#!/usr/bin/env python3
"""
export_bitnet.py — Convert BitNet b1.58 Hugging Face checkpoints to the `.bitnet`
format consumed by src/model_loader.c.

Supported inputs (auto-detected per tensor):
  * Offline-packed ternary checkpoints, e.g. microsoft/bitnet-b1.58-2B-4T:
      linear weights are U8 [out/4, in] with a `<name>_scale` (= beta) tensor.
      These are re-packed losslessly (no re-quantization).
  * Full-precision master weights, e.g. microsoft/bitnet-b1.58-2B-4T-bf16 or
    1bitLLM/bitnet_b1_58-large: linear weights are BF16/F16/F32 [out, in] and are
    quantized here with absmean: beta = mean|W|, W_q = clip(round(W / beta), -1, 1).
  * --mock: a synthetic checkpoint in either flavor, written as a real HF-style
    directory and then converted through the exact same code path.

Only numpy is required. Safetensors are parsed directly (the format is a JSON
header plus raw little-endian data); torch is used only for pytorch_model*.bin.

.bitnet format, version 1 (all little-endian):

  [0, 256)        header (see HEADER_FMT; header_crc32 covers these 256 bytes
                  with the crc field zeroed)
  [256, ...)      tensor table: n_tensors x 96-byte entries (table_crc32)
  data_offset...  tensor payloads, each 64-byte aligned, each with its own crc32

  Tensor dtypes:
    0 F32      row-major float32
    1 F16      row-major IEEE half
    2 TERNARY  rows x ceil(cols/4) bytes; weight k of a row lives in byte k/4,
               bits 2*(k%4)..2*(k%4)+1 (LSB first); 00 = 0, 01 = +1, 10 = -1.
               `scale` holds beta. Matches src/ternary_dot.h exactly.

  Tensor names: tok_embeddings, output_norm, output (only if untied),
    layers.{i}.{attn_norm, attn.wq, attn.wk, attn.wv, attn.wo, attn_sub_norm,
                ffn_norm, ffn.w_gate, ffn.w_up, ffn.w_down, ffn_sub_norm}
  Sub-norms are present only when header flag FLAG_SUB_NORMS is set.

A `--ref` sidecar records, for every ternary tensor, a random int8 input and the
exact int32 output computed from the unpacked ternary matrix, plus a few
embedding rows, so the C loader can prove bit-exact layout compatibility.
"""

import argparse
import json
import os
import re
import struct
import sys
import tempfile
import time
import zlib

import numpy as np

MAGIC = b"BITN"  # 0x42 0x49 0x54 0x4E
VERSION = 2          # written when any tensor uses the I128 layout; ROW4-only files stay v1
HEADER_SIZE = 256
ENTRY_SIZE = 96
NAME_LEN = 48
ALIGN = 64

DT_F32, DT_F16, DT_TERNARY, DT_TERNARY_I128, DT_Q8 = 0, 1, 2, 3, 4
DT_NAMES = {DT_F32: "f32", DT_F16: "f16", DT_TERNARY: "ternary", DT_TERNARY_I128: "tern-i128",
            DT_Q8: "q8"}
Q8_BLOCK = 32
TERNARY_DTYPES = (DT_TERNARY, DT_TERNARY_I128)

FLAG_TIED_EMBEDDINGS = 1 << 0
FLAG_SUB_NORMS = 1 << 1

ACT_SILU, ACT_RELU2 = 0, 1  # gated FFN activation: SwiGLU or squared-ReLU GLU

# magic, version, header_size, flags, vocab, dim, hidden, n_layers, n_heads,
# n_kv_heads, max_seq_len, ffn_act, norm_eps, rope_theta, n_tensors, entry_size,
# table_offset, data_offset, file_size, header_crc32, table_crc32
HEADER_FMT = "<4s11Iff2I3Q2I"
# name, dtype, ndim, rows, cols, offset, nbytes, scale, crc32, reserved
ENTRY_FMT = "<48s4IQQfI8x"
assert struct.calcsize(HEADER_FMT) == 96 and struct.calcsize(ENTRY_FMT) == ENTRY_SIZE
HEADER_CRC_OFFSET = 88

REF_MAGIC = b"BREF"
REF_VERSION = 1

GAMMA_MIN = 1e-5  # matches BITLINEAR_GAMMA_MIN / torch .clamp_(min=1e-5)
MAX_K = 1 << 17   # int32 -> float32 exactness limit from step3_bitlinear.c


def die(msg):
    sys.exit(f"export_bitnet: error: {msg}")


def align_up(x, a=ALIGN):
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def bf16_bits_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x):
    """Round-to-nearest-even float32 -> bfloat16 bit pattern (finite inputs)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounding = ((b >> 16) & 1) + np.uint32(0x7FFF)
    return ((b + rounding) >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------
# Ternary quantization and packing
# ---------------------------------------------------------------------------

def absmean_quantize(w):
    """BitNet b1.58 absmean weight quantization (matches the reference
    `scale = 1 / w.abs().mean().clamp(1e-5); (w * scale).round().clamp(-1, 1)`,
    with round = half-to-even like torch.round)."""
    w = np.asarray(w, dtype=np.float32)
    beta = np.float32(max(float(np.mean(np.abs(w), dtype=np.float64)), GAMMA_MIN))
    inv = np.float32(1.0) / beta
    q = np.clip(np.rint(w * inv), -1, 1).astype(np.int8)
    return q, float(beta)


def unpack_microsoft_u8(packed):
    """Inverse of the offline packing used by microsoft/bitnet-b1.58-2B-4T
    (transformers `unpack_weights`): U8 [R, C] -> int8 [4R, C]; rows
    [i*R, (i+1)*R) come from bits 2i..2i+1, stored as (w + 1).
    Verified against the bf16 master weights of that model (layer-0 k_proj)."""
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 2:
        die(f"packed weight must be 2-D, got shape {packed.shape}")
    r = packed.shape[0]
    out = np.empty((4 * r, packed.shape[1]), dtype=np.int8)
    for i in range(4):
        codes = (packed >> (2 * i)) & 3
        if np.any(codes == 3):
            die("packed weight contains invalid 2-bit code 3")
        out[i * r:(i + 1) * r] = codes.astype(np.int8) - 1
    return out


def pack_microsoft_u8(q):
    """Forward Microsoft packing (used only for --mock and self-tests)."""
    q = np.asarray(q, dtype=np.int8)
    if q.shape[0] % 4:
        die("Microsoft packing needs rows divisible by 4")
    r = q.shape[0] // 4
    out = np.zeros((r, q.shape[1]), dtype=np.uint8)
    for i in range(4):
        out |= ((q[i * r:(i + 1) * r] + 1).astype(np.uint8) << (2 * i))
    return out


def pack_ternary_i128(q):
    """int8 {-1,0,1} [M, K] -> I128 bytes [M, ceil(K/128)*32]: in each block of
    128 weights, byte j field f (bits 2f..2f+1) holds weight 32f + j. Matches
    src/ternary_dot.h (TERNARY_I128)."""
    q = np.asarray(q, dtype=np.int8)
    if q.ndim != 2:
        die("ternary tensor must be 2-D")
    if not np.all((q >= -1) & (q <= 1)):
        die("ternary tensor has values outside {-1, 0, 1}")
    m, k = q.shape
    nb = (k + 127) // 128
    codes = np.zeros((m, nb * 128), dtype=np.uint8)
    codes[:, :k][q == 1] = 1
    codes[:, :k][q == -1] = 2
    c = codes.reshape(m, nb, 4, 32)                      # [row, block, field, j]
    out = c[:, :, 0, :] | (c[:, :, 1, :] << 2) | (c[:, :, 2, :] << 4) | (c[:, :, 3, :] << 6)
    return out.reshape(m, nb * 32).astype(np.uint8)


def unpack_ternary_i128(packed, k):
    """Inverse of pack_ternary_i128 (self-test only)."""
    p = np.asarray(packed, dtype=np.uint8)
    m = p.shape[0]
    b = p.reshape(m, -1, 32)
    fields = np.stack([(b >> (2 * f)) & 3 for f in range(4)], axis=2)  # [row, block, field, j]
    lut = np.array([0, 1, -1, 0], dtype=np.int8)
    return lut[fields.reshape(m, -1)][:, :k]


def quantize_q8(w):
    """float32 [M, K] (K % 32 == 0) -> (int8 quants [M, K], float32 scales [M, K/32]),
    w ~= q * scale per block of 32 (src/q8.h). Round-half-to-even, clamp to +-127."""
    m, k = w.shape
    blocks = w.astype(np.float32).reshape(m, k // Q8_BLOCK, Q8_BLOCK)
    amax = np.max(np.abs(blocks), axis=2)
    scale = (amax / np.float32(127.0)).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(amax > 0, np.float32(127.0) / amax, np.float32(0.0)).astype(np.float32)
    q = np.clip(np.rint(blocks * inv[:, :, None]), -127, 127).astype(np.int8)
    return q.reshape(m, k), scale


def dequantize_q8(q, scale):
    m, k = q.shape
    return (q.reshape(m, k // Q8_BLOCK, Q8_BLOCK).astype(np.float32)
            * scale[:, :, None]).reshape(m, k)


def pack_ternary(q):
    """int8 {-1,0,1} [M, K] -> .bitnet TERNARY bytes [M, ceil(K/4)]."""
    q = np.asarray(q, dtype=np.int8)
    if q.ndim != 2:
        die("ternary tensor must be 2-D")
    if not np.all((q >= -1) & (q <= 1)):
        die("ternary tensor has values outside {-1, 0, 1}")
    m, k = q.shape
    k4 = (k + 3) // 4
    codes = np.zeros((m, k4 * 4), dtype=np.uint8)
    codes[:, :k][q == 1] = 1
    codes[:, :k][q == -1] = 2
    c = codes.reshape(m, k4, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)


def unpack_ternary(packed, k):
    """Inverse of pack_ternary (self-test only)."""
    p = np.asarray(packed, dtype=np.uint8)
    codes = np.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(p.shape[0], -1)[:, :k]
    lut = np.array([0, 1, -1, 0], dtype=np.int8)
    return lut[codes]


# ---------------------------------------------------------------------------
# Checkpoint readers
# ---------------------------------------------------------------------------

class SafetensorsSource:
    """Zero-copy safetensors reader (single file or sharded with index.json)."""

    _NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "U8": np.uint8,
           "I8": np.int8, "F64": np.float64}

    def __init__(self, files):
        self.tensors = {}
        for path in files:
            with open(path, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(hlen))
            header.pop("__metadata__", None)
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            base = 8 + hlen
            for name, t in header.items():
                if t["dtype"] not in self._NP:
                    die(f"{name}: unsupported safetensors dtype {t['dtype']}")
                a, b = t["data_offsets"]
                self.tensors[name] = (t["dtype"], tuple(t["shape"]), mm, base + a, base + b)

    def names(self):
        return set(self.tensors)

    def dtype(self, name):
        return self.tensors[name][0]

    def shape(self, name):
        return self.tensors[name][1]

    def raw(self, name):
        dt, shape, mm, a, b = self.tensors[name]
        arr = mm[a:b].view(self._NP[dt])
        return arr.reshape(shape) if shape else arr.reshape(())

    def f32(self, name):
        dt = self.dtype(name)
        arr = self.raw(name)
        if dt == "BF16":
            return bf16_bits_to_f32(arr)
        return arr.astype(np.float32)


class TorchBinSource:
    """pytorch_model*.bin reader (requires torch)."""

    def __init__(self, files):
        try:
            import torch
        except ImportError:
            die("pytorch_model*.bin checkpoints need torch (pip install torch); "
                "safetensors checkpoints do not")
        self.tensors = {}
        for path in files:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            for k, v in sd.items():
                self.tensors[k] = v

    def names(self):
        return set(self.tensors)

    def dtype(self, name):
        t = self.tensors[name]
        return "U8" if str(t.dtype) == "torch.uint8" else "F32"

    def shape(self, name):
        return tuple(self.tensors[name].shape)

    def raw(self, name):
        t = self.tensors[name]
        return t.numpy() if str(t.dtype) == "torch.uint8" else t.float().numpy()

    def f32(self, name):
        return self.tensors[name].float().numpy()


def open_checkpoint(path):
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return SafetensorsSource([path]), os.path.dirname(path)
    if not os.path.isdir(path):
        die(f"input {path!r} is neither a directory nor a .safetensors file")
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return SafetensorsSource([os.path.join(path, s) for s in shards]), path
    st = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if st:
        return SafetensorsSource([os.path.join(path, s) for s in st]), path
    bins = sorted(f for f in os.listdir(path) if re.match(r"pytorch_model.*\.bin$", f))
    if bins:
        return TorchBinSource([os.path.join(path, b) for b in bins]), path
    die(f"no .safetensors or pytorch_model*.bin files in {path!r}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def read_config(ckpt_dir, max_seq_len_override=None):
    path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(path):
        die(f"missing {path}")
    with open(path) as f:
        c = json.load(f)
    act = c.get("hidden_act", "silu")
    acts = {"silu": ACT_SILU, "swish": ACT_SILU, "relu2": ACT_RELU2}
    if act not in acts:
        die(f"unsupported hidden_act {act!r} (supported: {sorted(acts)})")
    cfg = {
        "vocab_size": int(c["vocab_size"]),
        "dim": int(c["hidden_size"]),
        "hidden_dim": int(c["intermediate_size"]),
        "n_layers": int(c["num_hidden_layers"]),
        "n_heads": int(c["num_attention_heads"]),
        "n_kv_heads": int(c.get("num_key_value_heads") or c["num_attention_heads"]),
        "max_seq_len": int(max_seq_len_override or c.get("max_position_embeddings", 2048)),
        "norm_eps": float(c.get("rms_norm_eps", 1e-5)),
        "rope_theta": float(c.get("rope_theta", 10000.0)),
        "ffn_act": acts[act],
        "tie_word_embeddings": bool(c.get("tie_word_embeddings", False)),
    }
    if c.get("attention_bias") or c.get("mlp_bias"):
        die("checkpoints with linear biases are not supported")
    if c.get("rope_scaling"):
        die(f"rope_scaling={c['rope_scaling']!r} is not supported yet")
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    for k in ("vocab_size", "dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "max_seq_len"):
        if cfg[k] <= 0:
            die(f"config {k} must be positive")
    if cfg["dim"] % cfg["n_heads"]:
        die("dim must be divisible by n_heads")
    if cfg["n_heads"] % cfg["n_kv_heads"]:
        die("n_heads must be divisible by n_kv_heads")
    if max(cfg["dim"], cfg["hidden_dim"]) >= MAX_K:
        die(f"dim/hidden_dim must be < {MAX_K} for exact int32 accumulation")


# ---------------------------------------------------------------------------
# Tensor plan: HF names -> .bitnet tensors
# ---------------------------------------------------------------------------

LINEARS = [
    ("attn.wq", "self_attn.q_proj"), ("attn.wk", "self_attn.k_proj"),
    ("attn.wv", "self_attn.v_proj"), ("attn.wo", "self_attn.o_proj"),
    ("ffn.w_gate", "mlp.gate_proj"), ("ffn.w_up", "mlp.up_proj"),
    ("ffn.w_down", "mlp.down_proj"),
]
SUB_NORM_ALIASES = {
    "attn_sub_norm": ["self_attn.attn_sub_norm", "self_attn.inner_attn_ln"],
    "ffn_sub_norm": ["mlp.ffn_sub_norm", "mlp.ffn_layernorm"],
}
IGNORABLE = [re.compile(r"\.rotary_emb\.inv_freq$")]


def plan_tensors(src, cfg, embed_dtype):
    """Returns (plan, flags). Each plan item: dict(name, dtype, shape, kind, hf)."""
    names = src.names()
    used = set()
    d, h, L = cfg["dim"], cfg["hidden_dim"], cfg["n_layers"]
    hd = d // cfg["n_heads"]
    q_out, kv_out = cfg["n_heads"] * hd, cfg["n_kv_heads"] * hd
    edt = {"f16": DT_F16, "f32": DT_F32, "q8": DT_Q8}[embed_dtype]
    if edt == DT_Q8 and d % Q8_BLOCK:
        die(f"--embed-dtype q8 needs dim % {Q8_BLOCK} == 0 (dim = {d}); use f16")

    def need(hf):
        if hf not in names:
            die(f"checkpoint is missing tensor {hf!r}")
        used.add(hf)
        return hf

    def check_shape(hf, want):
        got = src.shape(hf)
        if tuple(got) != tuple(want):
            die(f"{hf}: expected shape {tuple(want)}, checkpoint has {tuple(got)}")

    plan = []

    def add_float(name, hf, shape, dtype=DT_F32):
        check_shape(hf, shape)
        plan.append(dict(name=name, dtype=dtype, shape=shape, kind="float", hf=hf))

    add_float("tok_embeddings", need("model.embed_tokens.weight"), (cfg["vocab_size"], d), edt)

    # Sub-norms: all-or-nothing across layers.
    sub = {}
    for ours, aliases in SUB_NORM_ALIASES.items():
        found = [a for a in aliases if f"model.layers.0.{a}.weight" in names]
        sub[ours] = found[0] if found else None
    has_sub = all(sub.values())
    if any(sub.values()) and not has_sub:
        die(f"checkpoint has only some sub-norms: {sub}")

    lin_shapes = {"attn.wq": (q_out, d), "attn.wk": (kv_out, d), "attn.wv": (kv_out, d),
                  "attn.wo": (d, q_out), "ffn.w_gate": (h, d), "ffn.w_up": (h, d),
                  "ffn.w_down": (d, h)}
    for i in range(L):
        p = f"model.layers.{i}."
        add_float(f"layers.{i}.attn_norm", need(p + "input_layernorm.weight"), (d,))
        for ours, hf_suffix in LINEARS[:4]:
            add_linear(plan, src, need, f"layers.{i}.{ours}", p + hf_suffix + ".weight",
                       lin_shapes[ours], used)
        if has_sub:
            add_float(f"layers.{i}.attn_sub_norm", need(p + sub["attn_sub_norm"] + ".weight"), (d,))
        add_float(f"layers.{i}.ffn_norm", need(p + "post_attention_layernorm.weight"), (d,))
        for ours, hf_suffix in LINEARS[4:]:
            add_linear(plan, src, need, f"layers.{i}.{ours}", p + hf_suffix + ".weight",
                       lin_shapes[ours], used)
        if has_sub:
            add_float(f"layers.{i}.ffn_sub_norm", need(p + sub["ffn_sub_norm"] + ".weight"), (h,))

    add_float("output_norm", need("model.norm.weight"), (d,))

    tied = cfg["tie_word_embeddings"]
    if "lm_head.weight" in names:
        used.add("lm_head.weight")
        if tied:
            print("note: tie_word_embeddings=true; ignoring lm_head.weight", file=sys.stderr)
        else:
            add_float("output", "lm_head.weight", (cfg["vocab_size"], d), edt)
    elif not tied:
        print("note: no lm_head.weight; treating embeddings as tied", file=sys.stderr)
        tied = True

    leftovers = sorted(n for n in names - used if not any(r.search(n) for r in IGNORABLE))
    if leftovers:
        die(f"{len(leftovers)} unrecognized tensors would be dropped, e.g. {leftovers[:5]}")

    flags = (FLAG_TIED_EMBEDDINGS if tied else 0) | (FLAG_SUB_NORMS if has_sub else 0)
    return plan, flags


def add_linear(plan, src, need, name, hf, shape, used):
    need(hf)
    if src.dtype(hf) == "U8":
        want = (shape[0] // 4, shape[1])
        if shape[0] % 4 or tuple(src.shape(hf)) != want:
            die(f"{hf}: packed U8 expected shape {want}, got {tuple(src.shape(hf))}")
        need(hf + "_scale")
        kind = "packed_ms"
    else:
        if tuple(src.shape(hf)) != tuple(shape):
            die(f"{hf}: expected shape {tuple(shape)}, got {tuple(src.shape(hf))}")
        kind = "master"
    plan.append(dict(name=name, dtype=DT_TERNARY, shape=shape, kind=kind, hf=hf))


def tensor_nbytes(item):
    m = item["shape"][0]
    k = item["shape"][1] if len(item["shape"]) > 1 else 1
    if item["dtype"] == DT_TERNARY:
        return m * ((k + 3) // 4)
    if item["dtype"] == DT_TERNARY_I128:
        return m * (((k + 127) // 128) * 32)
    if item["dtype"] == DT_Q8:
        return m * k + m * (k // Q8_BLOCK) * 4
    return m * k * (4 if item["dtype"] == DT_F32 else 2)


def ternary_values(src, item):
    """Returns (int8 [M, K] ternary matrix, beta)."""
    if item["kind"] == "packed_ms":
        beta = float(src.f32(item["hf"] + "_scale").reshape(-1)[0])
        if not np.isfinite(beta) or beta <= 0:
            die(f"{item['hf']}_scale is not a positive finite number: {beta}")
        return unpack_microsoft_u8(src.raw(item["hf"])), beta
    return absmean_quantize(src.f32(item["hf"]))


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def export(src, cfg, out_path, ref_path=None, embed_dtype="q8", seed=1234, quiet=False,
           layout="i128"):
    plan, flags = plan_tensors(src, cfg, embed_dtype)
    if layout == "i128":
        for it in plan:
            if it["dtype"] == DT_TERNARY:
                it["dtype"] = DT_TERNARY_I128
    version = 2 if any(it["dtype"] in (DT_TERNARY_I128, DT_Q8) for it in plan) else 1

    table_off = HEADER_SIZE
    off = align_up(table_off + ENTRY_SIZE * len(plan))
    data_off = off
    for item in plan:
        if len(item["name"].encode()) >= NAME_LEN:
            die(f"tensor name too long: {item['name']}")
        item["offset"] = off
        item["nbytes"] = tensor_nbytes(item)
        off = align_up(off + item["nbytes"])
    file_size = off

    rng = np.random.default_rng(seed)
    ref_ternary, ref_embed = [], []
    tmp_path = out_path + ".tmp"
    t0 = time.time()
    counts = {DT_F32: 0, DT_F16: 0, DT_TERNARY: 0, DT_TERNARY_I128: 0, DT_Q8: 0}
    with open(tmp_path, "wb") as f:
        f.write(b"\0" * data_off)
        for n, item in enumerate(plan):
            f.seek(item["offset"])
            crc = 0
            if item["dtype"] in TERNARY_DTYPES:
                q, beta = ternary_values(src, item)
                packer = pack_ternary_i128 if item["dtype"] == DT_TERNARY_I128 else pack_ternary
                payload = packer(q).tobytes()
                crc = zlib.crc32(payload)
                f.write(payload)
                item["scale"] = beta
                if ref_path:
                    x = rng.integers(-128, 128, size=q.shape[1], dtype=np.int8)
                    y = q.astype(np.int32) @ x.astype(np.int32)
                    ref_ternary.append((item["name"], q.shape, beta, x, y.astype(np.int32)))
            else:
                item["scale"] = 1.0
                arr = src.f32(item["hf"]) if src.dtype(item["hf"]) != "BF16" else None
                rows = item["shape"][0]
                step = max(1, (1 << 24) // max(1, item["nbytes"] // rows))  # ~16 MiB chunks
                q8_scales = []  # Q8: quants are streamed, scales follow them in the payload
                for r0 in range(0, rows, step):
                    if arr is None:  # BF16: convert chunk-wise to bound memory
                        chunk = bf16_bits_to_f32(src.raw(item["hf"])[r0:r0 + step])
                    else:
                        chunk = arr[r0:r0 + step]
                    if item["dtype"] == DT_Q8:
                        if not np.all(np.isfinite(chunk)):
                            die(f"{item['hf']} contains NaN/Inf")
                        q, sc = quantize_q8(chunk)
                        q8_scales.append(sc)
                        chunk = dequantize_q8(q, sc)  # what the engine sees (for --ref)
                        b = q.tobytes()
                        crc = zlib.crc32(b, crc)
                        f.write(b)
                        if ref_path and item["name"] == "tok_embeddings":
                            for tok in sorted({0, rows // 2, rows - 1}):
                                if r0 <= tok < r0 + step:
                                    ref_embed.append((tok, chunk[tok - r0].astype(np.float32)))
                        continue
                    if item["dtype"] == DT_F16:
                        if np.any(np.abs(chunk) > 65504.0):
                            die(f"{item['hf']} overflows float16; use --embed-dtype f32")
                        chunk = chunk.astype(np.float16)
                    else:
                        chunk = chunk.astype(np.float32)
                    if not np.all(np.isfinite(chunk)):
                        die(f"{item['hf']} contains NaN/Inf")
                    b = np.ascontiguousarray(chunk).tobytes()
                    crc = zlib.crc32(b, crc)
                    f.write(b)
                    if ref_path and item["name"] == "tok_embeddings":
                        for tok in sorted({0, rows // 2, rows - 1}):
                            if r0 <= tok < r0 + step:
                                ref_embed.append((tok, chunk[tok - r0].astype(np.float32)))
                if q8_scales:
                    b = np.concatenate(q8_scales).astype("<f4").tobytes()
                    crc = zlib.crc32(b, crc)
                    f.write(b)
            item["crc"] = crc
            counts[item["dtype"]] += 1
            if not quiet and (n % 25 == 0 or n == len(plan) - 1):
                print(f"  [{n + 1:4d}/{len(plan)}] {item['name']:28s} {DT_NAMES[item['dtype']]:7s} "
                      f"{str(item['shape']):16s} ({time.time() - t0:.1f}s)", file=sys.stderr)
        f.truncate(file_size)

        table = b"".join(
            struct.pack(ENTRY_FMT, it["name"].encode(), it["dtype"], len(it["shape"]),
                        it["shape"][0], it["shape"][1] if len(it["shape"]) > 1 else 1,
                        it["offset"], it["nbytes"], it["scale"], it["crc"])
            for it in plan)
        header = bytearray(HEADER_SIZE)
        struct.pack_into(HEADER_FMT, header, 0, MAGIC, version, HEADER_SIZE, flags,
                         cfg["vocab_size"], cfg["dim"], cfg["hidden_dim"], cfg["n_layers"],
                         cfg["n_heads"], cfg["n_kv_heads"], cfg["max_seq_len"], cfg["ffn_act"],
                         cfg["norm_eps"], cfg["rope_theta"], len(plan), ENTRY_SIZE,
                         table_off, data_off, file_size, 0, zlib.crc32(table))
        struct.pack_into("<I", header, HEADER_CRC_OFFSET, zlib.crc32(bytes(header)))
        f.seek(0)
        f.write(header)
        f.write(table)
    os.replace(tmp_path, out_path)

    if ref_path:
        write_ref(ref_path, cfg["dim"], ref_ternary, ref_embed)

    if not quiet:
        print(f"wrote {out_path}: {file_size / 2**20:.1f} MiB, {len(plan)} tensors "
              f"v{version} ({counts[DT_TERNARY] + counts[DT_TERNARY_I128]} ternary "
              f"[{'i128' if counts[DT_TERNARY_I128] else 'row4'}], {counts[DT_F32]} f32, {counts[DT_F16]} f16, "
              f"{counts[DT_Q8]} q8), "
              f"flags={'tied ' if flags & FLAG_TIED_EMBEDDINGS else ''}"
              f"{'sub_norms' if flags & FLAG_SUB_NORMS else ''}, {time.time() - t0:.1f}s",
              file=sys.stderr)
        if ref_path:
            print(f"wrote {ref_path}: {len(ref_ternary)} GEMV vectors, {len(ref_embed)} embedding rows",
                  file=sys.stderr)
    return plan, flags


def write_ref(path, dim, ternary, embed):
    """BREF v1: magic, version, n_ternary, n_embed, dim; then per ternary tensor
    name[48], M, K, beta(f32), x[K] int8, y[M] int32; then per embedding row
    token(u32), row[dim] f32. Packed, little-endian."""
    with open(path, "wb") as f:
        f.write(struct.pack("<4s4I", REF_MAGIC, REF_VERSION, len(ternary), len(embed), dim))
        for name, (m, k), beta, x, y in ternary:
            f.write(struct.pack("<48sIIf", name.encode(), m, k, beta))
            f.write(x.astype(np.int8).tobytes())
            f.write(y.astype("<i4").tobytes())
        for tok, row in embed:
            f.write(struct.pack("<I", tok))
            f.write(row.astype("<f4").tobytes())


# ---------------------------------------------------------------------------
# Mock checkpoint
# ---------------------------------------------------------------------------

def write_safetensors(path, tensors):
    """tensors: dict name -> (dtype_str, np.ndarray with matching storage)."""
    header, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {"format": "pt"}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def make_mock_checkpoint(out_dir, flavor, dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                         vocab, max_seq_len, seed):
    """Writes config.json + model.safetensors in HF naming. flavor:
    'master' = BF16 full-precision linears; 'packed' = Microsoft U8 + weight_scale."""
    rng = np.random.default_rng(seed)
    hd = dim // n_heads
    os.makedirs(out_dir, exist_ok=True)
    cfg = {
        "architectures": ["BitNetForCausalLM"], "model_type": "bitnet",
        "hidden_act": "relu2", "hidden_size": dim, "intermediate_size": hidden_dim,
        "num_hidden_layers": n_layers, "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads, "vocab_size": vocab,
        "max_position_embeddings": max_seq_len, "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0, "tie_word_embeddings": True,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    def bf16(a):
        return ("BF16", f32_to_bf16_bits(a))

    t = {"model.embed_tokens.weight": bf16(rng.normal(0, 1, (vocab, dim)).astype(np.float32)),
         "model.norm.weight": bf16(rng.uniform(0.5, 1.5, dim).astype(np.float32))}
    shapes = {"self_attn.q_proj": (n_heads * hd, dim), "self_attn.k_proj": (n_kv_heads * hd, dim),
              "self_attn.v_proj": (n_kv_heads * hd, dim), "self_attn.o_proj": (dim, n_heads * hd),
              "mlp.gate_proj": (hidden_dim, dim), "mlp.up_proj": (hidden_dim, dim),
              "mlp.down_proj": (dim, hidden_dim)}
    for i in range(n_layers):
        p = f"model.layers.{i}."
        for n, d in (("input_layernorm", dim), ("post_attention_layernorm", dim),
                     ("self_attn.attn_sub_norm", dim), ("mlp.ffn_sub_norm", hidden_dim)):
            t[p + n + ".weight"] = bf16(rng.uniform(0.5, 1.5, d).astype(np.float32))
        for n, shp in shapes.items():
            w = rng.normal(0, 1 / np.sqrt(shp[1]), shp).astype(np.float32)
            if flavor == "master":
                t[p + n + ".weight"] = bf16(w)
            else:
                q, beta = absmean_quantize(bf16_bits_to_f32(f32_to_bf16_bits(w)))
                t[p + n + ".weight"] = ("U8", pack_microsoft_u8(q))
                t[p + n + ".weight_scale"] = bf16(np.array([beta], np.float32))
    t["model.layers.0.self_attn.rotary_emb.inv_freq"] = ("F32", np.ones(hd // 2, np.float32))
    write_safetensors(os.path.join(out_dir, "model.safetensors"), t)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(0)
    for m, k in [(1, 1), (3, 5), (8, 64), (7, 691), (64, 2560)]:
        q = rng.integers(-1, 2, size=(m, k), dtype=np.int8)
        p = pack_ternary(q)
        assert np.array_equal(unpack_ternary_i128(pack_ternary_i128(q), k), q)
        assert p.shape == (m, (k + 3) // 4)
        assert np.array_equal(unpack_ternary(p, k), q)
    # Bit layout spot-check against ternary_dot.h: weights [+1, -1, 0, +1] -> 0b01_00_10_01
    assert pack_ternary(np.array([[1, -1, 0, 1]], np.int8))[0, 0] == 0b01001001
    # Microsoft packing round-trip and code convention (0 -> -1, 1 -> 0, 2 -> +1).
    q = rng.integers(-1, 2, size=(640, 96), dtype=np.int8)
    assert np.array_equal(unpack_microsoft_u8(pack_microsoft_u8(q)), q)
    assert unpack_microsoft_u8(np.array([[0b10_01_00_10]], np.uint8)).ravel().tolist() == [1, -1, 0, 1]
    # absmean: beta = mean|W| and clipping to {-1, 0, 1}.
    w = np.array([0.5, -0.5, 1.5, 0.0, 3.0, -3.0], np.float32)
    q, beta = absmean_quantize(w)
    assert abs(beta - 1.4166666) < 1e-6 and q.tolist() == [0, 0, 1, 0, 1, -1]
    # bf16 round trip.
    x = rng.normal(size=1000).astype(np.float32)
    y = bf16_bits_to_f32(f32_to_bf16_bits(x))
    assert np.max(np.abs(x - y) / np.abs(x)) <= 2 ** -8
    # Q8: round trip within half a step per block; exact zeros; +-127 at the block max.
    w = rng.normal(size=(5, 96)).astype(np.float32)
    w[1, 32:64] = 0.0
    q, sc = quantize_q8(w)
    assert q.dtype == np.int8 and sc.shape == (5, 3) and np.all(np.abs(q) <= 127)
    err = np.abs(dequantize_q8(q, sc) - w).reshape(5, 3, 32).max(axis=2)
    assert np.all(err <= sc * 0.5 + 1e-7) and np.all(q[1, 32:64] == 0) and sc[1, 1] == 0
    assert np.all(np.abs(q).reshape(5, 3, 32).max(axis=2)[sc > 0] == 127)
    print("export_bitnet self-test: PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="HF checkpoint directory or .safetensors file")
    ap.add_argument("--output", help="output .bitnet path")
    ap.add_argument("--ref", help="also write a verification sidecar for src/step4_loader.c")
    ap.add_argument("--layout", choices=["i128", "row4"], default="i128",
                    help="ternary weight layout: i128 (format v2, fast SIMD kernels; default) "
                         "or row4 (format v1, readable by bitnet.c <= 1.1)")
    ap.add_argument("--embed-dtype", choices=["q8", "f16", "f32"], default="q8",
                    help="storage for embeddings / untied lm_head: q8 (int8 blocks of 32, "
                         "1.125 bytes/weight, format v2; default), f16 or f32")
    ap.add_argument("--max-seq-len", type=int, help="override max_position_embeddings")
    ap.add_argument("--seed", type=int, default=1234, help="seed for --ref inputs and --mock")
    ap.add_argument("--self-test", action="store_true", help="run packing unit tests and exit")
    mk = ap.add_argument_group("mock checkpoint (no download needed)")
    mk.add_argument("--mock", action="store_true", help="synthesize a checkpoint instead of --input")
    mk.add_argument("--mock-flavor", choices=["packed", "master"], default="packed")
    mk.add_argument("--mock-save-hf", help="keep the synthetic HF checkpoint in this directory")
    mk.add_argument("--dim", type=int, default=256)
    mk.add_argument("--hidden-dim", type=int, default=688)
    mk.add_argument("--n-layers", type=int, default=2)
    mk.add_argument("--n-heads", type=int, default=8)
    mk.add_argument("--n-kv-heads", type=int, default=2)
    mk.add_argument("--vocab", type=int, default=1000)
    a = ap.parse_args(argv)

    if a.self_test:
        self_test()
        return
    if not a.output:
        ap.error("--output is required")
    if bool(a.mock) == bool(a.input):
        ap.error("pass exactly one of --input or --mock")

    if a.mock:
        with tempfile.TemporaryDirectory() as tmp:
            hf_dir = a.mock_save_hf or tmp
            make_mock_checkpoint(hf_dir, a.mock_flavor, a.dim, a.hidden_dim, a.n_layers,
                                 a.n_heads, a.n_kv_heads, a.vocab, a.max_seq_len or 256, a.seed)
            src, d = open_checkpoint(hf_dir)
            export(src, read_config(d, a.max_seq_len), a.output, a.ref, a.embed_dtype, a.seed,
                   layout=a.layout)
    else:
        src, d = open_checkpoint(a.input)
        export(src, read_config(d, a.max_seq_len), a.output, a.ref, a.embed_dtype, a.seed,
               layout=a.layout)


if __name__ == "__main__":
    main()
