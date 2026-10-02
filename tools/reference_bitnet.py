#!/usr/bin/env python3
"""
reference_bitnet.py — Independent numpy reference forward pass for
microsoft/bitnet-b1.58-2B-4T, used to validate src/transformer.c.

Independence from the C path:
  * reads the ORIGINAL Hugging Face checkpoint (model.safetensors, U8 packed
    weights + weight_scale) via Microsoft's unpack order, not our .bitnet file;
  * full-sequence (prefill) forward with an explicit causal mask, so the C
    incremental KV-cache path is checked against a cache-free computation;
  * float64 activations everywhere except where HF semantics define float32
    (activation quantization scale, RoPE angles).

Semantics follow transformers `modeling_bitnet.py` + `integrations/bitnet.py`
(AutoBitLinear, offline mode): BitLinear(x) = F.linear(ActQuant(x), W_q) *
weight_scale with ActQuant = per-token absmax int8, round-half-to-even.

Usage:
  python3 tools/reference_bitnet.py models/hf/bitnet-b1.58-2B-4T \
      --text "The capital of France is Paris. The capital of Germany is" \
      --out models/bitnet_2b4t.ref_logits
  (--text needs the `tokenizers` package; or pass --tokens 128000,791,...)

Output (.ref_logits, little-endian): b"BLOG", u32 version=1, u32 T, u32 vocab,
i32 tokens[T], f32 logits[T][vocab].
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_bitnet import SafetensorsSource, unpack_microsoft_u8  # noqa: E402


def rmsnorm(x, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return w * (x / np.sqrt(var + eps))


def act_quant(x):
    """HF ActQuant: float32 per-row absmax scale, round-half-to-even, clamp."""
    x32 = x.astype(np.float32)
    amax = np.maximum(np.max(np.abs(x32), axis=-1, keepdims=True), np.float32(1e-5))
    scale = (np.float32(127.0) / amax).astype(np.float32)
    q = np.clip(np.rint(x32 * scale), -128, 127).astype(np.float32)
    return q, scale.astype(np.float64)


def bitlinear(x, wq, beta):
    """(T, K) float64 -> (T, M). q @ W^T is an exact integer sum in float32
    (|sum| <= 128 * K < 2^24), then dequantized in float64."""
    q, scale = act_quant(x)
    y = (q @ wq.T).astype(np.float64)
    return y / scale * beta


def rope_tables(T, head_dim, theta):
    inv_freq = (np.float32(1.0) / (np.float32(theta) ** (np.arange(0, head_dim, 2, dtype=np.float32)
                                                          / np.float32(head_dim)))).astype(np.float32)
    ang = (np.arange(T, dtype=np.float32)[:, None] * inv_freq[None, :]).astype(np.float32)
    ang = np.concatenate([ang, ang], axis=-1).astype(np.float64)
    return np.cos(ang), np.sin(ang)


def apply_rope(x, cos, sin):
    """x: (T, H, D); rotate_half layout."""
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt", help="HF checkpoint dir (config.json, model.safetensors)")
    ap.add_argument("--text", help="prompt (BOS is prepended)")
    ap.add_argument("--tokens", help="comma-separated token ids (used as-is)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                    help="activation precision (float32 measures the numerical noise floor)")
    a = ap.parse_args()
    DT = np.float64 if a.dtype == "float64" else np.float32

    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    if a.tokens:
        tokens = [int(t) for t in a.tokens.split(",")]
    elif a.text is not None:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(a.ckpt, "tokenizer.json"))
        tokens = [cfg["bos_token_id"]] + tok.encode(a.text, add_special_tokens=False).ids
    else:
        ap.error("pass --text or --tokens")

    d, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, eps, L, V = d // nh, cfg["rms_norm_eps"], cfg["num_hidden_layers"], cfg["vocab_size"]
    assert cfg["hidden_act"] == "relu2" and cfg["tie_word_embeddings"]
    T = len(tokens)
    src = SafetensorsSource([os.path.join(a.ckpt, "model.safetensors")])
    t0 = time.time()

    def w(name):
        return src.f32(name).astype(DT)

    def tern(prefix):
        q = unpack_microsoft_u8(src.raw(prefix + ".weight")).astype(np.float32)
        beta = float(src.f32(prefix + ".weight_scale").reshape(-1)[0])
        return q, beta

    emb = src.raw("model.embed_tokens.weight")  # BF16 bits, (V, d)
    x = np.stack([(emb[t].astype(np.uint32) << 16).view(np.float32) for t in tokens]).astype(DT)
    cos, sin = (c.astype(DT) for c in rope_tables(T, hd, cfg["rope_theta"]))
    mask = np.triu(np.full((T, T), -np.inf), k=1)
    group = nh // nkv

    for l in range(L):
        p = f"model.layers.{l}."
        h = rmsnorm(x, w(p + "input_layernorm.weight"), eps)
        q = bitlinear(h, *tern(p + "self_attn.q_proj")).astype(DT).reshape(T, nh, hd)
        k = bitlinear(h, *tern(p + "self_attn.k_proj")).astype(DT).reshape(T, nkv, hd)
        v = bitlinear(h, *tern(p + "self_attn.v_proj")).astype(DT).reshape(T, nkv, hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = np.repeat(k, group, axis=1), np.repeat(v, group, axis=1)   # repeat_kv
        s = (np.einsum("thd,uhd->htu", q, k) / DT(np.sqrt(hd)) + mask[None]).astype(DT)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        att = np.einsum("htu,uhd->thd", s, v).reshape(T, nh * hd)
        att = rmsnorm(att, w(p + "self_attn.attn_sub_norm.weight"), eps)
        x = (x + bitlinear(att, *tern(p + "self_attn.o_proj"))).astype(DT)

        h = rmsnorm(x, w(p + "post_attention_layernorm.weight"), eps)
        g = bitlinear(h, *tern(p + "mlp.gate_proj")).astype(DT)
        u = bitlinear(h, *tern(p + "mlp.up_proj")).astype(DT)
        f = np.square(np.maximum(g, 0.0)) * u                              # relu2(gate) * up
        f = rmsnorm(f, w(p + "mlp.ffn_sub_norm.weight"), eps)
        x = (x + bitlinear(f, *tern(p + "mlp.down_proj"))).astype(DT)
        print(f"  layer {l + 1:2d}/{L}  |x| rms {np.sqrt(np.mean(x * x)):.3f}  ({time.time() - t0:.1f}s)",
              file=sys.stderr)

    x = rmsnorm(x, w("model.norm.weight"), eps)
    logits = np.empty((T, V), dtype=np.float64)
    for r0 in range(0, V, 16384):                                          # tied lm_head, chunked
        chunk = (emb[r0:r0 + 16384].astype(np.uint32) << 16).view(np.float32).astype(DT)
        logits[:, r0:r0 + 16384] = x @ chunk.T

    with open(a.out, "wb") as f:
        f.write(struct.pack("<4s3I", b"BLOG", 1, T, V))
        f.write(np.asarray(tokens, dtype="<i4").tobytes())
        f.write(logits.astype("<f4").tobytes())
    top = np.argsort(-logits, axis=-1)[:, :5]
    print(f"wrote {a.out}: T={T} tokens={tokens} ({time.time() - t0:.1f}s)", file=sys.stderr)
    for t in range(T):
        print(f"  pos {t:2d} token {tokens[t]:6d} -> top5 {top[t].tolist()}", file=sys.stderr)


if __name__ == "__main__":
    main()
