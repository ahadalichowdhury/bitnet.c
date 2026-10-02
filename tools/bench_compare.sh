#!/usr/bin/env bash
#
# bench_compare.sh — bitnet.c vs Microsoft bitnet.cpp on this machine: same model
# (microsoft/bitnet-b1.58-2B-4T), same thread counts, llama-bench's metrics.
#
#   ./tools/bench_compare.sh                         pp512 + tg128, 5 repetitions
#   ./tools/bench_compare.sh --threads 2,4 --pp 256 --tg 64 --reps 3
#
# Steps (each is skipped when already done, so re-running is cheap):
#   1. builds bitnet.c's build/bench_llama (needs models/bitnet_2b4t.bitnet from
#      ./tools/download_model.sh)
#   2. clones github.com/microsoft/BitNet at a pinned commit into $COMPARE_DIR,
#      generates its kernels and builds llama-bench for the i2_s model, CPU only,
#      exactly as its setup_env.py would (cmake + clang; no Python packages)
#   3. downloads bitnet.cpp's official model ggml-model-i2_s.gguf (1.2 GB,
#      resumable, SHA-256 checked)
#   4. runs both engines at every thread count and prints a table
#
# Methodology (the same for both engines, llama-bench's definitions):
#   ppN  N random prompt tokens from an empty KV cache, processed as one batch
#   tgN  N tokens decoded one at a time from an empty KV cache, full logits each
#   one warm-up run, then --reps timed runs; mean +- standard deviation of tokens/s
# Thread counts default to 4 and all CPUs: on CPUs with efficiency cores (Apple
# M-series) llama.cpp is often fastest on the performance cores only, so each
# engine is compared at its own best setting as well as at equal threads.
#
# bitnet.cpp version: BITNET_CPP_REF (default 01eb415, March 2026, llama.cpp
# fork Eddie-Wang1120/llama.cpp with the I2_S kernels). Its main branch since
# July 2026 moved to a llama.cpp fork for embedding models that reads this GGUF
# as type "Q1_0" and runs it ~40x slower, so it is not a fair baseline. On some
# CPUs that version crashes (SIGBUS) in prompt processing with micro-batches of
# 128+ tokens; the script then re-runs with -ub 64 and says so in the table.
#
# Options: --threads LIST (comma-separated, default "4,<all CPUs>")
#          --pp N (512)  --tg N (128)  --reps N (5)
# Environment: COMPARE_DIR (default <repo>/build/compare), CC / CXX (compilers for
# bitnet.cpp; default: newest clang found), BITNET_CPP_REF, HF_BASE.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE_DIR="${COMPARE_DIR:-$ROOT/build/compare}"
HF_BASE="${HF_BASE:-https://huggingface.co}"
BITNET_CPP_REF="${BITNET_CPP_REF:-01eb415}"
MODEL_C="$ROOT/models/bitnet_2b4t.bitnet"
GGUF_DIR="$COMPARE_DIR/models/BitNet-b1.58-2B-4T"
GGUF="$GGUF_DIR/ggml-model-i2_s.gguf"
GGUF_URL="$HF_BASE/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf"
GGUF_SHA256="4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
GGUF_SIZE=1187801280

ncpu() { getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; }
NCPU="$(ncpu)"
THREADS="" PP=512 TG=128 REPS=5
while [ $# -gt 0 ]; do
    case "$1" in
        --threads) THREADS="$2"; shift ;;
        --pp)      PP="$2"; shift ;;
        --tg)      TG="$2"; shift ;;
        --reps)    REPS="$2"; shift ;;
        -h|--help) sed -n '3,41p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$THREADS" ] || { if [ "$NCPU" -gt 4 ]; then THREADS="4,$NCPU"; else THREADS="$NCPU"; fi; }
for v in PP TG REPS; do
    case "${!v}" in ''|*[!0-9]*) echo "--$(echo "$v" | tr 'A-Z' 'a-z') must be a number" >&2; exit 2 ;; esac
done
case "$THREADS" in ''|*[!0-9,]*|,*|*,|*,,*) echo "--threads must be a comma-separated list of numbers" >&2; exit 2 ;; esac
[ "$REPS" -ge 1 ] && [ $((PP + TG)) -gt 0 ] || { echo "need --reps >= 1 and --pp or --tg > 0" >&2; exit 2; }

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B= G= Y= R= N=; fi
step() { printf '%s==>%s %s\n' "$B" "$N" "$*"; }
ok()   { printf '    %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '    %s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '%serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
sha256() { if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi; }
filesize() { wc -c < "$1" | tr -d ' '; }

for tool in git cmake curl python3 make; do command -v "$tool" >/dev/null || die "$tool is required"; done
mkdir -p "$COMPARE_DIR" "$GGUF_DIR"

# ---- 1. bitnet.c -------------------------------------------------------------------
step "Building bitnet.c (build/bench_llama)"
[ -f "$MODEL_C" ] || die "missing ${MODEL_C#$ROOT/}: run ./tools/download_model.sh first"
make -C "$ROOT" -s build/bench_llama
ok "bitnet.c $(git -C "$ROOT" describe --tags --always --dirty 2>/dev/null || echo)"

# ---- 2. bitnet.cpp -----------------------------------------------------------------
SRC="$COMPARE_DIR/BitNet-$BITNET_CPP_REF"
BENCH="$SRC/build/bin/llama-bench"
step "Building Microsoft bitnet.cpp $BITNET_CPP_REF (llama-bench, i2_s, CPU only)"
if [ ! -x "$BENCH" ]; then
    if [ ! -d "$SRC/.git" ]; then
        git clone -q https://github.com/microsoft/BitNet.git "$SRC" || die "git clone of microsoft/BitNet failed"
    fi
    git -C "$SRC" checkout -q "$BITNET_CPP_REF" || die "unknown BITNET_CPP_REF $BITNET_CPP_REF"
    git -C "$SRC" submodule update -q --init --recursive --depth 1 || die "submodule checkout failed"

    if [ -z "${CC:-}" ]; then
        for c in clang-21 clang-20 clang-19 clang-18 clang; do
            if command -v "$c" >/dev/null; then CC="$c"; CXX="${c/clang/clang++}"; break; fi
        done
    fi
    [ -n "${CC:-}" ] || die "clang is required to build bitnet.cpp (Ubuntu: sudo apt install clang)"
    CXX="${CXX:-clang++}"
    command -v "$CXX" >/dev/null || die "$CXX not found (install the matching clang++ or set CXX)"
    echo "    $("$CC" --version | head -1)"

    # setup_env.py's gen_code() for BitNet-b1.58-2B-4T (it runs for every
    # quantization type; i2_s does not use the generated LUT kernels, but the
    # build needs the header).
    case "$(uname -m)" in
        arm64|aarch64)
            ARCH_FLAG="-DBITNET_ARM_TL1=OFF"
            CODEGEN=(utils/codegen_tl1.py --model bitnet_b1_58-3B --BM 160,320,320 --BK 64,128,64 --bm 32,64,32) ;;
        x86_64|amd64)
            ARCH_FLAG="-DBITNET_X86_TL2=OFF"
            CODEGEN=(utils/codegen_tl2.py --model bitnet_b1_58-3B --BM 160,320,320 --BK 96,96,96 --bm 32,32,32) ;;
        *) die "unsupported CPU architecture $(uname -m)" ;;
    esac
    (cd "$SRC" && python3 "${CODEGEN[@]}") > "$COMPARE_DIR/build.log" 2>&1 \
        || { tail -20 "$COMPARE_DIR/build.log"; die "kernel codegen failed"; }

    # setup_env.py's cmake configuration, plus llama.cpp's examples (off when
    # it is a subproject), static libraries, and CPU only (no Metal / BLAS /
    # CUDA) so both engines run on the same cores.
    cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release "$ARCH_FLAG" \
          -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
          -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=OFF \
          -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF >> "$COMPARE_DIR/build.log" 2>&1 \
        || { tail -20 "$COMPARE_DIR/build.log"; die "cmake configure failed (log: $COMPARE_DIR/build.log)"; }
    cmake --build "$SRC/build" --config Release --target llama-bench -j "$NCPU" >> "$COMPARE_DIR/build.log" 2>&1 \
        || { tail -20 "$COMPARE_DIR/build.log"; die "build failed (log: $COMPARE_DIR/build.log)"; }
fi
[ -x "$BENCH" ] || die "llama-bench was not built (log: $COMPARE_DIR/build.log)"
ok "microsoft/BitNet $(git -C "$SRC" rev-parse --short HEAD), llama.cpp $(git -C "$SRC/3rdparty/llama.cpp" rev-parse --short HEAD)"

# ---- 3. bitnet.cpp's model -----------------------------------------------------------
step "Model for bitnet.cpp: ${GGUF#$ROOT/}"
if [ -f "$GGUF" ] && [ "$(filesize "$GGUF")" = "$GGUF_SIZE" ]; then
    ok "already downloaded"
else
    curl -fL --retry 5 --retry-delay 3 -C - --progress-bar -o "$GGUF.part" "$GGUF_URL" \
        || die "download failed (re-run to resume)"
    [ "$(filesize "$GGUF.part")" = "$GGUF_SIZE" ] || die "download incomplete (re-run to resume)"
    [ "$(sha256 "$GGUF.part")" = "$GGUF_SHA256" ] || die "SHA-256 mismatch; delete $GGUF.part and re-run"
    mv "$GGUF.part" "$GGUF"
    ok "downloaded, SHA-256 verified"
fi

# ---- 4. run ------------------------------------------------------------------------------
OUT="$COMPARE_DIR/results"
rm -rf "$OUT" && mkdir -p "$OUT"
step "Benchmark: pp$PP / tg$TG, threads $THREADS, $REPS repetitions each (warm-up excluded)"
for t in ${THREADS//,/ }; do
    echo "    $t threads: bitnet.c ..."
    "$ROOT/build/bench_llama" "$MODEL_C" -p "$PP" -n "$TG" -t "$t" -r "$REPS" --json > "$OUT/c_t$t.json"
    echo "    $t threads: bitnet.cpp ..."
    rc=0
    ( "$BENCH" -m "$GGUF" -p "$PP" -n "$TG" -t "$t" -r "$REPS" -ngl 0 -o json \
            > "$OUT/cpp_t$t.json" 2> "$OUT/cpp_t$t.log" ) 2>/dev/null || rc=$?
    if [ "$rc" != 0 ]; then
        warn "llama-bench crashed (exit $rc); retrying with -ub 64 (see --help)"
        "$BENCH" -m "$GGUF" -p "$PP" -n "$TG" -t "$t" -r "$REPS" -ngl 0 -ub 64 -o json \
            > "$OUT/cpp_t$t.json" 2> "$OUT/cpp_t$t.log" || { tail -5 "$OUT/cpp_t$t.log"; die "llama-bench failed"; }
    fi
done

CPU_NAME="$( (sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2) | sed 's/^ *//')"
python3 - "$OUT" "$PP" "$TG" "$THREADS" "$MODEL_C" "$GGUF" "$BITNET_CPP_REF" "$CPU_NAME ($NCPU logical CPUs)" <<'PY'
import json, os, platform, sys
out, pp, tg, threads, model_c, gguf, ref, cpu = sys.argv[1:9]
pp, tg, threads = int(pp), int(tg), [int(t) for t in threads.split(",")]

def llama(path):
    text = open(path).read()
    res = {}
    for r in json.loads(text[text.index("["):]):
        key = ("pp", r["n_prompt"]) if r.get("n_gen", 0) == 0 else ("tg", r["n_gen"])
        res[key] = (r["avg_ts"], r["stddev_ts"], r.get("n_ubatch"))
        res["cpu"] = r.get("cpu_info", "")
    return res

rows, peak = [], 0
best = {}
for t in threads:
    c = json.load(open(os.path.join(out, f"c_t{t}.json")))
    peak = max(peak, c["peak_rss_bytes"])
    m = llama(os.path.join(out, f"cpp_t{t}.json"))
    for kind, n in (("pp", pp), ("tg", tg)):
        if n == 0:
            continue
        a = (c[f"{kind}_ts"], c[f"{kind}_sd"])
        b = m.get((kind, n))
        rows.append((f"{kind}{n}", t, a, b))
        for eng, v in (("c", a), ("cpp", b)):
            if v and v[0] > best.get((kind, eng), (0, 0))[0]:
                best[(kind, eng)] = (v[0], t)

fmt = lambda v: f"{v[0]:.2f} ± {v[1]:.2f}" if v else "n/a"
note = lambda test, b: " (-ub 64)" if b and b[2] == 64 and test.startswith("pp") else ""
mib = lambda n: n / 1048576
print()
print(f"CPU: {cpu} | model: microsoft/bitnet-b1.58-2B-4T")
print(f"bitnet.c  : {os.path.basename(model_c)} ({mib(os.path.getsize(model_c)):.0f} MiB file, peak RSS {mib(peak):.0f} MiB)")
print(f"bitnet.cpp: {os.path.basename(gguf)} ({mib(os.path.getsize(gguf)):.0f} MiB file), microsoft/BitNet {ref}")
print()
print(f"| {'test':<7} | {'threads':>7} | {'bitnet.c (t/s)':>16} | {'bitnet.cpp (t/s)':>24} | {'ratio':>6} |")
print(f"|{'-'*9}|{'-'*9}|{'-'*18}|{'-'*26}|{'-'*8}|")
for test, t, a, b in rows:
    ratio = f"{a[0] / b[0]:.2f}x" if b and b[0] > 0 else "n/a"
    print(f"| {test:<7} | {t:>7} | {fmt(a):>16} | {fmt(b) + note(test, b):>24} | {ratio:>6} |")
print()
for kind, n in (("pp", pp), ("tg", tg)):
    if n and (kind, "c") in best and (kind, "cpp") in best:
        (a, ta), (b, tb) = best[(kind, "c")], best[(kind, "cpp")]
        print(f"best {kind}{n}: bitnet.c {a:.2f} t/s ({ta} threads) vs bitnet.cpp {b:.2f} t/s ({tb} threads): {a / b:.2f}x")
print(f"\nraw results: {out}/")
PY
