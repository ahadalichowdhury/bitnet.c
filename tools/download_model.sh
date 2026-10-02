#!/usr/bin/env bash
#
# download_model.sh — Fetch microsoft/bitnet-b1.58-2B-4T and convert it to .bitnet.
#
#   ./tools/download_model.sh              download (~1.2 GB) + convert -> models/bitnet_2b4t.bitnet
#   ./tools/download_model.sh --test-data  also write the reference files `make test` needs
#
# Re-running is safe: interrupted downloads resume (curl -C -), finished files are
# verified (size + SHA-256 for the weights) and skipped, and the conversion only
# runs when its output is missing. Needs curl, python3 and numpy (for conversion).
#
# Environment: MODELS_DIR (default: <repo>/models), HF_BASE (mirror of huggingface.co).

set -euo pipefail

REPO="microsoft/bitnet-b1.58-2B-4T"
HF_BASE="${HF_BASE:-https://huggingface.co}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
HF_DIR="$MODELS_DIR/hf/bitnet-b1.58-2B-4T"
OUT="$MODELS_DIR/bitnet_2b4t.bitnet"

WEIGHTS_SHA256="8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b"
WEIGHTS_SIZE=1178623988
FILES=(config.json generation_config.json tokenizer.json tokenizer_config.json
       special_tokens_map.json LICENSE model.safetensors)

TEST_DATA=0
for arg in "$@"; do
    case "$arg" in
        --test-data) TEST_DATA=1 ;;
        -h|--help) sed -n '3,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
    esac
done

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B= G= Y= R= N=; fi
step() { printf '%s==>%s %s\n' "$B" "$N" "$*"; }
ok()   { printf '    %s✓%s %s\n' "$G" "$N" "$*"; }
die()  { printf '%serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

command -v curl >/dev/null || die "curl is required"
sha256() { if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi; }
filesize() { wc -c < "$1" | tr -d ' '; }

mkdir -p "$HF_DIR"

# ---- 1. download ----------------------------------------------------------------
step "Downloading $REPO into ${HF_DIR#$ROOT/}"
# Every file is downloaded to <name>.part and renamed only when complete, so an
# existing final file is always whole; the weights resume from their .part file.
for f in "${FILES[@]}"; do
    dest="$HF_DIR/$f"
    if [ -f "$dest" ]; then
        ok "$f already downloaded"
        continue
    fi
    if [ "$f" = model.safetensors ]; then
        have=0; [ -f "$dest.part" ] && have="$(filesize "$dest.part")"
        [ "$have" -gt 0 ] && echo "    resuming $f at $((have / 1048576)) MiB" || echo "    $f (1.1 GiB)"
        curl -fL --retry 5 --retry-delay 3 -C - --progress-bar -o "$dest.part" \
             "$HF_BASE/$REPO/resolve/main/$f" || die "download of $f failed (re-run this script to resume)"
        [ "$(filesize "$dest.part")" = "$WEIGHTS_SIZE" ] || die "$f is incomplete (re-run to resume)"
    else
        curl -fsSL --retry 5 --retry-delay 3 -o "$dest.part" "$HF_BASE/$REPO/resolve/main/$f" \
            || die "download of $f failed"
    fi
    mv "$dest.part" "$dest"
    ok "$f"
done

step "Verifying model.safetensors"
got="$(sha256 "$HF_DIR/model.safetensors")"
if [ "$got" != "$WEIGHTS_SHA256" ]; then
    die "SHA-256 mismatch for model.safetensors (got $got); delete it and re-run"
fi
ok "SHA-256 $WEIGHTS_SHA256"

# ---- 2. convert -------------------------------------------------------------------
step "Converting to ${OUT#$ROOT/}"
# Format version (u32 at byte 4): v1 files are upgraded to the v2 SIMD layout.
fmt_version() { python3 -c "import struct,sys; print(struct.unpack('<I', open(sys.argv[1],'rb').read(8)[4:8])[0])" "$1" 2>/dev/null || echo 0; }
if [ -f "$OUT" ] && [ "$(fmt_version "$OUT")" -lt 2 ]; then
    printf '    %s!%s %s is format v1: re-converting to v2 (faster SIMD weight layout)\n' "$Y" "$N" "${OUT#$ROOT/}"
    rm -f "$OUT"
fi
if [ -f "$OUT" ] && [ -f "${OUT%.bitnet}.ref" ]; then
    ok "already converted (format v$(fmt_version "$OUT"))"
else
    command -v python3 >/dev/null || die "python3 is required for the conversion"
    python3 -c "import numpy" 2>/dev/null || die "numpy is required: python3 -m pip install numpy"
    python3 "$ROOT/tools/export_bitnet.py" --input "$HF_DIR" --output "$OUT" --ref "${OUT%.bitnet}.ref"
    ok "wrote ${OUT#$ROOT/}"
fi

# ---- 3. optional: reference logits for `make test` ---------------------------------
if [ "$TEST_DATA" = 1 ]; then
    REF_LOGITS="${OUT%.bitnet}.ref_logits"
    step "Writing test reference logits (${REF_LOGITS#$ROOT/})"
    if [ -f "$REF_LOGITS" ]; then
        ok "already present"
    else
        # Tokens of "<|begin_of_text|>The capital of France is Paris. The capital of Germany is"
        python3 "$ROOT/tools/reference_bitnet.py" "$HF_DIR" --out "$REF_LOGITS" \
            --tokens 128000,791,6864,315,9822,374,12366,13,578,6864,315,10057,374 2>/dev/null
        ok "wrote ${REF_LOGITS#$ROOT/}"
    fi
fi

echo
printf '%sReady.%s Try: %s./bitnet -p "Hello"%s\n' "$G" "$N" "$B" "$N"
[ "$TEST_DATA" = 1 ] || printf '%s(run with --test-data before `make test`)%s\n' "$Y" "$N"
