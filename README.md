# BitNet-C

A dependency-free C11 inference engine for **BitNet b1.58** ternary models
(`{-1, 0, +1}` weights), hand-tuned for Apple Silicon with ARM NEON. Runs
[`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
at ~33 tokens/s on an M1, with output matching an independent reference of the
Hugging Face implementation.

```
$ ./bitnet -p "What is the capital of Japan? Answer in one sentence." --temp 0
The capital of Japan is Tokyo.
```

## Quick start

```sh
# 1. Get and convert the model (numpy only; ~1.2 GB download)
mkdir -p models/hf/bitnet-b1.58-2B-4T && cd models/hf/bitnet-b1.58-2B-4T
for f in config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -LO https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/main/$f; done
cd - && python3 tools/export_bitnet.py --input models/hf/bitnet-b1.58-2B-4T \
                                      --output models/bitnet_2b4t.bitnet --ref models/bitnet_2b4t.ref

# 2. Build and run
make
./bitnet -i                       # interactive chat (Ctrl+C stops a reply, Ctrl+D quits)
./bitnet -p "Explain RoPE briefly." --stats
./bitnet --bench                  # TTFT, prefill/decode tok/s, memory
make test                         # every verification suite (needs the exported model)
make test-mock                    # model-free suites + mock model (what CI runs)
```

`./bitnet --help` lists all options (`--temp`, `--top-p`, `--top-k`, `--seed`,
`--threads`, `--ctx`, `--system`, `--raw`, `-n`).

## Library

`include/bitnet.h` is the whole public API; compile with `-Iinclude` and link `build/libbitnet.a -lpthread -lm`.

```c
#include "bitnet.h"

static void print(const char *piece, void *user) { fputs(piece, stdout); fflush(stdout); }

BitNetContext *ctx = bitnet_init("models/bitnet_2b4t.bitnet",
                                 "models/hf/bitnet-b1.58-2B-4T/tokenizer.json",
                                 bitnet_default_config());
bitnet_chat_turn(ctx, "Hello!", bitnet_default_params(), print, NULL);   /* multi-turn */
bitnet_generate(ctx, "Once upon a time", bitnet_default_params(), print, NULL);
bitnet_cancel(ctx);        /* from any thread or a signal handler */
bitnet_free(ctx);
```

Callbacks always receive complete UTF-8; generation performs no heap allocation.

## Layout

| Path | What |
|---|---|
| `src/ternary_dot.h`, `src/gemv.h` | NEON ternary dot product and multi-row GEMV (2-bit weights, SDOT) |
| `src/bitlinear.h` | int8 activation quantization / dequantization (round-half-even, HF parity) |
| `src/model_loader.[ch]` | zero-copy mmap loader for the validated `.bitnet` format |
| `src/tokenizer.[ch]`, `src/unicode_tables.h` | Llama-3 byte-level BPE, bit-exact with HF `tokenizers` |
| `src/transformer.[ch]`, `src/threadpool.[ch]` | forward pass (RMSNorm, RoPE, GQA flash-decoding attention, relu² FFN), allocation-free pool |
| `src/generate.[ch]` | sampling, UTF-8-safe streaming, prefill/decode loop, chat template |
| `include/bitnet.h`, `src/bitnet.c`, `app/main.c` | public API and the `bitnet` CLI |
| `src/simd.h`, `src/platform.h` | NEON vs scalar selection; macOS / Linux abstraction (clock, RSS, fp16) |
| `tests/test_*.c` | per-module verification + benchmark suites, `tests/test_api.c` for the public API |
| `tools/` | exporter, numpy reference model, golden-data and Unicode-table generators |

## Portability

Apple Silicon is the primary target (NEON + SDOT kernels, GCD for the standalone
GEMV). Every NEON kernel has a scalar fallback (`src/simd.h`), and OS specifics
live in `src/platform.h`, so the engine also builds on Linux and x86-64; CI runs
the model-free and mock-model suites on `ubuntu-latest`. `-DBITNET_FORCE_SCALAR`
and `-DBITNET_PORTABLE` force those fallbacks on a Mac for testing.

See `ROADMAP.md` for the 8 build steps.
