# bitnet.c

[![CI](https://github.com/ahadalichowdhury/bitnet.c/actions/workflows/ci.yml/badge.svg)](https://github.com/ahadalichowdhury/bitnet.c/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C11](https://img.shields.io/badge/C-C11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-ARM%20NEON-black.svg)](#architecture)
[![x86-64](https://img.shields.io/badge/x86--64-AVX2-blue.svg)](#architecture)

A dependency-free C11 inference engine for **BitNet b1.58** ternary models
(`{-1, 0, +1}` weights), hand-tuned for Apple Silicon with ARM NEON, with AVX2
kernels for Intel/AMD x86-64. It runs
[`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
at **~33 tokens/s on an M1**, with output matching an independent float64
reference of the Hugging Face implementation.

```
$ ./bitnet -p "What is the capital of Japan? Answer in one sentence." --temp 0
The capital of Japan is Tokyo.
```

## Quickstart

```sh
# 1. Clone and build (clang + make; nothing else)
git clone https://github.com/ahadalichowdhury/bitnet.c.git && cd bitnet.c
make

# 2. Download the model (~1.2 GB, resumable) and convert it to .bitnet (needs numpy)
./tools/download_model.sh

# 3. Run
./bitnet -p "Hello"
```

More ways to run it:

```sh
./bitnet -i                        # interactive multi-turn chat (Ctrl+C stops a reply, Ctrl+D quits)
./bitnet -p "Explain RoPE briefly." --temp 0.7 --top-p 0.9 --stats
./bitnet -p "List ten fruits." --repeat-penalty 1.2 --stop "\n\n"   # curb loops; stop at a blank line
./bitnet --bench                   # TTFT, prefill/decode tok/s, memory footprint
./bitnet --help                    # all options: sampling, penalties, --stop, --threads, --ctx, --system, --raw, -n
```

## Performance

Apple M1 (8 GB, 4 performance + 4 efficiency cores, 8 threads), `bitnet-b1.58-2B-4T`:

| Metric | Result |
|---|---|
| Decode throughput | **~33 tokens/s** (35 tok/s at short context, 30 tok/s at ~500 tokens) |
| Model load (mmap + validation) | **0.27 ms**, zero-copy (+0.03 MiB resident until weights are touched) |
| Time to first token | ~0.4 s for a short chat prompt (warm) |
| Prompt prefill | ~53 tokens/s |
| Heap allocations per token | **0** (measured on all threads during forward passes and `generate()`) |
| Peak memory | **~1.2 GB** resident (1.1 GB memory-mapped weights + KV cache in use) |
| Binary size | 140 KB CLI, 123 KB shared library |

Reproduce with `./bitnet --bench`. Correctness: identical top-1 predictions to a
float64 reference at every tested position, bit-exact Hugging Face tokenization
on 3,108 test cases, and clean AddressSanitizer / UndefinedBehaviorSanitizer /
ThreadSanitizer / `leaks` runs.

## Architecture

bitnet.c is written in **pure C11** with no dependencies beyond libc and
pthreads: no BLAS, no C++, no Python at runtime.

```
 text ──► tokenizer ──► transformer_forward (×30 layers) ──► sampler ──► UTF-8 stream ──► text
          (BPE, pure C)  RMSNorm → BitLinear Q/K/V → RoPE →   temp/top-k/   (never splits a
                         GQA attention → BitLinear O →        top-p, seeded  multi-byte char)
                         RMSNorm → BitLinear gate/up →
                         relu² → BitLinear down
```

- **Ternary NEON kernels.** Weights are stored as 2 bits each (4 per byte). One
  16-byte load is decoded into 64 weights with shifts, a mask and a single
  `TBL` lookup, then multiplied against int8 activations with the ARMv8.4
  `SDOT` instruction, four rows at a time (`src/ternary_dot.h`, `src/gemv.h`).
- **AVX2 kernels (x86-64).** The same ternary dot product and 4-row GEMV in
  256-bit form: one `PSHUFB` + two `AND`/`CMPEQ` masks decode 32 weights, and
  `MADDUBS`/`MADD` accumulate exactly (`maddubs(w+1, a) - maddubs(1, a)`, which
  stays correct for `a = -128` where `PSIGNB` would overflow). CI checks that
  AVX2 and scalar builds produce bit-identical logits.
- **SIMD weight layout (`.bitnet` v2).** Weights are packed in 128-weight
  blocks (`I128`) so each shift + mask of a 32-byte load yields 32 consecutive
  weights, with no per-byte shuffles. Older v1 files still load;
  `./tools/download_model.sh` upgrades them automatically.
- **BitLinear.** Activations are quantized per token to int8 (absmax,
  round-half-to-even, matching PyTorch) and the int32 result is rescaled once
  (`src/bitlinear.h`).
- **Attention.** Grouped-query attention reads each key/value row once for the
  four query heads sharing it, splits long contexts into chunks across cores
  and merges them with a log-sum-exp rule (flash-decoding style).
- **Zero-copy loading.** The `.bitnet` file is memory-mapped read-only; every
  tensor pointer points into the mapping after the header, table and tensor
  bounds are validated (`src/model_loader.c`).
- **Zero-allocation runtime.** All activations, the static KV cache and RoPE
  tables live in one pre-allocated arena; a custom pthread pool replaces GCD
  because `dispatch_apply` allocates on every call (`src/threadpool.c`).
- **Portable.** Every SIMD kernel has a scalar fallback (`src/simd.h`) and OS
  specifics live in `src/platform.h`, so it also builds and passes its suites
  on Linux x86-64 (CI).

## C library

`include/bitnet.h` is the entire public API. `make lib` builds both
`build/libbitnet.a` and the shared `build/libbitnet.dylib` (`.so` on Linux),
which exports only the `bitnet_*` functions.

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

Compile with `-Iinclude` and link `build/libbitnet.a -lpthread -lm`. Callbacks
always receive complete UTF-8.

## Python

`tools/bitnet.py` wraps the shared library with the standard-library `ctypes`
module (no pip packages):

```python
import sys; sys.path.insert(0, "tools")
from bitnet import BitNet

with BitNet() as llm:
    for piece in llm.generate("The capital of France is", max_new_tokens=16):
        print(piece, end="", flush=True)        # streams as it is generated
    print(llm.chat("Name three primary colors."))  # multi-turn chat with history
    print(llm.chat("Count to 10.", stop=["5"], repetition_penalty=1.1))
    print(llm.stats)                            # TTFT, tokens/s, stop reason, ...
```

Run `make lib && python3 tools/bitnet.py` for its self-test. Leaving a
`generate` loop early cancels the generation in C.

## Development

```sh
make                    # CLI, static + shared library, all test programs
make test               # every suite on the real model (first: ./tools/download_model.sh --test-data)
make test-mock          # model-free suites + a generated mock model (what CI runs)
make asan               # AddressSanitizer + UndefinedBehaviorSanitizer builds and runs
make leaks              # macOS leaks check
make bench              # ./bitnet --bench
```

| Path | What |
|---|---|
| `include/bitnet.h` | public C API |
| `src/ternary_dot.h`, `src/gemv.h` | NEON / AVX2 ternary dot product and multi-row GEMV |
| `src/bitlinear.h` | int8 activation quantization / dequantization |
| `src/model_loader.[ch]` | zero-copy mmap loader for the validated `.bitnet` format |
| `src/tokenizer.[ch]`, `src/unicode_tables.h` | Llama-3 byte-level BPE (Unicode 16 tables) |
| `src/transformer.[ch]`, `src/threadpool.[ch]` | forward pass and the allocation-free thread pool |
| `src/generate.[ch]` | sampling, UTF-8-safe streaming, prefill/decode loop, chat template |
| `src/bitnet.c`, `app/main.c` | public API implementation and the `bitnet` CLI |
| `src/simd.h`, `src/platform.h` | NEON vs scalar selection; macOS / Linux abstraction |
| `tests/test_*.c` | per-module verification + benchmark suites |
| `tools/` | model downloader, exporter, Python bindings, numpy reference, data generators |

On x86-64 the Makefile adds `-mavx2 -mfma` (use `make X86_SIMD=` on CPUs without
AVX2). `-DBITNET_FORCE_SCALAR` and `-DBITNET_PORTABLE` force the scalar and non-Apple
code paths on a Mac for testing. See `ROADMAP.md` for the 8 build steps.

## License

This project is licensed under the [MIT License](LICENSE). Model weights are not
included: `microsoft/bitnet-b1.58-2B-4T` is distributed separately under its own
license (see its Hugging Face model card).
