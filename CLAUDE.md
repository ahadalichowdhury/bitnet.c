# Project: bitnet.c — BitNet b1.58 Inference Engine (Apple Silicon ARM NEON)

## Project Overview
A standalone, dependency-free C inference runtime for 1-bit / 1.58-bit ternary models (`{-1, 0, 1}`) optimized for Apple Silicon (ARM NEON).

## Core Architecture & Memory Guidelines
- **Language**: Pure C (C11 standard). No external C++ dependencies or heavy libraries.
- **Target Hardware**: macOS ARM64 (Apple Silicon).
- **SIMD Intrinsics**: Strictly use ``. Never use x86 AVX intrinsics unless in explicit cross-platform abstractions.
- **Memory Safety & Layout**:
  - Memory alignment: Always use 64-byte or 16-byte alignment (`posix_memalign` / `aligned_alloc`).
  - Zero Dynamic Allocation in Inner Loops: Avoid `malloc`/`free` inside hot decode loops; use fixed arena memory or pre-allocated buffers.
  - Weights: Packed ternary format (4 weights per byte, 2 bits each).
  - Activations: Signed 8-bit integers (`int8_t`).

## Build & Test Commands
- **Compiler**: `clang`
- **Recommended Flags**: `-O3 -mcpu=native -Wall -Wextra -std=c11`
- **Sanitizers for Debugging**: `-fsanitize=address,undefined`
- **Verification Rule**: Any newly introduced SIMD kernel MUST include an identical scalar reference function and an automated assert-based verification check before benchmarking.
- **Layout**: `include/bitnet.h` (public API), `src/` (engine internals), `app/main.c` (CLI), `tests/test_*.c` (suites), `tools/` (Python exporter/reference/generators).
- **Make targets**: `make` (CLI `./bitnet`, `build/libbitnet.a`, test programs), `make test` (all suites; needs the exported model in `models/`), `make test-mock` (no model weights; what CI runs), `make bench`, `make asan`, `make leaks`, `make clean`.
- **Portability**: gate NEON code behind `BITNET_NEON` (`src/simd.h`) with a scalar fallback; put OS-specific code in `src/platform.h`. Verify fallbacks with `make BUILD=build-scalar CFLAGS="-O3 -mcpu=native -Wall -Wextra -std=c11 -DBITNET_FORCE_SCALAR -DBITNET_PORTABLE" test-mock`.
- **Model files**: `models/bitnet_2b4t.bitnet` (+ `.ref`, `.ref_logits`) come from `tools/export_bitnet.py` / `tools/reference_bitnet.py`; dev-only Python tools (HF `tokenizers`, `jinja2`) are never runtime dependencies.
- **Heap-allocation checks**: `tests/test_alloc_hook.h` counts allocations on all threads; forward passes and `generate()` must stay at zero. Do not use GCD (`dispatch_apply` allocates); use `src/threadpool.h`.

## Coding Style Rules
- Keep code clean, modular, and self-contained.
- Do not produce half-finished implementations or placeholders (`// TODO: implement later`) in numerical compute kernels.
- Prioritize cache locality and sequential memory access.