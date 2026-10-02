# bitnet.c Development Roadmap (8 Steps)

- [x] **Step 1: ARM NEON Ternary Dot-Product Kernel**
  - Implement scalar dot-product baseline
  - Implement ARM NEON vectorized dot-product
  - Verification test (scalar vs NEON match check)
  - Latency & speedup benchmark

- [x] **Step 2: Matrix-Vector Multiplication (GEMV) & Cache Tiling**
  - M x K matrix-vector kernel
  - L1/L2 cache tiling
  - Multi-threading support

- [x] **Step 3: BitLinear Scaling & Activation Quantization Math**
  - Symmetric 8-bit dynamic quantization for activations
  - BitLinear layer forward pass with scaling factor
  - RMSE validation against float reference

- [x] **Step 4: Model Exporter & Memory-Mapped Loader**
  - Python weight conversion script (to custom binary)
  - Zero-copy mmap model loader in C

- [x] **Step 5: Minimal BPE Tokenizer**
  - Byte-pair encoding and decoding in pure C
  - Token lookup and vocab loader

- [x] **Step 6: Transformer Layer & Attention Block**
  - Vectorized RMSNorm
  - Rotary Position Embeddings (RoPE)
  - KV-Cache management
  - Multi-Head Attention forward pass

- [x] **Step 7: Autoregressive Decode Loop & Sampling**
  - Prefill and decode separation
  - Temperature, Top-K, Top-P sampling
  - Real-time stdout token streaming

- [x] **Step 8: Production CLI & Single-Header API**
  - Command-line interface (`main.c`)
  - Standalone header interface (`bitnet.h`)
  - Final execution profiling (TPS, memory footprint)

---

# Phase 2

One feature per branch, each with tests, `make test` + `make test-mock` + CI
green before merging.

## A. Output quality & usability

- [x] **P1: Repetition / frequency / presence penalties + stop strings** (v1.2.0)
  - HF `repetition_penalty` and OpenAI frequency/presence penalties over a
    window of recent tokens; stop strings matched across token boundaries,
    zero allocation; CLI flags, `bitnet.h` and Python bindings
- [ ] **P2: Line editing + history in the REPL** — small built-in editor
  (arrows, Home/End, backspace, up/down history) on a TTY, plain reads when
  piped; no new dependencies
- [x] **P3: float16 KV cache** (after v1.2.0)
  - KV cache 600 -> 300 MiB at 4K context; NEON / F16C / scalar conversions
    bit-identical for every float32; next-token KL vs float32 cache at the
    float-reordering noise floor. The float32 cache is not kept selectable.
- [ ] **P4: Save / restore session (KV cache) to disk** — versioned,
  checksummed file tied to the model; reload skips re-prefilling; CLI
  `--save-session/--load-session`, API `bitnet_session_save/load`

## B. Performance

- [x] **P5: F16C float16 output layer on x86** (v1.2.0)
- [x] **SIMD-friendly I128 weight layout (`.bitnet` v2)** (v1.2.0) — 1.93x
  faster AVX2 ternary GEMV; v1 files still load
- [x] **P6: Batched prefill** (v1.2.0) — ternary GEMM, 16 tokens per chunk;
  prefill ~3x on M1, ~1.9x on 4-vCPU AVX2; KV cache and logits bit-identical
  to token-by-token. Attention is still per token (batched attention open).
- [x] **P7: int8 output layer** (v1.2.0) — Q8 blocks of 32 with f32 scales;
  file 1125 -> 851 MiB, M1 decode ~33 -> ~40 tok/s, perplexity +0.13%
- [ ] **P8: Benchmark vs Microsoft bitnet.cpp** — same machine, same model,
  documented methodology and script (`tools/bench_compare.sh`)

## C. Reach

- [ ] **P9: OpenAI-compatible HTTP server** — `bitnet --server`:
  `/v1/chat/completions`, `/v1/completions`, `/v1/models`, SSE streaming,
  pure C sockets, request size limits, one request at a time (queued)
- [ ] **P10: Fuzzing** — libFuzzer harnesses for the model loader,
  tokenizer.json parser and BPE encoder; corpus + CI smoke run
- [ ] **P11: More models** — SentencePiece `tokenizer.model` (Llama-2 style,
  e.g. 1bitLLM) and SiLU/SwiGLU BitNet checkpoints end to end; 7B-class
  ternary models (exporter + loader for their layouts)
- [ ] **P12: Packaging** — Homebrew formula, `pyproject.toml` for the Python
  bindings, `make install` (publishing needs the maintainer's accounts)

## D. Large

- [ ] **P13: Metal GPU backend** — design doc first (kernel layout,
  unified-memory zero-copy weights), then implementation behind a runtime flag
