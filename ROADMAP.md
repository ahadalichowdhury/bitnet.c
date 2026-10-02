# BitNet-C Development Roadmap (8 Steps)

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