/*
 * simd.h — Selects the SIMD code path.
 *
 * BITNET_NEON is defined when ARM NEON is available (Apple Silicon, ARM64
 * Linux); BITNET_AVX2 on x86-64 CPUs with AVX2. Every SIMD kernel has a scalar
 * fallback. Define BITNET_FORCE_SCALAR to build the scalar fallbacks even when
 * SIMD is available (used to verify the SIMD paths bit-for-bit).
 */
#ifndef BITNET_SIMD_H
#define BITNET_SIMD_H

#if defined(__ARM_NEON) && !defined(BITNET_FORCE_SCALAR)
#define BITNET_NEON 1
#include <arm_neon.h>
#elif defined(__AVX2__) && !defined(BITNET_FORCE_SCALAR)
/* x86-64 with AVX2: the ternary dot product / GEMV (the integer hot path) use
 * 256-bit kernels; the float kernels use the portable scalar code, which the
 * compiler auto-vectorizes. */
#define BITNET_AVX2 1
#include <immintrin.h>
#endif

#endif /* BITNET_SIMD_H */
