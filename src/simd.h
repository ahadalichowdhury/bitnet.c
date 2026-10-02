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
/* F16C (VCVTPH2PS) for the float16 output layer; BITNET_NO_F16C keeps the
 * scalar GEMV (F16C reorders its float sum) so AVX2 and scalar builds can be
 * compared bit-for-bit. */
#if defined(__F16C__) && !defined(BITNET_NO_F16C)
#define BITNET_F16C 1
#endif
/* F16C row conversions (float16 KV cache). These round exactly like the
 * scalar float_to_half, so they stay on even with BITNET_NO_F16C. */
#if defined(__F16C__)
#define BITNET_F16C_CVT 1
#endif
#if defined(__FMA__)
#define BITNET_FMADD(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define BITNET_FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif
#endif

#endif /* BITNET_SIMD_H */
