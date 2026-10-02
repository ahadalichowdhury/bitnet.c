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
/* VNNI (VPDPBUSD: u8 x s8 products, 4 per int32 lane, accumulated without
 * saturation) replaces MADDUBS + MADD + ADD in the integer kernels. AVX-VNNI
 * (Alder Lake+, Zen 5) or AVX512-VNNI with VL on 256-bit registers (Ice Lake,
 * Sapphire Rapids, Zen 4). Exact like the AVX2 path; BITNET_NO_VNNI turns it off. */
#if !defined(BITNET_NO_VNNI)
#if defined(__AVXVNNI__)
#define BITNET_VNNI 1
#define BITNET_DPBUSD(acc, u, s) _mm256_dpbusd_avx_epi32((acc), (u), (s))
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define BITNET_VNNI 1
#define BITNET_DPBUSD(acc, u, s) _mm256_dpbusd_epi32((acc), (u), (s))
#endif
#endif
/* AVX2 + FMA float kernels (attention, exp) whose float sums are ordered
 * differently from the scalar code; gated like the F16C output layer. */
#if defined(BITNET_F16C) && defined(__FMA__)
#define BITNET_AVX2_FLOAT 1
#endif
#if defined(__FMA__)
#define BITNET_FMADD(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define BITNET_FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif
#endif

#endif /* BITNET_SIMD_H */
