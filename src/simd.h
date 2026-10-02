/*
 * simd.h — Selects the SIMD code path.
 *
 * BITNET_NEON is defined when ARM NEON is available (Apple Silicon, ARM64
 * Linux). Every NEON kernel in the engine has a scalar fallback used when it
 * is not, e.g. on x86-64. Define BITNET_FORCE_SCALAR to build the scalar
 * fallbacks on ARM as well (used to verify them against the NEON paths).
 */
#ifndef BITNET_SIMD_H
#define BITNET_SIMD_H

#if defined(__ARM_NEON) && !defined(BITNET_FORCE_SCALAR)
#define BITNET_NEON 1
#include <arm_neon.h>
#endif

#endif /* BITNET_SIMD_H */
