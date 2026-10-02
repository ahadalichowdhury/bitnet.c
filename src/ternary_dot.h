/*
 * ternary_dot.h — Shared ternary (BitNet b1.58) x int8 dot-product primitives.
 *
 * Weight packing (4 weights per byte, LSB first):
 *   weight i lives in byte i/4, bits [2*(i%4) .. 2*(i%4)+1]
 *   code 00 = 0, 01 = +1, 10 = -1, 11 = reserved (decoded as 0)
 *
 * NEON block = 64 weights = 16 packed bytes = 64 activations.
 *   - vld1q_u8 loads 16 packed bytes p[j], j = 0..15.
 *   - Field k of p[j] is weight 4j+k. Extracting field k across all 16 lanes
 *     ((p >> 2k) & 3) yields weights {k, 4+k, 8+k, ...}.
 *   - vld4q_s8 de-interleaves activations the same way: val[k][j] = act[4j+k],
 *     so lanes line up with no shuffles.
 *   - vqtbl1q_s8 maps 2-bit codes -> {0, +1, -1, 0} in one instruction.
 *   - SDOT (vdotq_s32) multiplies int8 x int8 and accumulates groups of 4 into
 *     int32 lanes, so -128 * -1 = 128 never overflows an int8 lane.
 *
 * Range: |dot| <= 128 * len, so len must stay below 2^24 to fit int32_t.
 */
#ifndef BITNET_TERNARY_DOT_H
#define BITNET_TERNARY_DOT_H

#include <stdint.h>

#include "simd.h"

static const int8_t k_ternary_lut[16] = {0, 1, -1, 0, 0, 0, 0, 0,
                                         0, 0, 0,  0, 0, 0, 0, 0};

/* Scalar dot product over weights [begin, end). begin must be a multiple of 4. */
static inline int32_t ternary_dot_scalar_range(const int8_t *act, const uint8_t *packed_w,
                                               int begin, int end) {
    int32_t sum = 0;
    for (int i = begin; i < end; i++) {
        const uint8_t code = (uint8_t)((packed_w[i >> 2] >> ((i & 3) * 2)) & 3);
        sum += (int32_t)act[i] * (int32_t)k_ternary_lut[code];
    }
    return sum;
}

#ifdef BITNET_NEON

#if defined(__ARM_FEATURE_DOTPROD)
#define TERNARY_DOT(acc, w, a) vdotq_s32((acc), (w), (a))
#define TERNARY_DOT_PATH "SDOT (vdotq_s32)"
#else
/* Fallback for cores without SDOT: widen to int16, pairwise-add into int32. */
static inline int32x4_t ternary_dot_fallback(int32x4_t acc, int8x16_t w, int8x16_t a) {
    const int16x8_t lo = vmull_s8(vget_low_s8(w), vget_low_s8(a));
    const int16x8_t hi = vmull_high_s8(w, a);
    acc = vpadalq_s16(acc, lo);
    return vpadalq_s16(acc, hi);
}
#define TERNARY_DOT(acc, w, a) ternary_dot_fallback((acc), (w), (a))
#define TERNARY_DOT_PATH "SMULL + SADALP fallback (no dotprod)"
#endif

/* Decodes 16 packed bytes into 64 ternary int8 weights, field-major
 * (val[k][j] = weight 4j+k), matching the vld4q_s8 activation layout. */
static inline int8x16x4_t ternary_decode64(uint8x16_t p) {
    const int8x16_t lut = vld1q_s8(k_ternary_lut);
    const uint8x16_t m3 = vdupq_n_u8(0x03);
    int8x16x4_t w;
    w.val[0] = vqtbl1q_s8(lut, vandq_u8(p, m3));
    w.val[1] = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p, 2), m3));
    w.val[2] = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p, 4), m3));
    w.val[3] = vqtbl1q_s8(lut, vshrq_n_u8(p, 6));
    return w;
}

/* Single-row NEON dot product. Four independent accumulators (one per field)
 * break the SDOT dependency chain. */
static inline int32_t ternary_dot_neon(const int8_t *act, const uint8_t *packed_w, int size) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    const int full = size & ~63;
    for (int i = 0; i < full; i += 64) {
        const int8x16x4_t a = vld4q_s8(act + i);
        const int8x16x4_t w = ternary_decode64(vld1q_u8(packed_w + (i >> 2)));
        acc0 = TERNARY_DOT(acc0, w.val[0], a.val[0]);
        acc1 = TERNARY_DOT(acc1, w.val[1], a.val[1]);
        acc2 = TERNARY_DOT(acc2, w.val[2], a.val[2]);
        acc3 = TERNARY_DOT(acc3, w.val[3], a.val[3]);
    }

    int32_t sum = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));

    /* Tail (< 64 weights); `full` is a multiple of 4 so byte alignment holds. */
    if (full < size)
        sum += ternary_dot_scalar_range(act, packed_w, full, size);
    return sum;
}

#else /* !BITNET_NEON: portable scalar build */

#define TERNARY_DOT_PATH "scalar (no NEON)"

static inline int32_t ternary_dot_neon(const int8_t *act, const uint8_t *packed_w, int size) {
    return ternary_dot_scalar_range(act, packed_w, 0, size);
}

#endif /* BITNET_NEON */

#endif /* BITNET_TERNARY_DOT_H */
