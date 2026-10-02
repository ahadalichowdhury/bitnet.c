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

#include <stddef.h>
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

#elif defined(BITNET_AVX2)

/* AVX2 block = 32 weights = 8 packed bytes = 32 activations.
 *   - One PSHUFB replicates packed byte j into output bytes 4j..4j+3 (both
 *     128-bit lanes hold the 8 bytes; lane 1 indexes bytes 4..7), so output
 *     byte t holds the byte that contains weight t.
 *   - AND + CMPEQ against {0x01,0x04,0x10,0x40} / {0x02,0x08,0x20,0x80}
 *     flags the "+1" bit and the "-1" bit of field t%4 as 0xFF; then
 *     w = isMinus - isPlus, which maps 00->0, 01->+1, 10->-1, 11->0 exactly
 *     like k_ternary_lut. No table lookup is needed.
 *   - _mm256_sign_epi8 would be wrong for a = -128 (-(-128) wraps), so the
 *     product uses MADDUBS with the unsigned u = w + 1 in {0,1,2}:
 *       sum(w*a) = maddubs(u, a) - maddubs(1, a)
 *     Both pair sums lie in [-512, 508] (no int16 saturation) and so does the
 *     difference, which MADD with ones widens into int32 lanes: exact. */
#define TERNARY_DOT_PATH "AVX2 (PSHUFB/CMPEQ decode + MADDUBS)"

static inline __m256i ternary_decode32_u_avx2(const uint8_t *p) {
    const __m256i rep = _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
    const __m256i pm = _mm256_set1_epi32(0x40100401);         /* bytes 01 04 10 40 */
    const __m256i mm = _mm256_set1_epi32((int)0x80200802u);   /* bytes 02 08 20 80 */
    const __m256i r = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i *)p)), rep);
    const __m256i plus = _mm256_cmpeq_epi8(_mm256_and_si256(r, pm), pm);
    const __m256i minus = _mm256_cmpeq_epi8(_mm256_and_si256(r, mm), mm);
    return _mm256_sub_epi8(_mm256_sub_epi8(minus, plus), _mm256_set1_epi8(-1)); /* w + 1 */
}

/* int32x8 partial sums of w*a for one block; asum16 = maddubs(1, a). */
static inline __m256i ternary_block32_avx2(__m256i u, __m256i a, __m256i asum16) {
    const __m256i d16 = _mm256_sub_epi16(_mm256_maddubs_epi16(u, a), asum16);
    return _mm256_madd_epi16(d16, _mm256_set1_epi16(1));
}

static inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* SIMD entry point (named for the NEON original; AVX2 here). */
static inline int32_t ternary_dot_neon(const int8_t *act, const uint8_t *packed_w, int size) {
    const __m256i ones8 = _mm256_set1_epi8(1);
    __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
    const int full = size & ~31;
    int i = 0;
    for (; i + 64 <= full; i += 64) { /* two independent chains */
        const __m256i a0 = _mm256_loadu_si256((const __m256i *)(act + i));
        const __m256i a1 = _mm256_loadu_si256((const __m256i *)(act + i + 32));
        acc0 = _mm256_add_epi32(acc0, ternary_block32_avx2(ternary_decode32_u_avx2(packed_w + (i >> 2)), a0,
                                                           _mm256_maddubs_epi16(ones8, a0)));
        acc1 = _mm256_add_epi32(acc1, ternary_block32_avx2(ternary_decode32_u_avx2(packed_w + (i >> 2) + 8), a1,
                                                           _mm256_maddubs_epi16(ones8, a1)));
    }
    for (; i < full; i += 32) {
        const __m256i a0 = _mm256_loadu_si256((const __m256i *)(act + i));
        acc0 = _mm256_add_epi32(acc0, ternary_block32_avx2(ternary_decode32_u_avx2(packed_w + (i >> 2)), a0,
                                                           _mm256_maddubs_epi16(ones8, a0)));
    }
    int32_t sum = hsum_epi32_avx2(_mm256_add_epi32(acc0, acc1));
    /* Tail (< 32 weights); `full` is a multiple of 4 so byte alignment holds. */
    if (full < size) sum += ternary_dot_scalar_range(act, packed_w, full, size);
    return sum;
}

#else /* no SIMD: portable scalar build */

#define TERNARY_DOT_PATH "scalar (no SIMD)"

static inline int32_t ternary_dot_neon(const int8_t *act, const uint8_t *packed_w, int size) {
    return ternary_dot_scalar_range(act, packed_w, 0, size);
}

#endif /* BITNET_NEON / BITNET_AVX2 */

/* ========================================================================= */
/* I128 layout (.bitnet v2): SIMD-friendly packing                           */
/*                                                                           */
/* Weights are grouped in blocks of 128 = 32 bytes. Within a block, byte j   */
/* (0..31) field f (bits 2f..2f+1) holds weight 32f + j. One 32-byte load +  */
/* shift/mask therefore yields 32 *consecutive* weights, matching 32         */
/* consecutive activations: no de-interleaving and no per-byte replication.  */
/* Codes are the same as ROW4 (00 = 0, 01 = +1, 10 = -1, 11 = 0). A row      */
/* takes ceil(K / 128) * 32 bytes (the padding in the last block is zero).   */
/* ========================================================================= */

typedef enum {
    TERNARY_ROW4 = 0, /* .bitnet v1: weight k at byte k/4, bits 2*(k%4) */
    TERNARY_I128 = 1, /* .bitnet v2: see above */
} ternary_layout;

static inline size_t ternary_row_bytes(int K, ternary_layout layout) {
    return layout == TERNARY_I128 ? (((size_t)K + 127) / 128) * 32 : ((size_t)K + 3) / 4;
}

/* Scalar reference over weights [begin, end) of an I128 row; begin must be a
 * multiple of 128 for block alignment of later SIMD use, any value works here. */
static inline int32_t ternary_dot_i128_scalar_range(const int8_t *act, const uint8_t *w, int begin,
                                                    int end) {
    int32_t sum = 0;
    for (int k = begin; k < end; k++) {
        const int r = k & 127;
        const uint8_t code = (uint8_t)((w[(size_t)(k >> 7) * 32 + (r & 31)] >> (2 * (r >> 5))) & 3);
        sum += (int32_t)act[k] * (int32_t)k_ternary_lut[code];
    }
    return sum;
}

#if defined(BITNET_NEON)

/* acc += dot of one 128-weight block (16-byte halves; one table lookup per
 * 16 codes; activations loaded contiguously). */
#define I128_NEON_FIELD(acc, ph, f, a)                                                   \
    acc = TERNARY_DOT(acc, vqtbl1q_s8(lut, (f) == 3 ? vshrq_n_u8(ph, 6)                  \
                                                    : vandq_u8(vshrq_n_u8(ph, 2 * (f)), m3)), (a))

#define TERNARY_I128_PATH TERNARY_DOT_PATH

static inline int32_t ternary_dot_i128(const int8_t *act, const uint8_t *w, int K) {
    const int8x16_t lut = vld1q_s8(k_ternary_lut);
    const uint8x16_t m3 = vdupq_n_u8(3);
    int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
    const int full = K & ~127;
    for (int b = 0; b < full; b += 128) {
        const uint8x16_t p0 = vld1q_u8(w + (b >> 2)), p1 = vld1q_u8(w + (b >> 2) + 16);
        const int8_t *a = act + b;
        c0 = TERNARY_DOT(c0, vqtbl1q_s8(lut, vandq_u8(p0, m3)), vld1q_s8(a));
        c1 = TERNARY_DOT(c1, vqtbl1q_s8(lut, vandq_u8(p1, m3)), vld1q_s8(a + 16));
        I128_NEON_FIELD(c2, p0, 1, vld1q_s8(a + 32));
        I128_NEON_FIELD(c3, p1, 1, vld1q_s8(a + 48));
        I128_NEON_FIELD(c0, p0, 2, vld1q_s8(a + 64));
        I128_NEON_FIELD(c1, p1, 2, vld1q_s8(a + 80));
        I128_NEON_FIELD(c2, p0, 3, vld1q_s8(a + 96));
        I128_NEON_FIELD(c3, p1, 3, vld1q_s8(a + 112));
    }
    int32_t sum = vaddvq_s32(vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)));
    if (full < K) sum += ternary_dot_i128_scalar_range(act, w, full, K);
    return sum;
}

#elif defined(BITNET_AVX2)

/* PSHUFB table: code -> w + 1 (unsigned for MADDUBS). */
static inline __m256i i128_lut_u(void) {
    return _mm256_setr_epi8(1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
}

/* int16 pair sums of (w+1)*a over the 4 fields of one block. Each MADDUBS
 * lane is in [-512, 508], so the 4-field sum stays inside int16. */
static inline __m256i i128_block_u16(const uint8_t *wb, const int8_t *a, __m256i lut, __m256i m3) {
    const __m256i p = _mm256_loadu_si256((const __m256i *)wb);
    __m256i s = _mm256_maddubs_epi16(_mm256_shuffle_epi8(lut, _mm256_and_si256(p, m3)),
                                     _mm256_loadu_si256((const __m256i *)a));
    s = _mm256_add_epi16(s, _mm256_maddubs_epi16(
            _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 2), m3)),
            _mm256_loadu_si256((const __m256i *)(a + 32))));
    s = _mm256_add_epi16(s, _mm256_maddubs_epi16(
            _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 4), m3)),
            _mm256_loadu_si256((const __m256i *)(a + 64))));
    s = _mm256_add_epi16(s, _mm256_maddubs_epi16(
            _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 6), m3)),
            _mm256_loadu_si256((const __m256i *)(a + 96))));
    return s;
}

/* int16 pair sums of 1*a over one block (the -sum(a) correction). */
static inline __m256i i128_block_asum16(const int8_t *a) {
    const __m256i one = _mm256_set1_epi8(1);
    __m256i s = _mm256_maddubs_epi16(one, _mm256_loadu_si256((const __m256i *)a));
    s = _mm256_add_epi16(s, _mm256_maddubs_epi16(one, _mm256_loadu_si256((const __m256i *)(a + 32))));
    s = _mm256_add_epi16(s, _mm256_maddubs_epi16(one, _mm256_loadu_si256((const __m256i *)(a + 64))));
    return _mm256_add_epi16(s, _mm256_maddubs_epi16(one, _mm256_loadu_si256((const __m256i *)(a + 96))));
}

/* acc += int32 partial sums of (w+1)*a over one block: VPDPBUSD (4 per block)
 * with VNNI, else the int16 MADDUBS sums widened by MADD. Both exact. */
static inline __m256i i128_block_acc(__m256i acc, const uint8_t *wb, const int8_t *a, __m256i lut,
                                     __m256i m3) {
#ifdef BITNET_VNNI
    const __m256i p = _mm256_loadu_si256((const __m256i *)wb);
    acc = BITNET_DPBUSD(acc, _mm256_shuffle_epi8(lut, _mm256_and_si256(p, m3)),
                        _mm256_loadu_si256((const __m256i *)a));
    acc = BITNET_DPBUSD(acc, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 2), m3)),
                        _mm256_loadu_si256((const __m256i *)(a + 32)));
    acc = BITNET_DPBUSD(acc, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 4), m3)),
                        _mm256_loadu_si256((const __m256i *)(a + 64)));
    return BITNET_DPBUSD(acc, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 6), m3)),
                         _mm256_loadu_si256((const __m256i *)(a + 96)));
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(i128_block_u16(wb, a, lut, m3), _mm256_set1_epi16(1)));
#endif
}

/* acc += int32 partial sums of a over one block (the -sum(a) correction). */
static inline __m256i i128_asum_acc(__m256i acc, const int8_t *a) {
#ifdef BITNET_VNNI
    const __m256i one = _mm256_set1_epi8(1);
    for (int f = 0; f < 4; f++)
        acc = BITNET_DPBUSD(acc, one, _mm256_loadu_si256((const __m256i *)(a + 32 * f)));
    return acc;
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(i128_block_asum16(a), _mm256_set1_epi16(1)));
#endif
}

#ifdef BITNET_VNNI
#define TERNARY_I128_PATH "AVX2 + VNNI (VPDPBUSD)"
#else
#define TERNARY_I128_PATH "AVX2 (MADDUBS)"
#endif

static inline int32_t ternary_dot_i128(const int8_t *act, const uint8_t *w, int K) {
    const __m256i lut = i128_lut_u(), m3 = _mm256_set1_epi8(3);
    __m256i acc = _mm256_setzero_si256(), asum = _mm256_setzero_si256();
    const int full = K & ~127;
    for (int b = 0; b < full; b += 128) {
        acc = i128_block_acc(acc, w + (b >> 2), act + b, lut, m3);
        asum = i128_asum_acc(asum, act + b);
    }
    int32_t sum = hsum_epi32_avx2(_mm256_sub_epi32(acc, asum)); /* sum((w+1)a) - sum(a) */
    if (full < K) sum += ternary_dot_i128_scalar_range(act, w, full, K);
    return sum;
}

#else

#define TERNARY_I128_PATH TERNARY_DOT_PATH

static inline int32_t ternary_dot_i128(const int8_t *act, const uint8_t *w, int K) {
    return ternary_dot_i128_scalar_range(act, w, 0, K);
}

#endif /* I128 kernels */

#endif /* BITNET_TERNARY_DOT_H */
