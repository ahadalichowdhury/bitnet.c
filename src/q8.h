/*
 * q8.h — Block-quantized int8 ("Q8") matrices for the output layer.
 *
 * The tied embedding / lm_head matrix [vocab, dim] dominates memory traffic
 * per token (f16: 2 bytes per weight). Q8 stores it at 1.125 bytes per weight:
 *
 *   tensor payload = rows x cols int8 quants (row-major)
 *                  + rows x (cols/32) float32 scales, one per block of 32
 *   weight[r][k]   = q[r][k] * scale[r][k/32]        (cols % 32 == 0)
 *
 * For the logits, the final hidden state x is quantized the same way (per
 * block of 32, absmax / 127, round-half-to-even), each block dot product is
 * an exact int32, and the float rescale uses a fixed 8-lane order:
 *
 *   acc[b % 8] += ((float)idot_b * xs[b]) * ws[b]      (no fused multiply-add)
 *   result      = ((acc0+acc4) + (acc2+acc6)) + ((acc1+acc5) + (acc3+acc7))
 *
 * so the scalar reference, NEON (SDOT) and AVX2 (MADDUBS) paths return
 * bit-identical floats. Quants are expected in [-127, 127] (the exporter
 * clamps); -128 in a weight is still computed exactly by every path.
 */
#ifndef BITNET_Q8_H
#define BITNET_Q8_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "simd.h"

#define Q8_BLOCK 32

/* Scales of a Q8 tensor start right after the quants. */
static inline const float *q8_scales(const void *data, int rows, int cols) {
    return (const float *)((const int8_t *)data + (size_t)rows * cols);
}

/* x[n] (n % 32 == 0) -> q[n] int8 in [-127, 127] and s[n/32]: x ~= q * s.
 * Shared by every SIMD path (one call per token, n = dim). */
static inline void q8_quantize(const float *x, int8_t *q, float *s, int n) {
    for (int b = 0; b < n / Q8_BLOCK; b++) {
        const float *xb = x + b * Q8_BLOCK;
        float amax = 0.0f;
        for (int j = 0; j < Q8_BLOCK; j++) amax = fmaxf(amax, fabsf(xb[j]));
        const float scale = amax / 127.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        s[b] = scale;
        for (int j = 0; j < Q8_BLOCK; j++) {
            float v = nearbyintf(xb[j] * inv); /* half-to-even (default rounding mode) */
            v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
            q[b * Q8_BLOCK + j] = (int8_t)v;
        }
    }
}

/* Dequantizes row `r` (embedding lookup): out[k] = q[k] * scale[k/32]. */
static inline void q8_dequant_row(const void *data, int rows, int cols, int r, float *out) {
    const int8_t *q = (const int8_t *)data + (size_t)r * cols;
    const float *s = q8_scales(data, rows, cols) + (size_t)r * (cols / Q8_BLOCK);
    for (int k = 0; k < cols; k++) out[k] = (float)q[k] * s[k / Q8_BLOCK];
}

static inline int32_t q8_block_dot_scalar(const int8_t *w, const int8_t *x) {
    int32_t d = 0;
    for (int j = 0; j < Q8_BLOCK; j++) d += (int32_t)w[j] * (int32_t)x[j];
    return d;
}

/* One lane update, as separate statements so it is never contracted to FMA. */
static inline void q8_lane_update(float *acc, int32_t d, float xs, float ws) {
    const float p = (float)d * xs;
    const float t = p * ws;
    *acc += t;
}

static inline float q8_reduce8(const float a[8]) {
    const float t0 = a[0] + a[4], t1 = a[1] + a[5], t2 = a[2] + a[6], t3 = a[3] + a[7];
    return (t0 + t2) + (t1 + t3);
}

/* Reference: dot(w_row, x) for one row of nb blocks. */
static inline float q8_dot_scalar(const int8_t *w, const float *ws, const int8_t *x,
                                  const float *xs, int nb) {
    float acc[8] = {0};
    for (int b = 0; b < nb; b++)
        q8_lane_update(&acc[b & 7], q8_block_dot_scalar(w + b * Q8_BLOCK, x + b * Q8_BLOCK),
                       xs[b], ws[b]);
    return q8_reduce8(acc);
}

#if defined(BITNET_NEON)

/* int32 dot of one 32-byte block, as 4 partial lanes (SDOT). */
static inline int32x4_t q8_block_neon(const int8_t *w, const int8_t *x) {
    int32x4_t d = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w), vld1q_s8(x));
    return vdotq_s32(d, vld1q_s8(w + 16), vld1q_s8(x + 16));
}

/* Block sums of 4 consecutive blocks: [sum d0, sum d1, sum d2, sum d3]. */
static inline int32x4_t q8_dot4_neon(const int8_t *w, const int8_t *x) {
    const int32x4_t d0 = q8_block_neon(w, x), d1 = q8_block_neon(w + 32, x + 32);
    const int32x4_t d2 = q8_block_neon(w + 64, x + 64), d3 = q8_block_neon(w + 96, x + 96);
    return vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
}

static inline float q8_dot(const int8_t *w, const float *ws, const int8_t *x, const float *xs,
                           int nb) {
    float32x4_t lo = vdupq_n_f32(0.0f), hi = lo;
    int b = 0;
    for (; b + 8 <= nb; b += 8) {
        const int8_t *wb = w + b * Q8_BLOCK, *xb = x + b * Q8_BLOCK;
        const float32x4_t dlo = vcvtq_f32_s32(q8_dot4_neon(wb, xb));        /* exact: |d| < 2^24 */
        const float32x4_t dhi = vcvtq_f32_s32(q8_dot4_neon(wb + 128, xb + 128));
        lo = vaddq_f32(lo, vmulq_f32(vmulq_f32(dlo, vld1q_f32(xs + b)), vld1q_f32(ws + b)));
        hi = vaddq_f32(hi, vmulq_f32(vmulq_f32(dhi, vld1q_f32(xs + b + 4)), vld1q_f32(ws + b + 4)));
    }
    float acc[8];
    vst1q_f32(acc, lo);
    vst1q_f32(acc + 4, hi);
    for (; b < nb; b++)
        q8_lane_update(&acc[b & 7], q8_block_dot_scalar(w + b * Q8_BLOCK, x + b * Q8_BLOCK),
                       xs[b], ws[b]);
    return q8_reduce8(acc);
}

#define Q8_PATH "NEON (SDOT)"

#elif defined(BITNET_AVX2)

/* int32 dot of one 32-byte block, as 8 partial lanes: |w| * sign(x, w)
 * (MADDUBS pairs stay below 2 * 128 * 127 < 2^15, so no saturation). */
static inline __m256i q8_block_avx2(const int8_t *w, const int8_t *x) {
    const __m256i wv = _mm256_loadu_si256((const __m256i *)w);
    const __m256i xv = _mm256_loadu_si256((const __m256i *)x);
#ifdef BITNET_VNNI
    return BITNET_DPBUSD(_mm256_setzero_si256(), _mm256_abs_epi8(wv), _mm256_sign_epi8(xv, wv));
#else
    const __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(xv, wv));
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
#endif
}

/* Block sums of 8 consecutive blocks, in block order. */
static inline __m256i q8_dot8_avx2(const int8_t *w, const int8_t *x) {
    __m256i d[8];
    for (int i = 0; i < 8; i++) d[i] = q8_block_avx2(w + i * Q8_BLOCK, x + i * Q8_BLOCK);
    const __m256i h01 = _mm256_hadd_epi32(d[0], d[1]), h23 = _mm256_hadd_epi32(d[2], d[3]);
    const __m256i h45 = _mm256_hadd_epi32(d[4], d[5]), h67 = _mm256_hadd_epi32(d[6], d[7]);
    const __m256i a = _mm256_hadd_epi32(h01, h23); /* [d0..d3 low half | d0..d3 high half] */
    const __m256i c = _mm256_hadd_epi32(h45, h67); /* [d4..d7 low half | d4..d7 high half] */
    return _mm256_add_epi32(_mm256_permute2x128_si256(a, c, 0x20),
                            _mm256_permute2x128_si256(a, c, 0x31));
}

static inline float q8_dot(const int8_t *w, const float *ws, const int8_t *x, const float *xs,
                           int nb) {
    __m256 acc8 = _mm256_setzero_ps();
    int b = 0;
    for (; b + 8 <= nb; b += 8) {
        const __m256 d = _mm256_cvtepi32_ps(q8_dot8_avx2(w + b * Q8_BLOCK, x + b * Q8_BLOCK));
        const __m256 p = _mm256_mul_ps(d, _mm256_loadu_ps(xs + b));
        acc8 = _mm256_add_ps(acc8, _mm256_mul_ps(p, _mm256_loadu_ps(ws + b)));
    }
    float acc[8];
    _mm256_storeu_ps(acc, acc8);
    for (; b < nb; b++)
        q8_lane_update(&acc[b & 7], q8_block_dot_scalar(w + b * Q8_BLOCK, x + b * Q8_BLOCK),
                       xs[b], ws[b]);
    return q8_reduce8(acc);
}

#ifdef BITNET_VNNI
#define Q8_PATH "AVX2 + VNNI (VPDPBUSD)"
#else
#define Q8_PATH "AVX2 (MADDUBS)"
#endif

#else

static inline float q8_dot(const int8_t *w, const float *ws, const int8_t *x, const float *xs,
                           int nb) {
    return q8_dot_scalar(w, ws, x, xs, nb);
}

#define Q8_PATH "scalar"

#endif

#endif /* BITNET_Q8_H */
