/*
 * bitlinear.h — BitLinear activation quantization and dequantization kernels
 * (BitNet b1.58; Step 3), shared by the BitLinear layer and the transformer.
 *
 *   gamma = max(max|x|, 1e-5)                         (per-token absmax)
 *   x_q   = clip(round(x * 127 / gamma), -128, 127)   -> int8_t
 *   y     = y_int * (gamma * beta / 127)               (beta = weight scale)
 *
 * round() is round-half-to-even (torch.round): rintf / FCVTNS. This matches
 * HF transformers AutoBitLinear exactly: ActQuant then F.linear * weight_scale.
 */
#ifndef BITNET_BITLINEAR_H
#define BITNET_BITLINEAR_H

#include <math.h>
#include <stdint.h>

#include "simd.h"

#define BITLINEAR_GAMMA_MIN 1e-5f
#define BITLINEAR_QMAX      127.0f

/* ------------------------------------------------------------------------- */
/* Activation quantization                                                   */
/* ------------------------------------------------------------------------- */

static inline float quant_scale_from_gamma(float gamma) { return BITLINEAR_QMAX / gamma; }

static inline int8_t quantize_one(float x, float scale) {
    float v = rintf(x * scale);
    if (v > 127.0f)  v = 127.0f;
    if (v < -128.0f) v = -128.0f;
    return (int8_t)v;
}

static inline float absmax_scalar(const float *x, int K) {
    float m = 0.0f;
    for (int k = 0; k < K; k++) {
        const float a = fabsf(x[k]);
        if (a > m) m = a;
    }
    return m;
}

/* Returns gamma (clamped absmax). */
static inline float quantize_act_scalar(const float *x, int8_t *x_q, int K) {
    const float gamma = fmaxf(absmax_scalar(x, K), BITLINEAR_GAMMA_MIN);
    const float scale = quant_scale_from_gamma(gamma);
    for (int k = 0; k < K; k++) x_q[k] = quantize_one(x[k], scale);
    return gamma;
}

#ifdef BITNET_NEON
static inline float absmax_neon(const float *x, int K) {
    float32x4_t m0 = vdupq_n_f32(0.0f), m1 = m0, m2 = m0, m3 = m0;
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(x + k)));
        m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(x + k + 4)));
        m2 = vmaxq_f32(m2, vabsq_f32(vld1q_f32(x + k + 8)));
        m3 = vmaxq_f32(m3, vabsq_f32(vld1q_f32(x + k + 12)));
    }
    float m = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
    for (; k < K; k++) {
        const float a = fabsf(x[k]);
        if (a > m) m = a;
    }
    return m;
}

/* Bit-exact with quantize_act_scalar: same single multiply, FCVTNS rounds
 * half-to-even like rintf, and SQXTN narrowing saturates to [-128, 127]. */
static inline float quantize_act_neon(const float *x, int8_t *x_q, int K) {
    const float gamma = fmaxf(absmax_neon(x, K), BITLINEAR_GAMMA_MIN);
    const float scale = quant_scale_from_gamma(gamma);
    const float32x4_t vs = vdupq_n_f32(scale);

    int k = 0;
    for (; k + 16 <= K; k += 16) {
        const int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k), vs));
        const int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k + 4), vs));
        const int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k + 8), vs));
        const int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k + 12), vs));
        const int16x8_t h0 = vqmovn_high_s32(vqmovn_s32(i0), i1);
        const int16x8_t h1 = vqmovn_high_s32(vqmovn_s32(i2), i3);
        vst1q_s8(x_q + k, vqmovn_high_s16(vqmovn_s16(h0), h1));
    }
    for (; k < K; k++) x_q[k] = quantize_one(x[k], scale);
    return gamma;
}

/* ------------------------------------------------------------------------- */
/* Dequantization                                                            */
/* ------------------------------------------------------------------------- */
#else
static inline float absmax_neon(const float *x, int K) { return absmax_scalar(x, K); }
static inline float quantize_act_neon(const float *x, int8_t *x_q, int K) {
    return quantize_act_scalar(x, x_q, K);
}
#endif

static inline float dequant_scale(float gamma, float beta) {
    return gamma * beta / BITLINEAR_QMAX;
}

static inline void dequantize_scalar(const int32_t *y_int, float *y, int M, float s) {
    for (int m = 0; m < M; m++) y[m] = (float)y_int[m] * s;
}

#ifdef BITNET_NEON
static inline void dequantize_neon(const int32_t *y_int, float *y, int M, float s) {
    int m = 0;
    for (; m + 16 <= M; m += 16) {
        vst1q_f32(y + m,      vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(y_int + m)), s));
        vst1q_f32(y + m + 4,  vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(y_int + m + 4)), s));
        vst1q_f32(y + m + 8,  vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(y_int + m + 8)), s));
        vst1q_f32(y + m + 12, vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(y_int + m + 12)), s));
    }
    for (; m < M; m++) y[m] = (float)y_int[m] * s;
}
#else
static inline void dequantize_neon(const int32_t *y_int, float *y, int M, float s) {
    dequantize_scalar(y_int, y, M, s);
}
#endif

#endif /* BITNET_BITLINEAR_H */
