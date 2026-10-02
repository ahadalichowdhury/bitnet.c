/*
 * transformer.c — BitNet b1.58 transformer forward pass (see transformer.h).
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE /* glibc: MAP_ANONYMOUS */
#define _DARWIN_C_SOURCE

#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "bitlinear.h"
#include "gemv.h"
#include "platform.h"
#include "q8.h"
#include "simd.h"

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

/* Attention runs on the calling thread below this many cached positions: the
 * work is tiny and fork/join would dominate. */
#define ATTN_PARALLEL_MIN_POS 64
/* Rows per work item for the f16 output projection. */
#define F16_ROW_BLOCK 64

_Static_assert(TRANSFORMER_BATCH >= 1 && TRANSFORMER_BATCH <= GEMM_MAX_T, "TRANSFORMER_BATCH must be in [1, GEMM_MAX_T]");

static inline uint64_t now_ns(void) { return platform_now_ns(); }

/* ========================================================================= */
/* Elementwise kernels                                                       */
/* ========================================================================= */

#ifdef BITNET_NEON

void rmsnorm_neon(float *out, const float *x, const float *w, int n, float eps) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const float32x4_t a = vld1q_f32(x + i), b = vld1q_f32(x + i + 4);
        const float32x4_t c = vld1q_f32(x + i + 8), d = vld1q_f32(x + i + 12);
        s0 = vfmaq_f32(s0, a, a);
        s1 = vfmaq_f32(s1, b, b);
        s2 = vfmaq_f32(s2, c, c);
        s3 = vfmaq_f32(s3, d, d);
    }
    float ss = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++) ss += x[i] * x[i];
    /* eps is added to the mean square inside the sqrt (HF BitNetRMSNorm). */
    const float r = 1.0f / sqrtf(ss / (float)n + eps);
    const float32x4_t vr = vdupq_n_f32(r);
    i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), vr), vld1q_f32(w + i)));
    for (; i < n; i++) out[i] = (x[i] * r) * w[i];
}

void residual_add_neon(float *x, const float *y, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(y + i)));
        vst1q_f32(x + i + 4, vaddq_f32(vld1q_f32(x + i + 4), vld1q_f32(y + i + 4)));
        vst1q_f32(x + i + 8, vaddq_f32(vld1q_f32(x + i + 8), vld1q_f32(y + i + 8)));
        vst1q_f32(x + i + 12, vaddq_f32(vld1q_f32(x + i + 12), vld1q_f32(y + i + 12)));
    }
    for (; i < n; i++) x[i] += y[i];
}

/* Cephes-style expf: e^x = 2^n * e^r, |r| <= ln2/2, degree-6 polynomial. */
static inline float32x4_t exp_f32x4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.0f));
    const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    const float32x4_t y = vaddq_f32(vfmaq_f32(r, p, vmulq_f32(r, r)), vdupq_n_f32(1.0f));
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}

void exp_neon(float *out, const float *x, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, exp_f32x4(vld1q_f32(x + i)));
    if (i < n) {
        float tmp[4] = {0};
        memcpy(tmp, x + i, (size_t)(n - i) * sizeof(float));
        vst1q_f32(tmp, exp_f32x4(vld1q_f32(tmp)));
        memcpy(out + i, tmp, (size_t)(n - i) * sizeof(float));
    }
}

void softmax_neon(float *x, int n) {
    float32x4_t vm = vdupq_n_f32(-INFINITY);
    int i = 0;
    for (; i + 4 <= n; i += 4) vm = vmaxq_f32(vm, vld1q_f32(x + i));
    float mx = vmaxvq_f32(vm);
    for (; i < n; i++) mx = x[i] > mx ? x[i] : mx;

    const float32x4_t vmx = vdupq_n_f32(mx);
    float32x4_t vs = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t e = exp_f32x4(vsubq_f32(vld1q_f32(x + i), vmx));
        vst1q_f32(x + i, e);
        vs = vaddq_f32(vs, e);
    }
    float sum = vaddvq_f32(vs);
    if (i < n) {
        float tmp[4] = {0};
        for (int k = 0; k < n - i; k++) tmp[k] = x[i + k] - mx;
        vst1q_f32(tmp, exp_f32x4(vld1q_f32(tmp)));
        for (int k = 0; k < n - i; k++) {
            x[i + k] = tmp[k];
            sum += tmp[k];
        }
    }
    const float32x4_t inv = vdupq_n_f32(1.0f / sum);
    i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), inv));
    for (; i < n; i++) x[i] *= 1.0f / sum;
}

void glu_neon(float *gate, const float *up, int n, int relu2) {
    int i = 0;
    if (relu2) {
        const float32x4_t zero = vdupq_n_f32(0.0f);
        for (; i + 4 <= n; i += 4) {
            const float32x4_t r = vmaxq_f32(vld1q_f32(gate + i), zero);
            vst1q_f32(gate + i, vmulq_f32(vmulq_f32(r, r), vld1q_f32(up + i)));
        }
        for (; i < n; i++) {
            const float r = gate[i] > 0.0f ? gate[i] : 0.0f;
            gate[i] = r * r * up[i];
        }
    } else {
        const float32x4_t one = vdupq_n_f32(1.0f);
        for (; i + 4 <= n; i += 4) {
            const float32x4_t g = vld1q_f32(gate + i);
            const float32x4_t sig = vdivq_f32(one, vaddq_f32(one, exp_f32x4(vnegq_f32(g))));
            vst1q_f32(gate + i, vmulq_f32(vmulq_f32(g, sig), vld1q_f32(up + i)));
        }
        for (; i < n; i++) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
    }
}

void apply_rope_neon(float *vec, int n_heads, int head_dim, const float *cos, const float *sin) {
    const int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *v1 = vec + (size_t)h * head_dim, *v2 = v1 + half;
        int i = 0;
        for (; i + 4 <= half; i += 4) {
            const float32x4_t x1 = vld1q_f32(v1 + i), x2 = vld1q_f32(v2 + i);
            const float32x4_t c = vld1q_f32(cos + i), s = vld1q_f32(sin + i);
            vst1q_f32(v1 + i, vfmsq_f32(vmulq_f32(x1, c), x2, s)); /* x1 c - x2 s */
            vst1q_f32(v2 + i, vfmaq_f32(vmulq_f32(x2, c), x1, s)); /* x2 c + x1 s */
        }
        for (; i < half; i++) {
            const float x1 = v1[i], x2 = v2[i];
            v1[i] = x1 * cos[i] - x2 * sin[i];
            v2[i] = x2 * cos[i] + x1 * sin[i];
        }
    }
}

static inline float dot_f32(const float *a, const float *b, int n) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline void axpy_f32(float *y, float a, const float *x, int n) {
    const float32x4_t va = vdupq_n_f32(a);
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
    for (; i < n; i++) y[i] += a * x[i];
}

/* exp(row[i] - mx) in place; returns the sum. */
static inline float exp_shift_sum(float *row, int n, float mx) {
    const float32x4_t vm = vdupq_n_f32(mx);
    float32x4_t vs = vdupq_n_f32(0.0f);
    int t = 0;
    for (; t + 4 <= n; t += 4) {
        const float32x4_t e = exp_f32x4(vsubq_f32(vld1q_f32(row + t), vm));
        vst1q_f32(row + t, e);
        vs = vaddq_f32(vs, e);
    }
    float l = vaddvq_f32(vs);
    for (; t < n; t++) {
        float tmp[4] = {row[t] - mx, 0.0f, 0.0f, 0.0f};
        vst1q_f32(tmp, exp_f32x4(vld1q_f32(tmp)));
        row[t] = tmp[0];
        l += tmp[0];
    }
    return l;
}

#else /* !BITNET_NEON: portable scalar kernels (same math, no SIMD) */

void rmsnorm_neon(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float r = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] * r) * w[i];
}

void residual_add_neon(float *x, const float *y, int n) {
    for (int i = 0; i < n; i++) x[i] += y[i];
}

void exp_neon(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++) out[i] = expf(x[i]);
}

static inline float exp_shift_sum(float *row, int n, float mx) {
    float l = 0.0f;
    for (int t = 0; t < n; t++) l += (row[t] = expf(row[t] - mx));
    return l;
}

void softmax_neon(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) mx = x[i] > mx ? x[i] : mx;
    const float inv = 1.0f / exp_shift_sum(x, n, mx);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void glu_neon(float *gate, const float *up, int n, int relu2) {
    for (int i = 0; i < n; i++) {
        if (relu2) {
            const float r = gate[i] > 0.0f ? gate[i] : 0.0f;
            gate[i] = r * r * up[i];
        } else {
            gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
        }
    }
}

void apply_rope_neon(float *vec, int n_heads, int head_dim, const float *cos, const float *sin) {
    const int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *v1 = vec + (size_t)h * head_dim, *v2 = v1 + half;
        for (int i = 0; i < half; i++) {
            const float x1 = v1[i], x2 = v2[i];
            v1[i] = x1 * cos[i] - x2 * sin[i];
            v2[i] = x2 * cos[i] + x1 * sin[i];
        }
    }
}

static inline float dot_f32(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline void axpy_f32(float *y, float a, const float *x, int n) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

#endif /* BITNET_NEON */

/* ========================================================================= */
/* f16 output projection                                                     */
/* ========================================================================= */

typedef struct {
    const uint16_t *W;
    const float    *x;
    float          *out;
    int             M, K;
} f16_ctx;

static void f16_rows(void *ctx_, size_t blk) {
    const f16_ctx *c = ctx_;
    const int r0 = (int)blk * F16_ROW_BLOCK;
    const int r1 = r0 + F16_ROW_BLOCK < c->M ? r0 + F16_ROW_BLOCK : c->M;
    for (int r = r0; r < r1; r++) {
        const uint16_t *w = c->W + (size_t)r * c->K;
#ifdef BITNET_NEON
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        int k = 0;
        for (; k + 16 <= c->K; k += 16) {
            const float16x8_t h0 = vreinterpretq_f16_u16(vld1q_u16(w + k));
            const float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(w + k + 8));
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(c->x + k));
            a1 = vfmaq_f32(a1, vcvt_high_f32_f16(h0), vld1q_f32(c->x + k + 4));
            a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(c->x + k + 8));
            a3 = vfmaq_f32(a3, vcvt_high_f32_f16(h1), vld1q_f32(c->x + k + 12));
        }
        float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#elif defined(BITNET_F16C)
        /* x86 F16C: VCVTPH2PS widens 8 halves per instruction (exactly, like
         * half_to_float); four independent FMA chains of 8 lanes. */
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int k = 0;
        for (; k + 32 <= c->K; k += 32) {
            a0 = BITNET_FMADD(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + k))),
                              _mm256_loadu_ps(c->x + k), a0);
            a1 = BITNET_FMADD(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + k + 8))),
                              _mm256_loadu_ps(c->x + k + 8), a1);
            a2 = BITNET_FMADD(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + k + 16))),
                              _mm256_loadu_ps(c->x + k + 16), a2);
            a3 = BITNET_FMADD(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + k + 24))),
                              _mm256_loadu_ps(c->x + k + 24), a3);
        }
        for (; k + 8 <= c->K; k += 8)
            a0 = BITNET_FMADD(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + k))),
                              _mm256_loadu_ps(c->x + k), a0);
        const __m256 v = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
        float s = _mm_cvtss_f32(h);
#else
        float s = 0.0f;
        int k = 0;
#endif
        for (; k < c->K; k++) s += half_to_float(w[k]) * c->x[k];
        c->out[r] = s;
    }
}

void gemv_f16_neon(const uint16_t *W, const float *x, float *out, int M, int K, threadpool *pool) {
    f16_ctx c = {W, x, out, M, K};
    threadpool_run(pool, ((size_t)M + F16_ROW_BLOCK - 1) / F16_ROW_BLOCK, &c, f16_rows);
}

typedef struct {
    const float *W;
    const float *x;
    float       *out;
    int          M, K;
} f32_ctx;

static void f32_rows(void *ctx_, size_t blk) {
    const f32_ctx *c = ctx_;
    const int r0 = (int)blk * F16_ROW_BLOCK;
    const int r1 = r0 + F16_ROW_BLOCK < c->M ? r0 + F16_ROW_BLOCK : c->M;
    for (int r = r0; r < r1; r++) c->out[r] = dot_f32(c->W + (size_t)r * c->K, c->x, c->K);
}

typedef struct {
    const int8_t *W;   /* [M, K] quants */
    const float  *ws;  /* [M, K/32] block scales */
    const int8_t *xq;  /* [K] quantized input */
    const float  *xs;  /* [K/32] input block scales */
    float        *out;
    int           M, K;
} q8_ctx;

static void q8_rows(void *ctx_, size_t blk) {
    const q8_ctx *c = ctx_;
    const int nb = c->K / Q8_BLOCK;
    const int r0 = (int)blk * F16_ROW_BLOCK;
    const int r1 = r0 + F16_ROW_BLOCK < c->M ? r0 + F16_ROW_BLOCK : c->M;
    for (int r = r0; r < r1; r++)
        c->out[r] = q8_dot(c->W + (size_t)r * c->K, c->ws + (size_t)r * nb, c->xq, c->xs, nb);
}

/* ========================================================================= */
/* BitLinear projections sharing one quantized input, one pool job          */
/* ========================================================================= */

typedef struct {
    gemv_ctx c[3];
    size_t   first_blk[4];
    int      n;
} multi_gemv;

static void multi_gemv_block(void *ctx_, size_t b) {
    multi_gemv *m = ctx_;
    int i = 0;
    while (b >= m->first_blk[i + 1]) i++;
    gemv_row_block(&m->c[i], b - m->first_blk[i]);
}

typedef struct {
    gemm_ctx c[3];
    size_t   first_blk[4];
    int      n;
} multi_gemm;

static void multi_gemm_block(void *ctx_, size_t b) {
    multi_gemm *m = ctx_;
    int i = 0;
    while (b >= m->first_blk[i + 1]) i++;
    gemm_row_block(&m->c[i], b - m->first_blk[i]);
}

/* For each matrix i and token t < T:
 *   out[i][t * out_stride[i] + r] = BitLinear(W[i], xq[t]) with scale gamma[t].
 * All row blocks of all matrices go into one pool job so small projections
 * (k, v) do not each pay a fork/join. One token uses the GEMV kernels; more
 * use the batched GEMM, which decodes each weight block once for GEMM_TB
 * tokens. Both produce identical int32 sums. */
static void bitlinear_batch(RunState *s, int T, const float *gamma, int K,
                            const bitnet_tensor *const *W, float *const *out,
                            const size_t *out_stride, int n) {
    size_t off = 0;
    if (T == 1) {
        multi_gemv m;
        m.n = n;
        m.first_blk[0] = 0;
        for (int i = 0; i < n; i++) {
            const ternary_layout L = W[i]->dtype == BITNET_DTYPE_TERNARY_I128 ? TERNARY_I128 : TERNARY_ROW4;
            m.c[i] = (gemv_ctx){s->xq, (const uint8_t *)W[i]->data, s->yi + off, W[i]->rows, K,
                                ternary_row_bytes(K, L), L};
            m.first_blk[i + 1] = m.first_blk[i] + ((size_t)W[i]->rows + GEMV_ROW_BLOCK - 1) / GEMV_ROW_BLOCK;
            off += (size_t)W[i]->rows;
        }
        threadpool_run(s->pool, m.first_blk[n], &m, multi_gemv_block);
    } else {
        multi_gemm m;
        m.n = n;
        m.first_blk[0] = 0;
        for (int i = 0; i < n; i++) {
            const ternary_layout L = W[i]->dtype == BITNET_DTYPE_TERNARY_I128 ? TERNARY_I128 : TERNARY_ROW4;
            m.c[i] = (gemm_ctx){s->xq, s->xq_stride, s->yi + off, s->yi_stride, T,
                                (const uint8_t *)W[i]->data, W[i]->rows, K, ternary_row_bytes(K, L), L};
            m.first_blk[i + 1] = m.first_blk[i] + ((size_t)W[i]->rows + GEMV_ROW_BLOCK - 1) / GEMV_ROW_BLOCK;
            off += (size_t)W[i]->rows;
        }
        threadpool_run(s->pool, m.first_blk[n], &m, multi_gemm_block);
    }
    for (int t = 0; t < T; t++) {
        off = 0;
        for (int i = 0; i < n; i++) {
            dequantize_neon(s->yi + (size_t)t * s->yi_stride + off, out[i] + (size_t)t * out_stride[i],
                            W[i]->rows, dequant_scale(gamma[t], W[i]->scale));
            off += (size_t)W[i]->rows;
        }
    }
}

/* ========================================================================= */
/* Attention (GQA-grouped, split-softmax over position chunks)               */
/* ========================================================================= */

typedef struct {
    float       *out;
    const float *q, *kc, *vc;
    float       *part;
    int          pos, n_kv, group, hd, kv_dim, n_chunks;
    float        scale;
} attn_job;

static size_t attn_slot(const attn_job *a, int head, int chunk) {
    return ((size_t)head * a->n_chunks + chunk) * (size_t)(a->hd + 2);
}

/* One (kv head, chunk) item: scores for every query head of the group, a local
 * softmax (max m, sum l) and the unnormalized value sum o, stored as
 * part[head][chunk] = {m, l, o[hd]}. Each K and V row is read once. */
static void attn_item(void *ctx, size_t item) {
    const attn_job *a = ctx;
    const int kvh = (int)item / a->n_chunks, c = (int)item % a->n_chunks;
    const int t0 = c * ATTN_CHUNK, t1 = t0 + ATTN_CHUNK < a->pos + 1 ? t0 + ATTN_CHUNK : a->pos + 1;
    const int nt = t1 - t0, hd = a->hd, G = a->group;
    const float *K = a->kc + (size_t)kvh * hd, *V = a->vc + (size_t)kvh * hd;
    float sc[ATTN_MAX_GROUP][ATTN_CHUNK];

    const float *qg = a->q + (size_t)kvh * G * hd;
#ifdef BITNET_NEON
    if (G == 4 && hd % 8 == 0) {
        /* Score pass for 4 query heads per K row: each K element is loaded once
         * and feeds 4 FMAs (vs. one load per FMA with per-head dot products). */
        for (int t = t0; t < t1; t++) {
            const float *k = K + (size_t)t * a->kv_dim;
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
            float32x4_t u0 = s0, u1 = s0, u2 = s0, u3 = s0;
            for (int i = 0; i < hd; i += 8) {
                const float32x4_t k0 = vld1q_f32(k + i), k1 = vld1q_f32(k + i + 4);
                s0 = vfmaq_f32(s0, vld1q_f32(qg + i), k0);
                u0 = vfmaq_f32(u0, vld1q_f32(qg + i + 4), k1);
                s1 = vfmaq_f32(s1, vld1q_f32(qg + hd + i), k0);
                u1 = vfmaq_f32(u1, vld1q_f32(qg + hd + i + 4), k1);
                s2 = vfmaq_f32(s2, vld1q_f32(qg + 2 * hd + i), k0);
                u2 = vfmaq_f32(u2, vld1q_f32(qg + 2 * hd + i + 4), k1);
                s3 = vfmaq_f32(s3, vld1q_f32(qg + 3 * hd + i), k0);
                u3 = vfmaq_f32(u3, vld1q_f32(qg + 3 * hd + i + 4), k1);
            }
            sc[0][t - t0] = vaddvq_f32(vaddq_f32(s0, u0)) * a->scale;
            sc[1][t - t0] = vaddvq_f32(vaddq_f32(s1, u1)) * a->scale;
            sc[2][t - t0] = vaddvq_f32(vaddq_f32(s2, u2)) * a->scale;
            sc[3][t - t0] = vaddvq_f32(vaddq_f32(s3, u3)) * a->scale;
        }
    } else
#endif
    {
        for (int t = t0; t < t1; t++) {
            const float *k = K + (size_t)t * a->kv_dim;
            for (int g = 0; g < G; g++) sc[g][t - t0] = dot_f32(qg + (size_t)g * hd, k, hd) * a->scale;
        }
    }
    float *o[ATTN_MAX_GROUP];
    for (int g = 0; g < G; g++) {
        float *row = sc[g];
        float mx = row[0];
        for (int t = 1; t < nt; t++) mx = row[t] > mx ? row[t] : mx;
        const float l = exp_shift_sum(row, nt, mx);
        float *slot = a->part + attn_slot(a, kvh * G + g, c);
        slot[0] = mx;
        slot[1] = l;
        o[g] = slot + 2;
        memset(o[g], 0, (size_t)hd * sizeof(float));
    }
#ifdef BITNET_NEON
    if (G == 4 && hd % 16 == 0) {
        /* Register-blocked value pass for the common 4-query-heads-per-KV-head
         * case: 4 heads x 16 dims of output stay in 16 NEON registers while the
         * chunk's V rows stream by, instead of a load/store per row. */
        for (int i0 = 0; i0 < hd; i0 += 16) {
            float32x4_t acc[4][4];
            for (int g = 0; g < 4; g++)
                for (int j = 0; j < 4; j++) acc[g][j] = vdupq_n_f32(0.0f);
            for (int t = t0; t < t1; t++) {
                const float *v = V + (size_t)t * a->kv_dim + i0;
                const float32x4_t v0 = vld1q_f32(v), v1 = vld1q_f32(v + 4);
                const float32x4_t v2 = vld1q_f32(v + 8), v3 = vld1q_f32(v + 12);
                for (int g = 0; g < 4; g++) {
                    const float p = sc[g][t - t0];
                    acc[g][0] = vfmaq_n_f32(acc[g][0], v0, p);
                    acc[g][1] = vfmaq_n_f32(acc[g][1], v1, p);
                    acc[g][2] = vfmaq_n_f32(acc[g][2], v2, p);
                    acc[g][3] = vfmaq_n_f32(acc[g][3], v3, p);
                }
            }
            for (int g = 0; g < 4; g++)
                for (int j = 0; j < 4; j++) vst1q_f32(o[g] + i0 + 4 * j, acc[g][j]);
        }
        return;
    }
#endif
    for (int t = t0; t < t1; t++) {
        const float *v = V + (size_t)t * a->kv_dim;
        for (int g = 0; g < G; g++) axpy_f32(o[g], sc[g][t - t0], v, hd);
    }
}

size_t attention_scratch_floats(int n_heads, int head_dim, int max_seq_len) {
    const size_t chunks = ((size_t)max_seq_len + ATTN_CHUNK - 1) / ATTN_CHUNK;
    return (size_t)n_heads * chunks * (size_t)(head_dim + 2);
}

void attention_neon(float *out, const float *q, const float *k_cache, const float *v_cache, int pos,
                    int n_heads, int n_kv_heads, int head_dim, int kv_dim, float *part, threadpool *pool) {
    attn_job a = {out, q, k_cache, v_cache, part, pos, n_kv_heads, n_heads / n_kv_heads, head_dim,
                  kv_dim, (pos + ATTN_CHUNK) / ATTN_CHUNK, 1.0f / sqrtf((float)head_dim)};
    const size_t items = (size_t)n_kv_heads * (size_t)a.n_chunks;
    if (pos + 1 >= ATTN_PARALLEL_MIN_POS) {
        threadpool_run(pool, items, &a, attn_item);
    } else {
        for (size_t i = 0; i < items; i++) attn_item(&a, i);
    }

    /* Merge chunk partials: out = sum_c e^(m_c - M) o_c / sum_c e^(m_c - M) l_c. */
    for (int h = 0; h < n_heads; h++) {
        float *dst = out + (size_t)h * head_dim;
        if (a.n_chunks == 1) {
            const float *slot = part + attn_slot(&a, h, 0);
            const float inv = 1.0f / slot[1];
            for (int i = 0; i < head_dim; i++) dst[i] = slot[2 + i] * inv;
            continue;
        }
        float M = -INFINITY;
        for (int c = 0; c < a.n_chunks; c++) {
            const float m = part[attn_slot(&a, h, c)];
            M = m > M ? m : M;
        }
        float L = 0.0f;
        memset(dst, 0, (size_t)head_dim * sizeof(float));
        for (int c = 0; c < a.n_chunks; c++) {
            const float *slot = part + attn_slot(&a, h, c);
            const float w = expf(slot[0] - M);
            L += w * slot[1];
            axpy_f32(dst, w, slot + 2, head_dim);
        }
        const float inv = 1.0f / L;
        for (int i = 0; i < head_dim; i++) dst[i] *= inv;
    }
}

/* ========================================================================= */
/* RunState                                                                  */
/* ========================================================================= */

static size_t align64(size_t x) { return (x + 63) & ~(size_t)63; }

int runstate_init(RunState *s, const BitNetModel *m, int max_seq_len, int n_threads, char *err,
                  size_t err_len) {
    memset(s, 0, sizeof(*s));
    const bitnet_config *c = &m->config;
    if (max_seq_len <= 0 || max_seq_len > c->max_seq_len) max_seq_len = c->max_seq_len;
    const int hd = c->head_dim, q_dim = c->n_heads * hd, kv_dim = c->n_kv_heads * hd;
    if (hd % 2 != 0) {
        snprintf(err, err_len, "head_dim %d must be even for RoPE", hd);
        return -1;
    }
    if (c->n_heads % c->n_kv_heads != 0 || c->n_heads / c->n_kv_heads > ATTN_MAX_GROUP) {
        snprintf(err, err_len, "GQA group %d/%d unsupported (max %d)", c->n_heads, c->n_kv_heads,
                 ATTN_MAX_GROUP);
        return -1;
    }
    if ((c->flags & BITNET_FLAG_SUB_NORMS) && q_dim != c->dim) {
        snprintf(err, err_len, "attn_sub_norm needs n_heads*head_dim == dim");
        return -1;
    }
    const int xq_len = c->dim > c->hidden_dim ? (c->dim > q_dim ? c->dim : q_dim)
                                              : (c->hidden_dim > q_dim ? c->hidden_dim : q_dim);
    const size_t cache = (size_t)c->n_layers * max_seq_len * kv_dim;
    const size_t rope = (size_t)max_seq_len * (hd / 2);

    /* Activation buffers hold TRANSFORMER_BATCH rows (prefill); single-token
     * passes use row 0. xq rows are padded to whole GEMM_TB groups. */
    const size_t B = TRANSFORMER_BATCH, BQ = (TRANSFORMER_BATCH + GEMM_TB - 1) / GEMM_TB * GEMM_TB;
    s->xq_stride = align64((size_t)xq_len);
    s->yi_stride = (size_t)q_dim + 2 * kv_dim + 2 * (size_t)c->hidden_dim + c->dim;
    struct { void **p; size_t bytes; } parts[] = {
        {(void **)&s->x, B * c->dim * 4},              {(void **)&s->xb, B * c->dim * 4},
        {(void **)&s->xb2, B * q_dim * 4},             {(void **)&s->q, B * q_dim * 4},
        {(void **)&s->attn_part, attention_scratch_floats(c->n_heads, hd, max_seq_len) * 4},
        {(void **)&s->hb, B * c->hidden_dim * 4},      {(void **)&s->hb2, B * c->hidden_dim * 4},
        {(void **)&s->logits, (size_t)c->vocab_size * 4},
        {(void **)&s->xq, BQ * s->xq_stride},
        {(void **)&s->yi, B * s->yi_stride * 4},
        {(void **)&s->key_cache, cache * 4},           {(void **)&s->value_cache, cache * 4},
        {(void **)&s->rope_cos, rope * 4},             {(void **)&s->rope_sin, rope * 4},
    };
    const int n_parts = (int)(sizeof(parts) / sizeof(parts[0]));
    size_t total = 0;
    for (int i = 0; i < n_parts; i++) total += align64(parts[i].bytes);

    /* One anonymous mapping: zero-filled, page-aligned, and the KV cache only
     * becomes resident as positions are written. */
    void *arena = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (arena == MAP_FAILED) {
        snprintf(err, err_len, "cannot map %.1f MiB for the run state", total / 1048576.0);
        return -1;
    }
    size_t off = 0;
    for (int i = 0; i < n_parts; i++) {
        *parts[i].p = (uint8_t *)arena + off;
        off += align64(parts[i].bytes);
    }
    s->arena = arena;
    s->arena_bytes = total;
    s->max_seq_len = max_seq_len;
    s->pool = threadpool_create(n_threads, err, err_len);
    if (!s->pool) {
        munmap(arena, total);
        memset(s, 0, sizeof(*s));
        return -1;
    }

    /* RoPE tables, computed like HF: inv_freq = 1 / theta^(2i/d) in float32,
     * angle = float(pos) * inv_freq in float32. */
    for (int i = 0; i < hd / 2; i++) {
        const float inv_freq = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)hd);
        for (int p = 0; p < max_seq_len; p++) {
            const float ang = (float)p * inv_freq;
            s->rope_cos[(size_t)p * (hd / 2) + i] = cosf(ang);
            s->rope_sin[(size_t)p * (hd / 2) + i] = sinf(ang);
        }
    }
    return 0;
}

void runstate_free(RunState *s) {
    threadpool_destroy(s->pool);
    if (s->arena) munmap(s->arena, s->arena_bytes);
    memset(s, 0, sizeof(*s));
}

/* ========================================================================= */
/* Forward pass                                                              */
/* ========================================================================= */

#define PROF_MARK(field)                                         \
    do {                                                         \
        if (s->profile) {                                        \
            const uint64_t t_ = now_ns();                        \
            s->prof.field += t_ - t_mark;                        \
            t_mark = t_;                                         \
        }                                                        \
    } while (0)

void transformer_forward(int token_id, int pos, const BitNetModel *m, RunState *s) {
    transformer_forward_ex(token_id, pos, m, s, 1);
}

void transformer_forward_ex(int token_id, int pos, const BitNetModel *m, RunState *s,
                            int compute_logits) {
    const int32_t tok = token_id;
    transformer_forward_batch(&tok, 1, pos, m, s, compute_logits);
}

/* Final norm + output projection of residual row x into s->logits. */
static void output_logits(const BitNetModel *m, RunState *s, const float *x) {
    const bitnet_config *c = &m->config;
    const int dim = c->dim;
    rmsnorm_neon(s->x, x, m->output_norm->data, dim, c->norm_eps);
    if (m->output->dtype == BITNET_DTYPE_Q8) {
        /* x -> int8 blocks (xq is free after the last layer; xb holds the scales). */
        q8_quantize(s->x, s->xq, s->xb, dim);
        q8_ctx qc = {m->output->data, q8_scales(m->output->data, c->vocab_size, dim), s->xq,
                     s->xb, s->logits, c->vocab_size, dim};
        threadpool_run(s->pool, ((size_t)c->vocab_size + F16_ROW_BLOCK - 1) / F16_ROW_BLOCK, &qc,
                       q8_rows);
    } else if (m->output->dtype == BITNET_DTYPE_F16) {
        gemv_f16_neon(m->output->data, s->x, s->logits, c->vocab_size, dim, s->pool);
    } else {
        f32_ctx fc = {m->output->data, s->x, s->logits, c->vocab_size, dim};
        threadpool_run(s->pool, ((size_t)c->vocab_size + F16_ROW_BLOCK - 1) / F16_ROW_BLOCK, &fc,
                       f32_rows);
    }
}

/* Up to TRANSFORMER_BATCH tokens at consecutive positions pos..pos+n-1. */
static void forward_chunk(const int32_t *tokens, int n, int pos, const BitNetModel *m, RunState *s,
                          int compute_logits) {
    const bitnet_config *c = &m->config;
    const int dim = c->dim, hd = c->head_dim, hidden = c->hidden_dim;
    const int q_dim = c->n_heads * hd, kv_dim = c->n_kv_heads * hd;
    const float eps = c->norm_eps;
    const int relu2 = c->ffn_act == BITNET_ACT_RELU2;
    const size_t xs = s->xq_stride;
    float gamma[TRANSFORMER_BATCH];
    const uint64_t t_start = s->profile ? now_ns() : 0;
    uint64_t t_mark = t_start;

#define ROW(buf, t, width) ((buf) + (size_t)(t) * (width))
    for (int t = 0; t < n; t++) bitnet_embedding_row(m, tokens[t], ROW(s->x, t, dim));
    PROF_MARK(embed);

    for (int l = 0; l < c->n_layers; l++) {
        const bitnet_layer *L = &m->layers[l];
        const size_t layer_off = (size_t)l * s->max_seq_len * kv_dim;
        float *kc = s->key_cache + layer_off + (size_t)pos * kv_dim;
        float *vc = s->value_cache + layer_off + (size_t)pos * kv_dim;

        /* ---- Attention: QKV projections (K/V written straight into the cache). */
        for (int t = 0; t < n; t++) {
            rmsnorm_neon(ROW(s->xb, t, dim), ROW(s->x, t, dim), L->attn_norm->data, dim, eps);
            gamma[t] = quantize_act_neon(ROW(s->xb, t, dim), ROW(s->xq, t, xs), dim);
        }
        {
            const bitnet_tensor *W[3] = {L->wq, L->wk, L->wv};
            float *out[3] = {s->q, kc, vc};
            const size_t os[3] = {(size_t)q_dim, (size_t)kv_dim, (size_t)kv_dim};
            bitlinear_batch(s, n, gamma, dim, W, out, os, 3);
        }
        for (int t = 0; t < n; t++) {
            const size_t rp = (size_t)(pos + t) * (hd / 2);
            apply_rope_neon(ROW(s->q, t, q_dim), c->n_heads, hd, s->rope_cos + rp, s->rope_sin + rp);
            apply_rope_neon(ROW(kc, t, kv_dim), c->n_kv_heads, hd, s->rope_cos + rp, s->rope_sin + rp);
        }
        PROF_MARK(attn_proj);

        /* ---- Causal attention for each token over positions 0..pos+t (GQA). */
        for (int t = 0; t < n; t++)
            attention_neon(ROW(s->xb2, t, q_dim), ROW(s->q, t, q_dim), s->key_cache + layer_off,
                           s->value_cache + layer_off, pos + t, c->n_heads, c->n_kv_heads, hd, kv_dim,
                           s->attn_part, s->pool);
        PROF_MARK(attention);

        for (int t = 0; t < n; t++) {
            float *a = ROW(s->xb2, t, q_dim);
            if (L->attn_sub_norm) rmsnorm_neon(a, a, L->attn_sub_norm->data, q_dim, eps);
            gamma[t] = quantize_act_neon(a, ROW(s->xq, t, xs), q_dim);
        }
        {
            const bitnet_tensor *W[1] = {L->wo};
            float *out[1] = {s->xb};
            const size_t os[1] = {(size_t)dim};
            bitlinear_batch(s, n, gamma, q_dim, W, out, os, 1);
        }
        for (int t = 0; t < n; t++) residual_add_neon(ROW(s->x, t, dim), ROW(s->xb, t, dim), dim);
        PROF_MARK(attn_proj);

        /* ---- Feed-forward: down(sub_norm(act(gate(h)) * up(h))). */
        for (int t = 0; t < n; t++) {
            rmsnorm_neon(ROW(s->xb, t, dim), ROW(s->x, t, dim), L->ffn_norm->data, dim, eps);
            gamma[t] = quantize_act_neon(ROW(s->xb, t, dim), ROW(s->xq, t, xs), dim);
        }
        {
            const bitnet_tensor *W[2] = {L->w_gate, L->w_up};
            float *out[2] = {s->hb, s->hb2};
            const size_t os[2] = {(size_t)hidden, (size_t)hidden};
            bitlinear_batch(s, n, gamma, dim, W, out, os, 2);
        }
        for (int t = 0; t < n; t++) {
            float *h = ROW(s->hb, t, hidden);
            glu_neon(h, ROW(s->hb2, t, hidden), hidden, relu2);
            if (L->ffn_sub_norm) rmsnorm_neon(h, h, L->ffn_sub_norm->data, hidden, eps);
            gamma[t] = quantize_act_neon(h, ROW(s->xq, t, xs), hidden);
        }
        {
            const bitnet_tensor *W[1] = {L->w_down};
            float *out[1] = {s->xb};
            const size_t os[1] = {(size_t)dim};
            bitlinear_batch(s, n, gamma, hidden, W, out, os, 1);
        }
        for (int t = 0; t < n; t++) residual_add_neon(ROW(s->x, t, dim), ROW(s->xb, t, dim), dim);
        PROF_MARK(ffn);
    }

    /* ---- Final norm + tied output projection, for the last token only. */
    if (compute_logits) {
        output_logits(m, s, ROW(s->x, n - 1, dim));
        PROF_MARK(logits);
    }
#undef ROW
    if (s->profile) {
        s->prof.total += now_ns() - t_start;
        s->prof.tokens += (uint64_t)n;
    }
}

void transformer_forward_batch(const int32_t *tokens, int n, int pos, const BitNetModel *m,
                               RunState *s, int compute_logits) {
    const bitnet_config *c = &m->config;
    if (n < 1 || pos < 0 || pos > s->max_seq_len - n) {
        fprintf(stderr, "transformer_forward: %d tokens at pos %d out of range\n", n, pos);
        abort();
    }
    for (int t = 0; t < n; t++)
        if (tokens[t] < 0 || tokens[t] >= c->vocab_size) {
            fprintf(stderr, "transformer_forward: token %d / pos %d out of range\n", tokens[t], pos + t);
            abort();
        }
    for (int i = 0; i < n; i += TRANSFORMER_BATCH) {
        const int k = n - i < TRANSFORMER_BATCH ? n - i : TRANSFORMER_BATCH;
        forward_chunk(tokens + i, k, pos + i, m, s, compute_logits && i + k == n);
    }
}
