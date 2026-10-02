/*
 * test_bitlinear.c — BitLinear (BitNet b1.58) forward pass: float in, float out.
 *
 *   gamma   = max(max|x|, 1e-5)                         (per-token absmax)
 *   x_q     = clip(round(x * 127 / gamma), -128, 127)   -> int8_t
 *   y_int   = W_q * x_q                                 (ternary GEMV, gemv.h)
 *   y       = y_int * (gamma * beta / 127)
 *
 * W_q is ternary {-1, 0, +1} with per-layer scale beta = mean|W| (absmean
 * quantization, as in the b1.58 paper / Microsoft reference code).
 *
 * Numerical conventions (chosen to match the PyTorch reference so exported
 * models reproduce bit-for-bit):
 *   - round() is round-half-to-even (torch.round), i.e. rintf / FCVTNS.
 *   - gamma is clamped to >= 1e-5 (torch: .clamp_(min=1e-5)); an all-zero
 *     input quantizes to zeros and produces a zero output, never NaN.
 *   - Inputs are assumed finite; NaN/Inf activations are not sanitized.
 *   - y_int -> float is exact while |y_int| < 2^24, i.e. K < 2^17.
 *
 * Quantization kernels live in bitlinear.h (shared with the transformer).
 *
 * Build & run: `make test` (binary: build/test_bitlinear); `make asan` for sanitizers.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#undef NDEBUG /* verification asserts must always run */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "platform.h"
#include "bitlinear.h"
#include "gemv.h"

/* ------------------------------------------------------------------------- */
/* Layer and workspace                                                       */
/* ------------------------------------------------------------------------- */

typedef struct {
    const uint8_t *W;    /* packed ternary weights, M rows x ceil(K/4) bytes */
    float          beta; /* weight scale: mean|W| of the original float weights */
    int            M;    /* output features */
    int            K;    /* input features */
} bitlinear_layer;

/* Pre-allocated scratch so the forward pass never allocates. */
typedef struct {
    int8_t  *x_q;   /* [K] */
    int32_t *y_int; /* [M] */
} bitlinear_workspace;

/* ------------------------------------------------------------------------- */
/* Forward pass                                                              */
/* ------------------------------------------------------------------------- */

/* Note: the full forward costs ~10 us more than quant + GEMV + dequant timed in
 * isolation. That is the GEMV workers pulling the freshly written x_q from the
 * calling core's cache; an isolated GEMV benchmark re-reads an already-shared
 * x_q and hides it. Fusing the dequant into the workers was measured and gave
 * no gain, so the stages stay separate. */
void bitlinear_forward_neon(const bitlinear_layer *L, const float *x, float *y,
                            bitlinear_workspace *ws) {
    const float gamma = quantize_act_neon(x, ws->x_q, L->K);
    gemv_bitnet_neon(ws->x_q, L->W, ws->y_int, L->M, L->K);
    dequantize_neon(ws->y_int, y, L->M, dequant_scale(gamma, L->beta));
}

void bitlinear_forward_scalar(const bitlinear_layer *L, const float *x, float *y,
                              bitlinear_workspace *ws) {
    const float gamma = quantize_act_scalar(x, ws->x_q, L->K);
    gemv_scalar(ws->x_q, L->W, ws->y_int, L->M, L->K);
    dequantize_scalar(ws->y_int, y, L->M, dequant_scale(gamma, L->beta));
}

/* ------------------------------------------------------------------------- */
/* Offline weight quantization (absmean) — used to build test layers         */
/* ------------------------------------------------------------------------- */

/* beta = max(mean|W|, 1e-5); W_q = clip(round(W / beta), -1, 1), packed 2-bit.
 * Optionally writes the dequantized weights W_q * beta to `w_deq`. */
static float quantize_weights_absmean(const float *W, int M, int K, uint8_t *packed,
                                      float *w_deq) {
    double sum = 0.0;
    for (size_t i = 0; i < (size_t)M * K; i++) sum += fabs((double)W[i]);
    const float beta = fmaxf((float)(sum / ((double)M * K)), BITLINEAR_GAMMA_MIN);
    const float inv = 1.0f / beta;

    const size_t stride = gemv_row_stride(K);
    memset(packed, 0, (size_t)M * stride);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            const size_t i = (size_t)m * K + k;
            float q = rintf(W[i] * inv);
            q = q > 1.0f ? 1.0f : (q < -1.0f ? -1.0f : q);
            const uint8_t code = q > 0.0f ? 0x1 : (q < 0.0f ? 0x2 : 0x0);
            packed[(size_t)m * stride + (k >> 2)] |= (uint8_t)(code << ((k & 3) * 2));
            if (w_deq) w_deq[i] = q * beta;
        }
    }
    return beta;
}

/* ------------------------------------------------------------------------- */
/* Float32 reference                                                         */
/* ------------------------------------------------------------------------- */

/* Plain float32 linear layer y = W x (float32 tensors; double accumulator so
 * the reference's own rounding error is negligible next to quantization). */
static void linear_f32_reference(const float *W, const float *x, float *y, int M, int K) {
    for (int m = 0; m < M; m++) {
        const float *row = W + (size_t)m * K;
        double acc = 0.0;
        for (int k = 0; k < K; k++) acc += (double)row[k] * (double)x[k];
        y[m] = (float)acc;
    }
}

/* ------------------------------------------------------------------------- */
/* Test utilities                                                            */
/* ------------------------------------------------------------------------- */

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static inline uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

static inline float rand_uniform(void) { /* (0, 1) */
    return ((float)(xorshift64() >> 40) + 0.5f) / (float)(1u << 24);
}

static inline float rand_normal(void) {
    const float u1 = rand_uniform(), u2 = rand_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static void *aligned_buf(size_t bytes) {
    void *p = NULL;
    if (bytes == 0) bytes = 64;
    if (posix_memalign(&p, 64, bytes) != 0) {
        fprintf(stderr, "posix_memalign(%zu) failed\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

typedef enum { ACT_GAUSSIAN, ACT_UNIFORM, ACT_OUTLIER, ACT_TINY, ACT_ZERO } act_dist;

static const char *act_dist_name(act_dist d) {
    switch (d) {
    case ACT_GAUSSIAN: return "gaussian";
    case ACT_UNIFORM:  return "uniform";
    case ACT_OUTLIER:  return "gauss+outliers";
    case ACT_TINY:     return "tiny (1e-3)";
    case ACT_ZERO:     return "all-zero";
    }
    return "?";
}

static void fill_act(float *x, int K, act_dist d) {
    for (int k = 0; k < K; k++) {
        switch (d) {
        case ACT_GAUSSIAN: x[k] = rand_normal(); break;
        case ACT_UNIFORM:  x[k] = 2.0f * rand_uniform() - 1.0f; break;
        case ACT_OUTLIER:  x[k] = rand_normal(); break;
        case ACT_TINY:     x[k] = 1e-3f * rand_normal(); break;
        case ACT_ZERO:     x[k] = 0.0f; break;
        }
    }
    if (d == ACT_OUTLIER) /* LLM-style channel outliers: a few features ~20 sigma */
        for (int i = 0; i < 4 && i < K; i++) x[(i * 997) % K] = (i & 1 ? -20.0f : 20.0f);
}

static double rms(const float *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
    return n ? sqrt(s / n) : 0.0;
}

typedef struct { double rmse, nrmse, cosine; } err_stats;

static err_stats compare(const float *y, const float *ref, int M) {
    double se = 0.0, dot = 0.0, ny = 0.0, nr = 0.0;
    for (int m = 0; m < M; m++) {
        const double d = (double)y[m] - ref[m];
        se  += d * d;
        dot += (double)y[m] * ref[m];
        ny  += (double)y[m] * y[m];
        nr  += (double)ref[m] * ref[m];
    }
    err_stats s;
    s.rmse   = sqrt(se / M);
    s.nrmse  = nr > 0.0 ? s.rmse / sqrt(nr / M) : (s.rmse == 0.0 ? 0.0 : INFINITY);
    s.cosine = (ny > 0.0 && nr > 0.0) ? dot / sqrt(ny * nr) : (ny == nr ? 1.0 : 0.0);
    return s;
}

/* ------------------------------------------------------------------------- */
/* Verification                                                              */
/* ------------------------------------------------------------------------- */

/* 1) NEON quantize/dequantize are bit-exact with scalar, incl. tails and ties. */
static void verify_quant_kernels(void) {
    enum { MAX_K = 11008 + 37 };
    float   *x  = aligned_buf(MAX_K * sizeof(float));
    int8_t  *qs = aligned_buf(MAX_K);
    int8_t  *qn = aligned_buf(MAX_K);
    int32_t *yi = aligned_buf(MAX_K * sizeof(int32_t));
    float   *ys = aligned_buf(MAX_K * sizeof(float));
    float   *yn = aligned_buf(MAX_K * sizeof(float));

    static const int sizes[] = {1, 3, 15, 16, 17, 31, 64, 100, 4095, 4096, 11008 + 37};
    int cases = 0;
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        const int K = sizes[s];
        for (int d = ACT_GAUSSIAN; d <= ACT_ZERO; d++) {
            fill_act(x, K, (act_dist)d);
            const float gs = quantize_act_scalar(x, qs, K);
            const float gn = quantize_act_neon(x, qn, K);
            assert(gs == gn);
            assert(memcmp(qs, qn, (size_t)K) == 0);
            assert(gs == fmaxf(absmax_scalar(x, K), BITLINEAR_GAMMA_MIN));
            cases++;
        }
    }

    /* Exact rounding ties: gamma = 127 -> scale = 1, so x = n + 0.5 is a tie. */
    {
        const int K = 64;
        for (int k = 0; k < K; k++) x[k] = (float)((k % 9) - 4) + 0.5f; /* -3.5 .. 4.5 */
        x[K - 1] = 127.0f;
        quantize_act_scalar(x, qs, K);
        quantize_act_neon(x, qn, K);
        assert(memcmp(qs, qn, (size_t)K) == 0);
        assert(qs[0] == -4 && qs[1] == -2 && qs[3] == 0 && qs[7] == 4 && qs[8] == 4);
        assert(qs[K - 1] == 127);
        cases++;
    }

    /* Range: max-magnitude element maps to exactly +/-127; nothing hits -128. */
    {
        const int K = 4096;
        fill_act(x, K, ACT_GAUSSIAN);
        x[123] = -50.0f;
        quantize_act_neon(x, qn, K);
        assert(qn[123] == -127);
        for (int k = 0; k < K; k++) assert(qn[k] >= -127 && qn[k] <= 127);
        cases++;
    }

    /* All-zero input: zeros out, finite zero output (no 0/0). */
    {
        const int K = 4096;
        fill_act(x, K, ACT_ZERO);
        const float g = quantize_act_neon(x, qn, K);
        assert(g == BITLINEAR_GAMMA_MIN);
        for (int k = 0; k < K; k++) assert(qn[k] == 0);
        cases++;
    }

    /* Dequantize: bit-exact NEON vs scalar, including the int32 extremes. */
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        const int M = sizes[s];
        for (int m = 0; m < M; m++) yi[m] = (int32_t)(xorshift64() % (2 * 128 * 11008 + 1)) - 128 * 11008;
        yi[0] = 128 * 11008;
        const float sc = dequant_scale(3.7f, 0.0213f);
        dequantize_scalar(yi, ys, M, sc);
        dequantize_neon(yi, yn, M, sc);
        assert(memcmp(ys, yn, (size_t)M * sizeof(float)) == 0);
        cases++;
    }

    free(x); free(qs); free(qn); free(yi); free(ys); free(yn);
    printf("Quant kernels:  PASSED (%d cases; NEON == scalar bit-exact, half-to-even ties, zero input)\n",
           cases);
}

/* 2) End-to-end accuracy against float32 references across layer shapes. */
static void verify_layers(void) {
    static const struct { int M, K; const char *name; } shapes[] = {
        {128,   512,   "toy 128x512"},
        {1000,  777,   "odd 1000x777"},
        {2048,  2048,  "2048x2048"},
        {4096,  4096,  "attn proj 4096x4096"},
        {11008, 4096,  "ffn up 11008x4096"},
        {4096,  11008, "ffn down 4096x11008"},
    };
    enum { MAX_M = 11008, MAX_K = 11008 };
    const size_t max_wn = (size_t)4096 * 11008;

    float   *W      = aligned_buf(max_wn * sizeof(float));
    float   *W_deq  = aligned_buf(max_wn * sizeof(float));
    uint8_t *Wp     = aligned_buf((size_t)MAX_M * gemv_row_stride(MAX_K));
    float   *x      = aligned_buf(MAX_K * sizeof(float));
    float   *x_deq  = aligned_buf(MAX_K * sizeof(float));
    float   *y_neon = aligned_buf(MAX_M * sizeof(float));
    float   *y_sc   = aligned_buf(MAX_M * sizeof(float));
    float   *y_tern = aligned_buf(MAX_M * sizeof(float));
    float   *y_fp   = aligned_buf(MAX_M * sizeof(float));
    float   *y_xdq  = aligned_buf(MAX_M * sizeof(float));
    bitlinear_workspace ws = {aligned_buf(MAX_K), aligned_buf(MAX_M * sizeof(int32_t))};

    printf("\nLayer accuracy (y = BitLinear NEON forward)\n");
    printf("  A: vs float32 layer using the ternary weights W_q*beta  -> activation-quant error only\n");
    printf("  B: vs float32 layer using the original float weights W  -> + ternary weight error (informational)\n");
    printf("  %-21s %-15s %10s %8s %10s %9s | %8s %9s\n", "layer", "activations",
           "A cos", "A nrmse", "A bound", "A rmse", "B cos", "B nrmse");

    int cases = 0;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        const int M = shapes[s].M, K = shapes[s].K;
        /* Kaiming-ish float weights, as a trained layer would roughly look. */
        const float wstd = 1.0f / sqrtf((float)K);
        for (size_t i = 0; i < (size_t)M * K; i++) W[i] = wstd * rand_normal();
        const bitlinear_layer L = {Wp, quantize_weights_absmean(W, M, K, Wp, W_deq), M, K};

        for (int d = ACT_GAUSSIAN; d <= ACT_ZERO; d++) {
            fill_act(x, K, (act_dist)d);

            bitlinear_forward_neon(&L, x, y_neon, &ws);
            bitlinear_forward_scalar(&L, x, y_sc, &ws);
            /* The integer path is exact, so NEON and scalar forward must agree bit-for-bit. */
            assert(memcmp(y_neon, y_sc, (size_t)M * sizeof(float)) == 0);

            linear_f32_reference(W_deq, x, y_tern, M, K);
            linear_f32_reference(W, x, y_fp, M, K);

            /* Sanity: dequantized activations through the same float layer
             * must equal the quantized forward up to float32 rounding. */
            const float gamma = quantize_act_scalar(x, ws.x_q, K);
            for (int k = 0; k < K; k++) x_deq[k] = (float)ws.x_q[k] * gamma / BITLINEAR_QMAX;
            linear_f32_reference(W_deq, x_deq, y_xdq, M, K);
            const err_stats e_alg = compare(y_neon, y_xdq, M);
            assert(e_alg.nrmse < 1e-5 || (d == ACT_ZERO && e_alg.rmse < 1e-12));

            const err_stats a = compare(y_neon, y_tern, M);
            const err_stats b = compare(y_neon, y_fp, M);

            /* Tolerance derived from the quantizer: step D = gamma/127 adds
             * uniform noise of rms D/sqrt(12) per input, so relative output
             * error ~ D / (sqrt(12) * rms(x)). Require <= 1.5x that bound. */
            const double rx = rms(x, K);
            const double bound = rx > 0.0
                ? 1.5 * ((double)gamma / BITLINEAR_QMAX) / (sqrt(12.0) * rx) : 0.0;

            if (d == ACT_ZERO) {
                for (int m = 0; m < M; m++) assert(y_neon[m] == 0.0f);
                printf("  %-21s %-15s %10s %8s %10s %9.2e | %8s %9s\n", shapes[s].name,
                       act_dist_name((act_dist)d), "-", "-", "-", a.rmse, "-", "-");
            } else {
                printf("  %-21s %-15s %10.6f %8.4f %10.4f %9.2e | %8.4f %9.4f\n", shapes[s].name,
                       act_dist_name((act_dist)d), a.cosine, a.nrmse, bound, a.rmse,
                       b.cosine, b.nrmse);
                assert(a.nrmse <= bound);
                assert(a.cosine >= 1.0 - 0.5 * bound * bound * 4.0); /* cos ~ 1 - nrmse^2/2 */
                if (d != ACT_OUTLIER) {
                    assert(a.cosine >= 0.9995); /* well-conditioned inputs: hard floor */
                    assert(a.nrmse <= 0.03);
                }
            }
            cases++;
        }
    }

    free(W); free(W_deq); free(Wp); free(x); free(x_deq);
    free(y_neon); free(y_sc); free(y_tern); free(y_fp); free(y_xdq);
    free(ws.x_q); free(ws.y_int);
    printf("Layer accuracy: PASSED (%d cases; NEON == scalar bit-exact, all within quantization bound)\n",
           cases);
}

/* ------------------------------------------------------------------------- */
/* Benchmark                                                                 */
/* ------------------------------------------------------------------------- */

static inline uint64_t now_ns(void) { return platform_now_ns(); }

static int cmp_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

typedef enum { STAGE_QUANT, STAGE_GEMV, STAGE_DEQUANT, STAGE_FORWARD, STAGE_FORWARD_SCALAR } stage;

/* Median per-call latency (ns). Rotates through `nmat` layers; nmat > 1 forces
 * weights to stream from DRAM. */
static uint64_t bench_stage(stage st, bitlinear_layer *layers, int nmat, const float *x,
                            float *y, bitlinear_workspace *ws, int calls) {
    uint64_t *t = aligned_buf((size_t)calls * sizeof(uint64_t));
    const bitlinear_layer *L0 = &layers[0];
    float gamma = quantize_act_neon(x, ws->x_q, L0->K);
    gemv_bitnet_neon(ws->x_q, L0->W, ws->y_int, L0->M, L0->K);

    for (int i = -nmat; i < calls; i++) { /* negative i = warm-up */
        const bitlinear_layer *L = &layers[(i + nmat) % nmat];
        __asm__ volatile("" : : "r"(x), "r"(y), "r"(L) : "memory");
        const uint64_t t0 = now_ns();
        switch (st) {
        case STAGE_QUANT:   gamma = quantize_act_neon(x, ws->x_q, L->K); break;
        case STAGE_GEMV:    gemv_bitnet_neon(ws->x_q, L->W, ws->y_int, L->M, L->K); break;
        case STAGE_DEQUANT: dequantize_neon(ws->y_int, y, L->M, dequant_scale(gamma, L->beta)); break;
        case STAGE_FORWARD: bitlinear_forward_neon(L, x, y, ws); break;
        case STAGE_FORWARD_SCALAR: bitlinear_forward_scalar(L, x, y, ws); break;
        }
        __asm__ volatile("" : : "r"(y), "r"(ws->x_q), "r"(ws->y_int) : "memory");
        if (i >= 0) t[i] = now_ns() - t0;
    }
    qsort(t, (size_t)calls, sizeof(uint64_t), cmp_u64);
    const uint64_t med = t[calls / 2];
    free(t);
    return med;
}

static void run_benchmark(int quick) {
    enum { M = 4096, K = 4096, NMAT_COLD = 24 }; /* 24 x 4 MiB = 96 MiB >> L2 + SLC */
    const int nmat_cold = quick ? 2 : NMAT_COLD;
    const int calls = quick ? 20 : 1000;
    const int scalar_calls = quick ? 3 : 15;

    bitlinear_layer layers[NMAT_COLD];
    float *Wf = aligned_buf((size_t)M * K * sizeof(float));
    for (int i = 0; i < nmat_cold; i++) {
        uint8_t *Wp = aligned_buf((size_t)M * gemv_row_stride(K));
        for (size_t j = 0; j < (size_t)M * K; j++) Wf[j] = rand_normal() / 64.0f;
        layers[i] = (bitlinear_layer){Wp, quantize_weights_absmean(Wf, M, K, Wp, NULL), M, K};
    }
    free(Wf);

    float *x = aligned_buf(K * sizeof(float));
    float *y = aligned_buf(M * sizeof(float));
    bitlinear_workspace ws = {aligned_buf(K), aligned_buf(M * sizeof(int32_t))};
    fill_act(x, K, ACT_GAUSSIAN);

    printf("\nBenchmark: BitLinear forward, M=%d, K=%d, median per call (us)\n", M, K);
    printf("  %-40s %12s %12s\n", "stage", "hot (L2)", "cold (DRAM)");

    static const struct { stage st; const char *name; } rows[] = {
        {STAGE_QUANT,   "quantize activations (NEON)"},
        {STAGE_GEMV,    "ternary GEMV only (x_q already shared)"},
        {STAGE_DEQUANT, "dequantize outputs (NEON)"},
        {STAGE_FORWARD, "full forward (NEON, multi-thread)"},
        {STAGE_FORWARD_SCALAR, "full forward (scalar baseline)"},
    };
    uint64_t fwd[2] = {0, 0}, fwd_sc[2] = {0, 0};
    for (size_t r = 0; r < sizeof(rows) / sizeof(rows[0]); r++) {
        uint64_t t[2];
        for (int regime = 0; regime < 2; regime++) {
            const int n = rows[r].st == STAGE_FORWARD_SCALAR ? scalar_calls : calls;
            t[regime] = bench_stage(rows[r].st, layers, regime ? nmat_cold : 1, x, y, &ws, n);
        }
        if (rows[r].st == STAGE_FORWARD)        { fwd[0] = t[0];    fwd[1] = t[1]; }
        if (rows[r].st == STAGE_FORWARD_SCALAR) { fwd_sc[0] = t[0]; fwd_sc[1] = t[1]; }
        printf("  %-40s %12.2f %12.2f\n", rows[r].name, t[0] / 1e3, t[1] / 1e3);
    }
    printf("  %-40s %11.1fx %11.1fx\n", "speedup (scalar / NEON forward)",
           (double)fwd_sc[0] / fwd[0], (double)fwd_sc[1] / fwd[1]);
    printf("  %-40s %12.1f %12.1f\n", "full forward GOP/s",
           2.0 * M * K / fwd[0], 2.0 * M * K / fwd[1]);

    for (int i = 0; i < nmat_cold; i++) free((void *)layers[i].W);
    free(x); free(y); free(ws.x_q); free(ws.y_int);
}

int main(int argc, char **argv) {
    const int quick = argc > 1 && strcmp(argv[1], "--quick") == 0;
    verify_quant_kernels();
    verify_layers();
    run_benchmark(quick);
    return 0;
}
