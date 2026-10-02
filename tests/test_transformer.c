/*
 * test_transformer.c — Verification and benchmark of the BitNet b1.58
 * transformer forward pass on the real exported model.
 *
 *   build/test_transformer MODEL.bitnet TOKENIZER.json [REF.ref_logits] [--quick]
 *
 *   1. Kernel unit tests (RMSNorm, RoPE, softmax, exp, GLU, residual, f16 GEMV)
 *      against float64 scalar references.
 *   2. Single-token forward (BOS @ pos 0): logits shape / finiteness / top-5.
 *   3. Reference parity: feeds the tokens of REF (written by
 *      tools/reference_bitnet.py from the original HF checkpoint, cache-free
 *      float64 prefill) one at a time through the KV cache and compares the
 *      logits at every position (argmax, top-5, cosine, error).
 *   4. KV cache: re-running is bit-identical; rewinding to an earlier
 *      position and replaying reproduces the same logits.
 *   5. Zero allocation: counts malloc calls during forward passes.
 *   6. Benchmarks: per-token latency with stage breakdown, latency vs context
 *      length, memory.
 *
 * Build & run: `make test` (binary: build/test_transformer); `make asan` for sanitizers.
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

#include <malloc/malloc.h>
#include <unistd.h>

#include "model_loader.h"
#include "platform.h"
#include "threadpool.h"
#include "tokenizer.h"
#include "transformer.h"
#include "test_alloc_hook.h"

static inline uint64_t now_ns(void) { return platform_now_ns(); }

static size_t resident_bytes(void) { return platform_resident_bytes(); }

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void) { /* uniform [-1, 1) */
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / (1 << 23)) - 1.0);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    assert(p);
    return p;
}

/* ------------------------------------------------------------------------- */
/* 1. Kernel unit tests                                                      */
/* ------------------------------------------------------------------------- */

static void kernel_tests(void) {
    enum { N = 6912 + 13 };
    float *x = xmalloc(N * 4), *w = xmalloc(N * 4), *y = xmalloc(N * 4), *z = xmalloc(N * 4);
    double max_rel = 0;

    /* RMSNorm vs float64, sizes with/without tails, incl. tiny inputs where eps dominates. */
    static const int sizes[] = {1, 3, 4, 15, 16, 17, 128, 2560, 6912, N};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        const int n = sizes[si];
        for (int scale_i = 0; scale_i < 3; scale_i++) {
            const float sc = scale_i == 0 ? 1.0f : scale_i == 1 ? 1e4f : 1e-4f;
            for (int i = 0; i < n; i++) { x[i] = frand() * sc; w[i] = 1.0f + 0.5f * frand(); }
            rmsnorm_neon(y, x, w, n, 1e-5f);
            double ss = 0;
            for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
            const double r = 1.0 / sqrt(ss / n + 1e-5);
            for (int i = 0; i < n; i++) {
                const double ref = x[i] * r * w[i];
                const double e = fabs(y[i] - ref) / (fabs(ref) + 1e-30);
                if (fabs(ref) > 1e-6 && e > max_rel) max_rel = e;
            }
            memcpy(z, x, (size_t)n * 4);
            rmsnorm_neon(z, z, w, n, 1e-5f); /* in place */
            assert(memcmp(y, z, (size_t)n * 4) == 0);
        }
    }
    assert(max_rel < 2e-6);
    const double rms_err = max_rel;

    /* RoPE vs float64 rotate_half formula; must preserve each pair's norm. */
    const int hd = 128, nh = 20;
    float cs[64], sn[64];
    for (int i = 0; i < 64; i++) {
        const double a = 4095.0 * pow(500000.0, -2.0 * i / hd);
        cs[i] = (float)cos(a);
        sn[i] = (float)sin(a);
    }
    for (int i = 0; i < nh * hd; i++) x[i] = frand();
    memcpy(y, x, (size_t)nh * hd * 4);
    apply_rope_neon(y, nh, hd, cs, sn);
    double rope_err = 0;
    for (int h = 0; h < nh; h++)
        for (int i = 0; i < 64; i++) {
            const double x1 = x[h * hd + i], x2 = x[h * hd + 64 + i];
            const double r1 = x1 * cs[i] - x2 * sn[i], r2 = x2 * cs[i] + x1 * sn[i];
            rope_err = fmax(rope_err, fmax(fabs(y[h * hd + i] - r1), fabs(y[h * hd + 64 + i] - r2)));
            const double n0 = x1 * x1 + x2 * x2;
            const double n1 = (double)y[h * hd + i] * y[h * hd + i] + (double)y[h * hd + 64 + i] * y[h * hd + 64 + i];
            assert(fabs(n0 - n1) <= 1e-5 * (n0 + 1e-12));
        }
    assert(rope_err < 1e-6);

    /* exp: relative error over [-87, 88]. */
    double exp_err = 0;
    for (int i = 0; i < N; i++) x[i] = -87.0f + 175.0f * (float)i / N;
    exp_neon(y, x, N);
    for (int i = 0; i < N; i++) exp_err = fmax(exp_err, fabs(y[i] - exp((double)x[i])) / exp((double)x[i]));
    assert(exp_err < 5e-7);

    /* softmax vs float64, incl. huge spread and a single element. */
    double sm_err = 0;
    for (int trial = 0; trial < 4; trial++) {
        const int n = trial == 0 ? 1 : trial == 1 ? 7 : trial == 2 ? 4096 : 13;
        for (int i = 0; i < n; i++) x[i] = frand() * (trial == 3 ? 500.0f : 20.0f);
        memcpy(y, x, (size_t)n * 4);
        softmax_neon(y, n);
        double mx = -INFINITY, sum = 0, tot = 0;
        for (int i = 0; i < n; i++) mx = fmax(mx, x[i]);
        for (int i = 0; i < n; i++) sum += exp((double)x[i] - mx);
        for (int i = 0; i < n; i++) {
            sm_err = fmax(sm_err, fabs(y[i] - exp((double)x[i] - mx) / sum));
            tot += y[i];
            assert(isfinite(y[i]) && y[i] >= 0.0f);
        }
        assert(fabs(tot - 1.0) < 1e-5);
    }
    assert(sm_err < 1e-6);

    /* GLU: relu2 exact; silu vs float64. */
    double glu_err = 0;
    for (int i = 0; i < N; i++) { x[i] = 8.0f * frand(); w[i] = 4.0f * frand(); }
    memcpy(y, x, N * 4);
    glu_neon(y, w, N, 1);
    for (int i = 0; i < N; i++) {
        const float r = x[i] > 0 ? x[i] : 0;
        assert(y[i] == r * r * w[i]);
    }
    memcpy(y, x, N * 4);
    glu_neon(y, w, N, 0);
    for (int i = 0; i < N; i++) {
        const double ref = x[i] / (1.0 + exp(-(double)x[i])) * w[i];
        glu_err = fmax(glu_err, fabs(y[i] - ref) / (fabs(ref) + 1e-3));
    }
    assert(glu_err < 2e-6);

    /* Residual add: exact. */
    memcpy(y, x, N * 4);
    residual_add_neon(y, w, N);
    for (int i = 0; i < N; i++) assert(y[i] == x[i] + w[i]);

    /* f16 GEMV vs float64 (M not a multiple of the row block, K with a tail). */
    const int M = 1000, K = 2570;
    uint16_t *W16 = xmalloc((size_t)M * K * 2);
    for (size_t i = 0; i < (size_t)M * K; i++) {
        W16[i] = float_to_half(0.1f * frand());
    }
    for (int k = 0; k < K; k++) x[k] = frand();
    gemv_f16_neon(W16, x, y, M, K, NULL);
    double f16_err = 0;
    for (int m = 0; m < M; m++) {
        double ref = 0, mag = 0;
        for (int k = 0; k < K; k++) {
            const double h = half_to_float(W16[(size_t)m * K + k]);
            ref += h * x[k];
            mag += fabs(h * x[k]);
        }
        f16_err = fmax(f16_err, fabs(y[m] - ref) / mag);
    }
    assert(f16_err < 1e-6);

    free(W16); free(x); free(w); free(y); free(z);
    printf("Kernels:        PASSED (max rel err: rmsnorm %.1e, rope %.1e, exp %.1e, softmax %.1e, "
           "silu %.1e, f16 gemv %.1e; relu2 & residual exact)\n",
           rms_err, rope_err, exp_err, sm_err, glu_err, f16_err);
}

/* ------------------------------------------------------------------------- */
/* Thread pool and attention unit tests                                      */
/* ------------------------------------------------------------------------- */

typedef struct {
    _Atomic int *hits;
} pool_probe;

static void pool_probe_fn(void *ctx, size_t i) {
    pool_probe *p = ctx;
    __atomic_fetch_add((int *)&p->hits[i], 1, __ATOMIC_RELAXED);
}

static void pool_tests(void) {
    char err[128];
    threadpool *pool = threadpool_create(0, err, sizeof(err));
    assert(pool);
    enum { MAXN = 5000 };
    _Atomic int *hits = calloc(MAXN, sizeof(*hits));
    pool_probe pr = {hits};
    static const size_t ns[] = {0, 1, 2, 3, 7, 8, 9, 64, 1000, MAXN};
    long runs = 0;
    /* Varying n between consecutive jobs (small after large and vice versa) is
     * what exposes stale-worker races; run enough jobs to hit them. */
    for (int rep = 0; rep < 60000; rep++) {
        const size_t n = ns[rep % (sizeof(ns) / sizeof(ns[0]))];
        memset((void *)hits, 0, MAXN * sizeof(*hits));
        threadpool_run(pool, n, &pr, pool_probe_fn);
        for (size_t i = 0; i < n; i++) assert(hits[i] == 1);
        for (size_t i = n; i < (n < MAXN ? n + 1 : n); i++) assert(hits[i] == 0);
        runs++;
        if (rep % 15000 == 14999) { /* let workers fall asleep, then wake them */
            struct timespec ts = {0, 20 * 1000 * 1000};
            nanosleep(&ts, NULL);
        }
    }
    printf("Thread pool:    PASSED (%d threads; %ld jobs incl. after sleep, every item exactly once)\n",
           threadpool_size(pool), runs);
    free((void *)hits);
    threadpool_destroy(pool);
}

static void attention_tests(void) {
    const int nh = 20, nkv = 5, hd = 128, kv_dim = nkv * hd, T = 4096, G = nh / nkv;
    float *q = xmalloc((size_t)nh * hd * 4), *out = xmalloc((size_t)nh * hd * 4);
    float *K = xmalloc((size_t)T * kv_dim * 4), *V = xmalloc((size_t)T * kv_dim * 4);
    float *part = xmalloc(attention_scratch_floats(nh, hd, T) * 4);
    double *sc = xmalloc((size_t)T * sizeof(double));
    for (int i = 0; i < nh * hd; i++) q[i] = 2.0f * frand();
    for (size_t i = 0; i < (size_t)T * kv_dim; i++) { K[i] = frand(); V[i] = frand(); }
    char err[128];
    threadpool *pool = threadpool_create(0, err, sizeof(err));
    assert(pool);

    static const int ps[] = {0, 1, 2, 62, 63, 64, ATTN_CHUNK - 1, ATTN_CHUNK, ATTN_CHUNK + 1, 700, 4095};
    double worst = 0;
    for (size_t pi = 0; pi < sizeof(ps) / sizeof(ps[0]); pi++) {
        const int pos = ps[pi];
        for (int use_pool = 0; use_pool < 2; use_pool++) {
            attention_neon(out, q, K, V, pos, nh, nkv, hd, kv_dim, part, use_pool ? pool : NULL);
            for (int h = 0; h < nh; h++) {
                const int kvh = h / G; /* HF repeat_kv: head h uses kv head h / group */
                double mx = -INFINITY, sum = 0;
                for (int t = 0; t <= pos; t++) {
                    double d = 0;
                    for (int i = 0; i < hd; i++)
                        d += (double)q[h * hd + i] * K[(size_t)t * kv_dim + kvh * hd + i];
                    sc[t] = d / sqrt((double)hd);
                    mx = fmax(mx, sc[t]);
                }
                for (int t = 0; t <= pos; t++) sum += (sc[t] = exp(sc[t] - mx));
                for (int i = 0; i < hd; i++) {
                    double ref = 0;
                    for (int t = 0; t <= pos; t++) ref += sc[t] * V[(size_t)t * kv_dim + kvh * hd + i];
                    ref /= sum;
                    worst = fmax(worst, fabs(out[h * hd + i] - ref));
                }
            }
        }
    }
    assert(worst < 2e-5);
    printf("Attention:      PASSED (GQA %d/%d, head_dim %d, %zu positions up to %d, chunk %d, pool and "
           "serial; max abs err vs float64 %.1e)\n", nh, nkv, hd, sizeof(ps) / sizeof(ps[0]), T - 1,
           ATTN_CHUNK, worst);
    threadpool_destroy(pool);
    free(q); free(out); free(K); free(V); free(part); free(sc);
}

/* ------------------------------------------------------------------------- */
/* Helpers on logits                                                         */
/* ------------------------------------------------------------------------- */

static void topk(const float *v, int n, int k, int *idx) {
    for (int j = 0; j < k; j++) idx[j] = -1;
    for (int i = 0; i < n; i++) {
        int j = k;
        while (j > 0 && (idx[j - 1] < 0 || v[i] > v[idx[j - 1]])) j--;
        if (j < k) {
            memmove(idx + j + 1, idx + j, (size_t)(k - j - 1) * sizeof(int));
            idx[j] = i;
        }
    }
}

static const char *tok_str(const Tokenizer *tk, int id, char *buf, size_t cap) {
    size_t l;
    const char *s = bpe_decode_bytes(tk, id, &l);
    size_t o = 0;
    for (size_t i = 0; s && i < l && o + 5 < cap; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '\n') { buf[o++] = '\\'; buf[o++] = 'n'; }
        else if (c >= 0x20) buf[o++] = (char)c;
        else o += (size_t)snprintf(buf + o, cap - o, "\\x%02x", c);
    }
    buf[o] = '\0';
    return buf;
}

/* ------------------------------------------------------------------------- */
/* 3. Reference parity                                                       */
/* ------------------------------------------------------------------------- */

typedef struct {
    int    T, V;
    int   *tokens;
    float *logits; /* [T][V] */
} ref_data;

static int load_ref(const char *path, ref_data *r) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    uint32_t ver, T, V;
    assert(fread(magic, 1, 4, f) == 4 && memcmp(magic, "BLOG", 4) == 0);
    assert(fread(&ver, 4, 1, f) == 1 && ver == 1);
    assert(fread(&T, 4, 1, f) == 1 && fread(&V, 4, 1, f) == 1);
    r->T = (int)T;
    r->V = (int)V;
    r->tokens = xmalloc(T * sizeof(int));
    r->logits = xmalloc((size_t)T * V * sizeof(float));
    assert(fread(r->tokens, 4, T, f) == T);
    assert(fread(r->logits, 4, (size_t)T * V, f) == (size_t)T * V);
    fclose(f);
    return 0;
}

static void verify_reference(const BitNetModel *m, const Tokenizer *tk, RunState *s, const ref_data *r) {
    const int V = m->config.vocab_size;
    assert(r->V == V);
    float *first = xmalloc((size_t)r->T * V * sizeof(float));
    double worst_cos = 1, worst_nrmse = 0;
    int top1 = 0, top5_overlap = 0;

    printf("\nReference parity (C incremental KV-cache path vs numpy float64 prefill of the HF checkpoint):\n");
    if (r->T <= 24)
        printf("  pos  token          C top-1           ref top-1        cosine     nRMSE   top5\n");
    for (int t = 0; t < r->T; t++) {
        transformer_forward(r->tokens[t], t, m, s);
        memcpy(first + (size_t)t * V, s->logits, (size_t)V * 4);
        const float *ref = r->logits + (size_t)t * V;
        double dot = 0, na = 0, nb = 0, se = 0;
        for (int i = 0; i < V; i++) {
            assert(isfinite(s->logits[i]));
            dot += (double)s->logits[i] * ref[i];
            na += (double)s->logits[i] * s->logits[i];
            nb += (double)ref[i] * ref[i];
            se += ((double)s->logits[i] - ref[i]) * ((double)s->logits[i] - ref[i]);
        }
        const double cosv = dot / sqrt(na * nb), nrmse = sqrt(se / nb);
        int a[5], b[5], ov = 0;
        topk(s->logits, V, 5, a);
        topk(ref, V, 5, b);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++) ov += a[i] == b[j];
        top1 += a[0] == b[0];
        top5_overlap += ov;
        worst_cos = fmin(worst_cos, cosv);
        worst_nrmse = fmax(worst_nrmse, nrmse);
        char s1[64], s2[64], s3[64];
        if (r->T <= 24) printf("  %3d  %-14s %-17s %-17s %.6f  %.2e   %d/5\n", t, tok_str(tk, r->tokens[t], s1, 64),
               tok_str(tk, a[0], s2, 64), tok_str(tk, b[0], s3, 64), cosv, nrmse, ov);
    }
    printf("  -> top-1 agreement %d/%d, mean top-5 overlap %.2f/5, worst cosine %.6f, worst nRMSE %.2e\n",
           top1, r->T, (double)top5_overlap / r->T, worst_cos, worst_nrmse);
    /* Tolerances come from the measured noise floor between two *correct*
     * implementations: the numpy reference in float32 vs float64 differs by up
     * to nRMSE 5.4e-2 / cosine 0.99929 on this 2B model (1.7e-2 / 0.99986 on a
     * 2-layer mock), because a 1-ulp difference can flip an int8 activation
     * rounding at a .5 boundary and the flip propagates through later layers.
     * Pos 0 has no such freedom and matches to ~3e-7. Allow 2x the floor;
     * structural bugs (GQA mapping, RoPE layout/sign, causal range, scale) give
     * nRMSE ~1 (see the mutation tests in the Step 6 report). */
    assert(top1 == r->T);
    assert((double)top5_overlap / r->T >= 4.5);
    assert(worst_cos > 0.995 && worst_nrmse < 0.12);

    /* KV cache determinism: a second run must be bit-identical. */
    for (int t = 0; t < r->T; t++) {
        transformer_forward(r->tokens[t], t, m, s);
        assert(memcmp(first + (size_t)t * V, s->logits, (size_t)V * 4) == 0);
    }
    /* Rewind: replay from position T/2 over the same cache; same logits. */
    for (int t = r->T / 2; t < r->T; t++) transformer_forward(r->tokens[t], t, m, s);
    assert(memcmp(first + (size_t)(r->T - 1) * V, s->logits, (size_t)V * 4) == 0);
    printf("KV cache:       PASSED (re-run bit-identical at all %d positions; rewind to pos %d and "
           "replay bit-identical)\n", r->T, r->T / 2);
    free(first);
}

/* ------------------------------------------------------------------------- */
/* 5/6. Allocation check and benchmarks                                      */
/* ------------------------------------------------------------------------- */

static void alloc_check(const BitNetModel *m, RunState *s, int token) {
#ifndef ALLOC_HOOK_AVAILABLE
    /* Sanitizer builds own the allocator; non-macOS has no malloc zones. */
    (void)m; (void)s; (void)token;
    printf("Allocations:    SKIPPED (allocation hook unavailable on this build/platform)\n");
#else
    if (hook_zones(1) != 0) {
        printf("Allocations:    SKIPPED (could not hook malloc zones)\n");
        return;
    }
    g_allocs = 0;
    void *volatile probe = malloc(24); /* proves the hooks observe allocations */
    free(probe);
    const long probe_seen = g_allocs;
    g_allocs = 0;
    for (int p = 0; p < 96; p++) transformer_forward(token, p, m, s); /* both attention paths */
    const long n = g_allocs;
    hook_zones(0);
    if (probe_seen == 0) {
        printf("Allocations:    SKIPPED (allocator hooks not reached, e.g. under ASan)\n");
        return;
    }
    printf("Allocations:    %s (%ld heap allocations during 96 forward passes, all threads; "
           "hook verified with a probe malloc)\n", n == 0 ? "PASSED" : "FAILED", n);
    assert(n == 0);
#endif
}

static void benchmark(const BitNetModel *m, const Tokenizer *tk, RunState *s, int quick) {
    const int V = m->config.vocab_size;
    /* Greedy continuation of a prompt: realistic short-context decode. */
    int32_t prompt[64];
    const int np = bpe_encode(tk, "The capital of France is", prompt + 1, 63) + 1;
    prompt[0] = 128000;
    const int gen = quick ? 8 : 64;
    memset(&s->prof, 0, sizeof(s->prof));
    s->profile = 1;
    int next = prompt[0];
    char text[1024] = "", piece[64];
    for (int p = 0; p < np + gen; p++) {
        transformer_forward(p < np ? prompt[p] : next, p, m, s);
        int best = 0;
        for (int i = 1; i < V; i++) best = s->logits[i] > s->logits[best] ? i : best;
        next = best; /* only consumed once the prompt is exhausted */
        if (p >= np - 1 && strlen(text) < 400)
            strncat(text, tok_str(tk, next, piece, 64), sizeof(text) - strlen(text) - 1);
    }
    s->profile = 0;
    const transformer_profile *P = &s->prof;
    const double per = (double)P->total / P->tokens / 1e6;
    printf("\nGreedy continuation: \"The capital of France is%s\"\n", text);
    printf("\nBenchmark: per-token forward, positions 0..%d (%d-thread pool)\n", np + gen - 1,
           threadpool_size(s->pool));
    printf("  time per token           %7.2f ms   (%.1f tokens/s)\n", per, 1e3 / per);
    printf("    embedding              %7.3f ms\n", (double)P->embed / P->tokens / 1e6);
    printf("    attn projections (qkvo)%7.3f ms\n", (double)P->attn_proj / P->tokens / 1e6);
    printf("    attention (scores,softmax,values) %6.3f ms\n", (double)P->attention / P->tokens / 1e6);
    printf("    ffn (gate,up,down)     %7.3f ms\n", (double)P->ffn / P->tokens / 1e6);
    printf("    logits (f16 tied, 128256x2560) %5.3f ms\n", (double)P->logits / P->tokens / 1e6);

    printf("\n  latency vs context length (single token at position p, median of %d):\n", quick ? 3 : 9);
    static const int ps[] = {16, 256, 1024, 2048, 4095};
    for (size_t i = 0; i < sizeof(ps) / sizeof(ps[0]); i++) {
        if (ps[i] >= s->max_seq_len) continue;
        uint64_t t[9];
        const int reps = quick ? 3 : 9;
        for (int r = 0; r < reps; r++) {
            const uint64_t t0 = now_ns();
            transformer_forward(128000, ps[i], m, s);
            t[r] = now_ns() - t0;
        }
        for (int a = 0; a < reps; a++)
            for (int b = a + 1; b < reps; b++)
                if (t[b] < t[a]) { const uint64_t x = t[a]; t[a] = t[b]; t[b] = x; }
        printf("    pos %5d   %7.2f ms\n", ps[i], t[reps / 2] / 1e6);
    }
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *tok_path = NULL, *ref_path = NULL;
    int quick = 0, n_threads = 0, unit = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) quick = 1;
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-unit-tests") == 0) unit = 0;
        else if (!model_path) model_path = argv[i];
        else if (!tok_path) tok_path = argv[i];
        else ref_path = argv[i];
    }
    setvbuf(stdout, NULL, _IOLBF, 0); /* progress survives a crash in logs */
    if (!model_path || !tok_path) {
        fprintf(stderr, "usage: %s MODEL.bitnet TOKENIZER.json [REF.ref_logits] [--quick] "
                "[--threads N] [--no-unit-tests]\n", argv[0]);
        return 2;
    }

    if (unit) {
        kernel_tests();
        pool_tests();
        attention_tests();
    }

    char err[256];
    const size_t rss0 = resident_bytes();
    bitnet_model model;
    if (bitnet_model_load(model_path, &model, err, sizeof(err)) != 0) {
        fprintf(stderr, "model: %s\n", err);
        return 1;
    }
    Tokenizer *tk = tokenizer_load(tok_path, err, sizeof(err));
    if (!tk) {
        fprintf(stderr, "tokenizer: %s\n", err);
        return 1;
    }
    RunState s;
    if (runstate_init(&s, &model, 0, n_threads, err, sizeof(err)) != 0) {
        fprintf(stderr, "runstate: %s\n", err);
        return 1;
    }
    const bitnet_config *c = &model.config;
    printf("Model: %d layers, dim %d, hidden %d, heads %d/%d kv (head_dim %d), vocab %d, %s FFN\n",
           c->n_layers, c->dim, c->hidden_dim, c->n_heads, c->n_kv_heads, c->head_dim, c->vocab_size,
           c->ffn_act == BITNET_ACT_RELU2 ? "relu2" : "SwiGLU");
    printf("Threads: %d (persistent pool incl. caller)\n", threadpool_size(s.pool));
    printf("RunState: %.1f MiB reserved (KV cache %d positions x %d layers x 2 x %d floats = %.1f MiB)\n",
           s.arena_bytes / 1048576.0, s.max_seq_len, c->n_layers, c->n_kv_heads * c->head_dim,
           2.0 * c->n_layers * s.max_seq_len * c->n_kv_heads * c->head_dim * 4 / 1048576.0);

    /* The tokenizer belongs to the real model; a mock model (Step 4 exporter)
     * has its own small vocabulary, so text-level checks are skipped for it. */
    const int real = tokenizer_vocab_size(tk) == c->vocab_size;
    const int bos = real ? tokenizer_token_id(tk, "<|begin_of_text|>") : 0;

    /* 2. Single-token forward. */
    transformer_forward(bos, 0, &model, &s);
    assert(model.output->rows == c->vocab_size);
    if (real) assert(c->vocab_size == 128256);
    int top[5];
    for (int i = 0; i < c->vocab_size; i++) assert(isfinite(s.logits[i]));
    topk(s.logits, c->vocab_size, 5, top);
    printf("Single token:   PASSED (BOS @ pos 0 -> %d finite logits; top-5:", c->vocab_size);
    char b[64];
    for (int i = 0; i < 5; i++) {
        if (real) printf(" \"%s\"", tok_str(tk, top[i], b, 64));
        else printf(" %d", top[i]);
    }
    printf(")\n");

    ref_data r;
    if (ref_path && load_ref(ref_path, &r) == 0) {
        verify_reference(&model, tk, &s, &r);
        free(r.tokens);
        free(r.logits);
    } else {
        printf("Reference parity: SKIPPED (no reference logits file)\n");
    }

    alloc_check(&model, &s, bos);
    if (real) benchmark(&model, tk, &s, quick);
    printf("\n  resident memory: %.0f MiB (model pages touched + KV cache in use)\n",
           ((double)resident_bytes() - (double)rss0) / 1048576.0);

    runstate_free(&s);
    tokenizer_free(tk);
    bitnet_model_free(&model);
    return 0;
}
