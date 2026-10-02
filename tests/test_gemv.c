/*
 * test_gemv.c — Ternary (BitNet b1.58) GEMV: out[M] = W[M x K] * act[K].
 *
 * Kernel implementation lives in gemv.h (shared with the engine).
 *
 * Layout:
 *   - W is row-major; each row holds K packed ternary weights (see ternary_dot.h)
 *     and starts on a byte boundary: row stride = ceil(K / 4) bytes.
 *   - act is int8_t[K], out is int32_t[M].
 *
 * Parallelism & tiling:
 *   - Rows are split into GEMV_ROW_BLOCK-row work items and distributed with
 *     Grand Central Dispatch (dispatch_apply_f). GCD keeps a persistent worker
 *     pool (no per-call thread creation) and balances items dynamically, which
 *     matters on Apple Silicon's mixed P/E-core clusters.
 *   - Within a work item, K is walked in GEMV_K_TILE-wide tiles. Each
 *     activation tile is reused by every row of the block while it is hot in
 *     L1; partial sums live in a per-block stack accumulator.
 *   - The inner micro-kernel processes 4 rows at once: one vld4q_s8 of
 *     activations feeds 4 rows of decoded weights (built on the
 *     primitives in ternary_dot.h).
 *
 * Build & run: `make test` (binary: build/test_gemv); `make asan` for sanitizers.
 *
 * Range: |out[m]| <= 128 * K, so K must stay below 2^24 to fit int32_t.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#undef NDEBUG /* verification asserts must always run */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "platform.h"
#include "gemv.h"

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

static void *aligned_buf(size_t bytes) {
    void *p = NULL;
    if (bytes == 0) bytes = 64;
    if (posix_memalign(&p, 64, bytes) != 0) {
        fprintf(stderr, "posix_memalign(%zu) failed\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

static inline size_t matrix_bytes(int M, int K) { return (size_t)M * gemv_row_stride(K); }

static void fill_act_random(int8_t *act, int K) {
    for (int k = 0; k < K; k++) act[k] = (int8_t)(uint8_t)xorshift64();
}

/* Random valid codes {00, 01, 10}; padding bits past K in each row stay zero. */
static void fill_weights_random(uint8_t *W, int M, int K) {
    const size_t stride = gemv_row_stride(K);
    memset(W, 0, matrix_bytes(M, K));
    for (int m = 0; m < M; m++) {
        uint8_t *row = W + (size_t)m * stride;
        uint64_t bits = 0;
        int avail = 0;
        for (int k = 0; k < K; k++) {
            if (avail == 0) { bits = xorshift64(); avail = 32; }
            uint8_t code = (uint8_t)(bits & 3);
            bits >>= 2;
            avail--;
            if (code == 3) code = (uint8_t)(xorshift64() % 3);
            row[k >> 2] |= (uint8_t)(code << ((k & 3) * 2));
        }
    }
}

/* Every weight set to `code` (0x0 = 0, 0x1 = +1, 0x2 = -1). */
static void fill_weights_const(uint8_t *W, int M, int K, uint8_t code) {
    const size_t stride = gemv_row_stride(K);
    const uint8_t full = (uint8_t)(code | code << 2 | code << 4 | code << 6);
    memset(W, 0, matrix_bytes(M, K));
    for (int m = 0; m < M; m++) {
        uint8_t *row = W + (size_t)m * stride;
        memset(row, full, (size_t)K / 4);
        for (int k = K & ~3; k < K; k++) row[k >> 2] |= (uint8_t)(code << ((k & 3) * 2));
    }
}

/* Independent reference: decodes via bit arithmetic, accumulates in int64. */
static void gemv_unpacked_reference(const int8_t *act, const uint8_t *W, int32_t *out,
                                    int M, int K) {
    const size_t stride = gemv_row_stride(K);
    for (int m = 0; m < M; m++) {
        int64_t sum = 0;
        for (int k = 0; k < K; k++) {
            const unsigned code = (W[(size_t)m * stride + (k >> 2)] >> (2 * (k & 3))) & 3u;
            const int w = (int)(code & 1u) - (int)((code >> 1) & 1u);
            sum += (code == 3u) ? 0 : (int64_t)act[k] * w;
        }
        out[m] = (int32_t)sum;
    }
}

static int first_mismatch(const int32_t *a, const int32_t *b, int M) {
    for (int m = 0; m < M; m++)
        if (a[m] != b[m]) return m;
    return -1;
}

typedef struct {
    int8_t  *act;
    uint8_t *W;
    int32_t *ref, *sc, *st, *mt;
} test_bufs;

static void check_all(const char *label, const test_bufs *b, int M, int K, int with_unpacked) {
    gemv_scalar(b->act, b->W, b->sc, M, K);
    gemv_bitnet_neon_st(b->act, b->W, b->st, M, K);
    gemv_bitnet_neon(b->act, b->W, b->mt, M, K);

    int bad = -1;
    if (with_unpacked) {
        gemv_unpacked_reference(b->act, b->W, b->ref, M, K);
        bad = first_mismatch(b->ref, b->sc, M);
        if (bad >= 0)
            fprintf(stderr, "MISMATCH [%s] M=%d K=%d row=%d ref=%d scalar=%d\n",
                    label, M, K, bad, b->ref[bad], b->sc[bad]);
        assert(bad < 0);
    }
    bad = first_mismatch(b->sc, b->st, M);
    if (bad >= 0)
        fprintf(stderr, "MISMATCH [%s] M=%d K=%d row=%d scalar=%d neon_st=%d\n",
                label, M, K, bad, b->sc[bad], b->st[bad]);
    assert(bad < 0);
    bad = first_mismatch(b->sc, b->mt, M);
    if (bad >= 0)
        fprintf(stderr, "MISMATCH [%s] M=%d K=%d row=%d scalar=%d neon_mt=%d\n",
                label, M, K, bad, b->sc[bad], b->mt[bad]);
    assert(bad < 0);
}

static void run_verification(void) {
    /* Buffer K must cover every shape below, including the K-tile-derived ones. */
    enum { MAX_M = 4096,
           MAX_K = (3 * GEMV_K_TILE + 17 > 11008) ? 3 * GEMV_K_TILE + 17 : 11008 };
    test_bufs b = {
        aligned_buf(MAX_K),
        aligned_buf(matrix_bytes(MAX_M, MAX_K)),
        aligned_buf(MAX_M * sizeof(int32_t)),
        aligned_buf(MAX_M * sizeof(int32_t)),
        aligned_buf(MAX_M * sizeof(int32_t)),
        aligned_buf(MAX_M * sizeof(int32_t)),
    };

    /* Random shapes: small/odd (tails in M and K), K-tile boundaries, layer sizes. */
    static const struct { int M, K, trials; } shapes[] = {
        {1, 1, 20},      {1, 63, 20},      {3, 64, 20},       {5, 65, 20},
        {7, 100, 20},    {31, 127, 20},    {33, 129, 10},     {128, 512, 20},
        {129, 1023, 10}, {256, 2048, 10},  {64, GEMV_K_TILE - 1, 5},
        {64, GEMV_K_TILE + 1, 5},          {100, 3 * GEMV_K_TILE + 17, 5},
        {1024, 1024, 5}, {2048, 4096, 3},  {4096, 4096, 3},   {4096, 11008, 1},
    };
    int cases = 0;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        const int M = shapes[s].M, K = shapes[s].K;
        for (int t = 0; t < shapes[s].trials; t++) {
            fill_act_random(b.act, K);
            fill_weights_random(b.W, M, K);
            check_all("random", &b, M, K, (size_t)M * (size_t)K <= (1u << 22));
            cases++;
        }
    }

    /* Extremes at full layer size: |out| = 128 * K hits the int8 x ternary max. */
    const int M = 4096, K = 4096;
    memset(b.act, -128, (size_t)K);
    fill_weights_const(b.W, M, K, 0x2);
    check_all("extreme -128 * -1", &b, M, K, 0);
    for (int m = 0; m < M; m++) assert(b.mt[m] == 128 * K);

    fill_weights_const(b.W, M, K, 0x1);
    check_all("extreme -128 * +1", &b, M, K, 0);
    for (int m = 0; m < M; m++) assert(b.mt[m] == -128 * K);

    memset(b.act, 127, (size_t)K);
    check_all("extreme 127 * +1", &b, M, K, 0);
    for (int m = 0; m < M; m++) assert(b.mt[m] == 127 * K);

    fill_weights_const(b.W, M, K, 0x0);
    check_all("all-zero weights", &b, M, K, 0);
    for (int m = 0; m < M; m++) assert(b.mt[m] == 0);

    /* Reserved code 0b11 must decode as 0 everywhere. */
    fill_act_random(b.act, K);
    for (size_t i = 0; i < matrix_bytes(M, K); i++) b.W[i] = (uint8_t)xorshift64();
    check_all("random bytes incl. reserved", &b, 512, K, 1);
    cases += 5;

    free(b.act);
    free(b.W);
    free(b.ref);
    free(b.sc);
    free(b.st);
    free(b.mt);
    printf("Verification: PASSED (%d cases, M x K from 1x1 to 4096x11008; scalar == SIMD-1T == SIMD-MT)\n",
           cases);
}

/* ------------------------------------------------------------------------- */
/* Benchmark                                                                 */
/* ------------------------------------------------------------------------- */

typedef void (*gemv_fn)(const int8_t *, const uint8_t *, int32_t *, int, int);

static inline uint64_t now_ns(void) { return platform_now_ns(); }

static int cmp_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Median per-call latency in ns. Cycles through `nmat` weight matrices so that
 * nmat > 1 defeats cache residency (cold-weight / DRAM-streaming regime). */
static uint64_t bench(gemv_fn fn, const int8_t *act, uint8_t *const *mats, int nmat,
                      int32_t *out, int M, int K, int calls) {
    uint64_t *t = aligned_buf((size_t)calls * sizeof(uint64_t));
    for (int i = 0; i < nmat; i++) fn(act, mats[i], out, M, K); /* warm-up */
    for (int i = 0; i < calls; i++) {
        const uint8_t *W = mats[i % nmat];
        __asm__ volatile("" : : "r"(act), "r"(W), "r"(out) : "memory");
        const uint64_t t0 = now_ns();
        fn(act, W, out, M, K);
        __asm__ volatile("" : : "r"(out) : "memory");
        t[i] = now_ns() - t0;
    }
    qsort(t, (size_t)calls, sizeof(uint64_t), cmp_u64);
    const uint64_t med = t[calls / 2];
    free(t);
    return med;
}

static void print_row(const char *name, uint64_t ns, uint64_t base_ns, int M, int K) {
    const double ms   = (double)ns / 1e6;
    const double gops = 2.0 * (double)M * (double)K / (double)ns;    /* mul + add */
    const double gbs  = (double)matrix_bytes(M, K) / (double)ns;     /* weight stream */
    printf("  %-22s %10.3f %10.2f %12.2f %9.2fx\n", name, ms, gops, gbs,
           (double)base_ns / (double)ns);
}

static void run_benchmark(int quick) {
    enum { M = 4096, K = 4096, NMAT_COLD = 24 }; /* 24 x 4 MiB = 96 MiB >> L2 + SLC */
    const int nmat_cold = quick ? 2 : NMAT_COLD;

    int8_t  *act = aligned_buf(K);
    int32_t *out = aligned_buf(M * sizeof(int32_t));
    uint8_t *mats[NMAT_COLD];
    for (int i = 0; i < nmat_cold; i++) {
        mats[i] = aligned_buf(matrix_bytes(M, K));
        fill_weights_random(mats[i], M, K);
    }
    fill_act_random(act, K);

    const int scalar_calls = quick ? 3 : 15;
    const int neon_calls   = quick ? 20 : 500;

    printf("\nBenchmark: M=%d, K=%d (%.1f MiB packed weights), median per call\n",
           M, K, (double)matrix_bytes(M, K) / (1 << 20));
    printf("  SIMD path: %s | row block %d | K tile %d | GCD over %d logical CPUs\n",
           TERNARY_DOT_PATH, GEMV_ROW_BLOCK, GEMV_K_TILE,
           (int)sysconf(_SC_NPROCESSORS_ONLN));

    for (int regime = 0; regime < 2; regime++) {
        const int nmat = regime == 0 ? 1 : nmat_cold;
        const uint64_t t_sc = bench(gemv_scalar,         act, mats, nmat, out, M, K, scalar_calls);
        const uint64_t t_st = bench(gemv_bitnet_neon_st, act, mats, nmat, out, M, K, neon_calls);
        const uint64_t t_mt = bench(gemv_bitnet_neon,    act, mats, nmat, out, M, K, neon_calls);

        printf("\n  [%s]\n", regime == 0
               ? "hot: same matrix every call (weights L2-resident)"
               : "cold: rotating matrices (weights streamed from DRAM)");
        printf("  %-22s %10s %10s %12s %10s\n", "kernel", "ms", "GOP/s", "W GB/s", "speedup");
        print_row("scalar",           t_sc, t_sc, M, K);
        print_row("NEON 1 thread",    t_st, t_sc, M, K);
        print_row("NEON multi-thread",t_mt, t_sc, M, K);
    }

    for (int i = 0; i < nmat_cold; i++) free(mats[i]);
    free(act);
    free(out);
}

int main(int argc, char **argv) {
    const int quick = argc > 1 && strcmp(argv[1], "--quick") == 0;
    run_verification();
    run_benchmark(quick);
    return 0;
}
