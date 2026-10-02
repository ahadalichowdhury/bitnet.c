/*
 * test_dot_product.c — Ternary (BitNet b1.58) x int8 dot-product kernel.
 *
 * Kernel implementation lives in ternary_dot.h (shared with the engine).
 *
 * Weight packing (4 weights per byte, LSB first):
 *   weight i lives in byte i/4, bits [2*(i%4) .. 2*(i%4)+1]
 *   code 00 = 0, 01 = +1, 10 = -1, 11 = reserved (decoded as 0)
 *
 * Activations: int8_t, one per weight.
 *
 * Build & run: `make test` (binary: build/test_dot_product); `make asan` for sanitizers.
 *
 * Range: |result| <= 128 * size, so size must stay below 2^24 to fit int32_t.
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

#include "platform.h"
#include "ternary_dot.h"

/* ------------------------------------------------------------------------- */
/* Kernels (implementation shared via ternary_dot.h)                         */
/* ------------------------------------------------------------------------- */

int32_t dot_product_scalar(const int8_t *act, const uint8_t *packed_w, int size) {
    return ternary_dot_scalar_range(act, packed_w, 0, size);
}

int32_t dot_product_neon(const int8_t *act, const uint8_t *packed_w, int size) {
    return ternary_dot_neon(act, packed_w, size);
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

static void *aligned_buf(size_t bytes) {
    void *p = NULL;
    if (bytes == 0) bytes = 64;
    if (posix_memalign(&p, 64, bytes) != 0) {
        fprintf(stderr, "posix_memalign(%zu) failed\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

static inline size_t packed_bytes(int size) { return ((size_t)size + 3) / 4; }

/* Packs ternary values {-1, 0, 1} into the 2-bit format. */
static void pack_ternary(const int8_t *w, uint8_t *packed, int size) {
    memset(packed, 0, packed_bytes(size));
    for (int i = 0; i < size; i++) {
        const uint8_t code = w[i] == 1 ? 0x1 : (w[i] == -1 ? 0x2 : 0x0);
        packed[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
    }
}

static void fill_random(int8_t *act, int8_t *w, int size) {
    for (int i = 0; i < size; i++) {
        act[i] = (int8_t)(uint8_t)xorshift64();
        w[i]   = (int8_t)((int)(xorshift64() % 3) - 1);
    }
}

/* Independent unpacked reference (does not share the decoding path). */
static int32_t dot_unpacked_reference(const int8_t *act, const int8_t *w, int size) {
    int64_t sum = 0;
    for (int i = 0; i < size; i++) sum += (int64_t)act[i] * w[i];
    return (int32_t)sum;
}

static void verify_case(const char *label, const int8_t *act, const int8_t *w,
                        uint8_t *packed, int size) {
    pack_ternary(w, packed, size);
    const int32_t ref  = dot_unpacked_reference(act, w, size);
    const int32_t sc   = dot_product_scalar(act, packed, size);
    const int32_t simd = dot_product_neon(act, packed, size);
    if (sc != ref || simd != ref) {
        fprintf(stderr, "MISMATCH [%s] N=%d ref=%d scalar=%d neon=%d\n",
                label, size, ref, sc, simd);
    }
    assert(sc == ref);
    assert(simd == ref);
}

static void run_verification(void) {
    enum { MAX_N = 4096 + 128 };
    int8_t  *act    = aligned_buf(MAX_N);
    int8_t  *w      = aligned_buf(MAX_N);
    uint8_t *packed = aligned_buf(packed_bytes(MAX_N));

    /* Primary target: N = 4096, many random trials. */
    for (int t = 0; t < 1000; t++) {
        fill_random(act, w, 4096);
        verify_case("random N=4096", act, w, packed, 4096);
    }

    /* Saturation extremes: |sum| = 128 * N, and -128 * -1 = +128 per lane. */
    for (int i = 0; i < 4096; i++) { act[i] = -128; w[i] = -1; }
    verify_case("extreme -128*-1", act, w, packed, 4096);
    assert(dot_product_neon(act, packed, 4096) == 128 * 4096);

    for (int i = 0; i < 4096; i++) { act[i] = -128; w[i] = 1; }
    verify_case("extreme -128*+1", act, w, packed, 4096);

    for (int i = 0; i < 4096; i++) { act[i] = 127; w[i] = 1; }
    verify_case("extreme 127*+1", act, w, packed, 4096);

    for (int i = 0; i < 4096; i++) { act[i] = 127; w[i] = 0; }
    verify_case("all-zero weights", act, w, packed, 4096);
    assert(dot_product_neon(act, packed, 4096) == 0);

    /* Tail handling: sizes not divisible by 64 (or by 4). */
    static const int sizes[] = {0, 1, 2, 3, 4, 5, 63, 64, 65, 127, 128, 129,
                                1000, 1023, 4095, 4096 + 17, 4096 + 127};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        for (int t = 0; t < 50; t++) {
            fill_random(act, w, sizes[s]);
            verify_case("tail", act, w, packed, sizes[s]);
        }
    }

    /* Reserved code 0b11 must decode as 0 in both kernels. */
    for (int t = 0; t < 100; t++) {
        for (int i = 0; i < 4096; i++) act[i] = (int8_t)(uint8_t)xorshift64();
        for (size_t b = 0; b < packed_bytes(4096); b++) packed[b] = (uint8_t)xorshift64();
        assert(dot_product_scalar(act, packed, 4096) == dot_product_neon(act, packed, 4096));
    }

    free(act);
    free(w);
    free(packed);
    printf("Verification: PASSED (N=4096 x1000 random, extremes, tails, reserved codes)\n");
}

/* ------------------------------------------------------------------------- */
/* Benchmark                                                                 */
/* ------------------------------------------------------------------------- */

typedef int32_t (*dot_fn)(const int8_t *, const uint8_t *, int);

static inline uint64_t now_ns(void) {
    return platform_now_ns();
}

/* Returns best-of-`reps` total wall time in ns for `iters` calls. */
static uint64_t bench(dot_fn fn, const int8_t *act, const uint8_t *packed, int size,
                      int iters, int reps, volatile int32_t *sink) {
    uint64_t best = UINT64_MAX;
    for (int r = 0; r < reps; r++) {
        int32_t acc = 0;
        const uint64_t t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            /* Opaque barrier: stops the compiler hoisting the pure call. */
            __asm__ volatile("" : : "r"(act), "r"(packed) : "memory");
            acc += fn(act, packed, size);
        }
        const uint64_t dt = now_ns() - t0;
        *sink = acc;
        if (dt < best) best = dt;
    }
    return best;
}

static void run_benchmark(void) {
    enum { N = 4096, ITERS = 200000, REPS = 5 };
    int8_t  *act    = aligned_buf(N);
    int8_t  *w      = aligned_buf(N);
    uint8_t *packed = aligned_buf(packed_bytes(N));
    volatile int32_t sink = 0;

    fill_random(act, w, N);
    pack_ternary(w, packed, N);

    /* Warm caches and branch predictors. */
    bench(dot_product_scalar, act, packed, N, ITERS / 10, 1, &sink);
    bench(dot_product_neon,   act, packed, N, ITERS / 10, 1, &sink);

    const uint64_t t_scalar = bench(dot_product_scalar, act, packed, N, ITERS, REPS, &sink);
    const uint64_t t_neon   = bench(dot_product_neon,   act, packed, N, ITERS, REPS, &sink);

    const double ms_scalar = (double)t_scalar / 1e6;
    const double ms_neon   = (double)t_neon / 1e6;
    const double ns_scalar = (double)t_scalar / ITERS;
    const double ns_neon   = (double)t_neon / ITERS;
    /* 2 ops (mul + add) per weight. */
    const double gops_scalar = 2.0 * N / ns_scalar;
    const double gops_neon   = 2.0 * N / ns_neon;

    printf("\nBenchmark: N=%d, %d iterations, best of %d runs\n", N, ITERS, REPS);
    printf("  %-8s %12s %14s %12s\n", "kernel", "total (ms)", "per call (ns)", "GOP/s");
    printf("  %-8s %12.3f %14.1f %12.2f\n", "scalar", ms_scalar, ns_scalar, gops_scalar);
    printf("  %-8s %12.3f %14.1f %12.2f\n", "neon",   ms_neon,   ns_neon,   gops_neon);
    printf("  Speedup (scalar / neon): %.2fx\n", ms_scalar / ms_neon);
    printf("  SIMD path: %s\n", TERNARY_DOT_PATH);
    (void)sink;

    free(act);
    free(w);
    free(packed);
}

int main(void) {
    run_verification();
    run_benchmark();
    return 0;
}
