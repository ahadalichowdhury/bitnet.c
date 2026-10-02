/*
 * gemv.h — Ternary (BitNet b1.58) GEMV: out[M] = W[M x K] * act[K].
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
 *     activations feeds 4 rows of decoded weights (built on the Step 1
 *     primitives in ternary_dot.h).
 *
 * Range: |out[m]| <= 128 * K, so K must stay below 2^24 to fit int32_t.
 */
#ifndef BITNET_GEMV_H
#define BITNET_GEMV_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ternary_dot.h"

/* GCD's persistent pool on macOS; elsewhere the standalone GEMV runs on the
 * calling thread (the transformer uses its own pthread pool, threadpool.h). */
#if defined(__APPLE__) && !defined(BITNET_PORTABLE)
#include <dispatch/dispatch.h>
#define GEMV_USE_GCD 1
#endif

/* Rows per GCD work item (multiple of 4). */
#ifndef GEMV_ROW_BLOCK
#define GEMV_ROW_BLOCK 32
#endif

/* K tile width in weights (multiple of 64). */
#ifndef GEMV_K_TILE
#define GEMV_K_TILE 2048
#endif

/* Below this many weights, run on the calling thread: dispatch overhead wins. */
#define GEMV_PARALLEL_MIN_WEIGHTS (64 * 1024)

_Static_assert(GEMV_ROW_BLOCK % 4 == 0, "GEMV_ROW_BLOCK must be a multiple of 4");
_Static_assert(GEMV_K_TILE % 64 == 0 && GEMV_K_TILE > 0, "GEMV_K_TILE must be a multiple of 64");

static inline size_t gemv_row_stride(int K) { return ((size_t)K + 3) / 4; }

/* ------------------------------------------------------------------------- */
/* Scalar reference                                                          */
/* ------------------------------------------------------------------------- */

static inline void gemv_scalar(const int8_t *act, const uint8_t *packed_weight_matrix,
                               int32_t *out, int M, int K) {
    const size_t stride = gemv_row_stride(K);
    for (int m = 0; m < M; m++)
        out[m] = ternary_dot_scalar_range(act, packed_weight_matrix + (size_t)m * stride, 0, K);
}

/* Scalar reference for either weight layout. */
static inline void gemv_scalar_layout(const int8_t *act, const uint8_t *W, int32_t *out, int M, int K,
                                      ternary_layout layout) {
    const size_t stride = ternary_row_bytes(K, layout);
    for (int m = 0; m < M; m++)
        out[m] = layout == TERNARY_I128
                     ? ternary_dot_i128_scalar_range(act, W + (size_t)m * stride, 0, K)
                     : ternary_dot_scalar_range(act, W + (size_t)m * stride, 0, K);
}

/* ------------------------------------------------------------------------- */
/* I128 layout micro-kernel: 4 rows x len weights (len: tile length; the     */
/* tile starts on a 128-weight block boundary)                               */
/* ------------------------------------------------------------------------- */

#if defined(BITNET_NEON)
static inline void ternary_dot4_i128_accumulate(const int8_t *act, const uint8_t *w, size_t stride,
                                                int len, int32_t out[4]) {
    const int8x16_t lut = vld1q_s8(k_ternary_lut);
    const uint8x16_t m3 = vdupq_n_u8(3);
    int32x4_t c[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
    const int full = len & ~127;
    for (int b = 0; b < full; b += 128) {
        const int8_t *a = act + b;
        int8x16_t av[8];
        for (int i = 0; i < 8; i++) av[i] = vld1q_s8(a + 16 * i);
        for (int r = 0; r < 4; r++) {
            const uint8_t *wb = w + (size_t)r * stride + (b >> 2);
            const uint8x16_t p0 = vld1q_u8(wb), p1 = vld1q_u8(wb + 16);
            int32x4_t acc = c[r];
            acc = TERNARY_DOT(acc, vqtbl1q_s8(lut, vandq_u8(p0, m3)), av[0]);
            acc = TERNARY_DOT(acc, vqtbl1q_s8(lut, vandq_u8(p1, m3)), av[1]);
            I128_NEON_FIELD(acc, p0, 1, av[2]);
            I128_NEON_FIELD(acc, p1, 1, av[3]);
            I128_NEON_FIELD(acc, p0, 2, av[4]);
            I128_NEON_FIELD(acc, p1, 2, av[5]);
            I128_NEON_FIELD(acc, p0, 3, av[6]);
            I128_NEON_FIELD(acc, p1, 3, av[7]);
            c[r] = acc;
        }
    }
    for (int r = 0; r < 4; r++) {
        out[r] += vaddvq_s32(c[r]);
        if (full < len) out[r] += ternary_dot_i128_scalar_range(act, w + (size_t)r * stride, full, len);
    }
}
#elif defined(BITNET_AVX2)
static inline void ternary_dot4_i128_accumulate(const int8_t *act, const uint8_t *w, size_t stride,
                                                int len, int32_t out[4]) {
    const __m256i lut = i128_lut_u(), m3 = _mm256_set1_epi8(3);
    __m256i c0 = _mm256_setzero_si256(), c1 = c0, c2 = c0, c3 = c0, asum = c0;
    const uint8_t *w0 = w, *w1 = w + stride, *w2 = w + 2 * stride, *w3 = w + 3 * stride;
    const int full = len & ~127;
    for (int b = 0; b < full; b += 128) {
        const int8_t *a = act + b;
        const int j = b >> 2;
        /* sum(w*a) = sum((w+1)*a) - sum(a); sum(a) is shared by the 4 rows. */
        asum = i128_asum_acc(asum, a);
        c0 = i128_block_acc(c0, w0 + j, a, lut, m3);
        c1 = i128_block_acc(c1, w1 + j, a, lut, m3);
        c2 = i128_block_acc(c2, w2 + j, a, lut, m3);
        c3 = i128_block_acc(c3, w3 + j, a, lut, m3);
    }
    const int32_t as = hsum_epi32_avx2(asum);
    out[0] += hsum_epi32_avx2(c0) - as;
    out[1] += hsum_epi32_avx2(c1) - as;
    out[2] += hsum_epi32_avx2(c2) - as;
    out[3] += hsum_epi32_avx2(c3) - as;
    if (full < len)
        for (int r = 0; r < 4; r++) out[r] += ternary_dot_i128_scalar_range(act, w + (size_t)r * stride, full, len);
}
#else
static inline void ternary_dot4_i128_accumulate(const int8_t *act, const uint8_t *w, size_t stride,
                                                int len, int32_t out[4]) {
    for (int r = 0; r < 4; r++) out[r] += ternary_dot_i128_scalar_range(act, w + (size_t)r * stride, 0, len);
}
#endif

/* ------------------------------------------------------------------------- */
/* Micro-kernel: 4 rows x len weights (NEON, AVX2, or scalar fallback)      */
/* ------------------------------------------------------------------------- */

/* out[r] += dot(row r) for 4 rows. `act` and `w` point at the same K offset
 * (a multiple of 4); rows are `stride` bytes apart. Two accumulators per row
 * halve the SDOT dependency chain. */
#ifdef BITNET_NEON
static inline void ternary_dot4_accumulate(const int8_t *act, const uint8_t *w, size_t stride,
                                           int len, int32_t out[4]) {
    const uint8_t *w0 = w;
    const uint8_t *w1 = w + stride;
    const uint8_t *w2 = w + 2 * stride;
    const uint8_t *w3 = w + 3 * stride;

    int32x4_t a0 = vdupq_n_s32(0), b0 = vdupq_n_s32(0);
    int32x4_t a1 = vdupq_n_s32(0), b1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), b2 = vdupq_n_s32(0);
    int32x4_t a3 = vdupq_n_s32(0), b3 = vdupq_n_s32(0);

#define ROW_STEP(acc_a, acc_b, wp)                                        \
    do {                                                                  \
        const int8x16x4_t d = ternary_decode64(vld1q_u8((wp) + (i >> 2))); \
        acc_a = TERNARY_DOT(acc_a, d.val[0], x.val[0]);                   \
        acc_b = TERNARY_DOT(acc_b, d.val[1], x.val[1]);                   \
        acc_a = TERNARY_DOT(acc_a, d.val[2], x.val[2]);                   \
        acc_b = TERNARY_DOT(acc_b, d.val[3], x.val[3]);                   \
    } while (0)

    const int full = len & ~63;
    for (int i = 0; i < full; i += 64) {
        const int8x16x4_t x = vld4q_s8(act + i);
        ROW_STEP(a0, b0, w0);
        ROW_STEP(a1, b1, w1);
        ROW_STEP(a2, b2, w2);
        ROW_STEP(a3, b3, w3);
    }
#undef ROW_STEP

    /* Horizontal reduce: lane r = sum over row r. */
    const int32x4_t s01 = vpaddq_s32(vaddq_s32(a0, b0), vaddq_s32(a1, b1));
    const int32x4_t s23 = vpaddq_s32(vaddq_s32(a2, b2), vaddq_s32(a3, b3));
    int32x4_t sums = vpaddq_s32(s01, s23);

    if (full < len) {
        const int32_t tail[4] = {
            ternary_dot_scalar_range(act, w0, full, len),
            ternary_dot_scalar_range(act, w1, full, len),
            ternary_dot_scalar_range(act, w2, full, len),
            ternary_dot_scalar_range(act, w3, full, len),
        };
        sums = vaddq_s32(sums, vld1q_s32(tail));
    }
    vst1q_s32(out, vaddq_s32(vld1q_s32(out), sums));
}
#elif defined(BITNET_AVX2)
/* AVX2: each 32-byte activation load and its maddubs(1, a) correction are
 * shared by the 4 rows (see ternary_dot.h for the exact decode/MADDUBS trick). */
static inline void ternary_dot4_accumulate(const int8_t *act, const uint8_t *w, size_t stride,
                                           int len, int32_t out[4]) {
    const __m256i ones8 = _mm256_set1_epi8(1);
    const uint8_t *w0 = w, *w1 = w + stride, *w2 = w + 2 * stride, *w3 = w + 3 * stride;
    __m256i c0 = _mm256_setzero_si256(), c1 = c0, c2 = c0, c3 = c0;
    const int full = len & ~31;
    for (int i = 0; i < full; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(act + i));
        const __m256i asum = _mm256_maddubs_epi16(ones8, a);
        const int j = i >> 2;
        c0 = _mm256_add_epi32(c0, ternary_block32_avx2(ternary_decode32_u_avx2(w0 + j), a, asum));
        c1 = _mm256_add_epi32(c1, ternary_block32_avx2(ternary_decode32_u_avx2(w1 + j), a, asum));
        c2 = _mm256_add_epi32(c2, ternary_block32_avx2(ternary_decode32_u_avx2(w2 + j), a, asum));
        c3 = _mm256_add_epi32(c3, ternary_block32_avx2(ternary_decode32_u_avx2(w3 + j), a, asum));
    }
    out[0] += hsum_epi32_avx2(c0);
    out[1] += hsum_epi32_avx2(c1);
    out[2] += hsum_epi32_avx2(c2);
    out[3] += hsum_epi32_avx2(c3);
    if (full < len) {
        out[0] += ternary_dot_scalar_range(act, w0, full, len);
        out[1] += ternary_dot_scalar_range(act, w1, full, len);
        out[2] += ternary_dot_scalar_range(act, w2, full, len);
        out[3] += ternary_dot_scalar_range(act, w3, full, len);
    }
}
#else
static inline void ternary_dot4_accumulate(const int8_t *act, const uint8_t *w, size_t stride,
                                           int len, int32_t out[4]) {
    for (int r = 0; r < 4; r++) out[r] += ternary_dot_scalar_range(act, w + (size_t)r * stride, 0, len);
}
#endif

/* ------------------------------------------------------------------------- */
/* Tiled, multi-threaded GEMV                                                */
/* ------------------------------------------------------------------------- */

typedef struct {
    const int8_t  *act;
    const uint8_t *W;
    int32_t       *out;
    int            M;
    int            K;
    size_t         stride;
    ternary_layout layout;
} gemv_ctx;

/* Computes rows [blk * GEMV_ROW_BLOCK, min(+GEMV_ROW_BLOCK, M)). */
static inline void gemv_row_block(void *ctx_, size_t blk) {
    const gemv_ctx *c = (const gemv_ctx *)ctx_;
    const int r0 = (int)blk * GEMV_ROW_BLOCK;
    const int r1 = (r0 + GEMV_ROW_BLOCK < c->M) ? r0 + GEMV_ROW_BLOCK : c->M;
    const size_t stride = c->stride;

    int32_t acc[GEMV_ROW_BLOCK] __attribute__((aligned(16))) = {0};

    for (int k0 = 0; k0 < c->K; k0 += GEMV_K_TILE) {
        const int len = (c->K - k0 < GEMV_K_TILE) ? c->K - k0 : GEMV_K_TILE;
        const int8_t *a = c->act + k0;
        const uint8_t *wtile = c->W + (size_t)(k0 >> 2);

        int r = r0;
        if (c->layout == TERNARY_I128) {
            for (; r + 4 <= r1; r += 4)
                ternary_dot4_i128_accumulate(a, wtile + (size_t)r * stride, stride, len, acc + (r - r0));
            for (; r < r1; r++)
                acc[r - r0] += ternary_dot_i128(a, wtile + (size_t)r * stride, len);
        } else {
            for (; r + 4 <= r1; r += 4)
                ternary_dot4_accumulate(a, wtile + (size_t)r * stride, stride, len, acc + (r - r0));
            for (; r < r1; r++)
                acc[r - r0] += ternary_dot_neon(a, wtile + (size_t)r * stride, len);
        }
    }

    memcpy(c->out + r0, acc, (size_t)(r1 - r0) * sizeof(int32_t));
}

static inline void gemv_neon_impl_layout(const int8_t *act, const uint8_t *packed_weight_matrix,
                                         int32_t *out, int M, int K, int allow_parallel,
                                         ternary_layout layout) {
    if (M <= 0) return;
    if (K <= 0) {
        memset(out, 0, (size_t)M * sizeof(int32_t));
        return;
    }

    gemv_ctx ctx = {act, packed_weight_matrix, out, M, K, ternary_row_bytes(K, layout), layout};
    const size_t nblk = ((size_t)M + GEMV_ROW_BLOCK - 1) / GEMV_ROW_BLOCK;

#ifdef GEMV_USE_GCD
    if (allow_parallel && (size_t)M * (size_t)K >= GEMV_PARALLEL_MIN_WEIGHTS && nblk > 1) {
        dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
        dispatch_apply_f(nblk, q, &ctx, gemv_row_block);
    } else
#else
    (void)allow_parallel;
#endif
    {
        for (size_t b = 0; b < nblk; b++) gemv_row_block(&ctx, b);
    }
}

static inline void gemv_neon_impl(const int8_t *act, const uint8_t *packed_weight_matrix,
                                  int32_t *out, int M, int K, int allow_parallel) {
    gemv_neon_impl_layout(act, packed_weight_matrix, out, M, K, allow_parallel, TERNARY_ROW4);
}

static inline void gemv_bitnet_neon(const int8_t *act, const uint8_t *packed_weight_matrix,
                                    int32_t *out, int M, int K) {
    gemv_neon_impl(act, packed_weight_matrix, out, M, K, 1);
}

/* SIMD GEMV for either layout (multi-threaded on macOS via GCD). */
static inline void gemv_bitnet_layout(const int8_t *act, const uint8_t *W, int32_t *out, int M, int K,
                                      ternary_layout layout, int allow_parallel) {
    gemv_neon_impl_layout(act, W, out, M, K, allow_parallel, layout);
}

/* Single-threaded variant: isolates SIMD speedup from threading speedup. */
static inline void gemv_bitnet_neon_st(const int8_t *act, const uint8_t *packed_weight_matrix,
                                       int32_t *out, int M, int K) {
    gemv_neon_impl(act, packed_weight_matrix, out, M, K, 0);
}

/* ------------------------------------------------------------------------- */
/* Batched GEMM for prompt prefill: out[t][m] = dot(W row m, act[t])         */
/*                                                                           */
/* Each 128-weight I128 block is decoded once and multiplied against          */
/* GEMM_TB tokens' activations, instead of once per token. All sums are      */
/* exact int32, so every token's result equals the GEMV result bit-for-bit.  */
/* ------------------------------------------------------------------------- */

#define GEMM_TB 8 /* tokens per register tile; activation buffers hold a multiple */

/* out[t] += sum over full 128-blocks [0, full) of row w . act[t], t < GEMM_TB.
 * On AVX2 the result is sum((w+1) * a); the caller subtracts sum(a) per token
 * (gemm_asum, shared by all rows). */
#if defined(BITNET_NEON)
static inline void ternary_gemm_i128(const int8_t *act, size_t as, const uint8_t *w, int full,
                                     int32_t out[GEMM_TB]) {
    const int8x16_t lut = vld1q_s8(k_ternary_lut);
    const uint8x16_t m3 = vdupq_n_u8(3);
    int32x4_t c[GEMM_TB];
    for (int t = 0; t < GEMM_TB; t++) c[t] = vdupq_n_s32(0);
    for (int b = 0; b < full; b += 128) {
        const uint8x16_t p0 = vld1q_u8(w + (b >> 2)), p1 = vld1q_u8(w + (b >> 2) + 16);
        const int8x16_t d0 = vqtbl1q_s8(lut, vandq_u8(p0, m3));
        const int8x16_t d1 = vqtbl1q_s8(lut, vandq_u8(p1, m3));
        const int8x16_t d2 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p0, 2), m3));
        const int8x16_t d3 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p1, 2), m3));
        const int8x16_t d4 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p0, 4), m3));
        const int8x16_t d5 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p1, 4), m3));
        const int8x16_t d6 = vqtbl1q_s8(lut, vshrq_n_u8(p0, 6));
        const int8x16_t d7 = vqtbl1q_s8(lut, vshrq_n_u8(p1, 6));
        for (int t = 0; t < GEMM_TB; t++) {
            const int8_t *a = act + (size_t)t * as + b;
            int32x4_t acc = c[t];
            acc = TERNARY_DOT(acc, d0, vld1q_s8(a));
            acc = TERNARY_DOT(acc, d1, vld1q_s8(a + 16));
            acc = TERNARY_DOT(acc, d2, vld1q_s8(a + 32));
            acc = TERNARY_DOT(acc, d3, vld1q_s8(a + 48));
            acc = TERNARY_DOT(acc, d4, vld1q_s8(a + 64));
            acc = TERNARY_DOT(acc, d5, vld1q_s8(a + 80));
            acc = TERNARY_DOT(acc, d6, vld1q_s8(a + 96));
            acc = TERNARY_DOT(acc, d7, vld1q_s8(a + 112));
            c[t] = acc;
        }
    }
    for (int t = 0; t < GEMM_TB; t++) out[t] += vaddvq_s32(c[t]);
}

static inline void gemm_asum(const int8_t *act, size_t as, int full, int32_t corr[GEMM_TB]) {
    (void)act; (void)as; (void)full;
    for (int t = 0; t < GEMM_TB; t++) corr[t] = 0;
}
#elif defined(BITNET_AVX2)
static inline void ternary_gemm_i128(const int8_t *act, size_t as, const uint8_t *w, int full,
                                     int32_t out[GEMM_TB]) {
    const __m256i lut = i128_lut_u(), m3 = _mm256_set1_epi8(3);
#ifndef BITNET_VNNI
    const __m256i ones16 = _mm256_set1_epi16(1);
#endif
    __m256i c[GEMM_TB];
    for (int t = 0; t < GEMM_TB; t++) c[t] = _mm256_setzero_si256();
    for (int b = 0; b < full; b += 128) {
        const __m256i p = _mm256_loadu_si256((const __m256i *)(w + (b >> 2)));
        const __m256i u0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(p, m3));
        const __m256i u1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 2), m3));
        const __m256i u2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 4), m3));
        const __m256i u3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 6), m3));
        for (int t = 0; t < GEMM_TB; t++) {
            const int8_t *a = act + (size_t)t * as + b;
#ifdef BITNET_VNNI
            __m256i ct = c[t];
            ct = BITNET_DPBUSD(ct, u0, _mm256_loadu_si256((const __m256i *)a));
            ct = BITNET_DPBUSD(ct, u1, _mm256_loadu_si256((const __m256i *)(a + 32)));
            ct = BITNET_DPBUSD(ct, u2, _mm256_loadu_si256((const __m256i *)(a + 64)));
            c[t] = BITNET_DPBUSD(ct, u3, _mm256_loadu_si256((const __m256i *)(a + 96)));
#else
            /* 4 MADDUBS lanes in [-512, 508] each: the int16 sum cannot overflow. */
            __m256i s = _mm256_maddubs_epi16(u0, _mm256_loadu_si256((const __m256i *)a));
            s = _mm256_add_epi16(s, _mm256_maddubs_epi16(u1, _mm256_loadu_si256((const __m256i *)(a + 32))));
            s = _mm256_add_epi16(s, _mm256_maddubs_epi16(u2, _mm256_loadu_si256((const __m256i *)(a + 64))));
            s = _mm256_add_epi16(s, _mm256_maddubs_epi16(u3, _mm256_loadu_si256((const __m256i *)(a + 96))));
            c[t] = _mm256_add_epi32(c[t], _mm256_madd_epi16(s, ones16));
#endif
        }
    }
    for (int t = 0; t < GEMM_TB; t++) out[t] += hsum_epi32_avx2(c[t]);
}

/* corr[t] = sum(act[t][0, full)): subtracted once per row (w = (w+1) - 1). */
static inline void gemm_asum(const int8_t *act, size_t as, int full, int32_t corr[GEMM_TB]) {
    for (int t = 0; t < GEMM_TB; t++) {
        __m256i s = _mm256_setzero_si256();
        for (int b = 0; b < full; b += 128) s = i128_asum_acc(s, act + (size_t)t * as + b);
        corr[t] = hsum_epi32_avx2(s);
    }
}
#else
static inline void ternary_gemm_i128(const int8_t *act, size_t as, const uint8_t *w, int full,
                                     int32_t out[GEMM_TB]) {
    for (int t = 0; t < GEMM_TB; t++)
        out[t] += ternary_dot_i128_scalar_range(act + (size_t)t * as, w, 0, full);
}

static inline void gemm_asum(const int8_t *act, size_t as, int full, int32_t corr[GEMM_TB]) {
    (void)act; (void)as; (void)full;
    for (int t = 0; t < GEMM_TB; t++) corr[t] = 0;
}
#endif

typedef struct {
    const int8_t  *act;        /* [T rounded up to GEMM_TB][act_stride]; rows >= T are ignored */
    size_t         act_stride;
    int32_t       *out;        /* out[t * out_stride + m] */
    size_t         out_stride;
    int            T;
    const uint8_t *W;
    int            M, K;
    size_t         stride;     /* weight row bytes */
    ternary_layout layout;
} gemm_ctx;

#define GEMM_MAX_T 32 /* tokens per gemm call */

/* Rows [blk * GEMV_ROW_BLOCK, +GEMV_ROW_BLOCK) for all T tokens. */
static inline void gemm_row_block(void *ctx_, size_t blk) {
    const gemm_ctx *c = (const gemm_ctx *)ctx_;
    const int r0 = (int)blk * GEMV_ROW_BLOCK;
    const int r1 = (r0 + GEMV_ROW_BLOCK < c->M) ? r0 + GEMV_ROW_BLOCK : c->M;
    const int T = c->T, ngroups = (T + GEMM_TB - 1) / GEMM_TB;
    int32_t acc[GEMV_ROW_BLOCK][GEMM_MAX_T];
    memset(acc, 0, sizeof(acc));

    for (int k0 = 0; k0 < c->K; k0 += GEMV_K_TILE) {
        const int len = (c->K - k0 < GEMV_K_TILE) ? c->K - k0 : GEMV_K_TILE;
        const uint8_t *wtile = c->W + (size_t)(k0 >> 2);
        if (c->layout != TERNARY_I128) { /* ROW4 (v1 files): per-token 4-row kernel */
            for (int t = 0; t < T; t++) {
                const int8_t *a = c->act + (size_t)t * c->act_stride + k0;
                int32_t tmp[GEMV_ROW_BLOCK] = {0};
                int r = r0;
                for (; r + 4 <= r1; r += 4)
                    ternary_dot4_accumulate(a, wtile + (size_t)r * c->stride, c->stride, len, tmp + (r - r0));
                for (; r < r1; r++) tmp[r - r0] += ternary_dot_neon(a, wtile + (size_t)r * c->stride, len);
                for (r = r0; r < r1; r++) acc[r - r0][t] += tmp[r - r0];
            }
            continue;
        }
        const int full = len & ~127;
        for (int g = 0; g < ngroups; g++) {
            const int8_t *a = c->act + (size_t)g * GEMM_TB * c->act_stride + k0;
            const int nt = T - g * GEMM_TB < GEMM_TB ? T - g * GEMM_TB : GEMM_TB;
            int32_t corr[GEMM_TB];
            gemm_asum(a, c->act_stride, full, corr);
            for (int r = r0; r < r1; r++) {
                const uint8_t *w = wtile + (size_t)r * c->stride;
                int32_t s[GEMM_TB] = {0};
                ternary_gemm_i128(a, c->act_stride, w, full, s);
                for (int t = 0; t < nt; t++) {
                    int32_t v = s[t] - corr[t];
                    if (full < len) v += ternary_dot_i128_scalar_range(a + (size_t)t * c->act_stride, w, full, len);
                    acc[r - r0][g * GEMM_TB + t] += v;
                }
            }
        }
    }
    for (int r = r0; r < r1; r++)
        for (int t = 0; t < T; t++) c->out[(size_t)t * c->out_stride + r] = acc[r - r0][t];
}

/* Single-threaded batched GEMM (the transformer runs gemm_row_block on its pool).
 * act must hold ceil(T / GEMM_TB) * GEMM_TB readable rows; 1 <= T <= GEMM_MAX_T. */
static inline void gemm_bitnet_layout(const int8_t *act, size_t act_stride, int T, const uint8_t *W,
                                      int32_t *out, size_t out_stride, int M, int K,
                                      ternary_layout layout) {
    gemm_ctx ctx = {act, act_stride, out, out_stride, T, W, M, K, ternary_row_bytes(K, layout), layout};
    for (size_t b = 0; b < ((size_t)M + GEMV_ROW_BLOCK - 1) / GEMV_ROW_BLOCK; b++) gemm_row_block(&ctx, b);
}

#endif /* BITNET_GEMV_H */
