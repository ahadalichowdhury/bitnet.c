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

/* ------------------------------------------------------------------------- */
/* Micro-kernel: 4 rows x len weights (NEON, or scalar fallback)            */
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
        for (; r + 4 <= r1; r += 4) {
            ternary_dot4_accumulate(a, wtile + (size_t)r * stride, stride, len, acc + (r - r0));
        }
        for (; r < r1; r++)
            acc[r - r0] += ternary_dot_neon(a, wtile + (size_t)r * stride, len);
    }

    memcpy(c->out + r0, acc, (size_t)(r1 - r0) * sizeof(int32_t));
}

static inline void gemv_neon_impl(const int8_t *act, const uint8_t *packed_weight_matrix,
                                  int32_t *out, int M, int K, int allow_parallel) {
    if (M <= 0) return;
    if (K <= 0) {
        memset(out, 0, (size_t)M * sizeof(int32_t));
        return;
    }

    gemv_ctx ctx = {act, packed_weight_matrix, out, M, K, gemv_row_stride(K)};
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

static inline void gemv_bitnet_neon(const int8_t *act, const uint8_t *packed_weight_matrix,
                                    int32_t *out, int M, int K) {
    gemv_neon_impl(act, packed_weight_matrix, out, M, K, 1);
}

/* Single-threaded variant: isolates SIMD speedup from threading speedup. */
static inline void gemv_bitnet_neon_st(const int8_t *act, const uint8_t *packed_weight_matrix,
                                       int32_t *out, int M, int K) {
    gemv_neon_impl(act, packed_weight_matrix, out, M, K, 0);
}

#endif /* BITNET_GEMV_H */
