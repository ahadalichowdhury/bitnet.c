/*
 * test_loader.c — Verification and benchmark driver for the .bitnet loader.
 *
 *   build/test_loader MODEL.bitnet [--ref MODEL.ref] [--corrupt-tests] [--tmpdir DIR]
 *
 *   1. Loads (mmap, zero-copy) and validates the model; prints config + tensors.
 *   2. Verifies every payload CRC.
 *   3. --ref: for every ternary tensor, runs gemv_bitnet_neon and gemv_scalar
 *      directly on the mapped weights with the exporter's random int8 input and
 *      requires the int32 output to equal the one Python computed from the
 *      unpacked ternary matrix (proves bit layout + scale compatibility). Also
 *      checks embedding rows bit-for-bit.
 *   4. --corrupt-tests: writes ~20 corrupted copies of the file and requires the
 *      loader to reject each with the expected diagnostic (no crash, no leak).
 *   5. Benchmarks load latency, resident memory before/after touching weights,
 *      CRC throughput and a pass of GEMVs over every ternary tensor.
 *
 * Build & run: `make test` (binary: build/test_loader); `make asan` for sanitizers.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#undef NDEBUG /* verification asserts must always run */
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


#include "gemv.h"
#include "model_loader.h"
#include "platform.h"

/* ------------------------------------------------------------------------- */
/* Utilities                                                                 */
/* ------------------------------------------------------------------------- */

static inline uint64_t now_ns(void) { return platform_now_ns(); }

static size_t resident_bytes(void) { return platform_resident_bytes(); }

static void *aligned_buf(size_t bytes) {
    void *p = NULL;
    if (bytes == 0) bytes = 64;
    if (posix_memalign(&p, 64, bytes) != 0) {
        fprintf(stderr, "posix_memalign(%zu) failed\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

static uint8_t *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(EXIT_FAILURE);
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = aligned_buf((size_t)n);
    if (n < 0 || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "%s: read failed\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(f);
    *size = (size_t)n;
    return buf;
}

/* ------------------------------------------------------------------------- */
/* 1. Load + summary                                                         */
/* ------------------------------------------------------------------------- */

static void print_summary(const bitnet_model *m) {
    const bitnet_config *c = &m->config;
    printf("Model: vocab=%d dim=%d hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d "
           "max_seq=%d\n", c->vocab_size, c->dim, c->hidden_dim, c->n_layers, c->n_heads,
           c->n_kv_heads, c->head_dim, c->max_seq_len);
    printf("       ffn=%s norm_eps=%g rope_theta=%g tied=%s sub_norms=%s\n",
           c->ffn_act == BITNET_ACT_RELU2 ? "relu2" : "silu", (double)c->norm_eps,
           (double)c->rope_theta, (c->flags & BITNET_FLAG_TIED_EMBEDDINGS) ? "yes" : "no",
           (c->flags & BITNET_FLAG_SUB_NORMS) ? "yes" : "no");

    size_t bytes[3] = {0}, count[3] = {0}, params_ternary = 0;
    for (int i = 0; i < m->n_tensors; i++) {
        const bitnet_tensor *t = &m->tensors[i];
        bytes[t->dtype] += t->nbytes;
        count[t->dtype]++;
        if (t->dtype == BITNET_DTYPE_TERNARY) params_ternary += (size_t)t->rows * t->cols;
    }
    printf("Tensors: %d total | ternary %zu (%.1f MiB, %.1f M params) | f16 %zu (%.1f MiB) | "
           "f32 %zu (%.2f MiB)\n", m->n_tensors, count[2], bytes[2] / 1048576.0,
           params_ternary / 1e6, count[1], bytes[1] / 1048576.0, count[0], bytes[0] / 1048576.0);
}

/* ------------------------------------------------------------------------- */
/* 3. Cross-language reference check                                         */
/* ------------------------------------------------------------------------- */

typedef struct {
    const uint8_t *p, *end;
} cursor;

static const void *take(cursor *c, size_t n) {
    if ((size_t)(c->end - c->p) < n) {
        fprintf(stderr, "ref file truncated\n");
        exit(EXIT_FAILURE);
    }
    const void *r = c->p;
    c->p += n;
    return r;
}

static uint32_t take_u32(cursor *c) {
    uint32_t v;
    memcpy(&v, take(c, 4), 4);
    return v;
}

static void verify_reference(const bitnet_model *m, const char *ref_path) {
    size_t size;
    uint8_t *buf = read_file(ref_path, &size);
    cursor c = {buf, buf + size};
    assert(memcmp(take(&c, 4), "BREF", 4) == 0);
    assert(take_u32(&c) == 1);
    const uint32_t n_tern = take_u32(&c), n_emb = take_u32(&c), dim = take_u32(&c);
    assert((int)dim == m->config.dim);

    int max_m = 0, max_k = 0;
    for (int i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].rows > max_m) max_m = m->tensors[i].rows;
        if (m->tensors[i].cols > max_k) max_k = m->tensors[i].cols;
    }
    int8_t  *x  = aligned_buf((size_t)max_k);
    int32_t *yn = aligned_buf((size_t)max_m * sizeof(int32_t));
    int32_t *ys = aligned_buf((size_t)max_m * sizeof(int32_t));
    int32_t *ye = aligned_buf((size_t)max_m * sizeof(int32_t));
    float   *row = aligned_buf(dim * sizeof(float));

    int n_tensors_checked = 0;
    for (uint32_t r = 0; r < n_tern; r++) {
        char name[BITNET_NAME_LEN + 1] = {0};
        memcpy(name, take(&c, BITNET_NAME_LEN), BITNET_NAME_LEN);
        const uint32_t M = take_u32(&c), K = take_u32(&c);
        float beta;
        memcpy(&beta, take(&c, 4), 4);
        memcpy(x, take(&c, K), K);
        memcpy(ye, take(&c, (size_t)M * 4), (size_t)M * 4);

        const bitnet_tensor *t = bitnet_model_find(m, name);
        if (!t) {
            fprintf(stderr, "ref tensor %s not in model\n", name);
            exit(EXIT_FAILURE);
        }
        assert(t->dtype == BITNET_DTYPE_TERNARY);
        assert(t->rows == (int)M && t->cols == (int)K);
        assert(t->scale == beta);
        assert(((uintptr_t)t->data & 63) == 0);

        gemv_bitnet_neon(x, (const uint8_t *)t->data, yn, (int)M, (int)K);
        gemv_scalar(x, (const uint8_t *)t->data, ys, (int)M, (int)K);
        for (uint32_t i = 0; i < M; i++) {
            if (yn[i] != ye[i] || ys[i] != ye[i]) {
                fprintf(stderr, "MISMATCH %s row %u: python=%d neon=%d scalar=%d\n",
                        name, i, ye[i], yn[i], ys[i]);
                exit(EXIT_FAILURE);
            }
        }
        n_tensors_checked++;
    }

    for (uint32_t r = 0; r < n_emb; r++) {
        const uint32_t tok = take_u32(&c);
        const float *want = take(&c, (size_t)dim * 4);
        assert((int)tok < m->config.vocab_size);
        bitnet_embedding_row(m, (int)tok, row);
        if (memcmp(row, want, (size_t)dim * 4) != 0) {
            fprintf(stderr, "MISMATCH embedding row %u\n", tok);
            exit(EXIT_FAILURE);
        }
    }
    assert(c.p == c.end);

    free(buf); free(x); free(yn); free(ys); free(ye); free(row);
    printf("Reference check: PASSED (%d ternary tensors: NEON GEMV == scalar GEMV == Python "
           "int32, bit-exact; %u embedding rows bit-exact; beta matches)\n",
           n_tensors_checked, n_emb);
}

/* ------------------------------------------------------------------------- */
/* 4. Corruption tests                                                       */
/* ------------------------------------------------------------------------- */

typedef struct {
    uint8_t *b;
    size_t   n;
} image;

static bitnet_file_header *hdr(image *im) { return (bitnet_file_header *)im->b; }

static bitnet_file_tensor *entry_named(image *im, const char *name) {
    bitnet_file_header *h = hdr(im);
    bitnet_file_tensor *e = (bitnet_file_tensor *)(im->b + h->table_offset);
    for (uint32_t i = 0; i < h->n_tensors; i++)
        if (strncmp(e[i].name, name, BITNET_NAME_LEN) == 0) return &e[i];
    fprintf(stderr, "corrupt test: no tensor %s\n", name);
    exit(EXIT_FAILURE);
}

static bitnet_file_tensor *entry_at(image *im, int i) {
    return (bitnet_file_tensor *)(im->b + hdr(im)->table_offset) + i;
}

static void fix_crcs(image *im) {
    bitnet_file_header *h = hdr(im);
    /* Re-sign the table only if it lies inside the image; an out-of-bounds table
     * (C_N_TENSORS) must be rejected by the loader's bounds check instead. */
    const uint64_t table_bytes = (uint64_t)h->n_tensors * BITNET_ENTRY_SIZE;
    if (h->table_offset <= im->n && table_bytes <= im->n - h->table_offset)
        h->table_crc32 = bitnet_crc32(0, im->b + h->table_offset, (size_t)table_bytes);
    h->header_crc32 = 0;
    h->header_crc32 = bitnet_crc32(0, h, sizeof(*h));
}

typedef enum {
    C_TRUNC_1, C_TRUNC_SMALL, C_MAGIC, C_VERSION, C_HEADER_FLIP, C_TABLE_FLIP, C_MISALIGNED,
    C_PAST_EOF, C_NBYTES_HUGE, C_SHAPE, C_DTYPE, C_NAME_UNTERMINATED, C_DUP_NAME, C_OVERLAP,
    C_HEADS, C_FLAGS, C_MISSING, C_N_TENSORS, C_NAN_SCALE, C_SUBNORM_FLAG, C_PAYLOAD_FLIP,
    C_COUNT
} corruption;

static const struct {
    const char *what;
    const char *expect; /* substring of the loader's error */
    int         fix_crc;
} k_cases[C_COUNT] = {
    [C_TRUNC_1]           = {"truncated by 1 byte",             "file size",        0},
    [C_TRUNC_SMALL]       = {"truncated to 100 bytes",          "smaller than",     0},
    [C_MAGIC]             = {"wrong magic",                     "bad magic",        0},
    [C_VERSION]           = {"future version",                  "unsupported version", 1},
    [C_HEADER_FLIP]       = {"header bit flip",                 "header crc",       0},
    [C_TABLE_FLIP]        = {"tensor table bit flip",           "table crc",        0},
    [C_MISALIGNED]        = {"misaligned tensor offset",        "aligned",          1},
    [C_PAST_EOF]          = {"tensor offset past EOF",          "out of bounds",    1},
    [C_NBYTES_HUGE]       = {"nbytes = UINT64_MAX",             "out of bounds",    1},
    [C_SHAPE]             = {"norm shape != config",            "expected",         1},
    [C_DTYPE]             = {"norm stored as f16",              "unexpected dtype", 1},
    [C_NAME_UNTERMINATED] = {"name without NUL",                "NUL-terminated",   1},
    [C_DUP_NAME]          = {"duplicate tensor name",           "duplicate",        1},
    [C_OVERLAP]           = {"overlapping payloads",            "overlap",          1},
    [C_HEADS]             = {"dim not divisible by n_heads",    "not divisible",    1},
    [C_FLAGS]             = {"unknown flag bit",                "unknown flags",    1},
    [C_MISSING]           = {"renamed (missing) tensor",        "missing tensor",   1},
    [C_N_TENSORS]         = {"n_tensors past end of file",      "out of bounds",    1},
    [C_NAN_SCALE]         = {"NaN ternary scale",               "scale",            1},
    [C_SUBNORM_FLAG]      = {"sub-norm flag cleared",           "unrecognized",     1},
    [C_PAYLOAD_FLIP]      = {"payload bit flip (load ok, crc)", "payload crc",      0},
};

static void apply(corruption k, image *im) {
    bitnet_file_header *h = hdr(im);
    switch (k) {
    case C_TRUNC_1:     im->n -= 1; break;
    case C_TRUNC_SMALL: im->n = 100; break;
    case C_MAGIC:       im->b[0] = 'X'; break;
    case C_VERSION:     h->version = 2; break;
    case C_HEADER_FLIP: h->dim ^= 1u; break;
    case C_TABLE_FLIP:  entry_at(im, 0)->rows ^= 1u; break;
    case C_MISALIGNED:  entry_at(im, 1)->offset += 1; break;
    case C_PAST_EOF:    entry_at(im, 1)->offset = (h->file_size + 128) & ~63ull; break;
    case C_NBYTES_HUGE: entry_at(im, 1)->nbytes = UINT64_MAX; break;
    case C_SHAPE: {
        bitnet_file_tensor *e = entry_named(im, "layers.0.attn_norm");
        e->rows -= 16;
        e->nbytes = (uint64_t)e->rows * 4;
        break;
    }
    case C_DTYPE: {
        bitnet_file_tensor *e = entry_named(im, "layers.0.ffn_norm");
        e->dtype = BITNET_DTYPE_F16;
        e->nbytes /= 2;
        break;
    }
    case C_NAME_UNTERMINATED: memset(entry_at(im, 0)->name, 'A', BITNET_NAME_LEN); break;
    case C_DUP_NAME: memcpy(entry_at(im, 2)->name, entry_at(im, 1)->name, BITNET_NAME_LEN); break;
    case C_OVERLAP:  entry_at(im, 2)->offset = entry_at(im, 1)->offset; break;
    case C_HEADS:    h->n_heads = 7; h->n_kv_heads = 7; break;
    case C_FLAGS:    h->flags |= 0x80u; break;
    case C_MISSING:  entry_named(im, "layers.1.ffn.w_down")->name[0] = 'X'; break;
    case C_N_TENSORS: h->n_tensors = 1u << 19; break;
    case C_NAN_SCALE: entry_named(im, "layers.0.attn.wq")->scale = NAN; break;
    case C_SUBNORM_FLAG: h->flags &= ~(uint32_t)BITNET_FLAG_SUB_NORMS; break;
    case C_PAYLOAD_FLIP: {
        bitnet_file_tensor *e = entry_named(im, "layers.0.ffn.w_up");
        im->b[e->offset + e->nbytes / 2] ^= 0x10;
        break;
    }
    case C_COUNT: break;
    }
}

static void run_corruption_tests(const char *path, const char *tmpdir) {
    size_t size;
    uint8_t *orig = read_file(path, &size);
    if (!(hdr(&(image){orig, size})->flags & BITNET_FLAG_SUB_NORMS)) {
        fprintf(stderr, "corruption tests expect a model with sub-norms (use --mock)\n");
        exit(EXIT_FAILURE);
    }
    uint8_t *work = aligned_buf(size);
    char tmpl[1024];
    printf("\nCorruption tests (loader must reject each with a precise diagnostic):\n");

    for (int k = 0; k < C_COUNT; k++) {
        memcpy(work, orig, size);
        image im = {work, size};
        apply((corruption)k, &im);
        if (k_cases[k].fix_crc) fix_crcs(&im);

        snprintf(tmpl, sizeof(tmpl), "%s/bitnet_corrupt_XXXXXX", tmpdir);
        const int fd = mkstemp(tmpl);
        if (fd < 0) {
            perror("mkstemp");
            exit(EXIT_FAILURE);
        }
        if (write(fd, im.b, im.n) != (ssize_t)im.n) {
            perror("write");
            exit(EXIT_FAILURE);
        }
        close(fd);

        bitnet_model m;
        char err[256] = {0};
        int rc = bitnet_model_load(tmpl, &m, err, sizeof(err));
        if (k == C_PAYLOAD_FLIP) {
            assert(rc == 0); /* payloads are only checked on demand */
            rc = bitnet_model_verify_crc(&m, err, sizeof(err));
            bitnet_model_free(&m);
        } else {
            /* A failed load must leave the model zeroed (nothing to free). */
            assert(m.map == NULL && m.tensors == NULL && m.layers == NULL);
        }
        unlink(tmpl);

        const int ok = rc != 0 && strstr(err, k_cases[k].expect) != NULL;
        printf("  %-34s %s  %s\n", k_cases[k].what, ok ? "rejected" : "NOT CAUGHT", err);
        assert(ok);
    }
    free(orig);
    free(work);
    printf("Corruption tests: PASSED (%d cases)\n", C_COUNT);
}

/* ------------------------------------------------------------------------- */
/* 5. Benchmarks                                                             */
/* ------------------------------------------------------------------------- */

static int cmp_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static double gemv_pass_ms(const bitnet_model *m, int8_t *x, int32_t *y) {
    const uint64_t t0 = now_ns();
    for (int i = 0; i < m->n_tensors; i++) {
        const bitnet_tensor *t = &m->tensors[i];
        if (t->dtype != BITNET_DTYPE_TERNARY) continue;
        gemv_bitnet_neon(x, (const uint8_t *)t->data, y, t->rows, t->cols);
        __asm__ volatile("" : : "r"(y) : "memory");
    }
    return (double)(now_ns() - t0) / 1e6;
}

static void run_benchmark(const char *path) {
    enum { LOAD_ITERS = 50 };
    char err[256];
    uint64_t t[LOAD_ITERS];
    for (int i = 0; i < LOAD_ITERS; i++) {
        bitnet_model m;
        const uint64_t t0 = now_ns();
        const int rc = bitnet_model_load(path, &m, err, sizeof(err));
        t[i] = now_ns() - t0;
        assert(rc == 0);
        bitnet_model_free(&m);
    }
    qsort(t, LOAD_ITERS, sizeof(t[0]), cmp_u64);

    const size_t rss0 = resident_bytes();
    bitnet_model m;
    assert(bitnet_model_load(path, &m, err, sizeof(err)) == 0);
    const size_t rss_loaded = resident_bytes();

    int max_m = 0, max_k = 0;
    size_t tern_bytes = 0;
    for (int i = 0; i < m.n_tensors; i++) {
        if (m.tensors[i].dtype != BITNET_DTYPE_TERNARY) continue;
        if (m.tensors[i].rows > max_m) max_m = m.tensors[i].rows;
        if (m.tensors[i].cols > max_k) max_k = m.tensors[i].cols;
        tern_bytes += m.tensors[i].nbytes;
    }
    int8_t *x = aligned_buf((size_t)max_k);
    int32_t *y = aligned_buf((size_t)max_m * sizeof(int32_t));
    for (int k = 0; k < max_k; k++) x[k] = (int8_t)(k * 37 + 11);

    const double first_ms = gemv_pass_ms(&m, x, y);
    const size_t rss_touched = resident_bytes();
    double warm_ms = 1e30;
    for (int r = 0; r < 5; r++) {
        const double ms = gemv_pass_ms(&m, x, y);
        if (ms < warm_ms) warm_ms = ms;
    }

    const uint64_t c0 = now_ns();
    assert(bitnet_model_verify_crc(&m, err, sizeof(err)) == 0);
    const double crc_s = (double)(now_ns() - c0) / 1e9;

    printf("\nBenchmark (%s, %.1f MiB)\n", path, m.map_size / 1048576.0);
    printf("  load + validate (mmap, zero-copy)   median %8.1f us   min %8.1f us\n",
           t[LOAD_ITERS / 2] / 1e3, t[0] / 1e3);
    printf("  resident memory after load           %+8.2f MiB (file is %.1f MiB: nothing copied)\n",
           ((double)rss_loaded - (double)rss0) / 1048576.0, m.map_size / 1048576.0);
    printf("  resident memory after GEMV pass      %+8.2f MiB (ternary weights %.1f MiB paged in)\n",
           ((double)rss_touched - (double)rss0) / 1048576.0, tern_bytes / 1048576.0);
    printf("  GEMV over all ternary tensors        first touch %8.2f ms   warm %8.2f ms\n",
           first_ms, warm_ms);
    printf("  payload CRC verify (all tensors)     %8.1f ms   (%.1f GB/s)\n",
           crc_s * 1e3, m.map_size / crc_s / 1e9);

    free(x);
    free(y);
    bitnet_model_free(&m);
}

/* ------------------------------------------------------------------------- */

int main(int argc, char **argv) {
    const char *model_path = NULL, *ref_path = NULL, *tmpdir = getenv("TMPDIR");
    int corrupt = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ref") == 0 && i + 1 < argc) ref_path = argv[++i];
        else if (strcmp(argv[i], "--corrupt-tests") == 0) corrupt = 1;
        else if (strcmp(argv[i], "--tmpdir") == 0 && i + 1 < argc) tmpdir = argv[++i];
        else if (argv[i][0] != '-' && !model_path) model_path = argv[i];
        else {
            fprintf(stderr, "usage: %s MODEL.bitnet [--ref MODEL.ref] [--corrupt-tests] "
                    "[--tmpdir DIR]\n", argv[0]);
            return 2;
        }
    }
    if (!model_path) {
        fprintf(stderr, "usage: %s MODEL.bitnet [--ref MODEL.ref] [--corrupt-tests]\n", argv[0]);
        return 2;
    }
    if (!tmpdir) tmpdir = "/tmp";

    bitnet_model m;
    char err[256];
    if (bitnet_model_load(model_path, &m, err, sizeof(err)) != 0) {
        fprintf(stderr, "load failed: %s\n", err);
        return 1;
    }
    print_summary(&m);
    if (bitnet_model_verify_crc(&m, err, sizeof(err)) != 0) {
        fprintf(stderr, "crc check failed: %s\n", err);
        bitnet_model_free(&m);
        return 1;
    }
    printf("Payload CRCs: PASSED (%d tensors)\n", m.n_tensors);
    if (ref_path) verify_reference(&m, ref_path);
    bitnet_model_free(&m);

    if (corrupt) run_corruption_tests(model_path, tmpdir);
    run_benchmark(model_path);
    return 0;
}
