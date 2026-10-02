/*
 * model_loader.c — Zero-copy mmap loader for `.bitnet` files (see model_loader.h).
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#include "model_loader.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "platform.h"
#include "simd.h"
#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

/* ------------------------------------------------------------------------- */
/* CRC-32 (zlib polynomial 0xEDB88320, reflected)                            */
/* ------------------------------------------------------------------------- */

#if !defined(__ARM_FEATURE_CRC32)
static uint32_t crc_table[256];
static int crc_table_ready;

static void crc_table_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        crc_table[i] = c;
    }
    crc_table_ready = 1;
}
#endif

uint32_t bitnet_crc32(uint32_t crc, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    crc = ~crc;
#if defined(__ARM_FEATURE_CRC32)
    /* ARMv8 CRC32 instructions implement exactly the zlib polynomial. */
    while (n && ((uintptr_t)p & 7)) { crc = __crc32b(crc, *p++); n--; }
    for (; n >= 32; n -= 32, p += 32) {
        uint64_t a, b, c, d;
        memcpy(&a, p, 8); memcpy(&b, p + 8, 8); memcpy(&c, p + 16, 8); memcpy(&d, p + 24, 8);
        crc = __crc32d(__crc32d(__crc32d(__crc32d(crc, a), b), c), d);
    }
    for (; n >= 8; n -= 8, p += 8) {
        uint64_t a;
        memcpy(&a, p, 8);
        crc = __crc32d(crc, a);
    }
    while (n--) crc = __crc32b(crc, *p++);
#else
    if (!crc_table_ready) crc_table_init();
    while (n--) crc = crc_table[(crc ^ *p++) & 0xFFu] ^ (crc >> 8);
#endif
    return ~crc;
}

/* ------------------------------------------------------------------------- */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------- */

static int fail(char *err, size_t err_len, const char *fmt, ...) {
    if (err && err_len) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, err_len, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static const char *dtype_name(uint32_t dt) {
    switch (dt) {
    case BITNET_DTYPE_F32:     return "f32";
    case BITNET_DTYPE_F16:     return "f16";
    case BITNET_DTYPE_TERNARY: return "ternary";
    }
    return "invalid";
}

/* Expected payload size; 0 on overflow or invalid dtype. */
static uint64_t expected_nbytes(uint32_t dtype, uint32_t rows, uint32_t cols) {
    const uint64_t r = rows, c = cols;
    switch (dtype) {
    case BITNET_DTYPE_F32:     return r * c * 4u;  /* rows, cols < 2^32: no overflow */
    case BITNET_DTYPE_F16:     return r * c * 2u;
    case BITNET_DTYPE_TERNARY: return r * ((c + 3u) / 4u);
    }
    return 0;
}

const bitnet_tensor *bitnet_model_find(const bitnet_model *model, const char *name) {
    for (int i = 0; i < model->n_tensors; i++)
        if (strcmp(model->tensors[i].name, name) == 0) return &model->tensors[i];
    return NULL;
}

static int cmp_by_offset(const void *a, const void *b) {
    const bitnet_tensor *x = *(const bitnet_tensor *const *)a;
    const bitnet_tensor *y = *(const bitnet_tensor *const *)b;
    return (x->data > y->data) - (x->data < y->data);
}

/* ------------------------------------------------------------------------- */
/* Validation                                                                */
/* ------------------------------------------------------------------------- */

static int validate_header(const bitnet_file_header *h, const uint8_t *map, size_t size,
                           char *err, size_t err_len) {
    if (memcmp(h->magic, BITNET_MAGIC, 4) != 0)
        return fail(err, err_len, "bad magic (not a .bitnet file)");
    if (h->version != BITNET_VERSION)
        return fail(err, err_len, "unsupported version %u (expected %u)", h->version, BITNET_VERSION);
    if (h->header_size != BITNET_HEADER_SIZE || h->entry_size != BITNET_ENTRY_SIZE)
        return fail(err, err_len, "unexpected header_size %u / entry_size %u",
                    h->header_size, h->entry_size);

    bitnet_file_header tmp = *h;
    tmp.header_crc32 = 0;
    const uint32_t crc = bitnet_crc32(0, &tmp, sizeof(tmp));
    if (crc != h->header_crc32)
        return fail(err, err_len, "header crc mismatch (stored %08x, computed %08x)",
                    h->header_crc32, crc);

    if (h->file_size != (uint64_t)size)
        return fail(err, err_len, "file size %zu does not match header (%llu): truncated or padded",
                    size, (unsigned long long)h->file_size);
    if (h->flags & ~(uint32_t)BITNET_KNOWN_FLAGS)
        return fail(err, err_len, "unknown flags 0x%x", h->flags & ~(uint32_t)BITNET_KNOWN_FLAGS);

    /* Tensor table bounds (all arithmetic in uint64, operands < 2^32). */
    if (h->n_tensors == 0 || h->n_tensors > (1u << 20))
        return fail(err, err_len, "implausible tensor count %u", h->n_tensors);
    if (h->table_offset < BITNET_HEADER_SIZE || h->table_offset % 8 != 0)
        return fail(err, err_len, "bad table offset %llu", (unsigned long long)h->table_offset);
    const uint64_t table_bytes = (uint64_t)h->n_tensors * BITNET_ENTRY_SIZE;
    if (h->table_offset > size || table_bytes > size - h->table_offset ||
        h->data_offset < h->table_offset + table_bytes || h->data_offset > size)
        return fail(err, err_len, "tensor table out of bounds");
    const uint32_t tcrc = bitnet_crc32(0, map + h->table_offset, (size_t)table_bytes);
    if (tcrc != h->table_crc32)
        return fail(err, err_len, "tensor table crc mismatch (stored %08x, computed %08x)",
                    h->table_crc32, tcrc);

    /* Model geometry. */
    if (h->vocab_size == 0 || h->vocab_size > (uint32_t)INT32_MAX ||
        h->dim == 0 || h->hidden_dim == 0 || h->n_layers == 0 || h->n_layers > 4096 ||
        h->n_heads == 0 || h->n_kv_heads == 0 || h->max_seq_len == 0 ||
        h->max_seq_len > (uint32_t)INT32_MAX)
        return fail(err, err_len, "config has zero or out-of-range dimensions");
    if (h->dim % h->n_heads != 0)
        return fail(err, err_len, "dim %u not divisible by n_heads %u", h->dim, h->n_heads);
    if (h->n_heads % h->n_kv_heads != 0)
        return fail(err, err_len, "n_heads %u not divisible by n_kv_heads %u",
                    h->n_heads, h->n_kv_heads);
    if (h->dim >= BITNET_MAX_K || h->hidden_dim >= BITNET_MAX_K)
        return fail(err, err_len, "dim/hidden_dim must be < %d", BITNET_MAX_K);
    if (h->ffn_act != BITNET_ACT_SILU && h->ffn_act != BITNET_ACT_RELU2)
        return fail(err, err_len, "unknown ffn activation %u", h->ffn_act);
    if (!isfinite(h->norm_eps) || h->norm_eps <= 0.0f ||
        !isfinite(h->rope_theta) || h->rope_theta <= 0.0f)
        return fail(err, err_len, "norm_eps / rope_theta must be finite and positive");
    return 0;
}

static int validate_tensor(const bitnet_file_tensor *e, int idx, const bitnet_file_header *h,
                           char *err, size_t err_len) {
    if (!memchr(e->name, '\0', BITNET_NAME_LEN) || e->name[0] == '\0')
        return fail(err, err_len, "tensor %d: name is empty or not NUL-terminated", idx);
    if (e->dtype > BITNET_DTYPE_TERNARY)
        return fail(err, err_len, "%s: invalid dtype %u", e->name, e->dtype);
    if (e->ndim < 1 || e->ndim > 2 || e->rows == 0 || e->cols == 0 ||
        (e->ndim == 1 && e->cols != 1) || e->rows > (uint32_t)INT32_MAX ||
        e->cols > (uint32_t)INT32_MAX)
        return fail(err, err_len, "%s: invalid shape (ndim %u, %u x %u)",
                    e->name, e->ndim, e->rows, e->cols);
    if (e->dtype == BITNET_DTYPE_TERNARY && e->ndim != 2)
        return fail(err, err_len, "%s: ternary tensors must be 2-D", e->name);
    if (e->offset % BITNET_ALIGN != 0)
        return fail(err, err_len, "%s: offset %llu not %u-byte aligned",
                    e->name, (unsigned long long)e->offset, BITNET_ALIGN);
    if (e->offset < h->data_offset || e->offset > h->file_size ||
        e->nbytes > h->file_size - e->offset)
        return fail(err, err_len, "%s: payload [%llu, +%llu) out of bounds", e->name,
                    (unsigned long long)e->offset, (unsigned long long)e->nbytes);
    const uint64_t want = expected_nbytes(e->dtype, e->rows, e->cols);
    if (e->nbytes != want)
        return fail(err, err_len, "%s: nbytes %llu does not match %s %u x %u (%llu)", e->name,
                    (unsigned long long)e->nbytes, dtype_name(e->dtype), e->rows, e->cols,
                    (unsigned long long)want);
    if (!isfinite(e->scale) || e->scale <= 0.0f)
        return fail(err, err_len, "%s: scale must be finite and positive", e->name);
    return 0;
}

/* Binds a named tensor with the expected dtype class and shape. `dtypes` is a
 * bitmask of acceptable bitnet_dtype values. */
static int bind(bitnet_model *m, const bitnet_tensor **slot, unsigned dtypes, int rows, int cols,
                int *bound, char *err, size_t err_len, const char *fmt, ...) {
    char name[BITNET_NAME_LEN + 16];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);

    const bitnet_tensor *t = bitnet_model_find(m, name);
    if (!t) return fail(err, err_len, "missing tensor %s", name);
    if (!(dtypes & (1u << t->dtype)))
        return fail(err, err_len, "%s: unexpected dtype %s", name, dtype_name(t->dtype));
    if (t->rows != rows || t->cols != cols)
        return fail(err, err_len, "%s: shape %d x %d, expected %d x %d",
                    name, t->rows, t->cols, rows, cols);
    *slot = t;
    (*bound)++;
    return 0;
}

#define DT(x) (1u << (x))

static int bind_model(bitnet_model *m, char *err, size_t err_len) {
    const bitnet_config *c = &m->config;
    const int d = c->dim, hd = c->hidden_dim;
    const int q_out = c->n_heads * c->head_dim, kv_out = c->n_kv_heads * c->head_dim;
    const unsigned F = DT(BITNET_DTYPE_F32), E = DT(BITNET_DTYPE_F32) | DT(BITNET_DTYPE_F16),
                   T = DT(BITNET_DTYPE_TERNARY);
    const int sub = (c->flags & BITNET_FLAG_SUB_NORMS) != 0;
    int bound = 0;

    if (bind(m, &m->tok_embeddings, E, c->vocab_size, d, &bound, err, err_len, "tok_embeddings") ||
        bind(m, &m->output_norm, F, d, 1, &bound, err, err_len, "output_norm"))
        return -1;

    if (c->flags & BITNET_FLAG_TIED_EMBEDDINGS) {
        if (bitnet_model_find(m, "output"))
            return fail(err, err_len, "tied embeddings but an 'output' tensor is present");
        m->output = m->tok_embeddings;
    } else if (bind(m, &m->output, E, c->vocab_size, d, &bound, err, err_len, "output")) {
        return -1;
    }

    for (int i = 0; i < c->n_layers; i++) {
        bitnet_layer *L = &m->layers[i];
        if (bind(m, &L->attn_norm, F, d, 1, &bound, err, err_len, "layers.%d.attn_norm", i) ||
            bind(m, &L->wq, T, q_out, d, &bound, err, err_len, "layers.%d.attn.wq", i) ||
            bind(m, &L->wk, T, kv_out, d, &bound, err, err_len, "layers.%d.attn.wk", i) ||
            bind(m, &L->wv, T, kv_out, d, &bound, err, err_len, "layers.%d.attn.wv", i) ||
            bind(m, &L->wo, T, d, q_out, &bound, err, err_len, "layers.%d.attn.wo", i) ||
            bind(m, &L->ffn_norm, F, d, 1, &bound, err, err_len, "layers.%d.ffn_norm", i) ||
            bind(m, &L->w_gate, T, hd, d, &bound, err, err_len, "layers.%d.ffn.w_gate", i) ||
            bind(m, &L->w_up, T, hd, d, &bound, err, err_len, "layers.%d.ffn.w_up", i) ||
            bind(m, &L->w_down, T, d, hd, &bound, err, err_len, "layers.%d.ffn.w_down", i))
            return -1;
        if (sub && (bind(m, &L->attn_sub_norm, F, d, 1, &bound, err, err_len,
                         "layers.%d.attn_sub_norm", i) ||
                    bind(m, &L->ffn_sub_norm, F, hd, 1, &bound, err, err_len,
                         "layers.%d.ffn_sub_norm", i)))
            return -1;
    }

    if (bound != m->n_tensors)
        return fail(err, err_len, "%d unrecognized tensors in file (e.g. sub-norms without "
                    "FLAG_SUB_NORMS)", m->n_tensors - bound);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Public API                                                                */
/* ------------------------------------------------------------------------- */

void bitnet_model_free(bitnet_model *model) {
    if (!model) return;
    if (model->map) munmap((void *)model->map, model->map_size);
    free(model->layers);
    free(model->tensors);
    memset(model, 0, sizeof(*model));
}

int bitnet_model_load(const char *path, bitnet_model *model, char *err, size_t err_len) {
    memset(model, 0, sizeof(*model));

    const int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(err, err_len, "open %s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) {
        const int e = errno;
        close(fd);
        return fail(err, err_len, "stat %s: %s", path, strerror(e));
    }
    if (!S_ISREG(st.st_mode) || st.st_size < (off_t)BITNET_HEADER_SIZE) {
        close(fd);
        return fail(err, err_len, "%s: not a regular file or smaller than the %u-byte header",
                    path, BITNET_HEADER_SIZE);
    }
    const size_t size = (size_t)st.st_size;
    /* MAP_SHARED + PROT_READ: on macOS a MAP_PRIVATE mapping of a 1.1 GiB file
     * costs 8-40 ms in mmap() itself (copy-on-write setup); a read-only shared
     * mapping costs ~1 us. We never write, so the two are equivalent except
     * that in-place edits to the file would become visible — the exporter
     * replaces files atomically (write + rename), so a mapped model never
     * changes underneath us. */
    void *map = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd); /* the mapping keeps the file referenced */
    if (map == MAP_FAILED) return fail(err, err_len, "mmap %s: %s", path, strerror(errno));
    model->map = (const uint8_t *)map;
    model->map_size = size;

    bitnet_file_header h;
    memcpy(&h, map, sizeof(h));
    if (validate_header(&h, model->map, size, err, err_len) != 0) goto fail;

    bitnet_config *c = &model->config;
    c->vocab_size  = (int)h.vocab_size;
    c->dim         = (int)h.dim;
    c->hidden_dim  = (int)h.hidden_dim;
    c->n_layers    = (int)h.n_layers;
    c->n_heads     = (int)h.n_heads;
    c->n_kv_heads  = (int)h.n_kv_heads;
    c->head_dim    = (int)(h.dim / h.n_heads);
    c->max_seq_len = (int)h.max_seq_len;
    c->norm_eps    = h.norm_eps;
    c->rope_theta  = h.rope_theta;
    c->ffn_act     = (bitnet_ffn_act)h.ffn_act;
    c->flags       = h.flags;

    model->n_tensors = (int)h.n_tensors;
    model->tensors = calloc((size_t)h.n_tensors, sizeof(bitnet_tensor));
    model->layers = calloc((size_t)h.n_layers, sizeof(bitnet_layer));
    if (!model->tensors || !model->layers) {
        fail(err, err_len, "out of memory");
        goto fail;
    }

    const bitnet_file_tensor *entries =
        (const bitnet_file_tensor *)(model->map + h.table_offset); /* 8-byte aligned */
    for (int i = 0; i < model->n_tensors; i++) {
        const bitnet_file_tensor *e = &entries[i];
        if (validate_tensor(e, i, &h, err, err_len) != 0) goto fail;
        model->tensors[i] = (bitnet_tensor){
            .name = e->name, .dtype = (bitnet_dtype)e->dtype, .rows = (int)e->rows,
            .cols = (int)e->cols, .scale = e->scale, .data = model->map + e->offset,
            .nbytes = (size_t)e->nbytes, .crc32 = e->crc32,
        };
    }

    /* Duplicate names and overlapping payloads. */
    {
        const bitnet_tensor **order = malloc((size_t)model->n_tensors * sizeof(*order));
        if (!order) {
            fail(err, err_len, "out of memory");
            goto fail;
        }
        for (int i = 0; i < model->n_tensors; i++) order[i] = &model->tensors[i];
        qsort(order, (size_t)model->n_tensors, sizeof(*order), cmp_by_offset);
        int bad = 0;
        for (int i = 1; i < model->n_tensors && !bad; i++) {
            const bitnet_tensor *a = order[i - 1], *b = order[i];
            if ((const uint8_t *)a->data + a->nbytes > (const uint8_t *)b->data) {
                fail(err, err_len, "tensors %s and %s overlap", a->name, b->name);
                bad = 1;
            }
        }
        free(order);
        if (bad) goto fail;
        for (int i = 0; i < model->n_tensors; i++)
            for (int j = i + 1; j < model->n_tensors; j++)
                if (strcmp(model->tensors[i].name, model->tensors[j].name) == 0) {
                    fail(err, err_len, "duplicate tensor name %s", model->tensors[i].name);
                    goto fail;
                }
    }

    if (bind_model(model, err, err_len) != 0) goto fail;
    return 0;

fail:
    bitnet_model_free(model);
    return -1;
}

int bitnet_model_verify_crc(const bitnet_model *model, char *err, size_t err_len) {
    for (int i = 0; i < model->n_tensors; i++) {
        const bitnet_tensor *t = &model->tensors[i];
        const uint32_t crc = bitnet_crc32(0, t->data, t->nbytes);
        if (crc != t->crc32)
            return fail(err, err_len, "%s: payload crc mismatch (stored %08x, computed %08x)",
                        t->name, t->crc32, crc);
    }
    return 0;
}

void bitnet_model_prefetch(const bitnet_model *model) {
    if (model->map) posix_madvise((void *)model->map, model->map_size, POSIX_MADV_WILLNEED);
}

void bitnet_embedding_row(const bitnet_model *model, int token, float *out) {
    const bitnet_tensor *t = model->tok_embeddings;
    const int dim = t->cols;
    if (t->dtype == BITNET_DTYPE_F32) {
        memcpy(out, (const float *)t->data + (size_t)token * dim, (size_t)dim * sizeof(float));
        return;
    }
    const uint16_t *src = (const uint16_t *)t->data + (size_t)token * dim;
    int i = 0;
#ifdef BITNET_NEON
    for (; i + 8 <= dim; i += 8) {
        const uint16x8_t h = vld1q_u16(src + i);
        vst1q_f32(out + i, vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))));
        vst1q_f32(out + i + 4, vcvt_high_f32_f16(vreinterpretq_f16_u16(h)));
    }
#endif
    for (; i < dim; i++) out[i] = half_to_float(src[i]); /* exact, like the NEON path */
}
