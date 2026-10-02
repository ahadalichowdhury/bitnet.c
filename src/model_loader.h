/*
 * model_loader.h — Zero-copy, memory-mapped loader for `.bitnet` model files.
 *
 * Files are produced by tools/export_bitnet.py. The whole file is mmap'ed
 * read-only; every tensor pointer points straight into the mapping, so loading
 * costs O(header + tensor table) regardless of model size and weights are
 * paged in lazily by the kernel on first use.
 *
 * .bitnet format, version 2 (little-endian; version 1 files are still read):
 *   [0, 256)        bitnet_file_header (header_crc32 covers it, crc field zeroed)
 *   [table_offset)  n_tensors x bitnet_file_tensor (table_crc32)
 *   [data_offset)   tensor payloads, each 64-byte aligned, each with a crc32
 *
 * Tensor dtypes:
 *   F32 / F16   row-major
 *   TERNARY     ROW4 layout: rows x ceil(cols/4) bytes, weight k in byte k/4
 *               at bits 2*(k%4) (00 = 0, 01 = +1, 10 = -1); `scale` = beta.
 *   TERNARY_I128 (v2 only) same codes, SIMD-friendly: each row is ceil(cols/128)
 *               blocks of 32 bytes; byte j of a block holds weights j, 32+j,
 *               64+j, 96+j at bits 0, 2, 4, 6, so one shift + mask yields 32
 *               consecutive weights (ternary_dot.h). Written by default.
 *   Q8          (v2 only; embeddings / lm_head) rows x cols int8 followed by
 *               rows x cols/32 float32 block scales; cols % 32 == 0 (q8.h).
 *
 * The loader validates everything before exposing a pointer: magic, version,
 * CRCs of header and table, config sanity, every tensor's bounds, alignment,
 * size, dtype and shape against the config, duplicate names and overlapping
 * payloads. A file that passes load can be fed to the kernels without further
 * checks. Payload CRCs are optional (bitnet_model_verify_crc) because checking
 * them touches every page.
 */
#ifndef BITNET_MODEL_LOADER_H
#define BITNET_MODEL_LOADER_H

#include <stddef.h>
#include <stdint.h>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the .bitnet format is little-endian; big-endian hosts are not supported"
#endif

#define BITNET_MAGIC        "BITN" /* bytes 0x42 0x49 0x54 0x4E */
#define BITNET_VERSION      2u /* newest format written; v1 files are still read */
#define BITNET_VERSION_MIN  1u
#define BITNET_HEADER_SIZE  256u
#define BITNET_ENTRY_SIZE   96u
#define BITNET_NAME_LEN     48u
#define BITNET_ALIGN        64u
#define BITNET_MAX_K        (1 << 17) /* int32 -> float32 exactness limit (Step 3) */

enum {
    BITNET_FLAG_TIED_EMBEDDINGS = 1u << 0,
    BITNET_FLAG_SUB_NORMS       = 1u << 1,
    BITNET_KNOWN_FLAGS          = BITNET_FLAG_TIED_EMBEDDINGS | BITNET_FLAG_SUB_NORMS,
};

typedef enum {
    BITNET_DTYPE_F32     = 0,
    BITNET_DTYPE_F16     = 1,
    BITNET_DTYPE_TERNARY = 2,      /* ROW4 layout (v1 and v2 files) */
    BITNET_DTYPE_TERNARY_I128 = 3, /* SIMD-friendly I128 layout, ternary_dot.h (v2 only) */
    BITNET_DTYPE_Q8      = 4,      /* int8 + f32 scale per 32-block, q8.h (v2 only) */
} bitnet_dtype;

typedef enum {
    BITNET_ACT_SILU  = 0, /* SwiGLU:       silu(gate) * up */
    BITNET_ACT_RELU2 = 1, /* squared ReLU: relu(gate)^2 * up (bitnet-b1.58-2B-4T) */
} bitnet_ffn_act;

/* ---- On-disk structures ------------------------------------------------- */

typedef struct {
    char     magic[4];
    uint32_t version;
    uint32_t header_size;
    uint32_t flags;
    uint32_t vocab_size;
    uint32_t dim;
    uint32_t hidden_dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t max_seq_len;
    uint32_t ffn_act;
    float    norm_eps;
    float    rope_theta;
    uint32_t n_tensors;
    uint32_t entry_size;
    uint64_t table_offset;
    uint64_t data_offset;
    uint64_t file_size;
    uint32_t header_crc32;
    uint32_t table_crc32;
    uint8_t  reserved[160];
} bitnet_file_header;

typedef struct {
    char     name[BITNET_NAME_LEN];
    uint32_t dtype;
    uint32_t ndim;
    uint32_t rows;
    uint32_t cols;
    uint64_t offset;
    uint64_t nbytes;
    float    scale;
    uint32_t crc32;
    uint8_t  reserved[8];
} bitnet_file_tensor;

_Static_assert(sizeof(bitnet_file_header) == BITNET_HEADER_SIZE, "header size");
_Static_assert(sizeof(bitnet_file_tensor) == BITNET_ENTRY_SIZE, "tensor entry size");

/* ---- In-memory model ---------------------------------------------------- */

typedef struct {
    const char  *name;   /* points into the mapping (NUL-terminated, validated) */
    bitnet_dtype dtype;
    int          rows;   /* output features for linears; length for 1-D */
    int          cols;   /* input features for linears; 1 for 1-D */
    float        scale;  /* beta for TERNARY, 1.0 otherwise */
    const void  *data;   /* 64-byte aligned, inside the mapping */
    size_t       nbytes;
    uint32_t     crc32;
} bitnet_tensor;

typedef struct {
    const bitnet_tensor *attn_norm;     /* F32 [dim] */
    const bitnet_tensor *wq;            /* TERNARY [n_heads*head_dim, dim] */
    const bitnet_tensor *wk;            /* TERNARY [n_kv_heads*head_dim, dim] */
    const bitnet_tensor *wv;            /* TERNARY [n_kv_heads*head_dim, dim] */
    const bitnet_tensor *wo;            /* TERNARY [dim, n_heads*head_dim] */
    const bitnet_tensor *attn_sub_norm; /* F32 [dim], NULL unless FLAG_SUB_NORMS */
    const bitnet_tensor *ffn_norm;      /* F32 [dim] */
    const bitnet_tensor *w_gate;        /* TERNARY [hidden_dim, dim] */
    const bitnet_tensor *w_up;          /* TERNARY [hidden_dim, dim] */
    const bitnet_tensor *w_down;        /* TERNARY [dim, hidden_dim] */
    const bitnet_tensor *ffn_sub_norm;  /* F32 [hidden_dim], NULL unless FLAG_SUB_NORMS */
} bitnet_layer;

typedef struct {
    int            vocab_size;
    int            dim;
    int            hidden_dim;
    int            n_layers;
    int            n_heads;
    int            n_kv_heads;
    int            head_dim;
    int            max_seq_len;
    float          norm_eps;
    float          rope_theta;
    bitnet_ffn_act ffn_act;
    uint32_t       flags;
} bitnet_config;

typedef struct {
    bitnet_config        config;
    const bitnet_tensor *tok_embeddings; /* F16, F32 or Q8 [vocab, dim] */
    const bitnet_tensor *output_norm;    /* F32 [dim] */
    const bitnet_tensor *output;         /* == tok_embeddings when tied */
    bitnet_layer        *layers;         /* [n_layers] */
    bitnet_tensor       *tensors;        /* [n_tensors], file order */
    int                  n_tensors;
    const uint8_t       *map;
    size_t               map_size;
} bitnet_model;

/* Maps and validates `path`. Returns 0 on success; on failure returns -1,
 * leaves *model zeroed and writes a message to err (if non-NULL). */
int bitnet_model_load(const char *path, bitnet_model *model, char *err, size_t err_len);

/* Unmaps and frees; safe on a zeroed or already-freed model. */
void bitnet_model_free(bitnet_model *model);

/* Exact-name lookup; NULL if absent. */
const bitnet_tensor *bitnet_model_find(const bitnet_model *model, const char *name);

/* Checks every payload against its stored crc32 (touches all pages).
 * Returns 0 if all match, else -1 with the first bad tensor named in err. */
int bitnet_model_verify_crc(const bitnet_model *model, char *err, size_t err_len);

/* Hints the kernel to start paging the whole file in (madvise WILLNEED). */
void bitnet_model_prefetch(const bitnet_model *model);

/* Writes embedding row `token` as float32 into out[dim] (F16 widened exactly). */
void bitnet_embedding_row(const bitnet_model *model, int token, float *out);

/* zlib-compatible CRC-32: crc = bitnet_crc32(0, data, n), chainable. */
uint32_t bitnet_crc32(uint32_t crc, const void *data, size_t n);

#endif /* BITNET_MODEL_LOADER_H */
