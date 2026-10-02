/*
 * transformer.h — BitNet b1.58 transformer forward pass (bitnet-b1.58-2B-4T).
 *
 * Mirrors HF transformers `modeling_bitnet.py` + `AutoBitLinear` exactly:
 *
 *   x  = embed[token]
 *   per layer:
 *     h  = RMSNorm(x, attn_norm)
 *     q,k,v = BitLinear(h)                      (one shared int8 activation quant)
 *     q,k   = RoPE(q,k, pos)                    (rotate_half layout: i <-> i + d/2)
 *     a  = softmax(q k^T / sqrt(d)) v           (causal, GQA: kv head = head / group)
 *     x += BitLinear_o(RMSNorm(a, attn_sub_norm))
 *     h  = RMSNorm(x, ffn_norm)
 *     g,u = BitLinear_gate(h), BitLinear_up(h)
 *     x += BitLinear_down(RMSNorm(act(g) * u, ffn_sub_norm))   act = relu^2 | silu
 *   logits = embed^T RMSNorm(x, output_norm)   (tied, f16 weights, f32 accumulate)
 *
 *   BitLinear(x) = (W_q . quant(x)) * gamma/127 * beta   (bitlinear.h + gemv.h)
 *
 * All activations are float32. RunState owns every buffer, including a static
 * KV cache sized for max_seq_len and RoPE tables, in one anonymous mapping,
 * plus a persistent worker pool; transformer_forward performs no heap
 * allocation on any thread (GCD is not used: dispatch_apply calloc()s).
 */
#ifndef BITNET_TRANSFORMER_H
#define BITNET_TRANSFORMER_H

#include <stddef.h>
#include <stdint.h>

#include "model_loader.h"
#include "threadpool.h"

typedef bitnet_model BitNetModel;

/* Per-stage wall time accumulated by transformer_forward (nanoseconds). */
typedef struct {
    uint64_t embed, attn_proj, attention, ffn, logits, total;
    uint64_t tokens;
} transformer_profile;

typedef struct {
    int max_seq_len; /* KV-cache capacity in positions */

    float   *x;      /* [dim]          residual stream */
    float   *xb;     /* [dim]          normed input / sub-layer output */
    float   *xb2;    /* [q_dim]        attention output (per head, concatenated) */
    float   *q;      /* [q_dim]        query for the current position */
    float   *attn_part; /* [n_heads][max_chunks][head_dim + 2] split-softmax partials */
    float   *hb;     /* [hidden_dim]   gate activations */
    float   *hb2;    /* [hidden_dim]   up activations */
    float   *logits; /* [vocab_size] */
    int8_t  *xq;     /* [max(dim, hidden_dim, q_dim)] quantized activations */
    int32_t *yi;     /* [q_dim + 2*kv_dim + 2*hidden_dim] int32 GEMV outputs */

    float   *key_cache;   /* [n_layers][max_seq_len][kv_dim] */
    float   *value_cache; /* [n_layers][max_seq_len][kv_dim] */
    float   *rope_cos;    /* [max_seq_len][head_dim/2] */
    float   *rope_sin;    /* [max_seq_len][head_dim/2] */

    threadpool *pool;     /* persistent workers shared by all parallel stages */
    transformer_profile prof;
    int      profile;     /* nonzero: accumulate prof (adds a few clock reads) */

    void    *arena;       /* single anonymous mapping backing all of the above */
    size_t   arena_bytes;
} RunState;

/* Allocates all buffers for `model` with a KV cache of max_seq_len positions
 * (0 = the model's max_seq_len) and a pool of n_threads threads including the
 * caller (0 = all CPUs). Returns 0, or -1 with a message in err. */
int runstate_init(RunState *s, const BitNetModel *model, int max_seq_len, int n_threads,
                  char *err, size_t err_len);
void runstate_free(RunState *s);

/* Runs one token at position pos (0 <= pos < s->max_seq_len), writing K/V for
 * pos into the cache and next-token logits into s->logits. Positions must be
 * fed in order; feeding pos again overwrites that cache slot (rewind). */
void transformer_forward(int token_id, int pos, const BitNetModel *model, RunState *s);

/* Same, but skips the final norm + output projection when compute_logits is 0
 * (s->logits is left untouched). Used for prompt tokens whose logits are not
 * needed: the 128256 x 2560 output layer is ~40% of a forward pass. */
void transformer_forward_ex(int token_id, int pos, const BitNetModel *model, RunState *s,
                            int compute_logits);

/* ---- Building blocks (exposed for unit tests) --------------------------- */

/* out = x / sqrt(mean(x^2) + eps) * w   (out may alias x) */
void rmsnorm_neon(float *out, const float *x, const float *w, int n, float eps);

/* In-place RoPE on n_heads consecutive heads of head_dim (rotate_half layout).
 * cos/sin hold head_dim/2 values for the current position. */
void apply_rope_neon(float *vec, int n_heads, int head_dim, const float *cos, const float *sin);

/* In-place numerically stable softmax over n values. */
void softmax_neon(float *x, int n);

/* x += y */
void residual_add_neon(float *x, const float *y, int n);

/* gate = act(gate) * up, act = relu(x)^2 (relu2 = 1) or x*sigmoid(x) (relu2 = 0) */
void glu_neon(float *gate, const float *up, int n, int relu2);

/* Vectorized exp (max rel. error ~2 ulp on [-87, 88]); out may alias x. */
void exp_neon(float *out, const float *x, int n);

/* out[m] = sum_k f16(W[m][k]) * x[k] (tied output layer); pool may be NULL. */
void gemv_f16_neon(const uint16_t *W, const float *x, float *out, int M, int K, threadpool *pool);

/* Positions per split-softmax chunk in attention_neon. */
#ifndef ATTN_CHUNK
#define ATTN_CHUNK 256
#endif
#define ATTN_MAX_GROUP 8 /* max n_heads / n_kv_heads */

/* Causal GQA attention of one query position over cached positions 0..pos.
 *   q:        [n_heads * head_dim] (RoPE applied)
 *   k_cache, v_cache: this layer's cache, position t at offset t * kv_dim
 *   out:      [n_heads * head_dim]
 *   part:     scratch of attention_scratch_floats(...) floats
 * Work items are (kv head, chunk of ATTN_CHUNK positions): each K/V row is
 * read once for all query heads of its group, and chunk results are merged
 * with the log-sum-exp rule (flash-decoding). pool may be NULL. */
void attention_neon(float *out, const float *q, const float *k_cache, const float *v_cache, int pos,
                    int n_heads, int n_kv_heads, int head_dim, int kv_dim, float *part, threadpool *pool);
size_t attention_scratch_floats(int n_heads, int head_dim, int max_seq_len);

#endif /* BITNET_TRANSFORMER_H */
