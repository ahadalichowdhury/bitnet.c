/*
 * bitnet.h — Public C11 API of the BitNet b1.58 inference engine.
 *
 * The only header an application needs. Everything else (kernels, loader,
 * tokenizer, transformer, sampler) stays behind the opaque BitNetContext.
 *
 *     BitNetContext *ctx = bitnet_init("models/bitnet_2b4t.bitnet",
 *                                      "models/hf/bitnet-b1.58-2B-4T/tokenizer.json",
 *                                      bitnet_default_config());
 *     if (!ctx) { fprintf(stderr, "%s\n", bitnet_last_error(NULL)); return 1; }
 *     bitnet_chat_turn(ctx, "Hello!", bitnet_default_params(), print_token, NULL);
 *     bitnet_free(ctx);
 *
 * Callbacks receive NUL-terminated pieces of text that are always complete,
 * valid UTF-8 (multi-byte characters split across tokens are reassembled).
 *
 * Threading: a context is not thread-safe; use one context per thread (they
 * share nothing). bitnet_cancel is the exception: it may be called from any
 * thread or from a signal handler.
 *
 * Build: link src/bitnet.c with the engine sources (see Makefile: libbitnet.a).
 */
#ifndef BITNET_H
#define BITNET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BITNET_VERSION_STRING "1.0.0"

typedef struct BitNetContext BitNetContext;

/* Engine configuration (fixed for the lifetime of a context). */
typedef struct {
    int         n_threads;     /* worker threads incl. caller; 0 = all CPUs */
    int         max_seq_len;   /* KV-cache positions; 0 = model maximum (4096) */
    const char *system_prompt; /* chat system prompt (copied); NULL = none */
} BitNetConfig;

/* Per-call sampling parameters. */
typedef struct {
    float    temperature;    /* 0 = greedy */
    int      top_k;          /* 0 = disabled */
    float    top_p;          /* 1 = disabled */
    uint64_t seed;           /* 0 = pick a fresh random seed per call */
    int      max_new_tokens; /* 0 = until end of turn or end of context */
} BitNetSampleParams;

typedef void (*bitnet_token_fn)(const char *token, void *user_data);

typedef enum {
    BITNET_STOP_END_OF_TURN = 0, /* model emitted <|eot_id|> / <|end_of_text|> */
    BITNET_STOP_MAX_TOKENS,
    BITNET_STOP_CONTEXT_FULL,
    BITNET_STOP_CANCELLED,       /* bitnet_cancel() */
    BITNET_STOP_ERROR,           /* see bitnet_last_error(ctx) */
} BitNetStopReason;

/* Statistics of the most recent bitnet_generate / bitnet_chat_turn. */
typedef struct {
    BitNetStopReason stop_reason;
    int              prompt_tokens;    /* tokens prefilled by this call */
    int              generated_tokens;
    int              context_used;     /* KV-cache positions in use afterwards */
    int              context_size;
    int              context_reset;    /* 1 if the chat history was dropped to fit */
    double           prefill_ms;
    double           ttft_ms;          /* call start -> first token delivered */
    double           decode_ms;
    double           prefill_tok_s;
    double           decode_tok_s;
    uint64_t         seed;             /* seed actually used */
} BitNetStats;

/* Memory footprint of a context. */
typedef struct {
    size_t model_mapped_bytes;   /* weights file, memory-mapped (shared, read-only) */
    size_t runstate_bytes;       /* activations + KV cache, reserved */
    size_t kv_cache_bytes;       /* part of runstate_bytes */
    size_t process_resident_bytes;
    size_t process_peak_resident_bytes;
} BitNetMemoryInfo;

BitNetConfig       bitnet_default_config(void);
BitNetSampleParams bitnet_default_params(void); /* model card: temp 0.6, top-p 0.9 */

/* Loads the model (mmap, zero-copy) and tokenizer and allocates all buffers.
 * Returns NULL on failure; bitnet_last_error(NULL) then explains why. */
BitNetContext *bitnet_init(const char *model_path, const char *tokenizer_path, BitNetConfig config);
void bitnet_free(BitNetContext *ctx); /* NULL is allowed */

/* Plain text completion of `prompt` (BOS + prompt, no chat template). Starts
 * from an empty KV cache, so it also discards the chat history. */
void bitnet_generate(BitNetContext *ctx, const char *prompt, BitNetSampleParams params,
                     bitnet_token_fn on_token, void *user_data);

/* One chat turn: appends "User: {user_msg}" to the conversation, streams the
 * assistant's reply. The KV cache is reused across turns, so only the new
 * message is prefilled. If the conversation no longer fits, it is restarted
 * with this message (stats.context_reset = 1). */
void bitnet_chat_turn(BitNetContext *ctx, const char *user_msg, BitNetSampleParams params,
                      bitnet_token_fn on_token, void *user_data);

/* Forgets the conversation (the system prompt is kept). */
void bitnet_reset_chat(BitNetContext *ctx);

/* Replaces the system prompt (copied; NULL = none) and resets the chat.
 * Returns 0, or -1 if out of memory. */
int bitnet_set_system_prompt(BitNetContext *ctx, const char *system_prompt);

/* Requests the running generation to stop after the current token.
 * Async-signal-safe and thread-safe. A no-op when nothing is running. */
void bitnet_cancel(BitNetContext *ctx);

/* Results of the last call. Returns 0, or -1 if ctx is NULL. */
int bitnet_last_stats(const BitNetContext *ctx, BitNetStats *out);
int bitnet_memory_info(const BitNetContext *ctx, BitNetMemoryInfo *out);

/* Human-readable model summary (e.g. "BitNet b1.58 2B: 30 layers, ..."). */
const char *bitnet_model_description(const BitNetContext *ctx);

/* Last error message for ctx, or of the last failed bitnet_init on this
 * thread when ctx is NULL. Never NULL ("" when there is no error). */
const char *bitnet_last_error(const BitNetContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BITNET_H */
