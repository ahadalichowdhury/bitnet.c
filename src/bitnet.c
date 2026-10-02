/*
 * bitnet.c — Public API implementation (see bitnet.h).
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#include "bitnet.h"

#include <limits.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "generate.h"
#include "model_loader.h"
#include "platform.h"
#include "tokenizer.h"
#include "transformer.h"

#define PIECE_MAX 1024

_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "cancel flag must be lock-free to be async-signal-safe");

struct BitNetContext {
    bitnet_model model;
    Tokenizer   *tok;
    RunState     rs;
    Sampler      sampler;
    char        *system;
    int32_t      stop[2]; /* <|end_of_text|>, <|eot_id|> */
    int32_t      eot;

    int32_t     *prompt;   /* grows on demand (outside the generation loop) */
    int          prompt_cap;

    /* Conversation state: positions [0, chat_pos) of the KV cache hold the
     * chat so far; open_token is a generated token not yet in the cache
     * (reply cut short by max tokens / cancel), -1 if none. */
    int          chat_pos;
    int32_t      open_token;

    _Atomic int           cancel; /* lock-free: safe from signal handlers and threads */
    bitnet_token_fn       cb;
    void                 *ud;
    char                  piece[PIECE_MAX + 1];
    uint64_t              seed_counter;

    BitNetStats  stats;
    char         err[256];
    char         desc[256];
};

static _Thread_local char g_init_err[256];

static void set_err(char *buf, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, 256, fmt, ap);
    va_end(ap);
}

BitNetConfig bitnet_default_config(void) {
    return (BitNetConfig){0, 0, "You are a helpful AI assistant."};
}

BitNetSampleParams bitnet_default_params(void) {
    return (BitNetSampleParams){.temperature = 0.6f, .top_p = 0.9f, .repetition_penalty = 1.0f};
}

/* ------------------------------------------------------------------------- */
/* Lifecycle                                                                 */
/* ------------------------------------------------------------------------- */

BitNetContext *bitnet_init(const char *model_path, const char *tokenizer_path, BitNetConfig cfg) {
    g_init_err[0] = '\0';
    if (!model_path || !tokenizer_path) {
        set_err(g_init_err, "model_path and tokenizer_path are required");
        return NULL;
    }
    BitNetContext *c = calloc(1, sizeof(*c));
    if (!c) {
        set_err(g_init_err, "out of memory");
        return NULL;
    }
    char err[256];
    if (bitnet_model_load(model_path, &c->model, err, sizeof(err)) != 0) {
        set_err(g_init_err, "model %s: %s", model_path, err);
        free(c);
        return NULL;
    }
    c->tok = tokenizer_load(tokenizer_path, err, sizeof(err));
    if (!c->tok) {
        set_err(g_init_err, "tokenizer %s: %s", tokenizer_path, err);
        bitnet_model_free(&c->model);
        free(c);
        return NULL;
    }
    if (tokenizer_vocab_size(c->tok) != c->model.config.vocab_size) {
        set_err(g_init_err, "tokenizer vocabulary (%d) does not match the model (%d)",
                tokenizer_vocab_size(c->tok), c->model.config.vocab_size);
        bitnet_free(c);
        return NULL;
    }
    c->stop[0] = tokenizer_token_id(c->tok, "<|end_of_text|>");
    c->stop[1] = c->eot = tokenizer_token_id(c->tok, "<|eot_id|>");
    if (c->stop[0] < 0 || c->eot < 0 || tokenizer_token_id(c->tok, "<|begin_of_text|>") < 0) {
        set_err(g_init_err, "tokenizer lacks the Llama-3 special tokens");
        bitnet_free(c);
        return NULL;
    }
    if (runstate_init(&c->rs, &c->model, cfg.max_seq_len, cfg.n_threads, err, sizeof(err)) != 0) {
        set_err(g_init_err, "run state: %s", err);
        bitnet_free(c);
        return NULL;
    }
    const SamplerConfig sc = {.temperature = 0.6f, .top_p = 0.9f, .seed = 1, .repetition_penalty = 1.0f};
    if (sampler_init(&c->sampler, c->model.config.vocab_size, &sc, err, sizeof(err)) != 0) {
        set_err(g_init_err, "sampler: %s", err);
        bitnet_free(c);
        return NULL;
    }
    c->prompt_cap = 1024;
    c->prompt = malloc((size_t)c->prompt_cap * sizeof(int32_t));
    if (!c->prompt || bitnet_set_system_prompt(c, cfg.system_prompt) != 0) {
        set_err(g_init_err, "out of memory");
        bitnet_free(c);
        return NULL;
    }
    const bitnet_config *mc = &c->model.config;
    /* Parameter count for the description (ternary + embeddings). */
    double params = (double)mc->vocab_size * mc->dim;
    for (int i = 0; i < c->model.n_tensors; i++)
        if (c->model.tensors[i].dtype == BITNET_DTYPE_TERNARY ||
            c->model.tensors[i].dtype == BITNET_DTYPE_TERNARY_I128)
            params += (double)c->model.tensors[i].rows * c->model.tensors[i].cols;
    snprintf(c->desc, sizeof(c->desc),
             "BitNet b1.58 %.1fB: %d layers, dim %d, hidden %d, %d/%d heads, vocab %d, context %d, "
             "%d threads", params / 1e9, mc->n_layers, mc->dim, mc->hidden_dim, mc->n_heads,
             mc->n_kv_heads, mc->vocab_size, c->rs.max_seq_len, threadpool_size(c->rs.pool));
    c->open_token = -1;
    c->stats.context_size = c->rs.max_seq_len;
    return c;
}

void bitnet_free(BitNetContext *c) {
    if (!c) return;
    sampler_free(&c->sampler);
    if (c->rs.arena) runstate_free(&c->rs);
    tokenizer_free(c->tok);
    bitnet_model_free(&c->model);
    free(c->system);
    free(c->prompt);
    free(c);
}

int bitnet_set_system_prompt(BitNetContext *c, const char *system) {
    char *copy = NULL;
    if (system && !(copy = strdup(system))) return -1;
    free(c->system);
    c->system = copy;
    bitnet_reset_chat(c);
    return 0;
}

void bitnet_reset_chat(BitNetContext *c) {
    if (!c) return;
    c->chat_pos = 0;
    c->open_token = -1;
}

void bitnet_cancel(BitNetContext *c) {
    if (c) atomic_store_explicit(&c->cancel, 1, memory_order_relaxed); /* async-signal-safe */
}

/* ------------------------------------------------------------------------- */
/* Generation plumbing                                                       */
/* ------------------------------------------------------------------------- */

/* Utf8Stream sink -> user callback with NUL-terminated pieces, split only on
 * character boundaries. */
static void deliver(void *user, const char *s, size_t n) {
    BitNetContext *c = user;
    while (n) {
        size_t k = n < PIECE_MAX ? n : PIECE_MAX;
        if (k < n)
            while (k > 0 && ((unsigned char)s[k] & 0xC0) == 0x80) k--;
        memcpy(c->piece, s, k);
        c->piece[k] = '\0';
        c->cb(c->piece, c->ud);
        s += k;
        n -= k;
    }
}

static uint64_t fresh_seed(BitNetContext *c) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t z = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec +
                 0x9E3779B97F4A7C15ull * ++c->seed_counter;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return z ? z : 1;
}

/* Clamps params into the sampler and resets per-call state. */
static void begin_call(BitNetContext *c, BitNetSampleParams p, bitnet_token_fn cb, void *ud) {
    SamplerConfig sc;
    sc.temperature = p.temperature > 0.0f ? p.temperature : 0.0f;
    sc.top_k = p.top_k > 0 ? p.top_k : 0;
    sc.top_p = (p.top_p > 0.0f && p.top_p < 1.0f) ? p.top_p : 1.0f;
    sc.seed = p.seed ? p.seed : fresh_seed(c);
    sc.repetition_penalty = p.repetition_penalty > 0.0f ? p.repetition_penalty : 1.0f;
    sc.frequency_penalty = p.frequency_penalty;
    sc.presence_penalty = p.presence_penalty;
    sc.penalty_last_n = p.penalty_last_n > SAMPLER_HISTORY ? SAMPLER_HISTORY : p.penalty_last_n;
    c->sampler.cfg = sc;
    sampler_reseed(&c->sampler, sc.seed);
    c->cb = cb;
    c->ud = ud;
    atomic_store_explicit(&c->cancel, 0, memory_order_relaxed);
    c->err[0] = '\0';
    memset(&c->stats, 0, sizeof(c->stats));
    c->stats.context_size = c->rs.max_seq_len;
    c->stats.seed = sc.seed;
}

static void fail_call(BitNetContext *c, const char *msg) {
    set_err(c->err, "%s", msg);
    c->stats.stop_reason = BITNET_STOP_ERROR;
    c->stats.context_used = c->chat_pos;
}

/* Checks params.stop; sets the error and returns -1 if invalid. */
static int check_stops(BitNetContext *c, const BitNetSampleParams *p) {
    if (p->n_stop <= 0) return 0;
    if (p->n_stop > BITNET_MAX_STOP_STRINGS || !p->stop) {
        fail_call(c, "too many stop strings (max 8)");
        return -1;
    }
    for (int i = 0; i < p->n_stop; i++) {
        const size_t l = p->stop[i] ? strlen(p->stop[i]) : 0;
        if (l == 0 || l > STOP_MAX_LEN) {
            fail_call(c, "stop strings must be 1..64 bytes");
            return -1;
        }
    }
    return 0;
}

static int ensure_prompt_cap(BitNetContext *c, int need) {
    if (need <= c->prompt_cap) return 0;
    int32_t *p = realloc(c->prompt, (size_t)need * sizeof(int32_t));
    if (!p) return -1;
    c->prompt = p;
    c->prompt_cap = need;
    return 0;
}

static BitNetStopReason map_reason(gen_stop_reason r) {
    switch (r) {
    case GEN_STOP_EOS:          return BITNET_STOP_END_OF_TURN;
    case GEN_STOP_MAX_TOKENS:   return BITNET_STOP_MAX_TOKENS;
    case GEN_STOP_CONTEXT_FULL: return BITNET_STOP_CONTEXT_FULL;
    case GEN_STOP_CANCELLED:    return BITNET_STOP_CANCELLED;
    case GEN_STOP_STRING:       return BITNET_STOP_STOP_STRING;
    }
    return BITNET_STOP_ERROR;
}

static int run_generation(BitNetContext *c, int n_prompt, int start_pos, const BitNetSampleParams *p,
                          GenerateStats *st) {
    const GenerateParams gp = {
        .max_new_tokens = p->max_new_tokens > 0 ? p->max_new_tokens : INT_MAX,
        .stop_tokens = c->stop, .n_stop = 2,
        .on_text = c->cb ? deliver : NULL, .user = c,
        .cancel = &c->cancel,
        .stop_strings = p->n_stop > 0 ? p->stop : NULL, .n_stop_strings = p->n_stop > 0 ? p->n_stop : 0,
    };
    if (generate(&c->model, c->tok, &c->rs, &c->sampler, c->prompt, n_prompt, start_pos, &gp, st) != 0)
        return -1;
    BitNetStats *o = &c->stats;
    o->stop_reason = map_reason(st->reason);
    o->prompt_tokens = n_prompt;
    o->generated_tokens = st->n_generated;
    o->prefill_ms = st->prefill_ms;
    o->ttft_ms = st->ttft_ms;
    o->decode_ms = st->decode_ms;
    o->prefill_tok_s = st->prefill_tok_s;
    o->decode_tok_s = st->decode_tok_s;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Public generation entry points                                            */
/* ------------------------------------------------------------------------- */

void bitnet_generate(BitNetContext *c, const char *prompt, BitNetSampleParams p, bitnet_token_fn cb,
                     void *ud) {
    if (!c) return;
    begin_call(c, p, cb, ud);
    bitnet_reset_chat(c); /* the completion overwrites the KV cache from position 0 */
    if (check_stops(c, &p) != 0) return;
    if (!prompt) {
        fail_call(c, "prompt is NULL");
        return;
    }
    const size_t len = strlen(prompt);
    const int need = bpe_encode_ex(c->tok, prompt, len, NULL, 0, TOKENIZER_NO_SPECIAL);
    if (need < 0 || ensure_prompt_cap(c, need + 1) != 0) {
        fail_call(c, "cannot encode prompt");
        return;
    }
    c->prompt[0] = tokenizer_token_id(c->tok, "<|begin_of_text|>");
    bpe_encode_ex(c->tok, prompt, len, c->prompt + 1, need, TOKENIZER_NO_SPECIAL);
    const int n = need + 1;
    if (n >= c->rs.max_seq_len) {
        fail_call(c, "prompt does not fit in the context window");
        return;
    }
    GenerateStats st;
    if (run_generation(c, n, 0, &p, &st) != 0) {
        fail_call(c, "generation failed");
        return;
    }
    c->stats.context_used = st.end_pos;
}

/* Builds the prompt for the next chat turn into c->prompt; returns its length
 * or -1. continuing: close the previous assistant turn first. */
static int build_turn(BitNetContext *c, const char *msg, int continuing) {
    for (;;) {
        int n = 0;
        if (continuing) {
            if (c->open_token >= 0) c->prompt[n++] = c->open_token;
            c->prompt[n++] = c->eot;
        }
        const int k = build_chat_prompt(c->tok, continuing ? NULL : c->system, msg, !continuing,
                                        c->prompt + n, c->prompt_cap - n);
        if (k < 0) return -1;
        if (n + k <= c->prompt_cap) return n + k;
        if (ensure_prompt_cap(c, n + k + 16) != 0) return -1;
    }
}

void bitnet_chat_turn(BitNetContext *c, const char *msg, BitNetSampleParams p, bitnet_token_fn cb,
                      void *ud) {
    if (!c) return;
    begin_call(c, p, cb, ud);
    if (check_stops(c, &p) != 0) return;
    if (!msg) {
        fail_call(c, "message is NULL");
        return;
    }
    int continuing = c->chat_pos > 0;
    int n = build_turn(c, msg, continuing);
    /* Need room for the prompt plus at least one generated token. */
    if (n >= 0 && continuing && c->chat_pos + n >= c->rs.max_seq_len) {
        continuing = 0;
        c->stats.context_reset = 1;
        n = build_turn(c, msg, 0);
    }
    if (n < 0) {
        fail_call(c, "cannot encode message");
        return;
    }
    if (n >= c->rs.max_seq_len) {
        fail_call(c, "message does not fit in the context window");
        return;
    }
    const int start = continuing ? c->chat_pos : 0;
    GenerateStats st;
    if (run_generation(c, n, start, &p, &st) != 0) {
        fail_call(c, "generation failed");
        return;
    }
    if (st.prefill_done) {
        /* The turn is part of the conversation. A reply that ended without a
         * stop token leaves its last token outside the cache. */
        c->chat_pos = st.end_pos;
        /* A stop string ends the turn like a stop token: its completing
         * token is dropped rather than fed. */
        c->open_token = (st.reason == GEN_STOP_EOS || st.reason == GEN_STOP_STRING) ? -1 : st.last_token;
    } else if (c->stats.context_reset) {
        bitnet_reset_chat(c); /* the old history was already being overwritten */
    }
    /* else: cancelled during the prompt -> the conversation is unchanged. */
    c->stats.context_used = c->chat_pos;
}

/* ------------------------------------------------------------------------- */
/* Introspection                                                             */
/* ------------------------------------------------------------------------- */

int bitnet_last_stats(const BitNetContext *c, BitNetStats *out) {
    if (!c || !out) return -1;
    *out = c->stats;
    return 0;
}

int bitnet_memory_info(const BitNetContext *c, BitNetMemoryInfo *out) {
    if (!c || !out) return -1;
    const bitnet_config *mc = &c->model.config;
    out->model_mapped_bytes = c->model.map_size;
    out->runstate_bytes = c->rs.arena_bytes;
    out->kv_cache_bytes = 2 * (size_t)mc->n_layers * c->rs.max_seq_len * mc->n_kv_heads * mc->head_dim *
                          sizeof(float);
    out->process_resident_bytes = platform_resident_bytes();
    struct rusage ru;
#ifdef __APPLE__
    const size_t maxrss_unit = 1;    /* macOS reports ru_maxrss in bytes */
#else
    const size_t maxrss_unit = 1024; /* Linux reports kilobytes */
#endif
    out->process_peak_resident_bytes = getrusage(RUSAGE_SELF, &ru) == 0 ? (size_t)ru.ru_maxrss * maxrss_unit : 0;
    return 0;
}

const char *bitnet_model_description(const BitNetContext *c) { return c ? c->desc : ""; }

const char *bitnet_last_error(const BitNetContext *c) { return c ? c->err : g_init_err; }
