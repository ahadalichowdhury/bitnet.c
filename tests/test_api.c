/*
 * test_api.c — Tests of the public bitnet.h API on the real model.
 *
 *   build/test_api [MODEL.bitnet] [TOKENIZER.json]
 *
 * Covers: init errors, raw generation, determinism and seeds, multi-turn chat
 * memory, reset and system prompts, cancellation (from the callback, from
 * another thread, during the prompt) with conversation consistency afterwards,
 * context overflow handling, UTF-8 validity of every callback piece, memory
 * info and NULL safety.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#undef NDEBUG
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bitnet.h"

#define DEFAULT_MODEL     "models/bitnet_2b4t.bitnet"
#define DEFAULT_TOKENIZER "models/hf/bitnet-b1.58-2B-4T/tokenizer.json"

typedef struct {
    char           text[16384];
    size_t         n;
    int            pieces;
    int            cancel_after; /* > 0: call bitnet_cancel after this many pieces */
    BitNetContext *ctx;
    _Atomic int    started;      /* set once the first piece arrived */
} collect;

static int valid_utf8(const unsigned char *s) {
    while (*s) {
        const unsigned char b = *s;
        const int L = b < 0x80 ? 1 : (b >= 0xC2 && b <= 0xDF) ? 2 : (b >= 0xE0 && b <= 0xEF) ? 3
                    : (b >= 0xF0 && b <= 0xF4) ? 4 : 0;
        if (!L) return 0;
        for (int k = 1; k < L; k++)
            if ((s[k] & 0xC0) != 0x80) return 0;
        s += L;
    }
    return 1;
}

static void on_piece(const char *piece, void *user) {
    collect *c = user;
    assert(piece && valid_utf8((const unsigned char *)piece));
    const size_t l = strlen(piece);
    assert(l > 0);
    if (c->n + l < sizeof(c->text)) {
        memcpy(c->text + c->n, piece, l);
        c->n += l;
        c->text[c->n] = '\0';
    }
    atomic_store(&c->started, 1);
    if (++c->pieces == c->cancel_after) bitnet_cancel(c->ctx);
}

static BitNetStats stats_of(BitNetContext *ctx) {
    BitNetStats s;
    assert(bitnet_last_stats(ctx, &s) == 0);
    return s;
}

static void reset_collect(collect *c, BitNetContext *ctx) {
    memset(c, 0, sizeof(*c));
    c->ctx = ctx;
}

typedef struct {
    BitNetContext      *ctx;
    int                 delay_ms;
    const _Atomic int  *wait_for; /* optional: start the delay only once *wait_for != 0 */
} canceller;

/* Cancels from another thread. Waiting for an event instead of a fixed time
 * keeps the test meaningful under slow (sanitizer) builds. */
static void *cancel_later(void *arg) {
    const canceller *c = arg;
    while (c->wait_for && !atomic_load(c->wait_for)) {
        struct timespec poll = {0, 1000000L};
        nanosleep(&poll, NULL);
    }
    struct timespec ts = {c->delay_ms / 1000, (long)(c->delay_ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
    bitnet_cancel(c->ctx);
    return NULL;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *model = argc > 1 ? argv[1] : DEFAULT_MODEL;
    const char *tok = argc > 2 ? argv[2] : DEFAULT_TOKENIZER;
    int checks = 0;

    /* ---- init errors and NULL safety */
    assert(bitnet_init("does/not/exist.bitnet", tok, bitnet_default_config()) == NULL);
    assert(strstr(bitnet_last_error(NULL), "does/not/exist.bitnet"));
    assert(bitnet_init(model, "missing.json", bitnet_default_config()) == NULL);
    assert(strstr(bitnet_last_error(NULL), "missing.json"));
    assert(bitnet_init(NULL, tok, bitnet_default_config()) == NULL);
    bitnet_free(NULL);
    bitnet_cancel(NULL);
    bitnet_reset_chat(NULL);
    BitNetStats dummy;
    assert(bitnet_last_stats(NULL, &dummy) == -1);
    checks += 8;

    BitNetContext *ctx = bitnet_init(model, tok, bitnet_default_config());
    if (!ctx) {
        fprintf(stderr, "init: %s\n", bitnet_last_error(NULL));
        return 1;
    }
    assert(bitnet_last_error(ctx)[0] == '\0');
    printf("%s\n", bitnet_model_description(ctx));
    collect c;
    const BitNetSampleParams greedy = {.temperature = 0.0f, .top_p = 1.0f, .seed = 1};

    /* ---- raw generation: greedy, deterministic */
    BitNetSampleParams p = greedy;
    p.max_new_tokens = 8;
    reset_collect(&c, ctx);
    bitnet_generate(ctx, "The capital of France is", p, on_piece, &c);
    BitNetStats s = stats_of(ctx);
    assert(strstr(c.text, "Paris") && s.stop_reason == BITNET_STOP_MAX_TOKENS);
    assert(s.prompt_tokens == 6 && s.generated_tokens == 8 && s.context_used == 6 + 7);
    char first[256];
    snprintf(first, sizeof(first), "%s", c.text);
    reset_collect(&c, ctx);
    bitnet_generate(ctx, "The capital of France is", p, on_piece, &c);
    assert(strcmp(first, c.text) == 0);
    bitnet_generate(ctx, "The capital of France is", p, NULL, NULL); /* NULL callback is fine */
    assert(stats_of(ctx).generated_tokens == 8);
    printf("raw generate:   \"The capital of France is%s\" (deterministic)\n", first);
    checks += 6;

    /* ---- seeds: explicit seed reproducible, seed 0 = fresh random seed */
    BitNetSampleParams hot = {.temperature = 0.9f, .top_p = 0.95f, .seed = 1234, .max_new_tokens = 24};
    char a[2048], b[2048];
    reset_collect(&c, ctx);
    bitnet_generate(ctx, "My favourite animal is", hot, on_piece, &c);
    snprintf(a, sizeof(a), "%s", c.text);
    reset_collect(&c, ctx);
    bitnet_generate(ctx, "My favourite animal is", hot, on_piece, &c);
    assert(strcmp(a, c.text) == 0 && stats_of(ctx).seed == 1234);
    hot.seed = 0;
    bitnet_generate(ctx, "My favourite animal is", hot, NULL, NULL);
    const uint64_t s1 = stats_of(ctx).seed;
    bitnet_generate(ctx, "My favourite animal is", hot, NULL, NULL);
    assert(s1 != 0 && stats_of(ctx).seed != s1);
    (void)b;
    checks += 3;

    /* ---- stop strings, penalties, and invalid stop configurations */
    {
        BitNetSampleParams sp = greedy;
        const char *stops[] = {"."};
        sp.stop = stops;
        sp.n_stop = 1;
        sp.max_new_tokens = 32;
        reset_collect(&c, ctx);
        bitnet_generate(ctx, "The capital of France is", sp, on_piece, &c);
        s = stats_of(ctx);
        assert(s.stop_reason == BITNET_STOP_STOP_STRING && strstr(c.text, "Paris") && !strchr(c.text, '.'));
        printf("stop string:    \"The capital of France is%s\" [stop_string]\n", c.text);

        sp.n_stop = 0;
        sp.stop = NULL;
        sp.repetition_penalty = 1.3f;
        sp.frequency_penalty = 0.2f;
        sp.presence_penalty = 0.1f;
        sp.penalty_last_n = 128;
        bitnet_generate(ctx, "The capital of France is", sp, NULL, NULL);
        assert(stats_of(ctx).stop_reason != BITNET_STOP_ERROR);

        char longs[80];
        memset(longs, 'x', sizeof(longs) - 1);
        longs[sizeof(longs) - 1] = '\0';
        const char *bad_long[] = {longs};
        const char *bad_empty[] = {""};
        sp = greedy;
        sp.stop = bad_long;
        sp.n_stop = 1;
        bitnet_generate(ctx, "x", sp, NULL, NULL);
        assert(stats_of(ctx).stop_reason == BITNET_STOP_ERROR && strstr(bitnet_last_error(ctx), "1..64"));
        sp.stop = bad_empty;
        bitnet_chat_turn(ctx, "x", sp, NULL, NULL);
        assert(stats_of(ctx).stop_reason == BITNET_STOP_ERROR);
        sp.n_stop = 9;
        bitnet_generate(ctx, "x", sp, NULL, NULL);
        assert(stats_of(ctx).stop_reason == BITNET_STOP_ERROR && strstr(bitnet_last_error(ctx), "max 8"));
        checks += 6;
    }

    /* ---- multi-turn chat memory */
    p = greedy;
    p.max_new_tokens = 32;
    reset_collect(&c, ctx);
    bitnet_chat_turn(ctx, "What is 7 + 5? Answer with just the number.", p, on_piece, &c);
    BitNetStats t1 = stats_of(ctx);
    printf("chat turn 1:    \"%s\" [%d prompt tokens]\n", c.text, t1.prompt_tokens);
    assert(strstr(c.text, "12") && t1.stop_reason == BITNET_STOP_END_OF_TURN);
    reset_collect(&c, ctx);
    bitnet_chat_turn(ctx, "Now multiply that result by 10. Answer with just the number.", p, on_piece, &c);
    BitNetStats t2 = stats_of(ctx);
    printf("chat turn 2:    \"%s\" [%d prompt tokens, context %d]\n", c.text, t2.prompt_tokens, t2.context_used);
    assert(strstr(c.text, "120"));                      /* needs turn 1 in memory */
    assert(t2.prompt_tokens < t1.prompt_tokens);        /* only the new message was prefilled */
    assert(t2.context_used == t1.context_used + t2.prompt_tokens + t2.generated_tokens);
    checks += 5;

    /* ---- cancel from the callback; the conversation must stay usable */
    p.max_new_tokens = 200;
    reset_collect(&c, ctx);
    c.cancel_after = 3;
    bitnet_chat_turn(ctx, "Write a long story about a dragon who learns to paint.", p, on_piece, &c);
    BitNetStats tc = stats_of(ctx);
    assert(tc.stop_reason == BITNET_STOP_CANCELLED && tc.generated_tokens <= 6);
    reset_collect(&c, ctx);
    p.max_new_tokens = 32;
    bitnet_chat_turn(ctx, "What was the number you answered in your second reply? Just the number.", p,
                     on_piece, &c);
    BitNetStats ta = stats_of(ctx);
    printf("after cancel:   \"%s\" [%s]\n", c.text, ta.stop_reason == BITNET_STOP_END_OF_TURN ? "end of turn" : "?");
    assert(ta.stop_reason == BITNET_STOP_END_OF_TURN && strstr(c.text, "120"));
    checks += 3;

    /* ---- cancel from another thread during decode */
    pthread_t th;
    reset_collect(&c, ctx);
    canceller cn = {ctx, 200, &c.started}; /* cancel 200 ms into the decode phase */
    p.max_new_tokens = 400;
    assert(pthread_create(&th, NULL, cancel_later, &cn) == 0);
    bitnet_generate(ctx, "Here is a very long essay about the history of mathematics:", p, on_piece, &c);
    pthread_join(th, NULL);
    s = stats_of(ctx);
    assert(s.stop_reason == BITNET_STOP_CANCELLED && s.generated_tokens > 0 && s.generated_tokens < 400);
    checks++;

    /* ---- cancel during the prompt: the chat turn is dropped, history kept */
    bitnet_reset_chat(ctx);
    p.max_new_tokens = 16;
    bitnet_chat_turn(ctx, "Remember the word 'banana'. Reply with OK.", p, NULL, NULL);
    const int used_before = stats_of(ctx).context_used;
    /* ~2000 tokens: fits the 4096-position context but takes seconds to
     * prefill, so the cancel lands in the middle of the prompt. */
    char *big = malloc(10000);
    size_t bl = 0;
    while (bl < 8000) bl += (size_t)snprintf(big + bl, 10000 - bl, "This is filler text number %zu. ", bl);
    cn.delay_ms = 150; /* prefilling ~2000 tokens takes seconds: lands mid-prompt */
    cn.wait_for = NULL;
    assert(pthread_create(&th, NULL, cancel_later, &cn) == 0);
    bitnet_chat_turn(ctx, big, p, NULL, NULL);
    pthread_join(th, NULL);
    s = stats_of(ctx);
    if (!(s.stop_reason == BITNET_STOP_CANCELLED && s.generated_tokens == 0 && s.context_used == used_before))
        fprintf(stderr, "prompt cancel: reason %d, generated %d, context %d (before %d), error '%s'\n",
                s.stop_reason, s.generated_tokens, s.context_used, used_before, bitnet_last_error(ctx));
    assert(s.stop_reason == BITNET_STOP_CANCELLED && s.generated_tokens == 0 && s.context_used == used_before);
    reset_collect(&c, ctx);
    bitnet_chat_turn(ctx, "Which word did I ask you to remember? Answer with just the word.", p, on_piece, &c);
    printf("prompt cancel:  history kept -> \"%s\"\n", c.text);
    assert(strstr(c.text, "anana") && stats_of(ctx).context_used > used_before);
    checks += 3;

    /* ---- reset and system prompt */
    p.max_new_tokens = 1;
    bitnet_reset_chat(ctx);
    bitnet_chat_turn(ctx, "Hi", p, NULL, NULL);
    const int with_default_system = stats_of(ctx).prompt_tokens;
    assert(bitnet_set_system_prompt(ctx, NULL) == 0);
    bitnet_chat_turn(ctx, "Hi", p, NULL, NULL);
    assert(stats_of(ctx).prompt_tokens < with_default_system && stats_of(ctx).context_used <= 12);
    assert(bitnet_set_system_prompt(ctx, "You are a pirate. Always talk like a pirate.") == 0);
    bitnet_chat_turn(ctx, "Hi", p, NULL, NULL);
    assert(stats_of(ctx).prompt_tokens > with_default_system);
    checks += 3;

    /* ---- memory info */
    BitNetMemoryInfo mi;
    assert(bitnet_memory_info(ctx, &mi) == 0);
    assert(mi.model_mapped_bytes > (size_t)800 << 20 && /* q8: 851 MiB, f16: 1125 MiB */
           mi.kv_cache_bytes == (size_t)2 * 30 * 4096 * 640 * 2); /* float16 */
    assert(mi.runstate_bytes > mi.kv_cache_bytes && mi.process_resident_bytes > 0);
    checks += 2;
    bitnet_free(ctx);

    /* ---- small context: overflow resets history, too-long input is an error */
    BitNetConfig small = bitnet_default_config();
    small.max_seq_len = 96;
    small.n_threads = 4;
    BitNetContext *sc = bitnet_init(model, tok, small);
    assert(sc);
    p = greedy;
    p.max_new_tokens = 20;
    int resets = 0;
    for (int turn = 0; turn < 8; turn++) {
        bitnet_chat_turn(sc, "Tell me one fact about the moon.", p, NULL, NULL);
        s = stats_of(sc);
        assert(s.stop_reason != BITNET_STOP_ERROR && s.context_used <= 96 && s.context_size == 96);
        resets += s.context_reset;
    }
    assert(resets > 0);
    bitnet_chat_turn(sc, big, p, NULL, NULL);
    s = stats_of(sc);
    assert(s.stop_reason == BITNET_STOP_ERROR && strstr(bitnet_last_error(sc), "context"));
    bitnet_chat_turn(sc, "Hi", p, NULL, NULL); /* still usable after an error */
    assert(stats_of(sc).stop_reason != BITNET_STOP_ERROR && bitnet_last_error(sc)[0] == '\0');
    printf("small context:  %d history resets over 8 turns in 96 positions; oversize input rejected\n", resets);
    checks += 4;
    bitnet_free(sc);
    free(big);

    printf("API tests:      PASSED (%d checks)\n", checks);
    return 0;
}
