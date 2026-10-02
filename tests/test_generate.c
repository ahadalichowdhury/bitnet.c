/*
 * test_generate.c — Text generation on the real BitNet b1.58 2B-4T model:
 * test suite, benchmark and interactive chat.
 *
 *   build/test_generate MODEL.bitnet TOKENIZER.json                  full test suite + benchmark
 *   build/test_generate MODEL TOKENIZER --prompt "text" [--chat]      one generation
 *   build/test_generate MODEL TOKENIZER --interactive                 multi-turn chat (stdin)
 *
 *   sampling: --temp T (0 = greedy) --top-k K --top-p P --seed S --max-new N
 *   chat:     --system "text" (default: the model card's system prompt)
 *   --quick:  shorter test suite (sanitizer builds)
 *   --unit-only: sampler + UTF-8 suites only, no model needed (CI)
 *
 * Build & run: `make test` (binary: build/test_generate); `make asan` for sanitizers.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#undef NDEBUG /* verification asserts must always run */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "generate.h"
#include "model_loader.h"
#include "platform.h"
#include "test_alloc_hook.h"
#include "tokenizer.h"
#include "transformer.h"

#define DEFAULT_SYSTEM "You are a helpful AI assistant."

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    assert(p);
    return p;
}

/* ------------------------------------------------------------------------- */
/* Output sinks                                                              */
/* ------------------------------------------------------------------------- */

/* Is buf[0..n) a sequence of complete, valid UTF-8 characters? */
static int utf8_valid(const uint8_t *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        const uint8_t b = s[i];
        int L = b < 0x80 ? 1 : (b >= 0xC2 && b <= 0xDF) ? 2 : (b >= 0xE0 && b <= 0xEF) ? 3
              : (b >= 0xF0 && b <= 0xF4) ? 4 : 0;
        if (!L || i + (size_t)L > n) return 0;
        for (int k = 1; k < L; k++) {
            const uint8_t c = s[i + k];
            if ((c & 0xC0) != 0x80) return 0;
            if (k == 1 && ((b == 0xE0 && c < 0xA0) || (b == 0xED && c > 0x9F) ||
                           (b == 0xF0 && c < 0x90) || (b == 0xF4 && c > 0x8F)))
                return 0;
        }
        i += (size_t)L;
    }
    return 1;
}

typedef struct {
    char  *buf;
    size_t n, cap;
    int    echo;          /* also write to stdout (live streaming) */
    int    chunks;        /* sink calls */
    int    invalid_chunks; /* chunks that were not complete valid UTF-8 */
} capture;

static void capture_sink(void *user, const char *s, size_t n) {
    capture *c = user;
    c->chunks++;
    if (!utf8_valid((const uint8_t *)s, n)) c->invalid_chunks++;
    if (c->buf) {
        const size_t k = c->n + n <= c->cap ? n : c->cap - c->n;
        memcpy(c->buf + c->n, s, k);
        c->n += k;
    }
    if (c->echo) {
        fwrite(s, 1, n, stdout);
        fflush(stdout);
    }
}

/* ------------------------------------------------------------------------- */
/* 1. Sampler tests                                                          */
/* ------------------------------------------------------------------------- */

/* Empirical distribution of n draws vs expected (max abs difference). */
static double sample_dist_err(const SamplerConfig *cfg, const float *logits, int V,
                              const double *expect, int n) {
    Sampler s;
    char err[64];
    assert(sampler_init(&s, V, cfg, err, sizeof(err)) == 0);
    float *tmp = xmalloc((size_t)V * sizeof(float));
    long *cnt = calloc((size_t)V, sizeof(long));
    for (int i = 0; i < n; i++) {
        memcpy(tmp, logits, (size_t)V * sizeof(float));
        const int32_t t = sampler_sample(&s, tmp);
        assert(t >= 0 && t < V);
        cnt[t]++;
    }
    double worst = 0;
    for (int i = 0; i < V; i++) {
        if (expect[i] == 0.0) assert(cnt[i] == 0); /* filtered tokens never appear */
        worst = fmax(worst, fabs((double)cnt[i] / n - expect[i]));
    }
    free(tmp);
    free(cnt);
    sampler_free(&s);
    return worst;
}

/* Reference distribution in float64: temperature, top-k, top-p (HF order). */
static void expected_dist(const float *logits, int V, const SamplerConfig *c, double *out) {
    int *ord = xmalloc((size_t)V * sizeof(int));
    for (int i = 0; i < V; i++) ord[i] = i;
    for (int i = 0; i < V; i++) /* selection sort by logit desc, id asc (V is small) */
        for (int j = i + 1; j < V; j++)
            if (logits[ord[j]] > logits[ord[i]] ||
                (logits[ord[j]] == logits[ord[i]] && ord[j] < ord[i])) {
                const int t = ord[i]; ord[i] = ord[j]; ord[j] = t;
            }
    int keep = (c->top_k > 0 && c->top_k < V) ? c->top_k : V;
    double mx = logits[ord[0]], sum = 0;
    for (int j = 0; j < keep; j++) sum += exp(((double)logits[ord[j]] - mx) / c->temperature);
    if (c->top_p < 1.0f) {
        double cum = 0;
        for (int j = 0; j < keep; j++) {
            cum += exp(((double)logits[ord[j]] - mx) / c->temperature) / sum;
            if (cum >= c->top_p) { keep = j + 1; break; }
        }
    }
    double tot = 0;
    for (int i = 0; i < V; i++) out[i] = 0;
    for (int j = 0; j < keep; j++) tot += out[ord[j]] = exp(((double)logits[ord[j]] - mx) / c->temperature);
    for (int j = 0; j < keep; j++) out[ord[j]] /= tot;
    free(ord);
}

static void sampler_tests(int quick) {
    const int V = 8, N = quick ? 40000 : 200000;
    const float logits[8] = {2.0f, 1.0f, 0.5f, 3.0f, -1.0f, 0.0f, 2.5f, -3.0f};
    static const SamplerConfig cfgs[] = {
        {1.0f, 0, 1.0f, 1},   /* plain softmax */
        {0.5f, 0, 1.0f, 2},   /* sharper */
        {2.0f, 0, 1.0f, 3},   /* flatter */
        {1.0f, 3, 1.0f, 4},   /* top-k */
        {0.7f, 0, 0.9f, 5},   /* nucleus */
        {1.0f, 0, 0.5f, 6},   /* small nucleus */
        {0.8f, 4, 0.8f, 7},   /* both */
    };
    double expect[8], worst = 0;
    for (size_t c = 0; c < sizeof(cfgs) / sizeof(cfgs[0]); c++) {
        expected_dist(logits, V, &cfgs[c], expect);
        worst = fmax(worst, sample_dist_err(&cfgs[c], logits, V, expect, N));
    }
    /* sigma of a frequency <= sqrt(0.25 / N); allow ~4.5 sigma. */
    const double tol = 4.5 * sqrt(0.25 / N);
    assert(worst < tol);

    char err[64];
    Sampler s;
    float buf[8];
    /* Greedy: argmax, ties -> lowest id; also temperature 0 ignores top-k/p. */
    const SamplerConfig greedy = {0.0f, 2, 0.5f, 9};
    assert(sampler_init(&s, V, &greedy, err, sizeof(err)) == 0);
    memcpy(buf, logits, sizeof(buf));
    assert(sampler_sample(&s, buf) == 3);
    const float ties[8] = {1, 5, 2, 5, 5, 0, 1, 2};
    memcpy(buf, ties, sizeof(buf));
    assert(sampler_sample(&s, buf) == 1);
    sampler_free(&s);
    /* top-k with ties keeps the lowest ids; top_k=1 is deterministic. */
    const SamplerConfig k1 = {1.0f, 1, 1.0f, 10};
    assert(sampler_init(&s, V, &k1, err, sizeof(err)) == 0);
    for (int i = 0; i < 100; i++) {
        memcpy(buf, ties, sizeof(buf));
        assert(sampler_sample(&s, buf) == 1);
    }
    sampler_free(&s);
    /* Uniform logits: every token reachable. */
    const float flat[8] = {0};
    for (int i = 0; i < V; i++) expect[i] = 1.0 / V;
    const SamplerConfig f = {1.0f, 0, 1.0f, 11};
    assert(sample_dist_err(&f, flat, V, expect, N) < tol);

    /* Full-vocab equivalence: the top-p prefilter + sort must pick exactly
     * the same token as a naive full sort, draw for draw. */
    const int BV = 128256, trials = quick ? 20 : 200;
    float *lg = xmalloc((size_t)BV * 4), *tmp = xmalloc((size_t)BV * 4), *p = xmalloc((size_t)BV * 4);
    int *ord = xmalloc((size_t)BV * sizeof(int));
    const SamplerConfig big = {0.8f, 0, 0.9f, 1234};
    assert(sampler_init(&s, BV, &big, err, sizeof(err)) == 0);
    uint64_t r = 99;
    double total_us = 0;
    for (int t = 0; t < trials; t++) {
        for (int i = 0; i < BV; i++) { /* heavy-tailed logits like a real LM */
            r = r * 6364136223846793005ull + 1442695040888963407ull;
            const double u = ((r >> 11) + 0.5) * 0x1.0p-53;
            lg[i] = (float)(-log(u) * (t % 3 == 0 ? 0.5 : 2.0));
        }
        /* Naive reference using the identical float probabilities. */
        float mx = lg[0];
        for (int i = 1; i < BV; i++) mx = lg[i] > mx ? lg[i] : mx;
        for (int i = 0; i < BV; i++) p[i] = (lg[i] - mx) * (1.0f / big.temperature);
        exp_neon(p, p, BV);
        double sum = 0;
        for (int i = 0; i < BV; i++) { sum += p[i]; ord[i] = i; }
        /* partial insertion is too slow for 128K: sort indices by p desc. */
        for (int gap = BV / 2; gap > 0; gap /= 2) /* shell sort, deterministic */
            for (int i = gap; i < BV; i++) {
                const int v = ord[i];
                int j = i;
                while (j >= gap && (p[ord[j - gap]] < p[v] || (p[ord[j - gap]] == p[v] && ord[j - gap] > v))) {
                    ord[j] = ord[j - gap];
                    j -= gap;
                }
                ord[j] = v;
            }
        double cum = 0, total = 0;
        int keep = BV;
        for (int j = 0; j < BV; j++) {
            cum += p[ord[j]];
            if (cum >= big.top_p * sum) { keep = j + 1; break; }
        }
        for (int j = 0; j < keep; j++) total += p[ord[j]];
        uint64_t saved[4];
        memcpy(saved, s.rng, sizeof(saved));
        const double ru = sampler_uniform(&s) * total;
        memcpy(s.rng, saved, sizeof(saved));
        double c2 = 0;
        int expect_tok = ord[keep - 1];
        for (int j = 0; j < keep; j++) {
            c2 += p[ord[j]];
            if (ru < c2) { expect_tok = ord[j]; break; }
        }
        memcpy(tmp, lg, (size_t)BV * 4);
        const uint64_t t0 = platform_now_ns();
        const int32_t got = sampler_sample(&s, tmp);
        total_us += (platform_now_ns() - t0) / 1e3;
        assert(got == expect_tok);
    }
    sampler_free(&s);

    /* Reproducibility: same seed -> same stream; reseed restarts it. */
    Sampler a, b;
    const SamplerConfig cr = {0.9f, 50, 0.95f, 777};
    assert(sampler_init(&a, BV, &cr, err, sizeof(err)) == 0 && sampler_init(&b, BV, &cr, err, sizeof(err)) == 0);
    int32_t first[64];
    for (int i = 0; i < 64; i++) {
        memcpy(tmp, lg, (size_t)BV * 4);
        first[i] = sampler_sample(&a, tmp);
        memcpy(tmp, lg, (size_t)BV * 4);
        assert(sampler_sample(&b, tmp) == first[i]);
    }
    sampler_reseed(&a, 777);
    for (int i = 0; i < 64; i++) {
        memcpy(tmp, lg, (size_t)BV * 4);
        assert(sampler_sample(&a, tmp) == first[i]);
    }
    sampler_free(&a);
    sampler_free(&b);
    free(lg); free(tmp); free(p); free(ord);

    printf("Sampler:        PASSED (7 configs: empirical vs exact distribution max err %.4f < %.4f over "
           "%d draws; greedy/ties/top-k=1; top-p fast path == full sort on %d 128K-vocab trials; "
           "seeded streams reproducible; %.0f us/sample at 128K vocab, temp+top-p)\n",
           worst, tol, N, trials, total_us / trials);
}

/* ------------------------------------------------------------------------- */
/* 2. UTF-8 streaming tests                                                  */
/* ------------------------------------------------------------------------- */

static void stream_all(const char *in, size_t n, const size_t *cuts, int ncuts, capture *c) {
    Utf8Stream u;
    utf8_stream_init(&u, capture_sink, c);
    size_t prev = 0;
    for (int i = 0; i <= ncuts; i++) {
        const size_t end = i < ncuts ? cuts[i] : n;
        utf8_stream_write(&u, in + prev, end - prev);
        prev = end;
    }
    utf8_stream_flush(&u);
}

static void utf8_tests(void) {
    char out1[512], out2[512];
    /* Valid text: every chunking reproduces it exactly, every emitted chunk is
     * complete valid UTF-8. */
    const char *text = "h\xc3\xa9llo \xe4\xb8\x96\xe7\x95\x8c \xf0\x9f\xa6\x80"
                       "\xf0\x9f\x91\xa8\xe2\x80\x8d\xf0\x9f\x91\xa9 \xce\xa9\xe2\x89\x88\xc3\xa7\xe2\x88\x9a!";
    const size_t n = strlen(text);
    int cases = 0;
    for (int mode = 0; mode < 1002; mode++) {
        size_t cuts[64];
        int nc = 0;
        if (mode == 0) { /* one write */ }
        else if (mode == 1) for (size_t i = 1; i < n; i++) cuts[nc++] = i; /* byte by byte */
        else {
            uint64_t r = (uint64_t)mode * 0x9E3779B97F4A7C15ull;
            for (size_t i = 1; i < n && nc < 60; i++) {
                r ^= r << 13; r ^= r >> 7; r ^= r << 17;
                if (r % 3 == 0) cuts[nc++] = i;
            }
        }
        capture c = {out1, 0, sizeof(out1), 0, 0, 0};
        stream_all(text, n, cuts, nc, &c);
        assert(c.n == n && memcmp(out1, text, n) == 0 && c.invalid_chunks == 0);
        cases++;
    }

    /* Invalid input: U+FFFD per maximal invalid subpart, identical to Python's
     * bytes.decode("utf-8", "replace"), regardless of chunking. */
    static const struct { const char *in; size_t n; const char *want; } bad[] = {
        {"\xff", 1, "\xef\xbf\xbd"},
        {"\x61\x80\x62", 3, "\x61\xef\xbf\xbd\x62"},
        {"\xe2\x82", 2, "\xef\xbf\xbd"},
        {"\xc0\xaf", 2, "\xef\xbf\xbd\xef\xbf\xbd"},
        {"\xed\xa0\x80", 3, "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd"},
        {"\xf4\x90\x80\x80", 4, "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd"},
        {"\xe2\x28\xa1", 3, "\xef\xbf\xbd\x28\xef\xbf\xbd"},
        {"\xf0\x9f\x98", 3, "\xef\xbf\xbd"},
        {"\x6f\x6b\xf0\x9f\x98\x80\xf0\x9f", 8, "\x6f\x6b\xf0\x9f\x98\x80\xef\xbf\xbd"},
        {"\xe0\x80\x80", 3, "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd"},
        {"\xc3\xa9\xc3", 3, "\xc3\xa9\xef\xbf\xbd"},
    };
    for (size_t b = 0; b < sizeof(bad) / sizeof(bad[0]); b++) {
        capture c1 = {out1, 0, sizeof(out1), 0, 0, 0}, c2 = {out2, 0, sizeof(out2), 0, 0, 0};
        stream_all(bad[b].in, bad[b].n, NULL, 0, &c1);
        size_t cuts[8];
        for (size_t i = 1; i < bad[b].n; i++) cuts[i - 1] = i;
        stream_all(bad[b].in, bad[b].n, cuts, (int)bad[b].n - 1, &c2);
        assert(c1.n == strlen(bad[b].want) && memcmp(out1, bad[b].want, c1.n) == 0);
        assert(c2.n == c1.n && memcmp(out1, out2, c1.n) == 0);
        assert(c1.invalid_chunks == 0 && c2.invalid_chunks == 0);
        cases += 2;
    }

    /* Random bytes: output is always valid UTF-8 and chunking-invariant. */
    uint64_t r = 7;
    for (int t = 0; t < 2000; t++) {
        char in[96];
        const size_t len = 1 + (size_t)(t % 96);
        for (size_t i = 0; i < len; i++) {
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            in[i] = (char)(t & 1 ? (r & 0xFF) : (0x80 | (r & 0x7F)) - (r % 5 == 0 ? 0x40 : 0));
        }
        capture c1 = {out1, 0, sizeof(out1), 0, 0, 0}, c2 = {out2, 0, sizeof(out2), 0, 0, 0};
        stream_all(in, len, NULL, 0, &c1);
        size_t cuts[96];
        for (size_t i = 1; i < len; i++) cuts[i - 1] = i;
        stream_all(in, len, cuts, (int)len - 1, &c2);
        assert(c1.n == c2.n && memcmp(out1, out2, c1.n) == 0);
        assert(utf8_valid((const uint8_t *)out1, c1.n) && c1.invalid_chunks == 0 && c2.invalid_chunks == 0);
        cases++;
    }
    printf("UTF-8 stream:   PASSED (%d cases: 1002 chunkings of emoji/CJK/ZWJ text, 11 malformed inputs "
           "== Python 'replace', 2000 random byte strings chunking-invariant; no partial character "
           "ever emitted)\n", cases);
}

/* ------------------------------------------------------------------------- */
/* 3. Generation on the real model                                           */
/* ------------------------------------------------------------------------- */

typedef struct {
    const BitNetModel *m;
    const Tokenizer   *tk;
    RunState          *s;
    int32_t            stop[2];
} ctx_t;

static const char *reason_str(gen_stop_reason r) {
    return r == GEN_STOP_EOS ? "end-of-turn" : r == GEN_STOP_MAX_TOKENS ? "max tokens" : "context full";
}

static void print_stats(const char *label, const GenerateStats *st) {
    printf("  [%s] prompt %d tok | prefill %.1f tok/s | TTFT %.0f ms | generated %d tok | decode %.1f "
           "tok/s | stop: %s\n", label, st->n_prompt, st->prefill_tok_s, st->ttft_ms, st->n_generated,
           st->decode_tok_s, reason_str(st->reason));
}

/* Runs one generation from position 0 with live streaming; returns stats and
 * fills tokens/text. */
static GenerateStats run(ctx_t *c, const char *title, const char *lead, const int32_t *prompt,
                         int n_prompt, const SamplerConfig *sc, int max_new, int echo, int32_t *out,
                         capture *cap) {
    Sampler smp;
    char err[64];
    assert(sampler_init(&smp, c->m->config.vocab_size, sc, err, sizeof(err)) == 0);
    GenerateParams gp = {max_new, c->stop, 2, capture_sink, cap, out, max_new, NULL};
    cap->echo = echo;
    if (echo && title) printf("\n--- %s\n%s", title, lead ? lead : "");
    GenerateStats st;
    assert(generate(c->m, c->tk, c->s, &smp, prompt, n_prompt, 0, &gp, &st) == 0);
    if (echo) printf("\n");
    sampler_free(&smp);
    return st;
}

static int encode_raw(const Tokenizer *tk, const char *text, int32_t *ids, int cap) {
    ids[0] = tokenizer_token_id(tk, "<|begin_of_text|>");
    const int n = bpe_encode_ex(tk, text, strlen(text), ids + 1, cap - 1, TOKENIZER_NO_SPECIAL);
    assert(n >= 0 && n < cap - 1);
    return n + 1;
}

static void model_tests(ctx_t *c, int quick) {
    enum { CAP = 4096 };
    int32_t prompt[CAP], out1[512], out2[512];
    char *text1 = xmalloc(65536), *text2 = xmalloc(65536);
    const SamplerConfig greedy = {0.0f, 0, 1.0f, 0};
    const int n_raw = encode_raw(c->tk, "The capital of France is", prompt, CAP);
    const int gen_n = quick ? 12 : 48;

    printf("\nGeneration on the real model (streamed live as it is produced):\n");

    /* A. Greedy raw completion, run twice: identical, and identical to a
     *    hand-rolled transformer_forward + argmax loop. */
    capture cap = {text1, 0, 65536, 1, 0, 0};
    GenerateStats a = run(c, "greedy, raw completion", "The capital of France is", prompt, n_raw,
                          &greedy, gen_n, 1, out1, &cap);
    print_stats("greedy", &a);
    capture cap2 = {text2, 0, 65536, 0, 0, 0};
    GenerateStats a2 = run(c, NULL, NULL, prompt, n_raw, &greedy, gen_n, 0, out2, &cap2);
    assert(a2.n_generated == a.n_generated && memcmp(out1, out2, (size_t)a.n_generated * 4) == 0);
    assert(cap.n == cap2.n && memcmp(text1, text2, cap.n) == 0);
    for (int i = 0; i < n_raw; i++) transformer_forward(prompt[i], i, c->m, c->s);
    for (int g = 0; g < a.n_generated; g++) {
        int best = 0;
        for (int i = 1; i < c->m->config.vocab_size; i++) best = c->s->logits[i] > c->s->logits[best] ? i : best;
        assert(best == out1[g]);
        transformer_forward(best, n_raw + g, c->m, c->s);
    }
    printf("  -> deterministic (2 runs identical) and == manual forward+argmax loop for all %d tokens\n",
           a.n_generated);

    /* B. Greedy chat: should end its turn by itself with <|eot_id|>. */
    int n = build_chat_prompt(c->tk, DEFAULT_SYSTEM, "What is the capital of France? Answer in one sentence.",
                              1, prompt, CAP);
    cap = (capture){text1, 0, 65536, 1, 0, 0};
    GenerateStats b = run(c, "greedy chat", "User: What is the capital of France? Answer in one sentence.\n"
                          "Assistant: ", prompt, n, &greedy, 64, 1, out1, &cap);
    print_stats("greedy chat", &b);
    assert(b.reason == GEN_STOP_EOS && b.stop_token == c->stop[1]);

    /* C. Creative sampling (temp 0.7, top-p 0.9): same seed reproduces the
     *    exact text; another seed diverges. */
    const SamplerConfig creative = {0.7f, 0, 0.9f, 42};
    n = build_chat_prompt(c->tk, DEFAULT_SYSTEM, "Write a short poem about the ocean.", 1, prompt, CAP);
    cap = (capture){text1, 0, 65536, 1, 0, 0};
    GenerateStats cr = run(c, "creative chat (temp 0.7, top-p 0.9, seed 42)",
                           "User: Write a short poem about the ocean.\nAssistant: ", prompt, n, &creative,
                           quick ? 24 : 120, 1, out1, &cap);
    print_stats("creative", &cr);
    cap2 = (capture){text2, 0, 65536, 0, 0, 0};
    GenerateStats cr2 = run(c, NULL, NULL, prompt, n, &creative, quick ? 24 : 120, 0, out2, &cap2);
    assert(cr2.n_generated == cr.n_generated && memcmp(out1, out2, (size_t)cr.n_generated * 4) == 0);
    SamplerConfig other = creative;
    other.seed = 43;
    cap2 = (capture){text2, 0, 65536, 0, 0, 0};
    GenerateStats cr3 = run(c, NULL, NULL, prompt, n, &other, quick ? 24 : 120, 0, out2, &cap2);
    const int same43 = cr3.n_generated == cr.n_generated && memcmp(out1, out2, (size_t)cr.n_generated * 4) == 0;
    printf("  -> seed 42 re-run: identical (%d tokens); seed 43: %s\n", cr.n_generated,
           same43 ? "identical (!)" : "different text");

    /* D. Emoji: tokens that carry partial UTF-8 characters are buffered; the
     *    streamed bytes equal the decoded tokens and every chunk is valid. */
    n = build_chat_prompt(c->tk, DEFAULT_SYSTEM,
                          "Reply with five emoji that describe a happy summer day, then the word done.",
                          1, prompt, CAP);
    cap = (capture){text1, 0, 65536, 1, 0, 0};
    GenerateStats d = run(c, "emoji streaming (greedy)",
                          "User: Reply with five emoji that describe a happy summer day, then the word done.\n"
                          "Assistant: ", prompt, n, &greedy, 48, 1, out1, &cap);
    print_stats("emoji", &d);
    int partial = 0;
    size_t off = 0;
    for (int i = 0; i < d.n_generated; i++) {
        size_t l;
        const char *bytes = bpe_decode_bytes(c->tk, out1[i], &l);
        partial += !utf8_valid((const uint8_t *)bytes, l);
        assert(off + l <= cap.n && memcmp(text1 + off, bytes, l) == 0);
        off += l;
    }
    assert(off == cap.n && cap.invalid_chunks == 0);
    printf("  -> %d of %d tokens were partial UTF-8 fragments; %d chunks streamed, all complete valid "
           "UTF-8, bytes identical to the decoded tokens\n", partial, d.n_generated, cap.chunks);

#ifdef ALLOC_HOOK_AVAILABLE
    /* E. The whole generate() call (prefill, sampling, streaming) allocates
     *    nothing once the sampler exists. */
    {
        Sampler smp;
        char err[64];
        assert(sampler_init(&smp, c->m->config.vocab_size, &creative, err, sizeof(err)) == 0);
        capture quiet = {text2, 0, 65536, 0, 0, 0};
        GenerateParams gp = {32, c->stop, 2, capture_sink, &quiet, out2, 32, NULL};
        GenerateStats st;
        assert(hook_zones(1) == 0);
        void *volatile probe = malloc(16);
        free(probe);
        const long probe_seen = g_allocs;
        g_allocs = 0;
        assert(generate(c->m, c->tk, c->s, &smp, prompt, n, 0, &gp, &st) == 0);
        const long allocs = g_allocs;
        hook_zones(0);
        sampler_free(&smp);
        assert(probe_seen > 0 && allocs == 0);
        printf("\nAllocations:    PASSED (0 heap allocations in generate(): %d-token prefill + %d sampled "
               "tokens + streaming, all threads)\n", st.n_prompt, st.n_generated);
    }
#else
    printf("\nAllocations:    SKIPPED (allocation hook unavailable on this build/platform)\n");
#endif
    free(text1);
    free(text2);
}

/* Prefill/decode throughput at a longer prompt; also the cost of computing
 * logits for every prompt token (what skipping them saves). */
static void benchmark(ctx_t *c, int quick) {
    enum { CAP = 4096 };
    int32_t *prompt = xmalloc(CAP * sizeof(int32_t)), out[256];
    char *buf = xmalloc(1 << 16);
    const char *para = "Large language models generate text one token at a time. Each new token "
                       "attends to every previous token through the key-value cache, so the cost of "
                       "attention grows with the length of the context. ";
    size_t bl = 0;
    while (bl + strlen(para) < (quick ? 600u : 2600u)) {
        memcpy(buf + bl, para, strlen(para));
        bl += strlen(para);
    }
    buf[bl] = '\0';
    const int n = build_chat_prompt(c->tk, DEFAULT_SYSTEM, buf, 1, prompt, CAP);
    const SamplerConfig sc = {0.6f, 0, 0.9f, 7}; /* model card defaults */
    capture cap = {NULL, 0, 0, 0, 0, 0};
    const int max_new = quick ? 16 : 128;
    GenerateStats st = run(c, NULL, NULL, prompt, n, &sc, max_new, 0, out, &cap);

    /* Same prompt with logits computed for every prompt token. */
    const uint64_t t0 = platform_now_ns();
    for (int i = 0; i < n; i++) transformer_forward(prompt[i], i, c->m, c->s);
    const double full_ms = (platform_now_ns() - t0) / 1e6;

    printf("\nBenchmark (Apple M1, %d threads, temp 0.6 / top-p 0.9):\n", threadpool_size(c->s->pool));
    printf("  prompt              %6d tokens\n", st.n_prompt);
    printf("  prefill             %6.1f tokens/s  (%.0f ms; %.1f tokens/s if logits were computed for "
           "every prompt token)\n", st.prefill_tok_s, st.prefill_ms, n / (full_ms / 1e3));
    printf("  time to first token %6.0f ms\n", st.ttft_ms);
    printf("  decode              %6.1f tokens/s  (%d tokens after the first, %.0f ms)\n",
           st.decode_tok_s, st.n_generated - 1, st.decode_ms);
    printf("  sampling            %6.3f ms/token  (real logits, 128K vocab, temp + top-p)\n",
           st.sample_ms / (st.n_generated + (st.reason == GEN_STOP_EOS)));
    free(prompt);
    free(buf);
}

/* ------------------------------------------------------------------------- */
/* Interactive chat (multi-turn, KV cache reused across turns)               */
/* ------------------------------------------------------------------------- */

static void interactive(ctx_t *c, const SamplerConfig *sc, const char *system, int max_new) {
    enum { CAP = 4096 };
    int32_t *prompt = xmalloc(CAP * sizeof(int32_t)), *out = xmalloc((size_t)max_new * 4);
    char line[4096];
    Sampler smp;
    char err[64];
    assert(sampler_init(&smp, c->m->config.vocab_size, sc, err, sizeof(err)) == 0);
    int pos = 0;
    GenerateStats last = {0};
    printf("BitNet b1.58 2B-4T chat (temp %.2f, top-p %.2f, top-k %d, seed %llu). Ctrl-D to quit.\n",
           sc->temperature, sc->top_p, sc->top_k, (unsigned long long)sc->seed);
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\n")] = '\0';
        if (!line[0]) continue;
        if (!isatty(STDIN_FILENO)) printf("%s\n", line); /* echo piped input */

        int n = 0;
        if (pos > 0) {
            /* Close the previous assistant turn: its last token (if it stopped on
             * max tokens) and <|eot_id|> were never fed to the cache. */
            if (last.reason != GEN_STOP_EOS && last.n_generated > 0) prompt[n++] = out[last.n_generated - 1];
            prompt[n++] = c->stop[1];
        }
        const int k = build_chat_prompt(c->tk, pos == 0 ? system : NULL, line, pos == 0, prompt + n, CAP - n);
        if (k < 0 || n + k > CAP - 1) {
            printf("(message too long)\n");
            continue;
        }
        n += k;
        if (pos + n + max_new > c->s->max_seq_len) {
            printf("(context full: starting a new conversation)\n");
            pos = 0;
            n = build_chat_prompt(c->tk, system, line, 1, prompt, CAP);
        }
        capture cap = {NULL, 0, 0, 1, 0, 0};
        GenerateParams gp = {max_new, c->stop, 2, capture_sink, &cap, out, max_new, NULL};
        assert(generate(c->m, c->tk, c->s, &smp, prompt, n, pos, &gp, &last) == 0);
        pos = last.end_pos;
        printf("\n  [prefill %d tok @ %.1f tok/s | TTFT %.0f ms | %d tok @ %.1f tok/s | context %d/%d]\n",
               last.n_prompt, last.prefill_tok_s, last.ttft_ms, last.n_generated, last.decode_tok_s,
               pos, c->s->max_seq_len);
    }
    printf("\n");
    sampler_free(&smp);
    free(prompt);
    free(out);
}

/* ------------------------------------------------------------------------- */

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *model_path = NULL, *tok_path = NULL, *prompt_text = NULL, *system = DEFAULT_SYSTEM;
    int quick = 0, chat = 0, inter = 0, max_new = 256, threads = 0, unit_only = 0;
    SamplerConfig sc = {0.6f, 0, 0.9f, (uint64_t)time(NULL)};
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int more = i + 1 < argc;
        if (!strcmp(a, "--quick")) quick = 1;
        else if (!strcmp(a, "--unit-only")) unit_only = 1;
        else if (!strcmp(a, "--chat")) chat = 1;
        else if (!strcmp(a, "--interactive")) inter = 1;
        else if (!strcmp(a, "--prompt") && more) prompt_text = argv[++i];
        else if (!strcmp(a, "--system") && more) system = argv[++i];
        else if (!strcmp(a, "--temp") && more) sc.temperature = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-k") && more) sc.top_k = atoi(argv[++i]);
        else if (!strcmp(a, "--top-p") && more) sc.top_p = (float)atof(argv[++i]);
        else if (!strcmp(a, "--seed") && more) sc.seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--max-new") && more) max_new = atoi(argv[++i]);
        else if (!strcmp(a, "--threads") && more) threads = atoi(argv[++i]);
        else if (!model_path) model_path = a;
        else if (!tok_path) tok_path = a;
        else {
            fprintf(stderr, "unknown argument %s\n", a);
            return 2;
        }
    }
    if (!unit_only && (!model_path || !tok_path || max_new < 1)) {
        fprintf(stderr, "usage: %s MODEL.bitnet TOKENIZER.json [--prompt TEXT [--chat]] [--interactive] "
                "[--system TEXT] [--temp T] [--top-k K] [--top-p P] [--seed S] [--max-new N] "
                "[--threads N] [--quick]\n", argv[0]);
        return 2;
    }

    const int test_mode = !prompt_text && !inter;
    if (test_mode || unit_only) {
        sampler_tests(quick);
        utf8_tests();
        if (unit_only) return 0; /* model-free suites only (CI) */
    }

    char err[256];
    bitnet_model m;
    if (bitnet_model_load(model_path, &m, err, sizeof(err)) != 0) {
        fprintf(stderr, "model: %s\n", err);
        return 1;
    }
    Tokenizer *tk = tokenizer_load(tok_path, err, sizeof(err));
    if (!tk) {
        fprintf(stderr, "tokenizer: %s\n", err);
        bitnet_model_free(&m);
        return 1;
    }
    RunState s;
    if (runstate_init(&s, &m, 0, threads, err, sizeof(err)) != 0) {
        fprintf(stderr, "runstate: %s\n", err);
        tokenizer_free(tk);
        bitnet_model_free(&m);
        return 1;
    }
    ctx_t c = {&m, tk, &s, {tokenizer_token_id(tk, "<|end_of_text|>"), tokenizer_token_id(tk, "<|eot_id|>")}};
    assert(c.stop[0] == 128001 && c.stop[1] == 128009);

    if (test_mode) {
        /* The first forward pass pages the memory-mapped weights in from the
         * page cache / disk; measure it separately so later TTFTs are warm. */
        const uint64_t t0 = platform_now_ns();
        transformer_forward(c.stop[1], 0, &m, &s);
        printf("\nCold start:     first forward pass %.0f ms (pages in %.0f MiB of mapped weights; "
               "warm passes ~30 ms)\n", (platform_now_ns() - t0) / 1e6,
               m.map_size / 1048576.0);
        model_tests(&c, quick);
        benchmark(&c, quick);
    } else if (inter) {
        interactive(&c, &sc, system, max_new);
    } else {
        int32_t *prompt = xmalloc(8192 * sizeof(int32_t)), *out = xmalloc((size_t)max_new * 4);
        const int n = chat ? build_chat_prompt(tk, system, prompt_text, 1, prompt, 8192)
                           : encode_raw(tk, prompt_text, prompt, 8192);
        assert(n > 0 && n <= 8192);
        if (!chat) printf("%s", prompt_text);
        capture cap = {NULL, 0, 0, 1, 0, 0};
        GenerateParams gp = {max_new, c.stop, 2, capture_sink, &cap, out, max_new, NULL};
        Sampler smp;
        assert(sampler_init(&smp, m.config.vocab_size, &sc, err, sizeof(err)) == 0);
        GenerateStats st;
        assert(generate(&m, tk, &s, &smp, prompt, n, 0, &gp, &st) == 0);
        printf("\n");
        print_stats("stats", &st);
        sampler_free(&smp);
        free(prompt);
        free(out);
    }

    runstate_free(&s);
    tokenizer_free(tk);
    bitnet_model_free(&m);
    return 0;
}
