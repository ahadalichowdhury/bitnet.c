/*
 * generate.c — Sampling, UTF-8-safe streaming and the generation loop
 * (see generate.h).
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#include "generate.h"

#include "platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double now_ms(void) { return (double)platform_now_ns() / 1e6; }

/* ========================================================================= */
/* RNG: xoshiro256** seeded with splitmix64                                  */
/* ========================================================================= */

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t xoshiro_next(uint64_t s[4]) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

void sampler_reseed(Sampler *s, uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) s->rng[i] = splitmix64(&x);
}

double sampler_uniform(Sampler *s) { return (double)(xoshiro_next(s->rng) >> 11) * 0x1.0p-53; }

/* ========================================================================= */
/* Sampler                                                                   */
/* ========================================================================= */

typedef struct {
    float   p;
    int32_t i;
} prob_index;

int sampler_init(Sampler *s, int vocab, const SamplerConfig *cfg, char *err, size_t err_len) {
    memset(s, 0, sizeof(*s));
    if (vocab < 1 || !cfg) {
        snprintf(err, err_len, "bad sampler arguments");
        return -1;
    }
    s->cfg = *cfg;
    s->vocab = vocab;
    s->probs = malloc((size_t)vocab * sizeof(float));
    s->idx = malloc((size_t)vocab * sizeof(int32_t));
    s->pairs = malloc((size_t)vocab * sizeof(prob_index));
    s->counts = calloc((size_t)vocab, sizeof(int32_t));
    if (!s->probs || !s->idx || !s->pairs || !s->counts) {
        sampler_free(s);
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    sampler_reseed(s, cfg->seed);
    return 0;
}

void sampler_free(Sampler *s) {
    free(s->probs);
    free(s->idx);
    free(s->pairs);
    free(s->counts);
    memset(s, 0, sizeof(*s));
}

void sampler_reset_history(Sampler *s) {
    s->hist_len = 0;
    s->hist_head = 0;
}

void sampler_accept(Sampler *s, int32_t token) {
    s->hist[s->hist_head] = token;
    s->hist_head = (s->hist_head + 1) % SAMPLER_HISTORY;
    if (s->hist_len < SAMPLER_HISTORY) s->hist_len++;
}

/* Applies repetition / frequency / presence penalties for the tokens in the
 * last penalty_last_n accepted positions. counts[] is all zero on entry and
 * restored to zero on exit, so this is O(window) with no allocation. */
static void apply_penalties(Sampler *s, float *logits) {
    const SamplerConfig *c = &s->cfg;
    const int rep = c->repetition_penalty > 0.0f && c->repetition_penalty != 1.0f;
    if (!rep && c->frequency_penalty == 0.0f && c->presence_penalty == 0.0f) return;
    int n = c->penalty_last_n > 0 ? c->penalty_last_n : SAMPLER_DEFAULT_LAST_N;
    if (n > s->hist_len) n = s->hist_len;
    if (n > SAMPLER_HISTORY) n = SAMPLER_HISTORY;
    const int start = (s->hist_head - n + SAMPLER_HISTORY) % SAMPLER_HISTORY;
    for (int i = 0; i < n; i++) {
        const int32_t t = s->hist[(start + i) % SAMPLER_HISTORY];
        if (t >= 0 && t < s->vocab) s->counts[t]++;
    }
    for (int i = 0; i < n; i++) {
        const int32_t t = s->hist[(start + i) % SAMPLER_HISTORY];
        if (t < 0 || t >= s->vocab || s->counts[t] == 0) continue; /* already handled */
        float l = logits[t];
        if (rep) l = l > 0.0f ? l / c->repetition_penalty : l * c->repetition_penalty;
        l -= c->frequency_penalty * (float)s->counts[t] + c->presence_penalty;
        logits[t] = l;
        s->counts[t] = 0; /* marks t done and restores the all-zero invariant */
    }
}

static int32_t argmax(const float *x, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i; /* strict: lowest id wins ties */
    return best;
}

/* "a ranks below b": smaller value, or equal value and larger id. */
static inline int ranks_below(float va, int32_t ia, float vb, int32_t ib) {
    return va < vb || (va == vb && ia > ib);
}

/* Top-k by a size-k min-heap over (value, id): O(V log k). Fills
 * probs[0..k) / idx[0..k) (heap order, not sorted). */
static int select_top_k(const float *logits, int V, int k, float *hv, int32_t *hi) {
    int n = 0;
    for (int32_t i = 0; i < V; i++) {
        const float v = logits[i];
        if (n < k) {
            int j = n++;
            while (j > 0) {
                const int p = (j - 1) / 2;
                if (!ranks_below(v, i, hv[p], hi[p])) break;
                hv[j] = hv[p]; hi[j] = hi[p];
                j = p;
            }
            hv[j] = v; hi[j] = i;
        } else if (ranks_below(hv[0], hi[0], v, i)) {
            int j = 0;
            for (;;) {
                int c = 2 * j + 1;
                if (c >= k) break;
                if (c + 1 < k && ranks_below(hv[c + 1], hi[c + 1], hv[c], hi[c])) c++;
                if (!ranks_below(hv[c], hi[c], v, i)) break;
                hv[j] = hv[c]; hi[j] = hi[c];
                j = c;
            }
            hv[j] = v; hi[j] = i;
        }
    }
    return n;
}

static int cmp_desc(const void *a, const void *b) {
    const prob_index *x = a, *y = b;
    if (x->p != y->p) return x->p > y->p ? -1 : 1;
    return (x->i > y->i) - (x->i < y->i);
}

static int32_t draw(Sampler *s, const float *p, const int32_t *id, int n, double total) {
    const double r = sampler_uniform(s) * total;
    double cum = 0.0;
    for (int j = 0; j < n; j++) {
        cum += p[j];
        if (r < cum) return id[j];
    }
    return id[n - 1]; /* rounding: r landed at the very top */
}

int32_t sampler_sample(Sampler *s, float *logits) {
    const int V = s->vocab;
    const SamplerConfig *c = &s->cfg;
    apply_penalties(s, logits);
    if (c->temperature <= 0.0f) return argmax(logits, V);

    float *p = s->probs;
    int32_t *id = s->idx;
    int n;
    if (c->top_k > 0 && c->top_k < V) {
        n = select_top_k(logits, V, c->top_k, p, id);
    } else {
        n = V;
        memcpy(p, logits, (size_t)V * sizeof(float));
        for (int32_t i = 0; i < V; i++) id[i] = i;
    }

    /* Temperature + softmax numerators (exp is monotonic, so top-k above was
     * taken on raw logits without changing the result). */
    const float inv_t = 1.0f / c->temperature;
    float mx = p[0];
    for (int j = 1; j < n; j++) mx = p[j] > mx ? p[j] : mx;
    for (int j = 0; j < n; j++) p[j] = (p[j] - mx) * inv_t;
    exp_neon(p, p, n);
    double sum = 0.0;
    for (int j = 0; j < n; j++) sum += p[j];

    if (c->top_p < 1.0f && n > 1) {
        /* Tokens with prob < (1 - top_p) / (n - 1) can never be inside the
         * nucleus (together they hold < 1 - top_p), so only the rest is
         * sorted. */
        prob_index *pairs = s->pairs;
        const double cutoff = (1.0 - c->top_p) / (double)(n - 1) * sum;
        int m = 0;
        for (int j = 0; j < n; j++)
            if (p[j] >= cutoff) {
                pairs[m].p = p[j];
                pairs[m].i = id[j];
                m++;
            }
        qsort(pairs, (size_t)m, sizeof(prob_index), cmp_desc);
        const double target = c->top_p * sum;
        double cum = 0.0;
        int keep = m;
        for (int j = 0; j < m; j++) {
            cum += pairs[j].p;
            if (cum >= target) {
                keep = j + 1;
                break;
            }
        }
        double total = 0.0;
        for (int j = 0; j < keep; j++) {
            p[j] = pairs[j].p;
            id[j] = pairs[j].i;
            total += p[j];
        }
        return draw(s, p, id, keep, total);
    }
    return draw(s, p, id, n, sum);
}

/* ========================================================================= */
/* UTF-8 streaming                                                           */
/* ========================================================================= */

static const char k_replacement[] = "\xEF\xBF\xBD"; /* U+FFFD */

void utf8_stream_init(Utf8Stream *u, text_sink sink, void *user) {
    memset(u, 0, sizeof(*u));
    u->sink = sink;
    u->user = user;
}

/* Length of the sequence started by lead byte b, or 0 if b cannot start one. */
static int seq_len(uint8_t b) {
    if (b < 0x80) return 1;
    if (b >= 0xC2 && b <= 0xDF) return 2;
    if (b >= 0xE0 && b <= 0xEF) return 3;
    if (b >= 0xF0 && b <= 0xF4) return 4;
    return 0;
}

/* Is b valid as continuation byte k (1-based) after `lead`? (RFC 3629:
 * no overlongs, no surrogates, nothing above U+10FFFF.) */
static int cont_ok(uint8_t lead, int k, uint8_t b) {
    if ((b & 0xC0) != 0x80) return 0;
    if (k == 1) {
        if (lead == 0xE0 && b < 0xA0) return 0;
        if (lead == 0xED && b > 0x9F) return 0;
        if (lead == 0xF0 && b < 0x90) return 0;
        if (lead == 0xF4 && b > 0x8F) return 0;
    }
    return 1;
}

static void emit(Utf8Stream *u, const void *p, size_t n) {
    if (n && u->sink) u->sink(u->user, (const char *)p, n);
}

void utf8_stream_write(Utf8Stream *u, const char *bytes, size_t len) {
    const uint8_t *in = (const uint8_t *)bytes;
    size_t i = 0;

    /* Complete (or reject) a character held back from the previous write. */
    while (u->n_pending && i < len) {
        const int need = seq_len(u->pending[0]);
        if (!cont_ok(u->pending[0], u->n_pending, in[i])) {
            emit(u, k_replacement, 3);
            u->n_pending = 0; /* in[i] is re-examined as a fresh byte below */
            break;
        }
        u->pending[u->n_pending++] = in[i++];
        if (u->n_pending == need) {
            emit(u, u->pending, (size_t)need);
            u->n_pending = 0;
        }
    }
    if (u->n_pending) return; /* still incomplete, input exhausted */

    /* Emit maximal runs of valid characters straight from the input. */
    size_t run = i;
    while (i < len) {
        const uint8_t b = in[i];
        const int L = seq_len(b);
        if (L == 1) { i++; continue; }
        if (L == 0) {
            emit(u, in + run, i - run);
            emit(u, k_replacement, 3);
            run = ++i;
            continue;
        }
        int k = 1;
        while (k < L && i + (size_t)k < len && cont_ok(b, k, in[i + k])) k++;
        if (k == L) { i += (size_t)L; continue; }
        emit(u, in + run, i - run);
        if (i + (size_t)k == len) { /* valid prefix cut off by the end: hold it */
            memcpy(u->pending, in + i, (size_t)k);
            u->n_pending = k;
            return;
        }
        emit(u, k_replacement, 3); /* invalid: replace the maximal bad prefix */
        i += (size_t)k;
        run = i;
    }
    emit(u, in + run, len - run);
}

void utf8_stream_flush(Utf8Stream *u) {
    if (u->n_pending) emit(u, k_replacement, 3);
    u->n_pending = 0;
}

/* ========================================================================= */
/* Generation loop                                                           */
/* ========================================================================= */

/* ========================================================================= */
/* Stop strings                                                              */
/* ========================================================================= */

int stop_matcher_init(StopMatcher *m, const char *const *stops, int n_stops, Utf8Stream *out) {
    memset(m, 0, sizeof(*m));
    m->matched = -1;
    m->out = out;
    if (n_stops < 0 || n_stops > STOP_MAX_STRINGS || (n_stops > 0 && !stops)) return -1;
    for (int i = 0; i < n_stops; i++) {
        if (!stops[i]) return -1;
        m->lens[i] = strlen(stops[i]);
        if (m->lens[i] == 0 || m->lens[i] > STOP_MAX_LEN) return -1;
    }
    m->stops = stops;
    m->n_stops = n_stops;
    return 0;
}

/* Bytes of the held text that must stay held: the longest suffix of
 * hold[0..n) that is a proper prefix of some stop string. */
static size_t stop_suffix_keep(const StopMatcher *m, const char *buf, size_t n) {
    size_t keep = 0;
    for (int i = 0; i < m->n_stops; i++) {
        const size_t max = m->lens[i] - 1 < n ? m->lens[i] - 1 : n;
        for (size_t k = max; k > keep; k--)
            if (memcmp(buf + n - k, m->stops[i], k) == 0) {
                keep = k;
                break;
            }
    }
    return keep;
}

int stop_matcher_write(StopMatcher *m, const char *bytes, size_t len) {
    if (m->matched >= 0) return 1;
    if (m->n_stops == 0) {
        utf8_stream_write(m->out, bytes, len);
        return 0;
    }
    /* Work buffer = held bytes + new bytes, processed in bounded windows. */
    char buf[2 * STOP_MAX_LEN + 256];
    size_t i = 0;
    while (i < len) {
        const size_t room = sizeof(buf) - m->n_hold;
        const size_t take = len - i < room ? len - i : room;
        memcpy(buf, m->hold, m->n_hold);
        memcpy(buf + m->n_hold, bytes + i, take);
        const size_t n = m->n_hold + take;
        i += take;

        /* Earliest occurrence of any stop string. */
        size_t best = n;
        for (int s = 0; s < m->n_stops; s++) {
            const size_t L = m->lens[s];
            for (size_t p = 0; p + L <= n && p < best; p++)
                if (memcmp(buf + p, m->stops[s], L) == 0) {
                    if (p < best) {
                        best = p;
                        m->matched = s;
                    }
                    break;
                }
        }
        if (m->matched >= 0) {
            utf8_stream_write(m->out, buf, best);
            m->n_hold = 0;
            return 1;
        }
        const size_t keep = stop_suffix_keep(m, buf, n);
        utf8_stream_write(m->out, buf, n - keep);
        memcpy(m->hold, buf + n - keep, keep);
        m->n_hold = keep;
    }
    return 0;
}

void stop_matcher_flush(StopMatcher *m) {
    if (m->matched < 0 && m->n_hold) utf8_stream_write(m->out, m->hold, m->n_hold);
    m->n_hold = 0;
}

static int is_stop(const GenerateParams *p, int32_t t) {
    for (int i = 0; i < p->n_stop; i++)
        if (p->stop_tokens[i] == t) return 1;
    return 0;
}

int generate(const BitNetModel *m, const Tokenizer *tk, RunState *s, Sampler *smp,
             const int32_t *prompt, int n_prompt, int start_pos, const GenerateParams *p,
             GenerateStats *st) {
    memset(st, 0, sizeof(*st));
    st->stop_token = -1;
    st->last_token = -1;
    st->n_prompt = n_prompt;
    if (n_prompt < 1 || start_pos < 0 || start_pos + n_prompt > s->max_seq_len) return -1;

    Utf8Stream us;
    utf8_stream_init(&us, p->on_text, p->user);
    StopMatcher sm;
    st->stop_string = -1;
    if (stop_matcher_init(&sm, p->stop_strings, p->n_stop_strings, &us) != 0) return -1;
    const int need_text = p->on_text || p->n_stop_strings > 0;

    /* Penalties look back over this call's prompt plus what it generates. */
    sampler_reset_history(smp);
    for (int i = 0; i < n_prompt; i++) sampler_accept(smp, prompt[i]);
    const double t0 = now_ms();

    /* Prefill: every prompt token goes through the KV cache; only the last
     * one needs logits. */
    for (int i = 0; i < n_prompt; i++) {
        if (p->cancel && atomic_load_explicit(p->cancel, memory_order_relaxed)) {
            st->reason = GEN_STOP_CANCELLED;
            st->end_pos = start_pos + i;
            st->prefill_ms = st->total_ms = now_ms() - t0;
            return 0;
        }
        transformer_forward_ex(prompt[i], start_pos + i, m, s, i == n_prompt - 1);
    }
    const double t_prefill = now_ms();
    st->prefill_ms = t_prefill - t0;
    st->prefill_done = 1;

    int pos = start_pos + n_prompt;
    double t_first = t_prefill;
    for (;;) {
        const double ts = now_ms();
        const int32_t tok = sampler_sample(smp, s->logits);
        st->sample_ms += now_ms() - ts;
        if (is_stop(p, tok)) {
            st->reason = GEN_STOP_EOS;
            st->stop_token = tok;
            if (st->n_generated == 0) t_first = now_ms();
            break;
        }
        if (p->out_tokens && st->n_generated < p->out_cap) p->out_tokens[st->n_generated] = tok;
        st->n_generated++;
        st->last_token = tok;
        sampler_accept(smp, tok);
        int hit = 0;
        if (need_text && !tokenizer_is_special(tk, tok)) {
            size_t len;
            const char *bytes = bpe_decode_bytes(tk, tok, &len);
            hit = stop_matcher_write(&sm, bytes, len);
        }
        if (st->n_generated == 1) {
            t_first = now_ms();
            st->ttft_ms = t_first - t0;
        }
        if (hit) { /* like a stop token: the completing token is not fed */
            st->reason = GEN_STOP_STRING;
            st->stop_string = sm.matched;
            break;
        }
        if (st->n_generated >= p->max_new_tokens) {
            st->reason = GEN_STOP_MAX_TOKENS;
            break;
        }
        if (pos >= s->max_seq_len) {
            st->reason = GEN_STOP_CONTEXT_FULL;
            break;
        }
        if (p->cancel && atomic_load_explicit(p->cancel, memory_order_relaxed)) {
            st->reason = GEN_STOP_CANCELLED;
            break;
        }
        transformer_forward(tok, pos, m, s);
        pos++;
    }
    stop_matcher_flush(&sm);
    utf8_stream_flush(&us);

    const double t_end = now_ms();
    if (st->n_generated == 0) st->ttft_ms = t_first - t0;
    st->end_pos = pos;
    st->decode_ms = t_end - t_first;
    st->total_ms = t_end - t0;
    st->prefill_tok_s = st->prefill_ms > 0 ? n_prompt / (st->prefill_ms / 1e3) : 0;
    st->decode_tok_s = st->decode_ms > 0 && st->n_generated > 1
                           ? (st->n_generated - 1) / (st->decode_ms / 1e3) : 0;
    return 0;
}

/* ========================================================================= */
/* Chat prompt                                                               */
/* ========================================================================= */

static int is_ws(char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

/* Encodes "{role}: {trim(content)}" (specials disabled) then <|eot_id|>. */
static int append_turn(const Tokenizer *tk, const char *role, const char *content, int32_t eot,
                       int32_t *tokens, int max, int n) {
    size_t a = 0, b = strlen(content);
    while (a < b && is_ws(content[a])) a++;
    while (b > a && is_ws(content[b - 1])) b--;
    const size_t rl = strlen(role), len = rl + 2 + (b - a);
    char *text = malloc(len + 1);
    if (!text) return -1;
    memcpy(text, role, rl);
    memcpy(text + rl, ": ", 2);
    memcpy(text + rl + 2, content + a, b - a);
    text[len] = '\0';
    const int room = n < max ? max - n : 0;
    const int k = bpe_encode_ex(tk, text, len, room ? tokens + n : NULL, room, TOKENIZER_NO_SPECIAL);
    free(text);
    if (k < 0) return -1;
    n += k;
    if (n < max) tokens[n] = eot;
    return n + 1;
}

int build_chat_prompt(const Tokenizer *tk, const char *system, const char *user, int with_bos,
                      int32_t *tokens, int max_tokens) {
    const int32_t bos = tokenizer_token_id(tk, "<|begin_of_text|>");
    const int32_t eot = tokenizer_token_id(tk, "<|eot_id|>");
    if (bos < 0 || eot < 0 || !user || max_tokens < 0) return -1;
    int n = 0;
    if (with_bos) {
        if (n < max_tokens) tokens[n] = bos;
        n++;
    }
    if (system && (n = append_turn(tk, "System", system, eot, tokens, max_tokens, n)) < 0) return -1;
    if ((n = append_turn(tk, "User", user, eot, tokens, max_tokens, n)) < 0) return -1;
    const int room = n < max_tokens ? max_tokens - n : 0;
    const int k = bpe_encode_ex(tk, "Assistant: ", 11, room ? tokens + n : NULL, room, 0);
    return k < 0 ? -1 : n + k;
}
