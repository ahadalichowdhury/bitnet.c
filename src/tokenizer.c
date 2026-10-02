/*
 * tokenizer.c — Byte-level BPE tokenizer (see tokenizer.h).
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include "tokenizer.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unicode_tables.h"

/* The only pre-tokenizer pattern implemented (Llama-3 / BitNet 2B-4T). */
static const char k_llama3_pattern[] =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

#define MAX_TOKENS_TABLE (1 << 24)
#define BPE_STACK_SYMBOLS 128

static void set_err(char *err, size_t err_len, const char *fmt, ...) {
    if (!err || !err_len) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

/* ========================================================================= */
/* Minimal JSON DOM (strings unescaped in place inside the file buffer)      */
/* ========================================================================= */

enum { J_NULL, J_FALSE, J_TRUE, J_NUM, J_STR, J_ARR, J_OBJ };
#define J_NONE UINT32_MAX
#define J_MAX_DEPTH 64

typedef struct {
    uint8_t  type;
    uint32_t len;   /* string bytes, or number of children (object: pairs) */
    uint32_t child; /* first child (object: first key; key.next = value) */
    uint32_t next;  /* next sibling */
    union {
        const char *s;
        double      num;
    } v;
} jnode;

typedef struct {
    char       *p, *end;
    jnode      *nodes;
    uint32_t    n, cap;
    const char *msg;
} jparser;

static uint32_t j_new(jparser *P, uint8_t type) {
    if (P->n == P->cap) {
        const uint32_t cap = P->cap ? P->cap * 2 : 1024;
        jnode *nn = realloc(P->nodes, (size_t)cap * sizeof(jnode));
        if (!nn) {
            P->msg = "out of memory";
            return J_NONE;
        }
        P->nodes = nn;
        P->cap = cap;
    }
    jnode *n = &P->nodes[P->n];
    memset(n, 0, sizeof(*n));
    n->type = type;
    n->child = n->next = J_NONE;
    return P->n++;
}

static void j_ws(jparser *P) {
    while (P->p < P->end && (*P->p == ' ' || *P->p == '\t' || *P->p == '\n' || *P->p == '\r'))
        P->p++;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int read_hex4(const char *p, const char *end, uint32_t *out) {
    if (end - p < 4) return -1;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        const int h = hexval(p[i]);
        if (h < 0) return -1;
        v = v << 4 | (uint32_t)h;
    }
    *out = v;
    return 0;
}

static int utf8_put(char *w, uint32_t cp) {
    if (cp < 0x80) { w[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        w[0] = (char)(0xC0 | cp >> 6);
        w[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        w[0] = (char)(0xE0 | cp >> 12);
        w[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        w[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    w[0] = (char)(0xF0 | cp >> 18);
    w[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    w[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    w[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* P->p is just past the opening quote. Unescapes in place (output never
 * outruns input) and NUL-terminates. */
static uint32_t j_string(jparser *P) {
    char *start = P->p, *w = P->p;
    for (;;) {
        if (P->p >= P->end) { P->msg = "unterminated string"; return J_NONE; }
        const char c = *P->p++;
        if (c == '"') break;
        if ((unsigned char)c < 0x20) { P->msg = "control character in string"; return J_NONE; }
        if (c != '\\') { *w++ = c; continue; }
        if (P->p >= P->end) { P->msg = "bad escape"; return J_NONE; }
        const char e = *P->p++;
        switch (e) {
        case '"': *w++ = '"'; break;
        case '\\': *w++ = '\\'; break;
        case '/': *w++ = '/'; break;
        case 'b': *w++ = '\b'; break;
        case 'f': *w++ = '\f'; break;
        case 'n': *w++ = '\n'; break;
        case 'r': *w++ = '\r'; break;
        case 't': *w++ = '\t'; break;
        case 'u': {
            uint32_t cp;
            if (read_hex4(P->p, P->end, &cp)) { P->msg = "bad \\u escape"; return J_NONE; }
            P->p += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                uint32_t lo;
                if (P->end - P->p < 6 || P->p[0] != '\\' || P->p[1] != 'u' ||
                    read_hex4(P->p + 2, P->end, &lo) || lo < 0xDC00 || lo > 0xDFFF) {
                    P->msg = "unpaired surrogate";
                    return J_NONE;
                }
                P->p += 6;
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                P->msg = "unpaired surrogate";
                return J_NONE;
            }
            w += utf8_put(w, cp);
            break;
        }
        default: P->msg = "bad escape"; return J_NONE;
        }
    }
    const uint32_t idx = j_new(P, J_STR);
    if (idx == J_NONE) return J_NONE;
    *w = '\0';
    P->nodes[idx].v.s = start;
    P->nodes[idx].len = (uint32_t)(w - start);
    return idx;
}

static uint32_t j_value(jparser *P, int depth);

static uint32_t j_container(jparser *P, int depth, int is_obj) {
    const uint32_t idx = j_new(P, is_obj ? J_OBJ : J_ARR);
    if (idx == J_NONE) return J_NONE;
    const char close = is_obj ? '}' : ']';
    uint32_t last = J_NONE, count = 0;
    j_ws(P);
    if (P->p < P->end && *P->p == close) { P->p++; return idx; }
    for (;;) {
        if (is_obj) {
            j_ws(P);
            if (P->p >= P->end || *P->p != '"') { P->msg = "expected object key"; return J_NONE; }
            P->p++;
            const uint32_t key = j_string(P);
            if (key == J_NONE) return J_NONE;
            j_ws(P);
            if (P->p >= P->end || *P->p != ':') { P->msg = "expected ':'"; return J_NONE; }
            P->p++;
            const uint32_t val = j_value(P, depth + 1);
            if (val == J_NONE) return J_NONE;
            P->nodes[key].next = val;
            if (last != J_NONE) P->nodes[last].next = key;
            else P->nodes[idx].child = key;
            last = val;
        } else {
            const uint32_t val = j_value(P, depth + 1);
            if (val == J_NONE) return J_NONE;
            if (last != J_NONE) P->nodes[last].next = val;
            else P->nodes[idx].child = val;
            last = val;
        }
        count++;
        j_ws(P);
        if (P->p < P->end && *P->p == ',') { P->p++; continue; }
        if (P->p < P->end && *P->p == close) { P->p++; break; }
        P->msg = is_obj ? "expected ',' or '}'" : "expected ',' or ']'";
        return J_NONE;
    }
    P->nodes[idx].len = count;
    return idx;
}

static uint32_t j_value(jparser *P, int depth) {
    if (depth > J_MAX_DEPTH) { P->msg = "nesting too deep"; return J_NONE; }
    j_ws(P);
    if (P->p >= P->end) { P->msg = "unexpected end of input"; return J_NONE; }
    const char c = *P->p;
    if (c == '{') { P->p++; return j_container(P, depth, 1); }
    if (c == '[') { P->p++; return j_container(P, depth, 0); }
    if (c == '"') { P->p++; return j_string(P); }
    if (c == '-' || (c >= '0' && c <= '9')) {
        char *e;
        const double d = strtod(P->p, &e);
        if (e == P->p) { P->msg = "bad number"; return J_NONE; }
        P->p = e;
        const uint32_t idx = j_new(P, J_NUM);
        if (idx != J_NONE) P->nodes[idx].v.num = d;
        return idx;
    }
    static const struct { const char *lit; uint8_t type; } lits[] = {
        {"true", J_TRUE}, {"false", J_FALSE}, {"null", J_NULL}};
    for (size_t i = 0; i < 3; i++) {
        const size_t l = strlen(lits[i].lit);
        if ((size_t)(P->end - P->p) >= l && memcmp(P->p, lits[i].lit, l) == 0) {
            P->p += l;
            return j_new(P, lits[i].type);
        }
    }
    P->msg = "unexpected character";
    return J_NONE;
}

static const jnode *jn(const jparser *P, uint32_t i) { return i == J_NONE ? NULL : &P->nodes[i]; }

/* Object member lookup; returns J_NONE if absent or obj is not an object. */
static uint32_t jget(const jparser *P, uint32_t obj, const char *key) {
    if (obj == J_NONE || P->nodes[obj].type != J_OBJ) return J_NONE;
    const size_t kl = strlen(key);
    for (uint32_t k = P->nodes[obj].child; k != J_NONE; k = P->nodes[P->nodes[k].next].next) {
        const jnode *kn = &P->nodes[k];
        if (kn->len == kl && memcmp(kn->v.s, key, kl) == 0) return kn->next;
    }
    return J_NONE;
}

static int jis_str(const jparser *P, uint32_t i, const char *s) {
    const jnode *n = jn(P, i);
    return n && n->type == J_STR && n->len == strlen(s) && memcmp(n->v.s, s, n->len) == 0;
}

static int jis_null_or_empty(const jparser *P, uint32_t i) {
    const jnode *n = jn(P, i);
    return !n || n->type == J_NULL || (n->type == J_STR && n->len == 0);
}

static int jis_true(const jparser *P, uint32_t i) {
    const jnode *n = jn(P, i);
    return n && n->type == J_TRUE;
}

/* ========================================================================= */
/* GPT-2 byte-level alphabet                                                 */
/* ========================================================================= */

/* bytes_to_unicode(): printable bytes map to themselves, the other 68 bytes
 * to U+0100.. in order. inverse[cp] = byte + 1 (0 = not in the alphabet). */
static void bytelevel_inverse(uint16_t inverse[324]) {
    memset(inverse, 0, 324 * sizeof(uint16_t));
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        const int printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
                              (b >= 0xAE && b <= 0xFF);
        const int cp = printable ? b : 256 + extra++;
        inverse[cp] = (uint16_t)(b + 1);
    }
}

/* Decodes one UTF-8 code point; returns its byte length, or 0 if invalid. */
static int utf8_decode(const uint8_t *s, size_t n, uint32_t *cp) {
    const uint8_t c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    int len;
    uint32_t v, min;
    if ((c & 0xE0) == 0xC0) { len = 2; v = c & 0x1F; min = 0x80; }
    else if ((c & 0xF0) == 0xE0) { len = 3; v = c & 0x0F; min = 0x800; }
    else if ((c & 0xF8) == 0xF0) { len = 4; v = c & 0x07; min = 0x10000; }
    else return 0;
    if ((size_t)len > n) return 0;
    for (int i = 1; i < len; i++) {
        if ((s[i] & 0xC0) != 0x80) return 0;
        v = v << 6 | (s[i] & 0x3F);
    }
    if (v < min || v > 0x10FFFF || (v >= 0xD800 && v <= 0xDFFF)) return 0;
    *cp = v;
    return len;
}

/* Byte-level vocab string -> raw bytes. Returns length or -1. */
static long bytelevel_decode(const uint16_t inverse[324], const char *s, size_t n, uint8_t *out) {
    size_t i = 0;
    long o = 0;
    while (i < n) {
        uint32_t cp;
        const int l = utf8_decode((const uint8_t *)s + i, n - i, &cp);
        if (!l || cp >= 324 || !inverse[cp]) return -1;
        out[o++] = (uint8_t)(inverse[cp] - 1);
        i += (size_t)l;
    }
    return o;
}

/* ========================================================================= */
/* Tokenizer                                                                 */
/* ========================================================================= */

enum { K_ABSENT = 0, K_VOCAB = 1, K_ADDED = 2, K_SPECIAL = 3 };

struct Tokenizer {
    int32_t   n_ids;
    uint32_t *off;    /* [n_ids] offset of decoded bytes in pool */
    uint32_t *len;    /* [n_ids] decoded length */
    uint8_t  *kind;   /* [n_ids] K_* */
    char     *pool;   /* decoded bytes, each NUL-terminated */

    uint32_t *vocab_slots; /* open addressing, id + 1 (0 = empty); model vocab only */
    uint32_t  vocab_mask;

    uint64_t *merge_keys;  /* (left << 32 | right), UINT64_MAX = empty */
    uint32_t *merge_rank;
    int32_t  *merge_out;
    uint32_t  merge_mask;

    int32_t   byte_id[256];
    int       ignore_merges;

    int32_t  *added;        /* added token ids grouped by first byte, longest first */
    uint32_t  added_start[257];
};

static uint64_t hash_bytes(const uint8_t *p, size_t n) {
    uint64_t h = 1469598103934665603ull; /* FNV-1a */
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h ^ (h >> 29);
}

static uint64_t mix64(uint64_t x) { /* splitmix64 finalizer */
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27; x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static uint32_t pow2_at_least(uint64_t n) {
    uint32_t c = 16;
    while (c < n) c <<= 1;
    return c;
}

static int32_t vocab_find(const Tokenizer *t, const uint8_t *s, size_t n) {
    for (uint32_t h = (uint32_t)hash_bytes(s, n) & t->vocab_mask;; h = (h + 1) & t->vocab_mask) {
        const uint32_t v = t->vocab_slots[h];
        if (!v) return -1;
        const int32_t id = (int32_t)(v - 1);
        if (t->len[id] == n && memcmp(t->pool + t->off[id], s, n) == 0) return id;
    }
}

static void vocab_insert(Tokenizer *t, int32_t id) {
    const uint8_t *s = (const uint8_t *)t->pool + t->off[id];
    uint32_t h = (uint32_t)hash_bytes(s, t->len[id]) & t->vocab_mask;
    while (t->vocab_slots[h]) h = (h + 1) & t->vocab_mask;
    t->vocab_slots[h] = (uint32_t)id + 1;
}

static inline int merge_find(const Tokenizer *t, int32_t l, int32_t r, uint32_t *rank, int32_t *out) {
    const uint64_t key = (uint64_t)(uint32_t)l << 32 | (uint32_t)r;
    for (uint32_t h = (uint32_t)mix64(key) & t->merge_mask;; h = (h + 1) & t->merge_mask) {
        if (t->merge_keys[h] == UINT64_MAX) return 0;
        if (t->merge_keys[h] == key) {
            *rank = t->merge_rank[h];
            *out = t->merge_out[h];
            return 1;
        }
    }
}

/* Inserts unless present (first = lowest rank wins). */
static void merge_insert(Tokenizer *t, int32_t l, int32_t r, uint32_t rank, int32_t out) {
    const uint64_t key = (uint64_t)(uint32_t)l << 32 | (uint32_t)r;
    uint32_t h = (uint32_t)mix64(key) & t->merge_mask;
    while (t->merge_keys[h] != UINT64_MAX) {
        if (t->merge_keys[h] == key) return;
        h = (h + 1) & t->merge_mask;
    }
    t->merge_keys[h] = key;
    t->merge_rank[h] = rank;
    t->merge_out[h] = out;
}

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    free(t->off); free(t->len); free(t->kind); free(t->pool);
    free(t->vocab_slots); free(t->merge_keys); free(t->merge_rank); free(t->merge_out);
    free(t->added);
    free(t);
}

static char *read_whole_file(const char *path, size_t *size, char *err, size_t err_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { set_err(err, err_len, "open %s: %s", path, strerror(errno)); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); set_err(err, err_len, "seek failed"); return NULL; }
    const long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); set_err(err, err_len, "seek failed"); return NULL; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); set_err(err, err_len, "out of memory"); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f); free(buf);
        set_err(err, err_len, "read %s failed", path);
        return NULL;
    }
    fclose(f);
    buf[n] = '\0';
    *size = (size_t)n;
    return buf;
}

/* Checks pre_tokenizer / normalizer / decoder are exactly what we implement. */
static int check_pipeline(const jparser *P, uint32_t root, char *err, size_t err_len) {
    const uint32_t norm = jget(P, root, "normalizer");
    if (norm != J_NONE && P->nodes[norm].type != J_NULL) {
        set_err(err, err_len, "unsupported: normalizer must be null");
        return -1;
    }
    const uint32_t pre = jget(P, root, "pre_tokenizer");
    const uint32_t seq = jget(P, pre, "pretokenizers");
    const jnode *sq = jn(P, seq);
    int ok = jis_str(P, jget(P, pre, "type"), "Sequence") && sq && sq->type == J_ARR && sq->len == 2;
    if (ok) {
        const uint32_t split = sq->child, bl = P->nodes[split].next;
        ok = jis_str(P, jget(P, split, "type"), "Split") &&
             jis_str(P, jget(P, jget(P, split, "pattern"), "Regex"), k_llama3_pattern) &&
             jis_str(P, jget(P, split, "behavior"), "Isolated") &&
             !jis_true(P, jget(P, split, "invert")) &&
             jis_str(P, jget(P, bl, "type"), "ByteLevel") &&
             !jis_true(P, jget(P, bl, "add_prefix_space")) &&
             !jis_true(P, jget(P, bl, "use_regex"));
    }
    if (!ok) {
        set_err(err, err_len, "unsupported pre_tokenizer: only the Llama-3 byte-level Split "
                "pattern is implemented");
        return -1;
    }
    if (!jis_str(P, jget(P, jget(P, root, "decoder"), "type"), "ByteLevel")) {
        set_err(err, err_len, "unsupported decoder: expected ByteLevel");
        return -1;
    }
    return 0;
}

static int cmp_added(const void *a, const void *b, void *ctx_) {
    const Tokenizer *t = ctx_;
    const int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    const uint8_t fx = (uint8_t)t->pool[t->off[x]], fy = (uint8_t)t->pool[t->off[y]];
    if (fx != fy) return fx < fy ? -1 : 1;
    if (t->len[x] != t->len[y]) return t->len[x] > t->len[y] ? -1 : 1; /* longest first */
    return (x > y) - (x < y);
}

/* Small insertion sort with context (qsort_r differs between BSD and glibc). */
static void sort_added(Tokenizer *t, int32_t *ids, int n) {
    for (int i = 1; i < n; i++) {
        const int32_t v = ids[i];
        int j = i - 1;
        while (j >= 0 && cmp_added(&ids[j], &v, t) > 0) { ids[j + 1] = ids[j]; j--; }
        ids[j + 1] = v;
    }
}

Tokenizer *tokenizer_load(const char *path, char *err, size_t err_len) {
    size_t size = 0;
    char *buf = read_whole_file(path, &size, err, err_len);
    if (!buf) return NULL;

    jparser P = {buf, buf + size, NULL, 0, 0, NULL};
    Tokenizer *t = NULL;
    uint8_t *tmp = NULL;

    j_ws(&P);
    if (P.p >= P.end || *P.p != '{') {
        set_err(err, err_len, "%s is not a tokenizer.json (SentencePiece tokenizer.model files "
                "are not supported)", path);
        goto fail;
    }
    const uint32_t root = j_value(&P, 0);
    if (root == J_NONE) {
        set_err(err, err_len, "JSON parse error at byte %ld: %s", (long)(P.p - buf), P.msg);
        goto fail;
    }

    const uint32_t model = jget(&P, root, "model");
    if (!jis_str(&P, jget(&P, model, "type"), "BPE")) {
        set_err(err, err_len, "unsupported model type (only BPE)");
        goto fail;
    }
    if (jis_true(&P, jget(&P, model, "byte_fallback")) ||
        !jis_null_or_empty(&P, jget(&P, model, "continuing_subword_prefix")) ||
        !jis_null_or_empty(&P, jget(&P, model, "end_of_word_suffix")) ||
        (jget(&P, model, "dropout") != J_NONE && P.nodes[jget(&P, model, "dropout")].type != J_NULL)) {
        set_err(err, err_len, "unsupported BPE options (byte_fallback / subword affixes / dropout)");
        goto fail;
    }
    if (check_pipeline(&P, root, err, err_len) != 0) goto fail;

    const uint32_t vocab = jget(&P, model, "vocab");
    const uint32_t merges = jget(&P, model, "merges");
    const uint32_t added = jget(&P, root, "added_tokens");
    if (!jn(&P, vocab) || P.nodes[vocab].type != J_OBJ || !jn(&P, merges) ||
        P.nodes[merges].type != J_ARR || (added != J_NONE && P.nodes[added].type != J_ARR)) {
        set_err(err, err_len, "missing model.vocab / model.merges / added_tokens");
        goto fail;
    }

    /* Size the id table and string pool. */
    int64_t max_id = -1;
    size_t pool_bytes = 0, max_str = 0;
    for (uint32_t k = P.nodes[vocab].child; k != J_NONE; k = P.nodes[P.nodes[k].next].next) {
        const jnode *v = &P.nodes[P.nodes[k].next];
        if (v->type != J_NUM || v->v.num < 0 || v->v.num >= MAX_TOKENS_TABLE ||
            v->v.num != (double)(int64_t)v->v.num) {
            set_err(err, err_len, "vocab id out of range");
            goto fail;
        }
        if ((int64_t)v->v.num > max_id) max_id = (int64_t)v->v.num;
        pool_bytes += P.nodes[k].len + 1;
        if (P.nodes[k].len > max_str) max_str = P.nodes[k].len;
    }
    for (uint32_t a = added == J_NONE ? J_NONE : P.nodes[added].child; a != J_NONE; a = P.nodes[a].next) {
        const jnode *id = jn(&P, jget(&P, a, "id")), *c = jn(&P, jget(&P, a, "content"));
        if (!id || id->type != J_NUM || id->v.num < 0 || id->v.num >= MAX_TOKENS_TABLE ||
            !c || c->type != J_STR || c->len == 0) {
            set_err(err, err_len, "malformed added_tokens entry");
            goto fail;
        }
        if (jis_true(&P, jget(&P, a, "lstrip")) || jis_true(&P, jget(&P, a, "rstrip")) ||
            jis_true(&P, jget(&P, a, "single_word"))) {
            set_err(err, err_len, "unsupported added token options (lstrip/rstrip/single_word) on %s",
                    c->v.s);
            goto fail;
        }
        if ((int64_t)id->v.num > max_id) max_id = (int64_t)id->v.num;
        pool_bytes += c->len + 1;
        if (c->len > max_str) max_str = c->len;
    }
    if (max_id < 255 || pool_bytes >= UINT32_MAX) {
        set_err(err, err_len, "vocabulary too small or too large");
        goto fail;
    }

    t = calloc(1, sizeof(*t));
    if (!t) { set_err(err, err_len, "out of memory"); goto fail; }
    t->n_ids = (int32_t)(max_id + 1);
    t->off = calloc((size_t)t->n_ids, sizeof(uint32_t));
    t->len = calloc((size_t)t->n_ids, sizeof(uint32_t));
    t->kind = calloc((size_t)t->n_ids, 1);
    t->pool = malloc(pool_bytes);
    t->vocab_mask = pow2_at_least((uint64_t)P.nodes[vocab].len * 2) - 1;
    t->vocab_slots = calloc((size_t)t->vocab_mask + 1, sizeof(uint32_t));
    tmp = malloc(max_str * 2 + 2);
    if (!t->off || !t->len || !t->kind || !t->pool || !t->vocab_slots || !tmp) {
        set_err(err, err_len, "out of memory");
        goto fail;
    }
    t->ignore_merges = jis_true(&P, jget(&P, model, "ignore_merges"));

    uint16_t inverse[324];
    bytelevel_inverse(inverse);

    /* Model vocabulary: byte-level strings -> raw bytes. */
    uint32_t pool_used = 0;
    for (uint32_t k = P.nodes[vocab].child; k != J_NONE; k = P.nodes[P.nodes[k].next].next) {
        const jnode *key = &P.nodes[k];
        const int32_t id = (int32_t)P.nodes[key->next].v.num;
        if (t->kind[id] != K_ABSENT) { set_err(err, err_len, "duplicate vocab id %d", id); goto fail; }
        const long n = bytelevel_decode(inverse, key->v.s, key->len, (uint8_t *)t->pool + pool_used);
        if (n < 0) {
            set_err(err, err_len, "vocab token %d is not in the byte-level alphabet", id);
            goto fail;
        }
        if (vocab_find(t, (const uint8_t *)t->pool + pool_used, (size_t)n) >= 0) {
            set_err(err, err_len, "duplicate vocab string for id %d", id);
            goto fail;
        }
        t->off[id] = pool_used;
        t->len[id] = (uint32_t)n;
        t->kind[id] = K_VOCAB;
        t->pool[pool_used + (uint32_t)n] = '\0';
        pool_used += (uint32_t)n + 1;
        vocab_insert(t, id);
    }

    /* Added tokens: stored verbatim (not byte-level encoded). */
    int n_added = 0;
    for (uint32_t a = added == J_NONE ? J_NONE : P.nodes[added].child; a != J_NONE; a = P.nodes[a].next) {
        const jnode *c = jn(&P, jget(&P, a, "content"));
        const int32_t id = (int32_t)P.nodes[jget(&P, a, "id")].v.num;
        const uint8_t kind = jis_true(&P, jget(&P, a, "special")) ? K_SPECIAL : K_ADDED;
        if (t->kind[id] == K_VOCAB) {
            /* Allowed only if it names the same bytes (e.g. GPT-2 <|endoftext|>). */
            if (t->len[id] != c->len || memcmp(t->pool + t->off[id], c->v.s, c->len) != 0) {
                set_err(err, err_len, "added token %d conflicts with vocab entry", id);
                goto fail;
            }
        } else if (t->kind[id] != K_ABSENT) {
            set_err(err, err_len, "duplicate added token id %d", id);
            goto fail;
        } else {
            memcpy(t->pool + pool_used, c->v.s, c->len);
            t->off[id] = pool_used;
            t->len[id] = c->len;
            t->pool[pool_used + c->len] = '\0';
            pool_used += c->len + 1;
        }
        t->kind[id] = kind;
        n_added++;
    }

    for (int b = 0; b < 256; b++) {
        const uint8_t byte = (uint8_t)b;
        t->byte_id[b] = vocab_find(t, &byte, 1);
        if (t->byte_id[b] < 0) {
            set_err(err, err_len, "vocab lacks single-byte token 0x%02x (not byte-level)", b);
            goto fail;
        }
    }

    /* Merges: "a b" strings or ["a", "b"] pairs; rank = position. */
    const uint32_t n_merges = P.nodes[merges].len;
    t->merge_mask = pow2_at_least((uint64_t)n_merges * 2) - 1;
    t->merge_keys = malloc(((size_t)t->merge_mask + 1) * sizeof(uint64_t));
    t->merge_rank = malloc(((size_t)t->merge_mask + 1) * sizeof(uint32_t));
    t->merge_out = malloc(((size_t)t->merge_mask + 1) * sizeof(int32_t));
    if (!t->merge_keys || !t->merge_rank || !t->merge_out) { set_err(err, err_len, "out of memory"); goto fail; }
    memset(t->merge_keys, 0xFF, ((size_t)t->merge_mask + 1) * sizeof(uint64_t));

    uint32_t rank = 0;
    for (uint32_t m = P.nodes[merges].child; m != J_NONE; m = P.nodes[m].next, rank++) {
        const jnode *mn = &P.nodes[m];
        const char *a, *b;
        size_t al, bl;
        if (mn->type == J_STR) {
            const char *sp = memchr(mn->v.s, ' ', mn->len);
            if (!sp) { set_err(err, err_len, "merge %u has no separator", rank); goto fail; }
            a = mn->v.s; al = (size_t)(sp - a);
            b = sp + 1;  bl = mn->len - al - 1;
        } else if (mn->type == J_ARR && mn->len == 2 && P.nodes[mn->child].type == J_STR &&
                   P.nodes[P.nodes[mn->child].next].type == J_STR) {
            const jnode *x = &P.nodes[mn->child], *y = &P.nodes[x->next];
            a = x->v.s; al = x->len; b = y->v.s; bl = y->len;
        } else {
            set_err(err, err_len, "merge %u is malformed", rank);
            goto fail;
        }
        const long la = bytelevel_decode(inverse, a, al, tmp);
        const long lb = la < 0 ? -1 : bytelevel_decode(inverse, b, bl, tmp + la);
        if (la <= 0 || lb <= 0) { set_err(err, err_len, "merge %u: bad token text", rank); goto fail; }
        const int32_t l = vocab_find(t, tmp, (size_t)la);
        const int32_t r = vocab_find(t, tmp + la, (size_t)lb);
        const int32_t o = vocab_find(t, tmp, (size_t)(la + lb));
        if (l < 0 || r < 0 || o < 0) {
            set_err(err, err_len, "merge %u references a token missing from the vocab", rank);
            goto fail;
        }
        merge_insert(t, l, r, rank, o);
    }

    /* Added-token match table: bucketed by first byte, longest first. */
    t->added = malloc((size_t)(n_added ? n_added : 1) * sizeof(int32_t));
    if (!t->added) { set_err(err, err_len, "out of memory"); goto fail; }
    int na = 0;
    for (int32_t id = 0; id < t->n_ids; id++)
        if (t->kind[id] == K_ADDED || t->kind[id] == K_SPECIAL) t->added[na++] = id;
    sort_added(t, t->added, na);
    memset(t->added_start, 0, sizeof(t->added_start));
    for (int i = 0; i < na; i++) t->added_start[(uint8_t)t->pool[t->off[t->added[i]]] + 1]++;
    for (int b = 0; b < 256; b++) t->added_start[b + 1] += t->added_start[b];

    free(tmp);
    free(P.nodes);
    free(buf);
    return t;

fail:
    tokenizer_free(t);
    free(tmp);
    free(P.nodes);
    free(buf);
    return NULL;
}

/* ========================================================================= */
/* Pre-tokenizer (Llama-3 regex, hand-compiled)                              */
/* ========================================================================= */

enum { CL_OTHER = 0, CL_LETTER = 1, CL_NUMBER = 2, CL_SPACE = 3 };

static int in_ranges(const uint32_t (*r)[2], int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid - 1;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

static int cp_class(uint32_t cp) {
    if (cp < 0x80) {
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return CL_LETTER;
        if (cp >= '0' && cp <= '9') return CL_NUMBER;
        if ((cp >= 0x09 && cp <= 0x0D) || cp == ' ') return CL_SPACE;
        return CL_OTHER;
    }
    if (cp == UINT32_MAX) return CL_OTHER; /* invalid UTF-8 byte */
    /* Unicode White_Space (what \s matches in the HF regex engine). */
    if (cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000)
        return CL_SPACE;
    if (in_ranges(k_unicode_letter, UNICODE_LETTER_RANGES, cp)) return CL_LETTER;
    if (in_ranges(k_unicode_number, UNICODE_NUMBER_RANGES, cp)) return CL_NUMBER;
    return CL_OTHER;
}

/* Next code point at s (n > 0): sets *cp (UINT32_MAX for an invalid byte) and
 * returns its length (1 for invalid). */
static inline int next_cp(const uint8_t *s, size_t n, uint32_t *cp) {
    if (s[0] < 0x80) { *cp = s[0]; return 1; }
    const int l = utf8_decode(s, n, cp);
    if (l) return l;
    *cp = UINT32_MAX;
    return 1;
}

/* Simple case folding for the contraction letters: (?i) also maps U+017F
 * LATIN SMALL LETTER LONG S to 's' (verified against HF). */
static inline uint32_t fold(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp == 0x17F) return 's';
    return cp;
}

static size_t run_of(const uint8_t *s, size_t n, int cls) {
    size_t i = 0;
    while (i < n) {
        uint32_t cp;
        const int l = next_cp(s + i, n - i, &cp);
        if (cp_class(cp) != cls) break;
        i += (size_t)l;
    }
    return i;
}

/* Length in bytes of the pre-token starting at s (n > 0). Alternatives are
 * tried in regex order; each mirrors the backtracking result of the pattern. */
static size_t pretoken_len(const uint8_t *s, size_t n) {
    uint32_t c0;
    const size_t l0 = (size_t)next_cp(s, n, &c0);
    const int k0 = cp_class(c0);

    /* 1. (?i:'s|'t|'re|'ve|'m|'ll|'d) */
    if (c0 == '\'' && n > 1) {
        uint32_t c1;
        const size_t l1 = (size_t)next_cp(s + 1, n - 1, &c1);
        const uint32_t f1 = fold(c1);
        if (f1 == 's' || f1 == 't' || f1 == 'm' || f1 == 'd') return 1 + l1;
        if ((f1 == 'r' || f1 == 'v' || f1 == 'l') && n > 1 + l1) {
            uint32_t c2;
            const size_t l2 = (size_t)next_cp(s + 1 + l1, n - 1 - l1, &c2);
            const uint32_t f2 = fold(c2);
            if (((f1 == 'r' || f1 == 'v') && f2 == 'e') || (f1 == 'l' && f2 == 'l'))
                return 1 + l1 + l2;
        }
    }

    /* 2. [^\r\n\p{L}\p{N}]?\p{L}+ */
    if (k0 == CL_LETTER) return l0 + run_of(s + l0, n - l0, CL_LETTER);
    if (k0 != CL_NUMBER && c0 != '\r' && c0 != '\n' && l0 < n) {
        const size_t r = run_of(s + l0, n - l0, CL_LETTER);
        if (r) return l0 + r;
    }

    /* 3. \p{N}{1,3} */
    if (k0 == CL_NUMBER) {
        size_t i = l0;
        for (int k = 1; k < 3 && i < n; k++) {
            uint32_t cp;
            const int l = next_cp(s + i, n - i, &cp);
            if (cp_class(cp) != CL_NUMBER) break;
            i += (size_t)l;
        }
        return i;
    }

    /* 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*` */
    {
        size_t i = 0;
        if (c0 == ' ' && n > 1) {
            uint32_t c1;
            next_cp(s + 1, n - 1, &c1);
            if (cp_class(c1) == CL_OTHER) i = 1;
        }
        const size_t r = run_of(s + i, n - i, CL_OTHER);
        if (r) {
            i += r;
            while (i < n && (s[i] == '\r' || s[i] == '\n')) i++;
            return i;
        }
    }

    /* 5-7. Whitespace run W (k0 is CL_SPACE here). */
    size_t i = 0, last_cp_start = 0, after_last_nl = 0, ncp = 0;
    while (i < n) {
        uint32_t cp;
        const int l = next_cp(s + i, n - i, &cp);
        if (cp_class(cp) != CL_SPACE) break;
        last_cp_start = i;
        i += (size_t)l;
        ncp++;
        if (cp == '\r' || cp == '\n') after_last_nl = i;
    }
    if (after_last_nl) return after_last_nl; /* 5. \s*[\r\n]+  */
    if (i == n) return i;                    /* 6. \s+(?!\S) at end of text */
    if (ncp >= 2) return last_cp_start;      /* 6. ... leaving one space for the next word */
    return i ? i : l0;                       /* 7. \s+ */
}

/* ========================================================================= */
/* BPE                                                                       */
/* ========================================================================= */

typedef struct {
    int32_t *tokens;
    int      max;
    long     n;
    int      oom;
} out_buf;

static inline void emit(out_buf *o, int32_t id) {
    if (o->n < o->max) o->tokens[o->n] = id;
    o->n++;
}

typedef struct {
    uint32_t rank;
    uint32_t pos;
    int32_t  left, right, merged;
} bpe_cand;

static inline int cand_less(const bpe_cand *a, const bpe_cand *b) {
    return a->rank != b->rank ? a->rank < b->rank : a->pos < b->pos;
}

static void heap_push(bpe_cand *h, uint32_t *n, bpe_cand c) {
    uint32_t i = (*n)++;
    while (i > 0) {
        const uint32_t p = (i - 1) / 2;
        if (!cand_less(&c, &h[p])) break;
        h[i] = h[p];
        i = p;
    }
    h[i] = c;
}

static bpe_cand heap_pop(bpe_cand *h, uint32_t *n) {
    const bpe_cand top = h[0];
    const bpe_cand last = h[--(*n)];
    uint32_t i = 0;
    for (;;) {
        uint32_t c = 2 * i + 1;
        if (c >= *n) break;
        if (c + 1 < *n && cand_less(&h[c + 1], &h[c])) c++;
        if (!cand_less(&h[c], &last)) break;
        h[i] = h[c];
        i = c;
    }
    if (*n) h[i] = last;
    return top;
}

static void push_pair(const Tokenizer *t, bpe_cand *heap, uint32_t *hn, const int32_t *id,
                      uint32_t pos, uint32_t next) {
    uint32_t rank;
    int32_t merged;
    if (merge_find(t, id[pos], id[next], &rank, &merged))
        heap_push(heap, hn, (bpe_cand){rank, pos, id[pos], id[next], merged});
}

/* Rank-ordered BPE over the bytes of one pre-token (HF merge_all semantics:
 * lowest rank first, ties to the leftmost pair, stale entries skipped). */
static void bpe_piece(const Tokenizer *t, const uint8_t *s, size_t n, out_buf *o) {
    if (n == 1) { emit(o, t->byte_id[s[0]]); return; }
    if (t->ignore_merges) {
        const int32_t id = vocab_find(t, s, n);
        if (id >= 0) { emit(o, id); return; }
    }

    int32_t  st_id[BPE_STACK_SYMBOLS], st_next[BPE_STACK_SYMBOLS], st_prev[BPE_STACK_SYMBOLS];
    bpe_cand st_heap[3 * BPE_STACK_SYMBOLS];
    int32_t *id = st_id, *next = st_next, *prev = st_prev;
    bpe_cand *heap = st_heap;
    void *block = NULL;
    if (n > BPE_STACK_SYMBOLS) {
        if (n > UINT32_MAX / 4) { o->oom = 1; return; }
        block = malloc(n * 3 * sizeof(int32_t) + 3 * n * sizeof(bpe_cand));
        if (!block) { o->oom = 1; return; }
        heap = (bpe_cand *)block;
        id = (int32_t *)(heap + 3 * n);
        next = id + n;
        prev = next + n;
    }

    for (size_t i = 0; i < n; i++) {
        id[i] = t->byte_id[s[i]];
        next[i] = i + 1 < n ? (int32_t)(i + 1) : -1;
        prev[i] = (int32_t)i - 1;
    }
    uint32_t hn = 0;
    for (uint32_t i = 0; i + 1 < n; i++) push_pair(t, heap, &hn, id, i, i + 1);

    while (hn) {
        const bpe_cand c = heap_pop(heap, &hn);
        const int32_t r = next[c.pos];
        if (id[c.pos] != c.left || r < 0 || id[r] != c.right) continue; /* stale */
        id[c.pos] = c.merged;
        id[r] = -1;
        next[c.pos] = next[r];
        if (next[r] >= 0) prev[next[r]] = (int32_t)c.pos;
        if (prev[c.pos] >= 0) push_pair(t, heap, &hn, id, (uint32_t)prev[c.pos], c.pos);
        if (next[c.pos] >= 0) push_pair(t, heap, &hn, id, c.pos, (uint32_t)next[c.pos]);
    }
    for (int32_t i = 0; i >= 0; i = next[i]) emit(o, id[i]);
    free(block);
}

static void encode_text(const Tokenizer *t, const uint8_t *s, size_t n, out_buf *o) {
    size_t i = 0;
    while (i < n && !o->oom) {
        const size_t l = pretoken_len(s + i, n - i);
        bpe_piece(t, s + i, l, o);
        i += l;
    }
}

/* Longest added token at s, or -1. */
static int32_t match_added(const Tokenizer *t, const uint8_t *s, size_t n, unsigned flags,
                           size_t *mlen) {
    for (uint32_t k = t->added_start[s[0]]; k < t->added_start[s[0] + 1]; k++) {
        const int32_t id = t->added[k];
        if ((flags & TOKENIZER_NO_SPECIAL) && t->kind[id] == K_SPECIAL) continue;
        const size_t l = t->len[id];
        if (l <= n && memcmp(t->pool + t->off[id], s, l) == 0) {
            *mlen = l;
            return id;
        }
    }
    return -1;
}

int bpe_encode_ex(const Tokenizer *t, const char *text, size_t len, int32_t *tokens,
                  int max_tokens, unsigned flags) {
    if (!t || (!text && len) || max_tokens < 0 || (!tokens && max_tokens > 0)) return -1;
    const uint8_t *s = (const uint8_t *)text;
    out_buf o = {tokens, max_tokens, 0, 0};
    size_t seg = 0, i = 0;
    while (i < len && !o.oom) {
        size_t mlen;
        const int32_t id = t->added_start[s[i]] == t->added_start[s[i] + 1]
                               ? -1 : match_added(t, s + i, len - i, flags, &mlen);
        if (id < 0) { i++; continue; }
        encode_text(t, s + seg, i - seg, &o);
        emit(&o, id);
        i += mlen;
        seg = i;
    }
    if (!o.oom) encode_text(t, s + seg, len - seg, &o);
    if (o.oom || o.n > INT_MAX) return -1;
    return (int)o.n;
}

int bpe_encode(const Tokenizer *t, const char *text, int32_t *tokens, int max_tokens) {
    if (!text) return -1;
    return bpe_encode_ex(t, text, strlen(text), tokens, max_tokens, 0);
}

/* ========================================================================= */
/* Decoding and lookup                                                       */
/* ========================================================================= */

const char *bpe_decode_bytes(const Tokenizer *t, int32_t token, size_t *len) {
    if (!t || token < 0 || token >= t->n_ids || t->kind[token] == K_ABSENT) return NULL;
    if (len) *len = t->len[token];
    return t->pool + t->off[token];
}

const char *bpe_decode(const Tokenizer *t, int32_t token) {
    return bpe_decode_bytes(t, token, NULL);
}

long bpe_decode_tokens(const Tokenizer *t, const int32_t *tokens, int n_tokens, char *out,
                       size_t out_cap) {
    if (!t || n_tokens < 0 || (!tokens && n_tokens)) return -1;
    size_t total = 0;
    for (int i = 0; i < n_tokens; i++) {
        size_t l;
        const char *p = bpe_decode_bytes(t, tokens[i], &l);
        if (!p) return -1;
        if (out_cap && total < out_cap - 1) {
            const size_t c = l < out_cap - 1 - total ? l : out_cap - 1 - total;
            memcpy(out + total, p, c);
        }
        total += l;
    }
    if (out_cap) out[total < out_cap - 1 ? total : out_cap - 1] = '\0';
    return total > LONG_MAX ? -1 : (long)total;
}

int tokenizer_vocab_size(const Tokenizer *t) { return t ? t->n_ids : 0; }

int32_t tokenizer_token_id(const Tokenizer *t, const char *text) {
    if (!t || !text) return -1;
    const size_t n = strlen(text);
    for (uint32_t k = 0; k < t->added_start[256]; k++) {
        const int32_t id = t->added[k];
        if (t->len[id] == n && memcmp(t->pool + t->off[id], text, n) == 0) return id;
    }
    return vocab_find(t, (const uint8_t *)text, n);
}

int tokenizer_is_special(const Tokenizer *t, int32_t token) {
    return t && token >= 0 && token < t->n_ids && t->kind[token] == K_SPECIAL;
}
