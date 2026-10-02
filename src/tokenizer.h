/*
 * tokenizer.h — Byte-level BPE tokenizer (Llama-3 / BitNet b1.58 2B-4T) in pure C.
 *
 * Loads a Hugging Face `tokenizer.json` and reproduces HF `tokenizers` encoding
 * exactly (verified against tokenizers 0.22.2 golden outputs), with no
 * dependencies beyond libc:
 *
 *   1. Added/special tokens (e.g. "<|eot_id|>") are matched leftmost-longest
 *      and emitted directly (HF default; disable with TOKENIZER_NO_SPECIAL).
 *   2. Remaining text is split by the Llama-3 pre-tokenizer regex
 *        (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
 *        ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 *      implemented by hand over UTF-8 with Unicode 16.0 \p{L}/\p{N} tables.
 *   3. Each piece is looked up whole (ignore_merges) or merged with
 *      rank-ordered BPE over raw bytes (O(n log n) heap, HF tie-breaking).
 *
 * Text is bytes: any input, including invalid UTF-8, round-trips exactly
 * (decode(encode(text)) == text); an invalid byte is pre-tokenized as a single
 * non-letter, non-digit, non-space character. Encoding does not add BOS.
 *
 * Only tokenizer.json with model.type "BPE" + byte-level pre-tokenizer/decoder
 * is supported; SentencePiece `tokenizer.model` files (Llama-2 style) are
 * rejected with an explicit error.
 */
#ifndef BITNET_TOKENIZER_H
#define BITNET_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct Tokenizer Tokenizer;

enum {
    TOKENIZER_NO_SPECIAL = 1 << 0, /* treat "<|...|>" in the text as plain text */
};

/* Loads tokenizer.json. Returns NULL on failure with a message in err. */
Tokenizer *tokenizer_load(const char *path, char *err, size_t err_len);

/* Frees everything; NULL is allowed. */
void tokenizer_free(Tokenizer *tok);

/* Encodes NUL-terminated `text` (special tokens recognized). Writes at most
 * max_tokens ids and returns the total number of tokens the full encoding
 * needs (so a return value > max_tokens means the output was truncated), or
 * -1 on invalid arguments. */
int bpe_encode(const Tokenizer *tok, const char *text, int32_t *tokens, int max_tokens);

/* Same, for `len` bytes (may contain NULs) with TOKENIZER_* flags. */
int bpe_encode_ex(const Tokenizer *tok, const char *text, size_t len, int32_t *tokens,
                  int max_tokens, unsigned flags);

/* Decoded bytes of one token as a NUL-terminated string owned by the
 * tokenizer (valid until tokenizer_free), or NULL for an invalid id. A token
 * may hold a partial UTF-8 sequence (e.g. one byte of an emoji); concatenating
 * consecutive tokens restores the text. The single-byte token for 0x00 decodes
 * to an empty C string — use bpe_decode_bytes when that matters. */
const char *bpe_decode(const Tokenizer *tok, int32_t token);

/* Like bpe_decode, also returning the exact byte length. */
const char *bpe_decode_bytes(const Tokenizer *tok, int32_t token, size_t *len);

/* Decodes a sequence into `out` (always NUL-terminated if out_cap > 0).
 * Returns the full decoded length in bytes (may exceed out_cap - 1), or -1 if
 * an id is invalid. */
long bpe_decode_tokens(const Tokenizer *tok, const int32_t *tokens, int n_tokens,
                       char *out, size_t out_cap);

/* Number of token ids (max id + 1, including added tokens). */
int tokenizer_vocab_size(const Tokenizer *tok);

/* Id of the token whose decoded text is exactly `text` (vocab or added
 * token, e.g. "<|eot_id|>"), or -1. */
int32_t tokenizer_token_id(const Tokenizer *tok, const char *text);

/* 1 if `token` is a special added token (e.g. <|begin_of_text|>), else 0. */
int tokenizer_is_special(const Tokenizer *tok, int32_t token);

#endif /* BITNET_TOKENIZER_H */
