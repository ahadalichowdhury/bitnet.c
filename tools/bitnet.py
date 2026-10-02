#!/usr/bin/env python3
"""
bitnet.py — Minimal, dependency-free Python bindings for bitnet.c (ctypes).

Wraps the shared library built by `make lib` (build/libbitnet.dylib on macOS,
build/libbitnet.so on Linux). Override the location with $BITNET_LIB.

    from bitnet import BitNet

    with BitNet() as llm:                                   # default model paths
        for piece in llm.generate("The capital of France is", max_new_tokens=16):
            print(piece, end="", flush=True)                # streamed, complete UTF-8
        print(llm.chat("Name three primary colors."))       # multi-turn chat
        print(llm.stats)                                    # TTFT, tok/s, ...

Streaming runs the C generation on a worker thread and yields pieces as they
arrive; leaving the loop early (or Ctrl+C) cancels the generation cleanly.

Self-test:  python3 tools/bitnet.py
"""

import ctypes
import os
import queue
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "models", "bitnet_2b4t.bitnet")
DEFAULT_TOKENIZER = os.path.join(ROOT, "models", "hf", "bitnet-b1.58-2B-4T", "tokenizer.json")

STOP_REASONS = ["end_of_turn", "max_tokens", "context_full", "cancelled", "error"]


# ---------------------------------------------------------------------------
# C declarations (mirror include/bitnet.h)
# ---------------------------------------------------------------------------

class _Config(ctypes.Structure):
    _fields_ = [("n_threads", ctypes.c_int), ("max_seq_len", ctypes.c_int),
                ("system_prompt", ctypes.c_char_p)]


class _Params(ctypes.Structure):
    _fields_ = [("temperature", ctypes.c_float), ("top_k", ctypes.c_int), ("top_p", ctypes.c_float),
                ("seed", ctypes.c_uint64), ("max_new_tokens", ctypes.c_int)]


class _Stats(ctypes.Structure):
    _fields_ = [("stop_reason", ctypes.c_int), ("prompt_tokens", ctypes.c_int),
                ("generated_tokens", ctypes.c_int), ("context_used", ctypes.c_int),
                ("context_size", ctypes.c_int), ("context_reset", ctypes.c_int),
                ("prefill_ms", ctypes.c_double), ("ttft_ms", ctypes.c_double),
                ("decode_ms", ctypes.c_double), ("prefill_tok_s", ctypes.c_double),
                ("decode_tok_s", ctypes.c_double), ("seed", ctypes.c_uint64)]


_TOKEN_FN = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


def _find_library():
    if os.environ.get("BITNET_LIB"):
        return os.environ["BITNET_LIB"]
    for name in ("libbitnet.dylib", "libbitnet.so"):
        path = os.path.join(ROOT, "build", name)
        if os.path.exists(path):
            return path
    raise OSError("libbitnet shared library not found: run `make lib` (or set $BITNET_LIB)")


def _load():
    lib = ctypes.CDLL(_find_library())
    lib.bitnet_default_config.restype = _Config
    lib.bitnet_default_params.restype = _Params
    lib.bitnet_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, _Config]
    lib.bitnet_init.restype = ctypes.c_void_p
    lib.bitnet_free.argtypes = [ctypes.c_void_p]
    for fn in (lib.bitnet_generate, lib.bitnet_chat_turn):
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _Params, _TOKEN_FN, ctypes.c_void_p]
        fn.restype = None
    lib.bitnet_reset_chat.argtypes = [ctypes.c_void_p]
    lib.bitnet_set_system_prompt.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.bitnet_cancel.argtypes = [ctypes.c_void_p]
    lib.bitnet_last_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Stats)]
    lib.bitnet_model_description.argtypes = [ctypes.c_void_p]
    lib.bitnet_model_description.restype = ctypes.c_char_p
    lib.bitnet_last_error.argtypes = [ctypes.c_void_p]
    lib.bitnet_last_error.restype = ctypes.c_char_p
    return lib


_lib = None


def _library():
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


class BitNetError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class BitNet:
    """One model context (not thread-safe; use one per thread)."""

    def __init__(self, model_path=DEFAULT_MODEL, tokenizer_path=DEFAULT_TOKENIZER, n_threads=0,
                 max_seq_len=0, system_prompt="You are a helpful AI assistant."):
        lib = _library()
        cfg = lib.bitnet_default_config()
        cfg.n_threads, cfg.max_seq_len = n_threads, max_seq_len
        cfg.system_prompt = system_prompt.encode() if system_prompt is not None else None
        self._ctx = lib.bitnet_init(os.fsencode(model_path), os.fsencode(tokenizer_path), cfg)
        if not self._ctx:
            raise BitNetError(lib.bitnet_last_error(None).decode(errors="replace"))
        self._lib = lib
        self.description = lib.bitnet_model_description(self._ctx).decode()

    # -- lifecycle ----------------------------------------------------------
    def close(self):
        if getattr(self, "_ctx", None):
            self._lib.bitnet_free(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    # -- generation ---------------------------------------------------------
    def generate(self, prompt, *, temperature=0.6, top_p=0.9, top_k=0, seed=0, max_new_tokens=256):
        """Plain completion of `prompt`; yields text pieces as they are generated."""
        return self._stream(self._lib.bitnet_generate, prompt, temperature, top_p, top_k, seed,
                            max_new_tokens)

    def chat_stream(self, message, *, temperature=0.6, top_p=0.9, top_k=0, seed=0, max_new_tokens=512):
        """One chat turn (conversation history is kept); yields the reply as it streams."""
        return self._stream(self._lib.bitnet_chat_turn, message, temperature, top_p, top_k, seed,
                            max_new_tokens)

    def chat(self, message, **kw):
        """One chat turn; returns the whole reply."""
        return "".join(self.chat_stream(message, **kw))

    def reset_chat(self):
        self._lib.bitnet_reset_chat(self._ctx)

    def set_system_prompt(self, text):
        self._lib.bitnet_set_system_prompt(self._ctx, text.encode() if text is not None else None)

    def cancel(self):
        self._lib.bitnet_cancel(self._ctx)

    @property
    def stats(self):
        s = _Stats()
        self._lib.bitnet_last_stats(self._ctx, ctypes.byref(s))
        out = {name: getattr(s, name) for name, _ in _Stats._fields_}
        out["stop_reason"] = STOP_REASONS[s.stop_reason]
        out["context_reset"] = bool(s.context_reset)
        return out

    def _stream(self, fn, text, temperature, top_p, top_k, seed, max_new_tokens):
        if not self._ctx:
            raise BitNetError("context is closed")
        params = _Params(temperature, top_k, top_p, seed, max_new_tokens)
        pieces = queue.Queue()
        done = object()

        @_TOKEN_FN
        def on_token(piece, _user):
            pieces.put(piece.decode("utf-8"))  # always complete UTF-8 (see bitnet.h)

        def run():
            try:
                fn(self._ctx, text.encode("utf-8"), params, on_token, None)
            finally:
                pieces.put(done)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            while True:
                item = pieces.get()
                if item is done:
                    break
                yield item
        finally:  # consumer stopped early, raised, or hit Ctrl+C: stop the C loop
            if worker.is_alive():
                self.cancel()
            worker.join()
            _ = on_token  # keep the callback alive until the C call has returned
        if self.stats["stop_reason"] == "error":
            raise BitNetError(self._lib.bitnet_last_error(self._ctx).decode(errors="replace"))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    print(f"library: {_find_library()}")
    try:  # error path works without a model (used by CI)
        BitNet("does-not-exist.bitnet", DEFAULT_TOKENIZER)
        raise AssertionError("expected BitNetError")
    except BitNetError as e:
        assert "does-not-exist.bitnet" in str(e)
        print(f"error path:  OK ({e})")
    if not os.path.exists(DEFAULT_MODEL):
        print("model not found: skipping generation tests (run tools/download_model.sh)")
        return

    with BitNet() as llm:
        print(f"model:       {llm.description}")
        text = "".join(llm.generate("The capital of France is", temperature=0, max_new_tokens=8))
        print(f"generate:    'The capital of France is{text}'")
        assert "Paris" in text and llm.stats["generated_tokens"] == 8

        print("chat:        ", end="", flush=True)
        for piece in llm.chat_stream("What is 7 + 5? Answer with just the number.", temperature=0):
            print(piece, end="", flush=True)
        print()
        assert llm.stats["stop_reason"] == "end_of_turn"
        reply = llm.chat("Multiply that by 10. Just the number.", temperature=0)
        print(f"follow-up:   {reply}  (history kept)")
        assert "120" in reply

        n = 0
        for _ in llm.generate("Write a long essay about the sea.", max_new_tokens=500, seed=1):
            n += 1
            if n == 5:
                break  # leaving the loop cancels generation in C
        assert llm.stats["stop_reason"] == "cancelled", llm.stats
        s = llm.stats
        print(f"early stop:  cancelled after {s['generated_tokens']} tokens")
        emoji = "".join(llm.chat_stream("Reply with three fruit emoji.", temperature=0, max_new_tokens=12))
        print(f"utf-8:       {emoji}")
        print(f"stats:       TTFT {s['ttft_ms']:.0f} ms, decode {s['decode_tok_s']:.1f} tok/s")
    print("Python bindings: PASSED")


if __name__ == "__main__":
    _self_test()
