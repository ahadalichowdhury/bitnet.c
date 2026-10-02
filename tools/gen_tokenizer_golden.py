#!/usr/bin/env python3
"""
gen_tokenizer_golden.py — Generate golden encodings for src/step5_tokenizer.c
using the reference Hugging Face `tokenizers` library (dev-time only; the C
tokenizer itself has no dependencies).

Usage:
  python3 -m venv .venv && .venv/bin/pip install tokenizers
  .venv/bin/python tools/gen_tokenizer_golden.py \
      models/hf/bitnet-b1.58-2B-4T/tokenizer.json tests/data/tokenizer_golden.txt

Output format (one case per line, after '#' comment lines):
  <label>\t<UTF-8 text as hex>\t<space-separated token ids>
"""

import os
import random
import sys
import time

try:
    import tokenizers
    from tokenizers import Tokenizer
except ImportError:
    sys.exit("this generator needs the reference library: pip install tokenizers")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CURATED = [
    # --- empty / trivial
    "", " ", "a", "A", "0", "\n", "\t", "\r\n", "  ", "\n\n\n",
    # --- English and contractions
    "Hello, world!", "The quick brown fox jumps over the lazy dog.",
    "I'm sure you've seen it; they'll say we'd won't can't shouldn't.",
    "HE'S SHE'LL WE'RE THEY'VE I'M I'D IT'S", "it's it'S IT's it'ſ x'ſa", "'s 's' ''s '''s",
    "rock'n'roll o'clock y'all ma'am", "don’t won’t (curly apostrophes)",
    "Mr. Smith paid $1,234.56 on 2024-02-29 at 13:45:07 (UTC+05:30).",
    # --- numbers
    "1 12 123 1234 12345 123456 1234567890", "3.14159265358979323846", "1e-9 -0.0 +42 0x1F 0b1010",
    "١٢٣٤ १२३ １２３", "x² + y³ = z⁴ ½ ⅓ Ⅷ Ⅻ",
    # --- whitespace edge cases
    "  leading", "trailing  ", "a  b   c    d", "a\tb\t\tc", "line1\nline2\r\nline3\rline4",
    "para\n\n\npara", "   \n   \n", " \n", "\n ", "x \n y", "x\t\n\ty", "end with spaces   ",
    "a b  c", "a　b　　c", "a b c", "a\u0085b", "a​b﻿c",
    "\u000b\u000c\u001c\u001f", "tabs\t\t\tand\t spaces \t mixed",
    # --- punctuation / symbols
    "!!!???...,,,;;;:::", "Wait... what?! (Really?!)", "--- *** ___ ~~~ ``` ###",
    "\"quoted\" 'single' “fancy” ‘fancy’ «guillemets»",
    "←→↑↓ ∀x∈ℝ: x²≥0 ∞ ∑ ∫",
    "$€£¥₹₿", "© ® ™ § ¶ †",
    # --- code
    "def f(x):\n    return x ** 2  # square\n",
    "int main(void) {\n\tprintf(\"%d\\n\", 42);\n\treturn 0;\n}\n",
    "for (int i = 0; i < n; ++i) { a[i] += b[i] * c[i]; }",
    "{\"key\": [1, 2.5, true, null], \"nested\": {\"a\": \"b\"}}",
    "<div class=\"x\"><a href=\"https://example.com/a?b=c&d=e#f\">link</a></div>",
    "SELECT id, name FROM users WHERE age >= 18 ORDER BY name DESC;",
    "git commit -m 'fix: handle \\r\\n' && ./run.sh --flag=1 2>&1 | tee log.txt",
    "const x = (a, b) => a ?? b; // nullish\n/* block */",
    "    indented four\n        indented eight\n\tindented tab",
    "$$\\int_0^\\infty e^{-x^2}\\,dx = \\frac{\\sqrt{\\pi}}{2}$$",
    "user@example.com https://www.example.org/path/to/page.html?q=1",
    # --- emojis
    "\U0001f600\U0001f603\U0001f604", "I ❤️ Rust \U0001f980!",
    "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466 family",
    "\U0001f44d\U0001f3fd \U0001f44b\U0001f3ff", "\U0001f1fa\U0001f1f8\U0001f1ef\U0001f1f5\U0001f1e7\U0001f1e9",
    "1️⃣ #️⃣", "\U0001f3f3️‍\U0001f308 \U0001faf6",
    # --- scripts
    "你好，世界！这是一个测试。",
    "こんにちは世界、カタカナとひらがな",
    "안녕하세요 세계", "Привет, мир!",
    "Γειά σου Κόσμε",
    "مرحبا بالعالم", "שלום עולם",
    "नमस्ते दुनिया", "สวัสดีชาวโลก",
    "Café vs Café (NFD vs NFC)", "Z͑ͫ̓ͪ̂ͫ̽͏̴a̐ĺgo",
    "ẛ İstanbul ı ß ẞ ﬁ ﬂ",
    "\U0001e290\U0001e291 \U0001e4d0 \U0001e5d0 ࢏౜",  # Unicode 14/15/16 letters, 17 letters
    "\U00016ac0\U00016ac1 \U00011f50",  # Unicode 14/15 digits
    # --- special tokens
    "<|begin_of_text|>Hello<|eot_id|>", "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>",
    "<|eot_id|><|eot_id|>", " <|eot_id|> ", "<|eot_id", "<|eot_id|", "<|not_a_token|>", "<<|eot_id|>>",
    "<|reserved_special_token_0|><|reserved_special_token_250|>", "text<|end_of_text|>",
    # --- markdown / prose
    "# Title\n\n- item 1\n- item 2\n\n> quote\n\n**bold** _italic_ `code`\n",
    "| a | b |\n|---|---|\n| 1 | 2 |\n",
]

ATOMS = [
    "a", "Z", "hello", "World", "x", "é", "ß", "ж", "λ", "中", "文", "あ",
    "가", "ا", "א", "क", "्", "́", "ก",
    "0", "7", "42", "123", "9999", "٣", "²", "½", "Ⅷ",
    " ", " ", " ", "  ", "\t", "\n", "\n", "\r", "\r\n", " ", "　", " ", "\u0085",
    "​", "﻿", "\u000b",
    "'", "'s", "'ll", "'RE", "'ſ", "’", "\"", ".", ",", "!", "?", "...", "-", "_", "/",
    "\\", "(", ")", "{", "}", "[", "]", "<", ">", "|", "<|", "|>", "#", "@", "$", "%", "^", "&",
    "*", "+", "=", "~", "`", ";", ":",
    "\U0001f600", "\U0001f980", "❤️", "\U0001f44d\U0001f3fd", "‍", "\U0001f1fa\U0001f1f8",
    "\U0001e290", "\U0001e5d0", "࢏",
    "<|eot_id|>", "<|begin_of_text|>",
]

STRESS = [
    ("5000 spaces", " " * 5000), ("3000 letters", "a" * 3000), ("2000 digits", "7" * 2000),
    ("500 newlines", "\n" * 500), ("mixed ws run", " \t" * 700 + "x"), ("1000 emojis", "\U0001f600" * 1000),
    ("long word", "pneumonoultramicroscopicsilicovolcanoconiosis" * 40),
    ("base64-ish", "".join(random.Random(5).choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/") for _ in range(4000))),
    ("CJK block", "中文测试" * 600),
]

FILES = [
    "models/hf/bitnet-b1.58-2B-4T/README.md", "models/hf/bitnet-b1.58-2B-4T/LICENSE",
    "CLAUDE.md", "ROADMAP.md", "src/tokenizer.c", "tools/export_bitnet.py",
]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    tok = Tokenizer.from_file(sys.argv[1])
    cases = [(f"curated[{i}]", s) for i, s in enumerate(CURATED)]
    rng = random.Random(1234)
    for i in range(3000):
        n = rng.randint(1, 40)
        cases.append((f"fuzz[{i}]", "".join(rng.choice(ATOMS) for _ in range(n))))
    cases += STRESS
    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cases.append((f"file:{rel}", f.read()))

    total_bytes = sum(len(s.encode()) for _, s in cases)
    t0 = time.perf_counter()
    encs = [tok.encode(s, add_special_tokens=False).ids for _, s in cases]
    dt = time.perf_counter() - t0

    with open(sys.argv[2], "w", encoding="ascii") as f:
        f.write(f"# golden encodings from HF tokenizers {tokenizers.__version__} "
                f"({os.path.basename(sys.argv[1])}), add_special_tokens=False\n")
        f.write(f"# {len(cases)} cases, {total_bytes} bytes; HF single-thread encode: "
                f"{dt * 1e3:.1f} ms ({total_bytes / dt / 1e6:.2f} MB/s)\n")
        for (label, s), ids in zip(cases, encs):
            f.write(f"{label}\t{s.encode('utf-8').hex()}\t{' '.join(map(str, ids))}\n")
    print(f"wrote {len(cases)} cases ({total_bytes} bytes) to {sys.argv[2]}; "
          f"HF encode {dt * 1e3:.1f} ms = {total_bytes / dt / 1e6:.2f} MB/s")


if __name__ == "__main__":
    main()
