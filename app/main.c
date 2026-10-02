/*
 * main.c — `bitnet` command-line interface (built on the public bitnet.h API only).
 *
 *   bitnet -p "Explain RoPE in one paragraph."      single chat turn, streamed
 *   bitnet -p "Once upon a time" --raw               plain text completion
 *   bitnet -i                                        interactive multi-turn chat
 *   bitnet --bench                                   TTFT / prefill / decode / memory profile
 *
 * Ctrl+C cancels the generation in progress (the REPL keeps running and the
 * conversation stays consistent); Ctrl+D or /exit leaves the REPL.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "bitnet.h"

#define DEFAULT_MODEL     "models/bitnet_2b4t.bitnet"
#define DEFAULT_TOKENIZER "models/hf/bitnet-b1.58-2B-4T/tokenizer.json"

/* ------------------------------------------------------------------------- */
/* Terminal styling                                                          */
/* ------------------------------------------------------------------------- */

static int g_color;
#define STYLE(code) (g_color ? "\033[" code "m" : "")
#define RESET       STYLE("0")
#define BOLD        STYLE("1")
#define DIM         STYLE("2")
#define GREEN       STYLE("1;32")
#define CYAN        STYLE("36")
#define YELLOW      STYLE("33")
#define RED         STYLE("1;31")

/* ------------------------------------------------------------------------- */
/* Signals: Ctrl+C cancels generation, never kills the REPL                  */
/* ------------------------------------------------------------------------- */

static BitNetContext *volatile g_ctx;
static volatile sig_atomic_t g_busy;   /* a generation is running */
static volatile sig_atomic_t g_sigint; /* Ctrl+C seen */

static void on_sigint(int sig) {
    (void)sig;
    g_sigint = 1;
    if (g_busy && g_ctx) bitnet_cancel(g_ctx); /* async-signal-safe */
}

static void install_signals(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* no SA_RESTART: a blocked read returns EINTR */
    sigaction(SIGINT, &sa, NULL);
}

/* ------------------------------------------------------------------------- */
/* Output                                                                    */
/* ------------------------------------------------------------------------- */

static void print_token(const char *piece, void *user) {
    (void)user;
    fputs(piece, stdout);
    fflush(stdout);
}

static void ignore_token(const char *piece, void *user) {
    (void)piece;
    (void)user;
}

static const char *reason_name(BitNetStopReason r) {
    switch (r) {
    case BITNET_STOP_END_OF_TURN:  return "end of turn";
    case BITNET_STOP_MAX_TOKENS:   return "max tokens";
    case BITNET_STOP_CONTEXT_FULL: return "context full";
    case BITNET_STOP_CANCELLED:    return "cancelled";
    case BITNET_STOP_ERROR:        return "error";
    }
    return "?";
}

static void print_stats(FILE *f, const BitNetStats *s) {
    char rate[32] = "n/a"; /* decode speed needs at least two generated tokens */
    if (s->generated_tokens > 1) snprintf(rate, sizeof(rate), "%.1f tok/s", s->decode_tok_s);
    fprintf(f, "%s[%d prompt tok @ %.1f tok/s | TTFT %.0f ms | %d tok @ %s | %s | context %d/%d"
               "%s]%s\n", DIM, s->prompt_tokens, s->prefill_tok_s, s->ttft_ms, s->generated_tokens,
            rate, reason_name(s->stop_reason), s->context_used, s->context_size,
            s->context_reset ? " | history reset" : "", RESET);
}

static void usage(FILE *f) {
    fprintf(f,
        "bitnet %s — BitNet b1.58 inference on Apple Silicon (pure C, ARM NEON)\n\n"
        "usage: bitnet [options] (-p PROMPT | -i | --bench)\n\n"
        "modes:\n"
        "  -p, --prompt TEXT     generate a reply to TEXT (chat template) and exit\n"
        "      --raw             with -p: plain text completion instead of chat\n"
        "  -i, --interactive     multi-turn chat REPL (default when no mode is given)\n"
        "      --bench           profile TTFT, prefill/decode speed and memory\n\n"
        "model:\n"
        "      --model PATH      .bitnet weights      (default %s)\n"
        "      --tokenizer PATH  tokenizer.json       (default %s)\n"
        "      --system TEXT     system prompt        (default \"You are a helpful AI assistant.\")\n"
        "      --ctx N           context length       (default: model maximum, 4096)\n"
        "      --threads N       worker threads       (default: all CPUs)\n\n"
        "sampling:\n"
        "      --temp T          temperature, 0 = greedy (default 0.6)\n"
        "      --top-p P         nucleus sampling      (default 0.9)\n"
        "      --top-k K         top-k, 0 = off        (default 0)\n"
        "      --seed S          RNG seed, 0 = random  (default 0)\n"
        "  -n, --max-new N       max tokens per reply  (default 512, 0 = until end of turn)\n\n"
        "other:\n"
        "      --stats           print speed statistics after each reply (stderr)\n"
        "      --no-color        disable ANSI colors (also: NO_COLOR environment variable)\n"
        "  -v, --version         print version and model information\n"
        "  -h, --help            this help\n",
        BITNET_VERSION_STRING, DEFAULT_MODEL, DEFAULT_TOKENIZER);
}

/* ------------------------------------------------------------------------- */
/* Modes                                                                     */
/* ------------------------------------------------------------------------- */

static int run_prompt(BitNetContext *ctx, const char *prompt, int raw, BitNetSampleParams p, int stats) {
    g_busy = 1;
    if (raw) {
        fputs(prompt, stdout);
        bitnet_generate(ctx, prompt, p, print_token, NULL);
    } else {
        bitnet_chat_turn(ctx, prompt, p, print_token, NULL);
    }
    g_busy = 0;
    BitNetStats s;
    bitnet_last_stats(ctx, &s);
    if (s.stop_reason == BITNET_STOP_CANCELLED) printf("%s [cancelled]%s", YELLOW, RESET);
    printf("\n");
    fflush(stdout); /* keep stdout/stderr ordered when both are redirected */
    if (s.stop_reason == BITNET_STOP_ERROR) {
        fprintf(stderr, "%serror:%s %s\n", RED, RESET, bitnet_last_error(ctx));
        return 1;
    }
    if (stats) print_stats(stderr, &s);
    return s.stop_reason == BITNET_STOP_CANCELLED ? 130 : 0;
}

/* Reads one line; returns 1 with a line, 0 on EOF, -1 if interrupted. */
static int read_line(char **buf, size_t *cap) {
    errno = 0;
    const ssize_t n = getline(buf, cap, stdin);
    if (n < 0) {
        if (errno == EINTR || g_sigint) {
            clearerr(stdin);
            return -1;
        }
        return 0;
    }
    while (n > 0 && ((*buf)[strlen(*buf) - 1] == '\n' || (*buf)[strlen(*buf) - 1] == '\r'))
        (*buf)[strlen(*buf) - 1] = '\0';
    return 1;
}

static void repl_help(void) {
    printf("%scommands:%s /help  /reset (new conversation)  /system TEXT  /stats (toggle)  /exit\n"
           "%sCtrl+C stops a reply; Ctrl+D exits.%s\n", DIM, RESET, DIM, RESET);
}

static int run_interactive(BitNetContext *ctx, BitNetSampleParams p, int stats) {
    const int tty = isatty(STDIN_FILENO);
    printf("%s%s%s\n", BOLD, bitnet_model_description(ctx), RESET);
    printf("%ssampling: temp %.2f, top-p %.2f, top-k %d, seed %llu%s\n", DIM, p.temperature, p.top_p,
           p.top_k, (unsigned long long)p.seed, RESET);
    repl_help();
    char *line = NULL;
    size_t cap = 0;
    for (;;) {
        g_sigint = 0;
        printf("\n%syou>%s ", GREEN, RESET);
        fflush(stdout);
        const int r = read_line(&line, &cap);
        if (r < 0) {
            printf("\n%s(Ctrl+C: use Ctrl+D or /exit to quit)%s\n", DIM, RESET);
            continue;
        }
        if (r == 0) break;
        if (!tty) printf("%s\n", line); /* echo piped input so transcripts read naturally */
        if (!line[0]) continue;

        if (line[0] == '/') {
            if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) break;
            else if (!strcmp(line, "/help")) repl_help();
            else if (!strcmp(line, "/reset")) {
                bitnet_reset_chat(ctx);
                printf("%s(new conversation)%s\n", DIM, RESET);
            } else if (!strncmp(line, "/system ", 8)) {
                bitnet_set_system_prompt(ctx, line + 8);
                printf("%s(system prompt set; new conversation)%s\n", DIM, RESET);
            } else if (!strcmp(line, "/stats")) {
                stats = !stats;
                printf("%s(stats %s)%s\n", DIM, stats ? "on" : "off", RESET);
            } else {
                printf("%sunknown command%s\n", YELLOW, RESET);
                repl_help();
            }
            continue;
        }

        printf("%sbitnet>%s %s", CYAN, RESET, CYAN);
        fflush(stdout);
        g_busy = 1;
        bitnet_chat_turn(ctx, line, p, print_token, NULL);
        g_busy = 0;
        printf("%s", RESET);
        BitNetStats s;
        bitnet_last_stats(ctx, &s);
        if (s.stop_reason == BITNET_STOP_CANCELLED) printf("%s [cancelled]%s", YELLOW, RESET);
        printf("\n");
        if (s.stop_reason == BITNET_STOP_ERROR)
            printf("%serror:%s %s\n", RED, RESET, bitnet_last_error(ctx));
        else if (s.context_reset)
            printf("%s(context window full: earlier turns were dropped)%s\n", DIM, RESET);
        if (stats) print_stats(stdout, &s);
    }
    free(line);
    printf("\n");
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Benchmark                                                                 */
/* ------------------------------------------------------------------------- */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double mib(size_t b) { return (double)b / 1048576.0; }

static int run_bench(const char *model, const char *tokenizer, BitNetConfig cfg) {
    printf("%sbitnet --bench%s\n\n", BOLD, RESET);
    const double t0 = now_ms();
    BitNetContext *ctx = bitnet_init(model, tokenizer, cfg);
    const double load_ms = now_ms() - t0;
    if (!ctx) {
        fprintf(stderr, "%serror:%s %s\n", RED, RESET, bitnet_last_error(NULL));
        return 1;
    }
    g_ctx = ctx;
    BitNetMemoryInfo mem0;
    bitnet_memory_info(ctx, &mem0);
    printf("%s\n\n", bitnet_model_description(ctx));

    BitNetSampleParams greedy = {0.0f, 0, 1.0f, 1, 0};
    BitNetStats s;

    /* Cold: the first pass pages the memory-mapped weights in. */
    greedy.max_new_tokens = 1;
    bitnet_chat_turn(ctx, "Hi", greedy, ignore_token, NULL);
    bitnet_last_stats(ctx, &s);
    const double cold_ttft = s.ttft_ms;

    /* Warm TTFT for a short chat prompt (median of 5). */
    double ttft[5];
    int short_tokens = 0;
    for (int i = 0; i < 5; i++) {
        bitnet_reset_chat(ctx);
        bitnet_chat_turn(ctx, "What is the capital of France?", greedy, ignore_token, NULL);
        bitnet_last_stats(ctx, &s);
        ttft[i] = s.ttft_ms;
        short_tokens = s.prompt_tokens;
    }
    qsort(ttft, 5, sizeof(double), cmp_d);

    /* Prefill throughput on a ~500-token document. */
    char *doc = malloc(1 << 15);
    size_t dl = 0;
    const char *para = "The history of computing is a story of abstraction layered on abstraction: "
                       "transistors become gates, gates become arithmetic units, and arithmetic units "
                       "become processors that run compilers, operating systems and, now, language "
                       "models that predict one token at a time. ";
    while (dl + strlen(para) < 2600) {
        memcpy(doc + dl, para, strlen(para));
        dl += strlen(para);
    }
    doc[dl] = '\0';
    bitnet_generate(ctx, doc, greedy, ignore_token, NULL);
    BitNetStats prefill;
    bitnet_last_stats(ctx, &prefill);

    /* Decode throughput: 128 tokens from a short prompt, and 64 tokens after
     * the long document (attention over ~550 positions). */
    BitNetSampleParams dec = {0.6f, 0, 0.9f, 7, 128};
    bitnet_generate(ctx, "Once upon a time, in a small village by the sea,", dec, ignore_token, NULL);
    BitNetStats decode_short;
    bitnet_last_stats(ctx, &decode_short);
    dec.max_new_tokens = 64;
    bitnet_generate(ctx, doc, dec, ignore_token, NULL);
    BitNetStats decode_long;
    bitnet_last_stats(ctx, &decode_long);
    free(doc);

    BitNetMemoryInfo mem;
    bitnet_memory_info(ctx, &mem);

    printf("%-34s %10.0f ms\n", "load (mmap + tokenizer + buffers)", load_ms);
    printf("%-34s %10.0f ms   (first pass pages in the weights)\n", "TTFT, cold", cold_ttft);
    printf("%-34s %10.0f ms   (%d-token chat prompt, median of 5)\n", "TTFT, warm", ttft[2], short_tokens);
    printf("%-34s %10.1f tok/s (%d-token prompt)\n", "prefill", prefill.prefill_tok_s, prefill.prompt_tokens);
    printf("%-34s %10.1f tok/s (%d tokens, short context, temp 0.6 / top-p 0.9)\n", "decode",
           decode_short.decode_tok_s, decode_short.generated_tokens);
    printf("%-34s %10.1f tok/s (%d tokens after a %d-token prompt)\n", "decode, long context",
           decode_long.decode_tok_s, decode_long.generated_tokens, decode_long.prompt_tokens);
    printf("\nmemory\n");
    printf("  %-32s %8.1f MiB  (memory-mapped, read-only, shared page cache)\n", "model file", mib(mem.model_mapped_bytes));
    printf("  %-32s %8.1f MiB  (incl. KV cache %.1f MiB for %d positions)\n", "run state reserved",
           mib(mem.runstate_bytes), mib(mem.kv_cache_bytes), s.context_size);
    printf("  %-32s %8.1f MiB  (after load: %.1f MiB)\n", "resident now", mib(mem.process_resident_bytes),
           mib(mem0.process_resident_bytes));
    printf("  %-32s %8.1f MiB\n", "peak resident", mib(mem.process_peak_resident_bytes));
    bitnet_free(ctx);
    g_ctx = NULL;
    return 0;
}

/* ------------------------------------------------------------------------- */

int main(int argc, char **argv) {
    const char *model = DEFAULT_MODEL, *tokenizer = DEFAULT_TOKENIZER, *prompt = NULL;
    BitNetConfig cfg = bitnet_default_config();
    BitNetSampleParams p = bitnet_default_params();
    p.max_new_tokens = 512;
    int interactive = 0, bench = 0, raw = 0, stats = 0, version = 0, no_color = 0;

    enum { O_MODEL = 256, O_TOK, O_SYSTEM, O_TEMP, O_TOPP, O_TOPK, O_SEED, O_THREADS, O_CTX, O_RAW,
           O_BENCH, O_STATS, O_NOCOLOR };
    static const struct option opts[] = {
        {"model", required_argument, NULL, O_MODEL},   {"tokenizer", required_argument, NULL, O_TOK},
        {"prompt", required_argument, NULL, 'p'},      {"interactive", no_argument, NULL, 'i'},
        {"system", required_argument, NULL, O_SYSTEM}, {"temp", required_argument, NULL, O_TEMP},
        {"top-p", required_argument, NULL, O_TOPP},    {"top-k", required_argument, NULL, O_TOPK},
        {"seed", required_argument, NULL, O_SEED},     {"threads", required_argument, NULL, O_THREADS},
        {"ctx", required_argument, NULL, O_CTX},       {"max-new", required_argument, NULL, 'n'},
        {"raw", no_argument, NULL, O_RAW},             {"bench", no_argument, NULL, O_BENCH},
        {"stats", no_argument, NULL, O_STATS},         {"no-color", no_argument, NULL, O_NOCOLOR},
        {"version", no_argument, NULL, 'v'},           {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int ch;
    char *end;
#define NUM(var, conv, lo, hi, name)                                                        \
    do {                                                                                    \
        errno = 0;                                                                          \
        const double v_ = conv(optarg, &end);                                               \
        if (errno || *end || end == optarg || v_ < (lo) || v_ > (hi)) {                     \
            fprintf(stderr, "bitnet: invalid %s '%s'\n", name, optarg);                     \
            return 2;                                                                       \
        }                                                                                   \
        var = v_;                                                                           \
    } while (0)
    while ((ch = getopt_long(argc, argv, "p:in:vh", opts, NULL)) != -1) {
        switch (ch) {
        case O_MODEL:   model = optarg; break;
        case O_TOK:     tokenizer = optarg; break;
        case 'p':       prompt = optarg; break;
        case 'i':       interactive = 1; break;
        case O_SYSTEM:  cfg.system_prompt = optarg; break;
        case O_TEMP:    NUM(p.temperature, strtod, 0, 10, "--temp"); break;
        case O_TOPP:    NUM(p.top_p, strtod, 0.0001, 1, "--top-p"); break;
        case O_TOPK:    NUM(p.top_k, strtod, 0, 1e7, "--top-k"); break;
        case O_SEED:
            errno = 0;
            p.seed = strtoull(optarg, &end, 10);
            if (errno || *end || end == optarg || optarg[0] == '-') {
                fprintf(stderr, "bitnet: invalid --seed '%s'\n", optarg);
                return 2;
            }
            break;
        case O_THREADS: NUM(cfg.n_threads, strtod, 0, 256, "--threads"); break;
        case O_CTX:     NUM(cfg.max_seq_len, strtod, 0, 1e6, "--ctx"); break;
        case 'n':       NUM(p.max_new_tokens, strtod, 0, 1e6, "--max-new"); break;
        case O_RAW:     raw = 1; break;
        case O_BENCH:   bench = 1; break;
        case O_STATS:   stats = 1; break;
        case O_NOCOLOR: no_color = 1; break;
        case 'v':       version = 1; break;
        case 'h':       usage(stdout); return 0;
        default:        usage(stderr); return 2;
        }
    }
    if (optind < argc) {
        fprintf(stderr, "bitnet: unexpected argument '%s' (did you mean -p \"...\"?)\n", argv[optind]);
        return 2;
    }
    if ((prompt != NULL) + interactive + bench > 1) {
        fprintf(stderr, "bitnet: choose one of -p, -i, --bench\n");
        return 2;
    }
    g_color = !no_color && isatty(STDOUT_FILENO) && !getenv("NO_COLOR");
    if (bench) return run_bench(model, tokenizer, cfg); /* Ctrl+C simply ends a benchmark */
    install_signals();

    BitNetContext *ctx = bitnet_init(model, tokenizer, cfg);
    if (!ctx) {
        fprintf(stderr, "%serror:%s %s\n", RED, RESET, bitnet_last_error(NULL));
        return 1;
    }
    g_ctx = ctx;
    int rc = 0;
    if (version) printf("bitnet %s\n%s\n", BITNET_VERSION_STRING, bitnet_model_description(ctx));
    else if (prompt) rc = run_prompt(ctx, prompt, raw, p, stats);
    else rc = run_interactive(ctx, p, stats);
    g_ctx = NULL;
    bitnet_free(ctx);
    return rc;
}
