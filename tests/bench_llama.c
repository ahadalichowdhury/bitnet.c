/*
 * bench_llama.c — Throughput in llama-bench's terms, for comparing bitnet.c
 * with llama.cpp-based engines such as Microsoft's bitnet.cpp
 * (tools/bench_compare.sh runs both).
 *
 *   build/bench_llama MODEL.bitnet [-p 512] [-n 128] [-t THREADS] [-r 5] [--json] [--profile]
 *
 * Same definitions as llama-bench:
 *   ppN  prompt processing: N random tokens from an empty KV cache in one
 *        batched call (logits for the last token), tokens/s = N / time
 *   tgN  text generation: N random tokens decoded one at a time from an empty
 *        KV cache, each producing full logits, tokens/s = N / time
 * One untimed warm-up run of each test (pages the weights in), then r timed
 * repetitions; reported as mean +- sample standard deviation of tokens/s.
 * Sampling and tokenization are excluded, as in llama-bench. --profile adds a
 * per-stage time breakdown (ms per token) of the timed runs.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "model_loader.h"
#include "platform.h"
#include "threadpool.h"
#include "transformer.h"

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

/* Peak resident set size (getrusage: KiB on Linux, bytes on macOS). */
static size_t peak_rss_bytes(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#ifdef __APPLE__
    return (size_t)ru.ru_maxrss;
#else
    return (size_t)ru.ru_maxrss * 1024;
#endif
}

static int32_t rand_token(int vocab) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (int32_t)(g_rng % (uint64_t)vocab);
}

static double run_pp(const BitNetModel *m, RunState *s, int32_t *toks, int n) {
    for (int i = 0; i < n; i++) toks[i] = rand_token(m->config.vocab_size);
    const uint64_t t0 = platform_now_ns();
    transformer_forward_batch(toks, n, 0, m, s, 1);
    return (double)(platform_now_ns() - t0) / 1e9;
}

static double run_tg(const BitNetModel *m, RunState *s, int n) {
    const uint64_t t0 = platform_now_ns();
    for (int i = 0; i < n; i++) transformer_forward(rand_token(m->config.vocab_size), i, m, s);
    return (double)(platform_now_ns() - t0) / 1e9;
}

/* ms per token of each stage accumulated in s->prof since the last reset. */
static void print_profile(const char *name, const RunState *s) {
    const transformer_profile *p = &s->prof;
    const double n = p->tokens ? (double)p->tokens : 1.0, ms = 1e-6;
    printf("  %-5s ms/token: total %.2f | BitLinear attn (qkv+o) %.2f | attention %.2f | BitLinear ffn %.2f"
           " | logits %.2f | other %.2f\n", name, p->total * ms / n, p->attn_proj * ms / n,
           p->attention * ms / n, p->ffn * ms / n, p->logits * ms / n,
           (double)(p->total - p->attn_proj - p->attention - p->ffn - p->logits) * ms / n);
}

static void stats(const double *ts, int r, double *mean, double *sd) {
    double s = 0, q = 0;
    for (int i = 0; i < r; i++) s += ts[i];
    *mean = s / r;
    for (int i = 0; i < r; i++) q += (ts[i] - *mean) * (ts[i] - *mean);
    *sd = r > 1 ? sqrt(q / (r - 1)) : 0.0;
}

int main(int argc, char **argv) {
    const char *path = NULL;
    int pp = 512, tg = 128, threads = 0, reps = 5, json = 0, profile = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc) pp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) tg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")) json = 1;
        else if (!strcmp(argv[i], "--profile")) profile = 1;
        else if (argv[i][0] != '-' && !path) path = argv[i];
        else {
            fprintf(stderr, "usage: %s MODEL.bitnet [-p 512] [-n 128] [-t THREADS] [-r 5] [--json]\n", argv[0]);
            return 2;
        }
    }
    if (!path || pp < 0 || tg < 0 || reps < 1 || threads < 0 || pp + tg == 0) {
        fprintf(stderr, "usage: %s MODEL.bitnet [-p 512] [-n 128] [-t THREADS] [-r 5] [--json]\n", argv[0]);
        return 2;
    }

    char err[256];
    BitNetModel m;
    if (bitnet_model_load(path, &m, err, sizeof(err)) != 0) {
        fprintf(stderr, "bench_llama: %s\n", err);
        return 1;
    }
    const int ctx = pp > tg ? pp : tg;
    RunState s;
    if (runstate_init(&s, &m, ctx, threads, err, sizeof(err)) != 0) {
        fprintf(stderr, "bench_llama: %s\n", err);
        return 1;
    }
    int32_t *toks = malloc((size_t)(pp > 0 ? pp : 1) * sizeof(int32_t));
    double *ts = malloc((size_t)reps * sizeof(double));
    double pp_mean = 0, pp_sd = 0, tg_mean = 0, tg_sd = 0;
    transformer_profile pp_prof = {0}, tg_prof = {0};

    if (pp > 0) {
        run_pp(&m, &s, toks, pp); /* warm-up */
        memset(&s.prof, 0, sizeof(s.prof));
        s.profile = profile;
        for (int r = 0; r < reps; r++) ts[r] = pp / run_pp(&m, &s, toks, pp);
        s.profile = 0;
        pp_prof = s.prof;
        stats(ts, reps, &pp_mean, &pp_sd);
    }
    if (tg > 0) {
        run_tg(&m, &s, tg); /* warm-up */
        memset(&s.prof, 0, sizeof(s.prof));
        s.profile = profile;
        for (int r = 0; r < reps; r++) ts[r] = tg / run_tg(&m, &s, tg);
        s.profile = 0;
        tg_prof = s.prof;
        stats(ts, reps, &tg_mean, &tg_sd);
    }

    const int nthreads = threadpool_size(s.pool);
    if (json) {
        printf("{\"engine\": \"bitnet.c\", \"model_bytes\": %zu, \"threads\": %d, \"reps\": %d, "
               "\"pp\": %d, \"pp_ts\": %.3f, \"pp_sd\": %.3f, \"tg\": %d, \"tg_ts\": %.3f, \"tg_sd\": %.3f, "
               "\"peak_rss_bytes\": %zu}\n",
               m.map_size, nthreads, reps, pp, pp_mean, pp_sd, tg, tg_mean, tg_sd,
               peak_rss_bytes());
    } else {
        printf("bitnet.c: %s (%.1f MiB), %d threads, %d repetitions\n", path, m.map_size / 1048576.0,
               nthreads, reps);
        printf("| test   |            t/s |\n|--------|----------------|\n");
        if (pp > 0) printf("| pp%-4d | %7.2f +- %4.2f |\n", pp, pp_mean, pp_sd);
        if (tg > 0) printf("| tg%-4d | %7.2f +- %4.2f |\n", tg, tg_mean, tg_sd);
        printf("peak resident: %.1f MiB\n", peak_rss_bytes() / 1048576.0);
        if (profile) {
            printf("profile (timed runs; clock reads add a little overhead):\n");
            if (pp > 0) { s.prof = pp_prof; print_profile("pp", &s); }
            if (tg > 0) { s.prof = tg_prof; print_profile("tg", &s); }
        }
    }
    free(toks);
    free(ts);
    runstate_free(&s);
    bitnet_model_free(&m);
    return 0;
}
