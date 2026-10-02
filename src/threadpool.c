/*
 * threadpool.c — see threadpool.h.
 *
 * Job protocol: the caller writes (fn, ctx, n_items) into job slot
 * [generation & 1], resets `done`, stores the work counter as
 * (generation << 32 | 0) and then bumps `generation`. Workers claim items with
 * a CAS on the tagged counter and read the job from the slot of the
 * generation they claim for.
 *
 * The slots are double-buffered because a worker that woke late for job g-1
 * can read the job description while the caller is already publishing job g
 * but has not yet re-tagged the counter: with a single slot it would see its
 * own (still valid) tag next to job g's larger n_items and claim an item of
 * job g, running it twice. With two slots, job g never touches g-1's slot, and
 * by the time slot (g-1)&1 is reused (job g+1) the counter carries tag g, so
 * the stale worker's CAS can only fail. Completion is counted per item, not
 * per worker, so the caller never waits for a sleeping worker to wake up.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#include "threadpool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <signal.h>

#include "platform.h"

#ifdef __APPLE__
#include <pthread/qos.h>
#endif

#define CACHE_LINE 128   /* Apple Silicon (also >= the 64-byte x86 line) */
#define SPIN_ITERS 20000 /* ~50-100 us of spinning before a worker sleeps */

struct threadpool {
    _Alignas(CACHE_LINE) _Atomic uint64_t next; /* (generation << 32) | next item */
    _Alignas(CACHE_LINE) _Atomic size_t done;   /* items completed in this job */
    _Alignas(CACHE_LINE) _Atomic uint32_t generation;
    _Atomic int sleepers;
    _Atomic int stop;
    /* Job description, double-buffered by generation parity (see top). The
     * fields are atomics: a stale worker may read a slot being rewritten for a
     * later job; its tagged CAS then fails, so the values are never used. */
    struct {
        _Atomic(threadpool_fn) fn;
        _Atomic(void *) ctx;
        _Atomic size_t n_items;
    } job[2];

    pthread_mutex_t mu;
    pthread_cond_t cv;
    int n_workers;
    pthread_t *threads;
};

static inline void cpu_relax(void) { platform_cpu_relax(); }

/* Claims and runs items of generation `gen` until none are left. */
static void run_items(threadpool *p, uint32_t gen) {
    const threadpool_fn fn = atomic_load_explicit(&p->job[gen & 1].fn, memory_order_relaxed);
    void *const ctx = atomic_load_explicit(&p->job[gen & 1].ctx, memory_order_relaxed);
    const size_t n = atomic_load_explicit(&p->job[gen & 1].n_items, memory_order_relaxed);
    uint64_t v = atomic_load_explicit(&p->next, memory_order_acquire);
    for (;;) {
        if ((uint32_t)(v >> 32) != gen || (v & 0xFFFFFFFFu) >= n) return;
        if (!atomic_compare_exchange_weak_explicit(&p->next, &v, v + 1, memory_order_acq_rel,
                                                   memory_order_acquire))
            continue;
        fn(ctx, (size_t)(v & 0xFFFFFFFFu));
        atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
        v = atomic_load_explicit(&p->next, memory_order_acquire);
    }
}

static void *worker_main(void *arg) {
    threadpool *p = arg;
    uint32_t seen = atomic_load(&p->generation);
    for (;;) {
        uint32_t g;
        int spins = 0;
        while ((g = atomic_load(&p->generation)) == seen) {
            if (atomic_load_explicit(&p->stop, memory_order_relaxed)) return NULL;
            if (++spins < SPIN_ITERS) {
                cpu_relax();
                continue;
            }
            pthread_mutex_lock(&p->mu);
            atomic_fetch_add(&p->sleepers, 1); /* seq_cst: pairs with run's generation bump */
            while (atomic_load(&p->generation) == seen && !atomic_load(&p->stop))
                pthread_cond_wait(&p->cv, &p->mu);
            atomic_fetch_sub(&p->sleepers, 1);
            pthread_mutex_unlock(&p->mu);
            spins = 0;
        }
        seen = g;
        if (atomic_load(&p->stop)) return NULL;
        run_items(p, g);
    }
}

threadpool *threadpool_create(int n_threads, char *err, size_t err_len) {
    if (n_threads <= 0) n_threads = platform_cpu_count(); /* POSIX sysconf: macOS + Linux */
    if (n_threads < 1) n_threads = 1;
    threadpool *p = NULL;
    if (posix_memalign((void **)&p, CACHE_LINE, sizeof(*p)) != 0) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    memset(p, 0, sizeof(*p));
    atomic_init(&p->next, 0);
    atomic_init(&p->done, 0);
    atomic_init(&p->generation, 0);
    atomic_init(&p->sleepers, 0);
    atomic_init(&p->stop, 0);
    for (int i = 0; i < 2; i++) {
        atomic_init(&p->job[i].fn, NULL);
        atomic_init(&p->job[i].ctx, NULL);
        atomic_init(&p->job[i].n_items, 0);
    }
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    p->n_workers = n_threads - 1;
    p->threads = calloc((size_t)(p->n_workers ? p->n_workers : 1), sizeof(pthread_t));
    if (!p->threads) {
        threadpool_destroy(p);
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    pthread_attr_t attr;
    pthread_attr_init(&attr);
#ifdef __APPLE__
    /* Prefer performance cores for the workers. */
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    /* Workers inherit a fully blocked signal mask, so asynchronous signals
     * (e.g. SIGINT for Ctrl+C) are always delivered to application threads
     * and can interrupt their blocking calls. */
    sigset_t all, old;
    sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, &old);
    for (int i = 0; i < p->n_workers; i++) {
        if (pthread_create(&p->threads[i], &attr, worker_main, p) != 0) {
            p->n_workers = i;
            pthread_sigmask(SIG_SETMASK, &old, NULL);
            pthread_attr_destroy(&attr);
            threadpool_destroy(p);
            snprintf(err, err_len, "pthread_create failed");
            return NULL;
        }
    }
    pthread_sigmask(SIG_SETMASK, &old, NULL);
    pthread_attr_destroy(&attr);
    return p;
}

void threadpool_destroy(threadpool *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    atomic_store(&p->stop, 1);
    atomic_fetch_add(&p->generation, 1);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->n_workers; i++) pthread_join(p->threads[i], NULL);
    pthread_cond_destroy(&p->cv);
    pthread_mutex_destroy(&p->mu);
    free(p->threads);
    free(p);
}

int threadpool_size(const threadpool *p) { return p ? p->n_workers + 1 : 1; }

void threadpool_run(threadpool *p, size_t n_items, void *ctx, threadpool_fn fn) {
    if (n_items == 0) return;
    if (!p || p->n_workers == 0 || n_items == 1) {
        for (size_t i = 0; i < n_items; i++) fn(ctx, i);
        return;
    }
    if (n_items > 0xFFFFFFFFu) abort();
    const uint32_t gen = atomic_load_explicit(&p->generation, memory_order_relaxed) + 1;
    atomic_store_explicit(&p->job[gen & 1].fn, fn, memory_order_relaxed);
    atomic_store_explicit(&p->job[gen & 1].ctx, ctx, memory_order_relaxed);
    atomic_store_explicit(&p->job[gen & 1].n_items, n_items, memory_order_relaxed);
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    atomic_store_explicit(&p->next, (uint64_t)gen << 32, memory_order_release);
    atomic_store(&p->generation, gen); /* seq_cst: publishes the job */
    if (atomic_load(&p->sleepers) > 0) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
    }
    run_items(p, gen);
    while (atomic_load_explicit(&p->done, memory_order_acquire) < n_items) cpu_relax();
}
