/*
 * threadpool.h — Persistent fork/join worker pool with zero per-job allocation.
 *
 * threadpool_run(pool, n, ctx, fn) calls fn(ctx, i) for i in [0, n) across
 * the workers and the calling thread, returning when all n calls are done.
 * Items are claimed from one atomic counter, so faster (P) cores simply take
 * more items than slower (E) cores. Idle workers spin briefly for the next job
 * (dispatches inside a forward pass are microseconds apart) and then sleep on
 * a condition variable, so an idle pool costs no CPU.
 *
 * Unlike GCD's dispatch_apply_f (which calloc()s on every call), running a
 * job performs no heap allocation. One job at a time: threadpool_run must not
 * be called concurrently on the same pool, nor from inside a job.
 */
#ifndef BITNET_THREADPOOL_H
#define BITNET_THREADPOOL_H

#include <stddef.h>

typedef struct threadpool threadpool;
typedef void (*threadpool_fn)(void *ctx, size_t item);

/* n_threads includes the calling thread; <= 0 selects all logical CPUs. */
threadpool *threadpool_create(int n_threads, char *err, size_t err_len);
void threadpool_destroy(threadpool *pool); /* NULL is allowed */
int threadpool_size(const threadpool *pool);

void threadpool_run(threadpool *pool, size_t n_items, void *ctx, threadpool_fn fn);

#endif /* BITNET_THREADPOOL_H */
