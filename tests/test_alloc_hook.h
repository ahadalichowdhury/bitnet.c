/*
 * test_alloc_hook.h — Test-only heap allocation counter (macOS).
 *
 * Wraps the malloc/calloc/realloc/valloc/memalign entry points of every
 * registered malloc zone (nano + scalable), so heap allocations from any
 * thread (ours, GCD's, libc's) are counted while installed. Used to prove that
 * forward passes and the generation loop never allocate.
 *
 *   if (hook_zones(1) == 0) { g_allocs = 0; ...; n = g_allocs; hook_zones(0); }
 *
 * Only available (ALLOC_HOOK_AVAILABLE) on macOS without sanitizers; callers
 * must skip the check otherwise.
 */
#ifndef BITNET_TEST_ALLOC_HOOK_H
#define BITNET_TEST_ALLOC_HOOK_H

#include <stdlib.h>
#include <unistd.h>

#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define SANITIZER_OWNS_MALLOC 1
#endif
#endif

/* The hook rewrites macOS malloc-zone tables; elsewhere (Linux) and in
 * sanitizer builds the zero-allocation checks report SKIPPED. */
#if defined(__APPLE__) && !defined(BITNET_PORTABLE) && !defined(SANITIZER_OWNS_MALLOC)
#define ALLOC_HOOK_AVAILABLE 1

#include <mach/mach.h>
#include <malloc/malloc.h>

#define MAX_ZONES 16
typedef struct {
    malloc_zone_t *zone;
    void *(*malloc_fn)(malloc_zone_t *, size_t);
    void *(*calloc_fn)(malloc_zone_t *, size_t, size_t);
    void *(*realloc_fn)(malloc_zone_t *, void *, size_t);
    void *(*valloc_fn)(malloc_zone_t *, size_t);
    void *(*memalign_fn)(malloc_zone_t *, size_t, size_t);
} zone_hooks;

static zone_hooks g_zones[MAX_ZONES];
static int g_nzones;
static volatile long g_allocs;

static const zone_hooks *orig(malloc_zone_t *z) {
    for (int i = 0; i < g_nzones; i++)
        if (g_zones[i].zone == z) return &g_zones[i];
    abort();
}
static void *h_malloc(malloc_zone_t *z, size_t n) {
    __atomic_fetch_add(&g_allocs, 1, __ATOMIC_RELAXED);
    return orig(z)->malloc_fn(z, n);
}
static void *h_calloc(malloc_zone_t *z, size_t c, size_t n) {
    __atomic_fetch_add(&g_allocs, 1, __ATOMIC_RELAXED);
    return orig(z)->calloc_fn(z, c, n);
}
static void *h_realloc(malloc_zone_t *z, void *p, size_t n) {
    __atomic_fetch_add(&g_allocs, 1, __ATOMIC_RELAXED);
    return orig(z)->realloc_fn(z, p, n);
}
static void *h_valloc(malloc_zone_t *z, size_t n) {
    __atomic_fetch_add(&g_allocs, 1, __ATOMIC_RELAXED);
    return orig(z)->valloc_fn(z, n);
}
static void *h_memalign(malloc_zone_t *z, size_t a, size_t n) {
    __atomic_fetch_add(&g_allocs, 1, __ATOMIC_RELAXED);
    return orig(z)->memalign_fn(z, a, n);
}

static int zone_writable(malloc_zone_t *z, int writable) {
    const vm_size_t page = (vm_size_t)sysconf(_SC_PAGESIZE);
    const vm_address_t start = (vm_address_t)z & ~(page - 1);
    const vm_size_t len = (((vm_address_t)z + sizeof(*z) + page - 1) & ~(page - 1)) - start;
    return vm_protect(mach_task_self(), start, len, 0,
                      writable ? VM_PROT_READ | VM_PROT_WRITE : VM_PROT_READ) == KERN_SUCCESS ? 0 : -1;
}

__attribute__((unused)) static int hook_zones(int install) {
    if (install) {
        vm_address_t *zones;
        unsigned count;
        if (malloc_get_all_zones(mach_task_self(), NULL, &zones, &count) != KERN_SUCCESS) return -1;
        g_nzones = 0;
        for (unsigned i = 0; i < count && g_nzones < MAX_ZONES; i++) {
            malloc_zone_t *z = (malloc_zone_t *)zones[i];
            g_zones[g_nzones] = (zone_hooks){z, z->malloc, z->calloc, z->realloc, z->valloc,
                                             z->version >= 5 ? z->memalign : NULL};
            g_nzones++;
        }
    }
    for (int i = 0; i < g_nzones; i++) {
        malloc_zone_t *z = g_zones[i].zone;
        if (zone_writable(z, 1) != 0) return -1;
        z->malloc = install ? h_malloc : g_zones[i].malloc_fn;
        z->calloc = install ? h_calloc : g_zones[i].calloc_fn;
        z->realloc = install ? h_realloc : g_zones[i].realloc_fn;
        z->valloc = install ? h_valloc : g_zones[i].valloc_fn;
        if (g_zones[i].memalign_fn) z->memalign = install ? h_memalign : g_zones[i].memalign_fn;
        zone_writable(z, 0);
    }
    return 0;
}

#endif /* ALLOC_HOOK_AVAILABLE */

#endif /* BITNET_TEST_ALLOC_HOOK_H */
