/*
 * platform.h — Small OS / CPU abstraction layer (macOS and Linux).
 *
 *   platform_now_ns()          monotonic clock in nanoseconds
 *   platform_resident_bytes()  current resident set size (0 if unknown)
 *   platform_cpu_count()       online logical CPUs (POSIX sysconf)
 *   platform_cpu_relax()       spin-wait hint (ARM `yield` / x86 `pause`)
 *   half_to_float / float_to_half   IEEE binary16 conversion in software
 *
 * Define BITNET_PORTABLE to use the generic POSIX implementations on macOS
 * too (exercises the Linux code paths on a Mac).
 */
#ifndef BITNET_PLATFORM_H
#define BITNET_PLATFORM_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__) && !defined(BITNET_PORTABLE)
#define BITNET_DARWIN 1
#include <mach/mach.h>
#endif

static inline uint64_t platform_now_ns(void) {
#ifdef BITNET_DARWIN
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

static inline size_t platform_resident_bytes(void) {
#ifdef BITNET_DARWIN
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return (size_t)info.resident_size;
#else
    /* Linux: second field of /proc/self/statm is resident pages. */
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long size = 0, resident = 0;
    const int ok = fscanf(f, "%lu %lu", &size, &resident) == 2;
    fclose(f);
    return ok ? (size_t)resident * (size_t)sysconf(_SC_PAGESIZE) : 0;
#endif
}

static inline int platform_cpu_count(void) {
    const long n = sysconf(_SC_NPROCESSORS_ONLN); /* POSIX: macOS and Linux */
    return n > 0 ? (int)n : 1;
}

static inline void platform_cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause");
#endif
}

/* IEEE 754 binary16 <-> binary32, exact for all inputs (incl. subnormals,
 * inf, NaN); float_to_half rounds to nearest even. */
static inline float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, bits;
    if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else if (exp != 0) {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    } else if (mant == 0) {
        bits = sign;
    } else { /* subnormal: normalize */
        exp = 113;
        while (!(mant & 0x400u)) {
            mant <<= 1;
            exp--;
        }
        bits = sign | (exp << 23) | ((mant & 0x3FFu) << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline uint16_t float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    const uint32_t abs = x & 0x7FFFFFFFu;
    if (abs >= 0x7F800000u) return sign | (abs > 0x7F800000u ? 0x7E00u : 0x7C00u); /* NaN / inf */
    if (abs >= 0x477FF000u) return sign | 0x7C00u;                                 /* overflow */
    if (abs < 0x38800000u) {                                                       /* subnormal / zero */
        if (abs < 0x33000000u) return sign;
        const uint32_t e = abs >> 23, m = (abs & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 126 - e; /* 14..24 */
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1u))) r++;
        return sign | (uint16_t)r;
    }
    uint32_t r = abs - 0x38000000u; /* rebias exponent 127 -> 15 */
    const uint32_t rem = r & 0x1FFFu;
    r >>= 13;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r++;
    return sign | (uint16_t)r;
}

#endif /* BITNET_PLATFORM_H */
