// Baseline harness for the mmap model-loading work.
//
// Three things move when the loader changes, so all three are measured here:
//   1. how long whisper_init_from_file_with_params takes,
//   2. what the weights cost in memory once loaded — and, crucially, whether
//      that cost is dirty (swap-backed) or clean (file-backed),
//   3. whether the model still computes the same numbers.
//
// (3) is not a nicety. Metal reads a misaligned buffer offset without any
// error and returns wrong values, so a bad mapping produces an app that runs
// and quietly transcribes worse. The fingerprint below runs the encoder and
// the decoder on Metal over a fixed synthetic take and hashes the logits, so
// that failure shows up as a changed hash instead of a bad transcript weeks
// later.
//
//   ./bench --model ~/Library/Application\ Support/Murmur/models/ggml-large-v3-turbo.bin

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <mach/mach.h>
#include <libproc.h>

#include "whisper.h"

#define SR          16000
#define TAKE_SEC    30
#define N_SAMPLES   (SR * TAKE_SEC)
#define N_LOGITS    4096
#define N_THREADS   4

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// phys_footprint is the number Activity Monitor shows. The split that matters
// here is internal vs external: internal is anonymous memory, which can only
// be reclaimed by writing it to swap, while external is file-backed and can be
// dropped for free and re-read from the model file. Today the weights are
// entirely internal. Moving them to external is the whole point of the patch.
typedef struct { uint64_t footprint, internal, compressed, external, resident; } vmsnap;

static vmsnap vm_now(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    vmsnap s = {0};
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        s.footprint  = info.phys_footprint;
        s.internal   = info.internal;
        s.compressed = info.compressed;
        s.external   = info.external;
    }
    mach_task_basic_info_data_t bi;
    mach_msg_type_number_t bc = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&bi, &bc) == KERN_SUCCESS) {
        s.resident = bi.resident_size;
    }
    return s;
}

static uint64_t peak_footprint(void) {
    struct rusage_info_v4 ri;
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ri) == 0) {
        return ri.ri_lifetime_max_phys_footprint;
    }
    return 0;
}

static uint64_t fnv1a(const void *p, size_t n) {
    const unsigned char *b = p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

// A fixed chirp. The point is not that whisper transcribes it — the point is
// that the same samples must drive the same weights to the same logits.
static void synth(float *out, int n) {
    for (int i = 0; i < n; i++) {
        double t = (double)i / SR;
        double f = 120.0 + 300.0 * sin(2.0 * M_PI * 0.25 * t);
        out[i] = (float)(0.4 * sin(2.0 * M_PI * f * t) +
                         0.2 * sin(2.0 * M_PI * 2.7 * f * t) * exp(-fmod(t, 1.0)));
    }
}

static double mb(int64_t b) { return b / 1048576.0; }
static int64_t delta(uint64_t a, uint64_t b) { return (int64_t)a - (int64_t)b; }

int main(int argc, char **argv) {
    const char *model = NULL;
    int iters = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)      model = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    }
    if (!model) { fprintf(stderr, "usage: bench --model PATH [--iters N]\n"); return 2; }

    float *pcm = malloc(N_SAMPLES * sizeof(float));
    synth(pcm, N_SAMPLES);

    printf("model : %s\n", model);
    printf("iters : %d   (take: %ds synthetic, %d threads)\n\n", iters, TAKE_SEC, N_THREADS);
    printf("%-4s %10s %10s %12s %12s %12s %18s\n",
           "it", "load_s", "encode_s", "footprint", "resident", "external", "logits_hash");
    printf("%-4s %10s %10s %12s %12s %12s %18s\n",
           "---", "--------", "--------", "--charged-", "-in RAM---", "--clean---", "----------------");

    double load_sum = 0, enc_sum = 0, load_min = 1e9;
    int64_t fp_sum = 0, load_fp_sum = 0;
    uint64_t first_hash = 0;
    int mismatches = 0;

    for (int it = 0; it < iters; it++) {
        vmsnap before = vm_now();

        struct whisper_context_params cp = whisper_context_default_params();
        cp.use_gpu     = true;   // matches WhisperEngine.swift
        cp.flash_attn  = false;

        double t0 = now_sec();
        struct whisper_context *ctx = whisper_init_from_file_with_params(model, cp);
        double t_load = now_sec() - t0;
        if (!ctx) { fprintf(stderr, "init failed\n"); return 1; }

        vmsnap after = vm_now();

        // Same weights, same input -> same logits. Anything else is corruption.
        if (whisper_pcm_to_mel(ctx, pcm, N_SAMPLES, N_THREADS) != 0) { fprintf(stderr, "mel failed\n"); return 1; }
        double t1 = now_sec();
        if (whisper_encode(ctx, 0, N_THREADS) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
        double t_enc = now_sec() - t1;

        whisper_token prompt[1] = { whisper_token_sot(ctx) };
        if (whisper_decode(ctx, prompt, 1, 0, N_THREADS) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

        vmsnap used = vm_now();

        const float *logits = whisper_get_logits(ctx);
        uint64_t hash = fnv1a(logits, N_LOGITS * sizeof(float));

        if (it == 0) first_hash = hash;
        else if (hash != first_hash) mismatches++;

        printf("%-4d %10.3f %10.3f %11.1fM %11.1fM %11.1fM %#18llx\n",
               it, t_load, t_enc,
               mb(delta(used.footprint, before.footprint)),
               mb(delta(used.resident, before.resident)),
               mb(delta(used.external, before.external)),
               (unsigned long long)hash);

        load_sum += t_load; enc_sum += t_enc;
        if (t_load < load_min) load_min = t_load;
        fp_sum += delta(used.footprint, before.footprint);
        load_fp_sum += delta(after.footprint, before.footprint);

        whisper_free(ctx);
    }

    printf("\nload   : mean %.3f s   best %.3f s\n", load_sum / iters, load_min);
    printf("encode : mean %.3f s\n", enc_sum / iters);
    printf("weights: mean %.1f MB right after load, %.1f MB after the model is used\n",
           mb(load_fp_sum / iters), mb(fp_sum / iters));
    printf("         dirty (internal) memory is swap-backed; clean (external) is not\n");
    printf("peak   : %.1f MB lifetime max footprint\n", mb((int64_t)peak_footprint()));
    printf("logits : %s\n", mismatches ? "*** UNSTABLE ACROSS ITERATIONS ***" : "stable across iterations");
    printf("\nRecord the hash. After the mmap patch it must be identical —\n"
           "a changed hash means the mapped weights are misaligned.\n");

    free(pcm);
    return mismatches ? 1 : 0;
}
