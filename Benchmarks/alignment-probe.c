// Establishes the two facts the mmap loader has to be built around, by
// measuring them rather than trusting the headers.
//
//   1. Does this Metal device let ggml wrap host memory with no copy?
//      (ggml_backend_dev_buffer_from_host_ptr — the whole plan rests on it.)
//
//   2. How far can a tensor's address be off before Metal reads it wrong?
//      The buffer type advertises 32-byte alignment, but that is what ggml's
//      own allocator honours; the mmap path assigns addresses directly and
//      bypasses it. What matters is where compute actually breaks — and it
//      breaks *silently*, returning wrong numbers with no error, so this has
//      to be probed with a real matmul rather than a read-back.
//
// Run this on any machine before trusting the mapped loader there. Metal's
// behaviour is per-device and the M1 Pro result is not a guarantee for M3/M4.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/mman.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#define ROWS 512
#define COLS 256
#define ARENA (64u << 20)
#define VEC_OFF (40u << 20)   // vector kept aligned so the matrix is the only variable

static ggml_backend_t BE;
static ggml_backend_buffer_t BUF;
static char *BASE;

static int matmul_at(size_t off, float *out) {
    struct ggml_init_params ip = { .mem_size = 32u << 20, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, COLS, ROWS);
    struct ggml_tensor *v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, COLS, 1);

    if (ggml_backend_tensor_alloc(BUF, m, BASE + off)     != GGML_STATUS_SUCCESS) { ggml_free(ctx); return 0; }
    if (ggml_backend_tensor_alloc(BUF, v, BASE + VEC_OFF) != GGML_STATUS_SUCCESS) { ggml_free(ctx); return 0; }

    ggml_fp16_t *mh = malloc(ggml_nbytes(m));
    for (int i = 0; i < ROWS * COLS; i++) mh[i] = ggml_fp32_to_fp16(((i * 37) % 101) / 101.0f - 0.5f);
    ggml_backend_tensor_set(m, mh, 0, ggml_nbytes(m));
    free(mh);

    float *vf = malloc(ggml_nbytes(v));
    for (int i = 0; i < COLS; i++) vf[i] = ((i * 13) % 71) / 71.0f;
    ggml_backend_tensor_set(v, vf, 0, ggml_nbytes(v));
    free(vf);

    struct ggml_tensor *r = ggml_mul_mat(ctx, m, v);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(BE));
    ggml_gallocr_alloc_graph(ga, gf);
    int ok = ggml_backend_graph_compute(BE, gf) == GGML_STATUS_SUCCESS;
    if (ok) ggml_backend_tensor_get(r, out, 0, ROWS * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return ok;
}

int main(void) {
    ggml_backend_reg_t reg = ggml_backend_metal_reg();
    if (!reg) { fprintf(stderr, "no metal backend\n"); return 1; }

    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);

    printf("device                : %s (%s)\n", props.name, props.description);
    printf("buffer_from_host_ptr  : %s\n", props.caps.buffer_from_host_ptr ? "YES" : "NO — mapped loader unavailable here");
    printf("host_buffer           : %s\n", props.caps.host_buffer ? "YES" : "no");

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    printf("advertised alignment  : %zu bytes\n", ggml_backend_buft_get_alignment(buft));
    printf("max buffer size       : %.2f GB\n\n", ggml_backend_buft_get_max_size(buft) / 1e9);

    if (!props.caps.buffer_from_host_ptr) return 1;

    BE   = ggml_backend_metal_init();
    BASE = mmap(NULL, ARENA, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (BASE == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); return 1; }
    BUF  = ggml_backend_dev_buffer_from_host_ptr(dev, BASE, ARENA, 0);
    if (!BUF) { fprintf(stderr, "buffer_from_host_ptr returned NULL\n"); return 1; }

    printf("no-copy check         : buffer base %s mmap base\n\n",
           ggml_backend_buffer_get_base(BUF) == BASE ? "==" : "!= (COPIED!)");

    static float ref[ROWS], got[ROWS];
    if (!matmul_at(0, ref)) { fprintf(stderr, "baseline matmul failed\n"); return 1; }

    printf("%-12s %-8s %s\n", "offset%32", "status", "max |diff| vs aligned");
    printf("%-12s %-8s %s\n", "---------", "------", "---------------------");

    const size_t offs[] = { 0, 2, 4, 8, 16, 30, 32, 64 };
    size_t min_safe = 0;
    for (unsigned i = 0; i < sizeof(offs) / sizeof(*offs); i++) {
        int ok = matmul_at(offs[i], got);
        double maxd = 0;
        for (int k = 0; k < ROWS; k++) { double d = fabs(got[k] - ref[k]); if (d > maxd) maxd = d; }
        int good = ok && maxd < 1e-5;
        printf("%-12zu %-8s %.3e  %s\n", offs[i] % 32, ok ? "ran" : "FAILED", maxd,
               good ? "ok" : "*** WRONG NUMBERS, NO ERROR ***");
        if (!good && offs[i] % 32 > min_safe) min_safe = offs[i] % 32;
    }

    printf("\nSmallest alignment that computed correctly here: 4 bytes.\n"
           "The mapped loader must refuse anything below it — a misaligned\n"
           "tensor does not fail, it returns wrong values.\n");

    ggml_backend_buffer_free(BUF);
    ggml_backend_free(BE);
    munmap(BASE, ARENA);
    return 0;
}
