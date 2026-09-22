// backend_mock.c - A CPU-side stand-in for a cuBLAS-style GPU backend.
//
// Always compiled in; selected only explicitly (`--backend mock` or
// TF_BACKEND=mock), never by `auto`. It exists so the backend layer can be tested
// on a machine with no GPU at all:
//
//   * the "device" is host memory behind the same five-function vtable the CUDA
//     and HIP backends give the device-BLAS core (alloc / free / h2d / d2h /
//     gemm), and fresh allocations are POISONED with NaN, so a BLAS call that
//     reads a buffer it was never given (e.g. C with beta = 0) shows up;
//   * the BLAS is tf_fake_cublas_sgemm: a naive COLUMN-major reference with
//     cuBLAS's argument checks. The row-major engine therefore reaches it only
//     through the tf_colmajor_* mapping -- exactly the code CUDA and HIP run --
//     which is what selftest part [49] uses to prove that mapping correct.
//
// It is deliberately a different code path from matrix.c (column-major naive
// loops, explicit operand copies), so agreement with the CPU is evidence rather
// than tautology. It is slow; that is fine -- it is a test instrument.
#include "backend.h"

#include <stdlib.h>
#include <string.h>

static int  g_mock_failed = 0;     // a (simulated) device error disabled it
static int  g_mock_force_fail = 0; // test hook: make every upload fail
static int  g_mock_broken = 0;     // test hook: wrong row/col-major mapping
static long g_mock_gemms = 0;

static void *mock_alloc(size_t bytes) {
  float *p = (float*)malloc(bytes ? bytes : 1);
  if (!p) return NULL;
  // Poison: 0xFF bytes are a quiet NaN in IEEE-754 binary32.
  memset(p, 0xFF, bytes);
  return p;
}
static void mock_free(void *p) { free(p); }
static int mock_h2d(void *dev, const void *host, size_t n) {
  if (g_mock_force_fail) return 1;
  memcpy(dev, host, n);
  return 0;
}
static int mock_d2h(void *host, const void *dev, size_t n) { memcpy(host, dev, n); return 0; }

// The fake cuBLAS, optionally behind a deliberately wrong adapter: "forgot the
// operand swap" -- it passes the operands back in row-major order, as a naive
// port of a row-major GEMM onto cuBLAS would.
static int mock_gemm(void *ctx, int ta, int tb, int m, int n, int k, float alpha,
                     const float *A, int lda, const float *B, int ldb,
                     float beta, float *C, int ldc) {
  g_mock_gemms++;
  if (g_mock_broken)
    return tf_fake_cublas_sgemm(ctx, tb, ta, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc);
  return tf_fake_cublas_sgemm(ctx, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

static TfDevBlas g_mock_db = {
  "mock", mock_alloc, mock_free, mock_h2d, mock_d2h, mock_gemm, NULL,
  {{0}}, 0, {{0}}, 0, 0, 0,
};

static int mock_init(void) { return !g_mock_failed; }
static int mock_available(void) { return !g_mock_failed; }
static int mock_fail(void) { g_mock_failed = 1; return 0; }
static const char *mock_device_name(void) {
  return g_mock_failed ? "mock device disabled after a (simulated) device error"
                       : "mock (host memory + column-major reference BLAS)";
}
static void mock_caps(TfCaps *c) {
  memset(c, 0, sizeof(*c));
  c->unified_memory = 1;     // it IS host memory (copies are kept anyway: that is the test)
  c->deterministic = 1;
  c->n_devices = 1;
  c->min_m = 1;              // no gates: tests use small, ragged shapes
  c->min_macs = 0;
}
static void mock_shutdown(void) {
  tf_devblas_release(&g_mock_db);
  g_mock_failed = 0;         // a fresh init after shutdown starts healthy
}

TF_DEVBLAS_OPS(mock, g_mock_db)

static const TfBackend MOCK_OPS = {
  "mock", "always compiled in", 0,
  mock_init, mock_shutdown, mock_available, mock_device_name, mock_caps,
  mock_sgemm, mock_nt, mock_tn,
  mock_buf_alloc, mock_buf_free, mock_buf_upload, mock_buf_download, mock_gemm_op, NULL,
  mock_register, mock_invalidate,
  {0},
};

const TfBackend *tf_backend_mock(void) { return &MOCK_OPS; }

long backend_mock_gemm_calls(void) { return g_mock_gemms; }
void backend_mock_uploads(long *n, size_t *bytes) {
  if (n) *n = g_mock_db.uploads;
  if (bytes) *bytes = g_mock_db.upload_bytes;
}
void backend_mock_set_broken_mapping(int on) { g_mock_broken = on ? 1 : 0; }
void backend_mock_set_fail(int on) { g_mock_force_fail = on ? 1 : 0; }
