// rmsnorm.c - Root Mean Square Layer Normalization (LLaMA-family)
#include "rmsnorm.h"

#include <math.h>
#include <stdlib.h>

#include "matrix.h"   // tf_parallel_for

// Rows are independent in the forward and in the input-gradient half of the
// backward, so big inputs are split by rows across the thread pool. The gamma
// gradient sums over ROWS, so it is a second pass split by COLUMNS instead: each
// gamma_j is still accumulated over i ascending with the same expression. Both
// keep every thread count bit-identical to the serial loops (selftest [43]).
#define RMS_THREAD_MIN (1L << 15)   // elements; below this, one thread

typedef struct {
  const Tensor *input, *gamma, *output; Tensor *gamma_grad;
  float epsilon; float *r;          // per-row 1/rms, backward only (may be NULL)
} RmsJob;

// y_j = x_j * r * gamma_j,   r = 1 / sqrt(mean(x^2) + epsilon)
static void rms_fwd_rows(int i0, int i1, void *ctx) {
  const RmsJob *t = (const RmsJob*)ctx;
  const int cols = t->input->shape[1];
  const float inv_n = 1.0f / (float)cols;
  for (int i = i0; i < i1; i++) {
    const float *x = &t->input->data[(size_t)i * cols];
    float ms = 0.0f;
    for (int j = 0; j < cols; j++) ms += x[j] * x[j];
    ms *= inv_n;
    float r = 1.0f / sqrtf(ms + t->epsilon);
    for (int j = 0; j < cols; j++)
      t->output->data[(size_t)i * cols + j] = x[j] * r * t->gamma->data[j];
  }
}

static int rms_big(const Tensor *t) {
  return t->shape[0] > 1 && (size_t)t->shape[0] * (size_t)t->shape[1] >= (size_t)RMS_THREAD_MIN;
}

void rmsnorm(const Tensor *input, const Tensor *gamma, Tensor *output, float epsilon) {
  RmsJob t = { input, gamma, output, NULL, epsilon, NULL };
  if (rms_big(input)) tf_parallel_for(input->shape[0], rms_fwd_rows, &t);
  else rms_fwd_rows(0, input->shape[0], &t);
}

// Recomputes r per row from input->data (like layernorm_backward). Derivation:
//   n_j = x_j * r,  y_j = gamma_j * n_j,  r = (ms+eps)^(-1/2),  ms = mean(x^2)
//   dr/dx_k = -r^3 * x_k / d   =>   dx_k = r * (g_k - x_k * r^2 * S / d)
// with g_k = dy_k * gamma_k and S = sum_j g_j * x_j.
static void rms_bwd_rows(int i0, int i1, void *ctx) {
  const RmsJob *t = (const RmsJob*)ctx;
  const Tensor *input = t->input, *gamma = t->gamma, *output = t->output;
  const int cols = input->shape[1];
  const float inv_n = 1.0f / (float)cols;

  for (int i = i0; i < i1; i++) {
    const float *x = &input->data[(size_t)i * cols];
    const float *dy = &output->grad[(size_t)i * cols];

    float ms = 0.0f;
    for (int j = 0; j < cols; j++) ms += x[j] * x[j];
    ms *= inv_n;
    float r = 1.0f / sqrtf(ms + t->epsilon);

    float S = 0.0f;                       // sum_j (dy_j * gamma_j) * x_j
    for (int j = 0; j < cols; j++) S += dy[j] * gamma->data[j] * x[j];
    const float r2S_n = r * r * S * inv_n;

    for (int j = 0; j < cols; j++) {
      float g = dy[j] * gamma->data[j];
      input->grad[(size_t)i * cols + j] += r * (g - x[j] * r2S_n);
    }
    if (t->r) t->r[i] = r;
    else                                  // serial: the gamma term in the same pass
      for (int j = 0; j < cols; j++)
        t->gamma_grad->grad[j] += dy[j] * x[j] * r;
  }
}

// gamma_grad[j] += sum_i dy[i,j] * x[i,j] * r_i, i ascending, for j in [j0, j1).
static void rms_bwd_gamma_cols(int j0, int j1, void *ctx) {
  const RmsJob *t = (const RmsJob*)ctx;
  const int rows = t->input->shape[0], cols = t->input->shape[1];
  float *gg = t->gamma_grad->grad;
  for (int i = 0; i < rows; i++) {
    const float *x = &t->input->data[(size_t)i * cols];
    const float *dy = &t->output->grad[(size_t)i * cols];
    const float r = t->r[i];
    for (int j = j0; j < j1; j++) gg[j] += dy[j] * x[j] * r;
  }
}

void rmsnorm_backward(const Tensor *input, const Tensor *gamma, const Tensor *output,
                      Tensor *gamma_grad, float epsilon) {
  const int rows = input->shape[0], cols = input->shape[1];
  RmsJob t = { input, gamma, output, gamma_grad, epsilon, NULL };
  if (rms_big(input)) t.r = (float*)malloc((size_t)rows * sizeof(float));
  if (!t.r) { rms_bwd_rows(0, rows, &t); return; }
  tf_parallel_for(rows, rms_bwd_rows, &t);
  tf_parallel_for(cols, rms_bwd_gamma_cols, &t);
  free(t.r);
}

// A (rows * cols / head_dim, head_dim) view of t sharing its data and grad, so a
// per-head norm is just the plain row norm over more, shorter rows.
static Tensor heads_view(const Tensor *t, int head_dim, int *shape) {
  Tensor v = *t;
  shape[0] = (int)(t->size / (size_t)head_dim);
  shape[1] = head_dim;
  v.shape = shape;
  v.ndim = 2;
  return v;
}

void rmsnorm_heads(const Tensor *input, const Tensor *gamma, Tensor *output, float epsilon) {
  const int hd = gamma->shape[0];
  int si[2], so[2];
  Tensor vi = heads_view(input, hd, si), vo = heads_view(output, hd, so);
  rmsnorm(&vi, gamma, &vo, epsilon);
}

void rmsnorm_heads_backward(const Tensor *input, const Tensor *gamma, const Tensor *output,
                            Tensor *gamma_grad, float epsilon) {
  const int hd = gamma->shape[0];
  int si[2], so[2];
  Tensor vi = heads_view(input, hd, si), vo = heads_view(output, hd, so);
  rmsnorm_backward(&vi, gamma, &vo, gamma_grad, epsilon);
}
