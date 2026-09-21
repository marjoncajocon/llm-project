// rmsnorm.c - Root Mean Square Layer Normalization (LLaMA-family)
#include "rmsnorm.h"

#include <math.h>

// y_j = x_j * r * gamma_j,   r = 1 / sqrt(mean(x^2) + epsilon)
void rmsnorm(const Tensor *input, const Tensor *gamma, Tensor *output, float epsilon) {
  const int rows = input->shape[0];
  const int cols = input->shape[1];
  const float inv_n = 1.0f / (float)cols;

  for (int i = 0; i < rows; i++) {
    const float *x = &input->data[i * cols];
    float ms = 0.0f;
    for (int j = 0; j < cols; j++) ms += x[j] * x[j];
    ms *= inv_n;
    float r = 1.0f / sqrtf(ms + epsilon);
    for (int j = 0; j < cols; j++)
      output->data[i * cols + j] = x[j] * r * gamma->data[j];
  }
}

// Recomputes r per row from input->data (like layernorm_backward). Derivation:
//   n_j = x_j * r,  y_j = gamma_j * n_j,  r = (ms+eps)^(-1/2),  ms = mean(x^2)
//   dr/dx_k = -r^3 * x_k / d   =>   dx_k = r * (g_k - x_k * r^2 * S / d)
// with g_k = dy_k * gamma_k and S = sum_j g_j * x_j.
void rmsnorm_backward(const Tensor *input, const Tensor *gamma, const Tensor *output,
                      Tensor *gamma_grad, float epsilon) {
  const int rows = input->shape[0];
  const int cols = input->shape[1];
  const float inv_n = 1.0f / (float)cols;

  for (int i = 0; i < rows; i++) {
    const float *x = &input->data[i * cols];
    const float *dy = &output->grad[i * cols];

    float ms = 0.0f;
    for (int j = 0; j < cols; j++) ms += x[j] * x[j];
    ms *= inv_n;
    float r = 1.0f / sqrtf(ms + epsilon);

    float S = 0.0f;                       // sum_j (dy_j * gamma_j) * x_j
    for (int j = 0; j < cols; j++) S += dy[j] * gamma->data[j] * x[j];
    const float r2S_n = r * r * S * inv_n;

    for (int j = 0; j < cols; j++) {
      float g = dy[j] * gamma->data[j];
      input->grad[i * cols + j] += r * (g - x[j] * r2S_n);
      gamma_grad->grad[j] += dy[j] * x[j] * r;
    }
  }
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
