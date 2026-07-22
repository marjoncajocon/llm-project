// quant.c - int8 weight quantization (per-output-channel, symmetric)
#include "quant.h"

#include <math.h>
#include <stdlib.h>
#include <stdint.h>

void quantize_int8_percol(const float *w, int rows, int cols, signed char *q, float *scale) {
  for (int j = 0; j < cols; j++) {
    float mx = 0.0f;
    for (int i = 0; i < rows; i++) {
      float a = fabsf(w[i * cols + j]);
      if (a > mx) mx = a;
    }
    float s = (mx > 0.0f) ? mx / 127.0f : 1.0f;
    scale[j] = s;
    float inv = 1.0f / s;
    for (int i = 0; i < rows; i++) {
      int v = (int)lrintf(w[i * cols + j] * inv);
      if (v > 127) v = 127;
      if (v < -127) v = -127;
      q[i * cols + j] = (signed char)v;
    }
  }
}

void dequantize_int8_percol(const signed char *q, const float *scale, int rows, int cols, float *w) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      w[i * cols + j] = (float)q[i * cols + j] * scale[j];
}

QTensor qtensor_create(int rows, int cols) {
  QTensor t;
  t.rows = rows; t.cols = cols;
  t.q = (signed char*)malloc((size_t)rows * cols);
  t.scale = (float*)malloc((size_t)cols * sizeof(float));
  return t;
}

void qtensor_free(QTensor *t) {
  free(t->q); free(t->scale);
  t->q = NULL; t->scale = NULL;
}

void qtensor_quantize(QTensor *t, const float *w) {
  quantize_int8_percol(w, t->rows, t->cols, t->q, t->scale);
}

// int8 x int8 -> int32 matmul (the fast quantized path). Activations are
// quantized to int8 per row on the fly, then the inner product is an integer
// multiply-accumulate into an int32 buffer -- the `acc[j] += av * qr[j]` loop is
// contiguous int8 x int8 and auto-vectorizes to many MACs per instruction
// (SSE/AVX). Result is dequantized by a_scale[m] * W.scale[j].
//   C[m,j] = a_scale[m] * W.scale[j] * sum_k qa[m,k] * qw[k,j]
// int32 accumulator is safe: max |qa*qw| = 127*127, times K (<= a few thousand)
// stays well under 2^31.
void qmatmul(const Tensor *A, const QTensor *W, Tensor *C) {
  const int M = A->shape[0], K = W->rows, N = W->cols;
  int32_t *acc = (int32_t*)malloc((size_t)N * sizeof(int32_t));
  signed char *qa = (signed char*)malloc((size_t)K);
  const float *ws = W->scale;

  for (int m = 0; m < M; m++) {
    const float *a = &A->data[(size_t)m * K];
    // quantize this activation row to int8 (per-row scale)
    float mx = 0.0f;
    for (int k = 0; k < K; k++) { float t = fabsf(a[k]); if (t > mx) mx = t; }
    float as = (mx > 0.0f) ? mx / 127.0f : 1.0f;
    float inv = 1.0f / as;
    for (int k = 0; k < K; k++) {
      int v = (int)lrintf(a[k] * inv);
      if (v > 127) v = 127;
      if (v < -127) v = -127;
      qa[k] = (signed char)v;
    }
    // integer i-k-j accumulate
    for (int j = 0; j < N; j++) acc[j] = 0;
    for (int k = 0; k < K; k++) {
      int av = qa[k];
      const signed char *qr = &W->q[(size_t)k * N];
      for (int j = 0; j < N; j++) acc[j] += av * (int)qr[j];
    }
    float *c = &C->data[(size_t)m * N];
    for (int j = 0; j < N; j++) c[j] = as * ws[j] * (float)acc[j];
  }
  free(acc); free(qa);
}
