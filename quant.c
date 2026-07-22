// quant.c - int8 weight quantization (per-output-channel, symmetric)
#include "quant.h"

#include <math.h>
#include <stdlib.h>

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

// i-k-j order (cache-friendly, matches matmul): stream a row of q and of C.
void qmatmul(const Tensor *A, const QTensor *W, Tensor *C) {
  const int M = A->shape[0], K = W->rows, N = W->cols;
  for (int m = 0; m < M; m++) {
    float *c = &C->data[(size_t)m * N];
    for (int j = 0; j < N; j++) c[j] = 0.0f;
    const float *a = &A->data[(size_t)m * K];
    for (int k = 0; k < K; k++) {
      float av = a[k];
      const signed char *qr = &W->q[(size_t)k * N];
      for (int j = 0; j < N; j++) c[j] += av * (float)qr[j];
    }
    for (int j = 0; j < N; j++) c[j] *= W->scale[j];
  }
}
