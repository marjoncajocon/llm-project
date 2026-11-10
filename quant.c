// quant.c - int8 weight quantization (per-output-channel, symmetric)
#include "quant.h"

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "matrix.h"

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
    // integer i-k-j accumulate (threaded over the output columns N)
    qacc_columns(qa, W->q, K, N, acc);
    float *c = &C->data[(size_t)m * N];
    for (int j = 0; j < N; j++) c[j] = as * ws[j] * (float)acc[j];
  }
  free(acc); free(qa);
}

// ---------------------------------------------------------------------------
// int4, group-wise (see quant.h)
// ---------------------------------------------------------------------------

Q4Tensor q4tensor_create(int rows, int cols, int group) {
  Q4Tensor t;
  t.rows = rows; t.cols = cols;
  t.group = group > 0 ? group : 32;
  t.n_groups = (rows + t.group - 1) / t.group;
  t.q = (unsigned char*)calloc((size_t)((rows + 1) / 2) * cols, 1);
  t.scale = (float*)calloc((size_t)t.n_groups * cols, sizeof(float));
  return t;
}

void q4tensor_free(Q4Tensor *t) {
  free(t->q); free(t->scale);
  t->q = NULL; t->scale = NULL;
}

size_t q4tensor_bytes(const Q4Tensor *t) {
  return (size_t)((t->rows + 1) / 2) * t->cols + (size_t)t->n_groups * t->cols * sizeof(float);
}

static int q4_code(const Q4Tensor *t, int i, int j) {
  const unsigned char b = t->q[(size_t)(i / 2) * t->cols + j];
  return (int)((i & 1) ? (b >> 4) : (b & 15)) - 8;
}

void q4tensor_quantize(Q4Tensor *t, const float *w) {
  const int R = t->rows, N = t->cols, G = t->group;
  memset(t->q, 0, (size_t)((R + 1) / 2) * N);
  for (int g = 0; g < t->n_groups; g++) {
    const int i0 = g * G, i1 = (i0 + G < R) ? i0 + G : R;
    for (int j = 0; j < N; j++) {
      float mx = 0.0f;
      for (int i = i0; i < i1; i++) { const float a = fabsf(w[(size_t)i * N + j]); if (a > mx) mx = a; }
      // Clipping search: max/7 lets ONE outlier set the step for all 32 weights.
      // Try shrinking the scale (the outlier then clips to +-7) and keep whichever
      // gives the least squared error over the group.
      float s = mx > 0.0f ? mx / 7.0f : 1.0f;
      if (mx > 0.0f) {
        float best = -1.0f;
        const float base = s;
        for (int f = 0; f <= 8; f++) {
          const float cs = base * (1.0f - 0.05f * (float)f), inv = 1.0f / cs;
          float err = 0.0f;
          for (int i = i0; i < i1; i++) {
            const float x = w[(size_t)i * N + j];
            int v = (int)lrintf(x * inv);
            if (v > 7) v = 7;
            if (v < -7) v = -7;
            const float e = x - cs * (float)v;
            err += e * e;
          }
          if (best < 0.0f || err < best) { best = err; s = cs; }
        }
      }
      t->scale[(size_t)g * N + j] = s;
      const float inv = 1.0f / s;
      for (int i = i0; i < i1; i++) {
        int v = (int)lrintf(w[(size_t)i * N + j] * inv);
        if (v > 7) v = 7;
        if (v < -7) v = -7;
        unsigned char *b = &t->q[(size_t)(i / 2) * N + j];
        *b |= (unsigned char)((v + 8) << ((i & 1) ? 4 : 0));
      }
    }
  }
}

void q4tensor_dequantize(const Q4Tensor *t, float *w) {
  const int N = t->cols;
  for (int i = 0; i < t->rows; i++) {
    const float *s = &t->scale[(size_t)(i / t->group) * N];
    for (int j = 0; j < N; j++) w[(size_t)i * N + j] = s[j] * (float)q4_code(t, i, j);
  }
}

void q4matmul(const Tensor *A, const Q4Tensor *W, Tensor *C) {
  const int M = A->shape[0], K = W->rows, N = W->cols, G = W->group;
  float *wg = (float*)malloc((size_t)G * N * sizeof(float));   // one dequantized group
  memset(C->data, 0, (size_t)M * N * sizeof(float));
  for (int g = 0; g < W->n_groups; g++) {
    const int i0 = g * G, i1 = (i0 + G < K) ? i0 + G : K;
    const float *s = &W->scale[(size_t)g * N];
    for (int i = i0; i < i1; i++) {
      float *row = &wg[(size_t)(i - i0) * N];
      const unsigned char *b = &W->q[(size_t)(i / 2) * N];
      const int sh = (i & 1) ? 4 : 0;
      for (int j = 0; j < N; j++) row[j] = s[j] * (float)((int)((b[j] >> sh) & 15) - 8);
    }
    for (int m = 0; m < M; m++) {
      const float *a = &A->data[(size_t)m * K];
      float *c = &C->data[(size_t)m * N];
      for (int i = i0; i < i1; i++) {
        const float av = a[i];
        const float *row = &wg[(size_t)(i - i0) * N];
        for (int j = 0; j < N; j++) c[j] += av * row[j];
      }
    }
  }
  free(wg);
}
