// matrix.c - Basic matrix operations
#include "matrix.h"

// Matrix Multiplication: C = A @ B
// A: (M x K), B: (K x N), C: (M x N)
// Uses ikj loop order: the inner loop streams over a row of B and a row of C
// sequentially (cache-friendly, and auto-vectorizable), instead of walking a
// column of B with stride N. Same result as the naive i-j-k form.
void matmul(const Tensor *A, const Tensor *B, Tensor *C) {
  const int M = A->shape[0];
  const int K = A->shape[1];
  const int N = B->shape[1];
  const float *Ad = A->data;
  const float *Bd = B->data;
  float *Cd = C->data;

  for (int i = 0; i < M; i++) {
    float *Ci = &Cd[i * N];
    for (int j = 0; j < N; j++) Ci[j] = 0.0f;
    const float *Ai = &Ad[i * K];
    for (int k = 0; k < K; k++) {
      const float a = Ai[k];
      const float *Bk = &Bd[k * N];
      for (int j = 0; j < N; j++) {
        Ci[j] += a * Bk[j];
      }
    }
  }
}

// Matrix Addition: C = A + B (element-wise)
void matadd(const Tensor *A, const Tensor *B, Tensor *C) {
  for (int i = 0; i < A->size; i++) {
    C->data[i] = A->data[i] + B->data[i];
  }
}

// Scale matrix: C = A * scalar
void matscale(const Tensor *A, float scalar, Tensor *C) {
  for (int i = 0; i < A->size; i++) {
    C->data[i] = A->data[i] * scalar;
  }
}

// Row-broadcast bias add: out[i,j] = x[i,j] + bias[j]
void bias_add(const Tensor *x, const Tensor *bias, Tensor *out) {
  const int M = x->shape[0];
  const int N = x->shape[1];
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      out->data[i * N + j] = x->data[i * N + j] + bias->data[j];
    }
  }
}

// ---------------------------------------------------------------------------
// Backward passes
// ---------------------------------------------------------------------------

// Backward of C = A @ B: dA += dC @ B^T, dB += A^T @ dC
void matmul_backward(const Tensor *A, const Tensor *B, const Tensor *C) {
  const int M = A->shape[0];
  const int K = A->shape[1];
  const int N = B->shape[1];

  // dA[i,k] += sum_j dC[i,j] * B[k,j]
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      float sum = 0.0f;
      for (int j = 0; j < N; j++) {
        sum += C->grad[i * N + j] * B->data[k * N + j];
      }
      A->grad[i * K + k] += sum;
    }
  }

  // dB[k,j] += sum_i A[i,k] * dC[i,j]
  for (int k = 0; k < K; k++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int i = 0; i < M; i++) {
        sum += A->data[i * K + k] * C->grad[i * N + j];
      }
      B->grad[k * N + j] += sum;
    }
  }
}

// Backward of C = A + B: dA += dC, dB += dC
void matadd_backward(const Tensor *C, const Tensor *A, const Tensor *B) {
  for (int i = 0; i < C->size; i++) {
    A->grad[i] += C->grad[i];
    B->grad[i] += C->grad[i];
  }
}

// Backward of bias_add: bias_grad[j] += sum_i out->grad[i,j]
void bias_add_backward(const Tensor *out, const Tensor *bias_grad) {
  const int M = out->shape[0];
  const int N = out->shape[1];
  for (int j = 0; j < N; j++) {
    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
      sum += out->grad[i * N + j];
    }
    bias_grad->grad[j] += sum;
  }
}
