// matrix.c - Basic matrix operations
#include "matrix.h"

// Matrix Multiplication: C = A @ B
// A: (M x K), B: (K x N), C: (M x N)
void matmul(const Tensor *A, const Tensor *B, Tensor *C) {
  const int M = A->shape[0];
  const int K = A->shape[1];
  const int N = B->shape[1];

  // GEMM: C[i,j] = sum(A[i,k] * B[k,j]) for k=0..K-1
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A->data[i * K + k] * B->data[k * N + j];
      }
      C->data[i * N + j] = sum;
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
