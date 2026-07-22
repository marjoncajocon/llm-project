// matrix.h - Basic matrix operations
#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

#include "tensor.h"

// Matrix Multiplication: C = A @ B
// A: (M x K), B: (K x N), C: (M x N)
void matmul(const Tensor *A, const Tensor *B, Tensor *C);

// Matrix Addition: C = A + B (element-wise)
void matadd(const Tensor *A, const Tensor *B, Tensor *C);

// Scale matrix: C = A * scalar
void matscale(const Tensor *A, float scalar, Tensor *C);

// Row-broadcast bias add: out[i,j] = x[i,j] + bias[j]
// x: (M, N), bias: (N), out: (M, N)
void bias_add(const Tensor *x, const Tensor *bias, Tensor *out);

// ---------------------------------------------------------------------------
// Backward passes (accumulate into the inputs' grad arrays; read out->grad)
// ---------------------------------------------------------------------------

// Backward of C = A @ B: dA += dC @ B^T, dB += A^T @ dC (reads C->grad).
void matmul_backward(const Tensor *A, const Tensor *B, const Tensor *C);

// Backward of C = A + B: dA += dC, dB += dC (reads C->grad).
void matadd_backward(const Tensor *C, const Tensor *A, const Tensor *B);

// Backward of bias_add: bias_grad[j] += sum_i out->grad[i,j] (reads out->grad,
// accumulates into bias_grad->grad).
void bias_add_backward(const Tensor *out, const Tensor *bias_grad);

// Threaded int8 column-accumulate for the quantized (decode) path:
//   acc[j] = sum_k qa[k] * qw[k*N + j]      for j in [0, N)
// Parallelizes over the OUTPUT COLUMNS N (using the same thread pool as matmul),
// so it speeds up single-token decode (M=1) — the case row-parallel matmul can't
// help. Result is bit-identical to the serial loop (each column's sum is
// independent of threading). qw is row-major (K x N).
void qacc_columns(const signed char *qa, const signed char *qw, int K, int N, int32_t *acc);

#endif // MATRIX_H
