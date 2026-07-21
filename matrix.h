// matrix.h - Basic matrix operations
#ifndef MATRIX_H
#define MATRIX_H

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

#endif // MATRIX_H
