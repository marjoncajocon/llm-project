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

#endif // MATRIX_H
