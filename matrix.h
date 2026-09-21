// matrix.h - Basic matrix operations
#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

#include "tensor.h"

// Matrix Multiplication: C = A @ B
// A: (M x K), B: (K x N), C: (M x N)
void matmul(const Tensor *A, const Tensor *B, Tensor *C);

// SIMD dispatch for matmul, matmul_backward, matmul_backward_dx and
// qacc_columns. On an x86 CPU with AVX2+FMA, register-blocked intrinsic kernels
// are chosen AT RUNTIME (CPUID), so portable baseline builds get them too; every
// other target runs the portable C kernels. Decided at the first matmul;
// $TF_SIMD=0 forces portable C. The SIMD kernels compute each output element as
// one FMA chain in k order, so they stay bit-exact across row counts (decode vs
// full forward) and thread counts -- but are NOT bit-identical to the portable
// kernels (~1e-7 relative; selftest part [31]).
int matmul_simd_available(void);     // this build AND this CPU can run them
int matmul_simd_enabled(void);       // currently in use
void matmul_set_simd(int on);        // force on/off (on is ignored if unavailable)
const char *matmul_simd_name(void);  // "avx2+fma" or "portable C"

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
//
// NUMERICS (portable kernels; the AVX2 ones are described above):
// dB accumulates in the same order as the naive scalar loop and is
// bit-exact. dA is a dot-product reduction computed with several independent
// accumulators so it can vectorize, which REASSOCIATES the sum — it is not
// bit-identical to a naive left-to-right scalar sum (it is typically slightly
// more accurate). Threading remains bit-exact: each output row is produced
// entirely by one thread, so results do not depend on the thread count.
void matmul_backward(const Tensor *A, const Tensor *B, const Tensor *C);

// Just the input-gradient half of matmul_backward: dA += dC @ B^T, and B->grad
// is neither read nor written. Use when B is FROZEN (e.g. the base weights under
// LoRA fine-tuning) — it skips half the backward's work.
void matmul_backward_dx(const Tensor *A, const Tensor *B, const Tensor *C);

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
