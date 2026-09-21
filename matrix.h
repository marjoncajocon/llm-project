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

// Few-row forward matmuls (M < 4 x threads -- single-token decode) are threaded
// over OUTPUT COLUMNS in 64-wide units instead of rows. Each element is still
// one FMA chain in k order, so this is bit-identical to one thread (selftest
// [38]).
//
// Test hook for that claim: 0 = normal dispatch; 1 = every forward matmul runs
// as ONE serial call over all rows; n >= 2 = every forward matmul takes the
// column path in n pieces, run one after another in the calling thread (so the
// column split is exercised even without a thread pool, e.g. on wasm).
void matmul_debug_split(int mode);

// ---------------------------------------------------------------------------
// The persistent thread pool, exported for the non-GEMM training ops (attention,
// softmax, SiLU, RMSNorm, the loss rows). tf_parallel_for(total, fn, ctx) splits
// [0, total) into tf_threads() CONTIGUOUS chunks and calls fn(lo, hi, ctx) once
// per chunk -- the workers take chunks 0..n-2, the caller the last -- then joins.
// Callers keep results bit-identical to one thread by giving every output
// element to exactly one chunk, computed in the same order as the serial loop.
// Not re-entrant: fn must not call tf_parallel_for (or a threaded matmul). On
// wasm (no threads) or with TF_THREADS=1 it is a plain call fn(0, total, ctx).
// ---------------------------------------------------------------------------
typedef void (*tf_range_fn)(int lo, int hi, void *ctx);
void tf_parallel_for(int total, tf_range_fn fn, void *ctx);
int tf_threads(void);
// Test hook (selftest [43]): n >= 1 makes every tf_parallel_for run as n
// contiguous pieces one after another in the calling thread (n = 1: one serial
// call), so chunk boundaries are exercised even with TF_THREADS=1 or on wasm;
// 0 = normal.
void tf_parallel_debug_chunks(int n);

// The lm_head weight-gradient reduction runs over the rows of A in blocks of
// this many rows (each block summed from zero, then added to dB) -- MM_KC for
// the AVX2 kernel, 0 = one unblocked sum for the portable one. A caller that
// drops all-zero rows of dC (llamatrain.c's compact lm_head) calls
// matmul_backward once per original block so the rounding is unchanged.
int matmul_backward_dw_block(void);

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

// C = A @ B^T with B stored (N, K): A (M,K), B (N,K), C (M,N). For a TIED output
// head (logits = hn @ wte^T, wte (vocab, d_model)) -- no transposed copy of B.
// Each output element is one dot product computed identically for any row/column
// split, so it is bit-exact across thread counts and across M (decode vs a full
// forward) -- within one kernel family: an AVX2+FMA kernel when available, else
// portable C (they differ by ~1e-7 relative, like matmul's). No GPU offload.
void matmul_nt(const Tensor *A, const Tensor *B, Tensor *C);

// Backward of C = A @ B^T: dA += dC @ B, and (want_dB) dB += dC^T @ A, where dB
// is B->grad -- for a tied head that is wte.grad, so the head's contribution adds
// to the embedding's. want_dB = 0 is the frozen-base (LoRA) path.
void matmul_nt_backward(const Tensor *A, const Tensor *B, const Tensor *C, int want_dB);

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
