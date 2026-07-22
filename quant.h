// quant.h - int8 weight quantization (per-output-channel, symmetric)
#ifndef QUANT_H
#define QUANT_H

#include "tensor.h"

// Per-column (per-output-channel) symmetric int8 quantization of a row-major
// (rows, cols) weight matrix. The column is the fast-varying index; one scale
// per column (length cols):
//   scale[j] = max_i |w[i,j]| / 127   (or 1 if the column is all zeros)
//   q[i,j]   = round(w[i,j] / scale[j]) clamped to [-127, 127]
//   w[i,j]  ~= q[i,j] * scale[j]
// Used for the LLMQ checkpoint (4x smaller than f32); convert_llama.py writes
// the same layout, the loader dequantizes with dequantize_int8_percol.
void quantize_int8_percol(const float *w, int rows, int cols, signed char *q, float *scale);
void dequantize_int8_percol(const signed char *q, const float *scale, int rows, int cols, float *w);

// A weight matrix kept int8 IN MEMORY (per-output-channel scale), for quantized
// inference: W is (rows, cols), consumed as x(M,rows) @ W -> (M,cols) via qmatmul,
// which dequantizes inline. Reads int8 weights (1 byte) instead of f32 (4 bytes)
// -> 4x less RAM and 4x less weight memory traffic in the matmul inner loop.
typedef struct {
  int rows, cols;
  signed char *q;   // rows*cols, row-major
  float *scale;     // cols (per output channel)
} QTensor;

QTensor qtensor_create(int rows, int cols);
void qtensor_free(QTensor *t);
void qtensor_quantize(QTensor *t, const float *w);   // fill from a (rows, cols) f32 weight

// C[m,j] = scale[j] * sum_k A[m,k] * (float)q[k,j].  A f32 (M, rows), C f32 (M, cols).
void qmatmul(const Tensor *A, const QTensor *W, Tensor *C);

#endif // QUANT_H
