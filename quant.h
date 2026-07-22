// quant.h - int8 weight quantization (per-output-channel, symmetric)
#ifndef QUANT_H
#define QUANT_H

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

#endif // QUANT_H
