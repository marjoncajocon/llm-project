// rmsnorm.h - Root Mean Square Layer Normalization (LLaMA-family)
#ifndef RMSNORM_H
#define RMSNORM_H

#include "tensor.h"

// RMSNorm: normalizes across the feature dimension by the RMS only — no mean
// subtraction and no bias. This is the LLaMA-family alternative to LayerNorm
// (cheaper, and it works as well in practice).
//   r    = 1 / sqrt(mean(x^2) + epsilon)
//   y_j  = x_j * r * gamma_j
void rmsnorm(const Tensor *input, const Tensor *gamma, Tensor *output, float epsilon);

// Backward of RMSNorm (reads output->grad and input->data). Accumulates:
//   input->grad      += dx      where dx_j = r * (g_j - x_j * r^2 * S / d),
//                                g_j = dy_j * gamma_j,  S = sum_k g_k * x_k
//   gamma_grad->grad += dy_j * (x_j * r)
void rmsnorm_backward(const Tensor *input, const Tensor *gamma, const Tensor *output,
                      Tensor *gamma_grad, float epsilon);

#endif // RMSNORM_H
