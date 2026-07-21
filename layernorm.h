// layernorm.h - Layer Normalization
#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "tensor.h"

// LayerNorm: normalizes activations across feature dimension
// y = (x - mean) / sqrt(var + epsilon) * gamma + beta
void layernorm(const Tensor *input, const Tensor *gamma, const Tensor *beta,
               Tensor *output, float epsilon);

// Backward of LayerNorm (reads output->grad; input->data). Accumulates:
//   input->grad     += dx
//   gamma_grad->grad += dy * xhat
//   beta_grad->grad  += dy
// gamma supplies the scale values; gamma_grad/beta_grad receive the parameter grads.
void layernorm_backward(const Tensor *input, const Tensor *gamma, const Tensor *output,
                        Tensor *gamma_grad, Tensor *beta_grad, float epsilon);

#endif // LAYERNORM_H
