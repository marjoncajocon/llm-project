// layernorm.h - Layer Normalization
#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "tensor.h"

// LayerNorm: normalizes activations across feature dimension
// y = (x - mean) / sqrt(var + epsilon) * gamma + beta
void layernorm(const Tensor *input, const Tensor *gamma, const Tensor *beta,
               Tensor *output, float epsilon);

#endif // LAYERNORM_H
