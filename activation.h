// activation.h - Activation functions
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"

// Row-wise Softmax with numerical stability
// Applies softmax to each row of input matrix:
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
void softmax_rowwise(const Tensor *input, Tensor *output);

// GELU activation (tanh approximation, as used by GPT-2), element-wise
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu(const Tensor *input, Tensor *output);

#endif // ACTIVATION_H
