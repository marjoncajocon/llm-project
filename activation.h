// activation.h - Activation functions
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"

// Row-wise Softmax with numerical stability
// Applies softmax to each row of input matrix:
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
void softmax_rowwise(const Tensor *input, Tensor *output);

#endif // ACTIVATION_H
