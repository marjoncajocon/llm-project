// feedforward.h - Position-wise feed-forward network (MLP)
#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "tensor.h"

// Position-wise feed-forward network: FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
// Applied independently to each row (token) of the input.
void feedforward(
    const Tensor *input,                 // (seq_len, d_model)
    const Tensor *W1, const Tensor *b1,  // (d_model, d_ff), (d_ff)
    const Tensor *W2, const Tensor *b2,  // (d_ff, d_model), (d_model)
    Tensor *output,                      // (seq_len, d_model)
    int d_ff                             // Hidden dimension (typically 4 * d_model)
);

#endif // FEEDFORWARD_H
