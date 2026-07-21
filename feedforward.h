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

// Training-mode forward: same result as feedforward(), but also fills the caches
// h1b (pre-activation = input@W1 + b1) and act (= gelu(h1b)) that backward needs.
void feedforward_train(
    const Tensor *input,
    const Tensor *W1, const Tensor *b1,
    const Tensor *W2, const Tensor *b2,
    Tensor *h1b, Tensor *act,            // caches (seq_len, d_ff)
    Tensor *output,
    int d_ff);

// Backward. Reads output->grad; accumulates into input->grad and every param grad
// (W1, b1, W2, b2). h1b/act are the caches from feedforward_train and their grad
// arrays are used as scratch (zero them beforehand).
void feedforward_backward(
    const Tensor *input,
    const Tensor *W1, const Tensor *b1,
    const Tensor *W2, const Tensor *b2,
    Tensor *h1b, Tensor *act,
    Tensor *output,
    int d_ff);

#endif // FEEDFORWARD_H
