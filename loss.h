// loss.h - Softmax cross-entropy loss
#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

// Softmax cross-entropy over rows. Each row of logits is one example; targets[i]
// is the correct class index for row i.
//   logits : (N, vocab_size)
//   targets: N class indices
//   probs  : (N, vocab_size) output, filled with row-softmax (for the backward pass)
// Returns the mean loss over the N rows.
float cross_entropy_forward(const Tensor *logits, const int *targets, Tensor *probs);

// Backward: logits_grad->grad[i,j] += (probs[i,j] - [j == targets[i]]) / N
void cross_entropy_backward(const Tensor *probs, const int *targets, Tensor *logits_grad);

#endif // LOSS_H
