// rope.h - Rotary Position Embeddings (RoPE, LLaMA-family)
#ifndef ROPE_H
#define ROPE_H

#include "tensor.h"

// RoPE rotates each row (a query/key head vector; head_dim = cols, must be even)
// by an angle that depends on the row's POSITION and the feature-pair index,
// encoding RELATIVE position directly into the Q·K dot product. Parameter-free —
// it replaces learned/absolute positional embeddings in LLaMA-family models.
//
// Half-split ("rotate_half") convention, matching HuggingFace LLaMA, with
// h = cols/2 and inv_freq_k = base^(-2k/cols):
//   y_k     = x_k * cos(pos*inv_freq_k) - x_{k+h} * sin(pos*inv_freq_k)
//   y_{k+h} = x_{k+h} * cos(pos*inv_freq_k) + x_k * sin(pos*inv_freq_k)
// The row index i is the position; `base` is typically 10000.
void rope(const Tensor *input, Tensor *output, float base);

// Backward: accumulates input->grad from output->grad. The rotation is
// orthogonal, so dx is the inverse rotation of dy (no parameters to learn).
void rope_backward(const Tensor *input, const Tensor *output, float base);

#endif // ROPE_H
