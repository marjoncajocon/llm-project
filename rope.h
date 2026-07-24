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

// In-place multi-head RoPE: x is (seq_len, n_heads * head_dim); row i is rotated
// by position (pos0 + i). rope_heads is the pos0 = 0 case (a full prefill);
// rope_heads_pos with a single row and pos0 = pos is used in incremental decode.
void rope_heads_pos(Tensor *x, int n_heads, float base, int pos0);
void rope_heads(Tensor *x, int n_heads, float base);

// Backward of the in-place multi-head RoPE: replaces x->grad with the INVERSE
// rotation of itself (dx = R(θ)^T dy), matching the in-place forward above.
// RoPE is parameter-free, so there is nothing else to accumulate.
//
// NOTE: unlike the other *_backward here, this TRANSFORMS x->grad in place rather
// than accumulating into a separate input — the in-place forward means the
// pre- and post-rotation gradients live in the same buffer. Call it exactly once,
// between the consumer's backward (which fills x->grad) and the producer's.
void rope_heads_backward_pos(Tensor *x, int n_heads, float base, int pos0);
void rope_heads_backward(Tensor *x, int n_heads, float base);

#endif // ROPE_H
