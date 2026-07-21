// attention.h - Self-attention and transformer block
#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"

// Single-head Self-Attention forward pass
// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
void self_attention_forward(
    const Tensor *input,          // (seq_len, d_model)
    const Tensor *W_q,            // (d_model, d_k)
    const Tensor *W_k,            // (d_model, d_k)
    const Tensor *W_v,            // (d_model, d_v)
    const Tensor *W_o,            // (d_v, d_model)
    Tensor *output,               // (seq_len, d_model)
    int d_k                       // Dimension of key/query
);

// Complete Transformer block: Self-Attention + LayerNorm
void transformer_block_forward(
    const Tensor *input,
    const Tensor *W_q, const Tensor *W_k, const Tensor *W_v, const Tensor *W_o,
    const Tensor *gamma1, const Tensor *beta1,
    const Tensor *gamma2, const Tensor *beta2,
    Tensor *output,
    int d_k
);

#endif // ATTENTION_H
