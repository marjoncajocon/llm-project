// attention.h - Self-attention and transformer block
#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"

// Multi-head causal Self-Attention forward pass.
// Projects the input into n_heads independent heads, applies scaled dot-product
// attention with a causal mask (each position attends only to itself and earlier
// positions), concatenates the heads, and projects back through W_o.
// All projection matrices are square (d_model, d_model); head_dim = d_model / n_heads.
void multihead_attention_forward(
    const Tensor *input,          // (seq_len, d_model)
    const Tensor *W_q,            // (d_model, d_model)
    const Tensor *W_k,            // (d_model, d_model)
    const Tensor *W_v,            // (d_model, d_model)
    const Tensor *W_o,            // (d_model, d_model)
    Tensor *output,               // (seq_len, d_model)
    int n_heads                   // Number of attention heads (must divide d_model)
);

// Training-mode multi-head attention: same result as multihead_attention_forward,
// but caches the intermediates backward needs:
//   Q, K, V : (seq_len, d_model)
//   weights : (n_heads * seq_len, seq_len)  softmax attention weights per head
//   concat  : (seq_len, d_model)            concatenated per-head outputs (pre-W_o)
void multihead_attention_train(
    const Tensor *input,
    const Tensor *W_q, const Tensor *W_k, const Tensor *W_v, const Tensor *W_o,
    Tensor *Q, Tensor *K, Tensor *V, Tensor *weights, Tensor *concat,
    Tensor *output,
    int n_heads);

// Backward of multi-head attention. Reads output->grad; accumulates input->grad and
// W_q/W_k/W_v/W_o grads. Q/K/V/concat grad arrays are used as scratch (zero beforehand).
void multihead_attention_backward(
    const Tensor *input,
    const Tensor *W_q, const Tensor *W_k, const Tensor *W_v, const Tensor *W_o,
    Tensor *Q, Tensor *K, Tensor *V, Tensor *weights, Tensor *concat,
    Tensor *output,
    int n_heads);

// Complete Transformer block (pre-norm): Multi-head Attention + FFN with residuals
//   a = input + MultiHeadAttention(LN1(input))
//   output = a + FFN(LN2(a))
void transformer_block_forward(
    const Tensor *input,
    const Tensor *W_q, const Tensor *W_k, const Tensor *W_v, const Tensor *W_o,
    const Tensor *gamma1, const Tensor *beta1,
    const Tensor *gamma2, const Tensor *beta2,
    const Tensor *ff_W1, const Tensor *ff_b1,
    const Tensor *ff_W2, const Tensor *ff_b2,
    Tensor *output,
    int n_heads, int d_ff
);

#endif // ATTENTION_H
