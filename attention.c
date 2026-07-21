// attention.c - Self-attention and transformer block
#include "attention.h"

#include <math.h>

#include "matrix.h"
#include "activation.h"
#include "layernorm.h"

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
) {
  const int seq_len = input->shape[0];
  const int d_v = W_v->shape[1];

  // Temporary tensors for intermediate computations
  int shape_q[2] = {seq_len, d_k};
  int shape_k[2] = {seq_len, d_k};
  int shape_v[2] = {seq_len, d_v};
  int shape_scores[2] = {seq_len, seq_len};
  int shape_attn[2] = {seq_len, d_v};

  Tensor Q = tensor_create(shape_q, 2);
  Tensor K = tensor_create(shape_k, 2);
  Tensor V = tensor_create(shape_v, 2);
  Tensor scores = tensor_create(shape_scores, 2);
  Tensor attn_weights = tensor_create(shape_scores, 2);
  Tensor attn_output = tensor_create(shape_attn, 2);

  // Step 1: Linear projections Q = X @ W_q, K = X @ W_k, V = X @ W_v
  matmul(input, W_q, &Q);
  matmul(input, W_k, &K);
  matmul(input, W_v, &V);

  // Step 2: Compute attention scores: scores = Q @ K^T / sqrt(d_k)
  // K^T is (d_k, seq_len), so we multiply Q (seq_len, d_k) by K^T
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < seq_len; j++) {
      float dot = 0.0f;
      for (int k = 0; k < d_k; k++) {
        dot += Q.data[i * d_k + k] * K.data[j * d_k + k];  // K transposed
      }
      scores.data[i * seq_len + j] = dot / sqrtf((float)d_k);
    }
  }

  // Step 3: Apply softmax to get attention weights
  softmax_rowwise(&scores, &attn_weights);

  // Step 4: Weighted sum: output = attn_weights @ V
  matmul(&attn_weights, &V, &attn_output);

  // Step 5: Final linear projection: output = attn_output @ W_o
  matmul(&attn_output, W_o, output);

  // Clean up temporary tensors
  tensor_free(&Q);
  tensor_free(&K);
  tensor_free(&V);
  tensor_free(&scores);
  tensor_free(&attn_weights);
  tensor_free(&attn_output);
}

// Complete Transformer block: Self-Attention + LayerNorm
void transformer_block_forward(
    const Tensor *input,
    const Tensor *W_q, const Tensor *W_k, const Tensor *W_v, const Tensor *W_o,
    const Tensor *gamma1, const Tensor *beta1,
    const Tensor *gamma2, const Tensor *beta2,
    Tensor *output,
    int d_k
) {
  int shape[2] = {input->shape[0], input->shape[1]};
  Tensor attn_out = tensor_create(shape, 2);
  Tensor norm1_out = tensor_create(shape, 2);
  Tensor residual1 = tensor_create(shape, 2);
  Tensor norm2_out = tensor_create(shape, 2);

  // Self-attention
  self_attention_forward(input, W_q, W_k, W_v, W_o, &attn_out, d_k);

  // Residual connection + LayerNorm 1
  matadd(input, &attn_out, &residual1);
  layernorm(&residual1, gamma1, beta1, &norm1_out, 1e-5f);

  // LayerNorm 2 (simplified - just another normalization)
  layernorm(&norm1_out, gamma2, beta2, output, 1e-5f);

  tensor_free(&attn_out);
  tensor_free(&norm1_out);
  tensor_free(&residual1);
  tensor_free(&norm2_out);
}
