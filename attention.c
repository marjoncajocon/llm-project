// attention.c - Self-attention and transformer block
#include "attention.h"

#include <math.h>
#include <string.h>

#include "matrix.h"
#include "activation.h"
#include "layernorm.h"
#include "feedforward.h"

// Multi-head causal Self-Attention forward pass.
// Attention(Q, K, V) = softmax(mask(Q @ K^T / sqrt(head_dim))) @ V, per head.
void multihead_attention_forward(
    const Tensor *input,          // (seq_len, d_model)
    const Tensor *W_q,            // (d_model, d_model)
    const Tensor *W_k,            // (d_model, d_model)
    const Tensor *W_v,            // (d_model, d_model)
    const Tensor *W_o,            // (d_model, d_model)
    Tensor *output,               // (seq_len, d_model)
    int n_heads                   // Number of attention heads (must divide d_model)
) {
  const int seq_len = input->shape[0];
  const int d_model = input->shape[1];
  const int head_dim = d_model / n_heads;
  const float scale = 1.0f / sqrtf((float)head_dim);

  // Full projections (all heads at once): Q,K,V = X @ W_{q,k,v}  -> (seq_len, d_model)
  int shape_proj[2] = {seq_len, d_model};
  Tensor Q = tensor_create(shape_proj, 2);
  Tensor K = tensor_create(shape_proj, 2);
  Tensor V = tensor_create(shape_proj, 2);
  Tensor concat = tensor_create(shape_proj, 2);  // concatenated per-head outputs

  matmul(input, W_q, &Q);
  matmul(input, W_k, &K);
  matmul(input, W_v, &V);

  // Per-head scratch buffers for scores and softmax weights
  int shape_scores[2] = {seq_len, seq_len};
  Tensor scores = tensor_create(shape_scores, 2);
  Tensor weights = tensor_create(shape_scores, 2);

  for (int h = 0; h < n_heads; h++) {
    const int off = h * head_dim;  // this head occupies columns [off, off + head_dim)

    // Scores = Q_h @ K_h^T * scale, with a causal mask (no attending to the future)
    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < seq_len; j++) {
        if (j > i) {
          scores.data[i * seq_len + j] = -INFINITY;  // mask future positions
        } else {
          float dot = 0.0f;
          for (int d = 0; d < head_dim; d++) {
            dot += Q.data[i * d_model + off + d] * K.data[j * d_model + off + d];
          }
          scores.data[i * seq_len + j] = dot * scale;
        }
      }
    }

    // Softmax over each row (masked entries -> exp(-inf) = 0)
    softmax_rowwise(&scores, &weights);

    // head_out = weights @ V_h, written into this head's columns of concat
    for (int i = 0; i < seq_len; i++) {
      for (int d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
          sum += weights.data[i * seq_len + j] * V.data[j * d_model + off + d];
        }
        concat.data[i * d_model + off + d] = sum;
      }
    }
  }

  // Output projection: output = concat @ W_o
  matmul(&concat, W_o, output);

  tensor_free(&Q);
  tensor_free(&K);
  tensor_free(&V);
  tensor_free(&concat);
  tensor_free(&scores);
  tensor_free(&weights);
}

// Training-mode multi-head attention: caches Q, K, V, per-head weights, and concat.
void multihead_attention_train(
    const Tensor *input,
    const Tensor *W_q, const Tensor *W_k, const Tensor *W_v, const Tensor *W_o,
    Tensor *Q, Tensor *K, Tensor *V, Tensor *weights, Tensor *concat,
    Tensor *output,
    int n_heads) {
  const int seq_len = input->shape[0];
  const int d_model = input->shape[1];
  const int head_dim = d_model / n_heads;
  const float scale = 1.0f / sqrtf((float)head_dim);

  matmul(input, W_q, Q);
  matmul(input, W_k, K);
  matmul(input, W_v, V);

  int shape_scores[2] = {seq_len, seq_len};
  Tensor scores = tensor_create(shape_scores, 2);
  Tensor w = tensor_create(shape_scores, 2);

  for (int h = 0; h < n_heads; h++) {
    const int off = h * head_dim;

    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < seq_len; j++) {
        if (j > i) {
          scores.data[i * seq_len + j] = -INFINITY;
        } else {
          float dot = 0.0f;
          for (int d = 0; d < head_dim; d++) {
            dot += Q->data[i * d_model + off + d] * K->data[j * d_model + off + d];
          }
          scores.data[i * seq_len + j] = dot * scale;
        }
      }
    }

    softmax_rowwise(&scores, &w);
    // Cache this head's attention weights
    memcpy(&weights->data[h * seq_len * seq_len], w.data,
           seq_len * seq_len * sizeof(float));

    for (int i = 0; i < seq_len; i++) {
      for (int d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
          sum += w.data[i * seq_len + j] * V->data[j * d_model + off + d];
        }
        concat->data[i * d_model + off + d] = sum;
      }
    }
  }

  matmul(concat, W_o, output);

  tensor_free(&scores);
  tensor_free(&w);
}

// Backward of multi-head attention.
void multihead_attention_backward(
    const Tensor *input,
    const Tensor *W_q, const Tensor *W_k, const Tensor *W_v, const Tensor *W_o,
    Tensor *Q, Tensor *K, Tensor *V, Tensor *weights, Tensor *concat,
    Tensor *output,
    int n_heads) {
  const int seq_len = input->shape[0];
  const int d_model = input->shape[1];
  const int head_dim = d_model / n_heads;
  const float scale = 1.0f / sqrtf((float)head_dim);

  // output = concat @ W_o
  matmul_backward(concat, W_o, output);  // concat->grad, W_o->grad

  int shape_scores[2] = {seq_len, seq_len};
  Tensor dw = tensor_create(shape_scores, 2);      // d(attention weights)
  Tensor dscore = tensor_create(shape_scores, 2);  // d(scores)

  for (int h = 0; h < n_heads; h++) {
    const int off = h * head_dim;
    const float *w = &weights->data[h * seq_len * seq_len];

    // concat[i,off+d] = sum_j w[i,j] * V[j,off+d]
    // dw[i,j] = sum_d dconcat[i,off+d] * V[j,off+d]
    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < seq_len; j++) {
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          s += concat->grad[i * d_model + off + d] * V->data[j * d_model + off + d];
        }
        dw.data[i * seq_len + j] = s;
      }
    }
    // dV[j,off+d] += sum_i w[i,j] * dconcat[i,off+d]
    for (int j = 0; j < seq_len; j++) {
      for (int d = 0; d < head_dim; d++) {
        float s = 0.0f;
        for (int i = 0; i < seq_len; i++) {
          s += w[i * seq_len + j] * concat->grad[i * d_model + off + d];
        }
        V->grad[j * d_model + off + d] += s;
      }
    }
    // Softmax backward: dscore[i,j] = w[i,j] * (dw[i,j] - sum_k dw[i,k]*w[i,k])
    for (int i = 0; i < seq_len; i++) {
      float dot = 0.0f;
      for (int k = 0; k < seq_len; k++) {
        dot += dw.data[i * seq_len + k] * w[i * seq_len + k];
      }
      for (int j = 0; j < seq_len; j++) {
        dscore.data[i * seq_len + j] = w[i * seq_len + j] * (dw.data[i * seq_len + j] - dot);
      }
    }
    // scores[i,j] = scale * sum_d Q[i,off+d]*K[j,off+d]
    // dQ[i,off+d] += scale * sum_j dscore[i,j]*K[j,off+d]
    for (int i = 0; i < seq_len; i++) {
      for (int d = 0; d < head_dim; d++) {
        float s = 0.0f;
        for (int j = 0; j < seq_len; j++) {
          s += dscore.data[i * seq_len + j] * K->data[j * d_model + off + d];
        }
        Q->grad[i * d_model + off + d] += scale * s;
      }
    }
    // dK[j,off+d] += scale * sum_i dscore[i,j]*Q[i,off+d]
    for (int j = 0; j < seq_len; j++) {
      for (int d = 0; d < head_dim; d++) {
        float s = 0.0f;
        for (int i = 0; i < seq_len; i++) {
          s += dscore.data[i * seq_len + j] * Q->data[i * d_model + off + d];
        }
        K->grad[j * d_model + off + d] += scale * s;
      }
    }
  }

  // Projections: Q/K/V = input @ W_{q,k,v}
  matmul_backward(input, W_q, Q);
  matmul_backward(input, W_k, K);
  matmul_backward(input, W_v, V);

  tensor_free(&dw);
  tensor_free(&dscore);
}

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
) {
  int shape[2] = {input->shape[0], input->shape[1]};
  Tensor norm1_out = tensor_create(shape, 2);
  Tensor attn_out = tensor_create(shape, 2);
  Tensor attn_residual = tensor_create(shape, 2);
  Tensor norm2_out = tensor_create(shape, 2);
  Tensor ff_out = tensor_create(shape, 2);

  // Sub-layer 1: attn_residual = input + MultiHeadAttention(LN1(input))
  layernorm(input, gamma1, beta1, &norm1_out, 1e-5f);
  multihead_attention_forward(&norm1_out, W_q, W_k, W_v, W_o, &attn_out, n_heads);
  matadd(input, &attn_out, &attn_residual);

  // Sub-layer 2: output = attn_residual + FFN(LN2(attn_residual))
  layernorm(&attn_residual, gamma2, beta2, &norm2_out, 1e-5f);
  feedforward(&norm2_out, ff_W1, ff_b1, ff_W2, ff_b2, &ff_out, d_ff);
  matadd(&attn_residual, &ff_out, output);

  tensor_free(&norm1_out);
  tensor_free(&attn_out);
  tensor_free(&attn_residual);
  tensor_free(&norm2_out);
  tensor_free(&ff_out);
}
