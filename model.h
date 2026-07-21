// model.h - GPT model: embeddings, stacked transformer blocks, output head
#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

// Learnable weights for a single transformer block.
typedef struct {
  Tensor W_q, W_k, W_v, W_o;      // attention projections (d_model, d_model)
  Tensor gamma1, beta1;           // LayerNorm 1 params (d_model)
  Tensor gamma2, beta2;           // LayerNorm 2 params (d_model)
  Tensor ff_W1, ff_b1;            // feed-forward layer 1 (d_model, d_ff), (d_ff)
  Tensor ff_W2, ff_b2;            // feed-forward layer 2 (d_ff, d_model), (d_model)
} BlockWeights;

// A minimal GPT-2 style language model.
typedef struct {
  int vocab_size;
  int d_model;
  int n_heads;
  int n_layers;
  int d_ff;
  int max_seq_len;

  Tensor wte;                     // token embeddings (vocab_size, d_model)
  Tensor wpe;                     // positional embeddings (max_seq_len, d_model)
  BlockWeights *blocks;           // n_layers transformer blocks
  Tensor ln_f_gamma, ln_f_beta;   // final LayerNorm params (d_model)
  Tensor lm_head;                 // output projection (d_model, vocab_size)
} GPT;

// Create a model and randomly initialize all weights.
GPT gpt_create(int vocab_size, int d_model, int n_heads,
               int n_layers, int d_ff, int max_seq_len);

// Free all weights owned by the model.
void gpt_free(GPT *m);

// Forward pass over a token sequence.
//   token_ids : seq_len token indices
//   logits    : (seq_len, vocab_size) output, caller-allocated
void gpt_forward(const GPT *m, const int *token_ids, int seq_len, Tensor *logits);

// Greedily generate n_new tokens, appending them in-place to token_ids.
// The token_ids buffer must hold at least (prompt_len + n_new) ints, and
// (prompt_len + n_new) must not exceed max_seq_len.
void gpt_generate(const GPT *m, int *token_ids, int prompt_len, int n_new);

#endif // MODEL_H
