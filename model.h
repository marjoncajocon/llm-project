// model.h - GPT model: embeddings, stacked transformer blocks, output head
#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

// Learnable weights for a single transformer block.
typedef struct {
  Tensor W_q, W_k, W_v, W_o;      // attention projections (d_model, d_model)
  Tensor b_q, b_k, b_v, b_o;      // attention biases (d_model) — GPT-2 c_attn/c_proj
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

// ---------------------------------------------------------------------------
// Training: cached activations, training forward/backward, parameter helpers
// ---------------------------------------------------------------------------

// Activations cached during a training forward pass, consumed by gpt_backward.
typedef struct {
  int seq_len;
  int n_layers;
  Tensor emb;         // (seq_len, d_model) token+positional embeddings
  Tensor ln_f_out;    // (seq_len, d_model) final LayerNorm output
  // Per-layer arrays (length n_layers):
  Tensor *ln1;        // (seq_len, d_model)
  Tensor *Q, *K, *V;  // (seq_len, d_model)
  Tensor *weights;    // (n_heads*seq_len, seq_len) attention weights
  Tensor *concat;     // (seq_len, d_model) concatenated head outputs
  Tensor *attn_out;   // (seq_len, d_model)
  Tensor *a_resid;    // (seq_len, d_model) input + attention (residual)
  Tensor *ln2;        // (seq_len, d_model)
  Tensor *ff_h1b;     // (seq_len, d_ff) FFN pre-activation
  Tensor *ff_act;     // (seq_len, d_ff) FFN post-GELU
  Tensor *ff_out;     // (seq_len, d_model)
  Tensor *block_out;  // (seq_len, d_model) block output (input to next layer)
} GPTActs;

GPTActs gpt_acts_create(const GPT *m, int seq_len);
void gpt_acts_free(GPTActs *a);

// Training-mode forward: fills the activation cache and logits (seq_len, vocab_size).
void gpt_forward_train(GPT *m, const int *token_ids, int seq_len,
                       GPTActs *a, Tensor *logits);

// Backward pass. Expects logits->grad to already hold dL/dlogits (from
// cross_entropy_backward). Zeros parameter and activation grads, then accumulates
// gradients into every model parameter.
void gpt_backward(GPT *m, const int *token_ids, int seq_len,
                  GPTActs *a, Tensor *logits);

// Fill list[] with pointers to all learnable parameter tensors, in canonical
// order. Returns the count (== 2 + 16*n_layers + 3). list must have room for that.
int gpt_collect_params(GPT *m, Tensor **list);

// Zero the gradient of every parameter.
void gpt_zero_grad(GPT *m);

// ---------------------------------------------------------------------------
// Incremental decoding with a key/value cache (fast autoregressive generation)
// ---------------------------------------------------------------------------

// Per-layer cache of the K and V projections for every generated position.
typedef struct {
  int n_layers, d_model, n_heads, max_seq_len;
  int len;       // number of positions currently cached
  Tensor *K;     // per layer: (max_seq_len, d_model)
  Tensor *V;     // per layer: (max_seq_len, d_model)
} KVCache;

KVCache gpt_kvcache_create(const GPT *m);
void gpt_kvcache_free(KVCache *kv);

// Decode one token at the next cached position: appends its K/V to the cache and
// writes the next-token logits (shape (1, vocab_size)). Advances kv->len.
void gpt_decode_step(const GPT *m, KVCache *kv, int token, Tensor *logits);

#endif // MODEL_H
