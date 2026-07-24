// llamatrain.h - Training (forward-with-cache + backward) for the modern
// LLaMA-family model in llama.h. This is the counterpart of the GPT training
// path in model.h (gpt_forward_train / gpt_backward), for the architecture that
// real small modern models actually use: RMSNorm + RoPE + GQA + SwiGLU, no biases
// on the output projections and no learned positional embedding.
//
// The per-op backward primitives already exist and are gradient-checked
// individually (rmsnorm.h, rope.h, gqa.h, swiglu.h); this module stitches them
// into a whole-model backward, which selftest.c then gradient-checks end to end.
//
// Usage mirrors the GPT path exactly:
//   LlamaModel m = llama_model_create_train(...);   // weights WITH grad arrays
//   LlamaActs a = llama_acts_create(&m, seq_len);
//   llama_forward_train(&m, ids, seq_len, &a, &logits);
//   float loss = cross_entropy_forward(&logits, targets, &probs);
//   tensor_zero_grad(&logits);
//   cross_entropy_backward(&probs, targets, &logits);
//   llama_backward(&m, ids, seq_len, &a, &logits);  // grads into every parameter
//   adam_step(&opt, params, n);
#ifndef LLAMATRAIN_H
#define LLAMATRAIN_H

#include "tensor.h"
#include "llama.h"
#include "gqa.h"
#include "swiglu.h"

// Activations cached by the training forward, consumed by llama_backward.
// One set of per-layer tensors, plus the two model-level ones. Every tensor here
// carries a grad array (they are all intermediate values backprop flows through).
typedef struct {
  int seq_len, n_layers;
  Tensor emb;         // (n, d_model) token embeddings (no positional term: RoPE)
  Tensor final_hn;    // (n, d_model) final RMSNorm output (input to lm_head)
  // Per-layer arrays (length n_layers):
  Tensor *xn;         // (n, d_model)          attention RMSNorm output
  Tensor *q;          // (n, n_heads*head_dim) POST-RoPE queries
  Tensor *k;          // (n, n_kv*head_dim)    POST-RoPE keys
  Tensor *v;          // (n, n_kv*head_dim)
  Tensor *o;          // (n, n_heads*head_dim) attention output (pre-Wo)
  Tensor *ao;         // (n, d_model)          o @ Wo
  Tensor *h;          // (n, d_model)          x + ao   (attention residual)
  Tensor *hn;         // (n, d_model)          FFN RMSNorm output
  Tensor *f;          // (n, d_model)          SwiGLU output
  Tensor *block_out;  // (n, d_model)          h + f    (input to the next layer)
  GqaCache *gc;       // per layer: cached softmax weights
  SwigluCache *sc;    // per layer: cached g, u, a, h
} LlamaActs;

LlamaActs llama_acts_create(const LlamaModel *m, int seq_len);
void llama_acts_free(LlamaActs *a);

// Training-mode forward: fills the activation cache and logits (seq_len, vocab).
// Causal, and identical in value to llama_model_forward.
void llama_forward_train(LlamaModel *m, const int *ids, int seq_len,
                         LlamaActs *a, Tensor *logits);

// Backward. Expects logits->grad to already hold dL/dlogits (from
// cross_entropy_backward). Zeros parameter and activation grads, then accumulates
// gradients into every model parameter (including the embedding rows used).
void llama_backward(LlamaModel *m, const int *ids, int seq_len,
                    LlamaActs *a, Tensor *logits);

// Number of learnable parameter tensors (== 3 + 12*n_layers). Use this to size
// the buffer passed to llama_collect_params.
int llama_param_count(const LlamaModel *m);

// Fill list[] with pointers to all learnable parameter tensors, in canonical
// order. Returns the count (== llama_param_count(m)).
int llama_collect_params(LlamaModel *m, Tensor **list);

// Zero the gradient of every parameter.
void llama_zero_grad(LlamaModel *m);

#endif // LLAMATRAIN_H
