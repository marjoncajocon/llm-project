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
#include "lora.h"

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

// Re-target the activation buffers at a SHORTER sequence length (<= the length
// passed to llama_acts_create). Only the logical row count shrinks — no realloc —
// so an SFT loop can run each example at its TRUE length instead of padding every
// one to the global max. Bit-exact vs padding (causal attention never reads padded
// rows; padded targets are -1 = zero loss/grad), just far less wasted compute.
void llama_acts_set_len(LlamaActs *a, int seq_len);

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

// ---------------------------------------------------------------------------
// LoRA fine-tuning: adapters on the four attention projections of every layer
// (Wq, Wk, Wv, Wo — the conventional choice), with the base model FROZEN.
//
//   LlamaLora lo = llama_lora_create(&m, 8, 16.0f);  llama_lora_init(&lo);
//   LlamaLoraActs la = llama_lora_acts_create(&m, &lo, seq_len);
//   llama_forward_train_lora(&m, ids, n, &acts, &logits, &lo, &la);
//   ... cross_entropy_backward ...
//   llama_backward_lora(&m, ids, n, &acts, &logits, &lo, &la);
//   adam_step(&opt, lora_params, n_lora);   // ONLY the adapters are optimized
//
// Freezing is achieved by handing Adam only the adapter tensors. The backward
// still fills the base gradients (matmul_backward computes dx and dW together,
// so dx cannot be had without dW), but nothing consumes them — so the saving is
// the Adam moments, which dominate optimizer memory, not the gradient buffers.
// ---------------------------------------------------------------------------
typedef struct {
  int n_layers, r;
  LoraAdapter *q, *k, *v, *o;   // n_layers each
} LlamaLora;

// r = rank, alpha = LoRA scaling (scale = alpha / r).
LlamaLora llama_lora_create(const LlamaModel *m, int r, float alpha);
void llama_lora_free(LlamaLora *lo);

// A <- small random, B <- 0, so the adapted model starts bit-identical to the base.
void llama_lora_init(LlamaLora *lo);

// Per-layer adapter activation caches for a given sequence length.
typedef struct {
  int n_layers;
  LoraCache *q, *k, *v, *o;
} LlamaLoraActs;

LlamaLoraActs llama_lora_acts_create(const LlamaModel *m, const LlamaLora *lo, int seq_len);
void llama_lora_acts_free(LlamaLoraActs *la);

// Re-target the LoRA activation caches at a shorter sequence length (see
// llama_acts_set_len). Call alongside it before a shorter forward.
void llama_lora_acts_set_len(LlamaLoraActs *la, int seq_len);

// Trainable adapter tensors (== 8 * n_layers: an A and a B per projection).
int llama_lora_param_count(const LlamaLora *lo);
int llama_lora_collect_params(LlamaLora *lo, Tensor **list);

// Fold the learned corrections into the base weights (W += scale * A @ B) and
// reset the adapters to zero, so the result is an ORDINARY LlamaModel that
// llama_model_save writes as a normal LLMA checkpoint and `transformer llama`
// runs with no adapter support and no inference-time cost.
void llama_lora_merge(LlamaModel *m, LlamaLora *lo);

// Forward / backward with adapters. Passing lo == NULL is exactly the plain path.
void llama_forward_train_lora(LlamaModel *m, const int *ids, int seq_len,
                              LlamaActs *a, Tensor *logits,
                              const LlamaLora *lo, LlamaLoraActs *la);
void llama_backward_lora(LlamaModel *m, const int *ids, int seq_len,
                         LlamaActs *a, Tensor *logits,
                         const LlamaLora *lo, LlamaLoraActs *la);

// Same, but does NOT zero the parameter gradients first, so B micro-batches
// accumulate into one effective batch of B. Zero once up front (llama_zero_grad,
// plus llama_lora_zero_grad when adapters are in use), run B of these, then
// scale_grads(params, n, 1/B) before clipping and stepping.
void llama_backward_accum(LlamaModel *m, const int *ids, int seq_len,
                          LlamaActs *a, Tensor *logits,
                          const LlamaLora *lo, LlamaLoraActs *la);

// Zero every adapter gradient (the LoRA counterpart of llama_zero_grad).
void llama_lora_zero_grad(const LlamaLora *lo);

#endif // LLAMATRAIN_H
