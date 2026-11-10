// deltanet.h - Gated DeltaNet: linear attention with a fixed-size state (Qwen3.5 /
// Qwen3-Next hybrid layers; Yang, Kautz & Hatamizadeh, "Gated Delta Networks", 2024).
//
// Softmax attention keeps every past key and value (the KV cache grows with the
// context, and each token attends over all of it). A DeltaNet head instead keeps
// ONE d x d matrix S that maps keys to values, and edits it with the delta rule:
//
//   S_t = alpha_t * (S_{t-1} - beta_t * k_t (k_t^T S_{t-1})) + beta_t * k_t v_t^T
//   o_t = S_t^T q_t
//
// i.e. read what S currently stores under key k_t, move it a fraction beta_t of the
// way toward v_t (the delta rule -- an online least-squares step), and decay the
// whole memory by alpha_t (the gate: forget stale associations). q and k are
// L2-normalized, so beta_t = 1 overwrites the slot exactly.
//
//   q, k, v  = x @ Wq, x @ Wk, x @ Wv        per head: head_dim each
//   alpha_t  = sigmoid(x @ Wa + ba)          per head (the paper uses exp(-A softplus);
//   beta_t   = sigmoid(x @ Wb + bb)           sigmoid is the simplest stable gate)
//   out      = O @ Wo
//
// Cost per token is O(n_heads * head_dim^2) whatever the position: decode needs
// no cache that grows, which is why hybrids keep only a few full-attention layers.
// This module is the layer on its own (forward, backward, O(1)-state decode);
// wiring it into LlamaModel as a hybrid block is a separate step.
#ifndef DELTANET_H
#define DELTANET_H

#include "tensor.h"

typedef struct {
  int d_model, n_heads, head_dim;
  Tensor Wq, Wk, Wv;   // (d_model, n_heads*head_dim)
  Tensor Wa, Wb;       // (d_model, n_heads)  gate logits
  Tensor ba, bb;       // (n_heads)           gate biases
  Tensor Wo;           // (n_heads*head_dim, d_model)
} DeltaNet;

DeltaNet deltanet_create(int d_model, int n_heads, int head_dim);
void deltanet_free(DeltaNet *dn);
// Small random weights; ba = +3 (alpha ~ 0.95: remember by default), bb = 0.
void deltanet_randomize(DeltaNet *dn);
void deltanet_zero_grad(DeltaNet *dn);
// The 8 learnable tensors, for an optimizer / gradient check. Returns 8.
int deltanet_collect_params(DeltaNet *dn, Tensor **list);

// Everything the backward needs from one forward of up to max_len positions.
typedef struct {
  int cap, n;
  Tensor q, k, v;      // (cap, H*hd) projections (with grads); q/k RAW (pre-norm)
  Tensor ga, gb;       // (cap, H) gate logits (with grads)
  Tensor o;            // (cap, H*hd) S_t^T q_t (with grad)
  float *qn, *kn;      // (cap, H*hd) normalized q, k
  float *qnorm, *knorm;// (cap, H)    their L2 norms
  float *alpha, *beta; // (cap, H)
  float *S;            // (cap + 1) * H * hd * hd: S_0 = 0, then every S_t
} DeltaNetCache;

DeltaNetCache deltanet_cache_create(const DeltaNet *dn, int max_len);
void deltanet_cache_free(DeltaNetCache *c);

// Forward over x (n, d_model) -> out (n, d_model), from a zero state.
void deltanet_forward(const DeltaNet *dn, const Tensor *x, Tensor *out, DeltaNetCache *c);

// Backward: reads out->grad, ACCUMULATES x->grad and every parameter grad.
void deltanet_backward(DeltaNet *dn, const Tensor *x, const Tensor *out, DeltaNetCache *c);

// O(1)-state decode: state is n_heads * head_dim * head_dim floats (zero to
// start a sequence). One token x (1, d_model) -> out (1, d_model); updates state.
// Bit-identical to the row of deltanet_forward at the same position.
void deltanet_step(const DeltaNet *dn, float *state, const Tensor *x, Tensor *out);

#endif // DELTANET_H
