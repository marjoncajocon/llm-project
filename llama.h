// llama.h - A modern (LLaMA-family) transformer block, assembled from the
// gradient-checked primitives: RMSNorm + RoPE + GQA + SwiGLU. No biases.
//
//   xn  = RMSNorm(x, attn_norm)
//   q,k,v = xn @ {Wq, Wk, Wv}          (RoPE applied per-head to q and k)
//   h   = x + GQA(q, k, v) @ Wo        (residual)
//   hn  = RMSNorm(h, ffn_norm)
//   out = h + SwiGLU(hn, Wgate, Wup, Wdown)   (residual)
//
// Forward only (inference) — the target use is loading real small modern models.
#ifndef LLAMA_H
#define LLAMA_H

#include "tensor.h"

typedef struct {
  int d_model, n_heads, n_kv, head_dim, d_ff;
  float rope_base, eps;
  // Attention
  Tensor attn_norm;   // (d_model)
  Tensor Wq;          // (d_model, n_heads*head_dim)
  Tensor Wk;          // (d_model, n_kv*head_dim)
  Tensor Wv;          // (d_model, n_kv*head_dim)
  Tensor Wo;          // (n_heads*head_dim, d_model)
  // Feed-forward (SwiGLU)
  Tensor ffn_norm;    // (d_model)
  Tensor Wgate;       // (d_model, d_ff)
  Tensor Wup;         // (d_model, d_ff)
  Tensor Wdown;       // (d_ff, d_model)
} LlamaBlock;

// Allocate a block with the given config (weights are zeroed — fill them via the
// returned tensors, or use llama_block_randomize for testing).
LlamaBlock llama_block_create(int d_model, int n_heads, int n_kv, int head_dim,
                              int d_ff, float rope_base, float eps);
void llama_block_free(LlamaBlock *b);

// Fill all weights with small deterministic pseudo-random values (for tests).
void llama_block_randomize(LlamaBlock *b);

// Forward: x (seq_len, d_model) -> out (seq_len, d_model). Causal.
void llama_block_forward(const LlamaBlock *b, const Tensor *x, Tensor *out);

#endif // LLAMA_H
