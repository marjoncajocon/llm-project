// moe.h - Mixture-of-Experts feed-forward layer (DeepSeek / Qwen3-MoE style)
#ifndef MOE_H
#define MOE_H

#include "tensor.h"
#include "swiglu.h"

// A drop-in replacement for one SwiGLU FFN: n_experts independent SwiGLU FFNs,
// a learned router that sends each token to only top_k of them, and an optional
// always-on shared expert. Parameters grow with n_experts; per-token compute
// grows only with top_k. That decoupling is the whole point of MoE.
//
//   logits = x @ W_router                       (n, E)
//   p      = softmax(logits)                    (n, E)   router probabilities
//   S_t    = the top_k experts of token t by p  (ties -> lower index)
//   g_te   = p_te / sum_{j in S_t} p_tj         (renorm=1)   or   p_te (renorm=0)
//   y_t    = sum_{e in S_t} g_te * SwiGLU_e(x_t)  [+ SwiGLU_shared(x_t)]
//
// Gate choice. renorm=1 (the default) makes each token's k gates sum to 1. That
// is exactly "softmax over the chosen k logits" (Mixtral), and Qwen3-MoE's
// norm_topk_prob=True; DeepSeek-V3 also normalizes its (sigmoid) scores over the
// chosen k. It keeps the output scale independent of k and of how peaked the
// router is. Its limit: with top_k == 1 the single gate is always exactly 1, so
// the TASK loss sends the router no gradient at all -- only the aux loss trains
// it. That is why Switch Transformer (k=1) uses the raw probability (renorm=0).
//
// Load balancing (Switch Transformer, eq. 4, generalized to top-k):
//   f_e   = (tokens routed to e) / (n * top_k)     -- fraction of assignments
//   P_e   = mean_t p_te                            -- mean router probability
//   aux   = E * sum_e f_e * P_e
// aux == 1 whenever routing is perfectly balanced (or p is uniform), and grows
// toward E as the router collapses onto one expert. f_e is a count, so it is
// piecewise constant and carries no gradient; the gradient flows through P_e.
// The caller adds aux_coef * aux to its loss (Switch uses ~1e-2).
typedef struct {
  int d_model, d_ff, n_experts, top_k;
  int renorm;          // 1: renormalize the chosen k gates to sum 1 (see above)
  int shared;          // 1: add one always-on shared expert (DeepSeek-style)
  Tensor W_router;     // (d_model, n_experts)
  Tensor *W_gate;      // n_experts x (d_model, d_ff)
  Tensor *W_up;        // n_experts x (d_model, d_ff)
  Tensor *W_down;      // n_experts x (d_ff, d_model)
  Tensor Ws_gate, Ws_up, Ws_down;   // shared expert; unallocated when shared == 0
} MoeLayer;

// Allocate with gradient arrays (trainable). Weights are Xavier-random.
MoeLayer moe_create(int d_model, int d_ff, int n_experts, int top_k,
                    int renorm, int shared);
void moe_free(MoeLayer *m);

// Collect pointers to every parameter tensor (router, then expert 0's gate/up/
// down, expert 1's, ..., then the shared expert's). `out` must hold
// moe_n_params(m) entries. Returns the count. For adam_create / clip_grad_norm.
int moe_n_params(const MoeLayer *m);
int moe_params(MoeLayer *m, Tensor **out);

// Per-forward state for the backward pass. Sized for a fixed token count n.
// Expert buffers are sized by how many tokens each expert actually received, so
// they are (re)allocated inside every forward -- an expert that got no tokens
// has no buffer and is never run.
typedef struct {
  int n, n_experts, top_k;
  Tensor logits, probs;  // (n, E)
  int *sel;              // (n, top_k) chosen expert ids, best first
  float *gate;           // (n, top_k) gate weight of each chosen expert
  float *dgate;          // (n, top_k) backward scratch: dL/dgate
  int *count;            // (E) tokens routed to each expert
  int *offset;           // (E+1) expert e's assignments are order[offset[e]..offset[e+1])
  int *order;            // (n*top_k) assignment ids a = t*top_k + j, grouped by expert
  Tensor *xe, *ye;       // per expert: gathered inputs / outputs (count[e], d_model)
  SwigluCache *ce;       // per expert SwiGLU cache (count[e] rows)
  int *rows_alloc;       // rows currently allocated per expert (0 = none)
  Tensor ys;             // (n, d_model) shared-expert output (if shared)
  SwigluCache cs;        // shared-expert cache
  float *f, *P;          // (E) aux-loss terms from the last forward
  float aux;             // aux loss of the last forward
  // Work actually done by the last forward, counted as 2 FLOPs per multiply-add
  // of each matmul. Expert work is summed over the experts that RAN, so it is a
  // measurement of the dispatch, not a formula.
  double flops_router, flops_experts, flops_shared;
  long long expert_rows; // total rows pushed through routed experts (== n*top_k)
} MoeCache;

MoeCache moe_cache_create(const MoeLayer *m, int n);
void moe_cache_free(MoeCache *c);

// Forward: x (n, d_model) -> y (n, d_model). Returns the (unscaled) aux loss,
// also stored in c->aux.
float moe_forward(const MoeLayer *m, const Tensor *x, MoeCache *c, Tensor *y);

// Backward: reads y->grad and the cache from the matching forward; ACCUMULATES
// into x->grad and every parameter's grad. aux_coef is the weight the caller
// gave the aux loss (total loss = task + aux_coef * aux); pass 0 to skip it.
void moe_backward(MoeLayer *m, const Tensor *x, MoeCache *c, const Tensor *y,
                  float aux_coef);

#endif // MOE_H
