// gqa.h - Grouped-Query Attention (causal, LLaMA-family)
#ifndef GQA_H
#define GQA_H

#include "tensor.h"

// Grouped-query attention: n_heads QUERY heads but only n_kv KEY/VALUE heads
// (n_kv divides n_heads). Query head h shares KV head (h / (n_heads/n_kv)), so a
// group of n_heads/n_kv query heads reuse one K/V head — this is what shrinks the
// KV cache in modern models. n_kv == n_heads is ordinary multi-head attention;
// n_kv == 1 is multi-query attention.
//
// Inputs are already-projected, per-head-concatenated:
//   Q: (n, n_heads * head_dim)      K, V: (n, n_kv * head_dim)      O: (n, n_heads * head_dim)
// Causal: position i attends only to j <= i. Scale = 1/sqrt(head_dim).
// The forward caches the softmax weights so the backward needs no recompute.
//
// Sliding window (optional): with window W > 0, position i attends only to the
// W most recent positions j in [max(0, i-W+1), i] — itself included, so W = 1
// is "attend to self only". Keys outside the window are masked exactly like
// future keys (weight 0, no gradient). W <= 0 is ordinary full causal attention
// and runs the IDENTICAL arithmetic as before the window existed (the loop just
// starts at j = 0), so it is bit-for-bit unchanged. W >= n is also full causal
// attention, bit-exactly (the window's left edge never rises above 0).
// gqa_cache_create sets window = 0; the caller sets c.window before forward.
typedef struct {
  int n, n_heads, n_kv, head_dim;
  int window;   // sliding-window size W (<= 0 = full causal attention)
  float *att;   // (n_heads, n, n) row-softmax weights (only lo(i)<=j<=i used)
} GqaCache;

// First key position query i may attend to under window W (0 when W <= 0).
// Shared by the batched forward/backward here and the KV-cache decode in llama.c
// so the two can never disagree about the window's edge.
static inline int gqa_window_lo(int i, int window) {
  return (window > 0 && i - window + 1 > 0) ? i - window + 1 : 0;
}

GqaCache gqa_cache_create(int n, int n_heads, int n_kv, int head_dim);
void gqa_cache_free(GqaCache *c);

void gqa_forward(const Tensor *Q, const Tensor *K, const Tensor *V, GqaCache *c, Tensor *O);

// Backward: reads O->grad and the cache; accumulates Q->grad, K->grad, V->grad
// (grads of query heads sharing a KV head accumulate into that head's K/V).
void gqa_backward(const Tensor *Q, const Tensor *K, const Tensor *V,
                  const GqaCache *c, const Tensor *O);

#endif // GQA_H
