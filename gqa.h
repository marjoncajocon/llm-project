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
typedef struct {
  int n, n_heads, n_kv, head_dim;
  float *att;   // (n_heads, n, n) row-softmax weights (only j<=i used)
} GqaCache;

GqaCache gqa_cache_create(int n, int n_heads, int n_kv, int head_dim);
void gqa_cache_free(GqaCache *c);

void gqa_forward(const Tensor *Q, const Tensor *K, const Tensor *V, GqaCache *c, Tensor *O);

// Backward: reads O->grad and the cache; accumulates Q->grad, K->grad, V->grad
// (grads of query heads sharing a KV head accumulate into that head's K/V).
void gqa_backward(const Tensor *Q, const Tensor *K, const Tensor *V,
                  const GqaCache *c, const Tensor *O);

#endif // GQA_H
