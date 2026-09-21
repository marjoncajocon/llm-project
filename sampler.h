// sampler.h - Token sampling from logits
#ifndef SAMPLER_H
#define SAMPLER_H

#include "tensor.h"

// Greedy sampler: return the index of the largest logit in the given row.
//   logits : (seq_len, vocab_size)
//   row    : which position's logits to sample from (0-based)
int sample_argmax(const Tensor *logits, int row);

// Small reproducible RNG (xorshift64), so sampling is deterministic given a seed.
typedef struct {
  unsigned long long state;
} Rng;

Rng rng_create(unsigned long long seed);
float rng_uniform(Rng *r);  // uniform float in [0, 1)

// Sample a token from one logits row with temperature + top-k + top-p (nucleus).
//   temperature: scales logits; <= 0 means greedy (argmax).
//   top_k      : keep only the k highest-probability tokens (<= 0 or >= vocab disables).
//   top_p      : keep the smallest set whose cumulative prob >= top_p (<= 0 or >= 1 disables).
//   rng        : RNG state, advanced on each call.
// Probabilities are ordered by (probability desc, token id asc) -- a total
// order, so equal probabilities never depend on the C library's qsort. Only the
// candidates that can matter are sorted (see sampler.c); the token is
// bit-for-bit the one a full sort gives.
int sample_token(const Tensor *logits, int row, float temperature,
                 int top_k, float top_p, Rng *rng);

// sample_token that also reports how many entries it sorted (*sorted; V when it
// sorted everything, 0 for greedy). For tests and benchmarks.
int sample_token_debug(const Tensor *logits, int row, float temperature,
                       int top_k, float top_p, Rng *rng, int *sorted);

#endif // SAMPLER_H
