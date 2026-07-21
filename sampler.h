// sampler.h - Token sampling from logits
#ifndef SAMPLER_H
#define SAMPLER_H

#include "tensor.h"

// Greedy sampler: return the index of the largest logit in the given row.
//   logits : (seq_len, vocab_size)
//   row    : which position's logits to sample from (0-based)
int sample_argmax(const Tensor *logits, int row);

#endif // SAMPLER_H
