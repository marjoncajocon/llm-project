// sampler.c - Token sampling from logits
#include "sampler.h"

// Greedy sampler: return the index of the largest logit in the given row.
int sample_argmax(const Tensor *logits, int row) {
  const int vocab_size = logits->shape[1];
  const float *r = &logits->data[row * vocab_size];

  int best = 0;
  float best_val = r[0];
  for (int j = 1; j < vocab_size; j++) {
    if (r[j] > best_val) {
      best_val = r[j];
      best = j;
    }
  }
  return best;
}
