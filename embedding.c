// embedding.c - Token + positional embedding lookup
#include "embedding.h"

// Embed a sequence of token IDs: output[i] = wte[token_ids[i]] + wpe[i]
void embed_tokens(const int *token_ids, int seq_len,
                  const Tensor *wte, const Tensor *wpe, Tensor *output) {
  const int d_model = wte->shape[1];

  for (int i = 0; i < seq_len; i++) {
    const int tok = token_ids[i];
    for (int j = 0; j < d_model; j++) {
      output->data[i * d_model + j] =
          wte->data[tok * d_model + j] + wpe->data[i * d_model + j];
    }
  }
}

// Backward (scatter-add gradients into the embedding tables).
void embedding_backward(const int *token_ids, int seq_len,
                        const Tensor *output, const Tensor *wte, const Tensor *wpe) {
  const int d_model = wte->shape[1];

  for (int i = 0; i < seq_len; i++) {
    const int tok = token_ids[i];
    for (int j = 0; j < d_model; j++) {
      float g = output->grad[i * d_model + j];
      wte->grad[tok * d_model + j] += g;
      wpe->grad[i * d_model + j] += g;
    }
  }
}
