// embedding.h - Token + positional embedding lookup
#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

// Embed a sequence of token IDs: output[i] = wte[token_ids[i]] + wpe[i]
//   token_ids : seq_len integer token indices (each in [0, vocab_size))
//   wte       : (vocab_size, d_model)  token embedding table
//   wpe       : (max_seq_len, d_model) positional embedding table
//   output    : (seq_len, d_model)     summed embeddings
void embed_tokens(const int *token_ids, int seq_len,
                  const Tensor *wte, const Tensor *wpe, Tensor *output);

// Backward of embed_tokens (scatter-add): reads output->grad and accumulates
//   wte->grad[token_ids[i]] += output->grad[i]
//   wpe->grad[i]            += output->grad[i]
void embedding_backward(const int *token_ids, int seq_len,
                        const Tensor *output, const Tensor *wte, const Tensor *wpe);

#endif // EMBEDDING_H
