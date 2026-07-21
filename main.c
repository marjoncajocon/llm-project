// main.c - Minimal GPT-2 Style Transformer in Pure C11 (demo entry point)
// Build: make        (see README.md and Makefile)
// Or:    gcc -std=c11 -O3 -march=native -o transformer *.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tensor.h"
#include "model.h"

// ============================================================================
// Main Function: build a tiny GPT and greedily generate tokens
// ============================================================================

int main() {
  printf("=== Minimal GPT-2 Style Transformer Demo ===\n\n");

  // Model configuration (tiny, randomly initialized - untrained)
  const int vocab_size = 16;      // Vocabulary size
  const int d_model = 8;          // Model dimension
  const int n_heads = 2;          // Attention heads (head_dim = d_model / n_heads)
  const int n_layers = 2;         // Number of stacked transformer blocks
  const int d_ff = 4 * d_model;   // Feed-forward hidden dimension
  const int max_seq_len = 16;     // Maximum supported sequence length

  // Seed random number generator for reproducibility
  srand(42);

  // Build the model
  GPT model = gpt_create(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len);
  printf("Model: vocab=%d d_model=%d heads=%d layers=%d d_ff=%d\n\n",
         vocab_size, d_model, n_heads, n_layers, d_ff);

  // Prompt: a short sequence of token IDs, plus room to append generated tokens
  const int prompt_len = 3;
  const int n_new = 5;
  int ids[16];  // capacity == max_seq_len
  ids[0] = 1;
  ids[1] = 5;
  ids[2] = 9;

  printf("Prompt tokens:  ");
  for (int i = 0; i < prompt_len; i++) printf("%d ", ids[i]);
  printf("\n");

  // Autoregressive greedy generation
  printf("Generating %d tokens (greedy argmax)...\n", n_new);
  gpt_generate(&model, ids, prompt_len, n_new);

  printf("Full sequence:  ");
  for (int i = 0; i < prompt_len + n_new; i++) printf("%d ", ids[i]);
  printf("\n\n");

  // Validate logits from a full forward pass (no NaN or Inf)
  int total_len = prompt_len + n_new;
  int logits_shape[2] = {total_len, vocab_size};
  Tensor logits = tensor_create(logits_shape, 2);
  gpt_forward(&model, ids, total_len, &logits);

  int valid = 1;
  for (int i = 0; i < logits.size; i++) {
    if (isnan(logits.data[i]) || isinf(logits.data[i])) {
      valid = 0;
      break;
    }
  }
  printf("Output validation: %s\n\n",
         valid ? "PASSED (no NaN/Inf)" : "FAILED (contains NaN/Inf)");

  // Clean up
  printf("Cleaning up memory...\n");
  tensor_free(&logits);
  gpt_free(&model);
  printf("Done! All memory freed successfully.\n");

  return 0;
}
