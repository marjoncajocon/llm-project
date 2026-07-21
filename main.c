// main.c - Minimal GPT-2 Style Transformer in Pure C11 (demo entry point)
// Build: make        (see README.md and Makefile)
// Or:    gcc -std=c11 -O3 -march=native -o transformer *.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tensor.h"
#include "attention.h"

// ============================================================================
// Main Function: Demo with sample data
// ============================================================================

int main() {
  printf("=== Minimal GPT-2 Style Transformer Demo ===\n\n");

  // Parameters
  const int seq_len = 4;         // Sequence length (number of tokens)
  const int d_model = 8;         // Model dimension
  const int d_k = 4;             // Key/Query dimension
  const int d_ff = 4 * d_model;  // Feed-forward hidden dimension

  // Seed random number generator for reproducibility
  srand(42);

  // Create input tensor (seq_len x d_model)
  int input_shape[2] = {seq_len, d_model};
  Tensor input = tensor_create(input_shape, 2);
  tensor_init_random(&input);
  printf("Input sequence:\n");
  tensor_print(&input, "Input");

  // Create weight matrices for self-attention
  int wq_shape[2] = {d_model, d_k};
  int wk_shape[2] = {d_model, d_k};
  int wv_shape[2] = {d_model, d_k};
  int wo_shape[2] = {d_k, d_model};

  Tensor W_q = tensor_create(wq_shape, 2);
  Tensor W_k = tensor_create(wk_shape, 2);
  Tensor W_v = tensor_create(wv_shape, 2);
  Tensor W_o = tensor_create(wo_shape, 2);

  tensor_init_random(&W_q);
  tensor_init_random(&W_k);
  tensor_init_random(&W_v);
  tensor_init_random(&W_o);

  // Create LayerNorm parameters
  int norm_shape[1] = {d_model};
  Tensor gamma1 = tensor_create(norm_shape, 1);
  Tensor beta1 = tensor_create(norm_shape, 1);
  Tensor gamma2 = tensor_create(norm_shape, 1);
  Tensor beta2 = tensor_create(norm_shape, 1);

  // Initialize gamma to 1, beta to 0
  for (int i = 0; i < d_model; i++) {
    gamma1.data[i] = 1.0f;
    gamma2.data[i] = 1.0f;
    beta1.data[i] = 0.0f;
    beta2.data[i] = 0.0f;
  }

  // Create feed-forward network weights: (d_model -> d_ff -> d_model)
  int ff_w1_shape[2] = {d_model, d_ff};
  int ff_b1_shape[1] = {d_ff};
  int ff_w2_shape[2] = {d_ff, d_model};
  int ff_b2_shape[1] = {d_model};

  Tensor ff_W1 = tensor_create(ff_w1_shape, 2);
  Tensor ff_b1 = tensor_create(ff_b1_shape, 1);
  Tensor ff_W2 = tensor_create(ff_w2_shape, 2);
  Tensor ff_b2 = tensor_create(ff_b2_shape, 1);

  tensor_init_random(&ff_W1);
  tensor_init_random(&ff_b1);
  tensor_init_random(&ff_W2);
  tensor_init_random(&ff_b2);

  // Output tensor
  int output_shape[2] = {seq_len, d_model};
  Tensor output = tensor_create(output_shape, 2);

  // Execute forward pass through transformer block
  printf("Executing forward pass...\n\n");
  transformer_block_forward(
      &input,
      &W_q, &W_k, &W_v, &W_o,
      &gamma1, &beta1,
      &gamma2, &beta2,
      &ff_W1, &ff_b1,
      &ff_W2, &ff_b2,
      &output,
      d_k, d_ff
  );

  // Print results
  printf("Transformer block output:\n");
  tensor_print(&output, "Output");

  // Verify outputs are valid (no NaN or Inf)
  int valid = 1;
  for (int i = 0; i < output.size; i++) {
    if (isnan(output.data[i]) || isinf(output.data[i])) {
      valid = 0;
      break;
    }
  }
  printf("Output validation: %s\n\n", valid ? "PASSED (no NaN/Inf)" : "FAILED (contains NaN/Inf)");

  // Clean up all allocated memory
  printf("Cleaning up memory...\n");
  tensor_free(&input);
  tensor_free(&W_q);
  tensor_free(&W_k);
  tensor_free(&W_v);
  tensor_free(&W_o);
  tensor_free(&gamma1);
  tensor_free(&beta1);
  tensor_free(&gamma2);
  tensor_free(&beta2);
  tensor_free(&ff_W1);
  tensor_free(&ff_b1);
  tensor_free(&ff_W2);
  tensor_free(&ff_b2);
  tensor_free(&output);

  printf("Done! All memory freed successfully.\n");

  return 0;
}
