// activation.c - Activation functions
#include "activation.h"

#include <math.h>

// Row-wise Softmax with numerical stability
// Applies softmax to each row of input matrix:
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
void softmax_rowwise(const Tensor *input, Tensor *output) {
  const int rows = input->shape[0];
  const int cols = input->shape[1];

  for (int i = 0; i < rows; i++) {
    // Step 1: Find maximum value in row for numerical stability
    float max_val = input->data[i * cols];
    for (int j = 1; j < cols; j++) {
      if (input->data[i * cols + j] > max_val) {
        max_val = input->data[i * cols + j];
      }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (int j = 0; j < cols; j++) {
      output->data[i * cols + j] = expf(input->data[i * cols + j] - max_val);
      sum_exp += output->data[i * cols + j];
    }

    // Step 3: Normalize by sum
    for (int j = 0; j < cols; j++) {
      output->data[i * cols + j] /= sum_exp;
    }
  }
}

// GELU activation (tanh approximation, as used by GPT-2), element-wise
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu(const Tensor *input, Tensor *output) {
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  for (int i = 0; i < input->size; i++) {
    float x = input->data[i];
    float inner = k * (x + 0.044715f * x * x * x);
    output->data[i] = 0.5f * x * (1.0f + tanhf(inner));
  }
}
