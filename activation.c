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
