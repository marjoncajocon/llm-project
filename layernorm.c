// layernorm.c - Layer Normalization
#include "layernorm.h"

#include <math.h>

// LayerNorm: normalizes activations across feature dimension
// y = (x - mean) / sqrt(var + epsilon) * gamma + beta
void layernorm(const Tensor *input, const Tensor *gamma, const Tensor *beta,
               Tensor *output, float epsilon) {
  const int rows = input->shape[0];
  const int cols = input->shape[1];

  for (int i = 0; i < rows; i++) {
    // Calculate mean
    float mean = 0.0f;
    for (int j = 0; j < cols; j++) {
      mean += input->data[i * cols + j];
    }
    mean /= cols;

    // Calculate variance
    float variance = 0.0f;
    for (int j = 0; j < cols; j++) {
      float diff = input->data[i * cols + j] - mean;
      variance += diff * diff;
    }
    variance /= cols;

    // Normalize: (x - mean) / sqrt(variance + epsilon)
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    for (int j = 0; j < cols; j++) {
      output->data[i * cols + j] = (input->data[i * cols + j] - mean) * inv_std;
    }

    // Scale and shift with learnable parameters
    for (int j = 0; j < cols; j++) {
      output->data[i * cols + j] = output->data[i * cols + j] * gamma->data[j] + beta->data[j];
    }
  }
}
