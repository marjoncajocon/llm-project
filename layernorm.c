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

// Backward of LayerNorm. Recomputes mean/variance from input->data per row.
void layernorm_backward(const Tensor *input, const Tensor *gamma, const Tensor *output,
                        Tensor *gamma_grad, Tensor *beta_grad, float epsilon) {
  const int rows = input->shape[0];
  const int cols = input->shape[1];
  const float inv_n = 1.0f / (float)cols;

  for (int i = 0; i < rows; i++) {
    const float *x = &input->data[i * cols];
    const float *dy = &output->grad[i * cols];

    // Recompute mean, variance, inv_std for this row
    float mean = 0.0f;
    for (int j = 0; j < cols; j++) mean += x[j];
    mean *= inv_n;

    float variance = 0.0f;
    for (int j = 0; j < cols; j++) {
      float d = x[j] - mean;
      variance += d * d;
    }
    variance *= inv_n;
    float inv_std = 1.0f / sqrtf(variance + epsilon);

    // First pass: parameter grads, and the two reduction terms for dx
    float mean_dxhat = 0.0f;       // mean over j of (dy * gamma)
    float mean_dxhat_xhat = 0.0f;  // mean over j of (dy * gamma) * xhat
    for (int j = 0; j < cols; j++) {
      float xhat = (x[j] - mean) * inv_std;
      gamma_grad->grad[j] += dy[j] * xhat;
      beta_grad->grad[j] += dy[j];
      float dxhat = dy[j] * gamma->data[j];
      mean_dxhat += dxhat;
      mean_dxhat_xhat += dxhat * xhat;
    }
    mean_dxhat *= inv_n;
    mean_dxhat_xhat *= inv_n;

    // Second pass: dx = inv_std * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    for (int j = 0; j < cols; j++) {
      float xhat = (x[j] - mean) * inv_std;
      float dxhat = dy[j] * gamma->data[j];
      input->grad[i * cols + j] +=
          inv_std * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
    }
  }
}
