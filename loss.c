// loss.c - Softmax cross-entropy loss
#include "loss.h"

#include <math.h>

#include "activation.h"

// Softmax cross-entropy over rows. Returns the mean loss over N rows.
float cross_entropy_forward(const Tensor *logits, const int *targets, Tensor *probs) {
  const int N = logits->shape[0];
  const int V = logits->shape[1];

  // probs = row-wise softmax of logits (numerically stable, from activation.c)
  softmax_rowwise(logits, probs);

  // loss = -1/N * sum_i log(probs[i, targets[i]])
  float loss = 0.0f;
  for (int i = 0; i < N; i++) {
    float p = probs->data[i * V + targets[i]];
    // Guard the log against exactly-zero probability
    loss -= logf(p > 1e-12f ? p : 1e-12f);
  }
  return loss / (float)N;
}

// Backward: dlogits[i,j] = (probs[i,j] - [j == targets[i]]) / N
void cross_entropy_backward(const Tensor *probs, const int *targets, Tensor *logits_grad) {
  const int N = probs->shape[0];
  const int V = probs->shape[1];
  const float inv_n = 1.0f / (float)N;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < V; j++) {
      float g = probs->data[i * V + j];
      if (j == targets[i]) g -= 1.0f;
      logits_grad->grad[i * V + j] += g * inv_n;
    }
  }
}
