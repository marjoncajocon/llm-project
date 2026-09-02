// loss.c - Softmax cross-entropy loss
#include "loss.h"

#include <math.h>

#include "activation.h"
#include "check.h"

// Softmax cross-entropy over rows. Rows whose target is < 0 are IGNORED (no loss,
// no gradient) — used to supervise only some positions (e.g. caption tokens in a
// multimodal sequence). Returns the mean loss over the non-ignored rows.
float cross_entropy_forward(const Tensor *logits, const int *targets, Tensor *probs) {
  const int N = logits->shape[0];
  const int V = logits->shape[1];

  // probs = row-wise softmax of logits (numerically stable, from activation.c)
  softmax_rowwise(logits, probs);

  float loss = 0.0f;
  int nvalid = 0;
  for (int i = 0; i < N; i++) {
    if (targets[i] < 0) continue;          // ignored position
    tf_check_target_id(targets[i], V);
    float p = probs->data[i * V + targets[i]];
    loss -= logf(p > 1e-12f ? p : 1e-12f);  // guard against log(0)
    nvalid++;
  }
  return nvalid > 0 ? loss / (float)nvalid : 0.0f;
}

// Backward: dlogits[i,j] = (probs[i,j] - [j == targets[i]]) / nvalid for supervised
// rows; ignored rows (target < 0) contribute zero gradient.
void cross_entropy_backward(const Tensor *probs, const int *targets, Tensor *logits_grad) {
  const int N = probs->shape[0];
  const int V = probs->shape[1];

  int nvalid = 0;
  for (int i = 0; i < N; i++) if (targets[i] >= 0) nvalid++;
  if (nvalid == 0) return;
  const float inv_n = 1.0f / (float)nvalid;

  for (int i = 0; i < N; i++) {
    if (targets[i] < 0) continue;          // ignored -> zero gradient
    tf_check_target_id(targets[i], V);
    for (int j = 0; j < V; j++) {
      float g = probs->data[i * V + j];
      if (j == targets[i]) g -= 1.0f;
      logits_grad->grad[i * V + j] += g * inv_n;
    }
  }
}

// Per-row-weighted version of the above: the row weight comes from coeff[i]
// instead of being the uniform 1/nvalid. See loss.h for why every log-likelihood
// objective collapses to this one loop.
void pg_backward(const Tensor *probs, const int *targets, const float *coeff,
                 Tensor *logits_grad) {
  const int N = probs->shape[0];
  const int V = probs->shape[1];

  for (int i = 0; i < N; i++) {
    if (targets[i] < 0) continue;          // masked position (e.g. the prompt)
    if (coeff[i] == 0.0f) continue;        // clipped token: contributes nothing
    tf_check_target_id(targets[i], V);
    const float c = coeff[i];
    for (int j = 0; j < V; j++) {
      float g = probs->data[i * V + j];
      if (j == targets[i]) g -= 1.0f;
      logits_grad->grad[i * V + j] += g * c;
    }
  }
}

float logprob_of(const Tensor *probs, int row, int id) {
  const int V = probs->shape[1];
  tf_check_target_id(id, V);
  float p = probs->data[(size_t)row * V + id];
  return logf(p > 1e-12f ? p : 1e-12f);
}
