// loss.c - Softmax cross-entropy loss
#include "loss.h"

#include <math.h>

#include "activation.h"
#include "check.h"
#include "matrix.h"   // tf_parallel_for

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

// The two backward loops below write one logits_grad row per supervised row and
// read nothing another row writes, so large ones are split by rows across the
// thread pool; every element is computed by the same expression -> bit-identical
// for any thread count (selftest [43]).
typedef struct {
  const Tensor *probs; const int *targets; const float *coeff; float inv_n;
  Tensor *logits_grad;
} RowGradJob;

// coeff == NULL: the uniform CE weight inv_n; else the per-row pg weight.
static void row_grad_rows(int r0, int r1, void *ctx) {
  const RowGradJob *t = (const RowGradJob*)ctx;
  const int V = t->probs->shape[1];
  for (int i = r0; i < r1; i++) {
    if (t->targets[i] < 0) continue;       // ignored / masked -> zero gradient
    if (t->coeff && t->coeff[i] == 0.0f) continue;   // clipped token: contributes nothing
    tf_check_target_id(t->targets[i], V);
    const float c = t->coeff ? t->coeff[i] : t->inv_n;
    const float *p = &t->probs->data[(size_t)i * V];
    float *g = &t->logits_grad->grad[(size_t)i * V];
    for (int j = 0; j < V; j++) {
      float d = p[j];
      if (j == t->targets[i]) d -= 1.0f;
      g[j] += d * c;
    }
  }
}

#define ROWGRAD_THREAD_MIN (1L << 15)   // elements; below this, one thread

static void row_grad_run(const RowGradJob *t) {
  const int N = t->probs->shape[0];
  if (N < 2 || (size_t)N * (size_t)t->probs->shape[1] < (size_t)ROWGRAD_THREAD_MIN)
    row_grad_rows(0, N, (void*)t);
  else
    tf_parallel_for(N, row_grad_rows, (void*)t);
}

// Backward: dlogits[i,j] = (probs[i,j] - [j == targets[i]]) / nvalid for supervised
// rows; ignored rows (target < 0) contribute zero gradient.
void cross_entropy_backward(const Tensor *probs, const int *targets, Tensor *logits_grad) {
  const int N = probs->shape[0];

  int nvalid = 0;
  for (int i = 0; i < N; i++) if (targets[i] >= 0) nvalid++;
  if (nvalid == 0) return;
  const float inv_n = 1.0f / (float)nvalid;
  RowGradJob t = { probs, targets, NULL, inv_n, logits_grad };
  row_grad_run(&t);
}

// Per-row-weighted version of the above: the row weight comes from coeff[i]
// instead of being the uniform 1/nvalid. See loss.h for why every log-likelihood
// objective collapses to this one loop.
void pg_backward(const Tensor *probs, const int *targets, const float *coeff,
                 Tensor *logits_grad) {
  RowGradJob t = { probs, targets, coeff, 0.0f, logits_grad };
  row_grad_run(&t);
}

float logprob_of(const Tensor *probs, int row, int id) {
  const int V = probs->shape[1];
  tf_check_target_id(id, V);
  float p = probs->data[(size_t)row * V + id];
  return logf(p > 1e-12f ? p : 1e-12f);
}
