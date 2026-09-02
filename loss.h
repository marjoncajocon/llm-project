// loss.h - Softmax cross-entropy loss
#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

// Softmax cross-entropy over rows. Each row of logits is one example; targets[i]
// is the correct class index for row i.
//   logits : (N, vocab_size)
//   targets: N class indices
//   probs  : (N, vocab_size) output, filled with row-softmax (for the backward pass)
// Returns the mean loss over the N rows.
float cross_entropy_forward(const Tensor *logits, const int *targets, Tensor *probs);

// Backward: logits_grad->grad[i,j] += (probs[i,j] - [j == targets[i]]) / N
void cross_entropy_backward(const Tensor *probs, const int *targets, Tensor *logits_grad);

// ---------------------------------------------------------------------------
// Policy-gradient backward (used by GRPO, grpo.h)
// ---------------------------------------------------------------------------

// The same gradient shape as cross_entropy_backward, but with a PER-ROW weight
// instead of the uniform 1/N:
//
//   logits_grad->grad[i,j] += coeff[i] * (probs[i,j] - [j == targets[i]])
//
// That is the whole trick behind the RL trainer. For a sampled token o_i the
// score function is d/dlogits log p(o_i) = onehot(o_i) - p, so ANY objective of
// the form sum_i w_i * log p(o_i) has gradient -w_i * (p - onehot) on the
// logits. Cross-entropy is the special case w_i = -1/N; a policy gradient is the
// case w_i = advantage (times an importance ratio, times 1/|response|). Rows
// with targets[i] < 0 are IGNORED, exactly as in cross_entropy_backward, which
// is how the prompt span is masked out of the update.
//
// coeff must have one entry per row of probs. A zero coefficient contributes
// nothing (that is how a clipped token is dropped).
void pg_backward(const Tensor *probs, const int *targets, const float *coeff,
                 Tensor *logits_grad);

// log p(id) at one row of an already-computed softmax (probs from
// cross_entropy_forward). Clamped away from log(0). Used to record the
// per-token log-probabilities a policy assigned to the tokens it sampled.
float logprob_of(const Tensor *probs, int row, int id);

#endif // LOSS_H
