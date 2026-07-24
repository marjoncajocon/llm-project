// optim.c - Adam / AdamW optimizer, gradient clipping, LR schedules
#include "optim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static size_t count_params(Tensor **params, int n_params) {
  size_t total = 0;
  for (int i = 0; i < n_params; i++) total += params[i]->size;
  return total;
}

Adam adam_create(Tensor **params, int n_params, float lr) {
  Adam o;
  o.total = count_params(params, n_params);
  o.m = (float*)malloc(o.total * sizeof(float));
  o.v = (float*)malloc(o.total * sizeof(float));
  if (!o.m || !o.v) {
    fprintf(stderr, "Error: Adam allocation failed (%zu params)\n", o.total);
    exit(EXIT_FAILURE);
  }
  memset(o.m, 0, o.total * sizeof(float));
  memset(o.v, 0, o.total * sizeof(float));
  o.t = 0;
  o.lr = lr;
  o.beta1 = 0.9f;
  o.beta2 = 0.999f;
  o.eps = 1e-8f;
  o.weight_decay = 0.0f;
  return o;
}

Adam adamw_create(Tensor **params, int n_params, float lr, float weight_decay) {
  Adam o = adam_create(params, n_params, lr);
  o.weight_decay = weight_decay;
  return o;
}

void adam_set_lr(Adam *o, float lr) { o->lr = lr; }

void adam_step(Adam *o, Tensor **params, int n_params) {
  o->t += 1;
  // Bias-correction factors for this step
  float bc1 = 1.0f - powf(o->beta1, (float)o->t);
  float bc2 = 1.0f - powf(o->beta2, (float)o->t);

  size_t idx = 0;  // running index into the flat m/v buffers
  for (int p = 0; p < n_params; p++) {
    Tensor *param = params[p];
    // Decoupled decay applies to weight matrices only, never to 1-D biases /
    // norm gains (see optim.h). wd == 0 leaves this a plain Adam step.
    const float wd = (o->weight_decay != 0.0f && param->ndim >= 2) ? o->weight_decay : 0.0f;
    for (size_t i = 0; i < param->size; i++, idx++) {
      float g = param->grad[i];
      o->m[idx] = o->beta1 * o->m[idx] + (1.0f - o->beta1) * g;
      o->v[idx] = o->beta2 * o->v[idx] + (1.0f - o->beta2) * g * g;
      float m_hat = o->m[idx] / bc1;
      float v_hat = o->v[idx] / bc2;
      float w = param->data[i];
      // AdamW: the decay term is NOT divided by sqrt(v_hat) — that decoupling is
      // the whole point, so regularization strength doesn't depend on grad scale.
      param->data[i] = w - o->lr * (m_hat / (sqrtf(v_hat) + o->eps) + wd * w);
    }
  }
}

// ---------------------------------------------------------------------------
// Gradient clipping
// ---------------------------------------------------------------------------

float clip_grad_norm(Tensor **params, int n_params, float max_norm) {
  // Accumulate in double: summing millions of squared float32 grads loses too
  // much precision otherwise, and this norm gates whether we scale at all.
  double sumsq = 0.0;
  for (int p = 0; p < n_params; p++) {
    const Tensor *t = params[p];
    if (!t->grad) continue;
    for (size_t i = 0; i < t->size; i++) {
      double g = (double)t->grad[i];
      sumsq += g * g;
    }
  }
  float total = (float)sqrt(sumsq);

  if (!isfinite(total)) {
    // An exploded/NaN step: drop the whole update rather than write NaNs into
    // the weights (from which no later step can recover).
    for (int p = 0; p < n_params; p++) {
      Tensor *t = params[p];
      if (t->grad) memset(t->grad, 0, t->size * sizeof(float));
    }
    return total;
  }
  if (max_norm <= 0.0f || total <= max_norm) return total;

  const float scale = max_norm / (total + 1e-6f);
  for (int p = 0; p < n_params; p++) {
    Tensor *t = params[p];
    if (!t->grad) continue;
    for (size_t i = 0; i < t->size; i++) t->grad[i] *= scale;
  }
  return total;
}

void scale_grads(Tensor **params, int n_params, float s) {
  for (int p = 0; p < n_params; p++) {
    Tensor *t = params[p];
    if (!t->grad) continue;
    for (size_t i = 0; i < t->size; i++) t->grad[i] *= s;
  }
}

// ---------------------------------------------------------------------------
// Learning-rate schedules
// ---------------------------------------------------------------------------

float lr_cosine_warmup(float base_lr, float min_lr, int step,
                       int warmup_steps, int total_steps) {
  if (step < 0) step = 0;
  if (warmup_steps > 0 && step < warmup_steps)
    return base_lr * (float)(step + 1) / (float)warmup_steps;
  if (total_steps <= warmup_steps) return base_lr;   // no decay phase to speak of
  if (step >= total_steps) return min_lr;

  const float PI = 3.14159265358979323846f;
  float progress = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
  float cosine = 0.5f * (1.0f + cosf(PI * progress));   // 1 -> 0
  return min_lr + (base_lr - min_lr) * cosine;
}

void adam_free(Adam *o) {
  free(o->m);
  free(o->v);
  o->m = NULL;
  o->v = NULL;
  o->total = 0;
}
