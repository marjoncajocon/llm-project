// optim.c - Adam optimizer
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
  return o;
}

void adam_step(Adam *o, Tensor **params, int n_params) {
  o->t += 1;
  // Bias-correction factors for this step
  float bc1 = 1.0f - powf(o->beta1, (float)o->t);
  float bc2 = 1.0f - powf(o->beta2, (float)o->t);

  size_t idx = 0;  // running index into the flat m/v buffers
  for (int p = 0; p < n_params; p++) {
    Tensor *param = params[p];
    for (size_t i = 0; i < param->size; i++, idx++) {
      float g = param->grad[i];
      o->m[idx] = o->beta1 * o->m[idx] + (1.0f - o->beta1) * g;
      o->v[idx] = o->beta2 * o->v[idx] + (1.0f - o->beta2) * g * g;
      float m_hat = o->m[idx] / bc1;
      float v_hat = o->v[idx] / bc2;
      param->data[i] -= o->lr * m_hat / (sqrtf(v_hat) + o->eps);
    }
  }
}

void adam_free(Adam *o) {
  free(o->m);
  free(o->v);
  o->m = NULL;
  o->v = NULL;
  o->total = 0;
}
