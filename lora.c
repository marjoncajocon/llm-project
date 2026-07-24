// lora.c - LoRA low-rank adapters (see lora.h)
#include "lora.h"

#include <stdlib.h>
#include <math.h>

#include "matrix.h"

static Tensor mk(int r, int c) { int s[2] = {r, c}; return tensor_create(s, 2); }

LoraAdapter lora_create(int d_in, int d_out, int r, float alpha) {
  LoraAdapter l;
  l.d_in = d_in; l.d_out = d_out; l.r = r;
  l.scale = (r > 0) ? alpha / (float)r : 0.0f;
  l.A = mk(d_in, r);
  l.B = mk(r, d_out);
  return l;
}

void lora_free(LoraAdapter *l) {
  tensor_free(&l->A);
  tensor_free(&l->B);
}

void lora_init(LoraAdapter *l) {
  // A gets small random values scaled by 1/sqrt(d_in) (keeps x @ A at a sane
  // magnitude regardless of width); B stays zero so the correction starts at 0.
  const float s = 1.0f / sqrtf((float)(l->d_in > 0 ? l->d_in : 1));
  for (size_t i = 0; i < l->A.size; i++)
    l->A.data[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * s;
  for (size_t i = 0; i < l->B.size; i++) l->B.data[i] = 0.0f;
}

LoraCache lora_cache_create(int n, int r, int d_out) {
  LoraCache c;
  c.xa = mk(n, r);
  c.y = mk(n, d_out);
  return c;
}

void lora_cache_free(LoraCache *c) {
  tensor_free(&c->xa);
  tensor_free(&c->y);
}

void lora_forward(const LoraAdapter *l, const Tensor *x, Tensor *y, LoraCache *c) {
  matmul(x, &l->A, &c->xa);            // xa = x @ A          (n, r)
  matmul(&c->xa, &l->B, &c->y);        // y_lora = xa @ B     (n, d_out)
  for (size_t i = 0; i < y->size; i++) y->data[i] += l->scale * c->y.data[i];
}

void lora_backward(const LoraAdapter *l, const Tensor *x, const Tensor *y, LoraCache *c) {
  // The forward added scale * (xa @ B) to y, so the gradient reaching the inner
  // matmul is scale * dy. Park it on the cache tensor and reuse matmul_backward
  // for both stages rather than hand-rolling the transposes.
  tensor_zero_grad(&c->xa);
  for (size_t i = 0; i < c->y.size; i++) c->y.grad[i] = l->scale * y->grad[i];

  matmul_backward(&c->xa, &l->B, &c->y);   // dxa += dY @ B^T ; dB += xa^T @ dY
  matmul_backward(x, &l->A, &c->xa);       // dx  += dxa @ A^T ; dA += x^T @ dxa
}
