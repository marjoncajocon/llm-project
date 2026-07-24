// lora.h - LoRA (Low-Rank Adaptation) adapters for fine-tuning a FROZEN base model.
//
// A full fine-tune has to carry a gradient and two Adam moments for every weight.
// LoRA instead freezes W and learns a low-rank correction beside it:
//
//   y = x @ W  +  scale * (x @ A) @ B          scale = alpha / r
//
// with A: (d_in, r) and B: (r, d_out) for a small rank r, so only r*(d_in+d_out)
// values are trained instead of d_in*d_out. At r=4 on a (512,512) projection that
// is 4k trainable values instead of 262k — a ~64x cut in optimizer state, which is
// what makes fine-tuning a real model feasible on a CPU.
//
// B is initialized to ZERO (and A randomly), so the correction starts at exactly
// zero: an adapted model is bit-identical to its base before the first step. That
// property is checked in the smoke test — it is what makes attaching adapters to a
// pretrained checkpoint safe.
#ifndef LORA_H
#define LORA_H

#include "tensor.h"

typedef struct {
  int d_in, d_out, r;
  float scale;    // alpha / r
  Tensor A;       // (d_in, r)  trainable
  Tensor B;       // (r, d_out) trainable — starts at zero
} LoraAdapter;

// Cached activations for one adapter at a given sequence length, so the backward
// needs no recomputation. `xa` holds x @ A; `y` carries the scaled gradient into
// the second matmul (its data is unused).
typedef struct {
  Tensor xa;      // (n, r)
  Tensor y;       // (n, d_out) — grad carrier only
} LoraCache;

// alpha scales the correction (scale = alpha / r); alpha == r means scale 1.
LoraAdapter lora_create(int d_in, int d_out, int r, float alpha);
void lora_free(LoraAdapter *l);

// A <- small random, B <- 0 (so the correction is exactly zero at init).
// Uses rand(); seed with srand() for reproducibility.
void lora_init(LoraAdapter *l);

LoraCache lora_cache_create(int n, int r, int d_out);
void lora_cache_free(LoraCache *c);

// y += scale * (x @ A) @ B. Call AFTER the frozen base projection has written y,
// so the adapter adds its correction onto it. x: (n, d_in), y: (n, d_out).
void lora_forward(const LoraAdapter *l, const Tensor *x, Tensor *y, LoraCache *c);

// Backward: reads y->grad, accumulates into x->grad and the adapter's A/B grads.
// Must be called with the SAME cache the forward filled.
void lora_backward(const LoraAdapter *l, const Tensor *x, const Tensor *y, LoraCache *c);

#endif // LORA_H
