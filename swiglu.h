// swiglu.h - SwiGLU gated feed-forward network (LLaMA-family)
#ifndef SWIGLU_H
#define SWIGLU_H

#include "tensor.h"

// SwiGLU FFN (LLaMA-family), replacing the GPT-2 GELU MLP. No biases:
//   g = x @ W_gate            (n, d_ff)
//   u = x @ W_up              (n, d_ff)
//   h = SiLU(g) (*) u         (elementwise; SiLU(z) = z * sigmoid(z))
//   y = h @ W_down            (n, d_model)
// W_gate, W_up: (d_model, d_ff);  W_down: (d_ff, d_model).
//
// The forward caches g, u, a=SiLU(g), h so the backward can run without
// recomputation; create/free one cache per activation shape.
typedef struct {
  Tensor g, u, a, h;   // each (n, d_ff)
} SwigluCache;

SwigluCache swiglu_cache_create(int n, int d_ff);
void swiglu_cache_free(SwigluCache *c);

void swiglu_forward(const Tensor *x, const Tensor *W_gate, const Tensor *W_up,
                    const Tensor *W_down, SwigluCache *c, Tensor *y);

// Backward: reads y->grad and the cache; accumulates x->grad, W_gate->grad,
// W_up->grad, W_down->grad. Zeros the cache's own grads internally.
void swiglu_backward(const Tensor *x, const Tensor *W_gate, const Tensor *W_up,
                     const Tensor *W_down, SwigluCache *c, const Tensor *y);

// Same, but only x->grad is produced — the three weight gradients are neither
// read nor written. For a FROZEN FFN (LoRA fine-tuning) that skips roughly half
// the sub-layer's matmul work, and the FFN dominates a layer's FLOPs.
void swiglu_backward_dx(const Tensor *x, const Tensor *W_gate, const Tensor *W_up,
                        const Tensor *W_down, SwigluCache *c, const Tensor *y);

#endif // SWIGLU_H
