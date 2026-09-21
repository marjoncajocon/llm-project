// swiglu.c - SwiGLU gated feed-forward network (LLaMA-family)
#include "swiglu.h"

#include <math.h>

#include "matrix.h"

// SiLU (a.k.a. swish): SiLU(z) = z * sigmoid(z).
static float silu(float z) {
  float s = 1.0f / (1.0f + expf(-z));
  return z * s;
}
// d/dz SiLU = s * (1 + z * (1 - s)),  s = sigmoid(z).
static float silu_prime(float z) {
  float s = 1.0f / (1.0f + expf(-z));
  return s * (1.0f + z * (1.0f - s));
}

// The elementwise gate (SiLU(g) * u) and its backward are independent per
// element, so a large one is cut into contiguous element ranges on the thread
// pool. Each element runs exactly the serial arithmetic (the backward's two
// loops fused per element: dg[i] only reads da[i], written just before), so the
// result is the same for every thread count.
static void silu_fwd_range(int i0, int i1, void *ctx) {
  SwigluCache *c = (SwigluCache*)ctx;
  for (int i = i0; i < i1; i++) {
    c->a.data[i] = silu(c->g.data[i]);       // a = SiLU(g)
    c->h.data[i] = c->a.data[i] * c->u.data[i];  // h = a (*) u
  }
}

static void silu_bwd_range(int i0, int i1, void *ctx) {
  SwigluCache *c = (SwigluCache*)ctx;
  for (int i = i0; i < i1; i++) {
    float dh = c->h.grad[i];
    c->a.grad[i] += dh * c->u.data[i];        // da += dh * u
    c->u.grad[i] += dh * c->a.data[i];        // du += dh * a
  }
  for (int i = i0; i < i1; i++)
    c->g.grad[i] += c->a.grad[i] * silu_prime(c->g.data[i]);  // dg += da * SiLU'(g)
}

#define SWIGLU_THREAD_MIN (1 << 14)   // elements; below this, one thread

static void ew_run(int nf, void (*fn)(int, int, void*), SwigluCache *c) {
  if (nf < SWIGLU_THREAD_MIN) fn(0, nf, c);
  else tf_parallel_for(nf, fn, c);
}

SwigluCache swiglu_cache_create(int n, int d_ff) {
  SwigluCache c;
  int sh[2] = {n, d_ff};
  c.g = tensor_create(sh, 2);
  c.u = tensor_create(sh, 2);
  c.a = tensor_create(sh, 2);
  c.h = tensor_create(sh, 2);
  return c;
}

void swiglu_cache_free(SwigluCache *c) {
  tensor_free(&c->g); tensor_free(&c->u); tensor_free(&c->a); tensor_free(&c->h);
}

void swiglu_forward_lora(const Tensor *x, const Tensor *W_gate, const Tensor *W_up,
                         const Tensor *W_down, SwigluCache *c, Tensor *y,
                         const SwigluLora *sl) {
  const int nf = c->g.shape[0] * c->g.shape[1];
  matmul(x, W_gate, &c->g);                 // g = x @ W_gate
  if (sl && sl->gate) lora_forward(sl->gate, x, &c->g, sl->cg);   // before SiLU
  matmul(x, W_up, &c->u);                    // u = x @ W_up
  if (sl && sl->up) lora_forward(sl->up, x, &c->u, sl->cu);
  ew_run(nf, silu_fwd_range, c);            // a = SiLU(g), h = a (*) u
  matmul(&c->h, W_down, y);                   // y = h @ W_down
  if (sl && sl->down) lora_forward(sl->down, &c->h, y, sl->cd);
}

void swiglu_forward(const Tensor *x, const Tensor *W_gate, const Tensor *W_up,
                    const Tensor *W_down, SwigluCache *c, Tensor *y) {
  swiglu_forward_lora(x, W_gate, W_up, W_down, c, y, NULL);
}

// `frozen` selects the input-gradient-only matmul backward: with the three FFN
// weights frozen (LoRA fine-tuning) their dW is computed and thrown away, and
// the FFN is the bulk of a layer's FLOPs (d_ff is typically 4*d_model across
// three matrices), so skipping it is the single biggest saving available.
static void swiglu_backward_impl(const Tensor *x, const Tensor *W_gate, const Tensor *W_up,
                                 const Tensor *W_down, SwigluCache *c, const Tensor *y,
                                 const SwigluLora *sl, int frozen) {
  const int nf = c->g.shape[0] * c->g.shape[1];
  tensor_zero_grad(&c->g); tensor_zero_grad(&c->u);
  tensor_zero_grad(&c->a); tensor_zero_grad(&c->h);

  if (frozen) matmul_backward_dx(&c->h, W_down, y);   // dh only
  else        matmul_backward(&c->h, W_down, y);      // dh (h->grad), dW_down
  // The down-adapter saw h too: its dh must be in h->grad BEFORE the elementwise
  // gate backward below reads it.
  if (sl && sl->down) lora_backward(sl->down, &c->h, y, sl->cd);
  ew_run(nf, silu_bwd_range, c);            // da, du, then dg = da * SiLU'(g)

  if (frozen) {
    matmul_backward_dx(x, W_gate, &c->g);     // dx += dg @ W_gate^T
    matmul_backward_dx(x, W_up, &c->u);       // dx += du @ W_up^T
  } else {
    matmul_backward(x, W_gate, &c->g);        // dx += dg @ W_gate^T, dW_gate += x^T @ dg
    matmul_backward(x, W_up, &c->u);          // dx += du @ W_up^T,   dW_up   += x^T @ du
  }
  // Gate/up adapters read dg/du (their outputs' grads) and add into dx.
  if (sl && sl->gate) lora_backward(sl->gate, x, &c->g, sl->cg);
  if (sl && sl->up)   lora_backward(sl->up, x, &c->u, sl->cu);
}

void swiglu_backward(const Tensor *x, const Tensor *W_gate, const Tensor *W_up,
                     const Tensor *W_down, SwigluCache *c, const Tensor *y) {
  swiglu_backward_impl(x, W_gate, W_up, W_down, c, y, NULL, 0);
}

void swiglu_backward_dx(const Tensor *x, const Tensor *W_gate, const Tensor *W_up,
                        const Tensor *W_down, SwigluCache *c, const Tensor *y) {
  swiglu_backward_impl(x, W_gate, W_up, W_down, c, y, NULL, 1);
}

void swiglu_backward_lora(const Tensor *x, const Tensor *W_gate, const Tensor *W_up,
                          const Tensor *W_down, SwigluCache *c, const Tensor *y,
                          const SwigluLora *sl, int frozen) {
  swiglu_backward_impl(x, W_gate, W_up, W_down, c, y, sl, frozen);
}
