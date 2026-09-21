// optim.h - Adam / AdamW optimizer, gradient clipping, LR schedules
#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"

// Adam optimizer state. Holds first/second moment estimates for every scalar in
// the parameter set, flattened into one contiguous buffer per moment.
typedef struct {
  float *m;        // first moment (mean of grads)
  float *v;        // second moment (mean of squared grads)
  size_t total;    // total number of scalar parameters (size_t: no 2^31 overflow)
  int t;           // time step (for bias correction)
  float lr;        // learning rate
  float beta1;     // first-moment decay
  float beta2;     // second-moment decay
  float eps;       // numerical epsilon
  float weight_decay;  // DECOUPLED weight decay (AdamW); 0 = plain Adam
  float *lr_mult;      // optional per-TENSOR LR multiplier [n_params] (owned copy),
                       // NULL (the default) = every tensor uses lr — the historical,
                       // bit-identical step. See adam_set_lr_mult.
} Adam;

// Create optimizer state sized for the given parameter tensors (uses each
// tensor's `size`). Default hyperparameters: beta1=0.9, beta2=0.999, eps=1e-8,
// weight_decay=0 (i.e. plain Adam — bit-identical to the pre-AdamW behaviour).
Adam adam_create(Tensor **params, int n_params, float lr);

// Same, with DECOUPLED weight decay (AdamW): the decay is applied straight to the
// weights (w -= lr * wd * w) instead of being folded into the gradient, so it is
// not scaled by the adaptive 1/sqrt(v) term. This is the correct way to regularize
// an Adam-trained transformer, and it matters for fine-tuning.
//
// Decay is applied ONLY to tensors with ndim >= 2 (the weight matrices). 1-D
// tensors — biases and RMSNorm/LayerNorm gains — are left alone, the standard
// rule: decaying a norm gain toward zero fights the normalization itself.
Adam adamw_create(Tensor **params, int n_params, float lr, float weight_decay);

// One Adam/AdamW update step using each parameter tensor's accumulated `grad`.
void adam_step(Adam *o, Tensor **params, int n_params);

// Set the learning rate (call once per step when using a schedule).
void adam_set_lr(Adam *o, float lr);

// Per-tensor learning-rate multipliers: tensor p (in the order adam_step is given
// them) is updated with lr * mult[p], decay included (it is decoupled, lr-scaled).
// Used for LoRA+ (lr_B = ratio * lr_A; llama_lora_plus_mult). The array is COPIED;
// mult == NULL removes it (back to the plain step). mult[p] == 1 for every p is
// bit-identical to no multiplier (x * 1.0f == x exactly in IEEE float).
void adam_set_lr_mult(Adam *o, const float *mult, int n_params);

// Release optimizer state.
void adam_free(Adam *o);

// ---------------------------------------------------------------------------
// Gradient clipping
// ---------------------------------------------------------------------------

// Global gradient-norm clipping. Computes ONE L2 norm over every parameter's
// grad (the whole model treated as a single flat vector — not a per-tensor norm),
// and if it exceeds max_norm scales EVERY gradient by max_norm / total_norm, so
// the update direction is preserved and only its length is capped.
//
// Returns the total norm BEFORE clipping (a useful training diagnostic: a
// spiking grad norm is the usual first sign of divergence). A non-finite norm
// (NaN/Inf from an exploded step) zeros all grads and returns that norm, so one
// bad batch cannot poison the weights.
float clip_grad_norm(Tensor **params, int n_params, float max_norm);

// Scale every parameter's gradient in place. Used after gradient ACCUMULATION:
// running B micro-batches sums B gradients, so scaling by 1/B turns the sum into
// the mean — the gradient of an actual batch of B. Do this BEFORE clipping, so
// the clip threshold means the same thing whatever the batch size.
void scale_grads(Tensor **params, int n_params, float s);

// ---------------------------------------------------------------------------
// Learning-rate schedules
// ---------------------------------------------------------------------------

// Cosine decay with linear warmup — the standard transformer schedule:
//   step < warmup_steps : linear ramp 0 -> base_lr   (stabilizes the first steps,
//                                                     when Adam's v is still cold)
//   otherwise           : cosine from base_lr down to min_lr at total_steps
//   step >= total_steps : min_lr
// Pass warmup_steps = 0 for pure cosine. Safe for total_steps <= warmup_steps
// (returns base_lr rather than dividing by zero).
float lr_cosine_warmup(float base_lr, float min_lr, int step,
                       int warmup_steps, int total_steps);

#endif // OPTIM_H
