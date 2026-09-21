// muon.h - Muon optimizer (MomentUm Orthogonalized by Newton-Schulz)
//
// Muon (Keller Jordan, 2024; scaled to LLM pre-training by Moonshot AI, "Muon is
// Scalable for LLM Training", 2025 — the optimizer behind Kimi K2) replaces
// Adam's per-SCALAR adaptive step with a per-MATRIX one. For each 2-D weight W
// with gradient G:
//
//   M  <- mu * M + G                       momentum (mu = 0.95)
//   U  <- G + mu * M                       Nesterov look-ahead (else U = M)
//   O  <- NewtonSchulz5(U)                 ~= the orthogonal factor U V^T of
//                                          U = U S V^T: every singular value
//                                          pushed to ~1, directions kept
//   W  <- W - lr * (0.2 * sqrt(max(rows, cols)) * O + wd * W)
//
// Why: a raw gradient/momentum matrix is very low-rank in practice — a few
// directions dominate and Adam's elementwise step follows them. Orthogonalizing
// gives every direction the same step length, so the rare-but-useful directions
// learn too. The 0.2*sqrt(max(rows,cols)) factor is Moonshot's RMS matching: an
// orthogonal (r x c) matrix has element RMS 1/sqrt(max(r,c)), so this makes the
// update's RMS ~0.2 — about what AdamW's update RMS is — and an LR / weight-decay
// already tuned for AdamW transfers without a new sweep.
//
// Muon is only defined for MATRICES. Everything else stays on AdamW, which this
// optimizer runs internally over the complement of the routing mask:
//   - 1-D tensors (RMSNorm/LayerNorm gains, biases): there is no matrix to
//     orthogonalize — for a vector U/|U| is just sign-like normalization;
//   - the token embedding and the output head (lm_head): these are 2-D but not
//     "hidden" linear maps. An embedding row only gets gradient when its token
//     occurs, so orthogonalizing across the vocab spreads update mass onto rows
//     of tokens that were never seen; the output head is similarly per-class.
//     Both Keller Jordan's and Moonshot's recipes keep them on AdamW.
// The caller decides the routing (muon_route below implements the standard rule).
//
// Numerics: Newton-Schulz runs in float32 through the ordinary matmul (so it
// gets the AVX2/threaded kernels). It does NOT converge to exactly orthogonal:
// the quintic coefficients (3.4445, -4.7750, 2.0315) are tuned for speed, and
// they leave the singular values in a band around 1 (roughly 0.7..1.2) rather
// than at 1 — by design; the step is insensitive to that.
#ifndef MUON_H
#define MUON_H

#include "tensor.h"
#include "optim.h"

typedef struct {
  int n_params;        // size of the parameter list this was created for
  int *is_muon;        // [n_params] routing mask: 1 = Muon, 0 = the inner AdamW
  float **mom;         // [n_params] momentum buffer per Muon tensor (NULL for AdamW ones)
  Tensor **sub;        // [n_params] scratch: the AdamW sub-list, rebuilt per step
  int n_adam;          // how many params the inner AdamW owns
  Adam adam;           // AdamW over the non-Muon params (same lr and weight decay)
  float lr;            // shared learning rate (Muon + inner AdamW)
  float momentum;      // mu (default 0.95)
  int nesterov;        // 1 = Nesterov momentum (default)
  int ns_steps;        // Newton-Schulz iterations (default 5)
  float weight_decay;  // decoupled, applied to BOTH groups' matrices (AdamW rule
                       // for its group: 1-D tensors are never decayed)
  float *lr_mult;      // optional per-tensor LR multiplier [n_params]; NULL = none
} Muon;

// Routing rule: mask[i] = 1 for every 2-D tensor NOT in `exclude` (pass the
// embedding and output head there), 0 otherwise. Returns the Muon count.
int muon_route(Tensor **params, int n_params, Tensor *const *exclude, int n_exclude,
               int *mask);

// Create the optimizer for this parameter list and routing mask (copied). Every
// Muon tensor must be 2-D. Defaults: momentum 0.95 Nesterov, 5 NS steps; the
// inner AdamW uses adamw_create's defaults (beta1 0.9, beta2 0.999, eps 1e-8).
Muon muon_create(Tensor **params, int n_params, const int *mask, float lr,
                 float weight_decay);

// One step on each tensor's accumulated grad. `params` must be the same list
// (same order) that muon_create saw.
void muon_step(Muon *o, Tensor **params, int n_params);

void muon_set_lr(Muon *o, float lr);   // sets the inner AdamW's LR too

// Per-tensor LR multipliers over the FULL parameter list (copied; NULL removes
// them): a Muon tensor p steps with lr * mult[p] (decay included), and the inner
// AdamW gets the matching sub-list. For LoRA+ under --optim muon.
void muon_set_lr_mult(Muon *o, const float *mult);
void muon_free(Muon *o);

// The orthogonalization on its own: O ~= NewtonSchulz(G / ||G||_F), steps
// iterations of X <- a X + (b A + c A A) X with A = X X^T. rows x cols row-major;
// works on the SMALLER Gram side (transposes internally when rows > cols). G and
// O may alias. A zero G gives a zero O.
void muon_newton_schulz(const float *G, int rows, int cols, float *O, int steps);

#endif // MUON_H
