// mtp.h - Multi-token prediction (MTP) head: predict token t+2 as well as t+1.
//
// DeepSeek-V3, Nemotron 3 and others train an extra head that predicts the token
// AFTER next. It densifies the training signal (every response row teaches two
// targets), and at inference the head's guess is a free speculative draft.
//
// This is the smallest honest version: depth 1, one d x d projection and an
// RMSNorm on the trunk's final hidden state, then the model's OWN output head:
//
//   z_t      = RMSNorm(final_hn_t @ W, g)        W (d, d) = identity at init, g = 1
//   logits2  = z_t @ lm_head   (or z_t @ wte^T when tied)
//   L_mtp    = lambda * CE(logits2_t, ids[t+2])  over rows whose t+2 is supervised
//
// The gradient flows into W, g and back into the TRUNK (through final_hn), so
// the shared layers learn representations that also carry the second token.
// The output head itself gets no MTP gradient: it is shared with the main loss
// and, under LoRA (the usual fine-tune here), frozen anyway.
//
// Identity init means step 0 predicts t+2 with the t+1 head -- a weak but sane
// start, instead of noise that would first have to be unlearned.
#ifndef MTP_H
#define MTP_H

#include "llama.h"
#include "tensor.h"

typedef struct {
  int d_model;
  Tensor W;   // (d_model, d_model), with grad
  Tensor g;   // (d_model), with grad
} MtpHead;

MtpHead mtp_create(int d_model);
void mtp_free(MtpHead *h);
void mtp_zero_grad(MtpHead *h);
// The two learnable tensors, for an optimizer: list[0] = &W, list[1] = &g.
int mtp_collect_params(MtpHead *h, Tensor **list);

// Scratch for one sequence of up to `rows` supervised rows.
typedef struct {
  int cap;
  Tensor hc, u, z, logits, probs;
} MtpActs;

MtpActs mtp_acts_create(int rows, int d_model, int vocab);
void mtp_acts_free(MtpActs *a);

// Forward + backward of lambda * CE on the rows where targets2[t] >= 0
// (targets2[t] = the token at t+2, or -1). final_hn is (n, d_model) -- the
// trunk's final RMSNorm output for the whole sequence. ADDS lambda * dL/dfinal_hn
// into dh (n x d_model floats; the caller zeroes it), and accumulates W.grad and
// g.grad. Returns the unweighted mean CE (0 when no row is supervised); writes
// how many of those rows the head predicted exactly (argmax) to *n_hit and the
// row count to *n_rows when non-NULL.
float mtp_loss_backward(MtpHead *h, const LlamaModel *m, const Tensor *final_hn,
                        const int *targets2, int n, float lambda, MtpActs *a, float *dh,
                        int *n_hit, int *n_rows);

// Save / load the head (magic 'MTPH', d_model, then W and g as float32).
int mtp_save(const char *path, const MtpHead *h);
int mtp_load(const char *path, MtpHead *out);

#endif // MTP_H
