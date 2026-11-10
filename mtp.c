// mtp.c - Multi-token prediction head (see mtp.h)
#include "mtp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "loss.h"
#include "matrix.h"
#include "rmsnorm.h"
#include "sampler.h"

#define MTP_MAGIC 0x4850544Du   // 'MTPH'

MtpHead mtp_create(int d_model) {
  MtpHead h;
  h.d_model = d_model;
  int ws[2] = {d_model, d_model}, gs[1] = {d_model};
  h.W = tensor_create(ws, 2);
  h.g = tensor_create(gs, 1);
  memset(h.W.data, 0, h.W.size * sizeof(float));
  for (int i = 0; i < d_model; i++) {
    h.W.data[(size_t)i * d_model + i] = 1.0f;   // identity: start as the t+1 head
    h.g.data[i] = 1.0f;
  }
  mtp_zero_grad(&h);
  return h;
}

void mtp_free(MtpHead *h) {
  tensor_free(&h->W);
  tensor_free(&h->g);
}

void mtp_zero_grad(MtpHead *h) {
  tensor_zero_grad(&h->W);
  tensor_zero_grad(&h->g);
}

int mtp_collect_params(MtpHead *h, Tensor **list) {
  list[0] = &h->W;
  list[1] = &h->g;
  return 2;
}

MtpActs mtp_acts_create(int rows, int d_model, int vocab) {
  MtpActs a;
  a.cap = rows > 0 ? rows : 1;
  int hs[2] = {a.cap, d_model}, ls[2] = {a.cap, vocab};
  a.hc = tensor_create(hs, 2);
  a.u = tensor_create(hs, 2);
  a.z = tensor_create(hs, 2);
  a.logits = tensor_create(ls, 2);
  a.probs = tensor_create_nograd(ls, 2);
  return a;
}

void mtp_acts_free(MtpActs *a) {
  tensor_free(&a->hc); tensor_free(&a->u); tensor_free(&a->z);
  tensor_free(&a->logits); tensor_free(&a->probs);
}

static void set_rows(Tensor *t, int rows) {
  t->shape[0] = rows;
  t->size = (size_t)rows * (size_t)t->shape[1];
}

float mtp_loss_backward(MtpHead *h, const LlamaModel *m, const Tensor *final_hn,
                        const int *targets2, int n, float lambda, MtpActs *a, float *dh,
                        int *n_hit, int *n_rows) {
  const int dm = h->d_model;
  int *rows = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
  int *tc = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
  int nr = 0;
  for (int t = 0; t < n && nr < a->cap; t++)
    if (targets2[t] >= 0) { rows[nr] = t; tc[nr] = targets2[t]; nr++; }
  if (n_hit) *n_hit = 0;
  if (n_rows) *n_rows = nr;
  if (nr == 0) { free(rows); free(tc); return 0.0f; }

  set_rows(&a->hc, nr); set_rows(&a->u, nr); set_rows(&a->z, nr);
  set_rows(&a->logits, nr); set_rows(&a->probs, nr);
  for (int r = 0; r < nr; r++)
    memcpy(&a->hc.data[(size_t)r * dm], &final_hn->data[(size_t)rows[r] * dm],
           (size_t)dm * sizeof(float));

  // Forward: u = hc @ W, z = RMSNorm(u, g), logits = head(z).
  matmul(&a->hc, &h->W, &a->u);
  rmsnorm(&a->u, &h->g, &a->z, m->eps);
  llama_head_forward(m, &a->z, &a->logits);
  const float loss = cross_entropy_forward(&a->logits, tc, &a->probs);
  if (n_hit)
    for (int r = 0; r < nr; r++)
      if (sample_argmax(&a->logits, r) == tc[r]) (*n_hit)++;

  // Backward, scaled by lambda at the logits (everything below is linear in it).
  tensor_zero_grad(&a->logits); tensor_zero_grad(&a->z);
  tensor_zero_grad(&a->u); tensor_zero_grad(&a->hc);
  cross_entropy_backward(&a->probs, tc, &a->logits);
  for (size_t i = 0; i < a->logits.size; i++) a->logits.grad[i] *= lambda;
  if (m->tied) matmul_nt_backward(&a->z, &m->wte, &a->logits, 0);   // dz only: head shared
  else         matmul_backward_dx(&a->z, &m->lm_head, &a->logits);
  rmsnorm_backward(&a->u, &h->g, &a->z, &h->g, m->eps);
  matmul_backward(&a->hc, &h->W, &a->u);
  for (int r = 0; r < nr; r++) {
    float *dst = &dh[(size_t)rows[r] * dm];
    const float *src = &a->hc.grad[(size_t)r * dm];
    for (int j = 0; j < dm; j++) dst[j] += src[j];
  }
  free(rows); free(tc);
  return loss;
}

int mtp_save(const char *path, const MtpHead *h) {
  FILE *f = fopen(path, "wb");
  if (!f) return 1;
  const unsigned int magic = MTP_MAGIC;
  const int d = h->d_model;
  int ok = fwrite(&magic, 4, 1, f) == 1 && fwrite(&d, 4, 1, f) == 1 &&
           fwrite(h->W.data, sizeof(float), h->W.size, f) == h->W.size &&
           fwrite(h->g.data, sizeof(float), h->g.size, f) == h->g.size;
  if (fclose(f) != 0) ok = 0;
  return ok ? 0 : 1;
}

int mtp_load(const char *path, MtpHead *out) {
  FILE *f = fopen(path, "rb");
  if (!f) return 1;
  unsigned int magic = 0;
  int d = 0;
  if (fread(&magic, 4, 1, f) != 1 || magic != MTP_MAGIC || fread(&d, 4, 1, f) != 1 ||
      d <= 0 || d > 65536) {
    fclose(f);
    return 1;
  }
  MtpHead h = mtp_create(d);
  const int ok = fread(h.W.data, sizeof(float), h.W.size, f) == h.W.size &&
                 fread(h.g.data, sizeof(float), h.g.size, f) == h.g.size;
  fclose(f);
  if (!ok) { mtp_free(&h); return 1; }
  *out = h;
  return 0;
}
