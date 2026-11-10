// deltanet.c - Gated DeltaNet layer (see deltanet.h)
#include "deltanet.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

#define DN_EPS 1e-6f

static Tensor mk(int r, int c) { int s[2] = {r, c}; return tensor_create(s, 2); }
static Tensor mk1(int n) { int s[1] = {n}; return tensor_create(s, 1); }

DeltaNet deltanet_create(int d_model, int n_heads, int head_dim) {
  DeltaNet dn;
  const int qd = n_heads * head_dim;
  dn.d_model = d_model; dn.n_heads = n_heads; dn.head_dim = head_dim;
  dn.Wq = mk(d_model, qd); dn.Wk = mk(d_model, qd); dn.Wv = mk(d_model, qd);
  dn.Wa = mk(d_model, n_heads); dn.Wb = mk(d_model, n_heads);
  dn.ba = mk1(n_heads); dn.bb = mk1(n_heads);
  dn.Wo = mk(qd, d_model);
  deltanet_zero_grad(&dn);
  return dn;
}

int deltanet_collect_params(DeltaNet *dn, Tensor **l) {
  l[0] = &dn->Wq; l[1] = &dn->Wk; l[2] = &dn->Wv; l[3] = &dn->Wa;
  l[4] = &dn->Wb; l[5] = &dn->ba; l[6] = &dn->bb; l[7] = &dn->Wo;
  return 8;
}

void deltanet_free(DeltaNet *dn) {
  Tensor *l[8];
  deltanet_collect_params(dn, l);
  for (int i = 0; i < 8; i++) tensor_free(l[i]);
}

void deltanet_zero_grad(DeltaNet *dn) {
  Tensor *l[8];
  deltanet_collect_params(dn, l);
  for (int i = 0; i < 8; i++) tensor_zero_grad(l[i]);
}

static float frand(float scale) { return scale * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f); }

void deltanet_randomize(DeltaNet *dn) {
  Tensor *l[8];
  deltanet_collect_params(dn, l);
  const float s = 1.0f / sqrtf((float)dn->d_model);
  for (int i = 0; i < 8; i++)
    for (size_t j = 0; j < l[i]->size; j++) l[i]->data[j] = frand(s);
  for (int h = 0; h < dn->n_heads; h++) { dn->ba.data[h] = 3.0f; dn->bb.data[h] = 0.0f; }
}

DeltaNetCache deltanet_cache_create(const DeltaNet *dn, int max_len) {
  DeltaNetCache c;
  const int H = dn->n_heads, qd = H * dn->head_dim, hh = dn->head_dim * dn->head_dim;
  c.cap = max_len; c.n = 0;
  c.q = mk(max_len, qd); c.k = mk(max_len, qd); c.v = mk(max_len, qd);
  c.ga = mk(max_len, H); c.gb = mk(max_len, H); c.o = mk(max_len, qd);
  c.qn = (float*)malloc((size_t)max_len * qd * sizeof(float));
  c.kn = (float*)malloc((size_t)max_len * qd * sizeof(float));
  c.qnorm = (float*)malloc((size_t)max_len * H * sizeof(float));
  c.knorm = (float*)malloc((size_t)max_len * H * sizeof(float));
  c.alpha = (float*)malloc((size_t)max_len * H * sizeof(float));
  c.beta = (float*)malloc((size_t)max_len * H * sizeof(float));
  c.S = (float*)calloc((size_t)(max_len + 1) * H * hh, sizeof(float));
  return c;
}

void deltanet_cache_free(DeltaNetCache *c) {
  tensor_free(&c->q); tensor_free(&c->k); tensor_free(&c->v);
  tensor_free(&c->ga); tensor_free(&c->gb); tensor_free(&c->o);
  free(c->qn); free(c->kn); free(c->qnorm); free(c->knorm);
  free(c->alpha); free(c->beta); free(c->S);
}

static float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

// L2-normalize n floats: out = x / |x|, returns |x|.
static float l2n(const float *x, float *out, int n) {
  float ss = 0.0f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  const float nr = sqrtf(ss + DN_EPS);
  for (int i = 0; i < n; i++) out[i] = x[i] / nr;
  return nr;
}

// One head, one token: the recurrence. Sp = S_{t-1}, Sn = S_t (hd x hd, row =
// key dim i, column = value dim j); qn/kn normalized; writes o = S_t^T qn.
// Shared by the forward and by deltanet_step, so the two are bit-identical.
static void dn_head_token(const float *Sp, float *Sn, const float *qn, const float *kn,
                          const float *v, float alpha, float beta, float *o, int hd) {
  for (int j = 0; j < hd; j++) {
    float u = 0.0f;                                   // u_j = (k^T S_{t-1})_j
    for (int i = 0; i < hd; i++) u += kn[i] * Sp[i * hd + j];
    const float w = beta * (v[j] - alpha * u);        // the column's write
    for (int i = 0; i < hd; i++)
      Sn[i * hd + j] = alpha * Sp[i * hd + j] + kn[i] * w;
  }
  for (int j = 0; j < hd; j++) {
    float acc = 0.0f;
    for (int i = 0; i < hd; i++) acc += Sn[i * hd + j] * qn[i];
    o[j] = acc;
  }
}

// Gate logits -> gates for one row (with biases already added by the caller).
static void project(const DeltaNet *dn, const Tensor *x, Tensor *q, Tensor *k, Tensor *v,
                    Tensor *ga, Tensor *gb) {
  matmul(x, &dn->Wq, q);
  matmul(x, &dn->Wk, k);
  matmul(x, &dn->Wv, v);
  matmul(x, &dn->Wa, ga);
  matmul(x, &dn->Wb, gb);
  const int H = dn->n_heads, n = x->shape[0];
  for (int t = 0; t < n; t++)
    for (int h = 0; h < H; h++) {
      ga->data[t * H + h] += dn->ba.data[h];
      gb->data[t * H + h] += dn->bb.data[h];
    }
}

static void set_rows(Tensor *t, int rows) {
  t->shape[0] = rows;
  t->size = (size_t)rows * (size_t)t->shape[1];
}

void deltanet_forward(const DeltaNet *dn, const Tensor *x, Tensor *out, DeltaNetCache *c) {
  const int n = x->shape[0], H = dn->n_heads, hd = dn->head_dim, qd = H * hd, hh = hd * hd;
  c->n = n;
  set_rows(&c->q, n); set_rows(&c->k, n); set_rows(&c->v, n);
  set_rows(&c->ga, n); set_rows(&c->gb, n); set_rows(&c->o, n);
  project(dn, x, &c->q, &c->k, &c->v, &c->ga, &c->gb);
  memset(c->S, 0, (size_t)H * hh * sizeof(float));   // S_0 = 0
  for (int t = 0; t < n; t++)
    for (int h = 0; h < H; h++) {
      const int r = t * qd + h * hd, g = t * H + h;
      c->qnorm[g] = l2n(&c->q.data[r], &c->qn[r], hd);
      c->knorm[g] = l2n(&c->k.data[r], &c->kn[r], hd);
      c->alpha[g] = sigmoidf_(c->ga.data[g]);
      c->beta[g] = sigmoidf_(c->gb.data[g]);
      dn_head_token(&c->S[((size_t)t * H + h) * hh], &c->S[((size_t)(t + 1) * H + h) * hh],
                    &c->qn[r], &c->kn[r], &c->v.data[r], c->alpha[g], c->beta[g],
                    &c->o.data[r], hd);
    }
  matmul(&c->o, &dn->Wo, out);
}

void deltanet_step(const DeltaNet *dn, float *state, const Tensor *x, Tensor *out) {
  const int H = dn->n_heads, hd = dn->head_dim, qd = H * hd, hh = hd * hd;
  Tensor q = mk(1, qd), k = mk(1, qd), v = mk(1, qd), ga = mk(1, H), gb = mk(1, H), o = mk(1, qd);
  project(dn, x, &q, &k, &v, &ga, &gb);
  float *qn = (float*)malloc((size_t)hd * sizeof(float));
  float *kn = (float*)malloc((size_t)hd * sizeof(float));
  float *Sn = (float*)malloc((size_t)hh * sizeof(float));
  for (int h = 0; h < H; h++) {
    l2n(&q.data[h * hd], qn, hd);
    l2n(&k.data[h * hd], kn, hd);
    dn_head_token(&state[(size_t)h * hh], Sn, qn, kn, &v.data[h * hd],
                  sigmoidf_(ga.data[h]), sigmoidf_(gb.data[h]), &o.data[h * hd], hd);
    memcpy(&state[(size_t)h * hh], Sn, (size_t)hh * sizeof(float));
  }
  matmul(&o, &dn->Wo, out);
  free(qn); free(kn); free(Sn);
  tensor_free(&q); tensor_free(&k); tensor_free(&v);
  tensor_free(&ga); tensor_free(&gb); tensor_free(&o);
}

// d(x/|x|) back to dx: dx = (dxn - xn (xn . dxn)) / |x|.
static void l2n_backward(const float *xn, float nr, const float *dxn, float *dx, int n) {
  float dot = 0.0f;
  for (int i = 0; i < n; i++) dot += xn[i] * dxn[i];
  for (int i = 0; i < n; i++) dx[i] += (dxn[i] - xn[i] * dot) / nr;
}

void deltanet_backward(DeltaNet *dn, const Tensor *x, const Tensor *out, DeltaNetCache *c) {
  const int n = c->n, H = dn->n_heads, hd = dn->head_dim, qd = H * hd, hh = hd * hd;
  tensor_zero_grad(&c->o); tensor_zero_grad(&c->q); tensor_zero_grad(&c->k);
  tensor_zero_grad(&c->v); tensor_zero_grad(&c->ga); tensor_zero_grad(&c->gb);
  matmul_backward(&c->o, &dn->Wo, out);               // dO, dWo

  float *G = (float*)malloc((size_t)hh * sizeof(float));
  float *u = (float*)malloc((size_t)hd * sizeof(float));
  float *w = (float*)malloc((size_t)hd * sizeof(float));
  float *dqn = (float*)malloc((size_t)hd * sizeof(float));
  float *dkn = (float*)malloc((size_t)hd * sizeof(float));
  for (int h = 0; h < H; h++) {
    memset(G, 0, (size_t)hh * sizeof(float));         // dL/dS_t, carried backward in time
    for (int t = n - 1; t >= 0; t--) {
      const int r = t * qd + h * hd, g = t * H + h;
      const float *Sp = &c->S[((size_t)t * H + h) * hh];
      const float *Sn = &c->S[((size_t)(t + 1) * H + h) * hh];
      const float *qn = &c->qn[r], *kn = &c->kn[r], *v = &c->v.data[r];
      const float *dO = &c->o.grad[r];
      const float a = c->alpha[g], b = c->beta[g];
      // o = S_t^T q: dq = S_t dO, and S_t collects q dO^T.
      for (int i = 0; i < hd; i++) {
        float acc = 0.0f;
        for (int j = 0; j < hd; j++) {
          acc += Sn[i * hd + j] * dO[j];
          G[i * hd + j] += qn[i] * dO[j];
        }
        dqn[i] = acc;
      }
      // S_t = a*S' + k w^T with w_j = b*(v_j - a*u_j), u = k^T S'.
      for (int j = 0; j < hd; j++) {
        float uj = 0.0f, wj = 0.0f;
        for (int i = 0; i < hd; i++) { uj += kn[i] * Sp[i * hd + j]; wj += G[i * hd + j] * kn[i]; }
        u[j] = uj; w[j] = wj;                         // w here = (G^T k)_j
      }
      float da = 0.0f, db = 0.0f;
      for (int i = 0; i < hd; i++) {
        float dk = 0.0f, sw = 0.0f;
        for (int j = 0; j < hd; j++) {
          const float Gij = G[i * hd + j];
          da += Gij * (Sp[i * hd + j] - b * kn[i] * u[j]);
          db += Gij * kn[i] * (v[j] - a * u[j]);
          dk += Gij * (v[j] - a * u[j]);
          sw += Sp[i * hd + j] * w[j];
        }
        dkn[i] = b * dk - a * b * sw;
      }
      for (int j = 0; j < hd; j++) c->v.grad[r + j] += b * w[j];
      l2n_backward(qn, c->qnorm[g], dqn, &c->q.grad[r], hd);
      l2n_backward(kn, c->knorm[g], dkn, &c->k.grad[r], hd);
      c->ga.grad[g] += da * a * (1.0f - a);
      c->gb.grad[g] += db * b * (1.0f - b);
      // dS' = a * (G - b * k w^T): pass the state gradient one step back.
      for (int i = 0; i < hd; i++)
        for (int j = 0; j < hd; j++)
          G[i * hd + j] = a * (G[i * hd + j] - b * kn[i] * w[j]);
    }
  }
  free(G); free(u); free(w); free(dqn); free(dkn);

  for (int t = 0; t < n; t++)
    for (int h = 0; h < H; h++) {
      dn->ba.grad[h] += c->ga.grad[t * H + h];
      dn->bb.grad[h] += c->gb.grad[t * H + h];
    }
  matmul_backward(x, &dn->Wq, &c->q);
  matmul_backward(x, &dn->Wk, &c->k);
  matmul_backward(x, &dn->Wv, &c->v);
  matmul_backward(x, &dn->Wa, &c->ga);
  matmul_backward(x, &dn->Wb, &c->gb);
}
