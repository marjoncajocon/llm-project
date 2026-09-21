// muon.c - Muon optimizer (Newton-Schulz orthogonalized momentum) + AdamW fallback
#include "muon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix.h"

// Quintic Newton-Schulz coefficients (Keller Jordan). Chosen to maximize the
// slope at 0 (so tiny singular values grow fast) while keeping every value in a
// band around 1 after 5 steps — not to converge exactly.
#define NS_A 3.4445f
#define NS_B (-4.7750f)
#define NS_C 2.0315f

static void *xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (!p) { fprintf(stderr, "Error: Muon allocation failed (%zu bytes)\n", n); exit(EXIT_FAILURE); }
  return p;
}

// A 2-D, grad-less view over a caller-owned buffer, so scratch arrays can go
// through the ordinary (SIMD / threaded) matmul without a tensor_create each.
static Tensor view2(float *data, int *shape, int r, int c) {
  Tensor t;
  shape[0] = r; shape[1] = c;
  t.data = data; t.grad = NULL; t.shape = shape; t.ndim = 2;
  t.size = (size_t)r * (size_t)c;
  return t;
}

void muon_newton_schulz(const float *G, int rows, int cols, float *O, int steps) {
  // Work on X of shape (r x c) with r <= c, so the Gram matrix A = X X^T is the
  // small (r x r) one: the whole iteration then costs O(r^2 c), not O(r c^2).
  const int tr = rows > cols;
  const int r = tr ? cols : rows, c = tr ? rows : cols;
  const size_t rc = (size_t)r * c, rr = (size_t)r * r;
  float *X = (float*)xmalloc(rc * sizeof(float));
  float *XT = (float*)xmalloc(rc * sizeof(float));
  float *A = (float*)xmalloc(rr * sizeof(float));
  float *AA = (float*)xmalloc(rr * sizeof(float));
  float *BX = (float*)xmalloc(rc * sizeof(float));

  // X = G (or G^T) / ||G||_F. Frobenius >= spectral norm, so every singular
  // value starts in [0, 1] — inside the iteration's basin.
  double ss = 0.0;
  for (size_t i = 0; i < rc; i++) ss += (double)G[i] * G[i];
  const float inv = (float)(1.0 / (sqrt(ss) + 1e-7));
  if (tr) {
    for (int i = 0; i < rows; i++)             // G is (rows x cols) = (c x r)
      for (int j = 0; j < cols; j++) X[(size_t)j * c + i] = G[(size_t)i * cols + j] * inv;
  } else {
    for (size_t i = 0; i < rc; i++) X[i] = G[i] * inv;
  }

  int sx[2], sxt[2], sa[2], saa[2], sbx[2];
  Tensor tX = view2(X, sx, r, c), tXT = view2(XT, sxt, c, r);
  Tensor tA = view2(A, sa, r, r), tAA = view2(AA, saa, r, r), tBX = view2(BX, sbx, r, c);
  for (int s = 0; s < steps; s++) {
    for (int i = 0; i < r; i++)                // XT = X^T
      for (int j = 0; j < c; j++) XT[(size_t)j * r + i] = X[(size_t)i * c + j];
    matmul(&tX, &tXT, &tA);                    // A  = X X^T        (r x r)
    matmul(&tA, &tA, &tAA);                    // AA = A A
    for (size_t i = 0; i < rr; i++) A[i] = NS_B * A[i] + NS_C * AA[i];   // B = bA + cA^2
    matmul(&tA, &tX, &tBX);                    // BX = B X          (r x c)
    for (size_t i = 0; i < rc; i++) X[i] = NS_A * X[i] + BX[i];
  }

  if (tr) {
    for (int j = 0; j < r; j++)                // back to (rows x cols)
      for (int i = 0; i < c; i++) O[(size_t)i * cols + j] = X[(size_t)j * c + i];
  } else {
    memcpy(O, X, rc * sizeof(float));
  }
  free(X); free(XT); free(A); free(AA); free(BX);
}

int muon_route(Tensor **params, int n_params, Tensor *const *exclude, int n_exclude,
               int *mask) {
  int n = 0;
  for (int i = 0; i < n_params; i++) {
    int m = (params[i]->ndim == 2);
    for (int e = 0; e < n_exclude && m; e++)
      if (params[i] == exclude[e]) m = 0;
    mask[i] = m;
    n += m;
  }
  return n;
}

// The AdamW group, in canonical order (the order its m/v buffers were laid out in).
static int adam_sublist(const Muon *o, Tensor **params) {
  int n = 0;
  for (int i = 0; i < o->n_params; i++)
    if (!o->is_muon[i]) o->sub[n++] = params[i];
  return n;
}

Muon muon_create(Tensor **params, int n_params, const int *mask, float lr,
                 float weight_decay) {
  Muon o;
  o.n_params = n_params;
  o.is_muon = (int*)xmalloc((size_t)n_params * sizeof(int));
  o.mom = (float**)xmalloc((size_t)n_params * sizeof(float*));
  o.sub = (Tensor**)xmalloc((size_t)n_params * sizeof(Tensor*));
  for (int i = 0; i < n_params; i++) {
    o.is_muon[i] = mask[i] ? 1 : 0;
    o.mom[i] = NULL;
    if (!o.is_muon[i]) continue;
    if (params[i]->ndim != 2) {
      fprintf(stderr, "Error: Muon param %d is %d-D (Muon needs matrices)\n", i, params[i]->ndim);
      exit(EXIT_FAILURE);
    }
    o.mom[i] = (float*)xmalloc(params[i]->size * sizeof(float));
    memset(o.mom[i], 0, params[i]->size * sizeof(float));
  }
  o.n_adam = adam_sublist(&o, params);
  if (o.n_adam > 0) {
    o.adam = adamw_create(o.sub, o.n_adam, lr, weight_decay);
  } else {                     // all-Muon (e.g. LoRA): no moments; malloc(0) may be NULL
    memset(&o.adam, 0, sizeof(o.adam));
    o.adam.lr = lr;
  }
  o.lr = lr;
  o.momentum = 0.95f;
  o.nesterov = 1;
  o.ns_steps = 5;
  o.weight_decay = weight_decay;
  return o;
}

void muon_set_lr(Muon *o, float lr) {
  o->lr = lr;
  adam_set_lr(&o->adam, lr);
}

void muon_step(Muon *o, Tensor **params, int n_params) {
  if (n_params != o->n_params) {
    fprintf(stderr, "Error: muon_step got %d params, created for %d\n", n_params, o->n_params);
    exit(EXIT_FAILURE);
  }
  // Largest Muon matrix sizes the one update buffer shared by all of them.
  size_t maxsz = 0;
  for (int i = 0; i < n_params; i++)
    if (o->is_muon[i] && params[i]->size > maxsz) maxsz = params[i]->size;
  float *U = maxsz ? (float*)xmalloc(maxsz * sizeof(float)) : NULL;

  const float mu = o->momentum;
  for (int p = 0; p < n_params; p++) {
    if (!o->is_muon[p]) continue;
    Tensor *w = params[p];
    const int rows = w->shape[0], cols = w->shape[1];
    float *M = o->mom[p];
    for (size_t i = 0; i < w->size; i++) {
      const float g = w->grad[i];
      M[i] = mu * M[i] + g;
      U[i] = o->nesterov ? g + mu * M[i] : M[i];
    }
    muon_newton_schulz(U, rows, cols, U, o->ns_steps);
    // Moonshot RMS matching: orthogonal => element RMS 1/sqrt(max(r,c)); scaling
    // by 0.2*sqrt(max(r,c)) lands the update at RMS ~0.2, AdamW's typical size.
    const float scale = 0.2f * sqrtf((float)(rows > cols ? rows : cols));
    const float decay = 1.0f - o->lr * o->weight_decay;
    for (size_t i = 0; i < w->size; i++)
      w->data[i] = w->data[i] * decay - o->lr * scale * U[i];
  }
  free(U);

  if (o->n_adam > 0) {
    adam_sublist(o, params);
    adam_step(&o->adam, o->sub, o->n_adam);
  }
}

void muon_free(Muon *o) {
  for (int i = 0; i < o->n_params; i++) free(o->mom[i]);
  free(o->mom); free(o->is_muon); free(o->sub);
  adam_free(&o->adam);
  o->mom = NULL; o->is_muon = NULL; o->sub = NULL;
  o->n_params = o->n_adam = 0;
}
