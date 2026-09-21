// gqa.c - Grouped-Query Attention (causal, optional sliding window, LLaMA-family)
#include "gqa.h"

#include <math.h>
#include <stdlib.h>

#include "matrix.h"   // tf_parallel_for

GqaCache gqa_cache_create(int n, int n_heads, int n_kv, int head_dim) {
  GqaCache c;
  c.n = n; c.n_heads = n_heads; c.n_kv = n_kv; c.head_dim = head_dim;
  c.window = 0;                         // full causal attention unless the caller sets one
  c.att = (float*)calloc((size_t)n_heads * n * n, sizeof(float));
  return c;
}

void gqa_cache_free(GqaCache *c) { free(c->att); c->att = NULL; }

// ---------------------------------------------------------------------------
// Layout and threading (bit-exact with the original scalar loops, selftest [43]):
//  * The q.k scores are computed as an AXPY over the keys j against a
//    TRANSPOSED copy of K (Kt[d][j]): each score is still ONE sum over d
//    ascending from 0, just eight keys at a time in vector lanes instead of one
//    64-long dependency chain. Same for <dO, V_j> in the backward (Vt).
//  * O = att @ V is an AXPY over contiguous V rows (j outer, d inner) instead of
//    a strided d-outer / j-inner walk; each O element still sums j ascending.
//  * The forward is threaded over QUERY HEADS (a head owns its att slice and its
//    O columns); the backward over KV HEADS (a KV group owns its dK/dV columns
//    and its query heads' dQ columns), visiting the group's heads in ascending
//    order -- exactly the order the serial loop adds them in. No races, no
//    reordered sums, so every thread count gives the same bits.
// ---------------------------------------------------------------------------

// X (n, KV*hd) -> Xt[(kv*hd + d)*n + j] = X[j, kv*hd + d]
static float *transpose_heads(const float *X, int n, int kd) {
  float *t = (float*)malloc((size_t)n * kd * sizeof(float));
  if (!t) return NULL;
  for (int j = 0; j < n; j++)
    for (int c = 0; c < kd; c++) t[(size_t)c * n + j] = X[(size_t)j * kd + c];
  return t;
}

// Below this much work (n^2 * H * hd) the pool's hand-off costs more than it saves.
#define GQA_THREAD_MIN (1L << 16)

typedef struct {
  const Tensor *Q, *K, *V; const GqaCache *c; const Tensor *O;
  const float *Kt, *Vt;       // transposed K (forward) / V (backward), may be NULL
} GqaJob;

// Scores s[j] = sum_d q[d] * K[j, d] for j in [lo, i], d ascending from 0.
static void qk_scores(const float *q, const float *K, const float *kt, int kd, int n,
                      int hd, int lo, int i, float *att) {
  if (kt) {
    for (int j = lo; j <= i; j++) att[j] = 0.0f;
    for (int d = 0; d < hd; d++) {
      const float qd = q[d];
      const float *kr = kt + (size_t)d * n;
      for (int j = lo; j <= i; j++) att[j] += qd * kr[j];
    }
  } else {                                  // no transposed copy (OOM): the plain loop
    for (int j = lo; j <= i; j++) {
      float dot = 0.0f;
      for (int d = 0; d < hd; d++) dot += q[d] * K[(size_t)j * kd + d];
      att[j] = dot;
    }
  }
}

static void gqa_fwd_heads(int h0, int h1, void *ctx) {
  const GqaJob *t = (const GqaJob*)ctx;
  const GqaCache *c = t->c;
  const int n = c->n, H = c->n_heads, KV = c->n_kv, hd = c->head_dim;
  const int group = H / KV;             // query heads per KV head
  const int qd = H * hd;                // Q / O row stride
  const int kd = KV * hd;               // K / V row stride
  const float scale = 1.0f / sqrtf((float)hd);
  const float *Qd = t->Q->data, *Kd = t->K->data, *Vd = t->V->data;
  float *Od = t->O->data;

  for (int h = h0; h < h1; h++) {
    const int kv = h / group;
    const int qoff = h * hd, koff = kv * hd;
    const float *kt = t->Kt ? t->Kt + (size_t)koff * n : NULL;
    for (int i = 0; i < n; i++) {
      float *att = &c->att[((size_t)h * n + i) * n];
      const int lo = gqa_window_lo(i, c->window);   // 0 unless a sliding window cuts in
      // scores over j = lo..i (causal + window), softmax
      for (int j = 0; j < lo; j++) att[j] = 0.0f;   // outside the window: masked
      qk_scores(&Qd[(size_t)i * qd + qoff], &Kd[koff], kt, kd, n, hd, lo, i, att);
      float maxs = -INFINITY;
      for (int j = lo; j <= i; j++) {
        float s = att[j] * scale;
        att[j] = s;
        if (s > maxs) maxs = s;
      }
      float sum = 0.0f;
      for (int j = lo; j <= i; j++) { float e = expf(att[j] - maxs); att[j] = e; sum += e; }
      float inv = 1.0f / sum;
      for (int j = lo; j <= i; j++) att[j] *= inv;
      for (int j = i + 1; j < n; j++) att[j] = 0.0f;
      // output O_h[i] = sum_j att[j] * V_kv[j]   (AXPY over contiguous V rows)
      float *o = &Od[(size_t)i * qd + qoff];
      for (int d = 0; d < hd; d++) o[d] = 0.0f;
      for (int j = lo; j <= i; j++) {
        const float a = att[j];
        const float *vr = &Vd[(size_t)j * kd + koff];
        for (int d = 0; d < hd; d++) o[d] += a * vr[d];
      }
    }
  }
}

void gqa_forward(const Tensor *Q, const Tensor *K, const Tensor *V, GqaCache *c, Tensor *O) {
  const int n = c->n, H = c->n_heads, kd = c->n_kv * c->head_dim;
  GqaJob t = { Q, K, V, c, O, transpose_heads(K->data, n, kd), NULL };
  if ((size_t)n * n * H * c->head_dim < (size_t)GQA_THREAD_MIN) gqa_fwd_heads(0, H, &t);
  else tf_parallel_for(H, gqa_fwd_heads, &t);
  free((void*)t.Kt);
}

// Copy the (rows x w) column block at src (row stride ld) into dst (row stride
// w), or back (to_src).
static void block_copy(float *dst, float *src, int rows, int w, int ld, int to_src) {
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < w; c++) {
      if (to_src) src[(size_t)r * ld + c] = dst[(size_t)r * w + c];
      else        dst[(size_t)r * w + c] = src[(size_t)r * ld + c];
    }
}

static void gqa_bwd_groups(int g0, int g1, void *ctx) {
  const GqaJob *t = (const GqaJob*)ctx;
  const GqaCache *c = t->c;
  const int n = c->n, H = c->n_heads, KV = c->n_kv, hd = c->head_dim;
  const int group = H / KV;
  const int qd = H * hd, kd = KV * hd;
  const float scale = 1.0f / sqrtf((float)hd);
  const float *Qd = t->Q->data, *Kd = t->K->data, *Vd = t->V->data;
  float *Qg = t->Q->grad, *Kg = t->K->grad, *Vg = t->V->grad;
  float *datt = (float*)malloc((size_t)n * sizeof(float));
  // The group's dQ / dK / dV columns are accumulated in PRIVATE packed buffers
  // (copied in, updated, copied back -- the same additions in the same order,
  // just at another address). Written in place, three threads update
  // neighbouring 256-byte column blocks of the same rows for every (i, j), and
  // the hardware prefetchers pull the neighbours' lines across cores: measured
  // SLOWER than one thread (n = 320: 11.5 ms at 3 threads vs 8.4 ms at 1).
  float *pk = (float*)malloc((size_t)n * hd * sizeof(float));
  float *pv = (float*)malloc((size_t)n * hd * sizeof(float));
  float *pq = (float*)malloc((size_t)group * n * hd * sizeof(float));
  const int packed = pk && pv && pq;

  for (int kv = g0; kv < g1; kv++) {
    const int koff = kv * hd;
    const float *vt = t->Vt ? t->Vt + (size_t)koff * n : NULL;
    // accumulator views: row j of dK/dV at kg0 + j*ks, row i of head hh's dQ at
    // qg0 + hh*qh + i*qs
    float *kg0 = Kg + koff, *vg0 = Vg + koff, *qg0 = Qg + (size_t)kv * group * hd;
    size_t ks = (size_t)kd, qs = (size_t)qd, qh = (size_t)hd;
    if (packed) {
      block_copy(pk, kg0, n, hd, kd, 0);
      block_copy(pv, vg0, n, hd, kd, 0);
      for (int hh = 0; hh < group; hh++)
        block_copy(pq + (size_t)hh * n * hd, qg0 + (size_t)hh * hd, n, hd, qd, 0);
      kg0 = pk; vg0 = pv; qg0 = pq;
      ks = (size_t)hd; qs = (size_t)hd; qh = (size_t)n * hd;
    }
    for (int hh = 0; hh < group; hh++) {
      const int h = kv * group + hh;
      const int qoff = h * hd;
      for (int i = 0; i < n; i++) {
        const float *att = &c->att[((size_t)h * n + i) * n];
        const float *dO = &t->O->grad[(size_t)i * qd + qoff];
        const int lo = gqa_window_lo(i, c->window);   // masked keys get no gradient

        // datt[j] = <dO, V_kv[j]>  ;  dV_kv[j] += att[j] * dO
        qk_scores(dO, &Vd[koff], vt, kd, n, hd, lo, i, datt);
        float dot_ad = 0.0f;
        for (int j = lo; j <= i; j++) {
          const float a = att[j];
          float *vg = vg0 + (size_t)j * ks;
          for (int d = 0; d < hd; d++) vg[d] += a * dO[d];
          dot_ad += a * datt[j];
        }
        // softmax backward: ds[j] = att[j] * (datt[j] - sum_k att[k] datt[k])
        // score s[j] = scale * <Q_i, K_j>  =>  dQ_i += ds*scale*K_j ; dK_j += ds*scale*Q_i
        float *qg = qg0 + (size_t)hh * qh + (size_t)i * qs;
        const float *qi = &Qd[(size_t)i * qd + qoff];
        for (int j = lo; j <= i; j++) {
          float ds = att[j] * (datt[j] - dot_ad);
          float dss = ds * scale;
          const float *kj = &Kd[(size_t)j * kd + koff];
          float *kg = kg0 + (size_t)j * ks;
          for (int d = 0; d < hd; d++) {
            qg[d] += dss * kj[d];
            kg[d] += dss * qi[d];
          }
        }
      }
    }
    if (packed) {
      block_copy(pk, Kg + koff, n, hd, kd, 1);
      block_copy(pv, Vg + koff, n, hd, kd, 1);
      for (int hh = 0; hh < group; hh++)
        block_copy(pq + (size_t)hh * n * hd, Qg + (size_t)(kv * group + hh) * hd, n, hd, qd, 1);
    }
  }
  free(pk); free(pv); free(pq);
  free(datt);
}

void gqa_backward(const Tensor *Q, const Tensor *K, const Tensor *V,
                  const GqaCache *c, const Tensor *O) {
  const int n = c->n, KV = c->n_kv, kd = KV * c->head_dim;
  GqaJob t = { Q, K, V, c, O, NULL, transpose_heads(V->data, n, kd) };
  if ((size_t)n * n * c->n_heads * c->head_dim < (size_t)GQA_THREAD_MIN) gqa_bwd_groups(0, KV, &t);
  else tf_parallel_for(KV, gqa_bwd_groups, &t);
  free((void*)t.Vt);
}
