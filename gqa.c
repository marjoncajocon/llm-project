// gqa.c - Grouped-Query Attention (causal, LLaMA-family)
#include "gqa.h"

#include <math.h>
#include <stdlib.h>

GqaCache gqa_cache_create(int n, int n_heads, int n_kv, int head_dim) {
  GqaCache c;
  c.n = n; c.n_heads = n_heads; c.n_kv = n_kv; c.head_dim = head_dim;
  c.att = (float*)calloc((size_t)n_heads * n * n, sizeof(float));
  return c;
}

void gqa_cache_free(GqaCache *c) { free(c->att); c->att = NULL; }

void gqa_forward(const Tensor *Q, const Tensor *K, const Tensor *V, GqaCache *c, Tensor *O) {
  const int n = c->n, H = c->n_heads, KV = c->n_kv, hd = c->head_dim;
  const int group = H / KV;             // query heads per KV head
  const int qd = H * hd;                // Q / O row stride
  const int kd = KV * hd;               // K / V row stride
  const float scale = 1.0f / sqrtf((float)hd);

  for (int h = 0; h < H; h++) {
    const int kv = h / group;
    const int qoff = h * hd, koff = kv * hd;
    for (int i = 0; i < n; i++) {
      float *att = &c->att[((size_t)h * n + i) * n];
      // scores over j = 0..i (causal), softmax
      float maxs = -INFINITY;
      for (int j = 0; j <= i; j++) {
        float dot = 0.0f;
        for (int d = 0; d < hd; d++) dot += Q->data[i * qd + qoff + d] * K->data[j * kd + koff + d];
        float s = dot * scale;
        att[j] = s;
        if (s > maxs) maxs = s;
      }
      float sum = 0.0f;
      for (int j = 0; j <= i; j++) { float e = expf(att[j] - maxs); att[j] = e; sum += e; }
      float inv = 1.0f / sum;
      for (int j = 0; j <= i; j++) att[j] *= inv;
      for (int j = i + 1; j < n; j++) att[j] = 0.0f;
      // output O_h[i] = sum_j att[j] * V_kv[j]
      for (int d = 0; d < hd; d++) {
        float o = 0.0f;
        for (int j = 0; j <= i; j++) o += att[j] * V->data[j * kd + koff + d];
        O->data[i * qd + qoff + d] = o;
      }
    }
  }
}

void gqa_backward(const Tensor *Q, const Tensor *K, const Tensor *V,
                  const GqaCache *c, const Tensor *O) {
  const int n = c->n, H = c->n_heads, KV = c->n_kv, hd = c->head_dim;
  const int group = H / KV;
  const int qd = H * hd, kd = KV * hd;
  const float scale = 1.0f / sqrtf((float)hd);
  float *datt = (float*)malloc((size_t)n * sizeof(float));

  for (int h = 0; h < H; h++) {
    const int kv = h / group;
    const int qoff = h * hd, koff = kv * hd;
    for (int i = 0; i < n; i++) {
      const float *att = &c->att[((size_t)h * n + i) * n];
      const float *dO = &O->grad[i * qd + qoff];

      // dV_kv[j] += att[j] * dO  ;  datt[j] = <dO, V_kv[j]>
      float dot_ad = 0.0f;
      for (int j = 0; j <= i; j++) {
        float dj = 0.0f;
        for (int d = 0; d < hd; d++) {
          dj += dO[d] * V->data[j * kd + koff + d];
          V->grad[j * kd + koff + d] += att[j] * dO[d];
        }
        datt[j] = dj;
        dot_ad += att[j] * dj;
      }
      // softmax backward: ds[j] = att[j] * (datt[j] - sum_k att[k] datt[k])
      // score s[j] = scale * <Q_i, K_j>  =>  dQ_i += ds*scale*K_j ; dK_j += ds*scale*Q_i
      for (int j = 0; j <= i; j++) {
        float ds = att[j] * (datt[j] - dot_ad);
        float dss = ds * scale;
        for (int d = 0; d < hd; d++) {
          Q->grad[i * qd + qoff + d] += dss * K->data[j * kd + koff + d];
          K->grad[j * kd + koff + d] += dss * Q->data[i * qd + qoff + d];
        }
      }
    }
  }
  free(datt);
}
