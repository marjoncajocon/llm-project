// sampler.c - Token sampling from logits
#include "sampler.h"

#include <stdlib.h>
#include <math.h>

// Greedy sampler: return the index of the largest logit in the given row.
int sample_argmax(const Tensor *logits, int row) {
  const int vocab_size = logits->shape[1];
  const float *r = &logits->data[row * vocab_size];

  int best = 0;
  float best_val = r[0];
  for (int j = 1; j < vocab_size; j++) {
    if (r[j] > best_val) {
      best_val = r[j];
      best = j;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// RNG (xorshift64)
// ---------------------------------------------------------------------------
Rng rng_create(unsigned long long seed) {
  Rng r;
  r.state = seed ? seed : 0x9E3779B97F4A7C15ULL;  // avoid the all-zero fixed point
  return r;
}

static unsigned long long xorshift64(unsigned long long *s) {
  unsigned long long x = *s;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  *s = x;
  return x;
}

float rng_uniform(Rng *r) {
  // Top 53 bits -> a double in [0,1), then narrow to float.
  return (float)((xorshift64(&r->state) >> 11) * (1.0 / 9007199254740992.0));
}

// ---------------------------------------------------------------------------
// Temperature + top-k + top-p sampling
// ---------------------------------------------------------------------------
typedef struct {
  float p;
  int idx;
} ProbIdx;

static int cmp_prob_desc(const void *a, const void *b) {
  float pa = ((const ProbIdx*)a)->p;
  float pb = ((const ProbIdx*)b)->p;
  return (pa < pb) - (pa > pb);  // descending order
}

int sample_token(const Tensor *logits, int row, float temperature,
                 int top_k, float top_p, Rng *rng) {
  const int V = logits->shape[1];
  const float *l = &logits->data[row * V];

  if (temperature <= 0.0f) {
    return sample_argmax(logits, row);  // greedy
  }

  // Temperature-scaled, numerically stable softmax into a (prob, index) array.
  float max_l = l[0];
  for (int j = 1; j < V; j++) if (l[j] > max_l) max_l = l[j];

  ProbIdx *pi = (ProbIdx*)malloc(V * sizeof(ProbIdx));
  float sum = 0.0f;
  for (int j = 0; j < V; j++) {
    float e = expf((l[j] - max_l) / temperature);
    pi[j].p = e;
    pi[j].idx = j;
    sum += e;
  }
  for (int j = 0; j < V; j++) pi[j].p /= sum;

  // Sort by probability, descending.
  qsort(pi, V, sizeof(ProbIdx), cmp_prob_desc);

  // How many of the top tokens to keep (top-k then top-p).
  int keep = V;
  if (top_k > 0 && top_k < keep) keep = top_k;
  if (top_p > 0.0f && top_p < 1.0f) {
    float cum = 0.0f;
    int kp = 0;
    for (int i = 0; i < keep; i++) {
      cum += pi[i].p;
      kp++;
      if (cum >= top_p) break;
    }
    keep = kp;
  }
  if (keep < 1) keep = 1;

  // Renormalize the kept mass and sample from it.
  float z = 0.0f;
  for (int i = 0; i < keep; i++) z += pi[i].p;
  float u = rng_uniform(rng) * z;

  int chosen = pi[keep - 1].idx;
  float acc = 0.0f;
  for (int i = 0; i < keep; i++) {
    acc += pi[i].p;
    if (u <= acc) { chosen = pi[i].idx; break; }
  }

  free(pi);
  return chosen;
}
