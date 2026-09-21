// sampler.c - Token sampling from logits
#include "sampler.h"

#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

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
//
// FAST PATH, SAME TOKEN. Sorting all V probabilities per generated token (V =
// 49152 for SmolLM2) costs a noticeable share of a 135M decode step, yet top-k /
// top-p only ever look at the head of the sorted order. So a single pass buckets
// the probabilities by float exponent, picks the largest threshold 2^e whose
// bucket-mass (or count, for top-k) clearly covers the cut, and sorts only the
// candidates p >= 2^e. Because the comparator is a total order (ties -> lower
// id first) and the candidates are exactly the p >= threshold, their sorted
// order IS the first n entries of the full sorted order, element for element.
// The cut-off loop, the renormalizing sum z and the inverse-CDF walk then run
// over identical entries in identical order, so every float sum -- and the
// chosen token for a given RNG draw -- is bit-for-bit what the full sort gives.
// If the top-p cumulative sum does not reach top_p inside the candidates (flat
// distributions), it falls back to sorting everything. Selftest [38] checks the
// fast path against the full sort.
// ---------------------------------------------------------------------------
typedef struct {
  float p;
  int idx;
} ProbIdx;

// Descending probability; ties broken by ascending token id. That makes it a
// TOTAL order, so the sorted array does not depend on qsort's (unspecified)
// handling of equal keys -- and the head of the full sort is exactly the sort of
// a threshold-selected subset, which the fast path relies on.
static int cmp_prob_desc(const void *a, const void *b) {
  const ProbIdx *x = (const ProbIdx*)a, *y = (const ProbIdx*)b;
  if (x->p != y->p) return (x->p < y->p) - (x->p > y->p);  // descending order
  return (x->idx > y->idx) - (x->idx < y->idx);
}

// Exponent field of a non-negative float: monotone in the value, so
// "field >= e" is the threshold test p >= 2^(e-127) with no float compare.
static int exp_field(float p) {
  uint32_t u;
  memcpy(&u, &p, sizeof u);
  return (int)((u >> 23) & 0xFFu);
}

// How many of the sorted (descending) entries to keep: top-k, then top-p --
// computed from the first n of them. *decided = 0 when the answer depends on
// entries past n (only possible for n < V): top-p's cumulative sum did not
// reach top_p inside them, or top-k alone asks for more than n.
static int keep_count(const ProbIdx *pi, int n, int V, int top_k, float top_p, int *decided) {
  int keep = V;
  if (top_k > 0 && top_k < keep) keep = top_k;
  *decided = 1;
  if (top_p > 0.0f && top_p < 1.0f) {
    float cum = 0.0f;
    int kp = 0, broke = 0;
    const int lim = (keep < n) ? keep : n;
    for (int i = 0; i < lim; i++) {
      cum += pi[i].p;
      kp++;
      if (cum >= top_p) { broke = 1; break; }
    }
    if (!broke && keep > n) *decided = 0;         // the full loop would run on past n
    keep = kp;
  } else if (keep > n) {
    *decided = 0;
  }
  if (keep < 1) keep = 1;
  return keep;
}

// Move the entries with exponent field >= e_low to the front (swapping, so the
// array stays a permutation of all V entries); returns how many there are.
static int partition_above(ProbIdx *pi, int V, int e_low) {
  int n = 0;
  for (int j = 0; j < V; j++)
    if (exp_field(pi[j].p) >= e_low) {
      const ProbIdx t = pi[n]; pi[n] = pi[j]; pi[j] = t;
      n++;
    }
  return n;
}

int sample_token_debug(const Tensor *logits, int row, float temperature,
                       int top_k, float top_p, Rng *rng, int *sorted) {
  const int V = logits->shape[1];
  const float *l = &logits->data[row * V];

  if (sorted) *sorted = 0;
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

  // Candidate selection (see the header comment). Only when something truncates
  // the distribution and the probabilities are finite; else n stays V.
  const int use_k = (top_k > 0 && top_k < V), use_p = (top_p > 0.0f && top_p < 1.0f);
  int n = V;
  if ((use_k || use_p) && isfinite(sum) && sum > 0.0f) {
    int cnt[256] = {0};
    double mass[256] = {0.0};
    for (int j = 0; j < V; j++) {
      const int e = exp_field(pi[j].p);
      cnt[e]++;
      mass[e] += pi[j].p;
    }
    const int need = use_k ? top_k : V;
    int c = 0, e_low = 0;
    double ms = 0.0;
    // Stop at the first bucket where top-k is covered, or the mass exceeds top_p
    // by a margin far above any float-summation-order effect (the cut is then
    // re-checked exactly by keep_count, falling back if it was not enough).
    for (int e = 255; e >= 1; e--) {
      c += cnt[e];
      ms += mass[e];
      if (c >= need || (use_p && ms >= (double)top_p + 1e-3)) { e_low = e; break; }
    }
    if (e_low >= 1) n = partition_above(pi, V, e_low);
  }

  qsort(pi, n, sizeof(ProbIdx), cmp_prob_desc);
  int decided;
  int keep = keep_count(pi, n, V, top_k, top_p, &decided);
  if (!decided) {                            // the candidates did not cover the cut
    n = V;
    qsort(pi, V, sizeof(ProbIdx), cmp_prob_desc);
    keep = keep_count(pi, V, V, top_k, top_p, &decided);
  }
  if (sorted) *sorted = n;

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

int sample_token(const Tensor *logits, int row, float temperature,
                 int top_k, float top_p, Rng *rng) {
  return sample_token_debug(logits, row, temperature, top_k, top_p, rng, NULL);
}
