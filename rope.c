// rope.c - Rotary Position Embeddings (RoPE, LLaMA-family)
#include "rope.h"

#include <math.h>
#include <stdlib.h>

#include "matrix.h"   // tf_parallel_for

// y = rotation(pos) applied to each row. Half-split convention (see rope.h).
void rope(const Tensor *input, Tensor *output, float base) {
  const int rows = input->shape[0];
  const int cols = input->shape[1];
  const int h = cols / 2;

  for (int i = 0; i < rows; i++) {            // row index i = position
    const float *x = &input->data[i * cols];
    float *y = &output->data[i * cols];
    for (int k = 0; k < h; k++) {
      float inv_freq = powf(base, -2.0f * (float)k / (float)cols);
      float ang = (float)i * inv_freq;
      float c = cosf(ang), s = sinf(ang);
      float x1 = x[k], x2 = x[k + h];
      y[k]     = x1 * c - x2 * s;
      y[k + h] = x2 * c + x1 * s;
    }
  }
}

// The per-pair forward is y = R(θ) x with R = [[c,-s],[s,c]]; backward is
// dx = R(θ)^T dy = [[c,s],[-s,c]] dy. Recomputes θ per (position, pair).
void rope_backward(const Tensor *input, const Tensor *output, float base) {
  const int rows = input->shape[0];
  const int cols = input->shape[1];
  const int h = cols / 2;

  for (int i = 0; i < rows; i++) {
    const float *dy = &output->grad[i * cols];
    float *dx = &input->grad[i * cols];
    for (int k = 0; k < h; k++) {
      float inv_freq = powf(base, -2.0f * (float)k / (float)cols);
      float ang = (float)i * inv_freq;
      float c = cosf(ang), s = sinf(ang);
      dx[k]     += dy[k] * c + dy[k + h] * s;
      dx[k + h] += -dy[k] * s + dy[k + h] * c;
    }
  }
}

// In-place, per-head. x: (seq, n_heads*head_dim); row i uses position pos0+i.
// Rotating (x1,x2) into locals before writing makes the in-place update safe.
//
// The angle of pair k at position pos is the same for every head, so each row
// computes its cos/sin table ONCE (the identical powf/cosf/sinf expressions, so
// the identical values) and all heads reuse it -- n_heads x fewer
// transcendental calls. Rows are independent, so they are split across the
// thread pool; every element is still produced by the same arithmetic.
typedef struct { Tensor *x; int n_heads; float base; int pos0, backward; } RopeJob;

#define ROPE_MAX_HALF 256          // head_dim/2 above this uses a heap table

static void rope_rows(int r0, int r1, void *ctx) {
  const RopeJob *t = (const RopeJob*)ctx;
  const int cols = t->x->shape[1];
  const int hd = cols / t->n_heads;   // per-head dim
  const int h = hd / 2;
  float stack_cs[2 * ROPE_MAX_HALF];
  float *cs = (h <= ROPE_MAX_HALF) ? stack_cs : (float*)malloc((size_t)2 * h * sizeof(float));
  if (!cs) return;
  float *cc = cs, *ss = cs + h;

  for (int i = r0; i < r1; i++) {
    const int pos = t->pos0 + i;
    for (int k = 0; k < h; k++) {
      float inv_freq = powf(t->base, -2.0f * (float)k / (float)hd);
      float ang = (float)pos * inv_freq;
      cc[k] = cosf(ang); ss[k] = sinf(ang);
    }
    for (int head = 0; head < t->n_heads; head++) {
      if (!t->backward) {
        float *xh = &t->x->data[(size_t)i * cols + head * hd];
        for (int k = 0; k < h; k++) {
          float c = cc[k], s = ss[k];
          float x1 = xh[k], x2 = xh[k + h];
          xh[k]     = x1 * c - x2 * s;
          xh[k + h] = x2 * c + x1 * s;
        }
      } else {
        float *gh = &t->x->grad[(size_t)i * cols + head * hd];
        for (int k = 0; k < h; k++) {
          float c = cc[k], s = ss[k];
          float g1 = gh[k], g2 = gh[k + h];
          gh[k]     =  g1 * c + g2 * s;
          gh[k + h] = -g1 * s + g2 * c;
        }
      }
    }
  }
  if (cs != stack_cs) free(cs);
}

// Below this many elements a row split is not worth the pool hand-off.
#define ROPE_THREAD_MIN (1L << 14)

static void rope_run(Tensor *x, int n_heads, float base, int pos0, int backward) {
  RopeJob t = { x, n_heads, base, pos0, backward };
  const int rows = x->shape[0];
  if ((size_t)rows * (size_t)x->shape[1] < (size_t)ROPE_THREAD_MIN) rope_rows(0, rows, &t);
  else tf_parallel_for(rows, rope_rows, &t);
}

void rope_heads_pos(Tensor *x, int n_heads, float base, int pos0) {
  rope_run(x, n_heads, base, pos0, 0);
}

void rope_heads(Tensor *x, int n_heads, float base) {
  rope_heads_pos(x, n_heads, base, 0);
}

// Mirror of rope_heads_pos on the gradient: the rotation is orthogonal, so the
// backward is the transposed (inverse) rotation of the same angle. Applied in
// place, for the same reason the forward is.
void rope_heads_backward_pos(Tensor *x, int n_heads, float base, int pos0) {
  rope_run(x, n_heads, base, pos0, 1);
}

void rope_heads_backward(Tensor *x, int n_heads, float base) {
  rope_heads_backward_pos(x, n_heads, base, 0);
}
