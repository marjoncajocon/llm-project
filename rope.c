// rope.c - Rotary Position Embeddings (RoPE, LLaMA-family)
#include "rope.h"

#include <math.h>

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
void rope_heads_pos(Tensor *x, int n_heads, float base, int pos0) {
  const int rows = x->shape[0];
  const int cols = x->shape[1];
  const int hd = cols / n_heads;   // per-head dim
  const int h = hd / 2;

  for (int i = 0; i < rows; i++) {
    const int pos = pos0 + i;
    for (int head = 0; head < n_heads; head++) {
      float *xh = &x->data[i * cols + head * hd];
      for (int k = 0; k < h; k++) {
        float inv_freq = powf(base, -2.0f * (float)k / (float)hd);
        float ang = (float)pos * inv_freq;
        float c = cosf(ang), s = sinf(ang);
        float x1 = xh[k], x2 = xh[k + h];
        xh[k]     = x1 * c - x2 * s;
        xh[k + h] = x2 * c + x1 * s;
      }
    }
  }
}

void rope_heads(Tensor *x, int n_heads, float base) {
  rope_heads_pos(x, n_heads, base, 0);
}
