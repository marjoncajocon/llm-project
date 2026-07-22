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
