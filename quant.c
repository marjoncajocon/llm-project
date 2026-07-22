// quant.c - int8 weight quantization (per-output-channel, symmetric)
#include "quant.h"

#include <math.h>

void quantize_int8_percol(const float *w, int rows, int cols, signed char *q, float *scale) {
  for (int j = 0; j < cols; j++) {
    float mx = 0.0f;
    for (int i = 0; i < rows; i++) {
      float a = fabsf(w[i * cols + j]);
      if (a > mx) mx = a;
    }
    float s = (mx > 0.0f) ? mx / 127.0f : 1.0f;
    scale[j] = s;
    float inv = 1.0f / s;
    for (int i = 0; i < rows; i++) {
      int v = (int)lrintf(w[i * cols + j] * inv);
      if (v > 127) v = 127;
      if (v < -127) v = -127;
      q[i * cols + j] = (signed char)v;
    }
  }
}

void dequantize_int8_percol(const signed char *q, const float *scale, int rows, int cols, float *w) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      w[i * cols + j] = (float)q[i * cols + j] * scale[j];
}
