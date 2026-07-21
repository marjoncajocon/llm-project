// imagegen.c - Image <-> token codec for generation (color-cube quantization)
#include "imagegen.h"

#include <stdlib.h>

int imagegen_vocab(int levels) {
  return levels * levels * levels;
}

// Quantize a value in [0,1] to a level index in [0, levels).
static int quantize(float v, int levels) {
  int q = (int)(v * levels);
  if (q < 0) q = 0;
  if (q > levels - 1) q = levels - 1;
  return q;
}

void image_to_tokens(const Image *im, int levels, int *tokens) {
  const int n = im->h * im->w;
  for (int i = 0; i < n; i++) {
    int r = quantize(im->data[i * 3 + 0], levels);
    int g = quantize(im->data[i * 3 + 1], levels);
    int b = quantize(im->data[i * 3 + 2], levels);
    tokens[i] = (r * levels + g) * levels + b;
  }
}

Image tokens_to_image(const int *tokens, int w, int h, int levels) {
  Image im;
  im.h = h; im.w = w; im.c = 3;
  im.data = (float*)malloc((size_t)w * h * 3 * sizeof(float));
  const float scale = (levels > 1) ? 1.0f / (float)(levels - 1) : 0.0f;
  for (int i = 0; i < w * h; i++) {
    int t = tokens[i];
    int b = t % levels;
    int g = (t / levels) % levels;
    int r = (t / (levels * levels)) % levels;
    im.data[i * 3 + 0] = r * scale;
    im.data[i * 3 + 1] = g * scale;
    im.data[i * 3 + 2] = b * scale;
  }
  return im;
}
