// video.c - Video = a sequence of frames (reuses the image patch front-end)
#include "video.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "patch.h"

Video video_load(const char *dir) {
  Video v = {NULL, 0};
  int cap = 8;
  v.frames = (Image*)malloc(cap * sizeof(Image));
  for (int i = 0;; i++) {
    char path[512];
    snprintf(path, sizeof(path), "%s/frame_%04d.ppm", dir, i);
    FILE *probe = fopen(path, "rb");
    if (!probe) break;                       // no more frames
    fclose(probe);
    Image im = image_load_ppm(path);
    if (!im.data) break;
    if (v.n == cap) { cap *= 2; v.frames = (Image*)realloc(v.frames, cap * sizeof(Image)); }
    v.frames[v.n++] = im;
  }
  return v;
}

void video_free(Video *v) {
  for (int i = 0; i < v->n; i++) image_free(&v->frames[i]);
  free(v->frames);
  v->frames = NULL;
  v->n = 0;
}

int video_patch_count(const Video *v, int patch) {
  if (v->n == 0) return 0;
  return v->n * patch_count(&v->frames[0], patch);
}

// Sinusoidal temporal encoding for frame f, dimension d of d_model.
static float temporal_enc(int f, int d, int dm) {
  float exponent = (float)(2 * (d / 2)) / (float)dm;
  float ang = (float)f / powf(10000.0f, exponent);
  return (d % 2 == 0) ? sinf(ang) : cosf(ang);
}

void video_embed(const Video *v, int patch, const Tensor *W, const Tensor *b, Tensor *out) {
  const int ppf = patch_count(&v->frames[0], patch);   // patches per frame
  const int dm = W->shape[1];

  int fs[2] = {ppf, dm};
  Tensor frame_emb = tensor_create(fs, 2);
  for (int f = 0; f < v->n; f++) {
    patch_embed(&v->frames[f], patch, W, b, &frame_emb);   // (ppf, dm)
    for (int p = 0; p < ppf; p++) {
      float *dst = &out->data[(f * ppf + p) * dm];
      const float *src = &frame_emb.data[p * dm];
      for (int d = 0; d < dm; d++) dst[d] = src[d] + temporal_enc(f, d, dm);
    }
  }
  tensor_free(&frame_emb);
}

void video_embed_backward(const Video *v, int patch, const Tensor *W,
                          const Tensor *b, const Tensor *out) {
  const int ppf = patch_count(&v->frames[0], patch);
  const int dm = W->shape[1];

  int fs[2] = {ppf, dm};
  Tensor frame_slice = tensor_create(fs, 2);
  for (int f = 0; f < v->n; f++) {
    // The temporal term is constant, so the gradient passes straight through:
    // copy this frame's upstream grad and run the patch-projection backward.
    memcpy(frame_slice.grad, &out->grad[f * ppf * dm], (size_t)ppf * dm * sizeof(float));
    patch_embed_backward(&v->frames[f], patch, W, b, &frame_slice);
  }
  tensor_free(&frame_slice);
}
