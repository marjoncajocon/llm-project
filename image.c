// image.c - Minimal image loading (binary PPM / P6)
#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

// Read the next integer from a PPM header, skipping whitespace and #comments.
static int ppm_next_int(FILE *f, int *ok) {
  int c;
  // skip whitespace and comment lines
  for (;;) {
    c = fgetc(f);
    if (c == EOF) { *ok = 0; return 0; }
    if (c == '#') { while (c != '\n' && c != EOF) c = fgetc(f); continue; }
    if (!isspace(c)) break;
  }
  int val = 0, any = 0;
  while (c != EOF && isdigit(c)) { val = val * 10 + (c - '0'); any = 1; c = fgetc(f); }
  *ok = any;
  return val;
}

Image image_load_ppm(const char *path) {
  Image im = {0, 0, 0, NULL};
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "image: cannot open '%s'\n", path); return im; }

  char magic[3] = {0};
  if (fread(magic, 1, 2, f) != 2 || magic[0] != 'P' || magic[1] != '6') {
    fprintf(stderr, "image: '%s' is not a binary PPM (P6)\n", path);
    fclose(f); return im;
  }

  int ok = 1;
  int w = ppm_next_int(f, &ok);
  int h = ppm_next_int(f, &ok);
  int maxval = ppm_next_int(f, &ok);
  if (!ok || w <= 0 || h <= 0 || maxval <= 0 || maxval > 255) {
    fprintf(stderr, "image: bad PPM header in '%s'\n", path);
    fclose(f); return im;
  }
  // ppm_next_int already consumed the single whitespace after maxval, so the
  // pixel data starts here — do NOT skip another byte.

  int n = w * h * 3;
  unsigned char *raw = (unsigned char*)malloc(n);
  if (fread(raw, 1, n, f) != (size_t)n) {
    fprintf(stderr, "image: '%s' is truncated\n", path);
    free(raw); fclose(f); return im;
  }
  fclose(f);

  im.h = h; im.w = w; im.c = 3;
  im.data = (float*)malloc(n * sizeof(float));
  float inv = 1.0f / (float)maxval;
  for (int i = 0; i < n; i++) im.data[i] = raw[i] * inv;
  free(raw);
  return im;
}

int image_save_ppm(const char *path, const Image *im) {
  FILE *f = fopen(path, "wb");
  if (!f) return 1;
  fprintf(f, "P6\n%d %d\n255\n", im->w, im->h);
  for (int i = 0; i < im->w * im->h * 3; i++) {
    int v = (int)(im->data[i] * 255.0f + 0.5f);
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    fputc(v, f);
  }
  fclose(f);
  return 0;
}

void image_free(Image *im) {
  free(im->data);
  im->data = NULL;
  im->h = im->w = im->c = 0;
}
