// patch.c - ViT-style patch embedding (image -> sequence of vectors)
#include "patch.h"

#include "matrix.h"

int patch_count(const Image *im, int patch) {
  return (im->h / patch) * (im->w / patch);
}

int patch_dim(const Image *im, int patch) {
  return patch * patch * im->c;
}

// Build the flattened patch matrix (np, pdim) from the image (HWC order).
static Tensor build_flat(const Image *im, int patch) {
  const int gh = im->h / patch, gw = im->w / patch;
  const int np = gh * gw;
  const int pdim = patch * patch * im->c;
  const int C = im->c, W_img = im->w;

  int shape[2] = {np, pdim};
  Tensor flat = tensor_create(shape, 2);
  int row = 0;
  for (int py = 0; py < gh; py++) {
    for (int px = 0; px < gw; px++) {
      int f = 0;
      for (int dy = 0; dy < patch; dy++) {
        for (int dx = 0; dx < patch; dx++) {
          int y = py * patch + dy;
          int x = px * patch + dx;
          const float *px_data = &im->data[(y * W_img + x) * C];
          for (int ch = 0; ch < C; ch++) flat.data[row * pdim + f++] = px_data[ch];
        }
      }
      row++;
    }
  }
  return flat;
}

void patch_embed(const Image *im, int patch, const Tensor *W, const Tensor *b, Tensor *out) {
  Tensor flat = build_flat(im, patch);
  matmul(&flat, W, out);      // (np, pdim) @ (pdim, d_model) -> (np, d_model)
  bias_add(out, b, out);
  tensor_free(&flat);
}

void patch_embed_backward(const Image *im, int patch, const Tensor *W,
                          const Tensor *b, const Tensor *out) {
  Tensor flat = build_flat(im, patch);
  // out = flat @ W + b, with out->grad already set.
  bias_add_backward(out, b);        // b->grad += sum rows out->grad
  matmul_backward(&flat, W, out);   // W->grad += flat^T @ out->grad
  tensor_free(&flat);
}
