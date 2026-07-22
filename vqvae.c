// vqvae.c - patch-based VQ-VAE image codec (learned; matmul only, no conv).
#include "vqvae.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "optim.h"

// --- patch <-> tensor helpers (raster over the patch grid; HWC within a patch) ---

static void extract_patches(const Image *im, int patch, Tensor *out) {
  const int gw = im->w / patch, c = im->c;
  const int pdim = patch * patch * c;
  const int gh = im->h / patch;
  for (int gy = 0; gy < gh; gy++)
    for (int gx = 0; gx < gw; gx++) {
      int pi = gy * gw + gx;
      for (int py = 0; py < patch; py++)
        for (int px = 0; px < patch; px++)
          for (int ch = 0; ch < c; ch++) {
            int iy = gy * patch + py, ix = gx * patch + px;
            out->data[(size_t)pi * pdim + ((size_t)(py * patch + px) * c + ch)] =
                im->data[((size_t)(iy * im->w + ix) * c + ch)];
          }
    }
}

static void patches_to_image(const Tensor *patches, int patch, Image *im) {
  const int gw = im->w / patch, c = im->c, gh = im->h / patch;
  const int pdim = patch * patch * c;
  for (int gy = 0; gy < gh; gy++)
    for (int gx = 0; gx < gw; gx++) {
      int pi = gy * gw + gx;
      for (int py = 0; py < patch; py++)
        for (int px = 0; px < patch; px++)
          for (int ch = 0; ch < c; ch++) {
            int iy = gy * patch + py, ix = gx * patch + px;
            float v = patches->data[(size_t)pi * pdim + ((size_t)(py * patch + px) * c + ch)];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            im->data[((size_t)(iy * im->w + ix) * c + ch)] = v;
          }
    }
}

// --- model ---

VQVAE vqvae_create(int patch, int channels, int ldim, int K) {
  VQVAE v;
  v.patch = patch; v.channels = channels;
  v.pdim = patch * patch * channels;
  v.ldim = ldim; v.K = K;
  int we[2] = {v.pdim, ldim}; v.W_enc = tensor_create(we, 2); tensor_init_random(&v.W_enc);
  int wd[2] = {ldim, v.pdim}; v.W_dec = tensor_create(wd, 2); tensor_init_random(&v.W_dec);
  int cb[2] = {K, ldim};      v.codebook = tensor_create(cb, 2); tensor_init_random(&v.codebook);
  return v;
}

void vqvae_free(VQVAE *v) {
  tensor_free(&v->W_enc); tensor_free(&v->W_dec); tensor_free(&v->codebook);
}

int vqvae_ncodes(const VQVAE *v, int w, int h) {
  return (w / v->patch) * (h / v->patch);
}

// Assign each latent row to its nearest codebook entry; fill codes[] and z_q.
static void quantize(const VQVAE *v, const Tensor *ze, int np, int *codes, Tensor *zq) {
  const int ld = v->ldim, K = v->K;
  for (int i = 0; i < np; i++) {
    const float *z = &ze->data[(size_t)i * ld];
    int best = 0;
    float bd = 1e30f;
    for (int k = 0; k < K; k++) {
      const float *cb = &v->codebook.data[(size_t)k * ld];
      float d = 0.0f;
      for (int j = 0; j < ld; j++) { float t = z[j] - cb[j]; d += t * t; }
      if (d < bd) { bd = d; best = k; }
    }
    codes[i] = best;
    if (zq) {
      for (int j = 0; j < ld; j++) zq->data[(size_t)i * ld + j] = v->codebook.data[(size_t)best * ld + j];
    }
  }
}

void vqvae_train(VQVAE *v, const Image *imgs, int n, int steps, float lr) {
  const int pdim = v->pdim, ld = v->ldim;
  const int np = vqvae_ncodes(v, imgs[0].w, imgs[0].h);
  const float beta = 0.25f;   // commitment weight

  Tensor *params[3] = {&v->W_enc, &v->W_dec, &v->codebook};
  Adam opt = adam_create(params, 3, lr);

  int ps[2] = {np, pdim}, zs[2] = {np, ld};
  Tensor P = tensor_create(ps, 2);
  Tensor ze = tensor_create(zs, 2);
  Tensor zq = tensor_create(zs, 2);       // straight-through decoder input (data = quantized)
  Tensor recon = tensor_create(ps, 2);
  int *codes = (int*)malloc((size_t)np * sizeof(int));

  srand(1234);
  const float invR = 1.0f / (float)(np * pdim);
  const float invZ = 1.0f / (float)(np * ld);
  for (int s = 0; s < steps; s++) {
    int e = (n > 1) ? rand() % n : 0;
    extract_patches(&imgs[e], v->patch, &P);

    matmul(&P, &v->W_enc, &ze);            // encode
    quantize(v, &ze, np, codes, &zq);      // nearest code -> zq (data used by decoder)
    matmul(&zq, &v->W_dec, &recon);        // decode from the quantized latent

    // reconstruction loss + its gradient into recon.
    tensor_zero_grad(&recon);
    float mse = 0.0f;
    for (int i = 0; i < np * pdim; i++) {
      float d = recon.data[i] - P.data[i];
      mse += d * d;
      recon.grad[i] = 2.0f * d * invR;
    }
    mse *= invR;

    // zero param + intermediate grads for this step.
    tensor_zero_grad(&v->W_enc); tensor_zero_grad(&v->W_dec); tensor_zero_grad(&v->codebook);
    tensor_zero_grad(&zq); tensor_zero_grad(&ze); tensor_zero_grad(&P);

    matmul_backward(&zq, &v->W_dec, &recon);   // -> zq.grad, W_dec.grad

    // straight-through: encoder gets the decoder's gradient wrt its (quantized)
    // input, plus the commitment pull toward the chosen code. The codebook gets
    // pulled toward the encoder output for the codes it was assigned.
    for (int i = 0; i < np; i++) {
      int k = codes[i];
      for (int j = 0; j < ld; j++) {
        size_t idx = (size_t)i * ld + j;
        float e_val = ze.data[idx], q_val = zq.data[idx];
        ze.grad[idx] = zq.grad[idx] + 2.0f * beta * (e_val - q_val) * invZ; // straight-through + commitment
        v->codebook.grad[(size_t)k * ld + j] += 2.0f * (q_val - e_val) * invZ; // codebook loss
      }
    }

    matmul_backward(&P, &v->W_enc, &ze);       // -> W_enc.grad (P.grad ignored)
    adam_step(&opt, params, 3);

    if (s % 100 == 0 || s == steps - 1)
      fprintf(stderr, "  vq step %d/%d recon_mse %.5f\n", s, steps, mse);
  }

  free(codes);
  tensor_free(&P); tensor_free(&ze); tensor_free(&zq); tensor_free(&recon);
  adam_free(&opt);
}

void vqvae_encode(const VQVAE *v, const Image *im, int *codes) {
  const int np = vqvae_ncodes(v, im->w, im->h);
  int ps[2] = {np, v->pdim}, zs[2] = {np, v->ldim};
  Tensor P = tensor_create(ps, 2), ze = tensor_create(zs, 2);
  extract_patches(im, v->patch, &P);
  matmul(&P, &v->W_enc, &ze);
  quantize(v, &ze, np, codes, NULL);
  tensor_free(&P); tensor_free(&ze);
}

Image vqvae_decode(const VQVAE *v, const int *codes, int w, int h) {
  const int np = vqvae_ncodes(v, w, h);
  int zs[2] = {np, v->ldim}, ps[2] = {np, v->pdim};
  Tensor zq = tensor_create(zs, 2), recon = tensor_create(ps, 2);
  for (int i = 0; i < np; i++)
    for (int j = 0; j < v->ldim; j++)
      zq.data[(size_t)i * v->ldim + j] = v->codebook.data[(size_t)codes[i] * v->ldim + j];
  matmul(&zq, &v->W_dec, &recon);

  Image im;
  im.h = h; im.w = w; im.c = v->channels;
  im.data = (float*)malloc((size_t)w * h * v->channels * sizeof(float));
  patches_to_image(&recon, v->patch, &im);
  tensor_free(&zq); tensor_free(&recon);
  return im;
}

// --- CLI test: train on one image, reconstruct it, report codes + error ---
int run_vqvae(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: transformer vqvae <in.ppm> <out.ppm> [steps]\n");
    return 2;
  }
  const char *in = argv[2], *out = argv[3];
  int steps = (argc > 4) ? atoi(argv[4]) : 1500;
  if (steps < 1) steps = 1500;

  Image im = image_load_ppm(in);
  if (!im.data) { fprintf(stderr, "error: cannot load '%s'\n", in); return 1; }
  const int patch = 2, ldim = 16, K = 128;
  if (im.w % patch || im.h % patch) { fprintf(stderr, "error: image dims must be divisible by %d\n", patch); image_free(&im); return 1; }

  VQVAE v = vqvae_create(patch, im.c, ldim, K);
  vqvae_train(&v, &im, 1, steps, 0.01f);

  int nc = vqvae_ncodes(&v, im.w, im.h);
  int *codes = (int*)malloc((size_t)nc * sizeof(int));
  vqvae_encode(&v, &im, codes);
  Image og = vqvae_decode(&v, codes, im.w, im.h);

  double err = 0.0;
  int npix = im.w * im.h * im.c;
  for (int i = 0; i < npix; i++) { double d = og.data[i] - im.data[i]; err += d * d; }
  err /= npix;
  int ok = (image_save_ppm(out, &og) == 0);
  printf("vqvae: %dx%d | %d codes (vs %d color-cube pixels), K=%d, ldim=%d | recon MSE %.5f -> %s (%s)\n",
         im.w, im.h, nc, im.w * im.h, K, ldim, err, out, ok ? "ok" : "FAILED");

  free(codes); image_free(&og); vqvae_free(&v); image_free(&im);
  return ok ? 0 : 1;
}
