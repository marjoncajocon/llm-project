// vqvae.h - A patch-based VQ-VAE image codec (learned, no diffusion, no conv).
//
// Splits an image into patches, linearly ENCODES each to a latent vector,
// QUANTIZES it to the nearest of K learned codebook entries (the discrete code),
// and linearly DECODES the codebook vector back to patch pixels. Trained with a
// reconstruction loss + codebook/commitment losses and the straight-through
// estimator — the standard VQ-VAE, using only matmul/matmul_backward. Gives a
// short grid of discrete codes (one per patch) with far better color fidelity
// than the per-pixel color-cube, which is the real quality lever for the
// autoregressive text->image model (t2i).
#ifndef VQVAE_H
#define VQVAE_H

#include "tensor.h"
#include "image.h"

typedef struct {
  int patch;        // patch size (square)
  int channels;     // 3 (RGB)
  int pdim;         // patch*patch*channels (flattened patch dim)
  int ldim;         // latent dim
  int K;            // codebook size (the code vocabulary)
  Tensor W_enc;     // (pdim, ldim)
  Tensor W_dec;     // (ldim, pdim)
  Tensor codebook;  // (K, ldim)
} VQVAE;

VQVAE vqvae_create(int patch, int channels, int ldim, int K);
void vqvae_free(VQVAE *v);

// Codes per image = (h/patch) * (w/patch).
int vqvae_ncodes(const VQVAE *v, int w, int h);

// Train the codec to reconstruct a set of same-size images.
void vqvae_train(VQVAE *v, const Image *imgs, int n, int steps, float lr);

// Encode an image to code indices [0,K) (one per patch; vqvae_ncodes of them).
void vqvae_encode(const VQVAE *v, const Image *im, int *codes);

// Decode code indices back to an image (caller frees via image_free).
Image vqvae_decode(const VQVAE *v, const int *codes, int w, int h);

// CLI smoke test: transformer vqvae <in.ppm> <out.ppm> [steps]
int run_vqvae(int argc, char **argv);

#endif // VQVAE_H
