// patch.h - ViT-style patch embedding (image -> sequence of vectors)
#ifndef PATCH_H
#define PATCH_H

#include "tensor.h"
#include "image.h"

// Number of patches for an image at the given patch size ((h/patch)*(w/patch)).
int patch_count(const Image *im, int patch);

// Flattened patch dimension: patch * patch * channels.
int patch_dim(const Image *im, int patch);

// Split the image into a grid of (patch x patch) patches (row-major over the
// grid), flatten each patch (patch*patch*C values, HWC order), and linearly
// project it with W (patch_dim, d_model) + b (d_model). Produces a sequence of
// patch embeddings the transformer can consume.
//   out : (patch_count, d_model), caller-allocated.
void patch_embed(const Image *im, int patch, const Tensor *W, const Tensor *b, Tensor *out);

// Backward of patch_embed. `out` must carry the upstream gradient in out->grad
// (e.g. img_emb->grad from gpt_backward_multimodal). Accumulates into W->grad and
// b->grad (the image pixels are not differentiated). Recomputes the flattened
// patches from the image, so no cache is needed.
void patch_embed_backward(const Image *im, int patch, const Tensor *W,
                          const Tensor *b, const Tensor *out);

#endif // PATCH_H
