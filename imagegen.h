// imagegen.h - Image <-> token codec for generation (color-cube quantization)
#ifndef IMAGEGEN_H
#define IMAGEGEN_H

#include "image.h"

// Discrete-token image generation, minimal version: each pixel is quantized to a
// color-cube token (each RGB channel to `levels` steps -> levels^3 tokens). No
// training needed for the codec; the GPT then generates the token sequence
// autoregressively, and we decode tokens back to pixels. (A learned VQ-VAE would
// give real quality; this is the dependency-free minimal path.)

// Vocabulary size = levels^3.
int imagegen_vocab(int levels);

// Quantize every pixel of `im` to a color-cube token. Writes im->h * im->w tokens
// (raster order) into `tokens`.
void image_to_tokens(const Image *im, int levels, int *tokens);

// Decode w*h tokens back into an Image (caller frees via image_free).
Image tokens_to_image(const int *tokens, int w, int h, int levels);

#endif // IMAGEGEN_H
