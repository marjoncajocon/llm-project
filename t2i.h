// t2i.h - `t2i` subcommand: dataset-trained text->image (autoregressive, the
// DALL-E 1 / Parti recipe at tiny scale — no diffusion).
#ifndef T2I_H
#define T2I_H

// transformer t2i <dataset-dir> <out.ppm> <caption> [steps] [levels]
//
// Trains ONE GPT on a folder of (caption, image) pairs named cap_0000.txt /
// img_0000.ppm, cap_0001.txt / img_0001.ppm, ... (all images the same size):
// each example is [caption text tokens | image color-cube tokens] and the image
// tokens are supervised, so the model learns to GENERATE the image from the
// caption. Then it generates an image for <caption> and writes it to <out.ppm>.
//
// This is a real (if toy) text->image MODEL living entirely in the engine — it
// generalizes within its training distribution instead of overfitting one pair.
// Quality is capped by scale + the crude color-cube codec (a learned VQ-VAE is
// the next step). Returns 0 on success.
int run_t2i(int argc, char **argv);

#endif // T2I_H
