// text2media.h - `text2media` subcommand: text -> image / audio (teach-then-
// reproduce, exposing the cond_gen conditional-generation path).
#ifndef TEXT2MEDIA_H
#define TEXT2MEDIA_H

// transformer text2media <target.ppm | target.wav> <out.ppm | out.wav> <text> [steps]
//
// Trains a tiny GPT to generate the target media's tokens from the text prefix
// (combined text+media vocab, overfit on this one pair), then generates the media
// from the text alone and writes it to <out>. This exposes the text->media
// pipeline (smoke test [11]); it is teach-then-reproduce on one example, NOT a
// zero-shot text-to-image model (that needs a trained diffusion/VQ-VAE + scale),
// so the output reproduces the target. Prints a status line to stdout.
int run_text2media(int argc, char **argv);

#endif // TEXT2MEDIA_H
