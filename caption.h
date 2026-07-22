// caption.h - `caption` subcommand: teach-then-reproduce media->text demo.
#ifndef CAPTION_H
#define CAPTION_H

// transformer caption <image.ppm | audio.wav | frames_dir/> <caption-text> [steps]
//
// Loads the media, builds embeddings with the multimodal front-ends (a randomly
// initialised projection), then trains a tiny GPT to caption it with the given
// text and generates the caption back. Prints "caption: <text>" to stdout and
// diagnostics to stderr. This proves the multimodal train->generate pipeline
// runs on real media end-to-end; it is NOT a trained captioner (the front-end is
// untrained and it overfits one example), so quality is toy-scale by design.
int run_caption(int argc, char **argv);

#endif // CAPTION_H
