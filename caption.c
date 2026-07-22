// caption.c - `caption` subcommand: teach-then-reproduce media->text demo.
//
// Reuses the multimodal path (patch/audio/video front-ends + gpt_*_multimodal +
// cross-entropy) exactly as selftest.c's media_to_text, but on a real media file
// and with byte-level caption tokens (so any caption string works). Additive:
// touches no training/backprop internals, so the gradient + KV-cache checks are
// unaffected.
#include "caption.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "image.h"
#include "patch.h"
#include "audio.h"
#include "video.h"
#include "model.h"
#include "loss.h"
#include "optim.h"
#include "sampler.h"

#define CAP_DMODEL 32   // media/model dimension (matches the selftest demo)

// Train a tiny GPT to caption media_emb with `cap` (C byte tokens), then generate
// the caption back autoregressively from the media alone. Prints the decoded
// caption to stdout and stats to stderr. Returns the number of reproduced bytes.
static int caption_generate(const char *kind, const Tensor *media_emb, int n_media,
                            const int *cap, int C, int steps) {
  const int v_text = 256;                    // byte-level caption vocab
  const int dm = media_emb->shape[1];
  const int total = n_media + C;

  GPT g = gpt_create(v_text, dm, 2, 2, dm * 4, total);
  GPTActs a = gpt_acts_create(&g, total);
  int lz[2] = {total, v_text};
  Tensor lo = tensor_create(lz, 2), pr = tensor_create(lz, 2);
  int *tgt = (int*)malloc(total * sizeof(int));
  for (int i = 0; i < total; i++) tgt[i] = -1;
  for (int k = 0; k < C; k++) tgt[n_media - 1 + k] = cap[k];   // predict caption after media
  Tensor *par[128];
  int np = gpt_collect_params(&g, par);
  Adam opt = adam_create(par, np, 0.01f);

  float l0 = 0.0f, lN = 0.0f;
  for (int s = 0; s < steps; s++) {
    gpt_forward_multimodal(&g, media_emb, cap, n_media, C, &a, &lo);
    float loss = cross_entropy_forward(&lo, tgt, &pr);
    if (s == 0) l0 = loss;
    lN = loss;
    tensor_zero_grad(&lo);
    cross_entropy_backward(&pr, tgt, &lo);
    gpt_backward_multimodal(&g, cap, n_media, C, &a, &lo, (Tensor*)media_emb);
    adam_step(&opt, par, np);
    if (s % 50 == 0 || s == steps - 1)
      fprintf(stderr, "  step %d/%d loss %.4f\n", s, steps, loss);
  }
  gpt_acts_free(&a); tensor_free(&lo); tensor_free(&pr); adam_free(&opt); free(tgt);

  // Autoregressively generate the caption from the media prefix.
  int *gen = (int*)malloc(C * sizeof(int));
  for (int t = 0; t < C; t++) {
    int tot = n_media + t;
    GPTActs ag = gpt_acts_create(&g, tot);
    int gz[2] = {tot, v_text};
    Tensor glo = tensor_create(gz, 2);
    gpt_forward_multimodal(&g, media_emb, gen, n_media, t, &ag, &glo);
    gen[t] = sample_argmax(&glo, tot - 1);
    tensor_free(&glo); gpt_acts_free(&ag);
  }

  int match = 0;
  char out[8192];
  int oi = 0;
  for (int t = 0; t < C; t++) {
    if (gen[t] == cap[t]) match++;
    if (gen[t] >= 0 && gen[t] < 256 && oi < (int)sizeof(out) - 1) out[oi++] = (char)gen[t];
  }
  out[oi] = '\0';

  printf("caption: %s\n", out);        // stdout = result (the webserver reads this)
  fflush(stdout);
  fprintf(stderr, "%s: %d media tokens + %d caption bytes | loss %.4f -> %.4f | reproduced %d/%d bytes\n",
          kind, n_media, C, l0, lN, match, C);

  free(gen); gpt_free(&g);
  return match;
}

static int ends_with(const char *s, const char *suf) {
  size_t ls = strlen(s), lf = strlen(suf);
  return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

int run_caption(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: transformer caption <image.ppm | audio.wav | frames_dir> <caption-text> [steps]\n");
    return 2;
  }
  const char *path = argv[2];
  const char *text = argv[3];
  int steps = (argc > 4) ? atoi(argv[4]) : 300;
  if (steps < 1) steps = 300;

  int C = (int)strlen(text);
  if (C < 1) { fprintf(stderr, "error: empty caption text\n"); return 1; }
  int *cap = (int*)malloc((size_t)C * sizeof(int));
  for (int i = 0; i < C; i++) cap[i] = (unsigned char)text[i];   // byte-level tokens

  const int MD = CAP_DMODEL;
  Tensor media_emb;
  int n_media = 0;
  const char *kind = NULL;

  if (ends_with(path, ".wav")) {
    Audio au = audio_load_wav(path);
    if (!au.samples) { fprintf(stderr, "error: cannot load audio '%s'\n", path); free(cap); return 1; }
    int fs = 32, hop = 32, nf = audio_frame_count(&au, fs, hop);
    if (nf < 1) { fprintf(stderr, "error: audio too short (need >= %d samples)\n", fs); audio_free(&au); free(cap); return 1; }
    int ws[2] = {fs, MD}; Tensor W = tensor_create(ws, 2); tensor_init_random(&W);
    int bs[1] = {MD};     Tensor b = tensor_create(bs, 1);
    int es[2] = {nf, MD}; media_emb = tensor_create(es, 2);
    audio_embed(&au, fs, hop, &W, &b, &media_emb);
    n_media = nf; kind = "audio";
    tensor_free(&W); tensor_free(&b); audio_free(&au);
  } else if (ends_with(path, ".ppm")) {
    Image im = image_load_ppm(path);
    if (!im.data) { fprintf(stderr, "error: cannot load image '%s'\n", path); free(cap); return 1; }
    int patch = 2, np = patch_count(&im, patch), pd = patch_dim(&im, patch);
    if (np < 1) { fprintf(stderr, "error: image too small for patches\n"); image_free(&im); free(cap); return 1; }
    int ws[2] = {pd, MD}; Tensor W = tensor_create(ws, 2); tensor_init_random(&W);
    int bs[1] = {MD};     Tensor b = tensor_create(bs, 1);
    int es[2] = {np, MD}; media_emb = tensor_create(es, 2);
    patch_embed(&im, patch, &W, &b, &media_emb);
    n_media = np; kind = "image";
    tensor_free(&W); tensor_free(&b); image_free(&im);
  } else {
    Video vd = video_load(path);
    if (vd.n < 1) { fprintf(stderr, "error: no frames (frame_0000.ppm) in '%s'\n", path); free(cap); return 1; }
    int patch = 2, nvp = video_patch_count(&vd, patch), pd = patch_dim(&vd.frames[0], patch);
    if (nvp < 1) { fprintf(stderr, "error: frames too small for patches\n"); video_free(&vd); free(cap); return 1; }
    int ws[2] = {pd, MD}; Tensor W = tensor_create(ws, 2); tensor_init_random(&W);
    int bs[1] = {MD};     Tensor b = tensor_create(bs, 1);
    int es[2] = {nvp, MD}; media_emb = tensor_create(es, 2);
    video_embed(&vd, patch, &W, &b, &media_emb);
    n_media = nvp; kind = "video";
    fprintf(stderr, "video: %d frames\n", vd.n);
    tensor_free(&W); tensor_free(&b); video_free(&vd);
  }

  caption_generate(kind, &media_emb, n_media, cap, C, steps);
  tensor_free(&media_emb);
  free(cap);
  return 0;
}
