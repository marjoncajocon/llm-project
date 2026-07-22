// text2media.c - text -> image / audio via the conditional-generation path.
//
// Mirrors selftest.c's cond_gen: build a combined [text|media] vocabulary, train
// a tiny GPT to predict the media tokens from the text prefix (overfit on one
// pair), then generate the media from the text alone and decode it to a file.
// Additive: uses only the existing forward/backward + codecs, so the gradient
// and KV-cache checks are unaffected.
#include "text2media.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"
#include "imagegen.h"
#include "audio.h"
#include "audiogen.h"
#include "model.h"
#include "loss.h"
#include "optim.h"
#include "sampler.h"

// Train text->media on one (text, media) pair, then generate M media tokens from
// the text prefix into gen[]. Returns the number reproduced exactly.
static int cond_generate(const int *text, int T, const int *media, int M,
                         int v_media, int v_text, int steps, int *gen) {
  const int V = v_media + v_text;
  const int total = T + M;
  int *seq = (int*)malloc((size_t)total * sizeof(int));
  int *tgt = (int*)malloc((size_t)total * sizeof(int));
  for (int i = 0; i < T; i++) seq[i] = text[i] + v_media;   // text ids live above media ids
  for (int j = 0; j < M; j++) seq[T + j] = media[j];
  for (int i = 0; i < total; i++) tgt[i] = -1;
  for (int k = 0; k < M; k++) tgt[T - 1 + k] = media[k];    // predict media_k after the prefix

  GPT g = gpt_create(V, 48, 3, 2, 192, total);
  GPTActs a = gpt_acts_create(&g, total);
  int lz[2] = {total, V};
  Tensor lo = tensor_create(lz, 2), pr = tensor_create(lz, 2);
  Tensor **par = (Tensor**)malloc((size_t)gpt_param_count(&g) * sizeof(Tensor*));
  int np = gpt_collect_params(&g, par);
  Adam opt = adam_create(par, np, 0.01f);

  float l0 = 0.0f, lN = 0.0f;
  for (int s = 0; s < steps; s++) {
    gpt_forward_train(&g, seq, total, &a, &lo);
    float loss = cross_entropy_forward(&lo, tgt, &pr);
    if (s == 0) l0 = loss;
    lN = loss;
    tensor_zero_grad(&lo);
    cross_entropy_backward(&pr, tgt, &lo);
    gpt_backward(&g, seq, total, &a, &lo);
    adam_step(&opt, par, np);
    if (s % 50 == 0 || s == steps - 1)
      fprintf(stderr, "  step %d/%d loss %.4f\n", s, steps, loss);
  }

  // Generate: feed the text prefix, then greedily decode M media tokens.
  KVCache kv = gpt_kvcache_create(&g);
  int gz[2] = {1, V};
  Tensor gl = tensor_create(gz, 2);
  for (int i = 0; i < T; i++) gpt_decode_step(&g, &kv, seq[i], &gl);
  int match = 0;
  for (int k = 0; k < M; k++) {
    gen[k] = sample_argmax(&gl, 0);
    if (gen[k] == media[k]) match++;
    if (k < M - 1) gpt_decode_step(&g, &kv, gen[k], &gl);
  }

  fprintf(stderr, "loss %.4f -> %.4f\n", l0, lN);
  tensor_free(&gl); gpt_kvcache_free(&kv); adam_free(&opt);
  tensor_free(&lo); tensor_free(&pr); gpt_acts_free(&a); gpt_free(&g);
  free(seq); free(tgt); free(par);
  return match;
}

static int ends_with(const char *s, const char *suf) {
  size_t ls = strlen(s), lf = strlen(suf);
  return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

int run_text2media(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: transformer text2media <target.ppm|target.wav> <out.ppm|out.wav> <text> [steps]\n");
    return 2;
  }
  const char *target = argv[2], *out = argv[3], *text = argv[4];
  int steps = (argc > 5) ? atoi(argv[5]) : 400;
  if (steps < 1) steps = 400;

  int T = (int)strlen(text);
  if (T < 1) { fprintf(stderr, "error: empty text prompt\n"); return 1; }
  int *txt = (int*)malloc((size_t)T * sizeof(int));
  for (int i = 0; i < T; i++) txt[i] = (unsigned char)text[i];   // byte-level text tokens
  const int v_text = 256;

  int rc = 0;
  if (ends_with(target, ".ppm")) {
    Image im = image_load_ppm(target);
    if (!im.data) { fprintf(stderr, "error: cannot load image '%s'\n", target); free(txt); return 1; }
    const int levels = 4;
    int v_media = imagegen_vocab(levels);
    int M = im.h * im.w;
    if (M < 1 || M > 1024) { fprintf(stderr, "error: image must be <= 1024 pixels (got %d)\n", M); image_free(&im); free(txt); return 1; }
    int *media = (int*)malloc((size_t)M * sizeof(int));
    int *gen = (int*)malloc((size_t)M * sizeof(int));
    image_to_tokens(&im, levels, media);
    int match = cond_generate(txt, T, media, M, v_media, v_text, steps, gen);
    Image og = tokens_to_image(gen, im.w, im.h, levels);
    int ok = (image_save_ppm(out, &og) == 0);
    printf("text2image: %dx%d, %d tokens | reproduced %d/%d | wrote %s (%s)\n",
           im.w, im.h, M, match, M, out, ok ? "ok" : "FAILED");
    if (!ok) rc = 1;
    image_free(&og); image_free(&im); free(media); free(gen);
  } else if (ends_with(target, ".wav")) {
    Audio au = audio_load_wav(target);
    if (!au.samples) { fprintf(stderr, "error: cannot load audio '%s'\n", target); free(txt); return 1; }
    int v_media = audiogen_vocab();
    int M = au.n;
    if (M < 1 || M > 1024) { fprintf(stderr, "error: audio must be <= 1024 samples (got %d)\n", M); audio_free(&au); free(txt); return 1; }
    int *media = (int*)malloc((size_t)M * sizeof(int));
    int *gen = (int*)malloc((size_t)M * sizeof(int));
    audio_to_tokens(&au, media);
    int match = cond_generate(txt, T, media, M, v_media, v_text, steps, gen);
    Audio og = tokens_to_audio(gen, M, au.sample_rate);
    int ok = (audio_save_wav(out, &og) == 0);
    printf("text2audio: %d samples @ %d Hz | reproduced %d/%d | wrote %s (%s)\n",
           M, au.sample_rate, match, M, out, ok ? "ok" : "FAILED");
    if (!ok) rc = 1;
    audio_free(&og); audio_free(&au); free(media); free(gen);
  } else {
    fprintf(stderr, "error: target must be a .ppm image or .wav audio\n");
    rc = 1;
  }
  free(txt);
  return rc;
}
