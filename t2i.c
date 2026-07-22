// t2i.c - dataset-trained text->image (autoregressive; no diffusion).
//
// Trains one GPT on (caption, image) pairs so it learns P(image tokens | caption)
// and can generate an image from a caption prompt. Same machinery as the rest of
// the engine (gpt_forward_train/backward + Adam + the color-cube codec); the only
// new thing is the training loop over a captioned image dataset.
#include "t2i.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"
#include "imagegen.h"
#include "model.h"
#include "loss.h"
#include "optim.h"
#include "sampler.h"

#define T2I_CAPTION_LEN 32     // captions are padded/truncated to a fixed length
#define T2I_MAX_PAIRS   4096

static int read_text_file(const char *path, char *buf, int cap) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  int n = (int)fread(buf, 1, (size_t)cap - 1, f);
  fclose(f);
  buf[n] = '\0';
  while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ')) buf[--n] = '\0';
  return n;
}

// Encode a caption string into T2I_CAPTION_LEN text tokens, offset above the media
// vocab so text and image ids never collide (padded with byte 0).
static void encode_caption(const char *s, int v_media, int *out) {
  int len = (int)strlen(s);
  for (int j = 0; j < T2I_CAPTION_LEN; j++) {
    int b = (j < len) ? (unsigned char)s[j] : 0;
    out[j] = b + v_media;
  }
}

int run_t2i(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: transformer t2i <dataset-dir> <out.ppm> <caption> [steps] [levels]\n"
                    "  dataset-dir holds cap_0000.txt/img_0000.ppm, cap_0001.txt/img_0001.ppm, ...\n");
    return 2;
  }
  const char *dir = argv[2], *out = argv[3], *caption = argv[4];
  int steps = (argc > 5) ? atoi(argv[5]) : 1500;
  int levels = (argc > 6) ? atoi(argv[6]) : 4;
  if (steps < 1) steps = 1500;
  if (levels < 2) levels = 4;

  const int T = T2I_CAPTION_LEN;
  const int v_media = imagegen_vocab(levels);
  const int v_text = 256;
  const int V = v_media + v_text;

  // --- load the (caption, image) dataset ---
  int **imgtok = (int**)malloc(T2I_MAX_PAIRS * sizeof(int*));
  int **txttok = (int**)malloc(T2I_MAX_PAIRS * sizeof(int*));
  int N = 0, w = 0, h = 0;
  for (int i = 0; i < T2I_MAX_PAIRS; i++) {
    char ip[600], cp[600], cap[600];
    snprintf(ip, sizeof(ip), "%s/img_%04d.ppm", dir, i);
    snprintf(cp, sizeof(cp), "%s/cap_%04d.txt", dir, i);
    Image im = image_load_ppm(ip);
    if (!im.data) break;
    if (read_text_file(cp, cap, sizeof(cap)) < 0) { image_free(&im); break; }
    if (N == 0) {
      w = im.w; h = im.h;
      if (w * h > 1024) { fprintf(stderr, "error: images must be <= 1024 px (got %dx%d)\n", w, h); image_free(&im); free(imgtok); free(txttok); return 1; }
    } else if (im.w != w || im.h != h) {
      fprintf(stderr, "skip img_%04d.ppm: size mismatch\n", i); image_free(&im); continue;
    }
    int M = w * h;
    imgtok[N] = (int*)malloc((size_t)M * sizeof(int));
    image_to_tokens(&im, levels, imgtok[N]);
    txttok[N] = (int*)malloc((size_t)T * sizeof(int));
    encode_caption(cap, v_media, txttok[N]);
    N++;
    image_free(&im);
  }
  if (N < 1) {
    fprintf(stderr, "error: no cap_0000.txt/img_0000.ppm pairs found in '%s'\n", dir);
    free(imgtok); free(txttok);
    return 1;
  }
  const int M = w * h;
  const int total = T + M;
  fprintf(stderr, "t2i: %d pairs, %dx%d (%d image tokens), vocab %d (%d img + %d text), %d steps\n",
          N, w, h, M, V, v_media, v_text, steps);

  // --- train one GPT over the whole dataset (SGD, one random pair per step) ---
  GPT g = gpt_create(V, 64, 4, 3, 256, total);
  GPTActs a = gpt_acts_create(&g, total);
  int lz[2] = {total, V};
  Tensor lo = tensor_create(lz, 2), pr = tensor_create(lz, 2);
  Tensor **par = (Tensor**)malloc((size_t)gpt_param_count(&g) * sizeof(Tensor*));
  int np = gpt_collect_params(&g, par);
  Adam opt = adam_create(par, np, 0.005f);
  int *seq = (int*)malloc((size_t)total * sizeof(int));
  int *tgt = (int*)malloc((size_t)total * sizeof(int));

  srand(1234);
  float lN = 0.0f;
  for (int s = 0; s < steps; s++) {
    int e = rand() % N;
    for (int i = 0; i < T; i++) seq[i] = txttok[e][i];
    for (int j = 0; j < M; j++) seq[T + j] = imgtok[e][j];
    for (int i = 0; i < total; i++) tgt[i] = -1;
    for (int k = 0; k < M; k++) tgt[T - 1 + k] = imgtok[e][k];   // supervise image tokens

    gpt_forward_train(&g, seq, total, &a, &lo);
    float loss = cross_entropy_forward(&lo, tgt, &pr);
    lN = loss;
    tensor_zero_grad(&lo);
    cross_entropy_backward(&pr, tgt, &lo);
    gpt_backward(&g, seq, total, &a, &lo);
    adam_step(&opt, par, np);
    if (s % 100 == 0 || s == steps - 1)
      fprintf(stderr, "  step %d/%d loss %.4f\n", s, steps, loss);
  }

  // --- generate an image for the query caption ---
  int *qtext = (int*)malloc((size_t)T * sizeof(int));
  encode_caption(caption, v_media, qtext);
  KVCache kv = gpt_kvcache_create(&g);
  int gz[2] = {1, V};
  Tensor gl = tensor_create(gz, 2);
  for (int i = 0; i < T; i++) gpt_decode_step(&g, &kv, qtext[i], &gl);
  int *gen = (int*)malloc((size_t)M * sizeof(int));
  for (int k = 0; k < M; k++) {
    gen[k] = sample_argmax(&gl, 0);
    if (gen[k] >= v_media) gen[k] = v_media - 1;   // clamp any stray text id out of the image range
    if (k < M - 1) gpt_decode_step(&g, &kv, gen[k], &gl);
  }
  Image og = tokens_to_image(gen, w, h, levels);
  int ok = (image_save_ppm(out, &og) == 0);
  printf("t2i: trained on %d pairs (final loss %.4f) | generated %dx%d for \"%s\" -> %s (%s)\n",
         N, lN, w, h, caption, out, ok ? "ok" : "FAILED");

  // --- cleanup ---
  image_free(&og); tensor_free(&gl); gpt_kvcache_free(&kv);
  free(gen); free(qtext);
  adam_free(&opt); tensor_free(&lo); tensor_free(&pr); gpt_acts_free(&a); gpt_free(&g);
  free(seq); free(tgt); free(par);
  for (int e = 0; e < N; e++) { free(imgtok[e]); free(txttok[e]); }
  free(imgtok); free(txttok);
  return ok ? 0 : 1;
}
