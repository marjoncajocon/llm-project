// t2i.c - dataset-trained text->image (autoregressive; no diffusion).
//
// Two learned stages, both from scratch:
//   1. a VQ-VAE codec (vqvae.c) tokenizes each image into a short grid of
//      discrete codes (one per patch — far fewer than per-pixel color-cube);
//   2. a GPT learns P(codes | caption) over [caption text | image codes].
// Generation: encode the caption -> autoregress the code grid -> VQ-decode to an
// image. This is the DALL-E / Parti recipe (transformer + learned tokenizer), at
// tiny scale.
#include "t2i.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"
#include "vqvae.h"
#include "model.h"
#include "loss.h"
#include "optim.h"
#include "sampler.h"

#define T2I_CAPTION_LEN 32
#define T2I_MAX_PAIRS   4096
#define T2I_PATCH       2
#define T2I_LDIM        16
#define T2I_K           256      // codebook size = the image-code vocabulary

static int read_text_file(const char *path, char *buf, int cap) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  int n = (int)fread(buf, 1, (size_t)cap - 1, f);
  fclose(f);
  buf[n] = '\0';
  while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ')) buf[--n] = '\0';
  return n;
}

// Caption -> fixed-length text tokens, offset above the image-code vocab so text
// and code ids never collide (padded with byte 0).
static void encode_caption(const char *s, int v_media, int *out) {
  int len = (int)strlen(s);
  for (int j = 0; j < T2I_CAPTION_LEN; j++) {
    int b = (j < len) ? (unsigned char)s[j] : 0;
    out[j] = b + v_media;
  }
}

int run_t2i(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: transformer t2i <dataset-dir> <out.ppm> <caption> [gpt-steps] [vq-steps]\n"
                    "  dataset-dir holds cap_0000.txt/img_0000.ppm, cap_0001.txt/img_0001.ppm, ...\n");
    return 2;
  }
  const char *dir = argv[2], *out = argv[3], *caption = argv[4];
  int gpt_steps = (argc > 5) ? atoi(argv[5]) : 1500;
  int vq_steps  = (argc > 6) ? atoi(argv[6]) : 1200;
  if (gpt_steps < 1) gpt_steps = 1500;
  if (vq_steps < 1)  vq_steps = 1200;

  const int T = T2I_CAPTION_LEN;
  const int v_media = T2I_K;      // image codes occupy [0, K)
  const int v_text = 256;         // byte-level caption tokens
  const int V = v_media + v_text;

  // --- load the (caption, image) dataset (keep the images for the VQ-VAE) ---
  Image *imgs = (Image*)malloc(T2I_MAX_PAIRS * sizeof(Image));
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
      if (w % T2I_PATCH || h % T2I_PATCH) { fprintf(stderr, "error: image dims must be divisible by %d\n", T2I_PATCH); image_free(&im); free(imgs); free(txttok); return 1; }
    } else if (im.w != w || im.h != h) {
      fprintf(stderr, "skip img_%04d.ppm: size mismatch\n", i); image_free(&im); continue;
    }
    imgs[N] = im;   // keep (freed at the end)
    txttok[N] = (int*)malloc((size_t)T * sizeof(int));
    encode_caption(cap, v_media, txttok[N]);
    N++;
  }
  if (N < 1) {
    fprintf(stderr, "error: no cap_0000.txt/img_0000.ppm pairs found in '%s'\n", dir);
    free(imgs); free(txttok);
    return 1;
  }

  // --- stage 1: train the VQ-VAE codec on the images, then encode each to codes ---
  VQVAE vq = vqvae_create(T2I_PATCH, imgs[0].c, T2I_LDIM, T2I_K);
  fprintf(stderr, "t2i: training VQ-VAE codec (%d imgs, %d steps)\n", N, vq_steps);
  vqvae_train(&vq, imgs, N, vq_steps, 0.01f);

  const int M = vqvae_ncodes(&vq, w, h);     // image code tokens (one per patch)
  const int total = T + M;
  int **codetok = (int**)malloc((size_t)N * sizeof(int*));
  for (int e = 0; e < N; e++) {
    codetok[e] = (int*)malloc((size_t)M * sizeof(int));
    vqvae_encode(&vq, &imgs[e], codetok[e]);
  }
  fprintf(stderr, "t2i: %d pairs, %dx%d -> %d codes/img (was %d color-cube px), vocab %d, GPT %d steps\n",
          N, w, h, M, w * h, V, gpt_steps);

  // --- stage 2: train the GPT on [caption | codes], supervising the codes ---
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
  for (int s = 0; s < gpt_steps; s++) {
    int e = rand() % N;
    for (int i = 0; i < T; i++) seq[i] = txttok[e][i];
    for (int j = 0; j < M; j++) seq[T + j] = codetok[e][j];
    for (int i = 0; i < total; i++) tgt[i] = -1;
    for (int k = 0; k < M; k++) tgt[T - 1 + k] = codetok[e][k];   // supervise the image codes

    gpt_forward_train(&g, seq, total, &a, &lo);
    float loss = cross_entropy_forward(&lo, tgt, &pr);
    lN = loss;
    tensor_zero_grad(&lo);
    cross_entropy_backward(&pr, tgt, &lo);
    gpt_backward(&g, seq, total, &a, &lo);
    adam_step(&opt, par, np);
    if (s % 100 == 0 || s == gpt_steps - 1)
      fprintf(stderr, "  gpt step %d/%d loss %.4f\n", s, gpt_steps, loss);
  }

  // --- generate codes for the query caption, then VQ-decode to an image ---
  int *qtext = (int*)malloc((size_t)T * sizeof(int));
  encode_caption(caption, v_media, qtext);
  KVCache kv = gpt_kvcache_create(&g);
  int gz[2] = {1, V};
  Tensor gl = tensor_create(gz, 2);
  for (int i = 0; i < T; i++) gpt_decode_step(&g, &kv, qtext[i], &gl);
  int *gen = (int*)malloc((size_t)M * sizeof(int));
  for (int k = 0; k < M; k++) {
    gen[k] = sample_argmax(&gl, 0);
    if (gen[k] >= v_media) gen[k] = v_media - 1;   // keep in the code range
    if (k < M - 1) gpt_decode_step(&g, &kv, gen[k], &gl);
  }
  Image og = vqvae_decode(&vq, gen, w, h);
  int ok = (image_save_ppm(out, &og) == 0);
  printf("t2i: %d pairs, VQ-VAE %d codes/img | GPT final loss %.4f | generated %dx%d for \"%s\" -> %s (%s)\n",
         N, M, lN, w, h, caption, out, ok ? "ok" : "FAILED");

  // --- cleanup ---
  image_free(&og); tensor_free(&gl); gpt_kvcache_free(&kv);
  free(gen); free(qtext);
  adam_free(&opt); tensor_free(&lo); tensor_free(&pr); gpt_acts_free(&a); gpt_free(&g);
  free(seq); free(tgt); free(par);
  for (int e = 0; e < N; e++) { free(codetok[e]); free(txttok[e]); image_free(&imgs[e]); }
  free(codetok); free(txttok); free(imgs);
  vqvae_free(&vq);
  return ok ? 0 : 1;
}
