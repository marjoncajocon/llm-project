// t2i.c - dataset-trained text->image (autoregressive; no diffusion).
//
// Two learned stages from scratch: a VQ-VAE codec (vqvae.c) tokenizes each image
// into a short grid of discrete codes, and a GPT learns P(codes | caption) over
// [caption text | image codes]. Generation encodes the caption, autoregresses the
// code grid, and VQ-decodes to an image (the DALL-E / Parti recipe, tiny scale).
//
// Subcommands: `t2i` trains+generates in one shot; `t2i-train` trains and SAVES a
// model (VQ-VAE + GPT); `t2i-gen` loads a saved model and generates fast (no
// retraining) — that split is what the webchat uses.
#include "t2i.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "image.h"
#include "vqvae.h"
#include "model.h"
#include "loss.h"
#include "optim.h"
#include "sampler.h"
#include "checkpoint.h"

#ifdef _WIN32
#include <direct.h>
#define T2I_MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define T2I_MKDIR(p) mkdir(p, 0755)
#endif

#define T2I_CAPTION_LEN 32
#define T2I_MAX_PAIRS   4096
#define T2I_PATCH       2
#define T2I_LDIM        16
#define T2I_K           256      // codebook size = the image-code vocabulary
#define T2I_MAGIC       0x54324954u  // 'TI2T'

static int read_text_file(const char *path, char *buf, int cap) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  int n = (int)fread(buf, 1, (size_t)cap - 1, f);
  fclose(f);
  buf[n] = '\0';
  while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ')) buf[--n] = '\0';
  return n;
}

// Caption -> fixed-length text tokens, offset above the image-code vocab.
static void encode_caption(const char *s, int v_media, int *out) {
  int len = (int)strlen(s);
  for (int j = 0; j < T2I_CAPTION_LEN; j++) {
    int b = (j < len) ? (unsigned char)s[j] : 0;
    out[j] = b + v_media;
  }
}

// Make each directory prefix of `prefix` (so gpt_save's file can be written).
static void ensure_dirs(const char *prefix) {
  char buf[600];
  snprintf(buf, sizeof(buf), "%s", prefix);
  for (char *p = buf + 1; *p; p++) {
    if (*p == '/' || *p == '\\') { char c = *p; *p = '\0'; T2I_MKDIR(buf); *p = c; }
  }
}

// ---------------------------------------------------------------------------
// Build (train the VQ-VAE + GPT on a captioned-image folder)
// ---------------------------------------------------------------------------
static int t2i_build(const char *dir, int gpt_steps, int vq_steps,
                     VQVAE *vq_out, GPT *g_out, int *w_out, int *h_out, int *T_out) {
  const int T = T2I_CAPTION_LEN;
  const int v_media = T2I_K, v_text = 256, V = v_media + v_text;

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
      if (w % T2I_PATCH || h % T2I_PATCH) { fprintf(stderr, "error: dims must be divisible by %d\n", T2I_PATCH); image_free(&im); free(imgs); free(txttok); return 1; }
    } else if (im.w != w || im.h != h) {
      fprintf(stderr, "skip img_%04d.ppm: size mismatch\n", i); image_free(&im); continue;
    }
    imgs[N] = im;
    txttok[N] = (int*)malloc((size_t)T * sizeof(int));
    encode_caption(cap, v_media, txttok[N]);
    N++;
  }
  if (N < 1) {
    fprintf(stderr, "error: no cap_0000.txt/img_0000.ppm pairs in '%s'\n", dir);
    free(imgs); free(txttok); return 1;
  }

  // stage 1: VQ-VAE codec
  VQVAE vq = vqvae_create(T2I_PATCH, imgs[0].c, T2I_LDIM, T2I_K);
  fprintf(stderr, "t2i: training VQ-VAE codec (%d imgs, %d steps)\n", N, vq_steps);
  vqvae_train(&vq, imgs, N, vq_steps, 0.01f);

  const int M = vqvae_ncodes(&vq, w, h);
  const int total = T + M;
  int **codetok = (int**)malloc((size_t)N * sizeof(int*));
  for (int e = 0; e < N; e++) {
    codetok[e] = (int*)malloc((size_t)M * sizeof(int));
    vqvae_encode(&vq, &imgs[e], codetok[e]);
  }
  fprintf(stderr, "t2i: %d pairs, %dx%d -> %d codes/img, vocab %d, GPT %d steps\n",
          N, w, h, M, V, gpt_steps);

  // stage 2: GPT over [caption | codes]
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
  for (int s = 0; s < gpt_steps; s++) {
    int e = rand() % N;
    for (int i = 0; i < T; i++) seq[i] = txttok[e][i];
    for (int j = 0; j < M; j++) seq[T + j] = codetok[e][j];
    for (int i = 0; i < total; i++) tgt[i] = -1;
    for (int k = 0; k < M; k++) tgt[T - 1 + k] = codetok[e][k];
    gpt_forward_train(&g, seq, total, &a, &lo);
    float loss = cross_entropy_forward(&lo, tgt, &pr);
    tensor_zero_grad(&lo);
    cross_entropy_backward(&pr, tgt, &lo);
    gpt_backward(&g, seq, total, &a, &lo);
    adam_step(&opt, par, np);
    if (s % 100 == 0 || s == gpt_steps - 1)
      fprintf(stderr, "  gpt step %d/%d loss %.4f\n", s, gpt_steps, loss);
  }

  adam_free(&opt); tensor_free(&lo); tensor_free(&pr); gpt_acts_free(&a);
  free(seq); free(tgt); free(par);
  for (int e = 0; e < N; e++) { free(codetok[e]); free(txttok[e]); image_free(&imgs[e]); }
  free(codetok); free(txttok); free(imgs);

  *vq_out = vq; *g_out = g; *w_out = w; *h_out = h; *T_out = T;
  return 0;
}

// ---------------------------------------------------------------------------
// Generate an image from a caption using a trained (vq, gpt)
// ---------------------------------------------------------------------------
static Image t2i_generate(const VQVAE *vq, GPT *g, int w, int h, int T, const char *caption) {
  const int v_media = vq->K;
  const int M = vqvae_ncodes(vq, w, h);
  const int V = g->vocab_size;
  int *qtext = (int*)malloc((size_t)T * sizeof(int));
  encode_caption(caption, v_media, qtext);
  KVCache kv = gpt_kvcache_create(g);
  int gz[2] = {1, V};
  Tensor gl = tensor_create(gz, 2);
  for (int i = 0; i < T; i++) gpt_decode_step(g, &kv, qtext[i], &gl);
  int *gen = (int*)malloc((size_t)M * sizeof(int));
  for (int k = 0; k < M; k++) {
    gen[k] = sample_argmax(&gl, 0);
    if (gen[k] >= v_media) gen[k] = v_media - 1;
    if (k < M - 1) gpt_decode_step(g, &kv, gen[k], &gl);
  }
  Image og = vqvae_decode(vq, gen, w, h);
  free(gen); free(qtext); tensor_free(&gl); gpt_kvcache_free(&kv);
  return og;
}

// ---------------------------------------------------------------------------
// Checkpoint: <prefix>.gpt (via gpt_save) + <prefix>.vqt (config + VQ weights)
// ---------------------------------------------------------------------------
static int t2i_save(const char *prefix, int w, int h, int T, const VQVAE *vq, const GPT *g) {
  char gp[600], vp[600];
  snprintf(gp, sizeof(gp), "%s.gpt", prefix);
  snprintf(vp, sizeof(vp), "%s.vqt", prefix);
  if (gpt_save(gp, g) != 0) return 1;
  FILE *f = fopen(vp, "wb");
  if (!f) return 1;
  uint32_t magic = T2I_MAGIC;
  int32_t cfg[7] = {w, h, vq->patch, vq->channels, vq->ldim, vq->K, T};
  int ok = (fwrite(&magic, 4, 1, f) == 1) && (fwrite(cfg, 4, 7, f) == 7);
  ok = ok && (fwrite(vq->W_enc.data, sizeof(float), vq->W_enc.size, f) == vq->W_enc.size);
  ok = ok && (fwrite(vq->W_dec.data, sizeof(float), vq->W_dec.size, f) == vq->W_dec.size);
  ok = ok && (fwrite(vq->codebook.data, sizeof(float), vq->codebook.size, f) == vq->codebook.size);
  fclose(f);
  return ok ? 0 : 1;
}

static int t2i_load(const char *prefix, int *w, int *h, int *T, VQVAE *vq, GPT *g) {
  char gp[600], vp[600];
  snprintf(gp, sizeof(gp), "%s.gpt", prefix);
  snprintf(vp, sizeof(vp), "%s.vqt", prefix);
  FILE *f = fopen(vp, "rb");
  if (!f) { fprintf(stderr, "error: cannot open '%s'\n", vp); return 1; }
  uint32_t magic;
  int32_t cfg[7];
  if (fread(&magic, 4, 1, f) != 1 || magic != T2I_MAGIC || fread(cfg, 4, 7, f) != 7) {
    fprintf(stderr, "error: '%s' is not a t2i checkpoint\n", vp); fclose(f); return 1;
  }
  *w = cfg[0]; *h = cfg[1]; *T = cfg[6];
  *vq = vqvae_create(cfg[2], cfg[3], cfg[4], cfg[5]);
  int ok = (fread(vq->W_enc.data, sizeof(float), vq->W_enc.size, f) == vq->W_enc.size)
        && (fread(vq->W_dec.data, sizeof(float), vq->W_dec.size, f) == vq->W_dec.size)
        && (fread(vq->codebook.data, sizeof(float), vq->codebook.size, f) == vq->codebook.size);
  fclose(f);
  if (!ok) { fprintf(stderr, "error: '%s' truncated\n", vp); vqvae_free(vq); return 1; }
  if (gpt_load(gp, g) != 0) { vqvae_free(vq); return 1; }
  return 0;
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------
int run_t2i(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: transformer t2i <dataset-dir> <out.ppm> <caption> [gpt-steps] [vq-steps]\n");
    return 2;
  }
  int gpt_steps = (argc > 5) ? atoi(argv[5]) : 1500;
  int vq_steps = (argc > 6) ? atoi(argv[6]) : 1200;
  VQVAE vq; GPT g; int w, h, T;
  if (t2i_build(argv[2], gpt_steps, vq_steps, &vq, &g, &w, &h, &T) != 0) return 1;
  Image og = t2i_generate(&vq, &g, w, h, T, argv[4]);
  int ok = (image_save_ppm(argv[3], &og) == 0);
  printf("t2i: generated %dx%d for \"%s\" -> %s (%s)\n", w, h, argv[4], argv[3], ok ? "ok" : "FAILED");
  image_free(&og); vqvae_free(&vq); gpt_free(&g);
  return ok ? 0 : 1;
}

int run_t2i_train(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: transformer t2i-train <dataset-dir> <model-prefix> [gpt-steps] [vq-steps]\n"
                    "  saves <prefix>.gpt and <prefix>.vqt (train once, then t2i-gen many times)\n");
    return 2;
  }
  int gpt_steps = (argc > 4) ? atoi(argv[4]) : 2000;
  int vq_steps = (argc > 5) ? atoi(argv[5]) : 1500;
  VQVAE vq; GPT g; int w, h, T;
  if (t2i_build(argv[2], gpt_steps, vq_steps, &vq, &g, &w, &h, &T) != 0) return 1;
  ensure_dirs(argv[3]);
  int rc = t2i_save(argv[3], w, h, T, &vq, &g);
  printf("t2i-train: %s -> %s.gpt + %s.vqt (%s)\n", argv[2], argv[3], argv[3], rc == 0 ? "saved" : "FAILED");
  vqvae_free(&vq); gpt_free(&g);
  return rc;
}

int run_t2i_gen(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: transformer t2i-gen <model-prefix> <out.ppm> <caption>\n");
    return 2;
  }
  VQVAE vq; GPT g; int w, h, T;
  if (t2i_load(argv[2], &w, &h, &T, &vq, &g) != 0) return 1;
  Image og = t2i_generate(&vq, &g, w, h, T, argv[4]);
  int ok = (image_save_ppm(argv[3], &og) == 0);
  printf("t2i-gen: generated %dx%d for \"%s\" -> %s (%s)\n", w, h, argv[4], argv[3], ok ? "ok" : "FAILED");
  image_free(&og); vqvae_free(&vq); gpt_free(&g);
  return ok ? 0 : 1;
}
