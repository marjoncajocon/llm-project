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

#define T2I_CAPTION_LEN 48
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
// Procedural dataset: diverse (caption, image) pairs of colored shapes so the
// model learns real color+shape composition. VQGAN / DALL-E need a real dataset
// to generalize; this is a from-scratch, pure-C stand-in (no downloads).
//   transformer t2i-data <out-dir> [n] [size]
// ---------------------------------------------------------------------------
typedef struct { const char *name; float r, g, b; } NamedColor;

static const NamedColor T2I_COLORS[] = {
  {"red", 0.85f, 0.12f, 0.12f}, {"green", 0.13f, 0.70f, 0.20f},
  {"blue", 0.15f, 0.30f, 0.90f}, {"yellow", 0.95f, 0.85f, 0.15f},
  {"cyan", 0.15f, 0.80f, 0.85f}, {"magenta", 0.85f, 0.20f, 0.80f},
  {"orange", 0.95f, 0.55f, 0.10f}, {"purple", 0.55f, 0.20f, 0.75f},
};
static const char *T2I_SHAPES[] = {"circle", "square", "triangle", "diamond", "cross"};
static const char *T2I_RELS[] = {"left of", "right of", "above", "below"};
#define T2I_NCOLORS ((int)(sizeof(T2I_COLORS) / sizeof(T2I_COLORS[0])))
#define T2I_NSHAPES ((int)(sizeof(T2I_SHAPES) / sizeof(T2I_SHAPES[0])))
#define T2I_NRELS   ((int)(sizeof(T2I_RELS) / sizeof(T2I_RELS[0])))

// Is point (dx,dy) relative to the shape center inside the shape of radius rad?
static int shape_hit(int shape, float dx, float dy, float rad) {
  float ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy;
  switch (shape) {
    case 0: return dx * dx + dy * dy <= rad * rad;                 // circle
    case 1: return ax <= rad && ay <= rad;                         // square
    case 2:                                                        // triangle (apex up)
      if (dy < -rad || dy > rad) return 0;
      return ax <= rad * (dy + rad) / (2.0f * rad);
    case 3: return ax + ay <= rad;                                 // diamond
    case 4: return (ax <= rad * 0.34f && ay <= rad) ||             // cross / plus
                   (ay <= rad * 0.34f && ax <= rad);
  }
  return 0;
}

// tiny LCG so per-variant jitter is reproducible without touching global rand().
static float urand(unsigned int *s) {
  *s = *s * 1103515245u + 12345u;
  return (float)(*s >> 8) / 16777216.0f;   // [0,1)
}

// Paint one filled shape (color) into `im` centered at (cx,cy) with radius rad.
static void paint_shape(Image *im, int size, float cx, float cy, float rad,
                        int shape, const NamedColor *col) {
  for (int y = 0; y < size; y++)
    for (int x = 0; x < size; x++)
      if (shape_hit(shape, (x + 0.5f) - cx, (y + 0.5f) - cy, rad)) {
        size_t p = ((size_t)y * size + x) * 3;
        im->data[p] = col->r; im->data[p + 1] = col->g; im->data[p + 2] = col->b;
      }
}

// Two-object scene with a spatial relation, caption "a <c1> <s1> <rel> a <c2> <s2>".
// obj1 is placed relative to obj2 as stated: "left of"/"above" put obj1 in the
// left/top half, "right of"/"below" put it in the right/bottom half. Each object
// is jittered within its half so the model learns the RELATION concept (which side
// each object sits on) rather than memorizing fixed pixels.
static void draw_scene(const char *dir, int idx, int size,
                       int c1, int s1, int c2, int s2, int rel, unsigned int seed) {
  Image im; im.h = size; im.w = size; im.c = 3;
  im.data = (float*)malloc((size_t)size * size * 3 * sizeof(float));
  const float bg = 0.08f;                                // fixed dark background
  for (int i = 0; i < size * size * 3; i++) im.data[i] = bg;

  unsigned int rng = seed;
  float rad = size * (0.11f + 0.03f * urand(&rng));      // smaller: two per image
  float near = size * 0.27f, far = size * 0.73f;         // the two half-centers
  float jit = size * 0.08f;
  int horizontal = (rel == 0 || rel == 1);               // left of / right of
  int first_far = (rel == 1 || rel == 3);                // right of / below -> obj1 far
  float x1, y1, x2, y2;
  if (horizontal) {
    x1 = first_far ? far : near; x2 = first_far ? near : far;
    y1 = y2 = size * 0.5f;
  } else {
    y1 = first_far ? far : near; y2 = first_far ? near : far;
    x1 = x2 = size * 0.5f;
  }
  x1 += jit * (urand(&rng) - 0.5f); y1 += jit * (urand(&rng) - 0.5f);
  x2 += jit * (urand(&rng) - 0.5f); y2 += jit * (urand(&rng) - 0.5f);
  paint_shape(&im, size, x1, y1, rad, s1, &T2I_COLORS[c1]);
  paint_shape(&im, size, x2, y2, rad, s2, &T2I_COLORS[c2]);

  char ip[600], cp[600];
  snprintf(ip, sizeof(ip), "%s/img_%04d.ppm", dir, idx);
  snprintf(cp, sizeof(cp), "%s/cap_%04d.txt", dir, idx);
  image_save_ppm(ip, &im);
  FILE *cf = fopen(cp, "wb");
  if (cf) {
    fprintf(cf, "a %s %s %s a %s %s", T2I_COLORS[c1].name, T2I_SHAPES[s1],
            T2I_RELS[rel], T2I_COLORS[c2].name, T2I_SHAPES[s2]);
    fclose(cf);
  }
  image_free(&im);
}

int run_t2i_data(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: transformer t2i-data <out-dir> [n] [size]\n"
                    "  writes n two-object-scene (caption,image) pairs with spatial\n"
                    "  relations left/right/above/below (default 768, 24px)\n");
    return 2;
  }
  const char *dir = argv[2];
  int n = (argc > 3) ? atoi(argv[3]) : 768;
  int size = (argc > 4) ? atoi(argv[4]) : 24;
  if (size % T2I_PATCH) { fprintf(stderr, "error: size must be divisible by %d\n", T2I_PATCH); return 1; }
  ensure_dirs(dir);
  T2I_MKDIR(dir);
  // Randomly sample (color1,shape1,rel,color2,shape2) per index with a reproducible
  // per-index seed. The space (8*5*4*8*5 = 6400) is far larger than n, so the model
  // must GENERALIZE the relation across object pairings it never saw together.
  for (int i = 0; i < n; i++) {
    unsigned int r = 2654435761u * (unsigned)(i + 1) + 12345u;
    int c1 = (int)(urand(&r) * T2I_NCOLORS); if (c1 >= T2I_NCOLORS) c1 = T2I_NCOLORS - 1;
    int s1 = (int)(urand(&r) * T2I_NSHAPES); if (s1 >= T2I_NSHAPES) s1 = T2I_NSHAPES - 1;
    int rel = (int)(urand(&r) * T2I_NRELS);  if (rel >= T2I_NRELS) rel = T2I_NRELS - 1;
    int c2 = (int)(urand(&r) * T2I_NCOLORS); if (c2 >= T2I_NCOLORS) c2 = T2I_NCOLORS - 1;
    int s2 = (int)(urand(&r) * T2I_NSHAPES); if (s2 >= T2I_NSHAPES) s2 = T2I_NSHAPES - 1;
    draw_scene(dir, i, size, c1, s1, c2, s2, rel, 1000u + (unsigned)i * 2654435761u);
  }
  printf("t2i-data: wrote %d two-object scenes (%dx%d, %d shapes x %d colors x %d relations) -> %s\n",
         n, size, size, T2I_NSHAPES, T2I_NCOLORS, T2I_NRELS, dir);
  return 0;
}

// ---------------------------------------------------------------------------
// Build (train the VQ-VAE + GPT on a captioned-image folder)
// ---------------------------------------------------------------------------
static int t2i_save(const char *prefix, int w, int h, int T, const VQVAE *vq, const GPT *g);

// ckpt_prefix != NULL + ckpt_every > 0 saves the model every ckpt_every GPT steps
// (and after the last), so a long run survives being interrupted with usable weights.
static int t2i_build(const char *dir, int gpt_steps, int vq_steps,
                     const char *ckpt_prefix, int ckpt_every,
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

  // stage 2: GPT over [caption | codes] — sized to hold the color x shape x
  // relation (two-object) distribution while staying fast enough to converge.
  GPT g = gpt_create(V, 128, 4, 4, 512, total);
  GPTActs a = gpt_acts_create(&g, total);
  int lz[2] = {total, V};
  Tensor lo = tensor_create(lz, 2), pr = tensor_create(lz, 2);
  Tensor **par = (Tensor**)malloc((size_t)gpt_param_count(&g) * sizeof(Tensor*));
  int np = gpt_collect_params(&g, par);
  Adam opt = adam_create(par, np, 0.003f);
  int *seq = (int*)malloc((size_t)total * sizeof(int));
  int *tgt = (int*)malloc((size_t)total * sizeof(int));

  // Minibatch gradient accumulation: gpt_backward zeros+fills the param grads for
  // ONE example, so we sum each example's grads into a shadow buffer and average
  // before the optimizer step. Batch-1 SGD is too noisy to converge on the exact
  // per-position code grids (colors learn either way; sharp shapes need this).
  const int BATCH = 8;
  float **gacc = (float**)malloc((size_t)np * sizeof(float*));
  for (int p = 0; p < np; p++) gacc[p] = (float*)calloc((size_t)par[p]->size, sizeof(float));

  const float base_lr = opt.lr;
  srand(1234);
  for (int s = 0; s < gpt_steps; s++) {
    // Linear LR decay to 0.1x: high LR early to move fast, low LR late so the
    // model settles cleanly instead of bouncing around the minimum.
    opt.lr = base_lr * (1.0f - 0.9f * (float)s / (float)(gpt_steps > 1 ? gpt_steps - 1 : 1));
    for (int p = 0; p < np; p++) memset(gacc[p], 0, (size_t)par[p]->size * sizeof(float));
    float loss = 0.0f;
    for (int b = 0; b < BATCH; b++) {
      int e = rand() % N;
      for (int i = 0; i < T; i++) seq[i] = txttok[e][i];
      for (int j = 0; j < M; j++) seq[T + j] = codetok[e][j];
      for (int i = 0; i < total; i++) tgt[i] = -1;
      for (int k = 0; k < M; k++) tgt[T - 1 + k] = codetok[e][k];
      gpt_forward_train(&g, seq, total, &a, &lo);
      loss += cross_entropy_forward(&lo, tgt, &pr);
      tensor_zero_grad(&lo);
      cross_entropy_backward(&pr, tgt, &lo);
      gpt_backward(&g, seq, total, &a, &lo);
      for (int p = 0; p < np; p++) {
        const float *gs = par[p]->grad; float *gd = gacc[p]; int sz = par[p]->size;
        for (int i = 0; i < sz; i++) gd[i] += gs[i];
      }
    }
    const float inv = 1.0f / (float)BATCH;
    for (int p = 0; p < np; p++) {
      float *gd = par[p]->grad; const float *ga = gacc[p]; int sz = par[p]->size;
      for (int i = 0; i < sz; i++) gd[i] = ga[i] * inv;
    }
    adam_step(&opt, par, np);
    if (s % 50 == 0 || s == gpt_steps - 1)
      fprintf(stderr, "  gpt step %d/%d loss %.4f\n", s, gpt_steps, loss * inv);
    if (ckpt_prefix && ckpt_every > 0 && s > 0 && s % ckpt_every == 0) {
      if (t2i_save(ckpt_prefix, w, h, T, &vq, &g) == 0)
        fprintf(stderr, "  [checkpoint] saved %s at step %d\n", ckpt_prefix, s);
    }
  }

  for (int p = 0; p < np; p++) free(gacc[p]);
  free(gacc);

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
// Autoregressively sample the image-code grid from the caption, then VQ-decode.
// Sampling (temperature + top-k) over image codes ONLY — the media-token logits
// are copied into a masked row so a text token can never be chosen (DALL-E/VQGAN
// sample image tokens; greedy argmax collapses to one washed-out output).
static Image t2i_generate(const VQVAE *vq, GPT *g, int w, int h, int T,
                          const char *caption, float temp, int top_k, unsigned long long seed) {
  const int v_media = vq->K;
  const int M = vqvae_ncodes(vq, w, h);
  const int V = g->vocab_size;
  int *qtext = (int*)malloc((size_t)T * sizeof(int));
  encode_caption(caption, v_media, qtext);
  KVCache kv = gpt_kvcache_create(g);
  int gz[2] = {1, V}, mz[2] = {1, v_media};
  Tensor gl = tensor_create(gz, 2);
  Tensor ml = tensor_create(mz, 2);           // logits restricted to image codes
  Rng rng = rng_create(seed);
  for (int i = 0; i < T; i++) gpt_decode_step(g, &kv, qtext[i], &gl);
  int *gen = (int*)malloc((size_t)M * sizeof(int));
  for (int k = 0; k < M; k++) {
    for (int j = 0; j < v_media; j++) ml.data[j] = gl.data[j];
    gen[k] = sample_token(&ml, 0, temp, top_k, 1.0f, &rng);
    if (k < M - 1) gpt_decode_step(g, &kv, gen[k], &gl);
  }
  Image og = vqvae_decode(vq, gen, w, h);
  free(gen); free(qtext); tensor_free(&gl); tensor_free(&ml); gpt_kvcache_free(&kv);
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
  if (t2i_build(argv[2], gpt_steps, vq_steps, NULL, 0, &vq, &g, &w, &h, &T) != 0) return 1;
  Image og = t2i_generate(&vq, &g, w, h, T, argv[4], 0.6f, 40, 1234);
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
  int gpt_steps = (argc > 4) ? atoi(argv[4]) : 1800;  // each step = a minibatch
  int vq_steps = (argc > 5) ? atoi(argv[5]) : 1500;
  VQVAE vq; GPT g; int w, h, T;
  ensure_dirs(argv[3]);                               // needed before mid-run checkpoints
  // checkpoint every ~300 steps so a long/interruptible run keeps usable weights.
  if (t2i_build(argv[2], gpt_steps, vq_steps, argv[3], 300, &vq, &g, &w, &h, &T) != 0) return 1;
  int rc = t2i_save(argv[3], w, h, T, &vq, &g);
  printf("t2i-train: %s -> %s.gpt + %s.vqt (%s)\n", argv[2], argv[3], argv[3], rc == 0 ? "saved" : "FAILED");
  vqvae_free(&vq); gpt_free(&g);
  return rc;
}

int run_t2i_recon(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: transformer t2i-recon <model-prefix> <in.ppm> <out.ppm>\n");
    return 2;
  }
  VQVAE vq; GPT g; int w, h, T;
  if (t2i_load(argv[2], &w, &h, &T, &vq, &g) != 0) return 1;
  Image im = image_load_ppm(argv[3]);
  if (!im.data) { fprintf(stderr, "error: cannot load '%s'\n", argv[3]); vqvae_free(&vq); gpt_free(&g); return 1; }
  int nc = vqvae_ncodes(&vq, im.w, im.h);
  int *codes = (int*)malloc((size_t)nc * sizeof(int));
  vqvae_encode(&vq, &im, codes);
  int distinct = 0; char *seen = (char*)calloc((size_t)vq.K, 1);
  for (int i = 0; i < nc; i++) if (!seen[codes[i]]) { seen[codes[i]] = 1; distinct++; }
  Image og = vqvae_decode(&vq, codes, im.w, im.h);
  double err = 0.0; int npix = im.w * im.h * im.c;
  for (int i = 0; i < npix; i++) { double d = og.data[i] - im.data[i]; err += d * d; }
  err /= npix;
  int ok = (image_save_ppm(argv[4], &og) == 0);
  printf("t2i-recon: %dx%d | %d codes, %d distinct of K=%d | recon MSE %.5f -> %s (%s)\n",
         im.w, im.h, nc, distinct, vq.K, err, argv[4], ok ? "ok" : "FAILED");
  free(codes); free(seen); image_free(&og); image_free(&im); vqvae_free(&vq); gpt_free(&g);
  return ok ? 0 : 1;
}

int run_t2i_gen(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: transformer t2i-gen <model-prefix> <out.ppm> <caption> [temp] [seed]\n");
    return 2;
  }
  float temp = (argc > 5) ? (float)atof(argv[5]) : 0.6f;
  unsigned long long seed = (argc > 6) ? strtoull(argv[6], NULL, 10) : 1234ull;
  VQVAE vq; GPT g; int w, h, T;
  if (t2i_load(argv[2], &w, &h, &T, &vq, &g) != 0) return 1;
  Image og = t2i_generate(&vq, &g, w, h, T, argv[4], temp, 40, seed);
  int ok = (image_save_ppm(argv[3], &og) == 0);
  printf("t2i-gen: generated %dx%d for \"%s\" (temp %.2f) -> %s (%s)\n", w, h, argv[4], temp, argv[3], ok ? "ok" : "FAILED");
  image_free(&og); vqvae_free(&vq); gpt_free(&g);
  return ok ? 0 : 1;
}
