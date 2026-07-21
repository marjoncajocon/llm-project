// main.c - Minimal GPT-2 Style Transformer in Pure C11 (demo entry point)
// Build: make        (see README.md and Makefile)
// Or:    gcc -std=c11 -O3 -march=native -o transformer *.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "tensor.h"
#include "loss.h"
#include "optim.h"
#include "sampler.h"
#include "model.h"
#include "checkpoint.h"
#include "tokenizer.h"
#include "train.h"
#include "agent.h"
#include "image.h"
#include "patch.h"
#include "video.h"
#include "audio.h"
#include "imagegen.h"
#include "audiogen.h"

// Portable "make directory if absent" (ignores already-exists).
#if defined(_WIN32)
#include <direct.h>
static void ensure_dir(const char *p) { _mkdir(p); }
#else
#include <sys/stat.h>
static void ensure_dir(const char *p) { mkdir(p, 0755); }
#endif

// A named model lives in model/<name>/ as model.bin + tok.bin. Fill the two
// path buffers (and create the folder when `make` is set, i.e. for training).
static void model_paths(const char *name, char *model_out, char *tok_out,
                        int cap, int make) {
  char dir[400];
  snprintf(dir, sizeof(dir), "model/%s", name);
  if (make) { ensure_dir("model"); ensure_dir(dir); }
  snprintf(model_out, cap, "%s/model.bin", dir);
  snprintf(tok_out, cap, "%s/tok.bin", dir);
}

// Conditional generation, minimal: train a GPT on the sequence
// [text tokens (offset by v_media)] ++ [media tokens], supervising the media
// positions (predict the next media token given the text prefix). Then feed the
// text prefix and autoregressively GENERATE the media tokens. Returns 1 if all M
// media tokens were reproduced. Used for text->image / text->audio / text->video
// (the only difference between modalities is the media codec).
static int cond_gen(const char *label, const int *text, int T,
                    const int *media, int M, int v_media, int v_text, int steps) {
  const int V = v_media + v_text;
  const int total = T + M;
  int *seq = (int*)malloc(total * sizeof(int));
  int *tgt = (int*)malloc(total * sizeof(int));
  for (int i = 0; i < T; i++) seq[i] = text[i] + v_media;   // text tokens live above media ids
  for (int j = 0; j < M; j++) seq[T + j] = media[j];
  for (int i = 0; i < total; i++) tgt[i] = -1;
  for (int k = 0; k < M; k++) tgt[T - 1 + k] = media[k];    // predict media_k after the prefix

  GPT g = gpt_create(V, 48, 3, 2, 192, total);
  GPTActs a = gpt_acts_create(&g, total);
  int lz[2] = {total, V};
  Tensor lo = tensor_create(lz, 2), pr = tensor_create(lz, 2);
  Tensor *par[256];
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
  }

  // Generate: feed the text prefix, then decode M media tokens greedily.
  KVCache kv = gpt_kvcache_create(&g);
  int gz[2] = {1, V};
  Tensor gl = tensor_create(gz, 2);
  for (int i = 0; i < T; i++) gpt_decode_step(&g, &kv, seq[i], &gl);
  int *gen = (int*)malloc(M * sizeof(int));
  for (int k = 0; k < M; k++) {
    gen[k] = sample_argmax(&gl, 0);
    if (k < M - 1) gpt_decode_step(&g, &kv, gen[k], &gl);
  }
  int match = 0;
  for (int k = 0; k < M; k++) if (gen[k] == media[k]) match++;
  printf("    %-12s loss %.4f -> %.4f | generated %d/%d media tokens -> %s\n",
         label, l0, lN, match, M, match == M ? "OK" : "FAIL");

  free(gen); tensor_free(&gl); gpt_kvcache_free(&kv);
  adam_free(&opt); tensor_free(&lo); tensor_free(&pr); gpt_acts_free(&a); gpt_free(&g);
  free(seq); free(tgt);
  return match == M;
}

// Media -> text (captioning / transcription): train a GPT to predict `caption`
// (C text tokens) from the precomputed media embeddings (media_emb: n_media x
// d_model), then autoregressively GENERATE the caption from the media alone.
// Returns 1 if all C caption tokens were reproduced. (media_emb's projection is
// fixed here; jointly training it is E4's job — this demonstrates text decoding.)
static int media_to_text(const char *label, const Tensor *media_emb, int n_media,
                         const int *caption, int C, int v_text, int steps) {
  const int dm = media_emb->shape[1];
  const int total = n_media + C;

  GPT g = gpt_create(v_text, dm, 2, 2, dm * 4, total);
  GPTActs a = gpt_acts_create(&g, total);
  int lz[2] = {total, v_text};
  Tensor lo = tensor_create(lz, 2), pr = tensor_create(lz, 2);
  int *tgt = (int*)malloc(total * sizeof(int));
  for (int i = 0; i < total; i++) tgt[i] = -1;
  for (int k = 0; k < C; k++) tgt[n_media - 1 + k] = caption[k];   // predict caption after media
  Tensor *par[128];
  int np = gpt_collect_params(&g, par);
  Adam opt = adam_create(par, np, 0.01f);

  float l0 = 0.0f, lN = 0.0f;
  for (int s = 0; s < steps; s++) {
    gpt_forward_multimodal(&g, media_emb, caption, n_media, C, &a, &lo);
    float loss = cross_entropy_forward(&lo, tgt, &pr);
    if (s == 0) l0 = loss;
    lN = loss;
    tensor_zero_grad(&lo);
    cross_entropy_backward(&pr, tgt, &lo);
    gpt_backward_multimodal(&g, caption, n_media, C, &a, &lo, (Tensor*)media_emb);
    adam_step(&opt, par, np);
  }
  gpt_acts_free(&a); tensor_free(&lo); tensor_free(&pr); adam_free(&opt); free(tgt);

  // Autoregressively generate the caption from the media prefix (grow one token
  // at a time; acts are resized per step since the sequence length changes).
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
  for (int t = 0; t < C; t++) if (gen[t] == caption[t]) match++;
  printf("    %-12s loss %.4f -> %.4f | generated %d/%d caption tokens -> %s\n",
         label, l0, lN, match, C, match == C ? "OK" : "FAIL");

  free(gen); gpt_free(&g);
  return match == C;
}

// Full-model loss: training forward + cross-entropy. Fills logits and probs.
static float gpt_loss(GPT *m, GPTActs *a, const int *ids, int seq_len,
                      const int *targets, Tensor *logits, Tensor *probs) {
  gpt_forward_train(m, ids, seq_len, a, logits);
  return cross_entropy_forward(logits, targets, probs);
}

static int run_demo(void) {
  printf("=== Minimal GPT-2 Style Transformer Demo ===\n\n");
  int all_ok = 1;
  srand(42);

  // --------------------------------------------------------------------------
  // Part 1: tokenizer + checkpoint round-trips (step 4 code, condensed)
  // --------------------------------------------------------------------------
  printf("[1] Tokenizer + checkpoint round-trips\n");
  // Train BPE on a sample text, check round-trip + compression, then save/load.
  const char *sample = "the cat sat on the mat the cat ran to the cat";
  Tokenizer tok = tokenizer_create_bytelevel();
  tokenizer_train(&tok, sample, 30);
  int tids[256], tids2[256];
  int n_tids = tokenizer_encode(&tok, sample, tids, 256);
  char decoded[512];
  tokenizer_decode(&tok, tids, n_tids, decoded, sizeof(decoded));
  int rt_ok = (strcmp(sample, decoded) == 0);
  int comp_ok = (n_tids < (int)strlen(sample));

  int save_load_ok = (tokenizer_save(&tok, "tok.bin") == 0);
  Tokenizer tok2;
  save_load_ok = save_load_ok && (tokenizer_load(&tok2, "tok.bin") == 0);
  if (save_load_ok) {
    int n2 = tokenizer_encode(&tok2, sample, tids2, 256);
    save_load_ok = (n2 == n_tids) && (memcmp(tids, tids2, n_tids * sizeof(int)) == 0);
    tokenizer_free(&tok2);
  }
  remove("tok.bin");
  int tok_ok = rt_ok && comp_ok && save_load_ok;
  printf("    tokenizer: train %d bytes -> %d tokens | round-trip %s | save/load %s\n",
         (int)strlen(sample), n_tids, rt_ok ? "OK" : "FAIL",
         save_load_ok ? "OK" : "FAIL");
  tokenizer_free(&tok);
  all_ok &= tok_ok;

  GPT probe = gpt_create(16, 8, 2, 2, 32, 16);
  int pseq[3] = {1, 5, 9};
  int plshape[2] = {3, 16};
  Tensor pla = tensor_create(plshape, 2), plb = tensor_create(plshape, 2);
  gpt_forward(&probe, pseq, 3, &pla);
  int ckpt_ok = (gpt_save("model.bin", &probe) == 0);
  GPT probe2;
  ckpt_ok = ckpt_ok && (gpt_load("model.bin", &probe2) == 0);
  if (ckpt_ok) {
    gpt_forward(&probe2, pseq, 3, &plb);
    ckpt_ok = (memcmp(pla.data, plb.data, pla.size * sizeof(float)) == 0);
    gpt_free(&probe2);
  }
  tensor_free(&pla); tensor_free(&plb); gpt_free(&probe); remove("model.bin");
  printf("    checkpoint: save/load logits identical -> %s\n\n",
         ckpt_ok ? "OK" : "FAIL");
  all_ok &= ckpt_ok;

  // --------------------------------------------------------------------------
  // Part 2: build the full GPT + a fixed next-token task
  // --------------------------------------------------------------------------
  const int vocab = 16, d_model = 8, n_heads = 2, n_layers = 2, d_ff = 32, max_seq = 16;
  const int seq_len = 6;
  GPT model = gpt_create(vocab, d_model, n_heads, n_layers, d_ff, max_seq);
  printf("[2] Full GPT training  (vocab=%d d_model=%d heads=%d layers=%d)\n",
         vocab, d_model, n_heads, n_layers);

  // Fixed token sequence; predict the next token at each position.
  int tokens[7] = {1, 2, 4, 8, 3, 5, 7};
  const int *ids = tokens;            // inputs  = tokens[0..seq_len-1]
  const int *targets = tokens + 1;    // targets = tokens[1..seq_len]

  GPTActs acts = gpt_acts_create(&model, seq_len);
  int lshape[2] = {seq_len, vocab};
  Tensor logits = tensor_create(lshape, 2);
  Tensor probs = tensor_create(lshape, 2);

  Tensor *params[64];
  int n_params = gpt_collect_params(&model, params);

  // --------------------------------------------------------------------------
  // Part 3: end-to-end gradient check (analytic vs numerical), whole model
  // --------------------------------------------------------------------------
  gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
  tensor_zero_grad(&logits);
  cross_entropy_backward(&probs, targets, &logits);
  gpt_backward(&model, ids, seq_len, &acts, &logits);  // analytic grads into params

  const float eps = 1e-2f;
  float max_rel = 0.0f, max_rel_cond = 0.0f;
  for (int p = 0; p < n_params; p++) {
    Tensor *param = params[p];
    int step = param->size > 2 ? param->size / 2 : 1;
    for (int k = 0; k < param->size; k += step) {
      float base = param->data[k];
      param->data[k] = base + eps;
      float lp = gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
      param->data[k] = base - eps;
      float lm = gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
      param->data[k] = base;
      float numeric = (lp - lm) / (2.0f * eps);
      float analytic = param->grad[k];
      float rel = fabsf(analytic - numeric) / (fabsf(analytic) + fabsf(numeric) + 1e-8f);
      if (rel > max_rel) max_rel = rel;
      // "conditioned" entries: gradient large enough that float32 finite differences
      // are trustworthy (avoids the O(1)-loss cancellation noise floor).
      if (fabsf(analytic) > 1e-2f && rel > max_rel_cond) max_rel_cond = rel;
    }
  }
  int grad_ok = (max_rel_cond < 1e-2f);
  printf("    gradient check: max rel error %.2e (all) | %.2e (|grad|>1e-2) -> %s\n",
         max_rel, max_rel_cond, grad_ok ? "OK" : "FAIL");
  all_ok &= grad_ok;

  // --------------------------------------------------------------------------
  // Part 4: train (Adam) on the next-token task, expect the loss to drop
  // --------------------------------------------------------------------------
  Adam opt = adam_create(params, n_params, 0.02f);
  float first_loss = gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
  const int steps = 600;
  for (int s = 0; s < steps; s++) {
    float loss = gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
    tensor_zero_grad(&logits);
    cross_entropy_backward(&probs, targets, &logits);
    gpt_backward(&model, ids, seq_len, &acts, &logits);
    adam_step(&opt, params, n_params);
    if (s % 120 == 0) printf("    step %3d  loss %.4f\n", s, loss);
  }
  float final_loss = gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
  int train_ok = (final_loss < first_loss * 0.25f);
  printf("    start %.4f -> final %.4f -> %s\n\n",
         first_loss, final_loss, train_ok ? "OK (loss dropped)" : "FAIL");
  all_ok &= train_ok;

  // --------------------------------------------------------------------------
  // Part 5: sampling sanity checks (deterministic)
  // --------------------------------------------------------------------------
  Rng rng = rng_create(123);
  int topk1_ok = 1;
  for (int r = 0; r < seq_len; r++) {
    // top_k=1 must always select the argmax token
    if (sample_token(&logits, r, 1.0f, 1, 1.0f, &rng) != sample_argmax(&logits, r)) {
      topk1_ok = 0;
      break;
    }
  }
  // Same seed -> same token (reproducibility)
  Rng ra = rng_create(7), rb = rng_create(7);
  int seed_ok = (sample_token(&logits, 0, 0.8f, 8, 0.95f, &ra) ==
                 sample_token(&logits, 0, 0.8f, 8, 0.95f, &rb));
  int sample_ok = topk1_ok && seed_ok;
  printf("[3] Sampling: top_k=1==argmax %s | seeded reproducible %s\n\n",
         topk1_ok ? "OK" : "FAIL", seed_ok ? "OK" : "FAIL");
  all_ok &= sample_ok;

  // --------------------------------------------------------------------------
  // Part 6: KV-cache decode must match the non-cached forward (logits row t)
  // --------------------------------------------------------------------------
  KVCache kv = gpt_kvcache_create(&model);
  int slog_shape[2] = {1, vocab};
  Tensor slog = tensor_create(slog_shape, 2);
  float kv_max_diff = 0.0f;
  for (int t = 0; t < seq_len; t++) {
    gpt_decode_step(&model, &kv, ids[t], &slog);
    for (int j = 0; j < vocab; j++) {
      float diff = fabsf(slog.data[j] - logits.data[t * vocab + j]);
      if (diff > kv_max_diff) kv_max_diff = diff;
    }
  }
  int kv_ok = (kv_max_diff < 1e-3f);
  printf("[4] KV cache: max |logit diff| vs full forward = %.2e -> %s\n\n",
         kv_max_diff, kv_ok ? "OK" : "FAIL");
  all_ok &= kv_ok;
  tensor_free(&slog);
  gpt_kvcache_free(&kv);

  // --------------------------------------------------------------------------
  // Part 7: image patch embedding (Phase E: image -> sequence of vectors)
  // --------------------------------------------------------------------------
  FILE *pf = fopen("img.ppm.tmp", "wb");
  if (pf) {
    fprintf(pf, "P6\n4 4\n255\n");
    for (int i = 0; i < 4 * 4 * 3; i++) fputc((unsigned char)((i * 7) & 0xFF), pf);
    fclose(pf);
  }
  Image im = image_load_ppm("img.ppm.tmp");
  int img_ok = (im.data != NULL && im.h == 4 && im.w == 4);
  if (img_ok) {
    const int patch = 2;
    int np = patch_count(&im, patch);              // 4 patches
    int pd = patch_dim(&im, patch);                // 2*2*3 = 12
    int ws[2] = {pd, d_model};
    Tensor Wp = tensor_create(ws, 2); tensor_init_random(&Wp);  // patch projection
    int bs[1] = {d_model};  Tensor bp = tensor_create(bs, 1);
    int os[2] = {np, d_model}; Tensor pemb = tensor_create(os, 2);

    // Caption tokens; supervise text positions to predict the NEXT caption token
    // (image positions + the last position are ignored via target -1).
    int cap[3] = {2, 3, 5};
    int mm_seq = 3, total = np + mm_seq;
    int tgt[64];
    for (int i = 0; i < total; i++) tgt[i] = -1;
    for (int t = 0; t < mm_seq - 1; t++) tgt[np + t] = cap[t + 1];

    GPTActs amm = gpt_acts_create(&model, total);
    int mmls[2] = {total, vocab};
    Tensor mmlog = tensor_create(mmls, 2);
    Tensor mmprobs = tensor_create(mmls, 2);

    // E3: forward [image patches] + [text tokens] through the GPT.
    patch_embed(&im, patch, &Wp, &bp, &pemb);
    gpt_forward_multimodal(&model, &pemb, cap, np, mm_seq, &amm, &mmlog);
    int finite = 1;
    for (int i = 0; i < mmlog.size; i++)
      if (isnan(mmlog.data[i]) || isinf(mmlog.data[i])) { finite = 0; break; }
    int e3_ok = (mmlog.size == total * vocab) && finite;
    printf("[5] Image: %d patches + %d text -> GPT multimodal logits (%d, %d) -> %s\n",
           np, mm_seq, total, vocab, e3_ok ? "OK" : "FAIL");

    // E4: overfit this image -> caption (trains GPT params + patch projection).
    Tensor *pm[64];
    int npar = gpt_collect_params(&model, pm);
    pm[npar++] = &Wp; pm[npar++] = &bp;
    Adam mopt = adam_create(pm, npar, 0.02f);
    float first = 0.0f, last = 0.0f;
    const int msteps = 150;
    for (int s = 0; s < msteps; s++) {
      patch_embed(&im, patch, &Wp, &bp, &pemb);    // recompute (Wp changes)
      gpt_forward_multimodal(&model, &pemb, cap, np, mm_seq, &amm, &mmlog);
      float loss = cross_entropy_forward(&mmlog, tgt, &mmprobs);
      if (s == 0) first = loss;
      last = loss;
      tensor_zero_grad(&mmlog);
      cross_entropy_backward(&mmprobs, tgt, &mmlog);
      tensor_zero_grad(&Wp); tensor_zero_grad(&bp);
      gpt_backward_multimodal(&model, cap, np, mm_seq, &amm, &mmlog, &pemb);
      patch_embed_backward(&im, patch, &Wp, &bp, &pemb);
      adam_step(&mopt, pm, npar);
    }
    int e4_ok = (last < first * 0.5f);
    printf("[6] Multimodal train: image->caption loss %.4f -> %.4f -> %s\n\n",
           first, last, e4_ok ? "OK" : "FAIL");
    img_ok = e3_ok && e4_ok;

    adam_free(&mopt);
    tensor_free(&mmprobs); tensor_free(&mmlog); gpt_acts_free(&amm);
    tensor_free(&Wp); tensor_free(&bp); tensor_free(&pemb);
  } else {
    printf("[5] Image: load FAIL\n\n");
  }
  image_free(&im);
  remove("img.ppm.tmp");
  all_ok &= img_ok;

  // --------------------------------------------------------------------------
  // Part 8: video (Phase F) — frames -> temporal patch embeddings -> train
  // --------------------------------------------------------------------------
  ensure_dir("vid_tmp");
  for (int fr = 0; fr < 2; fr++) {
    char vp[64];
    snprintf(vp, sizeof(vp), "vid_tmp/frame_%04d.ppm", fr);
    FILE *vfp = fopen(vp, "wb");
    if (vfp) {
      fprintf(vfp, "P6\n4 4\n255\n");
      for (int i = 0; i < 4 * 4 * 3; i++) fputc((unsigned char)(((i * 7) + fr * 40) & 0xFF), vfp);
      fclose(vfp);
    }
  }
  Video vid = video_load("vid_tmp");
  int vid_ok = (vid.n == 2);
  if (vid_ok) {
    const int patch = 2;
    int vpc = video_patch_count(&vid, patch);          // 2 frames * 4 = 8
    int vpd = patch_dim(&vid.frames[0], patch);        // 12
    int vws[2] = {vpd, d_model}; Tensor Wv = tensor_create(vws, 2); tensor_init_random(&Wv);
    int vbs[1] = {d_model};      Tensor bv = tensor_create(vbs, 1);
    int ves[2] = {vpc, d_model}; Tensor vemb = tensor_create(ves, 2);

    int vcap[3] = {4, 6, 8};
    int vseq = 3, vtot = vpc + vseq;                   // 8 + 3 = 11 <= max_seq 16
    int vtgt[64];
    for (int i = 0; i < vtot; i++) vtgt[i] = -1;
    for (int t = 0; t < vseq - 1; t++) vtgt[vpc + t] = vcap[t + 1];

    GPTActs va = gpt_acts_create(&model, vtot);
    int vls[2] = {vtot, vocab};
    Tensor vlog = tensor_create(vls, 2), vprb = tensor_create(vls, 2);
    Tensor *vpar[64];
    int nvp = gpt_collect_params(&model, vpar);
    vpar[nvp++] = &Wv; vpar[nvp++] = &bv;
    Adam vopt = adam_create(vpar, nvp, 0.02f);

    float v0 = 0.0f, vl = 0.0f;
    for (int s = 0; s < 120; s++) {
      video_embed(&vid, patch, &Wv, &bv, &vemb);
      gpt_forward_multimodal(&model, &vemb, vcap, vpc, vseq, &va, &vlog);
      float loss = cross_entropy_forward(&vlog, vtgt, &vprb);
      if (s == 0) v0 = loss;
      vl = loss;
      tensor_zero_grad(&vlog);
      cross_entropy_backward(&vprb, vtgt, &vlog);
      tensor_zero_grad(&Wv); tensor_zero_grad(&bv);
      gpt_backward_multimodal(&model, vcap, vpc, vseq, &va, &vlog, &vemb);
      video_embed_backward(&vid, patch, &Wv, &bv, &vemb);
      adam_step(&vopt, vpar, nvp);
    }
    vid_ok = (vpc == 8) && (vl < v0 * 0.5f);
    printf("[7] Video: %d frames -> %d patches + %d text | train loss %.4f -> %.4f -> %s\n\n",
           vid.n, vpc, vseq, v0, vl, vid_ok ? "OK" : "FAIL");

    adam_free(&vopt);
    tensor_free(&vlog); tensor_free(&vprb); gpt_acts_free(&va);
    tensor_free(&Wv); tensor_free(&bv); tensor_free(&vemb);
  } else {
    printf("[7] Video: load FAIL\n\n");
  }
  video_free(&vid);
  remove("vid_tmp/frame_0000.ppm");
  remove("vid_tmp/frame_0001.ppm");
  all_ok &= vid_ok;

  // --------------------------------------------------------------------------
  // Part 9: audio (Phase F) — waveform -> framed embeddings -> train
  // --------------------------------------------------------------------------
  FILE *awf = fopen("aud.wav.tmp", "wb");
  if (awf) {
    int nsamp = 256; uint32_t dsz = (uint32_t)(nsamp * 2), x; uint16_t u16;
    fwrite("RIFF", 1, 4, awf); x = 36 + dsz; fwrite(&x, 4, 1, awf); fwrite("WAVE", 1, 4, awf);
    fwrite("fmt ", 1, 4, awf); x = 16; fwrite(&x, 4, 1, awf);
    u16 = 1; fwrite(&u16, 2, 1, awf); u16 = 1; fwrite(&u16, 2, 1, awf);     // PCM, mono
    x = 8000; fwrite(&x, 4, 1, awf); x = 16000; fwrite(&x, 4, 1, awf);      // rate, byte rate
    u16 = 2; fwrite(&u16, 2, 1, awf); u16 = 16; fwrite(&u16, 2, 1, awf);    // block align, bits
    fwrite("data", 1, 4, awf); fwrite(&dsz, 4, 1, awf);
    for (int i = 0; i < nsamp; i++) {
      int16_t s = (int16_t)(sinf(i * 0.1f) * 10000.0f);
      fwrite(&s, 2, 1, awf);
    }
    fclose(awf);
  }
  Audio aud = audio_load_wav("aud.wav.tmp");
  int aud_ok = (aud.samples != NULL && aud.n == 256);
  if (aud_ok) {
    const int fsz = 64, hop = 64;
    int afc = audio_frame_count(&aud, fsz, hop);        // (256-64)/64 + 1 = 4
    int aws[2] = {fsz, d_model}; Tensor Wa = tensor_create(aws, 2); tensor_init_random(&Wa);
    int abz[1] = {d_model};      Tensor bva = tensor_create(abz, 1);
    int aez[2] = {afc, d_model}; Tensor aemb = tensor_create(aez, 2);

    int acap[3] = {5, 7, 9};
    int aseq = 3, atot = afc + aseq;                    // 4 + 3 = 7
    int atgt[64];
    for (int i = 0; i < atot; i++) atgt[i] = -1;
    for (int t = 0; t < aseq - 1; t++) atgt[afc + t] = acap[t + 1];

    GPTActs aa = gpt_acts_create(&model, atot);
    int alz[2] = {atot, vocab};
    Tensor alog = tensor_create(alz, 2), aprb = tensor_create(alz, 2);
    Tensor *apar[64];
    int nap = gpt_collect_params(&model, apar);
    apar[nap++] = &Wa; apar[nap++] = &bva;
    Adam aopt = adam_create(apar, nap, 0.02f);

    float a0 = 0.0f, al = 0.0f;
    for (int s = 0; s < 120; s++) {
      audio_embed(&aud, fsz, hop, &Wa, &bva, &aemb);
      gpt_forward_multimodal(&model, &aemb, acap, afc, aseq, &aa, &alog);
      float loss = cross_entropy_forward(&alog, atgt, &aprb);
      if (s == 0) a0 = loss;
      al = loss;
      tensor_zero_grad(&alog);
      cross_entropy_backward(&aprb, atgt, &alog);
      tensor_zero_grad(&Wa); tensor_zero_grad(&bva);
      gpt_backward_multimodal(&model, acap, afc, aseq, &aa, &alog, &aemb);
      audio_embed_backward(&aud, fsz, hop, &Wa, &bva, &aemb);
      adam_step(&aopt, apar, nap);
    }
    aud_ok = (afc == 4) && (al < a0 * 0.5f);
    printf("[8] Audio: %d frames + %d text | train loss %.4f -> %.4f -> %s\n\n",
           afc, aseq, a0, al, aud_ok ? "OK" : "FAIL");

    adam_free(&aopt);
    tensor_free(&alog); tensor_free(&aprb); gpt_acts_free(&aa);
    tensor_free(&Wa); tensor_free(&bva); tensor_free(&aemb);
  } else {
    printf("[8] Audio: load FAIL\n\n");
  }
  audio_free(&aud);
  remove("aud.wav.tmp");
  all_ok &= aud_ok;

  // --------------------------------------------------------------------------
  // Part 10: image GENERATION (Phase G, GI minimal) — pixels -> tokens, train a
  // GPT to autoregress them, then generate and decode back to pixels.
  // --------------------------------------------------------------------------
  {
    const int LV = 4;                          // 4 levels/channel -> 64 color tokens
    int gv = imagegen_vocab(LV);
    const int giw = 8, gih = 8, P = giw * gih; // 64 pixels = 64 tokens
    Image gim; gim.h = gih; gim.w = giw; gim.c = 3;
    gim.data = (float*)malloc((size_t)P * 3 * sizeof(float));
    for (int y = 0; y < gih; y++)
      for (int x = 0; x < giw; x++) {
        int i = y * giw + x;
        gim.data[i * 3 + 0] = (float)x / (giw - 1);   // R gradient across x
        gim.data[i * 3 + 1] = (float)y / (gih - 1);   // G gradient across y
        gim.data[i * 3 + 2] = 0.5f;
      }
    int itok[64];
    image_to_tokens(&gim, LV, itok);

    // Train a small GPT to memorize the image's token sequence.
    GPT gg = gpt_create(gv, 32, 2, 2, 128, P);
    GPTActs ga = gpt_acts_create(&gg, P);
    int ilz[2] = {P, gv};
    Tensor ilog = tensor_create(ilz, 2), iprb = tensor_create(ilz, 2);
    int itgt[64];
    for (int i = 0; i < P; i++) itgt[i] = (i < P - 1) ? itok[i + 1] : -1;
    Tensor *ipar[128];
    int nip = gpt_collect_params(&gg, ipar);
    Adam iopt = adam_create(ipar, nip, 0.01f);
    float i0 = 0.0f, il = 0.0f;
    for (int s = 0; s < 400; s++) {
      gpt_forward_train(&gg, itok, P, &ga, &ilog);
      float loss = cross_entropy_forward(&ilog, itgt, &iprb);
      if (s == 0) i0 = loss;
      il = loss;
      tensor_zero_grad(&ilog);
      cross_entropy_backward(&iprb, itgt, &ilog);
      gpt_backward(&gg, itok, P, &ga, &ilog);
      adam_step(&iopt, ipar, nip);
    }

    // Autoregressively GENERATE the image tokens (greedy) from the first pixel.
    KVCache ikv = gpt_kvcache_create(&gg);
    int gtok[64]; gtok[0] = itok[0];
    int glz[2] = {1, gv};
    Tensor gl = tensor_create(glz, 2);
    gpt_decode_step(&gg, &ikv, itok[0], &gl);
    for (int i = 1; i < P; i++) {
      gtok[i] = sample_argmax(&gl, 0);
      gpt_decode_step(&gg, &ikv, gtok[i], &gl);
    }
    int match = 0;
    for (int i = 0; i < P; i++) if (gtok[i] == itok[i]) match++;
    Image gout = tokens_to_image(gtok, giw, gih, LV);   // decode -> pixels
    int gen_ok = (match == P) && (gout.data != NULL);
    printf("[10] Image GEN: train loss %.4f -> %.4f | generated %d/%d tokens match -> %s\n\n",
           i0, il, match, P, gen_ok ? "OK" : "FAIL");

    image_free(&gout); tensor_free(&gl); gpt_kvcache_free(&ikv);
    adam_free(&iopt); tensor_free(&ilog); tensor_free(&iprb); gpt_acts_free(&ga);
    gpt_free(&gg); image_free(&gim);
    all_ok &= gen_ok;
  }

  // --------------------------------------------------------------------------
  // Part 11: CONDITIONAL generation (Phase G) — text -> image / audio / video.
  // Same mechanism for all three; only the media codec differs.
  // --------------------------------------------------------------------------
  printf("[11] Conditional generation (text -> media)\n");
  int cg_ok = 1;
  const int LV = 4, IMG_V = imagegen_vocab(LV);   // 64 color tokens
  const int TXT_V = 16;

  // text -> image: a 4x4 image = 16 tokens
  {
    Image im4; im4.h = 4; im4.w = 4; im4.c = 3;
    im4.data = (float*)malloc(16 * 3 * sizeof(float));
    for (int y = 0; y < 4; y++)
      for (int x = 0; x < 4; x++) {
        int i = y * 4 + x;
        im4.data[i * 3 + 0] = (float)x / 3.0f;
        im4.data[i * 3 + 1] = (float)y / 3.0f;
        im4.data[i * 3 + 2] = 0.3f;
      }
    int mt[16]; image_to_tokens(&im4, LV, mt);
    int txt[2] = {1, 2};
    cg_ok &= cond_gen("text->image", txt, 2, mt, 16, IMG_V, TXT_V, 300);
    image_free(&im4);
  }
  // text -> audio: 32 mu-law samples = 32 tokens
  {
    Audio a32; a32.n = 32; a32.sample_rate = 8000;
    a32.samples = (float*)malloc(32 * sizeof(float));
    for (int i = 0; i < 32; i++) a32.samples[i] = sinf(i * 0.3f) * 0.7f;
    int mt[32]; audio_to_tokens(&a32, mt);
    int txt[2] = {3, 4};
    cg_ok &= cond_gen("text->audio", txt, 2, mt, 32, audiogen_vocab(), TXT_V, 300);
    audio_free(&a32);
  }
  // text -> video: 2 frames of 4x4 = 32 image tokens
  {
    int mt[32];
    for (int f = 0; f < 2; f++) {
      Image fr; fr.h = 4; fr.w = 4; fr.c = 3;
      fr.data = (float*)malloc(16 * 3 * sizeof(float));
      for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
          int i = y * 4 + x;
          fr.data[i * 3 + 0] = (float)x / 3.0f;
          fr.data[i * 3 + 1] = (float)(f) / 1.0f;   // frame differs over time
          fr.data[i * 3 + 2] = (float)y / 3.0f;
        }
      int ft[16]; image_to_tokens(&fr, LV, ft);
      for (int i = 0; i < 16; i++) mt[f * 16 + i] = ft[i];
      image_free(&fr);
    }
    int txt[2] = {5, 6};
    cg_ok &= cond_gen("text->video", txt, 2, mt, 32, IMG_V, TXT_V, 300);
  }
  printf("\n");
  all_ok &= cg_ok;

  // --------------------------------------------------------------------------
  // Part 12: media -> text (captioning): image/audio/video -> caption tokens,
  // generated autoregressively from the media.
  // --------------------------------------------------------------------------
  printf("[12] Media -> text (captioning)\n");
  int mt_ok = 1;
  const int MD = 32;                              // media/model dimension

  // image -> text: 8x8 image -> 16 patch embeddings -> 3 caption tokens
  {
    Image im8; im8.h = 8; im8.w = 8; im8.c = 3;
    im8.data = (float*)malloc(64 * 3 * sizeof(float));
    for (int y = 0; y < 8; y++)
      for (int x = 0; x < 8; x++) {
        int i = y * 8 + x;
        im8.data[i * 3 + 0] = (float)x / 7.0f;
        im8.data[i * 3 + 1] = (float)y / 7.0f;
        im8.data[i * 3 + 2] = 0.4f;
      }
    int patch = 2, npi = patch_count(&im8, patch), pdi = patch_dim(&im8, patch);
    int wi[2] = {pdi, MD}; Tensor Wi = tensor_create(wi, 2); tensor_init_random(&Wi);
    int bi[1] = {MD};      Tensor bmi = tensor_create(bi, 1);
    int ei[2] = {npi, MD}; Tensor emi = tensor_create(ei, 2);
    patch_embed(&im8, patch, &Wi, &bmi, &emi);
    int cap[3] = {7, 8, 9};
    mt_ok &= media_to_text("image->text", &emi, npi, cap, 3, 32, 250);
    tensor_free(&Wi); tensor_free(&bmi); tensor_free(&emi); image_free(&im8);
  }
  // audio -> text: 128 samples -> 4 frame embeddings -> 3 caption tokens
  {
    Audio au; au.n = 128; au.sample_rate = 8000;
    au.samples = (float*)malloc(128 * sizeof(float));
    for (int i = 0; i < 128; i++) au.samples[i] = sinf(i * 0.25f) * 0.7f;
    int fs = 32, hop = 32, nfa = audio_frame_count(&au, fs, hop);
    int wa[2] = {fs, MD}; Tensor Wa = tensor_create(wa, 2); tensor_init_random(&Wa);
    int ba[1] = {MD};     Tensor bma = tensor_create(ba, 1);
    int ea[2] = {nfa, MD}; Tensor ema = tensor_create(ea, 2);
    audio_embed(&au, fs, hop, &Wa, &bma, &ema);
    int cap[3] = {10, 11, 12};
    mt_ok &= media_to_text("audio->text", &ema, nfa, cap, 3, 32, 250);
    tensor_free(&Wa); tensor_free(&bma); tensor_free(&ema); audio_free(&au);
  }
  // video -> text: 2 frames (4x4) -> 8 patch embeddings -> 3 caption tokens
  {
    Video vd; vd.n = 2; vd.frames = (Image*)malloc(2 * sizeof(Image));
    for (int f = 0; f < 2; f++) {
      vd.frames[f].h = 4; vd.frames[f].w = 4; vd.frames[f].c = 3;
      vd.frames[f].data = (float*)malloc(16 * 3 * sizeof(float));
      for (int i = 0; i < 16 * 3; i++) vd.frames[f].data[i] = (float)((i + f * 20) % 100) / 100.0f;
    }
    int patch = 2, nvp = video_patch_count(&vd, patch), pdv = patch_dim(&vd.frames[0], patch);
    int wv[2] = {pdv, MD}; Tensor Wv = tensor_create(wv, 2); tensor_init_random(&Wv);
    int bv[1] = {MD};      Tensor bmv = tensor_create(bv, 1);
    int ev[2] = {nvp, MD}; Tensor emv = tensor_create(ev, 2);
    video_embed(&vd, patch, &Wv, &bmv, &emv);
    int cap[3] = {13, 14, 15};
    mt_ok &= media_to_text("video->text", &emv, nvp, cap, 3, 32, 250);
    tensor_free(&Wv); tensor_free(&bmv); tensor_free(&emv); video_free(&vd);
  }
  printf("\n");
  all_ok &= mt_ok;

  adam_free(&opt);
  tensor_free(&logits);
  tensor_free(&probs);
  gpt_acts_free(&acts);
  gpt_free(&model);

  // --------------------------------------------------------------------------
  printf("Output validation: %s\n\n", all_ok ? "PASSED" : "FAILED");
  printf("Done! All memory freed successfully.\n");
  return all_ok ? 0 : 1;
}

// ============================================================================
// `generate` subcommand: autoregressive generation driven by KV cache + sampler.
//   transformer generate                       -> random toy model, token IDs in
//   transformer generate model.bin             -> loaded weights, token IDs in
//   transformer generate model.bin tok.bin     -> TEXT in / TEXT out (tokenizer)
//   transformer generate --name X              -> model/X/ (text if tok.bin present)
// Reads one prompt per stdin line.
// ============================================================================
static int run_generate(int argc, char **argv) {
  // Resolve model + optional tokenizer paths (supports --name <folder>).
  const char *model_path = NULL, *tok_path = NULL;
  static char nm[512], nt[512];
  if (argc > 3 && strcmp(argv[2], "--name") == 0) {
    model_paths(argv[3], nm, nt, sizeof(nm), 0);
    model_path = nm;
    FILE *tf = fopen(nt, "rb");           // text mode only if the tokenizer exists
    if (tf) { fclose(tf); tok_path = nt; }
  } else {
    if (argc > 2) model_path = argv[2];
    if (argc > 3) tok_path = argv[3];
  }

  GPT model;
  if (model_path) {
    if (gpt_load(model_path, &model) != 0) return 1;
    fprintf(stderr, "loaded %s\n", model_path);
  } else {
    srand(42);
    model = gpt_create(16, 8, 2, 2, 32, 64);        // toy fallback (random weights)
    fprintf(stderr, "no checkpoint given; using a random toy model (vocab=16)\n");
  }

  // Optional tokenizer -> text mode (stdin lines are text, not token IDs).
  Tokenizer tok;
  int text_mode = 0;
  if (tok_path) {
    if (tokenizer_load(&tok, tok_path) != 0) {
      fprintf(stderr, "cannot load tokenizer %s\n", tok_path);
      gpt_free(&model);
      return 1;
    }
    text_mode = 1;
    fprintf(stderr, "text mode (tokenizer %s, vocab %d)\n", tok_path, tok.vocab_size);
  }

  const int vocab = model.vocab_size;
  const int max_seq = model.max_seq_len;
  const int n_new = 40;                   // tokens to generate per prompt
  const float temperature = 0.8f;
  const int top_k = 40;
  const float top_p = 0.95f;
  Rng rng = rng_create(1234);

  int lshape[2] = {1, vocab};
  Tensor logits = tensor_create(lshape, 2);
  int *prompt = (int*)malloc(max_seq * sizeof(int));
  int *gen = (int*)malloc(max_seq * sizeof(int));

  char line[8192];
  while (fgets(line, sizeof(line), stdin)) {
    char *p = line;
    // Strip a leading UTF-8 BOM (common when piping on Windows).
    if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB &&
        (unsigned char)p[2] == 0xBF) {
      p += 3;
    }

    int n = 0;
    if (text_mode) {
      size_t L = strlen(p);                 // strip trailing newline
      while (L > 0 && (p[L - 1] == '\n' || p[L - 1] == '\r')) p[--L] = '\0';
      n = tokenizer_encode(&tok, p, prompt, max_seq);
    } else {
      for (char *t = strtok(p, " \t\r\n"); t && n < max_seq; t = strtok(NULL, " \t\r\n")) {
        int id = atoi(t);
        if (id < 0 || id >= vocab) {
          fprintf(stderr, "skipping out-of-range token %d (vocab=%d)\n", id, vocab);
          continue;
        }
        prompt[n++] = id;
      }
    }
    if (n == 0) continue;

    KVCache kv = gpt_kvcache_create(&model);
    for (int i = 0; i < n; i++) gpt_decode_step(&model, &kv, prompt[i], &logits);
    int ng = 0;
    for (int s = 0; s < n_new && kv.len < max_seq; s++) {
      int next = sample_token(&logits, 0, temperature, top_k, top_p, &rng);
      gen[ng++] = next;
      gpt_decode_step(&model, &kv, next, &logits);
    }

    if (text_mode) {
      char out[8192];
      tokenizer_decode(&tok, gen, ng, out, sizeof(out));
      printf("%s%s\n", p, out);                    // prompt text + continuation
    } else {
      printf("gen:");
      for (int i = 0; i < n; i++) printf(" %d", prompt[i]);
      for (int i = 0; i < ng; i++) printf(" [%d]", gen[i]);
      printf("\n");
    }
    fflush(stdout);
    gpt_kvcache_free(&kv);
  }

  free(prompt);
  free(gen);
  tensor_free(&logits);
  if (text_mode) tokenizer_free(&tok);
  gpt_free(&model);
  return 0;
}

// ============================================================================
// `agent` subcommand: the parse -> approve -> execute -> observe loop with a
// pluggable model backend.
//   transformer agent                 -> built-in scripted backend (demo)
//   transformer agent --cmd "<prog>"  -> run <prog> as the model; its stdout is
//                                        each response (the transcript is passed
//                                        as a file arg). Plug in a real LLM here.
//   --ask   approve each tool action interactively   --goal "..."  set the goal
// ============================================================================
static int run_agent(int argc, char **argv) {
  const char *goal = "Create a note file, then read it back to verify.";
  const char *cmd = NULL;
  ApprovalMode mode = APPROVAL_AUTO;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--ask") == 0) mode = APPROVAL_ASK;
    else if (strcmp(argv[i], "--cmd") == 0 && i + 1 < argc) cmd = argv[++i];
    else if (strcmp(argv[i], "--goal") == 0 && i + 1 < argc) goal = argv[++i];
  }

  int rc;
  if (cmd) {
    CommandBackend b = command_backend(cmd);
    rc = agent_run(goal, &b.base, mode);
  } else {
    // Built-in scripted "model" (the toy GPT can't emit tool calls).
    const char *responses[] = {
      "I'll create the file.\n"
      "<tool name=\"write_file\" path=\"agent_demo.txt\">Hello from the C agent!\n</tool>",
      "Now I'll read it back.\n"
      "<tool name=\"read_file\" path=\"agent_demo.txt\"></tool>",
      "<done>Created agent_demo.txt and verified its contents.</done>"
    };
    ScriptedBackend b = scripted_backend(responses, 3);
    rc = agent_run(goal, &b.base, mode);
  }
  remove("agent_demo.txt");  // clean up the demo artifact
  return rc;
}

// ============================================================================
// `train` subcommand: train on a text file or directory, configured by flags.
//   transformer train <path> [--out F] [--tok F] [--merges N] [--steps N]
//                            [--d-model N] [--layers N] [--heads N] [--d-ff N]
//                            [--ctx N] [--lr F] [--seed N]
// Scale up (bigger/slower/better):  --d-model 256 --layers 6 --heads 8 --ctx 128 --steps 20000
// ============================================================================
static void train_usage(void) {
  fprintf(stderr,
    "usage: transformer train <path> [flags]\n"
    "  <path> : text file OR directory of *.txt\n"
    "  --name X      save to model/X/ (model.bin + tok.bin); use `generate --name X`\n"
    "  --out F       output model (default model.bin)   --tok F     output tokenizer (default tok.bin)\n"
    "  --merges N    BPE merges (default 300)            --steps N   Adam steps (default 1000)\n"
    "  --d-model N   model dim (default 128)             --layers N  blocks (default 3)\n"
    "  --heads N     attention heads (default 4)         --d-ff N    FFN dim (default 4*d_model)\n"
    "  --ctx N       context window (default 64)         --lr F      learning rate (default 3e-4)\n"
    "  --seed N      RNG seed (default 1234)             --val-frac F  held-out fraction (default 0.1)\n"
    "  --resume F    continue training a saved model     --checkpoint-every N  save every N steps\n");
}

static int run_train(int argc, char **argv) {
  if (argc < 3 || strncmp(argv[2], "--", 2) == 0) { train_usage(); return 2; }
  const char *data = argv[2];
  TrainConfig cfg = train_default_config();
  const char *name = NULL;
  static char name_model[512], name_tok[512];

  for (int i = 3; i < argc; i++) {
    const char *f = argv[i];
    int has_val = (i + 1 < argc);
#define NEED_VAL() do { if (!has_val) { fprintf(stderr, "missing value for %s\n", f); return 2; } } while (0)
    if      (strcmp(f, "--name") == 0)    { NEED_VAL(); name = argv[++i]; }
    else if (strcmp(f, "--out") == 0)     { NEED_VAL(); cfg.out_model = argv[++i]; }
    else if (strcmp(f, "--tok") == 0)     { NEED_VAL(); cfg.out_tok = argv[++i]; }
    else if (strcmp(f, "--merges") == 0)  { NEED_VAL(); cfg.num_merges = atoi(argv[++i]); }
    else if (strcmp(f, "--steps") == 0)   { NEED_VAL(); cfg.steps = atoi(argv[++i]); }
    else if (strcmp(f, "--d-model") == 0) { NEED_VAL(); cfg.d_model = atoi(argv[++i]); }
    else if (strcmp(f, "--layers") == 0)  { NEED_VAL(); cfg.n_layers = atoi(argv[++i]); }
    else if (strcmp(f, "--heads") == 0)   { NEED_VAL(); cfg.n_heads = atoi(argv[++i]); }
    else if (strcmp(f, "--d-ff") == 0)    { NEED_VAL(); cfg.d_ff = atoi(argv[++i]); }
    else if (strcmp(f, "--ctx") == 0)     { NEED_VAL(); cfg.ctx = atoi(argv[++i]); }
    else if (strcmp(f, "--lr") == 0)      { NEED_VAL(); cfg.lr = (float)atof(argv[++i]); }
    else if (strcmp(f, "--seed") == 0)    { NEED_VAL(); cfg.seed = (unsigned)atoi(argv[++i]); }
    else if (strcmp(f, "--resume") == 0)  { NEED_VAL(); cfg.resume = argv[++i]; }
    else if (strcmp(f, "--checkpoint-every") == 0) { NEED_VAL(); cfg.checkpoint_every = atoi(argv[++i]); }
    else if (strcmp(f, "--val-frac") == 0){ NEED_VAL(); cfg.val_frac = (float)atof(argv[++i]); }
    else { fprintf(stderr, "unknown flag: %s\n", f); train_usage(); return 2; }
#undef NEED_VAL
  }
  // --name X saves to model/X/{model,tok}.bin (unless --out/--tok were given).
  if (name) {
    model_paths(name, name_model, name_tok, sizeof(name_model), 1);
    cfg.out_model = name_model;
    cfg.out_tok = name_tok;
    printf("model dir: model/%s/\n", name);
  }
  return train_run(data, &cfg);
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "generate") == 0) return run_generate(argc, argv);
  if (argc > 1 && strcmp(argv[1], "agent") == 0) return run_agent(argc, argv);
  if (argc > 1 && strcmp(argv[1], "train") == 0) return run_train(argc, argv);
  return run_demo();
}
