// selftest.c - The built-in smoke test (default `transformer` run), split out
// of main.c so main.c is just CLI dispatch. Exercises every module end-to-end.
// See selftest.h. Correctness gates enforced here: the conditioned gradient
// check (rel err < 1e-2) and the bit-exact KV-cache check (0.00e+00).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "selftest.h"
#include "tensor.h"
#include "loss.h"
#include "optim.h"
#include "sampler.h"
#include "model.h"
#include "checkpoint.h"
#include "tokenizer.h"
#include "image.h"
#include "patch.h"
#include "video.h"
#include "audio.h"
#include "imagegen.h"
#include "audiogen.h"
#include "rmsnorm.h"
#include "rope.h"
#include "swiglu.h"
#include "gqa.h"
#include "llama.h"
#include "llamatrain.h"
#include "quant.h"
#include "matrix.h"
#include "activation.h"
#include "reward.h"
#include "grpo.h"
#include "gpu.h"
#include "moe.h"
#include "muon.h"
#include "train.h"

// Scalar loss L = sum(w * rmsnorm(x, gamma)) — used by the [13] gradient check.
static float rms_scalar_loss(const Tensor *x, const Tensor *gm, Tensor *y,
                             const float *w, float eps) {
  rmsnorm(x, gm, y, eps);
  int n = y->shape[0] * y->shape[1];
  float l = 0.0f;
  for (int i = 0; i < n; i++) l += w[i] * y->data[i];
  return l;
}

// Scalar loss L = sum(w * rope(x)) — used by the [14] gradient check.
static float rope_scalar_loss(const Tensor *x, Tensor *y, const float *w, float base) {
  rope(x, y, base);
  int n = y->shape[0] * y->shape[1];
  float l = 0.0f;
  for (int i = 0; i < n; i++) l += w[i] * y->data[i];
  return l;
}

// Forward + mean cross-entropy for the LLaMA-family training path — used by the
// [22] end-to-end gradient check (re-run per perturbed parameter).
static float llama_ce_loss(LlamaModel *m, LlamaActs *a, const int *ids, int seq_len,
                           const int *targets, Tensor *logits, Tensor *probs) {
  llama_forward_train(m, ids, seq_len, a, logits);
  return cross_entropy_forward(logits, targets, probs);
}

// Central-difference gradient check of params[0..n) against their .grad, for the
// LLaMA-family CE loss with optional adapters (lo == NULL: the plain model) —
// used by [31]. Tensors of at most `full` entries are checked at EVERY entry,
// larger ones at two. Writes the max rel error over all entries and over the
// entries clearing the float32 noise floor (|grad| > 1e-2, the rule of [2] and
// [22]); returns how many entries cleared it (0 = a vacuous check).
static int llama_grad_check(LlamaModel *m, LlamaActs *a, const LlamaLora *lo,
                            LlamaLoraActs *la, const int *ids, int seq, const int *targets,
                            Tensor *logits, Tensor *probs, Tensor **params, int n,
                            size_t full, float *max_all, float *max_cond) {
  const float eps = 1e-2f;
  int judged = 0;
  *max_all = 0.0f; *max_cond = 0.0f;
  for (int p = 0; p < n; p++) {
    Tensor *param = params[p];
    size_t step = (param->size <= full) ? 1 : (param->size > 2 ? param->size / 2 : 1);
    for (size_t k = 0; k < param->size; k += step) {
      float base = param->data[k];
      param->data[k] = base + eps;
      llama_forward_train_lora(m, ids, seq, a, logits, lo, la);
      float lp = cross_entropy_forward(logits, targets, probs);
      param->data[k] = base - eps;
      llama_forward_train_lora(m, ids, seq, a, logits, lo, la);
      float lmn = cross_entropy_forward(logits, targets, probs);
      param->data[k] = base;
      float numeric = (lp - lmn) / (2.0f * eps);
      float analytic = param->grad[k];
      float rel = fabsf(analytic - numeric) / (fabsf(analytic) + fabsf(numeric) + 1e-8f);
      if (rel > *max_all) *max_all = rel;
      if (fabsf(analytic) > 1e-2f) {
        judged++;
        if (rel > *max_cond) *max_cond = rel;
      }
    }
  }
  return judged;
}

// Portable "make directory if absent" (ignores already-exists).
#if defined(_WIN32)
#include <direct.h>
static void ensure_dir(const char *p) { _mkdir(p); }
#else
#include <sys/stat.h>
static void ensure_dir(const char *p) { mkdir(p, 0755); }
#endif
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

// ---- MoE helpers for part [31] -------------------------------------------

// Deterministic fill (independent of rand() state): a scrambled sawtooth in
// [-scale, scale). `mul` and `off` decorrelate tensors from each other.
static void moe_fill(Tensor *T, int mul, int off, float scale) {
  for (size_t i = 0; i < T->size; i++)
    T->data[i] = ((float)(((int)i * mul + off) % 97) / 48.5f - 1.0f) * scale;
}

// L = sum(w * y) + aux_coef * aux. Also reports whether the routing (the chosen
// expert ids of every token) differs from `ref_sel` -- a finite-difference probe
// that flips a top-k choice measures a jump, not a derivative.
static float moe_scalar_loss(const MoeLayer *m, const Tensor *x, MoeCache *c, Tensor *y,
                             const float *w, float aux_coef, const int *ref_sel,
                             int *flipped) {
  float aux = moe_forward(m, x, c, y);
  float l = aux_coef * aux;
  for (size_t i = 0; i < y->size; i++) l += w[i] * y->data[i];
  if (ref_sel && memcmp(ref_sel, c->sel, (size_t)c->n * c->top_k * sizeof(int)) != 0)
    *flipped = 1;
  return l;
}

// Full numerical gradient check of one MoE configuration over x and EVERY
// parameter. The routing is NOT frozen: the real forward (softmax + top-k) runs
// at every probe, and each probe's selection is compared against the base one.
// The inputs are chosen so the k-th / (k+1)-th probability gap (`margin`) is
// far larger than anything an h=1e-3 nudge can close, and the test then
// PROVES that no probe flipped (`flips` must be 0) rather than assuming it.
// use_task=0 zeroes w, isolating the aux-loss gradient.
static float moe_gradcheck(int E, int k, int renorm, int shared, int use_task,
                           float aux_coef, int *flips, float *margin, int *checked) {
  const int n = 5, dm = 4, df = 5;
  MoeLayer m = moe_create(dm, df, E, k, renorm, shared);
  int np = moe_n_params(&m);
  Tensor **ps = (Tensor**)malloc((np + 1) * sizeof(Tensor*));
  moe_params(&m, ps);
  for (int i = 0; i < np; i++) moe_fill(ps[i], 13 + 6 * i, 7 * i, 0.5f);
  moe_fill(&m.W_router, 29, 3, 2.0f);   // a decisive router: wide top-k margins
  int xs[2] = {n, dm};
  Tensor x = tensor_create(xs, 2), y = tensor_create(xs, 2);
  moe_fill(&x, 31, 11, 1.0f);
  ps[np] = &x;
  MoeCache c = moe_cache_create(&m, n);
  float *w = (float*)malloc(n * dm * sizeof(float));
  for (int i = 0; i < n * dm; i++) w[i] = use_task ? ((i * 23 % 100) / 50.0f) - 1.0f : 0.0f;

  // Analytic: forward, dL/dy = w, backward with the aux term.
  moe_forward(&m, &x, &c, &y);
  for (int i = 0; i <= np; i++) tensor_zero_grad(ps[i]);
  memcpy(y.grad, w, n * dm * sizeof(float));
  moe_backward(&m, &x, &c, &y, aux_coef);

  int *ref = (int*)malloc((size_t)n * k * sizeof(int));
  memcpy(ref, c.sel, (size_t)n * k * sizeof(int));
  *margin = 1.0f;
  if (k < E)
    for (int t = 0; t < n; t++) {
      const float *p = c.probs.data + (size_t)t * E;
      float kth = p[ref[t * k + k - 1]], best_out = 0.0f;
      for (int e = 0; e < E; e++) {
        int in = 0;
        for (int j = 0; j < k; j++) if (ref[t * k + j] == e) in = 1;
        if (!in && p[e] > best_out) best_out = p[e];
      }
      if (kth - best_out < *margin) *margin = kth - best_out;
    }

  const float h = 1e-3f;
  float max_rel = 0.0f;
  *flips = 0; *checked = 0;
  for (int pi = 0; pi <= np; pi++) {
    Tensor *T = ps[pi];
    for (size_t q = 0; q < T->size; q++) {
      float s = T->data[q];
      int fl = 0;
      T->data[q] = s + h; float lp = moe_scalar_loss(&m, &x, &c, &y, w, aux_coef, ref, &fl);
      T->data[q] = s - h; float lm = moe_scalar_loss(&m, &x, &c, &y, w, aux_coef, ref, &fl);
      T->data[q] = s;
      *flips += fl;
      float num = (lp - lm) / (2.0f * h), g = T->grad[q];
      float rel = fabsf(num - g) / (fabsf(g) + 1e-3f);
      if (fabsf(g) > 1e-2f) { (*checked)++; if (rel > max_rel) max_rel = rel; }
    }
  }
  free(ref); free(w); free(ps);
  moe_cache_free(&c); tensor_free(&x); tensor_free(&y); moe_free(&m);
  return max_rel;
}

// ---- Muon helpers for part [35] ------------------------------------------

// Cyclic Jacobi eigen-decomposition of a symmetric n x n matrix (double, in
// place): on return S's diagonal holds the eigenvalues and V (n x n, row-major)
// the eigenvectors as COLUMNS. Tiny n only — a reference, not a kernel.
static void jacobi_eig(double *S, double *V, int n) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) V[i * n + j] = (i == j);
  for (int sweep = 0; sweep < 100; sweep++) {
    double off = 0.0;
    for (int p = 0; p < n; p++)
      for (int q = p + 1; q < n; q++) off += S[p * n + q] * S[p * n + q];
    if (off < 1e-26) break;
    for (int p = 0; p < n; p++)
      for (int q = p + 1; q < n; q++) {
        double apq = S[p * n + q];
        if (fabs(apq) < 1e-300) continue;
        double th = (S[q * n + q] - S[p * n + p]) / (2.0 * apq);
        double t = (th >= 0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1.0));
        double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < n; k++) {            // S <- S J  (columns p, q)
          double skp = S[k * n + p], skq = S[k * n + q];
          S[k * n + p] = c * skp - s * skq; S[k * n + q] = s * skp + c * skq;
        }
        for (int k = 0; k < n; k++) {            // S <- J^T S  (rows p, q)
          double spk = S[p * n + k], sqk = S[q * n + k];
          S[p * n + k] = c * spk - s * sqk; S[q * n + k] = s * spk + c * sqk;
        }
        for (int k = 0; k < n; k++) {            // V <- V J
          double vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
        }
      }
  }
}

// For an r x c matrix M (float, row-major), k = min(r, c): the singular values
// of M (via eigenvalues of the small-side Gram) into sv[k], max |Gram - I|, and
// (if polar != NULL) the EXACT polar factor U V^T of M in double — the matrix
// Newton-Schulz approximates — into polar[r*c].
static double small_gram_svd(const float *M, int r, int c, double *sv, double *polar) {
  const int tall = r > c, k = tall ? c : r;
  double *S = (double*)malloc((size_t)k * k * sizeof(double));
  double *V = (double*)malloc((size_t)k * k * sizeof(double));
  for (int i = 0; i < k; i++)
    for (int j = 0; j < k; j++) {
      double acc = 0.0;
      if (tall) for (int t = 0; t < r; t++) acc += (double)M[t * c + i] * M[t * c + j];  // M^T M
      else      for (int t = 0; t < c; t++) acc += (double)M[i * c + t] * M[j * c + t];  // M M^T
      S[i * k + j] = acc;
    }
  double dev = 0.0;
  for (int i = 0; i < k; i++)
    for (int j = 0; j < k; j++) {
      double d = fabs(S[i * k + j] - (i == j));
      if (d > dev) dev = d;
    }
  jacobi_eig(S, V, k);
  for (int i = 0; i < k; i++) sv[i] = sqrt(S[i * k + i] > 0 ? S[i * k + i] : 0);
  if (polar) {
    // Gram^{-1/2} = V diag(1/sv) V^T; polar = M Gram^{-1/2} (tall) or Gram^{-1/2} M (wide).
    double *P = (double*)malloc((size_t)k * k * sizeof(double));
    for (int i = 0; i < k; i++)
      for (int j = 0; j < k; j++) {
        double acc = 0.0;
        for (int t = 0; t < k; t++) acc += V[i * k + t] * V[j * k + t] / sv[t];
        P[i * k + j] = acc;
      }
    for (int i = 0; i < r; i++)
      for (int j = 0; j < c; j++) {
        double acc = 0.0;
        if (tall) for (int t = 0; t < k; t++) acc += (double)M[i * c + t] * P[t * k + j];
        else      for (int t = 0; t < k; t++) acc += P[i * k + t] * (double)M[t * c + j];
        polar[i * c + j] = acc;
      }
    free(P);
  }
  free(S); free(V);
  return dev;
}

int run_selftest(void) {
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

  // Use throwaway names (NOT the train defaults tok.bin/model.bin) so running
  // the self-test can never overwrite or delete a user's trained model.
  int save_load_ok = (tokenizer_save(&tok, "selftest_tok.tmp") == 0);
  Tokenizer tok2;
  save_load_ok = save_load_ok && (tokenizer_load(&tok2, "selftest_tok.tmp") == 0);
  if (save_load_ok) {
    int n2 = tokenizer_encode(&tok2, sample, tids2, 256);
    save_load_ok = (n2 == n_tids) && (memcmp(tids, tids2, n_tids * sizeof(int)) == 0);
    tokenizer_free(&tok2);
  }
  remove("selftest_tok.tmp");
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
  int ckpt_ok = (gpt_save("selftest_model.tmp", &probe) == 0);
  GPT probe2;
  ckpt_ok = ckpt_ok && (gpt_load("selftest_model.tmp", &probe2) == 0);
  if (ckpt_ok) {
    gpt_forward(&probe2, pseq, 3, &plb);
    ckpt_ok = (memcmp(pla.data, plb.data, pla.size * sizeof(float)) == 0);
    gpt_free(&probe2);
  }
  tensor_free(&pla); tensor_free(&plb); gpt_free(&probe); remove("selftest_model.tmp");
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
    size_t step = param->size > 2 ? param->size / 2 : 1;
    for (size_t k = 0; k < param->size; k += step) {
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
    for (size_t i = 0; i < mmlog.size; i++)
      if (isnan(mmlog.data[i]) || isinf(mmlog.data[i])) { finite = 0; break; }
    int e3_ok = (mmlog.size == (size_t)(total * vocab)) && finite;
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
  // Part 9: instruction tuning (SFT) — the recipe step (stage 3, see memory.md)
  // that makes a base model FOLLOW instructions instead of just continuing text.
  // Train on (prompt -> response) pairs with the PROMPT positions MASKED (loss
  // only on the response tokens; target < 0 = ignore, same trick real SFT uses
  // on the completion), then feed each prompt and check the model generates the
  // right response. Two distinct prompts -> two distinct responses = instruction
  // following in miniature. (Toy scale: it memorizes; the *technique* is real.)
  // --------------------------------------------------------------------------
  {
    const int Vs = 16, dm = 32, L = 5;
    // Both examples padded to the same length L=5 so one acts/logits buffer serves both.
    //   A: prompt [1,2]   -> response [7,8,9]     B: prompt [3,4,5] -> response [11,12]
    int seqA[5] = {1, 2, 7, 8, 9},    tgtA[5] = {-1, 7, 8, 9, -1};   // supervise response only
    int seqB[5] = {3, 4, 5, 11, 12},  tgtB[5] = {-1, -1, 11, 12, -1};
    int promptA[2] = {1, 2},    respA[3] = {7, 8, 9};
    int promptB[3] = {3, 4, 5}, respB[2] = {11, 12};

    GPT gs = gpt_create(Vs, dm, 2, 2, dm * 4, L);
    GPTActs as = gpt_acts_create(&gs, L);
    int lz[2] = {L, Vs};
    Tensor lo = tensor_create(lz, 2), pr = tensor_create(lz, 2);
    Tensor *par[256];
    int np = gpt_collect_params(&gs, par);
    Adam opt = adam_create(par, np, 0.01f);

    float first = 0.0f, last = 0.0f;
    for (int s = 0; s < 800; s++) {
      int *seq = (s & 1) ? seqB : seqA;
      int *tgt = (s & 1) ? tgtB : tgtA;
      gpt_forward_train(&gs, seq, L, &as, &lo);
      float loss = cross_entropy_forward(&lo, tgt, &pr);
      if (s == 0) first = loss;
      last = loss;
      tensor_zero_grad(&lo);
      cross_entropy_backward(&pr, tgt, &lo);
      gpt_backward(&gs, seq, L, &as, &lo);
      adam_step(&opt, par, np);
    }

    // Feed each prompt through the KV cache, then greedily generate the response.
    int genA[3], genB[2];
    int gz[2] = {1, Vs};
    Tensor gl = tensor_create(gz, 2);
    KVCache kvA = gpt_kvcache_create(&gs);
    for (int i = 0; i < 2; i++) gpt_decode_step(&gs, &kvA, promptA[i], &gl);
    for (int r = 0; r < 3; r++) { genA[r] = sample_argmax(&gl, 0); if (r < 2) gpt_decode_step(&gs, &kvA, genA[r], &gl); }
    gpt_kvcache_free(&kvA);
    KVCache kvB = gpt_kvcache_create(&gs);
    for (int i = 0; i < 3; i++) gpt_decode_step(&gs, &kvB, promptB[i], &gl);
    for (int r = 0; r < 2; r++) { genB[r] = sample_argmax(&gl, 0); if (r < 1) gpt_decode_step(&gs, &kvB, genB[r], &gl); }
    gpt_kvcache_free(&kvB);

    int okA = (genA[0] == respA[0] && genA[1] == respA[1] && genA[2] == respA[2]);
    int okB = (genB[0] == respB[0] && genB[1] == respB[1]);
    int sft_ok = okA && okB;
    printf("[9] Instruction tuning (SFT): loss %.4f -> %.4f | prompt A -> %d %d %d %s | "
           "prompt B -> %d %d %s -> %s\n\n",
           first, last, genA[0], genA[1], genA[2], okA ? "OK" : "BAD",
           genB[0], genB[1], okB ? "OK" : "BAD", sft_ok ? "OK" : "FAIL");
    all_ok &= sft_ok;

    tensor_free(&gl); adam_free(&opt);
    tensor_free(&lo); tensor_free(&pr); gpt_acts_free(&as); gpt_free(&gs);
  }

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
  // Part 13: RMSNorm (LLaMA-family primitive, modern-architecture roadmap) —
  // forward + numerical gradient check (analytic backward vs central difference).
  // --------------------------------------------------------------------------
  {
    printf("[13] RMSNorm (LLaMA-family): forward + gradient check\n");
    const int R = 3, D = 8;
    int sh[2] = {R, D}, gsh[1] = {D};
    Tensor x = tensor_create(sh, 2), y = tensor_create(sh, 2);
    Tensor gm = tensor_create(gsh, 1), gg = tensor_create(gsh, 1);
    float w[24];
    for (int i = 0; i < R * D; i++) {
      x.data[i] = ((i * 37 % 100) / 50.0f) - 1.0f;    // deterministic pseudo-random in [-1,1)
      w[i] = ((i * 53 % 100) / 50.0f) - 1.0f;
    }
    for (int j = 0; j < D; j++) gm.data[j] = 0.5f + (j % 3) * 0.25f;
    const float eps = 1e-5f;

    // Analytic grads: L = sum(w * y) so dL/dy = w.
    rmsnorm(&x, &gm, &y, eps);
    for (int i = 0; i < R * D; i++) { x.grad[i] = 0.0f; y.grad[i] = w[i]; }
    for (int j = 0; j < D; j++) gg.grad[j] = 0.0f;
    rmsnorm_backward(&x, &gm, &y, &gg, eps);

    // Numerical grads (central difference) vs analytic.
    const float h = 1e-3f;
    float max_all = 0.0f, max_cond = 0.0f;
    for (int k = 0; k < R * D; k++) {
      float s = x.data[k];
      x.data[k] = s + h; float lp = rms_scalar_loss(&x, &gm, &y, w, eps);
      x.data[k] = s - h; float lm = rms_scalar_loss(&x, &gm, &y, w, eps);
      x.data[k] = s;
      float num = (lp - lm) / (2.0f * h), rel = fabsf(num - x.grad[k]) / (fabsf(x.grad[k]) + 1e-3f);
      if (rel > max_all) max_all = rel;
      if (fabsf(x.grad[k]) > 1e-2f && rel > max_cond) max_cond = rel;
    }
    for (int j = 0; j < D; j++) {
      float s = gm.data[j];
      gm.data[j] = s + h; float lp = rms_scalar_loss(&x, &gm, &y, w, eps);
      gm.data[j] = s - h; float lm = rms_scalar_loss(&x, &gm, &y, w, eps);
      gm.data[j] = s;
      float num = (lp - lm) / (2.0f * h), rel = fabsf(num - gg.grad[j]) / (fabsf(gg.grad[j]) + 1e-3f);
      if (rel > max_all) max_all = rel;
      if (fabsf(gg.grad[j]) > 1e-2f && rel > max_cond) max_cond = rel;
    }
    int rms_ok = (max_cond < 1e-2f);
    printf("    gradient check: max rel error %.2e (all) | %.2e (|grad|>1e-2) -> %s\n\n",
           max_all, max_cond, rms_ok ? "OK" : "FAIL");
    all_ok &= rms_ok;
    tensor_free(&x); tensor_free(&y); tensor_free(&gm); tensor_free(&gg);
  }

  // --------------------------------------------------------------------------
  // Part 14: RoPE (rotary positions, LLaMA-family) — numerical gradient check +
  // the DEFINING RoPE property: <RoPE(q,m), RoPE(k,n)> depends only on (m-n), so
  // the same relative offset at different absolute positions gives equal Q·K.
  // --------------------------------------------------------------------------
  {
    printf("[14] RoPE (rotary positions): gradient check + relative-position property\n");
    const int R = 6, D = 8;
    const float base = 10000.0f;
    int sh[2] = {R, D};
    Tensor x = tensor_create(sh, 2), y = tensor_create(sh, 2);
    float w[48];
    for (int i = 0; i < R * D; i++) {
      x.data[i] = ((i * 29 % 100) / 50.0f) - 1.0f;
      w[i] = ((i * 61 % 100) / 50.0f) - 1.0f;
    }
    // Analytic grads (L = sum(w * y), dL/dy = w). RoPE is parameter-free -> dx only.
    rope(&x, &y, base);
    for (int i = 0; i < R * D; i++) { x.grad[i] = 0.0f; y.grad[i] = w[i]; }
    rope_backward(&x, &y, base);
    const float h = 1e-3f;
    float max_cond = 0.0f;
    for (int k = 0; k < R * D; k++) {
      float s = x.data[k];
      x.data[k] = s + h; float lp = rope_scalar_loss(&x, &y, w, base);
      x.data[k] = s - h; float lm = rope_scalar_loss(&x, &y, w, base);
      x.data[k] = s;
      float num = (lp - lm) / (2.0f * h), rel = fabsf(num - x.grad[k]) / (fabsf(x.grad[k]) + 1e-3f);
      if (fabsf(x.grad[k]) > 1e-2f && rel > max_cond) max_cond = rel;
    }

    // Relative-position property: q at positions 1 and 4, k at positions 2 and 5
    // (both have offset n-m = 1), so <R_1 q, R_2 k> should equal <R_4 q, R_5 k>.
    Tensor p = tensor_create(sh, 2), pr = tensor_create(sh, 2);
    for (int i = 0; i < R * D; i++) p.data[i] = 0.0f;
    for (int j = 0; j < D; j++) {
      float q = ((j * 17 % 100) / 50.0f) - 1.0f, kk = ((j * 41 % 100) / 50.0f) - 1.0f;
      p.data[1 * D + j] = q;  p.data[4 * D + j] = q;    // query at pos 1 and 4
      p.data[2 * D + j] = kk; p.data[5 * D + j] = kk;   // key   at pos 2 and 5
    }
    rope(&p, &pr, base);
    float dot1 = 0.0f, dot2 = 0.0f;
    for (int j = 0; j < D; j++) {
      dot1 += pr.data[1 * D + j] * pr.data[2 * D + j];
      dot2 += pr.data[4 * D + j] * pr.data[5 * D + j];
    }
    float rel_dp = fabsf(dot1 - dot2) / (fabsf(dot1) + 1e-3f);
    int rope_ok = (max_cond < 1e-2f) && (rel_dp < 1e-3f);
    printf("    gradient %.2e (|grad|>1e-2) | rel-pos <q,k>: pos(1,2)=%.4f == pos(4,5)=%.4f (rel %.2e) -> %s\n\n",
           max_cond, dot1, dot2, rel_dp, rope_ok ? "OK" : "FAIL");
    all_ok &= rope_ok;
    tensor_free(&x); tensor_free(&y); tensor_free(&p); tensor_free(&pr);
  }

  // --------------------------------------------------------------------------
  // Part 15: SwiGLU (LLaMA-family gated FFN, replaces the GELU MLP) — forward +
  // numerical gradient check over all four tensors (x, W_gate, W_up, W_down).
  // --------------------------------------------------------------------------
  {
    printf("[15] SwiGLU (LLaMA-family FFN): forward + gradient check\n");
    const int n = 3, dm = 4, df = 6;
    int xs[2] = {n, dm}, wgs[2] = {dm, df}, wds[2] = {df, dm}, ys[2] = {n, dm};
    Tensor x = tensor_create(xs, 2), y = tensor_create(ys, 2);
    Tensor Wg = tensor_create(wgs, 2), Wu = tensor_create(wgs, 2), Wd = tensor_create(wds, 2);
    SwigluCache c = swiglu_cache_create(n, df);
    for (int i = 0; i < n * dm; i++) x.data[i] = ((i * 13 % 100) / 50.0f) - 1.0f;
    for (int i = 0; i < dm * df; i++) {
      Wg.data[i] = ((i * 7 % 100) / 50.0f - 1.0f) * 0.5f;
      Wu.data[i] = ((i * 19 % 100) / 50.0f - 1.0f) * 0.5f;
    }
    for (int i = 0; i < df * dm; i++) Wd.data[i] = ((i * 11 % 100) / 50.0f - 1.0f) * 0.5f;
    float w[12];
    for (int i = 0; i < n * dm; i++) w[i] = ((i * 23 % 100) / 50.0f) - 1.0f;

    // Analytic grads (L = sum(w * y), dL/dy = w).
    swiglu_forward(&x, &Wg, &Wu, &Wd, &c, &y);
    tensor_zero_grad(&x); tensor_zero_grad(&Wg); tensor_zero_grad(&Wu); tensor_zero_grad(&Wd);
    for (int i = 0; i < n * dm; i++) y.grad[i] = w[i];
    swiglu_backward(&x, &Wg, &Wu, &Wd, &c, &y);

    // Numerical grads over all four parameter tensors.
    const float h = 1e-3f;
    float max_cond = 0.0f;
    Tensor *ps[4] = {&x, &Wg, &Wu, &Wd};
    for (int pi = 0; pi < 4; pi++) {
      Tensor *T = ps[pi];
      int sz = T->shape[0] * T->shape[1];
      for (int k = 0; k < sz; k++) {
        float s = T->data[k];
        T->data[k] = s + h; swiglu_forward(&x, &Wg, &Wu, &Wd, &c, &y);
        float lp = 0.0f; for (int i = 0; i < n * dm; i++) lp += w[i] * y.data[i];
        T->data[k] = s - h; swiglu_forward(&x, &Wg, &Wu, &Wd, &c, &y);
        float lm = 0.0f; for (int i = 0; i < n * dm; i++) lm += w[i] * y.data[i];
        T->data[k] = s;
        float num = (lp - lm) / (2.0f * h), rel = fabsf(num - T->grad[k]) / (fabsf(T->grad[k]) + 1e-3f);
        if (fabsf(T->grad[k]) > 1e-2f && rel > max_cond) max_cond = rel;
      }
    }
    int sw_ok = (max_cond < 1e-2f);
    printf("    gradient check (x, W_gate, W_up, W_down): max rel error %.2e (|grad|>1e-2) -> %s\n\n",
           max_cond, sw_ok ? "OK" : "FAIL");
    all_ok &= sw_ok;
    swiglu_cache_free(&c);
    tensor_free(&x); tensor_free(&y); tensor_free(&Wg); tensor_free(&Wu); tensor_free(&Wd);
  }

  // --------------------------------------------------------------------------
  // Part 16: GQA (grouped-query attention, LLaMA-family) — causal attention with
  // FEWER KV heads than query heads. Forward + numerical gradient check (through
  // softmax + causal mask) over Q, K, V.
  // --------------------------------------------------------------------------
  {
    const int n = 4, H = 4, KV = 2, hd = 3;   // 4 query heads, 2 KV heads (group size 2)
    printf("[16] GQA (grouped-query attn): %d query heads, %d KV heads (group %d) + gradient check\n",
           H, KV, H / KV);
    int qs[2] = {n, H * hd}, ks[2] = {n, KV * hd};
    Tensor Q = tensor_create(qs, 2), O = tensor_create(qs, 2);
    Tensor K = tensor_create(ks, 2), V = tensor_create(ks, 2);
    GqaCache c = gqa_cache_create(n, H, KV, hd);
    for (int i = 0; i < n * H * hd; i++) Q.data[i] = ((i * 13 % 100) / 50.0f) - 1.0f;
    for (int i = 0; i < n * KV * hd; i++) {
      K.data[i] = ((i * 17 % 100) / 50.0f) - 1.0f;
      V.data[i] = ((i * 23 % 100) / 50.0f) - 1.0f;
    }
    float w[48];
    for (int i = 0; i < n * H * hd; i++) w[i] = ((i * 31 % 100) / 50.0f) - 1.0f;

    // Analytic grads (L = sum(w * O), dL/dO = w).
    gqa_forward(&Q, &K, &V, &c, &O);
    tensor_zero_grad(&Q); tensor_zero_grad(&K); tensor_zero_grad(&V);
    for (int i = 0; i < n * H * hd; i++) O.grad[i] = w[i];
    gqa_backward(&Q, &K, &V, &c, &O);

    // Numerical grads over Q, K, V.
    const float h = 1e-3f;
    float max_cond = 0.0f;
    Tensor *ps[3] = {&Q, &K, &V};
    for (int pi = 0; pi < 3; pi++) {
      Tensor *T = ps[pi];
      int sz = T->shape[0] * T->shape[1];
      for (int k = 0; k < sz; k++) {
        float s = T->data[k];
        T->data[k] = s + h; gqa_forward(&Q, &K, &V, &c, &O);
        float lp = 0.0f; for (int i = 0; i < n * H * hd; i++) lp += w[i] * O.data[i];
        T->data[k] = s - h; gqa_forward(&Q, &K, &V, &c, &O);
        float lm = 0.0f; for (int i = 0; i < n * H * hd; i++) lm += w[i] * O.data[i];
        T->data[k] = s;
        float num = (lp - lm) / (2.0f * h), rel = fabsf(num - T->grad[k]) / (fabsf(T->grad[k]) + 1e-3f);
        if (fabsf(T->grad[k]) > 1e-2f && rel > max_cond) max_cond = rel;
      }
    }
    int gqa_ok = (max_cond < 1e-2f);
    printf("    gradient check (Q, K, V): max rel error %.2e (|grad|>1e-2) -> %s\n\n",
           max_cond, gqa_ok ? "OK" : "FAIL");
    all_ok &= gqa_ok;
    gqa_cache_free(&c);
    tensor_free(&Q); tensor_free(&K); tensor_free(&V); tensor_free(&O);
  }

  // --------------------------------------------------------------------------
  // Part 17: modern (LLaMA-family) transformer block assembled from the four
  // primitives (RMSNorm + RoPE + GQA + SwiGLU + residuals). Forward + an
  // end-to-end CAUSALITY check: perturbing the LAST position must leave every
  // earlier output row unchanged (proves the causal mask survives assembly).
  // --------------------------------------------------------------------------
  {
    printf("[17] Modern block (RMSNorm+RoPE+GQA+SwiGLU): forward + causality\n");
    const int n = 5, dm = 8, nh = 4, nkv = 2, hd = 2, dff = 16;
    srand(7);
    LlamaBlock blk = llama_block_create(dm, nh, nkv, hd, dff, 10000.0f, 1e-5f);
    llama_block_randomize(&blk);
    int xs[2] = {n, dm};
    Tensor X = tensor_create(xs, 2), X2 = tensor_create(xs, 2);
    Tensor O1 = tensor_create(xs, 2), O2 = tensor_create(xs, 2);
    for (int i = 0; i < n * dm; i++) { X.data[i] = ((i * 13 % 100) / 50.0f) - 1.0f; X2.data[i] = X.data[i]; }
    for (int j = 0; j < dm; j++) X2.data[(n - 1) * dm + j] += 0.5f;   // perturb the last position only

    llama_block_forward(&blk, &X, &O1);
    llama_block_forward(&blk, &X2, &O2);

    int finite = 1;
    for (int i = 0; i < n * dm; i++) if (!isfinite(O1.data[i])) finite = 0;
    float past_diff = 0.0f, last_diff = 0.0f;
    for (int i = 0; i < n - 1; i++)
      for (int j = 0; j < dm; j++) {
        float d = fabsf(O1.data[i * dm + j] - O2.data[i * dm + j]);
        if (d > past_diff) past_diff = d;
      }
    for (int j = 0; j < dm; j++) {
      float d = fabsf(O1.data[(n - 1) * dm + j] - O2.data[(n - 1) * dm + j]);
      if (d > last_diff) last_diff = d;
    }
    int blk_ok = finite && (past_diff < 1e-6f) && (last_diff > 1e-4f);
    printf("    finite=%d | causality: past rows max|diff|=%.2e (==0) last row |diff|=%.2e (>0) -> %s\n\n",
           finite, past_diff, last_diff, blk_ok ? "OK" : "FAIL");
    all_ok &= blk_ok;
    llama_block_free(&blk);
    tensor_free(&X); tensor_free(&X2); tensor_free(&O1); tensor_free(&O2);
  }

  // --------------------------------------------------------------------------
  // Part 18: full modern MODEL (embedding -> N blocks -> RMSNorm -> head).
  // Forward on token ids + model-level causality: changing the LAST token id
  // must leave the logits at every earlier position unchanged.
  // --------------------------------------------------------------------------
  {
    const int vocab = 32, dm = 8, nl = 3, nh = 4, nkv = 2, hd = 2, dff = 16, seq = 6;
    printf("[18] Modern model: %d layers, vocab %d -> forward + causality\n", nl, vocab);
    srand(11);
    LlamaModel lm = llama_model_create(vocab, dm, nl, nh, nkv, hd, dff, 64, 10000.0f, 1e-5f);
    llama_model_randomize(&lm);
    int ids[6]  = {3, 7, 1, 9, 2, 5};
    int ids2[6] = {3, 7, 1, 9, 2, 8};   // differs only in the LAST token
    int ls[2] = {seq, vocab};
    Tensor L1 = tensor_create(ls, 2), L2 = tensor_create(ls, 2);
    llama_model_forward(&lm, ids, seq, &L1);
    llama_model_forward(&lm, ids2, seq, &L2);

    int finite = 1;
    for (int i = 0; i < seq * vocab; i++) if (!isfinite(L1.data[i])) finite = 0;
    float past_diff = 0.0f, last_diff = 0.0f;
    for (int i = 0; i < seq - 1; i++)
      for (int j = 0; j < vocab; j++) {
        float d = fabsf(L1.data[i * vocab + j] - L2.data[i * vocab + j]);
        if (d > past_diff) past_diff = d;
      }
    for (int j = 0; j < vocab; j++) {
      float d = fabsf(L1.data[(seq - 1) * vocab + j] - L2.data[(seq - 1) * vocab + j]);
      if (d > last_diff) last_diff = d;
    }
    int lm_ok = finite && (past_diff < 1e-6f) && (last_diff > 1e-4f);
    printf("    finite=%d | causality: earlier-token logits max|diff|=%.2e (==0) last-token |diff|=%.2e (>0) -> %s\n",
           finite, past_diff, last_diff, lm_ok ? "OK" : "FAIL");
    all_ok &= lm_ok;

    // LLMA checkpoint round-trip: save -> load -> forward must be bit-identical.
    int rt_ok = 0;
    if (llama_model_save("llama_ckpt.tmp", &lm) == 0) {
      LlamaModel lm2;
      if (llama_model_load("llama_ckpt.tmp", &lm2) == 0) {
        Tensor L3 = tensor_create(ls, 2);
        llama_model_forward(&lm2, ids, seq, &L3);
        float md = 0.0f;
        for (int i = 0; i < seq * vocab; i++) { float d = fabsf(L3.data[i] - L1.data[i]); if (d > md) md = d; }
        rt_ok = (md == 0.0f);
        printf("    LLMA checkpoint round-trip: max |logit diff| = %.2e -> %s\n\n", md, rt_ok ? "OK" : "FAIL");
        tensor_free(&L3);
        llama_model_free(&lm2);
      }
    }
    remove("llama_ckpt.tmp");
    all_ok &= rt_ok;

    llama_model_free(&lm);
    tensor_free(&L1); tensor_free(&L2);
  }

  // --------------------------------------------------------------------------
  // Part 19: LlamaModel KV-cache decode must match the full forward. Run the
  // whole sequence through llama_model_forward, then token-by-token through
  // llama_decode_step, and compare per-position logits (correctness gate for
  // the incremental decode used by `transformer llama` generation).
  // --------------------------------------------------------------------------
  {
    printf("[19] LlamaModel KV-cache decode == full forward\n");
    const int vocab = 32, dm = 8, nl = 2, nh = 4, nkv = 2, hd = 2, dff = 16, seq = 6;
    srand(13);
    LlamaModel lm = llama_model_create(vocab, dm, nl, nh, nkv, hd, dff, 64, 10000.0f, 1e-5f);
    llama_model_randomize(&lm);
    int ids[6] = {3, 7, 1, 9, 2, 5};

    int fs[2] = {seq, vocab};
    Tensor Lf = tensor_create(fs, 2);
    llama_model_forward(&lm, ids, seq, &Lf);

    LlamaKV kv = llama_kv_create(&lm);
    int ds[2] = {1, vocab};
    Tensor Ld = tensor_create(ds, 2);
    float maxd = 0.0f;
    for (int i = 0; i < seq; i++) {
      llama_decode_step(&lm, &kv, ids[i], &Ld);
      for (int j = 0; j < vocab; j++) {
        float d = fabsf(Ld.data[j] - Lf.data[i * vocab + j]);
        if (d > maxd) maxd = d;
      }
    }
    int kv_ok = (maxd < 1e-3f);
    printf("    max |logit diff| over all %d positions = %.2e -> %s\n\n", seq, maxd, kv_ok ? "OK" : "FAIL");
    all_ok &= kv_ok;
    tensor_free(&Lf); tensor_free(&Ld);
    llama_kv_free(&kv); llama_model_free(&lm);
  }

  // --------------------------------------------------------------------------
  // Part 20: int8 weight quantization (the LLMQ checkpoint format). Quantize a
  // weight per-column to int8, dequantize, and check the LINEAR-LAYER output is
  // preserved: A @ dequant(quant(W)) vs A @ W. int8 weights are 4x smaller.
  // --------------------------------------------------------------------------
  {
    printf("[20] int8 weight quantization: A @ dequant(quant(W)) vs A @ W\n");
    const int n = 4, rows = 64, cols = 32;    // realistic reduction dim (noise averages out)
    int as[2] = {n, rows}, ws[2] = {rows, cols}, ys[2] = {n, cols};
    Tensor A = tensor_create(as, 2), W = tensor_create(ws, 2), Wdq = tensor_create(ws, 2);
    Tensor Y = tensor_create(ys, 2), Yq = tensor_create(ys, 2);
    for (int i = 0; i < n * rows; i++) A.data[i] = ((i * 13 % 100) / 50.0f) - 1.0f;
    for (int i = 0; i < rows * cols; i++) W.data[i] = ((i * 29 % 100) / 50.0f - 1.0f) * 0.3f;

    signed char *q = (signed char*)malloc((size_t)rows * cols);
    float *scale = (float*)malloc((size_t)cols * sizeof(float));
    quantize_int8_percol(W.data, rows, cols, q, scale);
    dequantize_int8_percol(q, scale, rows, cols, Wdq.data);
    matmul(&A, &W, &Y);
    matmul(&A, &Wdq, &Yq);
    float num = 0.0f, den = 0.0f;                 // L2 relative error ||Yq-Y|| / ||Y||
    for (int i = 0; i < n * cols; i++) {
      float d = Yq.data[i] - Y.data[i];
      num += d * d;
      den += Y.data[i] * Y.data[i];
    }
    float rel = sqrtf(num / den);
    int q_ok = (rel < 0.03f);
    printf("    matmul output L2 rel error = %.2e (int8 weights = 4x smaller) -> %s\n\n",
           rel, q_ok ? "OK" : "FAIL");
    all_ok &= q_ok;
    free(q); free(scale);
    tensor_free(&A); tensor_free(&W); tensor_free(&Wdq); tensor_free(&Y); tensor_free(&Yq);
  }

  // --------------------------------------------------------------------------
  // Part 21: the in-memory int8 quantized model (QLlamaModel + qmatmul) must
  // match the fp32 model within quantization error. Quantize a small fp32 model,
  // run q_llama_decode_step per token, and compare to llama_model_forward.
  // --------------------------------------------------------------------------
  {
    printf("[21] Quantized (int8, in-memory) LlamaModel decode ~= fp32 forward\n");
    const int vocab = 32, dm = 8, nl = 2, nh = 4, nkv = 2, hd = 2, dff = 16, seq = 6;
    srand(17);
    LlamaModel fm = llama_model_create(vocab, dm, nl, nh, nkv, hd, dff, 64, 10000.0f, 1e-5f);
    llama_model_randomize(&fm);
    int ids[6] = {3, 7, 1, 9, 2, 5};
    int fs[2] = {seq, vocab};
    Tensor Lf = tensor_create(fs, 2);
    llama_model_forward(&fm, ids, seq, &Lf);

    QLlamaModel qm = q_llama_quantize_from(&fm);
    LlamaKV kv = q_llama_kv_create(&qm);
    int ds[2] = {1, vocab};
    Tensor Ld = tensor_create(ds, 2);
    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < seq; i++) {
      q_llama_decode_step(&qm, &kv, ids[i], &Ld);
      for (int j = 0; j < vocab; j++) {
        float d = Ld.data[j] - Lf.data[i * vocab + j];
        num += d * d;
        den += Lf.data[i * vocab + j] * Lf.data[i * vocab + j];
      }
    }
    float rel = sqrtf(num / den);
    int qm_ok = (rel < 0.15f);   // tiny dims amplify int8 noise (real models ~<1%)
    printf("    int8 decode vs f32 forward: L2 rel error = %.2e -> %s\n\n", rel, qm_ok ? "OK" : "FAIL");
    all_ok &= qm_ok;
    tensor_free(&Lf); tensor_free(&Ld);
    llama_kv_free(&kv); q_llama_model_free(&qm); llama_model_free(&fm);
  }

  // --------------------------------------------------------------------------
  // Part 22: TRAINING the modern (LLaMA-family) model — llamatrain.c. Three
  // gates, in increasing strength:
  //   (a) the caching training forward must agree BIT-EXACTLY with the plain
  //       inference forward (otherwise the cached activations describe a
  //       different function than the one being differentiated),
  //   (b) an end-to-end numerical gradient check over EVERY parameter tensor —
  //       the same conditioned rel-err < 1e-2 gate the GPT path uses in [2],
  //   (c) Adam must actually drive the loss down on a next-token task.
  // --------------------------------------------------------------------------
  {
    const int vocab = 24, dm = 8, nl = 2, nh = 4, nkv = 2, hd = 2, dff = 16, seq = 5;
    printf("[22] Modern model TRAINING (RMSNorm+RoPE+GQA+SwiGLU backward)\n");
    srand(23);
    LlamaModel tm = llama_model_create_train(vocab, dm, nl, nh, nkv, hd, dff, 64,
                                             10000.0f, 1e-5f);
    llama_model_randomize(&tm);
    LlamaActs acts = llama_acts_create(&tm, seq);
    int ids[5]     = {2, 9, 4, 1, 7};
    int targets[5] = {9, 4, 1, 7, 3};       // next-token task
    int ls[2] = {seq, vocab};
    Tensor logits = tensor_create(ls, 2), probs = tensor_create(ls, 2);
    Tensor ref = tensor_create(ls, 2);

    // (a) training forward == inference forward, bit for bit.
    llama_model_forward(&tm, ids, seq, &ref);
    llama_forward_train(&tm, ids, seq, &acts, &logits);
    float fwd_diff = 0.0f;
    for (int i = 0; i < seq * vocab; i++) {
      float d = fabsf(logits.data[i] - ref.data[i]);
      if (d > fwd_diff) fwd_diff = d;
    }
    int fwd_ok = (fwd_diff == 0.0f);
    printf("    train forward == inference forward: max |diff| = %.2e -> %s\n",
           fwd_diff, fwd_ok ? "OK" : "FAIL");
    all_ok &= fwd_ok;

    // (b) analytic vs central-difference gradients over every parameter.
    Tensor **lps = (Tensor**)malloc((size_t)llama_param_count(&tm) * sizeof(Tensor*));
    int n_lp = llama_collect_params(&tm, lps);
    llama_ce_loss(&tm, &acts, ids, seq, targets, &logits, &probs);
    tensor_zero_grad(&logits);
    cross_entropy_backward(&probs, targets, &logits);
    llama_backward(&tm, ids, seq, &acts, &logits);

    const float leps = 1e-2f;
    float lmax_rel = 0.0f, lmax_cond = 0.0f;
    for (int p = 0; p < n_lp; p++) {
      Tensor *param = lps[p];
      size_t step = param->size > 2 ? param->size / 2 : 1;
      for (size_t k = 0; k < param->size; k += step) {
        float base = param->data[k];
        param->data[k] = base + leps;
        float lp = llama_ce_loss(&tm, &acts, ids, seq, targets, &logits, &probs);
        param->data[k] = base - leps;
        float lmn = llama_ce_loss(&tm, &acts, ids, seq, targets, &logits, &probs);
        param->data[k] = base;
        float numeric = (lp - lmn) / (2.0f * leps);
        float analytic = param->grad[k];
        float rel = fabsf(analytic - numeric) / (fabsf(analytic) + fabsf(numeric) + 1e-8f);
        if (rel > lmax_rel) lmax_rel = rel;
        // Only entries whose gradient clears the float32 finite-difference noise
        // floor are judged (same rule as the GPT check in [2]).
        if (fabsf(analytic) > 1e-2f && rel > lmax_cond) lmax_cond = rel;
      }
    }
    int lgrad_ok = (lmax_cond < 1e-2f);
    printf("    gradient check (%d param tensors): max rel error %.2e (all) | %.2e (|grad|>1e-2) -> %s\n",
           n_lp, lmax_rel, lmax_cond, lgrad_ok ? "OK" : "FAIL");
    all_ok &= lgrad_ok;

    // (c) train: the loss must fall.
    Adam lopt = adam_create(lps, n_lp, 0.02f);
    float lfirst = llama_ce_loss(&tm, &acts, ids, seq, targets, &logits, &probs);
    float llast = lfirst;
    for (int s = 0; s < 200; s++) {
      llast = llama_ce_loss(&tm, &acts, ids, seq, targets, &logits, &probs);
      tensor_zero_grad(&logits);
      cross_entropy_backward(&probs, targets, &logits);
      llama_backward(&tm, ids, seq, &acts, &logits);
      adam_step(&lopt, lps, n_lp);
    }
    int ltrain_ok = (llast < lfirst * 0.5f) && isfinite(llast);
    printf("    train 200 steps (Adam): loss %.4f -> %.4f -> %s\n\n",
           lfirst, llast, ltrain_ok ? "OK" : "FAIL");
    all_ok &= ltrain_ok;

    adam_free(&lopt);
    free(lps);
    tensor_free(&logits); tensor_free(&probs); tensor_free(&ref);
    llama_acts_free(&acts);
    llama_model_free(&tm);
  }

  // --------------------------------------------------------------------------
  // Part 23: training-recipe essentials in optim.c — AdamW decoupled weight
  // decay, global gradient-norm clipping, and the cosine+warmup LR schedule.
  // Each is checked against its CLOSED-FORM value, not just "it ran".
  // --------------------------------------------------------------------------
  {
    printf("[23] Optimizer: AdamW decay + grad clipping + cosine/warmup LR\n");

    // (a) Decoupled decay with ZERO gradient must scale weights by exactly
    //     (1 - lr*wd) per step — and must leave 1-D tensors (biases / norm
    //     gains) untouched.
    int ms[2] = {2, 2}, vs[1] = {3};
    Tensor W = tensor_create(ms, 2), B = tensor_create(vs, 1);
    for (int i = 0; i < 4; i++) W.data[i] = 1.0f;
    for (int i = 0; i < 3; i++) B.data[i] = 1.0f;
    tensor_zero_grad(&W); tensor_zero_grad(&B);
    Tensor *wps[2] = {&W, &B};
    const float wlr = 0.1f, wwd = 0.5f;
    Adam wopt = adamw_create(wps, 2, wlr, wwd);
    adam_step(&wopt, wps, 2);
    float want = 1.0f * (1.0f - wlr * wwd);            // 0.95
    float dec_err = fabsf(W.data[0] - want);
    float bias_err = fabsf(B.data[0] - 1.0f);          // must be exactly untouched
    int wd_ok = (dec_err < 1e-6f) && (bias_err == 0.0f);
    printf("    AdamW: matrix %.6f (want %.6f) | 1-D bias %.6f (want 1.000000, undecayed) -> %s\n",
           W.data[0], want, B.data[0], wd_ok ? "OK" : "FAIL");
    all_ok &= wd_ok;
    adam_free(&wopt);

    // Plain adam_create must stay pure Adam: zero grad => zero movement.
    for (int i = 0; i < 4; i++) W.data[i] = 1.0f;
    tensor_zero_grad(&W);
    Adam popt = adam_create(wps, 1, wlr);
    adam_step(&popt, wps, 1);
    int plain_ok = (W.data[0] == 1.0f);
    printf("    plain Adam unchanged by the AdamW addition (wd=0): w = %.6f -> %s\n",
           W.data[0], plain_ok ? "OK" : "FAIL");
    all_ok &= plain_ok;
    adam_free(&popt);

    // (b) Global grad-norm clipping. Grads: four 3.0 => norm = sqrt(4*9) = 6.
    for (int i = 0; i < 4; i++) W.grad[i] = 3.0f;
    for (int i = 0; i < 3; i++) B.grad[i] = 0.0f;
    float n_under = clip_grad_norm(wps, 2, 10.0f);     // above max => untouched
    int under_ok = (fabsf(n_under - 6.0f) < 1e-5f) && (W.grad[0] == 3.0f);
    float n_over = clip_grad_norm(wps, 2, 2.0f);       // clip 6 -> 2
    float post = 0.0f;
    for (int i = 0; i < 4; i++) post += W.grad[i] * W.grad[i];
    post = sqrtf(post);
    int clip_ok = under_ok && (fabsf(n_over - 6.0f) < 1e-5f) && (fabsf(post - 2.0f) < 1e-4f);
    printf("    clip: norm %.4f (want 6) | no-op under max, grad still %.1f | after clip to 2: %.4f -> %s\n",
           n_over, 3.0f, post, clip_ok ? "OK" : "FAIL");
    all_ok &= clip_ok;

    // A NaN gradient must zero every grad (drop the batch) rather than poison weights.
    W.grad[0] = 0.0f / 0.0f;
    float n_nan = clip_grad_norm(wps, 2, 1.0f);
    int nan_ok = !isfinite(n_nan) && (W.grad[0] == 0.0f) && (W.grad[1] == 0.0f);
    printf("    clip: NaN grad -> norm %s, all grads zeroed -> %s\n",
           isfinite(n_nan) ? "finite(BAD)" : "non-finite", nan_ok ? "OK" : "FAIL");
    all_ok &= nan_ok;
    tensor_free(&W); tensor_free(&B);

    // (c) Cosine schedule with linear warmup: base 1.0, min 0.1, warmup 10, total 110.
    const float b_lr = 1.0f, m_lr = 0.1f;
    const int wu = 10, tot = 110;
    float s0 = lr_cosine_warmup(b_lr, m_lr, 0, wu, tot);      // first warmup step: 0.1
    float speak = lr_cosine_warmup(b_lr, m_lr, wu - 1, wu, tot);  // end of warmup: 1.0
    float smid = lr_cosine_warmup(b_lr, m_lr, 60, wu, tot);   // halfway: 0.55
    float send = lr_cosine_warmup(b_lr, m_lr, tot, wu, tot);  // clamped: 0.1
    int mono = 1;                                             // non-increasing after warmup
    float prev = speak;
    for (int s = wu; s <= tot; s++) {
      float cur = lr_cosine_warmup(b_lr, m_lr, s, wu, tot);
      if (cur > prev + 1e-6f) mono = 0;
      prev = cur;
    }
    int sched_ok = fabsf(s0 - 0.1f) < 1e-6f && fabsf(speak - 1.0f) < 1e-6f &&
                   fabsf(smid - 0.55f) < 1e-4f && fabsf(send - m_lr) < 1e-6f && mono;
    printf("    LR: warmup[0]=%.4f (0.1) peak=%.4f (1.0) mid=%.4f (0.55) end=%.4f (0.1) monotone=%d -> %s\n\n",
           s0, speak, smid, send, mono, sched_ok ? "OK" : "FAIL");
    all_ok &= sched_ok;
  }

  // --------------------------------------------------------------------------
  // Part 24: LoRA fine-tuning of a FROZEN base model. Four gates:
  //   (a) at init (B == 0) the adapted forward is BIT-IDENTICAL to the base —
  //       the property that makes attaching adapters to a pretrained checkpoint
  //       safe,
  //   (b) numerical gradient check over every adapter tensor,
  //   (c) training only the adapters drives the loss down,
  //   (d) the base weights are BYTE-for-BYTE unchanged afterwards (truly frozen).
  // --------------------------------------------------------------------------
  {
    const int vocab = 24, dm = 8, nl = 2, nh = 4, nkv = 2, hd = 2, dff = 16, seq = 5;
    const int rank = 4;
    printf("[24] LoRA fine-tuning (rank %d) on a frozen LLaMA-family base\n", rank);
    srand(31);
    LlamaModel bm = llama_model_create_train(vocab, dm, nl, nh, nkv, hd, dff, 64,
                                             10000.0f, 1e-5f);
    llama_model_randomize(&bm);
    LlamaActs bacts = llama_acts_create(&bm, seq);
    LlamaLora lo = llama_lora_create(&bm, rank, 8.0f);
    llama_lora_init(&lo);
    LlamaLoraActs la = llama_lora_acts_create(&bm, &lo, seq);

    int ids[5]     = {5, 2, 8, 1, 6};
    int targets[5] = {2, 8, 1, 6, 4};
    int ls[2] = {seq, vocab};
    Tensor blog = tensor_create(ls, 2), bprob = tensor_create(ls, 2);
    Tensor base_ref = tensor_create(ls, 2);

    // (a) B == 0 at init => adapted output must equal the base exactly.
    llama_forward_train(&bm, ids, seq, &bacts, &base_ref);
    llama_forward_train_lora(&bm, ids, seq, &bacts, &blog, &lo, &la);
    float init_diff = 0.0f;
    for (int i = 0; i < seq * vocab; i++) {
      float d = fabsf(blog.data[i] - base_ref.data[i]);
      if (d > init_diff) init_diff = d;
    }
    int init_ok = (init_diff == 0.0f);
    printf("    at init (B=0): adapted == base, max |diff| = %.2e -> %s\n",
           init_diff, init_ok ? "OK" : "FAIL");
    all_ok &= init_ok;

    // Snapshot every base weight so (d) can prove nothing moved.
    Tensor **bp = (Tensor**)malloc((size_t)llama_param_count(&bm) * sizeof(Tensor*));
    int n_bp = llama_collect_params(&bm, bp);
    size_t base_total = 0;
    for (int p = 0; p < n_bp; p++) base_total += bp[p]->size;
    float *snapshot = (float*)malloc(base_total * sizeof(float));
    size_t at = 0;
    for (int p = 0; p < n_bp; p++) {
      memcpy(&snapshot[at], bp[p]->data, bp[p]->size * sizeof(float));
      at += bp[p]->size;
    }

    Tensor **lp = (Tensor**)malloc((size_t)llama_lora_param_count(&lo) * sizeof(Tensor*));
    int n_lp2 = llama_lora_collect_params(&lo, lp);

    // Give B some nonzero values first: with B == 0 the gradient wrt A is
    // identically zero, so a check at init would be vacuous.
    for (int p = 0; p < n_lp2; p++)
      for (size_t k = 0; k < lp[p]->size; k++)
        lp[p]->data[k] += 0.05f * (float)((int)((k * 37 + (size_t)p * 11) % 13) - 6);

    llama_forward_train_lora(&bm, ids, seq, &bacts, &blog, &lo, &la);
    cross_entropy_forward(&blog, targets, &bprob);
    tensor_zero_grad(&blog);
    cross_entropy_backward(&bprob, targets, &blog);
    llama_backward_lora(&bm, ids, seq, &bacts, &blog, &lo, &la);

    // (b) central-difference check over the adapter tensors.
    const float ceps = 1e-2f;
    float lo_max = 0.0f, lo_cond = 0.0f;
    for (int p = 0; p < n_lp2; p++) {
      Tensor *param = lp[p];
      size_t step = param->size > 2 ? param->size / 2 : 1;
      for (size_t k = 0; k < param->size; k += step) {
        float base = param->data[k];
        param->data[k] = base + ceps;
        llama_forward_train_lora(&bm, ids, seq, &bacts, &blog, &lo, &la);
        float lpv = cross_entropy_forward(&blog, targets, &bprob);
        param->data[k] = base - ceps;
        llama_forward_train_lora(&bm, ids, seq, &bacts, &blog, &lo, &la);
        float lmv = cross_entropy_forward(&blog, targets, &bprob);
        param->data[k] = base;
        float numeric = (lpv - lmv) / (2.0f * ceps);
        float analytic = param->grad[k];
        float rel = fabsf(analytic - numeric) / (fabsf(analytic) + fabsf(numeric) + 1e-8f);
        if (rel > lo_max) lo_max = rel;
        if (fabsf(analytic) > 1e-2f && rel > lo_cond) lo_cond = rel;
      }
    }
    int lo_grad_ok = (lo_cond < 1e-2f);
    printf("    gradient check (%d adapter tensors): max rel error %.2e (all) | %.2e (|grad|>1e-2) -> %s\n",
           n_lp2, lo_max, lo_cond, lo_grad_ok ? "OK" : "FAIL");
    all_ok &= lo_grad_ok;

    // (c) The realistic scenario: PRETRAIN the base on task A, freeze it, then
    //     adapt to a DIFFERENT task B with adapters only. (Adapting a randomly
    //     initialized base is not a fair test — attention-only adapters have no
    //     useful representation to steer, and the loss plateaus near ln(vocab).)
    Tensor **bpar = (Tensor**)malloc((size_t)llama_param_count(&bm) * sizeof(Tensor*));
    int n_bpar = llama_collect_params(&bm, bpar);
    Adam pre = adam_create(bpar, n_bpar, 0.02f);
    for (int s = 0; s < 300; s++) {
      llama_forward_train(&bm, ids, seq, &bacts, &blog);
      cross_entropy_forward(&blog, targets, &bprob);
      tensor_zero_grad(&blog);
      cross_entropy_backward(&bprob, targets, &blog);
      llama_backward(&bm, ids, seq, &bacts, &blog);
      adam_step(&pre, bpar, n_bpar);
    }
    llama_forward_train(&bm, ids, seq, &bacts, &blog);
    float task_a = cross_entropy_forward(&blog, targets, &bprob);
    adam_free(&pre); free(bpar);

    // Fresh adapters (A random, B = 0) on the now-pretrained base.
    llama_lora_init(&lo);

    // Re-snapshot AFTER pretraining — that is the base the adapters must not touch.
    at = 0;
    for (int p = 0; p < n_bp; p++) {
      memcpy(&snapshot[at], bp[p]->data, bp[p]->size * sizeof(float));
      at += bp[p]->size;
    }

    // Task B: a different next-token mapping the frozen base was never trained on.
    int targets_b[5] = {7, 3, 5, 2, 9};
    llama_forward_train_lora(&bm, ids, seq, &bacts, &blog, &lo, &la);
    float lo_first = cross_entropy_forward(&blog, targets_b, &bprob);

    const int lo_steps = 400;
    Adam lo_opt = adam_create(lp, n_lp2, 0.05f);
    float lo_last = lo_first;
    for (int s = 0; s < lo_steps; s++) {
      llama_forward_train_lora(&bm, ids, seq, &bacts, &blog, &lo, &la);
      lo_last = cross_entropy_forward(&blog, targets_b, &bprob);
      tensor_zero_grad(&blog);
      cross_entropy_backward(&bprob, targets_b, &blog);
      llama_backward_lora(&bm, ids, seq, &bacts, &blog, &lo, &la);
      adam_step(&lo_opt, lp, n_lp2);
    }
    int lo_train_ok = (lo_last < lo_first * 0.5f) && isfinite(lo_last);
    printf("    base pretrained on task A (loss %.4f), then ADAPTERS ONLY on a new task B:\n"
           "      %d steps, loss %.4f -> %.4f -> %s\n",
           task_a, lo_steps, lo_first, lo_last, lo_train_ok ? "OK" : "FAIL");
    all_ok &= lo_train_ok;

    // (d) the frozen base must be byte-for-byte identical.
    at = 0;
    int frozen = 1;
    for (int p = 0; p < n_bp && frozen; p++) {
      if (memcmp(&snapshot[at], bp[p]->data, bp[p]->size * sizeof(float)) != 0) frozen = 0;
      at += bp[p]->size;
    }
    size_t n_lora_scalars = 0;
    for (int p = 0; p < n_lp2; p++) n_lora_scalars += lp[p]->size;
    printf("    base frozen: %zu base scalars unchanged=%d | trained %zu adapter scalars (%.1f%%) -> %s\n",
           base_total, frozen, n_lora_scalars,
           100.0 * (double)n_lora_scalars / (double)base_total, frozen ? "OK" : "FAIL");
    all_ok &= frozen;

    // (e) Merging the adapters into the base must reproduce the adapted output —
    //     that equivalence is what lets a fine-tune ship as an ordinary LLMA
    //     checkpoint with no adapter support at inference time.
    llama_forward_train_lora(&bm, ids, seq, &bacts, &blog, &lo, &la);
    Tensor merged = tensor_create(ls, 2);
    llama_lora_merge(&bm, &lo);
    llama_forward_train(&bm, ids, seq, &bacts, &merged);
    float mdiff = 0.0f, mscale = 0.0f;
    for (int i = 0; i < seq * vocab; i++) {
      float d = fabsf(merged.data[i] - blog.data[i]);
      if (d > mdiff) mdiff = d;
      if (fabsf(blog.data[i]) > mscale) mscale = fabsf(blog.data[i]);
    }
    // Not bit-exact: merging reassociates the arithmetic (W + scale*A@B applied
    // once, vs the correction added after the projection), so judge relatively.
    int merge_ok = (mdiff / (mscale + 1e-6f) < 1e-4f);
    printf("    merge adapters into base: max |diff| %.2e vs |logit| %.2e (rel %.2e) -> %s\n\n",
           mdiff, mscale, mdiff / (mscale + 1e-6f), merge_ok ? "OK" : "FAIL");
    all_ok &= merge_ok;
    tensor_free(&merged);

    adam_free(&lo_opt);
    free(snapshot); free(bp); free(lp);
    tensor_free(&blog); tensor_free(&bprob); tensor_free(&base_ref);
    llama_lora_acts_free(&la); llama_lora_free(&lo);
    llama_acts_free(&bacts); llama_model_free(&bm);
  }

  // --------------------------------------------------------------------------
  // Part 25: gradient ACCUMULATION (--batch). The claim being tested is exact:
  // accumulating two different sequences and scaling by 1/2 must equal the MEAN
  // of their individually computed gradients. If that failed, --batch would
  // silently train on something other than the batch gradient.
  // --------------------------------------------------------------------------
  {
    const int vocab = 20, dm = 8, nl = 2, nh = 4, nkv = 2, hd = 2, dff = 16, seq = 4;
    printf("[25] Gradient accumulation (--batch): accum/B == mean of per-sample grads\n");
    srand(41);
    LlamaModel am = llama_model_create_train(vocab, dm, nl, nh, nkv, hd, dff, 32,
                                             10000.0f, 1e-5f);
    llama_model_randomize(&am);
    LlamaActs aa = llama_acts_create(&am, seq);
    int ls2[2] = {seq, vocab};
    Tensor alog = tensor_create(ls2, 2), aprob = tensor_create(ls2, 2);

    int idsA[4] = {3, 1, 7, 2}, tgtA[4] = {1, 7, 2, 9};
    int idsB[4] = {5, 8, 2, 6}, tgtB[4] = {8, 2, 6, 4};

    Tensor **ap = (Tensor**)malloc((size_t)llama_param_count(&am) * sizeof(Tensor*));
    int n_ap = llama_collect_params(&am, ap);
    size_t tot = 0;
    for (int p = 0; p < n_ap; p++) tot += ap[p]->size;
    float *gA = (float*)malloc(tot * sizeof(float));
    float *gB = (float*)malloc(tot * sizeof(float));

    // Individual gradients.
    llama_forward_train(&am, idsA, seq, &aa, &alog);
    cross_entropy_forward(&alog, tgtA, &aprob);
    tensor_zero_grad(&alog); cross_entropy_backward(&aprob, tgtA, &alog);
    llama_backward(&am, idsA, seq, &aa, &alog);
    size_t at2 = 0;
    for (int p = 0; p < n_ap; p++) { memcpy(&gA[at2], ap[p]->grad, ap[p]->size * sizeof(float)); at2 += ap[p]->size; }

    llama_forward_train(&am, idsB, seq, &aa, &alog);
    cross_entropy_forward(&alog, tgtB, &aprob);
    tensor_zero_grad(&alog); cross_entropy_backward(&aprob, tgtB, &alog);
    llama_backward(&am, idsB, seq, &aa, &alog);
    at2 = 0;
    for (int p = 0; p < n_ap; p++) { memcpy(&gB[at2], ap[p]->grad, ap[p]->size * sizeof(float)); at2 += ap[p]->size; }

    // Accumulated over both, then scaled by 1/2 — exactly what --batch 2 does.
    llama_zero_grad(&am);
    llama_forward_train(&am, idsA, seq, &aa, &alog);
    cross_entropy_forward(&alog, tgtA, &aprob);
    tensor_zero_grad(&alog); cross_entropy_backward(&aprob, tgtA, &alog);
    llama_backward_accum(&am, idsA, seq, &aa, &alog, NULL, NULL);
    llama_forward_train(&am, idsB, seq, &aa, &alog);
    cross_entropy_forward(&alog, tgtB, &aprob);
    tensor_zero_grad(&alog); cross_entropy_backward(&aprob, tgtB, &alog);
    llama_backward_accum(&am, idsB, seq, &aa, &alog, NULL, NULL);
    scale_grads(ap, n_ap, 0.5f);

    float amax = 0.0f, gmax = 0.0f;
    at2 = 0;
    for (int p = 0; p < n_ap; p++) {
      for (size_t k = 0; k < ap[p]->size; k++) {
        float want = 0.5f * (gA[at2 + k] + gB[at2 + k]);
        float d = fabsf(ap[p]->grad[k] - want);
        if (d > amax) amax = d;
        if (fabsf(want) > gmax) gmax = fabsf(want);
      }
      at2 += ap[p]->size;
    }
    int acc_ok = (amax / (gmax + 1e-9f) < 1e-5f);
    printf("    accum(2)/2 vs mean of separate grads: max |diff| %.2e vs |grad| %.2e (rel %.2e) -> %s\n",
           amax, gmax, amax / (gmax + 1e-9f), acc_ok ? "OK" : "FAIL");
    all_ok &= acc_ok;

    // A single accumulated micro-batch must equal the plain backward exactly.
    llama_zero_grad(&am);
    llama_forward_train(&am, idsA, seq, &aa, &alog);
    cross_entropy_forward(&alog, tgtA, &aprob);
    tensor_zero_grad(&alog); cross_entropy_backward(&aprob, tgtA, &alog);
    llama_backward_accum(&am, idsA, seq, &aa, &alog, NULL, NULL);
    float one_max = 0.0f;
    at2 = 0;
    for (int p = 0; p < n_ap; p++) {
      for (size_t k = 0; k < ap[p]->size; k++) {
        float d = fabsf(ap[p]->grad[k] - gA[at2 + k]);
        if (d > one_max) one_max = d;
      }
      at2 += ap[p]->size;
    }
    int one_ok = (one_max == 0.0f);
    printf("    batch=1 accumulate == plain backward: max |diff| %.2e -> %s\n\n",
           one_max, one_ok ? "OK" : "FAIL");
    all_ok &= one_ok;

    free(gA); free(gB); free(ap);
    tensor_free(&alog); tensor_free(&aprob);
    llama_acts_free(&aa); llama_model_free(&am);
  }

  // --------------------------------------------------------------------------
  // Part 26: the LoRA-aware KV-cache decode must match the LoRA training
  // forward. This is the gate that keeps RL rollouts honest: the rollout is
  // sampled through llama_decode_step_lora and the update runs through
  // llama_forward_train_lora, so if the two apply the adapters differently (a
  // wrong order relative to RoPE, say) then every importance ratio GRPO computes
  // is measuring two different policies against each other.
  // --------------------------------------------------------------------------
  {
    printf("[26] LoRA KV-cache decode == LoRA training forward\n");
    const int vocab = 32, dm = 8, nl = 2, nh = 4, nkv = 2, hd = 2, dff = 16, seq = 6;
    srand(17);
    LlamaModel lm = llama_model_create_train(vocab, dm, nl, nh, nkv, hd, dff, 64,
                                             10000.0f, 1e-5f);
    llama_model_randomize(&lm);
    LlamaLora lo = llama_lora_create(&lm, 4, 8.0f);
    llama_lora_init(&lo);
    LlamaLoraActs la = llama_lora_acts_create(&lm, &lo, seq);
    LlamaActs la_acts = llama_acts_create(&lm, seq);

    // B starts at zero, which would make this test pass trivially against the
    // base path — give the adapters a real, nonzero correction first.
    Tensor **lp = (Tensor**)malloc((size_t)llama_lora_param_count(&lo) * sizeof(Tensor*));
    int n_lp = llama_lora_collect_params(&lo, lp);
    for (int p = 0; p < n_lp; p++)
      for (size_t k = 0; k < lp[p]->size; k++)
        lp[p]->data[k] += 0.05f * (float)((int)((k * 29 + (size_t)p * 7) % 11) - 5);

    int ids[6] = {3, 7, 1, 9, 2, 5};
    int fs[2] = {seq, vocab};
    Tensor Lf = tensor_create(fs, 2);
    llama_forward_train_lora(&lm, ids, seq, &la_acts, &Lf, &lo, &la);

    LlamaDecodeLora dv = llama_lora_decode_view(&lo);
    LlamaKV kv = llama_kv_create(&lm);
    int ds[2] = {1, vocab};
    Tensor Ld = tensor_create(ds, 2);
    float maxd = 0.0f;
    for (int i = 0; i < seq; i++) {
      llama_decode_step_lora(&lm, &kv, ids[i], &Ld, &dv);
      for (int j = 0; j < vocab; j++) {
        float d = fabsf(Ld.data[j] - Lf.data[i * vocab + j]);
        if (d > maxd) maxd = d;
      }
    }
    int lkv_ok = (maxd < 1e-3f);
    printf("    max |logit diff| over all %d positions = %.2e -> %s\n", seq, maxd,
           lkv_ok ? "OK" : "FAIL");
    all_ok &= lkv_ok;

    // A NULL view must be bit-identical to the plain, adapter-free decode.
    LlamaKV kv2 = llama_kv_create(&lm);
    LlamaKV kv3 = llama_kv_create(&lm);
    Tensor La = tensor_create(ds, 2), Lb = tensor_create(ds, 2);
    float nulld = 0.0f;
    for (int i = 0; i < seq; i++) {
      llama_decode_step_lora(&lm, &kv2, ids[i], &La, NULL);
      llama_decode_step(&lm, &kv3, ids[i], &Lb);
      for (int j = 0; j < vocab; j++) {
        float d = fabsf(La.data[j] - Lb.data[j]);
        if (d > nulld) nulld = d;
      }
    }
    int null_ok = (nulld == 0.0f);
    printf("    NULL adapters == plain decode: max |diff| %.2e -> %s\n\n",
           nulld, null_ok ? "OK" : "FAIL");
    all_ok &= null_ok;

    free(lp);
    tensor_free(&Lf); tensor_free(&Ld); tensor_free(&La); tensor_free(&Lb);
    llama_kv_free(&kv); llama_kv_free(&kv2); llama_kv_free(&kv3);
    llama_lora_acts_free(&la); llama_lora_free(&lo);
    llama_acts_free(&la_acts); llama_model_free(&lm);
  }

  // --------------------------------------------------------------------------
  // Part 27: the verifiable reward (reward.h). An RL run is only as trustworthy
  // as its reward, so this probes the adversarial cases the reward EXISTS to
  // catch — a hallucinated filename, an empty write body, the ~5% char-slip
  // (demo_note -> demonstra_note) — alongside responses that must score full
  // marks. Mirrors the clean_sft adversarial probe the SFT data gate uses.
  // --------------------------------------------------------------------------
  {
    printf("[27] Verifiable reward: grounding, format and action match\n");
    const char *prompt =
      "USER: create notes.txt containing hello world and read it back\nMODEL:";
    const char *ref_write = "<tool name=\"write_file\" path=\"notes.txt\">hello world</tool>";

    struct { const char *resp; const char *ref; int want_full; const char *what; } cases[] = {
      // Must score a perfect 1.0 — well formed, grounded, and the right action.
      {"<tool name=\"write_file\" path=\"notes.txt\">hello world</tool>", ref_write, 1,
       "exact correct action"},
      {"<think>I should write the file first</think><tool name=\"write_file\" "
       "path=\"notes.txt\">hello world</tool>", ref_write, 1, "reasoning then action"},
      {"<tool name=\"read_file\" path=\"notes.txt\"></tool>",
       "<tool name=\"read_file\" path=\"notes.txt\"></tool>", 1, "read_file"},
      {"<done>wrote and read notes.txt</done>", "<done>all done</done>", 1,
       "done (summary text is free-form)"},
      // Must LOSE marks — each for a specific, verifiable reason.
      {"<tool name=\"write_file\" path=\"other.txt\">hello world</tool>", ref_write, 0,
       "hallucinated path"},
      {"<tool name=\"write_file\" path=\"notes.txt\"></tool>", ref_write, 0,
       "empty write body"},
      {"<tool name=\"write_file\" path=\"demonstra_note.txt\">hello world</tool>", ref_write, 0,
       "char-slipped filename"},
      {"<tool name=\"read_file\" path=\"notes.txt\">", ref_write, 0, "unterminated block"},
      {"<tool name=\"read_file\" path=\"notes.txt\"></tool> and then I will stop", ref_write, 0,
       "trailing rambling"},
      {"<tool name=\"write_file\" path=\"notes.txt\">hello world</tool>",
       "<tool name=\"read_file\" path=\"notes.txt\"></tool>", 0, "wrong tool"},
    };
    const int ncase = (int)(sizeof(cases) / sizeof(cases[0]));
    int rw_ok = 1;
    for (int i = 0; i < ncase; i++) {
      RewardParts p;
      float r = reward_score(prompt, cases[i].resp, cases[i].ref, &p);
      const int full = (r > 0.999f);
      if (full != cases[i].want_full) {
        rw_ok = 0;
        printf("    MISMATCH: %-30s r=%.2f (fmt %.2f grd %.2f mat %.2f) %s\n",
               cases[i].what, (double)r, (double)p.format, (double)p.grounded,
               (double)p.match, p.reason);
      }
    }
    printf("    %d/%d cases scored as expected (4 perfect, 6 penalized) -> %s\n",
           rw_ok ? ncase : ncase - 1, ncase, rw_ok ? "OK" : "FAIL");
    all_ok &= rw_ok;

    // The whole-token rule: "ok" must not match inside "cooking".
    int tok_rule = reward_grounded_in("i am cooking dinner", "ok") == 0 &&
                   reward_grounded_in("read notes.txt now", "notes.txt") == 1 &&
                   reward_grounded_in("Read NOTES.TXT now", "notes.txt") == 1;
    printf("    whole-token grounding ('ok' not inside 'cooking', case-insensitive) -> %s\n\n",
           tok_rule ? "OK" : "FAIL");
    all_ok &= tok_rule;
  }

  // --------------------------------------------------------------------------
  // Part 28: the GRPO objective as arithmetic (grpo.h). Three gates:
  //   (a) a group whose rewards are all equal carries NO signal — every
  //       advantage is zero and the trainer must skip it,
  //   (b) advantages are the group's own z-scores (mean 0), which is what makes
  //       the group's mean the baseline that replaces a value network,
  //   (c) the ratio clip fires one-sidedly: it zeroes a token only when the
  //       policy has ALREADY moved too far in the direction the advantage wants.
  // --------------------------------------------------------------------------
  {
    printf("[28] GRPO objective: group-relative advantage + ratio clipping\n");
    GrpoConfig gc = grpo_default_config();
    gc.clip_eps = 0.2f;
    gc.kl_beta = 0.0f;

    // (a) zero variance => no signal
    float flat_r[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float adv[4];
    int live_flat = grpo_advantages(flat_r, 4, &gc, adv);
    int flat_ok = (live_flat == 0);
    for (int i = 0; i < 4; i++) if (adv[i] != 0.0f) flat_ok = 0;
    printf("    all-equal rewards -> skipped, advantages all zero -> %s\n",
           flat_ok ? "OK" : "FAIL");
    all_ok &= flat_ok;

    // (b) advantages are z-scores: mean 0, better-than-average positive.
    float rew[4] = {1.0f, 0.0f, 0.5f, 0.5f};
    int live = grpo_advantages(rew, 4, &gc, adv);
    double amean = 0.0;
    for (int i = 0; i < 4; i++) amean += adv[i];
    int adv_ok = live && fabs(amean) < 1e-5 && adv[0] > 0.0f && adv[1] < 0.0f &&
                 fabsf(adv[2]) < 1e-5f && fabsf(adv[3]) < 1e-5f;
    printf("    advantages %.3f %.3f %.3f %.3f (mean %.1e; best>0, worst<0) -> %s\n",
           (double)adv[0], (double)adv[1], (double)adv[2], (double)adv[3], amean,
           adv_ok ? "OK" : "FAIL");
    all_ok &= adv_ok;

    // (c) clipping. Four tokens: ratio well inside the band, far above it, and
    //     far below it — checked against both a positive and a negative advantage.
    const float lp_old[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // rho = exp(lp_new - lp_old): 1.0, 1.5 (above 1+eps), 0.5 (below 1-eps), 1.0
    const float lp_new[4] = {0.0f, 0.405465f, -0.693147f, 0.0f};
    float cpos[4], cneg[4];
    const float n14 = grpo_loss_norm(&gc, 1, 4, 0);     // seq: G * len = 4
    grpo_token_coeffs(+1.0f, n14, 4, lp_new, lp_old, NULL, &gc, cpos);
    grpo_token_coeffs(-1.0f, n14, 4, lp_new, lp_old, NULL, &gc, cneg);
    // Positive advantage: the HIGH ratio is clipped away, the low one is kept.
    // Negative advantage: the LOW ratio is clipped away, the high one is kept.
    int clip_ok = (cpos[0] != 0.0f) && (cpos[1] == 0.0f) && (cpos[2] != 0.0f) &&
                  (cneg[0] != 0.0f) && (cneg[1] != 0.0f) && (cneg[2] == 0.0f);
    // And an unclipped token is exactly adv * rho / (G * len).
    float want = 1.0f * 1.0f / (1.0f * 4.0f);
    clip_ok &= (fabsf(cpos[0] - want) < 1e-6f);
    printf("    clip eps 0.2: adv>0 drops rho=1.5, adv<0 drops rho=0.5, "
           "unclipped = adv*rho/(G*len) -> %s\n", clip_ok ? "OK" : "FAIL");
    all_ok &= clip_ok;

    // (d) DAPO clip-higher: rho = 1.25 sits between 1+0.2 and 1+0.28. Symmetric
    //     clipping drops it for adv > 0; clip-higher keeps it. The LOWER bound
    //     must not move: rho = 0.75 with adv < 0 is dropped either way.
    {
      const float hi_new[2] = {0.223144f, -0.287682f};   // rho 1.25, 0.75
      const float hi_old[2] = {0.0f, 0.0f};
      GrpoConfig sym = gc, dapo = gc;
      grpo_preset_dapo(&dapo);
      float s_pos[2], d_pos[2], s_neg[2], d_neg[2];
      grpo_token_coeffs(+1.0f, 2.0f, 2, hi_new, hi_old, NULL, &sym, s_pos);
      grpo_token_coeffs(+1.0f, 2.0f, 2, hi_new, hi_old, NULL, &dapo, d_pos);
      grpo_token_coeffs(-1.0f, 2.0f, 2, hi_new, hi_old, NULL, &sym, s_neg);
      grpo_token_coeffs(-1.0f, 2.0f, 2, hi_new, hi_old, NULL, &dapo, d_neg);
      int hi_ok = (s_pos[0] == 0.0f) && (d_pos[0] != 0.0f) &&
                  (s_neg[1] == 0.0f) && (d_neg[1] == 0.0f) &&
                  fabsf(d_pos[0] - 1.25f / 2.0f) < 1e-5f;
      printf("    clip-higher 0.28: adv>0 keeps rho=1.25 (symmetric drops it), "
             "lower bound unchanged -> %s\n", hi_ok ? "OK" : "FAIL");
      all_ok &= hi_ok;
    }

    // (e) Dr. GRPO advantage: r - mean, no std division. For {1, 0, .5, .5} the
    //     mean is 0.5, so exactly {+.5, -.5, 0, 0}.
    {
      GrpoConfig dr = gc;
      grpo_preset_dr_grpo(&dr);
      float dadv[4], dflat[4];
      int dlive = grpo_advantages(rew, 4, &dr, dadv);
      int dr_ok = dlive && fabsf(dadv[0] - 0.5f) < 1e-6f && fabsf(dadv[1] + 0.5f) < 1e-6f &&
                  fabsf(dadv[2]) < 1e-6f && fabsf(dadv[3]) < 1e-6f;
      // ...and a flat group is still flat without the std.
      dr_ok &= (grpo_advantages(flat_r, 4, &dr, dflat) == 0);
      printf("    no-std advantages %.3f %.3f %.3f %.3f = r - mean, flat still skipped -> %s\n",
             (double)dadv[0], (double)dadv[1], (double)dadv[2], (double)dadv[3],
             dr_ok ? "OK" : "FAIL");
      all_ok &= dr_ok;
    }

    // (f) the three loss normalizers, for G = 4, max_new = 96, a 10-token rollout,
    //     60 live tokens in the step: seq 40, token 60, const 384.
    {
      GrpoConfig a = gc;
      a.group = 4; a.max_new = 96;
      a.loss_agg = GRPO_AGG_SEQ;   const float ns = grpo_loss_norm(&a, 4, 10, 60);
      a.loss_agg = GRPO_AGG_TOKEN; const float nt = grpo_loss_norm(&a, 4, 10, 60);
      a.loss_agg = GRPO_AGG_CONST; const float nc = grpo_loss_norm(&a, 4, 10, 60);
      int agg_ok = ns == 40.0f && nt == 60.0f && nc == 384.0f;
      printf("    loss-agg normalizers seq %.0f / token %.0f / const %.0f -> %s\n",
             (double)ns, (double)nt, (double)nc, agg_ok ? "OK" : "FAIL");
      all_ok &= agg_ok;
    }

    // pg_backward with a uniform 1/N coefficient must reproduce cross-entropy
    // exactly — the identity the whole design rests on.
    {
      const int N = 4, Vv = 6;
      int sh[2] = {N, Vv};
      Tensor lg = tensor_create(sh, 2), pr = tensor_create(sh, 2);
      Tensor lg2 = tensor_create(sh, 2);
      for (int i = 0; i < N * Vv; i++) {
        lg.data[i] = 0.1f * (float)((i * 7) % 11) - 0.5f;
        lg2.data[i] = lg.data[i];
      }
      int tg[4] = {1, 3, -1, 5};                 // one masked row
      cross_entropy_forward(&lg, tg, &pr);
      tensor_zero_grad(&lg); tensor_zero_grad(&lg2);
      cross_entropy_backward(&pr, tg, &lg);
      float co[4];
      for (int i = 0; i < N; i++) co[i] = 1.0f / 3.0f;   // 3 unmasked rows
      pg_backward(&pr, tg, co, &lg2);
      float pmax = 0.0f;
      for (int i = 0; i < N * Vv; i++) {
        float d = fabsf(lg.grad[i] - lg2.grad[i]);
        if (d > pmax) pmax = d;
      }
      int pg_ok = (pmax < 1e-7f);
      printf("    pg_backward(coeff=1/N) == cross_entropy_backward: max |diff| %.2e -> %s\n\n",
             pmax, pg_ok ? "OK" : "FAIL");
      all_ok &= pg_ok;
      tensor_free(&lg); tensor_free(&lg2); tensor_free(&pr);
    }
  }

  // --------------------------------------------------------------------------
  // Part 29: the policy-gradient path end to end, in two gates.
  //   (a) NUMERICAL GRADIENT CHECK of pg_backward composed with the model
  //       backward. With coefficients held fixed, the surrogate is the scalar
  //       J = sum_t coeff[t] * (-log p(target_t)), so it can be finite-
  //       differenced like every other gradient in this file.
  //   (b) a real (tiny) GRPO loop: sample groups from the policy, score them
  //       with a synthetic verifiable reward, and check the mean reward RISES.
  // --------------------------------------------------------------------------
  {
    printf("[29] Policy gradient: gradient check + a closed GRPO loop\n");
    const int vocab = 16, dm = 8, nl = 2, nh = 4, nkv = 2, hd = 2, dff = 16, seq = 6;
    srand(23);
    LlamaModel pm = llama_model_create_train(vocab, dm, nl, nh, nkv, hd, dff, 64,
                                             10000.0f, 1e-5f);
    llama_model_randomize(&pm);
    LlamaActs pa = llama_acts_create(&pm, seq);
    int ps[2] = {seq, vocab};
    Tensor plog = tensor_create(ps, 2), pprob = tensor_create(ps, 2);

    int pids[6] = {2, 5, 1, 7, 3, 4};
    int ptgt[6] = {-1, -1, 1, 7, 3, -1};      // a masked prompt span, then a "response"
    float pcoef[6] = {0.0f, 0.0f, 0.31f, -0.17f, 0.44f, 0.0f};

    Tensor **pp = (Tensor**)malloc((size_t)llama_param_count(&pm) * sizeof(Tensor*));
    int n_pp = llama_collect_params(&pm, pp);

    llama_forward_train(&pm, pids, seq, &pa, &plog);
    softmax_rowwise(&plog, &pprob);
    tensor_zero_grad(&plog);
    pg_backward(&pprob, ptgt, pcoef, &plog);
    llama_backward(&pm, pids, seq, &pa, &plog);

    const float peps = 1e-2f;
    float pg_max = 0.0f, pg_cond = 0.0f;
    for (int p = 0; p < n_pp; p++) {
      Tensor *param = pp[p];
      size_t step = param->size > 2 ? param->size / 2 : 1;
      for (size_t k = 0; k < param->size; k += step) {
        float base = param->data[k];
        param->data[k] = base + peps;
        llama_forward_train(&pm, pids, seq, &pa, &plog);
        softmax_rowwise(&plog, &pprob);
        float jp = 0.0f;
        for (int i = 0; i < seq; i++)
          if (ptgt[i] >= 0) jp += pcoef[i] * -logprob_of(&pprob, i, ptgt[i]);
        param->data[k] = base - peps;
        llama_forward_train(&pm, pids, seq, &pa, &plog);
        softmax_rowwise(&plog, &pprob);
        float jm = 0.0f;
        for (int i = 0; i < seq; i++)
          if (ptgt[i] >= 0) jm += pcoef[i] * -logprob_of(&pprob, i, ptgt[i]);
        param->data[k] = base;
        float numeric = (jp - jm) / (2.0f * peps);
        float analytic = param->grad[k];
        float rel = fabsf(analytic - numeric) / (fabsf(analytic) + fabsf(numeric) + 1e-8f);
        if (rel > pg_max) pg_max = rel;
        if (fabsf(analytic) > 1e-2f && rel > pg_cond) pg_cond = rel;
      }
    }
    int pgrad_ok = (pg_cond < 1e-2f);
    printf("    gradient check (%d param tensors): max rel error %.2e (all) | %.2e (|grad|>1e-2) -> %s\n",
           n_pp, pg_max, pg_cond, pgrad_ok ? "OK" : "FAIL");
    all_ok &= pgrad_ok;
    free(pp);
    tensor_free(&plog); tensor_free(&pprob);
    llama_acts_free(&pa); llama_model_free(&pm);

    // ---- (b) A closed GRPO loop on a toy verifiable reward ----
    // Task: from a fixed prompt, emit the token GOAL. The reward is the fraction
    // of sampled tokens that are GOAL — objective, computable, and nothing the
    // model can talk its way into. If the loop is wired correctly the mean group
    // reward must climb.
    {
      const int V2 = 12, GOAL = 7, RLEN = 4, GG = 6, STEPS = 40;
      const int pn = 2;
      srand(29);
      LlamaModel gm = llama_model_create_train(V2, 16, 2, 4, 2, 4, 32, 32, 10000.0f, 1e-5f);
      llama_model_randomize(&gm);
      LlamaActs ga = llama_acts_create(&gm, pn + RLEN);
      int gs[2] = {pn + RLEN, V2};
      Tensor glog = tensor_create(gs, 2), gprob = tensor_create(gs, 2);
      int d1[2] = {1, V2};
      Tensor dlog = tensor_create(d1, 2), dprob = tensor_create(d1, 2);

      Tensor **gp = (Tensor**)malloc((size_t)llama_param_count(&gm) * sizeof(Tensor*));
      int n_gp = llama_collect_params(&gm, gp);
      Adam gopt = adamw_create(gp, n_gp, 0.02f, 0.0f);

      GrpoConfig cfg = grpo_default_config();
      cfg.group = GG;
      cfg.clip_eps = 0.2f;
      cfg.kl_beta = 0.0f;
      cfg.temperature = 1.0f;
      cfg.top_k = 0;
      cfg.top_p = 1.0f;

      LlamaKV gkv = llama_kv_create(&gm);
      Rng grng = rng_create(7);
      int prompt_ids[2] = {1, 2};
      int rid[6][4];
      float rlp[6][4];
      float rrew[6], radv[6], rcoef[4];
      int rlens[6];
      float first_mean = 0.0f, last_mean = 0.0f;

      for (int s = 0; s < STEPS; s++) {
        // Sample the group with the KV-cache decode (the real rollout path).
        double gsum = 0.0;
        for (int i = 0; i < GG; i++) {
          gkv.len = 0;
          for (int j = 0; j < pn; j++) llama_decode_step(&gm, &gkv, prompt_ids[j], &dlog);
          int hits = 0;
          for (int t = 0; t < RLEN; t++) {
            softmax_rowwise(&dlog, &dprob);
            int tk = sample_token(&dlog, 0, cfg.temperature, cfg.top_k, cfg.top_p, &grng);
            rlp[i][t] = logprob_of(&dprob, 0, tk);
            rid[i][t] = tk;
            if (tk == GOAL) hits++;
            llama_decode_step(&gm, &gkv, tk, &dlog);
          }
          rlens[i] = RLEN;
          rrew[i] = (float)hits / (float)RLEN;
          gsum += rrew[i];
        }
        const float mean_r = (float)(gsum / GG);
        if (s == 0) first_mean = mean_r;
        if (s == STEPS - 1) last_mean = mean_r;

        if (!grpo_advantages(rrew, GG, &cfg, radv)) continue;   // no spread, no update

        llama_zero_grad(&gm);
        int seq2[6], tgt2[6];
        float coef2[6];
        for (int i = 0; i < GG; i++) {
          const int L = pn + RLEN;
          seq2[0] = prompt_ids[0]; seq2[1] = prompt_ids[1];
          for (int t = 0; t < RLEN; t++) seq2[pn + t] = rid[i][t];
          for (int j = 0; j < L; j++) { tgt2[j] = -1; coef2[j] = 0.0f; }
          for (int j = pn - 1; j <= L - 2; j++) tgt2[j] = seq2[j + 1];

          llama_forward_train(&gm, seq2, L, &ga, &glog);
          softmax_rowwise(&glog, &gprob);
          float lp_new[4];
          for (int t = 0; t < RLEN; t++)
            lp_new[t] = logprob_of(&gprob, pn - 1 + t, rid[i][t]);
          grpo_token_coeffs(radv[i], grpo_loss_norm(&cfg, GG, rlens[i], 0), rlens[i],
                            lp_new, rlp[i], NULL, &cfg, rcoef);
          for (int t = 0; t < RLEN; t++) coef2[pn - 1 + t] = rcoef[t];

          tensor_zero_grad(&glog);
          pg_backward(&gprob, tgt2, coef2, &glog);
          llama_backward_accum(&gm, seq2, L, &ga, &glog, NULL, NULL);
        }
        clip_grad_norm(gp, n_gp, 1.0f);
        adam_step(&gopt, gp, n_gp);
      }

      // Chance is 1/V2 ~ 0.083. The loop must get meaningfully above that.
      int rl_ok = (last_mean > first_mean) && (last_mean > 0.5f);
      printf("    closed loop (%d steps, G=%d, reward = fraction of tokens == %d):\n"
             "      mean reward %.3f -> %.3f (chance %.3f) -> %s\n\n",
             STEPS, GG, GOAL, (double)first_mean, (double)last_mean,
             1.0 / (double)V2, rl_ok ? "OK" : "FAIL");
      all_ok &= rl_ok;

      adam_free(&gopt); free(gp);
      llama_kv_free(&gkv);
      tensor_free(&glog); tensor_free(&gprob);
      tensor_free(&dlog); tensor_free(&dprob);
      llama_acts_free(&ga); llama_model_free(&gm);
    }
  }

  // --------------------------------------------------------------------------
  // Part 30: the optional GPU backend (gpu.h). Self-skipping — on a CPU-only
  // build, or a GPU build with no device, it reports that and leaves all_ok
  // alone, so the same binary is a valid test everywhere.
  //
  // The GPU must produce the SAME NUMBERS as the CPU, not merely produce them
  // faster. Shapes are deliberately RAGGED (not multiples of the 16-wide tile or
  // of the forward kernel's 64-column block) because a tiled GEMM's classic bug
  // is exactly the boundary tile — and it hides completely on tidy dimensions.
  // --------------------------------------------------------------------------
  {
    printf("[30] GPU backend: results must match the CPU\n");
    if (!gpu_compiled_in()) {
      printf("    skipped (CPU-only build; rebuild with GPU=1 to exercise this)\n\n");
    } else if (!gpu_available()) {
      printf("    skipped (no GPU device: %s)\n", gpu_device_name());
      printf("    the CPU fallback carried every matmul in this run -- that IS the\n"
             "    contract being checked here\n\n");
      // Reaching this line at all proves the fallback works: every part above
      // ran through matrix.c with the GPU backend compiled in and inactive.
    } else {
      printf("    device: %s\n", gpu_device_name());
      // Ragged on purpose: none of these is a multiple of the 16-wide tile, and
      // 320 is not a multiple of the forward kernel's 64-column (TS*WPT) block.
      const int SH[3][3] = { {40, 33, 47}, {130, 129, 131}, {96, 257, 320} };
      float worst_fwd = 0.0f, worst_da = 0.0f, worst_db = 0.0f, worst_dx = 0.0f;
      int acc_ok = 1;
      // Compared as max|diff| / max|value| over the whole tensor, the same metric
      // parts [24]/[25] use. A PER-ELEMENT relative error is the wrong tool here:
      // matrix.h documents that the CPU's dA reassociates its dot product to
      // vectorize, so CPU and GPU legitimately differ in the last bits, and an
      // entry that happens to land near zero then shows a huge relative error
      // while its absolute error is negligible.
      #define CMP(nhost, host, ref, worst) do {                       \
        float _md = 0.0f, _mv = 0.0f;                                 \
        for (size_t _i = 0; _i < (nhost); _i++) {                     \
          float _d = fabsf((host)[_i] - (ref)[_i]);                   \
          if (_d > _md) _md = _d;                                     \
          if (fabsf((ref)[_i]) > _mv) _mv = fabsf((ref)[_i]);          \
        }                                                             \
        float _r = _md / (_mv + 1e-9f);                               \
        if (_r > (worst)) (worst) = _r;                               \
      } while (0)

      for (int s = 0; s < 3; s++) {
        const int M = SH[s][0], K = SH[s][1], N = SH[s][2];
        int sa[2] = {M, K}, sb[2] = {K, N}, sc[2] = {M, N};
        Tensor A = tensor_create(sa, 2), B = tensor_create(sb, 2), C = tensor_create(sc, 2);
        for (size_t i = 0; i < A.size; i++) A.data[i] = 0.02f * (float)((i * 31) % 17) - 0.16f;
        for (size_t i = 0; i < B.size; i++) B.data[i] = 0.02f * (float)((i * 41) % 19) - 0.18f;
        for (size_t i = 0; i < C.size; i++) C.grad[i] = 0.01f * (float)((i * 23) % 13) - 0.06f;

        // --- forward ---
        gpu_set_enabled(0);
        matmul(&A, &B, &C);
        float *ref = (float*)malloc(C.size * sizeof(float));
        memcpy(ref, C.data, C.size * sizeof(float));
        gpu_set_enabled(1);
        memset(C.data, 0, C.size * sizeof(float));
        matmul(&A, &B, &C);
        CMP(C.size, C.data, ref, worst_fwd);

        // --- backward, INCLUDING that it accumulates rather than overwrites ---
        // Both grads are pre-filled with a nonzero pattern: a kernel that stores
        // instead of accumulating passes a zero-initialized test and silently
        // discards every previously accumulated gradient in real training.
        float *gA0 = (float*)malloc(A.size * sizeof(float));
        float *gB0 = (float*)malloc(B.size * sizeof(float));
        for (size_t i = 0; i < A.size; i++) gA0[i] = 0.03f * (float)((i * 7) % 11) - 0.15f;
        for (size_t i = 0; i < B.size; i++) gB0[i] = 0.03f * (float)((i * 5) % 9) - 0.12f;

        memcpy(A.grad, gA0, A.size * sizeof(float));
        memcpy(B.grad, gB0, B.size * sizeof(float));
        gpu_set_enabled(0);
        matmul_backward(&A, &B, &C);
        float *rA = (float*)malloc(A.size * sizeof(float));
        float *rB = (float*)malloc(B.size * sizeof(float));
        memcpy(rA, A.grad, A.size * sizeof(float));
        memcpy(rB, B.grad, B.size * sizeof(float));

        memcpy(A.grad, gA0, A.size * sizeof(float));
        memcpy(B.grad, gB0, B.size * sizeof(float));
        gpu_set_enabled(1);
        matmul_backward(&A, &B, &C);
        CMP(A.size, A.grad, rA, worst_da);
        CMP(B.size, B.grad, rB, worst_db);
        // If the kernel had overwritten instead of accumulated, the GPU result
        // would equal the CPU's only when gA0 was zero -- so also assert the
        // result actually moved away from the seed.
        int moved = 0;
        for (size_t i = 0; i < A.size && !moved; i++)
          if (fabsf(A.grad[i] - gA0[i]) > 1e-6f) moved = 1;
        if (!moved) acc_ok = 0;

        // --- dx-only (the frozen-B / LoRA path) ---
        memcpy(A.grad, gA0, A.size * sizeof(float));
        gpu_set_enabled(0);
        matmul_backward_dx(&A, &B, &C);
        memcpy(rA, A.grad, A.size * sizeof(float));
        memcpy(A.grad, gA0, A.size * sizeof(float));
        gpu_set_enabled(1);
        matmul_backward_dx(&A, &B, &C);
        CMP(A.size, A.grad, rA, worst_dx);

        free(ref); free(gA0); free(gB0); free(rA); free(rB);
        tensor_free(&A); tensor_free(&B); tensor_free(&C);
      }

      #undef CMP
      const float tol = 1e-4f;
      int g_ok = (worst_fwd < tol) && (worst_da < tol) && (worst_db < tol) &&
                 (worst_dx < tol) && acc_ok;
      printf("    3 ragged shapes | fwd %.1e | dA %.1e | dB %.1e | dx %.1e (tol %.0e) -> %s\n",
             worst_fwd, worst_da, worst_db, worst_dx, tol, g_ok ? "OK" : "FAIL");
      printf("    backward ACCUMULATES into pre-seeded grads (not overwrite) -> %s\n\n",
             acc_ok ? "OK" : "FAIL");
      all_ok &= g_ok;
      gpu_set_enabled(1);
    }
  }

  // --------------------------------------------------------------------------
  // Part 31: Mixture-of-Experts FFN (moe.h) -- the FFN of DeepSeek-V3, Qwen3-MoE,
  // Kimi, GLM. A router picks top-k of E SwiGLU experts per token. Four gates:
  //   (a) numerical gradient check of x and EVERY parameter (router, all
  //       experts, shared expert) through the real softmax + top-k forward.
  //       Routing is not frozen; instead each probe's selection is compared to
  //       the base selection and the gate REQUIRES zero flips (a flip would make
  //       the finite difference measure a jump). Three configs: k=2 renormalized
  //       gates + shared expert; k=1 raw-probability gates (Switch); aux-only.
  //   (b) the aux loss has its defining values: 1 for a uniform router, ~E for
  //       a router collapsed onto one expert.
  //   (c) compute scales with k, not E: measured expert FLOPs per token are
  //       6*k*d*d_ff for every E, and unchosen experts are never run -- filled
  //       with NaN, they leave the output bit-identical and the backward finite.
  //   (d) the aux gradient does its job: Adam on the aux loss alone takes a
  //       router that sends most tokens to one expert and balances it.
  // --------------------------------------------------------------------------
  {
    printf("[31] Mixture-of-Experts FFN: router + top-k SwiGLU experts + aux loss\n");

    // (a) gradient checks.
    int moe_ok = 1;
    {
      struct { int E, k, renorm, shared, task; float aux; const char *name; } cf[3] = {
        {4, 2, 1, 1, 1, 0.1f, "k=2 renorm + shared, task+aux"},
        {4, 1, 0, 0, 1, 0.1f, "k=1 raw gate (Switch), task+aux"},
        {4, 2, 1, 0, 0, 1.0f, "k=2, aux loss ONLY"},
      };
      for (int i = 0; i < 3; i++) {
        int flips, checked; float margin;
        float rel = moe_gradcheck(cf[i].E, cf[i].k, cf[i].renorm, cf[i].shared,
                                  cf[i].task, cf[i].aux, &flips, &margin, &checked);
        int ok = rel < 1e-2f && flips == 0 && checked > 0;
        printf("    grad check %-31s: max rel %.2e over %3d entries | top-k margin %.3f,"
               " routing flips %d -> %s\n", cf[i].name, rel, checked, margin, flips,
               ok ? "OK" : "FAIL");
        moe_ok &= ok;
      }
    }

    // (b) aux-loss values at the two ends.
    {
      const int n = 16, dm = 4, E = 4;
      int xs[2] = {n, dm};
      Tensor x = tensor_create(xs, 2), y = tensor_create(xs, 2);
      moe_fill(&x, 31, 5, 1.0f);
      MoeLayer m = moe_create(dm, 6, E, 1, 0, 0);
      MoeCache c = moe_cache_create(&m, n);
      memset(m.W_router.data, 0, m.W_router.size * sizeof(float));
      float aux_uniform = moe_forward(&m, &x, &c, &y);
      for (int t = 0; t < n; t++) x.data[t * dm] = 1.0f;          // a constant feature...
      m.W_router.data[0 * E + 0] = 20.0f;                          // ...that says "expert 0"
      float aux_collapsed = moe_forward(&m, &x, &c, &y);
      int ok = fabsf(aux_uniform - 1.0f) < 1e-5f && aux_collapsed > 0.95f * E;
      printf("    aux loss: uniform router %.4f (expect 1) | collapsed router %.3f"
             " (expect ~E=%d) -> %s\n", aux_uniform, aux_collapsed, E, ok ? "OK" : "FAIL");
      moe_ok &= ok;
      moe_cache_free(&c); moe_free(&m); tensor_free(&x); tensor_free(&y);
    }

    // (c) compute scales with k, not E.
    {
      const int n = 16, dm = 8, df = 16;
      int xs[2] = {n, dm};
      Tensor x = tensor_create(xs, 2), y = tensor_create(xs, 2);
      moe_fill(&x, 37, 1, 1.0f);
      int scale_ok = 1;
      printf("    expert FLOPs/token, measured:  E=4    E=16   (expect 6*k*d*d_ff)\n");
      for (int k = 1; k <= 4; k *= 2) {
        double per_tok[2]; long long rows[2]; int Es[2] = {4, 16};
        for (int ei = 0; ei < 2; ei++) {
          MoeLayer m = moe_create(dm, df, Es[ei], k, 1, 0);
          MoeCache c = moe_cache_create(&m, n);
          moe_forward(&m, &x, &c, &y);
          per_tok[ei] = c.flops_experts / n; rows[ei] = c.expert_rows;
          moe_cache_free(&c); moe_free(&m);
        }
        double want = 6.0 * k * dm * df;
        int ok = per_tok[0] == want && per_tok[1] == want &&
                 rows[0] == (long long)n * k && rows[1] == (long long)n * k;
        printf("      k=%d                          %-6.0f %-6.0f (%.0f) -> %s\n",
               k, per_tok[0], per_tok[1], want, ok ? "OK" : "FAIL");
        scale_ok &= ok;
      }
      moe_ok &= scale_ok;

      // NaN poisoning: route 3 tokens through E=8, k=2 (at most 6 of 8 experts
      // used), then fill every unused expert with NaN. If the implementation ran
      // an unchosen expert at all -- even to multiply it by a zero gate -- NaN*0
      // is NaN and the output would change.
      const int n2 = 3, E2 = 8, k2 = 2;
      int xs2[2] = {n2, dm};
      Tensor x2 = tensor_create(xs2, 2), y2 = tensor_create(xs2, 2);
      moe_fill(&x2, 41, 9, 1.0f);
      MoeLayer m = moe_create(dm, df, E2, k2, 1, 1);
      MoeCache c = moe_cache_create(&m, n2);
      moe_forward(&m, &x2, &c, &y2);
      float ref[3 * 8];
      memcpy(ref, y2.data, sizeof ref);
      int used[8] = {0}, n_unused = 0;
      for (int a = 0; a < n2 * k2; a++) used[c.sel[a]] = 1;
      for (int e = 0; e < E2; e++) if (!used[e]) {
        n_unused++;
        for (size_t i = 0; i < m.W_gate[e].size; i++) m.W_gate[e].data[i] = NAN;
        for (size_t i = 0; i < m.W_up[e].size; i++)   m.W_up[e].data[i] = NAN;
        for (size_t i = 0; i < m.W_down[e].size; i++) m.W_down[e].data[i] = NAN;
      }
      moe_forward(&m, &x2, &c, &y2);
      int same = memcmp(ref, y2.data, sizeof ref) == 0;
      tensor_zero_grad(&x2);
      for (size_t i = 0; i < y2.size; i++) y2.grad[i] = 1.0f;
      moe_backward(&m, &x2, &c, &y2, 0.01f);
      int finite = 1;
      for (size_t i = 0; i < x2.size; i++) if (!isfinite(x2.grad[i])) finite = 0;
      int ok = same && finite && n_unused > 0;
      printf("    %d of %d experts unchosen, filled with NaN: output bit-identical %s,"
             " backward finite %s -> %s\n", n_unused, E2, same ? "yes" : "NO",
             finite ? "yes" : "NO", ok ? "OK" : "FAIL");
      moe_ok &= ok;
      moe_cache_free(&c); moe_free(&m);
      tensor_free(&x); tensor_free(&y); tensor_free(&x2); tensor_free(&y2);
    }

    // (d) the aux gradient balances a lopsided router (router trained alone).
    {
      const int n = 32, dm = 8, E = 4;
      int xs[2] = {n, dm};
      Tensor x = tensor_create(xs, 2), y = tensor_create(xs, 2);
      moe_fill(&x, 43, 2, 1.0f);
      for (int t = 0; t < n; t++) x.data[t * dm] = 1.0f;   // shared feature = bias
      MoeLayer m = moe_create(dm, 8, E, 1, 0, 0);
      moe_fill(&m.W_router, 47, 4, 0.5f);
      m.W_router.data[0 * E + 0] = 3.0f;                    // most tokens -> expert 0
      MoeCache c = moe_cache_create(&m, n);
      Tensor *rp[1] = {&m.W_router};
      Adam opt = adam_create(rp, 1, 0.05f);
      memset(y.grad, 0, y.size * sizeof(float));            // no task loss
      float aux0 = moe_forward(&m, &x, &c, &y);
      int max0 = 0;
      for (int e = 0; e < E; e++) if (c.count[e] > max0) max0 = c.count[e];
      for (int step = 0; step < 150; step++) {
        moe_forward(&m, &x, &c, &y);
        tensor_zero_grad(&m.W_router); tensor_zero_grad(&x);
        moe_backward(&m, &x, &c, &y, 1.0f);
        adam_step(&opt, rp, 1);
      }
      float auxN = moe_forward(&m, &x, &c, &y);
      int maxN = 0;
      for (int e = 0; e < E; e++) if (c.count[e] > maxN) maxN = c.count[e];
      int ok = auxN < aux0 && maxN < max0 && maxN <= n / 2;
      printf("    Adam on aux only (150 steps): aux %.3f -> %.3f, busiest expert %d/%d"
             " -> %d/%d tokens -> %s\n\n", aux0, auxN, max0, n, maxN, n, ok ? "OK" : "FAIL");
      moe_ok &= ok;
      adam_free(&opt); moe_cache_free(&c); moe_free(&m);
      tensor_free(&x); tensor_free(&y);
    }
    all_ok &= moe_ok;
  }

  // --------------------------------------------------------------------------
  // Part 32: SLIDING-WINDOW attention (gqa.h window, per layer via
  // llama_model_set_window). Position i attends only to [i-W+1, i]. Gates:
  //   (a) numerical gradient check of GQA WITH a window that actually cuts in
  //       (the masked keys must get exactly no gradient, the rest the right one),
  //   (b) W >= seq_len is full causal attention BIT-EXACTLY — at the GQA level
  //       and through a whole model — so a "large enough" window is free,
  //   (c) a token outside the window has ZERO influence: perturbing an
  //       out-of-window K/V row leaves that output row bit-identical, and in a
  //       stacked all-windowed model the receptive field is exactly L*(W-1)
  //       positions back (perturbing token 0 changes positions <= L*(W-1) and
  //       leaves every later position bit-identical),
  //   (d) KV-cache decode == full forward, BIT-EXACT, with the window on and a
  //       global layer mixed in, running long enough that the per-layer ring
  //       buffer (cap = W slots) wraps several times; plus the int8 decode path,
  //       the training forward, and a v3 checkpoint round-trip.
  // --------------------------------------------------------------------------
  {
    printf("[32] Sliding-window attention (per-layer window, local/global mix)\n");

    // (a) + (b) + (c, GQA level)
    {
      const int n = 6, H = 4, KV = 2, hd = 3, W = 2;
      int qs[2] = {n, H * hd}, ks[2] = {n, KV * hd};
      Tensor Q = tensor_create(qs, 2), O = tensor_create(qs, 2), Of = tensor_create(qs, 2);
      Tensor K = tensor_create(ks, 2), V = tensor_create(ks, 2);
      GqaCache c = gqa_cache_create(n, H, KV, hd);
      for (int i = 0; i < n * H * hd; i++) Q.data[i] = ((i * 13 % 100) / 50.0f) - 1.0f;
      for (int i = 0; i < n * KV * hd; i++) {
        K.data[i] = ((i * 17 % 100) / 50.0f) - 1.0f;
        V.data[i] = ((i * 23 % 100) / 50.0f) - 1.0f;
      }
      float w[72];
      for (int i = 0; i < n * H * hd; i++) w[i] = ((i * 31 % 100) / 50.0f) - 1.0f;

      // (a) analytic grads (L = sum(w * O)) vs central differences, window W.
      c.window = W;
      gqa_forward(&Q, &K, &V, &c, &O);
      tensor_zero_grad(&Q); tensor_zero_grad(&K); tensor_zero_grad(&V);
      for (int i = 0; i < n * H * hd; i++) O.grad[i] = w[i];
      gqa_backward(&Q, &K, &V, &c, &O);
      const float h = 1e-3f;
      float max_cond = 0.0f;
      Tensor *ps[3] = {&Q, &K, &V};
      for (int pi = 0; pi < 3; pi++) {
        Tensor *T = ps[pi];
        int sz = T->shape[0] * T->shape[1];
        for (int k = 0; k < sz; k++) {
          float s = T->data[k];
          T->data[k] = s + h; gqa_forward(&Q, &K, &V, &c, &O);
          float lp = 0.0f; for (int i = 0; i < n * H * hd; i++) lp += w[i] * O.data[i];
          T->data[k] = s - h; gqa_forward(&Q, &K, &V, &c, &O);
          float lm = 0.0f; for (int i = 0; i < n * H * hd; i++) lm += w[i] * O.data[i];
          T->data[k] = s;
          float num = (lp - lm) / (2.0f * h), rel = fabsf(num - T->grad[k]) / (fabsf(T->grad[k]) + 1e-3f);
          if (fabsf(T->grad[k]) > 1e-2f && rel > max_cond) max_cond = rel;
        }
      }
      // The window must actually change the result (else (a) proves nothing new).
      gqa_forward(&Q, &K, &V, &c, &O);
      c.window = 0;
      gqa_forward(&Q, &K, &V, &c, &Of);
      float cut = 0.0f;
      for (int i = 0; i < n * H * hd; i++) { float d = fabsf(O.data[i] - Of.data[i]); if (d > cut) cut = d; }
      int grad_ok = (max_cond < 1e-2f) && (cut > 1e-4f);
      printf("    (a) gradient check W=%d, n=%d (Q, K, V): max rel error %.2e | window changes output by %.2e -> %s\n",
             W, n, max_cond, cut, grad_ok ? "OK" : "FAIL");
      all_ok &= grad_ok;

      // (b) W >= n must equal full causal attention exactly (W = n and W = n+5).
      float big = 0.0f;
      const int bigw[2] = {n, n + 5};
      for (int t = 0; t < 2; t++) {
        c.window = bigw[t];
        gqa_forward(&Q, &K, &V, &c, &O);
        for (int i = 0; i < n * H * hd; i++) { float d = fabsf(O.data[i] - Of.data[i]); if (d > big) big = d; }
      }
      int big_ok = (big == 0.0f);
      printf("    (b) GQA: W=%d and W=%d vs full causal: max |diff| = %.2e (must be exactly 0) -> %s\n",
             n, n + 5, big, big_ok ? "OK" : "FAIL");
      all_ok &= big_ok;

      // (c, GQA level) perturb K/V of position 1: with W=2, rows i >= 3 cannot
      // see it (bit-identical), rows 1 and 2 can (must change).
      c.window = W;
      gqa_forward(&Q, &K, &V, &c, &Of);
      for (int d = 0; d < KV * hd; d++) { K.data[1 * KV * hd + d] += 0.7f; V.data[1 * KV * hd + d] -= 0.9f; }
      gqa_forward(&Q, &K, &V, &c, &O);
      float out_win = 0.0f, in_win = 0.0f;
      for (int i = 0; i < n; i++)
        for (int j = 0; j < H * hd; j++) {
          float d = fabsf(O.data[i * H * hd + j] - Of.data[i * H * hd + j]);
          if (i >= 1 + W) { if (d > out_win) out_win = d; }
          else if (i >= 1 && d > in_win) in_win = d;
        }
      int iso_ok = (out_win == 0.0f) && (in_win > 1e-4f);
      printf("    (c) GQA: perturb K/V at pos 1 (W=%d): rows >= %d max|diff| = %.2e (==0), rows 1..%d |diff| = %.2e (>0) -> %s\n",
             W, 1 + W, out_win, W, in_win, iso_ok ? "OK" : "FAIL");
      all_ok &= iso_ok;
      gqa_cache_free(&c);
      tensor_free(&Q); tensor_free(&K); tensor_free(&V); tensor_free(&O); tensor_free(&Of);
    }

    const int vocab = 32, dm = 8, nh = 4, nkv = 2, hd = 2, dff = 16;

    // (b, model level) + (c, model level): receptive field of a stacked window.
    {
      const int nl = 2, W = 2, seq = 8;
      srand(31);
      LlamaModel lm = llama_model_create(vocab, dm, nl, nh, nkv, hd, dff, 64, 10000.0f, 1e-5f);
      llama_model_randomize(&lm);
      int ids[8]  = {3, 7, 1, 9, 2, 5, 11, 4};
      int ids2[8] = {20, 7, 1, 9, 2, 5, 11, 4};   // differs only in token 0
      int ls[2] = {seq, vocab};
      Tensor Lfull = tensor_create(ls, 2), L1 = tensor_create(ls, 2), L2 = tensor_create(ls, 2);

      llama_model_forward(&lm, ids, seq, &Lfull);
      llama_model_set_window(&lm, seq, 2);          // W = seq_len on the local layers
      llama_model_forward(&lm, ids, seq, &L1);
      float big = 0.0f;
      for (int i = 0; i < seq * vocab; i++) { float d = fabsf(L1.data[i] - Lfull.data[i]); if (d > big) big = d; }
      int big_ok = (big == 0.0f);
      printf("    (b) model: window = seq_len (%d) vs no window: max |logit diff| = %.2e (must be exactly 0) -> %s\n",
             seq, big, big_ok ? "OK" : "FAIL");
      all_ok &= big_ok;

      llama_model_set_window(&lm, W, 0);            // every layer windowed (Mistral-style)
      llama_model_forward(&lm, ids, seq, &L1);
      llama_model_forward(&lm, ids2, seq, &L2);
      const int reach = nl * (W - 1);               // last position token 0 can reach
      float far = 0.0f, near = 1e30f;
      for (int i = 0; i < seq; i++) {
        float mx = 0.0f;
        for (int j = 0; j < vocab; j++) { float d = fabsf(L1.data[i * vocab + j] - L2.data[i * vocab + j]); if (d > mx) mx = d; }
        if (i > reach) { if (mx > far) far = mx; }
        else if (mx < near) near = mx;
      }
      int rf_ok = (far == 0.0f) && (near > 1e-6f);
      printf("    (c) model: %d layers all W=%d, perturb token 0: positions > %d max|diff| = %.2e (==0), positions 0..%d min|diff| = %.2e (>0) -> %s\n",
             nl, W, reach, far, reach, near, rf_ok ? "OK" : "FAIL");
      all_ok &= rf_ok;

      // With one GLOBAL layer mixed in, token 0 reaches every position again.
      llama_model_set_window(&lm, W, 2);            // layer 1 global
      llama_model_forward(&lm, ids, seq, &L1);
      llama_model_forward(&lm, ids2, seq, &L2);
      float last = 0.0f;
      for (int j = 0; j < vocab; j++) { float d = fabsf(L1.data[(seq - 1) * vocab + j] - L2.data[(seq - 1) * vocab + j]); if (d > last) last = d; }
      int glob_ok = (last > 1e-6f);
      printf("        with layer 1 global (--global-every 2): last position |diff| = %.2e (>0) -> %s\n",
             last, glob_ok ? "OK" : "FAIL");
      all_ok &= glob_ok;
      tensor_free(&Lfull); tensor_free(&L1); tensor_free(&L2);
      llama_model_free(&lm);
    }

    // (d) decode == forward, bit-exact, ring buffer wrapping; int8 path;
    // training forward; checkpoint round-trip.
    {
      const int nl = 3, W = 3, seq = 11;            // layers 0,1 local (W=3), layer 2 global
      srand(37);
      LlamaModel lm = llama_model_create_train(vocab, dm, nl, nh, nkv, hd, dff, 32, 10000.0f, 1e-5f);
      llama_model_randomize(&lm);
      llama_model_set_window(&lm, W, 3);
      int ids[11] = {3, 7, 1, 9, 2, 5, 11, 4, 30, 17, 6};
      int fs[2] = {seq, vocab}, ds[2] = {1, vocab};
      Tensor Lf = tensor_create(fs, 2), Ld = tensor_create(ds, 2), Lt = tensor_create(fs, 2);
      llama_model_forward(&lm, ids, seq, &Lf);

      LlamaKV kv = llama_kv_create(&lm);
      float maxd = 0.0f;
      for (int i = 0; i < seq; i++) {
        llama_decode_step(&lm, &kv, ids[i], &Ld);
        for (int j = 0; j < vocab; j++) { float d = fabsf(Ld.data[j] - Lf.data[i * vocab + j]); if (d > maxd) maxd = d; }
      }
      int dec_ok = (maxd == 0.0f) && kv.cap[0] == W && kv.cap[1] == W && kv.cap[2] == kv.max_seq;
      printf("    (d) KV decode (%d positions, ring cap %d/%d/%d slots of max_seq %d) vs full forward: max |logit diff| = %.2e -> %s\n",
             seq, kv.cap[0], kv.cap[1], kv.cap[2], kv.max_seq, maxd, dec_ok ? "OK" : "FAIL");
      all_ok &= dec_ok;

      // Reusing the cache for a new sequence (len = 0) must still be exact.
      kv.len = 0;
      float again = 0.0f;
      for (int i = 0; i < seq; i++) {
        llama_decode_step(&lm, &kv, ids[i], &Ld);
        for (int j = 0; j < vocab; j++) { float d = fabsf(Ld.data[j] - Lf.data[i * vocab + j]); if (d > again) again = d; }
      }
      int again_ok = (again == 0.0f);
      printf("        cache reused (len reset to 0): max |diff| = %.2e -> %s\n", again, again_ok ? "OK" : "FAIL");
      all_ok &= again_ok;

      // int8 decode honours the window too. It carries quantization error, so
      // the gate is relative: it must track the WINDOWED fp32 forward (within
      // 2% — three layers of int8 at d_model 8 sit near 1%, cf. [21]) and sit
      // clearly closer to it than to the full-attention forward.
      Tensor Lnw = tensor_create(fs, 2);
      llama_model_set_window(&lm, 0, 0);
      llama_model_forward(&lm, ids, seq, &Lnw);
      llama_model_set_window(&lm, W, 3);
      QLlamaModel qm = q_llama_quantize_from(&lm);
      LlamaKV qkv = q_llama_kv_create(&qm);
      double num = 0.0, den = 0.0, num_nw = 0.0;
      for (int i = 0; i < seq; i++) {
        q_llama_decode_step(&qm, &qkv, ids[i], &Ld);
        for (int j = 0; j < vocab; j++) {
          double d = (double)Ld.data[j] - Lf.data[i * vocab + j];
          double e = (double)Ld.data[j] - Lnw.data[i * vocab + j];
          num += d * d; num_nw += e * e;
          den += (double)Lf.data[i * vocab + j] * Lf.data[i * vocab + j];
        }
      }
      double qrel = sqrt(num / (den + 1e-30)), qrel_nw = sqrt(num_nw / (den + 1e-30));
      int q_ok = (qrel < 2e-2) && (qrel < 0.5 * qrel_nw) && qkv.cap[0] == W;
      printf("        int8 decode (windowed): L2 rel error %.2e vs windowed fp32 (<2e-2), %.2e vs full-attention fp32 -> %s\n",
             qrel, qrel_nw, q_ok ? "OK" : "FAIL");
      all_ok &= q_ok;
      tensor_free(&Lnw);

      // The training forward must see the same window (else the backward would
      // differentiate a different function than inference runs).
      LlamaActs acts = llama_acts_create(&lm, seq);
      llama_forward_train(&lm, ids, seq, &acts, &Lt);
      float tr = 0.0f;
      for (int i = 0; i < seq * vocab; i++) { float d = fabsf(Lt.data[i] - Lf.data[i]); if (d > tr) tr = d; }
      int tr_ok = (tr == 0.0f);
      printf("        train forward == inference forward (windowed): max |diff| = %.2e -> %s\n",
             tr, tr_ok ? "OK" : "FAIL");
      all_ok &= tr_ok;

      // Checkpoint: a windowed model saves as v3 and round-trips exactly; an
      // unwindowed one still saves as v2 (readable by pre-window builds).
      int rt_ok = 0;
      uint32_t ver_w = 0, ver_0 = 0;
      if (llama_model_save("swa_ckpt.tmp", &lm) == 0) {
        FILE *f = fopen("swa_ckpt.tmp", "rb");
        if (f) { uint32_t hdr[2]; if (fread(hdr, sizeof(uint32_t), 2, f) == 2) ver_w = hdr[1]; fclose(f); }
        LlamaModel lm2;
        if (llama_model_load("swa_ckpt.tmp", &lm2) == 0) {
          Tensor L3 = tensor_create(fs, 2);
          llama_model_forward(&lm2, ids, seq, &L3);
          float md = 0.0f;
          for (int i = 0; i < seq * vocab; i++) { float d = fabsf(L3.data[i] - Lf.data[i]); if (d > md) md = d; }
          int same_w = 1;
          for (int l = 0; l < nl; l++) if (lm2.blocks[l].window != lm.blocks[l].window) same_w = 0;
          rt_ok = (md == 0.0f) && same_w && ver_w == 3;
          tensor_free(&L3);
          llama_model_free(&lm2);
        }
      }
      llama_model_set_window(&lm, 0, 0);            // window off -> must write v2
      if (llama_model_save("swa_ckpt.tmp", &lm) == 0) {
        FILE *f = fopen("swa_ckpt.tmp", "rb");
        if (f) { uint32_t hdr[2]; if (fread(hdr, sizeof(uint32_t), 2, f) == 2) ver_0 = hdr[1]; fclose(f); }
      }
      remove("swa_ckpt.tmp");
      rt_ok = rt_ok && ver_0 == 2;
      printf("        checkpoint: windowed -> v%u, round-trip windows + logits exact; no window -> v%u -> %s\n\n",
             ver_w, ver_0, rt_ok ? "OK" : "FAIL");
      all_ok &= rt_ok;

      llama_acts_free(&acts);
      llama_kv_free(&qkv); q_llama_model_free(&qm);
      llama_kv_free(&kv);
      tensor_free(&Lf); tensor_free(&Ld); tensor_free(&Lt);
      llama_model_free(&lm);
    }
  }

  // --------------------------------------------------------------------------
  // Part 33: QK-norm (Qwen3) — a per-head RMSNorm on q and k, with one learned
  // head_dim gain each shared across heads, applied after the projection (+bias,
  // +LoRA) and BEFORE RoPE. It is an optional per-model feature, so this part
  // turns it on and re-runs every gate the plain path has, on the new path:
  //   (a) it is live: the same weights without it give different logits,
  //   (b) training forward == inference forward, bit for bit,
  //   (c) end-to-end gradient check over every parameter (q_norm/k_norm at
  //       every entry), and over LoRA adapters that now feed through the norm,
  //   (d) KV-cache decode == full forward, bit for bit (plain and with LoRA),
  //   (e) LLMA v4 checkpoint round-trip (flag + gains survive; a model without
  //       any feature is still v2, windows-only still v3), and the int8 path,
  //   (f) QK-norm COMBINED with sliding windows ([32]): v4 with both flag bits
  //       -> reload -> logits bit-exact, and decode == forward with the rings
  //       wrapping.
  // --------------------------------------------------------------------------
  {
    const int vocab = 24, dm = 8, nl = 2, nh = 4, nkv = 2, hd = 4, dff = 16, seq = 5;
    printf("[33] QK-norm (Qwen3): per-head q/k RMSNorm before RoPE\n");
    srand(41);
    LlamaModel qm = llama_model_create_train(vocab, dm, nl, nh, nkv, hd, dff, 64,
                                             10000.0f, 1e-5f);
    llama_model_set_qk_norm(&qm, 1);          // BEFORE randomize: gains get drawn too
    llama_model_randomize(&qm);
    // The gains only steer the attention PATTERN; at randomize's small scale the
    // attended values barely differ and the head is tiny, so the gains' gradients
    // sit below the 1e-2 judging floor and (c) would be vacuous. Amplify the
    // value path (Wv, Wo) and the output head.
    for (int l = 0; l < nl; l++) {
      for (size_t i = 0; i < qm.blocks[l].Wv.size; i++) qm.blocks[l].Wv.data[i] *= 4.0f;
      for (size_t i = 0; i < qm.blocks[l].Wo.size; i++) qm.blocks[l].Wo.data[i] *= 4.0f;
    }
    for (size_t i = 0; i < qm.lm_head.size; i++) qm.lm_head.data[i] *= 8.0f;
    LlamaActs qa = llama_acts_create(&qm, seq);
    int ids[5]     = {2, 9, 4, 1, 7};
    int targets[5] = {9, 4, 1, 7, 3};
    int ls[2] = {seq, vocab};
    Tensor logits = tensor_create(ls, 2), probs = tensor_create(ls, 2);
    Tensor ref = tensor_create(ls, 2), off = tensor_create(ls, 2);

    // (a) Flip only the per-block switch (gains untouched) to get the same
    //     weights WITHOUT QK-norm; the logits must move.
    llama_model_forward(&qm, ids, seq, &ref);
    for (int l = 0; l < nl; l++) qm.blocks[l].qk_norm = 0;
    llama_model_forward(&qm, ids, seq, &off);
    for (int l = 0; l < nl; l++) qm.blocks[l].qk_norm = 1;
    float live = 0.0f;
    for (int i = 0; i < seq * vocab; i++) {
      float d = fabsf(ref.data[i] - off.data[i]);
      if (d > live) live = d;
    }
    int live_ok = (live > 1e-3f);
    printf("    QK-norm is live: max |logit diff| vs same weights without it = %.2e -> %s\n",
           live, live_ok ? "OK" : "FAIL");
    all_ok &= live_ok;

    // (b) training forward == inference forward.
    llama_forward_train(&qm, ids, seq, &qa, &logits);
    float fwd_diff = 0.0f;
    for (int i = 0; i < seq * vocab; i++) {
      float d = fabsf(logits.data[i] - ref.data[i]);
      if (d > fwd_diff) fwd_diff = d;
    }
    int fwd_ok = (fwd_diff == 0.0f);
    printf("    train forward == inference forward: max |diff| = %.2e -> %s\n",
           fwd_diff, fwd_ok ? "OK" : "FAIL");
    all_ok &= fwd_ok;

    // (c) gradient check: every parameter tensor, then the gains at every entry.
    const int n_qp = llama_param_count(&qm);
    Tensor **qp = (Tensor**)malloc((size_t)n_qp * sizeof(Tensor*));
    llama_collect_params(&qm, qp);
    Tensor **gains = (Tensor**)malloc((size_t)(2 * nl) * sizeof(Tensor*));
    for (int l = 0; l < nl; l++) {
      gains[2 * l] = &qm.blocks[l].q_norm;
      gains[2 * l + 1] = &qm.blocks[l].k_norm;
    }
    llama_ce_loss(&qm, &qa, ids, seq, targets, &logits, &probs);
    tensor_zero_grad(&logits);
    cross_entropy_backward(&probs, targets, &logits);
    llama_backward(&qm, ids, seq, &qa, &logits);
    float g_all, g_cond, n_all, n_cond;
    llama_grad_check(&qm, &qa, NULL, NULL, ids, seq, targets, &logits, &probs,
                     qp, n_qp, 0, &g_all, &g_cond);
    int n_judged = llama_grad_check(&qm, &qa, NULL, NULL, ids, seq, targets, &logits, &probs,
                                    gains, 2 * nl, (size_t)hd, &n_all, &n_cond);
    int qgrad_ok = (g_cond < 1e-2f) && (n_cond < 1e-2f) && (n_judged > 0);
    printf("    gradient check (%d param tensors = 3 + 14/layer): max rel error %.2e (all) | %.2e (|grad|>1e-2)\n",
           n_qp, g_all, g_cond);
    printf("    q_norm/k_norm gains, all %d entries: max rel error %.2e (all) | %.2e (%d with |grad|>1e-2) -> %s\n",
           2 * nl * hd, n_all, n_cond, n_judged, qgrad_ok ? "OK" : "FAIL");
    all_ok &= qgrad_ok;

    // LoRA through QK-norm: adapters land BEFORE the norm, so their gradient
    // must pass back through it. Nonzero B first (at B == 0 dA is identically 0).
    LlamaLora lo = llama_lora_create(&qm, 4, 8.0f);
    llama_lora_init(&lo);
    LlamaLoraActs la = llama_lora_acts_create(&qm, &lo, seq);
    const int n_lp = llama_lora_param_count(&lo);
    Tensor **lp = (Tensor**)malloc((size_t)n_lp * sizeof(Tensor*));
    llama_lora_collect_params(&lo, lp);
    for (int p = 0; p < n_lp; p++)
      for (size_t k = 0; k < lp[p]->size; k++)
        lp[p]->data[k] += 0.05f * (float)((int)((k * 29 + (size_t)p * 7) % 11) - 5);
    llama_forward_train_lora(&qm, ids, seq, &qa, &logits, &lo, &la);
    cross_entropy_forward(&logits, targets, &probs);
    tensor_zero_grad(&logits);
    cross_entropy_backward(&probs, targets, &logits);
    llama_backward_lora(&qm, ids, seq, &qa, &logits, &lo, &la);
    float l_all, l_cond;
    llama_grad_check(&qm, &qa, &lo, &la, ids, seq, targets, &logits, &probs,
                     lp, n_lp, 0, &l_all, &l_cond);
    int lgrad_ok = (l_cond < 1e-2f);
    printf("    LoRA gradient check through QK-norm (%d adapter tensors): %.2e (all) | %.2e (|grad|>1e-2) -> %s\n",
           n_lp, l_all, l_cond, lgrad_ok ? "OK" : "FAIL");
    all_ok &= lgrad_ok;

    // (d) KV-cache decode vs full forward, plain and with the adapters above.
    int ds[2] = {1, vocab};
    Tensor Ld = tensor_create(ds, 2);
    LlamaKV kv = llama_kv_create(&qm);
    float kv_diff = 0.0f;
    for (int i = 0; i < seq; i++) {
      llama_decode_step(&qm, &kv, ids[i], &Ld);
      for (int j = 0; j < vocab; j++) {
        float d = fabsf(Ld.data[j] - ref.data[i * vocab + j]);
        if (d > kv_diff) kv_diff = d;
      }
    }
    llama_forward_train_lora(&qm, ids, seq, &qa, &logits, &lo, &la);
    LlamaDecodeLora dv = llama_lora_decode_view(&lo);
    LlamaKV kv2 = llama_kv_create(&qm);
    float lkv_diff = 0.0f;
    for (int i = 0; i < seq; i++) {
      llama_decode_step_lora(&qm, &kv2, ids[i], &Ld, &dv);
      for (int j = 0; j < vocab; j++) {
        float d = fabsf(Ld.data[j] - logits.data[i * vocab + j]);
        if (d > lkv_diff) lkv_diff = d;
      }
    }
    int kv_ok = (kv_diff == 0.0f), lkv_ok = (lkv_diff < 1e-3f);
    printf("    KV-cache decode == full forward: max |logit diff| = %.2e -> %s\n",
           kv_diff, kv_ok ? "OK" : "FAIL");
    printf("    LoRA KV-cache decode == LoRA training forward: max |logit diff| = %.2e -> %s\n",
           lkv_diff, lkv_ok ? "OK" : "FAIL");
    all_ok &= kv_ok & lkv_ok;

    // (e) checkpoint: QK-norm model -> v4 with flags 0x1; reload must keep the
    //     flag and give bit-identical logits. A plain model must still be v2.
    int rt_ok = 0;
    uint32_t hq[3] = {0, 0, 0}, hp[2] = {0, 0};
    if (llama_model_save("qknorm_ckpt.tmp", &qm) == 0) {
      FILE *hf = fopen("qknorm_ckpt.tmp", "rb");
      if (hf) {
        if (fread(hq, sizeof(uint32_t), 2, hf) == 2 && fseek(hf, 8 + 32 + 8, SEEK_SET) == 0 &&
            fread(&hq[2], sizeof(uint32_t), 1, hf) == 1) { /* header read */ }
        fclose(hf);
      }
      LlamaModel qm2;
      if (llama_model_load("qknorm_ckpt.tmp", &qm2) == 0) {
        Tensor L2 = tensor_create(ls, 2);
        llama_model_forward(&qm2, ids, seq, &L2);
        float md = 0.0f;
        for (int i = 0; i < seq * vocab; i++) { float d = fabsf(L2.data[i] - ref.data[i]); if (d > md) md = d; }
        rt_ok = (md == 0.0f) && qm2.qk_norm && qm2.blocks[nl - 1].qk_norm &&
                hq[1] == 4u && hq[2] == 1u;
        printf("    LLMA v%u (flags 0x%x) round-trip: qk_norm=%d, max |logit diff| = %.2e",
               hq[1], hq[2], qm2.qk_norm, md);
        tensor_free(&L2);

        // int8: quantize the QK-norm model; its decode must track the f32 forward.
        QLlamaModel q8 = q_llama_quantize_from(&qm2);
        LlamaKV kv8 = q_llama_kv_create(&q8);
        float num = 0.0f, den = 0.0f;
        for (int i = 0; i < seq; i++) {
          q_llama_decode_step(&q8, &kv8, ids[i], &Ld);
          for (int j = 0; j < vocab; j++) {
            float d = Ld.data[j] - ref.data[i * vocab + j];
            num += d * d;
            den += ref.data[i * vocab + j] * ref.data[i * vocab + j];
          }
        }
        float rel8 = sqrtf(num / den);
        int q8_ok = (rel8 < 0.15f);          // same tiny-dims tolerance as [21]
        rt_ok &= q8_ok;
        printf(" | int8 decode L2 rel %.2e", rel8);
        llama_kv_free(&kv8); q_llama_model_free(&q8);
        llama_model_free(&qm2);
      }
    }
    remove("qknorm_ckpt.tmp");
    LlamaModel pm = llama_model_create(vocab, dm, nl, nh, nkv, hd, dff, 64, 10000.0f, 1e-5f);
    if (llama_model_save("plain_ckpt.tmp", &pm) == 0) {
      FILE *hf = fopen("plain_ckpt.tmp", "rb");
      if (hf) { if (fread(hp, sizeof(uint32_t), 2, hf) != 2) hp[1] = 0; fclose(hf); }
    }
    remove("plain_ckpt.tmp");
    llama_model_free(&pm);
    rt_ok &= (hp[1] == 2u);
    printf(" | plain model still v%u -> %s\n", hp[1], rt_ok ? "OK" : "FAIL");
    all_ok &= rt_ok;

    // (f) window + QK-norm together. 3 layers (local, local, global), W = 2 and
    //     7 positions, so the 2-slot rings wrap several times.
    {
      const int cl = 3, cseq = 7, W = 2;
      int cids[7] = {3, 7, 1, 9, 2, 5, 8};
      srand(43);
      LlamaModel cm = llama_model_create(vocab, dm, cl, nh, nkv, hd, dff, 64, 10000.0f, 1e-5f);
      llama_model_set_qk_norm(&cm, 1);
      llama_model_randomize(&cm);
      llama_model_set_window(&cm, W, 3);          // layer 2 global, layers 0-1 windowed
      int cs[2] = {cseq, vocab};
      Tensor Lc = tensor_create(cs, 2), Lc2 = tensor_create(cs, 2);
      llama_model_forward(&cm, cids, cseq, &Lc);

      // A windows-only model (same weights, QK-norm off) must still be v3.
      uint32_t hw[2] = {0, 0};
      for (int l = 0; l < cl; l++) cm.blocks[l].qk_norm = 0;
      cm.qk_norm = 0;
      if (llama_model_save("win_ckpt.tmp", &cm) == 0) {
        FILE *hf = fopen("win_ckpt.tmp", "rb");
        if (hf) { if (fread(hw, sizeof(uint32_t), 2, hf) != 2) hw[1] = 0; fclose(hf); }
      }
      remove("win_ckpt.tmp");
      for (int l = 0; l < cl; l++) cm.blocks[l].qk_norm = 1;   // gains untouched
      cm.qk_norm = 1;

      int c_ok = 0;
      uint32_t hc[2] = {0, 0}, cflags = 0;
      int32_t cw[3] = {-1, -1, -1};
      float c_md = -1.0f, c_kv = -1.0f;
      if (llama_model_save("combo_ckpt.tmp", &cm) == 0) {
        FILE *hf = fopen("combo_ckpt.tmp", "rb");
        if (hf) {
          if (fread(hc, sizeof(uint32_t), 2, hf) == 2 && fseek(hf, 8 + 32 + 8, SEEK_SET) == 0 &&
              fread(&cflags, sizeof(uint32_t), 1, hf) == 1 &&
              fread(cw, sizeof(int32_t), 3, hf) == 3) { /* header read */ }
          fclose(hf);
        }
        LlamaModel cm2;
        if (llama_model_load("combo_ckpt.tmp", &cm2) == 0) {
          llama_model_forward(&cm2, cids, cseq, &Lc2);
          c_md = 0.0f;
          for (int i = 0; i < cseq * vocab; i++) {
            float d = fabsf(Lc2.data[i] - Lc.data[i]);
            if (d > c_md) c_md = d;
          }
          LlamaKV ckv = llama_kv_create(&cm2);
          c_kv = 0.0f;
          for (int i = 0; i < cseq; i++) {
            llama_decode_step(&cm2, &ckv, cids[i], &Ld);
            for (int j = 0; j < vocab; j++) {
              float d = fabsf(Ld.data[j] - Lc.data[i * vocab + j]);
              if (d > c_kv) c_kv = d;
            }
          }
          int same_cfg = cm2.qk_norm;
          for (int l = 0; l < cl; l++)
            same_cfg &= (cm2.blocks[l].window == cm.blocks[l].window) && cm2.blocks[l].qk_norm;
          c_ok = same_cfg && (c_md == 0.0f) && (c_kv == 0.0f) && hc[1] == 4u &&
                 cflags == 3u && cw[0] == W && cw[1] == W && cw[2] == 0 && hw[1] == 3u;
          llama_kv_free(&ckv);
          llama_model_free(&cm2);
        }
      }
      remove("combo_ckpt.tmp");
      printf("    window+QK-norm: v%u flags 0x%x windows {%d,%d,%d} round-trip max |diff| %.2e | "
             "decode == forward (rings wrap) %.2e | windows-only still v%u -> %s\n\n",
             hc[1], cflags, cw[0], cw[1], cw[2], c_md, c_kv, hw[1], c_ok ? "OK" : "FAIL");
      all_ok &= c_ok;
      tensor_free(&Lc); tensor_free(&Lc2);
      llama_model_free(&cm);
    }

    free(qp); free(gains); free(lp);
    llama_kv_free(&kv); llama_kv_free(&kv2);
    llama_lora_acts_free(&la); llama_lora_free(&lo);
    tensor_free(&logits); tensor_free(&probs); tensor_free(&ref); tensor_free(&off);
    tensor_free(&Ld);
    llama_acts_free(&qa);
    llama_model_free(&qm);
  }

  // --------------------------------------------------------------------------
  // Part 34: the AVX2+FMA matmul kernels (matrix.c, roadmap B4) vs the portable
  // C ones. Self-skipping on a CPU/build without AVX2+FMA. Every part above ran
  // on whichever path the dispatch picked; this one runs BOTH on the same inputs.
  //
  // Shapes are awkward on purpose: 1 row, 1 column, K=1, sizes that are not
  // multiples of 8 or 16 (the masked tail strip), 1-3 leftover rows next to
  // 4-row blocks, N > 128 (more than one packed panel) and K, N or M > 256 (more
  // than one p block: the forward's resumed chain, dA's and dB's added blocks).
  // Four gates:
  //  (a) SIMD vs portable within tol, as max|diff| / max|ref| over the tensor --
  //      they legitimately differ in the last bits (FMA vs mul+add depending on
  //      how the portable code was compiled, and a different dA/dB association),
  //      so this is a tolerance, not equality. Both are also measured against a
  //      double-precision reference, so "SIMD differs" can't hide "SIMD is worse".
  //  (b) backward/dx ACCUMULATE into pre-seeded grads (the seed is in the
  //      reference, so an overwriting kernel fails (a) outright);
  //  (c) SIMD row i of an M-row matmul == the same row computed as a 1-row
  //      matmul, BIT-EXACT: M=1 decode and the M=T forward take different code
  //      paths (AXPY vs packed tiles), and parts [4]/[19]/[26] need them equal;
  //  (d) the int8 column accumulate (qmatmul's core) is integer arithmetic, so
  //      SIMD and portable must agree EXACTLY.
  // --------------------------------------------------------------------------
  {
    printf("[34] SIMD matmul kernels (AVX2+FMA) vs portable C\n");
    if (!matmul_simd_available()) {
      printf("    skipped (%s: no AVX2+FMA path on this CPU/build)\n\n", matmul_simd_name());
    } else {
      const int was_on = matmul_simd_enabled();
      const int gpu_was = gpu_is_enabled();
      gpu_set_enabled(0);                         // compare CPU kernels, nothing else
      const int SH[][3] = {                       // M, K, N
        {1, 1, 1}, {1, 7, 9}, {2, 3, 1}, {3, 17, 5}, {4, 16, 16}, {5, 13, 31},
        {9, 33, 47}, {7, 1, 300}, {130, 5, 3}, {1, 300, 70}, {6, 260, 129},
        {5, 20, 300}, {300, 6, 17}, {21, 67, 145},
      };
      const int nsh = (int)(sizeof(SH) / sizeof(SH[0]));
      double w_fwd = 0, w_da = 0, w_db = 0, w_dx = 0;        // SIMD vs portable
      double ref_simd = 0, ref_port = 0;                     // each vs double ref
      int rows_exact = 1;
      unsigned lcg = 12345u;
      #define RNDF() ((float)((lcg = lcg * 1103515245u + 12345u) >> 9 & 0xFFFF) / 32768.0f - 1.0f)
      // max|x - ref| / max|ref|, ref in double
      #define RELD(n, x, r, out) do {                                    \
        double _md = 0, _mv = 0;                                         \
        for (size_t _i = 0; _i < (size_t)(n); _i++) {                    \
          double _d = fabs((double)(x)[_i] - (double)(r)[_i]);           \
          if (_d > _md) _md = _d;                                        \
          if (fabs((double)(r)[_i]) > _mv) _mv = fabs((double)(r)[_i]);  \
        }                                                                \
        (out) = _md / (_mv + 1e-30);                                     \
      } while (0)
      for (int s = 0; s < nsh; s++) {
        const int M = SH[s][0], K = SH[s][1], N = SH[s][2];
        int sa[2] = {M, K}, sb[2] = {K, N}, sc[2] = {M, N};
        Tensor A = tensor_create(sa, 2), B = tensor_create(sb, 2), C = tensor_create(sc, 2);
        for (size_t i = 0; i < A.size; i++) A.data[i] = RNDF();
        for (size_t i = 0; i < B.size; i++) B.data[i] = RNDF();
        for (size_t i = 0; i < C.size; i++) C.grad[i] = RNDF();
        float *gA0 = (float*)malloc(A.size * sizeof(float));
        float *gB0 = (float*)malloc(B.size * sizeof(float));
        for (size_t i = 0; i < A.size; i++) gA0[i] = RNDF();
        for (size_t i = 0; i < B.size; i++) gB0[i] = RNDF();

        // double-precision references (grads include their seeds)
        double *rC = (double*)malloc(C.size * sizeof(double));
        double *rA = (double*)malloc(A.size * sizeof(double));
        double *rB = (double*)malloc(B.size * sizeof(double));
        for (int i = 0; i < M; i++)
          for (int j = 0; j < N; j++) {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)A.data[i * K + k] * B.data[k * N + j];
            rC[i * N + j] = acc;
          }
        for (int i = 0; i < M; i++)
          for (int k = 0; k < K; k++) {
            double acc = gA0[i * K + k];
            for (int j = 0; j < N; j++) acc += (double)C.grad[i * N + j] * B.data[k * N + j];
            rA[i * K + k] = acc;
          }
        for (int k = 0; k < K; k++)
          for (int j = 0; j < N; j++) {
            double acc = gB0[k * N + j];
            for (int i = 0; i < M; i++) acc += (double)A.data[i * K + k] * C.grad[i * N + j];
            rB[k * N + j] = acc;
          }

        float *out[2][4];                         // [portable|simd][C, dA, dB, dx]
        for (int mode = 0; mode < 2; mode++) {
          matmul_set_simd(mode);
          out[mode][0] = (float*)malloc(C.size * sizeof(float));
          out[mode][1] = (float*)malloc(A.size * sizeof(float));
          out[mode][2] = (float*)malloc(B.size * sizeof(float));
          out[mode][3] = (float*)malloc(A.size * sizeof(float));
          matmul(&A, &B, &C);
          memcpy(out[mode][0], C.data, C.size * sizeof(float));
          memcpy(A.grad, gA0, A.size * sizeof(float));
          memcpy(B.grad, gB0, B.size * sizeof(float));
          matmul_backward(&A, &B, &C);
          memcpy(out[mode][1], A.grad, A.size * sizeof(float));
          memcpy(out[mode][2], B.grad, B.size * sizeof(float));
          memcpy(A.grad, gA0, A.size * sizeof(float));
          matmul_backward_dx(&A, &B, &C);
          memcpy(out[mode][3], A.grad, A.size * sizeof(float));
        }
        double e;
        RELD(C.size, out[1][0], out[0][0], e); if (e > w_fwd) w_fwd = e;
        RELD(A.size, out[1][1], out[0][1], e); if (e > w_da) w_da = e;
        RELD(B.size, out[1][2], out[0][2], e); if (e > w_db) w_db = e;
        RELD(A.size, out[1][3], out[0][3], e); if (e > w_dx) w_dx = e;
        for (int mode = 0; mode < 2; mode++) {
          double *worst = mode ? &ref_simd : &ref_port;
          RELD(C.size, out[mode][0], rC, e); if (e > *worst) *worst = e;
          RELD(A.size, out[mode][1], rA, e); if (e > *worst) *worst = e;
          RELD(B.size, out[mode][2], rB, e); if (e > *worst) *worst = e;
          RELD(A.size, out[mode][3], rA, e); if (e > *worst) *worst = e;
        }

        // (c) row-by-row (M=1) SIMD forward == the full SIMD forward, exactly
        matmul_set_simd(1);
        int s1[2] = {1, K}, s2[2] = {1, N};
        Tensor a1 = tensor_create(s1, 2), c1 = tensor_create(s2, 2);
        for (int i = 0; i < M; i++) {
          memcpy(a1.data, &A.data[(size_t)i * K], (size_t)K * sizeof(float));
          matmul(&a1, &B, &c1);
          if (memcmp(c1.data, &out[1][0][(size_t)i * N], (size_t)N * sizeof(float)) != 0)
            rows_exact = 0;
        }
        tensor_free(&a1); tensor_free(&c1);

        for (int mode = 0; mode < 2; mode++)
          for (int q = 0; q < 4; q++) free(out[mode][q]);
        free(rC); free(rA); free(rB); free(gA0); free(gB0);
        tensor_free(&A); tensor_free(&B); tensor_free(&C);
      }
      #undef RELD

      // (d) int8 column accumulate: exact integer equality on ragged K, N
      const int QS[][2] = { {1, 1}, {1, 64}, {2, 17}, {33, 77}, {64, 100}, {5, 200}, {129, 1000} };
      int q_exact = 1;
      for (int s = 0; s < (int)(sizeof(QS) / sizeof(QS[0])); s++) {
        const int K = QS[s][0], N = QS[s][1];
        signed char *qa = (signed char*)malloc((size_t)K);
        signed char *qw = (signed char*)malloc((size_t)K * N);
        int32_t *acc0 = (int32_t*)malloc((size_t)N * sizeof(int32_t));
        int32_t *acc1 = (int32_t*)malloc((size_t)N * sizeof(int32_t));
        for (int k = 0; k < K; k++) qa[k] = (signed char)((int)(RNDF() * 127.0f));
        for (size_t i = 0; i < (size_t)K * N; i++) qw[i] = (signed char)((int)(RNDF() * 127.0f));
        qa[0] = -127; qw[0] = -127;               // the extremes, once
        matmul_set_simd(0); qacc_columns(qa, qw, K, N, acc0);
        matmul_set_simd(1); qacc_columns(qa, qw, K, N, acc1);
        if (memcmp(acc0, acc1, (size_t)N * sizeof(int32_t)) != 0) q_exact = 0;
        free(qa); free(qw); free(acc0); free(acc1);
      }
      #undef RNDF

      matmul_set_simd(was_on);
      gpu_set_enabled(gpu_was);
      const double tol = 1e-5;
      const int s_ok = (w_fwd < tol) && (w_da < tol) && (w_db < tol) && (w_dx < tol) &&
                       (ref_simd < tol);
      printf("    %d awkward shapes | SIMD vs portable: fwd %.1e | dA %.1e | dB %.1e | dx %.1e"
             " (tol %.0e) -> %s\n", nsh, w_fwd, w_da, w_db, w_dx, tol, s_ok ? "OK" : "FAIL");
      printf("    vs a double reference (incl. pre-seeded grads): SIMD %.1e | portable %.1e\n",
             ref_simd, ref_port);
      printf("    1-row matmuls == rows of the M-row matmul (bit-exact) -> %s\n",
             rows_exact ? "OK" : "FAIL");
      printf("    int8 column accumulate: SIMD == portable (exact) -> %s\n\n",
             q_exact ? "OK" : "FAIL");
      all_ok &= s_ok && rows_exact && q_exact;
    }
  }

  // --------------------------------------------------------------------------
  // Part 35: Muon optimizer (muon.h) — Newton-Schulz orthogonalized momentum on
  // the hidden weight matrices, AdamW on everything else.
  //  (a) Newton-Schulz: for well-conditioned random matrices (tall, wide and
  //      square) every singular value of the output lands in the documented
  //      band around 1 — measured EXACTLY, from a double Jacobi eigensolve of
  //      the small-side Gram O^T O (tall) / O O^T (wide) — and O points the
  //      same way as the exact polar factor U V^T (cosine). Tall == wide
  //      transposed, bit for bit; a zero matrix maps to zero.
  //  (b) one Muon step: zero grad => weights scale by exactly (1 - lr*wd)
  //      (decoupled decay); a random grad => update RMS / lr ~= 0.2 (Moonshot
  //      RMS matching, what lets AdamW LRs transfer).
  //  (c) the SAME toy next-token task, init, LR (0.02) and step count (200) as
  //      part [22]: Adam vs Muon final loss, reported as measured. Muon is
  //      gated only on training (loss halves, finite) — NOT on beating Adam.
  //  (d) the default path is untouched: train_default_config() selects Adam,
  //      the routing rule sends exactly the 14 hidden matrices to Muon, and an
  //      all-AdamW mask makes muon_step bit-identical to adam_step.
  // --------------------------------------------------------------------------
  {
    printf("[35] Muon optimizer (Newton-Schulz orthogonalized momentum)\n");
    unsigned lcg = 777u;
    #define RNDF() ((float)((lcg = lcg * 1103515245u + 12345u) >> 9 & 0xFFFF) / 32768.0f - 1.0f)

    // (a) Newton-Schulz on tall / wide / square. A random tall r x c with r >> c
    //     is well-conditioned (sigma_max/sigma_min ~ (sqrt r + sqrt c)/(sqrt r - sqrt c));
    //     the square case is made so explicitly (I + 0.3 * noise).
    const int NS_SH[][2] = { {48, 12}, {12, 48}, {16, 16}, {64, 32} };
    const float sv_lo = 0.6f, sv_hi = 1.25f;       // the tolerance: sigma(O) in [0.6, 1.25]
    double sv_min = 1e9, sv_max = 0.0, cos_min = 1.0, gram_dev = 0.0;
    for (int s = 0; s < 4; s++) {
      const int r = NS_SH[s][0], c = NS_SH[s][1], k = r < c ? r : c;
      float *G = (float*)malloc((size_t)r * c * sizeof(float));
      float *O = (float*)malloc((size_t)r * c * sizeof(float));
      for (int i = 0; i < r * c; i++) G[i] = RNDF();
      if (r == c) for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) G[i * c + j] = (i == j) + 0.3f * G[i * c + j];
      double *sv = (double*)malloc((size_t)k * sizeof(double));
      double *polar = (double*)malloc((size_t)r * c * sizeof(double));
      small_gram_svd(G, r, c, sv, polar);          // exact polar factor of G
      muon_newton_schulz(G, r, c, O, 5);
      double dev = small_gram_svd(O, r, c, sv, NULL);
      if (dev > gram_dev) gram_dev = dev;
      for (int i = 0; i < k; i++) {
        if (sv[i] < sv_min) sv_min = sv[i];
        if (sv[i] > sv_max) sv_max = sv[i];
      }
      double dot = 0, no = 0, np = 0;
      for (int i = 0; i < r * c; i++) {
        dot += O[i] * polar[i]; no += (double)O[i] * O[i]; np += polar[i] * polar[i];
      }
      double cs = dot / sqrt(no * np + 1e-300);
      if (cs < cos_min) cos_min = cs;
      free(G); free(O); free(sv); free(polar);
    }
    // tall == transposed wide: NS(G^T)^T must equal NS(G) exactly.
    int tr_exact = 1;
    {
      const int r = 40, c = 10;
      float *G = (float*)malloc(r * c * sizeof(float)), *GT = (float*)malloc(r * c * sizeof(float));
      float *O = (float*)malloc(r * c * sizeof(float)), *OT = (float*)malloc(r * c * sizeof(float));
      for (int i = 0; i < r * c; i++) G[i] = RNDF();
      for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) GT[j * r + i] = G[i * c + j];
      muon_newton_schulz(G, r, c, O, 5);
      muon_newton_schulz(GT, c, r, OT, 5);
      for (int i = 0; i < r; i++) for (int j = 0; j < c; j++)
        if (O[i * c + j] != OT[j * r + i]) tr_exact = 0;
      for (int i = 0; i < r * c; i++) G[i] = 0.0f;
      muon_newton_schulz(G, r, c, O, 5);           // zero in => zero out, no NaN
      for (int i = 0; i < r * c; i++) if (O[i] != 0.0f) tr_exact = 0;
      free(G); free(GT); free(O); free(OT);
    }
    int ns_ok = (sv_min >= sv_lo) && (sv_max <= sv_hi) && (cos_min > 0.95) && tr_exact;
    printf("    Newton-Schulz (5 steps; 48x12, 12x48, 16x16, 64x32): sigma(O) in [%.3f, %.3f]"
           " (tol [%.2f, %.2f]) | max|O^T O - I| %.3f\n", sv_min, sv_max, sv_lo, sv_hi, gram_dev);
    printf("    cos(O, exact polar U V^T) >= %.4f | tall == wide^T bit-exact, zero -> zero: %s -> %s\n",
           cos_min, tr_exact ? "yes" : "NO", ns_ok ? "OK" : "FAIL");
    all_ok &= ns_ok;

    // (b) one step: decoupled decay, then the RMS matching.
    int ws[2] = {48, 12};
    Tensor W = tensor_create(ws, 2);
    for (size_t i = 0; i < W.size; i++) W.data[i] = 1.0f;
    tensor_zero_grad(&W);
    Tensor *wp[1] = {&W};
    int one = 1;
    const float mlr = 0.1f, mwd = 0.5f;
    Muon dm = muon_create(wp, 1, &one, mlr, mwd);
    muon_step(&dm, wp, 1);
    float want = 1.0f - mlr * mwd, wgot = W.data[0];
    int dec_ok = 1;
    for (size_t i = 0; i < W.size; i++) if (fabsf(W.data[i] - want) > 1e-6f) dec_ok = 0;
    muon_free(&dm);
    for (size_t i = 0; i < W.size; i++) { W.data[i] = 0.0f; W.grad[i] = RNDF(); }
    const float rlr = 1e-3f;
    Muon rm = muon_create(wp, 1, &one, rlr, 0.0f);
    muon_step(&rm, wp, 1);
    double ss = 0.0;
    for (size_t i = 0; i < W.size; i++) ss += (double)W.data[i] * W.data[i];
    float urms = (float)(sqrt(ss / (double)W.size) / rlr);
    int rms_ok = (urms > 0.14f) && (urms < 0.26f);
    muon_free(&rm);
    tensor_free(&W);
    printf("    step: zero grad -> w %.6f (want %.6f, decoupled decay) | update RMS / lr = %.4f"
           " (want ~0.2) -> %s\n", wgot, want, urms, (dec_ok && rms_ok) ? "OK" : "FAIL");
    all_ok &= dec_ok && rms_ok;
    #undef RNDF

    // (c) Adam vs Muon on part [22]'s task: same model, init (srand 23), data, LR, steps.
    const int vocab = 24, dmod = 8, nl = 2, nh = 4, nkv = 2, hd = 2, dff = 16, seq = 5;
    int ids[5]     = {2, 9, 4, 1, 7};
    int targets[5] = {9, 4, 1, 7, 3};
    float first[2] = {0}, mid[2] = {0}, last[2] = {0};
    int nmuon = 0, ntot = 0, bit_same = 1;
    for (int run = 0; run < 3; run++) {          // 0 Adam, 1 Muon, 2 Muon with all-AdamW mask
      srand(23);
      LlamaModel tm = llama_model_create_train(vocab, dmod, nl, nh, nkv, hd, dff, 64,
                                               10000.0f, 1e-5f);
      llama_model_randomize(&tm);
      LlamaActs acts = llama_acts_create(&tm, seq);
      int ls[2] = {seq, vocab};
      Tensor logits = tensor_create(ls, 2), probs = tensor_create(ls, 2);
      Tensor **lps = (Tensor**)malloc((size_t)llama_param_count(&tm) * sizeof(Tensor*));
      int n_lp = llama_collect_params(&tm, lps);
      ntot = n_lp;
      int *mask = (int*)malloc((size_t)n_lp * sizeof(int));
      Tensor *ex[2] = { &tm.wte, &tm.lm_head };
      if (run == 1) nmuon = muon_route(lps, n_lp, ex, 2, mask);
      else for (int i = 0; i < n_lp; i++) mask[i] = 0;
      Adam aopt = adam_create(lps, n_lp, 0.02f);
      Muon mopt = muon_create(lps, n_lp, mask, 0.02f, 0.0f);
      const int steps = (run == 2) ? 20 : 200;
      float l0 = 0.0f, l = 0.0f;
      for (int s = 0; s < steps; s++) {
        l = llama_ce_loss(&tm, &acts, ids, seq, targets, &logits, &probs);
        if (s == 0) l0 = l;
        if (s == 50 && run < 2) mid[run] = l;
        tensor_zero_grad(&logits);
        cross_entropy_backward(&probs, targets, &logits);
        llama_backward(&tm, ids, seq, &acts, &logits);
        if (run == 0) adam_step(&aopt, lps, n_lp);
        else          muon_step(&mopt, lps, n_lp);
      }
      if (run < 2) { first[run] = l0; last[run] = l; }
      if (run == 2) {
        // Replay with plain Adam on a fresh copy of the same init: must match exactly.
        srand(23);
        LlamaModel am = llama_model_create_train(vocab, dmod, nl, nh, nkv, hd, dff, 64,
                                                 10000.0f, 1e-5f);
        llama_model_randomize(&am);
        Tensor **aps = (Tensor**)malloc((size_t)n_lp * sizeof(Tensor*));
        llama_collect_params(&am, aps);
        Adam a2 = adam_create(aps, n_lp, 0.02f);
        for (int s = 0; s < steps; s++) {
          llama_ce_loss(&am, &acts, ids, seq, targets, &logits, &probs);
          tensor_zero_grad(&logits);
          cross_entropy_backward(&probs, targets, &logits);
          llama_backward(&am, ids, seq, &acts, &logits);
          adam_step(&a2, aps, n_lp);
        }
        for (int p = 0; p < n_lp; p++)
          if (memcmp(lps[p]->data, aps[p]->data, lps[p]->size * sizeof(float)) != 0) bit_same = 0;
        adam_free(&a2); free(aps); llama_model_free(&am);
      }
      adam_free(&aopt); muon_free(&mopt);
      free(mask); free(lps);
      tensor_free(&logits); tensor_free(&probs);
      llama_acts_free(&acts); llama_model_free(&tm);
    }
    int mtrain_ok = (last[1] < first[1] * 0.5f) && isfinite(last[1]);
    printf("    [22]'s task, lr 0.02, 200 steps, loss at step 0 / 50 / 199: Adam %.4f / %.4f / %.4f"
           " | Muon %.4f / %.4f / %.4f\n", first[0], mid[0], last[0], first[1], mid[1], last[1]);
    printf("    (reported as measured, NOT gated on Muon winning; gate: Muon halves the loss) -> %s\n",
           mtrain_ok ? "OK" : "FAIL");
    all_ok &= mtrain_ok;

    // (d) default path untouched.
    TrainConfig dc = train_default_config();
    int def_ok = (dc.muon == 0) && (nmuon == 14) && bit_same;
    printf("    default optim = %s | routing: %d/%d tensors to Muon (the 14 hidden matrices;"
           " wte, lm_head, norms, biases -> AdamW) | all-AdamW mask == adam_step bit-exact: %s -> %s\n\n",
           dc.muon ? "muon" : "adam", nmuon, ntot,
           bit_same ? "yes" : "NO", def_ok ? "OK" : "FAIL");
    all_ok &= def_ok;
  }

  // --------------------------------------------------------------------------
  printf("Output validation: %s\n\n", all_ok ? "PASSED" : "FAILED");
  printf("Done! All memory freed successfully.\n");
  return all_ok ? 0 : 1;
}
