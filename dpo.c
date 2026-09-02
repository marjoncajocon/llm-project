// dpo.c - Direct Preference Optimization (see dpo.h)
#include "dpo.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "activation.h"
#include "hftok.h"
#include "llama.h"
#include "llamatrain.h"
#include "loss.h"
#include "optim.h"
#include "sampler.h"
#include "tensor.h"

#define MAX_IDS 4096

DpoConfig dpo_default_config(void) {
  DpoConfig c;
  memset(&c, 0, sizeof(c));
  c.out_model = "model/dpo.bin";
  c.beta = 0.1f;
  c.len_norm = 0;
  c.steps = 200;
  c.batch = 4;
  c.lr = 5e-6f;          // DPO is a sharp objective; SFT learning rates diverge here
  c.weight_decay = 0.0f;
  c.grad_clip = 1.0f;
  c.warmup = 0;
  c.lr_min = -1.0f;
  c.lora_r = 8;
  c.lora_alpha = 16.0f;
  c.seed = 1;
  c.verbose = 0;
  return c;
}

// ---------------------------------------------------------------------------
// The objective (pure arithmetic - what the self-test gradient-checks)
// ---------------------------------------------------------------------------
float dpo_pair_loss(float pi_chosen, float pi_rejected,
                    float ref_chosen, float ref_rejected,
                    float beta, float *g_out, float *margin_out, int *acc_out) {
  const float s_w = pi_chosen - ref_chosen;
  const float s_l = pi_rejected - ref_rejected;
  const float margin = s_w - s_l;
  const float h = beta * margin;

  // softplus(-h), computed the stable way: for large positive h the naive
  // log(1+exp(-h)) underflows to 0 and for large negative h exp(-h) overflows.
  const float loss = (h > 0.0f) ? logf(1.0f + expf(-h)) : (-h + logf(1.0f + expf(h)));

  // g = beta * sigmoid(-h). Near beta while the model still prefers the wrong
  // completion, near 0 once it has learned the preference: the update anneals
  // itself away pair by pair.
  const float sig_neg = (h > 0.0f) ? (expf(-h) / (1.0f + expf(-h)))
                                   : (1.0f / (1.0f + expf(h)));
  if (g_out) *g_out = beta * sig_neg;
  if (margin_out) *margin_out = margin;
  if (acc_out) *acc_out = (h > 0.0f) ? 1 : 0;
  return loss;
}

// ---------------------------------------------------------------------------
// Preference pairs
// ---------------------------------------------------------------------------
typedef struct {
  int *p, *c, *r;      // token ids: prompt, chosen, rejected
  int np, nc, nr;      // lengths (chosen/rejected already include the EOS slot)
  float ref_c, ref_r;  // reference log-probs, precomputed once
} Pair;

static char *dup_str(const char *s) {
  const size_t n = strlen(s) + 1;
  char *p = (char*)malloc(n);
  if (p) memcpy(p, s, n);
  return p;
}

// Read `prompt<TAB>chosen<TAB>rejected` lines into raw text triples.
typedef struct { char *p, *c, *r; } RawPair;

static RawPair *pairs_load(const char *path, int *n_out) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "error: cannot open pair file '%s'\n", path); return NULL; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (len <= 0) { fclose(f); fprintf(stderr, "error: pair file is empty\n"); return NULL; }
  char *buf = (char*)malloc((size_t)len + 1);
  if (!buf) { fclose(f); return NULL; }
  size_t got = fread(buf, 1, (size_t)len, f);
  buf[got] = '\0';
  fclose(f);

  int cap = 64, n = 0;
  RawPair *rp = (RawPair*)malloc((size_t)cap * sizeof(RawPair));
  if (!rp) { free(buf); return NULL; }

  for (char *line = buf; *line; ) {
    char *nl = strchr(line, '\n');
    if (nl) *nl = '\0';
    size_t ll = strlen(line);
    if (ll && line[ll - 1] == '\r') line[ll - 1] = '\0';

    char *t1 = strchr(line, '\t');
    char *t2 = t1 ? strchr(t1 + 1, '\t') : NULL;
    if (t1 && t2) {
      *t1 = '\0'; *t2 = '\0';
      const char *prompt = line, *chosen = t1 + 1, *rejected = t2 + 1;
      if (prompt[0] && chosen[0] && rejected[0]) {
        if (n == cap) {
          cap *= 2;
          RawPair *bigger = (RawPair*)realloc(rp, (size_t)cap * sizeof(RawPair));
          if (!bigger) break;
          rp = bigger;
        }
        rp[n].p = dup_str(prompt);
        rp[n].c = dup_str(chosen);
        rp[n].r = dup_str(rejected);
        if (rp[n].p && rp[n].c && rp[n].r) n++;
      }
    }
    if (!nl) break;
    line = nl + 1;
  }
  free(buf);
  *n_out = n;
  if (n == 0) {
    fprintf(stderr, "error: no 'prompt<TAB>chosen<TAB>rejected' lines found\n"
                    "       (three tab-separated columns are required, not two)\n");
    free(rp);
    return NULL;
  }
  return rp;
}

// ---------------------------------------------------------------------------
// Model plumbing (mirrors grpo.c)
// ---------------------------------------------------------------------------
static void set_rows(Tensor *t, int rows, int cols) {
  t->shape[0] = rows;
  t->shape[1] = cols;
  t->size = (size_t)rows * (size_t)cols;
}

static int copy_weights(LlamaModel *dst, LlamaModel *src) {
  const int n = llama_param_count(dst);
  if (n != llama_param_count(src)) return 0;
  Tensor **sp = (Tensor**)malloc((size_t)n * sizeof(Tensor*));
  Tensor **dp = (Tensor**)malloc((size_t)n * sizeof(Tensor*));
  if (!sp || !dp) { free(sp); free(dp); return 0; }
  llama_collect_params(src, sp);
  llama_collect_params(dst, dp);
  int ok = 1;
  for (int i = 0; i < n; i++) {
    if (sp[i]->size != dp[i]->size) { ok = 0; break; }
    memcpy(dp[i]->data, sp[i]->data, sp[i]->size * sizeof(float));
  }
  free(sp); free(dp);
  return ok;
}

// Build prompt ++ completion into seq, and the target mask. Scored rows are
// T-1 .. L-2 (row j predicts token j+1), exactly as the SFT trainer and GRPO do:
// the prompt is context, not something the policy is graded on.
static int build_seq(const int *prompt, int T, const int *comp, int R,
                     int *seq, int *tgt) {
  const int L = T + R;
  memcpy(seq, prompt, (size_t)T * sizeof(int));
  memcpy(&seq[T], comp, (size_t)R * sizeof(int));
  for (int j = 0; j < L; j++) tgt[j] = -1;
  for (int j = T - 1; j <= L - 2; j++) tgt[j] = seq[j + 1];
  return L;
}

// Forward one sequence and return the SUMMED log-probability of its scored
// tokens. lo == NULL runs the reference policy (adapters off).
static float seq_logprob(LlamaModel *m, const int *seq, int L, int T, int R,
                         LlamaActs *acts, Tensor *logits, Tensor *probs,
                         const LlamaLora *lo, LlamaLoraActs *la, int len_norm) {
  llama_acts_set_len(acts, L);
  if (lo) llama_lora_acts_set_len(la, L);
  set_rows(logits, L, m->vocab);
  set_rows(probs, L, m->vocab);
  llama_forward_train_lora(m, seq, L, acts, logits, lo, lo ? la : NULL);
  softmax_rowwise(logits, probs);
  float sum = 0.0f;
  for (int t = 0; t < R; t++) sum += logprob_of(probs, T - 1 + t, seq[T + t]);
  return len_norm ? sum / (float)R : sum;
}

int dpo_run(const char *pairs_path, const DpoConfig *cfg) {
  if (!cfg->model || !cfg->hf_tok) {
    fprintf(stderr, "error: dpo needs --model <LLMA checkpoint> and --hf-tok <vocab.hf>\n");
    return 1;
  }
  if (cfg->lora_r <= 0) {
    fprintf(stderr,
            "error: dpo requires --lora-r > 0.\n"
            "       DPO needs a FROZEN reference policy. With LoRA that is free -- it is\n"
            "       the same weights with the adapters switched off. Without adapters it\n"
            "       would mean holding a second full copy of the model in memory, which\n"
            "       this trainer deliberately does not do.\n");
    return 1;
  }
  if (cfg->beta <= 0.0f) {
    fprintf(stderr, "error: --beta must be > 0 (it is the KL strength; 0.1 is the usual default)\n");
    return 1;
  }

  int n_raw = 0;
  RawPair *raw = pairs_load(pairs_path, &n_raw);
  if (!raw) return 1;

  HFTok tok;
  if (hftok_load(&tok, cfg->hf_tok) != 0) {
    fprintf(stderr, "error: cannot load HF tokenizer '%s'\n", cfg->hf_tok);
    for (int i = 0; i < n_raw; i++) { free(raw[i].p); free(raw[i].c); free(raw[i].r); }
    free(raw);
    return 1;
  }

  LlamaModel loaded;
  if (llama_model_load(cfg->model, &loaded) != 0) {
    fprintf(stderr, "error: cannot load model '%s'\n", cfg->model);
    hftok_free(&tok);
    for (int i = 0; i < n_raw; i++) { free(raw[i].p); free(raw[i].c); free(raw[i].r); }
    free(raw);
    return 1;
  }
  LlamaModel model = llama_model_create_train(loaded.vocab, loaded.d_model, loaded.n_layers,
                                              loaded.n_heads, loaded.n_kv, loaded.head_dim,
                                              loaded.d_ff, loaded.max_seq, loaded.rope_base,
                                              loaded.eps);
  if (!copy_weights(&model, &loaded)) {
    fprintf(stderr, "error: checkpoint geometry did not survive the trainable rebuild\n");
    llama_model_free(&loaded); llama_model_free(&model); hftok_free(&tok);
    for (int i = 0; i < n_raw; i++) { free(raw[i].p); free(raw[i].c); free(raw[i].r); }
    free(raw);
    return 1;
  }
  llama_model_free(&loaded);

  const int V = model.vocab;
  const int EOS = tok.eos;

  // Tokenize every pair; drop any that cannot fit the context.
  Pair *pairs = (Pair*)calloc((size_t)n_raw, sizeof(Pair));
  int npairs = 0, maxL = 0;
  int scratch[MAX_IDS];
  for (int i = 0; i < n_raw; i++) {
    int T = hftok_encode(&tok, raw[i].p, scratch, MAX_IDS);
    if (T < 1) continue;
    int *P = (int*)malloc((size_t)T * sizeof(int));
    if (!P) continue;
    memcpy(P, scratch, (size_t)T * sizeof(int));

    int NC = hftok_encode(&tok, raw[i].c, scratch, MAX_IDS - 1);
    int *C = NULL;
    if (NC >= 1) {
      C = (int*)malloc((size_t)(NC + 1) * sizeof(int));
      if (C) { memcpy(C, scratch, (size_t)NC * sizeof(int)); C[NC] = EOS; NC += 1; }
    }
    int NR = hftok_encode(&tok, raw[i].r, scratch, MAX_IDS - 1);
    int *R = NULL;
    if (NR >= 1) {
      R = (int*)malloc((size_t)(NR + 1) * sizeof(int));
      if (R) { memcpy(R, scratch, (size_t)NR * sizeof(int)); R[NR] = EOS; NR += 1; }
    }
    const int Lc = T + NC, Lr = T + NR;
    if (!C || !R || NC < 2 || NR < 2 || Lc > model.max_seq || Lr > model.max_seq) {
      free(P); free(C); free(R);
      continue;
    }
    pairs[npairs].p = P;  pairs[npairs].np = T;
    pairs[npairs].c = C;  pairs[npairs].nc = NC;
    pairs[npairs].r = R;  pairs[npairs].nr = NR;
    if (Lc > maxL) maxL = Lc;
    if (Lr > maxL) maxL = Lr;
    npairs++;
  }
  for (int i = 0; i < n_raw; i++) { free(raw[i].p); free(raw[i].c); free(raw[i].r); }
  free(raw);

  if (npairs == 0) {
    fprintf(stderr, "error: no pair fits the model's %d-position context\n", model.max_seq);
    free(pairs); llama_model_free(&model); hftok_free(&tok);
    return 1;
  }

  LlamaActs acts = llama_acts_create(&model, maxL);
  int lshape[2] = {maxL, V};
  Tensor logits = tensor_create(lshape, 2), probs = tensor_create(lshape, 2);

  LlamaLora lora = llama_lora_create(&model, cfg->lora_r,
                                     cfg->lora_alpha > 0.0f ? cfg->lora_alpha : (float)cfg->lora_r);
  llama_lora_init(&lora);
  LlamaLoraActs lacts = llama_lora_acts_create(&model, &lora, maxL);
  Tensor **params = (Tensor**)malloc((size_t)llama_lora_param_count(&lora) * sizeof(Tensor*));
  int n_params = llama_lora_collect_params(&lora, params);
  Adam opt = adamw_create(params, n_params, cfg->lr, cfg->weight_decay);

  int *seq = (int*)malloc((size_t)maxL * sizeof(int));
  int *tgt = (int*)malloc((size_t)maxL * sizeof(int));
  float *coeff = (float*)malloc((size_t)maxL * sizeof(float));

  printf("DPO: %d pairs (of %d lines) | beta %.3g | %d steps x batch %d | max seq %d\n",
         npairs, n_raw, (double)cfg->beta, cfg->steps, cfg->batch, maxL);
  printf("  model %s (vocab %d, d_model %d, layers %d) | tok %s\n",
         cfg->model, V, model.d_model, model.n_layers, cfg->hf_tok);
  printf("  LoRA: rank %d alpha %g on Wq/Wk/Wv/Wo, base FROZEN (%d adapter tensors)\n",
         cfg->lora_r, (double)(cfg->lora_alpha > 0.0f ? cfg->lora_alpha : (float)cfg->lora_r),
         n_params);
  printf("  reference policy = the same weights with the adapters OFF (no second copy)\n");
  printf("  lr %g | clip %.2g | length-normalized log-probs: %s\n",
         (double)cfg->lr, (double)cfg->grad_clip, cfg->len_norm ? "yes" : "no (standard DPO)");
  fflush(stdout);

  // ---- Precompute the reference log-probs, once ----
  // pi_ref never changes, so computing it inside the training loop would repeat
  // the same two forwards every step for no reason. This is the standard DPO
  // implementation trick and it roughly halves the cost of a run.
  {
    const clock_t t0 = clock();
    for (int i = 0; i < npairs; i++) {
      int L = build_seq(pairs[i].p, pairs[i].np, pairs[i].c, pairs[i].nc, seq, tgt);
      pairs[i].ref_c = seq_logprob(&model, seq, L, pairs[i].np, pairs[i].nc,
                                   &acts, &logits, &probs, NULL, NULL, cfg->len_norm);
      L = build_seq(pairs[i].p, pairs[i].np, pairs[i].r, pairs[i].nr, seq, tgt);
      pairs[i].ref_r = seq_logprob(&model, seq, L, pairs[i].np, pairs[i].nr,
                                   &acts, &logits, &probs, NULL, NULL, cfg->len_norm);
      if (npairs > 20 && (i + 1) % (npairs / 10) == 0) {
        printf("  reference pass: %d/%d pairs\n", i + 1, npairs);
        fflush(stdout);
      }
    }
    printf("  reference log-probs cached for %d pairs in %.1fs\n",
           npairs, (double)(clock() - t0) / CLOCKS_PER_SEC);
    fflush(stdout);
  }

  srand(cfg->seed);
  const int report = cfg->steps < 10 ? 1 : cfg->steps / 10;
  double run_loss = 0.0, run_margin = 0.0;
  int run_acc = 0, run_n = 0;
  int rc = 0;

  for (int s = 0; s < cfg->steps; s++) {
    llama_zero_grad(&model);
    llama_lora_zero_grad(&lora);
    double step_loss = 0.0, step_margin = 0.0;
    int step_acc = 0;

    for (int b = 0; b < cfg->batch; b++) {
      const Pair *pr = &pairs[rand() % npairs];
      const int T = pr->np;

      // 1. Policy log-prob of the CHOSEN completion.
      int Lc = build_seq(pr->p, T, pr->c, pr->nc, seq, tgt);
      const float pi_c = seq_logprob(&model, seq, Lc, T, pr->nc,
                                     &acts, &logits, &probs, &lora, &lacts, cfg->len_norm);

      // 2. Policy log-prob of the REJECTED completion. This leaves the activation
      //    cache holding the REJECTED sequence, which is why its backward runs
      //    first below - it saves one forward pass per pair.
      int Lr = build_seq(pr->p, T, pr->r, pr->nr, seq, tgt);
      const float pi_r = seq_logprob(&model, seq, Lr, T, pr->nr,
                                     &acts, &logits, &probs, &lora, &lacts, cfg->len_norm);

      // 3. One scalar decides both updates.
      float g = 0.0f, margin = 0.0f;
      int acc = 0;
      const float loss = dpo_pair_loss(pi_c, pi_r, pr->ref_c, pr->ref_r,
                                       cfg->beta, &g, &margin, &acc);
      step_loss += loss;
      step_margin += margin;
      step_acc += acc;

      // 4. Rejected: push DOWN (coefficient -g). The cache already holds it.
      for (int j = 0; j < Lr; j++) coeff[j] = 0.0f;
      for (int j = T - 1; j <= Lr - 2; j++) coeff[j] = -g;
      tensor_zero_grad(&logits);
      pg_backward(&probs, tgt, coeff, &logits);
      llama_backward_accum(&model, seq, Lr, &acts, &logits, &lora, &lacts);

      // 5. Chosen: push UP (coefficient +g). Re-forward to refill the cache.
      Lc = build_seq(pr->p, T, pr->c, pr->nc, seq, tgt);
      (void)seq_logprob(&model, seq, Lc, T, pr->nc,
                        &acts, &logits, &probs, &lora, &lacts, cfg->len_norm);
      for (int j = 0; j < Lc; j++) coeff[j] = 0.0f;
      for (int j = T - 1; j <= Lc - 2; j++) coeff[j] = g;
      tensor_zero_grad(&logits);
      pg_backward(&probs, tgt, coeff, &logits);
      llama_backward_accum(&model, seq, Lc, &acts, &logits, &lora, &lacts);
    }

    if (cfg->batch > 1) scale_grads(params, n_params, 1.0f / (float)cfg->batch);
    if (cfg->grad_clip > 0.0f) clip_grad_norm(params, n_params, cfg->grad_clip);
    float lr = cfg->lr;
    if (cfg->warmup > 0 || cfg->lr_min >= 0.0f)
      lr = lr_cosine_warmup(cfg->lr, cfg->lr_min < 0.0f ? cfg->lr : cfg->lr_min,
                            s, cfg->warmup, cfg->steps);
    adam_set_lr(&opt, lr);
    adam_step(&opt, params, n_params);

    run_loss += step_loss / cfg->batch;
    run_margin += step_margin / cfg->batch;
    run_acc += step_acc;
    run_n += cfg->batch;

    if (s % report == 0 || s == cfg->steps - 1) {
      printf("  step %5d/%d  loss %.4f  margin %+.4f  pref-acc %.0f%%\n",
             s, cfg->steps, step_loss / cfg->batch, step_margin / cfg->batch,
             100.0 * step_acc / cfg->batch);
      fflush(stdout);
    }
  }

  if (cfg->steps > 0)
    printf("  over the run: mean loss %.4f | mean margin %+.4f | preference accuracy %.0f%%\n",
           run_loss / cfg->steps, run_margin / cfg->steps,
           run_n > 0 ? 100.0 * run_acc / run_n : 0.0);
  printf("  NOTE: loss falling and margin rising means the PREFERENCE was learned.\n"
         "        It does not mean the model got better at the task -- only\n"
         "        tools/remeasure.py can tell you that. Gate promotion on it.\n");

  llama_lora_merge(&model, &lora);
  printf("  LoRA: merged adapters into the base weights\n");
  if (llama_model_save(cfg->out_model, &model) != 0) {
    fprintf(stderr, "error saving model\n");
    rc = 1;
  } else {
    printf("saved DPO model -> %s (tokenizer: %s, unchanged)\n", cfg->out_model, cfg->hf_tok);
  }

  free(seq); free(tgt); free(coeff); free(params);
  for (int i = 0; i < npairs; i++) { free(pairs[i].p); free(pairs[i].c); free(pairs[i].r); }
  free(pairs);
  llama_lora_acts_free(&lacts); llama_lora_free(&lora);
  adam_free(&opt);
  tensor_free(&logits); tensor_free(&probs);
  llama_acts_free(&acts);
  llama_model_free(&model);
  hftok_free(&tok);
  return rc;
}
