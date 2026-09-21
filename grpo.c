// grpo.c - Group Relative Policy Optimization (see grpo.h)
#include "grpo.h"

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
#include "reward.h"
#include "sampler.h"
#include "tensor.h"

#define MAX_PROMPT_IDS 4096
#define TEXT_CAP 16384         // detokenized rollout buffer (max_new tokens of text)
#define LINE_CAP 400           // how much of a rollout --verbose echoes

GrpoConfig grpo_default_config(void) {
  GrpoConfig c;
  memset(&c, 0, sizeof(c));
  c.out_model = "model/grpo.bin";
  c.group = 8;
  c.max_new = 96;
  c.temperature = 1.0f;
  c.top_k = 0;
  c.top_p = 0.95f;
  c.steps = 100;
  c.groups_per_step = 1;
  c.inner_epochs = 1;
  c.lr = 1e-5f;
  c.weight_decay = 0.0f;
  c.grad_clip = 1.0f;      // ON by default: an RL gradient is far spikier than an
                           // SFT one, and this trainer diverges without it.
  c.warmup = 0;
  c.lr_min = -1.0f;        // constant LR
  c.lora_r = 8;
  c.lora_alpha = 16.0f;
  c.clip_eps = 0.2f;
  c.clip_eps_high = -1.0f;   // symmetric
  c.kl_beta = 0.0f;
  c.std_norm = 1;
  c.loss_agg = GRPO_AGG_SEQ;
  c.resample = 0;
  c.seed = 1;
  c.verbose = 0;
  return c;
}

void grpo_preset_dapo(GrpoConfig *c) {
  c->clip_eps_high = 0.28f;  // the DAPO paper's value, with clip_eps 0.2 below
  c->loss_agg = GRPO_AGG_TOKEN;
  c->resample = 2;
}

void grpo_preset_dr_grpo(GrpoConfig *c) {
  c->std_norm = 0;
  c->loss_agg = GRPO_AGG_CONST;
}

float grpo_loss_norm(const GrpoConfig *cfg, int G, int len, int step_tokens) {
  switch (cfg->loss_agg) {
    case GRPO_AGG_TOKEN: return (float)(step_tokens > 0 ? step_tokens : 1);
    case GRPO_AGG_CONST: return (float)G * (float)(cfg->max_new > 0 ? cfg->max_new : 1);
    default:             return (float)G * (float)(len > 0 ? len : 1);
  }
}

// ---------------------------------------------------------------------------
// The objective (pure arithmetic — the part the self-test gradient-checks)
// ---------------------------------------------------------------------------
int grpo_advantages(const float *rewards, int G, const GrpoConfig *cfg, float *adv_out) {
  if (G <= 0) return 0;
  for (int i = 0; i < G; i++) adv_out[i] = 0.0f;

  double mean = 0.0;
  for (int i = 0; i < G; i++) mean += rewards[i];
  mean /= G;
  double var = 0.0;
  for (int i = 0; i < G; i++) {
    const double d = rewards[i] - mean;
    var += d * d;
  }
  var /= G;                                  // population std: the group IS the set
  const double sd = sqrt(var);

  // No spread => no signal. Every rollout was equally good (or equally bad), so
  // the group's own mean baseline cancels it exactly. Report it and skip.
  if (sd < 1e-6) return 0;

  const double scale = cfg->std_norm ? 1.0 / (sd + 1e-4) : 1.0;
  for (int i = 0; i < G; i++)
    adv_out[i] = (float)((rewards[i] - mean) * scale);
  return 1;
}

void grpo_token_coeffs(float adv, float norm, int len, const float *logp_new,
                       const float *logp_old, const float *logp_ref,
                       const GrpoConfig *cfg, float *coeff_out) {
  if (len <= 0) return;
  const float inv = 1.0f / (norm > 0.0f ? norm : 1.0f);
  const float eps_high = cfg->clip_eps_high >= 0.0f ? cfg->clip_eps_high : cfg->clip_eps;
  for (int t = 0; t < len; t++) {
    // Importance ratio. Even at inner_epochs == 1 this is not exactly 1: the
    // rollout came from the KV-cache decode path and the update runs the
    // training forward, and the two agree only to float tolerance.
    float rho = expf(logp_new[t] - logp_old[t]);
    if (!isfinite(rho)) rho = 1.0f;
    if (rho > 20.0f) rho = 20.0f;            // guard a degenerate ratio

    int clipped = 0;
    if (cfg->clip_eps > 0.0f) {
      // The clip is one-sided per token: it only removes gradient that would push
      // a token further in the direction it has ALREADY moved too far.
      // The upper bound may be wider (clip-higher): see GrpoConfig.clip_eps_high.
      if (adv > 0.0f && rho > 1.0f + eps_high) clipped = 1;
      if (adv < 0.0f && rho < 1.0f - cfg->clip_eps) clipped = 1;
    }
    const float pg = clipped ? 0.0f : adv * rho;

    float kl = 0.0f;
    if (cfg->kl_beta > 0.0f && logp_ref) {
      // Schulman's k3 estimator: D = rho_ref - log(rho_ref) - 1, whose derivative
      // with respect to log pi_theta is -(rho_ref - 1).
      float rho_ref = expf(logp_ref[t] - logp_new[t]);
      if (!isfinite(rho_ref)) rho_ref = 1.0f;
      if (rho_ref > 20.0f) rho_ref = 20.0f;
      kl = cfg->kl_beta * (rho_ref - 1.0f);
    }
    coeff_out[t] = (pg + kl) * inv;
  }
}

int grpo_group_coeffs(const float *rewards, int G, const int *lens,
                      const float *logp_new, const float *logp_old,
                      const float *logp_ref, const GrpoConfig *cfg,
                      float *adv_out, float *coeff_out) {
  if (G <= 0) return 0;
  int total = 0;
  for (int i = 0; i < G; i++) total += lens[i];
  for (int t = 0; t < total; t++) coeff_out[t] = 0.0f;

  float *adv = (float*)malloc((size_t)G * sizeof(float));
  if (!adv) return 0;
  const int live = grpo_advantages(rewards, G, cfg, adv);
  if (adv_out) memcpy(adv_out, adv, (size_t)G * sizeof(float));
  if (!live) { free(adv); return 0; }

  // A lone group: its own tokens are the whole step's tokens.
  int off = 0;
  for (int i = 0; i < G; i++) {
    const float norm = grpo_loss_norm(cfg, G, lens[i], total);
    grpo_token_coeffs(adv[i], norm, lens[i], &logp_new[off], &logp_old[off],
                      logp_ref ? &logp_ref[off] : NULL, cfg, &coeff_out[off]);
    off += lens[i];
  }
  free(adv);
  return 1;
}

// ---------------------------------------------------------------------------
// Task file
// ---------------------------------------------------------------------------
typedef struct {
  char *prompt;
  char *reference;
} Task;

// strdup is POSIX, not C11, and this tree builds -std=c11 on four platforms.
static char *dup_str(const char *s) {
  const size_t n = strlen(s) + 1;
  char *p = (char*)malloc(n);
  if (p) memcpy(p, s, n);
  return p;
}

// Read `prompt<TAB>reference` lines. Same format as the SFT trainer, so a corpus
// from tools/clean_sft.py drops straight in.
static Task *tasks_load(const char *path, int *n_out) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "error: cannot open task file '%s'\n", path); return NULL; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (len <= 0) { fclose(f); fprintf(stderr, "error: task file is empty\n"); return NULL; }
  char *buf = (char*)malloc((size_t)len + 1);
  if (!buf) { fclose(f); return NULL; }
  size_t got = fread(buf, 1, (size_t)len, f);
  buf[got] = '\0';
  fclose(f);

  int cap = 64, n = 0;
  Task *ts = (Task*)malloc((size_t)cap * sizeof(Task));
  if (!ts) { free(buf); return NULL; }

  char *save = buf;
  for (char *line = buf; *line; ) {
    char *nl = strchr(line, '\n');
    if (nl) *nl = '\0';
    size_t ll = strlen(line);
    if (ll && line[ll - 1] == '\r') line[ll - 1] = '\0';
    char *tab = strchr(line, '\t');
    if (tab) {
      *tab = '\0';
      if (line[0] && tab[1]) {
        if (n == cap) {
          cap *= 2;
          Task *bigger = (Task*)realloc(ts, (size_t)cap * sizeof(Task));
          if (!bigger) break;
          ts = bigger;
        }
        ts[n].prompt = dup_str(line);
        ts[n].reference = dup_str(tab + 1);
        if (ts[n].prompt && ts[n].reference) n++;
      }
    }
    if (!nl) break;
    line = nl + 1;
  }
  free(save);
  *n_out = n;
  if (n == 0) {
    fprintf(stderr, "error: task file has no 'prompt<TAB>reference' lines\n");
    free(ts);
    return NULL;
  }
  return ts;
}

static void tasks_free(Task *ts, int n) {
  for (int i = 0; i < n; i++) { free(ts[i].prompt); free(ts[i].reference); }
  free(ts);
}

// ---------------------------------------------------------------------------
// Model plumbing
// ---------------------------------------------------------------------------

// Re-target a preallocated (rows_max, cols) buffer at fewer rows, exactly as the
// SFT trainer does — no realloc, just a smaller logical row count.
static void set_rows(Tensor *t, int rows, int cols) {
  t->shape[0] = rows;
  t->shape[1] = cols;
  t->size = (size_t)rows * (size_t)cols;
}

// Copy every parameter of `src` into the same-geometry `dst`. GRPO always starts
// from a real checkpoint and never grows the vocabulary (it uses the model's own
// tokenizer), so this is a straight 1:1 copy in canonical parameter order.
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

// One rollout: prefill the prompt through the KV cache, then sample up to
// max_new tokens from the CURRENT policy (adapters included). Records the token
// ids and the log-probability the policy assigned to each sampled token — the
// pi_old the importance ratio is measured against.
//
typedef enum { STOP_CAP = 0, STOP_EOS = 1, STOP_TAG = 2 } StopReason;

// Returns the number of sampled tokens, writes the detokenized text to `text`,
// and reports HOW the rollout ended in *stop.
//
// Sampling stops at a closing </tool> or </done> as well as at EOS, because that
// is what the deployed agent actually does: toolcall_parse takes the FIRST block
// and agent.c executes only that, so any text the model emits afterwards is
// discarded before it can have an effect. Scoring — and training on — tokens the
// harness throws away would optimize something that does not exist at runtime.
// It also cuts rollout cost sharply: a 135M model that has said everything it
// needs to say tends to keep going regardless.
//
// STOP_CAP is the case worth watching: the rollout was cut off at --max-new
// having never closed its block, and so loses the format term for a reason that
// is the budget's fault rather than the policy's.
static int rollout(const LlamaModel *m, const LlamaDecodeLora *lo, LlamaKV *kv,
                   const HFTok *tok, const int *prompt_ids, int n_prompt,
                   const GrpoConfig *cfg, Rng *rng, Tensor *logits, Tensor *probs,
                   int *out_ids, float *out_logp, char *text, int text_cap,
                   StopReason *stop) {
  kv->len = 0;
  for (int i = 0; i < n_prompt; i++)
    llama_decode_step_lora(m, kv, prompt_ids[i], logits, lo);

  int n = 0;
  *stop = STOP_CAP;
  text[0] = '\0';
  const int room = m->max_seq - n_prompt;
  const int cap = (cfg->max_new < room) ? cfg->max_new : room;
  while (n < cap) {
    // The log-prob is taken from the UNTEMPERED softmax: temperature/top-k/top-p
    // shape the exploration, but the policy whose ratio we measure — and whose
    // gradient we take — is the model's own distribution.
    softmax_rowwise(logits, probs);
    const int t = sample_token(logits, 0, cfg->temperature, cfg->top_k, cfg->top_p, rng);
    out_logp[n] = logprob_of(probs, 0, t);
    out_ids[n++] = t;
    if (hftok_is_eog(tok, t)) { *stop = STOP_EOS; break; }
    // Re-decode the whole response each step rather than tracking a partial
    // decode: a BPE piece can straddle the tag, and n is at most --max-new, so
    // this is nothing beside a 30-layer forward.
    hftok_decode(tok, out_ids, n, text, text_cap);
    if (strstr(text, "</tool>") || strstr(text, "</done>")) { *stop = STOP_TAG; break; }
    llama_decode_step_lora(m, kv, t, logits, lo);
  }
  hftok_decode(tok, out_ids, n, text, text_cap);
  return n;
}

// Copy up to `cap` bytes of `s` onto one line for the verbose log: control
// characters (a write_file body is full of newlines) become spaces.
static void one_line(const char *s, char *out, int cap) {
  int n = 0;
  for (const char *p = s; *p && n < cap - 1; p++)
    out[n++] = ((unsigned char)*p < 0x20) ? ' ' : *p;
  out[n] = '\0';
}

int grpo_run(const char *tasks_path, const GrpoConfig *cfg) {
  if (!cfg->model || !cfg->hf_tok) {
    fprintf(stderr, "error: grpo needs --model <LLMA checkpoint> and --hf-tok <vocab.hf>\n");
    return 1;
  }
  if (cfg->group < 2) {
    fprintf(stderr, "error: --group must be >= 2 (a group of one has no mean to be relative to)\n");
    return 1;
  }

  int n_tasks = 0;
  Task *tasks = tasks_load(tasks_path, &n_tasks);
  if (!tasks) return 1;

  HFTok tok;
  if (hftok_load(&tok, cfg->hf_tok) != 0) {
    fprintf(stderr, "error: cannot load HF tokenizer '%s'\n", cfg->hf_tok);
    tasks_free(tasks, n_tasks);
    return 1;
  }

  LlamaModel loaded;
  if (llama_model_load(cfg->model, &loaded) != 0) {
    fprintf(stderr, "error: cannot load model '%s'\n", cfg->model);
    hftok_free(&tok); tasks_free(tasks, n_tasks);
    return 1;
  }

  // The loaded model has no gradient arrays; rebuild it as a trainable one and
  // copy the weights across (the same two-step the SFT trainer's --resume does).
  LlamaModel model = llama_model_create_train(loaded.vocab, loaded.d_model, loaded.n_layers,
                                              loaded.n_heads, loaded.n_kv, loaded.head_dim,
                                              loaded.d_ff, loaded.max_seq, loaded.rope_base,
                                              loaded.eps);
  if (!copy_weights(&model, &loaded)) {
    fprintf(stderr, "error: checkpoint geometry did not survive the trainable rebuild\n");
    llama_model_free(&loaded); llama_model_free(&model);
    hftok_free(&tok); tasks_free(tasks, n_tasks);
    return 1;
  }
  llama_model_copy_window(&model, &loaded);   // keep its sliding-window config
  llama_model_free(&loaded);

  srand(cfg->seed);
  Rng rng = rng_create(cfg->seed ? cfg->seed : 1);

  const int G = cfg->group;
  const int V = model.vocab;

  // Tokenize the prompts once. Tasks whose prompt leaves no room for a rollout
  // are dropped here rather than silently producing empty responses.
  int **pids = (int**)malloc((size_t)n_tasks * sizeof(int*));
  int *plen = (int*)malloc((size_t)n_tasks * sizeof(int));
  int usable = 0, maxL = 0;
  int scratch[MAX_PROMPT_IDS];
  for (int i = 0; i < n_tasks; i++) {
    const int T = hftok_encode(&tok, tasks[i].prompt, scratch, MAX_PROMPT_IDS);
    pids[i] = NULL; plen[i] = 0;
    if (T < 1 || T + 2 >= model.max_seq) continue;
    pids[i] = (int*)malloc((size_t)T * sizeof(int));
    if (!pids[i]) continue;
    memcpy(pids[i], scratch, (size_t)T * sizeof(int));
    plen[i] = T;
    const int room = model.max_seq - T;
    const int newcap = (cfg->max_new < room) ? cfg->max_new : room;
    if (T + newcap > maxL) maxL = T + newcap;
    usable++;
  }
  if (usable == 0) {
    fprintf(stderr, "error: no task prompt fits the model's %d-position context\n", model.max_seq);
    for (int i = 0; i < n_tasks; i++) free(pids[i]);
    free(pids); free(plen);
    llama_model_free(&model); hftok_free(&tok); tasks_free(tasks, n_tasks);
    return 1;
  }

  LlamaActs acts = llama_acts_create(&model, maxL);
  int lshape[2] = {maxL, V};
  Tensor logits = tensor_create(lshape, 2), probs = tensor_create(lshape, 2);
  int dshape[2] = {1, V};
  Tensor dlogits = tensor_create(dshape, 2), dprobs = tensor_create(dshape, 2);

  const int use_lora = (cfg->lora_r > 0);
  LlamaLora lora;
  LlamaLoraActs lacts;
  LlamaDecodeLora dview;
  memset(&dview, 0, sizeof(dview));
  Tensor **params;
  int n_params;
  if (use_lora) {
    lora = llama_lora_create(&model, cfg->lora_r,
                             cfg->lora_alpha > 0.0f ? cfg->lora_alpha : (float)cfg->lora_r);
    llama_lora_init(&lora);
    lacts = llama_lora_acts_create(&model, &lora, maxL);
    dview = llama_lora_decode_view(&lora);
    params = (Tensor**)malloc((size_t)llama_lora_param_count(&lora) * sizeof(Tensor*));
    n_params = llama_lora_collect_params(&lora, params);
  } else {
    params = (Tensor**)malloc((size_t)llama_param_count(&model) * sizeof(Tensor*));
    n_params = llama_collect_params(&model, params);
  }
  Adam opt = adamw_create(params, n_params, cfg->lr, cfg->weight_decay);
  LlamaKV kv = llama_kv_create(&model);

  // Rollout storage for a whole step: groups_per_step groups of G rollouts each,
  // kept for the duration of the step so inner epochs can reuse them.
  const int cap_new = cfg->max_new;
  const int NR = cfg->groups_per_step * G;              // rollouts held per step
  const size_t flat_cap = (size_t)cfg->groups_per_step * G * cap_new;
  int **rid = (int**)malloc((size_t)NR * sizeof(int*));
  for (int i = 0; i < NR; i++) rid[i] = (int*)malloc((size_t)cap_new * sizeof(int));
  int *rlen = (int*)malloc((size_t)NR * sizeof(int));
  float *rrew = (float*)malloc((size_t)NR * sizeof(float));
  float *radv = (float*)malloc((size_t)NR * sizeof(float));
  float *flat_new = (float*)malloc(flat_cap * sizeof(float));
  float *flat_old = (float*)malloc(flat_cap * sizeof(float));
  float *flat_ref = (float*)malloc(flat_cap * sizeof(float));
  int *task_of_group = (int*)malloc((size_t)cfg->groups_per_step * sizeof(int));
  int *group_live = (int*)malloc((size_t)cfg->groups_per_step * sizeof(int));
  int *seq = (int*)malloc((size_t)maxL * sizeof(int));
  int *tgt = (int*)malloc((size_t)maxL * sizeof(int));
  float *coeff_row = (float*)malloc((size_t)maxL * sizeof(float));
  char *text = (char*)malloc(TEXT_CAP);
  char *line = (char*)malloc(LINE_CAP);

  printf("GRPO: %d usable tasks (of %d) | group %d | %d steps x %d group(s)/step x mu=%d\n",
         usable, n_tasks, G, cfg->steps, cfg->groups_per_step, cfg->inner_epochs);
  printf("  model %s (vocab %d, d_model %d, layers %d) | tok %s\n",
         cfg->model, V, model.d_model, model.n_layers, cfg->hf_tok);
  printf("  sampling: temp %.2f top-k %d top-p %.2f max-new %d | max seq %d\n",
         (double)cfg->temperature, cfg->top_k, (double)cfg->top_p, cfg->max_new, maxL);
  printf("  objective: clip-eps %.2f/%.2f | kl-beta %.3g%s | lr %g | clip %.2g\n",
         (double)cfg->clip_eps,
         (double)(cfg->clip_eps_high >= 0.0f ? cfg->clip_eps_high : cfg->clip_eps),
         (double)cfg->kl_beta,
         cfg->kl_beta > 0.0f ? " (reference = adapters off)" : " (off)",
         (double)cfg->lr, (double)cfg->grad_clip);
  printf("  advantage %s | loss-agg %s | resample flat groups up to %d time%s\n",
         cfg->std_norm ? "(r-mean)/std" : "r-mean (no std)",
         cfg->loss_agg == GRPO_AGG_TOKEN ? "token" :
         (cfg->loss_agg == GRPO_AGG_CONST ? "const" : "seq"),
         cfg->resample, cfg->resample == 1 ? "" : "s");
  if (use_lora)
    printf("  LoRA: rank %d alpha %g on Wq/Wk/Wv/Wo, base FROZEN (%d adapter tensors)\n",
           cfg->lora_r, (double)(cfg->lora_alpha > 0.0f ? cfg->lora_alpha : (float)cfg->lora_r),
           n_params);
  fflush(stdout);

  double run_reward = 0.0;
  int run_groups = 0, skipped = 0, resampled = 0, capped = 0, rollouts = 0;
  const int report = cfg->steps < 10 ? 1 : cfg->steps / 10;
  int rc = 0;

  for (int s = 0; s < cfg->steps; s++) {
    // Collect groups_per_step task groups, then take inner_epochs updates on them.
    // Rollouts are the expensive half (G x max_new decode steps each), so they are
    // sampled ONCE and reused across the inner epochs — that is what mu > 1 buys.
    double step_reward = 0.0;
    int step_groups = 0;
    const clock_t t0 = clock();

    // --- Sample and score every group ONCE (the expensive half of a step) ---
    // Every inner epoch reuses these rollouts (that reuse is the whole point of
    // mu > 1). A task is drawn per group; with --resample, a group that comes back
    // FLAT is thrown away and a fresh task drawn in its place (DAPO's dynamic
    // sampling), so the step's gradient is not diluted by groups that have none.
    // With resample == 0 the rand() sequence is exactly the original one.
    for (int g = 0; g < cfg->groups_per_step; g++) {
     for (int attempt = 0; attempt <= cfg->resample; attempt++) {
      int ti;
      do { ti = rand() % n_tasks; } while (plen[ti] == 0);
      task_of_group[g] = ti;
      const int T = plen[ti];
      int off = 0;
      for (int i = 0; i < G; i++) {
        const int gi = g * G + i;
        StopReason stop = STOP_CAP;
        rlen[gi] = rollout(&model, use_lora ? &dview : NULL, &kv, &tok,
                           pids[ti], T, cfg, &rng, &dlogits, &dprobs,
                           rid[gi], &flat_old[(size_t)g * G * cap_new + off],
                           text, TEXT_CAP, &stop);
        if (stop == STOP_CAP) capped++;
        RewardParts parts;
        rrew[gi] = reward_score(tasks[ti].prompt, text, tasks[ti].reference, &parts);
        if (cfg->verbose && i == 0) {
          one_line(text, line, LINE_CAP);
          printf("      sample r=%.2f (fmt %.2f grd %.2f mat %.2f) [%s %d tok] %s\n"
                 "        %s\n",
                 (double)rrew[gi], (double)parts.format, (double)parts.grounded,
                 (double)parts.match,
                 stop == STOP_EOS ? "eos" : (stop == STOP_TAG ? "tag" : "CAP"),
                 rlen[gi], parts.reason[0] ? parts.reason : "ok", line);
        }
        off += rlen[gi];
      }
      // A discarded flat group still counts toward the reported reward: it is what
      // the policy actually scored, and leaving it out would flatter the number.
      double gm = 0.0;
      for (int i = 0; i < G; i++) gm += rrew[g * G + i];
      step_reward += gm / G;
      step_groups++;
      rollouts += G;

      // Advantages depend on the rewards alone, so they settle here — which is
      // what lets the update below take exactly one forward per rollout.
      group_live[g] = grpo_advantages(&rrew[g * G], G, cfg, &radv[g * G]);
      if (group_live[g]) break;
      if (attempt < cfg->resample) resampled++;
      else skipped++;
     }
    }

    // Token-level aggregation normalizes by every live response token in the step.
    int step_tokens = 0;
    for (int g = 0; g < cfg->groups_per_step; g++)
      if (group_live[g])
        for (int i = 0; i < G; i++) step_tokens += rlen[g * G + i];

    for (int ep = 0; ep < cfg->inner_epochs; ep++) {
      llama_zero_grad(&model);
      if (use_lora) llama_lora_zero_grad(&lora);
      int live_groups = 0;

      for (int g = 0; g < cfg->groups_per_step; g++) {
        if (!group_live[g]) continue;         // no spread => no gradient to take
        live_groups++;
        const int ti = task_of_group[g];
        const int T = plen[ti];
        const size_t gbase = (size_t)g * G * cap_new;

        int off = 0;
        for (int i = 0; i < G; i++) {
          const int gi = g * G + i;
          const int R = rlen[gi];
          if (R == 0) continue;
          const int L = T + R;
          memcpy(seq, pids[ti], (size_t)T * sizeof(int));
          memcpy(&seq[T], rid[gi], (size_t)R * sizeof(int));
          for (int j = 0; j < L; j++) { tgt[j] = -1; coeff_row[j] = 0.0f; }
          // Row j predicts token j+1, so the rows that produced the RESPONSE are
          // T-1 .. L-2. Everything else stays masked: the prompt is context, not
          // something the policy chose.
          for (int j = T - 1; j <= L - 2; j++) tgt[j] = seq[j + 1];

          llama_acts_set_len(&acts, L);
          if (use_lora) llama_lora_acts_set_len(&lacts, L);
          set_rows(&logits, L, V);
          set_rows(&probs, L, V);

          // The reference policy — the frozen base, i.e. the same weights with the
          // adapters switched off — is evaluated FIRST, because the policy forward
          // that follows is the one whose activation cache the backward consumes.
          if (cfg->kl_beta > 0.0f && use_lora) {
            llama_forward_train_lora(&model, seq, L, &acts, &logits, NULL, NULL);
            softmax_rowwise(&logits, &probs);
            for (int t = 0; t < R; t++)
              flat_ref[gbase + off + t] = logprob_of(&probs, T - 1 + t, rid[gi][t]);
          }

          llama_forward_train_lora(&model, seq, L, &acts, &logits,
                                   use_lora ? &lora : NULL, use_lora ? &lacts : NULL);
          softmax_rowwise(&logits, &probs);
          for (int t = 0; t < R; t++)
            flat_new[gbase + off + t] = logprob_of(&probs, T - 1 + t, rid[gi][t]);

          grpo_token_coeffs(radv[gi], grpo_loss_norm(cfg, G, R, step_tokens), R,
                            &flat_new[gbase + off],
                            &flat_old[gbase + off],
                            cfg->kl_beta > 0.0f ? &flat_ref[gbase + off] : NULL,
                            cfg, &coeff_row[T - 1]);

          tensor_zero_grad(&logits);
          pg_backward(&probs, tgt, coeff_row, &logits);
          llama_backward_accum(&model, seq, L, &acts, &logits,
                               use_lora ? &lora : NULL, use_lora ? &lacts : NULL);
          off += R;
        }
      }

      if (live_groups > 0) {
        // Token-level loss is already averaged over the whole step's tokens; the
        // other two are per-group, so average the groups.
        if (live_groups > 1 && cfg->loss_agg != GRPO_AGG_TOKEN)
          scale_grads(params, n_params, 1.0f / (float)live_groups);
        if (cfg->grad_clip > 0.0f) clip_grad_norm(params, n_params, cfg->grad_clip);
        float lr = cfg->lr;
        if (cfg->warmup > 0 || cfg->lr_min >= 0.0f)
          lr = lr_cosine_warmup(cfg->lr, cfg->lr_min < 0.0f ? cfg->lr : cfg->lr_min,
                                s, cfg->warmup, cfg->steps);
        adam_set_lr(&opt, lr);
        adam_step(&opt, params, n_params);
      }
    }

    if (step_groups > 0) { run_reward += step_reward / step_groups; run_groups++; }
    if (s % report == 0 || s == cfg->steps - 1) {
      const double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
      printf("  step %5d/%d  reward %.4f  (running %.4f, %d flat group%s skipped",
             s, cfg->steps,
             step_groups > 0 ? step_reward / step_groups : 0.0,
             run_groups > 0 ? run_reward / run_groups : 0.0,
             skipped, skipped == 1 ? "" : "s");
      if (cfg->resample > 0) printf(", %d resampled", resampled);
      printf(")  %.1fs\n", secs);
      fflush(stdout);
    }
  }

  if (run_groups > 0) {
    printf("  mean reward over the run: %.4f | %d group%s carried no signal\n",
           run_reward / run_groups, skipped, skipped == 1 ? "" : "s");
    if (cfg->resample > 0)
      printf("  dynamic sampling: %d flat group%s replaced by a fresh task\n",
             resampled, resampled == 1 ? "" : "s");
    if (skipped > 0)
      printf("  NOTE: a skipped group means all %d rollouts scored identically — GRPO\n"
             "        learns from spread, so raise --temperature to explore more.\n", G);
    if (rollouts > 0 && capped * 4 > rollouts)
      printf("  WARNING: %d of %d rollouts (%.0f%%) hit the --max-new %d cap instead of\n"
             "        stopping on EOS. A truncated rollout cannot close its action block,\n"
             "        so it loses the format term for a reason that is the BUDGET's fault,\n"
             "        not the policy's. Raise --max-new.\n",
             capped, rollouts, 100.0 * capped / rollouts, cfg->max_new);
  }

  if (use_lora) {
    llama_lora_merge(&model, &lora);
    printf("  LoRA: merged adapters into the base weights\n");
  }
  if (llama_model_save(cfg->out_model, &model) != 0) {
    fprintf(stderr, "error saving model\n");
    rc = 1;
  } else {
    printf("saved GRPO model -> %s (tokenizer: %s, unchanged)\n", cfg->out_model, cfg->hf_tok);
  }

  free(text); free(line); free(coeff_row); free(seq); free(tgt);
  free(flat_new); free(flat_old); free(flat_ref);
  free(task_of_group); free(group_live);
  free(rlen); free(rrew); free(radv);
  for (int i = 0; i < NR; i++) free(rid[i]);
  free(rid);
  llama_kv_free(&kv);
  free(params);
  if (use_lora) { llama_lora_acts_free(&lacts); llama_lora_free(&lora); }
  adam_free(&opt);
  tensor_free(&logits); tensor_free(&probs);
  tensor_free(&dlogits); tensor_free(&dprobs);
  llama_acts_free(&acts);
  llama_model_free(&model);
  for (int i = 0; i < n_tasks; i++) free(pids[i]);
  free(pids); free(plen);
  hftok_free(&tok);
  tasks_free(tasks, n_tasks);
  return rc;
}
