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
#include "toolcall.h"

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
  c.lora_mlp = 0;          // attention-only adapters unless --lora-mlp
  c.lora_plus = 0.0f;      // one LR for A and B unless --lora-plus
  c.clip_eps = 0.2f;
  c.clip_eps_high = -1.0f;   // symmetric
  c.kl_beta = 0.0f;
  c.std_norm = 1;
  c.loss_agg = GRPO_AGG_SEQ;
  c.resample = 0;
  c.prio = 0;              // uniform rand() task draw unless --prio
  c.prio_alpha = 0.3f;
  c.prio_wmin = 0.05f;
  c.prio_hi = 0.9f;
  c.diag = 0;              // report lines exactly as before unless --diag / --prio
  c.reward_version = 1;    // the original reward; v2 is opt-in (--reward v2)
  c.eval_only = 0;
  c.eval_k = 1;
  c.eval_limit = 0;
  c.eval_dump = NULL;
  c.constrain = 0;         // unconstrained sampling unless --constrain (eval only)
  c.con = constrain_default_opts();
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
  grpo_token_coeffs_ex(adv, norm, len, logp_new, logp_old, logp_ref, cfg, coeff_out, NULL);
}

void grpo_token_coeffs_ex(float adv, float norm, int len, const float *logp_new,
                          const float *logp_old, const float *logp_ref,
                          const GrpoConfig *cfg, float *coeff_out, int *n_clipped) {
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
    if (clipped && n_clipped) (*n_clipped)++;
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

double grpo_row_entropy(const float *p, int V) {
  double h = 0.0;
  for (int v = 0; v < V; v++) {
    const double q = p[v];
    if (q > 0.0) h -= q * log(q);
  }
  return h;
}

// ---------------------------------------------------------------------------
// Difficulty-aware task sampling (--prio; the formula is documented in grpo.h)
// ---------------------------------------------------------------------------
static unsigned long long splitmix64(unsigned long long x);

GrpoPrio grpo_prio_create(int n_tasks, const GrpoConfig *cfg) {
  GrpoPrio p;
  memset(&p, 0, sizeof(p));
  p.n = n_tasks > 0 ? n_tasks : 0;
  p.st = (GrpoTaskStat*)calloc((size_t)(p.n > 0 ? p.n : 1), sizeof(GrpoTaskStat));
  // A stream of its own, so turning --prio on never shifts rand() or the rollout
  // RNG, and the same --seed always gives the same draws. The xor keeps it apart
  // from the eval-mode per-rollout seeds, which hash the same --seed.
  p.rng = splitmix64(0x7072696FULL ^ (unsigned long long)cfg->seed);
  p.alpha = (cfg->prio_alpha > 0.0f && cfg->prio_alpha <= 1.0f) ? cfg->prio_alpha : 0.3f;
  p.w_min = cfg->prio_wmin;
  if (!(p.w_min >= GRPO_PRIO_W_FLOOR)) p.w_min = GRPO_PRIO_W_FLOOR;
  if (p.w_min > 1.0f) p.w_min = 1.0f;
  p.hi = cfg->prio_hi;
  return p;
}

void grpo_prio_free(GrpoPrio *p) {
  free(p->st);
  p->st = NULL;
  p->n = 0;
}

float grpo_prio_weight(const GrpoPrio *p, int i) {
  const GrpoTaskStat *t = &p->st[i];
  if (t->seen == 0) return GRPO_PRIO_W_UNSEEN;
  float w = 1.0f - t->succ;                      // Kimi k1.5: weight ~ (1 - success)
  if (w < p->w_min) w = p->w_min;
  if (w > 1.0f) w = 1.0f;
  // GRESO-style: a streak of flat groups predicts another flat group. Mastered
  // (all correct) fades fast; too hard (all equally wrong) is retried, rarely.
  const int hs = t->streak_hi < GRPO_PRIO_STREAK_CAP ? t->streak_hi : GRPO_PRIO_STREAK_CAP;
  const int ls = t->streak_lo < GRPO_PRIO_STREAK_CAP ? t->streak_lo : GRPO_PRIO_STREAK_CAP;
  for (int k = 0; k < hs; k++) w *= GRPO_PRIO_DECAY_HI;
  for (int k = 0; k < ls; k++) w *= GRPO_PRIO_DECAY_LO;
  if (w < GRPO_PRIO_W_FLOOR) w = GRPO_PRIO_W_FLOOR;
  return w;
}

void grpo_prio_update(GrpoPrio *p, int i, float mean_reward, int flat) {
  if (i < 0 || i >= p->n) return;
  GrpoTaskStat *t = &p->st[i];
  float r = mean_reward;
  if (!(r >= 0.0f)) r = 0.0f;                    // also maps NaN to 0
  if (r > 1.0f) r = 1.0f;
  const float f = flat ? 1.0f : 0.0f;
  if (t->seen == 0) {
    t->succ = r;
    t->flat = f;
  } else {
    t->succ += p->alpha * (r - t->succ);
    t->flat += p->alpha * (f - t->flat);
  }
  t->seen++;
  if (flat && r >= p->hi)  { t->streak_hi++; t->streak_lo = 0; }   // all correct
  else if (flat)           { t->streak_lo++; t->streak_hi = 0; }   // all equally wrong
  else                     { t->streak_hi = 0; t->streak_lo = 0; } // spread: learnable
}

int grpo_draw_task(GrpoPrio *p, const int *plen, int n_tasks) {
  if (!p) {
    // The original draw, unchanged: the default run's rand() stream (and so its
    // task sequence and log) is exactly what it was before --prio existed.
    int ti;
    do { ti = rand() % n_tasks; } while (plen[ti] == 0);
    return ti;
  }
  double total = 0.0;
  int last = -1;
  for (int i = 0; i < n_tasks; i++)
    if (plen[i] != 0) { total += grpo_prio_weight(p, i); last = i; }
  const unsigned long long r = splitmix64(p->rng);
  p->rng += 0x9E3779B97F4A7C15ULL;
  p->draws++;
  if (last < 0) return 0;                        // grpo_run guarantees usable > 0
  const double u = (double)(r >> 11) * (1.0 / 9007199254740992.0);   // [0, 1)
  double target = u * total, acc = 0.0;
  for (int i = 0; i < n_tasks; i++) {
    if (plen[i] == 0) continue;
    acc += grpo_prio_weight(p, i);
    if (target < acc) return i;
  }
  return last;                                   // float round-off at the very top
}

void grpo_prio_summary(const GrpoPrio *p, const int *plen, int *seen, int *mastered,
                       int *too_hard) {
  int s = 0, m = 0, h = 0;
  for (int i = 0; i < p->n; i++) {
    if (plen && plen[i] == 0) continue;
    const GrpoTaskStat *t = &p->st[i];
    if (t->seen == 0) continue;
    s++;
    if (t->streak_hi > 0) m++;
    if (t->streak_lo > 0) h++;
  }
  if (seen) *seen = s;
  if (mastered) *mastered = m;
  if (too_hard) *too_hard = h;
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

// Prefill: decode the prompt into `kv` from position 0, token by token, through
// the CURRENT adapters, leaving the last prompt position's logits in `logits`.
// Token-by-token (not one batched llama_forward_train_lora) on purpose: the
// batched forward is only ~1e-3-close to decode (selftest [26]), and the sampled
// rollouts must stay bit-identical to what a per-rollout decode would draw.
// Only the last position's logits are ever read, so the earlier steps pass NULL
// and skip the final norm + lm_head (576 x 49152, the largest matmul of a 135M
// step); the KV cache they write is unchanged.
static void prefill_prompt(const LlamaModel *m, const LlamaDecodeLora *lo, LlamaKV *kv,
                           const int *prompt_ids, int n_prompt, Tensor *logits) {
  kv->len = 0;
  for (int i = 0; i < n_prompt; i++)
    llama_decode_step_lora(m, kv, prompt_ids[i], (i == n_prompt - 1) ? logits : NULL, lo);
}

// One rollout: start from the prefilled prompt, then sample up to max_new tokens
// from the CURRENT policy (adapters included). Records the token ids and the
// log-probability the policy assigned to each sampled token — the pi_old the
// importance ratio is measured against.
//
// Shared prefill: every rollout of a group starts from the SAME prompt, and the
// prompt's KV entries and last logits depend only on the prompt and the weights
// (no sampling), so grpo_run prefills it ONCE per group into a template cache
// (`tmpl`, `tmpl_logits`) and each rollout forks it with llama_kv_copy. With
// ~110-token prompts and ~40-token responses that removes most of the decode
// work. The copy is bit-exact, and the prefill draws nothing from the RNG, so
// the sampled tokens are identical to prefilling per rollout (selftest [37]).
// tmpl == NULL prefills here instead (the old per-rollout path, kept for [37]).
//
typedef enum { STOP_CAP = 0, STOP_EOS = 1, STOP_TAG = 2, STOP_ERR = 3 } StopReason;

// Constrained decoding for one rollout (grpo --eval-only --constrain): the task's
// span index, the vocabulary view, and counters for the eval summary. NULL => the
// plain sampler, exactly as before.
typedef struct {
  const Constrain *c;
  const ConVocab *v;
  long picks, full_scans, errors;
} ConCtx;

// Test hooks only (grpo_debug_rollout_group): with no tokenizer, sampling this
// token ends a rollout as EOS would. -1 (always, outside that hook) = never.
static int S_debug_stop_tok = -1;

// Returns the number of sampled tokens, writes the detokenized text to `text`,
// and reports HOW the rollout ended in *stop. tok == NULL (tests only) disables
// the EOS / closing-tag stops and leaves `text` empty.
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
                   const LlamaKV *tmpl, const Tensor *tmpl_logits,
                   const HFTok *tok, const int *prompt_ids, int n_prompt,
                   const GrpoConfig *cfg, Rng *rng, Tensor *logits, Tensor *probs,
                   int *out_ids, float *out_logp, char *text, int text_cap,
                   StopReason *stop, ConCtx *cx) {
  if (tmpl) {
    llama_kv_copy(kv, tmpl);
    memcpy(logits->data, tmpl_logits->data, logits->size * sizeof(float));
  } else {
    prefill_prompt(m, lo, kv, prompt_ids, n_prompt, logits);
  }

  int n = 0;
  *stop = STOP_CAP;
  text[0] = '\0';
  const int room = m->max_seq - n_prompt;
  const int cap = (cfg->max_new < room) ? cfg->max_new : room;
  ConState cst;
  if (cx) constrain_init(cx->c, &cst);
  while (n < cap) {
    // The log-prob is taken from the UNTEMPERED softmax: temperature/top-k/top-p
    // shape the exploration, but the policy whose ratio we measure — and whose
    // gradient we take — is the model's own distribution.
    softmax_rowwise(logits, probs);
    int t;
    if (cx) {
      // Constrained: the highest-ranked candidate that keeps the output a valid
      // action prefix (constrain.h); the budget is the tokens this rollout has left.
      int full = 0;
      t = constrain_pick(cx->c, cx->v, &cst, logits->data, cfg->temperature, cfg->top_k,
                         cfg->top_p, rng, cap - n, &full);
      cx->picks++;
      cx->full_scans += full;
      if (t < 0) {
        fprintf(stderr, "grpo: constrained decode: no token of the vocabulary continues a valid action\n");
        cx->errors++;
        *stop = STOP_ERR;
        break;
      }
    } else {
      t = sample_token(logits, 0, cfg->temperature, cfg->top_k, cfg->top_p, rng);
    }
    out_logp[n] = logprob_of(probs, 0, t);
    out_ids[n++] = t;
    if (tok) {
      if (hftok_is_eog(tok, t)) { *stop = STOP_EOS; break; }
      // Re-decode the whole response each step rather than tracking a partial
      // decode: a BPE piece can straddle the tag, and n is at most --max-new, so
      // this is nothing beside a 30-layer forward.
      hftok_decode(tok, out_ids, n, text, text_cap);
      if (strstr(text, "</tool>") || strstr(text, "</done>")) { *stop = STOP_TAG; break; }
    } else if (t == S_debug_stop_tok) { *stop = STOP_EOS; break; }   // tests only
    if (cx && constrain_complete(&cst)) { *stop = STOP_TAG; break; }
    llama_decode_step_lora(m, kv, t, logits, lo);
  }
  if (tok) hftok_decode(tok, out_ids, n, text, text_cap);
  return n;
}

// Batched rollouts (--batch-rollouts): the G rollouts of one group, decoded in
// LOCKSTEP. Each round, every live rollout samples its next token from its own
// logits row with its OWN rng (rngs[i]) and runs exactly rollout()'s stop logic
// (EOS, closing tag, cap); the survivors are then decoded together by ONE
// llama_decode_batch_lora call, whose row for rollout i is bit-identical to the
// llama_decode_step_lora rollout() would run. The weights are read once per
// round instead of once per rollout -- the whole point: a 135M decode step is
// bound by streaming ~540 MB of weights, not by arithmetic.
//
// So with the same rngs[i], rollout i here produces exactly the ids, log-probs,
// text and stop reason that rollout() produces for it alone (selftest [48]). What
// batching CANNOT preserve is a SHARED rng: interleaving the draws of G rollouts
// reorders it. Hence the callers give every rollout its own stream.
//
// One difference in work, none in output: rollout() decodes the last sampled
// token of a rollout that hits the cap (its logits are never read); this skips it.
//
// kvs: G caches (each able to hold n_prompt + cap positions); blogits: (G, V)
// scratch; probs: (1, V). Rollout i's outputs go to out_ids[i], out_logp[i],
// text[i]. Returns the number of batched decode calls made.
static long rollout_batch(const LlamaModel *m, const LlamaDecodeLora *lo, LlamaKV *kvs, int G,
                          const LlamaKV *tmpl, const Tensor *tmpl_logits, const HFTok *tok,
                          int n_prompt, const GrpoConfig *cfg, Rng *rngs, Tensor *blogits,
                          Tensor *probs, int *const *out_ids, float *const *out_logp,
                          char *const *text, int text_cap, StopReason *stop, int *out_len,
                          ConCtx *cx) {
  const int V = m->vocab;
  // Constrained decoding (eval only): the per-task index and vocabulary view are
  // read-only and shared; the decoder state is per rollout, and each pick draws
  // from that rollout's own rng -- exactly what rollout() does for it alone.
  ConState *cst = cx ? (ConState*)malloc((size_t)G * sizeof(ConState)) : NULL;
  int *live = (int*)malloc((size_t)G * sizeof(int));
  int *row = (int*)malloc((size_t)G * sizeof(int));      // rollout i's row in blogits
  int *dtok = (int*)malloc((size_t)G * sizeof(int));
  LlamaKV **dkv = (LlamaKV**)malloc((size_t)G * sizeof(LlamaKV*));
  const int room = m->max_seq - n_prompt;
  const int cap = (cfg->max_new < room) ? cfg->max_new : room;
  for (int i = 0; i < G; i++) {
    llama_kv_copy(&kvs[i], tmpl);
    memcpy(&blogits->data[(size_t)i * V], tmpl_logits->data, (size_t)V * sizeof(float));
    out_len[i] = 0; stop[i] = STOP_CAP; text[i][0] = '\0';
    row[i] = i; live[i] = (cap > 0);
    if (cx) constrain_init(cx->c, &cst[i]);
  }
  long calls = 0;
  for (;;) {
    int nd = 0;
    for (int i = 0; i < G; i++) {
      if (!live[i]) continue;
      // Row i as a (1, V) tensor: softmax_rowwise / logprob_of see exactly the
      // single-rollout (1, V) logits, so the log-prob is bit-identical too.
      int vs[2] = {1, V};
      Tensor lr = *blogits;
      lr.data = &blogits->data[(size_t)row[i] * V]; lr.grad = NULL;
      lr.shape = vs; lr.ndim = 2; lr.size = (size_t)V;
      softmax_rowwise(&lr, probs);
      const int n = out_len[i];
      int t;
      if (cx) {
        int full = 0;
        t = constrain_pick(cx->c, cx->v, &cst[i], lr.data, cfg->temperature, cfg->top_k,
                           cfg->top_p, &rngs[i], cap - n, &full);
        cx->picks++;
        cx->full_scans += full;
        if (t < 0) {
          fprintf(stderr, "grpo: constrained decode: no token of the vocabulary continues a valid action\n");
          cx->errors++;
          stop[i] = STOP_ERR; live[i] = 0; continue;
        }
      } else {
        t = sample_token(&lr, 0, cfg->temperature, cfg->top_k, cfg->top_p, &rngs[i]);
      }
      out_logp[i][n] = logprob_of(probs, 0, t);
      out_ids[i][n] = t;
      out_len[i] = n + 1;
      if (tok) {
        if (hftok_is_eog(tok, t)) { stop[i] = STOP_EOS; live[i] = 0; continue; }
        hftok_decode(tok, out_ids[i], n + 1, text[i], text_cap);
        if (strstr(text[i], "</tool>") || strstr(text[i], "</done>")) {
          stop[i] = STOP_TAG; live[i] = 0; continue;
        }
      } else if (t == S_debug_stop_tok) { stop[i] = STOP_EOS; live[i] = 0; continue; }
      if (cx && constrain_complete(&cst[i])) { stop[i] = STOP_TAG; live[i] = 0; continue; }
      if (n + 1 >= cap) { live[i] = 0; continue; }       // STOP_CAP: nothing left to decode
      dkv[nd] = &kvs[i]; dtok[nd] = t; row[i] = nd; nd++;
    }
    if (nd == 0) break;
    // The survivors' logits land in rows 0..nd-1, in rollout order. Rows are
    // only overwritten here, after every live rollout has sampled from its row.
    int bs[2] = {nd, V};
    Tensor bl = *blogits;
    bl.grad = NULL; bl.shape = bs; bl.size = (size_t)nd * V;
    llama_decode_batch_lora(m, dkv, dtok, nd, &bl, lo);
    calls++;
  }
  if (tok)
    for (int i = 0; i < G; i++) hftok_decode(tok, out_ids[i], out_len[i], text[i], text_cap);
  free(live); free(row); free(dtok); free(dkv); free(cst);
  return calls;
}

// Scratch for rollout_batch: G caches sized for prompt + response only (the
// template forks into them: llama_kv_copy), (G, V) logits, G text buffers.
typedef struct {
  int G;
  LlamaKV *kv;
  Tensor logits;
  Rng *rng;
  char **text;
  int **ids;          // G x cap_new
  float **logp;       // G x cap_new
  StopReason *stop;
  int *len;
} RolloutBatch;

static RolloutBatch rollout_batch_create(const LlamaModel *m, int G, int max_len,
                                         int cap_new, int text_cap) {
  RolloutBatch b;
  b.G = G;
  b.kv = (LlamaKV*)malloc((size_t)G * sizeof(LlamaKV));
  for (int i = 0; i < G; i++) b.kv[i] = llama_kv_create_len(m, max_len);
  int s[2] = {G, m->vocab};
  b.logits = tensor_create_nograd(s, 2);
  b.rng = (Rng*)malloc((size_t)G * sizeof(Rng));
  b.text = (char**)malloc((size_t)G * sizeof(char*));
  b.ids = (int**)malloc((size_t)G * sizeof(int*));
  b.logp = (float**)malloc((size_t)G * sizeof(float*));
  for (int i = 0; i < G; i++) {
    b.text[i] = (char*)malloc((size_t)text_cap);
    b.ids[i] = (int*)malloc((size_t)cap_new * sizeof(int));
    b.logp[i] = (float*)malloc((size_t)cap_new * sizeof(float));
  }
  b.stop = (StopReason*)malloc((size_t)G * sizeof(StopReason));
  b.len = (int*)malloc((size_t)G * sizeof(int));
  return b;
}

static void rollout_batch_free(RolloutBatch *b) {
  for (int i = 0; i < b->G; i++) {
    llama_kv_free(&b->kv[i]);
    free(b->text[i]); free(b->ids[i]); free(b->logp[i]);
  }
  free(b->kv); free(b->rng); free(b->text); free(b->ids); free(b->logp);
  free(b->stop); free(b->len);
  tensor_free(&b->logits);
}

static long rollout_batch_run(const LlamaModel *m, const LlamaDecodeLora *lo, RolloutBatch *b,
                              const LlamaKV *tmpl, const Tensor *tmpl_logits, const HFTok *tok,
                              int n_prompt, const GrpoConfig *cfg, Tensor *probs, int text_cap,
                              ConCtx *cx) {
  return rollout_batch(m, lo, b->kv, b->G, tmpl, tmpl_logits, tok, n_prompt, cfg, b->rng,
                       &b->logits, probs, b->ids, b->logp, b->text, text_cap, b->stop, b->len,
                       cx);
}

long grpo_debug_rollout_group(const LlamaModel *m, const LlamaDecodeLora *lo,
                              const int *prompt_ids, int n_prompt, const GrpoConfig *cfg,
                              const unsigned long long *seeds, int batched, int stop_tok,
                              int *out_ids, float *out_logp, int *out_len) {
  const int G = cfg->group, V = m->vocab;
  int ds[2] = {1, V};
  Tensor logits = tensor_create(ds, 2), probs = tensor_create(ds, 2);
  Tensor plogits = tensor_create(ds, 2);
  LlamaKV pkv = llama_kv_create_len(m, n_prompt);
  S_debug_stop_tok = stop_tok;
  prefill_prompt(m, lo, &pkv, prompt_ids, n_prompt, &plogits);
  long calls = 0;
  if (batched) {
    RolloutBatch b = rollout_batch_create(m, G, n_prompt + cfg->max_new, cfg->max_new, 1);
    for (int i = 0; i < G; i++) b.rng[i] = rng_create(seeds[i]);
    calls = rollout_batch_run(m, lo, &b, &pkv, &plogits, NULL, n_prompt, cfg, &probs, 1, NULL);
    for (int i = 0; i < G; i++) {
      out_len[i] = b.len[i];
      memcpy(&out_ids[(size_t)i * cfg->max_new], b.ids[i], (size_t)b.len[i] * sizeof(int));
      memcpy(&out_logp[(size_t)i * cfg->max_new], b.logp[i], (size_t)b.len[i] * sizeof(float));
    }
    rollout_batch_free(&b);
  } else {
    LlamaKV kv = llama_kv_create(m);
    char text[1];
    for (int i = 0; i < G; i++) {
      StopReason stop;
      Rng rng = rng_create(seeds[i]);
      out_len[i] = rollout(m, lo, &kv, &pkv, &plogits, NULL, prompt_ids, n_prompt, cfg, &rng,
                           &logits, &probs, &out_ids[(size_t)i * cfg->max_new],
                           &out_logp[(size_t)i * cfg->max_new], text, 1, &stop, NULL);
    }
    llama_kv_free(&kv);
  }
  S_debug_stop_tok = -1;
  llama_kv_free(&pkv);
  tensor_free(&logits); tensor_free(&probs); tensor_free(&plogits);
  return calls;
}

long grpo_debug_sample_group(const LlamaModel *m, const LlamaDecodeLora *lo,
                             const int *prompt_ids, int n_prompt, const GrpoConfig *cfg,
                             int shared_prefill, Rng *rng,
                             int *out_ids, float *out_logp, int *out_len) {
  const int G = cfg->group, V = m->vocab;
  int ds[2] = {1, V};
  Tensor logits = tensor_create(ds, 2), probs = tensor_create(ds, 2);
  Tensor plogits = tensor_create(ds, 2);
  LlamaKV kv = llama_kv_create(m), pkv = llama_kv_create_len(m, n_prompt);
  char text[1];
  long prefill_steps = 0;
  if (shared_prefill) {
    prefill_prompt(m, lo, &pkv, prompt_ids, n_prompt, &plogits);
    prefill_steps += n_prompt;
  }
  for (int i = 0; i < G; i++) {
    StopReason stop;
    out_len[i] = rollout(m, lo, &kv, shared_prefill ? &pkv : NULL, &plogits, NULL,
                         prompt_ids, n_prompt, cfg, rng, &logits, &probs,
                         &out_ids[(size_t)i * cfg->max_new],
                         &out_logp[(size_t)i * cfg->max_new], text, 1, &stop, NULL);
    if (!shared_prefill) prefill_steps += n_prompt;
  }
  llama_kv_free(&kv); llama_kv_free(&pkv);
  tensor_free(&logits); tensor_free(&probs); tensor_free(&plogits);
  return prefill_steps;
}

// Copy up to `cap` bytes of `s` onto one line for the verbose log: control
// characters (a write_file body is full of newlines) become spaces.
static void one_line(const char *s, char *out, int cap) {
  int n = 0;
  for (const char *p = s; *p && n < cap - 1; p++)
    out[n++] = ((unsigned char)*p < 0x20) ? ' ' : *p;
  out[n] = '\0';
}

// The reward-hacking monitor: what KIND of action the rollouts took, and the mean
// of each reward term. A policy that has found a cheap way to score (answer every
// task with <done>, emit prose, ramble to the cap) shows up here as a share that
// climbs while the match term stays flat — long before the total looks wrong.
// Pure bookkeeping: it never feeds back into training.
typedef struct {
  int n, capped;
  int act[REWARD_ACT_COUNT];
  double fmt, grd, mat;
} RewardMonitor;

static void mon_add(RewardMonitor *m, const RewardParts *p, int capped) {
  m->n++;
  m->capped += capped;
  if (p->action >= 0 && p->action < REWARD_ACT_COUNT) m->act[p->action]++;
  m->fmt += p->format;
  m->grd += p->grounded;
  m->mat += p->match;
}

static void mon_print(const char *lead, const RewardMonitor *m) {
  if (m->n == 0) return;
  printf("%sactions:", lead);
  for (int a = 0; a < REWARD_ACT_COUNT; a++)
    printf(" %s %.0f%%", reward_action_name(a), 100.0 * m->act[a] / m->n);
  printf(" | capped %.0f%% | mean fmt %.3f grd %.3f mat %.3f (%d rollouts)\n",
         100.0 * m->capped / m->n, m->fmt / m->n, m->grd / m->n, m->mat / m->n, m->n);
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
  // Under LoRA the base is frozen: it gets gradient arrays only on its 1-D
  // gains/biases (llama_model_create_train_frozen), not ~650 MB of matrices.
  LlamaModel model = (cfg->lora_r > 0 ? llama_model_create_train_frozen : llama_model_create_train)(
      loaded.vocab, loaded.d_model, loaded.n_layers, loaded.n_heads, loaded.n_kv,
      loaded.head_dim, loaded.d_ff, loaded.max_seq, loaded.rope_base, loaded.eps);
  llama_model_set_qk_norm(&model, loaded.qk_norm);   // same parameter set as the file
  llama_model_set_tied(&model, loaded.tied);        // tied head: wte only, as in the file
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
  // --prio draws from the per-task statistics; --diag (implied by --prio) only
  // keeps them for the coverage summary. Neither touches rand() or `rng`.
  const int diag = cfg->diag || cfg->prio;
  GrpoPrio prio;
  memset(&prio, 0, sizeof(prio));
  if (diag) prio = grpo_prio_create(n_tasks, cfg);

  // Tokenize the prompts once. Tasks whose prompt leaves no room for a rollout
  // are dropped here rather than silently producing empty responses.
  int **pids = (int**)malloc((size_t)n_tasks * sizeof(int*));
  int *plen = (int*)malloc((size_t)n_tasks * sizeof(int));
  int usable = 0, maxL = 0, maxP = 0;
  int scratch[MAX_PROMPT_IDS];
  for (int i = 0; i < n_tasks; i++) {
    const int T = hftok_encode(&tok, tasks[i].prompt, scratch, MAX_PROMPT_IDS);
    pids[i] = NULL; plen[i] = 0;
    if (T < 1 || T + 2 >= model.max_seq) continue;
    pids[i] = (int*)malloc((size_t)T * sizeof(int));
    if (!pids[i]) continue;
    memcpy(pids[i], scratch, (size_t)T * sizeof(int));
    plen[i] = T;
    if (T > maxP) maxP = T;
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
  // The update takes logits for the RESPONSE rows only (compact lm_head, see
  // llama_forward_train_rows), and a response is at most max_new tokens -- so
  // the logits buffers hold that many rows, not max-seq. probs needs no grad.
  int lshape[2] = {cfg->max_new < maxL ? cfg->max_new : maxL, V};
  Tensor logits = tensor_create(lshape, 2), probs = tensor_create_nograd(lshape, 2);
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
    lora = llama_lora_create_ex(&model, cfg->lora_r,
                                cfg->lora_alpha > 0.0f ? cfg->lora_alpha : (float)cfg->lora_r,
                                cfg->lora_mlp);
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
  if (use_lora && cfg->lora_plus > 0.0f) {       // LoRA+: B gets lora_plus x the LR
    float *mult = (float*)malloc((size_t)n_params * sizeof(float));
    llama_lora_plus_mult(&lora, cfg->lora_plus, mult);
    adam_set_lr_mult(&opt, mult, n_params);
    free(mult);
  }
  // --batch-rollouts: G caches of maxL positions (prompt + response) replace the
  // one max_seq cache, which then shrinks to a 1-position stub.
  const int batched = cfg->batch_rollouts;
  LlamaKV kv = batched ? llama_kv_create_len(&model, 1) : llama_kv_create(&model);
  // The group's prompt, prefilled ONCE (with the current adapters) and forked
  // into every rollout of the group — see rollout(). Sized for the longest prompt
  // only, so it costs maxP positions of KV rather than a second max_seq cache.
  LlamaKV pkv = llama_kv_create_len(&model, maxP);
  Tensor plogits = tensor_create(dshape, 2);
  RolloutBatch rb;
  memset(&rb, 0, sizeof(rb));
  if (batched) rb = rollout_batch_create(&model, G, maxL, cfg->max_new, TEXT_CAP);
  long batch_group = 0;                     // groups sampled so far (seeds the streams)

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
  int *tgt = (int*)malloc((size_t)maxL * sizeof(int));        // compact: one per response row
  int *lm_rows = (int*)malloc((size_t)maxL * sizeof(int));
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
  if (cfg->prio)
    printf("  prio: w = max(%.2f, clamp(1 - succ_ema, %.2f, 1) x %.1f^mastered_streak x %.1f^too_hard_streak),"
           " unseen %.1f | ema %.2f | mastered = flat at >= %.2f\n",
           (double)GRPO_PRIO_W_FLOOR, (double)prio.w_min, (double)GRPO_PRIO_DECAY_HI,
           (double)GRPO_PRIO_DECAY_LO, (double)GRPO_PRIO_W_UNSEEN, (double)prio.alpha,
           (double)prio.hi);
  else if (diag)
    printf("  diagnostics: on (task draw: uniform, as default)\n");
  if (batched)
    printf("  rollouts: BATCHED -- the %d of a group decoded in lockstep; rollout i of group g"
           " samples from its own RNG = hash(seed, g, i)\n", G);
  if (use_lora) {
    printf("  LoRA: rank %d alpha %g on Wq/Wk/Wv/Wo%s, base FROZEN (%d adapter tensors)\n",
           cfg->lora_r, (double)(cfg->lora_alpha > 0.0f ? cfg->lora_alpha : (float)cfg->lora_r),
           cfg->lora_mlp ? "+Wgate/Wup/Wdown" : "", n_params);
    if (cfg->lora_plus > 0.0f)
      printf("  LoRA+: lr(B) = %g x lr(A)\n", (double)cfg->lora_plus);
  }
  fflush(stdout);

  if (cfg->reward_version == 2)
    printf("  reward: v2 = format * (0.30 grounded + 0.70 match), capped rollout => 0\n");
  RewardMonitor mon_win, mon_run;               // since the last report / whole run
  memset(&mon_win, 0, sizeof(mon_win));
  memset(&mon_run, 0, sizeof(mon_run));

  double run_reward = 0.0;
  int run_groups = 0, skipped = 0, resampled = 0, capped = 0, rollouts = 0;
  const int report = cfg->steps < 10 ? 1 : cfg->steps / 10;
  int rc = 0;
  // Diagnostics (--diag / --prio): window = since the last report line.
  double win_ent = 0.0, run_ent = 0.0;          // summed per-token entropy (nats)
  long win_ent_tok = 0, run_ent_tok = 0;
  long win_clip = 0, win_ctok = 0, run_clip = 0, run_ctok = 0;
  int win_sampled = 0, win_live = 0, run_sampled = 0, run_live = 0;
  int win_slots = 0, win_slots_live = 0;
  long adam_updates = 0;

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
      ti = grpo_draw_task(cfg->prio ? &prio : NULL, plen, n_tasks);
      task_of_group[g] = ti;
      const int T = plen[ti];
      // The adapters only change in the update phase below, so one prefill serves
      // all G rollouts of this group. It draws nothing from the RNG.
      prefill_prompt(&model, use_lora ? &dview : NULL, &pkv, pids[ti], T, &plogits);
      if (batched) {
        // --batch-rollouts: all G in lockstep, rollout i on its own stream.
        for (int i = 0; i < G; i++)
          rb.rng[i] = rng_create(grpo_batch_rollout_seed(cfg->seed, batch_group, i));
        batch_group++;
        rollout_batch_run(&model, use_lora ? &dview : NULL, &rb, &pkv, &plogits, &tok, T,
                          cfg, &dprobs, TEXT_CAP, NULL);
      }
      int off = 0;
      for (int i = 0; i < G; i++) {
        const int gi = g * G + i;
        StopReason stop = STOP_CAP;
        const char *rtext = text;
        if (batched) {                     // copy out in the unbatched layout
          rlen[gi] = rb.len[i];
          stop = rb.stop[i];
          rtext = rb.text[i];
          memcpy(rid[gi], rb.ids[i], (size_t)rb.len[i] * sizeof(int));
          memcpy(&flat_old[(size_t)g * G * cap_new + off], rb.logp[i],
                 (size_t)rb.len[i] * sizeof(float));
        } else {
          rlen[gi] = rollout(&model, use_lora ? &dview : NULL, &kv, &pkv, &plogits, &tok,
                             pids[ti], T, cfg, &rng, &dlogits, &dprobs,
                             rid[gi], &flat_old[(size_t)g * G * cap_new + off],
                             text, TEXT_CAP, &stop, NULL);
        }
        if (stop == STOP_CAP) capped++;
        RewardParts parts;
        rrew[gi] = reward_score_version(cfg->reward_version, tasks[ti].prompt, rtext,
                                        tasks[ti].reference, stop == STOP_CAP, &parts);
        mon_add(&mon_win, &parts, stop == STOP_CAP);
        mon_add(&mon_run, &parts, stop == STOP_CAP);
        if (cfg->verbose && i == 0) {
          one_line(rtext, line, LINE_CAP);
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
      if (diag) {
        grpo_prio_update(&prio, ti, (float)(gm / G), !group_live[g]);
        win_sampled++;
        win_live += group_live[g];
      }
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
          // Row j predicts token j+1, so the rows that produced the RESPONSE are
          // T-1 .. L-2. Everything else is masked -- the prompt is context, not
          // something the policy chose -- so only those R rows get logits at all
          // (compact lm_head): logits/probs row t is position T-1+t, and tgt /
          // coeff_row are indexed the same way. Bit-identical to computing every
          // row and masking (selftest [43]).
          for (int t = 0; t < R; t++) { lm_rows[t] = T - 1 + t; tgt[t] = seq[T + t]; }

          llama_acts_set_len(&acts, L);
          if (use_lora) llama_lora_acts_set_len(&lacts, L);
          set_rows(&logits, R, V);
          set_rows(&probs, R, V);

          // The reference policy — the frozen base, i.e. the same weights with the
          // adapters switched off — is evaluated FIRST, because the policy forward
          // that follows is the one whose activation cache the backward consumes.
          if (cfg->kl_beta > 0.0f && use_lora) {
            llama_forward_train_rows(&model, seq, L, &acts, &logits, NULL, NULL, lm_rows, R);
            softmax_rowwise(&logits, &probs);
            for (int t = 0; t < R; t++)
              flat_ref[gbase + off + t] = logprob_of(&probs, t, rid[gi][t]);
          }

          llama_forward_train_rows(&model, seq, L, &acts, &logits,
                                   use_lora ? &lora : NULL, use_lora ? &lacts : NULL, lm_rows, R);
          softmax_rowwise(&logits, &probs);
          for (int t = 0; t < R; t++)
            flat_new[gbase + off + t] = logprob_of(&probs, t, rid[gi][t]);
          // Policy entropy over the sampled response tokens, read off the softmax
          // this forward already built (no extra forward). First epoch only, so
          // mu > 1 does not count the same tokens twice. (Compact rows: probs row t
          // IS response token t -- the same row values as the all-rows path.)
          if (diag && ep == 0) {
            for (int t = 0; t < R; t++)
              win_ent += grpo_row_entropy(&probs.data[(size_t)t * V], V);
            win_ent_tok += R;
          }

          int nclip = 0;
          grpo_token_coeffs_ex(radv[gi], grpo_loss_norm(cfg, G, R, step_tokens), R,
                               &flat_new[gbase + off],
                               &flat_old[gbase + off],
                               cfg->kl_beta > 0.0f ? &flat_ref[gbase + off] : NULL,
                               cfg, coeff_row, &nclip);
          win_clip += nclip;
          win_ctok += R;

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
        adam_updates++;
      }
    }

    if (step_groups > 0) { run_reward += step_reward / step_groups; run_groups++; }
    win_slots += cfg->groups_per_step;
    for (int g = 0; g < cfg->groups_per_step; g++) win_slots_live += group_live[g];
    if (s % report == 0 || s == cfg->steps - 1) {
      const double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
      printf("  step %5d/%d  reward %.4f  (running %.4f, %d flat group%s skipped",
             s, cfg->steps,
             step_groups > 0 ? step_reward / step_groups : 0.0,
             run_groups > 0 ? run_reward / run_groups : 0.0,
             skipped, skipped == 1 ? "" : "s");
      if (cfg->resample > 0) printf(", %d resampled", resampled);
      printf(")  %.1fs\n", secs);
      mon_print("           ", &mon_win);
      memset(&mon_win, 0, sizeof(mon_win));
      if (diag) {
        printf("           diag: entropy ");
        if (win_ent_tok > 0) printf("%.4f nats/tok (%ld tok)", win_ent / win_ent_tok, win_ent_tok);
        else printf("n/a (no live group)");
        printf(" | live groups %d/%d sampled (%.0f%%), %d/%d step slots"
               " | clip frac %.4f (%ld/%ld tok) | adam updates %ld\n",
               win_live, win_sampled, win_sampled > 0 ? 100.0 * win_live / win_sampled : 0.0,
               win_slots_live, win_slots,
               win_ctok > 0 ? (double)win_clip / (double)win_ctok : 0.0, win_clip, win_ctok,
               adam_updates);
        int seen = 0, mast = 0, hard = 0;
        grpo_prio_summary(&prio, plen, &seen, &mast, &hard);
        printf("           tasks: seen %d/%d | mastered %d | too-hard %d", seen, usable, mast, hard);
        if (cfg->prio) {
          double wmin = 2.0, wmax = 0.0, wsum = 0.0;
          for (int i = 0; i < n_tasks; i++) {
            if (plen[i] == 0) continue;
            const double w = grpo_prio_weight(&prio, i);
            if (w < wmin) wmin = w;
            if (w > wmax) wmax = w;
            wsum += w;
          }
          printf(" | prio weight min %.3f mean %.3f max %.3f", wmin, wsum / usable, wmax);
        }
        printf("\n");
        run_ent += win_ent; run_ent_tok += win_ent_tok;
        run_clip += win_clip; run_ctok += win_ctok;
        run_sampled += win_sampled; run_live += win_live;
        win_ent = 0.0; win_ent_tok = 0; win_clip = 0; win_ctok = 0;
        win_sampled = 0; win_live = 0; win_slots = 0; win_slots_live = 0;
      }
      fflush(stdout);
    }
  }

  if (run_groups > 0) {
    printf("  mean reward over the run: %.4f | %d group%s carried no signal\n",
           run_reward / run_groups, skipped, skipped == 1 ? "" : "s");
    mon_print("  run mix: ", &mon_run);
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
  if (diag) {
    int seen = 0, mast = 0, hard = 0;
    grpo_prio_summary(&prio, plen, &seen, &mast, &hard);
    printf("  task coverage: %d distinct of %d usable tasks seen | mastered %d (last group flat,"
           " all >= %.2f) | too hard %d (last group flat below %.2f)\n",
           seen, usable, mast, (double)prio.hi, hard, (double)prio.hi);
    printf("  diagnostics over the run: live groups %d/%d sampled (%.0f%%) | adam updates %ld"
           " | clip frac %.4f (%ld/%ld tok) | mean entropy ",
           run_live, run_sampled, run_sampled > 0 ? 100.0 * run_live / run_sampled : 0.0,
           adam_updates, run_ctok > 0 ? (double)run_clip / (double)run_ctok : 0.0,
           run_clip, run_ctok);
    if (run_ent_tok > 0) printf("%.4f nats/tok\n", run_ent / run_ent_tok);
    else printf("n/a\n");
    if (cfg->prio) printf("  prio: %ld weighted task draws\n", prio.draws);
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

  free(text); free(line); free(coeff_row); free(seq); free(tgt); free(lm_rows);
  free(flat_new); free(flat_old); free(flat_ref);
  free(task_of_group); free(group_live);
  if (diag) grpo_prio_free(&prio);
  free(rlen); free(rrew); free(radv);
  for (int i = 0; i < NR; i++) free(rid[i]);
  free(rid);
  llama_kv_free(&kv); llama_kv_free(&pkv);
  tensor_free(&plogits);
  free(params);
  if (use_lora) { llama_lora_acts_free(&lacts); llama_lora_free(&lora); }
  adam_free(&opt);
  if (batched) rollout_batch_free(&rb);
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

// ---------------------------------------------------------------------------
// Evaluation (grpo --eval-only; see grpo.h)
// ---------------------------------------------------------------------------

static unsigned long long splitmix64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

unsigned long long grpo_eval_rollout_seed(unsigned seed, int task, int k) {
  unsigned long long h = splitmix64((unsigned long long)seed);
  h = splitmix64(h ^ (unsigned long long)(unsigned)task);
  h = splitmix64(h ^ ((unsigned long long)(unsigned)k << 32));
  return h ? h : 1ULL;
}

unsigned long long grpo_batch_rollout_seed(unsigned seed, long group, int i) {
  // 'BTCH' domain tag: never the eval stream of the same (seed, task, k).
  unsigned long long h = splitmix64(0x4254434800000000ULL ^ (unsigned long long)seed);
  h = splitmix64(h ^ (unsigned long long)group);
  h = splitmix64(h ^ ((unsigned long long)(unsigned)i << 32));
  return h ? h : 1ULL;
}

// One evaluation rollout of task `task`, draw `kk`, from the prefilled template
// (pkv, plogits). Its RNG is created HERE from (seed, task, kk) and dies with the
// rollout, so no other rollout — of this checkpoint or any other — can shift it:
// a checkpoint that answers task 3 in 12 tokens instead of 40 consumes 28 fewer
// draws, and with one shared RNG every later task would then be sampled with
// different noise. Here task 4 is sampled with the same noise either way.
static int eval_rollout(const LlamaModel *m, LlamaKV *kv, const LlamaKV *pkv,
                        const Tensor *plogits, const HFTok *tok, const int *pids, int T,
                        int task, int kk, const GrpoConfig *cfg, Tensor *logits,
                        Tensor *probs, int *ids, float *logp, char *text, int text_cap,
                        StopReason *stop, ConCtx *cx) {
  Rng rng = rng_create(grpo_eval_rollout_seed(cfg->seed, task, kk));
  return rollout(m, NULL, kv, pkv, plogits, tok, pids, T, cfg, &rng, logits, probs,
                 ids, logp, text, text_cap, stop, cx);
}

void grpo_debug_eval_task(const LlamaModel *m, const int *prompt_ids, int n_prompt,
                          int task, const GrpoConfig *cfg,
                          int *out_ids, float *out_logp, int *out_len) {
  grpo_debug_eval_task_ex(m, NULL, NULL, prompt_ids, n_prompt, task, cfg,
                          out_ids, out_logp, out_len, NULL);
}

long grpo_debug_eval_task_ex(const LlamaModel *m, const HFTok *tok, const char *prompt_text,
                             const int *prompt_ids, int n_prompt, int task, const GrpoConfig *cfg,
                             int *out_ids, float *out_logp, int *out_len, int *out_stop) {
  const int K = cfg->eval_k;
  int ds[2] = {1, m->vocab};
  Tensor logits = tensor_create(ds, 2), probs = tensor_create(ds, 2);
  Tensor plogits = tensor_create(ds, 2);
  LlamaKV kv = llama_kv_create(m), pkv = llama_kv_create_len(m, n_prompt);
  const int text_cap = tok ? TEXT_CAP : 1;
  char *text = (char*)malloc((size_t)text_cap);
  // The constraint exactly as grpo_eval_run builds it: vocabulary view from the
  // tokenizer (clamped to the model's vocab), grounded in the task's prompt text.
  ConVocab cvoc;
  memset(&cvoc, 0, sizeof(cvoc));
  ConCtx cx;
  memset(&cx, 0, sizeof(cx));
  cx.v = &cvoc;
  Constrain *tcon = NULL;
  if (cfg->constrain && tok && prompt_text && convocab_from_hftok(&cvoc, tok) == 0) {
    if (cvoc.V > m->vocab) cvoc.V = m->vocab;
    tcon = constrain_create(prompt_text, &cfg->con);
    cx.c = tcon;
  }
  prefill_prompt(m, NULL, &pkv, prompt_ids, n_prompt, &plogits);
  if (cfg->batch_rollouts) {                 // the K rollouts in lockstep
    RolloutBatch b = rollout_batch_create(m, K, n_prompt + cfg->max_new, cfg->max_new, text_cap);
    for (int kk = 0; kk < K; kk++) b.rng[kk] = rng_create(grpo_eval_rollout_seed(cfg->seed, task, kk));
    rollout_batch_run(m, NULL, &b, &pkv, &plogits, tok, n_prompt, cfg, &probs, text_cap,
                      tcon ? &cx : NULL);
    for (int kk = 0; kk < K; kk++) {
      out_len[kk] = b.len[kk];
      if (out_stop) out_stop[kk] = (int)b.stop[kk];
      memcpy(&out_ids[(size_t)kk * cfg->max_new], b.ids[kk], (size_t)b.len[kk] * sizeof(int));
      memcpy(&out_logp[(size_t)kk * cfg->max_new], b.logp[kk], (size_t)b.len[kk] * sizeof(float));
    }
    rollout_batch_free(&b);
  }
  for (int kk = 0; kk < K && !cfg->batch_rollouts; kk++) {
    StopReason stop = STOP_CAP;
    out_len[kk] = eval_rollout(m, &kv, &pkv, &plogits, tok, prompt_ids, n_prompt, task, kk,
                               cfg, &logits, &probs, &out_ids[(size_t)kk * cfg->max_new],
                               &out_logp[(size_t)kk * cfg->max_new], text, text_cap, &stop,
                               tcon ? &cx : NULL);
    if (out_stop) out_stop[kk] = (int)stop;
  }
  constrain_free(tcon);
  convocab_free(&cvoc);
  free(text);
  llama_kv_free(&kv); llama_kv_free(&pkv);
  tensor_free(&logits); tensor_free(&probs); tensor_free(&plogits);
  return cx.picks;
}

// A short task "family" from the REFERENCE action — what the task asks for:
// write / read / run / delete (a run_command whose command is `rm ...`) / done /
// prose. NULL when there is no reference or it does not parse.
static const char *task_family(const char *reference) {
  if (!reference || !reference[0]) return NULL;
  const char *s = reference, *e;
  while ((e = strstr(s, "</think>")) != NULL) s = e + 8;   // skip the reasoning
  ToolCall tc;
  toolcall_parse(s, &tc);
  switch (tc.kind) {
    case TOOL_WRITE_FILE: return "write";
    case TOOL_READ_FILE:  return "read";
    case TOOL_DONE:       return "done";
    case TOOL_RUN_COMMAND: {
      const char *c = tc.command;
      while (*c == ' ') c++;
      return (strncmp(c, "rm ", 3) == 0) ? "delete" : "run";
    }
    default:
      return (strstr(s, "<tool") || strstr(s, "<done")) ? NULL : "prose";
  }
}

// Length of the well-formed UTF-8 sequence at p, or 0 if it is not one.
static int utf8_len(const unsigned char *p) {
  int n;
  unsigned cp;
  if      ((p[0] & 0xE0) == 0xC0) { n = 2; cp = p[0] & 0x1Fu; }
  else if ((p[0] & 0xF0) == 0xE0) { n = 3; cp = p[0] & 0x0Fu; }
  else if ((p[0] & 0xF8) == 0xF0) { n = 4; cp = p[0] & 0x07u; }
  else return 0;
  for (int i = 1; i < n; i++) {                 // stops at the NUL, too
    if ((p[i] & 0xC0) != 0x80) return 0;
    cp = (cp << 6) | (p[i] & 0x3Fu);
  }
  if ((n == 2 && cp < 0x80) || (n == 3 && cp < 0x800) ||
      (n == 4 && (cp < 0x10000 || cp > 0x10FFFF)) || (cp >= 0xD800 && cp <= 0xDFFF))
    return 0;
  return n;
}

// Write `s` as a JSON string. A byte-level BPE model can emit half a UTF-8
// character; such a byte is written as \u00XX so the file stays valid JSON (and
// two different broken bytes still compare different).
static void json_str(FILE *f, const char *s) {
  fputc('"', f);
  const unsigned char *p = (const unsigned char*)s;
  while (*p) {
    const unsigned char c = *p;
    if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); p++; }
    else if (c == '\n') { fputs("\\n", f); p++; }
    else if (c == '\r') { fputs("\\r", f); p++; }
    else if (c == '\t') { fputs("\\t", f); p++; }
    else if (c < 0x20 || c == 0x7f) { fprintf(f, "\\u%04x", c); p++; }
    else if (c < 0x80) { fputc(c, f); p++; }
    else {
      const int n = utf8_len(p);
      if (n > 0) { fwrite(p, 1, (size_t)n, f); p += n; }
      else { fprintf(f, "\\u%04x", c); p++; }
    }
  }
  fputc('"', f);
}

// FNV-1a over prompt TAB reference: lets evalcard.py refuse to pair two dumps
// made from different task files.
static unsigned long long task_hash(const Task *t) {
  unsigned long long h = 1469598103934665603ULL;
  for (const unsigned char *p = (const unsigned char*)t->prompt; *p; p++) { h ^= *p; h *= 1099511628211ULL; }
  h ^= '\t'; h *= 1099511628211ULL;
  for (const unsigned char *p = (const unsigned char*)t->reference; *p; p++) { h ^= *p; h *= 1099511628211ULL; }
  return h;
}

#define EVAL_FAMILIES 7
static const char *const eval_family_names[EVAL_FAMILIES] =
  {"write", "read", "run", "delete", "done", "prose", "(none)"};

static int family_index(const char *fam) {
  if (fam)
    for (int i = 0; i < EVAL_FAMILIES - 1; i++)
      if (strcmp(fam, eval_family_names[i]) == 0) return i;
  return EVAL_FAMILIES - 1;
}

int grpo_eval_run(const char *tasks_path, const GrpoConfig *cfg) {
  if (!cfg->model || !cfg->hf_tok) {
    fprintf(stderr, "error: grpo --eval-only needs --model <LLMA checkpoint> and --hf-tok <vocab.hf>\n");
    return 1;
  }
  const int K = cfg->eval_k > 0 ? cfg->eval_k : 1;
  GrpoConfig ec = *cfg;
  ec.eval_k = K;

  int n_tasks = 0;
  Task *tasks = tasks_load(tasks_path, &n_tasks);
  if (!tasks) return 1;
  const int n_eval = (cfg->eval_limit > 0 && cfg->eval_limit < n_tasks) ? cfg->eval_limit : n_tasks;

  HFTok tok;
  if (hftok_load(&tok, cfg->hf_tok) != 0) {
    fprintf(stderr, "error: cannot load HF tokenizer '%s'\n", cfg->hf_tok);
    tasks_free(tasks, n_tasks);
    return 1;
  }
  // Inference-only load: no gradient arrays, no adapters, no optimizer state.
  LlamaModel model;
  if (llama_model_load(cfg->model, &model) != 0) {
    fprintf(stderr, "error: cannot load model '%s'\n", cfg->model);
    hftok_free(&tok); tasks_free(tasks, n_tasks);
    return 1;
  }

  FILE *dump = NULL;
  if (cfg->eval_dump) {
    dump = fopen(cfg->eval_dump, "wb");       // "wb": LF line ends on every platform
    if (!dump) {
      fprintf(stderr, "error: cannot write '%s'\n", cfg->eval_dump);
      llama_model_free(&model); hftok_free(&tok); tasks_free(tasks, n_tasks);
      return 1;
    }
  }

  // Tokenize the evaluated prompts once; a prompt that leaves no room for a
  // response is reported and skipped (it gets no dump line).
  int **pids = (int**)calloc((size_t)n_eval, sizeof(int*));
  int *plen = (int*)calloc((size_t)n_eval, sizeof(int));
  int scratch[MAX_PROMPT_IDS];
  int maxP = 1, too_long = 0;
  for (int i = 0; i < n_eval; i++) {
    const int T = hftok_encode(&tok, tasks[i].prompt, scratch, MAX_PROMPT_IDS);
    if (T < 1 || T + 2 >= model.max_seq) { too_long++; continue; }
    pids[i] = (int*)malloc((size_t)T * sizeof(int));
    if (!pids[i]) continue;
    memcpy(pids[i], scratch, (size_t)T * sizeof(int));
    plen[i] = T;
    if (T > maxP) maxP = T;
  }

  const int V = model.vocab;
  int dshape[2] = {1, V};
  Tensor logits = tensor_create(dshape, 2), probs = tensor_create(dshape, 2);
  Tensor plogits = tensor_create(dshape, 2);
  const int batched = cfg->batch_rollouts;
  const int cap_new = cfg->max_new > 0 ? cfg->max_new : 1;
  // --batch-rollouts: K caches of prompt + response positions replace the one
  // max_seq cache (then a 1-position stub).
  LlamaKV kv = batched ? llama_kv_create_len(&model, 1) : llama_kv_create(&model);
  LlamaKV pkv = llama_kv_create_len(&model, maxP);
  RolloutBatch rb;
  memset(&rb, 0, sizeof(rb));
  if (batched) rb = rollout_batch_create(&model, K, maxP + cap_new, cap_new, TEXT_CAP);
  int *ids = (int*)malloc((size_t)cap_new * sizeof(int));
  float *logp = (float*)malloc((size_t)cap_new * sizeof(float));
  char *etext = (char*)malloc(TEXT_CAP);
  char *line = (char*)malloc(LINE_CAP);
  const float match_max = reward_match_max(cfg->reward_version);
  const int rv = cfg->reward_version == 2 ? 2 : 1;
  ConVocab cvoc;
  memset(&cvoc, 0, sizeof(cvoc));
  ConCtx cx;
  memset(&cx, 0, sizeof(cx));
  cx.v = &cvoc;
  int con_fail = cfg->constrain && convocab_from_hftok(&cvoc, &tok) != 0;
  if (cvoc.V > V) cvoc.V = V;                 // only ids the model has logits for

  printf("GRPO EVAL: tasks 0..%d of %d in file order | k %d per task | reward v%d | seed %u\n",
         n_eval - 1, n_tasks, K, rv, cfg->seed);
  printf("  model %s (vocab %d, d_model %d, layers %d) | tok %s\n",
         cfg->model, V, model.d_model, model.n_layers, cfg->hf_tok);
  if (cfg->temperature <= 0.0f)
    printf("  sampling: GREEDY (temp 0) | max-new %d\n", cfg->max_new);
  else
    printf("  sampling: temp %.2f top-k %d top-p %.2f max-new %d | per-rollout RNG = hash(seed, task, k)\n",
           (double)cfg->temperature, cfg->top_k, (double)cfg->top_p, cfg->max_new);
  printf("  inference only: no LoRA, no optimizer, no backward, nothing saved%s%s\n",
         dump ? " | dump -> " : "", dump ? cfg->eval_dump : "");
  if (batched)
    printf("  rollouts: BATCHED -- the %d of a task decoded in lockstep (same per-rollout RNGs,"
           " identical output)\n", K);
  if (cfg->temperature <= 0.0f && K > 1)
    printf("  NOTE: temp 0 is greedy, so all %d rollouts of a task are identical; use --k 1\n", K);
  if (cfg->constrain)
    printf("  CONSTRAINED decoding: one action block%s%s%s | think <= %d bytes | window %d\n",
           cfg->con.ground_path ? ", path = whole-token prompt span" : "",
           cfg->con.ground_body ? ", write body = prompt span" : "",
           cfg->con.ground_cmd ? ", command args = prompt spans" : "",
           cfg->con.think_max, cfg->con.window);
  fflush(stdout);

  RewardMonitor mon;
  memset(&mon, 0, sizeof(mon));
  double sum_total = 0.0;
  long n_exact = 0, n_roll = 0;
  int tasks_done = 0, tasks_any_exact = 0, tasks_major_exact = 0;
  double fam_total[EVAL_FAMILIES] = {0};
  long fam_exact[EVAL_FAMILIES] = {0}, fam_n[EVAL_FAMILIES] = {0};
  int fam_tasks[EVAL_FAMILIES] = {0};
  const clock_t t0 = clock();

  for (int ti = 0; ti < n_eval && !con_fail; ti++) {
    if (plen[ti] == 0) continue;
    const int T = plen[ti];
    const char *fam = task_family(tasks[ti].reference);
    const int fi = family_index(fam);
    const unsigned long long th = task_hash(&tasks[ti]);
    fam_tasks[fi]++;
    // One prefill per task, forked into each of the K rollouts.
    prefill_prompt(&model, NULL, &pkv, pids[ti], T, &plogits);
    Constrain *tcon = NULL;
    if (cfg->constrain) {
      tcon = constrain_create(tasks[ti].prompt, &cfg->con);   // grounding text = the task prompt
      if (!tcon) { con_fail = 1; break; }
      cx.c = tcon;
    }
    if (batched) {                  // all K at once, rollout kk on its eval stream
      for (int kk = 0; kk < K; kk++) rb.rng[kk] = rng_create(grpo_eval_rollout_seed(ec.seed, ti, kk));
      rollout_batch_run(&model, NULL, &rb, &pkv, &plogits, &tok, T, &ec, &probs, TEXT_CAP,
                        tcon ? &cx : NULL);
    }
    int task_exact = 0;
    for (int kk = 0; kk < K; kk++) {
      StopReason stop = STOP_CAP;
      int n;
      const char *text = etext;
      if (batched) {
        n = rb.len[kk]; stop = rb.stop[kk]; text = rb.text[kk];
      } else {
        n = eval_rollout(&model, &kv, &pkv, &plogits, &tok, pids[ti], T, ti, kk, &ec,
                         &logits, &probs, ids, logp, etext, TEXT_CAP, &stop,
                         tcon ? &cx : NULL);
      }
      RewardParts parts;
      const float r = reward_score_version(cfg->reward_version, tasks[ti].prompt, text,
                                           tasks[ti].reference, stop == STOP_CAP, &parts);
      const int exact = (tasks[ti].reference[0] && parts.match >= match_max - 1e-4f);
      mon_add(&mon, &parts, stop == STOP_CAP);
      sum_total += r; n_exact += exact; n_roll++; task_exact += exact;
      fam_total[fi] += r; fam_exact[fi] += exact; fam_n[fi]++;
      const char *sname = stop == STOP_EOS ? "eos" : (stop == STOP_TAG ? "tag"
                        : (stop == STOP_ERR ? "err" : "cap"));
      if (dump) {
        fprintf(dump, "{\"task\":%d,\"k\":%d,\"family\":", ti, kk);
        if (fam) json_str(dump, fam); else fputs("null", dump);
        fprintf(dump, ",\"reward\":\"v%d\",\"total\":%.6g,\"format\":%.6g,\"grounded\":%.6g,"
                      "\"match\":%.6g,\"exact\":%d,\"action\":\"%s\",\"stop\":\"%s\",\"tokens\":%d,"
                      "\"temp\":%g,\"seed\":%u,\"task_hash\":\"%016llx\",\"response\":",
                rv, (double)r, (double)parts.format, (double)parts.grounded, (double)parts.match,
                exact, reward_action_name(parts.action), sname, n,
                (double)(cfg->temperature > 0.0f ? cfg->temperature : 0.0f), cfg->seed, th);
        json_str(dump, text);
        fputs("}\n", dump);
      }
      if (cfg->verbose) {
        one_line(text, line, LINE_CAP);
        printf("  task %4d k %d [%s] r=%.2f (fmt %.2f grd %.2f mat %.2f) [%s %d tok] %s\n      %s\n",
               ti, kk, fam ? fam : "-", (double)r, (double)parts.format, (double)parts.grounded,
               (double)parts.match, sname, n, parts.reason[0] ? parts.reason : "ok", line);
      }
    }
    constrain_free(tcon);
    tasks_done++;
    if (task_exact > 0) tasks_any_exact++;
    if (2 * task_exact > K) tasks_major_exact++;
    if (!cfg->verbose && (tasks_done % 25 == 0)) {
      printf("  ... %d tasks, mean reward %.4f, exact %.1f%%\n", tasks_done,
             sum_total / (double)n_roll, 100.0 * (double)n_exact / (double)n_roll);
      fflush(stdout);
    }
  }

  const double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
  if (n_roll > 0) {
    printf("EVAL SUMMARY: %d tasks x %d = %ld rollouts (%d skipped: prompt too long) %.1fs\n",
           tasks_done, K, n_roll, too_long, secs);
    printf("  mean reward %.4f | exact-match %.2f%% (%ld/%ld rollouts)",
           sum_total / (double)n_roll, 100.0 * (double)n_exact / (double)n_roll, n_exact, n_roll);
    if (K > 1)
      printf(" | tasks: majority-exact %.2f%%, any-of-%d exact %.2f%%",
             100.0 * tasks_major_exact / tasks_done, K, 100.0 * tasks_any_exact / tasks_done);
    printf("\n");
    mon_print("  ", &mon);
    if (cfg->constrain)
      printf("  constrained: %ld picks, %ld needed a full-vocab scan, %ld dead ends (stop \"err\")\n",
             cx.picks, cx.full_scans, cx.errors);
    printf("  per family (from the reference action):\n");
    for (int f = 0; f < EVAL_FAMILIES; f++) {
      if (fam_n[f] == 0) continue;
      printf("    %-7s %4d tasks  mean %.4f  exact %6.2f%%\n", eval_family_names[f], fam_tasks[f],
             fam_total[f] / (double)fam_n[f], 100.0 * (double)fam_exact[f] / (double)fam_n[f]);
    }
    if (dump) printf("  wrote %ld lines -> %s (compare with: python tools/evalcard.py A.jsonl B.jsonl)\n",
                     n_roll, cfg->eval_dump);
  } else {
    printf("EVAL SUMMARY: no task could be evaluated (%d skipped: prompt too long)\n", too_long);
  }

  if (con_fail) fprintf(stderr, "error: out of memory building the decode constraint\n");
  int rc = (n_roll > 0 && !con_fail) ? 0 : 1;
  if (dump && fclose(dump) != 0) { fprintf(stderr, "error writing '%s'\n", cfg->eval_dump); rc = 1; }
  free(ids); free(logp); free(etext); free(line);
  if (batched) rollout_batch_free(&rb);
  convocab_free(&cvoc);
  llama_kv_free(&kv); llama_kv_free(&pkv);
  tensor_free(&logits); tensor_free(&probs); tensor_free(&plogits);
  llama_model_free(&model);
  for (int i = 0; i < n_eval; i++) free(pids[i]);
  free(pids); free(plen);
  hftok_free(&tok);
  tasks_free(tasks, n_tasks);
  return rc;
}
