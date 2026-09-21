// grpo.h - GRPO: Group Relative Policy Optimization (RL from verifiable rewards)
//
// The engine's other trainer (train.h) is SUPERVISED: it is shown the right
// answer and pushed toward it. GRPO instead lets the model ACT, scores what it
// did against a verifiable reward (reward.h), and pushes it toward its own
// better-than-average attempts and away from its worse-than-average ones.
//
// The loop, per task:
//   1. sample G rollouts from the current policy (KV-cache decode WITH the LoRA
//      adapters applied, so the samples come from the policy being updated;
//      the shared prompt is prefilled ONCE per group and its KV cache forked
//      into each rollout),
//   2. score each with reward.h — no reward model, no value network, no critic,
//      which is exactly why this fits a 135M model on a CPU,
//   3. normalize the rewards WITHIN the group:  Â_i = (r_i - mean) / (std + eps).
//      This is the "group relative" part and it replaces the value network: the
//      group's own mean is the baseline.
//   4. one policy-gradient step, clipped like PPO and optionally pulled toward a
//      reference policy by a KL penalty.
//
// Why the group matters: if every rollout scores the same, every advantage is
// zero and the group teaches nothing — it is SKIPPED before any forward pass.
// GRPO learns from VARIANCE. Where the model is uniformly right or uniformly
// wrong there is no signal, which is the honest limit of this method at 135M.
//
// The relationship to the STaR loop (tools/self_train.py): STaR keeps the
// successful rollouts as SFT rows and discards the failures. GRPO uses the
// failures too — a below-average rollout gets a NEGATIVE advantage, actively
// suppressing it. That suppression is the capability rejection sampling does not
// have.
//
// The reference policy for the KL term is free here: with LoRA the base weights
// are frozen, so running the same model with the adapters switched off IS the
// reference. No second copy of the checkpoint.
#ifndef GRPO_H
#define GRPO_H

#include "llama.h"
#include "sampler.h"

typedef struct {
  // --- What to train, and from what ---
  const char *model;        // LLMA checkpoint to start from (required)
  const char *hf_tok;       // the model's own .hf/.tok vocabulary (required)
  const char *out_model;    // where the updated checkpoint is written

  // --- Rollouts ---
  int group;                // G: rollouts sampled per task (>= 2; the group needs
                            // at least two members to have a mean to be relative to)
  int max_new;              // hard cap on sampled tokens per rollout
  float temperature;        // sampling temperature — this is the EXPLORATION knob.
                            // Too low and every rollout in the group is identical,
                            // the advantage is zero and nothing is learned.
  int top_k;
  float top_p;

  // --- Optimization ---
  int steps;                // optimizer steps
  int groups_per_step;      // task groups accumulated into one step (like --batch)
  int inner_epochs;         // mu: updates taken per set of rollouts (1 = on-policy)
  float lr, weight_decay, grad_clip;
  int warmup;
  float lr_min;             // < 0 => constant LR
  int lora_r;               // LoRA rank (0 => full fine-tune; not recommended here)
  float lora_alpha;
  int lora_mlp;             // 1 => adapters on Wgate/Wup/Wdown too (0 => attention only)
  float lora_plus;          // LoRA+: lr_B = lora_plus * lr, A keeps lr (0 => off)

  // --- The GRPO objective ---
  float clip_eps;           // PPO-style ratio clip, 1 +/- clip_eps (0 => no clipping)
  float clip_eps_high;      // DAPO "clip-higher": the UPPER bound becomes 1 + clip_eps_high
                            // (< 0 => symmetric, i.e. clip_eps). A wider upper bound lets
                            // a low-probability token that turned out good grow faster,
                            // which is what keeps the policy from collapsing onto the
                            // tokens it already favours (entropy collapse).
  float kl_beta;            // KL-to-reference penalty (0 => off, and then the extra
                            // reference forward per rollout is skipped entirely)
  int std_norm;             // 1 => Â = (r - mean) / std (original GRPO); 0 => Â = r - mean
                            // (Dr. GRPO). Dividing by std up-weights groups that are
                            // NEARLY flat — the easiest and the hardest tasks — which
                            // is a difficulty bias, not a signal.
  int loss_agg;             // how per-token terms are averaged — a GrpoAgg value
  int resample;             // DAPO dynamic sampling: when a group comes back flat, draw a
                            // fresh task and resample, up to this many times, so the step
                            // is not spent on a group with no gradient (0 => off)

  // --- Difficulty-aware task sampling (--prio; see GrpoPrio below) ---
  int prio;                 // 1 => draw tasks by grpo_prio_weight instead of uniform
                            // rand(); 0 (default) => the original rand() draw, untouched
  float prio_alpha;         // EMA rate for the per-task statistics (default 0.3)
  float prio_wmin;          // lower clip of the (1 - success) term (default 0.05)
  float prio_hi;            // a flat group whose reward is >= this is "all correct"
                            // (mastered); a flat group below it is "all wrong"
                            // (too hard for now). Default 0.9.
  int diag;                 // 1 => append training diagnostics to the report lines
                            // (entropy, live-group fraction, Adam updates, clip
                            // fraction) and a task-coverage summary at the end.
                            // Implied by --prio. 0 (default) => the log is unchanged.

  // --- The reward ---
  int reward_version;       // 1 (default) => reward_score, the original reward;
                            // 2 => reward_score_v2: format gates, capped rollouts
                            // score 0, <done>/prose earn grounding only when the
                            // reference is the same kind, stricter command grounding
                            // (reward.h). Opt-in so v1 runs reproduce.

  // --- Evaluation (grpo --eval-only; see grpo_eval_run) ---
  int eval_only;            // 1 => grpo_eval_run: no optimizer, no LoRA, no backward
  int eval_k;               // rollouts per task (default 1; with temp 0 that is greedy)
  int eval_limit;           // evaluate only the first N tasks of the file (0 => all)
  const char *eval_dump;    // JSONL path: one line per rollout (NULL => no dump)

  // --- Run control ---
  unsigned seed;
  int verbose;              // 1 => print a scored sample rollout per group
} GrpoConfig;

// How a rollout's per-token terms are normalized. This is not a detail: it decides
// how much a LONG rollout counts against a short one.
typedef enum {
  // 1 / (G * len_i) — the original GRPO. Every rollout carries equal total weight,
  // so each token of a long rollout counts for less. A long WRONG answer is
  // therefore punished less per token than a short wrong one, which quietly
  // rewards rambling when the answer is bad.
  GRPO_AGG_SEQ = 0,
  // 1 / (total tokens of every live rollout in the step) — DAPO. Every TOKEN
  // carries equal weight, whichever rollout it came from.
  GRPO_AGG_TOKEN = 1,
  // 1 / (G * max_new) — Dr. GRPO. A constant, so length has no effect on the weight.
  GRPO_AGG_CONST = 2
} GrpoAgg;

// The presets, applied on top of whatever else is in *c (CLI flags that come AFTER
// the preset still win, so `--dapo --clip-high 0.3` works).
//   DAPO:     clip-higher 0.28, token-level loss, up to 2 resamples of a flat group.
//   Dr. GRPO: no std division, constant normalizer.
void grpo_preset_dapo(GrpoConfig *c);
void grpo_preset_dr_grpo(GrpoConfig *c);

// The per-token denominator for rollout i under cfg->loss_agg. `step_tokens` is the
// response-token total of every live rollout in the step (only GRPO_AGG_TOKEN reads it).
float grpo_loss_norm(const GrpoConfig *cfg, int G, int len, int step_tokens);
// There is deliberately no mid-run checkpointing. Saving would mean merging the
// adapters into the base, and llama_lora_merge resets B to zero — after which
// dA is identically zero and training stalls. A run saves once, at the end.

// Defaults: G=8, 1 group/step, mu=1, clip 0.2, KL off, LoRA rank 8, lr 1e-5 —
// and the ORIGINAL objective (symmetric clip, std-normalized advantage, per-sequence
// loss, no resampling), so a run from before the DAPO/Dr. GRPO options reproduces.
GrpoConfig grpo_default_config(void);

// Run GRPO over a task file of `prompt<TAB>reference_action` lines — the SAME
// format the SFT trainer reads, so any corpus built by tools/clean_sft.py works
// unchanged (the response column becomes the reference action the reward scores
// against). Saves the adapted model to cfg->out_model.
// Returns 0 on success, nonzero on error.
int grpo_run(const char *tasks_path, const GrpoConfig *cfg);

// ---------------------------------------------------------------------------
// The objective, as pure arithmetic — no tensors, no model, no tokenizer — so it
// can be unit-tested and gradient-checked on its own. This is the whole of GRPO
// that is not just "run the model".
//
// For each rollout i with reward r_i and length lens[i]:
//     Â_i     = (r_i - mean_r) / (std_r + 1e-4)          (std_norm = 1)
//             =  r_i - mean_r                            (std_norm = 0)
// and for each of its tokens t:
//     rho     = exp(logp_new - logp_old)                  (importance ratio)
//     clipped = (Â > 0 && rho > 1+eps_high) || (Â < 0 && rho < 1-eps)
//     pg      = clipped ? 0 : Â * rho
//     kl      = beta * (rho_ref - 1),  rho_ref = exp(logp_ref - logp_new)
//     coeff   = (pg + kl) / norm_i,   norm_i = grpo_loss_norm(cfg, G, lens[i], ...)
//
// coeff is the per-token weight to hand to pg_backward (loss.h). The sign
// convention is loss-like: pg_backward adds coeff * (p - onehot), and (p - onehot)
// is the gradient of -log p, so a POSITIVE coeff raises the token's probability.
//
// The flat arrays are laid out rollout-major: rollout i's tokens occupy
// [sum(lens[0..i-1]), + lens[i]). logp_ref may be NULL when cfg->kl_beta == 0.
// adv_out (length G) may be NULL if the advantages are not wanted.
//
// Returns 1 if the group carries signal, or 0 if every reward was identical — in
// which case coeff_out is all zeros and the caller should skip the group's
// forward and backward passes entirely.
int grpo_group_coeffs(const float *rewards, int G, const int *lens,
                      const float *logp_new, const float *logp_old,
                      const float *logp_ref, const GrpoConfig *cfg,
                      float *adv_out, float *coeff_out);

// The two halves of the above, exposed separately because the ADVANTAGE depends
// only on the rewards — not on any log-probability. That lets the trainer settle
// the whole group's advantages up front and then do exactly ONE forward per
// rollout (recompute log-probs, weight them, backpropagate) instead of one pass
// to collect log-probs and a second to use them.
//
// grpo_advantages fills adv_out (length G) and returns 1 if the group has spread,
// 0 if every reward was identical (advantages all zero — skip the group).
// It reads only cfg->std_norm.
int grpo_advantages(const float *rewards, int G, const GrpoConfig *cfg, float *adv_out);

// Per-token coefficients for ONE rollout of the group, given its advantage and its
// denominator from grpo_loss_norm. coeff_out must have room for `len` entries.
// logp_ref may be NULL.
void grpo_token_coeffs(float adv, float norm, int len, const float *logp_new,
                       const float *logp_old, const float *logp_ref,
                       const GrpoConfig *cfg, float *coeff_out);

// grpo_token_coeffs, also counting the tokens whose policy-gradient term the ratio
// clip removed (the "clip fraction" numerator). *n_clipped is ADDED to, so a caller
// can accumulate across rollouts; NULL => not counted. Same coeff_out either way.
void grpo_token_coeffs_ex(float adv, float norm, int len, const float *logp_new,
                          const float *logp_old, const float *logp_ref,
                          const GrpoConfig *cfg, float *coeff_out, int *n_clipped);

// Shannon entropy -sum p log p (nats) of one probability row of length V. Zero
// entries contribute nothing. The trainer applies it to the rows of the softmax
// the update forward ALREADY built (no extra forward) for the diagnostics line.
double grpo_row_entropy(const float *p, int V);

// ---------------------------------------------------------------------------
// Difficulty-aware task sampling (--prio)
//
// The problem it answers: with uniform draws, a large share of groups come back
// FLAT (every rollout scores the same => zero advantage => no gradient). The
// 135M run in logs/v3/train_G1_grpo.log had 47 of 100 groups flat. Two ideas from
// the literature, combined:
//   - Kimi k1.5 curriculum sampling: weight a problem by (1 - success rate), so
//     the steps go to what the policy does not solve yet;
//   - GRESO (arXiv 2506.02177): a prompt with a HISTORY of zero-variance groups
//     will very likely be zero-variance again, so skip it with a probability
//     that grows with the streak.
//
// Per task i the sampler keeps (all start "unseen"):
//   succ_i   EMA of the group's mean reward (rewards are in [0, 1])
//   flat_i   EMA of the indicator "the group was flat"         (reported only)
//   hi_i     current streak of consecutive flat groups at reward >= prio_hi
//            (all correct: MASTERED); reset by any other group
//   lo_i     current streak of consecutive flat groups below prio_hi
//            (all equally wrong: TOO HARD for now); reset by any other group
// with EMA x <- (1 - alpha) x + alpha * new, the first observation taken as is.
//
// The draw weight:
//   unseen task:  w_i = GRPO_PRIO_W_UNSEEN (= 1, the maximum: every task is looked at)
//   seen task:    base = clamp(1 - succ_i, prio_wmin, 1)
//                 w_i  = max(GRPO_PRIO_W_FLOOR, base * DECAY_HI^min(hi_i, 8)
//                                                    * DECAY_LO^min(lo_i, 8))
//   with DECAY_HI = 0.5 (mastered fades fast), DECAY_LO = 0.7 (too-hard is
//   retried, but rarely), W_FLOOR = 0.01 (no task ever becomes unreachable, so a
//   task the policy has since started to make progress on can come back).
// Only one of hi_i / lo_i is nonzero at a time. Every weight lies in
// [GRPO_PRIO_W_FLOOR, 1]. Task i is drawn with probability w_i / sum_j w_j over
// the usable tasks, from the sampler's own SplitMix64 stream seeded by --seed:
// deterministic, and independent of rand() and of the rollout RNG.
// ---------------------------------------------------------------------------
#define GRPO_PRIO_W_UNSEEN 1.0f
#define GRPO_PRIO_W_FLOOR  0.01f
#define GRPO_PRIO_DECAY_HI 0.5f
#define GRPO_PRIO_DECAY_LO 0.7f
#define GRPO_PRIO_STREAK_CAP 8

typedef struct {
  int seen;                 // groups observed for this task (0 => unseen)
  float succ;               // EMA of mean group reward
  float flat;               // EMA of "group was flat"
  int streak_hi, streak_lo; // consecutive flat-all-correct / flat-below-top groups
} GrpoTaskStat;

typedef struct {
  int n;                    // number of tasks
  GrpoTaskStat *st;         // n entries
  unsigned long long rng;   // SplitMix64 state (the draw stream)
  float alpha, w_min, hi;
  long draws;               // weighted draws taken
} GrpoPrio;

GrpoPrio grpo_prio_create(int n_tasks, const GrpoConfig *cfg);
void grpo_prio_free(GrpoPrio *p);
// The draw weight of task i (formula above); always in [GRPO_PRIO_W_FLOOR, 1].
float grpo_prio_weight(const GrpoPrio *p, int i);
// Record one scored group of task i: its mean reward and whether it was flat.
void grpo_prio_update(GrpoPrio *p, int i, float mean_reward, int flat);
// Draw a task index whose plen[] is nonzero. p == NULL => the ORIGINAL draw,
// `do ti = rand() % n_tasks; while (plen[ti] == 0);`, exactly (the default
// path); otherwise a weighted draw from p's own stream (consumes no rand()).
int grpo_draw_task(GrpoPrio *p, const int *plen, int n_tasks);
// Coverage summary over tasks with plen[i] != 0 (plen may be NULL => all):
// distinct tasks seen, mastered (last group flat at >= prio_hi), too hard (last
// group flat below prio_hi).
void grpo_prio_summary(const GrpoPrio *p, const int *plen, int *seen, int *mastered,
                       int *too_hard);

// Test hook (selftest [37]): sample one group of cfg->group rollouts from `prompt_ids`
// through the SAME rollout() grpo_run uses, with no tokenizer (so no EOS / closing-tag
// stops: every rollout runs to min(cfg->max_new, max_seq - n_prompt) tokens).
//   shared_prefill = 1: prefill the prompt once, fork it into each rollout with
//                       llama_kv_copy (what grpo_run does);
//   shared_prefill = 0: re-prefill per rollout (the path grpo_run used before).
// out_ids / out_logp are rollout-major with stride cfg->max_new; out_len has G
// entries. Returns the number of prompt-prefill decode steps taken.
long grpo_debug_sample_group(const LlamaModel *m, const LlamaDecodeLora *lo,
                             const int *prompt_ids, int n_prompt, const GrpoConfig *cfg,
                             int shared_prefill, Rng *rng,
                             int *out_ids, float *out_logp, int *out_len);

// ---------------------------------------------------------------------------
// Evaluation: `transformer grpo <tasks> --eval-only ...`
//
// `grpo --lr 0` is NOT an evaluation: it draws tasks WITH replacement (40 draws
// cover ~36 of 200 tasks), shares one sampling RNG across the run (after the
// first rollout whose length differs between two checkpoints, every later draw
// is unpaired noise), still runs the backward pass, and never logs per-task
// rewards. grpo_eval_run instead:
//   - loads the checkpoint for inference only (no gradient arrays, no LoRA, no
//     optimizer, no backward, nothing saved);
//   - visits the tasks in FILE ORDER (the first cfg->eval_limit, if > 0), each
//     exactly once;
//   - samples cfg->eval_k rollouts per task, rollout (i, k) drawing from its OWN
//     Rng seeded by grpo_eval_rollout_seed(cfg->seed, i, k) — so the same draw
//     is used for the same (task, k) by every checkpoint, whatever happened on
//     other tasks (a paired design: sampling noise cancels in the difference);
//   - prefills each task's prompt once and forks the KV cache per rollout (the
//     same shared-prefill path grpo_run uses);
//   - scores with cfg->reward_version and writes one JSON line per rollout to
//     cfg->eval_dump (compare two dumps with tools/evalcard.py).
// Returns 0 on success.
int grpo_eval_run(const char *tasks_path, const GrpoConfig *cfg);

// The per-rollout RNG seed: a SplitMix64 hash of (seed, task index, k). Never 0.
unsigned long long grpo_eval_rollout_seed(unsigned seed, int task, int k);

// Test hook (selftest [39]): evaluate ONE task exactly as grpo_eval_run does —
// shared prefill, cfg->eval_k rollouts, rollout k drawing from
// rng_create(grpo_eval_rollout_seed(cfg->seed, task, k)) — with no tokenizer (no
// EOS / closing-tag stop). out_ids / out_logp are rollout-major with stride
// cfg->max_new; out_len has eval_k entries.
void grpo_debug_eval_task(const LlamaModel *m, const int *prompt_ids, int n_prompt,
                          int task, const GrpoConfig *cfg,
                          int *out_ids, float *out_logp, int *out_len);

#endif // GRPO_H
