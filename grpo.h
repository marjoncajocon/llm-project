// grpo.h - GRPO: Group Relative Policy Optimization (RL from verifiable rewards)
//
// The engine's other trainer (train.h) is SUPERVISED: it is shown the right
// answer and pushed toward it. GRPO instead lets the model ACT, scores what it
// did against a verifiable reward (reward.h), and pushes it toward its own
// better-than-average attempts and away from its worse-than-average ones.
//
// The loop, per task:
//   1. sample G rollouts from the current policy (KV-cache decode WITH the LoRA
//      adapters applied, so the samples come from the policy being updated),
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

  // --- The GRPO objective ---
  float clip_eps;           // PPO-style ratio clip, 1 +/- clip_eps (0 => no clipping)
  float kl_beta;            // KL-to-reference penalty (0 => off, and then the extra
                            // reference forward per rollout is skipped entirely)

  // --- Run control ---
  unsigned seed;
  int verbose;              // 1 => print a scored sample rollout per group
} GrpoConfig;
// There is deliberately no mid-run checkpointing. Saving would mean merging the
// adapters into the base, and llama_lora_merge resets B to zero — after which
// dA is identically zero and training stalls. A run saves once, at the end.

// Defaults: G=8, 1 group/step, mu=1, clip 0.2, KL off, LoRA rank 8, lr 1e-5.
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
//     Â_i     = (r_i - mean_r) / (std_r + 1e-4)
// and for each of its tokens t:
//     rho     = exp(logp_new - logp_old)                  (importance ratio)
//     clipped = (Â > 0 && rho > 1+eps) || (Â < 0 && rho < 1-eps)
//     pg      = clipped ? 0 : Â * rho
//     kl      = beta * (rho_ref - 1),  rho_ref = exp(logp_ref - logp_new)
//     coeff   = (pg + kl) / (G * lens[i])
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
int grpo_advantages(const float *rewards, int G, float *adv_out);

// Per-token coefficients for ONE rollout of the group, given its advantage.
// coeff_out must have room for `len` entries. logp_ref may be NULL.
void grpo_token_coeffs(float adv, int G, int len, const float *logp_new,
                       const float *logp_old, const float *logp_ref,
                       const GrpoConfig *cfg, float *coeff_out);

#endif // GRPO_H
