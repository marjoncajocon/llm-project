// dpo.h - Direct Preference Optimization (Rafailov et al., 2023)
//
// The third way this engine learns. To place it:
//
//   llama-train --sft   here is the right answer, imitate it
//   grpo                act, score what you did, push toward better-than-average
//   dpo                 here are two answers, one is better -- prefer it
//
// WHY THIS EXISTS ALONGSIDE GRPO. GRPO must SAMPLE to learn: G rollouts per task,
// each a full KV-cache decode, which on this CPU costs 12-40 s per step. DPO
// needs no sampling at all. The preferences are collected offline, once, and
// every training step is a handful of forward passes over text that already
// exists. On a CPU that difference is the whole ballgame.
//
// THE OBJECTIVE. For a prompt x with a preferred completion y_w and a rejected
// one y_l, with a FROZEN reference policy pi_ref:
//
//   s_w = log pi(y_w|x) - log pi_ref(y_w|x)      (implicit reward, chosen)
//   s_l = log pi(y_l|x) - log pi_ref(y_l|x)      (implicit reward, rejected)
//   h   = beta * (s_w - s_l)                     (the preference margin)
//   L   = -log sigmoid(h) = softplus(-h)
//
// The insight of the paper is that this is exactly the RLHF objective with the
// reward model solved for in closed form -- so no reward model and no rollouts,
// just a classification loss over pairs.
//
// HOW IT REACHES THE WEIGHTS. d/d logits of log pi(y_t) is (onehot - p), the same
// score function GRPO uses, so the whole backward is again pg_backward (loss.h)
// with a per-token coefficient. Differentiating L gives one scalar shared by
// every token of the pair:
//
//   g = beta * sigmoid(-h)
//   coeff = +g  on every scored token of the CHOSEN completion   (push it up)
//   coeff = -g  on every scored token of the REJECTED completion (push it down)
//
// Note what g does: sigmoid(-h) is near 1 when the model still prefers the WRONG
// answer and near 0 once the preference is already learned. The update is
// self-annealing -- pairs the model has understood stop contributing gradient.
//
// THE REFERENCE IS FREE. With LoRA the base weights are frozen, so pi_ref is just
// the same model with the adapters switched off. No second copy of a 621 MB
// checkpoint. That is why this trainer REQUIRES --lora-r > 0: without adapters
// there is no reference policy to be had without doubling memory.
#ifndef DPO_H
#define DPO_H

typedef struct {
  // --- What to train, and from what ---
  const char *model;        // LLMA checkpoint to start from (required)
  const char *hf_tok;       // the model's own .hf/.tok vocabulary (required)
  const char *out_model;    // where the updated checkpoint is written

  // --- The objective ---
  float beta;               // KL strength. The ONLY knob unique to DPO: it sets how
                            // far the policy may move from the reference. 0.1 is the
                            // paper's default; lower drifts further and can degrade,
                            // higher stays cautious and learns less.
  int len_norm;             // 1 => divide each completion's log-prob by its token
                            // count. Standard DPO sums (len_norm = 0), which gives a
                            // known LENGTH BIAS: summing more negative terms makes
                            // long completions look worse regardless of quality. Turn
                            // this on when chosen and rejected differ a lot in length.

  // --- Optimization ---
  int steps;
  int batch;                // preference pairs accumulated per optimizer step
  float lr, weight_decay, grad_clip;
  int warmup;
  float lr_min;             // < 0 => constant LR
  int lora_r;               // REQUIRED > 0 (see above)
  float lora_alpha;

  // --- Run control ---
  unsigned seed;
  int verbose;              // print a sample pair with its margin each report
} DpoConfig;

// Defaults: beta 0.1, batch 4, lr 5e-6, LoRA rank 8 / alpha 16, clip 1.0.
DpoConfig dpo_default_config(void);

// Train on a file of `prompt<TAB>chosen<TAB>rejected` lines -- the SFT format with
// one extra column. Returns 0 on success, nonzero on error.
int dpo_run(const char *pairs_path, const DpoConfig *cfg);

// ---------------------------------------------------------------------------
// The objective as pure arithmetic, so it can be unit-tested and finite-
// difference checked without a model. Given the four summed log-probabilities:
//
//   returns the loss softplus(-h), and writes to *g_out the shared coefficient
//   beta*sigmoid(-h) -- apply it as +g on the chosen tokens and -g on the
//   rejected ones. *margin_out receives h/beta = s_w - s_l, which is the number
//   worth logging: it starts near 0 and should climb as the model learns the
//   preference. *acc_out receives 1 if the model currently ranks the pair
//   correctly (h > 0), which is the accuracy metric DPO runs are judged on.
// ---------------------------------------------------------------------------
float dpo_pair_loss(float pi_chosen, float pi_rejected,
                    float ref_chosen, float ref_rejected,
                    float beta, float *g_out, float *margin_out, int *acc_out);

#endif // DPO_H
