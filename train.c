// train.c - Train a GPT on a text file or directory
#include "train.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "tokenizer.h"
#include "model.h"
#include "loss.h"
#include "optim.h"
#include "checkpoint.h"
#include "corpus.h"

TrainConfig train_default_config(void) {
  TrainConfig c;
  c.out_model = "model.bin";
  c.out_tok = "tok.bin";
  c.num_merges = 300;
  c.steps = 1000;
  c.d_model = 128;
  c.n_heads = 4;
  c.n_layers = 3;
  c.d_ff = 0;          // 0 => 4 * d_model
  c.ctx = 64;
  c.lr = 3e-4f;
  c.seed = 1234;
  c.resume = NULL;
  c.checkpoint_every = 0;
  c.val_frac = 0.1f;
  c.sft = 0;
  // Recipe options default to OFF so existing runs reproduce bit-for-bit; opt in
  // with --weight-decay / --clip / --warmup / --min-lr.
  c.weight_decay = 0.0f;
  c.grad_clip = 0.0f;
  c.warmup = 0;
  c.lr_min = -1.0f;
  return c;
}

// A schedule is active once warmup or a cosine floor is requested; otherwise the
// LR stays exactly cfg->lr (the historical behaviour).
static int sched_on(const TrainConfig *cfg) {
  return cfg->warmup > 0 || cfg->lr_min >= 0.0f;
}

// Per-step LR for the configured schedule (constant when none is configured).
static float sched_lr(const TrainConfig *cfg, int step) {
  if (!sched_on(cfg)) return cfg->lr;
  float lo = (cfg->lr_min >= 0.0f) ? cfg->lr_min : 0.0f;
  return lr_cosine_warmup(cfg->lr, lo, step, cfg->warmup, cfg->steps);
}

// One-line summary of whichever recipe options are enabled (empty when none are).
static void print_recipe(const TrainConfig *cfg) {
  if (cfg->weight_decay > 0.0f) printf("  weight decay (AdamW, matrices only): %g\n", cfg->weight_decay);
  if (cfg->grad_clip > 0.0f)    printf("  grad-norm clip: %g\n", cfg->grad_clip);
  if (sched_on(cfg))
    printf("  LR schedule: warmup %d steps -> cosine %g -> %g\n",
           cfg->warmup, cfg->lr, (cfg->lr_min >= 0.0f) ? cfg->lr_min : 0.0f);
}

// ---------------------------------------------------------------------------
// Masked instruction tuning (SFT). Input file: one example per line, formatted
// "prompt<TAB>response". We build the sequence [prompt ++ response] and supervise
// ONLY the response tokens (prompt positions and end-padding get target -1 =
// ignored in cross-entropy). This is the real SFT loss: the model is trained to
// PRODUCE the response given the prompt, not to reconstruct the prompt. (Smoke
// test part [9] is the same idea, hand-built and deterministic.)
// ---------------------------------------------------------------------------
static int train_sft(char *text, const TrainConfig *cfg) {
  if (cfg->d_model % cfg->n_heads != 0) {
    fprintf(stderr, "error: d_model (%d) must be divisible by heads (%d)\n",
            cfg->d_model, cfg->n_heads);
    return 1;
  }
  srand(cfg->seed);

  Tokenizer tok = tokenizer_create_bytelevel();
  tokenizer_train(&tok, text, cfg->num_merges);

  // Upper bound on the number of examples = number of newlines + 1.
  int maxex = 1;
  for (char *p = text; *p; p++) if (*p == '\n') maxex++;
  int **P = (int**)malloc(maxex * sizeof(int*)), *PL = (int*)malloc(maxex * sizeof(int));
  int **R = (int**)malloc(maxex * sizeof(int*)), *RL = (int*)malloc(maxex * sizeof(int));
  int nex = 0, maxL = 0;

  char *line = strtok(text, "\n");
  while (line) {
    char *tab = strchr(line, '\t');
    if (tab) {
      *tab = '\0';
      char *prompt = line, *resp = tab + 1;
      size_t rl = strlen(resp);                  // strip a trailing CR (Windows files)
      if (rl && resp[rl - 1] == '\r') resp[rl - 1] = '\0';
      if (prompt[0] && resp[0]) {
        int t[4096], r[4096];
        int T = tokenizer_encode(&tok, prompt, t, 4096);
        int Rn = tokenizer_encode(&tok, resp, r, 4096);
        if (T >= 1 && Rn >= 1 && T + Rn <= 4096) {
          P[nex] = (int*)malloc(T * sizeof(int));  memcpy(P[nex], t, T * sizeof(int)); PL[nex] = T;
          R[nex] = (int*)malloc(Rn * sizeof(int)); memcpy(R[nex], r, Rn * sizeof(int)); RL[nex] = Rn;
          if (T + Rn + 1 > maxL) maxL = T + Rn + 1;   // +1 for the EOS token
          nex++;
        }
      }
    }
    line = strtok(NULL, "\n");
  }

  if (nex == 0) {
    fprintf(stderr, "error: SFT file has no 'prompt<TAB>response' lines\n");
    free(P); free(PL); free(R); free(RL); tokenizer_free(&tok); return 1;
  }

  // EOS lives at id = tokenizer vocab (the model gets one extra slot). Appending it
  // to every response and supervising its prediction teaches the model to STOP;
  // `generate` then halts when it samples any id >= the tokenizer's vocab.
  const int EOS = tok.vocab_size;

  // Build padded, masked training sequences (all length maxL): [prompt ++ response ++ EOS ++ pad].
  int **seq = (int**)malloc(nex * sizeof(int*));
  int **tgt = (int**)malloc(nex * sizeof(int*));
  for (int e = 0; e < nex; e++) {
    seq[e] = (int*)malloc(maxL * sizeof(int));
    tgt[e] = (int*)malloc(maxL * sizeof(int));
    int T = PL[e], Rn = RL[e];
    for (int j = 0; j < T; j++) seq[e][j] = P[e][j];
    for (int j = 0; j < Rn; j++) seq[e][T + j] = R[e][j];
    seq[e][T + Rn] = EOS;
    for (int j = T + Rn + 1; j < maxL; j++) seq[e][j] = 0;  // pad
    for (int j = 0; j < maxL; j++) tgt[e][j] = -1;          // mask everything...
    for (int j = T - 1; j <= T + Rn - 1; j++) tgt[e][j] = seq[e][j + 1];  // ...supervise response + EOS
  }

  const int d_ff = (cfg->d_ff > 0) ? cfg->d_ff : 4 * cfg->d_model;
  GPT model = gpt_create(tok.vocab_size + 1, cfg->d_model, cfg->n_heads, cfg->n_layers, d_ff, maxL);
  printf("SFT: %d examples | vocab %d (+EOS) | seq len (padded) %d | d_model=%d layers=%d lr=%g\n",
         nex, tok.vocab_size, maxL, cfg->d_model, cfg->n_layers, cfg->lr);

  GPTActs acts = gpt_acts_create(&model, maxL);
  int lshape[2] = {maxL, model.vocab_size};   // model vocab (= tok vocab + EOS)
  Tensor logits = tensor_create(lshape, 2), probs = tensor_create(lshape, 2);
  Tensor **params = (Tensor**)malloc(gpt_param_count(&model) * sizeof(Tensor*));
  int n_params = gpt_collect_params(&model, params);
  Adam opt = adamw_create(params, n_params, cfg->lr, cfg->weight_decay);
  print_recipe(cfg);

  const int steps = cfg->steps;
  const int report = steps < 10 ? 1 : steps / 10;
  for (int s = 0; s < steps; s++) {
    int e = rand() % nex;
    gpt_forward_train(&model, seq[e], maxL, &acts, &logits);
    float loss = cross_entropy_forward(&logits, tgt[e], &probs);   // averages over response tokens only
    tensor_zero_grad(&logits);
    cross_entropy_backward(&probs, tgt[e], &logits);
    gpt_backward(&model, seq[e], maxL, &acts, &logits);
    if (cfg->grad_clip > 0.0f) clip_grad_norm(params, n_params, cfg->grad_clip);
    adam_set_lr(&opt, sched_lr(cfg, s));
    adam_step(&opt, params, n_params);
    if (s % report == 0 || s == steps - 1)
      printf("  step %5d/%d  sft-loss %.4f\n", s, steps, loss);
  }

  int rc = 0;
  if (gpt_save(cfg->out_model, &model) != 0) { fprintf(stderr, "error saving model\n"); rc = 1; }
  if (tokenizer_save(&tok, cfg->out_tok) != 0) { fprintf(stderr, "error saving tokenizer\n"); rc = 1; }
  if (rc == 0) printf("saved SFT model -> %s, tokenizer -> %s\n", cfg->out_model, cfg->out_tok);

  for (int e = 0; e < nex; e++) { free(P[e]); free(R[e]); free(seq[e]); free(tgt[e]); }
  free(P); free(PL); free(R); free(RL); free(seq); free(tgt);
  free(params);
  adam_free(&opt); tensor_free(&logits); tensor_free(&probs);
  gpt_acts_free(&acts); gpt_free(&model); tokenizer_free(&tok);
  return rc;
}

// Average cross-entropy over `nwin` random windows whose start lies in [lo, hi-ctx).
// Forward-only (no gradients). Returns -1 if the region can't fit a window.
static float eval_split(GPT *m, const int *toks, int lo, int hi, int ctx,
                        GPTActs *acts, Tensor *logits, Tensor *probs,
                        int *inp, int *tgt, int nwin) {
  int count = hi - ctx - lo;   // number of valid window starts in [lo, hi-ctx)
  if (count < 1) return -1.0f;
  float sum = 0.0f;
  for (int w = 0; w < nwin; w++) {
    int start = lo + (rand() % count);
    for (int j = 0; j < ctx; j++) { inp[j] = toks[start + j]; tgt[j] = toks[start + j + 1]; }
    gpt_forward_train(m, inp, ctx, acts, logits);
    sum += cross_entropy_forward(logits, tgt, probs);
  }
  return sum / nwin;
}

int train_run(const char *data_path, const TrainConfig *cfg) {
  const int resuming = (cfg->resume != NULL);
  if (!resuming && cfg->d_model % cfg->n_heads != 0) {
    fprintf(stderr, "error: d_model (%d) must be divisible by heads (%d)\n",
            cfg->d_model, cfg->n_heads);
    return 1;
  }

  long text_len = 0;
  char *text = corpus_read(data_path, &text_len);   // file or directory of *.txt
  if (!text) { fprintf(stderr, "error: cannot read '%s'\n", data_path); return 1; }
  if (text_len < 8) { fprintf(stderr, "error: '%s' is too small\n", data_path); free(text); return 1; }

  // Masked instruction-tuning path ("prompt<TAB>response" lines); builds fresh (no resume).
  if (cfg->sft) {
    int rc = train_sft(text, cfg);
    free(text);
    return rc;
  }

  srand(cfg->seed);

  // 1. Tokenizer + model: either resume from a checkpoint or build fresh.
  Tokenizer tok;
  GPT model;
  if (resuming) {
    if (gpt_load(cfg->resume, &model) != 0) {
      fprintf(stderr, "error: cannot load model '%s'\n", cfg->resume); free(text); return 1;
    }
    if (tokenizer_load(&tok, cfg->out_tok) != 0) {
      fprintf(stderr, "error: cannot load tokenizer '%s' (needed to resume)\n", cfg->out_tok);
      gpt_free(&model); free(text); return 1;
    }
    if (tok.vocab_size != model.vocab_size) {
      fprintf(stderr, "error: tokenizer vocab %d != model vocab %d\n",
              tok.vocab_size, model.vocab_size);
      tokenizer_free(&tok); gpt_free(&model); free(text); return 1;
    }
    printf("resumed model %s (vocab %d, d_model %d, layers %d)\n",
           cfg->resume, model.vocab_size, model.d_model, model.n_layers);
  } else {
    tok = tokenizer_create_bytelevel();
    tokenizer_train(&tok, text, cfg->num_merges);
    const int d_ff = (cfg->d_ff > 0) ? cfg->d_ff : 4 * cfg->d_model;
    int mk = cfg->ctx;                       // model max_seq_len
    model = gpt_create(tok.vocab_size, cfg->d_model, cfg->n_heads, cfg->n_layers, d_ff, mk);
    printf("model: d_model=%d heads=%d layers=%d d_ff=%d ctx=%d lr=%g\n",
           cfg->d_model, cfg->n_heads, cfg->n_layers, d_ff, mk, cfg->lr);
  }

  // 2. Encode the corpus with the (loaded or freshly trained) tokenizer.
  int *toks = (int*)malloc((text_len + 1) * sizeof(int));
  int n_toks = tokenizer_encode(&tok, text, toks, (int)text_len);
  free(text);

  // Context window: bounded by the model's max_seq_len and the tokens available.
  int ctx = model.max_seq_len;
  if (n_toks - 1 < ctx) ctx = n_toks - 1;
  if (ctx < 2) { fprintf(stderr, "error: corpus too short to train\n"); free(toks); tokenizer_free(&tok); gpt_free(&model); return 1; }

  // 3. Train / validation split (val is the tail of the token stream).
  int n_val = (int)(n_toks * cfg->val_frac);
  int n_train = n_toks - n_val;
  int has_val = (n_val > ctx + 1) && (n_train > ctx + 1);
  if (!has_val) { n_train = n_toks; n_val = 0; }   // corpus too small to hold out
  printf("corpus: %ld bytes -> %d tokens | vocab %d | train %d / val %d | ctx %d\n",
         text_len, n_toks, tok.vocab_size, n_train, n_val, ctx);

  GPTActs acts = gpt_acts_create(&model, ctx);
  int lshape[2] = {ctx, tok.vocab_size};
  Tensor logits = tensor_create(lshape, 2);
  Tensor probs = tensor_create(lshape, 2);
  Tensor **params = (Tensor**)malloc(gpt_param_count(&model) * sizeof(Tensor*));
  int n_params = gpt_collect_params(&model, params);
  Adam opt = adamw_create(params, n_params, cfg->lr, cfg->weight_decay);
  print_recipe(cfg);

  int *inp = (int*)malloc(ctx * sizeof(int));
  int *tgt = (int*)malloc(ctx * sizeof(int));

  // 4. Training loop over random windows from the TRAIN region.
  const int steps = cfg->steps;
  const int report = steps < 10 ? 1 : steps / 10;
  const int train_starts = n_train - ctx;    // >= 1 (guaranteed above)
  for (int s = 0; s < steps; s++) {
    int start = rand() % train_starts;
    for (int j = 0; j < ctx; j++) { inp[j] = toks[start + j]; tgt[j] = toks[start + j + 1]; }
    gpt_forward_train(&model, inp, ctx, &acts, &logits);
    float loss = cross_entropy_forward(&logits, tgt, &probs);
    tensor_zero_grad(&logits);
    cross_entropy_backward(&probs, tgt, &logits);
    gpt_backward(&model, inp, ctx, &acts, &logits);
    if (cfg->grad_clip > 0.0f) clip_grad_norm(params, n_params, cfg->grad_clip);
    adam_set_lr(&opt, sched_lr(cfg, s));
    adam_step(&opt, params, n_params);

    if (s % report == 0 || s == steps - 1) {
      if (has_val) {
        float vl = eval_split(&model, toks, n_train, n_toks, ctx, &acts, &logits, &probs, inp, tgt, 8);
        printf("  step %5d/%d  train %.4f  val %.4f\n", s, steps, loss, vl);
      } else {
        printf("  step %5d/%d  train %.4f  val n/a\n", s, steps, loss);
      }
    }

    // Periodic checkpoint so long runs are safe / resumable. Saves are atomic
    // (temp + rename), so a failed/interrupted save keeps the previous good one.
    if (cfg->checkpoint_every > 0 && s > 0 && s % cfg->checkpoint_every == 0) {
      int cok = (gpt_save(cfg->out_model, &model) == 0);
      cok = (tokenizer_save(&tok, cfg->out_tok) == 0) && cok;
      if (cok) printf("  [checkpoint] saved at step %d -> %s\n", s, cfg->out_model);
      else fprintf(stderr, "  [checkpoint] FAILED at step %d (previous checkpoint kept)\n", s);
    }
  }

  // 5. Final save.
  int rc = 0;
  if (gpt_save(cfg->out_model, &model) != 0) { fprintf(stderr, "error saving model\n"); rc = 1; }
  if (tokenizer_save(&tok, cfg->out_tok) != 0) { fprintf(stderr, "error saving tokenizer\n"); rc = 1; }
  if (rc == 0) printf("saved model -> %s, tokenizer -> %s\n", cfg->out_model, cfg->out_tok);

  free(inp); free(tgt); free(toks);
  free(params);
  adam_free(&opt);
  tensor_free(&logits); tensor_free(&probs);
  gpt_acts_free(&acts);
  gpt_free(&model);
  tokenizer_free(&tok);
  return rc;
}
