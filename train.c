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
#include "llama.h"
#include "llamatrain.h"
#include "hftok.h"

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
  c.batch = 1;
  c.n_kv = 0;              // 0 => n_heads (plain multi-head, no GQA sharing)
  c.rope_base = 10000.0f;
  c.lora_r = 0;            // 0 => full fine-tune (no adapters)
  c.lora_alpha = 0.0f;     // 0 => alpha = r, i.e. scale 1
  c.hf_tok = NULL;         // NULL => train/load our own byte-level BPE
  return c;
}

// ---------------------------------------------------------------------------
// Fine-tuning a REAL model has to use that model's OWN tokenizer, so the SFT
// path accepts either. A tagged union keeps one training loop with concrete
// types (the project bans untyped void* in APIs) instead of duplicating it.
// ---------------------------------------------------------------------------
typedef struct {
  int is_hf;
  Tokenizer bpe;   // valid when !is_hf
  HFTok hf;        // valid when  is_hf
} AnyTok;

static int anytok_encode(const AnyTok *t, const char *s, int *ids, int max_ids) {
  return t->is_hf ? hftok_encode(&t->hf, s, ids, max_ids)
                  : tokenizer_encode(&t->bpe, s, ids, max_ids);
}

static int anytok_vocab(const AnyTok *t) {
  return t->is_hf ? t->hf.n_vocab : t->bpe.vocab_size;
}

// End-of-response id. A real model already HAS an EOS token in its vocabulary;
// our byte-level BPE does not, so one extra slot is appended past its vocab.
static int anytok_eos(const AnyTok *t) {
  if (!t->is_hf) return t->bpe.vocab_size;
  return (t->hf.eos >= 0) ? t->hf.eos : t->hf.n_vocab - 1;
}

// Model vocabulary needed to host that EOS id.
static int anytok_model_vocab(const AnyTok *t) {
  return t->is_hf ? t->hf.n_vocab : t->bpe.vocab_size + 1;
}

static void anytok_free(AnyTok *t) {
  if (t->is_hf) hftok_free(&t->hf); else tokenizer_free(&t->bpe);
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

// Micro-batches per optimizer step (>= 1).
static int batch_of(const TrainConfig *cfg) {
  return (cfg->batch > 1) ? cfg->batch : 1;
}

// One-line summary of whichever recipe options are enabled (empty when none are).
static void print_recipe(const TrainConfig *cfg) {
  if (cfg->batch > 1) printf("  batch: %d sequences per step (gradient accumulation)\n", cfg->batch);
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
  const int B = batch_of(cfg);
  for (int s = 0; s < steps; s++) {
    gpt_zero_grad(&model);
    float loss = 0.0f;
    for (int b = 0; b < B; b++) {
      int e = rand() % nex;
      gpt_forward_train(&model, seq[e], maxL, &acts, &logits);
      loss += cross_entropy_forward(&logits, tgt[e], &probs);   // averages over response tokens only
      tensor_zero_grad(&logits);
      cross_entropy_backward(&probs, tgt[e], &logits);
      gpt_backward_accum(&model, seq[e], maxL, &acts, &logits);
    }
    loss /= (float)B;
    if (B > 1) scale_grads(params, n_params, 1.0f / (float)B);
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
  const int B = batch_of(cfg);
  for (int s = 0; s < steps; s++) {
    gpt_zero_grad(&model);
    float loss = 0.0f;
    for (int b = 0; b < B; b++) {
      int start = rand() % train_starts;
      for (int j = 0; j < ctx; j++) { inp[j] = toks[start + j]; tgt[j] = toks[start + j + 1]; }
      gpt_forward_train(&model, inp, ctx, &acts, &logits);
      loss += cross_entropy_forward(&logits, tgt, &probs);
      tensor_zero_grad(&logits);
      cross_entropy_backward(&probs, tgt, &logits);
      gpt_backward_accum(&model, inp, ctx, &acts, &logits);
    }
    loss /= (float)B;
    if (B > 1) scale_grads(params, n_params, 1.0f / (float)B);
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

// ===========================================================================
// LLaMA-family training driver (modern architecture: RMSNorm + RoPE + GQA +
// SwiGLU). Same corpus/val/checkpoint/resume/SFT/recipe behaviour as the GPT
// path above; the model type and its forward/backward differ, which C can't
// abstract over without void* (banned here), so the loop is written out.
// ===========================================================================

// Geometry shared by every entry point: head_dim = d_model / n_heads, and n_kv
// defaults to n_heads (plain multi-head). Returns 0 if the config is invalid.
static int llama_geometry(const TrainConfig *cfg, int *head_dim, int *n_kv, int *d_ff) {
  if (cfg->n_heads <= 0 || cfg->d_model % cfg->n_heads != 0) {
    fprintf(stderr, "error: d_model (%d) must be divisible by heads (%d)\n",
            cfg->d_model, cfg->n_heads);
    return 0;
  }
  *n_kv = (cfg->n_kv > 0) ? cfg->n_kv : cfg->n_heads;
  if (*n_kv > cfg->n_heads || cfg->n_heads % *n_kv != 0) {
    fprintf(stderr, "error: heads (%d) must be a multiple of kv heads (%d)\n",
            cfg->n_heads, *n_kv);
    return 0;
  }
  *head_dim = cfg->d_model / cfg->n_heads;
  if (*head_dim % 2 != 0) {
    fprintf(stderr, "error: head_dim (%d) must be even (RoPE rotates feature pairs)\n", *head_dim);
    return 0;
  }
  *d_ff = (cfg->d_ff > 0) ? cfg->d_ff : 4 * cfg->d_model;
  return 1;
}

// Forward-only mean loss over `nwin` random windows starting in [lo, hi-ctx).
// `adapters`/`ladapt` may be NULL (full fine-tune); when set, the val loss
// includes the LoRA correction, as it must to be comparable with the train loss.
static float llama_eval_split(LlamaModel *m, const int *toks, int lo, int hi, int ctx,
                              LlamaActs *acts, Tensor *logits, Tensor *probs,
                              int *inp, int *tgt, int nwin,
                              const LlamaLora *adapters, LlamaLoraActs *ladapt) {
  int count = hi - ctx - lo;
  if (count < 1) return -1.0f;
  float sum = 0.0f;
  for (int w = 0; w < nwin; w++) {
    int start = lo + (rand() % count);
    for (int j = 0; j < ctx; j++) { inp[j] = toks[start + j]; tgt[j] = toks[start + j + 1]; }
    llama_forward_train_lora(m, inp, ctx, acts, logits, adapters, ladapt);
    sum += cross_entropy_forward(logits, tgt, probs);
  }
  return sum / nwin;
}

// LoRA scale convention: alpha defaults to r, i.e. scale 1.
static float lora_alpha_of(const TrainConfig *cfg) {
  return (cfg->lora_alpha > 0.0f) ? cfg->lora_alpha : (float)cfg->lora_r;
}

// Copy every weight from a loaded (inference, grad == NULL) model into a trainable
// one. The two agree on every dimension EXCEPT possibly the vocabulary: SFT needs
// one extra slot for EOS, so a base trained with vocab V is grown to V+1 here.
// wte is (vocab, d_model) — rows, so the old block is contiguous — but lm_head is
// (d_model, vocab), where vocab is the STRIDE, so it must be copied row by row.
static void copy_weights_grow(LlamaModel *dst, LlamaModel *src) {
  Tensor **s = (Tensor**)malloc((size_t)llama_param_count(src) * sizeof(Tensor*));
  Tensor **d = (Tensor**)malloc((size_t)llama_param_count(dst) * sizeof(Tensor*));
  int ns = llama_collect_params(src, s), nd = llama_collect_params(dst, d);
  for (int i = 0; i < ns && i < nd; i++) {
    if (s[i]->size == d[i]->size) {
      memcpy(d[i]->data, s[i]->data, s[i]->size * sizeof(float));
    } else if (s[i]->ndim == 2 && d[i]->ndim == 2 && s[i]->shape[0] == d[i]->shape[0]) {
      // Same rows, wider columns (lm_head): copy each row's prefix.
      const int rows = s[i]->shape[0], sc = s[i]->shape[1], dc = d[i]->shape[1];
      for (int r = 0; r < rows; r++)
        memcpy(&d[i]->data[(size_t)r * dc], &s[i]->data[(size_t)r * sc], (size_t)sc * sizeof(float));
    } else {
      // Fewer rows, same columns (wte): the old block is contiguous.
      memcpy(d[i]->data, s[i]->data, s[i]->size * sizeof(float));
    }
  }
  free(s); free(d);
}

// Masked instruction tuning for the LLaMA path — same "prompt<TAB>response"
// format, EOS slot and -1 masking as train_sft() above.
static int llama_train_sft(char *text, const TrainConfig *cfg) {
  const int resuming = (cfg->resume != NULL);
  int head_dim = 0, n_kv = 0, d_ff = 0;
  if (!resuming && !llama_geometry(cfg, &head_dim, &n_kv, &d_ff)) return 1;
  srand(cfg->seed);

  // Fine-tuning MUST reuse the base model's tokenizer — a freshly trained one
  // would assign different ids to the same text, making the pretrained
  // embeddings meaningless. Fresh runs train a tokenizer on the SFT text.
  AnyTok tok;
  tok.is_hf = 0;
  LlamaModel loaded;
  if (cfg->hf_tok) {                        // a real model's own vocabulary
    if (hftok_load(&tok.hf, cfg->hf_tok) != 0) {
      fprintf(stderr, "error: cannot load HF tokenizer '%s'\n", cfg->hf_tok);
      return 1;
    }
    tok.is_hf = 1;
    printf("tokenizer: %s (%d tokens, eos %d)\n", cfg->hf_tok, tok.hf.n_vocab, tok.hf.eos);
  }
  if (resuming) {
    if (llama_model_load(cfg->resume, &loaded) != 0) {
      fprintf(stderr, "error: cannot load model '%s'\n", cfg->resume);
      if (tok.is_hf) anytok_free(&tok);
      return 1;
    }
    if (!tok.is_hf && tokenizer_load(&tok.bpe, cfg->out_tok) != 0) {
      fprintf(stderr, "error: cannot load tokenizer '%s' (needed to resume)\n", cfg->out_tok);
      llama_model_free(&loaded);
      return 1;
    }
    if (loaded.vocab < anytok_vocab(&tok)) {
      fprintf(stderr, "error: model vocab %d < tokenizer vocab %d\n",
              loaded.vocab, anytok_vocab(&tok));
      anytok_free(&tok); llama_model_free(&loaded);
      return 1;
    }
    printf("resumed LLMA model %s (vocab %d, d_model %d, layers %d, heads %d, kv %d)\n",
           cfg->resume, loaded.vocab, loaded.d_model, loaded.n_layers, loaded.n_heads, loaded.n_kv);
  } else if (!tok.is_hf) {
    tok.bpe = tokenizer_create_bytelevel();
    tokenizer_train(&tok.bpe, text, cfg->num_merges);
  }

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
      size_t rl = strlen(resp);
      if (rl && resp[rl - 1] == '\r') resp[rl - 1] = '\0';
      if (prompt[0] && resp[0]) {
        int t[4096], r[4096];
        int T = anytok_encode(&tok, prompt, t, 4096);
        int Rn = anytok_encode(&tok, resp, r, 4096);
        if (T >= 1 && Rn >= 1 && T + Rn <= 4096) {
          P[nex] = (int*)malloc(T * sizeof(int));  memcpy(P[nex], t, T * sizeof(int)); PL[nex] = T;
          R[nex] = (int*)malloc(Rn * sizeof(int)); memcpy(R[nex], r, Rn * sizeof(int)); RL[nex] = Rn;
          if (T + Rn + 1 > maxL) maxL = T + Rn + 1;
          nex++;
        }
      }
    }
    line = strtok(NULL, "\n");
  }

  if (nex == 0) {
    fprintf(stderr, "error: SFT file has no 'prompt<TAB>response' lines\n");
    free(P); free(PL); free(R); free(RL); anytok_free(&tok);
    if (resuming) llama_model_free(&loaded);
    return 1;
  }
  if (resuming && maxL > loaded.max_seq) {
    fprintf(stderr, "error: SFT sequences need %d positions but the base model's ctx is %d\n",
            maxL, loaded.max_seq);
    for (int e = 0; e < nex; e++) { free(P[e]); free(R[e]); }
    free(P); free(PL); free(R); free(RL); anytok_free(&tok); llama_model_free(&loaded);
    return 1;
  }

  const int EOS = anytok_eos(&tok);
  int **seq = (int**)malloc(nex * sizeof(int*));
  int **tgt = (int**)malloc(nex * sizeof(int*));
  for (int e = 0; e < nex; e++) {
    seq[e] = (int*)malloc(maxL * sizeof(int));
    tgt[e] = (int*)malloc(maxL * sizeof(int));
    int T = PL[e], Rn = RL[e];
    for (int j = 0; j < T; j++) seq[e][j] = P[e][j];
    for (int j = 0; j < Rn; j++) seq[e][T + j] = R[e][j];
    seq[e][T + Rn] = EOS;
    for (int j = T + Rn + 1; j < maxL; j++) seq[e][j] = 0;
    for (int j = 0; j < maxL; j++) tgt[e][j] = -1;
    for (int j = T - 1; j <= T + Rn - 1; j++) tgt[e][j] = seq[e][j + 1];
  }

  LlamaModel model;
  if (resuming) {
    // Grow the vocab by one for the EOS slot if the base doesn't already have it,
    // keeping every other dimension (and max_seq, so the KV geometry still fits).
    // A real model already has an EOS in its vocabulary; our byte BPE does not,
    // so the vocab grows by one slot in that case (and only that case).
    const int want = anytok_model_vocab(&tok);
    const int vocab = (loaded.vocab > want) ? loaded.vocab : want;
    model = llama_model_create_train(vocab, loaded.d_model, loaded.n_layers, loaded.n_heads,
                                     loaded.n_kv, loaded.head_dim, loaded.d_ff,
                                     loaded.max_seq, loaded.rope_base, loaded.eps);
    llama_model_randomize(&model);          // any new row/column starts random
    copy_weights_grow(&model, &loaded);     // ...then every pretrained weight lands on top
    llama_model_free(&loaded);
    printf("LLaMA SFT (fine-tune): %d examples | vocab %d -> %d | seq len %d | d_model=%d layers=%d heads=%d kv=%d eos=%d lr=%g\n",
           nex, anytok_vocab(&tok), vocab, maxL, model.d_model, model.n_layers,
           model.n_heads, model.n_kv, EOS, cfg->lr);
  } else {
    model = llama_model_create_train(anytok_model_vocab(&tok), cfg->d_model, cfg->n_layers,
                                     cfg->n_heads, n_kv, head_dim, d_ff, maxL,
                                     cfg->rope_base, 1e-5f);
    llama_model_randomize(&model);
    printf("LLaMA SFT: %d examples | vocab %d (+EOS) | seq len (padded) %d | d_model=%d layers=%d heads=%d kv=%d lr=%g\n",
           nex, anytok_vocab(&tok), maxL, cfg->d_model, cfg->n_layers, cfg->n_heads, n_kv, cfg->lr);
  }

  LlamaActs acts = llama_acts_create(&model, maxL);
  int lshape[2] = {maxL, model.vocab};
  Tensor logits = tensor_create(lshape, 2), probs = tensor_create(lshape, 2);

  // LoRA: freeze the base and hand Adam only the adapters.
  const int use_lora = (cfg->lora_r > 0);
  LlamaLora lora; LlamaLoraActs lacts;
  Tensor **params; int n_params;
  if (use_lora) {
    lora = llama_lora_create(&model, cfg->lora_r, lora_alpha_of(cfg));
    llama_lora_init(&lora);
    lacts = llama_lora_acts_create(&model, &lora, maxL);
    params = (Tensor**)malloc((size_t)llama_lora_param_count(&lora) * sizeof(Tensor*));
    n_params = llama_lora_collect_params(&lora, params);
    printf("  LoRA: rank %d alpha %g on Wq/Wk/Wv/Wo, base FROZEN (%d adapter tensors)\n",
           cfg->lora_r, lora_alpha_of(cfg), n_params);
  } else {
    params = (Tensor**)malloc((size_t)llama_param_count(&model) * sizeof(Tensor*));
    n_params = llama_collect_params(&model, params);
  }
  Adam opt = adamw_create(params, n_params, cfg->lr, cfg->weight_decay);
  print_recipe(cfg);

  const int steps = cfg->steps;
  const int report = steps < 10 ? 1 : steps / 10;
  const int B = batch_of(cfg);
  for (int s = 0; s < steps; s++) {
    llama_zero_grad(&model);
    if (use_lora) llama_lora_zero_grad(&lora);
    float loss = 0.0f;
    for (int b = 0; b < B; b++) {
      int e = rand() % nex;
      llama_forward_train_lora(&model, seq[e], maxL, &acts, &logits,
                               use_lora ? &lora : NULL, use_lora ? &lacts : NULL);
      loss += cross_entropy_forward(&logits, tgt[e], &probs);
      tensor_zero_grad(&logits);
      cross_entropy_backward(&probs, tgt[e], &logits);
      llama_backward_accum(&model, seq[e], maxL, &acts, &logits,
                           use_lora ? &lora : NULL, use_lora ? &lacts : NULL);
    }
    loss /= (float)B;
    if (B > 1) scale_grads(params, n_params, 1.0f / (float)B);
    if (cfg->grad_clip > 0.0f) clip_grad_norm(params, n_params, cfg->grad_clip);
    adam_set_lr(&opt, sched_lr(cfg, s));
    adam_step(&opt, params, n_params);
    if (s % report == 0 || s == steps - 1)
      printf("  step %5d/%d  sft-loss %.4f\n", s, steps, loss);
  }

  // Fold the adapters into the weights so the result saves as an ordinary LLMA
  // checkpoint that `transformer llama` runs with no adapter support.
  if (use_lora) {
    llama_lora_merge(&model, &lora);
    printf("  LoRA: merged adapters into the base weights\n");
  }

  int rc = 0;
  if (llama_model_save(cfg->out_model, &model) != 0) { fprintf(stderr, "error saving model\n"); rc = 1; }
  // An HF vocabulary is the model's own file and is not rewritten — only our
  // byte-level BPE is owned by the run and needs saving alongside the weights.
  if (!tok.is_hf && tokenizer_save(&tok.bpe, cfg->out_tok) != 0) {
    fprintf(stderr, "error saving tokenizer\n"); rc = 1;
  }
  if (rc == 0) {
    if (tok.is_hf) printf("saved LLMA SFT model -> %s (tokenizer: %s, unchanged)\n",
                          cfg->out_model, cfg->hf_tok);
    else printf("saved LLMA SFT model -> %s, tokenizer -> %s\n", cfg->out_model, cfg->out_tok);
  }

  for (int e = 0; e < nex; e++) { free(P[e]); free(R[e]); free(seq[e]); free(tgt[e]); }
  free(P); free(PL); free(R); free(RL); free(seq); free(tgt);
  free(params);
  if (use_lora) { llama_lora_acts_free(&lacts); llama_lora_free(&lora); }
  adam_free(&opt); tensor_free(&logits); tensor_free(&probs);
  llama_acts_free(&acts); llama_model_free(&model); anytok_free(&tok);
  return rc;
}

int llama_train_run(const char *data_path, const TrainConfig *cfg) {
  const int resuming = (cfg->resume != NULL);
  int head_dim = 0, n_kv = 0, d_ff = 0;
  if (!resuming && !llama_geometry(cfg, &head_dim, &n_kv, &d_ff)) return 1;

  long text_len = 0;
  char *text = corpus_read(data_path, &text_len);
  if (!text) { fprintf(stderr, "error: cannot read '%s'\n", data_path); return 1; }
  if (text_len < 8) { fprintf(stderr, "error: '%s' is too small\n", data_path); free(text); return 1; }

  if (cfg->sft) {
    int rc = llama_train_sft(text, cfg);
    free(text);
    return rc;
  }

  srand(cfg->seed);

  // 1. Tokenizer + model: resume an LLMA checkpoint, or build fresh.
  Tokenizer tok;
  LlamaModel model;
  if (resuming) {
    if (llama_model_load(cfg->resume, &model) != 0) {
      fprintf(stderr, "error: cannot load model '%s'\n", cfg->resume); free(text); return 1;
    }
    if (tokenizer_load(&tok, cfg->out_tok) != 0) {
      fprintf(stderr, "error: cannot load tokenizer '%s' (needed to resume)\n", cfg->out_tok);
      llama_model_free(&model); free(text); return 1;
    }
    if (tok.vocab_size != model.vocab) {
      fprintf(stderr, "error: tokenizer vocab %d != model vocab %d\n", tok.vocab_size, model.vocab);
      tokenizer_free(&tok); llama_model_free(&model); free(text); return 1;
    }
    // A checkpoint loaded for INFERENCE has no gradient arrays, so it cannot be
    // trained: rebuild a trainable model and copy the weights across.
    LlamaModel t = llama_model_create_train(model.vocab, model.d_model, model.n_layers,
                                            model.n_heads, model.n_kv, model.head_dim,
                                            model.d_ff, model.max_seq, model.rope_base, model.eps);
    copy_weights_grow(&t, &model);
    llama_model_free(&model);
    model = t;
    printf("resumed LLMA model %s (vocab %d, d_model %d, layers %d, heads %d, kv %d)\n",
           cfg->resume, model.vocab, model.d_model, model.n_layers, model.n_heads, model.n_kv);
  } else {
    tok = tokenizer_create_bytelevel();
    tokenizer_train(&tok, text, cfg->num_merges);
    model = llama_model_create_train(tok.vocab_size, cfg->d_model, cfg->n_layers,
                                     cfg->n_heads, n_kv, head_dim, d_ff, cfg->ctx,
                                     cfg->rope_base, 1e-5f);
    llama_model_randomize(&model);
    printf("LLaMA model: d_model=%d heads=%d kv=%d head_dim=%d layers=%d d_ff=%d ctx=%d rope_base=%g lr=%g\n",
           cfg->d_model, cfg->n_heads, n_kv, head_dim, cfg->n_layers, d_ff, cfg->ctx,
           cfg->rope_base, cfg->lr);
  }

  // 2. Encode the corpus.
  int *toks = (int*)malloc((text_len + 1) * sizeof(int));
  int n_toks = tokenizer_encode(&tok, text, toks, (int)text_len);
  free(text);

  int ctx = model.max_seq;
  if (n_toks - 1 < ctx) ctx = n_toks - 1;
  if (ctx < 2) {
    fprintf(stderr, "error: corpus too short to train\n");
    free(toks); tokenizer_free(&tok); llama_model_free(&model); return 1;
  }

  // 3. Train / validation split (val is the tail of the token stream).
  int n_val = (int)(n_toks * cfg->val_frac);
  int n_train = n_toks - n_val;
  int has_val = (n_val > ctx + 1) && (n_train > ctx + 1);
  if (!has_val) { n_train = n_toks; n_val = 0; }
  printf("corpus: %ld bytes -> %d tokens | vocab %d | train %d / val %d | ctx %d\n",
         text_len, n_toks, tok.vocab_size, n_train, n_val, ctx);

  LlamaActs acts = llama_acts_create(&model, ctx);
  int lshape[2] = {ctx, model.vocab};
  Tensor logits = tensor_create(lshape, 2), probs = tensor_create(lshape, 2);

  // LoRA: freeze the base and optimize only the adapters.
  const int use_lora = (cfg->lora_r > 0);
  LlamaLora lora; LlamaLoraActs lacts;
  Tensor **params; int n_params;
  if (use_lora) {
    lora = llama_lora_create(&model, cfg->lora_r, lora_alpha_of(cfg));
    llama_lora_init(&lora);
    lacts = llama_lora_acts_create(&model, &lora, ctx);
    params = (Tensor**)malloc((size_t)llama_lora_param_count(&lora) * sizeof(Tensor*));
    n_params = llama_lora_collect_params(&lora, params);
    printf("  LoRA: rank %d alpha %g on Wq/Wk/Wv/Wo, base FROZEN (%d adapter tensors)\n",
           cfg->lora_r, lora_alpha_of(cfg), n_params);
    if (cfg->checkpoint_every > 0)
      printf("  LoRA: --checkpoint-every ignored (adapters are merged only at the end)\n");
  } else {
    params = (Tensor**)malloc((size_t)llama_param_count(&model) * sizeof(Tensor*));
    n_params = llama_collect_params(&model, params);
  }
  Adam opt = adamw_create(params, n_params, cfg->lr, cfg->weight_decay);
  print_recipe(cfg);

  int *inp = (int*)malloc(ctx * sizeof(int));
  int *tgt = (int*)malloc(ctx * sizeof(int));

  // 4. Training loop over random windows from the TRAIN region.
  const int steps = cfg->steps;
  const int report = steps < 10 ? 1 : steps / 10;
  const int train_starts = n_train - ctx;
  const int B = batch_of(cfg);
  for (int s = 0; s < steps; s++) {
    // Accumulate B micro-batches, then take ONE optimizer step on their mean.
    llama_zero_grad(&model);
    if (use_lora) llama_lora_zero_grad(&lora);
    float loss = 0.0f;
    for (int b = 0; b < B; b++) {
      int start = rand() % train_starts;
      for (int j = 0; j < ctx; j++) { inp[j] = toks[start + j]; tgt[j] = toks[start + j + 1]; }
      llama_forward_train_lora(&model, inp, ctx, &acts, &logits,
                               use_lora ? &lora : NULL, use_lora ? &lacts : NULL);
      loss += cross_entropy_forward(&logits, tgt, &probs);
      tensor_zero_grad(&logits);
      cross_entropy_backward(&probs, tgt, &logits);
      llama_backward_accum(&model, inp, ctx, &acts, &logits,
                           use_lora ? &lora : NULL, use_lora ? &lacts : NULL);
    }
    loss /= (float)B;
    if (B > 1) scale_grads(params, n_params, 1.0f / (float)B);
    if (cfg->grad_clip > 0.0f) clip_grad_norm(params, n_params, cfg->grad_clip);
    adam_set_lr(&opt, sched_lr(cfg, s));
    adam_step(&opt, params, n_params);

    if (s % report == 0 || s == steps - 1) {
      if (has_val) {
        float vl = llama_eval_split(&model, toks, n_train, n_toks, ctx, &acts, &logits, &probs, inp, tgt, 8,
                                    use_lora ? &lora : NULL, use_lora ? &lacts : NULL);
        printf("  step %5d/%d  train %.4f  val %.4f\n", s, steps, loss, vl);
      } else {
        printf("  step %5d/%d  train %.4f  val n/a\n", s, steps, loss);
      }
    }

    // Intermediate checkpoints are skipped under LoRA: merging mid-run would zero
    // B and desync Adam's moments for it, so adapters are folded in only at the end.
    if (!use_lora && cfg->checkpoint_every > 0 && s > 0 && s % cfg->checkpoint_every == 0) {
      int cok = (llama_model_save(cfg->out_model, &model) == 0);
      cok = (tokenizer_save(&tok, cfg->out_tok) == 0) && cok;
      if (cok) printf("  [checkpoint] saved at step %d -> %s\n", s, cfg->out_model);
      else fprintf(stderr, "  [checkpoint] FAILED at step %d (previous checkpoint kept)\n", s);
    }
  }

  // 5. Fold adapters in (if any), then save.
  if (use_lora) {
    llama_lora_merge(&model, &lora);
    printf("  LoRA: merged adapters into the base weights\n");
  }
  int rc = 0;
  if (llama_model_save(cfg->out_model, &model) != 0) { fprintf(stderr, "error saving model\n"); rc = 1; }
  if (tokenizer_save(&tok, cfg->out_tok) != 0) { fprintf(stderr, "error saving tokenizer\n"); rc = 1; }
  if (rc == 0) printf("saved LLMA model -> %s, tokenizer -> %s\n", cfg->out_model, cfg->out_tok);

  free(inp); free(tgt); free(toks);
  free(params);
  if (use_lora) { llama_lora_acts_free(&lacts); llama_lora_free(&lora); }
  adam_free(&opt);
  tensor_free(&logits); tensor_free(&probs);
  llama_acts_free(&acts);
  llama_model_free(&model);
  tokenizer_free(&tok);
  return rc;
}
