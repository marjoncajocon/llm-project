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
  return c;
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
  Tensor *params[1024];
  int n_params = gpt_collect_params(&model, params);
  Adam opt = adam_create(params, n_params, cfg->lr);

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
    adam_step(&opt, params, n_params);

    if (s % report == 0 || s == steps - 1) {
      if (has_val) {
        float vl = eval_split(&model, toks, n_train, n_toks, ctx, &acts, &logits, &probs, inp, tgt, 8);
        printf("  step %5d/%d  train %.4f  val %.4f\n", s, steps, loss, vl);
      } else {
        printf("  step %5d/%d  train %.4f  val n/a\n", s, steps, loss);
      }
    }

    // Periodic checkpoint so long runs are safe / resumable.
    if (cfg->checkpoint_every > 0 && s > 0 && s % cfg->checkpoint_every == 0) {
      gpt_save(cfg->out_model, &model);
      tokenizer_save(&tok, cfg->out_tok);
      printf("  [checkpoint] saved at step %d -> %s\n", s, cfg->out_model);
    }
  }

  // 5. Final save.
  int rc = 0;
  if (gpt_save(cfg->out_model, &model) != 0) { fprintf(stderr, "error saving model\n"); rc = 1; }
  if (tokenizer_save(&tok, cfg->out_tok) != 0) { fprintf(stderr, "error saving tokenizer\n"); rc = 1; }
  if (rc == 0) printf("saved model -> %s, tokenizer -> %s\n", cfg->out_model, cfg->out_tok);

  free(inp); free(tgt); free(toks);
  adam_free(&opt);
  tensor_free(&logits); tensor_free(&probs);
  gpt_acts_free(&acts);
  gpt_free(&model);
  tokenizer_free(&tok);
  return rc;
}
