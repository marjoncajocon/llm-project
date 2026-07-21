// train.c - Train a GPT on a text file
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
  return c;
}

int train_run(const char *data_path, const TrainConfig *cfg) {
  if (cfg->d_model % cfg->n_heads != 0) {
    fprintf(stderr, "error: d_model (%d) must be divisible by heads (%d)\n",
            cfg->d_model, cfg->n_heads);
    return 1;
  }
  const int d_ff = (cfg->d_ff > 0) ? cfg->d_ff : 4 * cfg->d_model;

  long text_len = 0;
  // Accepts a single file or a directory tree of *.txt files.
  char *text = corpus_read(data_path, &text_len);
  if (!text) { fprintf(stderr, "error: cannot read '%s'\n", data_path); return 1; }
  if (text_len < 8) { fprintf(stderr, "error: '%s' is too small\n", data_path); free(text); return 1; }

  // 1. Learn a BPE tokenizer on the corpus.
  Tokenizer tok = tokenizer_create_bytelevel();
  tokenizer_train(&tok, text, cfg->num_merges);
  const int vocab = tok.vocab_size;

  // 2. Encode the whole corpus into a token stream.
  int *toks = (int*)malloc((text_len + 1) * sizeof(int));
  int n_toks = tokenizer_encode(&tok, text, toks, (int)text_len);
  free(text);
  printf("corpus: %ld bytes -> %d tokens | vocab %d (%d merges)\n",
         text_len, n_toks, vocab, cfg->num_merges);

  // 3. Build the model. Context window is capped by the tokens available.
  int ctx = cfg->ctx;
  if (n_toks <= ctx + 1) ctx = n_toks - 1;
  if (ctx < 2) { fprintf(stderr, "error: corpus too short to train\n"); free(toks); tokenizer_free(&tok); return 1; }

  srand(cfg->seed);
  GPT model = gpt_create(vocab, cfg->d_model, cfg->n_heads, cfg->n_layers, d_ff, ctx);
  printf("model: d_model=%d heads=%d layers=%d d_ff=%d ctx=%d lr=%g\n",
         cfg->d_model, cfg->n_heads, cfg->n_layers, d_ff, ctx, cfg->lr);

  GPTActs acts = gpt_acts_create(&model, ctx);
  int lshape[2] = {ctx, vocab};
  Tensor logits = tensor_create(lshape, 2);
  Tensor probs = tensor_create(lshape, 2);
  Tensor *params[1024];
  int n_params = gpt_collect_params(&model, params);
  Adam opt = adam_create(params, n_params, cfg->lr);

  int *inp = (int*)malloc(ctx * sizeof(int));
  int *tgt = (int*)malloc(ctx * sizeof(int));

  // 4. Train on random context windows: predict each token from the previous ones.
  const int steps = cfg->steps;
  int report = steps < 10 ? 1 : steps / 10;
  for (int s = 0; s < steps; s++) {
    int start = rand() % (n_toks - ctx);
    for (int j = 0; j < ctx; j++) {
      inp[j] = toks[start + j];
      tgt[j] = toks[start + j + 1];
    }
    gpt_forward_train(&model, inp, ctx, &acts, &logits);
    float loss = cross_entropy_forward(&logits, tgt, &probs);
    tensor_zero_grad(&logits);
    cross_entropy_backward(&probs, tgt, &logits);
    gpt_backward(&model, inp, ctx, &acts, &logits);
    adam_step(&opt, params, n_params);

    if (s % report == 0 || s == steps - 1) {
      printf("  step %5d/%d  loss %.4f\n", s, steps, loss);
    }
  }

  // 5. Save the trained model + tokenizer.
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
