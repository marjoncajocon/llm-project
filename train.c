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

int train_run(const char *data_path, const char *out_model, const char *out_tok,
              int num_merges, int steps) {
  long text_len = 0;
  // Accepts a single file or a directory tree of *.txt files.
  char *text = corpus_read(data_path, &text_len);
  if (!text) { fprintf(stderr, "error: cannot read '%s'\n", data_path); return 1; }
  if (text_len < 8) { fprintf(stderr, "error: '%s' is too small\n", data_path); free(text); return 1; }

  // 1. Learn a BPE tokenizer on the corpus.
  Tokenizer tok = tokenizer_create_bytelevel();
  tokenizer_train(&tok, text, num_merges);
  const int vocab = tok.vocab_size;

  // 2. Encode the whole corpus into a token stream.
  int *toks = (int*)malloc((text_len + 1) * sizeof(int));
  int n_toks = tokenizer_encode(&tok, text, toks, (int)text_len);
  free(text);
  printf("corpus: %ld bytes -> %d tokens | vocab %d (%d merges)\n",
         text_len, n_toks, vocab, num_merges);

  // 3. Build the model (small, CPU-friendly). Context window <= tokens available.
  const int d_model = 64, n_heads = 4, n_layers = 2, d_ff = 256, max_seq = 128;
  int ctx = 32;
  if (n_toks <= ctx + 1) ctx = n_toks - 1;
  if (ctx < 2) { fprintf(stderr, "error: corpus too short to train\n"); free(toks); tokenizer_free(&tok); return 1; }

  srand(1234);
  GPT model = gpt_create(vocab, d_model, n_heads, n_layers, d_ff, max_seq);
  printf("model: d_model=%d heads=%d layers=%d d_ff=%d ctx=%d\n",
         d_model, n_heads, n_layers, d_ff, ctx);

  GPTActs acts = gpt_acts_create(&model, ctx);
  int lshape[2] = {ctx, vocab};
  Tensor logits = tensor_create(lshape, 2);
  Tensor probs = tensor_create(lshape, 2);
  Tensor *params[512];
  int n_params = gpt_collect_params(&model, params);
  Adam opt = adam_create(params, n_params, 1e-3f);

  int *inp = (int*)malloc(ctx * sizeof(int));
  int *tgt = (int*)malloc(ctx * sizeof(int));

  // 4. Train on random context windows: predict each token from the previous ones.
  float running = 0.0f;
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

    running += loss;
    if (s % report == 0 || s == steps - 1) {
      printf("  step %5d/%d  loss %.4f\n", s, steps, loss);
    }
  }
  (void)running;

  // 5. Save the trained model + tokenizer.
  int rc = 0;
  if (gpt_save(out_model, &model) != 0) { fprintf(stderr, "error saving model\n"); rc = 1; }
  if (tokenizer_save(&tok, out_tok) != 0) { fprintf(stderr, "error saving tokenizer\n"); rc = 1; }
  if (rc == 0) printf("saved model -> %s, tokenizer -> %s\n", out_model, out_tok);

  free(inp); free(tgt); free(toks);
  adam_free(&opt);
  tensor_free(&logits); tensor_free(&probs);
  gpt_acts_free(&acts);
  gpt_free(&model);
  tokenizer_free(&tok);
  return rc;
}
