// main.c - Minimal GPT-2 Style Transformer in Pure C11 (demo entry point)
// Build: make        (see README.md and Makefile)
// Or:    gcc -std=c11 -O3 -march=native -o transformer *.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tensor.h"
#include "loss.h"
#include "optim.h"
#include "sampler.h"
#include "model.h"
#include "checkpoint.h"
#include "tokenizer.h"

// Full-model loss: training forward + cross-entropy. Fills logits and probs.
static float gpt_loss(GPT *m, GPTActs *a, const int *ids, int seq_len,
                      const int *targets, Tensor *logits, Tensor *probs) {
  gpt_forward_train(m, ids, seq_len, a, logits);
  return cross_entropy_forward(logits, targets, probs);
}

static int run_demo(void) {
  printf("=== Minimal GPT-2 Style Transformer Demo ===\n\n");
  int all_ok = 1;
  srand(42);

  // --------------------------------------------------------------------------
  // Part 1: tokenizer + checkpoint round-trips (step 4 code, condensed)
  // --------------------------------------------------------------------------
  printf("[1] Tokenizer + checkpoint round-trips\n");
  Tokenizer tok = tokenizer_create_bytelevel();
  int ll = tokenizer_add_merge(&tok, 'l', 'l');
  tokenizer_add_merge(&tok, 'e', ll);
  int tids[64];
  int n_tids = tokenizer_encode(&tok, "hello hello", tids, 64);
  char decoded[128];
  tokenizer_decode(&tok, tids, n_tids, decoded, sizeof(decoded));
  int tok_ok = (strcmp("hello hello", decoded) == 0);
  tokenizer_free(&tok);
  all_ok &= tok_ok;

  GPT probe = gpt_create(16, 8, 2, 2, 32, 16);
  int pseq[3] = {1, 5, 9};
  int plshape[2] = {3, 16};
  Tensor pla = tensor_create(plshape, 2), plb = tensor_create(plshape, 2);
  gpt_forward(&probe, pseq, 3, &pla);
  int ckpt_ok = (gpt_save("model.bin", &probe) == 0);
  GPT probe2;
  ckpt_ok = ckpt_ok && (gpt_load("model.bin", &probe2) == 0);
  if (ckpt_ok) {
    gpt_forward(&probe2, pseq, 3, &plb);
    ckpt_ok = (memcmp(pla.data, plb.data, pla.size * sizeof(float)) == 0);
    gpt_free(&probe2);
  }
  tensor_free(&pla); tensor_free(&plb); gpt_free(&probe); remove("model.bin");
  printf("    tokenizer: %s | checkpoint: %s\n\n",
         tok_ok ? "OK" : "FAIL", ckpt_ok ? "OK" : "FAIL");
  all_ok &= ckpt_ok;

  // --------------------------------------------------------------------------
  // Part 2: build the full GPT + a fixed next-token task
  // --------------------------------------------------------------------------
  const int vocab = 16, d_model = 8, n_heads = 2, n_layers = 2, d_ff = 32, max_seq = 16;
  const int seq_len = 6;
  GPT model = gpt_create(vocab, d_model, n_heads, n_layers, d_ff, max_seq);
  printf("[2] Full GPT training  (vocab=%d d_model=%d heads=%d layers=%d)\n",
         vocab, d_model, n_heads, n_layers);

  // Fixed token sequence; predict the next token at each position.
  int tokens[7] = {1, 2, 4, 8, 3, 5, 7};
  const int *ids = tokens;            // inputs  = tokens[0..seq_len-1]
  const int *targets = tokens + 1;    // targets = tokens[1..seq_len]

  GPTActs acts = gpt_acts_create(&model, seq_len);
  int lshape[2] = {seq_len, vocab};
  Tensor logits = tensor_create(lshape, 2);
  Tensor probs = tensor_create(lshape, 2);

  Tensor *params[64];
  int n_params = gpt_collect_params(&model, params);

  // --------------------------------------------------------------------------
  // Part 3: end-to-end gradient check (analytic vs numerical), whole model
  // --------------------------------------------------------------------------
  gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
  tensor_zero_grad(&logits);
  cross_entropy_backward(&probs, targets, &logits);
  gpt_backward(&model, ids, seq_len, &acts, &logits);  // analytic grads into params

  const float eps = 1e-2f;
  float max_rel = 0.0f, max_rel_cond = 0.0f;
  for (int p = 0; p < n_params; p++) {
    Tensor *param = params[p];
    int step = param->size > 2 ? param->size / 2 : 1;
    for (int k = 0; k < param->size; k += step) {
      float base = param->data[k];
      param->data[k] = base + eps;
      float lp = gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
      param->data[k] = base - eps;
      float lm = gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
      param->data[k] = base;
      float numeric = (lp - lm) / (2.0f * eps);
      float analytic = param->grad[k];
      float rel = fabsf(analytic - numeric) / (fabsf(analytic) + fabsf(numeric) + 1e-8f);
      if (rel > max_rel) max_rel = rel;
      // "conditioned" entries: gradient large enough that float32 finite differences
      // are trustworthy (avoids the O(1)-loss cancellation noise floor).
      if (fabsf(analytic) > 1e-2f && rel > max_rel_cond) max_rel_cond = rel;
    }
  }
  int grad_ok = (max_rel_cond < 1e-2f);
  printf("    gradient check: max rel error %.2e (all) | %.2e (|grad|>1e-2) -> %s\n",
         max_rel, max_rel_cond, grad_ok ? "OK" : "FAIL");
  all_ok &= grad_ok;

  // --------------------------------------------------------------------------
  // Part 4: train (Adam) on the next-token task, expect the loss to drop
  // --------------------------------------------------------------------------
  Adam opt = adam_create(params, n_params, 0.02f);
  float first_loss = gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
  const int steps = 600;
  for (int s = 0; s < steps; s++) {
    float loss = gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
    tensor_zero_grad(&logits);
    cross_entropy_backward(&probs, targets, &logits);
    gpt_backward(&model, ids, seq_len, &acts, &logits);
    adam_step(&opt, params, n_params);
    if (s % 120 == 0) printf("    step %3d  loss %.4f\n", s, loss);
  }
  float final_loss = gpt_loss(&model, &acts, ids, seq_len, targets, &logits, &probs);
  int train_ok = (final_loss < first_loss * 0.25f);
  printf("    start %.4f -> final %.4f -> %s\n\n",
         first_loss, final_loss, train_ok ? "OK (loss dropped)" : "FAIL");
  all_ok &= train_ok;

  // --------------------------------------------------------------------------
  // Part 5: sampling sanity checks (deterministic)
  // --------------------------------------------------------------------------
  Rng rng = rng_create(123);
  int topk1_ok = 1;
  for (int r = 0; r < seq_len; r++) {
    // top_k=1 must always select the argmax token
    if (sample_token(&logits, r, 1.0f, 1, 1.0f, &rng) != sample_argmax(&logits, r)) {
      topk1_ok = 0;
      break;
    }
  }
  // Same seed -> same token (reproducibility)
  Rng ra = rng_create(7), rb = rng_create(7);
  int seed_ok = (sample_token(&logits, 0, 0.8f, 8, 0.95f, &ra) ==
                 sample_token(&logits, 0, 0.8f, 8, 0.95f, &rb));
  int sample_ok = topk1_ok && seed_ok;
  printf("[3] Sampling: top_k=1==argmax %s | seeded reproducible %s\n\n",
         topk1_ok ? "OK" : "FAIL", seed_ok ? "OK" : "FAIL");
  all_ok &= sample_ok;

  // --------------------------------------------------------------------------
  // Part 6: KV-cache decode must match the non-cached forward (logits row t)
  // --------------------------------------------------------------------------
  KVCache kv = gpt_kvcache_create(&model);
  int slog_shape[2] = {1, vocab};
  Tensor slog = tensor_create(slog_shape, 2);
  float kv_max_diff = 0.0f;
  for (int t = 0; t < seq_len; t++) {
    gpt_decode_step(&model, &kv, ids[t], &slog);
    for (int j = 0; j < vocab; j++) {
      float diff = fabsf(slog.data[j] - logits.data[t * vocab + j]);
      if (diff > kv_max_diff) kv_max_diff = diff;
    }
  }
  int kv_ok = (kv_max_diff < 1e-3f);
  printf("[4] KV cache: max |logit diff| vs full forward = %.2e -> %s\n\n",
         kv_max_diff, kv_ok ? "OK" : "FAIL");
  all_ok &= kv_ok;
  tensor_free(&slog);
  gpt_kvcache_free(&kv);

  adam_free(&opt);
  tensor_free(&logits);
  tensor_free(&probs);
  gpt_acts_free(&acts);
  gpt_free(&model);

  // --------------------------------------------------------------------------
  printf("Output validation: %s\n\n", all_ok ? "PASSED" : "FAILED");
  printf("Done! All memory freed successfully.\n");
  return all_ok ? 0 : 1;
}

// ============================================================================
// `generate` subcommand: a token-level REPL driven by the KV cache + sampler.
// Reads lines of space-separated token IDs from stdin (the prompt) and streams
// generated tokens. Pipe-testable, e.g.:  echo "1 5 9" | transformer generate
// With a checkpoint:  transformer generate model.bin
// (Becomes text-in/text-out once a real tokenizer is wired — see notes.md A1.)
// ============================================================================
static int run_generate(int argc, char **argv) {
  GPT model;
  if (argc > 2) {
    if (gpt_load(argv[2], &model) != 0) return 1;   // load real weights
    fprintf(stderr, "loaded %s\n", argv[2]);
  } else {
    srand(42);
    model = gpt_create(16, 8, 2, 2, 32, 64);        // toy fallback (random weights)
    fprintf(stderr, "no checkpoint given; using a random toy model (vocab=16)\n");
  }

  const int vocab = model.vocab_size;
  const int max_seq = model.max_seq_len;
  const int n_new = 20;                   // tokens to generate per prompt
  const float temperature = 0.8f;
  const int top_k = 40;
  const float top_p = 0.95f;
  Rng rng = rng_create(1234);

  int lshape[2] = {1, vocab};
  Tensor logits = tensor_create(lshape, 2);
  int *prompt = (int*)malloc(max_seq * sizeof(int));

  char line[8192];
  while (fgets(line, sizeof(line), stdin)) {
    char *p = line;
    // Strip a leading UTF-8 BOM (common when piping on Windows) so the first
    // token doesn't get mangled.
    if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB &&
        (unsigned char)p[2] == 0xBF) {
      p += 3;
    }
    // Parse space/newline-separated token IDs (bounds-checked to [0, vocab)).
    int n = 0;
    for (char *t = strtok(p, " \t\r\n"); t && n < max_seq; t = strtok(NULL, " \t\r\n")) {
      int id = atoi(t);
      if (id < 0 || id >= vocab) {
        fprintf(stderr, "skipping out-of-range token %d (vocab=%d)\n", id, vocab);
        continue;
      }
      prompt[n++] = id;
    }
    if (n == 0) continue;

    KVCache kv = gpt_kvcache_create(&model);
    printf("gen:");
    for (int i = 0; i < n; i++) {                 // feed the prompt
      gpt_decode_step(&model, &kv, prompt[i], &logits);
      printf(" %d", prompt[i]);
    }
    for (int s = 0; s < n_new && kv.len < max_seq; s++) {  // autoregress
      int next = sample_token(&logits, 0, temperature, top_k, top_p, &rng);
      printf(" [%d]", next);                       // brackets mark generated tokens
      gpt_decode_step(&model, &kv, next, &logits);
    }
    printf("\n");
    fflush(stdout);
    gpt_kvcache_free(&kv);
  }

  free(prompt);
  tensor_free(&logits);
  gpt_free(&model);
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "generate") == 0) {
    return run_generate(argc, argv);
  }
  return run_demo();
}
