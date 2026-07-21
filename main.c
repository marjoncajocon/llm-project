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
#include "train.h"
#include "agent.h"

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
  // Train BPE on a sample text, check round-trip + compression, then save/load.
  const char *sample = "the cat sat on the mat the cat ran to the cat";
  Tokenizer tok = tokenizer_create_bytelevel();
  tokenizer_train(&tok, sample, 30);
  int tids[256], tids2[256];
  int n_tids = tokenizer_encode(&tok, sample, tids, 256);
  char decoded[512];
  tokenizer_decode(&tok, tids, n_tids, decoded, sizeof(decoded));
  int rt_ok = (strcmp(sample, decoded) == 0);
  int comp_ok = (n_tids < (int)strlen(sample));

  int save_load_ok = (tokenizer_save(&tok, "tok.bin") == 0);
  Tokenizer tok2;
  save_load_ok = save_load_ok && (tokenizer_load(&tok2, "tok.bin") == 0);
  if (save_load_ok) {
    int n2 = tokenizer_encode(&tok2, sample, tids2, 256);
    save_load_ok = (n2 == n_tids) && (memcmp(tids, tids2, n_tids * sizeof(int)) == 0);
    tokenizer_free(&tok2);
  }
  remove("tok.bin");
  int tok_ok = rt_ok && comp_ok && save_load_ok;
  printf("    tokenizer: train %d bytes -> %d tokens | round-trip %s | save/load %s\n",
         (int)strlen(sample), n_tids, rt_ok ? "OK" : "FAIL",
         save_load_ok ? "OK" : "FAIL");
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
  printf("    checkpoint: save/load logits identical -> %s\n\n",
         ckpt_ok ? "OK" : "FAIL");
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
// `generate` subcommand: autoregressive generation driven by KV cache + sampler.
//   transformer generate                       -> random toy model, token IDs in
//   transformer generate model.bin             -> loaded weights, token IDs in
//   transformer generate model.bin tok.bin     -> TEXT in / TEXT out (tokenizer)
// Reads one prompt per stdin line.
// ============================================================================
static int run_generate(int argc, char **argv) {
  GPT model;
  if (argc > 2) {
    if (gpt_load(argv[2], &model) != 0) return 1;   // load trained weights
    fprintf(stderr, "loaded %s\n", argv[2]);
  } else {
    srand(42);
    model = gpt_create(16, 8, 2, 2, 32, 64);        // toy fallback (random weights)
    fprintf(stderr, "no checkpoint given; using a random toy model (vocab=16)\n");
  }

  // Optional tokenizer -> text mode (stdin lines are text, not token IDs).
  Tokenizer tok;
  int text_mode = 0;
  if (argc > 3) {
    if (tokenizer_load(&tok, argv[3]) != 0) {
      fprintf(stderr, "cannot load tokenizer %s\n", argv[3]);
      gpt_free(&model);
      return 1;
    }
    text_mode = 1;
    fprintf(stderr, "text mode (tokenizer %s, vocab %d)\n", argv[3], tok.vocab_size);
  }

  const int vocab = model.vocab_size;
  const int max_seq = model.max_seq_len;
  const int n_new = 40;                   // tokens to generate per prompt
  const float temperature = 0.8f;
  const int top_k = 40;
  const float top_p = 0.95f;
  Rng rng = rng_create(1234);

  int lshape[2] = {1, vocab};
  Tensor logits = tensor_create(lshape, 2);
  int *prompt = (int*)malloc(max_seq * sizeof(int));
  int *gen = (int*)malloc(max_seq * sizeof(int));

  char line[8192];
  while (fgets(line, sizeof(line), stdin)) {
    char *p = line;
    // Strip a leading UTF-8 BOM (common when piping on Windows).
    if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB &&
        (unsigned char)p[2] == 0xBF) {
      p += 3;
    }

    int n = 0;
    if (text_mode) {
      size_t L = strlen(p);                 // strip trailing newline
      while (L > 0 && (p[L - 1] == '\n' || p[L - 1] == '\r')) p[--L] = '\0';
      n = tokenizer_encode(&tok, p, prompt, max_seq);
    } else {
      for (char *t = strtok(p, " \t\r\n"); t && n < max_seq; t = strtok(NULL, " \t\r\n")) {
        int id = atoi(t);
        if (id < 0 || id >= vocab) {
          fprintf(stderr, "skipping out-of-range token %d (vocab=%d)\n", id, vocab);
          continue;
        }
        prompt[n++] = id;
      }
    }
    if (n == 0) continue;

    KVCache kv = gpt_kvcache_create(&model);
    for (int i = 0; i < n; i++) gpt_decode_step(&model, &kv, prompt[i], &logits);
    int ng = 0;
    for (int s = 0; s < n_new && kv.len < max_seq; s++) {
      int next = sample_token(&logits, 0, temperature, top_k, top_p, &rng);
      gen[ng++] = next;
      gpt_decode_step(&model, &kv, next, &logits);
    }

    if (text_mode) {
      char out[8192];
      tokenizer_decode(&tok, gen, ng, out, sizeof(out));
      printf("%s%s\n", p, out);                    // prompt text + continuation
    } else {
      printf("gen:");
      for (int i = 0; i < n; i++) printf(" %d", prompt[i]);
      for (int i = 0; i < ng; i++) printf(" [%d]", gen[i]);
      printf("\n");
    }
    fflush(stdout);
    gpt_kvcache_free(&kv);
  }

  free(prompt);
  free(gen);
  tensor_free(&logits);
  if (text_mode) tokenizer_free(&tok);
  gpt_free(&model);
  return 0;
}

// ============================================================================
// `agent` subcommand: demo the parse -> approve -> execute -> observe loop with
// a scripted "model" (the toy GPT can't emit tool calls; a real LLM backend
// would replace the `responses` script). Use `agent --ask` for interactive
// approval; default auto-approves so it runs non-interactively.
// ============================================================================
static int run_agent(int argc, char **argv) {
  const char *goal = "Create a note file, then read it back to verify.";
  const char *responses[] = {
    "I'll create the file.\n"
    "<tool name=\"write_file\" path=\"agent_demo.txt\">Hello from the C agent!\n</tool>",
    "Now I'll read it back.\n"
    "<tool name=\"read_file\" path=\"agent_demo.txt\"></tool>",
    "<done>Created agent_demo.txt and verified its contents.</done>"
  };
  int n = 3;

  ApprovalMode mode = APPROVAL_AUTO;
  if (argc > 2 && strcmp(argv[2], "--ask") == 0) mode = APPROVAL_ASK;

  int rc = agent_run(goal, responses, n, mode);
  remove("agent_demo.txt");  // clean up the demo artifact
  return rc;
}

// ============================================================================
// `train` subcommand: train on a text file or directory, configured by flags.
//   transformer train <path> [--out F] [--tok F] [--merges N] [--steps N]
//                            [--d-model N] [--layers N] [--heads N] [--d-ff N]
//                            [--ctx N] [--lr F] [--seed N]
// Scale up (bigger/slower/better):  --d-model 256 --layers 6 --heads 8 --ctx 128 --steps 20000
// ============================================================================
static void train_usage(void) {
  fprintf(stderr,
    "usage: transformer train <path> [flags]\n"
    "  <path> : text file OR directory of *.txt\n"
    "  --out F       output model (default model.bin)   --tok F     output tokenizer (default tok.bin)\n"
    "  --merges N    BPE merges (default 300)            --steps N   Adam steps (default 1000)\n"
    "  --d-model N   model dim (default 128)             --layers N  blocks (default 3)\n"
    "  --heads N     attention heads (default 4)         --d-ff N    FFN dim (default 4*d_model)\n"
    "  --ctx N       context window (default 64)         --lr F      learning rate (default 3e-4)\n"
    "  --seed N      RNG seed (default 1234)             --val-frac F  held-out fraction (default 0.1)\n"
    "  --resume F    continue training a saved model     --checkpoint-every N  save every N steps\n");
}

static int run_train(int argc, char **argv) {
  if (argc < 3 || strncmp(argv[2], "--", 2) == 0) { train_usage(); return 2; }
  const char *data = argv[2];
  TrainConfig cfg = train_default_config();

  for (int i = 3; i < argc; i++) {
    const char *f = argv[i];
    int has_val = (i + 1 < argc);
#define NEED_VAL() do { if (!has_val) { fprintf(stderr, "missing value for %s\n", f); return 2; } } while (0)
    if      (strcmp(f, "--out") == 0)     { NEED_VAL(); cfg.out_model = argv[++i]; }
    else if (strcmp(f, "--tok") == 0)     { NEED_VAL(); cfg.out_tok = argv[++i]; }
    else if (strcmp(f, "--merges") == 0)  { NEED_VAL(); cfg.num_merges = atoi(argv[++i]); }
    else if (strcmp(f, "--steps") == 0)   { NEED_VAL(); cfg.steps = atoi(argv[++i]); }
    else if (strcmp(f, "--d-model") == 0) { NEED_VAL(); cfg.d_model = atoi(argv[++i]); }
    else if (strcmp(f, "--layers") == 0)  { NEED_VAL(); cfg.n_layers = atoi(argv[++i]); }
    else if (strcmp(f, "--heads") == 0)   { NEED_VAL(); cfg.n_heads = atoi(argv[++i]); }
    else if (strcmp(f, "--d-ff") == 0)    { NEED_VAL(); cfg.d_ff = atoi(argv[++i]); }
    else if (strcmp(f, "--ctx") == 0)     { NEED_VAL(); cfg.ctx = atoi(argv[++i]); }
    else if (strcmp(f, "--lr") == 0)      { NEED_VAL(); cfg.lr = (float)atof(argv[++i]); }
    else if (strcmp(f, "--seed") == 0)    { NEED_VAL(); cfg.seed = (unsigned)atoi(argv[++i]); }
    else if (strcmp(f, "--resume") == 0)  { NEED_VAL(); cfg.resume = argv[++i]; }
    else if (strcmp(f, "--checkpoint-every") == 0) { NEED_VAL(); cfg.checkpoint_every = atoi(argv[++i]); }
    else if (strcmp(f, "--val-frac") == 0){ NEED_VAL(); cfg.val_frac = (float)atof(argv[++i]); }
    else { fprintf(stderr, "unknown flag: %s\n", f); train_usage(); return 2; }
#undef NEED_VAL
  }
  return train_run(data, &cfg);
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "generate") == 0) return run_generate(argc, argv);
  if (argc > 1 && strcmp(argv[1], "agent") == 0) return run_agent(argc, argv);
  if (argc > 1 && strcmp(argv[1], "train") == 0) return run_train(argc, argv);
  return run_demo();
}
