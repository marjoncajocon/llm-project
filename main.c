// main.c - Minimal GPT-2 Style Transformer in Pure C11 (demo entry point)
// Build: make        (see README.md and Makefile)
// Or:    gcc -std=c11 -O3 -march=native -o transformer *.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "tensor.h"
#include "loss.h"
#include "optim.h"
#include "sampler.h"
#include "model.h"
#include "checkpoint.h"
#include "tokenizer.h"
#include "train.h"
#include "grpo.h"
#include "agent.h"
#include "toolexec.h"
#include "risk.h"
#include "caption.h"
#include "text2media.h"
#include "t2i.h"
#include "vqvae.h"
#include "hftok.h"
#include "gen.h"
#include "image.h"
#include "patch.h"
#include "video.h"
#include "audio.h"
#include "imagegen.h"
#include "audiogen.h"
#include "llama.h"
#include "matrix.h"
#include "gpu.h"
#include "selftest.h"

// Portable "make directory if absent" (ignores already-exists).
#if defined(_WIN32)
#include <direct.h>
static void ensure_dir(const char *p) { _mkdir(p); }
#else
#include <sys/stat.h>
static void ensure_dir(const char *p) { mkdir(p, 0755); }
#endif

// A named model lives in model/<name>/ as model.bin + tok.bin. Fill the two
// path buffers (and create the folder when `make` is set, i.e. for training).
static void model_paths(const char *name, char *model_out, char *tok_out,
                        int cap, int make) {
  char dir[400];
  snprintf(dir, sizeof(dir), "model/%s", name);
  if (make) { ensure_dir("model"); ensure_dir(dir); }
  snprintf(model_out, cap, "%s/model.bin", dir);
  snprintf(tok_out, cap, "%s/tok.bin", dir);
}


// ============================================================================
// `generate` subcommand: autoregressive generation driven by KV cache + sampler.
//   transformer generate                       -> random toy model, token IDs in
//   transformer generate model.bin             -> loaded weights, token IDs in
//   transformer generate model.bin tok.bin     -> TEXT in / TEXT out (tokenizer)
//   transformer generate --name X              -> model/X/ (text if tok.bin present)
// Reads one prompt per stdin line.
// ============================================================================
static int run_generate(int argc, char **argv) {
  // Resolve model + optional tokenizer paths (supports --name <folder>).
  const char *model_path = NULL, *tok_path = NULL;
  static char nm[512], nt[512];
  if (argc > 3 && strcmp(argv[2], "--name") == 0) {
    model_paths(argv[3], nm, nt, sizeof(nm), 0);
    model_path = nm;
    FILE *tf = fopen(nt, "rb");           // text mode only if the tokenizer exists
    if (tf) { fclose(tf); tok_path = nt; }
  } else {
    if (argc > 2) model_path = argv[2];
    if (argc > 3) tok_path = argv[3];
  }

  GPT model;
  if (model_path) {
    if (gpt_load(model_path, &model) != 0) return 1;
    fprintf(stderr, "loaded %s\n", model_path);
  } else {
    srand(42);
    model = gpt_create(16, 8, 2, 2, 32, 64);        // toy fallback (random weights)
    fprintf(stderr, "no checkpoint given; using a random toy model (vocab=16)\n");
  }

  // Optional tokenizer -> text mode (stdin lines are text, not token IDs).
  Tokenizer tok;
  int text_mode = 0;
  if (tok_path) {
    if (tokenizer_load(&tok, tok_path) != 0) {
      fprintf(stderr, "cannot load tokenizer %s\n", tok_path);
      gpt_free(&model);
      return 1;
    }
    text_mode = 1;
    fprintf(stderr, "text mode (tokenizer %s, vocab %d)\n", tok_path, tok.vocab_size);
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
      // EOS = any id >= the tokenizer's vocab (SFT models reserve one extra slot).
      // Stop cleanly and drop it (nothing to decode). Non-EOS models never hit this.
      if (text_mode && next >= tok.vocab_size) break;
      gen[ng++] = next;
      if (!text_mode) { printf("t %d\n", next); fflush(stdout); }  // stream ids (gpt2)
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
// `risk` subcommand: the run_command risk classifier (risk.h) on its own.
//   transformer risk "<command>"   -> prints "LEVEL<TAB>reason"; exit status
//                                     0 SAFE, 1 REVIEW, 2 BLOCK
//   transformer risk --stdin       -> one command per input line; prints one
//                                     "LEVEL<TAB>reason" line per input line
//                                     (exit 0). Used by tools/risk_filter.py so
//                                     the training-data filter and the runtime
//                                     gate are the SAME code.
// Classifies strings only; nothing is executed.
// ============================================================================
static int run_risk(int argc, char **argv) {
  char why[256];
  if (argc == 3 && strcmp(argv[2], "--stdin") == 0) {
    static char line[65536 + 4];
    while (fgets(line, sizeof(line), stdin)) {
      size_t n = strlen(line);
      int whole = n > 0 && line[n - 1] == '\n';
      while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
      RiskLevel lv;
      if (!whole && !feof(stdin)) {                 // over-long line: never analyzed
        int c;
        while ((c = getchar()) != EOF && c != '\n') {}
        lv = RISK_BLOCK;
        snprintf(why, sizeof(why), "line too long to analyze");
      } else {
        lv = risk_classify(line, why, (int)sizeof(why));
      }
      for (char *p = why; *p; p++) if (*p == '\t' || *p == '\n' || *p == '\r') *p = ' ';
      printf("%s\t%s\n", risk_level_name(lv), why);
    }
    return 0;
  }
  if (argc != 3) {
    fprintf(stderr, "usage: transformer risk \"<command>\"   (exit 0 SAFE, 1 REVIEW, 2 BLOCK)\n"
                    "       transformer risk --stdin       (one command per line)\n");
    return 64;
  }
  RiskLevel lv = risk_classify(argv[2], why, (int)sizeof(why));
  printf("%s\t%s\n", risk_level_name(lv), why);
  return (int)lv;
}

// ============================================================================
// `agent` subcommand: the parse -> approve -> execute -> observe loop with a
// pluggable model backend.
//   transformer agent                 -> built-in scripted backend (demo)
//   transformer agent --cmd "<prog>"  -> run <prog> as the model; its stdout is
//                                        each response (the transcript is passed
//                                        as a file arg). Plug in a real LLM here.
//   --goal "..."      set the goal
//   --workspace DIR   root for every tool path and the cwd of run_command
//                     (default: ./workspace, created if missing)
//   --auto            auto-approve write_file/read_file (default: ask y/N for
//                     every action). run_command ALWAYS asks, even with --auto,
//                     and auto-approved writes refuse .git/, tools/ and
//                     executable/script extensions. Every run_command is
//                     risk-classified (risk.h): BLOCK needs a typed "yes",
//                     and with --auto it is refused without asking.
//   --ask             accepted for compatibility (it is the default)
//   --timeout SEC     wall-clock limit per run_command (default 60)
//   --no-compact      stop when the transcript is full instead of compacting it
//                     (clip old observations, then drop the oldest turns)
// Exit status: 0 normal end, 2 transcript full, 3 the --cmd backend failed.
// ============================================================================
static int run_agent(int argc, char **argv) {
  const char *goal = "Create a note file, then read it back to verify.";
  const char *cmd = NULL;
  const char *wsdir = "workspace";
  AgentConfig cfg = agent_default_config();       // APPROVAL_ASK
  cfg.compact = 1;                                // a full transcript is compacted, not fatal
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--ask") == 0) cfg.mode = APPROVAL_ASK;
    else if (strcmp(argv[i], "--no-compact") == 0) cfg.compact = 0;
    else if (strcmp(argv[i], "--auto") == 0) cfg.mode = APPROVAL_AUTO;
    else if (strcmp(argv[i], "--cmd") == 0 && i + 1 < argc) cmd = argv[++i];
    else if (strcmp(argv[i], "--goal") == 0 && i + 1 < argc) goal = argv[++i];
    else if (strcmp(argv[i], "--workspace") == 0 && i + 1 < argc) wsdir = argv[++i];
    else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) cfg.timeout_s = atoi(argv[++i]);
    else {
      fprintf(stderr, "agent: unknown option '%s'\n"
              "usage: transformer agent [--goal \"...\"] [--cmd \"<prog>\"] [--workspace DIR]\n"
              "                         [--auto] [--timeout SEC] [--no-compact]\n", argv[i]);
      return 1;
    }
  }
  char ws[1024];
  if (toolexec_workspace_init(wsdir, ws, sizeof(ws)) != 0) {
    fprintf(stderr, "agent: cannot create/open workspace '%s'\n", wsdir);
    return 1;
  }
  cfg.workspace = ws;
  if (strcmp(wsdir, ".") == 0 || strcmp(wsdir, "./") == 0)
    fprintf(stderr, "agent: warning: the workspace is the current directory; the model can "
                    "write any file under it\n");
  printf("[%s mode] tools are confined to %s\n",
         cfg.mode == APPROVAL_AUTO ? "auto-approve" : "ask", ws);

  int rc;
  if (cmd) {
    CommandBackend b = command_backend(cmd);
    rc = agent_run(goal, &b.base, &cfg);
  } else {
    // Built-in scripted "model" (the toy GPT can't emit tool calls).
    const char *responses[] = {
      "I'll create the file.\n"
      "<tool name=\"write_file\" path=\"agent_demo.txt\">Hello from the C agent!\n</tool>",
      "Now I'll read it back.\n"
      "<tool name=\"read_file\" path=\"agent_demo.txt\"></tool>",
      "<done>Created agent_demo.txt and verified its contents.</done>"
    };
    ScriptedBackend b = scripted_backend(responses, 3);
    rc = agent_run(goal, &b.base, &cfg);
  }
  char demo[1100];
  snprintf(demo, sizeof(demo), "%s/agent_demo.txt", ws);
  remove(demo);  // clean up the demo artifact
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
    "  --name X      save to model/X/ (model.bin + tok.bin); use `generate --name X`\n"
    "  --out F       output model (default model.bin)   --tok F     output tokenizer (default tok.bin)\n"
    "  --merges N    BPE merges (default 300)            --steps N   Adam steps (default 1000)\n"
    "  --d-model N   model dim (default 128)             --layers N  blocks (default 3)\n"
    "  --heads N     attention heads (default 4)         --d-ff N    FFN dim (default 4*d_model)\n"
    "  --ctx N       context window (default 64)         --lr F      learning rate (default 3e-4)\n"
    "  --seed N      RNG seed (default 1234)             --val-frac F  held-out fraction (default 0.1)\n"
    "  --resume F    continue training a saved model     --checkpoint-every N  save every N steps\n"
    "  --sft         masked instruction tuning: file is 'prompt<TAB>response' lines,\n"
    "                loss on the response only (true SFT). See data/sft.txt.\n"
    "  Training recipe (all OFF by default, so existing runs reproduce exactly):\n"
    "  --weight-decay F  AdamW decoupled decay on weight matrices (biases/norms exempt)\n"
    "  --clip F          global grad-norm clip (1.0 is the usual choice)\n"
    "  --warmup N        linear LR warmup steps      --min-lr F  cosine floor LR\n"
    "  --batch N         sequences per optimizer step (gradient accumulation;\n"
    "                    default 1). Costs N forward+backward passes per step but\n"
    "                    gives a far less noisy gradient.\n"
    "\n"
    "`llama-train` takes the SAME flags but trains the modern LLaMA-family\n"
    "architecture (RMSNorm+RoPE+GQA+SwiGLU) and writes an LLMA checkpoint that\n"
    "`transformer llama` generates from. Extra flags:\n"
    "  --kv-heads N      GQA key/value heads (default = --heads, i.e. plain MHA)\n"
    "  --rope-base F     RoPE frequency base (default 10000)\n"
    "  --window W        sliding-window attention: each position attends only to\n"
    "                    its W most recent positions (default 0 = full attention)\n"
    "  --global-every N  with --window: every Nth layer stays full/global\n"
    "                    (2 = alternate like Gemma 2; 6 = 5 local : 1 global like\n"
    "                    Gemma 3; default 0 = every layer windowed, like Mistral)\n"
    "  --lora-r N        LoRA rank: FREEZE the base and train only low-rank adapters\n"
    "                    on Wq/Wk/Wv/Wo (merged into the weights before saving).\n"
    "                    Pair with --resume to fine-tune a pretrained checkpoint.\n"
    "  --lora-alpha F    LoRA scaling (scale = alpha / r; default alpha = r)\n"
    "  --lora-mlp        with --lora-r: ALSO put adapters on the SwiGLU MLP\n"
    "                    (Wgate/Wup/Wdown). Attention-only LoRA underperforms\n"
    "                    (\"LoRA Without Regret\", 2025). Default off.\n"
    "  --lora-plus F     LoRA+: adapter B matrices train at F x --lr (A keeps\n"
    "                    --lr); 16 is the paper's choice. Default off.\n"
    "  --hf-tok F        fine-tune with a REAL model's own vocabulary (.hf, see\n"
    "                    tools/export_tokenizer.py) instead of training a byte BPE.\n"
    "                    Its real EOS token is used, so no vocab slot is added.\n"
    "  --optim adam|muon optimizer (default adam). muon = Muon (Newton-Schulz\n"
    "                    orthogonalized momentum, Moonshot RMS-matched so the\n"
    "                    same --lr/--weight-decay apply) on the hidden weight\n"
    "                    matrices; embedding, lm_head, norms and biases stay\n"
    "                    on AdamW. With --lora-r every adapter uses Muon.\n");
}

// Shared by `train` (GPT) and `llama-train` (modern LLaMA-family) — the flags are
// identical apart from the two GQA/RoPE ones, so only the driver at the end differs.
static int run_train_arch(int argc, char **argv, int llama) {
  if (argc < 3 || strncmp(argv[2], "--", 2) == 0) { train_usage(); return 2; }
  const char *data = argv[2];
  TrainConfig cfg = train_default_config();
  const char *name = NULL;
  static char name_model[512], name_tok[512];

  for (int i = 3; i < argc; i++) {
    const char *f = argv[i];
    int has_val = (i + 1 < argc);
#define NEED_VAL() do { if (!has_val) { fprintf(stderr, "missing value for %s\n", f); return 2; } } while (0)
    if      (strcmp(f, "--name") == 0)    { NEED_VAL(); name = argv[++i]; }
    else if (strcmp(f, "--out") == 0)     { NEED_VAL(); cfg.out_model = argv[++i]; }
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
    else if (strcmp(f, "--sft") == 0)     { cfg.sft = 1; }
    else if (strcmp(f, "--weight-decay") == 0) { NEED_VAL(); cfg.weight_decay = (float)atof(argv[++i]); }
    else if (strcmp(f, "--clip") == 0)    { NEED_VAL(); cfg.grad_clip = (float)atof(argv[++i]); }
    else if (strcmp(f, "--warmup") == 0)  { NEED_VAL(); cfg.warmup = atoi(argv[++i]); }
    else if (strcmp(f, "--min-lr") == 0)  { NEED_VAL(); cfg.lr_min = (float)atof(argv[++i]); }
    else if (strcmp(f, "--batch") == 0)   { NEED_VAL(); cfg.batch = atoi(argv[++i]); }
    else if (strcmp(f, "--kv-heads") == 0 && llama) { NEED_VAL(); cfg.n_kv = atoi(argv[++i]); }
    else if (strcmp(f, "--rope-base") == 0 && llama) { NEED_VAL(); cfg.rope_base = (float)atof(argv[++i]); }
    else if (strcmp(f, "--window") == 0 && llama) { NEED_VAL(); cfg.window = atoi(argv[++i]); }
    else if (strcmp(f, "--global-every") == 0 && llama) { NEED_VAL(); cfg.global_every = atoi(argv[++i]); }
    else if (strcmp(f, "--lora-r") == 0 && llama) { NEED_VAL(); cfg.lora_r = atoi(argv[++i]); }
    else if (strcmp(f, "--lora-alpha") == 0 && llama) { NEED_VAL(); cfg.lora_alpha = (float)atof(argv[++i]); }
    else if (strcmp(f, "--lora-mlp") == 0 && llama) { cfg.lora_mlp = 1; }
    else if (strcmp(f, "--lora-plus") == 0 && llama) { NEED_VAL(); cfg.lora_plus = (float)atof(argv[++i]); }
    else if (strcmp(f, "--hf-tok") == 0 && llama) { NEED_VAL(); cfg.hf_tok = argv[++i]; }
    else if (strcmp(f, "--optim") == 0 && llama) {
      NEED_VAL(); const char *o = argv[++i];
      if (strcmp(o, "adam") == 0) cfg.muon = 0;
      else if (strcmp(o, "muon") == 0) cfg.muon = 1;
      else { fprintf(stderr, "--optim must be adam or muon (got %s)\n", o); return 2; }
    }
    else { fprintf(stderr, "unknown flag: %s\n", f); train_usage(); return 2; }
#undef NEED_VAL
  }
  // --name X saves to model/X/{model,tok}.bin (unless --out/--tok were given).
  if (name) {
    model_paths(name, name_model, name_tok, sizeof(name_model), 1);
    cfg.out_model = name_model;
    cfg.out_tok = name_tok;
    printf("model dir: model/%s/\n", name);
  }
  return llama ? llama_train_run(data, &cfg) : train_run(data, &cfg);
}

static int run_train(int argc, char **argv) { return run_train_arch(argc, argv, 0); }
static int run_llama_train(int argc, char **argv) { return run_train_arch(argc, argv, 1); }

static void grpo_usage(void) {
  printf(
    "Usage: transformer grpo <tasks.txt> --model M.bin --hf-tok V.hf [options]\n"
    "\n"
    "Reinforcement learning from VERIFIABLE rewards (GRPO). Where `llama-train\n"
    "--sft` is shown the right answer and pushed toward it, this lets the model\n"
    "ACT, scores what it did, and pushes it toward its own better-than-average\n"
    "attempts and away from its worse-than-average ones.\n"
    "\n"
    "<tasks.txt> is the SAME prompt<TAB>response format the SFT trainer reads --\n"
    "the response column becomes the REFERENCE ACTION the reward scores against,\n"
    "so a corpus from tools/clean_sft.py drops straight in. Build one that is\n"
    "known-grounded with: python tools/grpo_tasks.py --out data/grpo_tasks.txt ...\n"
    "\n"
    "Required:\n"
    "  --model F         LLMA checkpoint to start from (the policy)\n"
    "  --hf-tok F        that model's own .hf vocabulary\n"
    "Output:\n"
    "  --out F           where to save (default model/grpo.bin)\n"
    "  --name X          shorthand for --out model/X/model.bin\n"
    "Rollouts:\n"
    "  --group N         G rollouts per task, >= 2 (default 8). The group's own\n"
    "                    mean is the baseline -- that is what replaces a value net.\n"
    "  --max-new N       cap on sampled tokens per rollout (default 96)\n"
    "  --temp F          sampling temperature (default 1.0). This is the EXPLORATION\n"
    "                    knob: too low and every rollout in a group is identical,\n"
    "                    the advantage is zero, and the group teaches nothing.\n"
    "  --top-k N / --top-p F   nucleus sampling (defaults 0 / 0.95)\n"
    "Optimization:\n"
    "  --steps N         optimizer steps (default 100)\n"
    "  --groups N        task groups accumulated per step (default 1)\n"
    "  --inner-epochs N  updates taken per set of rollouts (default 1 = on-policy).\n"
    "                    > 1 reuses the expensive rollouts; the ratio clip is what\n"
    "                    keeps that honest.\n"
    "  --lr F            learning rate (default 1e-5 -- RL wants far less than SFT)\n"
    "  --clip F          grad-norm clip (default 1.0; 0 disables -- not advised)\n"
    "  --weight-decay F  AdamW decoupled decay (default 0)\n"
    "  --warmup N / --min-lr F   cosine schedule (default: constant LR)\n"
    "  --lora-r N        LoRA rank, base frozen (default 8; 0 = full fine-tune)\n"
    "  --lora-alpha F    LoRA scaling (default 16)\n"
    "  --lora-mlp        adapters on the MLP (Wgate/Wup/Wdown) too (default off)\n"
    "  --lora-plus F     LoRA+: B matrices train at F x --lr (16 typical; default off)\n"
    "Objective:\n"
    "  --clip-eps F      PPO-style ratio clip, 1 +/- eps (default 0.2; 0 = off)\n"
    "  --clip-high F     DAPO clip-higher: upper bound 1 + F instead of 1 + eps\n"
    "                    (default: symmetric). Wider lets a rare-but-good token\n"
    "                    grow faster -- the guard against entropy collapse.\n"
    "  --no-std          advantage = r - mean, not (r - mean)/std (Dr. GRPO)\n"
    "  --loss-agg M      seq   = 1/(G*len) per rollout (default, original GRPO)\n"
    "                    token = every response token in the step weighs the same (DAPO)\n"
    "                    const = 1/(G*max_new), length-independent (Dr. GRPO)\n"
    "  --resample N      when a group comes back FLAT, draw a fresh task and\n"
    "                    resample, up to N times (DAPO dynamic sampling; default 0)\n"
    "  --dapo            preset: --clip-high 0.28 --loss-agg token --resample 2\n"
    "  --dr-grpo         preset: --no-std --loss-agg const\n"
    "                    (flags after a preset override it)\n"
    "Task sampling:\n"
    "  --prio            difficulty-aware draw instead of uniform: weight per task\n"
    "                    w = max(0.01, clamp(1 - success_ema, wmin, 1)\n"
    "                                 x 0.5^mastered_streak x 0.7^too_hard_streak),\n"
    "                    unseen tasks weight 1. A streak counts consecutive FLAT\n"
    "                    groups (no gradient) all-correct (mastered) or all equally\n"
    "                    wrong (too hard for now). Kimi k1.5 + GRESO. Own RNG\n"
    "                    stream from --seed. Implies --diag. Works with --dapo.\n"
    "  --prio-alpha F    EMA rate of the per-task stats (default 0.3)\n"
    "  --prio-wmin F     lower clip of the (1 - success) term (default 0.05)\n"
    "  --prio-hi F       a flat group at reward >= F is mastered (default 0.9)\n"
    "  --diag            append diagnostics to every report: policy entropy over the\n"
    "                    sampled tokens, live-group fraction, Adam updates, ratio-clip\n"
    "                    fraction; task coverage summary at the end (default off)\n"
    "  --kl F            KL penalty toward the reference policy (default 0 = off).\n"
    "                    The reference is the FROZEN BASE -- the same weights with\n"
    "                    the adapters off -- so it costs one extra forward per\n"
    "                    rollout and no extra memory.\n"
    "Reward:\n"
    "  --reward v1|v2    v1 (default) = the original reward; reproduces old runs.\n"
    "                    v2 = format * (0.3 grounded + 0.7 match): a rollout cut\n"
    "                    off at --max-new scores 0, <done>/prose are grounded only\n"
    "                    when the reference is the same kind, every command\n"
    "                    argument must be grounded, leaked MODEL:/OBSERVATION: fail\n"
    "Run control:\n"
    "  --seed N          RNG seed (default 1)\n"
    "  --verbose         print a scored sample rollout per group\n"
    "Evaluation (no training):\n"
    "  --eval-only       score the model instead of training it: no LoRA, no\n"
    "                    optimizer, no backward, nothing saved. Tasks are visited\n"
    "                    once each in FILE ORDER; rollout (task i, draw k) samples\n"
    "                    from its own RNG seeded by hash(--seed, i, k), so two\n"
    "                    checkpoints see PAIRED sampling noise. --temp defaults to 0\n"
    "                    (greedy) here. Training-only flags are ignored.\n"
    "  --k N             rollouts per task (default 1; use with --temp > 0)\n"
    "  --limit N         only the first N tasks of the file\n"
    "  --dump F.jsonl    one JSON line per rollout (task, k, family, reward terms,\n"
    "                    exact, action, stop, tokens, response); compare two dumps\n"
    "                    with: python tools/evalcard.py A.jsonl B.jsonl\n"
    "  --constrain       constrained decoding (eval only): every rollout is ONE valid\n"
    "                    action block -- [<think>..</think>] then read_file / write_file\n"
    "                    / run_command / <done> in canonical form -- and path=\"...\" is\n"
    "                    a whole-token span of the task prompt (byte-exact). A token is\n"
    "                    allowed iff its bytes keep the output a valid prefix; the top\n"
    "                    candidates are walked in logit order, then the whole vocab.\n"
    "  --constrain-body  also: a write_file body must be one prompt span (implies --constrain)\n"
    "  --constrain-cmd   also: run_command arguments other than verbs and -flags must be\n"
    "                    prompt spans (implies --constrain)\n"
    "  --constrain-think N  <think> bytes before </think> is forced (default 256; 0 = no cap)\n"
    "\n"
    "The run saves ONCE, at the end: mid-run saving would mean merging the\n"
    "adapters, which resets B to zero and stalls training.\n");
}

static int run_grpo(int argc, char **argv) {
  if (argc < 3 || strncmp(argv[2], "--", 2) == 0) { grpo_usage(); return 2; }
  const char *tasks = argv[2];
  GrpoConfig cfg = grpo_default_config();
  const char *name = NULL;
  static char name_model[512], name_tok[512];
  int temp_set = 0;

  for (int i = 3; i < argc; i++) {
    const char *f = argv[i];
    int has_val = (i + 1 < argc);
#define NEED_VAL() do { if (!has_val) { fprintf(stderr, "missing value for %s\n", f); return 2; } } while (0)
    if      (strcmp(f, "--model") == 0)   { NEED_VAL(); cfg.model = argv[++i]; }
    else if (strcmp(f, "--hf-tok") == 0)  { NEED_VAL(); cfg.hf_tok = argv[++i]; }
    else if (strcmp(f, "--out") == 0)     { NEED_VAL(); cfg.out_model = argv[++i]; }
    else if (strcmp(f, "--name") == 0)    { NEED_VAL(); name = argv[++i]; }
    else if (strcmp(f, "--group") == 0)   { NEED_VAL(); cfg.group = atoi(argv[++i]); }
    else if (strcmp(f, "--max-new") == 0) { NEED_VAL(); cfg.max_new = atoi(argv[++i]); }
    else if (strcmp(f, "--temp") == 0)    { NEED_VAL(); cfg.temperature = (float)atof(argv[++i]); temp_set = 1; }
    else if (strcmp(f, "--top-k") == 0)   { NEED_VAL(); cfg.top_k = atoi(argv[++i]); }
    else if (strcmp(f, "--top-p") == 0)   { NEED_VAL(); cfg.top_p = (float)atof(argv[++i]); }
    else if (strcmp(f, "--steps") == 0)   { NEED_VAL(); cfg.steps = atoi(argv[++i]); }
    else if (strcmp(f, "--groups") == 0)  { NEED_VAL(); cfg.groups_per_step = atoi(argv[++i]); }
    else if (strcmp(f, "--inner-epochs") == 0) { NEED_VAL(); cfg.inner_epochs = atoi(argv[++i]); }
    else if (strcmp(f, "--lr") == 0)      { NEED_VAL(); cfg.lr = (float)atof(argv[++i]); }
    else if (strcmp(f, "--clip") == 0)    { NEED_VAL(); cfg.grad_clip = (float)atof(argv[++i]); }
    else if (strcmp(f, "--weight-decay") == 0) { NEED_VAL(); cfg.weight_decay = (float)atof(argv[++i]); }
    else if (strcmp(f, "--warmup") == 0)  { NEED_VAL(); cfg.warmup = atoi(argv[++i]); }
    else if (strcmp(f, "--min-lr") == 0)  { NEED_VAL(); cfg.lr_min = (float)atof(argv[++i]); }
    else if (strcmp(f, "--lora-r") == 0)  { NEED_VAL(); cfg.lora_r = atoi(argv[++i]); }
    else if (strcmp(f, "--lora-alpha") == 0) { NEED_VAL(); cfg.lora_alpha = (float)atof(argv[++i]); }
    else if (strcmp(f, "--lora-mlp") == 0) { cfg.lora_mlp = 1; }
    else if (strcmp(f, "--lora-plus") == 0) { NEED_VAL(); cfg.lora_plus = (float)atof(argv[++i]); }
    else if (strcmp(f, "--clip-eps") == 0){ NEED_VAL(); cfg.clip_eps = (float)atof(argv[++i]); }
    else if (strcmp(f, "--kl") == 0)      { NEED_VAL(); cfg.kl_beta = (float)atof(argv[++i]); }
    else if (strcmp(f, "--clip-high") == 0) { NEED_VAL(); cfg.clip_eps_high = (float)atof(argv[++i]); }
    else if (strcmp(f, "--no-std") == 0)  { cfg.std_norm = 0; }
    else if (strcmp(f, "--resample") == 0){ NEED_VAL(); cfg.resample = atoi(argv[++i]); }
    else if (strcmp(f, "--dapo") == 0)    { grpo_preset_dapo(&cfg); }
    else if (strcmp(f, "--dr-grpo") == 0) { grpo_preset_dr_grpo(&cfg); }
    else if (strcmp(f, "--prio") == 0)    { cfg.prio = 1; }
    else if (strcmp(f, "--prio-alpha") == 0) { NEED_VAL(); cfg.prio_alpha = (float)atof(argv[++i]); }
    else if (strcmp(f, "--prio-wmin") == 0)  { NEED_VAL(); cfg.prio_wmin = (float)atof(argv[++i]); }
    else if (strcmp(f, "--prio-hi") == 0)    { NEED_VAL(); cfg.prio_hi = (float)atof(argv[++i]); }
    else if (strcmp(f, "--diag") == 0)    { cfg.diag = 1; }
    else if (strcmp(f, "--loss-agg") == 0) {
      NEED_VAL();
      const char *m = argv[++i];
      if      (strcmp(m, "seq") == 0)   cfg.loss_agg = GRPO_AGG_SEQ;
      else if (strcmp(m, "token") == 0) cfg.loss_agg = GRPO_AGG_TOKEN;
      else if (strcmp(m, "const") == 0) cfg.loss_agg = GRPO_AGG_CONST;
      else { fprintf(stderr, "--loss-agg must be seq, token or const (got '%s')\n", m); return 2; }
    }
    else if (strcmp(f, "--reward") == 0) {
      NEED_VAL();
      const char *v = argv[++i];
      if      (strcmp(v, "v1") == 0 || strcmp(v, "1") == 0) cfg.reward_version = 1;
      else if (strcmp(v, "v2") == 0 || strcmp(v, "2") == 0) cfg.reward_version = 2;
      else { fprintf(stderr, "--reward must be v1 or v2 (got '%s')\n", v); return 2; }
    }
    else if (strcmp(f, "--seed") == 0)    { NEED_VAL(); cfg.seed = (unsigned)atoi(argv[++i]); }
    else if (strcmp(f, "--verbose") == 0) { cfg.verbose = 1; }
    else if (strcmp(f, "--eval-only") == 0) { cfg.eval_only = 1; }
    else if (strcmp(f, "--k") == 0)       { NEED_VAL(); cfg.eval_k = atoi(argv[++i]); }
    else if (strcmp(f, "--limit") == 0)   { NEED_VAL(); cfg.eval_limit = atoi(argv[++i]); }
    else if (strcmp(f, "--dump") == 0)    { NEED_VAL(); cfg.eval_dump = argv[++i]; }
    else if (strcmp(f, "--constrain") == 0) { cfg.constrain = 1; }
    else if (strcmp(f, "--constrain-body") == 0) { cfg.constrain = 1; cfg.con.ground_body = 1; }
    else if (strcmp(f, "--constrain-cmd") == 0)  { cfg.constrain = 1; cfg.con.ground_cmd = 1; }
    else if (strcmp(f, "--constrain-think") == 0) { NEED_VAL(); cfg.con.think_max = atoi(argv[++i]); }
    else { fprintf(stderr, "unknown flag: %s\n", f); grpo_usage(); return 2; }
#undef NEED_VAL
  }
  if (name) {
    model_paths(name, name_model, name_tok, sizeof(name_model), 1);
    cfg.out_model = name_model;
    printf("model dir: model/%s/\n", name);
  }
  if (cfg.eval_only) {
    if (cfg.eval_k < 1) { fprintf(stderr, "--k must be >= 1\n"); return 2; }
    if (!temp_set) cfg.temperature = 0.0f;     // greedy unless asked otherwise
    return grpo_eval_run(tasks, &cfg);
  }
  if (cfg.constrain) {
    fprintf(stderr, "error: --constrain is for --eval-only: a masked sampler is not the policy\n"
                    "       whose probability ratio GRPO trains on\n");
    return 2;
  }
  if (cfg.eval_k != 1 || cfg.eval_limit != 0 || cfg.eval_dump)
    fprintf(stderr, "note: --k / --limit / --dump only apply with --eval-only\n");
  if (cfg.groups_per_step < 1) cfg.groups_per_step = 1;
  if (cfg.inner_epochs < 1) cfg.inner_epochs = 1;
  if (cfg.resample < 0) cfg.resample = 0;
  return grpo_run(tasks, &cfg);
}

// ============================================================================
// Media generation to a file: quantize the input to tokens, train a small GPT to
// model that token sequence, then autoregressively regenerate it and decode. The
// codec (imagegen/audiogen) is the only difference between imggen and audgen.
// (A single-example generative model — overfits the input; a real generative
// model needs a dataset. Kept small: O(n^2) attention, so short inputs only.)
// ============================================================================
static int autoregress_reconstruct(const int *tok, int P, int V, int steps, int *gen) {
  int *tgt = (int*)malloc(P * sizeof(int));
  for (int i = 0; i < P; i++) tgt[i] = (i < P - 1) ? tok[i + 1] : -1;
  srand(1234);
  GPT g = gpt_create(V, 64, 4, 2, 256, P);
  GPTActs a = gpt_acts_create(&g, P);
  int lz[2] = {P, V};
  Tensor lo = tensor_create(lz, 2), pr = tensor_create(lz, 2);
  Tensor *par[256];
  int np = gpt_collect_params(&g, par);
  Adam opt = adam_create(par, np, 0.005f);
  int report = (steps >= 10) ? steps / 10 : 1;
  for (int s = 0; s < steps; s++) {
    gpt_forward_train(&g, tok, P, &a, &lo);
    float loss = cross_entropy_forward(&lo, tgt, &pr);
    tensor_zero_grad(&lo);
    cross_entropy_backward(&pr, tgt, &lo);
    gpt_backward(&g, tok, P, &a, &lo);
    adam_step(&opt, par, np);
    if (s % report == 0) fprintf(stderr, "  step %d/%d  loss %.4f\n", s, steps, loss);
  }
  KVCache kv = gpt_kvcache_create(&g);
  int gz[2] = {1, V};
  Tensor gl = tensor_create(gz, 2);
  gen[0] = tok[0];
  gpt_decode_step(&g, &kv, tok[0], &gl);
  for (int i = 1; i < P; i++) {
    gen[i] = sample_argmax(&gl, 0);
    gpt_decode_step(&g, &kv, gen[i], &gl);
  }
  int match = 0;
  for (int i = 0; i < P; i++) if (gen[i] == tok[i]) match++;
  tensor_free(&gl); gpt_kvcache_free(&kv); adam_free(&opt);
  tensor_free(&lo); tensor_free(&pr); gpt_acts_free(&a); gpt_free(&g); free(tgt);
  return match;
}

// Dataset mode: train ONE GPT on N equal-length token sequences (one per media
// file in a folder) and GENERATE a novel sample. A BOS token (index V, so the
// model vocab is V+1) is prepended to every sequence, so the model learns the
// distribution of first tokens too and can generate unconditionally. Generation
// samples P tokens from BOS with temperature + top-k/top-p (sampler.c); BOS is
// never emitted into the media stream. Unlike autoregress_reconstruct (one
// example, greedy), this learns from a set -- quality still needs real scale.
static void dataset_autoregress(int **toks, int n, int P, int V, int steps, int *gen) {
  const int MV = V + 1;      // media vocab + BOS
  const int BOS = V;
  const int L = P + 1;       // BOS + P media tokens
  int *seq = (int*)malloc(L * sizeof(int));
  int *tgt = (int*)malloc(L * sizeof(int));
  srand(1234);
  GPT g = gpt_create(MV, 64, 4, 2, 256, L);
  GPTActs a = gpt_acts_create(&g, L);
  int lz[2] = {L, MV};
  Tensor lo = tensor_create(lz, 2), pr = tensor_create(lz, 2);
  Tensor *par[256];
  int np = gpt_collect_params(&g, par);
  Adam opt = adam_create(par, np, 0.005f);
  int report = (steps >= 10) ? steps / 10 : 1;
  for (int s = 0; s < steps; s++) {
    const int *e = toks[rand() % n];              // random example per step (SGD)
    seq[0] = BOS;
    for (int i = 0; i < P; i++) { seq[i + 1] = e[i]; tgt[i] = e[i]; }
    tgt[P] = -1;                                    // last position predicts nothing
    gpt_forward_train(&g, seq, L, &a, &lo);
    float loss = cross_entropy_forward(&lo, tgt, &pr);
    tensor_zero_grad(&lo);
    cross_entropy_backward(&pr, tgt, &lo);
    gpt_backward(&g, seq, L, &a, &lo);
    adam_step(&opt, par, np);
    if (s % report == 0) fprintf(stderr, "  step %d/%d  loss %.4f\n", s, steps, loss);
  }
  // Generate: feed BOS, then sample P media tokens.
  KVCache kv = gpt_kvcache_create(&g);
  int gz[2] = {1, MV};
  Tensor gl = tensor_create(gz, 2);
  Rng rng = rng_create(1234);
  gpt_decode_step(&g, &kv, BOS, &gl);
  for (int i = 0; i < P; i++) {
    int t = sample_token(&gl, 0, 0.9f, 40, 0.95f, &rng);
    if (t >= V) t = V - 1;                          // never emit BOS into the media stream
    gen[i] = t;
    gpt_decode_step(&g, &kv, t, &gl);
  }
  tensor_free(&gl); gpt_kvcache_free(&kv); adam_free(&opt);
  tensor_free(&lo); tensor_free(&pr); gpt_acts_free(&a); gpt_free(&g);
  free(seq); free(tgt);
}

// imggen dataset mode: load dir/img_0000.ppm, img_0001.ppm, ... (all same size),
// train one GPT on the set, and generate a novel image. Triggered when the input
// path is a folder rather than a single PPM.
static int imggen_dataset(const char *dir, const char *out, int levels, int steps) {
  int cap = 16, n = 0, P = 0, W = 0, H = 0;
  int **toks = (int**)malloc(cap * sizeof(int*));
  for (int i = 0; ; i++) {
    char path[512];
    snprintf(path, sizeof(path), "%s/img_%04d.ppm", dir, i);
    Image im = image_load_ppm(path);
    if (!im.data) break;
    if (n == 0) { W = im.w; H = im.h; P = W * H; }
    else if (im.w != W || im.h != H) {
      fprintf(stderr, "  skip %s: %dx%d != %dx%d\n", path, im.w, im.h, W, H);
      image_free(&im); continue;
    }
    if (n == cap) { cap *= 2; toks = (int**)realloc(toks, cap * sizeof(int*)); }
    int *t = (int*)malloc(P * sizeof(int));
    image_to_tokens(&im, levels, t);
    toks[n++] = t; image_free(&im);
  }
  if (n == 0) {
    fprintf(stderr, "imggen: '%s' is neither a PPM nor a folder of img_0000.ppm, img_0001.ppm, ...\n", dir);
    free(toks); return 1;
  }
  int V = imagegen_vocab(levels);
  if (P > 1024) {
    fprintf(stderr, "images too large (%d px); use <= ~32x32\n", P);
    for (int i = 0; i < n; i++) free(toks[i]);
    free(toks); return 1;
  }
  fprintf(stderr, "imggen dataset: %d images %dx%d, %d tokens, vocab %d(+BOS), %d steps\n",
          n, W, H, P, V, steps);
  int *gen = (int*)malloc(P * sizeof(int));
  dataset_autoregress(toks, n, P, V, steps, gen);
  Image outim = tokens_to_image(gen, W, H, levels);
  int rc = image_save_ppm(out, &outim);
  fprintf(stderr, "generated a %dx%d image from %d examples; wrote %s (%s)\n",
          W, H, n, out, rc ? "FAILED" : "ok");
  image_free(&outim); free(gen);
  for (int i = 0; i < n; i++) free(toks[i]);
  free(toks);
  return rc;
}

static int run_imggen(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: transformer imggen <in.ppm | dir/> <out.ppm> [levels=4] [steps=500]\n"
                    "  single PPM: reconstruct it; a folder of img_0000.ppm,... : learn + generate\n");
    return 2;
  }
  int levels = (argc > 4) ? atoi(argv[4]) : 4;
  int steps = (argc > 5) ? atoi(argv[5]) : 500;
  Image im = image_load_ppm(argv[2]);
  if (!im.data) return imggen_dataset(argv[2], argv[3], levels, steps);
  int P = im.h * im.w, V = imagegen_vocab(levels);
  if (P > 1024) {
    fprintf(stderr, "image too large (%d px); use <= ~32x32 (img_to_ppm.py --size 32)\n", P);
    image_free(&im); return 1;
  }
  fprintf(stderr, "imggen: %dx%d, %d tokens, vocab %d, %d steps\n", im.w, im.h, P, V, steps);
  int *tok = (int*)malloc(P * sizeof(int));
  int *gen = (int*)malloc(P * sizeof(int));
  image_to_tokens(&im, levels, tok);
  int match = autoregress_reconstruct(tok, P, V, steps, gen);
  Image out = tokens_to_image(gen, im.w, im.h, levels);
  int rc = image_save_ppm(argv[3], &out);
  fprintf(stderr, "generated %d/%d tokens; wrote %s (%s)\n",
          match, P, argv[3], rc ? "FAILED" : "ok");
  image_free(&out); image_free(&im); free(tok); free(gen);
  return rc;
}

// audgen dataset mode: load dir/aud_0000.wav, aud_0001.wav, ... (all same length),
// train one GPT on the set, and generate a novel clip. Triggered when the input
// path is a folder rather than a single WAV.
static int audgen_dataset(const char *dir, const char *out, int steps) {
  int cap = 16, n = 0, P = 0, rate = 0;
  int **toks = (int**)malloc(cap * sizeof(int*));
  for (int i = 0; ; i++) {
    char path[512];
    snprintf(path, sizeof(path), "%s/aud_%04d.wav", dir, i);
    Audio a = audio_load_wav(path);
    if (!a.samples) break;
    if (n == 0) { P = a.n; rate = a.sample_rate; }
    else if (a.n != P) {
      fprintf(stderr, "  skip %s: %d samples != %d\n", path, a.n, P);
      audio_free(&a); continue;
    }
    if (n == cap) { cap *= 2; toks = (int**)realloc(toks, cap * sizeof(int*)); }
    int *t = (int*)malloc(P * sizeof(int));
    audio_to_tokens(&a, t);
    toks[n++] = t; audio_free(&a);
  }
  if (n == 0) {
    fprintf(stderr, "audgen: '%s' is neither a WAV nor a folder of aud_0000.wav, aud_0001.wav, ...\n", dir);
    free(toks); return 1;
  }
  int V = audiogen_vocab();
  if (P > 1024) {
    fprintf(stderr, "audio too long (%d samples); use <= ~1024 samples\n", P);
    for (int i = 0; i < n; i++) free(toks[i]);
    free(toks); return 1;
  }
  fprintf(stderr, "audgen dataset: %d clips of %d samples @ %d Hz, vocab %d(+BOS), %d steps\n",
          n, P, rate, V, steps);
  int *gen = (int*)malloc(P * sizeof(int));
  dataset_autoregress(toks, n, P, V, steps, gen);
  Audio outa = tokens_to_audio(gen, P, rate);
  int rc = audio_save_wav(out, &outa);
  fprintf(stderr, "generated a %d-sample clip from %d examples; wrote %s (%s)\n",
          P, n, out, rc ? "FAILED" : "ok");
  audio_free(&outa); free(gen);
  for (int i = 0; i < n; i++) free(toks[i]);
  free(toks);
  return rc;
}

static int run_audgen(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: transformer audgen <in.wav | dir/> <out.wav> [steps=500]\n"
                    "  single WAV: reconstruct it; a folder of aud_0000.wav,... : learn + generate\n");
    return 2;
  }
  int steps = (argc > 4) ? atoi(argv[4]) : 500;
  Audio a = audio_load_wav(argv[2]);
  if (!a.samples) return audgen_dataset(argv[2], argv[3], steps);
  int P = a.n, V = audiogen_vocab();
  if (P > 1024) {
    fprintf(stderr, "audio too long (%d samples); use a short clip (<= ~1024 samples)\n", P);
    audio_free(&a); return 1;
  }
  fprintf(stderr, "audgen: %d samples @ %d Hz, %d tokens, %d steps\n", a.n, a.sample_rate, P, steps);
  int *tok = (int*)malloc(P * sizeof(int));
  int *gen = (int*)malloc(P * sizeof(int));
  audio_to_tokens(&a, tok);
  int match = autoregress_reconstruct(tok, P, V, steps, gen);
  Audio out = tokens_to_audio(gen, P, a.sample_rate);
  int rc = audio_save_wav(argv[3], &out);
  fprintf(stderr, "generated %d/%d tokens; wrote %s (%s)\n",
          match, P, argv[3], rc ? "FAILED" : "ok");
  audio_free(&out); audio_free(&a); free(tok); free(gen);
  return rc;
}

// ============================================================================
// `llama` subcommand: load an LLMA checkpoint (modern LlamaModel — see
// tools/convert_llama.py) and autoregressively GENERATE tokens. Reads a prompt
// as space-separated token IDs per stdin line, samples `n_new` continuation
// tokens (temp/top-k/top-p), and prints `gen: <all ids>`. Token-in/token-out;
// tools/llama_tok.py + llama_chat.py bridge text<->tokens.
//   transformer llama <checkpoint.bin> [n_new=40]
// Note: re-runs the full forward per step (no KV cache yet) — slow on big models.
// ============================================================================
// Parse one stdin line of space-separated token IDs into seq[]; returns count.
static int llama_parse_ids(char *line, int *seq, int vocab, int max_seq) {
  char *p = line;
  if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF) p += 3;
  int n = 0;
  for (char *t = strtok(p, " \t\r\n"); t && n < max_seq; t = strtok(NULL, " \t\r\n")) {
    int id = atoi(t);
    if (id < 0 || id >= vocab) { fprintf(stderr, "skip out-of-range id %d (vocab %d)\n", id, vocab); continue; }
    seq[n++] = id;
  }
  return n;
}

static void llama_print_gen(const int *seq, int n) {
  printf("gen:");
  for (int i = 0; i < n; i++) printf(" %d", seq[i]);
  printf("\n");
  fflush(stdout);
}

// Text-in/text-out generation for an LLMA model paired with one of OUR tokenizers
// (the pairing `llama-train` writes). Mirrors `generate model.bin tok.bin`:
// encode the line, decode-step through it, sample, and stop on any id at or above
// the tokenizer's vocab (the SFT EOS slot).
static int llama_text_chat(LlamaModel *m, const char *tok_path, int n_new,
                           float temperature, int top_k, float top_p, Rng *rng) {
  Tokenizer tok;
  if (tokenizer_load(&tok, tok_path) != 0) {
    fprintf(stderr, "Error: cannot load tokenizer '%s'\n", tok_path);
    return 1;
  }
  fprintf(stderr, "text mode: tokenizer '%s' (vocab %d)\n", tok_path, tok.vocab_size);
  LlamaKV kv = llama_kv_create(m);
  int lv[2] = {1, m->vocab};
  Tensor logits = tensor_create(lv, 2);
  int *seq = (int*)malloc((size_t)m->max_seq * sizeof(int));
  char line[8192];
  char *out = (char*)malloc((size_t)m->max_seq * 8 + 1);

  while (fgets(line, sizeof(line), stdin)) {
    size_t ll = strlen(line);
    while (ll && (line[ll - 1] == '\n' || line[ll - 1] == '\r')) line[--ll] = '\0';
    if (ll == 0) continue;
    int n = tokenizer_encode(&tok, line, seq, m->max_seq);
    if (n <= 0) continue;
    kv.len = 0;
    for (int i = 0; i < n; i++) llama_decode_step(m, &kv, seq[i], &logits);
    int gen0 = n;
    for (int s = 0; s < n_new && n < m->max_seq; s++) {
      int nxt = sample_token(&logits, 0, temperature, top_k, top_p, rng);
      if (nxt >= tok.vocab_size) break;         // EOS slot => stop cleanly
      seq[n++] = nxt;
      llama_decode_step(m, &kv, nxt, &logits);
    }
    tokenizer_decode(&tok, &seq[gen0], n - gen0, out, (size_t)m->max_seq * 8 + 1);
    printf("%s\n", out);
    fflush(stdout);
  }

  free(out); free(seq); tensor_free(&logits); llama_kv_free(&kv); tokenizer_free(&tok);
  return 0;
}

static int run_llama(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: transformer llama <checkpoint.bin> [n_new=40] [tok.bin]\n"
                    "  LLMA (f32) or LLMQ (int8, in-memory quantized) checkpoint;\n"
                    "  reads space-separated token IDs per stdin line; prints 'gen: <ids>'\n"
                    "  Pass a tokenizer (as written by `llama-train`) for TEXT in / TEXT out.\n");
    return 2;
  }
  const int n_new = (argc > 3) ? atoi(argv[3]) : 40;
  const char *tok_path = (argc > 4) ? argv[4] : NULL;

  // Peek the magic to pick the f32 (LLMA) vs int8 in-memory (LLMQ) path.
  FILE *pf = fopen(argv[2], "rb");
  if (!pf) { fprintf(stderr, "Error: cannot open '%s'\n", argv[2]); return 1; }
  unsigned int magic = 0;
  if (fread(&magic, sizeof(unsigned int), 1, pf) != 1) { fclose(pf); fprintf(stderr, "Error: '%s' too small\n", argv[2]); return 1; }
  fclose(pf);
  const int is_q = (magic == 0x514D4C4Cu);   // 'LLMQ'

  const float temperature = 0.8f;
  const int top_k = 40;
  const float top_p = 0.95f;
  Rng rng = rng_create(1234);
  char line[8192];

  if (is_q) {
    QLlamaModel m;
    if (q_llama_model_load(argv[2], &m) != 0) return 1;
    fprintf(stderr, "loaded LLMQ (int8, in-memory): vocab %d d_model %d layers %d heads %d kv %d head_dim %d d_ff %d qk_norm %d\n",
            m.vocab, m.d_model, m.n_layers, m.n_heads, m.n_kv, m.head_dim, m.d_ff, m.qk_norm);
    LlamaKV kv = q_llama_kv_create(&m);
    int lv[2] = {1, m.vocab};
    Tensor logits = tensor_create(lv, 2);
    int *seq = (int*)malloc(m.max_seq * sizeof(int));
    while (fgets(line, sizeof(line), stdin)) {
      int n = llama_parse_ids(line, seq, m.vocab, m.max_seq);
      if (n == 0) continue;
      kv.len = 0;
      for (int i = 0; i < n; i++) q_llama_decode_step(&m, &kv, seq[i], &logits);
      for (int s = 0; s < n_new && n < m.max_seq; s++) {
        int nxt = sample_token(&logits, 0, temperature, top_k, top_p, &rng);
        seq[n++] = nxt;
        printf("t %d\n", nxt); fflush(stdout);   // stream each token as generated
        q_llama_decode_step(&m, &kv, nxt, &logits);
      }
      llama_print_gen(seq, n);
    }
    free(seq); tensor_free(&logits); llama_kv_free(&kv); q_llama_model_free(&m);
  } else {
    LlamaModel m;
    if (llama_model_load(argv[2], &m) != 0) return 1;
    fprintf(stderr, "loaded LLMA (f32): vocab %d d_model %d layers %d heads %d kv %d head_dim %d d_ff %d qk_norm %d tied %d\n",
            m.vocab, m.d_model, m.n_layers, m.n_heads, m.n_kv, m.head_dim, m.d_ff, m.qk_norm, m.tied);
    if (tok_path) {
      int rc = llama_text_chat(&m, tok_path, n_new, temperature, top_k, top_p, &rng);
      llama_model_free(&m);
      return rc;
    }
    LlamaKV kv = llama_kv_create(&m);
    int lv[2] = {1, m.vocab};
    Tensor logits = tensor_create(lv, 2);
    int *seq = (int*)malloc(m.max_seq * sizeof(int));
    while (fgets(line, sizeof(line), stdin)) {
      int n = llama_parse_ids(line, seq, m.vocab, m.max_seq);
      if (n == 0) continue;
      kv.len = 0;
      for (int i = 0; i < n; i++) llama_decode_step(&m, &kv, seq[i], &logits);
      for (int s = 0; s < n_new && n < m.max_seq; s++) {
        int nxt = sample_token(&logits, 0, temperature, top_k, top_p, &rng);
        seq[n++] = nxt;
        printf("t %d\n", nxt); fflush(stdout);   // stream each token as generated
        llama_decode_step(&m, &kv, nxt, &logits);
      }
      llama_print_gen(seq, n);
    }
    free(seq); tensor_free(&logits); llama_kv_free(&kv); llama_model_free(&m);
  }
  return 0;
}

// ============================================================================
// `hfgen` subcommand: pure-C text chat for HF models (GPT-2 / LLaMA-family) using
// the engine's own byte-level BPE tokenizer (hftok) — NO Python at runtime.
// Reads the prompt text from stdin, tokenizes it, generates, and streams each
// token's decoded text as a JSON delta {"d":"piece"} then {"done":true}. The
// generation core (prompt fitting, stop-at-tag, multi-sample) is gen.c.
//   transformer hfgen <model.bin> <tok.hf> [n_new] [temp] [seed]
//                     [--stop-tags] [--samples N] [--constrain[-body|-cmd]]
//                     [--constrain-think N]
// ============================================================================
static int run_hfgen(int argc, char **argv) {
  // Flags may appear anywhere after <tok.hf>; the rest stay positional.
  const char *pos[8];
  int npos = 0, stop_tags = 0, n_samples = 1, bad = 0, constrain = 0;
  ConstrainOpts con = constrain_default_opts();
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--stop-tags") == 0) stop_tags = 1;
    else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) n_samples = atoi(argv[++i]);
    else if (strcmp(argv[i], "--constrain") == 0) constrain = 1;
    else if (strcmp(argv[i], "--constrain-body") == 0) { constrain = 1; con.ground_body = 1; }
    else if (strcmp(argv[i], "--constrain-cmd") == 0) { constrain = 1; con.ground_cmd = 1; }
    else if (strcmp(argv[i], "--constrain-think") == 0 && i + 1 < argc) con.think_max = atoi(argv[++i]);
    else if (strncmp(argv[i], "--", 2) == 0) bad = 1;
    else if (npos < 8) pos[npos++] = argv[i];
  }
  if (npos < 2 || bad || n_samples < 1 || n_samples > 64) {
    fprintf(stderr, "usage: transformer hfgen <model.bin> <tok.hf> [n_new] [temp] [seed]"
                    " [--stop-tags] [--samples N] [--constrain]\n"
                    "  temp defaults to 0.8; use a low value (e.g. 0.1) for an\n"
                    "  instruction-tuned model that should answer deterministically.\n"
                    "  seed defaults to 1234; vary it for different samples at temp>0.\n"
                    "  --stop-tags  stop once the first action block is closed by </tool>\n"
                    "               or </done> (the agent executes only that block).\n"
                    "  --samples N  N samples (1..64) from ONE model load and ONE prompt\n"
                    "               prefill, sample i seeded seed+i; each framed by\n"
                    "               {\"sample\":i,\"seed\":S} ... {\"end\":i}.\n"
                    "  --constrain  constrained decoding: the reply is ONE valid action block\n"
                    "               ([<think>..</think>] + read_file/write_file/run_command/\n"
                    "               <done>) whose path=\"...\" is a whole-token span of the\n"
                    "               prompt; stops when the block is complete. Options:\n"
                    "               --constrain-body (write body = prompt span), --constrain-cmd\n"
                    "               (command args = prompt spans), --constrain-think N (think\n"
                    "               bytes before </think> is forced; default 256, 0 = no cap).\n"
                    "  A prompt longer than the context keeps its most recent tokens (with a\n"
                    "  warning on stderr); if nothing can be generated it prints\n"
                    "  {\"error\":...} and exits nonzero.\n");
    return 2;
  }
  HFGenOpts o = hfgen_default_opts();
  o.n_new = (npos > 2) ? atoi(pos[2]) : 128;
  o.temp = (npos > 3) ? (float)atof(pos[3]) : 0.8f;
  o.seed = (npos > 4) ? strtoull(pos[4], NULL, 10) : 1234ull;
  o.stop_tags = stop_tags;
  o.n_samples = n_samples;
  o.constrain = constrain;
  o.con = con;
  HFTok tok;
  if (hftok_load(&tok, pos[1]) != 0) return 1;

  // Read the WHOLE prompt from stdin (no size cap: an over-long prompt is cut
  // from the FRONT after tokenizing, keeping its newest text) and tokenize it
  // (special-token aware).
  size_t pcap = 1 << 16, plen = 0, r;
  char *prompt = (char*)malloc(pcap);
  while ((r = fread(prompt + plen, 1, pcap - 1 - plen, stdin)) > 0) {
    plen += r;
    if (plen == pcap - 1) { pcap *= 2; prompt = (char*)realloc(prompt, pcap); }
  }
  prompt[plen] = '\0';
  int cap = (int)plen + 16;
  int *pids = (int*)malloc((size_t)cap * sizeof(int));
  int np = hftok_encode(&tok, prompt, pids, cap);
  o.con_prompt = prompt;                     // grounding text for --constrain
  int rc = hfgen_file(&tok, pos[0], pids, np, &o);
  free(prompt);
  free(pids); hftok_free(&tok);
  return rc;
}

// `hfchat`: pure-C chat that applies the model's OWN chat template. Reads the
// conversation from stdin as turns — each turn "role\x1fcontent" separated by
// "\x1e" — renders it via hftok_chat_render, tokenizes, and generates. This is
// the model-agnostic fix for the "rambling / fake conversation" behavior: the
// prompt is in the exact format the model was trained on and it stops on its own
// end token.
//   transformer hfchat <model.bin> <tok.hf> [n_new]
static int run_hfchat(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: transformer hfchat <model.bin> <tok.hf> [n_new]\n");
    return 2;
  }
  int n_new = (argc > 4) ? atoi(argv[4]) : 256;
  HFTok tok;
  if (hftok_load(&tok, argv[3]) != 0) return 1;

  static char in[1 << 16];
  int ilen = (int)fread(in, 1, sizeof(in) - 1, stdin);
  in[ilen] = '\0';

  const char *roles[256], *contents[256];
  int nt = 0;
  char *p = in;
  while (*p && nt < 256) {
    char *rec_end = strchr(p, '\x1e');
    if (rec_end) *rec_end = '\0';
    char *us = strchr(p, '\x1f');
    if (us) { *us = '\0'; roles[nt] = p; contents[nt] = us + 1; nt++; }
    if (!rec_end) break;
    p = rec_end + 1;
  }

  static char rendered[1 << 16];
  hftok_chat_render(&tok, roles, contents, nt, rendered, sizeof(rendered));
  int cap = (int)strlen(rendered) + 16;
  int *pids = (int*)malloc((size_t)cap * sizeof(int));
  int np = hftok_encode(&tok, rendered, pids, cap);
  HFGenOpts o = hfgen_default_opts();
  o.n_new = n_new;
  int rc = hfgen_file(&tok, argv[2], pids, np, &o);
  free(pids); hftok_free(&tok);
  return rc;
}

// ============================================================================
// GPU: report what the backend found, and benchmark it against the CPU.
// ============================================================================

static int run_gpu_info(int argc, char **argv) {
  (void)argc; (void)argv;
  if (!gpu_compiled_in()) {
    printf("GPU backend: NOT COMPILED IN (this is a CPU-only build)\n");
    printf("  Rebuild with:  make GPU=1        (native)\n");
    printf("                 make cross-gpu    (Linux x86_64 binary for Colab)\n");
    return 0;
  }
  printf("GPU backend: compiled in (OpenCL, loaded at runtime)\n");
  const int ok = gpu_available();
  printf("  device: %s\n", gpu_device_name());
  if (!ok) {
    printf("  status: NO USABLE DEVICE -- every matmul will run on the CPU.\n");
    printf("          This is a correct, supported state, not a failure.\n");
    printf("          On Linux/Colab, an OpenCL ICD loader is usually all that is missing:\n");
    printf("            apt-get install -y ocl-icd-libopencl1\n");
    printf("            mkdir -p /etc/OpenCL/vendors\n");
    printf("            echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd\n");
    return 0;
  }
  printf("  compute units: %d\n", gpu_compute_units());
  printf("  global memory: %.1f GB\n", (double)gpu_global_mem() / (1024.0 * 1024.0 * 1024.0));
  printf("  status: READY (enabled=%d)\n", gpu_is_enabled());
  printf("  thresholds: TF_GPU_MIN_M (default 8), TF_GPU_MIN_MACS (default 4194304)\n");
  printf("              Below either one a matmul stays on the CPU on purpose --\n");
  printf("              single-token decode (M=1) is bandwidth-bound and loses on a GPU.\n");
  return 0;
}

// Monotonic wall time in seconds. clock() measures CPU time, which undercounts a
// blocking wait on a device queue -- exactly the thing being measured here.
static double wall_now(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Summarized as the GEOMETRIC MEAN of the per-shape speedups, not as
// sum(cpu)/sum(gpu). Summing absolute times lets one huge shape decide the
// verdict: the lm_head GEMM alone is ~100x the work of a layer projection, so an
// arithmetic summary reported "1.24x, use the GPU" while the GPU was actually
// losing 0.25-0.45x on the three shapes a transformer layer really runs -- and a
// layer's projections execute n_layers times per forward against lm_head's once.
static double g_bench_log_ratio = 0.0;
static int    g_bench_shapes = 0;
static double g_bench_worst_err = 0.0;
static double g_bench_best_ratio = 0.0;

static void bench_one(int M, int K, int N, int reps) {
  int sa[2] = {M, K}, sb[2] = {K, N}, sc[2] = {M, N};
  Tensor A = tensor_create(sa, 2), B = tensor_create(sb, 2), C = tensor_create(sc, 2);
  for (size_t i = 0; i < A.size; i++) A.data[i] = (float)((i * 37) % 19) * 0.01f - 0.09f;
  for (size_t i = 0; i < B.size; i++) B.data[i] = (float)((i * 53) % 23) * 0.01f - 0.11f;

  const double flops = 2.0 * (double)M * (double)N * (double)K * (double)reps;

  gpu_set_enabled(0);
  double t0 = wall_now();
  for (int r = 0; r < reps; r++) matmul(&A, &B, &C);
  const double cpu_s = wall_now() - t0;
  // Keep one CPU result to compare against, so the benchmark also proves the two
  // paths agree -- a fast wrong answer is not a speedup.
  float *ref = (float*)malloc(C.size * sizeof(float));
  memcpy(ref, C.data, C.size * sizeof(float));

  gpu_set_enabled(1);
  const int have = gpu_available();
  double gpu_s = 0.0, maxrel = 0.0;
  if (have) {
    matmul(&A, &B, &C);                     // warm up: first call builds/uploads
    t0 = wall_now();
    for (int r = 0; r < reps; r++) matmul(&A, &B, &C);
    gpu_s = wall_now() - t0;
    for (size_t i = 0; i < C.size; i++) {
      double d = fabs((double)C.data[i] - (double)ref[i]);
      double m = fabs((double)ref[i]);
      double rel = d / (m > 1e-6 ? m : 1e-6);
      if (rel > maxrel) maxrel = rel;
    }
  }

  printf("  %5d x %5d x %5d | CPU %7.1f ms (%6.1f GFLOP/s)", M, K, N,
         cpu_s * 1000.0, flops / cpu_s / 1e9);
  if (have) {
    printf(" | GPU %7.1f ms (%6.1f GFLOP/s) | %5.2fx | max rel err %.1e\n",
           gpu_s * 1000.0, flops / gpu_s / 1e9, cpu_s / gpu_s, maxrel);
    const double ratio = cpu_s / gpu_s;
    g_bench_log_ratio += log(ratio);
    g_bench_shapes++;
    if (ratio > g_bench_best_ratio) g_bench_best_ratio = ratio;
    if (maxrel > g_bench_worst_err) g_bench_worst_err = maxrel;
  } else {
    printf(" | GPU: no device\n");
  }

  free(ref);
  tensor_free(&A); tensor_free(&B); tensor_free(&C);
}

static int run_gpu_bench(int argc, char **argv) {
  int reps = (argc > 2) ? atoi(argv[2]) : 5;
  if (reps < 1) reps = 1;
  printf("matmul benchmark, %d reps per shape (shapes taken from SmolLM2-135M)\n", reps);
  if (!gpu_compiled_in())
    printf("  NOTE: CPU-only build -- this measures the CPU path only.\n");
  else if (!gpu_available())
    printf("  NOTE: no GPU device found (%s) -- CPU path only.\n", gpu_device_name());
  printf("      M x     K x     N |\n");
  bench_one(256, 576, 1536, reps);      // SwiGLU up/gate projection
  bench_one(256, 1536, 576, reps);      // SwiGLU down projection
  bench_one(512, 576, 576, reps);       // attention projections, longer sequence
  bench_one(256, 576, 49152, reps);     // lm_head -- the big one
  printf("\nDecode (M=1) is deliberately NOT benchmarked on the GPU: it is memory-\n"
         "bandwidth-bound, so it stays on the CPU by design (TF_GPU_MIN_M).\n");

  // A benchmark that makes you infer the conclusion is only half a benchmark.
  // "GPU present" does not mean "GPU worth using" -- an integrated GPU sharing
  // the CPU's memory bus routinely loses to a well-threaded CPU.
  if (gpu_compiled_in() && gpu_available() && g_bench_shapes > 0) {
    const double geo = exp(g_bench_log_ratio / g_bench_shapes);
    printf("\nVERDICT: %.2fx typical (geometric mean over %d shapes; best single shape "
           "%.2fx)\n         worst rel err vs CPU: %.1e\n",
           geo, g_bench_shapes, g_bench_best_ratio, g_bench_worst_err);
    if (g_bench_worst_err > 1e-4)
      printf("  !! The GPU results DISAGREE with the CPU beyond tolerance. Do not\n"
             "     train with this device; report the shape that failed.\n");
    else if (geo >= 1.15)
      printf("  Use the GPU on this machine: run training/rollouts as-is.\n");
    else
      printf("  This GPU is SLOWER than this CPU on the shapes a transformer layer\n"
             "  actually runs. Prefer --no-gpu here (or build without GPU=1).\n"
             "  That is the normal result for an INTEGRATED GPU: it shares the CPU's\n"
             "  memory bus and has a fraction of the FLOPs. The GPU build is still\n"
             "  worth having -- run this same command on a discrete GPU (a Colab T4,\n"
             "  say) to see what it is for.\n");
    if (geo < 1.15 && g_bench_best_ratio >= 1.15)
      printf("  (One shape DID win -- the big lm_head GEMM, where this CPU does worst.\n"
             "   It does not carry the verdict: a layer's projections run n_layers\n"
             "   times per forward, lm_head once.)\n");
  }
  return 0;
}

// ---------------------------------------------------------------------------
// simd-bench: the portable C kernels vs the AVX2+FMA ones (matrix.c), same
// binary, same shapes (SmolLM2-135M: d_model 576, d_ff 1536, vocab 49152).
// Reported per shape as the MEDIAN of `reps` timed runs, with the two kernels
// INTERLEAVED run by run, so a background load drifting during the benchmark
// hits both sides alike instead of whichever happened to run second. Thread
// count is whatever $TF_THREADS says -- run it with TF_THREADS=1 to compare
// the kernels rather than the thread pool.
// ---------------------------------------------------------------------------
enum { SB_FWD, SB_BWD, SB_DX, SB_Q8 };

static double sb_median(double *v, int n) {
  for (int i = 1; i < n; i++)                          // insertion sort, n is tiny
    for (int j = i; j > 0 && v[j] < v[j - 1]; j--) { double x = v[j]; v[j] = v[j - 1]; v[j - 1] = x; }
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void sb_one(int op, int M, int K, int N, int reps) {
  int sa[2] = {M, K}, sb[2] = {K, N}, sc[2] = {M, N};
  Tensor A = tensor_create(sa, 2), B = tensor_create(sb, 2), C = tensor_create(sc, 2);
  for (size_t i = 0; i < A.size; i++) A.data[i] = (float)((i * 37) % 19) * 0.01f - 0.09f;
  for (size_t i = 0; i < B.size; i++) B.data[i] = (float)((i * 53) % 23) * 0.01f - 0.11f;
  for (size_t i = 0; i < C.size; i++) C.grad[i] = (float)((i * 29) % 17) * 0.01f - 0.08f;
  signed char *qa = NULL, *qw = NULL;
  int32_t *qacc = NULL;
  if (op == SB_Q8) {
    qa = (signed char*)malloc((size_t)K);
    qw = (signed char*)malloc((size_t)K * N);
    qacc = (int32_t*)malloc((size_t)N * sizeof(int32_t));
    for (int k = 0; k < K; k++) qa[k] = (signed char)((k * 37) % 255 - 127);
    for (size_t i = 0; i < (size_t)K * N; i++) qw[i] = (signed char)((i * 53) % 255 - 127);
  }
  // The output the two kernels are compared on.
  const size_t n_out = (op == SB_FWD) ? C.size : (op == SB_Q8) ? (size_t)N : A.size + B.size;
  float *out[2] = { (float*)malloc(n_out * sizeof(float)), (float*)malloc(n_out * sizeof(float)) };
  double *ts[2] = { (double*)malloc((size_t)reps * sizeof(double)),
                    (double*)malloc((size_t)reps * sizeof(double)) };

  // Tiny products (decode shapes take ~50 us) are repeated inside one timed
  // sample so the clock's resolution is not what gets measured. Backward runs
  // then accumulate `iters` times -- harmless, the error check below uses the
  // last sample, and for it iters is 1 on every shape benchmarked here.
  const double work = (op == SB_BWD ? 4.0 : 2.0) * (double)M * (double)K * (double)N;
  const int iters = (work < 2e8) ? (int)(2e8 / work) : 1;
  for (int r = -1; r < reps; r++) {                    // r = -1 is an untimed warm-up
    for (int mode = 0; mode < 2; mode++) {
      matmul_set_simd(mode);
      if (op == SB_BWD || op == SB_DX) {               // backward ACCUMULATES: reset
        memset(A.grad, 0, A.size * sizeof(float));
        memset(B.grad, 0, B.size * sizeof(float));
      }
      const double t0 = wall_now();
      for (int it = 0; it < iters; it++) {
        switch (op) {
          case SB_FWD: matmul(&A, &B, &C); break;
          case SB_BWD: matmul_backward(&A, &B, &C); break;
          case SB_DX:  matmul_backward_dx(&A, &B, &C); break;
          default:     qacc_columns(qa, qw, K, N, qacc); break;
        }
      }
      const double dt = (wall_now() - t0) / iters;
      if (r >= 0) ts[mode][r] = dt;
      if (r == reps - 1) {
        if (op == SB_FWD) memcpy(out[mode], C.data, C.size * sizeof(float));
        else if (op == SB_Q8) for (int j = 0; j < N; j++) out[mode][j] = (float)qacc[j];
        else {
          memcpy(out[mode], A.grad, A.size * sizeof(float));
          memcpy(out[mode] + A.size, B.grad, B.size * sizeof(float));
        }
      }
    }
  }
  // Error as max|diff| / max|ref| over the whole output (the metric of parts
  // [24]/[25]/[30]/[31]); for dx the B half is untouched zeros on both sides.
  double md = 0.0, mv = 0.0;
  for (size_t i = 0; i < n_out; i++) {
    const double d = fabs((double)out[1][i] - (double)out[0][i]);
    if (d > md) md = d;
    if (fabs((double)out[0][i]) > mv) mv = fabs((double)out[0][i]);
  }
  const double tp = sb_median(ts[0], reps), tv = sb_median(ts[1], reps);
  static const char *names[] = { "fwd ", "bwd ", "dx  ", "int8" };
  printf("  %s %4d x %5d x %5d | portable %8.2f ms %6.1f G/s | avx2 %8.2f ms %6.1f G/s | %5.2fx | err %.1e\n",
         names[op], M, K, N, tp * 1e3, work / tp / 1e9, tv * 1e3, work / tv / 1e9,
         tp / tv, md / (mv + 1e-30));
  fflush(stdout);
  free(out[0]); free(out[1]); free(ts[0]); free(ts[1]);
  free(qa); free(qw); free(qacc);
  tensor_free(&A); tensor_free(&B); tensor_free(&C);
}

static int run_simd_bench(int argc, char **argv) {
  int reps = (argc > 2) ? atoi(argv[2]) : 5;
  if (reps < 1) reps = 1;
  const char *thr = getenv("TF_THREADS");
  printf("SIMD matmul benchmark: portable C vs AVX2+FMA, median of %d interleaved runs\n", reps);
  printf("  threads: TF_THREADS=%s\n", thr ? thr : "(unset: all cores)");
#if defined(__AVX2__) && defined(__FMA__)
  printf("  portable C in THIS binary was compiled for AVX2+FMA (a native build), so it\n"
         "  is already auto-vectorized: this measures register blocking, not SIMD width.\n");
#else
  printf("  portable C in THIS binary was compiled for the baseline ISA (no -mavx2), as\n"
         "  in every cross/portable build: the AVX2 path comes from runtime dispatch.\n");
#endif
  if (!matmul_simd_available()) {
    printf("  this CPU/build has no AVX2+FMA path -- nothing to compare.\n");
    return 0;
  }
  printf("  G/s = GFLOP/s (int8: GOP/s, 1 MAC = 2 ops); err = max|diff| / max|ref|.\n");
  const int T = 128;                                   // one training sequence
  sb_one(SB_FWD, T, 576, 1536, reps);                  // SwiGLU gate / up
  sb_one(SB_FWD, T, 1536, 576, reps);                  // SwiGLU down
  sb_one(SB_FWD, T, 576, 576, reps);                   // attention projections
  sb_one(SB_FWD, T, 576, 49152, reps);                 // lm_head
  sb_one(SB_FWD, 1, 576, 1536, reps);                  // single-token decode (M=1)
  sb_one(SB_FWD, 1, 1536, 576, reps);
  sb_one(SB_FWD, 1, 576, 49152, reps);
  sb_one(SB_BWD, T, 576, 1536, reps);                  // full backward: dA + dB
  sb_one(SB_BWD, T, 1536, 576, reps);
  sb_one(SB_BWD, T, 576, 576, reps);
  sb_one(SB_DX, T, 576, 49152, reps);                  // lm_head is frozen under LoRA
  sb_one(SB_Q8, 1, 576, 1536, reps);                   // int8 decode (qmatmul's core)
  sb_one(SB_Q8, 1, 1536, 576, reps);
  sb_one(SB_Q8, 1, 576, 49152, reps);
  matmul_set_simd(1);
  return 0;
}

int main(int argc, char **argv) {
  // Global --gpu / --no-gpu, honoured by every subcommand. They are stripped from
  // argv before dispatch so the per-subcommand parsers never see them (they
  // reject unknown flags). --no-gpu on a GPU build is how you get an honest A/B.
  static char *fargv[256];
  int fargc = 0;
  for (int i = 0; i < argc && fargc < 256; i++) {
    if (i > 0 && strcmp(argv[i], "--no-gpu") == 0) { gpu_set_enabled(0); continue; }
    if (i > 0 && strcmp(argv[i], "--gpu") == 0)    { gpu_set_enabled(1); continue; }
    fargv[fargc++] = argv[i];
  }
  argv = fargv;
  argc = fargc;

  if (argc > 1 && strcmp(argv[1], "gpu-info") == 0) return run_gpu_info(argc, argv);
  if (argc > 1 && strcmp(argv[1], "gpu-bench") == 0) return run_gpu_bench(argc, argv);
  if (argc > 1 && strcmp(argv[1], "simd-bench") == 0) return run_simd_bench(argc, argv);
  if (argc > 1 && strcmp(argv[1], "hfgen") == 0) return run_hfgen(argc, argv);
  if (argc > 1 && strcmp(argv[1], "hfchat") == 0) return run_hfchat(argc, argv);
  if (argc > 1 && strcmp(argv[1], "llama") == 0) return run_llama(argc, argv);
  if (argc > 1 && strcmp(argv[1], "generate") == 0) return run_generate(argc, argv);
  if (argc > 1 && strcmp(argv[1], "agent") == 0) return run_agent(argc, argv);
  if (argc > 1 && strcmp(argv[1], "risk") == 0) return run_risk(argc, argv);
  if (argc > 1 && strcmp(argv[1], "train") == 0) return run_train(argc, argv);
  if (argc > 1 && strcmp(argv[1], "imggen") == 0) return run_imggen(argc, argv);
  if (argc > 1 && strcmp(argv[1], "audgen") == 0) return run_audgen(argc, argv);
  if (argc > 1 && strcmp(argv[1], "caption") == 0) return run_caption(argc, argv);
  if (argc > 1 && strcmp(argv[1], "text2media") == 0) return run_text2media(argc, argv);
  if (argc > 1 && strcmp(argv[1], "t2i") == 0) return run_t2i(argc, argv);
  if (argc > 1 && strcmp(argv[1], "t2i-data") == 0) return run_t2i_data(argc, argv);
  if (argc > 1 && strcmp(argv[1], "t2i-recon") == 0) return run_t2i_recon(argc, argv);
  if (argc > 1 && strcmp(argv[1], "grpo") == 0) return run_grpo(argc, argv);
  if (argc > 1 && strcmp(argv[1], "llama-train") == 0) return run_llama_train(argc, argv);
  if (argc > 1 && strcmp(argv[1], "t2i-train") == 0) return run_t2i_train(argc, argv);
  if (argc > 1 && strcmp(argv[1], "t2i-gen") == 0) return run_t2i_gen(argc, argv);
  if (argc > 1 && strcmp(argv[1], "vqvae") == 0) return run_vqvae(argc, argv);
  if (argc > 1 && strcmp(argv[1], "hftok") == 0) return run_hftok(argc, argv);
  // Bare `transformer` (no args) runs the built-in self-test. An UNRECOGNIZED
  // subcommand must NOT fall through to it (a typo like `trian` would otherwise
  // silently run the demo) — print usage and fail instead.
  if (argc > 1) {
    fprintf(stderr, "unknown command '%s'\n", argv[1]);
    fprintf(stderr, "usage: transformer [train|llama-train|grpo|generate|agent|imggen|audgen|llama] ...\n");
    fprintf(stderr, "       transformer [gpu-info|gpu-bench]   (GPU backend status / CPU-vs-GPU timing)\n");
    fprintf(stderr, "       transformer simd-bench [reps]      (portable C vs AVX2+FMA kernel timing)\n");
    fprintf(stderr, "       transformer            (no args: run the built-in self-test)\n");
    return 1;
  }
  return run_selftest();
}
