// main.c - Minimal GPT-2 Style Transformer in Pure C11 (demo entry point)
// Build: make        (see README.md and Makefile)
// Or:    gcc -std=c11 -O3 -march=native -o transformer *.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
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
#include "caption.h"
#include "text2media.h"
#include "t2i.h"
#include "vqvae.h"
#include "hftok.h"
#include "image.h"
#include "patch.h"
#include "video.h"
#include "audio.h"
#include "imagegen.h"
#include "audiogen.h"
#include "llama.h"
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
// `agent` subcommand: the parse -> approve -> execute -> observe loop with a
// pluggable model backend.
//   transformer agent                 -> built-in scripted backend (demo)
//   transformer agent --cmd "<prog>"  -> run <prog> as the model; its stdout is
//                                        each response (the transcript is passed
//                                        as a file arg). Plug in a real LLM here.
//   --ask   approve each tool action interactively   --goal "..."  set the goal
// ============================================================================
static int run_agent(int argc, char **argv) {
  const char *goal = "Create a note file, then read it back to verify.";
  const char *cmd = NULL;
  ApprovalMode mode = APPROVAL_AUTO;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--ask") == 0) mode = APPROVAL_ASK;
    else if (strcmp(argv[i], "--cmd") == 0 && i + 1 < argc) cmd = argv[++i];
    else if (strcmp(argv[i], "--goal") == 0 && i + 1 < argc) goal = argv[++i];
  }

  int rc;
  if (cmd) {
    CommandBackend b = command_backend(cmd);
    rc = agent_run(goal, &b.base, mode);
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
    rc = agent_run(goal, &b.base, mode);
  }
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
    "  --name X      save to model/X/ (model.bin + tok.bin); use `generate --name X`\n"
    "  --out F       output model (default model.bin)   --tok F     output tokenizer (default tok.bin)\n"
    "  --merges N    BPE merges (default 300)            --steps N   Adam steps (default 1000)\n"
    "  --d-model N   model dim (default 128)             --layers N  blocks (default 3)\n"
    "  --heads N     attention heads (default 4)         --d-ff N    FFN dim (default 4*d_model)\n"
    "  --ctx N       context window (default 64)         --lr F      learning rate (default 3e-4)\n"
    "  --seed N      RNG seed (default 1234)             --val-frac F  held-out fraction (default 0.1)\n"
    "  --resume F    continue training a saved model     --checkpoint-every N  save every N steps\n"
    "  --sft         masked instruction tuning: file is 'prompt<TAB>response' lines,\n"
    "                loss on the response only (true SFT). See data/sft.txt.\n");
}

static int run_train(int argc, char **argv) {
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
  return train_run(data, &cfg);
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

static int run_llama(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: transformer llama <checkpoint.bin> [n_new=40]\n"
                    "  LLMA (f32) or LLMQ (int8, in-memory quantized) checkpoint;\n"
                    "  reads space-separated token IDs per stdin line; prints 'gen: <ids>'\n");
    return 2;
  }
  const int n_new = (argc > 3) ? atoi(argv[3]) : 40;

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
    fprintf(stderr, "loaded LLMQ (int8, in-memory): vocab %d d_model %d layers %d heads %d kv %d head_dim %d d_ff %d\n",
            m.vocab, m.d_model, m.n_layers, m.n_heads, m.n_kv, m.head_dim, m.d_ff);
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
    fprintf(stderr, "loaded LLMA (f32): vocab %d d_model %d layers %d heads %d kv %d head_dim %d d_ff %d\n",
            m.vocab, m.d_model, m.n_layers, m.n_heads, m.n_kv, m.head_dim, m.d_ff);
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

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "llama") == 0) return run_llama(argc, argv);
  if (argc > 1 && strcmp(argv[1], "generate") == 0) return run_generate(argc, argv);
  if (argc > 1 && strcmp(argv[1], "agent") == 0) return run_agent(argc, argv);
  if (argc > 1 && strcmp(argv[1], "train") == 0) return run_train(argc, argv);
  if (argc > 1 && strcmp(argv[1], "imggen") == 0) return run_imggen(argc, argv);
  if (argc > 1 && strcmp(argv[1], "audgen") == 0) return run_audgen(argc, argv);
  if (argc > 1 && strcmp(argv[1], "caption") == 0) return run_caption(argc, argv);
  if (argc > 1 && strcmp(argv[1], "text2media") == 0) return run_text2media(argc, argv);
  if (argc > 1 && strcmp(argv[1], "t2i") == 0) return run_t2i(argc, argv);
  if (argc > 1 && strcmp(argv[1], "t2i-train") == 0) return run_t2i_train(argc, argv);
  if (argc > 1 && strcmp(argv[1], "t2i-gen") == 0) return run_t2i_gen(argc, argv);
  if (argc > 1 && strcmp(argv[1], "vqvae") == 0) return run_vqvae(argc, argv);
  if (argc > 1 && strcmp(argv[1], "hftok") == 0) return run_hftok(argc, argv);
  // Bare `transformer` (no args) runs the built-in self-test. An UNRECOGNIZED
  // subcommand must NOT fall through to it (a typo like `trian` would otherwise
  // silently run the demo) — print usage and fail instead.
  if (argc > 1) {
    fprintf(stderr, "unknown command '%s'\n", argv[1]);
    fprintf(stderr, "usage: transformer [train|generate|agent|imggen|audgen|llama] ...\n");
    fprintf(stderr, "       transformer            (no args: run the built-in self-test)\n");
    return 1;
  }
  return run_selftest();
}
