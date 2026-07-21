// main.c - Minimal GPT-2 Style Transformer in Pure C11 (demo entry point)
// Build: make        (see README.md and Makefile)
// Or:    gcc -std=c11 -O3 -march=native -o transformer *.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tensor.h"
#include "model.h"
#include "checkpoint.h"
#include "tokenizer.h"

// ============================================================================
// Demo: byte-level BPE tokenizer + checkpoint save/load + generation
// ============================================================================

int main() {
  printf("=== Minimal GPT-2 Style Transformer Demo ===\n\n");

  int all_ok = 1;  // gates the final validation line

  // --------------------------------------------------------------------------
  // Part 1: Byte-level BPE tokenizer round-trip
  // --------------------------------------------------------------------------
  printf("[1] Byte-level BPE tokenizer\n");
  Tokenizer tok = tokenizer_create_bytelevel();

  // Teach it a few merges so encoding actually compresses.
  //   'l'+'l' -> "ll",  'e'+'ll' -> "ell",  'h'+'ell' -> "hell",  'o'+' ' -> "o "
  int ll = tokenizer_add_merge(&tok, 'l', 'l');
  int ell = tokenizer_add_merge(&tok, 'e', ll);
  int hell = tokenizer_add_merge(&tok, 'h', ell);
  (void)hell;

  const char *text = "hello hello";
  int ids[64];
  int n_ids = tokenizer_encode(&tok, text, ids, 64);

  char decoded[128];
  tokenizer_decode(&tok, ids, n_ids, decoded, sizeof(decoded));

  int byte_len = (int)strlen(text);
  int roundtrip_ok = (strcmp(text, decoded) == 0);
  int compressed = (n_ids < byte_len);

  printf("    text     : \"%s\" (%d bytes)\n", text, byte_len);
  printf("    encoded  : %d tokens ->", n_ids);
  for (int i = 0; i < n_ids; i++) printf(" %d", ids[i]);
  printf("\n");
  printf("    decoded  : \"%s\"\n", decoded);
  printf("    round-trip: %s | compressed: %s\n\n",
         roundtrip_ok ? "OK" : "FAIL", compressed ? "yes" : "no");
  all_ok &= roundtrip_ok && compressed;

  tokenizer_free(&tok);

  // --------------------------------------------------------------------------
  // Part 2: Build a model, generate, and checkpoint round-trip
  // --------------------------------------------------------------------------
  printf("[2] GPT model\n");
  const int vocab_size = 16;
  const int d_model = 8;
  const int n_heads = 2;
  const int n_layers = 2;
  const int d_ff = 4 * d_model;
  const int max_seq_len = 16;

  srand(42);
  GPT model = gpt_create(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len);
  printf("    config: vocab=%d d_model=%d heads=%d layers=%d d_ff=%d\n",
         vocab_size, d_model, n_heads, n_layers, d_ff);

  // Greedy generation from a small prompt
  const int prompt_len = 3;
  const int n_new = 5;
  int seq[16];
  seq[0] = 1; seq[1] = 5; seq[2] = 9;
  gpt_generate(&model, seq, prompt_len, n_new);
  printf("    generate: 1 5 9 ->");
  for (int i = 0; i < prompt_len + n_new; i++) printf(" %d", seq[i]);
  printf("\n");

  // Reference logits from the original model
  int total_len = prompt_len + n_new;
  int logits_shape[2] = {total_len, vocab_size};
  Tensor logits_a = tensor_create(logits_shape, 2);
  gpt_forward(&model, seq, total_len, &logits_a);

  // NaN/Inf validation
  int logits_valid = 1;
  for (int i = 0; i < logits_a.size; i++) {
    if (isnan(logits_a.data[i]) || isinf(logits_a.data[i])) { logits_valid = 0; break; }
  }
  printf("    logits  : %s\n\n", logits_valid ? "no NaN/Inf" : "contains NaN/Inf!");
  all_ok &= logits_valid;

  // --------------------------------------------------------------------------
  // Part 3: Checkpoint save/load round-trip (reloaded logits must be identical)
  // --------------------------------------------------------------------------
  printf("[3] Checkpoint save/load\n");
  const char *path = "model.bin";
  int save_ok = (gpt_save(path, &model) == 0);

  GPT loaded;
  int load_ok = save_ok && (gpt_load(path, &loaded) == 0);

  int ckpt_ok = 0;
  if (load_ok) {
    Tensor logits_b = tensor_create(logits_shape, 2);
    gpt_forward(&loaded, seq, total_len, &logits_b);
    ckpt_ok = (memcmp(logits_a.data, logits_b.data, logits_a.size * sizeof(float)) == 0);
    tensor_free(&logits_b);
    gpt_free(&loaded);
  }
  printf("    saved '%s': %s | reloaded logits identical: %s\n\n",
         path, save_ok ? "OK" : "FAIL", ckpt_ok ? "yes" : "no");
  all_ok &= save_ok && load_ok && ckpt_ok;

  // --------------------------------------------------------------------------
  // Final result
  // --------------------------------------------------------------------------
  printf("Output validation: %s\n\n",
         all_ok ? "PASSED (no NaN/Inf)" : "FAILED");

  printf("Cleaning up memory...\n");
  tensor_free(&logits_a);
  gpt_free(&model);
  remove(path);
  printf("Done! All memory freed successfully.\n");

  return all_ok ? 0 : 1;
}
