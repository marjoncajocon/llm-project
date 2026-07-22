// checkpoint.c - Save/load GPT model weights to a flat binary file
#include "checkpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

#define CKPT_MAGIC   0x43545047u  // 'GPTC' (little-endian bytes G,P,T,C)
#define CKPT_VERSION 2u           // v2 adds attention biases (b_q/b_k/b_v/b_o)

// Maximum number of weight tensors we ever collect: 2 embeddings + 16 per layer
// + 2 final-LayerNorm + 1 output head.
#define TENSORS_PER_LAYER 16

// List every weight tensor of the model, in a fixed canonical order shared by
// save and load (so the two can never drift). Returns the tensor count.
static int collect_tensors(const GPT *m, const Tensor **list) {
  int n = 0;
  list[n++] = &m->wte;
  list[n++] = &m->wpe;
  for (int l = 0; l < m->n_layers; l++) {
    const BlockWeights *b = &m->blocks[l];
    list[n++] = &b->W_q;
    list[n++] = &b->W_k;
    list[n++] = &b->W_v;
    list[n++] = &b->W_o;
    list[n++] = &b->b_q;
    list[n++] = &b->b_k;
    list[n++] = &b->b_v;
    list[n++] = &b->b_o;
    list[n++] = &b->gamma1;
    list[n++] = &b->beta1;
    list[n++] = &b->gamma2;
    list[n++] = &b->beta2;
    list[n++] = &b->ff_W1;
    list[n++] = &b->ff_b1;
    list[n++] = &b->ff_W2;
    list[n++] = &b->ff_b2;
  }
  list[n++] = &m->ln_f_gamma;
  list[n++] = &m->ln_f_beta;
  list[n++] = &m->lm_head;
  return n;
}

// Allocate a tensor-pointer list large enough for this model's weights.
static const Tensor **alloc_tensor_list(const GPT *m) {
  int max = 2 + TENSORS_PER_LAYER * m->n_layers + 3;
  return (const Tensor**)malloc(max * sizeof(const Tensor*));
}

// Atomically replace dst with tmp: rename is atomic on POSIX; on Windows rename
// fails if dst exists, so fall back to remove+rename. Returns 0 on success.
static int replace_file(const char *tmp, const char *dst) {
  if (rename(tmp, dst) == 0) return 0;
  remove(dst);
  return rename(tmp, dst);
}

int gpt_save(const char *path, const GPT *m) {
  // Write to a temp file, then atomically rename over `path`, so a failed or
  // interrupted save never truncates an existing good checkpoint.
  char *tmp = (char*)malloc(strlen(path) + 5);
  if (!tmp) { fprintf(stderr, "Error: out of memory saving '%s'\n", path); return 1; }
  sprintf(tmp, "%s.tmp", path);

  FILE *f = fopen(tmp, "wb");
  if (!f) {
    fprintf(stderr, "Error: cannot open '%s' for writing\n", tmp);
    free(tmp);
    return 1;
  }

  uint32_t header[2] = {CKPT_MAGIC, CKPT_VERSION};
  int32_t config[6] = {
      m->vocab_size, m->d_model, m->n_heads,
      m->n_layers, m->d_ff, m->max_seq_len};

  int ok = 1;
  ok &= (fwrite(header, sizeof(uint32_t), 2, f) == 2);
  ok &= (fwrite(config, sizeof(int32_t), 6, f) == 6);

  const Tensor **list = alloc_tensor_list(m);
  int count = collect_tensors(m, list);
  for (int i = 0; i < count && ok; i++) {
    size_t size = (size_t)list[i]->size;
    ok &= (fwrite(list[i]->data, sizeof(float), size, f) == size);
  }
  free(list);
  ok &= (fflush(f) == 0);
  ok &= (fclose(f) == 0);          // catch deferred write errors (e.g. disk full)

  if (ok) ok = (replace_file(tmp, path) == 0);
  if (!ok) {
    remove(tmp);
    fprintf(stderr, "Error: failed while writing '%s'\n", path);
    free(tmp);
    return 1;
  }
  free(tmp);
  return 0;
}

// Tensor.size is size_t (allocation is overflow-safe), but the embedding/forward
// path still indexes these weights with int arithmetic (e.g. token*d_model), so a
// weight with more than INT_MAX elements would overflow those indices. Reject such
// configs up front until the forward index math is migrated to size_t too.
static int prod_fits_int(long long a, long long b) {
  return a > 0 && b > 0 && a <= (long long)INT_MAX / b;
}

// Validate the 6 config ints of a GPTC header before building a model from them.
// config = {vocab, d_model, n_heads, n_layers, d_ff, max_seq}. Guards against
// corrupt/hostile checkpoints (negative/zero/huge dims, non-divisible heads,
// element counts that overflow int). Returns 1 if sane, 0 otherwise.
static int gpt_config_valid(const int32_t *c) {
  const int32_t MAXD = 1000000;
  for (int i = 0; i < 6; i++)
    if (c[i] <= 0 || c[i] > MAXD) return 0;
  int vocab = c[0], d_model = c[1], n_heads = c[2], d_ff = c[4];
  if (d_model % n_heads != 0) return 0;                 // heads split d_model evenly
  if (!prod_fits_int(vocab, d_model)) return 0;         // wte / lm_head
  if (!prod_fits_int(d_model, d_ff)) return 0;          // FFN weights
  return 1;
}

int gpt_load(const char *path, GPT *out) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Error: cannot open '%s' for reading\n", path);
    return 1;
  }

  uint32_t header[2];
  int32_t config[6];
  if (fread(header, sizeof(uint32_t), 2, f) != 2 ||
      fread(config, sizeof(int32_t), 6, f) != 6) {
    fprintf(stderr, "Error: '%s' is truncated (bad header)\n", path);
    fclose(f);
    return 1;
  }
  if (header[0] != CKPT_MAGIC || header[1] != CKPT_VERSION) {
    fprintf(stderr, "Error: '%s' is not a v%u GPTC checkpoint\n", path, CKPT_VERSION);
    fclose(f);
    return 1;
  }
  if (!gpt_config_valid(config)) {
    fprintf(stderr, "Error: '%s' has an invalid/unsupported config\n", path);
    fclose(f);
    return 1;
  }

  // Build a matching model, then overwrite its weights from the file.
  *out = gpt_create(config[0], config[1], config[2], config[3], config[4], config[5]);

  int ok = 1;
  const Tensor **list = alloc_tensor_list(out);
  int count = collect_tensors(out, list);
  for (int i = 0; i < count && ok; i++) {
    size_t size = (size_t)list[i]->size;
    ok &= (fread(list[i]->data, sizeof(float), size, f) == size);
  }
  free(list);
  fclose(f);

  if (!ok) {
    fprintf(stderr, "Error: '%s' is truncated (bad weights)\n", path);
    gpt_free(out);
    return 1;
  }
  return 0;
}
