// checkpoint.c - Save/load GPT model weights to a flat binary file
#include "checkpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define CKPT_MAGIC   0x43545047u  // 'GPTC' (little-endian bytes G,P,T,C)
#define CKPT_VERSION 1u

// Maximum number of weight tensors we ever collect: 2 embeddings + 12 per layer
// + 2 final-LayerNorm + 1 output head.
#define TENSORS_PER_LAYER 12

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

int gpt_save(const char *path, const GPT *m) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "Error: cannot open '%s' for writing\n", path);
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
  fclose(f);

  if (!ok) {
    fprintf(stderr, "Error: failed while writing '%s'\n", path);
    return 1;
  }
  return 0;
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
