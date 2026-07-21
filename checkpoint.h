// checkpoint.h - Save/load GPT model weights to a flat binary file
#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "model.h"

// Binary checkpoint layout (little-endian, host float32):
//   magic  uint32 = 'GPTC'
//   version uint32 = 1
//   config: vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len (6 x int32)
//   every weight tensor's raw float32 data, in a fixed canonical order

// Save all model weights to path. Returns 0 on success, nonzero on error.
int gpt_save(const char *path, const GPT *m);

// Load a model from path. Reads the config, allocates the model (gpt_create),
// then fills every weight from the file. On success *out holds a model the
// caller must release with gpt_free. Returns 0 on success, nonzero on error.
int gpt_load(const char *path, GPT *out);

#endif // CHECKPOINT_H
