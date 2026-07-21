// train.h - Train a GPT on a text file or directory
#ifndef TRAIN_H
#define TRAIN_H

// Model + training hyperparameters. Use train_default_config() then override.
typedef struct {
  const char *out_model;   // where to write the model checkpoint
  const char *out_tok;     // where to write the tokenizer
  int num_merges;          // BPE merges (vocab = 256 + num_merges)
  int steps;               // Adam training steps
  int d_model;             // model / embedding dimension
  int n_heads;             // attention heads (must divide d_model)
  int n_layers;            // transformer blocks
  int d_ff;                // feed-forward hidden dim (<=0 => 4 * d_model)
  int ctx;                 // context window = model max_seq_len
  float lr;                // Adam learning rate
  unsigned seed;           // RNG seed for reproducibility
} TrainConfig;

// Sensible defaults (a small-but-real model).
TrainConfig train_default_config(void);

// Train on data_path (a text file OR a directory of *.txt) using cfg, then save
// the model to cfg->out_model and tokenizer to cfg->out_tok.
// Returns 0 on success, nonzero on error.
int train_run(const char *data_path, const TrainConfig *cfg);

#endif // TRAIN_H
