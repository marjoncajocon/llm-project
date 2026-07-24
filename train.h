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
  float lr;                // Adam learning rate (peak LR when a schedule is on)
  unsigned seed;           // RNG seed for reproducibility
  const char *resume;      // NULL => fresh model; else path to a checkpoint to continue
  int checkpoint_every;    // save every N steps (0 => only at the end)
  float val_frac;          // fraction of tokens held out for validation loss (e.g. 0.1)
  int sft;                 // 1 => masked instruction-tuning (SFT) mode (prompt<TAB>response lines)
  // --- Training-recipe options (all default to OFF, so runs stay reproducible) ---
  float weight_decay;      // AdamW decoupled decay on weight matrices (0 => plain Adam)
  float grad_clip;         // global grad-norm clip (0 => off; 1.0 is the usual choice)
  int warmup;              // linear warmup steps (0 => none)
  float lr_min;            // cosine floor LR; < 0 => constant LR (no cosine decay)
  // --- LLaMA-family only (llama_train_run); ignored by the GPT trainer ---
  int n_kv;                // KV heads for GQA (0 => n_heads, i.e. plain multi-head)
  float rope_base;         // RoPE frequency base (10000 is standard)
  int lora_r;              // LoRA rank; > 0 => freeze the base, train adapters only
  float lora_alpha;        // LoRA scaling (scale = alpha / r); 0 => use r (scale 1)
  const char *hf_tok;      // .hf/.tok vocab of a REAL model (hftok.h): fine-tune with
                           // the model's OWN tokenizer instead of training a byte BPE
} TrainConfig;

// Sensible defaults (a small-but-real model).
TrainConfig train_default_config(void);

// Train on data_path (a text file OR a directory of *.txt) using cfg, then save
// the model to cfg->out_model and tokenizer to cfg->out_tok.
// Returns 0 on success, nonzero on error.
int train_run(const char *data_path, const TrainConfig *cfg);

// Same, but trains the MODERN (LLaMA-family) architecture — RMSNorm + RoPE + GQA
// + SwiGLU, via llamatrain.h — and writes an 'LLMA' checkpoint that
// `transformer llama` can generate from. Honours the same corpus/val-split,
// checkpointing, resume, SFT and recipe options, plus cfg->n_kv and
// cfg->rope_base. head_dim is d_model / n_heads.
// Returns 0 on success, nonzero on error.
int llama_train_run(const char *data_path, const TrainConfig *cfg);

#endif // TRAIN_H
