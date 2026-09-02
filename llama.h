// llama.h - A modern (LLaMA-family) transformer block, assembled from the
// gradient-checked primitives: RMSNorm + RoPE + GQA + SwiGLU. No biases.
//
//   xn  = RMSNorm(x, attn_norm)
//   q,k,v = xn @ {Wq, Wk, Wv}          (RoPE applied per-head to q and k)
//   h   = x + GQA(q, k, v) @ Wo        (residual)
//   hn  = RMSNorm(h, ffn_norm)
//   out = h + SwiGLU(hn, Wgate, Wup, Wdown)   (residual)
//
// This module is the forward/inference path (loading and running real small
// modern models). The matching training path — cached-activation forward,
// backward, and parameter collection — lives in llamatrain.h.
#ifndef LLAMA_H
#define LLAMA_H

#include "tensor.h"
#include "quant.h"
#include "lora.h"

typedef struct {
  int d_model, n_heads, n_kv, head_dim, d_ff;
  float rope_base, eps;
  // Attention
  Tensor attn_norm;   // (d_model)
  Tensor Wq;          // (d_model, n_heads*head_dim)
  Tensor Wk;          // (d_model, n_kv*head_dim)
  Tensor Wv;          // (d_model, n_kv*head_dim)
  Tensor b_q;         // (n_heads*head_dim)  q/k/v biases (zero for LLaMA/SmolLM;
  Tensor b_k;         // (n_kv*head_dim)      nonzero for Qwen2). o_proj has none.
  Tensor b_v;         // (n_kv*head_dim)
  Tensor Wo;          // (n_heads*head_dim, d_model)
  // Feed-forward (SwiGLU)
  Tensor ffn_norm;    // (d_model)
  Tensor Wgate;       // (d_model, d_ff)
  Tensor Wup;         // (d_model, d_ff)
  Tensor Wdown;       // (d_ff, d_model)
} LlamaBlock;

// Allocate a block with the given config (weights are zeroed — fill them via the
// returned tensors, or use llama_block_randomize for testing).
LlamaBlock llama_block_create(int d_model, int n_heads, int n_kv, int head_dim,
                              int d_ff, float rope_base, float eps);

// Same, but every weight also gets a gradient array so the block can be TRAINED
// (see llamatrain.h). Inference-only blocks skip the grads to halve their memory.
LlamaBlock llama_block_create_train(int d_model, int n_heads, int n_kv, int head_dim,
                                    int d_ff, float rope_base, float eps);
void llama_block_free(LlamaBlock *b);

// Fill all weights with small deterministic pseudo-random values (for tests).
void llama_block_randomize(LlamaBlock *b);

// Forward: x (seq_len, d_model) -> out (seq_len, d_model). Causal.
void llama_block_forward(const LlamaBlock *b, const Tensor *x, Tensor *out);

// ---------------------------------------------------------------------------
// Full modern model: token embedding -> n_layers blocks -> final RMSNorm ->
// output head. No positional embedding (RoPE handles position inside attention).
// ---------------------------------------------------------------------------
typedef struct {
  int vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq;
  float rope_base, eps;
  Tensor wte;          // (vocab, d_model)   token embedding
  LlamaBlock *blocks;  // n_layers
  Tensor final_norm;   // (d_model)          RMSNorm before the head
  Tensor lm_head;      // (d_model, vocab)    output projection (tie = wte^T)
} LlamaModel;

LlamaModel llama_model_create(int vocab, int d_model, int n_layers, int n_heads,
                              int n_kv, int head_dim, int d_ff, int max_seq,
                              float rope_base, float eps);

// Same, but with gradient arrays on every weight — a TRAINABLE model (llamatrain.h).
LlamaModel llama_model_create_train(int vocab, int d_model, int n_layers, int n_heads,
                                    int n_kv, int head_dim, int d_ff, int max_seq,
                                    float rope_base, float eps);
void llama_model_free(LlamaModel *m);
void llama_model_randomize(LlamaModel *m);

// Forward: token ids (length seq_len) -> logits (seq_len, vocab). Causal.
void llama_model_forward(const LlamaModel *m, const int *ids, int seq_len, Tensor *logits);

// Checkpoint I/O: flat binary 'LLMA' format (magic + version + config + every
// weight's float32 in a fixed canonical order shared by save and load).
// llama_model_load builds a matching model, then reads the weights into it.
// Return 0 on success, nonzero on error.
int llama_model_save(const char *path, const LlamaModel *m);
int llama_model_load(const char *path, LlamaModel *out);

// ---------------------------------------------------------------------------
// KV cache for incremental decode: caches K and V (n_kv * head_dim per position)
// for every layer, so each new token computes only its own Q/K/V and attends
// over the cache instead of re-running the whole prefix. O(n) per token.
// ---------------------------------------------------------------------------
typedef struct {
  int n_layers, max_seq, kv_dim;   // kv_dim = n_kv * head_dim
  int len;                         // number of positions currently cached
  float **k;                       // n_layers x (max_seq * kv_dim)
  float **v;
} LlamaKV;

LlamaKV llama_kv_create(const LlamaModel *m);
void llama_kv_free(LlamaKV *kv);

// Decode ONE token at position kv->len: append its K/V to the cache, attend over
// positions 0..kv->len, and write logits (1, vocab). Advances kv->len.
void llama_decode_step(const LlamaModel *m, LlamaKV *kv, int token, Tensor *logits);

// A read-only view of the four per-layer adapter arrays of a LlamaLora, so the
// decode path can apply adapters WITHOUT depending on llamatrain.h (which
// includes this header). Build one with llama_lora_decode_view (llamatrain.h),
// or by hand: each field is either NULL or an array of n_layers adapters.
typedef struct {
  const LoraAdapter *q, *k, *v, *o;
} LlamaDecodeLora;

// KV-cache decode WITH LoRA adapters applied — the incremental counterpart of
// llama_forward_train_lora, needed to SAMPLE from an adapted policy without
// re-running the whole prefix per token (which is what makes RL rollouts
// affordable; see grpo.h). Passing lo == NULL is exactly llama_decode_step.
//
// The adapters must be applied in the same places and the same order as the
// training forward — after the q/k/v bias but BEFORE RoPE, and onto the output
// projection's result — or the two paths disagree and the importance ratio a
// policy-gradient update computes is meaningless.
void llama_decode_step_lora(const LlamaModel *m, LlamaKV *kv, int token,
                            Tensor *logits, const LlamaDecodeLora *lo);

// ---------------------------------------------------------------------------
// Quantized (int8, in-memory) model: same architecture, but the big weights
// (embedding, all block linears, output head) are kept as int8 QTensors and
// consumed via qmatmul — 4x less RAM and less memory traffic. Norms stay f32.
// Loads the 'LLMQ' checkpoint WITHOUT dequantizing. Decode-only (for generation).
// ---------------------------------------------------------------------------
typedef struct {
  Tensor attn_norm, ffn_norm;                      // f32
  Tensor b_q, b_k, b_v;                            // f32 q/k/v biases (small; kept f32)
  QTensor Wq, Wk, Wv, Wo, Wgate, Wup, Wdown;       // int8
} QLlamaBlock;

typedef struct {
  int vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq;
  float rope_base, eps;
  QTensor wte;          // int8 (dequantized one row per token at embed time)
  QLlamaBlock *blocks;
  Tensor final_norm;    // f32
  QTensor lm_head;      // int8
} QLlamaModel;

void q_llama_model_free(QLlamaModel *m);

// Build a quantized model by quantizing an fp32 LlamaModel's weights (for tests).
QLlamaModel q_llama_quantize_from(const LlamaModel *src);

// Load an 'LLMQ' checkpoint into a quantized model (int8 kept in memory).
// Returns 0 on success, nonzero on error (incl. a non-LLMQ file).
int q_llama_model_load(const char *path, QLlamaModel *out);

// Incremental decode with the (f32) KV cache, using int8 weights via qmatmul.
void q_llama_decode_step(const QLlamaModel *m, LlamaKV *kv, int token, Tensor *logits);
LlamaKV q_llama_kv_create(const QLlamaModel *m);

#endif // LLAMA_H
