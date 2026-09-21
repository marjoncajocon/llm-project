// llama.h - A modern (LLaMA-family) transformer block, assembled from the
// gradient-checked primitives: RMSNorm + RoPE + GQA + SwiGLU. No biases.
//
//   xn  = RMSNorm(x, attn_norm)
//   q,k,v = xn @ {Wq, Wk, Wv}          (RoPE applied per-head to q and k)
//   [QK-norm, Qwen3: q,k = RMSNorm per head (q_norm/k_norm) BEFORE RoPE]
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
  int qk_norm;        // 1 = Qwen3 QK-norm on (q_norm/k_norm used); 0 = LLaMA/Qwen2
  int window;         // sliding-window attention size W (gqa.h); <= 0 = full causal.
                      // 0 after create; set per model via llama_model_set_window.
  // Attention
  Tensor attn_norm;   // (d_model)
  Tensor Wq;          // (d_model, n_heads*head_dim)
  Tensor Wk;          // (d_model, n_kv*head_dim)
  Tensor Wv;          // (d_model, n_kv*head_dim)
  Tensor b_q;         // (n_heads*head_dim)  q/k/v biases (zero for LLaMA/SmolLM;
  Tensor b_k;         // (n_kv*head_dim)      nonzero for Qwen2). o_proj has none.
  Tensor b_v;         // (n_kv*head_dim)
  Tensor q_norm;      // (head_dim)  QK-norm gains, shared by every head; applied
  Tensor k_norm;      // (head_dim)  only when qk_norm (always allocated: tiny)
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
  int qk_norm;         // per-head q/k RMSNorm (Qwen3); mirrored in every block
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

// A base for LoRA fine-tuning, whose weights stay FROZEN: gradient arrays only
// on the 1-D gains/biases (the dx-only backward still accumulates into those),
// none on wte, lm_head or any projection -- for SmolLM-135M that is ~650 MB of
// gradient memory never allocated, and never re-zeroed by llama_zero_grad. The
// llamatrain.h backward runs such a model through the dx-only path always.
// Never hand its matrices to an optimizer or a full-model backward.
LlamaModel llama_model_create_train_frozen(int vocab, int d_model, int n_layers, int n_heads,
                                           int n_kv, int head_dim, int d_ff, int max_seq,
                                           float rope_base, float eps);
void llama_model_free(LlamaModel *m);
void llama_model_randomize(LlamaModel *m);

// Turn QK-norm on/off for the model and every block. Models are created with it
// OFF (the LLaMA/Qwen2 behavior); turning it on resets every q_norm/k_norm gain
// to 1 — load/copy the real gains afterwards. It also changes the parameter set
// (llama_param_count), so set it BEFORE collecting params or building acts.
void llama_model_set_qk_norm(LlamaModel *m, int on);

// Sliding-window attention, per layer. The REPRESENTATION is one int per layer
// (LlamaBlock.window: W > 0 = that layer attends only to its W most recent
// positions, <= 0 = full causal attention), because real models disagree about
// the pattern — Mistral windows every layer, Gemma 2 / gpt-oss alternate
// local/global, Gemma 3 runs 5 local : 1 global, Qwen2 windows only the upper
// layers — and a per-layer value covers all of them. This helper builds the
// common "every Nth layer is global" pattern:
//   window <= 0          -> every layer full attention (the default; turns it off)
//   global_every <= 0    -> every layer windowed (Mistral)
//   global_every = N > 0 -> layer l is GLOBAL when (l + 1) % N == 0, else windowed
//                           (N = 2: local,global,local,...  = Gemma 2 / gpt-oss;
//                            N = 6: 5 local then 1 global    = Gemma 3)
// Any other pattern: set m->blocks[l].window directly.
void llama_model_set_window(LlamaModel *m, int window, int global_every);

// Copy the per-layer windows of src onto dst (same n_layers). The trainers rebuild
// a loaded checkpoint as a trainable model; this keeps its window config.
void llama_model_copy_window(LlamaModel *dst, const LlamaModel *src);

// Forward: token ids (length seq_len) -> logits (seq_len, vocab). Causal.
void llama_model_forward(const LlamaModel *m, const int *ids, int seq_len, Tensor *logits);

// Checkpoint I/O: flat binary 'LLMA' format (magic + version + config + every
// weight's float32 in a fixed canonical order shared by save and load).
// llama_model_load builds a matching model, then reads the weights into it.
// Versions (the writer picks the lowest one that can hold the model):
//   v2  no optional feature (byte-identical to before, so older builds and
//       tools keep reading it);
//   v3  = v2 + n_layers int32 sliding windows right after the float config;
//   v4  = v2 + a u32 feature-flags word after the float config (bit 0 = QK-norm,
//       bit 1 = per-layer windows present), then the n_layers int32 windows iff
//       bit 1; with bit 0 each layer stores q_norm, k_norm right after b_v.
//       Written whenever QK-norm is on.
// v2/v3/v4 all load; unknown versions or flag bits are rejected.
// Return 0 on success, nonzero on error.
int llama_model_save(const char *path, const LlamaModel *m);
int llama_model_load(const char *path, LlamaModel *out);

// ---------------------------------------------------------------------------
// KV cache for incremental decode: caches K and V (n_kv * head_dim per position)
// for every layer, so each new token computes only its own Q/K/V and attends
// over the cache instead of re-running the whole prefix. O(n) per token.
// ---------------------------------------------------------------------------
//
// A sliding-window layer only ever reads its last W positions, so its cache is
// a RING BUFFER of cap = min(W, max_seq) slots (position p lives in slot p % cap)
// — memory for that layer is bounded by W, not max_seq. Full-attention layers
// keep cap = max_seq (slot p % max_seq == p). Decode visits positions in the same
// order as the full forward, so it stays bit-exact with or without the window.
typedef struct {
  int n_layers, max_seq, kv_dim;   // kv_dim = n_kv * head_dim
  int len;                         // number of positions currently cached (the
                                   // next token's absolute position; reset to 0
                                   // to start a new sequence)
  int *cap;                        // n_layers: slots in that layer's cache
  float **k;                       // n_layers x (cap[l] * kv_dim)
  float **v;
} LlamaKV;

LlamaKV llama_kv_create(const LlamaModel *m);
// Same, but holding only max_len positions (clamped to 1..m->max_seq): full layers
// get max_len slots, windowed ones min(W, max_len). For a cache that will never
// grow past max_len — e.g. a prompt-prefill template — at a fraction of the RAM.
LlamaKV llama_kv_create_len(const LlamaModel *m, int max_len);
void llama_kv_free(LlamaKV *kv);

// Copy src's cached state into dst: len plus every layer's POPULATED slots
// (min(len, cap[l]) per layer — correct for sliding-window ring buffers too,
// since positions 0..len-1 occupy a slot prefix). Both caches must come from
// the same model (n_layers, kv_dim), dst must hold src->len positions, and each
// layer must map positions 0..len-1 to the same slots — equal caps, or a ring
// that has not wrapped in either (so a llama_kv_create_len template can fork
// into a full llama_kv_create cache); otherwise it aborts. Slots beyond the populated prefix are left untouched: they
// are never read, because decode only visits positions < len (and writes a
// slot before reading it). Used to prefill a shared prompt once and fork it
// (grpo.c: one prefill per group, copied into each of the G rollouts).
void llama_kv_copy(LlamaKV *dst, const LlamaKV *src);

// Decode ONE token at position kv->len: append its K/V to the cache, attend over
// positions 0..kv->len, and write logits (1, vocab). Advances kv->len.
// logits == NULL skips the final norm and lm_head (the costliest matmul of a
// small model): prompt prefill needs only the LAST position's logits. The KV
// cache is written exactly as with logits.
void llama_decode_step(const LlamaModel *m, LlamaKV *kv, int token, Tensor *logits);

// A read-only view of the four per-layer adapter arrays of a LlamaLora, so the
// decode path can apply adapters WITHOUT depending on llamatrain.h (which
// includes this header). Build one with llama_lora_decode_view (llamatrain.h),
// or by hand: q/k/v/o must be arrays of n_layers adapters; gate/up/down are
// either NULL (no MLP adapters) or arrays of n_layers adapters.
typedef struct {
  const LoraAdapter *q, *k, *v, *o;
  const LoraAdapter *gate, *up, *down;
} LlamaDecodeLora;

// KV-cache decode WITH LoRA adapters applied — the incremental counterpart of
// llama_forward_train_lora, needed to SAMPLE from an adapted policy without
// re-running the whole prefix per token (which is what makes RL rollouts
// affordable; see grpo.h). Passing lo == NULL is exactly llama_decode_step.
//
// The adapters must be applied in the same places and the same order as the
// training forward — after the q/k/v bias but BEFORE QK-norm and RoPE, onto
// the output projection's result, and (MLP adapters) onto g and u before SiLU and
// onto the down projection's result — or the two paths disagree and the importance
// ratio a policy-gradient update computes is meaningless.
void llama_decode_step_lora(const LlamaModel *m, LlamaKV *kv, int token,
                            Tensor *logits, const LlamaDecodeLora *lo);

// ---------------------------------------------------------------------------
// Quantized (int8, in-memory) model: same architecture, but the big weights
// (embedding, all block linears, output head) are kept as int8 QTensors and
// consumed via qmatmul — 4x less RAM and less memory traffic. Norms stay f32.
// Loads the 'LLMQ' checkpoint WITHOUT dequantizing. Decode-only (for generation).
// ---------------------------------------------------------------------------
typedef struct {
  int window;                                      // sliding window (as LlamaBlock)
  Tensor attn_norm, ffn_norm;                      // f32
  Tensor b_q, b_k, b_v;                            // f32 q/k/v biases (small; kept f32)
  Tensor q_norm, k_norm;                           // f32 QK-norm gains (head_dim)
  QTensor Wq, Wk, Wv, Wo, Wgate, Wup, Wdown;       // int8
} QLlamaBlock;

typedef struct {
  int vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq;
  float rope_base, eps;
  int qk_norm;          // Qwen3 per-head q/k RMSNorm (see LlamaModel)
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
