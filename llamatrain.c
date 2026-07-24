// llamatrain.c - Training forward/backward for the LLaMA-family model (llama.h)
#include "llamatrain.h"

#include <stdlib.h>

#include "matrix.h"
#include "rmsnorm.h"
#include "rope.h"
#include "gqa.h"
#include "swiglu.h"

// Every activation here is an intermediate that gradients flow through, so all of
// them are allocated WITH a grad array (unlike the inference tensors in llama.c).
static Tensor act(int r, int c) { int s[2] = {r, c}; return tensor_create(s, 2); }

static Tensor *layer_acts(int n_layers, int rows, int cols) {
  Tensor *arr = (Tensor*)malloc((size_t)n_layers * sizeof(Tensor));
  for (int l = 0; l < n_layers; l++) arr[l] = act(rows, cols);
  return arr;
}

static void free_layer_acts(Tensor *arr, int n_layers) {
  for (int l = 0; l < n_layers; l++) tensor_free(&arr[l]);
  free(arr);
}

LlamaActs llama_acts_create(const LlamaModel *m, int seq_len) {
  LlamaActs a;
  const int L = m->n_layers, dm = m->d_model;
  const int qd = m->n_heads * m->head_dim, kvd = m->n_kv * m->head_dim;
  a.seq_len = seq_len;
  a.n_layers = L;
  a.emb = act(seq_len, dm);
  a.final_hn = act(seq_len, dm);
  a.xn = layer_acts(L, seq_len, dm);
  a.q = layer_acts(L, seq_len, qd);
  a.k = layer_acts(L, seq_len, kvd);
  a.v = layer_acts(L, seq_len, kvd);
  a.o = layer_acts(L, seq_len, qd);
  a.ao = layer_acts(L, seq_len, dm);
  a.h = layer_acts(L, seq_len, dm);
  a.hn = layer_acts(L, seq_len, dm);
  a.f = layer_acts(L, seq_len, dm);
  a.block_out = layer_acts(L, seq_len, dm);
  a.gc = (GqaCache*)malloc((size_t)L * sizeof(GqaCache));
  a.sc = (SwigluCache*)malloc((size_t)L * sizeof(SwigluCache));
  for (int l = 0; l < L; l++) {
    a.gc[l] = gqa_cache_create(seq_len, m->n_heads, m->n_kv, m->head_dim);
    a.sc[l] = swiglu_cache_create(seq_len, m->d_ff);
  }
  return a;
}

void llama_acts_free(LlamaActs *a) {
  tensor_free(&a->emb);
  tensor_free(&a->final_hn);
  free_layer_acts(a->xn, a->n_layers);
  free_layer_acts(a->q, a->n_layers);
  free_layer_acts(a->k, a->n_layers);
  free_layer_acts(a->v, a->n_layers);
  free_layer_acts(a->o, a->n_layers);
  free_layer_acts(a->ao, a->n_layers);
  free_layer_acts(a->h, a->n_layers);
  free_layer_acts(a->hn, a->n_layers);
  free_layer_acts(a->f, a->n_layers);
  free_layer_acts(a->block_out, a->n_layers);
  for (int l = 0; l < a->n_layers; l++) {
    gqa_cache_free(&a->gc[l]);
    swiglu_cache_free(&a->sc[l]);
  }
  free(a->gc); free(a->sc);
  a->gc = NULL; a->sc = NULL;
}

static void llama_acts_zero_grad(LlamaActs *a) {
  tensor_zero_grad(&a->emb);
  tensor_zero_grad(&a->final_hn);
  for (int l = 0; l < a->n_layers; l++) {
    tensor_zero_grad(&a->xn[l]);
    tensor_zero_grad(&a->q[l]);
    tensor_zero_grad(&a->k[l]);
    tensor_zero_grad(&a->v[l]);
    tensor_zero_grad(&a->o[l]);
    tensor_zero_grad(&a->ao[l]);
    tensor_zero_grad(&a->h[l]);
    tensor_zero_grad(&a->hn[l]);
    tensor_zero_grad(&a->f[l]);
    tensor_zero_grad(&a->block_out[l]);
  }
}

// ---------------------------------------------------------------------------
// Forward (caching)
// ---------------------------------------------------------------------------

void llama_forward_train(LlamaModel *m, const int *ids, int seq_len,
                         LlamaActs *a, Tensor *logits) {
  const int dm = m->d_model;

  // Token embedding. No positional term — RoPE carries position inside attention.
  for (int i = 0; i < seq_len; i++)
    for (int j = 0; j < dm; j++)
      a->emb.data[i * dm + j] = m->wte.data[ids[i] * dm + j];

  for (int l = 0; l < m->n_layers; l++) {
    LlamaBlock *b = &m->blocks[l];
    const Tensor *x = (l == 0) ? &a->emb : &a->block_out[l - 1];

    // --- Attention sub-layer ---
    rmsnorm(x, &b->attn_norm, &a->xn[l], b->eps);
    matmul(&a->xn[l], &b->Wq, &a->q[l]); bias_add(&a->q[l], &b->b_q, &a->q[l]);
    matmul(&a->xn[l], &b->Wk, &a->k[l]); bias_add(&a->k[l], &b->b_k, &a->k[l]);
    matmul(&a->xn[l], &b->Wv, &a->v[l]); bias_add(&a->v[l], &b->b_v, &a->v[l]);
    rope_heads(&a->q[l], b->n_heads, b->rope_base);   // in place: q/k become POST-RoPE
    rope_heads(&a->k[l], b->n_kv, b->rope_base);
    gqa_forward(&a->q[l], &a->k[l], &a->v[l], &a->gc[l], &a->o[l]);
    matmul(&a->o[l], &b->Wo, &a->ao[l]);
    matadd(x, &a->ao[l], &a->h[l]);                   // residual

    // --- Feed-forward sub-layer ---
    rmsnorm(&a->h[l], &b->ffn_norm, &a->hn[l], b->eps);
    swiglu_forward(&a->hn[l], &b->Wgate, &b->Wup, &b->Wdown, &a->sc[l], &a->f[l]);
    matadd(&a->h[l], &a->f[l], &a->block_out[l]);     // residual
  }

  rmsnorm(&a->block_out[m->n_layers - 1], &m->final_norm, &a->final_hn, m->eps);
  matmul(&a->final_hn, &m->lm_head, logits);          // (seq_len, vocab)
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

void llama_backward(LlamaModel *m, const int *ids, int seq_len,
                    LlamaActs *a, Tensor *logits) {
  const int dm = m->d_model;

  llama_zero_grad(m);
  llama_acts_zero_grad(a);

  // logits = final_hn @ lm_head
  matmul_backward(&a->final_hn, &m->lm_head, logits);
  // final_hn = RMSNorm(block_out[L-1], final_norm)
  rmsnorm_backward(&a->block_out[m->n_layers - 1], &m->final_norm, &a->final_hn,
                   &m->final_norm, m->eps);

  for (int l = m->n_layers - 1; l >= 0; l--) {
    LlamaBlock *b = &m->blocks[l];
    const Tensor *x = (l == 0) ? &a->emb : &a->block_out[l - 1];

    // --- Feed-forward sub-layer (reverse) ---
    matadd_backward(&a->block_out[l], &a->h[l], &a->f[l]);
    swiglu_backward(&a->hn[l], &b->Wgate, &b->Wup, &b->Wdown, &a->sc[l], &a->f[l]);
    // Completes h->grad: the FFN branch adds to the residual term already there.
    rmsnorm_backward(&a->h[l], &b->ffn_norm, &a->hn[l], &b->ffn_norm, b->eps);

    // --- Attention sub-layer (reverse) ---
    matadd_backward(&a->h[l], x, &a->ao[l]);          // dx += dh (residual), dao += dh
    matmul_backward(&a->o[l], &b->Wo, &a->ao[l]);
    gqa_backward(&a->q[l], &a->k[l], &a->v[l], &a->gc[l], &a->o[l]);
    // RoPE is in place, so this REPLACES q/k grads with their pre-rotation form.
    rope_heads_backward(&a->q[l], b->n_heads, b->rope_base);
    rope_heads_backward(&a->k[l], b->n_kv, b->rope_base);
    bias_add_backward(&a->q[l], &b->b_q);
    bias_add_backward(&a->k[l], &b->b_k);
    bias_add_backward(&a->v[l], &b->b_v);
    matmul_backward(&a->xn[l], &b->Wq, &a->q[l]);
    matmul_backward(&a->xn[l], &b->Wk, &a->k[l]);
    matmul_backward(&a->xn[l], &b->Wv, &a->v[l]);
    rmsnorm_backward(x, &b->attn_norm, &a->xn[l], &b->attn_norm, b->eps);
  }

  // Embedding: scatter each position's gradient into its token's row of wte.
  for (int i = 0; i < seq_len; i++)
    for (int j = 0; j < dm; j++)
      m->wte.grad[ids[i] * dm + j] += a->emb.grad[i * dm + j];
}

// ---------------------------------------------------------------------------
// Parameter helpers
// ---------------------------------------------------------------------------

int llama_param_count(const LlamaModel *m) {
  return 3 + 12 * m->n_layers;   // wte, final_norm, lm_head + 12 per layer
}

int llama_collect_params(LlamaModel *m, Tensor **list) {
  int n = 0;
  list[n++] = &m->wte;
  for (int l = 0; l < m->n_layers; l++) {
    LlamaBlock *b = &m->blocks[l];
    list[n++] = &b->attn_norm;
    list[n++] = &b->Wq;
    list[n++] = &b->Wk;
    list[n++] = &b->Wv;
    list[n++] = &b->b_q;
    list[n++] = &b->b_k;
    list[n++] = &b->b_v;
    list[n++] = &b->Wo;
    list[n++] = &b->ffn_norm;
    list[n++] = &b->Wgate;
    list[n++] = &b->Wup;
    list[n++] = &b->Wdown;
  }
  list[n++] = &m->final_norm;
  list[n++] = &m->lm_head;
  return n;
}

void llama_zero_grad(LlamaModel *m) {
  Tensor **list = (Tensor**)malloc((size_t)llama_param_count(m) * sizeof(Tensor*));
  int n = llama_collect_params(m, list);
  for (int i = 0; i < n; i++) tensor_zero_grad(list[i]);
  free(list);
}
