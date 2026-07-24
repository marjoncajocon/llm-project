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

// Reduce a tensor's logical shape to `rows` (<= its allocated rows). Row-major
// stride is shape[1] (cols), independent of the row count, so shrinking shape[0]
// and size just makes every shape-driven op process fewer rows — the underlying
// buffer (sized for the create-time max) is untouched.
static void set_rows(Tensor *t, int rows) {
  t->shape[0] = rows;
  t->size = (size_t)rows * (size_t)t->shape[1];
}

void llama_acts_set_len(LlamaActs *a, int seq_len) {
  a->seq_len = seq_len;
  set_rows(&a->emb, seq_len);
  set_rows(&a->final_hn, seq_len);
  for (int l = 0; l < a->n_layers; l++) {
    set_rows(&a->xn[l], seq_len);
    set_rows(&a->q[l], seq_len);
    set_rows(&a->k[l], seq_len);
    set_rows(&a->v[l], seq_len);
    set_rows(&a->o[l], seq_len);
    set_rows(&a->ao[l], seq_len);
    set_rows(&a->h[l], seq_len);
    set_rows(&a->hn[l], seq_len);
    set_rows(&a->f[l], seq_len);
    set_rows(&a->block_out[l], seq_len);
    a->gc[l].n = seq_len;                 // GqaCache strides its att buffer by n
    set_rows(&a->sc[l].g, seq_len);       // SwigluCache tensors stride by shape[0]
    set_rows(&a->sc[l].u, seq_len);
    set_rows(&a->sc[l].a, seq_len);
    set_rows(&a->sc[l].h, seq_len);
  }
}

void llama_lora_acts_set_len(LlamaLoraActs *la, int seq_len) {
  for (int l = 0; l < la->n_layers; l++) {
    set_rows(&la->q[l].xa, seq_len); set_rows(&la->q[l].y, seq_len);
    set_rows(&la->k[l].xa, seq_len); set_rows(&la->k[l].y, seq_len);
    set_rows(&la->v[l].xa, seq_len); set_rows(&la->v[l].y, seq_len);
    set_rows(&la->o[l].xa, seq_len); set_rows(&la->o[l].y, seq_len);
  }
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

void llama_forward_train_lora(LlamaModel *m, const int *ids, int seq_len,
                              LlamaActs *a, Tensor *logits,
                              const LlamaLora *lo, LlamaLoraActs *la) {
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
    if (lo) {                                         // low-rank corrections on q/k/v
      lora_forward(&lo->q[l], &a->xn[l], &a->q[l], &la->q[l]);
      lora_forward(&lo->k[l], &a->xn[l], &a->k[l], &la->k[l]);
      lora_forward(&lo->v[l], &a->xn[l], &a->v[l], &la->v[l]);
    }
    rope_heads(&a->q[l], b->n_heads, b->rope_base);   // in place: q/k become POST-RoPE
    rope_heads(&a->k[l], b->n_kv, b->rope_base);
    gqa_forward(&a->q[l], &a->k[l], &a->v[l], &a->gc[l], &a->o[l]);
    matmul(&a->o[l], &b->Wo, &a->ao[l]);
    if (lo) lora_forward(&lo->o[l], &a->o[l], &a->ao[l], &la->o[l]);
    matadd(x, &a->ao[l], &a->h[l]);                   // residual

    // --- Feed-forward sub-layer ---
    rmsnorm(&a->h[l], &b->ffn_norm, &a->hn[l], b->eps);
    swiglu_forward(&a->hn[l], &b->Wgate, &b->Wup, &b->Wdown, &a->sc[l], &a->f[l]);
    matadd(&a->h[l], &a->f[l], &a->block_out[l]);     // residual
  }

  rmsnorm(&a->block_out[m->n_layers - 1], &m->final_norm, &a->final_hn, m->eps);
  matmul(&a->final_hn, &m->lm_head, logits);          // (seq_len, vocab)
}

void llama_forward_train(LlamaModel *m, const int *ids, int seq_len,
                         LlamaActs *a, Tensor *logits) {
  llama_forward_train_lora(m, ids, seq_len, a, logits, NULL, NULL);
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

// zero_params = 0 leaves the parameter gradients alone so several micro-batches
// accumulate into one effective batch (activation grads are still zeroed — those
// belong to the current sequence).
static void llama_backward_impl(LlamaModel *m, const int *ids, int seq_len,
                                LlamaActs *a, Tensor *logits,
                                const LlamaLora *lo, LlamaLoraActs *la,
                                int zero_params) {
  const int dm = m->d_model;

  if (zero_params) {
    llama_zero_grad(m);
    if (lo) llama_lora_zero_grad(lo);
  }
  llama_acts_zero_grad(a);

  // Under LoRA the base is frozen, so every base weight gradient is dead work:
  // take the input-gradient-only path for each big projection (roughly halves
  // the backward) and skip the embedding scatter entirely.
  const int frozen = (lo != NULL);

  // logits = final_hn @ lm_head
  if (frozen) matmul_backward_dx(&a->final_hn, &m->lm_head, logits);
  else        matmul_backward(&a->final_hn, &m->lm_head, logits);
  // final_hn = RMSNorm(block_out[L-1], final_norm)
  rmsnorm_backward(&a->block_out[m->n_layers - 1], &m->final_norm, &a->final_hn,
                   &m->final_norm, m->eps);

  for (int l = m->n_layers - 1; l >= 0; l--) {
    LlamaBlock *b = &m->blocks[l];
    const Tensor *x = (l == 0) ? &a->emb : &a->block_out[l - 1];

    // --- Feed-forward sub-layer (reverse) ---
    matadd_backward(&a->block_out[l], &a->h[l], &a->f[l]);
    if (frozen) swiglu_backward_dx(&a->hn[l], &b->Wgate, &b->Wup, &b->Wdown, &a->sc[l], &a->f[l]);
    else        swiglu_backward(&a->hn[l], &b->Wgate, &b->Wup, &b->Wdown, &a->sc[l], &a->f[l]);
    // Completes h->grad: the FFN branch adds to the residual term already there.
    rmsnorm_backward(&a->h[l], &b->ffn_norm, &a->hn[l], &b->ffn_norm, b->eps);

    // --- Attention sub-layer (reverse) ---
    matadd_backward(&a->h[l], x, &a->ao[l]);          // dx += dh (residual), dao += dh
    if (frozen) matmul_backward_dx(&a->o[l], &b->Wo, &a->ao[l]);
    else        matmul_backward(&a->o[l], &b->Wo, &a->ao[l]);
    if (lo) lora_backward(&lo->o[l], &a->o[l], &a->ao[l], &la->o[l]);
    gqa_backward(&a->q[l], &a->k[l], &a->v[l], &a->gc[l], &a->o[l]);
    // RoPE is in place, so this REPLACES q/k grads with their pre-rotation form.
    rope_heads_backward(&a->q[l], b->n_heads, b->rope_base);
    rope_heads_backward(&a->k[l], b->n_kv, b->rope_base);
    bias_add_backward(&a->q[l], &b->b_q);
    bias_add_backward(&a->k[l], &b->b_k);
    bias_add_backward(&a->v[l], &b->b_v);
    if (frozen) {
      matmul_backward_dx(&a->xn[l], &b->Wq, &a->q[l]);
      matmul_backward_dx(&a->xn[l], &b->Wk, &a->k[l]);
      matmul_backward_dx(&a->xn[l], &b->Wv, &a->v[l]);
    } else {
      matmul_backward(&a->xn[l], &b->Wq, &a->q[l]);
      matmul_backward(&a->xn[l], &b->Wk, &a->k[l]);
      matmul_backward(&a->xn[l], &b->Wv, &a->v[l]);
    }
    if (lo) {   // q/k/v adapters saw the SAME xn, and add into its gradient too
      lora_backward(&lo->q[l], &a->xn[l], &a->q[l], &la->q[l]);
      lora_backward(&lo->k[l], &a->xn[l], &a->k[l], &la->k[l]);
      lora_backward(&lo->v[l], &a->xn[l], &a->v[l], &la->v[l]);
    }
    rmsnorm_backward(x, &b->attn_norm, &a->xn[l], &b->attn_norm, b->eps);
  }

  // Embedding: scatter each position's gradient into its token's row of wte.
  // Skipped when the base is frozen — nothing consumes wte's gradient.
  if (!frozen)
    for (int i = 0; i < seq_len; i++)
      for (int j = 0; j < dm; j++)
        m->wte.grad[ids[i] * dm + j] += a->emb.grad[i * dm + j];
}

void llama_backward_lora(LlamaModel *m, const int *ids, int seq_len,
                         LlamaActs *a, Tensor *logits,
                         const LlamaLora *lo, LlamaLoraActs *la) {
  llama_backward_impl(m, ids, seq_len, a, logits, lo, la, 1);
}

void llama_backward_accum(LlamaModel *m, const int *ids, int seq_len,
                          LlamaActs *a, Tensor *logits,
                          const LlamaLora *lo, LlamaLoraActs *la) {
  llama_backward_impl(m, ids, seq_len, a, logits, lo, la, 0);
}

void llama_backward(LlamaModel *m, const int *ids, int seq_len,
                    LlamaActs *a, Tensor *logits) {
  llama_backward_impl(m, ids, seq_len, a, logits, NULL, NULL, 1);
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

// ---------------------------------------------------------------------------
// LoRA adapters over a LlamaModel
// ---------------------------------------------------------------------------

static LoraAdapter *adapters(int n_layers, int d_in, int d_out, int r, float alpha) {
  LoraAdapter *arr = (LoraAdapter*)malloc((size_t)n_layers * sizeof(LoraAdapter));
  for (int l = 0; l < n_layers; l++) arr[l] = lora_create(d_in, d_out, r, alpha);
  return arr;
}

static void free_adapters(LoraAdapter *arr, int n_layers) {
  for (int l = 0; l < n_layers; l++) lora_free(&arr[l]);
  free(arr);
}

LlamaLora llama_lora_create(const LlamaModel *m, int r, float alpha) {
  LlamaLora lo;
  const int dm = m->d_model;
  const int qd = m->n_heads * m->head_dim, kvd = m->n_kv * m->head_dim;
  lo.n_layers = m->n_layers;
  lo.r = r;
  lo.q = adapters(m->n_layers, dm, qd, r, alpha);   // Wq: (d_model, n_heads*head_dim)
  lo.k = adapters(m->n_layers, dm, kvd, r, alpha);  // Wk/Wv: (d_model, n_kv*head_dim)
  lo.v = adapters(m->n_layers, dm, kvd, r, alpha);
  lo.o = adapters(m->n_layers, qd, dm, r, alpha);   // Wo: (n_heads*head_dim, d_model)
  return lo;
}

void llama_lora_free(LlamaLora *lo) {
  free_adapters(lo->q, lo->n_layers);
  free_adapters(lo->k, lo->n_layers);
  free_adapters(lo->v, lo->n_layers);
  free_adapters(lo->o, lo->n_layers);
  lo->q = lo->k = lo->v = lo->o = NULL;
}

void llama_lora_init(LlamaLora *lo) {
  for (int l = 0; l < lo->n_layers; l++) {
    lora_init(&lo->q[l]); lora_init(&lo->k[l]);
    lora_init(&lo->v[l]); lora_init(&lo->o[l]);
  }
}

static LoraCache *caches(int n_layers, int seq_len, int r, int d_out) {
  LoraCache *arr = (LoraCache*)malloc((size_t)n_layers * sizeof(LoraCache));
  for (int l = 0; l < n_layers; l++) arr[l] = lora_cache_create(seq_len, r, d_out);
  return arr;
}

static void free_caches(LoraCache *arr, int n_layers) {
  for (int l = 0; l < n_layers; l++) lora_cache_free(&arr[l]);
  free(arr);
}

LlamaLoraActs llama_lora_acts_create(const LlamaModel *m, const LlamaLora *lo, int seq_len) {
  LlamaLoraActs la;
  const int dm = m->d_model;
  const int qd = m->n_heads * m->head_dim, kvd = m->n_kv * m->head_dim;
  la.n_layers = m->n_layers;
  la.q = caches(m->n_layers, seq_len, lo->r, qd);
  la.k = caches(m->n_layers, seq_len, lo->r, kvd);
  la.v = caches(m->n_layers, seq_len, lo->r, kvd);
  la.o = caches(m->n_layers, seq_len, lo->r, dm);
  return la;
}

void llama_lora_acts_free(LlamaLoraActs *la) {
  free_caches(la->q, la->n_layers);
  free_caches(la->k, la->n_layers);
  free_caches(la->v, la->n_layers);
  free_caches(la->o, la->n_layers);
  la->q = la->k = la->v = la->o = NULL;
}

// W += scale * (A @ B), then zero B so re-merging is a no-op.
static void merge_one(Tensor *W, LoraAdapter *l) {
  int s[2] = {l->d_in, l->d_out};
  Tensor delta = tensor_create(s, 2);
  matmul(&l->A, &l->B, &delta);
  for (size_t i = 0; i < W->size; i++) W->data[i] += l->scale * delta.data[i];
  for (size_t i = 0; i < l->B.size; i++) l->B.data[i] = 0.0f;
  tensor_free(&delta);
}

void llama_lora_merge(LlamaModel *m, LlamaLora *lo) {
  for (int l = 0; l < lo->n_layers; l++) {
    merge_one(&m->blocks[l].Wq, &lo->q[l]);
    merge_one(&m->blocks[l].Wk, &lo->k[l]);
    merge_one(&m->blocks[l].Wv, &lo->v[l]);
    merge_one(&m->blocks[l].Wo, &lo->o[l]);
  }
}

void llama_lora_zero_grad(const LlamaLora *lo) {
  for (int l = 0; l < lo->n_layers; l++) {
    tensor_zero_grad(&lo->q[l].A); tensor_zero_grad(&lo->q[l].B);
    tensor_zero_grad(&lo->k[l].A); tensor_zero_grad(&lo->k[l].B);
    tensor_zero_grad(&lo->v[l].A); tensor_zero_grad(&lo->v[l].B);
    tensor_zero_grad(&lo->o[l].A); tensor_zero_grad(&lo->o[l].B);
  }
}

int llama_lora_param_count(const LlamaLora *lo) {
  return 8 * lo->n_layers;   // A and B for each of q, k, v, o
}

int llama_lora_collect_params(LlamaLora *lo, Tensor **list) {
  int n = 0;
  for (int l = 0; l < lo->n_layers; l++) {
    list[n++] = &lo->q[l].A; list[n++] = &lo->q[l].B;
    list[n++] = &lo->k[l].A; list[n++] = &lo->k[l].B;
    list[n++] = &lo->v[l].A; list[n++] = &lo->v[l].B;
    list[n++] = &lo->o[l].A; list[n++] = &lo->o[l].B;
  }
  return n;
}
