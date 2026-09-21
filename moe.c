// moe.c - Mixture-of-Experts feed-forward layer (see moe.h)
//
// Dispatch, not masking. The simple way to write an MoE is to run every expert
// on every token and multiply the unchosen outputs by a zero gate. That gives
// the right numbers and throws away the entire reason MoE exists: compute would
// scale with n_experts. Here each expert gathers only the rows routed to it,
// runs the ordinary SwiGLU forward/backward from swiglu.c on that packed block,
// and the results are scattered back. An expert that received no tokens is
// never touched -- the self-test proves it by filling such experts with NaN.
#include "moe.h"

#include <stdlib.h>
#include <string.h>

#include "activation.h"
#include "check.h"
#include "matrix.h"

static Tensor mat(int r, int c) {
  int sh[2] = {r, c};
  return tensor_create(sh, 2);
}

MoeLayer moe_create(int d_model, int d_ff, int n_experts, int top_k,
                    int renorm, int shared) {
  TF_ASSERT(n_experts >= 1 && top_k >= 1 && top_k <= n_experts);
  MoeLayer m;
  memset(&m, 0, sizeof m);
  m.d_model = d_model; m.d_ff = d_ff; m.n_experts = n_experts; m.top_k = top_k;
  m.renorm = renorm; m.shared = shared;
  m.W_router = mat(d_model, n_experts);
  tensor_init_random(&m.W_router);
  m.W_gate = (Tensor*)malloc(n_experts * sizeof(Tensor));
  m.W_up   = (Tensor*)malloc(n_experts * sizeof(Tensor));
  m.W_down = (Tensor*)malloc(n_experts * sizeof(Tensor));
  for (int e = 0; e < n_experts; e++) {
    m.W_gate[e] = mat(d_model, d_ff); tensor_init_random(&m.W_gate[e]);
    m.W_up[e]   = mat(d_model, d_ff); tensor_init_random(&m.W_up[e]);
    m.W_down[e] = mat(d_ff, d_model); tensor_init_random(&m.W_down[e]);
  }
  if (shared) {
    m.Ws_gate = mat(d_model, d_ff); tensor_init_random(&m.Ws_gate);
    m.Ws_up   = mat(d_model, d_ff); tensor_init_random(&m.Ws_up);
    m.Ws_down = mat(d_ff, d_model); tensor_init_random(&m.Ws_down);
  }
  return m;
}

void moe_free(MoeLayer *m) {
  tensor_free(&m->W_router);
  for (int e = 0; e < m->n_experts; e++) {
    tensor_free(&m->W_gate[e]); tensor_free(&m->W_up[e]); tensor_free(&m->W_down[e]);
  }
  free(m->W_gate); free(m->W_up); free(m->W_down);
  if (m->shared) {
    tensor_free(&m->Ws_gate); tensor_free(&m->Ws_up); tensor_free(&m->Ws_down);
  }
  m->W_gate = m->W_up = m->W_down = NULL;
}

int moe_n_params(const MoeLayer *m) {
  return 1 + 3 * m->n_experts + (m->shared ? 3 : 0);
}

int moe_params(MoeLayer *m, Tensor **out) {
  int k = 0;
  out[k++] = &m->W_router;
  for (int e = 0; e < m->n_experts; e++) {
    out[k++] = &m->W_gate[e]; out[k++] = &m->W_up[e]; out[k++] = &m->W_down[e];
  }
  if (m->shared) { out[k++] = &m->Ws_gate; out[k++] = &m->Ws_up; out[k++] = &m->Ws_down; }
  return k;
}

MoeCache moe_cache_create(const MoeLayer *m, int n) {
  const int E = m->n_experts, k = m->top_k;
  MoeCache c;
  memset(&c, 0, sizeof c);
  c.n = n; c.n_experts = E; c.top_k = k;
  c.logits = mat(n, E);
  c.probs  = mat(n, E);
  c.sel    = (int*)calloc((size_t)n * k, sizeof(int));
  c.gate   = (float*)calloc((size_t)n * k, sizeof(float));
  c.dgate  = (float*)calloc((size_t)n * k, sizeof(float));
  c.order  = (int*)calloc((size_t)n * k, sizeof(int));
  c.count  = (int*)calloc(E, sizeof(int));
  c.offset = (int*)calloc(E + 1, sizeof(int));
  c.rows_alloc = (int*)calloc(E, sizeof(int));
  c.xe = (Tensor*)calloc(E, sizeof(Tensor));
  c.ye = (Tensor*)calloc(E, sizeof(Tensor));
  c.ce = (SwigluCache*)calloc(E, sizeof(SwigluCache));
  c.f  = (float*)calloc(E, sizeof(float));
  c.P  = (float*)calloc(E, sizeof(float));
  if (m->shared) {
    c.ys = mat(n, m->d_model);
    c.cs = swiglu_cache_create(n, m->d_ff);
  }
  return c;
}

static void free_expert_buffers(MoeCache *c, int e) {
  if (c->rows_alloc[e] == 0) return;
  tensor_free(&c->xe[e]); tensor_free(&c->ye[e]); swiglu_cache_free(&c->ce[e]);
  c->rows_alloc[e] = 0;
}

void moe_cache_free(MoeCache *c) {
  for (int e = 0; e < c->n_experts; e++) free_expert_buffers(c, e);
  tensor_free(&c->logits); tensor_free(&c->probs);
  free(c->sel); free(c->gate); free(c->dgate); free(c->order);
  free(c->count); free(c->offset); free(c->rows_alloc);
  free(c->xe); free(c->ye); free(c->ce); free(c->f); free(c->P);
  if (c->ys.data) { tensor_free(&c->ys); swiglu_cache_free(&c->cs); }
  memset(c, 0, sizeof *c);
}

float moe_forward(const MoeLayer *m, const Tensor *x, MoeCache *c, Tensor *y) {
  const int n = c->n, d = m->d_model, E = m->n_experts, k = m->top_k;
  TF_ASSERT(x->shape[0] == n && x->shape[1] == d);
  TF_ASSERT(y->shape[0] == n && y->shape[1] == d);

  // --- router ---
  matmul(x, &m->W_router, &c->logits);
  softmax_rowwise(&c->logits, &c->probs);
  c->flops_router = 2.0 * n * d * E;

  // --- top-k per token: k passes of argmax over the not-yet-chosen experts.
  // k is small (1-8 in practice), so O(k*E) beats a sort. Strict '>' makes
  // ties go to the lower index, which keeps routing deterministic.
  for (int t = 0; t < n; t++) {
    const float *p = c->probs.data + (size_t)t * E;
    int *s = c->sel + (size_t)t * k;
    float sum = 0.0f;
    for (int j = 0; j < k; j++) {
      int best = -1;
      for (int e = 0; e < E; e++) {
        int taken = 0;
        for (int q = 0; q < j; q++) if (s[q] == e) { taken = 1; break; }
        if (!taken && (best < 0 || p[e] > p[best])) best = e;
      }
      s[j] = best;
      sum += p[best];
    }
    for (int j = 0; j < k; j++)
      c->gate[(size_t)t * k + j] = m->renorm ? p[s[j]] / sum : p[s[j]];
  }

  // --- group the n*k assignments by expert (a counting sort) ---
  memset(c->count, 0, E * sizeof(int));
  for (int a = 0; a < n * k; a++) c->count[c->sel[a]]++;
  c->offset[0] = 0;
  for (int e = 0; e < E; e++) c->offset[e + 1] = c->offset[e] + c->count[e];
  {
    int *fill = (int*)malloc(E * sizeof(int));
    memcpy(fill, c->offset, E * sizeof(int));
    for (int a = 0; a < n * k; a++) c->order[fill[c->sel[a]]++] = a;
    free(fill);
  }

  // --- run each expert on exactly its own tokens, scatter gate * output ---
  memset(y->data, 0, (size_t)n * d * sizeof(float));
  c->flops_experts = 0.0;
  c->expert_rows = 0;
  for (int e = 0; e < E; e++) {
    const int cnt = c->count[e];
    if (c->rows_alloc[e] != cnt) {
      free_expert_buffers(c, e);
      if (cnt > 0) {
        c->xe[e] = mat(cnt, d);
        c->ye[e] = mat(cnt, d);
        c->ce[e] = swiglu_cache_create(cnt, m->d_ff);
        c->rows_alloc[e] = cnt;
      }
    }
    if (cnt == 0) continue;   // never computed: the compute saving of MoE

    for (int r = 0; r < cnt; r++) {
      int t = c->order[c->offset[e] + r] / k;
      memcpy(c->xe[e].data + (size_t)r * d, x->data + (size_t)t * d, d * sizeof(float));
    }
    swiglu_forward(&c->xe[e], &m->W_gate[e], &m->W_up[e], &m->W_down[e],
                   &c->ce[e], &c->ye[e]);
    c->flops_experts += 6.0 * cnt * d * m->d_ff;   // three (cnt x d x d_ff) matmuls
    c->expert_rows += cnt;

    for (int r = 0; r < cnt; r++) {
      int a = c->order[c->offset[e] + r], t = a / k;
      float g = c->gate[a];
      const float *src = c->ye[e].data + (size_t)r * d;
      float *dst = y->data + (size_t)t * d;
      for (int i = 0; i < d; i++) dst[i] += g * src[i];
    }
  }

  // --- shared expert: every token, gate fixed at 1 (DeepSeek-style). It holds
  // the knowledge every token needs, so the routed experts can specialize.
  c->flops_shared = 0.0;
  if (m->shared) {
    swiglu_forward(x, &m->Ws_gate, &m->Ws_up, &m->Ws_down, &c->cs, &c->ys);
    for (size_t i = 0; i < (size_t)n * d; i++) y->data[i] += c->ys.data[i];
    c->flops_shared = 6.0 * n * d * m->d_ff;
  }

  // --- aux load-balancing loss ---
  float aux = 0.0f;
  for (int e = 0; e < E; e++) {
    float Pe = 0.0f;
    for (int t = 0; t < n; t++) Pe += c->probs.data[(size_t)t * E + e];
    c->P[e] = Pe / n;
    c->f[e] = (float)c->count[e] / (float)(n * k);
    aux += c->f[e] * c->P[e];
  }
  c->aux = aux * E;
  return c->aux;
}

// Where the gradient goes, and where it cannot.
//
// The top-k choice S_t is a step function of the logits: nudging a logit either
// leaves S_t unchanged (derivative 0) or flips it (no derivative). So no
// gradient flows "through the selection" -- we treat S_t as a constant, exactly
// as every MoE implementation does. The router still learns, because the GATE
// VALUES g_te are smooth functions of the probabilities: if raising expert e's
// weight on token t would lower the loss, dL/dg_te < 0 pushes p_te up, and a
// large enough push eventually changes the selection itself. The aux loss is
// the same story: f_e (a count) is constant, P_e (a mean probability) carries
// the gradient.
//
// Per token t, with dg_j = dL/dg_tj = <dy_t, SwiGLU_j(x_t)>:
//   renorm=0:  dL/dp_tj = dg_j
//   renorm=1:  g_j = p_j / S, S = sum_{i in S_t} p_i
//              dL/dp_tj = (dg_j - sum_i dg_i g_i) / S      (j in S_t; 0 otherwise)
//   aux:       dL/dp_te += aux_coef * E * f_e / n           (every e)
// then the softmax backward dz = p (*) (dp - <p, dp>) and the router matmul.
void moe_backward(MoeLayer *m, const Tensor *x, MoeCache *c, const Tensor *y,
                  float aux_coef) {
  const int n = c->n, d = m->d_model, E = m->n_experts, k = m->top_k;

  // --- routed experts: dgate, then each expert's own SwiGLU backward ---
  for (int e = 0; e < E; e++) {
    const int cnt = c->count[e];
    if (cnt == 0) continue;
    for (int r = 0; r < cnt; r++) {
      int a = c->order[c->offset[e] + r], t = a / k;
      const float *dy = y->grad + (size_t)t * d;
      const float *ye = c->ye[e].data + (size_t)r * d;
      float *dye = c->ye[e].grad + (size_t)r * d;
      float g = c->gate[a], dot = 0.0f;
      for (int i = 0; i < d; i++) { dot += dy[i] * ye[i]; dye[i] = g * dy[i]; }
      c->dgate[a] = dot;
    }
    tensor_zero_grad(&c->xe[e]);
    swiglu_backward(&c->xe[e], &m->W_gate[e], &m->W_up[e], &m->W_down[e],
                    &c->ce[e], &c->ye[e]);
    for (int r = 0; r < cnt; r++) {
      int t = c->order[c->offset[e] + r] / k;
      const float *src = c->xe[e].grad + (size_t)r * d;
      float *dst = x->grad + (size_t)t * d;
      for (int i = 0; i < d; i++) dst[i] += src[i];
    }
  }

  // --- gates -> router probabilities (+ aux) -> logits -> W_router, x ---
  tensor_zero_grad(&c->probs);
  tensor_zero_grad(&c->logits);
  for (int t = 0; t < n; t++) {
    const float *p = c->probs.data + (size_t)t * E;
    float *dp = c->probs.grad + (size_t)t * E;
    const int *s = c->sel + (size_t)t * k;
    const float *g = c->gate + (size_t)t * k, *dg = c->dgate + (size_t)t * k;
    if (m->renorm) {
      float S = 0.0f, gdg = 0.0f;
      for (int j = 0; j < k; j++) { S += p[s[j]]; gdg += g[j] * dg[j]; }
      for (int j = 0; j < k; j++) dp[s[j]] += (dg[j] - gdg) / S;
    } else {
      for (int j = 0; j < k; j++) dp[s[j]] += dg[j];
    }
    if (aux_coef != 0.0f)
      for (int e = 0; e < E; e++) dp[e] += aux_coef * E * c->f[e] / n;

    float pdp = 0.0f;
    for (int e = 0; e < E; e++) pdp += p[e] * dp[e];
    float *dz = c->logits.grad + (size_t)t * E;
    for (int e = 0; e < E; e++) dz[e] = p[e] * (dp[e] - pdp);
  }
  matmul_backward(x, &m->W_router, &c->logits);

  // --- shared expert: an ordinary SwiGLU on all tokens, reading y->grad ---
  if (m->shared)
    swiglu_backward(x, &m->Ws_gate, &m->Ws_up, &m->Ws_down, &c->cs, y);
}
