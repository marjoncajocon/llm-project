// main.c - Minimal GPT-2 Style Transformer in Pure C11 (demo entry point)
// Build: make        (see README.md and Makefile)
// Or:    gcc -std=c11 -O3 -march=native -o transformer *.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tensor.h"
#include "matrix.h"
#include "activation.h"
#include "layernorm.h"
#include "loss.h"
#include "optim.h"
#include "model.h"
#include "checkpoint.h"
#include "tokenizer.h"

#define LN_EPS 1e-5f

// ============================================================================
// A tiny trainable MLP that exercises every backward op we implemented:
//   logits = LayerNorm(GELU(x @ W1 + b1)) @ W2 + b2 ;  loss = cross-entropy
// ============================================================================
typedef struct {
  int N, in_dim, hidden, n_classes;
  Tensor W1, b1, gamma, beta, W2, b2;                 // parameters
  Tensor h1, h1b, g, ln, z, logits, probs;            // intermediates
} MLP;

static MLP mlp_create(int N, int in_dim, int hidden, int n_classes) {
  MLP m;
  m.N = N; m.in_dim = in_dim; m.hidden = hidden; m.n_classes = n_classes;

  int s_w1[2] = {in_dim, hidden};   m.W1 = tensor_create(s_w1, 2);
  int s_b1[1] = {hidden};           m.b1 = tensor_create(s_b1, 1);
  int s_g[1]  = {hidden};           m.gamma = tensor_create(s_g, 1);
  int s_be[1] = {hidden};           m.beta = tensor_create(s_be, 1);
  int s_w2[2] = {hidden, n_classes};m.W2 = tensor_create(s_w2, 2);
  int s_b2[1] = {n_classes};        m.b2 = tensor_create(s_b2, 1);

  tensor_init_random(&m.W1);
  tensor_init_random(&m.b1);
  tensor_init_random(&m.W2);
  tensor_init_random(&m.b2);
  for (int i = 0; i < hidden; i++) { m.gamma.data[i] = 1.0f; m.beta.data[i] = 0.0f; }

  int s_h[2]  = {N, hidden};        m.h1 = tensor_create(s_h, 2);
  int s_hb[2] = {N, hidden};        m.h1b = tensor_create(s_hb, 2);
  int s_gg[2] = {N, hidden};        m.g = tensor_create(s_gg, 2);
  int s_ln[2] = {N, hidden};        m.ln = tensor_create(s_ln, 2);
  int s_z[2]  = {N, n_classes};     m.z = tensor_create(s_z, 2);
  int s_lo[2] = {N, n_classes};     m.logits = tensor_create(s_lo, 2);
  int s_pr[2] = {N, n_classes};     m.probs = tensor_create(s_pr, 2);
  return m;
}

static void mlp_free(MLP *m) {
  tensor_free(&m->W1); tensor_free(&m->b1); tensor_free(&m->gamma);
  tensor_free(&m->beta); tensor_free(&m->W2); tensor_free(&m->b2);
  tensor_free(&m->h1); tensor_free(&m->h1b); tensor_free(&m->g);
  tensor_free(&m->ln); tensor_free(&m->z); tensor_free(&m->logits);
  tensor_free(&m->probs);
}

// Forward pass, returns the mean cross-entropy loss.
static float mlp_forward(MLP *m, const Tensor *x, const int *targets) {
  matmul(x, &m->W1, &m->h1);
  bias_add(&m->h1, &m->b1, &m->h1b);
  gelu(&m->h1b, &m->g);
  layernorm(&m->g, &m->gamma, &m->beta, &m->ln, LN_EPS);
  matmul(&m->ln, &m->W2, &m->z);
  bias_add(&m->z, &m->b2, &m->logits);
  return cross_entropy_forward(&m->logits, targets, &m->probs);
}

static void copy_grad(Tensor *dst, const Tensor *src) {
  memcpy(dst->grad, src->grad, dst->size * sizeof(float));
}

// Backward pass: zero all grads, then propagate from the loss to every parameter.
static void mlp_backward(MLP *m, const Tensor *x, const int *targets) {
  tensor_zero_grad(&m->W1); tensor_zero_grad(&m->b1);
  tensor_zero_grad(&m->gamma); tensor_zero_grad(&m->beta);
  tensor_zero_grad(&m->W2); tensor_zero_grad(&m->b2);
  tensor_zero_grad(&m->h1); tensor_zero_grad(&m->h1b); tensor_zero_grad(&m->g);
  tensor_zero_grad(&m->ln); tensor_zero_grad(&m->z); tensor_zero_grad(&m->logits);
  tensor_zero_grad((Tensor*)x);

  cross_entropy_backward(&m->probs, targets, &m->logits);
  bias_add_backward(&m->logits, &m->b2);
  copy_grad(&m->z, &m->logits);
  matmul_backward(&m->ln, &m->W2, &m->z);
  layernorm_backward(&m->g, &m->gamma, &m->ln, &m->gamma, &m->beta, LN_EPS);
  gelu_backward(&m->h1b, &m->g);
  bias_add_backward(&m->h1b, &m->b1);
  copy_grad(&m->h1, &m->h1b);
  matmul_backward(x, &m->W1, &m->h1);
}

// ============================================================================
// Main
// ============================================================================
int main() {
  printf("=== Minimal GPT-2 Style Transformer Demo ===\n\n");
  int all_ok = 1;
  srand(42);

  // --------------------------------------------------------------------------
  // Part 1: tokenizer + checkpoint round-trips (step 4 code, condensed)
  // --------------------------------------------------------------------------
  printf("[1] Tokenizer + checkpoint round-trips\n");
  Tokenizer tok = tokenizer_create_bytelevel();
  int ll = tokenizer_add_merge(&tok, 'l', 'l');
  tokenizer_add_merge(&tok, 'e', ll);
  int ids[64];
  int n_ids = tokenizer_encode(&tok, "hello hello", ids, 64);
  char decoded[128];
  tokenizer_decode(&tok, ids, n_ids, decoded, sizeof(decoded));
  int tok_ok = (strcmp("hello hello", decoded) == 0);
  tokenizer_free(&tok);
  all_ok &= tok_ok;

  GPT gpt = gpt_create(16, 8, 2, 2, 32, 16);
  int seq[3] = {1, 5, 9};
  int lshape[2] = {3, 16};
  Tensor la = tensor_create(lshape, 2), lb = tensor_create(lshape, 2);
  gpt_forward(&gpt, seq, 3, &la);
  int ckpt_ok = (gpt_save("model.bin", &gpt) == 0);
  GPT gpt2;
  ckpt_ok = ckpt_ok && (gpt_load("model.bin", &gpt2) == 0);
  if (ckpt_ok) {
    gpt_forward(&gpt2, seq, 3, &lb);
    ckpt_ok = (memcmp(la.data, lb.data, la.size * sizeof(float)) == 0);
    gpt_free(&gpt2);
  }
  tensor_free(&la); tensor_free(&lb); gpt_free(&gpt); remove("model.bin");
  printf("    tokenizer: %s | checkpoint: %s\n\n",
         tok_ok ? "OK" : "FAIL", ckpt_ok ? "OK" : "FAIL");
  all_ok &= ckpt_ok;

  // --------------------------------------------------------------------------
  // Part 2: build the trainable MLP + a fixed synthetic dataset
  // --------------------------------------------------------------------------
  const int N = 8, in_dim = 6, hidden = 16, n_classes = 4;
  MLP m = mlp_create(N, in_dim, hidden, n_classes);

  int xs[2] = {N, in_dim};
  Tensor x = tensor_create(xs, 2);
  tensor_init_random(&x);
  int targets[8];
  for (int i = 0; i < N; i++) targets[i] = i % n_classes;

  Tensor *params[6] = {&m.W1, &m.b1, &m.gamma, &m.beta, &m.W2, &m.b2};
  const int n_params = 6;

  // --------------------------------------------------------------------------
  // Part 3: gradient check (analytic grads vs numerical finite differences)
  // --------------------------------------------------------------------------
  printf("[2] Gradient check (analytic vs numerical)\n");
  mlp_forward(&m, &x, targets);
  mlp_backward(&m, &x, targets);  // fills param->grad at the base point

  const float eps = 1e-3f;
  float max_rel = 0.0f;
  for (int p = 0; p < n_params; p++) {
    Tensor *param = params[p];
    // sample up to 3 indices per parameter tensor
    int step = param->size > 3 ? param->size / 3 : 1;
    for (int k = 0; k < param->size; k += step) {
      float base = param->data[k];
      param->data[k] = base + eps; float lp = mlp_forward(&m, &x, targets);
      param->data[k] = base - eps; float lm = mlp_forward(&m, &x, targets);
      param->data[k] = base;
      float numeric = (lp - lm) / (2.0f * eps);
      float analytic = param->grad[k];
      float rel = fabsf(analytic - numeric) / (fabsf(analytic) + fabsf(numeric) + 1e-8f);
      if (rel > max_rel) max_rel = rel;
    }
  }
  int grad_ok = (max_rel < 1e-2f);
  printf("    max relative error: %.2e  (tol 1e-2) -> %s\n\n",
         max_rel, grad_ok ? "OK" : "FAIL");
  all_ok &= grad_ok;

  // --------------------------------------------------------------------------
  // Part 4: train (Adam) and check the loss actually drops
  // --------------------------------------------------------------------------
  printf("[3] Training (Adam, overfit %d examples)\n", N);
  Adam opt = adam_create(params, n_params, 0.05f);
  float first_loss = mlp_forward(&m, &x, targets);
  const int steps = 300;
  for (int s = 0; s < steps; s++) {
    float loss = mlp_forward(&m, &x, targets);
    mlp_backward(&m, &x, targets);
    adam_step(&opt, params, n_params);
    if (s % 60 == 0) printf("    step %3d  loss %.4f\n", s, loss);
  }
  float final_loss = mlp_forward(&m, &x, targets);
  int train_ok = (final_loss < first_loss * 0.1f) && (final_loss < 0.05f);
  printf("    start %.4f -> final %.4f -> %s\n\n",
         first_loss, final_loss, train_ok ? "OK (loss dropped)" : "FAIL");
  all_ok &= train_ok;

  adam_free(&opt);
  mlp_free(&m);
  tensor_free(&x);

  // --------------------------------------------------------------------------
  printf("Output validation: %s\n\n", all_ok ? "PASSED" : "FAILED");
  printf("Done! All memory freed successfully.\n");
  return all_ok ? 0 : 1;
}
