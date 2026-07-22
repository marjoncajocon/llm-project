// check.h - Debug-only validation for tensor shapes and index bounds.
//
// These catch programming errors (shape mismatches, out-of-range token/target
// ids) with a clear message and an abort AT THE POINT OF FAILURE, instead of the
// silent heap corruption they would otherwise cause — e.g. the scatter-add in
// embedding_backward writing to wte->grad[bad_token * d_model].
//
// Cost: O(1) per matmul/bias call, O(seq_len) in embedding — negligible next to
// the O(M*N*K) work they guard. They compile to NOTHING when NDEBUG is defined
// (add -DNDEBUG to CFLAGS for a checks-off "release" build); the default build
// leaves them ON so bugs surface immediately.
#ifndef CHECK_H
#define CHECK_H

#include "tensor.h"

#ifdef NDEBUG

// Release: every check is a no-op (arguments consumed so -Wall stays clean).
#define TF_ASSERT(cond) ((void)0)
static inline void tf_check_matmul(const Tensor *a, const Tensor *b, const Tensor *c) { (void)a; (void)b; (void)c; }
static inline void tf_check_same_size(const Tensor *a, const Tensor *b) { (void)a; (void)b; }
static inline void tf_check_bias_add(const Tensor *x, const Tensor *bias, const Tensor *out) { (void)x; (void)bias; (void)out; }
static inline void tf_check_token_id(int id, int vocab_size) { (void)id; (void)vocab_size; }
static inline void tf_check_target_id(int id, int vocab_size) { (void)id; (void)vocab_size; }

#else

#include <stdio.h>
#include <stdlib.h>

// One-off boolean check (for conditions without a dedicated helper below).
#define TF_ASSERT(cond)                                                    \
  do {                                                                     \
    if (!(cond)) {                                                         \
      fprintf(stderr, "TF_ASSERT failed: %s\n  at %s:%d\n",                \
              #cond, __FILE__, __LINE__);                                  \
      abort();                                                             \
    }                                                                      \
  } while (0)

// C = A(M,K) @ B(K,N): all 2D, inner dims agree, C sized (M,N).
static inline void tf_check_matmul(const Tensor *a, const Tensor *b, const Tensor *c) {
  if (a->ndim != 2 || b->ndim != 2 || c->ndim != 2 ||
      a->shape[1] != b->shape[0] ||
      c->shape[0] != a->shape[0] || c->shape[1] != b->shape[1]) {
    fprintf(stderr, "tf_check_matmul: bad shapes A(%d,%d) @ B(%d,%d) -> C(%d,%d)\n",
            a->shape[0], a->ndim > 1 ? a->shape[1] : -1,
            b->shape[0], b->ndim > 1 ? b->shape[1] : -1,
            c->shape[0], c->ndim > 1 ? c->shape[1] : -1);
    abort();
  }
}

// Element-wise ops: A and B must hold the same number of elements.
static inline void tf_check_same_size(const Tensor *a, const Tensor *b) {
  if (a->size != b->size) {
    fprintf(stderr, "tf_check_same_size: %zu != %zu\n", a->size, b->size);
    abort();
  }
}

// out[i,j] = x[i,j] + bias[j]: x/out same 2D shape, bias length == columns.
static inline void tf_check_bias_add(const Tensor *x, const Tensor *bias, const Tensor *out) {
  if (x->ndim != 2 || out->ndim != 2 ||
      out->shape[0] != x->shape[0] || out->shape[1] != x->shape[1] ||
      bias->size != (size_t)x->shape[1]) {
    fprintf(stderr, "tf_check_bias_add: x(%d,%d) bias(%zu) / out(%d,%d) mismatch\n",
            x->shape[0], x->ndim > 1 ? x->shape[1] : -1, bias->size,
            out->shape[0], out->ndim > 1 ? out->shape[1] : -1);
    abort();
  }
}

// Input token id must index a real embedding row: 0 <= id < vocab_size.
static inline void tf_check_token_id(int id, int vocab_size) {
  if (id < 0 || id >= vocab_size) {
    fprintf(stderr, "tf_check_token_id: token id %d out of range [0,%d)\n", id, vocab_size);
    abort();
  }
}

// Target id may be negative (ignored position) but must never index past the
// vocab — an id >= vocab_size silently reads OOB in cross-entropy.
static inline void tf_check_target_id(int id, int vocab_size) {
  if (id >= vocab_size) {
    fprintf(stderr, "tf_check_target_id: target id %d >= vocab %d\n", id, vocab_size);
    abort();
  }
}

#endif // NDEBUG

#endif // CHECK_H
