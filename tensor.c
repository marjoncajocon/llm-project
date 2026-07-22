// tensor.c - Tensor structure and memory management
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdalign.h>

// Portable 64-byte aligned allocation. C11 aligned_alloc() is not available in
// the Windows CRT (which uses _aligned_malloc, and whose blocks must be released
// with _aligned_free, never plain free()), so pick the right pair per platform.
#if defined(_WIN32)
#include <malloc.h>
#define ALIGNED_ALLOC(align, size) _aligned_malloc((size), (align))
#define ALIGNED_FREE(ptr)          _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(align, size) aligned_alloc((align), (size))
#define ALIGNED_FREE(ptr)          free(ptr)
#endif

// Allocate aligned memory for cache-line optimization (64-byte alignment)
static inline float* aligned_malloc(size_t size) {
  float *ptr = (float*)ALIGNED_ALLOC(64, size);
  if (!ptr) {
    fprintf(stderr, "Error: aligned allocation failed\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

// Shared allocator: with_grad=0 leaves grad == NULL (inference-only tensors).
static Tensor tensor_alloc(int *shape, int ndim, int with_grad) {
  Tensor t;
  t.ndim = ndim;
  t.shape = (int*)malloc(ndim * sizeof(int));
  if (!t.shape) {
    fprintf(stderr, "Error: shape allocation failed\n");
    exit(EXIT_FAILURE);
  }
  memcpy(t.shape, shape, ndim * sizeof(int));

  // Calculate total size (accumulate in size_t so the product never wraps).
  t.size = 1;
  for (int i = 0; i < ndim; i++) {
    t.size *= (size_t)shape[i];
  }

  // Allocate aligned memory for data (and gradients if requested), zero-filled.
  t.data = (float*)aligned_malloc(t.size * sizeof(float));
  memset(t.data, 0, t.size * sizeof(float));
  if (with_grad) {
    t.grad = (float*)aligned_malloc(t.size * sizeof(float));
    memset(t.grad, 0, t.size * sizeof(float));
  } else {
    t.grad = NULL;
  }

  return t;
}

// Create a tensor with given shape (with a gradient array, for training).
Tensor tensor_create(int *shape, int ndim) {
  return tensor_alloc(shape, ndim, 1);
}

// Create a tensor WITHOUT a gradient array (grad == NULL). For inference-only
// tensors — weights loaded only for generation, KV caches, decode scratch —
// this halves the memory of a forward-only model. Never pass one to a *_backward
// that writes into grad.
Tensor tensor_create_nograd(int *shape, int ndim) {
  return tensor_alloc(shape, ndim, 0);
}

// Free tensor memory
void tensor_free(Tensor *t) {
  ALIGNED_FREE(t->data);      // data/grad came from the aligned allocator
  if (t->grad) ALIGNED_FREE(t->grad);   // grad is NULL for inference-only tensors
  free(t->shape);             // shape came from plain malloc
  t->data = NULL;
  t->grad = NULL;
  t->shape = NULL;
}

// Fill tensor with random values (Xavier initialization)
void tensor_init_random(Tensor *t) {
  float scale = sqrtf(2.0f / t->shape[t->ndim - 1]);
  for (size_t i = 0; i < t->size; i++) {
    // Divide in double so RAND_MAX (which may exceed float's exact range on
    // some platforms) doesn't lose precision. Result r is in [0, 1].
    float r = (float)((double)rand() / (double)RAND_MAX);
    t->data[i] = (r * 2.0f - 1.0f) * scale;
  }
}

// Zero the gradient array (call before accumulating grads for a new step).
// No-op for inference-only tensors (grad == NULL).
void tensor_zero_grad(Tensor *t) {
  if (t->grad) memset(t->grad, 0, t->size * sizeof(float));
}

// Print tensor (for debugging)
void tensor_print(const Tensor *t, const char *name) {
  printf("%s shape: (", name);
  for (int i = 0; i < t->ndim; i++) {
    printf("%d%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
  }
  printf(")\n");

  if (t->ndim == 2) {
    // Print 2D matrix
    for (int i = 0; i < t->shape[0] && i < 8; i++) {  // Limit to 8 rows
      printf("[");
      for (int j = 0; j < t->shape[1] && j < 8; j++) {  // Limit to 8 cols
        printf("%7.4f", t->data[i * t->shape[1] + j]);
        if (j < t->shape[1] - 1 && j < 7) printf(", ");
      }
      if (t->shape[1] > 8) printf(", ...");
      printf("]\n");
    }
    if (t->shape[0] > 8) printf("... (%d more rows)\n", t->shape[0] - 8);
  }
  printf("\n");
}
