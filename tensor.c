// tensor.c - Tensor structure and memory management
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdalign.h>

// Allocate aligned memory for cache-line optimization (64-byte alignment)
static inline void* aligned_malloc(size_t size) {
  void *ptr = aligned_alloc(64, size);
  if (!ptr) {
    fprintf(stderr, "Error: aligned_alloc failed\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

// Create a tensor with given shape
Tensor tensor_create(int *shape, int ndim) {
  Tensor t;
  t.ndim = ndim;
  t.shape = (int*)malloc(ndim * sizeof(int));
  memcpy(t.shape, shape, ndim * sizeof(int));

  // Calculate total size
  t.size = 1;
  for (int i = 0; i < ndim; i++) {
    t.size *= shape[i];
  }

  // Allocate aligned memory for data and gradients
  t.data = (float*)aligned_malloc(t.size * sizeof(float));
  t.grad = (float*)aligned_malloc(t.size * sizeof(float));

  // Initialize to zeros
  memset(t.data, 0, t.size * sizeof(float));
  memset(t.grad, 0, t.size * sizeof(float));

  return t;
}

// Free tensor memory
void tensor_free(Tensor *t) {
  free(t->data);
  free(t->grad);
  free(t->shape);
  t->data = NULL;
  t->grad = NULL;
  t->shape = NULL;
}

// Fill tensor with random values (Xavier initialization)
void tensor_init_random(Tensor *t) {
  float scale = sqrtf(2.0f / t->shape[t->ndim - 1]);
  for (int i = 0; i < t->size; i++) {
    t->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
  }
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
