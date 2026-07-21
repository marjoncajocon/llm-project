// tensor.h - Tensor structure and memory management
#ifndef TENSOR_H
#define TENSOR_H

// ============================================================================
// Tensor Structure Definition
// ============================================================================
typedef struct {
  float *data;      // Data array (row-major layout)
  float *grad;      // Gradient array (for future backprop)
  int *shape;       // Array of dimension sizes
  int ndim;         // Number of dimensions
  int size;         // Total number of elements
} Tensor;

// ============================================================================
// Memory Management Functions
// ============================================================================

// Create a tensor with given shape
Tensor tensor_create(int *shape, int ndim);

// Free tensor memory
void tensor_free(Tensor *t);

// Fill tensor with random values (Xavier initialization)
void tensor_init_random(Tensor *t);

// Print tensor (for debugging)
void tensor_print(const Tensor *t, const char *name);

#endif // TENSOR_H
