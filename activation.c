// activation.c - Activation functions
#include "activation.h"

#include <math.h>

#include "matrix.h"   // tf_parallel_for

// Row-wise Softmax with numerical stability
// Applies softmax to each row of input matrix:
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
// Rows are independent, so large inputs (the training logits: rows x vocab) are
// split across the thread pool by rows; each row runs the identical serial
// max / exp-sum / divide sequence, so the result does not depend on the thread
// count (selftest [43]).
typedef struct { const Tensor *in; Tensor *out; } SoftmaxJob;

static void softmax_rows(int r0, int r1, void *ctx) {
  const SoftmaxJob *t = (const SoftmaxJob*)ctx;
  const Tensor *input = t->in;
  Tensor *output = t->out;
  const int cols = input->shape[1];

  for (int i = r0; i < r1; i++) {
    const float *x = &input->data[(size_t)i * cols];
    float *y = &output->data[(size_t)i * cols];
    // Step 1: Find maximum value in row for numerical stability
    float max_val = x[0];
    for (int j = 1; j < cols; j++) {
      if (x[j] > max_val) {
        max_val = x[j];
      }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (int j = 0; j < cols; j++) {
      y[j] = expf(x[j] - max_val);
      sum_exp += y[j];
    }

    // Step 3: Normalize by sum
    for (int j = 0; j < cols; j++) {
      y[j] /= sum_exp;
    }
  }
}

#define SOFTMAX_THREAD_MIN (1L << 15)   // elements; below this, one thread

void softmax_rowwise(const Tensor *input, Tensor *output) {
  const int rows = input->shape[0];
  SoftmaxJob t = { input, output };
  if (rows < 2 || (size_t)rows * (size_t)input->shape[1] < (size_t)SOFTMAX_THREAD_MIN)
    softmax_rows(0, rows, &t);
  else
    tf_parallel_for(rows, softmax_rows, &t);
}

// GELU activation (tanh approximation, as used by GPT-2), element-wise
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu(const Tensor *input, Tensor *output) {
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  for (size_t i = 0; i < input->size; i++) {
    float x = input->data[i];
    float inner = k * (x + 0.044715f * x * x * x);
    output->data[i] = 0.5f * x * (1.0f + tanhf(inner));
  }
}

// Backward of GELU: input->grad += output->grad * d gelu/dx
void gelu_backward(const Tensor *input, const Tensor *output) {
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  const float a = 0.044715f;
  for (size_t i = 0; i < input->size; i++) {
    float x = input->data[i];
    float inner = k * (x + a * x * x * x);
    float t = tanhf(inner);
    float d_inner = k * (1.0f + 3.0f * a * x * x);
    // d/dx [0.5 x (1 + tanh(inner))]
    float dgelu = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * d_inner;
    input->grad[i] += output->grad[i] * dgelu;
  }
}
