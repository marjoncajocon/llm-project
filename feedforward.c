// feedforward.c - Position-wise feed-forward network (MLP)
#include "feedforward.h"

#include "matrix.h"
#include "activation.h"

// Position-wise feed-forward network: FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
// Applied independently to each row (token) of the input.
void feedforward(
    const Tensor *input,                 // (seq_len, d_model)
    const Tensor *W1, const Tensor *b1,  // (d_model, d_ff), (d_ff)
    const Tensor *W2, const Tensor *b2,  // (d_ff, d_model), (d_model)
    Tensor *output,                      // (seq_len, d_model)
    int d_ff                             // Hidden dimension (typically 4 * d_model)
) {
  const int seq_len = input->shape[0];
  const int d_model = output->shape[1];

  // Temporary tensors for the hidden layer
  int hidden_shape[2] = {seq_len, d_ff};
  Tensor hidden = tensor_create(hidden_shape, 2);
  Tensor activated = tensor_create(hidden_shape, 2);

  // Step 1: First linear layer: hidden = input @ W1 + b1 (bias broadcast over rows)
  matmul(input, W1, &hidden);
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < d_ff; j++) {
      hidden.data[i * d_ff + j] += b1->data[j];
    }
  }

  // Step 2: GELU activation
  gelu(&hidden, &activated);

  // Step 3: Second linear layer: output = activated @ W2 + b2 (bias broadcast over rows)
  matmul(&activated, W2, output);
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < d_model; j++) {
      output->data[i * d_model + j] += b2->data[j];
    }
  }

  tensor_free(&hidden);
  tensor_free(&activated);
}
