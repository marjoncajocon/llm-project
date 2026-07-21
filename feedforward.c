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

// Training-mode forward: caches h1b (pre-activation) and act (= gelu(h1b)).
void feedforward_train(
    const Tensor *input,
    const Tensor *W1, const Tensor *b1,
    const Tensor *W2, const Tensor *b2,
    Tensor *h1b, Tensor *act,
    Tensor *output,
    int d_ff) {
  (void)d_ff;
  matmul(input, W1, h1b);        // h1b = input @ W1
  bias_add(h1b, b1, h1b);        // h1b += b1 (in place)
  gelu(h1b, act);               // act = gelu(h1b)
  matmul(act, W2, output);       // output = act @ W2
  bias_add(output, b2, output);  // output += b2 (in place)
}

// Backward. Reads output->grad; accumulates input->grad and W1/b1/W2/b2 grads.
void feedforward_backward(
    const Tensor *input,
    const Tensor *W1, const Tensor *b1,
    const Tensor *W2, const Tensor *b2,
    Tensor *h1b, Tensor *act,
    Tensor *output,
    int d_ff) {
  (void)d_ff;
  // output = (act @ W2) + b2  ->  d(act@W2) == output->grad
  bias_add_backward(output, b2);
  matmul_backward(act, W2, output);      // act->grad, W2->grad
  // act = gelu(h1b)
  gelu_backward(h1b, act);               // h1b->grad
  // h1b = (input @ W1) + b1  ->  d(input@W1) == h1b->grad
  bias_add_backward(h1b, b1);
  matmul_backward(input, W1, h1b);       // input->grad, W1->grad
}
