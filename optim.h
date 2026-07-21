// optim.h - Adam optimizer
#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"

// Adam optimizer state. Holds first/second moment estimates for every scalar in
// the parameter set, flattened into one contiguous buffer per moment.
typedef struct {
  float *m;        // first moment (mean of grads)
  float *v;        // second moment (mean of squared grads)
  int total;       // total number of scalar parameters
  int t;           // time step (for bias correction)
  float lr;        // learning rate
  float beta1;     // first-moment decay
  float beta2;     // second-moment decay
  float eps;       // numerical epsilon
} Adam;

// Create optimizer state sized for the given parameter tensors (uses each
// tensor's `size`). Default hyperparameters: beta1=0.9, beta2=0.999, eps=1e-8.
Adam adam_create(Tensor **params, int n_params, float lr);

// One Adam update step using each parameter tensor's accumulated `grad`.
void adam_step(Adam *o, Tensor **params, int n_params);

// Release optimizer state.
void adam_free(Adam *o);

#endif // OPTIM_H
