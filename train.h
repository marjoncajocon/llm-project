// train.h - Train a GPT on a text file
#ifndef TRAIN_H
#define TRAIN_H

// Train on the text in data_path:
//   1. learn a byte-level BPE tokenizer (num_merges merges) on the text,
//   2. build a GPT sized to that vocabulary,
//   3. train for `steps` Adam steps on random context windows,
//   4. save the model to out_model and the tokenizer to out_tok.
// Returns 0 on success, nonzero on error.
int train_run(const char *data_path, const char *out_model, const char *out_tok,
              int num_merges, int steps);

#endif // TRAIN_H
