// selftest.h - Built-in smoke test (the default `transformer` run).
// Exercises every module end-to-end: tokenizer round-trips, full-model gradient
// check, training loss drop, bit-exact KV cache, multimodal train/generate, and
// media generation/captioning. Prints "Output validation: PASSED" on success.
#ifndef SELFTEST_H
#define SELFTEST_H

// Run the full self-test. Returns 0 on success (all checks PASSED), 1 otherwise.
int run_selftest(void);

#endif // SELFTEST_H
