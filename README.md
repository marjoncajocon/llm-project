# Minimal GPT-2 Style Transformer (Pure C11)

A tiny, dependency-free implementation of a single-head self-attention transformer
block written in portable C11. It exists as a readable reference — every piece
(tensors, matrix math, softmax, LayerNorm, attention) is small enough to read in
one sitting.

## Project layout

Inspired by the Lua source tree: **every source file lives flat in the project
root — there are no subfolders.** Each module is one concern, declared in a `.h`
and defined in the matching `.c`.

| File                         | Responsibility                                     |
| ---------------------------- | -------------------------------------------------- |
| `tensor.h` / `tensor.c`      | `Tensor` struct + memory management                |
| `matrix.h` / `matrix.c`      | `matmul`, `matadd`, `matscale`                     |
| `activation.h` / `activation.c` | `softmax_rowwise`                               |
| `layernorm.h` / `layernorm.c`   | `layernorm`                                     |
| `feedforward.h` / `feedforward.c` | Position-wise feed-forward MLP: Linear → GELU → Linear |
| `attention.h` / `attention.c`   | `multihead_attention_forward` (causal), `transformer_block_forward` |
| `embedding.h` / `embedding.c`   | Token + positional embedding lookup                |
| `sampler.h` / `sampler.c`       | Token sampling from logits (greedy argmax)         |
| `model.h` / `model.c`           | `GPT` model: embeddings, stacked blocks, output head, generation |
| `main.c`                     | Demo entry point                                   |
| `Makefile`                   | Build rules                                        |

## Build & run

```sh
make            # compiles all modules into ./transformer
./transformer   # runs the demo forward pass
make clean      # removes object files and the binary
```

Or compile directly, without make:

```sh
gcc -std=c11 -O3 -march=native -o transformer *.c -lm
```

`clang` works as a drop-in replacement for `gcc`.

## Coding rules (for contributors)

1. **Indentation is 2 spaces.** Never use tabs.
2. **Do not create subfolders.** All source files stay flat in the project root
   (Lua-style layout). A pull request that adds a directory will be rejected.
3. **One concern per module.** Declare the public API in the `.h`, define it in the
   matching `.c`, and give every header an include guard (`#ifndef MODULE_H`).
4. Keep helpers that are used by only one file `static` inside that `.c`.
5. **Never use untyped `void *` parameters** in a function's public API. Pass concrete,
   typed pointers (`Tensor *`, `float *`, …) instead — untyped pointers defeat the
   compiler's type checking and lead to slow, hard-to-predict runtime errors.
6. Code must compile cleanly under `-std=c11 -Wall` (no warnings).
7. When you add a new module, add its `.o` to `OBJS` in the `Makefile` and list its
   header dependencies there.
