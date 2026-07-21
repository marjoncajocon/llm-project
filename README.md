# Minimal GPT-2 Style Transformer (Pure C11)

A tiny, dependency-free GPT-2–style language model written in portable C11. It exists
as a readable reference — every piece (tensors, matrix math, softmax, GELU, LayerNorm,
multi-head causal attention, feed-forward, embeddings, sampling, checkpointing, and a
byte-level BPE tokenizer) is small enough to read in one sitting.

The model is currently **untrained** (weights are random) — the point is a correct,
end-to-end forward/generation pipeline you can build on. See `notes.md` for the roadmap.

## Project layout

Inspired by the Lua source tree: **every C source file lives flat in the project root —
there are no subfolders for code.** Each module is one concern, declared in a `.h` and
defined in the matching `.c`.

| File                              | Responsibility                                                |
| --------------------------------- | ------------------------------------------------------------- |
| `tensor.h` / `tensor.c`           | `Tensor` struct + aligned memory management                   |
| `matrix.h` / `matrix.c`           | `matmul`, `matadd`, `matscale`                                |
| `activation.h` / `activation.c`   | `softmax_rowwise`, `gelu`                                      |
| `layernorm.h` / `layernorm.c`     | `layernorm`                                                    |
| `feedforward.h` / `feedforward.c` | Position-wise feed-forward MLP: Linear → GELU → Linear        |
| `attention.h` / `attention.c`     | `multihead_attention_forward` (causal), `transformer_block_forward` |
| `embedding.h` / `embedding.c`     | Token + positional embedding lookup                           |
| `sampler.h` / `sampler.c`         | Token sampling from logits (greedy argmax)                    |
| `model.h` / `model.c`             | `GPT` model: embeddings, stacked blocks, output head, generation |
| `checkpoint.h` / `checkpoint.c`   | Save/load model weights (`gpt_save` / `gpt_load`, GPTC format) |
| `tokenizer.h` / `tokenizer.c`     | Self-contained byte-level BPE (`encode` / `decode`)           |
| `main.c`                          | Demo entry point                                              |
| `Makefile`                        | Build rules                                                   |
| `tools/`                          | Helper scripts in **other** languages (Python/Go/…) — not C   |

## Build & run

```sh
make            # compiles all modules into ./transformer
./transformer   # runs the demo (tokenizer + model + checkpoint round-trips)
make clean      # removes object files and the binary
```

Or compile directly, without make:

```sh
gcc -std=c11 -O3 -march=native -o transformer *.c -lm
```

`clang` works as a drop-in replacement for `gcc`. On a machine without gcc/clang, the
project also builds with `zig cc -std=c11 -O2 -Wall -o transformer *.c -lm`.

---

## CLI guide

### `transformer` (the demo binary)

Runs the full pipeline and prints a report. It takes no arguments today; edit the config
constants at the top of `main()` in `main.c` to change vocab size, dimensions, heads,
layers, prompt, or number of generated tokens.

```sh
$ ./transformer
=== Minimal GPT-2 Style Transformer Demo ===

[1] Byte-level BPE tokenizer
    text     : "hello hello" (11 bytes)
    encoded  : 5 tokens -> 258 111 32 258 111
    decoded  : "hello hello"
    round-trip: OK | compressed: yes

[2] GPT model
    config: vocab=16 d_model=8 heads=2 layers=2 d_ff=32
    generate: 1 5 9 -> 1 5 9 11 6 11 14 14
    logits  : no NaN/Inf

[3] Checkpoint save/load
    saved 'model.bin': OK | reloaded logits identical: yes

Output validation: PASSED (no NaN/Inf)
```

Exit code is `0` when all checks pass, `1` otherwise — handy in scripts:

```sh
./transformer && echo "smoke test ok"
```

### `tools/convert_to_gptc.py` (weight importer)

Converts a `.npz` of named weight arrays into the C checkpoint format so `gpt_load()`
can read it. This is the bridge for importing external weights (e.g. a converted GPT-2).

```sh
python tools/convert_to_gptc.py <weights.npz> <model.bin> --n-heads <N>
```

| Argument       | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| `weights.npz`  | input archive; array names/shapes must match the model params |
| `model.bin`    | output GPTC checkpoint                                         |
| `--n-heads N`  | number of attention heads (can't be inferred from shapes)     |

Expected `.npz` keys: `wte`, `wpe`, `blocks.{l}.{W_q,W_k,W_v,W_o,gamma1,beta1,gamma2,beta2,ff_W1,ff_b1,ff_W2,ff_b2}`, `ln_f_gamma`, `ln_f_beta`, `lm_head` (see the script's docstring for shapes). Load the result in C with `gpt_load("model.bin", &m)`.

---

## Programmatic guide (using the C API)

### Tokenizer

```c
#include "tokenizer.h"

Tokenizer tok = tokenizer_create_bytelevel();   // 256 byte tokens
int ll = tokenizer_add_merge(&tok, 'l', 'l');   // learn merges -> new token ids
tokenizer_add_merge(&tok, 'e', ll);

int ids[64];
int n = tokenizer_encode(&tok, "hello", ids, 64);

char out[128];
tokenizer_decode(&tok, ids, n, out, sizeof(out)); // "hello" (byte-exact round-trip)

tokenizer_free(&tok);
```

### Model: build, run, generate

```c
#include "model.h"

// vocab, d_model, n_heads, n_layers, d_ff, max_seq_len
GPT m = gpt_create(16, 8, 2, 2, 32, 16);

// Forward pass -> logits (seq_len, vocab_size)
int ids[3] = {1, 5, 9};
int shape[2] = {3, m.vocab_size};
Tensor logits = tensor_create(shape, 2);
gpt_forward(&m, ids, 3, &logits);

// Autoregressive greedy generation (appends in place; buffer must fit prompt+new)
int seq[16] = {1, 5, 9};
gpt_generate(&m, seq, /*prompt_len=*/3, /*n_new=*/5);

tensor_free(&logits);
gpt_free(&m);
```

### Checkpoints

```c
#include "checkpoint.h"

gpt_save("model.bin", &m);        // persist weights

GPT loaded;
gpt_load("model.bin", &loaded);   // rebuilds config + weights; free with gpt_free
```

### Importing external weights (e.g. GPT-2) — end-to-end

1. In Python, assemble the model's weights into a `.npz` with the key names above.
2. Convert: `python tools/convert_to_gptc.py weights.npz model.bin --n-heads N`.
3. In C: `GPT m; gpt_load("model.bin", &m);` then `gpt_forward` / `gpt_generate`.

> Note: loading *real* GPT-2 additionally requires matching its exact parameterization
> (attention biases, fused QKV, weight-tied output head). That is tracked as a follow-up
> in `notes.md`; the pieces above are the transport layer it will use.

---

## Coding rules (for contributors)

1. **Indentation is 2 spaces.** Never use tabs.
2. **The source root is C-only, and flat.** All `.c`/`.h` files live directly in the
   project root with no subfolders — do not mix other languages into the root. A pull
   request that adds a code subfolder or non-C source to the root will be rejected.
3. **Non-C helper / third-party scripts go in `tools/`.** Anything in another language
   (Python, Go, shell, …) — converters, downloaders, experiments — belongs under
   `tools/`, never in the root next to the C sources.
4. **One concern per module.** Declare the public API in the `.h`, define it in the
   matching `.c`, and give every header an include guard (`#ifndef MODULE_H`).
5. Keep helpers that are used by only one file `static` inside that `.c`.
6. **Never use untyped `void *` parameters** in a function's public API. Pass concrete,
   typed pointers (`Tensor *`, `float *`, …) instead — untyped pointers defeat the
   compiler's type checking and lead to slow, hard-to-predict runtime errors.
7. Code must compile cleanly under `-std=c11 -Wall` (no warnings).
8. When you add a new module, add its `.o` to `OBJS` in the `Makefile` and list its
   header dependencies there.
