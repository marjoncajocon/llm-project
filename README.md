# Minimal GPT-2 Style Transformer (Pure C11)

A tiny, dependency-free GPT-2–style language model written in portable C11. It exists
as a readable reference — every piece (tensors, matrix math, softmax, GELU, LayerNorm,
multi-head causal attention, feed-forward, embeddings, sampling, checkpointing, and a
byte-level BPE tokenizer) is small enough to read in one sitting.

The model starts from random weights and now **trains end-to-end** — full backpropagation
through attention, feed-forward, LayerNorm and embeddings, with Adam and a cross-entropy
loss, verified by numerical gradient checking. The demo overfits a tiny next-token task to
~0 loss. See `notes.md` for the roadmap.

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
| `sampler.h` / `sampler.c`         | Token sampling: greedy argmax + temperature / top-k / top-p (seeded RNG) |
| `model.h` / `model.c`             | `GPT` model: embeddings, stacked blocks, output head, generation |
| `checkpoint.h` / `checkpoint.c`   | Save/load model weights (`gpt_save` / `gpt_load`, GPTC format) |
| `tokenizer.h` / `tokenizer.c`     | Self-contained byte-level BPE (`encode` / `decode`)           |
| `loss.h` / `loss.c`               | Softmax cross-entropy (forward + backward)                    |
| `optim.h` / `optim.c`             | Adam optimizer                                                |
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

### Training (backprop + Adam)

Every forward op has a matching `*_backward` that **accumulates** into the `grad` array of
its inputs/parameters (call `tensor_zero_grad` before each step). Cross-entropy is in
`loss.c`, the optimizer in `optim.c`. A training step looks like:

```c
#include "loss.h"
#include "optim.h"

Tensor *params[] = {&W1, &b1, &W2, &b2};
Adam opt = adam_create(params, 4, /*lr=*/0.05f);

for (int step = 0; step < n_steps; step++) {
  // forward: ... -> logits
  float loss = cross_entropy_forward(&logits, targets, &probs);

  // backward: zero grads, then propagate from the loss back to every parameter
  cross_entropy_backward(&probs, targets, &logits);
  // ... matmul_backward / bias_add_backward / gelu_backward / layernorm_backward ...

  adam_step(&opt, params, 4);   // updates params using their .grad
}
adam_free(&opt);
```

The **full `GPT` trains end-to-end**: `gpt_forward_train` (in `model.c`) caches activations,
`gpt_backward` propagates through the output head, final LayerNorm, every stacked block
(multi-head causal attention + FFN + residuals), and the embeddings, and `gpt_collect_params`
feeds Adam. A training step:

```c
#include "model.h"
Tensor *params[64];
int n = gpt_collect_params(&m, params);
Adam opt = adam_create(params, n, 0.02f);
GPTActs acts = gpt_acts_create(&m, seq_len);

gpt_forward_train(&m, ids, seq_len, &acts, &logits);
float loss = cross_entropy_forward(&logits, targets, &probs);
tensor_zero_grad(&logits);
cross_entropy_backward(&probs, targets, &logits);
gpt_backward(&m, ids, seq_len, &acts, &logits);   // grads into every parameter
adam_step(&opt, params, n);
```

Gradients are verified in `main.c` by **end-to-end numerical gradient checking** (analytic
`grad` vs `(L(+ε) − L(−ε)) / 2ε`) across the whole model, then it trains the GPT on a
next-token task and overfits it to ~0 loss. (Note: float32 finite-difference checking has a
noise floor for tiny gradients, so the check trusts well-conditioned entries; see `notes.md`.)

### Importing external weights (e.g. GPT-2) — end-to-end

The model is **GPT-2-architecture-compatible**: pre-norm blocks, multi-head causal attention
**with biases** (`b_q/b_k/b_v/b_o`, matching GPT-2's `c_attn`/`c_proj`), GELU FFN, learned
positional embeddings, and a final LayerNorm. The output head is a separate `lm_head`, so
GPT-2's tied head is written as `lm_head = wte^T` at conversion time.

**Real GPT-2 (124M):**
```sh
pip install numpy safetensors
huggingface-cli download gpt2 model.safetensors --local-dir gpt2/
python tools/convert_gpt2.py gpt2/model.safetensors gpt2.npz     # HF -> our layout
python tools/convert_to_gptc.py gpt2.npz gpt2.bin --n-heads 12   # -> GPTC checkpoint
```
Then in C: `GPT m; gpt_load("gpt2.bin", &m);` and `gpt_forward` / `gpt_generate`.
(Config: vocab 50257, d_model 768, 12 layers, 12 heads, d_ff 3072, ctx 1024.)

**Your own weights:** assemble a `.npz` with the documented key names (incl. `b_q…b_o`) and
run `convert_to_gptc.py` directly. The GPTC format is **v2** (adds the attention biases).

> Not yet wired: GPT-2's real BPE tokenizer (`encoder.json` / `vocab.bpe`) — the byte-level
> BPE here is self-contained. That plus running the full 124M weights is the next milestone
> in `notes.md`.

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
