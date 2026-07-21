# Project Notes

## Step 1 — Complete the transformer block (FFN + GELU + pre-norm)

**Status: DONE — smoke test passed.**

### What changed
The transformer block was previously `attention → +residual → LN1 → LN2`, where the
second LayerNorm stood in where the feed-forward network belongs, using post-norm
ordering. It is now a real GPT-2 pre-norm block with a feed-forward MLP:

```
a = input + Attention(LN1(input))
output = a + FFN(LN2(a))       where FFN(h) = GELU(h @ W1 + b1) @ W2 + b2
```

- **GELU** activation added to `activation.c` (GPT-2 tanh approximation).
- New flat module **`feedforward.h` / `feedforward.c`**: Linear → GELU → Linear with
  biases, hidden size `d_ff = 4 * d_model`.
- **`transformer_block_forward`** rewritten to pre-norm ordering with the FFN sub-layer;
  signature extended with the four FFN weight tensors and `d_ff`.
- **`main.c`** creates, initializes, passes, and frees the FFN weights.
- **Portability fix** (needed to build/smoke-test on Windows): `tensor.c` now uses a
  per-platform aligned allocator — `_aligned_malloc`/`_aligned_free` on Windows,
  `aligned_alloc`/`free` elsewhere. Previously `aligned_alloc` failed to compile on the
  Windows CRT, and blocks were freed with the wrong deallocator.

### Progress checklist
- [x] GELU activation (`activation.h` / `activation.c`)
- [x] Feed-forward module (`feedforward.h` / `feedforward.c`)
- [x] Pre-norm block reorder + FFN (`attention.h` / `attention.c`)
- [x] Wire FFN weights into `main.c`
- [x] `Makefile` updated (`feedforward.o` + deps)
- [x] `README.md` updated (module table + no-`void *`-parameter rule)
- [x] Portable aligned alloc/free in `tensor.c`
- [x] `.gitignore` added
- [x] Smoke test (build + run → `Output validation: PASSED (no NaN/Inf)`)

### Smoke test result (2026-07-21)
Built with `zig cc` (no gcc/clang/make on this machine; `zig cc` is a C11-compliant
clang driver). `-Wall` clean, program ran to completion:

```
Output shape: (4, 8)
Output validation: PASSED (no NaN/Inf)
Done! All memory freed successfully.
```

Build command used:
```sh
zig cc -std=c11 -O2 -Wall -o transformer.exe *.c -lm
```
The `Makefile` (`make`) produces the same result on a machine with gcc/clang installed.

## Step 2 — Multi-head attention + causal masking

**Status: DONE — smoke test passed.**

### What changed
Single-head `self_attention_forward` was replaced by
**`multihead_attention_forward`**, which:
- Projects the input into `n_heads` heads (`head_dim = d_model / n_heads`) using square
  `(d_model, d_model)` weight matrices `W_q/W_k/W_v/W_o`.
- Runs scaled dot-product attention per head with a **causal mask**: position `i` may
  attend only to positions `j <= i`; future positions get score `-INFINITY`, so after
  softmax their weight is exactly 0 (autoregressive behavior, like GPT-2).
- Concatenates the per-head outputs and projects through `W_o`.

`transformer_block_forward` now calls the multi-head function; its `d_k` parameter became
`n_heads`. `main.c` uses `n_heads = 2` and square attention weights.

### Progress checklist
- [x] Multi-head projection + per-head attention (`attention.h` / `attention.c`)
- [x] Causal masking (future positions -> -INFINITY before softmax)
- [x] `transformer_block_forward` updated (`d_k` -> `n_heads`)
- [x] `main.c` updated (square weights, `n_heads = 2`)
- [x] `README.md` updated (attention module description)
- [x] Smoke test (build + run -> `Output validation: PASSED`, `-Wall` clean)

### Smoke test result (2026-07-21)
Built with `zig cc -std=c11 -O2 -Wall`; `-Wall` clean, ran to completion with
`Output validation: PASSED (no NaN/Inf)`.

## Next steps (not started)
- Step 3: token + positional embeddings, stacked blocks, output projection, sampler.
- Step 4: BPE tokenizer + load real GPT-2 weights.
- Step 5: backprop + training (uses the unused `Tensor.grad` field).
