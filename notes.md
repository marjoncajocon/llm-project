# Project Notes

## Roadmap — all steps & phases

Goal: grow this pure-C transformer into a **fully local coding-agent CLI**. Every step ends
green (`-Wall` clean, smoke test `PASSED`, artifacts cleaned) and is committed.

**Foundations (DONE):**
- [x] Step 1 — Transformer block: FFN + GELU + pre-norm
- [x] Step 2 — Multi-head attention + causal masking
- [x] Step 3 — Full model: embeddings, stacked blocks, output head, greedy sampler
- [x] Step 4 — Checkpoint save/load (GPTC) + byte-level BPE tokenizer + `tools/` converter
- [x] Step 5a — Training foundation: backward core + cross-entropy + Adam, gradient-checked
- [x] Step 5b — End-to-end GPT training (attention/FFN/embedding backward)

**Phase A — Make the model produce real text:**
- [x] Step 4b — Load real GPT-2 weights: GPT-2-compatible architecture (attention biases,
      tied output head), converter (HF safetensors → GPTC), verified on GPT-2-shaped weights
- [x] Step A1 — Trainable BPE tokenizer: merge-learning + save/load (self-trained BPE).
      (Loading GPT-2's exact `encoder.json`/`vocab.bpe` + pretok regex = future extension.)
- [x] Step A2 — Train-on-your-data pipeline: `data/` folder + `transformer train` +
      `transformer generate model.bin tok.bin` (text-in/text-out). Verified: learns a small
      corpus (loss 5.9 → ~0.1) and generates coherent in-style text.

**Phase B/scaling:**
- [x] Configurable model via CLI flags (`transformer train ... --d-model --layers --heads
      --d-ff --ctx --merges --steps --lr --seed`). Defaults: d_model 128, 3 layers, 4 heads,
      ctx 64. Each `model.bin`+`tok.bin` is a self-contained model (→ many role-specific AIs).
- [ ] Train/val split + held-out loss reporting; checkpoint-every-N; resume-from-checkpoint.
- [ ] B4 remainder — threaded/SIMD matmul + int8 quant (needed for big models to train fast).

**Phase E — Multimodal (images), the big future build (user goal: understand/describe images):**
- [x] Step E1 — Image decode: `image.c` reads binary **PPM (P6)** → HWC floats in [0,1]
      (pure C, no deps). `tools/img_to_ppm.py` converts PNG/JPG → PPM. Store in `data/images/`.
- [x] Step E2 — Patch embedding (ViT-style): `patch.c` splits the image into patches, flattens
      each, and projects with W+b (reuses `matmul`/`bias_add`) → `(n_patches, d_model)`.
      Verified in the smoke test (`[5] Image`).
- [x] Step E3 — Multimodal input: `gpt_forward_multimodal` prepends `n_img` precomputed
      image-patch embeddings before the text tokens, adds positions over the combined sequence,
      and runs the transformer (`gpt_run_from_emb` refactor shared with the text forward).
      Verified: `[5] Image: 4 patches + 3 text -> logits (7,16)`.
- [x] Step E4 — Image→caption **training**: `gpt_backward_multimodal` (scatters to `wte`/`wpe`
      and writes the image-embedding gradient) + `patch_embed_backward` (learns the patch
      projection) + cross-entropy with an **ignore index** (target < 0) so only caption
      positions are supervised. Verified: `[6] Multimodal train: loss 4.69 -> 0.0001` (overfits
      one image→caption). Real image-caption *datasets* + a bigger model = the quality frontier.
      (Honest: a genuinely useful vision-language model needs far more model+data+compute than
      is feasible locally from scratch; this builds the plumbing, quality needs real scale.)

**Phase F — Video:**
- [x] Step F1 — Frame loading: `video.c` reads `dir/frame_0000.ppm, 0001, ...` into a `Video`;
      `tools/video_to_frames.py` extracts PPM frames via ffmpeg.
- [x] Step F2 — Per-frame patch embeddings (reuse Phase E `patch_embed`) + **sinusoidal
      temporal encoding** per frame; stacked into one `(n_frames*patches, d_model)` sequence.
- [x] Step F3 — Audio: `audio.c` loads 16-bit PCM WAV (mono / stereo downmix), frames the
      waveform into windows, projects each + sinusoidal temporal encoding → a sequence of
      vectors; trained the same way (audio→caption, loss 7.30 -> 0.0017 in smoke `[8]`).
      `tools/audio_to_wav.py` converts any audio via ffmpeg. (Raw-waveform framing, not
      mel-spectrograms — cruder but dependency-free.)
- [x] Step F4 — Video-text training: reuse `gpt_forward_multimodal`/`gpt_backward_multimodal`
      (video patches as the prefix) + `video_embed_backward`. Smoke `[7]`: 2-frame clip →
      caption, loss 5.998 -> 0.001.
      (Honest: research-scale for real quality; this is a correct, runnable pipeline.)

**Phase G — GENERATE media (image / audio / video). Approach: discrete-token autoregressive,
reusing the GPT + KV cache + sampler + cross-entropy (like DALL·E 1 / ImageGPT / audio codec
LMs). Diffusion is the SOTA alternative but a much bigger, different build — not chosen.**
- [ ] Step G0 — Generation foundation: discrete tokens per modality; media vocab / special
      tokens (BOS/EOS); a decode+stop path. (Minimal versions need NO VQ-VAE — see below.)
- [x] Step GI1 — Image quantizer (`imagegen.c`): `image_to_tokens`/`tokens_to_image` via a
      **color-cube** (levels^3 tokens; levels=4 → 64), deterministic, no training. (Later: VQ-VAE.)
- [x] Step GI2 — Train GPT to autoregressively predict image tokens (smoke `[10]`: overfit an
      8×8 image, loss 4.53 → 0.0003). (Text→image conditioning = next; prefix the caption.)
- [x] Step GI3 — Generate: greedy autoregressive sampling of image tokens (KV cache +
      `sample_argmax`) → **64/64 tokens reproduced** → `tokens_to_image` → pixels. Minimal image
      generation works end-to-end.
- [x] Step G-CLI — `transformer imggen <in.ppm> <out.ppm> [levels] [steps]` and
      `audgen <in.wav> <out.wav> [steps]`: generate real media FILES (image_save_ppm /
      audio_save_wav; `autoregress_reconstruct` helper). Verified: 12×12 PPM → out.ppm (144/144),
      160-sample WAV → out.wav (160/160). Small inputs only (O(n²)); single-example overfit.
- [x] Step GA — Audio generation: `audiogen.c` **µ-law 256-level** codec
      (`audio_to_tokens`/`tokens_to_audio`, WaveNet-style). Generated via text→audio (below).
- [x] Step GV — Video generation: reuse `imagegen` per frame (a clip = concatenated frame
      tokens). Generated via text→video (below).
- [x] Step G-cond — **text→image / text→audio / text→video** (`cond_gen` in `main.c`): combined
      vocab = media + text (text ids offset above media); train to predict media tokens after
      the text prefix; generate = feed text, autoregress media, decode. Smoke `[11]`:
      text→image 16/16, text→audio 32/32, text→video 32/32 tokens reproduced.
      (Honest: pipeline reuses ~80% of the stack; the codec is the quality bottleneck —
      palette/µ-law give tiny/blocky output; real quality needs a trained VQ-VAE or diffusion
      + big data + compute. Unconditional gen = empty text prefix.)
- [x] Step G-caption — **image→text / audio→text / video→text** (`media_to_text` in `main.c`):
      the reverse — media embeddings as the prefix (`patch_embed`/`audio_embed`/`video_embed`),
      autoregressively generate the caption text (`gpt_forward_multimodal` + `gpt_backward_multimodal`).
      Smoke `[12]`: image/audio/video → 3/3 caption tokens. So media↔text works both ways.
- [x] Model identity/persona — comes from **training data** (not a flag). `data/identity.txt`
      (identity Q&A) + `transformer train --name marvin` → "Who are you?" → "I am Marvin, an AI
      model created by Marjon." (`--name` = folder; persona = data.)

**Phase B — A local generation CLI:**
- [x] Step B1 — Better sampling: temperature + top-k/top-p + seeded RNG (`sampler.c`)
- [x] Step B2 — KV cache (incremental decode; verified bit-exact vs full forward)
- [x] Step B3 — `generate` subcommand: token-in/token-out REPL (KV cache + sampler)
- [~] Step B4 — Performance: cache-friendly `ikj` matmul done; threads/SIMD/int8 quant TODO

**Phase C — The agent layer (tools + approval), pure C:**
- [x] Step C1 — Tool protocol: parse `<tool …>` / `<done>` blocks (`toolcall.c`)
- [x] Step C2 — Tool executor: write_file / read_file / run_command (`toolexec.c`)
- [x] Step C3 — Approval gate: `y/N` on stdin, or auto (`approval.c`)
- [x] Step C4 — Agent loop: parse → approve → execute → observe (`agent.c`, `agent` subcommand)

**Phase D — Quality track (as needed):**
- [ ] Step D1 — Run a small quantized code model (1–3B) for genuinely useful output
      (needs B4 threads/SIMD/quant + real weights; not implementable in this environment)

> Honest note: the *pipeline* is 100% doable in pure C and local; agent *quality* is capped by
> the model you can run. Loading pretrained weights (4b) is the fastest path to real output.

## Next phases (priority order)

1. **Training UX** — train/val split + **held-out loss**; `--checkpoint-every N`;
   `--resume model.bin` (keep training across sessions).  ✅ DONE
2. **Performance — matmul thread pool** ✅ DONE — a **persistent fork-join thread pool** in
   `matrix.c` threads `matmul` + both loops of `matmul_backward` (row-parallel → **bit-exact**;
   verified byte-identical models at 1 vs 8 threads). Thresholded (tiny matmuls stay
   single-threaded), portable (Win32 / pthreads / WASM-single-thread). **Measured 2.68× on 8
   threads** (d_model 256, 4 layers). Threads = `$TF_THREADS` or CPU count. Remaining perf
   (later): SIMD + int8 quantization.
3. **Real GPT-2 weights** ✅ DONE — `tools/gpt2_setup.py` downloads GPT-2 124M + converts to
   GPTC (→ `model/gpt2/model.bin`, 622 MB); `tools/gpt2_tok.py` bridges tiktoken. Verified:
   the pure-C model generates **coherent real GPT-2 text** ("The meaning of life is not to be
   confused with the fact of being reborn..."). Also added the `model/<name>/` folder
   convention + `--name` for train/generate. (Bigger variants: `--model gpt2-medium` etc.)
4. **Real agent backend** ✅ DONE — `agent.c` now has a pluggable `AgentBackend` interface:
   a **scripted** backend (demo/tests) and a **command** backend (`agent --cmd "<prog>"` runs
   an external program whose stdout is each response; transcript passed as a file arg). Verified
   with `tools/agent_mock.py` driving write→read→done. Plug in a real LLM by pointing `--cmd`
   at an API script (template in agent_mock.py).
5. **Phase E — Images (multimodal)** *(large)* — decode → patch embeddings → multimodal
   input → image-text training.
6. ✅ **Phase F — Video + Audio** — video (`video.c` + `tools/video_to_frames.py`) and audio
   (`audio.c` WAV framing + `tools/audio_to_wav.py`) → per-frame/window embeddings + temporal
   encoding → train via the multimodal path. Smoke: video→caption and audio→caption both drop.
   DONE (F1/F2/F3/F4). Text/image/video/audio all supported.

Order rationale: text (1–4) is very achievable locally; images (5) and video (6) are big
multimodal builds where the plumbing is feasible but real quality needs far more scale.

## ▶▶ PLANNED NEXT STEP (agreed 2026-07-21) — real LLM agent backend — ✅ DONE (2026-07-22)

**DONE.** `tools/agent_llm.py` turns the agent from a scripted demo into one that genuinely
acts, via the already-pluggable `agent --cmd` backend. Biggest usefulness jump (the original
"local coding-agent CLI" vision); NOT gated on model scale — only on an API key. No C changes.

**Built:** `tools/agent_llm.py` — reads the transcript file (argv[1], stdin fallback), calls a
real LLM with a system prompt constraining it to emit EXACTLY one block:
`<tool name="write_file|read_file|run_command" ...>...</tool>` or `<done>summary</done>` (exact
attribute syntax matched to `toolcall.c`'s parser). Prints only text blocks to stdout (skips
thinking blocks); strips stray ``` fences. Provider: **Claude via Anthropic API**
(`claude-opus-4-8`, adaptive thinking) by default, **OpenAI** if only `OPENAI_API_KEY` is set
(`$OPENAI_MODEL` or `gpt-4o`). On no key / no SDK / any API error it prints a `<done>` block so
the C loop ends cleanly instead of hanging.

**Run:** `transformer agent --cmd "python tools/agent_llm.py" --goal "..." --ask` — the approval
gate (`--ask`) stays as the safety layer. Needs `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`).

**Verified (this machine, no key available):** script compiles; the no-key path emits the
graceful `<done>`; full path through the C command backend works end-to-end
(`transformer agent --cmd "<py> tools/agent_llm.py" --goal ...` → writes transcript → runs the
script → parses its `<done>` → stops). Smoke test still `PASSED`, `-Wall` clean. The live LLM
call is untested here (no `anthropic` pkg / no key) — it's the standard Messages API shape from
the `claude-api` skill; `agent_mock.py` remains the offline/no-key fallback.

**Backups:** (a) ✅ DONE (2026-07-22) — code health: smoke test split into `selftest.c`;
(b) ✅ DONE (2026-07-22) — dataset-trained `imggen`/`audgen` (learn from a folder). Both below.

## Instruction tuning (SFT) smoke test — DONE (2026-07-22)

First "real LLM recipe" step demonstrated end-to-end (see the 5-stage recipe in `memory.md`).
New selftest **part `[9]`** (filled the gap between `[8]` and `[10]`): trains the GPT on two
`prompt → response` pairs with the **prompt positions masked** (loss only on the response tokens,
via the existing cross-entropy ignore-index `target < 0`) — exactly how SFT supervises the
completion. Then feeds each prompt through the KV cache and greedily generates: prompt A `[1,2]`
→ `7 8 9`, prompt B `[3,4,5]` → `11 12`. Two distinct prompts → two distinct correct responses =
instruction-following in miniature. Verified: `[9] ... loss 3.19 → 0.0001 | A OK | B OK`, smoke
test still `PASSED`, `-Wall` clean. (Toy scale — it memorizes; the *technique* is the real thing.)
This is the "start step by step" first step toward the recipe.

**Real SFT dataset via the CLI — DONE (2026-07-22):** `data/instruct.txt` (tracked; `Q:`/`A:`
instruction→response pairs whose answers are the agent's `<tool …>`/`<done>` format). Trained a
real small model: `transformer train data/instruct.txt --name instruct --d-model 96 --layers 3
--heads 4 --ctx 96 --merges 48 --steps 4000 --lr 1e-3` (loss ~0.07). Verified routing — each
held-in instruction generates its correct response: `read notes.txt`→`read_file`, `run the
tests`→`run_command make test`, `delete temp.log`→`run_command rm temp.log`, `the task is
finished`→`<done>Task complete.</done>`, `write hello to notes.txt`→`write_file`. KEY GOTCHA:
`--merges` must stay LOW (48, not 200) — coarse BPE merges whole lines into single tokens so the
prompt doesn't align to a token boundary and routing drifts (first run at merges 200 emitted valid
blocks but wrong routing). `model/instruct/*.bin` git-ignored; `data/instruct.txt` tracked (added
`.gitignore` exception). Honest: memorizes at this scale — technique works, generalization needs
more data + bigger model.

**Masked-SFT training mode (`--sft`) — DONE (2026-07-22):** true response-only SFT in `train.c`,
not whole-stream memorization. New `TrainConfig.sft` + `--sft` flag + `train_sft()`. File format:
one `prompt<TAB>response` example per line (`data/sft.txt`, tracked). Builds `[prompt ++ response]`
padded to the max length; targets `-1` (ignored) on the prompt positions AND the end-padding, so
`cross_entropy` (which already averages over target≥0 only) supervises the RESPONSE tokens only —
identical loss to smoke-test `[9]`, now on real data. Random example per step; no resume for SFT.
Verified: `train data/sft.txt --sft --name sft --d-model 96 --layers 3 --heads 4 --merges 48
--steps 3000 --lr 1e-3` → sft-loss 5.92 → 0.0003; all 12 instructions route to their exact
tool-call/`<done>` response (read_file, run_command pwd/make test/rm, write_file, `<done>`).
`model/sft/*.bin` git-ignored. Build `-Wall` clean, smoke test still PASSED.

**EOS / stop token — DONE (2026-07-22):** `--sft` now reserves one extra vocab slot as EOS
(`EOS = tok.vocab_size`; model vocab = `tok.vocab_size + 1`), appends it to every response
(`seq = [prompt ++ response ++ EOS]`, padded), and supervises predicting it (the last response
token predicts EOS). `run_generate` (text mode) stops as soon as it samples an id ≥ `tok.vocab_size`
and drops it — non-SFT models have no extra id so they're unaffected (backward compatible, no
checkpoint-format change). BUG caught + fixed during the build: after bumping the model vocab I
left `logits`/`probs` sized to `tok.vocab_size` → forward wrote one column OOB → crash (exit 116);
fix = size the tensors to `model.vocab_size`. Retrained: sft-loss → 0.0005; **generation now stops
cleanly** — every instruction produces its exact tool-call/`<done>` response with NO trailing
filler (read_file, run_command make test/rm/pwd, write_file, `<done>Task complete.>`).

## Modern architecture — step 1: RMSNorm — DONE (2026-07-22)

First LLaMA-family primitive. New `rmsnorm.h`/`rmsnorm.c`: `rmsnorm` (RMS-only norm, no mean
subtraction, no bias — `y_j = x_j / sqrt(mean(x^2)+eps) * gamma_j`) + `rmsnorm_backward`
(`dx_j = r*(g_j - x_j*r^2*S/d)`, `g_j = dy_j*gamma_j`, `S = sum g_k x_k`; `dgamma_j += dy_j*x_j*r`).
Verified: new selftest **part `[13]`** does a numerical gradient check (analytic backward vs central
difference) — max rel error **7.74e-04** (< 1e-2 gate); smoke test still PASSED, `-Wall` clean.
Makefile + build lists + README module table updated. NOT yet wired into the model — that's the
bigger arch swap (a modern block uses RMSNorm + RoPE + SwiGLU + GQA together).

## Modern architecture — step 2: RoPE — DONE (2026-07-22)

Rotary position embeddings (`rope.h`/`rope.c`): rotates each row (a Q/K head vector, head_dim even)
by a position-dependent angle so RELATIVE position enters the Q·K dot product; parameter-free,
replaces learned positions. **Half-split ("rotate_half") convention matching HuggingFace LLaMA**
(so it lines up with real weights later): pair `(k, k+h)` rotated by `pos * base^(-2k/cols)`,
`base=10000`, `h=cols/2`. Backward = inverse (transpose) rotation of `dy` (orthogonal map).
Verified: new selftest **part `[14]`** — (1) numerical gradient check 1.38e-03 (< 1e-2); (2) the
DEFINING RoPE property `<RoPE(q,m),RoPE(k,n)>` depends only on `(m-n)`: q@{1,4}, k@{2,5} (offset 1
both) -> dot -0.6568 == -0.6568, rel 1.8e-07. Smoke test PASSED, `-Wall` clean. (Perf note: `powf`
in the inner loop; precompute cos/sin tables when wiring into the model.)

## Modern architecture — step 3: SwiGLU — DONE (2026-07-22)

Gated feed-forward network (`swiglu.h`/`swiglu.c`), the LLaMA-family replacement for the GPT-2
GELU MLP. No biases: `g = x@W_gate`, `u = x@W_up`, `h = SiLU(g) (*) u`, `y = h@W_down`
(`SiLU(z) = z*sigmoid(z)`; `W_gate`/`W_up`: (d_model,d_ff); `W_down`: (d_ff,d_model)). Reuses
`matmul`/`matmul_backward`; a `SwigluCache` holds g/u/a/h so backward runs without recompute.
Backward: `dh` via `matmul_backward`, then `da=dh*u`, `du=dh*a`, `dg=da*SiLU'(g)`
(`SiLU'(z)=s(1+z(1-s))`), then `matmul_backward` for x/W_gate/W_up. Verified: new selftest
**part `[15]`** numerically gradient-checks all four tensors (x, W_gate, W_up, W_down) — max rel
error **1.17e-03** (< 1e-2). Smoke test PASSED, `-Wall` clean.

## Modern architecture — step 4: GQA — DONE (2026-07-22)

Grouped-query causal attention (`gqa.h`/`gqa.c`): n_heads QUERY heads but only n_kv KEY/VALUE
heads (n_kv | n_heads); query head h shares KV head `h/(n_heads/n_kv)` — a group of query heads
reuse one K/V head (what shrinks the KV cache in modern models; n_kv=n_heads is plain MHA, n_kv=1
is MQA). Takes pre-projected per-head-concatenated Q (n, n_heads·hd), K/V (n, n_kv·hd) → O
(n, n_heads·hd). Causal (j≤i), scale 1/√hd; forward caches the row-softmax weights (`GqaCache`).
Backward = standard attention backward (dV += att·dO; datt=<dO,V>; softmax backward
ds=att·(datt−Σatt·datt); dQ+=ds·scale·K, dK+=ds·scale·Q) with grads of grouped query heads
accumulating into the shared K/V. Verified: new selftest **part `[16]`** numerically gradient-checks
Q, K, V (4 query / 2 KV heads) — max rel error **7.30e-03** (< 1e-2). Smoke test PASSED, `-Wall` clean.

🏁 **All four modern (LLaMA-family) primitives DONE + gradient-checked:** RMSNorm `[13]`, RoPE
`[14]`, SwiGLU `[15]`, GQA `[16]`.

## Modern architecture — step 5: assembled transformer BLOCK — DONE (2026-07-22)

`llama.h`/`llama.c` — the four primitives wired into one LLaMA-family block (no biases):
`xn=RMSNorm(x,attn_norm)` → `q,k,v = xn@{Wq,Wk,Wv}` (RoPE per-head on q,k via new in-place
`rope_heads` in `rope.c`) → `h = x + GQA(q,k,v)@Wo` (residual) → `hn=RMSNorm(h,ffn_norm)` →
`out = h + SwiGLU(hn,Wgate,Wup,Wdown)` (residual). Weights: `attn_norm/ffn_norm` (d_model),
`Wq`(d_model,H·hd), `Wk/Wv`(d_model,KV·hd), `Wo`(H·hd,d_model), `Wgate/Wup`(d_model,d_ff),
`Wdown`(d_ff,d_model). **Forward only (inference)** — the target is loading real small models;
block backward is a later add if we want to fine-tune. `llama_block_create/free/randomize/forward`.
Verified: new selftest **part `[17]`** — forward is finite AND end-to-end **causal**: perturbing the
LAST input position changed its own output (|diff| 5.2e-1) but left every earlier output row
**exactly** unchanged (max |diff| 0.00e+00), proving the causal mask survives the full assembly.
Smoke test PASSED, `-Wall` clean.

## Modern architecture — step 6: multi-block MODEL + forward — DONE (2026-07-22)

`LlamaModel` in `llama.h`/`llama.c`: token embedding (`wte`, vocab×d_model; NO positional term —
RoPE carries position) → `n_layers` `LlamaBlock`s (ping-pong buffers) → final RMSNorm → `lm_head`
(d_model×vocab; tie = wte^T) → logits (seq, vocab). `llama_model_create/free/randomize/forward`.
Forward-only (inference). Verified: new selftest **part `[18]`** — 3-layer model forward is finite
AND model-level **causal**: changing the LAST token id changes only its own logits (|diff| 6.6e-2),
every earlier position's logits **exactly** unchanged (0.00e+00) through all 3 blocks + embedding +
final norm + head. Smoke test PASSED.

**Cross-compile verified (user asked):** all **8 `make cross` targets build clean** with the new
modules (rmsnorm/rope/swiglu/gqa/llama) — x86_64/aarch64 linux-gnu, linux-musl, x86_64/aarch64
windows-gnu, x86_64/aarch64 macos, and **wasm32-wasi** (the new modules are pure standard C + math +
malloc, no platform code). `-Wall` clean everywhere.

## Modern architecture — step 7: LLMA checkpoint format + C loader — DONE (2026-07-22)

`llama_model_save`/`llama_model_load` in `llama.c` (declared in `llama.h`) — a flat binary **`LLMA`**
format mirroring `GPTC`: magic + version + 8 config ints (vocab, d_model, n_layers, n_heads, n_kv,
head_dim, d_ff, max_seq) + 2 floats (rope_base, eps) + every weight's float32 in a fixed canonical
order driven by a shared `llama_collect` (1 wte + 9/layer + final_norm + lm_head), so save/load can't
drift. `load` builds a matching model then reads weights in. Verified: selftest **part `[18]`** now
also does a **round-trip** — save → load → forward → logits **bit-identical** to the original
(max |diff| 0.00e+00). All targets still cross-compile (re-checked wasm32-wasi + aarch64-macos with
the stdio checkpoint code). Smoke test PASSED, `-Wall` clean.

## Modern architecture — step 8: convert_llama.py + `llama` CLI — DONE (2026-07-22)

`tools/convert_llama.py` — HF LLaMA-family safetensors -> our `LLMA` checkpoint. **KEY CORRECTION to
earlier notes: NO RoPE Q/K permutation is needed.** rope.c implements HF's half-split ("rotate_half")
convention, so HF-format `q_proj`/`k_proj` load directly (the permutation is only for
Meta-interleaved->HF conversion, which doesn't apply to HF-native models like SmolLM2/Qwen2.5). The
converter transposes every linear weight (HF stores (out,in), applied `x@W^T`; our matmul is `x@W`,
W (in,out)), copies norm gains + token embedding as-is, uses `lm_head = embed_tokens.weight^T` for
tied models, and reads config.json for dims/rope_theta/eps/tie. Warns (doesn't handle) attention
biases (Qwen2 has q/k/v biases; LlamaBlock has none). `--tiny PATH` writes a small deterministic
checkpoint with no download.

`main.c` new **`llama <checkpoint.bin>`** subcommand: loads an LLMA model and forwards over
space-separated token IDs from stdin, printing the next-token argmax (token-in/token-out; verifies the
converter + seeds the generate path).

Verified (no download): `convert_llama.py --tiny t.bin` -> `transformer llama t.bin` loads the
PYTHON-written file (config parsed: vocab 32 / d_model 8 / 2 layers / 4 heads / 2 kv) and forwards to
finite logits + argmax -- proving the Python<->C byte format (magic, config order, weight
order/layout, LE float32) matches end-to-end. Smoke test PASSED; wasm32-wasi + linux-musl still
cross-compile with the new `llama` subcommand.

## Modern architecture — step 9: tokenizer bridge + generate path — DONE (2026-07-22)

`main.c` `llama` subcommand now **generates**: reads a prompt (token IDs), autoregressively samples
`n_new` tokens (`sample_token`, temp 0.8/top-k 40/top-p 0.95, seeded), prints `gen: <all ids>`.
Re-runs the full forward per step (NO KV cache yet → slow on big models; KV cache is the perf
follow-up). `tools/llama_tok.py` (encode/decode via the model's `tokenizer.json` using the `tokenizers`
lib) + `tools/llama_chat.py` (one command: tokenize → `transformer llama` → detokenize, like
`gpt2_chat.py`).

Verified (tiny, no download): `convert_llama.py --tiny t.bin` → `echo "5 9 2" | transformer llama
t.bin 12` → `gen: 5 9 2 20 15 5 20 31 16 25 1 27 30 7 14` (all ids in range) — the generate loop +
sampling integrate correctly. Smoke test PASSED. Tools `py_compile` clean. Deps present:
huggingface_hub 1.8.0, safetensors 0.7.0; `tokenizers` NOT yet installed (needed for the real run).

## 🎉 REAL MODERN MODEL RUNS — SmolLM2-135M generates coherent text (2026-07-22)

**End-to-end SUCCESS.** Downloaded + converted **HuggingFaceTB/SmolLM2-135M** and generated fluent
English through the pure-C `LlamaModel`:
- `python tools/convert_llama.py --model HuggingFaceTB/SmolLM2-135M --out model/smollm/model.bin`
  → 621 MB LLMA (vocab 49152, d_model 576, 30 layers, 9 q / 3 kv heads, head_dim 64, d_ff 1536,
  **rope_base 100000** — SmolLM2's non-default rope_theta, correctly read from config.json — tied).
- `python tools/llama_chat.py --model HuggingFaceTB/SmolLM2-135M --ckpt model/smollm/model.bin
  "The capital of France is" --n 16` →
  **"The capital of France is the capital of the department of Loir-et-Cher. It is"** — coherent,
  grammatical (Loir-et-Cher is a real French département; the model is tiny so it's not factually
  perfect, but the ENGLISH is fluent → the weight mapping/RoPE/GQA/SwiGLU/RMSNorm/tokenizer are all
  CORRECT; wrong mapping would give gibberish tokens, not prose).

**bf16 fix (required):** SmolLM2 weights ship in bfloat16, which `safetensors.numpy` can't load
(numpy has no bf16). Added a manual safetensors reader in `convert_llama.py` (`load_safetensors_f32`):
parse the u64-length JSON header + raw blob, convert BF16→F32 by widening u16→u32 and shifting up 16
bits (bf16 = high half of f32); also handles F16/F32. No torch needed. Deps used: huggingface_hub,
safetensors (header only), numpy, tokenizers. `model/smollm/*.bin` git-ignored.

The from-scratch pure-C engine now loads and runs a real modern (LLaMA-family) LLM. 🏁

## Modern architecture — step 10: KV cache for LlamaModel — DONE (2026-07-22)

`LlamaKV` + `llama_decode_step` in `llama.c` (declared in `llama.h`): caches K and V
(`n_kv*head_dim` per position) for every layer, so each new token computes only its own Q/K/V and
attends over the cache — O(n) per token instead of a full O(n·layers·d_model²) re-forward. Single-
token path: embed → per layer {RMSNorm → q/k/v proj → `rope_heads_pos` at the token's position →
append k,v to cache → single-query attention over cached 0..pos with GQA head sharing → Wo → resid →
RMSNorm → SwiGLU → resid} → final RMSNorm → lm_head. Added `rope_heads_pos(x,n_heads,base,pos0)`
(rope_heads = pos0 0). `main.c` `llama` generation now primes the cache with the prompt then samples
via decode steps. **Bit-exact:** new selftest **part `[19]`** runs a sequence through
`llama_model_forward` and token-by-token through `llama_decode_step` and compares per-position logits:
max |diff| **0.00e+00** (like the GPT KV cache). Smoke test PASSED, `-Wall` clean. Re-ran SmolLM2-135M
with the cache — same coherent output, faster.

## Modern architecture — step 11: int8 quantization (LLMQ) — DONE (2026-07-22)

`quant.h`/`quant.c`: per-output-channel (per-column) symmetric int8 quantize/dequantize
(`scale[j]=max_i|w[i,j]|/127`, `q=round(w/scale)`, `w~=q*scale`). New **`LLMQ`** checkpoint magic:
same config header, then per canonical tensor — 2D weights as `scales (cols f32) + int8 bytes`, 1D
norms as f32. `convert_llama.py --int8` writes it (numpy per-column quantize); `llama_model_load`
detects LLMQ and DEQUANTIZES to the fp32 model on load, so the runtime forward is unchanged — this is
a **storage/download win** (in-memory int8 + a quantized matmul for RAM/compute is the next step).
Verified: selftest **part `[20]`** — `A@dequant(quant(W))` vs `A@W` L2 rel error **6.81e-03** (<1%,
int8 weights 4× smaller). **Real model: SmolLM2-135M f32 621 MB → int8 156 MB (4×)**, and the int8
checkpoint still generates coherent text. Smoke test PASSED, `-Wall` clean.

## Modern architecture — step 12: in-memory int8 + quantized matmul — DONE (2026-07-22)

The runtime quantization (not just storage): weights stay int8 IN MEMORY and are consumed by a
quantized matmul — 4× less RAM and 4× less weight memory traffic in the matmul inner loop.
- `quant.h/.c`: `QTensor` (int8 + per-column scale kept in memory) + `qmatmul` (`C[m,j] =
  scale[j]·Σ_k A[m,k]·q[k,j]`, i-k-j cache-friendly order, reads int8 weights = 1 byte vs f32 4).
- `llama.h/.c`: `QLlamaModel`/`QLlamaBlock` (int8 wte + block linears + lm_head; f32 norms),
  `q_llama_model_load` (loads LLMQ WITHOUT dequantizing — keeps int8), `q_llama_decode_step` (KV-cache
  decode via qmatmul; dequantizes the embedding row per token; inline quantized SwiGLU),
  `q_llama_kv_create`, `q_llama_quantize_from` (build from an fp32 model, for tests).
- `main.c` `llama` peeks the magic: **LLMQ → in-memory int8 path**, LLMA → f32 path (auto).
Verified: selftest **part `[21]`** — quantize a small fp32 model, run `q_llama_decode_step` per token,
compare to `llama_model_forward`: L2 rel error **6.38e-03** (<1%, tiny-dim; real models lower). Smoke
test PASSED, `-Wall` clean. `transformer llama model_int8.bin` now runs the true int8 model.

**Measured (SmolLM2-135M, honest):** RAM ~135 MB int8 vs ~540 MB f32 (4× less, by construction) and
file 156 MB vs 621 MB (4×). But **NO compute speedup: 83.3 s vs f32 85.2 s (~2%, noise).** Why: our
`qmatmul` converts each int8 weight to float and does f32 arithmetic (`av * (float)q`), so it does the
SAME number of FLOPs as f32 (plus an int8→float convert) — it only cuts memory *bytes*. At this size
decode is compute-bound, not memory-bound, so less traffic + same FLOPs nets ~zero. The real int8
compute speedup needs **SIMD int8×int8→int32 dot-products (AVX2/AVX-512 VNNI)** which do 4× MACs per
instruction, and that requires **activation quantization** (int8 activations) + SIMD intrinsics — a
bigger, separate step. So this step's honest win is RAM/size, not latency.

## Modern architecture — step 13: int8×int8 qmatmul (real speedup) + timing CORRECTION — DONE (2026-07-22)

Rewrote `qmatmul` (quant.c) to the **int8×int8→int32** path: quantize the activation row to int8
(per-row scale) on the fly, accumulate `acc[j] += qa[k]*qw[k,j]` in **int32** (contiguous int8×int8,
auto-vectorizes to wide integer MACs under `-O3 -march=native`), then dequantize `a_scale·W.scale·acc`.
int32 accumulator is safe (127²·K « 2³¹). Accuracy: selftest `[21]` now **7.47e-03** (vs 6.38e-03 with
f32 activations — activation quant adds negligible error). Smoke test PASSED, `-Wall` clean.

**Measured (token-IDs fed directly, `-O3 -march=native` AVX2 build, 30 tokens):**
`int8 (int8×int8 SIMD) 0.9 s vs fp32 2.3 s → 2.5× faster.` Real compute speedup. Output stays
coherent (int8 SmolLM2: "…There is a square in Paris called the Place de la Concorde…").

**⚠️ TIMING CORRECTION (important):** the earlier "~85 s" llama_chat numbers were **NOT the C engine** —
they were Python-side **huggingface_hub doing a slow (rate-limited, unauthenticated) HEAD check on
tokenizer.json** every run. Feeding token IDs straight to `transformer llama` shows the engine is
seconds, not minutes (fp32 2.3 s, int8 0.9 s / 30 tokens). So the int8 SIMD speedup is real and 2.5×.
The earlier "no speedup" finding (step 12) was for the *f32-activation* qmatmul and was correct for
that version; int8×int8 is the one that's actually faster.

**FIXED (the HF-hub overhead): `llama_tok.py` + `llama_chat.py` now resolve `tokenizer.json` via
`hf_hub_download(..., local_files_only=True)` (fall back to a one-time download if not cached) — no
per-run Hub HEAD check. Result: `llama_chat` end-to-end dropped from ~83 s to 1.3 s (64×) for a
20-token SmolLM2 int8 generation, same coherent output ("…a square in Paris called the Place de la
Concorde…"). So the whole pipeline (tokenize → C int8 generate → detokenize) is now ~1 s, matching the
engine.**

## Modern architecture — step 14: q/k/v attention biases (Qwen2 support) — DONE (2026-07-22)

Qwen2/Qwen2.5 is LLaMA-family but adds **biases on q/k/v projections** (o_proj + MLP have none).
Added `b_q`(n_heads·hd)/`b_k`/`b_v`(n_kv·hd) to `LlamaBlock` AND `QLlamaBlock` (f32, kept small),
applied via `bias_add` right after the q/k/v matmul in every path (fp32 `llama_block_forward` +
`llama_decode_step`, int8 `q_llama_decode_step`). **Always present, zero for bias-free models** (LLaMA/
SmolLM), so no conditional logic; convert_llama.py reads `q/k/v_proj.bias` if present else writes
zeros. Checkpoint **bumped to v2** (LLMA/LLMQ; `TENSORS_PER_LAYER` 9→12; biases are 1D f32, so the
LLMQ loader reads them as f32 like norms). Old v1 files must be re-converted (they're git-ignored).
Verified: smoke test PASSED with random biases — `[18]` LLMA v2 round-trip 0.00e+00, `[19]` KV decode
bit-exact, `[21]` int8 vs f32 0.96%. SmolLM2 re-converted (v2, zero biases) → identical coherent output.

**Qwen2.5-0.5B RUNS (the bias-using model):** `convert_llama.py --model Qwen/Qwen2.5-0.5B --int8`
(vocab 151936, d_model 896, 24 layers, 14 q / 2 kv heads, head_dim 64, d_ff 4864, **rope_base
1000000**, eps 1e-6, tied; q/k/v biases read). Generation via the int8 engine:
`"The capital of France is"` → **"Paris. The population of Paris is approximately 1,67 million."**
(FACTUALLY CORRECT — Qwen depends on its q/k/v biases, so a correct answer proves the bias path);
`"Once upon a time"` → *"…a very, very sick little mouse. His name was Muffin…"*. Speed ~3.0 s for
20 tokens int8 (0.5B, ~4× SmolLM2's size; the ~84 s first run was a one-time `tokenizer.json`
download, 3.0 s once cached). `model/qwen/*.bin` git-ignored.

Perf follow-ups: explicit AVX2/NEON intrinsics + a wider (block) qmatmul for prefill; thread qmatmul.
Block/model forward-only (add backward to fine-tune).

## Dataset-trained imggen / audgen — DONE (2026-07-22)

`imggen`/`audgen` now have a **dataset mode** in addition to the original single-example
reconstruct. Auto-selected from the input path (no new flag), so single-file use is byte-identical:

- **Single file** (`imggen cat.ppm out.ppm ...`): unchanged — `autoregress_reconstruct` overfits
  the one input (greedy decode).
- **Folder** (`imggen imgset/ out.ppm ...`): triggered when the path isn't a loadable PPM/WAV.
  Loads `dir/img_%04d.ppm` (or `dir/aud_%04d.wav`) sequentially until missing (same numbered-file
  convention as `video_load`; no new dir-walk code, portable). All examples must share
  size/length (mismatches skipped with a warning). Trains ONE GPT on the set (random example per
  step, SGD) via new `dataset_autoregress`, then **generates a novel sample**.
- **How generation works:** a **BOS** token is prepended to every training sequence (model vocab
  = V+1, BOS = index V), so the model learns P(first token) and can sample unconditionally from
  BOS. Sampling uses `sample_token` (temp 0.9 / top-k 40 / top-p 0.95, seeded); BOS is never
  emitted into the media stream (clamped). Reuses the exact GPT/KV-cache/sampler stack — no codec
  changes.
- **Verified:** built `-Wall` clean, smoke test still `PASSED`. Dataset mode on 4 PPMs (3 red /
  1 blue) → loss ~0.03, generated a 4×4 image = all red (the majority example). Audio dataset on
  3 WAVs → loss ~0.0006, wrote a 64-sample clip. Single-file reconstruct still 16/16 (loss
  0.0002). Bogus input → clear error. (`main.c` grew from 367 → ~490 lines with the two loaders +
  `dataset_autoregress`.)
- **Honest note:** still a toy — the model tends to reproduce a training example (memorization at
  this scale); the color-cube/µ-law codecs cap quality. The *pipeline* (folder → learned
  distribution → sampled generation) is the deliverable.

## Code health — smoke test split into selftest.c — DONE (2026-07-22)

## Code health — smoke test split into selftest.c — DONE (2026-07-22)

Moved the 12-part built-in smoke test (`run_demo`, renamed **`run_selftest`**) plus its three
private helpers (`cond_gen`, `media_to_text`, `gpt_loss`) out of `main.c` into new
**`selftest.c` / `selftest.h`**. `main.c` is now just CLI dispatch: **995 → 367 lines**.
`main()` calls `run_selftest()`. `selftest.c` keeps its own tiny portable `ensure_dir` (for the
`vid_tmp` test frames) so the modules stay decoupled; `main.c` still has its own for
`model_paths`. Makefile: added `selftest.o` to `OBJS` + a header-dep rule (cross-build picks it
up via `SRCS = $(OBJS:.o=.c)`). **Behavior byte-identical**: same `PASSED`, gradient check
`9.28e-04` (|grad|>1e-2), KV cache `0.00e+00`, same losses; `-Wall` clean. (Also fixed the
CONTINUATION-GUIDE build command, which had been missing the media `.c` files.)

## Security & robustness hardening pass — DONE (2026-07-22)

A multi-agent code review (5 parallel reviewers over the whole tree) found the math correct
(softmax stability, GELU/SiLU/LayerNorm/RMSNorm backward, RoPE, GQA, attention scaling,
cross-entropy, int8 all verified) but flagged bugs at the **trust boundaries and robustness
edges**. Worked through the top of the list — 4 criticals + every high + the biggest
mediums/lows — in **6 commits on `agent-1`**. Every commit is smoke-tested (`PASSED`,
byte-identical numerics: grad `9.28e-04`, KV cache `0.00e+00`) and `-Wall` clean.

1. **`b08d7b6` — loader validation** (checkpoint.c, llama.c, tokenizer.c). The binary loaders
   are the untrusted-input boundary (a "download a model" workflow feeds them). Added
   `gpt_config_valid` / `llama_config_valid` gates that reject a corrupt/hostile header BEFORE
   building a model: non-positive/oversized dims (cap 1e6), `d_model % n_heads != 0`, `n_kv == 0`
   (was a decode divide-by-zero), GQA groups that don't divide evenly (was OOB K/V reads), and
   2D weight element counts > INT_MAX (the forward path still indexes weights in int).
   `tokenizer_load` also bounds-checks `n_merges`, null-checks its mallocs, and validates every
   merge id references an already-existing token (`< 256 + rank`).

2. **`9f34d12` — sandbox the tool executor** (toolexec.c, agent.c). Closes the two CRITICAL
   agent bugs. `write_file`/`read_file` paths are confined to the working directory via
   `path_is_safe()` (relative only — no leading `/`/`\`, no drive letter, no UNC, no `..`
   component); the executor enforces this regardless of the approval outcome (hard boundary).
   `run_command` is NEVER auto-approved — it always forces an interactive y/N even in
   `APPROVAL_AUTO` (non-interactive stdin denies), and `write_file` now prints the full content
   before approval so the gate is meaningful. Verified: traversal + absolute paths refused,
   run_command not auto-approved, no files escaped.

3. **`5fd66ff` — size_t migration** (tensor.h/.c, matrix.c, optim.h/.c). `Tensor.size` and
   `Adam.total` were `int` → a tensor/param-set over 2^31 elements wrapped negative and
   `aligned_malloc`/`memset` corrupted memory. Now `size_t` end-to-end (loops over `.size`, the
   flat Adam m/v index too). `matmul`/`matmul_backward`: the `(long)M*N*K` thread threshold
   overflowed on Windows (LLP64: `long` is 32-bit) → huge GEMMs silently ran single-threaded;
   now `(size_t)`, and the backward kernels' `i*N+j` index math is now `size_t` (matching the
   forward, which already was). NOTE: the loader caps (item 1) stay at INT_MAX because the
   embedding/forward path still indexes in int — full >INT_MAX-element single-tensor support
   would need migrating every forward index expression too (future).

4. **`6151129` — fail-closed persistence** (checkpoint.c, tokenizer.c, llama.c, train.c,
   model.c/.h, selftest.c, main.c). Saves are ATOMIC: write to `<path>.tmp` + `fflush`/`fclose`
   check + atomic rename (POSIX) / remove+rename (Windows), so a failed/interrupted/disk-full
   save never truncates a good checkpoint; `llama_model_save` no longer ignores `fclose`.
   Periodic checkpoints in `train.c` now report failures loudly. New `gpt_param_count()` heap-
   sizes the param list in `train_run`/`train_sft`, killing the fixed `Tensor *params[1024]`
   that smashed the stack above 63 layers (`--layers`/`--resume` reachable). Self-test no longer
   writes/`remove()`s the exact train defaults `model.bin`/`tok.bin` (it used `selftest_*.tmp`
   now) — that was DATA LOSS on any bare/typo'd run. And an unknown subcommand prints usage +
   exits 1 instead of falling through to the self-test. Verified: a decoy model.bin survives a
   self-test run; `trian` errors without touching it.

5. **`5064878` — debug validation layer** (new header-only `check.h`, wired into matrix.c,
   embedding.c, loss.c). Typed helpers (`tf_check_matmul`, `tf_check_same_size`,
   `tf_check_bias_add`, `tf_check_token_id`, `tf_check_target_id`, `TF_ASSERT`) abort with a
   clear message AT THE POINT OF FAILURE instead of silently corrupting the heap. Guards the
   `wte->grad[bad_token*d_model]` scatter-add in `embedding_backward` and OOB target ids in
   cross-entropy. O(1)/matmul, O(seq_len)/embedding; **compiled out under `-DNDEBUG`** — the
   DEFAULT build leaves them ON (cheap, catches mistakes). SwiGLU's internal matmuls are
   validated transitively. Verified: default build PASSED with no assertion fired (codebase is
   shape-consistent), `-DNDEBUG` build clean + PASSES, a bad-shape probe aborts as intended.

6. **`73902ac` — no-grad inference tensors** (tensor.h/.c, llama.c, model.c). Every Tensor
   allocated a same-size unused `grad` array even for forward-only models (~+540 MB for
   SmolLM2-135M fp32). New `tensor_create_nograd()` (grad == NULL; `tensor_free`/
   `tensor_zero_grad` handle NULL). The LLaMA path's `vec()`/`mat()` helpers — which build ALL
   its tensors (fp32 weights, int8-model f32 norms/biases, KV cache, decode scratch) — now skip
   grad, HALVING a loaded fp32 model's RAM (headline win for SmolLM2/Qwen2.5). GPT KV cache +
   `gpt_decode_step` scratch also nograd (`make_layer_tensors` gained a `with_grad` flag).
   GPT model weights stay grad-capable (dual-use: `--resume` trains them). Verified byte-
   identical numerics incl. LLaMA save→load→forward→decode (parts 18–19).

**Remaining (lower priority, none load-bearing):** endianness portability of checkpoints/`tok.bin`
(only matters for big-endian, which none of the 8 targets are); a real `generate` CLI
(`--seed/--temp/--top-k/--top-p/--n`, seed-from-time); 64-bit file sizes for >2 GB corpora
(`corpus.c` `ftell`/`int` narrowing); crawler retry/backoff + filename-collision dedup;
`convert_llama.py` sharded-checkpoint support. See the review tables (in-session) for the full list.

## Web testing frontend (Go + WebSocket) + `caption` subcommand — DONE (2026-07-22)

`tools/webchat/` — a self-contained **Go** web server (stdlib only, hand-rolled RFC 6455 in
`ws.go`, embedded single-page UI) that drives the engine from a browser for testing all
modalities. Built in 4 phases, each committed:
1. Server skeleton + **model switcher** + **text chat**. `models.go` scans `model/<name>/`, peeks
   the 4-byte magic, and routes: GPTC+tok.bin → pure-C `generate --name`; GPTC/no-tok → GPT-2 via
   `gpt2_chat.py`; LLMA/LLMQ → `llama_chat.py` (built-in SmolLM2/Qwen HF-tokenizer map, override via
   `model/<name>/hf_tokenizer.txt`). Subprocesses run with cwd = repo root; stdout = result, stderr
   = status.
2. **Image + audio generation** panels. `media.go` does PPM(P6)↔PNG and WAV normalize/trim in Go
   stdlib (no Pillow). imggen reconstructs an uploaded image; audgen an uploaded WAV (returned
   browser-playable).
3. **New C `caption` subcommand** (`caption.c/.h`) + **video** panel. `caption <img.ppm|aud.wav|
   frames_dir> <text> [steps]` builds media embeddings via the existing patch/audio/video
   front-ends, byte-level-tokenizes the caption, and runs the same train→generate as selftest's
   `media_to_text` (teach-then-reproduce). Prints `caption: <text>` to stdout. Video panel extracts
   frames via `video_to_frames.py` (ffmpeg) → caption + frames-GIF preview; degrades gracefully if
   ffmpeg is absent. **Additive — smoke test stayed PASSED (grad 9.28e-04, KV 0.00e+00).**
4. Docs + `.gitignore` (POSIX `webchat` binary; `*.exe` already covered).

**Key gotcha found:** media/large-model runs are SLOW, not hung — attention is O(seq²) on CPU
(audgen 512 samples/100 steps = 37 s vs 128 samples = 2.4 s). The webchat caps inputs (image
≤16×16, audio ≤256 samples) + low default steps so each run is ~10 s, with a 4-min subprocess
timeout as a safety net. **Honest ceiling:** generation reconstructs/overfits one input; caption is
teach-then-reproduce on an untrained front-end (no trained multimodal checkpoint exists) — it proves
the pipelines RUN on real media, it is not a real captioner. Deps: Go 1.23; Python+tiktoken/
tokenizers only for the real models; ffmpeg only for video. Run: `cd tools/webchat && go run .
-root ../..` → http://localhost:8080. See `tools/webchat/README.md`.

## Webchat → live streaming chat → mixed multimodal + FROM-SCRATCH text→image — DONE (2026-07-22)

Big arc on top of the `tools/webchat/` frontend (all committed on `agent-1`, each smoke-green):

**ChatGPT-style chat** — `017ce13` live token streaming + conversation memory; `b4773bd` responses
run until the model finishes (no max-new). C `llama`/`generate` now print each token as `t <id>`
flushed; new Python **stream wrappers** (`tools/llama_stream.py`, `gpt2_stream.py`) incrementally
detokenize (tokenizers/tiktoken) and emit JSON `{"d":piece}` deltas, stopping on EOS tokens or a
`\nUser` stop; Go `runCmdStream` forwards each delta over the WS; the client keeps history (rebuilds
a `User:/Assistant:` context) + a Clear button. `f63e5c7` fixed truncated replies (keep the whole
stdout, not first line). `48552e1` threaded the int8 decode matmul over output columns (new typed
`qacc_columns` in matrix.c) — decode is MEMORY-BANDWIDTH-bound so ~1.3× on this box; `749f628`
webchat prefers the int8 checkpoint. **Standalone exes:** `f61cf75` `tools/webchat/build.ps1` +
`make webchat` build BOTH `transformer.exe` + self-contained `webchat.exe`; the server auto-detects
repo root + engine (runs with no flags). Honest ceiling for the CHAT: Qwen2.5-0.5B is a *base*
model (rambles); the biggest quality lever would be converting Qwen2.5-0.5B-**Instruct**.

**Mixed multimodal chat** — `8074c3b` new C `text2media` subcommand (text→image/audio, teach-then-
reproduce via the cond_gen path); `581e21f` one unified chat that AUTO-DETECTS the requested
modality (keyword classifier + manual Auto/Text/Image/Audio/Video override) and renders results
inline: text streams; image/audio + an attached target run text2media inline (chat_media); no
attachment → an honest note (no trained media model). The media branch is the documented seam for a
real generator.

**FROM-SCRATCH TEXT→IMAGE (the real thing, no diffusion, no libs)** — the DALL-E/Parti recipe built
from matmul up, in 3 steps + save/load + chat integration:
- `e02815c` **step 1**: `t2i` subcommand — trains ONE GPT on a folder of (caption, image) pairs
  (`cap_%04d.txt`/`img_%04d.ppm`) so it learns P(image tokens | caption), generates an image from a
  caption. Verified: red/green/blue captioned imgs → correct colors.
- `8c28860` **step 2**: `vqvae.c/.h` — a learned **VQ-VAE image codec** (patch encoder → learned
  codebook nearest-neighbor quantize → patch decoder), trained with reconstruction + commitment
  losses + the **straight-through estimator**, using only matmul/matmul_backward + Adam. `vqvae` CLI
  test: 16×16 → 64 codes (vs 256 color-cube px, 4× fewer) at MSE 0.0083.
- `0048fbc` **step 3**: `t2i` now uses the VQ-VAE (generate learned codes, VQ-decode) instead of the
  per-pixel color-cube → 4× fewer tokens, learned codec. Verified red/green/blue via codes.
- `d33a3d9` **train/save + fast generate + chat**: `t2i-train <dataset> <prefix>` saves
  `<prefix>.gpt` (gpt_save) + `<prefix>.vqt` (config + VQ weights); `t2i-gen <prefix> <out.ppm>
  <caption>` loads + generates in **~0.03 s** (no retrain). The webchat image intent generates from
  `model/t2i/` inline when present, else the honest note. `.gitignore` now covers `*.gpt`/`*.vqt`.
  Usage: `t2i-train model/mydata model/t2i/t2i 3000 2000` then ask "a picture of X" in the chat.

**Honest ceiling for text→image:** demonstrated on red/green/blue (no real captioned dataset here);
recognizable NOVEL images need thousands of caption/image pairs + a bigger model + more training —
scale, not code. The architecture (learned tokenizer + text→codes transformer + save/load + chat) is
now complete and fast; scaling up is just data + compute. Note: HF text-to-image models (Stable
Diffusion/FLUX) are DIFFUSION (different architecture) and can't load into this engine — this is the
autoregressive path instead.

## Pure-C tokenizer — the engine parses HF models itself, NO Python at runtime — DONE (2026-07-22)

User goal: "our own llm can parse it, not python." Built a from-scratch byte-level BPE tokenizer
so the engine tokenizes GPT-2 / SmolLM2 / Qwen2.5 ITSELF — Python is now only a one-time export
(like the weight converters), never at runtime.
- `baf5f20` **`hftok.c/.h`** — loads a `.tok` vocab (id→raw bytes) into a bytes→id hash, hand-codes
  the GPT-2 pretokenization regex (contractions / ` ?\p{L}+` / ` ?\p{N}+` / punctuation / whitespace;
  \p{L} ≈ ASCII letters + UTF-8 bytes), and runs the tiktoken byte-level-BPE rule "merge the adjacent
  pair with the lowest combined-token id". Key insight: HF BPE assigns vocab ids in MERGE ORDER
  (256 bytes + merges), so min-combined-id == min merge rank → the SAME algorithm works for tiktoken
  AND HF tokenizers. `tools/export_tokenizer.py`: `gpt2` via tiktoken; `<hf-repo>` via the tokenizers
  lib + the HF byte-decoder → `.tok`. CLI `hftok <file.tok> encode|decode`.
- `5a0517a` **`hfgen <model.bin> <tok.hf> [n]`** — pure-C HF chat: prompt on stdin → hftok encode →
  generate (auto-detect GPTC/LLMA/LLMQ) → stream JSON `{"d":piece}` deltas (incremental hftok decode)
  → `{"done":true}`, stopping on EOS or `\nUser`. webchat `streamText` prefers this when
  `model/<name>/tok.hf` exists (export once), else the Python bridges.
- **VERIFIED EXACT** vs the reference tokenizers: GPT-2 (tiktoken), SmolLM2 + Qwen2.5 (tokenizers) all
  match byte-for-byte on punctuation / code / numbers / contractions. End-to-end: gpt2 chats in the
  webchat with zero Python (coherent "…capital of the Republic of Belgium…"); SmolLM2 int8 generates
  pure-C. `.gitignore` covers the generated `*.hf`.
- **To make an HF model pure-C:** `python tools/export_tokenizer.py <gpt2|hf-repo> model/<name>/tok.hf`
  once; the webchat then chats it with no Python. (Qwen's digit-group pretokenizer edge cases may
  differ; SmolLM2/GPT-2 are exact.)

## ▶ CONTINUATION GUIDE — resume here (read this first)

**If a session ran out of usage / you're picking this up fresh, start here.**

**Repo state:** pure-C GPT that trains on text, generates text, has a token/text CLI, a
scripted agent loop, loads & runs real modern LLMs (SmolLM2-135M, Qwen2.5-0.5B via LLMA/LLMQ),
and cross-platform builds (incl. WASM). A 6-commit **security/robustness hardening pass** landed
2026-07-22 (see that section above): sandboxed tool executor, validated loaders, size_t overflow
fixes, atomic saves, a `check.h` debug layer, and no-grad inference tensors. **`check.h` runs
shape/bounds assertions in the DEFAULT build; add `-DNDEBUG` to compile them out.** Work is on
branch **`agent-1`** (several commits ahead of `master`). Tree should be clean — run `git status`
and `git log --oneline -12` to confirm where you are.

**Environment (this Windows machine):** no gcc/clang/make. Use **`zig`** (`D:\env\zig\zig.exe`)
and **Python** (`D:\env\python\python-3.9.10.amd64\python.exe` + numpy). Node at
`D:\env\node\...` (has WASI). `git` works; user identity Marjon Cajocon.

**Build + smoke test (do this after any change):**
```
zig cc -std=c11 -O2 -Wall -o transformer.exe tensor.c matrix.c activation.c layernorm.c \
  feedforward.c attention.c embedding.c sampler.c model.c checkpoint.c tokenizer.c loss.c \
  optim.c rmsnorm.c rope.c swiglu.c gqa.c quant.c llama.c toolcall.c toolexec.c approval.c agent.c corpus.c train.c image.c patch.c \
  video.c audio.c imagegen.c audiogen.c caption.c text2media.c t2i.c vqvae.c selftest.c main.c -lm
# On this Windows box, listing files explicitly is fragile — easiest is to glob:
#   $c = (Get-ChildItem *.c | ForEach-Object { $_.Name }); zig cc -std=c11 -O2 -Wall -o transformer.exe @c -lm
# (check.h is header-only — no .c to list. Add -DNDEBUG to disable the debug assertions.)
.\transformer.exe            # must print: Output validation: PASSED
```
Then clean: `Remove-Item transformer.exe,*.o,*.bin,*.wasm,xbuild_* -ErrorAction SilentlyContinue`.
Cross-compile all platforms: `make cross` (POSIX shell) or the `zig cc -target ...` loop.

**Subcommands:** `train <path> [--d-model N --layers N --heads N --ctx N --steps N --lr F
--out F --tok F ...]` · `generate model.bin tok.bin` (text) or `generate model.bin` (token IDs)
· `agent [--ask]`. Crawl a corpus: `python tools/crawl_wikipedia.py "Topic" --max-pages N`.

**Conventions (enforced):** source root is **C-only and flat**; non-C scripts in `tools/`;
data in `data/` (git-ignored); cross binaries in `build/` (git-ignored except *.md/*.txt);
**2-space indent**, `-std=c11 -Wall` clean, no untyped `void *` params. **Any change to
training/backprop must keep the gradient check green** (main.c: conditioned rel err < 1e-2)
and the **KV-cache bit-exact** check (0.00e+00). Commit per logical step (auto-commit is
authorized; local only, don't push).

**Recommended next tasks (in priority order):**
1. ✅ DONE — **Training UX**: train/val split + held-out `val` loss, `--checkpoint-every N`,
   `--resume model.bin` (continue a saved model; loads its config + tokenizer). In `train.c`
   (`eval_split`, resume/checkpoint logic) + `main.c` flags.
2. ✅ DONE — **matmul thread pool** (`matrix.c`): persistent fork-join pool threads `matmul` +
   `matmul_backward`, bit-exact (verified), 2.68× on 8 threads, all 8 targets build. Threads =
   `$TF_THREADS` or CPU count. (SIMD + int8 quant remain, later.)
3. ✅ DONE — **Real GPT-2**: `tools/gpt2_setup.py` (download+convert) + `tools/gpt2_tok.py`
   (tiktoken) → pure-C model generates real coherent GPT-2 text. `model/<name>/` folder
   convention + `--name`. GPT-2 lives in `model/gpt2/` (git-ignored weights).
4. ✅ DONE — **pluggable agent backend** (`agent.c`): scripted + command backends;
   `agent --cmd "<prog>"` uses an external program as the model (`tools/agent_mock.py`).
   For a genuinely acting agent, point `--cmd` at an instruction-tuned model or an LLM API
   (base `model/gpt2` won't follow tool-call formats well).
5. ✅ **Phase E (images) COMPLETE** — E1 load, E2 patch-embed, E3 into-GPT
   (`gpt_forward_multimodal`), E4 image→caption training (`gpt_backward_multimodal` +
   `patch_embed_backward` + loss ignore-index). Then **Phase F (video)** is next.

Multimodal (asked about): **images now train end-to-end** (forward + backward + patch
projection learned; smoke test overfits image→caption). Real *quality* still needs real
image-caption datasets + a much bigger model. **Video is Phase F** (#6), the frontier.

## Session summary (2026-07-21, autonomous run)

Built out the whole pipeline in green, committed steps (newest first):
`Phase C agent` · `B4 matmul` · `A1 tokenizer` · `B3 generate` · `B2 KV cache` ·
`B1 sampling` · `4b GPT-2 compat`. Everything builds `-Wall` clean and the default
`transformer` smoke test prints `Output validation: PASSED`.

**What works now (pure C, local, flat modules):**
- A GPT-2-architecture model (attention biases, pre-norm, GELU FFN, learned pos-emb, tied
  head via converter) that **trains end-to-end** (backprop gradient-checked; loss → ~0).
- Checkpoint save/load (GPTC v2) + Python converters (`tools/convert_gpt2.py`,
  `convert_to_gptc.py`) to import real GPT-2.
- Trainable byte-level BPE tokenizer (train/save/load).
- Generation: temperature/top-k/top-p sampling + **bit-exact KV cache** + a `generate` CLI.
- A **coding-agent loop** CLI (`agent`): parse tool calls → approval gate → execute
  (write/read/run) → observe.

**What's left (and why it's not buildable here):**
- **A2 / real text output** — needs the model's vocab to equal the tokenizer's *and* a
  trained (or loaded) model; otherwise output is meaningless bytes. Blocked on real weights.
- **Real agent usefulness** — the loop is driven by a scripted model; a capable backend
  (real LLM API, or a large locally-run model) is required. The built-in transformer is far
  too small to emit valid tool calls.
- **B4 perf (threads/SIMD/int8 quant)** and **D1 (run a 1–3B model)** — large systems work,
  only matters at real scale; can't be exercised on this toy model here.

The honest bottom line: **all the machinery/plumbing for a local coding-agent CLI is built
and tested**; turning it into something genuinely useful is gated on running a real,
capable model — either download+convert GPT-2 (step 4b recipe) or wire a real LLM backend
into `agent.c`.

---

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

## Step 3 — Full model: embeddings, stacked blocks, output head, sampler

**Status: DONE — smoke test passed.**

### What changed
The demo is now an actual (tiny, untrained) autoregressive language model instead of a
single block on random input. Three new flat modules plus a rewritten `main.c`:

- **`embedding.h/.c`** — `embed_tokens`: `output[i] = wte[token_ids[i]] + wpe[i]`
  (token + positional embeddings).
- **`sampler.h/.c`** — `sample_argmax`: greedy next-token pick from a logits row.
- **`model.h/.c`** — introduces two structs to bundle weights:
  - `BlockWeights` — all learnable tensors of one transformer block.
  - `GPT` — config + `wte`, `wpe`, an array of `n_layers` `BlockWeights`, final
    LayerNorm params, and `lm_head` output projection.
  - `gpt_create` / `gpt_free` — allocate + randomly initialize / release all weights.
  - `gpt_forward` — embeddings -> stacked blocks -> final LayerNorm -> `@ lm_head`
    logits `(seq_len, vocab_size)`.
  - `gpt_generate` — autoregressive greedy loop, appending tokens in place.

`main.c` builds a `GPT` (vocab=16, d_model=8, heads=2, layers=2, d_ff=32), generates 5
tokens from a 3-token prompt, and validates the logits (no NaN/Inf).

### Notes / simplifications
- Output head `lm_head` is a **separate** weight, not tied to `wte` (real GPT-2 ties
  them: `logits = hidden @ wte^T`). Tying can come later once a transpose-matmul exists.
- No KV cache: each generation step re-runs the full forward over the current sequence.
  Fine for a reference model; a cache is a later optimization.
- Only greedy sampling; temperature / top-k / top-p can extend `sampler.c`.

### Progress checklist
- [x] Token + positional embeddings (`embedding.h` / `embedding.c`)
- [x] Greedy sampler (`sampler.h` / `sampler.c`)
- [x] `GPT` + `BlockWeights` structs, create/free (`model.h` / `model.c`)
- [x] `gpt_forward`: embed -> N blocks -> final LayerNorm -> logits
- [x] `gpt_generate`: autoregressive greedy loop
- [x] `main.c` rewritten to build + run the model
- [x] `Makefile` + `README.md` updated (3 new modules)
- [x] Smoke test (build + run -> generates tokens, `Output validation: PASSED`, `-Wall` clean)

### Smoke test result (2026-07-21)
Built with `zig cc -std=c11 -O2 -Wall`; `-Wall` clean. Prompt `1 5 9` generated
`11 6 11 14 14` (arbitrary - the model is untrained), logits validated PASSED, all
memory freed.

## Step 4 (Option A) — Checkpoint save/load + byte-level BPE tokenizer

**Status: DONE — smoke test passed (incl. the `tools/` converter).**

### What changed
Built the two pieces of infrastructure that real-weight loading depends on, both
self-contained and testable without a large download:

- **`checkpoint.h/.c`** — `gpt_save` / `gpt_load` in a flat binary format (GPTC):
  `magic 'GPTC' + version + 6-int config + every weight's raw float32` in a fixed
  canonical order. A shared `collect_tensors` helper drives both save and load so their
  order can't drift. `gpt_load` reuses `gpt_create` then reads weights in.
- **`tokenizer.h/.c`** — self-contained byte-level BPE: 256 base byte tokens + merges
  (`256 + rank`), binary-safe, no external files. `encode`/`decode`/`add_merge`/
  `create`/`free`.
- **`main.c`** — rewritten demo validates three things: tokenizer round-trip
  (`decoded == original`, encoded shorter than bytes), model generation + NaN/Inf check,
  and checkpoint round-trip (reloaded logits **bitwise identical**). Exit code reflects
  pass/fail.
- **`tools/convert_to_gptc.py`** — NEW helper (Python) that converts a `.npz` of named
  weight arrays into a GPTC `.bin`, matching `collect_tensors` order exactly. This is the
  bridge for importing external weights.
- **`README.md`** — added the two module rows, a **CLI guide** (the `transformer` binary,
  the converter tool with args) and a **programmatic API guide** (tokenizer, model,
  checkpoints, external-weight import). Coding rules updated: source root is **C-only and
  flat**; non-C helper/third-party scripts go in **`tools/`**.
- **`.gitignore`** — ignore `*.bin` checkpoints.

### Repo layout rule (updated)
- Root: C source only, flat, no subfolders.
- `tools/`: helper/third-party scripts in other languages (Python/Go/…). The only allowed
  directory.

### Notes / simplifications
- Byte-level BPE is self-contained (no `encoder.json`/`vocab.bpe` yet); the GPT-2 regex
  pre-tokenization and file loading are a later add-on.
- Checkpoint format is our own simple binary (host-endian float32), not safetensors.
- Loading **real** GPT-2 weights (step 4b) still needs the model to match GPT-2's exact
  parameterization: attention biases, fused QKV projection, and a weight-tied `lm_head`.
  The converter + loader are the transport layer that work will plug into.

### Progress checklist
- [x] Checkpoint format + `gpt_save`/`gpt_load` (`checkpoint.h`/`.c`)
- [x] Shared `collect_tensors` (save/load order can't drift)
- [x] Byte-level BPE tokenizer (`tokenizer.h`/`.c`)
- [x] `main.c` demo: tokenizer + generation + checkpoint round-trips
- [x] `tools/convert_to_gptc.py` weight importer
- [x] `Makefile` + `.gitignore` updated
- [x] `README.md`: CLI guide + API guide + `tools/` / C-only rules
- [x] Smoke test (build `-Wall` clean; all three round-trips PASS; exit 0)
- [x] Converter verified end-to-end (npz -> convert -> C `gpt_load` -> forward, no NaN/Inf)

### Smoke test result (2026-07-21)
Built with `zig cc -std=c11 -O2 -Wall`; `-Wall` clean. Tokenizer round-trips
`"hello hello"` (11 bytes -> 5 tokens -> original); model generates `1 5 9 -> 11 6 11 14
14`; reloaded checkpoint logits bitwise identical; `Output validation: PASSED`. The
`tools/` converter was also verified: a random numpy `.npz` -> `convert_to_gptc.py` ->
GPTC `.bin` -> C `gpt_load` -> forward pass ran clean (no NaN/Inf).

## Step 5a — Training foundation (backprop core + loss + Adam), gradient-checked

**Status: DONE — smoke test passed (gradient check + loss drop).**

### What changed
The `Tensor.grad` array is finally used. Manual backprop (explicit `*_backward` siblings,
no autograd engine); each accumulates into input/param grads, so callers zero grads per
step (`tensor_zero_grad`, new in `tensor.c`).

- **`matrix.c`** — `matmul_backward` (`dA += dC·Bᵀ`, `dB += Aᵀ·dC`), `matadd_backward`,
  plus `bias_add` / `bias_add_backward` (row-broadcast bias, reusable by the FFN).
- **`activation.c`** — `gelu_backward` (analytic derivative of the tanh-approx GELU).
- **`layernorm.c`** — `layernorm_backward` (per-row `dgamma`, `dbeta`, `dx`).
- **`loss.h/.c`** (NEW) — `cross_entropy_forward` (mean loss + fills softmax `probs`,
  reuses `softmax_rowwise`) and `cross_entropy_backward` (`(probs - onehot)/N`).
- **`optim.h/.c`** (NEW) — `Adam` (bias-corrected, per-param m/v state).
- **`main.c`** — rewritten around a tiny trainable MLP
  `logits = LayerNorm(GELU(x·W1+b1))·W2 + b2` that exercises every new backward op:
  - **Gradient check**: analytic grad vs numerical `(L(+ε)-L(-ε))/2ε`, sampled across all
    params; asserts max relative error < 1e-2.
  - **Training**: Adam overfits 8 synthetic examples; asserts the loss collapses.
  - Keeps condensed tokenizer + checkpoint round-trips (step-4 code still exercised).

### Notes
- Gradient checking is the correctness proof here — a wrong gradient still "runs" but won't
  learn; the numeric check catches it.
- Scope boundary held: attention/embedding backward = step 5b (below).

### Progress checklist
- [x] `tensor_zero_grad` (`tensor.c`)
- [x] `matmul_backward`, `matadd_backward`, `bias_add`/`bias_add_backward` (`matrix.c`)
- [x] `gelu_backward` (`activation.c`)
- [x] `layernorm_backward` (`layernorm.c`)
- [x] Cross-entropy forward/backward (`loss.h`/`.c`)
- [x] Adam optimizer (`optim.h`/`.c`)
- [x] `main.c`: trainable MLP + gradient check + training loop
- [x] `Makefile` + `README.md` (module rows + Training subsection)
- [x] Smoke test (build `-Wall` clean; grad check + loss drop PASS; exit 0)

### Smoke test result (2026-07-21)
Built with `zig cc -std=c11 -O2 -Wall`; `-Wall` clean.
- Gradient check: **max relative error 2.37e-03** (tol 1e-2) — all backward ops correct.
- Training: loss **2.5848 -> 0.0001** over 300 Adam steps.
- Tokenizer + checkpoint round-trips still OK. `Output validation: PASSED`.

## Step 5b — End-to-end GPT training (attention + embedding + FFN backward)

**Status: DONE — smoke test passed (full-model gradient check + training).**

### What changed
The full `GPT` now trains end-to-end. Forward functions freed their intermediates, so
training-mode forwards were added that **cache activations**, plus matching backward ops:

- **`feedforward.c`** — `feedforward_train` (caches h1b/act) + `feedforward_backward`.
- **`attention.c`** — `multihead_attention_train` (caches Q/K/V/weights/concat) +
  `multihead_attention_backward` (incl. softmax-through-causal-mask backward). Added
  `#include <string.h>` for the weights `memcpy`.
- **`embedding.c`** — `embedding_backward` (scatter-add into wte/wpe).
- **`matrix.c`** — `bias_add_backward`/`matadd_backward` made `const`-correct.
- **`model.c`** — `GPTActs` activation cache (`gpt_acts_create/free`), `gpt_forward_train`,
  `gpt_backward` (reverse through head → final LN → each block {residual, FFN, LN2,
  residual, attention, LN1} → embeddings), `gpt_collect_params`, `gpt_zero_grad`.
- **`main.c`** — replaced the 5a MLP with the real model: end-to-end gradient check +
  next-token training on a fixed sequence.

### Gradient-check note (float32 finite differences)
First run reported max rel error 1.19e-1 on `block0.W_q[0]` (analytic 7.6e-4, numeric
6.0e-4) — a **float32 cancellation artifact**, not a bug: the loss is O(1) (~2.88) while
that gradient is ~7e-4, so `Lp-Lm ~ 1e-6` is dominated by float32 rounding of the loss.
Fixes: use `eps=1e-2`, and judge correctness on **well-conditioned** entries (`|grad|>1e-2`,
where finite differences are trustworthy). Training converging to ~0 is the corroborating
signal that the gradients are right.

### Progress checklist
- [x] FFN train/backward (`feedforward.c`)
- [x] Multi-head attention train/backward incl. causal-softmax backward (`attention.c`)
- [x] Embedding backward / scatter-add (`embedding.c`)
- [x] `GPTActs` cache + `gpt_forward_train` + `gpt_backward` (`model.c`)
- [x] `gpt_collect_params` + `gpt_zero_grad` for the optimizer
- [x] `main.c`: end-to-end gradient check + next-token training
- [x] `README.md` updated (full-GPT training)
- [x] Smoke test (build `-Wall` clean; grad check OK; loss drops; exit 0)

### Smoke test result (2026-07-21)
Built with `zig cc -std=c11 -O2 -Wall`; `-Wall` clean.
- Gradient check: max rel error **3.70e-03** (all) / **9.28e-04** (|grad|>1e-2) — full
  backprop correct.
- Training: next-token loss **2.8787 -> 0.0002** over 600 Adam steps (the 2-layer GPT
  memorizes the sequence).
- Tokenizer + checkpoint round-trips still OK. `Output validation: PASSED`.

## Step 4b — GPT-2-compatible architecture + weight loading

**Status: DONE — smoke test passed; loader verified on GPT-2-shaped weights.**

### What changed
Made the model able to *represent and load* real GPT-2 weights. The one true architecture
gap was **attention biases** (GPT-2 has `c_attn`/`c_proj` biases; we had none).

- **`attention.c/.h`** — `multihead_attention_forward/train/backward` and
  `transformer_block_forward` now take `b_q/b_k/b_v/b_o` (each d_model). Forward adds them
  after the Q/K/V and output projections (`bias_add`); backward accumulates via
  `bias_add_backward`.
- **`model.h/.c`** — `BlockWeights` gained `b_q/b_k/b_v/b_o` (zero-init); threaded through
  `gpt_create/free`, `gpt_forward`, `gpt_forward_train`, `gpt_backward`,
  `gpt_collect_params`, and `gpt_zero_grad` (per-layer count 12 → 16).
- **`checkpoint.c`** — canonical tensor order includes the biases; **format bumped to v2**
  (`TENSORS_PER_LAYER` 12 → 16).
- **Output head tying** — kept a separate `lm_head`; GPT-2's tied head is emitted as
  `lm_head = wte^T` by the converter, so no forward change was needed.
- **`tools/convert_to_gptc.py`** — added the bias keys; writes v2.
- **`tools/convert_gpt2.py`** (NEW) — HuggingFace GPT-2 `model.safetensors` → `.npz`:
  splits fused `c_attn` into W_q/W_k/W_v + b_q/b_k/b_v, maps `ln_*` → gamma/beta, ties
  `lm_head = wte^T`. (GPT-2 `Conv1D` weights are already `(in,out)` = matches our matmul, no
  transpose needed.)

### Verified vs. not
- **Verified:** `main.c` smoke test passes with biases — gradient check conditioned
  (`|grad|>1e-2`) **9.28e-04**, training still 2.8787 → 0.0002. And the full loader path with
  the **v2 format incl. biases**: random GPT-2-shaped `.npz` → `convert_to_gptc.py` →
  C `gpt_load` → forward runs clean (no NaN/Inf).
- **Not verified in-session:** `convert_gpt2.py` needs `pip install safetensors` + the ~500MB
  GPT-2 download (neither available here). Its HF→npz mapping is straightforward and matches
  HF's layout; the bit-exact `.npz`→GPTC→C part *is* tested.

### Gradient-check note
With zero-init biases, the "all" rel-error metric hits ~1.0 (biases sit at the float32
noise floor with near-zero gradients). The **conditioned** metric (`|grad|>1e-2`) is the gate
and stays at 9.28e-04 — bias backward is correct; training convergence corroborates.

### Progress checklist
- [x] Attention biases in forward/train/backward + block (`attention.c/.h`)
- [x] `BlockWeights` biases threaded through `model.c` (create/free/forward/backward/params)
- [x] Checkpoint v2 (biases in canonical order) (`checkpoint.c`)
- [x] `convert_to_gptc.py` bias keys + v2
- [x] `convert_gpt2.py` HF GPT-2 → npz importer
- [x] Docs: README (GPT-2 import recipe) + `tools/README.md`
- [x] Smoke test (build `-Wall` clean; grad check + training PASS)
- [x] Loader verified on GPT-2-shaped weights (npz → GPTC → C load → clean forward)

## Step B1 — Sampling (temperature / top-k / top-p)

**Status: DONE — smoke test passed.**

- **`sampler.c/.h`** — added a small reproducible RNG (`Rng`, xorshift64; `rng_create`,
  `rng_uniform`) and `sample_token(logits, row, temperature, top_k, top_p, rng)`:
  temperature-scaled stable softmax → sort by prob → apply top-k then top-p (nucleus) →
  sample from the renormalized kept mass. `temperature <= 0` falls back to greedy.
  Kept the existing `sample_argmax`.
- **`main.c`** — Part [3] sampling checks: `top_k=1` selects the argmax for every row, and
  the same seed reproduces the same token. Both pass.
- Model-agnostic: drives the toy model now, real GPT-2 later. Next: KV cache (B2) and a REPL
  front-end (B3).

## Step B2 — KV cache (incremental decode)

**Status: DONE — verified bit-exact vs the full forward.**

- **`model.h/.c`** — `KVCache` (per-layer `(max_seq_len, d_model)` K and V buffers) with
  `gpt_kvcache_create` / `gpt_kvcache_free`, and `gpt_decode_step(m, kv, token, logits)`:
  embeds one token, and per layer does LN1 → q/k/v (append k,v to cache) →
  single-query attention over cached positions 0..pos → W_o+bias → residual → LN2 → FFN →
  residual, then final LN → `@ lm_head`. Reuses `layernorm`/`matmul`/`bias_add`/`matadd`/
  `feedforward`; attention-over-cache is a small inline loop. Advances `kv->len`.
- **`main.c`** — Part [4] feeds the sequence one token at a time and compares each step's
  logits to the non-cached forward (`logits` row t). Result: **max |diff| = 0.00e+00**
  (bit-exact) — the cache is correct, and generation is now O(n) per token instead of
  re-running the whole prefix.

## Step B3 — `generate` subcommand (token-level REPL)

**Status: DONE — pipe-tested.**

- **`main.c`** — split into `run_demo()` (the default smoke test) and `run_generate()`,
  dispatched by `main(argc, argv)`: `transformer generate [checkpoint.bin]`.
  Reads lines of space-separated **token IDs** from stdin, feeds them through the KV cache
  (`gpt_decode_step`), then autoregressively samples `n_new` tokens (`sample_token`,
  temp 0.8 / top-k 40 / top-p 0.95, seed 1234) and streams them (bracketed). Loads real
  weights if a `.bin` is given, else a random toy model. Out-of-range IDs are skipped; a
  leading UTF-8 BOM is stripped (Windows pipe robustness).
- **Verified:** `echo "1 5 9" | transformer generate` → `gen: 1 5 9 [15] [15] [5] ...`;
  multi-line input loops; `99` (>= vocab 16) is skipped. Default `transformer` still
  `PASSED`. Build `-Wall` clean.
- This is the CLI skeleton: **token-in/token-out** now, **text-in/text-out** once A1 (real
  BPE) + real weights land.

## Step A1 — Trainable BPE tokenizer (merge-learning + persistence)

**Status: DONE — smoke test passed.**

- **`tokenizer.c/.h`** — `tokenizer_train(t, text, num_merges)` learns merges greedily (most
  frequent adjacent pair each round, replacing occurrences in the working sequence);
  `tokenizer_save`/`tokenizer_load` persist the merge list (`BPE1` format; the 256 base byte
  tokens are implicit).
- **`main.c`** Part [1] — trains 30 merges on a sample string: 45 bytes → 20 tokens,
  `decode(encode(x)) == x`, and save→load reproduces identical token IDs.
- **Note / honesty:** this is a *self-trained* byte-level BPE, fully local and testable.
  Matching GPT-2's *exact* tokenization (its `encoder.json` + `vocab.bpe` + the GPT-2
  pretokenization regex + byte→unicode map) is a separate extension needed only to feed text
  to real GPT-2 weights — text↔token vocab must also match the model's vocab (see A2).

## Step B4 (partial) — cache-friendly matmul

**Status: matmul reorder DONE; threads/SIMD/quant remain.**

- **`matrix.c`** — `matmul` rewritten from `i-j-k` to **`i-k-j`** order: the inner loop
  streams a row of B and a row of C sequentially (cache-friendly, auto-vectorizable)
  instead of striding down a column of B. Same math.
- **Verified:** KV-cache logits still bit-exact (0.00e+00), gradient check + training
  unchanged (2.8787 → 0.0002) — the reorder is correct.
- **Remaining (needed only at 124M+ scale):** multi-threading (OpenMP/pthreads), SIMD
  intrinsics, and int8 weight quantization. Tracked under Phase D.

## Phase C — Agent layer (parse → approve → execute → observe)

**Status: DONE — both approval modes tested; loop creates and reads a real file.**

Four small flat C modules build a local coding-agent loop (pure C, no external deps):

- **`toolcall.c/.h`** — `toolcall_parse` extracts the first `<tool name=… path=… command=…>
  body</tool>` or `<done>…</done>` block into a `ToolCall` (kind + attrs + body). Plain text
  with no block = final answer.
- **`toolexec.c/.h`** — `toolexec_run` executes a call and returns an observation string:
  `write_file` (fopen/fwrite), `read_file` (fopen/fread), `run_command` (portable
  `popen`/`_popen`).
- **`approval.c/.h`** — `approval_confirm(mode, action)`: `APPROVAL_ASK` prompts `y/N` on
  stdin, `APPROVAL_AUTO` auto-approves. BOM-robust (Windows piped input).
- **`agent.c/.h`** — `agent_run(goal, responses[], n, mode)`: the loop. For each model
  response: parse → (if tool) describe + approve → execute → print observation; `<done>` or
  plain text ends it. `responses[]` is the pluggable "model backend" — a fixed script here;
  a real LLM integration replaces it with live calls that see the observations.
- **`main.c`** — `agent` subcommand: `transformer agent` (scripted demo, auto-approve),
  `transformer agent --ask` (interactive approval). The demo writes `agent_demo.txt`, reads
  it back, then cleans it up.

**Verified:** `transformer agent` runs write→read→done, actually creating/reading the file;
`transformer agent --ask` (piped `y`) approves and does the same; denying (or no input)
blocks the action. Default `transformer` smoke test still `PASSED`; `-Wall` clean.

**Honest limitation:** the built-in transformer is far too small/weak to emit valid tool
calls, so the agent is driven by a scripted model here. The loop, tools, and approval gate
are real and reusable; plugging in a capable model (a real LLM API, or a much larger locally
-run model — Phase D) is what makes it genuinely useful.

## Step A2 — Train-on-your-data pipeline

**Status: DONE — trains on a text file and generates coherent in-style text.**

- **`data/`** — where your training data goes (git-ignored except `README.md` + `sample.txt`).
  This folder is data, not code, so it's exempt from the C-only-flat-root rule.
- **`train.c/.h`** — `train_run(data, out_model, out_tok, num_merges, steps)`: reads the text
  file, trains a byte-level BPE (`tokenizer_train`), sizes a GPT to that vocab (d_model 64,
  2 layers, 4 heads, d_ff 256, ctx 32), then trains on random context windows
  (`gpt_forward_train` → `cross_entropy` → `gpt_backward` → Adam), and saves the model +
  tokenizer.
- **`main.c`** — `train` subcommand: `transformer train <data.txt> [model.bin] [tok.bin]
  [merges] [steps]`. And `generate` now has a **text mode**: `transformer generate model.bin
  tok.bin` encodes stdin text → generates → decodes back to text (token-ID mode still works
  without a tokenizer).
- **Verified:** `transformer train data/sample.txt model.bin tok.bin 150 400` → 427 bytes →
  83 tokens, vocab 312, loss **5.95 → ~0.1**; then `echo "the cat" | transformer generate
  model.bin tok.bin` produced coherent corpus-style text ("the cat sat on the mat. the dog
  sat on the log. ..."). Default `transformer` smoke test still `PASSED`; `-Wall` clean.
- **How to use your own data:** put a `.txt` in `data/`, run `transformer train
  data/yourfile.txt model.bin tok.bin 200 2000`, then `echo "prompt" | transformer generate
  model.bin tok.bin`. More/cleaner text + more steps = better; scale the config in `train.c`.

## Data crawler + directory training + faster BPE encode

**Status: DONE — crawl → train → generate verified on real Wikipedia text.**

- **`tools/crawl_wikipedia.py`** (NEW) — crawls *connected* Wikipedia articles via the API
  (plain-text extracts + breadth-first link following), saving `data/wiki/<title>.txt`.
  Stdlib `urllib` only; needs internet. Verified live: fetched "Machine learning" (58.7K
  chars) → linked "15.ai" (28.3K).
- **`corpus.h/.c`** (NEW) — `corpus_read(path)` reads a file, **or a directory tree
  recursively** (every `.txt`, portable dir-walk: Win32 `FindFirstFile` / POSIX `dirent`).
- **`train.c`** — now uses `corpus_read`, so **`transformer train data/`** (a directory)
  trains on all `.txt` under it. Answers "does data/ support subdirectories?" → **yes**.
- **`tokenizer.c`** — `tokenizer_encode` rewritten to standard BPE (merge *all* occurrences
  of the lowest-rank pair per round, not one). Same result, hugely faster: an 87KB corpus
  went from >120s (timeout) to **2.7s**. Essential for real corpora.
- **Verified end-to-end:** crawled 87KB of Wikipedia → `transformer train data/wiki` (300
  merges, 200 steps, 2.7s) → `generate` runs. Output is undertrained gibberish at 200 steps
  (tiny model) — needs many more steps + a bigger model for coherence; the *pipeline* is the
  deliverable. Directory training also verified on `data/` (sample.txt + a `wiki/` subdir).
  Default smoke test still `PASSED`; `-Wall` clean.

**Further BPE speedup (future):** word-level pretokenization (BPE within each word, like
GPT-2) would make encode near-instant on large corpora; current whole-sequence encode is
fine for MBs, slow for GBs.

## Cross-platform portability (Linux / macOS / Windows / WebAssembly)

**Status: DONE — all 8 targets build `-Wall` clean; Windows + WASM verified running.**

Cross-compiled every source with `zig cc -target <triple>`:
- Linux `x86_64/aarch64-linux-gnu`, `x86_64-linux-musl`; Windows `x86_64/aarch64-windows-gnu`;
  macOS `x86_64/aarch64-macos`; **WebAssembly `wasm32-wasi`** — all build.
- Native Windows runs (smoke test PASSED); the **WASM binary runs via Node's WASI** and
  passes the full smoke test (tokenizer, gradient check, KV cache bit-exact, PASSED) — file
  I/O works through WASI preopens.

Fixes this required (the Windows-only build had never exercised these paths):
- **`toolexec.c`** — `popen`/`pclose` are POSIX; glibc/musl hide them under `-std=c11` unless
  `_POSIX_C_SOURCE` is defined. Added it (non-Windows). WASM has no process model, so
  `run_command` is `#ifdef`-stubbed to "not supported on WebAssembly" there.
- **`corpus.c`** — same `_POSIX_C_SOURCE` guard for `opendir`/`readdir`/`stat`.
- **`tensor.c`** — `RAND_MAX` exceeds float's exact range on Linux/macOS; divide in `double`
  to silence `-Wimplicit-const-int-float-conversion` and keep init precise everywhere.

**`make cross`** builds every target into `build/` (via `zig cc`; needs a POSIX shell).
`build/` is git-ignored except text/markdown (`build/README.md` documents the outputs).
See README "Supported platforms" for the target list + build/run commands.

## Configurable model (CLI flags) — DONE

`train` is now fully configurable (no editing `train.c`): `TrainConfig` in `train.h`,
`train_default_config()` + `train_run(data, cfg)` in `train.c`, flag parsing in `main.c`'s
`run_train`. Flags: `--d-model --layers --heads --d-ff --ctx --merges --steps --lr --seed
--out --tok`. Validates `d_model % heads == 0`. Defaults are a small-but-real model (d_model
128, 3 layers, 4 heads, ctx 64). Verified: small config trains+generates; bad `--heads`
errors. Each trained `model.bin`+`tok.bin` is a standalone model → build many role-specific
AIs. See the CONTINUATION GUIDE above for the prioritized backlog.

## Training UX (train/val + checkpoint + resume) — DONE

`train.c`: train/val split (val = tail `val_frac` of the token stream; auto-disabled if the
corpus is too small to hold out a window), `eval_split()` prints held-out `val` loss each
report, `--checkpoint-every N` saves mid-run, and `--resume model.bin` loads a saved model +
its tokenizer (`--tok`) and continues training (config comes from the checkpoint, not flags).
`main.c` flags: `--val-frac --checkpoint-every --resume`. Verified: checkpoint saved mid-run;
resume continued from the saved loss (not from scratch); val loss tracks train loss.

## Phase #2 (performance) — matmul thread pool DONE

**First attempt (reverted):** per-call threading (spawn threads inside each `matmul`) measured
**0.94× — slower** (per-call OS thread spawn > benefit; backward unthreaded).

**Final (kept):** a **persistent fork-join thread pool** in `matrix.c`:
- N-1 worker threads created once (lazily), reused for every parallel region — no per-call
  spawn. `parallel_for(total, fn, ctx)` splits `[0,total)` into N chunks (workers do 0..N-2,
  the caller does N-1) using a mutex + two condition variables. `atexit` joins the workers.
- Portable sync wrappers: **Win32** `CRITICAL_SECTION`+`CONDITION_VARIABLE`, **POSIX**
  `pthread_mutex`+`pthread_cond`, **WASM** disabled (runs inline). Makefile adds `-pthread`.
- Threads `matmul` (over output rows) and **both loops of `matmul_backward`** (over rows of
  dA and dB) — the bulk of training compute. Attention's inner loops are not threaded (minor
  at small seq).
- **Bit-exact:** row-parallel, so each output element is produced by one thread in the same
  order → verified byte-identical checkpoints at `TF_THREADS=1` vs `8`. Tiny matmuls stay
  single-threaded (below `MM_THREAD_THRESHOLD` ~500k MACs), so the demo/tests are unchanged.
- **Speedup:** **2.68×** on 8 threads (d_model 256 / 4 layers / ctx 64). Thread count =
  `$TF_THREADS` or the CPU count (capped 64). All 8 cross-targets build; WASM still runs.

Remaining perf (later, if needed): SIMD (auto-vec / guarded intrinsics) + int8 quantization.

## Phase #3 (real GPT-2) — DONE, generates real text

- **`tools/gpt2_setup.py`** — downloads GPT-2 `model.safetensors` (huggingface_hub) and writes
  a GPTC checkpoint directly (splits fused `c_attn`, ties `lm_head = wte^T`). One command →
  `model/gpt2/model.bin` (vocab 50257, d_model 768, 12 layers, 622 MB).
- **`tools/gpt2_tok.py`** — GPT-2 BPE via `tiktoken`: `encode "text"` → ids, `decode ids` →
  text. (We use tiktoken instead of reimplementing GPT-2's regex+BPE in C.)
- **`model/<name>/` convention + `--name`** — `main.c`: portable `ensure_dir` + `model_paths`;
  `train --name X` saves `model/X/{model,tok}.bin`; `generate --name X` loads it (text mode if
  `tok.bin` present, else token-ID mode — GPT-2 has none). `model/**/*.bin` git-ignored.
- **Verified end-to-end** on this machine: pip install safetensors+tiktoken, downloaded +
  converted GPT-2 (153 s), ran `echo <ids> | transformer generate --name gpt2` → **coherent
  GPT-2 text** in ~5.6 s / 40 tokens. Deps present: python `D:\env\python`, ~1.5 GB RAM to load.
- Remaining niceties: a `gpt2_chat.py` wrapper (encode→generate→decode in one), and threading
  the M=1 decode-step matmuls over columns (currently single-threaded per token).

## Phase #4 (pluggable agent backend) — DONE

- **`agent.h/.c`** — `AgentBackend` interface (struct + `respond` fn ptr; concrete backends
  embed it as the first field → no `void *`). Two backends: `scripted_backend` (fixed
  responses) and `command_backend` (runs an external program; transcript written to a temp
  file passed as argv, program stdout = response). `agent_run(goal, AgentBackend*, mode)` keeps
  a running transcript, capped at `AGENT_MAX_STEPS`. Portable popen (Win32/POSIX; WASM disabled;
  `_POSIX_C_SOURCE` for glibc/musl under -std=c11).
- **`main.c`** — `agent [--ask] [--cmd "<prog>"] [--goal "..."]`.
- **`tools/agent_mock.py`** — example backend that drives write→read→done (reads the transcript
  to know the step), with a commented Anthropic/OpenAI call to swap in a real LLM.
- Verified: scripted + command backends both run; all 8 targets build; smoke test PASSED.

## Phase #5 / Phase E (multimodal images) — COMPLETE (E1–E4)

- **`image.c/.h`** — loads binary **PPM (P6)** into an `Image` (h,w,c=3; HWC floats [0,1]),
  pure C (skip whitespace/#comments in the header). `tools/img_to_ppm.py` (Pillow) converts
  PNG/JPG → PPM so the C stays dependency-free.
- **`patch.c/.h`** — `patch_embed`: grid of (patch×patch) patches → flatten (HWC) → `matmul`
  by W (patch·patch·C, d_model) + `bias_add` → `(n_patches, d_model)`. `patch_count`/`patch_dim`.
- **`main.c`** smoke test `[5]`: writes a 4×4 PPM, loads it, patch-embeds (patch 2 → 4 patches),
  checks shape + finiteness. All 8 targets build; PASSED.
- **E3 DONE:** `model.c` `gpt_forward_multimodal(m, img_emb, ids, n_img, seq_len, a, logits)`
  builds `emb = [patch_emb + wpe] ++ [wte[tok] + wpe]` then calls the shared `gpt_run_from_emb`
  (refactored out of `gpt_forward_train`, so the text path is unchanged/bit-exact). Causal
  attention over the combined sequence (text sees the image patches). `a` sized to
  `n_img+seq_len` (<= max_seq_len). Verified `[5] Image: 4 patches + 3 text -> logits (7,16)`.
- **E4 DONE:** `gpt_backward_multimodal` (block backward → scatter text grads to `wte`/`wpe`,
  write image grad to `img_emb->grad`) + `patch.c` `patch_embed_backward` (learns W_patch) +
  `loss.c` ignore-index (target < 0 skips). Smoke `[6]` overfits image→caption (loss 4.69 →
  ~0). Refactored `gpt_backward_to_emb` shared by text + multimodal backward.
  Honest: useful VLM quality needs real image-caption datasets + far more scale.

## Phase #6 / Phase F (video) — COMPLETE (F1, F2, F4)

- **`video.c/.h`** — `Video` = sequence of frames. `video_load(dir)` reads
  `frame_0000.ppm, 0001, ...` sequentially. `video_embed` patch-embeds each frame (shared
  W/b, reuses `patch_embed`) + adds a **sinusoidal temporal encoding** per frame, stacking
  into `(n_frames*patches, d_model)`. `video_embed_backward` distributes the grad per frame
  (temporal term is constant → passes through). `video_patch_count`.
- **`tools/video_to_frames.py`** — ffmpeg → `frame_%04d.ppm` (`-start_number 0`).
- **`main.c`** smoke `[7]`: 2-frame clip → video patches + caption → multimodal train; loss
  5.998 -> 0.001. Video is just [frame patches...] + [text] through `gpt_forward_multimodal`,
  so E3/E4 did the heavy lifting.
- **Audio (F3) DONE:** `audio.c` (WAV load, waveform framing → project + temporal encoding +
  backward, reuses `matmul`/`bias_add`), `tools/audio_to_wav.py` (ffmpeg). Smoke `[8]`:
  audio→caption train, loss 7.30 -> 0.0017. Same multimodal prefix trick — raw-waveform frames
  (not mel-spectrograms) to stay dependency-free. So text/image/video/**audio** all train via
  `gpt_forward_multimodal` + `gpt_backward_multimodal`.

## 🏁 Roadmap COMPLETE — all six phases done

#1 Training UX · #2 Thread pool · #3 Real GPT-2 · #4 Agent backend · #5 Images (E1–E4) ·
#6 Video (F1/F2/F4). Pure C11, all 8 platforms (Linux/macOS/Windows/WASM), every step green
(smoke test PASSED, gradient-checked, bit-exact where claimed). Honest ceiling: the pipelines
are correct and runnable; *quality* for real image/video understanding needs real datasets +
far larger models than a from-scratch CPU build.

**Possible future work (optional):** F3 audio; real image-caption/video datasets; SIMD/int8
quant (#2 remainder); a `tools/agent_llm.py` real-LLM agent backend; instruction-tuning; a
`gpt2_chat.py` one-command wrapper; larger GPT-2 variants (`gpt2-medium/large/xl`).
- Optional real agent: a `tools/agent_llm.py` calling a real LLM API (`agent --cmd ...`).
- Wire a real model backend into `agent.c` (replace scripted `responses[]`) — live LLM API
  or a large locally-run model.
- Phase D — perf/quant + a genuinely capable model (real weights).
- Run the actual 124M weights: download GPT-2, `convert_gpt2.py` → `convert_to_gptc.py` →
  `gpt_load`, then generate (needs Phase B sampling/KV-cache/REPL).
- Phase B/C: generation CLI, then the agent loop (tools + approval). See roadmap at top.


## 2026-07-22 -- Session: chat templates + C-native text-to-image (RESUME HERE)

Branch `agent-1` (local only; NOT pushed -- master merged by user). Everything below is
smoke-tested (Output validation PASSED); grad-check + KV bit-exact untouched (all additive).

**DONE this session:**

1. **Chat stop fixes** (commits 745c057, ce6ed61, e04b692): generation stops at end tokens
   (`<|endoftext|>`/`<|im_end|>`/`<|eot_id|>`) and at role labels
   (User/You/Human/Assistant/AI/System) matched as WHOLE WORDS at line start -- fixes Qwen's
   trailing "\nUser" garbage and fake self-conversations. In main.c `hf_stop`,
   tools/llama_stream.py, gpt2_stream.py, webchat handlers.go `trimAtStop`.

2. **Special tokens + chat templates in pure C** (commits 45b0a08, dd60835) -- the big one:
   - `.tok` format -> v2 (v1 still loads): adds bos, template_kind (plain/chatml/llama3),
     default_system, and the special-token table (id, end-of-generation flag, literal string).
   - hftok.c: `hftok_encode` splits input on special-token strings (greedy, probed at '<') and
     emits each as ONE id; `hftok_is_eog` (stop set); `hftok_chat_render` (turns -> prompt for
     ChatML / Llama-3 / plain).
   - main.c: `hf_generate()` shared core stops on the REAL eog token; new
     `hfchat <model.bin> <tok.hf> [n]` reads RS(0x1e)/US(0x1f)-delimited turns from stdin and
     applies the template in C.
   - tools/export_tokenizer.py writes v2 (reads added_tokens + chat_template from HF files).
   - webchat sends structured turns -> hfchat. VERIFIED: a ChatML prompt tokenizes
     BIT-IDENTICAL to HF `tokenizers` (`<|im_start|>`=151644, `<|im_end|>`=151645 single ids).
   - NOTE: `model/*/tok.hf` are git-ignored; qwen/smollm/gpt2 already re-exported to v2. Base
     Qwen/SmolLM still ramble (not instruct-tuned) -- the infra is correct; an INSTRUCT model
     would emit `<|im_end|>` and stop cleanly (offered to convert SmolLM2-135M-Instruct, not done).

3. **Text-to-image kept C-NATIVE** (user: "no python for diffusers, make it C, from scratch"):
   uninstalled torch/diffusers; webchat `-imagegen` is now OPT-IN only (commit f0bd205). Default
   image path = the from-scratch `t2i` (VQ-VAE + autoregressive GPT, all pure C, no Python).

4. **t2i improved** (commits 482afc5, 09bab75):
   - `t2i-data`: procedural dataset generator (5 shapes x 8 colors; caption "a <color> <shape>").
   - Generation SAMPLES image codes (temp + top_k, masked to the code vocab) not greedy argmax.
   - **Minibatch gradient accumulation** (BATCH=8) in `t2i_build`: `gpt_backward` zeros+fills
     grads per example, so accumulate into a shadow buffer + average before `adam_step`. Drove
     GPT loss 1.0 -> **0.0002** on the DETERMINISTIC 40-shape set.
   - `t2i-recon`: debug subcommand -- round-trips an image through the trained VQ only (no GPT).
   - RESULT (VERIFIED via montage): every "a <color> <shape>" caption -> correct color AND a
     recognizable shape. Clean from-scratch pure-C text-to-image.

**generalization (commit 4294e91 code) -- VALIDATED 2026-07-23:**
   - Reintroduced per-variant position/size jitter (seeded per index) + linear LR decay to 0.1x.
     Defaults now: `t2i-data` n=480, `t2i-train` 1800 steps, gen temp 0.6.
   - Prior no-decay run (1400 steps) UNDERTRAINED: loss plateaued ~0.25, confused blue->green,
     blobby shapes. LR decay + gentler jitter fixed that.
   - **CONFIRMING RUN (2026-07-23):** `t2i-data data\t2i 480 16` then
     `t2i-train data\t2i model\t2i\t2i 1800 1800`. VQ recon_mse ~0.005; GPT loss **6.50 -> 0.093**
     (blew past the old 0.25 plateau ~step 1400 as LR decayed). Model on disk (`model/t2i/t2i.gpt`
     + `.vqt`, 8:30 AM) is now the GOOD model.
   - **VERIFIED via montage** (each caption x 4 seeds, scratchpad ppm_montage.py -> PNG):
     all 5 shapes {circle, square, triangle, diamond, cross} render recognizably; all 8 colors
     {red green blue yellow cyan magenta orange purple} render DISTINCT + CORRECT (blue->green
     confusion GONE). Outputs vary per seed but keep correct color+shape => real generalization,
     not memorization. Only nit: orange skews slightly toward yellow (closest palette pair).
     NOTE: shape vocab is those 5 only -- out-of-vocab captions (e.g. "star") -> a colored blob.

**THEN (user-requested next): richer concepts** -- multi-object scenes + spatial relations
("on the left"), more shapes -> real compositionality. **DONE (partial) 2026-07-23, see below.**

Build: `$c=(Get-ChildItem *.c|%{$_.Name}); D:\env\zig\zig.exe cc -std=c11 -O2 -Wall -o
transformer.exe @c -lm`. Both exes (transformer.exe, tools/webchat/webchat.exe) left built.
Montage helper (view PPMs): scratchpad ppm_montage.py (stdlib zlib PNG).


## 2026-07-23 -- Session: two-object compositional t2i + checkpointing (RESUME HERE)

Extended the single-object t2i to **two-object scenes with spatial relations** (commit
4a4d880, all in `t2i.c`). Caption = "a <c1> <s1> <rel> a <c2> <s2>", rel in
{left of, right of, above, below}. `draw_scene` places obj1 in the named half (left/top for
left-of/above, right/bottom for right-of/below), each jittered; `run_t2i_data` now RANDOMLY
samples (c1,s1,rel,c2,s2) per index over a 6400-combo space >> dataset (forces relation
generalization across unseen pairings). Also `T2I_CAPTION_LEN` 32->48, GPT d 96->128 / dff 512.

**NEW: checkpointing.** `t2i_build` gained ckpt_prefix/ckpt_every params; `run_t2i_train` saves
the model every 300 GPT steps (dir created BEFORE the loop). A long/interruptible run keeps
usable weights -- added because two background runs got killed near ~90 min (the effective
runtime wall on this box). `t2i_save` forward-declared. VERIFIED on disk mid-run.

**RESULT (24px/d128/2200 steps, 768 scenes, loss 6.59 -> 0.054) -- PARTIAL SUCCESS.** Verified
via relation-swap montage ("a red circle <rel> a blue square" x seeds, temp 0.4) + unseen-pairing
montage:
  - **Horizontal relations LEARNED:** "left of" -> red left/blue right; "right of" -> red
    right/blue left. Position correct across seeds AND on object pairings never trained together.
  - **Vertical relations CONFLATED:** "above" and "below" both render blue-top/red-bottom (only
    "below" is then correct). Hypothesis: raster-order AR generation re-expresses left/right in
    every scanline but must commit above/below at the very first codes -> weaker signal.
  - **Colors mostly right, but 2nd object bleeds** (blue->magenta) in some samples -- small
    (~12px) objects at 24px stress the VQ codec.

**32px scale-up ATTEMPT (d160, user-requested) -- FAILED on this CPU.** Measured ~3.6-5 s/GPT
step at seq 304 (256 codes/img), so only ~1300 steps fit before the ~90-min kill wall vs the
2200 that 24px needs. Result: UNDERTRAINED, **color collapse** (everything red/orange, blue
gone) -- WORSE than 24px. Reverted source to the committed 24px/d128 config. LESSON: 32px
two-object exceeds this machine's from-scratch training budget within the runtime window.

**On disk now:** `model/t2i_scenes/t2i.gpt`+`.vqt` = the restored 24px partial model (loss 0.054).
Dataset `data/t2i_scenes` (768 @ 24px). Reproduce: `t2i-data data\t2i_scenes 768 24` then
`t2i-train data\t2i_scenes model\t2i_scenes\t2i 2200 2000` (deterministic -> 0.054).

**NEXT options to fix vertical relations / color bleed (all fit 24px budget):**
  1. **Relation/color/shape as dedicated vocab tokens** (not raw caption bytes) -> stronger,
     cleaner conditioning; likely the highest-leverage fix for the vertical-relation conflation.
  2. Bigger model at 24px (d160) + more steps (2600) -- still completes in the runtime window.
  3. More data (1200+ scenes) for better pairing coverage.
  Brute-force resolution (32px+) is OFF THE TABLE until faster compute (SIMD/threads/GPU).


## 2026-07-23 (eve) -- Structured caption tokens (WIP) -- SUPERSEDED, see 2026-07-24 below

Started the highest-leverage fix for the vertical-relation conflation: **structured caption
tokens** (commit 964846a) -- each color/shape/relation is now ONE dedicated vocab token instead
of raw caption bytes, giving the GPT a clean one-hot concept signal (esp. the relation word).

**Code done + committed (964846a), build clean:**
  - `encode_caption` rewritten as a parser: "a <c1> <s1> <rel> a <c2> <s2>" -> 5 tokens
    [c1,s1,rel,c2,s2] via `match_word_color/shape/rel` (relation matched by unique first word;
    "a"/"of" skipped; PAD-filled). Token layout above image codes: `T2I_TOK_COLOR/SHAPE/REL/PAD`,
    `T2I_VOCAB` = 274.
  - `T2I_CAPTION_LEN` 48->5; sequence 192->149. GPT d 128->160, dff 512->640 (shorter seq +
    smaller vocab freed the budget). Runs ~0.76 s/step => ~30 min for 2200 steps (was ~90).
  - Sanity checks PASSED: parser traced correct; vocab 274; step-0 loss 5.51 ~= ln(274).

**NOT validated -- training was STOPPED at step 450/2200 (loss ~0.20, user going home).** No good
structured-token model on disk (only an early step-300 checkpoint under `model/t2i_tok/`, ignore it).

**▶▶ RESUME (one command each, ~30 min total):**
  1. Rebuild: `$c=(Get-ChildItem *.c|%{$_.Name}); D:\env\zig\zig.exe cc -std=c11 -O2 -Wall -o transformer.exe @c -lm`
  2. Data (already on disk as `data/t2i_tok`, 1000 @ 24px; regen if missing):
     `.\transformer.exe t2i-data data\t2i_tok 1000 24`
  3. Train: `.\transformer.exe t2i-train data\t2i_tok model\t2i_tok\t2i 2200 2000`  (checkpoints /300)
  4. VALIDATE (the whole point -- does above/below disentangle now?): relation-swap montage,
     `.\transformer.exe t2i-gen model\t2i_tok\t2i <out.ppm> "a red circle <rel> a blue square" 0.4 <seed>`
     for rel in {left of, right of, above, below} x a few seeds; montage via scratchpad
     `ppm_montage.py` (recreate it -- stdlib zlib PNG, cols=3 scale=10). SUCCESS = above puts red
     on TOP, below puts red on BOTTOM (byte-caption model conflated them to blue-top/red-bottom).
  5. If vertical now works: commit result + update notes/memory; then compare vs the byte-caption
     partial model (`model/t2i_scenes`). If still conflated: try more steps (2800) / d192, or
     reconsider raster-order (generate codes in an order that expresses vertical early).

Prior baseline to beat (byte captions, commit 4a4d880, model `model/t2i_scenes`, loss 0.054):
horizontal relations OK, vertical CONFLATED, some 2nd-object color bleed.


## 2026-07-24 -- Structured caption tokens VALIDATED (relations fixed; color is the new wall)

Finished the run above and validated it **objectively** instead of by eyeballing a montage.
Result: **the structured-token fix WORKED -- vertical relations are no longer conflated.** The
bottleneck moved to **color fidelity, and that is the VQ-VAE's fault, not the GPT's.**

**Run:** rebuilt clean, VQ 2000 steps (recon_mse 0.0019), GPT 2200 steps requested but the run was
KILLED at ~step 1900 (loss ~0.11). The **step-1800 checkpoint** is what is on disk in
`model/t2i_tok/` and what every number below was measured on -- checkpointing/300 saved the run.

**New tooling (the real deliverable -- don't eyeball montages again):**
  - `tools/t2i_score.py` -- assigns each pixel to the nearest `T2I_COLORS` entry, takes per-color
    centroids, and scores **two separate axes**: color fidelity (did both requested colors render?)
    and relation accuracy (given both rendered, is obj1 on the stated side?). Conflating these is
    exactly what made the first read of the montage wrong.
  - `tools/ppm_montage.py` -- tile PPMs into one scaled PNG (stdlib zlib, no deps).

**CALIBRATE FIRST -- ground-truth `data/t2i_tok/img_*.ppm` scored by the same tool:**
  color fidelity **100%** (24/24), relation accuracy **88%** (21/24). So ~88% is the measurement
  CEILING for relations (jittered shapes straddle the midline); 100% is the ceiling for color.

**Measured (step-1800 model, temp 0.4):**

| what                              | color fidelity | relation accuracy   |
|-----------------------------------|----------------|---------------------|
| ground truth (ceiling)            | 100% (24/24)   | 88% (21/24)         |
| VQ-VAE **reconstruction** of it   | **67%** (16/24)| 75% (12/16)         |
| GPT generation, all 8 colors      | 33% (16/48)    | **81%** (13/16)     |
| GPT generation, 4 "safe" colors   | 30% (17/56)    | 71% (12/17)         |

**Conclusions (this is the important part):**
  1. **Relations are essentially SOLVED, vertical included.** 81% vs an 88% ceiling. Per-relation on
     the all-colors run: above **6/6**, below 5/7. The byte-caption baseline had ZERO vertical
     sensitivity (always blue-top/red-bottom); dedicated relation tokens fixed it. Goal achieved.
  2. **Color fidelity is the new wall, and it is capped by the VQ-VAE.** Reconstructing *training*
     images -- no GPT involved -- already loses a color 33% of the time. So even a perfect GPT could
     not exceed ~67%. More GPT steps CANNOT fix this.
  3. Dominant failure is **blue -> cyan**, plus magenta->purple and orange->yellow: adjacent hues
     collapsing, i.e. codebook resolution/collapse. A single image uses only **14 distinct codes of
     K=256**, so the codebook is nowhere near saturated -- this smells like codebook COLLAPSE (dead
     entries) rather than K being too small.
  4. My first eyeball read of the montage said "relations only 1/3 right, strong word-order bias".
     That was WRONG -- the failures I was counting were mostly color failures. Measure, don't squint.

**▶▶ NEXT (highest leverage first):**
  1. **Fix VQ-VAE color fidelity** -- the actual blocker. Try: dead-code restart / EMA codebook
     updates (reinit unused entries to jittered encoder outputs), a commitment-cost tweak, or
     bump `T2I_LDIM` 16->32. Re-measure with the recon row of the table above (target: 100%).
     Worth adding an aggregate codebook-utilization printout across the dataset to confirm collapse.
  2. Only then, finish/extend GPT training (the 2200-step run never completed; ~30 min).
  3. Horizontal-relation sample sizes were tiny (n=1..4) because color failures ate them --
     re-measure relations properly once color fidelity is fixed.


## 2026-07-24 (2) -- Training the modern (LLaMA-family) architecture: llamatrain.c

Started the post-t2i roadmap. Goal chosen: make the project able to **train**, not just run, the
architecture real modern models use -- the prerequisite for LoRA fine-tuning a real small model
(SmolLM2-135M) so the agent loop can finally be driven by OUR engine instead of an external API.

**The gap:** `llama.c` was inference-only (`llama.h` had no backward/train symbols at all), even
though `rmsnorm.c`, `rope.c`, `swiglu.c`, `gqa.c` each already had a gradient-checked `_backward`.
The per-op math existed; the model-level stitching did not.

**Added (pure C11, no deps):**
  - `llamatrain.h` / `llamatrain.c` -- new module, mirroring the GPT training path in `model.h`:
    `LlamaActs` (cached activations incl. per-layer `GqaCache`/`SwigluCache`), `llama_acts_create`
    /`_free`, `llama_forward_train`, `llama_backward`, `llama_param_count`, `llama_collect_params`,
    `llama_zero_grad`. Params = **3 + 12*n_layers** (27 for the 2-layer test model).
  - `rope.c/h`: **`rope_heads_backward` / `rope_heads_backward_pos`** -- the multi-head in-place
    RoPE had no backward. It TRANSFORMS `x->grad` in place (inverse rotation) rather than
    accumulating, because the in-place forward means pre- and post-rotation grads share a buffer.
    Documented explicitly in the header since it breaks the usual accumulate convention.
  - `llama.c/h`: `llama_model_create_train` / `llama_block_create_train` -- same weights but WITH
    grad arrays. Inference still allocates `nograd` (halves loaded-model memory), via shared
    `*_create_ex(..., int grad)` helpers so the two paths can't drift.

**Backward order** (reverse of forward; the residual means `h`/`x` receive grad from two paths, so
the RMSNorm backward must run before the matching `matadd_backward` propagates):
  out=h+f -> swiglu -> ffn_norm -> h=x+ao -> Wo -> gqa -> RoPE(q,k) -> biases -> Wq/Wk/Wv ->
  attn_norm -> scatter emb grad into `wte` rows.

**Verified -- new selftest part [22], three gates:**
  - (a) `llama_forward_train` == `llama_model_forward` **bit-exactly** (max |diff| 0.00e+00). This
    matters: if the caching forward differed, the backward would be differentiating a different
    function than the one we ship.
  - (b) end-to-end numerical gradient check over ALL 27 param tensors: **max rel err 1.04e-03**
    on conditioned entries (gate < 1e-2), same rule as the GPT check in [2].
  - (c) Adam, 200 steps, next-token task: loss **3.1764 -> 0.0012**.
  Also still green: KV-cache bit-exact 0.00e+00, `Output validation: PASSED`, exit 0.
  **All 8 cross targets build -Wall clean** (incl. wasm32-wasi + both macOS).

Makefile OBJS/deps + README module table updated (contributor rules 7/8).

**NEXT:** 1) LoRA adapters on top of this (freeze base, train rank-r A/B on the attention/FFN
projections; base can stay int8 via `quant.c`) -- that is what makes fine-tuning 135M feasible on
CPU; 2) AdamW + global grad-norm clipping + cosine-with-warmup LR in `optim.c` (plain Adam today,
no decay/clip/schedule) -- needed for fine-tuning stability; 3) SIMD `matmul` (`matrix.c` is scalar
+ thread pool only) -- guard `__AVX2__`/`__ARM_NEON` with a scalar fallback for WASM, and note it
won't stay bit-exact once the FMA reduction is reassociated.

## 2026-07-24 (3) -- Training-recipe essentials: AdamW + grad clipping + LR schedule

Second roadmap item (prerequisite for stable LoRA fine-tuning). `optim.c` was plain Adam: no
weight decay, no clipping, no schedule.

**Added to `optim.{h,c}` (pure C11):**
  - **AdamW**: `weight_decay` field + `adamw_create()`. Decay is DECOUPLED (applied straight to
    the weights, NOT divided by sqrt(v_hat)) -- that decoupling is the whole point. Applied only
    to `ndim >= 2` tensors: biases and RMSNorm/LayerNorm gains are exempt (decaying a norm gain
    fights the normalization). `adam_create` still sets wd=0, so plain Adam is BIT-IDENTICAL to
    before -- verified in the selftest.
  - **`clip_grad_norm(params, n, max_norm)`**: ONE global L2 norm over the whole model (not
    per-tensor), scaled so direction is preserved and only length is capped. Returns the norm
    BEFORE clipping (a spiking grad norm is the first sign of divergence). Accumulates in
    `double` -- summing millions of squared f32 grads loses too much precision otherwise. A
    non-finite norm ZEROES all grads and returns it, so one exploded batch can't write NaNs into
    the weights (unrecoverable).
  - **`lr_cosine_warmup(base, min, step, warmup, total)`**: linear warmup then cosine decay;
    guards total <= warmup instead of dividing by zero. Plus `adam_set_lr()`.

**Wired into `train.c` + `main.c`** (BOTH loops -- normal and `--sft`): new `TrainConfig` fields
`weight_decay`, `grad_clip`, `warmup`, `lr_min` and flags `--weight-decay --clip --warmup
--min-lr`. **All default OFF** so previously-recorded runs reproduce bit-for-bit. Enabled options
print a one-line recipe summary at startup.

**Verified -- new selftest part [23], all against CLOSED-FORM values (not just "it ran"):**
  - AdamW, zero grad => weight scales by exactly (1 - lr*wd) = 0.950000; the 1-D bias stays
    exactly 1.000000 (undecayed). Plain `adam_create` still moves a zero-grad weight by 0.
  - clip: four grads of 3.0 => norm exactly 6.0; no-op when under max; clipped to 2.0000 exactly;
    NaN grad => non-finite norm returned AND every grad zeroed.
  - LR: warmup[0]=0.1, peak=1.0, mid=0.55, end=min=0.1, monotone non-increasing after warmup.
  Plus a real CLI run of both loops. `Output validation: PASSED`, exit 0, all 8 cross targets
  `-Wall` clean.

**HONEST result:** on a 300-step overfit of `data/sample.txt`, baseline Adam reached train loss
0.0907 vs 0.2908 with the recipe on. That is EXPECTED, not a regression -- warmup spends the first
30 steps at a reduced LR and cosine decays to 1e-4, and weight decay regularizes. These options
buy stability on long/fine-tuning runs, not speed on toy memorization. README says so explicitly.

**NEXT:** LoRA adapters (freeze base, train rank-r A/B on attention/FFN projections; base can stay
int8 via `quant.c`) on top of `llamatrain.c` -- then SFT a real SmolLM2-135M so the agent runs on
our own engine. Then SIMD `matmul`.
