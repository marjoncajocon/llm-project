// gen.h - Text generation for HF-vocabulary models (the `hfgen` / `hfchat` core)
//
// Loads a GPTC / LLMA / LLMQ checkpoint, prefills a tokenized prompt and samples
// a continuation, streaming it as JSON deltas ({"d":"piece"} lines, then
// {"done":true}) — the format tools/webchat and tools/agent_local.py consume.
//
// Beyond plain generation it handles three things the tool-using agent needs:
//   * a prompt longer than the model's context keeps its MOST RECENT tokens
//     (the latest turns), reserving room for the reply, and says so on stderr;
//   * opt-in stop once the first action block is closed by </tool> or </done>
//     (the agent executes only that block, so decoding past it is wasted work);
//   * N samples from ONE load and ONE prefill (the KV cache is forked per
//     sample), sample i seeded with seed+i — bit-identical to N separate runs.
#ifndef GEN_H
#define GEN_H

#include "constrain.h"
#include "hftok.h"
#include "llama.h"

typedef struct {
  int n_new;                 // max new tokens per sample
  float temp;                // sampling temperature (<= 0: greedy)
  int top_k;                 // 40 in the CLI
  float top_p;               // 0.95 in the CLI
  unsigned long long seed;   // sample i uses rng_create(seed + i)
  int stop_tags;             // 1: stop once the text contains </tool> or </done>
  int n_samples;             // >= 1; > 1 frames each sample (see hfgen_file)
  int emit;                  // 1: stream JSON deltas to stdout
  int *out_ids;              // optional: n_samples * n_new sampled ids (tests)
  int *out_len;              // optional: n_samples lengths
  int constrain;             // 1: constrained decoding of the action (constrain.h); needs tok.
                             //    Each sample stops once its action is complete; the role-label
                             //    text stop is not applied (the grammar decides the end).
  ConstrainOpts con;         // its options (constrain_default_opts())
  const char *con_prompt;    // text the arguments are grounded in (NULL: the decoded prompt ids)
} HFGenOpts;

HFGenOpts hfgen_default_opts(void);   // 128 tokens, temp 0.8, k 40, p 0.95, seed 1234, 1 sample, emit,
                                      // no constraint

// How much of an np-token prompt to keep for a model with context max_seq when
// n_new tokens are wanted. If the prompt plus the reply does not fit, the OLDEST
// tokens are dropped, keeping the last (max_seq - reserve) with
// reserve = clamp(n_new, 1, max_seq / 2), so a long reply never squeezes the
// prompt below half the context. Writes the index of the first kept token to
// *first and returns the number kept; 0 means nothing can be generated (empty
// prompt, or a context of fewer than 2 positions).
int hfgen_fit_prompt(int np, int max_seq, int n_new, int *first);

// Stop-at-tag: if dec holds a complete action block, cut it right after the
// block's closing tag and return 1; else return 0. The block is the one
// toolcall_parse takes: the first "<tool" or "<done" opened, closed by its own
// "</tool>" / "</done>" (so a write_file body mentioning "</done>" survives);
// before anything is opened, the earliest "</tool>" or "</done>" (the GRPO rule).
int hfgen_cut_at_tag(char *dec);

// Generate from an in-memory f32 llama model (tok may be NULL: no EOG / text
// stops and no decoding — tests only). Returns 0, or 1 if nothing could be
// generated (an {"error":...} line is emitted when opts->emit).
int hfgen_llama(const LlamaModel *m, const HFTok *tok, const int *pids, int np,
                const HFGenOpts *opts);

// Load mpath (GPTC / LLMA / LLMQ by magic) and generate. Output on stdout when
// opts->emit:
//   n_samples == 1 : {"d":...} lines, then {"done":true}        (unchanged format)
//   n_samples  > 1 : per sample {"sample":i,"seed":S}, its {"d":...} lines,
//                    {"end":i}; then one {"done":true}
//   failure        : {"error":"..."} then {"done":true}, and a nonzero return
int hfgen_file(const HFTok *tok, const char *mpath, const int *pids, int np,
               const HFGenOpts *opts);

#endif // GEN_H
