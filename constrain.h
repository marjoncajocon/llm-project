// constrain.h - Opt-in constrained decoding of the agent's action (hfgen/grpo --constrain)
//
// The live model is a tool-using agent: every reply must be ONE action block
//
//     [<think>free reasoning</think>] <tool name="read_file"   path="P"></tool>
//                                     <tool name="write_file"  path="P">BODY</tool>
//                                     <tool name="run_command" command="CMD"></tool>
//                                     <done>TEXT</done>
//
// and the arguments it copies (above all the path) must come from the prompt.
// Sampling alone never fixed the residual char-slips (ctngh4 -> cngh4, lever #4 in
// notes.md): they are systematic, low-variance errors. This module removes them at
// the source by masking the decoder: a token is ALLOWED iff appending its bytes
// keeps the output a valid prefix of the language above. BPE pieces are handled
// byte by byte, so a piece may cross any boundary (`.log">`, `"></tool>`, ...).
//
// The language (canonical, the form the SFT data and toolcall.c use):
//   * optional <think> block first; its text is free except that '<' may only begin
//     "</think>" (so no tag can hide in it), and after think_max bytes the block
//     must close. After </think> the action follows immediately.
//   * path="P": with ground_path (default) P must stay a prefix of a WHOLE-TOKEN
//     span of the prompt, byte-exact: the span starts where the prompt is not
//     preceded by a word char [0-9A-Za-z_] (nor by '<': "/tool>" of a tag is not a
//     path), holds no whitespace / quote / <>|`,;&()
//     and the closing quote is allowed only where the span may END as a whole token
//     (end of span or a non-word char next) -- reward.c's `grounded` rule, made
//     case- and byte-exact. Implemented as a suffix array of the span starts sorted
//     by span text: a partial value is a contiguous range, narrowed per byte by
//     binary search, built once per prompt.
//   * write body: free (any bytes) unless ground_body; it may not contain "<tool",
//     "</done>", "<think>" or "</think>" (each would change how the reply is parsed
//     or where decoding stops), must hold a non-space byte, and ends at "</tool>".
//   * command: free (no '"', '<', '>' or control bytes) unless ground_cmd; with it every
//     argument that is not the verb of a | ; & segment and not a '-' flag must be a
//     whole-token prompt span (reward v2's rule). Optional constraints are OFF by
//     default -- a command verb or a new file body is legitimately new content.
//   * done text: free; may not contain "<done", "</tool>", "<think>" or "</think>";
//     ends at "</done>".
//
// Never a dead end: the grammar always has a shortest completion (e.g. `</think>`,
// a grounded close, `"></tool>`), and when the remaining token budget is known a
// token is allowed only if the shortest completion still fits in it (one byte per
// token, worst case) -- so the action is closed before --max-new runs out. The
// picker walks the top candidates in logit order; if none is allowed it scans the
// whole vocabulary; only if nothing at all is allowed does it return -1 (an
// explicit error for the caller).
#ifndef CONSTRAIN_H
#define CONSTRAIN_H

#include "hftok.h"
#include "sampler.h"

typedef struct {
  int ground_path;   // 1 (default): path="..." must be a whole-token prompt span
  int ground_body;   // 1: the write_file body must be ONE whole-token prompt span (default 0)
  int ground_cmd;    // 1: run_command arguments (not verbs / -flags) must be prompt spans (default 0)
  int think_max;     // <think> text bytes before </think> is forced (default 256; 0 = no cap)
  int window;        // greedy: candidates walked in logit order before a full scan (default 64)
} ConstrainOpts;

ConstrainOpts constrain_default_opts(void);

typedef struct Constrain Constrain;   // the per-prompt index (span suffix arrays)

// Build the index for one prompt (the text the arguments must be grounded in).
// Returns NULL only on allocation failure.
Constrain *constrain_create(const char *prompt, const ConstrainOpts *o);
void constrain_free(Constrain *c);

// Decoder state: a small POD, copied freely (a candidate token is tested on a copy).
typedef struct {
  int st;            // CS_* (constrain.c)
  int set, li;       // literal set and bytes matched in it
  unsigned alive;    // literal alternatives still matching
  int next;          // what follows a path: read / write
  int k;             // terminator match length in free text
  unsigned char fm[4];  // forbidden-tag match lengths in free text
  int count, nonws;  // bytes / non-space bytes of the current free region
  int lo, hi, L;     // span range in the suffix array and value length
  int sub;           // command sub-state
  int nbytes;        // total bytes accepted
} ConState;

void constrain_init(const Constrain *c, ConState *s);
// Feed n bytes. Returns 1 if the output is still a valid prefix (s advanced), 0 if
// not (s is then undefined: feed a copy).
int constrain_feed(const Constrain *c, ConState *s, const unsigned char *b, int n);
// 1 once a complete action has been produced (nothing more may follow).
int constrain_complete(const ConState *s);
// Bytes in the shortest completion from s (0 when complete).
int constrain_min_left(const Constrain *c, const ConState *s);
// Convenience for tests: 2 = text is a complete action, 1 = valid proper prefix,
// 0 = invalid.
int constrain_check(const Constrain *c, const char *text);

// The vocabulary as the constraint sees it: token bytes, and a ban list (empty
// pieces, special / end-of-generation tokens, pieces holding a NUL byte).
typedef struct {
  int V;
  const int *len;
  unsigned char *const *bytes;
  unsigned char *ban;
} ConVocab;

int convocab_from_hftok(ConVocab *v, const HFTok *t);   // 0 on success
void convocab_free(ConVocab *v);

// Pick the next token under the constraint and advance *s.
//   temp <= 0 : greedy -- the highest-logit allowed token (ties: lower id).
//   temp  > 0 : the top_k candidates in logit order (all if top_k <= 0), keep the
//               allowed ones, renormalize softmax(logit / temp) over them, apply
//               top_p, draw one rng_uniform. None allowed among them -> the same
//               over the top_k ALLOWED tokens of the whole vocabulary.
//   budget    : tokens left including this one (<= 0: unknown). If the shortest
//               completion fits, a token is allowed only if it still fits after it.
// Returns the id, or -1 if no token of the vocabulary is allowed (*s unchanged).
// *full_scan (may be NULL) is set to 1 when the whole vocabulary had to be scanned.
int constrain_pick(const Constrain *c, const ConVocab *v, ConState *s, const float *logits,
                   float temp, int top_k, float top_p, Rng *rng, int budget, int *full_scan);

#endif // CONSTRAIN_H
