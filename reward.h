// reward.h - Verifiable reward for a sampled agent response (used by GRPO).
//
// Reinforcement learning is only as honest as its reward. Every term here is
// checked against something OUTSIDE the model: the literal text of the prompt,
// or the reference action recorded on the task line. The model's own opinion of
// its work is never consulted — the same rule tools/self_train.py follows, and
// the reason self-training here sharpens behaviour instead of amplifying the
// model's errors.
//
// Three terms, each independently verifiable:
//
//   format   (0.20)  the response is exactly ONE well-formed <tool .../> or
//                    <done> block, properly closed, with nothing trailing.
//   grounded (0.30)  every argument the tool call carries — path, write body,
//                    filenames inside a command — occurs in the PROMPT as a
//                    whole token. This is the C port of clean_sft.py's verify(),
//                    and it is what catches a hallucinated or char-slipped
//                    filename (demo_note -> demonstra_note).
//   match    (0.50)  the parsed action equals the task's reference action.
//                    Partial credit for the right tool with the wrong argument.
//
// Scoring is pure: no files are opened, nothing is executed, nothing is written.
// A reward function with side effects inside a training loop is a blast radius,
// and a nondeterministic one makes a run unreproducible.
#ifndef REWARD_H
#define REWARD_H

// Per-term breakdown, so a run can report WHY reward moved rather than just that
// it did (reward hacking shows up here first: a total that climbs while `match`
// stays flat means the model found a cheap way to satisfy format + grounding).
typedef struct {
  float format;
  float grounded;
  float match;
  float total;        // format + grounded + match, in [0, 1]
  char reason[96];    // short note on the first term that failed ("" if perfect)
} RewardParts;

// Score `response` against `prompt` and `reference`. Returns the total in [0, 1]
// and, if parts is non-NULL, fills in the breakdown.
//
// `reference` may be NULL or empty — the task has no known-good action. The match
// term is then dropped and the other two are rescaled to keep the total in [0, 1]
// (the group-relative advantage is scale-free, but a comparable reward keeps the
// training log readable).
float reward_score(const char *prompt, const char *response,
                   const char *reference, RewardParts *parts);

// Exposed for the self-test: is `value` present in `prompt` as a whole token,
// after both are lowercased and their whitespace collapsed? Token boundaries stop
// "ok" from matching inside "cook" while still matching "notes.txt" exactly.
int reward_grounded_in(const char *prompt, const char *value);

#endif // REWARD_H
