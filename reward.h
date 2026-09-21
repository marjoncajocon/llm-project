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
  float total;        // v1: format + grounded + match, in [0, 1]
                      // v2: format * (grounded + match) (0 if capped), in [0, 1]
  char reason[96];    // short note on the first term that failed ("" if perfect)
  int action;         // what the response WAS — a RewardAction (both versions)
} RewardParts;

// The kind of action a response took, for the reward-hacking monitor in grpo.c:
// a policy that learns to answer every task with <done>, or with prose, shows up
// as a share that climbs while the match term stays flat.
typedef enum {
  REWARD_ACT_DONE = 0,
  REWARD_ACT_PROSE,       // no block at all
  REWARD_ACT_READ,
  REWARD_ACT_WRITE,
  REWARD_ACT_RUN,
  REWARD_ACT_MALFORMED,   // failed the format term (unclosed, trailing text, unknown tool)
  REWARD_ACT_COUNT
} RewardAction;

// Score `response` against `prompt` and `reference`. Returns the total in [0, 1]
// and, if parts is non-NULL, fills in the breakdown.
//
// `reference` may be NULL or empty — the task has no known-good action. The match
// term is then dropped and the other two are rescaled to keep the total in [0, 1]
// (the group-relative advantage is scale-free, but a comparable reward keeps the
// training log readable).
float reward_score(const char *prompt, const char *response,
                   const char *reference, RewardParts *parts);

// ---------------------------------------------------------------------------
// Reward v2 — opt-in (grpo --reward v2). reward_score above is v1 and unchanged.
//
// v1 had a free floor: a response that did nothing useful still banked format +
// grounding. Prose, or junk cut off at --max-new, scored 0.5 with zero match; a
// premature <done> scored 0.5 too (always "grounded" — no arguments to check),
// which made stopping the SAFE choice under uncertainty; and a command was only
// checked for tokens with a file extension, so `docker run -e xx=yy | MODEL:`
// (invented flag value, leaked transcript marker) earned full grounding.
//
//   total = format * (0.30 * grounded + 0.70 * match)
//
//   format   GATE in {0, 1}: same test as v1. A malformed action earns nothing —
//            there is no partial credit for a block the agent loop cannot run.
//   grounded in {0, 1} (reported weighted, i.e. 0 or 0.30):
//            tool call: v1's checks, plus — no leaked transcript marker (MODEL:,
//              OBSERVATION:, GOAL:, USER:) or '<' in any argument; a path must be
//              file-shaped (has '.' or '/') unless the reference path is bare;
//              EVERY command argument token must occur in the prompt as a whole
//              token (or of the reference command, so `ps aux` for "show
//              running processes" passes) — the command is split into segments
//              at | ; & and in each segment the first token (the verb) and '-'
//              flags are exempt.
//            prose / <done>: grounded only if the reference is the same kind.
//              With no reference: 0.5 (nothing to check the decision against).
//   match    v1's action_match (0 / 0.4 / 0.7 / 1), weighted 0.70.
//   capped   a rollout that hit --max-new (grpo's STOP_CAP) scores 0 outright.
//
// Consequences: premature <done> on a tool-call task = 0; truncated anything = 0;
// the right tool with ungrounded args but otherwise matching can still earn match
// credit, and a grounded wrong tool earns at most 0.30. The match weight
// dominates, so the ONLY route to a high score is the reference action.
//
// With no reference, the match term is dropped and grounding is rescaled to 1.0.
float reward_score_v2(const char *prompt, const char *response,
                      const char *reference, int capped, RewardParts *parts);

// Dispatch: version 2 => reward_score_v2, anything else => reward_score (v1,
// which ignores `capped`).
float reward_score_version(int version, const char *prompt, const char *response,
                           const char *reference, int capped, RewardParts *parts);

// The largest value the (weighted) match term can take under `version`: 0.50 for
// v1, 0.70 for v2. A rollout is an EXACT match when parts.match reaches it.
float reward_match_max(int version);

// "done", "prose", "read", "write", "run", "malformed".
const char *reward_action_name(int action);

// Exposed for the self-test: is `value` present in `prompt` as a whole token,
// after both are lowercased and their whitespace collapsed? Token boundaries stop
// "ok" from matching inside "cook" while still matching "notes.txt" exactly.
int reward_grounded_in(const char *prompt, const char *value);

#endif // REWARD_H
