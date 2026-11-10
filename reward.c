// reward.c - Verifiable reward for a sampled agent response (see reward.h).
#include "reward.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "toolcall.h"

#define W_FORMAT   0.20f
#define W_GROUND   0.30f
#define W_MATCH    0.50f

// Filename extensions that make a token inside a run_command "data the command
// operates on", which must be grounded. Mirrors clean_sft.py's EXT_RE. Command
// VERBS (ls, git, make) are deliberately not checked — they are canonical, not
// copied from the prompt.
static const char *const EXTS[] = {"txt", "md", "log", "json", "csv", NULL};

static int is_word_ch(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
         (c >= 'A' && c <= 'Z') || c == '_';
}

// Lowercase + collapse every whitespace run to a single space + trim: the space
// in which grounding is tested (clean_sft.py's norm()). Writes at most out_size-1
// bytes. Returns the length written.
static int norm_into(const char *s, char *out, int out_size) {
  int n = 0, pending_space = 0;
  if (out_size <= 0) return 0;
  for (const char *p = s; *p && n < out_size - 1; p++) {
    unsigned char c = (unsigned char)*p;
    if (isspace(c)) {
      if (n > 0) pending_space = 1;       // never leading; collapses runs
      continue;
    }
    if (pending_space) { out[n++] = ' '; pending_space = 0; if (n >= out_size - 1) break; }
    out[n++] = (char)tolower(c);
  }
  out[n] = '\0';                          // trailing space never emitted
  return n;
}

// Whole-token search of `needle` (already normalized) inside `hay` (normalized).
static int find_whole_token(const char *hay, const char *needle) {
  const size_t nlen = strlen(needle);
  if (nlen == 0) return 0;
  for (const char *p = strstr(hay, needle); p; p = strstr(p + 1, needle)) {
    const char before = (p == hay) ? '\0' : p[-1];
    const char after = p[nlen];
    if (!is_word_ch(before) && !is_word_ch(after)) return 1;
  }
  return 0;
}

// Normalize both sides, then whole-token search. Allocates, so callers that test
// many values against one prompt normalize the prompt once and use the static
// helper above instead.
int reward_grounded_in(const char *prompt, const char *value) {
  const size_t pl = strlen(prompt) + 1, vl = strlen(value) + 1;
  char *pn = (char*)malloc(pl), *vn = (char*)malloc(vl);
  int ok = 0;
  if (pn && vn) {
    norm_into(prompt, pn, (int)pl);
    norm_into(value, vn, (int)vl);
    ok = find_whole_token(pn, vn);
  }
  free(pn); free(vn);
  return ok;
}

// Grounding against an ALREADY-normalized prompt.
static int grounded(const char *prompt_norm, const char *value) {
  const size_t vl = strlen(value) + 1;
  char *vn = (char*)malloc(vl);
  int ok = 0;
  if (vn) {
    norm_into(value, vn, (int)vl);
    ok = find_whole_token(prompt_norm, vn);
  }
  free(vn);
  return ok;
}

// Drop <think>...</think> spans into `out`: ephemeral rationale, not arguments.
// An unclosed <think> swallows the rest, matching the DOTALL regex's behaviour on
// a truncated generation (there is nothing to score after it anyway).
static void strip_reasoning(const char *resp, char *out, int out_size) {
  int n = 0;
  const char *p = resp;
  while (*p && n < out_size - 1) {
    if (strncmp(p, "<think>", 7) == 0) {
      const char *end = strstr(p, "</think>");
      if (!end) break;
      p = end + 8;
      continue;
    }
    out[n++] = *p++;
  }
  out[n] = '\0';
  // trim
  while (n > 0 && isspace((unsigned char)out[n - 1])) out[--n] = '\0';
}

// Exactly one properly-closed block, nothing but whitespace after it.
static int format_ok(const char *resp, const ToolCall *tc) {
  if (tc->kind == TOOL_NONE) return 0;
  const char *mark = (tc->kind == TOOL_DONE) ? "<done" : "<tool";
  const char *tag  = (tc->kind == TOOL_DONE) ? "</done>" : "</tool>";
  const char *open = strstr(resp, mark);
  if (!open) return 0;
  const char *close = strstr(open, tag);
  if (!close) return 0;                          // unterminated block
  const char *dup = strstr(open + 1, mark);
  if (dup && dup < close) return 0;              // a second opener inside the block
  // Anything but whitespace after the close is either the model rambling past its
  // action or a second action the agent loop would never execute.
  for (const char *p = close + strlen(tag); *p; p++)
    if (!isspace((unsigned char)*p)) return 0;
  return 1;
}

// Every filename-like token inside a command must be grounded (clean_sft's
// EXT_RE loop). Returns 1 if all are, else 0 and copies the offender to `bad`.
static int command_files_grounded(const char *cmd, const char *prompt_norm,
                                  char *bad, int bad_size) {
  for (const char *p = cmd; *p; p++) {
    if (*p != '.') continue;
    int matched = 0;
    for (int e = 0; EXTS[e] && !matched; e++) {
      const size_t el = strlen(EXTS[e]);
      // case-insensitive extension compare, then a non-word char (or end) after
      size_t i = 0;
      while (i < el && p[1 + i] && tolower((unsigned char)p[1 + i]) == EXTS[e][i]) i++;
      if (i != el) continue;
      if (is_word_ch(p[1 + el])) continue;
      matched = 1;
      // walk back over [\w./-] to the start of the whole token
      const char *start = p;
      while (start > cmd) {
        char c = start[-1];
        if (is_word_ch(c) || c == '.' || c == '/' || c == '-') start--;
        else break;
      }
      const char *end = p + 1 + el;
      const int len = (int)(end - start);
      char *tok = (char*)malloc((size_t)len + 1);
      if (!tok) return 1;                        // OOM: do not fail the rollout
      memcpy(tok, start, (size_t)len);
      tok[len] = '\0';
      int ok = grounded(prompt_norm, tok);
      if (!ok) {
        snprintf(bad, (size_t)bad_size, "ungrounded-cmd-file:%s", tok);
        free(tok);
        return 0;
      }
      free(tok);
    }
  }
  return 1;
}

// The C port of clean_sft.py verify(): 1 if the call's args are all grounded.
static int grounding_ok(const char *prompt_norm, const ToolCall *tc,
                        char *reason, int reason_size) {
  if (tc->kind == TOOL_DONE || tc->kind == TOOL_NONE) return 1;   // no args to ground
  if (tc->path[0] && !grounded(prompt_norm, tc->path)) {
    snprintf(reason, (size_t)reason_size, "ungrounded-path:%.48s", tc->path);
    return 0;
  }
  if (tc->kind == TOOL_WRITE_FILE) {
    const char *b = tc->body;
    while (*b && isspace((unsigned char)*b)) b++;
    if (!*b) { snprintf(reason, (size_t)reason_size, "empty-write-content"); return 0; }
    if (!grounded(prompt_norm, tc->body)) {
      snprintf(reason, (size_t)reason_size, "ungrounded-content:%.40s", tc->body);
      return 0;
    }
  }
  if (tc->kind == TOOL_RUN_COMMAND && tc->command[0])
    if (!command_files_grounded(tc->command, prompt_norm, reason, reason_size)) return 0;
  return 1;
}

// Whitespace/case-insensitive string equality.
static int same_norm(const char *a, const char *b) {
  const size_t al = strlen(a) + 1, bl = strlen(b) + 1;
  char *an = (char*)malloc(al), *bn = (char*)malloc(bl);
  int eq = 0;
  if (an && bn) {
    norm_into(a, an, (int)al);
    norm_into(b, bn, (int)bl);
    eq = (strcmp(an, bn) == 0);
  }
  free(an); free(bn);
  return eq;
}

// How close the sampled action is to the reference action, in [0, 1].
//   0.0  wrong tool (or a tool call where prose was wanted, and vice versa)
//   0.4  right tool, wrong primary argument
//   0.7  right tool and primary argument, wrong write body
//   1.0  identical action
// A <done> summary and a prose answer are scored on the ROUTING decision only:
// their text is free-form, and grading free text against a reference string is
// imitation, not verification — that job belongs to the SFT loss, not here.
static float action_match(const ToolCall *got, const ToolCall *want) {
  if (got->kind != want->kind) return 0.0f;
  switch (want->kind) {
    case TOOL_DONE:
    case TOOL_NONE:
      return 1.0f;
    case TOOL_READ_FILE:
      return same_norm(got->path, want->path) ? 1.0f : 0.4f;
    case TOOL_RUN_COMMAND:
      return same_norm(got->command, want->command) ? 1.0f : 0.4f;
    case TOOL_WRITE_FILE:
      if (!same_norm(got->path, want->path)) return 0.4f;
      return same_norm(got->body, want->body) ? 1.0f : 0.7f;
  }
  return 0.0f;
}

// Which kind of action a parsed response is, for the reward-hacking monitor.
// Anything that failed the format term counts as MALFORMED, whatever it parsed as.
static int action_kind(const ToolCall *tc, int had_tag, int format_passed) {
  if (!format_passed) return REWARD_ACT_MALFORMED;
  switch (tc->kind) {
    case TOOL_DONE:        return REWARD_ACT_DONE;
    case TOOL_READ_FILE:   return REWARD_ACT_READ;
    case TOOL_WRITE_FILE:  return REWARD_ACT_WRITE;
    case TOOL_RUN_COMMAND: return REWARD_ACT_RUN;
    case TOOL_NONE:        return had_tag ? REWARD_ACT_MALFORMED : REWARD_ACT_PROSE;
  }
  return REWARD_ACT_MALFORMED;
}

float reward_score(const char *prompt, const char *response,
                   const char *reference, RewardParts *parts) {
  RewardParts p;
  memset(&p, 0, sizeof(p));
  p.action = REWARD_ACT_MALFORMED;

  const int has_ref = (reference && reference[0]);
  // No reference => rescale the two remaining terms so a perfect response still
  // scores 1.0 and the log stays comparable across task files.
  const float wf = has_ref ? W_FORMAT : W_FORMAT / (W_FORMAT + W_GROUND);
  const float wg = has_ref ? W_GROUND : W_GROUND / (W_FORMAT + W_GROUND);

  const size_t rl = strlen(response) + 1;
  char *clean = (char*)malloc(rl);
  const size_t pl = strlen(prompt) + 1;
  char *pnorm = (char*)malloc(pl);
  if (!clean || !pnorm) {                 // OOM: score 0, never crash a long run
    free(clean); free(pnorm);
    if (parts) { snprintf(p.reason, sizeof(p.reason), "oom"); *parts = p; }
    return 0.0f;
  }
  strip_reasoning(response, clean, (int)rl);
  norm_into(prompt, pnorm, (int)pl);

  ToolCall tc;
  toolcall_parse(clean, &tc);
  // toolcall_parse reports TOOL_NONE both for "no block at all" (a prose answer)
  // and for "<tool name=...>" naming a tool that does not exist. Only the former
  // is a legitimate action, so an unknown tool name is a format failure.
  const int had_tag = (strstr(clean, "<tool") != NULL || strstr(clean, "<done") != NULL);

  if (tc.kind == TOOL_NONE && !had_tag) {
    p.format = wf;                        // a plain prose answer is well-formed
  } else if (format_ok(clean, &tc)) {
    p.format = wf;
  } else {
    snprintf(p.reason, sizeof(p.reason), "malformed-action");
  }

  char why[96];
  why[0] = '\0';
  if (grounding_ok(pnorm, &tc, why, (int)sizeof(why))) {
    p.grounded = wg;
  } else if (!p.reason[0]) {
    snprintf(p.reason, sizeof(p.reason), "%s", why);
  }

  if (has_ref) {
    const size_t fl = strlen(reference) + 1;
    char *rclean = (char*)malloc(fl);
    if (rclean) {
      strip_reasoning(reference, rclean, (int)fl);
      ToolCall ref;
      toolcall_parse(rclean, &ref);
      const float m = action_match(&tc, &ref);
      p.match = W_MATCH * m;
      if (m < 1.0f && !p.reason[0])
        snprintf(p.reason, sizeof(p.reason), "action-mismatch(%.1f)", (double)m);
      free(rclean);
    }
  }

  p.total = p.format + p.grounded + p.match;
  p.action = action_kind(&tc, had_tag, p.format > 0.0f);
  free(clean); free(pnorm);
  if (parts) *parts = p;
  return p.total;
}

// ===========================================================================
// Reward v2 (see reward.h). Everything above this line is v1 and is left as it
// was (plus filling RewardParts.action), so a v1 run reproduces bit for bit.
// ===========================================================================

#define W2_GROUND  0.30f
#define W2_MATCH   0.70f

// Transcript markers a rollout has no business emitting inside an argument: they
// mean the model ran past its own turn and started writing the NEXT one (the
// observation, the user's reply) into its action. Tested on the normalized
// (lowercased) argument, so "Model:" is caught too.
static const char *const MARKERS[] = {"model:", "observation:", "goal:", "user:", NULL};

// 1 if `value` carries a leaked transcript marker or a '<' (the start of a tag —
// a second action, or a half-written closing tag, smuggled inside an attribute
// or a body). Copies a short reason to `why`.
static int has_leak(const char *value, char *why, int why_size) {
  const size_t vl = strlen(value) + 1;
  char *vn = (char*)malloc(vl);
  if (!vn) return 0;
  norm_into(value, vn, (int)vl);
  int leak = 0;
  if (strchr(vn, '<')) {
    snprintf(why, (size_t)why_size, "leaked-tag:%.40s", value);
    leak = 1;
  }
  for (int i = 0; !leak && MARKERS[i]; i++)
    if (strstr(vn, MARKERS[i])) {
      snprintf(why, (size_t)why_size, "leaked-marker:%s", MARKERS[i]);
      leak = 1;
    }
  free(vn);
  return leak;
}

// A path must look like a file: carry an extension dot or a directory slash. This
// is what stops read_file path="list" from passing on the strength of the word
// "list" appearing somewhere in the goal. Exempt only when the REFERENCE path is
// itself bare (Makefile, LICENSE): then a bare path is the expected shape.
static int path_shaped(const char *path) {
  return strchr(path, '.') != NULL || strchr(path, '/') != NULL;
}

// Every argument token of a command must be grounded. The command is split into
// pipeline segments at | ; & (so "ps -aux | grep docker" is two segments, each
// with its own verb). In each segment the FIRST token is the verb and is exempt —
// "list the files" legitimately becomes `ls`, and v1 exempted verbs too — and a
// token beginning with '-' is a flag and is exempt. Every other token (subcommand,
// file, branch, value) must occur in the prompt as a whole token — or in the
// REFERENCE command (`ref_norm`, may be NULL): "show running processes" -> `ps aux`
// is correct although "aux" is nowhere in the prompt, and the reference is
// verified ground truth, not the model's opinion. An invented token that is in
// neither still fails. Returns 1 if all are grounded; else 0 with the offender in
// `bad`.
static int command_args_grounded(const char *cmd, const char *prompt_norm,
                                 const char *ref_norm, char *bad, int bad_size) {
  const size_t cl = strlen(cmd) + 1;
  char *buf = (char*)malloc(cl);
  if (!buf) return 1;                            // OOM: do not fail the rollout
  memcpy(buf, cmd, cl);
  int ok = 1, first_in_segment = 1, any_token = 0;
  char *p = buf;
  while (*p && ok) {
    if (*p == '|' || *p == ';' || *p == '&') { first_in_segment = 1; p++; continue; }
    if (isspace((unsigned char)*p)) { p++; continue; }
    char *start = p;
    while (*p && !isspace((unsigned char)*p) && *p != '|' && *p != ';' && *p != '&') p++;
    const char saved = *p;
    *p = '\0';
    any_token = 1;
    if (first_in_segment) {
      first_in_segment = 0;                      // the verb: canonical, not copied
    } else if (start[0] != '-') {
      if (!grounded(prompt_norm, start) && !(ref_norm && grounded(ref_norm, start))) {
        snprintf(bad, (size_t)bad_size, "ungrounded-cmd-arg:%.40s", start);
        ok = 0;
      }
    }
    *p = saved;
  }
  free(buf);
  if (ok && !any_token) { snprintf(bad, (size_t)bad_size, "empty-command"); ok = 0; }
  return ok;
}

// v2 grounding of a TOOL call's arguments. Stricter than v1 in three ways: leaked
// markers/tags fail, paths must be file-shaped, and every command argument (not
// just the ones with a known file extension) must be grounded.
static int grounding_ok_v2(const char *prompt_norm, const ToolCall *tc,
                           const ToolCall *ref, char *reason, int reason_size) {
  if (tc->kind == TOOL_WRITE_FILE || tc->kind == TOOL_READ_FILE) {
    if (!tc->path[0]) { snprintf(reason, (size_t)reason_size, "empty-path"); return 0; }
    if (has_leak(tc->path, reason, reason_size)) return 0;
    if (!path_shaped(tc->path) && !(ref && ref->path[0] && !path_shaped(ref->path))) {
      snprintf(reason, (size_t)reason_size, "not-a-path:%.48s", tc->path);
      return 0;
    }
    if (!grounded(prompt_norm, tc->path)) {
      snprintf(reason, (size_t)reason_size, "ungrounded-path:%.48s", tc->path);
      return 0;
    }
  }
  if (tc->kind == TOOL_WRITE_FILE) {
    const char *b = tc->body;
    while (*b && isspace((unsigned char)*b)) b++;
    if (!*b) { snprintf(reason, (size_t)reason_size, "empty-write-content"); return 0; }
    if (has_leak(tc->body, reason, reason_size)) return 0;
    if (!grounded(prompt_norm, tc->body)) {
      snprintf(reason, (size_t)reason_size, "ungrounded-content:%.40s", tc->body);
      return 0;
    }
  }
  if (tc->kind == TOOL_RUN_COMMAND) {
    if (has_leak(tc->command, reason, reason_size)) return 0;
    char *rn = NULL;
    if (ref && ref->kind == TOOL_RUN_COMMAND && ref->command[0]) {
      const size_t rl = strlen(ref->command) + 1;
      rn = (char*)malloc(rl);
      if (rn) norm_into(ref->command, rn, (int)rl);
    }
    const int ok = command_args_grounded(tc->command, prompt_norm, rn, reason, reason_size);
    free(rn);
    if (!ok) return 0;
  }
  return 1;
}

// ===========================================================================
// Reward v3 helpers (see reward.h). v3 is v2's body with two switches flipped,
// so v2 itself is untouched and reproduces.
// ===========================================================================

#define W3_REDUNDANT 0.25f

// Character-level LCS F1 of two already-normalized strings. Two-row DP: the
// arguments are short (paths, one-line commands, small write bodies), and the
// 4096-byte body cap bounds the worst case.
static float lcs_f1_norm(const char *a, const char *b) {
  const int al = (int)strlen(a), bl = (int)strlen(b);
  if (al == 0 && bl == 0) return 1.0f;
  if (al == 0 || bl == 0) return 0.0f;
  int *prev = (int*)calloc((size_t)bl + 1, sizeof(int));
  int *cur = (int*)calloc((size_t)bl + 1, sizeof(int));
  if (!prev || !cur) { free(prev); free(cur); return 0.0f; }
  for (int i = 1; i <= al; i++) {
    for (int j = 1; j <= bl; j++) {
      if (a[i - 1] == b[j - 1]) cur[j] = prev[j - 1] + 1;
      else cur[j] = prev[j] > cur[j - 1] ? prev[j] : cur[j - 1];
    }
    int *t = prev; prev = cur; cur = t;
  }
  const int lcs = prev[bl];
  free(prev); free(cur);
  return 2.0f * (float)lcs / (float)(al + bl);
}

float reward_lcs_f1(const char *a, const char *b) {
  const size_t al = strlen(a) + 1, bl = strlen(b) + 1;
  char *an = (char*)malloc(al), *bn = (char*)malloc(bl);
  float f = 0.0f;
  if (an && bn) {
    norm_into(a, an, (int)al);
    norm_into(b, bn, (int)bl);
    f = lcs_f1_norm(an, bn);
  }
  free(an); free(bn);
  return f;
}

// v3's match: like action_match, but the right tool earns 0.3 + 0.7 * argument
// similarity instead of a flat 0.4 / 0.7. An identical action is exactly 1.0
// (F1 of equal normalized strings is 1), so the exact-match threshold holds.
static float action_match_soft(const ToolCall *got, const ToolCall *want) {
  if (got->kind != want->kind) return 0.0f;
  float s = 1.0f;
  switch (want->kind) {
    case TOOL_DONE:
    case TOOL_NONE:        return 1.0f;
    case TOOL_READ_FILE:   s = reward_lcs_f1(got->path, want->path); break;
    case TOOL_RUN_COMMAND: s = reward_lcs_f1(got->command, want->command); break;
    case TOOL_WRITE_FILE:
      s = 0.5f * (reward_lcs_f1(got->path, want->path) +
                  reward_lcs_f1(got->body, want->body));
      break;
  }
  return s >= 1.0f ? 1.0f : 0.3f + 0.7f * s;
}

// Same tool, same normalized arguments.
static int same_action(const ToolCall *a, const ToolCall *b) {
  if (a->kind != b->kind) return 0;
  switch (a->kind) {
    case TOOL_READ_FILE:   return same_norm(a->path, b->path);
    case TOOL_RUN_COMMAND: return same_norm(a->command, b->command);
    case TOOL_WRITE_FILE:  return same_norm(a->path, b->path) && same_norm(a->body, b->body);
    default:               return 1;
  }
}

// Does the transcript in `prompt` already show `tc` as a past MODEL: action?
// Only tool calls count — <done> and prose end a turn, they cannot be repeated.
static int repeats_past_action(const char *prompt, const ToolCall *tc) {
  if (tc->kind == TOOL_DONE || tc->kind == TOOL_NONE) return 0;
  ToolCall *past = (ToolCall*)malloc(sizeof(ToolCall));
  if (!past) return 0;
  int hit = 0;
  for (const char *p = strstr(prompt, "MODEL:"); p && !hit; p = strstr(p + 6, "MODEL:")) {
    // The block right after this marker: parse from here, but only accept it if
    // it starts before the next marker (else it belongs to a later turn).
    const char *next = strstr(p + 6, "MODEL:");
    const char *blk = strstr(p, "<tool");
    if (!blk || (next && blk > next)) continue;
    if (toolcall_parse(blk, past) && same_action(past, tc)) hit = 1;
  }
  free(past);
  return hit;
}

static float score_v2_or_v3(int v3, const char *prompt, const char *response,
                            const char *reference, int capped, RewardParts *parts) {
  RewardParts p;
  memset(&p, 0, sizeof(p));
  p.action = REWARD_ACT_MALFORMED;

  const int has_ref = (reference && reference[0]);
  const size_t rl = strlen(response) + 1;
  char *clean = (char*)malloc(rl);
  const size_t pl = strlen(prompt) + 1;
  char *pnorm = (char*)malloc(pl);
  if (!clean || !pnorm) {                 // OOM: score 0, never crash a long run
    free(clean); free(pnorm);
    if (parts) { snprintf(p.reason, sizeof(p.reason), "oom"); *parts = p; }
    return 0.0f;
  }
  strip_reasoning(response, clean, (int)rl);
  norm_into(prompt, pnorm, (int)pl);

  ToolCall tc;
  toolcall_parse(clean, &tc);
  const int had_tag = (strstr(clean, "<tool") != NULL || strstr(clean, "<done") != NULL);

  ToolCall ref;
  memset(&ref, 0, sizeof(ref));
  ref.kind = TOOL_NONE;
  if (has_ref) {
    const size_t fl = strlen(reference) + 1;
    char *rclean = (char*)malloc(fl);
    if (rclean) {
      strip_reasoning(reference, rclean, (int)fl);
      toolcall_parse(rclean, &ref);
      free(rclean);
    }
  }

  // format: a GATE (0 or 1), not a weight — a malformed action earns nothing.
  int fmt = 0;
  if (tc.kind == TOOL_NONE && !had_tag) fmt = 1;        // plain prose
  else if (format_ok(clean, &tc)) fmt = 1;
  if (!fmt) snprintf(p.reason, sizeof(p.reason), "malformed-action");
  p.format = (float)fmt;
  p.action = action_kind(&tc, had_tag, fmt);

  // grounded: a tool call by its arguments. Prose and <done> carry no arguments,
  // so they are "grounded" only when the reference is the SAME kind — the decision
  // to stop (or to answer in prose) is exactly what has to be right. With no
  // reference there is nothing to check that decision against: half credit.
  float g = 0.0f;
  char why[96];
  why[0] = '\0';
  if (tc.kind == TOOL_DONE || tc.kind == TOOL_NONE) {
    if (!fmt) g = 0.0f;
    else if (!has_ref) g = 0.5f;
    else if (ref.kind == tc.kind) g = 1.0f;
    else snprintf(why, sizeof(why), "%s-but-reference-differs",
                  tc.kind == TOOL_DONE ? "done" : "prose");
  } else if (grounding_ok_v2(pnorm, &tc, has_ref ? &ref : NULL, why, (int)sizeof(why))) {
    g = 1.0f;
  }
  if (g < 1.0f && why[0] && !p.reason[0]) snprintf(p.reason, sizeof(p.reason), "%s", why);
  p.grounded = W2_GROUND * g;

  // match: v1's action_match, reweighted. No reference => grounding is the whole
  // (ungated) remainder, rescaled so a perfect response still scores 1.0.
  float gscale = 1.0f;
  if (has_ref) {
    const float m = v3 ? action_match_soft(&tc, &ref) : action_match(&tc, &ref);
    p.match = W2_MATCH * m;
    if (m < 1.0f && !p.reason[0])
      snprintf(p.reason, sizeof(p.reason), v3 ? "action-mismatch(%.2f)" : "action-mismatch(%.1f)",
               (double)m);
  } else {
    gscale = 1.0f / W2_GROUND;
  }

  p.total = p.format * (p.grounded * gscale + p.match);
  if (v3 && fmt && repeats_past_action(prompt, &tc) &&
      !(has_ref && same_action(&tc, &ref))) {
    p.redundant = W3_REDUNDANT;
    p.total = p.total > W3_REDUNDANT ? p.total - W3_REDUNDANT : 0.0f;
    if (!p.reason[0] || strncmp(p.reason, "action-mismatch", 15) == 0)
      snprintf(p.reason, sizeof(p.reason), "redundant-repeat");
  }
  if (capped) {
    // Cut off at --max-new: whatever was produced is not a finished action, and a
    // truncated rollout must never be a safe score to fall back on.
    p.total = 0.0f;
    snprintf(p.reason, sizeof(p.reason), "capped");
  }
  free(clean); free(pnorm);
  if (parts) *parts = p;
  return p.total;
}

float reward_score_v2(const char *prompt, const char *response,
                      const char *reference, int capped, RewardParts *parts) {
  return score_v2_or_v3(0, prompt, response, reference, capped, parts);
}

float reward_score_v3(const char *prompt, const char *response,
                      const char *reference, int capped, RewardParts *parts) {
  return score_v2_or_v3(1, prompt, response, reference, capped, parts);
}

float reward_score_version(int version, const char *prompt, const char *response,
                           const char *reference, int capped, RewardParts *parts) {
  if (version == 2) return reward_score_v2(prompt, response, reference, capped, parts);
  if (version == 3) return reward_score_v3(prompt, response, reference, capped, parts);
  return reward_score(prompt, response, reference, parts);
}

float reward_match_max(int version) {
  return (version == 2 || version == 3) ? W2_MATCH : W_MATCH;
}

const char *reward_action_name(int action) {
  static const char *const names[REWARD_ACT_COUNT] =
    {"done", "prose", "read", "write", "run", "malformed"};
  return (action >= 0 && action < REWARD_ACT_COUNT) ? names[action] : "?";
}
