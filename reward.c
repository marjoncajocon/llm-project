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

float reward_score(const char *prompt, const char *response,
                   const char *reference, RewardParts *parts) {
  RewardParts p;
  memset(&p, 0, sizeof(p));

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
  free(clean); free(pnorm);
  if (parts) *parts = p;
  return p.total;
}
