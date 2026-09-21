// approval.c - Approve or deny a tool action before it runs
#include "approval.h"

#include <stdio.h>

// Skip a leading UTF-8 BOM (can appear on piped Windows input).
static const char *skip_bom(const char *p) {
  if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF)
    return p + 3;
  return p;
}

// One answer line; the rest of an over-long line is drained so it cannot
// answer the NEXT prompt. Returns 0 on EOF.
static int read_answer(char *buf, int cap) {
  if (!fgets(buf, cap, stdin)) return 0;
  int nl = 0;
  for (const char *p = buf; *p; p++) if (*p == '\n') nl = 1;
  if (!nl) {
    int c;
    while ((c = getchar()) != EOF && c != '\n') {}
  }
  return 1;
}

int approval_confirm(ApprovalMode mode, const char *action) {
  if (mode == APPROVAL_AUTO) {
    printf("  [auto-approved] %s\n", action);
    return 1;
  }
  printf("  approve? %s  [y/N] ", action);
  fflush(stdout);
  char buf[16];
  if (!read_answer(buf, (int)sizeof(buf))) return 0;
  return approval_answer_ok(RISK_SAFE, buf);
}

int approval_answer_ok(RiskLevel level, const char *answer) {
  if (!answer) return 0;
  const char *p = skip_bom(answer);
  if (level != RISK_BLOCK) return p[0] == 'y' || p[0] == 'Y';
  while (*p == ' ' || *p == '\t') p++;
  const char *yes = "yes";
  for (int i = 0; i < 3; i++, p++) {
    char c = *p;
    if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
    if (c != yes[i]) return 0;
  }
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
  return *p == '\0';
}

int approval_confirm_risk(ApprovalMode mode, const char *action, RiskLevel level,
                          const char *why) {
  printf("  risk: %s -- %s\n", risk_level_name(level), why ? why : "");
  if (level == RISK_BLOCK && mode == APPROVAL_AUTO) {
    printf("  [refused] BLOCK-level commands are never run in --auto mode: %s\n", action);
    return 0;
  }
  if (mode == APPROVAL_AUTO) {
    printf("  [auto-approved] %s\n", action);
    return 1;
  }
  char buf[64];
  if (level == RISK_BLOCK) {
    printf("  approve? %s\n  This command is classified BLOCK. Type the full word 'yes' to run it: ",
           action);
    fflush(stdout);
    if (!read_answer(buf, (int)sizeof(buf))) return 0;
    return approval_answer_ok(RISK_BLOCK, buf);
  }
  printf("  approve? %s  [y/N] ", action);
  fflush(stdout);
  if (!read_answer(buf, (int)sizeof(buf))) return 0;
  return approval_answer_ok(level, buf);
}
