// approval.c - Approve or deny a tool action before it runs
#include "approval.h"

#include <stdio.h>

int approval_confirm(ApprovalMode mode, const char *action) {
  if (mode == APPROVAL_AUTO) {
    printf("  [auto-approved] %s\n", action);
    return 1;
  }
  printf("  approve? %s  [y/N] ", action);
  fflush(stdout);
  char buf[16];
  if (!fgets(buf, sizeof(buf), stdin)) return 0;
  char *p = buf;
  // Skip a leading UTF-8 BOM (can appear on piped Windows input).
  if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB &&
      (unsigned char)p[2] == 0xBF) {
    p += 3;
  }
  return (p[0] == 'y' || p[0] == 'Y');
}
