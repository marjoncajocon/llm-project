// approval.h - Approve or deny a tool action before it runs
#ifndef APPROVAL_H
#define APPROVAL_H

typedef enum {
  APPROVAL_ASK,   // prompt the user y/N on stdin
  APPROVAL_AUTO   // auto-approve (non-interactive / trusted)
} ApprovalMode;

// Ask whether `action` may proceed. Returns 1 to allow, 0 to deny.
int approval_confirm(ApprovalMode mode, const char *action);

#endif // APPROVAL_H
