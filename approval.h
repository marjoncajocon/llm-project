// approval.h - Approve or deny a tool action before it runs
#ifndef APPROVAL_H
#define APPROVAL_H

#include "risk.h"

typedef enum {
  APPROVAL_ASK,   // prompt the user y/N on stdin
  APPROVAL_AUTO   // auto-approve (non-interactive / trusted)
} ApprovalMode;

// Ask whether `action` may proceed. Returns 1 to allow, 0 to deny.
int approval_confirm(ApprovalMode mode, const char *action);

// Approval for a shell command with its risk level (risk_classify). The prompt
// shows the command text, the level and the classifier's reason -- never the
// model's own explanation of why the command is fine, so a persuasive model
// cannot talk its way past the gate. SAFE/REVIEW ask y/N; BLOCK requires the
// full word "yes"; in APPROVAL_AUTO a BLOCK command is refused without asking.
// Returns 1 to allow, 0 to deny.
int approval_confirm_risk(ApprovalMode mode, const char *action, RiskLevel level,
                          const char *why);

// Is `answer` (one line as typed, with or without the newline / a UTF-8 BOM)
// an approval at this risk level? BLOCK: exactly "yes" (any case, surrounding
// blanks ignored). SAFE/REVIEW: starts with 'y' or 'Y'.
int approval_answer_ok(RiskLevel level, const char *answer);

#endif // APPROVAL_H
