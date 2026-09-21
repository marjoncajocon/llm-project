// toolexec.h - Execute a parsed tool call, confined to a workspace directory
#ifndef TOOLEXEC_H
#define TOOLEXEC_H

#include "toolcall.h"

// Where and how tool calls run.
//   workspace  : root directory for every tool path and the cwd of run_command.
//                NULL/"" means the current directory (callers should normally
//                pass a dedicated directory; see toolexec_workspace_init).
//   auto_mode  : 1 when tool calls are auto-approved (no human looks at them).
//                Enables the extra write policy (toolexec_write_policy).
//   timeout_s  : wall-clock limit for run_command (<= 0 means the default, 60).
typedef struct {
  const char *workspace;
  int auto_mode;
  int timeout_s;
} ToolExecPolicy;

// Execute a tool call, writing a human-readable observation (the tool's result,
// or an error message) into obs as a NUL-terminated string. Oversized output is
// cut with a "[truncated N bytes]" note; run_command appends "[exit status N]"
// when the command fails and "[timed out ...]" when it is killed.
// Returns 0 on success, nonzero on failure.
int toolexec_run(const ToolCall *tc, const ToolExecPolicy *pol, char *obs, int obs_size);

// Lexical check of a model-supplied relative path. Returns NULL if acceptable,
// else a static string saying why it is refused (absolute / drive / UNC / "..",
// ':' (drive-relative, NTFS alternate data streams), control or wildcard
// characters, Windows reserved device names (CON, NUL.txt, COM1, ...), and
// components ending in '.' or ' ').
const char *toolexec_path_reason(const char *p);

// Extra refusal for writes that are auto-approved: anything inside a .git
// directory, a top-level tools/ directory, or a file with an executable /
// script extension (.py .ps1 .bat .cmd .sh .exe .dll .so ...). Case-insensitive.
// Returns NULL if allowed, else the reason.
const char *toolexec_write_policy(const char *p);

// Create `dir` if missing (one level) and write its canonical absolute path
// into canon. Returns 0 on success.
int toolexec_workspace_init(const char *dir, char *canon, int cap);

#endif // TOOLEXEC_H
