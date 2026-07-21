// toolexec.h - Execute a parsed tool call
#ifndef TOOLEXEC_H
#define TOOLEXEC_H

#include "toolcall.h"

// Execute a tool call, writing a human-readable observation (the tool's result,
// or an error message) into obs as a NUL-terminated string.
// Returns 0 on success, nonzero on failure.
int toolexec_run(const ToolCall *tc, char *obs, int obs_size);

#endif // TOOLEXEC_H
