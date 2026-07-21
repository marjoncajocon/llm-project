// agent.h - Minimal agent loop with a pluggable model backend
#ifndef AGENT_H
#define AGENT_H

#include "approval.h"

// A model backend produces the next response given the running transcript.
// Concrete backends embed AgentBackend as their FIRST field (so a pointer to
// the backend can be cast to its concrete type) — no void* needed.
//   respond(): write the model's next response into out (NUL-terminated);
//   return 0 on success, nonzero on failure.
typedef struct AgentBackend AgentBackend;
struct AgentBackend {
  int (*respond)(AgentBackend *self, const char *transcript, char *out, int out_cap);
};

// Scripted backend: returns a fixed list of responses in order (demos/tests).
typedef struct {
  AgentBackend base;
  const char **responses;
  int n;
  int idx;
} ScriptedBackend;
ScriptedBackend scripted_backend(const char **responses, int n);

// Command backend: runs an external program; its stdout is the model's response.
// The transcript is passed as a file-path argument, so you can plug in a real
// LLM via a small script (e.g. `python tools/agent_llm.py`). Not available on
// WebAssembly (no process spawning).
typedef struct {
  AgentBackend base;
  const char *cmd;
} CommandBackend;
CommandBackend command_backend(const char *cmd);

// Run the agent loop: for each step, ask the backend for a response given the
// transcript, parse a tool call, gate it through approval, execute it, and feed
// the observation back into the transcript. A <done> block or a plain-text
// response ends the loop (also capped by a max step count).
int agent_run(const char *goal, AgentBackend *backend, ApprovalMode mode);

#endif // AGENT_H
