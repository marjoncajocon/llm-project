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
//
// Safety: every tool path resolves inside cfg->workspace (see toolexec.h); the
// model's text and every observation are stripped of terminal control
// sequences before they are printed or stored; each observation is wrapped in
// a per-run nonce delimiter and lines inside it that look like transcript role
// markers (MODEL:, GOAL:, OBSERVATION:, USER:, ...) are neutralized, so tool
// output cannot impersonate the goal or a model turn. A transcript that would
// overflow stops the run with an error (never silently truncated).
// Returns 0 on a normal end, 2 if the transcript filled up.
typedef struct {
  ApprovalMode mode;          // APPROVAL_ASK (default) or APPROVAL_AUTO
  const char *workspace;      // tool root; NULL/"" = current directory
  int timeout_s;              // run_command limit (<= 0: default 60 s)
  int quiet;                  // 1: no console output (tests)
  char *transcript_out;       // optional: receives the final transcript
  int transcript_out_cap;
} AgentConfig;

AgentConfig agent_default_config(void);
int agent_run(const char *goal, AgentBackend *backend, const AgentConfig *cfg);

// --- exposed for tests ---
// Strip C0/C1 control characters (keeping \n and \t), DEL, bidi overrides
// and invalid UTF-8 (-> '?') in place.
void agent_sanitize(char *s);
// Format one observation into out:
//   "OBSERVATION: [[obs:NONCE]]\n<sanitized, neutralized text>\n[[/obs:NONCE]]\n"
// Returns 0, or -1 if it did not fit in cap.
int agent_format_observation(const char *obs, const char *nonce, char *out, int cap);

#endif // AGENT_H
