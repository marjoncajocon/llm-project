// agent.c - Minimal agent loop with a pluggable model backend

// popen/pclose are POSIX; glibc/musl hide them under -std=c11 without this.
#if !defined(_WIN32) && !defined(__wasi__) && !defined(__wasm__)
#define _POSIX_C_SOURCE 200809L
#endif

#include "agent.h"

#include <stdio.h>
#include <string.h>

#include "toolcall.h"
#include "toolexec.h"

// Portable process pipe (for the command backend). WASM has no process model.
#if defined(__wasi__) || defined(__wasm__)
#define NO_COMMAND_BACKEND 1
#elif defined(_WIN32)
#define POPEN  _popen
#define PCLOSE _pclose
#else
#define POPEN  popen
#define PCLOSE pclose
#endif

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

static int scripted_respond(AgentBackend *self, const char *transcript,
                            char *out, int out_cap) {
  (void)transcript;
  ScriptedBackend *b = (ScriptedBackend*)self;
  if (b->idx >= b->n) { snprintf(out, out_cap, "<done>(script exhausted)</done>"); return 0; }
  snprintf(out, out_cap, "%s", b->responses[b->idx++]);
  return 0;
}

ScriptedBackend scripted_backend(const char **responses, int n) {
  ScriptedBackend b;
  b.base.respond = scripted_respond;
  b.responses = responses;
  b.n = n;
  b.idx = 0;
  return b;
}

static int command_respond(AgentBackend *self, const char *transcript,
                           char *out, int out_cap) {
  CommandBackend *b = (CommandBackend*)self;
#if defined(NO_COMMAND_BACKEND)
  (void)b; (void)transcript;
  snprintf(out, out_cap, "<done>(command backend unavailable on this platform)</done>");
  return 1;
#else
  // Pass the transcript to the program via a temp file argument; read its stdout.
  const char *tmp = "agent_transcript.tmp";
  FILE *w = fopen(tmp, "wb");
  if (!w) { snprintf(out, out_cap, "<done>(cannot write transcript)</done>"); return 1; }
  fwrite(transcript, 1, strlen(transcript), w);
  fclose(w);

  char full[1200];
  snprintf(full, sizeof(full), "%s %s", b->cmd, tmp);
  FILE *p = POPEN(full, "r");
  if (!p) { remove(tmp); snprintf(out, out_cap, "<done>(cannot run backend)</done>"); return 1; }
  int n = (int)fread(out, 1, out_cap - 1, p);
  out[n] = '\0';
  PCLOSE(p);
  remove(tmp);
  return 0;
#endif
}

CommandBackend command_backend(const char *cmd) {
  CommandBackend b;
  b.base.respond = command_respond;
  b.cmd = cmd;
  return b;
}

// ---------------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------------

#define AGENT_MAX_STEPS 12
#define TRANSCRIPT_CAP  16384

static void tappend(char *t, int cap, const char *s) {
  int len = (int)strlen(t);
  snprintf(t + len, cap - len, "%s", s);
}

int agent_run(const char *goal, AgentBackend *backend, ApprovalMode mode) {
  char transcript[TRANSCRIPT_CAP];
  char response[8192];
  char obs[4096];

  snprintf(transcript, sizeof(transcript), "GOAL: %s\n", goal);
  printf("Goal: %s\n\n", goal);

  for (int step = 0; step < AGENT_MAX_STEPS; step++) {
    printf("--- step %d ---\n", step + 1);
    if (backend->respond(backend, transcript, response, sizeof(response)) != 0) {
      printf("backend error; stopping.\n");
      break;
    }
    tappend(transcript, sizeof(transcript), "MODEL: ");
    tappend(transcript, sizeof(transcript), response);
    tappend(transcript, sizeof(transcript), "\n");

    ToolCall tc;
    if (!toolcall_parse(response, &tc)) {
      printf("model (final): %s\n", response);
      break;
    }
    if (tc.kind == TOOL_DONE) {
      printf("done: %s\n", tc.body);
      break;
    }

    char action[1400];
    // run_command executes arbitrary shell input from the model, so it is NEVER
    // auto-approved: it always requires an explicit interactive y/N, even in
    // APPROVAL_AUTO mode (where a non-interactive stdin denies it by default).
    ApprovalMode step_mode = mode;
    switch (tc.kind) {
      case TOOL_WRITE_FILE:
        snprintf(action, sizeof(action), "write_file '%s' (%zu bytes)", tc.path, strlen(tc.body));
        // Show the full content being written so approval is meaningful.
        printf("--- content of %s ---\n%s\n--- end content ---\n", tc.path, tc.body);
        break;
      case TOOL_READ_FILE:
        snprintf(action, sizeof(action), "read_file '%s'", tc.path);
        break;
      case TOOL_RUN_COMMAND:
        snprintf(action, sizeof(action), "run_command: %s", tc.command);
        step_mode = APPROVAL_ASK;
        break;
      default:
        snprintf(action, sizeof(action), "unknown");
        break;
    }
    printf("tool requested: %s\n", action);

    if (!approval_confirm(step_mode, action)) {
      snprintf(obs, sizeof(obs), "denied by user");
    } else {
      toolexec_run(&tc, obs, sizeof(obs));
    }
    printf("observation: %s\n\n", obs);

    tappend(transcript, sizeof(transcript), "OBSERVATION: ");
    tappend(transcript, sizeof(transcript), obs);
    tappend(transcript, sizeof(transcript), "\n");
  }
  return 0;
}
