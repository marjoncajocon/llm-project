// agent.c - Minimal agent loop: parse -> approve -> execute -> observe
#include "agent.h"

#include <stdio.h>
#include <string.h>

#include "toolcall.h"
#include "toolexec.h"

int agent_run(const char *goal, const char **responses, int n_responses, ApprovalMode mode) {
  printf("Goal: %s\n\n", goal);
  char obs[4096];

  for (int step = 0; step < n_responses; step++) {
    printf("--- step %d ---\n", step + 1);

    ToolCall tc;
    if (!toolcall_parse(responses[step], &tc)) {
      printf("model (final): %s\n", responses[step]);   // plain-text answer, no tool
      break;
    }
    if (tc.kind == TOOL_DONE) {
      printf("done: %s\n", tc.body);
      break;
    }

    // Describe the requested action for the approval prompt.
    char action[1400];
    switch (tc.kind) {
      case TOOL_WRITE_FILE:
        snprintf(action, sizeof(action), "write_file '%s' (%zu bytes)",
                 tc.path, strlen(tc.body));
        break;
      case TOOL_READ_FILE:
        snprintf(action, sizeof(action), "read_file '%s'", tc.path);
        break;
      case TOOL_RUN_COMMAND:
        snprintf(action, sizeof(action), "run_command: %s", tc.command);
        break;
      default:
        snprintf(action, sizeof(action), "unknown");
        break;
    }
    printf("tool requested: %s\n", action);

    if (!approval_confirm(mode, action)) {
      snprintf(obs, sizeof(obs), "denied by user");
    } else {
      toolexec_run(&tc, obs, sizeof(obs));
    }
    printf("observation: %s\n\n", obs);
    // A real agent would append `obs` to the model context and request the next step.
  }
  return 0;
}
