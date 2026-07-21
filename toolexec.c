// toolexec.c - Execute a parsed tool call
#include "toolexec.h"

#include <stdio.h>
#include <string.h>

// Portable process pipe (Windows uses the _-prefixed names).
#if defined(_WIN32)
#define POPEN  _popen
#define PCLOSE _pclose
#else
#define POPEN  popen
#define PCLOSE pclose
#endif

int toolexec_run(const ToolCall *tc, char *obs, int obs_size) {
  switch (tc->kind) {
    case TOOL_WRITE_FILE: {
      FILE *f = fopen(tc->path, "wb");
      if (!f) { snprintf(obs, obs_size, "error: cannot write '%s'", tc->path); return 1; }
      size_t n = strlen(tc->body);
      fwrite(tc->body, 1, n, f);
      fclose(f);
      snprintf(obs, obs_size, "wrote %zu bytes to %s", n, tc->path);
      return 0;
    }
    case TOOL_READ_FILE: {
      FILE *f = fopen(tc->path, "rb");
      if (!f) { snprintf(obs, obs_size, "error: cannot read '%s'", tc->path); return 1; }
      int n = (int)fread(obs, 1, obs_size - 1, f);
      obs[n] = '\0';
      fclose(f);
      return 0;
    }
    case TOOL_RUN_COMMAND: {
      FILE *p = POPEN(tc->command, "r");
      if (!p) { snprintf(obs, obs_size, "error: cannot run '%s'", tc->command); return 1; }
      int n = (int)fread(obs, 1, obs_size - 1, p);
      obs[n] = '\0';
      PCLOSE(p);
      return 0;
    }
    default:
      snprintf(obs, obs_size, "error: unknown tool");
      return 1;
  }
}
