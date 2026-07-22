// toolexec.c - Execute a parsed tool call

// popen/pclose are POSIX, not ISO C, so glibc/musl hide them under -std=c11
// unless a feature-test macro is set. WebAssembly (WASI) has no process model,
// so it has no popen at all — guarded below.
#if !defined(_WIN32) && !defined(__wasi__) && !defined(__wasm__)
#define _POSIX_C_SOURCE 200809L
#endif

#include "toolexec.h"

#include <stdio.h>
#include <string.h>

// Portable process pipe. WebAssembly cannot spawn processes.
#if defined(__wasi__) || defined(__wasm__)
#define NO_RUN_COMMAND 1
#elif defined(_WIN32)
#define POPEN  _popen
#define PCLOSE _pclose
#else
#define POPEN  popen
#define PCLOSE pclose
#endif

// Confine model-supplied file paths to the working directory. A path is safe
// only if it is relative (no leading '/' or '\', no drive letter, no UNC) and
// contains no ".." component — so a tool call can never read or overwrite files
// outside the workspace (e.g. C:\Windows\..., ../../.ssh/authorized_keys).
static int path_is_safe(const char *p) {
  if (!p || !p[0]) return 0;
  if (p[0] == '/' || p[0] == '\\') return 0;   // POSIX-absolute, Windows-root, UNC
  if (p[1] == ':') return 0;                    // drive-letter absolute (C:...)
  for (const char *s = p; *s; ) {
    // s is always at the start of a path component here.
    if (s[0] == '.' && s[1] == '.' && (s[2] == '\0' || s[2] == '/' || s[2] == '\\'))
      return 0;                                 // whole ".." component -> traversal
    while (*s && *s != '/' && *s != '\\') s++;   // skip to the separator
    while (*s == '/' || *s == '\\') s++;         // skip the separator(s)
  }
  return 1;
}

int toolexec_run(const ToolCall *tc, char *obs, int obs_size) {
  switch (tc->kind) {
    case TOOL_WRITE_FILE: {
      if (!path_is_safe(tc->path)) {
        snprintf(obs, obs_size, "error: refused unsafe path '%s' (must be relative and stay within the working directory)", tc->path);
        return 1;
      }
      FILE *f = fopen(tc->path, "wb");
      if (!f) { snprintf(obs, obs_size, "error: cannot write '%s'", tc->path); return 1; }
      size_t n = strlen(tc->body);
      fwrite(tc->body, 1, n, f);
      fclose(f);
      snprintf(obs, obs_size, "wrote %zu bytes to %s", n, tc->path);
      return 0;
    }
    case TOOL_READ_FILE: {
      if (!path_is_safe(tc->path)) {
        snprintf(obs, obs_size, "error: refused unsafe path '%s' (must be relative and stay within the working directory)", tc->path);
        return 1;
      }
      FILE *f = fopen(tc->path, "rb");
      if (!f) { snprintf(obs, obs_size, "error: cannot read '%s'", tc->path); return 1; }
      int n = (int)fread(obs, 1, obs_size - 1, f);
      obs[n] = '\0';
      fclose(f);
      return 0;
    }
    case TOOL_RUN_COMMAND: {
#if defined(NO_RUN_COMMAND)
      snprintf(obs, obs_size, "error: run_command is not supported on WebAssembly");
      return 1;
#else
      FILE *p = POPEN(tc->command, "r");
      if (!p) { snprintf(obs, obs_size, "error: cannot run '%s'", tc->command); return 1; }
      int n = (int)fread(obs, 1, obs_size - 1, p);
      obs[n] = '\0';
      PCLOSE(p);
      return 0;
#endif
    }
    default:
      snprintf(obs, obs_size, "error: unknown tool");
      return 1;
  }
}
