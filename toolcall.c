// toolcall.c - Parse tool-call blocks emitted by the model
#include "toolcall.h"

#include <string.h>

// Copy an attribute value: find key="value" within [seg, end) into out.
// Returns 1 if found.
static int get_attr(const char *seg, const char *end, const char *key,
                    char *out, int out_size) {
  out[0] = '\0';
  size_t klen = strlen(key);
  for (const char *p = seg; p + klen + 2 < end; p++) {
    if (strncmp(p, key, klen) == 0 && p[klen] == '=' && p[klen + 1] == '"') {
      const char *v = p + klen + 2;
      int n = 0;
      while (v < end && *v != '"' && n < out_size - 1) out[n++] = *v++;
      out[n] = '\0';
      return 1;
    }
  }
  return 0;
}

// Copy the inner body [start, stop) into out (bounded).
static void copy_body(const char *start, const char *stop, char *out, int out_size) {
  int n = 0;
  for (const char *p = start; p < stop && n < out_size - 1; p++) out[n++] = *p;
  out[n] = '\0';
}

int toolcall_parse(const char *text, ToolCall *tc) {
  memset(tc, 0, sizeof(*tc));
  tc->kind = TOOL_NONE;

  const char *t = strstr(text, "<tool");
  const char *d = strstr(text, "<done");

  // <done>...</done> — final answer
  if (d && (!t || d < t)) {
    const char *gt = strchr(d, '>');
    if (!gt) return 0;
    const char *close = strstr(gt, "</done>");
    copy_body(gt + 1, close ? close : (gt + 1 + strlen(gt + 1)), tc->body, sizeof(tc->body));
    tc->kind = TOOL_DONE;
    return 1;
  }

  if (!t) return 0;
  const char *gt = strchr(t, '>');
  if (!gt) return 0;

  get_attr(t, gt, "name", tc->name, sizeof(tc->name));
  get_attr(t, gt, "path", tc->path, sizeof(tc->path));
  get_attr(t, gt, "command", tc->command, sizeof(tc->command));

  const char *close = strstr(gt, "</tool>");
  copy_body(gt + 1, close ? close : (gt + 1), tc->body, sizeof(tc->body));

  if (strcmp(tc->name, "write_file") == 0) tc->kind = TOOL_WRITE_FILE;
  else if (strcmp(tc->name, "read_file") == 0) tc->kind = TOOL_READ_FILE;
  else if (strcmp(tc->name, "run_command") == 0) tc->kind = TOOL_RUN_COMMAND;
  else tc->kind = TOOL_NONE;

  return tc->kind != TOOL_NONE;
}
