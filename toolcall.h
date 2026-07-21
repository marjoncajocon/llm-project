// toolcall.h - Parse tool-call blocks emitted by the model
#ifndef TOOLCALL_H
#define TOOLCALL_H

// The model drives the agent by emitting blocks in its text output:
//   <tool name="write_file" path="a.txt">...file body...</tool>
//   <tool name="read_file" path="a.txt"></tool>
//   <tool name="run_command" command="ls"></tool>
//   <done>final answer / summary</done>
// A response with no such block is treated as a plain final answer.

typedef enum {
  TOOL_NONE,          // no tool block found (plain text answer)
  TOOL_WRITE_FILE,    // path + body
  TOOL_READ_FILE,     // path
  TOOL_RUN_COMMAND,   // command
  TOOL_DONE           // body = final answer
} ToolKind;

typedef struct {
  ToolKind kind;
  char name[64];
  char path[512];
  char command[1024];
  char body[4096];    // inner text (write_file content, or done summary)
} ToolCall;

// Parse the first tool/done block in `text` into *tc. Returns 1 if a block was
// found (tc->kind set accordingly), 0 if none (tc->kind == TOOL_NONE).
int toolcall_parse(const char *text, ToolCall *tc);

#endif // TOOLCALL_H
