// risk.h - Deterministic risk classifier for shell commands (agent run_command)
//
// DEFENCE IN DEPTH, NOT A SANDBOX. This looks at the command TEXT only and
// sorts it into SAFE / REVIEW / BLOCK so the approval prompt can warn the
// human, demand a typed "yes" for BLOCK, and refuse BLOCK outright in --auto
// mode. A classifier over strings can never be complete: a shell can build a
// command at run time (variables, aliases, scripts written then executed,
// interpreters, encodings) in ways no static reading of the text can follow.
// The real containment is elsewhere: the per-run workspace as cwd, the
// process limits in toolexec.c, and above all a human who reads the prompt.
// Where the text is ambiguous the classifier fails CLOSED (an unknown command
// is REVIEW, an unanalyzable one BLOCK), never open.
//
// Levels:
//   RISK_SAFE    read-only and confined: ls/dir, cat/type, head, grep, wc,
//                pwd, echo without redirection, git status/log/diff/show ...
//   RISK_REVIEW  changes something recoverable or reaches outside: writes,
//                moves, copies, mkdir, redirection into the workspace, package
//                installs, builds, network tools, interpreters, and every
//                command the classifier does not recognize (fail closed).
//   RISK_BLOCK   destroys data or the system, or escapes the workspace:
//                recursive/forced/wildcard deletes (rm -rf, del /s, rd /s,
//                Remove-Item -Recurse, find -delete, xargs rm), format/mkfs/dd
//                of=, git reset --hard / clean -f / push --force / checkout -- .
//                / restore ., chmod/chown -R, download piped into a shell,
//                shutdown/reboot, reg delete, and any absolute, home (~), drive
//                or '..'-escaping path (including redirection targets).
//
// How: the text is tokenized three times -- POSIX sh, cmd.exe and PowerShell
// quoting rules -- because the same string means different things to each
// (a POSIX-quoted ';' is a real separator to cmd.exe); the highest level wins.
// Each tokenization is split into subcommands on ; & && || | newlines and
// parentheses/braces; each subcommand is classified after stripping env-var
// prefixes (A=1 cmd) and wrappers (sudo, env, nice, timeout, xargs, nohup,
// start ...). Command substitutions $(...) / `...` / <(...) and the command
// strings given to sh -c, cmd /c, powershell -Command, eval and iex are
// classified recursively (max level wins).
#ifndef RISK_H
#define RISK_H

typedef enum {
  RISK_SAFE = 0,
  RISK_REVIEW = 1,
  RISK_BLOCK = 2
} RiskLevel;

// Classify one command line. Writes a short human-readable reason for the
// level into why (may be NULL) -- derived from the command text only, never
// from anything the model said about it. NULL/empty cmd -> SAFE.
RiskLevel risk_classify(const char *cmd, char *why, int why_cap);

// "SAFE" / "REVIEW" / "BLOCK".
const char *risk_level_name(RiskLevel level);

#endif // RISK_H
