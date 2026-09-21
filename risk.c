// risk.c - Deterministic risk classifier for shell commands (see risk.h)
//
// Defence in depth, not a sandbox: see the caveats in risk.h. Everything here
// is a pure function of the command string (no filesystem, no environment).
#include "risk.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RISK_MAX_DEPTH 6       // nested $(...), sh -c "...", eval ... levels
#define RISK_MAX_INPUT 65536   // longer command lines are not analyzed (BLOCK)

#define RISK_MAX_WORK 200000  // classification steps per command line (then BLOCK)

typedef struct {
  RiskLevel lvl;
  char why[200];
  long *work;            // shared step counter: bounds pathological inputs
} Verdict;

// Raise v to level l with a reason. The first reason recorded at the final
// level is the one reported (a SAFE reason only fills an empty slot).
static void raise_to(Verdict *v, RiskLevel l, const char *fmt, ...) {
  if (l < v->lvl) return;
  if (l == v->lvl && v->why[0]) return;
  v->lvl = l;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(v->why, sizeof(v->why), fmt, ap);
  va_end(ap);
}

// ---------------------------------------------------------------------------
// Small string helpers (ASCII case folding only; commands are matched ASCII)
// ---------------------------------------------------------------------------

static int lc(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
static int is_alpha(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static int is_digit(int c) { return c >= '0' && c <= '9'; }
static int is_alnum(int c) { return is_alpha(c) || is_digit(c); }

static int eq_ci(const char *a, const char *b) {
  for (; *a && *b; a++, b++) if (lc((unsigned char)*a) != lc((unsigned char)*b)) return 0;
  return *a == *b;
}

static int starts_ci(const char *s, const char *pre) {
  for (; *pre; s++, pre++) if (lc((unsigned char)*s) != lc((unsigned char)*pre)) return 0;
  return 1;
}

// p (non-empty) is a case-insensitive prefix of full: PowerShell parameter
// abbreviation (-r / -rec / -Recurse all mean -Recurse).
static int abbrev_of(const char *p, const char *full) {
  if (!*p) return 0;
  size_t n = strlen(p);
  if (n > strlen(full)) return 0;
  for (size_t i = 0; i < n; i++) if (lc((unsigned char)p[i]) != full[i]) return 0;
  return 1;
}

static int contains_ci(const char *hay, const char *needle) {
  for (; *hay; hay++) if (starts_ci(hay, needle)) return 1;
  return 0;
}

// Short POSIX option bundle ("-rf", "-Rv") containing any of chars.
static int bundle_has(const char *a, const char *chars) {
  if (a[0] != '-' || a[1] == '-' || !a[1]) return 0;
  for (const char *p = a + 1; *p && *p != '='; p++) {
    if (!is_alpha((unsigned char)*p)) return 0;       // "-1", "-n5": not a letter bundle
  }
  for (const char *p = a + 1; *p && *p != '='; p++) if (strchr(chars, *p)) return 1;
  return 0;
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

// Does this argument name something outside the workspace? NULL if not, else
// why. Lexical only: absolute, home, drive, UNC, registry/file: URLs, and any
// '..' that climbs above the starting directory.
static const char *path_escape(const char *a) {
  if (!a[0]) return NULL;
  if (a[0] == '/' || a[0] == '\\') return "absolute path";
  if (a[0] == '~') return "home-directory path";
  if (is_alpha((unsigned char)a[0]) && a[1] == ':') return "drive-absolute path";
  if (starts_ci(a, "file:")) return "file: URL";
  static const char *const reg[] = { "hklm:", "hkcu:", "hkcr:", "hku:", "hkcc:", "registry::",
                                     "hkey_" };
  for (size_t i = 0; i < sizeof(reg) / sizeof(reg[0]); i++)
    if (starts_ci(a, reg[i])) return "registry path";
  int depth = 0;
  const char *p = a;
  while (*p) {
    const char *q = p;
    while (*q && *q != '/' && *q != '\\') q++;
    size_t L = (size_t)(q - p);
    if (L == 2 && p[0] == '.' && p[1] == '.') {
      if (--depth < 0) return "'..' escapes the workspace";
    } else if (!(L == 0 || (L == 1 && p[0] == '.'))) {
      depth++;
    }
    p = *q ? q + 1 : q;
  }
  return NULL;
}

// The workspace itself (or everything in it) as a delete target.
static int is_whole_dir(const char *a) {
  static const char *const w[] = { ".", "./", ".\\", "./.", "*", "./*", ".\\*", "*.*", "", ".//" };
  for (size_t i = 0; i < sizeof(w) / sizeof(w[0]); i++) if (strcmp(a, w[i]) == 0) return 1;
  size_t n = strlen(a);
  if (n >= 2 && a[n - 1] == '.' && (a[n - 2] == '/' || a[n - 2] == '\\')) return 1;
  return 0;
}

// Output-redirection sinks that are not files.
static int is_null_sink(const char *a) {
  return eq_ci(a, "/dev/null") || eq_ci(a, "/dev/stdout") || eq_ci(a, "/dev/stderr") ||
         eq_ci(a, "nul") || eq_ci(a, "$null") || eq_ci(a, "con");
}

// Windows-style "/x" switch (dir /s, del /q, robocopy /MIR, shutdown /t).
// Common POSIX root directories are never switches.
static int is_win_switch(const char *a) {
  if (a[0] != '/' || !a[1]) return 0;
  char body[24];
  size_t n = 0;
  for (const char *p = a + 1; *p && *p != ':'; p++) {
    if (*p == '/' || *p == '\\' || *p == '.') return 0;
    if (n + 1 >= sizeof(body)) return 0;
    body[n++] = (char)lc((unsigned char)*p);
  }
  body[n] = '\0';
  if (n == 0) return 0;
  static const char *const roots[] = { "tmp", "etc", "usr", "var", "bin", "dev", "opt", "srv",
                                       "mnt", "sys", "lib", "run", "home", "root", "proc", "boot",
                                       "sbin", "media", "users", "windows" };
  for (size_t i = 0; i < sizeof(roots) / sizeof(roots[0]); i++) if (strcmp(body, roots[i]) == 0) return 0;
  if (n <= 3) return 1;
  static const char *const longsw[] = { "mir", "purge", "move", "grant", "deny", "remove", "reset",
                                        "inheritance", "create", "delete", "query", "change", "add",
                                        "domain", "exclude", "copyall", "copy", "enum", "deletevalue",
                                        "setintegritylevel", "save", "restore", "all", "release",
                                        "renew", "flushdns", "displaydns", "wait", "high", "normal",
                                        "min", "max", "shadow", "quiet" };
  for (size_t i = 0; i < sizeof(longsw) / sizeof(longsw[0]); i++) if (strcmp(body, longsw[i]) == 0) return 1;
  return 0;
}

// ---------------------------------------------------------------------------
// Command table
// ---------------------------------------------------------------------------

enum {
  K_SAFE,     // read-only: args are path-checked
  K_TEXT,     // prints its arguments (echo): args are not paths
  K_PATTERN,  // read-only, first positional is a pattern (grep, sed)
  K_REVIEW,   // writes / moves / installs / builds
  K_NET,      // network access
  K_BLOCK,    // always destructive (format, mkfs, shutdown ...)
  K_DELETE,   // rm / del / rd / Remove-Item ...
  K_GIT,
  K_FIND,
  K_WRAP,     // runs another command (sudo, env, xargs, nice ...)
  K_KEYWORD,  // shell keyword that precedes a command (if, then, do ...)
  K_NOOP,     // shell syntax with no command of its own (for, case, fi ...)
  K_SHELL,    // sh / bash / source / .
  K_CMDEXE,   // cmd
  K_PWSH,     // powershell / pwsh
  K_INTERP,   // python / perl / node ...
  K_EVAL      // eval / iex / Invoke-Expression
};

#define O_WIN    1u   // takes /x switches
#define O_FETCH  2u   // downloads content
#define O_DECODE 4u   // decodes content (base64 ...)
#define O_ELEV   8u   // privilege elevation
#define O_VER   16u   // "<tool> --version" is SAFE
#define O_ECHO  32u   // prints its arguments to stdout

typedef struct {
  const char *name;
  unsigned char kind;
  unsigned char opt;
} CmdInfo;

static const CmdInfo CMDS[] = {
  // --- read-only ---
  {"ls", K_SAFE, 0}, {"dir", K_SAFE, O_WIN}, {"cat", K_SAFE, 0}, {"type", K_SAFE, 0},
  {"head", K_SAFE, 0}, {"tail", K_SAFE, 0}, {"less", K_SAFE, 0}, {"more", K_SAFE, O_WIN},
  {"pwd", K_SAFE, 0}, {"cd", K_SAFE, O_WIN}, {"chdir", K_SAFE, O_WIN}, {"pushd", K_SAFE, 0},
  {"popd", K_SAFE, 0}, {"wc", K_SAFE, 0}, {"cut", K_SAFE, 0}, {"tr", K_TEXT, 0},
  {"nl", K_SAFE, 0}, {"file", K_SAFE, 0}, {"stat", K_SAFE, 0}, {"du", K_SAFE, 0},
  {"df", K_SAFE, 0}, {"tree", K_SAFE, O_WIN}, {"which", K_SAFE, 0}, {"where", K_SAFE, O_WIN},
  {"whereis", K_SAFE, 0}, {"whoami", K_SAFE, O_WIN}, {"hostname", K_SAFE, 0}, {"uname", K_SAFE, 0},
  {"printenv", K_SAFE, 0}, {"id", K_SAFE, 0}, {"basename", K_TEXT, 0}, {"dirname", K_TEXT, 0},
  {"realpath", K_SAFE, 0}, {"readlink", K_SAFE, 0}, {"md5sum", K_SAFE, 0}, {"sha1sum", K_SAFE, 0},
  {"sha256sum", K_SAFE, 0}, {"sha512sum", K_SAFE, 0}, {"cksum", K_SAFE, 0}, {"b2sum", K_SAFE, 0},
  {"diff", K_SAFE, 0}, {"cmp", K_SAFE, 0}, {"comm", K_SAFE, 0}, {"test", K_SAFE, 0},
  {"[", K_SAFE, 0}, {"[[", K_SAFE, 0}, {"]", K_NOOP, 0}, {"]]", K_NOOP, 0},
  {"true", K_SAFE, 0}, {"false", K_SAFE, 0}, {":", K_TEXT, 0}, {"sleep", K_SAFE, 0},
  {"ver", K_SAFE, 0}, {"vol", K_SAFE, 0}, {"column", K_SAFE, 0}, {"fold", K_SAFE, 0},
  {"fmt", K_SAFE, 0}, {"rev", K_SAFE, 0}, {"tac", K_SAFE, 0}, {"od", K_SAFE, 0},
  {"hexdump", K_SAFE, 0}, {"strings", K_SAFE, 0}, {"man", K_TEXT, 0}, {"help", K_TEXT, 0},
  {"clear", K_SAFE, 0}, {"cls", K_SAFE, 0}, {"seq", K_TEXT, 0}, {"nproc", K_SAFE, 0},
  {"free", K_SAFE, 0}, {"uptime", K_SAFE, 0}, {"lsblk", K_SAFE, 0}, {"lscpu", K_SAFE, 0},
  {"netstat", K_SAFE, O_WIN}, {"ss", K_SAFE, 0}, {"ps", K_SAFE, 0}, {"tasklist", K_SAFE, O_WIN},
  {"systeminfo", K_SAFE, O_WIN}, {"ipconfig", K_SAFE, O_WIN}, {"history", K_SAFE, 0},
  {"jobs", K_SAFE, 0}, {"base64", K_SAFE, O_DECODE}, {"xxd", K_SAFE, O_DECODE},
  {"sort", K_SAFE, O_WIN}, {"uniq", K_SAFE, 0}, {"date", K_TEXT, 0}, {"cal", K_SAFE, 0},
  {"attrib", K_SAFE, O_WIN}, {"crontab", K_SAFE, 0}, {"locale", K_SAFE, 0}, {"groups", K_SAFE, 0},
  {"w", K_SAFE, 0}, {"who", K_SAFE, 0}, {"last", K_SAFE, 0}, {"env", K_WRAP, 0},
  {"get-childitem", K_SAFE, 0}, {"gci", K_SAFE, 0}, {"get-content", K_SAFE, 0}, {"gc", K_SAFE, 0},
  {"get-location", K_SAFE, 0}, {"gl", K_SAFE, 0}, {"get-process", K_SAFE, 0}, {"gps", K_SAFE, 0},
  {"measure-object", K_SAFE, 0}, {"measure", K_SAFE, 0}, {"get-item", K_SAFE, 0}, {"gi", K_SAFE, 0},
  {"test-path", K_SAFE, 0}, {"get-date", K_SAFE, 0}, {"resolve-path", K_SAFE, 0},
  {"get-filehash", K_SAFE, 0}, {"sort-object", K_SAFE, 0}, {"select-object", K_SAFE, 0},
  {"select", K_SAFE, 0}, {"where-object", K_SAFE, 0}, {"foreach-object", K_SAFE, 0},
  {"%", K_SAFE, 0}, {"?", K_SAFE, 0}, {"format-table", K_SAFE, 0}, {"ft", K_SAFE, 0},
  {"format-list", K_SAFE, 0}, {"fl", K_SAFE, 0}, {"format-wide", K_SAFE, 0},
  {"out-string", K_SAFE, 0}, {"out-host", K_SAFE, 0}, {"get-command", K_SAFE, 0},
  {"gcm", K_SAFE, 0}, {"get-help", K_SAFE, 0}, {"get-alias", K_SAFE, 0}, {"get-member", K_SAFE, 0},
  {"gm", K_SAFE, 0}, {"get-psdrive", K_SAFE, 0}, {"get-itemproperty", K_SAFE, 0},
  {"get-acl", K_SAFE, 0}, {"get-service", K_SAFE, 0}, {"get-volume", K_SAFE, 0},
  {"get-computerinfo", K_SAFE, 0}, {"convertto-json", K_SAFE, 0}, {"convertfrom-json", K_SAFE, 0},
  {"get-unique", K_SAFE, 0}, {"group-object", K_SAFE, 0}, {"compare-object", K_SAFE, 0},
  {"get-variable", K_SAFE, 0}, {"get-history", K_SAFE, 0}, {"set-location", K_SAFE, 0},
  {"sl", K_SAFE, 0}, {"push-location", K_SAFE, 0}, {"pop-location", K_SAFE, 0},
  // --- print their arguments ---
  {"echo", K_TEXT, O_ECHO}, {"echo.", K_TEXT, O_ECHO}, {"printf", K_TEXT, O_ECHO},
  {"write-output", K_TEXT, O_ECHO}, {"write-host", K_TEXT, O_ECHO}, {"write", K_TEXT, O_ECHO},
  {"set", K_TEXT, 0}, {"export", K_TEXT, 0}, {"unset", K_TEXT, 0}, {"expr", K_TEXT, 0},
  {"title", K_TEXT, 0}, {"color", K_TEXT, 0}, {"chcp", K_TEXT, 0}, {"exit", K_TEXT, 0},
  {"return", K_TEXT, 0}, {"shift", K_TEXT, 0}, {"local", K_TEXT, 0}, {"declare", K_TEXT, 0},
  {"typeset", K_TEXT, 0}, {"readonly", K_TEXT, 0}, {"read", K_TEXT, 0}, {"setlocal", K_TEXT, 0},
  {"endlocal", K_TEXT, 0}, {"goto", K_TEXT, 0}, {"pause", K_TEXT, 0}, {"wait", K_TEXT, 0},
  // --- read-only with a pattern first ---
  {"grep", K_PATTERN, 0}, {"egrep", K_PATTERN, 0}, {"fgrep", K_PATTERN, 0}, {"rg", K_PATTERN, 0},
  {"ag", K_PATTERN, 0}, {"ack", K_PATTERN, 0}, {"findstr", K_PATTERN, O_WIN},
  {"select-string", K_PATTERN, 0}, {"sls", K_PATTERN, 0}, {"sed", K_PATTERN, 0},
  {"jq", K_PATTERN, 0}, {"awk", K_PATTERN, 0}, {"gawk", K_PATTERN, 0}, {"mawk", K_PATTERN, 0},
  // --- writes / moves / installs / builds (REVIEW) ---
  {"mv", K_REVIEW, 0}, {"cp", K_REVIEW, 0}, {"copy", K_REVIEW, O_WIN}, {"xcopy", K_REVIEW, O_WIN},
  {"robocopy", K_REVIEW, O_WIN}, {"move", K_REVIEW, O_WIN}, {"ren", K_REVIEW, O_WIN},
  {"rename", K_REVIEW, O_WIN}, {"mkdir", K_REVIEW, O_WIN}, {"md", K_REVIEW, O_WIN},
  {"touch", K_REVIEW, 0}, {"ln", K_REVIEW, 0}, {"mklink", K_REVIEW, O_WIN}, {"tee", K_REVIEW, 0},
  {"tee-object", K_REVIEW, 0}, {"truncate", K_REVIEW, 0}, {"install", K_REVIEW, 0},
  {"patch", K_REVIEW, 0}, {"tar", K_REVIEW, 0}, {"zip", K_REVIEW, 0}, {"unzip", K_REVIEW, 0},
  {"gzip", K_REVIEW, 0}, {"gunzip", K_REVIEW, 0}, {"7z", K_REVIEW, 0}, {"make", K_REVIEW, O_VER},
  {"cmake", K_REVIEW, O_VER}, {"ninja", K_REVIEW, O_VER}, {"gcc", K_REVIEW, O_VER},
  {"cc", K_REVIEW, O_VER}, {"clang", K_REVIEW, O_VER}, {"g++", K_REVIEW, O_VER},
  {"zig", K_REVIEW, O_VER}, {"javac", K_REVIEW, O_VER}, {"java", K_REVIEW, O_VER},
  {"rustc", K_REVIEW, O_VER}, {"go", K_REVIEW, O_VER}, {"cargo", K_REVIEW, O_VER},
  {"rustup", K_REVIEW, O_VER}, {"npm", K_REVIEW, O_VER}, {"npx", K_REVIEW, O_VER},
  {"yarn", K_REVIEW, O_VER}, {"pnpm", K_REVIEW, O_VER}, {"pip", K_REVIEW, O_VER},
  {"pip3", K_REVIEW, O_VER}, {"conda", K_REVIEW, O_VER}, {"poetry", K_REVIEW, O_VER},
  {"uv", K_REVIEW, O_VER}, {"apt", K_REVIEW, 0}, {"apt-get", K_REVIEW, 0}, {"dnf", K_REVIEW, 0},
  {"yum", K_REVIEW, 0}, {"pacman", K_REVIEW, 0}, {"brew", K_REVIEW, O_VER}, {"choco", K_REVIEW, 0},
  {"winget", K_REVIEW, 0}, {"scoop", K_REVIEW, 0}, {"gem", K_REVIEW, O_VER},
  {"bundle", K_REVIEW, O_VER}, {"dotnet", K_REVIEW, O_VER}, {"mvn", K_REVIEW, O_VER},
  {"gradle", K_REVIEW, O_VER}, {"docker", K_REVIEW, O_VER}, {"podman", K_REVIEW, O_VER},
  {"kubectl", K_REVIEW, 0}, {"terraform", K_REVIEW, O_VER}, {"tsc", K_REVIEW, O_VER},
  {"code", K_REVIEW, 0}, {"vim", K_REVIEW, 0}, {"vi", K_REVIEW, 0}, {"nano", K_REVIEW, 0},
  {"notepad", K_REVIEW, 0}, {"open", K_REVIEW, 0}, {"xdg-open", K_REVIEW, 0},
  {"explorer", K_REVIEW, 0}, {"set-content", K_REVIEW, 0}, {"add-content", K_REVIEW, 0},
  {"out-file", K_REVIEW, 0}, {"new-item", K_REVIEW, 0}, {"ni", K_REVIEW, 0},
  {"copy-item", K_REVIEW, 0}, {"cpi", K_REVIEW, 0}, {"move-item", K_REVIEW, 0},
  {"mi", K_REVIEW, 0}, {"rename-item", K_REVIEW, 0}, {"rni", K_REVIEW, 0},
  {"clear-content", K_REVIEW, 0}, {"clc", K_REVIEW, 0}, {"set-item", K_REVIEW, 0},
  {"set-itemproperty", K_REVIEW, 0}, {"new-itemproperty", K_REVIEW, 0},
  {"remove-itemproperty", K_REVIEW, 0}, {"stop-process", K_REVIEW, 0}, {"kill", K_REVIEW, 0},
  {"killall", K_REVIEW, 0}, {"pkill", K_REVIEW, 0}, {"taskkill", K_REVIEW, O_WIN},
  {"start-process", K_REVIEW, 0}, {"saps", K_REVIEW, 0}, {"setx", K_REVIEW, 0},
  {"alias", K_REVIEW, 0}, {"set-alias", K_REVIEW, 0}, {"chmod", K_REVIEW, 0}, {"chown", K_REVIEW, 0}, {"chgrp", K_REVIEW, 0},
  {"chattr", K_REVIEW, 0}, {"icacls", K_REVIEW, O_WIN}, {"takeown", K_REVIEW, O_WIN},
  {"set-acl", K_REVIEW, 0}, {"mount", K_REVIEW, 0}, {"umount", K_REVIEW, 0},
  {"useradd", K_REVIEW, 0}, {"userdel", K_REVIEW, 0}, {"passwd", K_REVIEW, 0},
  {"net", K_REVIEW, O_WIN}, {"sc", K_REVIEW, O_WIN}, {"schtasks", K_REVIEW, O_WIN},
  {"service", K_REVIEW, 0}, {"systemctl", K_REVIEW, 0}, {"at", K_REVIEW, 0},
  {"reg", K_REVIEW, O_WIN}, {"dd", K_REVIEW, 0}, {"set-executionpolicy", K_REVIEW, 0},
  {"unblock-file", K_REVIEW, 0}, {"expand-archive", K_REVIEW, 0},
  {"compress-archive", K_REVIEW, 0}, {"invoke-item", K_REVIEW, 0}, {"ii", K_REVIEW, 0},
  {"invoke-command", K_REVIEW, 0}, {"icm", K_REVIEW, 0}, {"sqlite3", K_REVIEW, 0},
  {"openssl", K_REVIEW, O_DECODE}, {"uudecode", K_REVIEW, O_DECODE}, {"cipher", K_REVIEW, O_WIN},
  {"vssadmin", K_REVIEW, O_WIN}, {"wbadmin", K_REVIEW, O_WIN}, {"wmic", K_REVIEW, O_WIN},
  {"init", K_REVIEW, 0}, {"mshta", K_REVIEW, 0}, {"rundll32", K_REVIEW, 0},
  {"regsvr32", K_REVIEW, O_WIN}, {"wscript", K_REVIEW, 0}, {"cscript", K_REVIEW, 0},
  // --- network ---
  {"curl", K_NET, O_FETCH | O_VER}, {"wget", K_NET, O_FETCH | O_VER},
  {"invoke-webrequest", K_NET, O_FETCH}, {"iwr", K_NET, O_FETCH},
  {"invoke-restmethod", K_NET, O_FETCH}, {"irm", K_NET, O_FETCH},
  {"start-bitstransfer", K_NET, O_FETCH}, {"bitsadmin", K_NET, O_FETCH | O_WIN},
  {"certutil", K_NET, O_FETCH | O_DECODE | O_WIN}, {"aria2c", K_NET, O_FETCH},
  {"fetch", K_NET, O_FETCH}, {"http", K_NET, O_FETCH}, {"ssh", K_NET, 0}, {"scp", K_NET, 0},
  {"sftp", K_NET, 0}, {"ftp", K_NET, 0}, {"telnet", K_NET, 0}, {"nc", K_NET, 0},
  {"ncat", K_NET, 0}, {"netcat", K_NET, 0}, {"socat", K_NET, 0}, {"ping", K_NET, 0},
  {"tracert", K_NET, 0}, {"traceroute", K_NET, 0}, {"nslookup", K_NET, 0}, {"dig", K_NET, 0},
  {"host", K_NET, 0}, {"rsync", K_NET, 0}, {"test-netconnection", K_NET, 0},
  // --- always destructive ---
  {"format", K_BLOCK, O_WIN}, {"format-volume", K_BLOCK, 0}, {"clear-disk", K_BLOCK, 0},
  {"initialize-disk", K_BLOCK, 0}, {"remove-partition", K_BLOCK, 0}, {"diskpart", K_BLOCK, 0},
  {"mke2fs", K_BLOCK, 0}, {"mkswap", K_BLOCK, 0}, {"fdisk", K_BLOCK, 0}, {"sfdisk", K_BLOCK, 0},
  {"parted", K_BLOCK, 0}, {"wipefs", K_BLOCK, 0}, {"shred", K_BLOCK, 0}, {"sdelete", K_BLOCK, 0},
  {"srm", K_BLOCK, 0}, {"wipe", K_BLOCK, 0}, {"shutdown", K_BLOCK, O_WIN}, {"reboot", K_BLOCK, 0},
  {"halt", K_BLOCK, 0}, {"poweroff", K_BLOCK, 0}, {"stop-computer", K_BLOCK, 0},
  {"restart-computer", K_BLOCK, 0}, {"bcdedit", K_BLOCK, O_WIN}, {"telinit", K_BLOCK, 0},
  {"clear-recyclebin", K_BLOCK, 0},
  // --- deletes ---
  {"rm", K_DELETE, 0}, {"unlink", K_DELETE, 0}, {"del", K_DELETE, 0}, {"erase", K_DELETE, 0},
  {"rd", K_DELETE, 0}, {"rmdir", K_DELETE, 0}, {"remove-item", K_DELETE, 0}, {"ri", K_DELETE, 0},
  {"trash", K_DELETE, 0}, {"trash-put", K_DELETE, 0}, {"rimraf", K_DELETE, 0},
  // --- structure ---
  {"git", K_GIT, 0}, {"find", K_FIND, 0},
  {"sudo", K_WRAP, O_ELEV}, {"doas", K_WRAP, O_ELEV}, {"runas", K_WRAP, O_ELEV | O_WIN},
  {"su", K_WRAP, O_ELEV}, {"pkexec", K_WRAP, O_ELEV}, {"gsudo", K_WRAP, O_ELEV},
  {"nice", K_WRAP, 0}, {"ionice", K_WRAP, 0}, {"nohup", K_WRAP, 0}, {"time", K_WRAP, 0},
  {"timeout", K_WRAP, O_WIN}, {"stdbuf", K_WRAP, 0}, {"watch", K_WRAP, 0}, {"xargs", K_WRAP, 0},
  {"parallel", K_WRAP, 0}, {"command", K_WRAP, 0}, {"builtin", K_WRAP, 0}, {"exec", K_WRAP, 0},
  {"busybox", K_WRAP, 0}, {"start", K_WRAP, O_WIN}, {"call", K_WRAP, 0}, {"strace", K_WRAP, 0},
  {"ltrace", K_WRAP, 0}, {"chroot", K_WRAP, O_ELEV}, {"unbuffer", K_WRAP, 0},
  {"flock", K_WRAP, 0}, {"wsl", K_WRAP, 0}, {"measure-command", K_WRAP, 0},
  {"if", K_KEYWORD, 0}, {"then", K_KEYWORD, 0}, {"do", K_KEYWORD, 0}, {"else", K_KEYWORD, 0},
  {"elif", K_KEYWORD, 0}, {"while", K_KEYWORD, 0}, {"until", K_KEYWORD, 0}, {"!", K_KEYWORD, 0},
  {"for", K_NOOP, 0}, {"case", K_NOOP, 0}, {"esac", K_NOOP, 0}, {"fi", K_NOOP, 0},
  {"done", K_NOOP, 0}, {"function", K_NOOP, 0}, {"foreach", K_NOOP, 0}, {"rem", K_NOOP, 0},
  {"::", K_NOOP, 0}, {"in", K_NOOP, 0}, {"end", K_NOOP, 0}, {"try", K_NOOP, 0},
  {"catch", K_NOOP, 0}, {"finally", K_NOOP, 0}, {"param", K_NOOP, 0},
  {"sh", K_SHELL, 0}, {"bash", K_SHELL, O_VER}, {"zsh", K_SHELL, 0}, {"dash", K_SHELL, 0},
  {"ksh", K_SHELL, 0}, {"fish", K_SHELL, 0}, {"csh", K_SHELL, 0}, {"tcsh", K_SHELL, 0},
  {"ash", K_SHELL, 0}, {"mksh", K_SHELL, 0}, {"source", K_SHELL, 0}, {".", K_SHELL, 0},
  {"cmd", K_CMDEXE, 0},
  {"powershell", K_PWSH, 0}, {"pwsh", K_PWSH, 0}, {"powershell_ise", K_PWSH, 0},
  {"python", K_INTERP, O_VER}, {"python2", K_INTERP, O_VER}, {"python3", K_INTERP, O_VER},
  {"py", K_INTERP, O_VER}, {"perl", K_INTERP, O_VER}, {"ruby", K_INTERP, O_VER},
  {"node", K_INTERP, O_VER}, {"nodejs", K_INTERP, O_VER}, {"php", K_INTERP, O_VER},
  {"lua", K_INTERP, O_VER}, {"deno", K_INTERP, O_VER}, {"bun", K_INTERP, O_VER},
  {"osascript", K_INTERP, 0}, {"tclsh", K_INTERP, 0}, {"rscript", K_INTERP, O_VER},
  {"eval", K_EVAL, 0}, {"iex", K_EVAL, 0}, {"invoke-expression", K_EVAL, 0},
};

static const CmdInfo *lookup(const char *name) {
  for (size_t i = 0; i < sizeof(CMDS) / sizeof(CMDS[0]); i++)
    if (strcmp(CMDS[i].name, name) == 0) return &CMDS[i];
  static const CmdInfo mkfs = {"mkfs", K_BLOCK, 0};
  static const CmdInfo python = {"python", K_INTERP, O_VER};
  if (strncmp(name, "mkfs", 4) == 0) return &mkfs;              // mkfs.ext4, mkfs.vfat ...
  if (strncmp(name, "python", 6) == 0) return &python;          // python3.11 ...
  return NULL;
}

// Command word -> lookup key: lower case, no leading '@' (cmd echo-off), no
// directory, no .exe/.com/.bat/.cmd suffix. "/bin/rm" and "RM.EXE" are rm.
static void normalize_name(const char *w, char *out, size_t cap) {
  while (*w == '@') w++;
  const char *base = w;
  for (const char *p = w; *p; p++) if (*p == '/' || *p == '\\') base = p + 1;
  if (!*base) base = w;                                       // "/" alone
  size_t n = 0;
  for (const char *p = base; *p && n + 1 < cap; p++) out[n++] = (char)lc((unsigned char)*p);
  out[n] = '\0';
  static const char *const ext[] = { ".exe", ".com", ".bat", ".cmd" };
  for (size_t i = 0; i < sizeof(ext) / sizeof(ext[0]); i++) {
    size_t L = strlen(ext[i]);
    if (n > L && strcmp(out + n - L, ext[i]) == 0) { out[n - L] = '\0'; break; }
  }
}

// ---------------------------------------------------------------------------
// Lexer: one command string -> words and operators, under one shell's rules
// ---------------------------------------------------------------------------

enum { M_POSIX, M_CMD, M_PS, M_COUNT };
static const char *const MODE_NAME[M_COUNT] = { "sh", "cmd", "PowerShell" };
// Tokenization order: the shell run_command really uses comes first, so on a
// tie its reading supplies the reason (toolexec.c: cmd.exe on Windows, else sh).
#if defined(_WIN32)
static const int MODE_ORDER[M_COUNT] = { M_CMD, M_PS, M_POSIX };
#else
static const int MODE_ORDER[M_COUNT] = { M_POSIX, M_CMD, M_PS };
#endif

enum { T_WORD, T_SEP, T_PIPE, T_OUT, T_IN, T_DUP, T_HEREDOC };

#define W_VAR    1u   // contains a variable / placeholder expansion
#define W_GLOB   2u   // unquoted wildcard
#define W_SUBST  4u   // contains a command/process substitution
#define W_QUOTED 8u

typedef struct {
  int type;
  char *s;
  unsigned fl;
} Tok;

typedef struct {
  const char *s;
  size_t n;
  int mode;
  int depth;
  Verdict *v;
  Tok *t;
  int nt, cap;
  char *buf;
  size_t bn;
  int inword;
  size_t wstart;
  unsigned wfl;
} Lexer;

static void classify_text(const char *s, size_t n, int depth, Verdict *v);

static void lx_push(Lexer *L, int type, char *s, unsigned fl) {
  if (L->nt >= L->cap) return;                   // cannot happen (cap = n + 8)
  L->t[L->nt].type = type;
  L->t[L->nt].s = s;
  L->t[L->nt].fl = fl;
  L->nt++;
}

static void lx_begin(Lexer *L) {
  if (!L->inword) { L->inword = 1; L->wstart = L->bn; L->wfl = 0; }
}
static void lx_char(Lexer *L, char c) { lx_begin(L); L->buf[L->bn++] = c; }
static void lx_mark(Lexer *L, unsigned f) { lx_begin(L); L->wfl |= f; }
static void lx_end(Lexer *L) {
  if (!L->inword) return;
  L->buf[L->bn++] = '\0';
  lx_push(L, T_WORD, L->buf + L->wstart, L->wfl);
  L->inword = 0;
}
static void lx_op(Lexer *L, int type) { lx_end(L); lx_push(L, type, NULL, 0); }

// Index of the ')' closing a '(' whose body starts at i (quotes skipped), or n.
static size_t match_paren(const char *s, size_t i, size_t n) {
  int d = 1;
  for (; i < n; i++) {
    char c = s[i];
    if (c == '\\' && i + 1 < n) { i++; continue; }
    if (c == '\'') { i++; while (i < n && s[i] != '\'') i++; continue; }
    if (c == '"') { i++; while (i < n && s[i] != '"') { if (s[i] == '\\') i++; i++; } continue; }
    if (c == '(') d++;
    else if (c == ')' && --d == 0) return i;
  }
  return n;
}

// Classify an embedded command (substitution body) and mark the current word.
static void lx_subst(Lexer *L, size_t b, size_t e, const char *what) {
  classify_text(L->s + b, e > b ? e - b : 0, L->depth + 1, L->v);
  raise_to(L->v, RISK_REVIEW, "%s runs a nested command", what);
  lx_char(L, '$');
  lx_char(L, '(');
  lx_char(L, ')');
  lx_mark(L, W_VAR | W_SUBST);
}

// '$' at L->s[i] (POSIX / PowerShell). Returns the index of the last byte used.
static size_t lx_dollar(Lexer *L, size_t i) {
  const char *s = L->s;
  size_t n = L->n;
  if (i + 1 < n && s[i + 1] == '(') {
    size_t e = match_paren(s, i + 2, n);
    lx_subst(L, i + 2, e, "$(...)");
    return e < n ? e : n - 1;
  }
  if (i + 1 < n && s[i + 1] == '{') {
    size_t j = i + 2;
    while (j < n && s[j] != '}') j++;
    for (size_t k = i; k <= j && k < n; k++) lx_char(L, s[k]);
    lx_mark(L, W_VAR);
    return j < n ? j : n - 1;
  }
  if (i + 1 < n && (is_alnum((unsigned char)s[i + 1]) || s[i + 1] == '_')) {
    size_t j = i + 1;
    while (j < n && (is_alnum((unsigned char)s[j]) || s[j] == '_' ||
                     (L->mode == M_PS && s[j] == ':' && j + 1 < n && is_alpha((unsigned char)s[j + 1]))))
      j++;
    for (size_t k = i; k < j; k++) lx_char(L, s[k]);
    lx_mark(L, W_VAR);
    return j - 1;
  }
  if (i + 1 < n && strchr("?#@*!$-", s[i + 1])) {
    lx_char(L, '$');
    lx_char(L, s[i + 1]);
    lx_mark(L, W_VAR);
    return i + 1;
  }
  lx_char(L, '$');
  return i;
}

// '>' or '<' (unquoted) at s[i]. Returns the index of the last byte used.
static size_t lx_redirect(Lexer *L, size_t i) {
  const char *s = L->s;
  size_t n = L->n;
  char c = s[i];
  if (L->mode != M_CMD && i + 1 < n && s[i + 1] == '(') {         // <(...) >(...)
    lx_end(L);
    size_t e = match_paren(s, i + 2, n);
    lx_subst(L, i + 2, e, "process substitution");
    lx_end(L);
    return e < n ? e : n - 1;
  }
  // A word made only of digits (or '*' in PowerShell) right before is an fd.
  if (L->inword) {
    int fd = L->bn > L->wstart && !(L->wfl & W_QUOTED);
    for (size_t k = L->wstart; k < L->bn && fd; k++)
      if (!is_digit((unsigned char)L->buf[k]) && !(L->mode == M_PS && L->buf[k] == '*')) fd = 0;
    if (fd) { L->bn = L->wstart; L->inword = 0; }
    else lx_end(L);
  }
  if (c == '>') {
    size_t j = i + 1;
    if (j < n && s[j] == '>') j++;
    if (j < n && s[j] == '|') j++;
    if (j < n && s[j] == '&') {
      size_t k = j + 1;
      if (k < n && (is_digit((unsigned char)s[k]) || s[k] == '-')) {
        while (k + 1 < n && is_digit((unsigned char)s[k + 1])) k++;
        lx_push(L, T_DUP, NULL, 0);
        return k;
      }
      lx_push(L, T_OUT, NULL, 0);
      return j;
    }
    lx_push(L, T_OUT, NULL, 0);
    return j - 1;
  }
  // '<'
  if (L->mode == M_PS && i + 1 < n && s[i + 1] == '#') {           // <# block comment #>
    size_t j = i + 2;
    while (j + 1 < n && !(s[j] == '#' && s[j + 1] == '>')) j++;
    return j + 1 < n ? j + 1 : n - 1;
  }
  if (i + 1 < n && s[i + 1] == '<') {
    size_t j = i + 2;
    if (j < n && (s[j] == '<' || s[j] == '-')) j++;
    lx_push(L, T_HEREDOC, NULL, 0);
    return j - 1;
  }
  if (i + 1 < n && s[i + 1] == '&') {
    size_t k = i + 2;
    while (k < n && (is_digit((unsigned char)s[k]) || s[k] == '-')) k++;
    lx_push(L, T_DUP, NULL, 0);
    return k - 1;
  }
  if (i + 1 < n && s[i + 1] == '>') { lx_push(L, T_OUT, NULL, 0); return i + 1; }
  lx_push(L, T_IN, NULL, 0);
  return i;
}

static void lex(Lexer *L) {
  const char *s = L->s;
  size_t n = L->n;
  const int m = L->mode;
  const char esc = m == M_POSIX ? '\\' : m == M_CMD ? '^' : '`';
  char q = 0;
  for (size_t i = 0; i < n; i++) {
    char c = s[i];
    if (q == '\'') {                                  // POSIX / PS single quotes: literal
      if (c == '\'') {
        if (m == M_PS && i + 1 < n && s[i + 1] == '\'') { lx_char(L, '\''); i++; }
        else q = 0;
      } else lx_char(L, c);
      continue;
    }
    if (q == '"') {
      if (c == '"') { q = 0; continue; }
      if (m == M_POSIX && c == '\\' && i + 1 < n && strchr("\"\\$`\n", s[i + 1])) {
        if (s[i + 1] != '\n') lx_char(L, s[i + 1]);
        i++;
        continue;
      }
      if (m == M_PS && c == '`' && i + 1 < n) { lx_char(L, s[i + 1]); i++; continue; }
      if (m != M_CMD && c == '$') { i = lx_dollar(L, i); continue; }
      if (m == M_POSIX && c == '`') {
        size_t e = i + 1;
        while (e < n && s[e] != '`') { if (s[e] == '\\') e++; e++; }
        lx_subst(L, i + 1, e < n ? e : n, "`...`");
        i = e < n ? e : n - 1;
        continue;
      }
      if (m == M_CMD && c == '%') lx_mark(L, W_VAR);
      lx_char(L, c);
      continue;
    }
    // --- unquoted ---
    if (c == esc) {
      if (i + 1 < n) {
        if (s[i + 1] == '\n' || s[i + 1] == '\r') { i++; if (i + 1 < n && s[i] == '\r' && s[i + 1] == '\n') i++; continue; }
        lx_char(L, s[i + 1]);
        lx_mark(L, W_QUOTED);
        i++;
      }
      continue;
    }
    if (c == '"') { q = '"'; lx_mark(L, W_QUOTED); continue; }
    if (c == '\'' && m != M_CMD) {
      q = '\'';
      lx_mark(L, W_QUOTED);
      continue;
    }
    if (c == ' ' || c == '\t' || c == '\v' || c == '\f') { lx_end(L); continue; }
    if (c == ';' && m == M_CMD) { lx_end(L); continue; }        // cmd.exe: a delimiter
    if (c == '\n' || c == '\r' || c == ';') { lx_op(L, T_SEP); continue; }
    if (c == '&') {
      if (i + 1 < n && s[i + 1] == '&') { lx_op(L, T_SEP); i++; continue; }
      if (m == M_POSIX && i + 1 < n && s[i + 1] == '>') {         // &> &>>
        lx_end(L);
        i++;
        if (i + 1 < n && s[i + 1] == '>') i++;
        lx_push(L, T_OUT, NULL, 0);
        continue;
      }
      lx_op(L, T_SEP);
      continue;
    }
    if (c == '|') {
      if (i + 1 < n && s[i + 1] == '|') { lx_op(L, T_SEP); i++; continue; }
      if (i + 1 < n && s[i + 1] == '&') i++;                      // |&
      lx_op(L, T_PIPE);
      continue;
    }
    if (c == '(' || c == ')') { lx_op(L, T_SEP); continue; }
    if (c == '{' && !L->inword) {
      if (i + 1 < n && s[i + 1] == '}') {                         // find -exec ... {}
        lx_char(L, '{'); lx_char(L, '}'); lx_mark(L, W_VAR); i++;
        continue;
      }
      lx_op(L, T_SEP);
      continue;
    }
    if (c == '}' && !L->inword) { lx_op(L, T_SEP); continue; }
    if (c == '>' || c == '<') { i = lx_redirect(L, i); continue; }
    if (c == '#' && !L->inword && m != M_CMD) {                    // comment to end of line
      while (i + 1 < n && s[i + 1] != '\n') i++;
      continue;
    }
    if (m != M_CMD && c == '$') {
      if (m == M_POSIX && i + 1 < n && (s[i + 1] == '\'' || s[i + 1] == '"')) {
        lx_begin(L);
        continue;                                    // $'..' / $".." : the quote follows
      }
      i = lx_dollar(L, i);
      continue;
    }
    if (m == M_POSIX && c == '`') {
      size_t e = i + 1;
      while (e < n && s[e] != '`') { if (s[e] == '\\') e++; e++; }
      lx_subst(L, i + 1, e < n ? e : n, "`...`");
      i = e < n ? e : n - 1;
      continue;
    }
    if (m == M_CMD && (c == '%' || c == '!')) lx_mark(L, W_VAR);
    if (c == '*' || c == '?' || (c == '[' && m == M_POSIX)) lx_mark(L, W_GLOB);
    lx_char(L, c);
  }
  lx_end(L);
}

// ---------------------------------------------------------------------------
// Segment classification
// ---------------------------------------------------------------------------

typedef struct {
  Verdict *v;
  int depth;
  int mode;
} Ctx;

typedef struct {
  int pos;          // stage index inside the current pipeline
  int fetch;        // an earlier stage downloads
  int decode;       // an earlier stage decodes (base64 ...)
  char *echo;       // text printed by the previous stage (echo/printf), or NULL
} Pipe;

static void classify_words(Ctx *C, char **w, unsigned *fl, int n, const Pipe *P, unsigned *out);

static int is_assignment(const char *a) {
  if (!is_alpha((unsigned char)a[0]) && a[0] != '_') return 0;
  const char *p = a;
  while (is_alnum((unsigned char)*p) || *p == '_') p++;
  return *p == '=';
}

// Join words [b, n) with single spaces into a fresh string (caller frees).
static char *join_words(char **w, int b, int n) {
  size_t len = 1;
  for (int i = b; i < n; i++) len += strlen(w[i]) + 3;
  char *out = (char*)malloc(len);
  if (!out) return NULL;
  out[0] = '\0';
  size_t o = 0;
  for (int i = b; i < n; i++) {
    if (i > b) out[o++] = ' ';
    size_t L = strlen(w[i]);
    memcpy(out + o, w[i], L);
    o += L;
  }
  out[o] = '\0';
  return out;
}

static void classify_joined(Ctx *C, char **w, int b, int n) {
  char *j = join_words(w, b, n);
  if (!j) { raise_to(C->v, RISK_BLOCK, "out of memory while analyzing"); return; }
  classify_text(j, strlen(j), C->depth + 1, C->v);
  free(j);
}

// Path-check the arguments of a command. skip_pattern: the first positional is
// a pattern/program, not a path. var_lvl: the level for a variable argument.
static void check_args(Ctx *C, const char *name, char **w, unsigned *fl, int n, int start,
                       unsigned opt, int skip_pattern, RiskLevel var_lvl) {
  int ddash = 0, pos = 0;
  for (int i = start; i < n; i++) {
    const char *a = w[i];
    if (!ddash && a[0] == '-' && a[1]) {
      if (strcmp(a, "--") == 0) { ddash = 1; continue; }
      if (skip_pattern && (strcmp(a, "-e") == 0 || strcmp(a, "-f") == 0 || strcmp(a, "--regexp") == 0 ||
                           strcmp(a, "--expression") == 0 || strcmp(a, "-Pattern") == 0 ||
                           eq_ci(a, "-pattern"))) {
        skip_pattern = 0;                           // explicit pattern: positionals are files
        if (i + 1 < n) i++;
        continue;
      }
      const char *eq = strchr(a, '=');
      if (eq) {
        const char *why = path_escape(eq + 1);
        if (why) raise_to(C->v, RISK_BLOCK, "%s: %s in '%s'", name, why, a);
      }
      continue;
    }
    if ((opt & O_WIN) && is_win_switch(a)) continue;
    if (skip_pattern && !pos) { pos = 1; continue; }
    pos = 1;
    const char *why = path_escape(a);
    if (!why && is_assignment(a)) why = path_escape(strchr(a, '=') + 1);
    if (why) { raise_to(C->v, RISK_BLOCK, "%s: %s '%s'", name, why, a); continue; }
    if ((fl[i] & W_VAR) && var_lvl > RISK_SAFE)
      raise_to(C->v, var_lvl, "%s: argument '%s' depends on a variable", name, a);
  }
}

// Something is feeding stdin into an interpreter (curl ... | sh).
static void pipe_interp(Ctx *C, const Pipe *P, const char *name) {
  if (!P || P->pos == 0) {
    raise_to(C->v, RISK_REVIEW, "%s: starts an interpreter", name);
    return;
  }
  if (P->fetch || P->decode) {
    raise_to(C->v, RISK_BLOCK, "%s: %s content piped into an interpreter", name,
             P->fetch ? "downloaded" : "decoded");
    return;
  }
  if (P->echo) classify_text(P->echo, strlen(P->echo), C->depth + 1, C->v);
  raise_to(C->v, RISK_REVIEW, "%s: executes text piped into it", name);
}

// rm / del / rd / rmdir / Remove-Item ...
static void check_delete(Ctx *C, const char *name, char **w, unsigned *fl, int n, const Pipe *P) {
  const int posix = strcmp(name, "rm") == 0 || strcmp(name, "unlink") == 0;
  const int winsw = strcmp(name, "del") == 0 || strcmp(name, "erase") == 0 ||
                    strcmp(name, "rd") == 0 || strcmp(name, "rmdir") == 0;
  int ddash = 0, npos = 0;
  for (int i = 1; i < n; i++) {
    const char *a = w[i];
    if (!ddash && a[0] == '-' && a[1]) {
      if (strcmp(a, "--") == 0) { ddash = 1; continue; }
      if (a[1] == '-') {
        if (starts_ci(a, "--recursive") || starts_ci(a, "--force") || starts_ci(a, "--no-preserve-root") ||
            starts_ci(a, "--dir")) {
          raise_to(C->v, RISK_BLOCK, "%s: recursive/forced delete (%s)", name, a);
        }
        continue;
      }
      if (posix && bundle_has(a, "rRf")) {
        raise_to(C->v, RISK_BLOCK, "%s: recursive/forced delete (%s)", name, a);
        continue;
      }
      char p[40];
      size_t k = 0;
      for (const char *x = a + 1; *x && *x != ':' && k + 1 < sizeof(p); x++) p[k++] = *x;
      p[k] = '\0';
      if (abbrev_of(p, "recurse") || abbrev_of(p, "force"))
        raise_to(C->v, RISK_BLOCK, "%s: recursive/forced delete (%s)", name, a);
      else if (eq_ci(p, "path") || eq_ci(p, "literalpath") || eq_ci(p, "include") || eq_ci(p, "filter"))
        ;                                              // the value is the target: next word
      continue;
    }
    if (winsw && a[0] == '/' && a[1] && (!a[2] || a[2] == ':')) {
      int s = lc((unsigned char)a[1]);
      if (s == 's' || s == 'f') {
        raise_to(C->v, RISK_BLOCK, "%s: %s delete (%s)", name, s == 's' ? "recursive" : "forced", a);
        continue;
      }
      if (s == 'q' || s == 'p' || s == 'a') continue;
    }
    npos++;
    if (fl[i] & W_VAR)
      raise_to(C->v, RISK_BLOCK, "%s: delete target '%s' is a variable/placeholder", name, a);
    else if ((fl[i] & W_GLOB) || strpbrk(a, "*?"))
      raise_to(C->v, RISK_BLOCK, "%s: wildcard delete '%s'", name, a);
    else if (is_whole_dir(a))
      raise_to(C->v, RISK_BLOCK, "%s: deletes the whole working directory", name);
    else {
      const char *why = path_escape(a);
      if (why) raise_to(C->v, RISK_BLOCK, "%s: deletes outside the workspace (%s '%s')", name, why, a);
    }
  }
  if (npos == 0 && P && P->pos > 0)
    raise_to(C->v, RISK_BLOCK, "%s: deletes whatever the pipeline feeds it", name);
  raise_to(C->v, RISK_REVIEW, "%s: deletes files", name);
}

static int arg_is(const char *a, const char *const *set, size_t k) {
  for (size_t i = 0; i < k; i++) if (strcmp(a, set[i]) == 0) return 1;
  return 0;
}
#define ARG_IS(a, ...) arg_is((a), (const char *const[]){ __VA_ARGS__ }, \
                              sizeof((const char *const[]){ __VA_ARGS__ }) / sizeof(const char*))

static int any_arg(char **w, int b, int n, const char *x) {
  for (int i = b; i < n; i++) if (strcmp(w[i], x) == 0) return 1;
  return 0;
}

static int is_broad_pathspec(const char *a, unsigned fl) {
  return is_whole_dir(a) || strcmp(a, ":/") == 0 || strcmp(a, ":") == 0 || (fl & W_GLOB) ||
         strpbrk(a, "*?") != NULL;
}

static void check_git(Ctx *C, char **w, unsigned *fl, int n) {
  int i = 1;
  while (i < n && w[i][0] == '-') {
    const char *a = w[i];
    if (strcmp(a, "-C") == 0 || strcmp(a, "--git-dir") == 0 || strcmp(a, "--work-tree") == 0) {
      if (i + 1 < n) {
        const char *why = path_escape(w[i + 1]);
        if (why) raise_to(C->v, RISK_BLOCK, "git: %s '%s'", why, w[i + 1]);
      }
      i += 2;
      continue;
    }
    if (strcmp(a, "-c") == 0 || strcmp(a, "--namespace") == 0) { i += 2; continue; }
    const char *eq = strchr(a, '=');
    if (eq) {
      const char *why = path_escape(eq + 1);
      if (why) raise_to(C->v, RISK_BLOCK, "git: %s in '%s'", why, a);
    }
    i++;
  }
  if (i >= n) { raise_to(C->v, RISK_SAFE, "git: read-only"); return; }
  char sub[40];
  normalize_name(w[i], sub, sizeof(sub));
  const int b = i + 1;                                    // first argument of the subcommand
  // Arguments: paths are checked; commit messages and grep patterns are not.
  int pos = 0;
  for (int k = b; k < n; k++) {
    const char *a = w[k];
    if (strcmp(a, "-m") == 0 || strcmp(a, "--message") == 0 || strcmp(a, "-e") == 0) { k++; continue; }
    if (a[0] == '-') {
      const char *eq = strchr(a, '=');
      if (eq && !starts_ci(a, "--format") && !starts_ci(a, "--pretty") && !starts_ci(a, "--grep")) {
        const char *why = path_escape(eq + 1);
        if (why) raise_to(C->v, RISK_BLOCK, "git %s: %s in '%s'", sub, why, a);
      }
      continue;
    }
    if (strcmp(sub, "grep") == 0 && !pos) { pos = 1; continue; }
    pos = 1;
    const char *why = path_escape(a);
    if (why) raise_to(C->v, RISK_BLOCK, "git %s: %s '%s'", sub, why, a);
  }

  if (ARG_IS(sub, "status", "log", "diff", "show", "blame", "annotate", "shortlog", "describe",
             "rev-parse", "rev-list", "ls-files", "ls-tree", "cat-file", "grep", "whatchanged",
             "count-objects", "var", "help", "version", "check-ignore", "merge-base", "name-rev",
             "for-each-ref", "show-ref", "verify-commit", "fsck", "show-branch", "cherry",
             "range-diff", "diff-tree", "diff-files", "diff-index", "--version", "--help")) {
    for (int k = b; k < n; k++)
      if (starts_ci(w[k], "--output")) { raise_to(C->v, RISK_REVIEW, "git %s: writes a file (%s)", sub, w[k]); return; }
    raise_to(C->v, RISK_SAFE, "git %s: read-only", sub);
    return;
  }
  if (strcmp(sub, "reset") == 0) {
    if (any_arg(w, b, n, "--hard")) raise_to(C->v, RISK_BLOCK, "git reset --hard: discards uncommitted work");
    raise_to(C->v, RISK_REVIEW, "git reset: moves HEAD / the index");
    return;
  }
  if (strcmp(sub, "clean") == 0) {
    for (int k = b; k < n; k++)
      if (strcmp(w[k], "--dry-run") == 0 || bundle_has(w[k], "n")) {
        raise_to(C->v, RISK_SAFE, "git clean -n: dry run (lists, deletes nothing)");
        return;
      }
    for (int k = b; k < n; k++)
      if (strcmp(w[k], "--force") == 0 || bundle_has(w[k], "fxXd")) {
        raise_to(C->v, RISK_BLOCK, "git clean %s: deletes untracked files", w[k]);
        return;
      }
    raise_to(C->v, RISK_REVIEW, "git clean: deletes untracked files");
    return;
  }
  if (strcmp(sub, "push") == 0) {
    for (int k = b; k < n; k++) {
      const char *a = w[k];
      if (starts_ci(a, "--force") || strcmp(a, "--mirror") == 0 || strcmp(a, "--delete") == 0 ||
          strcmp(a, "--prune") == 0 || bundle_has(a, "fd") ||
          (a[0] == '+' && a[1]) || (a[0] == ':' && a[1])) {
        raise_to(C->v, RISK_BLOCK, "git push %s: force/destructive push rewrites the remote", a);
        return;
      }
    }
    raise_to(C->v, RISK_REVIEW, "git push: publishes commits");
    return;
  }
  if (strcmp(sub, "checkout") == 0 || strcmp(sub, "restore") == 0) {
    int staged_only = 0;
    if (strcmp(sub, "restore") == 0) {
      int st = any_arg(w, b, n, "--staged") || any_arg(w, b, n, "-S");
      int wt = any_arg(w, b, n, "--worktree") || any_arg(w, b, n, "-W");
      staged_only = st && !wt;
    }
    for (int k = b; k < n; k++) {
      const char *a = w[k];
      if (strcmp(a, "--force") == 0 || (strcmp(sub, "checkout") == 0 && bundle_has(a, "f"))) {
        raise_to(C->v, RISK_BLOCK, "git %s %s: discards uncommitted work", sub, a);
        return;
      }
      if (a[0] != '-' && is_broad_pathspec(a, fl[k]) && !staged_only) {
        raise_to(C->v, RISK_BLOCK, "git %s %s: discards every uncommitted change", sub, a);
        return;
      }
    }
    if (strcmp(sub, "checkout") == 0 && any_arg(w, b, n, "--"))
      raise_to(C->v, RISK_REVIEW, "git checkout -- <path>: discards changes to that path");
    raise_to(C->v, RISK_REVIEW, "git %s: changes the working tree", sub);
    return;
  }
  if (strcmp(sub, "switch") == 0) {
    for (int k = b; k < n; k++)
      if (strcmp(w[k], "--force") == 0 || strcmp(w[k], "--discard-changes") == 0 || strcmp(w[k], "-f") == 0) {
        raise_to(C->v, RISK_BLOCK, "git switch %s: discards uncommitted work", w[k]);
        return;
      }
    raise_to(C->v, RISK_REVIEW, "git switch: changes branch");
    return;
  }
  if (strcmp(sub, "branch") == 0) {
    int del = 0, force = 0, other = 0;
    for (int k = b; k < n; k++) {
      const char *a = w[k];
      if (strcmp(a, "-D") == 0 || (a[0] == '-' && a[1] != '-' && strchr(a, 'D'))) { del = force = 1; continue; }
      if (strcmp(a, "-d") == 0 || strcmp(a, "--delete") == 0) { del = 1; continue; }
      if (strcmp(a, "-f") == 0 || strcmp(a, "--force") == 0) { force = 1; continue; }
      if (ARG_IS(a, "-a", "-r", "-v", "-vv", "--list", "-l", "--all", "--remotes", "--show-current",
                 "--merged", "--no-merged", "--contains", "--verbose") || starts_ci(a, "--sort") ||
          starts_ci(a, "--format"))
        continue;
      other = 1;
    }
    if (del && force) raise_to(C->v, RISK_BLOCK, "git branch -D: force-deletes a branch (unmerged commits are lost)");
    if (del || other || force) raise_to(C->v, RISK_REVIEW, "git branch: creates/deletes/renames a branch");
    else raise_to(C->v, RISK_SAFE, "git branch: lists branches");
    return;
  }
  if (strcmp(sub, "stash") == 0) {
    const char *a = b < n ? w[b] : "";
    if (strcmp(a, "clear") == 0) raise_to(C->v, RISK_BLOCK, "git stash clear: discards every stash");
    else if (strcmp(a, "list") == 0 || strcmp(a, "show") == 0) raise_to(C->v, RISK_SAFE, "git stash %s: read-only", a);
    else raise_to(C->v, RISK_REVIEW, "git stash: changes the working tree / stashes");
    return;
  }
  if (strcmp(sub, "tag") == 0 || strcmp(sub, "remote") == 0 || strcmp(sub, "config") == 0 ||
      strcmp(sub, "reflog") == 0 || strcmp(sub, "worktree") == 0 || strcmp(sub, "submodule") == 0) {
    int listing = 1;
    for (int k = b; k < n; k++) {
      const char *a = w[k];
      if (strcmp(sub, "reflog") == 0 && k == b && (strcmp(a, "expire") == 0 || strcmp(a, "delete") == 0)) {
        raise_to(C->v, RISK_BLOCK, "git reflog %s: destroys recovery history", a);
        return;
      }
      if ((strcmp(sub, "worktree") == 0 || strcmp(sub, "submodule") == 0) &&
          (strcmp(a, "-f") == 0 || strcmp(a, "--force") == 0)) {
        raise_to(C->v, RISK_BLOCK, "git %s %s: forced removal discards work", sub, a);
        return;
      }
      if (ARG_IS(a, "-l", "--list", "-n", "-v", "--verbose", "show", "get-url", "list", "status") ||
          starts_ci(a, "--get") || starts_ci(a, "--show") || (strcmp(sub, "tag") == 0 && starts_ci(a, "--sort")))
        continue;
      if (strcmp(sub, "config") == 0 && k > b && listing && (any_arg(w, b, n, "--get") || any_arg(w, b, n, "--get-all")))
        continue;
      listing = 0;
    }
    if (listing) raise_to(C->v, RISK_SAFE, "git %s: read-only", sub);
    else raise_to(C->v, RISK_REVIEW, "git %s: changes repository settings/state", sub);
    return;
  }
  if (strcmp(sub, "rm") == 0) {
    for (int k = b; k < n; k++) {
      if (strcmp(w[k], "--force") == 0 || bundle_has(w[k], "f")) {
        raise_to(C->v, RISK_BLOCK, "git rm %s: deletes files with uncommitted changes", w[k]);
        return;
      }
      if (w[k][0] != '-' && is_broad_pathspec(w[k], fl[k]) && !any_arg(w, b, n, "--cached")) {
        raise_to(C->v, RISK_BLOCK, "git rm %s: deletes every tracked file", w[k]);
        return;
      }
    }
    raise_to(C->v, RISK_REVIEW, "git rm: deletes tracked files");
    return;
  }
  if (strcmp(sub, "filter-branch") == 0 || strcmp(sub, "filter-repo") == 0) {
    raise_to(C->v, RISK_BLOCK, "git %s: rewrites all history", sub);
    return;
  }
  if (strcmp(sub, "update-ref") == 0 && any_arg(w, b, n, "-d")) {
    raise_to(C->v, RISK_BLOCK, "git update-ref -d: deletes a ref");
    return;
  }
  if (strcmp(sub, "gc") == 0 || strcmp(sub, "prune") == 0) {
    for (int k = b; k < n; k++)
      if (starts_ci(w[k], "--prune=now") || starts_ci(w[k], "--prune=all") ||
          (strcmp(sub, "prune") == 0 && strcmp(w[k], "--expire") == 0)) {
        raise_to(C->v, RISK_BLOCK, "git %s %s: permanently drops unreachable commits", sub, w[k]);
        return;
      }
  }
  raise_to(C->v, RISK_REVIEW, "git %s: changes the repository", sub);
}

static void check_find(Ctx *C, char **w, unsigned *fl, int n, const Pipe *P) {
  int preds = 0;
  (void)P;
  for (int i = 1; i < n; i++) {
    const char *a = w[i];
    if (a[0] == '-' && a[1]) {
      preds = 1;
      if (strcmp(a, "-delete") == 0) { raise_to(C->v, RISK_BLOCK, "find -delete: deletes every match"); continue; }
      if (ARG_IS(a, "-exec", "-execdir", "-ok", "-okdir")) {
        int j = i + 1;
        while (j < n && strcmp(w[j], ";") != 0 && strcmp(w[j], "+") != 0 && strcmp(w[j], "\\") != 0 &&
               strcmp(w[j], "\\;") != 0 && strcmp(w[j], "`;") != 0)
          j++;
        if (j > i + 1) {
          Pipe none = {0, 0, 0, NULL};
          unsigned o = 0;
          classify_words(C, w + i + 1, fl + i + 1, j - (i + 1), &none, &o);
        }
        raise_to(C->v, RISK_REVIEW, "find %s: runs a command on every match", a);
        i = j;
        continue;
      }
      if (ARG_IS(a, "-fprint", "-fprint0", "-fprintf", "-fls")) {
        raise_to(C->v, RISK_REVIEW, "find %s: writes a file", a);
        if (i + 1 < n) {
          const char *why = path_escape(w[i + 1]);
          if (why) raise_to(C->v, RISK_BLOCK, "find %s: %s '%s'", a, why, w[i + 1]);
        }
        i++;
        continue;
      }
      if (ARG_IS(a, "-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename", "-regex",
                 "-iregex", "-type", "-size", "-mtime", "-mmin", "-atime", "-amin", "-ctime", "-cmin",
                 "-user", "-group", "-perm", "-maxdepth", "-mindepth", "-newer", "-printf", "-lname",
                 "-ilname", "-uid", "-gid", "-links", "-inum", "-samefile", "-used", "-fstype"))
        i++;
      continue;
    }
    if (preds) continue;
    const char *why = path_escape(a);
    if (why) raise_to(C->v, RISK_BLOCK, "find: %s '%s'", why, a);
    else if (fl[i] & W_VAR) raise_to(C->v, RISK_REVIEW, "find: start point '%s' depends on a variable", a);
  }
  raise_to(C->v, RISK_SAFE, "find: read-only search");
}

// sh / bash / zsh ... / source / .
static void check_shell(Ctx *C, const char *name, char **w, unsigned *fl, int n, const Pipe *P) {
  const int src = strcmp(name, "source") == 0 || strcmp(name, ".") == 0;
  for (int i = 1; i < n; i++) {
    const char *a = w[i];
    if (!src && (a[0] == '-' || a[0] == '+') && a[1] && a[1] != '-') {
      int has_c = 0;
      for (const char *p = a + 1; *p; p++) if (*p == 'c') has_c = 1;
      if (has_c) {
        if (i + 1 >= n) { raise_to(C->v, RISK_REVIEW, "%s -c: runs a command string", name); return; }
        if (fl[i + 1] & W_SUBST) {
          raise_to(C->v, RISK_BLOCK, "%s -c: runs the output of a command substitution", name);
          return;
        }
        classify_text(w[i + 1], strlen(w[i + 1]), C->depth + 1, C->v);
        raise_to(C->v, RISK_REVIEW, "%s -c: runs a command string", name);
        return;
      }
      if (strcmp(a, "-o") == 0 || strcmp(a, "+o") == 0 || strcmp(a, "-O") == 0 || strcmp(a, "+O") == 0) i++;
      continue;
    }
    if (a[0] == '-' && a[1] == '-') continue;
    if (strcmp(a, "-") == 0) break;                    // explicit stdin
    if (fl[i] & W_SUBST) {
      raise_to(C->v, RISK_BLOCK, "%s: runs a script produced by a command substitution", name);
      return;
    }
    const char *why = path_escape(a);
    if (why) raise_to(C->v, RISK_BLOCK, "%s: runs a script outside the workspace (%s '%s')", name, why, a);
    raise_to(C->v, RISK_REVIEW, "%s: runs a script (%s)", name, a);
    return;
  }
  pipe_interp(C, P, name);
}

static void check_cmdexe(Ctx *C, char **w, unsigned *fl, int n, const Pipe *P) {
  (void)fl;
  for (int i = 1; i < n; i++) {
    if (eq_ci(w[i], "/c") || eq_ci(w[i], "/k") || eq_ci(w[i], "/r")) {
      if (i + 1 < n) classify_joined(C, w, i + 1, n);
      raise_to(C->v, RISK_REVIEW, "cmd /c: runs a command string");
      return;
    }
  }
  pipe_interp(C, P, "cmd");
}

static void check_pwsh(Ctx *C, const char *name, char **w, unsigned *fl, int n, const Pipe *P) {
  for (int i = 1; i < n; i++) {
    const char *a = w[i];
    if (a[0] == '-' || a[0] == '/') {
      char p[40];
      size_t k = 0;
      for (const char *x = a + 1; *x && *x != ':' && k + 1 < sizeof(p); x++) p[k++] = (char)lc((unsigned char)*x);
      p[k] = '\0';
      if (!p[0]) continue;
      if (abbrev_of(p, "command")) {
        for (int j = i + 1; j < n; j++)
          if (fl[j] & W_SUBST) raise_to(C->v, RISK_BLOCK, "%s -Command: runs the output of a command substitution", name);
        if (i + 1 < n) classify_joined(C, w, i + 1, n);
        raise_to(C->v, RISK_REVIEW, "%s -Command: runs a PowerShell command string", name);
        return;
      }
      if (abbrev_of(p, "encodedcommand") || strcmp(p, "ec") == 0 || strcmp(p, "enc") == 0) {
        raise_to(C->v, RISK_BLOCK, "%s -EncodedCommand: an encoded command cannot be inspected", name);
        return;
      }
      if (abbrev_of(p, "file") && strlen(p) >= 1) {
        if (i + 1 < n) {
          const char *why = path_escape(w[i + 1]);
          if (why) raise_to(C->v, RISK_BLOCK, "%s -File: %s '%s'", name, why, w[i + 1]);
        }
        raise_to(C->v, RISK_REVIEW, "%s -File: runs a PowerShell script", name);
        return;
      }
      if (abbrev_of(p, "executionpolicy") || strcmp(p, "ep") == 0 || abbrev_of(p, "windowstyle") ||
          abbrev_of(p, "outputformat") || abbrev_of(p, "inputformat") || strcmp(p, "of") == 0 ||
          strcmp(p, "if") == 0 || abbrev_of(p, "configurationname") || abbrev_of(p, "workingdirectory") ||
          strcmp(p, "wd") == 0 || abbrev_of(p, "psconsolefile") || abbrev_of(p, "version") ||
          abbrev_of(p, "settingsfile"))
        i++;
      continue;
    }
    // A bare argument is the command (implicit -Command).
    classify_joined(C, w, i, n);
    raise_to(C->v, RISK_REVIEW, "%s: runs a PowerShell command string", name);
    return;
  }
  pipe_interp(C, P, name);
}

static void check_interp(Ctx *C, const char *name, char **w, unsigned *fl, int n, const Pipe *P) {
  const int py = strncmp(name, "py", 2) == 0;
  for (int i = 1; i < n; i++) {
    const char *a = w[i];
    int code = py ? (strcmp(a, "-c") == 0)
                  : (strcmp(a, "-e") == 0 || strcmp(a, "-E") == 0 || strcmp(a, "--eval") == 0 ||
                     strcmp(a, "-p") == 0 || strcmp(a, "--print") == 0 || strcmp(a, "-r") == 0);
    if (code) {
      if (i + 1 >= n) break;
      const char *src = w[i + 1];
      if (fl[i + 1] & W_SUBST) {
        raise_to(C->v, RISK_BLOCK, "%s: runs code produced by a command substitution", name);
        return;
      }
      static const char *const bad[] = {
        "rmtree", "os.remove", "os.unlink", "os.rmdir", "removedirs", "unlink", "rmdir",
        "rmsync", "rimraf", "fs.rm", "remove-item", "rm -r", "rm -f", "del /", "rd /", "mkfs",
        "format(", "shutil.move", ".delete(", "truncate(", "rm -rf", "git reset --hard",
        "git clean", "shutdown", "system(\"rm", "system('rm"
      };
      for (size_t k = 0; k < sizeof(bad) / sizeof(bad[0]); k++)
        if (contains_ci(src, bad[k])) {
          raise_to(C->v, RISK_BLOCK, "%s: inline code deletes/destroys data (%s)", name, bad[k]);
          return;
        }
      raise_to(C->v, RISK_REVIEW, "%s: runs inline code", name);
      return;
    }
    if (a[0] == '-') {
      if (py && strcmp(a, "-m") == 0) { raise_to(C->v, RISK_REVIEW, "%s -m: runs a module", name); return; }
      continue;
    }
    if (fl[i] & W_SUBST) {
      raise_to(C->v, RISK_BLOCK, "%s: runs a script produced by a command substitution", name);
      return;
    }
    const char *why = path_escape(a);
    if (why) raise_to(C->v, RISK_BLOCK, "%s: runs a script outside the workspace (%s '%s')", name, why, a);
    raise_to(C->v, RISK_REVIEW, "%s: runs a script (%s)", name, a);
    return;
  }
  pipe_interp(C, P, name);
}

// Wrapper (sudo, env, nice, timeout, xargs ...): classify the wrapped command.
// The wrapped command starts at the first word that names a known command; to
// stay closed when an option takes an argument we did not model, every later
// word that names a destructive/executing command is classified from there too.
static void check_wrap(Ctx *C, const char *name, unsigned opt, char **w, unsigned *fl, int n,
                       const Pipe *P) {
  if (opt & O_ELEV) raise_to(C->v, RISK_REVIEW, "%s: runs with elevated privileges", name);
  if (strcmp(name, "command") == 0 && n >= 2 && (strcmp(w[1], "-v") == 0 || strcmp(w[1], "-V") == 0)) {
    raise_to(C->v, RISK_SAFE, "command -v: lookup only");
    return;
  }
  // su -c "cmd", env -S "cmd": a command string.
  for (int i = 1; i + 1 < n; i++) {
    if ((strcmp(name, "su") == 0 && strcmp(w[i], "-c") == 0) ||
        (strcmp(name, "env") == 0 && strcmp(w[i], "-S") == 0) ||
        (strcmp(name, "wsl") == 0 && (strcmp(w[i], "-e") == 0 || strcmp(w[i], "--exec") == 0 ||
                                      strcmp(w[i], "--") == 0))) {
      classify_joined(C, w, i + 1, n);
      raise_to(C->v, RISK_REVIEW, "%s: runs a command string", name);
      return;
    }
  }
  const int xargs = strcmp(name, "xargs") == 0 || strcmp(name, "parallel") == 0;
  int found = 0, only_opts = 1;
  for (int i = 1; i < n; i++) {
    const char *a = w[i];
    if (a[0] != '-' && !is_assignment(a) && !is_digit((unsigned char)a[0])) only_opts = 0;
    if (a[0] == '-' || (a[0] == '/' && is_win_switch(a)) || is_assignment(a)) continue;
    char nm[64];
    normalize_name(a, nm, sizeof(nm));
    const CmdInfo *ci = lookup(nm);
    if (!ci) continue;
    int danger = !(ci->kind == K_SAFE || ci->kind == K_TEXT || ci->kind == K_PATTERN ||
                   ci->kind == K_NOOP || ci->kind == K_KEYWORD);
    if (found && !danger) continue;
    if (*C->v->work > RISK_MAX_WORK) {
      raise_to(C->v, RISK_BLOCK, "command too complex to analyze");
      return;
    }
    // For xargs, the arguments arrive on stdin: a synthetic placeholder target.
    int m = n - i + (xargs ? 1 : 0);
    char **ww = (char**)malloc((size_t)m * sizeof(char*));
    unsigned *ff = (unsigned*)malloc((size_t)m * sizeof(unsigned));
    if (!ww || !ff) { free(ww); free(ff); raise_to(C->v, RISK_BLOCK, "out of memory while analyzing"); return; }
    static char placeholder[] = "{stdin}";
    for (int k = 0; k < n - i; k++) { ww[k] = w[i + k]; ff[k] = fl[i + k]; }
    if (xargs) { ww[m - 1] = placeholder; ff[m - 1] = W_VAR; }
    unsigned o = 0;
    Ctx sub = *C;
    sub.depth = C->depth + 1;
    if (sub.depth > RISK_MAX_DEPTH) {
      raise_to(C->v, RISK_BLOCK, "%s: wrappers nested too deeply to analyze", name);
    } else {
      classify_words(&sub, ww, ff, m, P, &o);
    }
    free(ww);
    free(ff);
    found = 1;
    // A wrapper/keyword scans the rest of the words itself: stop here so a
    // chain of wrappers costs linear, not quadratic, work.
    if (ci->kind == K_WRAP || ci->kind == K_KEYWORD) break;
  }
  check_args(C, name, w, fl, n, 1, opt | O_WIN, 0, RISK_SAFE);
  if (!found) {
    if (only_opts) raise_to(C->v, RISK_SAFE, "%s: no command", name);
    else raise_to(C->v, RISK_REVIEW, "%s: runs an unrecognized command", name);
  }
}

// Name-specific rules for commands that are otherwise SAFE / REVIEW / NET.
// Returns 1 when it decided the level itself.
static int check_special(Ctx *C, const char *name, char **w, unsigned *fl, int n) {
  (void)fl;
  if (strcmp(name, "dd") == 0) {
    for (int i = 1; i < n; i++)
      if (starts_ci(w[i], "of=")) { raise_to(C->v, RISK_BLOCK, "dd %s: raw write over a file/device", w[i]); return 1; }
    return 0;
  }
  if (strcmp(name, "chmod") == 0 || strcmp(name, "chown") == 0 || strcmp(name, "chgrp") == 0) {
    for (int i = 1; i < n; i++)
      if (strcmp(w[i], "--recursive") == 0 || bundle_has(w[i], "R")) {
        raise_to(C->v, RISK_BLOCK, "%s %s: recursive permission/ownership change", name, w[i]);
        return 1;
      }
    return 0;
  }
  if (strcmp(name, "icacls") == 0 || strcmp(name, "takeown") == 0) {
    for (int i = 1; i < n; i++)
      if (eq_ci(w[i], "/t") || eq_ci(w[i], "/r")) {
        raise_to(C->v, RISK_BLOCK, "%s %s: recursive permission/ownership change", name, w[i]);
        return 1;
      }
    return 0;
  }
  if (strcmp(name, "sed") == 0) {
    for (int i = 1; i < n; i++)
      if (starts_ci(w[i], "--in-place") || (w[i][0] == '-' && w[i][1] != '-' && strchr(w[i], 'i'))) {
        raise_to(C->v, RISK_REVIEW, "sed -i: edits files in place");
        return 0;
      }
    return 0;
  }
  if (strcmp(name, "awk") == 0 || strcmp(name, "gawk") == 0 || strcmp(name, "mawk") == 0) {
    raise_to(C->v, RISK_REVIEW, "%s: programs can run commands and write files", name);
    return 0;
  }
  if (strcmp(name, "sort") == 0) {
    for (int i = 1; i < n; i++)
      if (strcmp(w[i], "-o") == 0 || starts_ci(w[i], "--output") || eq_ci(w[i], "/o"))
        raise_to(C->v, RISK_REVIEW, "sort -o: writes a file");
    return 0;
  }
  if (strcmp(name, "uniq") == 0) {
    int pos = 0;
    for (int i = 1; i < n; i++) if (w[i][0] != '-') pos++;
    if (pos >= 2) raise_to(C->v, RISK_REVIEW, "uniq IN OUT: writes a file");
    return 0;
  }
  if (strcmp(name, "date") == 0) {
    for (int i = 1; i < n; i++)
      if (strcmp(w[i], "-s") == 0 || starts_ci(w[i], "--set")) raise_to(C->v, RISK_REVIEW, "date -s: sets the clock");
    return 0;
  }
  if (strcmp(name, "crontab") == 0) {
    for (int i = 1; i < n; i++) {
      if (bundle_has(w[i], "r")) { raise_to(C->v, RISK_BLOCK, "crontab -r: deletes every scheduled job"); return 1; }
      if (!bundle_has(w[i], "l")) { raise_to(C->v, RISK_REVIEW, "crontab: changes scheduled jobs"); return 0; }
    }
    return 0;
  }
  if (strcmp(name, "attrib") == 0) {
    for (int i = 1; i < n; i++)
      if ((w[i][0] == '+' || w[i][0] == '-') && w[i][1]) {
        raise_to(C->v, RISK_REVIEW, "attrib %s: changes file attributes", w[i]);
        return 0;
      }
    return 0;
  }
  if (strcmp(name, "kill") == 0) {
    for (int i = 1; i < n; i++)
      if (strcmp(w[i], "-1") == 0 && i > 1) { raise_to(C->v, RISK_BLOCK, "kill -1: signals every process"); return 1; }
    return 0;
  }
  if (strcmp(name, "nc") == 0 || strcmp(name, "ncat") == 0 || strcmp(name, "netcat") == 0) {
    for (int i = 1; i < n; i++)
      if (strcmp(w[i], "-e") == 0 || strcmp(w[i], "-c") == 0 || starts_ci(w[i], "--exec") ||
          starts_ci(w[i], "--sh-exec")) {
        raise_to(C->v, RISK_BLOCK, "%s %s: serves a shell over the network", name, w[i]);
        return 1;
      }
    return 0;
  }
  if (strcmp(name, "rsync") == 0) {
    for (int i = 1; i < n; i++)
      if (starts_ci(w[i], "--delete") || starts_ci(w[i], "--remove-source-files")) {
        raise_to(C->v, RISK_BLOCK, "rsync %s: deletes files", w[i]);
        return 1;
      }
    return 0;
  }
  if (strcmp(name, "robocopy") == 0) {
    for (int i = 1; i < n; i++)
      if (eq_ci(w[i], "/mir") || eq_ci(w[i], "/purge") || eq_ci(w[i], "/mov") || eq_ci(w[i], "/move")) {
        raise_to(C->v, RISK_BLOCK, "robocopy %s: deletes files at the destination/source", w[i]);
        return 1;
      }
    return 0;
  }
  if (strcmp(name, "reg") == 0) {
    const char *s = n > 1 ? w[1] : "";
    if (eq_ci(s, "delete")) { raise_to(C->v, RISK_BLOCK, "reg delete: deletes registry keys/values"); return 1; }
    if (eq_ci(s, "query")) { raise_to(C->v, RISK_SAFE, "reg query: read-only"); return 1; }
    return 0;
  }
  if (strcmp(name, "systemctl") == 0) {
    for (int i = 1; i < n; i++) {
      if (ARG_IS(w[i], "poweroff", "reboot", "halt", "kexec", "rescue", "emergency", "suspend",
                 "hibernate", "hybrid-sleep")) {
        raise_to(C->v, RISK_BLOCK, "systemctl %s: stops the machine", w[i]);
        return 1;
      }
      if (ARG_IS(w[i], "status", "list-units", "list-unit-files", "show", "is-active", "is-enabled", "cat")) {
        raise_to(C->v, RISK_SAFE, "systemctl %s: read-only", w[i]);
        return 1;
      }
    }
    return 0;
  }
  if (strcmp(name, "init") == 0) {
    for (int i = 1; i < n; i++)
      if (strcmp(w[i], "0") == 0 || strcmp(w[i], "6") == 0) {
        raise_to(C->v, RISK_BLOCK, "init %s: halts/reboots the machine", w[i]);
        return 1;
      }
    return 0;
  }
  if (strcmp(name, "cipher") == 0) {
    for (int i = 1; i < n; i++)
      if (starts_ci(w[i], "/w")) { raise_to(C->v, RISK_BLOCK, "cipher /w: wipes free space"); return 1; }
    return 0;
  }
  if (strcmp(name, "vssadmin") == 0 || strcmp(name, "wbadmin") == 0 || strcmp(name, "wmic") == 0) {
    for (int i = 1; i < n; i++)
      if (eq_ci(w[i], "delete") || eq_ci(w[i], "resize")) {
        raise_to(C->v, RISK_BLOCK, "%s %s: deletes backups/shadow copies/system objects", name, w[i]);
        return 1;
      }
    return 0;
  }
  if (strcmp(name, "alias") == 0 || strcmp(name, "set-alias") == 0) {
    for (int i = 1; i < n; i++) {
      const char *eq = strchr(w[i], '=');
      const char *body = eq ? eq + 1 : (strcmp(name, "set-alias") == 0 && i == n - 1 ? w[i] : NULL);
      if (!body) continue;
      Verdict t = { RISK_SAFE, "", C->v->work };
      classify_text(body, strlen(body), C->depth + 1, &t);
      if (t.lvl == RISK_BLOCK) raise_to(C->v, RISK_BLOCK, "alias to a BLOCK command (%s)", t.why);
    }
    return 0;
  }
  if (strcmp(name, "docker") == 0 || strcmp(name, "podman") == 0) {
    // Volumes hold user data: pruning or removing them is not recoverable.
    int vol = 0, sys = 0, prune = 0, rmv = 0;
    for (int i = 1; i < n; i++) {
      if (strcmp(w[i], "volume") == 0) vol = 1;
      else if (strcmp(w[i], "system") == 0) sys = 1;
      else if (strcmp(w[i], "prune") == 0) prune = 1;
      else if (strcmp(w[i], "rm") == 0) rmv = 1;
    }
    if ((vol && (prune || rmv)) || (sys && prune)) {
      raise_to(C->v, RISK_BLOCK, "%s %s %s: deletes volumes/data", name, vol ? "volume" : "system",
               prune ? "prune" : "rm");
      return 1;
    }
    return 0;
  }
  if (strcmp(name, "net") == 0) {
    for (int i = 1; i < n; i++)
      if (eq_ci(w[i], "/delete")) { raise_to(C->v, RISK_BLOCK, "net ... /delete: deletes an account/share"); return 1; }
    return 0;
  }
  return 0;
}

static int is_version_flag(const char *a) {
  return ARG_IS(a, "--version", "-version", "-V", "version", "--help", "-h", "/?", "-v");
}

// Classify one simple command (words already split, no operators). out gets
// O_FETCH / O_DECODE / O_ECHO bits for the pipeline bookkeeping.
static void classify_words(Ctx *C, char **w, unsigned *fl, int n, const Pipe *P, unsigned *out) {
  *out = 0;
  if (++*C->v->work > RISK_MAX_WORK) {
    raise_to(C->v, RISK_BLOCK, "command too complex to analyze");
    return;
  }
  while (n > 0 && is_assignment(w[0])) {                     // A=1 cmd
    if (starts_ci(w[0], "PATH=") || starts_ci(w[0], "LD_") || starts_ci(w[0], "DYLD_"))
      raise_to(C->v, RISK_REVIEW, "sets %.*s for the command", (int)(strchr(w[0], '=') - w[0]), w[0]);
    w++; fl++; n--;
  }
  if (n >= 2 && strcmp(w[1], "=") == 0 && (fl[0] & W_VAR)) { w += 2; fl += 2; n -= 2; }   // PS: $x = cmd
  if (n <= 0) return;

  {
    const char *sl = strchr(w[0], '/');
    if (sl && sl > w[0] && !(fl[0] & W_VAR) && (size_t)(sl - w[0]) < 32) {
      char head[32];
      memcpy(head, w[0], (size_t)(sl - w[0]));
      head[sl - w[0]] = '\0';
      char hn[32];
      normalize_name(head, hn, sizeof(hn));
      const CmdInfo *hi = lookup(hn);
      if (hi && ((hi->opt & O_WIN) || hi->kind == K_DELETE || hi->kind == K_CMDEXE)) {
        // Split "rd/s/q" into "rd" "/s" "/q" (cmd.exe accepts it) and re-classify.
        int k = 1;
        for (const char *p = sl; *p; p++) if (*p == '/') k++;
        char **ww = (char**)malloc((size_t)(n + k) * sizeof(char*));
        unsigned *ff = (unsigned*)malloc((size_t)(n + k) * sizeof(unsigned));
        char *store = (char*)malloc(strlen(w[0]) + (size_t)k + 2);
        if (!ww || !ff || !store) {
          free(ww); free(ff); free(store);
          raise_to(C->v, RISK_BLOCK, "out of memory while analyzing");
          return;
        }
        int m = 0;
        char *o = store;
        memcpy(o, head, strlen(head) + 1);
        ww[m] = o; ff[m++] = fl[0]; o += strlen(head) + 1;
        for (const char *p = sl; *p; ) {
          const char *q = strchr(p + 1, '/');
          size_t L = q ? (size_t)(q - p) : strlen(p);
          memcpy(o, p, L);
          o[L] = '\0';
          ww[m] = o; ff[m++] = 0; o += L + 1;
          p += L;
        }
        for (int i = 1; i < n; i++) { ww[m] = w[i]; ff[m++] = fl[i]; }
        classify_words(C, ww, ff, m, P, out);
        free(ww); free(ff); free(store);
        return;
      }
    }
  }
  char name[64];
  normalize_name(w[0], name, sizeof(name));
  if (strcmp(name, "cd..") == 0 || strcmp(name, "cd\\") == 0) {
    raise_to(C->v, RISK_BLOCK, "%s: leaves the workspace", w[0]);
    return;
  }
  if (fl[0] & W_VAR) {
    // The program is only known at run time. Judge its arguments as if it
    // were rm: "$X -rf ." is still a recursive delete of the workspace.
    Ctx sub = *C;
    Verdict tmp = { RISK_SAFE, "", C->v->work };
    sub.v = &tmp;
    char *save = w[0];
    static char rm_name[] = "rm";
    w[0] = rm_name;
    check_delete(&sub, "rm", w, fl, n, P);
    w[0] = save;
    if (tmp.lvl == RISK_BLOCK)
      raise_to(C->v, RISK_BLOCK, "command name '%s' is a variable, arguments look destructive", w[0]);
    // rm$IFS-rf$IFS. / ${x}rm: re-read the line with each expansion as a blank.
    char *j = join_words(w, 0, n);
    if (j) {
      char *o = j;
      for (char *p = j; *p; ) {
        if (*p == '$' && p[1] == '{') { char *e = strchr(p, '}'); p = e ? e + 1 : p + strlen(p); *o++ = ' '; continue; }
        if (*p == '$' && (is_alnum((unsigned char)p[1]) || p[1] == '_')) {
          p++;
          while (is_alnum((unsigned char)*p) || *p == '_') p++;
          *o++ = ' ';
          continue;
        }
        if (*p == '%' && strchr(p + 1, '%')) { p = strchr(p + 1, '%') + 1; *o++ = ' '; continue; }
        if (*p == '$' && p[1] == '(' && p[2] == ')') { p += 3; *o++ = ' '; continue; }   // substitution marker
        *o++ = *p++;
      }
      *o = '\0';
      Verdict t2 = { RISK_SAFE, "", C->v->work };
      classify_text(j, strlen(j), C->depth + 1, &t2);
      free(j);
      if (t2.lvl == RISK_BLOCK && strncmp(t2.why, "command built from", 18) != 0)
        raise_to(C->v, RISK_BLOCK, "command built from variables reads as: %s", t2.why);
      else if (t2.lvl == RISK_BLOCK)
        raise_to(C->v, RISK_BLOCK, "%s", t2.why);
    }
    raise_to(C->v, RISK_REVIEW, "command name '%s' is a variable", w[0]);
    check_args(C, "command", w, fl, n, 1, 0, 0, RISK_REVIEW);
    return;
  }
  if (strchr(w[0], '/') || strchr(w[0], '\\')) {
    const char *why = path_escape(w[0]);
    if (why) raise_to(C->v, why[0] == '\'' ? RISK_BLOCK : RISK_REVIEW, "runs a program by path (%s '%s')", why, w[0]);
  }
  if (strncmp(name, "git-", 4) == 0 && name[4]) {
    // git-clean -f == git clean -f: re-classify in the two-word form.
    char **ww = (char**)malloc((size_t)(n + 1) * sizeof(char*));
    unsigned *ff = (unsigned*)malloc((size_t)(n + 1) * sizeof(unsigned));
    if (!ww || !ff) { free(ww); free(ff); raise_to(C->v, RISK_BLOCK, "out of memory while analyzing"); return; }
    static char git_word[] = "git";
    char sub[64];
    snprintf(sub, sizeof(sub), "%s", name + 4);
    ww[0] = git_word; ff[0] = 0;
    ww[1] = sub; ff[1] = 0;
    for (int i = 1; i < n; i++) { ww[i + 1] = w[i]; ff[i + 1] = fl[i]; }
    check_git(C, ww, ff, n + 1);
    free(ww);
    free(ff);
    return;
  }
  const CmdInfo *ci = lookup(name);
  if (!ci) {
    raise_to(C->v, RISK_REVIEW, "unrecognized command '%s'", name);
    check_args(C, name, w, fl, n, 1, 0, 0, RISK_REVIEW);
    return;
  }
  *out = ci->opt & (O_FETCH | O_DECODE | O_ECHO);
  if (((ci->opt & O_VER) || ci->kind == K_INTERP) && n == 2 && is_version_flag(w[1])) {
    raise_to(C->v, RISK_SAFE, "%s %s: version/help query", name, w[1]);
    return;
  }
  switch (ci->kind) {
    case K_NOOP:
      return;
    case K_KEYWORD:
      if (n > 1) check_wrap(C, name, 0, w, fl, n, P);
      return;
    case K_WRAP:
      check_wrap(C, name, ci->opt, w, fl, n, P);
      return;
    case K_BLOCK:
      raise_to(C->v, RISK_BLOCK, "%s: destroys data or stops the system", name);
      return;
    case K_DELETE:
      check_delete(C, name, w, fl, n, P);
      return;
    case K_GIT:
      check_git(C, w, fl, n);
      return;
    case K_FIND:
      check_find(C, w, fl, n, P);
      return;
    case K_SHELL:
      check_shell(C, name, w, fl, n, P);
      return;
    case K_CMDEXE:
      check_cmdexe(C, w, fl, n, P);
      return;
    case K_PWSH:
      check_pwsh(C, name, w, fl, n, P);
      return;
    case K_INTERP:
      check_interp(C, name, w, fl, n, P);
      return;
    case K_EVAL:
      for (int i = 1; i < n; i++)
        if (fl[i] & W_SUBST) raise_to(C->v, RISK_BLOCK, "%s: executes the output of a command", name);
      if (n > 1) {
        classify_joined(C, w, 1, n);
        raise_to(C->v, RISK_REVIEW, "%s: executes a command string", name);
      } else {
        pipe_interp(C, P, name);
      }
      return;
    default:
      break;
  }
  if (check_special(C, name, w, fl, n)) return;
  const int text = ci->kind == K_TEXT;
  if (!text)
    check_args(C, name, w, fl, n, 1, ci->opt, ci->kind == K_PATTERN, RISK_REVIEW);
  if (ci->kind == K_REVIEW) raise_to(C->v, RISK_REVIEW, "%s: writes/moves/installs or runs code", name);
  else if (ci->kind == K_NET) raise_to(C->v, RISK_REVIEW, "%s: network access", name);
  else raise_to(C->v, RISK_SAFE, "%s: read-only", name);
}

// One output/input redirection target.
static void check_redirect(Ctx *C, int type, const Tok *t) {
  if (!t) return;
  if (type == T_OUT) {
    if (is_null_sink(t->s)) return;
    const char *why = path_escape(t->s);
    if (why) raise_to(C->v, RISK_BLOCK, "redirect writes outside the workspace (%s '%s')", why, t->s);
    else if (t->fl & W_VAR) raise_to(C->v, RISK_BLOCK, "redirect target '%s' is a variable", t->s);
    else raise_to(C->v, RISK_REVIEW, "redirect creates/overwrites '%s'", t->s);
  } else if (type == T_IN) {
    const char *why = path_escape(t->s);
    if (why) raise_to(C->v, RISK_BLOCK, "reads a file outside the workspace (%s '%s')", why, t->s);
  }
}

// Classify one tokenization: split into pipelines and simple commands.
static void classify_tokens(Ctx *C, Tok *t, int nt) {
  char **w = (char**)malloc((size_t)(nt + 1) * sizeof(char*));
  unsigned *fl = (unsigned*)malloc((size_t)(nt + 1) * sizeof(unsigned));
  if (!w || !fl) { free(w); free(fl); raise_to(C->v, RISK_BLOCK, "out of memory while analyzing"); return; }
  Pipe P = {0, 0, 0, NULL};
  int nw = 0, wrote = 0;
  for (int i = 0; i <= nt; i++) {
    const int type = i < nt ? t[i].type : T_SEP;
    if (type == T_WORD) { w[nw] = t[i].s; fl[nw] = t[i].fl; nw++; continue; }
    if (type == T_OUT || type == T_IN || type == T_HEREDOC) {
      const Tok *target = (i + 1 < nt && t[i + 1].type == T_WORD) ? &t[i + 1] : NULL;
      if (type != T_HEREDOC) check_redirect(C, type, target);
      if (type == T_OUT && target && !is_null_sink(target->s)) wrote = 1;
      if (target) i++;
      continue;
    }
    if (type == T_DUP) continue;
    // T_SEP or T_PIPE: end of a simple command
    unsigned o = 0;
    if (nw > 0) classify_words(C, w, fl, nw, &P, &o);
    if ((o & O_ECHO) && wrote && nw > 1) {
      // echo 'rm -rf .' > x.sh && sh x.sh : judge the text being written.
      char *txt = join_words(w, 1, nw);
      if (txt) {
        Verdict t = { RISK_SAFE, "", C->v->work };
        classify_text(txt, strlen(txt), C->depth + 1, &t);
        free(txt);
        if (t.lvl == RISK_BLOCK) raise_to(C->v, RISK_BLOCK, "writes a BLOCK command to a file (%s)", t.why);
      }
    }
    if (type == T_PIPE) {
      if (o & O_FETCH) P.fetch = 1;
      if (o & O_DECODE) P.decode = 1;
      free(P.echo);
      P.echo = (o & O_ECHO) && nw > 1 ? join_words(w, (nw > 2 && w[1][0] == '-') ? 2 : 1, nw) : NULL;
      P.pos++;
    } else {
      free(P.echo);
      P.echo = NULL;
      P.pos = P.fetch = P.decode = 0;
    }
    nw = 0;
    wrote = 0;
  }
  free(P.echo);
  free(w);
  free(fl);
}

// Whole-string patterns that survive tokenization poorly.
static void classify_raw(const char *s, Verdict *v) {
  for (const unsigned char *p = (const unsigned char*)s; *p; p++)
    if ((*p < 0x20 && *p != '\t' && *p != '\n' && *p != '\r') || *p == 0x7F) {
      raise_to(v, RISK_REVIEW, "contains control characters");
      break;
    }
  if ((strstr(s, "(){") || strstr(s, "() {")) && strchr(s, '|') && strchr(s, '&'))
    raise_to(v, RISK_BLOCK, "fork-bomb pattern (self-piping function)");
  // PowerShell download-and-execute written without a pipe: iex (iwr ...),
  // IEX(New-Object Net.WebClient).DownloadString(...)
  int iex = 0;
  for (const char *p = s; *p; p++) {
    if ((p == s || !is_alnum((unsigned char)p[-1])) && starts_ci(p, "iex") && !is_alnum((unsigned char)p[3]) &&
        p[3] != '-')
      iex = 1;
  }
  if (contains_ci(s, "invoke-expression")) iex = 1;
  if (iex && (contains_ci(s, "downloadstring") || contains_ci(s, "downloadfile") || contains_ci(s, "iwr") ||
              contains_ci(s, "invoke-webrequest") || contains_ci(s, "irm ") || contains_ci(s, "invoke-restmethod") ||
              contains_ci(s, "http://") || contains_ci(s, "https://") || contains_ci(s, "curl") ||
              contains_ci(s, "wget") || contains_ci(s, "frombase64string")))
    raise_to(v, RISK_BLOCK, "downloads/decodes code and executes it (iex)");
  static const char *const calls[] = { ".delete(", "::delete(", "deletefile(", "removedirectory(" };
  for (size_t i = 0; i < sizeof(calls) / sizeof(calls[0]); i++)
    if (contains_ci(s, calls[i])) raise_to(v, RISK_BLOCK, "calls a delete API (%s)", calls[i]);
}

static void classify_text(const char *s, size_t n, int depth, Verdict *v) {
  if (depth > RISK_MAX_DEPTH) {
    raise_to(v, RISK_BLOCK, "command nesting too deep to analyze");
    return;
  }
  if ((*v->work += 1 + (long)n / 64) > RISK_MAX_WORK) {
    raise_to(v, RISK_BLOCK, "command too complex to analyze");
    return;
  }
  char *copy = (char*)malloc(n + 1);
  if (!copy) { raise_to(v, RISK_BLOCK, "out of memory while analyzing"); return; }
  memcpy(copy, s, n);
  copy[n] = '\0';
  n = strlen(copy);                                // stop at an embedded NUL
  classify_raw(copy, v);

  Verdict best = { RISK_SAFE, "", v->work };
  int best_mode = -1;
  for (int mi = 0; mi < M_COUNT; mi++) {
    const int m = MODE_ORDER[mi];
    Verdict vm = { RISK_SAFE, "", v->work };
    Lexer L;
    memset(&L, 0, sizeof(L));
    L.s = copy;
    L.n = n;
    L.mode = m;
    L.depth = depth;
    L.v = &vm;
    L.cap = (int)n + 8;
    L.t = (Tok*)malloc((size_t)L.cap * sizeof(Tok));
    L.buf = (char*)malloc(2 * n + 16);
    if (!L.t || !L.buf) {
      free(L.t); free(L.buf); free(copy);
      raise_to(v, RISK_BLOCK, "out of memory while analyzing");
      return;
    }
    lex(&L);
    Ctx C = { &vm, depth, m };
    classify_tokens(&C, L.t, L.nt);
    free(L.t);
    free(L.buf);
    if (best_mode < 0 || vm.lvl > best.lvl) { best = vm; best_mode = m; }
  }
  free(copy);
  if (best.lvl > RISK_SAFE && best_mode != MODE_ORDER[0] && best.why[0])
    raise_to(v, best.lvl, "%s (as %s)", best.why, MODE_NAME[best_mode]);
  else if (best.why[0])
    raise_to(v, best.lvl, "%s", best.why);
  else
    raise_to(v, best.lvl, "no command");
}

RiskLevel risk_classify(const char *cmd, char *why, int why_cap) {
  long work = 0;
  Verdict v = { RISK_SAFE, "", &work };
  if (!cmd || !cmd[0]) {
    raise_to(&v, RISK_SAFE, "empty command");
  } else {
    size_t n = strlen(cmd);
    if (n > RISK_MAX_INPUT) raise_to(&v, RISK_BLOCK, "command too long to analyze (%zu bytes)", n);
    else classify_text(cmd, n, 0, &v);
  }
  if (why && why_cap > 0) snprintf(why, (size_t)why_cap, "%s", v.why);
  return v.lvl;
}

const char *risk_level_name(RiskLevel level) {
  switch (level) {
    case RISK_SAFE: return "SAFE";
    case RISK_REVIEW: return "REVIEW";
    case RISK_BLOCK: return "BLOCK";
  }
  return "BLOCK";
}
