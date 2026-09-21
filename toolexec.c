// toolexec.c - Execute a parsed tool call, confined to a workspace directory
//
// Safety model (defense in depth; the model's output is untrusted input):
//   1. Lexical check (toolexec_path_reason): relative paths only, no "..", no
//      ':' (drive-relative "C:x", NTFS alternate data streams "a:b"), no control
//      or wildcard characters, no Windows device names (CON, NUL.txt, COM1...),
//      no component ending in '.' or ' ' (Windows silently strips those, so
//      "tools." would alias "tools").
//   2. Resolution: the path is joined onto the workspace, the file is OPENED,
//      and the canonical path of the opened object (after every symlink and
//      junction) must lie inside the canonical workspace. Windows: the handle's
//      GetFinalPathNameByHandleW. POSIX: realpath() of the parent directory,
//      then open(O_NOFOLLOW) of the final component. Hard-linked files (link
//      count > 1) are refused, since a hard link can alias a file outside the
//      workspace. The data is then read/written through that same
//      handle, so the checked object is the one used.
//   3. Auto-approved writes additionally refuse .git/, a top-level tools/ and
//      executable/script extensions (toolexec_write_policy), re-checked on the
//      canonical path so 8.3 short names / case tricks do not slip through.
//   4. run_command runs with the workspace as its cwd, stdin from the null
//      device, stdout+stderr captured, drained to EOF with a size cap, and a
//      wall-clock timeout (Windows: Job Object; POSIX: process group + kill).
// WebAssembly has no symlinks worth resolving and no processes: stage 2 is a
// plain fopen of the lexically checked path, and run_command is unavailable.

// realpath() is XSI (musl/glibc hide it under plain _POSIX_C_SOURCE); Darwin
// hides O_NOFOLLOW unless _DARWIN_C_SOURCE is set.
#if !defined(_WIN32) && !defined(__wasi__) && !defined(__wasm__)
#define _XOPEN_SOURCE 700
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#endif

#include "toolexec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__wasi__) || defined(__wasm__)
#define TE_WASM 1
#include <sys/stat.h>
#elif defined(_WIN32)
#define TE_WIN 1
#include <windows.h>
#else
#define TE_POSIX 1
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#endif

#define TE_DEFAULT_TIMEOUT 60
#define TE_NOTE_RESERVE    96     // room kept in obs for the trailing notes
#define TE_PATH_MAX        1024

// ---------------------------------------------------------------------------
// Lexical checks (portable)
// ---------------------------------------------------------------------------

static int is_sep(char c) { return c == '/' || c == '\\'; }

static int lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// Case-insensitive: does s[0..n) equal the NUL-terminated word w?
static int ci_eqn(const char *s, size_t n, const char *w) {
  size_t wn = strlen(w);
  if (n != wn) return 0;
  for (size_t i = 0; i < n; i++) if (lower((unsigned char)s[i]) != lower((unsigned char)w[i])) return 0;
  return 1;
}

// Windows device names are reserved in EVERY directory and with ANY extension
// ("nul.txt" is still the null device). Base name = up to the first '.', with
// trailing spaces ignored.
static int is_reserved_name(const char *c, size_t n) {
  size_t b = 0;
  while (b < n && c[b] != '.') b++;
  while (b > 0 && c[b - 1] == ' ') b--;
  static const char *const fixed[] = { "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$" };
  for (size_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]); i++)
    if (ci_eqn(c, b, fixed[i])) return 1;
  if (b >= 4 && (ci_eqn(c, 3, "COM") || ci_eqn(c, 3, "LPT"))) {
    if (b == 4 && c[3] >= '0' && c[3] <= '9') return 1;
    // superscript digits 1-3 (UTF-8 C2 B9 / C2 B2 / C2 B3) are reserved too
    if (b == 5 && (unsigned char)c[3] == 0xC2 &&
        ((unsigned char)c[4] == 0xB9 || (unsigned char)c[4] == 0xB2 || (unsigned char)c[4] == 0xB3))
      return 1;
  }
  return 0;
}

const char *toolexec_path_reason(const char *p) {
  if (!p || !p[0]) return "empty path";
  size_t n = strlen(p);
  if (n > 400) return "path too long";
  if (is_sep(p[0])) return "absolute/UNC path (must be relative to the workspace)";
  for (size_t i = 0; i < n; i++) {
    unsigned char c = (unsigned char)p[i];
    if (c < 0x20 || c == 0x7F) return "control character in path";
    if (c == ':') return "':' is not allowed (drive letters, alternate data streams)";
    if (strchr("*?\"<>|", c)) return "wildcard/reserved character in path";
  }
  const char *s = p;
  while (*s) {
    const char *c = s;
    while (*s && !is_sep(*s)) s++;
    size_t len = (size_t)(s - c);
    while (is_sep(*s)) s++;
    if (len == 0) continue;
    if (len == 1 && c[0] == '.') continue;
    if (len == 2 && c[0] == '.' && c[1] == '.') return "'..' component (escapes the workspace)";
    if (c[len - 1] == '.' || c[len - 1] == ' ') return "path component ends in '.' or ' '";
    if (is_reserved_name(c, len)) return "Windows reserved device name";
  }
  return NULL;
}

const char *toolexec_write_policy(const char *p) {
  static const char *const exts[] = {
    "py", "pyw", "ps1", "psm1", "bat", "cmd", "sh", "bash", "exe", "dll", "so",
    "dylib", "com", "scr", "vbs", "msi", "lnk"
  };
  if (!p) return "empty path";
  const char *s = p, *last = NULL;
  size_t last_len = 0;
  int first = 1;
  while (*s) {
    const char *c = s;
    while (*s && !is_sep(*s)) s++;
    size_t len = (size_t)(s - c);
    int more = 0;
    while (is_sep(*s)) { s++; more = 1; }
    if (len == 0 || (len == 1 && c[0] == '.')) continue;
    if (ci_eqn(c, len, ".git")) return "refusing to write inside .git/";
    if (first && more && ci_eqn(c, len, "tools")) return "refusing to write inside tools/";
    first = 0;
    last = c; last_len = len;
  }
  if (last) {
    const char *dot = NULL;
    for (size_t i = 0; i < last_len; i++) if (last[i] == '.') dot = last + i;
    if (dot) {
      size_t el = last_len - (size_t)(dot + 1 - last);
      for (size_t i = 0; i < sizeof(exts) / sizeof(exts[0]); i++)
        if (ci_eqn(dot + 1, el, exts[i])) return "refusing to write an executable/script file";
    }
  }
  return NULL;
}

// Join workspace + '/' + rel into out. Returns 0 on success.
static int join_path(const char *ws, const char *rel, char *out, size_t cap) {
  if (!ws || !ws[0]) ws = ".";
  int n = snprintf(out, cap, "%s/%s", ws, rel);
  return (n < 0 || (size_t)n >= cap) ? -1 : 0;
}

// Output collector: keeps up to cap bytes, counts everything past that.
typedef struct { char *buf; size_t cap, len; unsigned long long dropped; } Sink;

static void sink_put(Sink *k, const char *d, size_t n) {
  size_t room = k->cap - k->len;
  size_t take = n < room ? n : room;
  memcpy(k->buf + k->len, d, take);
  k->len += take;
  k->dropped += (unsigned long long)(n - take);
}

static int content_cap(int obs_size) {
  int c = obs_size - 1 - TE_NOTE_RESERVE;
  return c < 0 ? 0 : c;
}

// Append "\n[truncated N bytes]" (if any were dropped) after the kept content.
static void finish_obs(char *obs, int obs_size, Sink *k) {
  obs[k->len] = '\0';
  if (k->dropped) {
    size_t l = strlen(obs);
    snprintf(obs + l, (size_t)obs_size - l, "\n[truncated %llu bytes]", k->dropped);
  }
}

#if !defined(TE_WASM)
static void append_note(char *obs, int obs_size, const char *note) {
  size_t l = strlen(obs);
  snprintf(obs + l, (size_t)obs_size - l, "%s%s", l ? "\n" : "", note);
}
#endif

// ---------------------------------------------------------------------------
// Workspace-confined open (platform specific)
// ---------------------------------------------------------------------------
// te_open_*: resolve rel inside ws, verify containment, and open it.

#if defined(TE_WIN)

static int to_w(const char *s, wchar_t *out, int cap) {
  return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, out, cap) > 0;
}
static int to_u8(const wchar_t *s, char *out, int cap) {
  return WideCharToMultiByte(CP_UTF8, 0, s, -1, out, cap, NULL, NULL) > 0;
}

// Canonical path of an open handle, "\\?\" prefix removed ("\\?\UNC\x" -> "\\x").
static int final_path(HANDLE h, wchar_t *out, DWORD cap) {
  wchar_t tmp[TE_PATH_MAX];
  DWORD n = GetFinalPathNameByHandleW(h, tmp, TE_PATH_MAX, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
  if (n == 0 || n >= TE_PATH_MAX) return 0;
  const wchar_t *s = tmp;
  if (wcsncmp(s, L"\\\\?\\UNC\\", 8) == 0) {
    if (wcslen(s + 6) + 1 > cap) return 0;
    out[0] = L'\\'; wcscpy(out + 1, s + 7);
    return 1;
  }
  if (wcsncmp(s, L"\\\\?\\", 4) == 0) s += 4;
  if (wcslen(s) + 1 > cap) return 0;
  wcscpy(out, s);
  return 1;
}

static int dir_final(const char *dir, wchar_t *out, DWORD cap) {
  wchar_t w[TE_PATH_MAX];
  if (!to_w(dir, w, TE_PATH_MAX)) return 0;
  HANDLE h = CreateFileW(w, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                         OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
  if (h == INVALID_HANDLE_VALUE) return 0;
  int ok = final_path(h, out, cap);
  CloseHandle(h);
  return ok;
}

// Is p strictly inside root? Returns a pointer to the part after "root\" or NULL.
static const wchar_t *w_inside(const wchar_t *root, const wchar_t *p) {
  size_t rn = wcslen(root);
  while (rn > 0 && root[rn - 1] == L'\\') rn--;
  if (_wcsnicmp(root, p, rn) != 0 || p[rn] != L'\\' || p[rn + 1] == 0) return NULL;
  return p + rn + 1;
}

int toolexec_workspace_init(const char *dir, char *canon, int cap) {
  wchar_t w[TE_PATH_MAX], f[TE_PATH_MAX];
  if (!dir || !dir[0]) dir = ".";
  if (!to_w(dir, w, TE_PATH_MAX)) return -1;
  if (!CreateDirectoryW(w, NULL) && GetLastError() != ERROR_ALREADY_EXISTS) return -1;
  if (!dir_final(dir, f, TE_PATH_MAX)) return -1;
  return to_u8(f, canon, cap) ? 0 : -1;
}

// Open rel inside ws for reading or (truncating) writing. On failure returns
// INVALID_HANDLE_VALUE with a message in err.
static HANDLE te_open(const char *ws, const char *rel, int write, int auto_mode,
                      char *err, int errcap) {
  char full[TE_PATH_MAX];
  wchar_t wfull[TE_PATH_MAX], wroot[TE_PATH_MAX], wfin[TE_PATH_MAX];
  wfin[0] = 0;
  if (join_path(ws, rel, full, sizeof(full)) != 0 || !to_w(full, wfull, TE_PATH_MAX)) {
    snprintf(err, errcap, "path too long or not valid UTF-8"); return INVALID_HANDLE_VALUE;
  }
  if (!dir_final((ws && ws[0]) ? ws : ".", wroot, TE_PATH_MAX)) {
    snprintf(err, errcap, "workspace directory is missing"); return INVALID_HANDLE_VALUE;
  }
  // Before anything is created: the parent directory itself must already
  // resolve inside the workspace (else OPEN_ALWAYS could create a file through
  // a junction). The final component is checked on the opened handle below.
  {
    char par[TE_PATH_MAX];
    wchar_t wpar[TE_PATH_MAX];
    snprintf(par, sizeof(par), "%s", full);
    size_t n = strlen(par);
    while (n > 0 && is_sep(par[n - 1])) par[--n] = 0;
    while (n > 0 && !is_sep(par[n - 1])) n--;
    if (n > 0) par[n - 1] = 0;
    if (!dir_final(par, wpar, TE_PATH_MAX)) {
      snprintf(err, errcap, write ? "cannot write '%s' (no such directory)" : "cannot read '%s'", rel);
      return INVALID_HANDLE_VALUE;
    }
    if (_wcsicmp(wpar, wroot) != 0 && !w_inside(wroot, wpar)) {
      snprintf(err, errcap, "refused '%s': resolves outside the workspace (symlink/junction?)", rel);
      return INVALID_HANDLE_VALUE;
    }
  }
  HANDLE h = CreateFileW(wfull, write ? GENERIC_WRITE : GENERIC_READ, FILE_SHARE_READ, NULL,
                         write ? OPEN_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) {
    snprintf(err, errcap, write ? "cannot write '%s'" : "cannot read '%s'", rel);
    return INVALID_HANDLE_VALUE;
  }
  int created = write && GetLastError() != ERROR_ALREADY_EXISTS;
  const char *why = NULL;
  BY_HANDLE_FILE_INFORMATION info;
  const wchar_t *relw = NULL;
  if (!GetFileInformationByHandle(h, &info) || GetFileType(h) != FILE_TYPE_DISK)
    why = "not a regular file";
  else if (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
    why = "is a directory";
  else if (!final_path(h, wfin, TE_PATH_MAX) || !(relw = w_inside(wroot, wfin)))
    why = "resolves outside the workspace (symlink/junction?)";
  else if (info.nNumberOfLinks > 1)
    why = "hard-linked file (may alias a file outside the workspace)";
  else if (write && auto_mode) {
    char relu[TE_PATH_MAX];
    if (!to_u8(relw, relu, sizeof(relu))) why = "unrepresentable path";
    else why = toolexec_write_policy(relu);        // canonical name: catches 8.3 / case tricks
  }
  if (why) {
    CloseHandle(h);
    if (created) DeleteFileW(wfin[0] ? wfin : wfull);
    snprintf(err, errcap, "refused '%s': %s", rel, why);
    return INVALID_HANDLE_VALUE;
  }
  if (write) {
    SetFilePointer(h, 0, NULL, FILE_BEGIN);
    SetEndOfFile(h);                               // truncate only after the checks passed
  }
  return h;
}

static int te_write(const char *ws, const char *rel, const char *data, size_t n, int auto_mode,
                    char *err, int errcap) {
  HANDLE h = te_open(ws, rel, 1, auto_mode, err, errcap);
  if (h == INVALID_HANDLE_VALUE) return -1;
  DWORD put = 0;
  BOOL ok = WriteFile(h, data, (DWORD)n, &put, NULL);
  CloseHandle(h);
  if (!ok || put != (DWORD)n) { snprintf(err, errcap, "write to '%s' failed", rel); return -1; }
  return 0;
}

static int te_read(const char *ws, const char *rel, Sink *k, char *err, int errcap) {
  HANDLE h = te_open(ws, rel, 0, 0, err, errcap);
  if (h == INVALID_HANDLE_VALUE) return -1;
  char buf[4096];
  DWORD got;
  while (ReadFile(h, buf, sizeof(buf), &got, NULL) && got > 0) sink_put(k, buf, got);
  CloseHandle(h);
  return 0;
}

#elif defined(TE_POSIX)

int toolexec_workspace_init(const char *dir, char *canon, int cap) {
  char r[PATH_MAX];
  if (!dir || !dir[0]) dir = ".";
  if (mkdir(dir, 0755) != 0 && errno != EEXIST) return -1;
  if (!realpath(dir, r)) return -1;
  int n = snprintf(canon, (size_t)cap, "%s", r);
  return (n < 0 || n >= cap) ? -1 : 0;
}

// Returns an fd, or -1 with err set.
static int te_open(const char *ws, const char *rel, int write, int auto_mode,
                   char *err, int errcap) {
  char full[TE_PATH_MAX], root[PATH_MAX], parent[PATH_MAX], target[PATH_MAX + 256];
  if (join_path(ws, rel, full, sizeof(full)) != 0) { snprintf(err, errcap, "path too long"); return -1; }
  if (!realpath((ws && ws[0]) ? ws : ".", root)) { snprintf(err, errcap, "workspace directory is missing"); return -1; }
  // Split off the final component (trailing separators are not a file name).
  size_t n = strlen(full);
  while (n > 0 && is_sep(full[n - 1])) full[--n] = '\0';
  char *slash = full + n;
  while (slash > full && !is_sep(slash[-1])) slash--;
  const char *base = slash;
  if (!base[0] || strcmp(base, ".") == 0) { snprintf(err, errcap, "refused '%s': not a file name", rel); return -1; }
  slash[-1] = '\0';                                // full is now the parent directory
  if (!realpath(full, parent)) {
    snprintf(err, errcap, write ? "cannot write '%s'" : "cannot read '%s'", rel); return -1;
  }
  size_t rn = strlen(root);
  if (strcmp(root, "/") == 0) rn = 0;
  if (strcmp(parent, root) != 0 && !(strncmp(parent, root, rn) == 0 && parent[rn] == '/')) {
    snprintf(err, errcap, "refused '%s': resolves outside the workspace (symlink?)", rel); return -1;
  }
  snprintf(target, sizeof(target), "%s/%s", parent, base);
  if (write && auto_mode) {                        // re-check on the canonical relative path
    const char *why = toolexec_write_policy(target + rn + 1);
    if (why) { snprintf(err, errcap, "refused '%s': %s", rel, why); return -1; }
  }
  int fd = open(target, write ? (O_WRONLY | O_CREAT | O_NOFOLLOW) : (O_RDONLY | O_NOFOLLOW), 0644);
  if (fd < 0) {
    if (errno == ELOOP) snprintf(err, errcap, "refused '%s': is a symlink", rel);
    else snprintf(err, errcap, write ? "cannot write '%s'" : "cannot read '%s'", rel);
    return -1;
  }
  struct stat st;
  const char *why = NULL;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) why = "not a regular file";
  else if (st.st_nlink > 1) why = "hard-linked file (may alias a file outside the workspace)";
  if (why) { close(fd); snprintf(err, errcap, "refused '%s': %s", rel, why); return -1; }
  if (write && ftruncate(fd, 0) != 0) { close(fd); snprintf(err, errcap, "cannot truncate '%s'", rel); return -1; }
  return fd;
}

static int te_write(const char *ws, const char *rel, const char *data, size_t n, int auto_mode,
                    char *err, int errcap) {
  int fd = te_open(ws, rel, 1, auto_mode, err, errcap);
  if (fd < 0) return -1;
  size_t off = 0;
  while (off < n) {
    ssize_t w = write(fd, data + off, n - off);
    if (w <= 0) { if (w < 0 && errno == EINTR) continue; break; }
    off += (size_t)w;
  }
  close(fd);
  if (off != n) { snprintf(err, errcap, "write to '%s' failed", rel); return -1; }
  return 0;
}

static int te_read(const char *ws, const char *rel, Sink *k, char *err, int errcap) {
  int fd = te_open(ws, rel, 0, 0, err, errcap);
  if (fd < 0) return -1;
  char buf[4096];
  for (;;) {
    ssize_t r = read(fd, buf, sizeof(buf));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) break;
    sink_put(k, buf, (size_t)r);
  }
  close(fd);
  return 0;
}

#else  // TE_WASM: lexical confinement only (no symlink resolution available)

int toolexec_workspace_init(const char *dir, char *canon, int cap) {
  if (!dir || !dir[0]) dir = ".";
  mkdir(dir, 0755);                              // EEXIST is fine; fopen reports real errors
  int n = snprintf(canon, (size_t)cap, "%s", dir);
  return (n < 0 || n >= cap) ? -1 : 0;
}

static int te_write(const char *ws, const char *rel, const char *data, size_t n, int auto_mode,
                    char *err, int errcap) {
  (void)auto_mode;
  char full[TE_PATH_MAX];
  if (join_path(ws, rel, full, sizeof(full)) != 0) { snprintf(err, errcap, "path too long"); return -1; }
  FILE *f = fopen(full, "wb");
  if (!f) { snprintf(err, errcap, "cannot write '%s'", rel); return -1; }
  size_t put = fwrite(data, 1, n, f);
  fclose(f);
  if (put != n) { snprintf(err, errcap, "write to '%s' failed", rel); return -1; }
  return 0;
}

static int te_read(const char *ws, const char *rel, Sink *k, char *err, int errcap) {
  char full[TE_PATH_MAX], buf[4096];
  if (join_path(ws, rel, full, sizeof(full)) != 0) { snprintf(err, errcap, "path too long"); return -1; }
  FILE *f = fopen(full, "rb");
  if (!f) { snprintf(err, errcap, "cannot read '%s'", rel); return -1; }
  size_t r;
  while ((r = fread(buf, 1, sizeof(buf), f)) > 0) sink_put(k, buf, r);
  fclose(f);
  return 0;
}

#endif

// ---------------------------------------------------------------------------
// run_command: cwd = workspace, stdin = null device, stdout+stderr captured,
// drained to EOF (excess counted, not kept), wall-clock timeout.
// Returns 0 if it ran (exit status in *status, *timed_out set), -1 if it could
// not be started (message in err).
// ---------------------------------------------------------------------------

#if defined(TE_WIN)

static int te_spawn(const char *ws, const char *cmd, int timeout_s, Sink *k,
                    int *status, int *timed_out, char *err, int errcap) {
  wchar_t comspec[MAX_PATH], wws[TE_PATH_MAX];
  static wchar_t line[8192];
  DWORD cn = GetEnvironmentVariableW(L"ComSpec", comspec, MAX_PATH);
  if (cn == 0 || cn >= MAX_PATH) {
    UINT sn = GetSystemDirectoryW(comspec, MAX_PATH - 16);
    if (sn == 0 || sn >= MAX_PATH - 16) { snprintf(err, errcap, "cannot locate cmd.exe"); return -1; }
    wcscat(comspec, L"\\cmd.exe");
  }
  wchar_t wcmd[4096];
  if (!to_w(cmd, wcmd, 4096) || !to_w((ws && ws[0]) ? ws : ".", wws, TE_PATH_MAX)) {
    snprintf(err, errcap, "command/workspace not valid UTF-8"); return -1;
  }
  // Same shape as _popen: cmd.exe /d /s /c "<command>"  (/d: skip AutoRun).
  _snwprintf(line, 8192, L"\"%ls\" /d /s /c \"%ls\"", comspec, wcmd);
  line[8191] = 0;

  SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
  HANDLE rd = NULL, wr = NULL;
  if (!CreatePipe(&rd, &wr, &sa, 0)) { snprintf(err, errcap, "cannot create pipe"); return -1; }
  SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
  HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                           OPEN_EXISTING, 0, NULL);
  HANDLE job = CreateJobObjectW(NULL, NULL);
  if (job) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION jl;
    memset(&jl, 0, sizeof(jl));
    jl.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &jl, sizeof(jl));
  }
  STARTUPINFOW si;
  PROCESS_INFORMATION pi;
  memset(&si, 0, sizeof(si));
  memset(&pi, 0, sizeof(pi));
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = nul;
  si.hStdOutput = wr;
  si.hStdError = wr;
  BOOL ok = CreateProcessW(comspec, line, NULL, NULL, TRUE, CREATE_SUSPENDED | CREATE_NO_WINDOW,
                           NULL, wws, &si, &pi);
  CloseHandle(wr);
  if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
  if (!ok) {
    CloseHandle(rd);
    if (job) CloseHandle(job);
    snprintf(err, errcap, "cannot run command (error %lu)", (unsigned long)GetLastError());
    return -1;
  }
  if (job) AssignProcessToJobObject(job, pi.hProcess);
  ResumeThread(pi.hThread);
  CloseHandle(pi.hThread);

  ULONGLONG deadline = GetTickCount64() + (ULONGLONG)timeout_s * 1000ULL;
  int exited = 0, idle = 0;
  *timed_out = 0;
  char buf[4096];
  for (;;) {
    DWORD avail = 0;
    if (!PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL)) break;   // all writers gone: EOF
    if (avail) {
      DWORD got = 0;
      if (!ReadFile(rd, buf, avail < sizeof(buf) ? avail : (DWORD)sizeof(buf), &got, NULL) || !got) break;
      sink_put(k, buf, got);
      idle = 0;
      continue;
    }
    // A background grandchild may keep the pipe open after cmd.exe exits:
    // stop after ~200 ms of silence; closing the job kills it.
    if (exited && ++idle > 20) break;
    if (WaitForSingleObject(pi.hProcess, 10) == WAIT_OBJECT_0) exited = 1;
    if (GetTickCount64() > deadline) {
      *timed_out = 1;
      if (job) TerminateJobObject(job, 1); else TerminateProcess(pi.hProcess, 1);
      break;
    }
  }
  // EOF can come before the process exits (it may close its output early):
  // keep honouring the deadline while waiting for the exit code.
  if (!*timed_out) {
    ULONGLONG now = GetTickCount64();
    DWORD left = now < deadline ? (DWORD)(deadline - now) : 0;
    if (WaitForSingleObject(pi.hProcess, left) != WAIT_OBJECT_0) {
      *timed_out = 1;
      if (job) TerminateJobObject(job, 1); else TerminateProcess(pi.hProcess, 1);
    }
  }
  WaitForSingleObject(pi.hProcess, 2000);
  DWORD code = 0;
  GetExitCodeProcess(pi.hProcess, &code);
  *status = (int)code;
  CloseHandle(pi.hProcess);
  CloseHandle(rd);
  if (job) CloseHandle(job);                       // KILL_ON_JOB_CLOSE: no stragglers
  return 0;
}

#elif defined(TE_POSIX)

static long long now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000LL + ts.tv_nsec / 1000000L;
}

static int te_spawn(const char *ws, const char *cmd, int timeout_s, Sink *k,
                    int *status, int *timed_out, char *err, int errcap) {
  int pfd[2];
  if (pipe(pfd) != 0) { snprintf(err, errcap, "cannot create pipe"); return -1; }
  pid_t pid = fork();
  if (pid < 0) { close(pfd[0]); close(pfd[1]); snprintf(err, errcap, "cannot fork"); return -1; }
  if (pid == 0) {
    setpgid(0, 0);                                 // own group, so a timeout kills descendants
    if (chdir((ws && ws[0]) ? ws : ".") != 0) _exit(126);
    int nul = open("/dev/null", O_RDONLY);
    if (nul >= 0) { dup2(nul, 0); close(nul); }
    dup2(pfd[1], 1);
    dup2(pfd[1], 2);
    close(pfd[0]);
    close(pfd[1]);
    execl("/bin/sh", "sh", "-c", cmd, (char*)NULL);
    _exit(127);
  }
  setpgid(pid, pid);
  close(pfd[1]);
  long long deadline = now_ms() + (long long)timeout_s * 1000LL;
  int exited = 0, idle = 0, st = 0;
  *timed_out = 0;
  char buf[4096];
  for (;;) {
    struct pollfd pf = { pfd[0], POLLIN, 0 };
    int pr = poll(&pf, 1, 10);
    if (pr > 0) {
      ssize_t r = read(pfd[0], buf, sizeof(buf));
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) break;                           // EOF
      sink_put(k, buf, (size_t)r);
      idle = 0;
      continue;
    }
    if (!exited && waitpid(pid, &st, WNOHANG) == pid) exited = 1;
    if (exited && ++idle > 20) break;              // grandchild holding the pipe
    if (now_ms() > deadline) { *timed_out = 1; break; }
  }
  close(pfd[0]);
  // EOF can come before the shell exits: reap it, still honouring the deadline.
  while (!exited && !*timed_out) {
    if (waitpid(pid, &st, WNOHANG) == pid) { exited = 1; break; }
    if (now_ms() > deadline) { *timed_out = 1; break; }
    poll(NULL, 0, 10);
  }
  kill(-pid, SIGKILL);                             // stragglers / timeout
  if (!exited) waitpid(pid, &st, 0);
  if (WIFEXITED(st)) *status = WEXITSTATUS(st);
  else if (WIFSIGNALED(st)) *status = 128 + WTERMSIG(st);
  else *status = -1;
  return 0;
}

#endif

// ---------------------------------------------------------------------------

int toolexec_run(const ToolCall *tc, const ToolExecPolicy *pol, char *obs, int obs_size) {
  static const ToolExecPolicy def = { NULL, 0, 0 };
  if (!pol) pol = &def;
  char err[768];
  obs[0] = '\0';
  switch (tc->kind) {
    case TOOL_WRITE_FILE:
    case TOOL_READ_FILE: {
      int wr = tc->kind == TOOL_WRITE_FILE;
      const char *why = toolexec_path_reason(tc->path);
      if (!why && wr && pol->auto_mode) why = toolexec_write_policy(tc->path);
      if (why) {
        snprintf(obs, obs_size, "error: refused unsafe path '%s': %s", tc->path, why);
        return 1;
      }
      if (wr) {
        size_t n = strlen(tc->body);
        if (te_write(pol->workspace, tc->path, tc->body, n, pol->auto_mode, err, sizeof(err)) != 0) {
          snprintf(obs, obs_size, "error: %s", err);
          return 1;
        }
        snprintf(obs, obs_size, "wrote %zu bytes to %s", n, tc->path);
        return 0;
      }
      Sink k = { obs, (size_t)content_cap(obs_size), 0, 0 };
      if (te_read(pol->workspace, tc->path, &k, err, sizeof(err)) != 0) {
        snprintf(obs, obs_size, "error: %s", err);
        return 1;
      }
      finish_obs(obs, obs_size, &k);
      return 0;
    }
    case TOOL_RUN_COMMAND: {
#if defined(TE_WASM)
      (void)err;
      snprintf(obs, obs_size, "error: run_command is not supported on WebAssembly");
      return 1;
#else
      int timeout = pol->timeout_s > 0 ? pol->timeout_s : TE_DEFAULT_TIMEOUT;
      int status = 0, timed_out = 0;
      Sink k = { obs, (size_t)content_cap(obs_size), 0, 0 };
      if (te_spawn(pol->workspace, tc->command, timeout, &k, &status, &timed_out, err, sizeof(err)) != 0) {
        snprintf(obs, obs_size, "error: %s", err);
        return 1;
      }
      finish_obs(obs, obs_size, &k);
      char note[96];
      if (timed_out) {
        snprintf(note, sizeof(note), "[timed out after %d s; killed]", timeout);
        append_note(obs, obs_size, note);
        return 1;
      }
      if (status != 0) {
        snprintf(note, sizeof(note), "[exit status %d]", status);
        append_note(obs, obs_size, note);
        return 1;
      }
      return 0;
#endif
    }
    default:
      snprintf(obs, obs_size, "error: unknown tool");
      return 1;
  }
}
