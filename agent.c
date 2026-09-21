// agent.c - Minimal agent loop with a pluggable model backend

// popen/pclose are POSIX; glibc/musl hide them under -std=c11 without this.
#if !defined(_WIN32) && !defined(__wasi__) && !defined(__wasm__)
#define _POSIX_C_SOURCE 200809L
#endif
// rand_s (CRT, backed by the OS CSPRNG) for the per-run observation nonce.
#if defined(_WIN32)
#define _CRT_RAND_S
#endif

#include "agent.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
  // Drain to EOF so the backend never blocks on a full pipe; keep what fits.
  int n = 0;
  char sink[4096];
  size_t r;
  while ((r = fread(n < out_cap - 1 ? out + n : sink, 1,
                    n < out_cap - 1 ? (size_t)(out_cap - 1 - n) : sizeof(sink), p)) > 0)
    if (n < out_cap - 1) n += (int)r;
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
// Sanitization / observation framing
// ---------------------------------------------------------------------------

// Strip everything a terminal could interpret: C0 controls except \n and \t
// (\r too: it can overwrite the visible line), DEL, C1 controls (U+0080-U+009F),
// bidi embedding/override/isolate controls (U+202A-202E, U+2066-2069: they
// reorder what the approver sees), and invalid UTF-8 bytes (-> '?'; a lone
// 0x9B is CSI on 8-bit terminals). In place; output is never longer.
void agent_sanitize(char *s) {
  unsigned char *r = (unsigned char*)s, *w = (unsigned char*)s;
  while (*r) {
    unsigned c = *r;
    if (c < 0x80) {
      if (c == '\n' || c == '\t' || (c >= 0x20 && c != 0x7F)) *w++ = (unsigned char)c;
      r++;
      continue;
    }
    int len = (c >= 0xC2 && c <= 0xDF) ? 2 : (c >= 0xE0 && c <= 0xEF) ? 3 : (c >= 0xF0 && c <= 0xF4) ? 4 : 0;
    unsigned cp = len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    int ok = len > 0;
    for (int k = 1; ok && k < len; k++) {
      if ((r[k] & 0xC0) != 0x80) ok = 0;          // also stops at the NUL
      else cp = (cp << 6) | (r[k] & 0x3F);
    }
    if (ok && ((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) ||
               (cp >= 0xD800 && cp <= 0xDFFF)))
      ok = 0;                                      // overlong / out of range / surrogate
    if (!ok) { *w++ = '?'; r++; continue; }
    int drop = (cp >= 0x80 && cp <= 0x9F) || (cp >= 0x202A && cp <= 0x202E) ||
               (cp >= 0x2066 && cp <= 0x2069);
    if (!drop) for (int k = 0; k < len; k++) *w++ = r[k];
    r += len;
  }
  *w = '\0';
}

static const char *const ROLE_MARKERS[] = {
  "MODEL:", "GOAL:", "OBSERVATION:", "USER:", "ASSISTANT:", "SYSTEM:", "HUMAN:"
};

static int starts_ci(const char *p, const char *w) {
  for (; *w; p++, w++) {
    int a = (unsigned char)*p, b = (unsigned char)*w;
    if (a >= 'a' && a <= 'z') a -= 32;
    if (a != b) return 0;
  }
  return 1;
}

// Copy src (already sanitized) to out, defusing anything that could be read as
// transcript structure: a role marker at the start of a line (or right after a
// '|', the separator tools/agent_local.py flattens lines with) gets a "> "
// prefix, and "[[obs:" / "[[/obs:" frame markers are broken up. Returns the
// number of bytes written, or -1 if out is too small.
static int neutralize(const char *src, char *out, int cap) {
  int n = 0, boundary = 1;
#define PUT(ch) do { if (n >= cap - 1) return -1; out[n++] = (ch); } while (0)
  for (const char *p = src; *p; p++) {
    if (boundary && *p != ' ' && *p != '\t') {
      for (size_t i = 0; i < sizeof(ROLE_MARKERS) / sizeof(ROLE_MARKERS[0]); i++)
        if (starts_ci(p, ROLE_MARKERS[i])) { PUT('>'); PUT(' '); break; }
      boundary = 0;
    }
    if (p[0] == '[' && p[1] == '[' && (strncmp(p + 2, "obs:", 4) == 0 || strncmp(p + 2, "/obs:", 5) == 0)) {
      PUT('['); PUT(' ');
      continue;
    }
    PUT(*p);
    if (*p == '\n' || *p == '|') boundary = 1;
  }
#undef PUT
  out[n] = '\0';
  return n;
}

int agent_format_observation(const char *obs, const char *nonce, char *out, int cap) {
  size_t len = strlen(obs);
  char *tmp = (char*)malloc(len + 1);
  if (!tmp) return -1;
  memcpy(tmp, obs, len + 1);
  agent_sanitize(tmp);
  int h = snprintf(out, (size_t)cap, "OBSERVATION: [[obs:%s]]\n", nonce);
  if (h < 0 || h >= cap) { free(tmp); return -1; }
  int b = neutralize(tmp, out + h, cap - h);
  free(tmp);
  if (b < 0) return -1;
  int t = snprintf(out + h + b, (size_t)(cap - h - b), "\n[[/obs:%s]]\n", nonce);
  return (t < 0 || t >= cap - h - b) ? -1 : 0;
}

// 64 bits from the OS CSPRNG when available, else a time/address mix. The
// nonce only has to be unpredictable to the tool output of this run.
static void make_nonce(char out[17]) {
  uint64_t v = 0;
  int got = 0;
#if defined(_WIN32)
  unsigned int a, b;
  if (rand_s(&a) == 0 && rand_s(&b) == 0) { v = ((uint64_t)a << 32) | b; got = 1; }
#elif !defined(__wasi__) && !defined(__wasm__)
  FILE *f = fopen("/dev/urandom", "rb");
  if (f) { got = fread(&v, sizeof(v), 1, f) == 1; fclose(f); }
#endif
  if (!got) {
    v = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)&v;
#if !defined(__wasi__) && !defined(__wasm__)
    v ^= (uint64_t)clock() << 20;
#endif
    v += 0x9E3779B97F4A7C15ULL;                    // splitmix64 finalizer
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
    v ^= v >> 31;
  }
  snprintf(out, 17, "%016llx", (unsigned long long)v);
}

// ---------------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------------

#define AGENT_MAX_STEPS 12
#define TRANSCRIPT_CAP  16384

// Append s; -1 (transcript untouched) if it does not fit.
static int tappend(char *t, int cap, const char *s) {
  size_t len = strlen(t), n = strlen(s);
  if (len + n + 1 > (size_t)cap) return -1;
  memcpy(t + len, s, n + 1);
  return 0;
}

AgentConfig agent_default_config(void) {
  AgentConfig c;
  memset(&c, 0, sizeof(c));
  c.mode = APPROVAL_ASK;
  return c;
}

#define SAY(...) do { if (!cfg->quiet) printf(__VA_ARGS__); } while (0)

int agent_run(const char *goal, AgentBackend *backend, const AgentConfig *cfg_in) {
  AgentConfig defcfg = agent_default_config();
  const AgentConfig *cfg = cfg_in ? cfg_in : &defcfg;
  ToolExecPolicy pol = { cfg->workspace, cfg->mode == APPROVAL_AUTO, cfg->timeout_s };
  static char transcript[TRANSCRIPT_CAP];
  static char response[8192];
  static char shown[8192];
  static char block[12288];
  char obs[4096];
  char nonce[17];
  int rc = 0;
  make_nonce(nonce);

  char g[1024];
  snprintf(g, sizeof(g), "%s", goal);
  agent_sanitize(g);
  for (char *q = g; *q; q++) if (*q == '\n') *q = ' ';   // the goal is one line
  snprintf(transcript, sizeof(transcript), "GOAL: %s\n", g);
  SAY("Goal: %s\n", g);
  if (cfg->workspace && cfg->workspace[0]) SAY("Workspace: %s\n", cfg->workspace);
  SAY("\n");

  for (int step = 0; step < AGENT_MAX_STEPS; step++) {
    SAY("--- step %d ---\n", step + 1);
    if (backend->respond(backend, transcript, response, sizeof(response)) != 0) {
      SAY("backend error; stopping.\n");
      break;
    }
    // The tool call is parsed from the raw response (a write_file body keeps its
    // bytes); everything printed or stored is the sanitized copy.
    snprintf(shown, sizeof(shown), "%s", response);
    agent_sanitize(shown);
    snprintf(block, sizeof(block), "MODEL: ");
    if (neutralize(shown, block + 7, (int)sizeof(block) - 8) < 0 ||
        tappend(block, sizeof(block), "\n") != 0 ||
        tappend(transcript, sizeof(transcript), block) != 0) {
      SAY("error: transcript full (%d bytes); stopping the run.\n", TRANSCRIPT_CAP);
      rc = 2;
      break;
    }

    ToolCall tc;
    if (!toolcall_parse(response, &tc)) {
      SAY("model (final): %s\n", shown);
      break;
    }
    // What the user approves must be exactly what runs.
    agent_sanitize(tc.path);
    agent_sanitize(tc.command);
    if (tc.kind == TOOL_DONE) {
      agent_sanitize(tc.body);
      SAY("done: %s\n", tc.body);
      break;
    }

    char action[1400];
    // run_command executes arbitrary shell input from the model, so it is NEVER
    // auto-approved: it always requires an explicit interactive y/N, even in
    // APPROVAL_AUTO mode (where a non-interactive stdin denies it by default).
    ApprovalMode step_mode = cfg->mode;
    switch (tc.kind) {
      case TOOL_WRITE_FILE:
        snprintf(action, sizeof(action), "write_file '%s' (%zu bytes)", tc.path, strlen(tc.body));
        // Show the full content being written so approval is meaningful.
        snprintf(shown, sizeof(shown), "%s", tc.body);
        agent_sanitize(shown);
        SAY("--- content of %s ---\n%s\n--- end content ---\n", tc.path, shown);
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
    SAY("tool requested: %s\n", action);

    int allowed;
    if (cfg->quiet) allowed = step_mode == APPROVAL_AUTO;   // quiet runs never read stdin
    else allowed = approval_confirm(step_mode, action);
    if (!allowed) snprintf(obs, sizeof(obs), "denied by user");
    else toolexec_run(&tc, &pol, obs, sizeof(obs));

    if (agent_format_observation(obs, nonce, block, sizeof(block)) != 0 ||
        tappend(transcript, sizeof(transcript), block) != 0) {
      SAY("error: transcript full (%d bytes); stopping the run.\n", TRANSCRIPT_CAP);
      rc = 2;
      break;
    }
    agent_sanitize(obs);
    SAY("observation: %s\n\n", obs);
  }
  if (cfg->transcript_out && cfg->transcript_out_cap > 0)
    snprintf(cfg->transcript_out, (size_t)cfg->transcript_out_cap, "%s", transcript);
  return rc;
}
