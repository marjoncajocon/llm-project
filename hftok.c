// hftok.c - pure-C byte-level BPE tokenizer (GPT-2 / LLaMA-family compatible).
#include "hftok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#include "unitab.h"

// --- load ------------------------------------------------------------------

static uint32_t hash_bytes(const unsigned char *b, int len) {
  uint32_t h = 2166136261u;
  for (int i = 0; i < len; i++) { h ^= b[i]; h *= 16777619u; }
  return h;
}

int hftok_load(HFTok *t, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "error: cannot open tokenizer '%s'\n", path); return 1; }
  char magic[4];
  uint32_t ver, n;
  int32_t eos;
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "HFTK", 4) != 0 ||
      fread(&ver, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1 || fread(&eos, 4, 1, f) != 1) {
    fprintf(stderr, "error: '%s' is not a .tok file\n", path); fclose(f); return 1;
  }
  if (n == 0 || n > 5000000) { fprintf(stderr, "error: bad vocab size\n"); fclose(f); return 1; }

  t->n_vocab = (int)n;
  t->eos = eos;
  t->bos = -1;
  t->n_special = 0;
  t->special_id = NULL; t->special_len = NULL; t->special_str = NULL; t->special_eog = NULL;
  t->template_kind = HFTOK_TMPL_PLAIN;
  t->default_system = NULL;
  t->pretok = HFTOK_PRE_GPT2;          // v1/v2 files carry no pretokenizer kind
  t->tok_len = (int*)malloc((size_t)n * sizeof(int));
  t->tok_bytes = (unsigned char**)malloc((size_t)n * sizeof(unsigned char*));
  for (int i = 0; i < 256; i++) t->byte2id[i] = -1;

  for (uint32_t i = 0; i < n; i++) {
    uint16_t len;
    if (fread(&len, 2, 1, f) != 1) { fprintf(stderr, "error: truncated tok\n"); fclose(f); return 1; }
    t->tok_len[i] = len;
    t->tok_bytes[i] = (unsigned char*)malloc(len ? len : 1);
    if (len && fread(t->tok_bytes[i], 1, len, f) != len) { fprintf(stderr, "error: truncated bytes\n"); fclose(f); return 1; }
    if (len == 1) t->byte2id[t->tok_bytes[i][0]] = (int)i;
  }

  // v2 tail: bos, template kind, default system prompt, special tokens.
  if (ver >= 2) {
    int32_t bos; uint8_t kind; uint16_t syslen; uint32_t nsp;
    if (fread(&bos, 4, 1, f) != 1 || fread(&kind, 1, 1, f) != 1 || fread(&syslen, 2, 1, f) != 1) {
      fprintf(stderr, "error: truncated v2 header\n"); fclose(f); return 1;
    }
    t->bos = bos; t->template_kind = kind;
    t->default_system = (char*)malloc((size_t)syslen + 1);
    if (syslen && fread(t->default_system, 1, syslen, f) != syslen) { fprintf(stderr, "error: truncated system\n"); fclose(f); return 1; }
    t->default_system[syslen] = '\0';
    if (fread(&nsp, 4, 1, f) != 1 || nsp > 100000) { fprintf(stderr, "error: bad n_special\n"); fclose(f); return 1; }
    t->n_special = (int)nsp;
    t->special_id = (int*)malloc((size_t)(nsp ? nsp : 1) * sizeof(int));
    t->special_len = (int*)malloc((size_t)(nsp ? nsp : 1) * sizeof(int));
    t->special_str = (unsigned char**)malloc((size_t)(nsp ? nsp : 1) * sizeof(unsigned char*));
    t->special_eog = (int*)malloc((size_t)(nsp ? nsp : 1) * sizeof(int));
    for (uint32_t i = 0; i < nsp; i++) {
      int32_t sid; uint8_t eog; uint16_t slen;
      if (fread(&sid, 4, 1, f) != 1 || fread(&eog, 1, 1, f) != 1 || fread(&slen, 2, 1, f) != 1) {
        fprintf(stderr, "error: truncated special\n"); fclose(f); return 1;
      }
      t->special_id[i] = sid; t->special_eog[i] = eog; t->special_len[i] = slen;
      t->special_str[i] = (unsigned char*)malloc(slen ? slen : 1);
      if (slen && fread(t->special_str[i], 1, slen, f) != slen) { fprintf(stderr, "error: truncated special str\n"); fclose(f); return 1; }
    }
  }
  // v3 tail: one byte, the pretokenizer kind (HFTOK_PRE_*). Unknown kinds are
  // rejected: tokenizing with the wrong split would silently change every id.
  if (ver >= 3) {
    uint8_t pk;
    if (fread(&pk, 1, 1, f) != 1) { fprintf(stderr, "error: truncated v3 pretokenizer kind\n"); fclose(f); return 1; }
    if (pk > HFTOK_PRE_QWEN2) {
      fprintf(stderr, "error: '%s' needs pretokenizer kind %u (this build knows 0..%d)\n",
              path, (unsigned)pk, HFTOK_PRE_QWEN2);
      fclose(f); return 1;
    }
    t->pretok = pk;
  }
  fclose(f);

  // build the bytes->id hash map (open addressing, power-of-two capacity).
  int cap = 1;
  while (cap < (int)n * 2) cap <<= 1;
  t->hcap = cap;
  t->hmap = (int*)malloc((size_t)cap * sizeof(int));
  for (int i = 0; i < cap; i++) t->hmap[i] = -1;
  for (int i = 0; i < (int)n; i++) {
    if (t->tok_len[i] == 0) continue;
    uint32_t h = hash_bytes(t->tok_bytes[i], t->tok_len[i]) & (cap - 1);
    while (t->hmap[h] != -1) h = (h + 1) & (cap - 1);
    t->hmap[h] = i;
  }
  return 0;
}

void hftok_free(HFTok *t) {
  for (int i = 0; i < t->n_vocab; i++) free(t->tok_bytes[i]);
  free(t->tok_bytes); free(t->tok_len); free(t->hmap);
  for (int i = 0; i < t->n_special; i++) free(t->special_str[i]);
  free(t->special_id); free(t->special_len); free(t->special_str); free(t->special_eog);
  free(t->default_system);
  t->tok_bytes = NULL; t->tok_len = NULL; t->hmap = NULL; t->n_vocab = 0;
  t->special_str = NULL; t->n_special = 0; t->default_system = NULL;
}

static int hftok_lookup(const HFTok *t, const unsigned char *b, int len) {
  uint32_t h = hash_bytes(b, len) & (t->hcap - 1);
  while (t->hmap[h] != -1) {
    int id = t->hmap[h];
    if (t->tok_len[id] == len && memcmp(t->tok_bytes[id], b, (size_t)len) == 0) return id;
    h = (h + 1) & (t->hcap - 1);
  }
  return -1;
}

// --- pretokenization (GPT-2 regex, hand-coded) -----------------------------
// 's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
// \p{L} approximated as ASCII letters + any byte >= 0x80 (UTF-8 lead/cont).

static int is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
static int is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static int is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
static int is_other(unsigned char c) { return !is_space(c) && !is_letter(c) && !is_digit(c); }

// Advance *pos over the next pretoken; set *cs (start) and *cl (length). Returns 0 at end.
static int next_chunk(const char *s, int len, int *pos, int *cs, int *cl) {
  int p = *pos;
  if (p >= len) return 0;
  const unsigned char *u = (const unsigned char*)s;

  // contractions
  if (u[p] == '\'' && p + 1 < len) {
    unsigned char a = u[p + 1];
    if (a == 's' || a == 't' || a == 'm' || a == 'd') { *cs = p; *cl = 2; *pos = p + 2; return 1; }
    if (p + 2 < len) {
      unsigned char b = u[p + 2];
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) { *cs = p; *cl = 3; *pos = p + 3; return 1; }
    }
  }

  int start = p;
  int sp = (u[p] == ' ') ? 1 : 0;   // optional single leading space
  int c = p + sp;
  if (c < len && is_letter(u[c])) {
    p = c; while (p < len && is_letter(u[p])) p++;
    *cs = start; *cl = p - start; *pos = p; return 1;
  }
  if (c < len && is_digit(u[c])) {
    p = c; while (p < len && is_digit(u[p])) p++;
    *cs = start; *cl = p - start; *pos = p; return 1;
  }
  if (c < len && is_other(u[c])) {
    p = c; while (p < len && is_other(u[p])) p++;
    *cs = start; *cl = p - start; *pos = p; return 1;
  }
  // whitespace run: leave the last space for the next run if a non-space follows.
  int ws = start;
  while (ws < len && is_space(u[ws])) ws++;
  if (ws < len && ws - 1 > start) { *cs = start; *cl = (ws - 1) - start; *pos = ws - 1; return 1; }
  *cs = start; *cl = ws - start; *pos = ws; return 1;
}

// --- pretokenization (Qwen2 / Qwen2.5 / Qwen3 regex, hand-coded) -----------
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// Differences from GPT-2: numbers are SINGLE code points; a letter run may take
// ONE leading char that is not CR/LF/letter/number (space, tab, '(', '.', an
// emoji...), not just a space; punctuation runs swallow trailing newlines;
// contractions are case-insensitive. Classes are real Unicode (unitab.h, generated
// by tools/gen_unitab.py), evaluated on decoded code points. The alternatives are
// tried in order at each position and the first that matches wins (a backtracking
// engine's leftmost-first rule, as in HF tokenizers' onig), which is what the
// branches below reproduce. Invalid UTF-8 bytes are treated as one "other" char
// each (HF never sees them: its input is a valid Rust String).

static int uni_in(const uint32_t (*tab)[2], int n, uint32_t cp) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    if (cp < tab[mid][0]) hi = mid - 1;
    else if (cp > tab[mid][1]) lo = mid + 1;
    else return 1;
  }
  return 0;
}
#define UNI_COUNT(t) ((int)(sizeof(t) / sizeof((t)[0])))

static int u_is_L(uint32_t c) {
  if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  return uni_in(UNI_L, UNI_COUNT(UNI_L), c);
}
static int u_is_N(uint32_t c) {
  if (c < 0x80) return c >= '0' && c <= '9';
  return uni_in(UNI_N, UNI_COUNT(UNI_N), c);
}
static int u_is_S(uint32_t c) {
  if (c < 0x80) return c == ' ' || (c >= 0x09 && c <= 0x0D);
  return uni_in(UNI_S, UNI_COUNT(UNI_S), c);
}
static int u_is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }

// Decode the code point at u[p] (p < len); *n = its byte length. Malformed or
// truncated sequences decode as one byte with cp 0xFFFFFFFF (an "other" char:
// not a letter, number or space).
static uint32_t u_decode(const unsigned char *u, int len, int p, int *n) {
  const unsigned char c = u[p];
  if (c < 0x80) { *n = 1; return c; }
  int k; uint32_t cp, minv;
  if ((c & 0xE0) == 0xC0)      { k = 2; cp = c & 0x1F; minv = 0x80; }
  else if ((c & 0xF0) == 0xE0) { k = 3; cp = c & 0x0F; minv = 0x800; }
  else if ((c & 0xF8) == 0xF0) { k = 4; cp = c & 0x07; minv = 0x10000; }
  else { *n = 1; return 0xFFFFFFFFu; }
  if (p + k > len) { *n = 1; return 0xFFFFFFFFu; }
  for (int i = 1; i < k; i++) {
    if ((u[p + i] & 0xC0) != 0x80) { *n = 1; return 0xFFFFFFFFu; }
    cp = (cp << 6) | (u[p + i] & 0x3F);
  }
  if (cp < minv || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) { *n = 1; return 0xFFFFFFFFu; }
  *n = k;
  return cp;
}

// Case-insensitive contraction letter test. Onig's (?i) folds U+017F (long s)
// to 's' too; 'K' (Kelvin) etc. fold to letters that no contraction uses.
static int ci_eq(uint32_t c, char lower) {
  if (c == (uint32_t)lower || c == (uint32_t)(lower - 'a' + 'A')) return 1;
  return lower == 's' && c == 0x17F;
}

static int next_chunk_qwen2(const char *s, int len, int *pos, int *cs, int *cl) {
  const int p = *pos;
  if (p >= len) return 0;
  const unsigned char *u = (const unsigned char*)s;
  int n0, n1 = 0, n2 = 0;
  const uint32_t c0 = u_decode(u, len, p, &n0);
  const uint32_t c1 = (p + n0 < len) ? u_decode(u, len, p + n0, &n1) : 0xFFFFFFFEu;   // FE = end
  *cs = p;

  // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
  if (c0 == '\'' && c1 != 0xFFFFFFFEu) {
    if (ci_eq(c1, 's') || ci_eq(c1, 't') || ci_eq(c1, 'm') || ci_eq(c1, 'd')) {
      *cl = n0 + n1; *pos = p + *cl; return 1;
    }
    if (p + n0 + n1 < len) {
      const uint32_t c2 = u_decode(u, len, p + n0 + n1, &n2);
      if ((ci_eq(c1, 'r') && ci_eq(c2, 'e')) || (ci_eq(c1, 'v') && ci_eq(c2, 'e')) ||
          (ci_eq(c1, 'l') && ci_eq(c2, 'l'))) {
        *cl = n0 + n1 + n2; *pos = p + *cl; return 1;
      }
    }
  }

  // 2. [^\r\n\p{L}\p{N}]?\p{L}+
  int q = -1;
  if (u_is_L(c0)) q = p;
  else if (!u_is_crlf(c0) && !u_is_N(c0) && c1 != 0xFFFFFFFEu && u_is_L(c1)) q = p + n0;
  if (q >= 0) {
    int e = q, n;
    while (e < len && u_is_L(u_decode(u, len, e, &n))) e += n;
    *cl = e - p; *pos = e; return 1;
  }

  // 3. \p{N}  (a single code point)
  if (u_is_N(c0)) { *cl = n0; *pos = p + n0; return 1; }

  // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
  {
    int r = p, n;
    if (c0 == ' ') r = p + n0;
    int e = r;
    while (e < len) {
      const uint32_t c = u_decode(u, len, e, &n);
      if (u_is_S(c) || u_is_L(c) || u_is_N(c)) break;
      e += n;
    }
    if (e > r) {
      while (e < len && (u[e] == '\r' || u[e] == '\n')) e++;
      *cl = e - p; *pos = e; return 1;
    }
  }

  // 5-7. Whitespace. c0 is \s here (every other char matched above).
  int e = p, last_nl = -1, last_start = p, n;
  while (e < len) {
    const uint32_t c = u_decode(u, len, e, &n);
    if (!u_is_S(c)) break;
    if (u_is_crlf(c)) last_nl = e;
    last_start = e;
    e += n;
  }
  if (e == p) { *cl = n0; *pos = p + n0; return 1; }   // unreachable safety net
  if (last_nl >= 0) {                        // 5. \s*[\r\n]+ : up to the LAST CR/LF
    *cl = last_nl + 1 - p; *pos = last_nl + 1; return 1;
  }
  if (e < len && last_start > p) {           // 6. \s+(?!\S) : leave the last char
    *cl = last_start - p; *pos = last_start; return 1;
  }
  *cl = e - p; *pos = e; return 1;           // 6 at end of text / 7. \s+
}

int hftok_pretok_split(int kind, const char *text, int len, int *starts, int *lens, int max) {
  int pos = 0, cs, cl, n = 0;
  while (n < max && (kind == HFTOK_PRE_QWEN2 ? next_chunk_qwen2(text, len, &pos, &cs, &cl)
                                             : next_chunk(text, len, &pos, &cs, &cl))) {
    starts[n] = cs; lens[n] = cl; n++;
  }
  return n;
}

// --- encode / decode -------------------------------------------------------

// Byte-level BPE encode a plain run text[start,end) (no special tokens inside).
static int encode_run(const HFTok *t, const char *text, int start, int end,
                      int *ids, int n, int max_ids, unsigned char *buf, int *pieces) {
  const char *s = text + start;
  int len = end - start;
  int pos = 0, cs, cl;
  const int qwen = (t->pretok == HFTOK_PRE_QWEN2);
  while (n < max_ids && (qwen ? next_chunk_qwen2(s, len, &pos, &cs, &cl)
                              : next_chunk(s, len, &pos, &cs, &cl))) {
    int npc = 0;
    for (int i = 0; i < cl; i++) {
      int id = t->byte2id[(unsigned char)s[cs + i]];
      pieces[npc++] = (id >= 0) ? id : 0;
    }
    // BPE: repeatedly merge the adjacent pair with the lowest combined-token id.
    for (;;) {
      int best_i = -1, best_id = INT_MAX;
      for (int i = 0; i + 1 < npc; i++) {
        int a = pieces[i], b = pieces[i + 1];
        int la = t->tok_len[a], lb = t->tok_len[b];
        memcpy(buf, t->tok_bytes[a], (size_t)la);
        memcpy(buf + la, t->tok_bytes[b], (size_t)lb);
        int mid = hftok_lookup(t, buf, la + lb);
        if (mid >= 0 && mid < best_id) { best_id = mid; best_i = i; }
      }
      if (best_i < 0) break;
      pieces[best_i] = best_id;
      for (int j = best_i + 1; j < npc - 1; j++) pieces[j] = pieces[j + 1];
      npc--;
    }
    for (int i = 0; i < npc && n < max_ids; i++) ids[n++] = pieces[i];
  }
  return n;
}

// Longest special token whose literal string matches text at pos (or -1).
static int match_special(const HFTok *t, const char *text, int pos, int len, int *out_len) {
  int best = -1, best_len = 0;
  for (int i = 0; i < t->n_special; i++) {
    int sl = t->special_len[i];
    if (sl > best_len && pos + sl <= len &&
        memcmp(text + pos, t->special_str[i], (size_t)sl) == 0) {
      best = t->special_id[i]; best_len = sl;
    }
  }
  *out_len = best_len;
  return best;
}

int hftok_encode(const HFTok *t, const char *text, int *ids, int max_ids) {
  int len = (int)strlen(text);
  int n = 0;
  unsigned char *buf = (unsigned char*)malloc((size_t)len + 2);   // concat scratch
  int *pieces = (int*)malloc(((size_t)len + 1) * sizeof(int));
  int run_start = 0, pos = 0;
  while (pos < len && n < max_ids) {
    // Special tokens all begin with '<'; try to match one greedily here.
    if (text[pos] == '<') {
      int slen = 0, sid = match_special(t, text, pos, len, &slen);
      if (sid >= 0) {
        n = encode_run(t, text, run_start, pos, ids, n, max_ids, buf, pieces);
        if (n < max_ids) ids[n++] = sid;
        pos += slen; run_start = pos;
        continue;
      }
    }
    pos++;
  }
  n = encode_run(t, text, run_start, len, ids, n, max_ids, buf, pieces);
  free(buf); free(pieces);
  return n;
}

int hftok_is_eog(const HFTok *t, int id) {
  if (id == t->eos) return 1;
  for (int i = 0; i < t->n_special; i++)
    if (t->special_id[i] == id && t->special_eog[i]) return 1;
  return 0;
}

// --- chat template ---------------------------------------------------------

static void sb_put(char *out, int *pos, int max, const char *s) {
  while (*s && *pos < max - 1) out[(*pos)++] = *s++;
}

int hftok_chat_render(const HFTok *t, const char **roles, const char **contents,
                      int nturns, char *out, int max_out) {
  int p = 0;
  int has_system = 0;
  for (int i = 0; i < nturns; i++)
    if (strcmp(roles[i], "system") == 0) has_system = 1;

  if (t->template_kind == HFTOK_TMPL_LLAMA3) {
    sb_put(out, &p, max_out, "<|begin_of_text|>");
    if (!has_system && t->default_system && t->default_system[0]) {
      sb_put(out, &p, max_out, "<|start_header_id|>system<|end_header_id|>\n\n");
      sb_put(out, &p, max_out, t->default_system);
      sb_put(out, &p, max_out, "<|eot_id|>");
    }
    for (int i = 0; i < nturns; i++) {
      sb_put(out, &p, max_out, "<|start_header_id|>");
      sb_put(out, &p, max_out, roles[i]);
      sb_put(out, &p, max_out, "<|end_header_id|>\n\n");
      sb_put(out, &p, max_out, contents[i]);
      sb_put(out, &p, max_out, "<|eot_id|>");
    }
    sb_put(out, &p, max_out, "<|start_header_id|>assistant<|end_header_id|>\n\n");
  } else if (t->template_kind == HFTOK_TMPL_CHATML) {
    if (!has_system && t->default_system && t->default_system[0]) {
      sb_put(out, &p, max_out, "<|im_start|>system\n");
      sb_put(out, &p, max_out, t->default_system);
      sb_put(out, &p, max_out, "<|im_end|>\n");
    }
    for (int i = 0; i < nturns; i++) {
      sb_put(out, &p, max_out, "<|im_start|>");
      sb_put(out, &p, max_out, roles[i]);
      sb_put(out, &p, max_out, "\n");
      sb_put(out, &p, max_out, contents[i]);
      sb_put(out, &p, max_out, "<|im_end|>\n");
    }
    sb_put(out, &p, max_out, "<|im_start|>assistant\n");
  } else {   // plain "User:/Assistant:" fallback
    for (int i = 0; i < nturns; i++) {
      const char *r = roles[i];
      const char *label = (strcmp(r, "user") == 0) ? "User: " :
                          (strcmp(r, "assistant") == 0) ? "Assistant: " : "System: ";
      sb_put(out, &p, max_out, label);
      sb_put(out, &p, max_out, contents[i]);
      sb_put(out, &p, max_out, "\n");
    }
    sb_put(out, &p, max_out, "Assistant:");
  }
  out[p] = '\0';
  return p;
}

int hftok_decode(const HFTok *t, const int *ids, int n, char *out, int max_out) {
  int pos = 0;
  for (int i = 0; i < n; i++) {
    int id = ids[i];
    if (id < 0 || id >= t->n_vocab) continue;
    for (int j = 0; j < t->tok_len[id] && pos < max_out - 1; j++) out[pos++] = (char)t->tok_bytes[id][j];
  }
  out[pos] = '\0';
  return pos;
}

// --- CLI test --------------------------------------------------------------
int run_hftok(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: transformer hftok <file.tok> encode \"<text>\"\n"
                    "       transformer hftok <file.tok> decode <id> <id> ...\n"
                    "       transformer hftok <file.tok> count   (text on stdin; prints its token\n"
                    "                                           count, tokenized as hfgen does)\n"
                    "       transformer hftok <file.tok> encode-hex|split-hex  (stdin: one hex-encoded\n"
                    "                                           UTF-8 text per line; prints the ids /\n"
                    "                                           pretoken byte lengths, one line each)\n");
    return 2;
  }
  HFTok t;
  if (hftok_load(&t, argv[2]) != 0) return 1;

  if (strcmp(argv[3], "encode-hex") == 0 || strcmp(argv[3], "split-hex") == 0) {
    // Batch mode for verification against a reference tokenizer: every stdin
    // line is one text, HEX-encoded UTF-8 (so newlines/CRs/NULs-free bytes survive
    // the line protocol); prints one line per text -- the token ids (encode-hex)
    // or the pretoken byte lengths under this file's pretokenizer (split-hex).
    const int split = (argv[3][0] == 's');
    size_t cap = 1 << 16;
    char *line = (char*)malloc(cap);
    for (;;) {
      size_t len = 0; int c;
      while ((c = getchar()) != EOF && c != '\n') {
        if (len + 2 >= cap) { cap *= 2; line = (char*)realloc(line, cap); }
        line[len++] = (char)c;
      }
      if (c == EOF && len == 0) break;
      while (len && (line[len - 1] == '\r' || line[len - 1] == ' ')) len--;
      int nb = (int)(len / 2);
      char *text = (char*)malloc((size_t)nb + 1);
      for (int i = 0; i < nb; i++) {
        unsigned v = 0;
        sscanf(line + 2 * i, "%2x", &v);
        text[i] = (char)v;
      }
      text[nb] = '\0';
      int *a = (int*)malloc(((size_t)nb + 16) * sizeof(int));
      int *b = (int*)malloc(((size_t)nb + 16) * sizeof(int));
      int n = split ? hftok_pretok_split(t.pretok, text, nb, a, b, nb + 16)
                    : hftok_encode(&t, text, a, nb + 16);
      for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", split ? b[i] : a[i]);
      printf("\n");
      free(a); free(b); free(text);
      if (c == EOF) break;
    }
    free(line);
  } else if (strcmp(argv[3], "count") == 0) {
    size_t cap = 1 << 16, len = 0, r;
    char *text = (char*)malloc(cap);
    while ((r = fread(text + len, 1, cap - 1 - len, stdin)) > 0) {
      len += r;
      if (len == cap - 1) { cap *= 2; text = (char*)realloc(text, cap); }
    }
    text[len] = '\0';
    int *ids = (int*)malloc((len + 16) * sizeof(int));
    printf("%d\n", hftok_encode(&t, text, ids, (int)len + 16));
    free(ids); free(text);
  } else if (strcmp(argv[3], "encode") == 0) {
    const char *text = (argc > 4) ? argv[4] : "";
    int cap = (int)strlen(text) + 16;
    int *ids = (int*)malloc((size_t)cap * sizeof(int));
    int n = hftok_encode(&t, text, ids, cap);
    for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", ids[i]);
    printf("\n");
    free(ids);
  } else if (strcmp(argv[3], "decode") == 0) {
    int n = argc - 4;
    int *ids = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    for (int i = 0; i < n; i++) ids[i] = atoi(argv[4 + i]);
    char out[65536];
    hftok_decode(&t, ids, n, out, sizeof(out));
    printf("%s\n", out);
    free(ids);
  } else {
    fprintf(stderr, "unknown op '%s' (encode|decode|count|encode-hex|split-hex)\n", argv[3]);
    hftok_free(&t); return 2;
  }
  hftok_free(&t);
  return 0;
}
