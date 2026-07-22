// hftok.c - pure-C byte-level BPE tokenizer (GPT-2 / LLaMA-family compatible).
#include "hftok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

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
  t->tok_bytes = NULL; t->tok_len = NULL; t->hmap = NULL; t->n_vocab = 0;
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

// --- encode / decode -------------------------------------------------------

int hftok_encode(const HFTok *t, const char *text, int *ids, int max_ids) {
  int len = (int)strlen(text);
  int n = 0;
  unsigned char *buf = (unsigned char*)malloc((size_t)len + 2);   // concat scratch
  int *pieces = (int*)malloc(((size_t)len + 1) * sizeof(int));
  int pos = 0, cs, cl;
  while (n < max_ids && next_chunk(text, len, &pos, &cs, &cl)) {
    int npc = 0;
    for (int i = 0; i < cl; i++) {
      int id = t->byte2id[(unsigned char)text[cs + i]];
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
  free(buf); free(pieces);
  return n;
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
                    "       transformer hftok <file.tok> decode <id> <id> ...\n");
    return 2;
  }
  HFTok t;
  if (hftok_load(&t, argv[2]) != 0) return 1;

  if (strcmp(argv[3], "encode") == 0) {
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
    fprintf(stderr, "unknown op '%s' (encode|decode)\n", argv[3]);
    hftok_free(&t); return 2;
  }
  hftok_free(&t);
  return 0;
}
