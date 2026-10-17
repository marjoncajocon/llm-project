// tok_test.c - compares the assembly tokenizer (tok.S) against the C reference
// (../hftok.c) id for id. Build + run:  ./build.sh test tok_test ../hftok.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include "hftok.h"
#include "unitab.h"

typedef struct Tok Tok;
Tok *tok_load(const wchar_t *path);
long long tok_encode(Tok *t, const unsigned char *text, long long len, int *ids, long long cap);
long long tok_piece(Tok *t, long long id, unsigned char *out);
long long tok_is_eog(Tok *t, long long id);
long long tok_vocab(Tok *t);
long long tok_max_piece(Tok *t);
void tok_free(Tok *t);

static long long n_cmp = 0, n_bad = 0, n_ids = 0;

static int *ca, *cb;
static size_t cab_cap = 0;
static void ensure(size_t n) {
  if (n <= cab_cap) return;
  cab_cap = n * 2;
  ca = (int*)realloc(ca, cab_cap * sizeof(int));
  cb = (int*)realloc(cb, cab_cap * sizeof(int));
}

// compare one text (must contain no NUL: the C API uses strlen) at a given cap
static void cmp_cap(HFTok *h, Tok *T, const char *s, int len, int cap, const char *what) {
  ensure((size_t)len + 32);
  int nc = hftok_encode(h, s, ca, cap);
  long long na = tok_encode(T, (const unsigned char*)s, len, cb, cap);
  n_cmp++;
  n_ids += nc;
  if (nc != na || memcmp(ca, cb, (size_t)nc * sizeof(int)) != 0) {
    if (n_bad++ < 10) {
      printf("  MISMATCH (%s) len=%d cap=%d: C n=%d asm n=%lld\n   text: \"", what, len, cap, nc, na);
      for (int i = 0; i < len && i < 120; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 32 && c < 127) putchar(c); else printf("\\x%02X", c);
      }
      printf("\"\n   C  :");
      for (int i = 0; i < nc && i < 40; i++) printf(" %d", ca[i]);
      printf("\n   asm:");
      for (int i = 0; i < na && i < 40; i++) printf(" %d", cb[i]);
      printf("\n");
    }
  }
}
static void cmp1(HFTok *h, Tok *T, const char *s, int len, const char *what) {
  cmp_cap(h, T, s, len, len + 16, what);
}

static char *read_file(const char *path, int *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char *b = (char*)malloc((size_t)n + 1);
  n = (long)fread(b, 1, (size_t)n, f);
  fclose(f);
  // the C API is NUL-terminated: replace stray NULs
  for (long i = 0; i < n; i++) if (!b[i]) b[i] = ' ';
  b[n] = 0; *len = (int)n;
  return b;
}

// list of corpus files
static char **files; static int nfiles;
static void add_file(const char *p) { files = (char**)realloc(files, (nfiles + 1) * sizeof(char*)); files[nfiles++] = _strdup(p); }
static void add_dir(const char *dir) {
  char pat[512]; snprintf(pat, sizeof pat, "%s/*.txt", dir);
  WIN32_FIND_DATAA fd; HANDLE hf = FindFirstFileA(pat, &fd);
  if (hf == INVALID_HANDLE_VALUE) return;
  do { char p[512]; snprintf(p, sizeof p, "%s/%s", dir, fd.cFileName); add_file(p); } while (FindNextFileA(hf, &fd));
  FindClose(hf);
}

static const char *hand[] = {
  "", " ", "  ", "a", "<", "<|", "<|im_start|>", "<|im_start|>user\nhi<|im_end|>",
  "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n",
  "<|endoftext|><|endoftext|>x<|endoftext|", "a<|im_end|>b", "<<|im_end|>>", "<|im_start|><|im_end|>",
  "Hello world", "Hello, world! How's it going? I'm fine, you'll see; they've, we're, he'd.",
  "IT'S THE CAT'S HAT. I'M SURE YOU'LL SEE. WE'RE HERE. THEY'VE GONE. HE'D GO.",
  "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D 'Re 'rE 'lL 'x '", "\xC5\xBF \x27\xC5\xBF don\x27\xC5\xBF",
  "12345678901234567890", "x 12345", "x  1", "3.14159 2,718 1e10 0x1F 100% $5",
  "\xD9\xA1\xD9\xA2\xD9\xA3 \xC2\xBD \xE2\x85\xAB \xE2\x91\xA0", "\r\n\r\n\r\n", "a\r\nb\r\n\r\nc", "\n\n\n", " \n \n ",
  "\t\t\tindent", "    four spaces", "trailing   ", "   ", "a  b   c    d", "x\t \t y", "\x0B\x0C vt ff",
  "\xC2\xA0nbsp\xC2\xA0\xC2\xA0x", "\xE3\x80\x80ideo\xE3\x80\x80", "\xC2\x85nel", "\xE2\x80\xA8ls\xE2\x80\xA9ps",
  "\xE4\xB8\xAD\xE6\x96\x87\xE6\xB5\x8B\xE8\xAF\x95 \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE3\x81\xAE\xE3\x83\x86\xE3\x82\xAD\xE3\x82\xB9\xE3\x83\x88 \xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4",
  "\xF0\x9F\x98\x80\xF0\x9F\x98\x81 emoji \xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD \xF0\x9F\x87\xBA\xF0\x9F\x87\xB8",
  "caf\xC3\xA9 na\xC3\xAFve r\xC3\xA9sum\xC3\xA9 \xC3\x9C" "ber Stra\xC3\x9F" "e \xC3\xA5ngstr\xC3\xB6m",
  "Ελληνικά Русский язык עברית العربية हिन्दी ไทย",
  "e\xCC\x81 combining a\xCC\x8A", "\xC3", "\x80", "\xFF\xFE\xFD", "\xC0\xAF", "\xE0\x80\xAF", "\xED\xA0\x80 surrogate",
  "\xF4\x90\x80\x80 too big", "\xF0\x9F\x98 truncated", "ab\xE4\xB8", "\xE4\xB8x", "\x01\x02\x03\x04\x05", "th\x04" "e",
  "\xF1\xF5\xF8", "(self.foo) [x] {y} <z> a.b,c;d:e!f?g", "...!!!???", "--->>>", "\"quoted\" 'single'",
  " (hello) .world 'tis", "x='s'", "don't won't can't shan't", "\xE2\x80\x99s curly\xE2\x80\x99ll",
  "foo\n\n  bar\r\n\tbaz  \n", "  \n\n  x", "a\n \n b", ")\n\n", ".\r\n\r\n", "! \r\n", "\xF0\x9F\x98\x80hello",
  "\xF0\x9F\x98\x80 hello", "$ ls -la | grep foo > out.txt && echo \"done\"", "<tool_call>{\"name\": \"x\"}</tool_call>",
  "<think>\n</think>", "<|fim_prefix|>", "\xE2\x80\x8B zero width", "\xEF\xBB\xBF" "bom",
};

static const char *frag[] = {
  "a","Z","e","s","t","l","r","v","m","d","x","foo","Hello"," world","\xC3\xA9","\xC3\x9F","\xE4\xB8\xAD","\xE6\x96\x87",
  "\xF0\x9F\x98\x80","\xF0\x9F\x98","\xC3","\x80","\xFF","\xC0\xAF","\xED\xA0\x80","\xF4\x90\x80\x80"," ","  ","\t","\n","\r",
  "\r\n","\xC2\xA0","\xE3\x80\x80","\xC2\x85","0","7","42","\xD9\xA1","\xC2\xBD","\xE2\x85\xAB","'","'s","'S","'ll","'LL",
  "'re","'Ve","'D","\xC5\xBF"," '","<","<|im_start|>","<|im_end|>","<|endoftext|>","<|im_",".",",","!","(",")","\xE2\x80\x99",
  "\x04","\x01","\xF1","\xF5","\x0B","\x0C","e\xCC\x81","\xEA\xB0\x80","-","_","#","\\","\"",
};

static unsigned long long rng = 88172645463325252ull;
static unsigned rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)(rng >> 11); }

static double now(void) {
  LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
}

// copy a .tok file as v3 with the given pretokenizer kind
static int make_v3(const char *src, const char *dst, int kind) {
  FILE *f = fopen(src, "rb"); if (!f) return 1;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  unsigned char *raw = (unsigned char*)malloc((size_t)n + 1);
  n = (long)fread(raw, 1, (size_t)n, f); fclose(f);
  raw[4] = 3; raw[5] = raw[6] = raw[7] = 0;
  f = fopen(dst, "wb"); if (!f) return 1;
  fwrite(raw, 1, (size_t)n, f); fputc(kind, f); fclose(f);
  free(raw);
  return 0;
}

static int uni_in(const uint32_t (*tab)[2], int n, uint32_t cp) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) { int mid = (lo + hi) >> 1;
    if (cp < tab[mid][0]) hi = mid - 1; else if (cp > tab[mid][1]) lo = mid + 1; else return 1; }
  return 0;
}
#define CNT(t) ((int)(sizeof(t) / sizeof((t)[0])))

int main(void) {
  SetConsoleOutputCP(65001);
  struct { const char *name, *path; int kind; } cfg[] = {
    {"qwen (v2 file: kind 0)",      "../model/qwen/tok.hf",   -1},
    {"smollm (v2 file: kind 0)",    "../model/smollm/tok.hf", -1},
    {"gpt2 (v2 file: kind 0)",      "../model/gpt2/tok.hf",   -1},
    {"qwen as v3 kind 1 (Qwen2)",   "../model/qwen/tok.hf",    1},
    {"smollm as v3 kind 2 (Smol)",  "../model/smollm/tok.hf",  2},
    {"qwen as v3 kind 2",           "../model/qwen/tok.hf",    2},
    {"smollm as v3 kind 1",         "../model/smollm/tok.hf",  1},
    {"gpt2 as v3 kind 2",           "../model/gpt2/tok.hf",    2},
  };
  const int ncfg = CNT(cfg);

  add_file("../data/sft_agent.txt"); add_file("../data/qa.txt"); add_file("../data/mixed.txt");
  add_file("../data/traj.txt"); add_file("../data/toolcalls.txt"); add_file("../data/so_commands.txt");
  add_file("../data/clean_reason.txt"); add_file("../data/dpo_pairs.txt");
  add_dir("../data/wiki"); add_dir("../data/stackoverflow");
  printf("corpus: %d files\n", nfiles);

  // big text for throughput: all wiki files concatenated
  size_t big_len = 0, big_cap = 1 << 20; char *big = (char*)malloc(big_cap);
  for (int i = 0; i < nfiles; i++) {
    if (!strstr(files[i], "/wiki/")) continue;
    int n; char *b = read_file(files[i], &n); if (!b) continue;
    while (big_len + n + 2 > big_cap) { big_cap *= 2; big = (char*)realloc(big, big_cap); }
    memcpy(big + big_len, b, (size_t)n); big_len += n; big[big_len++] = '\n'; free(b);
  }
  big[big_len] = 0;

  int fail_any = 0;
  for (int c = 0; c < ncfg; c++) {
    char cpath[512]; wchar_t wpath[512];
    if (cfg[c].kind >= 0) {
      snprintf(cpath, sizeof cpath, "build/tok_v3_%d_%d.hf", c, cfg[c].kind);
      if (make_v3(cfg[c].path, cpath, cfg[c].kind)) { printf("cannot write %s\n", cpath); return 1; }
    } else snprintf(cpath, sizeof cpath, "%s", cfg[c].path);
    for (int i = 0; ; i++) { wpath[i] = (wchar_t)cpath[i]; if (!cpath[i]) break; }

    printf("\n== %s  [%s]\n", cfg[c].name, cpath);
    HFTok h;
    if (hftok_load(&h, cpath) != 0) { printf("  C load failed\n"); fail_any = 1; continue; }
    Tok *T = tok_load(wpath);
    if (!T) { printf("  asm load FAILED\n"); fail_any = 1; continue; }
    long long bad0 = n_bad, cmp0 = n_cmp, ids0 = n_ids;

    // header / vocab / pieces / eog
    if (tok_vocab(T) != h.n_vocab) { printf("  vocab mismatch %lld vs %d\n", tok_vocab(T), h.n_vocab); n_bad++; }
    printf("  vocab %lld, pretok %d, specials %d, longest token %lld bytes\n",
           tok_vocab(T), h.pretok, h.n_special, tok_max_piece(T));
    {
      static char cbuf[70000]; static unsigned char abuf[70000];
      long long pbad = 0, ebad = 0;
      for (int id = -3; id < h.n_vocab + 3; id++) {
        int lc = hftok_decode(&h, &id, 1, cbuf, sizeof cbuf);
        long long la = tok_piece(T, id, abuf);
        if (lc != la || memcmp(cbuf, abuf, (size_t)lc) != 0) pbad++;
        if (hftok_is_eog(&h, id) != tok_is_eog(T, id)) ebad++;
      }
      printf("  tok_piece vs hftok_decode: %d ids, %lld mismatches; tok_is_eog: %lld mismatches\n",
             h.n_vocab + 6, pbad, ebad);
      n_bad += pbad + ebad;
    }
    // Unicode class table (peek at Tok+96) vs unitab.h binary search, once
    if (c == 0) {
      const unsigned char *uc = *(const unsigned char**)((const char*)T + 96);
      long long ubad = 0;
      for (uint32_t cp = 0; cp < 0x110000; cp++) {
        int want = uni_in(UNI_L, CNT(UNI_L), cp) | uni_in(UNI_N, CNT(UNI_N), cp) << 1 | uni_in(UNI_S, CNT(UNI_S), cp) << 2;
        if (uc[cp] != want) ubad++;
      }
      printf("  Unicode class table: 0x110000 code points, %lld mismatches\n", ubad);
      n_bad += ubad;
    }

    const int perf_only = getenv("TOK_PERF") != NULL;   // TOK_PERF=1: skip the comparisons
    if (!perf_only) {
    // hand-written strings, at full cap and several truncated caps
    for (int i = 0; i < CNT(hand); i++) {
      int len = (int)strlen(hand[i]);
      cmp1(&h, T, hand[i], len, "hand");
      for (int cap = 0; cap < 6; cap++) cmp_cap(&h, T, hand[i], len, cap, "hand/cap");
    }
    // random strings from fragments + raw random bytes
    {
      char s[600];
      for (int it = 0; it < 30000; it++) {
        int len = 0, nf = 1 + rnd() % 40;
        for (int k = 0; k < nf && len < 500; k++) {
          if (rnd() % 8 == 0) { unsigned char b = (unsigned char)(1 + rnd() % 255); s[len++] = (char)b; }
          else { const char *f = frag[rnd() % CNT(frag)]; size_t fl = strlen(f); memcpy(s + len, f, fl); len += (int)fl; }
        }
        s[len] = 0;
        cmp1(&h, T, s, len, "fuzz");
        if (it % 10 == 0) cmp_cap(&h, T, s, len, (int)(rnd() % (len + 2)), "fuzz/cap");
      }
    }
    // corpus: every line (with and without its '\n') and every file whole
    long long lines = 0;
    for (int i = 0; i < nfiles; i++) {
      int n; char *b = read_file(files[i], &n); if (!b) continue;
      cmp1(&h, T, b, n, files[i]);
      int st = 0;
      for (int p = 0; p <= n; p++) {
        if (p == n || b[p] == '\n') {
          int e = (p < n) ? p + 1 : p;
          char sv = b[e]; b[e] = 0;
          cmp1(&h, T, b + st, e - st, files[i]);
          if (p < n) { b[p] = 0; cmp1(&h, T, b + st, p - st, files[i]); b[p] = '\n'; }
          b[e] = sv; lines++;
          st = p + 1;
        }
      }
      free(b);
    }
    printf("  encode: %lld texts compared (%lld corpus lines x2 + %d files + %d hand x7 + fuzz), %lld ids, %lld mismatches\n",
           n_cmp - cmp0, lines, nfiles, CNT(hand), n_ids - ids0, n_bad - bad0);

    }
    // throughput on the concatenated wiki text
    {
      ensure(big_len + 32);
      double t0 = now();
      int nc = hftok_encode(&h, big, ca, (int)big_len + 16);
      double tc = now() - t0;
      // a freshly loaded Tok, so the first run starts with an empty word cache
      Tok *T2 = tok_load(wpath);
      double ta = 1e30, tcold = 0; long long na = 0;
      for (int r = 0; r < 3; r++) {
        t0 = now();
        na = tok_encode(T2, (const unsigned char*)big, (long long)big_len, cb, (long long)big_len + 16);
        double d = now() - t0; if (d < ta) ta = d; if (r == 0) tcold = d;
      }
      tok_free(T2);
      int same = (na == nc) && memcmp(ca, cb, (size_t)nc * sizeof(int)) == 0;
      if (!same) n_bad++;
      printf("  throughput on %.2f MB (%d ids, %s): C %.2f MB/s | asm cold %.2f MB/s (%.1fx), warm %.2f MB/s (%.1fx)\n",
             big_len / 1e6, nc, same ? "identical" : "DIFFERENT", big_len / 1e6 / tc,
             big_len / 1e6 / tcold, tc / tcold, big_len / 1e6 / ta, tc / ta);
      if (perf_only) {   // hit-path speed: the first 64 KB of the text repeated (word cache all hits)
        size_t rl = 0; char *rep = (char*)malloc(big_len + 1);
        while (rl + 65536 <= big_len) { memcpy(rep + rl, big, 65536); rl += 65536; }
        rep[rl] = 0;
        double tr = 1e30;
        for (int r = 0; r < 3; r++) { t0 = now(); tok_encode(T, (const unsigned char*)rep, (long long)rl, cb, (long long)rl + 16);
          double d = now() - t0; if (d < tr) tr = d; }
        printf("  repeated 64 KB text (all word-cache hits): asm %.2f MB/s\n", rl / 1e6 / tr);
        free(rep);
      }
    }
    tok_free(T);
    hftok_free(&h);
  }

  // error handling
  {
    Tok *T = tok_load(L"../model/does_not_exist.hf");
    printf("\nmissing file -> %p (expect 0)\n", (void*)T);
    if (T) n_bad++;
    T = tok_load(L"../data/sample.txt");
    printf("non-.tok file -> %p (expect 0)\n", (void*)T);
    if (T) n_bad++;
    // truncated copy
    FILE *f = fopen("../model/gpt2/tok.hf", "rb"); FILE *g = fopen("build/tok_trunc.hf", "wb");
    char buf[4096]; size_t r = fread(buf, 1, sizeof buf, f); fwrite(buf, 1, r, g); fclose(f); fclose(g);
    T = tok_load(L"build/tok_trunc.hf");
    printf("truncated file -> %p (expect 0)\n", (void*)T);
    if (T) n_bad++;
    // v3 with an unknown pretokenizer kind
    make_v3("../model/gpt2/tok.hf", "build/tok_badkind.hf", 3);
    T = tok_load(L"build/tok_badkind.hf");
    printf("v3 kind 3 -> %p (expect 0)\n", (void*)T);
    if (T) n_bad++;
  }

  printf("\nTOTAL: %lld encode comparisons, %lld ids, %lld mismatches -> %s\n",
         n_cmp, n_ids, n_bad, (n_bad == 0 && !fail_any) ? "PASS" : "FAIL");
  return (n_bad == 0 && !fail_any) ? 0 : 1;
}
