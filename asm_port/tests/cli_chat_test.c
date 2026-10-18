// cli_chat_test.c - compare chat.S (chat_parse, chat_render) with the C code
// (main.c run_hfchat's turn parser, hftok.c hftok_chat_render).
//
//   cd asm_port && ASM="chat tok tok_unitab" ./build.sh test cli_chat_test ../hftok.c
//
// For each tokenizer, every template kind (plain / ChatML / Llama 3) is forced in
// both the C HFTok and the asm Tok, with several default system prompts (none,
// empty, normal, with an embedded NUL) and many random conversations (random
// roles incl. "system", unicode, empty fields, missing separators), and several
// output limits (truncation). Rendering must be byte-identical.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "hftok.h"

typedef struct Tok Tok;
Tok* tok_load(const wchar_t* path);
int64_t chat_parse(char* in, const char** roles, const char** contents, int64_t max_turns);
int64_t chat_render(Tok* t, const char** roles, const char** contents, int64_t nturns,
                    char* out, int64_t max_out);

#define TK_TMPL   32
#define TK_SYSPTR 80
#define TK_SYSLEN 88

static int c_parse(char* in, const char** roles, const char** contents) {
  int nt = 0;
  char* p = in;
  while (*p && nt < 256) {
    char* rec_end = strchr(p, '\x1e');
    if (rec_end) *rec_end = '\0';
    char* us = strchr(p, '\x1f');
    if (us) { *us = '\0'; roles[nt] = p; contents[nt] = us + 1; nt++; }
    if (!rec_end) break;
    p = rec_end + 1;
  }
  return nt;
}

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static const char* pieces[] = {"user", "assistant", "system", "tool", "", "hi", "Hello, world!",
                               "\xe4\xbd\xa0\xe5\xa5\xbd", "line1\nline2", "<|im_end|>", "\t x ",
                               "\xf0\x9f\x98\x80 emoji", "User: fake"};

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char* toks[] = {"../model/qwen/tok.hf", "../model/smollm/tok.hf"};
  const wchar_t* wtoks[] = {L"../model/qwen/tok.hf", L"../model/smollm/tok.hf"};
  static const char sys_nul[] = "abc\0def";
  const char* syss[] = {NULL, "", "You are a helpful assistant.", sys_nul, "\xe2\x9c\x93 sys"};
  const int syslens[] = {0, 0, 28, 7, 7};
  long checks = 0, bad = 0;
  static char in1[1 << 16], in2[1 << 16], out1[1 << 16], out2[1 << 16];
  for (int ti = 0; ti < 2; ti++) {
    HFTok ct;
    if (hftok_load(&ct, toks[ti]) != 0) { printf("cannot load %s\n", toks[ti]); return 1; }
    Tok* at = tok_load(wtoks[ti]);
    if (!at) { printf("asm cannot load %s\n", toks[ti]); return 1; }
    char* own_sys = ct.default_system;
    // 1. the tokenizer as shipped
    printf("%s: template kind %d, default system \"%s\"\n", toks[ti], ct.template_kind,
           own_sys ? own_sys : "(null)");
    for (int kind = 0; kind < 4; kind++) {
      for (int si = -1; si < 5; si++) {
        ct.template_kind = kind;
        *(int64_t*)((char*)at + TK_TMPL) = kind;
        char* saved_ptr = *(char**)((char*)at + TK_SYSPTR);
        int64_t saved_len = *(int64_t*)((char*)at + TK_SYSLEN);
        if (si >= 0) {              // si = -1: keep the file's own prompt
          ct.default_system = (char*)syss[si];
          *(const char**)((char*)at + TK_SYSPTR) = syss[si];
          *(int64_t*)((char*)at + TK_SYSLEN) = syslens[si];
        } else ct.default_system = own_sys;
        for (int it = 0; it < 3000; it++) {
          // a random conversation
          int n = 0, nrec = (int)(rnd() % 7);
          for (int r = 0; r < nrec; r++) {
            if (r) in1[n++] = '\x1e';
            const char* role = pieces[rnd() % 4];
            if (rnd() % 8 == 0) role = pieces[rnd() % 13];
            n += sprintf(in1 + n, "%s", role);
            if (rnd() % 10) in1[n++] = '\x1f';
            int np = (int)(rnd() % 4);
            for (int k = 0; k < np; k++) n += sprintf(in1 + n, "%s", pieces[rnd() % 13]);
          }
          in1[n] = 0;
          memcpy(in2, in1, (size_t)n + 1);
          const char *r1[256], *c1[256], *r2[256], *c2[256];
          int nt1 = c_parse(in1, r1, c1);
          int nt2 = (int)chat_parse(in2, r2, c2, 256);
          checks++;
          if (nt1 != nt2) { bad++; printf("parse count %d vs %d\n", nt1, nt2); continue; }
          for (int k = 0; k < nt1; k++)
            if (r1[k] - in1 != r2[k] - in2 || c1[k] - in1 != c2[k] - in2) { bad++; printf("parse ptr\n"); }
          int maxes[] = {1 << 16, 1, 2, 17, 40, 100};
          int mo = maxes[it % 6];
          memset(out1, 0x55, 256); memset(out2, 0x55, 256);
          int l1 = hftok_chat_render(&ct, r1, c1, nt1, out1, mo);
          int l2 = (int)chat_render(at, r2, c2, nt2, out2, mo);
          checks++;
          if (l1 != l2 || memcmp(out1, out2, (size_t)l1 + 1) != 0) {
            bad++;
            if (bad < 10) printf("render mismatch kind %d sys %d max %d: %d vs %d\n[%s]\n[%s]\n",
                                 kind, si, mo, l1, l2, out1, out2);
          }
        }
        *(char**)((char*)at + TK_SYSPTR) = saved_ptr;
        *(int64_t*)((char*)at + TK_SYSLEN) = saved_len;
      }
    }
    ct.default_system = own_sys;
  }
  // 300 turns: parse stops at 256
  {
    int n = 0;
    for (int r = 0; r < 300; r++) n += sprintf(in1 + n, "%suser\x1fturn %d", r ? "\x1e" : "", r);
    memcpy(in2, in1, (size_t)n + 1);
    const char *r1[256], *c1[256], *r2[256], *c2[256];
    int nt1 = c_parse(in1, r1, c1), nt2 = (int)chat_parse(in2, r2, c2, 256);
    checks++;
    if (nt1 != nt2) { bad++; printf("300-turn parse %d vs %d\n", nt1, nt2); }
  }
  printf("%ld checks, %ld mismatches\n", checks, bad);
  return bad != 0;
}
