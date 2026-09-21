// gen.c - Text generation for HF-vocabulary models (see gen.h)

#include "gen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "checkpoint.h"
#include "model.h"
#include "sampler.h"
#include "tensor.h"

HFGenOpts hfgen_default_opts(void) {
  HFGenOpts o;
  memset(&o, 0, sizeof(o));
  o.n_new = 128;
  o.temp = 0.8f;
  o.top_k = 40;
  o.top_p = 0.95f;
  o.seed = 1234ull;
  o.n_samples = 1;
  o.emit = 1;
  o.con = constrain_default_opts();
  return o;
}

int hfgen_fit_prompt(int np, int max_seq, int n_new, int *first) {
  *first = 0;
  if (np <= 0 || max_seq < 2) return 0;
  int reserve = n_new;
  if (reserve > max_seq / 2) reserve = max_seq / 2;
  if (reserve < 1) reserve = 1;
  if (np + reserve <= max_seq) return np;
  const int keep = max_seq - reserve;
  *first = np - keep;
  return keep;
}

int hfgen_cut_at_tag(char *dec) {
  // Close the block toolcall_parse will take: the first "<tool" or "<done"
  // opened decides which closing tag ends it (a write_file body may mention
  // "</done>"). Nothing opened yet: either closing tag ends it, as in GRPO.
  char *ot = strstr(dec, "<tool"), *od = strstr(dec, "<done");
  char *cut;
  if (ot && (!od || ot < od)) cut = strstr(ot, "</tool>");
  else if (od) cut = strstr(od, "</done>");
  else {
    char *a = strstr(dec, "</tool>"), *b = strstr(dec, "</done>");
    cut = (a && (!b || a < b)) ? a : b;
  }
  if (!cut) return 0;
  cut[7] = '\0';                             // both tags are 7 bytes long
  return 1;
}

static void hf_emit_delta(const char *full, int *prevlen) {
  int fl = (int)strlen(full);
  fputs("{\"d\":\"", stdout);
  for (int i = *prevlen; i < fl; i++) {
    unsigned char c = (unsigned char)full[i];
    if (c == '"') fputs("\\\"", stdout);
    else if (c == '\\') fputs("\\\\", stdout);
    else if (c == '\n') fputs("\\n", stdout);
    else if (c == '\r') fputs("\\r", stdout);
    else if (c == '\t') fputs("\\t", stdout);
    else if (c < 0x20) printf("\\u%04x", c);
    else putchar(c);
  }
  fputs("\"}\n", stdout);
  fflush(stdout);
  *prevlen = fl;
}

// Truncate the decoded text at an end-of-turn marker or a hallucinated next turn.
// Catches: literal end tokens (some models emit <|endoftext|>/<|im_end|> as text);
// and a role label on its own line (User/You/Human/Assistant/AI/System) as a WHOLE
// WORD — followed by ':', whitespace, or end — so "User"/"You" stop it but
// "Users"/"Youtube" in real text don't. Cuts at the EARLIEST match.
static int hf_stop(char *dec) {
  char *cut = NULL;
  static const char *ends[] = {"<|endoftext|>", "<|im_end|>", "<|eot_id|>", NULL};
  for (int i = 0; ends[i]; i++) {
    char *s = strstr(dec, ends[i]);
    if (s && (!cut || s < cut)) cut = s;
  }
  static const char *labels[] = {"User", "You", "Human", "Assistant", "AI", "System", "Q", NULL};
  for (char *p = dec; (p = strchr(p, '\n')) != NULL; p++) {
    char *w = p + 1;
    for (int i = 0; labels[i]; i++) {
      int ll = (int)strlen(labels[i]);
      if (strncmp(w, labels[i], (size_t)ll) == 0) {
        char a = w[ll];
        if (a == '\0' || a == ':' || a == ' ' || a == '\t' || a == '\r' || a == '\n') {
          if (!cut || p < cut) cut = p;
          break;
        }
      }
    }
  }
  if (cut) { *cut = '\0'; return 1; }
  return 0;
}

static void emit_error(const HFGenOpts *o, const char *msg) {
  fprintf(stderr, "hfgen: error: %s\n", msg);
  if (o->emit) { printf("{\"error\":\"%s\"}\n", msg); fflush(stdout); }
}

// ---------------------------------------------------------------------------
// A decoder: one model + one KV cache, with a snapshot of the prefilled state
// so several samples can start from one prefill.
// ---------------------------------------------------------------------------
typedef enum { DEC_GPT, DEC_LLAMA, DEC_QLLAMA } DecKind;

typedef struct {
  DecKind kind;
  const GPT *gpt;
  const LlamaModel *llama;
  const QLlamaModel *qllama;
  KVCache gkv;               // DEC_GPT
  LlamaKV kv, tmpl;          // DEC_LLAMA / DEC_QLLAMA
  int has_tmpl, mark_len;
  int max_seq, vocab;
} Dec;

static void dec_step(Dec *d, int token, Tensor *logits) {
  switch (d->kind) {
    case DEC_GPT:    gpt_decode_step(d->gpt, &d->gkv, token, logits); break;
    case DEC_LLAMA:  llama_decode_step(d->llama, &d->kv, token, logits); break;
    case DEC_QLLAMA: q_llama_decode_step(d->qllama, &d->kv, token, logits); break;
  }
}

// Snapshot the current (prefilled) cache. A GPT cache is linear: decoding past
// position mark_len never writes below it, so rewinding is just resetting len.
// A llama cache may hold sliding-window rings that later positions overwrite,
// so its populated slots are copied into a template sized for just the prompt.
static void dec_mark(Dec *d) {
  if (d->kind == DEC_GPT) { d->mark_len = d->gkv.len; return; }
  const LlamaKV *s = &d->kv;
  LlamaKV t;
  t.n_layers = s->n_layers;
  t.max_seq = s->max_seq;
  t.kv_dim = s->kv_dim;
  t.len = 0;
  t.cap = (int*)malloc((size_t)s->n_layers * sizeof(int));
  t.k = (float**)malloc((size_t)s->n_layers * sizeof(float*));
  t.v = (float**)malloc((size_t)s->n_layers * sizeof(float*));
  for (int l = 0; l < s->n_layers; l++) {
    int c = s->len < s->cap[l] ? s->len : s->cap[l];
    if (c < 1) c = 1;
    t.cap[l] = c;
    t.k[l] = (float*)malloc((size_t)c * s->kv_dim * sizeof(float));
    t.v[l] = (float*)malloc((size_t)c * s->kv_dim * sizeof(float));
  }
  llama_kv_copy(&t, s);
  d->tmpl = t;
  d->has_tmpl = 1;
}

static void dec_rewind(Dec *d) {
  if (d->kind == DEC_GPT) d->gkv.len = d->mark_len;
  else llama_kv_copy(&d->kv, &d->tmpl);
}

// Prefill, then sample opts->n_samples continuations.
static int dec_generate(Dec *d, const HFTok *tok, const int *pids, int np,
                        const HFGenOpts *o) {
  int first = 0;
  const int keep = hfgen_fit_prompt(np, d->max_seq, o->n_new, &first);
  if (keep == 0) {
    emit_error(o, np <= 0 ? "empty prompt: nothing to generate from"
                          : "model context too small to generate");
    return 1;
  }
  if (first > 0)
    fprintf(stderr, "hfgen: warning: prompt is %d tokens but the context is %d; dropped the "
            "oldest %d tokens (kept the last %d, room for %d new)\n",
            np, d->max_seq, first, keep, d->max_seq - keep);

  // Constrained decoding (opt-in): one span index for the prompt, one vocab view.
  Constrain *con = NULL;
  ConVocab cv;
  memset(&cv, 0, sizeof(cv));
  if (o->constrain) {
    if (!tok) {
      emit_error(o, "constrained decoding needs the model's tokenizer");
      return 1;
    }
    char *ptxt = NULL;
    if (!o->con_prompt) {
      int cap_txt = 16;
      for (int i = 0; i < np; i++)
        if (pids[i] >= 0 && pids[i] < tok->n_vocab) cap_txt += tok->tok_len[pids[i]];
      ptxt = (char*)malloc((size_t)cap_txt);
      if (ptxt) hftok_decode(tok, pids, np, ptxt, cap_txt);
    }
    con = constrain_create(o->con_prompt ? o->con_prompt : (ptxt ? ptxt : ""), &o->con);
    free(ptxt);
    if (!con || convocab_from_hftok(&cv, tok) != 0) {
      constrain_free(con);
      convocab_free(&cv);
      emit_error(o, "out of memory building the decode constraint");
      return 1;
    }
    if (cv.V > d->vocab) cv.V = d->vocab;     // only ids the model has logits for
  }

  const int N = o->n_samples > 1 ? o->n_samples : 1;
  int rc = 0;
  int lz[2] = {1, d->vocab};
  Tensor logits = tensor_create(lz, 2), plogits = tensor_create(lz, 2);
  const int cap = o->n_new > 0 ? o->n_new : 0;
  int *gen = (int*)malloc((size_t)(cap + 1) * sizeof(int));
  char *dec = tok ? (char*)malloc(1 << 20) : NULL;

  for (int i = 0; i < keep; i++) dec_step(d, pids[first + i], &logits);
  if (N > 1) {
    dec_mark(d);
    memcpy(plogits.data, logits.data, logits.size * sizeof(float));
  }

  for (int smp = 0; smp < N; smp++) {
    if (smp > 0) {
      dec_rewind(d);
      memcpy(logits.data, plogits.data, logits.size * sizeof(float));
    }
    int pos = keep, gn = 0, prevlen = 0;
    Rng rng = rng_create(o->seed + (unsigned long long)smp);
    if (o->emit && N > 1) { printf("{\"sample\":%d,\"seed\":%llu}\n", smp, o->seed + (unsigned long long)smp); fflush(stdout); }
    ConState cst;
    if (con) constrain_init(con, &cst);
    for (int s = 0; s < o->n_new && pos < d->max_seq; s++) {
      int nx;
      if (con) {
        const int left_new = o->n_new - s, left_ctx = d->max_seq - pos;
        nx = constrain_pick(con, &cv, &cst, logits.data, o->temp, o->top_k, o->top_p, &rng,
                            left_new < left_ctx ? left_new : left_ctx, NULL);
        if (nx < 0) {
          emit_error(o, "constrained decode: no token of the vocabulary continues a valid action");
          rc = 1;
          break;
        }
      } else {
        nx = sample_token(&logits, 0, o->temp, o->top_k, o->top_p, &rng);
      }
      if (tok && hftok_is_eog(tok, nx)) break;
      gen[gn++] = nx;
      int stop = 0;
      if (tok) {
        hftok_decode(tok, gen, gn, dec, 1 << 20);
        if (!con) stop = hf_stop(dec);
        if (o->stop_tags && hfgen_cut_at_tag(dec)) stop = 1;
        if (con && constrain_complete(&cst)) stop = 1;
        if (o->emit) hf_emit_delta(dec, &prevlen);
      }
      if (stop) break;                         // no need to decode a token we stop at
      dec_step(d, nx, &logits);
      pos++;
    }
    if (o->out_ids) memcpy(o->out_ids + (size_t)smp * cap, gen, (size_t)gn * sizeof(int));
    if (o->out_len) o->out_len[smp] = gn;
    if (o->emit && N > 1) { printf("{\"end\":%d}\n", smp); fflush(stdout); }
  }

  if (d->has_tmpl) { llama_kv_free(&d->tmpl); d->has_tmpl = 0; }
  tensor_free(&logits); tensor_free(&plogits);
  free(gen); free(dec);
  constrain_free(con);
  convocab_free(&cv);
  return rc;
}

int hfgen_llama(const LlamaModel *m, const HFTok *tok, const int *pids, int np,
                const HFGenOpts *opts) {
  Dec d;
  memset(&d, 0, sizeof(d));
  d.kind = DEC_LLAMA;
  d.llama = m;
  d.kv = llama_kv_create(m);
  d.max_seq = m->max_seq;
  d.vocab = m->vocab;
  int rc = dec_generate(&d, tok, pids, np, opts);
  llama_kv_free(&d.kv);
  return rc;
}

int hfgen_file(const HFTok *tok, const char *mpath, const int *pids, int np,
               const HFGenOpts *o) {
  unsigned int magic = 0;
  FILE *pf = fopen(mpath, "rb");
  if (!pf) { fprintf(stderr, "cannot open '%s'\n", mpath); return 1; }
  if (fread(&magic, 4, 1, pf) != 1) magic = 0;
  fclose(pf);

  int rc = 0;
  Dec d;
  memset(&d, 0, sizeof(d));
  if (magic == 0x43545047u) {                 // 'GPTC' — GPT-2 family
    GPT m;
    if (gpt_load(mpath, &m) != 0) { emit_error(o, "cannot load model"); rc = 1; goto done; }
    d.kind = DEC_GPT; d.gpt = &m;
    d.gkv = gpt_kvcache_create(&m);
    d.max_seq = m.max_seq_len; d.vocab = m.vocab_size;
    rc = dec_generate(&d, tok, pids, np, o);
    gpt_kvcache_free(&d.gkv); gpt_free(&m);
  } else if (magic == 0x514D4C4Cu) {          // 'LLMQ' — int8 llama
    QLlamaModel m;
    if (q_llama_model_load(mpath, &m) != 0) { emit_error(o, "cannot load model"); rc = 1; goto done; }
    d.kind = DEC_QLLAMA; d.qllama = &m;
    d.kv = q_llama_kv_create(&m);
    d.max_seq = m.max_seq; d.vocab = m.vocab;
    rc = dec_generate(&d, tok, pids, np, o);
    llama_kv_free(&d.kv); q_llama_model_free(&m);
  } else if (magic == 0x414D4C4Cu) {          // 'LLMA' — f32 llama
    LlamaModel m;
    if (llama_model_load(mpath, &m) != 0) { emit_error(o, "cannot load model"); rc = 1; goto done; }
    rc = hfgen_llama(&m, tok, pids, np, o);
    llama_model_free(&m);
  } else {
    fprintf(stderr, "error: '%s' is not a GPTC/LLMA/LLMQ checkpoint\n", mpath);
    if (o->emit) { printf("{\"error\":\"not a GPTC/LLMA/LLMQ checkpoint\"}\n"); }
    rc = 1;
  }

done:
  if (o->emit) { printf("{\"done\":true}\n"); fflush(stdout); }
  return rc;
}
