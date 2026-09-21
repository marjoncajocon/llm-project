// constrain.c - Opt-in constrained decoding of the agent's action (see constrain.h)
#include "constrain.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define INF_LEFT (1 << 28)

enum { CS_LIT = 0, CS_THINK, CS_PATH, CS_BODY, CS_CMD, CS_DONE, CS_ACCEPT };
enum { NX_READ = 0, NX_WRITE = 1 };
enum { SUB_VSTART = 0, SUB_VERB, SUB_GAP, SUB_FREE, SUB_ARG };

// ---------------------------------------------------------------------------
// Literal sets: alternatives matched byte by byte (none is a prefix of another).
// ---------------------------------------------------------------------------
typedef struct { const char *str; int target; int next; } Alt;

static const Alt SET_START_ALTS[] = {
  {"<think>", CS_THINK, 0},
  {"<tool name=\"read_file\" path=\"", CS_PATH, NX_READ},
  {"<tool name=\"write_file\" path=\"", CS_PATH, NX_WRITE},
  {"<tool name=\"run_command\" command=\"", CS_CMD, 0},
  {"<done>", CS_DONE, 0},
};
static const Alt SET_RCLOSE_ALTS[] = {{"></tool>", CS_ACCEPT, 0}};
static const Alt SET_WOPEN_ALTS[] = {{">", CS_BODY, 0}};
static const Alt SET_BCLOSE_ALTS[] = {{"/tool>", CS_ACCEPT, 0}};

enum { SET_START = 0, SET_POST, SET_RCLOSE, SET_WOPEN, SET_BCLOSE };
// SET_POST = SET_START without <think> (bit 0 masked off).
static const Alt *set_alts(int set, int *n) {
  switch (set) {
    case SET_START: case SET_POST: *n = 5; return SET_START_ALTS;
    case SET_RCLOSE: *n = 1; return SET_RCLOSE_ALTS;
    case SET_WOPEN:  *n = 1; return SET_WOPEN_ALTS;
    default:         *n = 1; return SET_BCLOSE_ALTS;
  }
}

static const char T_THINK[] = "</think>";   // 8
static const char T_TOOL[] = "</tool>";     // 7
static const char T_DONE[] = "</done>";     // 7
// Tags a free text may not contain: each would change how the reply is parsed
// (toolcall_parse / reward.c format_ok and strip_reasoning / the stop at a closing
// tag). "</tool>" in a body and "</done>" in a done text are the terminators.
#define N_FORBID 4
static const char *const FORBID_BODY[N_FORBID] = {"<tool", "</done>", "<think>", "</think>"};
static const char *const FORBID_DONE[N_FORBID] = {"<done", "</tool>", "<think>", "</think>"};

// ---------------------------------------------------------------------------
// Prompt span index: the starts of candidate spans, sorted by span text.
// ---------------------------------------------------------------------------
typedef struct { int s, e; const unsigned char *p; } Span;
typedef struct { Span *a; int n; int min0; } SpanIx;

struct Constrain {
  ConstrainOpts o;
  unsigned char *p;
  int n;
  SpanIx path, body;
};

static int is_word(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static int is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

static int path_ch(unsigned char c) {
  if (c <= 0x20 || c == 0x7f) return 0;
  return !strchr("\"'<>|`,;&()", c);
}
static int body_ch(unsigned char c) {
  if (c == '\t') return 1;
  if (c < 0x20 || c == 0x7f) return 0;
  return c != '<' && c != '|';
}

static int span_cmp(const void *x, const void *y) {
  const Span *a = (const Span*)x, *b = (const Span*)y;
  const int la = a->e - a->s, lb = b->e - b->s, m = la < lb ? la : lb;
  const int r = memcmp(a->p + a->s, b->p + b->s, (size_t)m);
  if (r) return r;
  if (la != lb) return la - lb;
  return a->s - b->s;                           // total order
}

// Whole-token end: the span may stop after L bytes.
static int close_at(const Span *sp, int L) {
  return L >= 1 && (sp->s + L == sp->e || !is_word(sp->p[sp->s + L]));
}

// Smallest L' >= max(L,1) at which the span may close (always <= its length).
static int span_min_close(const Span *sp, int L) {
  int x = L < 1 ? 1 : L;
  const int len = sp->e - sp->s;
  while (x < len && !close_at(sp, x)) x++;
  return x;
}

static void ix_build(SpanIx *ix, const unsigned char *p, int n, int body) {
  ix->a = (Span*)malloc((size_t)(n > 0 ? n : 1) * sizeof(Span));
  ix->n = 0;
  ix->min0 = INF_LEFT;
  if (!ix->a) return;
  for (int s = 0; s < n; s++) {
    const unsigned char c = p[s];
    if (body ? (!body_ch(c) || c == ' ' || c == '\t') : !path_ch(c)) continue;
    if (s > 0 && (is_word(p[s - 1]) || p[s - 1] == '<')) continue;   // not "/tool>" of a tag
    int e = s;
    while (e < n && (body ? body_ch(p[e]) : path_ch(p[e]))) e++;
    Span sp = {s, e, p};
    ix->a[ix->n++] = sp;
    const int mc = span_min_close(&sp, 0);
    if (mc < ix->min0) ix->min0 = mc;
  }
  qsort(ix->a, (size_t)ix->n, sizeof(Span), span_cmp);
}

// Within [lo,hi) (all spans share the first L bytes, all at least L long) keep
// those whose byte L is b. Order inside the range: length-L spans first, then by
// byte L -- a monotone key, so two binary searches.
static int rank_at(const Span *sp, int L) {
  return (sp->e - sp->s == L) ? -1 : (int)sp->p[sp->s + L];
}
static int ix_narrow(const SpanIx *ix, int *lo, int *hi, int L, unsigned char b) {
  int a = *lo, z = *hi;
  while (a < z) { int m = (a + z) / 2; if (rank_at(&ix->a[m], L) < (int)b) a = m + 1; else z = m; }
  int a2 = a, z2 = *hi;
  while (a2 < z2) { int m = (a2 + z2) / 2; if (rank_at(&ix->a[m], L) <= (int)b) a2 = m + 1; else z2 = m; }
  *lo = a; *hi = a2;
  return a < a2;
}
static int ix_closable(const SpanIx *ix, int lo, int hi, int L) {
  for (int i = lo; i < hi; i++) if (close_at(&ix->a[i], L)) return 1;
  return 0;
}
static int ix_min_left(const SpanIx *ix, int lo, int hi, int L) {
  int best = INF_LEFT;
  for (int i = lo; i < hi && best > 0; i++) {
    const int x = span_min_close(&ix->a[i], L) - L;
    if (x < best) best = x;
  }
  return best;
}

// ---------------------------------------------------------------------------
ConstrainOpts constrain_default_opts(void) {
  ConstrainOpts o;
  o.ground_path = 1;
  o.ground_body = 0;
  o.ground_cmd = 0;
  o.think_max = 256;
  o.window = 64;
  return o;
}

Constrain *constrain_create(const char *prompt, const ConstrainOpts *o) {
  Constrain *c = (Constrain*)calloc(1, sizeof(Constrain));
  if (!c) return NULL;
  c->o = o ? *o : constrain_default_opts();
  if (c->o.window < 1) c->o.window = 1;
  const size_t n = prompt ? strlen(prompt) : 0;
  c->p = (unsigned char*)malloc(n + 1);
  if (!c->p) { free(c); return NULL; }
  if (n) memcpy(c->p, prompt, n);
  c->p[n] = 0;
  c->n = (int)n;
  ix_build(&c->path, c->p, c->n, 0);
  ix_build(&c->body, c->p, c->n, 1);
  if (!c->path.a || !c->body.a) { constrain_free(c); return NULL; }
  return c;
}

void constrain_free(Constrain *c) {
  if (!c) return;
  free(c->path.a); free(c->body.a); free(c->p);
  free(c);
}

// ---------------------------------------------------------------------------
// The state machine
// ---------------------------------------------------------------------------
static unsigned set_initial_alive(const Constrain *c, int set) {
  if (set == SET_START || set == SET_POST) {
    unsigned m = (set == SET_START) ? 0x1Fu : 0x1Eu;
    if (c->o.ground_path && c->path.n == 0) m &= ~0x6u;   // no span: no read / write
    if (c->o.ground_body && c->body.n == 0) m &= ~0x4u;   // no span: no write
    return m;
  }
  return 1u;
}

static void enter_set(const Constrain *c, ConState *s, int set) {
  s->st = CS_LIT; s->set = set; s->li = 0; s->alive = set_initial_alive(c, set);
}

static void enter(const Constrain *c, ConState *s, int target, int next) {
  s->st = target;
  s->k = s->count = s->nonws = 0;
  memset(s->fm, 0, sizeof(s->fm));
  s->L = 0; s->sub = SUB_VSTART;
  switch (target) {
    case CS_PATH: s->next = next; s->lo = 0; s->hi = c->path.n; break;
    case CS_BODY: s->lo = 0; s->hi = c->body.n; break;
    default: break;
  }
}

static int pat_step(const char *pat, int k, unsigned char b) {
  if ((unsigned char)pat[k] == b) return k + 1;
  return b == '<' ? 1 : 0;           // every pattern has '<' only at index 0
}

// One byte of free text: -1 if it completes a forbidden tag, 1 if it completes
// the terminator, else 0.
static int free_step(ConState *s, const char *term, const char *const *forbid, unsigned char b) {
  s->count++;
  int bad = 0;
  for (int i = 0; i < N_FORBID; i++) {
    s->fm[i] = (unsigned char)pat_step(forbid[i], s->fm[i], b);
    if (forbid[i][s->fm[i]] == '\0') bad = 1;
  }
  s->k = pat_step(term, s->k, b);
  if (bad) return -1;
  return term[s->k] == '\0' ? 1 : 0;
}

// Leave a command: the closing quote.
static int cmd_close(const Constrain *c, ConState *s) {
  if (s->nonws == 0) return 0;
  enter_set(c, s, SET_RCLOSE);
  return 1;
}

static int cmd_byte(const Constrain *c, ConState *s, unsigned char b) {
  if (b == '>' || b == '<' || b == 0x7f || (b < 0x20 && b != '\t')) return 0;
  if (!c->o.ground_cmd) {
    if (b == '"') return cmd_close(c, s);
    if (!is_space(b)) s->nonws++;
    return 1;
  }
  const int sep = (b == '|' || b == ';' || b == '&');
  if (s->sub == SUB_ARG) {
    if (b == ' ' || b == '\t' || b == '"' || sep) {
      if (!ix_closable(&c->path, s->lo, s->hi, s->L)) return 0;
      if (b == '"') return cmd_close(c, s);
      s->sub = sep ? SUB_VSTART : SUB_GAP;
      return 1;
    }
    if (!ix_narrow(&c->path, &s->lo, &s->hi, s->L, b)) return 0;
    s->L++; s->nonws++;
    return 1;
  }
  if (b == '"') return cmd_close(c, s);
  if (sep) { s->sub = SUB_VSTART; s->nonws++; return 1; }
  if (b == ' ' || b == '\t') {
    if (s->sub == SUB_VERB || s->sub == SUB_FREE) s->sub = SUB_GAP;
    return 1;
  }
  switch (s->sub) {
    case SUB_VSTART: s->sub = SUB_VERB; break;
    case SUB_GAP:
      if (b == '-') { s->sub = SUB_FREE; break; }
      s->lo = 0; s->hi = c->path.n; s->L = 0;
      if (!ix_narrow(&c->path, &s->lo, &s->hi, 0, b)) return 0;
      s->L = 1; s->sub = SUB_ARG;
      break;
    default: break;                       // SUB_VERB / SUB_FREE: any byte
  }
  s->nonws++;
  return 1;
}

static int feed_byte(const Constrain *c, ConState *s, unsigned char b) {
  if (b == 0) return 0;
  switch (s->st) {
    case CS_LIT: {
      int na;
      const Alt *alts = set_alts(s->set, &na);
      unsigned alive = 0;
      for (int i = 0; i < na; i++)
        if ((s->alive >> i & 1u) && (unsigned char)alts[i].str[s->li] == b) alive |= 1u << i;
      if (!alive) return 0;
      s->alive = alive;
      s->li++;
      for (int i = 0; i < na; i++)
        if ((alive >> i & 1u) && alts[i].str[s->li] == '\0') {
          if (alts[i].target == CS_ACCEPT) { s->st = CS_ACCEPT; return 1; }
          enter(c, s, alts[i].target, alts[i].next);
          return 1;
        }
      return 1;
    }
    case CS_THINK:
      if (s->k == 0) {
        if (b == '<') { s->k = 1; return 1; }
        if (c->o.think_max > 0 && s->count >= c->o.think_max) return 0;   // must close now
        s->count++;
        return 1;
      }
      if ((unsigned char)T_THINK[s->k] != b) return 0;   // '<' only begins </think>
      if (++s->k == 8) enter_set(c, s, SET_POST);
      return 1;
    case CS_PATH:
      if (!c->o.ground_path) {
        if (b == '"') {
          if (s->L == 0) return 0;
          enter_set(c, s, s->next == NX_READ ? SET_RCLOSE : SET_WOPEN);
          return 1;
        }
        if (!path_ch(b)) return 0;
        s->L++;
        return 1;
      }
      if (b == '"') {
        if (!ix_closable(&c->path, s->lo, s->hi, s->L)) return 0;
        enter_set(c, s, s->next == NX_READ ? SET_RCLOSE : SET_WOPEN);
        return 1;
      }
      if (!ix_narrow(&c->path, &s->lo, &s->hi, s->L, b)) return 0;
      s->L++;
      return 1;
    case CS_BODY:
      if (c->o.ground_body) {
        if (b == '<') {
          if (!ix_closable(&c->body, s->lo, s->hi, s->L)) return 0;
          enter_set(c, s, SET_BCLOSE);
          return 1;
        }
        if (!ix_narrow(&c->body, &s->lo, &s->hi, s->L, b)) return 0;
        s->L++;
        return 1;
      }
    {
      const int r = free_step(s, T_TOOL, FORBID_BODY, b);
      if (!is_space(b)) s->nonws++;
      if (r < 0) return 0;
      if (r > 0) {
        if (s->nonws - 7 <= 0) return 0;      // an empty body is not an action
        s->st = CS_ACCEPT;
      }
      return 1;
    }
    case CS_CMD:
      return cmd_byte(c, s, b);
    case CS_DONE: {
      const int r = free_step(s, T_DONE, FORBID_DONE, b);
      if (r < 0) return 0;
      if (r > 0) s->st = CS_ACCEPT;
      return 1;
    }
    default:
      return 0;                               // CS_ACCEPT: nothing may follow
  }
}

void constrain_init(const Constrain *c, ConState *s) {
  memset(s, 0, sizeof(*s));
  enter_set(c, s, SET_START);
}

int constrain_feed(const Constrain *c, ConState *s, const unsigned char *b, int n) {
  for (int i = 0; i < n; i++) {
    if (!feed_byte(c, s, b[i])) return 0;
    s->nbytes++;
  }
  return 1;
}

int constrain_complete(const ConState *s) { return s->st == CS_ACCEPT; }

int constrain_check(const Constrain *c, const char *text) {
  ConState s;
  constrain_init(c, &s);
  if (!constrain_feed(c, &s, (const unsigned char*)text, (int)strlen(text))) return 0;
  return constrain_complete(&s) ? 2 : 1;
}

// ---------------------------------------------------------------------------
// Shortest completion (bytes)
// ---------------------------------------------------------------------------
static int path_fresh(const Constrain *c) { return c->o.ground_path ? c->path.min0 : 1; }
static int body_fresh(const Constrain *c) { return c->o.ground_body ? c->body.min0 + 7 : 8; }
static int path_tail(const Constrain *c, int next) {        // after the value: quote + rest
  return next == NX_READ ? 1 + 8 : 1 + 1 + body_fresh(c);
}

static int target_left(const Constrain *c, int target, int next) {
  switch (target) {
    case CS_THINK:  return 8 + 13;                          // </think><done></done>
    case CS_PATH:   return path_fresh(c) + path_tail(c, next);
    case CS_CMD:    return 1 + 9;                           // x"></tool>
    case CS_DONE:   return 7;
    case CS_BODY:   return body_fresh(c);
    default:        return 0;
  }
}

int constrain_min_left(const Constrain *c, const ConState *s) {
  switch (s->st) {
    case CS_LIT: {
      int na, best = INF_LEFT;
      const Alt *alts = set_alts(s->set, &na);
      for (int i = 0; i < na; i++) {
        if (!(s->alive >> i & 1u)) continue;
        const int x = (int)strlen(alts[i].str) - s->li + target_left(c, alts[i].target, alts[i].next);
        if (x < best) best = x;
      }
      return best;
    }
    case CS_THINK: return (8 - s->k) + 13;
    case CS_PATH:
      if (!c->o.ground_path) return (s->L > 0 ? 0 : 1) + path_tail(c, s->next);
      return ix_min_left(&c->path, s->lo, s->hi, s->L) + path_tail(c, s->next);
    case CS_BODY:
      if (c->o.ground_body) return ix_min_left(&c->body, s->lo, s->hi, s->L) + 7;
      return (s->nonws - s->k > 0) ? 7 - s->k : 8;
    case CS_CMD:
      if (c->o.ground_cmd && s->sub == SUB_ARG) return ix_min_left(&c->path, s->lo, s->hi, s->L) + 9;
      return (s->nonws > 0 ? 0 : 1) + 9;
    case CS_DONE: return 7 - s->k;
    default: return 0;
  }
}

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------
int convocab_from_hftok(ConVocab *v, const HFTok *t) {
  memset(v, 0, sizeof(*v));
  v->V = t->n_vocab;
  v->len = t->tok_len;
  v->bytes = t->tok_bytes;
  v->ban = (unsigned char*)calloc((size_t)(t->n_vocab > 0 ? t->n_vocab : 1), 1);
  if (!v->ban) return 1;
  for (int id = 0; id < t->n_vocab; id++) {
    if (t->tok_len[id] <= 0 || hftok_is_eog(t, id)) { v->ban[id] = 1; continue; }
    if (memchr(t->tok_bytes[id], 0, (size_t)t->tok_len[id])) v->ban[id] = 1;
  }
  for (int i = 0; i < t->n_special; i++)
    if (t->special_id[i] >= 0 && t->special_id[i] < t->n_vocab) v->ban[t->special_id[i]] = 1;
  return 0;
}

void convocab_free(ConVocab *v) {
  free(v->ban);
  memset(v, 0, sizeof(*v));
}

// ---------------------------------------------------------------------------
// The picker
// ---------------------------------------------------------------------------
static int allowed(const Constrain *c, const ConVocab *v, const ConState *s, int id,
                   int budget, ConState *out) {
  if (v->ban[id]) return 0;
  *out = *s;
  if (!constrain_feed(c, out, v->bytes[id], v->len[id])) return 0;
  if (budget > 0 && constrain_min_left(c, out) > budget - 1) return 0;
  return 1;
}

// (logit desc, id asc): a total order, as in sampler.c.
static int better(const float *l, int a, int b) {
  if (l[a] != l[b]) return l[a] > l[b];
  return a < b;
}

typedef struct { float l; int id; } LI;
static int li_cmp(const void *x, const void *y) {
  const LI *a = (const LI*)x, *b = (const LI*)y;
  if (a->l != b->l) return (a->l < b->l) - (a->l > b->l);
  return (a->id > b->id) - (a->id < b->id);
}

// The top W ids of cand[0..n) by (logit desc, id asc) into out; returns the count.
static int top_w(const float *l, const int *cand, int n, int W, int *out) {
  if (W > n) W = n;
  if (W <= 0) return 0;
  if (W > 256) {
    LI *a = (LI*)malloc((size_t)n * sizeof(LI));
    if (!a) return 0;
    for (int i = 0; i < n; i++) { const int id = cand ? cand[i] : i; a[i].l = l[id]; a[i].id = id; }
    qsort(a, (size_t)n, sizeof(LI), li_cmp);
    for (int i = 0; i < W; i++) out[i] = a[i].id;
    free(a);
    return W;
  }
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int id = cand ? cand[i] : i;
    if (isnan(l[id])) continue;
    if (m == W && !better(l, id, out[m - 1])) continue;
    int q = (m < W) ? m++ : m - 1;
    while (q > 0 && better(l, id, out[q - 1])) { out[q] = out[q - 1]; q--; }
    out[q] = id;
  }
  return m;
}

int constrain_pick(const Constrain *c, const ConVocab *v, ConState *s, const float *logits,
                   float temp, int top_k, float top_p, Rng *rng, int budget, int *full_scan) {
  const int V = v->V;
  if (full_scan) *full_scan = 0;
  if (budget > 0 && constrain_min_left(c, s) > budget) budget = 0;   // cannot fit: grammar only
  ConState tmp;

  if (temp <= 0.0f) {                                   // greedy
    int *idx = (int*)malloc((size_t)c->o.window * sizeof(int));
    if (!idx) return -1;
    const int m = top_w(logits, NULL, V, c->o.window, idx);
    for (int i = 0; i < m; i++)
      if (allowed(c, v, s, idx[i], budget, &tmp)) { const int id = idx[i]; free(idx); *s = tmp; return id; }
    free(idx);
    if (full_scan) *full_scan = 1;
    int best = -1;
    for (int id = 0; id < V; id++) {
      if (isnan(logits[id])) continue;
      if (best >= 0 && !better(logits, id, best)) continue;
      if (allowed(c, v, s, id, budget, &tmp)) best = id;
    }
    if (best < 0) return -1;
    allowed(c, v, s, best, budget, &tmp);
    *s = tmp;
    return best;
  }

  const int W = (top_k > 0 && top_k < V) ? top_k : V;
  int *idx = (int*)malloc((size_t)W * sizeof(int));
  int *ok = (int*)malloc((size_t)W * sizeof(int));
  if (!idx || !ok) { free(idx); free(ok); return -1; }
  int m = top_w(logits, NULL, V, W, idx), na = 0;
  for (int i = 0; i < m; i++) if (allowed(c, v, s, idx[i], budget, &tmp)) ok[na++] = idx[i];
  if (na == 0) {                                        // full scan: the top W allowed
    if (full_scan) *full_scan = 1;
    int *all = (int*)malloc((size_t)V * sizeof(int));
    if (!all) { free(idx); free(ok); return -1; }
    int nall = 0;
    for (int id = 0; id < V; id++)
      if (!isnan(logits[id]) && allowed(c, v, s, id, budget, &tmp)) all[nall++] = id;
    na = top_w(logits, all, nall, W, ok);
    free(all);
  }
  free(idx);
  if (na == 0) { free(ok); return -1; }

  // Renormalized softmax(logit / temp) over the allowed candidates, then top-p.
  const float mx = logits[ok[0]];
  float z = 0.0f;
  float *w = (float*)malloc((size_t)na * sizeof(float));
  if (!w) { free(ok); return -1; }
  for (int i = 0; i < na; i++) { w[i] = expf((logits[ok[i]] - mx) / temp); z += w[i]; }
  int keep = na;
  if (top_p > 0.0f && top_p < 1.0f && z > 0.0f) {
    float cum = 0.0f;
    for (int i = 0; i < na; i++) { cum += w[i]; if (cum / z >= top_p) { keep = i + 1; break; } }
  }
  float zk = 0.0f;
  for (int i = 0; i < keep; i++) zk += w[i];
  const float u = rng_uniform(rng) * zk;
  int chosen = ok[keep - 1];
  float acc = 0.0f;
  for (int i = 0; i < keep; i++) { acc += w[i]; if (u <= acc) { chosen = ok[i]; break; } }
  free(w); free(ok);
  allowed(c, v, s, chosen, budget, &tmp);
  *s = tmp;
  return chosen;
}
