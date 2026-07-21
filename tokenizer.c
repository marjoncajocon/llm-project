// tokenizer.c - Self-contained byte-level BPE tokenizer
#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>

// Store the byte string for a token id (copies len bytes).
static void set_token_bytes(Tokenizer *t, int id, const unsigned char *bytes, int len) {
  t->token_bytes[id] = (unsigned char*)malloc(len > 0 ? len : 1);
  memcpy(t->token_bytes[id], bytes, len);
  t->token_len[id] = len;
}

Tokenizer tokenizer_create_bytelevel(void) {
  Tokenizer t;
  t.cap_tokens = 256;
  t.token_bytes = (unsigned char**)malloc(t.cap_tokens * sizeof(unsigned char*));
  t.token_len = (int*)malloc(t.cap_tokens * sizeof(int));

  // 256 base tokens, one per byte value
  for (int i = 0; i < 256; i++) {
    unsigned char b = (unsigned char)i;
    set_token_bytes(&t, i, &b, 1);
  }
  t.vocab_size = 256;

  t.cap_merges = 16;
  t.merge_a = (int*)malloc(t.cap_merges * sizeof(int));
  t.merge_b = (int*)malloc(t.cap_merges * sizeof(int));
  t.n_merges = 0;

  return t;
}

int tokenizer_add_merge(Tokenizer *t, int a, int b) {
  // Grow the merge arrays if needed
  if (t->n_merges == t->cap_merges) {
    t->cap_merges *= 2;
    t->merge_a = (int*)realloc(t->merge_a, t->cap_merges * sizeof(int));
    t->merge_b = (int*)realloc(t->merge_b, t->cap_merges * sizeof(int));
  }
  // Grow the token tables if needed
  if (t->vocab_size == t->cap_tokens) {
    t->cap_tokens *= 2;
    t->token_bytes = (unsigned char**)realloc(t->token_bytes,
                                              t->cap_tokens * sizeof(unsigned char*));
    t->token_len = (int*)realloc(t->token_len, t->cap_tokens * sizeof(int));
  }

  int rank = t->n_merges;
  t->merge_a[rank] = a;
  t->merge_b[rank] = b;
  t->n_merges++;

  // New token's bytes = bytes(a) followed by bytes(b)
  int new_id = t->vocab_size;
  int len = t->token_len[a] + t->token_len[b];
  unsigned char *buf = (unsigned char*)malloc(len > 0 ? len : 1);
  memcpy(buf, t->token_bytes[a], t->token_len[a]);
  memcpy(buf + t->token_len[a], t->token_bytes[b], t->token_len[b]);
  set_token_bytes(t, new_id, buf, len);
  free(buf);

  t->vocab_size++;
  return new_id;
}

// Return the merge rank for adjacent tokens (a, b), or -1 if they never merge.
static int merge_rank(const Tokenizer *t, int a, int b) {
  for (int r = 0; r < t->n_merges; r++) {
    if (t->merge_a[r] == a && t->merge_b[r] == b) {
      return r;
    }
  }
  return -1;
}

int tokenizer_encode(const Tokenizer *t, const char *text, int *ids, int max_ids) {
  int n = (int)strlen(text);
  if (n > max_ids) n = max_ids;

  // Start from the raw byte tokens
  for (int i = 0; i < n; i++) {
    ids[i] = (unsigned char)text[i];
  }

  // Repeatedly merge the adjacent pair with the lowest (best) rank
  while (n >= 2) {
    int best_rank = -1;
    int best_pos = -1;
    for (int i = 0; i < n - 1; i++) {
      int r = merge_rank(t, ids[i], ids[i + 1]);
      if (r != -1 && (best_rank == -1 || r < best_rank)) {
        best_rank = r;
        best_pos = i;
      }
    }
    if (best_pos == -1) break;  // no more applicable merges

    // Merge the pair at best_pos into its combined token, shift the tail left
    ids[best_pos] = 256 + best_rank;
    for (int i = best_pos + 1; i < n - 1; i++) {
      ids[i] = ids[i + 1];
    }
    n--;
  }

  return n;
}

int tokenizer_decode(const Tokenizer *t, const int *ids, int n, char *out, int max_out) {
  int pos = 0;
  for (int i = 0; i < n; i++) {
    int id = ids[i];
    int len = t->token_len[id];
    for (int j = 0; j < len; j++) {
      if (pos >= max_out - 1) {  // leave room for the NUL terminator
        out[pos] = '\0';
        return pos;
      }
      out[pos++] = (char)t->token_bytes[id][j];
    }
  }
  out[pos] = '\0';
  return pos;
}

void tokenizer_free(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->token_bytes[i]);
  }
  free(t->token_bytes);
  free(t->token_len);
  free(t->merge_a);
  free(t->merge_b);
  t->token_bytes = NULL;
  t->token_len = NULL;
  t->merge_a = NULL;
  t->merge_b = NULL;
  t->vocab_size = 0;
  t->n_merges = 0;
}
