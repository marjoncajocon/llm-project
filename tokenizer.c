// tokenizer.c - Self-contained byte-level BPE tokenizer
#include "tokenizer.h"

#include <stdio.h>
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

  // Standard BPE: each round, find the lowest-rank applicable pair, then merge
  // ALL of its occurrences in a single pass. Same result as one-at-a-time, but
  // O(rounds) instead of O(merge-operations) passes — essential for large text.
  while (n >= 2) {
    int best_rank = -1;
    for (int i = 0; i < n - 1; i++) {
      int r = merge_rank(t, ids[i], ids[i + 1]);
      if (r != -1 && (best_rank == -1 || r < best_rank)) best_rank = r;
    }
    if (best_rank == -1) break;  // no applicable merges remain

    const int a = t->merge_a[best_rank];
    const int b = t->merge_b[best_rank];
    const int new_id = 256 + best_rank;
    int w = 0;
    for (int i = 0; i < n; ) {
      if (i < n - 1 && ids[i] == a && ids[i + 1] == b) {
        ids[w++] = new_id;
        i += 2;
      } else {
        ids[w++] = ids[i];
        i += 1;
      }
    }
    n = w;
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

// Learn up to num_merges BPE merges: greedily merge the most frequent adjacent pair.
void tokenizer_train(Tokenizer *t, const char *text, int num_merges) {
  int len = (int)strlen(text);
  if (len < 2 || num_merges <= 0) return;

  int *seq = (int*)malloc(len * sizeof(int));
  for (int i = 0; i < len; i++) seq[i] = (unsigned char)text[i];

  const int maxV = 256 + num_merges;         // upper bound on token ids we create
  long *count = (long*)malloc((size_t)maxV * maxV * sizeof(long));

  for (int m = 0; m < num_merges && len >= 2; m++) {
    memset(count, 0, (size_t)maxV * maxV * sizeof(long));
    for (int i = 0; i < len - 1; i++) {
      int a = seq[i], b = seq[i + 1];
      if (a < maxV && b < maxV) count[(size_t)a * maxV + b]++;
    }
    // Most frequent adjacent pair
    long best = 1;  // require the pair to occur at least twice to be worth merging
    int ba = -1, bb = -1;
    for (int a = 0; a < maxV; a++) {
      for (int b = 0; b < maxV; b++) {
        long c = count[(size_t)a * maxV + b];
        if (c > best) { best = c; ba = a; bb = b; }
      }
    }
    if (ba < 0) break;  // no repeated pair left

    int new_id = tokenizer_add_merge(t, ba, bb);
    // Replace every occurrence of (ba,bb) in the sequence with the merged token
    int w = 0;
    for (int i = 0; i < len; ) {
      if (i < len - 1 && seq[i] == ba && seq[i + 1] == bb) {
        seq[w++] = new_id;
        i += 2;
      } else {
        seq[w++] = seq[i];
        i += 1;
      }
    }
    len = w;
  }

  free(seq);
  free(count);
}

int tokenizer_save(const Tokenizer *t, const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f) return 1;
  const char magic[4] = {'B', 'P', 'E', '1'};
  int ok = 1;
  ok &= (fwrite(magic, 1, 4, f) == 4);
  ok &= (fwrite(&t->n_merges, sizeof(int), 1, f) == 1);
  ok &= (fwrite(t->merge_a, sizeof(int), t->n_merges, f) == (size_t)t->n_merges);
  ok &= (fwrite(t->merge_b, sizeof(int), t->n_merges, f) == (size_t)t->n_merges);
  fclose(f);
  return ok ? 0 : 1;
}

int tokenizer_load(Tokenizer *t, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return 1;
  char magic[4];
  int n_merges = 0;
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BPE1", 4) != 0 ||
      fread(&n_merges, sizeof(int), 1, f) != 1) {
    fclose(f);
    return 1;
  }
  // Reject a corrupt/hostile count before allocating or reading (10M is far above
  // any real BPE vocab; negatives would make malloc/fread wildly misbehave).
  if (n_merges < 0 || n_merges > 10000000) {
    fprintf(stderr, "Error: '%s' has an invalid merge count\n", path);
    fclose(f);
    return 1;
  }
  int *a = (int*)malloc((size_t)n_merges * sizeof(int) + 1);
  int *b = (int*)malloc((size_t)n_merges * sizeof(int) + 1);
  if (!a || !b) { free(a); free(b); fclose(f); return 1; }
  int ok = (fread(a, sizeof(int), n_merges, f) == (size_t)n_merges) &&
           (fread(b, sizeof(int), n_merges, f) == (size_t)n_merges);
  fclose(f);

  *t = tokenizer_create_bytelevel();
  if (ok) {
    for (int i = 0; i < n_merges; i++) {
      // Merge rank i may only reference token ids that already exist (0..256+i-1);
      // an out-of-range id would make tokenizer_add_merge read past its tables.
      if (a[i] < 0 || a[i] >= 256 + i || b[i] < 0 || b[i] >= 256 + i) {
        fprintf(stderr, "Error: '%s' has an out-of-range merge at rank %d\n", path, i);
        ok = 0;
        break;
      }
      tokenizer_add_merge(t, a[i], b[i]);
    }
  }
  free(a);
  free(b);
  if (!ok) { tokenizer_free(t); }
  return ok ? 0 : 1;
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
