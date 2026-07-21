// tokenizer.h - Self-contained byte-level BPE tokenizer
#ifndef TOKENIZER_H
#define TOKENIZER_H

// A byte-level BPE tokenizer. The base vocabulary is the 256 single-byte tokens
// (ids 0-255); each merge added creates a new token with id 256 + rank. This is
// binary-safe (any byte round-trips) and needs no external vocabulary files.
// Loading GPT-2's encoder.json / vocab.bpe is a later add-on.
typedef struct {
  int vocab_size;                // 256 + n_merges

  // Decode table: token id -> byte string.
  unsigned char **token_bytes;   // token_bytes[id] = bytes of that token
  int *token_len;                // token_len[id]   = length in bytes
  int cap_tokens;                // allocated capacity of the tables above

  // Merge rules: pair (merge_a[r], merge_b[r]) merges into token 256 + r.
  int *merge_a;
  int *merge_b;
  int n_merges;
  int cap_merges;                // allocated capacity of the merge arrays
} Tokenizer;

// Create a tokenizer with the 256 base byte tokens and no merges.
Tokenizer tokenizer_create_bytelevel(void);

// Add a merge rule combining tokens a and b. Returns the new token's id.
int tokenizer_add_merge(Tokenizer *t, int a, int b);

// Learn up to num_merges BPE merges from a training text: repeatedly merge the
// most frequent adjacent token pair. Call on a fresh byte-level tokenizer.
void tokenizer_train(Tokenizer *t, const char *text, int num_merges);

// Persist / load learned merges (the 256 base byte tokens are implicit).
// tokenizer_load fills *t (like create + the saved merges); free with tokenizer_free.
// Both return 0 on success, nonzero on error.
int tokenizer_save(const Tokenizer *t, const char *path);
int tokenizer_load(Tokenizer *t, const char *path);

// Encode text into token ids (greedy lowest-rank-pair merging over the bytes).
// Writes at most max_ids ids into ids[]; returns the number of tokens produced.
int tokenizer_encode(const Tokenizer *t, const char *text, int *ids, int max_ids);

// Decode token ids back into bytes. Writes a NUL-terminated string into out
// (at most max_out bytes including the terminator); returns bytes written.
int tokenizer_decode(const Tokenizer *t, const int *ids, int n, char *out, int max_out);

// Release all memory owned by the tokenizer.
void tokenizer_free(Tokenizer *t);

#endif // TOKENIZER_H
