// hftok.h - Pure-C byte-level BPE tokenizer (loads a model's exported vocab), so
// the engine tokenizes GPT-2 / LLaMA-family models ITSELF with no Python at
// runtime. The vocab is written once by tools/export_tokenizer.py into a .tok
// file (id -> bytes); encoding is the tiktoken/byte-level-BPE rule "merge the
// adjacent pair whose combined bytes have the lowest id".
#ifndef HFTOK_H
#define HFTOK_H

typedef struct {
  int n_vocab;
  int eos;                     // end-of-text token id (-1 if none)
  int *tok_len;                // bytes-length of each token
  unsigned char **tok_bytes;   // raw bytes of each token
  int *hmap;                   // open-addressing hash: bucket -> id (or -1)
  int hcap;                    // hash capacity (power of two)
  int byte2id[256];            // id of each single-byte token (-1 if absent)
} HFTok;

// Load a .tok file. Returns 0 on success.
int hftok_load(HFTok *t, const char *path);
void hftok_free(HFTok *t);

// Encode UTF-8 text into token ids (writes up to max_ids). Returns the count.
int hftok_encode(const HFTok *t, const char *text, int *ids, int max_ids);

// Decode ids back into bytes (NUL-terminated). Returns the byte length.
int hftok_decode(const HFTok *t, const int *ids, int n, char *out, int max_out);

// CLI: transformer hftok <file.tok> encode "<text>"
//      transformer hftok <file.tok> decode <id> <id> ...
int run_hftok(int argc, char **argv);

#endif // HFTOK_H
