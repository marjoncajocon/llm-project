// hftok.h - Pure-C byte-level BPE tokenizer (loads a model's exported vocab), so
// the engine tokenizes GPT-2 / LLaMA-family / Qwen models ITSELF with no Python at
// runtime. The vocab is written once by tools/export_tokenizer.py into a .tok
// file (id -> bytes); encoding is the tiktoken/byte-level-BPE rule "merge the
// adjacent pair whose combined bytes have the lowest id".
//
// v2 also carries a model's SPECIAL tokens (<|im_start|>, <|im_end|>, ...) so
// encode emits them as single ids (not byte-split), the end-of-generation set so
// generation stops on the real token (not a string heuristic), and the chat
// template kind so the engine can format a conversation itself.
#ifndef HFTOK_H
#define HFTOK_H

// Chat template kinds (how a conversation is turned into a prompt).
#define HFTOK_TMPL_PLAIN  0   // "User: ...\nAssistant: ..." fallback
#define HFTOK_TMPL_CHATML 1   // <|im_start|>role\n...<|im_end|> (Qwen, SmolLM2)
#define HFTOK_TMPL_LLAMA3 2   // <|start_header_id|>role<|end_header_id|>...<|eot_id|>

// Pretokenizer kinds (the regex that splits text into words before BPE). Stored
// in .tok v3; v1/v2 files are GPT-2.
#define HFTOK_PRE_GPT2  0   // GPT-2 regex (' ?\p{L}+| ?\p{N}+|...'): GPT-2, SmolLM2
#define HFTOK_PRE_QWEN2 1   // Qwen2/2.5/3 regex: single digits, [^\r\n\p{L}\p{N}]?\p{L}+, ...

typedef struct {
  int n_vocab;
  int eos;                     // end-of-text token id (-1 if none)
  int bos;                     // begin-of-sequence id (-1 if none)
  int *tok_len;                // bytes-length of each token
  unsigned char **tok_bytes;   // raw bytes of each token
  int *hmap;                   // open-addressing hash: bucket -> id (or -1)
  int hcap;                    // hash capacity (power of two)
  int byte2id[256];            // id of each single-byte token (-1 if absent)

  // special/added tokens, matched greedily in encode (all begin with '<').
  int n_special;
  int *special_id;             // vocab id of each special
  int *special_len;            // byte length of its literal string
  unsigned char **special_str; // literal string bytes (e.g. "<|im_start|>")
  int *special_eog;            // 1 if emitting this id ends generation

  int template_kind;           // HFTOK_TMPL_*
  char *default_system;        // model's default system prompt ("" if none)
  int pretok;                  // HFTOK_PRE_* (v3 files; GPT-2 for v1/v2)
} HFTok;

// Load a .tok file (v1, v2 or v3). Returns 0 on success. v3 = v2 + one trailing
// byte, the pretokenizer kind (unknown kinds are rejected).
int hftok_load(HFTok *t, const char *path);
void hftok_free(HFTok *t);

// Encode UTF-8 text into token ids (writes up to max_ids). Special-token strings
// present in the text are emitted as their single id. Returns the count.
int hftok_encode(const HFTok *t, const char *text, int *ids, int max_ids);

// Split text[0,len) into pretokens with the given HFTOK_PRE_* rule (no special-
// token handling): starts[i]/lens[i] are byte offsets/lengths, at most max of them.
// Returns the count. Exposed for tests (selftest [45], `hftok split`).
int hftok_pretok_split(int kind, const char *text, int len, int *starts, int *lens, int max);

// Decode ids back into bytes (NUL-terminated). Returns the byte length.
int hftok_decode(const HFTok *t, const int *ids, int n, char *out, int max_out);

// 1 if id is an end-of-generation token (eos or a special flagged eog).
int hftok_is_eog(const HFTok *t, int id);

// Render a conversation into a prompt STRING (with special markers) using the
// model's template_kind, ending at the point the assistant should continue.
// roles[i]/contents[i] are the nturns turns ("system"/"user"/"assistant"). If no
// system turn is given and default_system is set, it is prepended. Returns length.
int hftok_chat_render(const HFTok *t, const char **roles, const char **contents,
                      int nturns, char *out, int max_out);

// CLI: transformer hftok <file.tok> encode "<text>"
//      transformer hftok <file.tok> decode <id> <id> ...
int run_hftok(int argc, char **argv);

#endif // HFTOK_H
