// corpus.h - Read training text from a file or a directory tree
#ifndef CORPUS_H
#define CORPUS_H

// Read training text from `path`:
//   - if `path` is a file, return its contents;
//   - if `path` is a directory, recursively concatenate every *.txt file under it
//     (each separated by a newline).
// Returns a NUL-terminated heap buffer (caller frees), or NULL on error.
// *out_len (if non-NULL) receives the length in bytes.
char *corpus_read(const char *path, long *out_len);

#endif // CORPUS_H
