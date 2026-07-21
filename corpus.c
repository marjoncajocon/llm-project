// corpus.c - Read training text from a file or a directory tree

// opendir/readdir/stat are POSIX; enable them under strict -std=c11 (non-Windows).
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "corpus.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Platform directory walking.
#if defined(_WIN32)
#include <windows.h>
static int path_is_dir(const char *p) {
  DWORD a = GetFileAttributesA(p);
  return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}
#else
#include <sys/stat.h>
#include <dirent.h>
static int path_is_dir(const char *p) {
  struct stat s;
  return stat(p, &s) == 0 && S_ISDIR(s.st_mode);
}
#endif

// Growable byte buffer.
typedef struct { char *data; long len; long cap; } Buf;

static void buf_ensure(Buf *b, long extra) {
  if (b->len + extra + 1 > b->cap) {
    long nc = (b->cap ? b->cap : 4096);
    while (nc < b->len + extra + 1) nc *= 2;
    b->data = (char*)realloc(b->data, nc);
    b->cap = nc;
  }
}

static int has_txt_ext(const char *name) {
  size_t n = strlen(name);
  return n >= 4 && strcmp(name + n - 4, ".txt") == 0;
}

// Append the contents of one file to the buffer, followed by a newline.
static void append_file(const char *path, Buf *b) {
  FILE *f = fopen(path, "rb");
  if (!f) return;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  buf_ensure(b, len + 1);
  long n = (long)fread(b->data + b->len, 1, len, f);
  b->len += n;
  b->data[b->len++] = '\n';
  b->data[b->len] = '\0';
  fclose(f);
}

// Recursively walk `dir`, appending every *.txt file's contents.
static void walk_dir(const char *dir, Buf *b) {
#if defined(_WIN32)
  char pattern[1024];
  snprintf(pattern, sizeof(pattern), "%s\\*", dir);
  WIN32_FIND_DATAA fd;
  HANDLE h = FindFirstFileA(pattern, &fd);
  if (h == INVALID_HANDLE_VALUE) return;
  do {
    const char *name = fd.cFileName;
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;
    char full[1024];
    snprintf(full, sizeof(full), "%s\\%s", dir, name);
    if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) walk_dir(full, b);
    else if (has_txt_ext(name)) append_file(full, b);
  } while (FindNextFileA(h, &fd));
  FindClose(h);
#else
  DIR *d = opendir(dir);
  if (!d) return;
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
    char full[1024];
    snprintf(full, sizeof(full), "%s/%s", dir, e->d_name);
    if (path_is_dir(full)) walk_dir(full, b);
    else if (has_txt_ext(e->d_name)) append_file(full, b);
  }
  closedir(d);
#endif
}

char *corpus_read(const char *path, long *out_len) {
  Buf b = {NULL, 0, 0};

  if (path_is_dir(path)) {
    walk_dir(path, &b);
  } else {
    append_file(path, &b);
  }

  if (b.len == 0) {
    free(b.data);
    return NULL;
  }
  if (out_len) *out_len = b.len;
  return b.data;
}
