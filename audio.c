// audio.c - Audio loading + framing into a sequence of vectors (Phase F: audio)
#include "audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "matrix.h"

// Load 16-bit PCM WAV. Walks RIFF chunks so extra chunks (LIST/fact/...) are
// skipped; stereo is downmixed to mono. Returns {NULL,0,0} on error.
Audio audio_load_wav(const char *path) {
  Audio a = {NULL, 0, 0};
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "audio: cannot open '%s'\n", path); return a; }

  char tag[4];
  uint32_t sz;
  if (fread(tag, 1, 4, f) != 4 || memcmp(tag, "RIFF", 4) != 0) { fclose(f); return a; }
  fseek(f, 4, SEEK_CUR);                         // RIFF chunk size
  if (fread(tag, 1, 4, f) != 4 || memcmp(tag, "WAVE", 4) != 0) { fclose(f); return a; }

  int channels = 1, rate = 8000, bits = 16, have_fmt = 0;
  for (;;) {
    if (fread(tag, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1) break;
    if (memcmp(tag, "fmt ", 4) == 0) {
      uint16_t fmt, ch, ba, bps; uint32_t sr, br;
      fread(&fmt, 2, 1, f); fread(&ch, 2, 1, f); fread(&sr, 4, 1, f);
      fread(&br, 4, 1, f);  fread(&ba, 2, 1, f); fread(&bps, 2, 1, f);
      channels = ch ? ch : 1; rate = sr; bits = bps; have_fmt = 1;
      if (sz > 16) fseek(f, sz - 16, SEEK_CUR);  // skip any extended fmt bytes
    } else if (memcmp(tag, "data", 4) == 0) {
      if (!have_fmt || bits != 16) {             // only 16-bit PCM supported
        fprintf(stderr, "audio: '%s' must be 16-bit PCM\n", path);
        fclose(f); return a;
      }
      int16_t *raw = (int16_t*)malloc(sz);
      if (fread(raw, 1, sz, f) != sz) { free(raw); fclose(f); return a; }
      int total = (int)(sz / 2);                 // total 16-bit samples (all channels)
      int nmono = total / channels;
      a.samples = (float*)malloc(nmono * sizeof(float));
      for (int i = 0; i < nmono; i++) {
        int acc = 0;
        for (int c = 0; c < channels; c++) acc += raw[i * channels + c];
        a.samples[i] = (float)acc / channels / 32768.0f;
      }
      a.n = nmono; a.sample_rate = rate;
      free(raw);
      break;
    } else {
      fseek(f, sz + (sz & 1), SEEK_CUR);         // skip unknown chunk (word-aligned)
    }
  }
  fclose(f);
  return a;
}

int audio_save_wav(const char *path, const Audio *a) {
  FILE *f = fopen(path, "wb");
  if (!f) return 1;
  uint32_t dsz = (uint32_t)(a->n * 2), x;
  uint16_t u16;
  fwrite("RIFF", 1, 4, f); x = 36 + dsz; fwrite(&x, 4, 1, f); fwrite("WAVE", 1, 4, f);
  fwrite("fmt ", 1, 4, f); x = 16; fwrite(&x, 4, 1, f);
  u16 = 1; fwrite(&u16, 2, 1, f); u16 = 1; fwrite(&u16, 2, 1, f);           // PCM, mono
  x = a->sample_rate; fwrite(&x, 4, 1, f);
  x = a->sample_rate * 2; fwrite(&x, 4, 1, f);                             // byte rate
  u16 = 2; fwrite(&u16, 2, 1, f); u16 = 16; fwrite(&u16, 2, 1, f);         // block align, bits
  fwrite("data", 1, 4, f); fwrite(&dsz, 4, 1, f);
  for (int i = 0; i < a->n; i++) {
    float s = a->samples[i];
    if (s > 1.0f) s = 1.0f;
    if (s < -1.0f) s = -1.0f;
    int16_t v = (int16_t)(s * 32767.0f);
    fwrite(&v, 2, 1, f);
  }
  fclose(f);
  return 0;
}

void audio_free(Audio *a) {
  free(a->samples);
  a->samples = NULL;
  a->n = a->sample_rate = 0;
}

int audio_frame_count(const Audio *a, int frame_size, int hop) {
  if (a->n < frame_size) return 0;
  return (a->n - frame_size) / hop + 1;
}

// Sinusoidal temporal encoding for frame f, dimension d of d_model.
static float temporal_enc(int f, int d, int dm) {
  float exponent = (float)(2 * (d / 2)) / (float)dm;
  float ang = (float)f / powf(10000.0f, exponent);
  return (d % 2 == 0) ? sinf(ang) : cosf(ang);
}

// Build the (n_frames, frame_size) matrix of windowed samples.
static Tensor build_frames(const Audio *a, int frame_size, int hop) {
  int nf = audio_frame_count(a, frame_size, hop);
  int shape[2] = {nf, frame_size};
  Tensor flat = tensor_create(shape, 2);
  for (int i = 0; i < nf; i++) {
    int start = i * hop;
    for (int j = 0; j < frame_size; j++) flat.data[i * frame_size + j] = a->samples[start + j];
  }
  return flat;
}

void audio_embed(const Audio *a, int frame_size, int hop,
                 const Tensor *W, const Tensor *b, Tensor *out) {
  Tensor flat = build_frames(a, frame_size, hop);
  matmul(&flat, W, out);          // (nf, frame_size) @ (frame_size, d_model)
  bias_add(out, b, out);
  const int nf = out->shape[0], dm = out->shape[1];
  for (int f = 0; f < nf; f++)
    for (int d = 0; d < dm; d++) out->data[f * dm + d] += temporal_enc(f, d, dm);
  tensor_free(&flat);
}

void audio_embed_backward(const Audio *a, int frame_size, int hop,
                          const Tensor *W, const Tensor *b, const Tensor *out) {
  Tensor flat = build_frames(a, frame_size, hop);
  bias_add_backward(out, b);
  matmul_backward(&flat, W, out);
  tensor_free(&flat);
}
