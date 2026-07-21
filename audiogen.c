// audiogen.c - Audio <-> token codec for generation (mu-law, 256 levels)
#include "audiogen.h"

#include <stdlib.h>
#include <math.h>

#define MU 255.0f

int audiogen_vocab(void) { return 256; }

// mu-law compand a sample in [-1,1] to a token in [0,255].
static int mulaw_encode(float x) {
  float s = (x < 0.0f) ? -1.0f : 1.0f;
  float a = fabsf(x); if (a > 1.0f) a = 1.0f;
  float y = s * logf(1.0f + MU * a) / logf(1.0f + MU);   // [-1,1]
  int q = (int)((y + 1.0f) * 0.5f * 255.0f + 0.5f);      // [0,255]
  if (q < 0) q = 0;
  if (q > 255) q = 255;
  return q;
}

// Inverse mu-law: token in [0,255] back to a sample in [-1,1].
static float mulaw_decode(int q) {
  float y = (float)q / 255.0f * 2.0f - 1.0f;             // [-1,1]
  float s = (y < 0.0f) ? -1.0f : 1.0f;
  float a = (powf(1.0f + MU, fabsf(y)) - 1.0f) / MU;     // [0,1]
  return s * a;
}

void audio_to_tokens(const Audio *a, int *tokens) {
  for (int i = 0; i < a->n; i++) tokens[i] = mulaw_encode(a->samples[i]);
}

Audio tokens_to_audio(const int *tokens, int n, int sample_rate) {
  Audio a;
  a.n = n;
  a.sample_rate = sample_rate;
  a.samples = (float*)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) a.samples[i] = mulaw_decode(tokens[i]);
  return a;
}
