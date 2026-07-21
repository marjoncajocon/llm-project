// audio.h - Audio loading + framing into a sequence of vectors (Phase F: audio)
#ifndef AUDIO_H
#define AUDIO_H

#include "tensor.h"

// A mono audio clip: n float samples in [-1, 1] at sample_rate Hz.
typedef struct {
  float *samples;
  int n;
  int sample_rate;
} Audio;

// Load a 16-bit PCM WAV (mono, or stereo downmixed to mono). Returns
// {NULL,0,0} on error. Use tools/audio_to_wav.py to convert other formats.
Audio audio_load_wav(const char *path);

// Write an Audio clip as a 16-bit PCM mono WAV. Returns 0 on success.
int audio_save_wav(const char *path, const Audio *a);

void audio_free(Audio *a);

// Number of frames: windows of `frame_size` samples advanced by `hop`.
int audio_frame_count(const Audio *a, int frame_size, int hop);

// Frame the waveform (windows of frame_size, stride hop), project each window
// with W (frame_size x d_model) + b, and add a per-frame sinusoidal temporal
// encoding — producing a sequence of vectors the transformer can consume.
//   out : (audio_frame_count, d_model), caller-allocated.
void audio_embed(const Audio *a, int frame_size, int hop,
                 const Tensor *W, const Tensor *b, Tensor *out);

// Backward: accumulate W->grad / b->grad from out->grad (temporal term is a
// constant, so it passes straight through).
void audio_embed_backward(const Audio *a, int frame_size, int hop,
                          const Tensor *W, const Tensor *b, const Tensor *out);

#endif // AUDIO_H
