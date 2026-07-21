// audiogen.h - Audio <-> token codec for generation (mu-law, 256 levels)
#ifndef AUDIOGEN_H
#define AUDIOGEN_H

#include "audio.h"

// Discrete-token audio generation, minimal version: each sample is companded
// with 8-bit mu-law into one of 256 tokens (WaveNet-style). The GPT then
// generates the token sequence autoregressively; we decode tokens back to a
// waveform. (A learned neural audio codec would give real quality; this is the
// dependency-free minimal path.)

// Vocabulary size (256).
int audiogen_vocab(void);

// mu-law encode every sample of `a` to a token in [0,255]. Writes a->n tokens.
void audio_to_tokens(const Audio *a, int *tokens);

// Decode n tokens back into an Audio clip at the given sample rate.
Audio tokens_to_audio(const int *tokens, int n, int sample_rate);

#endif // AUDIOGEN_H
