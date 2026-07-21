// video.h - Video = a sequence of frames (reuses the image patch front-end)
#ifndef VIDEO_H
#define VIDEO_H

#include "tensor.h"
#include "image.h"

// A clip is a sequence of frames (each an RGB Image, all the same size).
typedef struct {
  Image *frames;
  int n;
} Video;

// Load frames dir/frame_0000.ppm, frame_0001.ppm, ... until one is missing.
// Use tools/video_to_frames.py to extract frames from a real video.
Video video_load(const char *dir);
void video_free(Video *v);

// Total patch tokens for the clip: n_frames * (per-frame patch count).
int video_patch_count(const Video *v, int patch);

// Patch-embed every frame with the shared projection (W: patch_dim x d_model,
// b: d_model) and stack them into one sequence, adding a per-frame sinusoidal
// TEMPORAL encoding so the model can tell frames apart.
//   out : (video_patch_count, d_model), caller-allocated.
void video_embed(const Video *v, int patch, const Tensor *W, const Tensor *b, Tensor *out);

// Backward: accumulate W->grad / b->grad from out->grad (per frame). The temporal
// encoding is a constant, so it doesn't affect the projection gradients.
void video_embed_backward(const Video *v, int patch, const Tensor *W,
                          const Tensor *b, const Tensor *out);

#endif // VIDEO_H
