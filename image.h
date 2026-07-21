// image.h - Minimal image loading (binary PPM / P6)
#ifndef IMAGE_H
#define IMAGE_H

// An RGB image: h*w*c floats in [0,1], row-major HWC layout (c is always 3).
typedef struct {
  int h, w, c;
  float *data;
} Image;

// Load a binary PPM (P6). Returns {0,0,0,NULL} on error. Free with image_free.
// PNG/JPG can be converted to PPM with tools/img_to_ppm.py.
Image image_load_ppm(const char *path);

// Write an Image as a binary PPM (P6). Returns 0 on success.
int image_save_ppm(const char *path, const Image *im);

void image_free(Image *im);

#endif // IMAGE_H
