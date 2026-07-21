#!/usr/bin/env python3
"""Convert an image (PNG/JPG/...) to a binary PPM the C image loader reads.

    pip install pillow
    python tools/img_to_ppm.py cat.png data/images/cat.ppm 64   # optional square resize

The C code (image.c) keeps things dependency-free by reading PPM; this tool does
the format/decoding in Python. Output goes wherever you point it — e.g. under
data/images/ for a future image-text training set.
"""
import os
import sys

from PIL import Image


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: img_to_ppm.py <in-image> <out.ppm> [size]")
    src, dst = sys.argv[1], sys.argv[2]
    im = Image.open(src).convert("RGB")
    if len(sys.argv) > 3:
        s = int(sys.argv[3])
        im = im.resize((s, s))
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    im.save(dst, format="PPM")   # Pillow writes binary P6 for .ppm
    print(f"wrote {dst}: {im.width}x{im.height} RGB")


if __name__ == "__main__":
    main()
