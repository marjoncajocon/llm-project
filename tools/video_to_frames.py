#!/usr/bin/env python3
"""Extract video frames as PPM for the C video loader (needs ffmpeg on PATH).

    python tools/video_to_frames.py clip.mp4 data/video/clip --fps 1 --size 64

Produces data/video/clip/frame_0000.ppm, frame_0001.ppm, ... which video_load()
reads. Then train video->caption the same way as images (see notes.md, Phase F).
"""
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Extract PPM frames from a video (via ffmpeg)")
    ap.add_argument("video", help="input video file")
    ap.add_argument("outdir", help="output frame directory")
    ap.add_argument("--fps", type=float, default=1.0, help="frames per second to sample")
    ap.add_argument("--size", type=int, default=0, help="square resize (0 = keep original)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    vf = f"fps={args.fps}"
    if args.size:
        vf += f",scale={args.size}:{args.size}"
    cmd = ["ffmpeg", "-i", args.video, "-vf", vf,
           "-start_number", "0", os.path.join(args.outdir, "frame_%04d.ppm")]
    print(" ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
