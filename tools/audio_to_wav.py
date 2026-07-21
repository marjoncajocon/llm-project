#!/usr/bin/env python3
"""Convert any audio to 16-bit PCM mono WAV for the C audio loader (needs ffmpeg).

    python tools/audio_to_wav.py speech.mp3 data/audio/speech.wav --rate 16000

Then frame + embed it in C (audio.c) and train audio->text the same way as
images/video (see notes.md, Phase F / audio).
"""
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Convert audio to 16-bit PCM mono WAV (via ffmpeg)")
    ap.add_argument("input", help="input audio file")
    ap.add_argument("output", help="output .wav path")
    ap.add_argument("--rate", type=int, default=16000, help="sample rate (Hz)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-i", args.input, "-ac", "1", "-ar", str(args.rate),
           "-c:a", "pcm_s16le", args.output]
    print(" ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
