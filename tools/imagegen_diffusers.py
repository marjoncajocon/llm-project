#!/usr/bin/env python3
"""Real text->image via HuggingFace `diffusers`, called by the webchat hook.

Reads the prompt on stdin, writes a PNG to argv[1]. This runs a real diffusion
model ALONGSIDE the C engine (the engine stays text-only) — the practical way to
get high-quality text-to-image, since diffusion models can't load into the
transformer engine.

    echo "a photo of the sun" | python tools/imagegen_diffusers.py out.png

Model:  $SD_MODEL  (default stabilityai/sd-turbo — 1-step, CPU-tolerant)
Steps:  $SD_STEPS  (default 1)
Deps:   pip install diffusers transformers accelerate torch
        (CPU works but is slow — tens of seconds; first run downloads the model.)
"""
import os
import sys


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: imagegen_diffusers.py <out.png>  (prompt on stdin)\n")
        sys.exit(2)
    out = sys.argv[1]
    prompt = sys.stdin.read().strip() or "a photo"

    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except Exception as e:
        sys.stderr.write(
            "text-to-image needs diffusers + torch (not installed): %s\n"
            "  pip install diffusers transformers accelerate torch\n" % e)
        sys.exit(3)

    model = os.environ.get("SD_MODEL", "stabilityai/sd-turbo")
    steps = int(os.environ.get("SD_STEPS", "1"))
    sys.stderr.write("loading %s (this can take a while on first run)...\n" % model)
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]
    image.save(out)
    sys.stderr.write("generated '%s' with %s (%d steps)\n" % (out, model, steps))


if __name__ == "__main__":
    main()
