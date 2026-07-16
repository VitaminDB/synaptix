"""Бенчмарк diffusers FLUX.1-dev 1024² — замер времени генерации для сравнения
с synaptix. enable_model_cpu_offload: трансформер резидентен на GPU во время
denoise (CLIP/T5/VAE оффлоадятся), как в типичном инференсе на 24GB."""

import time

import torch
from diffusers import FluxPipeline

FLUX = "models/black-forest-labs/FLUX.1-dev"


def main():
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    print(f"load {time.time()-t0:.1f}s")

    steps = 50
    # warmup compile/alloc — НЕ считаем (но FLUX без compile; первый прогон включает оффлоад-разогрев)
    t1 = time.time()
    img = pipe(
        "a photo of a red apple on a wooden table, cinematic lighting",
        height=1024, width=1024,
        guidance_scale=3.5,
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    dt = time.time() - t1
    img.save("/tmp/flux_python_1024.png")
    print(f"PYTHON FLUX 1024² {steps} steps: {dt:.1f}s ({dt/steps:.2f}s/step)")
    print(f"max VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()
