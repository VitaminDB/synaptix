"""Эталонная генерация FLUX.1-dev (Python/diffusers) для сравнения качества/времени
с synaptix. 1024², 28 шагов, guidance 3.5, seed 0. enable_model_cpu_offload
(полный FLUX 34GB не влезает в 24GB). Замеряет ТОЛЬКО денойз+VAE (без загрузки)."""
import time
import torch
from diffusers import FluxPipeline

FLUX = "models/black-forest-labs/FLUX.1-dev"
PROMPT = "a photorealistic portrait of a young woman with freckles, soft natural window light, 85mm lens, shallow depth of field"
H = W = 1024
STEPS = 28
OUT = "/tmp/flux_python_1024.png"


@torch.no_grad()
def main():
    pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    # warm-load дешёвых компонентов уже в from_pretrained; меряем чистую генерацию
    gen = torch.Generator("cpu").manual_seed(0)
    t0 = time.time()
    img = pipe(PROMPT, height=H, width=W, num_inference_steps=STEPS,
               guidance_scale=3.5, generator=gen).images[0]
    dt = time.time() - t0
    img.save(OUT)
    print(f"PYTHON_FLUX 1024x1024 {STEPS} steps: {dt:.1f}s ({dt/STEPS:.2f}s/step) -> {OUT}")


if __name__ == "__main__":
    main()
