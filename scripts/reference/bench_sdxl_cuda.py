"""Бенч diffusers SDXL на CUDA (BF16) для сравнения с нативным synaptix.
1024²×25, тот же model dir / prompt / seed. Меряем generation (без load),
warmup + torch.cuda.synchronize."""
import time
import torch
from diffusers import StableDiffusionXLPipeline

MODEL = "models/stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "a red apple on a wooden table, photorealistic, studio lighting, highly detailed"
H = W = 1024
STEPS = 25
CFG = 5.0
SEED = 42

t0 = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True
).to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"load: {time.time()-t0:.2f}s  scheduler={type(pipe.scheduler).__name__}")


def gen(steps):
    g = torch.Generator("cuda").manual_seed(SEED)
    torch.cuda.synchronize()
    t = time.time()
    img = pipe(PROMPT, height=H, width=W, num_inference_steps=steps,
               guidance_scale=CFG, generator=g).images[0]
    torch.cuda.synchronize()
    return time.time() - t, img

# warmup (компиляция/инициализация CUDA-аллокаторов)
gen(3)
dt, img = gen(STEPS)
print(f"diffusers SDXL CUDA bf16 {W}x{H} {STEPS} steps: {dt:.1f}s ({dt/STEPS:.2f}s/step)")
img.save("/tmp/sdxl_diffusers.png")
print("saved /tmp/sdxl_diffusers.png")
