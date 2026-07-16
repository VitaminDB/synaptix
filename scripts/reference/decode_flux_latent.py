"""Декод сохранённого Python final_latent (из io.safetensors, инжект-инпуты) →
PNG-эталон. Те же входы, что синапс-инжект гоняет → прямое сравнение зерна."""
import pathlib
import torch
from safetensors.torch import load_file
from diffusers import FluxPipeline

FLUX = "models/black-forest-labs/FLUX.1-dev"
IO = "tests/reference_data/flux_io/io.safetensors"
H = W = 512
OUT = "/tmp/flux_python_ref.png"


@torch.no_grad()
def main():
    d = load_file(IO)
    lat = d["final_latent"].to("cuda", torch.bfloat16)  # [1,1024,64] packed
    pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16, transformer=None, text_encoder=None, text_encoder_2=None)
    pipe.vae.to("cuda")
    lat = pipe._unpack_latents(lat, H, W, pipe.vae_scale_factor)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat, return_dict=False)[0]
    img = pipe.image_processor.postprocess(img, output_type="pil")[0]
    img.save(OUT)
    print("saved", OUT, img.size)


if __name__ == "__main__":
    main()
