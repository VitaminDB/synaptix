"""Reference data for synaptix FLUX VAE decoder (bit-exact check on CUDA).

FLUX VAE = AutoencoderKL с 16 латентными каналами, БЕЗ quant_conv/post_quant_conv
(use_quant_conv=false), scaling_factor=0.3611, shift_factor=0.1159. Загружаем в
float32 (апкаст из bf16, чтобы совпасть с synaptix-валидацией) и декодируем
фиксированный латент. Дампим:
  - decode: z[1,16,16,16] (вход декодера, БЕЗ scaling/shift) + sample[1,3,128,128]
    (= vae.decode(z).sample — голый выход декодера).
  - pipe_decode: z_pipe[1,16,16,16] (как из denoise) + image[1,3,128,128] после
    pipeline-шага latents = z/scaling + shift, затем decode (полный путь pipeline).

Запуск из корня synaptix через reference venv:
  scripts/reference/.venv/bin/python scripts/reference/gen_flux_vae.py
"""

import pathlib

import torch
from diffusers import AutoencoderKL

FLUX = "models/black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = pathlib.Path("tests/reference_data/flux_vae")
LATENT_HW = 16  # -> image 128x128 (16 * 2^3), задействует все up-блоки


def save_case(name, tensors):
    from safetensors.torch import save_file

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def main():
    vae = AutoencoderKL.from_pretrained(f"{FLUX}/vae", torch_dtype=torch.float32).float()
    vae.eval()
    sf = vae.config.scaling_factor
    shf = getattr(vae.config, "shift_factor", None)
    print("latent_channels", vae.config.latent_channels, "scaling", sf, "shift", shf,
          "use_quant_conv", vae.config.use_quant_conv,
          "use_post_quant_conv", vae.config.use_post_quant_conv)

    # --- decode: голый декодер z -> image (без scaling/shift) ---
    torch.manual_seed(0)
    z = torch.randn(1, vae.config.latent_channels, LATENT_HW, LATENT_HW, dtype=torch.float32)
    with torch.no_grad():
        sample = vae.decode(z, return_dict=True).sample  # [1,3,128,128]
    print("z", tuple(z.shape), "sample", tuple(sample.shape),
          "range", float(sample.min()), float(sample.max()))
    save_case("decode", {"z": z, "sample": sample})

    # --- pipe_decode: полный pipeline-путь (scaling/shift внутри) ---
    torch.manual_seed(2)
    z_pipe = torch.randn(1, vae.config.latent_channels, LATENT_HW, LATENT_HW, dtype=torch.float32)
    with torch.no_grad():
        lat = z_pipe / sf + (shf if shf is not None else 0.0)
        image = vae.decode(lat, return_dict=True).sample
    print("pipe image range", float(image.min()), float(image.max()))
    save_case("pipe_decode", {"z_pipe": z_pipe, "image": image})


if __name__ == "__main__":
    main()
