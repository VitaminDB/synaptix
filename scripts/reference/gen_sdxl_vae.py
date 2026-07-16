"""Reference data for synaptix SDXL VAE decoder (bit-exact check).

Loads the SDXL AutoencoderKL in float32 (from fp16 weights, upcast to match
synaptix) and decodes a fixed random latent. Dumps the latent `z` and the raw
decoder output `sample` (== `vae.decode(z).sample`, which applies
post_quant_conv then the decoder, WITHOUT the scaling_factor division).

Run from the synaptix repo root with the reference venv.
"""

import pathlib

import torch
from diffusers import AutoencoderKL

SDXL = "models/stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = pathlib.Path("tests/reference_data/sdxl_vae")
LATENT_HW = 8  # -> image 64x64 (8 * 2^3), exercises every up-block


def save_case(name, tensors):
    from safetensors.torch import save_file

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def main():
    vae = AutoencoderKL.from_pretrained(
        f"{SDXL}/vae", torch_dtype=torch.float16, variant="fp16"
    ).float()
    vae.eval()
    print("scaling_factor", vae.config.scaling_factor)

    # --- decode: латент -> image (raw decoder output) ---
    torch.manual_seed(0)
    z = torch.randn(1, vae.config.latent_channels, LATENT_HW, LATENT_HW, dtype=torch.float32)
    with torch.no_grad():
        sample = vae.decode(z, return_dict=True).sample  # [1,3,64,64]
    print("z", tuple(z.shape), "sample", tuple(sample.shape), "range",
          float(sample.min()), float(sample.max()))
    save_case("decode", {"z": z, "sample": sample})

    # --- encode: image -> moments (mean|logvar после quant_conv) ---
    torch.manual_seed(1)
    img = LATENT_HW * 8  # 64
    x = torch.randn(1, vae.config.in_channels, img, img, dtype=torch.float32)
    with torch.no_grad():
        moments = vae.encode(x).latent_dist.parameters  # [1,8,8,8]
    print("x", tuple(x.shape), "moments", tuple(moments.shape))
    save_case("encode", {"x": x, "moments": moments})


if __name__ == "__main__":
    main()
