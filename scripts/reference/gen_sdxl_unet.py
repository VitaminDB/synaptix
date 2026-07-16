"""Reference data for synaptix SDXL UNet (UNet2DConditionModel) bit-exact check.

Loads the SDXL UNet in float32 (from fp16 weights), runs one forward on a small
latent with random conditioning and dumps all inputs + the eps-prediction output.

Run from the synaptix repo root with the reference venv.
"""

import pathlib

import torch
from diffusers import UNet2DConditionModel

SDXL = "models/stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = pathlib.Path("tests/reference_data/sdxl_unet")
LAT = 16  # latent H=W (small; 2 downsamples -> 4x4 at the 1280-ch bottleneck)


def save_case(name, tensors):
    from safetensors.torch import save_file

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def main():
    unet = UNet2DConditionModel.from_pretrained(
        f"{SDXL}/unet", torch_dtype=torch.float16, variant="fp16"
    ).float()
    unet.eval()

    torch.manual_seed(0)
    sample = torch.randn(1, unet.config.in_channels, LAT, LAT, dtype=torch.float32)
    timestep = torch.tensor([981.0], dtype=torch.float32)
    ehs = torch.randn(1, 77, unet.config.cross_attention_dim, dtype=torch.float32)
    text_embeds = torch.randn(1, 1280, dtype=torch.float32)
    time_ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]], dtype=torch.float32)

    added = {"text_embeds": text_embeds, "time_ids": time_ids}
    with torch.no_grad():
        out = unet(sample, timestep, encoder_hidden_states=ehs, added_cond_kwargs=added).sample
    print("out", tuple(out.shape), "range", float(out.min()), float(out.max()))

    save_case(
        "forward",
        {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": ehs,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
            "out": out,
        },
    )


if __name__ == "__main__":
    main()
