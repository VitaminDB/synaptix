"""Generate reference SafeTensors for synaptix-nn VAE primitives."""

import pathlib

import torch
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_vae")


def save_case(name, tensors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_pixel_norm():
    torch.manual_seed(900)
    x = torch.randn(2, 3, 8)
    eps = 1e-8
    sq = x * x
    mean = sq.mean(dim=-1, keepdim=True)
    denom = (mean + eps).sqrt()
    out = x / denom
    save_case("pixel_norm", {"x": x, "output": out})


def case_per_channel_stats():
    torch.manual_seed(901)
    x = torch.randn(2, 4, 8, 8)
    mean = torch.tensor([0.5, -0.3, 1.0, 0.0])
    std = torch.tensor([1.2, 0.7, 1.5, 1.0])
    m = mean.view(1, 4, 1, 1)
    s = std.view(1, 4, 1, 1)
    normed = (x - m) / s
    denormed = normed * s + m
    save_case(
        "per_channel_stats",
        {"x": x, "mean": mean, "std": std, "normalized": normed, "denormalized": denormed},
    )


def case_reparameterize():
    torch.manual_seed(902)
    mean = torch.randn(2, 4)
    logvar = torch.randn(2, 4) * 0.5
    eps = torch.randn(2, 4)
    std = (0.5 * logvar).exp()
    out = mean + eps * std
    save_case(
        "reparameterize",
        {"mean": mean, "logvar": logvar, "eps": eps, "output": out},
    )


def case_kl_divergence():
    torch.manual_seed(903)
    mean = torch.randn(2, 4)
    logvar = torch.randn(2, 4) * 0.5
    kl = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1.0)
    save_case(
        "kl_divergence",
        {"mean": mean, "logvar": logvar, "output": kl},
    )


def main():
    print("Generating nn-vae reference data...")
    case_pixel_norm()
    case_per_channel_stats()
    case_reparameterize()
    case_kl_divergence()
    print("Done.")


if __name__ == "__main__":
    main()
