"""Generate reference SafeTensors for Session 2 — Normalization layers.

Run:
    python scripts/reference/gen_norm.py

Outputs data/ref/norm/<case>.safetensors containing input, weight/bias, output.
Uses PyTorch nn modules or transformers LlamaRMSNorm as ground truth.
"""

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/norm")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_rms_norm() -> None:
    torch.manual_seed(0)
    hidden = 256
    x = torch.randn(4, 32, hidden, dtype=torch.float32)
    weight = torch.ones(hidden, dtype=torch.float32) + torch.randn(hidden) * 0.02
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + 1e-6)
    out = x_norm * weight
    save_case("rms_norm", {"input": x, "weight": weight, "output": out})


def case_rms_norm_qwen() -> None:
    torch.manual_seed(1)
    hidden = 256
    x = torch.randn(4, 32, hidden, dtype=torch.float32)
    weight = torch.ones(hidden, dtype=torch.float32) + torch.randn(hidden) * 0.02
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_norm = x_f32 * torch.rsqrt(variance + 1e-6)
    out = (weight + 1.0) * x_norm
    save_case("rms_norm_qwen", {"input": x, "weight": weight, "output": out.to(x.dtype)})


def case_rms_norm_gated() -> None:
    torch.manual_seed(2)
    hidden = 256
    x = torch.randn(4, 32, hidden, dtype=torch.float32)
    gate = torch.randn(4, 32, hidden, dtype=torch.float32)
    weight = torch.ones(hidden, dtype=torch.float32)
    gated = x * F.silu(gate)
    variance = gated.pow(2).mean(-1, keepdim=True)
    gated_norm = gated * torch.rsqrt(variance + 1e-6)
    out = gated_norm * weight
    save_case("rms_norm_gated", {"input": x, "gate": gate, "weight": weight, "output": out})


def case_layer_norm() -> None:
    torch.manual_seed(3)
    hidden = 256
    x = torch.randn(4, 32, hidden, dtype=torch.float32)
    ln = nn.LayerNorm(hidden, eps=1e-5, elementwise_affine=True)
    out = ln(x)
    save_case(
        "layer_norm",
        {
            "input": x,
            "weight": ln.weight.data,
            "bias": ln.bias.data,
            "output": out.detach(),
        },
    )


def case_group_norm() -> None:
    torch.manual_seed(4)
    channels, groups = 256, 8
    x = torch.randn(4, channels, 32, dtype=torch.float32)
    gn = nn.GroupNorm(groups, channels, eps=1e-5)
    out = gn(x)
    save_case(
        "group_norm",
        {
            "input": x,
            "weight": gn.weight.data,
            "bias": gn.bias.data,
            "output": out.detach(),
        },
    )


def case_batch_norm_inference() -> None:
    torch.manual_seed(5)
    channels = 64
    x = torch.randn(8, channels, 32, dtype=torch.float32)
    bn = nn.BatchNorm1d(channels, eps=1e-5, momentum=0.1)
    bn.eval()
    bn.running_mean.data = torch.randn(channels)
    bn.running_var.data = torch.rand(channels) + 0.5
    out = bn(x)
    save_case(
        "batch_norm_inference",
        {
            "input": x,
            "weight": bn.weight.data,
            "bias": bn.bias.data,
            "running_mean": bn.running_mean.data,
            "running_var": bn.running_var.data,
            "output": out.detach(),
        },
    )


def case_instance_norm() -> None:
    torch.manual_seed(6)
    channels = 64
    x = torch.randn(4, channels, 32, dtype=torch.float32)
    inst = nn.InstanceNorm1d(channels, eps=1e-5, affine=True)
    out = inst(x)
    save_case(
        "instance_norm",
        {
            "input": x,
            "weight": inst.weight.data,
            "bias": inst.bias.data,
            "output": out.detach(),
        },
    )


def case_adaln_zero() -> None:
    torch.manual_seed(7)
    hidden = 256
    cond_dim = 128
    x = torch.randn(4, 32, hidden, dtype=torch.float32)
    cond = torch.randn(4, cond_dim, dtype=torch.float32)
    mlp = nn.Sequential(nn.Linear(cond_dim, hidden * 2), nn.SiLU())
    with torch.no_grad():
        scale_shift = mlp(cond).unsqueeze(1)
    scale, shift = scale_shift.chunk(2, dim=-1)
    ln = nn.LayerNorm(hidden, eps=1e-6)
    out = ln(x) * (1.0 + scale) + shift
    save_case(
        "adaln_zero",
        {
            "input": x,
            "cond": cond,
            "scale": scale.detach(),
            "shift": shift.detach(),
            "ln_weight": ln.weight.data,
            "ln_bias": ln.bias.data,
            "output": out.detach(),
        },
    )


def case_pixel_norm() -> None:
    torch.manual_seed(8)
    x = torch.randn(4, 32, 256, dtype=torch.float32)
    eps = 1e-8
    out = x / (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    save_case("pixel_norm", {"input": x, "output": out})


def main() -> None:
    print("Generating norm reference data...")
    case_rms_norm()
    case_rms_norm_qwen()
    case_rms_norm_gated()
    case_layer_norm()
    case_group_norm()
    case_batch_norm_inference()
    case_instance_norm()
    case_adaln_zero()
    case_pixel_norm()
    print("Done.")


if __name__ == "__main__":
    main()
