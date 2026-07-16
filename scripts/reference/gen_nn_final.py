"""Reference SafeTensors для двух последних стабов synaptix-nn: dit_joint и squeezeformer.

Run:
    scripts/reference/.venv/bin/python scripts/reference/gen_nn_final.py
"""

import pathlib
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_final")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def squeezeformer_forward(x, proj_w, proj_b):
    """Mirror src/squeezeformer.rs::Squeezeformer::forward."""
    b, t, _c = x.shape
    if t < 2:
        return F.linear(x, proj_w, proj_b)
    half = t // 2
    # subsample stride 2: indices 0, 2, 4, ..., 2*half-2
    sub = x[:, :2 * half:2, :]
    projected = F.linear(sub, proj_w, proj_b)
    # repeat-interleave по seq на 2
    upsampled = projected.repeat_interleave(2, dim=1)
    cur_t = upsampled.shape[1]
    if cur_t == t:
        return upsampled
    if cur_t < t:
        last = upsampled[:, cur_t - 1:cur_t, :]
        pads = last.expand(-1, t - cur_t, -1)
        return torch.cat([upsampled, pads], dim=1)
    return upsampled[:, :t, :]


def case_squeezeformer_even() -> None:
    torch.manual_seed(1300)
    b, t, in_ch, hidden = 2, 6, 4, 8  # T чётное
    proj_w = torch.randn(hidden, in_ch) * 0.1
    proj_b = torch.randn(hidden) * 0.05
    x = torch.randn(b, t, in_ch)
    out = squeezeformer_forward(x, proj_w, proj_b)
    save_case("squeezeformer_even", {
        "proj_w": proj_w, "proj_b": proj_b,
        "x": x, "output": out,
    })


def case_squeezeformer_odd() -> None:
    torch.manual_seed(1301)
    b, t, in_ch, hidden = 1, 5, 3, 6  # T нечётное → padding
    proj_w = torch.randn(hidden, in_ch) * 0.1
    proj_b = torch.randn(hidden) * 0.05
    x = torch.randn(b, t, in_ch)
    out = squeezeformer_forward(x, proj_w, proj_b)
    save_case("squeezeformer_odd", {
        "proj_w": proj_w, "proj_b": proj_b,
        "x": x, "output": out,
    })


def main() -> None:
    print("Generating nn-final (squeezeformer) reference data...")
    case_squeezeformer_even()
    case_squeezeformer_odd()
    print("Done.")


if __name__ == "__main__":
    main()
