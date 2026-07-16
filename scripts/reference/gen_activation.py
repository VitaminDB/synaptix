"""Generate reference SafeTensors for Session 2 — Activation functions.

Run:
    python scripts/reference/gen_activation.py

Each case saves input F32/F16/BF16 + output for 13 activation functions.
Output dir: data/ref/activation/<name>_<dtype>.safetensors
"""

import pathlib
from typing import Callable

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/activation")

DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def gen_activation(
    fn_name: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    seed: int = 0,
    positive_only: bool = False,
) -> None:
    torch.manual_seed(seed)
    base = torch.randn(8, 256, dtype=torch.float32)
    if positive_only:
        base = base.abs() + 0.01
    for dtype_name, dtype in DTYPES.items():
        x = base.to(dtype)
        out = fn(x.float()).to(dtype)
        save_case(f"{fn_name}_{dtype_name}", {"input": x, "output": out})


def main() -> None:
    print("Generating activation reference data...")

    gen_activation("relu", F.relu, seed=0)
    gen_activation("gelu_tanh", lambda x: F.gelu(x, approximate="tanh"), seed=1)
    gen_activation("gelu_exact", lambda x: F.gelu(x, approximate="none"), seed=2)
    gen_activation("silu", F.silu, seed=3)
    gen_activation("mish", F.mish, seed=4)
    gen_activation("elu", lambda x: F.elu(x, alpha=1.0), seed=5)
    gen_activation("leaky_relu", lambda x: F.leaky_relu(x, negative_slope=0.01), seed=6)
    gen_activation("hardswish", F.hardswish, seed=7)
    gen_activation("sigmoid", torch.sigmoid, seed=8)
    gen_activation("tanh", torch.tanh, seed=9)
    gen_activation(
        "softmax",
        lambda x: F.softmax(x, dim=-1),
        seed=10,
    )
    gen_activation(
        "log_softmax",
        lambda x: F.log_softmax(x, dim=-1),
        seed=11,
    )
    gen_activation("softplus", F.softplus, seed=12)

    print("Done.")


if __name__ == "__main__":
    main()
