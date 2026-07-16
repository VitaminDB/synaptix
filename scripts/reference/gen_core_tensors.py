"""Generate reference SafeTensors for Session 1 — Core Tensor ops.

Run:
    python scripts/reference/gen_core_tensors.py

Outputs data/ref/core_tensors/<case>.safetensors, each containing:
    - 'input' (or 'input_a' / 'input_b' for binary ops)
    - 'output' — PyTorch result
    - optional auxiliary tensors (mask, indices, etc.)
"""

import pathlib

import numpy as np
import torch
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/core_tensors")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_matmul_2d() -> None:
    torch.manual_seed(0)
    a = torch.randn(64, 128, dtype=torch.float32)
    b = torch.randn(128, 256, dtype=torch.float32)
    out = torch.mm(a, b)
    save_case("matmul_2d", {"input_a": a, "input_b": b, "output": out})


def case_matmul_batch() -> None:
    torch.manual_seed(1)
    a = torch.randn(4, 32, 64, dtype=torch.float32)
    b = torch.randn(4, 64, 128, dtype=torch.float32)
    out = torch.bmm(a, b)
    save_case("matmul_batch", {"input_a": a, "input_b": b, "output": out})


def case_reduce_sum_dim1() -> None:
    torch.manual_seed(2)
    x = torch.randn(8, 256, dtype=torch.float32)
    out = x.sum(dim=1)
    save_case("reduce_sum_dim1", {"input": x, "output": out})


def case_reduce_sum_all() -> None:
    torch.manual_seed(3)
    x = torch.randn(16, 64, dtype=torch.float32)
    out = x.sum().reshape(1)
    save_case("reduce_sum_all", {"input": x, "output": out})


def case_reduce_mean() -> None:
    torch.manual_seed(4)
    x = torch.randn(8, 64, dtype=torch.float32)
    out = x.mean(dim=-1)
    save_case("reduce_mean", {"input": x, "output": out})


def case_argmax() -> None:
    torch.manual_seed(5)
    x = torch.randn(4, 128, dtype=torch.float32)
    out = x.argmax(dim=-1).to(torch.int64)
    save_case("argmax", {"input": x, "output": out})


def case_broadcast_add() -> None:
    torch.manual_seed(6)
    a = torch.randn(4, 1, 64, dtype=torch.float32)
    b = torch.randn(1, 8, 64, dtype=torch.float32)
    out = a + b
    save_case("broadcast_add", {"input_a": a, "input_b": b, "output": out})


def case_gather_2d() -> None:
    torch.manual_seed(7)
    x = torch.randn(16, 256, dtype=torch.float32)
    indices = torch.randint(0, 256, (16, 32), dtype=torch.int64)
    out = torch.gather(x, dim=1, index=indices)
    save_case("gather_2d", {"input": x, "indices": indices, "output": out})


def case_masked_fill() -> None:
    torch.manual_seed(8)
    x = torch.randn(4, 64, dtype=torch.float32)
    mask = torch.randint(0, 2, (4, 64), dtype=torch.bool)
    out = x.masked_fill(mask, -1e9)
    save_case("masked_fill", {"input": x, "mask": mask.to(torch.uint8), "output": out})


def case_cat_dim0() -> None:
    torch.manual_seed(9)
    a = torch.randn(8, 64, dtype=torch.float32)
    b = torch.randn(12, 64, dtype=torch.float32)
    out = torch.cat([a, b], dim=0)
    save_case("cat_dim0", {"input_a": a, "input_b": b, "output": out})


def case_cast_f32_bf16_f32() -> None:
    torch.manual_seed(10)
    x = torch.randn(16, 128, dtype=torch.float32)
    bf16 = x.to(torch.bfloat16)
    out = bf16.to(torch.float32)
    save_case("cast_f32_bf16_f32", {"input": x, "output": out})


def main() -> None:
    print("Generating core tensor reference data...")
    case_matmul_2d()
    case_matmul_batch()
    case_reduce_sum_dim1()
    case_reduce_sum_all()
    case_reduce_mean()
    case_argmax()
    case_broadcast_add()
    case_gather_2d()
    case_masked_fill()
    case_cat_dim0()
    case_cast_f32_bf16_f32()
    print("Done.")


if __name__ == "__main__":
    main()
