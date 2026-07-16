"""Generate reference SafeTensors for Session 1 — CPU Kernels.

Run:
    python scripts/reference/gen_kernels_cpu.py

Outputs data/ref/kernels_cpu/<case>.safetensors.
Q4_0 dequant reference uses a manual quantization mirroring GGML block format:
  block = 32 floats → scale (f16) + 16 bytes (nibbles).
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/kernels_cpu")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_unary_sqrt_f32() -> None:
    torch.manual_seed(0)
    x = torch.rand(8, 256, dtype=torch.float32) + 0.01
    out = torch.sqrt(x)
    save_case("unary_sqrt_f32", {"input": x, "output": out})


def case_unary_silu_f32() -> None:
    torch.manual_seed(1)
    x = torch.randn(8, 256, dtype=torch.float32)
    out = F.silu(x)
    save_case("unary_silu_f32", {"input": x, "output": out})


def case_binary_broadcast_add() -> None:
    torch.manual_seed(2)
    a = torch.randn(4, 1, 128, dtype=torch.float32)
    b = torch.randn(4, 8, 128, dtype=torch.float32)
    out = a + b
    save_case("binary_bcast_add", {"input_a": a, "input_b": b, "output": out})


def case_gemm_f32() -> None:
    torch.manual_seed(3)
    a = torch.randn(128, 512, dtype=torch.float32)
    b = torch.randn(512, 256, dtype=torch.float32)
    out = torch.mm(a, b)
    save_case("gemm_f32", {"input_a": a, "input_b": b, "output": out})


def case_gemm_bf16() -> None:
    torch.manual_seed(4)
    a = torch.randn(64, 256, dtype=torch.bfloat16)
    b = torch.randn(256, 128, dtype=torch.bfloat16)
    out = torch.mm(a, b)
    save_case("gemm_bf16", {"input_a": a, "input_b": b, "output": out})


def case_reduction_sum_dim0() -> None:
    torch.manual_seed(5)
    x = torch.randn(32, 128, dtype=torch.float32)
    out = x.sum(dim=0)
    save_case("reduce_sum_dim0", {"input": x, "output": out})


def _quantize_q4_0(x_f32: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Quantize using GGML Q4_0 format, matching Rust quantize_block_q4_0.

    d = max_signed / -8.0  (element with largest abs preserves sign)
    nibble = clamp(floor(x / d + 8.5), 0, 15)  stored in lower/upper 4 bits
    Layout: first-half elements in lower nibbles, second-half in upper nibbles.
    """
    n = x_f32.size
    assert n % 32 == 0, "length must be multiple of 32"
    blocks = x_f32.reshape(-1, 32)
    raw_bytes = bytearray()
    reconstructed = np.zeros_like(x_f32)

    for i, block in enumerate(blocks):
        abs_vals = np.abs(block)
        max_idx = int(np.argmax(abs_vals))
        max_val = float(block[max_idx])

        d = max_val / -8.0
        id_ = (1.0 / d) if d != 0.0 else 0.0
        d_f16 = np.float16(d)
        raw_bytes += d_f16.tobytes()

        nibbles = bytearray(16)
        recon = np.zeros(32, dtype=np.float32)
        for j in range(16):
            x0 = float(block[j]) * id_
            x1 = float(block[j + 16]) * id_
            xi0 = int(min(max(x0 + 8.5, 0.0), 15.0))
            xi1 = int(min(max(x1 + 8.5, 0.0), 15.0))
            nibbles[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4)
            recon[j] = (xi0 - 8) * float(d_f16)
            recon[j + 16] = (xi1 - 8) * float(d_f16)

        raw_bytes += bytes(nibbles)
        reconstructed[i * 32 : i * 32 + 32] = recon

    return bytes(raw_bytes), reconstructed


def case_q4_0_dequant_matmul() -> None:
    torch.manual_seed(6)
    m, k, n = 16, 128, 64
    x = torch.randn(m, k, dtype=torch.float32)
    w = torch.randn(n, k, dtype=torch.float32)
    w_np = w.numpy().flatten()
    _, w_dequant = _quantize_q4_0(w_np)
    w_dequant_t = torch.from_numpy(w_dequant.reshape(n, k))
    out = torch.mm(x, w_dequant_t.T)
    save_case(
        "q4_0_matmul",
        {
            "input": x,
            "weight_f32": w,
            "weight_dequant": w_dequant_t,
            "output": out,
        },
    )


def main() -> None:
    print("Generating CPU kernels reference data...")
    case_unary_sqrt_f32()
    case_unary_silu_f32()
    case_binary_broadcast_add()
    case_gemm_f32()
    case_gemm_bf16()
    case_reduction_sum_dim0()
    case_q4_0_dequant_matmul()
    print("Done.")


if __name__ == "__main__":
    main()
