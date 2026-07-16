"""Generate reference SafeTensors for Session 3 — Positional encodings.

Run:
    python scripts/reference/gen_pos.py

Uses transformers for RoPE/YaRN/LongRoPE/ALiBi and plain PyTorch for sinusoidal/T5.
Outputs data/ref/pos/<case>.safetensors.
"""

import math
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/pos")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def _rope_split(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rot = torch.cat([-q2, q1], dim=-1)
    k_rot = torch.cat([-k2, k1], dim=-1)
    q_out = q * cos + q_rot * sin
    k_out = k * cos + k_rot * sin
    return q_out, k_out


def case_rope_split() -> None:
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 2, 8, 32, 64
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    theta = 10000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    positions = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(positions, theta)
    cos = freqs.cos()[None, None, :, :].expand(batch, 1, seq, head_dim // 2)
    sin = freqs.sin()[None, None, :, :].expand(batch, 1, seq, head_dim // 2)
    cos_full = cos.repeat(1, 1, 1, 2)
    sin_full = sin.repeat(1, 1, 1, 2)
    q_out, k_out = _rope_split(q, k, cos_full, sin_full)
    save_case(
        "rope_split",
        {"q": q, "k": k, "cos": cos_full, "sin": sin_full, "q_out": q_out, "k_out": k_out},
    )


def case_rope_interleaved() -> None:
    torch.manual_seed(1)
    batch, heads, seq, head_dim = 2, 8, 32, 64
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    theta = 10000.0 ** (-2.0 * torch.arange(0, head_dim // 2, dtype=torch.float32) / head_dim)
    positions = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(positions, theta)
    cos = freqs.cos()
    sin = freqs.sin()
    q_r = q.reshape(batch, heads, seq, head_dim // 2, 2)
    k_r = k.reshape(batch, heads, seq, head_dim // 2, 2)
    cos_e = cos[None, None, :, :, None]
    sin_e = sin[None, None, :, :, None]
    q_out = torch.stack(
        [q_r[..., 0] * cos_e[..., 0] - q_r[..., 1] * sin_e[..., 0],
         q_r[..., 1] * cos_e[..., 0] + q_r[..., 0] * sin_e[..., 0]],
        dim=-1,
    ).reshape(batch, heads, seq, head_dim)
    k_out = torch.stack(
        [k_r[..., 0] * cos_e[..., 0] - k_r[..., 1] * sin_e[..., 0],
         k_r[..., 1] * cos_e[..., 0] + k_r[..., 0] * sin_e[..., 0]],
        dim=-1,
    ).reshape(batch, heads, seq, head_dim)
    save_case(
        "rope_interleaved",
        {"q": q, "k": k, "cos": cos, "sin": sin, "q_out": q_out, "k_out": k_out},
    )


def case_yarn() -> None:
    torch.manual_seed(2)
    batch, heads, seq, head_dim = 2, 8, 64, 64
    original_max_pos = 4096
    scale = 4.0
    alpha, beta = 1.0, 32.0
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    base = 10000.0
    dim_range = torch.arange(0, head_dim // 2, dtype=torch.float32)
    freq_orig = 1.0 / (base ** (dim_range * 2.0 / head_dim))
    ramp = torch.clamp((dim_range / (head_dim // 2) * (beta - alpha) - alpha) / (beta - alpha), 0.0, 1.0)
    freq_interp = freq_orig / scale
    freq = freq_interp * (1.0 - ramp) + freq_orig * ramp
    positions = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(positions, freq)
    cos_half = freqs.cos()
    sin_half = freqs.sin()
    cos = cos_half.repeat(1, 2)[None, None, :, :]
    sin = sin_half.repeat(1, 2)[None, None, :, :]
    q_out, k_out = _rope_split(q, k, cos, sin)
    save_case(
        "yarn",
        {"q": q, "k": k, "cos": cos, "sin": sin, "q_out": q_out, "k_out": k_out},
    )


def case_longrope() -> None:
    torch.manual_seed(3)
    batch, heads, seq, head_dim = 2, 4, 32, 64
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    short_factor = torch.ones(head_dim // 2, dtype=torch.float32)
    long_factor = short_factor * 2.0
    base = 10000.0
    dim_range = torch.arange(0, head_dim // 2, dtype=torch.float32)
    freq = 1.0 / (base ** (dim_range * 2.0 / head_dim) * long_factor)
    positions = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(positions, freq)
    cos = freqs.cos().repeat(1, 2)[None, None, :, :]
    sin = freqs.sin().repeat(1, 2)[None, None, :, :]
    q_out, k_out = _rope_split(q, k, cos, sin)
    save_case(
        "longrope",
        {
            "q": q,
            "k": k,
            "long_factor": long_factor,
            "cos": cos,
            "sin": sin,
            "q_out": q_out,
            "k_out": k_out,
        },
    )


def case_alibi() -> None:
    heads = 8
    seq = 32
    slopes = 2.0 ** (-torch.arange(1, heads + 1, dtype=torch.float32) * 8.0 / heads)
    positions = torch.arange(seq, dtype=torch.float32)
    bias = -slopes[:, None, None] * (positions[None, :] - positions[:, None]).abs().unsqueeze(0)
    save_case("alibi", {"slopes": slopes, "bias": bias})


def case_sinusoidal() -> None:
    seq, dim = 128, 256
    positions = torch.arange(seq, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(0, dim, 2, dtype=torch.float32)
    div = torch.exp(dims * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(positions * div)
    pe[:, 1::2] = torch.cos(positions * div)
    save_case("sinusoidal", {"output": pe})


def case_t5_relative() -> None:
    heads = 8
    seq_q, seq_k = 16, 16
    num_buckets = 32
    max_distance = 128
    q_pos = torch.arange(seq_q, dtype=torch.int64)
    k_pos = torch.arange(seq_k, dtype=torch.int64)
    relative = k_pos[None, :] - q_pos[:, None]
    relative_buckets = torch.zeros_like(relative)
    num_buckets_half = num_buckets // 2
    relative_buckets = torch.where(
        relative > 0,
        relative_buckets + num_buckets_half,
        relative_buckets,
    )
    relative = relative.abs()
    max_exact = num_buckets_half // 2
    is_small = relative < max_exact
    relative_large = torch.clamp(
        (max_exact + (torch.log(relative.float().clamp(min=max_exact) / max_exact)
                      / math.log(max_distance / max_exact)
                      * (num_buckets_half - max_exact)).long()),
        max=num_buckets_half - 1,
    )
    relative_buckets = relative_buckets + torch.where(is_small, relative, relative_large)
    save_case("t5_relative", {"relative_buckets": relative_buckets.to(torch.int32)})


def main() -> None:
    print("Generating positional encoding reference data...")
    case_rope_split()
    case_rope_interleaved()
    case_yarn()
    case_longrope()
    case_alibi()
    case_sinusoidal()
    case_t5_relative()
    print("Done.")


if __name__ == "__main__":
    main()
