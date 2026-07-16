"""Generate reference SafeTensors for Session 4 — Attention variants.

Run:
    python scripts/reference/gen_attention.py

Uses F.scaled_dot_product_attention as the ground truth.
Outputs data/ref/attention/<case>.safetensors.
"""

import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/attention")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_scaled_dot_no_mask() -> None:
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 2, 8, 32, 64
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    v = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
    save_case("scaled_dot_no_mask", {"q": q, "k": k, "v": v, "output": out})


def case_scaled_dot_causal() -> None:
    torch.manual_seed(1)
    batch, heads, seq, head_dim = 2, 8, 32, 64
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    v = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
    save_case("scaled_dot_causal", {"q": q, "k": k, "v": v, "output": out})


def case_gqa() -> None:
    torch.manual_seed(2)
    batch, q_heads, kv_heads, seq, head_dim = 2, 8, 2, 32, 64
    q = torch.randn(batch, q_heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, kv_heads, seq, head_dim, dtype=torch.float32)
    v = torch.randn(batch, kv_heads, seq, head_dim, dtype=torch.float32)
    repeat = q_heads // kv_heads
    k_exp = k.repeat_interleave(repeat, dim=1)
    v_exp = v.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
    save_case("gqa", {"q": q, "k": k, "v": v, "output": out})


def case_mqa() -> None:
    torch.manual_seed(3)
    batch, q_heads, seq, head_dim = 2, 8, 32, 64
    q = torch.randn(batch, q_heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, 1, seq, head_dim, dtype=torch.float32)
    v = torch.randn(batch, 1, seq, head_dim, dtype=torch.float32)
    k_exp = k.expand(batch, q_heads, seq, head_dim)
    v_exp = v.expand(batch, q_heads, seq, head_dim)
    out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
    save_case("mqa", {"q": q, "k": k, "v": v, "output": out})


def case_sliding_window() -> None:
    torch.manual_seed(4)
    batch, heads, seq, head_dim = 2, 4, 32, 64
    window = 8
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    v = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    mask = torch.full((seq, seq), float("-inf"), dtype=torch.float32)
    for i in range(seq):
        lo = max(0, i - window + 1)
        mask[i, lo : i + 1] = 0.0
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    save_case(
        "sliding_window",
        {"q": q, "k": k, "v": v, "mask": mask, "output": out},
    )


def case_cross_attention() -> None:
    torch.manual_seed(5)
    batch, heads, q_seq, kv_seq, head_dim = 2, 8, 16, 32, 64
    q = torch.randn(batch, heads, q_seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, kv_seq, head_dim, dtype=torch.float32)
    v = torch.randn(batch, heads, kv_seq, head_dim, dtype=torch.float32)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
    save_case("cross_attention", {"q": q, "k": k, "v": v, "output": out})


def main() -> None:
    print("Generating attention reference data...")
    case_scaled_dot_no_mask()
    case_scaled_dot_causal()
    case_gqa()
    case_mqa()
    case_sliding_window()
    case_cross_attention()
    print("Done.")


if __name__ == "__main__":
    main()
