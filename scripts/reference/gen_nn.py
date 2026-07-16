"""Generate reference SafeTensors for Session 5 — NN Module API.

Run:
    python scripts/reference/gen_nn.py

Covers: linear_forward, linear_no_bias, sequential_forward, transformer_block, lora_forward, lora_merge.
All module weights and inputs are saved so Rust can reproduce the exact forward pass.
Outputs data/ref/nn/<case>.safetensors.
"""

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_linear_forward() -> None:
    torch.manual_seed(0)
    batch, in_features, out_features = 4, 256, 512
    lin = nn.Linear(in_features, out_features, bias=True)
    x = torch.randn(batch, in_features, dtype=torch.float32)
    out = lin(x)
    save_case(
        "linear_forward",
        {"input": x, "weight": lin.weight.data, "bias": lin.bias.data, "output": out.detach()},
    )


def case_linear_no_bias() -> None:
    torch.manual_seed(1)
    batch, in_features, out_features = 4, 256, 512
    lin = nn.Linear(in_features, out_features, bias=False)
    x = torch.randn(batch, in_features, dtype=torch.float32)
    out = lin(x)
    save_case(
        "linear_no_bias",
        {"input": x, "weight": lin.weight.data, "output": out.detach()},
    )


def case_sequential_forward() -> None:
    torch.manual_seed(2)
    hidden = 256
    x = torch.randn(4, 16, hidden, dtype=torch.float32)
    lin1 = nn.Linear(hidden, hidden * 2)
    ln = nn.LayerNorm(hidden * 2, eps=1e-5)
    lin2 = nn.Linear(hidden * 2, hidden)
    with torch.no_grad():
        h = F.gelu(lin1(x), approximate="tanh")
        h = ln(h)
        out = lin2(h)
    save_case(
        "sequential_forward",
        {
            "input": x,
            "lin1_weight": lin1.weight.data,
            "lin1_bias": lin1.bias.data,
            "ln_weight": ln.weight.data,
            "ln_bias": ln.bias.data,
            "lin2_weight": lin2.weight.data,
            "lin2_bias": lin2.bias.data,
            "output": out,
        },
    )


def _make_transformer_block(hidden: int, heads: int, intermediate: int, eps: float):
    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden, eps=eps)
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, hidden, bias=False)
            self.v_proj = nn.Linear(hidden, hidden, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)
            self.norm2 = nn.LayerNorm(hidden, eps=eps)
            self.w_gate = nn.Linear(hidden, intermediate, bias=False)
            self.w_up = nn.Linear(hidden, intermediate, bias=False)
            self.w_down = nn.Linear(intermediate, hidden, bias=False)
            self.n_heads = heads
            self.head_dim = hidden // heads

        def forward(self, x):
            b, s, _ = x.shape
            r = self.norm1(x)
            q = self.q_proj(r).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(r).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(r).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(b, s, -1)
            x = x + self.o_proj(attn)
            r = self.norm2(x)
            ffn = self.w_down(F.silu(self.w_gate(r)) * self.w_up(r))
            return x + ffn

    return TransformerBlock()


def case_transformer_block() -> None:
    torch.manual_seed(3)
    hidden, heads, intermediate, seq = 256, 8, 512, 16
    model = _make_transformer_block(hidden, heads, intermediate, eps=1e-5)
    model.eval()
    x = torch.randn(2, seq, hidden, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    tensors = {"input": x, "output": out}
    for name, param in model.named_parameters():
        tensors[name.replace(".", "_")] = param.data
    save_case("transformer_block", tensors)


def case_lora_forward() -> None:
    torch.manual_seed(4)
    in_f, out_f, rank, alpha = 256, 512, 8, 16.0
    lin = nn.Linear(in_f, out_f, bias=False)
    lora_a = torch.randn(rank, in_f, dtype=torch.float32) * 0.02
    lora_b = torch.zeros(out_f, rank, dtype=torch.float32)
    x = torch.randn(4, in_f, dtype=torch.float32)
    scale = alpha / rank
    out = F.linear(x, lin.weight) + F.linear(F.linear(x, lora_a), lora_b) * scale
    save_case(
        "lora_forward",
        {
            "input": x,
            "base_weight": lin.weight.data,
            "lora_a": lora_a,
            "lora_b": lora_b,
            "scale": torch.tensor(scale),
            "output": out,
        },
    )


def case_lora_merge() -> None:
    torch.manual_seed(5)
    in_f, out_f, rank, alpha = 256, 512, 8, 16.0
    base_weight = torch.randn(out_f, in_f, dtype=torch.float32) * 0.02
    lora_a = torch.randn(rank, in_f, dtype=torch.float32) * 0.02
    lora_b = torch.randn(out_f, rank, dtype=torch.float32) * 0.001
    x = torch.randn(4, in_f, dtype=torch.float32)
    scale = alpha / rank
    merged_weight = base_weight + (lora_b @ lora_a) * scale
    out_unmerged = F.linear(x, base_weight) + F.linear(F.linear(x, lora_a), lora_b) * scale
    out_merged = F.linear(x, merged_weight)
    save_case(
        "lora_merge",
        {
            "input": x,
            "base_weight": base_weight,
            "lora_a": lora_a,
            "lora_b": lora_b,
            "scale": torch.tensor(scale),
            "merged_weight": merged_weight,
            "output_unmerged": out_unmerged,
            "output_merged": out_merged,
        },
    )


def main() -> None:
    print("Generating NN module reference data...")
    case_linear_forward()
    case_linear_no_bias()
    case_sequential_forward()
    case_transformer_block()
    case_lora_forward()
    case_lora_merge()
    print("Done.")


if __name__ == "__main__":
    main()
