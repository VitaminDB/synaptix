"""Reference SafeTensors для synaptix-nn/multimodal.

Run:
    python scripts/reference/gen_nn_multimodal.py

Reference воспроизводит:
- MlpProjector: fc1 → F.gelu(approximate="none") → fc2 (LLaVA/Qwen-VL pattern).
- CrossModalAttention: multi-head Q from x, K/V from context.
- VlmBlock: layer_norm(x) → cross-attn(fused KV) → +skip.

Полные модели (transformers.Blip2QFormerModel, перцивер из openflamingo) —
Phase O.
"""

import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_multimodal")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_mlp_projector():
    torch.manual_seed(800)
    in_dim, hidden_dim, out_dim = 8, 16, 12
    fc1_w = torch.randn(hidden_dim, in_dim) * 0.1
    fc1_b = torch.randn(hidden_dim) * 0.05
    fc2_w = torch.randn(out_dim, hidden_dim) * 0.1
    fc2_b = torch.randn(out_dim) * 0.05
    x = torch.randn(2, 5, in_dim)
    h = F.linear(x, fc1_w, fc1_b)
    h = F.gelu(h, approximate="none")
    out = F.linear(h, fc2_w, fc2_b)
    save_case("mlp_projector", {
        "fc1_w": fc1_w, "fc1_b": fc1_b,
        "fc2_w": fc2_w, "fc2_b": fc2_b,
        "x": x, "output": out,
    })


def cross_modal_attn(x, context, q_w, k_w, v_w, o_w, num_heads):
    b, sq, query_dim = x.shape
    sk = context.shape[1]
    head_dim = query_dim // num_heads
    q = F.linear(x, q_w).reshape(b, sq, num_heads, head_dim).permute(0, 2, 1, 3)
    k = F.linear(context, k_w).reshape(b, sk, num_heads, head_dim).permute(0, 2, 1, 3)
    v = F.linear(context, v_w).reshape(b, sk, num_heads, head_dim).permute(0, 2, 1, 3)
    scale = 1.0 / (head_dim ** 0.5)
    attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
    out_h = attn @ v
    merged = out_h.permute(0, 2, 1, 3).reshape(b, sq, query_dim)
    return F.linear(merged, o_w)


def case_cross_modal_attention():
    torch.manual_seed(801)
    query_dim, context_dim, num_heads = 8, 16, 2
    q_w = torch.randn(query_dim, query_dim) * 0.1
    k_w = torch.randn(query_dim, context_dim) * 0.1
    v_w = torch.randn(query_dim, context_dim) * 0.1
    o_w = torch.randn(query_dim, query_dim) * 0.1
    x = torch.randn(2, 4, query_dim)
    context = torch.randn(2, 6, context_dim)
    out = cross_modal_attn(x, context, q_w, k_w, v_w, o_w, num_heads)
    save_case("cross_modal_attention", {
        "q_w": q_w, "k_w": k_w, "v_w": v_w, "o_w": o_w,
        "x": x, "context": context, "output": out,
    })


def case_vlm_block():
    """VlmBlock = layer_norm(x) → cross-attn(fused KV) → +skip."""
    torch.manual_seed(802)
    hidden_size, context_dim, num_heads = 8, 16, 2
    norm_w = torch.rand(hidden_size) + 0.5
    norm_b = torch.randn(hidden_size) * 0.1
    q_w = torch.randn(hidden_size, hidden_size) * 0.1
    kv_w = torch.randn(hidden_size * 2, context_dim) * 0.1
    o_w = torch.randn(hidden_size, hidden_size) * 0.1
    x = torch.randn(2, 4, hidden_size)
    context = torch.randn(2, 6, context_dim)

    normed = F.layer_norm(x, (hidden_size,), norm_w, norm_b, 1e-5)
    b, sq, _ = x.shape
    sk = context.shape[1]
    head_dim = hidden_size // num_heads
    q = F.linear(normed, q_w).reshape(b, sq, num_heads, head_dim).permute(0, 2, 1, 3)
    kv = F.linear(context, kv_w)
    k = kv[..., :hidden_size].reshape(b, sk, num_heads, head_dim).permute(0, 2, 1, 3)
    v = kv[..., hidden_size:].reshape(b, sk, num_heads, head_dim).permute(0, 2, 1, 3)
    scale = 1.0 / (head_dim ** 0.5)
    attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
    out_h = attn @ v
    merged = out_h.permute(0, 2, 1, 3).reshape(b, sq, hidden_size)
    attn_out = F.linear(merged, o_w)
    output = x + attn_out

    save_case("vlm_block", {
        "norm_w": norm_w, "norm_b": norm_b,
        "q_w": q_w, "kv_w": kv_w, "o_w": o_w,
        "x": x, "context": context, "output": output,
    })


def main():
    print("Generating nn-multimodal reference data...")
    case_mlp_projector()
    case_cross_modal_attention()
    case_vlm_block()
    print("Done.")


if __name__ == "__main__":
    main()
