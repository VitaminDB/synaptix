"""Generate reference SafeTensors for synaptix-nn ViT."""

import math
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_vit")


def save_case(name, tensors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def vit_block_ref(x, n1_w, n1_b, n2_w, n2_b, q_w, k_w, v_w, o_w, ff1_w, ff1_b, ff2_w, ff2_b, num_heads):
    b, s, h = x.shape
    head_dim = h // num_heads
    eps = 1e-6
    h1 = F.layer_norm(x, (h,), weight=n1_w, bias=n1_b, eps=eps)
    q = F.linear(h1, q_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    k = F.linear(h1, k_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    v = F.linear(h1, v_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    scale = 1.0 / math.sqrt(head_dim)
    attn = F.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v
    attn = attn.transpose(1, 2).contiguous().view(b, s, h)
    attn = F.linear(attn, o_w)
    x = x + attn

    h2 = F.layer_norm(x, (h,), weight=n2_w, bias=n2_b, eps=eps)
    mlp = F.linear(gelu_tanh(F.linear(h2, ff1_w, ff1_b)), ff2_w, ff2_b)
    return x + mlp


def case_vit():
    torch.manual_seed(700)
    b, c, h_img, w_img = 1, 3, 8, 8
    patch_size = 2
    hidden = 16
    num_heads = 4
    ffn = 32

    x = torch.randn(b, c, h_img, w_img)
    patch_w = torch.randn(hidden, patch_size * patch_size * c) * 0.1
    patch_b = torch.randn(hidden) * 0.05

    n1_w = torch.rand(hidden) + 0.5
    n1_b = torch.randn(hidden) * 0.1
    n2_w = torch.rand(hidden) + 0.5
    n2_b = torch.randn(hidden) * 0.1
    q_w = torch.randn(hidden, hidden) * 0.1
    k_w = torch.randn(hidden, hidden) * 0.1
    v_w = torch.randn(hidden, hidden) * 0.1
    o_w = torch.randn(hidden, hidden) * 0.1
    ff1_w = torch.randn(ffn, hidden) * 0.1
    ff1_b = torch.randn(ffn) * 0.05
    ff2_w = torch.randn(hidden, ffn) * 0.1
    ff2_b = torch.randn(hidden) * 0.05
    final_w = torch.rand(hidden) + 0.5
    final_b = torch.randn(hidden) * 0.1

    # Reference patchify
    p = patch_size
    nh = h_img // p
    nw = w_img // p
    reshaped = x.reshape(b, c, nh, p, nw, p)
    permuted = reshaped.permute(0, 2, 4, 1, 3, 5).contiguous()
    tokens = permuted.reshape(b, nh * nw, c * p * p)
    embedded = F.linear(tokens, patch_w, patch_b)

    block_out = vit_block_ref(
        embedded, n1_w, n1_b, n2_w, n2_b,
        q_w, k_w, v_w, o_w,
        ff1_w, ff1_b, ff2_w, ff2_b, num_heads,
    )
    final = F.layer_norm(block_out, (hidden,), weight=final_w, bias=final_b, eps=1e-6)

    save_case(
        "vit",
        {
            "x": x,
            "patch_w": patch_w, "patch_b": patch_b,
            "n1_w": n1_w, "n1_b": n1_b,
            "n2_w": n2_w, "n2_b": n2_b,
            "q_w": q_w, "k_w": k_w, "v_w": v_w, "o_w": o_w,
            "ff1_w": ff1_w, "ff1_b": ff1_b,
            "ff2_w": ff2_w, "ff2_b": ff2_b,
            "final_w": final_w, "final_b": final_b,
            "output": final,
        },
    )


def main():
    print("Generating nn-vit reference data...")
    case_vit()
    print("Done.")


if __name__ == "__main__":
    main()
