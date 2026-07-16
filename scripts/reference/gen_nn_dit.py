"""Generate reference SafeTensors for synaptix-nn DiT block.

Run:
    python scripts/reference/gen_nn_dit.py
"""

import math
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_dit")


def save_case(name, tensors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def dit_block_ref(
    x, cond,
    q_w, k_w, v_w, o_w,
    ff1_w, ff1_b, ff2_w, ff2_b,
    adaln_w, adaln_b,
    num_heads,
):
    b, s, h = x.shape
    head_dim = h // num_heads
    eps = 1e-6

    mod_out = F.linear(F.silu(cond), adaln_w, adaln_b)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_out.chunk(6, dim=1)

    # attention branch
    h1 = F.layer_norm(x, (h,), eps=eps)
    h1 = modulate(h1, shift_msa, scale_msa)
    q = F.linear(h1, q_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    k = F.linear(h1, k_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    v = F.linear(h1, v_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    scale = 1.0 / math.sqrt(head_dim)
    attn = F.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v
    attn = attn.transpose(1, 2).contiguous().view(b, s, h)
    attn = F.linear(attn, o_w)
    x = x + gate_msa.unsqueeze(1) * attn

    # MLP branch
    h2 = F.layer_norm(x, (h,), eps=eps)
    h2 = modulate(h2, shift_mlp, scale_mlp)
    mlp_out = F.linear(gelu_tanh(F.linear(h2, ff1_w, ff1_b)), ff2_w, ff2_b)
    return x + gate_mlp.unsqueeze(1) * mlp_out


def case_dit_block():
    torch.manual_seed(600)
    b, s, h = 2, 8, 16
    num_heads = 4
    ffn = 32
    cond_dim = 12
    x = torch.randn(b, s, h)
    cond = torch.randn(b, cond_dim)

    q_w = torch.randn(h, h) * 0.1
    k_w = torch.randn(h, h) * 0.1
    v_w = torch.randn(h, h) * 0.1
    o_w = torch.randn(h, h) * 0.1
    ff1_w = torch.randn(ffn, h) * 0.1
    ff1_b = torch.randn(ffn) * 0.05
    ff2_w = torch.randn(h, ffn) * 0.1
    ff2_b = torch.randn(h) * 0.05
    adaln_w = torch.randn(6 * h, cond_dim) * 0.1
    adaln_b = torch.randn(6 * h) * 0.05

    y = dit_block_ref(
        x, cond,
        q_w, k_w, v_w, o_w,
        ff1_w, ff1_b, ff2_w, ff2_b,
        adaln_w, adaln_b, num_heads,
    )
    save_case(
        "block",
        {
            "x": x, "cond": cond,
            "q_w": q_w, "k_w": k_w, "v_w": v_w, "o_w": o_w,
            "ff1_w": ff1_w, "ff1_b": ff1_b,
            "ff2_w": ff2_w, "ff2_b": ff2_b,
            "adaln_w": adaln_w, "adaln_b": adaln_b,
            "output": y,
        },
    )


def case_final_layer():
    torch.manual_seed(601)
    b, s, h = 1, 16, 16
    cond_dim = 12
    patch_size = 2
    out_ch = 3
    out_dim = patch_size * patch_size * out_ch
    x = torch.randn(b, s, h)
    cond = torch.randn(b, cond_dim)

    linear_w = torch.randn(out_dim, h) * 0.1
    linear_b = torch.randn(out_dim) * 0.05
    adaln_w = torch.randn(2 * h, cond_dim) * 0.1
    adaln_b = torch.randn(2 * h) * 0.05

    mod_out = F.linear(F.silu(cond), adaln_w, adaln_b)
    shift, scale = mod_out.chunk(2, dim=1)
    hn = F.layer_norm(x, (h,), eps=1e-6)
    hn = modulate(hn, shift, scale)
    y = F.linear(hn, linear_w, linear_b)

    save_case(
        "final_layer",
        {
            "x": x, "cond": cond,
            "linear_w": linear_w, "linear_b": linear_b,
            "adaln_w": adaln_w, "adaln_b": adaln_b,
            "output": y,
        },
    )


def case_patchify():
    torch.manual_seed(602)
    b, c, h, w = 1, 3, 8, 8
    patch_size = 2
    hidden_size = 12
    x = torch.randn(b, c, h, w)

    # Reference: reshape into patches
    p = patch_size
    nh = h // p
    nw = w // p
    # [B, C, H, W] → [B, C, nh, p, nw, p]
    reshaped = x.reshape(b, c, nh, p, nw, p)
    # → [B, nh, nw, C, p, p]
    permuted = reshaped.permute(0, 2, 4, 1, 3, 5).contiguous()
    # → [B, N, C*p*p]
    tokens = permuted.reshape(b, nh * nw, c * p * p)

    weight = torch.randn(hidden_size, c * p * p) * 0.1
    bias = torch.randn(hidden_size) * 0.05
    out = F.linear(tokens, weight, bias)

    # Unpatchify reference: out of size [B, N, out_dim] where out_dim = p*p*c_out
    out_ch = 3
    out_dim = p * p * out_ch
    unproj_w = torch.randn(out_dim, hidden_size) * 0.1
    unproj_b = torch.randn(out_dim) * 0.05
    pre_unpatch = F.linear(out, unproj_w, unproj_b)
    # [B, N, p*p*c] → [B, c, H, W]
    r1 = pre_unpatch.reshape(b, nh, nw, out_ch, p, p)
    r2 = r1.permute(0, 3, 1, 4, 2, 5).contiguous()
    img = r2.reshape(b, out_ch, h, w)

    save_case(
        "patchify",
        {
            "x": x,
            "patch_weight": weight, "patch_bias": bias,
            "tokens_output": out,
            "unproj_w": unproj_w, "unproj_b": unproj_b,
            "pre_unpatch": pre_unpatch,
            "img_output": img,
        },
    )


def main():
    print("Generating nn-dit reference data...")
    case_dit_block()
    case_final_layer()
    case_patchify()
    print("Done.")


if __name__ == "__main__":
    main()
