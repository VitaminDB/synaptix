"""Generate reference SafeTensors for synaptix-nn audio encoders."""

import math
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_audio")


def save_case(name, tensors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def transformer_block_ref(x, n1_w, n1_b, q_w, k_w, v_w, o_w, n2_w, n2_b, fc1_w, fc1_b, fc2_w, fc2_b, num_heads):
    b, s, h = x.shape
    head_dim = h // num_heads
    h1 = F.layer_norm(x, (h,), weight=n1_w, bias=n1_b, eps=1e-5)
    q = F.linear(h1, q_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    k = F.linear(h1, k_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    v = F.linear(h1, v_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    scale = 1.0 / math.sqrt(head_dim)
    attn = F.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v
    attn = attn.transpose(1, 2).contiguous().view(b, s, h)
    attn = F.linear(attn, o_w)
    x = x + attn
    h2 = F.layer_norm(x, (h,), weight=n2_w, bias=n2_b, eps=1e-5)
    mlp = F.linear(gelu_tanh(F.linear(h2, fc1_w, fc1_b)), fc2_w, fc2_b)
    return x + mlp


def case_whisper():
    torch.manual_seed(800)
    b, n_mels, t = 1, 4, 32
    d_model = 16
    num_heads = 4
    ffn = 32

    mel = torch.randn(b, n_mels, t)
    c1_w = torch.randn(d_model, n_mels, 3) * 0.1
    c1_b = torch.randn(d_model) * 0.05
    c2_w = torch.randn(d_model, d_model, 3) * 0.1
    c2_b = torch.randn(d_model) * 0.05

    x = F.conv1d(mel, c1_w, c1_b, stride=1, padding=1)
    x = gelu_tanh(x)
    x = F.conv1d(x, c2_w, c2_b, stride=2, padding=1)
    x = gelu_tanh(x)
    x = x.transpose(1, 2).contiguous()

    n1_w = torch.rand(d_model) + 0.5
    n1_b = torch.randn(d_model) * 0.1
    n2_w = torch.rand(d_model) + 0.5
    n2_b = torch.randn(d_model) * 0.1
    q_w = torch.randn(d_model, d_model) * 0.1
    k_w = torch.randn(d_model, d_model) * 0.1
    v_w = torch.randn(d_model, d_model) * 0.1
    o_w = torch.randn(d_model, d_model) * 0.1
    fc1_w = torch.randn(ffn, d_model) * 0.1
    fc1_b = torch.randn(ffn) * 0.05
    fc2_w = torch.randn(d_model, ffn) * 0.1
    fc2_b = torch.randn(d_model) * 0.05

    x = transformer_block_ref(x, n1_w, n1_b, q_w, k_w, v_w, o_w, n2_w, n2_b, fc1_w, fc1_b, fc2_w, fc2_b, num_heads)

    final_w = torch.rand(d_model) + 0.5
    final_b = torch.randn(d_model) * 0.1
    out = F.layer_norm(x, (d_model,), weight=final_w, bias=final_b, eps=1e-5)

    save_case(
        "whisper",
        {
            "mel": mel,
            "c1_w": c1_w, "c1_b": c1_b,
            "c2_w": c2_w, "c2_b": c2_b,
            "n1_w": n1_w, "n1_b": n1_b,
            "n2_w": n2_w, "n2_b": n2_b,
            "q_w": q_w, "k_w": k_w, "v_w": v_w, "o_w": o_w,
            "fc1_w": fc1_w, "fc1_b": fc1_b,
            "fc2_w": fc2_w, "fc2_b": fc2_b,
            "final_w": final_w, "final_b": final_b,
            "output": out,
        },
    )


def case_conformer():
    torch.manual_seed(801)
    b, s, h = 1, 8, 16
    num_heads = 4
    ffn = 32
    x = torch.randn(b, s, h)

    weights = {}
    def rand_w(name, shape, scale=0.1):
        t = torch.randn(*shape) * scale
        weights[name] = t
        return t

    ff1_n_w = torch.rand(h) + 0.5; weights["ff1_n_w"] = ff1_n_w
    ff1_n_b = torch.randn(h) * 0.1; weights["ff1_n_b"] = ff1_n_b
    ff1_in_w = rand_w("ff1_in_w", (ffn, h))
    ff1_in_b = rand_w("ff1_in_b", (ffn,), 0.05)
    ff1_out_w = rand_w("ff1_out_w", (h, ffn))
    ff1_out_b = rand_w("ff1_out_b", (h,), 0.05)

    attn_n_w = torch.rand(h) + 0.5; weights["attn_n_w"] = attn_n_w
    attn_n_b = torch.randn(h) * 0.1; weights["attn_n_b"] = attn_n_b
    q_w = rand_w("q_w", (h, h))
    k_w = rand_w("k_w", (h, h))
    v_w = rand_w("v_w", (h, h))
    o_w = rand_w("o_w", (h, h))

    ff2_n_w = torch.rand(h) + 0.5; weights["ff2_n_w"] = ff2_n_w
    ff2_n_b = torch.randn(h) * 0.1; weights["ff2_n_b"] = ff2_n_b
    ff2_in_w = rand_w("ff2_in_w", (ffn, h))
    ff2_in_b = rand_w("ff2_in_b", (ffn,), 0.05)
    ff2_out_w = rand_w("ff2_out_w", (h, ffn))
    ff2_out_b = rand_w("ff2_out_b", (h,), 0.05)

    final_n_w = torch.rand(h) + 0.5; weights["final_n_w"] = final_n_w
    final_n_b = torch.randn(h) * 0.1; weights["final_n_b"] = final_n_b

    # forward
    h1 = F.layer_norm(x, (h,), weight=ff1_n_w, bias=ff1_n_b, eps=1e-5)
    ff1 = F.linear(F.silu(F.linear(h1, ff1_in_w, ff1_in_b)), ff1_out_w, ff1_out_b)
    cur = x + 0.5 * ff1

    h2 = F.layer_norm(cur, (h,), weight=attn_n_w, bias=attn_n_b, eps=1e-5)
    head_dim = h // num_heads
    q = F.linear(h2, q_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    k = F.linear(h2, k_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    v = F.linear(h2, v_w).view(b, s, num_heads, head_dim).transpose(1, 2)
    scale = 1.0 / math.sqrt(head_dim)
    attn = F.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v
    attn = attn.transpose(1, 2).contiguous().view(b, s, h)
    attn = F.linear(attn, o_w)
    cur = cur + attn

    h3 = F.layer_norm(cur, (h,), weight=ff2_n_w, bias=ff2_n_b, eps=1e-5)
    ff2 = F.linear(F.silu(F.linear(h3, ff2_in_w, ff2_in_b)), ff2_out_w, ff2_out_b)
    cur = cur + 0.5 * ff2

    out = F.layer_norm(cur, (h,), weight=final_n_w, bias=final_n_b, eps=1e-5)
    weights["x"] = x
    weights["output"] = out
    save_case("conformer", weights)


def main():
    print("Generating nn-audio reference data...")
    case_whisper()
    case_conformer()
    print("Done.")


if __name__ == "__main__":
    main()
