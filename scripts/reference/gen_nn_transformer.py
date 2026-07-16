"""Generate reference SafeTensors for synaptix-nn transformer block.

Run:
    python scripts/reference/gen_nn_transformer.py
"""

import math
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_transformer")


def save_case(name, tensors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def gelu_tanh(x):
    # exact matches synaptix gelu_tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def transformer_block_ref(
    x, n1_w, n1_b, q_w, k_w, v_w, o_w, n2_w, n2_b, fc1_w, fc1_b, fc2_w, fc2_b,
    num_heads, mask=None,
):
    b, s, h = x.shape
    head_dim = h // num_heads
    eps = 1e-5

    # pre-norm
    h1 = F.layer_norm(x, (h,), weight=n1_w, bias=n1_b, eps=eps)

    q = F.linear(h1, q_w)
    k = F.linear(h1, k_w)
    v = F.linear(h1, v_w)

    q = q.view(b, s, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D]
    k = k.view(b, s, num_heads, head_dim).transpose(1, 2)
    v = v.view(b, s, num_heads, head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = (q @ k.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    probs = F.softmax(scores, dim=-1)
    attn = probs @ v
    attn = attn.transpose(1, 2).contiguous().view(b, s, h)

    out_attn = F.linear(attn, o_w)
    x = x + out_attn

    h2 = F.layer_norm(x, (h,), weight=n2_w, bias=n2_b, eps=eps)
    ff = gelu_tanh(F.linear(h2, fc1_w, fc1_b))
    ff = F.linear(ff, fc2_w, fc2_b)
    return x + ff


def case_transformer_block():
    torch.manual_seed(500)
    b, s, h = 2, 6, 16
    num_heads = 4
    ffn = 32
    x = torch.randn(b, s, h)

    n1_w = torch.rand(h) + 0.5
    n1_b = torch.randn(h) * 0.1
    n2_w = torch.rand(h) + 0.5
    n2_b = torch.randn(h) * 0.1
    q_w = torch.randn(h, h) * 0.1
    k_w = torch.randn(h, h) * 0.1
    v_w = torch.randn(h, h) * 0.1
    o_w = torch.randn(h, h) * 0.1
    fc1_w = torch.randn(ffn, h) * 0.1
    fc1_b = torch.randn(ffn) * 0.05
    fc2_w = torch.randn(h, ffn) * 0.1
    fc2_b = torch.randn(h) * 0.05

    y = transformer_block_ref(
        x, n1_w, n1_b, q_w, k_w, v_w, o_w, n2_w, n2_b,
        fc1_w, fc1_b, fc2_w, fc2_b, num_heads,
    )
    save_case(
        "block",
        {
            "x": x,
            "n1_w": n1_w, "n1_b": n1_b,
            "q_w": q_w, "k_w": k_w, "v_w": v_w, "o_w": o_w,
            "n2_w": n2_w, "n2_b": n2_b,
            "fc1_w": fc1_w, "fc1_b": fc1_b,
            "fc2_w": fc2_w, "fc2_b": fc2_b,
            "output": y,
        },
    )


def case_transformer_encoder():
    torch.manual_seed(501)
    n_layers = 2
    b, s, h = 1, 4, 8
    num_heads = 2
    ffn = 16
    x = torch.randn(b, s, h)
    tensors = {"x": x}
    cur = x
    for layer in range(n_layers):
        n1_w = torch.rand(h) + 0.5
        n1_b = torch.randn(h) * 0.1
        n2_w = torch.rand(h) + 0.5
        n2_b = torch.randn(h) * 0.1
        q_w = torch.randn(h, h) * 0.1
        k_w = torch.randn(h, h) * 0.1
        v_w = torch.randn(h, h) * 0.1
        o_w = torch.randn(h, h) * 0.1
        fc1_w = torch.randn(ffn, h) * 0.1
        fc1_b = torch.randn(ffn) * 0.05
        fc2_w = torch.randn(h, ffn) * 0.1
        fc2_b = torch.randn(h) * 0.05
        cur = transformer_block_ref(
            cur, n1_w, n1_b, q_w, k_w, v_w, o_w, n2_w, n2_b,
            fc1_w, fc1_b, fc2_w, fc2_b, num_heads,
        )
        tensors[f"l{layer}_n1_w"] = n1_w
        tensors[f"l{layer}_n1_b"] = n1_b
        tensors[f"l{layer}_n2_w"] = n2_w
        tensors[f"l{layer}_n2_b"] = n2_b
        tensors[f"l{layer}_q_w"] = q_w
        tensors[f"l{layer}_k_w"] = k_w
        tensors[f"l{layer}_v_w"] = v_w
        tensors[f"l{layer}_o_w"] = o_w
        tensors[f"l{layer}_fc1_w"] = fc1_w
        tensors[f"l{layer}_fc1_b"] = fc1_b
        tensors[f"l{layer}_fc2_w"] = fc2_w
        tensors[f"l{layer}_fc2_b"] = fc2_b
    tensors["output"] = cur
    save_case("encoder_2layers", tensors)


def main():
    print("Generating nn-transformer reference data...")
    case_transformer_block()
    case_transformer_encoder()
    print("Done.")


if __name__ == "__main__":
    main()
