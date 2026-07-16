"""Reference SafeTensors для synaptix-nn/conformer.

Run:
    scripts/reference/.venv/bin/python scripts/reference/gen_nn_conformer.py

Reference воспроизводит три модуля Conformer (torchaudio.models.Conformer
схема, see https://pytorch.org/audio/main/generated/torchaudio.models.Conformer.html):

- FeedForwardModule (macaron half-step): LN → fc1 → SiLU → fc2, output =
  x + 0.5 · fc(LN(x)).
- ConvolutionModule: LN → pw1(2C) → GLU(dim=1) → depthwise(K, pad=(K-1)/2)
  → BatchNorm1d → SiLU → pw2(C), output = x + conv(LN(x)).
- SelfAttentionModule: LN → MHA + opt. additive rel-pos bias → out_proj,
  output = x + attn(LN(x)).
"""

import pathlib
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_conformer")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_ff_module():
    torch.manual_seed(900)
    h, ffn = 8, 16
    norm_w = torch.rand(h) + 0.5
    norm_b = torch.randn(h) * 0.1
    fc1_w = torch.randn(ffn, h) * 0.1
    fc1_b = torch.randn(ffn) * 0.05
    fc2_w = torch.randn(h, ffn) * 0.1
    fc2_b = torch.randn(h) * 0.05
    x = torch.randn(2, 5, h)

    normed = F.layer_norm(x, (h,), norm_w, norm_b, 1e-5)
    hh = F.linear(normed, fc1_w, fc1_b)
    hh = F.silu(hh)
    hh = F.linear(hh, fc2_w, fc2_b)
    output = x + 0.5 * hh

    save_case("ff_module", {
        "norm_w": norm_w, "norm_b": norm_b,
        "fc1_w": fc1_w, "fc1_b": fc1_b,
        "fc2_w": fc2_w, "fc2_b": fc2_b,
        "x": x, "output": output,
    })


def case_conv_module():
    torch.manual_seed(901)
    h, k = 4, 5
    norm_w = torch.rand(h) + 0.5
    norm_b = torch.randn(h) * 0.1
    # Conv1d weight: [C_out, C_in, K]. Pointwise → K=1.
    pw1_w = torch.randn(2 * h, h, 1) * 0.1
    dw_w = torch.randn(h, 1, k) * 0.1
    bn_mean = torch.randn(h) * 0.05
    bn_var = torch.rand(h) + 0.5
    bn_w = torch.rand(h) + 0.5
    bn_b = torch.randn(h) * 0.1
    pw2_w = torch.randn(h, h, 1) * 0.1
    x = torch.randn(2, 6, h)

    pad = (k - 1) // 2
    normed = F.layer_norm(x, (h,), norm_w, norm_b, 1e-5)
    hh = normed.permute(0, 2, 1)  # [B, C, S]
    hh = F.conv1d(hh, pw1_w)  # [B, 2C, S]
    hh = F.glu(hh, dim=1)  # [B, C, S]
    hh = F.conv1d(hh, dw_w, padding=pad, groups=h)  # depthwise
    hh = F.batch_norm(hh, bn_mean, bn_var, weight=bn_w, bias=bn_b, training=False, eps=1e-5)
    hh = F.silu(hh)
    hh = F.conv1d(hh, pw2_w)
    hh = hh.permute(0, 2, 1)  # [B, S, C]
    output = x + hh

    save_case("conv_module", {
        "norm_w": norm_w, "norm_b": norm_b,
        "pw1_w": pw1_w, "dw_w": dw_w,
        "bn_mean": bn_mean, "bn_var": bn_var, "bn_w": bn_w, "bn_b": bn_b,
        "pw2_w": pw2_w,
        "x": x, "output": output,
    })


def case_attention_module():
    """MHA без rel-pos bias (torchaudio Conformer baseline)."""
    torch.manual_seed(902)
    h, nh = 8, 2
    head_dim = h // nh
    norm_w = torch.rand(h) + 0.5
    norm_b = torch.randn(h) * 0.1
    q_w = torch.randn(h, h) * 0.1
    q_b = torch.randn(h) * 0.05
    k_w = torch.randn(h, h) * 0.1
    k_b = torch.randn(h) * 0.05
    v_w = torch.randn(h, h) * 0.1
    v_b = torch.randn(h) * 0.05
    o_w = torch.randn(h, h) * 0.1
    o_b = torch.randn(h) * 0.05
    x = torch.randn(2, 4, h)

    normed = F.layer_norm(x, (h,), norm_w, norm_b, 1e-5)
    b, s, _ = x.shape
    q = F.linear(normed, q_w, q_b).reshape(b, s, nh, head_dim).permute(0, 2, 1, 3)
    k = F.linear(normed, k_w, k_b).reshape(b, s, nh, head_dim).permute(0, 2, 1, 3)
    v = F.linear(normed, v_w, v_b).reshape(b, s, nh, head_dim).permute(0, 2, 1, 3)
    scale = 1.0 / (head_dim ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out_h = attn @ v
    merged = out_h.permute(0, 2, 1, 3).reshape(b, s, h)
    out = F.linear(merged, o_w, o_b)
    output = x + out

    save_case("attention_module", {
        "norm_w": norm_w, "norm_b": norm_b,
        "q_w": q_w, "q_b": q_b,
        "k_w": k_w, "k_b": k_b,
        "v_w": v_w, "v_b": v_b,
        "o_w": o_w, "o_b": o_b,
        "x": x, "output": output,
    })


def case_attention_module_relpos():
    """MHA с дополнительным additive rel-pos bias [num_heads, S, S]."""
    torch.manual_seed(903)
    h, nh = 8, 2
    head_dim = h // nh
    s_len = 4
    norm_w = torch.rand(h) + 0.5
    norm_b = torch.randn(h) * 0.1
    q_w = torch.randn(h, h) * 0.1
    q_b = torch.randn(h) * 0.05
    k_w = torch.randn(h, h) * 0.1
    k_b = torch.randn(h) * 0.05
    v_w = torch.randn(h, h) * 0.1
    v_b = torch.randn(h) * 0.05
    o_w = torch.randn(h, h) * 0.1
    o_b = torch.randn(h) * 0.05
    rel_bias = torch.randn(nh, s_len, s_len) * 0.5
    x = torch.randn(2, s_len, h)

    normed = F.layer_norm(x, (h,), norm_w, norm_b, 1e-5)
    b, s, _ = x.shape
    q = F.linear(normed, q_w, q_b).reshape(b, s, nh, head_dim).permute(0, 2, 1, 3)
    k = F.linear(normed, k_w, k_b).reshape(b, s, nh, head_dim).permute(0, 2, 1, 3)
    v = F.linear(normed, v_w, v_b).reshape(b, s, nh, head_dim).permute(0, 2, 1, 3)
    scale = 1.0 / (head_dim ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale
    scores = scores + rel_bias.unsqueeze(0)  # broadcast [1, nh, S, S]
    attn = torch.softmax(scores, dim=-1)
    out_h = attn @ v
    merged = out_h.permute(0, 2, 1, 3).reshape(b, s, h)
    out = F.linear(merged, o_w, o_b)
    output = x + out

    save_case("attention_module_relpos", {
        "norm_w": norm_w, "norm_b": norm_b,
        "q_w": q_w, "q_b": q_b,
        "k_w": k_w, "k_b": k_b,
        "v_w": v_w, "v_b": v_b,
        "o_w": o_w, "o_b": o_b,
        "rel_bias": rel_bias,
        "x": x, "output": output,
    })


def main():
    print("Generating nn-conformer reference data...")
    case_ff_module()
    case_conv_module()
    case_attention_module()
    case_attention_module_relpos()
    print("Done.")


if __name__ == "__main__":
    main()
