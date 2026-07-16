"""Reference SafeTensors для synaptix-nn/unet.

Run:
    scripts/reference/.venv/bin/python scripts/reference/gen_nn_unet.py

Покрытие — все 6 раннее заглушенных модулей:

- TimeEmbedding: sinusoidal HF-diffusers (flip_sin_to_cos=True, downscale_freq_shift=1)
  → fc1 → SiLU → fc2.
- ResNetBlock (Linear-stub): LN→silu→fc + time_emb → LN→silu→fc + shortcut.
- UNetAttnBlock: Pre-LN MHA + residual.
- UNetCrossAttnBlock: Pre-LN cross-attention (Q from x, K/V from context) + residual.
- UNet2d: conv_in → ResNet(t_emb) → attn → cross_attn → conv_out.
- UNet3d: conv_in → ResNet(t_emb) → temporal-attn → conv_out.
"""

import pathlib
import math
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_unet")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def sinusoidal_embedding(t: torch.Tensor, in_dim: int) -> torch.Tensor:
    half = in_dim // 2
    denom = max(half - 1, 1)
    exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / denom
    freqs = exponent.exp()
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.cos(), args.sin()], dim=-1)


def mha_split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, s, h = x.shape
    head_dim = h // num_heads
    return x.view(b, s, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def mha_merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, nh, s, hd = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(b, s, nh * hd)


def self_attention(x: torch.Tensor, qw, kw, vw, ow, num_heads: int) -> torch.Tensor:
    q = mha_split_heads(F.linear(x, qw), num_heads)
    k = mha_split_heads(F.linear(x, kw), num_heads)
    v = mha_split_heads(F.linear(x, vw), num_heads)
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-1, -2) * scale
    probs = F.softmax(scores, dim=-1)
    out = mha_merge_heads(probs @ v)
    return F.linear(out, ow)


def cross_attention(x, ctx, qw, kw, vw, ow, num_heads: int) -> torch.Tensor:
    q = mha_split_heads(F.linear(x, qw), num_heads)
    k = mha_split_heads(F.linear(ctx, kw), num_heads)
    v = mha_split_heads(F.linear(ctx, vw), num_heads)
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-1, -2) * scale
    probs = F.softmax(scores, dim=-1)
    out = mha_merge_heads(probs @ v)
    return F.linear(out, ow)


def case_time_embedding() -> None:
    torch.manual_seed(1000)
    in_dim, hidden, out_dim = 8, 16, 12
    fc1_w = torch.randn(hidden, in_dim) * 0.1
    fc1_b = torch.randn(hidden) * 0.05
    fc2_w = torch.randn(out_dim, hidden) * 0.1
    fc2_b = torch.randn(out_dim) * 0.05
    timesteps = torch.tensor([0.0, 50.0, 100.0, 999.0], dtype=torch.float32)
    emb = sinusoidal_embedding(timesteps, in_dim)
    h = F.linear(emb, fc1_w, fc1_b)
    h = F.silu(h)
    out = F.linear(h, fc2_w, fc2_b)
    save_case("time_embedding", {
        "fc1_w": fc1_w, "fc1_b": fc1_b,
        "fc2_w": fc2_w, "fc2_b": fc2_b,
        "timesteps": timesteps,
        "sin_emb": emb,
        "output": out,
    })


def case_sinusoidal_only() -> None:
    torch.manual_seed(1001)
    in_dim = 16
    timesteps = torch.linspace(0.0, 1000.0, 5)
    emb = sinusoidal_embedding(timesteps, in_dim)
    save_case("sinusoidal_only", {"timesteps": timesteps, "output": emb})


def case_resnet_block() -> None:
    torch.manual_seed(1002)
    b, t, in_ch, out_ch, te = 2, 5, 6, 8, 12
    eps = 1e-5

    n1w = torch.rand(in_ch) + 0.5
    n1b = torch.randn(in_ch) * 0.1
    c1w = torch.randn(out_ch, in_ch) * 0.1
    c1b = torch.randn(out_ch) * 0.05
    n2w = torch.rand(out_ch) + 0.5
    n2b = torch.randn(out_ch) * 0.1
    c2w = torch.randn(out_ch, out_ch) * 0.1
    c2b = torch.randn(out_ch) * 0.05
    tew = torch.randn(out_ch, te) * 0.1
    teb = torch.randn(out_ch) * 0.05
    sw = torch.randn(out_ch, in_ch) * 0.1

    x = torch.randn(b, t, in_ch)
    time_emb = torch.randn(b, te) * 0.5

    h = F.layer_norm(x, (in_ch,), n1w, n1b, eps)
    h = F.silu(h)
    h = F.linear(h, c1w, c1b)

    te1 = F.silu(time_emb)
    te1 = F.linear(te1, tew, teb).unsqueeze(1)
    h = h + te1

    h = F.layer_norm(h, (out_ch,), n2w, n2b, eps)
    h = F.silu(h)
    h = F.linear(h, c2w, c2b)

    skip = F.linear(x, sw)
    out = h + skip

    save_case("resnet_block", {
        "norm1_w": n1w, "norm1_b": n1b,
        "conv1_w": c1w, "conv1_b": c1b,
        "norm2_w": n2w, "norm2_b": n2b,
        "conv2_w": c2w, "conv2_b": c2b,
        "time_emb_proj_w": tew, "time_emb_proj_b": teb,
        "shortcut_w": sw,
        "x": x, "time_emb": time_emb, "output": out,
    })


def case_attn_block() -> None:
    torch.manual_seed(1003)
    b, t, h, nh = 2, 6, 8, 2
    eps = 1e-5
    nw = torch.rand(h) + 0.5
    nb = torch.randn(h) * 0.1
    qw = torch.randn(h, h) * 0.1
    kw = torch.randn(h, h) * 0.1
    vw = torch.randn(h, h) * 0.1
    ow = torch.randn(h, h) * 0.1
    x = torch.randn(b, t, h)

    hn = F.layer_norm(x, (h,), nw, nb, eps)
    out = self_attention(hn, qw, kw, vw, ow, nh)
    out = x + out
    save_case("attn_block", {
        "norm_w": nw, "norm_b": nb,
        "q_w": qw, "k_w": kw, "v_w": vw, "o_w": ow,
        "x": x, "output": out,
    })


def case_cross_attn_block() -> None:
    torch.manual_seed(1004)
    b, t, h, ctx_s, ctx_d, nh = 2, 4, 8, 6, 12, 2
    eps = 1e-5
    nw = torch.rand(h) + 0.5
    nb = torch.randn(h) * 0.1
    qw = torch.randn(h, h) * 0.1
    kw = torch.randn(h, ctx_d) * 0.1
    vw = torch.randn(h, ctx_d) * 0.1
    ow = torch.randn(h, h) * 0.1
    x = torch.randn(b, t, h)
    ctx = torch.randn(b, ctx_s, ctx_d)

    hn = F.layer_norm(x, (h,), nw, nb, eps)
    out = cross_attention(hn, ctx, qw, kw, vw, ow, nh)
    out = x + out
    save_case("cross_attn_block", {
        "norm_w": nw, "norm_b": nb,
        "q_w": qw, "k_w": kw, "v_w": vw, "o_w": ow,
        "x": x, "context": ctx, "output": out,
    })


def case_unet_2d() -> None:
    torch.manual_seed(1005)
    b, t, in_ch, out_ch, hidden, nh = 2, 4, 6, 6, 8, 2
    ctx_s, ctx_d = 5, 12
    time_in, time_hid = 8, 16
    eps = 1e-5

    conv_in_w = torch.randn(hidden, in_ch) * 0.1
    conv_in_b = torch.randn(hidden) * 0.05

    fc1_w = torch.randn(time_hid, time_in) * 0.1
    fc1_b = torch.randn(time_hid) * 0.05
    fc2_w = torch.randn(hidden, time_hid) * 0.1
    fc2_b = torch.randn(hidden) * 0.05

    # ResNet (square in/out = hidden)
    n1w = torch.rand(hidden) + 0.5
    n1b = torch.randn(hidden) * 0.1
    c1w = torch.randn(hidden, hidden) * 0.1
    c1b = torch.randn(hidden) * 0.05
    n2w = torch.rand(hidden) + 0.5
    n2b = torch.randn(hidden) * 0.1
    c2w = torch.randn(hidden, hidden) * 0.1
    c2b = torch.randn(hidden) * 0.05
    tew = torch.randn(hidden, hidden) * 0.1
    teb = torch.randn(hidden) * 0.05

    # Attn
    a_nw = torch.rand(hidden) + 0.5
    a_nb = torch.randn(hidden) * 0.1
    a_qw = torch.randn(hidden, hidden) * 0.1
    a_kw = torch.randn(hidden, hidden) * 0.1
    a_vw = torch.randn(hidden, hidden) * 0.1
    a_ow = torch.randn(hidden, hidden) * 0.1

    # Cross attn
    c_nw = torch.rand(hidden) + 0.5
    c_nb = torch.randn(hidden) * 0.1
    c_qw = torch.randn(hidden, hidden) * 0.1
    c_kw = torch.randn(hidden, ctx_d) * 0.1
    c_vw = torch.randn(hidden, ctx_d) * 0.1
    c_ow = torch.randn(hidden, hidden) * 0.1

    conv_out_w = torch.randn(out_ch, hidden) * 0.1
    conv_out_b = torch.randn(out_ch) * 0.05

    x = torch.randn(b, t, in_ch)
    timesteps = torch.tensor([20.0, 500.0], dtype=torch.float32)
    text_ctx = torch.randn(b, ctx_s, ctx_d)

    h = F.linear(x, conv_in_w, conv_in_b)

    emb = sinusoidal_embedding(timesteps, time_in)
    te = F.silu(F.linear(emb, fc1_w, fc1_b))
    te = F.linear(te, fc2_w, fc2_b)

    # ResNet
    r1 = F.silu(F.layer_norm(h, (hidden,), n1w, n1b, eps))
    r1 = F.linear(r1, c1w, c1b)
    t1 = F.linear(F.silu(te), tew, teb).unsqueeze(1)
    r1 = r1 + t1
    r2 = F.silu(F.layer_norm(r1, (hidden,), n2w, n2b, eps))
    r2 = F.linear(r2, c2w, c2b)
    h = r2 + h  # shortcut=identity (in=out)

    # Attn
    hn = F.layer_norm(h, (hidden,), a_nw, a_nb, eps)
    h = h + self_attention(hn, a_qw, a_kw, a_vw, a_ow, nh)

    # Cross attn
    hn = F.layer_norm(h, (hidden,), c_nw, c_nb, eps)
    h = h + cross_attention(hn, text_ctx, c_qw, c_kw, c_vw, c_ow, nh)

    out = F.linear(h, conv_out_w, conv_out_b)

    save_case("unet_2d", {
        "conv_in_w": conv_in_w, "conv_in_b": conv_in_b,
        "fc1_w": fc1_w, "fc1_b": fc1_b,
        "fc2_w": fc2_w, "fc2_b": fc2_b,
        "n1w": n1w, "n1b": n1b,
        "c1w": c1w, "c1b": c1b,
        "n2w": n2w, "n2b": n2b,
        "c2w": c2w, "c2b": c2b,
        "tew": tew, "teb": teb,
        "a_nw": a_nw, "a_nb": a_nb,
        "a_qw": a_qw, "a_kw": a_kw, "a_vw": a_vw, "a_ow": a_ow,
        "c_nw": c_nw, "c_nb": c_nb,
        "c_qw": c_qw, "c_kw": c_kw, "c_vw": c_vw, "c_ow": c_ow,
        "conv_out_w": conv_out_w, "conv_out_b": conv_out_b,
        "x": x, "timesteps": timesteps, "text_ctx": text_ctx,
        "output": out,
    })


def case_unet_3d() -> None:
    torch.manual_seed(1006)
    b, t_dim, s_dim, in_ch, out_ch, hidden, nh = 2, 3, 4, 6, 6, 8, 2
    time_in, time_hid = 8, 16
    eps = 1e-5

    conv_in_w = torch.randn(hidden, in_ch) * 0.1
    conv_in_b = torch.randn(hidden) * 0.05

    fc1_w = torch.randn(time_hid, time_in) * 0.1
    fc1_b = torch.randn(time_hid) * 0.05
    fc2_w = torch.randn(hidden, time_hid) * 0.1
    fc2_b = torch.randn(hidden) * 0.05

    n1w = torch.rand(hidden) + 0.5
    n1b = torch.randn(hidden) * 0.1
    c1w = torch.randn(hidden, hidden) * 0.1
    c1b = torch.randn(hidden) * 0.05
    n2w = torch.rand(hidden) + 0.5
    n2b = torch.randn(hidden) * 0.1
    c2w = torch.randn(hidden, hidden) * 0.1
    c2b = torch.randn(hidden) * 0.05
    tew = torch.randn(hidden, hidden) * 0.1
    teb = torch.randn(hidden) * 0.05

    a_nw = torch.rand(hidden) + 0.5
    a_nb = torch.randn(hidden) * 0.1
    a_qw = torch.randn(hidden, hidden) * 0.1
    a_kw = torch.randn(hidden, hidden) * 0.1
    a_vw = torch.randn(hidden, hidden) * 0.1
    a_ow = torch.randn(hidden, hidden) * 0.1

    conv_out_w = torch.randn(out_ch, hidden) * 0.1
    conv_out_b = torch.randn(out_ch) * 0.05

    x = torch.randn(b, t_dim, s_dim, in_ch)
    timesteps = torch.tensor([10.0, 200.0], dtype=torch.float32)

    x_flat = x.reshape(b, t_dim * s_dim, in_ch)
    h = F.linear(x_flat, conv_in_w, conv_in_b)

    emb = sinusoidal_embedding(timesteps, time_in)
    te = F.silu(F.linear(emb, fc1_w, fc1_b))
    te = F.linear(te, fc2_w, fc2_b)

    r1 = F.silu(F.layer_norm(h, (hidden,), n1w, n1b, eps))
    r1 = F.linear(r1, c1w, c1b)
    t1 = F.linear(F.silu(te), tew, teb).unsqueeze(1)
    r1 = r1 + t1
    r2 = F.silu(F.layer_norm(r1, (hidden,), n2w, n2b, eps))
    r2 = F.linear(r2, c2w, c2b)
    h = r2 + h

    hn = F.layer_norm(h, (hidden,), a_nw, a_nb, eps)
    h = h + self_attention(hn, a_qw, a_kw, a_vw, a_ow, nh)

    out_flat = F.linear(h, conv_out_w, conv_out_b)
    out = out_flat.reshape(b, t_dim, s_dim, out_ch)

    save_case("unet_3d", {
        "conv_in_w": conv_in_w, "conv_in_b": conv_in_b,
        "fc1_w": fc1_w, "fc1_b": fc1_b,
        "fc2_w": fc2_w, "fc2_b": fc2_b,
        "n1w": n1w, "n1b": n1b,
        "c1w": c1w, "c1b": c1b,
        "n2w": n2w, "n2b": n2b,
        "c2w": c2w, "c2b": c2b,
        "tew": tew, "teb": teb,
        "a_nw": a_nw, "a_nb": a_nb,
        "a_qw": a_qw, "a_kw": a_kw, "a_vw": a_vw, "a_ow": a_ow,
        "conv_out_w": conv_out_w, "conv_out_b": conv_out_b,
        "x": x, "timesteps": timesteps,
        "output": out,
    })


def main() -> None:
    print("Generating nn-unet reference data...")
    case_time_embedding()
    case_sinusoidal_only()
    case_resnet_block()
    case_attn_block()
    case_cross_attn_block()
    case_unet_2d()
    case_unet_3d()
    print("Done.")


if __name__ == "__main__":
    main()
