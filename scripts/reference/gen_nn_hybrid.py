"""Reference SafeTensors для synaptix-nn/hybrid (Mamba/Attention гибриды).

Run:
    scripts/reference/.venv/bin/python scripts/reference/gen_nn_hybrid.py

Все 6 моделей реализованы как Pre-LN + projection + residual stub'ы (полные
архитектуры — это многочасовая работа, здесь зацементирован публичный API +
минимальная семантика для bit-exact теста).

- FalconMamba: x + fc2(silu(fc1(LN(x))))
- GriffinBlock: x + fc_out(silu(a) ⊙ b), где [a, b] = split(fc_in(LN(x))) (SwiGLU).
- Hymba: x + fuse(cat([silu(attn_proj(LN(x))), silu(ssm_proj(LN(x)))]))
- Jamba: x + sum_e(softmax(gate(LN(x)))[e] · expert_e(LN(x)))
- Samba: x + sigmoid(window_gate) · fc_out(silu(a) ⊙ b), [a,b] = split(fc_in(LN(x)))
- Zamba: x + out_proj(silu(mamba(LN(x))) + shared_attn(LN(x)))
"""

import pathlib
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_hybrid")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_falcon_mamba() -> None:
    torch.manual_seed(1200)
    b, t, h = 2, 4, 8
    eps = 1e-5
    nw = torch.rand(h) + 0.5
    nb = torch.randn(h) * 0.1
    fc1_w = torch.randn(h, h) * 0.1
    fc1_b = torch.randn(h) * 0.05
    fc2_w = torch.randn(h, h) * 0.1
    fc2_b = torch.randn(h) * 0.05
    x = torch.randn(b, t, h)

    hn = F.layer_norm(x, (h,), nw, nb, eps)
    h1 = F.linear(hn, fc1_w, fc1_b)
    h1 = F.silu(h1)
    out = x + F.linear(h1, fc2_w, fc2_b)
    save_case("falcon_mamba", {
        "norm_w": nw, "norm_b": nb,
        "fc1_w": fc1_w, "fc1_b": fc1_b,
        "fc2_w": fc2_w, "fc2_b": fc2_b,
        "x": x, "output": out,
    })


def case_griffin_block() -> None:
    torch.manual_seed(1201)
    b, t, h = 2, 4, 8
    eps = 1e-5
    inner = h
    nw = torch.rand(h) + 0.5
    nb = torch.randn(h) * 0.1
    fc_in_w = torch.randn(inner * 2, h) * 0.1
    fc_in_b = torch.randn(inner * 2) * 0.05
    fc_out_w = torch.randn(h, inner) * 0.1
    fc_out_b = torch.randn(h) * 0.05
    x = torch.randn(b, t, h)

    hn = F.layer_norm(x, (h,), nw, nb, eps)
    ab = F.linear(hn, fc_in_w, fc_in_b)
    a, b_split = ab[..., :inner], ab[..., inner:]
    gated = F.silu(a) * b_split
    out = x + F.linear(gated, fc_out_w, fc_out_b)
    save_case("griffin_block", {
        "norm_w": nw, "norm_b": nb,
        "fc_in_w": fc_in_w, "fc_in_b": fc_in_b,
        "fc_out_w": fc_out_w, "fc_out_b": fc_out_b,
        "x": x, "output": out,
    })


def case_hymba() -> None:
    torch.manual_seed(1202)
    b, t, h = 2, 4, 8
    eps = 1e-5
    nw = torch.rand(h) + 0.5
    nb = torch.randn(h) * 0.1
    attn_w = torch.randn(h, h) * 0.1
    attn_b = torch.randn(h) * 0.05
    ssm_w = torch.randn(h, h) * 0.1
    ssm_b = torch.randn(h) * 0.05
    fuse_w = torch.randn(h, 2 * h) * 0.1
    fuse_b = torch.randn(h) * 0.05
    x = torch.randn(b, t, h)

    hn = F.layer_norm(x, (h,), nw, nb, eps)
    a = F.silu(F.linear(hn, attn_w, attn_b))
    s = F.silu(F.linear(hn, ssm_w, ssm_b))
    cat = torch.cat([a, s], dim=-1)
    out = x + F.linear(cat, fuse_w, fuse_b)
    save_case("hymba", {
        "norm_w": nw, "norm_b": nb,
        "attn_proj_w": attn_w, "attn_proj_b": attn_b,
        "ssm_proj_w": ssm_w, "ssm_proj_b": ssm_b,
        "fuse_w": fuse_w, "fuse_b": fuse_b,
        "x": x, "output": out,
    })


def case_jamba() -> None:
    torch.manual_seed(1203)
    b, t, h = 2, 4, 8
    eps = 1e-5
    nw = torch.rand(h) + 0.5
    nb = torch.randn(h) * 0.1
    gate_w = torch.randn(2, h) * 0.1
    e0_w = torch.randn(h, h) * 0.1
    e0_b = torch.randn(h) * 0.05
    e1_w = torch.randn(h, h) * 0.1
    e1_b = torch.randn(h) * 0.05
    x = torch.randn(b, t, h)

    hn = F.layer_norm(x, (h,), nw, nb, eps)
    g = F.linear(hn, gate_w)
    probs = F.softmax(g, dim=-1)
    w0 = probs[..., 0:1]
    w1 = probs[..., 1:2]
    e0 = F.linear(hn, e0_w, e0_b)
    e1 = F.linear(hn, e1_w, e1_b)
    blended = e0 * w0 + e1 * w1
    out = x + blended
    save_case("jamba", {
        "norm_w": nw, "norm_b": nb,
        "gate_w": gate_w,
        "expert0_w": e0_w, "expert0_b": e0_b,
        "expert1_w": e1_w, "expert1_b": e1_b,
        "x": x, "output": out,
    })


def case_samba() -> None:
    torch.manual_seed(1204)
    b, t, h = 2, 4, 8
    eps = 1e-5
    nw = torch.rand(h) + 0.5
    nb = torch.randn(h) * 0.1
    fc_in_w = torch.randn(h * 2, h) * 0.1
    fc_in_b = torch.randn(h * 2) * 0.05
    fc_out_w = torch.randn(h, h) * 0.1
    fc_out_b = torch.randn(h) * 0.05
    window_gate = torch.tensor([0.5])
    x = torch.randn(b, t, h)

    hn = F.layer_norm(x, (h,), nw, nb, eps)
    ab = F.linear(hn, fc_in_w, fc_in_b)
    a, b_split = ab[..., :h], ab[..., h:]
    gated = F.silu(a) * b_split
    out_proj = F.linear(gated, fc_out_w, fc_out_b)
    scale = torch.sigmoid(window_gate).item()
    out = x + scale * out_proj
    save_case("samba", {
        "norm_w": nw, "norm_b": nb,
        "fc_in_w": fc_in_w, "fc_in_b": fc_in_b,
        "fc_out_w": fc_out_w, "fc_out_b": fc_out_b,
        "window_gate": window_gate,
        "x": x, "output": out,
    })


def case_zamba() -> None:
    torch.manual_seed(1205)
    b, t, h = 2, 4, 8
    eps = 1e-5
    nw = torch.rand(h) + 0.5
    nb = torch.randn(h) * 0.1
    m_w = torch.randn(h, h) * 0.1
    m_b = torch.randn(h) * 0.05
    s_w = torch.randn(h, h) * 0.1
    s_b = torch.randn(h) * 0.05
    o_w = torch.randn(h, h) * 0.1
    o_b = torch.randn(h) * 0.05
    x = torch.randn(b, t, h)

    hn = F.layer_norm(x, (h,), nw, nb, eps)
    m = F.silu(F.linear(hn, m_w, m_b))
    s = F.linear(hn, s_w, s_b)
    sum_ms = m + s
    out = x + F.linear(sum_ms, o_w, o_b)
    save_case("zamba", {
        "norm_w": nw, "norm_b": nb,
        "mamba_w": m_w, "mamba_b": m_b,
        "shared_attn_w": s_w, "shared_attn_b": s_b,
        "out_w": o_w, "out_b": o_b,
        "x": x, "output": out,
    })


def main() -> None:
    print("Generating nn-hybrid reference data...")
    case_falcon_mamba()
    case_griffin_block()
    case_hymba()
    case_jamba()
    case_samba()
    case_zamba()
    print("Done.")


if __name__ == "__main__":
    main()
