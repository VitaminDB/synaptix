"""Reference SafeTensors для synaptix-nn/ssm_block (Mamba2Block + XLstmBlock).

Run:
    python scripts/reference/gen_nn_ssm_block.py

Outputs в tests/reference_data/nn_ssm_block/<case>.safetensors.

Reference implementation воспроизводит ту же step-loop математику, что и Rust
реализация (mamba_step + slstm_step + mlstm_step из synaptix-ops::ssm). Это
self-consistency check: тот же алгоритм, та же арифметика → bit-exact (atol=1e-5)
при F32.

Для канонического сравнения с mamba_ssm.modules.mamba2.Mamba2 / xlstm_pytorch
нужны pretrained веса (откладывается на Phase O).
"""

import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_ssm_block")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


# ── Mamba2Block reference (порт нашей Rust step-loop логики) ────────────────
def mamba2_forward(
    x, in_proj_w, conv_w, conv_b, out_proj_w,
    a_log, d_param, dt_bias, norm_w,
    hidden_size, d_state, num_heads, head_dim, d_conv, norm_eps,
):
    b, l, _ = x.shape
    d_inner = num_heads * head_dim

    projected = F.linear(x, in_proj_w)
    z = projected[..., 0:d_inner]
    x_proj = projected[..., d_inner:2*d_inner]
    b_mat = projected[..., 2*d_inner:2*d_inner+d_state]
    c_mat = projected[..., 2*d_inner+d_state:2*d_inner+2*d_state]
    dt_raw = projected[..., 2*d_inner+2*d_state:]

    # Causal conv1d (depthwise): pad по L слева, conv1d(groups=d_inner)
    x_conv_in = x_proj.permute(0, 2, 1).contiguous()  # [B, D_inner, L]
    pad_left = torch.zeros(b, d_inner, d_conv - 1, dtype=x.dtype)
    x_pad = torch.cat([pad_left, x_conv_in], dim=2)
    x_conv = F.conv1d(x_pad, conv_w, conv_b, stride=1, groups=d_inner)
    x_acted = F.silu(x_conv)
    x_seq = x_acted.permute(0, 2, 1).contiguous()  # [B, L, D_inner]

    dt_biased = dt_raw + dt_bias  # broadcast по B,L
    dt = F.softplus(dt_biased, beta=1.0, threshold=20.0)  # [B, L, num_heads]
    a_full = torch.exp(-a_log)  # [num_heads]
    # Expand: a_full → [D_inner, d_state]; D → [D_inner]
    a_expanded = a_full.repeat_interleave(head_dim).unsqueeze(1).expand(-1, d_state)
    d_expanded = d_param.repeat_interleave(head_dim)

    # SSM scan
    h = torch.zeros(b, d_inner, d_state, dtype=x.dtype)
    ys = []
    for t in range(l):
        xt = x_seq[:, t, :]
        bt = b_mat[:, t, :]
        ct = c_mat[:, t, :]
        dt_t = dt[:, t, :]
        dt_t_expanded = dt_t.repeat_interleave(head_dim, dim=1)  # [B, D_inner]
        # ZOH discretization
        dt_t3 = dt_t_expanded.unsqueeze(2)  # [B, D_inner, 1]
        a_bar = torch.exp(dt_t3 * a_expanded.unsqueeze(0))  # [B, D_inner, d_state]
        delta_b = dt_t3 * bt.unsqueeze(1)  # [B, D_inner, d_state]
        delta_b_u = delta_b * xt.unsqueeze(2)
        h = a_bar * h + delta_b_u
        y_ssm = (ct.unsqueeze(1) * h).sum(dim=2)  # [B, D_inner]
        y = y_ssm + d_expanded * xt  # skip
        ys.append(y.unsqueeze(1))
    y_seq = torch.cat(ys, dim=1)  # [B, L, D_inner]

    # rms_norm_gated (порт synaptix-ops::norm::rms_norm::rms_norm_gated):
    # n = y * silu(z); rms = sqrt(mean(n^2) + eps); out = n * weight / rms
    n = y_seq * F.silu(z)
    var = n.pow(2).mean(dim=-1, keepdim=True)
    rms_inv = (var + norm_eps).rsqrt()
    gated = n * rms_inv * norm_w
    out = F.linear(gated, out_proj_w)
    return out


def case_mamba2_block() -> None:
    torch.manual_seed(500)
    hidden_size, d_state = 8, 4
    num_heads, head_dim, d_conv = 2, 4, 3
    d_inner = num_heads * head_dim
    in_dim = 2 * d_inner + 2 * d_state + num_heads

    in_proj_w = torch.randn(in_dim, hidden_size)
    conv_w = torch.randn(d_inner, 1, d_conv) * 0.1
    conv_b = torch.randn(d_inner) * 0.01
    out_proj_w = torch.randn(hidden_size, d_inner) * 0.1
    a_log = torch.randn(num_heads).abs() + 0.5
    d_param = torch.randn(num_heads).abs() + 0.5
    dt_bias = torch.randn(num_heads) * 0.1
    norm_w = torch.rand(d_inner) + 0.5
    x = torch.randn(2, 4, hidden_size) * 0.5

    out = mamba2_forward(
        x, in_proj_w, conv_w, conv_b, out_proj_w,
        a_log, d_param, dt_bias, norm_w,
        hidden_size, d_state, num_heads, head_dim, d_conv, 1e-5,
    )
    save_case("mamba2_block", {
        "in_proj_w": in_proj_w,
        "conv_w": conv_w,
        "conv_b": conv_b,
        "out_proj_w": out_proj_w,
        "a_log": a_log,
        "d": d_param,
        "dt_bias": dt_bias,
        "norm_w": norm_w,
        "x": x,
        "output": out,
    })


# ── XLstmBlock sLSTM reference ──────────────────────────────────────────────
def slstm_forward(x, gate_w, gate_b, out_w, hidden_size):
    b, l, h = x.shape
    gates = F.linear(x, gate_w, gate_b)  # [B, L, 4H]

    state_h = torch.zeros(b, h, dtype=x.dtype)
    state_c = torch.zeros(b, h, dtype=x.dtype)
    ys = []
    for t in range(l):
        gate_t = gates[:, t, :]
        z = torch.tanh(gate_t[:, 0:h])
        i = torch.sigmoid(gate_t[:, h:2*h])
        f = torch.sigmoid(gate_t[:, 2*h:3*h])
        o = torch.sigmoid(gate_t[:, 3*h:4*h])
        c_new = f * state_c + i * z
        h_new = o * torch.tanh(c_new)
        state_h = h_new
        state_c = c_new
        ys.append(h_new.unsqueeze(1))
    y_seq = torch.cat(ys, dim=1)
    return F.linear(y_seq, out_w)


def mlstm_forward(x, gate_w, gate_b, out_w, hidden_size):
    b, l, h = x.shape
    gates = F.linear(x, gate_w, gate_b)  # [B, L, 3H]

    state_h = torch.zeros(b, h, dtype=x.dtype)
    state_c = torch.zeros(b, h * h, dtype=x.dtype)
    ys = []
    for t in range(l):
        gate_t = gates[:, t, :]
        q = gate_t[:, 0:h]
        k = gate_t[:, h:2*h]
        v = gate_t[:, 2*h:3*h]
        # C_new[b,i,j] = C[b,i,j] + v[b,i]*k[b,j]
        c_mat = state_c.view(b, h, h)
        c_new = c_mat + v.unsqueeze(2) * k.unsqueeze(1)
        n_new = state_h + k  # [B, H]
        den = (n_new * q).sum(dim=1, keepdim=True)  # [B, 1]
        denom = torch.clamp(den.abs(), min=1.0)
        num = (c_new * q.unsqueeze(1)).sum(dim=2)  # [B, H]
        out_step = num / denom
        state_h = out_step
        state_c = c_new.view(b, h * h)
        ys.append(out_step.unsqueeze(1))
    y_seq = torch.cat(ys, dim=1)
    return F.linear(y_seq, out_w)


def case_xlstm_slstm() -> None:
    torch.manual_seed(501)
    hidden_size = 4
    gate_w = torch.randn(4 * hidden_size, hidden_size) * 0.3
    gate_b = torch.randn(4 * hidden_size) * 0.2
    out_w = torch.randn(hidden_size, hidden_size) * 0.3
    x = torch.randn(2, 3, hidden_size) * 0.5
    out = slstm_forward(x, gate_w, gate_b, out_w, hidden_size)
    save_case("xlstm_slstm", {
        "gate_w": gate_w, "gate_b": gate_b, "out_w": out_w,
        "x": x, "output": out,
    })


def case_xlstm_mlstm() -> None:
    torch.manual_seed(502)
    hidden_size = 4
    gate_w = torch.randn(3 * hidden_size, hidden_size) * 0.3
    gate_b = torch.randn(3 * hidden_size) * 0.2
    out_w = torch.randn(hidden_size, hidden_size) * 0.3
    x = torch.randn(2, 3, hidden_size) * 0.5
    out = mlstm_forward(x, gate_w, gate_b, out_w, hidden_size)
    save_case("xlstm_mlstm", {
        "gate_w": gate_w, "gate_b": gate_b, "out_w": out_w,
        "x": x, "output": out,
    })


def main() -> None:
    print("Generating nn-ssm_block reference data...")
    case_mamba2_block()
    case_xlstm_slstm()
    case_xlstm_mlstm()
    print("Done.")


if __name__ == "__main__":
    main()
