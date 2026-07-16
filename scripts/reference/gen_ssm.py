"""Generate reference SafeTensors for SSM ops (Mamba scan, RWKV-4 WKV).

Run:
    python scripts/reference/gen_ssm.py

Outputs tests/reference_data/ssm/<case>.safetensors.
"""

import pathlib

import torch
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/ssm")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_mamba_scan() -> None:
    torch.manual_seed(200)
    b, l, d_inner, n = 2, 6, 8, 4
    x = torch.randn(b, l, d_inner)
    a_disc = -0.5 * torch.rand(d_inner, n) - 0.1  # already discrete (negative)
    bb = torch.randn(b, l, n)
    cc = torch.randn(b, l, n)
    d = torch.randn(d_inner)
    h = torch.zeros(b, d_inner, n)
    ys = []
    for t in range(l):
        xt = x[:, t]  # [B, D]
        bt = bb[:, t]  # [B, N]
        ct = cc[:, t]  # [B, N]
        # h_new = a_disc * h + b_x where b_x[b,d,n] = b[b,n] * x[b,d]
        b_x = bt.unsqueeze(1) * xt.unsqueeze(2)  # [B, D, N]
        h = a_disc.unsqueeze(0) * h + b_x
        y_ssm = (ct.unsqueeze(1) * h).sum(dim=-1)  # [B, D]
        y = y_ssm + d.unsqueeze(0) * xt
        ys.append(y)
    out = torch.stack(ys, dim=1)  # [B, L, D]
    save_case(
        "mamba_scan",
        {"x": x, "a": a_disc, "b": bb, "c": cc, "d": d, "output": out},
    )


def case_mamba_step() -> None:
    torch.manual_seed(201)
    b, d_inner, n = 2, 8, 4
    x = torch.randn(b, d_inner)
    a = torch.randn(d_inner, n)
    bb = torch.randn(b, n)
    cc = torch.randn(b, n)
    dt = torch.rand(b, d_inner) + 0.1  # positive timestep
    h_in = torch.randn(b, d_inner, n)

    # ZOH discretization: a_bar = exp(dt * a), delta_B = dt * b
    dt3 = dt.unsqueeze(2)  # [B, D, 1]
    a_bar = (dt3 * a.unsqueeze(0)).exp()  # [B, D, N]
    delta_b = dt3 * bb.unsqueeze(1)  # [B, D, N]
    delta_b_u = delta_b * x.unsqueeze(2)  # [B, D, N]
    h_new = a_bar * h_in + delta_b_u
    y = (cc.unsqueeze(1) * h_new).sum(dim=-1)  # [B, D]
    save_case(
        "mamba_step",
        {
            "x": x, "a": a, "b": bb, "c": cc, "dt": dt,
            "h_in": h_in, "h_out": h_new, "y": y,
        },
    )


def case_rwkv_wkv() -> None:
    torch.manual_seed(202)
    b, l, d = 2, 6, 4
    k = torch.randn(b, l, d)
    v = torch.randn(b, l, d)
    r = torch.randn(b, l, d)
    time_decay = torch.randn(d) * 0.1  # log-decay free param
    time_first = torch.randn(d) * 0.1

    w_neg = -time_decay.exp()  # negative scalar log-decay
    aa = torch.zeros(b, d)
    bb_state = torch.zeros(b, d)
    pp = torch.full((b, d), -1.0e30)
    out = torch.zeros(b, l, d)
    for t in range(l):
        kt = k[:, t]
        vt = v[:, t]
        rt = r[:, t]
        ww_out = time_first + kt
        p_out = torch.maximum(pp, ww_out)
        e1_out = (pp - p_out).exp()
        e2_out = (ww_out - p_out).exp()
        num = e1_out * aa + e2_out * vt
        den = e1_out * bb_state + e2_out
        wkv = num / den
        out[:, t] = rt.sigmoid() * wkv

        ww_st = pp + w_neg
        p_st = torch.maximum(ww_st, kt)
        e1_st = (ww_st - p_st).exp()
        e2_st = (kt - p_st).exp()
        aa = e1_st * aa + e2_st * vt
        bb_state = e1_st * bb_state + e2_st
        pp = p_st

    save_case(
        "rwkv_wkv",
        {
            "k": k, "v": v, "r": r,
            "time_decay": time_decay, "time_first": time_first,
            "output": out,
        },
    )


def case_rwkv_time_mix() -> None:
    torch.manual_seed(203)
    b, d = 2, 8
    x = torch.randn(b, d)
    x_prev = torch.randn(b, d)
    mix_k = torch.rand(d)
    mix_v = torch.rand(d)
    mix_r = torch.rand(d)
    xk = x * mix_k + x_prev * (1 - mix_k)
    xv = x * mix_v + x_prev * (1 - mix_v)
    xr = x * mix_r + x_prev * (1 - mix_r)
    packed = torch.cat([xk, xv, xr], dim=-1)
    save_case(
        "rwkv_time_mix",
        {
            "x": x, "x_prev": x_prev,
            "mix_k": mix_k, "mix_v": mix_v, "mix_r": mix_r,
            "output": packed,
        },
    )


# ───────────────────────── расширенное SSM-семейство ─────────────────────────
# Все эталоны воспроизводят раскладку и порядок обновления Rust-реализаций
# в crates/synaptix-ops/src/ssm/.


def case_s4() -> None:
    torch.manual_seed(210)
    b, l, d, n = 2, 6, 4, 3
    x = torch.randn(b, l, d)
    a = 0.5 * torch.rand(d, n) + 0.3  # стабильный положительный decay (0.3..0.8)
    bb = torch.randn(d, n)
    cc = torch.randn(d, n)
    out = torch.zeros(b, l, d)
    h = torch.zeros(b, d, n)
    for t in range(l):
        xt = x[:, t]  # [b,d]
        h = a.unsqueeze(0) * h + bb.unsqueeze(0) * xt.unsqueeze(2)  # [b,d,n]
        out[:, t] = (cc.unsqueeze(0) * h).sum(dim=-1)
    save_case("s4", {"x": x, "a": a, "b": bb, "c": cc, "output": out})


def case_s5() -> None:
    torch.manual_seed(211)
    b, l, hsz, n = 2, 6, 4, 3
    x = torch.randn(b, l, hsz)
    lam = 0.5 * torch.rand(n) + 0.3
    bb = torch.randn(n, hsz)
    cc = torch.randn(hsz, n)
    d = torch.randn(hsz)
    out = torch.zeros(b, l, hsz)
    h = torch.zeros(b, n)
    for t in range(l):
        xt = x[:, t]  # [b,hsz]
        h = lam.unsqueeze(0) * h + xt @ bb.t()  # [b,n]
        out[:, t] = h @ cc.t() + d.unsqueeze(0) * xt
    save_case("s5", {"x": x, "lambda": lam, "b": bb, "c": cc, "d": d, "output": out})


def case_h3() -> None:
    torch.manual_seed(212)
    b, l, d = 2, 6, 4
    x = torch.randn(b, l, d)
    k = torch.randn(b, l, d)
    q = torch.randn(b, l, d)
    a = 0.5 * torch.rand(d) + 0.3
    out = torch.zeros(b, l, d)
    s = torch.zeros(b, d)
    for t in range(l):
        u = k[:, t] * x[:, t]
        s = a.unsqueeze(0) * s + u
        out[:, t] = q[:, t] * s
    save_case("h3", {"x": x, "k": k, "q": q, "a": a, "output": out})


def case_ttt() -> None:
    torch.manual_seed(213)
    b, l, d = 2, 5, 4
    lr = 0.1
    x = torch.randn(b, l, d)
    w = torch.randn(d, d) * 0.1
    out = torch.zeros(b, l, d)
    for bi in range(b):
        wmat = w.clone()
        for t in range(l):
            xt = x[bi, t]
            pred = wmat @ xt
            err = pred - xt
            wmat = wmat - lr * torch.outer(err, xt)
            out[bi, t] = wmat @ xt
    save_case("ttt", {"x": x, "w": w, "output": out})


def case_liquid() -> None:
    torch.manual_seed(214)
    b, d = 2, 4
    x = torch.randn(b, d)
    state = torch.randn(b, d)
    tau = torch.rand(d) + 1.0  # > 1, чтобы 1/tau < 1
    out = state + (1.0 / tau).unsqueeze(0) * (x.tanh() - state)
    save_case("liquid", {"x": x, "state": state, "tau": tau, "output": out})


def case_titans() -> None:
    torch.manual_seed(215)
    b, d = 2, 4
    x = torch.randn(b, d)
    mem = torch.randn(b, d)
    surprise = torch.randn(b, d)
    r = torch.sigmoid(x)
    out = r * mem + (1.0 - r) * surprise
    save_case("titans", {"x": x, "mem": mem, "surprise": surprise, "output": out})


def case_slstm() -> None:
    torch.manual_seed(216)
    b, d = 2, 4
    x = torch.randn(b, 4 * d)  # [z, i, f, o]
    h = torch.randn(b, d)
    c = torch.randn(b, d)
    z = x[:, :d].tanh()
    i = torch.sigmoid(x[:, d : 2 * d])
    f = torch.sigmoid(x[:, 2 * d : 3 * d])
    o = torch.sigmoid(x[:, 3 * d : 4 * d])
    c_new = f * c + i * z
    out = o * c_new.tanh()
    save_case("slstm", {"x": x, "h": h, "c": c, "output": out})


def case_mlstm() -> None:
    torch.manual_seed(217)
    b, d = 2, 4
    x = torch.randn(b, 3 * d)  # [q, k, v]
    h = torch.randn(b, d)  # normalizer n
    c = torch.randn(b, d * d)  # matrix state
    q = x[:, :d]
    k = x[:, d : 2 * d]
    v = x[:, 2 * d : 3 * d]
    cmat = c.reshape(b, d, d)
    c_new = cmat + v.unsqueeze(2) * k.unsqueeze(1)  # [b,d,d]
    n_new = h + k
    num = (c_new * q.unsqueeze(1)).sum(dim=-1)  # [b,d]
    den = (n_new * q).sum(dim=-1)  # [b]
    denom = den.abs().clamp(min=1.0)
    out = num / denom.unsqueeze(1)
    save_case("mlstm", {"x": x, "h": h, "c": c, "output": out})


def case_monarch() -> None:
    torch.manual_seed(218)
    b, l, d1, d2 = 2, 3, 3, 4
    d = d1 * d2
    x = torch.randn(b, l, d)
    m1 = torch.randn(d1, d1)
    m2 = torch.randn(d2, d2)
    X = x.reshape(b, l, d1, d2)
    y = torch.einsum("ip,blpq,jq->blij", m1, X, m2)
    out = y.reshape(b, l, d)
    save_case("monarch", {"x": x, "m1": m1, "m2": m2, "output": out})


def main() -> None:
    print("Generating SSM reference data...")
    case_mamba_scan()
    case_mamba_step()
    case_rwkv_wkv()
    case_rwkv_time_mix()
    # расширенное семейство
    case_s4()
    case_s5()
    case_h3()
    case_ttt()
    case_liquid()
    case_titans()
    case_slstm()
    case_mlstm()
    case_monarch()
    print("Done.")


if __name__ == "__main__":
    main()
