"""Generate reference SafeTensors for advanced attention variants.

Run:
    python scripts/reference/gen_attention_advanced.py

Outputs data/ref/attention_advanced/<case>.safetensors.
"""

import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/attention_advanced")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_mla() -> None:
    torch.manual_seed(100)
    b, h, s, d_nope, d_rope, d_v = 2, 4, 8, 32, 16, 32
    q_nope = torch.randn(b, h, s, d_nope)
    q_rope = torch.randn(b, h, s, d_rope)
    k_nope = torch.randn(b, h, s, d_nope)
    k_rope = torch.randn(b, 1, s, d_rope)
    v = torch.randn(b, h, s, d_v)
    scale = 1.0 / ((d_nope + d_rope) ** 0.5)
    k_rope_exp = k_rope.expand(b, h, s, d_rope)
    scores = (
        q_nope @ k_nope.transpose(-1, -2) + q_rope @ k_rope_exp.transpose(-1, -2)
    ) * scale
    probs = F.softmax(scores, dim=-1)
    out = probs @ v
    save_case(
        "mla",
        {
            "q_nope": q_nope, "q_rope": q_rope,
            "k_nope": k_nope, "k_rope": k_rope,
            "v": v, "output": out,
        },
    )


def case_lightning_no_causal() -> None:
    torch.manual_seed(101)
    b, h, s, dk, dv = 2, 4, 16, 32, 32
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    # no causal, no slope
    s_full = torch.einsum("bhsd,bhsv->bhdv", k, v)
    out = torch.einsum("bhsd,bhdv->bhsv", q, s_full)
    save_case("lightning_no_causal", {"q": q, "k": k, "v": v, "output": out})


def case_lightning_causal() -> None:
    torch.manual_seed(102)
    b, h, s, dk, dv = 2, 4, 16, 32, 32
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    slope = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out = torch.zeros(b, h, s, dv)
    for bi in range(b):
        for hi in range(h):
            lam = slope[hi].item()
            decay = float(torch.tensor(-lam).exp())
            state = torch.zeros(dk, dv)
            for t in range(s):
                state = state * decay
                state = state + torch.outer(k[bi, hi, t], v[bi, hi, t])
                out[bi, hi, t] = q[bi, hi, t] @ state
    save_case(
        "lightning_causal",
        {"q": q, "k": k, "v": v, "slope": slope, "output": out},
    )


def case_ring() -> None:
    torch.manual_seed(103)
    b, h, s, d = 2, 4, 16, 32
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)
    # standard SDPA (no causal)
    scores = (q @ k.transpose(-1, -2)) * scale
    probs = F.softmax(scores, dim=-1)
    out = probs @ v
    save_case("ring_no_causal", {"q": q, "k": k, "v": v, "output": out})


def case_ring_causal() -> None:
    torch.manual_seed(104)
    b, h, s, d = 2, 4, 16, 32
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    save_case("ring_causal", {"q": q, "k": k, "v": v, "output": out})


def case_nsa() -> None:
    torch.manual_seed(105)
    b, h, s, d = 2, 4, 16, 32
    block_size = 4
    window = 4
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)

    # cmp branch
    nb = s // block_size
    k_cmp = k.reshape(b, h, nb, block_size, d).mean(dim=3)
    v_cmp = v.reshape(b, h, nb, block_size, d).mean(dim=3)
    # block-causal mask: query at position i can attend to block 0..i//B
    mask_cmp = torch.full((s, nb), float("-inf"))
    for i in range(s):
        bi = i // block_size
        for j in range(nb):
            if j <= bi:
                mask_cmp[i, j] = 0.0
    scores_cmp = (q @ k_cmp.transpose(-1, -2)) * scale + mask_cmp
    y_cmp = F.softmax(scores_cmp, dim=-1) @ v_cmp

    # sliding-window branch
    mask_win = torch.full((s, s), float("-inf"))
    for i in range(s):
        lo = max(0, i - window + 1)
        mask_win[i, lo : i + 1] = 0.0
    y_win = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_win, is_causal=False)

    out = 0.5 * (y_cmp + y_win)
    save_case("nsa", {"q": q, "k": k, "v": v, "output": out})


def case_differential() -> None:
    torch.manual_seed(106)
    b, h, s, d = 2, 4, 8, 32
    q1 = torch.randn(b, h, s, d)
    q2 = torch.randn(b, h, s, d)
    k1 = torch.randn(b, h, s, d)
    k2 = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)
    lam = 0.5
    s1 = (q1 @ k1.transpose(-1, -2)) * scale
    s2 = (q2 @ k2.transpose(-1, -2)) * scale
    p1 = F.softmax(s1, dim=-1)
    p2 = F.softmax(s2, dim=-1)
    out = (p1 - lam * p2) @ v
    save_case(
        "differential",
        {"q1": q1, "q2": q2, "k1": k1, "k2": k2, "v": v, "output": out},
    )


def case_streaming_sink() -> None:
    torch.manual_seed(107)
    b, h, s, d = 2, 4, 16, 32
    sinks = 2
    window = 4
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)
    mask = torch.full((s, s), float("-inf"))
    for i in range(s):
        for j in range(s):
            if j > i:
                continue
            in_sink = j < sinks
            in_window = (j + window) > i
            if in_sink or in_window:
                mask[i, j] = 0.0
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    save_case("streaming_sink", {"q": q, "k": k, "v": v, "output": out})


def case_stripe() -> None:
    torch.manual_seed(108)
    b, h, s, d = 2, 4, 12, 32
    stripe = 3
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)
    mask = torch.full((s, s), float("-inf"))
    for i in range(s):
        for j in range(s):
            if j > i:
                continue
            if (i % stripe) == (j % stripe):
                mask[i, j] = 0.0
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    save_case("stripe", {"q": q, "k": k, "v": v, "output": out})


def case_blockwise() -> None:
    torch.manual_seed(109)
    b, h, s, d = 2, 4, 16, 32
    block = 4
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)
    mask = torch.full((s, s), float("-inf"))
    for i in range(s):
        for j in range(s):
            if j > i:
                continue
            if (i // block) == (j // block):
                mask[i, j] = 0.0
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    save_case("blockwise", {"q": q, "k": k, "v": v, "output": out})


def case_longformer() -> None:
    torch.manual_seed(110)
    b, h, s, d = 2, 4, 16, 32
    window = 2
    globals_ = [0, 5]
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)
    mask = torch.full((s, s), float("-inf"))
    for i in range(s):
        for j in range(s):
            if j > i:
                continue
            local = abs(i - j) <= window
            is_global = (i in globals_) or (j in globals_)
            if local or is_global:
                mask[i, j] = 0.0
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    save_case("longformer", {"q": q, "k": k, "v": v, "output": out})


def case_strided() -> None:
    torch.manual_seed(111)
    b, h, s, d = 2, 4, 16, 32
    stride = 4
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)
    mask = torch.full((s, s), float("-inf"))
    for i in range(s):
        for j in range(s):
            if j > i:
                continue
            local = (j + stride) > i and j <= i
            strd = (i - j) % stride == 0
            if local or strd:
                mask[i, j] = 0.0
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    save_case("strided", {"q": q, "k": k, "v": v, "output": out})


def case_bigbird() -> None:
    torch.manual_seed(112)
    b, h, s, d = 2, 4, 16, 32
    window = 2
    num_global = 2
    random_per_row = 0  # disable random for deterministic test
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)
    mask = torch.full((s, s), float("-inf"))
    for i in range(s):
        for j in range(s):
            if j > i:
                continue
            local = abs(i - j) <= window
            g_q = i < num_global
            g_k = j < num_global
            if local or g_q or g_k:
                mask[i, j] = 0.0
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    save_case("bigbird", {"q": q, "k": k, "v": v, "output": out})


def case_reformer_lsh() -> None:
    torch.manual_seed(113)
    b, h, s, d = 2, 4, 16, 32
    num_buckets = 4
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    buckets = torch.randint(0, num_buckets, (b, h, s), dtype=torch.int64)
    scale = 1.0 / (d ** 0.5)
    mask = torch.full((b, h, s, s), float("-inf"))
    for bi in range(b):
        for hi in range(h):
            for i in range(s):
                for j in range(s):
                    if j > i:
                        continue
                    if buckets[bi, hi, i] == buckets[bi, hi, j]:
                        mask[bi, hi, i, j] = 0.0
    scores = (q @ k.transpose(-1, -2)) * scale + mask
    probs = F.softmax(scores, dim=-1)
    out = probs @ v
    save_case(
        "reformer_lsh",
        {"q": q, "k": k, "v": v, "buckets": buckets, "output": out},
    )


# ───────────────────────── linear-attention family ─────────────────────────
# Все эталоны воспроизводят ту же раскладку состояния S[r,c] и порядок
# обновления, что и Rust-реализации в crates/synaptix-ops/src/attention/linear/.


def case_naive_linear() -> None:
    torch.manual_seed(114)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    s_full = torch.einsum("bhsd,bhsv->bhdv", k, v)
    out = torch.einsum("bhsd,bhdv->bhsv", q, s_full)
    save_case("naive_linear", {"q": q, "k": k, "v": v, "output": out})


def case_retnet() -> None:
    torch.manual_seed(115)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    gamma = 0.9
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    out = torch.zeros(b, h, s, dv)
    for bi in range(b):
        for hi in range(h):
            state = torch.zeros(dk, dv)
            for t in range(s):
                state = gamma * state + torch.outer(k[bi, hi, t], v[bi, hi, t])
                out[bi, hi, t] = q[bi, hi, t] @ state
    save_case("retnet", {"q": q, "k": k, "v": v, "output": out})


def case_gla() -> None:
    torch.manual_seed(116)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    gate = -0.1 * torch.rand(b, h, s, dk)  # log-decay <= 0
    out = torch.zeros(b, h, s, dv)
    for bi in range(b):
        for hi in range(h):
            state = torch.zeros(dk, dv)
            for t in range(s):
                decay = gate[bi, hi, t].exp().unsqueeze(-1)  # [dk,1]
                state = decay * state + torch.outer(k[bi, hi, t], v[bi, hi, t])
                out[bi, hi, t] = q[bi, hi, t] @ state
    save_case("gla", {"q": q, "k": k, "v": v, "gate": gate, "output": out})


def case_delta_net() -> None:
    torch.manual_seed(117)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    beta = 0.5
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    out = torch.zeros(b, h, s, dv)
    for bi in range(b):
        for hi in range(h):
            state = torch.zeros(dk, dv)  # rows=key, cols=value
            for t in range(s):
                kt, vt, qt = k[bi, hi, t], v[bi, hi, t], q[bi, hi, t]
                kv_old = state.transpose(0, 1) @ kt  # [dv]
                delta = beta * (vt - kv_old)
                state = state + torch.outer(kt, delta)
                out[bi, hi, t] = state.transpose(0, 1) @ qt
    save_case("delta_net", {"q": q, "k": k, "v": v, "output": out})


def case_gated_delta_net() -> None:
    torch.manual_seed(118)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    g = -0.1 * torch.rand(b, h, s)  # log-decay
    beta = torch.rand(b, h, s)
    q_scale = dk ** -0.5

    def l2n(x):
        return x / (x.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()

    qn = l2n(q) * q_scale
    kn = l2n(k)
    out = torch.zeros(b, h, s, dv)
    for bi in range(b):
        for hi in range(h):
            state = torch.zeros(dk, dv)
            for t in range(s):
                gt = float(g[bi, hi, t].exp())
                bt = float(beta[bi, hi, t])
                sg = gt * state
                kv = sg.transpose(0, 1) @ kn[bi, hi, t]
                delta = bt * (v[bi, hi, t] - kv)
                state = sg + torch.outer(kn[bi, hi, t], delta)
                out[bi, hi, t] = state.transpose(0, 1) @ qn[bi, hi, t]
    save_case(
        "gated_delta_net",
        {"q": q, "k": k, "v": v, "g": g, "beta": beta, "output": out},
    )


def case_chunk_scan() -> None:
    torch.manual_seed(119)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    # эталон = рекуррентный causal linear-scan (без decay)
    out = torch.zeros(b, h, s, dv)
    for bi in range(b):
        for hi in range(h):
            state = torch.zeros(dk, dv)
            for t in range(s):
                state = state + torch.outer(k[bi, hi, t], v[bi, hi, t])
                out[bi, hi, t] = q[bi, hi, t] @ state
    save_case("chunk_scan", {"q": q, "k": k, "v": v, "output": out})


def case_based() -> None:
    torch.manual_seed(120)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)

    def phi(x):  # [...,dk] -> [...,1+2dk]
        ones = torch.ones_like(x[..., :1])
        return torch.cat([ones, x, (x * x) / (2 ** 0.5)], dim=-1)

    pq, pk = phi(q), phi(k)
    m = 1 + 2 * dk
    out = torch.zeros(b, h, s, dv)
    for bi in range(b):
        for hi in range(h):
            state = torch.zeros(m, dv)
            z = torch.zeros(m)
            for t in range(s):
                state = state + torch.outer(pk[bi, hi, t], v[bi, hi, t])
                z = z + pk[bi, hi, t]
                num = pq[bi, hi, t] @ state
                den = pq[bi, hi, t] @ z
                out[bi, hi, t] = num / (den + 1e-6)
    save_case("based", {"q": q, "k": k, "v": v, "output": out})


def case_cosformer() -> None:
    import math

    torch.manual_seed(121)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    qr, kr = q.relu(), k.relu()
    idx = torch.arange(s).float()
    theta = (math.pi / 2) * (idx / s)
    cos, sin = theta.cos(), theta.sin()
    out = torch.zeros(b, h, s, dv)
    for bi in range(b):
        for hi in range(h):
            sc = torch.zeros(dk, dv)
            ss = torch.zeros(dk, dv)
            zc = torch.zeros(dk)
            zs = torch.zeros(dk)
            for t in range(s):
                ct, st = float(cos[t]), float(sin[t])
                sc = sc + ct * torch.outer(kr[bi, hi, t], v[bi, hi, t])
                ss = ss + st * torch.outer(kr[bi, hi, t], v[bi, hi, t])
                zc = zc + ct * kr[bi, hi, t]
                zs = zs + st * kr[bi, hi, t]
                num = ct * (qr[bi, hi, t] @ sc) + st * (qr[bi, hi, t] @ ss)
                den = ct * (qr[bi, hi, t] @ zc) + st * (qr[bi, hi, t] @ zs)
                out[bi, hi, t] = num / (den + 1e-6)
    save_case("cosformer", {"q": q, "k": k, "v": v, "output": out})


def case_performer() -> None:
    torch.manual_seed(122)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    m = 16
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    proj = torch.randn(m, dk)

    def phi(x):  # x: [...,dk] -> [...,m]
        x = x * (dk ** -0.25)
        proj_x = x @ proj.t()  # [...,m]
        norm = (x * x).sum(-1, keepdim=True) / 2
        return torch.exp(proj_x - norm) / (m ** 0.5)

    pq, pk = phi(q), phi(k)
    skv = torch.einsum("bhsm,bhsv->bhmv", pk, v)
    sk = pk.sum(2)  # [b,h,m]
    num = torch.einsum("bhsm,bhmv->bhsv", pq, skv)
    den = torch.einsum("bhsm,bhm->bhs", pq, sk).unsqueeze(-1)
    out = num / (den + 1e-6)
    save_case("performer", {"q": q, "k": k, "v": v, "proj": proj, "output": out})


def case_linformer() -> None:
    torch.manual_seed(123)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    r = 4
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    e_proj = torch.randn(r, s)
    f_proj = torch.randn(r, s)
    kp = torch.einsum("rs,bhsd->bhrd", e_proj, k)  # [b,h,r,dk]
    vp = torch.einsum("rs,bhsv->bhrv", f_proj, v)  # [b,h,r,dv]
    scores = (q @ kp.transpose(-1, -2)) * (dk ** -0.5)  # [b,h,s,r]
    out = F.softmax(scores, dim=-1) @ vp
    save_case(
        "linformer",
        {"q": q, "k": k, "v": v, "e_proj": e_proj, "f_proj": f_proj, "output": out},
    )


def case_tnn() -> None:
    torch.manual_seed(124)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    rel = torch.randn(h, s)  # kernel[h, lag]
    out = torch.zeros(b, h, s, dv)
    for bi in range(b):
        for hi in range(h):
            for i in range(s):
                for j in range(i + 1):
                    out[bi, hi, i] += rel[hi, i - j] * v[bi, hi, j]
    save_case("tnn", {"q": q, "k": k, "v": v, "rel_kernel": rel, "output": out})


def _synthesizer(seed: int, name: str, causal: bool) -> None:
    torch.manual_seed(seed)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    synth = torch.randn(h, s, s)
    scores = synth.unsqueeze(0).expand(b, h, s, s).clone()
    if causal:
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    out = F.softmax(scores, dim=-1) @ v
    save_case(name, {"q": q, "k": k, "v": v, "synth": synth, "output": out})


def case_hyena() -> None:
    torch.manual_seed(127)
    b, h, s, d = 2, 4, 8, 16  # dk == dv
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    filt = torch.randn(h, s)
    u = k * v
    c = torch.zeros(b, h, s, d)
    for bi in range(b):
        for hi in range(h):
            for i in range(s):
                for j in range(i + 1):
                    c[bi, hi, i] += filt[hi, i - j] * u[bi, hi, j]
    out = q * c
    save_case("hyena", {"q": q, "k": k, "v": v, "filt": filt, "output": out})


def case_abc() -> None:
    torch.manual_seed(128)
    b, h, s, dk, dv = 2, 4, 8, 16, 16
    m = 4
    q = torch.randn(b, h, s, dk)
    k = torch.randn(b, h, s, dk)
    v = torch.randn(b, h, s, dv)
    slot_proj = torch.randn(dk, m)
    sl = torch.einsum("bhsd,dm->bhsm", k, slot_proj)  # [b,h,s,m]
    phi = F.softmax(sl, dim=2)  # softmax по оси последовательности
    mem = torch.einsum("bhsm,bhsv->bhmv", phi, v)  # [b,h,m,dv]
    memk = torch.einsum("bhsm,bhsd->bhmd", phi, k)  # [b,h,m,dk]
    alpha = F.softmax((q @ memk.transpose(-1, -2)) * (dk ** -0.5), dim=-1)  # [b,h,s,m]
    out = alpha @ mem
    save_case("abc", {"q": q, "k": k, "v": v, "slot_proj": slot_proj, "output": out})


def case_flash_v2_nomask() -> None:
    torch.manual_seed(129)
    b, h, s, d = 2, 4, 8, 16
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    out = F.scaled_dot_product_attention(q, k, v)  # scale = 1/sqrt(d), no mask
    save_case("flash_v2_nomask", {"q": q, "k": k, "v": v, "output": out})


def case_flash_v2_causal() -> None:
    torch.manual_seed(130)
    b, h, s, d = 2, 4, 8, 16
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    # additive causal маска [S,S]: 0 при j<=i, иначе -inf
    mask = torch.full((s, s), float("-inf"))
    for i in range(s):
        for j in range(i + 1):
            mask[i, j] = 0.0
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    save_case("flash_v2_causal", {"q": q, "k": k, "v": v, "mask": mask, "output": out})


def case_flash_decode() -> None:
    torch.manual_seed(131)
    b, h, sq, sk, d = 2, 4, 2, 8, 16
    q = torch.randn(b, h, sq, d)
    k_cache = torch.randn(b, h, sk, d)
    v_cache = torch.randn(b, h, sk, d)
    out = F.scaled_dot_product_attention(q, k_cache, v_cache)  # no mask
    save_case("flash_decode", {"q": q, "k_cache": k_cache, "v_cache": v_cache, "output": out})


def main() -> None:
    print("Generating advanced attention reference data...")
    case_mla()
    case_lightning_no_causal()
    case_lightning_causal()
    case_ring()
    case_ring_causal()
    case_nsa()
    case_differential()
    case_streaming_sink()
    case_stripe()
    case_blockwise()
    case_longformer()
    case_strided()
    case_bigbird()
    case_reformer_lsh()
    # linear-attention family
    case_naive_linear()
    case_retnet()
    case_gla()
    case_delta_net()
    case_gated_delta_net()
    case_chunk_scan()
    case_based()
    case_cosformer()
    case_performer()
    case_linformer()
    case_tnn()
    _synthesizer(125, "synthesizer_no_causal", causal=False)
    _synthesizer(126, "synthesizer_causal", causal=True)
    case_hyena()
    case_abc()
    case_flash_v2_nomask()
    case_flash_v2_causal()
    case_flash_decode()
    print("Done.")


if __name__ == "__main__":
    main()
