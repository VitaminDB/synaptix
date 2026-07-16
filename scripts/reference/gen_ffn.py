"""Generate reference SafeTensors for Session 4 — FFN variants.

Run:
    python scripts/reference/gen_ffn.py

Covers: mlp_gelu, swiglu, geglu, reglu.
All weights randomized with fixed seed and saved alongside inputs/outputs.
Outputs data/ref/ffn/<case>.safetensors.
"""

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/ffn")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_mlp_gelu() -> None:
    torch.manual_seed(0)
    batch, seq, hidden, intermediate = 2, 16, 256, 512
    x = torch.randn(batch, seq, hidden, dtype=torch.float32)
    w1 = torch.randn(intermediate, hidden, dtype=torch.float32) * 0.02
    w2 = torch.randn(hidden, intermediate, dtype=torch.float32) * 0.02
    b1 = torch.zeros(intermediate, dtype=torch.float32)
    b2 = torch.zeros(hidden, dtype=torch.float32)
    h = F.gelu(F.linear(x, w1, b1), approximate="tanh")
    out = F.linear(h, w2, b2)
    save_case(
        "mlp_gelu",
        {"input": x, "w1": w1, "b1": b1, "w2": w2, "b2": b2, "output": out},
    )


def case_swiglu() -> None:
    torch.manual_seed(1)
    batch, seq, hidden, intermediate = 2, 16, 256, 512
    x = torch.randn(batch, seq, hidden, dtype=torch.float32)
    w1 = torch.randn(intermediate, hidden, dtype=torch.float32) * 0.02
    w2 = torch.randn(hidden, intermediate, dtype=torch.float32) * 0.02
    w3 = torch.randn(intermediate, hidden, dtype=torch.float32) * 0.02
    gate = F.silu(F.linear(x, w1))
    hidden_states = gate * F.linear(x, w3)
    out = F.linear(hidden_states, w2)
    save_case(
        "swiglu",
        {"input": x, "w_gate": w1, "w_up": w3, "w_down": w2, "output": out},
    )


def case_geglu() -> None:
    torch.manual_seed(2)
    batch, seq, hidden, intermediate = 2, 16, 256, 512
    x = torch.randn(batch, seq, hidden, dtype=torch.float32)
    w_combined = torch.randn(intermediate * 2, hidden, dtype=torch.float32) * 0.02
    w_down = torch.randn(hidden, intermediate, dtype=torch.float32) * 0.02
    proj = F.linear(x, w_combined)
    x1, x2 = proj.chunk(2, dim=-1)
    hidden_states = x1 * F.gelu(x2, approximate="tanh")
    out = F.linear(hidden_states, w_down)
    save_case(
        "geglu",
        {"input": x, "w_combined": w_combined, "w_down": w_down, "output": out},
    )


def case_reglu() -> None:
    torch.manual_seed(3)
    batch, seq, hidden, intermediate = 2, 16, 256, 512
    x = torch.randn(batch, seq, hidden, dtype=torch.float32)
    w_combined = torch.randn(intermediate * 2, hidden, dtype=torch.float32) * 0.02
    w_down = torch.randn(hidden, intermediate, dtype=torch.float32) * 0.02
    proj = F.linear(x, w_combined)
    x1, x2 = proj.chunk(2, dim=-1)
    hidden_states = x1 * F.relu(x2)
    out = F.linear(hidden_states, w_down)
    save_case(
        "reglu",
        {"input": x, "w_combined": w_combined, "w_down": w_down, "output": out},
    )


# ───────────────────────── ffn/moe расширение ─────────────────────────


def case_d_gate_net() -> None:
    torch.manual_seed(10)
    n, d = 4, 6
    x = torch.randn(n, d)
    gate_weight = torch.randn(d, d)
    out = x * torch.sigmoid(x @ gate_weight)
    save_case("d_gate_net", {"x": x, "gate_weight": gate_weight, "output": out})


def case_monarch_mixer() -> None:
    torch.manual_seed(11)
    n, d1, d2 = 4, 3, 4
    d = d1 * d2
    x = torch.randn(n, d)
    m1 = torch.randn(d1, d1)
    m2 = torch.randn(d2, d2)
    X = x.reshape(n, d1, d2)
    y = torch.einsum("ip,npq,jq->nij", m1, X, m2)
    out = y.reshape(n, d)
    save_case("monarch_mixer", {"x": x, "m1": m1, "m2": m2, "output": out})


def _bspline_basis(t: float, grid, degree: int):
    m = len(grid)
    b = [0.0] * (m - 1)
    for i in range(m - 1):
        in_iv = grid[i] <= t and (t < grid[i + 1] or (i == m - 2 and t <= grid[i + 1]))
        b[i] = 1.0 if in_iv else 0.0
    for p in range(1, degree + 1):
        length = m - 1 - p
        b2 = [0.0] * length
        for i in range(length):
            d1 = grid[i + p] - grid[i]
            d2 = grid[i + p + 1] - grid[i + 1]
            t1 = (t - grid[i]) / d1 * b[i] if d1 > 0 else 0.0
            t2 = (grid[i + p + 1] - t) / d2 * b[i + 1] if d2 > 0 else 0.0
            b2[i] = t1 + t2
        b = b2
    return b


def case_kan() -> None:
    torch.manual_seed(12)
    n, d = 3, 5
    degree = 3
    grid = torch.linspace(-1.0, 1.0, 8)  # num_basis = 8 - 3 - 1 = 4
    num_basis = len(grid) - degree - 1
    coeff = torch.randn(num_basis)
    x = (torch.rand(n, d) * 1.8) - 0.9  # интерьер (-0.9, 0.9)
    g = grid.tolist()
    c = coeff.tolist()
    lo, hi = g[0], g[-1]
    out = torch.zeros(n, d)
    flat = x.flatten().tolist()
    res = []
    for v in flat:
        t = min(max(v, lo), hi)
        basis = _bspline_basis(t, g, degree)
        res.append(sum(cv * bv for cv, bv in zip(c, basis)))
    out = torch.tensor(res, dtype=torch.float32).reshape(n, d)
    save_case("kan", {"x": x, "grid": grid, "coeff": coeff, "output": out})


def case_expert() -> None:
    torch.manual_seed(13)
    n, d, h = 4, 6, 8
    x = torch.randn(n, d)
    fc1 = torch.randn(h, d)
    fc2 = torch.randn(d, h)
    out = F.relu(x @ fc1.t()) @ fc2.t()
    save_case("expert", {"x": x, "fc1": fc1, "fc2": fc2, "output": out})


def case_shared_expert() -> None:
    torch.manual_seed(14)
    n, d, h = 4, 6, 8
    x = torch.randn(n, d)
    fc1 = torch.randn(h, d)
    fc2 = torch.randn(d, h)
    out = F.relu(x @ fc1.t()) @ fc2.t()
    save_case("shared_expert", {"x": x, "fc1": fc1, "fc2": fc2, "output": out})


def case_fine_grained_moe() -> None:
    torch.manual_seed(15)
    n, d, e, h, k = 4, 6, 5, 8, 2
    x = torch.randn(n, d)
    router_w = torch.randn(d, e)
    experts_fc1 = torch.randn(e, h, d)
    experts_fc2 = torch.randn(e, d, h)
    logits = x @ router_w  # [N,E]
    vals, idx = logits.topk(k, dim=-1)
    w = torch.softmax(vals, dim=-1)  # [N,k]
    out = torch.zeros(n, d)
    for ni in range(n):
        for pos in range(k):
            ei = int(idx[ni, pos])
            hid = F.relu(x[ni] @ experts_fc1[ei].t())
            ye = hid @ experts_fc2[ei].t()
            out[ni] += float(w[ni, pos]) * ye
    save_case(
        "fine_grained_moe",
        {
            "x": x,
            "router_w": router_w,
            "experts_fc1": experts_fc1,
            "experts_fc2": experts_fc2,
            "output": out,
        },
    )


def case_soft_router() -> None:
    torch.manual_seed(16)
    n, e = 4, 5
    logits = torch.randn(n, e)
    out = torch.softmax(logits, dim=-1)
    save_case("soft_router", {"logits": logits, "output": out})


def case_top_k_router() -> None:
    torch.manual_seed(17)
    n, e, k = 4, 6, 3
    logits = torch.randn(n, e)
    vals, idx = logits.topk(k, dim=-1)  # desc по значению
    w = torch.softmax(vals, dim=-1)
    save_case(
        "top_k_router",
        {"logits": logits, "indices": idx.float(), "weights": w},
    )


def case_expert_choice_router() -> None:
    torch.manual_seed(18)
    n, e, cap = 6, 4, 3
    logits = torch.randn(n, e)
    mask = torch.zeros(n, e)
    for j in range(e):
        _, tok = logits[:, j].topk(cap)
        mask[tok, j] = 1.0
    save_case("expert_choice_router", {"logits": logits, "output": mask})


def case_auxiliary_loss() -> None:
    torch.manual_seed(19)
    n, e = 8, 4
    router_probs = torch.softmax(torch.randn(n, e), dim=-1)
    expert_indices = router_probs.argmax(dim=-1)
    f = torch.bincount(expert_indices, minlength=e).float() / n
    p = router_probs.mean(dim=0)
    loss = e * (f * p).sum()
    save_case(
        "auxiliary_loss",
        {
            "router_probs": router_probs,
            "expert_indices": expert_indices.float(),
            "output": loss.reshape(1),
        },
    )


def case_z_loss() -> None:
    torch.manual_seed(20)
    n, e = 8, 4
    logits = torch.randn(n, e)
    lse = torch.logsumexp(logits, dim=-1)
    loss = (lse * lse).mean()
    save_case("z_loss", {"router_logits": logits, "output": loss.reshape(1)})


def main() -> None:
    print("Generating FFN reference data...")
    case_mlp_gelu()
    case_swiglu()
    case_geglu()
    case_reglu()
    # ffn/moe расширение
    case_d_gate_net()
    case_monarch_mixer()
    case_kan()
    case_expert()
    case_shared_expert()
    case_fine_grained_moe()
    case_soft_router()
    case_top_k_router()
    case_expert_choice_router()
    case_auxiliary_loss()
    case_z_loss()
    print("Done.")


if __name__ == "__main__":
    main()
