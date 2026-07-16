"""Generate reference data for synaptix-train optimizers + losses.

Run:
    python scripts/reference/gen_train_optimizers.py

Tests 3 update steps for each optimizer with fixed grads + initial params.
"""

import math
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/train")


def save_case(name, tensors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_cross_entropy():
    torch.manual_seed(900)
    b, vocab = 8, 32
    logits = torch.randn(b, vocab) * 2.0
    targets = torch.randint(0, vocab, (b,), dtype=torch.int64)
    loss_none = F.cross_entropy(logits, targets, reduction="none")
    loss_mean = F.cross_entropy(logits, targets, reduction="mean")
    loss_sum = F.cross_entropy(logits, targets, reduction="sum")
    save_case("cross_entropy", {
        "logits": logits, "targets": targets,
        "loss_none": loss_none, "loss_mean": loss_mean, "loss_sum": loss_sum,
    })


def case_cross_entropy_3d():
    torch.manual_seed(901)
    b, s, vocab = 2, 4, 16
    logits = torch.randn(b, s, vocab)
    targets = torch.randint(0, vocab, (b, s), dtype=torch.int64)
    targets[0, 0] = -100  # ignore_index
    loss_mean = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), ignore_index=-100, reduction="mean")
    save_case("cross_entropy_3d", {
        "logits": logits, "targets": targets, "loss_mean": loss_mean,
    })


def case_mse():
    torch.manual_seed(902)
    input_t = torch.randn(4, 8)
    target = torch.randn(4, 8)
    mse_mean = F.mse_loss(input_t, target, reduction="mean")
    mse_sum = F.mse_loss(input_t, target, reduction="sum")
    l1_mean = F.l1_loss(input_t, target, reduction="mean")
    smooth = F.smooth_l1_loss(input_t, target, beta=1.0, reduction="mean")
    save_case("mse_l1", {
        "input": input_t, "target": target,
        "mse_mean": mse_mean, "mse_sum": mse_sum,
        "l1_mean": l1_mean, "smooth_l1_mean": smooth,
    })


def adamw_ref_step(params, grads, m, v, step, lr, betas, eps, wd):
    b1, b2 = betas
    m = b1 * m + (1 - b1) * grads
    v = b2 * v + (1 - b2) * grads * grads
    bc1 = 1.0 - b1 ** step
    bc2 = 1.0 - b2 ** step
    m_hat = m / bc1
    v_hat = v / bc2
    update = m_hat / (v_hat.sqrt() + eps)
    params = params * (1 - lr * wd) - update * lr
    return params, m, v


def case_adamw():
    torch.manual_seed(903)
    params = torch.randn(4, 8)
    grads = [torch.randn(4, 8) for _ in range(3)]
    lr, betas, eps, wd = 0.01, (0.9, 0.999), 1e-8, 0.01
    m = torch.zeros_like(params)
    v = torch.zeros_like(params)
    p = params.clone()
    for step, g in enumerate(grads, 1):
        p, m, v = adamw_ref_step(p, g, m, v, step, lr, betas, eps, wd)
    save_case("adamw", {
        "params_init": params,
        "grad_0": grads[0], "grad_1": grads[1], "grad_2": grads[2],
        "params_final": p,
    })


def lion_ref_step(params, grads, m_state, lr, betas, wd):
    b1, b2 = betas
    update = (b1 * m_state + (1 - b1) * grads).sign()
    m_state = b2 * m_state + (1 - b2) * grads
    params = params * (1 - lr * wd) - update * lr
    return params, m_state


def case_lion():
    torch.manual_seed(904)
    params = torch.randn(4, 8)
    grads = [torch.randn(4, 8) for _ in range(3)]
    lr, betas, wd = 0.001, (0.9, 0.99), 0.0
    m = torch.zeros_like(params)
    p = params.clone()
    for g in grads:
        p, m = lion_ref_step(p, g, m, lr, betas, wd)
    save_case("lion", {
        "params_init": params,
        "grad_0": grads[0], "grad_1": grads[1], "grad_2": grads[2],
        "params_final": p,
    })


def adafactor_ref_step(params, grads, r, c, step, lr, eps1, decay_rate, clip_threshold, wd):
    t = step
    beta2_t = 1.0 - t ** decay_rate
    g_sq = grads * grads + eps1
    r = beta2_t * r + (1 - beta2_t) * g_sq.mean(dim=1)
    c = beta2_t * c + (1 - beta2_t) * g_sq.mean(dim=0)
    r_sum = max(r.sum().item(), eps1)
    r_norm = r / r_sum
    v_2d = r_norm.unsqueeze(1) * c.unsqueeze(0)
    update = grads / v_2d.sqrt()
    n = update.numel()
    upd_rms = (update.pow(2).sum() / n).sqrt()
    clip_scale = 1.0 / max(float(upd_rms / clip_threshold), 1.0)
    update = update * clip_scale
    params = params - update * lr
    if wd > 0:
        params = params * (1 - lr * wd)
    return params, r, c


def case_adafactor():
    torch.manual_seed(905)
    params = torch.randn(4, 8)
    grads = [torch.randn(4, 8) for _ in range(3)]
    lr = 0.01
    eps1, decay_rate, clip_threshold, wd = 1e-30, -0.8, 1.0, 0.0
    r = torch.zeros(4)
    c = torch.zeros(8)
    p = params.clone()
    for step, g in enumerate(grads, 1):
        p, r, c = adafactor_ref_step(p, g, r, c, step, lr, eps1, decay_rate, clip_threshold, wd)
    save_case("adafactor", {
        "params_init": params,
        "grad_0": grads[0], "grad_1": grads[1], "grad_2": grads[2],
        "params_final": p,
    })


def newton_schulz_zeropower(g, steps=5):
    transposed = g.shape[0] > g.shape[1]
    x = g.T.contiguous() if transposed else g.clone()
    frob = max(x.pow(2).sum().sqrt().item(), 1e-7)
    x = x / frob
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xt = x.T.contiguous()
        a_mat = x @ xt
        b_mat = a_mat @ a_mat
        part = b * a_mat + c * b_mat
        x = a * x + part @ x
    return x.T.contiguous() if transposed else x


def muon_ref_step(params, grads, momentum_buf, lr, momentum, nesterov, ns_steps, wd):
    buf = momentum * momentum_buf + grads
    momentum_buf = buf.clone()
    update_input = grads + momentum * buf if nesterov else buf
    update = newton_schulz_zeropower(update_input, steps=ns_steps) if update_input.dim() == 2 else update_input
    params = params - update * lr
    if wd > 0:
        params = params * (1 - lr * wd)
    return params, momentum_buf


def case_muon():
    torch.manual_seed(906)
    params = torch.randn(4, 8)
    grads = [torch.randn(4, 8) for _ in range(3)]
    lr, momentum, ns_steps, wd = 0.02, 0.95, 5, 0.0
    buf = torch.zeros_like(params)
    p = params.clone()
    for g in grads:
        p, buf = muon_ref_step(p, g, buf, lr, momentum, True, ns_steps, wd)
    save_case("muon", {
        "params_init": params,
        "grad_0": grads[0], "grad_1": grads[1], "grad_2": grads[2],
        "params_final": p,
    })


# ───────────────────────── расширение: optimizers ─────────────────────────


def case_adem_amix():
    torch.manual_seed(907)
    params = torch.randn(4, 8)
    grads = [torch.randn(4, 8) for _ in range(3)]
    lr, (b1, b2, b3), alpha, eps, wd = 0.01, (0.9, 0.999, 0.9999), 2.0, 1e-8, 0.0
    m1 = torch.zeros_like(params)
    m2 = torch.zeros_like(params)
    v = torch.zeros_like(params)
    p = params.clone()
    for step, g in enumerate(grads, 1):
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b3 * m2 + (1 - b3) * g
        v = b2 * v + (1 - b2) * g * g
        bc1 = 1 - b1 ** step
        bc2 = 1 - b2 ** step
        upd = (m1 / bc1 + alpha * m2) / (torch.sqrt(v / bc2) + eps)
        p = p * (1 - lr * wd) - lr * upd
    save_case("adem_amix", {
        "params_init": params,
        "grad_0": grads[0], "grad_1": grads[1], "grad_2": grads[2],
        "params_final": p,
    })


def case_sophia():
    torch.manual_seed(908)
    params = torch.randn(4, 8)
    grads = [torch.randn(4, 8) for _ in range(3)]
    lr, (b1, b2), rho, eps, wd = 0.01, (0.96, 0.99), 0.04, 1e-12, 0.0
    m = torch.zeros_like(params)
    h = torch.zeros_like(params)
    p = params.clone()
    for g in grads:
        m = b1 * m + (1 - b1) * g
        h = b2 * h + (1 - b2) * g * g
        upd = torch.clamp(m / (rho * h + eps), -1.0, 1.0)
        p = p * (1 - lr * wd) - lr * upd
    save_case("sophia", {
        "params_init": params,
        "grad_0": grads[0], "grad_1": grads[1], "grad_2": grads[2],
        "params_final": p,
    })


def case_adam8bit():
    torch.manual_seed(909)
    params = torch.randn(4, 8)
    grads = [torch.randn(4, 8) for _ in range(3)]
    lr, (b1, b2), eps, wd = 0.01, (0.9, 0.999), 1e-8, 0.0
    m = torch.zeros_like(params)
    v = torch.zeros_like(params)
    p = params.clone()
    for step, g in enumerate(grads, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        bc1 = 1 - b1 ** step
        bc2 = 1 - b2 ** step
        upd = (m / bc1) / (torch.sqrt(v / bc2) + eps)
        p = p * (1 - lr * wd) - lr * upd
    save_case("adam8bit", {
        "params_init": params,
        "grad_0": grads[0], "grad_1": grads[1], "grad_2": grads[2],
        "params_final": p,
    })


def case_grad_clip_norm():
    torch.manual_seed(910)
    g0 = torch.randn(4, 8)
    g1 = torch.randn(3)
    max_norm = 1.0
    total = torch.sqrt(g0.pow(2).sum() + g1.pow(2).sum())
    scale = max_norm / (total + 1e-6)
    g0c = g0 * scale
    g1c = g1 * scale
    save_case("grad_clip_norm", {
        "g0": g0, "g1": g1,
        "total_norm": total.reshape(1),
        "g0_clipped": g0c, "g1_clipped": g1c,
    })


def case_grad_clip_value():
    torch.manual_seed(911)
    g = torch.randn(4, 8)
    max_val = 0.5
    save_case("grad_clip_value", {
        "g": g, "max_val": torch.tensor([max_val]),
        "clipped": torch.clamp(g, -max_val, max_val),
    })


# ───────────────────────── расширение: RLHF losses ─────────────────────────


def case_dpo():
    torch.manual_seed(912)
    n = 6
    beta = 0.1
    pc = torch.randn(n)
    pr = torch.randn(n)
    rc = torch.randn(n)
    rr = torch.randn(n)
    logits = beta * ((pc - pr) - (rc - rr))
    loss = -F.logsigmoid(logits).mean()
    save_case("dpo", {
        "policy_chosen": pc, "policy_rejected": pr,
        "ref_chosen": rc, "ref_rejected": rr,
        "output": loss.reshape(1),
    })


def case_orpo():
    torch.manual_seed(913)
    n = 6
    lam = 0.1
    c = -torch.rand(n) - 0.1  # log-probs < 0
    r = -torch.rand(n) - 0.1
    log_odds = (c - torch.log1p(-torch.exp(c))) - (r - torch.log1p(-torch.exp(r)))
    sft = (-c).mean()
    or_loss = (-F.logsigmoid(log_odds)).mean()
    loss = sft + lam * or_loss
    save_case("orpo", {
        "chosen_logps": c, "rejected_logps": r,
        "output": loss.reshape(1),
    })


def case_kto():
    torch.manual_seed(914)
    nc, nr = 4, 4
    beta = 0.1
    pc = torch.randn(nc)
    rc = torch.randn(nc)
    pr = torch.randn(nr)
    rr = torch.randn(nr)
    chosen_lr = pc - rc
    rejected_lr = pr - rr
    kl = torch.clamp((chosen_lr.sum() + rejected_lr.sum()) / (nc + nr), min=0.0)
    loss_c = 1.0 - torch.sigmoid(beta * (chosen_lr - kl))
    loss_r = 1.0 - torch.sigmoid(beta * (kl - rejected_lr))
    loss = (loss_c.sum() + loss_r.sum()) / (nc + nr)
    save_case("kto", {
        "policy_chosen": pc, "ref_chosen": rc,
        "policy_rejected": pr, "ref_rejected": rr,
        "output": loss.reshape(1),
    })


def main():
    print("Generating train ref data...")
    case_cross_entropy()
    case_cross_entropy_3d()
    case_mse()
    case_adamw()
    case_lion()
    case_adafactor()
    case_muon()
    # расширение
    case_adem_amix()
    case_sophia()
    case_adam8bit()
    case_grad_clip_norm()
    case_grad_clip_value()
    case_dpo()
    case_orpo()
    case_kto()
    print("Done.")


if __name__ == "__main__":
    main()
