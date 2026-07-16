"""Generate reference data for Session 7 — Autograd numerical gradient check.

Run:
    python scripts/reference/gen_autograd.py

Uses F64 for all numerical gradient checks (h=1e-5).
Outputs data/ref/autograd/<case>.safetensors with inputs, analytical grads, and numerical grads.
Also outputs loss curves for MLP training convergence test.
"""

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/autograd")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def numerical_grad(fn, x: torch.Tensor, h: float = 1e-5) -> torch.Tensor:
    x = x.detach().double()
    grad = torch.zeros_like(x)
    it = x.view(-1)
    for i in range(it.numel()):
        orig = it[i].item()
        it[i] = orig + h
        f_plus = fn(x).sum().item()
        it[i] = orig - h
        f_minus = fn(x).sum().item()
        it[i] = orig
        grad.view(-1)[i] = (f_plus - f_minus) / (2.0 * h)
    return grad


def case_matmul_grad() -> None:
    torch.manual_seed(0)
    a = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    b = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)
    out = torch.mm(a, b)
    out.sum().backward()
    grad_a_analytical = a.grad.detach().clone()
    grad_b_analytical = b.grad.detach().clone()
    grad_a_numerical = numerical_grad(lambda x: torch.mm(x, b.detach()), a.detach())
    grad_b_numerical = numerical_grad(lambda x: torch.mm(a.detach(), x), b.detach())
    save_case(
        "matmul_grad",
        {
            "a": a.detach().float(),
            "b": b.detach().float(),
            "grad_a_analytical": grad_a_analytical.float(),
            "grad_b_analytical": grad_b_analytical.float(),
            "grad_a_numerical": grad_a_numerical.float(),
            "grad_b_numerical": grad_b_numerical.float(),
        },
    )


def case_rms_norm_grad() -> None:
    torch.manual_seed(1)
    hidden = 64
    x = torch.randn(4, hidden, dtype=torch.float64, requires_grad=True)
    weight = torch.ones(hidden, dtype=torch.float64, requires_grad=True)

    def rms_norm_fn(inp):
        variance = inp.pow(2).mean(-1, keepdim=True)
        return inp * torch.rsqrt(variance + 1e-6) * weight.detach()

    out = rms_norm_fn(x)
    out.sum().backward()
    grad_x_analytical = x.grad.detach().clone()
    grad_x_numerical = numerical_grad(rms_norm_fn, x.detach())
    save_case(
        "rms_norm_grad",
        {
            "input": x.detach().float(),
            "weight": weight.detach().float(),
            "grad_x_analytical": grad_x_analytical.float(),
            "grad_x_numerical": grad_x_numerical.float(),
        },
    )


def case_gelu_grad() -> None:
    torch.manual_seed(2)
    x = torch.randn(8, 64, dtype=torch.float64, requires_grad=True)
    out = F.gelu(x, approximate="tanh")
    out.sum().backward()
    grad_analytical = x.grad.detach().clone()
    grad_numerical = numerical_grad(lambda inp: F.gelu(inp, approximate="tanh"), x.detach())
    save_case(
        "gelu_grad",
        {
            "input": x.detach().float(),
            "grad_analytical": grad_analytical.float(),
            "grad_numerical": grad_numerical.float(),
        },
    )


def case_softmax_ce_grad() -> None:
    torch.manual_seed(3)
    batch, classes = 8, 32
    logits = torch.randn(batch, classes, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, classes, (batch,))
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    grad_analytical = logits.grad.detach().clone()
    grad_numerical = numerical_grad(
        lambda x: F.cross_entropy(x, targets).unsqueeze(0), logits.detach()
    )
    save_case(
        "softmax_ce_grad",
        {
            "logits": logits.detach().float(),
            "targets": targets,
            "grad_analytical": grad_analytical.float(),
            "grad_numerical": grad_numerical.float(),
        },
    )


def case_mlp_training() -> None:
    torch.manual_seed(4)
    in_f, hidden_f, out_f = 32, 64, 16
    n_samples = 64
    x = torch.randn(n_samples, in_f, dtype=torch.float32)
    y = torch.randint(0, out_f, (n_samples,))
    w1 = (torch.randn(hidden_f, in_f, dtype=torch.float32) * 0.1).requires_grad_(True)
    b1 = torch.zeros(hidden_f, dtype=torch.float32).requires_grad_(True)
    w2 = (torch.randn(out_f, hidden_f, dtype=torch.float32) * 0.1).requires_grad_(True)
    b2 = torch.zeros(out_f, dtype=torch.float32).requires_grad_(True)
    lr = 0.01
    losses = []
    for _ in range(100):
        h = F.relu(F.linear(x, w1, b1))
        logits = F.linear(h, w2, b2)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            w1 -= lr * w1.grad
            b1 -= lr * b1.grad
            w2 -= lr * w2.grad
            b2 -= lr * b2.grad
        w1.grad = None
        b1.grad = None
        w2.grad = None
        b2.grad = None
    save_case(
        "mlp_training",
        {
            "loss_curve": torch.tensor(losses, dtype=torch.float32),
            "final_loss": torch.tensor([losses[-1]], dtype=torch.float32),
        },
    )
    print(f"    MLP final loss: {losses[-1]:.4f}")


def case_chain_rule() -> None:
    torch.manual_seed(5)
    hidden = 64
    x = torch.randn(4, hidden, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(hidden, hidden, dtype=torch.float64) * 0.1
    weight_rms = torch.ones(hidden, dtype=torch.float64)

    def chain_fn(inp):
        norm_var = inp.pow(2).mean(-1, keepdim=True)
        normed = inp * torch.rsqrt(norm_var + 1e-6) * weight_rms
        lin = F.linear(normed, weight)
        return F.relu(lin)

    out = chain_fn(x)
    out.sum().backward()
    grad_analytical = x.grad.detach().clone()
    grad_numerical = numerical_grad(chain_fn, x.detach())
    save_case(
        "chain_rule",
        {
            "input": x.detach().float(),
            "weight": weight.float(),
            "weight_rms": weight_rms.float(),
            "grad_analytical": grad_analytical.float(),
            "grad_numerical": grad_numerical.float(),
        },
    )


def case_broadcast_grad() -> None:
    torch.manual_seed(6)
    a = torch.randn(4, 1, 32, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 8, 32, dtype=torch.float64, requires_grad=True)
    out = (a + b).sum()
    out.backward()
    save_case(
        "broadcast_grad",
        {
            "a": a.detach().float(),
            "b": b.detach().float(),
            "grad_a": a.grad.detach().float(),
            "grad_b": b.grad.detach().float(),
        },
    )


def main() -> None:
    print("Generating autograd reference data...")
    case_matmul_grad()
    case_rms_norm_grad()
    case_gelu_grad()
    case_softmax_ce_grad()
    case_mlp_training()
    case_chain_rule()
    case_broadcast_grad()
    print("Done.")


if __name__ == "__main__":
    main()
