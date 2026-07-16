"""Reference SafeTensors для synaptix-nn/diffusion.

Run:
    scripts/reference/.venv/bin/python scripts/reference/gen_nn_diffusion.py

Покрытие — все 7 раннее заглушенных модулей:

- CFG: `uncond + scale · (cond − uncond)`.
- PAG: `cond + scale · (cond − perturbed)`.
- APG: stateless orthogonal-only (rescale ‖diff‖ ≤ norm_threshold + проекция
  компоненты вдоль cond).
- ControlNet: `x + conditioning_scale · proj(control)`.
- T2I-Adapter: `x + scale · proj(condition)`.
- IP-Adapter: `x + scale · broadcast(proj(image_emb))`.
- GLIGEN: `x + tanh(gate) · scale · broadcast(mean(box_proj(boxes) + entity_proj(entity_emb), dim=N))`.
"""

import pathlib
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_diffusion")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_cfg() -> None:
    torch.manual_seed(900)
    cond = torch.randn(2, 4, 8)
    uncond = torch.randn(2, 4, 8)
    scale = 7.5
    out = uncond + scale * (cond - uncond)
    save_case("cfg", {"cond": cond, "uncond": uncond, "output": out})


def case_pag() -> None:
    torch.manual_seed(901)
    cond = torch.randn(2, 4, 8)
    perturbed = torch.randn(2, 4, 8)
    scale = 3.0
    out = cond + scale * (cond - perturbed)
    save_case("pag", {"cond": cond, "perturbed": perturbed, "output": out})


def apg_orthogonal(cond, uncond, scale=7.5, norm_threshold=2.5, eps=1e-8):
    diff = cond - uncond
    norm = diff.flatten().norm(p=2)
    if norm.item() > norm_threshold and norm.item() > 0:
        diff = diff * (norm_threshold / norm)
    dot = (diff * cond).sum()
    cond_sq = (cond * cond).sum()
    denom = max(cond_sq.item(), eps)
    parallel = (dot / denom) * cond
    ortho = diff - parallel
    return cond + scale * ortho


def case_apg() -> None:
    torch.manual_seed(902)
    cond = torch.randn(1, 3, 4) * 0.5
    uncond = torch.randn(1, 3, 4) * 0.5
    out = apg_orthogonal(cond, uncond, scale=7.5, norm_threshold=2.5)
    save_case("apg", {"cond": cond, "uncond": uncond, "output": out})


def case_apg_rescale_active() -> None:
    torch.manual_seed(903)
    # большая разница → rescale активен.
    cond = torch.randn(1, 3, 4) * 2.0
    uncond = torch.randn(1, 3, 4) * 2.0
    out = apg_orthogonal(cond, uncond, scale=4.0, norm_threshold=1.0)
    save_case("apg_rescale_active", {"cond": cond, "uncond": uncond, "output": out})


def case_controlnet() -> None:
    torch.manual_seed(904)
    in_channels, hidden, scale = 6, 8, 0.75
    proj_w = torch.randn(hidden, in_channels) * 0.1
    proj_b = torch.randn(hidden) * 0.05
    x = torch.randn(2, 5, hidden)
    control = torch.randn(2, 5, in_channels)
    projected = F.linear(control, proj_w, proj_b)
    out = x + scale * projected
    save_case("controlnet", {
        "proj_w": proj_w, "proj_b": proj_b,
        "x": x, "control": control, "output": out,
    })


def case_t2i_adapter() -> None:
    torch.manual_seed(905)
    in_channels, hidden, scale = 4, 8, 1.2
    proj_w = torch.randn(hidden, in_channels) * 0.1
    proj_b = torch.randn(hidden) * 0.05
    x = torch.randn(2, 5, hidden)
    cond = torch.randn(2, 5, in_channels)
    projected = F.linear(cond, proj_w, proj_b)
    out = x + scale * projected
    save_case("t2i_adapter", {
        "proj_w": proj_w, "proj_b": proj_b,
        "x": x, "condition": cond, "output": out,
    })


def case_ip_adapter() -> None:
    torch.manual_seed(906)
    img_dim, hidden, scale = 12, 8, 0.6
    proj_w = torch.randn(hidden, img_dim) * 0.1
    proj_b = torch.randn(hidden) * 0.05
    x = torch.randn(2, 5, hidden)
    image_emb = torch.randn(2, img_dim)
    projected = F.linear(image_emb, proj_w, proj_b)  # [B, hidden]
    expanded = projected.unsqueeze(1).expand(-1, x.shape[1], -1).contiguous()
    out = x + scale * expanded
    save_case("ip_adapter", {
        "proj_w": proj_w, "proj_b": proj_b,
        "x": x, "image_emb": image_emb, "output": out,
    })


def case_gligen() -> None:
    torch.manual_seed(907)
    entity_dim, hidden, num_entities, scale = 6, 8, 3, 1.5
    e_w = torch.randn(hidden, entity_dim) * 0.1
    e_b = torch.randn(hidden) * 0.05
    b_w = torch.randn(hidden, 4) * 0.1
    b_b = torch.randn(hidden) * 0.05
    gate = torch.tensor([0.4])

    x = torch.randn(2, 5, hidden)
    boxes = torch.randn(2, num_entities, 4)
    entity_emb = torch.randn(2, num_entities, entity_dim)

    e = F.linear(entity_emb, e_w, e_b)
    p = F.linear(boxes, b_w, b_b)
    grounded = e + p
    pooled = grounded.mean(dim=1, keepdim=True)         # [B, 1, hidden]
    pooled_b = pooled.expand(-1, x.shape[1], -1).contiguous()
    g = torch.tanh(gate)
    out = x + g.item() * scale * pooled_b

    save_case("gligen", {
        "entity_w": e_w, "entity_b": e_b,
        "box_w": b_w, "box_b": b_b,
        "gate": gate,
        "x": x, "boxes": boxes, "entity_emb": entity_emb,
        "output": out,
    })


def main() -> None:
    print("Generating nn-diffusion reference data...")
    case_cfg()
    case_pag()
    case_apg()
    case_apg_rescale_active()
    case_controlnet()
    case_t2i_adapter()
    case_ip_adapter()
    case_gligen()
    print("Done.")


if __name__ == "__main__":
    main()
