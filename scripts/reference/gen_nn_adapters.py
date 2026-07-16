"""Reference SafeTensors для synaptix-nn/adapters (PEFT-семейство).

Run:
    scripts/reference/.venv/bin/python scripts/reference/gen_nn_adapters.py

Покрытие — все 8 раннее заглушенных адаптеров:

- QLoRA: эквивалент LoRA над уже dequantized-базой.
- VeRA: shared frozen `A`/`B` + per-module diagonal `λ_d`, `λ_b`.
- OFT: full Cayley orthogonal `R = (I+Q)(I−Q)^−1` поверх `W`.
- BOFT: block-diagonal Cayley orthogonal по `num_blocks` блокам.
- GaLore: inference-stub (forward = base.linear; ранг хранится для loader).
- LoReFT: `h' = h + R^T · ((W·h + b) − R·h)`.
- PromptTuning: prepend `[num_tokens, hidden]` к `[B, T, H]`.
- P-TuningV2: reparam MLP на префикс-эмбеддингах → per-layer (K, V).
"""

import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_adapters")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_qlora() -> None:
    torch.manual_seed(800)
    in_f, out_f, r = 8, 8, 4
    alpha = 8.0
    scaling = alpha / r
    base_w = torch.randn(out_f, in_f)
    a = torch.randn(r, in_f) * 0.1
    b = torch.randn(out_f, r) * 0.1
    x = torch.randn(2, in_f)
    out = F.linear(x, base_w) + F.linear(F.linear(x, a), b) * scaling
    save_case("qlora", {"base_w": base_w, "lora_a": a, "lora_b": b, "x": x, "output": out})


def case_vera() -> None:
    torch.manual_seed(801)
    in_f, out_f, r = 8, 8, 4
    base_w = torch.randn(out_f, in_f)
    a_shared = torch.randn(r, in_f)
    b_shared = torch.randn(out_f, r)
    lambda_d = torch.rand(r) + 0.5
    lambda_b = torch.rand(out_f) - 0.5
    x = torch.randn(3, in_f)

    base_out = F.linear(x, base_w)
    ax = F.linear(x, a_shared)
    ax_scaled = ax * lambda_d
    bx = F.linear(ax_scaled, b_shared)
    bx_scaled = bx * lambda_b
    out = base_out + bx_scaled
    save_case("vera", {
        "base_w": base_w, "a_shared": a_shared, "b_shared": b_shared,
        "lambda_d": lambda_d, "lambda_b": lambda_b,
        "x": x, "output": out,
    })


def cayley_orthogonal(q_raw: torch.Tensor) -> torch.Tensor:
    n = q_raw.shape[0]
    q = q_raw - q_raw.t()
    eye = torch.eye(n, dtype=q.dtype)
    return (eye + q) @ torch.linalg.inv(eye - q)


def case_oft() -> None:
    torch.manual_seed(802)
    in_f, out_f = 8, 8
    base_w = torch.randn(out_f, in_f)
    q_raw = torch.randn(out_f, out_f) * 0.05
    r = cayley_orthogonal(q_raw)
    w_eff = r @ base_w
    x = torch.randn(2, in_f)
    out = F.linear(x, w_eff)
    save_case("oft", {
        "base_w": base_w, "q_raw": q_raw, "r_matrix": r,
        "x": x, "output": out,
    })


def case_boft() -> None:
    torch.manual_seed(803)
    in_f, out_f, num_blocks = 8, 8, 2
    block_size = out_f // num_blocks
    base_w = torch.randn(out_f, in_f)
    q_blocks = [torch.randn(block_size, block_size) * 0.05 for _ in range(num_blocks)]
    r_full = torch.zeros(out_f, out_f)
    for k, q_raw in enumerate(q_blocks):
        r_k = cayley_orthogonal(q_raw)
        off = k * block_size
        r_full[off:off + block_size, off:off + block_size] = r_k
    w_eff = r_full @ base_w
    x = torch.randn(2, in_f)
    out = F.linear(x, w_eff)
    save_case("boft", {
        "base_w": base_w,
        "q_block0": q_blocks[0], "q_block1": q_blocks[1],
        "r_matrix": r_full,
        "x": x, "output": out,
    })


def case_galore() -> None:
    torch.manual_seed(804)
    in_f, out_f = 8, 8
    base_w = torch.randn(out_f, in_f)
    x = torch.randn(2, in_f)
    out = F.linear(x, base_w)
    save_case("galore", {"base_w": base_w, "x": x, "output": out})


def case_reft() -> None:
    torch.manual_seed(805)
    hidden = 8
    r = 3
    r_proj = torch.randn(r, hidden)
    w = torch.randn(r, hidden)
    b = torch.randn(r) * 0.1
    h = torch.randn(2, 4, hidden)

    rh = F.linear(h, r_proj)
    wh = F.linear(h, w, b)
    diff = wh - rh
    delta = diff @ r_proj
    out = h + delta
    save_case("reft", {
        "r_proj": r_proj, "w": w, "b": b,
        "h": h, "output": out,
    })


def case_prompt_tuning() -> None:
    torch.manual_seed(806)
    batch, t, h = 2, 5, 8
    num_tokens = 3
    soft = torch.randn(num_tokens, h) * 0.02
    x = torch.randn(batch, t, h)
    sp_b = soft.unsqueeze(0).expand(batch, -1, -1)
    out = torch.cat([sp_b, x], dim=1)
    save_case("prompt_tuning", {"soft_prompts": soft, "x": x, "output": out})


def case_p_tuning_v2() -> None:
    torch.manual_seed(807)
    prefix_len, hidden, num_layers = 4, 6, 3
    emb = torch.randn(prefix_len, hidden) * 0.02
    reparam_w = torch.randn(num_layers * 2 * hidden, hidden) * 0.05
    proj = F.linear(emb, reparam_w)                      # [prefix_len, L*2*H]
    four = proj.view(prefix_len, num_layers, 2, hidden)  # [P, L, 2, H]
    full = four.permute(1, 2, 0, 3).contiguous()         # [L, 2, P, H]
    layer = 1
    layer_k = full[layer, 0].clone().contiguous()
    layer_v = full[layer, 1].clone().contiguous()
    save_case("p_tuning_v2", {
        "embeddings": emb, "reparam_w": reparam_w,
        "full": full,
        "layer_k": layer_k, "layer_v": layer_v,
    })


def main() -> None:
    print("Generating nn-adapters reference data...")
    case_qlora()
    case_vera()
    case_oft()
    case_boft()
    case_galore()
    case_reft()
    case_prompt_tuning()
    case_p_tuning_v2()
    print("Done.")


if __name__ == "__main__":
    main()
