"""Reference data for synaptix FLUX MMDiT (FluxTransformer2DModel) — CUDA bf16.

Transformer 23GB bf16 (f32=46GB не влезает в 24GB) → reference и валидация в bf16.
Малый вход: img-сетка 8x8=64 токена, txt=32 токена. Дампим входы + выход +
ПРОМЕЖУТОЧНЫЕ (через forward-hooks: x_embedder, time_text_embed, context_embedder,
transformer_blocks[0], single_transformer_blocks[0]) — для локализации багов в bf16.

Запуск: scripts/reference/.venv/bin/python scripts/reference/gen_flux_transformer.py
"""

import pathlib

import torch
from diffusers import FluxTransformer2DModel

FLUX = "models/black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = pathlib.Path("tests/reference_data/flux_transformer")
IMG_H, IMG_W = 32, 32       # 1024 img-токена (проверка scale-зависимости)
TXT_SEQ = 512
DEV = "cuda"
DT = torch.bfloat16


def save_case(name, tensors):
    from safetensors.torch import save_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path))
    print(f"  saved {path} ({list(tensors.keys())})")


def latent_image_ids(h, w):
    ids = torch.zeros(h, w, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(h)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w)[None, :]
    return ids.reshape(h * w, 3)


def main():
    model = FluxTransformer2DModel.from_pretrained(f"{FLUX}/transformer", torch_dtype=DT).to(DEV)
    model.eval()
    c = model.config
    print("layers", c.num_layers, "single", c.num_single_layers, "heads",
          c.num_attention_heads, "head_dim", c.attention_head_dim, "guidance", c.guidance_embeds)

    g = torch.Generator(device="cpu").manual_seed(0)
    img_seq = IMG_H * IMG_W
    hs = torch.randn(1, img_seq, c.in_channels, generator=g, dtype=torch.float32).to(DEV, DT)
    ehs = torch.randn(1, TXT_SEQ, c.joint_attention_dim, generator=g, dtype=torch.float32).to(DEV, DT) * 0.5
    pooled = torch.randn(1, c.pooled_projection_dim, generator=g, dtype=torch.float32).to(DEV, DT)
    timestep = torch.tensor([0.5], dtype=torch.float32).to(DEV, DT)   # доля [0,1]
    guidance = torch.tensor([3.5], dtype=torch.float32).to(DEV)        # f32
    img_ids = latent_image_ids(IMG_H, IMG_W).to(DEV, DT)
    txt_ids = torch.zeros(TXT_SEQ, 3).to(DEV, DT)

    inter = {}
    def hook(name):
        def f(_m, _inp, out):
            if isinstance(out, tuple):
                for i, o in enumerate(out):
                    if torch.is_tensor(o):
                        inter[f"{name}_{i}"] = o.detach().float().cpu()
            elif torch.is_tensor(out):
                inter[name] = out.detach().float().cpu()
        return f
    model.x_embedder.register_forward_hook(hook("x_emb"))
    model.context_embedder.register_forward_hook(hook("ctx_emb"))
    model.time_text_embed.register_forward_hook(hook("temb"))
    model.transformer_blocks[0].register_forward_hook(hook("db0"))
    model.single_transformer_blocks[0].register_forward_hook(hook("sb0"))

    with torch.no_grad():
        out = model(
            hidden_states=hs,
            encoder_hidden_states=ehs,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]
    print("out", tuple(out.shape), "range", float(out.min()), float(out.max()))

    save_case("io", {
        "hidden_states": hs, "encoder_hidden_states": ehs, "pooled": pooled,
        "timestep": timestep, "guidance": guidance,
        "img_ids": img_ids, "txt_ids": txt_ids,
        "out": out.float(),
    })
    save_case("inter", inter)


if __name__ == "__main__":
    main()
