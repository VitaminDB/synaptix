"""Reference SafeTensors для synaptix-nn/vision (CLIP-style ViT pattern).

Run:
    python scripts/reference/gen_nn_vision.py

Reference воспроизводит математику CLIP vision encoder: patchify →
ViT-blocks → final LN → CLS-pool (первый токен) → visual_projection.
Полные модели (transformers.CLIPVisionModel, timm.create_model) с
pretrained весами — Phase O.

Свёрточные backbones (ResNet, EfficientNet) — ref в Phase O (нужны
BatchNorm running stats + dropout calibration).
"""

import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/nn_vision")


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def vit_block(x, n1_w, n1_b, n2_w, n2_b, q_w, k_w, v_w, o_w, ff1_w, ff1_b, ff2_w, ff2_b, num_heads):
    h = F.layer_norm(x, (x.shape[-1],), n1_w, n1_b, 1e-6)
    head_dim = h.shape[-1] // num_heads
    b, s, _ = h.shape
    q = F.linear(h, q_w).reshape(b, s, num_heads, head_dim).permute(0, 2, 1, 3)
    k = F.linear(h, k_w).reshape(b, s, num_heads, head_dim).permute(0, 2, 1, 3)
    v = F.linear(h, v_w).reshape(b, s, num_heads, head_dim).permute(0, 2, 1, 3)
    scale = 1.0 / (head_dim ** 0.5)
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    attn = (attn @ v).permute(0, 2, 1, 3).reshape(b, s, h.shape[-1])
    attn_out = F.linear(attn, o_w)
    x = x + attn_out
    h2 = F.layer_norm(x, (x.shape[-1],), n2_w, n2_b, 1e-6)
    mlp = F.linear(F.gelu(F.linear(h2, ff1_w, ff1_b), approximate="tanh"), ff2_w, ff2_b)
    return x + mlp


def case_clip_vision_minimal():
    """ViT с 1 блоком + CLIP cls_pool + visual_projection."""
    torch.manual_seed(700)
    in_channels, patch_size, image_size = 3, 4, 16
    hidden, num_heads, num_patches = 32, 4, (image_size // patch_size) ** 2
    embed_dim = 16

    patch_w = torch.randn(hidden, in_channels * patch_size * patch_size)
    patch_b = torch.randn(hidden)
    n1_w = torch.rand(hidden) + 0.5
    n1_b = torch.randn(hidden) * 0.1
    n2_w = torch.rand(hidden) + 0.5
    n2_b = torch.randn(hidden) * 0.1
    q_w = torch.randn(hidden, hidden) * 0.1
    k_w = torch.randn(hidden, hidden) * 0.1
    v_w = torch.randn(hidden, hidden) * 0.1
    o_w = torch.randn(hidden, hidden) * 0.1
    ffn_dim = hidden * 4
    ff1_w = torch.randn(ffn_dim, hidden) * 0.1
    ff1_b = torch.randn(ffn_dim) * 0.05
    ff2_w = torch.randn(hidden, ffn_dim) * 0.1
    ff2_b = torch.randn(hidden) * 0.05
    norm_w = torch.rand(hidden) + 0.5
    norm_b = torch.randn(hidden) * 0.1
    proj_w = torch.randn(embed_dim, hidden) * 0.1

    image = torch.randn(2, in_channels, image_size, image_size)
    # Patchify
    b, c, h, w = image.shape
    p = patch_size
    nh, nw = h // p, w // p
    patches = image.reshape(b, c, nh, p, nw, p).permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.reshape(b, nh * nw, c * p * p)
    # Embed
    tokens = F.linear(patches, patch_w, patch_b)
    # Block
    tokens = vit_block(
        tokens, n1_w, n1_b, n2_w, n2_b,
        q_w, k_w, v_w, o_w, ff1_w, ff1_b, ff2_w, ff2_b, num_heads,
    )
    # Final LN
    tokens = F.layer_norm(tokens, (hidden,), norm_w, norm_b, 1e-6)
    # CLS-pool (первый токен)
    pooled = tokens[:, 0, :]
    # Visual projection
    output = F.linear(pooled, proj_w)

    save_case("clip_vision_minimal", {
        "image": image,
        "patch_w": patch_w, "patch_b": patch_b,
        "n1_w": n1_w, "n1_b": n1_b, "n2_w": n2_w, "n2_b": n2_b,
        "q_w": q_w, "k_w": k_w, "v_w": v_w, "o_w": o_w,
        "ff1_w": ff1_w, "ff1_b": ff1_b, "ff2_w": ff2_w, "ff2_b": ff2_b,
        "norm_w": norm_w, "norm_b": norm_b,
        "proj_w": proj_w,
        "output": output,
    })


def main():
    print("Generating nn-vision reference data...")
    case_clip_vision_minimal()
    print("Done.")


if __name__ == "__main__":
    main()
