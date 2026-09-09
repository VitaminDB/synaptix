"""Эталон башни зрения Gemma-4: препроцессинг + tower + проекция в LM.

Пишет в out/ бинарники float32 (little-endian), которые читает Rust-тест.
Запуск: muse_ref_venv/bin/python gemma4_vision_ref.py <hf_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
)

hf_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

cfg = Gemma4Config.from_pretrained(hf_dir)
torch.set_grad_enabled(False)

# ── детерминированная картинка ──────────────────────────────────────────────
H, W = 96, 144  # кратно pooling_kernel_size * patch_size = 48 → resize не тронет
yy, xx = np.mgrid[0:H, 0:W]
rgb = np.stack(
    [
        (xx * 255 // (W - 1)).astype(np.uint8),
        (yy * 255 // (H - 1)).astype(np.uint8),
        (((xx + yy) * 255) // (H + W - 2)).astype(np.uint8),
    ],
    axis=-1,
)
img = Image.fromarray(rgb, mode="RGB")
img.save(out_dir / "image.png")

# Препроцессинг руками (torchvision в окружении нет, а resize тут не нужен:
# стороны уже кратны pooling_kernel_size * patch_size).
PATCH = cfg.vision_config.patch_size
POOL = cfg.vision_config.pooling_kernel_size
MAX_SOFT = 280
MAX_PATCHES = MAX_SOFT * POOL**2

chw = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0
ph, pw = H // PATCH, W // PATCH
patched = chw.reshape(3, ph, PATCH, pw, PATCH).permute(1, 3, 2, 4, 0).reshape(ph * pw, -1)
grid = torch.meshgrid(torch.arange(pw), torch.arange(ph), indexing="xy")
real_positions = torch.stack(grid, dim=-1).reshape(ph * pw, 2)

pixel_small = patched[None]
positions_small = real_positions[None]
pixel_values = torch.nn.functional.pad(patched, (0, 0, 0, MAX_PATCHES - patched.shape[0]))[None]
position_ids = torch.nn.functional.pad(
    real_positions, (0, 0, 0, MAX_PATCHES - patched.shape[0]), value=-1
)[None]
print("patches", patched.shape, "→ padded", pixel_values.shape)

# ── веса башни ──────────────────────────────────────────────────────────────
tower = Gemma4VisionModel(cfg.vision_config).to(torch.float32).eval()
embedder = Gemma4MultimodalEmbedder(cfg.vision_config, cfg.text_config).to(torch.float32).eval()

want_tower = {f"model.vision_tower.{k}": k for k in tower.state_dict()}
want_emb = {f"model.embed_vision.{k}": k for k in embedder.state_dict()}
loaded_tower, loaded_emb = {}, {}
for shard in sorted(hf_dir.glob("*.safetensors")):
    with safe_open(shard, framework="pt") as f:
        for name in f.keys():
            if name in want_tower:
                loaded_tower[want_tower[name]] = f.get_tensor(name).to(torch.float32)
            elif name in want_emb:
                loaded_emb[want_emb[name]] = f.get_tensor(name).to(torch.float32)
missing_t = set(tower.state_dict()) - set(loaded_tower)
missing_e = set(embedder.state_dict()) - set(loaded_emb)
print("не найдено в башне:", sorted(missing_t))
print("не найдено в проекции:", sorted(missing_e))
tower.load_state_dict(loaded_tower, strict=True)
embedder.load_state_dict(loaded_emb, strict=True)

soft_padded = tower(pixel_values=pixel_values, pixel_position_ids=position_ids).last_hidden_state
soft = tower(pixel_values=pixel_small, pixel_position_ids=positions_small).last_hidden_state
print("padded vs unpadded max|Δ| =", (soft_padded - soft).abs().max().item())
projected = embedder(soft)
print("soft", tuple(soft.shape), "projected", tuple(projected.shape))


def dump(name, t):
    a = t.detach().to(torch.float32).contiguous().numpy()
    a.tofile(out_dir / f"{name}.f32")
    meta[name] = list(a.shape)


meta = {}
dump("pixel_values", pixel_small[0])
dump("soft_tokens", soft)
dump("projected", projected)
positions_small[0].to(torch.int32).numpy().tofile(out_dir / "position_ids.i32")
meta["position_ids"] = list(positions_small[0].shape)
meta["image_size"] = [H, W]
(out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
print("готово:", meta)
print("soft[0,:6]", soft[0, :6].tolist())
print("projected[0,:6]", projected[0, :6].tolist())
