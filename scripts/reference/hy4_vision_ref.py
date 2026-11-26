"""Эталон башни зрения Hy4-flash на CPU (по спецификации transformers HYV4VL).

  python hy4_vision_ref.py --model _hf/Hy4-flash --dir dump/
Читает patches.f32 и grid.txt (выход примера hy4_vision), считает башню и perceive
по BF16-весам и пишет ref.f32 [N, 3072]; печатает сравнение с features.f32.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dir", required=True)
a = ap.parse_args()
idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
files = {}
def t(name):
    f = idx[name]
    if f not in files:
        files[f] = safe_open(os.path.join(a.model, f), "pt")
    return files[f].get_tensor(name)

gh, gw = map(int, open(os.path.join(a.dir, "grid.txt")).read().split())
px = torch.from_numpy(np.fromfile(os.path.join(a.dir, "patches.f32"), np.float32).reshape(gh * gw, 768))
dt = torch.bfloat16
x = px.to(dt)
pw = t("vit.embeddings.patch_embedding.weight").reshape(1152, 768)
x = (x.float() @ pw.float().T + t("vit.embeddings.patch_embedding.bias").float()).to(dt)
P = t("vit.embeddings.position_embedding.weight")[1:].float().reshape(128, 128, 1152).permute(2, 0, 1)[None]
pos = F.interpolate(P, scale_factor=((gh + 0.1) / 128, (gw + 0.1) / 128), mode="bilinear", align_corners=False)
pos = pos[0].permute(1, 2, 0).reshape(gh * gw, 1152).to(dt)
x = (x.float() + pos.float()).to(dt)
def lin(v, k):
    return (v.float() @ t(k + ".weight").float().T + t(k + ".bias").float()).to(dt)
def ln(v, k):
    return F.layer_norm(v.float(), (1152,), t(k + ".weight").float(), t(k + ".bias").float(), 1e-5).to(dt)
n = gh * gw
for l in range(27):
    p = f"vit.layers.{l}"
    h = ln(x, p + ".input_layernorm")
    q, k, v = (lin(h, f"{p}.self_attn.{z}_proj").float().reshape(n, 16, 72).transpose(0, 1) for z in "qkv")
    o = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(n, 1152).to(dt)
    x = (x.float() + lin(o, p + ".self_attn.o_proj").float()).to(dt)
    h = ln(x, p + ".post_attention_layernorm")
    h = F.gelu(lin(h, p + ".mlp.dense_h_to_4h").float()).to(dt)
    x = (x.float() + lin(h, p + ".mlp.dense_4h_to_h").float()).to(dt)
    print("слой", l, flush=True)
xf = x.float()
y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(dt) * t("vit.perceive.norm.weight")
Y = y.float().reshape(gh, gw, 1152).permute(2, 0, 1)[None]
Z = F.conv2d(Y, t("vit.perceive.proj.0.weight").float(), t("vit.perceive.proj.0.bias").float(), stride=2)
Z = F.gelu(Z.to(dt).float())
U = F.conv2d(Z, t("vit.perceive.proj.2.weight").float(), t("vit.perceive.proj.2.bias").float())[0]  # [3072, r, c]
nl = t("vit.perceive.image_newline").float()[:, None, None].expand(3072, U.shape[1], 1)
U = torch.cat([U, nl], 2).permute(1, 2, 0).reshape(-1, 3072)
ref = U.numpy().astype(np.float32)
ref.tofile(os.path.join(a.dir, "ref.f32"))
got = np.fromfile(os.path.join(a.dir, "features.f32"), np.float32).reshape(ref.shape)
cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-30)
print("формы", ref.shape, "rel %.3e" % (np.linalg.norm(ref - got) / np.linalg.norm(ref)), "cos min %.5f mean %.5f" % (cos.min(), cos.mean()))
