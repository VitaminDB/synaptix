"""Веса Seed-VC v1 (whisper-small wavenet) → каталог компонентов safetensors для `synaptix convert`.

python -I seedvc_v1_convert.py OUT_DIR
    затем: synaptix convert OUT_DIR seed-vc-v1.syn --arch seed_vc_v1 --id seed-vc-v1

Компоненты (f32): cfm — DiT (`estimator.*`, weight-norm свёрнут в `weight`) и регулятор длины
(`length_regulator.*`); whisper — энкодер whisper-small (`model.encoder.*`) + whisper/config.json;
campplus, bigvgan — как в бандле v2.
"""
import json
import os
import shutil
import sys
import torch
from safetensors.torch import save_file, load_file

HF = os.path.expanduser("~/Storage/Нейросети/Модели/_hf")
MAIN = os.path.join(HF, "Seed-VC")
EXT = os.path.join(HF, "Seed-VC-ext")
out = sys.argv[1]


def strip(sd, prefix="module."):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}


def save(name, sd, dropped):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    sd = {k: v.to(torch.float32).contiguous() for k, v in sd.items()}
    save_file(sd, os.path.join(d, "model.safetensors"))
    print(f"{name}: {len(sd)} тензоров, {sum(v.numel() for v in sd.values()) / 1e6:.1f} М; отброшено {len(dropped)}: {sorted(set(dropped))[:8]}")


ck = torch.load(os.path.join(MAIN, "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"), map_location="cpu", weights_only=True)["net"]
src = strip(ck["cfm"])
cfm, dropped = {}, []
for k, v in src.items():
    if k.startswith(("estimator.input_pos", "estimator.cond_embedder", "estimator.f0_embedder", "estimator.content_mask_embedder", "estimator.x_embedder")):
        dropped.append(".".join(k.split(".")[:2]))
    elif k.endswith(".weight_g"):
        base = k[: -len("_g")]
        g, w = v, src[base + "_v"]
        norm = w.reshape(w.shape[0], -1).norm(dim=1).reshape([-1] + [1] * (w.dim() - 1))
        cfm[base] = w * (g.reshape(norm.shape) / norm)
    elif k.endswith(".weight_v"):
        pass
    else:
        cfm[k] = v
for k, v in strip(ck["length_regulator"]).items():
    if k in ("mask_token", "embedding.weight"):
        dropped.append("length_regulator." + k)
    else:
        cfm["length_regulator." + k] = v
dropped += ["style_encoder.*", "vq.*"]
save("cfm", cfm, dropped)

sd = load_file(os.path.join(EXT, "whisper-small/model.safetensors"))
enc = {k: v for k, v in sd.items() if k.startswith("model.encoder.")}
save("whisper", enc, ["model.decoder.*", "proj_out"] if len(enc) != len(sd) else [])
shutil.copy(os.path.join(EXT, "whisper-small/config.json"), os.path.join(out, "whisper", "config.json"))

sd = torch.load(os.path.join(EXT, "campplus/campplus_cn_common.bin"), map_location="cpu", weights_only=True)
keep = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}
save("campplus", keep, [k for k in sd if k not in keep])

sd = torch.load(os.path.join(EXT, "bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt"), map_location="cpu", weights_only=True)["generator"]
keep = {k: v for k, v in sd.items() if not k.endswith(".filter")}
save("bigvgan", keep, ["*.filter"])

json.dump({
    "_class_name": "SeedVcV1",
    "source": "Plachta/Seed-VC (DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth), openai/whisper-small (энкодер), funasr/campplus, nvidia/bigvgan_v2_22khz_80band_256x",
    "sample_rate": 22050, "hop": 256, "n_mels": 80,
    "license": "GPL-3.0 (Seed-VC); Apache-2.0 (Whisper, CAM++); MIT (BigVGAN)",
}, open(os.path.join(out, "model_index.json"), "w"), ensure_ascii=False, indent=1)
print("готово:", out)
