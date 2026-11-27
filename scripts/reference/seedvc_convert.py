"""Веса Seed-VC v2 (.pth/.bin/.pt) → каталог компонентов safetensors для `synaptix convert`.

python -I seedvc_convert.py OUT_DIR
    затем: synaptix convert OUT_DIR seed-vc-v2.syn --arch seed_vc_v2 --id seed-vc-v2

Компоненты (подкаталог/model.safetensors, имена тензоров — как в оригинале, f32):
  ar            — трансформер манеры + эмбеддинги узких токенов (`length_regulator.embedding.weight`)
                  + таблица RoPE `model.freqs_cis` (эталон считает её в bf16; кладём готовой);
  cfm           — DiT (`estimator.*`) + регулятор длины (`length_regulator.*`);
  astral_wide, astral_narrow — ConvNeXt + `quantizer.project_in` (BSQ 2048 и 32);
  hubert        — HuBERT-large, первые 18 слоёв (+ hubert/config.json с num_hidden_layers = 18);
  campplus      — CAM++ (вектор диктора);
  bigvgan       — генератор BigVGAN v2 22 кГц (фильтры активаций не кладём — считаются в движке).
Неиспользуемые при инференсе тензоры отброшены (перечень печатается).
"""
import json
import os
import sys
import torch
from safetensors.torch import save_file

HF = os.path.expanduser("~/Storage/Нейросети/Модели/_hf")
MAIN = os.path.join(HF, "Seed-VC")
EXT = os.path.join(HF, "Seed-VC-ext")
out = sys.argv[1]
HUBERT_LAYERS = 18


def strip(sd, prefix="module."):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}


def save(name, sd, dropped):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    sd = {k: v.to(torch.float32).contiguous() for k, v in sd.items()}
    save_file(sd, os.path.join(d, "model.safetensors"))
    n = sum(v.numel() for v in sd.values())
    print(f"{name}: {len(sd)} тензоров, {n / 1e6:.1f} М параметров; отброшено {len(dropped)}: {sorted(set(dropped))[:8]}")


def rope_table(seq_len, n_elem, base=10000.0):
    """precompute_freqs_cis эталона (modules/v2/ar.py), с округлением до bf16."""
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    freqs = torch.outer(torch.arange(seq_len), freqs)
    c = torch.polar(torch.ones_like(freqs), freqs)
    return torch.stack([c.real, c.imag], dim=-1).to(torch.bfloat16)


# ar
ck = torch.load(os.path.join(MAIN, "v2/ar_base.pth"), map_location="cpu", weights_only=True)["net"]
ar, dropped = {}, []
for k, v in strip(ck["ar"]).items():
    if k.startswith(("style_in.", "model.codebook_")):
        dropped.append(k.split(".")[0] + "." + k.split(".")[1])
    else:
        ar[k] = v
for k, v in strip(ck["length_regulator"]).items():
    if k == "embedding.weight":
        ar["length_regulator." + k] = v
    else:
        dropped.append("length_regulator." + k)
ar["model.freqs_cis"] = rope_table(4096, 64)
save("ar", ar, dropped)

# cfm
ck = torch.load(os.path.join(MAIN, "v2/cfm_small.pth"), map_location="cpu", weights_only=True)["net"]
cfm, dropped = {}, []
for k, v in strip(ck["cfm"]).items():
    if k.startswith(("estimator.transformer.causal_mask", "estimator.content_mask_embedder", "estimator.skip_linear", "estimator.x_embedder")):
        dropped.append(".".join(k.split(".")[:3]))
    else:
        cfm[k] = v
for k, v in strip(ck["length_regulator"]).items():
    if k == "mask_token":
        dropped.append("length_regulator." + k)
    else:
        cfm["length_regulator." + k] = v
save("cfm", cfm, dropped)

# astral
for kind, name in (("bsq2048", "astral_wide"), ("bsq32", "astral_narrow")):
    sd = torch.load(os.path.join(EXT, f"ASTRAL-quantization/{kind}/{kind}_light.pth"), map_location="cpu", weights_only=True)
    keep = {k: v for k, v in sd.items() if k.startswith("encoder.") or k.startswith("quantizer.project_in.")}
    save(name, keep, [k for k in sd if k not in keep])

# hubert
sd = torch.load(os.path.join(EXT, "hubert-large-ll60k/pytorch_model.bin"), map_location="cpu", weights_only=True)
keep, dropped = {}, []
for k, v in sd.items():
    parts = k.split(".")
    if k == "masked_spec_embed" or (k.startswith("encoder.layers.") and int(parts[2]) >= HUBERT_LAYERS):
        dropped.append(".".join(parts[:3]))
    else:
        keep[k] = v
save("hubert", keep, dropped)
cfg = json.load(open(os.path.join(EXT, "hubert-large-ll60k/config.json")))
cfg["num_hidden_layers"] = HUBERT_LAYERS
json.dump(cfg, open(os.path.join(out, "hubert", "config.json"), "w"), indent=1)

# campplus
p = os.path.join(EXT, "campplus/campplus_cn_common.bin")
sd = torch.load(p, map_location="cpu", weights_only=True)
keep = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}
save("campplus", keep, [k for k in sd if k not in keep])

# bigvgan
sd = torch.load(os.path.join(EXT, "bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt"), map_location="cpu", weights_only=True)["generator"]
keep = {k: v for k, v in sd.items() if not k.endswith(".filter")}
save("bigvgan", keep, ["*.filter"] if len(keep) != len(sd) else [])

json.dump({
    "_class_name": "SeedVcV2",
    "source": "Plachta/Seed-VC (v2/ar_base.pth, v2/cfm_small.pth), Plachta/ASTRAL-quantization (bsq32, bsq2048), facebook/hubert-large-ll60k, funasr/campplus, nvidia/bigvgan_v2_22khz_80band_256x",
    "sample_rate": 22050, "hop": 256, "n_mels": 80,
    "license": "GPL-3.0 (Seed-VC, ASTRAL); Apache-2.0 (HuBERT, CAM++); MIT (BigVGAN)",
}, open(os.path.join(out, "model_index.json"), "w"), ensure_ascii=False, indent=1)
print("готово:", out)
