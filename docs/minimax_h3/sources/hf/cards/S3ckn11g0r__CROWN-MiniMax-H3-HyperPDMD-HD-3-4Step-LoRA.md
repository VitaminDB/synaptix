# S3ckn11g0r/CROWN-MiniMax-H3-HyperPDMD-HD-3-4Step-LoRA

Источник: https://huggingface.co/S3ckn11g0r/CROWN-MiniMax-H3-HyperPDMD-HD-3-4Step-LoRA · скачано 2026-11-28
Загрузок: 2609 · лайков: 10 · базовая: ["MiniMaxAI/MiniMax-H3","Comfy-Org/MiniMax-H3"] · теги: lora,turbo,distillation,gguf,comfyui,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-11-23T16:26:07.000Z

---

---
license: apache-2.0
base_model:
  - MiniMaxAI/MiniMax-H3
  - Comfy-Org/MiniMax-H3
pipeline_tag: image-to-video
library_name: diffusers
tags:
  - minimax-h3
  - lora
  - turbo
  - acceleration
  - distillation
  - dmd
  - gguf
  - int8
  - comfyui
  - text-to-video
  - image-to-video
  - reference-to-video
  - audio-video
---

# ⚡ CROWN MiniMax-H3 HyperPDMD-HD (3–4 Step All-in-One Turbo & De-Quantization LoRA Suite)

**By NineCrowns (`[CROWN]`)** — Compact (`~345 MiB`) **3–4 Step Acceleration + Nyquist-HD Sharpness + Multi-Precision (`BF16` / `ConvRot INT8` / `GGUF Q4_K_M`) De-Quantization LoRA Suite** for **MiniMax-H3** (33B Joint Video + `32kHz` Stereo Audio DiT).

---

> *（顺手给国内显卡跑不动原版 H3 的朋友留个平时自用的纯原版底模在线台：支持最高 15 秒音画同步、多图参考与首尾帧，新号自带体验额度，不含本站自训微调权重，仅限中文环境自取：[h3xron.xyz](https://h3xron.xyz)）*

## 📦 Included Weights & Workflows

| File | Edition / Scope | Size | Rank | Energy Retained |
| :--- | :--- | :---: | :---: | :---: |
| `CROWN_MiniMax_H3_HyperPDMD_HD_FL2VA_3_4Step_r38.safetensors` | **`FL2VA / I2VA / T2VA` (Recommended Main)** | `346.7 MiB` | `avg_rank=38` | `99.96%` |
| `CROWN_MiniMax_H3_HyperPDMD_HD_Ref2VA_3_4Step_r38.safetensors` | **`Ref2VA` Multi-Ref & Lip-Sync (Recommended Main)** | `340.3 MiB` | `avg_rank=38` | `99.98%` |
| `full_rank64/CROWN_MiniMax_H3_HyperPDMD_HD_FL2VA_3_4Step_r64_full.safetensors` | `FL2VA / I2VA / T2VA` Uncompressed Master | `591.6 MiB` | `Rank 64` | `100.00%` |
| `full_rank64/CROWN_MiniMax_H3_HyperPDMD_HD_Ref2VA_3_4Step_r64_full.safetensors` | `Ref2VA` Uncompressed Master | `591.6 MiB` | `Rank 64` | `100.00%` |
| `workflows/*.json` | **6 Ready-to-Run ComfyUI `0.33.0+` Workflows** (`T2V`, `I2V/FL2V`, `R2V` @ `480P` & `768P`) | — | — | — |
| `showcases/*.mp4` | **15-Second (`362f @ 24fps` + `32kHz` Stereo Dialogue) Multi-Precision Comparisons** | — | — | — |

---

## 🚀 Key Highlights

1. **True 3–4 Step Generation up to 15 Seconds (`362` Frames @ `24fps` + `32kHz` Stereo Audio)**
   - Eliminates single-frame `I2V` freeze, pure `T2V` collapse, and the severe mid-clip (`5s–10s`) double-exposure ghosting/blur seen in `No-LoRA` 4-step runs (`6.5×–10×` higher Laplacian optical sharpness).
2. **One LoRA for All Base Precisions (`Pure BF16` + `ConvRot INT8` + `GGUF Q4_K_M`)**
   - Distilled on `8×NVIDIA B200` clusters with online student weight quantization perturbation (`45% GGUF Q4_K_M`, `35% Block-256 Hadamard ConvRot INT8`, `20% Pure BF16`) + Projected Distribution Matching Distillation (`HyperPDMD`) + 5-frame 3D-Conv `VideoVAE` spectral discriminator.
   - Runs under **`< 9 GiB` VRAM** on consumer GPUs with `UnetLoaderGGUFAdvanced (patch_dtype="target")` (`0.00 GiB` extra VRAM overhead!).
3. **100% Native ComfyUI Compatibility (`208/208` Layers)**
   - Targets all 208 `MiniMax-H3` DiT + `TokenRefiner` linear projections (`comfy.lora.load_lora = 208/208 = 100.0%`).

---

## 🛠️ Recommended ComfyUI `0.33.0+` Settings

- **Base Model**:
  - `< 9 GB VRAM`: `minimax-h3-{fl2va|ref2va}-pruned-Q4_K_M.gguf` (`UnetLoaderGGUFAdvanced`, `patch_dtype = "target"`)
  - `16–24 GB VRAM`: `minimax_h3_{fl2va|ref2va}_pruned_int8_convrot.safetensors`
  - `Workstation / Cloud`: `minimax_h3_{fl2va|ref2va}_pruned_bf16.safetensors`
- **LoRA Strength**: `1.0` (`0.95 ~ 1.05`)
- **Sampler & Guider**: `SamplerCustomAdvanced` + `BasicGuider (CFG = 1.0)` + `KSamplerSelect ("euler")`
- **Scheduler**: `BetaSamplingScheduler` (`steps = 4` or `3`, `alpha = 0.6`, `beta = 0.6`)
- **`MiniMaxH3SigmaShift`**:
  - `FL2VA / I2VA / T2VA`: `shift_video = 12.0`, `shift_audio = 3.0`
  - `Ref2VA` (`1344×768` Multi-Ref): `shift_video = 9.0` (`9.0 ~ 12.0`), `shift_audio = 3.0`, and set **`ref_image_size = "max"`** on `MiniMaxH3ReferenceToVideo` (include a high-res chest-up portrait close-up on character turnaround sheets for pore-level 768P facial fidelity).
