# OliviaRossi/Minimax-H3-Sulphur3-10Erosmax-SingularityV2-TriPrism-TurboFusion

Источник: https://huggingface.co/OliviaRossi/Minimax-H3-Sulphur3-10Erosmax-SingularityV2-TriPrism-TurboFusion · скачано 2026-11-28
Загрузок: 0 · лайков: 138 · базовая: ["MiniMaxAI/MiniMax-H3","SulphurAI/Sulphur-3","TenStrip/10Eros-Max","WarmBloodAban/Minimax-h3_Singularity"] · теги: comfyui · обновлено: 2026-11-25T17:00:04.000Z

---

---
library_name: comfyui
license: other
license_name: minimax-h3-community-license
base_model:
  - MiniMaxAI/MiniMax-H3
  - SulphurAI/Sulphur-3
  - TenStrip/10Eros-Max
  - WarmBloodAban/Minimax-h3_Singularity
tags:
  - minimax-h3
  - comfyui
  - text-to-video
  - image-to-video
  - model-merge
  - prism
  - int8
  - w6a8
  - w4a8
  - convrot
  - h3-tiny
---

# Minimax-H3-Sulphur3-10Erosmax-SingularityV2-TriPrism-TurboFusion

3-way **Tri-PRISM** merge of `Sulphur-3`, `10Eros-Max 8stepTurbo`, and `Singularity V2 (ref2va)` with exact rank-8 AdaLN curve folding (`canonical_t1`), **LightX2V Turbo-Fusion 8-step** (`@ 1.0`), and **MMH3 Realism Slider** (`@ 1.5`) baked in.

<div align="center">
  <video autoplay loop playsinline width="50%">
    <source src="https://huggingface.co/OliviaRossi/Minimax-H3-Sulphur3-10Erosmax-SingularityV2-TriPrism-TurboFusion/resolve/main/SkinSample01.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

<div align="center">
  <video autoplay loop muted playsinline width="50%">
    <source src="https://huggingface.co/OliviaRossi/Minimax-H3-Sulphur3-10Erosmax-SingularityV2-TriPrism-TurboFusion/resolve/main/SkinSample02.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>


## Files:

- `Sulphur3_10ErosMax_SingularityV2_TriPRISM_TurboFusion_realism_bf16.safetensors` — Pruned 50-block ComfyUI BF16 diffusion model (FP32 rank-8 AdaLN projections).
- `s3_10ErosMax_SingularityV2_TriPRISM_turbofusion8s_realism15_int8_convrot.safetensors` — 50-block native INT8 ConvRot (`int8_tensorwise`, `convrot_groupsize=256`).
- `Sulphur3_10ErosMax_SingularityV2_TriPRISM_TurboFusion_realism_pruned_w6a8_g32.safetensors` — 50-block **W6A8** (`w6a8_int8`, `group_size=32`, `convrot_groupsize=256`, 208 quantized layers + INT8 `token_refiner`, mean rel L2 err `2.289%`). Requires ComfyUI with native `w6a8_int8` support (`>= v0.45.0`, Ampere SM 80+ recommended).
- `Sulphur3_10ErosMax_SingularityV2_TriPRISM_TurboFusion_realism_pruned_w4a8_mixed.safetensors` — 50-block **W4A8 Mixed** (`asym_w4a8_int8`, `group_size=16`, `convrot_groupsize=256`, 16-entry codebook, 200 quantized layers, mean rel L2 err `7.315%`). Loads natively on ComfyUI `>= v0.31.0` (or via `ComfyUI-W4A8-Loader`).
- `Sulphur3_10ErosMax_SingularityV2_TriPRISM_TurboFusion_realism_h3tiny17b_int8_convrot.safetensors` — **H3-Tiny 17B-style Depth-Pruned INT8 ConvRot** (42 blocks; drops `[6, 7, 8, 9, 10, 11, 13, 15]` training-free per `fuyi25/MiniMax-H3-Tiny`, then MSE-clipped `int8_tensorwise` ConvRot across 176 layers, mean rel L2 err `0.914%`). Loads natively via standard **Load Diffusion Model**.
- `merge_report.json` / `quant_report.json` — Merge and quantization telemetry.

## ComfyUI Usage

Place any `.safetensors` file in `ComfyUI/models/diffusion_models/` and select it in **Load Diffusion Model** (`UNETLoader`). Recommended LightX2V Turbo-Fusion settings: **8 steps**, **Euler**, video flow shift **12**, audio flow shift **3**.

## License

Inherits the **MiniMax H3 Community License** and upstream source model/LoRA/H3-Tiny licenses.
