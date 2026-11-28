# lightx2v/Minimax-H3-Turbo-Fusion

Источник: https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion · скачано 2026-11-28
Загрузок: 24351 · лайков: 116 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,comfyui · обновлено: 2026-11-24T11:10:21.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/blob/main/LICENSE
library_name: diffusion-single-file
base_model: MiniMaxAI/MiniMax-H3
tags:
  - MiniMax-H3
  - video-generation
  - audio-video-generation
  - lora
  - comfyui
---

<h1 align="center">MiniMax-H3 Turbo Fusion</h1>

<div align="center">

[![LightX2V](https://img.shields.io/badge/LightX2V-181717?style=flat&logo=github&logoColor=white)](https://github.com/ModelTC/LightX2V) [![LightX2V Studio](https://img.shields.io/badge/LightX2V_Studio-2563EB?style=flat&logo=lightning&logoColor=white)](https://x2v.light-ai.top/) [![MiniMax-H3 Turbo](https://img.shields.io/badge/MiniMax--H3_Turbo-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/lightx2v/Minimax-h3-Turbo)

</div>

Motivated by [Pika's Flex Turbo LoRA release](https://huggingface.co/Pika-Labs/Pika-H3-Flex-Turbo-LoRA), we combine **our latest LightX2V Ref2VA distillation checkpoint**, [Pika's Ref2VA Flex adapter](https://huggingface.co/Pika-Labs/Pika-H3-Flex), and [Larry's Turbo v4 step-600 EMA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) into **one LoRA**. No inference-time stacking of three adapters is needed. This is an independent LightX2V fusion, not an official Pika release.

| Release | Fusion recipe |
|---|---|
| **4-step v1.0** | LightX2V turbo LoRA checkpoint × **1.0** + Pika Ref2VA Flex × **0.4** + Larry Turbo v4 EMA × **0.4** |
| **8-step v1.0** | LightX2V turbo LoRA checkpoint × **1.0** + Pika Ref2VA Flex × **0.4** + Larry Turbo v4 EMA × **0.4** |

The **Lite r32** files are per-layer QR-product truncated-SVD approximations. They retain 98.41% (4-step) and 97.80% (8-step) of the Full updates' spectral energy *before* BF16 rounding—weight-space statistics, **not** perceptual quality scores.

## Download

`_comfyui` identifies the ComfyUI key layout; without it, use the LightX2V/Diffusers-style layout. `lite_r32` means the compressed adapter.

| Steps | Variant | LightX2V / Diffusers | ComfyUI |
|:---:|---|---|---|
| 4 | Full | [Download](minimax_h3_ref2v_turbo_fusion_4step_v1.0.safetensors) | [Download](minimax_h3_ref2v_turbo_fusion_4step_v1.0_comfyui.safetensors) |
| 4 | Lite r32 | [Download](minimax_h3_ref2v_turbo_fusion_4step_v1.0_lite_r32.safetensors) | [Download](minimax_h3_ref2v_turbo_fusion_4step_v1.0_lite_r32_comfyui.safetensors) |
| 8 | Full | [Download](minimax_h3_ref2v_turbo_fusion_8step_v1.0.safetensors) | [Download](minimax_h3_ref2v_turbo_fusion_8step_v1.0_comfyui.safetensors) |
| 8 | Lite r32 | [Download](minimax_h3_ref2v_turbo_fusion_8step_v1.0_lite_r32.safetensors) | [Download](minimax_h3_ref2v_turbo_fusion_8step_v1.0_lite_r32_comfyui.safetensors) |

Download **one** variant for your loader. Do not add the component LoRAs on top of it: their updates are already included. The ComfyUI conversion fuses Q/K/V and preserves each module's effective `alpha / rank` scale.

## Inference settings

Use **Euler** sampling with **video flow shift 12** and **audio flow shift 3**.

## Ref2VA showcase

### Case 1

**Reference images, in input order:** [Picture 1–7](https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/tree/main/assets/case1).

**Prompt:** “The target video is a high-speed, big-budget live-action brawl sequence inside a giant high-end banquet hall.” [Read the full prompt](https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/blob/main/assets/case1/prompt.txt).

**4-step comparison**

| MiniMax-H3 base · 29 steps | [pika_h3_flex_turbo_4step](https://huggingface.co/Pika-Labs/Pika-H3-Flex-Turbo-LoRA/blob/main/pika_h3_flex_turbo_4step_ref2va.safetensors) | LightX2V Turbo Fusion · 4 steps |
|---|---|---|
| <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case1/base29.mp4" controls playsinline preload="none" width="100%"></video> | <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case1/pika_mix4.mp4" controls playsinline preload="none" width="100%"></video> | <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case1/ours_mix4.mp4" controls playsinline preload="none" width="100%"></video> |

**8-step comparison**

| MiniMax-H3 base · 29 steps | [pika_h3_flex_turbo_8step](https://huggingface.co/Pika-Labs/Pika-H3-Flex-Turbo-LoRA/blob/main/pika_h3_flex_turbo_8step_hd_ref2va.safetensors) | LightX2V Turbo Fusion · 8 steps |
|---|---|---|
| <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case1/base29.mp4" controls playsinline preload="none" width="100%"></video> | <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case1/pika_mix8.mp4" controls playsinline preload="none" width="100%"></video> | <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case1/ours_mix8.mp4" controls playsinline preload="none" width="100%"></video> |

### Case 2

**Reference images, in input order:** [Picture 1–5](https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/tree/main/assets/case2).

**Prompt:** “15秒竖版梦核音乐录像高潮与结尾。前半段保持两位人物身份，后半段旧友完全化为蒲公英种子，现实卧室中只能出现主唱一人。” [Read the full prompt](https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/blob/main/assets/case2/prompt.txt).

**4-step comparison**

| MiniMax-H3 base · 29 steps | [pika_h3_flex_turbo_4step](https://huggingface.co/Pika-Labs/Pika-H3-Flex-Turbo-LoRA/blob/main/pika_h3_flex_turbo_4step_ref2va.safetensors) | LightX2V Turbo Fusion · 4 steps |
|---|---|---|
| <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case2/base29.mp4" controls playsinline preload="none" width="100%"></video> | <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case2/pika_mix4.mp4" controls playsinline preload="none" width="100%"></video> | <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case2/ours_mix4.mp4" controls playsinline preload="none" width="100%"></video> |

**8-step comparison**

| MiniMax-H3 base · 29 steps | [pika_h3_flex_turbo_8step](https://huggingface.co/Pika-Labs/Pika-H3-Flex-Turbo-LoRA/blob/main/pika_h3_flex_turbo_8step_hd_ref2va.safetensors) | LightX2V Turbo Fusion · 8 steps |
|---|---|---|
| <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case2/base29.mp4" controls playsinline preload="none" width="100%"></video> | <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case2/pika_mix8.mp4" controls playsinline preload="none" width="100%"></video> | <video src="https://huggingface.co/lightx2v/Minimax-H3-Turbo-Fusion/resolve/main/assets/comparisons/case2/ours_mix8.mp4" controls playsinline preload="none" width="100%"></video> |

## Credits

The source updates are credited to the [LightX2V](https://huggingface.co/lightx2v/Minimax-h3-Turbo), [Pika](https://huggingface.co/Pika-Labs/Pika-H3-Flex), and [Larry](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) contributors. This release modifies and combines those adapters; see `LICENSE`, `NOTICE`, and the linked upstream model pages before redistribution or use.
