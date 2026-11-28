# releaseworthy/dasiwa-minimax-h3-turbo-int8

Источник: https://huggingface.co/releaseworthy/dasiwa-minimax-h3-turbo-int8 · скачано 2026-11-28
Загрузок: 20102 · лайков: 0 · базовая:  · теги: comfyui,turbo · обновлено: 2026-11-12T19:11:53.000Z

---

---
license: other
license_name: civitai-model-license
license_link: https://civitai.com/models/2877206
pipeline_tag: image-to-video
tags:
  - minimax-h3
  - comfyui
  - int8
  - turbo
  - mirror
---

# DaSiWa MiniMax H3 GXS v5 Turbo int8 mirror

A private-use mirror of the DaSiWa MiniMax H3 Turbo int8 checkpoint published on Civitai
([model 2877206](https://civitai.com/models/2877206), version `DaSiWa GXS v5 Turbo`, id 3457969), kept here so rented
GPU instances can pull it at full speed; the Civitai CDN throttles a single host and answers HTTP 429 after many GB.

- File: `DasiwaMinimaxH3_dasiwaGXSV5Turbo_int8.safetensors` (20,967,669,120 bytes,
  sha256 `d73ac7f98c32db783400687ea0d0dca3f562b1d2ad99baf36cacd79b4df1efcc`), Civitai file
  `dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo_int8_row-wise_convrot_runtime_mixed.safetensors`
- Runs in ComfyUI with the DaSiWa MiniMax H3 workflows (Civitai 2831978: T2VA, FL2VA, REF2VA) and node pack; the
  model card lists 4 steps (fast preview, degraded audio), 6 (basic) and 8 (quality)
- The previous Hybrid v4 Turbo int8 file is in this repository's commit history
- Text encoder, VAEs and latent upscaler are not mirrored here: they come from
  [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (`qwen3vl_32b_minimax_h3_int8_convrot`),
  [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental) and
  [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

All credit for the checkpoint goes to its Civitai author; this repository only re-hosts the file under the licence it
was published with.
