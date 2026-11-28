# pottokao/MiniMax-H3-FL2VA-turbo-4step-v1.2-768p-NVFP4-rotated-T1

Источник: https://huggingface.co/pottokao/MiniMax-H3-FL2VA-turbo-4step-v1.2-768p-NVFP4-rotated-T1 · скачано 2026-11-28
Загрузок: 0 · лайков: 1 · базовая:  · теги: comfyui,turbo,distilled · обновлено: 2026-09-21T02:28:35.000Z

---

---
license: apache-2.0
pipeline_tag: image-to-video
library_name: comfyui
tags:
- minimax-h3
- fl2va
- turbo
- 4-step
- distilled
- nvfp4
- rotated
- nunchaku
- w4a4
- comfyui
---

# MiniMax-H3 FL2VA — Turbo 4-step v1.2 (768p) · rotated-NVFP4 (T1, self-contained)

Single-file ComfyUI diffusion model for **MiniMax-H3 FL2VA** (First-&-Last-frame → Video + Audio), with the **Turbo 4-step v1.2 768p** LoRA fused in — a 4-step distilled build.

- **Self-contained**: one `.safetensors` (base transformer + turbo LoRA fused bf16-round-first + official STRIPPED shell with curve-basis AdaLN rank-8). No separate base/LoRA needed.
- **Quantization**: **rotated-NVFP4**, tier **T1** (all-fp4 blocks), group-16 e4m3 scales, **nunchaku W4A4** pack.
- **Rotation**: block-256 Sylvester Walsh–Hadamard × diag(±1), seed = 100000 + blk·17 + site_id.
- **Size**: ~12.5 GB — fits a single 16 GB GPU.

## ⚠️ Requires the custom loader
Vanilla ComfyUI **cannot** load this — the rotation + W4A4 pack must be decoded by:
- **[pottokao/H3-RotNVFP4-ComfyUI-Loader](https://huggingface.co/pottokao/H3-RotNVFP4-ComfyUI-Loader)** (nunchaku path)

## Companions
- **Text encoder (recommended: Q3)**: [pottokao/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-GGUF](https://huggingface.co/pottokao/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-GGUF) — use the **Q3_K_S** GGUF. For ComfyUI the **visual-merged** variant (`*_vis.gguf`) is required (the visual tower is what ComfyUI uses to detect the H3 text encoder).
- MiniMax-H3 **video VAE** (fp16) + **audio VAE** (fp32)

## Usage
Drop the `.safetensors` into `ComfyUI/models/diffusion_models/`, load it through the RotNVFP4 loader node, wire in the TE + VAEs. 4 sampling steps.
