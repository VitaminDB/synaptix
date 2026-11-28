# ulifisherde/MiniMax-H3-Turbo-Lora-Pruned-Fixed

Источник: https://huggingface.co/ulifisherde/MiniMax-H3-Turbo-Lora-Pruned-Fixed · скачано 2026-11-28
Загрузок: 1033 · лайков: 0 · базовая: MiniMaxAI/MiniMax-H3 · теги: turbo,lora,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-09-03T18:54:14.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: quantized
pipeline_tag: text-to-video
library_name: diffusers
tags:
  - minimax-h3
  - turbo
  - lora
  - video-generation
  - comfyui
---

# MiniMax-H3 Turbo LoRA (Pruned-Base Compatible, File-Fixed)

Fixed copy of [`Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI`](https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI)
(4-step turbo LoRA, original trainer [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)).

## Why this repo exists

Every `.safetensors` in the source repo has **64 extra trailing bytes** after the last tensor
(declared data ends at byte 620,285,592; files are 620,285,656 bytes), which makes strict safetensors
loaders fail with:

```
SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered
```

This file is byte-identical to the source file with the 64 trailing bytes removed. All 416 tensors
(`attn.qkv_proj` / `attn.out_proj` / `mlp.fc1` / `mlp.fc2` per block × 50 blocks + `token_refiner`,
**no `adaln` targets** — designed for pruned / curve-form MiniMax-H3 checkpoints) load and match a
pruned FP8 base (e.g. `Comfy-Org/MiniMax-H3` `*_pruned_fp8_scaled`).

## Usage (ComfyUI)

`Load LoRA` node → `minimax_h3_turbo_4step_ckpt600_ema_V4_fixed.safetensors`, strength 0.8–1.8,
Steps 8–12, sampler `res_multistep`. (Settings from the author's workflow.)
