# NaukNauk/minimax-h3-turbo-fl2va-merged

Источник: https://huggingface.co/NaukNauk/minimax-h3-turbo-fl2va-merged · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: lora · обновлено: 2026-08-18T15:43:29.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE
base_model:
  - MiniMaxAI/MiniMax-H3
library_name: sglang
pipeline_tag: image-text-to-video
tags:
  - minimax-h3
  - image-to-video
  - first-last-frame-to-video
  - audio-video-generation
  - lora
---

# MiniMax-H3 FL2VA + Larry v4 Turbo (merged transformer)

This repository contains the premerged **FL2VA transformer only** for native
MiniMax-H3 first-frame, last-frame, and first+last-frame video-and-audio generation.
It must be assembled with the non-transformer components from the pinned
`MiniMaxAI/MiniMax-H3` FL2VA partition before serving.

## Pinned provenance

- Base: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`, partition `FL2VA`
- LoRA: `larryvrh/MiniMax-H3-Turbo-Lora@43a74557ac3f6539db8e0f2a959d03feb7a81480`
- File: `minimax_h3_turbo_v4_step600_ema.safetensors`
- LoRA SHA-256: `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`
- Merge scale: `1.0`
- Merged targets: `259`
- Arithmetic: base dtype output with FP32 chunked `W + B @ A` accumulation

The Larry v4 model card recommends 4–8 denoiser evaluations. With SGLang's H3
scheduler contract, `num_inference_steps: 9` produces 8 denoiser evaluations.

## SGLang request mode

Serve the assembled model with `--model-variant fl2va`. A classic image-to-video
request supplies one image condition with `role: "keyframe"` and `frame_index: 0`.
Use `-1` for a last-frame condition, or ordered `[0, -1]` conditions for both.

This derivative remains subject to the MiniMax-H3 community license linked above.
The source LoRA is Apache-2.0 licensed. Repository ID: `NaukNauk/minimax-h3-turbo-fl2va-merged`.
