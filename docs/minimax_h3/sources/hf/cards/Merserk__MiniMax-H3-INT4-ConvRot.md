# Merserk/MiniMax-H3-INT4-ConvRot

Источник: https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot · скачано 2026-11-28
Загрузок: 166669 · лайков: 54 · базовая:  · теги: comfyui · обновлено: 2026-08-03T20:56:34.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
pipeline_tag: image-text-to-video
tags:
  - comfyui
  - minimax-h3
  - video-generation
  - int4
  - convrot
---

<div align="center">

# MiniMax H3 · INT4 ConvRot

<img
  src="https://cdn-uploads.huggingface.co/production/uploads/63f9610670d45efb75a30a19/1WvVDGq04j4IhumL81Yy2.png"
  alt="MiniMax H3 INT4 ConvRot"
  width="600"
/>

**Low-memory INT4 ConvRot weights for running MiniMax H3 locally in ComfyUI**

[![Original Model](https://img.shields.io/badge/Original-MiniMax_H3-blue)](https://huggingface.co/MiniMaxAI/MiniMax-H3)
[![ComfyUI Guide](https://img.shields.io/badge/ComfyUI-Workflow_Guide-green)](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
[![Civitai](https://img.shields.io/badge/Civitai-Model_Page-orange)](https://civitai.com/models/2830162/minimax-h3-int4-convrot-or-12gb-vram)

</div>

> [!NOTE]
> For **12 GB GPUs**, start with a **pruned** checkpoint. ComfyUI will rely on model offloading, so **32 GB+ system RAM** and a fast NVMe SSD are recommended.

## Included files

| File | Purpose | Size |
|---|---|---:|
| `minimax_h3_fl2va_pruned_int4_convrot.safetensors` | T2V / first-last-frame video | 11.3 GB |
| `minimax_h3_ref2va_pruned_int4_convrot.safetensors` | Reference-driven video | 11.3 GB |
| `minimax_h3_fl2va_int4_convrot.safetensors` | Full FL2VA checkpoint | 17.9 GB |
| `minimax_h3_ref2va_int4_convrot.safetensors` | Full Ref2VA checkpoint | 17.9 GB |
| `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` | INT4 text encoder | 15.0 GB |

## Installation

```bash
hf download Merserk/MiniMax-H3-INT4-ConvRot \
  --local-dir MiniMax-H3-INT4-ConvRot
```

Place the files in:

```text
ComfyUI/models/diffusion_models/   # FL2VA / Ref2VA checkpoints
ComfyUI/models/text_encoders/      # Qwen3-VL text encoder
ComfyUI/models/vae/                # Official MiniMax H3 video + audio VAEs
```

Update to **ComfyUI 0.30.0+**, then open **Template Library → Video → MiniMax H3**.

## Model choice

- **FL2VA** — text-to-video, first-frame, last-frame, or first-and-last-frame generation.
- **Ref2VA** — generation guided by reference images, videos, and audio.

> INT4 quantization may differ from the original BF16 model. These files are community conversions; the original **MiniMax H3 Community License** still applies.
