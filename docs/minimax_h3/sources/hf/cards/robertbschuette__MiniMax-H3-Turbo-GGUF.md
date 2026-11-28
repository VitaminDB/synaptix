# robertbschuette/MiniMax-H3-Turbo-GGUF

Источник: https://huggingface.co/robertbschuette/MiniMax-H3-Turbo-GGUF · скачано 2026-11-28
Загрузок: 623 · лайков: 0 · базовая: MiniMaxAI/MiniMax-H3 · теги: gguf,comfyui,turbo,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-10-20T09:06:57.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
tags:
  - comfyui
  - text-to-video
  - image-to-video
  - image-text-to-video
  - video-to-video
  - text-to-audio-video
  - image-to-audio-video
  - image-text-to-audio-video
  - video-to-audio-video
  - audio-to-audio-video
  - audio-video-generation
  - multimodal
  - synchronized-audio-video
  - gguf
  - turbo
  - 4step
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: quantized
---

# MiniMax-H3-Turbo (FL2VA 4-step v1.0 768p GGUF Quantizations)

This repository provides GGUF quantized versions and ComfyUI workflows for the **FL2VA Turbo 4-step v1.0 768p** variant of [MiniMax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo). MiniMax H3 is an omni-modal generative system supporting unified text, image, video, and audio generation with native stereo audio.

The Turbo distillation compresses inference down to **4 steps (NFE)** at **768p** resolution with shifted schedules.

> [!IMPORTANT]
> **Full Merged Checkpoints:** These GGUF files contain **full merged weights** (the base model + distilled turbo weights merged directly into a standalone model checkpoint), not separate LoRA adapters. You load them directly as diffusion models in ComfyUI without needing additional base weights or LoRA loaders.

## Model Specifications

| Parameter | Specification |
|---|---|
| **Distilled Variant** | FL2VA Turbo 4-step v1.0 768p |
| **Tasks** | FL2VA / I2VA / T2VA (First/Last Frame & Text to Audio-Video) |
| **Training Resolution** | 768p (1344×768) |
| **Training Shifts (Video / Audio)** | 6 / 3 |
| **Distillation Steps (NFE)** | 4 |
| **Recommended Inference Steps (NFE)** | 4 |
| **Frame Rate** | 24 FPS |
| **Audio Output** | 32 kHz stereo |

## Repository Contents

### GGUF Models

* `minimax_h3_fl2v_turbo_4step_v1.0_768p_Q4_0.gguf` (11.4 GB)
* `minimax_h3_fl2v_turbo_4step_v1.0_768p_Q8_0.gguf` (21.4 GB)
* `minimax_h3_fl2v_turbo_4step_v1.0_768p_Q8_CR.gguf` (20.2 GB)

### Quantization Formats

> [!NOTE]
> `Q8_CR` is specifically created to work with the [molbal/ComfyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF) ComfyUI custom nodes.

| Format | Quality | Notes |
|---|---|---|
| **Q4_0** | Lower | Standard 4-bit GGUF quantization. |
| **Q8_0** | High | Standard 8-bit GGUF quantization. |
| **Q8_CR** | High (≈ Q8_0) | Q8_0 quality with **INT8 ConvRot** weights. |

## ComfyUI Usage & Node Requirements

To run these GGUF models in ComfyUI, install the following custom node packages:

1. **GGUF Loader Node (one of the following):**
   * [ComfyUI-GGUF (GitHub)](https://github.com/molbal/ComfyUI-GGUF) or install via ComfyUI Node Registry: [comfyui-gguf-reboot](https://registry.comfy.org/publishers/molbal/nodes/comfyui-gguf-reboot)
   * Used to load the quantized `.gguf` UNet/diffusion models.
2. **MiniMax H3 Turbo Sampler (one of the following):**
   * [ComfyUI-MiniMax-H3-Turbo (GitHub)](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo) or install via ComfyUI Node Registry: [comfyui-minimax-h3-turbo](https://registry.comfy.org/nodes/comfyui-minimax-h3-turbo)
   * Provides the dedicated Turbo sampling schedules and shift calculations for 4-step generation.

### Model Placement

Place the downloaded GGUF files in your ComfyUI models directory:
```
ComfyUI/models/diffusion_models/minimax_h3_fl2v_turbo_4step_v1.0_768p_Q8_CR.gguf
```
*(or `ComfyUI/models/unet/` depending on your GGUF loader configuration)*

## Shift & Sampling Schedule

For `NFE = 4`, `video shift = 6`, and `audio shift = 3`:
* Define unshifted evaluation points: $q_i = rac{4 - i}{4}$ for $i = 0, 1, 2, 3 ightarrow [1.0, 0.75, 0.5, 0.25]$
* Recommended inference step count: **4 steps**.

## Supported Input Modes (FL2VA)

* **Text-to-Audio-Video (T2VA):** No image input connected.
* **Image-to-Audio-Video (I2VA):** One image connected to `first_frame`.
* **First-and-Last-Frame-to-Audio-Video (FL2VA):** Two images connected to `first_frame` and `last_frame`.

## ComfyUI Workflows

Example workflows are provided in the `workflows/` directory:
* `workflows/minimax_h3_t2v-turbo-gguf.json` — Text-to-Audio-Video (T2VA) Turbo workflow.
* `workflows/minimax_h3_i2v-turbo-gguf.json` — Image-to-Audio-Video (I2VA / FL2VA) Turbo workflow.

## License

MiniMax H3 is released under the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
