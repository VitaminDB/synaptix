# molbal/MiniMax-H3-GGUF

Источник: https://huggingface.co/molbal/MiniMax-H3-GGUF · скачано 2026-11-28
Загрузок: 75253 · лайков: 73 · базовая: MiniMaxAI/MiniMax-H3 · теги: gguf,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-12T20:47:00.000Z

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
  - reference-to-audio-video
  - gguf
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: quantized
---

# MiniMax H3 (GGUF Quantizations)

This repository (`molbal/MiniMax-H3-GGUF`) provides GGUF quantized versions and necessary component files for the MiniMax H3 model. MiniMax H3 is a general-purpose, omni-modal generative system that supports unified understanding of multimodal contexts composed of text, images, video, and audio. It can generate video with native stereo audio at resolutions up to 2K and durations of up to 15 seconds.

## Repository Contents

### UNet Models (GGUF)
This repository includes quantized versions of both the `FL2VA` (First-and-last-frame mode) and `Ref2VA` (Omni-reference mode) base models. The FL2VA builds are pruned to FP8 first and then quantized to GGUF.

* **FL2VA Models (pruned to FP8):**
  * `minimax_h3_fl2va_pruned_fp8_Q4_0.gguf` (11.4 GB)
  * `minimax_h3_fl2va_pruned_fp8_Q8_0.gguf` (20.2 GB)
  * `minimax_h3_fl2va_pruned_fp8_Q8_CR.gguf` (20.2 GB)
  * `minimax_h3_fl2va_pruned_fp8_U16G.gguf` (15.0 GB)
* **Ref2VA Models:**
  * `minimax-h3-ref2va-Q4_0.gguf` (11.4 GB)
  * `minimax-h3-ref2va-Q8_0.gguf` (21.4 GB)
  * `minimax-h3-ref2va-Q8_CR.gguf` (20.2 GB)
  * `minimax-h3-ref2va-U16G.gguf` (15.0 GB)

### Quantization formats

Note: Q8_CR and U16G are specifically created to work with the [molbal/CofmyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF) ComfyUI custom nodes and will probably not work elsewhere.

| Format | Quality | Notes |
|---|---|---|
| **Q4_0** | Lower | Standard 4-bit GGUF quantization. |
| **Q8_0** | High | Standard 8-bit GGUF quantization. |
| **Q8_CR** | High (≈ Q8_0) | Q8_0 quality with **INT8 ConvRot** weights. |
| **U16G** | Good | Mixed INT8 + Q4_0 weights, sized to fit comfortably in 16 GB VRAM cards. Despite being larger than Q4_0, it is faster, than that on 16GB> VRAM cards. |

## Model Variants and Input Specifications

* **H3-Base-FL2VA:** First-and-last-frame mode. Supports zero, one, or two input images.
  * No image input: Text-to-video mode.
  * One image input: First-frame-to-video or last-frame-to-video generation.
  * Two image inputs: First-and-last-frame-to-video generation.
* **H3-Base-Ref2VA:** Omni-reference mode. Supports multi-modal reference inputs:
  * **Images:** ≤ 9 images.
  * **Videos:** ≤ 3 clips; each clip must be 2–15 seconds long; total duration ≤ 15 seconds.
  * **Audio:** ≤ 3 clips; audio must be accompanied by image or video input and cannot be used as the sole input; each clip must be 2–15 seconds long; total duration ≤ 15 seconds.
  * **Mixed inputs:** Maximum number of files across all input types is 12.

## Output Specifications
* **Output duration:** 4–15 seconds.
* **Output aspect ratio:** Supports a wide range of aspect ratios, including but not limited to 21:9, 16:9, 4:3, 1:1, 3:4, and 9:16.
* **Output resolution:** Supports various resolution dimensions. The shorter side is set to 768 pixels by default. 2K generation can be achieved with H3-Regenerate-2K.
* **Output frame rate:** 24 FPS.
* **Output audio:** 32 kHz stereo.
* **Supported dialogue languages:** Stable support for 11 languages: Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, and Spanish. Additional languages are also supported to varying degrees.

## License
MiniMax H3 is released under the MiniMax H3 Community License Agreement.
