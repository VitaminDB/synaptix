# Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot

Источник: https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot · скачано 2026-11-28
Загрузок: 122734 · лайков: 277 · базовая: MiniMaxAI/MiniMax-H3 · теги: quantized,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-15T12:44:09.000Z

---

---
pipeline_tag: image-text-to-video
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
library_name: diffusers
tags:
- text-to-video
- image-to-video
- image-text-to-video
- video-to-video
- text-to-audio-video
- multimodal
- quantized
- comfyui
- int4
- nvfp4
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: quantized
language:
- th
---
# MiniMax H3 - Quantized & Pruned Collection (INT4 / INT8 / NVFP4)

This repository is a community-compiled collection of quantized and pruned weights for **[MiniMax H3 (Hailuo 3.0)](https://huggingface.co/MiniMaxAI/MiniMax-H3)**, optimized for local inference environments like ComfyUI. 

By unifying various quantization formats (INT4, INT8, Mixed, and NVFP4) into a single structured repository, this hub makes it easier for users with consumer GPUs (16GB - 24GB VRAM) to experiment with MiniMax H3's powerful omni-modal text/image/audio-to-video generation capabilities.

---
## 💻 Hardware & GPU Selection Guide (Which file to download?)

If you are new to local generation and aren't sure what to download, use this guide based on your graphics card. 

> ⚠️ **Note on VAEs:** Regardless of your GPU, **everyone** must download both files in the `/vae` folder (`minimax_h3_audio_vae_fp32.safetensors` and `minimax_h3_video_vae_fp16.safetensors`).

### 1. Standard High-End GPUs (16GB VRAM)
*Perfect for RTX 4070 Ti Super, RTX 4080, etc.*
*   **Diffusion Model:** Download the **INT4** (`MiniMax_H3_[...]_pruned_int4_convrot.safetensors`) or the **MIXED** (`_mixed_int4_int8_`) version.
*   **Text Encoder:** Download `qwen3vl_32b_minimax_h3_int4_convrot.safetensors`

### 2. Enthusiast GPUs (24GB VRAM)
*Perfect for RTX 3090, RTX 4090, etc.*
*   **Diffusion Model:** Download the **INT8** (`MiniMax_H3_[...]_pruned_int8_convrot.safetensors`) for the highest pruned quality.
*   **Text Encoder:** Download `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`

### 3. Nvidia Blackwell GPUs ONLY 
*Exclusively for RTX 5090, PRO 6000, and other next-gen Blackwell cards. Do not download these for 30/40 series cards!*
*   **Diffusion Model:** Download the **NVFP4** (`MiniMax_H3_[...]_pruned_nvfp4.safetensors`).
*   **Text Encoder:** Download `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`

### FL2VA vs. Ref2VA (What's the difference?)
*   **FL2VA:** Download this if you just want to do standard Text-to-Video or simple Image-to-Video (using one or two starting/ending images).
*   **Ref2VA:** Download this if you want to use advanced omni-references (multiple images, video clips, or audio inputs).

---
## 📦 Included Files & Formats

This repository uses a unified naming convention (`MiniMax_H3_FL2VA_*` and `MiniMax_H3_Ref2VA_*`) for easy integration into your local pipelines. 

### 1. Diffusion Models
*   **MIXED INT4/INT8 :** mixed-precision models. Requires 15+ VRAM (~15.5 GB).
    *   `MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors`
    *   `MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors`
*   **INT8 :** Highest quality pruned models. Requires 24GB+ VRAM (~21 GB).
    *   `MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors`
    *   `MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors`
*   **INT4 :** High-compression models for standard consumer hardware (~11.3 GB).
    *   `MiniMax_H3_FL2VA_pruned_int4_convrot.safetensors`
    *   `MiniMax_H3_Ref2VA_pruned_int4_convrot.safetensors`
*   **NVFP4 :** Experimental Nvidia 4-bit floating point models (Blackwell Architecture Required). 
    *   `MiniMax_H3_Ref2VA_pruned_nvfp4.safetensors` (12.5 GB)
    *   `MiniMax_H3_Ref2VA_nvfp4_mixed.safetensors` (24.4 GB)
    *   `MiniMax_H3_FL2VA_pruned_nvfp4.safetensors` (12.5 GB)

### 2. Text Encoders (`/text_encoders`)
*   `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` (15.0 GB) - *Recommended for 16GB GPUs*
*   `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (27.1 GB) - *Recommended for 24GB GPUs*
*   `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (15.7 GB) - *For Blackwell GPUs*

### 3. VAEs (`/vae`)
*   `minimax_h3_audio_vae_fp32.safetensors` (605 MB - Audio Decoder)
*   `minimax_h3_video_vae_fp16.safetensors` (5.21 GB - Video Decoder)
  
---
## Original MiniMax H3 System Overview

MiniMax H3 is a general-purpose, omni-modal generative system. It supports unified understanding of multimodal contexts composed of text, images, video, and audio, and can generate video with native stereo audio at resolutions up to 2K and durations of up to 15 seconds. Thanks to its task-generalization-oriented system design, H3 already possesses broad multimodal context understanding and generation capabilities at the pre-training stage, enabling outstanding performance in following complex multimodal instructions.

H3 supports the following input and output specifications:

| Category | Specification |
|---|---|
| Output duration | 4–15 seconds |
| Output aspect ratio | Supports a wide range of aspect ratios, including but not limited to 21:9, 16:9, 4:3, 1:1, 3:4, and 9:16 |
| Output resolution | Supports various resolution dimensions. The shorter side is set to 768 pixels by default. 2K generation can be achieved with H3-Regenerate-2K |
| Output frame rate | 24 FPS |
| Output audio | 32 kHz stereo |
| Supported dialogue languages | Stable support for 11 languages: Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, and Spanish. |

### Model Variants and Input Specifications

| Model Variant | Input Mode | Specifications |
|---|---|---|
| **H3-Base-FL2VA** | First-and-last-frame mode | Supports zero, one, or two input images. <br><br>- No image input: Text-to-video mode <br>- One image input: First-frame-to-video or last-frame-to-video generation <br>- Two image inputs: First-and-last-frame-to-video generation |
| **H3-Base-Ref2VA** | Omni-reference mode | Supports multi-modal reference inputs: <br><br>- **Images:** ≤ 9 images <br>- **Videos:** ≤ 3 clips; each clip must be 2–15 seconds long; total duration ≤ 15 seconds <br>- **Audio:** ≤ 3 clips; audio must be accompanied by image or video input and cannot be used as the sole input; each clip must be 2–15 seconds long; total duration ≤ 15 seconds <br>- **Mixed inputs:** Maximum number of files across all input types is 12 |

## Model Architecture

### H3-Context-IR
H3-Context-IR is a hosted preprocessing and orchestration system designed for free-form multimodal inputs. It interprets the relationships among text, images, audio, and reference videos, as well as how these materials relate to the intended generation output. H3-Context-IR serializes its understanding of the context into a structured representation accepted by H3-Base.

### H3-Base
- H3-Base encodes different modalities using their corresponding encoders or VAEs and organizes the encoded representations into a unified packed multimodal sequence. 
- Specifically, text is encoded by the H3-Encoder; visual inputs are encoded by both the H3-Encoder and the H3-VisualVAE; and audio is encoded solely by the H3-AudioVAE.
- The H3-Omni-Transformer jointly predicts video and audio latents, which are then decoded into video and stereo audio, respectively.

### H3-Encoder
- The H3-Encoder uses the full pretrained weights of Qwen3-VL-32B and provides the hidden states from its 50th layer to the H3-Omni-Transformer.

### H3-VAE
- **H3-VisualVAE** is a temporally causal video autoencoder with a spatial compression factor of 16×, a temporal compression factor of 4×, and 24 latent channels (f16t4d24). 
- **H3-AudioVAE** uses the same encoder and decoder for both the left and right audio channels while processing each channel independently. It compresses 32 kHz audio into a sequence of latent tokens with a temporal rate of 40 Hz.

### H3-Omni-Transformer
- H3-Omni-Transformer is a 33B-parameter dense, single-stream Transformer.
- The model uses three-dimensional Multimodal Rotary Position Embeddings (MM-RoPE) to represent positional relationships across the temporal and two spatial dimensions, `(t, h, w)`.

## License & Legal

MiniMax H3 (and its quantized derivatives) is released under the **[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)**. 

User-submitted text, images and videos, as well as enhanced prompts, are subject to automated moderation when using the official APIs. Content suspected of being unlawful, pornographic, or infringing third-party rights may be blocked. These guardrails do not affect the Licensee's obligations under the MiniMax H3 Community License, especially those relating to lawful use and use restrictions.

**Original Creators / Contact**
- Global API: [platform.minimax.io](https://platform.minimax.io)
- Contact: [model@minimax.io](mailto:model@minimax.io)
