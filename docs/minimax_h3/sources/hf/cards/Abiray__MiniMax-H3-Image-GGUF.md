# Abiray/MiniMax-H3-Image-GGUF

Источник: https://huggingface.co/Abiray/MiniMax-H3-Image-GGUF · скачано 2026-11-28
Загрузок: 338350 · лайков: 366 · базовая: ["MiniMaxAI/MiniMax-H3-Image"] · теги: gguf,comfyui,quantized,base_model:quantized:MiniMaxAI/MiniMax-H3-Image · обновлено: 2026-11-11T14:28:22.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE
language:
- en
tags:
- text-to-image
- image-to-image
- image-editing
- gguf
- minimax
- minimax-h3
- minimax-h3-image
- comfyui
- dit
- quantized
base_model:
- MiniMaxAI/MiniMax-H3-Image
pipeline_tag: text-to-image
widget:
  - text: "Showcase Image 1"
    output:
      url: images/minimax_h3_image_t2i_00003_.png
  - text: "Showcase Image 2"
    output:
      url: images/minimax_h3_image_t2i_00004_.png
  - text: "Showcase Image 3"
    output:
      url: images/minimax_h3_image_t2i_00005_.png
  - text: "Showcase Image 4"
    output:
      url: images/minimax_h3_image_t2i_00006_.png
  - text: "Showcase Image 5"
    output:
      url: images/minimax_h3_image_t2i_00007_.png
  - text: "Showcase Image 6"
    output:
      url: images/minimax_h3_image_t2i_00008_.png
---

# MiniMax-H3-Image (Pruned DiT) — GGUF Quantizations

Quantized **GGUF** weights for **[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)**, converted directly from the structurally pruned BF16 master checkpoint provided by **[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)** (`minimax_h3_image_pruned_bf16.safetensors`).

MiniMax-H3-Image is a state-of-the-art **33.1B parameter single-stream Diffusion Transformer (DiT)** designed for ultra-high-resolution text-to-image generation and multi-reference image editing (up to 16 MP / 4096 × 4096) in just **6 distilled sampling steps**.

All GGUFs in this repository have sensitive layers (the 24-channel input patchifier `video_patch_proj`, the 6-step PDD output head `final_layer`, and the low-rank AdaLN modulation basis `adaln_basis` / `adaln_t_table`) strictly preserved in **unquantized FP32** to eliminate quantization noise and grid artifacts.

> **✅ Update — Native Stock ComfyUI-GGUF Support:** All GGUF files in this repository now include the embedded `config` (`pdd_sigmas` 6-step schedule) header metadata and load natively out of the box in stock **ComfyUI-GGUF (`Unet Loader (GGUF)`)** with zero custom scripts or patches required! *(Note: [`minimax_h3_image_gguf_fix.py`](./minimax_h3_image_gguf_fix.py) is kept in the repository solely as an optional fallback for users on older versions of ComfyUI-GGUF).*
---

## Model Files & Quantization Overview

| File Name | Size | Quant Type | Recommended Hardware / VRAM | Quality / Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **`MiniMax-H3-Image-Pruned-Q2_K.gguf`** | **6.88 GB** | `Q2_K` | 8 GB – 10 GB VRAM | Maximum memory savings. Fits easily on consumer laptops & entry GPUs. |
| **`MiniMax-H3-Image-Pruned-Q3_K_M.gguf`** | **8.91 GB** | `Q3_K_M` | 12 GB VRAM | High efficiency with significantly lower quantization loss than 2-bit. |
| **`MiniMax-H3-Image-Pruned-Q4_K_S.gguf`** | **11.60 GB** | `Q4_K_S` | 12 GB – 16 GB VRAM | Fast 4-bit small variant. Well-balanced footprint. |
| **`MiniMax-H3-Image-Pruned-Q4_K_M.gguf`** | **11.60 GB** | `Q4_K_M` | 16 GB VRAM | **Recommended sweet spot.** Near-lossless visual quality for standard GPUs. |
| **`MiniMax-H3-Image-Pruned-Q5_K_M.gguf`** | **14.10 GB** | `Q5_K_M` | 16 GB – 24 GB VRAM | Maximum fidelity. Indistinguishable from full BF16. |

---

## 🖼️ Sample Generations (Q4_K_M)
<Gallery />

## Required Companion Files

MiniMax-H3-Image requires a dedicated **24-channel Image VAE** and the **Qwen3-VL-32B** text encoder (reading layer 50 hidden states).

### 1. Image VAE (24-Channel)
* **Official / Comfy-Org FP16 VAE:** [`minimax_h3_image_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae)  
  *(Place inside `ComfyUI/models/vae/`)*

> **Note:** The Image VAE uses 24 latent channels with a 2D ViT decoder. Do **not** use standard 4-channel SD/Flux VAEs or the 16-channel video VAE from earlier H3 releases.

### 2. Text Encoder (Qwen3-VL-32B)
You can choose between pre-quantized safetensors or standalone GGUF text encoders:

* **GGUF Text Encoders (by Abiray):**
  * **Q4_K_M (14.6 GB):** [`qwen3vl_32b_minimax_h3-Q4_K_M.gguf`](https://huggingface.co/Abiray/MiniMax-H3-GGUF/blob/main/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf)
  * **Q3_K_M (12 GB):** [`Qwen3vl_32b_h3-Q3_K_M.gguf`](https://huggingface.co/Abiray/Qwen3-VL-32B-Instruct-GGUF/blob/main/Qwen3vl_32b_h3-Q3_K_M.gguf)
* **Official / Comfy-Org Text Encoders:**
  * **NVFP4 / AWQ / BF16:** Available in the [Comfy-Org Text Encoders directory](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/text_encoders) (`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` or `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`)

*(Place inside `ComfyUI/models/clip/` or `ComfyUI/models/text_encoders/`)*


---
### 📋 Example Workflow

Download and import the ready-to-use ComfyUI workflow directly from this repository:
👉 [image_minimax_h3_image_t2i_gguf.json](https://huggingface.co/Abiray/MiniMax-H3-Image-GGUF/blob/main/image_minimax_h3_image_t2i_gguf.json)

---
## Installation & File Paths in ComfyUI

Ensure you have updated to **ComfyUI v0.45.0+** and have **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** installed.

```text
ComfyUI/
├── models/
│   ├── unet/ (or diffusion_models/)
│   │   └── MiniMax-H3-Image-Pruned-Q4_K_M.gguf
│   ├── vae/
│   │   └── minimax_h3_image_vae_fp16.safetensors
│   └── clip/ (or text_encoders/)
│       ├── qwen3vl_32b_minimax_h3-Q4_K_M.gguf
│       └── (or qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors)
```

---

## Recommended Generation Parameters

MiniMax-H3-Image is a distilled **Piecewise Decoupled Diffusion (PDD)** model trained specifically for single-stream 6-step inference:

- **Sampler:** `euler`
- **Scheduler:** `simple` or `sgm_uniform`
- **Sampling Steps:** `6` (the fused PDD heads are calibrated for exactly 6 steps)
- **CFG Scale:** `1.0` (**No CFG**; do not use high CFG or negative prompt scaling)
- **Denoise:** `1.0`
- **Latent Dimensions:** Native 24 channels (2 × 2 patchification, 32 × 32 pixels per token)
- **Resolution:** 1024 × 1024 (1 MP), 2048 × 2048 (4 MP), or up to 4096 × 4096 (16 MP)

---

## Prompting Guide

MiniMax-H3-Image responds best to **dense, descriptive English descriptions** that specify spatial layout, material properties, lighting conditions, and typography.

### Example Prompt (Packaging & Typography)
```text
The image is a cheerful toy-packaging poster built around a handmade plush fox floating above a summer meadow, drawn in a warm, fairly saturated palette: clear sky blue deepening toward the top, rich burnt-sienna and terracotta plush for the fox, warm cream, fresh sage green in the meadow, cornflower and cherry-red accents on the small props, and buttercup yellow in the lettering, with bright afternoon sun from the upper left and crisp short shadows. The fox is the hero, sewn from long-staple mohair-type plush in warm burnt sienna and hand-stuffed so its silhouette is slightly lumpy; the pile is long and irregular, single fibres catching the light along its outline as a bright hairy halo against the sky, with cream muzzle and sock paws. Across the top of the frame the word "MINIMAX" is inflatable and rounded in thick felted wool; directly beneath, smaller and centred, "H3" sits between two small dots. Along the bottom, a heavy bold sans-serif logotype reads "TEXT TO IMAGE" in saturated cornflower blue. All lettering is English only, with no Chinese characters.
```

### Example Prompt (Photorealistic Macro Still Life)
```text
A raw, high-resolution 35mm documentary portrait of an artisan ceramicist in her sunlit studio loft, shot on a Leica M11 with a 50mm f/1.4 lens. Bathed in cool morning daylight streaming from a tall steel-framed window on the left. Authentic, unretouched skin texture with visible pores, fine smile lines, and a faint dusting of dry white clay powder across her cheekbone. Her hands rest on a weathered pine workbench scattered with trimming tools and a half-empty stoneware coffee mug. Organic shallow depth of field, natural color grading, no plastic skin smoothing or HDR halos.
```
---

## License & Usage

This model is subject to the **[MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)**.

* Free for research, personal, and commercial creation within the terms of the Community License Agreement.
* If your monthly/annual revenue exceeds the threshold specified in Section IV of the license, or if you provide hosted API services, refer to the commercial licensing terms at [platform.minimax.io/license](https://platform.minimax.io/license).

---

## Credits & Acknowledgements

* **[MiniMax AI](https://huggingface.co/MiniMaxAI)** for developing and releasing the weights of MiniMax-H3-Image.
* **[city96](https://github.com/city96/ComfyUI-GGUF)** for the `ComfyUI-GGUF` loader suite.
* Quantized by **[Abiray](https://huggingface.co/Abiray)**.
