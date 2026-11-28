# Zakyrie/minimax-h3-beauty-clarity-enhancer

Источник: https://huggingface.co/Zakyrie/minimax-h3-beauty-clarity-enhancer · скачано 2026-11-28
Загрузок: 1518 · лайков: 4 · базовая: minimax/h3 · теги: lora · обновлено: 2026-11-05T15:40:21.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://platform.minimax.io/h3-license
tags:
  - lora
  - minimax
  - minimax-h3
  - text-to-video
  - video-generation
  - detailer
  - enhancer
base_model: minimax/h3
---

# MiniMax H3 – Beauty & Clarity Enhancer LoRA

> **Attribution & Source:**  
> This repository mirrors the model weights created by **vrgamedevgirl** on Civitai.  
> 🔗 **Original Civitai Page:** [MiniMaxH3_Enhancers (Model #3025511)](https://civitai.com/models/3025511/minimaxh3enhancers)  
> *All credit for dataset preparation, training, and workflow design belongs entirely to the original creator.*

---

## 🎬 Video Comparison & Showcase

Watch the full side-by-side comparison across multiple scenes (**No LoRA vs. Epoch 5 vs. Epoch 10**):

<div align="center">
  <video controls autoplay loop muted playsinline style="max-width: 800px; width: 100%; height: auto; border-radius: 8px;">
    <source src="https://huggingface.co/Zakyrie/minimax-h3-beauty-clarity-enhancer/resolve/main/assets/Beauty%20%26%20Clarity%20-%20Enhancer%20LoRA.mp4" type="video/mp4">
  </video>
  <p><small><i>Side-by-side comparison across various dynamic, lighting, and character scenes</i></small></p>
</div>

---

## Overview

This detailer LoRA is designed to improve the overall visual quality of MiniMax H3 generations, producing sharper, cleaner, and more aesthetically refined results without altering core prompt intent.

### What It Improves:
* **Sharper Details:** Improves overall scene clarity and resolves softness and unwanted blur.
* **Beauty Enhancement:** Enhances facial features, skin texture, and aesthetic appeal while avoiding an artificial plastic look.
* **Cleaner Backgrounds:** Cleans up muddy, messy, or artifact-heavy background elements.
* **Polished Visual Quality:** Delivers a more cinematic and intentional aesthetic while preserving photorealism.

---

## Available Versions

| Weight File | Variant | Characteristics |
| :--- | :--- | :--- |
| `MiniMaxH3_Enhancer_minimaxh3_3432130_epoch_5.safetensors` | **Epoch 5 (Lighter)** | Gentle sharpening, subtle beauty adjustments, and background cleanup without noticeably altering original character likeness. |
| `MiniMaxH3_Enhancer_minimaxh3_3432130_epoch_10.safetensors` | **Epoch 10 (Stronger)** | More pronounced facial beauty enhancement, sharper edges, and higher contrast background polish. |

---

## ⚙️ Recommendations & Usage

* **Choosing a Version:**
  * Start with **Epoch 5** if you prioritize naturalism and strict facial fidelity.
  * Use **Epoch 10** when you want stronger cosmetic refinement and maximum edge clarity.
* **LoRA Strength:**
  * All example test clips were generated at **strength 1.0**.
  * If you prefer an even subtler touch, start at **0.5** and scale upward until you find the ideal balance.

---

## 🛠️ Reference 2-Pass Generation Workflow

The comparison clips were produced with the following ComfyUI 2-pass configuration:

### Core Models:
* **Base Model:** MiniMax H3 FL2VA (`minimax_h3_fl2va_pruned_int8_convrot`)
* **Text Encoder:** `qwen3vl_32b_minimax_h3_nvfp4_awq`
* **VAEs:** `minimax_h3_video_vae_fp16` and `minimax_h3_audio_vae_fp32`

### Pass 1 — Generation:
* **Enhancer LoRA:** Strength `1.0` (*Pass 1 only*)
* **Generation Resolution:** 960 × 544 (half target resolution)
* **Steps:** 20
* **Sampler:** `res_multistep`
* **Scheduler:** `simple`
* **Denoise:** 1.0
* **Guider:** `BasicGuider` (no CFG, no negative prompt)

### Latent Upscale:
* **Upscaler:** Learned 3D latent upscaler (`minimax_h3_latent_upscaler_3d_conv_v1_bf16`), **×2**

### Pass 2 — Refinement:
* **Turbo LoRA:** `minimax_h3_fl2v_turbo_8step_v1.0_768p` at strength `1.0` (*Pass 2 only, no Enhancer*)
* **Steps:** 4
* **Denoise:** 0.35
* **Sampler:** `res_multistep`
* **Scheduler:** `simple`

### Final Output:
* **Resolution:** 1920 × 1088 (or 1088 × 1920 vertical), 24 fps
* **Clip Length:** 6–8 seconds
* **Audio:** Decoded directly from the Pass 1 latent using MiniMax H3's built-in audio engine.
