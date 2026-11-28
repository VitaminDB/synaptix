# Abiray/MiniMax-H3-Pruned-GGUF

Источник: https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF · скачано 2026-11-28
Загрузок: 936048 · лайков: 98 · базовая: MiniMaxAI/MiniMax-H3 · теги: gguf,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-07T19:16:23.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: quantized
pipeline_tag: image-text-to-video
tags:
  - comfyui
  - gguf
  - text-to-video
  - image-to-video
  - image-text-to-video
  - video-to-video
  - text-to-audio-video
  - image-to-audio-video
  - synchronized-audio-video
  - reference-to-audio-video
  - multimodal
---

# MiniMax H3 Pruned GGUF

<video controls autoplay loop muted style="width: 100%; border-radius: 8px;" src="https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF/resolve/main/video/MiniMax_H3_00002-audio.mp4"></video>

This repository (`Abiray/MiniMax-H3-Pruned-GGUF`) provides pruned and quantized GGUF weights for the **MiniMax H3** omni-modal generative model. MiniMax H3 is designed for unified multimodal context processing, capable of generating synchronized high-definition video and 32 kHz stereo audio from text, image, audio, and video inputs.

---

## 🌟 Key Highlights

* **VRAM Efficiency:** Pruned architecture compressed down to **8.9 GB – 21.6 GB**, bringing MiniMax H3 execution to consumer-tier GPUs.
* **Synchronized Omni-Modal Output:** Simultaneous generation of video (24 FPS) and native stereo audio (32 kHz).
* **Native ComfyUI Support:** Directly compatible with standard `ComfyUI-GGUF` workflows using the native MiniMax backend.
* **Dual Pipeline Variants:** Full quant suites for both `FL2VA` (First/Last Frame) and `Ref2VA` (Omni-Reference) modes.

---

## 📂 Repository Weights & Quantization Breakdown

> **Note:** For optimal performance, **Q4_K_M** is recommended for 16 GB GPUs, while **Q5_K_M** is recommended for GPUs with 24 GB VRAM.

### FL2VA Models (First-and-Last-Frame Mode)

| Quantization | File Name | Size | Target Hardware / Use Case |
| :--- | :--- | :--- | :--- |
| **Q3_K_M** | `MiniMax-H3-FL2VA-Pruned-Q3_K_M.gguf` | 8.9 GB | Low VRAM (~12 GB GPUs) |
| **Q4_K_M** | `MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf` | 11.6 GB | **Recommended Balance** (16 GB GPUs) |
| **Q4_K_S** | `MiniMax-H3-FL2VA-Pruned-Q4_K_S.gguf` | 11.6 GB | Compact 4-bit K-Quant |
| **Q5_K_M** | `MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf` | 14.1 GB | **High Quality Balance** (24 GB GPUs) |
| **Q5_K_S** | `MiniMax-H3-FL2VA-Pruned-Q5_K_S.gguf` | 14.1 GB | Compact 5-bit K-Quant |
| **Q6_K** | `MiniMax-H3-FL2VA-Pruned-Q6_K.gguf` | 16.7 GB | Near-Lossless Precision |
| **Q8_0** | `MiniMax-H3-FL2VA-Pruned-Q8_0.gguf` | 21.6 GB | Maximum Fidelity / Reference |

### Ref2VA Models (Omni-Reference Mode)

| Quantization | File Name | Size | Target Hardware / Use Case |
| :--- | :--- | :--- | :--- |
| **Q3_K_M** | `MiniMax-H3-Ref2VA-Pruned-Q3_K_M.gguf` | 8.9 GB | Low VRAM (~12 GB GPUs) |
| **Q4_K_M** | `MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf` | 11.6 GB | **Recommended Balance** (16 GB GPUs) |
| **Q4_K_S** | `MiniMax-H3-Ref2VA-Pruned-Q4_K_S.gguf` | 11.6 GB | Compact 4-bit K-Quant |
| **Q5_K_M** | `MiniMax-H3-Ref2VA-Pruned-Q5_K_M.gguf` | 14.1 GB | **High Quality Balance** (24 GB GPUs) |
| **Q5_K_S** | `MiniMax-H3-Ref2VA-Pruned-Q5_K_S.gguf` | 14.1 GB | Compact 5-bit K-Quant |
| **Q6_K** | `MiniMax-H3-Ref2VA-Pruned-Q6_K.gguf` | 16.7 GB | Near-Lossless Precision |
| **Q8_0** | `MiniMax-H3-Ref2VA-Pruned-Q8_0.gguf` | 21.6 GB | Maximum Fidelity / Reference |

---

## ⚙️ Quickstart & ComfyUI Deployment

### Requirements
* **ComfyUI:** Version `v0.30.0` or higher is required for native MiniMax-H3 architecture support.
* **Extension:** `ComfyUI-GGUF` custom node package installed.

### Setup Steps
1. Download your desired `.gguf` variant from the table above.
2. Place the downloaded `.gguf` file into the `ComfyUI/models/unet/` directory.
3. In your ComfyUI workflow, load the model using the **`UnetLoaderGGUF`** node.

---

## 📋 Model Variants & Input Specifications

* **H3-Base-FL2VA (First-and-Last-Frame Mode):**
  * **No image input:** Operates as standard Text-to-Video / Text-to-Audio-Video.
  * **Single image input:** Generates video conditioned on the initial frame or final frame.
  * **Dual image input:** Interpolates seamlessly between designated first and last reference frames.

* **H3-Base-Ref2VA (Omni-Reference Mode):**
  * **Images:** Up to 9 reference images.
  * **Videos:** Up to 3 video clips (2–15 seconds per clip; total combined duration $\le$ 15s).
  * **Audio:** Up to 3 audio clips (must be accompanied by visual input; duration 2–15s).
  * **Combined Context:** Maximum of 12 multi-modal input assets allowed per prompt.

---

## 📊 Output Specifications

* **Duration:** 4 – 15 seconds.
* **Aspect Ratios:** Flexible ratio support (16:9, 9:16, 4:3, 3:4, 1:1, 21:9).
* **Native Resolution:** 768p base resolution (scalable to 2K via H3-Regenerate-2K).
* **Frame Rate:** 24 FPS.
* **Audio Track:** 32 kHz stereo audio output.
* **Multilingual Dialogue:** Stable support across 11 primary languages (English, Chinese, French, German, Japanese, Korean, Spanish, Russian, Portuguese, Italian, Arabic).

---

## 📜 Legal & License

MiniMax H3 is released under the **MiniMax H3 Community License Agreement**. Please refer to the [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) repository and the repository's `LICENSE` file for full terms and commercial usage guidelines.
