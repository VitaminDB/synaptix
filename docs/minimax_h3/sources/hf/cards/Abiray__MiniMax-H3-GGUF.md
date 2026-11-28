# Abiray/MiniMax-H3-GGUF

Источник: https://huggingface.co/Abiray/MiniMax-H3-GGUF · скачано 2026-11-28
Загрузок: 206197 · лайков: 167 · базовая: MiniMaxAI/MiniMax-H3 · теги: gguf,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-08T07:18:36.000Z

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
<video controls autoplay loop muted style="width: 100%;" src="https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/Video/MiniMax_H3_00002_.mp4"></video>
This repository (`Abiray/MiniMax-H3-GGUF`) provides GGUF quantized versions and necessary component files for the MiniMax H3 model. MiniMax H3 is a general-purpose, omni-modal generative system that supports unified understanding of multimodal contexts composed of text, images, video, and audio. It can generate video with native stereo audio at resolutions up to 2K and durations of up to 15 seconds.

---
## 📌 Note: Pruned Versions for Consumer GPUs
If you are looking for a smaller model with the same great quality that fits better on consumer-tier GPUs, please check out the **[MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF)** repository. 
The pruned architecture is compressed down to 8.9 GB – 21.6 GB, bringing MiniMax H3 execution directly to consumer hardware. 
**Available Pruned Quants:**
| Bit Size | Quantization Type | File Size |
| :--- | :--- | :--- |
| **3-bit** | `Q3_K_M` | 8.9 GB |
| **4-bit** | `Q4_K_S`, `Q4_K_M` | 11.6 GB |
| **5-bit** | `Q5_K_S`, `Q5_K_M` | 14.1 GB |
| **6-bit** | `Q6_K` | 16.7 GB |
| **8-bit** | `Q8_0` | 21.6 GB |

---
## Repository Contents
### UNet Models (GGUF)
This repository includes quantized versions of both the `FL2VA` (First-and-last-frame mode) and `Ref2VA` (Omni-reference mode) base models. 
* **FL2VA Models:**
  * `MiniMax-H3-FL2VA-Q3_K_M.gguf` (15.6 GB)
  * `MiniMax-H3-FL2VA-Q3_K_S.gguf` (15.6 GB)
  * `MiniMax-H3-FL2VA-Q4_0.gguf` (18.6 GB)
  * `MiniMax-H3-FL2VA-Q4_K_M.gguf` (19.9 GB)
  * `MiniMax-H3-FL2VA-Q4_K_S.gguf` (19.9 GB)
  * `MiniMax-H3-FL2VA-Q5_0.gguf` (22.8 GB)
  * `MiniMax-H3-FL2VA-Q5_K_M.gguf` (23.9 GB)
  * `MiniMax-H3-FL2VA-Q5_K_S.gguf` (23.9 GB)
  * `MiniMax-H3-FL2VA-Q6_K.gguf` (28.2 GB)
  * `MiniMax-H3-FL2VA-Q8_0.gguf` (36 GB)
* **Ref2VA Models:**
  * `MiniMax-H3-Ref2VA-Q3_K_M.gguf` (15.6 GB)
  * `MiniMax-H3-Ref2VA-Q3_K_S.gguf` (15.6 GB)
  * `MiniMax-H3-Ref2VA-Q4_0.gguf` (18.6 GB)
  * `MiniMax-H3-Ref2VA-Q4_K_M.gguf` (19.9 GB)
  * `MiniMax-H3-Ref2VA-Q4_K_S.gguf` (19.9 GB)
  * `MiniMax-H3-Ref2VA-Q5_0.gguf` (22.8 GB)
  * `MiniMax-H3-Ref2VA-Q5_K_M.gguf` (23.9 GB)
  * `MiniMax-H3-Ref2VA-Q5_K_S.gguf` (23.9 GB)
  * `MiniMax-H3-Ref2VA-Q6_K.gguf` (28.2 GB)
  * `MiniMax-H3-Ref2VA-Q8_0.gguf` (36 GB)
### Text Encoders
* `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` (15 GB)
* `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (27.1 GB)
* `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (14.6 GB)
### VAE Models
* `minimax_h3_audio_vae_fp32.safetensors` (605 MB)
* `minimax_h3_video_vae_fp16.safetensors` (5.21 GB)
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
