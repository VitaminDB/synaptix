# Abiray/Minimax-h3-Singularity-V2-Pruned-GGUF

Источник: https://huggingface.co/Abiray/Minimax-h3-Singularity-V2-Pruned-GGUF · скачано 2026-11-28
Загрузок: 31130 · лайков: 14 · базовая: WarmBloodAban/Minimax-h3_Singularity · теги: gguf,comfyui,quantized,base_model:quantized:WarmBloodAban/Minimax-h3_Singularity · обновлено: 2026-10-22T06:01:36.000Z

---

---
license: other
license_name: minimax-h3-community
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE
tags:
- text-to-video
- image-to-video
- gguf
- comfyui
- minimax-h3
- singularity
- quantized
base_model: WarmBloodAban/Minimax-h3_Singularity
---

# Minimax-H3 Singularity V2 (Pruned) - GGUF Quantizations

Quantized GGUF versions of [WarmBloodAban/Minimax-H3 Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity), optimized for [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

---

## 📁 Available GGUF Files & Sizes

| Filename | File Size | Quantization Type | Recommended VRAM / Use Case |
| :--- | :--- | :--- | :--- |
| `Minimax-h3-Singularity-ref2va-v2-Pruned-Q3_K_M.gguf` | **8.9 GB** | `Q3_K_M` (3-bit Medium) | Low-VRAM systems (12 GB–16 GB VRAM) |
| `Minimax-h3-Singularity-ref2va-v2-Pruned-Q4_K_M.gguf` | **11.6 GB** | `Q4_K_M` (4-bit Medium) | **Sweet Spot** for standard GPUs (16 GB–24 GB VRAM) |
| `Minimax-h3-Singularity-ref2va-v2-Pruned-Q5_K_M.gguf` | **14.1 GB** | `Q5_K_M` (5-bit Medium) | High fidelity for 24 GB+ VRAM |
| `Minimax-h3-Singularity-ref2va-v2-Pruned-Q6_K.gguf` | **16.7 GB** | `Q6_K` (6-bit) | Near-lossless quality |

---

## 🛠️ Engineering & Pruning Details

- **Curve-Pruned AdaLN Architecture:** The 51 unpruned AdaLN modulation modules (`[Out, 2688]`) from the original 34GB checkpoint were projected onto the canonical rank-8 Ref2VA subspace (`adaln_basis` + `adaln_mean`), reducing parameters by **13.03 Billion** and saving ~26 GB of disk space.
- **Lossless Precision Guards:** All normalization layers (`RMSNorm`, `LayerNorm`), timestep embeddings (`adaln_t_table`), and modality input/output heads (`video_patch_proj`, `audio_patch_proj`, `final_layer.video_out`, `final_layer.audio_out`) are preserved in **lossless FP32**.
- **ConvRot Un-Rotation:** Source INT8 weights were accurately un-rotated using ComfyUI's exact **Regular Hadamard ($H_4^{\otimes 4}$)** transform in `float64` before K-quantization, eliminating quantization artifacts and color shifts.
- **LoRA Compatibility:** Includes the canonical `adaln_affine.safetensors` curve basis, ensuring 100% compatibility with the ComfyUI pruned H3 LoRA and PowerLoRA stack ecosystem.

---

## 🚀 How to Use in ComfyUI

1. Place your chosen `.gguf` file into your ComfyUI models folder:
   `ComfyUI/models/diffusion_models/`
2. Load the model using **`UnetLoaderGGUF`**.
3. Pair it with the Qwen3-VL text encoder and MiniMax-H3 Video + Audio VAEs.

---

## 📜 Acknowledgments & License
- **Original Model:** [WarmBloodAban/Minimax-H3_Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity)
- **Base Architecture:** MiniMax H3 by MiniMax, licensed under the **MiniMax H3 Community License Agreement**.
- **GGUF Conversion & Pruning:** By **Abiray**.
