# Abiray/Minimax-h3-Singularity-V2-Pruned

Источник: https://huggingface.co/Abiray/Minimax-h3-Singularity-V2-Pruned · скачано 2026-11-28
Загрузок: 20654 · лайков: 27 · базовая: WarmBloodAban/Minimax-h3_Singularity · теги: comfyui,quantized,base_model:quantized:WarmBloodAban/Minimax-h3_Singularity · обновлено: 2026-10-22T06:09:51.000Z

---

---
license: other
license_name: minimax-h3-community
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE
tags:
- text-to-video
- image-to-video
- safetensors
- comfyui
- minimax-h3
- int8
- w4a8
- quantized
base_model: WarmBloodAban/Minimax-h3_Singularity
base_model_relation: quantized

---

# Minimax-H3 Singularity V2 (Pruned Safetensors)

Pruned and optimized `.safetensors` versions of [WarmBloodAban/Minimax-H3 Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity), formatted for native ComfyUI custom nodes (such as `comfyui-mixed-quantizer` and native INT8 loaders).

## 📁 Available Safetensors Files & Sizes

| Filename | File Size | Quantization / Format | Recommended Use Case / Hardware |
| :--- | :--- | :--- | :--- |
| `Minimax-h3_Singularity_ref2va_v2_Pruned_int8.safetensors` | **21.0 GB** | **INT8 Backbone** + Lossless Pruned AdaLN | **100% Lossless Quality** (Direct INT8 copy with 13B AdaLN parameters pruned) |
| `Minimax-h3_Singularity_ref2va_v2_Pruned_w4a8.safetensors` | **12.5 GB** | **Asym W4A8 INT8** (ComfyUI `asym_w4a8_int8`) | **Low-VRAM Hardware** (Fitted 16-level Lloyd-Max codebook) |

---

## 🛠️ Engineering & Pruning Details

* **Curve-Pruned AdaLN Architecture:** The 51 unpruned AdaLN modulation modules (`[Out, 2688]`) from the original 34GB checkpoint were projected onto the canonical rank-8 Ref2VA subspace (`adaln_basis` + `adaln_mean`), removing **13.03 Billion parameters** and saving \~13 GB of disk space without affecting motion dynamics.
* **Lossless Precision Guards:** All normalization layers (`RMSNorm`), timestep embeddings (`adaln_t_table`), and modality input/output heads are preserved in **lossless FP32 / BF16**.
* **Regular Hadamard Un-Rotation ($H_4^{\otimes 4}$):** Source ConvRot-256 weights were accurately un-rotated using ComfyUI's exact butterfly transform in `float64` before re-quantization to eliminate color shifts and grid artifacts.
* **LoRA Compatibility:** Fully compatible with the canonical `adaln_affine.safetensors` curve basis, ensuring 100% compatibility with the ComfyUI pruned H3 LoRA and PowerLoRA stack ecosystem.

---

## 🚀 How to Use in ComfyUI

1. Place your chosen `.safetensors` file into your ComfyUI models folder:
   `ComfyUI/models/diffusion_models/`
2. Load the model using your preferred native ComfyUI loader or mixed-quantizer custom nodes.
3. Pair it with the Qwen3-VL text encoder and MiniMax-H3 Video + Audio VAEs.

---

## 📜 Acknowledgments & License

* **Original Model:** [WarmBloodAban/Minimax-H3_Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity)
* **Base Architecture:** MiniMax H3 by MiniMax, licensed under the **MiniMax H3 Community License Agreement**.
* **Pruning, Conversion & Optimization:** By **Abiray**.
