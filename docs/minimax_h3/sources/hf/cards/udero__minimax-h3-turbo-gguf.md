# udero/minimax-h3-turbo-gguf

Источник: https://huggingface.co/udero/minimax-h3-turbo-gguf · скачано 2026-11-28
Загрузок: 4358 · лайков: 1 · базовая: ["unsloth/MiniMax-H3-GGUF"] · теги: gguf,turbo,base_model:unsloth/MiniMax-H3-GGUF,base_model:quantized:unsloth/MiniMax-H3-GGUF · обновлено: 2026-11-08T01:40:21.000Z

---

---
base_model:
- unsloth/MiniMax-H3-GGUF
tags:
- gguf
- text-to-video
- image-to-video
- video-generation
- audio-generation
- minimax-h3
- turbo
- 8-step
pipeline_tag: text-to-video
license: other
---

# MiniMax-H3 FL2VA 8-Step Turbo (Permanently Baked GGUF Suite)

This repository hosts standalone, permanently baked **8-step distilled Turbo** versions of the **MiniMax-H3 FL2VA** foundation model (generating high-definition video with native synchronized stereo audio).

### Why Baked GGUFs?
In standard workflows, MiniMax-H3 requires 50 steps and loading separate LoRA adapters at runtime. Applying runtime adapters to quantized GGUF integer weights causes either massive VRAM dequantization spikes (triggering OOM crashes on consumer GPUs) or slow dual-branch execution.

These models were created by permanently fusing the official **MiniMax-H3 8-step Turbo LoRA** (`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`) directly into the base weights offline via **FP32 matrix fusion**:

$$W_{	ext{baked}} = 	ext{Quantize}\left(	ext{Dequantize}(W_{	ext{base}}) + rac{lpha}{r}(B 	imes A)ight)$$

Zero runtime LoRA overhead, zero extra VRAM required, and 8-step generation works out of the box.

---

### Complete Model Collection & Direct Download Links

All five models are available in the [Files and versions](https://huggingface.co/udero/minimax-h3-turbo-gguf/tree/main) tab:

| Quantization | Baked Model File (Download) | Unsloth Base Model File | Size | VRAM / Hardware Profile |
| :--- | :--- | :--- | :--- | :--- |
| **UD-Q3_K_XL** | [**`minimax_h3_fl2va_turbo_baked-UD-Q3_K_XL.gguf`**](https://huggingface.co/udero/minimax-h3-turbo-gguf/resolve/main/minimax_h3_fl2va_turbo_baked-UD-Q3_K_XL.gguf) | `minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf` | **8.90 GiB** | **Ultra-lightweight champion**: Unsloth Dynamic mixed-precision (Q5_K key projections, Q2_K secondary). Lowest memory footprint (~3.1 GB headroom on 12GB GPUs) with near-Q4 fidelity. |
| **Q4_K_M** | [**`minimax_h3_fl2va_turbo_baked-Q4_K_M.gguf`**](https://huggingface.co/udero/minimax-h3-turbo-gguf/resolve/main/minimax_h3_fl2va_turbo_baked-Q4_K_M.gguf) | `minimax_h3_fl2va_pruned-Q4_K.gguf` | **10.64 GiB** | **Fast & balanced**: fastest PCIe dynamic streaming; fits easily on 12 GB GPUs. |
| **Q5_0** | [**`minimax_h3_fl2va_turbo_baked-Q5_0.gguf`**](https://huggingface.co/udero/minimax-h3-turbo-gguf/resolve/main/minimax_h3_fl2va_turbo_baked-Q5_0.gguf) | `minimax_h3_fl2va_pruned-Q5_0.gguf` | **12.97 GiB** | **Sweet spot**: exceptional balance of speed, memory footprint, and quality. |
| **Q6_K** | [**`minimax_h3_fl2va_turbo_baked-Q6_K.gguf`**](https://huggingface.co/udero/minimax-h3-turbo-gguf/resolve/main/minimax_h3_fl2va_turbo_baked-Q6_K.gguf) | `minimax_h3_fl2va_pruned-Q6_K.gguf` | **15.45 GiB** | **High fidelity**: sharper micro-textures, feathers, text, and edge definition. |
| **Q8_0** | [**`minimax_h3_fl2va_turbo_baked-Q8_0.gguf`**](https://huggingface.co/udero/minimax-h3-turbo-gguf/resolve/main/minimax_h3_fl2va_turbo_baked-Q8_0.gguf) | `minimax_h3_fl2va_pruned-Q8_0.gguf` | **19.97 GiB** | **Near-lossless**: closest fidelity to unquantized FP16; recommended for 64GB+ RAM rigs. |

---

### Model Details
* **Base Model Lineage**: [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) (FL2VA: First-frame Latent / Image-to-Video & Stereo Audio variant)
* **Fused Adapter**: Official MiniMax-H3 8-Step Turbo LoRA (`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`)
* **Sampling Schedule**: **8 steps** (Euler) with CFG 1.0
* **Output**: Synchronized video + 32 kHz native stereo audio
* **VRAM Friendly**: Runs on a single 12 GB GPU (e.g., RTX 3060 12GB) via dynamic layer streaming (`--auto-fit on`).

---

### How to Use

#### 1. Quick Download via `hf` CLI
```bash
# Download the ultra-lightweight Unsloth Dynamic UD-Q3_K_XL (8.90 GiB):
hf download udero/minimax-h3-turbo-gguf minimax_h3_fl2va_turbo_baked-UD-Q3_K_XL.gguf

# Or download the balanced Q4_K_M or Q5_0:
hf download udero/minimax-h3-turbo-gguf minimax_h3_fl2va_turbo_baked-Q4_K_M.gguf
hf download udero/minimax-h3-turbo-gguf minimax_h3_fl2va_turbo_baked-Q5_0.gguf

# Or download the high-fidelity Q6_K:
hf download udero/minimax-h3-turbo-gguf minimax_h3_fl2va_turbo_baked-Q6_K.gguf
```

#### 2. Running in `stable-diffusion.cpp` (`sd-cli`)
```bash
sd-cli -M vid_gen \
  --diffusion-model minimax_h3_fl2va_turbo_baked-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "A cinematic drone shot flying over misty mountains at sunrise" \
  --cfg-scale 1.0 \
  --steps 8 \
  --video-frames 73 \
  --fps 24 \
  --auto-fit on \
  -o output.mp4
```

#### 3. Running in ComfyUI
Load via standard `UnetLoaderGGUF` directly into the sampling pipeline or dual-GPU splitter. No `LoraLoader` node is needed.

---

### Additional Resources & Deep Dive
> 📖 **Technical Write-up & Benchmarks**: Read the full article detailing the experiments, memory analysis, and performance tuning:  
> [**Running MiniMax-H3 on a Single RTX 3060 12GB GPU: My Experiment with GGUF, LoRA, and Dynamic Loading**](https://udero.io/blog/running-minimax-h3-on-a-single-rtx-3060-12gb-gpu-my-experiment-with-gguf-lora-and-dynamic-loading)

- **Base Model Repository**: [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)
- **Turbo Adapter**: [MiniMax-H3 8-Step Turbo LoRA (Comfy-Org)](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/loras)
