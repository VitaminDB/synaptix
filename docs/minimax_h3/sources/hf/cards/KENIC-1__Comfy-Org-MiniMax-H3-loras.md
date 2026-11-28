# KENIC-1/Comfy-Org-MiniMax-H3-loras

Источник: https://huggingface.co/KENIC-1/Comfy-Org-MiniMax-H3-loras · скачано 2026-11-28
Загрузок: 132 · лайков: 0 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: comfyui · обновлено: 2026-08-16T15:14:22.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
tags:
- comfyui
- diffusion-single-file
base_model:
- MiniMaxAI/MiniMax-H3
---

# MiniMax H3

Repackaged model files for ComfyUI.

Original model repository: 
- https://huggingface.co/MiniMaxAI/MiniMax-H3
- https://huggingface.co/lightx2v/Minimax-h3-Turbo

The Qwen3-VL-32B `nvfp4_awq` quant is converted from: https://huggingface.co/cybermotaz/Qwen3-VL-32B-Instruct-NVFP4

This `nvfp4` text encoder does not require Blackwell GPU to use.

For diffusion models prefer int8_convrot if you are able to use pytorch with cu130.

`fp8_scaled` should only be used if you cannot use `int8_convrot`.

---

Place the files in the following folders:

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   ├── minimax_h3_fl2va_bf16.safetensors
│   │   ├── minimax_h3_fl2va_int8_convrot.safetensors
│   │   ├── minimax_h3_fl2va_pruned_bf16.safetensors
│   │   ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors
│   │   ├── minimax_h3_fl2va_pruned_fp8_scaled.safetensors
│   │   ├── minimax_h3_ref2va_bf16.safetensors
│   │   ├── minimax_h3_ref2va_int8_convrot.safetensors
│   │   ├── minimax_h3_ref2va_pruned_bf16.safetensors
│   │   ├── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   │   └── minimax_h3_ref2va_pruned_fp8_scaled.safetensors
│   ├── 📂 text_encoders/
│   │   ├── qwen3vl_32b_minimax_h3_bf16.safetensors
│   │   ├── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 loras/
│   │   ├── minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
│   │   ├── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
│   │   └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_audio_vae_fp32.safetensors
│   │   └── minimax_h3_video_vae_fp16.safetensors
```

## Workflows

- I2V：https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json
- T2V：https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json
- R2V：https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json
