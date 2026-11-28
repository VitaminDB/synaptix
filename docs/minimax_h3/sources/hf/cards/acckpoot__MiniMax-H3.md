# acckpoot/MiniMax-H3

Источник: https://huggingface.co/acckpoot/MiniMax-H3 · скачано 2026-11-28
Загрузок: 0 · лайков: 1 · базовая:  · теги: comfyui · обновлено: 2026-08-04T09:05:32.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
tags:
- comfyui
---

# MiniMax H3

Repackaged model files for ComfyUI.

Original model repository: https://huggingface.co/MiniMaxAI/MiniMax-H3

Place the files in the following folders:

```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   ├── minimax_h3_fl2va_bf16.safetensors
│   │   ├── minimax_h3_fl2va_int8_convrot.safetensors
│   │   ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors
│   │   ├── minimax_h3_fl2va_pruned_fp8_scaled.safetensors
│   │   ├── minimax_h3_ref2va_bf16.safetensors
│   │   ├── minimax_h3_ref2va_int8_convrot.safetensors
│   │   ├── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   │   └── minimax_h3_ref2va_pruned_fp8_scaled.safetensors
│   ├── 📂 text_encoders/
│   │   ├── qwen3vl_32b_minimax_h3_bf16.safetensors
│   │   ├── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_audio_vae_fp32.safetensors
│   │   └── minimax_h3_video_vae_fp16.safetensors
```

## Workflows

- I2V：https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json
- T2V：https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json
- R2V：https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json
