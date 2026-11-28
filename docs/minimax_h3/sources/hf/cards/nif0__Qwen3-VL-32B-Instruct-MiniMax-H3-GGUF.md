# nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF

Источник: https://huggingface.co/nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF · скачано 2026-11-28
Загрузок: 76143 · лайков: 26 · базовая: ["unsloth/Qwen3-VL-32B-Instruct-GGUF"] · теги: gguf,base_model:unsloth/Qwen3-VL-32B-Instruct-GGUF,base_model:quantized:unsloth/Qwen3-VL-32B-Instruct-GGUF · обновлено: 2026-08-08T21:48:19.000Z

---

---
license: apache-2.0
base_model:
- unsloth/Qwen3-VL-32B-Instruct-GGUF
---
Pruned layers after 49 as they arent needed for use with MiniMax H3.

You need only the model for T2V and model + mmproj for I2V. place both the model and the mmproj file in text_encoder folder.

Can be used in ComfyUI with my fork of City96's ComfyUI-GGUF. 
https://github.com/Nif00/ComfyUI-GGUF
