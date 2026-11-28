# zzzDDs/MiniMax-H3_experimental

Источник: https://huggingface.co/zzzDDs/MiniMax-H3_experimental · скачано 2026-11-28
Загрузок: 47 · лайков: 0 · базовая: ["MiniMaxAI/MiniMax-H3","lightx2v/Minimax-h3-Turbo"] · теги: lora,comfyui · обновлено: 2026-08-26T02:04:30.000Z

---

---
license: apache-2.0
base_model:
- MiniMaxAI/MiniMax-H3
- lightx2v/Minimax-h3-Turbo
base_model_relation: adapter
pipeline_tag: image-text-to-video
tags:
- lora
- t2v
- i2v
- comfyui
- minimax-h3
---


**Contents**

| file | size |
|---|---|
| `loras/minimax_h3_fl2v_lightx2v_turbo_8step_merge_0821_bf16.safetensors` | 1.96 GB |

**FLFV 8 Step Merge 0821**

| source | weight |
|---|---|
| `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors` | 0.5 |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 1.0 |

The 4 step LoRA was trained using shift 6 and the 8 step was trained using shift 12. I made this merge to use as a video refiner only so I recommend using v1.1 on its own for a fresh generation. If you want to try using this for a full generation, use shift 6, 8 steps and 1.0 strength.
