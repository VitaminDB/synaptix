# kentskooking/MiniMax-H3-AnimateDiff-LoRAs

Источник: https://huggingface.co/kentskooking/MiniMax-H3-AnimateDiff-LoRAs · скачано 2026-11-28
Загрузок: 290 · лайков: 7 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,comfyui · обновлено: 2026-11-23T00:24:15.000Z

---

---
base_model: MiniMaxAI/MiniMax-H3
pipeline_tag: image-to-video
tags:
  - lora
  - comfyui
  - minimax-h3
---

# AnimateDiff motion LoRAs for MiniMax H3

Use either LoRA with **H3 FL2VA**, with a Turbo LoRA for fast sampling or at full steps without Turbo. Both work with an empty prompt.

| LoRA | Recommended use | Optional trigger |
| --- | --- | --- |
| [General AnimateDiff FLF 768](./H3_i2v_General_AnimateDiff_FLF_768.safetensors) | Supply a first and last frame. Start at strength **1.0** and try up to **2.0** for stronger motion between them. | `animatediff` |
| [Single AnimateDiff Loop 768](./H3_i2v_Single_AnimateDiff_Loop_768.safetensors) | Supply one starting image for looping motion. Leave the prompt empty to help preserve its colors. | `pinkmorph` can turn the scene pink. |

## Technical specifications

Both LoRAs were trained with [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit) on MiniMax H3 FL2VA.

| LoRA | Optimizer steps |
| --- | ---: |
| General AnimateDiff FLF 768 | 2,500 |
| Single AnimateDiff Loop 768 | 2,000 |

Each LoRA trained rank-16 adapters on four projections per block. These were the combined query/key/value projection (`attn.qkv_proj`), the attention output (`attn.out_proj`), and both feed-forward layers (`mlp.fc1` and `mlp.fc2`). This covers all 50 main transformer blocks (0-49) and both token-refiner blocks (0-1), for **208 adapted layers** per LoRA. The text encoder was frozen.
