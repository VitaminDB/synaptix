# angelarose210/Block-split_turbo_LoRA_MiniMax_H3_hybrid_blocks_25-49_only

Источник: https://huggingface.co/angelarose210/Block-split_turbo_LoRA_MiniMax_H3_hybrid_blocks_25-49_only · скачано 2026-11-28
Загрузок: 0 · лайков: 1 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги:  · обновлено: 2026-10-22T11:33:24.000Z

---

---
license: apache-2.0
base_model:
- MiniMaxAI/MiniMax-H3
---
--- Block-split turbo LoRA for MiniMax H3 hybrid models: blocks 25-49 only.

Source: RunningHubAI minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16 on Hugging Face. Not retrained. The original weights were copied with blocks 0-24 and token_refiner removed. All credit to the original author.

Made for hybrids whose blocks 25-49 come from Ref2VA, e.g. minimax_h3_hybrid_fl2va_ref2va_b25-49. This half covers the Ref2VA blocks.

Use with: H3_hybrids_blocks_0-24_plus_token_refiner_only (the other half). Chain both with plain LoraLoaderModelOnly nodes.
Contains: blocks 25-49 only.
Starting strength: 1.0 each. Lower to 0.7-0.8 if the result looks overbaked.
Steps: 8.
use with https://huggingface.co/angelarose210/A_MiniMax_H3_hybrids_blocks_0-24_plus_token_refiner_only
license: apache-2.0
base_model:
- MiniMaxAI/MiniMax-H3
license: apache-2.0
---
