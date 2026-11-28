# Alissonerdx/Qwen3.5-9B-Minimax-H3

Источник: https://huggingface.co/Alissonerdx/Qwen3.5-9B-Minimax-H3 · скачано 2026-11-28
Загрузок: 0 · лайков: 32 · базовая:  · теги: comfyui,lora · обновлено: 2026-11-25T18:54:56.000Z

---

---
language:
- en
license: apache-2.0
library_name: comfyui
pipeline_tag: text-to-video
tags:
- minimax-h3
- qwen3.5
- text-encoder
- projection
- lora
- comfyui
---

# Qwen3.5-9B → MiniMax-H3 Conditioning (Projection + LoRA)

> [!IMPORTANT]
> **Always update [`ComfyUI-Qwen3.5-H3`](https://github.com/alisson-anjos/ComfyUI-Qwen3.5-H3) to the latest version (`v0.2.7+`)** before pulling new weights (`v0.2.7` adds the `weights_version` selector between `v2 (latest)` and `v1`, and `v0.2.6+` fixes `tap=20` layer selection).
> - **Latest `v2` weights (`v3 Phase 2 Final — step 5289`, `score: 0.9292`) are now the default** (`minimax_h3_qwen3.5_9b_{projection,lora}_v2.safetensors`, also mirrored at the un-suffixed root names for `v0.2.6` compatibility).
> - **Original `v1` baseline is preserved at root** as `minimax_h3_qwen3.5_9b_projection_v1.safetensors` & `minimax_h3_qwen3.5_9b_lora_v1.safetensors`.
> - **Training scale:** Phase 2 distilled **21,156 training samples** (`5,289` steps) across `1,138` v1 cached samples + `10,339` MiniMax-H3 self-generated samples + `1,679` Higgsfield captions + `8,000` VidProM prompts (`text: 0.9562`, `vision: 0.9021`, `score: 0.9292`, `train_loss: 0.3109`). [`experimental/`](experimental/) holds intermediate checkpoints.

Weights to use **Qwen/Qwen3.5-9B** as a multimodal (`<Picture 1..N>` + text) conditioning encoder for **MiniMax-H3 Ref2VA** in ComfyUI.

- **ComfyUI Custom Node (GitHub):** [alisson-anjos/ComfyUI-Qwen3.5-H3](https://github.com/alisson-anjos/ComfyUI-Qwen3.5-H3) *(auto-downloads these weights)*
- **Base Qwen3.5-9B (Hugging Face):** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **ComfyUI Qwen3.5 Text Encoders (`bf16` / `int8_convrot` / `w4a8_gptq`):** [Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders)

## Root Weights

- [`minimax_h3_qwen3.5_9b_projection.safetensors`](minimax_h3_qwen3.5_9b_projection.safetensors) — `4096 -> 5120` linear + residual MLP projection (`tap=20`, Epoch 3 best)
- [`minimax_h3_qwen3.5_9b_lora.safetensors`](minimax_h3_qwen3.5_9b_lora.safetensors) — Rank-16 LoRA (`34` linear modules) trained jointly with the projection (Epoch 3 best)

### What is `minimax_h3_qwen3.5_9b_lora.safetensors`?

It is a **rank-16 LoRA** applied to `Qwen/Qwen3.5-9B` and trained **jointly** with the `4096 -> 5120` projection against MiniMax-H3's `Qwen3-VL-32B` teacher:
- **`model.visual.merger` (`linear_fc1`, `linear_fc2`)**: adapts how `<Picture 1..N>` reference image patches are projected into the language stream, boosting reference/identity fidelity (`vision` cosine `0.7873 -> 0.8879`).
- **Language layers `12–19` (`o_proj`, `gate_proj`, `up_proj`, `down_proj`, `out_proj`)**: aligns the hidden states right before the layer-20 tap (`text` cosine `0.8919 -> 0.9507`).

At load time, the ComfyUI node merges the LoRA directly into the base linear weights (`W += (alpha / r) * B @ A`), so there is **zero extra inference overhead**.

All training logs, metrics, and comparison samples are stored under [`training_artifacts/`](training_artifacts/).

## Quantitative Metrics

| Variant | Text Cosine | Vision (`refs`) Cosine | Overall Score | Video Mean PSNR | Video Frame Corr |
|---|---|---|---|---|---|
| Projection Only (No LoRA) | `0.8919` | `0.7873` | `0.8396` | `12.877 dB` | `0.4410` |
| Joint LoRA + Projection — Epoch 1 | `0.9417` | `0.8801` | `0.9109` | — | — |
| Joint LoRA + Projection — Epoch 2 | `0.9492` | `0.8867` | `0.9180` | — | — |
| **Joint LoRA + Projection — Epoch 3 (Current)** | **`0.9507`** | **`0.8879`** | **`0.9193`** | **`14.764 dB`** | **`0.6310`** |

## Side-by-Side Video Samples (Epoch 3 LoRA + Projection, Clean H3 Prompts)

### Case 01 (1 reference — PSNR `18.353 dB`, Frame Corr `0.8359`)
<video controls width="100%" src="https://huggingface.co/Alissonerdx/Qwen3.5-9B-Minimax-H3/resolve/main/training_artifacts/samples/case_01_proper_comparison.mp4"></video>

### Case 02 (2 references — PSNR `14.858 dB`, Frame Corr `0.8223`)
<video controls width="100%" src="https://huggingface.co/Alissonerdx/Qwen3.5-9B-Minimax-H3/resolve/main/training_artifacts/samples/case_02_proper_comparison.mp4"></video>

### Case 03 (3 references — PSNR `14.743 dB`, Frame Corr `0.3852`)
<video controls width="100%" src="https://huggingface.co/Alissonerdx/Qwen3.5-9B-Minimax-H3/resolve/main/training_artifacts/samples/case_03_proper_comparison.mp4"></video>

### Case 04 (4 references — PSNR `11.104 dB`, Frame Corr `0.4808`)
<video controls width="100%" src="https://huggingface.co/Alissonerdx/Qwen3.5-9B-Minimax-H3/resolve/main/training_artifacts/samples/case_04_proper_comparison.mp4"></video>
