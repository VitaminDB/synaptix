# QrusherZA/H3_Turbo_ComfyUI

Источник: https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI · скачано 2026-11-28
Загрузок: 1915 · лайков: 45 · базовая: larryvrh/MiniMax-H3-Turbo-Lora · теги: comfyui,lora,base_model:larryvrh/MiniMax-H3-Turbo-Lora,base_model:adapter:larryvrh/MiniMax-H3-Turbo-Lora · обновлено: 2026-11-17T10:45:41.000Z

---

---
base_model: larryvrh/MiniMax-H3-Turbo-Lora
license: apache-2.0
pipeline_tag: text-to-video
tags:
- comfyui
- minimax-h3
- lora
- audio-video
---

# MiniMax H3 Turbo LoRA for ComfyUI

This repository provides ComfyUI conversions of **Larryvrh’s MiniMax H3 Turbo LoRA**, an adapter for few-step MiniMax H3 video generation with synchronized audio. The original Turbo training and weights are Larryvrh’s work. My contribution is the ComfyUI key conversion and the pruned-compatible variants. [Original Turbo LoRA and training notes](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)

## Choose a LoRA

Start with **v4 step600 EMA** and match the file to your **base model’s layout**. Use the standard conversion for a full-width base and the `_pruned` conversion for a pruned base. A base can be both quantized and full-width, so check the model name or card for its layout.

| Download | Size | Use |
| --- | ---: | --- |
| [`minimax_h3_turbo_v4_step600_ema_comfy_pruned.safetensors`](https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI/resolve/main/minimax_h3_turbo_v4_step600_ema_comfy_pruned.safetensors) | 620 MB | Recommended v4 EMA starting point for pruned bases. |
| [`minimax_h3_turbo_v4_step600_ema_comfy.safetensors`](https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI/resolve/main/minimax_h3_turbo_v4_step600_ema_comfy.safetensors) | 780 MB | Recommended v4 EMA starting point for full-width bases. |
| [`minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors`](https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors) | 620 MB | Earlier ckpt500 release for pruned bases. |
| [`minimax_h3_turbo_4step_ckpt500_comfyui.safetensors`](https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt500_comfyui.safetensors) | 780 MB | Earlier ckpt500 release for full-width bases. |
| [`minimax_h3_turbo_4step_comfyui_pruned.safetensors`](https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI/resolve/main/minimax_h3_turbo_4step_comfyui_pruned.safetensors) | 620 MB | Initial 4-step release for pruned bases. |

In `v4_step600` and `ckpt500`, the numbers refer to training checkpoints. Set the sampling step count in your workflow.

## Install in ComfyUI

1. Start with a working MiniMax H3 workflow and install its base diffusion model, text encoder, and video and audio VAEs. The [ComfyUI MiniMax H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) has workflows and model setup instructions.
2. Download one LoRA from the table and put it in `ComfyUI/models/loras/`.
3. Insert **Load LoRA (Model Only)** (`LoraLoaderModelOnly`) on the `MODEL` connection from your diffusion model loader. Select the downloaded file and set `strength_model` to `1.0`.
4. Connect the LoRA node’s `MODEL` output to the parts of the workflow that consume the model, such as the guider and scheduler. Keep the H3 workflow’s video and audio sampling path intact.

## Sampling settings

These are starting points from the upstream Turbo release. Results depend on your base model, the LoRA variant, and the sampler in your workflow.

| Setting | Starting point |
| --- | --- |
| LoRA strength | `1.0` |
| Steps | `6–8` for v4 EMA, or `4` for a quick test |
| Scheduler | `simple` |

If v4 produces trailing or ghosting during fast movement at four steps, try six to eight steps on the same seed. For Larryvrh’s workflow and sampler implementation, use the original weights with the [upstream Turbo custom nodes](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo).

## What changed in the conversion

- The converted files add ComfyUI’s `diffusion_model.` prefix to the LoRA keys so the loader can map the adapter to the H3 diffusion model.
- The `_pruned` files omit AdaLN-related entries and other incompatible entries so the retained LoRA patches can be applied to pruned H3 layouts.

I shared the conversion and stripping scripts in [the repository discussion](https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI/discussions/3).

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| LoRA is missing from the dropdown | Check that the file is in `ComfyUI/models/loras/` and refresh the model list or restart ComfyUI. |
| Console reports unmatched LoRA keys or shape mismatches | Check that the base is MiniMax H3 and that the LoRA variant matches its full-width or pruned layout. Custom merges can change which target layers are available. |
| Clip smears during fast movement at four steps | Try six to eight steps with v4 EMA and compare the same seed. |

When reporting an issue, include the full LoRA and base-model filenames, your ComfyUI version, your sampler and scheduler, and the relevant console output. [Open a discussion](https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI/discussions)

## Credits and support

Credit for the Turbo LoRA goes to [Larryvrh](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora). Credit for the base model goes to [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3). QrusherZA maintains these ComfyUI conversions. The upstream adapter is released under Apache 2.0.

If these ComfyUI conversions helped you, you can [support QrusherZA on PayPal](https://paypal.me/ThirenGovender).
