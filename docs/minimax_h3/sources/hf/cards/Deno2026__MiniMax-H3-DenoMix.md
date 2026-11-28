# Deno2026/MiniMax-H3-DenoMix

Источник: https://huggingface.co/Deno2026/MiniMax-H3-DenoMix · скачано 2026-11-28
Загрузок: 98993 · лайков: 45 · базовая: ["MiniMaxAI/MiniMax-H3","WarmBloodAban/Minimax-h3_Singularity"] · теги: comfyui · обновлено: 2026-11-06T01:09:05.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model:
- MiniMaxAI/MiniMax-H3
- WarmBloodAban/Minimax-h3_Singularity
base_model_relation: merge
pipeline_tag: image-text-to-video
library_name: diffusion-single-file
tags:
- minimax-h3
- reference-to-video
- video-generation
- comfyui
- int8
- merge
---

# MiniMax-H3 DenoMix — R2V pruned INT8 v3

A MiniMax H3 reference-to-video (R2V) diffusion model. It is built on the original MiniMax H3 R2V weights and was fine-tuned by merging the [Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity) fine-tune together with a few additional weights into a single file.

Powered by MiniMax H3.

## License

This model follows the [MiniMax H3 Community License Agreement](LICENSE), since it is built on the original MiniMax H3 weights (modified by DENO, October 2026 — see [NOTICE](NOTICE)).

## File

| | |
|---|---|
| File | `MiniMax-H3_DenoMix_ref2va_pruned_int8_v3.safetensors` |
| Size | 20,967,647,432 bytes |
| SHA-256 | `6fc735bb1d2e9e44951106b3ec6a4c063df1535c567bc7d44aaff4c9052add52` |
| Format | pruned INT8 (ConvRot) diffusion model, one file |

## Use in ComfyUI

1. Put the file in `ComfyUI/models/diffusion_models/`.
2. In your MiniMax H3 R2V workflow, select it in **Load Diffusion Model** (`UNETLoader`) in place of the MiniMax H3 R2V pruned INT8 diffusion model. The text encoder and VAEs stay the same.

A ComfyUI version that supports the MiniMax H3 pruned INT8 (ConvRot) format is required.

## ComfyUI workflow

[`workflows/Deno_MiniMax_H3_R2V_16GB_Tiled_Sampler.json`](workflows/Deno_MiniMax_H3_R2V_16GB_Tiled_Sampler.json) — a 16GB-VRAM MiniMax H3 R2V workflow built around this model:
it generates at 960×544 with the Pika LoRA (6 steps), upscales the latent ×2 with ComfyUI's built-in MiniMax H3 latent upscaler, and then the **DENO Tiled Sampler** finishes the last DMD step tile by tile at 1920×1088.
Tested on an RTX 4060 Ti 16GB with 32GB system RAM: a 15-second clip at 24 fps. It needs ComfyUI v0.44.0 or newer and [DENO Custom Nodes](https://github.com/Deno2026/comfyui-deno-custom-nodes) v0.7.126 or newer for the tiled sampler node.

[`workflows/Deno_MiniMax_H3_R2V_Turbo_8plus2.json`](workflows/Deno_MiniMax_H3_R2V_Turbo_8plus2.json) — a MiniMax H3 R2V workflow built around this model:
it first generates at a low resolution with the Alibaba turbo LoRA (8 steps), upscales the latent ×2, and then refines with the DMD turbo LoRA and NVIDIA's refiner (Kijai's light extraction) over the last 2 steps.
Each workflow's note lists every model file it uses with its download link. Drag the file onto the ComfyUI canvas to open it.

Workflow license: [GPL-3.0-only](workflows/LICENSE). The models and custom nodes it loads keep their own licenses.

## Credits

- [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) by MiniMax — base model.
- [Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity) by WarmBloodAban ([Apache-2.0](LICENSE-Singularity-Apache-2.0.txt)) — merged into this model.
