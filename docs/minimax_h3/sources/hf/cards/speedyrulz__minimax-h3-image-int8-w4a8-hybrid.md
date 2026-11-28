# speedyrulz/minimax-h3-image-int8-w4a8-hybrid

Источник: https://huggingface.co/speedyrulz/minimax-h3-image-int8-w4a8-hybrid · скачано 2026-11-28
Загрузок: 4256 · лайков: 7 · базовая: ["MiniMaxAI/MiniMax-H3-Image"] · теги: comfyui · обновлено: 2026-11-10T20:07:35.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE
tags:
- comfyui
- diffusion-single-file
base_model:
- MiniMaxAI/MiniMax-H3-Image
pipeline_tag: text-to-image
---

# MiniMax-H3-Image int8 + w4a8 hybrid for ComfyUI

`diffusion_models/minimax_h3_image_pruned_int8_w4a8_hybrid.safetensors` is a mixed-precision repack of the
[Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) diffusion model, sized for 16 GB
GPUs. It is a drop-in replacement in the official text-to-image workflow (ComfyUI 0.45.0 or newer).

| Part of the model | Format | Taken from |
|---|---|---|
| Blocks 0-29: `qkv_proj`, `out_proj`, `fc1`, `fc2` | `asym_w4a8_int8` | `minimax_h3_image_pruned_w4a8` |
| Blocks 30-49 and every other tensor | `int8_tensorwise` (ConvRot) | `minimax_h3_image_pruned_int8_convrot` |

The tensors are copied unchanged from the two Comfy-Org files; nothing was requantized. SHA-256:
`ea736af3b44f0f8be32e3ec632e375fea0449a8c0b2e05fe121ede5719a0137b`.

## Why this split

The last 20 blocks are the ones that suffer from 4-bit weights, so they stay int8. Measured against the bf16
model at 2048x2048, 6 steps:

| Model | Size | Output error per step | SSIM to bf16 image |
|---|---|---|---|
| int8 ConvRot (official) | 19.5 GiB | 3.5% | 0.888 |
| **this hybrid** | **14.8 GiB** | **5.2%** | **0.844** |
| w4a8 (official) | 11.7 GiB | 9.1% | 0.767 |

Output error is the relative error of the model's prediction when every variant is fed the bf16 model's own
inputs, averaged over 6 steps and 2 prompts. SSIM is the mean over 6 images (2 prompts, 3 seeds) generated with
the model's default attention.

## Speed

6 steps at 2048x2048 with the official workflow. The smaller files finish sooner because less has to be moved
onto the GPU: on the RTX 5060 Ti that shortens only the load before the first step, on the RTX 4060 Ti (slower
PCIe link) every step is faster too.

| GPU | int8 | hybrid | w4a8 |
|---|---|---|---|
| RTX 5060 Ti 16 GB | 1.86 s/step, 21-23 s total | 1.85 s/step, 16-17 s total | 1.85 s/step, 16 s total |
| RTX 4060 Ti 16 GB | 2.72 s/step, 27-28 s total | 2.33 s/step, 20-21 s total | 2.27 s/step, 19-20 s total |

Totals are the six sampling steps including loading the model onto the GPU.

## Why there is no NVFP4 in it

NVFP4 was tested for every layer type and left out. On this model an NVFP4 layer loses 8-12% of its output on
real inputs, against about 1% for int8 ConvRot and 5-6% for w4a8, and half of that loss comes from NVFP4's 4-bit
activations, so better weight rounding or calibration cannot remove it. The best NVFP4 mix found (calibrated NVFP4
on `qkv_proj`, `out_proj` and `fc1` of blocks 0-29, int8 elsewhere) reached 8.1% output error at 15.5 GiB and only
runs fast on RTX 50 cards.

## Use

Put the file in `ComfyUI/models/diffusion_models` and select it in the workflow's diffusion model loader. It
uses ComfyUI's int8 kernels only, so it does not need FP4 hardware.

Original model: [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), under the MiniMax
H3 Image community license.
