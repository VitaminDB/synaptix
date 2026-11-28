# BennyDaBall/MiniMax-H3-Image-ComfyUI-NVFP4

Источник: https://huggingface.co/BennyDaBall/MiniMax-H3-Image-ComfyUI-NVFP4 · скачано 2026-11-28
Загрузок: 20219 · лайков: 10 · базовая: ["MiniMaxAI/MiniMax-H3-Image","Comfy-Org/MiniMax-H3-Image"] · теги: comfyui,quantized,base_model:Comfy-Org/MiniMax-H3-Image,base_model:quantized:Comfy-Org/MiniMax-H3-Image · обновлено: 2026-11-16T16:30:16.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE
base_model:
- MiniMaxAI/MiniMax-H3-Image
- Comfy-Org/MiniMax-H3-Image
base_model_relation: quantized
pipeline_tag: text-to-image
library_name: diffusion-single-file
tags:
- comfyui
- nvfp4
- quantized
- diffusion-single-file
- image-editing
- blackwell
- schur-replay
---

# ⚡ MiniMax H3 Image - NVFP4 for ComfyUI

![MiniMax H3 Image NVFP4 for ComfyUI, by BennyDaBall](banner.png)

**Follow me on X [@BennyDaBall_OG](https://x.com/BennyDaBall_OG) !**

**Want to support my work? Buy me a coffee on [Ko-fi](https://ko-fi.com/bennydaball) !**

**Full W4A4. 12.5 GB. 16 megapixels in 13 seconds on one RTX 5090.**

A native ComfyUI NVFP4 build of [MiniMax H3 Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), the image model that grew out of the H3 video model. Weights and activations are both 4-bit. On finished text-to-image pictures it lands as close to the full-precision model as the official W4A8 file, and it renders up to 20% faster.

- **Built for Blackwell, on Blackwell:** made on two RTX PRO 6000 Blackwell cards and one RTX 5090, for every Blackwell machine (RTX 50 series, RTX PRO Blackwell, DGX Spark).
- **True W4A4:** all 200 transformer matrices are NVFP4, with 4-bit weights and 4-bit activations. Nothing is held back at 8 bits.
- **Text to image and editing:** calibrated on both, at 4 MP and at 16 MP, on real 6-step denoiser runs.
- **Drop-in:** one file, the stock loader, the stock text encoder and VAE. No custom nodes.
- **Down to the activation floor:** from step 3 of 6 on, the only error left is the 4-bit activation rounding of the format itself. Numbers below.

![A woman in an emerald evening gown and welding gauntlets beside a ship hull at night, sparks falling behind her](assets/hero/hero01_gown_welder_preview.jpg)

Every picture on this page came out of this file in stock ComfyUI at 16 MP, 6 steps, no prompt expander. Full-size PNGs are in [assets/hero](assets/hero), and the prompts are in [assets/PROMPTS.md](assets/PROMPTS.md).

![A cream and green diesel train running on rails across the open sea at sunset, a fisherman waving from a rowing boat](assets/hero/hero02_sea_train_preview.jpg)

![A rusted container ship ploughing through a golden wheat field, wheat curling off the bow like a wave](assets/hero/hero03_wheat_freighter_preview.jpg)

![An old teal door standing alone on a salt flat with a waterfall pouring through it](assets/hero/hero04_door_waterfall_preview.jpg)

![A Boston Terrier the size of a building asleep in a city intersection, traffic stopped around it](assets/hero/hero05_giant_terrier_preview.jpg)

## Editing

Same file, same six steps. Each hero picture above went back in as `<Picture 1>` with a written edit prompt, at 16 MP.

![Gown changed from emerald to scarlet](assets/edits/edit01_gown_welder_before_after.jpg)

![Sunset changed to a moonlit night](assets/edits/edit02_sea_train_before_after.jpg)

![Wheat field changed to lavender in bloom](assets/edits/edit03_wheat_freighter_before_after.jpg)

![Waterfall changed to lava](assets/edits/edit04_door_waterfall_before_after.jpg)

![Red knitted sweater and snowfall added](assets/edits/edit05_giant_terrier_before_after.jpg)

Full-size edits are in [assets/edits](assets/edits).

## The numbers

Measured on my RTX 5090 in stock ComfyUI 0.45.0 with the stock workflow graph: euler, simple, 6 steps, CFG 1. Time is the whole prompt, warm, median of 5: sampling, VAE decode and PNG save.

| Transformer file | Size | 2048x2048 | 4096x4096 |
|---|---:|---:|---:|
| Original BF16 | 40.23 GB | 8.24 s | 27.40 s |
| Official W4A8 | 12.55 GB | 3.68 s | 14.52 s |
| **NVFP4, this repo** | **12.54 GB** | **3.07 s** | **12.93 s** |

- **2.7x the BF16 speed** at 2048x2048 and 2.1x at 4096x4096.
- **20% faster than the official W4A8** at 2048x2048 and 12% faster at 4096x4096.
- **Level on text-to-image quality:** over 32 prompts at 16 MP, mean LPIPS against the BF16 render is 0.303 for this file and 0.304 for the official W4A8.

Side-by-side sheets against BF16 and the official W4A8, the per-set tables and the edit scores are in [comparisons](comparisons).

### Down to the activation floor

NVFP4 rounds the input of every quantized layer to 4 bits, on top of the 4-bit weights. That activation rounding costs something even with perfect weights, and no weight work can remove it. So I measured it: the error of one denoiser call against the full-precision model, first with the original weights and only the 4-bit activations switched on (plus the INT8 attention every shipped file uses), then with this file.

| Step | 4-bit activations only | This file |
|---:|---:|---:|
| 1 | 0.201 | 0.230 |
| 2 | 0.139 | 0.206 |
| 3 | 0.109 | 0.108 |
| 4 | 0.063 | 0.063 |
| 5 | 0.041 | 0.042 |
| 6 | 0.024 | 0.024 |

From step 3 on the two columns are the same number. The weight error is gone there, and what is left is the format. (Relative error of the predicted clean image, 24 calls at 4 MP, lower is better.)

## How I did it

The shape of it, without the recipe.

- **Teacher and student:** the full-precision model is the teacher, the parent. The 4-bit file is the student, the child. Nothing is rounded and hoped for: each of the 200 matrices is solved so the student does what the teacher does on real denoiser runs.
- **Calibrated the way the model is used:** text to image and editing, at 4 MP and at 16 MP, across the whole 6-step schedule.
- **GPTQ, taken further:** the weights come out of my own complex derivatives of GPTQ ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323)). Every rounding choice is corrected against what the layer actually sees, instead of rounding weight by weight.
- **Schur Replay scoring with Four Over Six:** each 4-bit block scale is picked by Schur Replay scoring ([arXiv 2609.36654](https://arxiv.org/abs/2609.36654)), with the Four Over Six points ([arXiv 2512.02010](https://arxiv.org/abs/2512.02010)) among the options. A scale wins by how well it survives the solver's own error correction, not by how well it fits the raw weights.
- **An E4 pass to finish:** one more pass over the 4-bit codes and their E4M3 block scales after the solve.
- **Checked like a release:** every tensor is read back after writing and decoded with the same math ComfyUI uses, then the file is scored on runs it never saw.

All three papers were written for language models. Carrying them over to an image transformer, and everything built on top, is my own work.

## Use it

| Download | ComfyUI folder | Size |
|---|---|---:|
| [Transformer (NVFP4)](diffusion_models/MiniMax-H3-Image-ComfyUI-NVFP4.safetensors) | `models/diffusion_models/` | 12.54 GB |
| [Text encoder (NVFP4)](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, from Comfy-Org | `models/text_encoders/` | 15.69 GB |
| [VAE](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/blob/main/vae/minimax_h3_image_vae_fp16.safetensors) `minimax_h3_image_vae_fp16.safetensors`, from Comfy-Org | `models/vae/` | 1.17 GB |

**29.40 GB for the whole NVFP4 stack.** The text encoder is Comfy-Org's own NVFP4 build of Qwen3-VL 32B for MiniMax H3. Their INT8 ConvRot encoder (`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`, 27.14 GB) works just as well if you already have it; pick it in the **Load CLIP** node.

1. Update ComfyUI, then put the three files in their folders.
2. Drag [MiniMax-H3-Image-NVFP4_Text_to_Image.json](workflows/MiniMax-H3-Image-NVFP4_Text_to_Image.json) into ComfyUI and click **Run**. It is the stock Comfy-Org template with this file and the NVFP4 text encoder selected and 16 MP set.
3. For editing, copy [input/hero05_giant_terrier.png](input/hero05_giant_terrier.png) into `ComfyUI/input/` and load [MiniMax-H3-Image-NVFP4_Image_Edit.json](workflows/MiniMax-H3-Image-NVFP4_Image_Edit.json).

Or pick the file in the **Load Diffusion Model** node of any MiniMax H3 Image workflow, `weight_dtype` default. Nothing else changes.

- **Settings:** euler, simple, 6 steps, CFG 1.
- **Resolution:** the model likes it big. 4 MP is the floor, 16 MP is where it shines.
- **Prompts:** the prompt expander is off in both workflows. Write a full paragraph: subject, setting, light, lens. For edits, say what changes and what stays.
- **Hardware:** the FP4 speed-up needs an NVIDIA Blackwell GPU (RTX 50 series, RTX PRO Blackwell, DGX Spark).

## Files

| File | Size | sha256 |
|---|---:|---|
| `diffusion_models/MiniMax-H3-Image-ComfyUI-NVFP4.safetensors` | 12,537,569,930 bytes | `c4580a9c095bd6d133f2d7f78f247bbd0f075d9a44c53de30fd8ecd1c7afc6a1` |

- **200 NVFP4 matrices:** `qkv_proj`, `out_proj`, `fc1` and `fc2` of all 50 blocks, in the stock ComfyUI `nvfp4` layout.
- **Everything else untouched:** the other 378 tensors are byte-identical to `minimax_h3_image_pruned_bf16.safetensors`, and the 6-step schedule stays in the file header.

## Test setup

ComfyUI 0.45.0, comfy-kitchen 0.2.41, torch 2.14.0 + CUDA 13.0, one RTX 5090, launched with `--use-ck-attention`. Text encoder `qwen3vl_32b_minimax_h3_int8_convrot` (the template default) and VAE `minimax_h3_image_vae_fp16` for every file, pictures and numbers alike. Same prompt and seed for every file in a comparison.

**Quantized by BennyDaBall. Built on two RTX PRO 6000 Blackwell cards and one RTX 5090.**

## License and credits

MiniMax H3 Image is made by [MiniMax](https://huggingface.co/MiniMaxAI) and released under the [MiniMax H3 Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE). This repository is a quantized derivative, built from the [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) repackaged BF16 file; see [NOTICE.txt](NOTICE.txt). It is not an official MiniMax or Comfy-Org product.

Thanks to MiniMax for the model, to Comfy-Org and comfy-kitchen for the NVFP4 runtime and the workflows, and to the authors of GPTQ, Schur Replay and Four Over Six.

**Follow me on X [@BennyDaBall_OG](https://x.com/BennyDaBall_OG) !**

**Want to support my work? Buy me a coffee on [Ko-fi](https://ko-fi.com/bennydaball) !**
