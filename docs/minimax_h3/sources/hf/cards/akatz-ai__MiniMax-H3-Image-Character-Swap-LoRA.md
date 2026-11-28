# akatz-ai/MiniMax-H3-Image-Character-Swap-LoRA

Источник: https://huggingface.co/akatz-ai/MiniMax-H3-Image-Character-Swap-LoRA · скачано 2026-11-28
Загрузок: 18604 · лайков: 95 · базовая: Comfy-Org/MiniMax-H3-Image · теги: lora,comfyui,base_model:Comfy-Org/MiniMax-H3-Image,base_model:adapter:Comfy-Org/MiniMax-H3-Image · обновлено: 2026-11-20T10:47:18.000Z

---

---
language:
- en
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: LICENSE
base_model: Comfy-Org/MiniMax-H3-Image
base_model_relation: adapter
pipeline_tag: image-to-image
library_name: diffusion-single-file
datasets:
- akatz-ai/H3-Character-Swap-v1
tags:
- minimax-h3-image
- lora
- character-swap
- image-editing
- comfyui
- ai-toolkit
---

# H3 Image Character Swap V1

[![Follow @akatz_ai on X](https://img.shields.io/badge/Follow-%40akatz__ai-111111?style=flat-square&logo=x&logoColor=white)](https://x.com/akatz_ai)
[![Patreon](https://img.shields.io/badge/Support-Patreon-FF424D?style=flat-square&logo=patreon&logoColor=white)](https://www.patreon.com/Akatz)
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-EA4AAA?style=flat-square&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/akatz-ai)
[![Ko-fi](https://img.shields.io/badge/Tip-Ko--fi-FF5E5B?style=flat-square&logo=kofi&logoColor=white)](https://ko-fi.com/akatz)

Replace a person in a still image with a character from a reference sheet. I trained this LoRA to bring over the character’s face, costume, and style while following the pose, framing, and lighting in the source.

[![Four matched character swaps in a 2×2 grid. Rain Rosette and Copper Eel Archivist are above Green Reptilian Adventurer and Rattler Rail Marshal. Each comparison shows the source, V1 LoRA, base H3 Image, and character reference.](assets/character-swap-comparisons.webp)](assets/character-swap-comparisons.webp)

Each comparison uses the same source, character reference, prompt, seed, and sampling settings. The V1 panel adds the LoRA at strength **1.0**. These are selected examples. Source footage for Rain Rosette comes from Blender Institute’s *Tears of Steel*, © Blender Foundation. [Project credits](https://mango.blender.org/about/) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)

Source footage for Copper Eel Archivist comes from Lacoste’s *Crocodile Inside*. Rattler Rail Marshal uses footage from *When the potion seller runs out of ingredients*. [Copper Eel source](https://www.youtube.com/watch?v=CHapY4muRRQ) · [Rattler source](https://www.youtube.com/watch?v=qin8q_sOANo)

**[Download V1 LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Image-Character-Swap-LoRA/resolve/main/h3_image_character_swap_v1.safetensors)** · Powered by MiniMax-H3-Image

## Use in ComfyUI

1. Install the files below and open the [H3 Image Character Swap V1 workflow](examples/H3-Image-Character-Swap-V1.json) in ComfyUI. The V1 LoRA is already connected at strength **1.0**.
2. To replay Rain Rosette, put the [source frame](examples/rain-rosette-source.webp) and [Rain Rosette character sheet](examples/rain-rosette-character.webp) in `ComfyUI/input/`, or upload them through the two **Load Image** nodes. Queue the workflow with the included prompt and fixed seed.
3. For your own swap, load the source into **Image 1** and the character reference into **Image 2**. Use `<Picture 1>` and `<Picture 2>` in the prompt. If the shot has multiple people, identify the target by position, clothing, or what they are doing. Set the output dimensions to around **1 megapixel** and match the source aspect ratio.

The workflow uses native nodes and was tested in ComfyUI **0.45.0** on an RTX 4090 with dynamic VRAM. It is adapted from the [official MiniMax H3 Image Edit template](https://github.com/Comfy-Org/workflow_templates/blob/c8e48bdcfc7241d7af588d20d4f924caf9d9d953/templates/image_minimax_h3_image_edit.json). [Upstream workflow license](examples/UPSTREAM-LICENSE.txt)

| Component | File | Folder in `ComfyUI/models/` |
| --- | --- | --- |
| H3 Image INT8 ConvRot | [`minimax_h3_image_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors) | `diffusion_models/` |
| Qwen3-VL text encoder | [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/89c635085f539d03ad35485ed542c5a286e3c6d6/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `text_encoders/` |
| H3 Image VAE | [`minimax_h3_image_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/vae/minimax_h3_image_vae_fp16.safetensors) | `vae/` |
| V1 adapter | [`h3_image_character_swap_v1.safetensors`](https://huggingface.co/akatz-ai/MiniMax-H3-Image-Character-Swap-LoRA/resolve/main/h3_image_character_swap_v1.safetensors) | `loras/` |

| Setting | Tested value |
| --- | --- |
| LoRA strength | 1.0 |
| Steps | 6 |
| CFG | 1 |
| Sampler | `euler` |
| Scheduler | `simple` |
| Denoise | 1.0 |
| Negative conditioning | `ConditioningZeroOut` |
| Prompt enhancer | Off |

### Example prompt

> Replace the woman in the dark jacket in `<Picture 1>` with the character from `<Picture 2>`. Keep her position, pose, and expression. Preserve the background and everyone else in the scene. Carry over the reference character’s costume and materials, and match the light in the source shot.

## Training and limitations

V1 is a rank-16 LoRA trained with AI Toolkit on paired image edits. Each pair uses a source image, a character sheet, an instruction, and the edited target. The earlier public V1 swap dataset is part of the training history. The final training stage used 140 pairs, with 18 separate validation examples. [Earlier V1 dataset](https://huggingface.co/datasets/akatz-ai/H3-Character-Swap-v1)

The adapter is intended for people and humanoid creatures. Check small or occluded figures, hands near props, and clothing that blends into the background. The edit can preserve pieces of the original outfit, change details around the target, or pick the wrong person. This release has been evaluated on still images with the INT8 ConvRot base. BF16 and other quantizations are unverified.

## File identity

| Field | Value |
| --- | --- |
| File | `h3_image_character_swap_v1.safetensors` |
| Size | 155,110,448 bytes |
| Rank | 16 |
| SHA-256 | `68378574bede5ca9d7d69df8bb8f8d9f7860d07ea45b26d9c67a7babf50d330f` |

## License and attribution

Use and redistribution are subject to the [MiniMax-H3-Image Community License Agreement](LICENSE), including its territorial restrictions and separate authorization requirements. See [NOTICE](NOTICE) for attribution.

## 💜 Support Akatz Labs

If this helps with your work, you can help fund the next round of model training and ComfyUI workflows. Thanks for trying it.

[GitHub Sponsors](https://github.com/sponsors/akatz-ai) · [Patreon](https://www.patreon.com/Akatz) · [Ko-fi](https://ko-fi.com/akatz) · [Explore Akatz Labs](https://akatz.ai)
