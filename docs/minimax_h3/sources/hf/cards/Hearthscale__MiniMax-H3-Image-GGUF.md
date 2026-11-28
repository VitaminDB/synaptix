# Hearthscale/MiniMax-H3-Image-GGUF

Источник: https://huggingface.co/Hearthscale/MiniMax-H3-Image-GGUF · скачано 2026-11-28
Загрузок: 840375 · лайков: 24 · базовая: MiniMaxAI/MiniMax-H3-Image · теги: gguf,quantized,base_model:quantized:MiniMaxAI/MiniMax-H3-Image · обновлено: 2026-11-10T22:14:50.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3-Image
base_model_relation: quantized
pipeline_tag: text-to-image
library_name: gguf
language:
- en
tags:
- gguf
- stable-diffusion.cpp
- text-to-image
- image-to-image
- image-editing
- minimax
- minimax-h3
- minimax-h3-image
- quantized
widget:
- text: 'A night market food stall in Taipei, a hand-painted menu board reading "BEEF NOODLES 120" and "SCALLION PANCAKE 60", steam rising from a pot, warm lantern light, 35mm photo'
  output:
    url: assets/menu.png
- text: 'Portrait of an elderly fisherman in a yellow raincoat on a harbour wall at dawn, light rain, weathered skin and kind eyes, fishing boats behind him, 85mm photo, shallow depth of field'
  output:
    url: assets/portrait.png
- text: 'Edit of the portrait: Give him a red knitted beanie. Keep everything else exactly the same.'
  output:
    url: assets/edit.png
- text: 'A cozy bookshop cafe on a rainy evening, warm lamps, shelves up to the ceiling, a chalkboard by the door reading "OPEN LATE", two people reading by the window, rain on the glass, cinematic photo (2048 × 2048)'
  output:
    url: assets/bookshop.jpg
- text: 'Concept art of a city built into the ribs of a giant fossil in a desert, caravans with sails crossing the dunes at sunset, lanterns in the windows, matte painting, wide shot'
  output:
    url: assets/city.png
- text: 'A minimalist poster for a jazz night, a saxophone made of brass leaves on a cream background, the words "HEARTH JAZZ" in bold letters at the top and "FRIDAY 9 PM" at the bottom, Swiss design'
  output:
    url: assets/poster.png
---

# MiniMax H3-Image GGUF

GGUF files of the pruned diffusion transformer of [MiniMax H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and [Hearthscale](https://hearthscale.com). MiniMax H3-Image is a model for text-to-image and picture editing, up to 16 megapixels, distilled to 6 steps.

The files are converted from `minimax_h3_image_pruned_bf16.safetensors` in [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) with the converter of stable-diffusion.cpp `master-1088-32d516c`. They load in stable-diffusion.cpp from `master-1088`, the first build that reads MiniMax H3-Image, with no patch.

<Gallery />

Every picture above comes from the Q4_K_M file with seed 42, the Q2_K_M text encoder and stable-diffusion.cpp's own `master-1088-32d516c` release build. The edit takes the portrait as its reference.

## Files

| File | Size | Block weights | Graphics memory to hold the whole model | Quality |
|---|---|---|---|---|
| [minimax_h3_image_pruned-Q8_0.gguf](minimax_h3_image_pruned-Q8_0.gguf) | 21.4 GB | Q8_0 | 24 GB | Closest to the bf16 weights. |
| [minimax_h3_image_pruned-Q5_K_M.gguf](minimax_h3_image_pruned-Q5_K_M.gguf) | 13.9 GB | Q5_K | 16 GB | Very close to Q8_0. |
| [minimax_h3_image_pruned-Q4_K_M.gguf](minimax_h3_image_pruned-Q4_K_M.gguf) | 11.4 GB | Q4_K | 16 GB | **Recommended.** Lettering and fine detail hold. |
| [minimax_h3_image_pruned-Q3_K_M.gguf](minimax_h3_image_pruned-Q3_K_M.gguf) | 8.8 GB | Q3_K | 12 GB | Softer detail. The smallest file that keeps lettering right. |

With less graphics memory, the weights stream from system memory at each step. Every file runs on a 10 GB graphics card with 32 GB of system memory, at the times below. A 2-bit file is not published: at Q2_K, lettering and detail break.

In every file, the attention and MLP weights of the 50 blocks and of the token refiner take the type in the name. The AdaLN projections stay in F16, the norms and the condition projection in BF16, and the output heads in F32. The input projection is F32, except in the Q8_0 file, where it is Q8_0. The naming follows [leejet/MiniMax-H3-GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF).

## Measured

On an NVIDIA RTX 3080 with 10 GB and 32 GB of system memory, with the Vulkan build, the Q2_K_M text encoder, `--auto-fit on --max-vram 6` and one picture of 6 steps:

| File | 1024 × 1024, in all | Sampling |
|---|---|---|
| Q8_0 | 285 s | 265 s |
| Q5_K_M | 98 s | 76 s |
| Q4_K_M | 61 s | 38 s |
| Q3_K_M | 52 s | 31 s |

With the same setup and Q4_K_M, a 2048 × 2048 picture takes 77 s, and an edit of a 1024 × 1024 picture takes 76 s.

## Hearthscale

1. Update Hearthscale to 0.3.2, which ships stable-diffusion.cpp `master-1088`.
2. Open the Server, go to **Explore**, and search for `Hearthscale/MiniMax-H3-Image-GGUF`.
3. Choose a file. The Server offers the VAE and the text encoder beside it, with the text encoder in the size your device's memory takes.
4. Draw and edit in Image studio. Each picture takes 6 steps, on the schedule the model was distilled for.

## stable-diffusion.cpp

You also need the VAE and the text encoder:

- VAE: `vae/minimax_h3_image_vae_fp16.safetensors` from [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae)
- Text encoder: `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (18.2 GB) or `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` (13.1 GB) from [leejet/MiniMax-H3-GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF)

```sh
sd-cli --diffusion-model minimax_h3_image_pruned-Q4_K_M.gguf \
  --vae minimax_h3_image_vae_fp16.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q2_K_M.gguf \
  -p "A silver tabby kitten holding a sign reading 'sd.cpp', natural daylight" \
  --cfg-scale 1 --sampling-method euler --sigmas minimax-h3-image \
  -W 1024 -H 1024 --auto-fit on
```

To edit a picture, add `-r picture.png` and say what to change, for example `-p "Give him a red knitted beanie. Keep everything else exactly the same."`. Repeat `-r` for more reference pictures, up to 16. See the [MiniMax-H3-Image guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/minimax_h3_image.md) of stable-diffusion.cpp.

## Licence

The files follow the licence of the original model, the [MiniMax-H3-Image Community License Agreement](LICENSE). Read it before you use them: it limits where the model may be used.
