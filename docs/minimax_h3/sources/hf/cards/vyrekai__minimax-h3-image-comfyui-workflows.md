# vyrekai/minimax-h3-image-comfyui-workflows

Источник: https://huggingface.co/vyrekai/minimax-h3-image-comfyui-workflows · скачано 2026-11-28
Загрузок: 0 · лайков: 128 · базовая:  · теги: comfyui · обновлено: 2026-11-15T03:07:21.000Z

---

---
license: apache-2.0
tags:
  - comfyui
  - workflow
  - minimax-h3
  - text-to-image
  - image-editing
---

# MiniMax H3 Image: ComfyUI workflows for 8, 12 and 16 GB VRAM

Three versions of one ComfyUI workflow for **MiniMax H3 Image**. **All three make full 16 MP images** (3:4 = 3552x4736, 16:9 = 5472x3072), even on an 8 GB card. ComfyUI streams the model through the card in parts, so a smaller card gives **the same image, only slower**. Pick the file that matches your VRAM and drag it into ComfyUI.

| File | For | Final image | Draft | VAE decode | Reference pictures |
|---|---|---|---|---|---|
| `minimax_h3_image_8gb_vram.json` | 8 GB cards | 16 MP | 4 MP | Tiled (512) | 10 slots, 4 or fewer is fastest |
| `minimax_h3_image_12gb_vram.json` | 12 GB cards | 16 MP | 4 MP | Tiled (768) | 10 slots, 4 or fewer is fastest |
| `minimax_h3_image_16gb_vram.json` | 16 GB+ cards | 16 MP | 4 MP | Normal | 10 slots |

### What you need for an 8 GB card

- **GPU:** NVIDIA with 8 GB VRAM (tested: ComfyUI used at most 3.7 GB VRAM for a 16 MP image).
- **System RAM:** 24 GB or more (tested with 24 GB), 32 GB is safer. The models are bigger than the card, so ComfyUI keeps them in RAM and streams them in parts.
- **Disk:** about 38 GB free, on an SSD.
- **ComfyUI:** a recent version (MiniMax H3 nodes are built in, no custom nodes).
- **3 model files**, the same full-quality files as the 16 GB version, **no GGUF**: the image model (21 GB), the main text encoder `nvfp4` (15.7 GB) and the VAE (1.2 GB). Links are in [Model files](#model-files).
- **Not needed:** the prompt upsampler `qwen3.5_9b_w4a8_gptq.safetensors`. It is switched off in the 8 GB workflow because it makes images very slow on 8 GB.

### Measured render times

Tested on an RTX 5070 Ti (16 GB) with 24 GB of system RAM. For 12 GB and 8 GB, part of the card's memory was filled by a dummy program, so ComfyUI only had the VRAM a real 12 GB or 8 GB card has free. With the same seed, all three gave **pixel-for-pixel the same image**.

| Job | 16 GB | 12 GB | 8 GB |
|---|---|---|---|
| Text-to-image, 16 MP | 66 s | 63 s | 60 s |
| Time per step (6 steps) | 4.4 s | 4.8 s | 5.7 s |
| 4 reference pictures, 16 MP | - | - | 228 s (about 4 min) |
| 4 reference pictures, 8 MP | - | - | 156 s (about 2.5 min) |

Real 8 GB and 12 GB cards (RTX 4060, 3060, 4070 ...) have slower chips than the test card, so expect longer times. The memory limit itself is not a problem.

## Example images

Made with this workflow, straight out of ComfyUI with no editing or upscaling. Click an image to open the full-size original PNG (5472x3072, 16 MP). All originals are in the [`examples`](https://huggingface.co/vyrekai/minimax-h3-image-comfyui-workflows/tree/main/examples) folder.

| | |
|---|---|
| [![Tortoise island](examples/previews/01_tortoise_island.jpg)](examples/01_tortoise_island.png) | [![Teacup tsunami](examples/previews/02_teacup_tsunami.jpg)](examples/02_teacup_tsunami.png) |
| [![Book canyon](examples/previews/03_book_canyon.jpg)](examples/03_book_canyon.png) | [![Moon hatching](examples/previews/04_moon_hatching.jpg)](examples/04_moon_hatching.png) |
| [![Lightbulb orchard](examples/previews/05_lightbulb_orchard.jpg)](examples/05_lightbulb_orchard.png) | [![Mirror desert](examples/previews/06_mirror_desert.jpg)](examples/06_mirror_desert.png) |
| [![Octopus pizza](examples/previews/07_octopus_pizza.jpg)](examples/07_octopus_pizza.png) | [![Bathroom mirror](examples/previews/08_bathroom_mirror.jpg)](examples/08_bathroom_mirror.png) |

The octopus pizza is 3840x2176 (8 MP); all others are 5472x3072 (16 MP).

## Prompt guide: let any AI write your prompts

[`MINIMAX_H3_IMAGE_PROMPT_GUIDE.md`](MINIMAX_H3_IMAGE_PROMPT_GUIDE.md) teaches an AI chatbot how to write prompts for this model. Drop the file into ChatGPT, Claude, Gemini or a local model, then ask for what you want, for example:

- *"Using this guide, write me a prompt for a rainy Tokyo street at night with a woman holding a red umbrella."*
- *"Using this guide, write an edit prompt: Picture 1 is my living room, make it sunset and add a cat on the sofa."*
- *"Using this guide, put the man from Picture 1 into the café from Picture 2."*

It answers with a finished prompt plus the settings to use (aspect ratio, draft or full size). Paste the prompt into the **PROMPT** box and leave *Enhance prompt* off, because the prompt is already long.

The guide covers text-to-image, single-picture edits and multi-reference scenes, plus realistic skin, hands, body direction, text in images, and fixes for common problems.

## What is in the workflow

- **Prompt box** with an optional **Enhance prompt** switch: a Qwen3.5 9B upsampler rewrites a short idea into a long, detailed caption (text-to-image only).
- **Aspect ratio picker** (1:1, 3:4, 4:3, 2:3, 3:2, 9:16, 16:9, 21:9) and a **full / draft switch**. Use the draft size to test prompts quickly.
- **Ten reference picture slots** for editing and multi-image prompts. They are all bypassed by default: select a slot, press **Ctrl+B** to switch it on, and refer to the pictures as `<Picture 1>`, `<Picture 2>`, ... in the prompt.
- **Sampler:** SamplerLCM (s_noise 1.1 -> 0.9) + BasicScheduler simple, **6 steps**, CFG **1.0**, no negative prompt. The model is distilled for 6 steps, so more steps do not help.
- A **How to use** note inside each workflow with the exact sizes for every aspect ratio.

## Model files

All four come from the official Comfy-Org repo: [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image). Click a file name to download it, then put it in the folder shown inside your ComfyUI `models` folder.

| Role | File (direct download) | Size | Folder |
|---|---|---|---|
| Diffusion model | [`minimax_h3_image_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors) | 21 GB | `models/diffusion_models/` |
| Text encoder | [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | 15.7 GB | `models/text_encoders/` |
| Prompt upsampler (optional, only for *Enhance prompt*. **8 GB: skip it**, it is switched off in the 8 GB workflow because it makes images very slow) | [`qwen3.5_9b_w4a8_gptq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors) | 7.3 GB | `models/text_encoders/` |
| VAE | [`minimax_h3_image_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors) | 1.2 GB | `models/vae/` |

About 38 GB in total, or 45 GB with the optional upsampler.

**No custom nodes needed.** Every node in these workflows is built into ComfyUI. Update ComfyUI to the latest version (tested on 0.45.3); older versions may not have the MiniMax H3 Image nodes.

## Low VRAM tips

- The model files are bigger than the card (about 21 GB diffusion + 16 GB text encoder). ComfyUI reads them in parts, so they do not need to fit in VRAM. Tested with **24 GB system RAM**. A fast SSD helps the first load.
- The **4 MP draft** is handy for testing prompts. Switching to 16 MP with the same seed gives a **different picture**, not the same one bigger, so if you like a 4 MP image, keep it.
- **Reference pictures:** all 10 slots work on every card, but each picture makes the steps slower. On 8 GB, 4 pictures take about 4 minutes at 16 MP; with many pictures, set *Full megapixels* to 8 to save time.
- Close other apps that use the GPU (games, video in the browser, OBS preview).
- If the decode runs out of memory, lower `tile_size` on the tiled VAE Decode node to 384 or 256.
- Prompts: write long, photo-like descriptions (who, where, light, camera). Short prompts give a glossy, doll-like face.
