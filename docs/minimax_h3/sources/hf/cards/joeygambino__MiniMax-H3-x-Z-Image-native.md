# joeygambino/MiniMax-H3-x-Z-Image-native

Источник: https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-native · скачано 2026-11-28
Загрузок: 22587 · лайков: 107 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: comfyui,comfy-native,comfy-quant · обновлено: 2026-08-24T08:28:31.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/Comfy-Org/MiniMax-H3
base_model:
- MiniMaxAI/MiniMax-H3
tags:
- comfyui
- comfy-native
- comfy-quant
- minimax-h3
- z-image
- video-generation
- audio-video
- merge
pipeline_tag: text-to-video
---

# MiniMax-H3 × Z-Image — spatial detail graft (comfy-native)

The comfy-native cuts of the [MiniMax-H3 × Z-Image graft](https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-GGUF):
Z-Image's spatial-attention profile on H3's engine — richer sets and textures, same identity,
no per-shot sharpening creep. Full story, demos and verification on the GGUF page.

Load with the plain **Load Diffusion Model** node, ComfyUI 0.32+. Files are the `pruned`
H3 builds with the graft baked in (`zs05` = late-block gains, dose 0.5):

- `bf16` — the master (ref2va)
- `comfy-fp8` / `fp8e5m2` — fp8 scaled
- `comfy-int8` / `int8_convrot` — the fast pick on RTX 50
- `comfy-w4a8` / `w4a4` / `nvfp4` — 4-bit family for 16 GB cards (`w4a8` is the quality pick; `nvfp4` is Blackwell-native, emulated elsewhere)
- `comfy-mxfp8` — 8-bit microscaling, Blackwell-specialized; little benefit on older architectures
- fl2va and ref2va variants where both exist; picking rules are identical to stock H3

RTX 30/40: the GGUF repo is 4–8× faster than any 4-bit comfy-native arm on Ampere.

Made by joeygambino. Questions: open a discussion — I answer.
