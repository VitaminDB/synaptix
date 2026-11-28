# yichengup/minimax_h3-tyc-zs05-ltxqnorm-int8-convrot

Источник: https://huggingface.co/yichengup/minimax_h3-tyc-zs05-ltxqnorm-int8-convrot · скачано 2026-11-28
Загрузок: 2431 · лайков: 8 · базовая: MiniMaxAI/MiniMax-H3 · теги: comfyui · обновлено: 2026-11-27T08:40:04.000Z

---

---
base_model: MiniMaxAI/MiniMax-H3
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3
tags:
- minimax-h3
- comfyui
- int8
- image-to-video
- audio-video
pipeline_tag: image-to-video
---

**English** | [简体中文](README.zh.md)

# MiniMax H3 — int8 checkpoint for ComfyUI

A single-file ComfyUI checkpoint for **MiniMax H3**, quantised to int8.

It is a variant of the official pruned release, tuned and quantized by **Yicheng**; it has not been retrained.

## Specs

| | |
|---|---|
| Base | MiniMax H3 |
| Format | safetensors, single file |
| Quantisation | int8 (tensorwise + convrot) |
| Size | 20.97 GB |
| Loader | `Load Diffusion Model` (ComfyUI 0.32+) |

Text encoder and VAEs are the standard H3 ones — nothing custom is required.

## What it is good at

- **Fast action gets speed and punch.** Quick beats carry more explosive force than the stock
  checkpoint — dashes, strikes and snaps land with real weight instead of drifting across the
  frame.
- **Ordinary motion stays natural.** Slow and everyday movement — walking, turning, small
  gestures — keeps an unhurried pace and does not turn stiff or over-driven.
- **Small movement stays alive.** Hair, cloth, foliage and other fine elements keep moving
  instead of freezing between key poses.
- **Pushes motion hard — and that has a cost.** Because the movement is aggressive, fast action
  can show tearing, blending or drift, and objects near a moving subject may get dragged along.
  Short shots and moderate camera moves work best.

## Notes

- The weights carry modifications relative to the official release; `NOTICE` records that the
  files were changed.

## Licence

Derivative of MiniMax H3, distributed under the **MiniMax H3 Community License Agreement**
(see `LICENSE` and `NOTICE`). Not for generating likenesses of real people without consent,
not for illegal content, and not for any use the upstream licence prohibits.
