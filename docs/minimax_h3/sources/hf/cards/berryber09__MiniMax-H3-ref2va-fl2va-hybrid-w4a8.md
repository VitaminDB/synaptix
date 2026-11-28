# berryber09/MiniMax-H3-ref2va-fl2va-hybrid-w4a8

Источник: https://huggingface.co/berryber09/MiniMax-H3-ref2va-fl2va-hybrid-w4a8 · скачано 2026-11-28
Загрузок: 2604 · лайков: 14 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: comfyui,quantized · обновлено: 2026-10-10T18:39:21.000Z

---

---
license: other
license_name: minimax-h3
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3
base_model:
  - MiniMaxAI/MiniMax-H3
tags:
  - minimax-h3
  - text-to-video
  - image-to-video
  - comfyui
  - hybrid
  - w4a8
  - quantized
---

# MiniMax-H3 · fl2va × ref2va Hybrid (w4a8, b20-49)

A **4-bit (`w4a8_mixed`)** block-splice hybrid of MiniMax-H3 that combines the **first/last-frame keyframe
model (`fl2va`)** with the **reference-conditioning model (`ref2va`)** in a single checkpoint — so one gen can
be **driven by a first frame *and* use reference images at the same time**.

~**12.5 GB** on disk — 4-bit (`w4a8_mixed`); fits a 16 GB card with headroom.

## Why

- `fl2va` — first/last-frame keyframe conditioning, higher visual/audio quality. No reference slots.
- `ref2va` — image/video/audio reference conditioning (subject/scene/style). Lower base quality.

Neither alone can do **"open on this frame, and have this person walk in later."** The hybrid can:
the first frame stays a hard temporal anchor while the reference supplies a subject's identity that the
model can introduce anywhere in the clip.

## Recipe (`b20-49`)

Whole-block splice of the two official-derived w4a8 checkpoints:

| Part | Source |
|------|--------|
| Input/output heads + all non-block tensors | `fl2va` |
| Transformer blocks **0–19** | `fl2va` |
| Transformer blocks **20–49** | `ref2va` |

`b20-49` leans toward **reference adherence** (30 of 50 blocks from `ref2va`) at a small cost to base
quality — a good pick when the reference is mandatory. The quality↔reference spectrum, most-quality →
most-reference: `b30-49 → b25-49 → b20-49 → b15-49`.

Per-layer quant metadata is merged correctly: `fl2va`'s `_quantization_metadata` is kept only for the
fl2va-sourced layers; the `ref2va` blocks carry their own `comfy_quant` tensors. Verified to load in ComfyUI
(`Detected mixed precision quantization`).

## Usage (ComfyUI)

Place in `ComfyUI/models/diffusion_models/`. It routes as a **`ref2va`** model, so both paths are active:

- **first_frame / last_frame** reference → wired as a keyframe (`MiniMaxH3AddKeyframes`) — hard-anchors that frame.
- **subject / scene / style** reference → wired into `MiniMaxH3ReferenceToVideo` — identity/style conditioning, not frame-locked.

Both can be supplied in one generation.

## Credit

- Base model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Hybrid block-splice idea: [smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models) (int8 / bf16-sized). This repo adds the **w4a8** variant, which that repo does not provide.

Inherits the MiniMax-H3 license.
