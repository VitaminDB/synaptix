# joeygambino/MiniMax-H3-x-Z-Image-GGUF

Источник: https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-GGUF · скачано 2026-11-28
Загрузок: 5188 · лайков: 41 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: gguf,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-24T08:33:06.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/Comfy-Org/MiniMax-H3
base_model:
- MiniMaxAI/MiniMax-H3
tags:
- gguf
- comfyui
- minimax-h3
- z-image
- video-generation
- audio-video
- merge
pipeline_tag: text-to-video
---

# MiniMax-H3 × Z-Image — the spatial detail graft

**Z-Image's eye for texture on MiniMax-H3's engine.** These are drop-in replacements for the
standard MiniMax-H3 checkpoints: same identity, same voices, same speed, same VRAM, same
workflows — but sets and surfaces render noticeably richer. Peeling paint peels harder, rust
bleeds further, water carries more light. Measured on chained scenes, the extra detail stays
**flat across joins** (no per-shot sharpening creep: 0.99 high-band ratio over 3 joins vs 1.11
on stock).

## What it looks like

*(demo videos — three-shot continuous takes rendered with the files on this page)*

<video src="https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-GGUF/resolve/main/demo/demo_greenhouse.mp4" controls width="640"></video>
<video src="https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-GGUF/resolve/main/demo/demo_engine.mp4" controls width="640"></video>
<video src="https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-GGUF/resolve/main/demo/demo_market_v2.mp4" controls width="640"></video>

## How it works

Z-Image (Lumina2, a 6B image model with exceptional texture rendering) and MiniMax-H3 both use
per-head Q normalisation in attention. This graft transplants the *magnitude profile* of
Z-Image's spatial attention — how sharply it commits to fine texture — onto H3's later blocks
by rescaling `q_norm` weights. No retraining, no new knowledge, no architecture change: H3
keeps everything it knows and attends to texture the way Z-Image does. Early blocks are left
untouched (grafting them produces a lattice artifact in regular textures — measured, not
guessed), K normalisation and feed-forward are never touched.

This is the second marriage in this line: [Joy-LTX 2.5](https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-comfy-native)
put JoyAI-Echo's performance on LTX-2.5's engine by weight-delta transplant. Here the donor
is a different architecture entirely, so what crosses is attention *statistics* rather than
weights - a mechanism of our own, block-gated and dose-controlled, verified against
same-seed baselines. A nod to TenStrip, whose H3 attention experiments sparked the
question of what an image model could donate.

## Which file

Same picking rules as standard H3 GGUFs. `curve` = the current recommended bakes.

| file | fits | notes |
|---|---|---|
| `*-curve-zs05-Q8_0.gguf` | 32 GB | closest to full precision |
| `*-curve-zs05-Q5_1.gguf` | 24–32 GB | the 24 GB pick |
| `*-curve-zs05-Q4_0.gguf` | 16–24 GB | the 16 GB pick |
| plain (non-curve) variants | as above | for workflows built on the original bakes |

**fl2va vs ref2va**: identical choice as stock H3 — ref2va when identity/voice must persist
(reference images, voice anchoring, identity bank), fl2va when a shot must land on a supplied
frame. Both chain.

## Install

Drop the file where your H3 checkpoints live, pick it in your loader. That's the whole
install — every H3 workflow works unchanged, including the
[MiniMax-H3 Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot) seamless-chain
canvases. comfy-native versions (bf16 / fp8 / int8 / w4a8 / nvfp4 and more):
see the companion repo.

## Verified

- Same-seed equivalence against the runtime-patch implementation (differences at run-noise level)
- Identity holds on faces across variants; facial texture *improves* over stock
- 4-shot chained scenes: detail flat across joins, scripted events land, no artifacts
- Every file: tensor-level verification of the graft math after baking

Made by joeygambino. Questions: open a discussion — I answer.
