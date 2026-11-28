# taurusduan/Pika-H3-Flex-Turbo-LoRA

Источник: https://huggingface.co/taurusduan/Pika-H3-Flex-Turbo-LoRA · скачано 2026-11-28
Загрузок: 1338 · лайков: 0 · базовая: ["Pika-Labs/Pika-H3-Flex","lightx2v/Minimax-h3-Turbo","MiniMaxAI/MiniMax-H3","WarmBloodAban/Minimax-h3_Singularity","Iwannapose/minimax_h3_pdmd_2nfe_comfyui"] · теги: turbo,lora,comfyui,base_model:Iwannapose/minimax_h3_pdmd_2nfe_comfyui,base_model:adapter:Iwannapose/minimax_h3_pdmd_2nfe_comfyui · обновлено: 2026-11-04T15:04:27.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
- Pika-Labs/Pika-H3-Flex
- lightx2v/Minimax-h3-Turbo
- MiniMaxAI/MiniMax-H3
- WarmBloodAban/Minimax-h3_Singularity
- Iwannapose/minimax_h3_pdmd_2nfe_comfyui
tags:
- minimax-h3
- pika
- pika-h3-flex
- turbo
- 4-step
- 8-step
- 2-step
- lora
- comfyui
- diffusers
- video-generation
- image-to-video
- reference-to-video
- audio-video-generation
- ml-intern
---

[![Built with ML Intern](https://huggingface.co/datasets/huggingface/badges/resolve/main/built-with-ml-intern-sm.svg)](https://hf.co/chat?mode=ml-intern)

# Pika H3 Flex Turbo LoRA — Pika's motion, one file, at every speed: 2-step draft → 4-step fast → 8-step max quality

[Pika H3 Flex](https://huggingface.co/Pika-H3-Flex) fused with the best few-step distills for MiniMax-H3 into **single standard LoRAs** (exact rank-concatenation — no custom nodes, no model patches). Drop one file onto any stock H3 setup — ComfyUI or Diffusers, bf16 / int8 / GGUF — or onto quality merges like [Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity), where it gets even better. Video **with synchronized audio**.

## ⭐ The combo: 4-step Turbo LoRA on Singularity vs Pika official on stock H3

<video src="https://huggingface.co/akhaliq/Pika-H3-Flex-Turbo-LoRA/resolve/main/showcase_5way_singularity_4step.mp4" controls muted loop style="max-width: 100%;"></video>

**Left: Pika official Flex adapter on stock H3, 6 steps. Right: this 4-step LoRA on [Singularity v2](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity), 4 steps.** Five prompts, same seed, 848×480 with stereo audio. Pika's motion + Singularity's quality tuning + turbo speed — in our tests the right side is both **faster and better-looking** than official. Uses the **Ref2VA** 4-step file (Singularity ships ref2va).

## ⚡ 2-step draft mode on Singularity — 5 fresh head-to-heads

<video src="https://huggingface.co/akhaliq/Pika-H3-Flex-Turbo-LoRA/resolve/main/showcase_5way_2step_vs_official.mp4" controls muted loop style="max-width: 100%;"></video>

**Left: official 6-step. Right: the 2-step draft LoRA on Singularity** — ballet leap, F1 cornering, diner kitchen, lighthouse storm, reef turtle. Flex × [pDMD 2-NFE](https://huggingface.co/Iwannapose/minimax_h3_pdmd_2nfe_comfyui), ~14 s of sampling per 5-s clip (2 evaluations at ~7 s/step on H200). First test batch ([5 more prompts](https://huggingface.co/akhaliq/Pika-H3-Flex-Turbo-LoRA/resolve/main/singularity_2step_test.mp4)): 4 of 5 held up well. **Use 2-step to iterate framing and motion, then render finals at 4 or 8 steps** — occasional scene failures happen, it's labeled draft for a reason.

## Speed: Pika official (6 steps) vs this LoRA (4 steps) on stock H3

<video src="https://huggingface.co/akhaliq/Pika-H3-Flex-Turbo-LoRA/resolve/main/showcase_5way_ours_vs_pika.mp4" controls muted loop style="max-width: 100%;"></video>

Same five prompts, stock int8 H3, seed 42: official 6-step (left) vs this LoRA 4-step (right).

## Max quality on stock: official (6 steps) vs 8-step HD

<video src="https://huggingface.co/akhaliq/Pika-H3-Flex-Turbo-LoRA/resolve/main/showcase_5way_8step_hd_vs_pika.mp4" controls muted loop style="max-width: 100%;"></video>

The 8-step HD variant (style 0.6) for maximum detail on stock H3. Also transfers to Singularity: [8-step HD on Singularity vs official](https://huggingface.co/akhaliq/Pika-H3-Flex-Turbo-LoRA/resolve/main/showcase_5way_singularity_vs_pika.mp4).

## High-action stress test

<video src="https://huggingface.co/akhaliq/Pika-H3-Flex-Turbo-LoRA/resolve/main/action_headtohead.mp4" controls muted loop style="max-width: 100%;"></video>

Martial arts sparring + skateboard hill-bomb: official 6-step (left) vs 4-step turbo (right). Few-step models are weakest on fast motion — judge it here.

## Benchmarks (measured — same base, seed, prompt, CFG 1, 848×480 × 121 frames + stereo audio)

| | Steps | End-to-end per 5-s clip | vs official |
|---|---:|---:|---:|
| **H200** — Pika official adapter | 6 | **60 s** | 1× |
| **H200** — this LoRA, 4-step | 4 | **40 s** | **1.5× faster** |
| **H200** — this LoRA, 2-step draft | 2 | **~25 s** (14 s sampling) | **~2.4× faster** |
| **H200** — this LoRA, 8-step HD | 8 | **70 s** | 0.86× — quality-max |
| **A100** — Pika official adapter | 6 | **109.3 s** | 1× |
| **A100** — this LoRA, 4-step | 4 | **66.0 s** | **1.66× faster** |
| Base MiniMax-H3 (~40 steps) | ~40 | ~500 s | **~8–12× slower** |

Same per-step cost on identical hardware (~7 s/step H200, ~12.5 s/step A100) — the speedup comes entirely from needing fewer evaluations.

**Quality:** every file is a mathematically exact composition of its source adapters (per-layer relative error ≈ 1–3e-7 — it *is* the stacked adapters, as one file), so fidelity to the stacked reference is exact by construction; Flex style is baked at 0.5 (2/4-step) / 0.6 (8-step HD), the community-validated operating points for few-step H3. Judge the look in the side-by-sides above.

**Lite variants:** rank-32 SVD compression (exact QR-product truncation) retains **96.1% of the fused spectral energy** at **0.31 GB** — 12.6× smaller.

## The lineup

| File | Transformer | Steps | Size | Pick this for |
|---|---|---:|---:|---|
| `pika_h3_flex_turbo_2step_draft_ref2va.safetensors` | Ref2VA | **2** | 3.9 GB | **Draft/iterate** — pDMD fusion, check framing & motion fast |
| `pika_h3_flex_turbo_4step_fl2va.safetensors` | FL2VA | **4** | 3.9 GB | **Speed** — fastest production Pika Flex |
| `pika_h3_flex_turbo_4step_ref2va.safetensors` | Ref2VA | **4** | 3.9 GB | Speed + refs · **the Singularity combo file** |
| `pika_h3_flex_turbo_8step_hd_fl2va.safetensors` | FL2VA | **8** | 3.9 GB | **Quality** — max detail, style 0.6 |
| `pika_h3_flex_turbo_8step_hd_ref2va.safetensors` | Ref2VA | **8** | 3.9 GB | Quality, with multi-reference identity |
| `pika_h3_flex_turbo_4step_fl2va_lite_r32.safetensors` | FL2VA | **4** | **0.31 GB** | **Tiny** — 96% spectral energy |
| `pika_h3_flex_turbo_4step_ref2va_lite_r32.safetensors` | Ref2VA | **4** | **~0.3 GB** | Tiny, with multi-reference identity |

Unlike patch-based or custom-node releases, these are **plain LoRAs** — they compose with your existing checkpoints, other (gentle) LoRAs, upscalers and workflows, in every H3-compatible loader.

## Usage (ComfyUI)

1. Download the file for your transformer into `ComfyUI/models/loras` (Ref2VA files for Singularity / reference workflows, FL2VA for stock T2V/I2V).
2. Load your usual MiniMax-H3 workflow, add a **LoraLoaderModelOnly**, strength **1.0**.
3. Sample: **2, 4, or 8 steps** (match the file), **CFG 1.0**, `euler`/`lcm`/`res_multistep`, `simple` scheduler. Flow shift: video **6.0** (FL2VA) / **12.0** (Ref2VA profile), audio shift **3.0**.

## Tips

- The 4-step files are tuned to be run at exactly 4 steps; the 8-step HD files reward 6–8 steps; the 2-step file is draft-only — expect occasional scene failures and use it for iteration.
- Flex style is baked at 0.5 (2/4-step) / 0.6 (8-step) — the empirically optimal strength for stacking a style adapter on few-step-distilled H3 (community consensus range 0.4–0.8, lower end for fewer steps).
- The 4-step turbo half is lightx2v **v1.2 768p** for FL2VA (v1.3 has an acknowledged stacking regression); Ref2VA uses the field-proven ref2v v0.1; the 2-step half is **pDMD 2-NFE** (Apache 2.0).
- Audio degrades below ~6 steps (true of every H3 turbo) — worst at 2 steps. Render finals at 4+ if audio matters.
- Avoid stacking additional strong style/motion LoRAs at high strength on the 2/4-step files.

## Why the quality holds: exact fusion, not an approximate merge

Each file is an exact **rank-concatenation** of the two adapters, per layer:

```
B_fused @ A_fused  ==  s·(B_flex @ A_flex) + 1.0·(B_turbo @ A_turbo)
```

Functionally identical to loading both LoRAs stacked — by construction, with **zero merge interference**. Naive additive merges collide in the output-side B matrix ([arXiv 2604.16826](https://arxiv.org/abs/2604.16826)); concatenation cannot. Verified numerically per layer before upload (worst relative error ~2.6e-7 through a bf16 round-trip), then verified end-to-end on GPU (the benchmarks and demos above).

Sources: [Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-H3-Flex) (Pika / Mellis, Inc.), [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo), [pdmd2026/pdmd_2NFE_lora](https://huggingface.co/pdmd2026/pdmd_2NFE_lora) via [Iwannapose's ComfyUI conversion](https://huggingface.co/Iwannapose/minimax_h3_pdmd_2nfe_comfyui) (Apache 2.0). Base model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). Demonstrated on [WarmBloodAban/Minimax-h3_Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity). Method background: PTD ([arXiv 2610.11498](https://arxiv.org/abs/2610.11498)), WaDi ([arXiv 2603.08258](https://arxiv.org/abs/2603.08258)). Reproduction scripts: [`scripts/`](https://huggingface.co/akhaliq/Pika-H3-Flex-Turbo-LoRA/tree/main/scripts).

## License

**MiniMax H3 Community License Agreement** — Model Derivative of MiniMax H3 (Section III.2); the same license applies to your use and derivatives. See the [base model license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Credit: MiniMax (base), Pika / Mellis, Inc. (Flex adapters), LightX2V (turbo distillation), pDMD authors (2-step distillation).
