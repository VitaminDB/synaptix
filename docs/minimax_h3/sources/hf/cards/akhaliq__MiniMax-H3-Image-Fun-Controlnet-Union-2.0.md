# akhaliq/MiniMax-H3-Image-Fun-Controlnet-Union-2.0

Источник: https://huggingface.co/akhaliq/MiniMax-H3-Image-Fun-Controlnet-Union-2.0 · скачано 2026-11-28
Загрузок: 504 · лайков: 22 · базовая: ["MiniMaxAI/MiniMax-H3-Image","alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0"] · теги: controlnet,controlnet-union,base_model:alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0,base_model:merge:alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0 · обновлено: 2026-11-19T08:16:35.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
library_name: diffusers
base_model:
  - MiniMaxAI/MiniMax-H3-Image
  - alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0
base_model_relation: merge
tags:
- minimax-h3
- videox_fun
- controlnet
- controlnet-union
- text-to-image
- image-to-image
- ml-intern
---

<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 0 2em"><a href="https://hf.co/chat?mode=ml-intern"><img src="https://hf.co/datasets/huggingface/badges/resolve/main/built-with-ml-intern-lg.svg" alt="Built with ML Intern" style="margin:0"></a><a href="https://hf.co/chat?mode=ml-intern&prompt=Fine-tune%20the%20control%20branch%20in%20akhaliq%2FMiniMax-H3-Image-Fun-Controlnet-Union-2.0%20so%20it%20steers%20H3-Image" style="display:inline-flex;align-items:center;gap:5px;padding:4px 12px;border:1.5px solid #eee;border-radius:16px;background:#f9fbfc;color:#c2410c;font-weight:600;font-size:14px;text-decoration:none">Adapt the control branch<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M13.5 6H5a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-8.5m-10.5 6L21 3m0 0h-5m5 0v5"/></svg></a><a href="https://hf.co/chat?mode=ml-intern&prompt=Test%20the%20control%20branch%20from%20akhaliq%2FMiniMax-H3-Image-Fun-Controlnet-Union-2.0%20on%20the%20native%20(non-distilled)%20H3-Image%20DiT%20at%2040%20steps" style="display:inline-flex;align-items:center;gap:5px;padding:4px 12px;border:1.5px solid #eee;border-radius:16px;background:#f9fbfc;color:#c2410c;font-weight:600;font-size:14px;text-decoration:none">Try the native DiT<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M13.5 6H5a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-8.5m-10.5 6L21 3m0 0h-5m5 0v5"/></svg></a></div>

# MiniMax-H3 + Fun ControlNet-Union 2.0 — video-model control, validated

**TL;DR: this repo is the ControlNet-Union-2.0 branch working with the MiniMax-H3 _video_ model** —
all 9 official demos reproduced, every condition validated, a ComfyUI workflow, and a
repeated-frame recipe that turns the video stack into a still-image controlnet (canny/gray/HED).

**What we tried that did NOT work:** driving the [MiniMax-H3-**Image**](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
model (the fast 6-step PDD-distilled image DiT) with this branch. The runtime wiring is verified
mathematically exact (scale-0 output is bit-identical to base), but the video-trained control
blocks collapse the spatial signal on the image model's distilled trajectory — it perturbs,
never steers (full evidence: [`assets/diag*`](assets), [`assets/image-dit-official/`](assets/image-dit-official)).

**Future work:** fine-tune the control branch on the H3-Image DiT (frozen DiT, zero-gated start —
all machinery in [`merge/`](merge)).

Contents: [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) packaged
with the [alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0)
control branch (8 union conditions + inpainting), plus verified, reproducible scripts and samples.

## Showcase: extreme poses → man in a suit

![pose showcase grid](assets/showcase2/pose_showcase_grid6.png)

Each row is one extreme pose; the task is the same pose as a **man in a tailored suit** in a new
setting. Columns, left to right: the text-to-video **source pose** (woman in athletic wear); its
**canny map**; the **H3 video model with the source image as a keyframe reference** (no
ControlNet); the **ControlNet-Union-2.0 video stack** (22-frame repeated canny, 40 steps,
`guidance_scale=1.0`, seed 904231, control scale 1.0 on row 1, 0.6 on rows 2–4); and **H3-Image
with the source image as a `<Picture 1>` reference**. With a reference image both baselines
*approximate* the poses — H3-Image's multi-reference editing is genuinely strong — but only the
ControlNet column matches the exact geometry every row (compare row 2's one-arm handstand
arm/leg angles). Identity and clothing follow the prompt at control scale 0.6. Variants
(0.6/0.8/1.0, animated control) and per-column frames:
[`assets/showcase2/variants/`](assets/showcase2/variants), [`assets/showcase2/`](assets/showcase2).

## Skeleton (pose) control on stills: partial, honestly

![skeleton showcase grid](assets/showcase3/skeleton_showcase_grid.png)

Same task (single-person poses → man in a suit, generic prompt, same seed). Columns: dataset
person, **pose skeleton input**, **H3 video + keyframe reference** (no ControlNet),
**ControlNet-Union-2.0** (22-frame repeated skeleton, scale 1.0), **H3-Image + `<Picture 1>`**
reference. Read it row by row: static skeletons steer only **coarse posture classes** — the
seated and kneeling rows produce sitting men while every baseline stands — but the lunge rows
are ignored, and on those rows H3-Image's reference editing actually lands the crouch better
than the sparse skeleton. The video keyframe path mostly ignores the reference or copies its
identity instead. For dense, reliable still-image control, **canny remains the working
condition** (grid above); real pose control wants motion (see the official dancer demo below)
or the v1-based P-H3 pack. Full-res cells: [`assets/showcase3/`](assets/showcase3).

## Official card demos, reproduced

All **9 official demos** from the upstream card, reproduced end-to-end: the official control
videos ([`alibaba-pai/.../asset/`](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/tree/main/asset))
at exact card settings — 40 steps, `guidance_scale=1.0`, `control_context_scale=1.0`, seed 43,
704×1280 budget, 24 fps — one H200, one model load
([`merge/card_repro_job.py`](merge/card_repro_job.py)). The upstream card publishes no prompts, so
ours are scene-matched. Mid-frames shown; compare with the official result videos on the
[upstream card](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0).

| Condition | Control | Reproduced frame |
|---|---|---|
| Canny | astronaut edges | ![g](assets/card-repro/gen_canny_mid.png) |
| Depth | dog + cat depth | ![g](assets/card-repro/gen_depth_mid.png) |
| HED | turtle edges | ![g](assets/card-repro/gen_hed_mid.png) |
| MLSD | architectural lines | ![g](assets/card-repro/gen_mlsd_mid.png) |
| Pose | dancer skeleton | ![g](assets/card-repro/gen_pose_mid.png) |
| Scribble | 3-student sketch | ![g](assets/card-repro/gen_scribble_mid.png) |
| Layout | colored boxes | ![g](assets/card-repro/gen_layout_mid.png) |
| Gray | coral luminance | ![g](assets/card-repro/gen_gray_mid.png) |
| Inpaint | source + mask | ![g](assets/card-repro/gen_inpaint_mid.png) |

First frames, extracted control maps and every sample: [`assets/card-repro/`](assets/card-repro).
More samples (same-seed on/off proof, static-frame coverage, diagnostics): [`assets/`](assets).

## Reproduce / use

- **All 9 demos:** run [`merge/card_repro_job.py`](merge/card_repro_job.py) on an H200 (~1 h
  incl. downloads).
- **Your own control:** [`merge/expC_video_pipeline.py`](merge/expC_video_pipeline.py) — repeat a
  control image into a 22-frame video and run the VideoX-Fun control pipeline; any output frame
  is a controlled image (~8 min on an H200).
- **All 8 conditions from one photo:** [`merge/coverage_job.py`](merge/coverage_job.py).

## ComfyUI

ComfyUI **≥ 0.35** ships the core **Apply MiniMax H3 Fun ControlNet** node — no custom nodes
needed. Drag [`comfyui/workflows/minimax-h3-fun-controlnet-union-2.0.json`](comfyui/workflows/minimax-h3-fun-controlnet-union-2.0.json)
onto the canvas (the official template, adapted to load the **2.0** union patch).

| File | Goes in |
|---|---|
| [`minimax_h3_ref2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) (19.5 GB) | `models/diffusion_models/` |
| [`minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors) (4.5 GB) — **the 2.0 patch** ([bf16](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors) also available) | `models/model_patches/` |
| [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (14.6 GB) | `models/text_encoders/` |
| [`minimax_h3_video_vae_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) + [`minimax_h3_audio_vae_fp32.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `models/vae/` |

Usage notes:

- **Weights caveat:** ComfyUI's pruned bases need the *curve-form* patches from Comfy-Org (table
  above). The checkpoint in this repo's `controlnet/` is the original full-width AdaLN release —
  use it with the VideoX-Fun pipeline (`merge/` scripts), not with ComfyUI's pruned models.
- `strength` = `control_context_scale` (1.0 strongest, 0 off); keep guidance at 1.0. Control is
  windowable via `start_percent`/`end_percent`.
- Frames snap to the **17n+5** grid at 24 fps. For single-image control, repeat the control
  image into a 22-frame clip — static frames steer for **canny/gray/HED** (see coverage below);
  the other conditions want real motion.
- Preprocessors are built in: Canny (Detect Edges), Depth (Depth Anything 3 / MoGe), Pose
  (SDPose, wired in the template). Inpainting: connect `mask` (+ optional `source_video`).

## What's in this repo

```
controlnet/                 ControlNet-Union-2.0 branch (13.6 GB) + config + license
transformer_diffusers/      H3-Image DiT (PDD, 6-step), unmodified
vae_diffusers/, tokenizer/, processor/, scheduler/, modular_model_index.json
merge/                      wiring + reproduction + coverage scripts
comfyui/workflows/          ComfyUI template adapted for the 2.0 union patch
assets/                     samples: card-repro, validation, diagnostics
```

## Official controls on the H3-Image DiT (negative result)

The same 8 official control frames through the **image** DiT (runtime wiring,
[`merge/validate_controlnet.py`](merge/validate_controlnet.py)), scene-matched prompts, seed 43:
outputs are excellent — but driven by the **prompt**, not the control. Base (no control) vs
controlled at the same seed:

| base_canny | ctrl_canny |
|---|---|
| ![b](assets/image-dit-official/base_canny.png) | ![c](assets/image-dit-official/ctrl_canny.png) |

PSNR(base, ctrl) per condition — large changes, none structurally aligned with the control
(consistent with the hint-collapse diagnostics in [`assets/diag2/`](assets/diag2)):

| canny | depth | hed | mlsd | pose | scribble | layout | gray |
|---|---|---|---|---|---|---|---|
| 10.3 | 14.3 | 13.0 | 11.1 | 19.7 | 11.0 | 13.0 | 14.5 dB |

All 24 images (controls, base, controlled): [`assets/image-dit-official/`](assets/image-dit-official).

## Notes

- The ControlNet is a **runtime branch** of the MiniMax-H3 video DiT (10 blocks injecting
  zero-gated skips at layers 0,5,…,45) — it cannot be folded into the base weights, and the
  image model is **not** a ControlNet out of the box: on the H3-Image DiT (PDD-distilled, 6
  steps) the branch attaches but does not steer without adaptation fine-tuning. Evidence and
  diagnostics: [`assets/validation*`](assets), [`assets/diag*`](assets).
- For image use via the video pipeline, static repeated-frame controls transfer for **canny and
  gray (HED partially)**; the other conditions need real moving control videos.

## Sources & license

- Base: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image · ControlNet: https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0 · Code: https://github.com/aigc-apps/VideoX-Fun
- Both upstream models are under the MiniMax-H3 Community License Agreement (`LICENSE`,
  `LICENSE.controlnet`); the same terms apply to this unofficial community repackaging. Not
  endorsed by MiniMax or the VideoX-Fun authors.
