# MATLOWAI/minimax-h3-nvfp4

Источник: https://huggingface.co/MATLOWAI/minimax-h3-nvfp4 · скачано 2026-11-28
Загрузок: 5689 · лайков: 4 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: comfyui · обновлено: 2026-08-24T00:58:32.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
- MiniMaxAI/MiniMax-H3
tags:
- comfyui
- diffusion-single-file
- nvfp4
- blackwell
---

# MiniMax H3 FL2VA pruned, NVFP4 (Blackwell)

An NVFP4 quantisation of the MiniMax H3 pruned FL2VA diffusion model for
ComfyUI. Every one of the 200 block linears (qkv, out, fc1, fc2 in all 50
blocks) is stored as fp4 e2m1 with fp8 e4m3 block scales of 16 and one fp32
per-tensor scale; embeds, token refiner, norms and heads stay bf16. On
Blackwell (sm120) the GEMMs run natively on the fp4 tensor cores.

**Modification notice (required by the license): this repository contains a
modified version of MiniMax H3. The modification is post-training weight
quantisation of the block linears to NVFP4, performed 2026-08-19. Built from
`diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` in
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), revision
`3f57e8291d2ef846f9a074b1b76d2767db434abe`.**

## Should you use this file?

- **RTX 50-series / Blackwell + pytorch cu130+, current ComfyUI**: yes, if you
  want the speed. Measured against the W4A8 file on the same seed and graph:
  0.84x wall at 29k tokens, 0.90x at 45k, on an RTX PRO 6000. Same file size
  as W4A8 (12.5 GB, both 4.54 bits per parameter), so this is a speed play,
  not a memory play.
- **Any other GPU**: no. ComfyUI falls back to dequantised matmuls, which is
  slower than `int8_convrot`. Use Comfy-Org's `int8_convrot` file instead.

Runs on stock ComfyUI, no custom nodes required: the file carries per-layer
`comfy_quant` metadata, the same mechanism as the NVFP4 text encoder that
Comfy-Org already ships.

## How to run it

1. Blackwell GPU (RTX 50 series or RTX PRO Blackwell), pytorch built for CUDA
   13.0 or newer, and a current ComfyUI updated together with its
   comfy-kitchen dependency.
2. Put the `.safetensors` in `models/diffusion_models/minimax_h3/` next to the
   usual H3 stack from Comfy-Org (text encoder, video VAE, audio VAE).
3. Use any H3 workflow and point `UNETLoader` at this file, `weight_dtype`
   default. No custom nodes; the per-layer metadata does the rest.
4. Check the load log for `Native ops: nvfp4`. If a render comes out slower
   than the int8 file you are on the dequant fallback (wrong pytorch or a
   stale comfy-kitchen).
5. The lightx2v turbo LoRA stacks cleanly and keeps the speed: measured
   12-step turbo, 210 s vs 239 s on the W4A8 file (0.88x), same graph and
   seed.

## What it costs in quality

Median weight error is 9.4 percent relative rms vs bf16 (the W4A8 file
carries 7.3, int8_convrot 1.0). In practice a same-seed render is a clean
sibling take: same scene and words, slightly different delivery. Side-by-side
pages with clips, pixel, flow and audio rulers, and a synced A/B player:
https://matlowai.github.io/ComfyUI-MAINodes/a6-review/ (the gold cards are
this checkpoint). Note that an int8 control lands in the same
distance-from-reference band, so within this model family that distance
measures which take you got, not how good it is. Judge with your eyes on your
own content.

## The other regimes, and why only this one is published

Three other real NVFP4 regimes were built and rendered on the same scene, seed
and graph (a fourth, `except_out`, was built but never rendered). All three are
dominated by this file on every axis that can be trusted, so none of them is
published as a checkpoint:

| regime | what stays fp4 | size | wall | VRAM |
|---|---|---|---|---|
| **all (this file)** | qkv, out, fc1, fc2 in all 50 blocks | **12.5 GB** | **421.2 s** | **59.3 GB** |
| mid | the same four, blocks 2 to 47 only | 14.7 GB | 432.2 s | 65.2 GB |
| nofc2 | qkv, out, fc1 (fc2 stays bf16) | 18.0 GB | 450.8 s | 63.9 GB |
| fc1 | fc1 only | 29.1 GB | 525.6 s | 75.8 GB |
| int8 convrot (the A8W8 file) | nothing, int8 everywhere | 21.0 GB | 458.5 s | 82.5 GB |

Leaving layers in higher precision makes the file bigger, the render slower and
the peak VRAM higher, which is what you would expect. What it does **not** do is
give you a defensible quality win, because within this family we cannot measure
one. Every arm is a sibling take of the reference: same scene and words,
slightly different delivery. The control that settles it is the int8 checkpoint
in the last row, which carries about 1 percent weight error and should be the
quality ceiling of the whole quantised family. It has the best pixel PSNR
(14.76) and the best flow agreement (0.549) of any arm, and simultaneously the
**worst** audio correlation (0.147 against 0.329 to 0.528 for the fp4 arms) and
the worst mel error. A distance metric that ranks the most faithful checkpoint
last on audio is not measuring quality. Judge with your eyes and ears on your
own content.

## Measurements and per-layer censuses

Published here because they are hard to obtain and easy to carry, not because
they rank anything:

- `census/` plus the `.census.json` beside the weights: the per-layer NVFP4
  weight error (`rel_rms_nvfp4`, one row per quantised linear) for six builds, with
  the source file, build time and scale mode in each `meta` block. Includes
  `..._all_staticscale`, a calibrated-static-`input_scale` build that is a
  **negative result** and is not published as weights: static scales did not
  make streamed inference exact (streamed-vs-unstreamed rel-rms 0.392 static
  against 0.364 dynamic, so slightly worse), which located the remaining
  streaming difference outside the linears.
- `measurements/`: the full metric set behind the review site, 59 arms over
  three scenes (`bakery` 6, `painters` 19, `spider` 34), each with pixel, flow
  and audio rulers plus per-frame curves. Most arms are fake-quantised
  activations on the shipped W4A8 model, one region at a time; the rows tagged
  `real checkpoint` are native fp4 kernels. Rendered side by side with a synced
  head-to-head player at
  https://matlowai.github.io/ComfyUI-MAINodes/a6-review/

## Rebuild it yourself

The 30-second builder script (quantises the Comfy-Org bf16 file with comfy's
own TensorCoreNVFP4Layout) ships in
[ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes) as
`tools/build_nvfp4_checkpoint.py`. The `.census.json` beside the weights holds
the per-layer weight error of this exact build.

## License

MiniMax H3 Community License Agreement (see LICENSE and NOTICE in this
repository, and the license link above). The license carries territory
restrictions and other conditions; read it before using or redistributing.
Powered by MiniMax H3.
