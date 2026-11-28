# PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot

Источник: https://huggingface.co/PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot · скачано 2026-11-28
Загрузок: 3587 · лайков: 13 · базовая: MiniMaxAI/MiniMax-H3 · теги: turbo · обновлено: 2026-08-21T11:52:39.000Z

---

---
language: en
tags:
- video-generation
- minimax-h3
- ref2va
- int8
- convrot
- input-major
- turbo
- h3ddle
- pulpcut
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/939557dc319dd91227e30195a763f272ba7f8765/LICENSE
base_model: MiniMaxAI/MiniMax-H3
pretty_name: PulpCut MiniMax H3 Ref2VA Turbo INT8 ConvRot
---

# MiniMax H3 Ref2VA Turbo · pruned INT8 ConvRot

## What this repository is

An optimized MiniMax H3 **Ref2VA** package centered on the omni-reference
diffusion transformer, which conditions generation on ordered reference
images. It has the lightx2v **turbo step-distillation merged into the weights**
and uses the same pruned **INT8 ConvRot** layout as the Comfy-Org release. The
primary transformer is a drop-in replacement for
`minimax_h3_ref2va_pruned_int8_convrot.safetensors` in any runtime that reads
the optimized INT8 layout — including
[H3ddle](https://github.com/AlexanderIstomin/h3ddle), the open-source native
macOS app it was built for.

The transformer is **not a standalone model**. It needs the rest of the optimized
package (Qwen3-VL-32B INT8 text encoder, video/audio VAEs, tokenizer) from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), and the
FL2VA transformer alongside it if you also want prompt-only and keyframe
generation.

## H3ddle full input-major checkpoint

`minimax_h3_ref2va_pruned_turbo_int8_convrot_input_major.safetensors` is the
H3ddle-optimized form of the primary transformer. All 200 quantized core
projections across 50 blocks are stored `[input, output]` instead of
`[output, input]`. No tensor is dequantized or numerically changed; scales,
ConvRot metadata, reference conditioning, and non-core tensors are preserved.

This layout is selected by a versioned marker and validated against every
projection shape before H3ddle runs it. It is **not** compatible with runtimes
that assume the original output-major Comfy layout. The original and full
input-major transformers remain in this repository for users and runtimes that
want exact Ref2VA weights.

## Measured performance versus the regular layout

On a 32 GiB M1 Pro, using the same prompt, reference, seed, 512x896 canvas,
eight passes, 50 blocks, and output settings:

| comparison | regular / baseline | optimized path | resulting gain |
|---|---:|---:|---:|
| transformer layout, matched eight-pass run | 541.2 s output-major | 515.3 s input-major | **4.8% (1.05x)** |
| complete generation, matched layout A/B | 631.4 s | 595.9 s | **5.6% (1.06x)** |
| regular Ref2VA INT8 vs Turbo Ref2VA INT8, matched quality run | 49.0 min / 20 passes | 20.2 min / 8 passes | **58.8% (2.43x)** |

The generated reference output remained good. Actual gains vary with canvas
size, memory pressure, and Mac.

## Compact H3ddle hybrid overlay

`minimax_h3_ref2va_pruned_int8_convrot_hybrid_adaln_25_49.safetensors` is a
43.55 MB optional overlay for H3ddle. It uses the complete FL2VA transformer as
the base and replaces only Ref2VA's compact AdaLN weight and bias in blocks
25–49. The overlay is 99.79% smaller than carrying a second 20.97 GB
transformer. Its bytes are identical whether extracted from the standard or
Turbo Ref2VA checkpoint, so the same file supports both FL2VA variants.

This is a storage and download optimization, not a denoising-speed
optimization. It intentionally produces a different sample than the full
Ref2VA transformer. A matched 512-square, eight-pass test on an M1 Pro found
good reference identity and image quality; transformer work remained the same
within run-to-run system variation. Starting with H3ddle 0.7.4, both managed
standard and Turbo + References packages download this overlay by default.
The full Turbo checkpoints above remain available here for manual use and
comparison, and the full standard checkpoint remains in Comfy-Org/MiniMax-H3.

## Why this merge was made and republished

Every published turbo LoRA for MiniMax H3 targets the **FL2VA** transformer.
Reference generation therefore had no fast path at all: on Apple silicon a
short reference clip costs roughly fifty minutes at the base model's twenty
denoising passes, with no distilled alternative to fall back on. Since the two
transformers are structurally identical — 932 tensors, identical names and
shapes — the FL2VA distillation can be merged into the Ref2VA weights, which
is what this file is.

## Does the distillation survive the transfer?

The adapter was distilled against FL2VA, and reference-identity transfer is
exactly the capability step distillation is most likely to damage, so this was
measured rather than assumed. Matched control on an M1 Pro — same prompt, same
reference image, same seed, 448², adapter as the only variable:

| comparison | regular / baseline | optimized path | resulting gain |
|---|---:|---:|---:|
| passes | 20 | 8 | 60% fewer |
| wall time | 49.0 min | 20.2 min | **58.8% (2.43x)** |
| identity vs reference (SSIM) | 0.426 | 0.476 | identity retained |

**2.4x faster with identity transfer intact.** The reference subject's
markings, build, and setting carry through; detail in fur and whiskers is
slightly sharper than the base at its full pass count. The SSIM difference is
modest and comes from a single prompt, so treat it as evidence that identity
transfer is *not degraded* rather than proof that it improves.

Recommended settings: 8 denoising passes, euler sampling, all 50 blocks.

## Known behavior and limitations

The pruned ComfyUI conversion of the turbo LoRA **drops all 51 AdaLN adapter
pairs** (the source targets AdaLN input dimension 2688, while pruned
"compact-curve" models use dimension 8), and its own metadata warns that
four-step distillation behaviour may therefore be degraded. We use 8 passes
rather than 4 for that reason. The same caveat applies to our FL2VA turbo
file; see
[ModelTC/Minimax-H3-Turbo#7](https://github.com/ModelTC/Minimax-H3-Turbo/issues/7).

## How the merge/quantization is done (high level)

For each of the 208 quantized projections, the BF16 pruned Ref2VA base weight
is merged with `strength × B·A` (rank-64, strength 1.0, `ema_pruned` variant),
rotated by the grouped 256-wide Hadamard transform used by the ConvRot runtime
kernels, and requantized with symmetric per-row absmax INT8 scales.
Token-refiner adapters merge losslessly in BF16. All other tensors are copied
byte-identical from the official INT8 file. The pipeline reproduces the
official quantizer exactly: run at strength 0 against the Ref2VA base it
regenerates the official file with all 3,046,400 scales identical and 1,711 of
19.27 billion int8 values differing (rounding ties).

## Source and attribution

- Original model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Pruned INT8 ConvRot base + shared package files: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- Turbo distillation LoRA: [ModelTC/Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo) (lightx2v team)
- Pruned ComfyUI LoRA conversion: [drbaph/MiniMax-H3-Turbo-Lora-ComfyUI](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI)

## Licensing

Derivative of MiniMax H3 weights; the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/939557dc319dd91227e30195a763f272ba7f8765/LICENSE)
applies. By downloading you agree to its terms.

## What these files are used for in H3ddle

H3ddle installs it in both managed reference-capable packages: the app verifies
the SHA-256 below and reuses the overlay and other shared package files it
already has via hardlinks. The Turbo package defaults to 8 passes; standard
keeps its regular generation profile. Published by
[PulpCut](https://huggingface.co/PulpCut), whose editor family shares the
local-first media generation stack that H3ddle implements in the open.

## Safety and intended use

Intended for local, personal video generation. The merge changes speed
characteristics, not the base model's content behavior; all usage restrictions
of the MiniMax H3 Community License apply unchanged.

## File inventory

| File | Bytes | SHA-256 |
|---|---|---|
| `minimax_h3_ref2va_pruned_turbo_int8_convrot.safetensors` | 20,970,379,854 | `e64cef63bc2785bcd72e6103c52aa78c6cd2c4f9870a7ce79675083fd65cf2e7` |
| `minimax_h3_ref2va_pruned_turbo_int8_convrot_input_major.safetensors` | 20,970,380,012 | `5ca6696fe1cd9a8f254594ac67ee541f151b2377735dea3557364bd868270463` |
| `minimax_h3_ref2va_pruned_int8_convrot_hybrid_adaln_25_49.safetensors` | 43,551,180 | `c3d80a9a2d17a30caf83e933262473cbf0b1ba7de4d29556646e9a92ab5f17aa` |

## Reproducibility references

The conversion is a single dependency-free Python script,
[`Scripts/convert-turbo-package.py`](https://github.com/AlexanderIstomin/h3ddle/blob/main/Scripts/convert-turbo-package.py)
in the H3ddle repository, including the strength-0 self-check used to validate
the pipeline against the official file.

The full optimized layout is reproducible with
[`Scripts/repack-h3-input-major.py`](https://github.com/AlexanderIstomin/h3ddle/blob/main/Scripts/repack-h3-input-major.py).

The compact overlay is reproducible with
[`Scripts/build-h3-hybrid-adaln.py`](https://github.com/AlexanderIstomin/h3ddle/blob/main/Scripts/build-h3-hybrid-adaln.py).

## Contact

Open an issue in the [H3ddle repository](https://github.com/AlexanderIstomin/h3ddle/issues).
