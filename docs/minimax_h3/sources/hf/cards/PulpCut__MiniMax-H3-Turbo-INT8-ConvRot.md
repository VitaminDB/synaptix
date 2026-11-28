# PulpCut/MiniMax-H3-Turbo-INT8-ConvRot

Источник: https://huggingface.co/PulpCut/MiniMax-H3-Turbo-INT8-ConvRot · скачано 2026-11-28
Загрузок: 3053 · лайков: 5 · базовая: MiniMaxAI/MiniMax-H3 · теги: turbo · обновлено: 2026-08-21T11:52:42.000Z

---

---
language: en
tags:
- video-generation
- minimax-h3
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
pretty_name: PulpCut MiniMax H3 Turbo INT8 ConvRot
---

# MiniMax H3 Turbo · pruned INT8 ConvRot

## What this repository is

An optimized MiniMax H3 FL2VA package centered on a diffusion transformer with
the lightx2v **turbo step-distillation merged into the weights**, quantized in
the same pruned **INT8 ConvRot** layout as the Comfy-Org release. The primary
transformer is a drop-in replacement for
`minimax_h3_fl2va_pruned_int8_convrot.safetensors` in any runtime that reads
the optimized INT8 layout — including
[H3ddle](https://github.com/AlexanderIstomin/h3ddle), the open-source native
macOS app it was built for.

The transformer is **not a standalone model**. It needs the rest of the optimized
package (Qwen3-VL-32B INT8 text encoder, video/audio VAEs, tokenizer) from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

## H3ddle full input-major checkpoint

`minimax_h3_fl2va_pruned_turbo_int8_convrot_input_major.safetensors` is the
H3ddle-optimized form of the primary transformer. All 200 quantized core
projections across 50 blocks are stored `[input, output]` instead of
`[output, input]`. No tensor is dequantized or numerically changed; scales,
ConvRot metadata, and all non-core tensors are copied byte-for-byte.

This layout is selected by a versioned marker and validated against every
projection shape before H3ddle runs it. It is **not** compatible with runtimes
that assume the original output-major Comfy layout. The original transformer
remains in this repository for those runtimes; current H3ddle managed installs
download only the full input-major checkpoint.

## Measured performance versus the regular layout

On a 32 GiB M1 Pro, using the same prompt, seed, 512x512 canvas, eight passes,
50 blocks, and output settings:

| comparison | regular / baseline | optimized path | resulting gain |
|---|---:|---:|---:|
| transformer layout, matched eight-pass run | 253.9 s output-major | 231.4 s input-major | **8.9% (1.10x)** |
| complete generation, matched layout A/B | 278.7 s | 256.5 s | **8.0% (1.09x)** |
| regular H3 INT8 vs 8-step Turbo INT8, published RTX 4070 mean | 272.97 s / 20 passes | 130.66 s / 8 passes | **52.1% (2.09x)** |
| regular H3 INT8 vs Turbo INT8 denoising work | 20 passes / 1,000 block evaluations | 8 passes / 400 block evaluations | **60% less work; 2.5x fewer passes** |

The generated images were identical. Actual gains vary with canvas size,
memory pressure, and Mac. The external wall-time row comes from a reproducible
[three-pattern RTX 4070 benchmark](https://github.com/sepiablue-ai/minimax-h3-turbo-lora-benchmark/tree/main/202608_larry_vs_lightx2v)
that used the regular `minimax_h3_fl2va_pruned_int8_convrot.safetensors` as its
272.97-second baseline. Its two published eight-step Turbo means were 130.30
and 131.02 seconds; their 130.66-second mean is shown above. That is external
NVIDIA/ComfyUI evidence, not a projected M1 time. The work-count row is exact,
and the Turbo checkpoint is intended to reach roughly the regular model's
20-pass fidelity in 6–8 passes.

## Why this merge was made and republished

Step-distilled turbo checkpoints reach roughly 20-pass visual quality in
6–8 denoising passes, which is the difference between usable and unusable
generation times on low-memory Apple-silicon machines. No hosted turbo
variant existed in the INT8 ConvRot format that memory-constrained
runtimes stream from disk, so we merged and requantized one. H3ddle's
managed model downloads also require a pinned, hash-verified hosted
artifact, which this repository provides.

## Known behavior and limitations

The pruned ComfyUI conversion of the turbo LoRA **drops all 51 AdaLN
adapter pairs** (the source targets AdaLN input dimension 2688, while
pruned "compact-curve" models use dimension 8), and its own metadata warns
that four-step distillation behaviour may therefore be degraded.

In our testing that gap did **not** produce a measurable prompt-adherence
penalty. Every prompt-following miss we observed at 256²–512² with 6–8
passes — wrong subject species, illustration-style output, text-like
artifacts — is reproduced by the **unmodified base package at matched
settings, seed, and canvas**, so those are properties of the base model at
low step counts rather than effects of the distillation. What the merge
does change is fidelity: detail, fur, and lighting improve substantially at
the same step count. We asked the turbo authors about a curve-compatible
variant in
[ModelTC/Minimax-H3-Turbo#7](https://github.com/ModelTC/Minimax-H3-Turbo/issues/7).

Recommended settings: 6–8 denoising passes, euler sampling, all 50 blocks.
A Beta(0.6, 0.6) sigma schedule is commonly paired with turbo checkpoints;
we measured no consistent difference against the released linear grid on
this package.

## How the merge/quantization is done (high level)

For each of the 200 quantized projections, the BF16 pruned base weight is
merged with `strength × B·A` (rank-64, strength 1.0, `ema_pruned`
variant), rotated by the grouped 256-wide Hadamard transform used by the
ConvRot runtime kernels, and requantized with symmetric per-row absmax
INT8 scales. Token-refiner adapters merge losslessly in BF16. All other
tensors are copied byte-identical from the official INT8 file. The
pipeline reproduces the official quantizer exactly: run at strength 0 it
regenerates the official file with all 3,046,400 scales identical and
1,682 of 19.27 billion int8 values differing (rounding ties).

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

H3ddle uses the input-major file as the FL2VA core of its **"MiniMax H3 ·
Turbo + Hybrid References"** managed model. The app verifies the SHA-256,
adds the compact reference overlay, reuses shared package files via hardlinks,
and defaults the model to 8 passes. Published by [PulpCut](https://huggingface.co/PulpCut),
whose editor family shares the local-first media generation stack that
H3ddle implements in the open.

## Safety and intended use

Intended for local, personal video generation. The merge changes speed
characteristics, not the base model's content behavior; all usage
restrictions of the MiniMax H3 Community License apply unchanged.

## File inventory

| File | Bytes | SHA-256 |
|---|---|---|
| `minimax_h3_fl2va_pruned_turbo_int8_convrot.safetensors` | 20,970,379,854 | `9ad5c98b533894c122050d32804a14f49fca8edc16c52564a281cdc5825ac934` |
| `minimax_h3_fl2va_pruned_turbo_int8_convrot_input_major.safetensors` | 20,970,380,012 | `1dfe28c517a937fb9876f0975f224fd6e7ecb8744219f89bb8ba954403e10dc3` |

## Reproducibility references

The conversion is a single dependency-free Python script,
[`Scripts/convert-turbo-package.py`](https://github.com/AlexanderIstomin/h3ddle/blob/main/Scripts/convert-turbo-package.py)
in the H3ddle repository, including the strength-0 self-check used to
validate the pipeline against the official file.

The full optimized layout is reproducible with
[`Scripts/repack-h3-input-major.py`](https://github.com/AlexanderIstomin/h3ddle/blob/main/Scripts/repack-h3-input-major.py).

## Contact

Open an issue in the [H3ddle repository](https://github.com/AlexanderIstomin/h3ddle/issues).
