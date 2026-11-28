# rockerBOO/minimax-h3-nvfp4-convrot

Источник: https://huggingface.co/rockerBOO/minimax-h3-nvfp4-convrot · скачано 2026-11-28
Загрузок: 34284 · лайков: 34 · базовая: MiniMaxAI/MiniMax-H3 · теги: quantized,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-21T17:17:16.000Z

---

---
license: other
license_name: minimax-h3-community
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
pipeline_tag: text-to-video
tags:
- minimax-h3
- nvfp4
- quantized
- text-to-video
- image-to-video
- video-generation
- audio-generation
- comfyui
- diffusion-single-file
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: quantized
---

# MiniMax H3 — NVFP4 / INT4 quants for ComfyUI

Quantized diffusion transformers for [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), a 33B
omni-modal video+audio generator, built from the ComfyUI repack at
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). Output is 768p / 24 fps /
4–15 s with synchronized 32 kHz stereo audio. (2K output requires the separate `H3-Regenerate-2K`
module, which is not part of this or Comfy-Org's release.)

These are **ComfyUI single-file checkpoints**, not diffusers models.

## Which file do I want?

Filenames follow `minimax_h3_<fl2va|ref2va>_<variant>.safetensors`.

- **`fl2va`** — first/last-frame mode. Zero images = text-to-video, one or two = frame-conditioned.
- **`ref2va`** — omni-reference mode (up to 9 images / 3 video clips / 3 audio clips).

Both get identical treatment; pick the one matching your workflow.

| Variant | Size | GPU | Status |
|---|---|---|---|
| **`pruned_nvfp4_convrot_int8`** | 20 GB | Blackwell only | ✅ **Recommended** — tested in ComfyUI |
| `pruned_nvfp4_fp8` | 20 GB | Blackwell only | Tested; benchmark variant — `attn.qkv_proj` as FP8 instead of INT8 ConvRot, for speed comparison |
| `pruned_nvfp4` | 20 GB | Blackwell only | Tested; slower than the above at the same size |
| `nvfp4` | 34 GB | Blackwell only | Unpruned base; only if you can't use the pruned path |
| `pruned_mixed_int4_int8_convrot_simple` | 20 GB | Any (see below) | ⚠️ Experimental, quality unevaluated |
| `pruned_int4_convrot_simple` | 17 GB | Any (see below) | ⚠️ Experimental, quality unevaluated |
| `int4_convrot_simple` | 25 GB | Any (see below) | ⚠️ Experimental, unpruned |

**The three INT4 variants exist for `fl2va` only.** All other variants exist for both.

### Tradeoffs at a glance

- **NVFP4 needs Blackwell** (SM ≥ 10.0/12.0 — RTX 50-series, B100/B200). No fallback on older cards.
  This applies to every `nvfp4` file, including the `convrot_int8` ones, which are NVFP4 everywhere
  except `attn.qkv_proj`.
- **Pruned vs. unpruned** — pruning replaces the AdaLN modulation network (~40% of parameters) with a
  precomputed lookup table. Comfy-Org reports no quality loss, and it's a 34 GB → 20 GB cut. The
  tradeoff is that it depends on ComfyUI's shape-based detection of the pruned architecture, so the
  unpruned files are the safer choice outside ComfyUI.
- **INT4 is the escape hatch for non-Blackwell GPUs**, at the cost of unvalidated quality. It needs a
  ComfyUI build with `comfy-kitchen`'s `TensorCoreConvRotW4A4Layout`. Note that `comfy-kitchen`'s
  W4A4 CUDA kernel only uses native INT4 tensor-core MMA on `major == 8` (Ampere/Ada); Blackwell
  always falls through to an INT8-activation path, so there is no speed argument for INT4 there.
- **The mixed INT4/INT8 file is the same 20 GB as `pruned_nvfp4`.** It is only worth choosing if you
  can't run NVFP4 — it trades away validated quality for nothing else.
- **Quality has not been measured anywhere in this repo.** The NVFP4 files are recommended because
  they use conservative, established formats and are confirmed working, not because of a benchmark.
  The INT4 files are confirmed only to load and generate end-to-end.

## Setup

Download the diffusion model plus the companion text encoder and VAE from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), into ComfyUI's
`models/diffusion_models`, `models/text_encoders`, and `models/vae`. Use the I2V / T2V / R2V
workflow templates from that repo.

The text encoder is Qwen3-VL 32B and is large in its own right; Comfy-Org ships an
`nvfp4_awq` version of it that pairs well with these files on Blackwell.

## Quantization method

Quantized with the `convert-to-quant` (`ctq`) CLI using a hand-built per-layer profile (no built-in
filter exists for this architecture): NVFP4 for the bulk MLP weights, FP8 for attention QKV and
AdaLN-modulation projections, and BF16 kept for the precision-sensitive layers — the first 2 and
last 3 transformer blocks, the attention output projection, and all input/output projections. Several
of those are stored as F32 in the original checkpoint, which was taken as a signal they're sensitive.

| Layers | Format |
|---|---|
| `blocks.{2..46}.mlp.fc1` / `mlp.fc2` — 90 layers | NVFP4 |
| `blocks.{2..46}.attn.qkv_proj`, `.adaln_proj.linear` — 90 layers | FP8, tensor-wise scale |
| `blocks.{0,1,47,48,49}.*`, `attn.out_proj`, `condition_proj`, `time_embedder.*`, `audio_patch_proj`, `video_patch_proj`, `final_layer.*`, `token_refiner.*` — 86 layers | BF16 (untouched) |

The full config is in this repo as `minimax_h3_layer_config.json`.

```
hf download Comfy-Org/MiniMax-H3 --include "diffusion_models/minimax_h3_fl2va_bf16.safetensors"

ctq -i minimax_h3_fl2va_bf16.safetensors \
    -o minimax_h3_fl2va_nvfp4.safetensors \
    --nvfp4 --custom-type fp8 \
    --layer-config minimax_h3_layer_config.json \
    --comfy_quant --save-quant-metadata
```

`--custom-type fp8` enables the mixed-format path that `--layer-config` drives; `--nvfp4` is the
default for anything the config doesn't match. Result: 985 tensors, 180 layers quantized (90 NVFP4 +
90 FP8), verified against the `_quantization_metadata` header.

## AdaLN pruning (`pruned_*`)

Comfy-Org found that H3's AdaLN modulation weights (13.1B of 33.1B parameters, 39%) depend only on
the diffusion timestep, not content — so across the fixed set of inference timesteps they collapse to
a small lookup table with "no loss in output quality"
([Comfy blog](https://blog.comfy.org/i/209313677/optimized-for-local-inference-in-comfyui)).

The pruned files here apply that to our NVFP4/FP8 checkpoint rather than to Comfy-Org's INT8 one, by
transplanting the pruned tensors directly:

- **Removed:** `time_embedder.*` and every `*.adaln_proj.linear.{weight,bias}` (`[96768, 2688]` per block)
- **Added,** copied verbatim from Comfy-Org's pruned release: `adaln_t_table` (`[1025, 8]`, F32) and
  the collapsed `*.adaln_proj.linear.{weight,bias}` (`[96768, 8]`, F16)
- Everything else is untouched

Result: 892 tensors, 20 GB. ComfyUI's `MiniMaxH3Model` detects the pruned architecture from
`adaln_t_table.shape[1]` → `time_embed_dim`.

> **Gotcha:** this only works because the file carries no `config` metadata block. `model_detection.py`
> applies `config` *after* shape-based detection, so a stale `time_embed_dim: 2688` inherited from the
> parent NVFP4 file silently clobbers the correct value and breaks loading.

## INT8 ConvRot `attn.qkv_proj` (`pruned_nvfp4_convrot_int8`)

ConvRot (group-wise Hadamard rotation) pre-rotates weights in fixed-size groups before INT8
quantization, spreading outlier energy so INT8's narrow range can hold it — closing most of the gap
to FP8 while running faster, since INT8 tensor-core throughput generally exceeds FP8's. Comfy-Org
uses it throughout their `*_pruned_int8_convrot` releases; this applies it just to `attn.qkv_proj`
on top of our pruned NVFP4 checkpoint. Same file size, faster compute.

The 45 `attn.qkv_proj` layers in blocks 2–46 are quantized **from the original BF16 weights**, not
from the FP8 tensors they replace — that would compound quantization error. Built as extract →
quantize → splice rather than a full re-quantization pass:

1. Extract the 45 `attn.qkv_proj.weight` tensors from the BF16 checkpoint into a small standalone
   file (no bias — `qkv_proj` has none).
2. Quantize just that file:
   ```
   ctq -i <extracted>.safetensors -o <convrot_out>.safetensors \
       --int8 --scaling-mode row --dynamic-convrot --convrot-group-size 256 \
       --comfy_quant --save-quant-metadata
   ```
   `--dynamic-convrot` picks each layer's largest compatible power-of-4 group size; all 45 tensors
   share `in_features=5376`, cleanly divisible by 256.
3. Splice the resulting `{weight,weight_scale,comfy_quant}` tensors into the pruned NVFP4 file.
   Tensor count is unchanged at 892; only those layers change format (`F8_E4M3` tensor-scale →
   `I8` row-scale + ConvRot metadata).

`adaln_proj.linear` deliberately gets no equivalent treatment — pruning has already collapsed it to a
`[*, 8]` F16 lookup table, so there's nothing left worth quantizing.

This motivated extending `ctq`'s `--layer-config` JSON to support ConvRot options
(`convrot`/`dynamic_convrot`/`convrot_group_size`), previously only reachable via flat CLI flags —
[upstream PR #54](https://github.com/silveroxides/convert_to_quant/pull/54). The extract/splice
pipeline above doesn't need it, but it's the right generalization.

## FP8 `attn.qkv_proj` (`pruned_nvfp4_fp8`) — speed comparison baseline

Same as `pruned_nvfp4_convrot_int8`, but `attn.qkv_proj` (blocks 2–46) is FP8 tensor-scale instead
of INT8 ConvRot. Exists to A/B inference speed between the two at identical size and quality tier —
not a quality claim.

Quantized fresh from the original BF16 weights (default `ctq` FP8 output, no format flags needed),
then spliced into a copy of `pruned_nvfp4_convrot_int8` in place of its ConvRot `attn.qkv_proj`
tensors. Tensor count unchanged at 892.

## ⚠️ Experimental: INT4 W4A4 ConvRot

**Quantized with `--simple` (RTN, no learned rounding). Quality has not been evaluated — only
confirmed to load and generate end-to-end in ComfyUI.** The NVFP4 variants remain the recommended
choice. These exist for people who want a smaller footprint on non-Blackwell GPUs, or who want to
help evaluate quality. `fl2va` only.

Based on [ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers
(arXiv:2512.03673)](https://arxiv.org/abs/2512.03673) — group-wise Hadamard rotation on both
activations and weights before INT4 quantization, enabling W4A4 without retraining. Requires a
ComfyUI build with `comfy-kitchen`'s `TensorCoreConvRotW4A4Layout`.

### `int4_convrot_simple` (25 GB, unpruned)

`mlp.fc1`/`fc2`, `attn.qkv_proj`, **and** `adaln_proj.linear` for blocks 2–46 — 180 layers, the same
scope as the base NVFP4/FP8 profile — quantized to INT4 W4A4 ConvRot instead, replacing that split
entirely. Same BF16 exclusion set as every other variant.

```
ctq -i minimax_h3_fl2va_bf16.safetensors \
    -o minimax_h3_fl2va_int4_convrot_simple.safetensors \
    --int4 --dynamic-convrot --convrot-group-size 64 \
    --layer-config minimax_h3_layer_config_int4_convrot.json \
    --comfy_quant --save-quant-metadata --simple
```

Config included as `minimax_h3_layer_config_int4_convrot.json`. The group-size floor of 64 (vs. 256
for the INT8 pass) only exists to make `adaln_proj.linear` (`in_features=2688`) ConvRot-compatible at
all; `qkv_proj`/`fc1` (5376) and `fc2` (14336) resolve to 256/1024 regardless, since
`--dynamic-convrot` always takes the largest compatible size above the floor.

> In hindsight, including `adaln_proj.linear` was wasted work — pruning discards it either way. Drop
> it from the config on any future run.

### `pruned_int4_convrot_simple` (17 GB)

The above with the same AdaLN lookup-table transplant used by every pruned variant here, sourced from
`minimax_h3_fl2va_pruned_nvfp4.safetensors`. 802 tensors. Every remaining quantized layer (135) stays
INT4 W4A4.

### `pruned_mixed_int4_int8_convrot_simple` (20 GB)

The above, with 48 layers promoted from INT4 to genuine INT8 ConvRot (`int8_tensorwise`, full-width,
not 4-bit packed) to recover quality.

Per the ConvRot paper, full W4A4 measurably degrades output versus BF16 (FID 10.07 → 12.32,
ImageReward 0.99 → 0.84 on FLUX) — not from outliers, which ConvRot handles, but from INT4's limited
representational capacity in smooth, low-frequency regions. The paper's fix is promoting ~20% of
layers to INT8, selected empirically by functional role rather than by outlier amplitude.

Rather than running an independent sweep, the selection was taken from
[Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot),
whose mixed file records its own list in `_quantization_metadata` (`int8_ratio: 0.2`, matching the
paper). The 48 overlapping tensors (41 `mlp.fc1`, 4 `mlp.fc2`, 3 `attn.qkv_proj`) were pulled from
Comfy-Org's `minimax_h3_fl2va_pruned_int8_convrot.safetensors` via targeted HTTP range requests
(~7 GB of a ~21 GB file) and spliced in, rather than requantized — they were already quantized fresh
from BF16 by the same method used elsewhere here.

Because that list is FLUX-derived rather than measured on H3, treat the promotion set as a starting
point, not a tuned result.

> **Gotcha:** a safetensors file can carry both per-tensor `comfy_quant` blobs *and* a global
> `__metadata__["_quantization_metadata"]["layers"]` blob describing the same layers. In
> ComfyUI/comfy-kitchen 0.2.26 the loader treats the global blob as authoritative. Four separate fixes
> patched the per-tensor bytes correctly and still failed identically, because the global blob still
> claimed the pre-promotion format. Always update both when hand-editing quantization metadata.

## License

[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
inherited from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (repackaged by
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)). A full copy is included in this
repo as `LICENSE`.

MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax.
All Rights Reserved.
