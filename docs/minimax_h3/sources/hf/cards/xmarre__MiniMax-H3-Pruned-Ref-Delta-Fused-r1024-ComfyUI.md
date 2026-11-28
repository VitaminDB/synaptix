# xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI

Источник: https://huggingface.co/xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI · скачано 2026-11-28
Загрузок: 24115 · лайков: 47 · базовая: ["diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024","Comfy-Org/MiniMax-H3","MiniMaxAI/MiniMax-H3"] · теги: comfyui,base_model:Comfy-Org/MiniMax-H3,base_model:merge:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-28T15:20:58.000Z

---

---
library_name: diffusion-single-file
pipeline_tag: image-text-to-video
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model:
  - diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024
  - Comfy-Org/MiniMax-H3
  - MiniMaxAI/MiniMax-H3
tags:
  - comfyui
  - diffusion-single-file
  - safetensors
  - minimax-h3
  - ref2va
  - fl2va
  - merge
  - int8
  - convrot
  - synchronized-audio-video
  - image-text-to-video
  - experimental
---

# MiniMax-H3 Pruned Ref-Delta Fused r1024 — ComfyUI Single File

Native ComfyUI-format single-file conversion of
[`diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024`](https://huggingface.co/diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024).

This repository contains the **MiniMax-H3 diffusion transformer only**. It does not include the text encoder, tokenizer, VAE, or the rest of the MiniMax-H3 pipeline.

The BF16 checkpoint is a state-dict/layout conversion of the immediate source checkpoint. Four native-ComfyUI INT8 derivatives are provided: full core-Linear INT8 and INT8 ConvRot variants, plus `fc2`-BF16 compatibility variants retained for older/problematic ComfyUI execution paths. No training, fine-tuning, additional pruning, or learned-weight adaptation was performed.

## Available checkpoints

| File | Variant | Notes |
|---|---|---|
| `MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy.safetensors` | BF16 | Native ComfyUI conversion; 40,235,519,464 bytes (~37.472 GiB); 534 container tensors |
| `MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8.safetensors` | INT8 | All four heavy Linear weights in each of the 50 main transformer blocks quantized; 200 core Linear layers total; ~21.0 GB |
| `MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors` | INT8 ConvRot | Same 200-layer policy with ConvRot groupsize 256; ~21.0 GB; tested successfully on the author's current ComfyUI core |
| `MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-fc2bf16.safetensors` | INT8 / BF16 mixed | Compatibility variant; 150 core Linear layers quantized; all 50 MLP `fc2` weights remain BF16; 832 tensors |
| `MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot-fc2bf16.safetensors` | INT8 ConvRot / BF16 mixed | Compatibility variant using the same 150-layer policy with ConvRot groupsize 256; all 50 MLP `fc2` weights remain BF16; 832 tensors |

The ~21.0 GB sizes above are the rounded sizes reported by `hf`/Xet during upload.

## SHA-256 checksums

BF16 full-file SHA-256:

```text
78b88298e241231b3bd95d752abde711efc9dd6517669a8a934faeb70baf6a98
```

BF16 tensor-data-region SHA-256:

```text
d5593b33e9920d038241697e871955e6ad118dce08ff49b6bccddfb85e9c06a0
```

Full INT8 full-file SHA-256 values:

```text
MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8.safetensors
b1aa96041103165fe18fffa4c6e6f9d6a23a090527762cbf6e9f607caac54321

MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors
00be5b0f995cc5a628921790f69cb22e138776c1e12235e3eab521941bb4b8c2
```

Compatibility-variant full-file SHA-256 values:

```text
MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-fc2bf16.safetensors
5e9bb588f8b025414121b3036b0ea39b0e88483bd90c5de645ff87449cb5fcee

MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot-fc2bf16.safetensors
304e808416e714a91c348b7ba70ad5098d8b294783596d304c9eec602f1f609c
```

## INT8 quantization policy

All four quantized checkpoints use ComfyUI's native per-layer `.comfy_quant` format and `TensorWiseINT8Layout`. No custom quantized-model loader is required.

The following weights are quantized in every one of the 50 main transformer blocks in **all** INT8 variants:

```text
blocks.N.attn.qkv_proj.weight
blocks.N.attn.out_proj.weight
blocks.N.mlp.fc1.weight
```

That accounts for **150 quantized core Linear layers**.

The full INT8 variants additionally quantize:

```text
blocks.N.mlp.fc2.weight
```

That adds another 50 layers for **200 quantized core Linear layers total**.

The `*-fc2bf16.safetensors` compatibility variants keep those 50 `fc2` weights in BF16 and therefore retain the original 150-layer INT8 policy.

All smaller/sensitive tensors remain in their source precision, including the pruned AdaLN table and projections, final-layer projections, norms, patch/text projections, and token refiner.

Regular INT8 uses tensor-wise scaling:

```text
format: int8_tensorwise
per_channel: false
convrot: false
```

INT8 ConvRot uses:

```text
format: int8_tensorwise
per_channel: true
convrot: true
convrot_groupsize: 256
```

The two Diffusers-only auxiliaries `adaln_basis` and `adaln_mean`, which are retained in the repaired BF16 artifact but unused by native ComfyUI inference, are omitted from all quantized derivatives.

## `fc2` compatibility history

The first full 200-layer INT8 test also quantized `blocks.N.mlp.fc2.weight`.

In the ComfyUI environment used for the initial conversion work, large MiniMax-H3 sequences could fail when the fused `linear_input_act(..., "swiglu")` path sent quantized `fc2` through `comfy_kitchen.int8_linear`. The observed failure involved a very large contiguous INT8 scratch allocation during dynamic activation quantization and could result in OOM or a hard WSL failure.

The `fc2`-BF16 variants were created to avoid that path while retaining INT8 for the other three heavy Linear weights in every transformer block.

After later ComfyUI core updates, the full INT8 ConvRot checkpoint has been running successfully in the author's current environment, including the same type of large MiniMax-H3 workloads that previously exposed the failure. The exact upstream change responsible has not been isolated here, so the `fc2`-BF16 files remain published as compatibility fallbacks for older or otherwise affected ComfyUI installations.

## Which INT8 file should I use?

For an up-to-date ComfyUI installation, start with:

```text
MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors
```

Use the regular full-INT8 counterpart if ConvRot is not desired:

```text
MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8.safetensors
```

If full INT8 produces an OOM, hard process failure, or another failure specifically around quantized `fc2` execution, use the corresponding `fc2bf16` compatibility variant:

```text
MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-fc2bf16.safetensors
MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot-fc2bf16.safetensors
```

The full INT8 ConvRot file is the variant with confirmed current-runtime use by the author. The regular full-INT8 file uses the same 200-layer quantization policy without ConvRot; it has not received the same current end-to-end runtime coverage described below.

## Provenance

| Layer | Repository / revision |
|---|---|
| Original model | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| ComfyUI pruned-model lineage | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| Immediate source | [`diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024`](https://huggingface.co/diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024) |
| Immediate-source revision | `c7d26373ecb070f1f1dc8811609d56d10d45d690` |
| Native conversion / quantized packaging | `xmarre` |

See [`PROVENANCE.md`](PROVENANCE.md) for the BF16 conversion record.

## Pruned MiniMax-H3 timestep conditioning

This checkpoint uses MiniMax-H3's pruned AdaLN-curve representation.

The native model contains:

```text
adaln_t_table  [1025, 8]  F32
```

and intentionally does not contain the full-model timestep MLP:

```text
time_embedder.proj_in.*
time_embedder.proj_out.*
```

The Diffusers source stores the precomputed curve table as `time_embedder.table`; the native conversion restores it as `adaln_t_table`.

All 50 transformer-block AdaLN projections and the final-layer AdaLN projection consume the 8-wide curve basis.

## Critical folded-bias conversion

The pruned Diffusers checkpoint stores required constant AdaLN modulation terms as separate `folded_bias` tensors. They are converted to the native ComfyUI bias parameters:

```text
transformer_blocks.N.adaln_proj.folded_bias
    -> blocks.N.adaln_proj.linear.bias          N = 0..49

norm_out.folded_bias
    -> final_layer.adaln_proj.linear.bias
```

All 51 native constant biases are F32. The learned AdaLN weights preserve their source dtype, BF16 in this checkpoint.

## Other structural conversion steps

The BF16 conversion also:

- removes stale Diffusers `config` safetensors metadata;
- recombines 52 Diffusers Q/K/V groups into native `qkv_proj` tensors;
- restores the native SwiGLU half ordering for 52 feed-forward tensors;
- reconstructs the non-learned `rope.inv_freq` buffer;
- preserves learned-weight dtypes.

The repaired BF16 artifact retains `adaln_basis` and `adaln_mean`; native ComfyUI does not consume them. A clean fresh conversion, and all quantized derivatives, omit those auxiliaries.

## Validation

The BF16 conversion passed structural validation for the pruned H3 architecture, including:

```text
adaln_t_table [1025, 8] F32: OK
full timestep MLP tensors: absent
all 50 block AdaLN input widths = 8: OK
final-layer AdaLN input width = 8: OK
all 51 native AdaLN biases: present / F32
Diffusers folded_bias keys: absent
stale config metadata: absent
required native MiniMax-H3 keys: present
rope.inv_freq [16] F32: OK
safetensors.safe_open: OK
```

The two `fc2`-BF16 compatibility variants were validated through full native ComfyUI execution. The tested workflow exercised:

```text
model_type FLOW_AV detection
H3 Continuum main sampling pass 1
H3 Continuum continuation sampling pass 2
Spectrum H3 actual/forecast execution
large 3D latent refine pass
video VAE decode
audio VAE decode
final Continuum assembly
video combine
```

Both `fc2`-BF16 variants completed that tested end-to-end workflow without the failures seen in the earlier full-INT8 execution path.

Following later ComfyUI core updates, the **full INT8 ConvRot** variant has also been running successfully in the author's current environment. This confirms practical compatibility for the current tested setup and removes the original reason to withhold the 200-layer ConvRot file from release.

The regular full-INT8 file is published as the non-ConvRot equivalent of the same 200-layer quantization policy. The current validation statement above specifically covers the full INT8 ConvRot variant; it should not be read as a separate quantitative quality or stability benchmark for every file on every ComfyUI revision.

This is functional compatibility validation for the tested ComfyUI paths, not a quantitative image/video quality benchmark.

## ComfyUI usage

Place the selected checkpoint in:

```text
ComfyUI/models/diffusion_models/
```

Then select that exact file in a diffusion-model loader that delegates to ComfyUI's native diffusion-model loading path. The conversion was tested around `DiffusionModelLoaderKJ` from [`xmarre/ComfyUI-KJNodes`](https://github.com/xmarre/ComfyUI-KJNodes).

For the INT8 files:

```text
weight_dtype: default
compute_dtype: default or bf16
```

Do not force an additional FP8 weight cast on top of the native INT8 checkpoint.

Use the appropriate MiniMax-H3 text encoder and VAE separately.

## Included conversion / repair tools

- [`tools/convert_h3_diffusers_to_comfy_v3.py`](tools/convert_h3_diffusers_to_comfy_v3.py) — corrected Diffusers -> native ComfyUI converter.
- [`tools/repair_h3_pruned_comfy_header.py`](tools/repair_h3_pruned_comfy_header.py) — repairs the pruned timestep-table mapping and removes stale Diffusers `config` metadata from an earlier conversion.
- [`tools/repair_h3_pruned_folded_biases.py`](tools/repair_h3_pruned_folded_biases.py) — repairs the 51 pruned AdaLN folded-bias names in an earlier converted artifact without rewriting the tensor payload.

## Scope

The BF16 file is a native ComfyUI state-dict conversion and packaging of the immediate source checkpoint.

The four INT8 files are post-conversion native-ComfyUI quantized derivatives of that BF16 checkpoint. The two full variants quantize all four heavy core Linear weights per main transformer block; the two `fc2`-BF16 variants retain `fc2` in BF16 as a compatibility fallback. They do not add trained capabilities and are not fine-tunes or additional pruning passes.

## License and use restrictions

MiniMax-H3 and derivatives are governed by the **MiniMax H3 Community License Agreement**. A copy is provided in [`LICENSE`](LICENSE), with the distribution notice in [`NOTICE`](NOTICE).

The upstream license contains territorial, use, distribution, commercial, and acceptable-use restrictions. Review the complete license before using or distributing these weights. This repository does not grant rights beyond the upstream license.

## Attribution

- Original MiniMax-H3: MiniMax / Nanonoble Pte. Ltd.
- ComfyUI pruned-model lineage: Comfy-Org
- Immediate Diffusers source: diffusers-modular
- Native ComfyUI conversion and quantized packaging: xmarre, 2026

## Related repositories

- https://huggingface.co/MiniMaxAI/MiniMax-H3
- https://huggingface.co/Comfy-Org/MiniMax-H3
- https://huggingface.co/diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024
- https://github.com/xmarre/ComfyUI-KJNodes
