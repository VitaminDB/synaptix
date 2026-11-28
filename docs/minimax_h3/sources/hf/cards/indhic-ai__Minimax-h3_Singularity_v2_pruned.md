# indhic-ai/Minimax-h3_Singularity_v2_pruned

Источник: https://huggingface.co/indhic-ai/Minimax-h3_Singularity_v2_pruned · скачано 2026-11-28
Загрузок: 6536 · лайков: 3 · базовая: WarmBloodAban/Minimax-h3_Singularity · теги: comfyui,base_model:quantized:WarmBloodAban/Minimax-h3_Singularity · обновлено: 2026-10-22T05:44:33.000Z

---

---
license: apache-2.0
base_model: WarmBloodAban/Minimax-h3_Singularity
base_model_relation: quantized
pipeline_tag: image-to-video
tags:
- minimax-h3
- comfyui
- int8
- convrot
- pruned
- reference-to-video
---

# Minimax-h3 Singularity ref2va **v2** — pruned int8 (curve form)

A **pruned** build of [WarmBloodAban/Minimax-h3_Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity)
`Minimax-h3_Singularity_ref2va_v2_int8.safetensors`. That repo ships v2 only in full form (34.0 GB); this is the same
model in the pruned "curve" layout, **21.05 GB**, which loads natively in ComfyUI exactly like the official
`minimax_h3_ref2va_pruned_int8_convrot` and the author's own pruned v1.3.

All credit for the model goes to **WarmBloodAban** (fine-tune) and **MiniMax** (MiniMax-H3). This repo only changes the
storage layout of the timestep-modulation path. Licence: Apache-2.0, as the source.

| file | size | sha256 |
|---|---|---|
| `Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors` | 21.05 GB | `c75cba0d4a7587166303ac038befc0b4f4aaea16686315cccecc16f9ca0c1519` |
| source: `Minimax-h3_Singularity_ref2va_v2_int8.safetensors` | 34.00 GB | `db3ec568863db6f5ec1840210d41653afef73d4df7b0cb489c52ad458b6f8a7f` |

**Use:** put it in `models/diffusion_models/` and pick it in the UNET loader, in place of the full v2. ComfyUI detects the
curve form from the `adaln_t_table` key. Needs a ComfyUI recent enough to load pruned MiniMax-H3 checkpoints.

## What "pruned" means here

Only the timestep path changes; every attention / MLP weight is copied **byte for byte** (200 layers, int8 tensorwise
convrot, group size 256).

| | full v2 | this file |
|---|---|---|
| `time_embedder.*` | present | removed |
| each of 50 `blocks.N.adaln_proj.linear.weight` | `[96768, 2688]` int8 (~260 MB) | `[96768, 8]` fp32 (3 MB) |
| `final_layer.adaln_proj.linear.weight` | `[10752, 2688]` | `[10752, 8]` fp32 |
| `adaln_t_table` | — | `[1025, 8]` fp32 |

The timestep `t` is one number in [0, 1], so what the adaLN layers consume, `silu(time_embedder(t))`, traces a 1-D curve
in 2688-dim space. Sampled on a 1025-point grid, that curve lies **exactly** in an 8-dim subspace (SVD residual energy
0.0 at k = 8). So: table = curve coordinates in that basis, and each adaLN weight is pre-multiplied into the basis
(`W' = W·Vₖᵀ`, `b' = b + W·mean`). ComfyUI interpolates the table at runtime.

## Fidelity

Error of the adaLN output vs the full model's own weights, over the whole 1025-point t grid, relative to the
t-dependent part of the output:

| | blocks (50) | final layer |
|---|---|---|
| this file (v2) | 1.4e-4 – 1.7e-4 | 2.3e-4 |
| same method on v1.3 | 1.9e-4 – 2.0e-4 | 2.7e-4 |
| for reference: the author's pruned v1.3 vs their full v1.3 | 3.5e-2 – 2.0e-1 | 1.5e-2 |

The 8-wide weights are stored in **fp32** deliberately: bf16 storage raises the error to ~2.5e-2.

**Renders.** On a fast 4-step test of a 15 s action sequence (fixed seed), each model is perfectly deterministic. This
file, the full v2, and the full v2 with its adaLN stored bf16 instead of int8 all differ from one another by about the
same amount (17.5–18.1 dB PSNR): the sampler amplifies any change in adaLN numerics, including just int8 vs bf16
storage in the original. A genuinely different model (Singularity v1.3) sits at 12.9 dB. Judge quality by eye, as with
any re-quantisation.

## Reproduce

`curve_prune.py` (+ `curve_prune_io.py`) converts any full-form MiniMax-H3 checkpoint, streaming, CPU only (~2 min for
this file); `curve_check.py` verifies keys/shapes against a reference pruned header and reports per-layer error. Run
with ComfyUI's Python from inside a ComfyUI install (uses `comfy_kitchen` for int8 convrot dequantisation).

```
python -s curve_prune.py Minimax-h3_Singularity_ref2va_v2_int8.safetensors Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors
python -s curve_check.py Minimax-h3_Singularity_ref2va_v2_int8.safetensors Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors
```

Note: HyperFlow on a pruned base needs a curve refit (ComfyUI-HyperFlow-H3 "Create HyperFlow H3 Curve Fit").
