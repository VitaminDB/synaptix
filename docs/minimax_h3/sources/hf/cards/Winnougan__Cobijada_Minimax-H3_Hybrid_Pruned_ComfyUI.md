# Winnougan/Cobijada_Minimax-H3_Hybrid_Pruned_ComfyUI

Источник: https://huggingface.co/Winnougan/Cobijada_Minimax-H3_Hybrid_Pruned_ComfyUI · скачано 2026-11-28
Загрузок: 448349 · лайков: 133 · базовая:  · теги: comfyui,quantization · обновлено: 2026-09-23T13:20:21.000Z

---

---
license: apache-2.0
tags:
  - comfyui
  - video-generation
  - image-to-video
  - text-to-video
  - model-merge
  - quantization
  - int8
  - w4a8
  - bf16
  - minimax-h3
---

# Cobijada MiniMax-H3 Hybrid (Pruned, Uncensored)

![Cobijada MiniMax-H3 Hybrid](https://huggingface.co/Winnougan/Cobijada_Minimax-H3_Hybrid_Pruned_ComfyUI/resolve/main/H3_Hybrid.png)

A merged variant of MiniMax H3, a joint audio+video diffusion transformer (DiT) for ComfyUI, combining the strengths of the two source conditioning modes — first/last-keyframe (FL) and reference-driven (REF) — into a single checkpoint.

## Why this merge exists

MiniMax H3 ships two distinct training regimes with identical architecture and tensor layout:

- **FL (first/last-keyframe conditioning)** — higher raw visual and audio output quality, but no support for reference-driven generation.
- **REF (reference conditioning)** — adds image/video/audio reference-driven generation, but at a real cost to output quality, even on non-reference tasks.

A tensor-by-tensor comparison of the two checkpoints shows the large majority of weights — attention projections, MLPs, norms, patch projections, rotary embeddings, and the token refiner — are effectively identical between them. The meaningful differences are concentrated almost entirely in the per-block `adaln_proj` weights: the AdaLN modulation projections that route conditioning signal into the residual stream at each transformer block.

That made a targeted, non-training weight-selection merge viable rather than a lossy compromise.

## What this model is

This hybrid uses the **FL checkpoint as the base** for the entire network, with the **`adaln_proj` weights for transformer blocks 25–49 (the later half of the network) taken from the REF checkpoint** instead. Everything else — including the earlier-block `adaln_proj` weights, the final AdaLN projection, attention, MLPs, norms, and output heads — remains on FL throughout.

The intent is to keep REF's reference-conditioning pathway, which is expressed primarily through those later-block AdaLN weights, while retaining FL's higher-fidelity weights everywhere else in the network.

No additional training, fine-tuning, or gradient-based optimization was performed — this is a weight-selection merge at the tensor level, not a fine-tune. Both source checkpoints were the uncensored, pruned variants before merging.

## Files

| File | Format | Notes |
|---|---|---|
| `Cobijada_H3_Hybrid_Uncensored_pruned_hybrid_BF16_pruned.safetensors` | BF16 | Full-precision merge output; base for the quantized variants below |
| `Cobijada_H3_Hybrid_Uncensored_pruned_hybrid_int8_convrot_pruned.safetensors` | INT8 (ConvRot) | Quantized from the BF16 hybrid |
| `Cobijada_H3_Hybrid_Uncensored_pruned_hybrid_w4a8_convrot_pruned.safetensors` | W4A8 (ConvRot) | Quantized from the BF16 hybrid |

All three are pruned variants.

## Usage (ComfyUI)

1. Download the format you want and place it in `ComfyUI/models/diffusion_models/`.
2. Load it with your normal MiniMax H3 checkpoint loader — architecture and tensor layout are unchanged from the source models, so no custom node is required to load the hybrid itself.
3. Reference-conditioned workflows (image/video/audio reference input) and standard first/last-keyframe workflows both work with this single checkpoint.

## Intended use

- Text/image/video/audio-to-video generation where you want reference conditioning while keeping output quality closer to the FL checkpoint.
- A drop-in replacement for the REF checkpoint in reference-conditioned workflows, for users who found REF's raw output quality lacking.

This model is not expected to exceed the FL checkpoint's quality on non-reference-conditioned generation, since the large majority of its weights are shared with FL to begin with.

## License

This merge, the quantized conversions, and this repository are released under **Apache 2.0**. The underlying MiniMax H3 weights remain subject to MiniMax's own original license terms — please review those separately before use.

## Credits

Merged and quantized by [Winnougan](https://huggingface.co/Winnougan). Built on the officially released MiniMax H3 FL and REF checkpoints.
