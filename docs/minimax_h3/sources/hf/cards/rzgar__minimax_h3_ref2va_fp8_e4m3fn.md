# rzgar/minimax_h3_ref2va_fp8_e4m3fn

Источник: https://huggingface.co/rzgar/minimax_h3_ref2va_fp8_e4m3fn · скачано 2026-11-28
Загрузок: 2958 · лайков: 5 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-08T05:13:53.000Z

---

---
license: apache-2.0
base_model:
- MiniMaxAI/MiniMax-H3
base_model_relation: quantized
pipeline_tag: image-to-video
library_name: diffusers
tags:
  - text-to-video
  - image-to-video
  - image-text-to-video
  - video-to-video
  - text-to-audio-video
  - image-to-audio-video
  - image-text-to-audio-video
  - video-to-audio-video
  - audio-to-audio-video
  - audio-video-generation
  - multimodal
  - synchronized-audio-video
  - reference-to-audio-video
---

# MiniMax H3 Ref2VA (mxfp8 & fp16attn_fp8)

Quantized versions of the **MiniMax H3 Ref2VA** (reference-to-video-audio)
diffusion model. Both keep the text-conditioning path (`token_refiner`,
conditioning projections, audio heads) in full precision.

**Latest comfy version improves audio quality at low steps.**

_minimal 5 step (talking heads, less motions) | balanced 8-steps (shift 8) | Action-rich scenes above 12 (shift max 12)_

<video controls width="1280" height="720">
  <source src="https://huggingface.co/rzgar/minimax_h3_ref2va_fp8_e4m3fn/resolve/main/demo/demo_comfyV031x.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<img src="https://huggingface.co/rzgar/minimax_h3_ref2va_fp8_e4m3fn/resolve/main/demo/ksampler.png" width="1280" height="720" />

| File | Download | Note |
|---|---|---|
| `minimax_h3_ref2va_fp16attn_fp8.safetensors` | [Download 28.7GB](https://huggingface.co/rzgar/minimax_h3_ref2va_fp8_e4m3fn/resolve/main/minimax_h3_ref2va_fp16attn_fp8.safetensors) | FP16 attention + FP8-E4M3 MLP |
| `minimax_h3_ref2va_mxfp8.safetensors` | [Download 47.6GB](https://huggingface.co/rzgar/minimax_h3_ref2va_fp8_e4m3fn/resolve/main/minimax_h3_ref2va_mxfp8.safetensors) | E4M3 weights, E8M0 block scales |

## Comparison

| | **fp16attn_fp8** | **mxfp8** |
|---|---|---|
| Size | 28.7 GB | 47.6 GB |
| Base checkpoint | pruned fp8_scaled (curve-form, `adaln_t_table`) | full bf16 (time-embedder) |
| Attention (qkv/out) | **FP16** (exact) | MXFP8 (block-quantized) |
| MLP (fc1/fc2) | FP8-E4M3 per-tensor absmax | MXFP8 block-quantized |
| Quant scale granularity | per-tensor | per-32 block (better outlier handling) |
| `input_scale` | calibrated (fc1), fpmm on fc2 | 1.0 identity (150 layers) |
| token_refiner / conditioning | full precision | full precision |
| Relative precision | higher than pruned fp8, attn lossless | highest of the two (finer scales) |
| Memory footprint | lighter (faster load, less streaming) | heavier |
| Best for | everyday use / speed-first setups | maximum quality per weight bit |
