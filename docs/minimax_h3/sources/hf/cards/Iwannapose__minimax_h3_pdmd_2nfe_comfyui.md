# Iwannapose/minimax_h3_pdmd_2nfe_comfyui

Источник: https://huggingface.co/Iwannapose/minimax_h3_pdmd_2nfe_comfyui · скачано 2026-11-28
Загрузок: 4214 · лайков: 20 · базовая: MiniMaxAI/MiniMax-H3 · теги: distillation,lora,comfyui · обновлено: 2026-09-30T21:50:41.000Z

---

---
license: apache-2.0
library_name: diffusers
base_model: MiniMaxAI/MiniMax-H3
pipeline_tag: text-to-video
tags:
  - video-generation
  - distillation
  - pdmd
  - lora
  - comfyui
---

# PDMD 2-NFE LoRA for MiniMax-H3 — ComfyUI format

ComfyUI-format conversion of the [PDMD 2-NFE LoRA](https://huggingface.co/pdmd2026/pdmd_2NFE_lora)
— the 2-step (2 NFE) student distilled from MiniMax-H3-33B with Projected Distribution Matching
Distillation, rank 128, covering attention projections and both feed-forward layers of all 50
transformer blocks + 2 token-refiner blocks.
(Companion repo for the 4-NFE variant: `Iwannapose/minimax_h3_pdmd_4nfe_comfyui`.)

## Files

| file | description |
|---|---|
| `minimax_h3_pdmd_2nfe_comfyui.safetensors` | ComfyUI conversion, **bf16** (recommended), stock-loader compatible. |
| `minimax_h3_pdmd_2nfe_comfyui_fp32.safetensors` | ComfyUI conversion, **fp32** (same values; larger, no precision benefit on GPU). |

The two are numerically identical (the source fp32 file is an exact cast of the bf16 one); the
bf16 file is the one to use. Both are larger than the source because the fused `qkv_proj`
stores the block-diagonal zero padding explicitly.

## What the conversion does

Source keys are Diffusers PEFT names (`transformer.transformer_blocks.N.attn.to_q.lora_A.weight` …);
target keys follow the ComfyUI H3 layout (same scheme as the proven
`minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`):

1. **q/k/v → fused `qkv_proj`**: `A = [A_q; A_k; A_v]` (rows), `B = block_diag(B_q, B_k, B_v)`,
   row order `[q; k; v]` (A `[384, 5376]`, B `[21504, 384]`).
2. **SwiGLU remap**: diffusers `SwiGLU` outputs `[value; gate]`, ComfyUI's H3 `_swiglu_eager`
   expects `[gate; up]` — the two 14336-row halves of every `mlp.fc1.lora_B` are swapped.
3. **alpha entries**: `qkv_proj = 384` (3 × 128), `out_proj/fc1/fc2 = 128`, so ComfyUI's
   `alpha/rank` scale = **1.0** = the PDMD fusion scale (`W += (B @ A)`, alpha/rank = 128/128).
4. `to_out.0 → attn.out_proj`, `ff.net.0.proj → mlp.fc1`, `ff.net.2 → mlp.fc2`;
   `transformer_blocks.N → blocks.N`, `token_refiner.refiner_blocks.N → token_refiner.blocks.N`.

Verified: all 468 numeric checks pass against each original (exact in fp32), the key set is
identical to the known-good turbo LoRA, and all 208 target keys exist with matching dimensions
in `minimax_h3_ref2va_int8_convrot.safetensors`. The conversion output was also diffed
tensor-by-tensor against an independent community converter — identical.

## Usage (ComfyUI)

- Node: **LoraLoaderModelOnly** (model part only — the H3 text encoder is loaded separately).
- **Strength: 1.0** (any other value scales the distillation delta and changes the 2-step
  behavior — do not use it as a partial-style LoRA).
- Sample at **2 denoising steps** (the student's operating point); H3 scheduler config
  (shift 12 video / 3 audio), no CFG (H3 is guidance-distilled).
