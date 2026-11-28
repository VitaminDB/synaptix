# Iwannapose/minimax_h3_pdmd_4nfe_comfyui

Источник: https://huggingface.co/Iwannapose/minimax_h3_pdmd_4nfe_comfyui · скачано 2026-11-28
Загрузок: 2494 · лайков: 19 · базовая: MiniMaxAI/MiniMax-H3 · теги: distillation,lora,comfyui · обновлено: 2026-09-30T19:45:44.000Z

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

# PDMD 4-NFE LoRA for MiniMax-H3 — ComfyUI format

ComfyUI-format conversion of the [PDMD 4-NFE LoRA](https://huggingface.co/pdmd2026/pdmd_4NFE_lora)
(`lora_model_0.safetensors`) — the 4-step (4 NFE) student distilled from MiniMax-H3-33B with
Projected Distribution Matching Distillation, rank 128, covering attention projections and both
feed-forward layers of all 50 transformer blocks + 2 token-refiner blocks.

## Files

| file | description |
|---|---|
| `minimax_h3_pdmd_4nfe_comfyui_v6.safetensors` | **Use this one.** Correct ComfyUI conversion (bf16). |

Earlier uploads (base / keys_v2 / v3 / v4 / v5) were broken conversions and have been removed:
most of their keys matched nothing in the ComfyUI loader (silent no-op), and v4 missed the
SwiGLU gate/value remap below.

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

## Usage (ComfyUI)

- Node: **LoraLoaderModelOnly** (model part only — the H3 text encoder is loaded separately).
- **Strength: 1.0** (any other value scales the distillation delta, which changes the 4-step
  behavior — do not use it as a partial-style LoRA).
- Sample at **4 denoising steps** (the student's operating point); H3 scheduler config
  (shift 12 video / 3 audio), no CFG (H3 is guidance-distilled).
- Works on the stock ComfyUI H3 loader; verified that all 208 target keys exist with matching
  dimensions in `minimax_h3_ref2va_int8_convrot.safetensors`, and every tensor is numerically
  identical to the original under the transforms above.
