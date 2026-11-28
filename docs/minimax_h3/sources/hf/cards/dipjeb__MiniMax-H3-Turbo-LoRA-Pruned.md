# dipjeb/MiniMax-H3-Turbo-LoRA-Pruned

Источник: https://huggingface.co/dipjeb/MiniMax-H3-Turbo-LoRA-Pruned · скачано 2026-11-28
Загрузок: 899 · лайков: 2 · базовая: larryvrh/MiniMax-H3-Turbo-Lora · теги: lora,comfyui,base_model:larryvrh/MiniMax-H3-Turbo-Lora,base_model:adapter:larryvrh/MiniMax-H3-Turbo-Lora · обновлено: 2026-10-03T13:17:57.000Z

---

---
license: apache-2.0
base_model: larryvrh/MiniMax-H3-Turbo-Lora
tags:
  - minimax-h3
  - lora
  - comfyui
  - video-generation
---

# MiniMax-H3 Turbo LoRA (v4 step600 EMA) for pruned H3 models in core ComfyUI

This is a modified copy of `minimax_h3_turbo_v4_step600_ema.safetensors` from
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).
It loads in **core ComfyUI** (`Load LoRA (Model Only)`) onto the **pruned** MiniMax-H3 checkpoints,
for example `minimax_h3_fl2va_pruned_int8_convrot.safetensors` from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). It needs no custom nodes.

## Why this exists

The pruned H3 checkpoints replace the time embedder with an 8-wide curve table (`adaln_t_table`).
Their `adaln_proj.linear` weights are therefore `[out, 8]` instead of `[out, 2688]`. The original
LoRA was trained on the full model. When core ComfyUI loads it onto a pruned model, the 51
`adaln_proj` patches fail to reshape. ComfyUI logs `ERROR lora ... adaln_proj` once per key on
every sampling step and skips that part of the LoRA.

## What was changed

- Each dense `adaln_proj.linear` LoRA pair (`lora_A [r, 2688]`, `lora_B [out, r]`) was rebased onto the
  pruned curve basis and stored as ComfyUI-native full deltas:
  - `<module>.diff` = `B @ (A @ V)` → `[out, 8]` (weight delta)
  - `<module>.diff_b` = `B @ (A @ c)` → `[out]` (bias delta)

  `c` and `V` come from a least-squares fit `silu(t_emb)[i] ≈ c + V @ adaln_t_table[i]` over the 1025-point
  time grid. The grid comes from the non-pruned `minimax_h3_fl2va_int8_convrot` checkpoint and the table from
  `minimax_h3_fl2va_pruned_int8_convrot`. The relative fit residual is **9.0e-05**.
- All other (backbone) LoRA tensors are copied unchanged.
- The deltas are stored in fp32.

The result applies the full LoRA, AdaLN included, through the standard core LoRA loader, with no log errors.

## Usage

- Put the file in `ComfyUI/models/loras/`.
- Use `Load LoRA (Model Only)` after `Load Diffusion Model`. The original workflow used strength 0.8.
- Sample with 4 steps, `KSamplerSelect` → `euler`, and `BasicScheduler` → `simple`.
- Use it **only with the pruned fl2va H3 model**. On a non-pruned model, use the
  [original LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) instead.

## License and credit

- The original LoRA is by **larryvrh**, released under Apache-2.0. This modified version is also Apache-2.0.
- MiniMax-H3 itself is under the
  [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
- The AdaLN rebasing method follows the "H3 AdaLN LoRA Fix" (port mode) approach.
