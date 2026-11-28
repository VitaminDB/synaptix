# SanDiegoDude/H3-Turbo-6-Step-LoRA-Comfy

Источник: https://huggingface.co/SanDiegoDude/H3-Turbo-6-Step-LoRA-Comfy · скачано 2026-11-28
Загрузок: 0 · лайков: 4 · базовая: Comfy-Org/MiniMax-H3 · теги: lora,comfyui,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-07T20:35:41.000Z

---

---
license: apache-2.0
base_model: Comfy-Org/MiniMax-H3
tags:
  - text-to-video
  - text-to-audio
  - audio-video
  - lora
  - minimax-h3
  - comfyui
pipeline_tag: text-to-video
---

# MiniMax-H3 Turbo LoRA — 6-step, native ComfyUI (BF16)

ComfyUI-native conversions of [**larryvrh/MiniMax-H3-Turbo-Lora**](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora),
so they load with the stock `LoraLoader` against core MiniMax-H3 support — no custom
nodes, no custom sampler.

All credit for the actual training goes to [**larryvrh**](https://huggingface.co/larryvrh).
These are just format conversions of that work. Base model is
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Requires BF16

These are BF16 and expect a BF16 pipeline. They will not behave correctly in FP16.

## Call it 6 steps, not 4

Upstream is named "4step". In practice 4 steps is a little weak — **6 steps** is the
sweet spot. The files here are renamed accordingly; the weights are unchanged.

## Files

| file | for base checkpoint |
|---|---|
| `minimax_h3_turbo_6step_ema.safetensors` | `minimax_h3_fl2va_bf16` |
| `minimax_h3_turbo_6step.safetensors` | `minimax_h3_fl2va_bf16` |
| `minimax_h3_turbo_6step_ema_fl2va_pruned.safetensors` | `minimax_h3_fl2va_pruned_int8_convrot` |
| `minimax_h3_turbo_6step_fl2va_pruned.safetensors` | `minimax_h3_fl2va_pruned_int8_convrot` |

**Start with the EMA one.** It is noticeably cleaner than the non-EMA variant.

The `_fl2va_pruned` pair are remapped for the pruned int8/convrot checkpoint rather
than the full BF16 one. Match the LoRA to the checkpoint you actually loaded.

## Also works on the ref model

Trained for `fl2va` (first/last-frame), but it works on the `ref2va`
(reference-to-video) path too, even though it was never trained for it. Pleasant
surprise — worth trying.

## Recommended settings

Sampler `euler`, scheduler `beta57`, **6 steps**.

Strongly recommended: schedule the LoRA strength rather than pinning it flat. Using
**SCG LoRA Scheduler** from [SanDiegoDude/scg-utils](https://github.com/SanDiegoDude/scg-utils/):

| setting | value |
|---|---|
| `strength_start` | 4.00 |
| `strength_end` | 1.00 |
| `start_percent` | 0.100 |
| `end_percent` | 0.990 |
| `interpolation` | linear |
| `cutoff_outside_window` | true |
| `strength_clip` | 1.00 |

Ramping down from 4.0 to 1.0 across the run gives the best output I've found so far.

## License

Apache-2.0, same as upstream.
