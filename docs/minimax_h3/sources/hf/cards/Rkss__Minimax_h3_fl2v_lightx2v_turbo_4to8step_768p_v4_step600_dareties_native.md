# Rkss/Minimax_h3_fl2v_lightx2v_turbo_4to8step_768p_v4_step600_dareties_native

Источник: https://huggingface.co/Rkss/Minimax_h3_fl2v_lightx2v_turbo_4to8step_768p_v4_step600_dareties_native · скачано 2026-11-28
Загрузок: 6549 · лайков: 2 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,comfyui · обновлено: 2026-09-16T20:40:19.000Z

---

---
tags:
  - video-generation
  - lora
  - comfyui
  - minimax-h3
  - experimental
base_model: MiniMaxAI/MiniMax-H3
---

# silveroxides's MiniMax H3  Turbo LoRA (Experimental)

Experimental native-pruned version of SilverOxides' MiniMax H3 Turbo LoRA for ComfyUI.

Minimax_h3_fl2v_lightx2v_turbo_4to8step_v0.1-v1.0_768p_v4_step600_dareties_fro095

Load with ComfyUI's LoRA loader node. 

Two versions. Pruned Adaln modules and a ported one.

**This is not mathematically equivalent to the original LoRA**

## Original LoRA

Source:  https://huggingface.co/silveroxides/MiniMax-H3_tests/tree/main/experimental

## Recommended strength

- **FL2VA:** `0.80 – 1.1`
- **ref2VA:** `1.0 – 1.1`

Low strength may cause artifacts, softness, or voice instability. Use when you see oversharpen/detailed

High strength may become overcooked.   Use when you see  blurry ghosting.

Results may vary between workflows.

## Recommended settings

- Steps: 4-8 . It does work with 4 steps, but higher is better
- Sampler: **Euler**
- Scheduler: You can experiment with **Beta57**
- If you want to use it with ref2va. Use the hybrid fl2va-ref2va model; recommend b20-49. https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models

## Notes

This is an **experimental build**.
