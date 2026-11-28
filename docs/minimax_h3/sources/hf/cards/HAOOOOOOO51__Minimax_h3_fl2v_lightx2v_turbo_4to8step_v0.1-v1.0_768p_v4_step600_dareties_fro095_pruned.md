# HAOOOOOOO51/Minimax_h3_fl2v_lightx2v_turbo_4to8step_v0.1-v1.0_768p_v4_step600_dareties_fro095_pruned

Источник: https://huggingface.co/HAOOOOOOO51/Minimax_h3_fl2v_lightx2v_turbo_4to8step_v0.1-v1.0_768p_v4_step600_dareties_fro095_pruned · скачано 2026-11-28
Загрузок: 5924 · лайков: 1 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,comfyui · обновлено: 2026-08-30T15:14:30.000Z

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

# silveroxides's MiniMax H3  Turbo LoRA — Pruned (Experimental)

Experimental native-pruned version of SilverOxides' MiniMax H3 Turbo LoRA for ComfyUI.

Minimax_h3_fl2v_lightx2v_turbo_4to8step_v0.1-v1.0_768p_v4_step600_dareties_fro095

**This is not mathematically equivalent to the original LoRA because the AdaLN modules were pruned.**

## Original LoRA

Source:  https://huggingface.co/silveroxides/MiniMax-H3_tests/tree/main/experimental

## Recommended strength

- **FL2VA:** `0.80 – 1.15`
- **ref2VA:** `1.02 – 1.20`

Low strength may cause artifacts, softness or voice instability. Use when you see oversharpen/detailed

High strength may become overcooked.   Use when you see  blurry ghosting / smear

Results may vary between workflows.

## Recommended settings

- Steps: 4-8
- Sampler: **Euler**
- Scheduler: **Simple**
- You can also experiment with **Beta57**


## Notes

This is an **experimental build**.
