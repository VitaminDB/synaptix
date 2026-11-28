# NeuralDreamer/h3ErosMax_beta5_turbo_lora

Источник: https://huggingface.co/NeuralDreamer/h3ErosMax_beta5_turbo_lora · скачано 2026-11-28
Загрузок: 0 · лайков: 17 · базовая:  · теги:  · обновлено: 2026-09-14T08:39:13.000Z

---

---
model:
    name: LoRA for 10Eros-Max
    type: LoRA
    base_model: TenStrip/10Eros-Max
    tags:
      - audio
      - lora
      - minimax
    license: other
    license_name: minimax-h3-community-license-agreement
    license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE

---



# LoRA for the [10Eros‑Max](https://huggingface.co/TenStrip/10Eros-Max)

This LoRA was created by simply extracting the weight difference between `10Eros_Max_h3_TURBO-hybrid_beta5` and `10Eros_Max_h3_hybrid_beta5`.

The goal is simple: to enable `10Eros_Max_h3_hybrid_beta5` to work with fewer steps (14–16) while maintaining audio quality.

## How to use

Just apply this LoRA (rank 128) with a weight of 0.4–0.6 to the `10Eros_Max_h3_hybrid_beta5` model and use `"spectrum"` to achieve a good balance between image/audio quality and generation speed.  
`"Res_multistep + Simple"` is a good starting point.

## Credits

- [MiniMax](https://huggingface.co/MiniMaxAI)
- [TenStrip](https://huggingface.co/TenStrip)
