# ms13d/Minimax-h3-Singularity-TURBO

Источник: https://huggingface.co/ms13d/Minimax-h3-Singularity-TURBO · скачано 2026-11-28
Загрузок: 0 · лайков: 3 · базовая:  · теги:  · обновлено: 2026-11-08T11:28:54.000Z

---

# Minimax-h3 Singularity TURBO (private)

`Minimax-h3_Singularity_ref2va_Pruned_v1.3_TURBO-eros5_int8.safetensors`

- Base: [WarmBloodAban/Minimax-h3_Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity) `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors`
- Turbo: 10Eros-Max `TURBO-hybrid_beta5` minus `hybrid_beta5` (bf16), strength 1.0, added to blocks.N qkv_proj / out_proj / fc1 / fc2, re-quantized int8_tensorwise convrot g256. Everything else unchanged.
- Sampling (Eros beta5): er_sde/beta 4, er_sde/beta57 4-6, res_multistep/simple 6-8, LCM/simple 6-8, LCM/beta 4-6, euler/simple 4-8.
