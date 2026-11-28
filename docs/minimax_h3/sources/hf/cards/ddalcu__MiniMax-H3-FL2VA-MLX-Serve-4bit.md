# ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit

Источник: https://huggingface.co/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit · скачано 2026-11-28
Загрузок: 1956 · лайков: 14 · базовая: MiniMaxAI/MiniMax-H3 · теги: base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-07T04:01:25.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: quantized
library_name: mlx
tags:
- mlx-serve
- text-to-video
- image-text-to-video
- audio-video-generation
---

# MiniMax-H3 FL2VA — MLX-Serve 4-bit

4-bit affine (group size 64) conversion of MiniMax-H3's FL2VA checkpoint for
[mlx-serve](https://github.com/ddalcu/mlx-serve), running natively on Apple
Silicon. Text-to-audio-video: the DiT denoises video and stereo audio jointly
in one packed sequence.

Self-contained: weights, both VAEs and the tokenizer in one directory. Upstream
splits these across `Comfy-Org/MiniMax-H3` (weights, no tokenizer) and
`MiniMaxAI/MiniMax-H3` (tokenizer).

Quantized: the DiT and text-encoder matmul weights. Kept dense: gathered
embedding tables, the checkpoint's fp32 islands (patch projections, output
heads, time embedder) and both VAEs.

Note that quantization here buys FOOTPRINT, not speed — the workload is
compute-bound at roughly 192,000 FLOPs per weight byte.

## Modifications

These are MODIFIED files. The transformer and text encoder are quantized to
4-bit; see MODIFICATIONS.md for the full list. The VAEs and tokenizer are
unmodified copies.

## Turbo LoRA

`turbo_lora.safetensors` is included. It is a 4-step distillation adapter
trained by [larryvrh](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
and licensed Apache-2.0, copied byte for byte from that repo's
`minimax_h3_turbo_4step_ema_ckpt850.safetensors` and renamed so mlx-serve finds it
inside the pack. Nothing about it was changed.

It samples in 4 to 8 steps instead of the usual 16 to 30 (larryvrh notes 4 is
already sharp on this checkpoint; earlier ones needed 6 to 8). Measured on an M4 Max
at 960x544, 39 frames, one prompt: 188s of sampling at 16 steps without it
against 95s at 4 steps with it, so roughly 2x on sampling and 1.8x on the whole
request once staging and VAE decode are counted. Per step it is about 20%
slower, because the adapter is summed at runtime rather than folded into the
weights and it turns off mlx-serve's step-caching recipe. The win is the step
count, not the step.

Treat it as a preview. This is the EMA weighting of checkpoint 850, which
larryvrh recommends as the cleanest of the round; the non-EMA file of the same
step is sharper still but over-sharpened, and their notes describe plastic skin
and over-sharp grain as the round's known failure modes. Their guidance if you
hit either: nudge LoRA strength down to 0.8-0.95 for grain, up to 1.05-1.2 for
blurry ghosting.

Request it with `"turbo": true`, or tick Turbo in the mlx-serve app's video
pane. It stacks with your own style LoRAs rather than replacing them.

## License

Powered by MiniMax H3. Licensed under the MiniMax H3 Community License
Agreement -- see LICENSE and NOTICE, both included here.

**Territorial restriction.** The Agreement defines the Applicable Territory as
worldwide EXCLUDING the European Union, the United Kingdom, the Republic of
Korea and the United States of America, and Section V.4 prohibits use,
reproduction, modification, distribution and display outside it. Check whether
your jurisdiction permits you to use these files before downloading them.
