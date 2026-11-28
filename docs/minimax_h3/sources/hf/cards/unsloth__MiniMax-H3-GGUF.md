# unsloth/MiniMax-H3-GGUF

Источник: https://huggingface.co/unsloth/MiniMax-H3-GGUF · скачано 2026-11-28
Загрузок: 3774406 · лайков: 460 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: gguf,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-10-16T12:52:16.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
- MiniMaxAI/MiniMax-H3
base_model_relation: quantized
language:
- en
- zh
library_name: gguf
pipeline_tag: image-text-to-video
tags:
- gguf
- text-to-video
- image-to-video
- video-generation
- stable-diffusion.cpp
- unsloth
- minimax
- multimodal
---

Instructions further below. GGUF for MiniMax-H3, compatible on most platforms including stablediffusion.cpp and Unsloth.

<img width="600" alt="minimax h3 in unsloth" src="https://cdn-uploads.huggingface.co/production/uploads/62ecdc18b72a69615d6bd857/mC1bt6NkQVUM0AINDpork.gif" />

# Run MiniMax-H3 locally
You can run MiniMax-H3 via Unsloth: https://github.com/unslothai/unsloth/


GGUF quantizations of [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)

MiniMax H3 is an omni-modal generative system that produces video with native stereo audio, up to
15 seconds at 24 FPS with 32 kHz stereo audio. Both halves of the runtime are in this repo: the
denoisers and the Qwen3-VL text encoder they need.

H3 ships two denoisers, and which one you load decides what the model can be given:

- **`fl2va_pruned`**, the H3-Base first-and-last-frame variant. Text, plus zero, one or two frames.
- **`ref2va_pruned`**, the reference variant. Text, plus reference pictures, videos and audio.

They are separate checkpoints, not settings, so pick the one that matches the task. Both are
quantized here at the same rungs, so a given quant costs about the same either way.

## Example

`UD-Q2_K_XL`, the smallest rung here, at 960x544, 124 frames, 24 FPS, 8 steps, guidance 1.0,
seed 11, on a single card.

> a red panda stepping along a mossy log in a misty forest, cinematic

<img src="assets/h3_gguf_ud_q2_k_xl.gif" width="480">

The GIF is downsampled and silent. For the full 960x544 clip with its native 32 kHz stereo audio
track, play
[`assets/h3_gguf_ud_q2_k_xl.mp4`](https://huggingface.co/unsloth/MiniMax-H3-GGUF/blob/main/assets/h3_gguf_ud_q2_k_xl.mp4).
H3 generates the audio jointly with the video, so the audio is part of the model output rather
than something added afterwards.

## Files

Text and frames, `fl2va_pruned`:

| File | Size |
|---|---|
| `minimax_h3_fl2va_pruned-Q2_K.gguf` | 6.26 GiB |
| `minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf` | 7.51 GiB |
| `minimax_h3_fl2va_pruned-Q3_K.gguf` | 8.16 GiB |
| `minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf` | 8.90 GiB |
| `minimax_h3_fl2va_pruned-Q4_K.gguf` | 10.64 GiB |
| `minimax_h3_fl2va_pruned-Q5_0.gguf` | 12.97 GiB |
| `minimax_h3_fl2va_pruned-Q6_K.gguf` | 15.45 GiB |
| `minimax_h3_fl2va_pruned-Q8_0.gguf` | 19.97 GiB |

References, `ref2va_pruned`:

| File | Size |
|---|---|
| `minimax_h3_ref2va_pruned-Q2_K.gguf` | 6.22 GiB |
| `minimax_h3_ref2va_pruned-Q3_K.gguf` | 8.12 GiB |
| `minimax_h3_ref2va_pruned-Q4_K.gguf` | 10.60 GiB |
| `minimax_h3_ref2va_pruned-Q5_0.gguf` | 12.94 GiB |
| `minimax_h3_ref2va_pruned-Q6_K.gguf` | 15.42 GiB |
| `minimax_h3_ref2va_pruned-Q8_0.gguf` | 19.94 GiB |

Text encoder, shared by both:

| File | Size |
|---|---|
| `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` | 12.20 GiB |
| `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 16.97 GiB |

The `UD-` rungs are dynamic, mixed-precision builds. The uniform rungs hold one type throughout.
Pair the `Q2_K_M` text encoder with the two smallest denoisers and the `Q4_K_M` one with everything
else. The text encoder and the VAEs are shared, so switching between the two denoisers costs one
denoiser download and nothing else. The VAEs are not duplicated here, take them from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

```bash
sd-cli --mode vid_gen \
  --diffusion-model minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf \
  --llm qwen3vl_32b_minimax_h3-Q2_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 25 --steps 4 --cfg-scale 1.0 \
  --backend te=cpu --diffusion-fa \
  --output out.webm
```

Three flags are not optional. `--mode vid_gen`, or H3 takes the image path and aborts. Explicit
`--cfg-scale 1.0`, because H3 is distilled and cfg-free and aborts above 1.0 while the default is
7.0. And `--backend te=cpu`, which keeps the 12 GB text encoder off the card. Add
`--offload-to-cpu` to fit a smaller GPU.

The pre-quantized PyTorch checkpoints are in
[unsloth/MiniMax-H3-FP8](https://huggingface.co/unsloth/MiniMax-H3-FP8).

## Licence

MiniMax H3 Community License Agreement, from MiniMax-H3. Full text in
[`LICENSE`](https://huggingface.co/unsloth/MiniMax-H3-GGUF/blob/main/LICENSE). Read it before use:
it defines an Applicable Territory and excludes some jurisdictions from it. MiniMax also publish a
[Q&A about the licence](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md).

These files are Model Derivatives, not a plain copy: the transformer and the text encoder are
quantized, which changes the numerics. Section III of the licence wants that stated, so
[`NOTICE`](https://huggingface.co/unsloth/MiniMax-H3-GGUF/blob/main/NOTICE) lists every change
along with the attribution. Not an official MiniMax product, and not endorsed by MiniMax.
