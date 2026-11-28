# NaukNauk/minimax-h3-turbo-ref2va-merged

Источник: https://huggingface.co/NaukNauk/minimax-h3-turbo-ref2va-merged · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая: ["MiniMaxAI/MiniMax-H3","lightx2v/Minimax-h3-Turbo"] · теги: lora · обновлено: 2026-09-04T19:22:17.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE
base_model:
  - MiniMaxAI/MiniMax-H3
  - lightx2v/Minimax-h3-Turbo
library_name: sglang
pipeline_tag: image-text-to-video
tags:
  - minimax-h3
  - ref2va
  - video-generation
  - audio-video-generation
  - lightx2v
  - lora
  - merged
  - bf16
---

# MiniMax H3 Ref2VA + LightX2V 8-step 768p (merged transformer)

This repository contains the **premerged Ref2VA transformer only** for native MiniMax H3 reference-to-audio-video generation. It combines the official MiniMax H3 Ref2VA BF16 transformer with LightX2V's `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` LoRA.

It must be assembled with the unchanged processor, tokenizer, text encoder, visual VAE, and audio VAE from the pinned `MiniMaxAI/MiniMax-H3` Ref2VA partition. It is intended for 8-denoiser-evaluation, 768p `ref2va` inference. This is not a standalone LoRA and is not the Diffusers-format checkpoint.

## Provenance

| Item | Value |
| --- | --- |
| Base model | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3), native `Ref2VA` checkpoint |
| Base revision | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| LoRA source | [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |
| LoRA file | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` |
| LoRA SHA-256 | `9bac880b1a5d7ac052171cf6cce769f0cceaaa42ffa51de4b8e41143a2bdd2d2` |
| LoRA rank / alpha | `128` / `8` |
| Effective merge scale | `8 / 128 = 0.0625` |
| Merged native tensors | 208 targets from 624 LoRA tensors |
| Weight precision | BF16 |

The adapter was converted from the Diffusers key layout to the native MiniMax H3 layout during the merge. Split Q/K/V deltas were interleaved into native per-head fused QKV tensors, and the Diffusers SwiGLU value/gate order was converted back to the native gate/value order. The machine-readable merge manifest is included as `nauknauk_ref2va_merge.json`.

The LoRA release and recommended 8-step 768p settings are documented in [LightX2V discussion #51](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/51).

## Assemble and deploy with SGLang

Download the pinned base partition, then overlay this repository's transformer:

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include "model_index.json" "Ref2VA/*" \
  --local-dir MiniMax-H3

hf download NaukNauk/minimax-h3-turbo-ref2va-merged \
  --local-dir MiniMax-H3/Ref2VA
```

The checkpoint was validated with SGLang `0.5.17` on 30 reference-video cases at native 768p using four H100 80GB GPUs, tensor parallelism 4, Ulysses degree 1, and no CPU offload. All 30 cases completed, including inputs with six reference images.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sglang serve \
  --model-type diffusion \
  --backend sglang \
  --model-path MiniMax-H3 \
  --pipeline-class-name MiniMaxH3Pipeline \
  --model-variant ref2va \
  --num-gpus 4 \
  --tp-size 4 \
  --ulysses-degree 1 \
  --performance-mode speed \
  --use-fsdp-inference false \
  --text-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --dit-cpu-offload false \
  --encoder-parallel auto \
  --enable-torch-compile false \
  --host 0.0.0.0 \
  --port 30010
```

Recommended generation settings:

- `num_inference_steps`: 9 in native SGLang, which includes the terminal zero and therefore executes 8 denoiser evaluations
- video `flow_shift`: 12
- audio `flow_shift`: 3
- output short edge: 768 pixels
- frame rate: 24 FPS

The exact request schema depends on the SGLang release; consult the current SGLang MiniMax H3 API documentation.

## License and use

The merged checkpoint remains subject to the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). The LightX2V repository is published under Apache-2.0. Users are responsible for complying with both upstream terms and applicable laws.

This repository is a convenience artifact prepared by NaukNauk. It is not an official MiniMax or LightX2V release. Repository ID: `NaukNauk/minimax-h3-turbo-ref2va-merged`.
