# mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA

Источник: https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA · скачано 2026-11-28
Загрузок: 0 · лайков: 94 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: lora · обновлено: 2026-08-20T16:32:37.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
  - MiniMaxAI/MiniMax-H3
pipeline_tag: text-to-video
tags:
  - autoregressive
  - acceleration
  - streaming
  - realtime
  - diffusion
  - lora
---

# RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO

[Yanzuo Lu](https://yanzuo.lu/) · [Ronglai Zuo](https://2000zrl.github.io/) · [Jiankang Deng](https://jiankangdeng.github.io/) — Imperial College London

Project page: <https://yanzuo.lu/raven>

[![arXiv](https://img.shields.io/badge/arXiv-2605.15190-b31b1b.svg)](https://arxiv.org/abs/2605.15190) [![GitHub](https://img.shields.io/badge/GitHub-mvp--ai--lab%2FRAVEN-181717)](https://github.com/mvp-ai-lab/RAVEN)

## Release Notes

- **20 Aug, 2026.**  We’ve released [ComfyUI nodes](https://github.com/YanzuoLu/ComfyUI-MiniMax-H3-RAVEN-Streaming) for RAVEN streaming generation, enabling 192-frame 1376×768 T2VA generation within a 24 GiB VRAM envelope. This path still places a substantial demand on system RAM, and further memory optimizations are on the way.
- Many thanks to @MiniMax_AI for releasing and making this exploration possible. This is a quite large model, so reaching real-time generation will **require further inference acceleration**. We’re actively working on it. Contributions are all very welcome.
- Our weights can also use training-free extensions such as capped RoPE to support **long generation**. Our focus is now on texture detail, so we haven't tested this capability yet. If you are interested, just experiment with it and share what you find!
- Literally we don't request higher requirements on compute resources versus base model, maybe **even lower** since we process one chunk by one dit forward. Everyone can give it a try but currently we're still trying to improve texture details as first priority.

## Overview

**MiniMax-H3 4-NFE streaming preview.** This is an initial preview release. The adapter is still undertrained and its texture details remain limited. It nevertheless establishes the complete end-to-end training pipeline for RAVEN-style streaming generation on MiniMax-H3. We validated the pipeline from training through generation. More to come.

The released weight is a LoRA adapter that turns [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) into a causal streaming generator. Video is produced chunk by chunk, and each chunk is extrapolated from previously generated content instead of denoised as one bidirectional clip. This adaptation is not one of the models evaluated in the paper.

Training, inference, and evaluation code live in the [code repository](https://github.com/mvp-ai-lab/RAVEN). Other artifacts are listed in the [RAVEN collection](https://huggingface.co/collections/mvp-lab/raven).

## Demo

<video controls loop muted playsinline preload="metadata" width="100%" src="https://github.com/user-attachments/assets/96047839-e7cb-4416-a82b-4acb59f8bab3"></video>

## Files

| File | Description |
| --- | --- |
| [MiniMax-H3-RAVEN-Streaming-LoRA-4NFE-Preview](https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA/blob/main/minimax_h3_raven_streaming_lora_4nfe_preview.safetensors) | PEFT-layout streaming LoRA adapter for MiniMax-H3, loaded through the `adapter` block of a RAVEN trial config. |

## Model details

| Item | Value |
| --- | --- |
| Base model | `MiniMaxAI/MiniMax-H3` |
| Adapter | LoRA, `r=128`, `lora_alpha=128` |
| Sampling | 4 NFE (video and audio grids) |
| Frames | 192 |
| Resolution | 768 x 1376 |
| Frame rate | 24 fps |
| Causal chunking | `sink=2`, `window=2` |

## Usage

### Setup

```sh
git clone https://github.com/mvp-ai-lab/RAVEN.git
cd RAVEN
conda env create -f tools/environment.yaml
bash tools/prepare_venv.sh
source venv/bin/activate
```

### Download

```sh
hf download mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA --local-dir /path/to/minimax-h3-raven-lora
```

### Configure and run

Trial YAML files keep site-specific absolute paths. Complete the following steps before running.

- Obtain the MiniMax-H3 base components separately under the MiniMax-H3 Community License.
- Update the tokenizer, DiT, text encoder, video VAE, audio VAE, and `models.backbone.adapter.weight` paths.
- The bundled trial is a validation-only `BaseEngine` run configured for one 8-GPU node (4-way unified parallelism, FSDP shard size 8). Training implementations live in `projects/minimax_h3/meta_models/`.

```sh
N=8 bash tools/multi_run.sh \
  projects/minimax_h3/trials/base/causal_minimax_h3_base/minimax_h3_raven_streaming_lora_4nfe_preview.yaml
```

## License

This adapter is released under the [MiniMax-H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Users are additionally responsible for complying with the licenses and terms of the base model, the code repository, and all dependencies.

## Citation

```bibtex
@article{lu2026raven,
  title = {RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO},
  author = {Lu, Yanzuo and Zuo, Ronglai and Deng, Jiankang},
  year = 2026,
  journal = {arXiv preprint arXiv:2605.15190}
}
```
