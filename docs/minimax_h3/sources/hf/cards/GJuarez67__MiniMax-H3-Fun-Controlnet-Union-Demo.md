# GJuarez67/MiniMax-H3-Fun-Controlnet-Union-Demo

Источник: https://huggingface.co/GJuarez67/MiniMax-H3-Fun-Controlnet-Union-Demo · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая:  · теги: controlnet · обновлено: 2026-08-24T17:46:41.000Z

---

---
title: MiniMax-H3-Fun-Controlnet-Union
description: Showcase Space for the MiniMax-H3-Fun-Controlnet-Union model — a single ControlNet-Union checkpoint for Canny, Depth, HED, MLSD, and Pose control on MiniMax-H3 video generation.
tags:
  - controlnet
  - video-generation
  - minimax-h3
  - video-to-video
  - gradio
---

# MiniMax-H3-Fun-Controlnet-Union — Hugging Face Space

A showcase Space for the [MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) model by alibaba-pai.

## What is this?

MiniMax-H3-Fun-Controlnet-Union is a **ControlNet-Union** checkpoint for the [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) video diffusion transformer. One single checkpoint handles **Canny, Depth, HED, MLSD, and Pose** control conditions for video-to-video generation — no per-condition checkpoint switching — and also supports video inpainting.

The control branch (~6.8 GB) is loaded on top of the base MiniMax-H3 transformer and injects control signals at 5 points (layers 0, 10, 20, 30, 40) of the 50-block architecture.

## Space contents

This Space shows:

- **Model overview** — description, file table, and key features
- **Inference defaults** — the settings used for all published results
- **5 result pairs** — control input + generated output for each control type (Canny, Depth, HED, MLSD, Pose), sourced directly from the model repo
- **Setup guide** — how to run inference via the [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) pipeline

## Inference requirements

Running the actual model requires:

- The base **MiniMax-H3** transformer (~62 GB)
- The **Qwen3-VL** text encoder (~62 GB)
- This ControlNet-Union checkpoint (~6.8 GB)
- VideoX-Fun installed with the matching `minimax_h3_control.yaml` config

The full model does **not** fit a single 80 GB GPU fully loaded — use `model_group_offload` or `model_cpu_offload_and_qfloat8`.

See the Space app for the full step-by-step inference guide.
