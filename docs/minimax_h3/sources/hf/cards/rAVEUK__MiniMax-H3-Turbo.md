# rAVEUK/MiniMax-H3-Turbo

Источник: https://huggingface.co/rAVEUK/MiniMax-H3-Turbo · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: turbo,comfyui · обновлено: 2026-10-18T18:26:08.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
- MiniMaxAI/MiniMax-H3
pipeline_tag: text-to-video
library_name: diffusers
tags:
- audio-video
- minimax-h3
- turbo
- comfyui
---

<h1 align="center" style="font-size: 42px; line-height: 1.15; font-weight: 800;">MiniMax-H3-Turbo</h1>

<p align="center">
  <a href="https://github.com/wuli-art/MiniMax-H3-Turbo-ComfyUI">
    <img src="assets/comfyui-badge.svg" alt="ComfyUI: 4 / 6 / 8 steps" width="288" height="44" class="inline-block" />
  </a>
</p>

Fixed-step MiniMax-H3 checkpoints for text-to-video with audio and first-frame-to-video with audio. This repository contains three variants:

- `turbo-4step/`: four-step LoRA with fused video and audio output projections.
- `turbo-8step/`: eight denoising steps.
- `turbo-6step/`: six denoising steps.

The 6- and 8-step variants store a complete Diffusers-format transformer. The
4-step variant stores a compact adapter for the official transformer, including
a backbone LoRA and fused video/audio output projections. The demo downloads
the remaining components from `MiniMaxAI/MiniMax-H3`. The output is one MP4
containing H.264 video and AAC audio; no separate WAV is written.

## Interactive Demo

[![MiniMax-H3 video comparison preview — click to open the interactive demo](https://cdn.wuli.art/image/minimax-h3-comparison-20261009-0ff4412f/assets/model-card-demo-preview-17a3f8ce4b65.jpg)](https://cdn.wuli.art/image/minimax-h3-comparison-20261009-0ff4412f/index.html)

[**Explore the MiniMax-H3 video comparison demo →**](https://cdn.wuli.art/image/minimax-h3-comparison-20261009-0ff4412f/index.html)

Compare **Wuli at 4, 6 and 8 steps** with Larry v4, LightX2V, FlashGen, DMAD, PAI and the original MiniMax-H3 at 50 steps. The demo includes **9 prompts, 11 configurations and 99 videos**, with synchronized playback, step-count filters, side-by-side comparisons and selectable audio.

## ComfyUI

Use the [MiniMax H3 Turbo ComfyUI node pack](https://github.com/wuli-art/MiniMax-H3-Turbo-ComfyUI) to generate video and audio with **4, 6, or 8 steps** and save them together as one MP4.

### 1. Install the nodes

Use a local ComfyUI installation with CUDA-enabled PyTorch. Clone the node pack into `ComfyUI/custom_nodes/` and install its dependencies with the **same Python environment that runs ComfyUI**:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/wuli-art/MiniMax-H3-Turbo-ComfyUI.git
cd MiniMax-H3-Turbo-ComfyUI
python -m pip install -r requirements.txt
```

For Windows Portable, run the dependency command from the `ComfyUI_windows_portable` directory using its embedded Python:

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI