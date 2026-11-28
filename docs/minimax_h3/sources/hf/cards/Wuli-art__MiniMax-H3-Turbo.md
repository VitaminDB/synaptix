# Wuli-art/MiniMax-H3-Turbo

Источник: https://huggingface.co/Wuli-art/MiniMax-H3-Turbo · скачано 2026-11-28
Загрузок: 10277 · лайков: 72 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: turbo,comfyui · обновлено: 2026-11-11T09:08:47.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
- MiniMaxAI/MiniMax-H3
pipeline_tag: text-to-video
library_name: minimax-h3
tags:
  - diffusers
  - audio-video
  - minimax-h3
  - turbo
  - comfyui
  - ref2va
  - reference-to-video
  - image-to-video
  - few-step
  - video-generation
---

<h1 align="center" style="font-size: 42px; line-height: 1.15; font-weight: 800;">Wuli MiniMax-H3 Turbo — 8-Step Ref2VA and 4/6/8-Step T2VA/FL2VA</h1>

<p align="center" style="white-space: nowrap; overflow-x: auto;">
  <a href="https://modelscope.cn/models/Wuli-Art/MiniMax-H3-Turbo">
    <img src="https://img.shields.io/badge/ModelScope-624AFF?style=for-the-badge" alt="ModelScope" height="33" style="height:33px; width:auto;" class="inline-block" />
  </a>
  &nbsp;
  <a href="https://github.com/wuli-art/MiniMax-H3-Turbo-ComfyUI">
    <img src="assets/comfyui-badge.svg" alt="ComfyUI: 4 / 6 / 8 steps" width="216" height="33" style="height:33px; width:auto;" class="inline-block" />
  </a>
  &nbsp;
  <a href="https://huggingface.co/Wuli-art/MiniMax-H3-Turbo">
    <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black" alt="Hugging Face" height="33" style="height:33px; width:auto;" class="inline-block" />
  </a>
</p>

**Update — 2026-10-23:** Added `ref2va-turbo-8step` for reference-conditioned video generation with audio, along with `run_ref2va.py` and usage examples. Ref2VA is now supported in ComfyUI; load the [8-step Ref workflow](https://raw.githubusercontent.com/wuli-art/MiniMax-H3-Turbo-ComfyUI/main/example_workflows/ref2va_8step.json).

Wuli MiniMax-H3 Turbo provides **8-step Ref2VA (reference-to-video with audio)** from reference images or videos, with optional audio references, alongside **4-, 6-, and 8-step T2VA/FL2VA (text-to-video and frame-conditioned video generation with audio)**. All variants generate video and audio together. This repository contains four variants:

- `turbo-4step/`: four-step LoRA with fused video and audio output projections.
- `turbo-8step/`: eight denoising steps.
- `turbo-6step/`: six denoising steps.
- `ref2va-turbo-8step/`: eight-step reference-conditioned video generation with audio.

The 6- and 8-step variants, including Ref, store a complete Diffusers-format transformer. The
4-step variant stores a compact adapter for the official transformer, including
a backbone LoRA and fused video/audio output projections. The demo downloads
the remaining components from `MiniMaxAI/MiniMax-H3`. The output is one MP4
containing H.264 video and AAC audio; no separate WAV is written.

## Interactive Demo

[![MiniMax-H3 video comparison preview — click to open the interactive demo](https://cdn.wuli.art/image/minimax-h3-comparison-20261009-0ff4412f/assets/model-card-demo-preview-17a3f8ce4b65.jpg)](https://cdn.wuli.art/image/minimax-h3-comparison-20261009-0ff4412f/index.html)

[**Explore the MiniMax-H3 video comparison demo →**](https://cdn.wuli.art/image/minimax-h3-comparison-20261009-0ff4412f/index.html)

Compare **Wuli at 4, 6 and 8 steps** with Larry v4, LightX2V, FlashGen, DMAD, PAI and the original MiniMax-H3 at 50 steps. The demo includes **9 prompts, 11 configurations and 99 videos**, with synchronized playback, step-count filters, side-by-side comparisons and selectable audio.

## ComfyUI

Use the [MiniMax H3 Turbo ComfyUI node pack](https://github.com/wuli-art/MiniMax-H3-Turbo-ComfyUI) to generate video and audio with **4, 6, or 8 steps** and save them together as one MP4. The node pack supports text-to-video, frame-conditioned generation, and **8-step Ref2VA**.

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