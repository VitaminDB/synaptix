# javawock7618/comfy-MiniMax-H3-workflows

Источник: https://huggingface.co/javawock7618/comfy-MiniMax-H3-workflows · скачано 2026-11-28
Загрузок: 0 · лайков: 178 · базовая:  · теги: comfyui,controlnet,controlnet-union,pika-h3-flex-turbo-lora,turbo-lora,quantization · обновлено: 2026-11-26T11:07:16.000Z

---

---
pipeline_tag: image-to-video
tags:
- comfyui
- llama-cpp
- minimax-h3
- latent-upscaler
- first-last-frame
- flf
- bridge
- reference-to-video
- r2v
- controlnet
- controlnet-union
- openpose
- canny
- depth
- v2v
- text-to-speech
- tts
- video-generation
- int8
- block-sparse-attention
- veda-sparse
- spectrum
- dmad
- pika-h3-flex-turbo-lora
- turbo-lora
- extend
- switchboard
- group-bypass
- workflow
- optimization
- quantization
---

# MiniMax-H3 INT8 FLF, R2V, Bridge & TTS Workflows Collection

Optimized **ComfyUI workflows** for MiniMax-H3, featuring INT8 inference, native Block Sparse Attention, Veda Sparse Attention, Spectrum, DMAD, Pika H3 Flex Turbo LoRA, native Extend, Fun ControlNet Union V2V control, Face Refine, TTS, Latent Upscaler, Bridge, and Group Bypass, along with `llama.cpp` and `comfyui-llamacpp` integration references.

**Veda Sparse Attention is supported by both the I2V and R2V workflows** as an optional attention optimization.

These workflows are designed to provide flexible local video and audio generation, with optional acceleration and refinement stages that can be enabled or bypassed as needed.

## Workflows

| Workflow | File | Description |
|---|---|---|
| First-Last-Frame (FLF) | `MiniMax_int8_I2V-javanoYYMM.x.json` | FLF/I2V video generation with Extend, ControlNet Union V2V, Latent Upscaler, and optional Veda Sparse Attention. |
| Reference-to-Video (R2V) | `MiniMax_int8_R2V-javanoYYMM.x.json` | Reference-guided video generation with optional Reference LoRA, ControlNet Union V2V, Extend, Latent Upscaler, and optional Veda Sparse Attention. |
| Bridge | `MiniMax_int8_Bridge-javanoYYMM.x.json` | Generates a transition between Video A and Video B. |
| Face Refine | `MiniMax_int8_FR-javanoYYMM.x.json` | Face refinement for generated video. |
| Text-to-Speech (TTS) | `MiniMax_int8_TTS-javanoYYMM.x.json` | Audio-only speech generation. |

**Base model:** https://huggingface.co/Comfy-Org/MiniMax-H3

## Features

### INT8 Inference

The workflows use INT8 MiniMax-H3 model weights to reduce memory requirements and improve local inference efficiency.

- Reduced model memory requirements compared with higher-precision weights.
- Improved memory efficiency for local inference.
- Suitable for systems with limited VRAM relative to full-precision configurations.

Actual generation speed and output quality depend on the GPU, attention backend, sampling configuration, and enabled optimizations.

### Native Block Sparse Attention

The FLF / I2V and R2V workflows use ComfyUI's native **Block Sparse Attention** node for optimized attention processing.

This is separate from Veda Sparse Attention, which is an additional optional attention optimization.

Native Block Sparse Attention does not require an additional custom node.

### Spectrum

**Repository:** https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3

Spectrum provides additional MiniMax-H3 sampling optimization.

The workflows support Spectrum alongside other compatible acceleration methods. Sampling steps, schedules, and adapter/LoRA strengths may require adjustment to balance speed and quality.

### DMAD

**Repository:** https://github.com/Yzmblog/DMAD

**Hugging Face:** https://huggingface.co/ZhengmingYu/DMAD

DMAD (Distribution Matching as Adversarial Distillation) provides step-distilled LoRA adapters for accelerated MiniMax-H3 generation (including 4-step and 8-step configurations).

Use the corresponding sampler and scheduler/sigma settings provided by the workflow or DMAD adapter configuration, and adjust steps and strength to balance speed and audio-visual quality.

### Pika H3 Flex Turbo LoRA

**Model:** https://huggingface.co/Pika-Labs/Pika-H3-Flex-Turbo-LoRA

Pika H3 Flex Turbo LoRA fuses Pika H3 Flex motion/style adapters with few-step distillation for MiniMax-H3 into single LoRA files (including 2-step draft, 4-step fast, 8-step HD, and rank-32 `lite` variants for FL2VA / Ref2VA).

- Place the matching `.safetensors` file in `ComfyUI/models/loras` and load it via `LoraLoaderModelOnly` (typically at strength `1.0`).
- Match the step count (`2`, `4`, or `8`) and recommended sampler/scheduler settings (`CFG 1.0`, `simple` scheduler, and appropriate flow/audio shift values) for the selected variant.
- Use 2-step draft mode for fast iteration and 4-step or 8-step variants when higher visual and synchronized audio fidelity is needed; avoid stacking conflicting acceleration LoRAs.

### Veda Sparse Attention — I2V and R2V

**Model:** https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview

**ComfyUI implementation:** https://github.com/veda-sparse/Veda-on-ComfyUI

Veda Sparse Attention is an optional attention optimization for MiniMax-H3 video generation. It is available in both the I2V and R2V workflows.

It uses a learned predictor to identify relevant visual attention tiles and reduce attention computation.

- Available for both I2V and R2V workflows.
- Requires the Veda-on-ComfyUI custom node and a compatible predictor checkpoint.
- Separate from the native ComfyUI Block Sparse Attention implementation.
- Does not replace the MiniMax-H3 base model or any optional R2V Reference LoRA.
- Optional; the workflows can be used without enabling Veda Sparse Attention.

The published R2VA Preview is an experimental configuration. Compatibility with other model, adapter/LoRA, and sampling configurations should not be assumed.

Follow the official repository instructions for installation, model placement, and supported inference backends.

### Native Extend

MiniMax-H3 supports native video extension for continuing existing video content.

The workflows use native Extend functionality rather than requiring the former Motion Context custom node.

### Fun ControlNet Union — V2V Control

**Model repository:** https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/controlnet

Integrated into the I2V and R2V workflows.

Supported control types:

- OpenPose
- Canny
- Depth

Available models:

- `minimax_h3_fun_controlnet_union_pruned_bf16.safetensors`
- `minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors`

Use the model variant compatible with your workflow and inference configuration.

### Latent Upscaler

**Custom node:** https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler

**Model:** https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler

Integrated into I2V and R2V.

Generation pipeline:

**Low-resolution generation → Latent Upscaling → High-resolution decoding**

The workflow uses a low-resolution generation stage followed by latent-space upscaling and refinement.

### Bridge

Generates a transition between two existing clips:

**Video A → Generated Bridge → Video B**

Designed to create intermediate footage connecting two video segments.

### Face Refine

**Custom node:** https://github.com/Carasibana/ComfyUI-H3-FaceRefine

Provides automatic face detection and refinement for generated video.

### Text-to-Speech

MiniMax-H3 can generate audio without running the video generation path.

The TTS workflow is intended for audio-only speech generation.

### Group Bypass

**Custom node:** https://github.com/Oratorian/ComfyUI-Switchboard

Used to enable or bypass optional workflow groups, including optional processing and optimization stages.

### `llama.cpp` & ComfyUI Integration

- **`llama.cpp` Releases:** https://github.com/ggml-org/llama.cpp/releases
- **ComfyUI `llama.cpp` Integration:** https://github.com/Setmaster/comfyui-llamacpp

For companion language/text tasks alongside ComfyUI pipelines, use `llama.cpp` releases and `comfyui-llamacpp` custom nodes rather than Ollama.

## Workflow Details

### First-Last-Frame (FLF)

`MiniMax_int8_I2V-javanoYYMM.x.json`

- First + Last Frame conditioning
- Fun ControlNet Union / V2V
- OpenPose / Canny / Depth control
- Native Extend
- Latent Upscaler
- Native Block Sparse Attention
- INT8 inference
- Spectrum
- DMAD
- Pika H3 Flex Turbo LoRA
- Optional Veda Sparse Attention
- Group Bypass

Veda Sparse Attention is optional and can be enabled when the compatible Veda implementation and predictor are installed.

### Reference-to-Video (R2V)

`MiniMax_int8_R2V-javanoYYMM.x.json`

- Reference image conditioning
- Optional Reference LoRA
- Fun ControlNet Union / V2V
- OpenPose / Canny / Depth control
- Native Extend
- Latent Upscaler
- Native Block Sparse Attention
- INT8 inference
- Spectrum
- DMAD
- Pika H3 Flex Turbo LoRA
- Optional Veda Sparse Attention
- Group Bypass

Optional Reference LoRA:

`minimax_h3_ref_lora_rank_256_bf16.safetensors`

Download from:
https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/loras

Veda Sparse Attention is an optional acceleration component for the R2V workflow.

### Bridge

`MiniMax_int8_Bridge-javanoYYMM.x.json`

**Video A → Generated Bridge → Video B**

Creates a generated transition between two existing video clips.

### Face Refine

`MiniMax_int8_FR-javanoYYMM.x.json`

Uses ComfyUI-H3-FaceRefine for automatic face refinement.

### Text-to-Speech

`MiniMax_int8_TTS-javanoYYMM.x.json`

Audio-only MiniMax-H3 speech generation.

## Workflow Compatibility

| Feature | FLF / I2V | R2V |
|---|---|---|
| INT8 inference | Yes | Yes |
| Native Block Sparse Attention | Yes | Yes |
| Spectrum | Yes | Yes |
| DMAD | Yes | Yes |
| Pika H3 Flex Turbo LoRA | Yes | Yes |
| Native Extend | Yes | Yes |
| Fun ControlNet Union | Yes | Yes |
| Latent Upscaler | Yes | Yes |
| Veda Sparse Attention | Optional | Optional |
| Reference LoRA | No | Optional |

## Required Custom Nodes

Install the custom nodes required by the workflows you intend to use.

| Custom node | Purpose |
|---|---|
| [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) | Spectrum sampling optimization |
| [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine) | Face refinement |
| [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | Latent upscaling |
| [ComfyUI-Switchboard](https://github.com/Oratorian/ComfyUI-Switchboard) | Group Bypass |
| [Veda-on-ComfyUI](https://github.com/veda-sparse/Veda-on-ComfyUI) | Optional Veda Sparse Attention for I2V and R2V |
| [comfyui-llamacpp](https://github.com/Setmaster/comfyui-llamacpp) | ComfyUI integration for `llama.cpp` |

**Native Block Sparse Attention** is provided by ComfyUI and requires no additional custom node.

Veda-on-ComfyUI is required only when enabling Veda Sparse Attention. It is not required for standard I2V / FLF, R2V, Bridge, Face Refine, or TTS generation.

## Required Models

### Main Model

https://huggingface.co/Comfy-Org/MiniMax-H3

Use the model variant compatible with the workflow's INT8 loader.

### R2V Reference LoRA (Optional)

`minimax_h3_ref_lora_rank_256_bf16.safetensors`

Download from:
https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/loras
