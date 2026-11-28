# greenslime/MiniMax-H3-Turbo-Lora-ComfyUI

Источник: https://huggingface.co/greenslime/MiniMax-H3-Turbo-Lora-ComfyUI · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая: Comfy-Org/MiniMax-H3 · теги: lora,comfyui,turbo,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-06T14:46:54.000Z

---

---
license: apache-2.0
base_model: Comfy-Org/MiniMax-H3
base_model_relation: adapter
pipeline_tag: text-to-video
tags:
  - minimax-h3
  - lora
  - adapter
  - comfyui
  - pruned
  - pruned-model
  - curve-form
  - text-to-video
  - text-to-audio
  - audio-video
  - video-generation
  - audio-generation
  - synchronized-audio
  - four-step
  - 4-step
  - turbo
  - accelerated-inference
  - further-trained
  - checkpoint-500
  - safetensors
  - bfloat16
  - bf16
  - partial-conversion
---

# MiniMax-H3 Turbo 4-Step — ComfyUI Pruned-Model LoRAs

This repository contains third-party ComfyUI compatibility conversions of the original **MiniMax-H3 Turbo LoRA — 4-step audio-video generation preview** created by **[larryvrh](https://huggingface.co/larryvrh)**.

It also includes two **further-trained checkpoint-500 variants** in the same pruned-model ComfyUI format.

These LoRAs are intended for the **pruned/curve-form MiniMax-H3 checkpoint used by ComfyUI**.

**also check https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo**

## ComfyUI Test Output

<video
  src="https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/assets/MiniMax_H3_00060-audio.mp4"
  controls
  loop
  width="100%">
</video>

[Download the ComfyUI workflow](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/fl_minimax_h3_turbo_lora_example_workflow.json?download=true)

![ComfyUI workflow screenshot](https://cdn-uploads.huggingface.co/production/uploads/63473b59e5c0717e6737b872/zfVV_yKP-njnOrGinGy8K.png)

## Original Project

The original LoRA weights, training work, four-step distillation method, dual video/audio sampling implementation, documentation and source files were created and released by **larryvrh**:

**Original repository:**  
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)

I did **not** train, distill or create the original Turbo LoRA weights. This repository provides modified compatibility versions intended to allow the compatible LoRA adapters to load with ComfyUI's built-in MiniMax-H3 LoRA loader when using the pruned/curve-form model.

Full credit for the original MiniMax-H3 Turbo LoRA and its distillation method belongs to **larryvrh**.

## Included Files

| File | Description |
|---|---|
| `minimax_h3_turbo_4step_pruned_comfyui.safetensors` | Initial partial ComfyUI compatibility conversion of the original non-EMA Turbo weights for the pruned/curve-form MiniMax-H3 checkpoint. The original author describes the non-EMA weights as crisper and better at holding fast motion. |
| `minimax_h3_turbo_4step_ema_pruned_comfyui.safetensors` | Initial partial ComfyUI compatibility conversion of the original EMA Turbo weights for the pruned/curve-form MiniMax-H3 checkpoint. The original author describes the early EMA weights as smoother but softer because the EMA had not fully matured. |
| `minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors` | Further-trained checkpoint-500 non-EMA variant in the pruned-model ComfyUI format. This continues training beyond the initial preview checkpoint. |
| `minimax_h3_turbo_4step_ema_ckpt500_pruned_comfyui.safetensors` | Further-trained checkpoint-500 EMA variant in the pruned-model ComfyUI format. This contains the time-averaged weights from the further-trained checkpoint. |
| `fl_minimax_h3_turbo_lora_example_workflow.json` | First-and-last-frame ComfyUI example workflow using the Turbo LoRA, with optional attention and memory optimizations. |

The EMA and non-EMA files remain separate and contain different tensor values.

The original converted files preserve the initial preview weights, while the `ckpt500` files provide the further-trained variants.

These are partial compatibility versions of the original full-model Turbo LoRAs for use with the pruned/curve-form MiniMax-H3 architecture.

## Which File Should I Use?

For the further-trained weights, start with:

`minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors`

Use the further-trained EMA variant for comparison:

`minimax_h3_turbo_4step_ema_ckpt500_pruned_comfyui.safetensors`

The two files without `ckpt500` preserve the earlier preview versions:

- `minimax_h3_turbo_4step_pruned_comfyui.safetensors`
- `minimax_h3_turbo_4step_ema_pruned_comfyui.safetensors`

The non-EMA variants generally preserve a sharper response and stronger fast-motion behavior, while the EMA variants are intended to provide smoother time-averaged weights.

## About the Original Turbo LoRA

The original MiniMax-H3 Turbo LoRA is an early preview checkpoint designed to generate joint video and synchronized stereo audio in approximately **4 sampling steps instead of the usual approximately 20 steps**, providing roughly a **5× reduction in sampling wall-clock time**.

The original author clearly states that the initial release is an unfinished preview checkpoint from an in-progress training run:

- it is under-trained;
- the initial EMA weights have not fully matured;
- quality is not representative of a completed run;
- it is intended as an early demonstration of the direction of the project.

The additional `ckpt500` files in this repository are further-trained variants beyond that initial preview checkpoint.

Refer to the original repository for the original author's documentation, generation script, scheduler implementation and future updates:

[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)

## Why This Conversion Was Made

The original Turbo LoRA files use tensor keys such as:

`blocks.0.attn.qkv_proj.lora_A.weight`

ComfyUI's built-in MiniMax-H3 LoRA resolver expects keys under the model namespace:

`diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight`

Without that namespace, ComfyUI reports warnings such as:

`[WARNING] lora key not loaded: blocks.0.attn.out_proj.lora_A.weight`

The following namespace conversion was applied to the retained tensors:

- `blocks.*` → `diffusion_model.blocks.*`
- `token_refiner.*` → `diffusion_model.token_refiner.*`

The retained tensor values, BF16 dtype, tensor shapes, LoRA rank and A/B orientation were preserved.

## Conversion Details

Each original full-model Turbo LoRA contained:

| Component | Count |
|---|---:|
| Source tensors | 518 |
| Source LoRA A/B pairs | 259 |

Each pruned-model compatibility conversion contains:

| Component | Count |
|---|---:|
| Retained tensors | 416 |
| Retained LoRA A/B pairs | 208 |
| Pruned AdaLN tensors | 102 |
| Pruned AdaLN A/B pairs | 51 |

The pruned adapters were:

- 50 AdaLN projection pairs from `blocks.0` through `blocks.49`;
- 1 AdaLN projection pair from `final_layer.adaln_proj.linear`.

The retained adapters include:

- all main-block attention adapters;
- all main-block MLP adapters;
- all token-refiner attention adapters;
- all token-refiner MLP adapters.

The retained attention, MLP and token-refiner tensors have dimensions compatible with the pruned/curve-form MiniMax-H3 checkpoint.

## Validation

The converted LoRA structure was checked for:

- complete LoRA A/B adapter pairs;
- no orphan A or B tensors;
- retained keys using the `diffusion_model.*` namespace;
- no duplicated `diffusion_model.diffusion_model.*` prefixes;
- no incompatible `adaln_proj` tensors;
- preserved BF16 dtype;
- preserved tensor shapes;
- preserved LoRA ranks;
- separate EMA and non-EMA weights;
- no unresolved retained keys during ComfyUI loader-level validation.

The original source files were not overwritten.

The LoRAs have been tested working in ComfyUI with the following optional acceleration methods:

- **SageAttention**
- **Sol Attention**
- **Gradient**
- **Spectrum**

These acceleration methods are optional and are not included in this repository.

## Important Limitation

These files are **partial ComfyUI compatibility versions for the pruned/curve-form MiniMax-H3 checkpoint**.

The further-trained `ckpt500` variants improve upon the initial compatibility versions, 

## Usage

Place the LoRA files inside:

`ComfyUI/models/loras/`

Use them with the **pruned/curve-form MiniMax-H3 checkpoint for ComfyUI**.

Recommended further-trained version:

`minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors`

Further-trained EMA version:

`minimax_h3_turbo_4step_ema_ckpt500_pruned_comfyui.safetensors`

Suggested ComfyUI settings:

- Video sigma shift: `12`
- Audio sigma shift: `4-6`
- Steps tested working: `8–10` / `6-8` (ckpt500)
- Recommended sampler: `res_multistep`
- Suggested LoRA strength: `0.8–1.8`
- A higher LoRA strength can generally be used with fewer sampling steps
- Use the same prompt, seed, resolution, model checkpoint, sampler and workflow when comparing files

Optional accelerators confirmed working:

- **SageAttention**
- **Sol Attention**
- **Gradient**
- **Spectrum**

Do not assume broken audio is caused by the LoRA before checking the sampler, scheduler and audio sigma-shift configuration.

## Audio and Scheduler Notes

MiniMax-H3 generates video and synchronized audio as separate streams with different scheduling requirements.

The original author's standalone generation implementation uses a dual video/audio scheduling system specifically designed for ultra-low-step generation.

For this ComfyUI pruned-model version, the tested settings are:

- Video sigma shift: `12`
- Audio sigma shift: `4-6`
- Steps: `8–10` / `6-8` (ckpt500)
- Sampler: `res_multistep`

Incorrect audio scheduling may result in:

- blown-out audio;
- severe distortion;
- noise-like audio;
- unstable audio energy;
- completely broken audio.

For the original full-model settings, model files, resolution guidance, frame-count rules and dual-scheduler implementation, follow the original README:

[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)

## Attribution

Original MiniMax-H3 Turbo LoRA author:

**[larryvrh](https://huggingface.co/larryvrh)**

Original project:

**[MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)**

This repository contains third-party, partial ComfyUI compatibility versions for the pruned/curve-form MiniMax-H3 checkpoint, including further-trained checkpoint-500 variants.

I do not claim authorship of:

- the original Turbo LoRA weights;
- the original LoRA training;
- the original distillation method;
- the original four-step sampling method;
- the original dual video/audio scheduler;
- the original `generate.py`;
- the original documentation;
- the MiniMax-H3 base model;
- the pruned/curve-form MiniMax-H3 checkpoint.

Please retain attribution to **larryvrh** and comply with the licences and terms of the original LoRA and MiniMax-H3 base model when using or redistributing these files.
