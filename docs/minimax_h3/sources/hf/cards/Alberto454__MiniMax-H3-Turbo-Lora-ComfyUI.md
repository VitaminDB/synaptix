# Alberto454/MiniMax-H3-Turbo-Lora-ComfyUI

Источник: https://huggingface.co/Alberto454/MiniMax-H3-Turbo-Lora-ComfyUI · скачано 2026-11-28
Загрузок: 0 · лайков: 1 · базовая: Comfy-Org/MiniMax-H3 · теги: lora,comfyui,turbo,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-11T12:57:20.000Z

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
  - few-step
  - four-step
  - 4-step
  - turbo
  - accelerated-inference
  - further-trained
  - checkpoint-500
  - checkpoint-600
  - checkpoint-850
  - v4
  - ema
  - euler
  - beta-scheduler
  - safetensors
  - bfloat16
  - bf16
  - partial-conversion
---

# MiniMax-H3 Turbo LoRA — ComfyUI Pruned-Model Compatibility Conversions

This repository contains third-party **ComfyUI compatibility conversions** of the original **MiniMax-H3 Turbo LoRA** created by **[larryvrh](https://huggingface.co/larryvrh)**.

The Turbo LoRA enables few-step MiniMax-H3 audio-video generation, producing joint video and synchronized audio using substantially fewer sampling steps than the standard MiniMax-H3 workflow.

This repository includes compatibility conversions of the initial Turbo weights as well as later **checkpoint-500**, **checkpoint-850**, and **v4 step-600** releases.

These LoRAs are intended for the **pruned/curve-form MiniMax-H3 checkpoint used by ComfyUI**.

> **Recommended checkpoint:** `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`
>
> **Recommended settings:** **8 steps · Euler sampler · Beta scheduler**

---

# Workflow Examples

## New Workflow Output

This is the output from the current recommended workflow using the newer **v4 step-600 EMA** checkpoint and recommended sampling configuration.

<video
src="https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/assets/MiniMax_H3_audio.mp4"
controls
loop
width="100%"> </video>

### New Recommended Workflow

[Download the current ComfyUI workflow](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/fl_minimax_h3_turbo_lora_example_workflow.json?download=true)

The current workflow is:

`fl_minimax_h3_turbo_lora_example_workflow.json`

Recommended starting configuration:

```text
Checkpoint: minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors
Steps: 6
Sampler: euler
Scheduler: beta
LoRA strength: 1.0
```

---

## Recommended Setup

For the best overall results in this repository, start with:

`minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`

### Recommended ComfyUI settings

| Setting           | Recommended                                                  |
| ----------------- | ------------------------------------------------------------ |
| LoRA              | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors` |
| Steps             | `6-8`                                                          |
| Sampler           | `euler`                                                      |
| Scheduler         | `beta`                                                       |                                                      |
| LoRA strength     | `1.0` starting point                                         |

The **v4 step-600 EMA** checkpoint is currently the recommended general-purpose version in this repository.

For most generations, **8 steps with Euler and the Beta scheduler** is the recommended starting configuration for these ComfyUI compatibility versions.

---

## Old Workflow Output

The following video and workflow are retained as an example of the previous Turbo LoRA configuration.

<video
src="https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/assets/MiniMax_H3_00060-audio.mp4"
controls
loop
width="100%"> </video>

### Old Workflow

[Download the old ComfyUI workflow](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/fl_minimax_h3_turbo_lora_example_workflow_old.json?download=true)

The previous workflow has been renamed to:

`fl_minimax_h3_turbo_lora_example_workflow_old.json`

![Old ComfyUI workflow screenshot](https://cdn-uploads.huggingface.co/production/uploads/63473b59e5c0717e6737b872/zfVV_yKP-njnOrGinGy8K.png)

---

## Original Project

The original LoRA weights, training work, distillation method, audio/video sampling implementation, documentation and source files were created and released by **larryvrh**.

**Original repository:**
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)

I did **not** train, distill or create the original Turbo LoRA weights.

This repository provides modified compatibility versions intended to allow the compatible LoRA adapters to load with ComfyUI's built-in MiniMax-H3 LoRA loader when using the pruned/curve-form model.

Full credit for the original MiniMax-H3 Turbo LoRA, training and distillation method belongs to **larryvrh**.

---

## Included Files

| File                                                            | Description                                                                                       |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`    | **Recommended.** Pruned-model ComfyUI compatibility conversion of the v4 step-600 EMA checkpoint. |
| `minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors`        | Non-EMA version of the v4 step-600 checkpoint for comparison.                                     |
| `minimax_h3_turbo_4step_ema_ckpt850_pruned_comfyui.safetensors` | EMA compatibility conversion of the checkpoint-850 weights.                                       |
| `minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors`     | Non-EMA compatibility conversion of the checkpoint-850 weights.                                   |
| `minimax_h3_turbo_4step_ema_ckpt500_pruned_comfyui.safetensors` | Further-trained checkpoint-500 EMA variant in the pruned-model ComfyUI format.                    |
| `minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors`     | Further-trained checkpoint-500 non-EMA variant in the pruned-model ComfyUI format.                |
| `minimax_h3_turbo_4step_ema_pruned_comfyui.safetensors`         | Compatibility conversion of the initial EMA Turbo release.                                        |
| `minimax_h3_turbo_4step_pruned_comfyui.safetensors`             | Compatibility conversion of the initial non-EMA Turbo release.                                    |
| `fl_minimax_h3_turbo_lora_example_workflow.json`                | **Current recommended ComfyUI example workflow.**                                                 |
| `fl_minimax_h3_turbo_lora_example_workflow_old.json`            | Previous example workflow, retained for reference and comparison.                                 |
| `assets/MiniMax_H3_audio.mp4`                                   | Output video from the **new recommended workflow**.                                               |
| `assets/MiniMax_H3_00060-audio.mp4`                             | Output video from the previous workflow.                                                          |

The EMA and non-EMA files remain separate and contain different tensor values.

The files without checkpoint identifiers preserve the earlier preview weights.

The `ckpt500` and `ckpt850` files belong to the older training line, while the `v4_step600` files use the newer **v4 training recipe**.

These files are partial compatibility versions of the original full-model Turbo LoRAs for use with the pruned/curve-form MiniMax-H3 architecture.

---

## Which File Should I Use?

### Recommended — v4 Step-600 EMA

Start with:

`minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`

Recommended configuration:

* **Steps:** `6-8`
* **Sampler:** `euler`
* **Scheduler:** `beta`
* **LoRA strength:** start at `1.0`

This is the **recommended checkpoint for most generations**.

### v4 Step-600 non-EMA

For comparison:

`minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors`

The non-EMA weights are provided primarily for testing and comparison against the recommended EMA checkpoint.

### Checkpoint-850

EMA:

`minimax_h3_turbo_4step_ema_ckpt850_pruned_comfyui.safetensors`

Non-EMA:

`minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors`

These are retained for comparison with the newer v4 training line.

### Checkpoint-500

EMA:

`minimax_h3_turbo_4step_ema_ckpt500_pruned_comfyui.safetensors`

Non-EMA:

`minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors`

These preserve the earlier further-trained checkpoint-500 release.

### Initial Preview Weights

The following files preserve the initial preview versions:

* `minimax_h3_turbo_4step_pruned_comfyui.safetensors`
* `minimax_h3_turbo_4step_ema_pruned_comfyui.safetensors`

They are kept primarily for comparison and reproducibility.

---

## Recommended Workflow

The current recommended workflow file is:

`fl_minimax_h3_turbo_lora_example_workflow.json`

Download:

[fl_minimax_h3_turbo_lora_example_workflow.json](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/fl_minimax_h3_turbo_lora_example_workflow.json?download=true)

It is intended to be used with:

`minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`

Recommended configuration:

```text
Steps: 8
Sampler: euler
Scheduler: beta
LoRA strength: 1.0
Video sigma shift: 12
Audio sigma shift: 4-6
```

The corresponding example output is:

`assets/MiniMax_H3_audio.mp4`

[View / download MiniMax_H3_audio.mp4](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/assets/MiniMax_H3_audio.mp4)

---

## Previous Workflow

The previous example workflow is retained as:

`fl_minimax_h3_turbo_lora_example_workflow_old.json`

Download:

[fl_minimax_h3_turbo_lora_example_workflow_old.json](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/fl_minimax_h3_turbo_lora_example_workflow_old.json?download=true)

Its corresponding example output is:

`assets/MiniMax_H3_00060-audio.mp4`

[View / download MiniMax_H3_00060-audio.mp4](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/assets/MiniMax_H3_00060-audio.mp4)

---

## About the Original Turbo LoRA

The MiniMax-H3 Turbo LoRA is designed to generate joint video and synchronized stereo audio using dramatically fewer sampling steps than the standard MiniMax-H3 generation process.

The original project demonstrated generation using substantially fewer sampling steps than the normal MiniMax-H3 workflow, providing a large reduction in sampling time.

The newer Turbo checkpoints continue to improve upon the initial preview release.

Refer to the original repository for the author's current documentation, training updates, standalone generation implementation and sampling recommendations:

[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)

---

## Why This Conversion Was Made

The original Turbo LoRA files use tensor keys such as:

`blocks.0.attn.qkv_proj.lora_A.weight`

ComfyUI's built-in MiniMax-H3 LoRA resolver expects keys under the model namespace:

`diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight`

Without that namespace, ComfyUI may report warnings such as:

`[WARNING] lora key not loaded: blocks.0.attn.out_proj.lora_A.weight`

The following namespace conversion was applied to the retained tensors:

* `blocks.*` → `diffusion_model.blocks.*`
* `token_refiner.*` → `diffusion_model.token_refiner.*`

The retained tensor values, BF16 dtype, tensor shapes, LoRA rank and A/B orientation were preserved.

---

## Conversion Details

Each original full-model Turbo LoRA contained:

| Component             | Count |
| --------------------- | ----: |
| Source tensors        |   518 |
| Source LoRA A/B pairs |   259 |

Each pruned-model compatibility conversion contains:

| Component               | Count |
| ----------------------- | ----: |
| Retained tensors        |   416 |
| Retained LoRA A/B pairs |   208 |
| Pruned AdaLN tensors    |   102 |
| Pruned AdaLN A/B pairs  |    51 |

The pruned adapters were:

* 50 AdaLN projection pairs from `blocks.0` through `blocks.49`;
* 1 AdaLN projection pair from `final_layer.adaln_proj.linear`.

The retained adapters include:

* all main-block attention adapters;
* all main-block MLP adapters;
* all token-refiner attention adapters;
* all token-refiner MLP adapters.

The retained attention, MLP and token-refiner tensors have dimensions compatible with the pruned/curve-form MiniMax-H3 checkpoint.

---

## Validation

The converted LoRA structure was checked for:

* complete LoRA A/B adapter pairs;
* no orphan A or B tensors;
* retained keys using the `diffusion_model.*` namespace;
* no duplicated `diffusion_model.diffusion_model.*` prefixes;
* no incompatible `adaln_proj` tensors;
* preserved BF16 dtype;
* preserved tensor shapes;
* preserved LoRA ranks;
* separate EMA and non-EMA weights;
* no unresolved retained keys during ComfyUI loader-level validation.

The original source files were not overwritten.

---

## Optional Acceleration

The LoRAs have been tested working in ComfyUI with optional acceleration methods including:

* **SageAttention**
* **Sol Attention**
* **Gradient**
* **Spectrum**

These acceleration methods are optional and are not included in this repository.

They may also be combined with the Turbo LoRA depending on your ComfyUI workflow and available custom nodes.

---

## Important Limitation

These files are **partial ComfyUI compatibility conversions intended specifically for the pruned/curve-form MiniMax-H3 architecture**.

The original full-model Turbo LoRAs contain AdaLN adapters that do not directly match the corresponding dimensions of the pruned/curve-form MiniMax-H3 checkpoint.

Those incompatible AdaLN tensors are therefore excluded from these compatibility versions.

The retained attention, MLP and token-refiner LoRA tensors remain compatible with the pruned architecture.

Because these are partial compatibility conversions, they should not be assumed to be numerically identical in behaviour to using the original full Turbo LoRA with its intended full MiniMax-H3 base.

---

## Usage

Place the LoRA files inside:

`ComfyUI/models/loras/`

Use them with the **pruned/curve-form MiniMax-H3 checkpoint for ComfyUI**.

### Recommended LoRA

`minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`

### Recommended settings

* **Steps:** `8`
* **Sampler:** `euler`
* **Scheduler:** `beta`
* **Video sigma shift:** `12`
* **Audio sigma shift:** `4-6`
* **LoRA strength:** `1.0` recommended starting point

The recommended workflow for these settings is:

[Download `fl_minimax_h3_turbo_lora_example_workflow.json`](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/fl_minimax_h3_turbo_lora_example_workflow.json?download=true)

The corresponding example output is:

[View `assets/MiniMax_H3_audio.mp4`](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/assets/MiniMax_H3_audio.mp4)

If necessary, LoRA strength can be adjusted slightly for individual generations.

A reasonable starting range is approximately:

`0.8-1.2`

When comparing checkpoints, use the same:

* prompt;
* seed;
* resolution;
* frame count;
* base model;
* sampler;
* scheduler;
* LoRA strength;
* workflow.

This gives a meaningful comparison between checkpoint versions.

---

## Sampling Step Guidance

The Turbo LoRA can operate at extremely low step counts, but image quality and motion consistency generally improve when moving beyond the minimum.

### 4 steps

Maximum speed.

Useful for testing and rapid previews.

### 6 steps

Good speed/quality compromise.

### 8 steps

**Recommended for the v4 step-600 EMA compatibility conversion.**

Use:

```text
8 steps
Euler sampler
Beta scheduler
```

This is the recommended configuration used by the current example workflow.

---

## Audio and Scheduler Notes

MiniMax-H3 generates video and synchronized audio as separate streams with different scheduling requirements.

Incorrect audio scheduling can result in:

* blown-out audio;
* severe distortion;
* noise-like audio;
* unstable audio energy;
* completely broken audio.

For the current recommended ComfyUI compatibility workflow in this repository:

```text
LoRA: minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors
Steps: 8
Sampler: euler
Scheduler: beta
Video sigma shift: 12
Audio sigma shift: 4-6
LoRA strength: 1.0
```

Do not automatically assume broken audio is caused by the LoRA.

Check:

* sampler configuration;
* scheduler configuration;
* audio sigma shift;
* video sigma shift;
* ComfyUI version;
* MiniMax-H3 sampling implementation;
* custom sampler nodes.

---

## Original Turbo Custom Node

The original Turbo author also provides a dedicated ComfyUI custom node:

[GitHub — Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)

The node includes a MiniMax-H3 Turbo LoRA loader and dedicated sampling functionality designed for MiniMax-H3's video/audio scheduling behaviour.

A workflow is also included in the custom-node repository.

The custom node evolves alongside the upstream Turbo weights, so users choosing that implementation should follow the settings documented in its current README.

The settings documented in this repository refer to the tested **pruned-model compatibility workflow**, where the recommended starting point is:

**v4 step-600 EMA · 8 steps · Euler · Beta scheduler**

---

## Compatibility

These LoRAs are intended for the **pruned/curve-form MiniMax-H3 checkpoint used by ComfyUI**.

The compatibility conversions are particularly useful when the original full-model LoRA cannot be loaded directly because of incompatible AdaLN dimensions.

Optional acceleration methods confirmed working with these conversions include:

* SageAttention
* Sol Attention
* Gradient
* Spectrum

Accelerators can generally be stacked where the corresponding ComfyUI implementations support it.

---

## Naming Notes

The upstream project contains multiple generations of Turbo training.

### Older naming

Files such as:

`minimax_h3_turbo_4step_ema_ckpt500.safetensors`

and:

`minimax_h3_turbo_4step_ema_ckpt850.safetensors`

belong to the older training line.

In these filenames, `4step` refers to the few-step Turbo sampling target.

### Newer v4 naming

Files such as:

`minimax_h3_turbo_v4_step600_ema.safetensors`

use the newer naming convention.

Here:

* `v4` refers to the newer training recipe;
* `step600` refers to training step 600;
* `ema` refers to the exponential-moving-average weights.

The corresponding pruned ComfyUI compatibility conversion in this repository is:

`minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`

This is the **recommended checkpoint**.

---

## Repository Workflow Files

### Current workflow

`fl_minimax_h3_turbo_lora_example_workflow.json`

[Download current workflow](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/fl_minimax_h3_turbo_lora_example_workflow.json?download=true)

Current workflow output:

`assets/MiniMax_H3_audio.mp4`

[View current workflow output](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/assets/MiniMax_H3_audio.mp4)

### Previous workflow

`fl_minimax_h3_turbo_lora_example_workflow_old.json`

[Download previous workflow](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/fl_minimax_h3_turbo_lora_example_workflow_old.json?download=true)

Previous workflow output:

`assets/MiniMax_H3_00060-audio.mp4`

[View previous workflow output](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/assets/MiniMax_H3_00060-audio.mp4)

---

## Attribution

Original MiniMax-H3 Turbo LoRA author:

**[larryvrh](https://huggingface.co/larryvrh)**

Original project:

**[MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)**

Original ComfyUI Turbo custom node:

**[ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)**

This repository contains third-party, partial ComfyUI compatibility versions for the pruned/curve-form MiniMax-H3 checkpoint, including:

* initial Turbo preview weights;
* checkpoint-500 variants;
* checkpoint-850 variants;
* v4 step-600 variants.

I do not claim authorship of:

* the original Turbo LoRA weights;
* the original LoRA training;
* the original distillation method;
* the original few-step sampling method;
* the original dual video/audio scheduling implementation;
* the original generation scripts;
* the original documentation;
* the MiniMax-H3 base model;
* the pruned/curve-form MiniMax-H3 checkpoint.

The modifications in this repository are compatibility conversions intended to make the compatible portions of the upstream Turbo LoRAs usable with the pruned/curve-form MiniMax-H3 architecture in ComfyUI.

Please retain attribution to **larryvrh** and comply with the licences and terms of the original LoRA and MiniMax-H3 base model when using or redistributing these files.
