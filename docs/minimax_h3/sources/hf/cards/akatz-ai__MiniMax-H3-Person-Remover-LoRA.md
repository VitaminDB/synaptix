# akatz-ai/MiniMax-H3-Person-Remover-LoRA

Источник: https://huggingface.co/akatz-ai/MiniMax-H3-Person-Remover-LoRA · скачано 2026-11-28
Загрузок: 2916 · лайков: 101 · базовая: Comfy-Org/MiniMax-H3 · теги: lora,comfyui,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-10-17T23:41:46.000Z

---

---
language:
- en
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model: Comfy-Org/MiniMax-H3
base_model_relation: adapter
pipeline_tag: video-to-video
library_name: diffusion-single-file
datasets:
- akatz-ai/H3-Person-Remover-v1
tags:
- minimax-h3
- ref2va
- lora
- person-removal
- video-editing
- comfyui
- ai-toolkit
---

# MiniMax H3 Person Remover LoRA V1

[![Follow @akatz_ai on X](https://img.shields.io/badge/Follow-%40akatz__ai-111111?style=flat-square&logo=x&logoColor=white)](https://x.com/akatz_ai)

![Person Remover: original video, green-masked person, and reconstructed background](assets/person-remover-showcase.gif)

**Original → Green mask → Person removed.** Eight selected examples, shown with their full frames and original timing. [Watch or download the 1080p comparison with original audio](assets/showcase-1080p-with-audio.mp4).

Remove a selected person from a video and reconstruct the background with MiniMax H3 Ref2VA. The accompanying ComfyUI workflow tracks the person with SAM 3.1, fills the mask green, and generates the replacement background in overlapping windows. Supply the original video and a clean version of its first frame.

**[Download the LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Person-Remover-LoRA/resolve/main/H3-Person-Remover-V1.safetensors?download=true)** · **[Download the ComfyUI workflow](examples/H3-Person-Removal-Relay-V2.json)** · **[Training dataset](https://huggingface.co/datasets/akatz-ai/H3-Person-Remover-v1)**

This is an experimental adapter trained by Akatz Labs. The showcase contains selected successful results, not a benchmark of all inputs. Background details, camera movement, shadows, and hard cuts can still change.

## Quick start

1. Use a current ComfyUI with native MiniMax H3 and SAM 3.1 support. Install [H3 Relay 2.0](https://github.com/akatz-ai/h3-relay) for the native-loop toolkit workflow.
2. Put `H3-Person-Remover-V1.safetensors` in `ComfyUI/models/loras/`.
3. Drag [examples/H3-Person-Removal-Relay-V2.json](examples/H3-Person-Removal-Relay-V2.json) into ComfyUI. The workflow includes model links, folder locations, and notes beside the relevant nodes.
4. Load a **24 fps** video with width and height divisible by **32**. Start with a short, continuous shot of roughly five seconds.
5. Make a **clean first frame** in an image editor or image-editing model: remove the person while retaining the original size and framing. Load it in the green Load Image node.
6. Describe the person in the green CLIP Text Encode node, such as `man in gray shirt`. Check the saved green-mask preview before running the full removal.
7. Run the workflow. Inspect the window previews and reroll a weak section as needed. See [the workflow guide](examples/README.md) for preview, reroll, and audio instructions.

The Relay 2.0 workflow starts with no saved window locks or cache references. It uses native ComfyUI sampling, reference conditioning, SAM masks, loops and video saving, plus reusable Relay timeline and review nodes. The LoRA weights and removal schedule are unchanged. The [previous workflow](examples/Person-Remover-Window-Reroll-V1.json) remains available.

## Required models

Base models are separate downloads. Select the matching filenames in the workflow loaders.

| Model | ComfyUI folder |
| --- | --- |
| [H3 Ref2VA pruned INT8 ConvRot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | `models/diffusion_models/` |
| [Qwen3VL 32B NVFP4 AWQ text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `models/text_encoders/` |
| [H3 video VAE FP16](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors) | `models/vae/` |
| [H3 audio VAE FP32](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `models/vae/` |
| [SAM 3.1 multiplex FP16](https://huggingface.co/Comfy-Org/sam3.1/blob/main/checkpoints/sam3.1_multiplex_fp16.safetensors), including its text encoder | `models/checkpoints/` |
| [Person Remover V1 LoRA](H3-Person-Remover-V1.safetensors) | `models/loras/` |

The Relay 2.0 workflow was tested on RTX 4090 with ComfyUI `79be670e2d9be63e238785af307369d2b9039ed1`, frontend 1.53.6 and PyTorch 2.11.0 + CUDA 12.8. ComfyUI 0.37.0 is the minimum schema version. Full GPU residency exceeded 24 GB; normal offloading and the included memory policies completed the unchanged render settings. See the workflow guide for the tested configuration.

## Starting settings

| Setting | Value |
| --- | --- |
| LoRA strength | 1.0 |
| Window length | 22 frames |
| Sampling steps | 12 |
| Sampler / scheduler | `er_sde` / `simple` |
| CFG | 1 |
| Video / audio sigma shift | 12 / 3 |
| Seed | 904234 |
| Mask expansion | 5 pixels |
| Frame rate | 24 fps |

No Turbo or VFX LoRA is required. The workflow uses H3 frame lengths `17n + 5`: 22, 39, 56, 73, and so on. A longer window uses more memory and is not necessarily better. Start with 22.

Each continuation uses the generated boundary frame as its next reference and carries 18 frames of generated video/audio history. The relay removes overlap and trims the final result to the source frame count. The adapter itself does not segment people or create the initial clean reference; SAM and the external first-frame edit provide those inputs.

Default removal prompt:

> Remove the green-masked person and reconstruct the background. Preserve the rest of the video, including its camera motion and frame timing.

The workflow's clean output is silent by default. Connect the original audio from Get Video Components to the final Create Video node to retain the soundtrack. The showcase MP4 uses this original-audio approach in postprocessing; it does not demonstrate generated audio fidelity or voice removal.

## Training record

V1 was trained for **2,000 optimizer steps**, including regularization updates.

| Setting | Recorded value |
| --- | --- |
| Architecture | `minimax_h3_ref2va`, pruned Ref2VA |
| Total optimizer updates | 2,000, including regularization updates |
| Rank / alpha | 16 / 16; exclude `adaln_proj` |
| Tensor format | BF16, 416 tensors across 208 adapter targets |
| Training base | Pruned BF16 Ref2VA, with the frozen v2 training assistant |
| Optimizer / learning rate | AdamW8bit / 5e-5 |
| Batch / accumulation | 1 / 1 |
| Text encoder | NVFP4 Qwen3VL |
| Task / regularization resolution budget | 1152 / 256, using aspect-ratio buckets |
| Task frame lengths | 5, 56, 73, 90, 124 at 24 fps |
| Regularization length | 107 frames at 24 fps, with audio |
| Runtime | AI Toolkit 0.13.23 with the recorded aligned-video-guide patch |

The first 250 updates used static five-frame pairs and preservation regularization. Training then continued to 2,000 steps with balanced task-duration scheduling and resumed optimizer state. The mixed phase's available training pool contains 128 static pairs, 80 moving-mask excerpts from 20 scenes, and six preservation clips. Related excerpts are not independent scenes, and pool size is not update frequency.

The [dataset card](https://huggingface.co/datasets/akatz-ai/H3-Person-Remover-v1) explains the construction, split, and review status. [training/](training/) contains the recorded configurations, schedules, model hashes, and runtime patch. The base model and frozen assistant are not merged into or included in this LoRA. Training used BF16 base weights; the workflow's INT8 base is an inference choice.

- File size: **155,110,584 bytes**.
- SHA-256: `b01dc9fe888c2f4e455a8eb3b4878dfd4038ee0e2320a7b186c7935e65c93ba5`.

## Limitations

- H3 regenerates the full scene. Unmasked areas are not locked to their original pixels, even though training pairs preserve pixels outside their masks.
- A missed hand, hair edge, shadow, reflection, or occluded body part can remain. A larger mask asks the model to invent more background.
- A poor clean first frame can propagate through later windows. Strong camera motion and hard cuts can break continuity.
- Rerolling a window changes its continuation. Inspect the rebuilt suffix as well as the rerolled window.
- Technical checks and successful training do not equal final human approval of every dataset example. The recorded final dataset-acceptance flag remains false.

## License and attribution

Powered by MiniMax H3. This is an Akatz Labs modification, not an official MiniMax release. The model is distributed under the [MiniMax H3 Community License Agreement](LICENSE); see [NOTICE](NOTICE). Its territorial, use, redistribution, and commercial provisions apply. The standard territorial grant excludes the US, EU, UK, and Republic of Korea and describes separate authorization. This repository does not expand those terms.

[H3 Relay](https://github.com/akatz-ai/h3-relay) is a separate GPL-3.0 project. Its code license does not relicense these model weights. Dataset component terms are documented separately in the dataset repository.
