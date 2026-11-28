# drbaph/MiniMax-H3-Turbo-Lora-ComfyUI

Источник: https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI · скачано 2026-11-28
Загрузок: 250564 · лайков: 551 · базовая: Comfy-Org/MiniMax-H3 · теги: lora,comfyui,turbo,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-09-19T05:11:01.000Z

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
  - text-to-video
  - reference-to-video
  - audio-video
  - synchronized-audio
  - few-step
  - turbo
  - accelerated-inference
  - hyperflow
  - lightx2v
  - dynamic-rank
  - svd
  - safetensors
  - bfloat16
  - pruned-model
  - curve-form
---


# MiniMax-H3 Turbo LoRAs for ComfyUI


This repository contains MiniMax-H3 Turbo and few-step LoRAs converted and optimized for ComfyUI.

These LoRAs accelerate MiniMax-H3 video and synchronized-audio generation by reducing the required number of sampling steps.


---


## Update — HyperFlow 8-Step LoRAs


Four HyperFlow 8-step MiniMax-H3 LoRAs are now included.

They are derived from the original HyperFlow release by Video Rebirth:

- [Video-Rebirth/hyperflow — GitHub](https://github.com/Video-Rebirth/hyperflow)
- [videorebirth/hyperflow — Hugging Face](https://huggingface.co/videorebirth/hyperflow)


HyperFlow is an 8-step MiniMax-H3 LoRA designed for the MiniMax-H3 T2VA, FL2VA and Ref2VA workflows.

The upstream HyperFlow implementation uses its own fixed 8-step sigma schedule. For these ComfyUI conversions, the recommended manual sigma sequence is:


`1.0, 0.9939097854, 0.9842874953, 0.9660637197, 0.9230769231, 0.8349423898, 0.6968520491, 0.4687533149, 0.0`


This is a nine-point sigma sequence defining eight denoising intervals.


### HyperFlow files


| File | Variant | Purpose |
| --- | --- | --- |
| [minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors) | Full BF16 | Full-size ComfyUI HyperFlow 8-step conversion. |
| [minimax_h3_hyperflow_8step_v1.0_comfyui_bf16_resized_avg_rank_20_bf16.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_hyperflow_8step_v1.0_comfyui_bf16_resized_avg_rank_20_bf16.safetensors) | Resized BF16, avg rank 20 | Compact resized version of the full HyperFlow ComfyUI conversion. |
| [minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16.safetensors) | Pruned BF16 | HyperFlow 8-step conversion for the pruned/curve-form MiniMax-H3 base. |
| [minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16_resized_avg_rank_20_bf16.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16_resized_avg_rank_20_bf16.safetensors) | Pruned resized BF16, avg rank 20 | Compact resized HyperFlow version for the pruned/curve-form MiniMax-H3 base. |


Use the non-pruned files with the corresponding full MiniMax-H3 base and the `pruned` files with the compatible pruned/curve-form base.


### Recommended manual HyperFlow schedule


- Denoising steps: `8`
- Sigma points: `9`
- LoRA strength: start at `1.0`
- Manual sigmas: `1.0, 0.9939097854, 0.9842874953, 0.9660637197, 0.9230769231, 0.8349423898, 0.6968520491, 0.4687533149, 0.0` OR euler / simple, normal


When using the original Video Rebirth implementation, follow the upstream HyperFlow scheduler and loader rather than replacing its built-in schedule manually.


---


## Update — TaoMate 3-Step EMA and VDN-H3 8-Step Extracts (experimental)


### TaoMate FL2VA 3-step EMA


Newly added LoRA, located in the `experimental/` folder:


[minimax_h3_taomate_fl2va_3step_ema_comfyui.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/experimental/minimax_h3_taomate_fl2va_3step_ema_comfyui.safetensors?download=true)


Manual recommended sigmas: 3-step `1.0, 0.961165, 0.853333, 0.0` <br> 4-step `1.0, 0.970874, 0.907249, 0.640000, 0.0`


### VDN-H3 8-step extracts


Three LoRAs extracted from VDN-H3 8 step:


- [minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/experimental/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors?download=true)
- [minimax_h3_dmd_fl2va_8step_turbo_pruned.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/experimental/minimax_h3_dmd_fl2va_8step_turbo_pruned.safetensors?download=true)
- [minimax_h3_dmd_8step_turbo.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/experimental/minimax_h3_dmd_8step_turbo.safetensors?download=true)


The main 8-step LoRA works on both FL2VA and Ref2VA. If you are running a pruned base, choose the pruned version that corresponds to your base — the pruned versions need their own matching pruned base.


---


## Update — LightX2V FL2V and Ref2V Dynamic-Rank LoRAs


Five dynamically resized BF16 LoRAs are now included. Their source weights were published in the correct MiniMax-H3 ComfyUI layout by:


[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main)


### New files


| File | Source | Purpose |
| --- | --- | --- |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors` | `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | Four-step 768p FL2V generation. |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | Eight-step FL2V generation with a greater quality focus. |
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_resized_avg_rank_21_bf16.safetensors` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | Four-step reference-to-video generation. |
| `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_resized_avg_rank_64_bf16.safetensors` | `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors` | Four-step 768p FL2V generation, high-fidelity dynamic-rank resize. |
| `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_resized_avg_rank_28_bf16.safetensors` | `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors` | Four-step 768p FL2V generation, compact dynamic-rank resize. |


The LightX2V files were already supplied in the correct ComfyUI namespace and QKV layout. No namespace conversion or AdaLN pruning was required.


### How the LoRAs were resized


Each LoRA was compressed using an exact compact SVD calculated from its LoRA A/B factors without materializing the full dense weight updates.


The process:


1. Treats every attention, MLP and token-refiner projection independently.
2. Separates fused QKV adapters into their original Q, K and V projections.
3. Calculates the exact singular spectrum of each effective `B @ A` update.
4. Selects a dynamic rank independently for every projection.
5. Preserves one additional safety rank beyond the Frobenius-energy threshold.
6. Rebuilds balanced factors as `A = sqrt(S) × Vh` and `B = U × sqrt(S)`.
7. Re-fuses Q, K and V using concatenated A factors and a block-diagonal B matrix.
8. Bakes the source `alpha / rank` scale into `lora_B`.
9. Removes the no-longer-required alpha tensors.
10. Saves all output tensors as BF16.


The three original v1.0/v0.1 dynamic-rank files target the same projection-level average rank:


- Rank sum: `6778`
- Projection count: `312`
- Average projection rank: `21.724359`
- Filename rank: `floor(21.724359) = 21`


Q, K and V are counted separately. The stored rank of a fused QKV adapter is the sum of its three independently selected ranks.


### Resize results


| LoRA | Output size | Average rank | Mean Frobenius norm retained | Baked scale |
| --- | ---: | ---: | ---: | ---: |
| FL2V 4-step v1.0 768p | 284.36 MiB | 21.724359 | 94.95% | 1.0 |
| FL2V 8-step v1.0 | 311.89 MiB | 21.724359 | 97.72% | 0.0625 |
| Ref2V 4-step v0.1 | 311.79 MiB | 21.724359 | 98.33% | 0.0625 |
| FL2V 4-step v1.1 768p (rank 64) | 891.65 MiB | 64.500000 | 98.18% | 1.0 |
| FL2V 4-step v1.1 768p (rank 28) | 375.55 MiB | 28.500000 | 93.59% | 1.0 |


The Ref2V source was reduced from approximately 1,865.57 MiB to 311.79 MiB, an 83.29% reduction. The v1.1 768p source (1,865.57 MiB) was reduced to 891.65 MiB (−52.21%) at rank 64 and to 375.55 MiB (−79.87%) at rank 28.


A separate Frobenius threshold was solved for each LoRA because their singular-value distributions differ. Reusing one fixed threshold would not produce the same average rank.


### Ref2V numerical validation


Direct comparison between the original and resized effective LoRA updates produced:


| Measurement | Result |
| --- | ---: |
| Global cosine similarity | 99.9190% |
| Global relative L2 error | 4.0234% |
| Mean projection Frobenius norm retained | 98.3332% |
| Minimum projection Frobenius norm retained | 96.4904% |
| Projection rank range | 2–105 |
| Size reduction | 83.29% |


These are numerical weight-reconstruction measurements, not a perceptual video-quality score.


### FL2V 4-step v1.1 768p numerical validation


Direct comparison between the v1.1 source and each resized variant produced:


| Measurement | Rank 64 | Rank 28 |
| --- | ---: | ---: |
| Global cosine similarity | 99.1169% | 97.3279% |
| Global relative L2 error | 13.2601% | 22.9625% |
| Mean projection Frobenius norm retained | 98.1774% | 93.5945% |
| Minimum projection Frobenius norm retained | 98.0817% | 92.6661% |
| Projection rank range | 2–120 | 2–99 |
| Size reduction | 52.21% | 79.87% |


These are numerical weight-reconstruction measurements, not a perceptual video-quality score.


### Structural validation


Every resized LightX2V file contains:


| Component | Count |
| --- | ---: |
| BF16 tensors | 416 |
| LoRA A/B pairs | 208 |
| Independently ranked projections | 312 |
| QKV modules | 52 |
| Alpha tensors | 0 |


Validation confirmed:


- all tensors contain finite BF16 values;
- every LoRA A tensor has a matching LoRA B tensor;
- ComfyUI recognizes all 208 adapters and all 416 tensors;
- no incompatible or unresolved tensors remain;
- QKV rank columns remain block-diagonal and ordered as Q, K and V;
- the original effective LoRA scale is preserved;
- the exact average projection rank is 21.724359.


---


## Workflow Example


The following output was generated with the recommended pruned-model workflow using the v4 step-600 EMA LoRA.


<video
src="https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/assets/MiniMax_H3_audio.mp4"
controls
loop
width="100%"> </video>


### Current workflow


[Download the current ComfyUI workflow](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/fl_minimax_h3_turbo_lora_example_workflow.json?download=true)


- Workflow: `fl_minimax_h3_turbo_lora_example_workflow.json`
- LoRA: `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`
- Steps: `6–8`
- Sampler: `Euler`
- Scheduler: `Beta`
- LoRA strength: `1.0`
- Video sigma shift: `12`
- Audio sigma shift: `4–6`


The previous workflow remains available here:


[Download the previous workflow](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/fl_minimax_h3_turbo_lora_example_workflow_old.json?download=true)


[View the previous workflow output](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/assets/MiniMax_H3_00060-audio.mp4)


---


## Included Files


### HyperFlow 8-step LoRAs


| File | Description |
| --- | --- |
| [minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors) | Full BF16 ComfyUI conversion of the HyperFlow 8-step LoRA. |
| [minimax_h3_hyperflow_8step_v1.0_comfyui_bf16_resized_avg_rank_20_bf16.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_hyperflow_8step_v1.0_comfyui_bf16_resized_avg_rank_20_bf16.safetensors) | Compact resized BF16 HyperFlow conversion with average rank 20. |
| [minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16.safetensors) | Pruned/curve-form-compatible BF16 HyperFlow conversion. |
| [minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16_resized_avg_rank_20_bf16.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16_resized_avg_rank_20_bf16.safetensors) | Compact resized HyperFlow conversion for the pruned/curve-form base, average rank 20. |


Recommended manual HyperFlow sigmas:


`1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0`


Original HyperFlow project:


- [Video-Rebirth/hyperflow — GitHub](https://github.com/Video-Rebirth/hyperflow)
- [videorebirth/hyperflow — Hugging Face](https://huggingface.co/videorebirth/hyperflow)


### LightX2V dynamic-rank LoRAs


| File | Description |
| --- | --- |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors` | Dynamic-SVD resize of the official LightX2V four-step 768p FL2V LoRA. |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors` | Dynamic-SVD resize of the official LightX2V eight-step FL2V LoRA. |
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_resized_avg_rank_21_bf16.safetensors` | Dynamic-SVD resize of the official LightX2V four-step Ref2V LoRA. |
| `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_resized_avg_rank_64_bf16.safetensors` | High-fidelity dynamic-SVD resize of the LightX2V four-step v1.1 768p FL2V LoRA. |
| `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_resized_avg_rank_28_bf16.safetensors` | Compact dynamic-SVD resize of the LightX2V four-step v1.1 768p FL2V LoRA. |


### Pruned/curve-form compatibility conversions


| File | Description |
| --- | --- |
| `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors` | **Recommended.** Pruned-model conversion of the v4 step-600 EMA checkpoint. |
| `minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors` | Non-EMA v4 step-600 variant. |
| `minimax_h3_turbo_4step_ema_ckpt850_pruned_comfyui.safetensors` | EMA checkpoint-850 conversion. |
| `minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors` | Non-EMA checkpoint-850 conversion. |
| `minimax_h3_turbo_4step_ema_ckpt500_pruned_comfyui.safetensors` | EMA checkpoint-500 conversion. |
| `minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors` | Non-EMA checkpoint-500 conversion. |
| `minimax_h3_turbo_4step_ema_pruned_comfyui.safetensors` | Initial EMA Turbo preview conversion. |
| `minimax_h3_turbo_4step_pruned_comfyui.safetensors` | Initial non-EMA Turbo preview conversion. |


### Workflows and assets


| File | Description |
| --- | --- |
| `fl_minimax_h3_turbo_lora_example_workflow.json` | Current recommended ComfyUI workflow. |
| `fl_minimax_h3_turbo_lora_example_workflow_old.json` | Previous workflow retained for comparison. |
| `assets/MiniMax_H3_audio.mp4` | Current workflow example output. |
| `assets/MiniMax_H3_00060-audio.mp4` | Previous workflow example output. |


EMA and non-EMA files contain different tensor values and are retained separately.


---


## Pruned/Curve-Form Compatibility Conversions


The older full-model Turbo LoRAs use keys such as:


`blocks.0.attn.qkv_proj.lora_A.weight`


ComfyUI expects:


`diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight`


The compatibility conversion applies:


- `blocks.*` → `diffusion_model.blocks.*`
- `token_refiner.*` → `diffusion_model.token_refiner.*`


The original full-model LoRAs also contain AdaLN adapters whose dimensions do not match the pruned/curve-form MiniMax-H3 checkpoint. Those incompatible adapters were removed.


| Component | Count |
| --- | ---: |
| Source tensors | 518 |
| Source LoRA A/B pairs | 259 |
| Retained tensors | 416 |
| Retained LoRA A/B pairs | 208 |
| Removed AdaLN tensors | 102 |
| Removed AdaLN A/B pairs | 51 |


The removed adapters consist of 50 AdaLN projection pairs from `blocks.0` through `blocks.49` and one pair from `final_layer.adaln_proj.linear`.


These files are partial compatibility conversions and should not be assumed to behave identically to the original full-model LoRA used with its intended complete MiniMax-H3 base model.


The LightX2V dynamic-rank LoRAs do not use this pruning process.


HyperFlow is provided in separate full and pruned variants. Use the variant intended for the corresponding MiniMax-H3 base rather than mixing full-model and pruned-model files.


---


## Usage


Place the LoRA files in:


`ComfyUI/models/loras/`


Load the selected LoRA with its compatible MiniMax-H3 model and workflow.


Recommended starting strength:


`1.0`


If adjustment is needed, test approximately:


`0.8–1.2`


When comparing files, keep the following identical:


- prompt;
- seed;
- resolution;
- frame count;
- base model;
- conditioning inputs;
- sampler;
- scheduler;
- LoRA strength;
- sigma shifts;
- workflow.


### HyperFlow manual sigmas


For the four HyperFlow 8-step ComfyUI files, use:


`1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0`


The list contains nine sigma points for eight denoising steps.


---


## Sampling Guidance


### Three steps


For the experimental TaoMate FL2VA 3-step EMA LoRA, the manual recommended sigma sequence is:


`1.0, 0.961165, 0.853333, 0.0`


### Four steps


Maximum speed and useful for rapid previews. Use the matching four-step LightX2V FL2V or Ref2V workflow.


For the experimental TaoMate LoRA when running four steps, the manual recommended sigma sequence is:


`1.0, 0.970874, 0.907249, 0.640000, 0.0`


### Six steps


A good speed-to-quality compromise for the v4 step-600 EMA pruned conversion.


### Eight steps


Recommended when prioritizing quality, motion consistency and synchronized-audio stability.


For the existing pruned-model workflow, start with:


- Steps: `8`
- Sampler: `Euler`
- Scheduler: `Beta`
- LoRA strength: `1.0`
- Video sigma shift: `12`
- Audio sigma shift: `4–6`


For the HyperFlow 8-step ComfyUI conversions, use the recommended manual sigma schedule:


`1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0`


HyperFlow upstream project:


- [Video-Rebirth/hyperflow](https://github.com/Video-Rebirth/hyperflow)
- [videorebirth/hyperflow](https://huggingface.co/videorebirth/hyperflow)


Follow the source project recommendations when using the LightX2V LoRAs:


[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main)


---


## Audio and Scheduler Notes


MiniMax-H3 generates video and synchronized audio as separate streams with different scheduling requirements.


Incorrect audio scheduling may cause:


- severe distortion;
- noise-like output;
- unstable or blown-out audio;
- completely broken audio.


If audio is unstable, check:


- sampler and scheduler configuration;
- audio and video sigma shifts;
- manual sigma schedule when using a LoRA that requires one;
- ComfyUI version;
- MiniMax-H3 sampling implementation;
- custom sampler nodes;
- whether the workflow matches the selected FL2V, Ref2V, FL2VA or Ref2VA LoRA.


Broken audio should not automatically be attributed to the LoRA.


---


## Optional Acceleration


These LoRAs may be used with compatible ComfyUI acceleration methods such as:


- Comfy Kitchen attention;
- SageAttention;
- Sol Attention;
- Spectrum.


These methods are optional and are not included in this repository. Some attention backends replace one another rather than stacking, and patch order depends on the implementation.


For few-step Turbo and HyperFlow workflows, establish a clean baseline before adding approximate attention or trajectory-forecasting methods.


---


## Attribution


### HyperFlow


The HyperFlow source weights and original implementation were released by Video Rebirth:


- [Video-Rebirth/hyperflow — GitHub](https://github.com/Video-Rebirth/hyperflow)
- [videorebirth/hyperflow — Hugging Face](https://huggingface.co/videorebirth/hyperflow)


Credit for the original HyperFlow weights, training/distillation work, scheduler design and upstream implementation belongs to Video Rebirth and its contributors.


The HyperFlow files in this repository are ComfyUI-oriented conversions and resized/pruned variants of that upstream work. They should not be confused with the original upstream HyperFlow weight file or its official loader.


The four HyperFlow files included here are:


- `minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors`
- `minimax_h3_hyperflow_8step_v1.0_comfyui_bf16_resized_avg_rank_20_bf16.safetensors`
- `minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16.safetensors`
- `minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16_resized_avg_rank_20_bf16.safetensors`


Recommended manual ComfyUI sigma schedule:


`1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0`


### LightX2V LoRAs


The dynamically resized FL2V and Ref2V LoRAs are derived from the official ComfyUI-compatible files released by:


[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main)


Credit for the source weights, training and release belongs to the LightX2V project and its contributors.


This repository provides only the dynamic-rank BF16 resizing, packaging and validation described above.


### Original Turbo LoRAs


The older Turbo LoRA weights, training work and distillation method were released by:


[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)


Original ComfyUI Turbo custom node:


[ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)


This repository does not claim authorship of:


- the original HyperFlow source weights;
- the original HyperFlow training or distillation work;
- the original LoRA weights;
- LoRA training or distillation;
- the few-step sampling methods;
- MiniMax-H3 video/audio scheduling;
- the MiniMax-H3 base model;
- the pruned/curve-form checkpoint;
- the LightX2V source weights;
- the original project documentation or generation code.


The modifications provided here are limited to:


- ComfyUI-oriented HyperFlow conversions;
- pruned/curve-form HyperFlow compatibility variants;
- resized average-rank HyperFlow variants;
- pruned/curve-form ComfyUI compatibility conversions;
- namespace adjustments;
- removal of incompatible AdaLN adapters where required;
- exact dynamic-rank SVD resizing of the LightX2V ComfyUI LoRAs;
- BF16 safetensors packaging and validation.


Please retain attribution and comply with the licenses and terms of the original HyperFlow, LightX2V, Turbo LoRA and MiniMax-H3 projects when using or redistributing these files.
