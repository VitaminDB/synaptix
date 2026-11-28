# BennyDaBall/MiniMax-H3-Workflows

Источник: https://huggingface.co/BennyDaBall/MiniMax-H3-Workflows · скачано 2026-11-28
Загрузок: 0 · лайков: 4 · базовая:  · теги: comfyui,turbo,lora · обновлено: 2026-11-23T18:43:49.000Z

---

---
license: apache-2.0
pipeline_tag: text-to-video
tags:
- comfyui
- workflow
- minimax-h3
- text-to-video
- video
- audio
- turbo
- lora
- upscaling
- open-h3-regenerate-2k
- pi-h3
- hyperflow
- nvfp4
- few-step
---

# MiniMax H3 — ComfyUI Workflows

![MiniMax H3 ComfyUI Workflows, video with native stereo audio, by BennyDaBall](banner.png)

**Follow me on X [@BennyDaBall_OG](https://x.com/BennyDaBall_OG) !**

**Want to support my work? Buy me a coffee on [Ko-fi](https://ko-fi.com/bennydaball) !**

Drag-and-drop ComfyUI workflows for [MiniMax H3](https://www.minimax.io/blog/minimax-h3), the omni-modal video model that generates video **with native stereo audio** (voice, SFX, room tone — all in one pass). Shared by [@BennyDaBall_OG](https://x.com/BennyDaBall_OG).

## Workflows

| File | What it does |
|---|---|
| [`piH3_HyperFlow_NVFP4_480p_MaxSpeed_T2VA_BennyDaBall.json`](./piH3_HyperFlow_NVFP4_480p_MaxSpeed_T2VA_BennyDaBall.json) | **pi-H3, 2 steps, on my HyperFlow H3 Flash NVFP4** at 864x480: 5s of video with sound in about **4.6s** |
| [`piH3_Official_480p_MaxSpeed_NVENC_T2VA_BennyDaBall.json`](./piH3_Official_480p_MaxSpeed_NVENC_T2VA_BennyDaBall.json) | The official **pi-H3** workflow at 864x480 with a GPU (NVENC) save: about **4.8s** per clip |
| [`piH3_HyperFlow_NVFP4_2step_T2VA_BennyDaBall.json`](./piH3_HyperFlow_NVFP4_2step_T2VA_BennyDaBall.json) | **pi-H3, 2 steps, on HyperFlow NVFP4** at 1344x768: about **12.7s** per 5s clip |
| [`piH3_HyperFlow_NVFP4_4step_T2VA_BennyDaBall.json`](./piH3_HyperFlow_NVFP4_4step_T2VA_BennyDaBall.json) | **pi-H3, 4 steps, on HyperFlow NVFP4** at 1344x768: about **20.2s** per 5s clip |
| [`OpenH3_FaceRegion_Prompt_to_2K_BennyDaBall.json`](./OpenH3_FaceRegion_Prompt_to_2K_BennyDaBall.json) | Prompt → 768p H3 draft → **2560x1440 regenerate that keeps the faces and the lip sync**, in one queue |
| [`MiniMax_H3_Turbo_v4_8step_T2VA_BennyDaBall.json`](./MiniMax_H3_Turbo_v4_8step_T2VA_BennyDaBall.json) | Text → video+audio with the **Turbo v4 LoRA** — 8 steps instead of 20, ~**1.9× faster**, audio stays clean |

## pi-H3 · 2 steps, no VAE decode, faster than real time at 480p

[pi-H3](https://huggingface.co/Lakonik/pi-H3) (Lakonik, pi-Flow v2) distills MiniMax H3 to 2 or 4 steps, and its DWD adapters predict pixels at the last step, so there is no VAE decode. All four workflows build on the author's DWD + Spark-Attn workflow from [ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow):

- **`piH3_Official_480p_MaxSpeed_NVENC`**: the author's workflow on the stock H3 INT8 base at 864x480, saved on the GPU (NVENC) instead of the CPU.
- **`piH3_HyperFlow_NVFP4_2step` / `4step`**: pi-H3 on my NVFP4 build of [HyperFlow H3 Flash](https://huggingface.co/BennyDaBall/HyperFlow-H3-Flash-NVFP4-ComfyUI) at 1344x768, through my **pi-H3 Adapter for HyperFlow** node.
- **`piH3_HyperFlow_NVFP4_480p_MaxSpeed`**: the same at 864x480 with the NVENC save.

**My numbers** (one RTX 5090, 5s with audio, Spark-Attn, warm, queue to saved MP4):

| Workflow | Resolution | Steps | Repeat clip | New prompt | Peak VRAM |
|---|---|---|---|---|---|
| **HyperFlow NVFP4, 480p max speed** | 864x480 | 2 | **4.6s** | **5.5s** | 28 GB |
| Official, 480p max speed | 864x480 | 2 | 4.8s | 8 to 14s | 31 GB |
| **HyperFlow NVFP4, 2 steps** | 1344x768 | 2 | **12.7s** | | 24 GB |
| Official pi-H3 workflow, as shipped | 1344x768 | 2 | 13.3 to 13.9s | | 29 GB |
| **HyperFlow NVFP4, 4 steps** | 1344x768 | 4 | **20.2s** | | 24 GB |
| Official pi-H3 workflow, as shipped | 1344x768 | 4 | 22.0s | | 29 GB |

A 5.2s clip at 480p renders in less time than it plays. *Repeat clip* is the same prompt with a new seed; *new prompt* includes the text encode. On the stock base the 15 GB text encoder pushes part of the model out of VRAM, so the first clip or two after a new prompt run slower. The HyperFlow NVFP4 model and the text encoder fit together, so its new-prompt clips stay fast.

**The recipe**

| Setting | Value |
|---|---|
| Adapter | `pi_h3_2step_dwd_adapter_int8_convrot` (or the 4-step one), switched by the adapter group's `steps` widget |
| Sampler | euler + *pi-Flow Scheduler*: max_raw_t 0.93 (2 steps) or 0.96 (4 steps), final step scale 0.6; sigma shift 12 / 3 |
| Attention | Spark-H3 SM120, speed mode |
| Decode | *DWD Decode*, no VAE decode (the video VAE only encodes first / last frames) |
| Save, 480p | VideoHelperSuite *Video Combine*, `video/nvenc_h264-mp4`, 10 Mbit. The file with sound is `-audio.mp4` |
| HyperFlow | official *HyperFlow Loader* + *pi-H3 Adapter for HyperFlow*: `time_path` pi-H3 (stock H3 time curve), `lora_mode` bypass |

**Custom nodes** (install with ComfyUI Manager or git clone into `custom_nodes`):
- [Lakonik/ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow) 1.4.2 or newer: *pi-Flow Scheduler*, *Load LakonLab Adapter*, *DWD Decode*
- [zechengtang/Spark-H3](https://github.com/zechengtang/Spark-H3) 0.2.0: Spark-Attn
- [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite): the NVENC save (480p workflows)
- HyperFlow workflows: [Video-Rebirth/ComfyUI-HyperFlow-Flash](https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash) (the loader) and [HyperFlow INT8 Nodes](https://github.com/BennyDaBall930/comfyui_hyperflow_int8_nodes) 0.2.0 or newer (mine): *pi-H3 Adapter for HyperFlow*

**Extra models**:

| File | Size | Folder |
|---|---|---|
| [`pi_h3_2step_dwd_adapter_int8_convrot.safetensors`](https://huggingface.co/Lakonik/pi-H3/resolve/main/pi_h3_2step_dwd_adapter_int8_convrot.safetensors) | 2.19 GB | `models/loras` |
| [`pi_h3_4step_dwd_adapter_int8_convrot.safetensors`](https://huggingface.co/Lakonik/pi-H3/resolve/main/pi_h3_4step_dwd_adapter_int8_convrot.safetensors) | 2.19 GB | `models/loras` |
| [`minimax_h3_video_vae_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) | 2.81 GB | `models/vae` |
| [`HyperFlow-H3-Flash-fl2va-NVFP4-ComfyUI.safetensors`](https://huggingface.co/BennyDaBall/HyperFlow-H3-Flash-NVFP4-ComfyUI/resolve/main/diffusion_models/HyperFlow-H3-Flash-fl2va-NVFP4-ComfyUI.safetensors) (HyperFlow workflows) | 10.94 GB | `models/diffusion_models` |

Plus the H3 INT8 base, the text encoder and the audio VAE from the table below. The HyperFlow workflows read only the base's time table, not its weights.

**Known limits**
- 480p is a speed setting: detail is well below 1344x768.
- pi-H3 has its own look, glossier and more saturated than stock H3.
- On HyperFlow the spoken lines come out less reliably than on the stock base. Whisper heard the scripted line word for word in 5 of 8 HyperFlow clips against 8 of 8 stock clips at 480p. Quiet scenes also come out louder.

## Open-H3 Face Region · prompt to 2K

NVIDIA's [Open-H3-Regenerate-2K](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/) re-draws an upscaled H3 clip from 91% noise. That adds a lot of detail, but it recasts the actors. This workflow adds an **automatic face mask**: faces start the 2K pass at 50% noise while everything else still gets NVIDIA's full regenerate, and the draft's own audio is held clean so the mouths keep talking.

**My numbers** (talking close-up, same draft and seeds, face match = ArcFace vs the 768p draft, lip sync = SyncNet LSE-C):

| Setting | Face match | Lip sync (draft 8.22) | Detail vs plain upscale |
|---|---|---|---|
| NVIDIA default (0.909 everywhere) | 0.52 | 8.58 | 15.5x |
| My first try (0.7 everywhere, 2 steps) | 0.85 | 8.14 | 4.9x |
| **This workflow** | **0.91** | **8.27** | **10.8x** |

**The recipe**

| Setting | Value |
|---|---|
| Draft | stock H3 T2VA, 1344×768, 124 frames, 20 steps res_multistep |
| Upscale | LBH 3D latent upscaler to 2560×1440 |
| Face mask | *Open-H3 Face Region Map* (OpenCV YuNet, CPU, a few seconds): `face_strength` 0.55 = faces start at 0.50; `extra_boxes` protects a pet or prop |
| 2K pass | NVIDIA release grid, 3 Euler steps from 0.909, CFG-free (*BasicGuider*), draft audio held clean |
| LoRA | Kijai's Open-H3-Regenerate-2K **refiner only** (r384; the 155 MB r16 matched it in my tests) @ 1.0, no Turbo or Sharp |
| VAE | H3 video VAE INT8 for encode and decode |
| Time | about 7 min cold on an RTX 5090 for draft + 2K (measured 410 s). **Fast mode:** un-bypass the Sol sparse attention node and set the 2K steps to 2 (~70 s for the 2K pass, same face match, a bit less detail) |

**Custom nodes** (install with ComfyUI Manager or git clone into `custom_nodes`):
- [ComfyUI-Open-H3-FaceRegion](https://github.com/BennyDaBall930/ComfyUI-Open-H3-FaceRegion): the face mask + region noise nodes (mine)
- [rdh9616/ComfyUI-Open-H3-Regenerate](https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate): draft prep, upscale, refine, sigmas
- [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)

**Extra models** (also linked from the loader nodes):

| File | Folder |
|---|---|
| [`minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors`](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors) (5.9 GB, or the [155 MB r16](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors)) | `models/loras` |
| [`minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors) | `models/latent_upscale_models` |
| [`minimax_h3_video_vae_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) | `models/vae` |

Plus the H3 int8 model, text encoder and audio VAE from the table below.

**Honest take:** only the masked faces (and any boxes you add) are protected, so background props can still change. Small faces in wide shots gain less than close-ups.

## Turbo v4 · 8-step T2VA

100% core ComfyUI nodes — **no custom node packs**. Verified with a same-seed A/B against the stock 20-step recipe: same choreography, healthy audio (peak ≈ −9 dB, zero clipping).

**The recipe**

| Setting | Value |
|---|---|
| LoRA | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui` @ **1.0** (built-in *LoraLoaderModelOnly*) |
| Steps / sampler / scheduler | **8 · euler · beta** (no sigma-shift node needed) |
| Guidance | CFG-free (*BasicGuider*) — write a `Negative:` line inside the prompt itself |
| Resolution | 1344×768 @ 24 fps |
| Length | frames on a 17k+5 grid: **124 ≈ 5 s**, 243 ≈ 10 s, 362 ≈ 15 s (model max) |

The bundled demo prompt ships with a **fixed seed** so your first render reproduces the demo clip — flip the seed widget to *randomize* for new takes.

## Getting started

1. Update ComfyUI to **≥ 0.30** (MiniMax H3 nodes are built in).
2. Drag the `.json` onto the ComfyUI canvas.
3. ComfyUI will offer to **download any missing models automatically** (every loader node carries its download URL). Manual links below.
4. Queue. First run: the console should show **zero** `lora key not loaded` warnings.

## Models

| File | Size | Folder |
|---|---|---|
| [`minimax_h3_fl2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | 19.5 GB | `models/diffusion_models` |
| [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | 14.6 GB | `models/text_encoders` |
| [`minimax_h3_video_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) | 1.4 GB | `models/vae` |
| [`minimax_h3_audio_vae_fp32.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) | 0.4 GB | `models/vae` |
| [`minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors) | 620 MB | `models/loras` |

## Tested / not tested

- Verified on **5-second clips at 1344×768**. Higher res (1920×1088) and longer clips are untested with this LoRA — earlier turbo versions fell apart there, so treat that as experimental.
- Few-step audio blowout is the classic turbo failure mode; this v4 @ 8 steps is the combo that passed. If you drop steps further, check your audio peaks.

## Prompt pattern that works

Style contract → timed beats (`0.0-0.6s: ...`) → camera lock line → an audio timeline starting with room tone at 0.0s → `Negative:` line. Dialogue inside `<d>[English] ... </d>` is spoken verbatim with lip-sync. Give the first spoken line ≥ 1.2 s of lead-in and anchor t=0 with room tone. The demo prompt in the workflow shows the full pattern.

## Credits

- **MiniMax H3** by [MiniMax](https://www.minimax.io/) · ComfyUI-repackaged weights by [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3)
- **Turbo LoRA** trained by [larryvrh](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) (v4_step600 EMA) · ComfyUI convert by [drbaph](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI)
- **Open-H3-Regenerate-2K** by NVIDIA ([Efficient-Large-Model](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)) · refiner LoRA slices by [Kijai](https://huggingface.co/Kijai/MiniMax-H3-experimental) · latent upscaler by [LBH-123-AI](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) · regenerate nodes by [rdh9616](https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate) · YuNet face detector by Shiqi Yu et al. ([opencv_zoo](https://github.com/opencv/opencv_zoo), MIT)
- **pi-H3** by Hansheng Chen ([Lakonik/pi-H3](https://huggingface.co/Lakonik/pi-H3)) · [ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow) · **Spark-H3** by [zechengtang](https://github.com/zechengtang/Spark-H3) · **HyperFlow H3 Flash** by [Video Rebirth](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash)
- Workflow + testing by [@BennyDaBall_OG](https://x.com/BennyDaBall_OG)

*License note: this repo covers the workflow files only. The H3 model weights and the LoRA carry their own licenses — check the linked repos before commercial use.*

**Follow me on X [@BennyDaBall_OG](https://x.com/BennyDaBall_OG) !**

**Want to support my work? Buy me a coffee on [Ko-fi](https://ko-fi.com/bennydaball) !**
