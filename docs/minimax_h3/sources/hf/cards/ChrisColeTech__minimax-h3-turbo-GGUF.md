# ChrisColeTech/minimax-h3-turbo-GGUF

Источник: https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF · скачано 2026-11-28
Загрузок: 25866 · лайков: 24 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: gguf,quantized,distilled,turbo,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-10-15T15:35:47.000Z

---

---
license: other
pipeline_tag: text-to-video
tags:
  - text-to-video
  - image-to-video
  - gguf
  - quantized
  - minimax-h3
  - distilled
  - turbo
  - lightx2v
  - clipproj
library_name: gguf
base_model:
  - MiniMaxAI/MiniMax-H3
---

# MiniMax H3 Turbo Merge GGUF (8-step LoRA baked-in)

The MiniMax H3 **FL2VA (pruned)** DiT with the **LightX2V Turbo 8-step v1.0** LoRA **baked in**.

and

The MiniMax H3 **REF2VA (pruned)** DiT with the **LightX2V Turbo 4-step v1.0** LoRA **baked in**.

**Modes:** `img2video`, `txt2video`, FL2VA / T2VA / I2VA / REF2VA.

---

## Samples (4B vs 8B Text Encoder)

Both samples use the **same** turbo **Q4_K_M turbo GGUF** diffusion model, prompt, and settings, only the **text encoder** changes (4B vs 8B).

### 4B Text Encoder + ClipProj (8-steps)
<video controls width="700" src="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/txt2video-red-car-4b-clipproj-q6.mp4"></video>
![frames at 0s / 1.25s / 2.5s / 3.7s](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/txt2video-red-car-4b-frames.png)
 - **(~2.3 min)** on an **RTX 5090 (32 GB)**.

## Components used to generate

| Component | File | ~Size | Download |
|---|---|---|---|
| Transformer | `minimax_h3_fl2va_turbo_Q4_K_M.gguf` | 11.4 GB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/diffusion_models/minimax_h3_fl2va_turbo_Q4_K_M.gguf) |
| **4B Text Encoder** | `qwen3vl_4b_fp8_scaled.safetensors` | 5.24 GB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) |
| **4B Clip Proj** | `mmh3-4b-ClipProj-v3-mlp.safetensors` | 304 MB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/text_encoders/mmh3-4b-ClipProj-v3-mlp.safetensors) |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB| [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/vae/minimax_h3_video_vae_fp16.safetensors) |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 605 MB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/vae/minimax_h3_audio_vae_fp32.safetensors) |



### 8B Text Encoder + ClipProj (8-steps)
<video controls width="700" src="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/txt2video-red-car-8b-clipproj-q6.mp4"></video>
![frames at 0s / 1.25s / 2.5s / 3.7s](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/txt2video-red-car-8b-frames.png)
 - **(~2.5 min)** on an **RTX 5090 (32 GB)**.

## Components used to generate

| Component | File | ~Size | Download |
|---|---|---|---|
| Transformer | `minimax_h3_fl2va_turbo_Q4_K_M.gguf` | 11.4 GB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/diffusion_models/minimax_h3_fl2va_turbo_Q4_K_M.gguf) |
| **8B Text Encoder** | `qwen3vl_8b_fp8_scaled.safetensors` | 10.6 GB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/text_encoders/qwen3vl_8b_fp8_scaled.safetensors) |
| **8B Clip Proj** | `mmh3-8b-ClipProj-v3-mlp.safetensors` | 386 MB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/text_encoders/mmh3-8b-ClipProj-v3-mlp.safetensors) |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB| [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/vae/minimax_h3_video_vae_fp16.safetensors) |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 605 MB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/vae/minimax_h3_audio_vae_fp32.safetensors) |

### Sample settings 

| Parameter | Value |
|---|---|
| Prompt | `cinematic tracking shot of a red sports car on a wet night street, neon reflections, engine rumble and rain on asphalt, shallow depth of field` |
| Resolution | **1280 × 720** |
| Frames | **90 frames**  → ~3.75 s |
| Steps | **8** |
| CFG | **1.0** |
| Seed | **42** |

---

## Image-to-Video (8-steps)

<table style="text-align: -webkit-center">
<tr >
<td style="align-content: center;text-align: center"><img src="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/ComfyUI_00003.png" style="width:400px"></td>
<td style="align-content: center;text-align: center"><video controls src="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/minimax-h3-turbo_00013_.mp4" style="height:400px;"></video></td></tr>
<tr><td colspan=2 style="align-content: center;text-align: center"><strong>Prompt:</strong> <br>Use this image as the first frame. <br>Make her fly. Full body shot. <br>High quality</td></tr>
</table>

Link to workflow [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/workflow_example/minimax_h3_turbo_i2va.json)

## Components used to generate

| Component | File | ~Size | Download |
|---|---|---|---|
| Transformer | `minimax_h3_fl2va_turbo_Q4_K_M.gguf` | 11.4 GB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/diffusion_models/minimax_h3_fl2va_turbo_Q4_K_M.gguf) |
| **8B Text Encoder** | `qwen3vl_8b_fp8_scaled.safetensors` | 10.6 GB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/text_encoders/qwen3vl_8b_fp8_scaled.safetensors) |
| **8B Clip Proj** | `mmh3-8b-ClipProj-v3-mlp.safetensors` | 386 MB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/text_encoders/mmh3-8b-ClipProj-v3-mlp.safetensors) |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB| [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/vae/minimax_h3_video_vae_fp16.safetensors) |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 605 MB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/vae/minimax_h3_audio_vae_fp32.safetensors) |

---

## REF2VA works with up to 9 input images

<table style="text-align: -webkit-center">
<tr><td style="height:300px; justify-items: center"><img src="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/a-red-vintage-car-parked-on-a-sunny-cobblestone-street-blue-20260813-031631.png" style="height:300px; justify-items: center" ></td><td style="height:300px; justify-items: center"><img src="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/a-solitary-glass-greenhouse-on-a-foggy-cliff-edge-at-dawn-20260808-122900.webp" style="height:300px"></td></tr>
<tr><td style="height:300px; justify-items: center"><img src="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/an-astronaut-sitting-in-a-field-of-tall-wildflowers-holding-20260808-123526.webp" style="height:300px; justify-items: center"></td><td style="height:300px; justify-items: center"><img src="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/close-up-portrait-of-an-elderly-fisherman-with-weathered-20260808-123813.webp"  style="height:300px"></td></tr>
</table>
<video controls src="https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/resolve/main/samples/ComfyUI_00008_.mp4" style="height:400px;"></video>

## Components used to generate

| Component | File | ~Size | Download |
|---|---|---|---|
| Transformer | `minimax_h3_ref2va_turbo_Q4_K_M.gguf` | 11.4 GB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/diffusion_models/minimax_h3_ref2va_turbo_Q4_K_M.gguf) |
| **8B Text Encoder** | `qwen3vl_8b_fp8_scaled.safetensors` | 10.6 GB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/text_encoders/qwen3vl_8b_fp8_scaled.safetensors) |
| **8B Clip Proj** | `mmh3-8b-ClipProj-v3-mlp.safetensors` | 386 MB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/text_encoders/mmh3-8b-ClipProj-v3-mlp.safetensors) |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB| [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/vae/minimax_h3_video_vae_fp16.safetensors) |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 605 MB | [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/split/vae/minimax_h3_audio_vae_fp32.safetensors) |

---

## ⚠ Must use the updated GGUF Loader ⚠ 

In comfy:
- Open the ComfyUI Manager
- Change the channel to "Channel (remote)"
- and search for comfyui-gguf-loader

![image](https://cdn-uploads.huggingface.co/production/uploads/6a550d1dc84594e3ada10828/SgiBz44CKsKT5m2sCXDnB.png)


Use the turbo workflow [Link](https://huggingface.co/ChrisColeTech/minimax-h3-turbo-GGUF/blob/main/workflow_example/minimax_h3_turbo.json)

![image](https://cdn-uploads.huggingface.co/production/uploads/6a550d1dc84594e3ada10828/cqME7svM_hbjMbeJjMAad.png)

---

## Recommended settings (turbo 8-step)

| Parameter | Value | Notes |
|---|---|---|
| `width` × `height` | **1280 × 720** | 720P max resolution |
| `num_frames` | `124` | `481 max frames` |
| `frame_rate` | `24.0` | |
| `steps` / NFE | **`8`** | 8-steps minimum, 16 steps max |
| `guidance` / CFG | **`1.0`** | always use 1.0 for turbo variants |
| `negative_prompt` | _(empty in samples)_ | optional |

---

## Sources

| | |
|---|---|
| **Base DiT** | MiniMax H3 FL2VA pruned `fp8_scaled` — family of [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| **Turbo LoRA** | **LightX2V FL2VA Turbo 8-step v1.0 (ComfyUI)** — `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` |
| **Turbo weights** | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |
| **Turbo docs / code** | [ModelTC/Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo) · [LightX2V](https://github.com/modeltc/lightx2v) |
| **Upstream 8-step specs** | FL2VA / T2VA · train 544p mixed AR · video/audio shifts **12 / 3** · distill **8 NFE** · recommended inference **8** (upstream also lists 4 for this LoRA) |
| **This bake** | Offline fuse @ strength **1.0** (208/208 keys) → stochastic re-FP8 → GGUF ladder |
| **Runtime LoRA** | **Not required** — already fused into these files |

---

## Notes
- **Known Limitations** [Here](https://github.com/nicolab28/ComfyUI-ClipProj#known-limitations)
- **ComfyUI + 4B/8B:** matching `mmh3-*-ClipProj-celeb-mlp` is REQUIRED for small TEs.
- **Audio is first-class** on H3 — sample mp4s include stereo AAC.
- **Frame grid:** `num_frames % 17 == 5` (90 and 124 are valid).
- **MiniMax H3** — custom / other terms; read upstream before commercial use.
- **LightX2V turbo LoRA** — see [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) and [ModelTC/Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo).
- **Qwen3-VL** — Apache-2.0 (Alibaba).
- **ClipProj** — MIT code ([nicolab28/ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj)); matrices per NicoLab’s card.
