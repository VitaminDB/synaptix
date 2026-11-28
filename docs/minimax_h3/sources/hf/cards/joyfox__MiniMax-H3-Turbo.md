# joyfox/MiniMax-H3-Turbo

Источник: https://huggingface.co/joyfox/MiniMax-H3-Turbo · скачано 2026-11-28
Загрузок: 0 · лайков: 68 · базовая: ["Comfy-Org/MiniMax-H3"] · теги: comfyui,lora,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-12T05:49:10.000Z

---

---
base_model:
- Comfy-Org/MiniMax-H3
pipeline_tag: image-to-video
tags:
  - comfyui
  - lora
  - minimax-h3
  - text-to-video
  - image-to-video
  - video-generation
  - audio-generation
  - diffusion
  - four-step
---

# MiniMax H3 Turbo — ComfyUI 4-Step T2V and I2V LoRA

This repository contains an inference acceleration LoRA for the ComfyUI release of **MiniMax H3**, covering **Text-to-Video (T2V)** and **Image-to-Video (I2V)**. It targets H3's joint audio-video diffusion path and uses a fixed **4-step Euler** inference contract.

This file is a LoRA, not a standalone diffusion checkpoint. It must be paired with the matching BF16 base model from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Important: ComfyUI v0.31.0 audio regression

ComfyUI v0.31.0 changed MiniMax H3 to the new `ModelSamplingAV` audio-carry sampling path. With the validated **4-step Euler + simple** workflow, this change can produce a much higher noise floor, unstable stereo imaging, and excessive high-frequency artifacts. Generation still completes and the saved video contains a valid audio track, but the resulting sound can be clearly degraded compared with ComfyUI v0.30.0.

This is a ComfyUI sampling compatibility issue, not a problem with this LoRA, the MiniMax H3 Audio VAE, or the video-saving node.

Use the following compatibility node to restore the MiniMax H3 audio sampling behavior from ComfyUI v0.30.0 while keeping the rest of the current ComfyUI installation:

> **[Fix MiniMax H3 audio with ComfyUI-MiniMax-H3-LegacySampling](https://github.com/starsFriday/ComfyUI-MiniMax-H3-LegacySampling)**

Place `MiniMax H3 Legacy Audio Sampling` after the LoRA and other model patches, connect its model output to both `BasicGuider` and `BasicScheduler`, and keep `shift_video = 12.0` and `shift_audio = 3.0`. Do not chain it with the official `ModelSamplingMiniMaxH3` node.

## Supported generation modes

- **Text-to-Video (T2V):** generate a video directly from a text prompt with the FL2VA model and LoRA.
- **Image-to-Video (I2V):** animate an input image using the FL2VA first-frame path; an optional last frame can also guide the ending.

## Comparison protocol

Within each comparison group, all results use the same MiniMax H3 BF16 base model, the same inference input, and the following standardized inference target. Measured generation times are shown directly in the corresponding comparison table.

- **Sampling steps:** 4
- **LoRA strength:** 1.0
- **Video duration:** 10 seconds
- **Output resolution:** approximately 0.9 megapixels per frame

## First comparison group

<table align="center">
  <tr>
    <th align="center" width="33%"><div align="center">No LoRA</div></th>
    <th align="center" width="33%"><div align="center"><a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo">lightx2v LoRA</a></div></th>
    <th align="center" width="33%"><div align="center">Ours LoRA</div></th>
  </tr>
  <tr>
    <td align="center"><strong>190 s</strong></td>
    <td align="center"><strong>178 s</strong></td>
    <td align="center"><strong>180 s</strong></td>
  </tr>
  <tr>
    <td align="center"><video src="https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/asset/asset1_official.mp4" controls loop muted playsinline width="100%"></video></td>
    <td align="center"><video src="https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/asset/asset1_lightx2v.mp4" controls loop muted playsinline width="100%"></video></td>
    <td align="center"><video src="https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/asset/asset1_ours.mp4" controls loop muted playsinline width="100%"></video></td>
  </tr>
</table>

## Second comparison group — Subway scene

<table align="center">
  <tr>
    <th align="center" width="33%"><div align="center">No LoRA</div></th>
    <th align="center" width="33%"><div align="center"><a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo">lightx2v LoRA</a></div></th>
    <th align="center" width="33%"><div align="center">Ours LoRA</div></th>
  </tr>
  <tr>
    <td align="center"><strong>177 s</strong></td>
    <td align="center"><strong>174 s</strong></td>
    <td align="center"><strong>175 s</strong></td>
  </tr>
  <tr>
    <td align="center" width="33%"><video src="https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/asset/asset2_official.mp4" controls loop muted playsinline width="100%"></video></td>
    <td align="center" width="33%"><video src="https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/asset/asset2_lightx2v.mp4" controls loop muted playsinline width="100%"></video></td>
    <td align="center" width="33%"><video src="https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/asset/asset2_ours.mp4" controls loop muted playsinline width="100%"></video></td>
  </tr>
</table>

## Third comparison group — Outdoor lawn scene

<table align="center">
  <tr>
    <th align="center" width="33%"><div align="center">No LoRA</div></th>
    <th align="center" width="33%"><div align="center"><a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo">lightx2v LoRA</a></div></th>
    <th align="center" width="33%"><div align="center">Ours LoRA</div></th>
  </tr>
  <tr>
    <td align="center"><strong>183 s</strong></td>
    <td align="center"><strong>174 s</strong></td>
    <td align="center"><strong>175 s</strong></td>
  </tr>
  <tr>
    <td align="center" width="33%"><video src="https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/asset/asset3_official.mp4" controls loop muted playsinline width="100%"></video></td>
    <td align="center" width="33%"><video src="https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/asset/asset3_lightx2v.mp4" controls loop muted playsinline width="100%"></video></td>
    <td align="center" width="33%"><video src="https://huggingface.co/joyfox/MiniMax-H3-Turbo/resolve/main/asset/asset3_ours.mp4" controls loop muted playsinline width="100%"></video></td>
  </tr>
</table>

## Recommended inference settings

| Setting | Value |
|---|---|
| LoRA strength | `1.0` |
| Steps | `4` |
| Sampler | `Euler` |
| Scheduler | `simple` |
| CFG | Disabled |
| Sigma path | `1.0 → 0.97297297 → 0.92307692 → 0.8 → 0.0` |

The 4-step sigma path is part of the model contract. Different samplers, schedulers, step counts, or LoRA strengths may work experimentally, but are outside the validated configuration.

## ComfyUI usage

1. Place the `.safetensors` file in `ComfyUI/models/loras/`.
2. Load `minimax_h3_fl2va_pruned_bf16.safetensors` as the diffusion model.
3. Add the LoRA with strength `1.0`.
4. On affected ComfyUI v0.31.0 builds, install and connect [ComfyUI-MiniMax-H3-LegacySampling](https://github.com/starsFriday/ComfyUI-MiniMax-H3-LegacySampling).
5. For T2V, use the FL2VA workflow without keyframe conditioning. For I2V, use `MiniMax H3 Image to Video` with the first-frame input.
6. Configure `Euler`, `simple`, and `4` sampling steps.

## How it works

### Four-step flow update

MiniMax H3 predicts a conditional velocity field for a noisy joint audio-video latent. At each sampling boundary, Euler advances the latent with:

`x_next = x + (sigma_next - sigma) * v(x, sigma, condition)`

Running the original model with fewer scheduler evaluations is not enough by itself: each interval becomes much larger, so small velocity errors accumulate quickly. This Turbo LoRA modifies the model's velocity prediction for the fixed four-interval path. The acceleration therefore comes from a learned model correction, not merely from skipping scheduler steps.

The supplied adapter is tied to the published sigma boundaries. `Euler + simple + 4 steps` keeps the network evaluations and integration intervals aligned with that contract.

### Joint audio-video denoising

H3 processes text and visual conditioning, audio latents, and video latents inside one denoising model. Shared transformer blocks model cross-modal timing and semantic relationships, while separate final projections predict audio and video velocity outputs. The LoRA covers both shared blocks and both output paths, so the four-step correction is applied coherently to the synchronized audio-video state.

### FL2VA conditioning

First and last keyframes are encoded into latent-space keyframe conditions aligned to positions on the target video timeline. Their visual embeddings are also presented to the text/vision encoder before the prompt. This path supports text-only generation as well as control from boundary frames.

## Technical profile

- ComfyUI-native LoRA key layout in `safetensors` format.
- Covers attention, MLP, conditioning modulation, video output, and audio output paths.
- Designed for MiniMax H3's synchronized audio-video latent denoising.
- Structural loading was verified against ComfyUI successfully.

## Notes and limitations

- The exact matching base model and the remaining MiniMax H3 components are required and are not included here.
- Four-step inference trades some fine detail and temporal stability for substantially lower sampling cost.
- Visual fidelity, motion, audio quality, and prompt adherence vary with input quality, aspect ratio, duration, and seed.
- This repository does not modify or redistribute the MiniMax H3 base checkpoints. Follow the upstream model's license and usage terms.
