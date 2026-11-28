# abakanai/Minimax_h3_hybrid

Источник: https://huggingface.co/abakanai/Minimax_h3_hybrid · скачано 2026-11-28
Загрузок: 2497 · лайков: 9 · базовая: ["MiniMaxAI/MiniMax-H3","Comfy-Org/MiniMax-H3"] · теги: comfyui,quantized,base_model:Comfy-Org/MiniMax-H3,base_model:finetune:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-09T06:16:22.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
language:
- en
library_name: diffusion-single-file
pipeline_tag: image-to-video
base_model:
- MiniMaxAI/MiniMax-H3
- Comfy-Org/MiniMax-H3
tags:
- comfyui
- minimax-h3
- image-to-video
- video-generation
- audio-video
- blackwell
- nvfp4
- quantized
- diffusion-single-file
widget:
- text: Conservative hybrid — FFN input in NVFP4, 16.38 GiB
  output:
    url: examples/astrella_conservative_16_38g_s20_audio_00001_.mp4
- text: Fast hybrid — QKV and FFN input in NVFP4, 14.03 GiB
  output:
    url: examples/astrella_fast_14_03g_s20_audio_00001_.mp4
- text: Matched-seed A/B contact sheet
  output:
    url: examples/minimax_h3_ab_s20_contact.png
- text: Fast hybrid high-resolution — simple, 1376 × 768, rendered in 568.67 s
  output:
    url: examples/highres/astrella_fast_14_03g_1376x768_s20_simple_audio_00001_.mp4
- text: Conservative hybrid high-resolution — simple, 1376 × 768, rendered in 598.66 s
  output:
    url: examples/highres/astrella_conservative_16_38g_1376x768_s20_simple_audio_00001_.mp4
---

# MiniMax H3 Hybrid NVFP4 for NVIDIA Blackwell

Two experimental mixed-precision quantizations of the **MiniMax H3 Ref2VA Pruned** diffusion model, optimized and tested locally on an NVIDIA GeForce RTX 5070 Ti 16 GB (Blackwell, SM 12.0).

This is an independent community optimization by **abakanai**. It is not an official MiniMax or ComfyUI release. The checkpoints are inference-only single-file diffusion weights for ComfyUI; they are not LoRAs and do not include the text encoder or VAEs.

## Checkpoints

| Checkpoint | Quantization layout | Size | Recommended use |
|---|---|---:|---|
| [`minimax_h3_ref2va_pruned_hybrid_ffn_nvfp4_blackwell.safetensors`](./minimax_h3_ref2va_pruned_hybrid_ffn_nvfp4_blackwell.safetensors) | FFN input: NVFP4; QKV and output projections: INT8 ConvRot | 16.38 GiB | Start here when output fidelity is the priority |
| [`minimax_h3_ref2va_pruned_hybrid_nvfp4_blackwell.safetensors`](./minimax_h3_ref2va_pruned_hybrid_nvfp4_blackwell.safetensors) | QKV and FFN input: NVFP4; output projections: INT8 ConvRot | 14.03 GiB | Lower storage/offload cost and faster linear projections |

The attention-output and FFN-output projections remain INT8 ConvRot in both variants. This preserves the fused SwiGLU path and avoids spending precision where native NVFP4 did not provide a measured speed benefit.

## Generated examples

All media below are AI-generated. The paired videos in each comparison use the same reference image, prompt, seed, resolution, duration, sampler, scheduler, and step count.

<Gallery />

### Matched-seed comparison

![Matched-seed A/B comparison between the 16.38 GiB conservative hybrid and the 14.03 GiB fast hybrid](./examples/minimax_h3_ab_s20_contact.png)

- [Conservative hybrid video with audio](./examples/astrella_conservative_16_38g_s20_audio_00001_.mp4)
- [Fast hybrid video with audio](./examples/astrella_fast_14_03g_s20_audio_00001_.mp4)

### A/B generation settings

| Setting | Value |
|---|---|
| Resolution | 864 × 480 |
| Frames / frame rate | 124 frames / 24 fps |
| Duration | 5.17 seconds |
| Sampler | `res_multistep` |
| Scheduler | `beta` |
| Steps | 20 |
| Seed | `788309604372424` |
| Reference inputs | 1 image |
| Reference sizing | `match` |
| Audio | Native H3 audio latent, stereo AAC output at 32 kHz |

<details>
<summary>Prompt used for both videos</summary>

```text
<Picture 1> The same young adult anime woman looks into the camera, blinks naturally, smiles, then slowly forms a heart with both hands in front of her chest. Her short black hair moves gently. Smooth expressive character animation, stable face and round glasses, consistent fingers and hands, clean pink-and-white background, locked camera, no scene cuts. No speech.
```

</details>

In this single-seed qualitative check, both variants preserved the character's face and glasses across the clip. The conservative variant held finger shapes slightly more consistently, while the fast variant produced a very similar result with somewhat larger changes in background composition and framing. This is one visual sample, not a statistically meaningful perceptual evaluation.

### 1376 × 768 `simple` follow-up

The two hybrid checkpoints were also tested at 1376 × 768 using the same reference, prompt, seed, 124-frame duration, 20 steps, and `res_multistep` sampler. The scheduler was changed to `simple` for both matched runs.

- [Fast 14.03 GiB high-resolution video with audio](./examples/highres/astrella_fast_14_03g_1376x768_s20_simple_audio_00001_.mp4)
- [Conservative 16.38 GiB high-resolution video with audio](./examples/highres/astrella_conservative_16_38g_1376x768_s20_simple_audio_00001_.mp4)

| Checkpoint | Resolution | Video duration | Total render time |
|---|---:|---:|---:|
| Fast 14.03 GiB | 1376 × 768 | 5.17 s | 568.67 s (9:28.67) |
| Conservative 16.38 GiB | 1376 × 768 | 5.17 s | 598.66 s (9:58.66) |

Both runs completed on the RTX 5070 Ti 16 GB through ComfyUI DynamicVRAM and produced 124-frame, 24 fps MP4 files with stereo AAC audio at 32 kHz. The fast checkpoint finished about 5% sooner in these sequential single runs. This is a practical observation rather than a controlled end-to-end benchmark because model and cache state were not fully reset between jobs.

## Research summary

### Test environment

- NVIDIA GeForce RTX 5070 Ti, 16 GB
- Blackwell, SM 12.0
- PyTorch 2.10.0+cu130
- CUDA 13.0
- comfy-kitchen 0.2.26
- BF16 compute
- Baseline: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`, 19.53 GiB

### Attention and INT8 baseline

ComfyUI already dispatched MiniMax H3 full attention to cuDNN SDPA. At a sequence length of 16,384 tokens, one attention call took approximately 88 ms. Changing SDPA backend priority and forcing contiguous Q/K/V layouts did not produce a stable improvement.

The original INT8 ConvRot checkpoint already used a fused CUTLASS kernel. Forcing the cuBLAS fallback was 15–30% slower depending on matrix shape, so the CUTLASS path was retained.

### Native NVFP4 projection benchmarks

Measurements below use a sequence length of 16,000 tokens and report one projection call from one H3 block.

| Operation | INT8 ConvRot | NVFP4 | Change |
|---|---:|---:|---:|
| QKV projection | 12.75 ms | 6.86 ms | −46% |
| Attention output projection | 4.71 ms | 3.15 ms | −33% |
| FFN input projection | 16.60 ms | 9.35 ms | −44% |
| FFN output + SwiGLU | 9.47 ms | 9.57 ms | No speedup |

A full conversion of every linear layer to NVFP4 was not selected. The FFN output projection did not become faster, while numerical error accumulated more strongly across the model's 50 blocks, especially in the audio channel.

For the fast hybrid, the combined time of these four major linear operations decreased from approximately 43.5 ms to 30.4 ms per block, a reduction of about 30%. Full-attention complexity remains the principal high-resolution bottleneck.

### 50-block numerical smoke test

Each candidate was compared with a freshly executed INT8 baseline on identical input tensors.

| Variant | INT8 → candidate | Speedup | Video cosine | Audio cosine |
|---|---:|---:|---:|---:|
| NVFP4 for FFN input only | 1.339 → 0.464 s | 2.89× | 0.992 | 0.937 |
| NVFP4 for QKV and FFN input | 1.111 → 0.287 s | 3.87× | 0.985 | 0.891 |

No NaN or Inf values were observed. These cosine measurements describe numerical similarity for a controlled smoke test; they do not replace full iterative visual and audio evaluation.

## ComfyUI usage

Place one checkpoint in:

```text
ComfyUI/models/diffusion_models/
```

The Ref2VA workflow also requires the companion files from [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3):

```text
ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors
ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors
```

In ComfyUI:

1. Load one of the checkpoints with `UNETLoader` and leave `weight_dtype` at `default`.
2. Use the standard MiniMax H3 reference-to-video conditioning path.
3. Keep the ComfyUI-selected PyTorch attention backend; on the tested setup this resolved to cuDNN SDPA.
4. Use `ref_image_size=match` when maximum reference-token detail is not required, particularly on 16 GB cards.
5. Do not force the INT8 cuBLAS fallback on the tested Blackwell configuration.

The checkpoints rely on native NVFP4 operations exposed by a recent ComfyUI/comfy-kitchen build. They are intended for NVIDIA Blackwell GPUs, including GeForce RTX 50-series cards. Compatibility and performance on pre-Blackwell NVIDIA GPUs, ROCm, MPS, DirectML, XPU, NPU, and other backends have not been validated.

## Limitations

- Tested on one RTX 5070 Ti configuration; results should not be generalized to every RTX 50-series GPU without measurement.
- MiniMax H3 uses full attention in the evaluated open implementation. NVFP4 linear-projection gains do not remove the quadratic attention cost at high token counts.
- The fast variant has lower numerical similarity, especially in the audio channel. Prefer the conservative checkpoint when fidelity matters more than model size.
- The A/B media represents one seed and one reference image. More prompts, motion patterns, identities, durations, and audio scenarios are needed for a broader quality evaluation.
- These checkpoints are experimental community conversions and are provided without warranty.

## License and use restrictions

The MiniMax H3 diffusion weights and derivatives are governed by the **[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)**, not Apache-2.0. The license contains territorial restrictions, an Acceptable Use Policy, redistribution requirements, and additional commercial terms. In particular, the defined applicable territory excludes the European Union, the United Kingdom, the Republic of Korea, and the United States of America. Read the full [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE) before downloading, using, or redistributing these files.

The separately distributed Qwen3-VL-32B encoder is licensed under Apache-2.0 by its respective authors.

## Acknowledgements

- Original model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- ComfyUI single-file release and integration: [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)
- Hybrid quantization, kernel benchmarking, numerical validation, and A/B generation: **abakanai**
