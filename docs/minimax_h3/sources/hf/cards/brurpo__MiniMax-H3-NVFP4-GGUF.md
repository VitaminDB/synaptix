# brurpo/MiniMax-H3-NVFP4-GGUF

Источник: https://huggingface.co/brurpo/MiniMax-H3-NVFP4-GGUF · скачано 2026-11-28
Загрузок: 5312 · лайков: 1 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: gguf,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-16T08:48:17.000Z

---

---
base_model:
- MiniMaxAI/MiniMax-H3
pipeline_tag: image-text-to-video
library_name: stable-diffusion.cpp
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
tags:
- minimax-h3
- gguf
- nvfp4
- text-to-video
- image-to-video
- audio-video-generation
---

# MiniMax H3 NVFP4 GGUF

Native `GGML_TYPE_NVFP4` GGUF diffusion-model weights for MiniMax H3. These files target the MiniMax H3 support in stable-diffusion.cpp and Blackwell-class NVIDIA GPUs.

This repository contains all four combinations of task and pruning level:

| File | Task | Bytes | NVFP4 tensors | SHA-256 |
|---|---|---:|---:|---|
| `minimax_h3_fl2va_nvfp4.gguf` | FL2VA, full | 18,685,049,568 | 470 | `40699037ee725b5fab15594694bbbd50abbaec65bf3c03420be3be171105e229` |
| `minimax_h3_fl2va_pruned_nvfp4.gguf` | FL2VA, pruned | 11,380,270,144 | 418 | `7400c9a10b667322a13cf9f874182e9f5f0daf9b7129eeeef3d90b65b56e176e` |
| `minimax_h3_ref2va_nvfp4.gguf` | Ref2VA, full | 18,685,049,568 | 470 | `4ea6459d6282b3be3a60274ba623431869d6d2299a470ad1fc8888997066e7dd` |
| `minimax_h3_ref2va_pruned_nvfp4.gguf` | Ref2VA, pruned | 11,380,270,144 | 418 | `aa3d1b849da4014eb2f09a07c09c99a416bb16499493bd22778e13e92fba32bb` |

## Compatibility

These are GGUF files with native NVFP4 tensors. They are not ComfyUI packed-NVFP4 safetensors. They were converted and tested with stable-diffusion.cpp commit `ea7f0c8`.

The diffusion model is not a complete standalone H3 package. In stable-diffusion.cpp it also needs:

- a MiniMax H3 Qwen3-VL text encoder, such as `qwen3vl_32b_minimax_h3-Q2_K_M.gguf`;
- `minimax_h3_video_vae_fp16.safetensors`;
- `minimax_h3_audio_vae_fp32.safetensors` when generating synchronized audio.

NVFP4 GPU execution requires a compatible Blackwell NVIDIA GPU and a CUDA build whose compiled architectures include the installed device. Other backends may be able to load the GGUF container but are not the target of this release.

## Provenance

The source tensors are the official BF16 releases under `Comfy-Org/MiniMax-H3`:

| Source file | Bytes | Official LFS SHA-256 |
|---|---:|---|
| `diffusion_models/minimax_h3_fl2va_bf16.safetensors` | 66,280,487,368 | `907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182` |
| `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | 40,225,724,176 | `a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4` |
| `diffusion_models/minimax_h3_ref2va_bf16.safetensors` | 66,280,487,368 | `e32c54c1a7b4f5f397f195cea267ccb18806303bb665678c4bee60953bdf3026` |
| `diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors` | 40,225,724,176 | `37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe` |

Conversion command pattern:

```text
sd-cli --mode convert --model SOURCE.safetensors --output OUTPUT.gguf --type nvfp4
```

## Validation

All four files passed the following checks before upload:

1. official source byte length and LFS SHA-256;
2. GGUF v3 structure, expected H3 tensor count, and native type-40 NVFP4 tensor inventory;
3. complete stable-diffusion.cpp load with the Q2 H3 text encoder plus video and audio VAEs;
4. one-step, 256x256, 9-frame CUDA video generation on an NVIDIA GeForce RTX 5070 Ti (compute capability 12.0);
5. successful diffusion sampling, VAE decode, and non-empty AVI output.

The full variants exceed 16 GiB if staged as one buffer. They passed with `--max-vram 14 --stream-layers`; the pruned variants passed with ordinary CPU parameter offload. Low-level backend tests also identified CUDA0 and passed native NVFP4 matrix multiplication plus the supported SM 12.0 flash-attention cases.

## License

The upstream MiniMax H3 Community License Agreement is included as `LICENSE`. Review it before downloading, redistributing, or using these derivatives.

## Related artifact

An independently published compatible NVFP4 GGUF exists at `convertor/minimax-h3-gguf` for the Ref2VA-pruned variant. This repository's purpose is to provide a consistently converted and validated four-variant set.
