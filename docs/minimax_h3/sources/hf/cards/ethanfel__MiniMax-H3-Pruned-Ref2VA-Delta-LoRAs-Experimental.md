# ethanfel/MiniMax-H3-Pruned-Ref2VA-Delta-LoRAs-Experimental

Источник: https://huggingface.co/ethanfel/MiniMax-H3-Pruned-Ref2VA-Delta-LoRAs-Experimental · скачано 2026-11-28
Загрузок: 0 · лайков: 16 · базовая: ["MiniMaxAI/MiniMax-H3","Comfy-Org/MiniMax-H3"] · теги: comfyui,lora,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-11T19:31:00.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
  - MiniMaxAI/MiniMax-H3
  - Comfy-Org/MiniMax-H3
tags:
  - comfyui
  - lora
  - minimax-h3
  - video-generation
  - audio-generation
  - experimental
---

# MiniMax-H3 pruned FL2VA ↔ Ref2VA delta LoRAs

> [!WARNING]
> **Highly experimental, mechanically extracted adapters.** These are randomized-SVD approximations of the weight difference between the official pruned BF16 Ref2VA and FL2VA checkpoints. They were not trained as LoRAs and have not been generation-tested. Only safetensors integrity and structural loading with ComfyUI's current LoRA parser were validated. Expect artifacts, capability loss, or unpredictable behavior.

These model-only ComfyUI LoRAs explore transferring behavior in either direction between the pruned FL2VA and Ref2VA models. The forward files approximate:

```text
delta = pruned Ref2VA BF16 - pruned FL2VA BF16
```

The reverse files are exact sign-inverted forms of those approximations:

```text
reverse delta = pruned FL2VA BF16 - pruned Ref2VA BF16
```

At strength `1.0`, applying an adapter to its matching base moves the model toward a rank-limited approximation of the other variant. Lower strengths interpolate only in the extracted weight direction; they do not isolate a clean semantic capability.

## Required checkpoints and directions

The filenames state both the checkpoint to load and the direction of the
adapter. `apply_to_fl2va__toward_ref2va` means exactly: load the pruned FL2VA
checkpoint, then apply that LoRA to move its weights toward Ref2VA.

For files named `apply_to_fl2va__toward_ref2va`, use this base:

```text
Comfy-Org/MiniMax-H3
diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
```

Base SHA-256:

```text
a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4
```

For files named `apply_to_ref2va__toward_fl2va`, use this base:

```text
diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors
SHA-256: 37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe
```

Do not expect correct results with the full-size BF16, FP8, or INT8 checkpoints.

## Files

| Apply to this base | Move toward | File | Size |
|---|---|---|---:|
| pruned FL2VA BF16 | Ref2VA | `minimax_h3_pruned_bf16__apply_to_fl2va__toward_ref2va__rank256.safetensors` | 2.589 GB |
| pruned FL2VA BF16 | Ref2VA | `minimax_h3_pruned_bf16__apply_to_fl2va__toward_ref2va__rank512.safetensors` | 5.075 GB |
| pruned FL2VA BF16 | Ref2VA | `minimax_h3_pruned_bf16__apply_to_fl2va__toward_ref2va__rank1024.safetensors` | 10.048 GB |
| pruned Ref2VA BF16 | FL2VA | `minimax_h3_pruned_bf16__apply_to_ref2va__toward_fl2va__rank256.safetensors` | 2.589 GB |
| pruned Ref2VA BF16 | FL2VA | `minimax_h3_pruned_bf16__apply_to_ref2va__toward_fl2va__rank512.safetensors` | 5.075 GB |
| pruned Ref2VA BF16 | FL2VA | `minimax_h3_pruned_bf16__apply_to_ref2va__toward_fl2va__rank1024.safetensors` | 10.048 GB |

### What rank preserves

| Rank | Median capture across compressed matrices | Mean capture across compressed matrices | Global energy-weighted capture |
|---:|---:|---:|---:|
| 256 | 14.32% | 16.77% | 99.1260% |
| 512 | 26.40% | 29.70% | 99.2417% |
| 1024 | 46.11% | 49.62% | 99.4311% |

The global weighted percentage is **not the full story and is not a quality
score**. It is dominated by a small number of high-energy tensors with low
effective rank. Of the 264 matrix deltas, 54 low-rank matrices are reconstructed
essentially exactly at every requested rank. The table's median and mean are
calculated over the remaining 210 matrices that are genuinely compressed;
typical large attention and MLP matrices retain much less energy. Rank 1024 can
therefore be materially more faithful than rank 256 even though their global
weighted figures differ by only about 0.3 percentage points. Weight-space
energy also does not predict which directions matter most during inference;
compare actual generations with identical prompts, inputs, seeds, and sampler
settings.

Each `.safetensors.json` file contains the complete per-layer extraction report.

## ComfyUI use

1. Put one `.safetensors` file in `ComfyUI/models/loras/`.
2. Follow the filename literally: load pruned FL2VA for an `apply_to_fl2va` file, or pruned Ref2VA for an `apply_to_ref2va` file.
3. Apply the adapter with a model-only LoRA loader.
4. Start at model strength `1.0`; compare `0.25`, `0.5`, `0.75`, and `1.0` using identical inputs.
5. Do not stack ranks or opposing directions together. They are alternative forms of the same underlying delta.

All six files loaded as 531 expected ComfyUI patches each: 264 matrix adapters and 267 exact vector/bias diff patches. `rope.inv_freq` was identical between the source checkpoints and was omitted.

## Extraction details

- Matrix method: deterministic randomized SVD
- Requested ranks: 256, 512, and 1024
- Oversampling: 32
- Power iterations: 2
- Factor dtype: BF16
- Vector and bias patches: FP32 ComfyUI `.diff` / `.diff_b`
- Source tensors: 532 matching tensors
- Peak extractor CUDA allocation: approximately 1.24 GB
- Reverse-direction construction: exact sign inversion of one LoRA factor and every `.diff` / `.diff_b` tensor

The included `extract_minimax_h3_ref2va_lora.py` performs checkpointed, one-tensor-at-a-time extraction. It avoids loading both 40 GB checkpoints into memory simultaneously. `invert_comfy_lora_direction.py` creates the opposite direction without repeating or degrading the SVD.

## License and territorial restrictions

These are modified MiniMax H3 model derivatives. They remain subject to the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) and its Acceptable Use Policy. The license contains territorial restrictions, redistribution conditions, and use restrictions. Review `LICENSE` and `NOTICE` before downloading, using, or redistributing these files.

This repository does not grant additional rights or remove any upstream restriction. Users and distributors are responsible for determining whether their access, use, and redistribution are authorized.

## Status

- Experimental: **yes**
- Trained LoRA: **no**
- Generation-tested: **no**
- Structurally parsed by ComfyUI: **yes**
- Affiliated with or endorsed by MiniMax or Comfy Org: **no**
