# quifox/H3-Turbo-Runtime

Источник: https://huggingface.co/quifox/H3-Turbo-Runtime · скачано 2026-11-28
Загрузок: 0 · лайков: 1 · базовая: ["MiniMaxAI/MiniMax-H3","lightx2v/Minimax-h3-Turbo"] · теги: comfyui,turbo · обновлено: 2026-11-22T14:50:51.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
  - MiniMaxAI/MiniMax-H3
  - lightx2v/Minimax-h3-Turbo
pipeline_tag: text-to-video
language:
  - en
  - zh
tags:
  - video-generation
  - image-to-video
  - native-audio-video
  - low-vram
  - consumer-gpu
  - fast-inference
  - inference-optimization
  - comfyui
  - pytorch
  - triton
  - w4a8
  - turbo
---

# H3 Turbo Runtime — Low-VRAM Inference Engine for MiniMax H3

[GitHub Source Repository](https://github.com/quifox/H3-Turbo-Runtime) · [Upstream Model Registry](MODELS.md) · [Documentation](README.md) · [English Documentation](README.en.md)

**Optimized inference runtime and compute engine, not a newly trained model.**  
H3 Turbo Runtime provides low-VRAM, kernel-fused execution for MiniMax H3 audiovisual generation. It integrates community W4A8 DiT quantization, LightX2V few-step Turbo LoRA distillation, and fused Triton execution kernels, enabling native audiovisual generation across consumer and datacenter GPUs as well as ComfyUI and CozyUI workflows.

---

## 1. Upstream Weight Lineage & External Dependencies

This repository does **not** distribute model weights. All inference relies on upstream community weights hosted on Hugging Face:

| Component | Pinned Checkpoint File | Upstream Author | Hugging Face Repository | Size | SHA-256 Digest |
|---|---|---|---|---:|---|
| **DiT Backbone (FL2VA)** | `MiniMax-H3-fl2va-pruned-comfy-w4a8.safetensors` | joeygambino | [joeygambino/MiniMax-H3-comfy-native-fl2va](https://huggingface.co/joeygambino/MiniMax-H3-comfy-native-fl2va) | 12.54 GB | `18ce05a9e588618db5ae6dca380f242bf8041d9d47f3ffb5cd80d9dec9f9b1ec` |
| **Few-Step Adapter** | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | lightx2v | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | 1.96 GB | `2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e` |
| **Video VAE** | `minimax_h3_video_vae_int8_convrot.safetensors` | Kijai | [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental) | 3.17 GB | `9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410` |
| **Audio VAE** | `minimax_h3_audio_vae_fp32.safetensors` | Comfy-Org | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | 0.61 GB | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| **Text Conditioning** | `qwen3vl_4b_int8_convrot.safetensors` | Stick9190 | [Stick9190/qwen3vl_4b_int8_convrot](https://huggingface.co/Stick9190/qwen3vl_4b_int8_convrot) | 5.37 GB | `ebbb3efdf4a6e1b766e32f6dc4fa92af9530edede0968c070b3e604aeca413aa` |
| **Projection Layer** | `mmh3-4b-ClipProj-v3.1.safetensors` | NicoLab28 | [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) | 26.3 MB | `0184e5c8d666a131962506d21949c2d8a8c6f33445b7b5e347e9a7e0a5baa819` |

For Ref2VA (multi-character reference to video), the backbone and LoRA are swappable with `Kutches/minmax` and `lightx2v/Minimax-h3-Turbo` (768p Ref2V LoRA) as documented in [MODELS.md](MODELS.md).

---

## 2. Empirical Hardware Benchmarks

All metrics reflect actual recorded wall times and hardware peak allocations:

| Task & Setup | Target GPU | Backend Strategy | Steps | Completed Native AV Wall Time | Peak Device VRAM |
|---|---|---|---:|---:|---:|
| 5s FL2VA (832×480) Upstream Baseline | Tesla T4 16GB | Default PyTorch (Unmodified) | 8 | 2,287.90 s | 14,649 MiB |
| 5s FL2VA (832×480) Stream Runtime | Tesla T4 16GB | Stream (Paging) | 8 | 670.12 s (stage sum) | 4,181 MiB |
| 5s FL2VA (832×480) Stream Runtime | Tesla T4 16GB | Stream (Paging) | 4 | 227.18 s (cached cond) | 4,043 MiB |
| 5s FL2VA (832×480) Fast Runtime | RTX 4060 Laptop 8GB | Ada Fast (Fused 148 Regs) | 4 | 161.54 s (cached cond) | 5,695 MiB |
| 5s FL2VA (960×544) Fast Runtime | RTX 4060 Laptop 8GB | Ada Fast (Fused 148 Regs) | 4 | 217.07 s | 7,088 MiB |
| 15s Single-pass FL2VA (832×480) | RTX 4060 Laptop 8GB | Ada Fast (Fused 148 Regs) | 4 | 694.70 s | 6,995 MiB |

---

## 3. Technical Architecture & Memory Lifecycle

1. **Dual-Backend Strategy**:
   - **Stream Backend**: Uses memory paging and streaming weight transfers. Operates under constrained memory footprints (~4.0–4.2 GiB VRAM), allowing broad execution on 6GB/8GB/12GB/16GB GPUs across Turing, Ampere, Ada, and datacenter hardware.
   - **Fast Backend**: Utilizes custom Triton fused kernels (FC1 + FC2 and QK-RoPE) with explicit register budgeting for maximum execution throughput.
2. **Staged Memory Execution**:
   - The multimodal conditioning frontend (Qwen3-VL INT8 Prefix-24 and ClipProj) executes and completely unloads before DiT backbone loading (~1.6 GiB torch peak).
   - Video and audio latents are sampled together, then decoded sequentially: Video VAE tiled INT8 decode runs first, followed by Audio VAE FP32 decode.
3. **Multi-Resolution Support**:
   - Native spatial patches enforce `width % 16 == 0` and `height % 16 == 0`. Supported and empirically verified aspect ratios include 832×480 (16:9), 960×544 (16:9), 480×832 (9:16 vertical), 624×624 (1:1), and 720×544 (4:3).
4. **Frame Geometry & Alignment**:
   - Video duration aligns to H3 temporal compression formula $N = 17k + 5$ ($k \in \mathbb{N}$) at 24fps. For example: 5s = 124 native frames (120 exported), 10s = 243 native frames (240 exported), 15s = 362 native frames (360 exported).
5. **Hardware Adaptation Guide**:
   - For tuning or porting to alternative NVIDIA architectures or custom memory budgets, follow the backend contract in `.agents/skills/h3-gpu-backend-porting/`.

---

## 4. Quickstart & Reproducibility

### Automated Checkpoint Download & Hash Verification
```bash
git clone https://github.com/quifox/H3-Turbo-Runtime.git
cd H3-Turbo-Runtime
uv sync

# Fetch all 6 upstream weights with resume support and SHA-256 verification
uv run python scripts/fetch_weights.py --fetch
```

### CLI Inference
```bash
# Verify environment and weights
uv run python deploy.py check --comfy /path/to/ComfyUI --models weights/

# Render native audiovisual output
uv run python deploy.py render \
  --comfy /path/to/ComfyUI \
  --models weights/ \
  --condition path/to/condition.safetensors \
  --duration 5 \
  --width 832 \
  --height 480 \
  --steps 4 \
  --strategy fast \
  --out runs/demo
```

---

## 5. Licensing and Academic Scope

- **Runtime Engine Source Code**: Licensed under [GPL-3.0-only](https://github.com/quifox/H3-Turbo-Runtime/blob/main/LICENSE).
- **Model Checkpoints**: Subject to the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Users are responsible for complying with upstream territorial and commercial usage restrictions.
- **Numerical Verification Scope**: "Bit-exact" verification refers strictly to our unaccelerated internal 148 4-step reference baseline vs. accelerated runtime kernel outputs under identical hardware and weight conditions. Cross-device and upstream comparisons remain empirical side-by-side observations.
