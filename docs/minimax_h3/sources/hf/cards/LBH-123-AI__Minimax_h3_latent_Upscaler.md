# LBH-123-AI/Minimax_h3_latent_Upscaler

Источник: https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler · скачано 2026-11-28
Загрузок: 452149 · лайков: 524 · базовая:  · теги: comfyui · обновлено: 2026-11-10T13:16:50.000Z

---

---
license: apache-2.0
tags:
  - latent-upscaler
  - latent-space-upscaling
  - minimax-h3
  - video-super-resolution
  - super-resolution
  - comfyui
  - safetensors
  - pytorch
  - 3d-convolution
  - temporal-convolution
  - latent-diffusion
---

<p align="center">
  <a href="/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/README.md"><strong>English</strong></a> ·
  <a href="/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/README_zh.md">中文</a>
</p>

<p align="center">
  <a href="https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FLBH-123-AI%2FMinimax_h3_latent_Upscaler&amp;query=%24.downloads&amp;label=downloads%20(last%2030d)&amp;color=blue&amp;logo=huggingface" alt="Downloads (last 30 days)"></a>
  <a href="https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FLBH-123-AI%2FMinimax_h3_latent_Upscaler&amp;query=%24.likes&amp;label=likes&amp;color=red&amp;logo=huggingface" alt="Likes"></a>
</p>

# Minimax H3 Latent Upscaler

> ### 🆕 2026-11-10 — 2D v1 released
>
> Added **`minimax_h3_latent_upscaler_2d_conv_v1`**:
>
> - **Half the parameters** — 182,140,312 params, **52.8%** of the 3D release (345,280,216), with checkpoint files roughly half the size.
> - **Beats the 3D release on most scales** — lower latent MSE on **32 of 36** scales; across all scales it cuts latent MSE by **84.7%** versus trilinear (3D v1: 61.1%).
> - **6-step adjustable sharpness** — `tier` 1 is the most faithful, 6 the sharpest. The 3D release has no such control (its sharpness sits at roughly 2D tier 1).

Neural latent-space upscaler for **Minimax H3** video generation. Works directly on Minimax H3's
**24-channel VAE latents** to upscale spatial resolution (H×W) while preserving the time dimension.

## Purpose

This model is designed to **accelerate high-resolution H3 video generation**:

1. Generate video at low resolution — far fewer latent tokens, much faster.
2. Upscale the latent in-place with this learned upscaler.
3. Re-sample / refine at the target resolution to recover detail.

By skipping the expensive `decode → pixel upscale → encode` round-trip through Minimax H3's
heavy ~5B-parameter VAE, this pipeline saves a significant amount of generation time. It also
avoids the **ghosting / double-image artifacts** that naive latent interpolation
(bilinear/bicubic) introduces.

## 📸 Examples

**Video upscale comparison** — click to play:

<video src="https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/examples/Minimax_h3_latent_Upscaler_001.mp4" controls width="640"></video>

*(If the player doesn't render, [download the video here](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/examples/Minimax_h3_latent_Upscaler_001.mp4).)*

**Image upscale comparison**:

![Minimax H3 latent upscale comparison](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/examples/Minimax_h3_latent_Upscaler_002.jpg)

## 2D v1 vs 3D v1

**Test setup**: 2,500 validation pairs (the same frozen split used during training); latent-domain
metrics computed on the normalized latents; pixel metrics decoded through the actual Minimax H3 VAE.
Coverage is **36 scales from 1.1× to 4.0×** (reference report generated 2026-11-06; 3D v1 degrades
noticeably below 1.5×, see the per-range table).

![Per-scale latent MSE across all 36 scales](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/examples/3D_Conv_v1_vs_2D_Conv_v1.png)

> **Above** (Part A of the full report): raw latent MSE at each of the 36 scales from 1.1× to 4.0×
> — grey is trilinear, orange is 3D_Conv_v1, blue is 2D_Conv_v1 at tier 1.
> 2D sits at the bottom for almost every scale; 3D only pulls back ahead at the four scales it was
> natively trained on (2×, 2.25×, 2.5×, 4×), which is why the orange line is so jagged.

📊 **Full comparison report**: [`examples/3D_Conv_v1_vs_2D_Conv_v1.html`](/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/examples/3D_Conv_v1_vs_2D_Conv_v1.html)
— per-scale data for all 36 scales, side-by-side images, the sharpness ladder and temporal consistency.
(HTML file, 9.3 MB; Hugging Face does not render HTML inline, so download it from that page and open
it locally in a browser to see the full charts.)

### Overall

| Metric | Trilinear | 3D_Conv_v1 | 2D_Conv_v1 |
| :--- | ---: | ---: | ---: |
| Latent MSE (mean improvement over trilinear, 36 scales) | baseline | **+61.1%** | **+84.7%** |
| Decoded PSNR (mean over 36 scales, higher is better) | 23.54 | 29.30 | **30.47** |
| Sharpness (gradient ratio, 1.0 = matches ground truth) | 1.429 ¹ | 0.869 | **0.897** |
| Chroma shift (lower is better) | 28.3% | 17.9% | **16.4%** |
| Parameters | — | 345,280,216 | **182,140,312 (52.8%)** |
| Adjustable sharpness tiers | — | none | **6 (t1–t6)** |

¹ Trilinear's sharpness is usually above 1.0 — that is interpolation ringing, not extra real detail.

### By scale range (latent MSE improvement over trilinear)

| Scale range | Scales | 3D_Conv_v1 | 2D_Conv_v1 |
| :--- | ---: | ---: | ---: |
| 1.1× – 1.4× | 5 | +28.1% | **+80.3%** |
| 1.5× – 2.0× | 7 | +60.9% | **+83.1%** |
| 2.1× – 3.0× | 12 | +64.3% | **+85.6%** |
| 3.1× – 4.0× | 12 | +70.3% | **+87.1%** |

### The only 4 scales where 3D v1 wins

| Scale | 3D advantage over 2D |
| :--- | ---: |
| 2× | 11.0% better |
| 2.25× | 7.5% better |
| 2.5× | 12.5% better |
| 4× | 30.5% better |

Those four are exactly the scales 3D v1 was trained on; the other 32 scales go to the 2D release.
**Which one to pick**: use 3D only when you need maximum quality at 2× / 2.25× / 2.5× / 4×.
For every other scale and for general use, prefer 2D.

### 2D v1 sharpness ladder

| Tier | Latent MSE | PSNR | Sharpness (grad ratio) | Chroma shift |
| :--- | ---: | ---: | ---: | ---: |
| t1 (most faithful) | **0.06835** | **31.18** | 0.947 | **14.81%** |
| t2 | 0.07605 | 30.71 | 1.092 | 15.61% |
| t3 | 0.08412 | 30.21 | 1.211 | 16.25% |
| t4 | 0.09880 | 29.54 | 1.345 | 17.29% |
| t5 | 0.11783 | 28.83 | 1.484 | 18.51% |
| t6 (sharpest) | 0.13624 | 28.16 | 1.622 | 19.68% |
| *3D_Conv_v1 (reference)* | *0.06209* | *31.15* | *0.904* | *14.57%* |

Higher tiers are sharper, at a small cost in MSE / PSNR / chroma. The 3D release corresponds to
roughly **t1**, and cannot be adjusted.

### Temporal consistency

Frame-diff ratio = mean absolute adjacent-frame difference of the output ÷ that of the ground truth.
**1.000 = frame-to-frame dynamics identical to the reference**; < 1 means the output is "quieter"
than the truth, > 1 means "jitterier" (flicker or amplified noise).

| Domain | LR as-is | 2D_v1 · t1 | 3D_v1 | Trilinear |
| :--- | ---: | ---: | ---: | ---: |
| Latent (mean of 1.5×/2×/3×/4×) | 1.038 | **0.850** | 0.867 | 0.900 |
| Pixel (same samples) | — | **0.869** | 0.880 | 1.046 |

Both releases beat trilinear; the 2D release is marginally more stable over time.

## Files in this repository

Each release keeps its checkpoints in its own versioned folder. Filenames **always carry `2d` / `3d`
and the version tag**, because ComfyUI loads every model from a single flat directory
(`models/latent_upscale_models/`) — once the file is on your disk, **the filename is the only
identifier**. Please do not rename them.

### 2D release (`minimax_h3_latent_upscaler_2d_conv_v1/`, recommended)

| File | Format | Precision | Size | Notes |
| :--- | :--- | :--- | ---: | :--- |
| `..._2d_conv_v1_bf16.safetensors` | SafeTensors | bfloat16 | ~364 MB | Best on GPUs with native bf16 |
| `..._2d_conv_v1_fp16.safetensors` | SafeTensors | float16 | ~364 MB | **Recommended default**, numerically identical to fp32 |
| `..._2d_conv_v1_fp32.pth` | PyTorch | float32 | ~729 MB | Reference / conversion / bit-exact checks |

### 3D release (`minimax_h3_latent_upscaler_3d_conv_v1/`)

| File | Format | Precision | Size | Notes |
| :--- | :--- | :--- | ---: | :--- |
| `..._3d_conv_v1_bf16.safetensors` | SafeTensors | bfloat16 | ~691 MB | Fastest on Ampere/Ada GPUs |
| `..._3d_conv_v1_fp16.safetensors` | SafeTensors | float16 | ~691 MB | Best speed / memory balance |
| `..._3d_conv_v1_fp32.pth` | PyTorch | float32 | ~1.38 GB | Highest numerical accuracy |

`config.json` inside each release folder describes that release's architecture.

`config.json` at the repository root is the **family index**: it records what every release shares
(the 24-channel H3 latent space and its normalization, the supported scale range, the ComfyUI node
mapping) and lists each release's status and checkpoint paths.

The ComfyUI node infers the architecture from the state dict on load and does **not** read either
file — they are machine-readable specifications for downstream tooling.

### Versioning

Each release gets its own folder, and older releases stay downloadable and unchanged.

## Usage

Use with the companion ComfyUI custom node:

**[LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)**

1. Install the custom node into `ComfyUI/custom_nodes/`.
2. Place the downloaded checkpoint in:
   ```text
   ComfyUI/models/latent_upscale_models/
   ```
3. Add either **"Minimax H3 Latent Upscaler (2D)"** or **"Minimax H3 Latent Upscaler (3D)"**
   from the `video/MinimaxH3` menu, connect a `LATENT`, pick the model, and set `scale`.
   - The **2D node** also exposes a `tier` parameter (1–6) controlling output sharpness:
     1 = most faithful, 6 = sharpest.
   - The **3D node** has no such parameter.

Supported upscale factors: **1.0× – 4.0×** (continuous, 0.1 step; default **2.0×**).

## Training Data

The two releases use separate datasets. All figures below are measured from disk
(391,710 `.latent` files, no parse failures).

### 2D release (`2d_conv_v1`)

**230,747 paired latent samples** (high-resolution latent + low-resolution latent of the same source),
built from video material.

| Component | Pairs | Notes |
| :--- | ---: | :--- |
| Main data (tier-1) | 146,945 | General multi-scale upscaling training |
| Tier data (tier 1–6) | 83,802 | 13,967 per tier, used for sharpness-tier training; ~80% are single-frame static shots |

Another 50 high-resolution samples have no matching low-resolution file and are excluded.

#### Scale distribution (main data, 146,945 pairs, 36 scales)

| Scale | Pairs | Share | Scale | Pairs | Share |
| :--- | ---: | ---: | :--- | ---: | ---: |
| 1.10× | 3,249 | 2.2% | 2.50× | 15,000 | 10.2% |
| 1.25× | 500 | 0.3% | 3.00× | 10,824 | 7.4% |
| 1.50× | 11,150 | 7.6% | 3.50× | 1,400 | 1.0% |
| **2.00×** | **32,624** | **22.2%** | **4.00×** | **10,500** | **7.2%** |

The full distribution has **three main peaks** — 2.00× at 22.2%, 2.50× at 10.2%, 4.00× at 7.2% —
plus two secondary peaks at 1.50× (7.6%) and 3.00× (7.4%); the remaining 0.25-step scales hold about
3,300 pairs each (2.25%). The six quarter-step scales (1.25× / 1.75× / 2.25× / 2.75× / 3.25× / 3.75×)
have only 400 pairs each (0.27%) and act as supplementary samples.

**Share by range**: < 1.5× **9.3%** | 1.5–2.5× **48.2%** | 2.5–3.5× **30.6%** | 3.5–4.0× **11.9%**

> 2.00× alone accounts for more than a fifth of the data, while samples below 1.5× are comparatively
> scarce (9.3%).

### 3D release (`3d_conv_v1`)

**~80,000 pairs** (~70,000 video pairs + ~8,000 2K image pairs).

## Architecture

- **2D release**: Conv2d backbone with shared temporal blocks (temporal Conv1d); 640 base channels,
  12+12 blocks; an additional `tier` (sharpness) conditioning embedding.
- **3D release**: pure 3D-convolution backbone with temporal convolution and trilinear interpolation;
  512 base channels, 12+12 blocks.

Both draw on and reference the **LTX 2.3 Spatial Upscaler**
(`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`) and the neural-latent-upscaling approach
pioneered by [Ttl / ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale).

## Changelog

| Date | Change |
| :--- | :--- |
| **2026-11-10** | Released **2D v1** (182M params, 6 sharpness tiers, better than 3D v1 on 32 of 36 scales); added the side-by-side comparison to this card. |
| 2026-09-17 | Moved checkpoints into versioned folders; added the family-index `config.json` and bilingual model cards. |

## License

Apache-2.0
