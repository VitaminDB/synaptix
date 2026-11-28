# LBH-123-AI__Comfyui_Minimax_h3_latent_Upscaler

Source: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
Fetched: 2026-11-28

Description: Neural latent upscaler for Minimax H3 (24ch). Bypasses costly 5B-param VAE decode/encode. Upscale low-res latents directly, then refine. Accelerates high-res video gen, outperforms naive interp.
Stars: 867  Updated: 2026-11-10T13:09:05Z



---

## FILE: README.md (https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/HEAD/README.md)

<p align="center">
  <a href="./README.md"><strong>English</strong></a> ·
  <a href="./README_zh.md">中文</a>
</p>

<div align="center">

# ComfyUI Minimax H3 Latent Upscaler

**Neural Latent Upscaler for Minimax H3 Video Generation**
Learned · High-fidelity · 2D & 3D Variants

</div>

## 📰 News

- [2026-11-10] 🚀 **New 2D-conv release — `minimax_h3_latent_upscaler_2d_conv_v1`**: a 2D conv backbone (Conv2d) with interleaved temporal Conv1d — **182M parameters** (3D: 345M, roughly half) and a **~364 MB** fp16 checkpoint (3D: ~691 MB). Across all 36 scales it cuts latent MSE by **84.7%** vs trilinear (3D: 61.1%); below 1.5× it still delivers ~**80%** (3D only 28%); pixel-domain PSNR averages **30.47 dB** (3D: 29.30 dB). It also offers **6 sharpness tiers** (the 3D model has none) and covers the full 1.1×–4.0× range. The 3D model only leads on the scales it was trained on (2×/2.25×/2.5×/4×), and that lead is small: measured on the shared trilinear baseline, 2D lands at **83.0%–86.4%** and 3D at **84.7%–89.1%** — a **1.0–3.3 percentage-point** gap (absolute MSE difference 0.004–0.024). The "3D is 7.5%–30.5% better" figure is a relative difference on top of that already tiny residual, not a difference in overall quality. Temporal consistency is only marginally better too (frame-diff ratio 0.867 vs 0.850). See the [full comparison report](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/examples/3D_Conv_v1_vs_2D_Conv_v1.html) and the per-scale chart below.
- [2026-11-10] 🧩 **2D node brought up to parity with 3D**: three resize modes (`scale by multiplier` / `target dimensions` / `megapixels`), `rocm` backend, `force_unload` VRAM offload, pixel-grid `align`; plus a new **Sharpness** tier selector (1–6) and **Temporal Chunking** (`chunked` / `single pass`).

- [2026-08-28] 🚀 **New node combo — MMH3 Split Upscale**: three new nodes (`MMH3 Temporal Split Params`, `MMH3 Spatial Split Params`, `MMH3 Split Upscale`) perform a tiled, hires-fix style latent re-sampling upscale for H3 video. Decomposed and optimized from the **Comfyui-MMH3-UltimateUpscale** project — splitting the AV latent into temporal chunks and spatial tiles, resampling each piece, and stitching with seam denoise, two-level color matching and temporal anchors. Key upgrades vs. original: more stable seams (`seam_denoise` + probe-gated seam polish), zero color drift (two-level matching + first-block/every-chunk pin-source), triple temporal anchors (frame-0/motion/identity), easier use (spatial params simplified to overlap/fade percentages snapping to the latent grid), and a lighter footprint (built-in upscale model removed in favor of an external pre-upscaled latent).
- [2026-08-26] 🚀 **3D node memory & UX optimizations**: zero-copy model loading plus ComfyUI's native `soft_empty_cache` and explicit `.contiguous()` calls reduce RAM/VRAM spikes during load and inference; the post-inference CPU offload is now an optional `force_unload` toggle (default on); `enable_chunking` is renamed to `enable_temporal_chunking` and its effective chunk stride raised from 24 to 32; model files can now live in subfolders (PR #30); `target dimensions` and `megapixels` upper limits raised to 8192 px and 16 MP.
- [2026-08-23] 🚀 **3D node improvements**: added an `enable_chunking` toggle (turn off for short clips to use full-context inference); fixed temporal-chunk edge artifacts with replicate padding and weighted overlap blending, eliminating end-frame flicker; added ROCm (AMD GPU) backend support via the new `rocm` device option.
- [2026-08-21] 🚀 **3D node optimization**: The model is automatically offloaded to CPU after execution to free VRAM for subsequent second-pass sampling; width and height are independently aligned to the align grid (default 32), fixing the bottom light band issue; normalization/denormalization is changed from in-place operations to standard operations; temporal chunking is retained to support long videos.
- [2026-08-19] 🚀 **3D node overhaul**: all three resize modes (`scale by multiplier`, `target dimensions`, `megapixels`) merged into a single node; fixed aspect-ratio mismatch in certain modes and edge artifacts at specific sizes; added a new example workflow and expanded the usage notes.
- [2026-08-18] 🔥 **Precision selector**: both 2D and 3D nodes now support `fp32` / `fp16` / `bf16` inference.
- [2026-08-17] 🎉 **Initial release**: Minimax H3 Latent Upscaler 2D + 3D nodes with bilingual README and inline examples.

A custom ComfyUI node that upscales **Minimax H3** VAE latents (24 channels) with a trained
neural network instead of naive interpolation. Its main purpose is to **accelerate
high-resolution video generation** and improve quality:

- **Skip the slow decode → pixel-upscale → encode round-trip.** Minimax H3 ships a heavy
  ~5B-parameter VAE, so decoding and re-encoding latents is expensive. Upscaling directly in
  latent space avoids that costly round-trip entirely.
- **Enable a faster generation pipeline:** generate at low resolution (far fewer latent tokens),
  upscale the latent with this node, then refine at the target resolution.

It also **avoids the ghosting / double-image artifacts** that naive latent interpolation
(bilinear/bicubic) introduces, and plays a role similar to the latent upscaler in **LTX2.3**.

⚠️ This saves **time, not VRAM** — the refinement pass still runs at the target resolution, so
peak memory is comparable to generating high-res directly. The win is purely speed.

Two node variants are provided, both registered under the `video/MinimaxH3` category. Their
parameters are **almost identical** — the 2D node adds a `sharpness` selector and exposes temporal
chunking as a dropdown, the 3D node as a boolean toggle:

- **Minimax H3 Latent Upscaler (2D)** — a 2D conv backbone with interleaved temporal convolutions.
  Spatial (H×W) upscaling only; the time dimension is preserved. Lightweight and fast. Pairs with
  the `*_2d_conv_v1` checkpoint and supports 6 sharpness tiers.
- **Minimax H3 Latent Upscaler (3D)** — a fully 3D-convolution backbone (3D conv blocks +
  TemporalConv + trilinear interpolation). Processes the spatiotemporal volume jointly for slightly
  better temporal coherence, at ~1.9× the parameters and compute. Pairs with `*_3d_conv_v1`.

> Weights and backbone must match: the 2D node loads 2D checkpoints, the 3D node loads 3D
> checkpoints. Picking the wrong one fails loudly with a structure mismatch instead of silently
> producing garbage.

Both nodes ship **three resize modes**:

- `scale by multiplier` — classic `scale` factor (1.0×–4.0×).
- `target dimensions` — directly set target pixel `width`/`height`.
- `megapixels` — set a target total pixel count in megapixels (e.g. `1.2`); keeps aspect ratio.

The `target dimensions` and `megapixels` modes align the output to a configurable pixel grid and
derive the effective scale automatically, so any target between 1.0×–4.0× works.

> Both nodes support **upscaling only** (`effective scale >= 1.0`). `scale = 1.0` returns the input
> unchanged; an effective scale below `1.0` raises an error.

### Sharpness tiers (2D checkpoints only)

Checkpoints with a tier channel expose a `Sharpness` parameter (1–6):

- **1** = closest to the reference (softest, lowest MSE) — the default and the recommended setting.
- **6** = sharpest (most detail, furthest from the exact target).

Older checkpoints without a tier channel silently ignore it (treated as 1) and print a one-line
notice in the console.

> Sharpness exists only in the `2d_conv_v1` weights: `3d_conv_v1` has no adjustable sharpness, and
> its output sharpness measures equivalent to 2D tier t1.

### MMH3 Split Upscale (combo)

> **Origin:** this combo is a **decomposed and re-optimized version of the
> Comfyui-MMH3-UltimateUpscale project**. The original monolithic node was broken into three composable nodes (temporal split /
> spatial split / main upscaler), and the enhancements below were layered on top of the original tiling
> logic to improve seam stability, color consistency, temporal coherence, usability, and footprint.

A separate three-node combo that does a **tiled, hires-fix style re-sampling upscale** directly
inside the diffusion sampler — instead of a pre-trained upscaler network. It takes the H3 **AV
latent** (nested video 24ch + audio 32ch), splits it into **temporal chunks** and **spatial tiles**,
runs the sampler on each piece, then stitches them back seamlessly with seam-denoise, two-level color
matching and temporal anchors so seams and ghosting don't appear. The two `* Split Params` nodes
configure the splitting and feed the main `MMH3 Split Upscale` node (both optional — leave them
unconnected for a single full-frame pass). It requires a `model` + `conditioning` + `sampler`/`sigmas`,
i.e. it re-runs sampling at the higher resolution.

**Key enhancements vs. the original project:**

- **🧩 More stable seams (Seam / Ghosting):** a seam-neighborhood denoise cap (`seam_denoise`) plus a probe-gated second-pass seam polish (`seam_polish`) remove seams and ghosting; even under high denoise + fast motion, moving objects are no longer sliced ("broken limbs").
- **🎨 Zero color drift (Color Zero-Drift):** spatial + temporal two-level color matching, with a global color reference pinned to the **first block** and to **every chunk** (`grade_pin`) — eliminates inter-tile flicker and the localized cyan/green tint (e.g. top-left corner).
- **⏱️ Stronger temporal continuity:** a **triple temporal anchor** — `frame-0 anchor` + `motion anchor` + `identity anchor` — automatically backs up identity consistency under high denoise, so subjects don't drift.
- **🖱️ Easier to use:** spatial parameters cut from 9 fields down to **percentages** (overlap ratio / fade ratio) that auto-snap to the latent grid; `negative` sits right next to `conditioning` for a more intuitive layout.
- **🪶 Lighter footprint:** the built-in upscale model was removed in favor of an **external pre-upscaled latent** input — more controllable VRAM, and a node that stays focused on its job.

**Modules added / optimized (relative to the original project):**

| Module | What it adds / optimizes |
| :--- | :--- |
| **Prevention (预防)** | Freeze-prefill overlap band + **triple temporal anchors** (frame-0 anchor + motion anchor + identity anchor) + cross-fade stitching — stops seams and forks from forming in the first place. |
| **Correction (校正)** | **Two-level (spatial + temporal) color matching** + first-block source reference + per-chunk global pin-source (`grade_pin`) — keeps color & brightness consistent across tiles and chunks, killing inter-tile flicker and localized cyan/green tint. |
| **Anti-forking (抗分叉)** | `seam_denoise` cap — under high denoise + fast motion, the seam neighborhood is re-sampled at *medium* denoise so moving objects aren't sliced ("broken limbs"); combined with the probe-gated second-pass seam polish, seams and ghosting are both eliminated (suggested 0.5–0.8; 1.0 = off). |
| **Fixes (修复)** | **Per-tile independent Guider** + cropped keyframes, and a **fixed probe-gated seam polish** (`seam_polish`) — each tile gets correctly-scoped conditioning/keyframes, and the polish gating no longer over/under-applies. |
| **Simplification (简化)** | `overlap` / `fade` are now **percentage parameters** (overlap ratio / fade ratio, resolution-independent) instead of absolute pixels, and auto-snap to the latent grid; `negative` sits next to `conditioning` for a more intuitive layout. |

---

## 📸 Examples

**Video upscale comparison**

<video src="examples/Minimax_h3_latent_Upscaler_001.mp4" controls width="640"></video>

**Image upscale comparison**

![](examples/Minimax_h3_latent_Upscaler_002.jpg)

**2D vs 3D across all 36 scales** — full MSE / PSNR / sharpness / temporal charts:
[`examples/3D_Conv_v1_vs_2D_Conv_v1.html`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/examples/3D_Conv_v1_vs_2D_Conv_v1.html).

---

## 📁 Project Structure

```text
Comfyui_Minimax_h3_latent_Upscaler/
├── examples/
│   ├── Minimax_h3_latent_Upscaler_001.mp4
│   ├── Minimax_h3_latent_Upscaler_002.jpg
│   └── 3D_Conv_v1_vs_2D_Conv_v1.png     # per-scale MSE chart (2D vs 3D)
├── workflow_templates/
│   └── minimax_h3_r2v_Latent Upscaler example workflow.json  # example workflow for ComfyUI templates
├── nodes/
│   ├── __init__.py                       # merges 2D/3D/Split node mappings
│   ├── minimax_h3_latent_upscaler_2d.py  # 2D backbone + temporal convs (3 resize modes + sharpness + chunking)
│   ├── minimax_h3_latent_upscaler_3d.py  # pure 3D convolution with 3 resize modes
│   └── MMH3_Split_Upscale.py             # MMH3 Split Upscale combo (decomposed from Comfyui-MMH3-UltimateUpscale)
├── README.md
├── README_zh.md
└── __init__.py
```

> The model weights are **not** included in this repo. Place them in your ComfyUI models folder
> (see Model Placement below).

---

## 🚀 Key Features

- ✅ **Learned latent upscaling** — neural network trained for Minimax H3 latents, far sharper
  than bilinear/bicubic interpolation.
- ✅ **Two backbones, two checkpoints** — `2d_conv_v1` covers every scale at half the size with
  adjustable sharpness; `3d_conv_v1` is more accurate on 2×/2.25×/2.5×/4× and marginally better on
  temporal consistency (frame-diff ratio 0.867 vs 0.850). Node and checkpoint pair one-to-one.
- ✅ **Three ways to set output size on both nodes** — `scale by multiplier`, `target dimensions`,
  or `megapixels`, all with pixel-grid alignment and aspect-ratio lock.
- ✅ **Sharpness tiers** — one checkpoint spans 1 (most faithful) to 6 (sharpest), so you don't need
  a separate file per look.
- ✅ **Temporal chunking** — `chunked` caps VRAM on long clips, `single pass` runs the whole clip in
  one forward pass; older workflows keep working.
- ✅ **Tiled hires-fix upscaler (MMH3 Split Upscale combo)** — re-sample the H3 AV latent at higher
  resolution by splitting into temporal chunks + spatial tiles, with seam-denoise, two-level color
  matching and temporal anchors to avoid seams and ghosting.
- ✅ **24-channel Minimax H3 latent** — uses the exact per-channel mean/std normalization from
  training.
- ✅ **Auto architecture detection** — reads `in_channels`, block counts, temporal config and
  kernel size straight from the checkpoint; no manual config needed.
- ✅ **Robust weight loader** — supports `.safetensors` and `.pth`; auto-converts FP8→FP16;
  tolerates the `upscaler.` prefix in merged checkpoints.
- ✅ **Strict weight validation** — missing key layers raise an error instead of silently
  random-initializing (which would produce noise).
- ✅ **Flexible precision/device** — `cuda`/`rocm`/`cpu` and fp32/fp16/bf16 options.
- ✅ **Plug-and-play** — standard ComfyUI node, no changes to your existing workflow.

Inference forces attention **off** (`attn=False`) for speed and stability. Loaded models are
cached by `(name, device, precision)` so repeated runs stay cheap.

---

## 📦 Installation

1. Clone this repository into ComfyUI's `custom_nodes` folder:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git
   ```

2. Required dependencies (`torch`, `einops`, `safetensors`) are already present in a standard
   ComfyUI environment — no extra install needed.

3. Restart ComfyUI.

---

## 🤖 Model Placement

The nodes scan and load weights from:

```text
ComfyUI/models/latent_upscale_models/
```

Put your Minimax H3 latent upscaler checkpoint (`.safetensors` or `.pth`) there. It will appear
automatically in the node's `model_name` dropdown.

Pre-trained checkpoints are available at:
[huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

Every release ships three precisions — pick one that fits your GPU:

| Release | fp16 / bf16 | fp32 (.pth) | Parameters |
| :--- | :--- | :--- | ---: |
| `minimax_h3_latent_upscaler_2d_conv_v1` (2D backbone) | ~364 MB | ~729 MB | 182M |
| `minimax_h3_latent_upscaler_3d_conv_v1` (3D backbone) | ~691 MB | ~1.38 GB | 345M |

**fp16 is the recommended default**: it is numerically almost identical to fp32. Use `bf16` on GPUs
without fp16 fast paths (same size, slightly lower accuracy); `fp32` is the reference checkpoint for
conversion and bit-exact verification.

The loader auto-detects the architecture (channel count, block counts, temporal config, tier
channel), so **checkpoints are not interchangeable between the two nodes**: `2d_conv_v1` only
loads in the 2D node, `3d_conv_v1` only in the 3D node.

### Which checkpoint: 2D or 3D?

Measured on the same frozen 2,500-pair validation split:

| Metric (average over 36 scales) | Trilinear | 2D_Conv_v1 | 3D_Conv_v1 |
| :--- | ---: | ---: | ---: |
| latent MSE improvement vs trilinear | — | **84.7%** | 61.1% |
| Improvement below 1.5× | — | **~80%** | 28% |
| Pixel-domain PSNR (after VAE decode) | 23.54 dB | **30.47 dB** | 29.30 dB |
| Temporal consistency (latent frame-diff ratio, truth = 1.000) | 0.900 | 0.850 | **0.867** |
| Adjustable sharpness | — | **6 tiers** | none |
| Parameters / fp16 size | — | **182M / 364 MB** | 345M / 691 MB |

**Raw latent MSE across all 36 scales (lower is better)**

![3D_Conv_v1 vs 2D_Conv_v1 · per-scale latent MSE](examples/3D_Conv_v1_vs_2D_Conv_v1.png)

> Grey = trilinear, blue = `2D_Conv_v1` (t1), orange = `3D_Conv_v1`. 2D stays pinned to the floor as
> an almost flat line, holding 11%–15% of trilinear's error at every scale; 3D only dips below 2D at
> the scales it was trained on (2×/2.25×/2.5×/4×) and bounces back elsewhere — at 1.1× it nearly
> coincides with trilinear (0.133 vs 0.138), and at 1.3×/2.2× it only improves by 30%–45%. This chart
> is the most direct illustration of "2D wins coverage, 3D wins specific scales".

- **2D wins on coverage**: a large advantage below 1.5×, adjustable sharpness, half the size.
- **3D wins on the scales it was trained on, but the margin is small**: only 2×/2.25×/2.5×/4× flip,
  details in the table below.

**The four scales where 3D leads (common baseline: improvement vs trilinear)**

| Scale | Trilinear MSE | 2D_Conv_v1 MSE (improvement) | 3D_Conv_v1 MSE (improvement) | 3D vs 2D |
| :--- | ---: | ---: | ---: | ---: |
| 2× | 0.32384 | 0.05489 (+83.0%) | **0.04947 (+84.7%)** | 11.0% lower |
| 2.25× | 0.40259 | 0.05475 (+86.4%) | **0.05091 (+87.4%)** | 7.5% lower |
| 2.5× | 0.46205 | 0.06644 (+85.6%) | **0.05906 (+87.2%)** | 12.5% lower |
| 4× | 0.73592 | 0.10464 (+85.8%) | **0.08016 (+89.1%)** | 30.5% lower |

How to read it: against the shared trilinear baseline both models land in the **83%–89%** band, and
3D is only **1.0–3.3 percentage points** ahead (absolute MSE difference 0.004–0.024). The 7.5%–30.5%
column is a relative difference computed on that already-tiny residual, not a 30% quality gap.

**Temporal consistency** (latent frame-diff ratio; 1.000 = same frame-to-frame dynamics as the truth,
<1 = "quieter" than the truth, >1 = "jitterier"):

| Scale | LR as-is | Trilinear | 2D_Conv_v1 | 3D_Conv_v1 | 3D vs 2D |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1.5× | 1.019 | 0.880 | 0.848 | 0.857 | closer to truth by 0.009 |
| 2× | 1.031 | 0.869 | 0.900 | 0.908 | closer to truth by 0.008 |
| 3× | 1.032 | 0.904 | 0.892 | 0.898 | closer to truth by 0.006 |
| 4× | 1.069 | 0.946 | 0.761 | 0.805 | closer to truth by 0.044 |
| **Average** | 1.038 | 0.900 | 0.850 | **0.867** | **+0.017** |

All three average below 1.000, i.e. every output is a little "quieter" than the truth (some fine
detail or motion gets smoothed away). 3D's 0.867 sits closer to 1.000 than 2D's 0.850, and the gap
peaks at 4× (0.805 vs 0.761); between 1.5× and 3× it stays under 0.01, i.e. not visible. **So the
"better temporal consistency" claim only carries real weight at 4×.** (At 2× trilinear's 0.869 is
closest to 1.000, but that is interpolation ringing amplifying frame noise, not more faithful detail.)

---

## 🧩 Usage

Add the node you need from the `video/MinimaxH3` menu, connect a `LATENT`, pick the model, set the
resize mode / scale, and decode.

**Typical workflow:**
- **Quick preview:** `[Minimax H3 Latent] → [H3 Latent Upscaler] → [VAE Decode]`
- **High quality / time-saving (recommended):** `[Low-res Latent] → [H3 Latent Upscaler] → [Refine / Re-sample] → [VAE Decode]`

Compared to the naive approach `[Latent] → [VAE Decode] → [Pixel Upscaler] → [VAE Encode] → …`,
upscaling directly in latent space skips the expensive VAE decode/encode round-trip. Minimax H3's
~5B-parameter VAE makes decode and re-encode notably slow, so this is where most of the time is
saved. It also avoids the **ghosting / double-image artifacts** that direct latent interpolation
(bilinear/bicubic) causes.

⚠️ **VRAM expectations:** this node only saves time — refinement still runs at the target resolution,
so peak memory is close to generating high-res directly. Use `Temporal Chunking` to cap peaks on long
clips.

### Node Reference — 2D

| Parameter | Type | Default | Range / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | Input Minimax H3 latent (B,C,T,H,W) or (B,C,H,W) |
| `model_name` | dropdown | auto | scanned files | Checkpoint in `latent_upscale_models/` |
| `mode` | dropdown | `scale by multiplier` | `scale by multiplier` / `target dimensions` / `megapixels` | How the output size is chosen |
| `scale` | FLOAT | 2.0 | 1.0 – 4.0 (step 0.05) | Used when `mode` is `scale by multiplier` |
| `width` | INT | 1280 | 64 – 8192 (step 8) | Target pixel width (used by `target dimensions`) |
| `height` | INT | 704 | 64 – 8192 (step 8) | Target pixel height (used by `target dimensions`) |
| `megapixels` | FLOAT | 1.0 | 0.1 – 16.0 (step 0.1) | Target total megapixels (used by `megapixels`); keeps aspect ratio |
| `sharpness` | INT | 1 | 1 – 6 | **Sharpness.** 1 = closest to the reference (softest, lowest MSE); 6 = sharpest (most detail, furthest from the exact target). Only checkpoints with a tier channel respond; older ones fall back to 1 |
| `align` | INT | 32 | 1 – 512 | Pixel-grid alignment: output W/H are independently rounded to multiples of this value (strictly recommended: 32) |
| `temporal_chunking` | dropdown | `chunked` | `chunked` / `single pass` | **Temporal Chunking.** Chunked inference caps VRAM on long clips; `single pass` runs one forward pass for the whole clip (most VRAM) |
| `chunk_frames` | INT | 32 | 4 – 256 (step 4) | **Frames per chunk.** Only shown under `chunked`. Each chunk actually forwards this value + 48 frames of temporal context, so don't go too small if you have headroom |
| `force_unload` | BOOLEAN | True | True / False | Unload model to CPU after inference to free VRAM for subsequent nodes; disable if you reuse this node repeatedly |
| `device` | dropdown | cuda | cuda / rocm / cpu | Inference backend (ROCm needs a HIP-enabled PyTorch build) |
| `precision` | dropdown | fp16 | fp32 / fp16 / bf16 | Inference precision |

**Output:** `LATENT` — the upscaled latent, ready for VAE decode.

### Node Reference — 3D

| Parameter | Type | Default | Range / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | Input Minimax H3 latent (B,C,T,H,W) or (B,C,H,W) |
| `model_name` | dropdown | auto | scanned files | Checkpoint in `latent_upscale_models/` |
| `mode` | dropdown | `scale by multiplier` | `scale by multiplier` / `target dimensions` / `megapixels` | How the output size is chosen |
| `scale` | FLOAT | 2.0 | 1.0 – 4.0 (step 0.05) | Used when `mode` is `scale by multiplier` |
| `width` | INT | 1280 | 64 – 8192 (step 8) | Target pixel width (used by `target dimensions`) |
| `height` | INT | 704 | 64 – 8192 (step 8) | Target pixel height (used by `target dimensions`) |
| `megapixels` | FLOAT | 1.0 | 0.1 – 16.0 (step 0.1) | Target total megapixels (used by `megapixels`); keeps aspect ratio |
| `align` | INT | 32 | 1 – 512 | Pixel-grid alignment: output W/H are independently rounded to multiples of this value (e.g. 16/32/64) |
| `enable_temporal_chunking` | BOOLEAN | True | True / False | Split long videos into temporal chunks to cap VRAM; disable for short clips for full-context inference |
| `force_unload` | BOOLEAN | True | True / False | Unload model to CPU after inference to free VRAM for subsequent nodes; disable if you reuse this node repeatedly to avoid reload overhead |
| `device` | dropdown | cuda | cuda / rocm / cpu | Inference backend (ROCm needs a HIP-enabled PyTorch build) |
| `precision` | dropdown | fp16 | fp32 / fp16 / bf16 | Inference precision |

**Output:** `LATENT` — the upscaled latent.

> **Which node to pick?** Default to **2D**: full 1.1×–4.0× coverage, a big advantage at low scales,
> half the file size, adjustable sharpness. Only switch to **3D** when you consistently run
> 2×/2.25×/2.5×/4× and want the extra accuracy — its latent MSE is 7.5%–30.5% lower there (absolute
> difference 0.004–0.024; on the trilinear baseline, 2D ≈ 85% vs 3D ≈ 87%, a 1–3 point gap), and its
> temporal consistency is only marginally better.

### Node Reference — MMH3 Split Upscale (combo)

A tiled, **hires-fix style** re-sampling upscaler for H3 video, **decomposed and optimized from the
Comfyui-MMH3-UltimateUpscale project**. Unlike the 2D/3D nodes (which use a
pre-trained upscaler network), this combo re-runs the diffusion sampler on a higher-resolution
version of the latent, split into pieces so it fits in VRAM. It operates on the H3 **AV latent**
(nested video 24ch + audio 32ch) and requires `model` + `conditioning` + `sampler`/`sigmas`.

**`MMH3 Temporal Split Params`** — how to cut the clip along time (H3's native 5+17·m frame/token grid):

| Parameter | Type | Default | Range / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `chunk_frames` | INT | 73 | 5 – 100000 | Chunk length in frames (snapped to H3 token grid) |
| `temporal_overlap_frames` | INT | 22 | 0 – 100000 | Overlap between consecutive chunks |
| `anchor_strength` | FLOAT | 0.999 | 0.0 – 1.0 | Temporal anchor strength |
| `motion_anchor_frames` | dropdown | 22 | 0 / 5 / 22 / 39 | Motion anchor length in frames |
| `identity_anchor_frames` | INT | 24 | 0 – 240 | Spacing of identity anchors |

Output: `temporal_split_param` (custom type, feeds the main node).

**`MMH3 Spatial Split Params`** — how to cut each frame into tiles:

| Parameter | Type | Default | Range / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `tile_width` | INT | 512 | 64 – 16384 (step 32) | Tile width in pixels |
| `tile_height` | INT | 512 | 64 – 16384 (step 32) | Tile height in pixels |
| `overlap_ratio` | FLOAT | 0.25 | 0.0 – 0.90 | Overlap as a fraction of the tile |
| `fade_ratio` | FLOAT | 0.50 | 0.0 – 1.0 | Fade band width within the overlap |
| `min_tile_size` | INT | 256 | 0 – 16384 (step 32) | Minimum tile edge; avoids tiny edge tiles |
| `seam_denoise` | FLOAT | 1.0 | 0.1 – 1.0 | Seam-neighborhood denoise cap; <1 prevents fast objects being cut at seams (suggest 0.5–0.8) |

Outputs: `spatial_split_param` (custom type) + `grid_preview` (string, shows the computed tile/overlap grid).

**`MMH3 Split Upscale`** — the main orchestrator:

| Parameter | Type | Default | Range / Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `model` | MODEL | — | — | Diffusion model used for re-sampling |
| `conditioning` | CONDITIONING | — | — | Positive conditioning |
| `negative` | CONDITIONING | — | optional | Negative conditioning |
| `latent` | LATENT | — | — | Input H3 AV latent (nested video 24ch + audio 32ch); batch 1 only |
| `noise` | NOISE | — | — | Noise source for the sampler |
| `sampler` | SAMPLER | — | — | Sampler |
| `sigmas` | SIGMAS | — | — | Sigma schedule |
| `cfg` | FLOAT | 1.0 | 0.0 – 100.0 | Classifier-free guidance scale |
| `temporal_split_param` | custom | — | optional | From `MMH3 Temporal Split Params` |
| `spatial_split_param` | custom | — | optional | From `MMH3 Spatial Split Params` |
| `seam_polish` | dropdown | off | off / auto / all | Extra de-seam pass on detected seams |
| `color_match` | BOOLEAN | True | True / False | Two-level color matching across tiles/chunks |

**Output:** `LATENT` — the upscaled H3 AV latent, ready for VAE decode.

> **Notes:** Both `* Split Params` nodes are optional — leave them unconnected for a single
> full-frame pass (no temporal/spatial splitting). The combo re-runs the sampler at the higher
> resolution, so it trades extra compute for VRAM headroom on long / high-res clips.

---

## 🧪 Model / Architecture

- **Latent format:** 24-channel Minimax H3 VAE latent, normalized per-channel with the training
  mean/std before inference and de-normalized after.
- **Backbone:** the 2D variant stacks Conv2d blocks (with a 7×7 depthwise large-kernel block every
  3 layers) and interleaves a temporal Conv1d every 2 layers; the 3D variant is pure 3D conv blocks +
  temporal convolutions. **Residuals exist only inside blocks** — each block returns `x + f(x)` via a
  1×1 skip or identity — while the backbone as a whole has **no global residual**: the output is
  produced entirely by the convolutional path with no interpolated/input base added on top.
- **Architecture is inferred from the checkpoint**: `in_channels`, block counts, `channels`, temporal
  config, kernel size, CoordConv/SE/large-kernel flags, learned-upsample branch and tier channel are
  all read from the `state_dict` shapes and key names — no manual config. Both releases use 12+12
  blocks, `temporal_every=2`, `temporal_kernel=5`, `attn=False`; 2D uses `channels=640` (182M
  parameters) and 3D uses `channels=512` (345M).
- **Upsampling:** if the checkpoint carries learned PixelShuffle kernels (`learned_upsample_ks`) they
  are used; otherwise it falls back to feature interpolation — bilinear for the 2D node, trilinear
  for the 3D node. The console prints which path was taken.
- **Temporal handling:** both variants preserve the time dimension (only H×W are scaled); chunked
  inference caps VRAM on long clips.
- **Sharpness tiers:** checkpoints with a tier channel encode `sharpness` into a 32-dim condition fed
  alongside the scale condition; the tier channel is zero-initialized, so checkpoints without it
  behave bit-identically to the previous version.

---

## 📊 Training Data

Training data totals **~160,000 independent pairs** (one low-res latent paired with one high-res
target, excluding sharpness copies): **146,945 pairs** drive the main (scale) training, and
**13,967 groups** of separate footage train the sharpness tiers — each group paired with 1–6 tier
targets, **83,802 pairs** in total.

**By upscale factor (share of the main training set, measured):**

| Scale band | Pairs | Share |
| :--- | ---: | ---: |
| < 1.5× | 13,649 | 9.3% |
| 1.5× – 2.5× | 70,773 | 48.2% |
| 2.5× – 3.5× | 45,023 | 30.6% |
| 3.5× – 4.0× | 17,500 | 11.9% |
| **Total** | **146,945** | 100% |

The five dominant factors are **2.0× (32,624 pairs, 22.2%), 2.5× (15,000, 10.2%), 4.0× (10,500,
7.2%), 1.5× (11,150, 7.6%) and 3.0× (10,824, 7.4%)**; the remaining 0.25-step scales mostly hold
2–3k pairs each (~2.25%). Beyond the fixed buckets, the set includes arbitrary decimal scales between
1.0× and 4.0× so the model generalizes to any continuous `scale` value instead of only the fixed
buckets.

---

## 🙏 Acknowledgments

This node follows the neural-latent-upscaling approach pioneered by
[ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale) by **Ttl**
(https://github.com/Ttl). The model architecture also draws on and references the
**LTX 2.3 Spatial Upscaler** (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`).
Thanks to both projects for the open-source foundation this work builds upon.


---

## FILE: README_zh.md (https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/HEAD/README_zh.md)

<p align="center">
  <a href="./README.md">English</a> ·
  <a href="./README_zh.md"><strong>中文</strong></a>
</p>

<div align="center">

# ComfyUI Minimax H3 Latent Upscaler

**Minimax H3 视频生成专用 Latent 神经网络放大节点**
学习型 · 高保真 · 2D 与 3D 双版本

</div>

## 📰 更新动态

- [2026-11-10] 🚀 **发布 2D 卷积版本 `minimax_h3_latent_upscaler_2d_conv_v1`**：新权重为 2D 卷积主干（Conv2d）+ 穿插时序卷积（Conv1d），参数量 **1.82 亿**（3D 版 3.45 亿，约为一半），fp16 权重约 **364 MB**（3D 版约 691 MB）。在全部 36 个倍率上，latent MSE 相对三线性降低 **84.7%**（3D 版 61.1%）；1.1×–1.4× 低倍率仍有约 **80%** 的改善（3D 版此时只有 28%）；像素域平均 PSNR **30.47 dB**（3D 版 29.30 dB）。同时支持 **6 档锐度**（3D 版无可调锐度），并覆盖 1.1×–4.0× 全倍率。3D 版只在它训练过的 2×/2.25×/2.5×/4× 上略有优势，且幅度有限：按统一口径「相对三线性的改善」，这四个倍率 2D 是 **83.0%–86.4%**、3D 是 **84.7%–89.1%**，只差 **1.0–3.3 个百分点**（绝对 MSE 差 0.004–0.024）——「3D 比 2D 低 7.5%–30.5%」是在这层很小残差上算的相对差，不是量级差距。时序一致性也略强（帧间差比值 0.867 vs 0.850，偏离真值 13.3% vs 15.0%），差距同样很小。明细见下方「选哪份权重」对比表。完整对比见[对比报告](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/examples/3D_Conv_v1_vs_2D_Conv_v1.html)。
- [2026-11-10] 🧩 **2D 节点能力补齐，与 3D 参数对齐**：2D 节点现支持三种缩放模式（`scale by multiplier` / `target dimensions` / `megapixels`）、`rocm` 后端、`force_unload` 显存卸载、像素网格 `align` 对齐，新增 **Sharpness 锐度档位**（1–6）与 **Temporal Chunking 时间分块**（`chunked` / `single pass`）。
- [2026-08-28] 🚀 **新增组合节点 — MMH3 Split Upscale**：新增三个节点（`MMH3 Temporal Split Params`、`MMH3 Spatial Split Params`、`MMH3 Split Upscale`），对 H3 视频做**分块式「高清重采样」放大**。该组合是在 **Comfyui-MMH3-UltimateUpscale** 项目基础上拆解优化而来——把 AV latent 沿时间切块、沿空间分块，逐块重采样后再无缝拼接；相比原版主要增强：更稳的拼接（`seam_denoise` + probe 门控二道缝修补）、颜色零漂移（空间+时间两级颜色匹配 + 首块/每 chunk 钉源）、三重时间锚（frame-0 / motion / identity）、更易用（空间参数简化为重叠率/渐变率百分比并自动吸附网格）、更轻量（移除内置放大模型，改接外部预放大 latent）。
- [2026-08-26] 🚀 **3D 节点显存与使用优化**：采用零拷贝模型加载，并接入 ComfyUI 原生 `soft_empty_cache` 与显式 `.contiguous()`，降低加载与推理期间的显存/内存峰值；推理后卸载到 CPU 改为可选开关 `force_unload`（默认开启）；`enable_chunking` 更名为 `enable_temporal_chunking`，分块有效帧数由 24 提升至 32；模型文件现可放入子目录（PR #30）；`target dimensions` 与 `megapixels` 上限分别放宽到 8192 像素与 16 MP。
- [2026-08-23] 🚀 **3D 节点改进**：新增 `enable_chunking` 开关（短片段可关闭，使用全上下文推理）；修复时间分块边缘伪影——采用 Replicate Padding 与重叠区域加权融合，消除末尾帧闪烁；新增 ROCm（AMD 显卡）后端支持，设备选项新增 `rocm`。
- [2026-08-21] 🚀 **3D 节点优化**：节点执行后自动将模型卸载到 CPU，为后续二次采样释放显存；宽高分别独立对齐到 `align` 网格（默认 32），修复底部光带问题；归一化/反归一化由原地操作改为标准运算，保留时间分块以支持长视频。
- [2026-08-19] 🚀 **3D 节点重构**：三种缩放模式（`scale by multiplier`、`target dimensions`、`megapixels`）合并进单一节点；修复特定模式下造成的比例不统一，修复各类特定尺寸造成的边缘伪影；新增示例工作流供参考。
- [2026-08-18] 🔥 **精度选择**：2D / 3D 节点均支持 `fp32` / `fp16` / `bf16` 推理。
- [2026-08-17] 🎉 **初始发布**：Minimax H3 Latent Upscaler 2D + 3D 双节点、双语 README、内联示例。

一个 ComfyUI 自定义节点，用训练好的神经网络（而非简单插值）对 **Minimax H3** 的 VAE latent
（24 通道）进行放大。它的**主要目的是加速高分辨率视频的生成**，并提升画质：

- **跳过「解码 → 像素放大 → 再编码」的慢速往返。** Minimax H3 自带约 5B 参数的笨重 VAE，
  对 latent 做解码和再编码都很耗时；直接在 latent 空间放大，就彻底省掉了这一往返开销。
- **支持更快的生成流程：** 先低分辨率生成（latent token 少得多），用本节点放大 latent，再在
  目标分辨率下二次采样 / 重绘精修。

同时，它**避免了直接对 latent 插值（双线性/双三次）带来的鬼影、重影问题**，作用类似于
**LTX2.3** 中的 latent 放大模型。

⚠️ 它节省的是**时间，不是显存** —— 精修阶段仍在目标分辨率下运行，峰值显存与直接高清生成接近，
收益纯粹是更快出片。

提供两个节点变体，均注册在 `video/MinimaxH3` 分类下，**参数几乎一致**（2D 多一个 `sharpness`，时间分块在 2D 是下拉、在 3D 是开关）：

- **Minimax H3 Latent Upscaler (2D)**：2D 卷积主干 + 穿插的时序卷积，仅做空间（H×W）放大，
  时间维度保持不变，轻量且快速；对应权重 `*_2d_conv_v1`，支持 6 档锐度。
- **Minimax H3 Latent Upscaler (3D)**：纯 3D 卷积主干（3D 卷积块 + 时序卷积 + 三线性插值），
  联合处理时空体，时间一致性略强，但参数与算力约为 2D 版的 1.9 倍；对应权重 `*_3d_conv_v1`。

> 权重与节点主干必须匹配：2D 节点加载 2D 权重，3D 节点加载 3D 权重。选错时加载器会直接报
> 结构不匹配，而不是静默出错误结果。

两个节点都内置**三种缩放模式**：

- `scale by multiplier`：经典 `scale` 倍数（1.0×–4.0×）。
- `target dimensions`：直接填目标像素 `width` / `height`。
- `megapixels`：填目标总像素（百万像素，例如 `1.2`），保持原宽高比。

`target dimensions` 与 `megapixels` 两种模式会按可配置的像素网格对齐输出，并自动反推等效倍率，
因此 1.0×–4.0× 之间任意目标都能用。

> 两个节点均**只支持放大**（等效 `scale >= 1.0`）。`scale = 1.0` 返回输入原样；等效倍率小于 1.0 会报错。

### Sharpness 锐度档位（仅 2D 权重）

带档位通道的权重暴露一个 `Sharpness` 参数（1–6）：

- **1** = 最贴近原图（最柔和，MSE 最低）——默认值，也是绝大多数场景的推荐值。
- **6** = 最锐（细节最多，但相对精确目标偏离最远）。

旧权重不含档位通道时该参数被静默忽略（按 1 处理），控制台会打印一行提示。

> 锐度档位目前只存在于 `2d_conv_v1` 权重：`3d_conv_v1` 实测没有可调锐度，其输出锐度等价于 2D 的 t1 档。

### MMH3 Split Upscale（组合节点）

> **来源：** 该组合是在 **Comfyui-MMH3-UltimateUpscale** 项目基础上**拆解优化**而来。原本的单节点被拆成三个可组合节点（时间切分 / 空间切分 / 主放大），
> 并在原有分块逻辑之上，针对拼接稳定性、色彩一致性、时间连续性、易用性与轻量化做了系统性增强。

一套独立的**三节点组合**，在扩散采样器内部做**分块式「高清重采样」放大**（而非用预训练放大网络）。
它接收 H3 的 **AV latent**（嵌套：视频 24 通道 + 音频 32 通道），沿**时间切块**、沿**空间分块**，
对每一块单独跑采样器，再用缝降噪、两级颜色匹配与时间锚点把结果**无缝拼接**回去，避免出现接缝和鬼影。
两个 `* Split Params` 节点负责配置切分方式，并喂给主节点 `MMH3 Split Upscale`（两者均可选——
不连接即为整段/整帧单次处理）。它需要 `model` + `conditioning` + `sampler`/`sigmas`，即会在更高
分辨率上重新跑一遍采样。

**相比原版主要增强（拆解优化点）：**

- **🔧 更稳的拼接（Seam / Ghosting）**：缝邻域降噪上限（`seam_denoise`）+ probe 门控的**二道缝修补**（`seam_polish`），消除接缝与鬼影；在高降噪 + 快速运动下，运动物体也不再被切断（"断肢"）。
- **🎨 颜色零漂移（Color Zero-Drift）**：**空间 + 时间两级颜色匹配**，并对**首块**与**每个 chunk** 做全局色基准钉源（`grade_pin`），彻底解决块间闪烁、左上角发青等问题。
- **⏱️ 更强的时间连续性（Temporal Continuity）**：`frame-0 锚` + `motion 锚` + `identity 锚` **三重时间锚**，高降噪时自动兜底身份一致性，人物不漂移。
- **🧩 更易用（Usability）**：空间参数由原本的 9 项精简为**百分比**（重叠率 / 渐变率），并自动吸附 latent 网格；`negative` 紧跟 `conditioning` 布局，连线更直观。
- **🪶 更轻量（Lightweight）**：移除内置放大模型，改接**外部预放大 latent**，显存更可控、节点更专注。

**新增 / 优化的模块（相对于原项目）：**

| 模块 | 新增 / 优化的内容 |
| :--- | :--- |
| **预防（Prevention）** | 冻结预填重叠带 + **三重时间锚**（frame-0 锚 + motion 锚 + identity 锚）+ cross-fade 拼接——让接缝与分叉在生成之初就不出现。 |
| **校正（Correction）** | **空间/时间两级颜色匹配** + 首块源参考 + 每块（chunk）全局钉源（`grade_pin`）——保证各瓦片、各时间块之间的色彩与亮度一致，杜绝块间闪烁与局部发青。 |
| **抗分叉（Anti-forking）** | `seam_denoise` 上限——在高降噪 + 快速运动下，缝邻域以*中等*降噪续写邻居内容，防止运动物体被切断（"断肢"）；配合 probe 门控的二道缝修补，接缝与鬼影一并消除（建议 0.5–0.8；1.0 = 关闭）。 |
| **修复（Fixes）** | **每瓦片独立 Guider** + 裁剪 keyframes，以及**修复版 probe 门控去缝**（`seam_polish`）——每个瓦片获得正确作用域的条件/关键帧，去缝门控不再过度或不足。 |
| **简化（Simplification）** | `overlap` / `fade` 改为**百分比参数**（重叠率 / 渐变率，与分辨率无关），替代原来的绝对像素值，并自动吸附 latent 网格；`negative` 紧跟 `conditioning`，布局更直观。 |

---

## 📸 示例

**视频放大对比**

<video src="examples/Minimax_h3_latent_Upscaler_001.mp4" controls width="640"></video>

**图像放大对比**

![](examples/Minimax_h3_latent_Upscaler_002.jpg)

**2D 与 3D 逐倍率对比**：完整的 36 档 MSE / PSNR / 锐度 / 时序图表见
[`examples/3D_Conv_v1_vs_2D_Conv_v1.html`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/examples/3D_Conv_v1_vs_2D_Conv_v1.html)。

---

## 📁 项目结构

```text
Comfyui_Minimax_h3_latent_Upscaler/
├── examples/
│   ├── Minimax_h3_latent_Upscaler_001.mp4
│   ├── Minimax_h3_latent_Upscaler_002.jpg
│   └── 3D_Conv_v1_vs_2D_Conv_v1.png     # 2D vs 3D 全倍率 MSE 曲线
├── workflow_templates/
│   └── minimax_h3_r2v_Latent Upscaler example workflow.json  # ComfyUI 模板示例工作流
├── nodes/
│   ├── __init__.py                       # 合并 2D/3D/组合节点映射
│   ├── minimax_h3_latent_upscaler_2d.py  # 2D 主干 + 时序卷积（三模式 + 锐度档位 + 时间分块）
│   ├── minimax_h3_latent_upscaler_3d.py  # 纯 3D 卷积，支持 3 种缩放模式
│   └── MMH3_Split_Upscale.py             # MMH3 Split Upscale 组合（基于 Comfyui-MMH3-UltimateUpscale 拆解优化）
├── README.md
├── README_zh.md
└── __init__.py
```

> 模型权重**不**随仓库提供，请按下方「模型放置」说明放入 ComfyUI 模型目录。

---

## 🚀 核心特性

- ✅ **学习型 latent 放大** — 针对 Minimax H3 latent 训练的神经网络，比双线性/双三次插值清晰得多。
- ✅ **两种主干、两种权重** — `2d_conv_v1` 覆盖全倍率、体积减半、锐度可调；`3d_conv_v1` 在 2×/2.25×/2.5×/4× 上更准，时序一致性只微弱领先（帧间差比值 0.867 vs 0.850）。按场景选，节点与权重一一对应。
- ✅ **两个节点都支持三种输出尺寸方式** — `scale by multiplier` / `target dimensions` /
  `megapixels`，均带像素网格对齐与宽高比锁定。
- ✅ **Sharpness 锐度档位** — 一个权重覆盖 1（最保真）到 6（最锐）六档，按需选择，不必为不同锐度各存一份权重。
- ✅ **长视频时间分块** — `chunked` 分块推理压低显存峰值，`single pass` 整段一次前向；旧工作流自动兼容。
- ✅ **分块式高清重采样（MMH3 Split Upscale 组合）** — 把 H3 的 AV latent 沿时间切块 + 空间分块，
  在更高分辨率上重采样，配合缝降噪、两级颜色匹配与时间锚点，避免接缝与鬼影。
- ✅ **24 通道 Minimax H3 latent** — 使用训练时一致的逐通道均值/标准差做归一化。
- ✅ **自动识别模型结构** — 直接从权重读取通道数、块数、时序配置、卷积核大小与是否带上采样/档位通道，无需手动配置。
- ✅ **严格的权重校验** — 关键层缺失时直接报错退出，而不是静默随机初始化（输出会变成噪声）。
- ✅ **鲁棒的权重加载器** — 支持 `.safetensors` 与 `.pth`；自动 FP8→FP16；兼容合并权重中的 `upscaler.` 前缀。
- ✅ **可选精度与设备** — `cuda` / `rocm` / `cpu` 与 `fp32` / `fp16` / `bf16` 选项。
- ✅ **即插即用** — 标准 ComfyUI 节点，不改变现有工作流。

推理时强制关闭注意力（`attn=False`）以提升速度与稳定性；已加载模型按 `(名称, 设备, 精度)`
缓存，重复调用开销很低。

---

## 📦 安装

1. 将仓库克隆到 ComfyUI 的 `custom_nodes` 文件夹：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git
   ```

2. 所需依赖（`torch`、`einops`、`safetensors`）在标准 ComfyUI 环境中已自带，无需额外安装。

3. 重启 ComfyUI。

---

## 🤖 模型放置

节点从这里扫描并加载权重：

```text
ComfyUI/models/latent_upscale_models/
```

把你的 Minimax H3 latent 放大模型权重（`.safetensors` 或 `.pth`）放进该目录，节点下拉框会自动列出。
支持放在子目录里，下拉框会带上子目录名。

预训练权重可在此下载：
[huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

每个版本都提供 fp32 / fp16 / bf16 三档精度，按显卡与工作流选一档即可：

| 版本 | fp16 / bf16 | fp32 (.pth) | 参数量 |
| :--- | :--- | :--- | ---: |
| `minimax_h3_latent_upscaler_2d_conv_v1`（2D 主干） | 约 364 MB | 约 729 MB | 1.82 亿 |
| `minimax_h3_latent_upscaler_3d_conv_v1`（3D 主干） | 约 691 MB | 约 1.38 GB | 3.45 亿 |

**推荐默认 fp16**：与 fp32 数值上几乎无差；`bf16` 给没有 fp16 快速路径的显卡（同体积、精度略低）；
`fp32` 是参考权重，用于转换与逐位校验。

加载器会自动识别结构（通道数、块数、时序配置、是否带锐度档位通道等）。两个版本结构不同、不能互换：`2d_conv_v1` 只接 2D 节点，`3d_conv_v1` 只接 3D 节点。

### 选哪份权重：2D 还是 3D？

同一套验证集（2,500 对冻结验证）上，两个版本的实测差异：

| 指标（36 个倍率平均） | 三线性 | 2D_Conv_v1 | 3D_Conv_v1 |
| :--- | ---: | ---: | ---: |
| latent MSE 相对三线性的改善 | — | **84.7%** | 61.1% |
| 1.1×–1.4× 低倍率改善 | — | **约 80%** | 28% |
| 像素域 PSNR（VAE 解码后） | 23.54 dB | **30.47 dB** | 29.30 dB |
| 时序一致性（latent 帧间差比值，真值 = 1.000） | 0.900 | 0.850 | **0.867** |
| 可调锐度档位 | — | **1–6 档** | 无 |
| 参数量 / fp16 体积 | — | **1.82 亿 / 364 MB** | 3.45 亿 / 691 MB |

**全 36 个倍率的 raw latent MSE（越低越好）**

![3D_Conv_v1 vs 2D_Conv_v1 · 全倍率 latent MSE](examples/3D_Conv_v1_vs_2D_Conv_v1.png)

> 灰 = 三线性，蓝 = `2D_Conv_v1`（t1），橙 = `3D_Conv_v1`。2D 全程贴底、几乎是一条平线，36 个倍率都稳定在三线性的 11%–15%；3D 只在训练过的 2×/2.25×/2.5×/4× 处短暂低于 2D，其余倍率明显反弹——1.1× 处几乎与三线性重合（0.133 vs 0.138），1.3×、2.2× 的改善也只有 30%–45%。这张图就是“2D 赢覆盖面、3D 赢特定倍率”最直观的说明。

- **2D 赢在覆盖面**：低倍率（< 1.5×）优势极大，且多档锐度可调，体积只有一半。
- **3D 赢在训练过的倍率，但幅度有限**：只有 2× / 2.25× / 2.5× / 4× 四个倍率反超，明细见下表。

**3D 占优的四个倍率（统一口径：相对三线性的改善）**

| 倍率 | 三线性 MSE | 2D_Conv_v1 MSE（相对三线性改善） | 3D_Conv_v1 MSE（相对三线性改善） | 3D 比 2D 再低 |
| :--- | ---: | ---: | ---: | ---: |
| 2× | 0.32384 | 0.05489（+83.0%） | **0.04947（+84.7%）** | 11.0% |
| 2.25× | 0.40259 | 0.05475（+86.4%） | **0.05091（+87.4%）** | 7.5% |
| 2.5× | 0.46205 | 0.06644（+85.6%） | **0.05906（+87.2%）** | 12.5% |
| 4× | 0.73592 | 0.10464（+85.8%） | **0.08016（+89.1%）** | 30.5% |

读法：换成「相对三线性改善」这个统一口径后，两者都是 **83%–89%** 的量级，3D 只多 **1.0–3.3 个百分点**
（绝对差 0.004–0.024）。表里最后一列的 7.5%–30.5% 是在这层残差上算的相对差，不是“画质差三成”。

- **时序一致性：3D 只领先一点点** —— 见下表，平均帧间差比值 0.867 vs 0.850，偏离真值
  **13.3% vs 15.0%**（相对改善约 11%，绝对差仅 0.017）。这个差距**不足以成为选 3D 的理由**，
  选 3D 主要还是为了那四个倍率上的 MSE 优势。
- 其余倍率（尤其 1.1×–2.1× 的非整数档、2.6×–3.9×）2D 明显更准。

**时序一致性明细**（latent 域帧间差比值；1.000 = 帧间动态与真值一致，<1 = 比真值更“静”、
细节或运动被抹平，>1 = 更“抖”、闪烁被放大）：

| 倍率 | LR 原样 | 三线性 | 2D_Conv_v1 | 3D_Conv_v1 | 3D 相对 2D |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1.5× | 1.019 | 0.880 | 0.848 | 0.857 | 更接近真值 0.009 |
| 2× | 1.031 | 0.869 | 0.900 | 0.908 | 更接近真值 0.008 |
| 3× | 1.032 | 0.904 | 0.892 | 0.898 | 更接近真值 0.006 |
| 4× | 1.069 | 0.946 | 0.761 | 0.805 | 更接近真值 0.044 |
| **平均** | 1.038 | 0.900 | 0.850 | **0.867** | **+0.017** |

读法：三个方案的平均值都低于 1，也就是输出都比真值“静”一些（细节/运动被抹平一点）。3D 的
0.867 比 2D 的 0.850 更接近 1，差距在 4× 时最明显（0.805 vs 0.761），1.5×–3× 基本在 0.01 以内、
肉眼看不出。**所以“时序更强”只在 4× 高倍率下才有实际意义。**
（2× 那栏三线性 0.869 反而最靠近 1，但那是插值振铃把帧间噪声一起放大造成的，并不代表更真实。）

---

## 🧩 使用方法

从 `video/MinimaxH3` 菜单中添加所需节点，连入一个 `LATENT`，选择模型、设置缩放模式/参数，再解码即可。

**典型流程：**
- **快速预览：** `[Minimax H3 Latent] → [H3 Latent Upscaler] → [VAE 解码]`
- **高品质 / 省时（推荐）：** `[低清 Latent] → [H3 Latent Upscaler] → [二次采样重绘/精修] → [VAE 解码]`

相比「[Latent] → [VAE 解码] → [像素放大] → [VAE 编码] → …」这种朴素做法，直接在 latent 空间
放大跳过了昂贵的 VAE 解码/编码往返。Minimax H3 的约 5B 参数 VAE 让解码与再编码都明显偏慢，时间
主要就省在这里。同时也避免了直接对 latent 插值（双线性/双三次）造成的**鬼影 / 重影**问题。

⚠️ **显存预期：** 本节点只省时间——精修仍在目标分辨率下跑，峰值显存与直接高清生成接近；长视频请用 `Temporal Chunking` 压峰值。

### 节点参数 — 2D

| 参数 | 类型 | 默认值 | 范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | 输入的 Minimax H3 latent（(B,C,T,H,W) 或 (B,C,H,W)） |
| `model_name` | 下拉框 | 自动 | 扫描到的文件 | `latent_upscale_models/` 中的模型 |
| `mode` | 下拉框 | `scale by multiplier` | `scale by multiplier` / `target dimensions` / `megapixels` | 输出尺寸选择方式 |
| `scale` | FLOAT | 2.0 | 1.0 – 4.0（步进 0.05） | `mode` 为 `scale by multiplier` 时使用 |
| `width` | INT | 1280 | 64 – 8192（步进 8） | 目标像素宽（`target dimensions` 使用） |
| `height` | INT | 704 | 64 – 8192（步进 8） | 目标像素高（`target dimensions` 使用） |
| `megapixels` | FLOAT | 1.0 | 0.1 – 16.0（步进 0.1） | 目标总百万像素（`megapixels` 使用），保持宽高比 |
| `sharpness` | INT | 1 | 1 – 6 | **Sharpness。** 1 = 最贴近原图（最柔和，MSE 最低）；6 = 最锐（细节最多，偏离精确目标最远）。只对带档位通道的权重生效，旧权重自动按 1 处理 |
| `align` | INT | 32 | 1 – 512 | 像素网格对齐：输出 W/H 分别独立取整到该值的倍数（严格建议 32） |
| `temporal_chunking` | 下拉框 | `chunked` | `chunked` / `single pass` | **Temporal Chunking。** 长视频分块推理以省显存；`single pass` 为整段一次前向（最费显存） |
| `chunk_frames` | INT | 32 | 4 – 256（步进 4） | **Frames per chunk。** 仅在 `chunked` 下出现。每块实际前向帧数 = 该值 + 48（时间感受野上下文），放得下就别调太小 |
| `force_unload` | BOOLEAN | True | True / False | 推理后将模型卸载到 CPU，为后续节点释放显存；若重复使用本节点可关闭，避免重复加载开销 |
| `device` | 下拉框 | cuda | cuda / rocm / cpu | 推理后端（ROCm 需要支持 HIP 的 PyTorch 构建） |
| `precision` | 下拉框 | fp16 | fp32 / fp16 / bf16 | 推理精度 |

**输出：** `LATENT` — 放大后的 latent，可直接送 VAE 解码。

### 节点参数 — 3D

| 参数 | 类型 | 默认值 | 范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | — | 输入的 Minimax H3 latent（(B,C,T,H,W) 或 (B,C,H,W)） |
| `model_name` | 下拉框 | 自动 | 扫描到的文件 | `latent_upscale_models/` 中的模型 |
| `mode` | 下拉框 | `scale by multiplier` | `scale by multiplier` / `target dimensions` / `megapixels` | 输出尺寸选择方式 |
| `scale` | FLOAT | 2.0 | 1.0 – 4.0（步进 0.05） | `mode` 为 `scale by multiplier` 时使用 |
| `width` | INT | 1280 | 64 – 8192（步进 8） | 目标像素宽（`target dimensions` 使用） |
| `height` | INT | 704 | 64 – 8192（步进 8） | 目标像素高（`target dimensions` 使用） |
| `megapixels` | FLOAT | 1.0 | 0.1 – 16.0（步进 0.1） | 目标总百万像素（`megapixels` 使用），保持宽高比 |
| `align` | INT | 32 | 1 – 512 | 像素网格对齐：输出 W/H 会分别独立取整到该值的倍数（如 16/32/64） |
| `enable_temporal_chunking` | BOOLEAN | True | True / False | 长视频时间分块以压低显存峰值；短片段可关闭，使用全上下文推理 |
| `force_unload` | BOOLEAN | True | True / False | 推理后将模型卸载到 CPU，为后续节点释放显存；若重复使用本节点可关闭，避免重复加载开销 |
| `device` | 下拉框 | cuda | cuda / rocm / cpu | 推理后端（ROCm 需要支持 HIP 的 PyTorch 构建） |
| `precision` | 下拉框 | fp16 | fp32 / fp16 / bf16 | 推理精度 |

**输出：** `LATENT` — 放大后的 latent。

> **选哪个节点？** 默认走 **2D**：覆盖 1.1×–4.0× 全倍率、低倍率优势大、体积只有一半、锐度可调。
> 只有**固定跑 2×（或 2.25×/2.5×/4×）想要更高精度**时才换 **3D**——这几个倍率的 latent MSE 比 2D
> 再低 7.5%–30.5%（绝对差 0.004–0.024；换成相对三线性的改善口径，2D 约 85%、3D 约 87%，只差 1–3 个百分点），
> 时序一致性也只是微弱领先。
> 具体数据见上文「选哪份权重：2D 还是 3D？」。

### 节点参数 — MMH3 Split Upscale（组合）

一套**分块式「高清重采样」**的 H3 视频放大方案，**在 Comfyui-MMH3-UltimateUpscale 项目基础上拆解优化而来**。和 2D/3D 节点（用预训练放大网络）不同，这套组合会在
latent 的更高分辨率版本上重跑扩散采样器，并切块以适配显存。它操作的是 H3 的 **AV latent**
（嵌套：视频 24 通道 + 音频 32 通道），需要 `model` + `conditioning` + `sampler`/`sigmas`。

**`MMH3 Temporal Split Params`** — 沿时间怎么切（对齐 H3 原生的 5+17·m 帧/token 网格）：

| 参数 | 类型 | 默认值 | 范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `chunk_frames` | INT | 73 | 5 – 100000 | 每块帧数（自动对齐 H3 token 网格） |
| `temporal_overlap_frames` | INT | 22 | 0 – 100000 | 相邻块之间的重叠帧数 |
| `anchor_strength` | FLOAT | 0.999 | 0.0 – 1.0 | 时间锚强度 |
| `motion_anchor_frames` | 下拉框 | 22 | 0 / 5 / 22 / 39 | 运动锚长度（帧） |
| `identity_anchor_frames` | INT | 24 | 0 – 240 | 身份锚间距 |

输出：`temporal_split_param`（自定义类型，喂给主节点）。

**`MMH3 Spatial Split Params`** — 每帧怎么切瓦片：

| 参数 | 类型 | 默认值 | 范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `tile_width` | INT | 512 | 64 – 16384（步进 32） | 瓦片宽（像素） |
| `tile_height` | INT | 512 | 64 – 16384（步进 32） | 瓦片高（像素） |
| `overlap_ratio` | FLOAT | 0.25 | 0.0 – 0.90 | 重叠占比（占瓦片） |
| `fade_ratio` | FLOAT | 0.50 | 0.0 – 1.0 | 重叠区内的渐变带宽度 |
| `min_tile_size` | INT | 256 | 0 – 16384（步进 32） | 最小瓦片边长；避免边缘出现极小瓦片 |
| `seam_denoise` | FLOAT | 1.0 | 0.1 – 1.0 | 缝邻域降噪上限；<1 时高降噪下防止快速物体被切断（建议 0.5–0.8） |

输出：`spatial_split_param`（自定义类型）+ `grid_preview`（字符串，显示计算出的瓦片/重叠网格）。

**`MMH3 Split Upscale`** — 主调度节点：

| 参数 | 类型 | 默认值 | 范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `model` | MODEL | — | — | 用于重采样的扩散模型 |
| `conditioning` | CONDITIONING | — | — | 正向条件 |
| `negative` | CONDITIONING | — | 可选 | 负向条件 |
| `latent` | LATENT | — | — | 输入的 H3 AV latent（嵌套：视频 24 通道 + 音频 32 通道）；仅支持 batch 1 |
| `noise` | NOISE | — | — | 采样器用的噪声源 |
| `sampler` | SAMPLER | — | — | 采样器 |
| `sigmas` | SIGMAS | — | — | 调度序列（sigma schedule） |
| `cfg` | FLOAT | 1.0 | 0.0 – 100.0 | 无分类器引导强度 |
| `temporal_split_param` | 自定义 | — | 可选 | 来自 `MMH3 Temporal Split Params` |
| `spatial_split_param` | 自定义 | — | 可选 | 来自 `MMH3 Spatial Split Params` |
| `seam_polish` | 下拉框 | off | off / auto / all | 对检测到的接缝额外做去缝处理 |
| `color_match` | BOOLEAN | True | True / False | 跨瓦片/块的两级颜色匹配 |

**输出：** `LATENT` — 放大后的 H3 AV latent，可直接送 VAE 解码。

> **注意：** 两个 `* Split Params` 节点都可不接（即为整段/整帧单次处理，不做时间/空间切分）。
> 该组合会在更高分辨率上重跑采样器，因此以额外算力换取长视频 / 高分辨率下的显存余量。

---

## 🧪 模型与架构

- **Latent 格式：** 24 通道 Minimax H3 VAE latent，推理前按训练均值/标准差逐通道归一化，推理后反归一化。
- **主干结构：** 2D 版由 Conv2d 卷积块堆叠（每 3 块插一个 7×7 大核深度可分离块），并每 2 层穿插
  一次时序 Conv1d；3D 版为纯 3D 卷积块 + 时序卷积。**残差只存在于块内**——每个卷积块的输出是
  `x + f(x)`（带 1×1 跳连或恒等映射），但**主干整体没有全局残差**：网络输出完全由卷积预测给出，
  不叠加插值/输入基座，不会出现“神经网络只学残差、基座靠插值”的情况。
- **结构由权重推断**：`in_channels` / `in_blocks` / `out_blocks` / `channels` / 时序配置 /
  卷积核大小 / 是否带 CoordConv、SE、大核、学习型上采样、锐度档位通道，全部从 `state_dict`
  的形状和键名反推，不需要手动配置。两个版本都是 12+12 块、`temporal_every=2`、
  `temporal_kernel=5`、`attn=False`；2D 版 `channels=640`（1.82 亿参数），3D 版 `channels=512`
  （3.45 亿参数）。
- **上采样方式：** 权重若带学习型上采样核（`learned_upsample_ks`，即 PixelShuffle 分支）则启用
  学习上采样；否则回退到特征插值——2D 节点双线性，3D 节点三线性。控制台会打印实际走的路径。
- **时间处理：** 两个节点均保持时间长度不变，仅放大空间分辨率（H×W）；长视频用时间分块控制显存。
- **锐度档位：** 带档位通道的权重在内部把 `sharpness` 编码成 32 维条件，与倍率条件一起送入主干；
  档位通道零初始化，所以旧权重 / 不带档位时行为与之前逐位一致。

---

## 📊 训练数据

训练语料共 **约 16 万个独立配对**（一份低分辨率 latent 配一份高分辨率目标，不含锐度副本）：
其中 **146,945 对**用于主干（倍率）训练，**13,967 组**独立语料专门训练锐度档位——
这 13,967 组每组再配 1–6 六个锐度档位的 HR，共 **83,802 对**。

**按缩放倍数（占主干训练量，实测）：**

| 倍数区间 | 对数 | 占比 |
| :--- | ---: | ---: |
| < 1.5× | 13,649 | 9.3% |
| 1.5× – 2.5× | 70,773 | 48.2% |
| 2.5× – 3.5× | 45,023 | 30.6% |
| 3.5× – 4.0× | 17,500 | 11.9% |
| **合计** | **146,945** | 100% |

其中 **2.0×（32,624 对，22.2%）、2.5×（15,000 对，10.2%）、4.0×（10,500 对，7.2%）、
1.5×（11,150 对，7.6%）、3.0×（10,824 对，7.4%）** 是五个主力倍数，其余 0.25 步进的档位多为
2–3 千对（每档约 2.25%）。除了固定档位，还包含 1.0×–4.0× 之间的任意小数倍率，用于提升对连续
`scale` 值的泛化能力，避免只在几个固定档位上表现好。


---

## 🙏 致谢

本节点沿用了 **Ttl** 提出的「神经网络 latent 放大」思路
（[ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale)，
https://github.com/Ttl）。本项目模型架构同时参考借鉴了 **LTX 2.3 Spatial Upscaler**
（`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`）。感谢这些开源工作为本项目奠定基础。


---

# ISSUES / PRs of LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler


### #70 [open] ISSUE: New 2D Conv v1 introduces a small vertical geometry change. Verified at 576 -> 1024, reproduced with the bundled SFW example
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/70  by LordLocksley 2026-11-17T19:37:23Z

Thanks for the new 2D model. Its detail reconstruction is useful, but I am seeing a small vertical geometry change that distorts the original latent aspect ratio. It becomes specially visible when joining segments for a continuation workflow to an existing video.

## Reproduction

Source: [the example MP4 bundled with this repository](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/f34cadb19f10fee7e36091b690cac39b4eb40573/examples/Minimax_h3_latent_Upscaler_001.mp4).

Reproduction package:
[SFW_2D_Upscaler_Geometry_Issue.zip](https://github.com/user-attachments/files/34991667/SFW_2D_Upscaler_Geometry_Issue.zip)

The included `sfw_source_686.mkv` is a lossless FFV1 copy of its first 56 frames at 24 fps, cropped from the top-left source panel at x=272, y=0, width=686, height=686. This removes the comparison labels and other panels. All test branches receive these exact same frames.

1. Load the attached `reproducer.json` workflow.
2. Select the included `sfw_source_686.mkv` in its video loader.
3. Select the video VAE and 2D checkpoint listed below, if necessary.
4. Run once and compare matching frames in the four output folders.

Branches:

- **A:** source resized directly to 1024 x 1024.
- **B:** A -> VAE Encode -> VAE Decode at 1024 x 1024.
- **C:** source resized to 576 x 576 -> VAE Encode -> LBH 2D latent upscale to 1024 x 1024 -> VAE Decode.
- **D:** the SAME low-resolution latent used by C -> VAE Decode at 576 x 576 -> ordinary Lanczos image resize to 1024 x 1024.

All outputs have identical 1024 x 1024 dimensions and 56 frames. Branches B and D preserve geometry closely; C introduces a small vertical compression relative to the horizontal scale. The image content changes geometry, rather than the file dimensions or display aspect ratio changing.

Comparision gif:
<img width="1024" height="1104" alt="blink_frame_0028" src="https://github.com/user-attachments/assets/f4a10636-9ecd-44d9-93d6-9b72e63ad5be" />

## Configuration

- ComfyUI 0.45.0; Windows; RTX 4080 16 GB.
- Installed node-pack revision: `f34cadb19f10fee7e36091b690cac39b4eb40573`. The installed 2D source matches the upstream file retrieved during this investigation.
- VAE: `minimax_h3_video_vae_int8_convrot.safetensors` through native VAELoader / VAEEncode / VAEDecode. This reproduction has NOT been repeated with a BF16 video VAE.
- Upscaler: `minimax_h3_latent_upscaler_2d_conv_v1_fp16.safetensors` (364,308,928 bytes; no full payload hash was computed for this investigation).
- Target dimensions: 1024 x 1024, from a 36 x 36 latent grid to 64 x 64; scale 16/9, approximately 1.777778 on BOTH axes.
- Sharpness: 1; alignment: 32; device: CUDA; precision: FP16; force_unload: true.
- Temporal chunking widget: chunked / 32. The 56 source frames encode to 17 latent frames, so the node takes its full-forward path (`T <= chunk_frames`); there is no internal temporal chunk boundary in this reproduction.

## Samples and measurements

The original matching frame-28 PNGs are attached as A/B/C/D. Their embedded workflow/prompt metadata was removed. `comparison_frame_0028.png` places D and C side by side at native pixel size without geometric registration. `blink_frame_0028.gif` alternates those same images; the GIF uses a shared 256-colour palette and is for geometry visualization, while the PNGs are the original 8-bit comparison data.

A SIFT feature match + RANSAC affine fit on four matching full-frame pairs gives:

| Frame (1-based) | C versus D: estimated vertical scale relative to horizontal | Estimated vertical change |
|---|---:|---:|
| 1 | 0.985072 | -1.49% |
| 28 | 0.991051 | -0.89% |
| 51 | 0.990575 | -0.94% |
| 56 | 0.989670 | -1.03% |

For these same frames, the maximum absolute relative-axis scale change is approximately 0.04% for B versus A and 0.06% for D versus A. C-versus-D fits retain 463-535 feature correspondences with median inlier residuals of approximately 0.85-0.99 pixels.

These affine estimates describe matched image features; they are not proof that every pixel undergoes one perfectly uniform global stretch. The CSV contains the fit details. Method: OpenCV SIFT defaults, BFMatcher with Lowe ratio 0.75, `estimateAffine2D` with RANSAC reprojection threshold 2 pixels. Only this source/scale/tier/precision was run for the public reproduction.

## Code inspection and question

I have not established a specific faulty line or a safe correction:

- Both target axes are square and equally scaled; the integer alignment does not introduce an aspect-ratio mismatch here.
- The checkpoint has no learned PixelShuffle bank, and `USE_LOCAL_ENSEMBLE` is false. Its mid-network resize therefore uses ordinary bilinear interpolation, with no active PixelShuffle crop.
- Axis order and per-axis scale calculation in the inspected path look consistent.
- The checkpoint header and detected architecture agree with the published 2D configuration: 24 channels, 640 hidden channels, 12 input / 12 output blocks, embedding dimension 128, and the 32-dimensional sharpness-tier input.
- The normalization constants match both the current ComfyUI H3 VAE constants and the published model configuration. The inputs come from native ComfyUI VAEEncode, rather than an external raw-latent pipeline.

Could this be a learned geometry bias at this scale, or a mismatch between the released inference implementation and the training-pair resize/crop or scale-embedding convention? A known-good input/output pair or the exact training preprocessing/forward would help distinguish those possibilities. The reproduction establishes the deformation in this pipeline; it does not distinguish a checkpoint issue from an inference/training compatibility issue.

This was first noticed at a continuation join, but the attached source-only test reproduces it without any continuation machinery.


> **Comment by LordLocksley (2026-11-17T19:37:33Z):**

@LBH-123-AI 


> **Comment by chanon (2026-11-21T14:45:59Z):**

First thank you so much to LBH-123-AI for creating the upscaler models.

Regarding the vertical geometry changes, I have also run into this issue. The behavior of the 2D latent upscaler in terms of geometry is quite unpredictable. 

This issue reports square dimensions in both source and target. But this problem is even more prominent when the dimensions requested are not exactly the same aspect ratio to the source - which is common with H3 because it can only do multiples of 32 pixels.

The 3D latent upscaler would stretch uniformly, while the 2D latent upscaler 

- sometimes crops
- sometimes invents new rows of pixels at the top and bottom
- and sometimes stretches non uniformly

For upscaling it is still usable, but I have just realized today that it is unusable for dual stage extensions. Will be reverting to the 3D upscaler for extensions for now.



### #69 [open] ISSUE: Docs: README '36 scales' reads as resolution levels but scale is an upscaling multiplier band (1.1-4.0x)
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/69  by kylixs 2026-11-16T00:44:25Z

## Documentation clarification: "36 scales" in the README refers to upscaling multiplier bands, not resolution levels

First — thank you for this project; the 2D-conv v1 weights have been very useful in our upscaling chain.

While integrating the node and reading the evaluation table in the README, we initially parsed **"36 scales" as 36 resolution levels**, and picked parameters accordingly — which produced wrong output sizes. After measuring on our side, the `scale` parameter behaves as an **upscaling multiplier band (continuous, roughly 1.1–4.0×)** rather than a target-resolution list.

Concretely: upscaling 1344×768 → 1920×1088 is a **1.43×** factor, and it lands in the 1.1–1.4× band of the table, consistent with multiplier semantics. We could not find this stated explicitly in the README, and at least two people on our side misread it the same way.

## Suggestion

A one-line clarification in the README (e.g. *"scale = upscaling factor band (1.1–4.0×), not a target resolution"*) next to the evaluation table would prevent this misreading. Happy to open a docs PR if that's easier.

## Environment

- Node used with H3-family latents, 24-channel latent upsampling path
- NVIDIA DGX Spark (Thor, SM110, 128 GB unified memory)



### #68 [open] ISSUE: Blocky artifacts and Blinking light when using Temporal Split Params
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/68  by xormander 2026-11-14T16:43:26Z

When using the MMH3 Temporal Split Params node "identity anchor" it has to be set to minus 5 to what is default set. Instead of 22 it should be set 17 or 34. Otherwise it will give blocky artifacts and blinking light.


### #67 [open] PR: Allow non-seperated AV latent
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/pull/67  by excosy 2026-11-12T02:53:17Z

Keep the origin noise mask for next nodes


### #66 [open] ISSUE: 麻烦作者支持下 comfyui 的  【自定义节点管理器】
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/66  by shifenhutu 2026-11-11T16:50:31Z

希望可以在里面直接安装

谢谢


<img width="1423" height="709" alt="image" src="https://github.com/user-attachments/assets/6256fbf5-d867-4421-a9d8-2b324c5b5687" />


参考：
https://docs.comfy.org/zh/manager/pack-management


### #65 [open] ISSUE: 【bug】某些放大会出现气泡状花屏现象
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/65  by lanbomo 2026-11-02T19:56:32Z

某些放大会出现花屏现象，区域内一块或两块会有这种气泡状的花屏，在某些采样潜空间就会出现，只要是同一个潜空间，就会稳定出现这种情况。如果把潜空间先VAE解码成图像再VAE编码成潜空间再放大，就不会有这种情况了。

<img width="246" height="161" alt="Image" src="https://github.com/user-attachments/assets/c22abc85-3ad0-4de4-919c-8a70d1843d0a" />

<img width="606" height="195" alt="Image" src="https://github.com/user-attachments/assets/3e928992-bc8a-48b9-9aaf-56f7250c8f10" />


### #64 [open] ISSUE: 3D v1 weights (minimax_h3_latent_upscaler_3d_conv_v1) produce structured noise with current main — known-good node version for v1?
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/64  by LujiaJin 2026-10-30T07:02:58Z

Hi, thanks for the great work on this upscaler.

I'm porting the 3D upscaler outside ComfyUI (pure torch, feeding it Minimax-H3 video latents from the diffusers pipeline) and hitting a blocker with the **v1 checkpoint**:

**Symptom**
- Input: clean H3 video latents `(1, 24, T, H, W)` (decoded directly they look correct; per-channel stats match `LATENTS_MEAN/STD` exactly).
- Output of `LatentResizer3D` (strictly loaded from `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`, arch auto-detected: 24ch / 12+12 blocks / 512 / temporal_every=2 / kernel=5, params = 345,280,216 as documented): std ≈ 5× input, spatially structured noise; correlation with the input only ≈ 0.38. Decoded: dense flickering colored dots, no semantic content.
- Per-layer activation trace: the **first ResBlockEmb3D already amplifies std from ~5 (after conv_in) to ~70**; growth after that is mild (70→576 over 35 blocks).

**Already verified NOT the cause** (happy to share any of these tests):
- my port: diffed line-by-line against `nodes/minimax_h3_latent_upscaler_3d.py` @ current main (0 differences in the model classes / forward);
- dtype: bf16 / fp16 / fp32 identical;
- temporal chunking on/off identical; input convention matrix (raw vs normalized domain, axis permutations, double-norm) all produce the same broken distribution;
- scale conditioning values are small and sane (`|scale| < 2, |shift| < 1.9`);
- GroupNorm num_groups swept 1…128 (no "snap" to correct behavior);
- in_blocks↔out_blocks key permutation makes it worse; zero-init signatures of `out_layers`/`pwconv` look trained, not swapped;
- bf16 vs fp16 weight files are identical up to dtype rounding, and per-tensor statistics look like healthy trained weights;
- also tried the `h3_patch_lattice` grid_sample resample from the xmarre Plus fork in place of the mid-network trilinear — no change.

**Question**
Was the v1 checkpoint trained with an **earlier architecture/forward revision** (the 08-19 "3D 节点重构" caught my eye) that is layout-compatible with current code but not numerically equivalent? If so:
- is there a **known-good commit** of `minimax_h3_latent_upscaler_3d.py` that matches v1, or
- any plan to re-export an updated checkpoint that matches current main?

Happy to share a minimal repro script (pure torch + the checkpoint, ~40 lines) if useful. Thanks!

> **Comment by LujiaJin (2026-10-30T10:07:12Z):**

**Found the root cause — it's a latent normalization convention mismatch, not the weights or the node version.**

TL;DR: this upscaler expects latents that are **already normalized** (`(x - mean) / std` with the H3 VAE stats — the "loop format" that official ComfyUI workflows pass around), and the network then applies a **second normalization with the same stats internally** (it was trained that way). Our test harness was feeding **raw / de-normalized** latents — e.g. what diffusers returns from the H3 pipeline with `output_type='latent'` — so the network received far out-of-distribution input and produced exactly the structured noise described above.

The pointer that cracked it: ComfyUI core's own implementation (`comfy/ldm/minimax/latent_upscaler.py`, ~line 137) explicitly notes the model was *"trained on VAE latents normalized a second time with the same stats"* — i.e. the expected input convention is: normalized latent in, second normalization inside.

Fix on our side was one line before invoking the upscaler:

```python
lat_loop = (lat_raw - mean) / std   # raw VAE latent -> normalized "loop" format
lat_up   = upscaler(lat_loop)        # model applies its own second norm internally
```

Diagnostics on a real H3 latent (output vs. bilinear-upsampled input reference):

| input to upscaler | std_ratio | corr |
|---|---|---|
| raw (de-normalized) latent | 4.66 | 0.38 |
| normalized (loop-format) latent | ~0.50 | 0.68 |

And end-to-end (low-res gen → **2x latent upscale** → renoise → refine from `sigma[4]` with the same prompt/seed): the output is now a clean, sharp upscaled video, and it clearly beats the trilinear-interpolation control run through the identical refine schedule.

So: **v1 weights are fine, the node code is fine** — the trap only exists when you feed the upscaler from outside ComfyUI and hand it the de-normalized latent. Match the format the official workflows use and the noise disappears. Resolved on our side; leaving this open in case the maintainer wants to add a note to the README for non-ComfyUI users.



### #63 [open] ISSUE: Enhancement: strict checkpoint validation, halo-aware temporal chunking, ComfyUI-native model lifecycle — ideas + reference from DaSiWa Nodes
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/63  by darksidewalker 2026-10-06T14:44:05Z

# Enhancement: strict checkpoint validation, halo-aware temporal chunking, ComfyUI-native model lifecycle — ideas + reference implementation from DaSiWa Nodes

First: **thank you for this work.** The learned 24-channel H3 latent upscaler (2D + 3D variants, ~80k training pairs, the 2×-heavy scale distribution) is exactly the missing piece between "naive bilinear latent resize" and the heavy decode→pixel-upscale→encode round-trip, and the `MMH3 Split Upscale` combo's seam-denoise / two-level color matching / triple temporal anchors are the most thorough de-seaming treatment I've seen in the H3 ecosystem.

I built my own single-node H3 upscale/refine node, **MiniMaxH3 Enhanced Upscale**, in my nodepack **[ComfyUI-DaSiWa-Nodes](https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes)** (Comfy Registry: `comfyui-dasiwa-nodes`). Your 3D upscaler architecture and normalization statistics are adapted in my code with the **MIT attribution retained in the source header**, and your repo is credited in my docs: [`docs/minimax_h3_tiled_upscale.md`](https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes/blob/main/docs/minimax_h3_tiled_upscale.md), upscaler core: [`nodes/h3_latent_upscale.py`](https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes/blob/main/nodes/h3_latent_upscale.py).

Suggestions tailored to your nodes — take what's useful:

### 1. Strict state-dict validation against a meta-device instantiation
Your auto architecture detection reads `in_channels`, block counts, temporal config from the checkpoint — great. A hardening step: after detection, build the model under `torch.device("meta")` (zero memory), take its expected `state_dict()`, and diff keys/shapes against the file: `missing`, `unexpected`, `shape_mismatch` → raise with the first few offending keys. A truncated, renamed-prefix, or foreign checkpoint then fails at selection with a precise message, not inside the first forward. Also accept the `upscaler.`-prefixed variant by stripping the prefix only when **all** keys carry it.

Reference: `_validated_state_dict()` in `nodes/h3_latent_upscale.py`.

### 2. Per-block architecture detection, not cadence
`temporal_every` + `temporal_kernel` assumes a uniform cadence. The state dict itself encodes the exact layout: for each block index, exactly one of `in_layers.2.weight` (residual), `dwconv.weight` (temporal, kernel size = `shape[2]`), `q.weight` (attention) is present. Rebuilding from that per-block signature reproduces any checkpoint exactly — including checkpoints where the temporal kernel size varies per block — and rejects anything ambiguous (`sum(markers) != 1`).

Reference: `_detect_arch()` in `nodes/h3_latent_upscale.py`.

### 3. Halo-aware temporal chunking for the 3D node
`enable_temporal_chunking` with a fixed stride (32) + overlap blending is a reasonable approximation, but the network's temporal receptive field is computable: `halo = sum((kernel_size - 1) // 2)` over all depthwise temporal convs across both stages. Chunk with `halo` frames of real context on each side, run the network on the padded window, and discard the halo output — the interior frames then see true temporal context instead of replicate padding. Worth documenting explicitly that GroupNorm over T·H·W, residual Conv3d, and optional attention still couple frames, so chunked ≠ full-video bit-exact — the halo is a bound, not a guarantee.

Reference: `H3LatentUpscaler.temporal_halo` + the halo-padded window loop in `nodes/nodes_minimax_h3_tiled_upscale.py`.

### 4. Precision `auto` that respects ComfyUI's global policy
A `precision: auto` option that resolves via `model_management.should_use_bf16()` / `should_use_fp16()` **and** honors `--force-fp16` (fp32 on CPU), with explicit `fp16/bf16/fp32` casting weights + compute without silent fallback. This keeps the upscaler consistent with the user's global ComfyUI flags instead of a per-node choice that silently diverges from the diffusion model's dtype policy.

Reference: `_compute_dtype()` in `nodes/h3_latent_upscale.py`.

### 5. ComfyUI-native model lifecycle instead of manual offload
The `force_unload` CPU offload + `soft_empty_cache` works, but wrapping the upscaler in a `ModelPatcher` and registering it via `model_management.LoadedModel.model_load()` gives you ComfyUI's own residency accounting for free: the upscaler participates in normal eviction, releases cleanly per execution (`model_unload` / `patcher.detach` / `patcher.cleanup` in a `finally`), and — used from a sampler-side node — lets you free VRAM by unloading **only the diffusion model and its clones** (`unload_model_and_clones(model, unload_additional_models=False)`) so unrelated models, VAEs, and services are never globally evicted. No global GPU cache, no manual `.cpu()` bookkeeping.

Reference: `H3LatentUpscaler` in `nodes/h3_latent_upscale.py`.

### 6. For the Split Upscale combo (shared with the tiling ecosystem)
Three mechanics from my per-step path that compose well with your anchors and color matching:
- **Per-step tile blending** (Mixture-of-Diffusers): blend tile predictions at every denoising step with global full-frame position coordinates and fp32 tent weights, instead of finishing tiles and stitching — seam bookkeeping largely disappears.
- **Phase-aligned chunk starts**: snap temporal chunk starts/overlaps to H3's `FRAME_PER_TOKEN` period so every chunk's local native grid has the same phase as the full video; single-token keyframe anchors keep exact global positions.
- **One full-frame noise field** sliced per window, so overlapping chunks share deterministic noise in the overlap.
- **Frozen audio x0**: when audio is masked to zero, keep the native x0 (`torch.where(audio_mask == 0, first_prediction, blended)`) instead of averaging per-tile audio predictions — frozen audio stays bit-exact, no summation drift.

Reference: `nodes/h3_tiled_sampling.py` (GPL-3.0 on my side; your adapted architecture keeps its MIT attribution).

### 7. Continuation-aware upscale (Director / continuity integration)

My node sits mid-pipeline in a multi-take Director workflow, not just on finished single clips — the same shape would slot into Split Upscale's anchor system:

- **`continuity_context`** — from the continuation pipeline's Append & Stage node (the already-assembled **cumulative** latent: source + overlap + new extension). Validated read-only against source checkpoint metadata and native timing (frame count = source + extension, overlap tokens matching the native layout, refine start period-aligned to a multiple of 5 tokens = 17 frames), then:
  - **Freeze the old prefix** — everything before the source-tail overlap gets denoise mask 0, so the continuation prompt cannot regenerate earlier scenes; only the overlapping source tail and the newly appended section are refined together. This composes directly with your triple temporal anchors: the anchors protect chunk seams, the prefix mask protects the *take* seam.
  - **Inject the learned-upscaled source tail as native multi-frame AV guide conditioning** at its cumulative timeline position — `resolved_frame_index` shifted by the overlap frame offset, audio slice cut at globally-rounded native boundaries. Your frame-0 anchor generalized: not one anchor frame, the whole upscaled tail as keyframe conditioning.
  - **Original cumulative audio preserved** — overlap never appended twice, no frames added/removed, audio never re-sampled at the join.
  - Optional **`continuity_soft_refine`** (opt-in, default off): a globally-indexed smoothstep mask ramps video refine strength across the existing source-tail overlap, full strength at the new section — reduces refinement-induced light/color steps at the take seam without RGB color matching (complementary to your `grade_pin` color matching, not a replacement).
- **`director_guide` + `vae`** — Director endpoint images (first/last frame) resized **in pixel space** and re-encoded with the video VAE at the final upscaled canvas size, replacing stretched old conditioning keyframe latents; optional `start_image` overrides the guide's first frame. Endpoint substitution only touches single-frame keyframes — multi-frame continuation tails keep their own temporal conditioning, never collapsed into a still.

Reference: `nodes/h3_upscale_continuity.py` (`continuity_plan`, `align_continuity_conditioning`) + `continuity_refine_mask()` / `encode_endpoints()` in `nodes/nodes_minimax_h3_tiled_upscale.py`.

### 8. Small hardening bits
- `pad_to_patch_size` on patchified keyframe/tile latents so odd edges never break the 2×2 patch grid.
- Validate the checkpoint combo value at load time: reject absolute paths and `..` components, re-check the extension on the realpath'd file (combo strings are user-controlled).
- `throw_exception_if_processing_interrupted()` between chunks/tiles so Cancel works mid-run.

---

These are suggestions from a downstream integrator, not a critique — your nodes are the foundation my pack builds on, and the training-data transparency (modality/scale breakdown) is genuinely appreciated. Happy to see any of this land under your license terms.

Thanks for the open-source foundation.

— Darksidewalker, [ComfyUI-DaSiWa-Nodes](https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes)

*This issue text was summarized/translated with AI help.*



### #62 [open] ISSUE: Subsequent run after the first run very slow
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/62  by MostafaBalba 2026-10-06T12:52:10Z

first run runs okay ,next run very very slow i have to close comfy and reopen it to run fast again why?


### #61 [open] ISSUE: Please add an example workflow for Split Upscale.
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/61  by gachapengames 2026-09-27T04:50:14Z




### #60 [open] ISSUE: normalize是不是做了两遍
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/60  by silverbulletmdc 2026-09-23T06:37:35Z

想把这个模块加到代码里，看到模块内部做了normalize和denormalize，于是给他传了vae的原始特征，解码时也直接解，但发现变成花屏了。仔细阅读了下，发现在comfyui里接的前置vae特征已经是被normalize过的了。也就是说，这个模型的特征实际上被normalize了两次？

当然，将错就错的话倒是不影响推理，但是如果不按官方comfyui接入本模块很容易遇到这个问题，放在这里给大家提个醒吧。


### #59 [open] ISSUE: Minimax H3 Latent Upscaler (2D)结果相比3D差非常大
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/59  by simplew2011 2026-09-20T06:02:30Z

- 只是修改一个节点Minimax H3 Latent Upscaler (3D) -> Minimax H3 Latent Upscaler (2D)
- 权重文件不变，调整precision生成的视频都是噪声没去除干净
<img width="1371" height="1267" alt="Image" src="https://github.com/user-attachments/assets/26d9f44f-5a31-45bf-afd5-4186d1f99c17" />


### #58 [open] ISSUE: 会出现花屏现象
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/58  by JoeAu 2026-09-20T00:41:58Z

<img width="1248" height="704" alt="Image" src="https://github.com/user-attachments/assets/ed6b475a-c461-4a8e-adb7-cc17f8b7aa0e" />


### #57 [open] ISSUE: Split Sigmas
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/57  by Dunc4n1dah0 2026-09-16T10:32:50Z

Can you explain what is "Split Sigmas" node in example workflows for?
In r2v example you use 4 step lora, then 8 step in sampler and take 4 step with high sigmas, then put denoised output from sampler into upscale node.

What should I do for example when using new r2v 8-step lora?

> **Comment by MattMundane (2026-09-18T07:24:55Z):**

Bump. I'm interested as well. Not much explanation between Manual Sigmas and SplitSigmas.


> **Comment by Pizzawookiee (2026-09-25T05:49:47Z):**

Sigmas are specific noise levels applied in the diffusion process.

For example, the sigmas for a three-step process at denoise level 1.0 would be the list **[1, 0.961165, 0.853333, 0.0]**. The first step takes you from noise level 1.0 to 0.961165, the second step takes you from noise 0.961165 to 0.85333 and so on; when you are at noise level 0 you have a latent that's ready to decode into video.

Split Sigmas separates the zero-indexed sigma list at the 'step' input. Using the example list above, if 'step' set to 1, the high_sigmas output would be the list [1, 0.961165], and the low_sigmas output would be the list [0.961165, 0.853333, 0.0]. Which means that high_sigmas is the equivalent of only doing the first step out of three and low_sigmas would be doing the last two steps out of three. You can connect a ComfyUI 'Preview as Text' node to any high_sigmas or low_sigmas output in the Split Sigmas node to see the lists.

Manual Sigmas allow you to supply a handmade list of sigmas. You can connect any Comfy node with a 'sigmas' output to a 'Preview as Text' node to see what a list of sigmas look like in general for manual tweaking.

In ComfyUI's sampler nodes, the 'output' is the latent returned with leftover noise and the 'denoised_output' is the latent with the leftover noise subtracted.

In the example workflow, the low res stage does first four of eight steps as specified by the high_sigmas list from the SplitSigmas node, which takes us roughly to a noise level of 0.9 (It's not necessary for the low-res stage to stop at noise level 0.9, and in fact it could be useful to fully denoise the low-res segment but it does save time on the low-res stage). We take the denoised_output from the low-res stage's SamplerCustomAdvanced probably because the leftover noise will cause problems with latent upscaling so it needs to be subtracted out. Then at the second stage we re-noise to bring us back to approximately 0.9 noise level (i.e. where we left off on the low res stage) and continue on using the sigmas list from the ManualSigmas node.

To adapt this to a new turbo LoRA, check the recommended sampler, scheduler and steps. For example, in the low-res stage's 8-step process you may need to adjust Split Sigmas to use the first 5 steps rather than the first 4 steps to get to roughly 0.9 noise level. Or, in the high-res stage's Manual Sigmas, you need to add a new list of sigmas to better reflect the 8-step LoRA's sigma schedule from 0.9 noise to 0.0 noise.







### #56 [closed] ISSUE: Fixing 'NestedTensor' object has no attribute 'dim'
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/56  by Tony2371 2026-09-15T17:40:22Z

For people who have the issue 'AttributeError: 'NestedTensor' object has no attribute 'dim'. Did you mean: 'ndim'? '

1. Add these helper functions near the top of your file, after the imports:
```
def _is_nested_tensor(obj):
    """Check if an object is a NestedTensor without relying on isinstance."""
    return getattr(obj, "is_nested", False) and callable(getattr(obj, "unbind", None))

def _tensor_ndim(obj):
    """Get the number of dimensions of a tensor or NestedTensor."""
    if _is_nested_tensor(obj):
        return obj.ndim  # NestedTensor only has .ndim, not .dim()
    return obj.dim()

def _split_av_latent(latent_samples):
    """
    Split a joint AV latent into video and audio components.
    Returns: (video_tensor, audio_tensor, original_container)
    """
    if not _is_nested_tensor(latent_samples):
        return latent_samples, None, None
    
    streams = list(latent_samples.unbind())
    if len(streams) != 2:
        raise ValueError(f"Expected 2 AV streams, got {len(streams)}")
    
    # Find the video stream (24 channels, 5D)
    video_idx = None
    for i, stream in enumerate(streams):
        if (torch.is_tensor(stream) and stream.ndim == 5 and stream.shape[1] == 24):
            video_idx = i
            break
    
    if video_idx is None:
        shapes = [tuple(s.shape) if torch.is_tensor(s) else None for s in streams]
        raise ValueError(f"No video stream (B,24,T,H,W) found in AV latent. Shapes: {shapes}")
    
    video = streams[video_idx]
    audio = streams[1 - video_idx]  # The other stream is audio
    return video, audio, latent_samples

def _rejoin_av_latent(video, audio, original_container):
    """Recombine upscaled video with the original audio into a NestedTensor."""
    if original_container is None or audio is None:
        return video
    
    rebuilt = [video, audio]
    try:
        # Reuse the original container's constructor (most reliable)
        return type(original_container)(rebuilt)
    except Exception:
        # Fallback to ComfyUI's NestedTensor
        try:
            from comfy.nested_tensor import NestedTensor
            return NestedTensor(rebuilt)
        except Exception:
            # Last resort: use torch's public API
            return torch.nested.as_nested_tensor(rebuilt)
```

2. Replace
```

 selected_mode = mode["mode"]
        src = latent["samples"]
        orig_dtype = src.dtype
        was_4d = (src.ndim() == 4)
```


with

```

selected_mode = mode["mode"]
        
        # --- Handle joint AV NestedTensor ---
        raw_samples = latent["samples"]
        video_src, audio_src, original_container = _split_av_latent(raw_samples)
        
        src = video_src  # Process only the video stream
        orig_dtype = src.dtype
        was_4d = (_tensor_ndim(src) == 4)  # Use helper, not .dim()
```


3. Replace
       ` return io.NodeOutput({"samples": out})`
with

```
# Rejoin with the untouched audio if we split an AV latent
        final_out = _rejoin_av_latent(out, audio_src, original_container)
        return io.NodeOutput({"samples": final_out})
```


This error  is an incompatibility between the node's code and newer PyTorch versions. Above instructions fixed the issue for me (pytorch 2.14.0 + cuda 13 as for the date of the issue)

> **Comment by kaalibro (2026-09-15T21:44:52Z):**

You simply didn't add the `LTXVSeparateAVLatent` and `LTXVConcatAVLatent` nodes before and after the `Minimax H3 Latent Upscaler` respectively, as shown in the [workflow template](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/workflow_templates/minimax_h3_r2v_Latent%20Upscaler%20example%20workflow3.json). Because of this, it receives a combined audio-visual latent as input, causing the `NestedTensor` error.

**Wrong:**

<img width="1780" height="997" alt="Image" src="https://github.com/user-attachments/assets/135f1f62-fbd3-4d4e-ba60-4950c12085ee" />

---

**Correct:**

<img width="1781" height="1081" alt="Image" src="https://github.com/user-attachments/assets/97e1f50d-b071-4204-b21a-2aad426bff54" />



### #55 [open] PR: Support Comfy's dynamic VRAM
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/pull/55  by Pizzawookiee 2026-09-11T05:19:07Z

## Summary

Improve ComfyUI dynamic VRAM support for the MiniMax H3 latent upscaler.

This updates the 2D/3D latent upscaler loading path to integrate more closely with ComfyUI's model-management system instead of keeping the upscaler weights resident through the node's own loading logic.

AFAIK dynamic VRAM only supported for safetensors checkpoints, non safetensors checkpoints should load but would not benefit from dynamic VRAM.

## Changes

* Use ComfyUI-compatible safetensors loading for the latent upscaler checkpoints.
* Replace applicable PyTorch layers with ComfyUI-managed equivalents so weights can participate in dynamic VRAM/offloading.
* Allow ComfyUI to load and offload upscaler weights according to available VRAM rather than requiring the entire model to stay resident.
* Preserve the existing `force_unload` option as a legacy/manual override.
* Keep existing node behavior and workflow compatibility otherwise unchanged.

## Why

The latent upscaler can consume a significant amount of VRAM alongside MiniMax H3, especially at larger latent resolutions and during temporally chunked upscaling.

Previously, its weights were largely outside ComfyUI's normal dynamic VRAM lifecycle, which could leave less memory available for H3 sampling and increase the likelihood of OOMs.

Moving the supported model components onto ComfyUI's managed loading path lets the upscaler cooperate with the same dynamic VRAM/offloading mechanisms used by the rest of ComfyUI, while retaining `force_unload` for users who prefer the previous explicit behavior.



> **Comment by abline11 (2026-10-22T08:01:03Z):**

## Confirmed Windows shared-GPU memory improvement with PR #55 — RTX 3070 Ti 8 GB / MiniMax H3

I have tested PR #55 against the original MiniMax H3 Latent Upscaler (3D) in my existing ComfyUI workflow and obtained a significant improvement in GPU memory management. I am reporting the findings because they appear directly relevant to the problem described in issue #62, where subsequent generations become extremely slow unless ComfyUI is restarted.

### Test environment

- Windows 11, NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB VRAM), 32 GB system RAM
- NVIDIA driver 610.60
- ComfyUI 0.42.1, revision `63be2f9b`
- Python 3.13.9, PyTorch 2.9.1+cu130, cuDNN 9.12
- comfy-aimdo 0.5.6, DynamicVRAM enabled
- MiniMax H3 video-generation workflow incorporating a first sampler, the 3D latent upscaler, Concat AV Latent and a second sampler
- FP16 upscaler execution with temporal chunking and `force_unload` enabled
- Launch parameters included **`--disable-pinned-memory` and `--disable-comfy-compiler`**, both retained during the successful patched test

### Problem with the original upscaler

During the first sampling pass, Windows Task Manager showed approximately **0.3 GB of shared GPU memory** in use.

As soon as execution reached the MiniMax H3 Latent Upscaler (3D), shared GPU memory increased above 1.1 GB. It subsequently dropped to approximately 1.1 GB but remained at that level throughout the second sampling pass.

Crucially, this **1.1 GB allocation persisted after generation completed**, despite model unloading, CUDA cache clearing, garbage collection and additional ComfyUI memory-cleanup nodes.

Dedicated GPU memory returned to approximately 0.3 GB, but shared GPU memory did not. Subsequent generations suffered substantial performance degradation, making consecutive queued generations impractical without restarting ComfyUI.

**Importantly, `--disable-pinned-memory` had already been enabled for several days before this investigation reached PR #55.** Although this flag had helped avoid earlier CUDA failures, it did not eliminate the persistent 1.1 GB shared GPU memory allocation.

This distinguishes the two problems: disabling pinned memory improved generation stability but did not correct the upscaler's memory-retention behaviour.

### Test with PR #55

I retained the existing ComfyUI installation, launch parameters, models and workflow settings. I replaced only the 3D upscaler implementation and added the `comfy_dynamic.py` helper from the contributor's PR #55 branch.

**The `--disable-pinned-memory` flag remained enabled throughout both the original and patched tests. No pinned-memory configuration was changed when installing PR #55.**

The patched upscaler loaded and executed successfully. The console confirmed:

```text
DynamicVRAM: on
Model LatentResizer3D prepared for dynamic VRAM loading.
658MB Staged.
[MinimaxH3-3D] temporal chunking: T=57 chunks=2 overlap=5
[MinimaxH3-3D] ✅ Legacy force_unload: model offloaded through ComfyUI.
```

### Results

**Shared GPU memory remained at approximately 0.3 GB after the patched upscaler executed and the second sampler began.**

After the entire generation completed, Windows Task Manager showed:

- Dedicated GPU memory: 0.3 GB
- Shared GPU memory: **0.3 GB**
- No persistent 1.1 GB shared-memory allocation

The generation completed successfully in 13 minutes 34 seconds, compared with 16 minutes 37 seconds for the previous unpatched single run. Those single-run timings are effectively equivalent; the important improvement is the elimination of the persistent shared-memory allocation that previously affected later generations.

This is a confirmed successful single-generation test. Longer sequences of consecutive queued generations have not yet been formally validated with the patch.

Because `--disable-pinned-memory` was enabled before and after the patch, **the observed improvement cannot be attributed to newly disabling pinned memory**. The improvement occurred after installing the DynamicVRAM-integrated upscaler.

### Conclusion and request

These results provide strong evidence that the original upscaler's memory-management implementation contributes to the persistent Windows shared-GPU memory problem, and that PR #55's integration with ComfyUI's DynamicVRAM management addresses it on this configuration.

The before-and-after difference is particularly clear because the memory increase previously occurred at the 3D upscaler, whereas the patched implementation allowed shared memory to remain at its original level through the second sampler and after final cleanup.

The pinned-memory setting is an important distinction: **`--disable-pinned-memory` alone did not solve the memory-retention problem, whereas installing PR #55 with that setting unchanged eliminated the observed 1.1 GB residue.**

**I recommend that the maintainers review and test PR #55 for inclusion in the official extension**, particularly for users running MiniMax H3 on Windows with limited dedicated VRAM. The problem significantly affects the practicality of multi-generation workflows.

Related report: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/62

I can provide the original and patched execution logs and Windows Task Manager screenshots if useful.



### #54 [open] ISSUE: Please add a LICENSE file
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/54  by JackZ2024 2026-09-10T21:38:30Z

Hi! Thanks for sharing this project.

I noticed that there is currently no LICENSE file in the repository. Could you please add an appropriate open-source license (such as MIT, if that's what you intend)?

This would help users understand whether they are allowed to use, modify, and redistribute the code, especially for integration into other projects.

Thanks!

> **Comment by ILG2021 (2026-09-17T08:40:24Z):**

@LBH-123-AI 


> **Comment by LBH-123-AI (2026-09-17T09:55:38Z):**

Thanks for the nudge, @JackZ2024 (and @ILG2021) — you're right, the repo had no license.

I've added an **MIT** license to the `main` branch (see [LICENSE](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/LICENSE)). You're free to use, modify, merge, publish and redistribute the code, including embedding it into other projects, as long as the copyright notice is retained.

Thanks for the suggestion!



### #53 [open] PR: Fix mixed BF16/FP16 checkpoint support in the 3D upscaler
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/pull/53  by ethanfel 2026-09-09T21:55:36Z

The 3D loader currently casts every parameter to the selected precision. With `precision=bf16`, this rounds away the 150 FP16 fallback tensors in [Asirus/Minimax-H3-Latent-Upscaler-BF16-MAXQUALITY](https://huggingface.co/Asirus/Minimax-H3-Latent-Upscaler-BF16-MAXQUALITY).

This change detects mixed BF16/FP16 or BF16/FP32 checkpoints from their tensor dtypes. Layers containing fallback parameters run in FP32, preserving their stored values and avoiding FP16 activation overflow; the remaining layers run in BF16. Detection includes both weights and biases, which can have different dtypes within the same layer.

- Cast activations at Conv3d, Linear, and GroupNorm boundaries, and retain fallback precision for embedding modulation and temporal-chunk blending.
- Keep uniform `fp16`/`fp32` selections and ordinary single-dtype checkpoint behavior.
- Add regression tests, a reproducible CUDA benchmark with raw results, and English/Chinese usage documentation.

Validation completed:

- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`: 8 tests passed.
- Verified the actual checkpoint's SHA256 and exact preservation of all 322 parameter tensors; CPU inference passed.
- Local RTX 5090 checks passed for all three sizing modes, image/video inputs, and GPU offload/reload with mixed dtypes retained.
- Actual-checkpoint GPU benchmarks produced finite outputs for short, larger, and temporally chunked inputs.

RTX 5090 / PyTorch 2.11.0 results, using 2× spatial upscaling and the median of seven runs after two warmups:

| Output size / latent time steps | Uniform BF16 | BF16 + FP32 fallback | Runtime increase |
| --- | ---: | ---: | ---: |
| 384×256 / 2 | 16.1 ms | 18.9 ms | 17.4% |
| 1280×768 / 8 | 83.0 ms | 88.1 ms | 6.1% |
| 640×384 / 33, temporal chunking | 142.3 ms | 159.9 ms | 12.4% |

Peak allocated GPU memory increased by 124.4 MiB and 149.5 MiB in the two larger cases. Timings use synthetic normalized latents and exclude loading, offload, and VAE decoding. The GPU was shared with an existing process. Rendered-video quality has not been evaluated. Raw samples and an experimental FP16 fallback comparison are included in `tests/benchmark_results/rtx5090_asirus.json`.



### #52 [open] ISSUE: Idea: place the tiles from the take's own motion (a focus tile on the subject, a ring of edge tiles), so the bands land on static background
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/52  by 535gfx 2026-09-09T12:17:26Z


This is a proposal with a working prototype, in case it is useful to you.

The 2x2 grid puts its bands wherever the arithmetic puts them, and when a band crosses a moving object the re-sample doubles it. The attached clip is the same take through 2x2 1568x896, where the horizontal band runs through the hand (the fingers multiply, the blade smears), and through 2x2 1568x1280, where the rows overlap 1024 px and the band sits off the hand (clean). On a static-camera shot, which is most of what H3 gets used for, the band position is the whole game.

What we built on top of your helpers (imported, not modified): a layout step that thresholds frame-to-frame motion over the whole take (threshold 20, blobs over 3 percent of the frame), takes their bounding box as a focus tile, and fills the rest of the frame with a ring of edge tiles. A node then renders the tiles in that order, focus first and ring after, with your band law generalised to two dimensions: chessboard distance to the seam, the done side frozen beyond the fade and ramped inside it, a crossfade over the band. Every pixel is refined, the only blend lines sit on background, and each job stays under the card's paging line. On the kitchen shot the two full-height columns at 132k to 138k tokens paged at 23.5 GB, while the focus tile at 129k plus three ring tiles of 60k to 102k rendered in 16 minutes for 192 frames.

The overlay attached is the layout tool's output on that shot: red is where the take moves, yellow the focus tile with its token count, cyan the ring tiles and their bands. No model in the loop, just frame differences. A face detector could pick the focus when several things move; we have not needed it yet.

One caveat we hit and fixed on the way: with seam_denoise below 1 the ring tiles pulse (#51); at 1.0 they are clean.

Happy to share the code, two small files, torch-only geometry plus your functions. I could not find a license file in the repo, so tell me which license you use and I will match it.

<img width="1344" height="768" alt="Image" src="https://github.com/user-attachments/assets/b5b3c97b-7d73-4df4-8d11-912183eae7bf" />
<img width="2756" height="930" alt="Image" src="https://github.com/user-attachments/assets/5d6c7856-04a4-4f9f-adbd-6a854c69b40d" />

https://github.com/user-attachments/assets/18f6cb15-93b2-422c-887e-d6da4c9d3da9


> **Comment by 535gfx (2026-09-09T13:00:07Z):**

One more number, since speed came up in #28: on the same 4090 the placed plan at your 0.6316 / 0.2 / 0 ladder does the whole 192-frame take in 12.0 min, every pixel refined, no temporal seam. Same take, same box: the chunked full-frame refine 15.5 to 19 min, the 2x2 1568x1280 grid 14.0, the placed plan at the 3-step ladder 16.0. Table attached, everything measured here; the SeedVR2 row is one attempt at batch 33 that paged, so take that one as unfinished.

<img width="2000" height="1673" alt="Image" src="https://github.com/user-attachments/assets/fe49ff61-dee2-4dba-9788-0f8283a29b74" />



### #51 [open] ISSUE: Split Upscale: seam_denoise below 1.0 puts a 17-frame brightness pulse into every tile (cause and fix measured)
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/51  by 535gfx 2026-09-09T12:15:54Z


Hi, and thanks for the pack. The 3D latent upscaler plus the 3-step refine is our main 2x path for 8-second H3 takes on a 4090, and Split Upscale is the only way we found to refine a 192-frame take without a temporal seam.

While testing it on a static kitchen shot we noticed a faint strobe in the tiles around the subject: the brightness drops a step every 17 frames and climbs back over the next 16. It shows on flat bright surfaces (a fridge door, a bare wall) and hides on dark, cluttered backgrounds. Measured as mean luma per frame of a static strip, frame-to-frame:

- native take: std 0.05, no period
- full-frame chunked refine, no tiles: 0.08, flat
- Split Upscale 2x2 1568x1280, seam_denoise 0.7: 0.19, a drop of about 0.65 at every block edge
- a narrow 672x1536 edge tile in our placed-tile variant of the node (same band law as yours, I'll open a separate issue about it): 0.91, a drop of 3.3 at frames 17, 34, 51 ... 187, autocorrelation 0.99 at lag 17

The period is the VAE's temporal block (17 pixel frames per 5 latent frames). Same-seed A/B on that 672x1536 tile:

- the tile alone, nothing frozen: flat (0.07)
- a 384-wide first tile beside it, then the tile at seam_denoise 0.7: the pulse, 3.1
- the same two tiles at seam_denoise 1.0: flat (0.06)

So it is not the tile size and not the colour match (the DC match is one constant per channel over all frames). It is the free-side re-sample. With seam_denoise below 1 the fresh side of the band gets a denoise mask that ramps from the cap to 1, and ComfyUI's inpaint wrapper mixes (1 - mask) of the tile's latent_image into the sample at every step and into the result. For fresh pixels that latent_image is the raw upscaler output, so the band ends up a per-voxel blend of the refined latent and the upscaler latent, and that blend decodes with a step at the start of every block. The narrower the fresh region relative to the band, the stronger the pulse: the 672 tile's whole fresh strip (18 latent columns) sat inside the 24-column ramp, which is why it was the worst.

seam_denoise 1.0 fixes it in our tests: the whole four-tile plan re-rendered at 1.0 has no 17-frame period left in any tile (that fridge strip goes from 0.91 to 0.11, the wall strip to the native's 0.04). The done-side fade still crossfades the seam, so a moving object still crosses it through the frozen side's ramp. If a free-side re-sample is wanted, I think it would need to re-sample against the tile's own first pass rather than against the source latent.

Attached: the placed file next to its frame-to-frame difference amplified 12x (the flash outside the focus tile every 17 frames), a still of that difference at f16 to f17, and the same difference view at 0.7 and 1.0 side by side.

Setup: ComfyUI 0.34.1, pack at d7c01b9, MiniMax H3 FL2VA bf16, sigmas 0.9035 / 0.6316 / 0.3158 / 0, res_multistep, cfg 1, 1344x768 to 2688x1536, 192 frames, RTX 4090 24 GB, reserve-vram 1.0. Related: #37.

https://github.com/user-attachments/assets/4c43ea1c-2880-4db9-9ea6-9283ab3b5358

<img width="4088" height="938" alt="Image" src="https://github.com/user-attachments/assets/455283a2-43fc-4c3c-a4f3-a8c50ad6de4f" />

https://github.com/user-attachments/assets/156f1766-c8bc-4515-ac15-28ff0508dc0f


### #50 [open] ISSUE: amd 8060s throw error
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/50  by wszgrcy 2026-09-06T01:08:38Z

```
[MinimaxH3-3D] Latent 30x54 -> 60x108 | Pixels 960x1728 | scale=2.000
[MinimaxH3-3D] temporal chunking: T=72 chunks=3 overlap=5
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.36s/it]
Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT

****************************************************************************
*** GPU HANG ANALYSIS - VGPU(0x207647f0) Queue(0x73f03e4b3000)
****************************************************************************
SWq=0x73f03e4b3000, HWq=0x73efa3000000, id=1, priority=1, Dispatch Header = 0xb02 (type=2, barrier=1, acquire=1, release=1), setup=3, grid=[122752, 56, 1], workgr
oup=[128, 1, 1], private_seg_size=0, group_seg_size=16, kernel_obj=0x73ef8c018780, kernarg_address=0x73efa1608880, completion_signal=0x0, rptr=20942038, wptr=2094
7410
****************************************************************************

:0:rocdevice.cpp            :3942: 565573844574 us:  [host: amd395, GPU index: 0, faulting addr: 0x73e067950000, kernel: quant_per_block_int8_kernel]
/home/chen/ComfyUI/comfy/nested_tensor.py:45: UserWarning: uncheckedSetDevice() ignoring error: CUDA error: an illegal memory access was encountered
Search for `hipErrorIllegalAddress' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing AMD_SERIALIZE_KERNEL=3
Device-side assertion tracking was not enabled by user.
Exception raised from SetDevice at /__w/rockrel/rockrel/external-builds/pytorch/pytorch/c10/hip/HIPFunctions.cpp:334 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x9d (0x73f1c6774f4d in /home/
chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x1402a (0x73f1f1d9f02a in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libc10_hip.so)
frame #2: c10::cuda::SetDevice(signed char, bool) + 0x65 (0x73f1f1decda5 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libc10_hip.so)
frame #3: <unknown function> + 0xe4a17b (0x73f1c764a17b in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_hip.so)
frame #4: <unknown function> + 0x1cb10cc (0x73f1c84b10cc in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_hip.so)
frame #5: <unknown function> + 0x1af92ae (0x73f1e3ef92ae in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so)
frame #6: at::native::copy_(at::Tensor&, at::Tensor const&, bool) + 0x30 (0x73f1e3efb850 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtor
ch_cpu.so)
frame #7: at::_ops::copy_::call(at::Tensor&, at::Tensor const&, bool) + 0x142 (0x73f1e4de5e32 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/l
ibtorch_cpu.so)
frame #8: at::native::_to_copy(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, boo
l, std::optional<c10::MemoryFormat>) + 0x16ca (0x73f1e42f33ea in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so)
frame #9: <unknown function> + 0x2e83fdc (0x73f1e5283fdc in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so)
frame #10: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::D
evice>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0xe2 (0x73f1e489c772 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/lib
torch_cpu.so)
frame #11: <unknown function> + 0x2bf3a5b (0x73f1e4ff3a5b in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so)
frame #12: at::_ops::_to_copy::call(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>
, bool, std::optional<c10::MemoryFormat>) + 0x1e4 (0x73f1e494a864 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so)
frame #13: at::native::to(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, bo
ol, std::optional<c10::MemoryFormat>) + 0xc4 (0x73f1e42dbb64 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so)
frame #14: <unknown function> + 0x2fce432 (0x73f1e53ce432 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so)
frame #15: at::_ops::to_dtype_layout::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional
<c10::Device>, std::optional<bool>, bool, bool, std::optional<c10::MemoryFormat>) + 0x10a (0x73f1e4a70d6a in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so)
frame #16: <unknown function> + 0x2bf3935 (0x73f1e4ff3935 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so)
frame #17: at::_ops::to_dtype_layout::call(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, bool, std::optional<c10::MemoryFormat>) + 0x1f9 (0x73f1e4af7389 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so)
frame #18: <unknown function> + 0x45994b (0x73f1f0a5994b in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_python.so)
frame #19: <unknown function> + 0x4bbd27 (0x73f1f0abbd27 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_python.so)
frame #20: python() [0x1831544]
<omitting python frames>
frame #22: python() [0x1817178]
frame #24: python() [0x19f158f]
frame #25: python() [0x4c36c8]
frame #26: python() [0x4c34b5]
frame #27: python() [0x1802e2c]
frame #28: python() [0x48f029]
frame #29: python() [0x48ef5a]
frame #30: python() [0x1803f55]
frame #32: python() [0x1817111]
frame #33: python() [0x19c03d2]
frame #34: python() [0x19c0324]
frame #35: <unknown function> + 0xa407a (0x73f3886a407a in /usr/lib/x86_64-linux-gnu/libc.so.6)
frame #36: <unknown function> + 0x13772c (0x73f38873772c in /usr/lib/x86_64-linux-gnu/libc.so.6)
 (Triggered internally at /__w/rockrel/rockrel/external-builds/pytorch/pytorch/c10/hip/impl/HIPGuardImpl.h:70.)
  o.tensors[i] = t.to(*args, **kwargs)   
[ERROR] !!! Exception during processing !!! CUDA error: an illegal memory access was encountered
Search for `hipErrorIllegalAddress' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing AMD_SERIALIZE_KERNEL=3
Device-side assertion tracking was not enabled by user.
[ERROR] Traceback (most recent call last):
  File "/home/chen/ComfyUI/execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chen/ComfyUI/execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chen/ComfyUI/execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)  
  File "/home/chen/ComfyUI/execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "/home/chen/ComfyUI/comfy_api/internal/__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "/home/chen/ComfyUI/comfy_api/latest/_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "/home/chen/ComfyUI/comfy_extras/nodes_custom_sampler.py", line 1067, in execute
    x0_out = guider.model_patcher.model.process_latent_out(x0.cpu())
                                                           ~~~~~~^^
  File "/home/chen/ComfyUI/comfy/nested_tensor.py", line 55, in cpu
    return self.to(device="cpu")
           ~~~~~~~^^^^^^^^^^^^^^
  File "/home/chen/ComfyUI/comfy/nested_tensor.py", line 45, in to
    o.tensors[i] = t.to(*args, **kwargs) 
                   ~~~~^^^^^^^^^^^^^^^^^ 
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `hipErrorIllegalAddress' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing AMD_SERIALIZE_KERNEL=3
Device-side assertion tracking was not enabled by user.

[INFO] Prompt executed in 449.49 seconds
[W906 00:48:37.636952817 CachingHostAllocator.cpp:27] Warning: Exception in pinned allocator free(), rethrowing (function free)
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: an illegal memory access was encountered
Search for `hipErrorIllegalAddress' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing AMD_SERIALIZE_KERNEL=3
Device-side assertion tracking was not enabled by user.
Exception raised from ExchangeDevice at /__w/rockrel/rockrel/external-builds/pytorch/pytorch/c10/hip/HIPFunctions.cpp:347 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x9d (0x73f1c6774f4d in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x1402a (0x73f1f1d9f02a in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libc10_hip.so)
frame #2: c10::cuda::ExchangeDevice(signed char) + 0x51 (0x73f1f1dec821 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libc10_hip.so)
frame #3: <unknown function> + 0xe6bebf (0x73f1c766bebf in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_hip.so)
frame #4: <unknown function> + 0xe70255 (0x73f1c7670255 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_hip.so)
frame #5: <unknown function> + 0xe68bdb (0x73f1c7668bdb in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_hip.so)
frame #6: <unknown function> + 0x888b4 (0x73f1c67538b4 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libc10.so)
frame #7: c10::TensorImpl::~TensorImpl() + 0x9 (0x73f1c674d3d9 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libc10.so)
frame #8: <unknown function> + 0x8a6695 (0x73f1f0ea6695 in /home/chen/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib/libtorch_python.so)
frame #9: python() [0x1ac1f1f]
frame #10: python() [0x19b014d]
<omitting python frames>
frame #12: python() [0x1817111]
frame #13: python() [0x19c03d2]
frame #14: python() [0x19c0324]
frame #15: <unknown function> + 0xa407a (0x73f3886a407a in /usr/lib/x86_64-linux-gnu/libc.so.6)
frame #16: <unknown function> + 0x13772c (0x73f38873772c in /usr/lib/x86_64-linux-gnu/libc.so.6)

Fatal Python error: Aborted

Stack (most recent call first):
  File "/home/chen/ComfyUI/main.py", line 437 in prompt_worker
  File "/home/chen/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/threading.py", line 995 in run
  File "/home/chen/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/threading.py", line 1044 in _bootstrap_inner
  File "/home/chen/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/lib/python3.13/threading.py", line 1015 in _bootstrap

Extension modules: sqlalchemy.cyextension.collections, sqlalchemy.cyextension.immutabledict, sqlalchemy.cyextension.processors, sqlalchemy.cyextension.resultproxy, sqlalchemy.cyextension.util, greenlet._greenlet, markupsafe._speedups, yaml._yaml, PIL._imaging, multidict._multidict, yarl._quoting_c, propcache._helpers_c, ai
ohttp._http_writer, aiohttp._http_parser, aiohttp._websocket.mask, aiohttp._websocket.reader_c, frozenlist._frozenlist, chardet.models, chardet.pipeline.ascii, chardet.pipeline.confusion, chardet.pipeline.escape, chardet.pipeline.magic, chardet.pipeline.statistical, chardet.pipeline.structural, chardet.pipeline.utf8, chardet.pipeline.utf1632, chardet.pipeline.validity, chardet.pipeline.orchestrator, charset_normalizer.md, charset_normalizer.cd, requests.packages.chardet.models, requests.packages.chardet.pipeline.ascii, requests.packages.chardet.pipeline.confusion, requests.packages.chardet.pipeline.escape, requests.packages.chardet.pipeline.magic, requests.packages.chardet.pipeline.statistical, requests.packages.chardet.pipeline.structural, requests.packages.chardet.pipeline.utf8, requests.packages.chardet.pipeline.utf1632, requests.packages.chardet.pipeline.validity, requests.packages.chardet.pipeline.orchestrator, numpy._core._multiarray_umath, numpy.linalg._umath_linalg, torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._pcg64, numpy.random._generator, numpy.random._mt19937, numpy.random._philox, numpy.random._sfc64, numpy.random.mtrand, psutil._psutil_linux, PIL._imagingft, _cyutility, scipy._cyutility, scipy._lib._ccallback_c, scipy.ndimage._nd_image, scipy.ndimage._rank_filter_1d, scipy.special._ufuncs_cxx, scipy.linalg._fblas, scipy.linalg._flapack, scipy.linalg._batched_linalg, scipy.linalg._cythonized_array_utils, scipy.linalg._solve_toeplitz, scipy.linalg._internal_matfuncs, scipy.linalg._linalg_pythran, scipy.linalg.cython_blas, scipy.linalg.cython_lapack, scipy.linalg._decomp_update, scipy.special._ellip_harm_2, scipy.special._special_ufuncs, scipy.special._gufuncs, scipy.special._ufuncs, scipy.special._specfun, scipy.special._comb, _ni_label, scipy.ndimage._ni_label, regex._regex, scipy.integrate._odepack, scipy.integrate._quadpack, scipy.integrate._vode, scipy.integrate._dop, scipy.sparse._sparsetools, _csparsetools, scipy.sparse._csparsetools, scipy.sparse.linalg._dsolve._superlu, scipy.sparse.linalg._eigen.arpack._arpacklib, scipy.sparse.linalg._propack, scipy.optimize._group_columns, scipy._lib.messagestream, scipy.optimize._trlib._trlib, scipy.optimize._lbfgsb, _moduleTNC, scipy.optimize._moduleTNC, scipy.optimize._slsqplib, scipy.optimize._minpack, scipy.optimize._lsq.givens_elimination, scipy.optimize._zeros, scipy._lib._uarray._uarray, scipy.linalg._decomp_interpolative, scipy.optimize._bglu_dense, scipy.optimize._lsap, scipy.spatial._ckdtree, scipy.spatial._hausdorff, scipy.spatial._distance_wrap, scipy.spatial._qhull, scipy.spatial._voronoi, scipy.spatial.transform._rotation_cy, scipy.spatial.transform._rigid_transform_cy, scipy.optimize._direct, scipy.interpolate._fitpack, scipy.interpolate._dierckx, scipy.interpolate._ppoly, scipy.interpolate._interpnd, scipy.interpolate._rbfinterp_pythran, scipy.interpolate._rgi_cython, scipy.special.cython_special, scipy.stats._stats, scipy.stats._biasedurn, scipy.stats._levy_stable.levyst, scipy.stats._stats_pythran, scipy.stats._ansari_swilk_statistics, scipy.sparse.csgraph._tools, scipy.sparse.csgraph._shortest_path, scipy.sparse.csgraph._traversal, scipy.sparse.csgraph._min_spanning_tree, scipy.sparse.csgraph._flow, scipy.sparse.csgraph._matching, scipy.sparse.csgraph._reordering, scipy.stats._sobol, scipy.stats._qmc_cy, scipy.stats._rcont.rcont, scipy.stats._qmvnt_cy, av._core, av.logging, av.buffer, av.audio.format, av.error, av.dictionary, av.container.pyio, av.format, av.index, av.utils, av.stream, av.container.streams, av.sidedata.encparams, av.sidedata.motionvectors, av.sidedata.sidedata, av.opaque, av.packet, av.container.input, av.bitstream, av.container.output, av.container.core, av.codec.context, av.video.format, av.video.reformatter, av.plane, av.video.plane, av.video.frame, av.video.stream, av.codec.hwaccel, av.codec.codec, av.frame, av.audio.layout, av.audio.plane, av.audio.frame, av.audio.stream, av.filter.link, av.filter.context, av.filter.graph, av.filter.filter, av.filter.loudnorm, av.audio.resampler, av.audio.codeccontext, av.audio.fifo, av.device, av.video.codeccontext, av.subtitles.stream, _cffi_backend, scipy.signal._sigtools, scipy.signal._max_len_seq_inner, scipy.signal._upfirdn_apply, scipy.signal._spline, scipy.signal._sosfilt, scipy.signal._peak_finding_utils, sklearn.__check_build._check_build, sklearn._cyutility, sklearn.utils._isfinite, sklearn.utils.sparsefuncs_fast, sklearn.utils.murmurhash, sklearn.utils._openmp_helpers, sklearn.metrics.cluster._expected_mutual_info_fast, sklearn.metrics._dist_metrics, sklearn.metrics._pairwise_distances_reduction._datasets_pair, sklearn.utils._cython_blas, sklearn.metrics._pairwise_distances_reduction._base, sklearn.metrics._pairwise_distances_reduction._middle_term_computer, sklearn.utils._heap, sklearn.utils._sorting, sklearn.metrics._pairwise_distances_reduction._argkmin, sklearn.metrics._pairwise_distances_reduction._argkmin_classmode, sklearn.utils._vector_sentinel, sklearn.metrics._pairwise_distances_reduction._radius_neighbors, sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode, sklearn.metrics._pairwise_fast, sklearn.preprocessing._csr_polynomial_expansion, sklearn.preprocessing._target_encoder_fast, simsimd, stringzilla, PIL._imagingcms, PIL._imagingmath, kiwisolver._cext, skimage.measure._ccomp, skimage.measure._moments_cy, skimage.measure._find_contours_cy, skimage.measure._marching_cubes_lewiner_cy, hip_utils (total: 230)
```


# ComfyUI Error Report
## Error Details
- **Node ID:** 144
- **Node Type:** SamplerCustomAdvanced
- **Exception Type:** torch.AcceleratorError
- **Exception Message:** torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `hipErrorIllegalAddress' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing AMD_SERIALIZE_KERNEL=3
Device-side assertion tracking was not enabled by user.

## Stack Trace
```
  File "/home/chen/ComfyUI/execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/chen/ComfyUI/execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/chen/ComfyUI/execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "/home/chen/ComfyUI/execution.py", line 306, in process_inputs
    result = f(**inputs)

  File "/home/chen/ComfyUI/comfy_api/internal/__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)

  File "/home/chen/ComfyUI/comfy_api/latest/_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)

  File "/home/chen/ComfyUI/comfy_extras/nodes_custom_sampler.py", line 1067, in execute
    x0_out = guider.model_patcher.model.process_latent_out(x0.cpu())
                                                           ~~~~~~^^

  File "/home/chen/ComfyUI/comfy/nested_tensor.py", line 55, in cpu
    return self.to(device="cpu")
           ~~~~~~~^^^^^^^^^^^^^^

  File "/home/chen/ComfyUI/comfy/nested_tensor.py", line 45, in to
    o.tensors[i] = t.to(*args, **kwargs)
                   ~~~~^^^^^^^^^^^^^^^^^

```
## System Information
- **ComfyUI Version:** 0.34.0
- **Arguments:** main.py --enable-manager --listen 0.0.0.0 --gpu-only --disable-mmap --cache-none --use-sage-attention --disable-pinned-memory --enable-cors-header *
- **OS:** linux
- **Python Version:** 3.13.14 (main, Jul 18 2026, 17:02:37) [Clang 22.1.3 ]
- **Embedded Python:** false
- **PyTorch Version:** 2.15.0a0+rocm10.1.0a20260905
## Devices

- **Name:** cuda:0 AMD Radeon 8060S Graphics : native
  - **Type:** cuda
  - **VRAM Total:** 130996502528
  - **VRAM Free:** 130240712704
  - **Torch VRAM Total:** 81788928
  - **Torch VRAM Free:** 2097152

## Logs
```
Failed to retrieve server logs
```
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
```
Workflow too large. Please manually upload the workflow from local file system.
```

## Additional Context
(Please add any additional context or steps to reproduce the error here)


When I was testing, I set a very small resolution/duration, and during the step (0.2/1/2), it ran successfully. However, once I adjusted it and tried to run for a longer time, it seemed to report an error. I'm not sure whether this is due to the parameters or an AMD driver issue.


### #49 [open] ISSUE: What model/turbo lora (or none) should your sigma values be used with?
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/49  by slikvik55 2026-09-04T14:33:04Z

I'm using lightx2v and wondering if the sigmas are wrong for that?

Also, can the custom sigmas be swapped for a scheduler and denoise (say 0.25) instead?


### #48 [open] ISSUE: [Bug] identity_anchor_frames > 0 导致周期性白闪 / 曝光跳变
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/48  by nipluo8495 2026-09-03T17:56:06Z

你好，感谢更新 MMH3 Split Upscale / Temporal Split Params，分段二采对显存帮助很大。

我在 MiniMax H3 测试中发现一个疑似与 identity_anchor_frames 有关的问题：

使用 MMH3 Temporal Split Params 时，只要 identity_anchor_frames 设置为任意大于 0 的值（我测试了 1、5、24），二采输出就会出现规律性的白色亮度闪烁 / 曝光跳变。设置为 0 后，闪烁立即完全消失。

已测试并排除：
- 一采原始视频稳定；
- 潜空间放大器关闭后仍会闪；
- color_match 开关无影响；
- BF16 / FP16 无影响；
- CFG 从 1.0 降到 0.6 仍闪；
- 不使用 LoRA 仍闪；
- 绕过 Sage Attention 仍闪；
- chunk_frames 设为 999（尝试单段处理）仍闪；
- motion_anchor_frames 设置为 5 或 22 均不闪。

当前稳定配置：
chunk_frames = 73
temporal_overlap_frames = 22
anchor_strength = 0.999
motion_anchor_frames = 22
identity_anchor_frames = 0

推测 identity anchor 在 H3 latent 中可能同时把前一段的曝光/高光信息写回了后续片段，而不只是身份细节，因此在白色高光数字人场景中会产生周期性白闪。若方便，建议检查 identity anchor 是否应仅在真实分段边界启用，并考虑降低默认强度或采用渐变混合。

感谢！


### #47 [open] ISSUE: Bug: MinimaxH3LatentUpscaler3D crashes on H3's joint video+audio AV latent (NestedTensor)
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/47  by AI-KIMCHY 2026-09-02T02:50:55Z

Summary

The 3D upscaler node has no handling at all for MiniMax H3's joint AV latent (video 24ch + audio 32ch packed together). Any workflow that feeds the raw output of an H3 sampler (SamplerCustomAdvanced, MiniMaxH3TurboSampler, etc.) directly into MinimaxH3LatentUpscaler3D crashes, because execute() treats latent["samples"] as a plain 4D/5D torch.Tensor and never checks whether it's actually the AV-nested object.

Environment
ComfyUI: 0.34.0
PyTorch: 2.13.0+cu130
GPU: RTX 5070 Ti (CUDA)
Node file: nodes/minimax_h3_latent_upscaler_3d.py (unmodified, from this repo)
Steps to reproduce
Build a workflow where an H3 sampler's LATENT output (video+audio NestedTensor) is wired directly into MinimaxH3LatentUpscaler3D (no audio-separation node in between).
Queue the prompt.
Sampling completes normally, then the upscaler node throws.
Observed error
TypeError: nested_to_padded_tensor(): argument 'input' (position 1) must be Tensor, not NestedTensor

(from inside the 3D conv backbone's forward pass — the tensor never gets unwrapped before reaching F.interpolate/Conv3d calls).

Root cause

execute() reads src = latent["samples"] and immediately does src.dim(), src.to(...), src.shape unpacking, etc., all of which assume a plain Tensor. When latent["samples"] is actually the AV-joint object H3 samplers emit (video (B,24,T,H,W) + audio (B,32,2,T_audio) bundled together), those calls either silently misbehave or blow up deep inside the model's forward pass.

Two additional wrinkles that are easy to trip over when fixing this:

The AV object doesn't pass isinstance(x, torch.Tensor). It behaves like a NestedTensor (.is_nested, .unbind(), .dtype, .ndim all work) but is not a torch.Tensor subclass in this environment/PyTorch build — presumably a project-specific wrapper class rather than torch.nested's own class. Any nested-detection code needs to key off hasattr(x, "is_nested") rather than isinstance(..., torch.Tensor).
It doesn't expose .dim(), only .ndim (PyTorch 2.13's own NestedTensor also dropped .dim() in favor of .ndim).
Video and audio have different rank (5D vs 4D) within the same AV object, and torch.nested.nested_tensor() / torch.nested.as_nested_tensor() both hard-reject that ("all tensors must have the same dimension"). So re-packing the upscaled video + untouched audio back into an AV object for downstream nodes (e.g. MMH3SplitUpscale) has to reuse the original AV wrapper class's own constructor, not torch's public nested-tensor constructors.
Proposed fix

Patch attached (fix_clean.patch) — tested locally, resolves the crash in a real workflow (H3 sampler → MinimaxH3LatentUpscaler3D → MMH3SplitUpscale → VAEDecodeAudio). Summary of the change:

Adds _is_nested_tensor() / _tensor_ndim() helpers that detect and measure the AV object without relying on isinstance(..., torch.Tensor) or .dim().
Adds _split_av_latent(): if latent["samples"] is AV-nested, unbind it, identify the video component by its 24-channel dim, and return (video_tensor, audio_tensor, original_type). Raises a clear error instead of failing silently if the structure doesn't match the expected [video, audio] shape (e.g. more than 2 components, or the video component still being nested).
Adds _rejoin_av_latent(): re-attaches the untouched audio to the upscaled video, reconstructing via the original AV wrapper class first (orig_type([video, audio])), falling back to torch.nested.as_nested_tensor() / torch.nested.nested_tensor() only if that's unavailable.
execute() now calls _split_av_latent() at the top and _rejoin_av_latent() right before returning; everything in between is unchanged, so image-only (non-AV) latents still take the exact same path as before.

Happy to open this as a PR instead of just attaching a patch if that's preferred — let me know.

Patch
diff
--- a/nodes/minimax_h3_latent_upscaler_3d.py
+++ b/nodes/minimax_h3_latent_upscaler_3d.py
(see attached fix_clean.patch for the full diff)

> **Comment by Filipus24 (2026-09-05T23:53:51Z):**

@AI-KIMCHY : the patch file isn't attached.



### #46 [closed] PR: 为 3D 放大节点增加可调时序分块大小，降低 8GB 显卡 Conv3D 峰值
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/pull/46  by gosick233-cloud 2026-09-01T05:11:12Z

## 背景

目前 3D latent 放大节点把时序分块大小固定为 32。对于显存较小的显卡，分块再加上前后 overlap 后，单次 Conv3D 的激活峰值仍可能超过显存。

本地复现环境：

- NVIDIA RTX 4060 8GB
- 输入 latent：`[1, 24, 37, 22, 38]`
- 2 倍放大：`608×352 → 1216×704`
- BF16，时序分块已开启
- 分块 32 时在 `_forward_seg()` 的 `Conv3D` 处 OOM

这与 #18 中反馈的长视频 3D Conv 激活峰值问题属于同一类情况。

## 改动

- 为 `MinimaxH3LatentUpscaler3D` 增加 `temporal_chunk_frames` 下拉选项：`4 / 8 / 12 / 24 / 32`。
- 默认值仍为 `32`，因此没有修改现有默认行为。
- 将选定值传入 `LatentResizer3D.forward()`，只替换原来写死的 `chunk = 32`。
- 完整保留现有的零拷贝加载、`.contiguous()`、`soft_empty_cache`、异步传输、子目录模型扫描和 `force_unload` 等优化。
- 更新中英文参数说明。

较小的分块可以降低 Conv3D 峰值显存，但会增加分块次数和运行时间。

## 验证

- `py_compile` 通过。
- 在 RTX 4060 8GB 上使用与失败任务相同的 latent 形状、2 倍尺寸和 BF16，设置 `temporal_chunk_frames=12`：
  - 4 个时间分块完成
  - 输出计算完成，无 OOM
  - PyTorch 峰值分配显存约 `3942.4 MiB`
  - 独立放大探针耗时约 `9.70 秒`

这个参数让 8GB 用户可以按视频长度和目标分辨率主动降低分块大小，同时保持 32 作为原有默认值。



### #45 [open] ISSUE: no issue, just reporting, codex cli made changes to include Mac MPS support.
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/45  by buggz2k 2026-08-31T18:19:55Z

Hello,
I had codex cli change the node, Minimax H3 Latent Upscaler (3D) and 2D, to use MPS on Mac
It is working great here.
Would you like the changes?
Tell me how to get these you you if you would like.
Thanks!

[MinimaxH3-3D] Latent 54x30 -> 86x48 | Pixels 1376x768 | scale=1.596
[MinimaxH3-3D] Loaded upscale model: minimax_h3_latent_upscaler_3d_fp16.safetensors
  Params: 345,280,216 | Attn: forced off | Temporal: on (every=2, kernel=5) | Backend: Apple MPS | Precision: fp16
[MinimaxH3-3D] temporal chunking: T=37 chunks=2 overlap=5




> **Comment by susky900213 (2026-09-20T23:58:01Z):**

Can you give me the node that supports MPS? thanks




### #44 [open] ISSUE: 和UltimateUpscale相比，采样速度有极大差异
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/44  by SKYGUMP 2026-08-31T16:58:08Z

<img width="1437" height="293" alt="Image" src="https://github.com/user-attachments/assets/03327398-3b17-46de-9a73-8ba9e904091b" />

<img width="1437" height="258" alt="Image" src="https://github.com/user-attachments/assets/9169e611-781d-40d6-b67f-c8a2c72e9709" />

希望尽快更新，让速度能拉平。


### #43 [open] ISSUE: Sound issue with last nodes update
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/43  by RuslanKuz 2026-08-31T10:25:10Z

Hi! Thank you for the great upscaler. After update I connected new node MMH3 Split Upscale instead of SamplerCustomAdvanced on 2nd pass. But sound becomes very bad. For example with SamlerCustomAdvanced I didn't have this issue. How can this be fixed?

<img width="1198" height="1263" alt="Image" src="https://github.com/user-attachments/assets/3d87f49d-d555-4c70-a394-f3133aa405c8" />

<!-- Failed to upload "friendly_minimax_00009-audio.mp4" -->

> **Comment by RuslanKuz (2026-08-31T10:40:10Z):**

This one with new node:
https://github.com/user-attachments/assets/61c8b327-a3a0-435f-a7c9-5026d6cab3f1

And this with SamlerCustomAdvanced:
https://github.com/user-attachments/assets/10ff00af-3499-4ebb-b8bb-4431d9a4de1e


> **Comment by huowenjie (2026-09-02T01:15:10Z):**

You can try bypassing the SplitSigmas node for the first sampling region, and directly connect the latent output from the first custom sampler to the final audio VAE decoder node.

<img width="688" height="720" alt="Image" src="https://github.com/user-attachments/assets/3f22dc2c-1d5b-4ee4-b685-8a6081216e66" />



> **Comment by RuslanKuz (2026-09-03T13:59:22Z):**

@huowenjie Thanks! I will try



### #42 [closed] ISSUE: Can't see how to load your upscale safetensor with the new tiled nodes?
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/42  by slikvik55 2026-08-30T09:46:57Z

This is possible with Ultimate Upscale, so I think I may be misunderstanding your version of it. And example might be good. =]

> **Comment by m8rr (2026-08-30T11:53:20Z):**

Just replace the second Sampler Advanced with MMH3 Split Upscale and you're done.
Of course, I mean in the workflow of this repository.

I think there's a bug. It sounds like there's only static/noise coming through.
Perhaps it's masking all the sound and passing it through. Actually, we could use a split sampler instead of a sync node, but passing through the sound like this creates an issue where we can't use it as a first sampler—and of course, it's also a bit disappointing that it doesn't improve the sound even when used as a second sampler.


> **Comment by slikvik55 (2026-08-30T13:27:49Z):**

I don't want to have both the original upscaler nodes AND the fork. I was hoping now that @LBH-123-AI has implemented tiled upscale, that I can get rid of the bbaudio version.


> **Comment by m8rr (2026-08-30T13:46:44Z):**

<img width="1408" height="353" alt="Image" src="https://github.com/user-attachments/assets/74705fa2-c21b-4d86-9f3a-1373af0bb37d" />
So you upscale it with the upscale 3D node and then feed it into the upscale sampler. They're all nodes from this repository, and you just swap out the sampler in the example workflow.

https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/tree/main/workflow_templates



### #41 [open] ISSUE: fl2va (first/last-frame) + latent upscaler → shape mismatch in MiniMax H3 DiT
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/41  by darksidewalker 2026-08-30T06:49:34Z

## fl2va (first/last-frame) + latent upscaler → "shape mismatch" in MiniMax H3 DiT

### Summary

When a MiniMax H3 **fl2va** workflow (first_frame / last_frame on `MiniMax H3 Image to Video`) is chained through one of the latent upscaler nodes in this repo (2D or 3D) and then through a stock `SamplerCustomAdvanced`, sampling crashes inside the H3 DiT with:

```
RuntimeError: shape mismatch: value tensor of shape [864, 96] cannot be broadcast
to indexing result of shape [3456, 96]
```

(reported at `comfy/ldm/minimax/model.py`, the keyframe-cond row assignment
`all_video_rows[~img_update] = cond_video_rows`).

The reported numbers are a clean **2× spatial upscale**: `3456 / 864 = 4` = the
2× area ratio. The video latent was enlarged but the keyframe condition was not.

### Reproduce

1. `MiniMax H3 Image to Video` with a `first_frame` (and optionally `last_frame`),
   e.g. 768×1152. This builds the target latent and encodes the keyframes at that
   resolution (`minimax_keyframes`).
2. Feed the **latent** through `Minimax H3 Latent Upscaler (2D)` or `(3D)` at a
   2.0× factor → target latent becomes 1536×2304.
3. Run the **upscaled latent** through `SamplerCustomAdvanced` (with the same
   fl2va conditioning).

Result: crash above, ~0–2 steps in.

### Root cause (in core ComfyUI)

`PackedLayout` (core `comfy/ldm/minimax/model.py`) sizes the keyframe/"cond"
sequence slots from the **target** grid:

```python
n = vt * frame_rows          # frame_rows = _frame_grid(target_lat_h, target_lat_w)
```

but the actual keyframe rows are produced by patchifying each keyframe latent at
**its own** spatial grid, independently of the target:

```python
for z in payload.get("cond_video_latents", []):
    r = patchify_video(z.to(torch.float32), self.patch_size)   # z's own H,W
```

There is no check that the keyframe latent's H×W equals the target's. When they
differ, the layout reserves N cond slots (for the target grid) but only M rows
arrive (for the keyframe grid), and `all_video_rows[~img_update] = cond_video_rows`
fails to broadcast. (The audio branch has the identical latent bug.)

Reference blocks (`minimax_refs`) are not affected because their layout slots use
the ref's *own* dims; only **keyframes** ride the target grid.

### Why the upscaler node can't avoid it

`MinimaxH3LatentUpscalerNode2D` and `MinimaxH3LatentUpscaler3D` have a **LATENT-in →
LATENT-out** schema only. They enlarge the video latent but have no access to the
`minimax_keyframes` riding in the separate **conditioning** graph, so the keyframes
stay at the pre-upscale resolution. Any fl2va graph wired as
`fl2va → upscaler → stock sampler` is therefore guaranteed to hit this.

(`MMH3SplitUpscale` in this repo does resample keyframes to its own grid — see its
`reanchor_conditioning` — but it runs its own sampling and does not hand a latent to a
stock sampler, so it doesn't cover this path.)

### Suggested fix

Add an optional **CONDITIONING in/out** to the 2D and 3D upscaler nodes that
resamples every keyframe's video latent to the new target grid (bilinear
interpolate each `minimax_keyframes[*]["latent"]` to the new H×W, keeping the
`resolved_frame_index`). The resample logic already exists in this repo in
`MMH3_Split_Upscale.reanchor_conditioning` / `crop_keyframes_to_tile` and can be
reused. When a conditioning input is connected, the node outputs the corrected
conditioning so the downstream sampler sees keyframes that match the upscaled latent.

Complementary (core-side): a small guard in `comfy/ldm/minimax/model.py` that raises a
readable error ("keyframe cond latent has M rows but the target grid reserved N —
re-encode the keyframes at the target resolution") instead of the raw broadcast
crash, for both the video and audio branches.

### Immediate user workarounds (until the fix)

- Drop the first/last frames and run t2va through the upscaler, or
- run `MiniMax H3 Image to Video` directly at the final resolution and skip the upscaler, or
- use the two-node ref2va trick referenced in #31.

### Related

- #17 Shape mismatch ([726,96] vs [1536,96], i2v)
- #27 "Add Guide for MiniMax H3" node support ([1620,96] vs [2247,96])
- #29 Refine second-stage crash, low-res cond + upscaled latent (IndexError variant)
- #31 [2064,96] vs [1672,96] (i2v)
- #38 "Add Latent Cond Sync back?" — user request for the conditioning-resample capability

### Environment (from the report)

- ComfyUI 0.34.2, PyTorch 2.13.0+cu130, Python 3.13, win32, RTX 5080
- Node: `SamplerCustomAdvanced`
- Crash location: `comfy/ldm/minimax/model.py` keyframe-cond row assignment


> **Comment by m8rr (2026-08-30T08:08:45Z):**

Instead of asking AI to write these duplicate issue reports, wouldn't it make more sense to just ask it to create a node that resizes keyframes in conditioning?


> **Comment by darksidewalker (2026-08-30T09:14:33Z):**

It's not just a duplicate, read closer, there are references to the problem. And why not use AI to write a good report with details?
An extra node to fix the current upscale node, doesn't make sense to me.


> **Comment by m8rr (2026-08-30T09:40:50Z):**

https://github.com/user-attachments/assets/d40cffb2-bd40-48d0-90d6-7c54a096d6d8

<img width="1304" height="415" alt="Image" src="https://github.com/user-attachments/assets/87359a3e-4291-4c93-8c74-c77b68fae5d3" />

Anyway, the AI fixed the sync node that wasn't doing anything before, but even though there are no errors, the result is terrible! There must be something wrong, but I don't know and I'm not desperate enough to dig any deeper.  

The AI fixed the incorrect size calculation part in H3_latent_upscaler_3d_v3.
```
        target_width = int(samples.shape[-1])
        target_height = int(samples.shape[-2])
```

And I asked it to fix the issue where H3LatentResize couldn't detect keyframes.
```
def _resize_keyframe_latent(latent, target_height, target_width):
    """
    Args:
        latent:
            4D: (B, C, H, W)
            5D: (B, C, T, H, W)

        target_height:
            target latent height

        target_width:
            target latent width

    Returns:
        resized latent
    """

    if not isinstance(latent, torch.Tensor):
        return latent

    if latent.ndim not in (4, 5):
        return latent

    if (
        int(latent.shape[-2]) == int(target_height)
        and int(latent.shape[-1]) == int(target_width)
    ):
        return latent

    orig_dtype = latent.dtype

    value = latent.to(torch.float32)

    if latent.ndim == 4:
        # (B, C, H, W)
        resized = F.interpolate(
            value,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

    else:
        # (B, C, T, H, W)
        b, c, t, h, w = value.shape

        value = (
            value
            .permute(0, 2, 1, 3, 4)
            .reshape(b * t, c, h, w)
        )

        value = F.interpolate(
            value,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

        resized = (
            value
            .reshape(b, t, c, target_height, target_width)
            .permute(0, 2, 1, 3, 4)
        )

    return resized.to(orig_dtype)


def _resize_conditioning(conditioning, width, height):
    """
    Args:
        conditioning:
            ComfyUI CONDITIONING

        width:
            target latent width

        height:
            target latent height

    Returns:
       CONDITIONING
    """

    if conditioning is None:
        return None

    out = []

    for item in conditioning:

        if not isinstance(item, (list, tuple)) or len(item) < 2:
            out.append(item)
            continue

        cond_tensor = item[0]
        params = item[1]

        if not isinstance(params, dict):
            out.append(item)
            continue

        new_params = dict(params)

        keyframes = new_params.get("minimax_keyframes")

        if not isinstance(keyframes, list):
            out.append([cond_tensor, new_params])
            continue

        new_keyframes = []

        for kf in keyframes:

            if not isinstance(kf, dict):
                new_keyframes.append(kf)
                continue

            new_kf = dict(kf)

            latent = new_kf.get("latent")

            if isinstance(latent, torch.Tensor):
                new_kf["latent"] = _resize_keyframe_latent(
                    latent,
                    target_height=height,
                    target_width=width,
                )

            new_keyframes.append(new_kf)

        new_params["minimax_keyframes"] = new_keyframes

        out.append([cond_tensor, new_params])

    return out

```



> **Comment by m8rr (2026-08-30T10:00:49Z):**

This is just with 384 input. 
https://github.com/user-attachments/assets/59ea3d58-9d08-44c8-9a99-73867f22e077

And this is with 768 input, downscaling the latent and conditioning. Is conditioning resizing really this bad? I'm not sure.
https://github.com/user-attachments/assets/9828dd4b-de76-4664-a340-1d16f2a5ff5c

https://github.com/user-attachments/assets/484e995d-996d-4269-bf04-6e8284b4779a
A 1.6x downscale looks way better than 2x. Anyway, the conditioning resize code itself seems fine, but if the quality is this poor, it's just better to use the Add Guide node twice—that's how LTX does it anyway.




> **Comment by VKatilis (2026-09-13T22:29:52Z):**

Ridiculous. What's the point of uploading a broken node?



### #40 [closed] ISSUE: Question about Sigma Values
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/40  by javawock7618 2026-08-30T02:15:34Z

Hi.

I have a question about the sigma values.

When I try the manual sigma values provided in the workflow sample, I don't get very good results. The output ends up with coarse block noise.

I’m currently using the BasicScheduler, with the Latent Upscaler(3D) node set to 3 steps / denoise 0.40. For the Split Upscale node, I’m using 1 step / denoise 0.20, following the example from ComfyUI-MMH3-UltimateUpscale.

By the way, my workflow is publicly available here:
https://huggingface.co/javawock7618/comfy-MiniMax-H3-workflows



### #39 [open] ISSUE: Please update the new workflow. thank you
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/39  by aigohello1836-lang 2026-08-28T14:26:47Z




### #38 [open] ISSUE: Add "Latent Cond Sync" back?
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/38  by JustTheGuy1998 2026-08-28T13:15:03Z

Please, it's the only option that works and doesn't split "shape mismatch"....

Thanks.


### #37 [closed] ISSUE: possible to add teporal and spatial split? segment and tile to preventing from OOM
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/37  by xyzDist 2026-08-28T12:35:13Z

<img width="768" height="963" alt="Image" src="https://github.com/user-attachments/assets/03785082-2e1b-43da-ba5c-d0fe8a53136e" />

Thanks a lot!!

> **Comment by LBH-123-AI (2026-08-28T13:02:13Z):**

@xyzDist 
Already added. Please update and try it.
<img width="916" height="772" alt="Image" src="https://github.com/user-attachments/assets/cfb36b86-c123-4810-8cd8-43cc9219e007" />


> **Comment by aigohello1836-lang (2026-08-28T14:02:15Z):**




> [@xyzDist](https://github.com/xyzDist) Already added. Please update and try it. <img alt="Image" width="916" height="772" src="https://private-user-images.githubusercontent.com/215098063/642739679-cfb36b86-c123-4810-8cd8-43cc9219e007.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODc5MjU5MzEsIm5iZiI6MTc4NzkyNTYzMSwicGF0aCI6Ii8yMTUwOTgwNjMvNjQyNzM5Njc5LWNmYjM2Yjg2LWMxMjMtNDgxMC04Y2Q4LTQzY2M5MjE5ZTAwNy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwODI4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDgyOFQxNDAwMzFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04MWUyZTY3YTdmNDQyYjg3MzRiNTQwMjBjODIzZGExNWE0ZTRiYTQ0Yzg4NThjYzdmOWQ1ZTAyMDdkM2QyNTk0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.IDo7E8vRMt4g1yzs4OQxeWQ0cNLIOedq1LEmGktLrm4">

Can you provide a  full example workflow ?


> **Comment by xyzDist (2026-08-28T16:22:08Z):**

> [@xyzDist](https://github.com/xyzDist) Already added. Please update and try it. <img alt="Image" width="916" height="772" src="https://private-user-images.githubusercontent.com/215098063/642739679-cfb36b86-c123-4810-8cd8-43cc9219e007.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODc5MzQzODUsIm5iZiI6MTc4NzkzNDA4NSwicGF0aCI6Ii8yMTUwOTgwNjMvNjQyNzM5Njc5LWNmYjM2Yjg2LWMxMjMtNDgxMC04Y2Q4LTQzY2M5MjE5ZTAwNy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwODI4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDgyOFQxNjIxMjVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lMzk1MjA5ZTQwZTI2YThlNjE2NzQxZjJiNTI1NzA3YzQ5YmNiNTU4YjM4OTc1ZTBhMjI0YjViOTk2MWQ3ODU1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.0JhLfWrcnCDkeU9zvvfDcRKctK4dUqGjH1lJdI8nXqE">

WOW... Thank you!!! will report back.



### #36 [open] ISSUE: Apple Silicon / MPS: flat, solid-color, or washed-out upscale output above ~224x480 — PyTorch MPS F.pad kernel bug (fix: ComfyUI PR 15902)
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/36  by jk212h20 2026-08-26T17:12:09Z

Documenting for searchability, since this failure looks like "the upscaler is broken" but isn't an upscaler bug.

**Symptom (Mac / Apple Silicon only):** upscaled (or re-encoded) H3 video comes out a flat solid color (uniform tan/grey), heavily washed out, or NaN-blank — with no error in the console. Below ~224x480 the same workflow is fine. Users of both the 2D and 3D nodes are affected, because the corruption happens inside the H3 VAE encode stage itself.

**Root cause:** PyTorch\'s MPS `F.pad(mode="constant")` applied along the temporal axis of 5D tensors silently writes wrong values into ~80% of the tensor once the padded spatial extent exceeds ~2^16 elements/frame (fp16 additionally produces NaNs). The H3 VAE\'s `CausalConv3d` uses exactly this padding, so latents above ~224x480 encode to plausible-looking garbage. Fully characterized + 5-line repro: pytorch/pytorch#194922.

**Fix:** Comfy-Org/ComfyUI#15902 replaces the temporal pad with a `torch.cat` zero-slab (bit-exact vs CPU on every backend; validated 3x end-to-end upscale, SSIM 0.996–0.998). Once merged, updating ComfyUI resolves it — no change needed in this repo\'s nodes. Until then, upscaling above ~224x480 on MPS will corrupt output.

If you hit this: update ComfyUI once PR 15902 lands (or apply its 4-line change to `comfy/ldm/minimax/vae.py`).


### #35 [open] ISSUE: Minimax H3 upscaler does not find models from subdirectories or extra_model_paths.yaml
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/35  by badgids 2026-08-26T10:46:06Z

## Problem

The Minimax H3 Latent Upscaler nodes do not correctly find models when `latent_upscale_models` uses more than one ComfyUI model path.

This includes external model folders that are set in:

```text
extra_model_paths.yaml
```

My models were already in a valid external `latent_upscale_models` folder. ComfyUI had registered that folder correctly, but the Minimax H3 nodes did not show the models in the `model_name` list.

Instead, the node only showed the default path, for example:

```text
(place models in: /path/to/ComfyUI/models/latent_upscale_models)
```

## Cause

Both the 2D and 3D nodes use code similar to:

```python
folder_paths.get_folder_paths("latent_upscale_models")[0]
```

The `[0]` selects only the first registered path.

The nodes then use `glob()` to search that one directory.

ComfyUI can register several paths for one model type. For example:

```text
ComfyUI/models/latent_upscale_models
/external/models/latent_upscale_models
```

The second path can come from `extra_model_paths.yaml`.

ComfyUI knows about both paths, but the current node searches only the first one.

There is also a second issue. The `glob()` search only checks the top level of the selected directory.

This can be found:

```text
latent_upscale_models/
└── model.safetensors
```

This can be missed:

```text
latent_upscale_models/
└── MinimaxH3/
    └── model.safetensors
```

## Expected behavior

The Minimax H3 nodes should use the same model path system that ComfyUI uses.

Models should be found in:

```text
ComfyUI/models/latent_upscale_models/
```

External paths from:

```text
extra_model_paths.yaml
```

And subdirectories such as:

```text
latent_upscale_models/MinimaxH3/
```

## Fix

I submitted PR #34 with a fix:

https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/pull/34

The fix changes both:

```text
nodes/minimax_h3_latent_upscaler_2d.py
nodes/minimax_h3_latent_upscaler_3d.py
```

Model discovery now uses ComfyUI's existing function:

```python
folder_paths.get_filename_list("latent_upscale_models")
```

This lets ComfyUI search all registered model paths and subdirectories.

Model loading now uses:

```python
folder_paths.get_full_path("latent_upscale_models", model_name)
```

This lets ComfyUI resolve the selected model from any registered `latent_upscale_models` path.

The fix does not change the upscaler model, inference code, workflow inputs, or model format. It only changes how the node finds and resolves model files.

## Result

With this fix, users can keep Minimax H3 upscaler models in external model storage that is already configured through `extra_model_paths.yaml`.

Users do not need to copy the same model files into the main ComfyUI models directory just to make this node find them.

PR with the fix:

https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/pull/34

Cheers!



### #34 [open] PR: Dev: Fix - fix: use ComfyUI model paths for Minimax H3 upscalers added workflow note English translation
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/pull/34  by badgids 2026-08-26T09:08:15Z

Use ComfyUI's model discovery and path resolution functions for the 2D and 3D Minimax H3 latent upscaler nodes.

This fixes model loading from extra_model_paths.yaml and supports models stored in subdirectories.

### Fixed

* Fixed Minimax H3 Latent Upscaler model discovery.
* Added support for all `latent_upscale_models` paths registered by ComfyUI.
* Fixed loading from paths defined in `extra_model_paths.yaml`.
* Added support for models stored in subdirectories.
* Kept support for `ComfyUI/models/latent_upscale_models`.
* Added English workflow note translation in minimax_h3_i2v_Latent Upscaler example workflow2.json

> **Comment by badgids (2026-08-26T09:13:09Z):**

This PR fixes model discovery for the Minimax H3 Latent Upscaler.

The 2D and 3D nodes did not correctly use model paths from ComfyUI's `extra_model_paths.yaml`.
Added English Translation workflow note to `minimax_h3_i2v_Latent Upscaler example workflow2.json`.

## Problem

The nodes used:

```python
folder_paths.get_folder_paths("latent_upscale_models")[0]
```

This selected only the first registered model directory.

The nodes then used `glob()` to search that one directory.

Because of this, models in other registered paths could not appear in the model list. This included valid external paths from `extra_model_paths.yaml`.

The old code also did not search subdirectories.

## Fix

The 2D and 3D nodes now use ComfyUI's normal model path functions.

Model discovery now uses:

```python
folder_paths.get_filename_list("latent_upscale_models")
```

Model loading now uses:

```python
folder_paths.get_full_path("latent_upscale_models", model_name)
```

These functions let ComfyUI search all registered `latent_upscale_models` paths.

## Result

The nodes now:

* Use paths from `extra_model_paths.yaml`.
* Search all registered `latent_upscale_models` directories.
* Find models in subdirectories.
* Continue to support `ComfyUI/models/latent_upscale_models`.
* Use ComfyUI's standard model path system instead of custom path scanning.

No workflow changes are required.

## Files Changed

```text
nodes/minimax_h3_latent_upscaler_2d.py
nodes/minimax_h3_latent_upscaler_3d.py
workflow_templates/minimax_h3_i2v_Latent Upscaler example workflow2.json
```

## Test

Test with a model stored only in an external `latent_upscale_models` path defined in `extra_model_paths.yaml`.

The model must appear in the node's model list and load without copying it into the main ComfyUI models directory.



> **Comment by badgids (2026-09-28T07:10:56Z):**

Have you possibly had time to review this PR?

It solves issue  #35, fixes the incompatibilities in PR #30, and adds an English translation note to the workflow example.


> **Comment by JoeGaffney (2026-10-02T11:44:27Z):**

Why is this not being reviewed and merged its an obvious fix. Ran into this issue when trying the node pack. It's now october and this was submited end of Aug.



### #33 [open] ISSUE: Disabling hardcoded CPU offload for high VRAM setups (RTX PRO 6000)
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/33  by franktytao23 2026-08-25T09:28:21Z

Hi! Thanks for the awesome node!

Quick question about the upscaler stage (MinimaxH3-3D node). I'm running an RTX PRO 6000 (97GB VRAM) with 192GB RAM. I noticed the node hardcodes a CPU offload after generating, which seems to slow down my subsequent runs. Specifically, this part in the code:

```
if dev.type == "cuda":
    model.to("cpu")
    torch.cuda.empty_cache()
```

Do you think it's a safe idea for my setup to just comment these lines out? 
For context, I'm usually making 10-second videos at 0.5mp with a 2x upscale. Thanks for your time!

> **Comment by LBH-123-AI (2026-08-26T07:21:05Z):**

@franktytao23 Hello — thank you for the kind words, and congratulations on that impressive setup! 97GB VRAM is certainly enviable.

Regarding the offload‑to‑CPU functionality: it was primarily designed for users with limited VRAM. If this offloading step does cause any performance overhead on high‑end hardware, I will add a toggle in a future update so that users can enable or disable it as needed.

In the meantime, if you'd like to keep the model on GPU, you can safely comment out those lines in the code. I'll make sure the upcoming toggle makes this more convenient.

Thanks again for your feedback!



### #32 [open] ISSUE: Chunking and tiled upscaling. Feature request.
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/32  by WiseDuckling 2026-08-25T07:59:36Z

I just wanted to bring this to your attention: https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale

This person vibe coded (their words) tiled upscaling and chunking and integrated this into your workflow.

This made it possible for me to do up to 18 seconds at 2560x1920 and I can go higher. I just haven't tested the limits yet. While with your workflow, its an instant OOM if I try 4mpx and 7 seconds. Example: https://storage.to/XvHWGhsCc

It would be awesome to see something like this get worked into your latent upscale workflow and become a permanent addition to the upscaler. This would make the upscaler usable on far more hardware but also raise the ceiling for how high more powerful hardware can go. 

> **Comment by LBH-123-AI (2026-08-26T07:16:39Z):**

@WiseDuckling Hello — thank you for sharing this project with me. I wasn't aware of it before, so I really appreciate the heads-up.

Looking through the repo, I think this is a genuinely impressive effort. The **tiled upscaling and chunking** approach they implemented — which enables handling high resolutions and longer videos — is quite clever. The fact that you're able to push 18 seconds at 2560×1920 without OOM speaks for itself. And effectively reducing seam artifacts at chunk boundaries is no small feat either.

That said, from my understanding of the approach, there are still some areas that could be refined. For instance, in multi‑person dialogue scenes, characters across different chunks may speak simultaneously because they're driven by the same audio reference, and the relatively low denoising strength can also cause a drop in character consistency. This is a difficult problem to solve elegantly.

Honestly, I think this project has a lot of potential, and I fully encourage you and the author to keep exploring and improving it. To be frank, I'm not sure I can optimize it any better myself — they've already done excellent work. If a more robust solution emerges in the future — one that balances high‑resolution scaling with reliable temporal consistency — I would definitely consider integrating it into this node pack.

That said, for now, I need to devote most of my energy to optimizing the 2D convolution model and pushing its performance as far as possible.

In the meantime, my suggestion is to keep experimenting with both workflows and see which one works best for your use cases. If you or the author discover any new breakthroughs, feel free to share them — I'm always open to learning from the community.

Thanks again for sharing this, and good luck with your testing!



### #31 [open] ISSUE: [ERROR] !!! Exception during processing !!! shape mismatch: value tensor of shape [2064, 96] cannot be broadcast to indexing result of shape [1672, 96]
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/31  by drowai 2026-08-25T06:55:32Z

Loaded up your I2V workflow. Immediately errored out.

ComfyUI 0.33.1 with the latest pull being today (08/24/2026).

> **Comment by JustTheGuy1998 (2026-08-25T09:07:02Z):**

Because it's broken at arrival, i2v nodes encode images differently.

Use two ref2vid nodes instead and good luck with prompting to get image as a first frame :)


> **Comment by WiseDuckling (2026-08-25T10:54:16Z):**

> Because it's broken at arrival, i2v nodes encode images differently.
> 
> Use two ref2vid nodes instead and good luck with prompting to get image as a first frame :)

So you just stick an imagine into the ref2v h3 node and then prompt for that to be the starting image? I have the same issue with the model throwing a mismatch error about 6-10 steps into the upscaling with images loaded in I2v mode.


> **Comment by MajorTom100 (2026-08-25T22:31:56Z):**

so how to get this working? Publishing a non-working workflow doesn't make sense.....  As the other workflow with ref2va also doesn't seem to work, hanging on 47% at PASS2 and nothing moves forward.


> **Comment by drowai (2026-08-26T00:58:51Z):**

> so how to get this working? Publishing a non-working workflow doesn't make sense..... As the other workflow with ref2va also doesn't seem to work, hanging on 47% at PASS2 and nothing moves forward.

What @Telllinex said worked. But I don't even think this is worth it. Yeah, it looks nice, but both videos I upscaled had serious issues.

The first one overlaid a logo in the middle of it (lol) for no reason, and the second one added music. And no, it's 100% not a prompt issue, because I saved the intermediate (un-upscaled) versions and they had neither of those problems. This is 100% an upscaler issue. It's half-baked and not ready for primetime.


> **Comment by JustTheGuy1998 (2026-08-26T17:26:28Z):**

> > so how to get this working? Publishing a non-working workflow doesn't make sense..... As the other workflow with ref2va also doesn't seem to work, hanging on 47% at PASS2 and nothing moves forward.
> 
> What [@Telllinex](https://github.com/Telllinex) said worked. But I don't even think this is worth it. Yeah, it looks nice, but both videos I upscaled had serious issues.
> 
> The first one overlaid a logo in the middle of it (lol) for no reason, and the second one added music. And no, it's 100% not a prompt issue, because I saved the intermediate (un-upscaled) versions and they had neither of those problems. This is 100% an upscaler issue. It's half-baked and not ready for primetime.

DId you use 3-4 step sigmas or basic scheduler with custom denoise value?

Higher denoise values tend to do what you described, while sigmas... just work and make audio even better for me.


> **Comment by drowai (2026-08-26T18:44:02Z):**

> DId you use 3-4 step sigmas or basic scheduler with custom denoise value?
> 
> Higher denoise values tend to do what you described, while sigmas... just work and make audio even better for me.

All I did was use the I2V workflow and replace the conditioning nodes with the REF2VA nodes (MiniMax H3 Reference to Video). Literally everything else was the same.

It looks like it goes BasicScheduler (8 steps, 1.0 denoise) -> SplitSigmas (4 steps) -> SamplerCustomAdvanced, then it uses the 3 step Sigmas for second stage sampling.

Happy to hear any recommendations for denoise/steps/sigma values.


> **Comment by JustTheGuy1998 (2026-08-26T18:49:46Z):**

I forgot about 4 steps at 1st sampler because it will never be good... 10+ steps is the way.

Then, I've never heard about the split sigmas node used- the first sampler uses sigmas from the Basic scheduler, and yes, the second sampler uses "3-step sigmas" from his WF, which are 0.8500, 0.6316, 0.3158, 0.0000

So far, I did not have any issues... Of course, if you use 4 steps for the 1st sampler, you'll get audio issues. Use 8 step lightx2v with at least 8 steps. It's not that demaning with sparse attention from Zironic


> **Comment by drowai (2026-08-26T19:36:14Z):**

Swapped to Lightx2v 8-step LoRA and pumped it to 10 steps. Like I said, it _looks_ good, but the audio is still kinda iffy (though I suspect that's entirely the Turbo LoRA). I still get really weird stuff like logos and music though on the second stage sampling, and they are not present on the first stage.

<img width="1055" height="672" alt="Image" src="https://github.com/user-attachments/assets/916d1486-28ff-485e-80af-ed1eacb47edf" />



Edit: Just noticed you gave different sigma values for the 3 step upscaling pass than what came in the default I2V workflow. The ones supplied are 0.9035, 0.6316, 0.3158, 0.0000, so I'll try yours instead.


> **Comment by drowai (2026-08-26T19:57:34Z):**

It seems you were correct. Too much noise from the first sigma value was introducing unnecessary / unwanted deviation from the original generation. Your 0.8500 value helped mitigate that. Thanks!



### #30 [open] PR: Support models in subfolders
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/pull/30  by eduvm 2026-08-24T20:45:35Z

## Summary
- discover `.pth` and `.safetensors` models recursively through ComfyUI's model-folder API
- preserve relative subfolder names in the dropdown
- resolve selected files through `get_full_path_or_raise` for both 2D and 3D nodes

## Tests
- compiled both node modules with `py_compile`
- verified both nodes discover and resolve real models stored under `latent_upscale_models/MiniMaxH3/`


### #29 [open] ISSUE: Refine (second-stage sampling) crashes: IndexError list index out of range in calc_cond_batch (low-res cond + upscaled latent)
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/29  by 0000TVshareU0000 2026-08-24T14:36:08Z

## Bug: Refine (second-stage sampling) crashes with `IndexError: list index out of range` in `calc_cond_batch`

**Environment**
- ComfyUI: v0.33.2 (also reproduced on v0.31.0)
- PyTorch: 2.13.0+cu130, Python 3.13.14
- GPU: RTX 4070 Ti SUPER 16GB
- Model: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` + `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` + dual VAE
- Plugin: latest main (git clone today)

**Workflow structure (t2v, API format)**
```
MiniMaxH3ImageToVideo (608x352) ──cond──> BasicGuider ──> SamplerCustomAdvanced #1 (20 steps, res_multistep) ──> clean latent
  └──latent──> LTXVSeparateAVLatent ──video──> MinimaxH3LatentUpscaler3D (target 1216x704, align 32, fp16) ──> LTXVConcatAVLatent
  ──> SamplerCustomAdvanced #2 (refine, 3 steps ManualSigmas [0.9035, 0.6316, 0.3158, 0.0], euler)
      guider = SAME low-res BasicGuider (conditioning from 608x352 MiniMaxH3ImageToVideo)
      latent_image = upscaled 1216x704 latent
```

**Error**
```
node_id: SamplerCustomAdvanced (refine)
exception_type: IndexError
exception_message: list index out of range
in comfy/k_diffusion/sampling.py, sample_euler -> denoised = model(x, sigma_hat * s_in, ...)
  -> comfy/samplers.py sampling_function -> calc_cond_batch(model, conds, x, ...)
```

- Stage 1 (608x352 + low-res cond) samples fine.
- The 3D upscaler runs fine (latent upscaled to 1216x704).
- Only the second-stage sampling (refine) crashes, in `calc_cond_batch` — same `IndexError` on both ComfyUI v0.31.0 and v0.33.2.
- Reproduced with both `euler` and `res_multistep` samplers, and with 3-step / 4-step / 5-step ManualSigmas.
- Removing the refine sampler and decoding the upscaled latent directly works fine (no crash).

**Hypothesis**
The H3 conditioning (from `MiniMaxH3ImageToVideo`) appears to be shape-bound to the latent size. Feeding an upscaled 1216x704 latent with the original 608x352 conditioning causes an out-of-range index inside `calc_cond_batch`. In your `workflow_templates/minimax_h3_i2v_Latent Upscaler example workflow2.json`, the refine sampler (node 25) also connects the low-res conditioning (node 19, 608x352) to the upscaled latent (node 18) — which seems to be the same pattern.

**Questions**
1. What is the correct conditioning wiring for the refine/second-stage sampling after upscaling? Should the refine use a *high-resolution* conditioning (e.g. from a separate 1216x704 `MiniMaxH3ImageToVideo`), or is there a known workaround?
2. Is the example workflow actually runnable end-to-end as shipped, or does it need any tweaks on 0.31+/0.33+?

Happy to provide the full workflow JSON or run any debug steps you suggest.


> **Comment by m8rr (2026-08-24T23:47:58Z):**

The repository already provides an example using two i2v nodes. However, using the add guide twice is better. If you don't need an exact first frame, you can just use r2v max mode.



### #28 [open] ISSUE: Possible future improvements for even higher latent upscale quality?
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/28  by JohnAlcatraz 2026-08-23T17:43:52Z

This already works very well, and it's also super fast, it just takes a few seconds to run even with the 3D node. And the quality is fine, but also not perfect, the output tends to look a bit "fuzzy", not fully sharp/clear.

So I am curious, do you maybe have plans to make a slower version, train a different latent upscale model, with even higher quality? Maybe something that takes 2-3x as long to run as this current 3D node, but with a significantly higher latent upscale quality? 

A higher latent upscale quality could mean that in the second H3 pass, fewer H3 inference steps are required to refine the output from the latent upscale to a perfect output, so overall even a 2-3x slower latent upscale could save time if it means fewer H3 steps are required afterwards to clean it up.

> **Comment by subspecs (2026-08-23T18:30:49Z):**

> This already works very well, and it's also super fast, it just takes a few seconds to run even with the 3D node. And the quality is fine, but also not perfect, the output tends to look a bit "fuzzy", not fully sharp/clear.
> 
> So I am curious, do you maybe have plans to make a slower version, train a different latent upscale model, with even higher quality? Maybe something that takes 2-3x as long to run as this current 3D node, but with a significantly higher latent upscale quality?
> 
> A higher latent upscale quality could mean that in the second H3 pass, fewer H3 inference steps are required to refine the output from the latent upscale to a perfect output, so overall even a 2-3x slower latent upscale could save time if it means fewer H3 steps are required afterwards to clean it up.

You can swap the sigma nodes and replace it with a '**Basic Sigma**' node.
That one allows you to change the refine step count (set to something like 16 or more per say) and the scheduler to '**normal**' for the **Basic Sigma** node only.
Does exactly what you want.


> **Comment by JohnAlcatraz (2026-08-23T18:42:57Z):**

> You can swap the sigma nodes and replace it with a '**Basic Sigma**' node. That one allows you to change the refine step count (set to something like 16 or more per say) and the scheduler to '**normal**' for the **Basic Sigma** node only. Does exactly what you want.

No, that would be the opposite of what I was talking about. I was talking about improving the quality of the *latent upscale* through architectural changes in the latent upscale model, not improving the quality of the H3 refinement step.

So the goal would be *reducing* the actual H3 refinement steps needed to only 1 or 2, and have more of the upscaling quality directly come from a higher-quality, slower than now, but still faster-than-H3 latent upscale model. Because even a 2-3x heavier latent upscale model would still be much faster to run than a proper H3 refinement step.


> **Comment by subspecs (2026-08-23T18:55:15Z):**

> > You can swap the sigma nodes and replace it with a '**Basic Sigma**' node. That one allows you to change the refine step count (set to something like 16 or more per say) and the scheduler to '**normal**' for the **Basic Sigma** node only. Does exactly what you want.
> 
> No, that would be the opposite of what I was talking about. I was talking about improving the quality of the _latent upscale_ through architectural changes in the latent upscale model, not improving the quality of the H3 refinement step.
> 
> So the goal would be _reducing_ the actual H3 refinement steps needed to only 1 or 2, and have more of the upscaling quality directly come from a higher-quality, slower than now, but still faster-than-H3 latent upscale model. Because even a 2-3x heavier latent upscale model would still be much faster to run than a proper H3 refinement step.

You do know there is a practical limit to this, right?
I am pretty sure that more training for the current refinement model might not return the desired results based on the amount of resources spent training.
From what I've seen it's already a bit better than the LTX upscaler.


> **Comment by JohnAlcatraz (2026-08-23T19:14:20Z):**

> You do know there is a practical limit to this, right? I am pretty sure that more training for the current refinement model might not return the desired results based on the amount of resources spent training. From what I've seen it's already a bit better than the LTX upscaler.

There is some limit, but it could be quite likely that the limit is significantly above the current quality. I think it might be plausible to get latent upscale quality at least to the same quality like pixel-space lanczos quality.


> **Comment by subspecs (2026-08-23T19:15:30Z):**

> > You do know there is a practical limit to this, right? I am pretty sure that more training for the current refinement model might not return the desired results based on the amount of resources spent training. From what I've seen it's already a bit better than the LTX upscaler.
> 
> There is some limit, but it could be quite likely that the limit is significantly above the current quality. I think it might be plausible to get latent upscale quality at least to the same quality like pixel-space lanczos quality.

I hope man, I hope :p



> **Comment by LBH-123-AI (2026-08-24T03:31:13Z):**

Hi @JohnAlcatraz  and  @subspecs  — thanks again for the ongoing discussion. I really appreciate the thoughtful exchange.

I wanted to share where I'm currently focusing my efforts: I'm actively working on a **2D convolution‑based latent upscale model** as an alternative to the current 3D version.

From my early observations, the 2D approach has a few interesting advantages:
- It maintains **sharper, clearer details** with fewer parameters.
- By adding just a **small amount of 3D temporal structure**, it can achieve temporal coherence comparable to the full 3D convolution version, but with much lower computational cost.

If this direction proves successful, I plan to push further and explore **scaling up to a larger parameter version** to see how far we can push quality.

That said, I need to be realistic: preparing a more refined and diverse training dataset takes time, and I want to get it right. This will likely take a while before I have a stable model ready for release. And who knows — by the time this is finished, Flux 3 might already be out and everyone will have moved on to the next big thing! 😄

Still, I'm committed to exploring this path and will keep the community posted on any meaningful progress. Thanks for your patience and support — it means a lot.


> **Comment by Fuhze73 (2026-08-24T08:55:32Z):**


>And who knows — by the time this is finished, Flux 3 might already be out and everyone will have moved on to the next big thing! 😄


There's a good chance that MMH3 will remain the gold standard for local video generation for a while longer, even alongside Flux3 when it's released.

Thanks for your work, it's really a huge help in improving H3.


> **Comment by m8rr (2026-08-24T09:21:28Z):**

Still, it probably won't be the final upscale, right? If we end up using a secondary sampler anyway, having something that looks pristine to our eyes might not make the final result any better. A slightly messy upscaler might actually yield a better end result.

Even though basic bicubic latent upscale looks like a mess to our eyes, when it works well, it can produce very sharp and detailed results. Even when using image upscalers, nearest-exact sometimes yields better results than rtx_vsr.

Anyway, the important thing isn't what looks good to our eyes, but upscaling so that the secondary sampler can produce good results. It's a fundamental point, but... so what should we do? I have no idea.


> **Comment by Illynir (2026-08-24T17:14:07Z):**

The 2D approach, which is "sharper" and "clearer", seems the most appropriate and logical to me, especially with a second sampling pass. The node already supports 4x upscaling; I don't really see the point of going beyond that, given the investment in time and money, other than to satisfy a lucky few with PCs powerful enough to handle it.

Thanks for that in any case.


> **Comment by xyzDist (2026-08-29T14:31:02Z):**

> This already works very well, and it's also super fast, it just takes a few seconds to run even with the 3D node. And the quality is fine, but also not perfect, the output tends to look a bit "fuzzy", not fully sharp/clear.
> 
> So I am curious, do you maybe have plans to make a slower version, train a different latent upscale model, with even higher quality? Maybe something that takes 2-3x as long to run as this current 3D node, but with a significantly higher latent upscale quality?
> 
> A higher latent upscale quality could mean that in the second H3 pass, fewer H3 inference steps are required to refine the output from the latent upscale to a perfect output, so overall even a 2-3x slower latent upscale could save time if it means fewer H3 steps are required afterwards to clean it up.

the node works great, I upscale latent to 2K, does this looks sharp to you? use a basicSheduler / step 3 / denoise 0.1-0.2 for sigmas
<img width="3361" height="1931" alt="Image" src="https://github.com/user-attachments/assets/074695a5-a33e-4ca1-9e79-60af19398514" />


> **Comment by 535gfx (2026-09-09T12:18:59Z):**


Since the thread is about the refine after the latent upscale, here is a same-seed A/B that may be useful. Our lane had been refining with ManualSigmas 0.9035 / 0.6316 / 0.3158 / 0 under the H3 sigma shift of 12. Converted through the shift, that first step is t = 0.44, so the refine throws away 90 percent of the upscaled latent's signal at step one. The graph in #37 starts at 0.6316, which is t = 0.125, and a BasicScheduler at denoise 0.1 to 0.2 lands at 0.57 to 0.75 with the shifted table (or 0.07 to 0.14 with the linear one). Nobody in this thread refines above 0.75.

We had blamed the refine for a doubled lower lip on a fast smile inside a single 107-frame chunk and for a general flicker on flat surfaces. Both turned out to be the first sigma. Kitchen shot, 1344x768 to 2688x1536, one 107-frame full-frame refine per ladder, same seed:

| ladder | wall (4090) | mouth sharpness vs native, worst moving frame / median | static detail (Laplacian var, native 2x = 18) | flat-surface flicker (native 0.061) |
|---|---|---|---|---|
| 0.9035 / 0.6316 / 0.3158 / 0 | 7.5 min | 0.74 / 0.88 | 37 | 0.098 |
| 0.6316 / 0.2 / 0 (the #37 graph) | 6.5 min | 0.93 / 1.09 | 35 | 0.056 |
| 0.3158 / 0.158 / 0.05 / 0 | 8.0 min | 0.98 / 1.27 | 39 | 0.060 |

The 0.6316 ladder removes the lip double, brings the flicker to the native's level, keeps 95 percent of the detail and is a step faster. The 0.3158 ladder measures even sharper, but by eye it lets the upscaler's own defects through (a dark streak on a coat collar, a band on the forehead) that both other ladders re-draw, which matches the "fuzzy" you describe: at that first sigma the refine keeps 68 percent of the upscaler latent, defects included. So for faces the useful range on our material is a first sigma around 0.63, not 0.9 and not 0.3. Clip attached: native 2x, the 0.9035 ladder with the doubled lip, and a 2x2 tile refine for comparison.

https://github.com/user-attachments/assets/689fe86d-a0a1-4bac-a15e-032314cb9a46

Edit, one more thing that may be worth checking on your side: the 0.9035 ladder also changes the acting. The brow lift and the forehead lines of the source get softened toward a neutral face, while 0.6316 keeps most of them (mean luma distance to the source in the brow box: 5.8 against 4.5; sheet below, the three frames where the brow moves most). If the 3-step ladder is still the one the README suggests, a face with real expression would be a good test for it. We are going to keep A/B-ing the ladders on more takes and will post what changes.

<img width="1650" height="882" alt="Image" src="https://github.com/user-attachments/assets/3b805202-4005-4cb3-bd23-df4bbb475acd" />




### #27 [open] ISSUE: ‘’Add Guide for MiniMax H3‘’ node support  [shape mismatch error]
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/27  by zwukong 2026-08-23T09:40:13Z

when using 'Add Guide for MiniMax H3', it will cause error in second sampler

 RuntimeError: shape mismatch: value tensor of shape [1620, 96] cannot be broadcast to indexing result of shape [2247, 96]

> **Comment by JustTheGuy1998 (2026-08-23T09:45:25Z):**

Use H3 Latent Cond Sync and ref2vid node to pass a higher-res image to conditioning, then wire to basic guider and to second sampler - works flawlessly. Basically like in an attached i2v workflow, but it doesn't work with i2v minimax H3 nodes...

<img width="1415" height="395" alt="Image" src="https://github.com/user-attachments/assets/df0fc811-2d7a-4af8-969d-3f9eb1d3a1cf" />


> **Comment by zwukong (2026-08-23T09:57:19Z):**

where is the ‘’ Latent Cond Sync‘’ node , i can't find it in this proj


> **Comment by LBH-123-AI (2026-08-23T10:28:32Z):**

@zwukong Sorry for the confusion — the `Latent Cond Sync` node was removed in a recent update. During testing, I found that it didn't provide any meaningful improvement to the output, so I decided to drop it to keep the node pack clean and focused.


> **Comment by zwukong (2026-08-23T10:56:12Z):**

Thanks for your project , can you fix this problem.It is ok without  ‘’Add Guide for MiniMax H3‘’ node @LBH-123-AI . I think something must be done to deal with the positive 


> **Comment by JustTheGuy1998 (2026-08-23T10:59:28Z):**

> [@zwukong](https://github.com/zwukong) Sorry for the confusion — the `Latent Cond Sync` node was removed in a recent update. During testing, I found that it didn't provide any meaningful improvement to the output, so I decided to drop it to keep the node pack clean and focused.

Really? It does lol. You pass the higher-res image to an additional ref2vid node, then pass the conditioning from it, and it really contains data about that image, improving the quality of the upscale drastically! There are suddenly small details like text or other things that are now rendered from the image, not guessed by the upscaler and turned into a mush...

You should really get that node back....

Without passing a high-res image, this upscaler basically turns into an advanced refinement tool, which does not add natural sharpness and higher-res details (though I am talking about I2V and ref2v); T2V is probably completely dependent on the upscaler itself.


> **Comment by m8rr (2026-08-23T11:54:05Z):**

@Telllinex I think if you're using the second r2v node to encode and send the image, it'll work even without the sync node.
Even though I'm not entirely sure, since the reference mode saves images regardless of the latent size, it should be fine to input high-resolution images right from the beginning.


> **Comment by JustTheGuy1998 (2026-08-23T11:56:23Z):**

> [@Telllinex](https://github.com/Telllinex) I think if you're using the second r2v node to encode and send the image, it'll work even without the sync node. Even though I'm not entirely sure, since the reference mode saves images regardless of the latent size, it should be fine to input high-resolution images right from the beginning.

Nice idea, didn't think of it, but I will try :) Thanks!


> **Comment by m8rr (2026-08-23T11:57:40Z):**

If sync nodes operate in the future, the sequence should probably look something like this.

<img width="1543" height="445" alt="Image" src="https://github.com/user-attachments/assets/ac96b38b-e778-4d1b-abbd-fe2910dffa59" />


> **Comment by JustTheGuy1998 (2026-08-23T12:00:47Z):**

Is "Add guide" used to fix a shape mismatch with img2vid node? And why would you upscale the latent beforehand if the latent upscaler does it later?

> If sync nodes operate in the future, the sequence should probably look something like this.
> 
> <img alt="Image" width="1543" height="445" src="https://private-user-images.githubusercontent.com/206154427/639989435-ac96b38b-e778-4d1b-abbd-fe2910dffa59.jpeg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODc0ODY1NjIsIm5iZiI6MTc4NzQ4NjI2MiwicGF0aCI6Ii8yMDYxNTQ0MjcvNjM5OTg5NDM1LWFjOTZiMzhiLWU3NzgtNGQxYi1hYmJkLWZlMjkxMGRmZmE1OS5qcGVnP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDgyMyUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjA4MjNUMTE1NzQyWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YTY4MDlkYTQ5YmU2MmQ0OGI4ZDlkNTVkZTMxOGQ0ZDJhNzI1YWI2MGIzNjY4OWUyZDFlZWE4ZjkyYTBjNGI3MyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmcmVzcG9uc2UtY29udGVudC10eXBlPWltYWdlJTJGanBlZyJ9.iQzTmlgRcy0KsggY4MF-OnCJZj0bY4Ao6Oc51sx-nxI">




> **Comment by m8rr (2026-08-23T12:04:27Z):**

The Add guide doesn't matter; it just shows the sequence. The important thing is to start with high resolution to preserve the original details. Do you see the difference between starting in high resolution, scaling down, and then returning to high resolution, versus starting in low resolution and upscaling a low-resolution reference image? Upscaling a low-resolution reference would result in a loss of image quality, wouldn't it?

This is about when sync nodes are added later; it's completely irrelevant right now, and I think r2v is more than enough.
What I mean is, your cond sync probably isn't going to do anything, and even if it does work, it's just upscaling a low-resolution image at that position, so it doesn't make much sense.   



> **Comment by JustTheGuy1998 (2026-08-23T12:14:32Z):**

> The Add guide doesn't matter; it just shows the sequence. The important thing is to start with high resolution to preserve the original details. Do you see the difference between starting in high resolution, scaling down, and then returning to high resolution, versus starting in low resolution and upscaling a low-resolution reference image? Upscaling a low-resolution reference would result in a loss of image quality, wouldn't it?
> 
> This is about when sync nodes are added later; it's completely irrelevant right now, and I think r2v is more than enough.

Yes, the plan is to have one image, but two resize nodes (scale to total image pixels), one will downscale the image to get w*h for the first img2vid node (0.5mp) for example, but the image goes from the second resize node, that I will resize to 2MP and pass to the first node - In theory, it will do 0.5mp first pass, then will take conditioning again and because the second pass resolution is higher, will use it the same way I have it set up now.




> **Comment by m8rr (2026-08-23T12:30:08Z):**

@zwukong We can use it like this for now, it's the LTX style. Still, it seems better than using two I2V nodes, right?
<img width="1314" height="495" alt="Image" src="https://github.com/user-attachments/assets/336c2113-38f0-4e65-8aad-5aea4513115e" />


> **Comment by zwukong (2026-08-23T12:47:25Z):**

interesting, but not a good solution. we have to wait for a common one


> **Comment by JustTheGuy1998 (2026-08-23T12:51:48Z):**

> interesting, but not a good solution. we have to wait for a common one

I tried the method discussed above and got "shape mismatch" haha :)

So yep, you can roll back when cond sync existed and use it in the meantime.


> **Comment by m8rr (2026-08-23T12:57:30Z):**

I've never seen a shape mismatch error when using the r2v node. It works fine with both match and max. Even with I2V or add guides, if you use them twice according to the resolution changes, they work just fine. Does it depend on the resolution? That's interesting.



### #26 [closed] ISSUE: Significant color/brightness flickering towards end of video
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/26  by JohnAlcatraz 2026-08-22T23:09:33Z

This is very good, except for one issue: I seem to always get significant brightness/color flickering towards the end of the video.

For example with this input video:

https://github.com/user-attachments/assets/d19861bd-5474-4093-98b5-183198544243

put through the Minimax Latent Upscaler 3D node, I get this output:

https://github.com/user-attachments/assets/7e3f0959-756c-4f4d-80de-ca6c75fff744

You see towards the end of the video, there is a very visible flicker where the video becomes darker for a few frames.

This is done using the default ComfyUI Minimax H3 Text-to-Video workflow, just adding these two nodes in before the VAE Decode:

<img width="708" height="284" alt="Image" src="https://github.com/user-attachments/assets/170eedc9-b597-4fbb-b0fd-d650c82fb670" />

I have tested BF16 vs FP16 and the issue is identical with both.

> **Comment by LBH-123-AI (2026-08-23T07:45:22Z):**

@JohnAlcatraz Thank you for your feedback — and for providing the before/after videos, that really helps illustrate the issue.

I've identified the cause: in previous versions, the temporal chunking logic wasn't handled properly. Specifically, the way frames were split and processed across time boundaries could cause inconsistent feature propagation near the end of the video, resulting in the brightness/color flickering you observed.

I'm happy to say that I've already addressed this in the latest update. The new version adds a **toggle to enable/disable temporal chunking**, and I've also completely reworked the chunking calculation to ensure smooth transitions across frame boundaries. The default behavior should now handle this much better.

I'd recommend updating to the latest commit and giving it another try with your test clip. The flickering should be resolved.

Let me know if the update works for you — and if you still see any issues, feel free to share another log or video. Thanks again for the detailed report!



> **Comment by JohnAlcatraz (2026-08-23T17:39:07Z):**

I can confirm the new version no longer seems to have this issue. Thanks for fixing it!



### #25 [open] ISSUE: Broken generation quality?
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/25  by subspecs 2026-08-21T20:02:34Z

<img width="1881" height="1069" alt="Image" src="https://github.com/user-attachments/assets/54330b71-5d2d-4911-8d85-05eaa0d85dac" />
So I tried to upscale from a 360p generation 2x to a 720p, as you can see.

This is the second attempt, but the first attempt can be seen on the bottom right.

This is with the current latest version just now tested.

@LBH-123-AI 

> **Comment by LBH-123-AI (2026-08-23T07:50:25Z):**

@subspecs Thank you for testing the latest version and for sharing your results.

I'm not entirely sure what issue you encountered during the upscale — could you please provide a bit more detail about what went wrong?

That said, I do want to note that the previous version had a default temporal chunking logic that could cause flickering between frames. I've already optimized this in the new version, so it should be much better now.

Looking forward to your clarification — feel free to attach screenshots or logs if that helps.


> **Comment by subspecs (2026-08-23T17:22:39Z):**

"could you please provide a bit more detail about what went wrong?"

Really not much more info to give apart from what you already see in the picture. It generates a video using MiniMaxH3 and then I try to upscale it using your 2D node. As you can see it shits itself during the upscale.
But using the 3D node words like intended tho.

I can only really note that I am using a GGUF version of H3 model at Q6_K scale, if that changes anything on your end.


> **Comment by m8rr (2026-08-26T11:03:29Z):**

I tried giving the 2D node a shot too and got equally messy results. Is there any special advantage to 2D anyway? I've only ever used 3D from the start, so I have no idea.



### #24 [open] ISSUE: 增加了3d版的amd显卡的rocm驱动
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/24  by youkai1981 2026-08-21T10:59:06Z

已经提交pull requests #23 

> **Comment by LBH-123-AI (2026-08-21T15:18:55Z):**

老哥 这个好像是基于前一个版本提交的，新版本修复了显存占用的问题



### #23 [open] PR: add rocm
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/pull/23  by youkai1981 2026-08-21T10:55:01Z




### #22 [open] ISSUE: Exception during processing !!! 'NestedTensor' object has no attribute 'dim'
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/22  by Sooroojdeen 2026-08-21T09:40:07Z

[ERROR] !!! Exception during processing !!! 'NestedTensor' object has no attribute 'dim'
[ERROR] Traceback (most recent call last):
  File "G:\InstallationsComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\InstallationsComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\InstallationsComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "G:\InstallationsComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "G:\InstallationsComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "G:\InstallationsComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "G:\InstallationsComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "G:\InstallationsComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\Comfyui_Minimax_h3_latent_Upscaler\nodes\minimax_h3_latent_upscaler_3d.py", line 452, in execute
    was_4d = (src.dim() == 4)
              ^^^^^^^
AttributeError: 'NestedTensor' object has no attribute 'dim'. Did you mean: 'ndim'?

> **Comment by VladiCz (2026-08-21T13:39:33Z):**

You must split A/V latents and upscale only the video part. AttributeError: 'NestedTensor' object has no attribute 'dim'. Did you mean: 'ndim' shows when you forget to split the A/V parts.


> **Comment by MyBrainHurtsAlways (2026-08-25T17:33:34Z):**

what node do you use to split the av latents?


> **Comment by VladiCz (2026-08-26T04:02:21Z):**

> what node do you use to split the av latents?l

You can use native LTXVSeparateAVLatent, it works for H3 too.



### #21 [closed] ISSUE: [ERROR] !!! Exception during processing !!! Error(s) in loading state_dict for LatentResizer3D:
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/21  by mdkberry 2026-08-21T06:11:54Z

Getting this error on the node

> [MinimaxH3-3D] Latent 60x24 -> 110x44 | Pixels 1760x704 | scale=1.848
> [ERROR] !!! Exception during processing !!! Error(s) in loading state_dict for LatentResizer3D:
>         Missing key(s) in state_dict: "conv_in.weight", "conv_in.bias", "embed.0.weight", "embed.0.bias", "embed.2.weight", "embed.2.bias", "in_blocks.0.in_layers.0.weight", "in_blocks.0.in_layers.0.bias",
> 
> [cut huge chung of similar info out ]
> 
> .norm2.bias", "res_blocks.2.norm2.weight", "res_blocks.3.conv1.bias", "res_blocks.3.conv1.weight", "res_blocks.3.conv2.bias", "res_blocks.3.conv2.weight", "res_blocks.3.norm1.bias", "res_blocks.3.norm1.weight", "res_blocks.3.norm2.bias", "res_blocks.3.norm2.weight", "upsampler.blur_down.kernel", "upsampler.conv.bias", "upsampler.conv.weight".
> [ERROR] Traceback (most recent call last):
>   File "S:\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute
>     output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
>                                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>   File "S:\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data
>     return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
>                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>   File "S:\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
>     await process_inputs(input_dict, i)
>   File "S:\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
>     result = f(**inputs)
>   File "S:\ComfyUI_windows_portable\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
>     return method(locked_class, **inputs)
>   File "S:\ComfyUI_windows_portable\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
>     to_return = cls.execute(*args, **kwargs)
>   File "S:\ComfyUI_windows_portable\ComfyUI\custom_nodes\Comfyui_Minimax_h3_latent_Upscaler\nodes\minimax_h3_latent_upscaler_3d.py", line 481, in execute
>     model = load_model(model_name, dev, precision)
>   File "S:\ComfyUI_windows_portable\ComfyUI\custom_nodes\Comfyui_Minimax_h3_latent_Upscaler\nodes\minimax_h3_latent_upscaler_3d.py", line 335, in load_model
>     model.load_state_dict(up_sd, strict=True)
>     ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
>   File "S:\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 2639, in load_state_dict
>     raise RuntimeError(
>     ...<3 lines>...
>     )
> RuntimeError: Error(s) in loading state_dict for LatentResizer3D:

> **Comment by mdkberry (2026-08-21T06:29:30Z):**

the reason was latent_upscale_models have to be in the top folder e.g. `\ComfyUI_windows_portable\ComfyUI\models\latent_upscale_models`
I had them in `\ComfyUI_windows_portable\ComfyUI\models\latent_upscale_models\LBH\` so it was picking my LTX upscalers and throwing the above error. Once I moved `minimax_h3_latent_upscaler_3d_bf16.safetensors` to the `latent_upscale_models` folder itself I  could see it and select it. 





### #20 [closed] ISSUE: After update started seeing weird light artifacts at bottom of videos and extended gen times
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/20  by SleepyBandit 2026-08-20T20:54:07Z

<img width="1916" height="38" alt="Image" src="https://github.com/user-attachments/assets/ba02106c-b43b-4f32-b644-b4ea198ab320" />

I ran an update this morning and replaced the old Minimax H3 Latent Upscaler (3D) node + jingchen573's custom node to align sizing with the new Minimax H3 Latent Upscaler (3D) node that does both. Certain models are now creating this weird light creep along the bottom of the video. I also feel like final video quality has dropped and gen time has increased. Only thing changed was updating the custom node.

I am switching back to the older commit to conduct some additional A/B testing.

> **Comment by SleepyBandit (2026-08-20T21:01:54Z):**

Yep. Ran a test with the only change being this node and there is a 100% a difference.

New 2-in-1 latent upscaler node - 0.8MP to 1.5x Upscale = 555.10 seconds - light banding on bottom and lower overall quality.
<img width="1919" height="829" alt="Image" src="https://github.com/user-attachments/assets/cbb92b9d-8a90-4c93-8be3-7bfd4fd8532f" />

Old latent upscaler node paired with jingchen573's sizing node - 0.8MP to 1.5x Upscale = 497.56 seconds - No light banding, better overall quality.
<img width="1919" height="830" alt="Image" src="https://github.com/user-attachments/assets/d4cb2ef9-6cc3-4901-8f8e-92aa7fa5ed9d" />

I ran a 4 second clip at 1MP to 1.5x Upscale this morning and it took 40 minutes to complete! I had been running the same clips yesterday and it was completing in under 15 minutes. I am attempting a new test now and will update with gen time.

For now, I don't recommend updating. The older version was better.


> **Comment by cvar66 (2026-08-21T01:33:18Z):**

The light strip at the bottom is actually the top of the video. Both length and width of the video need to be divisible by 32. This plugin only aligns the length to 32. I bet if you check your video the height is only divisible by 16. You need to nudge the upscale so the height will will also divide by 32. In my instance i changed from upscale x 1.35 to 1.37 and it fixed this problem.


> **Comment by SleepyBandit (2026-08-21T01:53:44Z):**

Thanks for the info. Jingchen573's custom node just worked, I don't really want to have to constantly be thinking about divisibility by 32. Also what about the huge increase in gen time and worse quality? Would that be impacted by the same thing?


> **Comment by cvar66 (2026-08-21T02:33:55Z):**

I'm not sure I just started using this yesterday. How far back is the old example? Like a few days ago?


> **Comment by SleepyBandit (2026-08-21T03:19:28Z):**

I produced them both today. One with the latest update. The old (better) one was commit 8b5058a from August 18th which I reverted back to.


> **Comment by LBH-123-AI (2026-08-21T08:31:25Z):**

@SleepyBandit Thanks for the detailed report — I've confirmed all three issues and have since optimized the code accordingly.

**Causes:**
1. **Banding** – only width aligned to 32; height from float division can be odd, causing asymmetric padding.
2. **Slower gen** – `MODEL_CACHE` not freed + odd sizes break cuDNN Tensor Core optimization.
3. **Quality drop** – in‑place ops under FP16 cause truncation errors in latent space.

**Fixes applied:** both dimensions now align to 32; in‑place ops replaced with safe tensor operations; aggressive VRAM cleanup added.

I'll push the update soon. Manually setting dimensions to multiples of 32 should help in the meantime. Thanks again!



### #19 [open] ISSUE: I2V workflow Mismatch issue https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/19  by LokeshNakka 2026-08-20T19:45:49Z

https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler in this Repo They didnot used second MiniMax H3 Image to Video node they used this node MiniMax H3 Latent Upscale Combined for conditioning adjustment


> **Comment by LBH-123-AI (2026-08-26T07:29:03Z):**

@LokeshNakka Thank you for sharing this — it's great to learn about such a well‑made project.

I'm afraid I don't currently have the bandwidth to test or optimize new nodes, as I'm dedicating most of my time and energy to refining another **2D convolution‑based latent upscale model**. That project is my main focus right now, and I want to see it through properly.

That said, I've taken note of this implementation and will definitely revisit it in the future when I have more capacity — though it may take some time!

Best regards



### #18 [open] ISSUE: 单采直出正常，二采放大到同分辨率爆显存，问AI帮我修复了。
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/18  by zhuhanbin-cn 2026-08-20T09:27:05Z

你好，我的设备4070ts16G+96G内存单采0.8分辨率，15秒竖屏视频可以控制在600秒以内，但是二采（1采0.4，2采0.8）却爆显存了。
然后我问了AI怎么修复，最后改好了，时间快了几十秒吧，总结以下：
改动文件:1 个

nodes/minimax_h3_latent_upscaler_3d.py(3D 节点)。2D 节点零改动,其余文件未触碰。

具体改动:forward 加时间分块

在 LatentResizer3D.forward(minimax_h3_latent_upscaler_3d.py:208)前段,插入了一段分块逻辑:

当 latent 时间维 T > 16 时:
  沿时间维每 16 帧切一段,每段前后各扩 2 帧 overlap
  → 每段独立跑完整网络(_forward_seg)
  → 裁掉 overlap 区域,按原位置拼回
当 T ≤ 16 时:直接整段跑,行为与原来完全一致

原网络主体被抽成了 _forward_seg 方法,内部逻辑逐行未动。

为什么要改

16GB 显卡上,15 秒视频的 latent 时间维(几十到上百帧)全量做 512 通道 3D 卷积,中间激活峰值超过显存,conv3d 处 OOM。分块后单次前向只处理最多 20 帧,激活峰值降到约 1/6。

为什么结果是等价的
GroupNorm 逐位置归一化(不跨时间)
TemporalConv 只看前后 2 帧局部窗口
每段带的 2 帧 overlap 保证了段内每一帧的时间上下文与整段前向完全一致
附带的小处理

3D 的 LatentResizer3D 构造时没保存 temporal_kernel 属性,所以分块实现改为从 in_blocks 里的 TemporalConv 权重视图直接推断卷积核大小(取 kernel//2 作为 overlap),不依赖原属性。
附件是工作流和AI修复的代码，我也不是很懂先传上来。
[minimax_h3_latent_upscaler_3d.py](https://github.com/user-attachments/files/31257887/minimax_h3_latent_upscaler_3d.py)

[Unsaved Workflow (1).json](https://github.com/user-attachments/files/31257162/Unsaved.Workflow.1.json)

> **Comment by LBH-123-AI (2026-08-20T13:22:49Z):**

@zhuhanbin-cn 好嘞 感谢，我回头测试一下没问题整合进去



### #17 [open] ISSUE: Shape mismatch
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/17  by JustTheGuy1998 2026-08-20T09:24:33Z

RuntimeError: shape mismatch: value tensor of shape [726, 96] cannot be broadcast to indexing result of shape [1536, 96]

This happens in the attached workflow for i2v, but adjusted with  "scale image to total pixels" and "get-image size," which is then passed to either h3 latent upscale (target resolution) or (megapixels).

It was working fine with the ref2v node, but here we get to the starting point of this node pack, where things break as soon as you don't follow 16x9 AR with the i2v minimax node.

> **Comment by LBH-123-AI (2026-08-20T13:26:24Z):**

Could you please share a screenshot of your workflow setup? That would help me better understand your configuration and pinpoint the issue more quickly.

Thank you!



> **Comment by JustTheGuy1998 (2026-08-20T13:55:13Z):**

<img width="2287" height="1200" alt="Image" src="https://github.com/user-attachments/assets/cc6e1dcc-0dd4-45a5-8319-ef87c1f3c7b2" />

<img width="1661" height="1170" alt="Image" src="https://github.com/user-attachments/assets/0dcd09d3-198c-4535-bd66-b0dff0cbbdc8" />

Thanks! 

I also tried passing width x height to an H3 Latent Upscale (Target Resolution) from that "Scale image to total pixels," hoping that it would help count the resolution correctly, but it seems like the error still traces to minimax\model. py.

<details>

`[ERROR] !!! Exception during processing !!! shape mismatch: value tensor of shape [1247, 96] cannot be broadcast to indexing result of shape [1536, 96] [ERROR] Traceback (most recent call last): File "X:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list await process_inputs(input_dict, i) File "X:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs result = f(**inputs) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func return method(locked_class, **inputs) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED to_return = cls.execute(*args, **kwargs) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1335, in sample output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute return self.original(*args, **kwargs) ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1262, in outer_sample output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1237, in inner_sample samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute return self.original(*args, **kwargs) ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1005, in sample samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options) File "X:\AI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context return func(*args, **kwargs) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\k_diffusion\sampling.py", line 205, in sample_euler denoised = model(x, sigma_hat * s_in, **extra_args) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 640, in __call__ out = self.inner_model(x, sigma, model_options=model_options, seed=seed) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1208, in __call__ return self.outer_predict_noise(*args, **kwargs) ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise ).execute(x, timestep, model_options, seed) ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute return self.original(*args, **kwargs) ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1218, in predict_noise return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 620, in sampling_function out = calc_cond_batch(model, conds, x, timestep, model_options) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer return executor.execute(model, conds, x_in, timestep, model_options) ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute return self.original(*args, **kwargs) ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks) ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 204, in apply_model return comfy.patcher_extension.WrapperExecutor.new_class_executor( ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ...<2 lines>... comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs) ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute return self.original(*args, **kwargs) ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 248, in _apply_model model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds) File "X:\AI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl return self._call_impl(*args, **kwargs) ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl return forward_call(*args, **kwargs) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 540, in forward out = comfy.patcher_extension.WrapperExecutor.new_class_executor( ...<2 lines>... comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options) ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs) File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute return self.original(*args, **kwargs) ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^ File "X:\AI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 654, in _forward all_video_rows[~img_update] = cond_video_rows ~~~~~~~~~~~~~~^^^^^^^^^^^^^ RuntimeError: shape mismatch: value tensor of shape [1247, 96] cannot be broadcast to indexing result of shape [1536, 96] [INFO] Prompt executed in 332.60 seconds`

<details>


> **Comment by Sooroojdeen (2026-08-26T05:27:09Z):**

I also have this issue. It randomly stopped working.


> **Comment by peterducan-hub (2026-09-18T14:57:26Z):**

@LBH-123-AI in rev2v is work perfect but when i use base model a fl2va i got always this error!! im not able to do upscale with fl2va model! any idea why?



### #16 [closed] ISSUE: RuntimeError: shape mismatch when running with Motion Context latent loader/continuation
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/16  by Covalent0714 2026-08-20T04:38:01Z

```
[MinimaxH3-3D] Latent 46x36 -> 72x56 | Pixels 1152x896 | scale=1.573
[MinimaxH3-3D] Loaded upscale model: minimax_h3_latent_upscaler_3d_bf16.safetensors
  Params: 345,280,216 | Attn: forced off | Temporal: on (every=2, kernel=5) | Precision: bf16
[INFO] Requested to load MiniMaxH3
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached. Force pre-loaded 210 weights: 1175 KB.
  0%|                                                                | 0/3 [00:00<?, ?it/s,   Model Initializing ...  ]
[ERROR] !!! Exception during processing !!! shape mismatch: value tensor of shape [5541, 96] cannot be broadcast to indexing result of shape [9699, 96]
[ERROR] Traceback (most recent call last):
  File "G:\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
  File "G:\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
  File "G:\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "G:\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "G:\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "G:\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "G:\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1063, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "G:\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "G:\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
  File "G:\ComfyUI\custom_nodes\ComfyUI-KJNodes\nodes\preview_override_node.py", line 685, in __call__
    return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "G:\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
  File "G:\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
  File "G:\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "G:\ComfyUI\comfy\samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "G:\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
  File "G:\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "G:\ComfyUI\venv\lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "G:\ComfyUI\comfy\k_diffusion\sampling.py", line 205, in sample_euler
    denoised = model(x, sigma_hat * s_in, **extra_args)
  File "G:\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "G:\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
  File "G:\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
  File "G:\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
  File "G:\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "G:\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "G:\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "G:\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
  File "G:\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
  File "G:\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
  File "G:\ComfyUI\comfy\model_base.py", line 203, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
  File "G:\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
  File "G:\ComfyUI\comfy\model_base.py", line 247, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "G:\ComfyUI\venv\lib\site-packages\torch\nn\modules\module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "G:\ComfyUI\venv\lib\site-packages\torch\nn\modules\module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "G:\ComfyUI\comfy\ldm\minimax\model.py", line 502, in forward
    out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
  File "G:\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
  File "G:\ComfyUI\custom_nodes\comfyui-minimax-h3-turbo\__init__.py", line 405, in wrap
    return executor(*args, **kwargs)
  File "G:\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
  File "G:\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
  File "G:\ComfyUI\custom_nodes\comfyui-minimax-h3-turbo\__init__.py", line 461, in wrap
    return executor(*args, **kwargs)
  File "G:\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
  File "G:\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
  File "G:\ComfyUI\comfy\ldm\minimax\model.py", line 581, in _forward
    all_video_rows[~img_update] = cond_video_rows
RuntimeError: shape mismatch: value tensor of shape [5541, 96] cannot be broadcast to indexing result of shape [9699, 96]
```

<img width="343" height="679" alt="Image" src="https://github.com/user-attachments/assets/65b9f0e7-3b37-4d48-913e-1310f61717c4" />

<img width="1308" height="865" alt="Image" src="https://github.com/user-attachments/assets/c0a537c2-e67f-4c44-a55f-d5ea90a47f21" />

Error occurs after loading previous latent for continuation. When passed to upscale after low resolution generation, it errors out at the second sampler node. Motion context latent loading node and subgraph with upscaling pass shown for context.

Does this mean that these two nodes are incompatible? Clip chaining is really powerful and it'd be a shame not to be able to do the upscale in the same pass.

> **Comment by VladiCz (2026-08-20T15:44:48Z):**

Same issue here after latest update with both Minimax H3 Latent Upscaler (2D) and Minimax H3 Latent Upscaler (3D).
It was wokrking fine before I updated today. I was updating multiple custom nodes.

[ERROR] !!! Exception during processing !!! shape mismatch: value tensor of shape [1678, 96] cannot be broadcast to indexing result of shape [2083, 96]

<img width="255" height="236" alt="Image" src="https://github.com/user-attachments/assets/011d070c-bab2-4d44-a380-af21443b2599" />

Edit: **the issue is not with this node pack, as the same issue is present with native Upscale Latent By node.**

<img width="222" height="128" alt="Image" src="https://github.com/user-attachments/assets/e6511b6a-95a5-4b4d-9077-e62ac523a33f" />


> **Comment by Covalent0714 (2026-08-20T17:09:28Z):**

Cross referenced this with the motion context node repo, seeing as it affects the native upscale latent node, too.


> **Comment by VladiCz (2026-08-20T17:13:23Z):**

> Cross referenced this with the motion context node repo, seeing as it affects the native upscale latent node, too.

I was testing today hybrid model with First Frame + Reference Images and the issue is only when First Frame is present causing latent inconsistency when upscaling latent. Without latent upscale sampler works fine. The latent probably must be normalized somehow before the latent upscale.


> **Comment by Covalent0714 (2026-08-20T17:14:56Z):**

I tried trimming the latent by the same amount as the context node trims it for the low resolution generation before passing it to the upscaler, but I still got the same error. I concur there has to be some other normalization method.


> **Comment by NikoDemon80 (2026-08-21T01:17:42Z):**

This is not the upscaler and it is not the context node. It is a core assumption.

The error comes from `comfy/ldm/minimax/model.py`, on the line that fills conditioning video rows into the sequence. Core sizes the conditioning slots from the **target latent's** spatial grid, but the conditioning latents themselves were encoded earlier, at whatever size the latent was then. Nothing checks that those still agree.

So any node that changes the latent's width or height between the H3 node and the sampler will throw this. It is not upscaler-specific, which is why it happens with all of them.

You can reproduce it with no custom nodes: stock MiniMax H3 Image to Video with a first frame, latent upscale between its LATENT output and KSampler, conditioning passed straight through. Same exception.

Clip 1 works because it has no conditioning video latents, so that line never runs. Clip 2 has pinned context rows, so it does.

Fix the wiring order: upscale the latent **before** it goes into H3 Motion Context, not after. The node takes the target latent as an input and sizes the pinned frames to it, so it needs to see the final resolution.

One catch. `context_latent` cannot cross a resolution change, because a latent cannot be resized. The node already refuses that with a clear message. On an upscaled chain, use `context_frames` instead. You eat a VAE round trip, which is the cost of changing resolution mid-chain.

If you are doing a low res pass, upscale, then a second pass on the same conditioning, that will not work with pinned context at all. The second pass needs conditioning built at the upscaled size.


> **Comment by NikoDemon80 (2026-08-21T01:28:05Z):**

The image explains it. This is a two-pass refine and both `Basic Guider` nodes are fed the same `conditioning`.

The first pass runs at the original size. Then the AV latent is split, the video latent is upscaled, and the second pass runs on the bigger latent using that same conditioning.

Core does not support that. In `comfy/ldm/minimax/model.py`, the slots for conditioning video rows are sized from the **target latent's** spatial grid, but the conditioning latents were encoded at the original size. On the second pass it allocates the big grid and gets handed the small one. That is your 9699 versus 5541.

Nothing checks this, which is why the failure is a raw shape mismatch instead of a message.

Clip 1 works because its conditioning has no pinned video rows, so that line never runs. Clip 2 goes through Motion Context, which pins real frames as conditioning, so it does. Same reason it happens with any upscaler.

`keep_proportion false` with `megapixels 1.0` changes both axes, so there is no configuration of the upscaler that avoids this.

Two ways out:

1. Drop the two-pass and generate at the final resolution. One Motion Context, sized to the latent the sampler actually runs on. Simplest, and with pinned motion context the refine pass is buying you less than you think.
2. Keep the two-pass and build conditioning twice, one Motion Context per pass, the second wired to the upscaled latent. Note that the second one cannot use `context_latent` unless the previous clip's saved latent is also post-upscale, since a latent cannot be resized. Otherwise use `context_frames` and accept the VAE round trip.

Worth mentioning: any node that changes latent width or height between the H3 node and the sampler will do this, with or without my nodes. Stock `MiniMax H3 Image to Video` with a first frame breaks the same way if you put an upscaler between its LATENT output and the sampler.


> **Comment by Covalent0714 (2026-08-21T01:50:42Z):**

So if I save the upscaled latent instead of the low-res one, I can decode it and use it for the context chaining? I assume I would just do a VAE Decode for both audio and video piped in from the load latent node.

I'd prefer if I could keep a two pass using this upscaler as I quite like it, so how might the wiring look for the two sets of conditioning?

Thank you for your insight! I feel a little bit foolish now.


> **Comment by Covalent0714 (2026-08-22T04:03:58Z):**

I've been able to devise a workflow based on the advice that incorporates a fresh conditioning pass (i.e., building conditioning twice) that avoids the shape mismatch, saves the lowres latent for clip continuation, and latent upscales. After trimming, I can concatenate the upscaled clips. Seems to work! Thanks for the help! I'll go ahead and close the issue seeing as I have resolved it.



### #15 [closed] ISSUE: AttributeError: NestedTensor has no attribute clone/dim on H3 latents
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/15  by badraymen 2026-08-19T22:20:19Z

## Bug Description
The LBH H3 Latent Upscaler nodes (2D and 3D) crash when receiving standard MiniMax H3 NestedTensor latents.

## Errors
### 2D Node
`AttributeError: 'NestedTensor' object has no attribute 'clone'`
at `minimax_h3_latent_upscaler_2d.py:438`: `s = latent["samples"].clone()`

### 3D Node
`AttributeError: 'NestedTensor' object has no attribute 'dim'`
at `minimax_h3_latent_upscaler_3d.py:419`: `was_4d = (src.dim() == 4)`

## Root Cause
MiniMax H3 latents are `NestedTensor` (video+audio pair). LBH nodes assume `samples` is a plain `torch.Tensor` and call tensor methods directly.

## Environment
- ComfyUI: 0.33.0
- OS: Windows 11
- PyTorch: 2.13.0+cu130
- GPU: RTX 4090 24GB
- H3 checkpoint: minimax_h3_fl2va_pruned_int8_convrot

## Suggested Fix
Unwrap NestedTensor before tensor ops:
```python
samples = latent["samples"]
if hasattr(samples, "video"):
    samples = samples.video
s = samples.clone()
```



### #14 [open] ISSUE: [Discussion] With 8-step turbo lora, the quality seems better than 4-step? (Other settings unchanged)
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/14  by NeverOccurs 2026-08-19T15:47:02Z

Other settings unchanged, 8-step turbo lora (generation still 8 steps with split sigmas 4 step) seems better than 4-step in example workflows. 

> **Comment by Rkkss (2026-08-20T02:19:38Z):**

4steps lora run with modelsamplingH3 - 6/3 I think



### #13 [open] ISSUE: AttributeError: 'NestedTensor' object has no attribute 'clone'
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/13  by ioritree 2026-08-19T08:48:03Z

ComfyUI-MiniMaxH3-Contex-Loop 節點。

<img width="1476" height="868" alt="Image" src="https://github.com/user-attachments/assets/0cb6e7d6-2f2a-4e87-bee5-82c9dc8dc621" />

[ERROR] Traceback (most recent call last):
  File "G:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "G:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "G:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\Comfyui_Minimax_h3_latent_Upscaler\nodes\minimax_h3_latent_upscaler_2d.py", line 438, in run
    s = latent["samples"].clone()
        ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NestedTensor' object has no attribute 'clone'

> **Comment by LBH-123-AI (2026-08-19T10:46:52Z):**

从您的截图来看，您当前使用的是 2D 卷积版本的节点。需要说明的是，这个模型目前还在训练调优阶段，可能还需要一段时间才能发布稳定版本。建议您先改用 **3D 版本**的节点，并配合 ComfyUI 自带的 **拆分 Latent（Split Latent）** 和 **合并 Latent（Merge Latent）** 节点一起使用（截图所示）。或者直接截取示例工作流中大latent放大部分。

<img width="1353" height="540" alt="Image" src="https://github.com/user-attachments/assets/7a0df25b-dd96-4c47-856d-32216bd506f3" />


> **Comment by badraymen (2026-08-19T22:22:18Z):**

Update: the 3D node also crashes with the same NestedTensor issue: `AttributeError: NestedTensor object has no attribute dim` at `minimax_h3_latent_upscaler_3d.py:419` (`was_4d = (src.dim() == 4)`). So both 2D and 3D nodes are currently unusable with standard H3 NestedTensor latents until the unwrap fix is applied.


> **Comment by hirak99 (2026-09-26T17:36:51Z):**

Works for me.

If you see the response, and the example workflow, you need to apply the upscale _only_ to the video latent.

You need to therefore use the `Separate AV Latent` and `Concat AV Latent` nodes.

<img width="1031" height="399" alt="Image" src="https://github.com/user-attachments/assets/e5007861-023a-40da-be45-82797d6a5701" />



### #12 [closed] ISSUE: Question About Calculating the Latent Size
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/12  by javawock7618 2026-08-19T06:48:30Z

Hello. When I set 0.9 megapixels in H3 Latent Upscaler (Megapixels), it returns a latent size of 80 × 48, which corresponds to 1280 × 768 pixels. I believe the H3 latent dimensions use a multiple of 16.

However, when using the native Resolution Selector, which calculates dimensions using a multiple of 32, it returns 1280 × 736.

Would it be possible to make the H3 Latent Upscaler node return the same dimensions as the native Resolution Selector?

I would like both nodes to produce 1280 × 736 when 0.9 megapixels is specified.


By the way, when using MiniMax H3 Latent Upscaler (3D) with a 2× scale, it simply doubles the dimensions as expected:
608 × 352 → 1216 × 704.

Thank you!

> **Comment by LBH-123-AI (2026-08-19T10:30:46Z):**

@javawock7618 

Hello, 

Just a quick update — I've refactored the 3D node as suggested. The three scaling modes (`scale by multiplier`, `target dimensions`, and `megapixels`) have now been merged into a single node, making the interface cleaner. I've also fixed a few bugs along the way.

Please give it a try and let me know how it works for you. I'm still polishing some details, so any feedback or further suggestions are very welcome.

Thanks again for your great idea!


> **Comment by javawock7618 (2026-08-19T11:37:58Z):**

Hello. Thank you for the quick update. I’ve checked the changes, and this node is working very well in my workflow as well. If I notice anything while using it, I’ll report it again. Thank you!



### #11 [closed] ISSUE: It works. I have a small request.
URL: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/11  by javawock7618 2026-08-19T03:12:19Z

Hello. Your Latent Upscaler has produced the best results of any latent upscaler I’ve tried. Thank you very much.

I have a small request:

Could you combine the scale options for Megapixels, Dimension, and Multiplier into a single node?

And this is an even smaller request:

Could you standardize all logs and node tips to English?

Thank you, and please keep up the great work!

> **Comment by LBH-123-AI (2026-08-19T03:28:04Z):**

Hello,

Thank you for your kind words — I'm really glad to hear that the Latent Upscaler has been working well for you!

Regarding your requests:

1. Combining Megapixels, Dimension, and Multiplier scale options into a single node is a good idea. I'll consider it for a future update to simplify the interface.
2. Standardizing all logs and node tips to English also makes sense for broader accessibility. I'll work on that as well.

However, please note that implementing these changes may take some time, as I need to carefully refactor the code and ensure everything stays stable. I'll try my best to get it done, but I can't promise a quick timeline.

Also, if you have any other creative ideas or suggestions, feel free to share them with me — I'm always open to new thoughts that could improve the tool.

Thank you again for your feedback, and I'll keep improving the tool. Stay tuned!

Best regards

