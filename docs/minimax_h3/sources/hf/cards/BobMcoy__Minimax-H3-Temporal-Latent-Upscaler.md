# BobMcoy/Minimax-H3-Temporal-Latent-Upscaler

Источник: https://huggingface.co/BobMcoy/Minimax-H3-Temporal-Latent-Upscaler · скачано 2026-11-28
Загрузок: 0 · лайков: 1 · базовая:  · теги:  · обновлено: 2026-11-24T09:54:29.000Z

---

# MiniMax H3 Temporal Latent Upscaler for ComfyUI

Fix fast-motion smear, ghosting, and frame stepping in MiniMax H3 videos—and optionally upscale resolution in the same pass—directly in latent space.

Instead of decoding to pixels and re-encoding like pixel De-RoPE, this pack takes the **latent output of your first H3 sampler**, detects which 17-frame blocks have fast motion, stretches only those moments to 2× or 3× frames during Pass 2 so H3 can redraw them cleanly, and recovers your exact original frame count, 24 fps timing, and untouched Pass-1 audio.

---

## See the Difference

### Live-Action · Source vs Adaptive `1.0×` vs Decoupled `1.5×` *(v2.1 Default)*
[![Looping live-action parkour comparison: H3 source, adaptive 1.0x repair, and adaptive + 1.5x spatial.](demos/parkour-live-v21-preview.webp)](demos/README.md#live-action-parkour--adaptive-10-repair--decoupled-15-spatial-v21-default)

**Watch the midair flip, sneakers, and landing face.** Left is raw H3 Pass 1 (`672×384`), middle is `1.0×` dense adaptive repair (`672×384`), and right is `1.5×` decoupled spatial + adaptive temporal repair (`1024×576`) with razor-straight brickwork and untouched stereo audio.

### 2D Anime · Source vs Adaptive `1×/2×/3×` + `1.5×` Spatial *(v2.1 Default)*
[![Looping 2D anime comparison: H3 source on the left and adaptive + 1.5x spatial on the right.](demos/anime-courtyard-v21-preview.webp)](demos/README.md#2d-anime-courtyard-slash--adaptive-123--decoupled-15-spatial-v21-default)

**Watch the midair flip and sword draw.** Left is raw H3 Pass 1 (collapsed flip & smeared face). Right is v2.1 adaptive `1×/2×/3×` + `1.5×` spatial repair (`1024×576`), playing at the exact same 24 fps timing and stereo audio.

👉 [Full comparison gallery (live-action parkour, 2D anime courtyard, backflip, basketball pixel-vs-latent, anime slash)](demos/README.md)

---

## Updates

### v2.1 — Native ComfyUI Core Polish, Multi-LoRA & `BlockSparseAttention` (Nov 2026)

Polished the unified workflow around ComfyUI's latest native MiniMax H3 nodes and hardened 8 GB VRAM handoffs:

- **Replaced deprecated Triton Sol-Attn with native `Model Sparse Attention` (`BlockSparseAttention`):** Pass 2 now uses ComfyUI core's `BlockSparseAttention` (`selection="sol-attn"`, `tau=1.3`, `min_tokens=20480`, `end_percent=0.95`, `extra_tokens=256`, `sink_conditioning="exact_kv_and_rows"`), eliminating `ComfyUI-sol-attn`. Raising `min_tokens` to `20480` keeps `0.25–0.40 MP` `1.0×` clips 100% dense (eliminating sparse-attention face/eye smudging), while `end_percent=0.95` automatically switches the final detail step (`sigma < 0.387`) to **100% dense** on `1.5×`/`2.0×` upscales for crisp edges and textures. `MiniMaxChunkFeedForward` runs on both passes while `MiniMaxLowVRAMAttention` is scoped to Pass 1 only so it never conflicts with Pass 2's native chunked sparse producer.
- **Cross-domain verified Pass-2 defaults (`res_multistep` + 5-step `beta 12 / 0.42`):** Switched Pass-2 sampler to deterministic **`res_multistep`** (matching ComfyUI's official MiniMax H3 templates and avoiding stochastic mid-step noise) paired with 5-step **`beta 12 / inject 0.42`** (`sigma=0.883`, final detail step `0.271` running 100% dense via `end_percent=0.95`), cleanly eliminating `3× + 1.5×` flip-foot smudging across both 2D anime and live-action parkour while staying fast (`5` steps).
- **Bypassed `FirstBlockCache` by default for ComfyUI 0.45.0+ safety:** Under ComfyUI 0.45.0 (`comfy-aimdo 0.6.0` `malloc_graph`), `ComfyUI-MiniMaxH3-FirstBlockCache` triggers a deterministic C++ abort in `clear_tensors` when freeing `remaining_blocks_residual`. Toggle `3. Pass-1 FirstBlockCache` now defaults to **OFF** until the upstream node wraps its allocation with `pause_malloc_graph()`.
- **MiniMax-native MP resolution & `17k + 5` duration snapping:** Uses native `ResolutionSelector` (aspect ratio + megapixels with `32px` snap, e.g. `0.25 MP` → `672×384`, `1.5×` → `1024×576`, `2.0×` → `1344×768`) and `PrimitiveFloat` seconds → collapsed `ComfyMathExpression` auto-snapping to MiniMax H3's `17k + 5` frame grid (`5.0s → 124f`, `6.5s → 158f`).
- **Independent Pass-1 & Pass-2 multi-LoRA stacks (`LoadLoraModel`):** Native `LoadLoraModel` stacks (up to 20 LoRAs per pass) placed after the shared trunk so Pass-1 Turbo/style LoRAs never pollute Pass 2, with independent Control Panel toggles (`2. Pass-1 LoRAs`, `4. Pass-2 LoRAs`).
- **Native lazy mode switching & audio decode:** Replaced `Any Switch (rgthree)` × 3 with a single native `PrimitiveBoolean` driving 3× lazy `ComfySwitchNode`s (`Active Model`, `Active Conditioning`, `Active Empty Latent`), switched audio decoding to native `VAEDecodeAudio` (dropping `ComfyUI-H3-Audio-Sanitize`), switched scheduler to native `beta` (dropping `RES4LYF`), and removed canvas `H3EvictTextEncoder` / `H3FreeCache` / `H3DyRoPE` nodes.
- **Fixed INT8 DiT eviction & 8 GB VRAM decode handoff:** Switched internal decorators from `@torch.inference_mode()` to `@torch.no_grad()` so ComfyUI's INT8 `_quantized_apply` eviction never hits `RuntimeError: Cannot set version_counter for inference tensor`, and updated `H3ExactVAEDecode` to unload `vae.patcher` and flush CUDA cache after decoding so `VAEDecodeAudio` runs cleanly with zero OOM on 8 GB GPUs.

### v2.0 — Speed, Quality & Unified Workflow Overhaul (Nov 2026)

A full code audit, multi-scene benchmark sweep, and workflow cleanup resolved several hidden bugs and simplified the whole repo down to **one toggleable production workflow**:

- **1.81× faster temporal upscaler, slabbed resampling & zero OOM:** Rewrote `temporal_checkpoint.py` onto ComfyUI core's optimized `MiniMaxH3LatentUpscaler` path (`channels_last_3d` convolutions, fused `comfy_kitchen` GroupNorm3D + SiLU, `SLAB=16` chunking for convs, temporal resampling, and spatial resizing, dynamic VRAM via `CoreModelPatcher`, and mtime caching). Removed forced `unload_all_models()` calls. (`3× + 1.5× spatial` dropped from OOM / `5.60s` down to **`3.10s / 1.16 GiB`**.)
- **Decoupled spatial upscaling (`spatial_checkpoint="auto"`):** Temporal fine-tuning in the `v2` temporal checkpoint degrades fractional `1.5×` spatial geometry (wavy/broken grids on fine lines) and 1-token FL2V/I2V keyframes. When `spatial_scale > 1.0`, the node now automatically detects the official MiniMax H3 spatial upscaler in `models/latent_upscale_models/`, runs temporal dilation at `1.0×`, and performs spatial upscaling + direct 1-token keyframe scaling once via the official spatial model — eliminating grid artifacts, cutting FL2V keyframe error by **-58%** (`MAE 11.0 → 4.6`), and speeding up prep by **-23%** (falls back gracefully to the temporal checkpoint if absent).
- **Sharper & faster `1.5×`/`2.0×` spatial upscaling (`untouched_spatial_denoise = 1.0`):** Defaulting untouched spatial denoise to `1.0` avoids baking unrefined resizer softness into calm blocks (**+26% to +79% sharper** untouched frames and **~15% faster** Pass-2 sampling).
- **Smarter adaptive motion detection:** Uses weighted top-2 step velocity for `2×` dilation (preventing calm boundary spillover) while keeping single-step peak velocity for `3×` bursts (preserving sharp kicks/slashes), with tightened short-tail gating and console token telemetry.
- **Fixed Pass-2 schedule bug & `H3DyRoPE` auto-skip cleanup (`H3RepairSchedule`):** Previously, `inject` was compared against sigma values instead of step counts, silently running only 1 Pass-2 step instead of 5. The new built-in `H3RepairSchedule` matches `H3InjectSchedule` bit-for-bit (true 5 steps on masked clips), **automatically skips Pass 2 in `0.02s`** on calm 1× clips, and cleanly disarms `H3DyRoPE` global state on 0-step skips.
- **Fixed FL2V first-frame drift:** Spatial upscaling of single-frame I2V/FL2V keyframe latents now takes the symmetric center token (`2:3`) instead of an edge token, bringing first-frame fidelity (`first_mae`) in line with the end frame.
- **Fixed 5-frame edge-case crash & tail over-dilation:** Fixed a crash on 5-frame clips in motion scoring and tightened the short-tail gate so quiet endings don't waste Pass-2 compute.
- **Added `H3ExactVAEDecode`:** Combines H3 video VAE decoding and exact hold-map frame recovery in one node, filters kept frames on GPU before pixel finalization (cutting decode RAM and GPU→CPU transfer by **2×–3×**, bit-exact vs `VAEDecode + H3ExactRecover`), skips trailing snap-pad chunks, and cleanly falls back to normal decoding if Pass 2 is bypassed.
- **Replaced 4 separate JSON graphs with 1 unified workflow:** T2V, I2V, FL2V, Ref2V, 8–24 GB Low-VRAM/Anti-OOM, FirstBlockCache (Pass 1 only), native `BlockSparseAttention` Sol-Attn (Pass 2 only), multi-LoRA stacks, and RTX VSR 2× are now all one-click toggles in [`workflows/H3_Unified_Production_Workflow.json`](workflows/H3_Unified_Production_Workflow.json).

---

## One Workflow for Everything

Open **[`workflows/H3_Unified_Production_Workflow.json`](workflows/H3_Unified_Production_Workflow.json)** in ComfyUI. Use the **Control Panel on the left**—no rewiring needed:

### 1. Generation Mode *(Native Lazy Switch)*
- **`1. MODE: REF2VA = OFF` (FL2VA — Default):** Standard T2V, I2V, and FL2V (`MiniMax-H3-FL2VA`).
- **`1. MODE: REF2VA = ON` (REF2VA):** Reference-to-video using `<Picture 1..3>` and `<Audio 1>` (`MiniMax-H3-Ref2VA`). Lazy `ComfySwitchNode` routing never loads the unused UNET.

### 2. Optional Media Inputs *(Yellow · all OFF = T2V)*
- **A. Image 1:** First frame (FL2VA) or `<Picture 1>` (REF2VA).
- **B. Image 2:** Last frame (FL2VA) or `<Picture 2>` (REF2VA).
- **C. Image 3:** `<Picture 3>` (REF2VA only).
- **D. Ref Audio:** `<Audio 1>` (REF2VA only).

### 3. Hardware, LoRAs, Speedups & Finish *(Brown)*
| Toggle | Default | What it does |
| --- | :---: | --- |
| **1. Low VRAM & Anti-OOM (8-24GB KJ Attn+FFN)** | **ON** | Bit-exact chunked FFN on both passes (`chunks=4`) + low-VRAM attention on Pass 1 (`head_chunks=2`). **Keep ON for 8–24 GB GPUs**; optionally turn OFF on 32–48 GB+ GPUs. |
| **2. Pass-1 LoRAs (Multi-LoRA)** | OFF | Native `LoadLoraModel` stack (up to 20 LoRAs) scoped strictly to Pass 1. For 4-step Turbo drafts, enable and set Pass-1 steps to `4`–`6`. |
| **3. Pass-1 FirstBlockCache (Fast Base)** | OFF | ~11% faster Pass 1 (`H3 Safe`) on older ComfyUI builds. **Keep OFF on ComfyUI 0.45.0+** due to a `comfy-aimdo` `malloc_graph` abort in the upstream node. |
| **4. Pass-2 LoRAs (Multi-LoRA)** | OFF | Independent native `LoadLoraModel` stack (up to 20 LoRAs) for Pass-2 detail or Turbo LoRAs without affecting Pass 1. |
| **5. Pass-2 Model Sparse Attention (Sol-Attn)** | **ON** | Native ComfyUI `BlockSparseAttention` (`sol-attn`, `tau=1.3`, `min_tokens=20480`, `end_percent=0.95`, `exact_kv_and_rows`) — keeps `1.0×` base clips 100% dense, accelerates `1.5×`/`2.0×` early/mid steps, and finishes the last detail step dense. |
| **6. Pass-2 Temporal Repair & Latent Upscale** | **ON** | Adaptive 1×/2×/3× motion repair + optional `spatial_scale` (`1.0` / `1.5` / `2.0`). Auto-skips calm clips in `0.02s`. Toggle OFF to output Pass 1 directly. |
| **7. RTX VSR (2x ULTRA Finish)** | OFF | ~1.4s NVIDIA RTX pixel polish (`672×384 → 1344×768`, or `1024×576 → 2048×1152`). |

### Quick Hardware Presets
- **8–24 GB VRAM (RTX 4060 / 3090 / 4090 — Default):** Keep defaults ON (`1`, `5`, `6` ON; `2`, `3`, `4`, `7` OFF). Toggle 1 is bit-exact and prevents FFN activation OOMs during long/upscaled Pass-2 runs.
- **32–48 GB+ VRAM:** Optionally turn **`1. Low VRAM & Anti-OOM`** OFF.
- **Long Clips (`124f–158f+`) at High Resolution:** Prefer `spatial_scale=1.0` or `1.5` + **`7. RTX VSR (2× ULTRA)`** (~1.4s finish) instead of `spatial_scale=2.0`, since attention cost grows quadratically with token count.
- **Maximum Zero-Approximation Dense Quality:** Turn **`3. Pass-1 FirstBlockCache`** and **`5. Pass-2 Model Sparse Attention`** OFF.
- **Fast Draft / Prompt Iteration:** Turn **`2. Pass-1 LoRAs`** ON (with a Turbo LoRA, Pass-1 steps `4`–`6`) and turn **`6. Pass-2 Temporal Repair`** OFF.

---

## Quick Start

### 1. Install Dependencies
Use a recent ComfyUI build (`0.33.0+`) with native MiniMax H3, `ResolutionSelector`, `LoadLoraModel`, `ComfySwitchNode`, `VAEDecodeAudio`, and `BlockSparseAttention` (`Model Sparse Attention`) support.

**Required custom nodes** (install into `ComfyUI/custom_nodes/`):
- **[ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes)** — H3 timeline helpers (`H3TemporalInsert`) used internally by `H3AdaptiveLatentRepair`.
- **[rgthree-comfy](https://github.com/rgthree/rgthree-comfy)** — Toggle panels (`Fast Groups Bypasser`) in the Control Panel.
- **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)** — Low-VRAM attention & FFN chunking (can be bypassed on 32 GB+ GPUs).
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** — Compact GGUF Qwen3-VL text encoder loader (or swap `CLIPLoaderGGUF` for native `CLIPLoader` with FP8/NVFP4 safetensors).

**Optional speedup / finish nodes** (safe to leave bypassed if not installed):
- **[ComfyUI-MiniMaxH3-FirstBlockCache](https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache)** — Pass-1 FirstBlockCache (`-11%` runtime).
- **[Nvidia_RTX_Nodes_ComfyUI](https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI)** — Optional 2× RTX Video Super Resolution finish.

*(No longer needed: `ComfyUI-sol-attn` is replaced by native `BlockSparseAttention`, `ComfyUI-H3-Audio-Sanitize` is replaced by native `VAEDecodeAudio`, and `RES4LYF` is replaced by native `beta`.)*

### 2. Install This Pack
Open a terminal inside `ComfyUI/custom_nodes/` and clone:

```sh
git clone https://github.com/HyRo34/Minimax-H3-Temporal-Latent-Upscaler.git
```

*(Or download the GitHub ZIP and extract it so the folder inside `custom_nodes/` directly contains `__init__.py`.)*

### 3. Download Models
Place the weights in the folders below (relative to `ComfyUI/`). If you already run MiniMax H3, you only need the **first row** (George0667's temporal checkpoint) and can select your existing H3 weights in the workflow.

| File | Folder | Required? | Download |
| --- | --- | :---: | --- |
| `minimax_h3_latent_upscaler_3d_conv_v2_temporal_bf16.safetensors` | `models/h3_temporal_models/` | **Yes** | [George0667 temporal v2 BF16](https://huggingface.co/George0667/MiniMax_Temporal_Upscaler/blob/b30591f7215e8198eabb1b05a7984b57bf046176/minimax_h3_latent_upscaler_3d_conv_v2_temporal/minimax_h3_latent_upscaler_3d_conv_v2_temporal_bf16.safetensors) |
| `minimax_h3_latent_upscaler_3d_fp16.safetensors` | `models/latent_upscale_models/` | Recommended (`1.5×`/`2.0×`) | [H3 3D Spatial Upscaler FP16](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/minimax_h3_latent_upscaler_3d_fp16.safetensors) (or [BF16](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/minimax_h3_latent_upscaler_3d_bf16.safetensors)) |
| `MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors` | `models/diffusion_models/` | **Yes** (FL2VA) | [DeepBeepMeep H3 FL2VA](https://huggingface.co/DeepBeepMeep/MiniMax-H3/blob/main/MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors) |
| `qwen3vl-32B-MiniMax-H3-Q2_K.gguf` | `models/text_encoders/` | **Yes** | [ncrzy H3 GGUF bucket](https://huggingface.co/buckets/ncrzy/MiniMax-H3_GGUFs-bucket) |
| `minimax_h3_video_vae_fp8mix.safetensors` | `models/vae/` | **Yes** | [DeepBeepMeep H3 Video VAE](https://huggingface.co/DeepBeepMeep/MiniMax-H3/blob/main/minimax_h3_video_vae_fp8mix.safetensors) |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | **Yes** | [Comfy-Org H3 Audio VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| `MiniMax-H3-Ref2VA-pruned_rank8_int8_convrot.safetensors` | `models/diffusion_models/` | Optional (REF2VA) | [DeepBeepMeep H3 Ref2VA](https://huggingface.co/DeepBeepMeep/MiniMax-H3/blob/main/MiniMax-H3-Ref2VA-pruned_rank8_int8_convrot.safetensors) |

### 4. Run
Restart ComfyUI, drag **`workflows/H3_Unified_Production_Workflow.json`** onto the canvas, edit your prompt, and click **Queue**. Finished MP4s with stereo audio save to `ComfyUI/output/video/`.

---

## Included Nodes

If you prefer wiring the nodes into your own custom H3 graph, this pack provides 4 clean nodes:

1. **`H3AdaptiveLatentRepair`** *(H3 Latent Auto Repair & Upscale)* — Scores motion per 17-frame block in latent space, runs George0667's temporal/spatial 3D latent upscaler on ComfyUI's fused CUDA path, builds the selective `noise_mask`, stretches audio conditioning, and remaps I2V/FL2V keyframes.
2. **`H3RepairSchedule`** *(H3 Repair Inject Schedule · Auto-Skip)* — Builds the Pass-2 denoising tail (`beta 12 / inject 0.42` → 5 steps from `sigma=0.883`, final detail step at `0.271`) and automatically outputs an empty 0-step schedule when nothing is masked (calm 1× clips skip Pass 2 in `0.02s`).
3. **`H3ExactVAEDecode`** *(H3 Exact VAE Decode & Recover)* — Decodes the video latent and recovers the exact original frame count in one step, skipping wasted trailing pad chunks and falling back cleanly to normal decode when `hold_map` is empty.
4. **`H3RepairDecodeAudio`** *(H3 Repair · Decode Original Audio)* — Optional fallback audio decoder with explicit NaN/Inf sanitization (the unified workflow uses native ComfyUI `VAEDecodeAudio` directly).

---

## How It Works (Simply)

1. **Detect motion directly in latents:** Each 17-frame H3 block is scored for motion without running the video VAE. Calm blocks stay `1×`, moderate bursts get `2×`, and extreme bursts (flips, kicks, fast sword arcs) get `3×`.
2. **Expand in latent space:** George0667's temporal checkpoint inserts intermediate latent frames (and optionally upscales spatial resolution `1.5×` or `2.0×`) in ~3 seconds.
3. **Selective 5-step Pass-2 redraw:** Only the expanded/upscaled regions are refined for 5 deterministic `res_multistep` denoising steps (`beta 12 / 0.42`). If the whole clip is calm at `1.0×`, Pass 2 skips automatically in `0.02s`.
4. **Exact timing + untouched audio:** `H3ExactVAEDecode` picks back the exact original frame count at 24 fps (and releases the Video VAE afterward for clean 8 GB handoff), and native `VAEDecodeAudio` + `CreateVideo` mux the untouched Pass-1 stereo audio track.

---

## Verified Benchmarks (RTX 4060 Laptop · 8 GB VRAM)

All measurements taken on an **RTX 4060 Laptop (8 GB VRAM)** using `MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors`:

### Temporal Checkpoint Upscaler (`3×` temporal + `1.5×` spatial)
| Implementation | Time | Peak VRAM | Result |
| --- | ---: | ---: | --- |
| Previous BF16 eager convs | `5.60s` (or OOM `17.4 GiB`) | `1.70 GiB` | Forced full model unloads |
| **Updated fused CUDA + slabbing** | **`3.10s`** (**1.81× faster**) | **`1.16 GiB`** (**-32%**) | Dynamic VRAM, zero OOM |

### Pass 1 (`640×384`, 90 frames, 20 steps, 7,028 tokens)
| Config | Time | Peak Act VRAM | Sharpness | Notes |
| --- | ---: | ---: | ---: | --- |
| Dense baseline (`KJ + FFN`) | `101.34s` | `0.54 GB` | `385.4` | Zero-approximation reference |
| **FirstBlockCache `H3 Safe`** *(Default)* | **`90.21s`** (**-11%**) | `0.95 GB` | `345.6` | Visually matched baseline (6/20 cached) |
| Sparse attention at 7k tokens | `99.7s`–`107.7s` | `0.62–0.71 GB` | `262–290` | **Rejected** — blurred limbs at low token count |

### Pass 2 (`960×576`, `1.5×` spatial, 175 expanded frames, 5 steps, 28,912 tokens)
| Config | Time | Peak Act VRAM | Img MAE vs Dense | Sharpness | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Dense baseline (`KJ + FFN`) | `189.34s` | `2.23 GB` | `0.00` | `115.1` | Reference |
| Legacy Triton Sol-Attn INT8 | `171.09s` | `2.56 GB` | `5.62 / 255` | `128.8` | Deprecated upstream |
| **Native `BlockSparseAttention` (`end_percent=1.0` → `0.95`)** *(Default)* | **`128.19s`** (`-32%`, +~11s w/ dense tail) | **`2.21 GB`** | `5.30 / 255` | `108.6` (`ep=1.0`) → **`+15–20%` w/ `ep=0.95`** | Native chunked INT8 producer; `end_percent=0.95` finishes the final step dense |
| Calm 1× clip (`H3RepairSchedule` auto-skip) | **`0.02s`** | — | `0.00` | exact | 0 Pass-2 steps |

### End-to-End Finishing
- **FL2V 1.5× 5-step (`960×576`, 90f, 35,875 tokens):** `3.14 GB` peak activation VRAM, symmetric keyframe fidelity (`first_mae=10.85`, `last_mae=10.30`), stereo 32 kHz MP4 export.
- **Live ComfyUI 0.45.0 Cross-Domain Verification (`res_multistep`, 5-step `beta 12 / 0.42` `0.883`, `BlockSparseAttention` `min_tokens=20480` / `end_percent=0.95` / `exact_kv_and_rows`, 90f / `3.75s` stereo 32 kHz audio):**
  - **2D High-Motion Anime (`[1,1,2,3,2,1]`):** Pass 1 (`672×384`, `192.2s`, `7.53 GiB`, sharp `174.2`) → **`1.5×` `res_multistep` 5st (`1024×576`, `+180.4s`, `7.29 GiB`, sharp `247.3`)** — sharpest result of all ablations, completely eliminating `f44`/`f45` flip-foot smudging and restoring crisp cel linework & faces with zero grid artifacts.
  - **Live-Action Parkour Vault/Flip (`[2,2,2,2,2,1]`):** Pass 1 (`672×384`, `208.2s`, `7.61 GiB`, sharp `326.0`) → **`1.0×` Dense (`672×384`, `+134.1s`, `7.08 GiB`, sharp `367.0`)** → **`1.5×` 5st (`1024×576`, `+198.2s`, `7.12 GiB`, sharp `381.0`, landing `f82: 270.1`)** — cleanly rescued motion-blurred landing faces, hands, and sneakers while keeping brick/railing lines razor-straight (`38s` faster than 6-step).
- **8 GB Decode Handoff (`H3ExactVAEDecode` 209f → 90f @ `1024×576` + native `VAEDecodeAudio` 158f):** Video VAE peaks at `5.58 GiB` and unloads to `0.15 GiB`; native `VAEDecodeAudio` follows cleanly in `3.17s` at `0.86 GiB` peak VRAM with zero OOM.
- **RTX VSR 2× ULTRA (`640×384 → 1280×768`, 90f):** **`1.36s`** upscale time.

---

## Tuning Tips

- **Motion modes:** Start with `auto_adaptive (1x/2x/3x)`. Use `auto_2x` for a lighter repair pass, `full_2x`/`full_3x` to repair the whole clip, or `manual` with frame ranges like `17-51` (or `all`).
- **Pass-2 sampler & repair strength:** Keep Pass-2 sampler on deterministic **`res_multistep`** with 5-step **`beta` `12` / `0.42`** (`sigma=0.883`, dense final detail step at `0.271` via `end_percent=0.95`) for the fastest, cleanest smear removal on `2×`/`3×` flips and `1.5×` upscales. Switch to **`beta` `16` / `0.35`** (`6` steps from `sigma=0.855`) or **`beta` `20` / `0.30`** (`6` steps from `sigma=0.787`) when you want tighter Pass-1 pose/identity lock on gentler clips.
- **Valid H3 sizes & lengths:** Controlled automatically in the unified workflow via `ResolutionSelector` (MP + `32px` snap) and `Duration (Seconds)` → `17k + 5` frame snap (`73`, `90`, `107`, `124`, `141`, `158`, `192`).

---

## Credits & Licensing

- **MATLOWAI** — [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes) (Motion Lab, De-RoPE, timeline/frame recovery, `H3DyRoPE`).
- **George0667** — [MiniMax Temporal Upscaler](https://huggingface.co/George0667/MiniMax_Temporal_Upscaler) temporal v2 weights (**CC BY-NC 4.0**).
- **LBH-123-AI** — [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) original 3D latent-resizer backbone ([MIT notice](LICENSES/LBH-MIT.txt)).
- **MiniMax**, **Comfy Org**, **Kijai**, **DeepBeepMeep**, **city96**, **ncrzy**, **rgthree**, **Saganaki22**, **duckyshell**, and **NVIDIA** for the H3 models, ComfyUI core integration, low-VRAM patches, GGUF tools, Sol-Attn, FirstBlockCache, and RTX VSR.
- **TK_999**, **Kijai**, and the **Banodoco** community for benchmark scenarios and feedback.

The [MIT notice](LICENSES/LBH-MIT.txt) covers the adapted LBH backbone. Model weights are distributed by their respective authors under their own licenses.
