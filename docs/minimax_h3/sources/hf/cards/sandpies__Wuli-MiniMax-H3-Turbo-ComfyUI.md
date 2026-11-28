# sandpies/Wuli-MiniMax-H3-Turbo-ComfyUI

Источник: https://huggingface.co/sandpies/Wuli-MiniMax-H3-Turbo-ComfyUI · скачано 2026-11-28
Загрузок: 5621 · лайков: 10 · базовая: MiniMaxAI/MiniMax-H3 · теги: comfyui,lora,step-distillation,turbo · обновлено: 2026-10-15T20:02:42.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
tags:
- comfyui
- lora
- minimax-h3
- text-to-audio-video
- image-to-audio-video
- step-distillation
- turbo
- int8
---

# Wuli MiniMax-H3 Turbo (4-step + 8-step) — ComfyUI conversions

**Comparisons (same prompt and seed per row, euler / simple / 8 steps).** Left to right: hybrid, hybrid + Realism, FL2VA FlashGen.

Rainy bookstore café:

<video controls src="https://huggingface.co/sandpies/Wuli-MiniMax-H3-Turbo-ComfyUI/resolve/main/8step-hybrid/prompt1_hybrid_comparison.mp4"></video>

Golden-hour rooftop:

<video controls src="https://huggingface.co/sandpies/Wuli-MiniMax-H3-Turbo-ComfyUI/resolve/main/8step-hybrid/prompt2_hybrid_comparison.mp4"></video>

> **Update (Oct 16):** **F2** replaces the earlier `reflora100` hybrid as the recommended reference-to-video model. reflora100 baked in Kijai's Ref2VA reference LoRA, a low-rank approximation of the Ref2VA−FL2VA difference. F2 bakes in the full, uncompressed difference for blocks 25-49 instead, so it carries all of Ref2VA's reference behaviour in those blocks rather than a squeezed version, and it held up better than reflora100 in my tests. Same settings, same file size.


ComfyUI-format conversions of **Wuli-Art's MiniMax-H3 Turbo** few-step audio+video distillations of
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3):

- **`4step/`** — Wuli's 4-step adapter (`turbo-4step/Wuli-MiniMax-H3-Turbo-4steps-V1.0-bf16.safetensors` from
  [Wuli-Art/MiniMax-H3-Turbo](https://modelscope.ai/models/Wuli-Art/MiniMax-H3-Turbo)): a LoRA + 4 per-step output heads.
- **`8step/`** — Wuli's 8-step transformer (`turbo-8step/transformer` from
  [Wuli-art/MiniMax-H3-Turbo](https://huggingface.co/Wuli-art/MiniMax-H3-Turbo/tree/main/turbo-8step/transformer)):
  a full model with its 8 timesteps baked in, plus an int8 quant and an approximate LoRA.
- **`8step-ref2va/`** — the same 8-step distill moved onto the **Ref2VA** model (reference-image conditioning):
  built as Ref2VA + (Wuli 8-step − FL2VA). A weight merge, not something Wuli trained. **Untested in generation.**
- **`8step-unlocked/`, `8step-ref2va-unlocked/`** — the same two 8-step int8 models with only the small time table smoothed,
  so other turbo step counts (about 6–12), schedulers and samplers work. The original 8-step schedule is unchanged.
  Each also has an **HQ** int8 (`..._int8_convrot_hq`, 25.0 GB) that keeps the most quantization-sensitive layers in bf16.
- **`8step-hybrid/`** - one model for both **text-to-video and reference-to-video**: the FL2VA FlashGen merge with Ref2VA
  grafted into late blocks. Plain and Realism-People versions (adaLN-only graft), plus **F2** with the full Ref2VA
  difference in blocks 25-49. HQ int8, 25.0 GB each.

All credit for the models goes to **Wuli-Art / the Alibaba Wuli Team** (with the Alibaba RTP-Compression Team)
and to **MiniMax**. This repo only re-packages the weights for ComfyUI; nothing was retrained.

**Requires ComfyUI >= 0.38.0.** **Status:** CPU-verified through stock ComfyUI 0.38 code paths (details per section).
**Not yet validated with a GPU render.**

> This repo was previously `sandpies/Wuli-MiniMax-H3-Turbo-4step-ComfyUI` (old links redirect here). The 8-step files
> previously lived in `sandpies/Wuli-MiniMax-H3-Turbo-8step-ComfyUI` and are now in `8step/`.

## Which file do I need?

| I want… | Get | Sampler settings |
|---|---|---|
| **Recommended for reference-to-video: 8 steps, F2 hybrid (full Ref2VA blocks 25-49 baked in)** (one model for text-to-video and reference-to-video) | `8step-hybrid/wuli_h3_turbo_8step_v1_F2_flbase_ref2va-b25-49-full_gatedwuli_flashgen050_int8_convrot_hq.safetensors` (25.0 GB) → `models/diffusion_models/` | euler, `simple`, **8 steps**, shift **12/3**, CFG 1 |
| **Recommended: 8 steps, Wuli unlocked + FlashGen 0.5 (tested by the repo owner, works well as-is)** | `8step-unlocked-flashgen/wuli_h3_turbo_8step_v1_unlocked_flashgen050_comfyui_int8_convrot_hq.safetensors` (25.0 GB) → `models/diffusion_models/` | shift **12/3**, `simple`, **8 steps**, euler, CFG 1 |
| 8 steps, **one model for text-to-video and reference-to-video** (hybrid: FL2VA FlashGen base + Ref2VA reference adaLN, blocks 25-49) | `8step-hybrid/wuli_h3_turbo_8step_v1_flbase_refadaln_b25-49_flashgen050_int8_convrot_hq.safetensors` (25.0 GB) → `models/diffusion_models/` | euler, `simple`, **8 steps**, shift **12/3**, CFG 1 |
| same hybrid **with Realism-People 1.0 baked in** (trigger `r34l1sm`) | `8step-hybrid/wuli_h3_turbo_8step_v1_flbase_refadaln_b25-49_flashgen050_realism_int8_convrot_hq.safetensors` (25.0 GB) → `models/diffusion_models/` | same |
| 8 steps **with Ref2VA reference conditioning + FlashGen 0.5 + Realism-People LoRA, merged** (experimental; use Ref2VA conditioning nodes) | `8step-ref2va-unlocked-flashgen-realism/wuli_h3_turbo_8step_v1_ref2va_unlocked_flashgen050_realism_comfyui_int8_convrot_hq.safetensors` (25.0 GB) → `models/diffusion_models/` | shift **12/3**, `simple`, **8 steps**, euler, CFG 1 (unlocked: 6–12 steps OK) |
| 8 steps, exact, plain Wuli (smallest int8) | `8step/wuli_h3_turbo_8step_v1_comfyui_int8_convrot.safetensors` (21.4 GB) → `models/diffusion_models/` | shift **12/3**, `simple`, **8 steps**, euler, CFG 1 |
| 8 steps, exact, full precision | `8step/wuli_h3_turbo_8step_v1_comfyui_bf16.safetensors` (40.6 GB) → `models/diffusion_models/` | same |
| 8 steps **with Ref2VA reference conditioning** (experimental merge, untested) | `8step-ref2va/wuli_h3_turbo_8step_v1_ref2va_comfyui_int8_convrot.safetensors` (21.4 GB) or `..._bf16` (40.6 GB) → `models/diffusion_models/` | same 8-step settings |
| **6–12 steps, any scheduler/sampler** (same 8-step model, smoothed time table) | `8step-unlocked/wuli_h3_turbo_8step_v1_unlocked_comfyui_int8_convrot.safetensors` (21.4 GB) → `models/diffusion_models/` | shift **12/3** recommended (adjustable), 6–12 steps, `simple`/`beta`/`sgm_uniform`, euler / dpmpp_2m / res_multistep, CFG 1 |
| same, **with Ref2VA** (experimental merge) | `8step-ref2va-unlocked/wuli_h3_turbo_8step_v1_ref2va_unlocked_comfyui_int8_convrot.safetensors` (21.4 GB) → `models/diffusion_models/` | same as above |
| unlocked, **higher quality int8 (HQ)**, 24 GB+ VRAM | `8step-unlocked/wuli_h3_turbo_8step_v1_unlocked_comfyui_int8_convrot_hq.safetensors` (25.0 GB) or Ref2VA: `8step-ref2va-unlocked/wuli_h3_turbo_8step_v1_ref2va_unlocked_comfyui_int8_convrot_hq.safetensors` (25.0 GB) → `models/diffusion_models/` | same as above |
| 8 steps on my own base (approximate) | `8step/wuli_h3_turbo_8step_v1_lora_r256_comfyui.safetensors` (2.6 GB) → `models/loras/` | same |
| **4 steps on any base, exact (stock nodes only)** | `4step/` `trunk_comfyui_bf16` + `heads_widen` + `heads_set` → `models/loras/` | shift **6/3**, `simple`, **4 steps**, euler, CFG 1 |
| 4 steps, one file, FL2VA only | `4step/wuli_h3_turbo_4step_v1_full_fl2varel_comfyui.safetensors` → `models/loras/` | same 4-step settings |

Everything works in **stock ComfyUI (>= 0.38) with core nodes only**. The custom node zip in `4step/` is optional/deprecated.

**The shifts matter:** 4-step = **6/3**, 8-step = **12/3** (the `ModelSamplingMiniMaxH3` defaults). Each model only works on its own schedule — except the **unlocked** 8-step files, which take other step counts and schedules.

---

## 8-step unlocked × FlashGen (recommended)

A merge of the **8-step unlocked** model with the **FlashGen 4-step LoRA** (FlashGen by Beidouqixing / Huawei Ascend, Apache-2.0; ComfyUI conversion by [Kijai](https://huggingface.co/Kijai/MiniMax-H3-experimental), `loras/minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_pruned_avg_rank_13_bf16.safetensors`) at **strength 0.5**, baked in before quantization.

| File | Size | Goes in |
|---|---|---|
| `8step-unlocked-flashgen/wuli_h3_turbo_8step_v1_unlocked_flashgen050_comfyui_int8_convrot_hq.safetensors` | 25.0 GB | `models/diffusion_models/` |

- **How it was made:** the LoRA was added at 0.5 in fp32 to the unlocked bf16 weights: the 200 block linears, the 8 token-refiner linears, every adaLN projection (its 8 time-curve columns, zero-padded to the 25-column layout; columns 0–7 of the unlocked table are the base curve, so this is exact) and the adaLN biases. Then the HQ int8 recipe (see *HQ int8 files* below) (66 linears bf16, 134 int8 convrot with clipping-searched scales). The unlocked time table itself is unchanged.
- **Settings:** **8 steps, shift 12/3, `simple`, euler, CFG 1.** Do not also load the FlashGen LoRA.
- **Also works well at 10 steps** with euler + `simple` / `beta` / `beta_57`.
- **Tested by the repo owner, works well as-is.**
- **Quick download:**

```
hf download sandpies/Wuli-MiniMax-H3-Turbo-ComfyUI 8step-unlocked-flashgen/wuli_h3_turbo_8step_v1_unlocked_flashgen050_comfyui_int8_convrot_hq.safetensors --local-dir ComfyUI/models/diffusion_models
```

  (lands in `ComfyUI/models/diffusion_models/8step-unlocked-flashgen/`; ComfyUI lists subfolders.)

---

## 8-step hybrid: text-to-video + reference-to-video

One checkpoint for **both text-to-video and reference-to-video**. The base is the recommended **FL2VA unlocked + FlashGen 0.5**
merge (Wuli 8-step unlocked time table). The plain and Realism hybrids only graft Ref2VA's per-block adaLN into **blocks 25-49**.
**F2** goes further: it bakes in the full Ref2VA−FL2VA difference for blocks 25-49 (attention, MLP, norms, and adaLN), with the
Wuli delta re-added and gated, then FlashGen 0.5 on those blocks. Blocks 0-24, the time table and the final layer stay identical
to the FL2VA FlashGen base. AdaLN grafts follow the FL2VA-base hybrid method from [smhfacct](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models).

| File | Size | Goes in |
|---|---|---|
| `8step-hybrid/wuli_h3_turbo_8step_v1_flbase_refadaln_b25-49_flashgen050_int8_convrot_hq.safetensors` | 25.0 GB | `models/diffusion_models/` |
| `8step-hybrid/wuli_h3_turbo_8step_v1_flbase_refadaln_b25-49_flashgen050_realism_int8_convrot_hq.safetensors` | 25.0 GB | `models/diffusion_models/` |
| `8step-hybrid/wuli_h3_turbo_8step_v1_F2_flbase_ref2va-b25-49-full_gatedwuli_flashgen050_int8_convrot_hq.safetensors` | 25.0 GB | `models/diffusion_models/` |

- **Plain version** (`..._flashgen050_int8_convrot_hq`): nothing else baked in. Load the
  [Realism-People LoRA](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA) yourself if you want it.
- **Realism version** (`..._flashgen050_realism_int8_convrot_hq`): fal Realism-People baked in at **1.0** (attention, all blocks
  and the token refiner). Trigger word `r34l1sm` (optional). Don't also load the Realism-People LoRA.
- **F2 version** (`..._F2_flbase_ref2va-b25-49-full_gatedwuli_flashgen050_int8_convrot_hq`): the FL2VA unlocked + FlashGen 0.5
  base with the **full** Ref2VA−FL2VA difference grafted into blocks 25-49 (not a LoRA squeeze): attention qkv/out, MLP fc1/fc2,
  q/k norms, norm1/2, and adaLN (cols 0-7 rotated into the FL time basis). The Wuli 8-step delta is re-added inside those blocks,
  gated per tensor by `max(0, cos(FL2VA, Ref2VA))`; FlashGen 0.5 is applied after the graft. No ref LoRA. Same settings as the
  other 8-step hybrids. Don't also load the FlashGen or Ref2VA reference LoRA.
- **Settings:** tested at **euler, `simple`, 8 steps**, `ModelSamplingMiniMaxH3` at its defaults (**shift 12/3**), **CFG 1**.
  The time table is the unlocked one, so other turbo step counts should work like the other unlocked files, but 8 steps is what was tested.
- **Reference-to-video:** use the Ref2VA conditioning nodes, same as the other Ref2VA files. For plain text-to-video, just leave the reference out.
- Do not also load the FlashGen LoRA (already merged). Same HQ int8 recipe as the other HQ files.
- **Credits:** Wuli-Art (8-step distill, MiniMax-H3 Community License); FlashGen by Beidouqixing / Huawei Ascend (Apache-2.0), ComfyUI
  conversion by [Kijai](https://huggingface.co/Kijai/MiniMax-H3-experimental); fal Realism-People LoRA; hybrid method by smhfacct;
  Ref2VA/FL2VA via Comfy-Org.
- **Quick download:**

```
hf download sandpies/Wuli-MiniMax-H3-Turbo-ComfyUI 8step-hybrid/wuli_h3_turbo_8step_v1_flbase_refadaln_b25-49_flashgen050_int8_convrot_hq.safetensors --local-dir ComfyUI/models/diffusion_models
hf download sandpies/Wuli-MiniMax-H3-Turbo-ComfyUI 8step-hybrid/wuli_h3_turbo_8step_v1_flbase_refadaln_b25-49_flashgen050_realism_int8_convrot_hq.safetensors --local-dir ComfyUI/models/diffusion_models
hf download sandpies/Wuli-MiniMax-H3-Turbo-ComfyUI 8step-hybrid/wuli_h3_turbo_8step_v1_F2_flbase_ref2va-b25-49-full_gatedwuli_flashgen050_int8_convrot_hq.safetensors --local-dir ComfyUI/models/diffusion_models
```


---

## 8-step

Wuli's 8-step ships as a **full transformer** with its 8 timesteps baked into fixed per-step adaLN tables, built on stock
MiniMax-H3 FL2VA. CPU-verified: layout matches the Comfy-Org MiniMax-H3 checkpoints key for key, ComfyUI detects the model
config, transformer weights equal Wuli's source, sampler timesteps match Wuli's to ~1e-7.

### Files (`8step/`)

| File | Size | Goes in |
|---|---|---|
| `wuli_h3_turbo_8step_v1_comfyui_int8_convrot.safetensors` | 21.4 GB | `models/diffusion_models/` |
| `wuli_h3_turbo_8step_v1_comfyui_bf16.safetensors` | 40.6 GB | `models/diffusion_models/` |
| `wuli_h3_turbo_8step_v1_lora_r256_comfyui.safetensors` | 2.6 GB | `models/loras/` |

- **`..._comfyui_bf16`** — Wuli's 8-step transformer in the ComfyUI MiniMax-H3 layout (532 keys, bf16). The fixed per-step
  adaLN tables are packed into a 25-column `adaln_t_table` (cols 0-7 = stock base curve, cols 8-24 = Wuli's 8 video steps,
  the visual-conditioning step t=0.999, and 8 audio steps).
- **`..._comfyui_int8_convrot`** — the same model with the 200 block linears quantized to int8 (tensorwise + convrot, group
  size 256), 932 keys, matching the Comfy-Org `pruned_int8_convrot` layout. Dequant error is the same as the Comfy-Org base
  int8 (~0.9% rel.). Same VRAM as any MiniMax-H3 int8.
- **`..._lora_r256_comfyui`** — a rank-256 LoRA extracted from (Wuli 8-step − stock FL2VA), plus a fitted adaLN time-table
  approximation, for other MiniMax-H3 bases. **Approximate:** Wuli's update is spread thinly over almost every weight, so
  rank 256 captures only **about half** of it; on another base you get *part of the way* to Wuli's 8-step. All 352 patches
  load on Singularity v1.3. For the real thing use the int8/bf16 checkpoint.

### Usage — full / int8 checkpoint
1. **Load Diffusion Model** (`UNETLoader`) → `wuli_h3_turbo_8step_v1_comfyui_int8_convrot` (or `_bf16`). No turbo LoRA on top.
2. `ModelSamplingMiniMaxH3`: **shift_video 12, shift_audio 3** (the defaults).
3. Scheduler **simple**, **8 steps**, denoise 1, sampler **euler**, **CFG 1.0**.

### Usage — LoRA
1. Load your MiniMax-H3 base (e.g. Singularity v1.3 int8) → `LoraLoaderModelOnly` `wuli_h3_turbo_8step_v1_lora_r256_comfyui` at **1.0**.
2. Same sampler settings (12/3, simple, 8 steps, euler, CFG 1).

### 8-step sigmas
`ModelSamplingMiniMaxH3` 12/3 + `simple` + 8 steps gives exactly Wuli's schedule. If you prefer `ManualSigmas`:

```
1.0, 0.988235, 0.972973, 0.952381, 0.923077, 0.878049, 0.8, 0.631579, 0.0
```

(audio sigmas are derived from the video sigma by ComfyUI with shift 3.)

### 8-step caveats
- **On-schedule only.** The checkpoint carries fixed per-step adaLN tables for Wuli's 8 timesteps; other step counts, shifts,
  schedulers or multi-stage samplers (res_2s, dpmpp, ...) evaluate off those tables. For other schedules use the
  **unlocked** files (below).
- **Time-embedding LoRAs (DMAD, PDD-style turbo/step LoRAs) don't fit the full/int8 checkpoint** — its `adaln_t_table` is
  **25 columns** wide instead of the stock 8. Plain style/character LoRAs stack fine.
- **Don't stack the 8-step LoRA on the 8-step checkpoint** — it is already in there.
- Guidance-distilled: CFG 1, exactly 8 model calls.
- Wuli trained on the FL2VA trunk; on ref2va-derived bases (e.g. Singularity) the LoRA is off-label — test it, especially audio.

---

## 8-step Ref2VA (experimental)

Wuli distilled the 8-step from the **FL2VA** model. Ref2VA (the reference-image model) differs from FL2VA by about 3% of each
weight — far more than the distill itself changed (~0.2%) — so the plain 8-step checkpoint only behaves like FL2VA with
reference images. These files move Wuli's change onto Ref2VA instead:

**W = Ref2VA + (Wuli 8-step − FL2VA)**, for every tensor (fp32 math, then cast to the stored dtype). Ref2VA and FL2VA are the
Comfy-Org `minimax_h3_{ref2va,fl2va}_pruned_bf16` checkpoints. The 25-column time table keeps Ref2VA's own 8 base-curve
columns plus Wuli's 17 per-step columns unchanged, so at each of Wuli's 8 timesteps the adaLN output is exactly
Ref2VA's + (Wuli's − FL2VA's), and off those timesteps it is plain Ref2VA.

| File | Size | Goes in |
|---|---|---|
| `wuli_h3_turbo_8step_v1_ref2va_comfyui_int8_convrot.safetensors` | 21.4 GB | `models/diffusion_models/` |
| `wuli_h3_turbo_8step_v1_ref2va_comfyui_bf16.safetensors` | 40.6 GB | `models/diffusion_models/` |

Same layout as the 8-step files (532 / 932 keys, 25-column `adaln_t_table`, same int8 convrot recipe); ComfyUI detects it as MiniMax-H3.

### Usage
1. **Load Diffusion Model** (`UNETLoader`) → `wuli_h3_turbo_8step_v1_ref2va_comfyui_int8_convrot` (or `_bf16`).
2. Your usual **Ref2VA conditioning nodes** (reference image(s) into the MiniMax-H3 image-to-video / Ref2VA conditioning).
3. Same as the 8-step: `ModelSamplingMiniMaxH3` **shift 12/3**, scheduler **simple**, **8 steps**, sampler **euler**, **CFG 1.0**.

### Caveats
- **Untested in generation** (CPU-verified weights only). It's a weight-space merge: it assumes Wuli's change carries over
  from FL2VA to Ref2VA, which is plausible since the two share the base model, but nobody trained this combination.
- Same 8-step caveats as above (on-schedule only, no time-embedding LoRAs, style LoRAs fine, CFG 1).
- Don't stack it with the 8-step LoRA or other turbo adapters.

---

## 8-step unlocked (6–12 steps, any scheduler / sampler)

**Same model, smoothed time table.** These are the 8-step and 8-step Ref2VA int8 checkpoints with every transformer weight
byte-for-byte identical. Only two small time-embedding tensors changed (`adaln_t_table` and `final_layer.adaln_proj.linear.weight`,
about 1.2 MB in total).

In the original files, Wuli's per-step adaLN adjustments only switch on at the exact 8 noise levels of the 12/3 `simple` 8-step
schedule. At any other noise level the model falls back to plain base modulation, so for example on 10 steps `simple` only about
20% of Wuli's time conditioning is actually applied. The unlocked files interpolate Wuli's adjustments smoothly between those
8 points, so any step count, scheduler or sampler gets Wuli's modulation for the noise level it is at.

| File | Size | Goes in |
|---|---|---|
| `8step-unlocked/wuli_h3_turbo_8step_v1_unlocked_comfyui_int8_convrot.safetensors` | 21.4 GB | `models/diffusion_models/` |
| `8step-ref2va-unlocked/wuli_h3_turbo_8step_v1_ref2va_unlocked_comfyui_int8_convrot.safetensors` (experimental Ref2VA merge) | 21.4 GB | `models/diffusion_models/` |
| `8step-unlocked/wuli_h3_turbo_8step_v1_unlocked_comfyui_int8_convrot_hq.safetensors` (HQ int8) | 25.0 GB | `models/diffusion_models/` |
| `8step-ref2va-unlocked/wuli_h3_turbo_8step_v1_ref2va_unlocked_comfyui_int8_convrot_hq.safetensors` (HQ int8, experimental Ref2VA merge) | 25.0 GB | `models/diffusion_models/` |

### HQ int8 files (`..._int8_convrot_hq`)
Same unlocked models, quantized more carefully (the same idea as the community "skip edges" / HQ int8 quants):
- **Kept in bf16 (66 of the 200 block linears):** all four linears of blocks 0, 1, 48 and 49 (the input/output "edges"), `attn.out_proj` in all 50 blocks, and `mlp.fc2` in blocks 29, 39, 44, 45 and 49.
- **The other 134** use the same stock `int8_tensorwise` convrot format (Hadamard group 256, per-row scales), but each row's scale is picked by a clipping search (absmax × 1.00…0.80, lowest error) instead of plain absmax. That trims the int8 error a little (about 0.94% → 0.91% relative on fc2).
- Size 25.0 GB vs 21.4 GB for the regular int8, so plan for **24 GB+ VRAM** or some offloading. Stock ComfyUI >= 0.38 loads the mixed bf16 + int8 layers with no extra node. Every non-linear tensor (time table, adaLN, norms, embedders) is byte-identical to the regular unlocked file, so settings are unchanged.

### Usage
1. **Load Diffusion Model** (`UNETLoader`) → the unlocked file. No node needed; stock ComfyUI >= 0.38 loads it like the 8-step.
2. `ModelSamplingMiniMaxH3`: **shift 12/3 recommended**. You can adjust it (about 8–14 for video); it just samples the same curve at
   different noise levels.
3. **6–12 steps** work best. Scheduler `simple`, `beta` or `sgm_uniform`; sampler **euler**, **dpmpp_2m** or **res_multistep**; **CFG 1.0**.
4. `simple` + 8 steps + 12/3 is still Wuli's original schedule and gives the same result as the original 8-step file.

### What changed
- **Transformer blocks:** Wuli's per-step adjustments are interpolated linearly between his 8 steps, measured on the unshifted
  schedule grid, where the 8 steps are evenly spaced. Video (shift 12) and audio (shift 3) are interpolated separately. At Wuli's
  exact timesteps the values are held flat, so on the original schedule the block modulation is **bit-identical** to the 8-step
  file (verified on all 50 blocks for every video, audio, conditioning-frame and reference-audio lookup).
- **Low noise (below Wuli's last sigma, 0.63):** the video modulation blends from Wuli's last-step values to Wuli's own
  clean-frame (visual-conditioning) values. Audio and text taper back to the base model at sigma 0, so reference-audio and
  conditioning frames behave exactly as before.
- **Final layer:** video and audio share one modulation slot, so it follows a single curve through all 16 of Wuli's video and
  audio points. On the original schedule it matches to within 1e-7 relative.

### Caveats
- **Untested in generation** (CPU-verified only). The trunk was distilled for 8 big jumps, so 6–12 steps should stay closest
  to the 8-step's behaviour; 4–5 or 16+ steps, or schedulers that pile many steps at very low noise (`karras`, `exponential`), are further from
  what it was trained for and may soften or overcook.
- Wuli's last step (the jump from sigma 0.63 to 0) has much larger adjustments than the others. With more steps, that last-step
  behaviour is spread over the low-noise steps by the interpolation.
- In the final layer, between about sigma 0.64 and 0.3, the curve is a compromise between Wuli's video and audio values.
- Everything else from the 8-step caveats still applies: CFG 1, no time-embedding LoRAs (25-column table), style LoRAs are fine,
  and don't stack turbo LoRAs on top.

---

## 8-step Ref2VA unlocked × FlashGen × Realism-People (experimental)

**Base is Ref2VA** (the `8step-ref2va-unlocked` model: Ref2VA + (Wuli 8-step − FL2VA), unlocked Ref2VA time table) with two LoRAs baked in before quantization:

- the **FlashGen 4-step LoRA** at **strength 0.5** (FlashGen by Beidouqixing / Huawei Ascend, Apache-2.0; ComfyUI conversion by [Kijai](https://huggingface.co/Kijai/MiniMax-H3-experimental), `loras/minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_pruned_avg_rank_13_bf16.safetensors`)
- the **Realism-People LoRA** at **strength 1.0** ([fal/MiniMax-H3-Realism-People-LoRA](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA), `h3-realism-people-t2v-i2v-r2v.safetensors`, MiniMax-H3 Community License)

| File | Size | Goes in |
|---|---|---|
| `8step-ref2va-unlocked-flashgen-realism/wuli_h3_turbo_8step_v1_ref2va_unlocked_flashgen050_realism_comfyui_int8_convrot_hq.safetensors` | 25.0 GB | `models/diffusion_models/` |

- **Use with the Ref2VA conditioning nodes** (reference image(s)), same as the other Ref2VA files.
- **Settings:** 8 steps, shift 12/3, `simple`, euler, CFG 1 (the unlocked table also allows 6–12 steps / other samplers). Do not also load the FlashGen or Realism-People LoRAs. The Realism-People trigger word `r34l1sm` at the start of the prompt is optional.
- **Also works well at 10 steps** with euler + `simple` / `beta` / `beta_57`.
- **How it was made:** both LoRA deltas summed in fp32 onto the bf16 weights (FlashGen: 200 block linears, 8 token-refiner linears, adaLN projections and biases; Realism-People: attention qkv/out projections of all blocks and the token refiner). FlashGen was trained against FL2VA, whose 8 adaLN time-curve columns differ from Ref2VA's, so its adaLN delta was re-expressed in the Ref2VA curve basis (least-squares 8×8 map, fit error 0.005%) before zero-padding to the 25-column layout. Then the same HQ int8 recipe as above (66 linears bf16, 134 int8 convrot with clipping-searched scales).
- **Untested in generation.** FlashGen was distilled on FL2VA and is applied here to a Ref2VA base; treat it as an experiment.

---

## 4-step

The original file is not a standalone model. It holds:
- a shared **rank-128 LoRA** (alpha 128) on all 50 transformer blocks plus the 2 token-refiner blocks
  (`to_q`, `to_k`, `to_v`, `to_out.0`, `ff.net.0.proj`, `ff.net.2`), in diffusers/peft naming, and
- **4 per-step output heads** per modality (`pdd.video.0-3`, `pdd.audio.0-3`) that replace `proj_out` /
  `audio_proj_out` at step *i*, with `head_i = pdd_i / span_i` on a 256-cell grid shifted by 6 (video) / 3 (audio).

Here the LoRA keys are renamed to ComfyUI naming and the heads are repacked as a ComfyUI MiniMax-H3 **head bank**, so core
ComfyUI swaps the head per step natively. CPU-verified against the math of Wuli's `run_fl2va.py` through stock ComfyUI 0.38
code paths (`LoraLoaderModelOnly`'s loader, ModelPatcher, `simple` scheduler, the MiniMax-H3 final layer): per-step outputs
match to about 1e-8, with the correct head selected at every step.

### Which 4-step files?
- **Recommended, exact on any base (FL2VA, Singularity, ref2va, finetunes), strength-proof:** three `LoraLoaderModelOnly`
  in this order: `wuli_h3_turbo_4step_v1_trunk_comfyui_bf16` (trunk, any strength, 1.0 = trained) →
  `wuli_h3_turbo_4step_v1_heads_widen` → `wuli_h3_turbo_4step_v1_heads_set` (both: any strength, it is ignored).
- **Quick start, one file (exact on FL2VA only):** `wuli_h3_turbo_4step_v1_full_fl2varel_comfyui.safetensors`, strength **1.0**.
- **Exact on Singularity v1.3 (two files):** `..._trunk_comfyui_bf16` + `..._pddbank_singularityv13rel`, both at **1.0**.

### Files (`4step/`)

| File | Size |
|---|---|
| `wuli_h3_turbo_4step_v1_trunk_comfyui_bf16.safetensors` | 1.96 GB |
| `wuli_h3_turbo_4step_v1_heads_widen.safetensors` | 49 KB |
| `wuli_h3_turbo_4step_v1_heads_set.safetensors` | 11.0 MB |
| `wuli_h3_turbo_4step_v1_full_fl2varel_comfyui.safetensors` | 1.97 GB |
| `wuli_h3_turbo_4step_v1_pddbank_fl2varel.safetensors` | 11.7 MB |
| `wuli_h3_turbo_4step_v1_pddbank_singularityv13rel.safetensors` | 11.7 MB |
| `wuli_h3_turbo_4step_v1_heads_abs.safetensors` | 11.0 MB |
| `wuli_h3_turbo_4step_v1_singlehead_mean_fl2varel.safetensors` | 2.8 MB |
| `wuli_h3_turbo_4step_v1_singlehead_mean_singularityv13rel.safetensors` | 2.8 MB |
| `ComfyUI-Wuli-H3-Turbo.zip` (node v1.2.0, optional) | 12.4 MB |

All `.safetensors` files go in `ComfyUI/models/loras/`; the zip goes in `ComfyUI/custom_nodes/`.

- **`trunk_comfyui_bf16`** — trunk LoRA only (rank 128, bf16, 50 blocks + 2 token-refiner blocks). Pair it with
  `heads_widen` + `heads_set` (exact on any base) or with a `pddbank_*` file.
- **`heads_widen`** — a zero LoRA with ComfyUI's `reshape_weight`: it only widens `final_layer.video_out/audio_out` to the
  [4x96 | 4x32] head-bank shape that core 0.38 detects as per-step heads. On its own it changes nothing. Strength has no effect.
- **`heads_set`** — the absolute 4-step head bank as core `set_weight` patches (row block 0 = head 0, blocks 1-3 = head_k − head 0).
  It overwrites the widened heads, so the result does not depend on the base model's own head. Strength is ignored.
  **Must be loaded after `heads_widen`** (the other order fails with a shape error at sampling time).
- **`full_fl2varel_comfyui`** — trunk LoRA + 4-step head bank in one file, heads relative to the official FL2VA output head.
  One `LoraLoaderModelOnly` at strength 1.0.
- **`pddbank_fl2varel` / `pddbank_singularityv13rel`** — head bank only, relative to the FL2VA / Singularity v1.3 output head.
  Load at 1.0 after the trunk.
- **`heads_abs`** — the 4 video + 4 audio heads as absolute weights (`head_i = pdd_i / span_i`). Reference/tooling only (not a LoRA).
- **`singlehead_mean_*`** — approximation: one averaged head as a plain same-shape patch. Not exact.
- **`ComfyUI-Wuli-H3-Turbo.zip`** — optional, deprecated convenience node v1.2.0 (loads the original Wuli file directly, sets
  shifts/sigmas, grid guard; loads through ComfyUI's own LoRA/ModelPatcher path), plus the converter, the
  `heads_widen`/`heads_set` builder and CPU tests. Not needed for any file in this repo.

### Usage A — stock nodes only (recommended)
1. `UNETLoader` (any MiniMax-H3 base) → [style LoRAs] → `LoraLoaderModelOnly` `wuli_h3_turbo_4step_v1_trunk_comfyui_bf16`
   (strength 1.0 = trained; lower is allowed) → `LoraLoaderModelOnly` `wuli_h3_turbo_4step_v1_heads_widen` →
   `LoraLoaderModelOnly` `wuli_h3_turbo_4step_v1_heads_set`. Keep this order.
2. `ModelSamplingMiniMaxH3`: **shift_video 6, shift_audio 3** (the default 12 is wrong for the 4-step).
3. **4 steps**, scheduler **simple**, sampler **euler**, **CFG 1.0** (KSampler, or BasicScheduler + BasicGuider + SamplerCustomAdvanced).

Core then picks Wuli's head *i* at step *i* by itself (no custom node, no metadata: the bank is detected from the head's shape
and read on the 6/3 shifts with the sampler's sigmas).

Alternatives with fewer files (heads stored relative to one base, so **strength must be exactly 1.0**):
`..._full_fl2varel_comfyui` alone (exact on FL2VA; ~7e-4 off on Singularity v1.3), or
`..._trunk_comfyui_bf16` then `..._pddbank_singularityv13rel` (exact on Singularity v1.3).

### Usage B — the optional node (deprecated)
Only needed to load Wuli's original peft-format file without converting it. Unzip `4step/ComfyUI-Wuli-H3-Turbo.zip` into
`ComfyUI/custom_nodes/`. `UNETLoader → [style LoRAs] → Wuli MiniMax-H3 Turbo 4-step (Apply) → BasicGuider (CFG 1) →
SamplerCustomAdvanced`, KSamplerSelect **euler**, node **sigmas** wired in. It produces the same patched heads as Usage A,
bit for bit, and sets shifts 6/3 itself (don't add `ModelSamplingMiniMaxH3` after it).

### 4-step sigmas
- video: `1.0, 0.947368, 0.857143, 0.666667, 0`
- audio: `1.0, 0.9, 0.75, 0.5, 0` (ComfyUI derives these from the video sigma)

### 4-step caveats
- Wuli trained on the **FL2VA** trunk. On ref2va-derived bases (e.g. Singularity) the trunk LoRA is off-label; test it,
  especially audio.
- Do **not** stack with other step-distillation / turbo adapters or step caches (DMAD, LongLive, PDD-Acc, LightX2V/turbo
  LoRAs, FastH3/VSA or other turbo checkpoints — including the 8-step above —, TeaCache/FirstBlockCache-style caches).
  Style LoRAs are fine.
- **euler only.** Multi-stage samplers (res_2s, dpmpp, ...) evaluate off the 4 trained sigmas.
- Guidance-distilled: CFG 1, exactly 4 model calls.

---

<details>
<summary>SHA256 checksums (also in <code>SHA256SUMS.txt</code>)</summary>

```
e55bf78f3c0cb20fc9a4eb2d305c1b38594002d7b30e36057c2a1458528cf73b  4step/wuli_h3_turbo_4step_v1_full_fl2varel_comfyui.safetensors
67def1bcf529acc9ec8c7b0ac95812edacc3658f3a82d84591a1440dc88cc4a4  4step/wuli_h3_turbo_4step_v1_heads_abs.safetensors
ddeea056b1c490bcf0b344c8326ebb38407532dfee701a8f1eb01786fef65ff4  4step/wuli_h3_turbo_4step_v1_pddbank_fl2varel.safetensors
fb5dbd5ee3913ad5936efb7e65410358afd98013524f130c6dd0dabd74edaaa2  4step/wuli_h3_turbo_4step_v1_pddbank_singularityv13rel.safetensors
66989df484dc416fa3a79f35f297e6b6d946fa0e49cbb32a2ce0d119d6000757  4step/wuli_h3_turbo_4step_v1_singlehead_mean_fl2varel.safetensors
4f91f12693cbe916fcb28f433895850bb626b40b4a536a3ba3cf109da974360f  4step/wuli_h3_turbo_4step_v1_singlehead_mean_singularityv13rel.safetensors
ee16350bd02f46627a15a64ebd37b31f8547dd507855141c9c446a13adead8a1  4step/wuli_h3_turbo_4step_v1_trunk_comfyui_bf16.safetensors
eb643e48b5849523ffdbd365977926e6fa9b2e4c8a564f034f665560edb30805  4step/wuli_h3_turbo_4step_v1_heads_widen.safetensors
36af67e8b334f89458234b29f80e1b23f7ffba44432c0450af03ee7780b60aa7  4step/wuli_h3_turbo_4step_v1_heads_set.safetensors
e0f029741c776c81da8413ade15bfac00333d792431d90f52507cdb652c73da4  4step/ComfyUI-Wuli-H3-Turbo.zip
b15df144d7896f4f13fadb423187c2451a1fa1a5cebde54c07aa9c434d7514af  8step/wuli_h3_turbo_8step_v1_comfyui_bf16.safetensors
6d81c5d4975b47de69e6854cb2c9664ac875beed25ce2ebe0f65b7081d0c737b  8step/wuli_h3_turbo_8step_v1_comfyui_int8_convrot.safetensors
c85844102bd76cb45e5de8bc28c8c4e933f08c87ef4fdc168e0e64ceacbf3fb3  8step/wuli_h3_turbo_8step_v1_lora_r256_comfyui.safetensors
f87f0579415efadf453187f90165c3a0f4c58d544996eb0f5dabaf1a6a91dbe4  8step-ref2va/wuli_h3_turbo_8step_v1_ref2va_comfyui_bf16.safetensors
cf2c1d6c090e7bd66f8f98bcdab18c79eef339c1dfff20505d379e0d55124de5  8step-ref2va/wuli_h3_turbo_8step_v1_ref2va_comfyui_int8_convrot.safetensors
684b28b41d52dee8d300cc4529eed8aac5dacb29e8c805ab9c58cc0b50ab4d47  8step-unlocked/wuli_h3_turbo_8step_v1_unlocked_comfyui_int8_convrot.safetensors
c7ba8c745f4da024b9016f61a60531d4c55522ef384044d0c8636942b9edca38  8step-ref2va-unlocked/wuli_h3_turbo_8step_v1_ref2va_unlocked_comfyui_int8_convrot.safetensors
94b2a2600ae8b2c0332edc02d30126c98a3c4139d37e4268e6d079e7c89b7fae  8step-unlocked/wuli_h3_turbo_8step_v1_unlocked_comfyui_int8_convrot_hq.safetensors
0b79089d2ab030d843c7308e7141603aa212c0aca9e891255fb79124d2c86c7b  8step-ref2va-unlocked/wuli_h3_turbo_8step_v1_ref2va_unlocked_comfyui_int8_convrot_hq.safetensors
e9e9a2e0cc3dc6fd36528ac5d17c33bfbf9b726b38119c99c7e2340099cb77bf  8step-unlocked-flashgen/wuli_h3_turbo_8step_v1_unlocked_flashgen050_comfyui_int8_convrot_hq.safetensors
61e2c804e52046513dc0c692143915a4e0e712aa82eb51bf6b82d7a5e401f07e  8step-ref2va-unlocked-flashgen-realism/wuli_h3_turbo_8step_v1_ref2va_unlocked_flashgen050_realism_comfyui_int8_convrot_hq.safetensors
a852255134e20112d63a23848be122bd0076e37b9ddac09e436d9d61abbba42b  8step-hybrid/wuli_h3_turbo_8step_v1_flbase_refadaln_b25-49_flashgen050_int8_convrot_hq.safetensors
1092174db3985778cd5b511327a9d805c72e4d25a08c7ee3bb3c4a6aebac2566  8step-hybrid/wuli_h3_turbo_8step_v1_flbase_refadaln_b25-49_flashgen050_realism_int8_convrot_hq.safetensors
2629be33712574ff5ad80ce6da3032ba711546b897366c251510fdcd4a7837db  8step-hybrid/wuli_h3_turbo_8step_v1_F2_flbase_ref2va-b25-49-full_gatedwuli_flashgen050_int8_convrot_hq.safetensors
```

</details>

## License

These are **modified** MiniMax H3 Works, distributed under the
**MiniMax H3 Community License Agreement** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) (sources and modification notice).

> MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

Your use is bound by that license, including:
- **Territory:** the license does not cover use in the European Union, the United Kingdom, the Republic of Korea or the
  United States of America (the "Excluded Territories"). Do not download or use these files there.
- **Acceptable Use Policy** (Exhibit A of the license), e.g. no harm to minors, no impersonation without consent,
  disclosure of machine-generated content published publicly, no military use.
- No use of the weights or outputs to improve other (non-MiniMax-H3) AI models.
- Commercial products with > US$20M yearly revenue need written authorization from MiniMax; commercial products must
  display "MiniMax H3".

By downloading or using these files you agree to the MiniMax H3 Community License Agreement and its use restrictions,
and you must pass the same restrictions on to anyone you redistribute them to.

Credits: Wuli-Art / the Alibaba Wuli Team (models), MiniMax (MiniMax H3). Powered by MiniMax H3.
