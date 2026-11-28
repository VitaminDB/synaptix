# cgc-0521/minimax-h3-fused-turbo-int8-convrot

Источник: https://huggingface.co/cgc-0521/minimax-h3-fused-turbo-int8-convrot · скачано 2026-11-28
Загрузок: 854 · лайков: 2 · базовая: ["xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI","diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024","Comfy-Org/MiniMax-H3","MiniMaxAI/MiniMax-H3"] · теги: comfyui,turbo,base_model:Comfy-Org/MiniMax-H3,base_model:merge:Comfy-Org/MiniMax-H3,base_model:xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI,base_model:merge:xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI · обновлено: 2026-09-18T10:20:51.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model:
  - xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI
  - diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024
  - Comfy-Org/MiniMax-H3
  - MiniMaxAI/MiniMax-H3
base_model_relation: merge
library_name: diffusion-single-file
pipeline_tag: image-text-to-video
tags:
  - minimax-h3
  - comfyui
  - int8
  - convrot
  - turbo
  - merge
  - synchronized-audio-video
---

# MiniMax-H3 Fused Turbo (INT8 ConvRot)

One 21 GB ComfyUI diffusion-model file that does MiniMax-H3 text/image-to-video **and** reference-to-video in
**4 steps**, with a distilled turbo and a motion-smoothing style LoRA already folded into the weights.

- **Base:** the pruned `fl2va` transformer with a rank-1024 SVD of the (`ref2va` - `fl2va`) weight delta fused in,
  so a single partition serves both first/last-frame and reference conditioning
  (`xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI`, ComfyUI conversion of
  `diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024`).
- **Merged in:** lightx2v FL2VA Turbo **8-step v1.0** at 1.0 (rank-24 resize from `Kijai/MiniMax-H3_comfy`) and
  **Mystic v2.0** at 0.7 (Civitai model 2856467, merged for motion smoothness).
- **Quantization:** ONE pass after the merge. All four heavy Linear weights in each of the 50 blocks
  (`qkv_proj`, `out_proj`, `fc1`, `fc2`) are INT8 with ConvRot (group 256, per-channel) in ComfyUI's native
  `comfy_quant` layout; everything else stays BF16/F32. No custom loader: `UNETLoader`, weight_dtype `default`.

`diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` -> `ComfyUI/models/diffusion_models/`

**Hardware.** The timings below are from a 96 GB card with everything resident. Smaller cards work too: ComfyUI's model
management loads and offloads the transformer, text encoder and VAEs as they are needed, so the same graphs run, just
slower, and the 21 GB file is the smallest footprint this recipe has. **AMD / ROCm:** the recipe was developed on an
AMD box; use patientx's ComfyUI ROCm fork, <https://github.com/patientx-cfz/comfyui-rocm>, which also carries a ROCm
build of the SLA attention node.

## Results and timings

Same prompt, same seed 42, 1152x640 x 243 frames (10 s), SLA sparse attention, one RTX PRO 6000 (96 GB). Only the
pipeline differs; every one of them produces the same take.

| pipeline | steps | wall | preview |
|---|---|---|---|
| **reference: single pass** | **4** | **76 s** | <video src="https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/resolve/main/previews/01_reference_4step_sla.mp4" controls width="420"></video> |
| **4 + 4 de-rope** (Motion Lab: pass 1 = reference, jerk oracle -> time smear -> seeded V2V -> pass 2 -> exact recover) | 4 + 4 | **~374 s** cold: ~65 s pass 1, ~65 s smear + VAE encode of the 685 dilated frames, ~240 s pass 2 + decode + recover | <video src="https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/resolve/main/previews/02_derope_4plus4_sla.mp4" controls width="420"></video> |
| **4 + 4 de-rope + DyRoPE** (`physical_blocks` 30-49) **+ warm375 adapter** @1.0 on pass 2 | 4 + 4 | ~374 s cold (DyRoPE and the adapter add no measurable time) | <video src="https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/resolve/main/previews/03_derope_dyrope_pb3049_warm375.mp4" controls width="420"></video> |
| single pass | 6 | 80 s | <video src="https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/resolve/main/previews/04_reference_6step_sla.mp4" controls width="420"></video> |
| single pass | 8 | 103 s | <video src="https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/resolve/main/previews/05_reference_8step_sla.mp4" controls width="420"></video> |
| single pass | 25 | 292 s | <video src="https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/resolve/main/previews/06_reference_25step_sla.mp4" controls width="420"></video> |

More steps sharpen the drawing a little and change nothing about the soundtrack. The de-rope is the reason to run
this at 4 steps: pass 2 only spends steps on the frames the jerk oracle flagged, so the two passes together cost about
the same as one 25-step pass, and the held regions come out cleaner.

## Don't want the bake? The live-LoRA path is equivalent

The same result comes from the public pieces stacked live in ComfyUI - measured on the same seed at 8 steps with SLA:
same wall time (103 s), the same take, audio meters identical within noise. Load:

1. base: `MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors` from
   [xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI](https://huggingface.co/xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI)
2. `LoraLoaderModelOnly`: `minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors` @ **1.0** from
   [Kijai/MiniMax-H3_comfy](https://huggingface.co/Kijai/MiniMax-H3_comfy/tree/main/loras) (full-rank original:
   [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo))
3. `LoraLoaderModelOnly`: Mystic v2.0 @ **0.7** from [Civitai model 2856467](https://civitai.com/models/2856467)
4. then the SLA node, then the shift node, then the samplers - exactly the order in the shipped workflows.

This file saves you the two loaders, one quantisation round-trip, and **about 21 GB of VRAM**: ComfyUI keeps an
unpatch backup of every LoRA-patched weight, and with 200 patched layers on a 21 GB INT8 model that backup is another
~21 GB (measured twice at 8 steps: 47.8 GB peak / 42 GB resident baked vs 68.9 GB peak / 64 GB resident live, same
wall time). Off `--gpu-only` that backup lands in host RAM instead.

### Why merge instead of stacking the LoRAs

The two LoRA files total ~540 MB, so the 21 GB is not the adapters. A LoRA is a low-rank *delta* that ComfyUI adds to
each target weight at load time; to be able to take it off again (swap strength, unload) it keeps a pristine copy of
every weight it touched. Patch all 200 core layers of an INT8 model and the "pristine copy" is the whole model again.
A merge does the addition once, offline, and writes the result as ordinary weights: nothing to unpatch, nothing to
back up, no per-layer patch pass at load, and - done in this order - a single quantisation of the *merged* weights
rather than a quantised base with float deltas patched onto it. You give up the ability to turn the LoRAs down;
if you need that, use the live path above. (The usual warning that folding a distillation LoRA loses precision
applies to bf16 folds; merging before the INT8 pass measured identical to live here.)

## Steps: it says 8, run it at 4

The merged turbo is lightx2v's *8-step* LoRA, but it is a 4-or-8 NFE model: **run it at 4 steps**, especially in
the de-rope (4 + 4). See the table at the top for the 4 / 6 / 8 / 25 ladder.

## The recipe that made the previews

| dial | value |
|---|---|
| sampler / scheduler | `res_multistep` / `simple`, denoise 1, `BasicGuider` (no CFG) |
| steps | **4** |
| shift | video 12 / audio 3 (`MiniMaxH3SigmaShift`, which is also the core default) |
| canvas | 16:9 at 0.7 MP (1152x640) |
| length | 10 s -> 243 frames (17k+5) |
| attention | **H3 SLA block-sparse, sparsity 0.90, block 64** (`ComfyUI-PlagueKind-Nodes-only-sparse`) |
| VAEs | video `minimax_h3_video_vae_int8_convrot`, audio `minimax_h3_audio_vae_fp32` |
| text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq` |

Two of these matter more than they look. Both were checked on the same seed with only that one setting changed:
**`res_multistep` beats `euler` for audio at 4 steps** (lightx2v's example graphs say euler), and
**SLA sparse attention is audible** - dense attention on the same weights gives a duller, flatter soundtrack, while SLA
restores the top-end detail (and runs ~40% faster). Dense works if you have no sparse pack; expect the audio to be
softer.

## Workflows (`workflows/`)

Each comes as a ComfyUI graph (`.json`, drag into the UI) and as an API-format graph (`.api.json`). One **PROMPT**
box at the top feeds every stage (pass 1 and pass 2 read the same text); it holds a placeholder, paste your own.

1. `01_reference_4step_sla` - the recipe above, single pass.
2. `02_derope_4plus4_sla` - the [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes) Motion Lab de-rope:
   pass 1 = the reference; jerk oracle -> time smear -> V2V init with the pass-1 audio seeded into the audio rows ->
   pass 2 on the same model (8-step schedule injected at 0.5, so 4 executed steps) -> exact recover with pass 2's foley.
3. `03_derope_dyrope_pb3049_warm375` - the same de-rope with **DyRoPE** (`physical_blocks`, blocks 30-49, the
   timing-fidelity preset) and the **step375 motion adapter** (`temporal_expansion_step375_r16`, "warm375" in the
   text below) at 1.0 on pass 2.
4. `04_i2v_fl2v_4step_sla` - image-to-video: a `LoadImage` on `first_frame`; wire a second one into `last_frame` for
   first/last-frame interpolation. Same recipe. (Tested here on the bake + SLA: 107 f at 1152x640 in 25 s.)
5. `05_ref2va_4step_sla` - reference-to-video with `MiniMaxH3ReferenceToVideo`: one image reference wired in, the node
   grows more image / video / audio slots as you connect them; the prompt box holds the reference-contract template
   (`<Subject 1>` from `<Picture 1>`). This is the reason the base is the fused single partition: the same file does
   ref2v without swapping transformers. (Tested here: mascot reference, 107 f, 34 s, identity held.)

Requires ComfyUI-MAINodes (the de-rope, DyRoPE and audio nodes) and the SLA node pack. The graphs name files at the
root of each models folder: `diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors`,
`vae/minimax_h3_video_vae_int8_convrot.safetensors` + `vae/minimax_h3_audio_vae_fp32.safetensors`,
`text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, and for `03` the adapter `loras/minimax_h3_temporal_expansion_step375_r16.safetensors`.
If you keep them in subfolders, re-pick them in the loader nodes.

**AMD / ROCm:** `workflows/rocm/` holds the same three graphs with the SLA node swapped to the ROCm build
(`H3SLAAttentionROCM`, the widget values from the original AMD graph: sparsity 0.9, block 64, min_seq 8192,
protect audio on). Use those on patientx's fork; the CUDA graphs will show a missing node there, and vice versa.

## Low VRAM (`workflows/lowvram/`)

The same five graphs with one extra node after the loader: KJNodes' `MiniMaxChunkFeedForward` (chunks 4, seq
threshold 4096), which splits the H3 feed-forward over the packed sequence so the big FFN intermediate and the fused
INT8 SwiGLU scratch never exist for all ~40k tokens at once. It is **exact**: same seed, the output is bit-identical to
the standard graph (PSNR infinite in our check). ROCm variants are in `workflows/lowvram/rocm/`.

What we measured on an honest 16 GB card (a 96 GB card with 79 GiB held by a balloon, 32 GB RAM cgroup, no
`--gpu-only`; ComfyUI streams the 21 GB model from RAM):

| graph | peak VRAM | wall |
|---|---|---|
| standard reference (4 steps, SLA) | 11.0 GiB | 65 s |
| lowvram reference (+ FFN chunker) | 10.9 GiB | 59 s |

So on a current ComfyUI the standard graphs already fit in 16 GB. If you still OOM inside `SamplerCustomAdvanced`
("Currently allocated 7.8 GiB, requested 6.6 GiB, free 0"), something else on the card is holding the difference
(a display, another process, an older core with the fc2-INT8 scratch path): use the lowvram set first, then 16:9 at
0.5 MP (960x544, the 8-step LoRA's own training size), then fewer frames, and for the de-rope `H3TimeSmear` dilation
4 -> 2 (pass 2 samples the dilated clip, 2-3x the tokens). We deliberately did NOT include KJNodes'
`MiniMaxLowVRAMAttention`: with SLA in the chain it changes the take (17 dB from the reference) and did not lower the
peak in this test.

## Motion adapters

The de-rope's pass 2 takes an optional motion adapter LoRA. The published ones are at
[MATLOWAI/MiniMax-H3-Motion-Adapter](https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter): the pilot, and the
`temporal_expansion/` ladder (`warm100_r16`, `step025_r16` ... `step375_r16`). "warm375" below is
`temporal_expansion_step375_r16`, "warm100" is `temporal_expansion_warm100_r16`. Two things to know before you reach for them on stylised content:

- they trade a little style for motion: on 2D-animation prompts they add a visible bias away from flat cel rendering
  toward photographic shading, and that bias grows with the adapter's step count (warm375 more than warm100);
- warm375 is the stronger stylist and the less disciplined one: on held regions it can duplicate a subject (a mirrored
  rat in the preview above). warm100 keeps the polish gain without that.

Without an adapter, `02_derope_4plus4_sla` is the safe default.

## Credits and links

- The recipe (this fused base + the 8-step turbo at 1.0 + Mystic v2 at 0.7, `res_multistep`, SLA sparse attention) is
  patientx's, uploaded here with their OK. It was the best turbo mix I had seen for animation, and it suits the
  de-rope's two-stage process well. Their ComfyUI ROCm fork: <https://github.com/patientx-cfz/comfyui-rocm>. What this
  repo adds is the measurements above and the single file.
- Fused base and the ComfyUI conversion: `xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI`, built on
  `diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024`.
- Turbo LoRA: `lightx2v/Minimax-h3-Turbo` (rank-24 resize from `Kijai/MiniMax-H3_comfy`).
- De-rope, DyRoPE and audio nodes: [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes).
- Motion adapters: [MATLOWAI/MiniMax-H3-Motion-Adapter](https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter).
- SLA sparse attention node: `ComfyUI-PlagueKind-Nodes-only-sparse` (CUDA build; a ROCm variant exists in the fork above).

## License

MiniMax H3 Community License Agreement (`LICENSE`), with the `NOTICE` this file is required to carry. Note the
license's territorial scope. The merged turbo LoRA is Apache-2.0 (`LICENSE-APACHE`); its attribution is in `NOTICE`.
The Mystic LoRA's Civitai permissions allow derivatives and relicensing.

Powered by MiniMax H3.
