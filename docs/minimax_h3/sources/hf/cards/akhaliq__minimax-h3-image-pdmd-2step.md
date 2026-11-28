# akhaliq/minimax-h3-image-pdmd-2step

Источник: https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step · скачано 2026-11-28
Загрузок: 48132 · лайков: 72 · базовая: MiniMaxAI/MiniMax-H3-Image · теги: lora,distilled,comfyui · обновлено: 2026-11-11T00:53:30.000Z

---

---
license: apache-2.0
base_model: MiniMaxAI/MiniMax-H3-Image
base_model_relation: adapter
library_name: diffusers
pipeline_tag: text-to-image
tags:
- text-to-image
- image-generation
- lora
- few-step
- distilled
- 2-step
- pdmd
- minimax-h3
- comfyui
- ml-intern
- arxiv:2609.35768
---

<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 0 2em"><a href="https://hf.co/chat?mode=ml-intern"><img src="https://hf.co/datasets/huggingface/badges/resolve/main/built-with-ml-intern-lg.svg" alt="Built with ML Intern" style="margin:0"></a><a href="https://hf.co/chat?mode=ml-intern&prompt=Render%20my%20prompts%20with%20akhaliq%2Fminimax-h3-image-pdmd-2step%20at%202%20steps%20and%20compare%20against%20the%20stock%20baseline" style="display:inline-flex;align-items:center;gap:5px;padding:4px 12px;border:1.5px solid #eee;border-radius:16px;background:#f9fbfc;color:#c2410c;font-weight:600;font-size:14px;text-decoration:none">A/B it on your prompts<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M13.5 6H5a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-8.5m-10.5 6L21 3m0 0h-5m5 0v5"/></svg></a><a href="https://hf.co/chat?mode=ml-intern&prompt=Package%20my%20LoRA%20as%20a%20single-file%20repo%20with%20sample%20grids%20the%20way%20akhaliq%2Fminimax-h3-image-pdmd-2step%20is%20built" style="display:inline-flex;align-items:center;gap:5px;padding:4px 12px;border:1.5px solid #eee;border-radius:16px;background:#f9fbfc;color:#c2410c;font-weight:600;font-size:14px;text-decoration:none">Package your LoRA like this<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M13.5 6H5a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-8.5m-10.5 6L21 3m0 0h-5m5 0v5"/></svg></a></div>

# MiniMax-H3-Image pDMD 2-Step LoRA

A **true 2-step mode** for [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image):
the [pDMD 2-NFE LoRA](https://huggingface.co/pdmd2026/pdmd_2NFE_lora)
([arXiv 2609.35768](https://arxiv.org/abs/2609.35768), rank 128 over attention + FFN of all 50 blocks)
merged into the official 33.1B image transformer, at the model's native **4096×4096 (16 MP)** — no CFG,
prompts as written. The stock checkpoint is step-distilled for its fixed 6-step PDD schedule; this LoRA
lands a finished image in **2 denoising evaluations** on a plain flow-matching grid.

This repo ships the LoRA in two layouts:

- `lora_model_0.safetensors` (bf16, byte-identical to the upstream pDMD release) — **diffusers-layout
  keys** (`transformer.transformer_blocks.N.attn.to_q…`, split QKV). Merge this into any diffusers
  `MiniMaxH3Transformer3DModel` — P-H3-Image's, or the official checkpoint after the on-the-fly
  conversion in [`render_grid.py`](render_grid.py).
- `minimax_h3_pdmd_2nfe_comfyui.safetensors` — the **ComfyUI/native-layout** conversion (fused
  block-diagonal `qkv_proj`, SwiGLU halves swapped back to `[gate; value]`), by
  [Iwannapose](https://huggingface.co/Iwannapose/minimax_h3_pdmd_2nfe_comfyui), mirrored here for
  one-stop use. This is the file for ComfyUI and other native-naming stacks.
- `minimax_h3_image_pdmd_2step_bf16.safetensors` — the **one-file ComfyUI checkpoint**: Comfy-Org's
  pruned bf16 diffusion model with the pDMD LoRA pre-merged (208 deltas in fp32, base metadata
  preserved), verified loading and rendering in stock ComfyUI. No LoRA node needed.
- `transformer/` — the **official MiniMax-H3-Image transformer pre-converted to diffusers layout**
  (sharded bf16 `MiniMaxH3Transformer3DModel`), so the stack below loads with a plain
  `from_pretrained(..., subfolder="transformer")` and no on-the-fly conversion. The official image
  checkpoint has no audio weights; the unused audio-branch tensors are zeros.

**Try it: [akhaliq/minimax-h3-image-2step](https://huggingface.co/spaces/akhaliq/minimax-h3-image-2step)**
— **realtime ~1 MP previews as you type** (a 2-step frame lands in ~1–1.5 s, with an idle detail
polish when you pause), full renders and edits at 2/3/4 steps up to native 16 MP, detail pass built
in (ZeroGPU). Live previews are something the stock model cannot do: its distilled schedule is fixed
at 6 steps, and at 2 steps it is still mid-denoise — the LoRA is what makes 2-step framing fast
enough to feel realtime.

<video src="https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/realtime_showcase.mp4" poster="https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/realtime_showcase_poster.jpg" controls muted loop playsinline width="100%"></video>

*Realtime edit mode: a Mona Lisa source, the prompt typed live, one 2-step frame per chunk (~0.9 s
per frame on the Space's FP8 build) — rendered with the Space's own realtime pipeline
([`make_realtime_video.py`](make_realtime_video.py)).*

## Usage (ComfyUI)

Two ways in — both verified end-to-end on latest ComfyUI master (2026-11-10) with the official
[Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) repack: **zero
unmatched keys**, ~20 s per 2048² image on an H200 (bf16 DiT, int8 text encoder).

The headline in ComfyUI is the same as everywhere else: **2 denoising evaluations instead of 6**, so
prompt and seed iteration is 2–3× faster per render — near-interactive at 2–4 MP — and the stock
checkpoint cannot be driven below its fixed 6-step schedule at all.

**Option A — one file (easiest).** `minimax_h3_image_pdmd_2step_bf16.safetensors` (40 GB) is the
Comfy-Org pruned bf16 checkpoint **with the pDMD LoRA pre-merged** (208 deltas, fp32 math, base
metadata preserved — merge script: [`merge_comfy.py`](merge_comfy.py)):

1. Update ComfyUI to latest master. Download `minimax_h3_image_pdmd_2step_bf16.safetensors` into
   `ComfyUI/models/diffusion_models/`, plus a text encoder (`qwen3vl_32b` int8 or nvfp4) and the VAE
   from [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image).
2. **Wire the graph**: *Load Diffusion Model* → **MiniMaxH3SigmaShift** (**shift_video 12**,
   shift_audio 3) → *Basic Guider*. Text: *Load CLIP* (type **minimax**) → *CLIP Text Encode*
   (positive only; guidance-distilled). Latent: **EmptyHunyuanLatentVideo** at your size — the H3
   image latent is a 1-frame video latent; 4096×4096 is native.
3. **Sample**: *SamplerCustomAdvanced* — *RandomNoise* (your seed), *KSamplerSelect* **euler**,
   *BasicScheduler* scheduler **simple**, **steps 2**, denoise 1.0 → *VAE Decode* → *Save Image*.

No LoRA node, no strength to get wrong. **Ready-made workflow:
[`workflows/minimax-h3-image-pdmd-2step-t2i.json`](workflows/minimax-h3-image-pdmd-2step-t2i.json)** —
drop it on the canvas after step 1; steps 2–3 are already wired in.

**Option B — base + LoRA (2 GB).** Already have the Comfy-Org checkpoint? Keep it and add the LoRA:

```bash
hf download akhaliq/minimax-h3-image-pdmd-2step minimax_h3_pdmd_2nfe_comfyui.safetensors --local-dir ComfyUI/models/loras
```

Insert **LoraLoaderModelOnly** (strength **1.0** — any other value scales the distillation delta)
between *Load Diffusion Model* and the sigma-shift node; everything else as in Option A. **Ready-made
workflow: [`workflows/minimax-h3-image-pdmd-2step-lora-t2i.json`](workflows/minimax-h3-image-pdmd-2step-lora-t2i.json)**
— the Option A graph with the LoRA node already at 1.0. The LoRA path is also the only one for the
`int8_convrot`/`w4a8` quantized checkpoints — rotated/quantized weights cannot take a pre-merge.

### Detail pass in ComfyUI (verified)

The same lightweight recipe the Space runs, in stock ComfyUI, **no extra files** — no refiner LoRA
needed: half-size draft at 2 steps → *VAE Decode* → pixel upscale (**lanczos**) to the final size →
**VAE Encode** → a second sample at **3 steps with denoise = σ₀**. The start sigma is the knob, not
“low denoise”: **0.4–0.55** polishes while the draft holds; **0.7** adds more detail but can drift
or speckle on flat areas. **Ready-made workflow:
[`workflows/minimax-h3-image-pdmd-2step-detail-t2i.json`](workflows/minimax-h3-image-pdmd-2step-detail-t2i.json)**
(draft save included so you can compare stages).

[![ComfyUI detail pass comparison: draft, denoise 0.5, denoise 0.7, plain](assets/_comfyui_detail_verify.jpg)](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/_comfyui_detail_verify.jpg)

*The same recipe headlessly verified in stock ComfyUI: the half-size draft (left, upscaled) is as
soft as a plain full-size 2-step render (right); the detail pass at denoise 0.5–0.7 (middle) is where
the fur, rock and tower detail lands. Why the naive “upscale the latent + low denoise” route
overcooks: a latent-space upscale without re-encoding blurs, and an arbitrary small denoise redraws
texture badly on top of the blur — pixel-upscale + VAE-encode + re-noise at a controlled σ₀ is the
validated fix.*

## Samples — 2 steps

The same twelve prompts throughout — **4096×4096, 2 denoising steps, flow shift 12, seed 3**, prompts as
written, **~17.7 s per image** — so the grids compare tile for tile with each other and with the sister
card [akhaliq/p-h3-image-pdmd-2step](https://huggingface.co/akhaliq/p-h3-image-pdmd-2step) (same prompt
set, same recipe).

**Why the LoRA matters**: the stock transformer at 2 steps is still mid-denoise — compositions are
recognizable (it is a step-distilled model, so far from noise), but soft, low-contrast and unfinished.
The merged LoRA lands composition, texture and fine detail in the same two evaluations:

| Stock MiniMax-H3-Image, 2 steps | **+ pDMD 2-step LoRA** |
|:---:|:---:|
| [![stock MiniMax-H3-Image at 2 steps](assets/stock_2step_grid.jpg)](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/stock_2step_grid.jpg) | [![2-step pDMD sample grid](assets/pdmd_2step_grid.jpg)](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/pdmd_2step_grid.jpg) |

*Rendered bf16 with FlashAttention-3 (Hub kernels) on one H200 (peak 84 GB, ~17.7 s/image, all 24 images
in ~13 min) via [`render_grid.py`](render_grid.py).*

## Showcase — 2 steps, native 16 MP at every aspect ratio

Twelve new prompts, each rendered at an official ~16 MP size in the aspect ratio that fits the subject
(6272×2688 21:9 down to 3072×5472 9:16), same recipe — pDMD LoRA, 2 steps, shift 12, seed 3 — at
~8 s/image:

[![pDMD 2-step showcase across aspect ratios](assets/pdmd_2step_showcase_grid.jpg)](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/pdmd_2step_showcase_grid.jpg)

*Mosaic assembled from the full-resolution renders via [`render_showcase.py`](render_showcase.py).*

## Optional detail pass — and which one to use

**Lightweight (enough for most people): the same-model draft + regenerate.** Render the draft at half
size with the 2-step pipeline, then regenerate at full size from `(1−σ₀)·latent(upscaled draft) +
σ₀·noise` with 3 steps on the refiner grid (σ₀ 0.5–0.7): ~1–2 s at 4 MP, composition and style hold,
and it is what the [demo Space](https://huggingface.co/spaces/akhaliq/minimax-h3-image-2step) runs by
default. For most prompts this is all the detail pass you need — what follows is the heavyweight
option for maximum micro-detail at high resolution.

**For edits, lower σ₀.** The same pass polishes an edit without moving it: at **σ₀ 0.4–0.55** the
source holds while texture sharpens; at σ₀ 0.7 it redraws more boldly — the face drifts and flat
areas can speckle (a glasses edit on the Mona Lisa at three settings; the anchored draft carries
the edit, the refine is text-only):

[![detail-pass sigma sweep on an edit](assets/edit_detail_sigma.jpg)](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/edit_detail_sigma.jpg)

The Space skips its realtime polish for edits for exactly this reason; for a full-quality edit with
the detail pass, use σ₀ ≤ 0.55.

### Heavyweight: Open-H3-Regenerate-2K-IC

The 2-step drafts can be polished further with
[Open-H3-Regenerate-2K-IC](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC), the
in-context (IC) successor of the Regenerate-2K refiner: a second copy of the official transformer
carrying **HyperFlow 8-step + LMS Sharp + the IC LoRA** (merged in the authors' `infer.py` order),
regenerating each draft **at its native ~16 MP size while the draft stays in the transformer's context
as reference rows** (the pipeline's edit path, noised to 0.999), starting from
`(1−σ₀)·latent(draft) + σ₀·noise`. These are video-refinement LoRAs — the refine pass, not a straight
t2i merge, is the way to use them.

**Settings, chosen by ablation** (three showcase images × three variants): the authors' full recipe —
3 steps on the `up3` grid, **σ₀ = 0.835** — against IC-LoRA-only at σ₀ 0.7 and 0.5. Composition and
lettering held in all three; the full recipe gave the crispest micro-detail, so it is what this grid
uses (~35 s per 16 MP image, peak 95 GB on one H200, [`render_refine.py`](render_refine.py);
[ablation sheet](assets/_refine_ablation.jpg): rows are draft, full recipe, IC-only 0.7, IC-only 0.5).
The lower-σ₀ IC-only variants stay closer to the draft when you want less redrawing.

Zoom crops at 100% (draft | refined) — fur, dials and lettering are where the pass earns its keep:

[![before/after detail crops](assets/refine_zoom_comparison.jpg)](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/refine_zoom_comparison.jpg)

The full refined showcase, same twelve prompts and layout as the mosaic above:

[![2-step drafts + Open-H3-Regenerate-2K-IC refine, full recipe](assets/pdmd_2step_showcase_refined_grid.jpg)](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/pdmd_2step_showcase_refined_grid.jpg)

The same pass over the original 4096×4096 comparison grid ([drafts](assets/pdmd_2step_grid.jpg) →
refined, composition holds tile-for-tile):

[![original 2-step grid + IC details pass](assets/pdmd_2step_refined_grid.jpg)](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/pdmd_2step_refined_grid.jpg)

[![grid zoom crops, draft vs refined](assets/refine_zoom_comparison_grid.jpg)](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/resolve/main/assets/refine_zoom_comparison_grid.jpg)

**Speed (measured, bf16, FA3, one H200, 4096×4096):** stock MiniMax-H3-Image on its native 6-step PDD
schedule **~25.1 s**/image; + pDMD 2-step **~17.4 s** (**1.4× faster**); the IC detail pass adds
**~33 s** on top (~50 s total — its in-context reference rows double the transformer's tokens, and
the draft is VAE-encoded at full 16 MP). The 2-step alone is the speed play; the detail pass is a
quality play, and *with* it the pipeline is about 2× slower than the stock 6-step model.

At interactive sizes the picture flips: on the [demo Space](https://huggingface.co/spaces/akhaliq/minimax-h3-image-2step)
(slim FP8 + fused-Triton build, ZeroGPU RTX PRO 6000 slice), **2048×2048 renders in ~4.6 s *with* the
detail pass on — vs ~7 s for the official 6-step Space at the same size** — because the Space's
built-in detail pass is the lightweight same-model regenerate (+~1–2 s at 4 MP), not the heavyweight
IC refiner measured above. One caveat at σ₀ = 0.835: on flat, clean areas (studio backdrops, smooth gradients) the pass can invent
speckle-like micro-texture — drop to IC-only at σ₀ 0.5–0.7, or skip the pass, for that content.

## How it was run (and how to reproduce)

The official `MiniMaxH3ImagePipeline` class is not in a public diffusers release yet, and the official
SGLang/vLLM serving path hard-pins 6 steps. So [`render_grid.py`](render_grid.py) takes the diffusers
route:

> **Update (2026-11-10):** native MiniMax-H3-Image support just landed in diffusers `main` via
> [PR #15230](https://github.com/huggingface/diffusers/pull/15230) — `MiniMaxH3ImageModularPipeline`
> loads the official repo directly (full 66 GB text encoder, the official 2D image VAE, a new
> `MiniMaxH3ImageTransformer2DModel`, auto CPU offload). These grids predate the merge by hours and
> use the validated P-H3-stack route below; the pDMD LoRA's key fit with the new 2D transformer class
> is untested.

1. **Convert the official transformer on the fly** — stream the 12 official shards and remap them into
   the diffusers `MiniMaxH3Transformer3DModel` layout with the key mapping vendored from
   [`diffusers/scripts/convert_minimax_h3_to_diffusers.py`](https://github.com/huggingface/diffusers/blob/main/scripts/convert_minimax_h3_to_diffusers.py)
   (per-head-interleaved fused QKV → `to_q/to_k/to_v`, `[gate; value]` halves of `mlp.fc1` swapped for
   diffusers' SwiGLU). All 531 checkpoint keys map; the only uncovered model keys are the 4 unused
   audio-branch tensors (the image checkpoint has no audio weights).
2. **Run it in the proven P-H3-Image stack** — the converted transformer replaces the transformer inside
   [PrunaAI/P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image)'s ModularPipeline (Pruna's Qwen3-VL-4B
   text encoder + text connector, video-VAE decoder, `MiniMaxH3Scheduler`), the same stack as the sister
   card. Both grids share conditioning, VAE, seed and scheduler — the comparison is transformer ± LoRA
   only.
3. **2 steps**: `scheduler.set_shift(12.0)`, `scheduler.set_timesteps(3)` (N+1 grid points for N
   evaluations) and pass `sigmas=scheduler.sigmas`; no CFG, no negative prompt. The stock grid is the
   identical call without the merge.

Stack: `torch 2.13.0`, `transformers 5.20.2`, `diffusers @ c55ec279`, H200 (141 GB). Peak GPU 84 GB; the
official 66 GB text encoder is *not* needed in this stack (Pruna's 4B encoder + connector condition the
transformer instead), so the whole run fits comfortably on one H200.

## Usage (diffusers)

`lora_model_0.safetensors` carries diffusers PEFT keys, so it merges straight into a diffusers
`MiniMaxH3Transformer3DModel` — `merge_lora` in [`render_grid.py`](render_grid.py) does it in place.
Load P-H3-Image's transformer, or this repo's pre-converted official `transformer/` with a plain
`MiniMaxH3Transformer3DModel.from_pretrained(...)` — no on-the-fly conversion needed. Then sample at
2 steps: `scheduler.set_shift(12.0)`, `scheduler.set_timesteps(3)` (N+1 grid points for N evaluations),
pass `sigmas=scheduler.sigmas`, no CFG, no negative prompt.

pDMD's operating point is **shift 12** and **N+1 timesteps for N denoising evaluations**; at 3–4 steps
the same merge gets closer to the fully-converged look if 2 steps drifts on your prompt.

## Prompting

MiniMax-H3-Image works best with **long, complete visual descriptions** — the kind its prompt-expansion
stage produces; every grid here renders the prompt as written. MiniMax's official
[h3-image-prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3-Image/tree/main/skills/h3-image-prompt-writing)
documents the T2I and edit prompt shapes (T2I: a full scene description; edit: preserve → change →
describe the target), and [MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE)
expands short prompts locally (also runnable as
[diffusers custom blocks](https://huggingface.co/diffusers/minimax-h3-image-pe-blocks)).

## License

- The pDMD 2-NFE LoRA is **Apache-2.0** (© the pDMD authors; this repo redistributes it unmodified).
- Anything rendered with or derived from MiniMax-H3-Image remains governed by the
  **[MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)** —
  read it before use; it carries regional and attribution terms beyond a standard permissive license.

## Citation

```bibtex
@misc{wang2026pdmdprojecteddistributionmatching,
      title={PDMD: Projected Distribution Matching Distillation for Video Diffusion Models},
      author={Zimo Wang and Junkun Yuan and Angtian Wang and Haotian Yang and Canyu Zhang and Siyuan Yuan and Xingchang Huang and Bo Liu and Yizhi Wang and Yiding Yang and Chongyang Ma and Gordon Guocheng Qian},
      year={2026},
      eprint={2609.35768},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.35768},
}
```

## Credits

- Base model: [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) (MiniMax)
- 2-step LoRA: [pdmd2026/pdmd_2NFE_lora](https://huggingface.co/pdmd2026/pdmd_2NFE_lora) ([arXiv 2609.35768](https://arxiv.org/abs/2609.35768))
- Inference stack: [PrunaAI/P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image) ModularPipeline (Pruna AI)
- Key conversion: [`diffusers/scripts/convert_minimax_h3_to_diffusers.py`](https://github.com/huggingface/diffusers/blob/main/scripts/convert_minimax_h3_to_diffusers.py) (Apache-2.0)
- ComfyUI LoRA conversion: [Iwannapose/minimax_h3_pdmd_2nfe_comfyui](https://huggingface.co/Iwannapose/minimax_h3_pdmd_2nfe_comfyui) (Apache-2.0)
- ComfyUI repack: [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)
- Details pass: [Efficient-Large-Model/Open-H3-Regenerate-2K-IC](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC) (in-context refiner)
- Demo Space: [akhaliq/minimax-h3-image-2step](https://huggingface.co/spaces/akhaliq/minimax-h3-image-2step)
- Sister card (same treatment for P-H3-Image): [akhaliq/p-h3-image-pdmd-2step](https://huggingface.co/akhaliq/p-h3-image-pdmd-2step)
- Merge tooling, validation and grids: built with ML Intern, November 2026
