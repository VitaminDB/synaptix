# coolthor/MiniMax-H3-pruned-NVFP4

Источник: https://huggingface.co/coolthor/MiniMax-H3-pruned-NVFP4 · скачано 2026-11-28
Загрузок: 7602 · лайков: 53 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: quantized,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-11-14T04:29:10.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model:
  - MiniMaxAI/MiniMax-H3
base_model_relation: quantized
library_name: diffusion-single-file
pipeline_tag: image-text-to-video
tags:
  - text-to-video
  - image-to-video
  - text-to-audio-video
  - minimax
  - minimax-h3
  - nvfp4
  - fp4
  - quantized
  - comfyui
  - blackwell
extra_gated_prompt: >-
  MiniMax H3 is released under the MiniMax H3 Community License Agreement. Its open-weight
  license does NOT cover the European Union, the United Kingdom, the Republic of Korea, or
  the United States of America. By requesting access you confirm that you will not use,
  reproduce, modify, distribute, or display these weights or their outputs in those
  territories, and that you accept the Acceptable Use Policy in Exhibit A of the license.
extra_gated_fields:
  I have read the MiniMax H3 Community License Agreement: checkbox
  I am not located in, and will not distribute into, the Excluded Territories: checkbox
---

# MiniMax-H3 (pruned) — NVFP4

NVFP4 (TensorCore block-scaled FP4) **weight-only** quantization of both **pruned MiniMax-H3**
transformers — FL2VA and Ref2VA. MiniMax-H3 is a 33 B omni-modal video model that generates
video **and native stereo audio in a single forward pass**: dialogue, sound effects and score
come out with the picture, not from a separate stage.

**11.67 GiB per transformer instead of 19.53 GiB, and slightly faster** on Blackwell.

✅ **Everything you need for H3 is in here as of 2026-09-13.** Both transformers, the text
encoder, both VAEs and the two 4-step turbo LoRAs, laid out as ComfyUI's `models/` tree, so a
single `hf download` is enough to start generating. The only thing you fetch elsewhere is LTX-2.5, and only
if you want the two-stage refine workflow (different license, so it is linked, not mirrored).

📝 **Full write-up, benchmarks & the gotchas in detail:**
[Running MiniMax-H3 on one RTX 5090](https://ai-muninn.com/en/blog/minimax-h3-nvfp4-rtx5090)
· [繁體中文版](https://ai-muninn.com/zh-TW/blog/minimax-h3-nvfp4-rtx5090)

🖥 **Both Blackwell machines use this same repo.** An RTX 5090 (sm_120) and a DGX Spark /
GB10 (sm_121) load the exact same files and run the exact same graphs — nothing here is
per-machine. The only difference is how much fits at once: the Spark's 121 GiB of unified memory
holds both stages of the refine pipeline, while a 32 GB 5090 has to evict and reload between them.

📝 **On a DGX Spark:**
[MiniMax H3 Super Acceleration on a DGX Spark](https://ai-muninn.com/en/blog/h3-ltx-two-stage-dgx-spark)
(H3 draft + LTX-2.5 refine: 118.8 s vs 237.7 s native, 73.3 s warm)
· [繁體中文版](https://ai-muninn.com/zh-TW/blog/h3-ltx-two-stage-dgx-spark)

## What this is

- **Format:** NVFP4 weight-only. 200 linear layers are packed FP4 (4-bit) with per-block FP8
  (E4M3) scales and per-tensor FP32 global scales — the standard NVFP4 layout. Norms,
  embeddings, biases and the pruned model's `adaln_t_table` stay at their original precision.
  (Header dtype profile: 1,132 tensors — 200 U8 packed-FP4 + 200 F8_E4M3 block scales +
  200 F32 global scales, alongside 219 BF16 / 51 F16 / 4 F32 left untouched.)
- **Base:** the **pruned** Comfy-Org repack of
  [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3). The pruned checkpoint
  matters here: most of its ~13 B-parameter AdaLN projection has been replaced by an
  equivalent timestep lookup table (13,039,369,728 → 43,642,368 parameters, ~299×; reduced
  `adaln_proj` tensors remain rather than being removed outright), so quantization never
  touches it. MiniMax document this themselves — the model card notes the AdaLN modulation
  outputs can be precomputed and cached and "do not need to be loaded for inference-only
  deployment". AdaLN emits the
  scale/shift applied to every residual stream and its error compounds across every block and
  every sampling step — the pruned base removes that risk instead of trying to survive it.
- **Size:** 11.67 GiB per transformer, down from 19.53 GiB. The two transformers are
  alternative task checkpoints and are never loaded together, so what has to fit is one
  transformer plus the text encoder: measured peak is 26,914 MiB on a 32 GB card.

## Honest caveats (read these)

**1. Blackwell only. On anything older this is a size play with no speed.**
NVFP4 has tensor-core support only on sm_120+. On Turing/Ampere/Ada, ComfyUI lists `nvfp4`
under `emulated ops`: the file loads and runs, but weights are expanded before the matmul, so
you get the smaller download and none of the benefit. For those cards, use INT8-ConvRot or a
W4A4-ConvRot build instead.

**2. A different quantization does not give you the same clip faster — it gives you a
different clip.**
Quantization changes the numerics, the denoising trajectory diverges, and a fixed seed lands
somewhere else. `samples/nvfp4_vs_int8_sidebyside.mp4` shows NVFP4 next to INT8-ConvRot from
the same seed and prompt: both are coherent — materials, lighting and the carved Chinese
characters hold up in each — but the framing, the bird's position and how much of the
inscription is legible are simply not the same shot. Neither is broken; they are two valid
solutions to the same prompt. If you have a seed you like on INT8, expect to re-roll after
switching.

**3. The timings below are thermally limited, so read them with the ambient in mind.**
Those runs were made at about 35 °C room temperature with a 500 W power limit. The card sat
roughly 2 °C under its throttle point and held only 84–87% of its maximum SM clock. On a cooler
machine both formats will be faster and the gap between them may shift. Treat the absolute
seconds as this machine on that day; the ratio is the part that transfers.

**4. Prompting is not optional here.**
H3 accepts a keyword list and quietly gives you something worse. It expects a structured
context — see the prompting section below.

## Files

| File | Size | Purpose |
|---|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors` | 11.67 GiB | `t2va` (text→video) and `fl2va` (first and/or last frame→video) |
| `diffusion_models/minimax_h3_ref2va_pruned_nvfp4.safetensors` | 11.67 GiB | `ref2va` — up to 9 reference images, 3 videos, 3 audio clips; keeps a character, style or voice consistent |
| `text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` | 14.61 GiB | Qwen3-VL-32B text encoder, **abliterated** variant, quantized to NVFP4 here. See the note below before you pick this one. |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 4.85 GiB | Video VAE |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.56 GiB | Audio VAE |
| `loras/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors` | 1.82 GiB | lightx2v 4-step turbo LoRA for FL2V. **Apache-2.0**, mirrored byte-identical from [`lightx2v/Minimax-h3-Turbo-SLA`](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) |
| `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.82 GiB | Same, for Ref2V. From [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |
| `workflows/minimax-h3-ltx-hybrid-refine.json` | 6 KB | The two-stage H3 → LTX-2.5 workflow, 32 nodes |
| `workflows/minimax-h3-ltx-hybrid-refine-nogemma.json` | 6 KB | Same graph with the LTX text encoder removed — 7.85 s faster, bit-identical output |
| `conditioning_cache/ltx_generic_refine.cond` | 434 KB | Precomputed stage-2 conditioning, which is what lets the no-gemma graph skip the LTX text encoder |
| `custom_nodes/ComfyUI-CondCache/__init__.py` | 1.7 KB | Two nodes (`SaveConditioningT` / `LoadConditioningT`) that read and write that file |

**On the text encoder.** This repo ships the *abliterated* Qwen3-VL-32B, quantized to NVFP4.
Measured on a 5090 at 480p it is **slower and slightly worse than the AWQ build**: 90.6 s
against 70.5 s. It is here because it does not refuse. If you have no use for that, take
`text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` from
[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) instead — same
`CLIPLoader` slot, same `type: minimax`.

The repo mirrors ComfyUI's `models/` tree, so one command puts everything where it belongs:

```bash
# everything, ~47 GiB
hf download coolthor/MiniMax-H3-pruned-NVFP4 --local-dir ComfyUI/models

# text-to-video and keyframe work only, ~33.5 GiB
hf download coolthor/MiniMax-H3-pruned-NVFP4 --local-dir ComfyUI/models \
  --include "diffusion_models/minimax_h3_fl2va*" "text_encoders/*" "vae/*" \
            "loras/minimax_h3_fl2v*"
```

⚠️ `custom_nodes/` lands under `ComfyUI/models/` with the command above, which is the wrong
place for it. Move it once:

```bash
mv ComfyUI/models/custom_nodes/ComfyUI-CondCache ComfyUI/custom_nodes/
```

## Usage (ComfyUI)

Single-stage generation requires **ComfyUI ≥ 0.30.0**. No special launch flags are needed.

⚠️ **The two-stage refine needs ≥ 0.34.0** and fails silently below it — see
[Environment](#environment-the-versions-are-load-bearing) before you run that graph.

| | |
|---|---|
| Loader | `UNETLoader` → the `.safetensors` above, `weight_dtype: default` |
| Text encoder | `CLIPLoader` → `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors`, **type `minimax`** |
| Sampler | `KSamplerSelect` → `res_multistep` |
| Scheduler | `BasicScheduler` → `simple`, **steps 10**, denoise 1.0 |
| Guider | `BasicGuider` — the weights are CFG-distilled, so there is no negative prompt and no CFG value |
| Video | `CreateVideo` → 24 fps, audio connected from `VAEDecodeAudio` |

**Steps: start at 10, not 20.** The ComfyUI template ships 20, but MiniMax publishes no
recommended step count — their reproducible request scripts contain no step parameter at all,
and the released weights are CFG-distilled. 10 steps showed no visible quality loss on a
10-second clip while cutting time roughly in half (324 s → 172 s, same resolution and seed).

Frame math:

- Frame counts sit on a **17k+5** grid at 24 fps (5, 22, … 124, … 243 …); the stock template
  computes this for you.
- Trained range is **124–362 frames** (5.17–15.08 s).
- Native canvas is a 768 px short edge, capped at 768×1344, rounded to a multiple of 32.

With `--use-sage-attention`, this file ran **7% faster** on the 5090 (172 s vs 185 s), and the
output was effectively unchanged — only 4 bytes differed in a 1.97 MB file. That is far from the
"roughly double" figure quoted elsewhere; the installed package was SageAttention 1.0.6, and
that claim refers to 2.x.

## Environment: the versions are load-bearing

**Single-stage H3 runs fine on ComfyUI 0.30.x. The two-stage refine does not, and it will not
tell you.**

On a DGX Spark (GB10) running ComfyUI 0.30.0, the LTX refine returned `status: success`, exit
code 0, no errors in the log, and a correctly shaped 1344×768 h264+aac file with 121 frames —
and every frame was garbage. Everything feeding it was verified good: the H3 draft decodes to
a clean picture, and so does the upsampled latent one node before the refiner. The post-refine
latent held zero NaNs and a healthy standard deviation of 0.80. It simply correlated 0.235 with
its own input, where that input plus noise correlates 0.74.

The same weights, same graph and same seed produced a correct clip on an RTX 5090 that was
already on 0.34.0. Matching the Spark to it fixed the Spark, and torch was never touched.

| | correct output | silent garbage |
|---|---|---|
| ComfyUI | 0.34.0 | 0.30.0 |
| comfy-kitchen | 0.2.31 | 0.2.26 |
| ComfyUI-Spectrum-MiniMax-H3 | `4783a22` (2026-08-18) | `fcf2332` (2026-08-05) |
| torch | 2.13.0+cu130 | 2.13.0+cu130 |

⚠️ That table is a record of a past debugging session. **Spectrum is no longer part of these
workflows** — see "Removed 2026-09-16" at the end of this card. You do not need to install it.

Treat that column as one unit. The parts are not independently selectable: ComfyUI 0.34.0 will
not even finish importing against comfy-kitchen 0.2.26 — it raises `module 'comfy_kitchen' has
no attribute 'int8_attention_is_available'` — so there is no configuration in which the blame
can be pinned on a single package, and no point hunting for one.

⚠️ **Do not** reach for `pip install -r requirements.txt --upgrade`. ComfyUI's
`requirements.txt` leaves `torch` unpinned (and `torchvision`, `torchaudio`), so `--upgrade`
can take your PyTorch and CUDA runtime packages with it. Install the pinned entries instead:

```bash
pip install comfy-kitchen==0.2.31 comfy-aimdo==0.4.15 \
            comfyui-frontend-package==1.49.6 \
            comfyui-workflow-templates==0.11.48 \
            comfyui-embedded-docs==0.5.10
```

**Look at a frame before you trust a number.** Every mechanical check passed on the broken
runs: exit code, stream layout, resolution, frame count, duration — and once the failure mode
shifted from black to coloured noise, a black-frame detector passed too. If you are timing this
pipeline, decode a frame and look at it, or you will benchmark a pipeline that produces nothing.

## Two-stage refine: draft small, enhance the one you keep

`workflows/` holds the graph from NVIDIA's recipe, rebuilt on a 5090 and measured. NVIDIA calls
the method [**MiniMax H3 Super Acceleration**](https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/)
and publishes it under Sol Engine, full name Sol Video Inference Engine. The DGX Spark build of
it is [Sol-H3-Spark](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/). Four terms travel
together here and they are not interchangeable:

| Name | What it actually is |
|---|---|
| **MiniMax H3 Super Acceleration** | the method — draft small with H3, refine with LTX-2.5 |
| **Sol Engine** | NVIDIA's acceleration framework, which the method is published under |
| **Sol-H3-Spark** | NVIDIA's own build of the method for a DGX Spark |
| **Sol-Attn** | an attention kernel you can use inside the stage-2 refine |

**The 672×384 draft here is NVIDIA's own Spark setting, not a deviation.** Their GB200 line
drafts at 896×512 instead; the two lines differ. The sigma 0.78 figure below was swept at
672×384 and lines up with the Spark build — carry it over to a 896×512 draft and it needs
re-checking.

For a coordinate: NVIDIA report 6.852 s for a 5-second 1344×768 clip on a single GB200. ⚠️ That
number sums separately measured stages and excludes model loading and warmup, so it is not the
same quantity as the submission-to-finish times below.

H3 never runs at full resolution: it draws a 672×384 draft in **25.5 s**, and [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) takes
3 refinement steps to reach 1344×768 in **40.3 s** total, cold.

The point is not the total. It is that the draft is cheap enough to roll repeatedly — you
pick the take you want at 25.5 s each, and pay for the refine once.

| | Cold | Warm |
|---|---|---|
| 672×384 draft on its own | 25.5 s | — |
| draft + LTX refine → 1344×768 | **40.3 s** | 30.6 s |
| H3 native at 1344×768, 4 steps | 51.5 s | 41.5 s |

**On a DGX Spark (GB10, 121 GiB unified memory) the same graph beats native H3 by 2.00×, where
on the 5090 it only manages 1.28×.** Measured 2026-09-14 on ComfyUI 0.34.0 with the GPU clock capped at 2200 MHz,
same prompt and seed throughout:

| DGX Spark (GB10) | Cold | Warm |
|---|---|---|
| H3 native, 1344×768, 4 steps | 237.7 s | — |
| draft + LTX refine → 1344×768 | **118.8 s** | **73.3 s** |

The two stages come to 53.6 GiB of weights. A 5090 cannot hold that at once, so it evicts and
reloads on every run, which eats the time the method saves. The Spark holds both and never
reloads them; measured peak is 104 GiB. The warm run re-executes both samplers and reuses 18
of 32 nodes — all of them loaders and configuration, which is exactly what residency buys.
Native H3 gains nothing from that memory, because it has no nodes to reuse.

⚠️ An earlier version of this file quoted 110.6 s / 208.5 s / 66.8 s for a 1.89× figure. Those
two-stage runs were made on ComfyUI 0.30.0 and **their output was garbage** — see
[Environment](#environment-the-versions-are-load-bearing). The numbers above replace them and
come from runs whose frames were decoded and looked at.

Two things worth knowing before you load it:

- **Start the refine at sigma 0.78, not the official 0.909.** At 0.909 the refine redraws the
  face into a different person and the background breathes. 0.78 fixes both and is faster.
- **The LTX text encoder is optional.** Stage-2 conditioning is one constant string, so it can
  be computed once and cached. `-nogemma.json` + `conditioning_cache/ltx_generic_refine.cond`
  do exactly that: **7.85 s saved (16.3%), and 121 of 121 frames come out bit-identical.**
  You need `custom_nodes/ComfyUI-CondCache` installed for that graph to load.

You also need the LTX-2.5 weights, which are **not** mirrored here — they carry the LTX-2.x
community license, not this repo's. Get them from
[`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5), and the handoff nodes from
[`T8mars/comfyui-minimax-h3-audio-T8`](https://github.com/T8mars/comfyui-minimax-h3-audio-T8).

📝 Full walkthrough, with every benchmark and how it was measured:
[Roll cheap, then enhance the keeper](https://ai-muninn.com/en/blog/minimax-h3-ltx-hybrid-refine-rtx5090)
· [繁體中文版](https://ai-muninn.com/zh-TW/blog/minimax-h3-ltx-hybrid-refine-rtx5090)

## Measured performance

RTX 5090 · ComfyUI 0.30.1 · torch 2.11.0+cu128 · 500 W power limit · SageAttention off

Both runs used the same job; only the quantization differed: **fl2va**, 864×480, 243 frames
(10.125 s), 10 steps, seed 16, `res_multistep` + `simple`, with the same text encoder and VAEs.

| | **NVFP4 (this repo)** | INT8-ConvRot |
|---|---|---|
| File size | **11.67 GiB** | 19.53 GiB |
| Generation time | **175 s** | 185 s |
| Peak VRAM | **26,914 MiB** | 28,581 MiB |
| Mean SM clock | **2,696 MHz** | 2,585 MHz |
| Peak GPU temp | 82 °C | 80 °C |

**≈5% faster, 40% smaller, 1.6 GB less VRAM.** Ref2va was not benchmarked separately; it is the
same architecture at the same size, so expect the same order of magnitude plus whatever your
reference images cost to encode.

## Prompting

H3 expects a structured context, not a keyword list. Text-only generation (the T2VA task)
uses three fields, in this order. Image-conditioned tasks (I2VA / FL2VA / L2VA) additionally
need an alignment instruction as the very first line, followed by a blank line.

```text
integrated_multimodal_description: [Shot 1] <style>, <framing>, <what happens> …
[Shot 2] At 00:06.000, the camera cuts to …

overall_soundscape: <ambience, action sounds, non-verbal human sounds>

non_diegetic_music: <instruments, tempo, rhythm, dynamics — no mood words>
```

Easy things to get wrong:

- Shot 1 carries **no** timestamp; later shots need strictly increasing cut times within the
  clip's duration.
- Camera motion is motion type + amplitude + speed, written as a natural sentence
  (medium amplitude and normal speed are usually omitted):
  `The camera pushes in with small amplitude at slow speed toward …`
- Dialogue: everything except the language tag and the verbatim line goes **outside** the tag —
  `The man, with a low, restrained voice (S1), says: <d>[Chinese] …</d>`
- All three fields are always present; what is mutually exclusive is which sound goes where.
  Dialogue and diegetic music belong in the description, ambience in `overall_soundscape`, and
  only audience-side score in `non_diegetic_music`. Write `N/A` when a field has nothing —
  though `overall_soundscape` takes `N/A` only when total silence was explicitly requested.
- On-screen text goes in English double quotes, verbatim, untranslated.
- **Describe what is visible, not what is absent.** Diffusion models cannot render absence, and
  they follow geometry far better than words like "left": `enters from the left edge of the
  frame` works where `his left hand` does not.

MiniMax ships the full specification as two prompt-writing guides with the base model.

## Samples

Generated with the files in this repo at the settings above, each with the complete prompt that
produced it. Audio is native — dialogue, ambience and score come out of the same forward pass as
the picture.

| File | What it is |
|---|---|
| [`samples/yangguo_nvfp4.mp4`](samples/yangguo_nvfp4.mp4) | NVFP4 (this repo) — 175 s |
| [`samples/yangguo_int8_convrot.mp4`](samples/yangguo_int8_convrot.mp4) | INT8-ConvRot, for comparison — 185 s |
| [`samples/nvfp4_vs_int8_sidebyside.mp4`](samples/nvfp4_vs_int8_sidebyside.mp4) | The two above side by side, NVFP4 left, INT8 right |
| [`samples/yangguo_prompt.txt`](samples/yangguo_prompt.txt) | The exact prompt, verbatim |

[`samples/README.md`](samples/README.md) covers what the clip exercises — two shots in one
generation, dialogue across the cut, carved characters, a one-armed character — and why each of
those is hard.

## Credits

- **Base model:** [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) — 33 B
  omni-modal video + native audio.
- **ComfyUI repack (pruned base, text encoder, VAEs):**
  [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3).
- **NVFP4 approach:** [`lilcheaty/MiniMax-H3-NVFP4`](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4).
- **4-step turbo LoRAs:** [`lightx2v`](https://huggingface.co/lightx2v) —
  [`Minimax-h3-Turbo-SLA`](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) (FL2V) and
  [`Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) (Ref2V). Apache-2.0,
  mirrored here unmodified so the repo is one download; the sha256 of both files matches theirs.
- **Two-stage refine:** the idea is NVIDIA's
  [Sol-H3-Spark](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/); the ComfyUI nodes that
  implement it are [`T8mars/comfyui-minimax-h3-audio-T8`](https://github.com/T8mars/comfyui-minimax-h3-audio-T8);
  stage 2 is [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5).
- Quantization, packaging and benchmarks by **coolthor**.

This is a **modified** redistribution: the transformer weights have been quantized to NVFP4 and
are not the originals published by MiniMax.

## License

Distributed under the **MiniMax H3 Community License Agreement**, Copyright © 2026 MiniMax.
All Rights Reserved. A complete copy is in `LICENSE` and the required notice is in `NOTICE`.

**Two exceptions, both in `loras/`.** lightx2v trained the two turbo LoRAs themselves and released them under **Apache-2.0**. They
are not MiniMax weights and they carry their own terms. Everything else in this repo is under
the agreement below.

The `workflows/`, `conditioning_cache/` and `custom_nodes/` files are my own work and carry no
restriction.

Key points carried over:

- **The open-weight license excludes the European Union, the United Kingdom, the Republic of
  Korea, and the United States of America.** Section V.4 prohibits use, reproduction,
  modification, distribution and display of the works *or their outputs* outside the Applicable
  Territory — which is why this repo is gated. If you are in an excluded territory, MiniMax
  accepts license applications; see their
  [Q&A](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md).
- Any derivative must ship with a complete copy of the Agreement, and modified files must carry
  a notice that they were changed (done above).
- Section V.2: before giving anyone access you must bind them to enforceable terms **at least
  as protective as the use restrictions in Section V and Exhibit A**, and tell them those
  restrictions apply. Section V.5 additionally requires safeguards if you expose generation to
  third parties.
- Section IV.2: a commercial product or service built on this must display "MiniMax H3"
  prominently in its user interface.
- Commercial products or services above **USD $20,000,000** in yearly revenue need separate
  written authorization from MiniMax.

---

☕ If this saved you GPU hours, you can [buy me a coffee](https://buymeacoffee.com/thorlin).


## Removed 2026-09-16: the Spectrum node

The workflows here used to include `SpectrumApplyMiniMaxH3`, which meant installing
`ComfyUI-Spectrum-MiniMax-H3` as an extra node pack. Both workflows have been rebuilt without it.

**It could never do anything in this graph.** The node was configured with `warmup_steps = 5`
while the draft runs **4 steps**, so it never finished warming up. Turn its `debug` on and it
reports this itself:

```text
actual_steps=4  forecast_steps=0  phase_counts=model:a4/f0
actual_transformer_calls=4  forecast_calls=0  bypassed_steps=0
```

`forecast_calls=0` — not one forecast, and the same number of transformer calls you get with the
node absent.

Removing it is **not** a speedup, and the honest numbers say so:

| | with Spectrum | without |
|---|---:|---:|
| RTX 5090, warm (interleaved, n=6 pairs) | 22.96 s | 22.37 s |
| RTX 2080 Ti 22 GB, warm (n=3 each) | 165.0 s | 165.0 s |

The 5090 medians differ by 0.59 s against a per-arm standard deviation of 2.88 s — that is noise,
not a gain. A first pass without interleaving and without discarding warm-up runs reported +7.1%;
that number was an artifact of run-to-run drift.

It came out because it costs you a node pack to install and reads as load-bearing when it is not.
Node counts went from 33 → 32 and 32 → 31. Credit for spotting the mechanism goes to a reader.

⚠️ If you are running an older copy of these workflows, nothing is broken — you were just paying
for a wrapper that never fired.

---

Made by Thor Lin. Full benchmarks, setup notes and the runs that failed are on [ai-muninn.com](https://ai-muninn.com). More quantized models: [huggingface.co/coolthor](https://huggingface.co/coolthor).

## Provenance

This section documents release [`caf421a3f8bba8db5442e85d211c33183c1896ff`](https://huggingface.co/coolthor/MiniMax-H3-pruned-NVFP4/tree/caf421a3f8bba8db5442e85d211c33183c1896ff), inspected on 2026-11-14. Where the exact upstream commit used at build time is unknown, it is listed as **not recorded** rather than backfilled with today's upstream `HEAD`.

### Source lineage

| Source repository | Revision used for this build | Role |
|---|---|---|
| [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | **not recorded** | Immediate pruned transformer repack; also the documented H3 companion source |
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | **not recorded** | Original upstream model |
| [lightx2v/Minimax-h3-Turbo-SLA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) | **not recorded** | Bundled FL2V turbo LoRA |
| [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | **not recorded** | Bundled Ref2V turbo LoRA |

### Build details

**Tools and versions:** Custom NVFP4 conversion; build-time tool identity/version and exact command not recorded in the recovered material.

Both pruned FL2VA/Ref2VA transformers use weight-only NVFP4: 200 Linear layers each. The upstream pruned repack replaces most AdaLN projections with timestep lookup tables. The package also contains an abliterated Qwen3-VL-32B text encoder, video/audio VAEs and two separately distributed turbo LoRAs. The exact immediate source repository of the Heretic encoder and each VAE was not recovered; do not treat the original MiniMax repository as proof of those components.

The documented coolthor step is NVFP4 conversion and packaging, with no additional fine-tuning or merged LoRA recorded. This package nevertheless includes upstream abliterated encoder weights and separately trained turbo LoRAs; it is not an unmodified copy of the original MiniMax weights. The bundled LoRAs are separate files, not evidence of a merge into the stored transformer.

**Exact build command:** not recorded in the recovered material. The method description above is not a complete executable reproduction recipe.

### Weight file SHA-256 checksums

The values below are Hugging Face LFS object SHA-256 values for the pinned release above. The public tree endpoint masks LFS digests for this gated repository; the hashes below come from the public revision-specific model metadata endpoint. File sizes agree with the pinned tree, but a second public digest source was unavailable. They identify full file contents, not LFS pointer text or Git blob IDs. Calibration imatrices are included when present.

| File | Bytes | SHA-256 |
|---|---:|---|
| `diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors` | 12528636800 | `9d49beb65ddc373a0df523b8ce715b61b73f88127bf3c8d3ffda76c5dda02bd4` |
| `diffusion_models/minimax_h3_ref2va_pruned_nvfp4.safetensors` | 12528636800 | `8eea02f43902e69904990c4405968d01a13c6656aa392d37ab80331e79b5df2f` |
| `loras/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors` | 1956192992 | `5cae6df40a06ea825f85fc8876c9ea1c9692c833a9af07bb8b3bac9ce2a71bac` |
| `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1956193000 | `5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c` |
| `text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` | 15683129587 | `a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f` |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 605254808 | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5207808496 | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |

### Verifying the files

Download the desired files from revision `caf421a3f8bba8db5442e85d211c33183c1896ff` and compare `sha256sum <file>` (Linux) or `shasum -a 256 <file>` (macOS) with the table. If cloning with Git LFS, hash the downloaded weight files, not the LFS pointer files. A matching hash confirms your files match this release; it does **not** prove the weights came only from the listed base model with no other changes.

Checking how the weights were derived requires the exact upstream commit, tool versions, recipe and any calibration or training data, then re-running the build and diffing the weights with a format-aware tool (or comparing file hashes where serialization is deterministic). Because of the gaps listed below, this build can't be reproduced exactly today. Quantized files are expected to differ numerically from BF16 sources, and a raw file hash cannot establish a base-to-quant weight diff. No quantization rerun or weight-diff audit was performed for this provenance review.

**Known gaps:** All build-time upstream commits; Build-time converter/script, version and command; Exact encoder/VAE source mapping.

Sources: [existing model card](https://huggingface.co/coolthor/MiniMax-H3-pruned-NVFP4/blob/caf421a3f8bba8db5442e85d211c33183c1896ff/README.md) and [HF metadata](https://huggingface.co/api/models/coolthor/MiniMax-H3-pruned-NVFP4/revision/caf421a3f8bba8db5442e85d211c33183c1896ff?blobs=true).
