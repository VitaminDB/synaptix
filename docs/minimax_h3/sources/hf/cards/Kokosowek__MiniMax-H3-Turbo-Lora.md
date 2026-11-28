# Kokosowek/MiniMax-H3-Turbo-Lora

Источник: https://huggingface.co/Kokosowek/MiniMax-H3-Turbo-Lora · скачано 2026-11-28
Загрузок: 256 · лайков: 1 · базовая: Comfy-Org/MiniMax-H3 · теги: lora,comfyui,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-10-04T13:44:16.000Z

---

---
license: apache-2.0
base_model: Comfy-Org/MiniMax-H3
tags:
  - text-to-video
  - text-to-audio
  - audio-video
  - lora
  - minimax-h3
  - comfyui
pipeline_tag: text-to-video
---

# MiniMax-H3 Turbo LoRA — few-step audio-video generation

A LoRA for [MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) that renders
joint **video + synchronized stereo audio** in as few as **4 sampling steps**
instead of the usual ~20 — a ~5× sampling speedup — and keeps getting better as
you add steps.

## Which checkpoint — `v4` (step 600) or `v1` (850)?

For **most** work, use **`minimax_h3_turbo_v4_step600_ema.safetensors`**. It's the
strongest checkpoint we've released: much better static and small-motion shots,
markedly better micro-detail (faces, fingers, fine texture), and the
over-sharpening / plastic look of the earlier `v1` (~850) line is **fully
resolved**.

v4 introduced a **static-frame enhancement** — a big win for static and
small-motion content. The one trade-off shows up **only at 4 steps with large,
fast motion**, where v4 can produce **motion-smear / trailing ghosting** (we're
actively fixing this). Two things address it:

- **Use 6–8 steps.** This **largely removes the smear** and is where v4 looks its
  best. v4 also tolerates higher step counts better than v1, which tends to
  over-sharpen at high steps + strength 1.0.
- For the specific case of **4 steps *and* heavy motion**, the older **`v1` ~850**
  checkpoint can still be the friendlier pick.

```
Using 6–8 steps?        ── yes ──►  v4-600  (recommended)
   │ no (4 steps)
   ▼
Heavy / fast motion?    ── no  ──►  v4-600  (recommended)
   │ yes
   ▼
                                    v1-850  (friendlier at 4-step heavy motion)
```

Still a preview — training continues; the two areas still being improved are
**audio** and **behaviour under fast, intense motion**.

## Steps and strength — read this

- **4 steps is the recommended *minimum*; 4–8 is the useful range.** 6–8 steps
  look noticeably better than 4, so add steps if you can afford them. Past **8
  steps** it stops helping and can start to introduce **over-sharp artifacts** —
  there's no benefit to going higher, so stay in **4–8**.
- **Keep strength at `1.0`.** It's tuned for 1.0 and holds up well across the 4–8
  step range. Only reach for the strength dial if a *specific* clip misbehaves —
  then **blurry ghosting / smear → nudge up** (`~1.05–1.2`), **over-sharp grain →
  nudge down** (`~0.8–0.95`).
- Keep the scheduler on `simple`.

## Use it in ComfyUI (recommended)

Custom nodes: **[Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)**
— or search **"MiniMax-H3 Turbo"** in ComfyUI-Manager. (Keep the node updated; it
evolves alongside these weights.)

1. Install the nodes (Manager, or `git clone` into `ComfyUI/custom_nodes`) and put
   a `.safetensors` from this repo into `ComfyUI/models/loras/`. You also need the
   base MiniMax-H3 model, VAEs and text encoder — see the
   [MiniMax-H3 tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3).
2. Start from the official MiniMax-H3 workflow (t2v or i2v) and make two changes:
   - insert **MiniMax-H3 Turbo LoRA** between the model loader and the sampler;
   - feed `SamplerCustomAdvanced` from **MiniMax-H3 Turbo Sampler**, and set the
     scheduler to `simple` at **≥ 4 steps**.

Everything else stays as in the official graph, so both text-to-video and
image-to-video work. A ready-made t2v workflow ships in the
[node repo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/tree/main/example_workflows)
(and here as `minimax_h3_t2v_turbo.json`) — drag it in.

- **Base model**: any MiniMax-H3 base — full (`bf16`, `int8_convrot`) **and the
  pruned/curve variants** (`pruned_int8`, `pruned_fp8`). The node auto-detects a
  pruned base and re-injects the time-conditioning at run time, so **one LoRA file
  covers every base**.
- **`low_vram`** switch: **off** applies the LoRA at run time (sharpest,
  recommended); **on** merges it into the weights for the lowest peak VRAM (a bit
  softer on quantized bases). Turn it on only if you run out of memory.
- The custom sampler **auto-adapts to your ComfyUI version**: MiniMax-H3 runs
  video and audio on two different flow schedules; recent ComfyUI handles that
  natively (`ModelSamplingAV`) and older ComfyUI doesn't — the Turbo Sampler
  detects which and does the right thing either way, so nothing to change when you
  update ComfyUI.

## Weights

All bf16, ~744 MB, applied as a plain low-rank update
(`W_eff = W + lora_B @ lora_A`, alpha = rank, so no extra scaling). **Prefer the
EMA files**; the non-EMA ones are for comparison.

| file | notes |
|---|---|
| **`minimax_h3_turbo_v4_step600_ema.safetensors`** | **recommended — current best.** Strong static/small-motion, good micro-detail, no over-sharpening. |
| `minimax_h3_turbo_v4_step600.safetensors` | v4-600 non-EMA (comparison). |
| `minimax_h3_turbo_v4_step150_ema.safetensors` | earlier v4 checkpoint. |
| `minimax_h3_turbo_4step_ema_ckpt850.safetensors` | `v1` line (~850) — over-sharpened / plastic in general, but the friendlier pick for **4-step heavy motion** (see above). |
| `minimax_h3_turbo_4step_ema_ckpt500.safetensors` | older `v1` (~500), softer. |
| `minimax_h3_turbo_4step_ema.safetensors` | initial release (~200). |

*Naming:* `v4` is the current training recipe and `stepN` is the training step.
Older files carry the previous `4step_ckptN` naming, where `4step` referred to the
sampler-step count.

## Standalone (no ComfyUI graph)

`generate.py` is a single self-contained file — it loads the base DiT + a LoRA,
encodes the prompt, runs the few-step dual-schedule sampler, decodes and muxes an
mp4. It still needs a ComfyUI checkout for the H3 model / VAE / text-encoder
definitions:

```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI && pip install -r requirements.txt && cd ..
pip install -r requirements.txt          # this repo: torch, safetensors, imageio-ffmpeg

# base weights from Comfy-Org/MiniMax-H3 into a models/ tree, then:
python generate.py \
  --comfyui ./ComfyUI \
  --base   models/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --lora   minimax_h3_turbo_v4_step600_ema.safetensors \
  --te     models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --video-vae models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --prompt "A corgi in a chef hat flipping a pancake, sizzling sounds and a cheerful bark." \
  --width 1344 --height 768 --frames 124 --steps 6 --out corgi.mp4
```

## Notes

- **Resolution / duration**: width and height are multiples of 32 (short edge
  typically 768). Frame count is at 24 fps and snaps to the model's 17·k+5 grid
  (124 ≈ 5 s). Validated range ~124–362 frames (~5–15 s).
- **VRAM**: the base model is large (~33 B); an 80 GB GPU is comfortable at the
  largest resolutions. The ComfyUI node streams the base and adds the `low_vram`
  switch, so it runs on much smaller GPUs. In the standalone script,
  `--offload-adaln` trades ~13 GB of VRAM for CPU RAM.
- **Audio**: 32 kHz stereo, aligned to the video; the two streams ride different
  flow schedules and are integrated each on its own clock. (Audio is one of the
  two areas still being improved — see the top.)
