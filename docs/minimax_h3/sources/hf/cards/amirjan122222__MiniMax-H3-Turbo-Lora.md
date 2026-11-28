# amirjan122222/MiniMax-H3-Turbo-Lora

Источник: https://huggingface.co/amirjan122222/MiniMax-H3-Turbo-Lora · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая: Comfy-Org/MiniMax-H3 · теги: lora,comfyui,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-07T22:04:38.000Z

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

# MiniMax-H3 Turbo LoRA — 4-step audio-video generation (preview)

A LoRA for [MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) that renders
joint **video + synchronized stereo audio** in **4 sampling steps** instead of
the usual ~20 — roughly a 5× speedup in sampling wall-clock.

> ⚠️ **Preview — sharp, but with known artifacts.** The current weights
> (`ckpt850`) are the **final checkpoint of this training round**. Detail and
> sharpness at 4 steps are now high — a large step up from earlier checkpoints —
> but serious issues are surfacing at this point: **plastic-looking skin and
> over-sharp grain/noise**. Training of this round is **paused while we address
> them**, so treat these as a sharp-but-imperfect preview, not a finished model.
> The ComfyUI nodes are also prototype code: **functionality and compatibility
> are not guaranteed.** **If something breaks, please open an issue** on the
> [node repo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues).

## Use it in ComfyUI (recommended)

Custom nodes: **[Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)**
— or search **"MiniMax-H3 Turbo"** in ComfyUI-Manager.

> 🔄 **Keep the node updated** — it's actively evolving and features land in new
> versions (e.g. pruned-base support arrived after the first release). Update via
> ComfyUI-Manager or `git pull`.

1. Install the nodes (Manager, or `git clone` into `ComfyUI/custom_nodes`).
2. Download a `.safetensors` from this repo into `ComfyUI/models/loras/`.
3. Start from the official [MiniMax-H3 workflow](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
   (text-to-video or image-to-video) and make two changes:
   - insert **MiniMax-H3 Turbo LoRA** between the model loader and the sampler;
   - replace the sampler feeding `SamplerCustomAdvanced` with **MiniMax-H3 Turbo
     Sampler (4-step)**, and set the scheduler to **4 steps** (`simple`).

Everything else stays as in the official workflow, so both t2v and i2v work. A
ready-made t2v workflow is included here (`minimax_h3_t2v_turbo.json`) and in the
[node repo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/tree/main/example_workflows)
— drag it into ComfyUI. The custom sampler **auto-adapts to your ComfyUI version**:
MiniMax-H3 runs video and audio on two different flow schedules; recent ComfyUI
handles that natively (`ModelSamplingAV`), so on those builds the Turbo Sampler
steps as a plain single-schedule sampler (and a stock `euler` works too), while on
older ComfyUI it steps each stream on its own clock so the audio doesn't break at
4 steps. Keep it in the workflow and it does the right thing either way.

- **Steps**: with `ckpt850`, **4 steps is already sharp** (earlier checkpoints
  needed 6–8 to firm up). Any count **≥ 4** is valid; more steps still help a
  little. Keep the scheduler on `simple`.
- **LoRA strength** (default `1.0`) is the dial for the sharpness/artifact
  trade-off: if the result shows **blurry ghosting / smear**, nudge strength
  **up** (e.g. `1.05–1.2`); if it shows **over-sharp grain / artifacts**, nudge
  it **down** (e.g. `0.8–0.95`).
- **Base model**: works with any MiniMax-H3 base — full (`bf16`, `int8_convrot`)
  **and the pruned/curve variants** (`pruned_int8`, `pruned_fp8`); the ComfyUI
  node auto-detects a pruned base and re-injects the time-conditioning at run
  time, so one LoRA covers every base.
- **`low_vram`** (node switch): off by default (applies the LoRA at run time —
  sharpest, some extra peak VRAM). Turn it **on** if you run out of memory: it
  merges the LoRA into the weights for the lowest peak VRAM, at the cost of a
  **softer result on quantized (`int8` / `fp8` / pruned) bases**. Lowering the
  resolution or frame count also helps.

## Weights

All bf16, ~744 MB, applied as a standard low-rank update
(`W_eff = W + lora_B @ lora_A`, alpha = rank so no extra scaling):

| file | ~step | notes |
|---|---|---|
| `minimax_h3_turbo_4step_ema_ckpt850.safetensors` | ~850 | **recommended — current final checkpoint** (time-averaged EMA, sharp at 4 steps) |
| `minimax_h3_turbo_4step_ckpt850.safetensors` | ~850 | ckpt850 non-EMA — even sharper but over-sharpened; for comparison/analysis |
| `minimax_h3_turbo_4step_ckpt500.safetensors` | ~500 | older, non-EMA (softer) |
| `minimax_h3_turbo_4step_ema_ckpt500.safetensors` | ~500 | older EMA |
| `minimax_h3_turbo_4step.safetensors` | ~200 | initial release, non-EMA |
| `minimax_h3_turbo_4step_ema.safetensors` | ~200 | initial release, EMA (superseded) |

`ckpt850` is the final checkpoint of this training round (paused — see the note
at the top). Prefer the EMA file for the cleanest result.

## Standalone (no ComfyUI graph)

`generate.py` is a single self-contained file — loads the base DiT + a LoRA,
encodes the prompt, runs the 4-step dual-schedule sampler, decodes and muxes an
mp4. It still needs a ComfyUI checkout for the H3 model / VAE / text-encoder
definitions:

```bash
# ComfyUI (pinned to the commit these weights were validated against)
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI && git checkout 14b05228cef127ce529bc0c08660770d4af3e9a8
pip install -r requirements.txt && cd ..
pip install -r requirements.txt   # this repo: torch, safetensors, imageio-ffmpeg

# base weights from Comfy-Org/MiniMax-H3 into a models/ tree, then:
python generate.py \
  --comfyui ./ComfyUI \
  --base   models/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --lora   minimax_h3_turbo_4step_ema_ckpt850.safetensors \
  --te     models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --video-vae models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --prompt "A corgi in a chef hat flipping a pancake, sizzling sounds and a cheerful bark." \
  --width 1344 --height 768 --frames 124 --out corgi.mp4
```

## Notes

- **Resolution / duration**: width/height are multiples of 32; the short edge is
  typically 768. Frame count is at 24 fps and snaps to the model's 17·k+5 grid
  (124 ≈ 5 s). Validated range ~124–362 frames (~5–15 s).
- **VRAM**: the base model is large (~33 B); an 80 GB GPU is comfortable at the
  largest resolutions. The ComfyUI node streams the base and adds a `low_vram`
  switch (see above), so it runs on much smaller GPUs. In the standalone script,
  `--offload-adaln` trades ~13 GB of VRAM for CPU RAM.
- **Audio**: 32 kHz stereo, aligned to the video; the two streams ride different
  flow schedules and are integrated each on its own clock.
