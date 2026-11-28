# MATLOWAI/MiniMax-H3-Variable-Clock

Источник: https://huggingface.co/MATLOWAI/MiniMax-H3-Variable-Clock · скачано 2026-11-28
Загрузок: 340 · лайков: 2 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-11-26T03:11:08.000Z

---

---
license: mit
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: adapter
tags:
  - minimax-h3
  - lora
  - video
  - motion
  - slow-motion
  - variable-frame-rate
pipeline_tag: text-to-video
---

# MiniMax-H3 variable clock (run f, step 2,000, r16)

An experimental rank-16 LoRA that teaches MiniMax-H3 to run on a **variable clock**: the model is told, latent
by latent, how much real time each latent covers. Each slow-motion stretch then gets more latents per second
of real time, and the model generates those frames directly instead of making a normal clip and stretching it.

This is research output, released as is. Results below are from **one seed per prompt** and are mixed; read
the "What it does and does not do yet" section before using it.

## Why

Slow motion with H3 today goes through a "de-rope" pass (time-smear). Each real frame is repeated several times
into a longer clip, the model re-denoises that clip, and the repeats are thrown away afterwards. Two problems:

1. **Wasted compute.** You generate frames only to discard them.
2. **Small things move too fast.** At higher denoise the model re-invents fine, fast detail (hair, sparks,
   leaves, birds) at its normal per-frame rhythm inside the stretched clip. After the repeats are dropped,
   that detail plays back too fast and looks stiff, even though the big motion was slowed correctly.

The goal here is to bake the clock into the model so it **natively** changes its time rate. Slowed stretches
get less temporal compression (more compute per second of real time), and that compute should go into
smoother, more coherent motion in the frames you keep, not into frames you throw away.

## How it works

**The clock.** H3's video VAE packs pixel frames into latents in a fixed (1, 4, 4, 4, 4) pattern per 17 frames
(legal lengths are 17k + 5 frames). Each latent gets a *span*, the real time it covers, which drives its
temporal RoPE position. With 5/3 RoPE units per real frame:

| a 4-frame latent that holds | real frames | span | reads as |
|---|---:|---:|---|
| 4 distinct real frames | 4 | 6.67 | normal speed |
| 2 real frames, each repeated twice | 2 | 3.33 | 2x slow motion |
| 1 real frame, repeated four times | 1 | 1.67 | 4x slow motion |

A **hold map** lists, for every real frame, how many generated frames it occupies (1, 2 or 4), aligned to
the latent grid. The spans come from the hold map. The `clock_maps/` folder holds the two maps used in the
videos below, in the format `{"hold_map_used": [...]}`.

**The training data ("option 1").** The stock VAE is kept unchanged. Training clips were built from real
24 fps footage by repeating real frames *inside* latents: in a 2x stretch each latent holds 2 real frames
(each shown twice), and in a 4x stretch it holds 1 real frame (shown four times). The training target is
therefore always real frames, never invented in-between frames. Before training we checked that the stock VAE
round-trips these repeated-frame latents at least as well as ordinary ones (kept-frame PSNR 35–38 dB, copies
inside a latent within 0.3–0.5 of 255; 2 clips).

- 84 training rows from 21 live-action sources, 124 and 141 frames, 1280x736. Each source gives one
  normal-speed row, one constant 2x or 4x row, and two variable rows (ramps such as normal → 2x → 4x → 2x →
  normal, slow intro, slow outro, bursts).
- Of the 4-frame latents, 44% are normal speed, 30% 2x and 26% 4x. The mix is balanced across clip lengths
  and across the start, middle and end of clips, and slowed stretches start at random positions in the footage.
- Captions are the same for every row from a source, so the speed comes only from the clock, never from the text.
- No animation in the training data.

**Training.** LoRA rank 16 / alpha 16, AdamW 5e-5 (constant, 10 warmup steps), 2,000 steps, batch 1, bf16,
seed 20260923, 2 temporal fine RoPE bands kept at native rate (`keep_bands = 2`). A native-clock anchor
(weight 30) keeps normal-speed behaviour close to the base model. About 78 GPU-hours on one RTX PRO 6000
(about 2.3 min per step).

**Inference.** Give the sampler the hold map, so the per-latent spans are set, with the LoRA at 1.0 and the
same 2 fine bands. Generate. Then keep the **first copy** of every real frame and play the result at 24 fps.
Normal stretches play at normal speed and slowed stretches play as slow motion.

## Files

| file | what it is |
|---|---|
| `minimax_h3_variable_clock_runf_step2000_r16.safetensors` | the LoRA as trained: all 50 blocks |
| `minimax_h3_variable_clock_runf_step2000_r16_lateattn.safetensors` | for turbo LoRAs on a generic schedule (with Pika's own sampler the full file does as well): blocks 0–34 as trained, blocks 35–49 attention only (their MLP parts removed; 170 of 200 modules). Less soft than the full LoRA, keeps part of the temporal steadiness. |
| `minimax_h3_variable_clock_runf_step2000_r16_noblock49.safetensors` | the full LoRA without block 49. A smaller sharpness gain than `lateattn`. |

Why the trimmed copies: stacked with a turbo LoRA, this LoRA softens the result. A block-by-block test with Pika
6-step (de-rope stage 2, inject 0.45, one seed, one clip, generic 6-step Euler rather than Pika's own knots) found:

| run f part used, clock on | fine detail vs source | copy drift in held stretches (lower = steadier) |
|---|---:|---:|
| none (Pika + clock only) | 1.08 | 5.7 |
| all blocks (full LoRA) | 0.96 | 3.3 |
| blocks 0–34 only | 1.08 | 5.8 |
| blocks 35–49 only | 1.00 | 3.4 |
| all but block 49 | 0.99 | 3.4 |
| **blocks 0–34 + attention of 35–49 (`lateattn`)** | **1.01** | **4.1** |
| blocks 0–44 | 1.06 | 4.3 |
| full LoRA at strength 0.6 | 1.04 | 5.0 |

The early and middle blocks do almost nothing here. Blocks 35–49 carry both the temporal steadiness and the
softness, so no trim removes one without the other. `lateattn` is the best trade we found, and it also had the
lowest flicker of the clock-on runs. This LoRA does not touch H3's FinalLayer (where the PDD distillation heads
live); neither do the Pika and lightx2v LoRA files we checked.

## Videos

Every video is a side-by-side with the same seed (20260923), the same prompt and the same clock: **stock
H3 + clock (left)** vs **this LoRA + clock (right)**. The banner shows the current speed. The prompts are in `assets/`.

**Punching, long.** 1344x768, 124 generated frames: about 1.4 s normal, about 2.25 s of 2x, about 2 s of
4x, 2x, normal. Bob and weave, slips, jabs, an uppercut, a spinning backfist and a hook-cross.

- `assets/punching_long_kept.mp4`: the kept frames, the video you would actually ship.
- `assets/punching_long_all_frames.mp4`: every generated frame, including the repeats, with the
  real-frame / copy index in the banner.

<video controls muted playsinline preload="metadata" width="100%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-Variable-Clock/resolve/main/assets/punching_long_kept_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-Variable-Clock/resolve/main/assets/punching_long_kept.mp4" type="video/mp4">
</video>

**The same clip with nothing thrown away.** Every generated frame, repeats included, played at 24 fps. In the
2x and 4x stretches you can see what the model does inside each latent. Ideally the copies of one real frame
would be identical; in practice motion keeps flowing across them.

<video controls muted playsinline preload="metadata" width="100%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-Variable-Clock/resolve/main/assets/punching_long_all_frames_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-Variable-Clock/resolve/main/assets/punching_long_all_frames.mp4" type="video/mp4">
</video>

Single full-resolution videos (no tiling or labels), in `assets/raw/`:

| | kept frames (73, real time) | all generated frames (124) |
|---|---|---|
| this LoRA + clock | [punching_long_runf_kept.mp4](assets/raw/punching_long_runf_kept.mp4) | [punching_long_runf_all_frames.mp4](assets/raw/punching_long_runf_all_frames.mp4) |
| stock H3 + clock | [punching_long_stock_kept.mp4](assets/raw/punching_long_stock_kept.mp4) | [punching_long_stock_all_frames.mp4](assets/raw/punching_long_stock_all_frames.mp4) |

**Punching, short.** 1344x768, 56 generated frames (2.3 s, shorter than H3's usual 5–15 s range),
normal → 2x → 4x → 2x → normal.

- `assets/punching_short_kept.mp4`
- `assets/punching_short_all_frames.mp4`

<video controls muted playsinline preload="metadata" width="100%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-Variable-Clock/resolve/main/assets/punching_short_kept_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-Variable-Clock/resolve/main/assets/punching_short_kept.mp4" type="video/mp4">
</video>

## What it does and does not do yet

All of the following are from one seed per prompt; treat them as observations, not measurements.

- **Short punching clip:** the stock model jumps between poses and draws see-through ghost fists. The LoRA
  moves through the punches more continuously, with motion blur on two frames of the fastest punch.
- **Long punching clip:** both versions handle the full spinning backfist. The stock model's frames look
  crisper; the LoRA shows more motion blur on the fastest punches.
- **Anime rooftop run (not shown here):** at step 2,000 the running figure stays crisp through the 2x / 4x
  stretch, where the stock model and an earlier time-smear-trained LoRA both ghosted or blurred.
- **Live-action dancer (not shown here):** a clean, coherent spin. This is the first checkpoint where kept
  frames in the slowed stretch move noticeably less per frame than in the normal stretch.
- **Not learned yet:** at 2x and 4x the copies inside a latent should come out as the same held frame,
  because that is how training repeated them. They don't yet. They differ by about 7–9 out of 255 on average,
  against about 0.4 for true repeats through the VAE, so motion keeps going across the copies. Using only the
  first copy of each real frame hides this.
- It **still costs more compute** on slowed stretches. It does not yet save compute anywhere; that is the next
  research step (variable compression in both directions, and in space for detail such as faces and hands).

## ComfyUI workflows

The `workflows/` folder has three API-format workflows. Drag a `.api.json` file onto the ComfyUI canvas to
load it.

| file | what it does |
|---|---|
| `stage1_t2v_clock.api.json` | **Stage 1:** text-to-video directly on the variable clock (LoRA + H3 True Clock), then keeps the first copy of every real frame |
| `stage2_derope_clock_pika.api.json` | **Stage 2 with Pika 6-step (recommended):** Pika's own sampler (its six knots + `lcm`), the last 3 knots (inject 0.45). Steps 1–2 run Pika + this LoRA + H3 True Clock; the last step runs Pika alone on the normal clock. See "With Pika 6-step" below |
| `stage2_derope_clock.api.json` | **Stage 2:** the usual de-rope pass (H3 Time Smear → partial denoise → H3 Exact Recover), with the clock and the LoRA switched on |
| `source_generation.api.json` | stock H3 at normal speed; its output is the example input for stage 2 |

Both clock workflows save two videos: `*_kept` (the real-time result) and `*_all_frames` (everything generated).
`build_workflows.py` regenerates all three from the prompt and the clock map.

**Example outputs** from these exact workflows (ComfyUI, int8 pruned DiT, 20 steps, seed 20260923), in
`workflows/example_outputs/`:

- `source_native.mp4`: the stage 2 input.
- `stage1_kept.mp4` / `stage1_all_frames.mp4`
- `stage2_kept.mp4` / `stage2_all_frames.mp4`: frames 20–92 of the source, re-denoised at inject 0.45 (the MAINodes default).
- `stage2_pika_kept.mp4` / `stage2_pika_all_frames.mp4`: the Pika workflow on the same source (90 s on the test machine).

On the test machine (one RTX PRO 6000 Max-Q), each run took about 4–5 minutes.

**What you need**

- ComfyUI with the core MiniMax-H3 nodes.
- **ComfyUI-MAINodes, current `main`** ([github.com/matlowai/ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes),
  version **1.2.12 or later**, on the Comfy registry and GitHub `main`).
  It adds one optional input to **H3 True Clock**: `keep_fine_bands`. Set it to **2** for this LoRA.
  It keeps the two fastest temporal RoPE bands on the normal grid, which is how the LoRA was trained.
  At 0, the default, the node behaves exactly as before.
- The LoRA in `models/loras/minimax_h3_variable_clock/`, and the H3 models named in the loaders. The
  workflows use the int8 pruned DiT and the nvfp4 text encoder; swap in what you have.

**Clock maps.** The `hold_map` string is `{"holds": [...], "world_len": N}`, with one entry per real frame.
The LoRA was trained only on maps that line up with H3's latent grid:

- Pixel frames are packed in a (1, 4, 4, 4, 4) pattern per 17 frames.
- In each 4-frame latent, every real frame has the same hold: either 4 frames held once (normal), 2 frames
  held twice (2x), or 1 frame held four times (4x).
- Each single-frame latent holds one real frame once.
- The holds must add up to the clip length: 17k + 5 frames, e.g. 124.

The maps in `clock_maps/` follow this rule. This snippet builds one from a per-latent speed list:

```python
PATTERN = (1, 4, 4, 4, 4)
def clock_map(latents, speeds):            # speeds: one of 4, 2, 1 per 4-frame latent (4 = normal)
    holds, k = [], 0
    for i in range(latents):               # latents = 5k + 2, e.g. 37 for 124 frames
        if PATTERN[i % 5] == 1:
            holds.append(1)
        else:
            s = speeds[k]; k += 1
            holds += [4 // s] * s
    return {"holds": holds, "world_len": len(holds)}
```

Untested: hold values other than 1/2/4, holds that straddle latents, clocks faster than normal, and strengths
other than 1.0.

## With Pika 6-step

Use Pika's own sampler. MAINodes' Pika preset is: manual sigmas `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0`,
sampler `lcm` (fresh noise every step) and `pika_h3_flex_fl2va_v1`. With a generic 6-step Euler schedule everything
came out softer and less faithful.

Then turn the LoRA **and** the clock off for the last step. `stage2_derope_clock_pika.api.json` does this as two
samplers: Pika + this LoRA + H3 True Clock for knots 0.8 → 0.571 → 0.333, then Pika alone with DisableNoise from
0.333 to 0. The idea is that timing is decided in the early steps and detail in the last.

![inference levers with Pika's sampler](https://huggingface.co/MATLOWAI/MiniMax-H3-Variable-Clock/resolve/main/assets/pika_sampler_levers_table.png)

From one seed on the punching clip, against the full LoRA with the clock on every step, this has a little more
fine detail (1.19 vs 1.17 of the source), the best fidelity (20.95 dB) and the least flicker of the clock runs.
Turning off only the LoRA for the last step adds detail but costs steadiness and adds flicker. With this sampler,
the `lateattn` trim performs the same as the full LoRA, so the full file is fine here.

Side by side (Pika only / Pika + clock / Pika + LoRA + clock / LoRA + clock off for the last step):
[`assets/pika_levers_sbs_kept.mp4`](assets/pika_levers_sbs_kept.mp4).

Not yet checked: whether small, fast detail still slows down correctly when the last step runs on the normal clock.
This clip has little such detail.

## How this compares to DyRoPE / H3 True Clock without a LoRA

- **The clock signal is the same.** MAINodes' H3 True Clock, DyRoPE and this LoRA all give each latent a RoPE
  time span equal to the real time it covers, computed from the hold map. Hold maps carry over between them.
- **DyRoPE works around an untrained model.** The stock model never saw a stretched time grid. Feeding the
  stretched grid to every block corrects the speed but adds seam flash and jitter. DyRoPE therefore applies it
  to some blocks only, or fades it in or out over the denoise schedule, trading speed correction against
  artifacts. It is inference-only, and only for re-denoising a time-smeared clip.
- **This LoRA trains the model to read the clock.** It was trained on real frames repeated inside latents, so
  a 4x latent means "one real frame of motion" to it. Every block gets the clock at every step. The only split
  left is by frequency (`keep_fine_bands = 2`): the two fastest temporal bands stay on the normal grid, so the
  within-latent frame rhythm stays as trained. DyRoPE splits by block or by denoise step; this splits by RoPE band.
- **What that adds:** variable-speed **text-to-video from scratch (stage 1)**, and a stage 2 de-rope where small,
  fast detail is meant to slow down with the clock instead of regenerating at normal speed.
- **What it does not change yet:** the compute cost. A 4x latent still costs a full latent for one real frame,
  the same cost per kept frame as de-rope.

## Notes

- The Musubi-side tests used per-latent true-clock spans and `keep_bands = 2`, at LoRA strength 1.0, on the
  pruned bf16 DiT. The ComfyUI workflows were checked end to end on the int8 pruned DiT.
- Base model: MiniMax-H3 (FL2VA, pruned). See `LICENSE-MiniMax-H3` for the base model's terms.

`minimax_h3_variable_clock_runf_step2000_r16_lateattn.safetensors` sha256
`71e5f27c0891aacb9c9c06795a28d5cec41bf288582b47896835cc8041768b1a`

`minimax_h3_variable_clock_runf_step2000_r16_noblock49.safetensors` sha256
`8d51f66245b2fbe8354004a81c619198e5c86cdb2e775df3f743542446b253b1`

`minimax_h3_variable_clock_runf_step2000_r16.safetensors` sha256
`c01235bd0b7492856162b9965431007e9f9828e77454324d7f246546aeb1e652`
