# MATLOWAI/MiniMax-H3-Motion-Adapter

Источник: https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter · скачано 2026-11-28
Загрузок: 5711 · лайков: 50 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,comfyui · обновлено: 2026-09-22T02:00:25.000Z

---

---
license: mit
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: adapter
tags:
  - minimax-h3
  - lora
  - video
  - comfyui
  - motion
library_name: diffusers
pipeline_tag: image-to-video
---

# MiniMax-H3 motion adapter (pilot, r16)

A rank-16 LoRA that makes the de-rope pass in
[ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes) behave better
on fast motion. It needs that pass; it is not a general-purpose motion LoRA.

## What it looks like

One 5 s clip, four ways. Every panel has its numbers burned in.

<video controls muted playsinline preload="metadata" width="100%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter/resolve/main/assets/adapter_t2c_insert_quad_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter/resolve/main/assets/adapter_t2c_insert_quad.mp4" type="video/mp4">
  <a href="https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter/resolve/main/assets/adapter_t2c_insert_quad.mp4">Download the comparison clip (mp4, 2.9 MB)</a>
</video>

|   | left | right |
|---|---|---|
| **top** | the source plate, straight out of H3 | the whole clip re-generated on a stretched clock, base model, 139.9 s |
| **bottom** | only the fast span re-generated, no adapter, inject 0.45. 63.6 s, alternation 0.370, rate 1.416 | the same span **with this adapter** at strength 0.75, inject 0.30. 49.9 s, alternation **0.134**, rate **1.011** |

Watch the bottom two panels, not the stills. Without the adapter the fighters
advance and snap, advance and snap, frame to frame. With it they keep moving.
Rate 1.011 means the motion comes out at the speed the shot called for; 1.416
means the pass invented about 40 percent more movement than it should have.

Two things this exhibit is not. It is not the adapter on its own: the stretched
clock comes from the [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes)
de-rope pass, and the adapter only changes how the model behaves once that pass
has stretched it. Load this file on a plain text-to-video render and you get the
costs listed below with none of the benefit. And the settings shown here are
0.75 / inject 0.30, the lower-invention cell; see "Settings that work" for when
to prefer strength 1.0.

## Reproduce the four-tile fight example

The original source plate, three API workflows, a runner, and a freshly verified
adapter result are now in [examples/fight_quad](examples/fight_quad/README.md).
For the normal ComfyUI canvas, download [the winner workflow](examples/fight_quad/adapter.canvas.json)
or [the complete four-tile workflow](examples/fight_quad/fight_quad.canvas.json),
then drag the JSON into ComfyUI.

Start with the included **seedhunt_20260963_00001_.mp4** source; the top-left tile
is that original video. The other three tiles are the full-clip base pass,
windowed v3.1 without the adapter, and the adapter at **0.75 / inject 0.30**.

[Download the source plate](examples/fight_quad/seedhunt_20260963_00001_.mp4) ·
[Download the verified adapter result](examples/fight_quad/winner_verified_20260921.mp4) ·
[Reproduction instructions](examples/fight_quad/README.md) ·
[Download the complete example ZIP](examples/fight_quad_example.zip)

The September 21 verification completed in **65.4 s** and decoded successfully
(124 frames, 1152 × 640, 24 fps). This is a fresh functional check, not a new
matched-cache benchmark. The original comparison's 139.9 / 63.6 / 49.9 s labels
remain historical measurements. Both supplied MP4s contain their API graph in
`prompt` metadata; separate JSON files are included for convenient reuse.

## How it is actually wired, and what the adapter is not responsible for

Same clip, same seed, same graph in all four tiles. An anime sword ring, which
is the content type it helps most.

<video controls muted playsinline preload="metadata" width="100%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter/resolve/main/assets/adapter_mage_pinned_quad_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter/resolve/main/assets/adapter_mage_pinned_quad.mp4" type="video/mp4">
  <a href="https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter/resolve/main/assets/adapter_mage_pinned_quad.mp4">Download the comparison clip (mp4, 2.7 MB)</a>
</video>

|   | left | right |
|---|---|---|
| **top** | the source plate | the burst window re-generated at denoise 0.70 with **nothing anchoring it**, so it drifts off the plate and jumps at both splices |
| **bottom** | the same graph with the window's **first and last frames pinned**. Two extra wires, no extra time (10m41s either way): jitter 0.103 to **0.006**, entry jank 2.05 to 1.58, invented objects 109 to **60** | pins **plus this adapter** at 0.75, one `LoraLoaderModelOnly`: alternation 0.129 to **0.093**, objects 60 to 54, for 39 s more wall time |

The order matters, and it is the opposite of what people expect. **The pins do
most of the work**, and they are free: pinning the window's boundary frames to
the plate frames they must butt against is what stops the splice jump, adapter
or no adapter. The adapter then converts spare capacity into smoothness on top
of that. If your seams are jumping, pin first and only then reach for this file.

The same tile also shows the cost honestly: warm colour goes 1.04 to -1.59 with
the adapter on. That is the colour muting described below, visible in the one
exhibit where everything else improves.

Play to the end for a card listing every setting: window, hold map, denoise,
sampler, seed, quantisation and wall time per tile. (It labels the LoRA by its
training name, `p4_pilot_k100.safetensors`; that is this file, published here as
`minimax_h3_motion_adapter_pilot_r16.safetensors`.)

More before/after clips, including the two cases where it makes things worse,
are on the [demo page](https://matlowai.github.io/ComfyUI-MAINodes/#adapter).

## About this release

It is a pilot: rank 16, trained for a few hours on
38 clips we generated ourselves. It works, it is measurable, and it has
clear rough edges listed below. We are publishing it as an intermediate
release while we try something considerably more ambitious that may or may
not work (an all-in-one adapter that internalises the whole two-pass
pipeline). If that lands it will replace this file; if it does not, this
one still does a useful job. Part of that work is finding out whether an
adapter can be made to actually assign more compute to the burst on its
own (a variable-rate attention behaviour learned inside the model, rather
than the nodes stretching the clock for it); this pilot does not do that,
see the mechanics note below.

## What it does

MAINodes' Motion Lab regenerates the fast span of a clip on a stretched
clock (the "de-rope") and recovers it back to real time. That pass has a
known bad habit: on very fast motion the base model over-produces, and the
result alternates between advance and snap frame to frame. This adapter
was trained on that exact task (frames held out of the middle of a burst,
the model asked to fill them back in), and it teaches the base model to
spend the extra clock on smoothness instead of invention. To be precise
about the mechanics: the LoRA is a plain rank-16 weight delta and adds no
compute and needs no custom node. The extra clock comes from the nodes
(the de-rope stretches the burst into more latent tokens); the LoRA
changes how the model behaves on that stretched clock, because it was
trained in that exact situation.

Measured on four clips the adapter never saw during training (a fight,
an anime sword ring, a character swap, a dialogue close-up), against the
same graph without the adapter and against a same-seed noise floor:

- advance/snap alternation improved on 4 of 4 clips (4.6 to 6.1 times the
  floor)
- frame-to-frame over-production dropped from 1.76 to 2.93 down to 1.09
  to 1.23 on all four
- blind operator picks, adapter vs no adapter, same seed: 7 for, 0
  against, 1 no preference
- it transfers to the Ref2VA checkpoint (span, alternation and jitter
  wins of the same size on the two clips we tried), so one file serves
  both fl2va and ref2va graphs
- one end-to-end run of the shipped graph on a scene it had never seen
  (the fight choreography moved to a neon rooftop, same seed for both
  arms): alternation 0.24 to 0.19, jitter 0.09 to 0.04, fewer invented
  objects, and it kept the plate's setting where the no-adapter pass
  rewrote the signage. One clip, and each graph at its own shipped inject
  (0.70 vs 0.45), so read it as a smoke test, not a study

Where it is not great (also measured):

- it costs about 1 dB of anchor fidelity on native keyframes and it
  over-corrects calm chains (it smooths things that did not need
  smoothing); keep it off on calm content
- it mutes strong colour and thins particle effects. A prismatic creature
  came out as a plain calico on the Ref2VA checkpoint (mean saturation 46
  to 38 at denoise 0.40, 54 to 40 at 0.70) and again on the fl2va
  checkpoint when the whole clip was de-roped at 3x. Worst on Ref2VA and
  on full-clip passes; a windowed de-rope on the same creature kept its
  colour. If the subject is the colour, lower the strength or accept the
  trade
- it is anime-heavy in effect: on the one very fast anime clip we
  measured it wins clearly at every setting; on live-action character
  work it needs the lower inject values below

## Settings that work

- Strength 1.0. Strength and inject turned out to be the same dial. 1.0
  wins on alternation in every paired cell we ran (8 of 8), while 0.75
  invents fewer objects (7 of 8), so 1.0 is the smoothness setting and
  0.75 the lower-invention one; the old 0.75 to 0.80 band was
  in-distribution tuning on one fight plate and does not hold elsewhere.
- Inject is the dial. Starting points: 0.45 for character or dialogue
  where the base model over-produces hard; 0.30 where identity or props
  are the deliverable and the base is only mildly inflated; on very fast
  anime it does not matter (every setting wins).
- Keep the tail guide on. Turning it off costs identity about 3 times the
  floor across seeds and checkpoints, adapter or not.
- Apply it to the de-rope pass only, not to the first (text-to-video)
  pass.

## Use in ComfyUI

Put the file in `models/loras/minimax_h3/`. Load it with a stock
`LoraLoaderModelOnly` at strength 1.0 on the model that feeds the de-rope
pass. A ready graph is in ComfyUI-MAINodes:
`examples/motion_pipeline_adapter_api.json` (API format, loads through the
API and through the frontend's API import). The stock
`LoraLoaderModelOnly` is fine for this file; the key mismatch that bites
turbo LoRAs does not apply here.

## Training facts

- Base: MiniMax-H3 (fl2va), rank 16, alpha 16, 375 steps, lr 1e-4,
  adamw8bit, bf16, about 4.6 h on one GPU
- Data: 38 holdout-infilling items cut from 15 clips we generated with
  MiniMax-H3 (physics scenes, fights, chains, a dialogue plate); no real
  footage, no third-party footage
- Task: hold out the frames inside a motion burst, keep the rest as clean
  context (kept_label 1.0), predict the held tokens
- Trainer: our ai-toolkit extension for H3 (span-aware packing verified
  bit-exact against ComfyUI)

## The temporal-expansion ladder (new, 2026-08-24)

The "considerably more ambitious" follow-up mentioned above has produced its
first shippable piece. The pilot learned its task from targets the model
generated itself; these checkpoints retrain the same recipe against real
intermediate frames, and every training step is published so you can pick
your own point on the curve.

**What changed.** Same rank-16, attention-only construction, warm-started
from the pilot. The training targets are now real: windows cut from the
public GOPRO_Large deblurring dataset (240 fps source), decimated to the
working frame rate and re-expanded, so the held tokens are scored against
frames a real camera actually captured rather than an interpolation. The
world clock in training is truthful (each token labelled with the real time
it spans).

**Measured, on scenes the adapter never saw** (four camera shoots fully
disjoint from training, held-token error in latent space, lower is better;
the no-adapter baseline is 1.23 and the plain interpolation the pass starts
from scores 0.26):

| step | 25 | 100 | 175 | 250 | 375 |
|---|---|---|---|---|---|
| error | 0.158 | 0.133 | 0.116 | 0.105 | 0.086 |

The curve never turns: more steps keep measuring better, through the last
checkpoint. And here is the honest part: **the numbers and the eye disagree
about where to stop.** Step 100 was the pick in playback: fast motion comes
out smooth and the clip still feels like itself. Step 375 measures best by
a wide margin, but in side-by-side viewing it can read over-smooth, as if
the motion had been graded. On a static-content probe the family does not
invent motion (it lands below the interpolation floor rather than above
it), so the disagreement is about feel, not correctness. We publish every
25th step so you can find your own point; if you only take one file, take
`warm100`.

**Watch instead of reading.** Six arms of one clip, every panel with its
settings and wall time burned in; the bottom-right panel is the full
de-rope end to end for cost comparison:

<video controls muted playsinline preload="metadata" width="100%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter/resolve/main/assets/temporal_expansion_2x3_tile_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter/resolve/main/assets/temporal_expansion_2x3_tile.mp4" type="video/mp4">
  <a href="https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter/resolve/main/assets/temporal_expansion_2x3_tile.mp4">Download the comparison tile (mp4, 11 MB)</a>
</video>

**Use.** Exactly like the pilot: drop a file from `temporal_expansion/`
into `models/loras/minimax_h3/`, load with a stock `LoraLoaderModelOnly`
at strength 1.0 on the model that feeds the de-rope or windowed-insertion
pass. Not for the first text-to-video pass.

**Files.** `temporal_expansion/minimax_h3_temporal_expansion_warm100_r16.safetensors`
(the playback pick, sha256 cbb23ad4...) and
`minimax_h3_temporal_expansion_step025_r16.safetensors` through
`step375_r16.safetensors`, every 25 steps of the same run (step375 sha256
28f3d41d...). 63 MB each.

**Training facts.** Warm-started from the pilot above; 375 steps, lr 1e-4,
adamw8bit, bf16; 48 samples derived from GOPRO_Large (Nah et al., the
public deblurring benchmark) plus the pilot's self-generated set for the
difficulty gate; the trainer's span-aware packing verified bit-exact
against ComfyUI. Unlike the pilot, this data includes real footage; the
source dataset is public and widely used for research, and the derivation
(decimate, re-expand, score held tokens) is described so it can be
reproduced or swapped for your own footage.

## Base model and licence

This is a LoRA for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
by MiniMax; it does nothing without their base weights and is trained on
clips generated with them. The adapter weights and the graph here are
released under the MIT licence (`LICENSE`); use of the base model, and so
of this adapter with it, is subject to the MiniMax H3 Community License
Agreement, included as `LICENSE-MiniMax-H3` (copied from the MiniMax-H3
repository at commit 42ed227e). Thanks to MiniMax for releasing H3.

## Files

- `minimax_h3_motion_adapter_pilot_r16.safetensors` (63 MB)
- `motion_pipeline_adapter_api.json` (the same graph as in the node pack)
- `LICENSE` (this repository, MIT) and `LICENSE-MiniMax-H3` (the base
  model's community licence)

Examples and before/after clips, including the colour-muting case, are on
the demo page: https://matlowai.github.io/ComfyUI-MAINodes/#adapter
The node pack, the example graph and the tuning notes are at
https://github.com/matlowai/ComfyUI-MAINodes (see README, "The motion
adapter (pilot)", and TUNING.md).
