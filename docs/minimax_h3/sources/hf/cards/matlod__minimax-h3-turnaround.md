# matlod/minimax-h3-turnaround

Источник: https://huggingface.co/matlod/minimax-h3-turnaround · скачано 2026-11-28
Загрузок: 2487 · лайков: 45 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-08-08T03:15:04.000Z

---

---
license: other
license_name: minimax
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3
base_model: MiniMaxAI/MiniMax-H3
tags:
  - lora
  - minimax-h3
  - multi-view
  - character-sheet
  - turnaround
  - reference-to-image
---

# H3 Turnaround LoRA — character sheets from one image

*The first release of **Contact-Sheet diffusion**: a video model's timeline repurposed as a slot axis for coordinated image sets.*

One reference image + one instruction → **five coherent, progressively
rotated views of the same subject**, jointly denoised in a single pass,
each view an independently-decoded crisp still. A character turnaround
from one photo in ~10 s (512²) to ~57 s (1024²) on one GPU.

![real photo cat → five views](images/photo_cat_sheet.jpg)
*A real photograph (DIV2K) the model never saw, → five views. The LoRA
trained only on H3-generated subjects — real-photo identity transfer is
out-of-distribution and works.*

## The trick

H3 is a video+audio model. Its VAE stores video as one keyframe plus
bundles of four frames, and the transformer's superpower is "keep
everything consistent along the timeline." This LoRA hijacks that: five
**standalone image latents** are packed where a 5-position video latent
would go, with timeline positions stretched so the model believes they
span one continuous shot. The model thinks it is making a tiny video —
but every "frame" is a full independent image at a different camera
angle, so its temporal-consistency machinery gets repurposed into
cross-view identity. **The timeline becomes a slot axis.**

Proof it really is "video brain, image body": apply this LoRA to a
*normal* video generation and motion breaks — spinning windmill blades
render as superimposed discrete positions, because the LoRA has learned
that time = poses.

## Files

| file | step | character |
|---|---|---|
| `minimax_h3_five_view_512_s1500.safetensors` | 1500 | **default** — best rotation geometry |
| `minimax_h3_five_view_512_s400_instruct.safetensors` | 400 | better *instruction following* (e.g. "neutral studio background" override honored), shallower rotation |
| `minimax_h3_five_view_1024cont_s600.safetensors` | 600 | 1024-finetuned variant — see methodology |

All: rank 16, attention-only (qkv/out on 50 blocks, 100 modules), 63 MB,
`ref2va_pruned` partition. Strength 1.0 for rotation; ~0.7 trades rotation
for scene fidelity.

## Checkpoint methodology — how the files differ

One dataset (90 self-generated orbit clips, 30 subjects × 3 seeds; 75
train / 15 val split by subject), one LoRA config (rank 16 α 16,
attention-only), one de-distillation training adapter (ostris alpha,
train-time only). The files differ **only** in training resolution and
schedule:

- **`512_s400_instruct` / `512_s1500`** — the same single run at **512²**,
  lr 5e-5, adamw8bit, bs 1 × grad-accum 4, flowmatch/shift, saved at
  steps 400 and 1500. Early = instruction-following era; late = rotation
  era. (~6.8 s/step; the whole 1500-step run is ~2.8 h on one card.)
- **`1024cont_s600`** — took the step-400 weights above and **continued
  200 steps at 1024²**, lr 1e-5 (a short calibration tail, PixArt-style).
  Measured against the 512 checkpoints on held-out subjects at matched
  seeds: image quality at 1024/2048 is comparable (512-trained weights
  transfer to high-res inference surprisingly well — that's the headline
  finding), the studio-override instruction response is weaker than
  `s400`, and the 2048 early-arc back-loading is **not** fixed by 1024
  training (it's a resolution-dependent sampling effect, not a weights
  gap). Published for completeness and for users generating primarily at
  1024+.
- A third arm (**1024 from scratch**, lr 5e-5, 400 steps) is *not*
  published yet: at its endpoint it measures comparably to the arms above
  at every inference size, but it's only 400 steps in (vs 1500 for the
  512 run) — parked mid-experiment in favor of higher-priority work, and
  worth extending to 1500 at some point. Full judgment writeup in the
  research log.

## Resolution transfer (measured)

Trained at 512 — **inference transfers to 1024 and 2048**:

![512-trained LoRA at 2048](images/iguana_2048_sheet.jpg)
*512-trained weights, five fresh 2048² views (shown downscaled). No
upscaling anywhere in the pipeline — all views generated from noise.*

![time scales linearly with area](images/sheet_time_scaling.png)

| per sheet | 512²/slot | 1024²/slot | 2048²/slot |
|---|---|---|---|
| time (RTX PRO 6000 @ 450 W cap, 28 steps) | ~10 s | ~57 s | ~227 s |
| peak VRAM | 41 GiB | 42 GiB | 47 GiB |

## Before / after

![before after fennec](images/before_after_fennec.jpg)
![before after toybot](images/before_after_toybot.jpg)
*Top rows: base model with the same prompt/seed. Bottom rows: + this LoRA
(step-400 weights).*

## How to run

> **ComfyUI users: there's a node.**
> [ComfyUI-H3-ContactSheet](https://github.com/matlowai/ComfyUI-H3-ContactSheet)
> — two custom nodes bracketing a stock sampler, LoRA loaded via the
> stock loader, output verified visually indistinguishable from the
> reference sampler. **Do NOT
> just drop the LoRA into `loras/` with a normal workflow**: stock
> samplers can't do the five-slot packing (and on normal video
> generation this LoRA actively degrades motion). The Python CLI below
> is the alternative path.

**Requirements (this CLI path):** a **≥48 GB VRAM** NVIDIA card — the
toolkit keeps the whole quantized base (~41 GiB) resident with no
streaming offload. **The ComfyUI node does not share this floor**: stock
ComfyUI offloads to system RAM on smaller cards, so any setup that runs
MiniMax-H3 video can run sheets (slower). Common to both: ~43 GB disk
for the base weights (auto-downloaded from the public Comfy-Org
repackage, no login needed), Linux, Python 3.10+.

```bash
# 1. install ai-toolkit
git clone https://github.com/ostris/ai-toolkit && cd ai-toolkit
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. install the five-view extension (folder name must match exactly)
git clone https://github.com/matlowai/h3-contact-sheet \
    extensions/minimax_h3_five_view

# 3. download a LoRA from this repo, then:
.venv/bin/python extensions/minimax_h3_five_view/inference.py \
    --lora minimax_h3_five_view_512_s1500.safetensors \
    --ref  my_character.png \
    --out  out/ --size 1024        # 512 / 1024 / 2048 all work
# -> out/my_character_contact_sheet.png  ([ref | view0..view4])
#    out/my_character_slot0..4.png       (the five views, full size)
```

First run stalls at "Loading..." while the 43 GB downloads — that's
normal. Key dials: `--strength` (1.0 = rotation, 0.7 = fidelity),
`--seed` (deterministic per seed), `--prompt` (append
`", neutral studio background, constant lighting"` with the
`_s400_instruct` weights to replace the background).

**The [extension README](https://github.com/matlowai/h3-contact-sheet#inference--sheets-from-a-trained-lora-full-walkthrough)
has the full walkthrough with a troubleshooting section** (OOM, module
naming, unconsumed-weights warnings, under-rotation). Python API:
`generate_five_views(...)` in `five_view_sampler.py`.

## Honest limitations

- **Rotation axis follows the subject's pose.** Training orbits always
  started square-on frontal, so "orbit" anchors to the subject's facing
  direction — a subject photographed looking upward gets a tilted arc.
  Stating the starting view in the prompt helps.
- **Some subjects under-rotate.** H3 under-delivers prompted arcs on some
  classes; strength >1.0 buys more arc at some fidelity cost.
- **At 2048 the rotation back-loads**: early views cluster near the
  reference and the arc arrives in the last slots (measured; the effect
  is resolution-dependent attention dilution, not a weights issue).
- **Do not stack with video generation** — it degrades motion (see "The
  trick" above; this is inherent, not a bug).
- Slight subject re-posing between views is in-distribution; this is a
  view set, not a photogrammetry rig.
- Trained purely on **self-generated H3 corpus data** (90 text-prompted
  orbit clips, 30 subjects; frame 0 of each clip is its reference).
  Angles were measured, not trusted from prompts.

## Roadmap — what we're building next

Same machinery, new tricks (the slot packing is agnostic to what varies
across slots):

- **ComfyUI custom node** — ✅ shipped:
  [ComfyUI-H3-ContactSheet](https://github.com/matlowai/ComfyUI-H3-ContactSheet).
  Drag-in workflow file still to come.
- **Tiled joint-diffusion upscale** — slots as overlapping *zoomed tiles*
  of one image instead of rotated views, jointly denoised so tiles agree,
  anchored to the low-res source, with true spatial RoPE offsets per
  tile. Goal: 4K+ reassembled outputs from ~1 MP passes — the same
  in-context-regeneration idea MiniMax uses for its own 2K, spatialized.
- **Fast-motion restoration** — a program to fix video motion blur/
  morphing using self-manufactured slow-motion accumulation pairs and
  motion-targeted repaint; the five-slot testbed doubles as its eval rig.
  (One arm already killed by a cheap pre-test — research log has the
  numbers.)
- v2 training recipe: caption mixture (heals the instruction-vs-rotation
  checkpoint split), ref dropout (text-only mode for free), measured-angle
  captions, 10-slot sheets, 180°/360° spans.

Follow along / argue with us: the full research log is public at
[research/five-view-docs](https://github.com/matlowai/ai-toolkit-1/tree/research/five-view-docs/docs/five-view-research).

## Provenance

Training data: self-manufactured (H3 generates its own supervision —
text-prompted constant-rate orbit clips at 25 steps, no LoRA). Recipe,
measurements, and the full research log are public:
[research/five-view-docs](https://github.com/matlowai/ai-toolkit-1/tree/research/five-view-docs/docs/five-view-research).
Base model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
(license follows the base model's).
