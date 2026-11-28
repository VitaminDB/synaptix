# MATLOWAI/MiniMax-H3-ORB360-CardSpin

Источник: https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin · скачано 2026-11-28
Загрузок: 4426 · лайков: 65 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,comfyui · обновлено: 2026-09-30T03:20:15.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: adapter
tags:
  - minimax-h3
  - lora
  - video
  - ref2va
  - orbit
  - camera-control
  - novel-view
  - musubi-tuner
  - comfyui
pipeline_tag: image-to-video
---

# MiniMax-H3 ORB360: 360 orbit + CardSpin

Rank-32 LoRAs for MiniMax-H3 **Ref2VA**. Give it a photo, get a smooth clockwise 360-degree camera orbit around the
frozen subject (`ORB360_CW`), or, with the CardSpin files, turn the photo in space like a thin physical card
(`ORB360_CARDSPIN`).

| File | What it is | Use it for |
|---|---|---|
| `minimax_h3_orb360_step1500.safetensors` | **New.** ORB360 orbit LoRA, 1500 steps on high-resolution Blender orbits, camera heights from below to above, 1-4 labelled reference pictures | **Clean 360 orbits** (sharpest; our pick for multi-view use) |
| `minimax_h3_orb360_cardspin_v2_step50.safetensors` | **New.** CardSpin v2 = step 1500 + 50 card-spin steps | The card spin |
| `minimax_h3_orb360_cardspin_step50.safetensors` | CardSpin v1 (the original release) = the old 512 x 512 orbit LoRA + 50 card-spin steps | The original card spin |

Both CardSpin files can still orbit with the orbit prompt, but they are noticeably softer; see
[Which file for a clean 360?](#which-file-for-a-clean-360).

Powered by MiniMax H3.

## New: CardSpin v2

<video controls muted playsinline loop preload="metadata" width="100%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_cardspin_v2_step50_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_cardspin_v2_step50.mp4" type="video/mp4">
  <a href="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_cardspin_v2_step50.mp4">Download the CardSpin v2 clip (mp4)</a>
</video>

`minimax_h3_orb360_cardspin_v2_step50.safetensors`, one reference image (`examples/whitecat.png`), the prompt in
`prompts/cardspin_caption.txt`, 1024 x 768, 124 frames, 20 steps, seed 20260926.

## New: the 360 from ORB360 step 1500

<video controls muted playsinline loop preload="metadata" width="100%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_orbit_step1500_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_orbit_step1500.mp4" type="video/mp4">
  <a href="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_orbit_step1500.mp4">Download the step-1500 orbit (mp4)</a>
</video>

`minimax_h3_orb360_step1500.safetensors`, same image and seed, prompt `prompts/orbit_realscene_caption.txt`. It was
trained only on grey-backdrop Blender renders, and it keeps the real snowy scene around the cat.

## Which file for a clean 360?

The same orbit (same photo, prompt, seed, 1024 x 768, 124 frames) from four LoRAs. Numbers are per-frame image
statistics averaged over the clip; "detail vs step 1500" is the median per-frame ratio of the Laplacian variance (fine
detail) to the step-1500 clip.

| White cat 360 | Orbit step 750 (old, not in this repo) | CardSpin v1 (750 + 50) | **ORB360 step 1500** | CardSpin v2 (1500 + 50) |
|---|---|---|---|---|
| Fine detail (Laplacian variance) | 30.2 | 19.8 | **56.1** | 20.7 |
| Detail vs step 1500 | 0.54x | 0.34x | **1.00x** | 0.35x |
| Edge energy (Tenengrad) | 1098 | 809 | **1343** | 841 |
| High-frequency share | 0.0035 | 0.0028 | **0.0050** | 0.0029 |
| First frame vs the photo, PSNR / SSIM | 28.0 dB / 0.87 | 24.5 dB / 0.85 | **28.5 dB / 0.87** | 22.5 dB / 0.77 |

![Same frames from the four LoRAs, native-pixel crops](https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_orbit_detail_4way.jpg)

- The high-resolution training nearly doubles the fine detail (step 750 -> step 1500: 1.85x, higher in all 124 frames).
- **The 50 card-spin steps soften everything**, whichever orbit LoRA they start from: both CardSpin files keep only
  about a third of step 1500's fine detail and drift further from the photo in the first frame. The single card-spin
  training clip is a soft generated video, and 50 steps on it most likely pull the whole look towards it.
- So: **step 1500 for orbits, CardSpin only for the card trick.**

## The original v1 videos

CardSpin v1 (`minimax_h3_orb360_cardspin_step50.safetensors`), card-spin prompt:

<video controls muted playsinline loop preload="metadata" width="100%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_cardspin_step50_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_cardspin_step50.mp4" type="video/mp4">
  <a href="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_cardspin_step50.mp4">Download the v1 card-spin clip (mp4)</a>
</video>

Same file, orbit prompt:

<video controls muted playsinline loop preload="metadata" width="100%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_orbit_step50_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_orbit_step50.mp4" type="video/mp4">
  <a href="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/whitecat_orbit_step50.mp4">Download the v1 orbit clip (mp4)</a>
</video>

## How the card spin happened

We were testing an orbit LoRA (the ORB360 project) on real photos instead of the grey Blender renders it was trained
on, and fed it an 1867 portrait of Sir John Herschel by Julia Margaret Cameron. Instead of orbiting a man, it decided
the *photograph* was the object. This is the clip that started it all:

<video controls muted playsinline loop preload="metadata" width="60%"
       poster="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/herschel_origin_glitch_orbit750_poster.jpg">
  <source src="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/herschel_origin_glitch_orbit750.mp4" type="video/mp4">
  <a href="https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/assets/herschel_origin_glitch_orbit750.mp4">Download the original glitch (mp4)</a>
</video>

It was too good to leave as a one-off. So we took that single generated video, wrote a caption describing what
happens in it with timestamps, and trained **50 more steps** on top of the orbit LoRA using only that one clip. After
50 steps the effect transferred to photos it had never seen: other Cameron portraits, and a colour photo of a cat in a
hat. CardSpin v1 did this on the old 750-step orbit LoRA; CardSpin v2 repeats the same recipe on top of step 1500.

## Using it

**ComfyUI:** load one file with the standard **Load LoRA** node (model only, strength 1.0) on a MiniMax-H3 **Ref2VA**
model. These are kohya-style LoRAs (`lora_unet_*`, `lora_down`/`lora_up`/`alpha`); all 200 modules map onto ComfyUI's
MiniMax-H3 weights. Give it the reference image(s) and paste one of the prompts from `prompts/` as the text.

**musubi-tuner** ([kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner), v0.3.5 or later):

```bash
python src/musubi_tuner/minimax_h3_generate_video.py --task ref2va \
  --dit minimax_h3_ref2va_bf16.safetensors --prune_adaln \
  --video_vae minimax_h3_video_vae_fp16.safetensors --audio_vae minimax_h3_audio_vae_fp32.safetensors \
  --text_encoder qwen3vl_32b_minimax_h3_int8_convrot.safetensors --text_encoder_attn_mode sdpa \
  --lora_weight minimax_h3_orb360_step1500.safetensors --lora_multiplier 1.0 --lora_runtime_attach \
  --video_size 768 1024 --video_length 124 --infer_steps 20 --attn_mode sdpa --seed 20260926 \
  --prompt "$(cat prompts/orbit_realscene_caption.txt)" --ref your_photo.png --save_path out/
```

For the card spin, use `minimax_h3_orb360_cardspin_v2_step50.safetensors` and `prompts/cardspin_caption.txt`.

- The first `--ref` becomes `<Picture 1>`: the first and last frame of the orbit.
- Keep **124 frames at 24 fps** for these prompts: their timestamps (for example "At 1.708333 seconds (zero-based
  frame 41) ... 120 degrees") assume that length.
- Prompts:
  - `orbit_realscene_caption.txt`: one photo, keeps the photo's real surroundings.
  - `orbit_3ref_caption.txt`: three views of the same subject (front, 120 and 240 degrees clockwise), for example
    three renders or a turnaround sheet. With real side and back views the model does not have to invent them.
  - `orbit_underneath_3ref_example.txt`, `orbit_crossheight_front_back_example.txt`: the exact caption format step
    1500 was trained on, taken from two of our held-out tests (an orbit from 30 degrees below with no floor, and front
    and back pictures taken from 45 degrees with the orbit at 5 degrees). The numbers in them (camera distance, how
    much of the frame the subject fills) describe those test objects; edit the heights, times and angles for your own
    use. **Height prompts are still being trained:** they work on grey-backdrop renders like the ones step 1500 was
    trained on, but on real photos the orbit currently stays at the photo's own height (see Limitations).
  - `cardspin_caption.txt`: the card spin.
- Resolutions we have seen work: 800 x 800, 1024 x 768 and 1152 x 768 landscapes, 672 x 832 and 832 x 1024
  portraits.
- The prompts follow MiniMax's official Ref2VA prompt layout (`subject_definitions`, `summary`,
  `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`). Both audio sections are
  `N/A`, which asks for silence; without them H3 tends to invent a soundtrack.

## How well step 1500 orbits

Tested on three objects that were never rendered for training (a vintage film camera, a stylised character and a
stone cat statue), one seed per test, against Blender renders of the true orbit:

All of these tests use grey-backdrop Blender renders. Height control has not yet carried over to real photos (see
Limitations).

| Test | Result |
|---|---|
| Orbit at 20 degrees, 3 views (0/120/240) | Within one frame (2.9 degrees) of the true orbit on every frame |
| Orbits from below (30 and 10 degrees below the centre, no floor) | Right height on 118-124 of 124 frames, one clean turn |
| Pictures taken from 45 degrees, orbit asked for at 5 degrees | Low orbit on all three objects (most frames within 5-10 degrees of the target), one turn |
| High orbit (45 degrees) from one front picture | High orbit (30-45 degrees on most frames); the unseen sides are invented |
| Pictures taken from 45 degrees, orbit asked for at 30 degrees below | **Fails**: drifts back towards the pictures' height |
| Height changing during the clip (for example 45 above to 30 below) | **Fails**: the object tumbles instead of the camera orbiting |

The old 750-step LoRA mostly copies the height of the reference pictures: it gets the underneath orbits right when the
pictures are taken from below, but none of the "pictures from 45 degrees, orbit lower" cases.

## Training

All stages used musubi-tuner (v0.3.5, `--task ref2va`; small local wrappers for resumable segments) with the same settings:

| | |
|---|---|
| Base | MiniMax-H3 Ref2VA, BF16 transformer (`minimax_h3_ref2va_bf16` repack), loaded with `--prune_adaln` |
| Training adapter | [ostris/minimax_h3_training_adapter](https://huggingface.co/ostris/minimax_h3_training_adapter) `minimax_h3_ref2va_training_adapter_v1` via `--base_weights` (training only; not used at inference) |
| LoRA | rank 32, alpha 32 (`networks.lora_minimax_h3`) |
| Optimizer | AdamW, lr 1e-4 constant, max grad norm 1.0 |
| Precision | bf16 mixed, gradient checkpointing, SDPA |
| Batch | 1, no accumulation, video only (no audio loss), seed 20260926 |
| Hardware | one NVIDIA RTX PRO 6000 (96 GB) |

**ORB360 step 1500 (new), from scratch.** 23 assets (Poly Haven CC0 props and a few stylised character models),
rendered in Blender (Cycles) as 345 one-turn clockwise orbits at three lengths, each at the highest resolution that
fits in memory: 124 frames at 800 x 800, 73 frames at 1024 x 1024, and 22 frames at 1600 x 1600. Camera heights from
45 degrees below to 55 degrees above the subject's centre (the ones below with the floor switched off), tight and wide
framing, varied start angles. Each clip appears in three training rows with different reference pictures (the 73-frame clips lost their four-picture sets to fit in memory): two sets
taken from the clip itself (front only, front and back, front and side, front, side and back, a four-way turnaround,
or thirds at 0/120/240 degrees) and one set rendered from a different height than the orbit. The captions label every
picture with its exact time, frame and angle, and describe the camera's height, distance, framing and lens. 960 rows,
1500 steps, about 62 s per step (about 26 hours).

**CardSpin v2, +50 steps.** Initialised from the step-1500 weights (fresh optimizer). One training clip: the Herschel
glitch above (generated at 832 x 1024, trained at 672 x 832, 124 frames), with the original Herschel photo as the
single reference and the caption in `prompts/cardspin_caption.txt`.

**Original release (CardSpin v1).** Stage 1: four Blender assets, one 124-frame, 512 x 512 orbit each, three
reference pictures at 0, 120 and 240 degrees, 750 steps. Stage 2: the same 50 card-spin steps as above.

## Limitations

- The card spin was learned from a single example: it always turns the same way with roughly the same timing, and
  some subjects or seeds commit to the card less than others. Both CardSpin files soften the image (table above).
- Step 1500 was trained on grey-backdrop renders of 23 objects. It orbits real photos well in our tests, but was
  evaluated on a handful of images and three held-out objects, one seed each, not a large benchmark.
- **Camera-height (angle offset) prompts do not hold on real photos yet.** They work on grey-backdrop renders, but in
  our test an eye-level real photo prompted for an orbit 35 degrees above or 20 degrees below orbited at the photo's
  own height both times. A follow-up run training straight-on references, tilted orbits and vertical loops is in
  progress.
- Big jumps between the pictures' height and the orbit's height, and orbits whose height changes during the clip,
  do not work yet.
- From one picture, the sides the camera has not seen are invented. More pictures (front and back, a turnaround, or
  three views) fix that.

## Files

- `minimax_h3_orb360_step1500.safetensors`: ORB360 orbit LoRA, step 1500 (fp32, 597 MB)
- `minimax_h3_orb360_cardspin_v2_step50.safetensors`: CardSpin v2 (fp32, 597 MB)
- `minimax_h3_orb360_cardspin_step50.safetensors`: CardSpin v1, the original release (fp32, 597 MB)
- `prompts/`: the prompts described above
- `examples/whitecat.png`: reference image for the cat clips
- `examples/herschel_1867_cameron_met263166.png`: the Herschel portrait (Julia Margaret Cameron, 1867; The
  Metropolitan Museum of Art, Open Access, CC0), resized
- `assets/`: the videos above, their poster frames, and the four-way detail comparison
- `LICENSE` (MiniMax H3 Community License Agreement), `NOTICE`

## Base model and licence

These are LoRAs for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) by MiniMax; they do nothing without
their base weights, and the card-spin stages were trained on a clip generated with them. They are distributed under
the [MiniMax H3 Community License Agreement](LICENSE), including its acceptable-use, distribution, commercial and
territorial provisions. **They are not MIT.** Read the complete upstream terms; this repository does not expand them.
See `NOTICE` for attribution and the modification notice. Thanks to MiniMax for releasing H3, to Ostris for the
training adapter, and to kohya-ss for musubi-tuner.
