# rehan-fal/minimax-h3-360-equirect-lora

Источник: https://huggingface.co/rehan-fal/minimax-h3-360-equirect-lora · скачано 2026-11-28
Загрузок: 0 · лайков: 24 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-09-30T06:01:01.000Z

---

---
license: other
license_name: minimax-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3
tags:
  - lora
  - text-to-video
  - video
  - audio
  - minimax
  - minimax-h3
  - "360"
  - 360-video
  - equirectangular
  - panorama
  - vr
  - fal
pipeline_tag: text-to-video
library_name: peft
---

# MiniMax H3 · 360° equirectangular LoRA

A LoRA for **MiniMax H3** that makes the model generate **360° video**: the whole sphere around the viewer
unwrapped into one equirectangular frame, straight ahead in the centre, directly behind at the left and right
edges, with H3's native audio. Squeeze the 21:9 output to 2:1, tag it as 360, and it plays in a Quest / DeoVR /
Skybox / YouTube 360 as a full sphere you can look around.

![base vs A vs B](samples/base_vs_A_vs_B.png)

*Same seed, same prompt. Left: base H3 (an ordinary wide shot). Middle: the alternate LoRA (A). Right: this LoRA
(B). The LoRAs produce true equirect geometry: curved horizon, stretched poles, left and right edges that meet.*

## Use it on fal

```json
POST https://queue.fal.run/minimax/h3/text-to-video/lora
{
  "prompt": "eqr360 360-degree equirectangular panorama video: the whole sphere around the viewer unwrapped into one frame, straight ahead in the centre, directly behind at the left and right edges which join seamlessly, the horizon running straight across the middle. Standing on a mossy path in the middle of a lush rainforest, giant ferns all around, a waterfall straight ahead, sunbeams through the canopy overhead, a stream behind, birdsong and rushing water",
  "loras": [{"path": "https://huggingface.co/rehan-fal/minimax-h3-360-equirect-lora/resolve/main/h3-360-equirect-lora-v1.safetensors", "scale": 0.75}],
  "aspect_ratio": "21:9",
  "resolution": "4K",
  "duration": 5,
  "prompt_expansion_mode": "disabled"
}
```

* **Pass the direct file URL** as above (use `…/h3-360-equirect-lora-v1-alt-896x384.safetensors` for the alternate).
  On fal, the repo-id form (`"path": "rehan-fal/minimax-h3-360-equirect-lora"`, with or without `weight_name`)
  loads the weights differently: with the same seed it gave a different video, while the URL form reproduced the
  trained file's output exactly (tested 2026-09-29).
* **Trigger:** `eqr360`, then the layout sentence above (it was in every training caption), then the scene.
  Describe what is ahead, to the sides, behind and overhead; the viewer can look everywhere.
* **Strength:** 0.75 is the sweet spot. On two same-seed test scenes it held the geometry slightly better than 1.0
  (seam correlation 0.96 vs 0.92, pole spread 0.23 vs 0.30) with no visible loss of detail; 1.0 also works.
* **Resolution:** a sphere spreads pixels thin, so go as high as you can. 4K gives 4368×1920 (about 11 px per
  degree once packaged at 4096×2048); a 10 s clip takes roughly 4–9 min and costs $2.00. 2K (2912×1280) is the
  middle ground, and 768P renders in about a minute for previews. 2K and 4K upscale the 768P pass, so a 768P
  preview with the same seed and duration shows you the scene you'll get.
* **Duration:** 5–15 s all hold the geometry. 5 s gives the tightest seam. Longer renders can drift darker in their
  last ~3 s on some scenes, and seam quality varies over a long clip, so for the cleanest result keep the first
  ~7.5 s of a 10 s render.

## Make it headset-ready

```bash
# 21:9 -> exact 2:1 equirect, then Spherical V2 360 metadata (mono). 4K output -> 4096x2048; 2K -> 2912x1456.
ffmpeg -i out.mp4 -vf "scale=4096:2048:flags=lanczos,setsar=1" -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a copy tmp.mp4
python spatialmedia -i --v2 -p equirectangular tmp.mp4 out_360.mp4
```

The source repo's packager also softens the wrap-around seam behind the viewer (H3 has no wrap-around attention,
so the two edges can disagree slightly), and can trim a long render and make it loop seamlessly by crossfading its
first second under the end, which helps because headset players loop short clips. Every clip in `samples/` has the
softened seam; the 4K ones are also trimmed (not looped).

## 4K samples

![4K samples, three directions each](samples/4k_views.jpg)

*Each row is one 360 clip seen three ways: straight ahead, 100° right and 100° left.*

Headset-ready 4096×2048 360 files, rendered at 4K with strength 0.75, seed 7 and duration 10, keeping the first
7–9 s. Put them on a Quest, or any 360 player, and look around:

| File | Scene (the prompt after the trigger and layout sentence) |
| --- | --- |
| `samples/4k_reef_360.mp4` | Underwater in a vibrant coral reef in clear turquoise water, a sea turtle gliding slowly past straight ahead, schools of bright tropical fish swirling on every side, colourful corals and swaying sea fans all around, shimmering sunbeams and the rippling surface overhead, the deep blue fading into the distance behind, soft bubbles and muffled underwater ambience |
| `samples/4k_tokyo_360.mp4` | Standing in a narrow alley in Tokyo at night in the rain, glowing neon signs and red paper lanterns on both sides, steam rising from a tiny ramen stall straight ahead, people with clear umbrellas walking past, wet pavement reflecting pink and blue light, a train rumbling across a bridge behind, rain pattering, sizzling food and distant city sounds |
| `samples/4k_balloons_360.mp4` | Drifting high above Cappadocia at sunrise, dozens of colourful hot air balloons floating all around at different heights, rocky fairy-chimney valleys far below, the golden sun rising straight ahead, soft pastel sky overhead, the roar of a balloon burner now and then and a gentle wind |
| `samples/4k_aurora_360.mp4` | Standing on a frozen snowy lake in Lapland at night, vivid green and violet northern lights rippling across the whole sky overhead, a snowy pine forest along the shore on every side, a small glowing wooden cabin behind, bright stars, crisp quiet with a soft wind |

## Measured geometry

Averaged over 6 prompts × 6 frames per clip (`training/*.json`):

| | seam correlation | seam difference | pole spread |
| --- | --- | --- | --- |
| real 360 training footage | 0.92 | 8.4 | 0.17 |
| base H3, no LoRA | 0.55 | 41.6 | 0.89 |
| alternate LoRA (A) | 0.95 | 7.8 | 0.37 |
| **this LoRA (B)** | **0.95** | **6.8** | **0.26** |

Seam correlation compares the left and right edge columns, which should be the same meridian. Pole spread
measures how well the top and bottom rows collapse to a single point.

## Training

fal's hosted trainer (`minimax/h3/t2v/trainer`): PEFT LoRA rank 32 on the attention projections of all 50 DiT
blocks plus the 2 text token-refiner blocks, joint video+audio loss, AdamW 2e-4 linear decay, 3500 steps.

| | this LoRA (B) | alternate (A), `…-alt-896x384.safetensors` |
| --- | --- | --- |
| Bucket | 1120×480, 73 frames (3 s) | 896×384, 124 frames (5.2 s) |
| Data | 181 clips, identical | 181 clips, identical |

**Data:** 360 candidates drawn from the [360-1M](https://github.com/MattWallingford/360-1M) curated index, one
video per channel and stratified across 13 categories. Only YouTube's HLS renditions were downloaded, because the
DASH "mesh" renditions of 360 video are cubemaps. Of 327 downloaded clips, 281 passed seam and pole checks for true
equirect. Gemini, via fal, captioned each clip across all directions and screened it; clips with text overlays, a
prominent camera operator, game captures, near-black frames or non-360 layouts were dropped. That left 181.

## Known limits

* A faint seam can remain directly behind the viewer on busy scenes; package with the seam blend.
* The training footage skews toward handheld and first-person uploads, so people close to the camera and a
  slightly high horizon can appear.
* Monoscopic only (no stereo depth). The sibling
  [VR180 stereo LoRA](https://huggingface.co/rehan-fal/minimax-h3-vr180-sbs-lora) covers depth in front of you.

## Licence and data notice

Weights derive from MiniMax H3 and are subject to the
[MiniMax Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Training clips came
from YouTube (41 of the 360 candidates were CC-BY, the rest standard licence) and are not redistributed; the
adapter learns a projection and layout, not the content of those videos.
