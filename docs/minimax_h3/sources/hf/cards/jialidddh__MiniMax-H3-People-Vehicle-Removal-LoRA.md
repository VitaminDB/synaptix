# jialidddh/MiniMax-H3-People-Vehicle-Removal-LoRA

Источник: https://huggingface.co/jialidddh/MiniMax-H3-People-Vehicle-Removal-LoRA · скачано 2026-11-28
Загрузок: 299 · лайков: 15 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-11-15T09:42:44.000Z

---

---
language:
- en
- zh
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: adapter
pipeline_tag: image-text-to-video
tags:
- minimax-h3
- h3
- ref2va
- lora
- people-removal
- vehicle-removal
- video-editing
- panorama
- "360"
- 360-video
- equirectangular
datasets:
- junchaoh-cs/SolarWM-Data
extra_gated_eu_disallowed: true
extra_gated_prompt: >-
  This is a free community research release based on MiniMax H3. Before requesting
  access, read the MiniMax H3 Community License Agreement, including Section V
  and Exhibit A: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE.
  Access and intended use must comply with its applicable-territory provisions.
extra_gated_fields:
  Country of intended use: country
  I accept the MiniMax H3 Community License Agreement including Section V and Exhibit A: checkbox
  My access and intended use comply with the H3 applicable-territory provisions or are separately authorized by MiniMax: checkbox
---

# MiniMax-H3 People & Vehicle Removal LoRA

[中文](README_zh.md)

A **MiniMax-H3 Ref2VA LoRA for removing people and vehicles from perspective and 360° panoramic scenes**. Supply scene references in temporal order and a removal prompt to generate a background video with people and vehicles removed.

Joint training on panoramic video reconstruction and perspective removal pairs transfers removal capability to panoramic scenes while retaining panoramic video generation capability.

Released as a **free, non-commercial research contribution to the community**. The author does not commercialize this project.

## Download

[**Download v1**](MiniMax-H3-People-Vehicle-Removal-Ref2VA-v1-step1000.safetensors) · BF16 · rank 32 · 125 MiB

## Usage

Load the **BF16 Ref2VA** base with the [DiffSynth-Studio H3 inference examples](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/minimax_h3), then add the LoRA:

```python
from diffsynth.pipelines.minimax_h3_audio_video import ModelConfig

pipe.load_lora(
    pipe.dit,
    ModelConfig(path="MiniMax-H3-People-Vehicle-Removal-Ref2VA-v1-step1000.safetensors"),
    alpha=1.0,
)
```

| Setting | Recommended value |
|---|---|
| Base | MiniMax-H3 Ref2VA, BF16 |
| LoRA strength | 1.0 |
| References | Supplied in temporal order to both Qwen and VAE |
| Inference CFG / sampling steps | **1.0 / 50** |
| Frames / frame rate | 107 / 24 fps |
| Panorama output / reference short edge | 960 × 480 / 960 |
| Perspective output / reference short edge | 832 × 480 / 768 |
| Video / audio flow shift | 12 / 3 |
| Image-video / audio conditioning noise augmentation | 0.999 / 1.0 |
| Output audio | Off |

The [panoramic street prompt](prompts/removal_360_street.txt) and [perspective street prompt](prompts/removal_pv_street.txt) are the full task prompts used in evaluation, not trigger words. Adapt the scene description and camera motion to your input while retaining the removal instructions. Panorama reference frames are 0/35/106; perspective reference frames are 0/53/106. Reported results use the BF16 base.

## Video comparisons

**Perspective people and vehicle removal**

![Perspective people and vehicle removal](media/pv_street_03_seed123.webp)

[Full-rate MP4](media/pv_street_03_seed123.mp4) · [Actual prompt](media/pv_street_03_prompt.txt)

Base retains a foreground car at the start and vehicles on the right near the end; these vehicles are removed in the LoRA output.

**360° near-camera body and nadir capture artifact removal**

![360 near-camera body and nadir capture artifact removal](media/pano_street_05_seed42.webp)

[Full panoramic MP4](media/pano_street_05_seed42.mp4) · [Actual prompt](media/pano_street_05_prompt.txt)

Base leaves the camera operator’s head, shoulders, and arm prominently visible at the bottom of the panorama. The LoRA removes these nearby body parts. This example highlights body-part removal and nadir cleanup, with the difference visible directly in the full panorama.

Top: the actual three reference images. Bottom, left to right: **Source video | Base H3 | This LoRA**. The full source video is shown only for comparison; the model input is three sparse images. Each pair uses identical prompt, references, seed and BF16 settings; the only difference is whether the LoRA is loaded. MP4 files retain all 107 frames at 24 fps; animated previews are 8 fps.

## Results

Five panoramic and five perspective street scenes, seeds 42 / 123, with identical inputs and BF16 inference settings in each comparison:

| Metric | Base | This LoRA |
|---|---:|---:|
| Panorama sampled frames with people/vehicle detections ↓ | 81.9% | **25.7%** |
| Perspective sampled frames with people/vehicle detections ↓ | 15.7% | **0.0%** |
| Perspective SSIM against pseudo-clean targets ↑ | 0.325 | **0.392** |

Detection rates are measured with SAM3 on 21 sampled frames per output.

## Training

6,635 clips: 3,598 panoramic reconstruction clips and 3,037 perspective people/vehicle removal pairs. No paired 360° removal targets were used; panoramic removal comes from task transfer.

BF16 Ref2VA; attention `qkv_proj` / `out_proj` LoRA, rank 32; AdamW, learning rate 5e-5, global batch 16. The released checkpoint is step 1,000. Training uses inverse guidance-preservation loss with guidance 3.5: the no-grad near-empty branch uses the current LoRA, near-empty text, no Qwen images, and the same VAE references.

## Data and acknowledgements

Perspective pairs were derived from the SpatialVID subset of [SolarWM-Data](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data). Panoramic clips were curated through [PanoVid / PanoWan](https://github.com/VariantConst/PanoWan) from sources including 360-1M, Imagine360, Panonut360 and WEB360. We thank the original dataset authors and video creators. Raw training media and pseudo-clean targets are not redistributed in this repository.

## License

Trained on MiniMax H3. The weights are distributed under the [MiniMax H3 Community License](LICENSE); attribution and modification notices are provided in [NOTICE](NOTICE). The upstream territorial, use, redistribution and commercial provisions apply.

The author's purpose is non-commercial research and community sharing; this statement does not replace or amend the upstream weight license. Source-data terms remain applicable: [SpatialVID](https://github.com/NJU-3DV/SpatialVID#license) uses [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Original media rights remain with their respective owners. This release grants no additional commercial rights to source data.

Download and use are subject to the H3 agreement and its Acceptable Use Policy. Unless separately authorized by MiniMax, the standard territorial grant excludes the EU, UK, Republic of Korea and US. Use authorized inputs and label generated edits when sharing results.

Author: jiaronghan.
