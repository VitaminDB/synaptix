# akatz-ai/MiniMax-H3-Character-Swap-LoRA

Источник: https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA · скачано 2026-11-28
Загрузок: 129428 · лайков: 1021 · базовая: Comfy-Org/MiniMax-H3 · теги: lora,comfyui,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-10-17T23:41:56.000Z

---

---
language:
- en
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model: Comfy-Org/MiniMax-H3
base_model_relation: adapter
pipeline_tag: video-to-video
library_name: diffusion-single-file
datasets:
- akatz-ai/H3-Character-Swap-v1
tags:
- minimax-h3
- lora
- character-swap
- video-editing
- ref2va
- comfyui
- ai-toolkit
---
# H3 Character Swap Lora

[![Follow @akatz_ai on X](https://img.shields.io/badge/Follow-%40akatz__ai-111111?style=flat-square&logo=x&logoColor=white)](https://x.com/akatz_ai)
[![Patreon](https://img.shields.io/badge/Support-Patreon-FF424D?style=flat-square&logo=patreon&logoColor=white)](https://www.patreon.com/Akatz)
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-EA4AAA?style=flat-square&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/akatz-ai)
[![Ko-fi](https://img.shields.io/badge/Tip-Ko--fi-FF5E5B?style=flat-square&logo=kofi&logoColor=white)](https://ko-fi.com/akatz)

![Character Swap: original video, character reference, and swapped result](assets/character-swap-showcase.gif)

**Original → Reference → Character swap.** Four selected examples.

**V1.5 is the recommended checkpoint for new character swaps.** It replaces a
selected person in a source performance with a character from an image reference.

- **[Download V1.5](h3_character_swap_v1.5.safetensors)** — recommended, strength 1.0.
- **[Download V1](h3_character_swap_pro4500_1000.safetensors)** — retained for
  existing workflows and reproducible comparisons.

## Versions

| Version | File | Status |
| --- | --- | --- |
| H3 Character Swap V1.5 | `h3_character_swap_v1.5.safetensors` | Recommended |
| H3 Character Swap V1 | `h3_character_swap_pro4500_1000.safetensors` | Previous release |

## Native workflow with H3 Relay 2.0

[Download the canvas workflow](examples/H3-Character-Swap-Relay-V2.json) or
[the API prompt](examples/H3-Character-Swap-Relay-V2.api.json). These use the
recommended V1.5 adapter, native ComfyUI model loading, reference conditioning,
SAM masking, sampling, loops and saving, plus reusable Relay timeline helpers.
The model weights are unchanged by this workflow update.

Install [H3 Relay 2.0](https://github.com/akatz-ai/h3-relay) and follow
[the workflow guide](examples/README.md) for models, runtime and input paths.

## Use in ComfyUI

1. Obtain the H3 Ref2VA base model, text encoder and VAEs separately.
2. Place `h3_character_swap_v1.5.safetensors` in your configured LoRA directory
   and select it in a compatible model-only LoRA loader at strength **1.0**.
3. Provide the source performance as `<Video 1>` and character reference as
   `<Picture 1>`. Describe the person to replace and the reference identity.
4. Begin with a short continuous shot at 24 fps. Use your runtime's supported H3
   frame grid and review identity, motion, expressions and joins before extending.

Example:

> Replace the selected person in <Video 1> with the character in <Picture 1>.
> Keep the reference character's face, hair and clothing. Follow the source
> performance's pose, gaze, expression and movement. Preserve the scene and the
> other people.

The H3 Character Swap tool handles masks, optional scene anchors, source-audio
conditioning and continuation history. Its tested recipe uses the Ref2VA INT8
base, this character adapter at strength 1.0, and the separate
`minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` at strength 1.0:
8 steps, CFG 1, `er_sde`, `normal`. The LoRA itself is not a standalone model
and does not include the base, training assistant, VAEs or Turbo weights.

## V1 vs V1.5 comparison

<video controls playsinline src="https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/resolve/main/assets/character-swap-lora_v1-vs-v1-5.mp4" width="100%"></video>

[Download the comparison video](assets/character-swap-lora_v1-vs-v1-5.mp4).
V1 is on the left; V1.5 is on the right. These selected examples show cases
where V1 loses the reference identity or returns to the source appearance,
while V1.5 retains more of the character. They illustrate improvements on
these shots, rather than an aggregate benchmark.

## Examples and limitations

This remains an experimental character-swap model. Identity, source timing,
expressions, occlusions and cuts can still drift. A recommended default does not
mean every shot improves over V1. Selected comparisons show improved retention
of stylized faces, hair and character identity. Broad evaluation is still in
progress; these observations are not a claim of uniform improvement.

The accompanying tool retains the original source soundtrack in its export.
That is an application feature, not evidence that the LoRA reproduces audio
accurately or that lip sync is exact. Long segment controls in the tool are not
a guarantee of long-shot quality or suitability for every GPU.

The existing showcase and examples below are **V1 results and workflows**.
They are retained as reference material, not presented as V1.5 demonstrations.

- [V1 showcase](assets/character-swap-showcase.gif)
- [V1 native workflow](examples/H3%20Character%20Swap%20v1%20Ref2VA.json)
- [V1 adaptive Turbo8 workflow](examples/akatz-labs-character-swap-turbo8-adaptive-window.json)
- [V1 model card and training details](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/a665bb7e3bb9804e487d2925f67744f554c8ad49/README.md)

## Training and provenance

[V1.5 provenance](training/v1.5/run-summary.json) records the release hash,
initialization, completed updates and training settings. The original
[V1 dataset](https://huggingface.co/datasets/akatz-ai/H3-Character-Swap-v1)
is part of the training lineage; it does not contain the complete V1.5 training
mix. Files directly under `training/` describe V1, while `training/v1.5/`
describes this release.

Original checkpoint bytes are retained. Some embedded metadata uses the internal
V2 name and the original 2,250-update budget; the saved step and final stop record
confirm **1,000 completed additional updates**.

## License and attribution

The adapter retains the [MiniMax H3 Community License Agreement](LICENSE).
See [NOTICE](NOTICE) for attribution and modification history. The app and its
components have their own licenses; those do not replace the model's terms.

## 💜 Support Akatz Labs

I build and release AI tools, models, LoRAs, and workflows so others can experiment, create, and build upon them.

If this model has been useful to you, consider helping fund the next experiment!

Your support goes toward GPU compute, model training, software development, and future open-source releases.

**Support the lab:** [GitHub Sponsors](https://github.com/sponsors/akatz-ai) · [Patreon](https://www.patreon.com/Akatz) · [Ko-fi](https://ko-fi.com/akatz)

Thanks for helping keep independent creative AI development moving! 💜

[Explore Akatz Labs →](https://akatz.ai)
