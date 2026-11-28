# H3-Tiny/MiniMax-H3-Tiny-17B-Pruned-FL2VA

Источник: https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-Pruned-FL2VA · скачано 2026-11-28
Загрузок: 2024 · лайков: 9 · базовая: MiniMaxAI/MiniMax-H3 · теги:  · обновлено: 2026-11-22T12:36:58.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-Pruned-FL2VA/blob/main/LICENSE-MiniMax-H3
library_name: diffusers
pipeline_tag: image-to-video
tags:
- minimax-h3
- video-generation
- audio-generation
- image-to-video
- text-to-video
- first-last-frame
- pruning
- h3-tiny
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: finetune
---

# MiniMax-H3-Tiny-17B-Pruned-FL2VA

[![Project Page](https://img.shields.io/badge/Project-Page-2f6feb)](https://fuyi25.github.io/MiniMax-H3-Tiny/) [![GitHub](https://img.shields.io/badge/GitHub-MiniMax--H3--Tiny-181717?logo=github)](https://github.com/fuyi25/MiniMax-H3-Tiny) [![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-H3--Tiny-ffcc4d)](https://huggingface.co/collections/H3-Tiny/h3-tiny-v01-6afeea67147c27cf3b80820a) [![ComfyUI](https://img.shields.io/badge/ComfyUI-Weights-f0883e)](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-ComfyUI)

Part of **[H3-Tiny](https://github.com/fuyi25/MiniMax-H3-Tiny)**: structured pruning of MiniMax-H3 for joint video + audio
generation. This checkpoint is the pruned version of the H3 transformer used for **FL2VA** (first + last frame to
video + audio) and **T2VA** (text to video + audio).

🧩 **Using ComfyUI?** Get the single-file version of this checkpoint from [MiniMax-H3-Tiny-ComfyUI](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-ComfyUI).

| | |
|---|---|
| Parameters | 17.07B (transformer) |
| Made by | MiniMax H3 FL2VA / T2VA transformer (33.12B) with AdaLN rank 8 and 8 of its 50 transformer blocks removed. **No training.** |
| Workflows | FL2VA, T2VA |
| Sampling steps | 50 |

MiniMax H3 ships two transformers with the same architecture: one for Ref2VA and one shared by T2VA and FL2VA.
[`MiniMax-H3-Tiny-17B-Pruned-Ref2VA`](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-Pruned-Ref2VA) is pruned from the Ref2VA one; this
repository is pruned from the other with the same recipe, calibrated on FL2VA and T2VA samples. The depth search
removes the same blocks, `[6, 7, 8, 9, 10, 11, 13, 15]`, for both transformers.

This repository holds the **transformer only** (diffusers layout: `config.json` + sharded safetensors). The text encoder,
VAEs and processors come from the official MiniMax H3 checkpoint in diffusers format.

## Usage

```bash
git clone https://github.com/fuyi25/MiniMax-H3-Tiny && cd MiniMax-H3-Tiny && pip install -e .
hf download H3-Tiny/MiniMax-H3-Tiny-17B-Pruned-FL2VA --local-dir ckpt/H3-Tiny-17B-Pruned-FL2VA

# FL2VA: first + last frame
python scripts/infer.py --weights ckpt/H3-Tiny-17B-Pruned-FL2VA --base-model /models/MiniMax-H3 --workflow fl2va \
    --image first.png --last-image last.png --num-frames 124 --out outputs/ \
    --prompt "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced, and at 5.12 seconds into the target video, <Picture 2> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: ..."

# T2VA: text only
python scripts/infer.py --weights ckpt/H3-Tiny-17B-Pruned-FL2VA --base-model /models/MiniMax-H3 --workflow t2va \
    --prompt "integrated_multimodal_description: ..." --num-frames 124 --out outputs/
```

```python
from h3tiny.pipeline import load_pipeline
pipe = load_pipeline("ckpt/H3-Tiny-17B-Pruned-FL2VA", "/models/MiniMax-H3", workflow="fl2va", device="cuda")
```

See the [GitHub README](https://github.com/fuyi25/MiniMax-H3-Tiny) for prompts, offloading, attention backends and the pruning
scripts.

## Contact

Questions and bug reports: [GitHub issues](https://github.com/fuyi25/MiniMax-H3-Tiny/issues) or the Community tab here. For anything else, email Haoshuai Fu at [fohoo@outlook.com](mailto:fohoo@outlook.com).

## License

These weights are Model Derivatives of MiniMax H3 and are distributed under the
[MiniMax H3 Community License Agreement](LICENSE-MiniMax-H3); see [NOTICE](NOTICE). Use of the weights must follow that
agreement, including its territorial scope and acceptable-use policy.

*H3-Tiny is an independent project and is not affiliated with or endorsed by MiniMax.*
