# H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1

Источник: https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1 · скачано 2026-11-28
Загрузок: 2633 · лайков: 4 · базовая: MiniMaxAI/MiniMax-H3 · теги:  · обновлено: 2026-11-22T12:37:00.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1/blob/main/LICENSE-MiniMax-H3
library_name: diffusers
pipeline_tag: text-to-video
tags:
- minimax-h3
- video-generation
- audio-generation
- reference-to-video
- pruning
- h3-tiny
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: finetune
---

# MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1

[![Project Page](https://img.shields.io/badge/Project-Page-2f6feb)](https://fuyi25.github.io/MiniMax-H3-Tiny/) [![GitHub](https://img.shields.io/badge/GitHub-MiniMax--H3--Tiny-181717?logo=github)](https://github.com/fuyi25/MiniMax-H3-Tiny) [![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-H3--Tiny-ffcc4d)](https://huggingface.co/collections/H3-Tiny/h3-tiny-v01-6afeea67147c27cf3b80820a) [![ComfyUI](https://img.shields.io/badge/ComfyUI-Weights-f0883e)](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-ComfyUI)

Part of **[H3-Tiny](https://github.com/fuyi25/MiniMax-H3-Tiny)**: structured pruning of the MiniMax-H3 Ref2VA transformer
(33B → 17B → 14B) for joint video + audio generation from text and reference images.

🧩 **Using ComfyUI?** Get the single-file version of this checkpoint from [MiniMax-H3-Tiny-ComfyUI](https://huggingface.co/H3-Tiny/MiniMax-H3-Tiny-ComfyUI) (it also needs our [ComfyUI nodes](https://github.com/fuyi25/MiniMax-H3-Tiny/tree/main/comfyui)).

| | |
|---|---|
| Parameters | 17.08B (transformer) |
| Workflow | Ref2VA (reference images to video + audio) |
| Made by | MiniMax-H3-Tiny-17B-Pruned-Ref2VA, then few-step distillation. |
| Sampling steps | 6 by default (4 or more) |

Sample with 6 steps by default; `--steps 4` is faster and `--steps 8` gives more detail.

This repository holds the **transformer only** (diffusers layout: `config.json` + sharded safetensors). The text encoder,
VAEs and processors come from the official MiniMax H3 checkpoint in diffusers format.

## Usage

```bash
git clone https://github.com/fuyi25/MiniMax-H3-Tiny && cd MiniMax-H3-Tiny && pip install -e .
hf download H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1 --local-dir ckpt/H3-Tiny-17B-FewStep-Ref2VA-v0.1
python scripts/infer.py --weights ckpt/H3-Tiny-17B-FewStep-Ref2VA-v0.1 --base-model /models/MiniMax-H3 \
    --prompt "<Picture 1> ..." --references ref1.png --num-frames 124 --out outputs/
```

```python
from h3tiny.pipeline import load_pipeline
pipe = load_pipeline("ckpt/H3-Tiny-17B-FewStep-Ref2VA-v0.1", "/models/MiniMax-H3", device="cuda")
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
