# taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs

Источник: https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs · скачано 2026-11-28
Загрузок: 20969 · лайков: 30 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: distillation,lora · обновлено: 2026-10-26T12:07:39.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
  - MiniMaxAI/MiniMax-H3
base_model_relation: adapter
pipeline_tag: image-to-video
tags:
  - video-generation
  - audio-generation
  - distillation
  - lora
  - minimax-h3
arxiv:
  - 2610.11498
---

# MiniMax-H3 PTD + RefAsymOPD Ref2VA Acceleration LoRA

A **few-step** Ref2VA (reference-image → video + audio) accelerator for MiniMax-H3, combining
**PTD (Parametric Trajectory Distillation)** with **RefAsymOPD** — Reference-Asymmetric On-Policy
Distillation. We release an **8-step** and a **6-step** variant.

It improves **background plausibility, sharpness and realism** over plain PTD at the same step count.

At inference each checkpoint is **structurally identical to a [plain PTD LoRA](https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs)**: one independent
low-rank adapter per inference step, applied on top of the per-interval output heads introduced by
Parallel Decoding Distillation (PDD). The 8-step variant runs in **8** transformer evaluations, the
6-step in **6**, needing no extra input and adding no cost over plain PTD.

> This is a sibling of [`MiniMax-H3-PTD-LoRAs`](https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs)
> (plain PTD, 8-step and 4-step), not a replacement. Because it is structurally identical, the
> inference adapter, Diffusers script and ComfyUI nodes are the same code. Pick the 8-step
> checkpoint for the highest quality, or the 6-step checkpoint for lower latency.

## Weights

| Name | Base Model | Task | NFE | Description |
| --- | --- | --- | --- | --- |
| `MiniMax-H3-Ref2VA-PTD-RefAsymOPD-8Step.safetensors` | [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Ref2VA | 8 | Per-hop PTD LoRA trained with RefAsymOPD, 8 hops (`rank=128`, `network_alpha=128`, BF16). |
| `MiniMax-H3-Ref2VA-PTD-RefAsymOPD-6Step.safetensors` | [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Ref2VA | 6 | Per-hop PTD LoRA trained with RefAsymOPD, 6 hops (`rank=128`, `network_alpha=128`, BF16). |

The checkpoint carries its configuration in the safetensors `__metadata__` (`ptd_num_hops`,
`pdd_num_steps=32`, `pdd_block_size`, `lora_rank`, `lora_alpha`, `lora_targets`); `apply_ptd_lora`
reads it and derives `num_inference_steps`.

## Demo (Ref2VA)

Five cases from the training-monitor conditioning set, **portrait 768×1344, 124 frames, 24 fps,
with audio**.

The first column is the **MiniMax-H3 base at 40 sequential steps with no LoRA** — the
quality ceiling the few-step students are chasing, at 5× the compute of the 8-step columns.

Still frames at the native 768×1344 so the detail is visible (mid-point of each clip):

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
    <thead>
        <tr>
            <th style="text-align: center;" width="32%"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMax-H3 base</a><br>40-step (teacher)</th>
            <th style="text-align: center;" width="32%"><b>PTD + RefAsymOPD<br>8-step</b></th>
            <th style="text-align: center;" width="32%"><b>PTD + RefAsymOPD<br>6-step</b></th>
        </tr>
    </thead>
    <tr>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-0_base_frame.jpg" width="100%"><br><sub>sample-0</sub></td>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-0_refasymopd_frame.jpg" width="100%"></td>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-0_refasymopd_6step_frame.jpg" width="100%"></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-1_base_frame.jpg" width="100%"><br><sub>sample-1</sub></td>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-1_refasymopd_frame.jpg" width="100%"></td>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-1_refasymopd_6step_frame.jpg" width="100%"></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-2_base_frame.jpg" width="100%"><br><sub>sample-2</sub></td>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-2_refasymopd_frame.jpg" width="100%"></td>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-2_refasymopd_6step_frame.jpg" width="100%"></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-3_base_frame.jpg" width="100%"><br><sub>sample-3</sub></td>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-3_refasymopd_frame.jpg" width="100%"></td>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-3_refasymopd_6step_frame.jpg" width="100%"></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-4_base_frame.jpg" width="100%"><br><sub>sample-4</sub></td>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-4_refasymopd_frame.jpg" width="100%"></td>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-4_refasymopd_6step_frame.jpg" width="100%"></td>
    </tr>
</table>

Full five-way video comparison (base, PDD, PTD, PTD + RefAsymOPD 8-step, PTD + RefAsymOPD 6-step) with audio:

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
    <thead>
        <tr>
            <th style="text-align: center;" width="15%">Reference images</th>
            <th style="text-align: center;" width="15%"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMax-H3 base</a><br>40-step (teacher)</th>
            <th style="text-align: center;" width="15%"><a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs">PDD 8-step</a></th>
            <th style="text-align: center;" width="15%"><a href="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs">PTD 8-step</a></th>
            <th style="text-align: center;" width="15%"><b>PTD + RefAsymOPD<br>8-step</b></th>
            <th style="text-align: center;" width="15%"><b>PTD + RefAsymOPD<br>6-step</b></th>
        </tr>
    </thead>
    <tr>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-0_ref0.jpg" width="24%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-0_ref1.jpg" width="24%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-0_ref2.jpg" width="24%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-0_ref3.jpg" width="24%"><br><sub>sample-0</sub></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-0_base_40step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-0_pdd_8step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-0_ptd_8hop.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-0_ptd_refasymopd_8step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-0_ptd_refasymopd_6step.mp4" width="100%" controls loop muted playsinline></video></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-1_ref0.jpg" width="32%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-1_ref1.jpg" width="32%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-1_ref2.jpg" width="32%"><br><sub>sample-1</sub></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-1_base_40step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-1_pdd_8step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-1_ptd_8hop.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-1_ptd_refasymopd_8step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-1_ptd_refasymopd_6step.mp4" width="100%" controls loop muted playsinline></video></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-2_ref0.jpg" width="32%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-2_ref1.jpg" width="32%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-2_ref2.jpg" width="32%"><br><sub>sample-2</sub></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-2_base_40step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-2_pdd_8step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-2_ptd_8hop.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-2_ptd_refasymopd_8step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-2_ptd_refasymopd_6step.mp4" width="100%" controls loop muted playsinline></video></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-3_ref0.jpg" width="24%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-3_ref1.jpg" width="24%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-3_ref2.jpg" width="24%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-3_ref3.jpg" width="24%"><br><sub>sample-3</sub></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-3_base_40step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-3_pdd_8step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-3_ptd_8hop.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-3_ptd_refasymopd_8step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-3_ptd_refasymopd_6step.mp4" width="100%" controls loop muted playsinline></video></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-4_ref0.jpg" width="24%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-4_ref1.jpg" width="24%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-4_ref2.jpg" width="24%"><img src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/asset/sample-4_ref3.jpg" width="24%"><br><sub>sample-4</sub></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-4_base_40step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-4_pdd_8step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-4_ptd_8hop.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-4_ptd_refasymopd_8step.mp4" width="100%" controls loop muted playsinline></video></td>
        <td><video src="https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs/resolve/main/results/sample-4_ptd_refasymopd_6step.mp4" width="100%" controls loop muted playsinline></video></td>
    </tr>
</table>

Everything needed to reproduce a row is in `asset/`: the reference images in presentation order
(`sample-N_ref{0..}.jpg`, addressed as `<Picture 1..>` in the prompt), the prompt
(`sample-N_prompt.txt`), and `demo_samples.json` with the seed, resolution and frame count per
sample.

## Quick Start

Identical to the [plain PTD release](https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs) —
the checkpoint format is the same.

```bash
# NFE is read from the checkpoint metadata automatically.
# --case selects one of the five demo rows above (1 = sample-0); demo cases default to this repo.
python predict_ref2v.py --ptd-lora MiniMax-H3-Ref2VA-PTD-RefAsymOPD-8Step.safetensors --case 1
python predict_ref2v.py --ptd-lora MiniMax-H3-Ref2VA-PTD-RefAsymOPD-6Step.safetensors --case 1
```

```python
import torch
from diffusers import ModularPipeline
from minimax_h3_ptd import apply_ptd_lora

pipeline = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", workflow="ref2va")
pipeline.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3")
# Encode references at the training short edge (768). The pipeline default is 2048, which feeds the
# distilled student ~5x the conditioning tokens it saw in training and degrades output.
pipeline.register_to_config(reference_image_short_edge=768)

nfe = apply_ptd_lora(
    pipeline.transformer_ref,
    "MiniMax-H3-Ref2VA-PTD-RefAsymOPD-8Step.safetensors",  # or -6Step
    pipeline.scheduler.shift,        # video_shift (12.0)
    pipeline.audio_scheduler.shift,  # audio_shift (3.0)
)
result = pipeline(prompt=..., references=[...], num_inference_steps=nfe + 1, ...)
```

Inference needs `diffusers >= 0.40.0` with the MiniMax-H3 `ref2va` workflow. The base model, text
encoder and VAEs are MiniMax's own weights and are downloaded separately — this repository ships
only the adapter.

## ComfyUI

`comfyui_node/ComfyUI-MiniMaxH3-PTD/` holds custom nodes for ComfyUI's native MiniMax-H3; they
require [ComfyUI-PTD](https://github.com/vita-epfl/PTD-inference) alongside them. This checkpoint
loads through them unchanged. See that directory's README for setup and the example workflow.

## Reference
<ol>
  <li id="ref1">Lan Feng, et al. "Parametric Trajectory Distillation for Few-Step Video Generation". Preprint (2026). <a href="https://alan-lanfeng.github.io/PTD/">https://alan-lanfeng.github.io/PTD/</a></li>
  <li id="ref2">Neta Shaul, et al. "Parallel Decoding Distillation for Fast Image and Video Generation". arXiv preprint arXiv:2607.26004 (2026).</li>
</ol>
