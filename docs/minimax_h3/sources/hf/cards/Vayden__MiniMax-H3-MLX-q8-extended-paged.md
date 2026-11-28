# Vayden/MiniMax-H3-MLX-q8-extended-paged

Источник: https://huggingface.co/Vayden/MiniMax-H3-MLX-q8-extended-paged · скачано 2026-11-28
Загрузок: 14 · лайков: 1 · базовая: MiniMaxAI/MiniMax-H3 · теги: comfyui,quantized · обновлено: 2026-08-08T17:21:43.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
gated: true
extra_gated_eu_disallowed: true
extra_gated_heading: Acknowledge the included model license
extra_gated_description: Review the license and eligibility requirements before requesting access.
extra_gated_button_content: Acknowledge and request access
extra_gated_prompt: >-
  Access is provided only to users who are eligible under every included license. Do not request
  access if your location or intended use is excluded.
extra_gated_fields:
  Country: country
  I have read the included license and confirm I am eligible to receive these files: checkbox
base_model: MiniMaxAI/MiniMax-H3
pipeline_tag: text-to-video
tags:
  - mlx
  - comfyui
  - minimax-h3
  - quantized
  - video
  - audio
---

# MiniMax H3 MLX q8-extended paged transformer

> **Project:** This checkpoint is built for
> [WeeTodd Nodes](https://github.com/wee-todd/WeeTodd-Nodes), an MLX-native MiniMax H3 custom-node
> suite for ComfyUI on Apple Silicon. Installation, workflow examples, loader behavior, and current
> compatibility notes are maintained in that project.
> Browse the companion artifacts in the
> [WeeTodd MiniMax H3 MLX collection](https://huggingface.co/collections/Vayden/weetodd-minimax-h3-mlx-for-comfyui-6a7765772401cdd54a992af0).

This repository contains a modified MiniMax H3 FL2VA transformer for WeeTodd Nodes on Apple
Silicon. It is not a complete MiniMax H3 checkpoint. The processor, tokenizer, Qwen3-VL text
encoder, video VAE, and audio VAE are required separately.

The checkpoint uses the WeeTodd `q8_extended` mixed-precision recipe. The runtime retains fixed
tensors and loads four transformer blocks at a time. The paged layout reduces active transformer
weight residency without changing the selected checkpoint precision.

## Compatibility

- WeeTodd Nodes commit `789de36` or later
- MLX 0.32.0 or later
- Apple Silicon
- MiniMax H3 FL2VA task

Place the complete repository directory at:

```text
ComfyUI/models/MiniMax-H3/transformers/q8_extended_paged/
```

Select `MiniMax-H3/transformers/q8_extended_paged` as the transformer override in the WeeTodd H3
Component Loader. The loader detects `paged_manifest.json` automatically.

Use it with the
[paged Qwen3-VL conditioner](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-paged) and,
optionally, the
[Q8 video VAE](https://huggingface.co/Vayden/MiniMax-H3-Video-VAE-MLX-Q8). The remaining H3
processor, tokenizer, and audio VAE must be supplied separately under their applicable licenses.

## Quantization and paging

- Profile: `q8_extended`
- Quantization: MLX affine Q8, group size 64
- Quantized modules: 82
- Paged format: `weetodd-h3-paged-v1`
- Transformer pages: 50 block files plus one fixed file
- Execution window: four consecutive blocks
- Tensor storage: approximately 33.38 GB

The quantized modules cover both MLP projections in blocks 21 through 37 and all four core
projections in blocks 38 through 49. Other tensors retain their source precision.

## Measured result

A clean 640 by 384 ComfyUI run used this transformer with paged Qwen, five schedule points, four
transformer evaluations, low-memory BF16 staging, and direct publication. Complete-process peak
was 14.951 GB. This result is a capacity measurement, not a native-resolution quality claim.

## Provenance

- Base model: `MiniMaxAI/MiniMax-H3`
- Base revision reviewed: `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`
- Source transformer SHA-256:
  `85f57a3a7a26def920999f4be786ebb91d0e677ddf327d1848b7724ca829d187`
- Conversion implementation: `wee-todd/WeeTodd-Nodes`
- Conversion profile and page hashes: `quant_config.json` and `paged_manifest.json`

Each page SHA-256 is recorded in `paged_manifest.json`. Modified-file details are in
`MODIFICATIONS.md`.

## License

The weights are a MiniMax H3 Model Derivative. They remain subject to the MiniMax H3 Community
License Agreement in `LICENSE`. Review its territorial, redistribution, notice, and acceptable-use
requirements before downloading or using this repository.

WeeTodd Nodes is an independent project and is not affiliated with MiniMax.
