# asadkhan89/Minimax-h3-Turbo-SLA

Источник: https://huggingface.co/asadkhan89/Minimax-h3-Turbo-SLA · скачано 2026-11-28
Загрузок: 0 · лайков: 1 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: lora,distillation · обновлено: 2026-09-07T17:49:37.000Z

---

---
license: apache-2.0
base_model:
  - MiniMaxAI/MiniMax-H3
pipeline_tag: image-to-video
language:
  - en
tags:
  - video-generation
  - image-to-video
  - audio-video-generation
  - lora
  - distillation
  - sparse-attention
  - sla
  - lightx2v
---

# MiniMax-H3 Turbo-SLA

MiniMax-H3 Turbo-SLA is a 4-step distilled FL2V checkpoint for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), enhanced with [SLA (Sparse–Linear Attention)](https://github.com/thu-ml/SLA) for more efficient inference.

The model uses an **85% attention sparsity ratio**. In our tested [LightX2V](https://github.com/ModelTC/LightX2V) setup, it delivers approximately **2.5× inference acceleration on an NVIDIA RTX 5090** while retaining competitive visual quality.

> This repository contains LoRA weights and requires the original MiniMax-H3 model for inference. Actual performance may vary with resolution, video length, software environment, and hardware configuration.

## Demo

Side-by-side comparison between the 30-step MiniMax-H3 base model and the 4-step MiniMax-H3 Turbo-SLA model:

<video controls loop muted playsinline width="100%">
  <source src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/HXV6GbED84PeLteFXqK5l.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Model Highlights

- **4-step distillation** for substantially reduced denoising steps.
- **SLA sparse attention** with an **85% sparsity ratio**.
- Approximately **2.5× inference acceleration on RTX 5090** in our tested LightX2V setup.
- **768p FL2V** generation support.
- Native LightX2V and ComfyUI-compatible BF16 checkpoints.

## Available Checkpoints

| Checkpoint | Format | Description |
|---|---|---|
| [`minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_bf16.safetensors`](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA/blob/main/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_bf16.safetensors) | LightX2V | Native BF16 LoRA checkpoint for LightX2V inference. |
| [`minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors`](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA/blob/main/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors) | ComfyUI | Converted BF16 LoRA checkpoint for ComfyUI workflows. |

## LightX2V Inference

For installation and general MiniMax-H3 inference instructions, refer to the [LightX2V MiniMax-H3 examples](https://github.com/ModelTC/LightX2V/tree/main/examples/minimax_h3).

Use the following configuration for SLA-enabled inference on an RTX 5090:

[`minimax_h3_fp8_4step_5090_with_fp8_vae_sla.json`](https://github.com/ModelTC/LightX2V/blob/main/configs/minimax_h3/dmd/minimax_h3_fp8_4step_5090_with_fp8_vae_sla.json)

Key SLA-related settings in this configuration include:

```json
{
  "attn_type": "dynamic_sparse_attn",
  "dynamic_sparse_attn_setting": {
    "sparsity_ratio": 0.85,
    "operator": "sage2"
  },
  "video_flow_shift": 6.0,
  "audio_flow_shift": 3.0,
  "h3_step_update": "training_euler"
}
```

Use the configuration file linked above as the source of truth for the complete inference setup, including FP8 DiT/VAE options and checkpoint paths.

## ComfyUI

For ComfyUI workflows, use:

[`minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors`](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA/blob/main/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors)

The ComfyUI checkpoint is a converted version of the SLA-enabled 4-step LoRA and is intended for compatible MiniMax-H3 ComfyUI workflows.

## Related Projects

- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [MiniMax-H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
- [LightX2V](https://github.com/ModelTC/LightX2V)
- [SLA: Sparse–Linear Attention](https://github.com/thu-ml/SLA)

## Acknowledgements

This work builds on MiniMax-H3, LightX2V, and SLA. We thank the respective authors and contributors for making their work available to the community.

## Citation

If you use SLA in your work, please cite:

```bibtex
@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention},
  author={Zhang, Jintao and Wang, Haoxu and Jiang, Kai and Yang, Shuo and Zheng, Kaiwen and Xi, Haocheng and Wang, Ziteng and Zhu, Hongzhou and Zhao, Min and Stoica, Ion and others},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
```

## License

The adapter weights in this repository are released under the Apache 2.0 License. Use of the MiniMax-H3 base model is also subject to its corresponding license and terms of use.
