# WAS/Minimax-h3_Singularity_v2_pruned

Источник: https://huggingface.co/WAS/Minimax-h3_Singularity_v2_pruned · скачано 2026-11-28
Загрузок: 105818 · лайков: 38 · базовая: ["WarmBloodAban/Minimax-h3_Singularity","MiniMaxAI/MiniMax-H3"] · теги: comfyui · обновлено: 2026-10-25T01:03:12.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model:
- WarmBloodAban/Minimax-h3_Singularity
- MiniMaxAI/MiniMax-H3
tags:
- minimax-h3
- comfyui
- video-generation
- text-to-video
- image-to-video
- reference-to-video
- pruned
- int8
- w6a8
- w4a8
---

# MiniMax H3 Singularity ref2va v2, pruned

Pruned and quantized versions of the ref2va v2 release of
[WarmBloodAban/Minimax-h3_Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity),
in the same layout as ComfyUI's own pruned MiniMax H3 releases. Load them in ComfyUI with
**Load Diffusion Model**, with `weight_dtype` left at `default`.

| File | Size | Built from | Transformer blocks | Token refiner |
|---|---:|---|---|---|
| `Minimax-h3_Singularity_ref2va_v2_pruned_int8_convrot_r2.safetensors` | 21.0 GB | bf16 release | int8 convrot | bf16 |
| `Minimax-h3_Singularity_ref2va_v2_pruned_w6a8_r2.safetensors` | 16.0 GB | bf16 release | w6a8, group 32 | int8 convrot |
| `Minimax-h3_Singularity_ref2va_v2_pruned_w4a8.safetensors` | 11.8 GB | int8 release | w4a8, group 16 | int8 convrot |

## Build r2

The int8 and w6a8 files are built from `Minimax-h3_Singularity_ref2va_v2_bf16.safetensors`. The
first build started from the int8 release. Measured against the bf16 release:

| | r2 | r1 |
|---|---:|---:|
| int8 weight error, sampled layers | 0.87% to 1.10% | 1.30% to 1.48% |
| Timestep modulation error | 0.02% | 0.3% to 0.6% |
| int8 render, mean PSNR over four seeds | 26.1 dB | 24.6 dB |
| w6a8 render, mean PSNR over four seeds | 24.4 dB | 23.9 dB |

The w4a8 file is unchanged. Built from bf16 it rendered further from the bf16 release on every seed
tested (20.2 dB against 21.5 dB), so the build from the int8 release stays.

The r1 int8 and w6a8 files remain in the repository history at commit
[`ad7eb66`](https://huggingface.co/WAS/Minimax-h3_Singularity_v2_pruned/tree/ad7eb66157ff5fbe3429ef4413921fb0c7c9323f).

## What was pruned

The timestep embedder and the full-width adaLN projection in each of the 50 transformer blocks,
13 billion weights in all, are replaced by a 1025 x 8 `adaln_t_table` and 8-wide adaLN weights:
the curve layout of ComfyUI's own `minimax_h3_ref2va_pruned_int8_convrot`. ComfyUI loads it
natively. In the r2 files the table comes from the bf16 release's own timestep embedder, and the
modulation it produces is within 0.03% of the bf16 release's at every timestep tested.

The r2 int8 file quantizes the transformer block linears from bf16 with each row's scale set for
the lowest error. The w6a8 file quantizes the block linears and `final_layer.video_out`, with the
token refiner in int8 convrot, as ComfyUI's pruned w6a8 release does.

## Comparison

![The bf16 release and the pruned files, first, middle and last frame](comparison/contact_sheet.png)

1344 x 768, 124 frames at 24 fps, euler, simple, 6 steps, DMAD 4-step LoRA at 0.7, PyTorch
attention, `minimax_h3_video_vae_int8_convrot` and the Qwen3-VL 32B MiniMax H3 text encoder. Mean
PSNR of every frame against the bf16 release rendered with the same seed:

| File | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Mean | Render, seed 42 |
|---|---:|---:|---:|---:|---:|---|
| bf16 release | | | | | | [reference_bf16.mp4](comparison/reference_bf16.mp4) |
| pruned int8 convrot r2 | 23.3 | 24.9 | 28.2 | 28.0 | 26.1 dB | [pruned_int8_convrot_r2.mp4](comparison/pruned_int8_convrot_r2.mp4) |
| pruned w6a8 r2 | 23.5 | 24.2 | 24.3 | 25.8 | 24.4 dB | [pruned_w6a8_r2.mp4](comparison/pruned_w6a8_r2.mp4) |
| pruned w4a8 | 21.9 | 21.1 | 21.7 | 21.4 | 21.5 dB | [pruned_w4a8.mp4](comparison/pruned_w4a8.mp4) |

The bf16 release rendered twice with the same settings is identical frame for frame, so every
difference in the table comes from the weights. A single seed can rank two close files either way;
the mean over four is the comparison.

LoRAs without adaLN or timestep-embedder keys apply unchanged, including the source card's
recommended `minimax_h3_ref2v_turbo_4step_v0.1`.

## Modified files

These files are modified versions of WarmBloodAban's Singularity ref2va v2 release (MiniMax H3
Community License Agreement, Section III.2): the int8 and w6a8 files of
`Minimax-h3_Singularity_ref2va_v2_bf16.safetensors`, the w4a8 file of
`Minimax-h3_Singularity_ref2va_v2_int8.safetensors`. The timestep path is replaced as described
above and the transformer weights are quantized. Each file's safetensors metadata states the same.

## Credits

Fine-tune: [WarmBloodAban](https://huggingface.co/WarmBloodAban). Base model:
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) by MiniMax.

## License

MiniMax H3 Community License Agreement. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
