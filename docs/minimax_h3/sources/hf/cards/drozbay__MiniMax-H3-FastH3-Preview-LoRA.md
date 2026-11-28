# drozbay/MiniMax-H3-FastH3-Preview-LoRA

Источник: https://huggingface.co/drozbay/MiniMax-H3-FastH3-Preview-LoRA · скачано 2026-11-28
Загрузок: 2169 · лайков: 7 · базовая: FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2 · теги: lora,distillation,comfyui · обновлено: 2026-08-23T20:02:01.000Z

---

---
license: other
license_name: minimax-h3-community
license_link: LICENSE
base_model: FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2
tags:
- video-generation
- minimax-h3
- lora
- distillation
- few-step
- comfyui
---

# MiniMax H3 FastH3 4-Step LoRA (ComfyUI)

LoRA extraction of [FastVideo-Minimax-FastH3-Preview-v0.2](https://huggingface.co/FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2),
the 4-step DMD2 distillation of MiniMax-H3 FL2VA, as the weight difference against the base
[Comfy-Org fl2va release](https://huggingface.co/Comfy-Org/MiniMax-H3). Apply to any fl2va base
model to sample in 4 steps instead of 50; guidance-distilled, so cfg 1.0.

## Files

| file | for | size |
|---|---|---|
| `loras/..._lora_pruned_rank128_fp16` | `minimax_h3_fl2va_pruned_*` bases (recommended) | 1.33 GB |
| `loras/..._lora_pruned_rank64_fp16` | same, smaller | 0.71 GB |
| `loras/..._lora_full_max256_avg253_fp16` | full (non-pruned) fl2va bases only | 5.06 GB |

The pruned and full variants are not interchangeable: the adaln projections differ between the
two base layouts (see below).

## Usage

Standard **LoRA Loader (model only)** at strength 1.0 on the matching fl2va base, then sample with
**4 steps, euler, simple scheduler, cfg 1.0** (this reproduces the trained ladder
`[999, 749, 500, 250]` under the model's shift-12/3 schedule). Other step counts are
off-distribution per the FastVideo model card. fl2va / t2va only; there is no ref2va student.

## Extraction

Layer-by-layer SVD of the weight difference (student minus base), following the
[KJNodes](https://github.com/kijai/ComfyUI-KJNodes) `LoraExtractKJ` recipe: rank 256 standard,
bias diffs included, no quantile clamp, fp16 output, with exact `svd_linalg` instead of
`svd_lowrank`. Lower ranks are truncations of the same SVD. Norm weights ship as `.diff`,
biases as `.diff_b`.

Two things specific to this model:

- **Pruned adaln refit.** The pruned bases replace the time embedder with an 8-dim curve table,
  so ordinary adaln LoRA keys cannot apply there. The pruned variant instead refits the student's
  adaln projections onto the base's `adaln_t_table` (least squares over the 1025-point grid,
  residual ~1e-5) and stores them as exact `.diff` keys, tiny at `[out, 8]`.
- **The delta is mostly not low-rank.** FastH3 is a full fine-tune saved in bf16; each layer's
  difference is a strong low-rank head plus a flat rounding-dither tail no rank can carry. The
  LoRA captures the distillation direction well but is not a perfect stand-in for the full
  checkpoint.

Measured on a live 4-step run (960x544, 73 frames, same seed/prompt, PSNR vs the actual FastH3
checkpoint; ceiling is the same student under a different quantization):

| run | PSNR |
|---|---|
| base, no LoRA | 14.6 dB |
| base + rank 64 | 17.1 dB |
| base + rank 128 | 17.6 dB |
| base + rank 256 | 17.7 dB |
| FastH3 itself, different quant (ceiling) | 19.9 dB |

Rank 128 is the sweet spot; rank 256 adds almost nothing at double the size. Visually all ranks
land on FastH3's composition and sharpness.

## Credits

- [FastVideo](https://github.com/hao-ai-lab/FastVideo) (hao-ai-lab) for the FastH3 distillation
- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMax) and the
  [Comfy-Org repack](https://huggingface.co/Comfy-Org/MiniMax-H3) used as the base
- [Kijai](https://github.com/kijai/ComfyUI-KJNodes) for the extraction recipe and node math
- Extraction planned, executed, and verified with the help of
  [Claude](https://claude.com/claude-code) (Claude Fable 5, via Claude Code)

Distributed under the MiniMax H3 Community License (see [LICENSE](LICENSE)), inherited from the
base model. Review its territory and acceptable-use terms before use or redistribution.
