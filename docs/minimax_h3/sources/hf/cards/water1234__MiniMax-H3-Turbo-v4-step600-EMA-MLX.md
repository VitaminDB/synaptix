# water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX

Источник: https://huggingface.co/water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX · скачано 2026-11-28
Загрузок: 0 · лайков: 5 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,turbo · обновлено: 2026-08-11T17:56:55.000Z

---

---
license: apache-2.0
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: adapter
tags:
  - mlx
  - apple-silicon
  - minimax-h3
  - lora
  - turbo
  - text-to-video
  - audio-video
---

# MiniMax-H3 Turbo v4-step600 EMA — MLX Native Streaming Adapter

BF16 MLX-native conversion of [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora), pinned to revision `43a74557ac3f6539db8e0f2a959d03feb7a81480` and source file `minimax_h3_turbo_v4_step600_ema.safetensors`.

## Conversion contract

- Source SHA-256: `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`
- MLX monolithic adapter SHA-256: `ed679d4ec8f39ab5c71902f3e13af5bc6a4792bd0317d61c5f6171e40411925e`
- 518/518 BF16 tensors consumed
- 259/259 LoRA A/B pairs
- Tensor values are exactly equal before/after conversion
- Attention/MLP rank: 64
- Block/final AdaLN rank: 16
- `alpha=rank`, runtime multiplier `1.0`
- Runtime implementation commit: `d5c1e7995e54bbc99cd8b5e7dbca7fb41d119a39`

The source already uses canonical fused H3 module paths. Conversion validates every tensor and adds MLX runtime metadata plus a streaming index; it does not quantize, merge or numerically alter the adapter.

## Files

- `adapter.safetensors.index.json`: MLX streaming map
- `shards/`: 50 main-block, two token-refiner, and one final-layer shard
- `conversion_manifest.json`: source identity, mapping, ranks and per-shard records
- `turbo_lora.safetensors`: optional monolithic archival copy (duplicates the shard payload)

For the smallest download, clients need the index, manifest and `shards/`; the monolithic file is optional.

## MLX usage

Use this adapter as a directory. Do not pass a fixed `--turbo-lora-alpha`; the runtime restores `alpha=rank` from the native contract.

```bash
python scripts/generate.py "A cinematic fox running through a misty forest" \
  --checkpoint /path/to/MiniMax-H3/FL2VA \
  --transformer /path/to/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --text-encoder /path/to/your/MLX-text-encoder \
  --turbo-lora /path/to/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --turbo-lora-scale 1.0 --steps 9 \
  --low-memory --stream-blocks --output out.mp4
```

`--steps 9` means eight denoiser evaluations. `--steps 7` is the faster six-evaluation setting.

## Validation

The actual converted adapter passed:

- mixed-rank native-provider strict validation
- per-component shard loading
- legacy Turbo streaming regression
- real calibrated INT8 base + real BF16 Turbo finite refiner computation
- all 50 streamed block AdaLN cache computations
- real block-0 attention/MLP application
- final-layer AdaLN application

The adapter remains separate from the INT8 base so its low-rank update is never rounded into quantized weights.

## Attribution and license

Turbo training and original release are by `larryvrh`. This repository changes packaging/runtime layout only. The source model card declares Apache-2.0 for the adapter. Use also requires a compatible MiniMax-H3 base model, whose separate Community License remains applicable.
