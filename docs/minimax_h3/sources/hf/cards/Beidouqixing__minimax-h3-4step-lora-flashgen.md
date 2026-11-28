# Beidouqixing/minimax-h3-4step-lora-flashgen

Источник: https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen · скачано 2026-11-28
Загрузок: 3673 · лайков: 32 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,distillation · обновлено: 2026-09-22T09:23:38.000Z

---

---
license: apache-2.0
base_model: MiniMaxAI/MiniMax-H3
pipeline_tag: image-to-video
language:
  - en
  - zh
tags:
  - minimax-h3
  - lora
  - t2va
  - distillation
  - 4step
  - flashgen
---

# MiniMax-H3 4-Step LoRA (FlashGen)

4-step distilled LoRA for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) text-to-video+audio (T2VA). Trained on Ascend NPU with data-free distillation; merge into the base model and inject `base_schedule` before vllm-omni / MindIE-SD deployment.

## Files

| File | Description |
|------|-------------|
| `minimax_h3_4step_lora_flashgen_v1.0_768p_bf16.safetensors` | Student LoRA, rank 64, bf16 |
| `merge_lora_ckpt.py` | Merge LoRA into base model and inject `base_schedule` |

## Training config

| Item | Value |
|------|-------|
| Method | Distribution matching distillation (VSD, no GAN), trained on Ascend NPU |
| Task | T2VA |
| Inference steps | 4 |
| Resolution | 1344×768, 5.2s |
| base_schedule | `[1.0, 0.7, 0.4, 0.15, 0.0]` |
| LoRA targets | `qkv_proj`, `out_proj`, `fc1`, `fc2`, `adaln_proj.linear` |
| LoRA rank | 64 |

## Usage

Requires base model [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).

Merge with `merge_lora_ckpt.py` before vllm-omni / MindIE deployment:

```bash
python merge_lora_ckpt.py \
  ${MODEL_ROOT}/FL2VA \
  minimax_h3_4step_lora_flashgen_v1.0_768p_bf16.safetensors \
  ./minimax-h3-4step-flashgen

export MODEL=./minimax-h3-4step-flashgen/FL2VA
```

## Deployment (MindIE-SD / vllm-omni)

For Ascend NPU inference and serving, see the MindIE-SD deployment guide:

- [MiniMax-H3 推理部署文档（含 4 步蒸馏权重合并与启动）](https://gitcode.com/Ascend/MindIE-SD/blob/dev/examples/minimax-h3/infer.md)
- MindIE-SD 仓库：[Ascend/MindIE-SD](https://gitcode.com/Ascend/MindIE-SD)
