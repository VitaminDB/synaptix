# SpXMerlin1D/MiniMaxH3-CondBridge-Qwen3.5-4B

Источник: https://huggingface.co/SpXMerlin1D/MiniMaxH3-CondBridge-Qwen3.5-4B · скачано 2026-11-28
Загрузок: 0 · лайков: 7 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: interface-distillation,comfyui · обновлено: 2026-08-16T14:18:23.000Z

---

---
language:
- en
- zh
tags:
- text-encoding
- video-generation
- interface-distillation
- minimax-h3
- comfyui
library_name: adapter
pipeline_tag: text-to-video
base_model:
- MiniMaxAI/MiniMax-H3
---

# Please Read Readme FIRST / 请先阅读 README

# CondBridge: Interface-Distilled Text Conditioning for MiniMax-H3

## A DeepSeek-V4-Flash 0731 Assisted Project

**A 1.14B adapter that bridges a lightweight Qwen3.5-4B student to the
MiniMax-H3 33B text encoder's injection space.** It converts the student's
hidden states into the exact CLIP-injection representation the H3 DiT expects
(post-`condition_proj` + `token_refiner`), replacing the 33B teacher encoder
end-to-end.

## Model Description

MiniMax-H3 video generation conditions the DiT on text embeddings produced by a
33B text encoder (`condition_proj` + 2-layer `token_refiner`). Running it locally
is heavy. **CondBridge distills that interface** into a 1.14B adapter that
consumes:

| Input | Shape | Source |
|---|---|---|
| `h3_ids` | `[S_T]` | H3 tokenizer (vocab 151,643) |
| `student_hidden` | `[S_S, 2560]` | Qwen3.5-4B `hidden_states[-1]` (post-final-norm) |

and outputs the teacher-equivalent representation `[1, S_T, 5376]`. Because the
output lives in the same space as the teacher's post-refiner embeddings, the DiT
consumes it directly

## Architecture

`H3Adapter` — 1.144B params total:

| Module | Params | Role |
|---|---|---|
| `source_projection` | 13.8M | student 2560 → 5376 (KV) |
| `query_embedding` | 40.3M | H3 ids → 5376 query (embed 151,936×256 + proj) |
| `cross_attention` | 319M | 32-head resampler, QK-norm + tanh gate |
| `token_refiner` | 751M | 2 pre-norm blocks + final RMSNorm (mirrors teacher) |

Forward: `kv = source_proj(student)` → `x = cross_attn(query_embed(h3_ids), kv)`
→ `token_refiner(x)`.

## Training Data

- 10,032 prompts generated from a 9-skill element-pool Cartesian combos
  (subjects/actions/scenes/camera/lighting/styles...), deduplicated to **10,027**
- Each prompt: H3 tokenizer ids + Qwen3.5-4B hidden states + teacher target
  (`condition_proj` + `token_refiner` output of the official encoder)
- Feature-space dedup + length bucketing (short/medium/long)

## Training Details

Two-stage fine-tuning (32GB GPU, bf16):

| Stage | Duration | Steps | Scope | Result |
|---|---|---|---|---|
| 1 | 4h | 18,874 | adapter body, refiner frozen (DiT init) | cos 0.8856 |
| 2 | 1.5h | 4,135 | all params, refiner lr×0.1 | **cos 0.9229** |

- Optimizer: Muon (2D weights) + AdamW8bit (1D), WSD schedule, wd 0.02
- Loss v3: `huber×1.0 + cos×0.8 + infonce×0.05 + sp×0.1 + mag×0.1 + stat×0.0002`
  with a curriculum ramp on the contrastive terms
- Scale handled via RMS target-norm alignment (`pn = normalize(pred) × target_norm`)

## Evaluation

Held-out 1,002 prompts (disjoint from train):

| Metric | Value |
|---|---|
| **cosine** (token-level) | **0.9229** |
| **MSE** | 0.8162 |
| **norm_ratio** (pred/target scale) | 1.013 |


## Usage

```python
import torch
from transformers import AutoTokenizer
from adapter.model import H3Adapter          # repo code
from safetensors.torch import load_file

adapter = H3Adapter().to(torch.bfloat16)
adapter.load_state_dict(load_file("condbridge.safetensors"))
adapter.eval()

h3_tok = AutoTokenizer.from_pretrained("<h3 tokenizer>")
# student_hidden: Qwen3.5-4B hidden_states[-1]  [S_S, 2560]
h3_ids = h3_tok(prompt, add_special_tokens=False)["input_ids"]
embeds = adapter(torch.tensor([h3_ids]), student_hidden.unsqueeze(0))  # [1, S_T, 5376]
```

Requires: `Qwen3.5-4B` (student), MiniMax-H3 tokenizer, the H3 DiT.

## ComfyUI Integration

Use with the [ComfyUI-MiniMaxH3-Adapter] node: `MiniMaxH3AdapterLoader`
(student dir + adapter .safetensors) → plug into official
`MiniMaxH3ImageToVideo`. The node exposes the adapter as a duck-typed CLIP.

** I use HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive but it worked for Qwen/Qwen3.5-4B and every Fine-Tune or Quantizations **


## Limitations

- Distilled fidelity is bounded by the student (Qwen3.5-4B): complex multi-shot
  or long cinematic prompts may drift from the 33B teacher.
- Trained on generated prompt combinations (9-skill template), not in-the-wild
  caption distributions.
- Intended for research / local use; check MiniMax-H3 terms of service.

## Training your own CondBridge

- Training Code to be released soon

# CondBridge：面向 MiniMax-H3 的接口蒸馏文本条件适配器

## 本项目在DeepSeek-V4-Flash 0731帮助下完成

**一个 1.14B 参数的适配器，为轻量级 Qwen3.5-4B 学生模型搭建通往 MiniMax-H3
33B 文本编码器注入空间的"桥"。** 它将学生模型的隐藏状态转换为 H3 DiT 期望的
CLIP 注入表示（`condition_proj` + `token_refiner` 之后的结果），端到端替代
33B 教师编码器。

## 模型描述

MiniMax-H3 视频生成用 33B 文本编码器（`condition_proj` + 2 层 `token_refiner`）
产生文本嵌入来条件化 DiT。本地运行它很重。**CondBridge 蒸馏了这套接口**，用
一个 1.14B 适配器消费：

| 输入 | 形状 | 来源 |
|---|---|---|
| `h3_ids` | `[S_T]` | H3 tokenizer（词表 151,643） |
| `student_hidden` | `[S_S, 2560]` | Qwen3.5-4B `hidden_states[-1]`（final-norm 后） |

并输出与教师等价的表示 `[1, S_T, 5376]`。由于输出与教师的 post-refiner 嵌入
处于同一空间，DiT 可直接消费

## 模型架构

`H3Adapter` — 共 1.144B 参数：

| 模块 | 参数量 | 作用 |
|---|---|---|
| `source_projection` | 13.8M | 学生 2560 → 5376（作为 KV） |
| `query_embedding` | 40.3M | H3 ids → 5376 查询（embed 151,936×256 + proj） |
| `cross_attention` | 319M | 32 头重采样器，QK-norm + tanh 门控 |
| `token_refiner` | 751M | 2 个 pre-norm block + final RMSNorm（镜像教师） |

前向：`kv = source_proj(student)` → `x = cross_attn(query_embed(h3_ids), kv)`
→ `token_refiner(x)`。

## 训练数据

- 从 9 类技能元素池笛卡尔组合生成 10,032 条 prompt
  （主体/动作/场景/镜头/灯光/风格...），去重后 **10,027** 条
- 每条 prompt 含：H3 tokenizer ids + Qwen3.5-4B 隐藏状态 + 教师目标
  （官方编码器的 `condition_proj` + `token_refiner` 输出）
- 特征空间去重 + 长度分桶（短/中/长）

## 训练细节

两阶段微调（32GB 显卡，bf16）：

| 阶段 | 时长 | 步数 | 范围 | 结果 |
|---|---|---|---|---|
| 1 | 4h | 18,874 | 适配器主体，refiner 冻结（DiT 初始化） | cos 0.8856 |
| 2 | 1.5h | 4,135 | 全参数，refiner lr×0.1 | **cos 0.9229** |

- 优化器：Muon（2D 权重）+ AdamW8bit（1D 权重），WSD 调度，wd 0.02
- Loss v3：`huber×1.0 + cos×0.8 + infonce×0.05 + sp×0.1 + mag×0.1 + stat×0.0002`，
  对比项带课程 ramp
- 尺度处理：RMS 目标范数对齐（`pn = normalize(pred) × target_norm`）

## 评测结果

Held-out 1,002 条 prompt（与训练集不重叠）：

| 指标 | 数值 |
|---|---|
| **cosine**（逐 token） | **0.9229** |
| **MSE** | 0.8162 |
| **norm_ratio**（预测/目标尺度） | 1.013 |


## 使用方法

```python
import torch
from transformers import AutoTokenizer
from adapter.model import H3Adapter          # 仓库代码
from safetensors.torch import load_file

adapter = H3Adapter().to(torch.bfloat16)
adapter.load_state_dict(load_file("condbridge.safetensors"))
adapter.eval()

h3_tok = AutoTokenizer.from_pretrained("<h3 tokenizer>")
# student_hidden: Qwen3.5-4B hidden_states[-1]  [S_S, 2560]
h3_ids = h3_tok(prompt, add_special_tokens=False)["input_ids"]
embeds = adapter(torch.tensor([h3_ids]), student_hidden.unsqueeze(0))  # [1, S_T, 5376]
```

依赖：`Qwen3.5-4B`（学生）、MiniMax-H3 tokenizer、H3 DiT。

## ComfyUI 集成

配合 [ComfyUI-MiniMaxH3-Adapter] 节点使用：`MiniMaxH3AdapterLoader`
（学生目录 + 适配器 .safetensors）→ 接入官方 `MiniMaxH3ImageToVideo` 节点。

** 我在测试时使用HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive，但理论上原版Qwen/Qwen3.5-4B的GGUF量化以及其任何微调的GGUF量化版本都可用 **

## 局限性

- 蒸馏保真度受限于学生（Qwen3.5-4B）：复杂多镜头或长电影级 prompt 可能
  偏离 33B 教师
- 训练数据是生成的 prompt 组合（9 类技能模板），非真实世界字幕分布
- 面向研究 / 本地使用；请遵守 MiniMax-H3 服务条款

# 我自己想训练一个CondBridge

- 训练代码整理后发布
