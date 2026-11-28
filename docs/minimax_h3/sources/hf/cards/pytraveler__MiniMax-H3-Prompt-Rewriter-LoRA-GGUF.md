# pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF

Источник: https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF · скачано 2026-11-28
Загрузок: 2181 · лайков: 26 · базовая: lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA · теги: gguf,lora,base_model:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA,base_model:adapter:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA · обновлено: 2026-08-18T20:47:58.000Z

---

---
base_model: lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA
library_name: gguf
tags:
  - gguf
  - lora
  - llama.cpp
  - prompt-rewriting
  - minimax-h3
  - text-to-audio-video
language:
  - en
---

[![GitHub](https://img.shields.io/badge/GitHub-ComfyUI%20Node-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)

# MiniMax-H3 T2VA Prompt Rewriter LoRA — GGUF

A GGUF conversion of
[lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA),
so the rewriter runs under [llama.cpp](https://github.com/ggml-org/llama.cpp)
against a quantized Qwen3.6-27B instead of the 52 GB bf16 checkpoint.

This repository contains no new training — only a format conversion. The tensor
values are unchanged.

## Why

The original adapter is a PEFT LoRA for `Qwen/Qwen3.6-27B`, which means a 52 GB
download before anything can run. Under llama.cpp the same adapter attaches to a
quantized base:

| Base | Download | VRAM with this adapter |
|---|---|---|
| `Qwen3.6-27B-Q4_K_M.gguf` | 15.7 GB | ~19 GB |
| `Qwen3.6-27B-IQ4_XS.gguf` | 14.4 GB | ~18 GB |
| `Qwen3.6-27B-UD-Q3_K_XL.gguf` | 13.5 GB | ~17 GB |
| `Qwen3.6-27B-UD-IQ2_M.gguf` | 10.1 GB | ~13 GB |

Offloading fewer layers lowers VRAM further at the cost of speed, and llama.cpp
brings ROCm, Metal and CPU along with CUDA.

Base quants come from
[unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF).

## Contents

| File | Size | Notes |
|---|---|---|
| `MiniMax-H3-Prompt-Rewriter-LoRA-F16.gguf` | 3.48 GB | 992 tensors, rank 256, `adapter.lora.alpha = 512` |
| `MiniMax-H3-Prompt-Rewriter-LoRA-Q8_0.gguf` | 1.85 GB | the same adapter, quantised |

Tensor count and alpha match the source adapter exactly.

Take `Q8_0` unless you have a reason not to. It is quantised from the `F16` in
this repository rather than converted separately, so tensor names, their order
and every metadata field are identical — only the stored values differ, by a
mean of 0.001 relative across all 992 tensors and 0.004 on the worst one.

Both behave the same where it matters: with either adapter attached the rewrite
comes back in the trained cut-point form, `[Shot 2] At 00:06.500`, while the
unadapted base model writes plain ranges like `(0s-4s)` instead.

## Use with llama.cpp

```bash
llama-cli \
  -m Qwen3.6-27B-Q4_K_M.gguf \
  --lora MiniMax-H3-Prompt-Rewriter-LoRA-Q8_0.gguf \
  -sysf system_prompt.txt \
  -p "resolution: 16:9
duration: 15s
original_prompt: A red fox walks through a snowy forest at dawn." \
  -st -n 1400 --temp 0 -ngl 99 -c 8192 --reasoning off
```

`system_prompt.txt` must hold the exact system prompt from
[`prompt_template.py`](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA/blob/main/prompt_template.py)
in the source repository — the adapter was trained on that wording, and changing
it degrades the rewrite.

`--reasoning off` matters: it corresponds to the `enable_thinking=False` that the
reference `infer.py` passes. Without it the model spends hundreds of tokens
reasoning before starting the rewrite.

Measured on an RTX 5090 with `Q4_K_M`: **50 tok/s** with the adapter, 78 tok/s
without it. That ~35% difference is llama.cpp computing the adapter's matmuls,
which is the simplest confirmation that it is actually applied.

## Use in ComfyUI

[MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)
picks this adapter up automatically when a GGUF base model is selected and
`llama-cpp-python` is installed. It fetches `F16` by default; name
`MiniMax-H3-Prompt-Rewriter-LoRA-Q8_0.gguf` in the options node's `adapter`
field to take the smaller one instead and it is downloaded from here.

## How it was converted

`convert_lora_to_gguf.py` from llama.cpp, plus a one-hunk fix to
`conversion/qwen.py`.

llama.cpp reorders the V heads of Qwen3.5's linear-attention layers. For
`in_proj_qkv`, `in_proj_z`, `in_proj_a` and `in_proj_b` that reorder lands on the
output dimension and a LoRA passes through untouched. For `out_proj` it lands on
the **input** dimension, where `LoraTorchTensor.reshape` refuses — it cannot
reshape the axis that lives on the `A` factor.

The fix is to express that one reorder as an index instead of
`reshape`/`permute`/`reshape`. `LoraTorchTensor.__getitem__` already routes a
last-axis index onto `A`, and

```
(B @ A)[..., p] == B @ A[..., p]
```

exactly, so nothing is approximated and no full-rank delta is materialised. For
ordinary dense tensors the two forms are bit-identical, so the base-model
conversion path is unaffected.

```python
elif ".out_proj." in name:
    col_perm = self._reorder_v_heads(
        torch.arange(num_v_heads * head_v_dim, dtype=torch.long).unsqueeze(0),
        1, num_k_heads, num_v_per_k, head_v_dim,
    ).squeeze(0)
    data_torch = data_torch[..., col_perm]
```

## Licence

The weights originate from
[lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA)
and were converted without modifying their values. Whatever terms that
repository carries apply here unchanged; this repository claims nothing
additional.

Use of MiniMax-H3 itself is governed by the licence and acceptable-use terms in
the [official MiniMax-H3 repository](https://huggingface.co/MiniMaxAI/MiniMax-H3).
