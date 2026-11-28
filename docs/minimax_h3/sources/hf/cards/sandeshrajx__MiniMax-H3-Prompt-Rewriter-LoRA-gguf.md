# sandeshrajx/MiniMax-H3-Prompt-Rewriter-LoRA-gguf

Источник: https://huggingface.co/sandeshrajx/MiniMax-H3-Prompt-Rewriter-LoRA-gguf · скачано 2026-11-28
Загрузок: 145 · лайков: 1 · базовая: Qwen/Qwen3.6-27B · теги: gguf,lora · обновлено: 2026-08-07T23:19:41.000Z

---

---
base_model: Qwen/Qwen3.6-27B
tags:
- gguf
- lora
- peft
- prompt-rewriting
- minimax-h3
- text-to-audio-video
library_name: llama.cpp
language:
- en
pipeline_tag: text-generation
license: apache-2.0
---

# MiniMax-H3 Prompt Enhancer & Rewriter (GGUF)

Local multimodal prompt enhancer for **MiniMax-H3** joint audio-video generation, running via `llama-server` and `llama.cpp`.

This repository documents the deployment, conversion, analysis, and usage of the **[lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA)** adapter converted to GGUF format and loaded onto a **Qwen3.6-27B GGUF** base model.

---

## 🔗 Quick Links

- **Hugging Face GGUF Repo:** [sandeshrajx/MiniMax-H3-Prompt-Rewriter-LoRA-gguf](https://huggingface.co/sandeshrajx/MiniMax-H3-Prompt-Rewriter-LoRA-gguf)
- **Prompt Samples:** [`samples.md`](samples.md)
- **Analysis Document:** [`analysis.md`](analysis.md)
- **Output Comparison Table:** [`table.md`](table.md)

---

## 🚀 Server Launch Command

To launch `llama-server` with GPU offloading, Flash Attention, Jinja chat template support, and the `Q8_0` GGUF LoRA adapter:

```powershell
llama-server -m "D:\gguf_models\Qwen3.6-27B-Q3_K_S.gguf" `
  --no-mmap `
  --n-gpu-layers 99 `
  --flash-attn auto `
  --jinja `
  -np 1 `
  -c 16000 `
  --temp 0.7 `
  --min-p 0.0 `
  --top-k 15 `
  --top-p 0.95 `
  --chat-template-kwargs "{\"enable_thinking\":true}" `
  --lora "F: