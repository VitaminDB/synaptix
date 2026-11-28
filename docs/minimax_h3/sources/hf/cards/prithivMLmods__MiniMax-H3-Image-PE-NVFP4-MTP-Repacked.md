# prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked

Источник: https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked · скачано 2026-11-28
Загрузок: 112 · лайков: 2 · базовая: ["MiniMaxAI/MiniMax-H3-Image-PE"] · теги: base_model:quantized:MiniMaxAI/MiniMax-H3-Image-PE · обновлено: 2026-11-11T00:37:58.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE
library_name: transformers
tags:
- mtp
- speculative-decoding
- qwen3_5
- text-generation-inference
- stable
- Model Optimizer
- ModelOpt
- llama-cpp
- nvfp4
- nvidia
- llama-cli
- 27B
- nvidia-blackwell
- rtx-pro-6000
- SM120
- GB202
- B200
- B300
- FP4
- blackwell
- Qwen3.5
- Qwen3.6
- Qwen3.8
- prompt-expansion
- text-to-image
- minimax-h3
- fp4
- sglang
language:
- en
base_model:
- MiniMaxAI/MiniMax-H3-Image-PE
pipeline_tag: image-text-to-text
---

# **MiniMax-H3-Image-PE-NVFP4-MTP-Repacked**

> **MiniMax-H3-Image-PE** is MiniMax's prompt-expansion model for its MiniMax-H3-Image image generator. It is fine-tuned from Qwen3.8-27B and released in FP4. It turns a short request into a detailed image description and recommends an aspect ratio, returned as JSON with `expanded_caption` and `aspect_ratio`, with the output constrained to a schema by XGrammar. It runs with thinking on by default (`reasoning_effort` xhigh, 4,096 max tokens including reasoning), and the reasoning is returned separately. The weights cover text-to-image expansion. For image editing, the release provides only a task prompt and a message-assembly example for a vision-language model of your choice, and the card says editing support is still under development. In editing mode the model describes what is preserved and changed and returns a `target_aspect_ratio`. It can be served with SGLang (0.5.23 or later) or run locally through the Modular Diffusers blocks, where the FP4 weights use about 19 GiB of GPU memory. The Diffusers path runs activations in bf16, so its outputs differ slightly from the SGLang deployment. The card gives no benchmarks. The license is the MiniMax-H3-Image Community License Agreement, not Apache 2.0, and some uses need separate written authorization from MiniMax. The Qwen3.8-27B base is Apache 2.0.

> [!NOTE]
> MTP — Speculative Decoding: Yes | GGUF Available Here — [MiniMax-H3-Image-PE-NVFP4-MTP(GGUF)](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-GGUF)

## MTP-Repacked [Transformers, BF16 • U8] [NVFP4]
 
| File Name | File Type | File Size | File Link | Description |
|-----------|-----------|-----------|-----------|-------------|
| chat_template.jinja | Jinja Template | 8.95 kB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/chat_template.jinja) | Chat template for formatting conversations and prompts. |
| config.json | JSON | 37.6 kB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/config.json) | Core model architecture and configuration parameters. |
| generation_config.json | JSON | 306 Bytes | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/generation_config.json) | Default text generation parameters (e.g., temperature, top_p). |
| hf_quant_config.json | JSON | 9.97 kB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/hf_quant_config.json) | Configuration file detailing the applied quantization (e.g., NVFP4) parameters. |
| model-00001-of-00004.safetensors | Safetensors | 5.19 GB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/model-00001-of-00004.safetensors) | Model weight shard 1/4. **Required** to load the model. |
| model-00002-of-00004.safetensors | Safetensors | 5.2 GB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/model-00002-of-00004.safetensors) | Model weight shard 2/4. **Required** to load the model. |
| model-00003-of-00004.safetensors | Safetensors | 5.17 GB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/model-00003-of-00004.safetensors) | Model weight shard 3/4. **Required** to load the model. |
| model-00004-of-00004.safetensors | Safetensors | 3.21 GB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/model-00004-of-00004.safetensors) | Model weight shard 4/4. **Required** to load the model. |
| model.safetensors.index.json | JSON | 238 kB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/model.safetensors.index.json) | Index file mapping model weights to their respective shards. |
| processor_config.json | JSON | 1.19 kB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/processor_config.json) | Configuration for the multimodal processor (text + vision/image). |
| requirements.txt | Text | 31 Bytes | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/requirements.txt) | List of Python dependencies required to run or load the model. |
| tokenizer.json | JSON | 20 MB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/tokenizer.json) | Complete tokenizer vocabulary and encoding rules. |
| tokenizer_config.json | JSON | 1.12 kB | [Link](https://huggingface.co/prithivMLmods/MiniMax-H3-Image-PE-NVFP4-MTP-Repacked/blob/main/tokenizer_config.json) | Tokenizer-specific configuration parameters. |

## License and Attribution

This model is based on and/or incorporates the following open-source projects and models:

* **Qwen3.8-27B (Base):** [https://huggingface.co/Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
* **MiniMax-H3-Image-PE:** [https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE)

This model is released under the [MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE).
