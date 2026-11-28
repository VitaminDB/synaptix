# Raretutor/Qwen3-VL-32B-MiniMax-H3-GGUF

Источник: https://huggingface.co/Raretutor/Qwen3-VL-32B-MiniMax-H3-GGUF · скачано 2026-11-28
Загрузок: 4641 · лайков: 2 · базовая: Qwen/Qwen3-VL-32B-Instruct · теги: gguf,quantized,base_model:quantized:Qwen/Qwen3-VL-32B-Instruct · обновлено: 2026-09-06T10:06:34.000Z

---

---
base_model: Qwen/Qwen3-VL-32B-Instruct
library_name: gguf
license: apache-2.0
tags:
- gguf
- quantized
- vision
- qwen
- qwen3
- image-text-to-text
pipeline_tag: image-text-to-text
---


# How to use it in ComfyUI ???

<a href="https://www.youtube.com/watch?v=zUk6mqruWdA" style="color: yellow;">https://www.youtube.com/watch?v=zUk6mqruWdA</a>


# Qwen3-VL-32B-Instruct - GGUF (Q2 Quantization)

## Model Overview
This repository contains the heavily quantized Q2 GGUF version of the multimodal [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) model. 

By aggressively compressing the weights to 2-bit, this 32-billion parameter model is optimized for extreme low-VRAM execution. The footprint is reduced enough to allow full GPU offloading on consumer hardware with 12GB of VRAM (like an RTX 3060), making it highly practical for local Python inference scripts, Gradio interfaces, and ComfyUI node-based workflows without hitting out-of-memory errors.

## File Details
* **Base Model:** `Qwen/Qwen3-VL-32B-Instruct`
* **Format:** GGUF
* **Quantization Level:** Q2 (2-bit)
* **Architecture:** Qwen3 Vision-Language

> **Important Note for Vision:** Because this is a Vision-Language model, you will likely need the multimodal projector (`mmproj`) GGUF file to process images. Ensure your pipeline loads both the main LLM GGUF and the corresponding projector file.

## Usage

You can run this model locally using [llama.cpp](https://github.com/ggerganov/llama.cpp) or its Python bindings (`llama-cpp-python`). 

### Example with `llama-cpp-python`
```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# Load the multimodal projector
chat_handler = Llava15ChatHandler(clip_model_path="path/to/mmproj-model-f16.gguf")

# Load the Q2 GGUF model
llm = Llama(
    model_path="path/to/qwen3-vl-32b-instruct-q2.gguf",
    chat_handler=chat_handler,
    n_ctx=4096, # Adjust based on your sequence needs
    n_gpu_layers=-1 # Offload all layers to GPU for maximum speed
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}},
                {"type": "text", "text": "Describe this image in detail."},
            ]
        }
    ]
)

print(response["choices"][0]["message"]["content"])
