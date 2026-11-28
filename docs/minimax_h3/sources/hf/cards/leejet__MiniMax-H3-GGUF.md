# leejet/MiniMax-H3-GGUF

Источник: https://huggingface.co/leejet/MiniMax-H3-GGUF · скачано 2026-11-28
Загрузок: 341034 · лайков: 38 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: gguf,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-11-14T11:42:05.000Z

---

---
language:
- en
- zh
base_model:
- MiniMaxAI/MiniMax-H3
tags:
- gguf
- text-to-video
- image-to-video
- video-generation
- stable-diffusion.cpp
---

# MiniMax-H3 GGUF quantized files

The license of the quantized files follows the license of the original model:
 - MiniMax-H3: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE

These files are converted using https://github.com/leejet/stable-diffusion.cpp

## Usage

### stable-diffusion.cpp

This model can be used with [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp).

For setup instructions and usage details, please refer to:

[MiniMax H3 Documentation](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/models/minimax_h3.md)

### ComfyUI

To use this model in **ComfyUI**, first install the following custom node:

[leejet/ComfyUI-GGUF](https://github.com/leejet/ComfyUI-GGUF)

> [!IMPORTANT]
> Please use the **leejet** version of `ComfyUI-GGUF`, rather than the version maintained by `city96`.
> The `city96` repository appears to no longer be actively maintained.

#### Example Workflow

An example ComfyUI workflow is available here:

[`minimax_h3_fl2v_gguf.json`](minimax_h3_fl2v_gguf.json)

#### Example Output

<video
src="https://huggingface.co/leejet/MiniMax-H3-GGUF/resolve/main/assets/example.mp4"
controls
muted
style="max-width: 100%; height: auto;"> </video>
