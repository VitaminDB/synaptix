# PurpleBlazey/Minimax-H3-pruned-GGUF

Источник: https://huggingface.co/PurpleBlazey/Minimax-H3-pruned-GGUF · скачано 2026-11-28
Загрузок: 4043 · лайков: 3 · базовая: ["MiniMaxAI/MiniMax-H3","Comfy-Org/MiniMax-H3","unsloth/Qwen3-VL-32B-Instruct-GGUF"] · теги: gguf,base_model:Comfy-Org/MiniMax-H3,base_model:quantized:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-22T16:51:14.000Z

---

---
base_model:
- MiniMaxAI/MiniMax-H3
- Comfy-Org/MiniMax-H3
- unsloth/Qwen3-VL-32B-Instruct-GGUF
pipeline_tag: image-text-to-video
license: apache-2.0
language:
- en
tags:
- imatrix
---

Made these for personal experimentation in KoboldCPP's sdui interface: https://github.com/LostRuins/koboldcpp/releases/latest

**Video/Image Model** 
Minimax H3 was quanted with sd-cli.exe: https://github.com/leejet/stable-diffusion.cpp
In theory, pruning is lossless for inference, and only negatively affects training.

**Text Encoder/Clip-1 File**
Qwen3VL-32b had layers 50 to 63 pruned to save on VRAM, which are unused for video encoding. 
All K and IQ quants use imatrix. All UD quants use modified tensor profiles based on: https://huggingface.co/unsloth/Qwen3-VL-32B-Instruct-GGUF

**Image/Video VAE and Audio VAE**
Cloned from here for convenience: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae


From my testing, the Qwen3VL text encoders work surprisingly well even at UD-IQ1_M.
The Minimax H3 quants took at least Q3_K to become usable. Q2_K is a meme quant, but maybe adding enough steps can salvage something decent? After Q4_K, quality bumps become less noticable. I find Q5_K is the quality sweetspot while allowing VRAM room for resolution & video length.


Qwen3VL-32b-Heretic is an experimental quant of https://huggingface.co/llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic 
I can confirm it works with Minimax H3, but I'd recommend a non-heretic version actually intended for it first.
