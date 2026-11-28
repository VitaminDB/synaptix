# convertor/minimax-h3-gguf

Источник: https://huggingface.co/convertor/minimax-h3-gguf · скачано 2026-11-28
Загрузок: 2594 · лайков: 2 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: gguf,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-11-11T17:28:34.000Z

---

---
base_model:
- MiniMaxAI/MiniMax-H3
---
![screenshot](https://raw.githubusercontent.com/gguf-org/gguf-desktop/master/demo16.gif)

```bash
ggk -M vid_gen --diffusion-model minimax_h3_ref2va_pruned-nvfp4.gguf --vae minimax_h3_video_vae_fp16-f16.gguf --audio-vae minimax_h3_audio_vae_fp32-f16.gguf --llm qwen3vl_32b_minimax_h3-q2_k_m.gguf --cfg-scale 1.0 -W 480 -H 480 --diffusion-fa -p "a running pig" --video-frames 124 -o video.avi
```

*gk engine is now adopting an auto residency streaming for machine doesn't has enough vram, it works for low tier gpu with even less than 6gb vram
