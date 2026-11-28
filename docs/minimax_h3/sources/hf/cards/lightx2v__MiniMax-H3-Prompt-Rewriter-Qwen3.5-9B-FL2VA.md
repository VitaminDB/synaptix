# lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA

Источник: https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA · скачано 2026-11-28
Загрузок: 4234 · лайков: 6 · базовая:  · теги:  · обновлено: 2026-10-29T15:05:19.000Z

---

---
tags:
  - minimax-h3
  - prompt-rewriting
  - prompt-enhancement
  - prompt-expansion
  - video-prompt
  - audio-video-prompt
  - multimodal
  - vision-language
  - sft
  - qwen3.5
  - vllm
  - t2va
  - i2va
  - l2va
  - fl2va
---

# MiniMax-H3 Prompt Rewriter (FL2VA)

This model rewrites prompts for MiniMax-H3 video and audio generation. It supports
text-only prompts and optional first/last-frame images.

This model was trained through supervised fine-tuning (SFT) of
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).

## Setup

Use a Python environment with vLLM and GPU support on the server:

```bash
pip install vllm requests pillow qwen-vl-utils
```

All scripts use `python` from the active environment. The writing guide is included
in this directory.

## Run

1. Check `MODEL_PATH` and GPU settings in `vllm_server.sh`, then start the server:

   ```bash
   bash vllm_server.sh
   ```

2. In another terminal, edit `post_vllm.sh`: set `PROMPT`, `TASK_TYPE`, `DURATION`,
   and any image paths, then run:

   ```bash
   bash post_vllm.sh
   ```

| Task | Required images |
| --- | --- |
| `T2VA` | None |
| `I2VA` | `FIRST_IMAGE` |
| `L2VA` | `LAST_IMAGE` |
| `FL2VA` | `FIRST_IMAGE` and `LAST_IMAGE` |

Image paths must be absolute. Keep client `MODEL` and `SERVER_URL` aligned with
the server's `MODEL_NAME`, `HOST`, and `PORT`. Commands above assume this directory
is the working directory; the scripts also accept absolute paths from elsewhere.

Both example servers use GPU 0 and port 8000. To run both models at once, assign
separate GPUs and ports and update each client's `SERVER_URL`.
The 32768-token context includes the input and output; the output limit is 16384.

The client saves the enhanced prompt, reasoning, input, generation parameters,
usage, and timing to `infer_outputs/result.json`, overwriting an existing file.
Run `python post_vllm.py --help` for all options.
