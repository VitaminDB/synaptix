# lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA

Источник: https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA · скачано 2026-11-28
Загрузок: 8509 · лайков: 14 · базовая:  · теги:  · обновлено: 2026-10-29T15:05:45.000Z

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
  - reference-conditioned
  - image-reference
  - video-reference
  - audio-reference
  - ref2va
---

# MiniMax-H3 Prompt Rewriter (Ref2VA)

This model rewrites prompts for MiniMax-H3 video and audio generation. It supports
Ref2VA prompts with image, video, and audio references.

This model was trained through supervised fine-tuning (SFT) of
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).

## Setup

Use a Python environment with vLLM and GPU support on the server:

```bash
pip install vllm requests pillow qwen-vl-utils
```

All scripts use `python` from the active environment. For video references, install
`ffmpeg` and `ffprobe` and add them to `PATH`. The writing guide is included here.

## Run

1. Check `MODEL_PATH` and GPU settings in
   `vllm_server.sh`, then start the server:

   ```bash
   bash vllm_server.sh
   ```

2. In another terminal, edit `post_vllm.sh`: set `PROMPT`, `DURATION`, and at least
   one absolute path in `IMAGES`, `VIDEOS`, or `AUDIOS`, then run:

   ```bash
   bash post_vllm.sh
   ```

For a dry run, add `--dry_run` to the Python command in `post_vllm.sh`.

Each modality is numbered from 1 in array order, matching references such as
`image 1`, `video 1`, and `audio 1` in the prompt. Videos are sampled into
`VIDEO_FRAMES` images each (default: 8); images plus sampled frames must fit both
the client and server image limits (default: 50). Audio files are checked for
existence and represented by labels; their contents are not read or uploaded.

Keep client `MODEL` and `SERVER_URL` aligned with the server's `MODEL_NAME`,
`HOST`, and `PORT`. Run the commands from this directory, or use absolute script
paths from elsewhere.

Both example servers use GPU 0 and port 8000. To run both models at once, assign
separate GPUs and ports and update each client's `SERVER_URL`.
The 32768-token context includes text, image/frame tokens, and output; the output
limit is 16384. Reference counts alone do not guarantee that a request fits.

Outputs in `infer_outputs/` are overwritten on each run: `result.request.json`
contains the request, `result.prompt.txt` previews its text, and `result.json`
contains the response, reasoning, parameters, usage, and timing. Dry runs save
only the request and preview. Run `python post_vllm.py --help` for all options.
