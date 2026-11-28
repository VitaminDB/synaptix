# lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni

Источник: https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni · скачано 2026-11-28
Загрузок: 273 · лайков: 59 · базовая: Qwen/Qwen2.5-Omni-7B · теги: lora · обновлено: 2026-08-26T07:51:50.000Z

---

---
base_model: Qwen/Qwen2.5-Omni-7B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
  - peft
  - lora
  - safetensors
  - qwen2.5-omni
  - multimodal
  - prompt-rewriting
  - minimax-h3
  - text-to-audio-video
  - image-to-audio-video
language:
  - en
  - zh
license: other
license_name: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/LICENSE
---

# MiniMax-H3 Omni Prompt Rewriter LoRA

This repository contains a PEFT LoRA adapter for
[Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B). It rewrites a
short user request and its optional visual, video, or audio references into a
structured, production-ready MiniMax-H3 audio-video prompt.

> [!IMPORTANT]
> This model is a **prompt rewriter**. It outputs text only; it does not render
> video or audio. Feed `enhanced_prompt` and the same conditioning assets into
> a compatible MiniMax-H3 generation pipeline to create the final result.

## Supported tasks

| Task | Input accepted by this adapter | Reference role | Output schema |
|---|---|---|---|
| **T2AV** | Text only | No reference media | Base three-field schema |
| **I2AV** | Text + exactly one image | Exact first frame | Base three-field schema |
| **L2AV** | Text + exactly one image | Exact last frame | Base three-field schema |
| **FL2AV** | Text + exactly two ordered images | Exact first and last frames | Base three-field schema |
| **Ref2AV** | Text + one or more ordered images, videos, and/or audio files | Full-reference subjects, composition, motion, timing, or sound | Ref six-section schema |

The Base tasks produce these fields in order:

```text
integrated_multimodal_description:
overall_soundscape:
non_diegetic_music:
```

I2AV, L2AV, and FL2AV additionally receive the task-specific image-alignment
instruction required by MiniMax-H3. Ref2AV produces these sections in order:

```text
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
```

Requests use an integer duration from 4 through 15 seconds. The script maps it
to MiniMax-H3's legal `17*n+5` frame grid at 24 fps and supplies the resulting
effective duration to the rewriter. Base tasks support `adaptive`, `21:9`,
`16:9`, `4:3`, `1:1`, `3:4`, and `9:16`; Ref2AV supports `16:9` and `9:16`.

## Repository layout

```text
.
├── .gitattributes
├── .gitignore
├── adapter_config.json
├── adapter_model.safetensors
├── infer.py
├── system_prompt.py
├── requirements.txt
└── assets/
    └── examples/
        ├── eval_cases.json
        ├── i2av/picture_1.jpg
        ├── l2av/picture_1.jpg
        ├── fl2av/picture_1.jpg
        ├── fl2av/picture_2.jpg
        ├── ref2av/picture_1.jpg
        └── ref2av/picture_2.jpg
```

`assets/examples/eval_cases.json` contains one bundled validation case for
each task. Media paths inside JSON or JSONL input are resolved relative to the
input file, so the file can be run directly from this repository.

## Installation

Clone or download this repository with Git LFS, enter its root directory, then
install the inference dependencies:

```bash
git lfs install
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A CUDA GPU with BF16 support is recommended. Install FFmpeg on the host when
using video or audio references. By default, `infer.py` downloads the base
model from `Qwen/Qwen2.5-Omni-7B`, loads the adapter from the current repository,
uses SDPA, and places the model with `device_map="auto"`. Local model and adapter
directories can be selected with `--model-path` and `--adapter-path`.

## Single-request inference

All commands below are run from the repository root. The generated prompt is
printed to stdout. When `--output` ends in `.json`, the script also writes a
structured JSON object; `.jsonl` writes the same object as one JSONL row. Any
other suffix writes plain prompt text.

### T2AV

```bash
python infer.py \
  --model_path /path/Qwen2.5-Omni-7B/ \
  --task t2av \
  --prompt "A cinematic fox walks through a snowy forest while distant branches crack in the wind." \
  --duration 15 \
  --resolution 16:9 \
  --output outputs/t2av.json
```

### I2AV

```bash
python infer.py \
  --model_path /path/Qwen2.5-Omni-7B/ \
  --task i2av \
  --prompt "The woman calmly gathers her wet hair into a ponytail in front of the mirror." \
  --duration 15 \
  --resolution adaptive \
  --first-frame assets/examples/i2av/picture_1.jpg \
  --output outputs/i2av.json
```

`--image` is an alias-style convenience input for the single required image,
but `--first-frame` makes its role explicit.

### L2AV

```bash
python infer.py \
  --model_path /path/Qwen2.5-Omni-7B/ \
  --task l2av \
  --prompt "Begin with an abstract blur and gradually reveal a sunlit field of small yellow flowers." \
  --duration 11 \
  --resolution adaptive \
  --last-frame assets/examples/l2av/picture_1.jpg \
  --output outputs/l2av.json
```

### FL2AV

```bash
python infer.py \
  --model_path /path/Qwen2.5-Omni-7B/ \
  --task fl2av \
  --prompt "Create a continuous macro shot of water droplets moving naturally across the green leaf." \
  --duration 5 \
  --resolution adaptive \
  --first-frame assets/examples/fl2av/picture_1.jpg \
  --last-frame assets/examples/fl2av/picture_2.jpg \
  --output outputs/fl2av.json
```

### Ref2AV

Ref2AV references preserve command-line order. Labels are numbered separately
by media type, so the first image is `<Picture 1>`, the first video is
`<Video 1>`, and the first audio file is `<Audio 1>`.

```bash
python infer.py \
  --model_path /path/Qwen2.5-Omni-7B/ \
  --task ref2av \
  --prompt "Create a podcast scene. Use <Picture 1> for the host and studio, and <Picture 2> for the guest and opposite seating position. The host speaks animatedly while the guest listens." \
  --duration 10 \
  --resolution 16:9 \
  --reference image=assets/examples/ref2av/picture_1.jpg \
  --reference image=assets/examples/ref2av/picture_2.jpg \
  --output outputs/ref2av.json
```

For mixed Ref2AV media, repeat `--reference TYPE=PATH` in the intended order:

```bash
python infer.py \
  --model_path /path/Qwen2.5-Omni-7B/ \
  --task ref2av \
  --prompt "Use <Picture 1> for the protagonist, <Video 1> for camera rhythm, and <Audio 1> for the soundtrack." \
  --duration 12 \
  --resolution 9:16 \
  --reference image=/path/to/character.jpg \
  --reference video=/path/to/motion.mp4 \
  --reference audio=/path/to/soundtrack.wav \
  --video-fps 1.0 \
  --output outputs/ref2av_mixed.json
```

Do not combine `--reference` with `--image`, `--video`, `--audio`,
`--first-frame`, or `--last-frame` in one command.

### Ref2AV label and media constraints

- The raw prompt must mention every supplied label and must not mention labels
  that were not supplied. For example, two images and one video require
  `<Picture 1>`, `<Picture 2>`, and `<Video 1>` in the prompt.
- Numbering is independent for pictures, videos, and audio, while asset order
  is the order of the repeated `--reference` arguments.
- Ref2AV requires at least one reference asset. Image-only, video-only,
  audio-only, and mixed-media requests are accepted.
- The script accepts at most 9 images, 3 videos, 3 audio files, and 12 total
  reference assets.
- Embedded video audio is not used. Supply a separate labeled audio reference
  when audio content must be retained or imitated.

## Batch inference

`--input-file` accepts either a JSON list or JSONL. The model is loaded once
and heterogeneous requests are generated sequentially. This avoids forcing
different multimodal shapes into one padded tensor batch.

Run all five bundled validation cases:

```bash
python infer.py \
  --model_path /path/Qwen2.5-Omni-7B/ \
  --input-file assets/examples/eval_cases.json \
  --output outputs/validation_rewrites.jsonl \
  --continue-on-error
```

A minimal JSON request has this shape:

```json
[
  {
    "id": "example-i2av-001",
    "task": "i2av",
    "prompt": "The subject turns toward the window as sunlight fills the room.",
    "duration": 8,
    "resolution": "16:9",
    "references": [
      {
        "type": "image",
        "path": "media/first_frame.jpg"
      }
    ]
  }
]
```

Every batch row receives an ID. If `id` is omitted, the loader assigns a stable
`row-000001`-style ID for that input order. Optional `order` and `label` fields
are validated when present; the script always derives their canonical values.

## Output format

A successful JSON result contains:

- `id` for a batch row or a single request supplied with `--id`;
- normalized `task`, original `prompt`, and `resolution`;
- `requested_duration`, `target_num_frames`, and `effective_duration`;
- canonical ordered `references` with type, label, order, and resolved path;
- `base_model` and `checkpoint` provenance;
- `schema_ok`, a lightweight check that required section names occur in order;
- `enhanced_prompt`, the rewritten MiniMax-H3 prompt.

`schema_ok` validates section presence and ordering only; it is not a semantic
quality score.

For a base-model comparison, add `--base-only`. This skips the LoRA adapter but
keeps the same processor, task formatting, system prompt, and validation path.

## Limitations

- This is a learned prompt rewriter, not the official MiniMax Context-IR API;
  outputs can omit details or require manual editing.
- The adapter does not generate video or audio and cannot assess the final
  render without a downstream generator.
- Reference identity, motion, dialogue, and timing retention depend on media
  quality, decoding, the requested duration, and downstream generation.
- Long or dense multimodal requests may approach the base model's context limit.
  The script reduces `max_new_tokens` when necessary, but the request itself
  must still fit the available context.
- Users are responsible for reference-media rights, consent, privacy, and
  compliance with the licenses and policies of both the base model and the
  downstream video generator.

## Acknowledgements

- [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) provides the
  multimodal base model.
- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) defines the target
  audio-video prompt format and generation model family.

This LoRA adapter does not change the licenses of the base model, training data,
or downstream generator. Review and follow all applicable terms before use.
