# lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B

Источник: https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B · скачано 2026-11-28
Загрузок: 408 · лайков: 51 · базовая: Qwen/Qwen3-VL-8B-Instruct · теги: lora · обновлено: 2026-08-18T12:17:20.000Z

---

---
base_model: Qwen/Qwen3-VL-8B-Instruct
base_model_relation: adapter
library_name: peft
pipeline_tag: image-text-to-text
tags:
- peft
- lora
- safetensors
- qwen3-vl
- minimax-h3
- prompt-rewriting
- multimodal
- audio-video-generation
- text-to-audio-video
- image-to-audio-video
language:
- en
---

# MiniMax-H3 Prompt Rewriter LoRA 8B

[![Model](https://img.shields.io/badge/Hugging%20Face-model-yellow)](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B)
[![LightX2V](https://img.shields.io/badge/LightX2V-GitHub-black)](https://github.com/ModelTC/LightX2V)
[![Base model](https://img.shields.io/badge/Base-Qwen3--VL--8B--Instruct-blue)](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

This repository contains a PEFT LoRA adapter that turns short user requests into production-oriented prompts for MiniMax-H3 joint audio-video generation. It is built on **Qwen3-VL-8B-Instruct** and supports text-only and keyframe-conditioned rewriting.

Compared with [MiniMax-H3-Prompt-Rewriter-LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA), this 8B variant accepts visual references and covers all four prompt-rewriting tasks used to train this adapter:

- **T2VA / T2AV**: text to audio-video.
- **I2VA / I2AV**: first-frame image plus text to audio-video.
- **L2VA / L2AV**: last-frame image plus text to audio-video.
- **FL2VA / FL2AV / FLF2V**: first- and last-frame images plus text to audio-video.

Other MiniMax-H3 conditioning modes, including Ref2VA, are not supported by this adapter release.

The rewriter produces a structured prompt with an integrated shot timeline, synchronized physical and ambient sound, and non-diegetic music guidance. This repository contains the prompt rewriter only; MiniMax-H3 weights and a compatible inference framework such as LightX2V are still required to generate the final video.

## Repository layout

- `adapter_config.json` and `adapter_model.safetensors`: LoRA files to place at the repository root.
- `infer.py`: single-sample inference for all four tasks.
- `prompt_template.py`: the single source of truth for the system prompt, task aliases, and image/message ordering.
- `requirements.txt`: minimal inference dependencies.

## Installation

```bash
git lfs install
hf auth login  # required while the repository is private
git clone https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B
cd MiniMax-H3-Prompt-Rewriter-LoRA-8B
pip install -r requirements.txt
```

The Qwen3-VL base weights are downloaded separately on first use. You can also pass a local base-model path with `--base-model`.

## Inference

The script automatically uses the adapter in the current checkout when `adapter_config.json` is present. Otherwise it downloads the adapter from this Hugging Face repository. Greedy decoding is recommended for deterministic prompt rewriting.

### T2VA

```bash
python infer.py \
  --task t2va \
  --prompt "A corgi runs through a rainy neon-lit alley." \
  --duration 5 \
  --resolution 16:9 \
  --greedy \
  --output outputs/t2va.json
```

### I2VA

```bash
python infer.py \
  --task i2va \
  --first-frame /path/to/first_frame.jpg \
  --prompt "The subject turns toward the camera and smiles." \
  --duration 5 \
  --resolution adaptive \
  --greedy \
  --output outputs/i2va.json
```

### L2VA

```bash
python infer.py \
  --task l2va \
  --last-frame /path/to/last_frame.jpg \
  --prompt "A paper airplane crosses the room and lands in the shown final pose." \
  --duration 5 \
  --resolution adaptive \
  --greedy \
  --output outputs/l2va.json
```

### FL2VA

```bash
python infer.py \
  --task fl2va \
  --first-frame /path/to/first_frame.jpg \
  --last-frame /path/to/last_frame.jpg \
  --prompt "Create a continuous cinematic transition between the two reference frames." \
  --duration 5 \
  --resolution adaptive \
  --greedy \
  --output outputs/fl2va.json
```

Use `--prompt-file prompt.txt` instead of `--prompt` for long input. A `.json` output records the request metadata and writes the rewritten text to `enhanced_prompt`; any other extension writes plain text. Run `python infer.py --help` for sampling, dtype, processor pixel-budget, device-map, and local-path options.

To compare against the unadapted base model, add `--base-only`. This intentionally skips the LoRA while keeping the same system prompt and task-specific input layout.

## Using the rewritten prompt

Feed `enhanced_prompt` to the corresponding MiniMax-H3 task in LightX2V, preserving the same duration, resolution, and reference-frame roles:

- T2VA uses no image.
- I2VA uses the same first frame.
- L2VA uses the same last frame.
- FL2VA uses the same first frame followed by the same last frame.

Keeping the task geometry and image order unchanged is important because the rewriter describes their exact temporal roles.

In LightX2V, the corresponding task and reference-image arguments are:

- T2VA: `--task t2av`
- I2VA: `--task i2av --image_path FIRST_FRAME`
- L2VA: `--task l2av --last_frame_path LAST_FRAME`
- FL2VA: `--task fl2av --image_path FIRST_FRAME --last_frame_path LAST_FRAME`

## System prompt

The complete system instruction and all task-dependent message construction live in `prompt_template.py`. `infer.py` imports that file directly and renders the conversation with `enable_thinking=False`, matching the training input format. Edit that one file if you need a custom prompting policy.

## Limitations

- The adapter is a learned approximation of prompt rewriting and is not the hosted MiniMax Context-IR service.
- It can add plausible creative detail beyond a very short request; review identity, dialogue, visible text, and safety-sensitive content before generation.
- Image-conditioned quality depends on the clarity and consistency of the supplied reference frames.
- Final audio-video quality depends on the downstream MiniMax-H3 checkpoint, inference settings, and task geometry.
- Use of the adapter, Qwen base model, and downstream video model remains subject to their respective licenses and terms.

## Acknowledgements

- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [LightX2V](https://github.com/ModelTC/LightX2V)
- [PEFT](https://github.com/huggingface/peft)
