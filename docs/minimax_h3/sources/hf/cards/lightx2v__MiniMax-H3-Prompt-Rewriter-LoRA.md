# lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA

Источник: https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA · скачано 2026-11-28
Загрузок: 198 · лайков: 200 · базовая: Qwen/Qwen3.6-27B · теги: lora · обновлено: 2026-10-16T06:45:17.000Z

---

---
base_model: Qwen/Qwen3.6-27B
library_name: peft
tags:
  - peft
  - lora
  - prompt-rewriting
  - minimax-h3
  - text-to-audio-video
  - audio-video-generation
language:
  - en
---

# MiniMax-H3 T2VA Prompt Rewriter LoRA

[![LightX2V](https://img.shields.io/badge/Inference-LightX2V-ff6b35)](https://github.com/ModelTC/LightX2V)
[![GitHub stars](https://img.shields.io/github/stars/ModelTC/LightX2V?style=social)](https://github.com/ModelTC/LightX2V)
[![MiniMax-H3](https://img.shields.io/badge/Base%20Generator-MiniMax--H3-yellow)](https://huggingface.co/MiniMaxAI/MiniMax-H3)

An open, local prompt rewriter for **text-to-audio-video (T2VA)** generation with
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), fine-tuned as a LoRA
adapter on top of [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B).

> **Generate the final audio-video with
> [LightX2V](https://github.com/ModelTC/LightX2V).** LightX2V provides optimized local
> MiniMax-H3 inference, including T2AV and other H3 tasks, memory offloading, tensor and
> sequence parallelism, quantized checkpoints, optimized attention/RMSNorm/RoPE kernels,
> and feature caching. If this project is useful, please consider giving
> **[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) a star**.

The complete local workflow is:

```text
Short prompt ──► this Prompt Rewriter LoRA ──► structured H3 prompt
                                                        │
Official MiniMax-H3 weights ──► LightX2V inference ◄─────┘
                                      │
                                      ▼
                          synchronized video + audio
```

The [official MiniMax-H3 repository](https://huggingface.co/MiniMaxAI/MiniMax-H3)
provides the upstream model weights. **LightX2V is the recommended inference entry
point for the workflow documented here.**

## What it does

The adapter turns a short prompt plus the requested aspect ratio and duration into a
structured audio-video description:

```text
Original prompt + aspect ratio + duration
                    │
                    ▼
          Qwen3.6-27B + this LoRA
                    │
                    ▼
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
```

The rewritten prompt expands shot structure, timing, composition, camera motion,
physical action, continuity, synchronized diegetic sound, and non-diegetic music while
preserving the original intent.

| Component | Purpose | Model |
|---|---|---|
| Inference framework | Runs MiniMax-H3 locally and generates the final synchronized video and audio | **[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V)** |
| Generator weights | Upstream MiniMax-H3 model weights consumed by LightX2V | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| Base language model | General instruction-following backbone | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| LoRA adapter | H3-oriented T2VA prompt rewriting | [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA) |

## Status and roadmap

- [x] T2VA prompt rewriting
- [ ] First/last-frame-to-audio-video (FL2VA) prompt rewriting
- [ ] Reference-to-audio-video (Ref2VA) prompt rewriting

## Installation

Clone this repository and install the inference dependencies:

```bash
git clone https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA
cd MiniMax-H3-Prompt-Rewriter-LoRA
pip install -r requirements.txt
```

The 27B base model is not included in this adapter repository and will be downloaded
from Hugging Face on first use. Ensure that the machine has enough GPU/CPU memory for
Qwen3.6-27B. `device_map="auto"` is used to distribute the model over available devices.

## Rewrite one prompt

```bash
python infer.py \
  --prompt "Epic space-opera teaser: a captain watches the last fleet jump away, leaving her alone." \
  --duration 10 \
  --resolution 16:9 \
  --greedy
```

Local paths are supported as well:

```bash
python infer.py \
  --base-model /path/to/Qwen3.6-27B \
  --adapter /path/to/MiniMax-H3-Prompt-Rewriter-LoRA \
  --prompt "A red fox walks through a snowy forest at dawn." \
  --duration 15 \
  --resolution 21:9 \
  --output rewritten_prompt.json
```

Use `--output result.txt` for plain text or a `.json` path for a record containing
the original prompt, conditions, model IDs, and rewritten prompt.

To run the base-model baseline without the LoRA:

```bash
python infer.py \
  --base-only \
  --prompt "A red fox walks through a snowy forest at dawn." \
  --duration 15 \
  --resolution 21:9 \
  --greedy
```

For fair comparisons, use the same decoding mode and seed for every rewriter. Greedy
decoding is the simplest deterministic setting; sampling is enabled when `--greedy`
is omitted.

## Generate audio-video with LightX2V

### 1. Install LightX2V

Building from source gives access to the latest MiniMax-H3 integration:

```bash
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
uv pip install -v .  # or: pip install -v .
```

Follow the [LightX2V Quick Start](https://lightx2v-en.readthedocs.io/en/latest/getting_started/quickstart.html)
to install the optional optimized attention and quantization operators required by the
selected configuration.

### 2. Download the MiniMax-H3 weights

The generator weights come from the upstream
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) repository:

```bash
hf download MiniMaxAI/MiniMax-H3 --local-dir /path/to/MiniMax-H3
```

The LoRA in this repository rewrites text; it does not replace or include the
MiniMax-H3 generator weights.

### 3. Rewrite the prompt

Run this command from the prompt-rewriter repository:

```bash
python infer.py \
  --prompt "A red fox walks through a snowy forest at dawn." \
  --duration 15 \
  --resolution 16:9 \
  --greedy \
  --output /tmp/h3_rewritten_prompt.txt
```

### 4. Generate synchronized video and audio

Run the following commands from the LightX2V repository root.

Before running:
- Set `num_frames`, `fps`, and `size` in `configs/minimax_h3/minimax_h3.json` to match the duration and aspect ratio used for prompt rewriting.
- Follow the [AdaLN cache instructions](https://github.com/ModelTC/LightX2V/blob/main/scripts/minimax_h3/README.md#adaln-cache) to generate the initial cache. Use the same model directory, inference configuration, and `--model-variant fl2av`.

```bash
lightx2v_path="$(pwd)"
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32

python -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3/minimax_h3.json" \
  --prompt "$(</tmp/h3_rewritten_prompt.txt)" \
  --save_result_path save_results/minimax_h3_t2av.mp4 \
  --seed 42
```

The generated MP4 includes video and synchronized audio.

Choose a configuration for your hardware:

| Mode | Configuration |
|---|---|
| Single GPU | `configs/minimax_h3/minimax_h3.json` |
| Block offload | `configs/minimax_h3/minimax_h3_block_offload.json` |
| Sequence parallelism | `configs/minimax_h3/minimax_h3_sp.json` |
| Tensor parallelism | `configs/minimax_h3/minimax_h3_tp.json` |
| Tensor + sequence parallelism | `configs/minimax_h3/minimax_h3_tp_sp.json` |

For multi-GPU inference, use the [parallel launcher](https://github.com/ModelTC/LightX2V/blob/main/scripts/minimax_h3/run_minimax_h3_t2av_parallel.sh). Set the configuration path, visible GPUs, and process count to match your chosen configuration.

See the [MiniMax-H3 guide](https://github.com/ModelTC/LightX2V/blob/main/scripts/minimax_h3/README.md) for more details.

## Qualitative comparison

The following gallery is designed to compare videos generated through
[LightX2V](https://github.com/ModelTC/LightX2V) with the same MiniMax-H3 checkpoint and
identical inference settings. Only the prompt-rewriting method should differ.

| Label | Meaning |
|---|---|
| Original Prompt (No Rewrite) | The raw user prompt is sent directly to MiniMax-H3 |
| MiniMax Context-IR | The raw prompt is enhanced by the official hosted Context-IR workflow |
| Qwen3.6-27B Base Rewriter | The same template is used with the base model and no LoRA |
| **H3-T2VA Context Rewriter LoRA (Ours)** | Qwen3.6-27B with this task-specific LoRA |

### Prompt 1

> **Original prompt:** Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as the last fleet gathers and jumps away in a blinding flash, the bridge shaking, leaving her behind.

| Original Prompt (No Rewrite) | MiniMax Context-IR | Qwen3.6-27B Base Rewriter | **H3-T2VA Context Rewriter LoRA (Ours)** |
|---|---|---|---|
| <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/_gF8xSGc7-Z-3Ggsyk6w9.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/haF7Ay70aadZkKfq-at2i.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/Ztp7sdxWxMHLy3Sda6bjT.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/2pFOcxgj_WjvsnRMUcVWO.mp4"></video> |

### Prompt 2

> **Original prompt:** A dynamic TV promo-style video showcasing a group of politicians at a nighttime event illuminated by vibrant neon lights. The politicians, dressed in formal attire, engage in conversations, shake hands, and exchange smiles as they navigate through a bustling room filled with guests. The scene is set in a modern, urban environment with a mix of soft ambient lighting and bright neon signs. The camera alternates between close-ups of expressive faces and wider shots capturing the energetic atmosphere. The visuals include fast-paced cuts and subtle zoom-ins to highlight key interactions and facial expressions, emphasizing the excitement and importance of the political networking event.

| Original Prompt (No Rewrite) | MiniMax Context-IR | Qwen3.6-27B Base Rewriter | **H3-T2VA Context Rewriter LoRA (Ours)** |
|---|---|---|---|
| <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/9F0i0ZwRT8sST-weN0L_q.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/_QevAfxiAf8_-fuwRibKX.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/xmdrQXfSwNyORY6bkMAZ7.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/fEhON5N5rM6JBLNqcAs1G.mp4"></video> |

### Prompt 3

> **Original prompt:** ASMR video featuring a close-up shot of a young woman's hands. The girl has long, neatly manicured nails and soft, smooth skin. She moves her hands gracefully in front of the camera, focusing on gentle hand movements such as tapping, sliding, and fluttering. The lighting is warm and soft, casting a gentle glow on her hands. The camera remains static but maintains a close-up focus on her hands throughout the sequence. The background is blurred to keep the viewer's attention on the hands. The movements are slow and deliberate, creating a soothing and relaxing atmosphere.

| Original Prompt (No Rewrite) | MiniMax Context-IR | Qwen3.6-27B Base Rewriter | **H3-T2VA Context Rewriter LoRA (Ours)** |
|---|---|---|---|
| <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/mOQ8QidVn7Fc64UYSBcZR.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/ULjXofNgjvs2aud70jdy8.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/JgjzFsmiMWEfI7P7hERaQ.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/Aze2ErBmM9MtsE7cRNAL5.mp4"></video> |

### Prompt 4

> **Original prompt:** A man in casual attire reaches for a gaming joystick from a shelf in a well-lit electronics store. He is standing in front of a display filled with various gaming consoles and accessories. The store background includes rows of shelves stocked with electronic gadgets, creating a bustling retail atmosphere. The man's posture is upright, with a focused and excited expression as he selects the joystick. The scene captures a close-up mid-shot, emphasizing the man's interaction with the joystick.  

| Original Prompt (No Rewrite) | MiniMax Context-IR | Qwen3.6-27B Base Rewriter | **H3-T2VA Context Rewriter LoRA (Ours)** |
|---|---|---|---|
| <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/PfSo5D_lIe-ERk1yUS5GZ.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/iJkM5xdnfOojBo2gKQyEI.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/mFuOxCyycFw9ZwwiJQM6k.mp4"></video> | <video controls src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/xnDSTpEpofplVIRCI5eBY.mp4"></video> |

Expected video paths and filenames are documented in
[`assets/examples/UPLOAD_LAYOUT.md`](assets/examples/UPLOAD_LAYOUT.md).

## Limitations

- This release supports text-only T2VA prompt rewriting; it does not consume images,
  videos, or audio references.
- It is a learned approximation, not an open-source release or exact replica of the
  official H3-Context-IR service.
- The generated rewrite may add details not explicitly present in a short prompt.
  Review the result before generation when identity, dialogue, timing, or composition
  must be exact.
- Final quality depends on both prompt rewriting and the MiniMax-H3 inference settings
  selected in LightX2V.
- Use of MiniMax-H3 is governed by the license and acceptable-use terms published in
  the [official MiniMax-H3 repository](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## Acknowledgements

This project is developed by the
[LightX2V](https://github.com/ModelTC/LightX2V) community to make local MiniMax-H3
prompt rewriting and audio-video generation easier and more efficient. It is built for
the [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) audio-video generation
model and fine-tuned from [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B).

If you use this adapter or the MiniMax-H3 integration, please visit and star
**[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V)**, where inference support,
performance optimizations, examples, and future task updates are maintained.
