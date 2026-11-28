# pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF

Источник: https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF · скачано 2026-11-28
Загрузок: 4453 · лайков: 5 · базовая: lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA · теги: gguf,base_model:quantized:lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA · обновлено: 2026-11-12T12:53:51.000Z

---

---
license: apache-2.0
base_model: lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
language:
  - en
tags:
  - gguf
  - minimax-h3
  - prompt-rewriting
  - multimodal
  - fl2va
---

# MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF

GGUF conversions of [lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA).
The model rewrites user requests into structured MiniMax-H3 video and audio prompts.
These are complete model weights with a separate vision projector. Load the language
file with `-m` and the projector with `--mmproj`.

**Tasks:** T2VA, I2VA, L2VA and FL2VA. Use first/last frames in the order required by the task.

**Output:** `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`, plus frame alignment for image tasks.

## Files

| File | Download size | Contents |
|---|---:|---|
| [MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-Q4_K_M.gguf](MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-Q4_K_M.gguf) | 5.629 GB | Q4_K_M |
| [MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-Q5_K_M.gguf](MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-Q5_K_M.gguf) | 6.468 GB | Q5_K_M |
| [mmproj-MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-F16.gguf](mmproj-MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-F16.gguf) | 0.918 GB | Vision projector, F16 |

Sizes are decimal GB of files on disk. Runtime memory also includes the context,
vision processing and backend allocations. Both quantizations use the same F16
projector extracted from this checkpoint.

`conversion.json` records source/output SHA-256 hashes and the conversion tools.
The BF16 control used for validation is not required to run these files.

## ComfyUI

Install [MiniMax-H3 Prompt Rewriter](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)
and add **MiniMax-H3 Qwen Rewriter** with **Qwen Rewriter Options**. Choose this
checkpoint in **Model list**, or put the model and projector together under
`ComfyUI/models/LLM/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF/` for local discovery.

The node provides input checkboxes, the ordered reference strip, **+ instr**,
aspect ratio, duration, bypass/repeat and the prompt library. Its `references`
output carries the original enabled conditioning to Reference Slots or Reference
Encode. Select a task supported by this checkpoint.

## llama.cpp

Use a build that supports `qwen35` and its Qwen3-VL vision projector. Conversion
used llama.cpp commit `cb26014d965036f0eacdab7387b222d28d36b9e6` (`b10310`).

```bash
hf download pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF --include '*Q5_K_M.gguf' 'mmproj-*.gguf' --local-dir .
llama-server -m MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-Q5_K_M.gguf \
  --mmproj mmproj-MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-F16.gguf --jinja \
  --ctx-size 32768 --n-gpu-layers 999 --host 127.0.0.1 --port 8081
```

Supply the matching official writing guide and task request, as in the
[upstream inference example](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA/blob/ea2a677e0597831a992a400224d98dde4c405fe6/post_vllm.py).
Image parts must stay interleaved with their labels. Video references are sampled
images with source timestamps. The ComfyUI node builds this request automatically.

The upstream defaults are `temperature=0.7`, `top_p=0.8`, `top_k=20` and
`max_tokens=16384`, with thinking enabled. A smaller budget can be used with
thinking disabled through `chat_template_kwargs.enable_thinking=false`.

**Audio:** this checkpoint has no audio encoder. Provide an explicit description
of the reference sound. The original Ref2VA helper labels audio but does not send
its waveform; these conversions do not add a hearing capability.

## Validation

Q5_K_M and Q4_K_M loaded on an NVIDIA GeForce RTX 5090 and returned the task sections from image references. The original SFT also completed a smoke test (BF16 for 9B, NF4 for 27B). Exact settings, timings and H3 findings are recorded in validation.json.

Quantization changes the weights and can change the wording. A successful smoke
test checks loading, media input and output structure; it is not a quality score.

## Conversion and source

Source: [lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA at ea2a677e0597831a992a400224d98dde4c405fe6](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA/tree/ea2a677e0597831a992a400224d98dde4c405fe6).
The official `convert_hf_to_gguf.py` produced a BF16 language-model control with
`--no-nextn` and an F16 vision projector with `--mmproj` from the same SFT
directory. `llama-quantize` produced Q5_K_M and Q4_K_M language files. The vision
projector was kept at F16; no additional LoRA is needed.

The writing guides are fetched from their original source by the node and are
not redistributed in this repository.

## Licence and credits

The original checkpoint includes an Apache 2.0 licence, copied here as `LICENSE`. Credit for the fine-tuned model belongs to LightX2V and for its base model to Qwen.

Prompt format and video generation: [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
