# pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF

Источник: https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF · скачано 2026-11-28
Загрузок: 1680 · лайков: 9 · базовая: lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni · теги: gguf,lora,base_model:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni,base_model:adapter:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni · обновлено: 2026-08-26T17:20:38.000Z

---

---
base_model: lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni
base_model_relation: adapter
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - lora
  - llama.cpp
  - qwen2.5-omni
  - multimodal
  - prompt-rewriting
  - minimax-h3
  - text-to-audio-video
  - image-to-audio-video
language:
  - en
---

[![GitHub](https://img.shields.io/badge/GitHub-ComfyUI%20Node-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)

# MiniMax-H3 Prompt Rewriter LoRA Omni — GGUF

A GGUF conversion of
[lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni),
so the omni-modal rewriter runs under
[llama.cpp](https://github.com/ggml-org/llama.cpp) against a quantized
Qwen2.5-Omni-7B.

This repository contains no new training — only a format conversion. The tensor
values are unchanged.

## What is different about this one

It is the first of the three that both **sees and hears**, and the first that
covers every task in one adapter.

| Adapter | Base | Reads | Tasks |
|---|---|---|---|
| [27B](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF) | Qwen3.6-27B | text | T2AV |
| [8B](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-8B-GGUF) | Qwen3-VL-8B | pictures | T2AV, I2AV, L2AV, FL2AV |
| this one | Qwen2.5-Omni-7B | pictures, clips, sound | all four, plus Ref2AV |

Ref2AV is the difference that matters. It is the full-reference mode — six
output fields instead of three, a retention analysis, and `<Subject N>`,
`<Picture N>`, `<Video N>` and `<Audio N>` labels the rewrite has to honour.
Until now a reference asset had to be described in words first; this base model
takes the asset itself, sound included.

| Task | References | Meaning |
|---|---|---|
| T2AV | none | text only |
| I2AV | 1 picture | the picture is the first frame |
| L2AV | 1 picture | the picture is the final frame |
| FL2AV | 2 pictures | first and last frame |
| Ref2AV | any mix | pictures, clips and sounds the target video reuses |

The two system prompts that go with these — one for the four frame tasks, one
for Ref2AV — are in
[`system_prompt.py`](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni/blob/main/system_prompt.py)
in the source repository. The adapter was trained on that exact wording.

## Contents

| File | Size | Notes |
|---|---|---|
| `MiniMax-H3-Prompt-Rewriter-LoRA-Omni-F16.gguf` | 0.65 GB | 392 tensors, rank 128, `adapter.lora.alpha = 128` |
| `MiniMax-H3-Prompt-Rewriter-LoRA-Omni-Q8_0.gguf` | 0.34 GB | the same adapter, quantised |

Tensor count and alpha match the source adapter exactly. The adapter touches the
thinker's text tower only — `q/k/v/o` and `gate/up/down` across 28 layers; the
vision and audio towers are untouched, which is why 392 tensors cover the whole
thing.

Take `Q8_0` unless you have a reason not to. Names, order and every metadata
field are identical to the `F16`; only the stored values differ, by a mean of
0.005 relative across all 392 tensors and 0.006 on the worst one.

## Which base to run it against

Both files in [ggml-org/Qwen2.5-Omni-7B-GGUF](https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF),
since a multimodal model is a model plus a projector and the two have to come
from the same conversion:

| File | Size |
|---|---|
| `Qwen2.5-Omni-7B-Q4_K_M.gguf` | 4.68 GB |
| `mmproj-Qwen2.5-Omni-7B-Q8_0.gguf` | 1.55 GB |

About 9 GB of VRAM with this adapter attached and a 12k context. The projector
is only needed for tasks that carry references; T2AV runs without it.

## Use with llama.cpp

**Render the chat template yourself.** This is the one thing worth knowing
before anything else. `-no-cnv` feeds the model raw text with no template at
all, and Qwen2.5-Omni answers that by continuing the prompt: the output loops,
never opens `integrated_multimodal_description:`, and looks exactly like a
broken adapter. It is not — it is a missing `<|im_start|>`. Use `--jinja`, or
render the template and pass the result with `--file`.

Rendered correctly, T2AV needs no projector:

```bash
llama-completion \
  -m Qwen2.5-Omni-7B-Q4_K_M.gguf \
  --lora MiniMax-H3-Prompt-Rewriter-LoRA-Omni-Q8_0.gguf \
  --jinja -sysf system_prompt.txt \
  -p "Rewrite request:
task: T2AV
resolution: 16:9
effective_duration: 10s
raw_prompt: A lone astronaut walks toward a ruined tower on a red desert plain." \
  -st -n 1400 --temp 0 -ngl 99 -c 12288
```

A task with references goes through `llama-mtmd-cli`, which applies the template
itself and splices each asset in where its marker sits. Note the user turn's
field names: `effective_duration` and `raw_prompt`, not the `duration` and
`original_prompt` the 8B adapter was trained on.

```
Picture 1 — exact first frame at 0.00 seconds:
<__media__>

Rewrite request:
task: I2AV
resolution: 16:9
effective_duration: 10s
raw_prompt: ...
```

For Ref2AV the block of references opens with `Ordered MiniMax-H3 references:`
and each asset is introduced by its own label line —
[`infer.py`](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni/blob/main/infer.py)
in the source repository builds all of it.

## Use in ComfyUI

[MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)
ships a node for this adapter. Pick a Qwen2.5-Omni base, choose the task,
connect the references, and this adapter is fetched from here on first use. The
options node's `adapter` dropdown lists both precisions.

## What to expect of it

The trained shape arrives on the first try. With the adapter off, the same base
model and the same system prompt still print the three field names — and then
fills `integrated_multimodal_description` with a single sentence, no `[Shot N]`
markers and no camera language. With it on, the same request produces a shot
list with composition, lighting, camera motion and timing. That contrast is the
simplest way to confirm the adapter is actually attached.

## How it was converted

`convert_lora_to_gguf.py` from llama.cpp, with nothing done to it. All 392
tensors map cleanly onto `blk.N.attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down.weight.lora_a|lora_b`
with no renaming, even though the adapter's own tensor names carry no `thinker.`
prefix.

One thing does differ from the 8B conversion: `--base-model-id` cannot be used.
Qwen2.5-Omni's `config.json` keeps the text parameters under
`thinker_config.text_config`, and the hub path does not flatten it, so the
converter stops at `could not find any of: ['n_layers', 'num_hidden_layers', ...]`.
Downloading that one 4 KB config into a directory and passing `--base` works,
because the local path does rename `thinker_config` for exactly this case. The
base weights are still not needed.

```bash
mkdir omni-base
curl -L https://huggingface.co/Qwen/Qwen2.5-Omni-7B/resolve/main/config.json -o omni-base/config.json

python convert_lora_to_gguf.py --base omni-base --outtype f16  lora-omni
python convert_lora_to_gguf.py --base omni-base --outtype q8_0 lora-omni
```

Both files are separate runs of the converter, not one quantised from the other.

## Licence

The weights originate from
[lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni)
and were converted without modifying their values. Whatever terms that
repository carries apply here unchanged; this repository claims nothing
additional.

Use of MiniMax-H3 itself is governed by the licence and acceptable-use terms in
the [official MiniMax-H3 repository](https://huggingface.co/MiniMaxAI/MiniMax-H3).
