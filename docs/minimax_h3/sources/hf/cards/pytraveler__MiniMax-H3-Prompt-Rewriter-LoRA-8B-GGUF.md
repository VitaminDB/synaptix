# pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-8B-GGUF

Источник: https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-8B-GGUF · скачано 2026-11-28
Загрузок: 1036 · лайков: 3 · базовая: lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B · теги: gguf,lora,base_model:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B,base_model:adapter:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B · обновлено: 2026-08-18T21:46:34.000Z

---

---
base_model: lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B
base_model_relation: adapter
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - lora
  - llama.cpp
  - qwen3-vl
  - multimodal
  - prompt-rewriting
  - minimax-h3
  - text-to-audio-video
  - image-to-audio-video
language:
  - en
---

[![GitHub](https://img.shields.io/badge/GitHub-ComfyUI%20Node-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)

# MiniMax-H3 Prompt Rewriter LoRA 8B — GGUF

A GGUF conversion of
[lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B),
so the multimodal rewriter runs under
[llama.cpp](https://github.com/ggml-org/llama.cpp) against a quantized
Qwen3-VL-8B-Instruct.

This repository contains no new training — only a format conversion. The tensor
values are unchanged.

## What is different about this one

The [27B rewriter](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF)
reads text and covers one task. This adapter's base model has a vision tower, so
it **looks at the reference frames** rather than being told about them in words,
and it covers four tasks:

| Task | Pictures | Meaning |
|---|---|---|
| T2AV | 0 | text only |
| I2AV | 1 | the picture is the first frame |
| L2AV | 1 | the picture is the final frame |
| FL2AV | 2 | first and last frame |

For everything but T2AV the answer opens with an alignment sentence that
MiniMax-H3 itself reads, with the timestamps already filled in.

It is also much smaller than the 27B route:

| Base | Download | VRAM with this adapter |
|---|---|---|
| `Qwen3VL-8B-Instruct-Q4_K_M.gguf` + projector | 4.7 + 0.7 GB | ~9 GB |
| `Qwen3VL-8B-Instruct-Q8_0.gguf` + projector | 8.1 + 0.7 GB | ~13 GB |

A multimodal model is two files, and the projector has to come from the same
conversion as the base. Both are in
[Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF);
`mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf` is the smaller of the two projectors and
is what the numbers above assume.

## Contents

| File | Size | Notes |
|---|---|---|
| `MiniMax-H3-Prompt-Rewriter-LoRA-8B-F16.gguf` | 1.30 GB | 504 tensors, rank 256, `adapter.lora.alpha = 256` |
| `MiniMax-H3-Prompt-Rewriter-LoRA-8B-Q8_0.gguf` | 0.69 GB | the same adapter, quantised |

Tensor count and alpha match the source adapter exactly. The adapter touches the
text tower only — `q/k/v/o` and `gate/up/down` across 36 layers; the vision tower
is untouched, which is why 504 tensors cover the whole thing.

Take `Q8_0` unless you have a reason not to. Names, order and every metadata
field are identical to the `F16`; only the stored values differ, by a mean of
0.005 relative across all 504 tensors and 0.006 on the worst one. Both produce
the trained output shape: the three fields, `[Shot 2] At 00:04.500` cut markers,
`(S1)` speaker ids and `<d>[English] ...</d>` dialogue tags.

There is no `Q4_K_M`. `llama-quantize` treats an adapter GGUF as a model and
stops at `key not found in model: qwen3vl.context_length`, so `F16` and `Q8_0`
are the only two formats the tooling offers for a LoRA.

## Use with llama.cpp

A task with pictures goes through `llama-mtmd-cli`, and `<__media__>` is where
each one is spliced in. The number of markers must equal the number of `--image`
arguments, or tokenization is refused.

```bash
llama-mtmd-cli \
  -m Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \
  --lora MiniMax-H3-Prompt-Rewriter-LoRA-8B-Q8_0.gguf \
  --image first.png \
  -sys "$(cat system_prompt.txt)" \
  -p "Picture 1 — exact first frame at 0.00 seconds:
<__media__>
task: i2av
resolution: 16:9
duration: 10s
original_prompt: A lone astronaut walks toward a ruined tower on a red desert plain." \
  -n 1400 --temp 0 -ngl 99 -c 8192
```

T2AV has no pictures, so it needs no projector and runs under plain
`llama-completion`:

```bash
llama-completion \
  -m Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  --lora MiniMax-H3-Prompt-Rewriter-LoRA-8B-Q8_0.gguf \
  -sysf system_prompt.txt \
  -p "task: t2av
resolution: 16:9
duration: 10s
original_prompt: A lone astronaut walks toward a ruined tower on a red desert plain." \
  -no-cnv -st -n 1400 --temp 0 -ngl 99 -c 8192
```

`system_prompt.txt` must hold the exact system prompt from
[`prompt_template.py`](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B/blob/main/prompt_template.py)
in the source repository — the adapter was trained on that wording, and changing
it degrades the rewrite. That file also builds the user turn, including which
line introduces each picture, which differs per task.

`-st` on the second command is worth keeping: `llama-cli` enters chat mode on
this model despite `-no-cnv` and then waits for a second turn, so the run never
ends.

## Use in ComfyUI

[MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)
ships a node for this adapter — **MiniMax-H3 Prompt Rewriter 8B (sees frames)**.
Pick a Qwen3-VL base, choose the task, connect `first_frame` and `last_frame`
where the task wants them, and this adapter is fetched from here on first use.
The options node's `adapter` dropdown lists both precisions.

## What to expect of it

All four tasks produce the trained shape, and with the adapter off the same base
model stops writing `[Shot 2]` cut markers and answers about a third as long —
which is the simplest confirmation that it is attached.

It is an 8B, and that shows in one place: the alignment line's timestamp is
sometimes formatted to three decimal places instead of two, and on FL2AV the
final picture is occasionally credited to `Shot 1` rather than the last shot. The
27B adapter does not do this.

## How it was converted

`convert_lora_to_gguf.py` from llama.cpp, with nothing done to it. Qwen3-VL is an
ordinary attention architecture, so the V-head reordering that makes the
[27B conversion](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF)
awkward never comes up here. Both files are separate runs of the converter, not
one quantised from the other.

```bash
python convert_lora_to_gguf.py lora8b --outtype f16
python convert_lora_to_gguf.py lora8b --outtype q8_0
```

No `--base` is needed: the adapter's own `adapter_config.json` names
`Qwen/Qwen3-VL-8B-Instruct`, and the converter reads the base config from the
hub. Only the config is fetched — the base weights are not required.

## Licence

The weights originate from
[lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B)
and were converted without modifying their values. Whatever terms that
repository carries apply here unchanged; this repository claims nothing
additional.

Use of MiniMax-H3 itself is governed by the licence and acceptable-use terms in
the [official MiniMax-H3 repository](https://huggingface.co/MiniMaxAI/MiniMax-H3).
