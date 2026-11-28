# pytraveler__MiniMax-H3-Prompt-Rewriter-ComfyUI

Source: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI
Fetched: 2026-11-28

Description: Local ComfyUI nodes for MiniMax-H3 prompt rewriting, multimodal reference captioning, and AI video/audio workflows.
Stars: 445  Updated: 2026-11-23T15:40:47Z



---

## FILE: README.md (https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/blob/HEAD/README.md)

# MiniMax-H3 Prompt Writers & Rewriters for ComfyUI

ComfyUI nodes for creating, rewriting and checking structured video-and-audio
prompts for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). Choose
LightX2V LoRAs, [full Qwen SFT/GGUF checkpoints](docs/qwen_rewriter.md), local
models following MiniMax's writing guide, or the [API Universal Writer](docs/api_writer.md)
with a configured local server or remote provider.

The pack also includes MiniMax-H3-Image and P-H3 prompt expansion, reference conditioning, saved
RefMods, prompt reduction, presets and prompt/model libraries. Local writers
run on your machine; API requests go to the provider you configure.

[Русская версия](README_RU.md) · [中文版](README_ZH.md) · [Changelog](CHANGELOG.md)

<p align="center">
  <a href="https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI?display_name=tag"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <a href="https://registry.comfy.org/publishers/darkil/nodes/minimax-h3-prompt-rewriter"><img alt="ComfyUI Registry" src="https://img.shields.io/badge/ComfyUI-Registry-1B98E0"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA"><img alt="Hugging Face" src="docs/badges/hf-lora.svg"></a>
  <a href="https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF"><img alt="GGUF adapter, 27B" src="docs/badges/gguf-27b.svg"></a>
  <a href="https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-8B-GGUF"><img alt="GGUF adapter, 8B" src="docs/badges/gguf-8b.svg"></a>
  <a href="https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF"><img alt="GGUF adapter, Omni" src="docs/badges/gguf-omni.svg"></a>
  <a href="https://www.youtube.com/watch?v=h3rZTIRB_G8"><img alt="Video review, in English" src="https://img.shields.io/badge/YouTube-review%20(EN)-FF0000?logo=youtube&logoColor=white"></a>
  <a href="https://www.youtube.com/watch?v=PZd9fWX15VA"><img alt="Video review, in Russian" src="https://img.shields.io/badge/YouTube-review%20(RU)-FF0000?logo=youtube&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA"><img alt="LightX2V SFT: Qwen3.5 9B Ref2VA" src="docs/badges/hf-sft-9b-ref2va.svg"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA"><img alt="LightX2V SFT: Qwen3.5 9B FL2VA" src="docs/badges/hf-sft-9b-fl2va.svg"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA"><img alt="LightX2V SFT: Qwen3.8 27B Ref2VA" src="docs/badges/hf-sft-27b-ref2va.svg"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA"><img alt="LightX2V SFT: Qwen3.8 27B FL2VA" src="docs/badges/hf-sft-27b-fl2va.svg"></a>
</p>

![The rewriter node in ComfyUI between its options node and a text viewer: a short prompt in the node, and on the right the structured shot-by-shot description, soundscape and music fields](docs/node_preview.png)

The prompt may be in any language the base model reads; the rewrite comes back in
English, which is what MiniMax-H3 expects.

```text
"A red fox walks through a snowy forest at dawn."  +  16:9  +  15s
                              │
                              ▼
              Prompt Writer / Rewriter
                              │
                              ▼
   integrated_multimodal_description: [Shot 1] ... [Shot 2] 0:06 ...
   overall_soundscape: ...
   non_diegetic_music: ...
                              │
                              ▼
              MiniMax-H3 video + synchronized audio
```

Choose a node family for your model and references:

| Node family | Model or service | Reference input | Tasks |
|---|---|---|---|
| LoRA Rewriter 27B | Qwen3.6-27B + LightX2V LoRA | Text descriptions | T2VA |
| LoRA Rewriter 8B | Qwen3-VL-8B-Instruct + LoRA | Reads first/last-frame images directly | T2VA, I2VA, FL2VA, L2VA |
| LoRA Rewriter Omni | Qwen2.5-Omni-7B + LoRA | Images; video and audio on the GGUF path | T2VA, I2VA, FL2VA, L2VA, Ref2VA |
| [Qwen Rewriter](docs/qwen_rewriter.md) | Full LightX2V Qwen3.5 9B / Qwen3.8 27B, SFT or GGUF | Reads images and sampled video frames; sound uses text descriptions | FL2VA checkpoint: T2VA, I2VA, FL2VA, L2VA; Ref2VA checkpoint: Ref2VA |
| Guide-based writers | Local instruction-following GGUF | Media descriptions from captioners | T2VA, I2VA, FL2VA, L2VA, Ref2VA |
| [API Universal Writer](docs/api_writer.md) | Configured local server or remote provider | Images, sampled video and audio according to model/server capabilities; descriptions mode is also available | T2VA, I2VA, FL2VA, L2VA, Ref2VA |

The three LoRA variants are also available in [Universal Rewriter](#minimax-h3-universal-rewriter),
where a tab chooses the adapter. For guide-based writers,
[Reference Caption](#minimax-h3-reference-caption) and
[Multi Reference Caption](#minimax-h3-multi-reference-caption) turn media into text.
[Universal Writer](#minimax-h3-universal-writer) combines captioning and prompt
writing in one node, with draggable chips controlling the reference order.

And when the shortest way to a good prompt is somebody else's:
[Prompt Presets](#minimax-h3-prompt-presets) hands over one of a thousand
finished MiniMax-H3 prompts that ship inside the pack, narrowed by look and
subject, each with the frame of the clip it was written for and that clip a click
away. No model is loaded for it and nothing is downloaded.

If your card has 8 GB, skip to [the writer nodes](#minimax-h3-prompt-writer-t2vai2vafl2val2va).

## Contents

- [What you need before installing](#what-you-need-before-installing)
- [Install](#install)
  - [Example workflows](#example-workflows)
- [Nodes](#nodes)
  - [MiniMax-H3 Prompt Rewriter](#minimax-h3-prompt-rewriter)
  - [MiniMax-H3 Prompt Rewriter 8B (sees frames)](#minimax-h3-prompt-rewriter-8b-sees-frames)
  - [MiniMax-H3 Prompt Rewriter Omni (sees and hears)](#minimax-h3-prompt-rewriter-omni-sees-and-hears)
  - [MiniMax-H3 Qwen Rewriter](#minimax-h3-qwen-rewriter)
  - [MiniMax-H3 Universal Rewriter](#minimax-h3-universal-rewriter)
  - [MiniMax-H3 Rewriter Options](#minimax-h3-rewriter-options)
  - [MiniMax-H3 Prompt Writer (T2VA/I2VA/FL2VA/L2VA)](#minimax-h3-prompt-writer-t2vai2vafl2val2va)
  - [MiniMax-H3 Prompt Writer (Ref2VA)](#minimax-h3-prompt-writer-ref2va)
  - [MiniMax-H3 Universal Writer](#minimax-h3-universal-writer)
  - [MiniMax-H3 Image Prompt Writer](#minimax-h3-image-prompt-writer)
  - [P-H3 Prompt Writer (images)](#p-h3-prompt-writer-images)
  - [MiniMax-H3 Reference Caption](#minimax-h3-reference-caption)
  - [MiniMax-H3 Multi Reference Caption](#minimax-h3-multi-reference-caption)
  - [Captioning with a model ComfyUI already has loaded](#captioning-with-a-model-comfyui-already-has-loaded)
  - [The captioner is loaded once, not once per reference](#the-captioner-is-loaded-once-not-once-per-reference)
  - [MiniMax-H3 Guide Prompt (any LLM)](#minimax-h3-guide-prompt-any-llm)
  - [MiniMax-H3 Prompt Check](#minimax-h3-prompt-check)
  - [MiniMax-H3 Prompt Reducer](#minimax-h3-prompt-reducer)
  - [MiniMax-H3 Reduce Prompt (any LLM)](#minimax-h3-reduce-prompt-any-llm)
  - [MiniMax-H3 Effect Embeddings](#minimax-h3-effect-embeddings)
  - [MiniMax-H3 LoRA Triggers](#minimax-h3-lora-triggers)
  - [MiniMax-H3 Reference Adapter](#minimax-h3-reference-adapter)
  - [MiniMax-H3 Reference Slots](#minimax-h3-reference-slots)
  - [MiniMax-H3 RefMod](#minimax-h3-refmod)
  - [MiniMax-H3 Reference Encode](#minimax-h3-reference-encode)
  - [MiniMax-H3 Prompt Presets](#minimax-h3-prompt-presets)
  - [The duration widget](#the-duration-widget)
  - [Repeating the last answer](#repeating-the-last-answer)
  - [Unloading a kept model](#unloading-a-kept-model)
  - [The answer is checked](#the-answer-is-checked)
  - [Acting on what it found](#acting-on-what-it-found)
  - [The prompt library](#the-prompt-library)
  - [The guides are fetched, not bundled](#the-guides-are-fetched-not-bundled)
  - [The model list](#the-model-list)
  - [Models you already pulled for Ollama](#models-you-already-pulled-for-ollama)
- [Where the weights go](#where-the-weights-go)
- [Using a model you already have](#using-a-model-you-already-have)
  - [Smaller repackings](#smaller-repackings)
  - [If the node says a package is missing](#if-the-node-says-a-package-is-missing)
- [Smallest download without any extra install](#smallest-download-without-any-extra-install)
- [GGUF — smaller still, and nothing to install](#gguf--smaller-still-and-nothing-to-install)
  - [Progress on the node](#progress-on-the-node)
  - [Environment variables](#environment-variables)
  - [Languages](#languages)
- [Notes](#notes)
- [Credits](#credits)
- [Licence](#licence)

## What you need before installing

The LoRA route is a 27-billion-parameter language model, not a small helper.
There is no way around the following, because the adapter is bound to one
specific base model. **The writer nodes have none of these requirements** — see
their table below.

| Resource | Requirement |
|---|---|
| Disk | **~52 GB** for `Qwen/Qwen3.6-27B` + **~3.5 GB** for the adapter — or **~10–16 GB** total on the GGUF route |
| VRAM (`nf4`, default) | **~16 GB** |
| VRAM (`int8`) | ~28 GB |
| VRAM (`bfloat16`) | ~54 GB, spills into system RAM via accelerate |
| VRAM (GGUF) | ~13–19 GB depending on the quant, lower still with fewer offloaded layers |
| Packages | `transformers`, `peft`, `accelerate`, and `bitsandbytes` for `nf4`/`int8`. **Nothing at all on the GGUF route:** `llama-cpp-python` is used when it happens to be installed, and the official llama.cpp binaries are fetched when it is not |

> **The MiniMax-H3 text encoder cannot be reused for this.** It is a different
> model (Qwen3-VL-32B, vocabulary 151936) from the LoRA's base (Qwen3.6-27B,
> vocabulary 248320); it contains none of the linear-attention `in_proj_*`
> modules the adapter targets; it is truncated to the first 50 of 64 layers; and
> it ships without `lm_head` or a final norm, so it cannot generate text at all.
> It only produces hidden states for the DiT.

## Install

Clone into `ComfyUI/custom_nodes/` and install the requirements into the same
Python environment ComfyUI runs on:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI
```

For ComfyUI portable on Windows:

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\MiniMax-H3-Prompt-Rewriter-ComfyUI\requirements.txt
```

Or install it from the Comfy registry through ComfyUI-Manager.

### Example workflows

Twelve workflows ship with the pack and appear in ComfyUI's template browser
(*Workflow → Browse Templates*) under this node pack's name once it is
installed. Each is a card of its own and each runs on its own: nothing is
bypassed on open, and there is no second branch to mute before pressing Run.

| | Template | What it needs |
|---|---|---|
| 1 | **Write a prompt** — one line of an idea in, a full H3 audio-video description out | one 2.6 GB GGUF |
| 2 | **Rewrite a prompt with the 27B LoRA** — the LightX2V adapter this pack is named after | one 15.7 GB GGUF |
| 3 | **Write a prompt from references** — the writer describes your pictures and writes from what it saw | two GGUFs, 6 GB together |
| 4 | **Ready-made prompts** — a thousand finished ones, picked in a browser with their frames | nothing at all |
| 5 | **Prompt to video** — ComfyUI's text-to-video template with the writer in front of it | the MiniMax-H3 weights |
| 6 | **References to video** — the same for Ref2VA, the pictures reaching the generator through the writer, in the order its strip numbers them | the MiniMax-H3 ref2va weights |
| 7 | **LoRA triggers and effects** — the words an adapter answers to, and MiniMax's ten effects, put into a finished prompt | nothing at all |
| 8 | **RefMods to video** — a [RefMod](#minimax-h3-refmod), a look and a voice saved already encoded: the writer sees and hears it, and the generator is given the latent it was saved as | the MiniMax-H3 ref2va weights, and a RefMod in `models/refmods` |
| 9 | **P-H3 text to image** | P-H3 image weights and prompt upsampler |
| 10 | **P-H3 image editing** | P-H3 image weights, upsampler and your picture |
| 11 | **Write through an API** | a configured local or remote API server |
| 12 | **Write with the full Qwen model** | 9B Q5_K_M and its F16 projector, 7.39 GB together |

5, 6 and 8 generate MiniMax-H3 video; 9 and 10 generate P-H3 images. The other
workflows produce or process prompts. 5 and 6 are ComfyUI's own gallery templates
with the generator folded into a single subgraph box, so what is on screen is
the prompt side plus one node — the checkpoints they name are the ones the
stock templates use, and ComfyUI offers to download any that are missing. 8 is
6 with [Reference Encode](#minimax-h3-reference-encode) in that box in place of
the native reference node, and the two VAEs outside it, shared with the RefMod.

Every one carries a **Read me first** note: what it does, what it downloads,
what to set before pressing Run, and where to go next. The image loaders in 3
and 6 and the RefMod picker in 8 ship **empty** on purpose — a file name saved
into a template points at something your machine does not have — so choose
your own first.

The card pictures the browser draws beside each name come from
`python tools/template_cards.py`, which writes them as `<name>.jpg` next to the
workflow, where the browser looks for them.

For a community take, [axiomgraph's workflows](https://github.com/axiomgraph/ComfyUIWorkflow)
pair the Omni rewriter with FL2VA (GPL-3.0, and they use a few extra node
packs — grab them from their repository).

## Nodes

### MiniMax-H3 Prompt Rewriter

The main node. It downloads whatever is missing, loads the model, generates, and
releases the VRAM again.

**Outputs**

| Name | Contents |
|---|---|
| `rewritten_prompt` | The full rewrite, ready to paste into a MiniMax-H3 text input |
| `integrated_multimodal_description` | Just the shot-by-shot visual section |
| `overall_soundscape` | Just the diegetic audio section |
| `non_diegetic_music` | Just the score section |

**Inputs**

- `prompt` — the short prompt to expand.
- `model` — the base model. The list holds every entry from your model list plus
  every Qwen3.6-27B already on disk (prefixed `on disk:`). Anything not present
  is downloaded on first use, resuming if interrupted. The **Model list**
  button edits the list in a window over the graph — see below.
- `resolution` / `duration` — conditions the rewrite is composed for. Keep them
  equal to what you pass to MiniMax-H3, or the shot pacing will not match. The
  list runs `48:9` down to `9:16`; the two ultrawides are the multi-monitor
  case — 32:9 is two 16:9 screens side by side, 48:9 three. They condition how
  the shot is composed, which is all this node decides; what a generator will
  render at that shape is its own question.
- `aspect_ratio` — **the same setting, on a socket**, and it wins while something
  is connected. Every writer and rewriter node in the pack has it, and it takes a
  `STRING` or a `COMBO` link, so the primitive that already sets ComfyUI's
  **Resolution Selector** can drive this from the same wire. It is there
  because the shape of the frame is usually decided elsewhere in the graph and
  spelled differently there: ComfyUI's own **Resolution Selector** calls 16:9
  `16:9 (Widescreen)`, a size node says `3840x1080`, a divider says `1.78`. All
  three are read, and a label around the pair is read through — the number pair
  is what counts. A frame size within 2% of a listed ratio is called by its
  name, so `1376x768` (which is what Resolution Selector produces for 16:9 at
  1 MP) arrives as `16:9` rather than as `43:24`; a ratio that is nothing on the
  list passes through as itself, which is how `2.39:1` or `5:4` gets in.
  Something that is not a ratio at all is refused by name rather than composed
  for. The `resolution` widget itself has no socket, so there is one way in and
  it is the one that reads what other nodes write. **The picker greys out while
  the wire is connected** — dimmed, nothing lit, clicks refused — because a lit
  square would be naming a ratio the run is not going to use. **Unplugging clears
  the field**: an upstream node writes its value into the widget — that is how a
  wire feeds a widget input at all — so the text would otherwise stay behind, and
  it is not inert while it sits there. Anything in `aspect_ratio` outranks the
  picker, including a leftover.
- `quantization` — how to load an *unquantized* checkpoint: `nf4` (default,
  ~16 GB VRAM), `int8` (~28 GB), `bfloat16` / `float16` (~54 GB). Ignored when
  the checkpoint brings its own quantization.
- `greedy` — on by default for deterministic output. Turn it off to sample.
- `seed`
- `keep_model_loaded` — **off by default.** The 27B model is released as soon as
  the rewrite finishes, so the same GPU can run H3 video generation next. Turn it
  on only when iterating on prompts back-to-back, and see
  [Unloading a kept model](#unloading-a-kept-model) for giving the card back
  afterwards.
- `options` — optional; connect a **MiniMax-H3 Rewriter Options** node.

### MiniMax-H3 Prompt Rewriter 8B (sees frames)

The same idea on a much smaller model that is also multimodal. LightX2V's second
adapter is trained on
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), so
where the 27B has to be *told* what a reference frame contains, this one is shown
the frame and writes the alignment line from what it sees. It covers four tasks
rather than one, and fits on a card the 27B cannot go near.

![The 8B rewriter node in ComfyUI, set to T2VA: the options node on the left, the rewriter with its first_frame and last_frame inputs in the middle, and the finished rewrite on the right with numbered shots, an (S1) speaker id and a <d>[English] Hello.</d> dialogue tag](docs/node_rewriter_8b.png)

**Outputs** are the same four as the rewriter above, so the two are
interchangeable downstream.

**Inputs**

- `prompt`, `resolution`, `duration`, `greedy`, `seed` — as above.
- `model` — a Qwen3-VL-8B base, in either shape the adapter is published for.
  A **GGUF** entry is two files from the same conversion, the model and its
  projector. A **safetensors** entry is the official
  [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
  folder, which is what the adapter was trained on and what it is published as.
  Entries prefixed `on disk:` are already in your model folders.
  **Only the 8B fits the adapter**; a Qwen3-VL of another size is refused by name
  and number before anything is downloaded.
- `quantization` — how to load a **safetensors** base: `nf4` needs about 8 GB of
  VRAM, `int8` about 13, `bfloat16` about 20. Ignored for GGUF, which carries its
  own.
- `task` — `T2VA`, `I2VA`, `FL2VA`, `L2VA`. The model's own name for these is
  T2AV, I2AV, FL2AV and L2AV; they are the same four tasks.
- `first_frame` / `last_frame` — optional IMAGE inputs. `I2VA` reads
  `first_frame`, `L2VA` reads `last_frame`, `FL2VA` reads both, `T2VA` reads
  neither. Connect the wrong one and the node says which is missing before
  anything loads — which end of the clip a picture belongs to is part of what
  the model is told.
- `keep_model_loaded` — on a **safetensors** base every task honours it: the
  model is loaded in ComfyUI's own process and stays there. On a **GGUF** base
  only `T2VA` can, because the three tasks with frames run through
  `llama-mtmd-cli`, a fresh process each time that takes the model with it when
  it exits. The node says which it did rather than ignoring the switch.
- `options` — the same options node as everything else. Its `adapter` dropdown
  lists both LoRAs; the first entry picks whichever one matches the base model
  you chose, so it needs no attention.

**What it costs**

| | Download | VRAM |
|---|---|---|
| Q4_K_M base + projector + Q8_0 adapter | 4.7 + 0.7 + 0.7 GB | ~9 GB |
| Q8_0 base + projector + F16 adapter | 8.1 + 0.7 + 1.3 GB | ~13 GB |
| safetensors base + adapter, `nf4` | 17.5 + 2.8 GB | ~8 GB |
| safetensors base + adapter, `bfloat16` | 17.5 + 2.8 GB | ~20 GB |

The GGUF rows pay for the runtime as well, once per machine: `I2VA`, `FL2VA` and
`L2VA` run through `llama-mtmd-cli`, so the first of them fetches the official
llama.cpp build — 34 MB, or 511 MB where `llama_backend` resolves to CUDA — and
an installed `llama-cpp-python` does not cover it. `T2VA` is the one task that
can take the wheel instead.

The GGUF route is much the smaller download and needs nothing installed. The
safetensors route is the shape the adapter was published in, keeps the model
resident for every task rather than only for `T2VA`, and is the one to reach for
if you already have the checkpoint — but it needs `transformers` and `peft`,
which the pack lists as dependencies.

**What to expect of it.** All four tasks produce the trained shape: `T2VA` starts
straight in on the three fields, and the other three open with the alignment
sentence MiniMax-H3 itself reads. The shot markers are the visible difference the
adapter makes — with `use_lora` off the same model still fills the three fields,
because the contract is in the system prompt, but it stops writing `[Shot 2]`
cut markers and the answer comes back about a third as long.

It is an 8B, and it shows in one place: the alignment line's timestamp is
sometimes formatted to three decimals instead of two, and on `FL2VA` the final
picture is occasionally credited to `Shot 1` rather than the last shot. The
27B does not do this. Nothing downstream parses that line, so it costs
correctness nowhere — but it is worth a glance before pasting.

### MiniMax-H3 Prompt Rewriter Omni (sees and hears)

LightX2V's third adapter, and the first that listens. It is trained on
[Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — the same model
this pack's captioners already use — so a reference reaches it as the asset
itself: the picture, the clip, or the sound. It is also the only one of the three
that covers **Ref2AV**, the full-reference task, which answers with six fields
instead of three.

![The Omni rewriter node set to REF2AV, fed by a Load Video and a Load Image node: four reference sockets down the left with a checkbox on each row, nine outputs on the right from rewritten_prompt down to references, and between them a strip of three coloured squares - a blue "pic 1" over ref_0, a purple "aud 1" over ref_1 and a blue "pic 2" over ref_2 - above the line "drag to reorder - that order numbers the labels - click a square to switch it off". Below that the five tasks with REF2AV lit, eight aspect-ratio rectangles drawn to proportion from 48:9 to 9:16 with 16:9 chosen, a duration of 10.0, a two-sentence prompt, a Qwen2.5-Omni-7B GGUF Q8_0 in the model dropdown, and five buttons at the bottom: Model list, Unload model, Save the last prompt, Edit the last prompt and Prompt library. On the right the finished rewrite fills all six Ref2AV fields, naming Subject 1, Subject 2, Picture 1, Picture 2 and Audio 1 across two shots](docs/node_rewriter_omni.png)

**Outputs.** Seven, and which of them fill depends on the task. The four frame
tasks return the same three as the other two rewriters, so they are
interchangeable downstream. `Ref2AV` returns six: `subject_definitions`,
`summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`
and `non_diegetic_music` — the same set the
[Ref2VA writer](#minimax-h3-prompt-writer-ref2va) produces, and the same meanings.
`rewritten_prompt` always carries the whole answer. After them comes
`references`, which is not text: the switched-on references in strip order, for
[Reference Slots](#minimax-h3-reference-slots) to hand on to
`MiniMaxH3ReferenceToVideo` numbered the way the prompt numbers them.

**Inputs**

- `prompt`, `resolution`, `greedy`, `seed` — as above.
- `references` — one growing socket that takes an IMAGE, a VIDEO or an AUDIO.
  There is no wrong socket to plug into: what a reference is called follows from
  what it is. Pictures are numbered among pictures and sounds among sounds, so
  connecting a sound between two pictures does not renumber them.
- `task` — `T2AV`, `I2AV`, `L2AV`, `FL2AV`, `REF2AV`. Only `REF2AV` takes clips
  and sound; the other four are written from pictures alone, and connecting a
  sound to one of them is refused by name before anything loads.
- `duration` — **the node snaps it.** MiniMax-H3 generates on a 17n+5 frame grid
  at 24 fps, so most lengths do not exist: ask for 10 seconds and it is 243
  frames, 10.13 s, and *that* is the number written into the turn and quoted back
  in the alignment line. The widget is what you meant; the line and the video
  agree because of the snapping.
- `model` — a Qwen2.5-Omni-7B base, as a **GGUF** pair (the model and its
  projector) or as the official **safetensors** folder. Entries prefixed
  `on disk:` are already in your model folders. Two kinds of near-miss are marked
  rather than hidden: a Qwen2.5-Omni-**3B** is `(wrong size for the adapter)`,
  and a **Qwen2.5-VL-7B** — which is the same architecture string, the same 28
  blocks and the same width, so the adapter *would* attach — is
  `(vision only, not an Omni build)`, because its projector has no audio encoder
  and the rewrite would be about sound that was never heard. The **Open model
  list** button edits the `models_omni` section — see below.
- `quantization` — how to load a **safetensors** base: `nf4` about 9 GB of VRAM,
  `int8` about 12, `bfloat16` about 20. Ignored for GGUF. **Pick the largest your
  card holds** — see below.
- `max_frames` — how many frames to take from a clip, spread evenly. Each frame
  is its own picture to the model.
- `reference_layout` — the strip's state as JSON. It is a widget so the
  arrangement travels with the workflow and through the API; the interface draws
  it as squares instead.
- `keep_model_loaded` — honoured on a **safetensors** base. On a **GGUF** base a
  task with references runs through `llama-mtmd-cli`, a fresh process that takes
  the model with it when it exits, and the node says so rather than ignoring the
  switch.
- `options` — the same options node as everything else.

**The strip is the ordering.** Every connected reference appears as a coloured
square — blue for a picture, green for a clip, purple for a sound — showing what
it will be called and which socket it came from. Drag to reorder, and that order
is what numbers the labels: dragging the second picture to the front is what
makes it `<Picture 1>`. Click a square to switch it off without unplugging it.
The task strip greys out a task the connected references cannot serve and says
why on hover, so `FL2AV` with one picture is visibly unavailable rather than a
failure two minutes later.

There is deliberately **no relabelling here**, unlike the Universal Writer's
strip. There, a picture can be told to stand for a subject or a clip, because the
guide-driven turn names its references by hand. Here the socket settles it — and
a *subject* is something this adapter **produces** in `subject_definitions`, not
something the request supplies.

**A clip's label band switches its own sound on.** On `Ref2AV` the band turns
purple and reads `vid+a1`: the clip's sound is now `<Audio 1>`, a reference of
its own, and the adapter hears it — handed over as a separate sound right before
the clip, which is the only way its upstream inference ever passes a clip's
sound. Clip sounds take the first `<Audio N>` numbers and standalone sounds
follow, because that is how `MiniMaxH3ReferenceToVideo` numbers them, so a sound
square ahead of a sounding clip in the strip is presented after it. On the other
tasks the band stays a paler purple with no number: set, and idle until
`Ref2AV`. A clip with no sound track is said on the node and left unnumbered.

**What it costs**

| | Download | VRAM |
|---|---|---|
| Q4_K_M base + projector + Q8_0 adapter | 4.4 + 1.4 + 0.34 GB | ~9 GB |
| Q8_0 base + projector + F16 adapter | 8.1 + 1.4 + 0.65 GB | ~13 GB |
| safetensors base + adapter, `nf4` | 22.4 + 1.3 GB | ~9 GB |
| safetensors base + adapter, `bfloat16` | 22.4 + 1.3 GB | ~20 GB |

A GGUF base pays for the llama.cpp runtime as well, once: every task but `T2VA`
carries references and therefore runs through `llama-mtmd-cli` — 34 MB, or
511 MB where `llama_backend` resolves to CUDA, and an installed
`llama-cpp-python` makes no difference to it.

The GGUF adapter is converted from LightX2V's own safetensors with llama.cpp's
`convert_lora_to_gguf.py` and published at
[pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF).
The `Q8_0` build is half the size of the `F16` and behaves the same.

**Quantization buys VRAM, not speed.** Which is the opposite of the intuition — a
smaller model should move fewer bytes and go faster. Measured on this adapter, on
one card, same prompt, same two pictures, same 256 tokens:

| | VRAM | Generation |
|---|---|---|
| `bfloat16` | 19.4 GB | **18.6 tok/s** |
| `nf4` | 8.6 GB | 15.1 tok/s |
| `int8` | 12.0 GB | 6.0 tok/s |

`int8` is the worst of the three on both counts: slower than `nf4` *and* larger
than it. That is not a quirk of this adapter — bitsandbytes' `load_in_8bit` is
LLM.int8(), which splits every matmul into an fp16 outlier part and an int8 part
and recombines them, casting the activations each time. It is a scheme for
fitting a model that would not fit, and it costs what it costs. `nf4` is the
better small option and dequantizes on every matmul too, which is why it does not
beat `bfloat16` either. So quantize only down to what the card actually holds:
with 24 GB or more, `bfloat16` is both the fastest and the most faithful.

The same applies to any safetensors base in this pack, since the mechanism is
bitsandbytes' rather than the model's. **And none of it applies to the GGUF
route**, where llama.cpp has real quantized kernels: the same FL2AV that takes
26 seconds through `bfloat16` safetensors takes 10 through `Q4_K_M`.

**Pictures are scaled before the model sees them.** LightX2V's own inference
script caps a picture at 301056 pixels — 384 tokens — and a frame from a clip at
100352, and this node does the same. It is not only a saving: a 1616×1616 picture
is 3249 tokens, two of them overflow an 8k context before a word of the prompt is
counted, and the model is being shown a shape it never saw in training. The
context is then sized to what the turn actually costs, so a `Ref2AV` with eight
references widens it instead of failing.

**The safetensors route shows pictures only.** ComfyUI's in-process Transformers
path has no way to hand the model a sound, so a clip or a sound on that base is
refused rather than silently dropped from the turn. Pick a GGUF base to use them.

### MiniMax-H3 Qwen Rewriter

Runs the full LightX2V **Qwen3.5 9B** and **Qwen3.8 27B** SFT checkpoints, or their
GGUF conversions with matching vision projectors. The FL2VA checkpoint serves
T2VA, I2VA, L2VA and FL2VA; the Ref2VA checkpoint serves Ref2VA. Pick original
SFT weights or GGUF through **Model list**, with generation and media settings
in the separate **Qwen Rewriter Options** node.

The node has the shared top reference strip, input checkboxes, **+ instr**,
aspect/duration controls, bypass/repeat badges, prompt library and `references`
output. It reads images and sampled video frames directly; audio uses explicit
descriptions because these checkpoints have no audio encoder.

| Checkpoint | Original SFT | GGUF Q5_K_M / Q4_K_M |
|---|---|---|
| Qwen3.5-9B FL2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF) |
| Qwen3.5-9B Ref2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA-GGUF) |
| Qwen3.8-27B FL2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA-GGUF) |
| Qwen3.8-27B Ref2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-GGUF) |

[Setup, reference roles, outputs and conversion](docs/qwen_rewriter.md).

![Qwen Rewriter Options, the full Qwen Rewriter and its generated H3 prompt](docs/qwen_rewriter_workflow.png)

### MiniMax-H3 Universal Rewriter

All three prompt-rewriter LoRAs in one node, with a tab at the top choosing which
one runs. [Prompt Rewriter](#minimax-h3-prompt-rewriter),
[Prompt Rewriter 8B](#minimax-h3-prompt-rewriter-8b-sees-frames) and
[Prompt Rewriter Omni](#minimax-h3-prompt-rewriter-omni-sees-and-hears) are
unchanged and still there — nothing you have already built stops working.

The three adapters are not three settings of one thing. The 27B is text:
Qwen3.6-27B, one task, and a reference frame reaches it only as a sentence
somebody wrote. The 8B is multimodal: Qwen3-VL-8B, four tasks, and the picture
itself. The Omni is multimodal and hears as well: Qwen2.5-Omni-7B, the same four
tasks and a fifth of its own. Different base, different size, different
download.

Which is exactly why choosing between them by hand is tedious. The prompt is the
same prompt, the aspect ratio is the same aspect ratio, the duration is the same
duration — so trying another adapter meant retyping all of it into a second node
and then keeping them in step.

![The Universal Rewriter on its Omni tab, running I2VA: four reference rows down the left — first_frame, last_frame, reference_video and reference_audio, each with a checkbox, and first_frame fed by a Load Image node — and nine outputs on the right, from rewritten_prompt down to references. Three tabs across the top with "Omni LoRA / sees, hears" lit, a task strip with I2VA lit, eight aspect-ratio rectangles drawn to proportion from 48:9 to 9:16 with 16:9 chosen, a duration slider at 10, the prompt "A woman says hello to a alien.", and below them model_omni on a Qwen2.5-Omni-7B GGUF Q8_0 with quantization_omni set to bfloat16, then aspect_ratio, repeat_last and five buttons: Model list, Unload model, Save the last prompt, Edit the last prompt and Prompt library. On the right the answer: a line saying Picture 1 is fully referenced from the first shot, then the three fields I2VA fills](docs/node_universal_rewriter.png)

*One node, three tabs. Nothing above the model rows belongs to a tab — the
task, the ratio, the duration and the prompt are one set of values, whichever
adapter is lit. Below them each tab holds its own model and quantization, still
set to whatever you last chose, including across a save and load. The task strip
is the other difference: the 27B tab lights `T2VA` alone, the 8B tab adds the
three frame tasks, and the Omni tab adds `Ref2VA` on top of them, which no other
tab can reach.*

**So the tab carries what differs, and nothing else:**

| Belongs to the tab | Shared between them |
|---|---|
| `model_27b` / `model_8b` / `model_omni` | `prompt`, `task`, `resolution`, `duration` |
| `quantization_27b` / `quantization_8b` / `quantization_omni` | `greedy`, `seed`, `keep_model_loaded`, `bypass`, `options`, both frames, the clip and the sound |

The widget the other tab uses is hidden rather than reset, so it is still set to
whatever you last chose when you switch back — including across a save and load.

> **No captioner on the 27B tab, deliberately.** Folding a description of a
> frame into the prompt does reach that adapter, and it is not wasted — the props,
> surfaces and light in it turn up in the shots, in the trained shape, with
> nothing leaking into the answer. But the picture is absorbed into the scene
> rather than pinned to 0.00 seconds, which is exactly what the
> [LoRA's own page](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA)
> says: T2VA is finished there and FL2VA is not. A widget for it on this node
> would have looked like the frame task the 27B cannot do. When you want it,
> [Reference Caption](#minimax-h3-reference-caption) writes the description and
> it goes in `prompt` like any other text; when the picture has to *be* a frame,
> the 8B tab is the one that was trained for it.

**The task switch is shared, and the 27B tab does not touch it.** On that tab it
shows `T2VA` lit with the three frame tasks greyed out, because that is the
honest picture of a text-only model, and clicking does nothing at all — the value
the other two tabs had is still there when you switch back. On the 8B and Omni
tabs a frame task greys out until the frame it is written from is actually
connected and switched on, the same way `Ref2VA` does on the Universal Writer.

**`Ref2VA` is on the Omni tab, with a clip socket and a sound socket to feed it.**
`reference_video` and `reference_audio` are read by that task and by nothing else
— the 27B and 8B adapters have no ear, and the four frame tasks take pictures
alone, so connecting a sound to `FL2VA` is refused by name rather than quietly
dropped. On `Ref2VA` everything connected becomes a reference the target video
reuses, in socket order: `first_frame` is `<Picture 1>`, `last_frame` is
`<Picture 2>`, then `<Video 1>`, then `<Audio 1>`. The answer comes back in six
fields instead of three; the four extra outputs sit **after** the three every
task fills, so nothing already wired to this node moves. Last of all is
`references`: the references the current tab actually read, for
[Reference Slots](#minimax-h3-reference-slots) to hand on to
`MiniMaxH3ReferenceToVideo`.

> **Four references, not twelve.** Order is the whole labelling rule, and with
> four sockets the order is the order of the sockets. Past that, arranging them
> by hand is the thing you actually want — which is what the draggable strip on
> [Prompt Rewriter Omni](#minimax-h3-prompt-rewriter-omni-sees-and-hears) is for,
> along with `max_frames` and any number of pictures. This tab is for the other
> thing: trying the same prompt on a different adapter without leaving the node.

**Four reference inputs, with a checkbox on each row.** `first_frame` and
`last_frame` are the same two the 8B node has; `reference_video` and
`reference_audio` are the two `Ref2VA` adds. A switched-off row counts as
unplugged — which is how a reference gets parked without dragging the wire off.
Run the 27B tab with any of them connected and the node says on itself that it is
not reading them, and where to put them instead, rather than leaving you to
wonder.

**`reference_video` has a second box, a speaker.** Tick it and the clip's own
sound becomes a reference of its own on the Omni tab's `Ref2VA`: `<Audio 1>`,
with `reference_audio` moving to `<Audio 2>` — the order
`MiniMaxH3ReferenceToVideo` gives them — and `references` carries it on to
[Reference Slots](#minimax-h3-reference-slots).

**`duration` is a slider**, reaching 30 seconds until you move it — right-click
the node, `duration`. See [The duration widget](#the-duration-widget). The three
adapters were trained on clips of a few seconds, so a number far past that is a
worse prompt rather than a longer video — MiniMax-H3 gets the length from its own
settings, not from this line.

**There is no "Open guide folder" button**, because none of the three adapters
reads a guide: the format is in the system prompt they were trained with. The
**Model list** button is there, and it opens all three of this node's lists as
tabs in one window — `models` for the 27B tab, `models_8b` for the 8B one,
`models_omni` for the Omni one.

> **If the interface script does not load**, the tab strip, the task switch and
> the ratio picker fall back to the plain dropdowns they are built on, and every
> widget is visible at once instead of a tab's worth at a time. The node runs
> exactly the same either way. Those three controls are HTML and work in both of
> ComfyUI's renderers; the checkboxes on the two frame rows are drawn on the
> canvas, which *Modern Node Design (Nodes 2.0)* does not run, so there a frame
> is left out by unplugging it.

> **This node needs a recent ComfyUI**, being written against the v3 node API,
> exactly as the Universal Writer is. On an older install it goes missing and the
> rest of the pack registers as before.

### MiniMax-H3 Rewriter Options

Everything you rarely touch, kept off the main node. Leave it unconnected and the
rewriter uses the decoding parameters the adapter was published with.

![Rewriter Options with Default guide and the System prompts button below the generation settings](docs/node_options.png)

| Input | Default | Purpose |
|---|---|---|
| `max_new_tokens` | 2048 | Generation cap |
| `temperature` / `top_p` / `top_k` | 0.7 / 0.8 / 20 | Sampling, used only when `greedy` is off |
| `repetition_penalty` | 1.05 | |
| `attn_implementation` | `sdpa` | `eager` or `flash_attention_2` if you have it |
| `adapter` | the LightX2V repo | Which build of the LoRA to apply — see below |
| `use_lora` | on | Turn off for the plain base-model baseline |
| `merge_lora` | `auto` | Fold the adapter into the weights at load — twice the tokens a second, see below |
| `auto_download` | on | Turn off to fail loudly instead of fetching 52 GB |
| `downloader` | `built-in` | Which code fetches a Hugging Face repository — see below |
| `device` | `auto` | Which GPU runs the language model — see below |
| `prompt_file` | `global` | Which set of saved prompts the nodes wired to this one work in — see [the prompt library](#the-prompt-library) |
| `self_check` | `warnings and notes` | How much of [the self-check](#the-answer-is-checked) is said out loud: everything, warnings only, or nothing |
| `fix_once` | `false` | Let the nodes act on what the check found — [one re-run, never a loop](#acting-on-what-it-found) |
| `captioner_thinking` | `off` | GGUF reference captions: `off` disables thinking, `auto` keeps the model template default, `on` enables it when supported. |

For GGUF captions, supported flags are read from each binary's `--help` once
per executable revision. Explicit `off`/`on` uses `--reasoning` and, when
available, `--jinja`. If `llama-mtmd-cli` cannot set thinking, a compatible
`llama-server` beside it is used even for a single reference. The server loads
once per strip and exits when the run completes or is cancelled.
`MINIMAX_H3_MTMD_SERVER=never` keeps the CLI route. If neither available route
can apply the setting, the run uses the template default and logs a warning;
it never sends an unsupported flag. The 1024-token caption cap remains in
place. An empty GGUF caption raises an error. This option does not change the
writer or CLIP caption settings.

The same options node feeds the writer nodes and the captioner; `adapter` and
`use_lora` simply do not apply there.

#### `merge_lora` — the adapter as weights rather than as a second matmul

PEFT keeps an attached LoRA beside the base model and computes it on every
token, on top of the base weights. Folding it in once at load does the same
arithmetic ahead of time, and the difference is not small. Measured here on
Qwen3-VL-8B with the 8B adapter, 128 tokens on a 5090:

| | adapter attached | folded in | merge costs |
|---|---|---|---|
| `bfloat16` | 8.98 s — 14.3 tokens/s | **5.13 s — 25.0 tokens/s** | 0.07 s |
| `nf4` | 9.43 s — 13.6 tokens/s | **5.09 s — 25.1 tokens/s** | 4.6 s |

`auto`, the default, takes the free half: it folds the adapter into an
unquantized base, where the merge costs a tenth of a second, and leaves it
attached on a bitsandbytes `nf4`/`int8` one, where merging means dequantizing
every layer and quantizing it back. `on` does it there too, if you would rather
pay 4.6 s at load and have the tokens; `off` is the old behaviour. Nothing here
touches the GGUF route, where llama.cpp applies the adapter its own way.

**A folded run is not word-for-word the attached one.** Merging reassociates the
arithmetic — `W + BA` computed once is not bit-for-bit `Wx + B(Ax)` computed per
token — so a token here and there falls differently and the rest of the sentence
follows it. Measured on the pair above at the same seed: two differences in 128
tokens on `bfloat16`, more on `nf4`, where the requantization adds error of its
own and PEFT prints a warning saying so. Neither answer is the better one, but
they are not the same answer, and a workflow you have tuned to the token is a
reason to leave this on `off`.

#### `adapter` — a dropdown, and what is in it

The list holds, in order:

- **`lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA`**, the default and the entry to
  leave alone. It means *whichever build the model list names for the base model
  you picked* — the PEFT adapter for a `transformers` base, the catalog's GGUF
  for a GGUF one, and the 8B adapter on the 8B node.
- **Every published precision of both LoRAs.** F16 and Q8_0 for each; the Q8_0 is
  half the download and rewrites the same. Picking one that is not on disk
  downloads it from the repository it belongs to.
- **Every `.gguf` LoRA already in your ComfyUI model folders**, prefixed
  `on disk:` with its architecture in the label, so `qwen35` and `qwen3vl` are
  told apart at a glance. This is how a LoRA you converted yourself gets in:
  drop the file in `models/LLM` and it appears.

Choosing a quantisation used to mean editing `models.json` by hand, which is a
power-user path rather than an interface. Saved workflows are unaffected: a combo
serialises as the string it displays, and the old default is still the first
entry, character for character.

Two things are still true of this field, because it is the one setting a
downloaded workflow gets any say in:

- **A network path is refused.** `\\host\share\...` and `//host/share/...` are
  rejected before anything reads them, because merely looking at one is an
  authentication attempt against whatever host is named. A share of your own is
  reachable by drive letter or mount point as usual, and a path in `models.json`
  is not restricted at all — that file is yours and does not travel.
- **The adapter that was applied is logged**, every run. Ask for one that is not
  the configured adapter and the console says so at warning level. A swapped
  LoRA is otherwise invisible: the node still runs and still fills every field,
  it just writes something else.

#### Model code is never run

A Transformers checkpoint can carry its own modelling code, named by `auto_map`
in its `config.json`, and loading such a model imports and runs that Python with
your user's rights. The node refuses such a model and says why. Nothing in the
shipped list is one: every `transformers` entry is a Qwen3.6-27B variant,
Qwen3-VL-8B or Qwen2.5-Omni-7B, architectures Transformers supports natively,
and the GGUF entries never reach Transformers at all.

Up to 0.30.0 a `trust_remote_code` switch here let such a model run. It was off
by default, but it was a widget, so a workflow could turn it on, and the model
list can be extended through ComfyUI's API, which asks for no login. On a
ComfyUI reachable from the network that was enough to make it download a model
and run the code inside. 0.31.0 removed the switch. A workflow saved with it
still loads: the Options node puts the other values back in place.

#### `device` — and why it changes `keep_model_loaded`

Every node here has carried the same warning: turn `keep_model_loaded` off,
because the card is needed for video generation the moment the rewrite finishes.
That warning exists only because both models want the same device. **With two
cards they need not.**

Values are `auto`, `cpu`, and one `cuda:N` per GPU ComfyUI can see. One spelling,
three backends:

| | `auto` | `cuda:1` | `cpu` |
|---|---|---|---|
| llama.cpp binaries | unchanged | `--device CUDA1` | `--device none`, 0 layers |
| llama-cpp-python | unchanged | `main_gpu=1`, split mode `NONE` | `n_gpu_layers=0` |
| Transformers | `device_map="auto"` | `device_map={"": "cuda:1"}` | `{"": "cpu"}` |

The important part is not the placement. **On another card, ComfyUI's own models
are no longer evicted first.** Every backend here unloads them unconditionally
today, which is right when both want the same VRAM and pure waste when they do
not — it costs a full reload of the diffusion model after every rewrite. Pick a
second card and the rewriter loads beside it, `keep_model_loaded` becomes worth
switching on, and nothing has to move.

Two details worth knowing. A device the machine does not have is **refused, not
quietly demoted** — a workflow carried over from a two-card machine says so,
instead of running on the wrong card and evicting a batch mid-flight. And the
numbering is ComfyUI's own: started with `--cuda-device 1`, ComfyUI sees exactly
one device and it is `cuda:0`, for the subprocesses too.

The values are deliberately plain rather than `cuda:1 · RTX 4090`: a label with
the card's name reads better and breaks every saved workflow the day the card is
replaced. The tooltip names what is in each slot.

#### `downloader` — the built-in transfer, or `huggingface_hub`

The default moves the bytes itself: one ranged HTTP connection at a time,
resumable, needing nothing installed. That is the right floor for something
nobody opted into, and on a fast link it is fine.

It is not fine on a 30 GB checkpoint over a connection that gives one stream 8
MB/s. The Comfy-Org repositories are **Xet-backed**, and `hf_xet` fetches the
chunks of a *single* file over many connections at once — which is the whole of
the difference behind "the CLI took five minutes and the node took an hour".
Setting `downloader` to `huggingface_hub` takes that route instead:

```
python -m pip install huggingface_hub hf_xet
```

into ComfyUI's own Python, then restart. Neither package is a dependency of this
pack, and **nothing fails if they are missing**: the node logs the install line,
notes it in the caption, and downloads with the built-in transfer. That matters
because the setting travels inside the workflow, and a graph authored on a
machine that has them should still run on one that does not.

**It covers Hugging Face repositories only** — base models, GGUF writers, LoRA
adapters, a model and its projector. The llama.cpp binaries come from GitHub
releases and the two writing guides are one small request each; neither goes
through this setting.

**Half-finished files are not interchangeable.** The built-in transfer parks
bytes in `<name>.part` beside the file; `huggingface_hub` parks them under
`.cache/huggingface` inside the destination folder. Switching in the middle of a
download restarts that one file and leaves the other's leftovers to delete by
hand. A file that is already complete is never fetched twice either way — this
pack decides what is missing before either backend is called, which also avoids
`huggingface_hub` re-hashing a finished 30 GB file to adopt it.

One thing worth knowing about disk: `hf_xet` keeps a chunk cache at
`~/.cache/huggingface/xet` (`HF_XET_CACHE`), which on Windows is on `C:` whatever
drive the models live on. The free-space check this pack runs before a download
only looks at the destination.

**Behind an antivirus that inspects HTTPS.** Antiviruses that scan encrypted
connections, and company proxies like them, re-sign huggingface.co with a root
certificate of their own and install that root in the system's certificate store,
which is why the browser still opens the site. The built-in transfer trusts that
store as well as the usual certifi bundle, so it downloads wherever the browser
does. `huggingface_hub` and `hf_xet` bring their own TLS and do not look there:
behind such a filter they stop at `certificate verify failed`, and the node says
so — switch back to the built-in transfer, or add huggingface.co to the filter's
exclusions.

### MiniMax-H3 Prompt Writer (T2VA/I2VA/FL2VA/L2VA)

The same three output fields, without the LoRA and without the 27B. MiniMax's own
[prompt-writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
goes into the system prompt and any instruction-following GGUF writes to it. A
5.2 GB Qwen3.5-9B fills all three fields, with correct camera vocabulary, speaker
IDs and `<d>[English] …</d>` dialogue, in about 20 seconds.

![The T2VA writer node between its options node and a text viewer: a one-line prompt, a model served by Ollama, and on the right the shot-by-shot description with its three numbered shots, the soundscape paragraph, and non_diegetic_music describing a sparse piano and a low cello](docs/node_t2va.png)

*T2VA, 10 seconds at 16:9, on a local Qwen3.6-27B — 11.2 s. The prompt went in
in Russian and the description came back in H3's English format, with the cut at
00:03.000 that the prompt asked for and `non_diegetic_music: N/A` because it
asked for no music.*

The trade is worth stating plainly. The LoRA **is** the format: a 27B was trained
until H3 output came out of it with a seven-line system prompt. Here the format is
carried by ~4 000 tokens of instructions the model has to follow, so expect the
LoRA's prose to be denser and its formatting more reliable. Expect this to run at
all on hardware the LoRA cannot touch.

Outputs are identical to the rewriter node's — same names, same order — so the two
are interchangeable in a saved workflow.

Beyond the rewriter's inputs:

- `task` — **T2VA**, **I2VA**, **FL2VA** or **L2VA**. Everything but T2VA also
  emits the alignment instruction line H3 requires as the very first line, with
  the duration already substituted to two decimals; only the final shot number is
  left to the model.
- `reference_material` — **this node reads text, not pixels.** For I2VA, FL2VA and
  L2VA, describe what the reference frames show, by hand or from the
  [Reference Caption](#minimax-h3-reference-caption) node upstream. Without it the
  model invents a first frame that has nothing to do with your image.

`model` offers the `writers` section of the model list plus every GGUF already in
your ComfyUI model folders, whatever its architecture. Nothing here has to be
Qwen3.6-27B, and nothing has to be installed: without `llama-cpp-python` the node
runs the official llama.cpp binaries, exactly as the GGUF route below does.

| Suggested writer | Download | VRAM with the guide in context |
|---|---|---|
| Qwen3.5-4B `Q4_K_M` | 2.6 GB | ~5 GB — start here on an 8 GB card |
| Qwen3.5-9B `Q4_K_M` | 5.3 GB | ~8 GB — best writing per gigabyte |
| Qwen3.5-9B Uncensored (HauhauCS) `Q4_K_M` | 5.2 GB | ~8 GB — does not decline scenes the stock model refuses |
| Gemma 3 12B Instruct `Q4_K_M` | 6.8 GB | ~10 GB — non-Qwen alternative |
| Mistral Small 3.2 24B Instruct `Q4_K_M` | 13.4 GB | ~17 GB — for 16 GB cards and up |

`n_ctx` is raised automatically: the base guide needs about 9 200 tokens of
context and the full-reference guide about 12 300, against the 8 192 default that
suits the LoRA's short system prompt. Letting llama.cpp truncate instead would
drop the *front* of the prompt — the guide and the output contract — and the
answer would come back in some other format with nothing to say why.

If a field comes back missing, the node still returns everything it got and says
which fields are absent, on the node. Lower the temperature or move up a size.

#### `system_prompt` — pointing the writer at another model

The guide is the whole of what makes this an H3 writer: it goes into the system
prompt and nothing else in the node knows the format. So replacing that text is
what turns these nodes into a writer for something else — LTX, Krea, Wan, or a
house style of your own. Text in `system_prompt` replaces the assembled guide.

The guide is then not even fetched, which matters on a machine kept offline:
there is no 24 KB document downloaded to be ignored.

The shortest way in is
[Guide Prompt (any LLM)](#minimax-h3-guide-prompt-any-llm), which hands you the
stock system prompt on an output. Take it, edit it, connect it back.

**Saved system prompts.** Open **System prompts...** on Rewriter Options,
API Writer Options or Qwen Rewriter Options. Create a draft, import a UTF-8
`.md` or `.txt`, or choose a task and click **Copy default guide** to edit a
copy. **Save and select** stores a Markdown file in
`<ComfyUI user>/minimax_h3_rewriter/system_prompts` and selects it in
`system_prompt_file`. If a workflow moves to a different server, copy the
file to that server's library too.

![System prompts editor with import, a default guide task selector, and Save and select](docs/system_prompts_editor.png)

Text in `system_prompt` takes precedence. When the field is empty, the writer
uses the selected file. **Default guide** uses the built-in guide for the task.

Two things do not move. The task message is never replaced — it carries the
prompt, the aspect ratio and the duration, which any guide needs. And the answer
is still split into the H3 sections, so a guide that replies with one paragraph
fills `rewritten_prompt` and leaves the section outputs empty. That is worth
knowing rather than worth avoiding.

### MiniMax-H3 Prompt Writer (Ref2VA)

Full-reference mode, from MiniMax's
[full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
Six sections instead of three, each its own output:

| Output | Contents |
|---|---|
| `subject_definitions` | what each `<Subject N>` / `<Picture N>` / `<Video N>` / `<Audio N>` label denotes |
| `summary` | the `[task type]` prefix and the reference relationships in one paragraph |
| `retention_analysis` | per label: `fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference`, `fully_copy`, … |
| `detailed_description` | the body, shot by shot, with labels cited where their roles apply |
| `overall_soundscape` | ambience and physical sound |
| `non_diegetic_music` | audience-only score |

![The Ref2VA writer node beside a text viewer showing all six sections: subject_definitions binding Subject 1 and Picture 1, the summary with its reference-generation task type, the retention analysis marking both labels fully_preserved, the shot-by-shot body, the soundscape and non_diegetic_music](docs/node_ref2va.png)

*The same clip as above with one `Picture 1:` line added to `reference_assets` —
17.4 s. The model picked the task type, bound `<Subject 1>` to the cat and
`<Picture 1>` to the drawing's style, and wrote the retention analysis itself.*

`reference_assets` is **required** and is refused when empty: full-reference mode
describes how a target video reuses assets, so with no assets it is simply T2VA
under another name. One per line —

```text
Picture 1: young woman, long dark hair, blue cardigan, thin silver necklace
Picture 2: corner cafe interior, brick wall, brass lamps, rain on the window
Audio 1: voice-timbre reference for the woman — low, unhurried, slight rasp
```

— and the model binds the labels, picks the task type, and writes the retention
analysis from them. Or let
[MiniMax-H3 Reference Caption](#minimax-h3-reference-caption) write those lines
for you from the assets themselves.

### MiniMax-H3 Universal Writer

One node for a whole shot: the references, the order they are in, and the
rewrite. It does what [Multi Reference Caption](#minimax-h3-multi-reference-caption)
and both writer nodes do, in a single box, for all five tasks. Those three stay
exactly as they are — nothing you have already built stops working.

For H3 writing from text, images, video frames, audio or RefMods through a running llama-server or a remote API, use the
additional [API Universal Writer](docs/api_writer.md). Configure connections and models
in **API providers**, and generation settings in **API Writer Options**. Keys are
encrypted on the Windows or Linux server; workflows store connection/model IDs.
Start with **11 - Write through an API** in the workflow templates.

![API Universal Writer with task, aspect ratio, provider and model selection, reference inputs and prompt outputs](docs/api_universal_writer_node.png)

![API Writer Options with generation limits, response format, video sampling, audio rate and repair settings](docs/api_options_node.png)

The reason to fold them together is **order**. `Picture 1` and `Picture 2` are
not interchangeable in FL2VA: one opens the video and the other closes it. Until
now the only thing deciding which was which was the order the caption node
happened to write them in, which came in turn from which slot each was plugged
into. Real, load-bearing, and nowhere on screen.

![The Universal Writer node, with a Load Video and a Load Image node feeding it: four reference rows each with a checkbox, three of them connected, then a strip of three coloured squares reading vid 1 over ref_0, aud 1 over ref_1 and pic 1 over ref_2, each with an "+ instr" button and a line of help under them, a task switch with Ref2VA lit, eight aspect-ratio rectangles drawn to proportion from 48:9 to 9:16 with 16:9 chosen, a duration slider at 15.1, the prompt, the captioner and writer dropdowns, and six buttons at the bottom: Model list, Open guide folder, Unload model, Save the last prompt, Edit the last prompt and Prompt library. On the right the six-field answer, written around <Subject 1>, <Picture 1>, <Video 1> and <Audio 1>](docs/node_universal_writer.png)

*Four references on one growing socket — an image used as a subject, an image
used as a frame, a clip and a sound — and on the right an answer written around
exactly those labels: `<Subject 1>`, `<Picture 1>`, `<Video 1>`, `<Audio 1>`. The
squares are in slot order here because nothing has been dragged; the number is a
position and renumbers the moment anything moves, while the slot name under it is
what stays with a square.*

**One socket takes an image, a clip or a sound**, and more slots appear as you
fill them. There is no wrong socket to plug into here, because which socket you
used no longer decides anything.

**The strip under the inputs decides it.** Every connected reference gets a
square, coloured by what it is for and numbered exactly as the block will number
it. Under the number is the slot it arrived on, and that is the part of a square
that stays with it: the number is a position, so it renumbers the moment anything
moves.

| Square | Label | What it means |
|---|---|---|
| `pic` | `<Picture N>` | an image serving as an actual frame — first, last, key, composition anchor |
| `subj` | `<Subject N>` | reusable visible content — a person, an animal, a place, a costume, a style |
| `vid` | `<Video N>` | a clip, or a batch of frames you want read as one |
| `aud` | `<Audio N>` | a voice timbre, music, ambience, an effect |

- **Drag a square** to move it. The numbering follows immediately.
- **Click its label** to change what an image is for: `pic` → `subj` → `vid` and
  round again. A sound is what it is and does not cycle.
- **Click a clip's label** to count the clip's own sound as a reference too. On
  `Ref2VA` the band turns purple and reads `vid+a1`: the captioner describes that
  sound as `Audio 1` in the block, just before the clip, and `references` tells
  [Reference Slots](#minimax-h3-reference-slots) to hand it to the generator.
  Clip sounds take the first `<Audio N>` numbers and standalone sounds follow,
  which is how `MiniMaxH3ReferenceToVideo` numbers them. The captioner has to
  hear for this — one that cannot is refused by name before anything loads — and
  a clip with no sound track is said on the node and left unnumbered.
- **Click its number** to switch that reference off without unplugging it — the
  checkbox on the slot's own row does the same thing from the other side.
- **Click the band under it** to ask that one reference something other than its
  role's usual question. `+ instr` adds your line to the role's question,
  `= instr` asks it instead; right-click takes it back, hovering reads it out.
  A square carrying one is ringed in amber for `+` and in orange for `=`, so it
  does not go unnoticed.

So one socket still produces the four labels Ref2VA allows, and the distinction
Multi Reference Caption makes structurally — a subject is not a frame — is made
here on the square instead.

**Every square shows what is plugged into it** — here, on Prompt Rewriter Omni
and on Multi Reference Caption. A picture shows its thumbnail, a clip its first
frame with a ▶ in the corner, a sound its waveform with a speaker and the file
name in small type; the number moves to the corner. Hovering plays a clip and
flips through a batch of frames, and the tooltip adds the file, the size, the
frame count and the length. A sound, after a moment's hover, plays its first six
seconds with the wave lighting up as it goes; a press on the square silences it. A square wired to a loader — Load Image, Load Video,
Load Audio or VideoHelperSuite's loaders, through reroutes too — shows the file
straight away, before anything runs. One fed through other nodes shows what
reached the node on its last run, and stays a plain square until then.

![The strip of a Universal Writer set to Ref2VA, with eleven reference rows, ref_9 unticked. Its squares, left to right: pic over ref_0 showing a black cat with a 1 in the corner, ringed in amber with an amber "+ instr" band; pic over ref_1 showing a wooden fence, 2, ringed in orange with an orange "= instr" band; pic over ref_2 showing a man in a car, 3; vid over ref_3 showing a man walking down a street with a play mark, 1; aud over ref_4 and ref_5, each a speaker and a waveform, 1 and 2; vid over ref_6 showing a figure in a dinosaur mask with a play mark, 2; on the second row pic over ref_7 showing a jewelled turtle, 4; aud over ref_8 with a speaker, a truncated file name, a waveform and 3; and aud over ref_9, dimmed, with its name and waveform still drawn and "--" for a number. Under the strip the task switch with Ref2VA lit, the aspect ratios with 16:9 chosen and duration 10.0](docs/node_ref_strip.png)

*Pictures, clips and sounds tell apart at a glance, and so do two sounds: ref_8 and
ref_9 come straight from a loader and show their file names, ref_4 and ref_5 came
through other nodes and show the wave from the last run. ref_9 is switched off —
dimmed, numbered `--`, still recognisable. The two squares with an instruction
are the ones ringed.*

**The task switch and the aspect-ratio picker are the same idea**: the choice is
the picture rather than a line of text in a dropdown. And the task switch reads
the strip: a task greys out while the strip cannot supply what it is written
from, so with nothing connected only `T2VA` is lit, one picture lights `I2VA` and
`L2VA`, and two light `FL2VA`. Three pictures grey all three, because three is as
impossible for `I2VA` as none — it is the same count the node refuses on, moved
from the run to the sight of it, and the greyed button carries the same sentence as a tooltip. Badges
are counted, not sockets: turn a picture into a subject and the task it was
blocking lights up. A task already chosen and no longer possible turns red rather
than quietly failing later.

Every rectangle in the picker is drawn to its own proportion within one budget,
so the wide end reads as how little height is left: `21:9`, `32:9` and `48:9`
come out at roughly 15, 10 and 6 pixels tall. A node too narrow to hold the whole
row wraps it onto a second line rather than cutting the last rectangles off, and
grows by exactly that much.

**`duration` is a slider, in tenths of a second**, and how far it reaches is
yours to set — right-click the node, `duration`. See
[The duration widget](#the-duration-widget).

**Two model widgets, because there are two jobs.** `caption_model` reads the
references and `writer_model` writes the prompt from MiniMax's guide; `clip`
works here exactly as it does on the caption nodes, and is used for the
references when connected. On T2VA no captioner is touched at all.

**What each task expects of the strip:**

| Task | Pictures | Everything else on the strip |
|---|---|---|
| T2VA | none | ignored entirely, and the node says so rather than describing them |
| I2VA | exactly one — the first frame | goes in as reference material |
| L2VA | exactly one — the final frame | goes in as reference material |
| FL2VA | exactly two — first, then last | goes in as reference material |
| Ref2VA | any number | at least one reference of some kind is required |

A mismatch is refused before anything is downloaded or loaded, and the message
names the strip rather than the sockets, because the strip is where the fix is:
an image that is in the way gets switched off or given a different label, not
unplugged.

**The outputs are the union of both writers'** — the three T2VA fields, the six
Ref2VA fields, and the reference block itself. A task that does not write a field
leaves it empty, because a node's outputs cannot change with the value of one of
its widgets. The last output, `references`, is not text: it is what the strip
numbered, for [Reference Slots](#minimax-h3-reference-slots) to wire into
`MiniMaxH3ReferenceToVideo` in the same order.

> **One difference from Multi Reference Caption.** That node writes the block in
> the guide's own order — subjects, pictures, videos, audio — whatever the wiring
> says. This one writes it in strip order, because the strip is the whole point.
> An untouched strip is in slot order.

> **If the interface script does not load**, the strip, the task switch and the
> ratio picker fall back to the plain widgets they are built on — one text field
> and two dropdowns — and the node still runs. Those three are HTML and work in
> both of ComfyUI's renderers. The checkboxes on the input rows are drawn on the
> canvas, which *Modern Node Design (Nodes 2.0)* does not run; there the number
> on each square is the way to switch a reference off. Multi Reference Caption
> draws the same squares and needs no checkboxes at all any more.
>
> The three drawn controls are also kept off the right-side **Parameters**
> panel, which has no way to draw a control like this and would have to take it
> off the node to try. The node is where they live. Everything else about the
> node -- the prompt, the models, the duration slider -- is in the panel as
> usual.

> **This node needs a recent ComfyUI**, for the same reason Multi Reference
> Caption does: its growing inputs are `io.Autogrow` from the v3 node API. On an
> older install these two go missing and the rest of the pack registers as
> before.

### MiniMax-H3 Image Prompt Writer

Expand an image request into a detailed caption for [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image).
With no active pictures the writer expands a text-to-image prompt. With one to
16 active pictures it writes an edit prompt. The first active picture is the base.
Reorder the chips, disable inputs with their checkboxes, and add instructions via
`+ instr`. An IMAGE batch contributes its first frame.

![Image Prompt Writer in Local GGUF mode with a base image and an edit request](docs/image_prompt_writer_node.png)

- **Local CLIP**: connect `qwen3.5_9b_w4a8_gptq.safetensors` through Load CLIP,
  type `minimax`. This is the prompt model used in the
  [ComfyUI H3 Image templates](https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image).
- **MiniMax Context IR**: set `MINIMAX_API_KEY` in the ComfyUI server environment,
  or use **API providers** to select a saved profile. The profile needs
  `https://api.minimax.io/v1` (global) or `https://api.minimax.cn/v1` (China),
  a credential, and `MiniMax-H3-Image` in **Allowed models**. This calls the
  [prompt enhancement endpoint](https://platform.minimax.io/docs/api-reference/image-generation-h3-context-ir)
  with the instruction and enabled pictures. MiniMax bills the expansion.
- **API Writer** uses the selected Chat Completions model on a configured local
  or remote server. Pick the profile and model, as in **API Universal Writer**.
  Text-to-image works with a text model. For Image Edit, enable `image` in the
  model's **Allowed models** settings and use a server that accepts interleaved
  text and images. Active pictures are sent in chip order.
  [MiniMax's public prompt expansion recipes](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE)
- **Local GGUF** loads the selected model through the pack's local llama.cpp
  runtime. For Text-to-Image, select a model from **Writers**. For Image Edit,
  select a vision model and matching projector from **Captioners**. Local files
  under `models/LLM` are detected by **Model list**. Connect **Rewriter Options**
  to `local_options` for context, sampling and GPU settings. Image editing uses
  the llama.cpp build with `llama-server` and the same ordered pictures as API Writer.

Connect **API Writer Options** in API Writer mode to set the token budget,
temperature, response format and timeout. With response format `text`, the writer
asks for JSON in the prompt itself. Use `json_object` or `json_schema` when the
selected model and server support it. The `summary` output shows the profile and
model used for the request.

Connect `prompt`, `width` and `height` to **MiniMax H3 Image Conditioning**
in the ComfyUI template. With `aspect_ratio=auto` the writer uses the expander's
recommendation. Choose a fixed ratio and `area` for control over the canvas,
or enter an exact `size` with edges divisible by 32 and leave the ratio on auto.
For editing, **Reference Slots** passes pictures 1 to 9 from `references` to
Conditioning. For 10 to 16 pictures, wire the originals to Conditioning in chip
order. Use **bypass** to pass through your text or **repeat_last** to reuse an
expansion from the session or prompt library.

### P-H3 Prompt Writer (images)

Writes prompts for [Pruna P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image-Comfy)
with its official Qwen3.5-9B upsampler. Requires ComfyUI 0.44.0 or newer.

![P-H3 Prompt Writer in Text2Image mode with use_ref_description enabled, a source reference preview and frame selection](docs/p_h3_prompt_writer.png)

Connect **Load CLIP**, type `minimax`, with a `p-h3-image_prompt_upsampler` file to
`upsampler`. The similarly named 4B text encoder belongs to the image generator,
not this writer. ComfyUI manages model loading and offloading.

- **Text2Image** writes from the request, aspect ratio and 4 or 8 MP canvas.
  Connected references are ignored by default and the `references` output is empty;
  switching modes does not require disconnecting them.
- **Image2Image** reads up to eight pictures in total. The first enabled picture
  is the base; drag the reference strip to change the order, click to switch a
  reference off, and use **role** to say what to take from it. **keep** names
  what must remain unchanged. By default the canvas follows the base's ratio.

Enable **use_ref_description** in Text2Image to write the prompt from the first
active reference and your request. Its chip is marked **source**; drag the chips
to choose another source, and use **role** to say which details to take. For an
IMAGE batch, VIDEO or visual RefMod, only the first selected frame is described;
**frame** opens a picker for one frame, and Automatic uses the first frame.
The same upsampler first describes the frame, then writes a standalone prompt
from that description and your request, with no image supplied to the second
step. Your request takes priority; an empty request describes the pictured scene.
The image generator receives text only, and `references` stays empty. `summary`
shows the generated visual description. Canvas
settings still come from the ratio picker or `aspect_ratio`. With `bypass` on,
the source is prepared for its preview and the original prompt passes through.

The reference chips sit above the mode tabs. Optional `aspect_ratio` accepts
STRING or COMBO values such as `16:9`, `3840x1080` or `1.78`, overriding both
the ratio picker and `match_image_ratio`. Custom ratios keep the selected
megapixel budget, with canvas dimensions rounded to multiples of 32.

Wire `prompt`, `width` and `height` to **p-h3-image Conditioning**. Pass
`references` through **MiniMax-H3 Reference Slots**, connecting `picture_1` to
`image_1`, `picture_2` to `image_2`, and so on. Use the P-H3 text encoder and VAE
on Conditioning. Its sampler uses P-H3's eight-step sigma schedule and
ModelSamplingMiniMaxH3 shifts of 1.0/1.0; see the example workflows.

Visual **RefMods** expand into ordinary pictures. `refmod_images` selects up to
that many evenly spaced photos or frames per RefMod; all expanded pictures count
toward the eight-picture limit. Files without stored or decoded pictures need
the original H3 video VAE on the loader or `refmod_vae`. Strength is applied to
the pictures. P-H3 re-encodes them: the original RefMod latent is not injected,
and decoding then re-encoding can lose detail. Audio RefMods are refused.
`summary` reports the exact Picture numbers after expansion.

Connect native **VIDEO** outputs or the **IMAGE** output of VHS Load Video to
the same reference sockets. `video_images` automatically selects evenly spaced
frames from VIDEO and IMAGE batches larger than eight; smaller IMAGE batches
keep all pictures. All sources share the eight-picture limit.

Use **frames** under a reference to choose exact frames: inspect the overview,
browse nearby frames or enter a frame number, then tick the pictures to keep.
**Apply frames** saves the choice in the workflow; **Automatic** restores the
socket's default selection. Frames pass to both the upsampler and `references`
in chronological order. For the first preview, run the input branch once with
`bypass`. Refresh it after changing the source, or if ComfyUI evicts the input
from its cache. The picker retains small thumbnails, not the full input batch.
Numbers start at one in the received reference: VHS numbering follows its
`skip_first_frames`, `select_every_nth` and other processing; RefMod numbering
follows its stored or decoded pictures.

![P-H3 frame picker with a large preview, numbered filmstrip and selected reference frames](docs/p_h3_frames_editor.png)

The checkpoint's own template and edit brief are kept intact. Image checks cover
empty output, repetition and invalid reference tags, without requiring video
fields, shot timings or sound. The prompt library supports saving, editing and
repeating the result; references and canvas always follow the current inputs.
`bypass` passes the request through while still preparing those inputs.

Start with **9 - P-H3 text to image** or **10 - P-H3 image editing**. For edits,
choose your image on Load Image before queuing. Increase `max_length` if the
upsampler uses its token budget on thinking and returns no final prompt.

![P-H3 image editing result in Save Image: a turtle wearing a red knitted scarf](docs/p_h3_result.png)

### MiniMax-H3 Reference Caption

The writer nodes read text, not pixels. This is where the text comes from:
connect an image, an audio clip or a video, and a small multimodal model
describes it into one labelled line of `reference_assets`.

Measured on a 3.4 GB Qwen2.5-Omni-3B: **3 s for a frame, 2 s for an audio clip,
5 s for a video.** It runs `llama-mtmd-cli`, which ships beside
`llama-completion` in the same archive, so a machine whose rewrites already ran
on the binaries downloads no runtime for this.

**A machine whose rewrites did not, pays for it here.** `llama-cpp-python` is
what `gguf_runtime = auto` picks whenever the wheel is importable — and a recent
ComfyUI portable ships one — while a safetensors base never touches llama.cpp at
all; neither route has ever fetched an archive, so the first caption is where it
arrives: 34 MB, or 511 MB where `llama_backend` resolves to CUDA. The wheel does
not spare it either, however it was compiled: multimodal input goes through
`llama-mtmd-cli`, a program, and the wheel is a set of shared libraries with no
executables in it. `gguf_runtime` is therefore a writer setting — the caption
nodes run the binaries whatever it says.

**Chain them by wiring** `reference_assets` into the next node's `previous`. Each
label is numbered *within its own category*, which is the guide's own rule
(«`<Video N>` and `<Audio N>` are numbered independently»), so four assets come
out as `Picture 1`, `Picture 2`, `Video 1`, `Audio 1` — not 1 through 4.

![Two Reference Caption nodes chained: a Load Image feeds the first with role Picture, a Load Audio feeds the second with role Audio, reference_assets runs from the first node into the second node's previous input, and the viewer on the right shows both lines, Picture 1 and Audio 1](docs/node_ref_caption.png)

*An image and an audio clip through Qwen2.5-Omni-7B — 4.4 s and 3.7 s. The
second node received the first one's block on `previous` and appended to it, and
the `Audio` role described the voice rather than transcribing it: "a male
speaker with a calm and measured delivery, speaking in Russian".*

`role` picks both the label and the question asked, and the questions differ on
purpose:

| `role` | What the model is asked for |
|---|---|
| `Subject` | the features that must stay consistent across shots — build, hair, clothing and their colours, carried objects |
| `Picture` | the frame as a shot — style, shot size, camera angle, placement, environment, lighting |
| `Video` | subjects, actions in order, camera movement, cuts and pacing |
| `Audio` | **the sound, not the words** — voice timbre, apparent age and gender, delivery and rate, instrumentation, tempo, ambience |

That last row is the one that matters. `<Audio N>` in the guide is usually a
*timbre* reference, and a transcript throws away exactly the part that is needed,
so the instruction says "do not transcribe" outright. If you also want the spoken
words verbatim for a `<d>` block, that is an ASR job — run Whisper alongside and
paste the line in.

Other inputs:

- `description` — type it yourself and **no model runs at all**. The fastest way
  to add an asset you can describe in six words.
- `instruction` — override the role's question entirely.
- `max_frames` — how many frames to take from an IMAGE batch **or a VIDEO**,
  spread evenly. All of them would overflow both the context and the wall clock:
  two seconds at 25 fps is 56 images through the vision tower, and thirty seconds
  is 750. This is what makes the cost of describing a clip independent of its
  length.
- `context_size` — `0` sizes the context from the references and from the card,
  which is not the same as the model's own: llama.cpp reserves the whole KV cache
  before it reads a pixel, and a model trained for 262144 tokens asks tens of
  gigabytes for it — Qwen3-VL-8B asks 36 GB, on a card that would have captioned
  the picture in three seconds. One 1024×1024 frame is already twenty-odd media
  chunks, so the count of references is half of the answer and the memory of the
  device is the other half. Type a number to say it yourself; too small a value
  fails the run instead of truncating it.

**A thinking model does not think into the block.** Whatever a captioner writes
between `<think>` and `</think>` is cut before the caption becomes a line of
`reference_assets`, on both the GGUF and the `CLIPLoader` path — otherwise an
empty `<think> </think>` opens every caption, and a model that really reasons
sends its whole deliberation on into the writer's prompt.

Cutting it is only half, and the other half cannot be a flag. The writers render
the chat template themselves and pass `enable_thinking=False`; a caption cannot,
because `llama-mtmd-cli` applies the template itself — that is what puts the
media tokens in the right place — and offers no switch for the thought channel.
The reasoning is therefore still charged against `--predict`, so every caption
question ends with a line asking for the description alone. If a model of yours
still insists, `instruction` is where to say so in your own words.

> **Not every multimodal GGUF works here**, and the ones that do not fail loudly.
> llama.cpp's `mtmd` has to understand the projector format: Gemma 4's aborts the
> process outright on `b10310` — with Google's own file and with unsloth's alike,
> on the CUDA and CPU builds alike — while `llama-completion` runs the same model
> as text with no trouble. So the `captioners` list holds only pairs that have
> actually been run:
>
> | Captioner | Download | Modalities |
> |---|---|---|
> | Qwen2.5-Omni-3B `Q4_K_M` + mmproj | 3.4 GB | image, audio, video |
> | Qwen2.5-Omni-7B `Q4_K_M` + mmproj | 5.8 GB | image, audio, video |
>
> A model and its `mmproj` sitting together in one folder under `models/LLM` is
> offered automatically, so you can try another without editing anything.

### MiniMax-H3 Multi Reference Caption

The same job for a whole shot, in one box. A chain is exact, but it grows: five
references are five nodes, five wires and five chances to leave the wrong role on
one of them. This node folds the chain into a single box and takes the role out
of your hands entirely — **the group an asset is plugged into is its label.**

That is the guide's vocabulary made structural. Ref2VA defines exactly four
reference labels and forbids inventing more, so four groups of inputs cover the
format completely, and describing an image as audio stops being *possible* rather
than merely discouraged.

| Group | Label | What belongs here |
|---|---|---|
| `subjects` | `<Subject N>` | reusable visible content — a person, an animal, a place, a costume, a style |
| `pictures` | `<Picture N>` | an image serving as an actual frame: first, last, key, composition anchor, storyboard panel |
| `videos` | `<Video N>` | whole-video relationships — an edit source, a continuation point, camera work and pacing |
| `audios` | `<Audio N>` | a voice timbre, music, ambience, an effect |

Slots grow as you fill them, one spare always waiting, so the node is as tall as
the shot needs and no taller.

![The Multi Reference Caption node beside a text viewer: seven reference slots each with a checkbox on its own row, the connected video_0 slot unchecked while subject_0 and audio_0 stay checked, and the viewer showing a block of exactly two lines, Subject 1 describing a turtle with gem-like stones on its shell and Audio 1 describing a male voice over birdsong and splashing water](docs/node_multi_ref_caption.png)

*Two assets through Qwen2.5-Omni-3B — 6.0 s. The video is wired up and switched
off, so the block came out as `Subject 1` and `Audio 1` with no `Video` line at
all: the reference stays in the graph, it just costs nothing on this run.*

**The strip under the inputs is one coloured square per connected reference**,
in the order the block will be written and labelled with what each will be
called. Click a square to switch that reference off without unplugging anything:
a caption costs a model load and seconds to minutes, which makes "everything
except this one" the ordinary thing to want, and pulling the wire out to get it
throws away the wiring you meant to keep. The state is saved with the workflow
and travels through the API like any other value.

Squares here do not drag and their labels do not cycle, unlike the
[Universal Writer](#minimax-h3-universal-writer)'s: this node writes the block in
the guide's own order, and the group an asset is plugged into is what names it.

**The band under a square asks that one reference something else.** Dark while it
is asked its role's usual question, amber once it is not, with a ring of the same
colour round the whole square. Click to write the
question, right-click to take it back, hover to read it. It is per reference
rather than per node on purpose — a node describing a picture, a clip and a sound
at once has no single question that suits all three.

The checkbox in that little window decides which of two things your text is, and
the band then says which: `+ instr` for a line **added** to the role's question,
`= instr`, on an orange band, for one asked **instead** of it. Both are needed.

| What you write | Which mode | Why the other one fails |
|---|---|---|
| `Do not mention the window.` | added | A rule is not a question. Asked on its own it leaves nothing wanting a description, and the model answers the rule: `Subject 1: No`. |
| `Always answer "blah blah blah".` | instead | Left after the role's question it contradicts it, and a small model settles that by describing anyway. |

Asked *instead*, your text also takes over from `length`: whoever writes the
question owns the shape of its answer, so state the length yourself if it
matters. Added, the role's question and the length preset both stay where they
were.

**A `videos` slot takes a `VIDEO` or an `IMAGE` batch**, whichever your loader
hands out — VideoHelperSuite's `Load Video (Upload)` wires straight in. Frames
are sampled evenly up to `max_frames` either way, so the cost of a clip stays
independent of its length.

**The block is written in the guide's order** — subjects, pictures, videos, audio
— rather than in wiring order, and each label is numbered within its own
category, continuing from whatever arrives on `previous`. So this node still sits
in a chain with single caption nodes on either side.

`model`, `length`, `seed`, `max_frames`, `context_size` and `bypass` are shared
by every asset in the node; the question is not, and lives on the band under each
square. `role` and `description` are gone on purpose: the group is the role, and
a description you write out yourself belongs to one asset at a time. Keep
[Reference Caption](#minimax-h3-reference-caption) for an asset you would rather
describe by hand.

> **This node needs a recent ComfyUI.** Its growing inputs are `io.Autogrow` from
> the v3 node API. On an older install it is the only node that goes missing —
> the rest of the pack registers exactly as before.

### Captioning with a model ComfyUI already has loaded

Both caption nodes and the Universal Writer have a `clip` input. Connect a multimodal text encoder from
`CLIPLoader` and every asset is described by *that* model instead of by the GGUF
in `model` — which then stays exactly where ComfyUI's allocator put it.

That is the whole point. The GGUF route runs `llama-mtmd-cli`, one process per
asset, so a shot with five references reads the weights off disk five times. A
loaded encoder is read **once**, and reruns while you tune the wording cost
nothing at all. Measured here: three assets — an image, a clip and its
soundtrack — through Gemma-4 12B in **24 s**, with the log showing a single
`Requested to load` for all of them.

Do not read that as "faster", though. The same three assets through the 3.4 GB
Qwen2.5-Omni-3B on the GGUF route took **16 s**, loads and all: a small model
loads quickly enough that three reloads cost less than a 12B costs to run. What
this route buys is a *bigger* model at no reload penalty, and reruns that skip
the loading entirely — and the gap widens with every asset and every rerun.

Which encoders qualify, and what each can take in:

| Encoder | Sees | Hears |
|---|---|---|
| Qwen3-VL 4B / 8B | yes — a batch of frames is read frame by frame | no |
| Gemma-4 12B | yes | **yes** — the "unified" build, audio projected straight in |
| Gemma-4 E2B / E4B | yes | has the parts, but see below |
| Gemma-4 31B | yes | no |

Gemma-4 is recognised from the checkpoint itself, so the `type` widget on
`CLIPLoader` does not matter for it. Connect a vision-only encoder with audio
wired up and the node refuses before anything runs, the same way it refuses a
projector without an audio tower.

> **Two things to know about Gemma-4 E2B/E4B**, both observed on ComfyUI 0.30
> with an fp8 E4B and both reproduced with ComfyUI's own `Generate Text` node on
> the same file — so neither is something this pack can fix:
>
> - **Audio does not arrive.** The caption comes back saying no clip was
>   provided. The node logs a warning when it sees this shape of encoder rather
>   than refusing outright, because another checkpoint may behave. Gemma-4 12B
>   takes audio correctly.
> - **It reasons out loud.** The prompt primes an empty thought channel and this
>   build fills it anyway. The caption nodes cut everything up to the closing
>   tag, so what reaches `reference_assets` is the answer alone — but if you wire
>   `Generate Text` up yourself, that is yours to strip.

Two widgets are inert on this route and say so in their tooltips: `model`, which
is not consulted at all, and `context_size`, which is a llama.cpp KV-cache knob.
`max_frames` still applies — the frames are thinned here, before the encoder sees
them, rather than being left to the encoder's own 1 fps subsampling.

The GGUF route is not going anywhere: it reaches models ComfyUI has no encoder
for, and it needs nothing loaded in advance. Leave `clip` unconnected and nothing
about either node changes.

### The captioner is loaded once, not once per reference

A strip of six references used to be six model loads. `llama-mtmd-cli` runs one
description and exits, which is right for a single picture and wasteful for a
strip: the model and its projector were read from disk again for every asset,
before a pixel was looked at.

Now the whole loop is served by one `llama-server`, started before the first
reference and stopped after the last. The model, the projector, the sampling,
the seed and the system turn are the same either way, and only the loading
changes. Three references on an 8B captioner with a warm file cache: 8.7 seconds
before, 3.6 after, and the gap widens with every reference you add and with
every gigabyte the captioner weighs.

Each path repeats itself exactly — the same graph at the same seed gives the
same caption every time. Between the two, a caption can still land a word apart,
*a blue circle centered on* against *a blue circle is centered on*, where two
tokens are near-tied and the two runtimes' batching breaks the tie differently.
That is float arithmetic rather than a different question being asked, and it is
why the system turn is stated rather than left out: left out, the difference was
not a word but a different description.

You do not switch this on, and nothing is downloaded for it: `llama-server`
ships in the same release archive as `llama-mtmd-cli`, so it is looked for
beside the captioner that has already been found — and only beside it, since a
server from another build could pair the model with an older mtmd that cannot
read the projector this one just accepted.

It is an optimisation and never a requirement. A build without it, a port that
will not bind, a server that never reports healthy: each is written to the log
and the run carries on one process at a time, slower and otherwise identical.
One reference does not start a server at all, there being no second load to
save. `MINIMAX_H3_MTMD_SERVER` takes `never` to keep the old behaviour and
`always` to use a server even for a single asset.

### MiniMax-H3 Guide Prompt (any LLM)

Builds the guide-based `system_prompt` and `user_prompt` and returns them as
strings, without running anything. Costs no VRAM and no time. Wire them into
whichever LLM node you already use — local, API, remote — when you would rather
not run the model here. It covers all five tasks, Ref2VA included.

A third output, `prompt`, is both of them in one string, because most LLM nodes
take exactly one — **including ComfyUI's own `Generate Text`**, which since 0.30
runs a language model in ComfyUI's own process off a model loaded by `CLIPLoader`.
That is the shortest route to this pack's output with no GGUF downloaded at all,
if you already keep a Qwen3-VL or Gemma-4 text encoder for an image model:

![Load CLIP feeding Generate Text, with the Guide Prompt node's third output wired into its prompt input and a text viewer on the right showing the finished rewrite: subject_definitions binding Subject 1 and Picture 1, the summary with its reference-generation task type, the retention analysis, the shot-by-shot body with timecodes, the soundscape and non_diegetic_music](docs/node_guide_prompt.png)

*Ref2VA through a 4B Qwen3.5 — 37.7 s, and all six sections came back. The node's
own status line is the number worth knowing: a 25 042-character system prompt is
**about 10 240 tokens of context** before the model writes a word, which is what
decides whether a given encoder can take this at all.*

Three settings on `Generate Text` decide whether it works:

- **`max_length`** — its default of 512 is the *output* budget, and six Ref2VA
  sections do not fit in it. Around **2048** is right, which is what this pack's
  own writers use (`max_new_tokens` in the options node). Measured on a 4B
  Qwen3-VL: a Ref2VA rewrite came back complete in **53 s including the model
  load**, stopping on its own at roughly 580 tokens — inside 2048 with room to
  spare, and past 512 by enough that the default would have cut it mid-section.
- **`thinking`** — off. The guide asks for fields and nothing else; reasoning
  spends the budget above on prose you then have to strip.
- **`use_default_template`** — leave it on, and leave `format` on `plain`.

`format` is the one knob worth knowing about. On `plain` the two prompts are
joined with a blank line and the LLM node wraps the result in the model's own
chat template, which puts the whole guide inside the *user* turn. On `chatml` the
turns are written out here instead: a Qwen text encoder skips its own template as
soon as the text starts with `<|im_start|>`, so the guide arrives as a real
system message. That branch also skips Qwen's thinking suppression, so the empty
`<think>` block is written for you. It is Qwen-shaped by construction — on Gemma
or anything else, stay on `plain`.

### MiniMax-H3 Prompt Check

Every other node here writes a prompt and then checks what it wrote. This one
only checks, and takes the text on a socket — so a prompt that came from
somewhere else is read by exactly the rules a rewrite from this pack is read by,
and split into the same fields. Anything that produces MiniMax-H3 prose can feed
it: another pack's node, a text file loaded into the graph, a prompt found
online, or something typed into the widget.

No model is loaded and nothing is generated. The run costs a few milliseconds.

![The MiniMax-H3 Prompt Check node: an options socket and three reference sockets labelled ref_0, ref_1 and ref_2 down the left, two of them connected; nine outputs down the right — prompt first, then integrated_multimodal_description, subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape and non_diegetic_music, with findings last. Below them a tall empty prompt box, a task dropdown reading Ref2VA, a duration of 9.9, and one button: Save the last prompt](docs/node_prompt_check.png)

*The outputs are the Universal Writer's, minus the two that describe its own captioning work — so a check node drops into a graph where a writer stood, and a writer drops in where a check node stood. The duration is 9.9 rather than a round number because it is whatever the prompt was written for, not what you would have asked a writer for. And there is one button rather than two: the node runs no model, so there is nothing to repeat and nothing to point at the library — but what it read, it can keep.*

| Input | What it is for |
| --- | --- |
| `prompt` | The text to read. It leaves the first output untouched, so the node can sit in the middle of a graph without changing what reaches the generator. |
| `task` | Which task the prompt was written for. It decides which fields the answer should have, how many references of each kind it may cite, and whether an alignment line belongs at the top — so getting it wrong makes the reading wrong rather than absent. |
| `duration` | How long the target video is. Cut times are read against it, so set it to what the prompt was written for rather than to what you would ask a writer for. |
| `references` | Optional. Only their kind and number are read — nothing is decoded and no captioner runs. With them connected, the reading also covers what the text cites against what is actually here; with nothing connected those two rules are skipped and the rest still apply. |
| `options` | Optional, and only `self_check` is read from it: how much is announced on screen. |

The outputs are the prompt, the seven fields any task can fill, and `findings` —
the same block the node writes under itself, `!` for a warning and `-` for a
note, empty when there is nothing to say. That output always carries everything
found: `self_check` governs what the nodes announce during a run, which is a
question about noise, and wiring the output is asking.

**Save the last prompt** is on it as well, so a prompt that arrived from
elsewhere and read clean can go into [the prompt library](#the-prompt-library)
under a name, a description and groups, exactly like one this pack wrote.

One reading is worth expecting. Most prompts written for H3 by hand are prose:
no field labels, no shot list, no tags. Read as `T2VA` such a text is reported
as missing its fields and its `[Shot 1]`, and that is a true reading rather than
a fault — a prompt like that is an *instruction to a rewriter*, not a finished
H3 answer. Feed it to one of the writers and check what comes back instead.

### MiniMax-H3 Prompt Reducer

Every other node here expands. This one contracts: it turns a finished H3 prompt
back into the short line it could have been written from.

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a low-angle medium
shot frames a sleek black cat walking steadily along the top of a weathered wooden
fence in a quiet suburban yard at dusk. The camera tracks right with small amplitude
at slow speed, following the feline as its soft fur catches the fading golden light...

overall_soundscape: A gentle evening breeze rustles through nearby grass and leaves...
non_diegetic_music: A sparse piano melody at a slow tempo...
```

comes back as

```
A black cat walks along a wooden fence in a yard at dusk.
```

Three things it is for. Changing one word of a prompt you liked without
rewriting the other four hundred — reduce, edit the line, feed it back to a
writer. Feeding a prompt written for H3 to a generator that wants a short one:
an H3 prompt is H3-shaped, and Wan, Hunyuan and Kling are not. And putting a
readable line on a prompt you saved, where the card shows a name rather than a
paragraph.

**Half of this is not a job for a model, and that is what makes it work.** An H3
prompt is not prose — it is a known shape: named fields, the fixed alignment
sentence at the top, `[Shot 2] At 0:03` markers through the body, `<Picture 1>`
tags wherever a reference is cited, dialogue fenced in `<d>`. All of that is
scaffolding, all of it is recognisable by rule, and all of it comes off before a
model is asked anything. What reaches the model is one paragraph of ordinary
description under a short instruction, which is why a 4B does this well. Asked
to "reverse this document" instead, it would not.

Reference bindings go off with the rest of the scaffolding, and that is
deliberate: for Ref2VA, `subject_definitions` and `retention_analysis` describe
*the assets*, and a short prompt that kept them would describe pictures the next
run will not have.

| Input | What it is for |
| --- | --- |
| `prompt` | The finished prompt to shorten. Any of the five tasks, and you do not have to say which: the text is split against every field name either family uses, and one with no field names at all is read whole as the description. |
| `model` | Any instruction-following GGUF, from the same list the writers use — including `on disk:` and `ollama:` entries — and the **Model list** button under the node edits it. This asks much less of a model than writing does, so the smallest entry in the list is a reasonable choice here even when it is not one there. |
| `detail` | How much comes back. `idea` is the bare line, ten words at most; `sentence` allows the place and the time of day; `paragraph` keeps one sentence per thing that happens, which is what a prompt with several shots needs if the order is to survive. |
| `subjects` | How specifically people are named: `as written`, `age and gender`, or `impersonal`. |
| `keep_camera` | Keep the shot size, the angle and the camera move. Off by default — the camera is usually the writer's invention rather than yours, and leaving it out lets the next rewrite choose again. |
| `keep_audio` | Fold the soundscape and the music into one sentence at the end. Off by default, and off means the sound sections never reach the model at all: they are dropped by the parser, not by the instruction. |
| `keep_style` | Keep the medium and the look the prompt opens with — live-action, animation, cinematic, documentary. |
| `language` | Which language the answer comes back in. Empty means the language of the input. On this node it is a second pass — the line is shortened first and translated afterwards — which is what makes it reliable; the sub-section below says why it has to be. |
| `greedy`, `seed`, `keep_model_loaded` | As on the writers. Greedy is worth keeping on here: sampling is what turns "a black cat" into "a sleek obsidian feline". |
| `options` | Optional. The usual [Rewriter Options](#minimax-h3-rewriter-options) node — `device`, `n_ctx`, `gpu_layers`, `gguf_runtime` and the rest. |
| `bypass` | Hand `prompt` straight to the output and run nothing. |
| `system_prompt` | Replace the whole assembled instruction with your own. `detail`, `subjects` and the three keeps then stop applying — they exist only to build the text you are overriding. `language` still applies, because it is a separate request rather than part of that text. The parsing still happens either way, so what your instruction is handed is the cleaned scene rather than the raw text. |

Two outputs: `short_prompt`, and `scene` — the description with the scaffolding
taken off and nothing else done to it. No model has touched `scene`. Wire it
when the deterministic half is all you wanted.

![The MiniMax-H3 Prompt Reducer node with a Show Any node beside it: an options socket on the left, short_prompt and scene as the two outputs on the right, a tall prompt box holding a full I2VA prompt with its integrated_multimodal_description and [Shot 1] marker, then the widgets — a 35B model on disk, detail on idea, subjects on impersonal, keep_camera and keep_audio true, keep_style false, language reading Chinese, greedy true, a seed set to randomize, keep_model_loaded and bypass false, an empty system_prompt box, and the Model list button under it. Under the node the status line reads '110 words in, 33 characters out - 1 shot markers dropped - translated into Chinese', and below it the two Chinese sentences themselves, which the Show Any node repeats](docs/node_reducer.png)

*Four hundred words of cat, fence and dusk down to two short Chinese sentences, in 25 seconds on a 35B. Three axes are doing visible work at once: `detail` is on `idea`, so what comes back is the bare line; `keep_camera` put the low-angle tracking shot back into it; `keep_audio` folded the whole soundscape into the second sentence; and `keep_style` is off, so the `Live-action, cinematic` the prompt opened with is gone. The count is in characters rather than words because Chinese does not write the spaces a word count needs.

One thing it also shows honestly: `subjects` is on `impersonal`, and the cat still comes back black. The rule asks for the bare kind and this model kept the colour — that axis leans hardest on the model of the three, and it is people it is really aimed at.*

#### What the axes actually do

One prompt, one 4B, the axes moved one at a time. The input was a documentary
I2VA prompt about an elderly fisherman hauling a net, two shots, with a
soundscape and a cello drone:

| Setting | What came back |
| --- | --- |
| default | An elderly fisherman with a thick grey beard and a faded yellow oilskin jacket hauls a dripping net over the gunwale of a small wooden trawler under a bruised pre-dawn sky. |
| `detail: idea` | An elderly fisherman hauls a net on a boat. |
| `subjects: age and gender` | An elderly man hauls a net on a boat under a pre-dawn sky. |
| `subjects: impersonal` | A subject hauls a net over the side of a boat under a pre-dawn sky. |
| `keep_camera` | A handheld close-up follows an elderly fisherman... |
| `keep_style` | Live-action, documentary style. An elderly fisherman... |
| `keep_audio` | ...under a pre-dawn sky. Waves slap the hull, rope creaks, gulls cry overhead, and the man breathes heavily. A low cello drone underneath. |

`impersonal` is for templates you fill in afterwards. Fed to a generator as it
stands, it produces exactly the anonymous nothing it asks for.

There are four axes rather than one abstraction dial because they are
independent. How long the answer is and how specifically it names people are
different questions — a one-line prompt can still say "a woman in a red coat" —
and the camera, the sound and the film look are each kept or dropped on their
own.

#### `language` is a second pass, and it has to be

Asking for the reduction and the language in one request does not work. The
worked example inside the instruction is written in English and cannot be
anything else — nothing here can translate it into a language typed into a
widget — and a model copying the demonstration copies its language along with
everything else it copies from it.

That failure is not uniform, which is what makes it worth describing. With the
three keeps *off*, the example is short and the language rule was obeyed. With
them *on*, which makes the example longer and more elaborate, three models in a
row answered in English however the rule was phrased and wherever it was placed
— including as the last line of the system prompt, after the example.

So the Reducer shortens first and translates the finished line afterwards, in
its own request. That request has one objective, no example to copy and nothing
to trade off against, and the same models obey it: every combination tried came
back in the language asked for, on a 4B, a 9B and a 35B, in Russian and in
Chinese. It costs one short generation on a model that is already loaded, and
the node keeps it loaded between the two whatever `keep_model_loaded` says about
afterwards.

Two things follow from the design:

- **`system_prompt` does not switch it off.** The three keeps, `detail` and
  `subjects` all stop applying when you replace the instruction, because they
  exist only to build the text you replaced. `language` is not in that text at
  all, so it still applies.
- **`Reduce Prompt (any LLM)` cannot do this**, because it builds one request
  and runs nothing. There the language is a rule inside the instruction, obeyed
  or not depending on the model — which is exactly the behaviour described
  above. If a short prompt comes back from that node in the wrong language, this
  is why, and the Reducer is the reliable path.

What remains is translation quality rather than translation refusal: a small
model leaves the occasional word untranslated. That is the model, and a larger
one leaves fewer.

#### Effect tokens through the reducer

An [effect embedding](#minimax-h3-effect-embeddings) written into the description
comes out the other side untouched, small `e` and all — the capitalisation that
tidies the start of a sentence knows to leave one alone. A token placed *above*
the first field label does not survive, because the head of a prompt is dropped
along with the alignment sentence and everything else that is not the description
or the sound. That is the reducer's contract rather than an oversight: what it
hands back is a reusable line, and the head of an H3 prompt binds it to
particular reference frames.

Worth knowing too: the word count on both this node and the check counts a token
as one word. Its real cost in prompt positions is on the embeddings node's
`tokens` output.

#### One loop worth knowing

Reducer → edit the line → writer → [Prompt Check](#minimax-h3-prompt-check).
If the cat is still on the fence at the end of that circuit, both halves of the
pack are behaving. It is also the cheapest way to find out whether a given small
model is good enough at this: run it and read the line.

### MiniMax-H3 Reduce Prompt (any LLM)

The same reduction with nothing run: it builds the `system_prompt` and
`user_prompt` and hands them back as strings for whichever LLM node you already
use — local, API, remote. Costs no VRAM and no time. The parsing still happens
here, so the scene your model receives is already clean.

![The MiniMax-H3 Reduce Prompt (any LLM) node with a Show Any node beside it displaying the assembled system prompt: the rules list — never upgrade a word, add nothing, drop the writing rather than the story, answer with one sentence, name the subjects as written, drop the camera, drop the medium, write the answer in Russian — then the output contract, then the worked example with its scene and its answer 'A black cat walks along a wooden fence in a yard at dusk', and last the two lines saying the example is in English and the answer is to be in Russian. Four outputs run down the right of the node: system_prompt, user_prompt, prompt and scene](docs/node_reduce_prompt.png)

*The same reduction with nothing run, in 0.01 s. The whole instruction is visible here, which is the point of the node: 2112 characters of system prompt, 632 of user prompt, and every rule in it put there by a widget. The last two lines are the single-request language attempt described above — this node can only build one request, so that is the best it can do.*

The same four axes steer it, `format` joins the two outputs the way it does on
[Guide Prompt](#minimax-h3-guide-prompt-any-llm), and the fourth output is
`scene`, as above.

### MiniMax-H3 Effect Embeddings

MiniMax published ten effect embeddings alongside H3 — bullet time, dark magic,
fire breath, the Truman Show pull-back, and six more. This node puts them into a
prompt.

They are not keywords, and the difference matters for everything below. Each file
holds a single tensor of shape `[N, 5120]`: fifty to a hundred and forty
positions of prompt that have **already been through the text encoder**, saved as
numbers. Writing `embedding:minimaxh3_bullet_time` in the text is not a request
for bullet time in words — it is an instruction to ComfyUI's tokenizer to splice
those positions into the sequence at that point. So an effect costs prompt
*length* rather than adjectives, it arrives at full strength or not at all, and
no rewording of it is possible. There is nothing to phrase.

| Effect | Positions | Effect | Positions |
| --- | --- | --- | --- |
| Art is explosion | 50 | Kiss camera | 97 |
| Dark magic | 59 | Fire breath | 118 |
| Truman show | 90 | Blooming flowers | 123 |
| Bullet time | 94 | Spiral ascent | 131 |
| Storm magic | 137 | Four seasons | 142 |

**This needs ComfyUI 0.33.0 or newer.** That is the release where `embedding:`
started reaching the MiniMax-H3 tokenizer; on anything older the token is read as
ordinary words. The node checks your version and says so.

![The MiniMax-H3 Effect Embeddings node beside a Show Any node. Two badges sit above the title bar: the pack's own, and a grey one reading "bypass". Three outputs run down the right edge — prompt, wired across to the other node, then tokens and findings. Below them a text box holds a three-field T2VA prompt: integrated_multimodal_description: A cat walks along a fence at dusk, overall_soundscape: Wind in the grass, non_diegetic_music: N/A. Under it the placement widget reads "start of the description". Then the grid of ten, each row a checkbox, a name and a cost: Art is explosion 50 tok, Blooming flowers 123 tok, Bullet time 94 tok, Dark magic 59 tok, Fire breath 118 tok, Four seasons 142 tok, Kiss camera 97 tok, Spiral ascent 131 tok, Storm magic 137 tok, Truman show 90 tok. Three are ticked — Art is explosion, Dark magic and Fire breath — their boxes green with a white tick. A line under the grid reads "3 selected, 227 tokens", and below it the bypass widget reads false. There is no download button. The Show Any node displays the result: the three tokens run together after the integrated_multimodal_description label and in front of the sentence, then overall_soundscape and non_diegetic_music exactly as they were written](docs/node_effect_embeddings.png)

*Nothing here is loaded or generated — the node writes a hundred characters into
a text and reads ten file headers. The download button is absent rather than
disabled: all ten are on disk, so there is nothing to fetch and no reason for the
row to take up space. The costs in the right-hand column are read out of the
files themselves, not from a table in this pack, and 50 + 59 + 118 is the 227
under the grid. This is the `start of the description` placement, which is why
the three tokens open the description field rather than sitting above it — and
why the blank lines that follow them will reach H3 as single spaces, which the
`findings` output says out loud.*

| Input | What it is for |
| --- | --- |
| `prompt` | The prompt to put the tokens into — a writer node's output, a loaded file, something typed. It is passed through character for character apart from the tokens themselves. |
| `placement` | Where the tokens go: the start of the description, the top of the prompt, or the end. Not a matter of taste — see below. |
| `effects` | The grid of ten. A tick, the name, what it costs, and a mark on the ones that are not downloaded yet. |
| `bypass` | Hand `prompt` through and put no tokens in it. `tokens` comes back 0, `findings` empty, no file header is read. The node's title bar carries the switch as a badge, like the writers. |

| Output | What it is |
| --- | --- |
| `prompt` | The prompt with the tokens in it. |
| `tokens` | How many positions the chosen effects take up — read out of each file's own header, so it is the real number rather than a table in this pack. |
| `findings` | What the node noticed, one per line. Empty when there is nothing to say. |

Files are fetched by the button on the node into ComfyUI's own
`models/embeddings` folder, which is the only place `embedding:` looks. All ten
come to about ten megabytes. Running the node twice does not stack the tokens —
it takes its own back out before it puts them in — so unticking an effect removes
it from a prompt that already had it.

#### Every way this fails is a silent one

That is the reason it is a node at all rather than something you type. ComfyUI
drops a token it cannot use with one line in the console and no other sign: the
video generates, it simply generates without the effect, looking exactly like a
prompt that never asked for one. Three ways that happens, all of them things the
self-check now looks for:

- **A capital letter.** The test is `word.startswith("embedding:")` and nothing
  else, so `Embedding:` is not a smaller mistake than a typo. (The Prompt Reducer
  used to introduce one by capitalising the start of a sentence. It no longer
  does.)
- **A token stuck to the word in front of it.** Tokens are found after
  whitespace, or at the very start of the text. `a cat.embedding:x` is four
  ordinary words.
- **A full stop on the end of the name.** A trailing comma is stripped by
  ComfyUI and forgiven; a full stop is looked for as part of the file name. This
  one is worth knowing precisely, because it is a trap that hides on Windows:
  the path layer there drops a trailing dot, so `embedding:minimaxh3_dark_magic.`
  loads on Windows and quietly loses the effect on Linux and macOS. The prompt
  you share is not the prompt they run.

Weights do not work either — `(embedding:x:0.8)` is read as literal text, because
H3 tokenizes with weights disabled. There is no way to ask for a weaker effect.

#### Placement is not a matter of taste

ComfyUI's tokenizer joins everything that follows a token onto one line: the text
after `embedding:name`, up to the next token, goes through `' '.join(text.split())`
before H3 sees it. A token at the top of a prompt therefore flattens every blank
line between the fields below it into a single space.

`start of the description` — the default — opens the description field itself,
after its label. That is the field the writers put the scene in and the field
every downstream node reads, and it is the position an effect was meant to
modify. The fields below it are flattened.

`end of the prompt` puts the tokens last. Nothing follows them, so nothing is
flattened, and the prompt reaches H3 with its field separators intact.

`top of the prompt` puts them above everything, before the alignment sentence.

A prompt with no field labels at all has no description to open, so it falls back
to the top and says so on the `findings` output.

One thing to know if you use the Reducer: a token above the first field label is
dropped by it, along with the rest of the prompt's head. The reducer keeps the
description and the sound, not what sits above them. A token inside the
description survives.

What is *not* known here: whether two of these combine into anything sensible.
They are separate pieces of encoded prompt, and asking for a spiral ascent
through four seasons is asking the model to reconcile them. The node allows any
combination because there is no reason to forbid one; whether it is worth
generating is yours to find out.

### MiniMax-H3 LoRA Triggers

A LoRA trained with a trigger word in its caption prefix does nothing at all
until that word is in the prompt. This node keeps the list of those words on the
node, with the workflow, and puts them back into a finished prompt.

Nothing upstream knows about them. The writers write prose, the reducer shortens
it, the self-check judges it — and a word whose only job is to wake an adapter
three nodes later is exactly the sort of thing all three drop. So it gets typed
in by hand after every run, and lost on the next one. That is the only place in
the whole pipeline where finished text has to be edited by hand, and it is the
one part of a prompt that cannot be regenerated.

One row is one adapter: a grip, a tick, a name for your own use, the words that
go into the prompt verbatim, a short mark saying where they go, and a cross to
remove the row. A row can hold several words separated by commas, which is what
an adapter with more than one trigger needs — they are checked and added one at a
time, so a row is never half-duplicated. Triggers that want different parts of
the prompt want different rows.

Drag a row by its grip to move it. The order is not decoration: rows sharing a
placement are written into the prompt in list order, which is the order they read
in.

![The MiniMax-H3 LoRA Triggers node beside a Show Any node. Two badges sit above the title bar: the pack's own, and a grey one reading "bypass". Three outputs run down the right edge — prompt, wired across to the other node, then added and findings. Below them a text box holds a three-field T2VA prompt: integrated_multimodal_description: A cat walks along a fence at dusk, overall_soundscape: Wind in the grass, non_diegetic_music: N/A. Under it two rows of the list. Each is a drag grip, a green checkbox with a white tick, a field for the name, a wider one for the words, a placement mark and a cross: "Facial realism" holding "ohwx face" marked body, and "Neon" holding "neon glow, wet asphalt" marked top. The two marks line up in a column, and so do the two pairs of fields. A line under them reads "2 on of 2", then the bypass widget reading false, then a full-width button, "add a trigger". The Show Any node displays the result: "neon glow, wet asphalt" alone on the first line, a blank line, then integrated_multimodal_description opening with "ohwx face." in front of the sentence, and overall_soundscape and non_diegetic_music exactly as they were written](docs/node_lora_triggers.png)

*Two rows, two placements, one pass. `ohwx face` opened the description — after
the label and in front of the prose, which is the position a caption-prefix
trigger was trained in — while the second row went to the top carrying two
triggers at once, checked and written as two. Nothing was taken out to manage
it: run the same text through again and `added` comes back empty, because both
words are already there. The placement marks are a fixed column, which is what
keeps the fields starting and ending on the same pixel in every row; drag a row
by its grip to change the order they are written in.*

| Input | What it is for |
| --- | --- |
| `prompt` | The prompt to put the words into — a writer node's output, a reducer's, a loaded file, something typed. It is passed through character for character apart from the inserted words. |
| `triggers` | The list. A grip, a tick, a name, the words, the placement mark, a cross; a button below adds a row. Saved with the workflow. |
| `bypass` | Hand `prompt` through and add nothing. The text goes out exactly as it came in; `added` and `findings` come back empty. The node's title bar carries the switch as a badge, like the writers. |

| Output | What it is |
| --- | --- |
| `prompt` | The prompt with the words in it. |
| `added` | The words this run actually put in, comma separated. Empty when every switched-on trigger was already there, which is the normal state on a second run. |
| `findings` | What the node did and noticed, one per line. Empty when there is nothing to say. |

Driving it from the API rather than from the interface, the widget takes the JSON
array of `{name, words, where, on}` objects the interface writes — and two
friendlier shapes as well, because they are what anyone writes first: a plain
array of strings is read as a list of trigger words, and text that is not JSON at
all is read as one row's worth of words.

#### Where each word goes

The mark in the row is short on purpose. Seeing where every word is bound for
without opening anything is half of what the list is for, and five full dropdowns
would not fit across a node at any sensible width. Clicking the mark opens the
choice.

| Mark | Placement | Where it lands |
| --- | --- | --- |
| `body` | `start of the description` | Opens the description field itself, after its label. |
| `top` | `top of the prompt` | Above everything, before the field labels. |
| `sound` | `start of the soundscape` | Opens `overall_soundscape`, after its label. |
| `music` | `start of the music` | Opens `non_diegetic_music`, after its label. |
| `end` | `end of the prompt` | Last, after everything. |

The description is the default and usually the right answer: it is the field the
writers put the scene in, the field every downstream node reads, and the position
a caption-prefix trigger was trained in. The top and the end are positions rather
than fields, which is why those two still work on a text carrying no labels at
all.

The field set differs by task — three for T2VA, six for Ref2VA — and the node
only ever sees a text, which does not say which task wrote it. So a placement
naming a field this particular prompt has not got is not an error: the words go
to the top instead and the `findings` output says so.

A field that is there but says `N/A` gets a note of its own. Ref2VA prompts are
full of `non_diegetic_music: N/A`, and words put at the start of one leave a
field that names something and then says there is none of it. The node says so
and changes nothing else. Replacing the marker would have the field claim there
*is* music, called `ohwx face` — and the self-check reads that field; dropping
the words would be the node overruling a placement you asked for. Both are
decisions about your prompt, and this node does not make those.

One difference from the effect embeddings worth knowing, because the habit does
not carry over: ordinary words do not flatten the line breaks below them.
ComfyUI's tokenizer joins everything after an `embedding:` token onto one line,
but only after one. The top of the prompt is a safe place here.

#### It never takes anything out

This is where the node parts company with `MiniMax-H3 Effect Embeddings`, and the
reason is worth stating. Those ten tokens are strings known in advance, so the
node can cut its own back out before it puts them in — which is how unticking an
effect removes it. These words are yours. Cutting out a trigger spelled `detail`
would eat a hole in the prose it was sitting in.

So the rule here is presence, not removal:

- A word already in the text is **not added a second time** — matched on word
  boundaries and without regard for case, so `man` is not found inside `woman`
  and `Ohwx man` counts as `ohwx man`. It is reported on `findings`.
- **Unticking a row stops it being added. It does not take it away.** The word
  may well have come from the writer, and removing somebody else's word is not
  this node's call.

Running the same text through twice therefore changes nothing, which is what the
node needs in order to be safe at the tail of a graph that gets run again and
again.

#### `bypass` rather than Ctrl+B

Both of these nodes carry the pack's own `bypass` switch, with the same purple
badge in the title bar the writers have: click the badge or tick the widget, and
the prompt goes through untouched.

ComfyUI's own bypass is not the same thing on a node like this. It stands an
input in for each output by matching types, and there are three outputs here and
one socket to fill them from. Measured on the triggers node: `prompt` arrives,
`findings` is handed the whole prompt as though it were a finding, and `added`
loses its link altogether — which leaves whatever it fed with no input at all.
On the embeddings node `tokens` is an INT with no INT input to stand in for it,
so that link goes the same way. None of it is reported anywhere.

#### Put it at the very end of the graph

After the reducer and after `MiniMax-H3 Prompt Check`. Triggers add words, and
the check counts words against a ceiling — a node that lengthens the text after
it has been measured is fine, one that lengthens it before is arguing with the
measurement. And anything that rewrites the prompt downstream of this will drop
the words again, which is the failure this node exists to prevent.

#### What it does not promise

Only LoRAs that were trained with a trigger in the caption prefix. Sliders and
turbo accelerators need no word at all — they work by weight — and on most
machines those are the majority of what is installed. A trigger list will do
nothing for them, and that is not a fault in the list.

Nothing is read out of the LoRA files, and that is a finding rather than a
shortfall. Of forty-four adapters on the machine this was written on, one carried
`trigger_word`, one carried an empty `trigger_words`, and none carried
`ss_tag_frequency`. The pack can read a safetensors header already, so this is
not about cost: the data is not there. And an adapter often answers to several
words where the metadata has room for one. The list is yours to keep, which is
also what makes it worth keeping.

### MiniMax-H3 Reference Adapter

The writers take one reference per slot, and that is what makes the strip under
them work: every asset gets a square, a role and a switch. It is also what puts
them out of reach of a node that hands its references over *together* — an image
batch, a list from a directory loader, a bundle assembled by another pack. One
value holding many has no slot shape to go into.

This node is the join. Collected in on one socket, separated out on nine picture
sockets, three clip sockets and three sound sockets — which is what the writers
take, so the strip carries on working exactly as it did.

![The MiniMax-H3 Reference Adapter node. Two inputs down the left: items, unconnected, and bundle, wired to something off the edge of the frame. Sixteen outputs down the right: picture_1 to picture_9, video_1 to video_3, audio_1 to audio_3, and summary. The first three picture outputs run to three preview nodes showing three unrelated photographs. One widget, split_batches, reads true; under it a line of text reads "3 picture(s), 0 clip(s), 0 sound(s)". The node's run time is 0.014s](docs/node_ref_adapter.png)

*One wire in, three out — and thirteen outputs sitting idle, which is the ordinary case rather than a sign of something unfinished. The run cost 14 milliseconds because nothing here is loaded, decoded or described: the three photographs were carried through as they arrived. The line under the widget is the same text as the `summary` output, so what came in and where it went is readable off the node without wiring anything to a preview.*

| Input | What it is for |
| --- | --- |
| `items` | References arriving together. Takes any type, because the nodes that produce collections mostly declare none; what each item is gets worked out from the value itself rather than from the wire, and anything that is not an image, a clip or a sound is skipped and counted. |
| `bundle` | A reference bundle from another pack, if you have one. Its pictures, clips and sounds are read out ahead of anything on `items`. A clip that arrives as frames goes out as a clip, not as pictures, with its paired audio track inside it; a track not paired with a clip is a sound of its own. |
| `split_batches` | Whether an image batch becomes one reference per frame, or stays one reference made of several frames. |

That last switch is a real choice rather than a formality. Split, six frames are
six things the video reuses, each described and numbered on its own. Kept, they
are one thing seen six times, described once — which is what a clip is. Turn it
off when the batch is frames of a single shot.

Nine, three and three are what Ref2VA can hold, so that is the room there is.
Anything past it is reported on the `summary` output rather than dropped in
silence, along with anything skipped. Outputs left over hand on nothing, and a
writer skips those already, so there is no harm in wiring a socket that turns out
empty — plug in all nine and let the ones you did not fill sit idle.

It is a node of its own for a reason worth knowing, because it is not a choice.
Receiving a real ComfyUI list means declaring `is_input_list`, and that flag is
not per input: it rewrites the shape of *every* argument the node receives. On a
writer it would change how the prompt, the duration and the options arrive. So it
lives here, on a node that has nothing else to lose by it.

### MiniMax-H3 Reference Slots

`MiniMaxH3ReferenceToVideo` — ComfyUI's own conditioning node for H3's reference
mode — numbers its references by the socket they arrive on: the first picture
socket is `<Picture 1>`, the first sound socket `<Audio 1>`. The writers here
number theirs by the strip, which you can drag. Wire the same assets to both by
hand and the two orders agree until the first time a square is dragged; from
then on the prompt describes one voice, the video is given another, and nothing
on screen says so. Switching a reference off goes wrong the same way — gone from
the prompt, still in the video.

So every writer that takes references hands on what it numbered, on a last
output called `references`, and this node puts each reference on the socket
carrying its number.

![The MiniMax-H3 Reference Slots node between a Universal Writer and MiniMax H3 Reference to Video. The writer, set to Ref2VA, has four references connected and a strip reading pic 1 over ref_0, aud 1 over ref_2, pic 2 over ref_1 and aud 2 over ref_3; its last output, references, runs into the Reference Slots node. Down the right of that node are nineteen outputs — picture_1 to picture_9, video_1 to video_3, video_audio_1 to video_audio_3, audio_1 to audio_3 and summary — and every media output is wired straight across to the generator's inputs in the same order, ref_image_0 to ref_image_8, ref_video_0 to ref_video_2, ref_video_audio_0 to ref_video_audio_2 and ref_audio_0 to ref_audio_2. The soundtracks switch reads false; under it, "2 picture(s), 0 clip(s), 2 sound(s)" and "picture_1 ← ref_0". The writer ran in 0.011s and the slots node in 0.019s](docs/node_reference_slots.png)

*Wired once, all the way across, and never touched again: the arranging happens in the writer's strip. The squares read ref_0, ref_2, ref_1, ref_3 — so `picture_2` carries ref_1 and `audio_1` carries ref_2, whichever socket they were plugged into, and the generator numbers them exactly as the prompt does. The empty outputs cost nothing: the generator skips an empty socket. Nineteen milliseconds, because pictures and sounds are passed through as they are and there was no clip to decode.*

```
Universal Writer ─ references ─▶ Reference Slots ─ picture_1 ─▶ ref_image_0 ┐
                                                 ─ picture_2 ─▶ ref_image_1 ├ MiniMaxH3ReferenceToVideo
                                                 ─ audio_1   ─▶ ref_audio_0 ┘
```

Wire it straight across once, in order — `picture_N` to the N-th `ref_image`,
`video_N` to the N-th `ref_video`, `video_audio_N` to the N-th
`ref_video_audio`, `audio_N` to the N-th `ref_audio` — and do the arranging in
the strip from then on.

| Output | What it is |
| --- | --- |
| `picture_1` … `picture_9` | The pictures, in the writer's order. |
| `video_1` … `video_3` | The clips, as frames at 24 fps. |
| `video_audio_1` … `video_audio_3` | A clip's own sound, paired by number — when the writer counted it, or for every clip with `soundtracks` on. |
| `audio_1` … `audio_3` | The sounds, as they came. |
| `summary` | Which output holds what and which writer slot it came from, plus anything that did not fit. |

**A switched-off reference drops out on both sides.** It is not in `references`
at all, so its socket hands on nothing, and `MiniMaxH3ReferenceToVideo` skips an
empty socket and closes up its numbering around the gap — the same thing the
writer did with its labels.

**A subject goes after the pictures.** On the Universal Writer a picture badged
as a subject is written as `Subject N`, never as `<Picture N>`. The generator
has no subject socket and numbers it as a picture like any other, so it goes
after the last picture: the only numbers the prompt is not already using.

**Clips are decoded here, not in the writer.** A writer runs whether or not
anything is wired to `references`, and fifteen seconds of 1080p as float frames
is several gigabytes, so the clip travels as the VIDEO it arrived as and costs
nothing until this node runs. Here it is read at 24 fps — a 30 fps clip loses one
frame in five, a 12 fps one shows every frame twice — scaled to the canvas
`MiniMaxH3ReferenceToVideo` would give it anyway, and cut at fifteen seconds, the
longest reference it is meant for. A trimmed VIDEO is read over its trim. An
image batch the writer was told to read as a clip is frames already and goes
through as it is.

**A clip's sound goes across when the writer counted it.** The generator gives a
clip's sound an `<Audio N>` of its own, numbered ahead of every standalone sound.
Click a clip's label band in the writer's strip — or tick the speaker box on the
Universal Rewriter's `reference_video` row — and the writer numbers that sound
the same way, has it described or heard, and says so in `references`. This node
then puts it on the clip's `video_audio_N`, and the prompt and the video agree.
A clip with no sound track has nothing to put there, and the writer has already
said so and left it unnumbered.

**`soundtracks` gives every clip its sound, whatever the writer did.** It is off
by default. A clip the writer left silent then takes an `<Audio N>` the prompt
never gave it, so the prompt's `<Audio 1>` is no longer the sound on `audio_1`;
the summary says by how many the labels moved.

What each writer hands on is what its prompt names, and nothing it did not read:

| Node | `references` carries |
| --- | --- |
| Universal Writer | Every switched-on square, in strip order. Nothing on T2VA. |
| Prompt Rewriter Omni | Every switched-on square, in strip order. Nothing on T2AV. |
| Universal Rewriter | Nothing on the 27B tab or on T2VA. The frames the task reads on the 8B tab. On the Omni tab the same frames — or, on Ref2VA, every switched-on row in row order. |

The output is built from the live inputs every time, `bypass`, a library pick
and `repeat_last` included, and is never stored: the library and the node's
memory keep text, and a saved prompt comes back with the references connected
now.

> **Not the Reference Adapter.** That node feeds the writers: it sorts whatever
> arrives by kind and hands a clip on as a VIDEO, because a VIDEO is what a writer
> describes. This one feeds the generator, which takes a clip as frames and a
> clip's sound on a socket of its own. `references` is a type of its own for the
> same reason — plugged into a node that expects another pack's bundle, a clip
> would arrive as a VIDEO and fail inside the generator.

> **One chain this does not cover.** A Universal Writer with a block on
> `previous` counts that block's labels first, while its `references` holds its
> own references only and starts from one. Use one writer per shot when the
> output is going to the generator.

> **A RefMod stops it.** `MiniMaxH3ReferenceToVideo` encodes the pictures, clips
> and sounds it is given, and a [RefMod](#minimax-h3-refmod) is encoded already,
> so there is no socket to put one on. The node says so and names
> [MiniMax-H3 Reference Encode](#minimax-h3-reference-encode), which takes the
> same `references` and replaces both nodes.

### MiniMax-H3 RefMod

A RefMod is a reference that was encoded once and kept: the H3 VAE's latent of a
picture, a clip, a stack of photos or a voice, in a `.safetensors` file under
`ComfyUI/models/refmods`, with a header saying which it is. The format is
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)'s.
The RefMods people share on Hugging Face, that pack's single-file bundles and the
files [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder)
writes are all read here. Making and editing them is what those two packs are
for; this one only uses them.

This node picks one and hands it to a writer's strip. The template
**8 - RefMods to video** has the whole chain, from the file to the video.

| Input | What it is for |
| --- | --- |
| `refmod` | The entry, by its path under `models/refmods`. A look and a voice saved together — one bundle file, or a pair named `<name>_visual` and `<name>_audio` — are one entry. Press **R** in the graph to list files added since. |
| `strength` | How strongly the generator follows it. 1 is as saved. Lower mixes the latent toward a heavily blurred copy of itself, so colour, layout and the broad shape stay while the detail goes, and the pictures H3's text encoder is shown are weakened the same way. The writer still reads it at full strength. |
| `vae` | The H3 video VAE. Most files keep no pictures of themselves — none of the ones on Hugging Face do — and a writer has to see a reference to describe it, so the look is decoded here, once. |
| `audio_vae` | The H3 audio VAE: the same for a voice, so a writer can hear it. |

| Output | What it is |
| --- | --- |
| `visual` | The picture, clip or stack of photos. Empty for a voice alone. |
| `audio` | The voice. Empty when the entry has none. |

Each output goes into a square of its own on the
[Universal Writer](#minimax-h3-universal-writer) or the
[Prompt Rewriter Omni](#minimax-h3-prompt-rewriter-omni-sees-and-hears), where it
is marked `mod`:

![The Universal Writer with a RefMod's two outputs on ref_0 and ref_1, and ref_2 empty and unticked. Its strip holds two squares marked mod: a green vid over ref_0 showing a decoded photo of the character with a play mark and a 1, and a purple aud over ref_1 with a speaker, the entry's name Yoda, a waveform and a 1, each with an "+ instr" band. Under them the line of help and the task switch with Ref2VA lit](docs/refmod_support_universal_writer.png)

![The Prompt Rewriter Omni with the same two RefMod outputs on ref_0 and ref_1: a green vid square and a purple aud square named Yoda, both marked mod and numbered 1, above the line "drag to reorder - that order numbers the labels - click a square to switch it off - a clip's label: its sound". Below them the five tasks with REF2AV lit, eight aspect-ratio rectangles drawn to proportion with 16:9 chosen, and a duration of 10.0](docs/refmod_support_omni_rewriter.png)

From there:

- **It is called what its file says it is.** A clip or a stack of photos is a
  `<Video N>` and a voice an `<Audio N>` whatever the badge reads, because that
  is what the generator calls it. A single picture can still be made a subject,
  but not a clip, and there is no sound of its own to switch on.
- **It is described from its pictures,** like any other reference. What the
  file says about it travels as a hint after the role's own question: a stack is
  described as one subject seen in separate photographs rather than as a
  sequence of events, a saved subject name is given to use, and a saved
  appearance or voice description is offered to keep to where the pictures
  agree. The Universal Writer writes a voice nobody decoded from its saved
  description, when it has one. The Omni adapter hears the sound itself, so it
  refuses such a voice and says to connect `audio_vae`.
- **The Universal Writer's writer is told what the file says,** in the
  reference block, ahead of the caption: that a stack is photos of one subject
  rather than footage, what it was saved as a reference for — a voice being the
  voice alone, not what plays behind it — and, when a look and the voice saved
  with it are both on the strip, which label holds the other half:
  `Audio 1: (saved as a reference for a voice, the voice alone and not what
  plays behind it; the voice of the subject in Video 1) ...`. The captioner
  reads the two apart, so neither caption says they belong together, and a
  voice saved over music is otherwise taken for the music and borrowed for the
  score. The generator is still handed the whole recording, music and all: the
  prompt only stops asking for more of it.
- **The RefMod itself is handed on** in `references`, latent and all, to
  [MiniMax-H3 Reference Encode](#minimax-h3-reference-encode).

A stack is decoded one photo at a time, because each photo was encoded on its
own, and a clip whole. Either way it happens once per pick: ComfyUI keeps the
node's output, and the pictures with it.

A RefMod is not free at sampling time. The node's caption says what it costs,
and some of the shared ones are heavy: a ten-photo stack at 1024x1824 is 18,240
tokens riding through every sampling step, as many as eighteen reference
pictures at 1344x768 or a two-and-a-half-second reference clip. Lowering
`strength` changes how closely it is followed, not what it costs.

### MiniMax-H3 Reference Encode

`MiniMaxH3ReferenceToVideo` encodes the references it is given. A RefMod is
encoded already: the generator needs its latent as a reference block of its own,
and H3's text encoder still needs its pictures. So this node does what
[Reference Slots](#minimax-h3-reference-slots) and `MiniMaxH3ReferenceToVideo` do
together, and for a RefMod the one thing neither can.

```
Universal Writer ─ references ───────▶ references ┐
                 ─ rewritten_prompt ─▶ prompt     ├ MiniMax-H3 Reference Encode ─ positive ─▶ sampler
clip, vae, audio_vae ────────────────────────────┘                              ─ latent   ─▶ sampler
```

Its settings are `MiniMaxH3ReferenceToVideo`'s — `width`, `height`, `length`,
`ref_image_size` — with the writer's `references` and the `soundtracks` switch
from Reference Slots. It gives `positive`, the empty `latent` to sample into,
and a `summary` of what went in, in order, and what it costs.

**Order is the whole contract.** H3's DiT has no index for a reference: a block
is matched to its label by where it sits. So everything goes in the order
`MiniMaxH3ReferenceToVideo` uses — every picture, then every clip with its own
sound just ahead of it, then every sound — and a RefMod takes the place the
writer numbered. A picture, a clip or a sound is prepared with that node's own
helpers, sizing and cutting included, so a strip without a RefMod in it encodes
to the same conditioning it did through Reference Slots.

**What the text encoder is shown of a RefMod.** A picture as it is. A clip at two
frames a second, as for any reference clip. A stack every fourth photo, two to a
block — the default of the pack that stores such pictures, and close to what two
a second of the same latent played as a clip would show. A file that stores its
pictures is shown those; one that does not is decoded here with `vae` when its
RefMod node had none.

`vae` and `audio_vae` keep `MiniMaxH3ReferenceToVideo`'s rule for ordinary
references: without them those reach the text encoder only, and the summary says
how many did. A RefMod reaches the generator either way.

### MiniMax-H3 Prompt Presets

A thousand finished MiniMax-H3 prompts ship inside the pack, and this node hands
one of them on. Press **Pick a preset** and the browser opens: twenty shooting
styles, twenty subjects, the shape of the frame, the number of shots and whether
anyone speaks, all as tags that narrow the thousand down, with a search box over
the words themselves. Each row carries the frame of the clip the prompt was
written for — click it and the clip plays.

They are finished T2VA prompts in the format the writers here produce, the same
three labelled fields, which makes them useful in two directions. Wire `prompt`
at the generator and it is used as it stands; wire it into a writer's own
`prompt` and it is the starting point for a rewrite. That choice is a wire rather
than a widget, which is why this is a node of its own.

![The MiniMax-H3 Prompt Presets node beside a Show Any node. Eight outputs run down the right of the preset node, prompt first and source last. A preset widget reads 000014; under it two buttons, "Pick a preset" and "Save to the library"; under those the frame of the clip — a motorcycle courier on a sun-bleached desert highway — beside the whole prompt, its three labelled fields legible in full. The prompt output is wired to the Show Any node, which prints the same three fields as plain text. The run took 0.030s](docs/node_prompt_presets.png)

*The preset is on the node rather than behind a dialog: the frame of the clip it was written for and the whole of its text are on the face of it before anything is run. The run itself took thirty milliseconds and loaded nothing — what came out of `prompt` is what the node had been showing all along.*

| Output | What it is |
| --- | --- |
| `prompt` | The whole thing, three labelled fields, exactly as a writer here would have produced it. |
| `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music` | The same three, separately. |
| `seconds`, `width`, `height` | The clip the prompt was written for. Every one of them is about five seconds, and the shot times inside the text are written against that. |
| `source` | Its number, both addresses the clip can be watched at, and who is owed the credit — for wiring into a text preview when a workflow is going somewhere else. |

**Nothing is downloaded, at install or during a run.** The prompts, their tags
and one 256-pixel frame each come to 6 MB inside the pack; they are read off disk
and unpacked the first time something asks, so a session that never opens the
browser pays nothing, and a run takes single-digit milliseconds. The one thing
that reaches the network is the clip in the preview, and only when somebody
clicks a frame: huggingface.co first, hf-mirror.com when that does not answer,
which is the address that answers from mainland China. Without a connection the
frames are still there and only the video is missing.

**Save to the library** puts a copy in one of your own prompt sets, through the
same editor the library uses — name, description, groups, and the self-check
running as you type. From that point it is an ordinary record: editable, filed
under your own groups, and available to every writer through the library window.
The copy says in its description where it came from. The bundled preset is not
touched by any of it.

A caution worth stating once: these are T2VA prompts for a five-second clip.
Nothing stops you handing one to a task with references or to a much longer
video, and nothing will complain, but the text describes neither.

The prompts and the clips they describe are ostris's work, carried here with the
author's permission and with credit —
[ostris/minimax_h3_1k](https://huggingface.co/datasets/ostris/minimax_h3_1k). The
prompts are his own and effectively MIT; the clips are MiniMax-H3 output, whose
licence forbids using it to train models, and the frames cut from them here carry
that same restriction. The shooting-style and subject tags are the
[H3 Atlas](https://cohub.live/baize/video-altas/w/h3-atlas)'s reading of that
collection. All thousand pass [the self-check](#the-answer-is-checked) with
nothing to report, which is one way of saying what the format of a good H3 prompt
actually looks like.

### The duration widget

Every node in the pack that writes or reads a prompt has one, and it means the
same thing on all nine: how long the target clip is meant to be, in seconds. It
is a hint rather than a setting — it reaches the model as one line of text,
`duration: 10s`, and decides how many shots the rewrite plans and how it paces
them. How long the clip really turns out to be is settled downstream, by whatever
samples it.

**It takes `FLOAT` and `INT` on the same socket.** The widget itself is a float,
in tenths, because the cut times a rewrite writes are quoted to a thousandth —
`At 00:02.378, the camera cuts to` — and asking for 7.5 seconds is as ordinary as
asking for 8. But most graphs carry a length as a whole number, so the socket
takes either: an `INT` primitive, a node that counts frames, or a `FLOAT` off a
maths node all wire straight in, with no converter between. The two Universal
nodes draw it as a slider and the rest as a number field you can type into; the
range and the menu are the same either way.

**Its upper end is yours to move.** Right-click the node — on the classic canvas,
right-click the widget itself and the entry comes first — and there is a
`duration` submenu with two items:

| Item | What it does |
| --- | --- |
| **Default value** | Puts the widget back to the ten seconds it starts at. |
| **Longest offered…** | Asks for a number and makes it the widget's top end. It is rounded to a tenth of a second and kept in the node, so it travels with the workflow. |

![The classic-canvas context menu over a Universal Writer, headed MiniMaxH3UniversalWriter with `duration` as its first entry, highlighted, and its submenu open to the right holding two items — “Default value (10 s)” and “Longest offered (now 30 s)…”. Below the highlighted entry the node's ordinary menu continues: Set, Get, two greyed-out rgthree queue entries, Run, Reload Node and “Favorite Widget: duration”. Behind the menu the node shows its aspect-ratio chips with 16:9 lit and the duration slider reading 10.0](docs/duration_widget.png)

*Right-clicking the widget itself puts `duration` at the top of the menu, as
here; right-clicking anywhere else on the node reaches the same submenu further
down. Both numbers in the submenu are the node's own and are saved with the
workflow.*

Thirty seconds until you change it, and the server accepts up to six hundred. Two
numbers rather than one because a widget's range is fixed when the node is
declared and no single range suits every graph: MiniMax's own guide is written
around clips of a few seconds, while the stretched pipelines the community has
built run well past that. So the server takes anything sane, and the widget spans
what *you* work with — a slider reaching ten minutes is a slider that cannot be
nudged to 9.

That number is the node's own `max_duration` property, so the Properties Panel
reaches the same setting; a workflow saved before this existed simply gets the
thirty.

### Repeating the last answer

Every writer, rewriter and captioner has a `repeat_last` switch. Turn it on and
the node hands back the answer it produced last instead of running the model
again — with nothing kept yet it runs once, keeps what it wrote and says so, and
from then on returns that same text for as long as the switch is on, whatever
else you change. That is the point of it: a fifty-second rewrite is not something
to pay for twice while you wire up everything downstream of it.

ComfyUI's own cache cannot do this. A node's cache key is its class, its
`IS_CHANGED` value and every input it received, so editing the prompt is exactly
what drops the entry — and `IS_CHANGED` can only add invalidation, never mask it.
So the answers are kept in a store of the pack's own, one per node.

That store lives in memory for the ComfyUI session and nowhere else: it is not
written to disk, it does not travel with the workflow, and a restart empties it.
Every run says where its answer came from in three places — the caption under the
node, the ComfyUI console, and the switch's own tooltip, which carries the time
it was kept, its length and the opening of the text. When the inputs no longer
match the ones that produced it, all three say so rather than pretending nothing
changed. `bypass` still wins over it.

**And what is kept can be edited.** `Edit the last prompt` opens the answer the
node is holding in the same editor [the library](#the-prompt-library) uses: the
prompt in a box you can write in, read as you type by the rules the run was read
by. Saving splits the fields out of the new text again, so the section outputs
stay in step with the prose; everything the node kept past its fields belongs to
the run rather than to the prose and stays as it was. Nothing reaches disk — this
is the session store, and an edit worth keeping is saved to the library
afterwards.

The node hands the edited text on the next time `repeat_last` asks it to. An edit
changes none of the node's inputs, so each writer reports what it is holding as
its `IS_CHANGED`, the same way it reports the library record it is pointed at;
without that ComfyUI would go on serving the answer from before the edit.

**Whether the edit reaches the output is a separate question**, and the editor
answers it before you type. A saved prompt wins over the node's own answer, so
while one is chosen the edit waits behind it — clear the choice, or edit that
record instead. With `repeat_last` off the next run simply writes a new answer
over the edit, and on a greedy seed with unchanged inputs that new answer is
byte for byte the old one, which looks exactly like being ignored; so saving
switches `repeat_last` on when nothing else is in the way, the same way choosing
a library record does.

On the two caption nodes it is the caption that is kept, not the assembled block:
the numbering belongs to the chain, so the line is written again around whatever
arrives on `previous`, and an asset connected since is described for real.

The node's own last answer is what the switch hands back by default, and
[the prompt library](#the-prompt-library) is how you point it at something else
instead. One switch either way: `repeat_last` decides whether a kept prompt is
handed on at all, the library window decides which one.

### Unloading a kept model

`keep_model_loaded` leaves the writer in memory after a run. That is what you
want while iterating on a prompt, and what is in the way once the text is right
and the card is wanted for video. Turning the switch off does not give the card
back cheaply: the switch is an input, a changed input is a new run, and the node
writes the same text a second time before it unloads. Three things let the model
go without running it again.

**`Unload model`** is a button on every node that has the switch. It unloads
whatever the pack is holding, at once. Pressed while a node is still writing, it
waits for that run and unloads after it — whatever `keep_model_loaded` says,
because the press came later.

**ComfyUI's own `Unload Models`** reaches the pack's models too. They are not
ComfyUI models and its model management does not know them, so the pack hooks
itself into the unload. The same goes for everything else that asks ComfyUI to
unload all models: the `/free` route, the out-of-memory handler, and the
clean-up nodes of other packs — put one between the writer and the sampler, and
the writer is gone before the video model loads. It cuts the other way as well.
A sampler that clears the card for itself, as several wrappers do, takes a kept
writer along with the rest; and under `--disable-smart-memory` ComfyUI unloads
everything after every workflow run, so a kept model lasts for that run and no
longer.

**`repeat_last` on with `keep_model_loaded` off** unloads as well. A node that
hands back a kept prompt does not run the model, but it still reads the switch:
the text comes back from the node's own store, the model is unloaded, and
nothing is written twice.

All three let go of everything the pack holds inside the ComfyUI process — a
Transformers checkpoint or a GGUF on `llama-cpp-python` — and not only the model
of the node the button was pressed on. A GGUF run by the llama.cpp binaries was
never kept: the model leaves with the process.

### The answer is checked

Every fresh answer is read back against the rules of MiniMax's own writing
guides before it leaves the node — a self-check, silent when it finds nothing.
When it does find something, it says so in three places: a toast in the ComfyUI
window with the first few findings, because the caption under a node is easy
to miss from the other end of the graph; the caption itself, which holds the
full list above the answer; and the console, for runs nobody is watching. What
it looks at:

- **Shot structure.** `[Shot 1]` opens the description and carries no cut time;
  later shots are numbered in order, each with an `At MM:SS.mmm` timestamp that
  is later than the one before it and inside the requested duration.
- **Dialogue markup.** `<d>` and `</d>` come in pairs, and every block starts
  with its `[Language]` tag.
- **Reference tags, both ways.** A tag the task cannot carry — `<Video 1>` in a
  frame task, any tag at all in T2VA — is flagged, and so is the opposite: a
  reference that reached the node but is never cited, which the generator still
  receives with no say in what it is for. The nodes that see their references
  check the numbers against what is actually connected; the text-only writers
  skip that half rather than guess.
- **Full-reference bookkeeping.** Every `<Subject N>` that
  `subject_definitions` introduces owes `retention_analysis` a line, and the
  guide's 350–500 words for `detailed_description` is noted when missed.
- **The alignment line.** I2VA, FL2VA and L2VA open with a fixed sentence
  telling H3 where the reference frames land; its absence is worth knowing
  about before a render finds out.
- **Effect embedding tokens.** The one rule here that is not MiniMax's: a
  capital `Embedding:`, a token stuck to the word in front of it, or a full stop
  on the end of a name is a token ComfyUI will drop in silence. It is in the
  self-check because the failure has the same shape as all the others — nothing
  in the output says so — and because that is where you look when
  [an effect did nothing](#minimax-h3-effect-embeddings). A name that is not one
  of the ten is a note rather than a warning: it may well be your own textual
  inversion.

![A warning toast titled "Self-check: MiniMax-H3 Universal Writer", listing three warnings marked with an exclamation point — 5 fields missing from the answer, naming subject_definitions, summary, retention_analysis, overall_soundscape and non_diegetic_music with the advice to lower the temperature or try a larger writer model; the description has no Shot 1, shots being how H3 reads structure; Picture 1 and Picture 2 connected but never cited, the model still receiving them with no say in what they are for — and one note marked with a dash: detailed_description is 1286 words where the guide suggests 350–500](docs/self_check_alert.png)

*One deliberately bad run, read back. The `!` lines are warnings — things H3
will likely misread; the `-` line is a note — something the guide merely
suggests. The toast shows the first few findings; the full list sits in the
caption under the node and in the console, where it outlasts the toast however
long the toast is given.*

A refusal is the one thing the nodes do enforce, and it is said the same way: the
sentence appears as a toast under **Stopped** before the exception goes up, so a
run that halts does not look like it failed for no reason.

Findings are said, never enforced: the answer ships exactly as written, because
the model is sometimes right to bend a rule and only you know whether this is
that time. Warnings (`!`) are things H3 will likely misread; notes (`-`) are
things the guide merely suggests. A prompt handed on from the library is not
re-checked — it was checked when it was written, and the caption stays with the
record's name instead.

**How much is said** is `self_check` on the Options node: `warnings and notes`
is everything; `warnings only` drops the guide's softer suggestions, such as the
350–500 words, and keeps what H3 will likely misread; `off` says nothing. The
reading itself always happens — it is regexes over text already in memory and
costs nothing — so this decides what is reported, not what is looked at. An
unreadable setting reports everything: a stale workflow should leave the check
louder than intended, never silent.

The same toast carries the nodes' own warnings, under **Heads-up** instead of
**Self-check**: a reference the chosen task will not read, a saved prompt whose
references no longer match what the node is shown, a captioner that came back
with nothing for a connected asset. Those are about your wiring rather than the
model's prose, so `self_check` does not cover them and they are always said.
They were always in the caption and the console; now they are also impossible to
miss.

**Some of it happens before the run.** A task that cannot use what is connected
has always been refused before any weights move — `I2VA` with two pictures, a
clip wired into a frame task. `Ref2VA` now refuses the same way when the strip
holds more than H3 can take: nine pictures, three clips, three sounds. It used to
be described first and flagged afterwards, which cost a full captioning pass and
a rewrite to learn something countable in advance. Subjects are counted apart
from pictures, so a badge is often the whole fix.

### Acting on what it found

`fix_once` on the Options node lets the nodes do something about the findings
rather than only report them. It is **off by default**, and off means a promise:
the answer you get is exactly what the model wrote, with nothing edited on the
way out.

Turned on, two things happen. **The alignment line goes back on** where it is
missing — that sentence is fixed text the node already formats when it builds
the prompt, so a model that dropped it has not made a judgement worth
respecting. No model runs, nothing can regress. Then, for the mechanical
findings — a cut time past the end, an unbalanced `<d>`, a tag pointing at a
reference that is not there — **the writer is asked once more**, with those
findings folded into the prompt as rules to obey.

Once, never a loop. A model that ignored a rule twice will ignore it a third
time, and every attempt costs a full generation. Three things keep the single
retry safe:

- **It is refused on a hopeless answer.** Half the fields missing means the
  model is not holding the format at all, and a second pass will not change
  that; the node says to try a larger writer instead of spending another
  minute.
- **The constraints travel inside the prompt**, not as a second conversational
  turn. That keeps the single-turn shape the LoRAs were trained on, so the
  re-run behaves the same on a trained adapter as on a guided model.
- **The second answer is kept only if it is better** — fewer warnings, or the
  same warnings and fewer findings overall. A tie keeps the first. So the worst
  case is a minute spent, never a worse prompt.

Whatever it did is said on the node: the line restored, the re-run and how much
it gained, or that the first answer stood.

### The prompt library

`repeat_last` holds one answer per node and forgets it when ComfyUI restarts. The
library is the other half: prompts you name and keep, in a JSON file under the
ComfyUI user directory, available to every workflow and every node.

**Saving.** The **Save the last prompt** button on any writer or rewriter opens a
box asking for a name, a description and any number of groups — the groups already
in the file are offered as chips, and a new one is a word and Enter. What gets
saved is the run itself: the text, the sections after it, the task, the model, the
ratio, the duration, the seed, and a 50x50 thumbnail of every reference the node
was shown.

Those thumbnails are taken **during the run**, which is the only moment they can
be. By the time the box opens the node has returned strings and the tensors are
gone, so the picture, the first frame of the clip and the measurements are captured
as the answer is produced and travel with it. What is measured is what genuinely
exists: an image has a size; a clip that arrived as a real `VIDEO` has frames,
seconds and a frame rate, while one that arrived as a batch of images has a frame
count and nothing else, because a batch has no container to ask; a sound has a
duration, a rate and a channel count. References are labelled by position —
`ref1-image`, `ref2-audio` — rather than by file name, which a node cannot see
anyway.

**Using one.** The **Prompt library** button opens the list: this node's own last
answer first, then everything in the file, newest first. Filter by group with the
chips, or type in the search box to keep only the records whose name, description,
prompt, groups, references or settings contain what you typed. **Use** points the
node at a record; **Copy** puts the prompt on the clipboard without pointing the
node at anything; **Edit** opens it for changing; **Delete** removes it from the
file for good, after asking. The list answers the keyboard too: the arrow keys walk
it without leaving the search box, Enter uses the lit record, and a double click on
a card does the same. It opens scrolled to the record this node is pointed at.

![The prompt library window, titled “Prompt library” over the line “Hand a saved prompt straight to this node’s output. No model is loaded.”: a search box across the top reading “Search names, descriptions, prompts, references” with the file dropdown beside it on `global`, and four group chips under it — bakery, dinosaur, fish, joke. First the node’s own Last Prompt row, its Use and Copy greyed out beneath “Nothing kept yet. Run this node once and its answer is what gets saved.” Then two saved records: “Дино и пеламида”, three thumbnails down its left and the line Ref2VA · 16:9 · 10s · 3 images + 1 audio · joke, dinosaur, fish above its description and the opening of its subject_definitions; and “Дино”, one thumbnail and the line I2VA · 16:9 · 15s · 1 image · joke, bakery, dinosaur, its card tinted blue because that is the record this node is pointed at. Use, Copy and Delete run down the right of every row, and Write a new one and Close sit at the foot of the window](docs/prompt_library.png)

*Everything on a card is searchable — the name, the description, the prompt, the
groups, the settings and the references — which is what keeps a file of a hundred
records usable. The line under the name ends with what the record was written for,
`3 images + 1 audio`, in the same words the node uses when it warns that what it is
being shown no longer matches; the thumbnails beside it say which pictures those
were. The tinted card is the current choice. The Last Prompt row sits above the
file because it is not in it — it is this session’s own answer, which is why it
has no **Delete**.*

The library, the editor, the **Model list** and the bundled prompts size like any
window: drag an edge or a corner, or fill the browser tab with the button in the top
corner, and each kind remembers its own size. The footer says how many records the
file holds, or how many a search has left, and what the last edit or delete did.

A chosen prompt reaches `rewritten_prompt` and the section outputs exactly as a
fresh one would — no model is loaded, nothing is generated, and the run takes about
a tenth of a second. A record written by another kind of node works too: when the
classes match the outputs come back verbatim, and otherwise the text is split into
sections by the node reading it, the same way it splits an answer it wrote itself.

**Editing one.** **Edit** opens the record: its name, its description, its groups,
and the prompt itself in a box you can write in. What *produced* the record — the
writer, the settings, the duration, the reference thumbnails — is shown but not
editable, and does not change. That half is the account of a run, and a card that
misreported its own provenance would be worse than no card.

The prompt is checked as you type, by the same rules the writers run over a model's
answer and out of the same module, and against this record's own task, duration and
references: a cut time that has drifted past the end of the video is named while you
are still looking at it. That is the reason to edit here rather than in the JSON
file — the file will take anything.

Escape and **Cancel** ask before they throw away anything typed, here and in the
other boxes that take typing — **Save the last prompt** and the entry form of the
**Model list**. With nothing typed they just close.

![The edit window, titled “Edit a saved prompt” over the line “The prompt itself, and what the card says about it. What produced this record -- the writer, the settings, the references -- stays as it was.”: a Name box reading “Дино и пеламида (changed)”, a Description under it, then four group chips with joke, dinosaur and fish lit and bakery not, an empty “New group, then Enter” box, and the prompt itself in a monospaced box open at subject_definitions and summary. Beneath it the self-check — “Self-check: 2 warning(s), 1 note(s)” over two amber lines, one saying &lt;Video 1&gt; is cited but no video reached this node and one saying &lt;Picture 2&gt; and &lt;Picture 3&gt; are connected but never cited, and a grey one saying detailed_description is 147 words where the guide suggests 350-500. At the foot the record’s own account, MiniMaxH3UniversalWriter · Ref2VA · 16:9 · 10s · 3 images + 1 audio · saved 29.08.2026, 15:07:10 · edited 01.09.2026, 11:07:04, with the four reference thumbnails below it, and Cancel and Save changes in the corner](docs/prompt_library_edit.png)

*The findings are read from the record rather than guessed at: the task is Ref2VA, the duration ten seconds, and the references are the four at the foot — which is how it knows that `<Video 1>` is named in a prompt no video ever reached, and that two of the connected pictures are never cited. Amber is a warning, grey a note the guide merely suggests. `self_check` on the Options node governs what the nodes say during a run; the editor always shows both, since you asked for them by opening the box. Everything below the line is the half that cannot be edited.*

Changing the text drops the writer's own split of that answer into fields, because
the split was made from the text as it was. Without it the node reading the record
splits the text itself, which is exactly what it already does with a prompt written
by a different kind of node, so the section outputs stay in step with what you
wrote. Everything else about the record survives, and its card gains the time of
the edit beside the time it was saved.

A node already pointed at the record hands on the new text at its next run.
ComfyUI decides whether to run a node at all from its inputs, and an edit changes
none of them — the pick is still the same id in the same file — so each writer
reports the record's own content as its `IS_CHANGED` and the run happens. Nothing
else about caching moves: with no saved prompt in play the value is constant, and
a record renamed rather than rewritten does not make every node holding it run
again.

**A record with references is pinned to them.** A T2VA prompt is self-contained
and travels anywhere. A prompt for one of the frame tasks or for Ref2VA is not:
it names its references and describes them — `<Subject 1>` *is* the blue dinosaur
with the textured scaly skin — so giving that prompt a different set of assets
hands the generator a description of something that is not in front of it. The
thumbnails on the card are how you see what a record was written for, and the
node checks as well: when the kinds and counts of what it is being shown no
longer match what the record was written for, it says so on the node and in the
console before handing the prompt on. It says it rather than refusing — reusing a
description as a template is a legitimate thing to do, and only you know whether
that is what you meant. What no check can catch is one picture swapped for
another of the same kind; that is what the thumbnail is for.

**One switch decides, the window decides what.** Choosing a record switches
`repeat_last` on, because that switch is what hands a kept prompt on at all.
Turning it off gives the node back to the model and leaves the choice waiting;
**Write a new one** does both, forgetting the choice as well. The button says which
it is — `Library: Storm at sea`, or the same with `(repeat_last is off)` after it —
and the badge beside the node title reads `REPEAT` or `LIBRARY`.

The choice itself lives in a hidden widget, so it is saved with the workflow and
reaches an API run: a graph reopened tomorrow returns the same prompt. If the
record has been deleted since, the run stops and says so rather than quietly
writing a new one — the graph asked for a particular prompt.

**Sets.** `prompt_file` on the Options node names the file the nodes wired to it
save into and list, `global` unless you say otherwise, and **New prompt file**
beside it makes another. One file is one working set, which is the cheapest way to
keep a project's prompts apart from everything else. They live in

```text
ComfyUI/user/minimax_h3_rewriter/prompts/
```

as plain JSON, so they can be edited by hand, copied between machines or deleted.
A record with one thumbnail and a full-length prompt is about 11 KB.

### The guides are fetched, not bundled

The two guides are not shipped inside this pack. They are MiniMax's
Documentation, and the MiniMax H3 Community License grants the right to
redistribute the Materials only within its "Applicable Territory" — worldwide
*excluding* the European Union, the United Kingdom, the Republic of Korea and the
United States — and only alongside a copy of the agreement and a NOTICE file. A
node pack on a registry cannot honour a territory boundary, so each installation
fetches its own copy directly from MiniMax on first use: 16 KB and 24 KB, once,
under the same `auto_download` switch as everything else.

They land in

```text
ComfyUI/user/minimax_h3_rewriter/guides/
```

which the **Open guide folder** button opens. Editing a copy changes the system
prompt, and a fetch never overwrites a file that is already there — trimming the
guide is the cheapest way to fit a small model's context.

### The model list

On a node with one model field, **Model list** opens its catalog. On a node with
several model fields, open the catalog using the button for that field.
Use **Add a model**, **Edit**, or **Delete** to manage catalog entries.
**Check it** inspects an entry before you download the model. The catalog is saved in

```text
ComfyUI/user/minimax_h3_rewriter/models.json
```

The editor and node dropdowns show **On disk**, **Download needed**, or **Files
missing**. A multimodal GGUF needs both the model and its `mmproj` to show as
**On disk**. In **Model list**, use the search field or **On disk only** to narrow the
list. Click the star beside a model to put it in **Favorites** and bring it to the
top of the corresponding node dropdowns.

**Use this** selects a model on the node that opened the list. On nodes with
several model fields, buttons such as **Captioner models** and **Writer models**
open the corresponding tab. **Use in writer_model** sets the named field.
The folder badge shows the local model directory or download destination, such as
`LLM/quant`.

Turn on **Search nested folders** to include all subfolders of the configured
model directories. This setting and your favorites are saved on the ComfyUI
server. Click **Rescan folders** after copying models to the server.

The file is seeded from the packaged copy on first use, so updating the node pack
never overwrites your edits. **Open models.json** in the window's footer still
opens that file in your desktop's JSON editor — on the machine running ComfyUI,
which is not necessarily the one looking at the browser tab — for the two things
the window deliberately leaves alone: the `adapters` sections, and a path to a
network share.

The window shows only the lists the node it was opened from actually reads,
because the adapters take different architectures and an entry from one list
will not load in another's node. The Universal Rewriter therefore opens on three
tabs and the reference captioner on one. Expand **Model requirements** to see
the architecture, model shape, and projector requirements for the open list.
**Found in model folders** shows additional models discovered in your ComfyUI
directories and Ollama stores. These models also appear in the node dropdown.
When a catalog entry points to a discovered file, its **On disk** row represents
that file in both places.

![Model list on the Guided writers tab with Search nested folders enabled. Installed models show On disk badges, relative folders, Use in writer_model buttons, and favorite stars.](docs/model_list_dialog.png)

The footer counts the entries in the open list and says what the last save,
delete or restore did — including how many nodes in this graph moved to an entry's
new label, or still hold the name of one that is gone. Those keep the name, so that
a run says the entry is missing rather than picking another model for them.

**Check it** answers as much as can be answered without moving any weights. A
file already on this machine is read outright, so it reports the architecture and
the shape, whether the model carries a chat template, and whether the projector
carries vision and audio:

```text
- 'Qwen2.5-Omni-7B-Q4_K_M.gguf' is a 'qwen2vl' model, 28 blocks of width 3584. That fits.
- It carries its own chat template.
- 'mmproj-Qwen2.5-Omni-7B-Q8_0.gguf' carries the vision and audio encoder.
```

The chat template matters more than it looks. The 27B rewriter and the guided
writers build their prompt from the model's own template, so a GGUF without one —
typically the base conversion of a model rather than the instruct one — reads as a
perfectly good model and then stops on the first run. There Check it refuses it,
and renders the template once on a system and a user turn, so a template this
pack cannot render is refused too. The captioners and the 8B and Omni rewriters
leave the template to `llama-mtmd-cli`, which falls back to ChatML without one, so
there a missing template is only a warning.

Anything only on Hugging Face is asked what its metadata can say: a transformers
repository is judged from its 4 KB `config.json`, and a GGUF repository is asked
whether the files you named are in it — which is what catches a typo that would
otherwise surface as a download failing minutes in, and fills in `download_gb`
for you while it is there. A GGUF's shape and chat template are in the file
itself, so for one still on the Hub those wait for the first run — or for Check
it again once the file is here.

Leave `file` empty and **Check it** answers a different question: what is in that
repository? It lists the GGUF files as buttons, smallest first. A click puts the
file into the form, and a name in the style of the packaged entries —
`Qwen2.5-Omni-7B Q4_K_M` — while you have not typed one of your own. A list that
runs a pair shows the projectors in a row of their own, for `mmproj`, and fills one
in by itself when the repository holds only one; once both halves are there the
check runs and adds up the download. A model split into parts,
`-00001-of-00003.gguf`, is counted and left out, because an entry names one file
and only that file is downloaded. A folder on this machine in `repo` is listed the
same way, and a file name the repository does not have brings up the same buttons.

Two things the window will not do. It refuses a **network path** in `repo`,
`file` or `mmproj`: it is reachable over the ComfyUI API, which has no CSRF
token and is routinely served on `--listen`, and merely looking at a UNC path is
an authentication attempt against the host it names. A path typed into
`models.json` by hand is still unrestricted — that file does not travel. And it
refuses to write at all while the file does not parse, since saving over it
would replace your own entries with the packaged list; it shows the parse error
and every button that would write is dead, leaving **Open models.json**. What it
lists in that state is the packaged copy, because that is what the dropdowns are
offering too until the file parses again.

Editing an entry's download size, VRAM note or note changes what the dropdown
reads, and saved workflows remember that string — but an entry is found by its
name, so a workflow holding the old label still runs it and moves to the new one
when it is opened. Renaming the entry is different, since the name is what it is
found by: the form says so before it commits, the graph you have open is moved
across for you, and other workflows lose the choice.

Deleting an entry leaves the nodes that used it as they are: a run stops and says
the entry is gone, rather than picking another model and starting a download
nobody asked for. The one exception is an entry whose file is already here —
its `on disk:` row is back in the dropdown, and the nodes move to it.

The file holds five lists with the same fields. **`models`** feeds the LoRA
rewriter and has to be Qwen3.6-27B:

```json
{
  "name": "Qwen3.6-27B FP8",
  "repo": "Qwen/Qwen3.6-27B-FP8",
  "download_gb": 28.8,
  "vram": "~29 GB, no extra package needed"
}
```

**`models_8b`** feeds the 8B rewriter, has to be Qwen3-VL-8B-Instruct, and needs
an `mmproj` beside the model — being multimodal, it is two files:

```json
{
  "name": "Qwen3-VL-8B-Instruct GGUF Q4_K_M",
  "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
  "file": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
  "mmproj": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
  "format": "gguf",
  "download_gb": 5.4,
  "vram": "~9 GB with the adapter"
}
```

**`models_omni`** feeds the Omni rewriter, has to be Qwen2.5-Omni-7B, and takes
the same two files — with the difference that its projector has to carry an audio
encoder, or the adapter attaches to a model that cannot hear:

```json
{
  "name": "Qwen2.5-Omni-7B GGUF Q4_K_M",
  "repo": "ggml-org/Qwen2.5-Omni-7B-GGUF",
  "file": "Qwen2.5-Omni-7B-Q4_K_M.gguf",
  "mmproj": "mmproj-Qwen2.5-Omni-7B-Q8_0.gguf",
  "format": "gguf",
  "download_gb": 6.2,
  "vram": "~9 GB with the adapter and a 12k context"
}
```

Three lists rather than more entries in `models`, because an entry from any of
them would fail to load in either other node — different architecture, different
adapter. The three adapters live apart for the same reason, under `adapters`,
`adapters_8b` and `adapters_omni`.

**`writers`** feeds the writer nodes and can be anything, as long as it is a GGUF
language model with a chat template:

```json
{
  "name": "Qwen3.5-4B",
  "repo": "unsloth/Qwen3.5-4B-GGUF",
  "file": "Qwen3.5-4B-Q4_K_M.gguf",
  "format": "gguf",
  "download_gb": 2.6,
  "vram": "~5 GB with the guide in context"
}
```

**`captioners`** feeds the reference caption node and needs one extra field,
`mmproj` — a multimodal model is two files and both come from the same
conversion:

```json
{
  "name": "Qwen2.5-Omni-3B",
  "repo": "ggml-org/Qwen2.5-Omni-3B-GGUF",
  "file": "Qwen2.5-Omni-3B-Q4_K_M.gguf",
  "mmproj": "mmproj-Qwen2.5-Omni-3B-Q8_0.gguf",
  "format": "gguf",
  "download_gb": 3.4
}
```

#### Pointing an entry at a file you already have

`repo` need not be a Hugging Face id. Give it a **folder on this machine** and the
file is read straight out of it, with nothing downloaded and nothing copied:

```json
{
  "name": "Gemma 4 26B",
  "repo": "X:/models/gemma-4-26B-A4B-it-UD-Q8_K_XL",
  "file": "gemma-4-26B.gguf",
  "format": "gguf"
}
```

Or put the whole path in `file` and leave `repo` out — both forms work, in all
three sections, because both are what people actually type. Two things to watch:

- **Backslashes must be doubled in JSON**, or written as forward slashes.
  `"X:\Programs\..."` is not valid JSON at all — `\P` is not an escape — and one
  bad character makes the *whole file* unreadable, not just that entry. Windows
  accepts `X:/Programs/...` everywhere, so that is the easier habit.
- A path that does not exist is **reported, not downloaded around**. The node
  names the file it could not find and stops.

For a whole folder of GGUFs you keep elsewhere, an entry each is the long way
round: point ComfyUI's `extra_model_paths.yaml` at it under the key `LLM` and
every file in it is offered automatically, with `on disk:` in front of the name.
If that folder comes first, new downloads go into it; a model already in any
`LLM` folder is found where it is and not fetched again.

Add an entry, **refresh the browser tab** — ComfyUI need not restart — and it is
in the dropdown. Keep `name` stable: it is what a saved workflow finds the entry
by. A node whose stored choice has vanished stops before anything runs and says
so by name, instead of silently picking something else.

#### When the list itself is broken

A syntax error in `models.json` used to be the quietest failure in the pack: the
parse threw, the packaged defaults were served instead, and the dropdown looked
ordinary. An edit that never took was indistinguishable from an edit that did
nothing.

Now the first entry in every model dropdown says so, with the line and column:

```text
!! models.json is not valid JSON — Invalid \escape: line 3 column 30 (char 46) — showing the packaged list instead
```

The rest of the list is still there and ComfyUI still runs; picking that first
entry and hitting Run repeats the message and names the file to fix. Fix it,
refresh the tab, and it goes away.

#### Updates add new entries without touching yours

Your copy is never overwritten, but models added to the pack after you installed
**are merged in** — otherwise "we will not touch your list" quietly becomes "you
will never see anything new", with nothing anywhere to say the node knew about
more.

The rule is set algebra, not a version check. Beside the lists your file records
`seed_offered`: every name the packaged list has ever put in front of this
installation. An update then adds

```text
names in the pack  −  names in your file  −  names you were already offered
```

so an entry you deleted stays deleted, one you renamed is not duplicated, and a
genuinely new one arrives. Deleting a name from `seed_offered` offers that entry
again on the next start, and **Restore the packaged entries** in the model-list
window is that same edit as a button: it drops the record for one list, and the
next read brings back every packaged entry the list is missing. Entries you
wrote yourself are untouched by it.

One exception, once: a file written before this existed has no record of what it
was offered, so on the first update everything missing comes back — including
anything you had deleted by hand. The previous file is kept beside it as
`models.json.bak`, and from then on your deletions stick. That backup records
the merge and nothing else: edits you make in the model-list window do not
spend it, or two clicks would be all it took to lose the copy it exists for.

Merges are logged to the ComfyUI console by name. A file the node cannot parse is
left exactly as it is and the packaged list is used for that session, so a stray
comma costs you a restart, not your edits.

### Models you already pulled for Ollama

If you run Ollama, the writers and captioners this pack wants are on your disk
already, and downloading the same quant a second time would be silly. They are
offered in the `writer_model` and `caption_model` dropdowns, prefixed `ollama:`
and named the way `ollama list` names them:

```text
ollama: qwen3:8b [qwen3, 4.7 GB]
ollama: moondream:latest [+mmproj, vision, 1.7 GB]
```

Nothing is copied, converted or downloaded, and Ollama itself does not have to be
running: what it pulls is a plain GGUF, and llama.cpp reads it where it lies.
This is the file on disk, not Ollama's API — the server can be stopped, disabled
or uninstalled and the models stay usable.

A multimodal model appears in both lists: as a writer without its projector, and
as a captioner with it. The pairing comes out of the model's own manifest, where
the two files are named together, so it is certain in a way that comparing file
names in a folder is not.

Three places are looked at, and Ollama's layout is the same on every platform:

```text
$OLLAMA_MODELS                     # if you set it
~/.ollama/models                   # the usual install, Windows included
/usr/share/ollama/.ollama/models   # Linux, installed as a service
```

A store anywhere else — inside WSL, in a container, on a drive the server knows
nothing about — is named by hand, in `models.json`

```json
"ollama_stores": ["\\\\wsl$\\Ubuntu\\usr\\share\\ollama\\.ollama\\models"]
```

or in `MINIMAX_H3_OLLAMA_MODELS`, which takes several paths separated the way
`PATH` separates them. Those are not looked for automatically on purpose:
reaching `\\wsl$\...` **starts a stopped WSL distribution**, and these lists are
rebuilt every time a dropdown is filled. Opening a ComfyUI tab should not boot a
virtual machine.

`ollama rm` takes a model out of the dropdowns the way deleting any other file
does. What your workflow saves is the label rather than the path, so re-pulling
the same tag leaves it working — the store is content-addressed, and the same
files land back where they were. The label carries the size, though, and a
different quantisation is a different size, so re-pulling `qwen3:8b` at another
quant comes back as a choice you have to pick again.

## Where the weights go

```text
ComfyUI/models/LLM/
├── Qwen3.6-27B/                          # base model, ~52 GB
├── MiniMax-H3-Prompt-Rewriter-LoRA/      # adapter, ~3.5 GB
├── Qwen3.5-9B-Q4_K_M.gguf                # a writer model, one file
└── Qwen2.5-Omni-3B/                      # a captioner: model + mmproj together

ComfyUI/user/minimax_h3_rewriter/
├── models.json                           # your model list
├── guides/                               # the two writing guides, 40 KB
└── runtime/                              # llama.cpp binaries, if fetched
```

**Already downloaded the LoRA?** Point the `adapter` widget at that folder
directly (an absolute path is accepted) and nothing is fetched again.

## Using a model you already have

52 GB is a lot to ask, so the node looks for the base model before offering to
download it. The `model` dropdown lists every Qwen3.6-27B found in

- all directories ComfyUI registers for `LLM`, `transformers`, `diffusers`,
  `text_encoders` and `clip` — including anything mapped in through
  `extra_model_paths.yaml`, and
- the Hugging Face cache (`HF_HOME` / `HF_HUB_CACHE` / `~/.cache/huggingface/hub`),

so a copy pulled earlier by any other tool is reused instead of downloaded twice.
Only directories whose weights are actually present are listed; a cache entry
holding nothing but `config.json` is not offered.

### Smaller repackings

Any repository you put in the model list is checked from a 4 KB `config.json`
**before** a single weight moves. Every repacking keeps
the same fingerprint (`model_type qwen3_5`, hidden 5120, 64 layers, vocab 248320),
so what actually decides the outcome is the quantization runtime:

| Build | Download | Runtime package | LoRA attaches? | Verdict |
|---|---|---|---|---|
| `quant_method: bitsandbytes` (nf4 repack) | **~17 GB** | `bitsandbytes`, already required | yes | **works as shipped — the low-resource route** |
| `Qwen/Qwen3.6-27B` (bf16) | 52 GB | none; `bitsandbytes` for `nf4`/`int8` | yes | **works as shipped** |
| `Qwen/Qwen3.6-27B-FP8` | 29 GB | `kernels`, plus a kernel fetched from the Hub | yes | advanced only; needs >29 GB of VRAM |
| `quant_method: awq` | ~19 GB | `autoawq` | yes | works after installing it |
| `quant_method: gptq` | ~19 GB | `gptqmodel` | yes | works after installing it |
| `quant_method: compressed-tensors` | ~19 GB | `compressed-tensors` | **no** | not supported |
| `quant_method: modelopt` (NVFP4) | ~20 GB | `nvidia-modelopt` | **no** | not supported |

PEFT ships LoRA dispatchers for bitsandbytes, AWQ, GPTQ, HQQ, EETQ, AQLM and
torchao layers, and plain `nn.Linear` covers bf16/fp16 and FP8. For
compressed-tensors and modelopt there is no dispatcher, so the adapter cannot be
attached at all — no package will fix that.

> A repository's **name** is not its format: `cyankiwi/Qwen3.6-27B-AWQ-INT4` and
> `unsloth/Qwen3.6-27B-NVFP4` are both `compressed-tensors`, so neither can take
> the LoRA. The node reports the real `quant_method` and the exact reason before
> downloading anything.

### If the node says a package is missing

Nothing in the recommended routes needs one. bf16, the nf4 repacks and the
`nf4`/`int8` options all run on packages this pack already declares, so a
missing-package message only appears for a build you went looking for.

When it does appear it names the package **and the command for the interpreter
that is actually running ComfyUI** — `pip install kernels` typed into an
ordinary terminal installs into whichever Python is on PATH, which on a portable
install is never the right one, and the node goes on refusing:

```
This base model cannot run the prompt-rewriter LoRA.
  - the 'fp8' checkpoint needs the 'kernels' package, which is not installed in
    this Python environment. Install it with:
      "…\python_embeded\python.exe" -m pip install kernels
    Note: installing it is only half of it: the FP8 matmul is a Triton kernel
    that transformers then downloads from 'kernels-community/finegrained-fp8'
    on the first generation, and that needs a build matching this torch and
    CUDA version.
```

Run the line as printed and restart ComfyUI. The pack installs nothing into your
environment on its own, and never will: a node that silently pip-installs is a
node that can break an unrelated part of ComfyUI while you watch a progress bar.

## Smallest download without any extra install

A `bitsandbytes` nf4 repack of Qwen3.6-27B is **~17 GB to download and ~16 GB of
VRAM**, and it needs nothing this node does not already require. That is the
route to point people at: same code path as the official checkpoint, no new
dependency, a third of the download.

Those repacks are third-party — the node verifies the architecture from
`config.json` before fetching anything, which proves the shape is right, not that
the uploader is trustworthy. Judge that yourself, or make your own repack once
from the official weights.

## GGUF — smaller still, and nothing to install

Pick a `[gguf]` entry from the model list and the rewriter runs under llama.cpp
instead of Transformers. **No pip install is involved.** If `llama-cpp-python`
happens to be in ComfyUI's environment the node uses it; if it is not, it runs an
llama.cpp the machine already has, and failing that fetches the official binaries
(~34 MB, or ~511 MB where `llama_backend` resolves to CUDA — see the table below)
into `ComfyUI/user/minimax_h3_rewriter/runtime/` and runs
`llama-completion` as a subprocess. Same download switch as the weights:
`auto_download`.

The subprocess reloads the model on every run, which costs nothing in practice —
the node's default is `keep_model_loaded = False`, because the card is needed
for video generation the moment the rewrite finishes, and the in-process backend
already unloads after every run too. What the binary backend genuinely cannot do
is honour `keep_model_loaded = True`. Two things come free with it: VRAM is
returned by the operating system rather than by a deallocator, and a llama.cpp
crash takes down a child process instead of ComfyUI and its queue.

Two options in the options node control this, and they answer different
questions. **`gguf_runtime`** picks *what runs the model*:

| `gguf_runtime` | Meaning |
|---|---|
| `auto` | llama-cpp-python if it is importable, the binaries otherwise |
| `llama-cpp-python` | force the wheel; fails with a clear message if it is absent, rather than quietly using something else |
| `llama.cpp` | force the binaries, even when a wheel is installed — the way out when the installed wheel is broken |

Only `llama-cpp-python` can honour `keep_model_loaded`; the binaries hand the
model back to the operating system when the subprocess exits.

**`llama_backend`** then picks *which official build to fetch*, and applies only
when the binaries are in use:

| `llama_backend` | Download | Notes |
|---|---|---|
| `auto` | 34 or 511 MB | **the fastest build this machine can run, not the smallest.** CUDA on Windows with an NVIDIA card of compute capability 8.6, 8.9, 12.0 or 12.1 — the archive carries native SASS for exactly those and no PTX to fall back on — and Vulkan everywhere else |
| `vulkan` | 34 MB | NVIDIA, AMD and Intel alike; about half the CUDA throughput. Pick it explicitly to keep the download small on a card `auto` would send to CUDA |
| `cuda` | 511 MB | ~2× faster on NVIDIA; **Windows only** — upstream publishes no Linux CUDA build, so on Linux you compile one yourself and the node runs it |
| `cpu` | 17 MB | no GPU at all |

**An llama.cpp you already have is run as it is.** Before fetching anything the
node looks in four places, in order: the path in `MINIMAX_H3_LLAMA_BIN`, the
path written in `ComfyUI/user/minimax_h3_rewriter/llama_bin.txt`, the copy it has
already unpacked **for the backend in use**, then `PATH`. A build you compiled
yourself therefore needs no setting at all — put its `build/bin` on `PATH`, or
name it outright:

```sh
export MINIMAX_H3_LLAMA_BIN=/opt/llama.cpp/build/bin   # the folder, or the binary in it
```

**On a server, prefer the file.** An export reaches a server started from that
same shell and nothing else — a systemd unit, a container entrypoint or a
launcher script hands the process an environment of its own, and never reads
your `~/.bashrc`. One line in `llama_bin.txt` is read by the node itself and
does not care who started ComfyUI:

```sh
echo /opt/llama.cpp/build/bin > ~/comfy/ComfyUI/user/minimax_h3_rewriter/llama_bin.txt
```

`llama_backend` then stops mattering: it picks which archive to download, not
what an existing binary was compiled against. This is the road to a CUDA
llama.cpp on Linux — build it once with `-DGGML_CUDA=ON`, name it here, and
`device = cuda:0` does what it says. The caption nodes look for
`llama-mtmd-cli` beside it; a build without that target sends the node back to
the archive for that one job.

**The unpacked copy is per backend**, in `runtime/<release>-<backend>`, so an
archive already sitting in `runtime/b10310-vulkan` does not answer for a run that
resolved to CUDA — the node fetches that one too. Changing `llama_backend` is
therefore a second download, and unpacking a build by hand means putting it in
the folder named for the backend you will actually run.

A path pointing at nothing is an error rather than a quiet download, so a typo
says so instead of costing half a gigabyte. And when nothing is found anywhere,
the refusal prints what the ComfyUI **process** actually had — the variable, the
file, the folder and its `PATH` — instead of repeating advice you may already
have taken in a shell it never saw.

> **Why not the `llama-cpp-python` CUDA wheels.** Both current ones fail on
> ordinary consumer hardware, in two unrelated ways:
>
> | Wheel | Build flags | What happens |
> |---|---|---|
> | `v0.3.34-cu130` | `AVX512 = 1`, `ARCHS = 750..900` | weights load, then `llama_init_from_model` dies with `0xC000001D` — no consumer Intel 12th–14th gen chip has AVX-512 |
> | `v0.3.34-cu132` | `AVX512 = 0`, `ARCHS = 750..900` | reaches the first kernel, then `the provided PTX was compiled with an unsupported toolchain` — no `sm_120` in the list, so an RTX 50-series card falls back to JIT, which a driver older than the build's toolkit refuses |
> | `v0.3.34-vulkan` | `AVX512 = 0`, no arch list | works, and picks up `NV_coopmat2` where the driver offers it |
>
> The official release archives have neither problem. They carry **14** CPU
> backend variants and choose one at run time, which is why the same model runs
> fine under `llama-cli` on the machine where the cu130 wheel dies. And their
> CUDA archive carries native SASS with no PTX at all —
> `cuobjdump --list-elf` reports `sm_86 sm_89 sm_120a sm_121a` — so the driver
> is never asked to compile anything.
>
> If you want the in-process backend anyway, the Vulkan wheel is the one that
> works: `pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-vulkan/llama_cpp_python-0.3.34-py3-none-win_amd64.whl`

| Base quant | Download | VRAM with the adapter |
|---|---|---|
| `Q4_K_M` | 15.7 GB | ~19 GB |
| `IQ4_XS` | 14.4 GB | ~18 GB |
| `UD-Q3_K_XL` | 13.5 GB | ~17 GB |
| `UD-IQ2_M` | 10.1 GB | ~13 GB, noticeably lower fidelity |

Lower `gpu_layers` in the options node to fit a smaller card, at the cost of
speed. With `Q4_K_M` fully offloaded to a high-end consumer NVIDIA card, CUDA
generates at roughly **50 tok/s** with the adapter against 78 tok/s without it —
that ~35% is llama.cpp doing the adapter's matmuls — and Vulkan at roughly half
the CUDA figure.

> **A smaller Qwen3.5 is not a substitute.** Qwen3.5-9B carries the same
> `general.architecture = qwen35` in its header, so it looks like a match and
> the model list will show it — but it has 32 blocks of width 4096 where the
> adapter needs 64 of 5120. llama.cpp refuses to attach the LoRA
> (`tensor 'blk.0.attn_gate.weight' has incorrect shape`) and the run fails.
> The node checks those two header numbers first and says so before anything is
> downloaded, and labels such files `(wrong size for the adapter)` in the
> dropdown. If you see a 9B producing a plausible-looking rewrite, it is running
> **without** the adapter: the format comes from the system prompt, not from the
> LoRA. That is not a dead end — it is precisely what the
> [writer nodes](#minimax-h3-prompt-writer-t2vai2vafl2val2va) do on purpose, with
> the full guide in the prompt instead of seven lines.

> **Neither is a later Qwen.** Qwen3.8-27B is `qwen35` with the same 64 blocks of
> width 5120, so the shape check passes it and llama.cpp attaches the LoRA
> without complaint — to weights it was never trained on. The adapter was
> trained on [Qwen3.6-27B](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA)
> and nothing else. The node reads the names a GGUF gives itself and warns, on
> the node and in Check it, when none of them is Qwen3.6-27B. It does not
> refuse, because those names are optional and written by whoever quantised the
> file. An **MTP build** of Qwen3.6-27B is fine: it reports 65 blocks because it
> carries a draft head for speculative decoding, and the node counts the head
> out — llama.cpp never loads it here, and the adapter has nothing for it.

The GGUF route uses a **converted** adapter, not the PEFT one — F16 and Q8_0 of
[the 27B LoRA](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF) and of
[the 8B one](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-8B-GGUF). It is fetched
without asking; to use one of your own, drop the `.gguf` into `models/LLM` and
pick it from the options node's `adapter` list, or set `adapters.gguf.repo` in
the model list. The prompt is built from the GGUF's own chat template with
`enable_thinking=False`, and the result is byte-identical to what
`transformers.apply_chat_template` produces — the model sees exactly the text the
LoRA was trained on.

When the checkpoint carries its own quantization, the `quantization` widget is
ignored — bitsandbytes is not stacked on top of AWQ or FP8.

### Progress on the node

Downloads, weight loading, and token generation all report onto the node itself
through ComfyUI's own progress channels — a bar plus a caption with the current
file, transferred size, speed and ETA. No custom frontend extension is involved,
so nothing breaks when the ComfyUI frontend updates.

### Environment variables

| Variable | Effect |
|---|---|
| `HF_TOKEN` | Access token for gated or private repositories |
| `HF_ENDPOINT` | Mirror to download from instead of `huggingface.co` |
| `MINIMAX_H3_LLAMA_BIN` | An llama.cpp already on this machine: the executable, or the folder holding it |
| `MINIMAX_H3_MTMD_SERVER` | `auto`, `never` or `always`: whether a caption run holds one model open instead of loading it per reference |

### Languages

Set ComfyUI to Russian (Settings → Comfy → Locale) and the nodes come up in
Russian: what each node is, and every widget's tooltip — the long explanations
are the point of the translation, since they are what you actually read.

**Widget names stay English on purpose.** `repeat_last`, `fix_once`,
`prompt_file` are identifiers, not prose: they are the keys in the workflow
JSON and in an API call, they are what this README and the issue tracker call
things, and the tooltips name widgets by them. Translating the label would
leave every one of those references pointing at nothing, so a Russian tooltip
can say `repeat_last` and it matches what is written on the node.

One tooltip is left in English deliberately — `device` on the Options node,
which is built at run time and ends with the machine's own GPU list. A static
file cannot carry that without shipping one person's hardware to everyone.

Translations live in `locales/<lang>/nodeDefs.json` at the root of the pack,
which is where ComfyUI looks. Adding a language is adding a folder; nothing in
the Python needs to know. Two things help keep one honest:

```text
python tools/locales.py report ru     what is missing, and what has gone stale
python tools/locales.py fill ru       add the missing keys, in English, to translate over
```

`report` asks a running ComfyUI what the nodes actually are, so it cannot drift
from the code the way a checked-in copy would; `fill` never overwrites a key
that already has a translation. The test suite covers the rest — that a file is
valid JSON, that it names only this pack's nodes, and that every key is one
ComfyUI actually reads, since a misspelled key fails silently and simply never
appears.

**A node already on the canvas keeps the title it was created with.** Switching
language does not rename it, because that title is saved in the workflow. Add
the node afresh to see the translated name.

## Notes

- **Speed.** Qwen3.6-27B is a hybrid model: 48 of its 64 layers use linear
  attention. Without `flash-linear-attention` and `causal-conv1d` installed,
  Transformers falls back to a slower pure-PyTorch path and says so in the
  console. The fallback is correct, just slower; both packages are optional and
  awkward to build on Windows.
- **Determinism.** With `greedy` on, the same prompt, resolution, duration and
  seed produce the same rewrite, and ComfyUI caches the node accordingly.
- **Interruption.** Cancelling a run stops both a download and a generation in
  progress; a partial download resumes on the next run.
- **Format on the writer nodes is followed, not guaranteed.** A general model is
  obeying instructions rather than reproducing a distribution it was trained on.
  Keep `greedy` on — small models drift out of the format as soon as they sample —
  and if a field is missing the node returns everything it did get and names the
  gap on the node rather than failing.
- **A chat template that rejects a system role still works.** Gemma's calls
  `raise_exception` on one; the guide is then folded into the first user turn, as
  those models' own cards prescribe.
- **Listing your GGUF models is nearly free.** Building the dropdown needs six
  values from the first few kilobytes of each file, but `gguf.GGUFReader`
  materialises the whole header the moment it opens one — including
  `tokenizer.ggml.tokens`, a quarter of a million strings. Ten files in a model
  folder cost 31 seconds, paid the first time ComfyUI answered `/object_info`.
  The header is now walked directly and skipped past, which is the same six
  values in **0.4 s**. A half-downloaded file is still refused, by checking its
  tensor offsets against its size rather than by failing to map them.
- **The captioner writes media to a temporary folder** — an IMAGE and a VIDEO
  become PNGs, an AUDIO a 16-bit WAV written with the standard library — and
  removes the folder afterwards, including when the child process crashes.
- **A VIDEO is sampled here, not passed to `llama-mtmd-cli --video`.** That flag
  feeds the file to `ffprobe` through *stdin*, and when the MP4 carries its `moov`
  atom at the front — which is what "faststart" means, and what ComfyUI, phones
  and most of the web produce — ffprobe has what it needs after a few kilobytes
  and exits without reading the rest. llama.cpp is still writing the remaining
  megabytes into that pipe and blocks there for good: no output, no error, no end.
  Same clip with `moov` moved to the end runs in six seconds. Frames are decoded
  in-process instead, which also makes `max_frames` mean something for a VIDEO.
- The rewrite may add details a short prompt never stated. Review it before
  generating when identity, dialogue, timing or composition must be exact.

## Credits

The model work is entirely [LightX2V](https://github.com/ModelTC/LightX2V)'s —
this repository only wires their adapter into ComfyUI. If you find it useful,
star **[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V)**, where the
MiniMax-H3 inference support and future rewriter tasks (FL2VA, Ref2VA) are
maintained.

| Component | Source |
|---|---|
| LoRA adapter | [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA) |
| LoRA adapter, 8B | [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B) |
| Base language model | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Base language model, 8B | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Video/audio generator | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| Prompt-writing guides | [MiniMaxAI/MiniMax-H3 `docs/`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/docs) — fetched at run time, see above |
| Inference framework | [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) |
| Bundled prompts, and the clips they describe | [ostris/minimax_h3_1k](https://huggingface.co/datasets/ostris/minimax_h3_1k) — used with the author's permission; the prompts are effectively MIT, the clips and the frames cut from them are MiniMax-H3 output and may not be used to train models |
| Shooting-style and subject tags for them | [H3 Atlas](https://cohub.live/baize/video-altas/w/h3-atlas) |
| RefMod format, and how its references are presented | [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) by Luisa (luisacaotica), MIT; the pictures stored for the text encoder, and the every-fourth-photo default, from [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder). Read by this pack's own code, none of theirs copied |

The prompt templates in `minimax_h3_rewriter/prompt_template.py` and
`prompt_template_8b.py` are reproduced byte-for-byte from their adapter
repositories; changing them degrades the rewrite.

Use of MiniMax-H3 is governed by the licence and acceptable-use terms in the
[official MiniMax-H3 repository](https://huggingface.co/MiniMaxAI/MiniMax-H3).

Thanks to [AxiomGraph](https://www.youtube.com/@AxiomGraph) for the first video
review of these nodes —
[«Stop Struggling With MiniMax H3 Prompts. Do This Instead.»](https://www.youtube.com/watch?v=h3rZTIRB_G8) —
and for the community workflows linked above.

Thanks to [ЭйАй Генератьон](https://www.youtube.com/@AyiTheDeer) for a second and more detailed one, in
Russian, which places these nodes among the other MiniMax-H3 packs rather than
on their own —
[«MiniMax H3 - как создать непрерывное длинное видео. Обзор наборов нод для ComfyUI + рерайтер промта»](https://www.youtube.com/watch?v=PZd9fWX15VA).

## Licence

MIT — see [LICENSE](LICENSE). This covers the ComfyUI integration code only; the
model weights carry their own licences.


---

## FILE: README_RU.md (https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/blob/HEAD/README_RU.md)

# MiniMax-H3 для ComfyUI: создание и переписывание промтов

Ноды ComfyUI для создания, переписывания и проверки структурированных промтов
к видео и звуку [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
Можно использовать LoRA LightX2V, [полные Qwen-чекпойнты SFT/GGUF](docs/qwen_rewriter_RU.md),
локальные модели с официальным гайдом MiniMax или [API Universal Writer](docs/api_writer_RU.md)
с настроенным локальным сервером либо удалённым провайдером.

В пак также входят расширение промтов для MiniMax-H3-Image и P-H3, кондиционирование референсов,
сохранённые RefMod, сокращение промтов, пресеты и библиотеки промтов и моделей.
Локальные писатели работают на вашей машине; API-запросы отправляются
выбранному провайдеру.

[English version](README.md) · [中文版](README_ZH.md) · [История изменений](CHANGELOG_RU.md)

<p align="center">
  <a href="https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI?display_name=tag"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <a href="https://registry.comfy.org/publishers/darkil/nodes/minimax-h3-prompt-rewriter"><img alt="ComfyUI Registry" src="https://img.shields.io/badge/ComfyUI-Registry-1B98E0"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA"><img alt="Hugging Face" src="docs/badges/hf-lora.svg"></a>
  <a href="https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF"><img alt="GGUF adapter, 27B" src="docs/badges/gguf-27b.svg"></a>
  <a href="https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-8B-GGUF"><img alt="GGUF adapter, 8B" src="docs/badges/gguf-8b.svg"></a>
  <a href="https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF"><img alt="GGUF adapter, Omni" src="docs/badges/gguf-omni.svg"></a>
  <a href="https://www.youtube.com/watch?v=h3rZTIRB_G8"><img alt="Video review, in English" src="https://img.shields.io/badge/YouTube-review%20(EN)-FF0000?logo=youtube&logoColor=white"></a>
  <a href="https://www.youtube.com/watch?v=PZd9fWX15VA"><img alt="Video review, in Russian" src="https://img.shields.io/badge/YouTube-review%20(RU)-FF0000?logo=youtube&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA"><img alt="LightX2V SFT: Qwen3.5 9B Ref2VA" src="docs/badges/hf-sft-9b-ref2va.svg"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA"><img alt="LightX2V SFT: Qwen3.5 9B FL2VA" src="docs/badges/hf-sft-9b-fl2va.svg"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA"><img alt="LightX2V SFT: Qwen3.8 27B Ref2VA" src="docs/badges/hf-sft-27b-ref2va.svg"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA"><img alt="LightX2V SFT: Qwen3.8 27B FL2VA" src="docs/badges/hf-sft-27b-fl2va.svg"></a>
</p>

![Нода ререйтера в ComfyUI между нодой настроек и просмотрщиком текста: в самой ноде короткий промпт, справа — структурированное покадровое описание, звуковой ландшафт и музыка](docs/node_preview.png)

Промпт можно писать на любом языке, который понимает базовая модель, — результат
всё равно придёт на английском, как того и ждёт MiniMax-H3.

```text
«Рыжая лиса идёт по заснеженному лесу на рассвете.»  +  16:9  +  15 с
                              │
                              ▼
              Писатель / рерайтер промтов
                              │
                              ▼
   integrated_multimodal_description: [Shot 1] ... [Shot 2] 0:06 ...
   overall_soundscape: ...
   non_diegetic_music: ...
                              │
                              ▼
              MiniMax-H3: видео + синхронный звук
```

Выберите семейство нод под вашу модель и референсы:

| Семейство нод | Модель или сервис | Референсы | Задачи |
|---|---|---|---|
| LoRA Rewriter 27B | Qwen3.6-27B + LoRA LightX2V | Текстовые описания | T2VA |
| LoRA Rewriter 8B | Qwen3-VL-8B-Instruct + LoRA | Сама читает изображения первого/последнего кадра | T2VA, I2VA, FL2VA, L2VA |
| LoRA Rewriter Omni | Qwen2.5-Omni-7B + LoRA | Изображения; на пути GGUF также видео и аудио | T2VA, I2VA, FL2VA, L2VA, Ref2VA |
| [Qwen Rewriter](docs/qwen_rewriter_RU.md) | Полные LightX2V Qwen3.5 9B / Qwen3.8 27B, SFT или GGUF | Сама читает изображения и выборку видеокадров; звук задаётся текстовым описанием | FL2VA-чекпойнт: T2VA, I2VA, FL2VA, L2VA; Ref2VA-чекпойнт: Ref2VA |
| Писатели по гайду | Локальная GGUF, умеющая следовать инструкциям | Описания медиа от капшенеров | T2VA, I2VA, FL2VA, L2VA, Ref2VA |
| [API Universal Writer](docs/api_writer_RU.md) | Настроенный локальный сервер или удалённый провайдер | Изображения, выборка видеокадров и аудио по возможностям модели/сервера; есть режим описаний | T2VA, I2VA, FL2VA, L2VA, Ref2VA |

Все три LoRA-варианта доступны и в [Universal Rewriter](#minimax-h3-universal-rewriter):
адаптер выбирается вкладкой. Для писателей по гайду
[Reference Caption](#minimax-h3-reference-caption) и
[Multi Reference Caption](#minimax-h3-multi-reference-caption) превращают медиа в текст.
[Universal Writer](#minimax-h3-universal-writer) совмещает описание референсов и
написание промта в одной ноде; порядок референсов задаётся перетаскиванием чипов.

А когда короткий путь к хорошему промту — чужой промт:
[Prompt Presets](#minimax-h3-prompt-presets) отдаёт один из тысячи готовых
промтов MiniMax-H3, которые лежат внутри пака, — с отбором по манере съёмки и
теме, с кадром того клипа, под который промт написан, и с самим клипом в одном
клике. Никакая модель для этого не грузится и ничего не скачивается.

Если у вас 8 ГБ видеопамяти, переходите сразу к [нодам писателя](#minimax-h3-prompt-writer-t2vai2vafl2val2va).

## Содержание

- [Что потребуется до установки](#что-потребуется-до-установки)
- [Установка](#установка)
  - [Готовые воркфлоу](#готовые-воркфлоу)
- [Ноды](#ноды)
  - [MiniMax-H3 Prompt Rewriter](#minimax-h3-prompt-rewriter)
  - [MiniMax-H3 Prompt Rewriter 8B (sees frames)](#minimax-h3-prompt-rewriter-8b-sees-frames)
  - [MiniMax-H3 Prompt Rewriter Omni (sees and hears)](#minimax-h3-prompt-rewriter-omni-sees-and-hears)
  - [MiniMax-H3 Qwen Rewriter](#minimax-h3-qwen-rewriter)
  - [MiniMax-H3 Universal Rewriter](#minimax-h3-universal-rewriter)
  - [MiniMax-H3 Rewriter Options](#minimax-h3-rewriter-options)
  - [MiniMax-H3 Prompt Writer (T2VA/I2VA/FL2VA/L2VA)](#minimax-h3-prompt-writer-t2vai2vafl2val2va)
  - [MiniMax-H3 Prompt Writer (Ref2VA)](#minimax-h3-prompt-writer-ref2va)
  - [MiniMax-H3 Universal Writer](#minimax-h3-universal-writer)
  - [MiniMax-H3 Image Prompt Writer](#minimax-h3-image-prompt-writer)
  - [P-H3 Prompt Writer (изображения)](#p-h3-prompt-writer-изображения)
  - [MiniMax-H3 Reference Caption](#minimax-h3-reference-caption)
  - [MiniMax-H3 Multi Reference Caption](#minimax-h3-multi-reference-caption)
  - [Описание моделью, которая у ComfyUI уже загружена](#описание-моделью-которая-у-comfyui-уже-загружена)
  - [Каптионер грузится один раз, а не по разу на референс](#каптионер-грузится-один-раз-а-не-по-разу-на-референс)
  - [MiniMax-H3 Guide Prompt (any LLM)](#minimax-h3-guide-prompt-any-llm)
  - [MiniMax-H3 Prompt Check](#minimax-h3-prompt-check)
  - [MiniMax-H3 Prompt Reducer](#minimax-h3-prompt-reducer)
  - [MiniMax-H3 Reduce Prompt (any LLM)](#minimax-h3-reduce-prompt-any-llm)
  - [MiniMax-H3 Effect Embeddings](#minimax-h3-effect-embeddings)
  - [MiniMax-H3 LoRA Triggers](#minimax-h3-lora-triggers)
  - [MiniMax-H3 Reference Adapter](#minimax-h3-reference-adapter)
  - [MiniMax-H3 Reference Slots](#minimax-h3-reference-slots)
  - [MiniMax-H3 RefMod](#minimax-h3-refmod)
  - [MiniMax-H3 Reference Encode](#minimax-h3-reference-encode)
  - [MiniMax-H3 Prompt Presets](#minimax-h3-prompt-presets)
  - [Виджет duration](#виджет-duration)
  - [Повтор последнего ответа](#повтор-последнего-ответа)
  - [Выгрузка оставленной модели](#выгрузка-оставленной-модели)
  - [Ответ проверяется](#ответ-проверяется)
  - [Действие по находкам](#действие-по-находкам)
  - [Библиотека промтов](#библиотека-промтов)
  - [Руководства скачиваются, а не поставляются в паке](#руководства-скачиваются-а-не-поставляются-в-паке)
  - [Список моделей](#список-моделей)
  - [Модели, уже скачанные вами для Ollama](#модели-уже-скачанные-вами-для-ollama)
- [Куда кладутся веса](#куда-кладутся-веса)
- [Использовать уже имеющуюся модель](#использовать-уже-имеющуюся-модель)
  - [Сборки поменьше](#сборки-поменьше)
  - [Если нода сообщает, что пакета не хватает](#если-нода-сообщает-что-пакета-не-хватает)
- [Самая лёгкая загрузка вообще без установки](#самая-лёгкая-загрузка-вообще-без-установки)
- [GGUF — ещё легче, и ставить ничего не надо](#gguf--ещё-легче-и-ставить-ничего-не-надо)
  - [Прогресс на ноде](#прогресс-на-ноде)
  - [Переменные окружения](#переменные-окружения)
  - [Языки](#языки)
- [Замечания](#замечания)
- [Благодарности](#благодарности)
- [Лицензия](#лицензия)

## Что потребуется до установки

Путь через LoRA — это языковая модель на 27 миллиардов параметров, а не лёгкий
помощник. Обойти приведённые ниже требования нельзя: адаптер привязан к одной
конкретной базовой модели. **К нодам писателя ничего из этого не относится** —
их требования в таблице ниже.

| Ресурс | Требование |
|---|---|
| Диск | **~52 ГБ** под `Qwen/Qwen3.6-27B` + **~3.5 ГБ** под адаптер — либо **~10–16 ГБ** всего на пути GGUF |
| VRAM (`nf4`, по умолчанию) | **~16 ГБ** |
| VRAM (`int8`) | ~28 ГБ |
| VRAM (`bfloat16`) | ~54 ГБ, излишек accelerate вытеснит в оперативную память |
| VRAM (GGUF) | ~13–19 ГБ в зависимости от кванта, меньше — если выгрузить на GPU не все слои |
| Пакеты | `transformers`, `peft`, `accelerate`, для `nf4`/`int8` — `bitsandbytes`. **На пути GGUF — ничего:** `llama-cpp-python` берётся, если он уже стоит, а если нет — скачиваются официальные бинарники llama.cpp |

> **Текстовый энкодер MiniMax-H3 для этого переиспользовать нельзя.** Это другая
> модель: Qwen3-VL-32B со словарём 151936 против базы адаптера Qwen3.6-27B со
> словарём 248320. В нём нет ни одного модуля linear-attention `in_proj_*`, на
> которые вешается адаптер; он обрезан до первых 50 слоёв из 64; и в нём нет ни
> `lm_head`, ни финальной нормализации, то есть генерировать текст он физически
> не может. Он отдаёт только скрытые состояния для DiT.

## Установка

Склонировать в `ComfyUI/custom_nodes/` и поставить зависимости в то же
Python-окружение, в котором работает ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI
```

Для portable-сборки ComfyUI под Windows:

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\MiniMax-H3-Prompt-Rewriter-ComfyUI\requirements.txt
```

Либо установить из реестра Comfy через ComfyUI-Manager.

### Готовые воркфлоу

Вместе с паком идут двенадцать воркфлоу; после установки они появляются в браузере
шаблонов ComfyUI (*Workflow → Browse Templates*) под именем этого пака. У
каждого своя карточка, и каждый запускается сам по себе: при открытии ничего
не отключено, и нет второй ветки, которую надо заглушить перед запуском.

| | Воркфлоу | Что нужно |
|---|---|---|
| 1 | **Write a prompt** — строчка замысла на входе, полное H3-описание со звуком на выходе | один GGUF на 2.6 ГБ |
| 2 | **Rewrite a prompt with the 27B LoRA** — тот самый адаптер LightX2V, которым назван пак | один GGUF на 15.7 ГБ |
| 3 | **Write a prompt from references** — писатель сам описывает ваши картинки и пишет по увиденному | два GGUF, 6 ГБ вместе |
| 4 | **Ready-made prompts** — тысяча готовых, с выбором в окне и кадром из клипа | вообще ничего |
| 5 | **Prompt to video** — штатный text-to-video ComfyUI с писателем перед ним | веса MiniMax-H3 |
| 6 | **References to video** — то же для Ref2VA: картинки попадают в генератор через писателя, в порядке его полосы | веса MiniMax-H3 ref2va |
| 7 | **LoRA triggers and effects** — слова, на которые отзывается адаптер, и десять эффектов MiniMax, дописанные в готовый промт | вообще ничего |
| 8 | **RefMods to video** — [RefMod](#minimax-h3-refmod), то есть внешность и голос, сохранённые уже закодированными: писатель видит и слышит его, а генератор получает тот латент, в котором он сохранён | веса MiniMax-H3 ref2va и RefMod в `models/refmods` |
| 9 | **P-H3 text to image** | веса P-H3 и prompt upsampler |
| 10 | **P-H3 image editing** | веса P-H3, upsampler и ваша картинка |
| 11 | **Write through an API** | настроенный локальный или удалённый API-сервер |
| 12 | **Write with the full Qwen model** | 9B Q5_K_M и F16-проектор, 7.39 ГБ вместе |

5, 6 и 8 создают видео MiniMax-H3; 9 и 10 - изображения P-H3. Остальные
воркфлоу создают или обрабатывают промты. 5 и 6 — штатные шаблоны из галереи
ComfyUI, у которых генератор свёрнут в один субграф, так что на экране
остаётся промтовая часть плюс одна нода. Чекпойнты в них те же, что в штатных
шаблонах, и недостающие ComfyUI предложит скачать сам. 8 — это 6, где в том же
субграфе вместо нативной референсной ноды стоит
[Reference Encode](#minimax-h3-reference-encode), а оба VAE вынесены наружу и
общие с RefMod'ом.

В каждом слева лежит заметка **Read me first**: что он делает, что скачивает,
что выставить перед запуском и куда идти дальше. Загрузчики картинок в 3 и 6 и
выбор RefMod'а в 8 намеренно **пустые** — имя файла, сохранённое в шаблон,
указывает на то, чего на вашей машине нет, — так что сначала выберите свои.

Картинки, которые браузер рисует рядом с именем, делает
`python tools/template_cards.py`: он кладёт их как `<имя>.jpg` рядом с
воркфлоу, там браузер их и ищет.

Из народного — [воркфлоу axiomgraph](https://github.com/axiomgraph/ComfyUIWorkflow):
Omni-рерайтер в паре с FL2VA (GPL-3.0, плюс несколько сторонних паков нод —
берите их из его репозитория).

## Ноды

### MiniMax-H3 Prompt Rewriter

Основная нода. Сама докачивает недостающее, загружает модель, генерирует и
освобождает VRAM обратно.

**Выходы**

| Имя | Содержимое |
|---|---|
| `rewritten_prompt` | Полный переписанный промпт — готов к подаче в MiniMax-H3 |
| `integrated_multimodal_description` | Только визуальная часть, разбитая по шотам |
| `overall_soundscape` | Только диегетический звук |
| `non_diegetic_music` | Только закадровая музыка |

**Входы**

- `prompt` — короткий промпт, который надо развернуть.
- `model` — базовая модель. В списке все записи из вашего файла моделей плюс все
  Qwen3.6-27B, уже лежащие на диске (с префиксом `on disk:`). Чего нет — скачается
  при первом запуске, прерванная загрузка продолжится. Кнопка **Model list**
  правит список в окне поверх графа — см. ниже.
- `resolution` / `duration` — условия, под которые составляется описание. Держите
  их равными тем, что подаёте в MiniMax-H3, иначе ритм шотов не совпадёт. Список
  идёт от `48:9` до `9:16`; две сверхширокие — это случай нескольких мониторов:
  32:9 — два экрана 16:9 рядом, 48:9 — три. Они задают, как компонуется кадр, и
  это всё, что решает нода; отрисует ли генератор такую форму — вопрос отдельный.
- `aspect_ratio` — **та же настройка, но проводом**, и пока к ней что-то
  подключено, она главнее. Есть у всех нод-райтеров и ререйтеров пака и принимает
  связь типа `STRING` или `COMBO`, так что тот же примитив, который уже задаёт
  пропорцию штатному **Resolution Selector**, тянется одним проводом и сюда. Она
  нужна потому, что форму кадра обычно решают в другом месте графа и пишут иначе:
  штатный **Resolution Selector** называет 16:9 как `16:9 (Widescreen)`, нода
  размеров скажет `3840x1080`, делитель — `1.78`. Читаются все три, и подпись
  вокруг пары читается насквозь: важна пара чисел. Размер кадра, отстоящий от
  записи в списке не больше чем на 2%, называется её именем — `1376x768` (а
  именно столько выдаёт Resolution Selector для 16:9 при 1 МП) приходит как
  `16:9`, а не как `43:24`. Пропорция, которой в списке нет, проходит как есть —
  так сюда попадают `2.39:1` или `5:4`. А то, что вовсе не пропорция, отклоняется
  с называнием значения, а не молча компонуется. У самого виджета `resolution`
  сокета нет: вход один, и это тот, который читает написанное другими нодами.
  **Пока провод подключён, выбор пропорций гаснет** — притушен, ничего не горит,
  клики не проходят: горящий квадрат называл бы пропорцию, которой на этом
  запуске не будет. **При отключении поле очищается**: нода сверху пишет своё
  значение прямо в виджет — так провод вообще и кормит виджетный вход, — поэтому
  текст иначе остался бы лежать. А лежит он не безобидно: что угодно в
  `aspect_ratio` главнее пикера, остаток в том числе.
- `quantization` — как грузить *неквантованный* чекпоинт: `nf4` (по умолчанию,
  ~16 ГБ VRAM), `int8` (~28 ГБ), `bfloat16` / `float16` (~54 ГБ). Игнорируется,
  если у чекпоинта своя квантизация.
- `greedy` — включён по умолчанию, результат детерминированный.
- `seed`
- `keep_model_loaded` — **по умолчанию выключен.** Модель на 27B выгружается сразу
  после генерации, чтобы на той же видеокарте следом отработала генерация видео
  H3. Включайте только когда подряд перебираете варианты промпта, а как потом
  вернуть карту — в разделе
  [Выгрузка оставленной модели](#выгрузка-оставленной-модели).
- `options` — опционально; подключите ноду **MiniMax-H3 Rewriter Options**.

### MiniMax-H3 Prompt Rewriter 8B (sees frames)

То же самое на гораздо меньшей модели, которая вдобавок мультимодальна. Второй
адаптер LightX2V обучен на
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct):
там, где 27B нужно *рассказать*, что на референсном кадре, этой показывают сам
кадр — и строку выравнивания она пишет по тому, что видит. Задач у неё четыре, а
не одна, и помещается она на карту, к которой 27B не подступится.

![Нода ререйтера 8B в ComfyUI в режиме T2VA: слева нода настроек, в центре сам ререйтер со входами first_frame и last_frame, справа готовый результат с пронумерованными кадрами, идентификатором говорящего (S1) и репликой в теге <d>[English] Hello.</d>](docs/node_rewriter_8b.png)

**Выходы** — те же четыре, что у ноды выше, так что в графе они взаимозаменяемы.

**Входы**

- `prompt`, `resolution`, `duration`, `greedy`, `seed` — как выше.
- `model` — база Qwen3-VL-8B в любой из двух форм, в которых издан адаптер.
  **GGUF** — это два файла из одной конвертации: модель и её проектор.
  **safetensors** — официальная папка
  [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), то
  самое, на чём адаптер обучался и в чём он опубликован. Записи с
  префиксом `on disk:` уже лежат в ваших папках моделей. **К адаптеру
  подходит только 8B**; Qwen3-VL другого размера отклоняется по имени и
  числам до того, как что-то скачается.
- `quantization` — как грузить базу в **safetensors**: `nf4` — около 8 ГБ VRAM,
  `int8` — около 13, `bfloat16` — около 20. Для GGUF игнорируется: там своя.
- `task` — `T2VA`, `I2VA`, `FL2VA`, `L2VA`. Сама модель называет их T2AV, I2AV,
  FL2AV и L2AV; это те же четыре задачи.
- `first_frame` / `last_frame` — опциональные входы IMAGE. `I2VA` читает
  `first_frame`, `L2VA` — `last_frame`, `FL2VA` — оба, `T2VA` — ни одного. Если
  подключить не тот, нода назовёт недостающий до загрузки: к какому концу клипа
  относится картинка — часть того, что модели сообщают.
- `keep_model_loaded` — на базе в **safetensors** работает для любой задачи:
  модель грузится в сам процесс ComfyUI и там же остаётся. На **GGUF** — только
  для `T2VA`: три задачи с кадрами идут через `llama-mtmd-cli`, каждый раз новым
  процессом, и модель уходит вместе с ним. Нода сообщает, как поступила, а не
  молча игнорирует переключатель.
- `options` — та же нода настроек, что и везде. В её списке `adapter` обе LoRA, а
  первая запись подставляет ту, что подходит выбранной базовой модели, — трогать
  её не нужно.

**Во что обходится**

| | Загрузка | VRAM |
|---|---|---|
| База Q4_K_M + проектор + адаптер Q8_0 | 4.7 + 0.7 + 0.7 ГБ | ~9 ГБ |
| База Q8_0 + проектор + адаптер F16 | 8.1 + 0.7 + 1.3 ГБ | ~13 ГБ |
| База safetensors + адаптер, `nf4` | 17.5 + 2.8 ГБ | ~8 ГБ |
| База safetensors + адаптер, `bfloat16` | 17.5 + 2.8 ГБ | ~20 ГБ |

Строки GGUF доплачивают ещё и за рантайм, один раз на машину: `I2VA`, `FL2VA` и
`L2VA` идут через `llama-mtmd-cli`, поэтому первая же из них скачает официальную
сборку llama.cpp — 34 МБ, а если `llama_backend` разрешится в CUDA, то 511 МБ, —
и установленный `llama-cpp-python` от этого не избавляет. Колесо умеет здесь
только `T2VA`.

Путь через GGUF — заметно меньшая загрузка, и ставить ничего не надо. Путь через
safetensors — та форма, в которой адаптер издан; модель остаётся в памяти на любой
задаче, а не только на `T2VA`, и это то, что стоит брать, если чекпоинт уже есть.
Взамен ему нужны `transformers` и `peft` — они у пака в зависимостях.

**Чего от неё ждать.** Все четыре задачи дают обученную форму: `T2VA` сразу
начинает с трёх полей, остальные три открываются той самой строкой выравнивания,
которую читает сам MiniMax-H3. Видимая работа адаптера — маркеры кадров: с
выключенным `use_lora` модель по-прежнему заполняет три поля, потому что контракт
лежит в системном промте, но перестаёт писать маркеры `[Shot 2]`, и ответ
получается примерно втрое короче.

Это всё-таки 8B, и в одном месте это видно: временную метку в строке выравнивания
она иногда пишет с тремя знаками после запятой вместо двух, а на `FL2VA` изредка
приписывает финальную картинку к `Shot 1`, а не к последнему кадру. 27B так не
делает. Дальше по графу эту строку никто не разбирает, так что на корректность
это не влияет, — но взглянуть перед вставкой стоит.

### MiniMax-H3 Prompt Rewriter Omni (sees and hears)

Третья лора LightX2V — и первая, которая слушает. Она обучена на
[Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), той самой
модели, которую капшенеры этого пака уже используют, поэтому референс доходит до
неё самим собой: картинкой, клипом, звуком. И только она из трёх умеет
**Ref2AV** — задачу полного референса, отвечающую шестью полями вместо трёх.

![Нода Omni-ререйтера в режиме REF2AV, к которой подключены ноды Load Video и Load Image: слева четыре референсных входа с чекбоксом на каждой строке, справа девять выходов от rewritten_prompt до references, а между ними стрип из трёх цветных квадратов - синий "pic 1" над ref_0, фиолетовый "aud 1" над ref_1 и синий "pic 2" над ref_2 - и строка подсказки под ними. Ниже пять задач с горящей REF2AV, восемь прямоугольников пропорций, нарисованных по соотношению сторон, от 48:9 до 9:16, с выбранным 16:9, длительность 10.0, промт из двух предложений, Qwen2.5-Omni-7B GGUF Q8_0 в списке моделей и пять кнопок внизу: Model list, Unload model, Save the last prompt, Edit the last prompt и Prompt library. Справа готовый ререйт заполняет все шесть полей Ref2AV, называя Subject 1, Subject 2, Picture 1, Picture 2 и Audio 1 в двух кадрах](docs/node_rewriter_omni.png)

**Выходы.** Семь, и какие из них заполнятся, зависит от задачи. Четыре кадровые
задачи возвращают те же три, что и остальные ререйтеры, поэтому они
взаимозаменяемы дальше по графу. `Ref2AV` возвращает шесть:
`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`,
`overall_soundscape` и `non_diegetic_music` — тот же набор, что производит
[райтер Ref2VA](#minimax-h3-prompt-writer-ref2va), и в том же смысле.
`rewritten_prompt` всегда содержит ответ целиком. За ними идёт `references`, и
это не текст: включённые референсы в порядке стрипа, которые
[Reference Slots](#minimax-h3-reference-slots) передаёт в
`MiniMaxH3ReferenceToVideo` с той же нумерацией, что в промте.

**Входы**

- `prompt`, `resolution`, `greedy`, `seed` — как выше.
- `references` — один растущий вход, принимающий IMAGE, VIDEO или AUDIO.
  Неправильного гнезда здесь нет: как назовут референс, следует из того, что он
  такое. Картинки нумеруются среди картинок, звуки среди звуков, поэтому звук,
  подключённый между двумя картинками, их не перенумерует.
- `task` — `T2AV`, `I2AV`, `L2AV`, `FL2AV`, `REF2AV`. Клипы и звук берёт только
  `REF2AV`; остальные четыре пишутся с одних картинок, и подключённый к ним звук
  отклоняется по имени ещё до того, как что-либо загрузится.
- `duration` — **нода его снапит.** MiniMax-H3 генерирует по сетке 17n+5 кадров
  на 24 fps, поэтому большинства длительностей не существует: просишь 10 секунд —
  это 243 кадра, 10.13 с, и именно *это* число попадает в ход и возвращается в
  строке выравнивания. Виджет — то, что вы имели в виду; строка и видео сходятся
  благодаря снапу.
- `model` — база Qwen2.5-Omni-7B: пара **GGUF** (модель и её проектор) или
  официальная папка **safetensors**. Записи с префиксом `on disk:` уже лежат в
  ваших папках моделей. Два похожих варианта помечены, а не спрятаны:
  Qwen2.5-Omni-**3B** — это `(wrong size for the adapter)`, а **Qwen2.5-VL-7B** —
  та же строка архитектуры, те же 28 блоков, та же ширина, то есть лора *и правда*
  прикрепится — это `(vision only, not an Omni build)`, потому что в его проекторе
  нет аудио-энкодера и ререйт вышел бы про звук, которого никто не слышал.
  Кнопка **Model list** правит секцию `models_omni` — см. ниже.
- `quantization` — как грузить базу **safetensors**: `nf4` около 9 ГБ VRAM,
  `int8` около 12, `bfloat16` около 20. Для GGUF игнорируется. **Берите самое
  крупное, что влезает в карту** — почему, ниже.
- `max_frames` — сколько кадров взять из клипа, равномерно по длине. Каждый кадр
  для модели — отдельная картинка.
- `reference_layout` — состояние стрипа в виде JSON. Это виджет, чтобы
  расстановка ехала вместе с воркфлоу и через API; интерфейс рисует вместо него
  квадраты.
- `keep_model_loaded` — соблюдается на базе **safetensors**. На базе **GGUF**
  задача с референсами идёт через `llama-mtmd-cli` — отдельный процесс, который
  уносит модель с собой, — и нода об этом говорит, а не молча игнорирует
  переключатель.
- `options` — та же нода опций, что и везде.

**Стрип — это и есть порядок.** Каждый подключённый референс появляется цветным
квадратом: синий — картинка, зелёный — клип, фиолетовый — звук, — и показывает,
как его назовут и из какого гнезда он пришёл. Перетаскивание меняет порядок, а
порядок нумерует ярлыки: перетащите вторую картинку в начало, и она станет
`<Picture 1>`. Клик по квадрату выключает референс, не выдёргивая провод. Полоса
задач гасит ту задачу, которую подключённые референсы обслужить не могут, и
говорит почему при наведении, — так что `FL2AV` с одной картинкой видно как
недоступный сразу, а не через две минуты в виде ошибки.

Переименования здесь намеренно **нет**, в отличие от стрипа Universal Writer.
Там картинке можно сказать стоять за субъекта или за клип, потому что ход,
собранный по руководству, называет референсы вручную. Здесь решает тип входа — а
*субъект* это то, что эта лора **производит** в `subject_definitions`, а не то,
что ей подают.

**Клик по подписи клипа включает его собственный звук.** На `Ref2AV` полоса
становится фиолетовой и читается `vid+a1`: звук клипа теперь `<Audio 1>`,
отдельный референс, и адаптер его слышит — звук подаётся отдельной записью прямо
перед клипом, другого способа передать звук клипа в его исходном инференсе нет.
Звуки клипов получают первые номера `<Audio N>`, отдельные звуки идут за ними —
так их нумерует `MiniMaxH3ReferenceToVideo`, — поэтому квадратик звука, стоящий в
стрипе раньше звучащего клипа, подаётся после него. На остальных задачах полоса
остаётся бледно-фиолетовой и без номера: включено, но ждёт `Ref2AV`. Клип без
звуковой дорожки нода назовёт и номера ему не даст.

**Во что обходится**

| | Загрузка | VRAM |
|---|---|---|
| База Q4_K_M + проектор + адаптер Q8_0 | 4.4 + 1.4 + 0.34 ГБ | ~9 ГБ |
| База Q8_0 + проектор + адаптер F16 | 8.1 + 1.4 + 0.65 ГБ | ~13 ГБ |
| База safetensors + адаптер, `nf4` | 22.4 + 1.3 ГБ | ~9 ГБ |
| База safetensors + адаптер, `bfloat16` | 22.4 + 1.3 ГБ | ~20 ГБ |

База GGUF доплачивает ещё и за рантайм llama.cpp, один раз: все задачи, кроме
`T2VA`, несут референсы и потому идут через `llama-mtmd-cli` — 34 МБ, а при
`llama_backend`, разрешившемся в CUDA, 511 МБ, и установленный
`llama-cpp-python` тут ничего не меняет.

GGUF-адаптер сконвертирован из собственных safetensors LightX2V через
`convert_lora_to_gguf.py` из llama.cpp и опубликован в
[pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF).
Сборка `Q8_0` вдвое меньше `F16` и ведёт себя так же.

**Квантование покупает видеопамять, а не скорость.** Что противоречит интуиции:
модель поменьше должна двигать меньше байт и идти быстрее. Замер на этой лоре, на
одной карте, тот же промт, те же две картинки, те же 256 токенов:

| | VRAM | Генерация |
|---|---|---|
| `bfloat16` | 19.4 ГБ | **18.6 ток/с** |
| `nf4` | 8.6 ГБ | 15.1 ток/с |
| `int8` | 12.0 ГБ | 6.0 ток/с |

`int8` — худший из трёх по обоим столбцам: медленнее `nf4` *и* крупнее его. Это
не особенность этой лоры: `load_in_8bit` в bitsandbytes — это LLM.int8(), который
разбивает каждый матмул на fp16-часть с выбросами и int8-часть, а потом сводит
их, каждый раз кастуя активации. Это способ уместить модель, которая иначе не
влезает, и он стоит того, чего стоит. `nf4` — лучший из «маленьких» вариантов, но
он тоже деквантует на каждом матмуле, поэтому и он не обгоняет `bfloat16`. Так что
квантовать стоит ровно до того, что реально влезает в карту: с 24 ГБ и больше
`bfloat16` и самый быстрый, и самый точный.

То же верно для любой базы safetensors в этом паке — механизм принадлежит
bitsandbytes, а не модели. **И ничего из этого не относится к маршруту GGUF**, где
у llama.cpp настоящие квантованные ядра: тот же FL2AV, который через `bfloat16`
safetensors идёт 26 секунд, через `Q4_K_M` идёт 10.

**Картинки масштабируются до того, как модель их увидит.** Собственный скрипт
инференса LightX2V ставит картинке потолок в 301056 пикселей — 384 токена, — а
кадру из клипа 100352, и эта нода делает так же. Дело не только в экономии:
картинка 1616x1616 — это 3249 токенов, две таких переполняют контекст 8k раньше,
чем посчитано хоть одно слово промпта, и модели показывают форму, которой она при
обучении не видела. Контекст после этого подбирается под реальную стоимость хода,
поэтому `Ref2AV` с восемью референсами его расширяет, а не падает.

**Путь safetensors показывает только картинки.** У внутрипроцессного пути
Transformers в ComfyUI нет способа передать модели звук, поэтому клип или звук на
такой базе отклоняются, а не выбрасываются молча из хода. Чтобы ими
воспользоваться, возьмите базу GGUF.

### MiniMax-H3 Qwen Rewriter

Работает с полными SFT-чекпойнтами LightX2V **Qwen3.5 9B** и **Qwen3.8 27B**,
а также с их GGUF-конверсиями и соответствующими vision-проекторами. Чекпойнт
FL2VA обслуживает T2VA, I2VA, L2VA и FL2VA, чекпойнт Ref2VA — Ref2VA.
Оригинальные SFT и GGUF выбираются через **Model list**; параметры генерации
и подготовки референсов вынесены в **Qwen Rewriter Options**.

Чипы расположены сверху; есть checkbox'ы входов, **+ instr**, разрешение,
длительность, bypass/repeat, библиотека и выход `references`. Изображения
и выбранные кадры видео читает сама модель. Аудио передаётся через явное
описание: у этих чекпойнтов нет аудиоэнкодера.

| Модель | Исходный SFT | GGUF Q5_K_M / Q4_K_M |
|---|---|---|
| Qwen3.5-9B FL2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF) |
| Qwen3.5-9B Ref2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA-GGUF) |
| Qwen3.8-27B FL2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA-GGUF) |
| Qwen3.8-27B Ref2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-GGUF) |

[Настройка, роли референсов, выходы и конверсия](docs/qwen_rewriter_RU.md).

![Qwen Rewriter Options, новая Qwen Rewriter и сгенерированный H3-промт](docs/qwen_rewriter_workflow.png)

### MiniMax-H3 Universal Rewriter

Все три лоры-ререйтера в одной ноде, а вкладка сверху выбирает, которая из них
запустится. [Prompt Rewriter](#minimax-h3-prompt-rewriter),
[Prompt Rewriter 8B](#minimax-h3-prompt-rewriter-8b-sees-frames) и
[Prompt Rewriter Omni](#minimax-h3-prompt-rewriter-omni-sees-and-hears) остались
как были — ничего из уже собранного не ломается.

Эти три адаптера — не три настройки одного и того же. 27B текстовая:
Qwen3.6-27B, одна задача, и референсный кадр доходит до неё только фразой,
которую кто-то написал. 8B мультимодальная: Qwen3-VL-8B, четыре задачи и сама
картинка. Omni мультимодальная и ещё слышит: Qwen2.5-Omni-7B, те же четыре
задачи и пятая своя. Другая база, другой размер, другая закачка.

Именно поэтому выбирать между ними руками утомительно. Промт — тот же промт,
соотношение сторон — то же соотношение, продолжительность — та же
продолжительность. Попробовать другой адаптер значило перенабрать всё это во
вторую ноду и потом держать их в согласии.

![Universal Rewriter на вкладке Omni в режиме I2VA: слева четыре референсные строки — first_frame, last_frame, reference_video и reference_audio, у каждой чекбокс, а к first_frame подключена нода Load Image, — справа девять выходов, от rewritten_prompt до references. Сверху три вкладки, горит «Omni LoRA / sees, hears»; полоса задач с горящей I2VA; восемь прямоугольников пропорций, нарисованных по соотношению сторон, от 48:9 до 9:16, с выбранным 16:9; слайдер длительности на 10; промт «A woman says hello to a alien.», а под ним model_omni с Qwen2.5-Omni-7B GGUF Q8_0 и quantization_omni в bfloat16, затем aspect_ratio, repeat_last и пять кнопок: Model list, Unload model, Save the last prompt, Edit the last prompt и Prompt library. Справа ответ: строка о том, что Picture 1 полностью использована начиная с первого кадра, и три поля, которые заполняет I2VA](docs/node_universal_rewriter.png)

*Одна нода, три вкладки. Выше строк с моделью вкладке не принадлежит ничего —
задача, пропорции, длительность и промт тут один комплект, какой бы адаптер ни
горел. Ниже каждая вкладка держит свою модель и своё квантование, и они стоят
там, где их оставили, в том числе после сохранения и загрузки. Второе отличие —
полоса задач: вкладка 27B зажигает один `T2VA`, вкладка 8B добавляет три
кадровые, а вкладка Omni сверх них — `Ref2VA`, до которого не дотягивается
никакая другая.*

**Так что вкладка несёт только то, что различается:**

| Принадлежит вкладке | Общее для всех |
|---|---|
| `model_27b` / `model_8b` / `model_omni` | `prompt`, `task`, `resolution`, `duration` |
| `quantization_27b` / `quantization_8b` / `quantization_omni` | `greedy`, `seed`, `keep_model_loaded`, `bypass`, `options`, оба кадра, клип и звук |

Виджет чужой вкладки прячется, а не сбрасывается, — так что при возврате он всё
ещё стоит там, куда вы его поставили, в том числе после сохранения и загрузки.

> **Captioner'а на вкладке 27B нет, и это намеренно.** Описание кадра, вложенное
> в промт, до этого адаптера доходит и не пропадает даром: предметы, фактуры и
> свет из него всплывают в кадрах, в обученной форме, и ничего не протекает в
> ответ. Но картинка растворяется в сцене, а не прибивается к отметке 0.00 —
> ровно то, что написано на
> [странице самой лоры](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA):
> T2VA там закрыт, FL2VA нет. Виджет на этой ноде выглядел бы обещанием кадровой
> задачи, которой 27B не умеет. Когда описание нужно,
> [Reference Caption](#minimax-h3-reference-caption) его напишет, и оно ляжет в
> `prompt` как любой другой текст; а когда картинка должна *быть* кадром — для
> этого обучена вкладка 8B.

**Переключатель задачи общий, и вкладка 27B его не трогает.** На ней горит
`T2VA`, а три кадровые задачи погашены — это честная картина текстовой модели, —
и клик там не делает ничего: значение, выставленное на других двух вкладках,
дождётся возврата. На вкладках 8B и Omni кадровая задача гаснет, пока кадр, из
которого она пишется, не подключён и не включён — ровно как `Ref2VA` на
Universal Writer.

**`Ref2VA` есть на вкладке Omni, и ради него добавлены гнёзда клипа и звука.**
`reference_video` и `reference_audio` читает только эта задача и больше никто —
у 27B и 8B нет слуха, а четыре кадровые задачи берут одни картинки, поэтому звук,
подключённый к `FL2VA`, отклоняется по имени, а не выбрасывается молча. На
`Ref2VA` всё подключённое становится референсом, который переиспользует целевое
видео, в порядке гнёзд: `first_frame` — это `<Picture 1>`, `last_frame` —
`<Picture 2>`, дальше `<Video 1>`, дальше `<Audio 1>`. Ответ приходит шестью
полями вместо трёх; четыре новых выхода стоят **после** тех трёх, что заполняет
любая задача, поэтому ничего уже подключённого к этой ноде не сдвинулось. Самый
последний — `references`: референсы, которые текущая вкладка действительно
прочла, для [Reference Slots](#minimax-h3-reference-slots), которая передаст их в
`MiniMaxH3ReferenceToVideo`.

> **Четыре референса, а не двенадцать.** Порядок — это и есть всё правило
> нумерации, а при четырёх гнёздах порядок задаётся самими гнёздами. Дальше уже
> хочется расставлять руками — для этого и есть перетаскиваемый стрип на
> [Prompt Rewriter Omni](#minimax-h3-prompt-rewriter-omni-sees-and-hears), вместе
> с `max_frames` и любым числом картинок. Эта вкладка — про другое: попробовать
> тот же промт на другом адаптере, не выходя из ноды.

**Четыре референсных входа, с чекбоксом на строке каждого.** `first_frame` и
`last_frame` те же самые, что у ноды 8B; `reference_video` и `reference_audio`
добавляет `Ref2VA`. Выключенная строка считается неподключённой — так референс
отставляется в сторону, не вытаскивая провод. Запустите вкладку 27B с чем-нибудь
подключённым — нода сама скажет, что не читает это и куда его деть, а не оставит
вас гадать.

**У `reference_video` второй квадратик, с динамиком.** Поставьте галочку, и на
`Ref2VA` вкладки Omni собственный звук клипа станет отдельным референсом:
`<Audio 1>`, а `reference_audio` сдвинется в `<Audio 2>` — в том порядке, который
даёт им `MiniMaxH3ReferenceToVideo`, — и `references` донесёт это до
[Reference Slots](#minimax-h3-reference-slots).

**`duration` — слайдер**, дотягивающийся до 30 секунд, пока вы это не поменяли:
правый клик по ноде, пункт `duration`. Смотрите
[Виджет duration](#виджет-duration). Все три адаптера учили на клипах в несколько
секунд, поэтому число далеко за этим пределом даёт худший промт, а не более
длинное видео: длину MiniMax-H3 берёт из своих настроек, а не из этой строки.

**Кнопки «Open guide folder» здесь нет** — ни один из трёх адаптеров не читает
руководство: формат зашит в системный промт, с которым их обучали. Кнопка **Model
list** есть и открывает все три списка этой ноды вкладками в одном окне: `models` для
вкладки 27B, `models_8b` для 8B, `models_omni` для Omni.

> **Если скрипт интерфейса не загрузился**, вкладки, переключатель задачи и
> выбор пропорций откатываются к обычным выпадающим спискам, на которых они
> построены, и все виджеты видны разом, а не по одной вкладке. Нода при этом
> работает точно так же. Эти три контрола — HTML и работают в обоих рендерерах
> ComfyUI; чекбоксы на двух кадровых строках рисуются на канвасе, которого
> *Modern Node Design (Nodes 2.0)* не запускает, — там кадр исключается
> отключением провода.

> **Ноде нужен свежий ComfyUI**: она написана на v3 node API, ровно как
> Universal Writer. На старой установке её не будет, а остальной пак
> зарегистрируется как обычно.

### MiniMax-H3 Rewriter Options

Всё, что трогают редко, убрано с основной ноды. Если её не подключать, ререйтер
работает с теми параметрами декодирования, с которыми адаптер был опубликован.

![Rewriter Options с Default guide и кнопкой System prompts под настройками генерации](docs/node_options.png)

| Вход | По умолчанию | Назначение |
|---|---|---|
| `max_new_tokens` | 2048 | Предел генерации |
| `temperature` / `top_p` / `top_k` | 0.7 / 0.8 / 20 | Сэмплирование, работает только при выключенном `greedy` |
| `repetition_penalty` | 1.05 | |
| `attn_implementation` | `sdpa` | `eager` или `flash_attention_2`, если установлен |
| `adapter` | репозиторий LightX2V | Какую сборку LoRA применять — см. ниже |
| `use_lora` | вкл | Выключите, чтобы сравнить с чистой базовой моделью |
| `merge_lora` | `auto` | Вплавить адаптер в веса при загрузке — вдвое больше токенов в секунду, см. ниже |
| `auto_download` | вкл | Выключите, чтобы нода падала с ошибкой вместо скачивания 52 ГБ |
| `downloader` | `built-in` | Каким кодом качается репозиторий с Hugging Face — см. ниже |
| `device` | `auto` | На какой карте считать языковую модель — см. ниже |
| `prompt_file` | `global` | В каком наборе сохранённых промтов работают подключённые ноды — см. [библиотеку промтов](#библиотека-промтов) |
| `self_check` | `warnings and notes` | Сколько из [самопроверки](#ответ-проверяется) произносится вслух: всё, только предупреждения или ничего |
| `fix_once` | `false` | Разрешить нодам действовать по находкам — [один заход, не цикл](#действие-по-находкам) |
| `captioner_thinking` | `off` | GGUF-капшенер: `off` отключает thinking, `auto` оставляет значение шаблона модели, `on` включает его при поддержке бинарником. |

Для GGUF-капшенера поддерживаемые флаги читаются через `--help` один раз для
каждой версии исполняемого файла. Режимы `off`/`on` передают `--reasoning` и,
если доступен, `--jinja`. Если `llama-mtmd-cli` не умеет менять thinking,
используется совместимый `llama-server` из той же папки, даже для одного
референса. Для нескольких референсов модель загружается один раз; сервер
закрывается после прогона или отмены. `MINIMAX_H3_MTMD_SERVER=never` сохраняет
CLI-путь. Если доступные бинарники не могут применить настройку, используется
значение шаблона с предупреждением в логе; неизвестный флаг не передается.
Лимит подписи остается 1024 токена. Пустая GGUF-подпись вызывает ошибку.
Настройка не меняет генерацию райтера и параметры CLIP-капшенера.

Эта же нода настроек подходит нодам писателя и капшенеру; `adapter` и `use_lora`
там просто не применяются.

#### `merge_lora` — адаптер как веса, а не как второй матмул

PEFT держит подключённую LoRA рядом с базовой моделью и считает её на каждом
токене, поверх базовых весов. Если вплавить её один раз при загрузке, та же
арифметика делается заранее, и разница вовсе не косметическая. Замерено здесь на
Qwen3-VL-8B с 8B-адаптером, 128 токенов на 5090:

| | адаптер подключён | вплавлен | слияние стоит |
|---|---|---|---|
| `bfloat16` | 8.98 с — 14.3 ток/с | **5.13 с — 25.0 ток/с** | 0.07 с |
| `nf4` | 9.43 с — 13.6 ток/с | **5.09 с — 25.1 ток/с** | 4.6 с |

`auto`, значение по умолчанию, берёт бесплатную половину: вплавляет адаптер в
неквантованную базу, где слияние стоит десятую долю секунды, и оставляет его
подключённым на базе bitsandbytes `nf4`/`int8`, где слияние означает деквантовать
каждый слой и заквантовать обратно. `on` делает это и там, если 4.6 с на загрузке
не жалко ради токенов; `off` — прежнее поведение. Пути GGUF всё это не касается:
там адаптер применяет llama.cpp по-своему.

**Вплавленный запуск — не дословно тот же, что подключённый.** Слияние
переставляет скобки в арифметике: `W + BA`, посчитанное один раз, не побитово то
же самое, что `Wx + B(Ax)` на каждом токене, — поэтому где-то выпадает другой
токен, а за ним расходится и остаток фразы. На той же паре и том же seed: два
расхождения на 128 токенов в `bfloat16` и больше в `nf4`, где добавляется ошибка
переквантования, о чём PEFT сам печатает предупреждение. Ни один из ответов не
«правильнее» другого, но это разные ответы, и выверенный до токена воркфлоу —
повод оставить `off`.

#### `adapter` — выпадающий список, и что в нём

По порядку:

- **`lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA`** — значение по умолчанию, и его
  лучше не трогать. Оно означает *ту сборку, которую список моделей называет для
  выбранной вами базовой модели*: PEFT-адаптер для базы `transformers`, GGUF из
  каталога — для GGUF, и адаптер 8B — на ноде 8B.
- **Все опубликованные точности обеих LoRA.** По F16 и Q8_0 на каждую; Q8_0 вдвое
  меньше и переписывает так же. Если выбранной нет на диске, она скачается из
  того репозитория, которому принадлежит.
- **Все `.gguf`-LoRA, уже лежащие в папках моделей ComfyUI** — с префиксом
  `on disk:` и архитектурой в подписи, чтобы `qwen35` и `qwen3vl` различались с
  первого взгляда. Так сюда попадает и LoRA, которую вы сконвертировали сами:
  положите файл в `models/LLM`, и он появится.

Раньше выбор кванта означал правку `models.json` руками — а это путь для
продвинутых, а не интерфейс. Сохранённые воркфлоу не пострадали: список
сериализуется той же строкой, которую показывает, а прежнее значение по умолчанию
осталось первой записью, символ в символ.

Два свойства у этого поля сохраняются — это единственная настройка, на которую
скачанный воркфлоу вообще может повлиять:

- **Сетевой путь отклоняется.** `\\хост\шара\...` и `//хост/шара/...` отбиваются
  до того, как их кто-либо прочитает: одно только обращение к такому пути — это
  попытка аутентификации на указанном хосте. Своя шара доступна по букве диска
  или точке монтирования, как и раньше, а путь в `models.json` не ограничен
  ничем — этот файл ваш и никуда не уезжает.
- **Применённый адаптер пишется в лог** на каждом запуске. Если запрошен не тот
  адаптер, что настроен, консоль скажет об этом на уровне warning. Подменённая
  LoRA иначе невидима: нода так же отрабатывает и так же заполняет все поля,
  просто пишет другое.

#### Код модели не исполняется

Чекпоинт для Transformers может нести собственный код модели: он объявлен через
`auto_map` в `config.json`, и при загрузке такой модели этот питон импортируется
и выполняется с правами вашего пользователя. Нода такую модель отклоняет и
говорит почему. В поставляемом списке таких моделей нет: все записи формата
`transformers` — это варианты Qwen3.6-27B, Qwen3-VL-8B и Qwen2.5-Omni-7B,
архитектуры, поддержанные в Transformers нативно, а записи GGUF до Transformers
вообще не доходят.

До 0.30.0 включительно здесь был переключатель `trust_remote_code`, который
разрешал такой модели исполниться. По умолчанию он был выключен, но это виджет,
и воркфлоу мог его включить, а список моделей можно пополнить через API ComfyUI,
которое не спрашивает логина. На ComfyUI, доступном из сети, этого хватало, чтобы
заставить его скачать модель и выполнить код из неё. В 0.31.0 переключатель
убран. Воркфлоу, сохранённый с ним, по-прежнему открывается: нода настроек
расставляет остальные значения по местам.

#### `device` — и почему он меняет смысл `keep_model_loaded`

У каждой ноды здесь было одно и то же предупреждение: выключите
`keep_model_loaded`, потому что карта понадобится под генерацию видео сразу после
ререйта. Это предупреждение существует только потому, что обеим моделям нужна
одна и та же карта. **Когда карт две — не нужна.**

Значения: `auto`, `cpu` и по одному `cuda:N` на каждое устройство, которое видит
ComfyUI. Одно написание, три бэкенда:

| | `auto` | `cuda:1` | `cpu` |
|---|---|---|---|
| бинарники llama.cpp | как было | `--device CUDA1` | `--device none`, 0 слоёв |
| llama-cpp-python | как было | `main_gpu=1`, split mode `NONE` | `n_gpu_layers=0` |
| Transformers | `device_map="auto"` | `device_map={"": "cuda:1"}` | `{"": "cpu"}` |

Главное здесь не размещение. **На другой карте модели самого ComfyUI больше не
выгружаются перед запуском.** Сейчас это делает безусловно каждый бэкенд — что
правильно, когда обеим нужна одна VRAM, и чистая потеря, когда нет: это полная
перезагрузка диффузионной модели после каждого ререйта. Выберите вторую карту — и
ререйтер грузится рядом, `keep_model_loaded` становится осмысленным, и ничему не
приходится двигаться.

Две детали. Устройство, которого на машине нет, **отклоняется, а не подменяется
молча** — воркфлоу, принесённый с двухкарточной машины, честно об этом скажет,
вместо того чтобы уехать на чужую карту и выселить батч на середине. И нумерация
здесь — та же, что у ComfyUI: запущенный с `--cuda-device 1` ComfyUI видит ровно
одно устройство, и это `cuda:0`, в том числе для подпроцессов.

Значения намеренно простые, а не `cuda:1 · RTX 4090`: подпись с названием карты
читается приятнее и ломает все сохранённые воркфлоу в день замены карты. Что
именно стоит в каждом слоте — написано в подсказке.

#### `downloader` — своя качалка или `huggingface_hub`

По умолчанию байты двигает сам пак: одно ranged-соединение за раз, с докачкой,
и ставить для этого ничего не нужно. Для того, во что никто не вызывался, это
правильный пол, и на быстром канале этого достаточно.

Недостаточно — на чекпойнте в 30 ГБ через канал, который одному потоку даёт 8
МБ/с. Репозитории Comfy-Org **Xet-backed**, и `hf_xet` тянет чанки *одного*
файла по многим соединениям сразу — в этом вся разница между «CLI справился за
пять минут, а нода за час». Значение `huggingface_hub` включает этот путь:

```
python -m pip install huggingface_hub hf_xet
```

в собственный Python вашего ComfyUI, затем перезапуск. Ни один из пакетов не
является зависимостью пака, и **без них ничего не ломается**: нода пишет команду
установки в лог, отмечает это в подписи и качает встроенным способом. Это важно
потому, что настройка уезжает внутри воркфлоу, и граф, собранный на машине с
этими пакетами, обязан работать на машине без них.

**Действует только на репозитории Hugging Face** — базовые модели, GGUF-писатели,
LoRA-адаптеры, модель вместе с проектором. Бинарники llama.cpp приходят из
GitHub Releases, а два гайда по написанию промптов — это по одному маленькому
запросу; через эту настройку не идёт ни то, ни другое.

**Недокачки не взаимозаменяемы.** Встроенная качалка кладёт байты в
`<имя>.part` рядом с файлом, `huggingface_hub` — в `.cache/huggingface` внутри
папки назначения. Переключение посреди закачки начинает этот файл заново и
оставляет чужие остатки, которые придётся удалить руками. Целый файл повторно не
качается ни в том, ни в другом случае: пак сам решает, чего не хватает, до того
как позовёт любой из бэкендов, — попутно это избавляет от пересчёта sha256
готового файла на 30 ГБ, которым `huggingface_hub` иначе его «удочеряет».

Про диск стоит знать одно: `hf_xet` держит чанковый кэш в
`~/.cache/huggingface/xet` (`HF_XET_CACHE`), а это на Windows диск `C:`, на каком
бы диске ни лежали модели. Проверка свободного места, которую пак делает перед
загрузкой, смотрит только на папку назначения.

**За антивирусом, который проверяет HTTPS.** Антивирусы с проверкой защищённых
соединений и корпоративные прокси того же рода переподписывают huggingface.co своим
корневым сертификатом и кладут этот корень в системное хранилище сертификатов —
поэтому браузер по-прежнему открывает сайт. Встроенная загрузка доверяет этому
хранилищу наравне с обычным набором certifi, так что качает везде, где качает
браузер. `huggingface_hub` и `hf_xet` приносят собственный TLS и туда не смотрят: за
таким фильтром они останавливаются на `certificate verify failed`, и нода об этом
говорит — верните встроенную загрузку или добавьте huggingface.co в исключения
фильтра.

### MiniMax-H3 Prompt Writer (T2VA/I2VA/FL2VA/L2VA)

Те же три поля на выходе — без LoRA и без 27B. В системный промт кладётся
[руководство MiniMax по написанию промтов](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md),
и по нему пишет любая GGUF-модель, умеющая следовать инструкциям. Qwen3.5-9B на
5.2 ГБ заполняет все три поля — с правильным словарём движений камеры,
идентификаторами говорящих и репликами `<d>[English] …</d>` — примерно за 20 секунд.

![Нода T2VA-писателя между нодой настроек и просмотрщиком текста: промт в одну строку, модель из Ollama, справа покадровое описание с тремя пронумерованными кадрами, абзац звукового ландшафта и non_diegetic_music с описанием фортепиано и виолончели](docs/node_t2va.png)

*T2VA, 10 секунд в 16:9, на локальном Qwen3.6-27B — 11.2 с. Промт был написан
по-русски, описание вернулось в английском формате H3 — со склейкой на
00:03.000, которую промт и просил, и с `non_diegetic_music: N/A`, потому что
музыки просили не давать.*

Компромисс стоит назвать прямо. LoRA **и есть** формат: 27B обучали до того, что
вывод H3 получается из семи строк системного промта. Здесь формат несут ~4000
токенов инструкций, которым модель должна следовать, — так что проза LoRA будет
плотнее, а форматирование надёжнее. Зато это работает на железе, к которому LoRA
не подступится.

Выходы полностью совпадают с нодой ререйтера — те же имена, тот же порядок, — так
что в сохранённом воркфлоу их можно менять местами.

Сверх входов ререйтера:

- `task` — **T2VA**, **I2VA**, **FL2VA** или **L2VA**. Всё, кроме T2VA, ещё и
  выдаёт строку выравнивания, которую H3 требует самой первой строкой, с уже
  подставленной длительностью до двух знаков; модели остаётся только номер
  финального кадра.
- `reference_material` — **эта нода читает текст, а не пиксели.** Для I2VA, FL2VA
  и L2VA опишите, что на референсных кадрах — руками или нодой
  [Reference Caption](#minimax-h3-reference-caption) выше по графу. Без этого
  модель придумает первый кадр, не имеющий отношения к вашей картинке.

В `model` попадают раздел `writers` из списка моделей и все GGUF, уже лежащие в
модельных папках ComfyUI, любой архитектуры. Ничему здесь не нужно быть
Qwen3.6-27B, и ставить тоже ничего не нужно: без `llama-cpp-python` нода запускает
официальные бинарники llama.cpp — так же, как на пути GGUF ниже.

| Модель-писатель | Загрузка | VRAM с руководством в контексте |
|---|---|---|
| Qwen3.5-4B `Q4_K_M` | 2.6 ГБ | ~5 ГБ — начните с неё на 8 ГБ |
| Qwen3.5-9B `Q4_K_M` | 5.3 ГБ | ~8 ГБ — лучший текст на гигабайт |
| Qwen3.5-9B Uncensored (HauhauCS) `Q4_K_M` | 5.2 ГБ | ~8 ГБ — не отказывается от сцен, которые отклоняет сток |
| Gemma 3 12B Instruct `Q4_K_M` | 6.8 ГБ | ~10 ГБ — альтернатива не от Qwen |
| Mistral Small 3.2 24B Instruct `Q4_K_M` | 13.4 ГБ | ~17 ГБ — для карт от 16 ГБ |

`n_ctx` поднимается автоматически: базовому руководству нужно около 9200 токенов
контекста, полнореференсному — около 12300, против значения по умолчанию 8192,
которое подходит короткому системному промту LoRA. Если позволить llama.cpp
обрезать, отрезано будет *начало* промта — руководство и контракт вывода, — и
ответ вернётся в каком-то другом формате без единого слова о причине.

Если поля не хватает, нода всё равно вернёт всё, что получила, и напишет на самой
ноде, каких полей нет. Снизьте температуру или возьмите модель на размер больше.

#### `system_prompt` — нацелить райтер на другую модель

Руководство и есть всё, что делает эту ноду райтером H3: оно уходит в системный
промт, и больше про формат в ноде не знает никто. Поэтому замена именно этого
текста превращает ноды в райтер подо что-то другое — LTX, Krea, Wan или ваш
собственный стиль. Текст в `system_prompt` заменяет собранное руководство.

Тогда руководство даже не скачивается — что важно на машине, которую держат
офлайн: не будет 24 КБ документа, скачанного ради того, чтобы его не прочли.

Короткий путь — [Guide Prompt (any LLM)](#minimax-h3-guide-prompt-any-llm): она
отдаёт штатный системный промт выходом. Возьмите, поправьте, верните обратно.

**Сохранённые системные промты.** Открой **System prompts...** в Rewriter Options,
API Writer Options или Qwen Rewriter Options. Можно начать с пустого черновика,
импортировать `.md` или `.txt` в UTF-8 либо выбрать задачу и отредактировать
копию штатного гайда через **Copy default guide**. Кнопка **Save and select**
сохраняет файл в `<ComfyUI user>/minimax_h3_rewriter/system_prompts` и выбирает его
в `system_prompt_file`. При переносе workflow на другой сервер скопируй туда
и файл гайда.

![Редактор System prompts с импортом файла, выбором задачи для копирования гайда и кнопкой Save and select](docs/system_prompts_editor.png)

Приоритет у текста в `system_prompt` райтера. Если поле пустое, райтер берёт
выбранный файл, а при выборе **Default guide** — штатный гайд задачи.

Две вещи не двигаются. Сообщение задачи не заменяется никогда — в нём промт,
соотношение сторон и длительность, нужные любому руководству. И ответ по-прежнему
разбирается на секции H3, поэтому руководство, отвечающее одним абзацем, наполнит
`rewritten_prompt` и оставит выходы секций пустыми. Это стоит знать, а не
избегать.

### MiniMax-H3 Prompt Writer (Ref2VA)

Полнореференсный режим по
[руководству MiniMax для full-reference](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
Шесть разделов вместо трёх, каждый — отдельный выход:

| Выход | Содержимое |
|---|---|
| `subject_definitions` | что обозначает каждая метка `<Subject N>` / `<Picture N>` / `<Video N>` / `<Audio N>` |
| `summary` | префикс `[тип задачи]` и референсные связи одним абзацем |
| `retention_analysis` | по метке: `fully_preserved`, `partially_preserved`, `attribute_transfer`, `weak_reference`, `fully_copy`, … |
| `detailed_description` | тело, кадр за кадром, со ссылками на метки там, где их роль действует |
| `overall_soundscape` | атмосфера и звуки действий |
| `non_diegetic_music` | музыка, слышная только зрителю |

![Нода Ref2VA-писателя рядом с просмотрщиком текста, где видны все шесть секций: subject_definitions с привязкой Subject 1 и Picture 1, summary с типом задачи reference generation, retention_analysis, где обе метки помечены fully_preserved, покадровое тело, звуковой ландшафт и non_diegetic_music](docs/node_ref2va.png)

*Тот же клип, что и выше, плюс одна строка `Picture 1:` в `reference_assets` —
17.4 с. Модель сама выбрала тип задачи, привязала `<Subject 1>` к коту, а
`<Picture 1>` — к стилю рисунка, и сама написала анализ удержания.*

`reference_assets` — **обязательное** поле, пустым не принимается:
полнореференсный режим описывает, как целевое видео переиспользует ассеты, а без
ассетов это просто T2VA под другим названием. По одному в строке —

```text
Picture 1: young woman, long dark hair, blue cardigan, thin silver necklace
Picture 2: corner cafe interior, brick wall, brass lamps, rain on the window
Audio 1: voice-timbre reference for the woman — low, unhurried, slight rasp
```

— а модель по ним расставит метки, выберет тип задачи и напишет retention-анализ.
Либо пусть эти строки напишет за вас
[MiniMax-H3 Reference Caption](#minimax-h3-reference-caption) — прямо из самих
ассетов.

### MiniMax-H3 Universal Writer

Для записи H3-промта через запущенную локальную LLM или удаленный API есть отдельный
[API Universal Writer](docs/api_writer_RU.md): пять задач, картинки, кадры видео,
звук и RefMod. Подключения и модели настраиваются в **API providers**, параметры
генерации - в **API Writer Options**. Ключи хранятся зашифрованно на сервере
Windows или Linux и не попадают в workflow. Начни с примера
**11 - Write through an API**.

![API Universal Writer: выбор задачи, пропорций, подключения и модели, входы референсов и выходы промта](docs/api_universal_writer_node.png)

![API Writer Options: лимиты генерации, формат ответа, выборка видео, частота аудио и исправление промта](docs/api_options_node.png)

Одна нода на весь кадр: референсы, их порядок и сам ререйт. Она делает то же, что
[Multi Reference Caption](#minimax-h3-multi-reference-caption) и обе ноды-писателя
вместе, для всех пяти задач сразу. Те три остаются как были — ничего из уже
собранного не ломается.

Складывать их в одну есть смысл ради **порядка**. `Picture 1` и `Picture 2` в
FL2VA не взаимозаменяемы: одна открывает видео, другая закрывает. До сих пор
единственным, что решало, какая из них какая, был порядок, в котором их записала
caption-нода, а он, в свою очередь, брался из того, в какой слот что воткнули.
Вещь настоящая, несущая — и нигде на экране не видная.

![Нода Universal Writer, к которой подключены ноды Load Video и Load Image: четыре референсные строки, у каждой свой чекбокс, три подключены; ниже полоса из трёх цветных квадратиков — vid 1 над ref_0, aud 1 над ref_1 и pic 1 над ref_2, у каждого кнопка «+ instr» — со строкой подсказки под ними; переключатель задачи с горящим Ref2VA; восемь прямоугольников пропорций, нарисованных по соотношению сторон, от 48:9 до 9:16, с выбранным 16:9; слайдер длительности на 15.1; промт, списки моделей капшенера и райтера, а внизу шесть кнопок: Model list, Open guide folder, Unload model, Save the last prompt, Edit the last prompt и Prompt library. Справа ответ из шести полей, написанный вокруг <Subject 1>, <Picture 1>, <Video 1> и <Audio 1>](docs/node_universal_writer.png)

*Четыре референса на одном растущем сокете — картинка как субъект, картинка как
кадр, клип и звук, — а справа ответ, написанный вокруг ровно этих меток:
`<Subject 1>`, `<Picture 1>`, `<Video 1>`, `<Audio 1>`. Здесь порядок квадратиков
совпадает с порядком слотов, потому что ничего не перетаскивали: номер — это
позиция, он меняется при любой перестановке, а имя слота под ним остаётся с
квадратиком.*

**Один сокет принимает картинку, клип или звук**, и слоты добавляются по мере
заполнения. Неправильного сокета здесь нет: то, куда именно вы подключились,
больше ничего не решает.

**Решает полоса под входами.** У каждого подключённого референса свой квадратик,
окрашенный по назначению и пронумерованный ровно так, как его пронумерует блок.
Под номером —
слот, с которого он пришёл, и именно это остаётся с квадратиком при перестановке:
номер — это позиция, и он меняется, стоит чему-нибудь сдвинуться.

| Квадратик | Метка | Что это значит |
|---|---|---|
| `pic` | `<Picture N>` | картинка как реальный кадр — первый, последний, ключевой, композиционный якорь |
| `subj` | `<Subject N>` | переиспользуемое содержимое — человек, животное, место, костюм, стиль |
| `vid` | `<Video N>` | клип или пачка кадров, которую нужно прочитать как один клип |
| `aud` | `<Audio N>` | тембр голоса, музыка, эмбиенс, эффект |

- **Тащите квадратик**, чтобы переставить. Нумерация меняется сразу.
- **Клик по подписи** меняет назначение картинки: `pic` → `subj` → `vid` и по
  кругу. Звук — это звук, у него подпись не переключается.
- **Клик по подписи клипа** засчитывает собственный звук клипа как ещё один
  референс. На `Ref2VA` полоса становится фиолетовой и читается `vid+a1`:
  капшенер описывает этот звук как `Audio 1` в блоке, прямо перед клипом, а
  `references` говорит [Reference Slots](#minimax-h3-reference-slots) отдать его
  генератору. Звуки клипов получают первые номера `<Audio N>`, отдельные звуки
  идут за ними — так их нумерует `MiniMaxH3ReferenceToVideo`. Капшенер для этого
  должен слышать: неслышащий останавливается с объяснением ещё до загрузки, а
  клип без звуковой дорожки нода назовёт и оставит без номера.
- **Клик по номеру** выключает референс, не выдёргивая провод; чекбокс на строке
  самого входа делает то же самое с другой стороны.
- **Клик по полосе под ним** спрашивает у этого одного референса не то, что
  обычно спрашивают у его роли. `+ instr` дописывает вашу строку к вопросу роли,
  `= instr` спрашивает вместо него; ПКМ возвращает как было, наведение
  показывает текст. Квадратик с такой строкой обведён янтарной рамкой для `+` и
  оранжевой для `=` — не заметить его трудно.

Так один сокет всё равно даёт все четыре метки, которые разрешает Ref2VA, а
различие, которое Multi Reference Caption проводит структурно — субъект это не
кадр, — здесь проводится на квадратике.

**Каждый квадратик показывает, что в него подключено** — здесь, в Prompt Rewriter
Omni и в Multi Reference Caption. У картинки миниатюра, у клипа первый кадр со
значком ▶ в углу, у звука волна со значком динамика и мелким названием файла;
номер уезжает в угол. При наведении клип проигрывается, а батч кадров
перелистывается; подсказка добавляет файл, размер, число кадров и длительность.
Звук после короткой задержки играет первые шесть секунд, и волна загорается по
мере проигрывания; нажатие на квадратик его глушит.
Квадратик, подключённый к загрузчику — Load Image, Load Video, Load Audio или
загрузчикам VideoHelperSuite, в том числе через reroute, — показывает файл сразу,
ещё до запуска. Подключённый через другие ноды показывает то, что пришло в ноду
при последнем запуске, а до него остаётся обычным квадратиком.

![Полоса Universal Writer в режиме Ref2VA, одиннадцать референсных строк, у ref_9 галочка снята. Квадратики слева направо: pic над ref_0 — чёрная кошка с 1 в углу, в янтарной рамке и с янтарной полосой «+ instr»; pic над ref_1 — деревянный забор, 2, в оранжевой рамке и с оранжевой полосой «= instr»; pic над ref_2 — мужчина в машине, 3; vid над ref_3 — мужчина идёт по улице, значок воспроизведения, 1; aud над ref_4 и ref_5 — значок динамика и волна, 1 и 2; vid над ref_6 — фигура в маске динозавра, значок воспроизведения, 2; во втором ряду pic над ref_7 — черепаха в самоцветах, 4; aud над ref_8 — динамик, обрезанное название файла, волна и 3; и aud над ref_9, притушенный, с тем же названием и волной и «--» вместо номера. Под полосой переключатель задачи с горящим Ref2VA, пропорции с выбранным 16:9 и duration 10.0](docs/node_ref_strip.png)

*Картинку, клип и звук видно с первого взгляда, и два звука друг от друга тоже:
ref_8 и ref_9 подключены прямо к загрузчику и показывают названия файлов, ref_4 и
ref_5 пришли через другие ноды и показывают волну с последнего запуска. ref_9
выключен — притушен, номер `--`, но узнаётся. Два квадратика с инструкцией — те,
что в рамке.*

**Переключатель задачи и выбор пропорций устроены так же**: выбор — это картинка,
а не строчка текста в выпадающем списке. А ещё переключатель задачи читает
полосу: задача гаснет, пока полоса не может дать то, из чего задача пишется. При
пустой полосе горит один `T2VA`, одна картинка зажигает `I2VA` и `L2VA`, две —
`FL2VA`. Три картинки гасят все три, потому что три для `I2VA` так же
невозможны, как ноль: это тот же счёт, на котором нода отказывает, переехавший
из запуска в видимое, а погашенная кнопка несёт ту же фразу подсказкой.
Считаются бейджи, а не сокеты: переключите картинку в субъекты — и задача,
которую она блокировала, загорится. Уже выбранная и ставшая невозможной задача
становится красной, а не падает потом исподтишка.

Каждый прямоугольник рисуется по своей пропорции в общем габарите, поэтому
широкий край читается по тому, сколько высоты от него осталось: `21:9`, `32:9` и
`48:9` выходят примерно в 15, 10 и 6 пикселей. Нода, которой не хватает ширины на
весь ряд, переносит его на вторую строку, а не обрезает последние прямоугольники,
и подрастает ровно на эту строку.

**`duration` — слайдер, с точностью до десятой секунды**, и докуда он
дотягивается — решать вам: правый клик по ноде, пункт `duration`. Смотрите
[Виджет duration](#виджет-duration).

**Два виджета моделей, потому что работы две.** `caption_model` читает референсы,
`writer_model` пишет промт по руководству MiniMax. Вход `clip` работает здесь так
же, как на caption-нодах, и при подключении используется для референсов. На T2VA
описатель не трогается вовсе.

**Что каждая задача ожидает от полосы:**

| Задача | Картинок | Всё остальное на полосе |
|---|---|---|
| T2VA | ни одной | игнорируется целиком, и нода об этом говорит, а не описывает их впустую |
| I2VA | ровно одна — первый кадр | уходит как справочный материал |
| L2VA | ровно одна — последний кадр | уходит как справочный материал |
| FL2VA | ровно две — сперва первый, потом последний | уходит как справочный материал |
| Ref2VA | сколько угодно | нужен хотя бы один референс любого рода |

Несовпадение отклоняется до того, как что-либо скачается или загрузится, и
сообщение называет полосу, а не сокеты: чинить нужно там. Лишнюю картинку
выключают или переименовывают квадратик, а не выдёргивают провод.

**Выходы — объединение выходов обеих нод-писателей**: три поля T2VA, шесть полей
Ref2VA и сам блок референсов. Задача, которая какое-то поле не пишет, оставляет
его пустым: выходы ноды не могут меняться от значения одного из её виджетов.
Последний выход, `references`, — не текст: это то, что пронумеровала полоса, для
[Reference Slots](#minimax-h3-reference-slots), чтобы подключить это к
`MiniMaxH3ReferenceToVideo` в том же порядке.

> **Одно отличие от Multi Reference Caption.** Та нода пишет блок в порядке
> самого руководства — субъекты, картинки, видео, звук, — что бы ни говорила
> разводка. Эта пишет в порядке полосы, потому что полоса и есть весь смысл. У
> нетронутой полосы порядок слотов.

> **Если скрипт интерфейса не загрузился**, полоса, переключатель задачи и выбор
> пропорций откатываются к тем обычным виджетам, на которых построены, — одно
> текстовое поле и два выпадающих списка, — и нода по-прежнему работает. Эти три
> контрола сделаны на HTML и работают в обоих рендерерах ComfyUI. А чекбоксы на
> строках входов рисуются на канвасе, которого *Modern Node Design (Nodes 2.0)*
> не выполняет: там референс выключают номером на квадратике. А Multi Reference
> Caption рисует те же квадратики, и чекбоксы ему больше не нужны вовсе.
>
> Три рисованных контрола также убраны из правой панели **Parameters**: рисовать
> такой контрол она не умеет, а чтобы попробовать, ей пришлось бы забрать его с
> ноды. Их место — на ноде. Всё остальное — промт, модели, слайдер
> длительности — в панели как обычно.

> **Ноде нужен свежий ComfyUI** — по той же причине, что и Multi Reference
> Caption: её растущие входы это `io.Autogrow` из v3-API нод. На старой сборке
> пропадают только эти две ноды, остальной пак регистрируется как прежде.

### MiniMax-H3 Image Prompt Writer

Разворачивает короткий запрос в подробный промпт для [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image).
Без активных картинок нода пишет промпт для генерации по тексту. Если подключить
от 1 до 16 картинок, она пишет промпт для редактирования. Первая активная картинка
служит основой. Перетаскивайте чипы для смены порядка, отключайте входы чекбоксами
и добавляйте указания через `+ instr`. Из IMAGE-батча берётся первый кадр.

![Image Prompt Writer в режиме Local GGUF с исходной картинкой и запросом на редактирование](docs/image_prompt_writer_node.png)

- **Local CLIP**: подключите `qwen3.5_9b_w4a8_gptq.safetensors` через Load CLIP
  с типом `minimax`. Это модель расширения промптов из
  [шаблонов H3 Image для ComfyUI](https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image).
- **MiniMax Context IR**: задайте `MINIMAX_API_KEY` в окружении сервера ComfyUI
  или выберите через **API providers** сохранённый профиль. Укажите в нём
  `https://api.minimax.io/v1` (global) или `https://api.minimax.cn/v1` (Китай),
  источник ключа и `MiniMax-H3-Image` в **Allowed models**. Запрос и активные
  картинки отправляются в [API расширения промптов](https://platform.minimax.io/docs/api-reference/image-generation-h3-context-ir).
  MiniMax тарифицирует расширение.
- **API Writer** использует выбранную модель на локальном или удалённом сервере
  Chat Completions. Профиль и модель выбираются как в **API Universal Writer**.
  Для генерации по тексту подойдёт текстовая модель. Для Image Edit выберите
  модель с поддержкой изображений, включите `image` в **Allowed models** и
  проверьте, что сервер принимает текст и картинки в одном запросе. Картинки
  отправляются в порядке активных чипов.
  [Открытые инструкции MiniMax для расширения промптов](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE)
- **Local GGUF** запускает выбранную модель через локальный рантайм llama.cpp.
  Для генерации по тексту выберите модель из **Writers**. Для Image Edit
  выберите пару GGUF и подходящего проектора из **Captioners**. В **Model list**
  отображаются модели из `models/LLM`. Подключите **Rewriter Options** к
  `local_options`, чтобы настроить контекст, сэмплирование и GPU. Для Image Edit
  используется сборка llama.cpp с `llama-server`, картинки идут в порядке чипов.

В режиме API Writer через **API Writer Options** задаются лимит токенов,
температура, формат ответа и таймаут. С форматом `text` нода просит вернуть JSON
в самой инструкции. Форматы `json_object` и `json_schema` можно выбрать, если их
поддерживают модель и сервер. Выход `summary` показывает использованные профиль
и модель.

Подключите `prompt`, `width` и `height` к **MiniMax H3 Image Conditioning** из
шаблона ComfyUI. При `aspect_ratio=auto` пропорции выбирает модель. Можно задать
свои пропорции и `area` или точный `size` со сторонами, кратными 32, оставив
пропорции на auto. При редактировании **Reference Slots** передаёт из
`references` картинки с 1-й по 9-ю. Для 10–16 картинок подключите оригиналы
к Conditioning в порядке чипов. **bypass** пропускает исходный текст, а
**repeat_last** подставляет результат из текущей сессии или библиотеки промптов.

### P-H3 Prompt Writer (изображения)

Пишет промпты для [Pruna P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image-Comfy)
официальным upsampler на Qwen3.5-9B. Нужен ComfyUI 0.44.0 или новее.

![P-H3 Prompt Writer в режиме Text2Image с включённым use_ref_description, превью источника и выбором кадра](docs/p_h3_prompt_writer.png)

Подключите **Load CLIP**, тип `minimax`, с файлом `p-h3-image_prompt_upsampler`
ко входу `upsampler`. Похожий по названию текстовый энкодер 4B нужен генератору
изображений, а не писателю. Загрузкой и выгрузкой модели управляет ComfyUI.

- **Text2Image** пишет по запросу, соотношению сторон и холсту на 4 или 8 МП.
  По умолчанию подключённые референсы игнорируются, выход `references` пустой. При переключении
  режима отсоединять их не нужно.
- **Image2Image** читает до восьми картинок суммарно. Первая включённая картинка
  служит основой; перетаскивайте элементы ленты для смены порядка, кликайте для
  отключения, через **role** указывайте, что взять из референса. Поле **keep**
  задаёт, что сохранить. По умолчанию холст повторяет пропорции основы.

Включите **use_ref_description** в Text2Image, чтобы составить промпт по первому
активному референсу и своему запросу. Его чип помечен **source**. Перетаскивание
меняет источник, а **role** уточняет, какие детали взять. Для батча IMAGE, VIDEO
или визуального RefMod описывается первый из выбранных кадров. Кнопка **frame**
открывает выбор одного кадра; Automatic берёт первый кадр. Тот же upsampler
сначала описывает кадр, затем по описанию и запросу пишет самостоятельный
промпт. Во второй проход картинка не передаётся. Запрос имеет приоритет;
пустой запрос даёт описание изображённой сцены. Генератор получает
только текст, выход `references` остаётся пустым. Полученное описание кадра
видно в `summary`. Пропорции холста задаются
пресетом или `aspect_ratio`. С `bypass` источник подготавливается для превью,
а исходный промпт передаётся без изменений.

Чипы референсов расположены над вкладками режимов. Необязательный `aspect_ratio`
принимает STRING или COMBO: `16:9`, `3840x1080`, `1.78`. Он имеет приоритет
над выбором пропорций и `match_image_ratio`. Для произвольных соотношений
сохраняется выбранное число мегапикселей, размеры округляются до кратных 32.

Соедините `prompt`, `width` и `height` с **p-h3-image Conditioning**.
Выход `references` передайте в **MiniMax-H3 Reference Slots**, затем
`picture_1` в `image_1`, `picture_2` в `image_2` и так далее. Conditioning нужны
текстовый энкодер и VAE от P-H3. Сэмплер использует восемь шагов с расписанием
P-H3 и сдвигами ModelSamplingMiniMaxH3 1.0/1.0; готовая схема есть в примерах.

Визуальные **RefMod** разворачиваются в обычные картинки. `refmod_images`
выбирает до указанного числа равномерно распределённых фотографий или кадров
из каждого RefMod. Все они входят в общий лимит восьми картинок. Файлам без
сохранённых или декодированных картинок нужен исходный H3 video VAE на загрузчике
или входе `refmod_vae`. Strength применяется к картинкам. P-H3 кодирует их заново:
исходный латент RefMod не внедряется, а декодирование с повторным кодированием
может потерять детали. Аудио-RefMod отклоняются. Выход `summary` показывает точную
нумерацию Picture после разворачивания.

К тем же входам референсов подключаются **VIDEO** и выход **IMAGE** из VHS Load
Video. `video_images` автоматически выбирает равномерно распределённые кадры из
VIDEO и батчей IMAGE крупнее восьми картинок; меньшие батчи сохраняются целиком.
Все источники делят общий лимит восьми картинок.

Кнопка **frames** под референсом открывает выбор точных кадров: обзорную ленту,
соседние кадры и ввод номера. Отметьте нужные картинки галочками и нажмите
**Apply frames**; выбор сохранится в workflow. **Automatic** возвращает
автоматический выбор для этого входа. В upsampler и выход `references` попадает
один набор кадров в хронологическом порядке. Для первого превью выполните ветку
входов с `bypass`. Повторите запуск после смены источника или выгрузки входа из
кеша ComfyUI. Сам выборщик хранит маленькие миниатюры, а не полный батч.
Номера начинаются с единицы в полученном референсе: у VHS это кадры после
`skip_first_frames`, `select_every_nth` и остальных преобразований, у RefMod -
сохранённые или декодированные картинки.

![Выбор кадров P-H3: большое превью, лента с номерами и выбранные кадры референса](docs/p_h3_frames_editor.png)

Собственный шаблон модели и её edit brief сохраняются целиком. Проверяются
пустой ответ, зацикливание и неверные ссылки на картинки; видеополя, тайминг шотов
и звук не требуются. Результат можно сохранять, редактировать и повторять через
библиотеку промптов. Референсы и холст при повторе берутся из текущих входов.
`bypass` передаёт запрос как есть, сохраняя подготовку референсов и холста.

Начните с **9 - P-H3 text to image** или **10 - P-H3 image editing**. Для
редактирования сначала выберите свою картинку в Load Image. Если upsampler
израсходовал бюджет на размышления и не выдал промпт, увеличьте `max_length`.

![Результат редактирования P-H3 в Save Image: черепаха в красном вязаном шарфе](docs/p_h3_result.png)

### MiniMax-H3 Reference Caption

Ноды-писатели читают текст, а не пиксели. Вот откуда берётся этот текст:
подключаете картинку, аудиоклип или видео — и маленькая мультимодальная модель
описывает их одной размеченной строкой `reference_assets`.

Замерено на Qwen2.5-Omni-3B в 3.4 ГБ: **3 с на кадр, 2 с на аудиоклип, 5 с на
видео.** Работает через `llama-mtmd-cli`, а он лежит в том же архиве, что и
`llama-completion`, — так что машина, где ререйты уже шли через бинарники,
рантайм для этого не качает.

**Машина, где они шли иначе, платит за него здесь.** `llama-cpp-python` — это
то, что берёт `gguf_runtime = auto`, если колесо импортируется (а в свежей
portable-сборке ComfyUI оно уже есть), а база safetensors вообще не касается
llama.cpp; ни тот, ни другой путь архива не скачивал, и приходит он на первом же
капшене: 34 МБ, а если `llama_backend` разрешится в CUDA, то 511 МБ. Колесо от
этого тоже не избавляет, с какими бы флагами оно ни собиралось: мультимодальный
вход идёт через `llama-mtmd-cli`, отдельную программу, а колесо — это набор
разделяемых библиотек, исполняемых файлов в нём нет вовсе. Поэтому
`gguf_runtime` — настройка нод-райтеров: капшенеры работают через бинарники, что
бы там ни стояло.

**Соединяются в цепочку** проводом из `reference_assets` во вход `previous`
следующей ноды. Каждая метка нумеруется **внутри своей категории** — это
собственное правило руководства («`<Video N>` and `<Audio N>` are numbered
independently»), поэтому четыре ассета выйдут как `Picture 1`, `Picture 2`,
`Video 1`, `Audio 1`, а не как 1–4.

![Две ноды Reference Caption в цепочке: Load Image кормит первую с ролью Picture, Load Audio — вторую с ролью Audio, провод reference_assets идёт из первой ноды во вход previous второй, а в просмотрщике справа видны обе строки — Picture 1 и Audio 1](docs/node_ref_caption.png)

*Картинка и аудиоклип через Qwen2.5-Omni-7B — 4.4 с и 3.7 с. Вторая нода
получила блок первой на `previous` и дописала в него свою строку, а роль `Audio`
описала голос, а не расшифровала его: «a male speaker with a calm and measured
delivery, speaking in Russian».*

`role` выбирает и метку, и вопрос к модели, и вопросы различаются намеренно:

| `role` | О чём спрашиваем модель |
|---|---|
| `Subject` | признаки, которые должны сохраняться между кадрами — телосложение, волосы, одежда и её цвета, предметы в руках |
| `Picture` | кадр как кадр — стиль, крупность, ракурс, положение в кадре, окружение, свет |
| `Video` | субъекты, действия по порядку, движение камеры, склейки и их ритм |
| `Audio` | **звук, а не слова** — тембр голоса, возраст и пол на слух, подача и темп, инструментал, темп, эмбиенс |

Последняя строка — самая важная. `<Audio N>` в руководстве чаще всего именно
*тембровый* референс, а транскрипт выбрасывает ровно ту часть, которая нужна, —
поэтому в инструкции прямо написано «do not transcribe». Если дословные слова
всё же нужны для блока `<d>`, это работа для ASR: прогоните Whisper отдельно и
вставьте строку руками.

Остальные входы:

- `description` — впишите описание сами, и **никакая модель не запустится вовсе**.
  Самый быстрый способ добавить ассет, который описывается шестью словами.
- `instruction` — полностью заменить вопрос роли.
- `max_frames` — сколько кадров взять из батча IMAGE **или из VIDEO**, равномерно.
  Все сразу переполнили бы и контекст, и терпение: две секунды при 25 fps — это
  56 картинок через визуальную башню, а тридцать секунд — 750. Именно это и
  делает стоимость описания клипа не зависящей от его длины.
- `context_size` — `0` подбирает контекст под референсы и под карту, а это не то
  же самое, что «контекст самой модели»: llama.cpp резервирует весь KV-кэш до
  того, как прочтёт первый пиксель, и модель, обученная на 262144 токена,
  требует под него десятки гигабайт — Qwen3-VL-8B ровно 36 ГБ, на карте, которая
  описала бы картинку за три секунды. Один кадр 1024×1024 — это уже больше двадцати
  медиа-чанков, поэтому число референсов — половина ответа, а память устройства —
  вторая. Вписанное число выполняется буквально; слишком малое роняет запуск,
  а не обрезает его.

> **Не всякий мультимодальный GGUF здесь работает**, и те, что не работают,
> падают громко. `mtmd` в llama.cpp должен понимать формат проектора: у Gemma 4
> он валит процесс целиком на `b10310` — и с файлом самой Google, и с файлом
> unsloth, и на CUDA-сборке, и на CPU, — тогда как `llama-completion` гоняет ту
> же модель как текстовую без единой жалобы. Поэтому в списке `captioners` лежат
> только пары, которые действительно запускались:
>
> | Капшенер | Загрузка | Модальности |
> |---|---|---|
> | Qwen2.5-Omni-3B `Q4_K_M` + mmproj | 3.4 ГБ | картинка, звук, видео |
> | Qwen2.5-Omni-7B `Q4_K_M` + mmproj | 5.8 ГБ | картинка, звук, видео |
>
> Модель и её `mmproj`, лежащие вместе в одной папке внутри `models/LLM`,
> предлагаются автоматически — так что попробовать другую можно ничего не правя.

### MiniMax-H3 Multi Reference Caption

Та же работа сразу на весь кадр, в одной ноде. Цепочка точна, но она разрастается:
пять референсов — это пять нод, пять проводов и пять шансов оставить где-нибудь не
ту роль. Эта нода сворачивает цепочку в одну коробку и вовсе забирает выбор роли:
**метка задаётся тем, в какую группу воткнут ассет.**

Это словарь руководства, ставший структурой. Ref2VA определяет ровно четыре
референсные метки и запрещает выдумывать новые — значит четыре группы входов
покрывают формат полностью, и описать картинку как звук становится *невозможно*,
а не просто нежелательно.

| Группа | Метка | Что сюда идёт |
|---|---|---|
| `subjects` | `<Subject N>` | переиспользуемое видимое содержимое — человек, животное, место, костюм, стиль |
| `pictures` | `<Picture N>` | картинка как настоящий кадр: первый, последний, ключевой, композиционный якорь, раскадровка |
| `videos` | `<Video N>` | отношения на уровне целого видео — исходник монтажа, точка продолжения, работа камеры и ритм |
| `audios` | `<Audio N>` | тембр голоса, музыка, эмбиенс, эффект |

Слоты растут по мере заполнения, один свободный всегда наготове, — нода ровно
такой высоты, какой требует кадр, и ни строкой больше.

![Нода Multi Reference Caption рядом с просмотрщиком текста: семь референсных слотов, у каждого чекбокс на своей строке, подключённый слот video_0 снят, а subject_0 и audio_0 отмечены, и в просмотрщике блок ровно из двух строк — Subject 1 с черепахой, панцирь которой усыпан самоцветами, и Audio 1 с мужским голосом на фоне птиц и плеска воды](docs/node_multi_ref_caption.png)

*Два ассета через Qwen2.5-Omni-3B — 6.0 с. Видео подключено и выключено, поэтому
блок вышел как `Subject 1` и `Audio 1`, без строки `Video` вовсе: референс
остаётся в графе, просто на этом прогоне он ничего не стоит.*

**Полоса под входами — это по одному цветному квадратику на подключённый
референс**, в том порядке, в каком будет написан блок, и с подписью, как каждый
будет назван. Клик по квадратику выключает референс, ничего не отсоединяя:
капшен стоит загрузки модели и от секунд до минут, поэтому «всё, кроме вот
этого» — самое обычное желание, а выдёргивать ради него провод значит терять
ровно ту разводку, которую вы и хотели сохранить. Состояние сохраняется с
воркфлоу и уходит через API, как любое другое значение.

Квадратики здесь не таскаются и подписи не переключаются, в отличие от
[Universal Writer](#minimax-h3-universal-writer): эта нода пишет блок в порядке
самого руководства, а имя референсу даёт группа, в которую он воткнут.

**Полоса под квадратиком спрашивает у этого одного референса другое.** Тёмная,
пока вопрос обычный для его роли, а когда он свой — янтарная, и весь квадратик
обведён рамкой того же цвета. Клик открывает окно
с текстом, ПКМ очищает, наведение показывает. Именно на референс, а не на ноду:
у ноды, которая разом описывает картинку, клип и звук, нет одного вопроса,
подходящего всем троим.

Галочка в этом окошке решает, чем ваш текст является, а полоса потом это
показывает: `+ instr` — строка **дописана** к вопросу роли, `= instr`, на
оранжевой подложке, — спрошена **вместо** него. Нужны оба режима.

| Что вы пишете | Режим | Почему другой не работает |
|---|---|---|
| `Не упоминай окно.` | дописать | Правило — не вопрос. Спрошенное само по себе, оно не оставляет ничего, что просило бы описание, и модель отвечает правилу: `Subject 1: No`. |
| `Всегда пиши "blah blah blah".` | вместо | Оставленный перед ним вопрос роли противоречит ему, и маленькая модель разрешает это в пользу описания. |

В режиме «вместо» ваш текст забирает себе и `length`: кто пишет вопрос, тот
владеет формой ответа, поэтому длину назовите сами, если она важна. В режиме
«дописать» и вопрос роли, и пресет длины остаются на месте.

**Слот `videos` принимает и `VIDEO`, и батч `IMAGE`** — смотря что отдаёт ваш
загрузчик; `Load Video (Upload)` из VideoHelperSuite подключается напрямую. Кадры
в обоих случаях сэмплируются равномерно до `max_frames`, так что стоимость клипа
по-прежнему не зависит от его длины.

**Блок пишется в порядке руководства** — субъекты, картинки, видео, аудио, — а не
в порядке проводов, и каждая метка нумеруется внутри своей категории, продолжая
то, что пришло на `previous`. Так что нода спокойно стоит в цепочке с одиночными
нодами по обе стороны.

`model`, `length`, `seed`, `max_frames`, `context_size` и `bypass` общие для всех
ассетов ноды; вопрос — нет, он живёт на полосе под каждым квадратиком. `role` и
`description` убраны намеренно: роль — это группа, а описание, написанное руками,
относится к одному ассету за раз. Для ассета, который проще описать самому,
остаётся [Reference Caption](#minimax-h3-reference-caption).

> **Этой ноде нужен свежий ComfyUI.** Её растущие входы — это `io.Autogrow` из
> v3-API нод. На старой сборке пропадёт только она одна — остальной пак
> зарегистрируется ровно как прежде.

### Описание моделью, которая у ComfyUI уже загружена

У обеих caption-нод и у Universal Writer есть вход `clip`. Подключите к нему мультимодальный текстовый
энкодер из `CLIPLoader` — и каждый ассет будет описан *этой* моделью, а не GGUF
из `model`; модель при этом остаётся ровно там, куда её положил аллокатор
ComfyUI.

В этом и весь смысл. Маршрут через GGUF поднимает `llama-mtmd-cli` по процессу на
ассет, поэтому кадр с пятью референсами читает веса с диска пять раз. Загруженный
энкодер читается **один раз**, а перезапуски при подборе формулировок не стоят
вообще ничего. Замерено здесь: три ассета — картинка, клип и его звуковая
дорожка — через Gemma-4 12B за **24 с**, и в логе на всех них один-единственный
`Requested to load`.

Только не читайте это как «быстрее». Те же три ассета через Qwen2.5-Omni-3B в
3.4 ГБ по маршруту GGUF заняли **16 с** вместе со всеми загрузками: маленькая
модель грузится достаточно быстро, чтобы три её перезагрузки обошлись дешевле
одного прохода 12B. Этот маршрут даёт не скорость, а *большую* модель без платы
за перезагрузку и перезапуски, которые загрузку пропускают вовсе, — и разрыв
растёт с каждым ассетом и каждым перезапуском.

Какие энкодеры подходят и что каждый принимает:

| Энкодер | Видит | Слышит |
|---|---|---|
| Qwen3-VL 4B / 8B | да — батч кадров читается покадрово | нет |
| Gemma-4 12B | да | **да** — сборка «unified», аудио проецируется напрямую |
| Gemma-4 E2B / E4B | да | детали на месте, но см. ниже |
| Gemma-4 31B | да | нет |

Gemma-4 опознаётся по самому чекпоинту, поэтому виджет `type` у `CLIPLoader` для
неё роли не играет. Подключите энкодер без слуха вместе со звуком — нода
откажется до запуска, ровно как отказывается на проекторе без аудиобашни.

> **Две вещи про Gemma-4 E2B/E4B.** Обе наблюдались на ComfyUI 0.30 с fp8-сборкой
> E4B и обе воспроизводятся штатной нодой ComfyUI `Generate Text` на том же
> файле — то есть ни одна из них не чинится силами этого пака:
>
> - **Аудио не доходит.** Капшен возвращается словами о том, что клип не был
>   предоставлен. Нода не отказывает, а пишет предупреждение в лог, когда видит
>   энкодер такой формы: на другом чекпоинте это может и работать. Gemma-4 12B
>   принимает звук правильно.
> - **Она размышляет вслух.** Промпт примирует пустой канал мыслей, а эта сборка
>   всё равно его заполняет. Caption-ноды срезают всё до закрывающего тега,
>   поэтому в `reference_assets` попадает только ответ, — но если вы собираете
>   `Generate Text` руками, срезать придётся вам.

Два виджета на этом маршруте мертвы, и в подсказках об этом сказано: `model`,
который вообще не читается, и `context_size` — это ручка KV-кэша llama.cpp.
`max_frames` работает по-прежнему: кадры прореживаются здесь, до того как их
увидит энкодер, а не отдаются на откуп его собственному сэмплированию до 1 fps.

Маршрут через GGUF никуда не девается: он достаёт модели, для которых у ComfyUI
энкодера нет, и ему ничего не нужно загружать заранее. Оставьте `clip`
неподключённым — и в обеих нодах не изменится ничего.

### Каптионер грузится один раз, а не по разу на референс

Полоска из шести референсов раньше означала шесть загрузок модели.
`llama-mtmd-cli` выполняет одно описание и выходит: для одиночной картинки это
верно, для полоски — расточительно, потому что модель и её проектор читались с
диска заново под каждый ассет, ещё до того, как кто-то посмотрел на пиксели.

Теперь весь цикл обслуживает один `llama-server`: поднимается перед первым
референсом, гасится после последнего. Модель, проектор, выборка, сид и системный
ход те же — меняется только загрузка. Три референса на каптионере 8B при
прогретом файловом кэше: было 8.7 секунды, стало 3.6, и разрыв растёт с каждым
добавленным референсом и с каждым гигабайтом веса каптионера.

Каждый путь повторяет сам себя точно — один и тот же граф на одном и том же сиде
даёт один и тот же капшен. А между собой капшен может разойтись на слово,
*a blue circle centered on* против *a blue circle is centered on*, — там, где два
токена почти равны и ничью решает то, как каждый рантайм батчит. Это арифметика
с плавающей точкой, а не другой заданный вопрос, — и именно поэтому системный ход
задаётся явно: без него расхождение было не в слове, а в самом описании.

Включать это не нужно, и ничего ради него не скачивается: `llama-server` лежит в
том же архиве релиза, что и `llama-mtmd-cli`, поэтому его ищут рядом с уже
найденным каптионером — и только рядом: сервер из другой сборки мог бы свести
модель со старым mtmd, который не прочитает только что принятый проектор.

Это ускорение, а не требование. Сборка без него, порт, который не поднялся,
сервер, так и не сказавший, что здоров, — каждый случай попадает в лог, а прогон
продолжается по процессу за раз: медленнее и во всём остальном так же. Один
референс сервера не поднимает вовсе — экономить нечего.
`MINIMAX_H3_MTMD_SERVER` принимает `never`, чтобы вернуть прежнее поведение, и
`always`, чтобы держать сервер даже ради одного ассета.

### MiniMax-H3 Guide Prompt (any LLM)

Собирает `system_prompt` и `user_prompt` по руководству и отдаёт их строками,
ничего не запуская. Не стоит ни видеопамяти, ни времени. Подключите их к той
LLM-ноде, которой вы уже пользуетесь — локальной, API, удалённой, — если модель
удобнее гонять не здесь. Поддерживает все пять задач, включая Ref2VA.

Третий выход, `prompt`, — это оба промпта одной строкой, потому что почти все
LLM-ноды принимают ровно один. **В том числе штатная `Generate Text`**, которая
с версии 0.30 гоняет языковую модель прямо в процессе ComfyUI, взяв её из
`CLIPLoader`. Это самый короткий путь к формату этого пака вообще без загрузки
GGUF — если текстовый энкодер Qwen3-VL или Gemma-4 у вас и так лежит ради
картиночной модели:

![Load CLIP кормит Generate Text, третий выход ноды Guide Prompt подключён в её вход prompt, а в просмотрщике справа готовый рерайт: subject_definitions с привязкой Subject 1 и Picture 1, summary с типом задачи reference generation, retention_analysis, тело по шотам с таймкодами, soundscape и non_diegetic_music](docs/node_guide_prompt.png)

*Ref2VA через Qwen3.5 в 4B — 37.7 с, и все шесть секций вернулись. Самое полезное
число — на статусной строке самой ноды: системный промпт в 25 042 знака это
**около 10 240 токенов контекста** до того, как модель напишет первое слово, и
именно это решает, потянет ли конкретный энкодер такую задачу вообще.*

Три настройки на `Generate Text` решают, заработает это или нет:

- **`max_length`** — её значение по умолчанию, 512, это бюджет *вывода*, и шесть
  секций Ref2VA в него не помещаются. Нужно порядка **2048** — столько же
  используют собственные писатели этого пака (`max_new_tokens` в ноде опций).
  Замерено на Qwen3-VL в 4B: переписывание Ref2VA вернулось целиком за **53 с
  вместе с загрузкой модели**, остановившись само примерно на 580 токенах — то
  есть внутри 2048 с запасом и настолько дальше 512, что значение по умолчанию
  оборвало бы ответ на середине секции.
- **`thinking`** — выключить. Руководство просит поля и ничего кроме, а
  размышления тратят тот самый бюджет на текст, который потом придётся срезать.
- **`use_default_template`** — оставить включённым, а `format` — на `plain`.

`format` — единственная ручка, о которой стоит знать. На `plain` два промпта
склеиваются пустой строкой, а LLM-нода заворачивает результат в собственный
шаблон модели, из-за чего всё руководство оказывается внутри *пользовательского*
хода. На `chatml` ходы расписываются здесь: токенизатор Qwen пропускает свой
шаблон, как только текст начинается с `<|im_start|>`, и руководство приходит
настоящим системным сообщением. В этой же ветке пропускается и подавление
размышлений у Qwen, поэтому пустой блок `<think>` дописывается за вас. По
устройству это заточено под Qwen — на Gemma и всём остальном оставайтесь на
`plain`.

### MiniMax-H3 Prompt Check

Все прочие ноды здесь пишут промт, а потом проверяют написанное. Эта только
проверяет и принимает текст на сокет — так что промт, пришедший откуда угодно,
читается ровно теми правилами, которыми читается переписывание из этого пака, и
разбирается на те же поля. Кормить её может всё, что производит прозу
MiniMax-H3: нода другого пака, загруженный в граф текстовый файл, промт,
найденный в сети, или набранное прямо в виджете.

Никакая модель не грузится и ничего не генерируется. Прогон стоит несколько
миллисекунд.

![Нода MiniMax-H3 Prompt Check: слева сокет options и три сокета референсов — ref_0, ref_1 и ref_2, два из них подключены; справа девять выходов — сначала prompt, затем integrated_multimodal_description, subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape и non_diegetic_music, и последним findings. Ниже — высокое пустое поле промта, список task на Ref2VA, длительность 9.9 и одна кнопка: Save the last prompt](docs/node_prompt_check.png)

*Выходы — те же, что у Universal Writer, минус два, которые рассказывают о его собственной работе каптионера. Поэтому нода проверки встаёт в граф на место писателя, а писатель — на место ноды проверки. Длительность 9.9, а не круглое число, потому что это то, для чего писался промт, а не то, что вы попросили бы у писателя. И кнопка одна, а не две: нода не запускает модель, повторять нечего и наводить на библиотеку нечего — но прочитанное можно сохранить.*

| Вход | Зачем |
| --- | --- |
| `prompt` | Текст для чтения. Из первого выхода он выходит нетронутым, так что нода может стоять в середине графа, ничего не меняя в том, что дойдёт до генератора. |
| `task` | Для какой задачи писался промт. Она решает, какие поля у ответа должны быть, сколько референсов каждого рода он вправе упоминать и ждать ли сверху строку выравнивания, — ошибка здесь делает чтение неверным, а не отсутствующим. |
| `duration` | Сколько длится целевое видео. Времена склеек читаются против него, поэтому ставьте то, для чего писался промт, а не то, что вы попросили бы у писателя. |
| `references` | Необязательные. Читаются только род и число — ничего не декодируется и никакой каптионер не запускается. С ними чтение покрывает ещё и то, что текст упоминает, против того, что тут на самом деле есть; без них эти два правила пропускаются, остальные работают. |
| `options` | Необязательные, и берётся оттуда только `self_check`: сколько сказано на экране. |

На выходах — промт, семь полей, которые может заполнить любая задача, и
`findings`: тот же блок, что нода пишет под собой, `!` для предупреждения и `-`
для заметки, пустой, когда сказать нечего. Этот выход всегда несёт всё
найденное: `self_check` управляет тем, что ноды объявляют во время прогона, а
это вопрос о шуме — подключить же выход значит спросить.

**Save the last prompt** на ней тоже есть, так что промт, пришедший со стороны и
прочитавшийся чисто, кладётся в [библиотеку](#библиотека-промтов) под именем, с
описанием и группами, ровно как написанный этим паком.

Одно чтение стоит ожидать заранее. Большинство промтов, написанных для H3
руками, — проза: ни меток полей, ни списка планов, ни тегов. Прочитанный как
`T2VA`, такой текст будет объявлен потерявшим поля и `[Shot 1]`, и это верное
чтение, а не поломка: подобный промт есть *указание переписывателю*, а не
готовый ответ H3. Скормите его одному из писателей и проверяйте то, что вернётся.

### MiniMax-H3 Prompt Reducer

Все остальные ноды здесь разворачивают. Эта сворачивает: превращает готовый
промт H3 обратно в короткую строку, из которой он мог быть написан.

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a low-angle medium
shot frames a sleek black cat walking steadily along the top of a weathered wooden
fence in a quiet suburban yard at dusk. The camera tracks right with small amplitude
at slow speed, following the feline as its soft fur catches the fading golden light...

overall_soundscape: A gentle evening breeze rustles through nearby grass and leaves...
non_diegetic_music: A sparse piano melody at a slow tempo...
```

возвращается как

```
A black cat walks along a wooden fence in a yard at dusk.
```

Пригождается в трёх случаях. Поменять одно слово в удачном промте, не переписывая
остальные четыреста, — сократить, поправить строку, отдать обратно писателю.
Отдать промт, написанный под H3, генератору, который хочет короткий: промт H3
имеет форму H3, а Wan, Hunyuan и Kling — нет. И подписать сохранённый промт
читаемой строкой, чтобы на карточке было имя, а не абзац.

**Половина этой работы — не работа для модели, и именно поэтому всё получается.**
Промт H3 — не проза, а известная форма: именованные поля, фиксированная строка
выравнивания сверху, метки `[Shot 2] At 0:03` по телу, теги `<Picture 1>` везде,
где цитируется референс, диалог в ограде `<d>`. Всё это обвязка, всё это
опознаётся по правилам, и всё это снимается до того, как модель о чём-то спросят.
До модели доходит один абзац обычного описания под короткой инструкцией — вот
почему с этим хорошо справляется 4B. Попроси её «обратить этот документ» — не
справилась бы.

Привязки к референсам уходят вместе с остальной обвязкой, и это намеренно: у
Ref2VA `subject_definitions` и `retention_analysis` описывают *ассеты*, и короткий
промт, сохранивший их, описывал бы картинки, которых у следующего прогона не
будет.

| Вход | Для чего |
| --- | --- |
| `prompt` | Готовый промт, который надо сократить. Любая из пяти задач, и говорить какая не нужно: текст разбирается по всем именам полей обоих семейств, а промт вообще без имён полей читается целиком как описание. |
| `model` | Любая инструкционная GGUF из того же списка, которым пользуются писатели, включая записи `on disk:` и `ollama:`; правится он кнопкой **Model list** под нодой. Здесь от модели требуется куда меньше, чем при письме, так что самая маленькая запись в списке тут разумный выбор, даже если там она им не была. |
| `detail` | Сколько возвращается. `idea` — голая строка, не длиннее десяти слов; `sentence` разрешает место и время суток; `paragraph` оставляет по предложению на каждое событие, а это то, что нужно промту с несколькими шотами, если порядок должен уцелеть. |
| `subjects` | Насколько подробно называются люди: `as written`, `age and gender` или `impersonal`. |
| `keep_camera` | Сохранить крупность, ракурс и движение камеры. По умолчанию выключено: камера обычно выдумка писателя, а не ваша, и без неё следующий рерайт выберет заново. |
| `keep_audio` | Свести звуковой ландшафт и музыку в одно предложение в конце. По умолчанию выключено, и выключено означает, что звуковые секции вообще не доходят до модели: их выбрасывает разборщик, а не инструкция. |
| `keep_style` | Сохранить носитель и look, с которых промт начинается: live-action, анимация, cinematic, документальность. |
| `language` | На каком языке вернётся ответ. Пусто — язык входа. На этой ноде это второй проход: строка сперва сокращается, потом переводится, — именно поэтому он надёжен; в подразделе ниже сказано, почему иначе не выходит. |
| `greedy`, `seed`, `keep_model_loaded` | Как у писателей. `greedy` тут стоит держать включённым: именно сэмплирование превращает «чёрного кота» в «лоснящегося обсидианового представителя кошачьих». |
| `options` | Необязательно. Обычная нода [Rewriter Options](#minimax-h3-rewriter-options) — `device`, `n_ctx`, `gpu_layers`, `gguf_runtime` и прочее. |
| `bypass` | Отдать `prompt` прямо на выход и не запускать ничего. |
| `system_prompt` | Заменить всю собранную инструкцию своей. `detail`, `subjects` и три галочки тогда перестают действовать — они существуют только чтобы собрать текст, который вы переопределяете. `language` продолжает работать: он отдельный запрос, а не часть этого текста. Разбор остаётся в любом случае, так что вашей инструкции достаётся очищенная сцена, а не исходный текст. |

Два выхода: `short_prompt` и `scene` — описание со снятой обвязкой и больше ничем.
Модель `scene` не касалась. Подключайте, когда нужна была только
детерминированная половина.

![The MiniMax-H3 Prompt Reducer node with a Show Any node beside it: an options socket on the left, short_prompt and scene as the two outputs on the right, a tall prompt box holding a full I2VA prompt with its integrated_multimodal_description and [Shot 1] marker, then the widgets — a 35B model on disk, detail on idea, subjects on impersonal, keep_camera and keep_audio true, keep_style false, language reading Chinese, greedy true, a seed set to randomize, keep_model_loaded and bypass false, an empty system_prompt box, and the Model list button under it. Under the node the status line reads '110 words in, 33 characters out - 1 shot markers dropped - translated into Chinese', and below it the two Chinese sentences themselves, which the Show Any node repeats](docs/node_reducer.png)

*Четыреста слов про кота, забор и сумерки в два коротких китайских предложения, за 25 секунд на 35B. Сразу три оси делают видимую работу: `detail` стоит на `idea`, поэтому возвращается голая строка; `keep_camera` вернул в неё нижний ракурс и проезд; `keep_audio` свернул весь звуковой ландшафт во второе предложение; а `keep_style` выключен, и `Live-action, cinematic`, с которого промт начинался, исчез. Счёт идёт в символах, а не в словах, потому что китайский не пишет пробелов, на которых держится подсчёт слов.

И одно, что кадр показывает честно: `subjects` стоит на `impersonal`, а кот всё равно вернулся чёрным. Правило просит голый род, и эта модель оставила цвет — из трёх осей эта сильнее всех опирается на модель, и целится она всё-таки в людей.*

#### Что оси делают на самом деле

Один промт, одна 4B, оси двигались по одной. На входе документальный промт I2VA
про пожилого рыбака, вытягивающего сеть, два шота, со звуковым ландшафтом и
виолончельным дроном:

| Настройка | Что вернулось |
| --- | --- |
| по умолчанию | An elderly fisherman with a thick grey beard and a faded yellow oilskin jacket hauls a dripping net over the gunwale of a small wooden trawler under a bruised pre-dawn sky. |
| `detail: idea` | An elderly fisherman hauls a net on a boat. |
| `subjects: age and gender` | An elderly man hauls a net on a boat under a pre-dawn sky. |
| `subjects: impersonal` | A subject hauls a net over the side of a boat under a pre-dawn sky. |
| `keep_camera` | A handheld close-up follows an elderly fisherman... |
| `keep_style` | Live-action, documentary style. An elderly fisherman... |
| `keep_audio` | ...under a pre-dawn sky. Waves slap the hull, rope creaks, gulls cry overhead, and the man breathes heavily. A low cello drone underneath. |

`impersonal` — для шаблонов, которые потом заполняют. Отданный генератору как
есть, он выдаст ровно то безличное ничто, которое просит.

Осей четыре, а не один регулятор абстракции, потому что они независимы. Длина
ответа и подробность имён — разные вопросы: однострочный промт вполне может
сказать «женщина в красном пальто», — а камера, звук и киношный look держатся
или выбрасываются каждый сам по себе.

#### `language` — это второй проход, и иначе не выходит

Просить сокращение и язык одним запросом не работает. Разобранный пример внутри
инструкции написан по-английски и другим быть не может — ничто здесь не переведёт
его на язык, набранный в виджете, — а модель, копирующая демонстрацию, копирует и
её язык вместе со всем остальным, что из неё берёт.

Отказ этот неравномерный, и именно поэтому его стоит описать. С тремя галочками
*выключенными* пример короткий, и правило языка соблюдалось. С *включенными* —
пример становится длиннее и наряднее, — три модели подряд ответили по-английски,
как правило ни формулируй и куда его ни ставь, в том числе последней строкой
системного промта, после примера.

Поэтому Reducer сперва сокращает, а потом переводит готовую строку — отдельным
запросом. У этого запроса одна цель, копировать нечего и не с чем торговаться, и
те же самые модели ему подчиняются: все испробованные сочетания вернулись на
запрошенном языке, на 4B, на 9B и на 35B, по-русски и по-китайски. Стоит это
одной короткой генерации на уже загруженной модели, и между двумя проходами нода
держит её в памяти, что бы ни говорил `keep_model_loaded` про «после».

Из устройства следуют две вещи:

- **`system_prompt` его не выключает.** Три галочки, `detail` и `subjects`
  перестают действовать, когда вы заменяете инструкцию, потому что они только для
  того и существуют, чтобы собрать заменённый текст. `language` в этом тексте не
  участвует вовсе, так что он продолжает работать.
- **`Reduce Prompt (any LLM)` так не умеет** — она собирает один запрос и ничего
  не запускает. Там язык остаётся правилом внутри инструкции, соблюдаемым или нет
  в зависимости от модели, то есть ровно тем поведением, что описано выше. Если
  короткий промт вернулся от этой ноды не на том языке, причина в этом, а надёжный
  путь — Reducer.

Остаётся качество перевода, а не отказ переводить: маленькая модель нет-нет да и
оставит слово непереведённым. Это модель, и у большей таких слов меньше.

#### Токены эффектов через редьюсер

[Эмбеддинг-эффект](#minimax-h3-effect-embeddings), вписанный в описание, выходит
с той стороны нетронутым, со строчной `e`: капитализация, которая приводит в
порядок начало предложения, знает, что такой токен трогать нельзя. Токен *выше*
первой метки поля прохода не переживёт — голову промта редьюсер выбрасывает
вместе со строкой выравнивания и всем остальным, что не описание и не звук. Это
контракт редьюсера, а не недосмотр: на выходе он даёт переиспользуемую строку, а
голова H3-промта привязывает её к конкретным референсным кадрам.

Ещё стоит знать: счёт слов и на этой ноде, и в проверке считает токен одним
словом. Настоящую цену в позициях промта говорит выход `tokens` ноды эффектов.

#### Один круг, который стоит знать

Reducer → правка строки → писатель → [Prompt Check](#minimax-h3-prompt-check).
Если в конце этого круга кот всё ещё идёт по забору, обе половины пака ведут себя
как надо. Это же самый дешёвый способ выяснить, достаточно ли хороша конкретная
маленькая модель для этой задачи: прогнать и прочитать строку.

### MiniMax-H3 Reduce Prompt (any LLM)

То же сокращение, но ничего не запускается: нода собирает `system_prompt` и
`user_prompt` и отдаёт их строками для той LLM-ноды, что у вас уже есть, —
локальной, API, удалённой. Не стоит ни VRAM, ни времени. Разбор при этом
происходит здесь, так что сцена, которая достанется вашей модели, уже чистая.

![The MiniMax-H3 Reduce Prompt (any LLM) node with a Show Any node beside it displaying the assembled system prompt: the rules list — never upgrade a word, add nothing, drop the writing rather than the story, answer with one sentence, name the subjects as written, drop the camera, drop the medium, write the answer in Russian — then the output contract, then the worked example with its scene and its answer 'A black cat walks along a wooden fence in a yard at dusk', and last the two lines saying the example is in English and the answer is to be in Russian. Four outputs run down the right of the node: system_prompt, user_prompt, prompt and scene](docs/node_reduce_prompt.png)

*То же сокращение, но ничего не запущено, за 0.01 с. Вся инструкция видна целиком, и в этом смысл ноды: 2112 символов системного промта, 632 пользовательского, и каждое правило в нём поставлено виджетом. Последние две строки — это та самая однозапросная попытка задать язык, о которой сказано выше: собрать больше одного запроса эта нода не умеет.*

Управляется теми же четырьмя осями, `format` склеивает два выхода так же, как в
[Guide Prompt](#minimax-h3-guide-prompt-any-llm), а четвёртый выход — `scene`,
как выше.

### MiniMax-H3 Effect Embeddings

Вместе с H3 MiniMax выложила десять эмбеддингов-эффектов — bullet time, dark
magic, fire breath, отъезд камеры как в «Шоу Трумана» и ещё шесть. Эта нода
вставляет их в промт.

Это не ключевые слова, и дальше от этого зависит всё. В каждом файле лежит один
тензор формы `[N, 5120]`: от пятидесяти до ста сорока двух позиций промта,
**уже прошедших через текстовый энкодер** и сохранённых числами. Строка
`embedding:minimaxh3_bullet_time` в тексте — не просьба про буллет-тайм словами,
а указание токенизатору ComfyUI вклеить эти позиции в последовательность именно
здесь. Поэтому эффект стоит *длины* промта, а не прилагательных, приходит
целиком или никак, и переформулировать его нельзя: формулировать нечего.

| Эффект | Позиций | Эффект | Позиций |
| --- | --- | --- | --- |
| Art is explosion | 50 | Kiss camera | 97 |
| Dark magic | 59 | Fire breath | 118 |
| Truman show | 90 | Blooming flowers | 123 |
| Bullet time | 94 | Spiral ascent | 131 |
| Storm magic | 137 | Four seasons | 142 |

**Нужен ComfyUI 0.33.0 или новее.** Именно там `embedding:` начал доходить до
токенизатора MiniMax-H3; на всём, что старше, токен читается обычными словами.
Нода проверяет вашу версию и говорит об этом.

![Нода MiniMax-H3 Effect Embeddings рядом с нодой Show Any. Над заголовком два бейджа: собственный бейдж пака и серый с надписью «bypass». По правому краю три выхода: prompt, проложенный проводом к соседней ноде, затем tokens и findings. Ниже текстовое поле с T2VA-промтом из трёх полей: integrated_multimodal_description: A cat walks along a fence at dusk, overall_soundscape: Wind in the grass, non_diegetic_music: N/A. Под ним виджет placement со значением «start of the description». Дальше сетка из десяти строк, в каждой галочка, имя и цена: Art is explosion 50 tok, Blooming flowers 123 tok, Bullet time 94 tok, Dark magic 59 tok, Fire breath 118 tok, Four seasons 142 tok, Kiss camera 97 tok, Spiral ascent 131 tok, Storm magic 137 tok, Truman show 90 tok. Отмечены три — Art is explosion, Dark magic и Fire breath, — их квадратики зелёные, с белой птичкой. Под сеткой строка «3 selected, 227 tokens», ещё ниже виджет bypass со значением false. Кнопки скачивания нет. В ноде Show Any результат: три токена подряд сразу после метки integrated_multimodal_description и перед самой фразой, затем overall_soundscape и non_diegetic_music ровно так, как были написаны](docs/node_effect_embeddings.png)

*Ничего не загружается и не генерируется — нода вписывает в текст сотню символов
и читает десять файловых заголовков. Кнопки скачивания нет, а не выключена: все
десять файлов на диске, качать нечего, и незачем занимать под это строку. Цены в
правой колонке прочитаны из самих файлов, а не из таблицы внутри пака, и 50 + 59
+ 118 — это те самые 227 под сеткой. Здесь placement «start of the description»,
поэтому три токена открывают поле описания, а не стоят над ним, — и поэтому
пустые строки ниже дойдут до H3 одиночными пробелами, о чём вывод `findings` и
сообщает вслух.*

| Вход | Зачем |
| --- | --- |
| `prompt` | Промт, в который вставляются токены — выход ноды-писателя, загруженный файл, набранный текст. Проходит насквозь символ в символ, кроме самих токенов. |
| `placement` | Куда именно вставлять: в начало описания, в самое начало промта или в конец. Это не вопрос вкуса — см. ниже. |
| `effects` | Сетка из десяти: галочка, имя, цена в позициях и пометка на тех, которых ещё нет на диске. |
| `bypass` | Пропустить `prompt` насквозь и не вставлять токенов. `tokens` вернётся 0, `findings` пустым, заголовки файлов читаться не будут. Переключатель вынесен бейджем в заголовок ноды, как у писателей. |

| Выход | Что это |
| --- | --- |
| `prompt` | Промт с токенами. |
| `tokens` | Сколько позиций займут выбранные эффекты — прочитано из заголовка каждого файла, то есть настоящее число, а не табличка внутри пака. |
| `findings` | Что нода заметила, по строке на находку. Пусто, когда говорить не о чем. |

Файлы качает кнопка на ноде — в собственную папку ComfyUI `models/embeddings`,
единственную, куда смотрит `embedding:`. Все десять весят около десяти мегабайт.
Повторный прогон токены не накапливает: нода сначала вынимает свои, потом
вставляет заново — поэтому снятая галочка убирает эффект из промта, где он уже
был.

#### Все способы сломаться здесь — молчаливые

Ради этого нода и существует, а не строка в текстовом поле. Токен, который
ComfyUI не смог использовать, выбрасывается с одной строкой в консоли и без
единого другого признака: видео сгенерируется — просто без эффекта, ровно как
промт, который эффекта и не просил. Три способа это получить, и все три теперь
ищет самопроверка:

- **Заглавная буква.** Проверка — это `word.startswith("embedding:")` и ничего
  больше, так что `Embedding:` не меньшая ошибка, чем опечатка. (Редьюсер раньше
  ставил её сам, поднимая первую букву предложения. Больше не ставит.)
- **Токен, приклеенный к предыдущему слову.** Токены ищутся после пробельного
  символа или в самом начале текста. `a cat.embedding:x` — это четыре обычных
  слова.
- **Точка в конце имени.** Запятую ComfyUI отрежет и простит, точку будет искать
  как часть имени файла. Про эту стоит знать точно, потому что на Windows
  ловушка прячется: путевой слой там отбрасывает финальную точку, так что
  `embedding:minimaxh3_dark_magic.` на Windows загрузится, а на Linux и macOS
  молча потеряет эффект. Промт, которым вы поделились, — не тот промт, который
  запустят.

Веса тоже не работают: `(embedding:x:0.8)` читается буквальным текстом, потому
что H3 токенизирует с отключёнными весами. Попросить эффект послабее нельзя.

#### Место вставки — не вопрос вкуса

Токенизатор ComfyUI склеивает всё, что идёт после токена, в одну строку: текст
после `embedding:name` и до следующего токена проходит через
`' '.join(text.split())` раньше, чем его увидит H3. Токен в начале промта тем
самым превращает все пустые строки между полями ниже в одиночные пробелы.

`start of the description` — по умолчанию — открывает само поле описания, сразу
после метки. Это поле, куда писатели кладут сцену и которое читают все ноды
ниже по графу, и именно его эффект и должен был менять. Поля под ним схлопнутся.

`end of the prompt` ставит токены последними. После них ничего нет, схлопывать
нечего, и промт доезжает до H3 с целыми разделителями полей.

`top of the prompt` ставит их выше всего, перед строкой выравнивания.

У промта совсем без меток полей нет описания, которое можно открыть, — тогда
вставка уезжает наверх, и нода говорит об этом на выходе `findings`.

Если пользуетесь редьюсером, знайте: токен выше первой метки поля он выбросит
вместе со всей головой промта. Редьюсер оставляет описание и звук, а не то, что
стоит над ними. Токен внутри описания переживает проход.

Чего здесь **не** известно: складываются ли два эффекта во что-то осмысленное.
Это отдельные куски закодированного промта, и просьба про спиральный подъём
сквозь четыре сезона — это просьба к модели их примирить. Нода разрешает любое
сочетание, потому что запрещать нет оснований; стоит ли оно генерации, выяснять
вам.

### MiniMax-H3 LoRA Triggers

LoRA, обученная со словом-триггером в префиксе подписи, не делает ровно ничего,
пока этого слова нет в промте. Эта нода держит список таких слов на себе, вместе
с воркфлоу, и возвращает их в готовый промт.

Выше по графу о них не знает никто. Писатели пишут прозу, редуктор её сокращает,
самопроверка её судит — и слово, вся работа которого в том, чтобы через три ноды
проснулся адаптер, ровно то, что все трое выбрасывают. Значит, его дописывают
руками после каждого прогона и теряют на следующем. Это единственное место во
всём конвейере, где готовый текст правят вручную, и единственная часть промта,
которую нельзя сгенерировать заново.

Одна строка — один адаптер: грип, галочка, имя для себя, слова, которые уйдут в
промт дословно, короткая пометка о том, куда они уйдут, и крестик, чтобы убрать
строку. В строке может стоять несколько слов через запятую — это то, что нужно
адаптеру с несколькими триггерами; проверяются и вставляются они по одному, так
что строка никогда не продублируется наполовину. Триггерам, которым нужны разные
части промта, нужны разные строки.

Строка перетаскивается за грип. Порядок — не украшение: строки с одним
размещением уходят в промт в порядке списка, и в этом же порядке они читаются.

![Нода MiniMax-H3 LoRA Triggers рядом с нодой Show Any. Над заголовком два бейджа: собственный бейдж пака и серый с надписью «bypass». По правому краю три выхода: prompt, проложенный проводом к соседней ноде, затем added и findings. Ниже текстовое поле с T2VA-промтом из трёх полей: integrated_multimodal_description: A cat walks along a fence at dusk, overall_soundscape: Wind in the grass, non_diegetic_music: N/A. Под ним две строки списка. В каждой грип для перетаскивания, зелёная галочка с белой птичкой, поле имени, поле пошире под слова, пометка размещения и крестик: «Facial realism» со словами «ohwx face» и пометкой body, и «Neon» со словами «neon glow, wet asphalt» и пометкой top. Обе пометки стоят в одну колонку, и обе пары полей тоже. Под ними строка «2 on of 2», ещё ниже виджет bypass со значением false, затем кнопка во всю ширину — «add a trigger». В ноде Show Any результат: «neon glow, wet asphalt» отдельной первой строкой, пустая строка, затем integrated_multimodal_description, открывающийся словами «ohwx face.» перед самой фразой, и overall_soundscape с non_diegetic_music ровно так, как были написаны](docs/node_lora_triggers.png)

*Две строки, два размещения, один прогон. «ohwx face» открыл описание — после
метки и перед прозой, то есть ровно в той позиции, в которой триггер из префикса
подписи и обучался, — а вторая строка ушла наверх, неся сразу два триггера:
проверены и записаны они по отдельности. Вынимать ради этого ничего не
пришлось: прогони тот же текст ещё раз, и `added` вернётся пустым, потому что
оба слова уже на месте. Пометки размещения стоят фиксированной колонкой — за
счёт этого поля во всех строках начинаются и кончаются на одном пикселе; порядок
записи меняется перетаскиванием строки за грип.*

| Вход | Для чего |
| --- | --- |
| `prompt` | Промт, в который дописываются слова, — вывод ноды-писателя, вывод редуктора, загруженный файл, набранное вручную. Передаётся символ в символ, кроме самих вставленных слов. |
| `triggers` | Список. Грип, галочка, имя, слова, пометка размещения, крестик; кнопка под списком добавляет строку. Сохраняется вместе с воркфлоу. |
| `bypass` | Пропустить `prompt` насквозь и ничего не дописывать. Текст выйдет ровно таким, каким вошёл; `added` и `findings` вернутся пустыми. Переключатель вынесен бейджем в заголовок ноды, как у писателей. |

| Выход | Что это |
| --- | --- |
| `prompt` | Промт со вставленными словами. |
| `added` | Слова, которые этот прогон действительно вставил, через запятую. Пусто, когда все включённые триггеры уже были на месте, — на повторном прогоне это нормально. |
| `findings` | Что нода сделала и что заметила, по строке на пункт. Пусто, когда сказать нечего. |

Если гонять ноду через API, а не через интерфейс: виджет принимает тот самый
JSON-массив объектов `{name, words, where, on}`, который пишет интерфейс, — и ещё
две формы попроще, потому что именно их пишут первыми. Обычный массив строк
читается как список триггеров, а текст, который не JSON вовсе, — как слова одной
строки.

#### Куда уезжает каждое слово

Пометка в строке короткая намеренно. Видеть с одного взгляда, куда что уедет,
ничего не раскрывая, — половина смысла списка, а пять полных выпадающих списков
не влезут в ширину ноды ни при какой разумной ширине. Клик по пометке открывает
выбор.

| Пометка | Размещение | Куда попадает |
| --- | --- | --- |
| `body` | `start of the description` | Открывает само поле описания, сразу после метки. |
| `top` | `top of the prompt` | Над всем, перед метками полей. |
| `sound` | `start of the soundscape` | Открывает `overall_soundscape`, сразу после метки. |
| `music` | `start of the music` | Открывает `non_diegetic_music`, сразу после метки. |
| `end` | `end of the prompt` | В самом конце, после всего. |

Описание стоит по умолчанию и обычно оно и есть верный ответ: это поле, куда
писатели кладут сцену, его же читают все ноды ниже, и именно в этой позиции
обучался триггер из префикса подписи. Верх и конец — позиции, а не поля, поэтому
только они и работают на тексте вовсе без меток.

Набор полей у задач разный — три у T2VA, шесть у Ref2VA, — а нода видит только
текст, и текст не говорит, какая задача его написала. Поэтому размещение, для
которого в этом промте поля нет, — не ошибка: слова уходят наверх, и `findings`
об этом говорит.

Поле, которое есть, но в котором стоит `N/A`, получает отдельную пометку. В
промтах Ref2VA сплошь и рядом `non_diegetic_music: N/A`, и слова, поставленные в
начало такого поля, оставляют поле, которое что-то называет и тут же говорит, что
ничего этого нет. Нода пишет об этом и больше не делает ничего. Заменить маркер
значило бы заставить поле утверждать, что музыка есть и называется «ohwx face», —
а это поле читает самопроверка; выбросить слова значило бы отменить размещение,
которое вы выбрали. И то и другое — решения о вашем промте, а их эта нода не
принимает.

Одно отличие от эффект-эмбеддингов, которое стоит знать, потому что привычка
оттуда сюда не переносится: обычные слова не схлопывают переносы строк под собой.
Токенизатор ComfyUI склеивает в одну строку всё, что идёт после токена
`embedding:`, — но только после него. Верх промта здесь безопасен.

#### Нода ничего не вынимает

Здесь она расходится с `MiniMax-H3 Effect Embeddings`, и причину стоит назвать.
Те десять токенов — строки, известные заранее, поэтому нода умеет выкусить свои
собственные перед вставкой: снятая галочка там убирает эффект. Здесь слова ваши.
Вырезать триггер, написанный как `detail`, значит выесть дыру в прозе, в которой
он стоял.

Поэтому правило другое — присутствие, а не удаление:

- Слово, которое в тексте уже есть, **второй раз не вставляется**: поиск по
  границам слова и без учёта регистра, так что `man` не находится внутри `woman`,
  а `Ohwx man` засчитывается как `ohwx man`. Об этом говорится в `findings`.
- **Снятая галочка перестаёт добавлять. Она не убирает.** Слово могло прийти и от
  писателя, а удалять чужое — не дело этой ноды.

Поэтому повторный прогон того же текста не меняет ничего, а это ровно то, что
ноде нужно, чтобы безопасно стоять в хвосте графа, который гоняют раз за разом.

#### `bypass`, а не Ctrl+B

У обеих нод есть собственный переключатель `bypass` пака, с тем же фиолетовым
бейджем в заголовке, что у писателей: клик по бейджу или галочка в виджете — и
промт идёт насквозь нетронутым.

Штатный бипасс ComfyUI на такой ноде делает не то. Он подставляет каждому выходу
вход по совпадению типа, а выходов здесь три при одном сокете. Замерено на ноде
триггеров: `prompt` доходит, в `findings` приезжает весь промт целиком, как будто
это находка, а `added` теряет связь совсем — и нода снизу остаётся без входа. У
ноды эффектов `tokens` — INT, и подставить под него нечего, так что связь уходит
туда же. Ни о чём из этого нигде не сообщается.

#### Ставить в самом хвосте графа

После редуктора и после `MiniMax-H3 Prompt Check`. Триггеры добавляют слова, а
проверка считает слова и сравнивает с потолком: нода, которая удлиняет текст
после замера, — это нормально, а та, что удлиняет его до замера, с ним спорит. И
всё, что переписывает промт ниже по графу, слова снова потеряет — а это ровно тот
провал, ради которого нода и существует.

#### Чего нода не обещает

Только LoRA, обученные с триггером в префиксе подписи. Слайдерам и
turbo-ускорителям слово не нужно вовсе — они работают силой, — а на большинстве
машин именно таких среди установленных большинство. Список триггеров им ничем не
поможет, и это не изъян списка.

Из файлов LoRA не читается ничего, и это вывод, а не недоделка. Из сорока четырёх
адаптеров на машине автора `trigger_word` нашёлся у одного, `trigger_words` — у
второго и пустой, `ss_tag_frequency` — ни у одного. Заголовок safetensors пак
читать умеет, так что дело не в стоимости: данных просто нет. А триггеров у
адаптера бывает несколько там, где в метаданных место ровно под одно. Список
ведёте вы — и это же и делает его тем, что стоит вести.

### MiniMax-H3 Reference Adapter

Писатели принимают по референсу на слот, и именно поэтому работает полоска под
ними: у каждого ассета есть квадратик, роль и выключатель. По той же причине они
недостижимы для ноды, которая отдаёт свои референсы *вместе* — батч картинок,
список от загрузчика папки, связка, собранная чужим паком. Одному значению,
внутри которого лежит много, некуда воткнуться.

Эта нода — стык. Собранное на входе, разложенное на выходе: девять сокетов под
картинки, три под клипы, три под звуки — ровно то, что принимают писатели, так
что полоска продолжает работать как работала.

![Нода MiniMax-H3 Reference Adapter. Слева два входа: items, ни к чему не подключённый, и bundle, к которому идёт провод из-за края кадра. Справа шестнадцать выходов: picture_1 — picture_9, video_1 — video_3, audio_1 — audio_3 и summary. Первые три картиночных выхода уходят в три ноды предпросмотра с тремя не связанными между собой фотографиями. Единственный виджет, split_batches, стоит в true; под ним строка «3 picture(s), 0 clip(s), 0 sound(s)». Время прогона ноды — 0.014s](docs/node_ref_adapter.png)

*Один провод внутрь, три наружу — и тринадцать выходов простаивают, что здесь обычное дело, а не признак недоделанного. Прогон занял 14 миллисекунд, потому что тут ничего не грузится, не декодируется и не описывается: три фотографии прошли насквозь как приехали. Строка под виджетом — тот же текст, что и на выходе `summary`, так что прочитать, что приехало и куда легло, можно прямо на ноде, ничего не подключая к предпросмотру.*

| Вход | Зачем он |
| --- | --- |
| `items` | Референсы, приехавшие вместе. Принимает любой тип, потому что ноды, отдающие подборки, чаще всего типа не объявляют; род каждого элемента выясняется по самому значению, а не по проводу, и всё, что не картинка, не клип и не звук, пропускается и попадает в счёт. |
| `bundle` | Связка референсов из чужого пака, если она у вас есть. Её картинки, клипы и звуки читаются раньше всего, что пришло на `items`. Клип, приехавший кадрами, уходит клипом, а не картинками, и его парная звуковая дорожка — внутри него; дорожка без пары с клипом остаётся самостоятельным звуком. |
| `split_batches` | Станет ли батч картинок референсом за кадр или останется одним референсом из нескольких кадров. |

Последний переключатель — настоящий выбор, а не формальность. Разобранные: шесть
кадров это шесть вещей, которые переиспользует видео, каждая описана и
пронумерована отдельно. Оставленные: это одна вещь, увиденная шесть раз,
описанная один раз, — так устроен клип. Выключайте, когда батч это кадры одного
шота.

Девять, три и три — потолок Ref2VA, столько места и есть. Всё сверх попадает в
выход `summary`, а не пропадает молча, — вместе со всем, что было пропущено.
Незанятые выходы отдают пустоту, а писатель такие и так пропускает, так что
подключить сокет, оказавшийся пустым, ничем не грозит: втыкайте все девять, а
незаполненные пусть простаивают.

Это отдельная нода по причине, которую стоит знать, потому что выбора тут не
было. Чтобы получать настоящий список ComfyUI, нужно объявить `is_input_list`, а
этот флаг не про один вход: он переписывает форму *каждого* аргумента, который
получает нода. На писателе он изменил бы то, как приезжают промт, длительность и
опции. Поэтому он живёт здесь, на ноде, которой нечего на этом терять.

### MiniMax-H3 Reference Slots

`MiniMaxH3ReferenceToVideo` — собственная нода ComfyUI для референсного режима H3
— нумерует референсы по сокету, на который они пришли: первый картиночный сокет
это `<Picture 1>`, первый звуковой — `<Audio 1>`. Писатели здесь нумеруют свои
по полосе, а полосу можно перетаскивать. Подключите одни и те же ассеты к обеим
нодам руками, и два порядка совпадают ровно до первого перетащенного квадратика:
дальше промт описывает один голос, видео получает другой, и ничто на экране об
этом не говорит. Выключенный референс ломается так же, только наоборот — из
промта пропал, в видео остался.

Поэтому каждый писатель, принимающий референсы, отдаёт то, что пронумеровал, на
последнем выходе `references`, а эта нода кладёт каждый референс на сокет с его
номером.

![Нода MiniMax-H3 Reference Slots между Universal Writer и MiniMax H3 Reference to Video. Писатель в режиме Ref2VA, к нему подключены четыре референса, а полоса читается так: pic 1 над ref_0, aud 1 над ref_2, pic 2 над ref_1 и aud 2 над ref_3; его последний выход, references, идёт в ноду Reference Slots. Справа у неё девятнадцать выходов — picture_1 — picture_9, video_1 — video_3, video_audio_1 — video_audio_3, audio_1 — audio_3 и summary, — и каждый медийный выход подключён напрямую к входам генератора в том же порядке: ref_image_0 — ref_image_8, ref_video_0 — ref_video_2, ref_video_audio_0 — ref_video_audio_2 и ref_audio_0 — ref_audio_2. Переключатель soundtracks стоит в false; под ним «2 picture(s), 0 clip(s), 2 sound(s)» и «picture_1 ← ref_0». Писатель отработал за 0.011s, нода слотов — за 0.019s](docs/node_reference_slots.png)

*Подключено один раз, насквозь, и больше не трогается: расстановка происходит в полосе писателя. Квадратики читаются как ref_0, ref_2, ref_1, ref_3 — значит, `picture_2` несёт ref_1, а `audio_1` несёт ref_2, в какой бы сокет их ни воткнули, и генератор нумерует их ровно так же, как промт. Пустые выходы ничего не стоят: пустой сокет генератор пропускает. Девятнадцать миллисекунд — потому что картинки и звуки проходят как есть, а клипа для декодирования не было.*

```
Universal Writer ─ references ─▶ Reference Slots ─ picture_1 ─▶ ref_image_0 ┐
                                                 ─ picture_2 ─▶ ref_image_1 ├ MiniMaxH3ReferenceToVideo
                                                 ─ audio_1   ─▶ ref_audio_0 ┘
```

Один раз подключите напрямую и по порядку — `picture_N` в N-й `ref_image`,
`video_N` в N-й `ref_video`, `video_audio_N` в N-й `ref_video_audio`, `audio_N` в
N-й `ref_audio`, — а дальше расставляйте всё в полосе.

| Выход | Что это |
| --- | --- |
| `picture_1` … `picture_9` | Картинки в порядке писателя. |
| `video_1` … `video_3` | Клипы, кадрами в 24 fps. |
| `video_audio_1` … `video_audio_3` | Собственный звук клипа, в пару по номеру — когда писатель его засчитал или для всех клипов при включённом `soundtracks`. |
| `audio_1` … `audio_3` | Звуки, как пришли. |
| `summary` | Что лежит на каком выходе и из какого слота писателя пришло, плюс всё, что не поместилось. |

**Выключенный референс выпадает с обеих сторон.** В `references` его нет вовсе,
поэтому его сокет ничего не отдаёт, а `MiniMaxH3ReferenceToVideo` пустой сокет
пропускает и смыкает нумерацию вокруг пропуска — ровно так же, как писатель
сомкнул свои метки.

**Субъект идёт после картинок.** В Universal Writer картинка с бейджем «субъект»
пишется как `Subject N` и никогда как `<Picture N>`. Сокета для субъектов у
генератора нет, и он нумерует такую картинку как любую другую, поэтому она
встаёт после последней картинки — туда, где номера промтом не заняты.

**Клипы декодируются здесь, а не в писателе.** Писатель работает независимо от
того, подключено ли что-нибудь к `references`, а пятнадцать секунд 1080p во
float-кадрах — это несколько гигабайт, поэтому клип едет тем VIDEO, каким
пришёл, и ничего не стоит, пока эта нода не запустится. Здесь он читается в 24
fps — у клипа в 30 fps пропадает каждый пятый кадр, у клипа в 12 fps каждый кадр
показывается дважды, — масштабируется до холста, который
`MiniMaxH3ReferenceToVideo` дал бы ему и сама, и обрезается на пятнадцати
секундах — самом длинном референсе, на который она рассчитана. Обрезанное VIDEO
читается по своей обрезке. Батч картинок, который писателю велели читать как
клип, уже состоит из кадров и проходит как есть.

**Звук клипа уходит дальше, когда его засчитал писатель.** Генератор даёт звуку
клипа собственный `<Audio N>`, который нумеруется раньше всех отдельных звуков.
Щёлкните по подписи клипа в стрипе писателя — или поставьте галочку с динамиком на
строке `reference_video` у Universal Rewriter, — и писатель пронумерует этот звук
так же, опишет его или услышит и отметит это в `references`. Эта нода тогда кладёт
его на `video_audio_N` клипа, и промт сходится с видео. Клипу без звуковой дорожки
класть нечего, а писатель уже сказал об этом и номера ему не дал.

**`soundtracks` отдаёт звук всем клипам, что бы ни решил писатель.** По умолчанию
он выключен. Клип, который писатель оставил беззвучным, тогда получает `<Audio N>`,
которого в промте нет, и `<Audio 1>` в промте — уже не тот звук, что на
`audio_1`; `summary` скажет, на сколько сдвинулись метки.

Каждый писатель отдаёт то, что названо в его промте, и ничего из того, чего он не
читал:

| Нода | Что несёт `references` |
| --- | --- |
| Universal Writer | Каждый включённый квадратик, в порядке полосы. На T2VA — ничего. |
| Prompt Rewriter Omni | Каждый включённый квадратик, в порядке полосы. На T2AV — ничего. |
| Universal Rewriter | Ничего на вкладке 27B и на T2VA. На вкладке 8B — кадры, которые читает задача. На вкладке Omni — те же кадры, а на Ref2VA — каждую включённую строку в порядке строк. |

Выход каждый раз собирается из текущих входов — в том числе при `bypass`, при
выборе из библиотеки и при `repeat_last` — и нигде не хранится: библиотека и
память ноды держат текст, а сохранённый промт возвращается с теми референсами,
что подключены сейчас.

> **Это не Reference Adapter.** Та нода кормит писателей: раскладывает всё
> пришедшее по родам и отдаёт клип как VIDEO, потому что писатель описывает
> именно VIDEO. Эта кормит генератор, который берёт клип кадрами, а звук клипа —
> на отдельном сокете. По той же причине у `references` собственный тип: воткнутый
> в ноду, которая ждёт связку чужого пака, клип приехал бы как VIDEO и упал бы уже
> внутри генератора.

> **Одну цепочку это не покрывает.** Universal Writer с блоком на `previous`
> сначала считает метки этого блока, а его `references` содержит только
> собственные референсы и нумерует с единицы. Когда выход идёт в генератор,
> держите по одному писателю на шот.

> **На RefMod нода останавливается.** `MiniMaxH3ReferenceToVideo` сама кодирует
> картинки, клипы и звуки, которые ей дали, а [RefMod](#minimax-h3-refmod) уже
> закодирован, и сокета для него нет. Нода говорит об этом и называет
> [MiniMax-H3 Reference Encode](#minimax-h3-reference-encode) — та принимает те же
> `references` и заменяет обе ноды.

### MiniMax-H3 RefMod

RefMod — это референс, закодированный один раз и сохранённый: латент H3 VAE
картинки, клипа, стопки фотографий или голоса в файле `.safetensors` в
`ComfyUI/models/refmods`, с заголовком, где сказано, что это. Формат —
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod).
Здесь читаются RefMod'ы, которыми делятся на Hugging Face, однофайловые бандлы
того пака и файлы, которые пишет
[ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder).
Создавать и править их — дело тех двух паков; этот ими только пользуется.

Нода выбирает один и отдаёт его в полосу писателя. Вся цепочка, от файла до
видео, собрана в шаблоне **8 - RefMods to video**.

| Вход | Зачем |
| --- | --- |
| `refmod` | Запись — по её пути внутри `models/refmods`. Внешность и голос, сохранённые вместе — один бандл или пара `<имя>_visual` и `<имя>_audio`, — это одна запись. Нажмите **R** в графе, чтобы увидеть добавленные файлы. |
| `strength` | Насколько сильно генератор ему следует. 1 — как сохранено. Меньше — латент подмешивается к сильно размытой копии себя: цвет, композиция и общая форма остаются, детали уходят, а картинки, которые показывают текстовому энкодеру H3, ослабляются так же. Писатель всё равно читает его в полную силу. |
| `vae` | Видео-VAE H3. Большинство файлов своих картинок не хранят — с Hugging Face не хранит ни один, — а писателю, чтобы описать референс, надо его увидеть, поэтому внешность декодируется здесь, один раз. |
| `audio_vae` | Аудио-VAE H3 — то же для голоса, чтобы писатель мог его услышать. |

| Выход | Что это |
| --- | --- |
| `visual` | Картинка, клип или стопка фотографий. Пуст, если в записи только голос. |
| `audio` | Голос. Пуст, если голоса в записи нет. |

Каждый выход идёт в свой квадратик полосы
[Universal Writer](#minimax-h3-universal-writer) или
[Prompt Rewriter Omni](#minimax-h3-prompt-rewriter-omni-sees-and-hears), где он
помечен `mod`:

![Universal Writer, у которого на ref_0 и ref_1 два выхода одного RefMod'а, а ref_2 пуст и без галочки. В полосе два квадратика с пометкой mod: зелёный vid над ref_0 с декодированной фотографией персонажа, значком воспроизведения и цифрой 1, и фиолетовый aud над ref_1 с динамиком, именем записи Yoda, волной и цифрой 1, у каждого полоска «+ instr». Под ними строка подсказки и переключатель задач, в котором горит Ref2VA](docs/refmod_support_universal_writer.png)

![Prompt Rewriter Omni с теми же двумя выходами RefMod'а на ref_0 и ref_1: зелёный квадратик vid и фиолетовый aud с именем Yoda, оба с пометкой mod и номером 1, над строкой «drag to reorder - that order numbers the labels - click a square to switch it off - a clip's label: its sound». Ниже пять задач с горящей REF2AV, восемь прямоугольников соотношений сторон в пропорции с выбранным 16:9 и длительность 10.0](docs/refmod_support_omni_rewriter.png)

Дальше:

- **Он называется тем, чем его называет файл.** Клип или стопка фотографий — это
  `<Video N>`, голос — `<Audio N>`, что бы ни было на бейдже: так его назовёт
  генератор. Одиночную картинку по-прежнему можно сделать субъектом, но не клипом,
  а собственного звука, который можно было бы включить, у RefMod'а нет.
- **Он описывается по своим картинкам,** как любой референс. То, что о нём сказано
  в файле, едет подсказкой после вопроса роли: стопка описывается как один субъект
  на отдельных фотографиях, а не как последовательность событий, сохранённое имя
  субъекта дано для использования, а сохранённое описание внешности или голоса
  предложено держаться там, где картинки с ним согласны. Universal Writer пишет
  голос, который никто не декодировал, по сохранённому описанию, если оно есть.
  Адаптер Omni слышит звук сам, поэтому такой голос он отклоняет и просит
  подключить `audio_vae`.
- **Писателю Universal Writer сказано, что говорит файл,** — в блоке
  референсов, перед подписью: что стопка — фотографии одного субъекта, а не
  съёмка, для чего RefMod сохранён (голос — это только голос, а не то, что
  звучит за ним) и — когда внешность и голос, сохранённый с ней, оба стоят в
  полосе, — под какой меткой вторая половина:
  `Audio 1: (saved as a reference for a voice, the voice alone and not what
  plays behind it; the voice of the subject in Video 1) ...`. Капшенер читает
  их порознь, поэтому ни одна подпись не говорит, что они вместе, а голос,
  записанный поверх музыки, иначе принимается за музыку и уходит в саундтрек.
  Генератор всё равно получает запись целиком, с музыкой: промпт лишь
  перестаёт просить её ещё.
- **Дальше уходит сам RefMod** — в `references`, вместе с латентом, в
  [MiniMax-H3 Reference Encode](#minimax-h3-reference-encode).

Стопка декодируется по одной фотографии, потому что каждая кодировалась отдельно,
а клип — целиком. В обоих случаях это происходит один раз на выбор: ComfyUI
хранит выход ноды, а вместе с ним и картинки.

На сэмплинге RefMod не бесплатен. Подпись ноды говорит, во что он обходится, и
некоторые из выложенных тяжёлые: стопка из десяти фотографий в 1024x1824 — это
18 240 токенов на каждом шаге сэмплинга, столько же, сколько восемнадцать
референсных картинок в 1344x768 или референсный клип на две с половиной
секунды. Уменьшение `strength` меняет то, насколько ему следуют, а не его цену.

### MiniMax-H3 Reference Encode

`MiniMaxH3ReferenceToVideo` кодирует референсы, которые ей дали. RefMod уже
закодирован: генератору нужен его латент отдельным референсным блоком, а
текстовому энкодеру H3 — по-прежнему его картинки. Поэтому эта нода делает то, что
[Reference Slots](#minimax-h3-reference-slots) и `MiniMaxH3ReferenceToVideo`
делают вместе, а для RefMod'а — то единственное, чего не может ни одна из них.

```
Universal Writer ─ references ───────▶ references ┐
                 ─ rewritten_prompt ─▶ prompt     ├ MiniMax-H3 Reference Encode ─ positive ─▶ сэмплер
clip, vae, audio_vae ────────────────────────────┘                              ─ latent   ─▶ сэмплер
```

Настройки у неё от `MiniMaxH3ReferenceToVideo` — `width`, `height`, `length`,
`ref_image_size`, — плюс `references` писателя и переключатель `soundtracks` от
Reference Slots. Отдаёт она `positive`, пустой `latent` для сэмплинга и `summary`:
что вошло, в каком порядке и во что обходится.

**Порядок — это весь договор.** У DiT H3 нет индекса референса: блок сопоставляется
со своей меткой по месту, на котором стоит. Поэтому всё идёт в порядке
`MiniMaxH3ReferenceToVideo` — все картинки, потом все клипы, у каждого собственный
звук прямо перед ним, потом все звуки, — а RefMod занимает место, которое
пронумеровал писатель. Картинка, клип или звук готовятся хелперами самой той ноды,
с её масштабированием и обрезкой, так что полоса без RefMod'ов кодируется в то же
условие, что и через Reference Slots.

**Что текстовый энкодер видит у RefMod'а.** Картинку — как есть. Клип — по два
кадра в секунду, как любой референсный клип. Стопку — каждую четвёртую фотографию,
по две в блоке: так по умолчанию делает пак, который хранит такие картинки, и это
близко к тому, что показали бы два кадра в секунду того же латента, проигранного
как клип. Файлу, который хранит свои картинки, показывают их; файл, который не
хранит, декодируется здесь через `vae`, если у его ноды RefMod VAE не было.

`vae` и `audio_vae` для обычных референсов работают по правилу
`MiniMaxH3ReferenceToVideo`: без них такие референсы доходят только до текстового
энкодера, и `summary` говорит, сколько их было. RefMod доходит до генератора в
любом случае.

### MiniMax-H3 Prompt Presets

Тысяча готовых промтов MiniMax-H3 лежит внутри пака, и эта нода отдаёт один из
них. Кнопка **Pick a preset** открывает окно отбора: двадцать манер съёмки,
двадцать тем, форма кадра, число планов и наличие реплик — всё это теги,
сужающие тысячу, плюс поиск по самим словам. В каждой строке кадр того клипа, под
который промт написан; клик по кадру проигрывает клип.

Это готовые T2VA-промты в том формате, который пишут здешние писатели, — те же
три поля с метками, — и потому они годятся в две стороны. Провод от `prompt` в
генератор — промт идёт как есть; провод во вход `prompt` писателя — он становится
отправной точкой для переписывания. Этот выбор — провод, а не виджет, поэтому
здесь и отдельная нода.

![Нода MiniMax-H3 Prompt Presets рядом с нодой Show Any. Справа по ноде заготовок идут восемь выходов, первый prompt, последний source. Виджет preset показывает 000014; под ним две кнопки, «Pick a preset» и «Save to the library»; ниже — кадр клипа: курьер на мотоцикле на выжженном солнцем шоссе в пустыне, а рядом весь промт, все три поля с метками читаются целиком. Выход prompt подключён к ноде Show Any, которая печатает те же три поля обычным текстом. Прогон занял 0.030s](docs/node_prompt_presets.png)

*Заготовка стоит на самой ноде, а не прячется за диалогом: кадр клипа, под который она написана, и весь её текст видны на лице ноды ещё до всякого прогона. Сам прогон занял тридцать миллисекунд и ничего не загружал — из `prompt` вышло ровно то, что нода и так показывала.*

| Выход | Что это |
| --- | --- |
| `prompt` | Всё целиком, три поля с метками, ровно как выдал бы писатель отсюда. |
| `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music` | Те же три, по отдельности. |
| `seconds`, `width`, `height` | Клип, под который промт написан. Все они около пяти секунд, и времена склеек внутри текста считаны от этого. |
| `source` | Номер, оба адреса, по которым клип можно посмотреть, и кому причитается указание авторства — чтобы вывести в текстовый превью, когда воркфлоу уходит дальше. |

**Ничего не скачивается ни при установке, ни во время прогона.** Промты, их теги
и по одному кадру в 256 пикселей на каждый — это 6 МБ внутри пака; они читаются с
диска и распаковываются при первом обращении, так что сессия, где окно не
открывали, не платит ничего, а прогон занимает единицы миллисекунд. В сеть уходит
единственное — клип в предпросмотре, и только по клику на кадр: сперва
huggingface.co, а если тот не отвечает — hf-mirror.com, адрес, отвечающий из
материкового Китая. Без сети кадры остаются на месте, пропадает только видео.

**Save to the library** кладёт копию в один из ваших наборов промтов — через тот
же редактор, которым пользуется библиотека: имя, описание, группы и проверка по
мере набора. Дальше это обычная запись: правится, лежит в ваших группах, доступна
любому писателю через окно библиотеки. В описании копии сказано, откуда она
взялась. Встроенной заготовки всё это не касается.

Одна оговорка, которую стоит сказать один раз: это T2VA-промты под пятисекундный
клип. Отдать такой промт задаче с референсами или сильно более длинному видео
никто не мешает, и никто не пожалуется, — но текст не описывает ни того, ни
другого.

Промты и клипы, которые они описывают, — работа ostris, перенесённая сюда с его
разрешения и с указанием авторства:
[ostris/minimax_h3_1k](https://huggingface.co/datasets/ostris/minimax_h3_1k).
Промты — его собственные и по сути MIT; клипы — выход MiniMax-H3, а та
лицензия запрещает обучать на нём модели, и нарезанные из них здесь
кадры несут то же ограничение. Теги манеры съёмки и темы — прочтение
подборки проектом
[H3 Atlas](https://cohub.live/baize/video-altas/w/h3-atlas). Вся тысяча проходит
[самопроверку](#ответ-проверяется) без единой находки — что заодно и есть
ответ на вопрос, как выглядит хороший промт для H3.

### Виджет duration

Он есть у каждой ноды пака, которая пишет или читает промт, и на всех девяти
означает одно и то же: сколько секунд должен длиться целевой клип. Это подсказка,
а не настройка — до модели он доходит одной строкой текста, `duration: 10s`, и
решает, сколько планов запланирует переписанный промт и в каком темпе их
расставит. Сколько клип продлится на самом деле, решают ниже по графу — тот, кто
его сэмплит.

**Он принимает `FLOAT` и `INT` на один и тот же сокет.** Сам виджет вещественный,
с точностью до десятой, потому что времена склеек в ответе пишутся до тысячных —
`At 00:02.378, the camera cuts to` — и попросить 7.5 секунды так же обычно, как
попросить 8. Но большинство графов носит длину целым числом, поэтому сокет берёт
и то, и другое: `INT`-примитив, нода, считающая кадры, или `FLOAT` с
математической ноды подключаются напрямую, без конвертера между ними. Две
Universal-ноды рисуют его слайдером, остальные — обычным полем ввода; диапазон и
меню при этом одинаковые.

**Верхний край двигаете вы.** Правый клик по ноде — на классическом холсте прямо
по виджету, тогда пункт встанет первым — и там подменю `duration` из двух
пунктов:

| Пункт | Что делает |
| --- | --- |
| **Default value** | Возвращает виджету те десять секунд, с которых он начинает. |
| **Longest offered…** | Спрашивает число и делает его верхним краем виджета. Оно округляется до десятой секунды и хранится в самой ноде, то есть путешествует вместе с воркфлоу. |

![Контекстное меню на классическом холсте над нодой Universal Writer: заголовок MiniMaxH3UniversalWriter, первым пунктом подсвечен `duration`, справа открыто его подменю из двух пунктов — «Default value (10 s)» и «Longest offered (now 30 s)…». Ниже подсвеченного пункта идёт обычное меню ноды: Set, Get, два приглушённых пункта очереди от rgthree, Run, Reload Node и «Favorite Widget: duration». За меню видно саму ноду — чипы соотношения сторон с горящим 16:9 и слайдер duration на значении 10.0](docs/duration_widget.png)

*Правый клик по самому виджету поднимает `duration` в начало меню, как здесь; клик
по любому другому месту ноды ведёт к тому же подменю, только ниже по списку. Оба
числа в подменю принадлежат ноде и сохраняются вместе с воркфлоу.*

Тридцать секунд, пока вы это не поменяли, и до шестисот на стороне сервера. Два
числа, а не одно, потому что диапазон виджета фиксируется при объявлении ноды, и
ни один диапазон не подходит всем графам сразу: руководство MiniMax написано
вокруг клипов в несколько секунд, а растянутые пайплайны, придуманные
сообществом, уходят далеко за это. Поэтому сервер берёт всё разумное, а виджет
охватывает то, с чем работаете *вы*: слайдер, дотягивающийся до десяти минут, —
это слайдер, который не подвинуть на 9.

Это же число — собственное свойство ноды `max_duration`, так что до той же
настройки достаёт и Properties Panel; воркфлоу, сохранённый до появления всего
этого, просто получает тридцатку.

### Повтор последнего ответа

У каждой ноды-райтера, рерайтера и капшенера есть переключатель `repeat_last`.
Включите его — и нода отдаст тот ответ, который выдала в прошлый раз, вместо того
чтобы снова гонять модель. Если ещё ничего не сохранено, она один раз считает
по-настоящему, запоминает написанное и говорит об этом, а дальше возвращает тот
же текст, пока переключатель включён, что бы вы ни меняли. В этом и смысл:
пятидесятисекундный рерайт не стоит оплачивать второй раз, пока вы собираете всё,
что идёт после него.

Собственный кеш ComfyUI так не умеет. Ключ кеша ноды — это её класс, значение
`IS_CHANGED` и все полученные входы, так что правка промта как раз и выбрасывает
запись, а `IS_CHANGED` умеет только добавлять инвалидацию, но не скрывать её.
Поэтому ответы лежат в собственном хранилище пака, по одному на ноду.

Хранилище живёт в памяти на время сессии ComfyUI и больше нигде: оно не пишется
на диск, не путешествует с воркфлоу и очищается перезапуском. Каждый запуск
честно говорит, откуда взялся ответ, в трёх местах — подпись под нодой, консоль
ComfyUI и всплывающая подсказка самого переключателя, где указаны время
сохранения, длина текста и его начало. Если входы разошлись с теми, на которых
ответ был получен, все три об этом скажут, а не сделают вид, что ничего не
изменилось. `bypass` по-прежнему главнее.

**И сохранённое можно править.** `Edit the last prompt` открывает ответ,
который держит нода, в том же редакторе, которым пользуется
[библиотека](#библиотека-промтов): промт в поле, где можно писать, и чтение по
мере набора — по тем правилам, по которым читался прогон. При сохранении поля
заново выделяются из нового текста, так что выходы-секции остаются в согласии с
прозой; всё, что нода сохранила после полей, принадлежит прогону, а не прозе, и
остаётся как было. На диск не попадает ничего — это сессионное хранилище, и
правку, которую стоит сохранить, потом кладут в библиотеку.

Отредактированный текст нода отдаст, как только `repeat_last` в следующий раз её
об этом попросит. Правка не меняет ни одного входа ноды, поэтому каждый писатель
сообщает в `IS_CHANGED` то, что держит, — так же, как сообщает запись
библиотеки, на которую наведён; без этого ComfyUI продолжал бы отдавать ответ,
каким он был до правки.

**Дойдёт ли правка до выхода — вопрос отдельный**, и редактор отвечает на него
ещё до того, как вы начали печатать. Сохранённый промт главнее собственного
ответа ноды, так что, пока выбор стоит, правка ждёт за ним: снимите выбор или
правьте ту запись. При выключенном `repeat_last` следующий прогон просто пишет
новый ответ поверх правки, а на жадном сиде и неизменных входах этот новый
ответ байт в байт совпадёт со старым — что выглядит ровно как игнорирование.
Поэтому сохранение включает `repeat_last`, когда больше ничто не мешает, — так
же, как это делает выбор записи в библиотеке.

На двух нодах-капшенерах хранится именно описание, а не собранный блок:
нумерация принадлежит цепочке, поэтому строка пишется заново вокруг того, что
пришло на `previous`, а ассет, подключённый уже после, описывается по-настоящему.

По умолчанию переключатель отдаёт собственный последний ответ ноды, а
[библиотека промтов](#библиотека-промтов) — это способ указать ему на что-то
другое. Ручка в обоих случаях одна: `repeat_last` решает, отдавать ли готовый
промт вообще, а окно библиотеки — какой именно.

### Выгрузка оставленной модели

`keep_model_loaded` оставляет райтера в памяти после прогона. Это то, что нужно,
пока вы подбираете промт, и то, что мешает, когда текст уже устраивает, а карта
нужна под видео. Выключить переключатель — не способ её вернуть: это вход ноды,
изменённый вход означает новый прогон, и нода сначала напишет тот же текст
второй раз и только потом выгрузит модель. Отпустить модель, не запуская её
заново, можно тремя способами.

**`Unload model`** — кнопка на каждой ноде, у которой есть этот переключатель.
Она сразу выгружает всё, что держит пак. Если нажать её, пока нода ещё пишет,
она дождётся конца этого прогона и выгрузит модель после него — что бы ни
говорил `keep_model_loaded`, потому что нажатие было позже.

**Штатная кнопка ComfyUI `Unload Models`** тоже достаёт модели пака. Это не
модели ComfyUI, и его управление памятью о них не знает, поэтому пак сам
встраивается в выгрузку. То же относится ко всему, что просит ComfyUI выгрузить
все модели: маршрут `/free`, обработчик нехватки памяти и ноды-чистильщики из
других паков — поставьте такую между райтером и сэмплером, и райтер уйдёт
раньше, чем загрузится видеомодель. Работает это и в обратную сторону. Сэмплер,
который сам расчищает карту под себя (так делают несколько обёрток), заберёт
вместе со всем остальным и оставленного райтера; а с `--disable-smart-memory`
ComfyUI выгружает всё после каждого запуска воркфлоу, так что оставленная
модель живёт до конца этого запуска и не дольше.

**Включённый `repeat_last` при выключенном `keep_model_loaded`** тоже выгружает.
Нода, которая отдаёт готовый промт, модель не запускает, но переключатель всё
равно читает: текст возвращается из собственного хранилища ноды, модель
выгружается, и ничего не пишется дважды.

Все три способа отпускают всё, что пак держит внутри процесса ComfyUI, —
чекпоинт Transformers или GGUF на `llama-cpp-python`, — а не только модель той
ноды, на которой нажата кнопка. GGUF, запущенный бинарниками llama.cpp, и не
оставался: модель уходит вместе с процессом.

### Ответ проверяется

Каждый свежий ответ, прежде чем покинуть ноду, перечитывается по правилам
собственных руководств MiniMax — самопроверка, которая молчит, когда всё чисто.
А когда нет — говорит в трёх местах: тостом в окне ComfyUI с первыми
находками, потому что подпись под нодой легко не заметить с другого конца
графа; самой подписью, где над ответом лежит полный список; и консолью — для
прогонов, за которыми никто не следит. На что она смотрит:

- **Структура шотов.** `[Shot 1]` открывает описание и не несёт таймкода;
  последующие шоты пронумерованы по порядку, у каждого таймкод `At MM:SS.mmm`,
  который позже предыдущего и внутри заданной длительности.
- **Разметка диалогов.** `<d>` и `</d>` парные, и каждый блок начинается со
  своего тега `[Language]`.
- **Теги референсов, в обе стороны.** Тег, которого задача не принимает, —
  `<Video 1>` в кадровой задаче, любой тег в T2VA — отмечается, как и обратное:
  референс, дошедший до ноды, но ни разу не процитированный, — генератор всё
  равно его получит, только без единого слова о том, зачем. Ноды, которые видят
  свои референсы, сверяют номера с реально подключённым; текстовые райтеры эту
  половину пропускают, а не гадают.
- **Бухгалтерия полного референса.** Каждый `<Subject N>`, введённый в
  `subject_definitions`, должен строку `retention_analysis`, а про 350–500 слов
  `detailed_description` из руководства напоминается, когда мимо.
- **Строка выравнивания.** I2VA, FL2VA и L2VA открываются фиксированным
  предложением, говорящим H3, куда ложатся референсные кадры; о её отсутствии
  лучше узнать до того, как узнает рендер.
- **Токены эмбеддингов-эффектов.** Единственное правило здесь не от MiniMax:
  заглавное `Embedding:`, токен, приклеенный к предыдущему слову, или точка в
  конце имени — это токен, который ComfyUI молча выбросит. Правило живёт в
  самопроверке, потому что отказ у него той же формы, что и у всех прочих — в
  выводе о нём ни слова, — и потому что именно сюда смотрят, когда
  [эффект ничего не сделал](#minimax-h3-effect-embeddings). Имя не из десяти —
  заметка, а не предупреждение: вполне может быть вашей собственной текстовой
  инверсией.

![Тост-предупреждение с заголовком «Self-check: MiniMax-H3 Universal Writer»: три предупреждения с восклицательным знаком — в ответе не хватает 5 полей, перечислены subject_definitions, summary, retention_analysis, overall_soundscape и non_diegetic_music с советом понизить temperature или взять модель покрупнее; в описании нет Shot 1, а шоты — то, как H3 читает структуру; Picture 1 и Picture 2 подключены, но ни разу не процитированы, и модель всё равно их получит без единого слова о том, зачем — и одна заметка с дефисом: detailed_description на 1286 слов при советуемых руководством 350–500](docs/self_check_alert.png)

*Один нарочно испорченный прогон, прочитанный обратно. Строки с `!` —
предупреждения, то, что H3, скорее всего, поймёт не так; строка с `-` —
заметка, то, что руководство лишь советует. Тост показывает первые находки;
полный список лежит в подписи под нодой и в консоли, где он переживает тост,
сколько бы тому ни было отпущено.*

Отказ — единственное, что ноды действительно навязывают, и говорится он так же:
фраза выходит тостом под меткой **Stopped**, прежде чем исключение уйдёт
наверх, — чтобы остановившийся прогон не выглядел упавшим без причины.

Находки сообщаются, но ничего не блокируют: ответ уходит ровно таким, каким
написан, потому что модель иногда права, нарушая правило, и только вы знаете,
тот ли это случай. Предупреждения (`!`) — то, что H3, скорее всего, прочтёт не
так; заметки (`-`) — то, что руководство лишь советует. Промт, отданный из
библиотеки, не перепроверяется: его проверили, когда он писался, а в подписи
остаётся имя записи.

**Сколько именно произносится** — это `self_check` на ноде Options:
`warnings and notes` — всё; `warnings only` отбрасывает мягкие советы
руководства вроде тех самых 350–500 слов и оставляет то, что H3, скорее всего,
поймёт не так; `off` не говорит ничего. Само чтение происходит всегда — это
регексы по тексту, который уже в памяти, и стоит оно нисколько, — так что
переключатель решает, о чём сообщать, а не на что смотреть. Нечитаемое значение
сообщает всё: устаревший воркфлоу должен оставить проверку громче задуманного,
но не немой.

Тем же тостом приходят и собственные предупреждения нод — под заголовком
**Heads-up** вместо **Self-check**: референс, который выбранная задача не
читает; сохранённый промт, чьи референсы разошлись с тем, что ноде показывают;
капшенер, вернувший пустоту для подключённого ассета. Они — про вашу проводку, а
не про прозу модели, поэтому `self_check` их не покрывает и говорятся они
всегда. И раньше всё это было в подписи и консоли — теперь это ещё и невозможно
пропустить.

**Часть проверок происходит до прогона.** Задача, которой не годится
подключённое, отклонялась и раньше, прежде чем сдвинется хоть один вес: `I2VA` с
двумя картинками, клип, воткнутый в кадровую задачу. Теперь так же отклоняется
`Ref2VA`, если в полосе больше, чем H3 способен принять: девять картинок, три
клипа, три звука. Раньше это сперва описывалось, а помечалось потом — полный
проход каптионера и переписывание ради того, что можно сосчитать заранее.
Субъекты считаются отдельно от картинок, так что бейдж часто и есть всё
лечение.

### Действие по находкам

`fix_once` на ноде Options разрешает нодам что-то с находками сделать, а не
только о них сообщить. По умолчанию **выключен**, и выключенный он — это
обещание: ответ, который вы получите, ровно такой, каким его написала модель,
по дороге в него никто не лазил.

Включённый делает две вещи. **Возвращает на место строку выравнивания**, если
её нет: это фиксированный текст, который нода и так формирует, когда собирает
промт, — модель, его потерявшая, не приняла решения, достойного уважения.
Никакая модель не запускается, регрессировать нечему. А дальше, для
механических находок — таймкод за пределами длительности, непарный `<d>`, тег,
указывающий на референс, которого нет, — **райтера просят написать ещё раз**, с
этими находками, вложенными в промт как правила.

Один раз, не цикл. Модель, дважды проигнорировавшая правило, проигнорирует его
и в третий, а каждая попытка стоит полной генерации. Единственный повтор
удерживают в безопасности три вещи:

- **На безнадёжном ответе он не запускается.** Половина полей отсутствует —
  значит, модель не держит формат вообще, и второй заход этого не изменит;
  нода посоветует взять райтера покрупнее, а не потратит ещё минуту.
- **Правила едут внутри промта**, а не вторым ходом диалога. Так сохраняется
  однооборотная форма, на которой обучались LoRA, и повтор ведёт себя на
  обученном адаптере так же, как на guided-модели.
- **Второй ответ оставляют, только если он лучше** — меньше предупреждений или
  столько же предупреждений при меньшем числе находок. Ничья оставляет первый.
  Худший случай — потраченная минута, но никогда не худший промт.

Что именно произошло, сказано на ноде: строка восстановлена, повтор и сколько
он выиграл, или что первый ответ устоял.

### Библиотека промтов

`repeat_last` держит по одному ответу на ноду и забывает его при перезапуске
ComfyUI. Библиотека — вторая половина: промты, которым вы дали имя и оставили в
JSON-файле в пользовательской папке ComfyUI, доступные любому воркфлоу и любой
ноде.

**Сохранение.** Кнопка **Save the last prompt** на любом райтере или рерайтере
открывает окно с именем, описанием и любым числом групп — группы, уже
встречающиеся в файле, предлагаются чипами, а новая добавляется словом и Enter.
Сохраняется сам прогон: текст, секции после него, задача, модель, пропорции,
длительность, seed и миниатюра 50x50 каждого референса, который видела нода.

Миниатюры снимаются **во время прогона** — единственный момент, когда это
возможно. К открытию окна нода уже вернула строки, а тензоров нет, поэтому
картинка, первый кадр клипа и все измерения снимаются в момент получения ответа и
путешествуют вместе с ним. Записывается то, что действительно существует: у
картинки есть размер; у клипа, пришедшего настоящим `VIDEO`, есть кадры, секунды
и частота кадров, а у пришедшего батчем картинок — только число кадров, потому
что у батча нет контейнера, которого можно об этом спросить; у звука есть
длительность, частота и число каналов. Референсы помечаются по позиции —
`ref1-image`, `ref2-audio`, — а не по имени файла, которого нода всё равно не
видит.

**Использование.** Кнопка **Prompt library** открывает список: сначала
собственный последний ответ этой ноды, потом всё содержимое файла, новые сверху.
Фильтр по группам — чипами, строка поиска оставляет только записи, у которых
введённый текст встречается в имени, описании, промте, группах, референсах или
настройках. **Use** указывает ноде на запись, **Copy** кладёт промт в буфер
обмена, ни на что ноду не наводя, **Edit** открывает её для правки, **Delete**
удаляет запись из файла насовсем, но сначала спрашивает. Со списком можно работать
и с клавиатуры: стрелки ходят по нему, не уводя курсор из строки поиска, Enter
применяет подсвеченную запись, двойной клик по карточке делает то же самое.
Открывается список прокрученным к записи, на которую наведена нода.

![Окно библиотеки промтов: заголовок «Prompt library» со строкой «Hand a saved prompt straight to this node’s output. No model is loaded.», под ним строка поиска «Search names, descriptions, prompts, references» и список файлов рядом с ней на `global`, ещё ниже четыре чипа групп — bakery, dinosaur, fish, joke. Первой идёт строка Last Prompt самой ноды: Use и Copy погашены под фразой «Nothing kept yet. Run this node once and its answer is what gets saved.» Дальше две сохранённые записи: «Дино и пеламида» с тремя миниатюрами слева и строкой Ref2VA · 16:9 · 10s · 3 images + 1 audio · joke, dinosaur, fish над описанием и началом subject_definitions; и «Дино» с одной миниатюрой и строкой I2VA · 16:9 · 15s · 1 image · joke, bakery, dinosaur, её карточка подсвечена синим, потому что именно на неё наведена нода. Справа у каждой записи Use, Copy и Delete, а внизу окна — Write a new one и Close](docs/prompt_library.png)

*Ищется всё, что есть на карточке: имя, описание, сам промт, группы,
настройки и референсы, — на этом и держится файл из сотни записей. Строка под
именем заканчивается тем, для чего запись писалась, — `3 images + 1 audio`, теми
же словами, которыми нода предупреждает, что показывают ей теперь не то; а
миниатюры рядом говорят, какие именно это были картинки. Подсвеченная
карточка — текущий выбор. Строка Last Prompt стоит выше файла, потому что
в файле её нет: это собственный ответ этой сессии, поэтому у неё и нет
**Delete**.*

Библиотека, редактор, **Model list** и встроенные промты меняют размер, как обычные
окна: тяните за край или угол, а кнопка в верхнем углу разворачивает окно на всю
вкладку браузера. Каждое окно запоминает свой размер. Внизу написано, сколько
записей в файле или сколько из них осталось после поиска, и что сделала последняя
правка или удаление.

Выбранный промт попадает в `rewritten_prompt` и в выходы-секции ровно так же, как
свежий: модель не грузится, ничего не генерируется, прогон занимает около
десятой доли секунды. Запись, сделанная нодой другого типа, тоже работает: при
совпадении классов выходы отдаются дословно, иначе текст разбирается на секции
средствами читающей ноды — так же, как она разбирает написанное сама.

**Правка.** **Edit** открывает запись: имя, описание, группы и сам промт в поле,
в котором можно писать. То, чем запись *сделана*, — писатель, настройки,
длительность, миниатюры референсов — показано, но не правится и не меняется. Эта
половина есть отчёт о прогоне, а карточка, совравшая о собственном
происхождении, была бы хуже, чем никакой.

Промт проверяется по мере набора — по тем же правилам, по которым писатели
читают ответ модели, и тем же самым модулем, а задачу, длительность и референсы
для проверки даёт сама запись: время склейки, уехавшее за конец видео, будет
названо, пока вы ещё на него смотрите. Ради этого и стоит править здесь, а не в
JSON-файле — файл примет что угодно.

Escape и **Cancel** спрашивают, прежде чем выбросить набранное, — здесь и в других
окнах, где что-то печатают: в **Save the last prompt** и в форме записи
**Model list**. Если ничего не набрано, окно просто закрывается.

![Окно правки: заголовок «Edit a saved prompt» со строкой «The prompt itself, and what the card says about it. What produced this record -- the writer, the settings, the references -- stays as it was.»; поле Name с «Дино и пеламида (changed)», под ним Description, дальше четыре чипа групп — joke, dinosaur и fish горят, bakery нет, — пустое поле «New group, then Enter» и сам промт моноширинным шрифтом, открытый на subject_definitions и summary. Ниже самопроверка: «Self-check: 2 warning(s), 1 note(s)» над двумя янтарными строками — &lt;Video 1&gt; упомянут, но видео до ноды не дошло, и &lt;Picture 2&gt;, &lt;Picture 3&gt; подключены, но не упомянуты — и серой о detailed_description в 147 слов при советуемых 350-500. Внизу отчёт самой записи: MiniMaxH3UniversalWriter · Ref2VA · 16:9 · 10s · 3 images + 1 audio · saved 29.08.2026, 15:07:10 · edited 01.09.2026, 11:07:04, под ним четыре миниатюры референсов, а в углу — Cancel и Save changes](docs/prompt_library_edit.png)

*Находки читаются из записи, а не угадываются: задача — Ref2VA, длительность — десять секунд, референсы — те четыре, что внизу. Оттуда и известно, что `<Video 1>` назван в промте, куда видео не дошло, а две подключённые картинки ни разу не упомянуты. Янтарный — предупреждение, серый — заметка, которую руководство лишь советует. `self_check` на ноде Options управляет тем, что ноды говорят во время прогона; редактор показывает и то и другое всегда — вы попросили об этом, открыв окно. Всё, что ниже черты, — та половина, которая не правится.*

Изменение текста снимает с записи авторскую разбивку ответа на поля: она была
сделана из текста, каким он был. Без неё читающая нода разбирает текст сама —
ровно то, что она уже делает с промтом, написанным нодой другого типа, — и
выходы-секции остаются в согласии с тем, что вы написали. Всё остальное в записи
переживает правку, а на карточке рядом со временем сохранения появляется время
изменения.

Нода, уже наведённая на эту запись, отдаст новый текст со следующим прогоном.
Запускать ли ноду вообще, ComfyUI решает по её входам, а правка не меняет ни
одного из них — выбор всё тот же id в том же файле, — поэтому каждый писатель
сообщает в `IS_CHANGED` содержимое самой записи, и прогон происходит. Больше в
кэшировании не меняется ничего: когда сохранённый промт не задействован,
значение постоянно, а переименованная, но не переписанная запись не заставляет
перезапускаться все ноды, что её держат.

**Запись с референсами приколота к ним.** Промт T2VA самодостаточен и переезжает
куда угодно. Промт кадровой задачи или Ref2VA — нет: он называет свои референсы и
описывает их — `<Subject 1>` *и есть* синий динозавр с текстурированной чешуёй, —
так что, подсунув такому промту другой набор ассетов, вы отдадите генератору
описание того, чего перед ним нет. Увидеть, для чего запись была написана,
помогают миниатюры на карточке, а нода проверяет и сама: если род и число того,
что ей показывают, разошлись с тем, для чего писался промт, она скажет об этом на
ноде и в консоли, прежде чем его отдать. Именно скажет, а не откажется:
переиспользовать описание как шаблон — законное занятие, и знаете об этом только
вы. Чего не поймает никакая проверка — подмену картинки на другую того же рода;
для этого и нужна миниатюра.

**Решает один переключатель, окно решает что именно.** Выбор записи включает
`repeat_last`, потому что именно он отдаёт готовый промт. Выключение возвращает
ноду модели и оставляет выбор ждать; **Write a new one** делает и то и другое,
заодно забывая выбор. Кнопка сообщает, что происходит: `Library: Storm at sea` —
или то же самое с `(repeat_last is off)` в конце, — а плашка рядом с заголовком
ноды показывает `REPEAT` или `LIBRARY`.

Сам выбор лежит в скрытом виджете, поэтому сохраняется с воркфлоу и доходит до
API-запуска: граф, открытый завтра, вернёт тот же промт. Если запись с тех пор
удалили, прогон останавливается и говорит об этом, а не пишет тихонько новый
промт: граф просил конкретный.

**Наборы.** `prompt_file` на ноде Options называет файл, в который сохраняют и
который листают подключённые к ней ноды, — `global`, пока не сказано иначе, а
кнопка **New prompt file** рядом создаёт другой. Один файл — один рабочий набор,
и это самый дешёвый способ держать промты проекта отдельно от всего остального.
Лежат они в

```text
ComfyUI/user/minimax_h3_rewriter/prompts/
```

обычным JSON, так что их можно править руками, копировать между машинами и
удалять. Запись с одной миниатюрой и промтом в полную длину весит около 11 КБ.

### Руководства скачиваются, а не поставляются в паке

Двух руководств внутри пака нет. Это Documentation компании MiniMax, а MiniMax H3
Community License разрешает распространять Materials только внутри своей
«Applicable Territory» — весь мир *за исключением* Европейского союза,
Великобритании, Республики Корея и США — и только вместе с копией соглашения и
файлом NOTICE. Пак нод в реестре не может соблюдать территориальную границу,
поэтому каждая установка при первом обращении скачивает свою копию напрямую у
MiniMax: 16 КБ и 24 КБ, один раз, под тем же переключателем `auto_download`.

Ложатся они в

```text
ComfyUI/user/minimax_h3_rewriter/guides/
```

и эту папку открывает кнопка **Open guide folder**. Правка копии меняет системный
промт, а скачивание никогда не перезаписывает уже лежащий файл — обрезать
руководство проще всего, когда контекста маленькой модели не хватает.

### Список моделей

На ноде с одним полем модели каталог открывает кнопка **Model list**. Если полей
несколько, нажмите кнопку нужного поля. Записи каталога можно добавлять, править
и удалять кнопками **Add a model**, **Edit** и **Delete**. **Check it** проверяет
запись перед загрузкой модели. Каталог хранится в

```text
ComfyUI/user/minimax_h3_rewriter/models.json
```

Редактор и выпадающие списки нод показывают **On disk**, **Download needed** или
**Files missing**. Для мультимодальной GGUF отметка **On disk** появляется, когда
найдены оба файла, модель и её `mmproj`. В **Model list** можно отфильтровать
модели по имени или оставить только локальные через **On disk only**. Звёздочка
добавляет модель в **Favorites** и поднимает её в начало соответствующих списков нод.

**Use this** выбирает модель в ноде, из которой открыт список. Если полей
несколько, кнопки **Captioner models** и **Writer models** открывают
соответствующие вкладки. **Use in writer_model** применяет модель к указанному
полю. Бейдж показывает папку модели или место будущей загрузки, например `LLM/quant`.

Включите **Search nested folders** для поиска во всех подпапках подключённых
папок моделей. Переключатель и избранное сохраняются на сервере ComfyUI.
После копирования моделей на сервер нажмите **Rescan folders**.

Файл засевается из поставки при первом обращении, поэтому обновление пака не затирает
ваши правки. **Open models.json** в подвале окна по-прежнему открывает этот файл в
системном редакторе JSON — **на той машине, где работает ComfyUI**, а не обязательно на
той, где открыт браузер, — для двух вещей, которые окно намеренно не трогает: секций
`adapters` и пути к сетевой шаре.

Окно показывает только те списки, которые действительно читает нода, из которой его
открыли: адаптеры рассчитаны на разные архитектуры, и запись из одного списка не
загрузится в ноде другого. Поэтому Universal Rewriter открывается на трёх вкладках, а
капшенер — на одной. Разверните **Model requirements**, чтобы посмотреть требования
к архитектуре, размерам модели и проектору. В **Found in model folders** показаны
модели из папок ComfyUI и хранилищ Ollama. Их тоже можно выбрать в выпадающем списке
ноды. Если найденный файл указан в каталоге, в обоих списках он представлен именем
записи с отметкой **On disk**.

![Вкладка Guided writers в окне Model list с включённым поиском во вложенных папках. У локальных моделей видны отметки On disk, относительные пути, кнопки Use in writer_model и звёздочки избранного.](docs/model_list_dialog.png)

Внизу окна написано, сколько записей в открытом списке и что сделали последнее
сохранение, удаление или восстановление, — в том числе сколько нод этого графа
перешли на новую подпись записи или всё ещё держат имя удалённой. Такие ноды имя
сохраняют, чтобы запуск сказал, что записи нет, а не выбрал за них другую модель.

**Check it** отвечает настолько, насколько на вопрос можно ответить, не двигая веса.
Файл, уже лежащий на этой машине, читается напрямую, поэтому сообщается архитектура,
форма, есть ли у модели шаблон чата и несёт ли проектор зрение и звук:

```text
- 'Qwen2.5-Omni-7B-Q4_K_M.gguf' is a 'qwen2vl' model, 28 blocks of width 3584. That fits.
- It carries its own chat template.
- 'mmproj-Qwen2.5-Omni-7B-Q8_0.gguf' carries the vision and audio encoder.
```

Шаблон чата важнее, чем кажется. Ререйтер 27B и писатели собирают промт по
собственному шаблону модели, поэтому GGUF без него — обычно базовая конвертация
модели, а не instruct — выглядит совершенно исправной моделью и останавливается на
первом же запуске. Там Check it такую модель отклоняет, а шаблон один раз прогоняет
на системной и пользовательской репликах, так что отклоняется и шаблон, который этот
пак отрендерить не может. Капшенеры и ререйтеры 8B и Omni оставляют шаблон
`llama-mtmd-cli`, а тот без шаблона берёт ChatML, так что там его отсутствие — только
предупреждение.

То, что есть только на Hugging Face, спрашивается о том, что может сказать метаданными:
репозиторий transformers оценивается по `config.json` в 4 КБ, а репозиторий GGUF —
вопросом, есть ли в нём названные вами файлы. Именно это ловит опечатку, которая иначе
всплыла бы отказом загрузки через несколько минут, и заодно заполняет `download_gb`.
Форма GGUF и его шаблон чата лежат в самом файле, поэтому для файла, который ещё на
Hub, они ждут первого запуска — или повторного Check it, когда файл уже скачан.

Если оставить `file` пустым, **Check it** отвечает на другой вопрос: что лежит в этом
репозитории? Он показывает GGUF-файлы кнопками, от меньшего к большему. Клик ставит
файл в форму, а пока своего имени вы не набрали — ещё и имя в стиле записей из
пакета, `Qwen2.5-Omni-7B Q4_K_M`. Списку, которому нужна пара, проекторы показываются
отдельной строкой для `mmproj`; если проектор в репозитории один, он подставляется
сам, а когда на месте обе половины, проверка запускается и складывает размер
загрузки. Модель, разбитая на части (`-00001-of-00003.gguf`), подсчитывается и
пропускается: запись называет один файл, и скачивается только он. Папка на этом
компьютере в `repo` перечисляется так же, а имя файла, которого в репозитории нет,
выводит те же кнопки.

Двух вещей окно не делает. Оно отказывает **сетевому пути** в `repo`, `file` и `mmproj`:
окно доступно через API ComfyUI, у которого нет CSRF-токена и который постоянно поднимают
с `--listen`, а один только взгляд на UNC-путь — уже попытка аутентификации на названном
хосте. Путь, вписанный в `models.json` руками, по-прежнему не ограничен: этот файл никуда
не уезжает. И оно отказывается писать, пока файл не разбирается, потому что запись поверх
заменила бы ваши записи поставляемым списком: показывается ошибка разбора, а всякая
кнопка, которая пишет, мертва — остаётся **Open models.json**. Показывает оно в этом
состоянии поставляемую копию, потому что именно её до починки файла предлагают и
выпадающие списки.

Правка размера загрузки, заметки о VRAM или примечания меняет то, что читает
выпадающий список, а сохранённые воркфлоу помнят именно эту строку — но запись
находится по имени, поэтому воркфлоу со старой подписью по-прежнему её запускает и
переходит на новую подпись, когда его открывают. Переименование записи — другое дело,
ведь по имени её и ищут: форма предупреждает об этом до сохранения, открытый у вас граф
переводится на новое имя сам, а другие воркфлоу выбор теряют.

Удаление записи оставляет использовавшие её ноды как есть: запуск остановится и скажет,
что записи больше нет, а не подставит другую модель и не начнёт загрузку, о которой
никто не просил. Исключение — запись, чей файл уже здесь: его строка `on disk:`
возвращается в выпадающий список, и ноды переходят на неё.

В файле пять списков с одинаковыми полями. **`models`** питает ноду ререйтера, и там
обязательно должна быть Qwen3.6-27B:

```json
{
  "name": "Qwen3.6-27B FP8",
  "repo": "Qwen/Qwen3.6-27B-FP8",
  "download_gb": 28.8,
  "vram": "~29 GB, no extra package needed"
}
```

**`models_8b`** питает ререйтер 8B, там обязательно должна быть
Qwen3-VL-8B-Instruct, и рядом с моделью нужен `mmproj`: мультимодальная модель —
это два файла:

```json
{
  "name": "Qwen3-VL-8B-Instruct GGUF Q4_K_M",
  "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
  "file": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
  "mmproj": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
  "format": "gguf",
  "download_gb": 5.4,
  "vram": "~9 GB with the adapter"
}
```

**`models_omni`** питает ререйтер Omni, там обязательно должна быть
Qwen2.5-Omni-7B, и файла тоже два — с той разницей, что её проектор обязан нести
аудио-энкодер, иначе лора прикрепится к модели, которая не слышит:

```json
{
  "name": "Qwen2.5-Omni-7B GGUF Q4_K_M",
  "repo": "ggml-org/Qwen2.5-Omni-7B-GGUF",
  "file": "Qwen2.5-Omni-7B-Q4_K_M.gguf",
  "mmproj": "mmproj-Qwen2.5-Omni-7B-Q8_0.gguf",
  "format": "gguf",
  "download_gb": 6.2,
  "vram": "~9 GB with the adapter and a 12k context"
}
```

Три отдельных списка, а не дополнительные записи в `models`, потому что запись из
любого из них не загрузилась бы в чужой ноде: другая архитектура, другой адаптер.
Адаптеры разведены по той же причине — `adapters`, `adapters_8b` и
`adapters_omni`.

**`writers`** питает ноды писателя, и там может быть что угодно — лишь бы это была
GGUF-модель языка с чат-шаблоном:

```json
{
  "name": "Qwen3.5-4B",
  "repo": "unsloth/Qwen3.5-4B-GGUF",
  "file": "Qwen3.5-4B-Q4_K_M.gguf",
  "format": "gguf",
  "download_gb": 2.6,
  "vram": "~5 GB with the guide in context"
}
```

**`captioners`** питает ноду Reference Caption, и там нужно одно дополнительное
поле — `mmproj`: мультимодальная модель это два файла, и оба должны быть из одной
конвертации:

```json
{
  "name": "Qwen2.5-Omni-3B",
  "repo": "ggml-org/Qwen2.5-Omni-3B-GGUF",
  "file": "Qwen2.5-Omni-3B-Q4_K_M.gguf",
  "mmproj": "mmproj-Qwen2.5-Omni-3B-Q8_0.gguf",
  "format": "gguf",
  "download_gb": 3.4
}
```

#### Как указать файл, который уже лежит на диске

`repo` не обязан быть идентификатором с Hugging Face. Укажите там **папку на этой
машине** — и файл будет прочитан прямо из неё, без скачивания и без копирования:

```json
{
  "name": "Gemma 4 26B",
  "repo": "X:/models/gemma-4-26B-A4B-it-UD-Q8_K_XL",
  "file": "gemma-4-26B.gguf",
  "format": "gguf"
}
```

Либо положите весь путь в `file`, а `repo` не указывайте вовсе — работают обе
формы и во всех трёх секциях, потому что обе люди и пишут. Два подводных камня:

- **Обратные слэши в JSON нужно удваивать** — или писать прямые. `"X:\Programs\..."`
  вообще не валидный JSON: `\P` не является escape-последовательностью, и один
  такой символ делает нечитаемым **весь файл**, а не одну запись. Windows
  прекрасно понимает `X:/Programs/...`, так что это привычка попроще.
- Несуществующий путь **называется вслух, а не обходится скачиванием**. Нода
  скажет, какой файл не нашла, и остановится.

Если у вас целая папка с GGUF где-то в стороне, заводить запись на каждый файл —
долгий путь: пропишите её в `extra_model_paths.yaml` под ключом `LLM`, и все
файлы оттуда появятся в списке сами, с префиксом `on disk:`. Если эта папка идёт
первой, новые загрузки пойдут в неё; модель, уже лежащая в любой из папок `LLM`,
находится там, где лежит, и заново не скачивается.

Добавили запись, **обновили вкладку браузера** — перезапускать ComfyUI не нужно —
и она в списке. `name` лучше не менять: по нему сохранённый воркфлоу находит
запись. Нода с пропавшим выбором остановится до запуска и честно скажет об этом
по имени, а не подставит молча что-то другое.

#### Когда сломан сам список

Синтаксическая ошибка в `models.json` была самым тихим отказом во всём паке:
разбор падал, вместо вашего списка подставлялся упакованный по умолчанию, и
выпадашка выглядела совершенно обычно. Правка, которая не применилась, была
неотличима от правки, которая ничего не изменила.

Теперь об этом говорит первая же строка в каждом списке моделей — с номером
строки и позиции:

```text
!! models.json is not valid JSON — Invalid \escape: line 3 column 30 (char 46) — showing the packaged list instead
```

Остальной список при этом на месте и ComfyUI работает; если выбрать эту строку и
нажать Run, сообщение повторится и назовёт файл, который надо починить. Почините,
обновите вкладку — и оно исчезнет.

#### Обновления добавляют новые записи, не трогая ваши

Ваш файл не перезаписывается, но модели, добавленные в пак после вашей установки,
**вливаются в него** — иначе «мы не тронем ваш список» тихо превращается в «вы
никогда не увидите ничего нового», причём без единого сигнала о том, что нода
знает о большем.

Правило — теория множеств, а не сравнение версий. Рядом со списками ваш файл
хранит `seed_offered`: все имена, которые поставляемый список когда-либо
предлагал именно этой установке. При обновлении добавляется

```text
имена в паке  −  имена в вашем файле  −  имена, которые вам уже предлагали
```

Поэтому удалённая вами запись остаётся удалённой, переименованная не двоится, а
действительно новая приезжает. Удалите имя из `seed_offered` — и эта запись будет
предложена снова при следующем запуске; **Restore the packaged entries** в окне списка
моделей — это та же правка кнопкой: она убирает запись для одного списка, и следующее
чтение возвращает все недостающие записи из поставки. Ваших собственных записей это
не касается.

Одно исключение, и оно однократное: у файла, написанного до появления этого
механизма, нет записи о том, что ему предлагали, — поэтому при первом обновлении
вернётся всё недостающее, включая то, что вы удаляли руками. Предыдущая копия
сохраняется рядом как `models.json.bak`, и дальше ваши удаления держатся. Эта копия
фиксирует именно слияние: правки, сделанные в окне списка моделей, её не расходуют —
иначе двух кликов хватило бы, чтобы потерять снимок, ради которого она и заводится.

Слияния пишутся в консоль ComfyUI поимённо. Файл, который нода не смогла
разобрать, остаётся нетронутым, а на эту сессию берётся поставляемый список — так
что лишняя запятая стоит вам перезапуска, а не правок.

### Модели, уже скачанные вами для Ollama

Если у вас стоит Ollama, писатели и капшенеры, которых хочет этот пак, уже лежат
у вас на диске, и качать тот же квант второй раз незачем. Они предлагаются в
выпадающих списках `writer_model` и `caption_model` с префиксом `ollama:` и под
теми же именами, что показывает `ollama list`:

```text
ollama: qwen3:8b [qwen3, 4.7 GB]
ollama: moondream:latest [+mmproj, vision, 1.7 GB]
```

Ничего не копируется, не конвертируется и не скачивается, и сама Ollama при этом
может быть выключена: то, что она скачивает, — обычный GGUF, и llama.cpp читает
его там, где он лежит. Читается файл на диске, а не API Ollama, — сервис можно
остановить, убрать из автозапуска или снести вовсе, модели останутся рабочими.

Мультимодальная модель попадает в оба списка: писателем — без проектора,
капшенером — вместе с ним. Пару даёт манифест самой модели, где оба файла названы
рядом, так что она достоверна — в отличие от сравнения имён файлов в папке.

Смотрим в три места, и раскладка Ollama одинакова на всех платформах:

```text
$OLLAMA_MODELS                     # если вы его задали
~/.ollama/models                   # обычная установка, в том числе Windows
/usr/share/ollama/.ollama/models   # Linux, установка сервисом
```

Хранилище, лежащее где-то ещё — внутри WSL, в контейнере, на диске, о котором
сервер не знает, — указывается руками: либо в `models.json`

```json
"ollama_stores": ["\\\\wsl$\\Ubuntu\\usr\\share\\ollama\\.ollama\\models"]
```

либо в `MINIMAX_H3_OLLAMA_MODELS`, который принимает несколько путей через тот же
разделитель, что и `PATH`. Автоматически они не ищутся намеренно: обращение к
`\\wsl$\...` **поднимает остановленный дистрибутив WSL**, а эти списки строятся
заново при каждом заполнении выпадающего списка. Открытие вкладки ComfyUI не
должно запускать виртуальную машину.

`ollama rm` убирает модель из списков так же, как удаление любого другого файла.
Воркфлоу сохраняет метку, а не путь, поэтому повторная закачка того же тега
оставляет его рабочим — хранилище адресуется по содержимому, и те же файлы
ложатся туда же. Но в метке стоит размер, а другой квант — это другой размер, так
что `qwen3:8b`, перекачанный в другом кванте, вернётся выбором, который придётся
сделать заново.

## Куда кладутся веса

```text
ComfyUI/models/LLM/
├── Qwen3.6-27B/                          # базовая модель, ~52 ГБ
├── MiniMax-H3-Prompt-Rewriter-LoRA/      # адаптер, ~3.5 ГБ
├── Qwen3.5-9B-Q4_K_M.gguf                # модель-писатель, один файл
└── Qwen2.5-Omni-3B/                      # капшенер: модель и mmproj вместе

ComfyUI/user/minimax_h3_rewriter/
├── models.json                           # ваш список моделей
├── guides/                               # два руководства, 40 КБ
└── runtime/                              # бинарники llama.cpp, если скачаны
```

**Уже скачали LoRA?** Укажите путь к этой папке прямо в виджете `adapter`
(абсолютный путь принимается) — повторно ничего не скачается.

## Использовать уже имеющуюся модель

52 ГБ — это много, поэтому нода сначала ищет базовую модель и только потом
предлагает её скачать. В списке `model` окажутся все найденные Qwen3.6-27B из:

- всех каталогов, которые ComfyUI регистрирует под именами `LLM`, `transformers`,
  `diffusers`, `text_encoders` и `clip` — включая подключённые через
  `extra_model_paths.yaml`, и
- кэша Hugging Face (`HF_HOME` / `HF_HUB_CACHE` / `~/.cache/huggingface/hub`),

так что копия, скачанная раньше любым другим инструментом, будет переиспользована,
а не скачана повторно. В список попадают только каталоги, где веса реально лежат:
запись в кэше, содержащая один лишь `config.json`, не предлагается.

### Сборки поменьше

Любой репозиторий, добавленный в список моделей, проверяется по `config.json`
весом 4 КБ **до того**, как сдвинется хоть один вес. Все перепаковки сохраняют
одинаковый отпечаток (`model_type qwen3_5`, hidden 5120, 64 слоя, словарь 248320),
поэтому исход решает рантайм квантизации:

| Сборка | Скачать | Нужный пакет | LoRA цепляется? | Итог |
|---|---|---|---|---|
| `quant_method: bitsandbytes` (nf4-перепаковка) | **~17 ГБ** | `bitsandbytes`, уже в зависимостях | да | **работает как есть — путь для слабых машин** |
| `Qwen/Qwen3.6-27B` (bf16) | 52 ГБ | ничего; `bitsandbytes` для `nf4`/`int8` | да | **работает как есть** |
| `Qwen/Qwen3.6-27B-FP8` | 29 ГБ | `kernels` плюс ядро, качаемое с Hub | да | только для продвинутых; нужно больше 29 ГБ VRAM |
| `quant_method: awq` | ~19 ГБ | `autoawq` | да | заработает после установки |
| `quant_method: gptq` | ~19 ГБ | `gptqmodel` | да | заработает после установки |
| `quant_method: compressed-tensors` | ~19 ГБ | `compressed-tensors` | **нет** | не поддерживается |
| `quant_method: modelopt` (NVFP4) | ~20 ГБ | `nvidia-modelopt` | **нет** | не поддерживается |

В PEFT есть диспетчеры LoRA для слоёв bitsandbytes, AWQ, GPTQ, HQQ, EETQ, AQLM и
torchao, а обычный `nn.Linear` покрывает bf16/fp16 и FP8. Для compressed-tensors
и modelopt диспетчера нет вовсе, то есть адаптер физически не на что вешать —
установкой пакетов это не лечится.

> **Название репозитория не равно формату:** `cyankiwi/Qwen3.6-27B-AWQ-INT4` и
> `unsloth/Qwen3.6-27B-NVFP4` — оба на деле `compressed-tensors`, и LoRA не
> цепляется ни к одному. Нода показывает настоящий `quant_method` и точную
> причину ещё до того, как что-либо скачает.

### Если нода сообщает, что пакета не хватает

Ни одному из рекомендованных путей это не грозит. bf16, nf4-перепаковки и опции
`nf4`/`int8` работают на пакетах, которые нода и так объявляет в зависимостях,
так что сообщение о недостающем пакете появится только у сборки, которую вы
выбрали сами.

А если появится — в нём будет и название пакета, и **команда для того самого
интерпретатора, на котором запущен ComfyUI**. Набранный в обычном терминале
`pip install kernels` ставит пакет в тот Python, который нашёлся в PATH, а в
портативной сборке это никогда не тот, и нода продолжает отказывать:

```
This base model cannot run the prompt-rewriter LoRA.
  - the 'fp8' checkpoint needs the 'kernels' package, which is not installed in
    this Python environment. Install it with:
      "…\python_embeded\python.exe" -m pip install kernels
    Note: installing it is only half of it: the FP8 matmul is a Triton kernel
    that transformers then downloads from 'kernels-community/finegrained-fp8'
    on the first generation, and that needs a build matching this torch and
    CUDA version.
```

Выполните строку как напечатано и перезапустите ComfyUI. Сама нода в ваше
окружение ничего не ставит и ставить не будет: нода, которая молча делает
pip install, — это нода, которая способна сломать посторонний кусок ComfyUI,
пока вы смотрите на полосу загрузки.

## Самая лёгкая загрузка вообще без установки

nf4-перепаковка Qwen3.6-27B через `bitsandbytes` — это **~17 ГБ загрузки и ~16 ГБ
VRAM**, и она не требует ничего сверх того, что нода и так требует. Именно этот
путь стоит советовать: тот же код, что и для официального чекпоинта, ноль новых
зависимостей, треть объёма.

Перепаковки эти сторонние. Нода проверяет архитектуру по `config.json` до начала
скачивания — это доказывает, что форма модели верная, но не что автор
добросовестен. Оценивайте сами или сделайте перепаковку один раз из официальных весов.

## GGUF — ещё легче, и ставить ничего не надо

Выберите в списке моделей запись `[gguf]`, и ререйтер пойдёт через llama.cpp
вместо Transformers. **Никакого `pip install` при этом нет.** Если
`llama-cpp-python` в окружении ComfyUI уже стоит — нода возьмёт его; если нет —
запустит llama.cpp, который на машине уже есть, а если нет и его — скачает
официальные бинарники (~34 МБ, а если `llama_backend` разрешится в CUDA, то
~511 МБ, — см. таблицу ниже) в `ComfyUI/user/minimax_h3_rewriter/runtime/` и
запустит `llama-completion` отдельным процессом. Под тем же флагом
`auto_download`, что и веса.

Модель при этом перезагружается на каждый запуск, и на практике это ничего не
стоит: по умолчанию у ноды `keep_model_loaded = False`, потому что сразу после
ререйта карта нужна под генерацию видео, — а в этом режиме встроенный движок и
так выгружает модель после каждого запуска. Чего бинарный бэкенд действительно
не умеет — так это `keep_model_loaded = True`. Зато бесплатно достаются две
вещи: VRAM возвращает операционная система, а не деструктор, и падение
llama.cpp уносит дочерний процесс, а не ComfyUI вместе с очередью.

За это отвечают две опции ноды настроек, и они про разное. **`gguf_runtime`**
выбирает, *чем считать*:

| `gguf_runtime` | Что делает |
|---|---|
| `auto` | llama-cpp-python, если он импортируется, иначе бинарники |
| `llama-cpp-python` | принудительно колесо; если его нет — внятная ошибка, а не тихая подмена |
| `llama.cpp` | принудительно бинарники, даже когда колесо установлено, — выход, если установленное колесо сломано |

`keep_model_loaded` умеет только `llama-cpp-python`: бинарники отдают модель
операционной системе вместе с завершением процесса.

**`llama_backend`** выбирает, *какую официальную сборку тянуть*, и действует
только когда работают бинарники:

| `llama_backend` | Загрузка | Замечания |
|---|---|---|
| `auto` | 34 или 511 МБ | **самая быстрая сборка, которую машина потянет, а не самая лёгкая.** CUDA — на Windows с картой NVIDIA compute capability 8.6, 8.9, 12.0 или 12.1 (архив несёт нативный SASS ровно под них и не несёт PTX, из которого можно было бы собрать остальное); во всех прочих случаях — Vulkan |
| `vulkan` | 34 МБ | NVIDIA, AMD и Intel одинаково; примерно вдвое медленнее CUDA. Выбирайте явно, чтобы удержать загрузку на карте, которую `auto` отправил бы в CUDA |
| `cuda` | 511 МБ | вдвое быстрее на NVIDIA; **только Windows** — под Linux upstream CUDA-сборок не публикует, так что под Linux вы собираете её сами, а нода её запускает |
| `cpu` | 17 МБ | без GPU вовсе |

**Уже собранный llama.cpp запускается как есть.** Прежде чем что-то качать,
нода смотрит в четыре места по порядку: путь из `MINIMAX_H3_LLAMA_BIN`, путь,
записанный в `ComfyUI/user/minimax_h3_rewriter/llama_bin.txt`, уже распакованную
копию **того бэкенда, который сейчас в работе**, потом `PATH`. Собственной сборке
поэтому не нужно никаких настроек — положите её `build/bin` в `PATH` или
назовите прямо:

```sh
export MINIMAX_H3_LLAMA_BIN=/opt/llama.cpp/build/bin   # папка или сам бинарник
```

**На сервере надёжнее файл.** `export` доходит до сервера, запущенного из этой же
оболочки, и больше никуда: юнит systemd, entrypoint контейнера или скрипт-лаунчер
выдают процессу собственное окружение и ваш `~/.bashrc` не читают вовсе. Строчку в
`llama_bin.txt` читает сама нода, и ей всё равно, кто запустил ComfyUI:

```sh
echo /opt/llama.cpp/build/bin > ~/comfy/ComfyUI/user/minimax_h3_rewriter/llama_bin.txt
```

После этого `llama_backend` перестаёт что-либо значить: он выбирает, какой архив
скачать, а не то, с чем скомпилирован уже имеющийся бинарник. Это и есть дорога
к CUDA-сборке llama.cpp под Linux — собрать один раз с `-DGGML_CUDA=ON`,
назвать здесь, и `device = cuda:0` начнёт делать то, что написано. Ноды-описатели
ищут рядом `llama-mtmd-cli`; сборка без этой цели отправит ноду за архивом ради
одной этой задачи.

**Распакованная копия своя у каждого бэкенда** — в `runtime/<релиз>-<бэкенд>`, —
так что архив, уже лежащий в `runtime/b10310-vulkan`, не годится для запуска,
разрешившегося в CUDA: его нода скачает тоже. Смена `llama_backend` поэтому
означает вторую загрузку, а сборку, распакованную руками, надо класть в папку с
именем того бэкенда, который вы действительно запустите.

Путь, указывающий в пустоту, — ошибка, а не тихая загрузка: опечатка скажет о
себе, а не обойдётся в полгигабайта. А если не нашлось нигде, отказ печатает, что
было у **процесса** ComfyUI на самом деле — переменная, файл, папка и его `PATH`, —
вместо того чтобы повторять совет, которому вы могли уже последовать в оболочке,
которой сервер не видел.

> **Почему не CUDA-колёса `llama-cpp-python`.** Оба нынешних не работают на
> обычном потребительском железе, причём по двум независимым причинам:
>
> | Колесо | Флаги сборки | Что происходит |
> |---|---|---|
> | `v0.3.34-cu130` | `AVX512 = 1`, `ARCHS = 750..900` | веса грузятся, затем `llama_init_from_model` падает с `0xC000001D` — AVX-512 нет ни у одного потребительского Intel 12–14 поколений |
> | `v0.3.34-cu132` | `AVX512 = 0`, `ARCHS = 750..900` | доходит до первого ядра и падает: `the provided PTX was compiled with an unsupported toolchain` — `sm_120` в списке нет, карта RTX 50-й серии уходит на JIT, а драйвер старше тулкита сборки его не принимает |
> | `v0.3.34-vulkan` | `AVX512 = 0`, списка архитектур нет | работает и подхватывает `NV_coopmat2`, если драйвер его предлагает |
>
> У официальных архивов нет ни той, ни другой болезни. В них **14** вариантов
> CPU-бэкенда с выбором в рантайме — поэтому та же модель прекрасно работает под
> `llama-cli` на машине, где колесо cu130 падает. А в их CUDA-архиве лежит
> нативная SASS вообще без PTX: `cuobjdump --list-elf` показывает
> `sm_86 sm_89 sm_120a sm_121a`, так что драйвера никогда ни о чём не просят.
>
> Если встроенный движок всё же нужен, рабочее колесо — Vulkan:
> `pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-vulkan/llama_cpp_python-0.3.34-py3-none-win_amd64.whl`

| Квант базы | Загрузка | VRAM вместе с адаптером |
|---|---|---|
| `Q4_K_M` | 15.7 ГБ | ~19 ГБ |
| `IQ4_XS` | 14.4 ГБ | ~18 ГБ |
| `UD-Q3_K_XL` | 13.5 ГБ | ~17 ГБ |
| `UD-IQ2_M` | 10.1 ГБ | ~13 ГБ, заметно ниже качество |

Уменьшайте `gpu_layers` в ноде настроек, чтобы влезть в меньшую карту ценой
скорости. При `Q4_K_M`, целиком выгруженной на топовую потребительскую карту
NVIDIA, CUDA даёт около **50 ток/с** с адаптером против 78 ток/с без него (эти
~35% и есть матричные умножения адаптера), Vulkan — примерно вдвое меньше.

> **Qwen3.5 поменьше — не замена.** У Qwen3.5-9B в заголовке та же строка
> `general.architecture = qwen35`, поэтому она выглядит подходящей и попадает в
> список — но у неё 32 блока шириной 4096, а адаптеру нужны 64 шириной 5120.
> llama.cpp отказывается цеплять LoRA
> (`tensor 'blk.0.attn_gate.weight' has incorrect shape`), и запуск падает.
> Нода сверяет эти два числа заранее и говорит об этом до всякой загрузки, а в
> списке помечает такие файлы как `(wrong size for the adapter)`.
> Если вы видите, как 9B выдаёт правдоподобный ререйт, — она работает **без**
> адаптера: формат задаёт системный промпт, а не LoRA. И это не тупик — именно
> это [ноды писателя](#minimax-h3-prompt-writer-t2vai2vafl2val2va) делают
> намеренно, только с полным руководством в промте вместо семи строк.

> **Более новая Qwen — тоже не замена.** Qwen3.8-27B — это `qwen35` с теми же 64
> блоками шириной 5120, поэтому проверка формы её пропускает, и llama.cpp молча
> цепляет LoRA — к весам, на которых та не обучалась. Адаптер обучен на
> [Qwen3.6-27B](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA)
> и больше ни на чём. Нода читает имена, которые GGUF даёт сам себе, и если ни
> одно из них не Qwen3.6-27B, предупреждает об этом на ноде и в Check it. Отказа
> нет: эти имена необязательны, и пишет их тот, кто квантовал файл.
> **MTP-сборка** Qwen3.6-27B подходит: она сообщает о 65 блоках, потому что несёт
> драфт-голову для спекулятивного декодирования, а нода эту голову не считает —
> здесь llama.cpp её не грузит, и у адаптера для неё ничего нет.

На пути GGUF используется **конвертированный** адаптер, а не PEFT-версия: F16 и
Q8_0 [для 27B](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF) и
[для 8B](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-8B-GGUF). Он
скачивается сам; чтобы взять свой, положите `.gguf` в `models/LLM` и выберите его
в списке `adapter` ноды настроек либо пропишите `adapters.gguf.repo` в файле
списка моделей. Промпт собирается по чат-шаблону из
самого GGUF с `enable_thinking=False`, и результат **побитово совпадает** с тем,
что выдаёт `transformers.apply_chat_template` — модель видит ровно тот текст, на
котором обучалась LoRA.

Если у чекпоинта есть собственная квантизация, виджет `quantization` игнорируется:
bitsandbytes поверх AWQ или FP8 не накладывается.

### Прогресс на ноде

Скачивание, загрузка весов и генерация токенов отображаются прямо на ноде через
штатные каналы прогресса ComfyUI: полоса плюс подпись с текущим файлом, объёмом,
скоростью и оценкой времени. Собственного фронтенд-расширения здесь нет, поэтому
обновление фронтенда ComfyUI ничего не сломает.

### Переменные окружения

| Переменная | Действие |
|---|---|
| `HF_TOKEN` | Токен доступа к закрытым или gated-репозиториям |
| `HF_ENDPOINT` | Зеркало, с которого качать вместо `huggingface.co` |
| `MINIMAX_H3_LLAMA_BIN` | llama.cpp, уже стоящая на машине: сам бинарник или папка с ним |
| `MINIMAX_H3_MTMD_SERVER` | `auto`, `never` или `always`: держать ли одну модель открытой на весь прогон вместо загрузки под каждый референс |

### Языки

Переключите ComfyUI на русский (Settings → Comfy → Locale) — и ноды поднимутся
по-русски: что такое каждая нода и подсказка к каждому виджету. Длинные
объяснения — это и есть смысл перевода: именно их вы читаете.

**Имена виджетов намеренно оставлены английскими.** `repeat_last`, `fix_once`,
`prompt_file` — идентификаторы, а не проза: это ключи в JSON воркфлоу и в
вызове API, так они называются в этом README и в issue на гитхабе, и подсказки
называют виджеты именно ими. Перевод подписи оставил бы все эти ссылки
указывающими в никуда — а так русская подсказка говорит `repeat_last`, и это
ровно то, что написано на ноде.

Одна подсказка оставлена по-английски осознанно — `device` на ноде Options: она
собирается на лету и заканчивается списком видеокарт этой машины. Статический
файл такого нести не может, не разослав всем чужое железо.

Переводы лежат в `locales/<язык>/nodeDefs.json` в корне пака — там, где ComfyUI
их и ищет. Добавить язык — значит добавить папку; питону об этом знать не надо.
Не сбиться помогают две команды:

```text
python tools/locales.py report ru     чего не хватает и что устарело
python tools/locales.py fill ru       дописать недостающие ключи по-английски, поверх переводить
```

`report` спрашивает у запущенного ComfyUI, какие ноды есть на самом деле, — так
он не разойдётся с кодом, как разошлась бы вторая копия определений; `fill`
никогда не затирает уже переведённый ключ. Остальное закрывают тесты: что файл
— валидный JSON, что в нём только ноды этого пака и что каждый ключ из тех,
которые ComfyUI действительно читает, — опечатка в ключе не ломает ничего, она
просто молча не показывается.

**Нода, уже стоящая на холсте, сохраняет заголовок, с которым её создали.**
Смена языка её не переименует: этот заголовок записан в воркфлоу. Чтобы увидеть
перевод, добавьте ноду заново.

## Замечания

- **Скорость.** Qwen3.6-27B — гибридная модель: 48 из 64 слоёв используют linear
  attention. Без установленных `flash-linear-attention` и `causal-conv1d`
  Transformers переключается на более медленный чистый PyTorch и пишет об этом в
  консоль. Результат корректный, просто медленнее; оба пакета опциональны и
  тяжело собираются под Windows.
- **Детерминированность.** При включённом `greedy` одинаковые промпт, разрешение,
  длительность и seed дают одинаковый результат, и ComfyUI закеширует ноду.
- **Прерывание.** Отмена задания останавливает и скачивание, и генерацию;
  недокачанный файл продолжится при следующем запуске.
- **На нодах писателя формат соблюдается, но не гарантируется.** Обычная модель
  следует инструкциям, а не воспроизводит распределение, на котором её обучали.
  Держите `greedy` включённым — маленькие модели уходят из формата, едва начав
  сэмплировать, — а если поля не хватает, нода вернёт всё, что получила, и
  назовёт пропуск на самой ноде, вместо того чтобы упасть.
- **Чат-шаблон, не принимающий системную роль, тоже работает.** У Gemma он
  вызывает `raise_exception`; руководство тогда вкладывается в первую
  пользовательскую реплику — как и предписано карточками таких моделей.
- **Перечисление ваших GGUF почти ничего не стоит.** Для выпадающего списка нужно
  шесть значений из первых килобайт каждого файла, но `gguf.GGUFReader` при
  открытии материализует весь заголовок целиком — включая `tokenizer.ggml.tokens`,
  четверть миллиона строк. Десять файлов в модельной папке стоили 31 секунду,
  и платились они при первой отдаче `/object_info`. Теперь заголовок читается
  напрямую с перемоткой мимо ненужного — те же шесть значений за **0.4 с**.
  Недокачанный файл по-прежнему отбраковывается: сверкой смещений тензоров с
  размером файла, а не тем, что их не удалось отобразить в память.
- **Капшенер пишет медиа во временную папку** — IMAGE и VIDEO становятся
  PNG-файлами, AUDIO — 16-битным WAV, написанным средствами стандартной
  библиотеки, — и удаляет её после себя, в том числе если дочерний процесс упал.
- **VIDEO разбирается на кадры здесь, а не отдаётся в `llama-mtmd-cli --video`.**
  Этот флаг скармливает файл `ffprobe` через *stdin*, и если у MP4 атом `moov`
  стоит в начале — а это и есть «faststart», и именно так пишут ComfyUI, телефоны
  и почти весь веб — ffprobe получает всё нужное за пару килобайт и выходит, не
  дочитав остальное. llama.cpp в этот момент всё ещё пишет оставшиеся мегабайты в
  тот же пайп и застревает там навсегда: ни вывода, ни ошибки, ни конца. Тот же
  клип с `moov` в конце отрабатывает за шесть секунд. Кадры декодируются
  внутри процесса, и заодно `max_frames` начинает что-то значить и для VIDEO.
- Ререйтер может добавить детали, которых в коротком промпте не было. Проверяйте
  результат, если внешность, реплики, тайминг или композиция должны быть точными.

## Благодарности

Вся модельная работа — заслуга [LightX2V](https://github.com/ModelTC/LightX2V),
этот репозиторий лишь подключает их адаптер к ComfyUI. Если пригодилось —
поставьте звезду **[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V)**, где
поддерживается инференс MiniMax-H3 и готовятся следующие задачи ререйтера
(FL2VA, Ref2VA).

| Компонент | Источник |
|---|---|
| LoRA-адаптер | [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA) |
| LoRA-адаптер, 8B | [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B) |
| Базовая языковая модель | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Базовая языковая модель, 8B | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Генератор видео и звука | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| Руководства по написанию промтов | [MiniMaxAI/MiniMax-H3 `docs/`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/docs) — скачиваются при работе, см. выше |
| Фреймворк инференса | [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) |
| Встроенные промты и клипы, которые они описывают | [ostris/minimax_h3_1k](https://huggingface.co/datasets/ostris/minimax_h3_1k) — с разрешения автора; промты по сути MIT, а клипы и нарезанные из них кадры — выход MiniMax-H3, обучать на нём модели нельзя |
| Теги манеры съёмки и темы к ним | [H3 Atlas](https://cohub.live/baize/video-altas/w/h3-atlas) |
| Формат RefMod и то, как показываются его референсы | [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod), автор Luisa (luisacaotica), MIT; картинки, хранимые для текстового энкодера, и правило «каждая четвёртая фотография» — от [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder). Читается собственным кодом пака, их код не скопирован |

Шаблоны промптов в `minimax_h3_rewriter/prompt_template.py` и
`prompt_template_8b.py` воспроизведены побайтово из репозиториев соответствующих
адаптеров; их изменение ухудшает качество результата.

Использование MiniMax-H3 регулируется лицензией и правилами допустимого
использования из [официального репозитория MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).

Спасибо [AxiomGraph](https://www.youtube.com/@AxiomGraph) за первый видеообзор
этих нод —
[«Stop Struggling With MiniMax H3 Prompts. Do This Instead.»](https://www.youtube.com/watch?v=h3rZTIRB_G8) —
и за народные воркфлоу, на которые есть ссылка выше.

Спасибо [ЭйАй Генератьон](https://www.youtube.com/@AyiTheDeer) за второй, гораздо более подробный, — в нём
эти ноды показаны не сами по себе, а рядом с остальными паками для MiniMax-H3 —
[«MiniMax H3 - как создать непрерывное длинное видео. Обзор наборов нод для ComfyUI + рерайтер промта»](https://www.youtube.com/watch?v=PZd9fWX15VA).

## Лицензия

MIT — см. [LICENSE](LICENSE). Она распространяется только на код интеграции с
ComfyUI; веса моделей имеют собственные лицензии.


---

## FILE: README_ZH.md (https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/blob/HEAD/README_ZH.md)

# MiniMax-H3 提示词生成与重写节点（ComfyUI）

用于生成、重写和检查 [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
结构化音视频提示词的 ComfyUI 节点包。可以选择 LightX2V LoRA、
[完整 Qwen SFT/GGUF 检查点](docs/qwen_rewriter_ZH.md)、按 MiniMax 官方指南写作的
本地模型，或通过 [API Universal Writer](docs/api_writer.md) 连接已配置的
本地服务器或远程服务。

节点包还提供 MiniMax-H3-Image 和 P-H3 提示词扩写、参考条件处理、已保存的 RefMod、提示词精简、
预设，以及提示词库和模型库。本地写作节点在你的机器上运行；API 请求发送到
你配置的服务。

[English version](README.md) · [Русская версия](README_RU.md) · [更新日志](CHANGELOG.md)

> **本文档为机器翻译。** 本项目作者不懂中文，这份中文 README 是由 AI 从英文版翻译过来的。
> 术语和措辞难免有不准确的地方；如果发现错误、意思偏了或者读不通顺的段落，欢迎到
> [issue](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues)
> 里指出来，我会改。两边说法有出入时，以[英文版](README.md)为准。

<p align="center">
  <a href="https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI?display_name=tag"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <a href="https://registry.comfy.org/publishers/darkil/nodes/minimax-h3-prompt-rewriter"><img alt="ComfyUI Registry" src="https://img.shields.io/badge/ComfyUI-Registry-1B98E0"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA"><img alt="Hugging Face" src="docs/badges/hf-lora.svg"></a>
  <a href="https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF"><img alt="GGUF adapter, 27B" src="docs/badges/gguf-27b.svg"></a>
  <a href="https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-8B-GGUF"><img alt="GGUF adapter, 8B" src="docs/badges/gguf-8b.svg"></a>
  <a href="https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF"><img alt="GGUF adapter, Omni" src="docs/badges/gguf-omni.svg"></a>
  <a href="https://www.youtube.com/watch?v=h3rZTIRB_G8"><img alt="Video review, in English" src="https://img.shields.io/badge/YouTube-review%20(EN)-FF0000?logo=youtube&logoColor=white"></a>
  <a href="https://www.youtube.com/watch?v=PZd9fWX15VA"><img alt="Video review, in Russian" src="https://img.shields.io/badge/YouTube-review%20(RU)-FF0000?logo=youtube&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA"><img alt="LightX2V SFT: Qwen3.5 9B Ref2VA" src="docs/badges/hf-sft-9b-ref2va.svg"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA"><img alt="LightX2V SFT: Qwen3.5 9B FL2VA" src="docs/badges/hf-sft-9b-fl2va.svg"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA"><img alt="LightX2V SFT: Qwen3.8 27B Ref2VA" src="docs/badges/hf-sft-27b-ref2va.svg"></a>
  <a href="https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA"><img alt="LightX2V SFT: Qwen3.8 27B FL2VA" src="docs/badges/hf-sft-27b-fl2va.svg"></a>
</p>

![ComfyUI 中的重写器节点，夹在选项节点和文本查看器之间：节点里是一句简短的提示词，右边是结构化的分镜描述、声景和音乐字段](docs/node_preview.png)

提示词可以用底座模型读得懂的任何语言写；重写的结果一律返回英文 —— MiniMax-H3
要的正是英文。

```text
"A red fox walks through a snowy forest at dawn."  +  16:9  +  15s
                              │
                              ▼
              提示词生成 / 重写节点
                              │
                              ▼
   integrated_multimodal_description: [Shot 1] ... [Shot 2] 0:06 ...
   overall_soundscape: ...
   non_diegetic_music: ...
                              │
                              ▼
              MiniMax-H3: 视频 + 同步音频
```

按模型和参考素材选择节点：

| 节点系列 | 模型或服务 | 参考素材输入 | 任务 |
|---|---|---|---|
| LoRA Rewriter 27B | Qwen3.6-27B + LightX2V LoRA | 文字描述 | T2VA |
| LoRA Rewriter 8B | Qwen3-VL-8B-Instruct + LoRA | 直接读取首尾帧图像 | T2VA, I2VA, FL2VA, L2VA |
| LoRA Rewriter Omni | Qwen2.5-Omni-7B + LoRA | 图像；GGUF 路径还可处理视频和音频 | T2VA, I2VA, FL2VA, L2VA, Ref2VA |
| [Qwen Rewriter](docs/qwen_rewriter_ZH.md) | 完整 LightX2V Qwen3.5 9B / Qwen3.8 27B，SFT 或 GGUF | 直接读取图像和采样视频帧；声音使用文字描述 | FL2VA 检查点：T2VA, I2VA, FL2VA, L2VA；Ref2VA 检查点：Ref2VA |
| 按指南写作的节点 | 本地指令跟随 GGUF 模型 | 由素材描述模型提供文字 | T2VA, I2VA, FL2VA, L2VA, Ref2VA |
| [API Universal Writer](docs/api_writer.md) | 已配置的本地服务器或远程服务 | 按模型/服务器能力提供图像、采样视频和音频，也可使用描述模式 | T2VA, I2VA, FL2VA, L2VA, Ref2VA |

三个 LoRA 版本也可以通过 [Universal Rewriter](#minimax-h3-universal-rewriter-通用重写器)
使用，在标签页中选择适配器。按指南写作的节点可通过
[Reference Caption](#minimax-h3-reference-caption-参考素材描述) 和
[Multi Reference Caption](#minimax-h3-multi-reference-caption-多参考素材描述) 把素材转为文字。
[Universal Writer](#minimax-h3-universal-writer-通用写作节点) 则在一个节点中完成素材描述和
提示词写作，参考素材的顺序由可拖动的芯片控制。

还有一种情况：通往好提示词的最短路径，是别人已经写好的那一条。
[Prompt Presets](#minimax-h3-prompt-presets-提示词预设) 会从节点包内置的一千条
成品 MiniMax-H3 提示词里递给你一条，可以按画面风格和主题筛选，每一条都配着当初
为它写的那段视频的画面，而那段视频也只有一次点击的距离。它不加载任何模型，
也不下载任何东西。

如果你的显卡只有 8 GB，直接跳到[写作节点](#minimax-h3-prompt-writer-t2vai2vafl2val2va-提示词写作节点)。

## 目录

- [安装前需要准备什么](#安装前需要准备什么)
- [安装](#安装)
  - [示例工作流](#示例工作流)
- [节点](#节点)
  - [MiniMax-H3 Prompt Rewriter 提示词重写器](#minimax-h3-prompt-rewriter-提示词重写器)
  - [MiniMax-H3 Prompt Rewriter 8B 看得见画面](#minimax-h3-prompt-rewriter-8b-看得见画面)
  - [MiniMax-H3 Prompt Rewriter Omni 看得见也听得见](#minimax-h3-prompt-rewriter-omni-看得见也听得见)
  - [MiniMax-H3 Qwen Rewriter](#minimax-h3-qwen-rewriter)
  - [MiniMax-H3 Universal Rewriter 通用重写器](#minimax-h3-universal-rewriter-通用重写器)
  - [MiniMax-H3 Rewriter Options 重写器选项](#minimax-h3-rewriter-options-重写器选项)
  - [MiniMax-H3 Prompt Writer (T2VA/I2VA/FL2VA/L2VA) 提示词写作节点](#minimax-h3-prompt-writer-t2vai2vafl2val2va-提示词写作节点)
  - [MiniMax-H3 Prompt Writer (Ref2VA) 提示词写作节点](#minimax-h3-prompt-writer-ref2va-提示词写作节点)
  - [MiniMax-H3 Universal Writer 通用写作节点](#minimax-h3-universal-writer-通用写作节点)
  - [MiniMax-H3 Image Prompt Writer](#minimax-h3-image-prompt-writer)
  - [P-H3 Prompt Writer (图像)](#p-h3-prompt-writer-图像)
  - [MiniMax-H3 Reference Caption 参考素材描述](#minimax-h3-reference-caption-参考素材描述)
  - [MiniMax-H3 Multi Reference Caption 多参考素材描述](#minimax-h3-multi-reference-caption-多参考素材描述)
  - [用 ComfyUI 已经加载好的模型来生成描述](#用-comfyui-已经加载好的模型来生成描述)
  - [描述模型只加载一次，不是每份素材加载一次](#描述模型只加载一次不是每份素材加载一次)
  - [MiniMax-H3 Guide Prompt (any LLM) 指南提示词](#minimax-h3-guide-prompt-any-llm-指南提示词)
  - [MiniMax-H3 Prompt Check 提示词检查](#minimax-h3-prompt-check-提示词检查)
  - [MiniMax-H3 Prompt Reducer 提示词精简器](#minimax-h3-prompt-reducer-提示词精简器)
  - [MiniMax-H3 Reduce Prompt (any LLM) 精简提示词](#minimax-h3-reduce-prompt-any-llm-精简提示词)
  - [MiniMax-H3 Effect Embeddings 特效嵌入](#minimax-h3-effect-embeddings-特效嵌入)
  - [MiniMax-H3 LoRA Triggers LoRA 触发词](#minimax-h3-lora-triggers-lora-触发词)
  - [MiniMax-H3 Reference Adapter 参考素材适配器](#minimax-h3-reference-adapter-参考素材适配器)
  - [MiniMax-H3 Reference Slots 参考素材插槽](#minimax-h3-reference-slots-参考素材插槽)
  - [MiniMax-H3 RefMod 已保存的参考素材](#minimax-h3-refmod-已保存的参考素材)
  - [MiniMax-H3 Reference Encode 参考素材编码](#minimax-h3-reference-encode-参考素材编码)
  - [MiniMax-H3 Prompt Presets 提示词预设](#minimax-h3-prompt-presets-提示词预设)
  - [时长小部件](#时长小部件)
  - [重复上一次的答案](#重复上一次的答案)
  - [卸载留在内存里的模型](#卸载留在内存里的模型)
  - [答案会被检查一遍](#答案会被检查一遍)
  - [照着发现的问题动手](#照着发现的问题动手)
  - [提示词库](#提示词库)
  - [写作指南是取下来的，不是随包附带的](#写作指南是取下来的不是随包附带的)
  - [模型列表](#模型列表)
  - [你为 Ollama 拉下来的那些模型](#你为-ollama-拉下来的那些模型)
- [权重都落在哪儿](#权重都落在哪儿)
- [用你已经有的模型](#用你已经有的模型)
  - [更小的重打包](#更小的重打包)
  - [如果节点说缺了某个包](#如果节点说缺了某个包)
- [什么都不用另装的最小下载量](#什么都不用另装的最小下载量)
- [GGUF —— 还要更小，而且什么都不用装](#gguf--还要更小而且什么都不用装)
  - [节点上的进度](#节点上的进度)
  - [环境变量](#环境变量)
  - [语言](#语言)
- [杂记](#杂记)
- [致谢](#致谢)
- [许可证](#许可证)

## 安装前需要准备什么

走 LoRA 这条路，面对的是一个 270 亿参数的语言模型，不是什么小助手。下面这些
要求绕不过去，因为适配器只和一个特定的底座模型绑定。**写作节点没有这些要求** ——
见下面它们自己的表格。

| 资源 | 要求 |
|---|---|
| 磁盘 | `Qwen/Qwen3.6-27B` 需要 **约 52 GB**，适配器另需 **约 3.5 GB** —— 走 GGUF 路线的话总共 **约 10–16 GB** |
| 显存（`nf4`，默认） | **约 16 GB** |
| 显存（`int8`） | 约 28 GB |
| 显存（`bfloat16`） | 约 54 GB，会通过 accelerate 溢出到系统内存 |
| 显存（GGUF） | 约 13–19 GB，取决于量化等级；卸载到 GPU 上的层数少一些，占用还能更低 |
| 软件包 | `transformers`、`peft`、`accelerate`，`nf4`/`int8` 还需要 `bitsandbytes`。**走 GGUF 路线什么都不用装：** 系统里恰好有 `llama-cpp-python` 就用它，没有就去取 llama.cpp 官方的二进制文件 |

> **MiniMax-H3 的文本编码器不能拿来干这件事。** 它和 LoRA 的底座是两个不同的模型
> （Qwen3-VL-32B，词表 151936；底座是 Qwen3.6-27B，词表 248320）；它里面没有适配器
> 要挂上去的那些线性注意力 `in_proj_*` 模块；它被截断到 64 层里的前 50 层；而且它
> 不带 `lm_head`，也没有最后的归一化层，所以它根本没法生成文本。它只为 DiT 产出
> 隐藏状态。

## 安装

克隆到 `ComfyUI/custom_nodes/`，再把依赖装进 ComfyUI 实际运行的那个 Python 环境：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI
```

Windows 上的 ComfyUI 便携版：

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\MiniMax-H3-Prompt-Rewriter-ComfyUI\requirements.txt
```

或者用 ComfyUI-Manager 从 Comfy registry 安装。

### 示例工作流

节点包自带十二个工作流。装好之后，它们会出现在 ComfyUI 的模板浏览器里
（*Workflow → Browse Templates*），归在本节点包的名字下面。每一个都是独立的一张
卡片，也都能独立跑起来：打开时没有任何节点被绕过，也不必先静音另一条分支才能
按下运行。

| | 模板 | 需要什么 |
|---|---|---|
| 1 | **Write a prompt** —— 输入一句想法，输出一段完整的 H3 音视频描述 | 一个 2.6 GB 的 GGUF |
| 2 | **Rewrite a prompt with the 27B LoRA** —— 本节点包名字的由来，LightX2V 的那个适配器 | 一个 15.7 GB 的 GGUF |
| 3 | **Write a prompt from references** —— 写作节点先描述你的图片，再照着看到的内容写 | 两个 GGUF，合计 6 GB |
| 4 | **Ready-made prompts** —— 一千条现成的提示词，在浏览器里连着画面一起挑 | 什么都不需要 |
| 5 | **Prompt to video** —— ComfyUI 自带的文生视频模板，前面接上写作节点 | MiniMax-H3 的权重 |
| 6 | **References to video** —— 同样的东西，换成 Ref2VA，图片经由写作节点、按素材条的编号顺序送到生成器 | MiniMax-H3 的 ref2va 权重 |
| 7 | **LoRA triggers and effects** —— 适配器要听的那些词，和 MiniMax 的十个特效，放进一段写好的提示词里 | 什么都不需要 |
| 8 | **RefMods to video** —— 一份 [RefMod](#minimax-h3-refmod-已保存的参考素材)，也就是已经编码好保存下来的外观和声音：写作节点能看见也能听见它，生成器拿到的就是它保存时的潜变量 | MiniMax-H3 的 ref2va 权重，以及 `models/refmods` 里的一份 RefMod |
| 9 | **P-H3 text to image** | P-H3 图像权重和提示词扩写模型 |
| 10 | **P-H3 image editing** | P-H3 权重、扩写模型和你的图像 |
| 11 | **Write through an API** | 已配置的本地或远程 API 服务器 |
| 12 | **Write with the full Qwen model** | 9B Q5_K_M 与 F16 视觉投影器，合计 7.39 GB |

5、6 和 8 生成 MiniMax-H3 视频，9 和 10 生成 P-H3 图像。其他工作流生成或处理提示词。5 和 6 是 ComfyUI 自己图库里的模板，把生成那一半折进了一个
子图方框，所以屏幕上剩下的就是提示词这一侧再加一个节点 —— 它们点名的检查点就是
官方模板用的那些，缺了哪个 ComfyUI 会主动提出替你下载。8 就是 6，只是那个方框里用
[Reference Encode](#minimax-h3-reference-encode-参考素材编码) 换掉了原生的参考节点，
两个 VAE 搬到了方框外面，和 RefMod 共用。

每一个都带着一张 **Read me first** 说明：它做什么、会下载什么、按下运行之前要设置
什么、接下来该往哪儿走。3 和 6 里面的图像加载器和 8 里的 RefMod 选择**故意是空的** ——
存进模板里的文件名指向的是你机器上没有的东西 —— 所以先挑你自己的。

浏览器在每个名字旁边画的那张卡片图来自 `python tools/template_cards.py`：它把图片
写成 `<name>.jpg`，放在工作流旁边，也就是浏览器会去找的地方。

社区那边也有一份：[axiomgraph 的工作流](https://github.com/axiomgraph/ComfyUIWorkflow)
把 Omni 重写器和 FL2VA 搭在一起（GPL-3.0 协议，另外还用了几个别的节点包 ——
去他们的仓库里拿）。

## 节点

### MiniMax-H3 Prompt Rewriter 提示词重写器

主力节点。缺什么就下载什么，加载模型，生成，然后把显存再还回去。

**输出**

| 名称 | 内容 |
|---|---|
| `rewritten_prompt` | 完整的重写结果，可以直接粘进 MiniMax-H3 的文本输入 |
| `integrated_multimodal_description` | 只有逐镜头的画面部分 |
| `overall_soundscape` | 只有画内声音的部分 |
| `non_diegetic_music` | 只有配乐的部分 |

**输入**

- `prompt` —— 要展开的那句短提示词。
- `model` —— 底座模型。列表里既有你模型列表中的每一项，也有硬盘上已经存在的每一个
  Qwen3.6-27B（前缀是 `on disk:`）。不在本地的会在第一次使用时下载，中断了可以续传。
  **Model list** 按钮会在画布上方开一个窗口来编辑这份列表 —— 见后文。
- `resolution` / `duration` —— 重写要照着哪些条件来写。把它们和你交给 MiniMax-H3 的
  设置保持一致，否则镜头节奏对不上。列表从 `48:9` 一直排到 `9:16`；两个超宽比例是
  多显示器的情形 —— 32:9 是两块 16:9 屏幕并排，48:9 是三块。它们决定的是镜头怎么构图，
  而这也正是本节点唯一决定的事；生成器在那个形状下会渲染出什么，是另一回事。
- `aspect_ratio` —— **同一个设置，只是挂在插槽上**，而且只要有东西连着它就说了算。
  节点包里每一个写作节点和重写节点都有它，它接受 `STRING` 或 `COMBO` 连线，所以那个
  已经在给 ComfyUI 的 **Resolution Selector** 设值的 Primitive 节点，可以用同一根线
  顺带驱动它。它存在的理由是：画面形状通常是在图里别的地方定下来的，而那边的写法
  各不相同 —— ComfyUI 自己的 **Resolution Selector** 把 16:9 叫作 `16:9 (Widescreen)`，
  尺寸节点说的是 `3840x1080`，除法节点说的是 `1.78`。这三种都能读，标签把数字对包在
  里面也能读穿 —— 真正算数的是那一对数字。画面尺寸只要和列表里某个比例相差不超过
  2%，就按那个比例的名字算，所以 `1376x768`（Resolution Selector 在 1 MP 下为 16:9
  给出的正是这个尺寸）到这里是 `16:9`，而不是 `43:24`；列表里完全没有的比例则按原样
  通过，`2.39:1` 或者 `5:4` 就是这么进来的。至于根本不是比例的东西，会被点名拒绝，
  而不是拿去照着构图。`resolution` 小部件本身没有插槽，所以入口只有一个，而这个入口
  读的正是别的节点写出来的东西。**连线接上时，选择器会变灰** —— 暗掉，没有任何一项
  亮着，点击也不响应 —— 因为亮着的那一格会指向一个这次运行根本不会用的比例。
  **拔掉连线会清空这个字段**：上游节点是把自己的值写进小部件里的 —— 连线喂给小部件
  输入本来就是这么工作的 —— 否则那段文字会留在原地，而它待在那儿并不是不起作用的。
  `aspect_ratio` 里的任何东西都压过选择器，残留值也算。
- `quantization` —— 怎么加载一个*未量化*的检查点：`nf4`（默认，约 16 GB 显存）、
  `int8`（约 28 GB）、`bfloat16` / `float16`（约 54 GB）。检查点自带量化时，这一项被忽略。
- `greedy` —— 默认开启，输出是确定性的。想采样就把它关掉。
- `seed`
- `keep_model_loaded` —— **默认关闭。** 重写一结束，27B 模型就被释放，好让同一块 GPU
  接着去跑 H3 视频生成。只有在连着不停地打磨提示词时才打开它；之后怎么把显卡要回来，
  见[卸载留在内存里的模型](#卸载留在内存里的模型)。
- `options` —— 可选；接一个 **MiniMax-H3 Rewriter Options** 节点。

### MiniMax-H3 Prompt Rewriter 8B 看得见画面

同样的思路，换到一个小得多、而且是多模态的模型上。LightX2V 的第二个适配器训练在
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 上，所以
27B 必须*被人告诉*参考帧里有什么，而这一个是直接看到那张画面，照着看到的内容写出
对齐行（开头那句告诉 H3 参考帧落在什么位置的固定语句）。它覆盖四个任务而不是一个，
而且装得进 27B 根本靠近不了的显卡。

![ComfyUI 中的 8B 重写器节点，任务设为 T2VA：左边是选项节点，中间是带 first_frame 和 last_frame 输入的重写器，右边是写好的结果，里面有编号的镜头、(S1) 说话人标识，以及一条 <d>[English] Hello.</d> 对白标签](docs/node_rewriter_8b.png)

**输出**和上面那个重写器完全一样的四个，所以两者在下游可以互换。

**输入**

- `prompt`、`resolution`、`duration`、`greedy`、`seed` —— 同上。
- `model` —— 一个 Qwen3-VL-8B 底座，适配器发布成什么形状，这里就支持什么形状。
  **GGUF** 条目是同一次转换产出的两个文件：模型本体和它的投影器。**safetensors**
  条目是官方的
  [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
  文件夹，适配器就是在它上面训练、也是按它发布的。带 `on disk:` 前缀的条目已经在
  你的模型目录里了。**只有 8B 才配得上这个适配器**；别的尺寸的 Qwen3-VL 会在下载
  任何东西之前，就按名字和数字被拒掉。
- `quantization` —— 怎么加载一个 **safetensors** 底座：`nf4` 需要大约 8 GB 显存，
  `int8` 大约 13 GB，`bfloat16` 大约 20 GB。GGUF 自带量化，这一项对它无效。
- `task` —— `T2VA`、`I2VA`、`FL2VA`、`L2VA`。模型自己给这几个任务起的名字是 T2AV、
  I2AV、FL2AV 和 L2AV；指的是同样这四个任务。
- `first_frame` / `last_frame` —— 可选的 IMAGE 输入。`I2VA` 读 `first_frame`，
  `L2VA` 读 `last_frame`，`FL2VA` 两个都读，`T2VA` 一个都不读。接错了的话，节点会在
  加载任何东西之前就说出缺的是哪一个 —— 一张图属于片段的哪一头，本身就是要告诉
  模型的信息之一。
- `keep_model_loaded` —— 在 **safetensors** 底座上，四个任务都认这个开关：模型加载在
  ComfyUI 自己的进程里，就留在那儿。在 **GGUF** 底座上，只有 `T2VA` 能做到，因为带
  画面的那三个任务是走 `llama-mtmd-cli` 的，每次都是一个新进程，退出时把模型一起带走。
  节点会说明它实际做了哪一种，而不是默默忽略这个开关。
- `options` —— 和其他节点用的是同一个选项节点。它的 `adapter` 下拉框里两个 LoRA 都在；
  第一项会自动挑中与你所选底座相匹配的那一个，所以不用管它。

**它的开销**

| | 下载 | 显存 |
|---|---|---|
| Q4_K_M 底座 + 投影器 + Q8_0 适配器 | 4.7 + 0.7 + 0.7 GB | 约 9 GB |
| Q8_0 底座 + 投影器 + F16 适配器 | 8.1 + 0.7 + 1.3 GB | 约 13 GB |
| safetensors 底座 + 适配器，`nf4` | 17.5 + 2.8 GB | 约 8 GB |
| safetensors 底座 + 适配器，`bfloat16` | 17.5 + 2.8 GB | 约 20 GB |

GGUF 那两行还要额外付一次运行时的钱，每台机器一次：`I2VA`、`FL2VA` 和 `L2VA` 走的是
`llama-mtmd-cli`，所以其中第一个跑起来时会去取 llama.cpp 的官方构建 —— 34 MB，如果
`llama_backend` 解析到 CUDA 则是 511 MB —— 而且已经装好的 `llama-cpp-python` 顶不了
这个位置。`T2VA` 是唯一一个可以改用 wheel 的任务。

GGUF 路线的下载量小得多，而且什么都不用装。safetensors 路线是适配器发布时的原始形状，
四个任务都能让模型常驻显存而不只是 `T2VA`，如果你本来就有这个检查点，那就该选它 ——
但它需要 `transformers` 和 `peft`，这两个包节点包已经列进依赖里了。

**该对它抱什么期待。** 四个任务产出的都是训练好的那个形状：`T2VA` 上来就直接写三个
字段，另外三个则以 MiniMax-H3 自己会读的那句对齐语句开头。适配器带来的可见差别是
镜头标记 —— 把 `use_lora` 关掉，同一个模型照样会把三个字段填满，因为格式约定写在
系统提示词里，但它不再写 `[Shot 2]` 这样的切分标记，而且答案的长度大约只剩三分之一。

它终究是个 8B，这一点在一个地方露了馅：对齐行里的时间戳有时会写成三位小数而不是
两位，而在 `FL2VA` 上，最后那张图偶尔会被记到 `Shot 1` 头上，而不是最后一个镜头。
27B 不会这样。下游没有任何东西会去解析那一行，所以它不会让任何地方出错 —— 但粘贴
之前值得扫一眼。

### MiniMax-H3 Prompt Rewriter Omni 看得见也听得见

LightX2V 的第三个适配器，也是第一个会听的。它训练在
[Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) 上 —— 正是本节点包
的描述节点已经在用的那个模型 —— 所以参考素材是以它本来的样子送到它面前的：图片、
片段，或者声音。它也是三个里面唯一覆盖 **Ref2AV** 这个全参考任务的，而这个任务回答
的是六个字段而不是三个。

![设为 REF2AV 的 Omni 重写器节点，左边接着一个 Load Video 节点和一个 Load Image 节点：左边一列四个参考插槽，每一行都带一个复选框；右边九个输出，从 rewritten_prompt 一直到 references；中间横着一条三个彩色方块的素材条 —— ref_0 上面是蓝色的 “pic 1”，ref_1 上面是紫色的 “aud 1”，ref_2 上面是蓝色的 “pic 2” —— 方块下面写着 “拖动可重新排序 —— 顺序决定标签的编号 —— 点击方块可以把它关掉”。再往下是五个任务，REF2AV 亮着；八个按真实比例画出来的画面比例方框，从 48:9 到 9:16，选中的是 16:9；时长 10.0；一段两句话的提示词；模型下拉框里是一个 Qwen2.5-Omni-7B GGUF Q8_0；底部五个按钮：Model list、Unload model、Save the last prompt、Edit the last prompt 和 Prompt library。右边写好的结果把 Ref2AV 的六个字段全都填满了，在两个镜头里点名了 Subject 1、Subject 2、Picture 1、Picture 2 和 Audio 1](docs/node_rewriter_omni.png)

**输出。** 一共七个，哪些会被填满取决于任务。四个带画面的任务返回的是和另外两个
重写器一样的那三个字段，所以下游可以互换。`Ref2AV` 返回六个：`subject_definitions`、
`summary`、`retention_analysis`、`detailed_description`、`overall_soundscape` 和
`non_diegetic_music` —— 和 [Ref2VA 写作节点](#minimax-h3-prompt-writer-ref2va-提示词写作节点)
产出的是同一组，含义也相同。`rewritten_prompt` 永远装着完整的答案。它们后面是
`references`，它不是文本：是按素材条顺序排好的、打开着的参考素材，交给
[Reference Slots](#minimax-h3-reference-slots-参考素材插槽)，按提示词里的编号递给
`MiniMaxH3ReferenceToVideo`。

**输入**

- `prompt`、`resolution`、`greedy`、`seed` —— 同上。
- `references` —— 一个会自动增长的插槽，接受 IMAGE、VIDEO 或者 AUDIO。这里不存在
  “插错插槽”：一份参考素材叫什么，是由它本身是什么决定的。图片在图片之间编号，声音
  在声音之间编号，所以在两张图片中间接进一段声音，不会把它们重新编号。
- `task` —— `T2AV`、`I2AV`、`L2AV`、`FL2AV`、`REF2AV`。只有 `REF2AV` 收片段和声音；
  另外四个只照着图片来写，往它们身上接一段声音会在加载任何东西之前就被点名拒绝。
- `duration` —— **节点会把它吸附到网格上。** MiniMax-H3 是在 24 fps 下按 17n+5 的
  帧数网格生成的，所以大多数长度根本不存在：你要 10 秒，实际是 243 帧、10.13 秒，
  而写进这一轮请求、并在对齐行里被引用的正是*后面这个*数字。小部件写的是你的本意；
  对齐行和视频能对上，靠的就是这次吸附。
- `model` —— 一个 Qwen2.5-Omni-7B 底座，可以是 **GGUF** 组合（模型本体加它的投影器），
  也可以是官方的 **safetensors** 文件夹。带 `on disk:` 前缀的条目已经在你的模型目录里。
  有两类“差一点就对”的情况是被标出来而不是被藏起来的：Qwen2.5-Omni-**3B** 会标成
  `(wrong size for the adapter)`；而 **Qwen2.5-VL-7B** —— 它的架构字符串一样、同样是
  28 个块、宽度也一样，所以适配器*确实*挂得上去 —— 会标成
  `(vision only, not an Omni build)`，因为它的投影器里没有音频编码器，写出来的重写
  会去描述一段从来没被听见过的声音。**Model list** 按钮编辑的是 `models_omni`
  这一节 —— 见后文。
- `quantization` —— 怎么加载一个 **safetensors** 底座：`nf4` 约 9 GB 显存，`int8`
  约 12 GB，`bfloat16` 约 20 GB。对 GGUF 无效。**挑你的显卡装得下的最大那个** ——
  见后文。
- `max_frames` —— 从一段片段里取多少帧，均匀分布。对模型来说，每一帧都是一张独立的图片。
- `reference_layout` —— 素材条的状态，以 JSON 保存。做成小部件是为了让这套排布能跟着
  工作流走，也能穿过 API；界面上则把它画成一排方块。
- `keep_model_loaded` —— 在 **safetensors** 底座上有效。在 **GGUF** 底座上，带参考
  素材的任务走的是 `llama-mtmd-cli`，那是一个新进程，退出时会把模型一起带走，节点
  会把这一点说出来，而不是默默忽略这个开关。
- `options` —— 和其他节点用的是同一个选项节点。

**素材条就是顺序本身。** 每一份连上来的参考素材都会显示成一个彩色方块 —— 图片是蓝色，
片段是绿色，声音是紫色 —— 上面写着它将被叫作什么、来自哪个插槽。拖动就能重新排序，
而这个顺序决定了标签的编号：把第二张图片拖到最前面，它就成了 `<Picture 1>`。点一下
方块可以在不拔线的情况下把它关掉。任务条会把当前这批参考素材伺候不了的任务置灰，
鼠标悬停时说明原因，所以只有一张图片时的 `FL2AV` 是明摆着不可用，而不是两分钟之后
才失败。

这里**故意没有重新贴标签这回事**，和 Universal Writer 的素材条不一样。在那边，一张
图片可以被指定去代表一个主体或者一段片段，因为那种由写作指南驱动的请求是手工点名
自己的参考素材的。而在这里，插槽就把事情定死了 —— 而且*主体*在这个适配器这儿是它
**产出**的东西，写在 `subject_definitions` 里，不是请求提供给它的东西。

**点片段的标签条，会把它自带的声音也打开。** 在 `Ref2AV` 上标签条变成紫色，写着
`vid+a1`：片段的声音现在是 `<Audio 1>`，一份独立的参考素材，适配器能听到它 —— 它作为
一段单独的声音紧挨在片段前面递进去，这也是它的上游推理传递片段声音的唯一方式。片段的
声音先拿到 `<Audio N>` 的编号，独立的声音排在后面，因为 `MiniMaxH3ReferenceToVideo`
就是这样编号的；所以素材条里排在发声片段前面的声音方块，会被放到它后面去递。在其他
任务上，标签条保持浅紫色、不带编号：已经设好，等到 `Ref2AV` 才生效。没有音轨的片段会在
节点上点名说明，不给编号。

**它的开销**

| | 下载 | 显存 |
|---|---|---|
| Q4_K_M 底座 + 投影器 + Q8_0 适配器 | 4.4 + 1.4 + 0.34 GB | 约 9 GB |
| Q8_0 底座 + 投影器 + F16 适配器 | 8.1 + 1.4 + 0.65 GB | 约 13 GB |
| safetensors 底座 + 适配器，`nf4` | 22.4 + 1.3 GB | 约 9 GB |
| safetensors 底座 + 适配器，`bfloat16` | 22.4 + 1.3 GB | 约 20 GB |

GGUF 底座还要为 llama.cpp 运行时付一次钱：除了 `T2VA`，每个任务都带参考素材，因而
都走 `llama-mtmd-cli` —— 34 MB，如果 `llama_backend` 解析到 CUDA 则是 511 MB，而且
装没装 `llama-cpp-python` 对这件事毫无影响。

GGUF 适配器是用 llama.cpp 的 `convert_lora_to_gguf.py` 从 LightX2V 自己的 safetensors
转出来的，发布在
[pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-Omni-GGUF)。
`Q8_0` 版本只有 `F16` 的一半大，表现完全一样。

**量化买来的是显存，不是速度。** 这和直觉正好相反 —— 更小的模型应该搬运更少的字节、
跑得更快才对。在这个适配器上、同一块显卡、同样的提示词、同样两张图片、同样 256 个
token 的实测结果：

| | 显存 | 生成速度 |
|---|---|---|
| `bfloat16` | 19.4 GB | **18.6 tok/s** |
| `nf4` | 8.6 GB | 15.1 tok/s |
| `int8` | 12.0 GB | 6.0 tok/s |

`int8` 在两项上都是三个里面最差的：比 `nf4` 慢，*而且*比它还大。这不是这个适配器
的怪癖 —— bitsandbytes 的 `load_in_8bit` 就是 LLM.int8()，它把每一次矩阵乘法拆成
一个 fp16 的离群值部分和一个 int8 部分再合并起来，每次都要给激活值做类型转换。它是
一套用来“把本来装不下的模型装进去”的方案，代价就是这么多。`nf4` 是更好的那个小选项，
但它同样在每次矩阵乘法时反量化，所以它也赢不了 `bfloat16`。因此只量化到显卡真正
装得下的那一档就行：有 24 GB 或更多的话，`bfloat16` 既最快也最忠实。

这条结论对本节点包里任何一个 safetensors 底座都成立，因为这个机制是 bitsandbytes
的，不是模型的。**而它对 GGUF 路线完全不适用**，那边 llama.cpp 有真正的量化算子：
同一个 FL2AV，走 `bfloat16` safetensors 要 26 秒，走 `Q4_K_M` 只要 10 秒。

**图片在模型看到之前会先被缩放。** LightX2V 自己的推理脚本把一张图片的上限定在
301056 像素 —— 也就是 384 个 token —— 片段里的一帧则是 100352，本节点照做。这不只是
省一点：一张 1616×1616 的图片是 3249 个 token，两张就能在提示词还没数进去一个字的
时候撑爆 8k 上下文，而且这等于是拿一个它训练时从没见过的形状去喂模型。上下文随后
会按这一轮请求的实际开销来定大小，所以带八份参考素材的 `Ref2AV` 是把上下文撑宽，
而不是直接失败。

**safetensors 路线只能看图。** ComfyUI 进程内的 Transformers 路径没有办法把一段声音
递给模型，所以在那种底座上接一段片段或者声音会被拒绝，而不是从请求里悄悄丢掉。
想用它们，就挑一个 GGUF 底座。

### MiniMax-H3 Qwen Rewriter

加载 LightX2V 的完整 **Qwen3.5 9B**、**Qwen3.8 27B** SFT 模型，
也支持带配套视觉投影器的 GGUF 转换。FL2VA 模型用于 T2VA、I2VA、L2VA 和 FL2VA；
Ref2VA 模型用于 Ref2VA。通过 **Model list** 选择原始 SFT 或 GGUF，
生成参数和参考素材设置位于独立的 **Qwen Rewriter Options** 节点。

参考芯片位于顶部，保留输入复选框、**+ instr**、宽高比、时长、bypass/repeat、
提示词库和 `references` 输出。模型直接读取图像和采样视频帧。
这些模型没有音频编码器，声音通过明确的文字描述提供。

| 模型 | 原始 SFT | GGUF Q5_K_M / Q4_K_M |
|---|---|---|
| Qwen3.5-9B FL2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF) |
| Qwen3.5-9B Ref2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA-GGUF) |
| Qwen3.8-27B FL2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA-GGUF) |
| Qwen3.8-27B Ref2VA | [LightX2V](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA) | [pytraveler](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-GGUF) |

[设置、参考素材角色、输出与转换说明](docs/qwen_rewriter_ZH.md)。

![Qwen Rewriter Options、完整 Qwen Rewriter 和生成的 H3 提示词](docs/qwen_rewriter_workflow.png)

### MiniMax-H3 Universal Rewriter 通用重写器

三个提示词重写 LoRA 装在一个节点里，顶上的标签页决定跑哪一个。
[Prompt Rewriter](#minimax-h3-prompt-rewriter-提示词重写器)、
[Prompt Rewriter 8B](#minimax-h3-prompt-rewriter-8b-看得见画面) 和
[Prompt Rewriter Omni](#minimax-h3-prompt-rewriter-omni-看得见也听得见)
原封不动，也仍然都在 —— 你已经搭好的东西不会因此停摆。

这三个适配器不是同一个东西的三档设置。27B 是纯文本的：Qwen3.6-27B，一个任务，参考帧
只能以别人写下的一句话的形式抵达它。8B 是多模态的：Qwen3-VL-8B，四个任务，收的是
图片本身。Omni 是多模态而且还会听：Qwen2.5-Omni-7B，同样那四个任务，外加它自己独有
的第五个。底座不同、尺寸不同、下载量也不同。

正因为如此，手工在它们之间来回切换才格外磨人。提示词还是那句提示词，画面比例还是
那个画面比例，时长还是那个时长 —— 于是“换个适配器试试”就意味着把这一切重新敲进
第二个节点，然后还得让两边一直保持同步。

![处在 Omni 标签页、正在跑 I2VA 的通用重写器：左边一列四行参考素材 —— first_frame、last_frame、reference_video 和 reference_audio，每一行都带一个复选框，first_frame 上接着一个 Load Image 节点 —— 右边九个输出，从 rewritten_prompt 一直到 references。顶上横排三个标签页，亮着的是 “Omni LoRA / sees, hears”；任务条上 I2VA 亮着；八个按真实比例画出的画面比例方框，从 48:9 到 9:16，选中的是 16:9；时长滑块停在 10；提示词是 “A woman says hello to a alien.”；下面 model_omni 选的是一个 Qwen2.5-Omni-7B GGUF Q8_0，quantization_omni 设为 bfloat16；再往下是 aspect_ratio、repeat_last 和五个按钮：Model list、Unload model、Save the last prompt、Edit the last prompt 和 Prompt library。右边是答案：先是一行，说明 Picture 1 从第一个镜头起被完整引用，然后是 I2VA 填的那三个字段](docs/node_universal_rewriter.png)

*一个节点，三个标签页。模型那几行以上的东西不属于任何标签页 —— 任务、比例、时长和
提示词是同一组值，无论哪个适配器亮着都一样。它们下面，每个标签页各有自己的模型和
量化设置，而且始终停在你上一次选的那个上，保存再打开也不变。任务条是另一处差别：
27B 标签页只点亮 `T2VA`，8B 标签页加上三个带画面的任务，Omni 标签页则在此之上再加
`Ref2VA` —— 那是别的标签页够不着的。*

**所以标签页装的只是不同的那部分，别的都不装：**

| 属于标签页 | 三者共用 |
|---|---|
| `model_27b` / `model_8b` / `model_omni` | `prompt`、`task`、`resolution`、`duration` |
| `quantization_27b` / `quantization_8b` / `quantization_omni` | `greedy`、`seed`、`keep_model_loaded`、`bypass`、`options`，两个画面帧、片段和声音 |

另一个标签页用的那个小部件是被藏起来，而不是被重置，所以你切回去的时候，它还停在
你上一次选的那个值上 —— 保存再打开也一样。

> **27B 标签页上故意没有描述节点。** 把一张画面的文字描述折进提示词里，确实能送到
> 那个适配器面前，也不算白费 —— 里面的道具、材质和光线会以训练好的那个形状出现在
> 镜头里，不会有什么东西渗进答案。但那张图片是被吸收进场景里，而不是被钉在 0.00 秒
> 那个位置上，而这正是 [LoRA 自己那一页](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA)
> 所说的：T2VA 在那边是完成了的，FL2VA 不是。要是在这个节点上给它做一个小部件，看起来
> 就像 27B 能做那个它其实做不了的画面任务。真需要的时候，
> [Reference Caption](#minimax-h3-reference-caption-参考素材描述) 会写出那段描述，
> 它像任何别的文字一样进 `prompt`；而当那张图片必须*就是*一帧画面时，8B 标签页才是
> 为此训练过的那个。

**任务开关是共用的，27B 标签页不碰它。** 在那个标签页上，它显示成 `T2VA` 亮着、三个
带画面的任务灰着，因为这才是一个纯文本模型的诚实模样，而且点了完全没有反应 ——
另外两个标签页原来的值还在，切回去就看得到。在 8B 和 Omni 标签页上，一个带画面的
任务会一直灰着，直到它要照着写的那一帧真的接上并且打开为止，和通用写作节点上的
`Ref2VA` 是一个道理。

**`Ref2VA` 在 Omni 标签页上，配着一个片段插槽和一个声音插槽喂给它。**
`reference_video` 和 `reference_audio` 只被这个任务读，别的谁都不读 —— 27B 和 8B
适配器没有耳朵，四个带画面的任务也只收图片，所以往 `FL2VA` 上接一段声音会被点名
拒绝，而不是悄悄丢掉。在 `Ref2VA` 上，凡是接上来的东西都会成为目标视频要复用的一份
参考素材，按插槽顺序排：`first_frame` 是 `<Picture 1>`，`last_frame` 是 `<Picture 2>`，
然后是 `<Video 1>`，再然后是 `<Audio 1>`。答案回来的是六个字段而不是三个；多出来的
四个输出排在每个任务都会填的那三个**后面**，所以已经接在这个节点上的线一根都不用挪。
排在最后的是 `references`：当前标签页真正读过的参考素材，交给
[Reference Slots](#minimax-h3-reference-slots-参考素材插槽) 递给
`MiniMaxH3ReferenceToVideo`。

> **四份参考素材，不是十二份。** 标签规则全在顺序上，而只有四个插槽时，顺序就是插槽
> 的顺序。再往上，你真正想要的其实是手工摆放它们 —— 那正是
> [Prompt Rewriter Omni](#minimax-h3-prompt-rewriter-omni-看得见也听得见) 上那条可拖动
> 素材条的用途，连同 `max_frames` 和任意多的图片。这个标签页管的是另一件事：不离开
> 节点，就把同一句提示词换个适配器试一遍。

**四个参考输入，每一行带一个复选框。** `first_frame` 和 `last_frame` 就是 8B 节点上
那两个；`reference_video` 和 `reference_audio` 是 `Ref2VA` 追加的那两个。关掉的一行
等同于没插 —— 想把一份参考素材先搁一边而又不用把线拽下来，就是这么做的。要是在
27B 标签页上连着它们中的任何一个开跑，节点会在自己身上写明它没在读这些东西，以及
该把它们放到哪儿去，而不是留你一个人纳闷。

**`reference_video` 那一行还有第二个框，画着一个喇叭。** 勾上它，在 Omni 标签页的
`Ref2VA` 上，片段自带的声音就成为一份独立的参考素材：`<Audio 1>`，而 `reference_audio`
顺延为 `<Audio 2>` —— 这正是 `MiniMaxH3ReferenceToVideo` 给它们的顺序 —— `references`
会把这一点带给 [Reference Slots](#minimax-h3-reference-slots-参考素材插槽)。

**`duration` 是一个滑块**，在你没动它之前一直能拉到 30 秒 —— 右键点节点，选
`duration`。见[时长小部件](#时长小部件)。三个适配器都是在几秒钟的片段上训练出来的，
所以一个远超这个范围的数字换来的是一句更糟的提示词，而不是一段更长的视频 ——
MiniMax-H3 的长度是从它自己的设置里拿的，不是从这一行拿的。

**这里没有 “Open guide folder” 按钮**，因为三个适配器谁也不读写作指南：格式就在
它们训练时用的系统提示词里。**Model list** 按钮是有的，它会把这个节点的三份列表当作
三个标签页在同一个窗口里打开 —— 27B 标签页用 `models`，8B 用 `models_8b`，Omni 用
`models_omni`。

> **如果界面脚本没能加载**，标签条、任务开关和比例选择器会回退成它们底下那些普通
> 下拉框，而且所有小部件会一次全部显示，而不是一次只显示一个标签页的量。不管哪种
> 情况，节点跑起来完全一样。这三个控件是 HTML 的，在 ComfyUI 的两种渲染器里都能用；
> 而两个画面行上的复选框是画在画布上的，*Modern Node Design (Nodes 2.0)* 不跑画布
> 绘制，所以在那种模式下，想排除一帧画面就得把它的线拔掉。

> **这个节点需要较新的 ComfyUI**，它是照着 v3 节点 API 写的，和通用写作节点一样。
> 装在旧版本上它会直接不出现，节点包的其余部分照常注册。

### MiniMax-H3 Rewriter Options 重写器选项

所有你很少去动的东西，都从主节点上挪到了这里。不接它，重写器就用适配器发布时
自带的那套解码参数。

![Rewriter Options 在生成设置下方显示 Default guide 和 System prompts 按钮](docs/node_options.png)

| 输入 | 默认值 | 作用 |
|---|---|---|
| `max_new_tokens` | 2048 | 生成长度上限 |
| `temperature` / `top_p` / `top_k` | 0.7 / 0.8 / 20 | 采样参数，只有 `greedy` 关掉时才用得上 |
| `repetition_penalty` | 1.05 | |
| `attn_implementation` | `sdpa` | 有的话也可以换成 `eager` 或 `flash_attention_2` |
| `adapter` | LightX2V 那个仓库 | 套用 LoRA 的哪一个构建 —— 见下文 |
| `use_lora` | 开 | 关掉它，就是不带适配器的底座模型基线 |
| `merge_lora` | `auto` | 加载时就把适配器折进权重里 —— 每秒 token 数翻倍，见下文 |
| `auto_download` | 开 | 关掉它，缺东西就直接报错，而不是去拉 52 GB |
| `downloader` | `built-in` | 由哪段代码去取 Hugging Face 仓库 —— 见下文 |
| `device` | `auto` | 哪一块 GPU 跑这个语言模型 —— 见下文 |
| `prompt_file` | `global` | 接在这个节点上的那些节点，在哪一套已保存的提示词里工作 —— 见[提示词库](#提示词库) |
| `self_check` | `warnings and notes` | [自检](#答案会被检查一遍)的结果说出多少：全说、只说警告，或者什么都不说 |
| `fix_once` | `false` | 让节点照着检查发现的问题动手 —— [只重跑一次，绝不成环](#照着发现的问题动手) |
| `captioner_thinking` | `off` | GGUF 参考描述：`off` 关闭思考，`auto` 保留模型模板默认值，`on` 在二进制支持时启用思考。 |

GGUF 描述会通过 `--help` 检查可用参数，每个可执行文件版本只检查一次。
`off`/`on` 使用 `--reasoning`，并在支持时启用 `--jinja`。如果
`llama-mtmd-cli` 无法控制思考，则使用同一目录中支持该参数的 `llama-server`，
单个参考也适用。多个参考共用一次模型加载；运行结束或取消后关闭服务器。
`MINIMAX_H3_MTMD_SERVER=never` 保留 CLI 路径。如果可用二进制均无法应用此设置，
则使用模板默认值并记录警告，不传入未知参数。描述仍最多生成 1024 个 token，
空的 GGUF 描述会报错。此设置不改变写作模型或 CLIP 描述路径的参数。

同一个选项节点也喂给写作节点和描述节点；在那些节点上，`adapter` 和 `use_lora`
单纯用不上。

#### `merge_lora` —— 把适配器当成权重，而不是第二次矩阵乘法

PEFT 会把挂上去的 LoRA 放在底座模型旁边，每出一个 token 就在底座权重之上把它算
一遍。在加载时一次性折进去，等于把同样的算术提前做完，而这个差别并不小。下面是在
Qwen3-VL-8B 加 8B 适配器、5090 上生成 128 个 token 的实测：

| | 适配器挂着 | 折进去 | 折叠本身的开销 |
|---|---|---|---|
| `bfloat16` | 8.98 秒 —— 14.3 tok/s | **5.13 秒 —— 25.0 tok/s** | 0.07 秒 |
| `nf4` | 9.43 秒 —— 13.6 tok/s | **5.09 秒 —— 25.1 tok/s** | 4.6 秒 |

默认的 `auto` 只拿白给的那一半：底座没量化时把适配器折进去 —— 那儿折叠只要十分之
一秒；底座是 bitsandbytes 的 `nf4`/`int8` 时就让它挂着 —— 那儿折叠意味着把每一层
反量化再量化回去。`on` 在后一种情况下也照折，如果你宁可在加载时付掉那 4.6 秒来换
token 速度；`off` 是以前的行为。这里没有任何一档碰得到 GGUF 路线，那边 llama.cpp
有它自己套用适配器的办法。

**折进去跑出来的答案，和挂着跑不是一字不差的。** 合并改变了算术的结合方式 ——
算一次的 `W + BA`，和每个 token 都算一遍的 `Wx + B(Ax)`，在比特层面并不相等 ——
于是偶尔有一个 token 落点不同，后面半句话就跟着它走了。就在上面那一对上、同一个
种子测得的结果：`bfloat16` 下 128 个 token 里差两处，`nf4` 下更多 —— 那边重新量化
自己还要添一份误差，PEFT 也会为此打印一句警告。两个答案谁也不比谁更好，但它们确实
不是同一个答案；如果你的工作流已经精调到了每一个 token，那就是把这一项留在 `off`
的理由。

#### `adapter` —— 一个下拉框，以及里面装着什么

列表里按顺序装着：

- **`lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA`**，默认项，也是那个不用去动的条目。
  它的意思是*模型列表为你所选的底座点名的那个构建* —— `transformers` 底座用 PEFT
  适配器，GGUF 底座用目录里的 GGUF，8B 节点上则是 8B 适配器。
- **两个 LoRA 发布过的每一种精度。** 每个都有 F16 和 Q8_0；Q8_0 下载量只有一半，
  重写出来一模一样。挑一个本地没有的，它会从它所属的那个仓库下载下来。
- **你 ComfyUI 模型目录里已有的每一个 `.gguf` LoRA**，前缀 `on disk:`，标签里带着
  它的架构，所以 `qwen35` 和 `qwen3vl` 一眼就分得开。你自己转出来的 LoRA 就是这么
  进来的：把文件丢进 `models/LLM`，它就出现了。

以前想换一个量化档位，得手工去改 `models.json` —— 那是高级用户的路子，不是界面。
已经保存的工作流不受影响：下拉框序列化出来的就是它显示的那串字，而原来的默认项
仍然是第一条，一个字符都没变。

关于这一项，有两件事一直成立，因为在所有设置里，它是从网上下载来的工作流唯一能
插上话的那一个：

- **网络路径会被拒绝。** `\\host\share\...` 和 `//host/share/...` 在任何东西去读
  它们之前就被挡下，因为光是去看一眼这种路径，就已经是在对那个被写出来的主机发起
  一次身份验证了。你自己的共享盘照旧可以用盘符或者挂载点访问；而 `models.json` 里
  的路径完全不受这条限制 —— 那个文件是你自己的，也不会跟着工作流走。
- **实际套用了哪个适配器会被记进日志**，每一次运行都记。要是请求的不是配置好的那个
  适配器，控制台会以警告级别把这件事说出来。否则被掉包的 LoRA 是看不见的：节点照跑，
  每个字段照填，只是写出来的是别的东西。

#### 模型自带的代码一律不运行

一个 Transformers 检查点可以自带建模代码，由它 `config.json` 里的 `auto_map` 点名，
而加载这样的模型，会以你这个用户的权限导入并运行那段 Python。节点会拒绝这样的模型，
并说明原因。随包附带的列表里没有一项是这样的：每个 `transformers` 条目都是
Qwen3.6-27B 的某个变体、Qwen3-VL-8B 或 Qwen2.5-Omni-7B，都是 Transformers 原生支持的
架构，而 GGUF 条目根本走不到 Transformers 那边去。

直到 0.30.0，这里都有一个 `trust_remote_code` 开关，允许这样的模型运行。它默认是关着的，
但它是一个小部件，工作流可以把它打开；而模型列表可以通过 ComfyUI 的 API 来扩充，这个
API 不要求登录。在一个能从网络访问到的 ComfyUI 上，这就足以让它下载一个模型，并运行
里面的代码。0.31.0 去掉了这个开关。带着它保存的工作流照样能打开：选项节点会把其余的值
放回原位。

#### `device` —— 以及它为什么改变了 `keep_model_loaded`

这里每一个节点都带着同一句提醒：把 `keep_model_loaded` 关掉，因为重写一结束，那块
卡就要拿去生成视频了。那句提醒之所以存在，只是因为两个模型抢的是同一块设备。
**有两块卡的话，它们就不必抢了。**

可选值是 `auto`、`cpu`，以及 ComfyUI 能看见的每一块 GPU 各一个 `cuda:N`。同一种
写法，三套后端：

| | `auto` | `cuda:1` | `cpu` |
|---|---|---|---|
| llama.cpp 二进制 | 不变 | `--device CUDA1` | `--device none`，0 层 |
| llama-cpp-python | 不变 | `main_gpu=1`，split mode 为 `NONE` | `n_gpu_layers=0` |
| Transformers | `device_map="auto"` | `device_map={"": "cuda:1"}` | `{"": "cpu"}` |

要紧的不是模型被放到了哪儿。**换到另一块卡上之后，ComfyUI 自己的模型不再被第一个
赶出去。** 今天这里的每一套后端都是无条件把它们卸掉的 —— 两边抢同一份显存时这么做
是对的，不抢的时候就是纯粹的浪费：代价是每重写一次，扩散模型就得整个重新加载一遍。
选第二块卡，重写器就在它旁边加载，`keep_model_loaded` 这时才值得打开，而且什么都
不用挪。

有两个细节值得知道。机器上不存在的设备会被**拒绝，而不是悄悄降级** —— 从一台双卡
机器带过来的工作流会把这件事说出来，而不是跑到错的那块卡上、半路把一批任务挤下去。
还有，编号用的是 ComfyUI 自己那一套：用 `--cuda-device 1` 启动时，ComfyUI 看到的
恰好只有一块设备，而它是 `cuda:0` —— 对子进程来说也是如此。

这些值是故意写得朴素的，而不是 `cuda:1 · RTX 4090`：带显卡名字的标签读起来更顺眼，
但换卡那天，它会把每一个保存过的工作流都弄坏。每个槽位里插着什么，工具提示会说。

#### `downloader` —— 内置传输，还是 `huggingface_hub`

默认这一档是自己搬字节：一次一条带 range 的 HTTP 连接，可以续传，什么都不用装。
对于一件没人主动选择过的事情来说，这是正确的底线，而且在快的线路上它挺好用。

换成一个 30 GB 的检查点、而线路给单条流只有 8 MB/s 时，它就不好用了。Comfy-Org
那些仓库是 **Xet 支撑的**，而 `hf_xet` 会用许多条连接同时去取*同一个*文件的各个
分块 —— “命令行五分钟就下完了，节点却下了一个小时”，差别全在这儿。把 `downloader`
设成 `huggingface_hub`，走的就是那条路：

```
python -m pip install huggingface_hub hf_xet
```

装进 ComfyUI 自己的 Python 里，然后重启。这两个包都不是本节点包的依赖，而且
**缺了它们不会有任何东西失败**：节点会把安装命令记进日志、在自己身上注明这件事，
然后用内置传输去下载。这一点很重要，因为这个设置是跟着工作流走的，而在装了它们的
机器上做出来的图，在没装的机器上也应该照样能跑。

**它只管 Hugging Face 仓库** —— 底座模型、GGUF 写作模型、LoRA 适配器，以及一个模型
连同它的投影器。llama.cpp 的二进制来自 GitHub releases，两份写作指南各自只是一个
很小的请求；这两样都不走这个设置。

**下到一半的文件在两边之间不通用。** 内置传输把字节停在文件旁边的 `<name>.part`
里；`huggingface_hub` 把它们停在目标文件夹内的 `.cache/huggingface` 下。下载途中
换一档，那个文件会从头开始，另一边剩下的东西得你手工去删。已经下完整的文件无论
哪一档都不会被取第二遍 —— 本节点包是在调用任何一个后端之前就决定了缺什么，这也
顺带免掉了 `huggingface_hub` 为了接管一个已经下完的 30 GB 文件而重新算一遍哈希。

关于硬盘有一件事值得知道：`hf_xet` 会在 `~/.cache/huggingface/xet`
（`HF_XET_CACHE`）存一份分块缓存，在 Windows 上它落在 `C:` 盘 —— 不管模型本身放在
哪个盘。本节点包在下载前做的剩余空间检查，只看目标位置。

**在会检查 HTTPS 的杀毒软件后面。** 扫描加密连接的杀毒软件以及同类的公司代理，
会用它们自己的根证书给 huggingface.co 重新签名，并把这个根证书装进系统的证书存储 ——
所以浏览器照样能打开这个网站。内置的传输在常用的 certifi 证书包之外也信任这个存储，
所以浏览器能下载的地方它也能下载。`huggingface_hub` 和 `hf_xet` 自带 TLS，不看那里：
在这种过滤器后面它们会停在 `certificate verify failed` 上，节点会把这一点说出来 ——
换回内置的传输，或者把 huggingface.co 加进过滤器的例外列表。

### MiniMax-H3 Prompt Writer (T2VA/I2VA/FL2VA/L2VA) 提示词写作节点

同样那三个输出字段，不用 LoRA，也不用 27B。MiniMax 官方的
[提示词写作指南](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
被放进系统提示词，任何一个会跟随指令的 GGUF 都能照着它写。一个 5.2 GB 的
Qwen3.5-9B 大约二十秒就能把三个字段都填满，镜头术语、说话人标识和
`<d>[English] …</d>` 对白标签都写得对。

![T2VA 写作节点，夹在选项节点和文本查看器之间：一行提示词，一个来自 Ollama 的模型，右边是逐镜头的描述，三个编号镜头、一段声景，以及描述钢琴和大提琴的 non_diegetic_music](docs/node_t2va.png)

*T2VA，16:9，10 秒，跑在本地一个 Qwen3.6-27B 上 —— 11.2 秒。提示词是用俄语递进去的，
描述按 H3 的英文格式回来，切点就在提示词要求的 00:03.000，而
`non_diegetic_music: N/A`，因为提示词说了不要配乐。*

这笔交换值得明说。LoRA **本身就是**那个格式：一个 27B 被一直训练到只用七行系统
提示词就能吐出 H3 的输出。而在这里，格式是由大约 4000 个 token 的指令扛着的，模型
得去跟随它们 —— 所以该预期 LoRA 写出来的文字更密实、格式更靠得住；也该预期这一个
在 LoRA 根本够不着的硬件上照样跑得起来。

输出和重写器节点的完全一致 —— 同样的名字，同样的顺序 —— 所以在保存好的工作流里，
两者可以互换。

在重写器那些输入之外：

- `task` —— **T2VA**、**I2VA**、**FL2VA** 或 **L2VA**。除 T2VA 以外，其余任务还会
  把 H3 要求的那句对齐指令写成答案的头一行，时长已经按两位小数代进去了；留给模型
  的只有最后那个镜头编号。
- `reference_material` —— **这个节点读的是文字，不是像素。** 做 I2VA、FL2VA 和
  L2VA 时，把参考帧上有什么描述出来，手写也行，或者从上游的
  [Reference Caption](#minimax-h3-reference-caption-参考素材描述) 节点接过来。没有
  它，模型会自己编一个和你那张图毫无关系的首帧。

`model` 给出的是模型列表里的 `writers` 一节，外加你 ComfyUI 模型目录里已有的每一个
GGUF，架构是什么都行。这里没有哪样东西非得是 Qwen3.6-27B，也没有哪样东西非装不可：
没装 `llama-cpp-python` 时，节点就去跑 llama.cpp 的官方二进制，和后文那条 GGUF 路线
一模一样。

| 推荐的写作模型 | 下载 | 指南进上下文之后的显存 |
|---|---|---|
| Qwen3.5-4B `Q4_K_M` | 2.6 GB | 约 5 GB —— 8 GB 的卡从这个开始 |
| Qwen3.5-9B `Q4_K_M` | 5.3 GB | 约 8 GB —— 每 GB 换来的文笔最好 |
| Qwen3.5-9B Uncensored (HauhauCS) `Q4_K_M` | 5.2 GB | 约 8 GB —— 原版会拒绝的场景它不拒绝 |
| Gemma 3 12B Instruct `Q4_K_M` | 6.8 GB | 约 10 GB —— 非 Qwen 的一个选择 |
| Mistral Small 3.2 24B Instruct `Q4_K_M` | 13.4 GB | 约 17 GB —— 给 16 GB 及以上的卡 |

`n_ctx` 会自动抬高：基础指南大约要 9200 个 token 的上下文，全参考指南大约 12300，
而默认的 8192 是照着 LoRA 那段很短的系统提示词定的。改成让 llama.cpp 去截断的话，
被砍掉的会是提示词的*前面*那一段 —— 指南，以及输出格式的约定 —— 于是答案会以某种
别的格式回来，而且没有任何东西说得出为什么。

要是某个字段没写回来，节点仍然把拿到的东西全部返回，并在自己身上写明缺的是哪几个
字段。把 temperature 调低，或者往上换一档大小。

#### `system_prompt` —— 把写作节点指向另一个模型

这个节点之所以是给 H3 写提示词的，全靠那份指南：它进系统提示词，而节点里再没有
别的东西知道这个格式。所以把那段文字换掉，就等于把这些节点变成给别的东西写提示词
的节点 —— LTX、Krea、Wan，或者你自己的一套风格。`system_prompt` 中的文字会替换组装好的指南。

那时指南连取都不会去取，这在一台常年离线的机器上是要紧的：不会为了把它忽略掉，先
下载一份 24 KB 的文档。

最短的入口是 [Guide Prompt](#minimax-h3-guide-prompt-any-llm-指南提示词)，它把原版
的系统提示词从一个输出交到你手上。拿走、改掉、再接回来。

**保存系统提示词。** 在 Rewriter Options、API Writer Options 或 Qwen Rewriter Options 中
打开 **System prompts...**。可以新建、导入 UTF-8 编码的 `.md` 或 `.txt` 文件，或者选择任务，
通过 **Copy default guide** 复制并编辑默认指南。点击 **Save and select** 会将文件保存到
`<ComfyUI user>/minimax_h3_rewriter/system_prompts`，并在 `system_prompt_file` 中选中。
迁移工作流时，也要将指南文件复制到目标服务器的库目录。

![System prompts 编辑器，含文件导入、默认指南的任务选择和 Save and select 按钮](docs/system_prompts_editor.png)

写作节点优先使用 `system_prompt` 中的文字。该字段为空时使用所选文件，选择
**Default guide** 时则使用当前任务的默认指南。

有两样东西不动。任务消息永远不会被替换 —— 它带着提示词、画面比例和时长，而任何一份
指南都需要这些。还有，答案照样会按 H3 的那几节切开，所以一份只回一段话的指南会把
`rewritten_prompt` 填满，而把分节的那些输出留空。这件事值得知道，而不是值得躲开。

### MiniMax-H3 Prompt Writer (Ref2VA) 提示词写作节点

全参考模式，出自 MiniMax 的
[全参考指南](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)。
六节而不是三节，每一节各占一个输出：

| 输出 | 内容 |
|---|---|
| `subject_definitions` | 每一个 `<Subject N>` / `<Picture N>` / `<Video N>` / `<Audio N>` 标签各自指代什么 |
| `summary` | `[task type]` 前缀，外加一段话说清各份参考素材之间的关系 |
| `retention_analysis` | 逐个标签：`fully_preserved`、`partially_preserved`、`attribute_transfer`、`weak_reference`、`fully_copy`…… |
| `detailed_description` | 正文，逐个镜头，在标签起作用的地方点名引用它们 |
| `overall_soundscape` | 环境声和画面内的物理发声 |
| `non_diegetic_music` | 只有观众听得见的配乐 |

![Ref2VA 写作节点和一个文本查看器并排，六节全在上面：subject_definitions 把 Subject 1 和 Picture 1 绑定起来，summary 带着它的参考生成任务类型，保留度分析把两个标签都标成 fully_preserved，逐镜头的正文，以及声景和 non_diegetic_music](docs/node_ref2va.png)

*和上面同一个片段，只是往 `reference_assets` 里加了一行 `Picture 1:` —— 17.4 秒。
模型自己挑出了任务类型，把 `<Subject 1>` 绑到那只猫、把 `<Picture 1>` 绑到那幅画的
风格上，保留度分析也是它自己写的。*

`reference_assets` 是**必填的**，留空会被拒绝：全参考模式讲的是一段目标视频如何复用
参考素材，没有素材，它就只是换了个名字的 T2VA。一行一份 ——

```text
Picture 1: young woman, long dark hair, blue cardigan, thin silver necklace
Picture 2: corner cafe interior, brick wall, brass lamps, rain on the window
Audio 1: voice-timbre reference for the woman — low, unhurried, slight rasp
```

—— 然后模型据此把标签绑定好、挑出任务类型，并写出保留度分析。也可以让
[MiniMax-H3 Reference Caption](#minimax-h3-reference-caption-参考素材描述)
直接从素材本身替你写出这几行。

### MiniMax-H3 Universal Writer 通用写作节点

另有独立的 [API Universal Writer](docs/api_writer.md)，可通过已运行的本地 LLM
或兼容的远程 API 编写 H3 提示词，支持图片、视频采样帧、音频和 RefMod。
在 **API providers** 中设置连接和模型，在 **API Writer Options** 中设置生成参数。
密钥在 Windows 或 Linux 服务器上加密保存，不写入工作流。
从 **11 - Write through an API** 模板开始，先配置自己的连接。

![API Universal Writer：任务、画面比例、连接和模型选择，以及参考素材输入和提示词输出](docs/api_universal_writer_node.png)

![API Writer Options：生成限制、响应格式、视频采样、音频采样率和提示词修复设置](docs/api_options_node.png)

一个节点管完一整个镜头：参考素材、它们的顺序，以及重写本身。
[Multi Reference Caption](#minimax-h3-multi-reference-caption-多参考素材描述)
和两个写作节点做的事，它在同一个盒子里全做了，五个任务都覆盖。那三个节点原封不动
——你已经搭好的东西不会因此停摆。

把它们折到一起的理由是**顺序**。在 FL2VA 里，`Picture 1` 和 `Picture 2` 不能互换：
一个开头，另一个收尾。而在此之前，决定谁是谁的只有描述节点碰巧把它们写下来的先后，
而那又取决于每一份素材插在了哪个插槽上。真实存在、而且是承重的，却在屏幕上哪儿都
看不见。

![通用写作节点，左边接着一个 Load Video 节点和一个 Load Image 节点：四行参考输入，每一行带一个复选框，其中三行接着东西；接着是一条三个彩色方块的素材条，ref_0 上面写着 vid 1、ref_1 上面 aud 1、ref_2 上面 pic 1，每个方块带一个 “+ instr” 按钮，底下还有一行说明文字；任务开关上 Ref2VA 亮着；八个按真实比例画出来的画面比例方框，从 48:9 到 9:16，选中的是 16:9；时长滑块停在 15.1；提示词；描述模型和写作模型两个下拉框；底部六个按钮：Model list、Open guide folder、Unload model、Save the last prompt、Edit the last prompt 和 Prompt library。右边是六个字段的答案，围着 <Subject 1>、<Picture 1>、<Video 1> 和 <Audio 1> 写成](docs/node_universal_writer.png)

*四份参考素材接在同一个会自动增长的插槽上 —— 一张当主体用的图片、一张当画面帧用的
图片、一段片段和一段声音 —— 右边的答案正是围着这几个标签写的：`<Subject 1>`、
`<Picture 1>`、`<Video 1>`、`<Audio 1>`。这里的方块排成插槽顺序，只是因为还没人拖过
它们；编号是位置，任何东西一动就重编，而编号下面那个插槽名才是跟着方块走的。*

**一个插槽收图片、片段或者声音**，你填一个，就多冒出一个。这里不存在“插错插槽”，
因为你用了哪个插槽，已经什么都不决定了。

**决定权在输入下面那条素材条上。** 每一份连上来的参考素材都得到一个方块，颜色按它
派什么用场来分，编号和参考素材块里将要给它的编号完全一致。编号底下写的是它从哪个
插槽来的，而这才是方块身上跟着它走的那一部分：编号是位置，只要有东西一动，它就
重新编。

| 方块 | 标签 | 含义 |
|---|---|---|
| `pic` | `<Picture N>` | 一张真的当画面帧用的图片 —— 首帧、末帧、关键帧、构图锚点 |
| `subj` | `<Subject N>` | 可以复用的可见内容 —— 一个人、一只动物、一个地方、一套服装、一种风格 |
| `vid` | `<Video N>` | 一段片段，或者一批你希望被当成一个整体去读的帧 |
| `aud` | `<Audio N>` | 一种音色、音乐、环境声，或者一个音效 |

- **拖动方块**就能挪动它。编号立刻跟上。
- **点它的标签**可以改一张图片派什么用场：`pic` → `subj` → `vid`，然后转回来。
  声音是什么就是什么，所以它不参与轮换。
- **点片段的标签**，会把片段自带的声音也算作一份参考素材。在 `Ref2VA` 上标签条变成
  紫色，写着 `vid+a1`：描述模型会在块里把这个声音写成 `Audio 1`，紧挨在片段前面，
  而 `references` 会让 [Reference Slots](#minimax-h3-reference-slots-参考素材插槽)
  把它交给生成节点。片段的声音先拿到 `<Audio N>` 的编号，独立的声音排在后面，这正是
  `MiniMaxH3ReferenceToVideo` 的编号方式。为此描述模型得能听 —— 听不了的会在加载任何
  东西之前就被点名拒绝 —— 没有音轨的片段会在节点上点名说明，不给编号。
- **点它的编号**可以在不拔线的情况下把这份参考素材关掉 —— 插槽那一行自己的复选框，
  是从另一头做同一件事。
- **点它下面那条带子**，可以就这一份参考素材问一个不同于它角色惯常那个问题的问题。
  `+ instr` 是把你写的这句加到角色的问题上，`= instr` 是拿它替掉那个问题；右键撤回，
  鼠标悬停就能读到它。带着这样一句的方块会围上一圈边框：`+` 是琥珀色，`=` 是橙色，
  想看漏都难。

所以一个插槽照样产出 Ref2VA 允许的那四种标签；而 Multi Reference Caption 用结构
做出来的那个区分 —— 主体不是画面帧 —— 在这里改由方块来做。

**每个方块都会显示插在它上面的是什么** —— 这里如此，Prompt Rewriter Omni 和
Multi Reference Caption 上也一样。图片显示缩略图，片段显示第一帧、角上带一个 ▶，
声音显示波形、一个喇叭图标和一行小字的文件名；编号挪到角上。鼠标悬停时片段会播放，
一批帧会轮流翻过去，提示框里再补上文件、尺寸、帧数和时长。声音在悬停片刻后播放开头
六秒，波形随着播放一格格亮起来；按下方块就会停。直接接在加载器上的方块 ——
Load Image、Load Video、Load Audio 或 VideoHelperSuite 的加载器，中间隔着 reroute
也行 —— 不用运行就能马上显示文件。经过别的节点接进来的方块，显示的是节点上一次运行时
收到的东西，在那之前就是一个普通方块。

![设为 Ref2VA 的通用写作节点的素材条，共十一行参考输入，ref_9 的复选框没勾。方块从左到右：ref_0 上的 pic，一只黑猫，角上是 1，围着琥珀色边框，带子是琥珀色的 “+ instr”；ref_1 上的 pic，一道木栅栏，2，围着橙色边框，带子是橙色的 “= instr”；ref_2 上的 pic，车里的一个男人，3；ref_3 上的 vid，一个在街上走的男人，带播放标记，1；ref_4 和 ref_5 上的 aud，各是一个喇叭和一段波形，1 和 2；ref_6 上的 vid，一个戴恐龙面具的人，带播放标记，2；第二行是 ref_7 上的 pic，一只镶满宝石的乌龟，4；ref_8 上的 aud，一个喇叭、被截断的文件名、一段波形和 3；还有 ref_9 上的 aud，颜色变暗，文件名和波形仍然画着，编号是 “--”。素材条下面是亮着 Ref2VA 的任务开关、选中 16:9 的画面比例，以及 duration 10.0](docs/node_ref_strip.png)

*图片、片段和声音一眼就分得出来，两段声音之间也一样：ref_8 和 ref_9 直接接在加载器上，
显示的是文件名；ref_4 和 ref_5 是经过别的节点进来的，显示的是上一次运行时的波形。ref_9
被关掉了 —— 颜色变暗、编号是 `--`，但照样认得出来。围着边框的，就是那两个带问题的方块。*

**任务开关和画面比例选择器是同一个想法**：让你选的是那张图，而不是下拉框里的一行
字。而且任务开关会去读素材条：素材条供不上某个任务所需要的东西时，那个任务就置灰
—— 所以什么都没接时只有 `T2VA` 亮着，一张图片点亮 `I2VA` 和 `L2VA`，两张点亮
`FL2VA`。三张图片会把这三个一起灰掉，因为对 `I2VA` 来说，三张和一张都没有同样不
可能 —— 这就是节点本来会在运行时拒绝的那个数量，只是从运行挪到了看得见的地方，而
灰掉的按钮把同一句话挂在工具提示里。数的是标签，不是插槽：把一张图片改成主体，它
原本挡住的那个任务就亮了。已经选中、而如今又不再可能的任务会变红，而不是过一会儿
悄悄失败。

选择器里的每一个方框，都是在同一份面积预算内按自己的真实比例画出来的，所以越往宽
的那一头，读出来的就是还剩多少高度：`21:9`、`32:9` 和 `48:9` 画出来大约只有 15、
10 和 6 像素高。节点窄到装不下整排时，它会把这一排折到第二行，而不是把最后几个方框
切掉，并且正好长高那么多。

**`duration` 是一个滑块，以十分之一秒为单位**，能拉到多远由你自己定 —— 右键点节点，
选 `duration`。见[时长小部件](#时长小部件)。

**两个模型小部件，因为这里有两份活。** `caption_model` 负责读参考素材，
`writer_model` 照着 MiniMax 的指南写提示词；`clip` 在这里的用法和在描述节点上完全
一样，接上了就用它来读参考素材。做 T2VA 时，描述模型根本不会被碰。

**每个任务对素材条的要求：**

| 任务 | 图片 | 素材条上的其他一切 |
|---|---|---|
| T2VA | 一张都不要 | 完全忽略，而且节点会把这件事说出来，而不是去描述它们 |
| I2VA | 恰好一张 —— 首帧 | 作为参考材料递进去 |
| L2VA | 恰好一张 —— 末帧 | 作为参考材料递进去 |
| FL2VA | 恰好两张 —— 先首帧，后末帧 | 作为参考材料递进去 |
| Ref2VA | 多少张都行 | 至少得有一份某种参考素材 |

对不上的时候，在下载或者加载任何东西之前就会被拒绝，而且消息点名的是素材条而不是
插槽，因为该动手的地方在素材条上：一张挡路的图片，是被关掉或者换一个标签，而不是
被拔掉。

**输出是两个写作节点输出的并集** —— T2VA 那三个字段、Ref2VA 那六个字段，以及参考
素材块本身。某个任务不写的字段就留空，因为一个节点的输出不能随着它某个小部件的
取值而改变。最后一个输出 `references` 不是文本：它是素材条编好号的东西，交给
[Reference Slots](#minimax-h3-reference-slots-参考素材插槽)，按同样的顺序接进
`MiniMaxH3ReferenceToVideo`。

> **和 Multi Reference Caption 有一处不同。** 那个节点是按指南自己的顺序写这个块的
> —— 主体、图片、片段、声音 —— 不管线是怎么接的。这一个按素材条的顺序写，因为
> 素材条正是它全部的用意所在。没被动过的素材条，就是插槽顺序。

> **如果界面脚本没能加载**，素材条、任务开关和比例选择器会回退成它们底下那些普通
> 小部件 —— 一个文本框和两个下拉框 —— 节点照样跑得起来。这三个控件是 HTML 的，
> 在 ComfyUI 的两种渲染器里都能用。输入行上的复选框是画在画布上的，而
> *Modern Node Design (Nodes 2.0)* 不跑画布绘制；在那种模式下，关掉一份参考素材
> 靠的是点每个方块上的编号。Multi Reference Caption 画的是同样这些方块，如今已经
> 完全不需要复选框了。
>
> 这三个画出来的控件同样被挡在右侧的 **Parameters** 面板之外：那个面板没有办法画
> 这样的控件，要试也得先把它从节点上拿下来。它们的家就在节点上。这个节点的其他
> 一切 —— 提示词、模型、时长滑块 —— 照常待在面板里。

> **这个节点需要较新的 ComfyUI**，理由和 Multi Reference Caption 一样：它那些会
> 自动增长的输入是 v3 节点 API 里的 `io.Autogrow`。装在旧版本上，这两个节点会直接
> 不出现，节点包的其余部分照常注册。

### MiniMax-H3 Image Prompt Writer

将简短请求扩写为 [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
所需的详细提示词。没有启用的图片时为文生图，启用 1 至 16 张图片时为图像编辑。
第一张启用的图片作为底图。拖动参考卡片可以调整顺序，用输入复选框切换启用状态，
通过 `+ instr` 添加单图指令。IMAGE 批次使用第一帧。

![Image Prompt Writer 的 Local GGUF 模式，使用底图和编辑指令](docs/image_prompt_writer_node.png)

- **Local CLIP**：通过 Load CLIP 加载 `qwen3.5_9b_w4a8_gptq.safetensors`，
  类型选择 `minimax`，连接到 `upsampler`。这是
  [ComfyUI H3 Image 模板](https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image)使用的扩写模型。
- **MiniMax Context IR**：在 ComfyUI 服务端环境设置 `MINIMAX_API_KEY`，
  或用 **API providers** 选择保存的 MiniMax 配置。配置需要
  `https://api.minimax.io/v1`（国际站）或 `https://api.minimax.cn/v1`（中国站）、
  凭证，以及 **Allowed models** 中的 `MiniMax-H3-Image`。
  [扩写 API](https://platform.minimax.io/docs/api-reference/image-generation-h3-context-ir)
  接收指令和启用的图片，MiniMax 对扩写计费。
- **API Writer** 使用所选本地或远程 Chat Completions 服务中的模型。和
  **API Universal Writer** 一样选择配置及模型。文生图扩写可使用文本模型。
  图像编辑扩写需要支持图像的模型和可接收交错图文输入的服务，并在
  **Allowed models** 中启用 `image`。启用的图片按参考卡片顺序发送。
  [MiniMax 公开的提示词扩写方法](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE)
- **Local GGUF** 通过节点包的本地 llama.cpp 运行所选模型。文生图从
  **Writers** 选择模型。图像编辑从 **Captioners** 选择支持图像的 GGUF
  及其匹配的投影器。**Model list** 会显示 `models/LLM` 中的模型。
  将 **Rewriter Options** 接到 `local_options` 可设置上下文、采样和显卡。
  图像编辑使用带 `llama-server` 的 llama.cpp 构建，并按参考卡片顺序传递图片。

在 API Writer 模式下，可连接 **API Writer Options** 以设置 token 上限、
温度、响应格式和超时时间。使用 `text` 格式时，节点在提示词中要求模型返回 JSON。
所选模型及服务支持时，可使用 `json_object` 或 `json_schema`。
`summary` 显示本次请求使用的配置及模型。

将 `prompt`、`width` 和 `height` 连到 ComfyUI 模板中的 **MiniMax H3 Image Conditioning**。
`aspect_ratio=auto` 使用扩写模型建议的比例。也可以固定比例并选择 `area`，
或保留 auto 并输入边长为 32 倍数的 `size`。编辑时 **Reference Slots** 可将
`references` 中的第 1 至 9 张图连到 Conditioning。使用 10 至 16 张图时，
按参考卡片顺序连接原图。**bypass** 传递原始文本，**repeat_last** 复用会话或提示词库中的结果。

### P-H3 Prompt Writer (图像)

节点界面和图像编辑结果。完整用法见[英文版说明](README.md#p-h3-prompt-writer-images)。

在 Text2Image 中启用 `use_ref_description`，可根据第一个启用的参考图和用户请求编写独立提示词。视频、图像批次或视觉 RefMod 使用选中帧中的第一帧；可通过 `frame` 选择单帧。生成器只接收文字，`references` 输出仍为空。

![启用 use_ref_description 的 Text2Image 模式 P-H3 Prompt Writer，显示来源参考图预览和帧选择](docs/p_h3_prompt_writer.png)

![P-H3 帧选择窗口，带大图预览、编号缩略图和已选参考帧](docs/p_h3_frames_editor.png)

![Save Image 中的 P-H3 图像编辑结果：一只戴着红色针织围巾的乌龟](docs/p_h3_result.png)

### MiniMax-H3 Reference Caption 参考素材描述

写作节点读的是文字，不是像素。文字就是从这儿来的：接上一张图片、一段声音或者一段
视频，一个小的多模态模型就会把它描述成 `reference_assets` 里带标签的一行。

在一个 3.4 GB 的 Qwen2.5-Omni-3B 上实测：**一帧画面 3 秒，一段声音 2 秒，一段视频
5 秒。** 它跑的是 `llama-mtmd-cli`，这个程序和 `llama-completion` 装在同一个压缩包
里，所以重写本来就跑在二进制上的机器，不必为它再下载什么运行时。

**重写没跑在二进制上的机器，这笔钱要在这儿付。** 只要 wheel 能被导入，
`gguf_runtime = auto` 挑的就是 `llama-cpp-python` —— 而较新的 ComfyUI portable 自带
了一个 —— 至于 safetensors 底座，它根本碰不到 llama.cpp；这两条路谁都没取过那个
压缩包，所以第一次生成描述，正是它到来的时刻：34 MB，或者在 `llama_backend` 解析到
CUDA 时是 511 MB。wheel 也省不掉这一笔，不管它是怎么编译出来的：多模态输入要走
`llama-mtmd-cli`，那是一个程序，而 wheel 是一组共享库，里面一个可执行文件都没有。
所以 `gguf_runtime` 是一个属于写作模型的设置 —— 不管它写着什么，描述节点跑的都是
二进制。

**把它们串起来靠的是接线**：把 `reference_assets` 接进下一个节点的 `previous`。每个
标签都是*在自己那一类里面*编号的，这是指南自己的规则（“`<Video N>` 和 `<Audio N>`
各自独立编号”），所以四份素材出来是 `Picture 1`、`Picture 2`、`Video 1`、`Audio 1`，
而不是 1 到 4。

![两个串起来的参考素材描述节点：一个 Load Image 喂给第一个，角色是 Picture；一个 Load Audio 喂给第二个，角色是 Audio；reference_assets 从第一个节点接进第二个节点的 previous 输入，右边的查看器里两行都在，Picture 1 和 Audio 1](docs/node_ref_caption.png)

*一张图片和一段声音，过的是 Qwen2.5-Omni-7B —— 4.4 秒和 3.7 秒。第二个节点从
`previous` 收到了第一个节点的块并追加在后面，而 `Audio` 这个角色描述的是嗓音本身，
没有去转写：“a male speaker with a calm and measured delivery, speaking in
Russian”。*

`role` 同时决定标签和要问的问题，而这些问题是故意问得不一样的：

| `role` | 要模型回答什么 |
|---|---|
| `Subject` | 跨镜头必须保持一致的那些特征 —— 体型、头发、衣着和它们的颜色、随身携带的物件 |
| `Picture` | 把这一帧当作一个镜头来看 —— 风格、景别、机位角度、位置关系、环境、光线 |
| `Video` | 主体、按先后顺序的动作、镜头运动、剪切和节奏 |
| `Audio` | **声音本身，不是词句** —— 音色、听上去的年龄和性别、语气和语速、乐器编配、速度、环境声 |

最后那一行才是要紧的。指南里的 `<Audio N>` 通常是一份*音色*参考，而一份逐字稿恰好
把需要的那一部分丢掉了，所以指令里干脆写明了“不要转写”。要是你还想要逐字的台词去
填一个 `<d>` 块，那是 ASR 的活 —— 在旁边跑一个 Whisper，把那一行粘进来。

其他输入：

- `description` —— 自己打字，**一个模型都不会跑**。要加一份六个词就能说清的素材，
  这是最快的办法。
- `instruction` —— 把这个角色的问题整个替掉。
- `max_frames` —— 从一批 IMAGE **或者一段 VIDEO** 里取多少帧，均匀分布。全都要的话，
  上下文和挂钟时间都会溢出：25 fps 下的两秒就是 56 张图片要过视觉塔，三十秒是 750
  张。正是这一项让描述一段片段的开销和它有多长无关。
- `context_size` —— `0` 表示按参考素材和显卡来定上下文大小，这和按模型自己的来定
  不是一回事：llama.cpp 是在读到一个像素之前就把整个 KV 缓存预留下来的，而一个按
  262144 个 token 训练出来的模型会为此要走几十个 GB —— Qwen3-VL-8B 要 36 GB，而
  那块卡本来三秒钟就能把那张图描述完。一帧 1024×1024 已经是二十来个媒体分块，所以
  答案的一半在参考素材的数量上，另一半在设备的显存上。想自己说了算就填一个数；填得
  太小会让这次运行直接失败，而不是把内容截掉。

**会思考的模型不会把思考写进这个块里。** 描述模型写在 `<think>` 和 `</think>` 之间
的东西，在这段描述变成 `reference_assets` 的一行之前就被剪掉了 —— GGUF 那条路和
`CLIPLoader` 那条路都一样 —— 否则每一条描述开头都会顶着一个空的 `<think> </think>`，
而一个真的会推理的模型，会把它整段的斟酌一路送进写作节点的提示词里。

剪掉它只是一半，另一半没法做成一个开关。写作节点是自己渲染聊天模板的，会传
`enable_thinking=False`；描述节点做不到，因为模板是 `llama-mtmd-cli` 自己套用的 ——
正是它把媒体 token 放到该在的位置上 —— 而它没有为思考通道留任何开关。于是那段推理
仍然要算在 `--predict` 的额度里，所以每一个描述问题的末尾都有一句，要求只给描述
本身。要是你手上某个模型还是不肯听，`instruction` 就是你用自己的话把这件事说清楚的
地方。

> **不是每一个多模态 GGUF 在这儿都能用**，而用不了的那些会大声失败。llama.cpp 的
> `mtmd` 必须看得懂投影器的格式：Gemma 4 的投影器在 `b10310` 上会让进程直接中止
> —— 用 Google 自己的文件和用 unsloth 的一样，CUDA 构建和 CPU 构建也一样 —— 而
> `llama-completion` 把同一个模型当纯文本跑却毫无问题。所以 `captioners` 列表里
> 只放真正跑通过的组合：
>
> | 描述模型 | 下载 | 支持的模态 |
> |---|---|---|
> | Qwen2.5-Omni-3B `Q4_K_M` + mmproj | 3.4 GB | 图片、声音、视频 |
> | Qwen2.5-Omni-7B `Q4_K_M` + mmproj | 5.8 GB | 图片、声音、视频 |
>
> 一个模型和它的 `mmproj` 一起待在 `models/LLM` 下的同一个文件夹里，就会被自动
> 列出来，所以想试别的，什么都不用改。

### MiniMax-H3 Multi Reference Caption 多参考素材描述

同一份活，一整个镜头，装在一个盒子里。串成链是准确的，但它会长大：五份参考素材就是
五个节点、五根线，以及五次把其中某一个的角色留错的机会。这个节点把那条链折进一个
盒子，并且把角色彻底从你手里拿走 —— **一份素材插进哪个分组，那就是它的标签。**

这是把指南的那套词汇做进了结构里。Ref2VA 恰好定义四种参考标签，并且禁止再发明新的，
所以四组输入就把这个格式完整覆盖了；而把一张图片当成声音去描述，从此不再是*做得到
但不推荐*，而是根本做不到。

| 分组 | 标签 | 什么该放这儿 |
|---|---|---|
| `subjects` | `<Subject N>` | 可以复用的可见内容 —— 一个人、一只动物、一个地方、一套服装、一种风格 |
| `pictures` | `<Picture N>` | 一张真的当画面帧用的图片：首帧、末帧、关键帧、构图锚点、分镜稿 |
| `videos` | `<Video N>` | 整段视频层面的关系 —— 剪辑来源、接续点、镜头运动和节奏 |
| `audios` | `<Audio N>` | 一种音色、音乐、环境声，或者一个音效 |

插槽随着你往里填而增长，永远多留一个空位，所以这个镜头需要节点有多高，它就有多高，
不会更高。

![多参考素材描述节点和一个文本查看器并排：七个参考插槽，每一行各带一个复选框，接着东西的 video_0 那一行没有勾上，而 subject_0 和 audio_0 勾着；查看器里的块恰好只有两行，Subject 1 描述的是一只壳上带宝石般石头的乌龟，Audio 1 描述的是鸟鸣和水花声之上的一个男声](docs/node_multi_ref_caption.png)

*两份素材过 Qwen2.5-Omni-3B —— 6.0 秒。那段视频接着但被关掉了，所以块里出来的是
`Subject 1` 和 `Audio 1`，一行 `Video` 都没有：那份参考素材还留在图里，只是这一次
运行不花它一分钱。*

**输入下面那条素材条上，每一份连上来的参考素材各占一个彩色方块**，顺序就是这个块
将被写出来的顺序，方块上写着它将被叫作什么。点一下方块，就能在什么都不拔的情况下
把这份参考素材关掉：生成一次描述要付一次模型加载，外加几秒到几分钟，于是“除了这一
个，其他都要”成了再平常不过的诉求，而为此把线拽下来，等于把你本来想留着的接线扔掉。
这个状态跟着工作流一起保存，也像任何别的值一样能穿过 API。

和[通用写作节点](#minimax-h3-universal-writer-通用写作节点)上的不同，这里的方块不能
拖动，标签也不轮换：这个节点按指南自己的顺序写这个块，而一份素材叫什么，由它插进
哪个分组来定。

**方块下面那条带子，是就这一份参考素材问点别的。** 它被问的还是本角色惯常那个问题
时，带子是暗的；一旦不是，它就变成琥珀色，整个方块也围上一圈同色的边框。点一下写下问题，右键撤回，鼠标悬停就能读到。
它做成每份素材一条、而不是每个节点一条，是有意的 —— 一个节点同时描述一张图片、
一段片段和一段声音，本来就不存在一个同时适合三者的问题。

那个小窗口里的复选框决定你写的东西属于两者中的哪一种，带子随后把它说出来：
`+ instr` 表示这一句是**加在**角色的问题上的，`= instr`（带子是橙色的）表示它是
**替掉**那个问题去问的。两种都少不了。

| 你写的东西 | 该用哪一种 | 换成另一种为什么不行 |
|---|---|---|
| `Do not mention the window.` | 加上去 | 一条规则不是一个问题。单拿它去问，就没有任何东西在等着被描述，于是模型回答的是这条规则：`Subject 1: No`。 |
| `Always answer "blah blah blah".` | 替掉 | 把它留在角色的问题后面，两者是矛盾的，而一个小模型摆平矛盾的办法，就是照样去描述。 |

用*替掉*这一种时，你的文字连 `length` 一起接管了：谁写的问题，谁就拥有答案的形状，
所以长度要紧的话，就自己把它说明白。用加上去那一种时，角色的问题和长度预设都留在
原处。

**`videos` 插槽既收 `VIDEO`，也收一批 `IMAGE`**，你的加载器给出哪一种都行 ——
VideoHelperSuite 的 `Load Video (Upload)` 可以直接接上。两种情况下帧都是均匀取样、
取到 `max_frames` 为止，所以一段片段的开销始终和它有多长无关。

**这个块是按指南的顺序写的** —— 主体、图片、片段、声音 —— 而不是按接线的顺序；
每个标签在自己那一类里编号，并且接着 `previous` 上送来的东西往下编。所以这个节点
照样能站在链条中间，两头都接单个的描述节点。

`model`、`length`、`seed`、`max_frames`、`context_size` 和 `bypass` 由节点里所有素材
共用；问题不共用，它住在每个方块下面那条带子上。`role` 和 `description` 是有意去掉
的：分组就是角色，而一段你自己写出来的描述，一次只属于一份素材。想手写描述的素材，
还是交给 [Reference Caption](#minimax-h3-reference-caption-参考素材描述)。

> **这个节点需要较新的 ComfyUI。** 它那些会自动增长的输入是 v3 节点 API 里的
> `io.Autogrow`。装在旧版本上，它是唯一一个会不出现的节点 —— 节点包的其余部分
> 和以前完全一样地注册。

### 用 ComfyUI 已经加载好的模型来生成描述

两个描述节点和通用写作节点都有一个 `clip` 输入。从 `CLIPLoader` 接一个多模态文本
编码器过来，每一份素材就改由*那个*模型来描述，而不是由 `model` 里的 GGUF ——
后者于是就一直待在 ComfyUI 的分配器把它放下的地方。

这正是它的全部意义。GGUF 那条路跑的是 `llama-mtmd-cli`，一份素材一个进程，所以一个
带五份参考素材的镜头，要把权重从硬盘上读五遍。已经加载好的编码器只被读**一次**，
而你打磨措辞时的那些重跑，一分钱都不花。这里实测：三份素材 —— 一张图片、一段片段
和它的声轨 —— 过 Gemma-4 12B 一共 **24 秒**，日志里三份素材总共只有一次
`Requested to load`。

不过别把这读成“更快”。同样这三份素材走 GGUF 那条路、过 3.4 GB 的
Qwen2.5-Omni-3B，连加载一起是 **16 秒**：一个小模型加载得够快，快到三次重新加载
加起来还不如跑一个 12B 贵。这条路买到的，是*更大*的模型而不必为重新加载付钱，以及
完全跳过加载的重跑 —— 而且每多一份素材、每多跑一遍，差距就拉开一点。

哪些编码器合用，各自又收得下什么：

| 编码器 | 看得见 | 听得见 |
|---|---|---|
| Qwen3-VL 4B / 8B | 能 —— 一批帧会被逐帧读进去 | 不能 |
| Gemma-4 12B | 能 | **能** —— 那个“统一”的构建，声音是直接投影进去的 |
| Gemma-4 E2B / E4B | 能 | 零件是齐的，但见下文 |
| Gemma-4 31B | 能 | 不能 |

Gemma-4 是从检查点本身认出来的，所以 `CLIPLoader` 上那个 `type` 小部件对它无关紧要。
接一个只能看的编码器、同时又接上了声音，节点会在任何东西跑起来之前就拒绝，和它拒绝
一个没有音频塔的投影器是一个道理。

> **关于 Gemma-4 E2B/E4B 有两件事要知道**，两件都是在 ComfyUI 0.30 上用一个 fp8 的
> E4B 观察到的，也都用 ComfyUI 自己的 `Generate Text` 节点在同一个文件上复现过 ——
> 所以这两件都不是本节点包能修的：
>
> - **声音送不到。** 描述回来时说没有给它片段。节点在看到这种形状的编码器时是打
>   一句警告，而不是直接拒绝，因为换一个检查点也许就正常。Gemma-4 12B 收声音是收
>   得对的。
> - **它会把推理说出声。** 提示词已经把思考通道预置成空的，而这个构建照样往里填。
>   描述节点会把闭合标签之前的一切剪掉，所以到达 `reference_assets` 的只有答案本身
>   —— 但要是你自己去接 `Generate Text`，剪掉它就是你的事了。

这条路上有两个小部件是不起作用的，工具提示里也这么写着：`model`，它根本不会被查看；
以及 `context_size`，那是 llama.cpp 的 KV 缓存旋钮。`max_frames` 仍然有效 —— 帧是
在这里、在编码器看到它们之前就被抽稀的，而不是留给编码器自己那套 1 fps 的降采样。

GGUF 那条路哪儿也不去：它够得着那些 ComfyUI 没有编码器的模型，而且不需要事先加载
任何东西。`clip` 不接，这两个节点就什么都不变。

### 描述模型只加载一次，不是每份素材加载一次

一条六份参考素材的素材条，从前就是六次模型加载。`llama-mtmd-cli` 生成一段描述就
退出，这对单独一张图片是对的，对一条素材条则是浪费：每一份素材都要把模型和它的
投影器从硬盘上重读一遍，然后才轮到去看一个像素。

现在整个循环由一个 `llama-server` 伺候，它在第一份参考素材之前启动，在最后一份之后
停下。模型、投影器、采样、种子和那一轮系统消息两边都一样，变的只有加载。文件缓存
是热的、8B 的描述模型、三份参考素材：以前 8.7 秒，现在 3.6 秒；而且你每多加一份
素材、描述模型每多重一个 GB，这个差距就更大。

两条路各自都能精确重现 —— 同一张图、同一个种子，每次给出同一段描述。而在两条路
之间，一段描述仍然可能差一个词，*a blue circle centered on* 对上
*a blue circle is centered on*：两个 token 的分数几乎打平，而两套运行时的批处理把
这个平局判给了不同的一方。这是浮点算术，不是问题被问得不一样了；也正因如此，那一轮
系统消息是明写出来的，而不是省掉：省掉之后，差别就不是一个词，而是一段不同的描述了。

这件事不用你去打开，也不会为它下载任何东西：`llama-server` 和 `llama-mtmd-cli` 装在
同一个发布压缩包里，所以它是在已经找到的那个描述程序旁边被找出来的 —— 而且只在它
旁边找，因为来自另一个构建的 server 可能把模型和一个更旧的 mtmd 配到一起，而那个
mtmd 读不了这一个刚刚接受的投影器。

它是一项优化，从来不是必要条件。一个不带它的构建、一个绑不上的端口、一个始终不报
健康的服务：每一种都会被写进日志，而这次运行会退回到一次一个进程地做下去，慢一些，
其余一模一样。只有一份参考素材时根本不会起 server，因为没有第二次加载可省。
`MINIMAX_H3_MTMD_SERVER` 收 `never`（保持旧行为）和 `always`（哪怕只有一份素材也
用 server）。

### MiniMax-H3 Guide Prompt (any LLM) 指南提示词

把基于指南的 `system_prompt` 和 `user_prompt` 拼出来，以字符串返回，什么都不跑。
不花显存，也不花时间。要是你不想在这儿跑模型，就把它们接进你本来就在用的那个 LLM
节点 —— 本地的、API 的、远程的都行。五个任务它全覆盖，Ref2VA 也在内。

第三个输出 `prompt` 是这两段合成的一个字符串，因为大多数 LLM 节点只收一个 ——
**包括 ComfyUI 自己的 `Generate Text`**，它从 0.30 起就能在 ComfyUI 自己的进程里，
用 `CLIPLoader` 加载好的模型跑一个语言模型。如果你本来就为某个图像模型留着一个
Qwen3-VL 或者 Gemma-4 文本编码器，那这就是拿到本节点包这份输出、而且一个 GGUF 都
不用下载的最短路径：

![Load CLIP 喂给 Generate Text，指南提示词节点的第三个输出接进它的 prompt 输入，右边的文本查看器里是写好的重写结果：subject_definitions 把 Subject 1 和 Picture 1 绑在一起，summary 带着它的参考生成任务类型，保留度分析，带时间码的逐镜头正文，以及声景和 non_diegetic_music](docs/node_guide_prompt.png)

*Ref2VA 过一个 4B 的 Qwen3.5 —— 37.7 秒，六节全都回来了。节点自己那行状态里的数字
才是值得知道的：一段 25042 个字符的系统提示词，在模型写下一个字之前就已经是
**大约 10240 个 token 的上下文**，而正是它决定了某个编码器到底吃不吃得下这件事。*

`Generate Text` 上有三个设置决定它能不能成：

- **`max_length`** —— 它默认的 512 是*输出*的额度，而 Ref2VA 那六节根本装不进去。
  **2048** 上下才对，本节点包自己的写作节点用的就是这个数（选项节点里的
  `max_new_tokens`）。在一个 4B 的 Qwen3-VL 上实测：一次 Ref2VA 重写**连模型加载
  一起 53 秒**完整回来，在大约 580 个 token 处自己停住 —— 离 2048 还很宽裕，但比
  512 多出不少，默认值会把它从某一节中间切断。
- **`thinking`** —— 关掉。指南要的就是那几个字段，别的都不要；推理会把上面那份额度
  花在你事后还得剪掉的散文上。
- **`use_default_template`** —— 就让它开着，`format` 也留在 `plain`。

`format` 是唯一一个值得知道的旋钮。在 `plain` 下，两段提示词之间空一行连起来，再由
LLM 节点用模型自己的聊天模板包一层 —— 于是整份指南落进了*用户*那一轮里。在
`chatml` 下，各轮消息改由这里写出来：只要文本以 `<|im_start|>` 开头，Qwen 的文本
编码器就会跳过它自己的模板，于是指南是作为一条真正的系统消息抵达的。那条分支同时也
跳过了 Qwen 对思考的抑制，所以那个空的 `<think>` 块会替你写好。它按构造就是照着
Qwen 来的 —— 在 Gemma 或者别的什么上，就留在 `plain`。

### MiniMax-H3 Prompt Check 提示词检查

这里其他每一个节点都是先写出一段提示词，再检查自己写出来的东西。这一个只检查，而且
文本是从插槽上收的 —— 于是一段从别处来的提示词，会被本节点包读自己重写结果时用的
那套规则原样读一遍，并切成同样那些字段。任何能产出 MiniMax-H3 那种文字的东西都能
喂给它：另一个节点包的节点、加载进图里的一个文本文件、网上找来的一段提示词，或者你
直接敲进小部件里的东西。

不加载任何模型，也不生成任何东西。跑一次只花几毫秒。

![MiniMax-H3 提示词检查节点：左边从上到下是一个 options 插槽和三个参考插槽 ref_0、ref_1、ref_2，其中两个接着东西；右边九个输出 —— 先是 prompt，然后是 integrated_multimodal_description、subject_definitions、summary、retention_analysis、detailed_description、overall_soundscape 和 non_diegetic_music，findings 排在最后。它们下面是一个高高的空提示词框、一个写着 Ref2VA 的任务下拉框、9.9 的时长，以及唯一一个按钮：Save the last prompt](docs/node_prompt_check.png)

*这些输出就是通用写作节点的输出，只去掉了描述它自己那份描述工作的那两个 —— 所以
写作节点站过的位置能放下一个检查节点，检查节点站过的位置也能放下一个写作节点。时长
是 9.9 而不是一个整数，因为它是这段提示词当初为之而写的那个数，不是你会向写作节点
要的那个数。而且按钮只有一个，不是两个：这个节点不跑模型，所以没有什么可重复的，
也没有什么可指向提示词库的 —— 但它读到的东西，它留得住。*

| 输入 | 用来做什么 |
|---|---|
| `prompt` | 要读的那段文字。它不会去动第一个输出，所以这个节点可以坐在图的中间，而不改变最终抵达生成器的东西。 |
| `task` | 这段提示词当初是为哪个任务写的。它决定答案该有哪些字段、每一类参考素材最多可以引用几份，以及顶上该不该有一行对齐语句 —— 所以填错了会让读出来的结果是错的，而不是没有。 |
| `duration` | 目标视频有多长。切点时间是拿它来对的，所以要按这段提示词当初为多长而写来填，而不是按你会向写作节点要的那个数。 |
| `references` | 可选。只读它们的种类和数量 —— 什么都不会被解码，也不会有描述模型跑起来。接上它们之后，读取还会把文字里引用的东西和这里实际有的东西对一遍；什么都不接，这两条规则就跳过，其余照旧。 |
| `options` | 可选，而且只从它里面读 `self_check`：屏幕上要宣告多少。 |

输出是那段提示词、任何任务都可能填的那七个字段，以及 `findings` —— 就是节点写在
自己底下的那个块，`!` 是警告，`-` 是提示，没什么可说时它是空的。这个输出永远装着
找到的全部内容：`self_check` 管的是节点在运行时往外说多少，那是一个关于噪音的问题，
而把这个输出接出去，本身就是在问。

**Save the last prompt** 按钮它也有，所以一段从别处来、读下来干干净净的提示词，可以
带着名字、说明和分组进[提示词库](#提示词库)，和本节点包自己写出来的那些一模一样。

有一种读法值得事先料到。大多数手写给 H3 的提示词都是散文：没有字段标签，没有镜头表，
没有标记。按 `T2VA` 去读，这样一段文字会被报成缺字段、缺 `[Shot 1]`，而这是一次
真实的读取，不是毛病 —— 那样的提示词是*给重写器的指令*，不是一份写完的 H3 答案。
把它喂给某个写作节点，再去检查回来的东西。

### MiniMax-H3 Prompt Reducer 提示词精简器

这里其他每一个节点都在展开。这一个在收缩：它把一段写好的 H3 提示词，还原成当初本
可以由之写出它来的那一行短句。

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a low-angle medium
shot frames a sleek black cat walking steadily along the top of a weathered wooden
fence in a quiet suburban yard at dusk. The camera tracks right with small amplitude
at slow speed, following the feline as its soft fur catches the fading golden light...

overall_soundscape: A gentle evening breeze rustles through nearby grass and leaves...
non_diegetic_music: A sparse piano melody at a slow tempo...
```

回来的是

```
A black cat walks along a wooden fence in a yard at dusk.
```

它有三个用处。改掉一段你喜欢的提示词里的一个词，而不必把另外四百个词重写一遍 ——
精简、改那一行、再喂回给写作节点。把一段为 H3 写的提示词送给一个只要短句的生成器：
H3 的提示词是 H3 形状的，而 Wan、混元和可灵不是。以及给你保存下来的提示词配上一行
读得懂的话，让卡片上显示的是一个名字，而不是一整段。

**这件事有一半根本不是模型的活，而这正是它管用的原因。** 一段 H3 提示词不是散文
—— 它是一个已知的形状：有名字的字段、顶上那句固定的对齐语句、正文里一路的
`[Shot 2] At 0:03` 标记、凡是引用参考素材的地方就有的 `<Picture 1>` 标签、用 `<d>`
围起来的对白。这些全是脚手架，全都能按规则认出来，也全都在向模型开口之前就被拆掉
了。递到模型面前的，是一小段普通描述加一句很短的指令 —— 这就是为什么一个 4B 也能
把它做好。换成让它“把这份文档反过来写”，它就做不好了。

参考素材的绑定关系和其余脚手架一起被拆掉，这是有意的：在 Ref2VA 里，
`subject_definitions` 和 `retention_analysis` 描述的是*那些素材*，而一段保留了它们
的短提示词，描述的会是下一次运行根本不会有的图片。

| 输入 | 用来做什么 |
|---|---|
| `prompt` | 要缩短的那段写好的提示词。五个任务的都行，而且你不必说是哪一个：文本会按两族格式用过的每一个字段名去切，而一个字段名都没有的文本，会被整段当作描述来读。 |
| `model` | 任何一个会跟随指令的 GGUF，来自写作节点用的同一份列表 —— `on disk:` 和 `ollama:` 条目也在内 —— 节点下面的 **Model list** 按钮就是用来编辑它的。这件事对模型的要求比写作低得多，所以列表里最小的那一条在这儿是个合理的选择，哪怕它在那边不是。 |
| `detail` | 回来多少东西。`idea` 是光秃秃的一行，至多十个词；`sentence` 允许带上地点和一天中的时辰；`paragraph` 给每一件发生的事各留一句话 —— 一段有好几个镜头的提示词，想让顺序活下来，就需要这一档。 |
| `subjects` | 人被点名到多具体：`as written`、`age and gender`，或者 `impersonal`。 |
| `keep_camera` | 保留景别、机位角度和镜头运动。默认关闭 —— 镜头通常是写作模型的发明，而不是你的，把它留在外面，下一次重写就能重新选一遍。 |
| `keep_audio` | 把声景和音乐折成结尾的一句话。默认关闭，而关闭意味着那几节声音根本到不了模型面前：它们是被解析器丢掉的，不是被指令丢掉的。 |
| `keep_style` | 保留提示词开头写明的那个媒介和观感 —— 实拍、动画、电影感、纪录片。 |
| `language` | 答案用哪种语言回来。留空表示跟着输入的语言走。在这个节点上它是第二遍 —— 先把那一行缩短，之后再翻译 —— 正是这一点让它靠得住；下面那个小节说明它为什么必须如此。 |
| `greedy`、`seed`、`keep_model_loaded` | 和写作节点上一样。`greedy` 在这儿值得一直开着：把“一只黑猫”变成“一只毛色如黑曜石般油亮的猫科动物”的，正是采样。 |
| `options` | 可选。就是平常那个 [Rewriter Options](#minimax-h3-rewriter-options-重写器选项) 节点 —— `device`、`n_ctx`、`gpu_layers`、`gguf_runtime` 等等。 |
| `bypass` | 把 `prompt` 直接递到输出，什么都不跑。 |
| `system_prompt` | 用你自己的指令替掉整段拼好的指令。`detail`、`subjects` 和那三个 keep 于是不再起作用 —— 它们存在的唯一意义，就是拼出你现在要替掉的那段文字。`language` 仍然有效，因为它是一次单独的请求，不是那段文字的一部分。两种情况下解析照做，所以递到你那条指令手上的是清理过的场景，而不是原始文本。 |

两个输出：`short_prompt`，以及 `scene` —— 就是拆掉脚手架、别的什么都没做过的那段
描述。没有任何模型碰过 `scene`。要是你想要的只是确定性的那一半，就接它。

![MiniMax-H3 提示词精简器节点，旁边是一个 Show Any 节点：左边一个 options 插槽，右边两个输出 short_prompt 和 scene，一个高高的提示词框里装着一整段 I2VA 提示词，带着它的 integrated_multimodal_description 和 [Shot 1] 标记，然后是那些小部件 —— 一个硬盘上的 35B 模型、detail 设为 idea、subjects 设为 impersonal、keep_camera 和 keep_audio 打开、keep_style 关闭、language 写着 Chinese、greedy 打开、种子设为 randomize、keep_model_loaded 和 bypass 关闭，以及一个空的 system_prompt 框，框下面是 Model list 按钮。节点下面那行状态写着 “110 words in, 33 characters out - 1 shot markers dropped - translated into Chinese”，再下面就是那两句中文本身，Show Any 节点把它们又显示了一遍](docs/node_reducer.png)

*四百个词的猫、篱笆和黄昏，缩成两句短短的中文，在一个 35B 上花了 25 秒。有三个轴
同时在做看得见的事：`detail` 在 `idea` 上，所以回来的是光秃秃的一行；`keep_camera`
把那个低角度的跟拍镜头放了回去；`keep_audio` 把整段声景折进了第二句；而 `keep_style`
是关的，所以提示词开头那句 `Live-action, cinematic` 不见了。计数用的是字符而不是词，
因为中文不写词数所需要的那些空格。*

*它还老老实实地展示了一件事：`subjects` 在 `impersonal` 上，而那只猫回来时仍然是
黑的。规则要的是光秃秃的种类，而这个模型把颜色留下了 —— 三个轴里，这一个最依赖
模型，而且它真正针对的是人。*

#### 这几个轴实际上做了什么

一段提示词，一个 4B，几个轴一次只动一个。输入是一段纪录片风格的 I2VA 提示词，讲
一位老渔民拉网，两个镜头，带一段声景和一段大提琴的持续低音：

| 设置 | 回来的是什么 |
|---|---|
| 默认 | An elderly fisherman with a thick grey beard and a faded yellow oilskin jacket hauls a dripping net over the gunwale of a small wooden trawler under a bruised pre-dawn sky. |
| `detail: idea` | An elderly fisherman hauls a net on a boat. |
| `subjects: age and gender` | An elderly man hauls a net on a boat under a pre-dawn sky. |
| `subjects: impersonal` | A subject hauls a net over the side of a boat under a pre-dawn sky. |
| `keep_camera` | A handheld close-up follows an elderly fisherman... |
| `keep_style` | Live-action, documentary style. An elderly fisherman... |
| `keep_audio` | ...under a pre-dawn sky. Waves slap the hull, rope creaks, gulls cry overhead, and the man breathes heavily. A low cello drone underneath. |

`impersonal` 是给那种事后再填空的模板用的。原样喂给生成器，它产出的正是它所要求的
那种匿名的空无。

这里是四个轴，而不是一个“抽象程度”旋钮，因为它们彼此独立。答案有多长，和它把人
点名到多具体，是两个不同的问题 —— 一行的提示词照样可以说 “a woman in a red coat”
—— 而镜头、声音和影像观感，是各自单独决定留还是不留的。

#### `language` 是第二遍，而且它只能是第二遍

把精简和语言放进同一次请求里去要，是行不通的。指令里那个示范例子是用英文写的，而且
不可能是别的语言 —— 这里没有任何东西能把它翻成你敲进小部件里的那种语言 —— 而一个
照着示范学的模型，会把它的语言连同它从中照抄的其他一切一起抄过来。

这个失败不是一律的，正因如此才值得写下来。三个 keep *关着*时，例子很短，语言那条
规则被遵守了。它们*开着*时，例子变长也变复杂，接连三个模型都用英文作答 —— 不管
那条规则怎么措辞、放在哪儿都一样，连放在系统提示词的最后一行、放在例子后面也一样。

所以精简器是先缩短，之后再用一次自己的请求把写好的那一行翻过去。那次请求只有一个
目标，没有例子可抄，也没有什么需要权衡，而同样这些模型就照办了：试过的每一种组合
都按要求的语言回来，4B、9B 和 35B 上都是，俄语和中文都是。它的代价是在一个已经
加载好的模型上多做一次很短的生成，而且不管 `keep_model_loaded` 对之后怎么说，节点
都会在这两次之间让模型保持加载。

这个设计带来两个结论：

- **`system_prompt` 关不掉它。** 三个 keep、`detail` 和 `subjects` 在你替掉指令时
  全都不再起作用，因为它们存在的唯一意义就是拼出你替掉的那段文字。`language` 根本
  不在那段文字里，所以它照样起作用。
- **`Reduce Prompt (any LLM)` 做不到这件事**，因为它只拼出一次请求，什么都不跑。
  在那儿，语言是指令内部的一条规则，遵不遵守取决于模型 —— 正是上面描述的那种行为。
  要是从那个节点回来的短提示词语言不对，原因就在这儿，而精简器是靠得住的那条路。

剩下的问题是翻译质量，而不是拒绝翻译：一个小模型偶尔会留下一两个没翻的词。那是模型
的事，换大一点的就少一些。

#### 特效 token 穿过精简器

写进描述里的一个[特效嵌入](#minimax-h3-effect-embeddings-特效嵌入)会原样从另一头
出来，连那个小写的 `e` 都在 —— 那条把句首整理成大写的规则，知道要放过它。而放在
第一个字段标签*上方*的 token 活不下来，因为提示词的头部会连同对齐语句、以及一切
既不是描述也不是声音的东西一起被丢掉。这是精简器的约定，不是疏漏：它交回来的是
一行可以复用的句子，而一段 H3 提示词的头部把它绑死在特定的那几张参考帧上。

还有一点值得知道：这个节点和检查节点上的词数统计，都把一个 token 算作一个词。它在
提示词位置上的真实开销，写在嵌入节点的 `tokens` 输出上。

#### 一条值得知道的循环

精简器 → 改那一行 → 写作节点 →
[Prompt Check](#minimax-h3-prompt-check-提示词检查)。转完这一圈，猫要是还在篱笆上，
那就说明节点包的两半都在好好干活。这也是最便宜的办法，用来弄清某个小模型在这件事上
够不够格：跑一遍，读那一行。

### MiniMax-H3 Reduce Prompt (any LLM) 精简提示词

同样的精简，但什么都不跑：它把 `system_prompt` 和 `user_prompt` 拼出来，以字符串
交给你本来就在用的那个 LLM 节点 —— 本地的、API 的、远程的都行。不花显存，也不花
时间。解析仍然在这里做，所以你的模型收到的场景已经是干净的了。

![MiniMax-H3 Reduce Prompt (any LLM) 节点，旁边一个 Show Any 节点显示着拼好的系统提示词：一串规则 —— 绝不把一个词往上升级、什么都不要加、宁可丢掉写法也不要丢掉故事、用一句话作答、按原文点名主体、丢掉镜头、丢掉媒介、答案用俄语写 —— 然后是输出格式的约定，然后是那个示范例子，带着它的场景和它的答案 “A black cat walks along a wooden fence in a yard at dusk”，最后是两行，说明例子是英文的、而答案要用俄语。节点右边从上到下四个输出：system_prompt、user_prompt、prompt 和 scene](docs/node_reduce_prompt.png)

*同样的精简，什么都不跑，0.01 秒。整段指令在这儿一览无余，而这正是这个节点的用意：
2112 个字符的系统提示词、632 个字符的用户提示词，里面每一条规则都是某个小部件放
进去的。最后那两行就是上面说的“一次请求搞定语言”的尝试 —— 这个节点只能拼出一次
请求，所以这已经是它能做到的最好了。*

掌舵的是同样那四个轴，`format` 把两个输出接在一起的方式和在
[指南提示词](#minimax-h3-guide-prompt-any-llm-指南提示词)上一样，第四个输出是
`scene`，含义同上。

### MiniMax-H3 Effect Embeddings 特效嵌入

MiniMax 随 H3 一起发布了十个特效嵌入 —— 子弹时间、黑魔法、喷火、《楚门的世界》那种
拉镜，还有另外六个。这个节点把它们放进提示词里。

它们不是关键词，而这个区别关系到下面的每一件事。每个文件里装着一个形状为
`[N, 5120]` 的张量：五十到一百四十个**已经过完文本编码器**的提示词位置，以数字的
形式存了下来。在文本里写 `embedding:minimaxh3_bullet_time`，不是用词句去请求子弹
时间 —— 它是给 ComfyUI 分词器的一条指令，要它把那些位置在这一点上拼进序列里。所以
一个特效花掉的是提示词的*长度*，而不是形容词；它要么以全强度到来，要么根本不来；
也不存在换个说法这回事。这里没有什么可措辞的。

| 特效 | 位置数 | 特效 | 位置数 |
|---|---|---|---|
| Art is explosion | 50 | Kiss camera | 97 |
| Dark magic | 59 | Fire breath | 118 |
| Truman show | 90 | Blooming flowers | 123 |
| Bullet time | 94 | Spiral ascent | 131 |
| Storm magic | 137 | Four seasons | 142 |

**这需要 ComfyUI 0.33.0 或者更新的版本。** 正是那一版让 `embedding:` 能到达
MiniMax-H3 的分词器；在更旧的版本上，这个 token 会被当成普通词句来读。节点会检查
你的版本并把这件事说出来。

![MiniMax-H3 特效嵌入节点，旁边一个 Show Any 节点。标题栏上方有两个徽标：节点包自己的那个，和一个写着 “bypass” 的灰色徽标。右侧从上到下三个输出 —— prompt（接到另一个节点去了）、tokens 和 findings。它们下面的文本框里是一段三个字段的 T2VA 提示词：integrated_multimodal_description: A cat walks along a fence at dusk、overall_soundscape: Wind in the grass、non_diegetic_music: N/A。再下面 placement 小部件写着 “start of the description”。然后是那十个一格的网格，每一行一个复选框、一个名字和一份开销：Art is explosion 50 tok、Blooming flowers 123 tok、Bullet time 94 tok、Dark magic 59 tok、Fire breath 118 tok、Four seasons 142 tok、Kiss camera 97 tok、Spiral ascent 131 tok、Storm magic 137 tok、Truman show 90 tok。打了勾的有三个 —— Art is explosion、Dark magic 和 Fire breath —— 它们的方框是绿的，里面一个白色的勾。网格下面一行写着 “3 selected, 227 tokens”，再下面 bypass 小部件写着 false。没有下载按钮。Show Any 节点显示着结果：三个 token 连成一串，跟在 integrated_multimodal_description 标签后面、那句话前面，然后是原样的 overall_soundscape 和 non_diegetic_music](docs/node_effect_embeddings.png)

*这里什么都不加载、什么都不生成 —— 节点往一段文字里写了一百个字符，读了十个文件头。
下载按钮是不见了，而不是变灰：十个全在硬盘上，所以没什么可取的，那一行也没有理由占
地方。右边那一列的开销是从文件本身读出来的，不是本节点包里的一张表；50 + 59 + 118
就是网格底下那个 227。这是 `start of the description` 那一档，所以这三个 token 是把
描述字段打开了，而不是待在它上面 —— 也正因如此，它们下面那些空行到达 H3 时会变成一个
个空格，而 `findings` 输出把这件事说了出来。*

| 输入 | 用来做什么 |
|---|---|
| `prompt` | 要把 token 放进去的那段提示词 —— 某个写作节点的输出、加载进来的一个文件，或者手打的东西。除了 token 本身，它是一个字符不差地穿过去的。 |
| `placement` | token 放在哪儿：描述的开头、提示词的顶上，或者末尾。这不是审美问题 —— 见下文。 |
| `effects` | 那个十格的网格。一个勾、一个名字、它花多少，以及还没下载的那些身上的标记。 |
| `bypass` | 让 `prompt` 直接穿过去，一个 token 也不放。`tokens` 返回 0，`findings` 为空，也不去读任何文件头。这个开关像写作节点那样，在节点标题栏上有一个徽标。 |

| 输出 | 是什么 |
|---|---|
| `prompt` | 放好 token 的那段提示词。 |
| `tokens` | 选中的这些特效占掉多少个位置 —— 是从每个文件自己的头里读出来的，所以是真实的数字，不是本节点包里的一张表。 |
| `findings` | 节点注意到的事，一行一条。没什么可说时它是空的。 |

文件由节点上的按钮取下来，放进 ComfyUI 自己的 `models/embeddings` 文件夹 —— 那是
`embedding:` 唯一会去找的地方。十个加起来大约十兆。把节点跑两遍不会把 token 叠起来
—— 它在放进去之前会先把自己上一次放的取出来 —— 所以把某个特效的勾去掉，就能把它从
一段已经带着它的提示词里拿走。

#### 它失败的每一种方式都是无声的

这正是它之所以是一个节点、而不是让你手打的原因。ComfyUI 碰到一个用不了的 token，
会往控制台里写一行，此外没有任何迹象：视频照样生成，只是生成时没有那个特效，看上去
和一段从来没要过特效的提示词一模一样。有三种情况会这样，而自检如今三种都会去找：

- **一个大写字母。** 判断条件只有 `word.startswith("embedding:")`，别的什么都不看，
  所以 `Embedding:` 并不比一个拼写错误更轻。（提示词精简器从前会因为把句首改成大写
  而弄出这种情况。现在不会了。）
- **token 粘在了前一个词上。** token 是在空白之后、或者在文本的最开头被找出来的。
  `a cat.embedding:x` 就是四个普通的词。
- **名字末尾的一个句号。** 末尾的逗号会被 ComfyUI 剥掉、放过；句号则会被当作文件名
  的一部分去找。这一条值得精确地知道，因为它是一个藏在 Windows 上的陷阱：那边的
  路径层会把末尾的点丢掉，所以 `embedding:minimaxh3_dark_magic.` 在 Windows 上能
  加载，到了 Linux 和 macOS 上就悄悄丢掉了那个特效。你分享出去的提示词，不是他们
  跑的那段提示词。

权重也不管用 —— `(embedding:x:0.8)` 会被当成字面文本来读，因为 H3 分词时权重是关掉
的。没有办法要一个弱一点的特效。

#### 放在哪儿不是审美问题

ComfyUI 的分词器会把一个 token 后面的所有东西并成一行：`embedding:name` 之后、直到
下一个 token 为止的那段文本，在 H3 看到它之前要先过一遍 `' '.join(text.split())`。
所以放在提示词最顶上的一个 token，会把它下面各字段之间的每一个空行都压成一个空格。

`start of the description` —— 默认这一档 —— 是在描述字段自己的标签之后把它开头。
那是写作节点放场景的字段，也是下游每个节点读的字段，而且正是一个特效本该修饰的位置。
它下面的那些字段会被压平。

`end of the prompt` 把 token 放在最后。它们后面什么都没有，所以什么都不会被压平，
提示词到达 H3 时，字段之间的分隔完好无损。

`top of the prompt` 把它们放在一切之上，在对齐语句之前。

一段完全没有字段标签的提示词，没有描述可以开头，于是它回退到顶部，并在 `findings`
输出上把这件事说出来。

用精简器的话，有一件事要知道：位于第一个字段标签之上的 token 会被它丢掉，连同提示词
头部的其余内容一起。精简器留下的是描述和声音，不是它们上面的东西。而描述内部的
token 活得下来。

这里*不*知道的是：这些特效两个凑在一起，能不能凑出什么讲得通的东西。它们是各自独立
的、已经编码好的提示词片段，而要一次穿过四季的螺旋上升，等于是让模型自己去调和它们。
节点允许任意组合，因为没有理由去禁止哪一种；值不值得生成，由你自己去发现。

### MiniMax-H3 LoRA Triggers LoRA 触发词

一个在标注前缀里带着触发词训练出来的 LoRA，只要那个词不在提示词里，就什么都不做。
这个节点把这些词的清单留在节点上、跟着工作流一起存下来，并把它们放回写好的提示词里。

上游没有谁知道它们的存在。写作节点写散文，精简器把它缩短，自检去评判它 —— 而一个
唯一的用处是让三个节点之后的适配器醒过来的词，正是这三者都会丢掉的东西。于是它在
每次运行之后被手打进去，又在下一次运行时丢掉。这是整条流水线上唯一一处要手改成品
文本的地方，也是提示词里唯一无法重新生成的部分。

一行就是一个适配器：一个抓手、一个勾、一个你自己看的名字、会原样进入提示词的那些词、
一个说明它们去哪儿的短标记，以及一个删掉这一行的叉。一行里可以放好几个词，用逗号隔开 ——
带不止一个触发词的适配器要的正是这个；它们是一个一个地检查、一个一个地放进去的，
所以一行永远不会只重复一半。要去提示词不同部位的触发词，要用不同的行。

拽住抓手就能把一行挪走。次序不是装饰：同一个放置档位上的几行，是按清单里的先后写进
提示词的，读起来也就是那个先后。

![MiniMax-H3 LoRA 触发词节点，旁边一个 Show Any 节点。标题栏上方有两个徽标：节点包自己的那个，和一个写着 “bypass” 的灰色徽标。右侧从上到下三个输出 —— prompt（接到另一个节点去了）、added 和 findings。它们下面的文本框里是一段三个字段的 T2VA 提示词：integrated_multimodal_description: A cat walks along a fence at dusk、overall_soundscape: Wind in the grass、non_diegetic_music: N/A。再下面是清单的两行。每一行都是一个拖拽抓手、一个打了白勾的绿色复选框、一个放名字的输入框、一个更宽的放词的输入框、一个放置标记和一个叉：“Facial realism” 里放着 “ohwx face”，标记是 body；“Neon” 里放着 “neon glow, wet asphalt”，标记是 top。两个标记排成一列，两对输入框也是。它们下面一行写着 “2 on of 2”，再下面 bypass 小部件写着 false，然后是一个通栏按钮 “add a trigger”。Show Any 节点显示着结果：第一行孤零零一个 “neon glow, wet asphalt”，一个空行，然后 integrated_multimodal_description 以 “ohwx face.” 开头、接着才是那句话，而 overall_soundscape 和 non_diegetic_music 原样不动](docs/node_lora_triggers.png)

*两行、两个档位、一趟跑完。`ohwx face` 把描述打开了 —— 在标签之后、散文之前，正是
一个来自标注前缀的触发词受训时所处的位置 —— 而第二行一次带着两个触发词去了顶上，
它们是分开检查、分开写进去的。为此并不需要拿走任何东西：把同一段文本再跑一遍，
`added` 会是空的，因为两个词都已经在里面了。放置标记排成固定的一列，正是它让每一行
的输入框都从同一个像素开始、到同一个像素结束；拽住某一行的抓手，就能改变它们被写进去
的先后。*

| 输入 | 用来做什么 |
|---|---|
| `prompt` | 要把词放进去的那段提示词 —— 某个写作节点的输出、精简器的输出、加载进来的一个文件，或者手打的东西。除了放进去的那些词，它是一个字符不差地穿过去的。 |
| `triggers` | 那份清单。一个抓手、一个勾、一个名字、那些词、放置标记、一个叉；下面的按钮加一行。跟工作流一起存下来。 |
| `bypass` | 让 `prompt` 直接穿过去，什么都不加。文字进来什么样出去就什么样；`added` 和 `findings` 返回空。这个开关像写作节点那样，在节点标题栏上有一个徽标。 |

| 输出 | 是什么 |
|---|---|
| `prompt` | 放好词的那段提示词。 |
| `added` | 这一次运行真正放进去的词，用逗号隔开。当所有打了勾的触发词本来就在里面时它是空的 —— 第二次运行时这是正常状态。 |
| `findings` | 节点做了什么、注意到什么，一行一条。没什么可说时它是空的。 |

如果是通过 API 而不是界面来驱动它：这个小部件收界面写出来的那种
`{name, words, where, on}` 对象的 JSON 数组 —— 另外还收两种更省事的写法，因为那是
任何人最先会写的东西。一个普通的字符串数组会被读成一份触发词清单；根本不是 JSON 的
一段文本，会被读成一行的那些词。

#### 每个词去哪儿

行里的标记短是故意的。什么都不展开就能一眼看出每个词要去哪儿，这是这份清单一半的
意义所在；而五个完整的下拉框，在任何合理的节点宽度上都排不下。点一下标记就能改。

| 标记 | 放置档位 | 落在哪儿 |
|---|---|---|
| `body` | `start of the description` | 打开描述字段本身，紧接在它的标签后面。 |
| `top` | `top of the prompt` | 在一切之上，在字段标签前面。 |
| `sound` | `start of the soundscape` | 打开 `overall_soundscape`，紧接在它的标签后面。 |
| `music` | `start of the music` | 打开 `non_diegetic_music`，紧接在它的标签后面。 |
| `end` | `end of the prompt` | 在最后，在一切之后。 |

描述是默认档，通常也是对的那个：那是写作节点放场景的字段，是下游每个节点都会读的
字段，也正是一个来自标注前缀的触发词受训时所处的位置。顶上和末尾是位置而不是字段
—— 所以在一段根本没有标签的文本上，还能用的只有这两个。

字段集合按任务而不同 —— T2VA 三个，Ref2VA 六个 —— 而节点看见的永远只是一段文字，
文字并不会说它是哪个任务写的。所以一个档位指名了这段提示词并没有的字段，这不是错误：
那些词改去顶上，`findings` 输出会把这件事说出来。

一个在、但写着 `N/A` 的字段，会另外得到一条提示。Ref2VA 的提示词里到处都是
`non_diegetic_music: N/A`，而放在这种字段开头的词，会留下一个既说出了什么、又紧接着
说这东西没有的字段。节点把这件事说出来，别的什么都不改。把那个标记换掉，等于让这个
字段声称音乐是有的、叫作 `ohwx face` —— 而自检会读这个字段；把那些词丢掉，又等于
节点推翻了你要的那个放置档位。这两样都是关于你这段提示词的决定，而这个节点不做这种
决定。

有一处和特效嵌入的区别值得知道，因为那边的习惯不能照搬过来：普通的词不会把它们下面
的换行压平。ComfyUI 的分词器只会把 `embedding:` token 之后的东西并成一行 —— 也只在
它之后。在这里，提示词的顶上是个安全的地方。

#### 它从不拿走任何东西

这正是它和 `MiniMax-H3 Effect Embeddings` 分道扬镳的地方，理由值得说清楚。那十个
token 是事先就知道的字符串，所以那个节点能在放进去之前先把自己上一次放的取出来 ——
去掉某个特效的勾之所以能把它拿走，就是这个道理。而这里的词是你自己的。把一个写成
`detail` 的触发词剪掉，就是在它待着的那段散文上剪出一个窟窿。

所以这里的规矩是另一条 —— 检查在不在，而不是拿走：

- 已经在文本里的词**不会被放第二遍** —— 按词边界匹配、不分大小写，所以 `man` 不会
  在 `woman` 里被找到，而 `Ohwx man` 算作 `ohwx man`。这件事会写进 `findings`。
- **去掉勾，只是不再添加。它不会拿走。** 那个词很可能本来就是写作节点写的，而删掉
  别人的词不是这个节点该做的决定。

因此，同一段文本跑两遍什么都不会变 —— 而这正是这个节点要待在一张会被一遍遍运行的图
的尾巴上所需要的。

#### 用 `bypass`，而不是 Ctrl+B

这两个节点都带着本节点包自己的 `bypass` 开关，标题栏上有和写作节点一样的紫色徽标：
点一下徽标，或者把那个小部件勾上，提示词就原封不动地穿过去。

在这样的节点上，ComfyUI 自己的旁路做的不是同一件事。它按类型给每个输出找一个输入
来顶替，而这里有三个输出、只有一个插槽可用。在触发词节点上实测：`prompt` 到了，
`findings` 收到的是整段提示词、仿佛那是一条发现，而 `added` 干脆把连线丢了 ——
于是它下面那个节点什么输入都没有。在特效嵌入节点上，`tokens` 是 INT，没有 INT 输入
能顶替它，那根连线也是同样的下场。这些事情哪儿都不会报出来。

#### 把它放在图的最末尾

放在精简器之后，也放在 `MiniMax-H3 Prompt Check` 之后。触发词会增加词数，而检查会
数词数并和上限比较：一个在文本被量过之后才把它加长的节点没问题，一个在量之前就加长
它的节点，是在和这次测量较劲。而任何在它下游重写提示词的东西，都会把这些词再丢一次
—— 那正是这个节点存在所要防的事。

#### 它不承诺什么

只管那些在标注前缀里带着触发词训练出来的 LoRA。滑块类和 turbo 加速类根本不需要词
—— 它们靠权重工作 —— 而在多数机器上，装着的多半正是这一类。一份触发词清单对它们
毫无用处，而这不是清单的毛病。

不会从 LoRA 文件里读任何东西，而这是一个结论，不是一处偷工。在写下这段话的这台机器
上的四十四个适配器里，有一个带着 `trigger_word`，有一个带着空的 `trigger_words`，
没有一个带着 `ss_tag_frequency`。这个节点包本来就会读 safetensors 的文件头，所以问题
不在成本上：数据根本就不在那儿。而元数据里只有一个词的位置，适配器却常常不止听一个词。
清单归你来管 —— 而这也正是它值得管的原因。

### MiniMax-H3 Reference Adapter 参考素材适配器

写作节点是一个插槽收一份参考素材，而这正是它们下面那条素材条得以成立的原因：每一份
素材都有一个方块、一个角色和一个开关。同样也正因如此，它们够不着那些把参考素材
*一起*递过来的节点 —— 一批图片、目录加载器给出的一个列表、另一个节点包拼好的一个
bundle。一个值里装着许多份，没有哪个插槽的形状接得住它。

这个节点就是那个接头。从一个插槽收进来，分到九个图片插槽、三个片段插槽和三个声音
插槽上出去 —— 那正是写作节点收得下的形状，所以素材条照旧一如既往地工作。

![MiniMax-H3 参考素材适配器节点。左边两个输入：没接东西的 items，以及接向画面之外某处的 bundle。右边十六个输出：picture_1 到 picture_9、video_1 到 video_3、audio_1 到 audio_3，还有 summary。前三个 picture 输出接到三个预览节点上，显示着三张互不相干的照片。只有一个小部件 split_batches，写着 true；它下面一行文字写着 “3 picture(s), 0 clip(s), 0 sound(s)”。节点的运行时间是 0.014 秒](docs/node_ref_adapter.png)

*一根线进来，三根出去 —— 另有十三个输出闲着，而这是常态，不是哪里没做完的迹象。这
一次运行花了 14 毫秒，因为这里什么都不加载、不解码、也不描述：那三张照片是原样被
带过去的。小部件底下那行字和 `summary` 输出是同一段文本，所以进来了什么、去了哪儿，
不用往预览节点上接线，在节点上就读得到。*

| 输入 | 用来做什么 |
|---|---|
| `items` | 一起到来的参考素材。什么类型都收，因为产出集合的那些节点大多根本不声明类型；每一项是什么，是从值本身推断出来的，而不是从线上；不是图片、片段或者声音的东西会被跳过，并被计数。 |
| `bundle` | 别的节点包给出的一个参考素材 bundle，如果你手上有的话。它里面的图片、片段和声音会先于 `items` 上的任何东西被读出来，以帧的形式到来的片段会作为片段输出，而不是一张张图片，与它配对的音轨也装在它里面；没有和片段配对的音轨，才作为独立的声音。 |
| `split_batches` | 一批图片是变成一帧一份参考素材，还是仍旧算一份由若干帧组成的参考素材。 |

最后那个开关是一个真正的选择，不是走个形式。拆开，六帧就是视频要复用的六样东西，
各自被描述、各自编号。不拆，它们是同一样东西被看了六次、只被描述一遍 —— 而这正是
一段片段的样子。当这一批就是同一个镜头的若干帧时，把它关掉。

九、三、三就是 Ref2VA 装得下的数量，所以位置就这么多。超出去的部分会在 `summary`
输出上被报出来，而不是无声地丢掉，被跳过的东西也一样。多余的输出什么都不递，而写作
节点本来就会跳过它们，所以接一个最后什么都没有的插槽毫无害处 —— 九个全接上，没填上
的就让它们闲着。

它自成一个节点，理由值得知道，因为这并不是一个选择。要接收一个真正的 ComfyUI 列表，
就得声明 `is_input_list`，而这个标志不是按输入来的：它会改写这个节点收到的*每一个*
参数的形状。放在写作节点上，它会改变提示词、时长和选项抵达的方式。所以它住在这里，
住在一个不会因此再损失什么的节点上。

### MiniMax-H3 Reference Slots 参考素材插槽

`MiniMaxH3ReferenceToVideo` —— ComfyUI 自带的、给 H3 参考模式用的条件节点 —— 是按
参考素材抵达的插槽来编号的：第一个图片插槽是 `<Picture 1>`，第一个声音插槽是
`<Audio 1>`。这里的写作节点则按素材条来编号，而素材条是可以拖动的。把同一批素材
手工分别接到两边，两套顺序只在第一次拖动方块之前是一致的；从那以后，提示词描述的是
一个声音，视频拿到的却是另一个，而屏幕上什么提示都没有。关掉一份参考素材也会以同样的
方式出错，只是方向反过来 —— 提示词里没了，视频里还在。

所以，每个接收参考素材的写作节点都会把它编好号的东西，放在最后一个名为 `references`
的输出上递出去，而这个节点把每份参考素材放到带着它编号的那个插槽上。

![MiniMax-H3 参考素材插槽节点，位于 Universal Writer 和 MiniMax H3 Reference to Video 之间。写作节点设为 Ref2VA，接着四份参考素材，素材条上依次是 ref_0 上的 pic 1、ref_2 上的 aud 1、ref_1 上的 pic 2 和 ref_3 上的 aud 2；它的最后一个输出 references 接进 Reference Slots 节点。这个节点右边有十九个输出 —— picture_1 到 picture_9、video_1 到 video_3、video_audio_1 到 video_audio_3、audio_1 到 audio_3，还有 summary —— 每个媒体输出都按同样的顺序直接接到生成节点的输入上：ref_image_0 到 ref_image_8、ref_video_0 到 ref_video_2、ref_video_audio_0 到 ref_video_audio_2，以及 ref_audio_0 到 ref_audio_2。soundtracks 开关写着 false；它下面是 “2 picture(s), 0 clip(s), 2 sound(s)” 和 “picture_1 ← ref_0”。写作节点运行了 0.011 秒，插槽节点 0.019 秒](docs/node_reference_slots.png)

*接一次，全部接通，之后再也不用碰：摆放全在写作节点的素材条上完成。方块依次是 ref_0、ref_2、ref_1、ref_3 —— 所以 `picture_2` 装的是 ref_1，`audio_1` 装的是 ref_2，不管它们当初插在哪个插槽上，生成节点给它们的编号都和提示词里的一模一样。空着的输出不花任何代价：生成节点会跳过空插槽。十九毫秒，因为图片和声音都是原样通过的，也没有需要解码的片段。*

```
Universal Writer ─ references ─▶ Reference Slots ─ picture_1 ─▶ ref_image_0 ┐
                                                 ─ picture_2 ─▶ ref_image_1 ├ MiniMaxH3ReferenceToVideo
                                                 ─ audio_1   ─▶ ref_audio_0 ┘
```

按顺序直接接一次 —— `picture_N` 接第 N 个 `ref_image`，`video_N` 接第 N 个
`ref_video`，`video_audio_N` 接第 N 个 `ref_video_audio`，`audio_N` 接第 N 个
`ref_audio` —— 之后的摆放全在素材条上做。

| 输出 | 是什么 |
|---|---|
| `picture_1` … `picture_9` | 图片，按写作节点的顺序。 |
| `video_1` … `video_3` | 片段，以 24 fps 的帧给出。 |
| `video_audio_1` … `video_audio_3` | 片段自带的声音，按编号配对 —— 写作节点算上了它的时候才有，或者打开 `soundtracks` 时每个片段都有。 |
| `audio_1` … `audio_3` | 声音，原样给出。 |
| `summary` | 哪个输出上放着什么、来自写作节点的哪个插槽，以及没放下的东西。 |

**关掉的参考素材在两边同时消失。** 它根本不在 `references` 里，所以它的插槽什么都
不递；`MiniMaxH3ReferenceToVideo` 会跳过空插槽，把编号在空位处合拢 —— 和写作节点对
自己的标签做的事情一模一样。

**主体排在图片后面。** 在 Universal Writer 上，被标成主体的图片写出来是 `Subject N`，
从来不是 `<Picture N>`。生成节点没有主体插槽，会像对待其他图片一样给它一个图片编号，
所以它排在最后一张图片之后：那里的编号是提示词没在用的。

**片段在这里解码，而不是在写作节点里。** 不管 `references` 有没有接东西，写作节点
都会运行，而十五秒 1080p 的浮点帧有好几个 GB，所以片段以它进来时的 VIDEO 形态传递，
在这个节点运行之前不花任何代价。在这里它以 24 fps 读取 —— 30 fps 的片段每五帧丢一帧，
12 fps 的片段每帧显示两次 —— 缩放到 `MiniMaxH3ReferenceToVideo` 本来就会给它的那块
画布上，并在十五秒处截断，那是它设计时所针对的最长参考素材。被裁剪过的 VIDEO 按它的
裁剪范围读取。写作节点被告知当作片段来读的一批图片本来就是帧，原样通过。

**写作节点算上了片段的声音，它才会被递过去。** 生成节点会给片段的声音单独一个
`<Audio N>`，编号排在所有独立声音之前。在写作节点的素材条里点片段的标签条 —— 或者在
Universal Rewriter 的 `reference_video` 那一行勾上喇叭框 —— 写作节点就会按同样的方式
给这个声音编号，让它被描述或被听到，并在 `references` 里注明。这个节点随后把它放到
片段的 `video_audio_N` 上，提示词和视频就对得上。没有音轨的片段这里没东西可放，写作
节点也已经说明过，没给它编号。

**`soundtracks` 不管写作节点怎么做，都给每个片段递出声音。** 它默认关闭。写作节点
没算声音的片段这时会拿到一个提示词里没有的 `<Audio N>`，于是提示词里的 `<Audio 1>`
就不再是 `audio_1` 上的那个声音了；`summary` 会说标签移动了几位。

每个写作节点递出去的，是它的提示词里点到名的东西，它没读过的一样都不递：

| 节点 | `references` 装着什么 |
|---|---|
| Universal Writer | 每一个打开的方块，按素材条顺序。T2VA 上什么都没有。 |
| Prompt Rewriter Omni | 每一个打开的方块，按素材条顺序。T2AV 上什么都没有。 |
| Universal Rewriter | 27B 标签页和 T2VA 上什么都没有。8B 标签页上是任务读取的那几帧。Omni 标签页上是同样的帧 —— 或者在 Ref2VA 上，按行的顺序给出每一个打开的行。 |

这个输出每次都从当前的输入构建，`bypass`、从提示词库里选取和 `repeat_last` 也不例外，
而且从不被保存：提示词库和节点的记忆只保存文本，一段保存过的提示词回来时，带的是此刻
接着的那些参考素材。

> **这不是 Reference Adapter。** 那个节点喂的是写作节点：它把进来的东西按种类分好，
> 把片段作为 VIDEO 递出去，因为写作节点描述的就是 VIDEO。这个节点喂的是生成节点，
> 生成节点要的是片段的帧，以及单独一个插槽上的片段声音。`references` 自成一个类型也是
> 这个原因 —— 如果插进一个期待别的节点包 bundle 的节点，片段会以 VIDEO 的形态抵达，
> 然后在生成节点内部出错。

> **有一种串联它不覆盖。** `previous` 上接着一个块的 Universal Writer，会先数这个块里
> 的标签，而它的 `references` 只装着它自己的参考素材，从一开始编号。当输出要去生成
> 节点时，每个镜头只用一个写作节点。

> **遇到 RefMod 它会停下。** `MiniMaxH3ReferenceToVideo` 自己编码交给它的图片、片段和
> 声音，而 [RefMod](#minimax-h3-refmod-已保存的参考素材) 已经编码过了，没有插槽可放。
> 这个节点会说明这一点，并指向
> [MiniMax-H3 Reference Encode](#minimax-h3-reference-encode-参考素材编码) —— 它接收同样的
> `references`，并取代这两个节点。

### MiniMax-H3 RefMod 已保存的参考素材

RefMod 是一份编码过一次并保存下来的参考素材：图片、片段、一组照片或一个声音经 H3 VAE
得到的潜变量，存在 `ComfyUI/models/refmods` 下的 `.safetensors` 文件里，文件头写明它是
什么。格式来自 [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)。
大家在 Hugging Face 上分享的 RefMod、那个节点包的单文件 bundle，以及
[ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder)
写出的文件，这里都能读。制作和编辑它们是那两个节点包的事；这个节点包只使用它们。

这个节点选出一个，交给写作节点的素材条。从文件到视频的整条链路，模板
**8 - RefMods to video** 里都已经搭好了。

| 输入 | 用途 |
|---|---|
| `refmod` | 条目，按它在 `models/refmods` 下的路径。一起保存的外观和声音 —— 一个 bundle 文件，或一对名为 `<名称>_visual` 和 `<名称>_audio` 的文件 —— 是同一个条目。之后新加的文件，在图里按 **R** 刷新列表。 |
| `strength` | 生成节点遵循它的强度。1 表示按保存的样子。更低的值会把潜变量向一份高度模糊的自身副本混合：颜色、构图和大致形状保留，细节消失；给 H3 文本编码器看的图片也以同样方式减弱。写作节点读到的始终是完整强度。 |
| `vae` | H3 视频 VAE。大多数文件不保存自身的图片 —— Hugging Face 上的一个都没有 —— 而写作节点要看见参考素材才能描述它，所以外观在这里解码一次。 |
| `audio_vae` | H3 音频 VAE：对声音做同样的事，让写作节点能听见它。 |

| 输出 | 是什么 |
|---|---|
| `visual` | 图片、片段或一组照片。条目只有声音时为空。 |
| `audio` | 声音。条目没有声音时为空。 |

每个输出接到 [Universal Writer](#minimax-h3-universal-writer-通用写作节点) 或
[Prompt Rewriter Omni](#minimax-h3-prompt-rewriter-omni-看得见也听得见) 素材条上各自的
方块里，方块上标着 `mod`：

![Universal Writer 的 ref_0 和 ref_1 接着同一份 RefMod 的两个输出，ref_2 空着、没有勾选。素材条上有两个标着 mod 的方块：ref_0 上方是绿色的 vid，显示一张解码出来的角色照片，带播放标记和数字 1；ref_1 上方是紫色的 aud，带扬声器图标、条目名 Yoda、波形和数字 1，每个下面都有一条“+ instr”。再往下是一行提示文字，以及点亮了 Ref2VA 的任务切换](docs/refmod_support_universal_writer.png)

![Prompt Rewriter Omni 的 ref_0 和 ref_1 接着同样两个 RefMod 输出：一个绿色的 vid 方块和一个名为 Yoda 的紫色 aud 方块，都标着 mod、编号为 1，上方是一行“drag to reorder - that order numbers the labels - click a square to switch it off - a clip's label: its sound”。下面是点亮了 REF2AV 的五个任务、按比例画出的八个宽高比矩形（选中 16:9），以及时长 10.0](docs/refmod_support_omni_rewriter.png)

之后：

- **它叫它文件里写的那个名字。** 片段或一组照片是 `<Video N>`，声音是 `<Audio N>`，不管
  标签上写着什么，因为生成节点就是这样叫它的。单张图片仍然可以标成主体，但不能标成片段；
  它也没有可以打开的自带声音。
- **它按自己的图片来描述，** 和任何参考素材一样。文件里关于它的内容作为提示跟在角色本身
  的问题后面：一组照片被描述成分开拍摄的同一个主体，而不是一串事件；保存的主体名字会给出
  以供使用；保存的外观或声音描述会被提供出来，在图片与之相符的地方照着写。没人解码过的
  声音，Universal Writer 会按它保存的描述来写（如果有的话）。Omni 适配器自己听声音，所以
  它会拒绝这样的声音，并提示接上 `audio_vae`。
- **Universal Writer 的写作模型会被告知文件里写的内容，** 写在参考素材块里、描述之前：
  一组照片是同一个主体的照片而不是一段画面，它是作为什么的参考保存的（声音参考只指声音本身，
  不包括它背后播放的东西）；如果一份外观和与它一起保存的声音都在素材条上，还会写明另一半在
  哪个标签下：`Audio 1: (saved as a reference for a voice, the voice alone and not what plays
  behind it; the voice of the subject in Video 1) ...`。描述模型是分开读这两者的，所以哪条
  描述都不会说它们属于一起；而叠在音乐上录的声音，否则会被当成音乐，被拿去写配乐。生成节点
  拿到的仍然是整段录音，连同音乐：提示词只是不再要求更多音乐。
- **交出去的是 RefMod 本身**，连同潜变量，放在 `references` 里，交给
  [MiniMax-H3 Reference Encode](#minimax-h3-reference-encode-参考素材编码)。

一组照片按每张照片分别解码，因为每张都是单独编码的；片段则整体解码。无论哪种，每次选择只
解码一次：ComfyUI 会保留节点的输出，图片也随之保留。

RefMod 在采样时并不免费。节点的说明文字会写出它的代价，而有些分享出来的很重：一组十张
1024x1824 的照片是 18,240 个 token，每个采样步都要带着，相当于十八张 1344x768 的参考
图片，或者一段两秒半的参考片段。降低 `strength` 改变的是遵循的程度，而不是代价。

### MiniMax-H3 Reference Encode 参考素材编码

`MiniMaxH3ReferenceToVideo` 会编码交给它的参考素材。RefMod 已经编码过了：生成节点需要
把它的潜变量作为单独的参考块，而 H3 的文本编码器仍然需要它的图片。所以这个节点做的是
[Reference Slots](#minimax-h3-reference-slots-参考素材插槽) 和
`MiniMaxH3ReferenceToVideo` 合起来做的事，对 RefMod 则做那件它俩都做不到的事。

```
Universal Writer ─ references ───────▶ references ┐
                 ─ rewritten_prompt ─▶ prompt     ├ MiniMax-H3 Reference Encode ─ positive ─▶ 采样器
clip, vae, audio_vae ────────────────────────────┘                              ─ latent   ─▶ 采样器
```

它的设置和 `MiniMaxH3ReferenceToVideo` 相同 —— `width`、`height`、`length`、
`ref_image_size` —— 再加上写作节点的 `references` 和来自 Reference Slots 的
`soundtracks` 开关。它输出 `positive`、用来采样的空 `latent`，以及一份 `summary`：
放进了什么、按什么顺序、代价多少。

**顺序就是全部约定。** H3 的 DiT 没有参考素材的索引：一个块靠它所在的位置对应到它的
标签。所以所有东西都按 `MiniMaxH3ReferenceToVideo` 的顺序放 —— 先是所有图片，然后是
所有片段（每个片段自带的声音紧挨在它前面），最后是所有声音 —— RefMod 占据写作节点给它
编的位置。图片、片段或声音用那个节点自己的辅助函数来准备，缩放和截断也一样，所以没有
RefMod 的素材条编码出来的条件和经过 Reference Slots 时相同。

**文本编码器看到的 RefMod 是什么样。** 图片原样给出。片段每秒两帧，和任何参考片段一样。
一组照片每四张取一张，每块两张 —— 这是保存这类图片的那个节点包的默认做法，也接近同一
潜变量作为片段播放时每秒两帧会给出的样子。保存了自身图片的文件给出那些图片；没保存的，
如果它的 RefMod 节点没接 VAE，就在这里用 `vae` 解码。

对普通参考素材，`vae` 和 `audio_vae` 沿用 `MiniMaxH3ReferenceToVideo` 的规则：没有它们，
这些素材只到达文本编码器，`summary` 会说有几个。RefMod 无论如何都会到达生成节点。

### MiniMax-H3 Prompt Presets 提示词预设

节点包里自带一千段写好的 MiniMax-H3 提示词，而这个节点把其中一段递出去。按
**Pick a preset**，浏览窗口就打开：二十种拍摄风格、二十类主体、画面的形状、镜头的
数量，以及有没有人说话，全都做成标签，把这一千段一层层收窄下来，上面还有一个直接搜
词句的搜索框。每一行都带着这段提示词当初为之而写的那个片段的一帧画面 —— 点它，片段
就播放起来。

它们是写好的 T2VA 提示词，格式正是这里的写作节点产出的那一种，同样那三个带标签的
字段 —— 这让它们在两个方向上都有用。把 `prompt` 接到生成器上，它就被原样使用；把它
接进某个写作节点自己的 `prompt`，它就是一次重写的起点。这个选择是一根线，而不是一个
小部件，这就是它自成一个节点的原因。

![MiniMax-H3 提示词预设节点，旁边一个 Show Any 节点。预设节点右侧从上到下八个输出，prompt 在最前，source 在最后。preset 小部件写着 000014；它下面是两个按钮，“Pick a preset” 和 “Save to the library”；再下面是那个片段的一帧画面 —— 一名摩托快递员行驶在被日光晒白的沙漠公路上 —— 旁边是完整的提示词，三个带标签的字段全都清晰可读。prompt 输出接到 Show Any 节点上，那边把同样这三个字段以纯文本打印出来。这次运行花了 0.030 秒](docs/node_prompt_presets.png)

*预设是摆在节点上的，而不是藏在一个对话框后面：它当初为之而写的那个片段的画面，以及
它的全部文字，在跑任何东西之前就已经在节点脸上了。运行本身花了三十毫秒，什么都没
加载 —— 从 `prompt` 出来的，就是节点一直在显示的那段东西。*

| 输出 | 是什么 |
|---|---|
| `prompt` | 整段东西，三个带标签的字段，和这里的写作节点会产出的一模一样。 |
| `integrated_multimodal_description`、`overall_soundscape`、`non_diegetic_music` | 同样这三段，分开来给。 |
| `seconds`、`width`、`height` | 这段提示词当初为之而写的那个片段。它们每一个都是五秒左右，而文字里的镜头时间就是照着这个写的。 |
| `source` | 它的编号、这段片段可以在哪两个地址上看到，以及功劳该记在谁头上 —— 工作流要送去别处时，可以把它接进一个文本预览。 |

**什么都不下载，安装时不下载，运行时也不下载。** 这些提示词、它们的标签，以及每段
各一帧 256 像素的画面，在节点包里一共 6 MB；它们是第一次有人要的时候才从硬盘读出来
解包的，所以一次从不打开浏览窗口的会话一分钱都不花，而跑一次只要个位数的毫秒。唯一
会碰到网络的是预览里的那段片段，而且只在有人点了某一帧的时候：先是 huggingface.co，
那边不应答时换 hf-mirror.com —— 后者正是在中国大陆应答的那个地址。没有网络时，那些
帧照样在，缺的只有视频。

**Save to the library** 会往你自己的某一套提示词里放一份副本，走的是提示词库用的
同一个编辑器 —— 名字、说明、分组，以及你打字时一直在跑的自检。从那一刻起，它就是
一条普通的记录：可以编辑，归在你自己的分组下，通过库窗口对每一个写作节点都可用。
副本会在自己的说明里写明它是从哪儿来的。自带的那条预设，这一切都碰不到。

有一句提醒值得说一次：这些是为一段五秒片段写的 T2VA 提示词。没有什么拦着你把其中
一段交给一个带参考素材的任务、或者一段长得多的视频，也不会有谁抱怨，但那段文字描述
的既不是前者也不是后者。

这些提示词，以及它们所描述的那些片段，是 ostris 的作品，经作者许可、并署名之后收在
这里 —— [ostris/minimax_h3_1k](https://huggingface.co/datasets/ostris/minimax_h3_1k)。
提示词是作者本人写的，实际上等同于 MIT；片段是 MiniMax-H3 的输出，而它的许可证禁止
拿它去训练模型，从中截下来的这些帧带着同一条限制。拍摄风格和主体的标签，是
[H3 Atlas](https://cohub.live/baize/video-altas/w/h3-atlas) 对这批素材的解读。这一千
段全都能通过[自检](#答案会被检查一遍)、且没有一条可报的意见 —— 这也算是从一个角度
说清了，一段好的 H3 提示词的格式究竟长什么样。

### 时长小部件

节点包里凡是写提示词或者读提示词的节点都有一个，而在这九个节点上它的含义完全一样：
目标片段打算做多长，以秒计。它是一个提示，不是一个设置 —— 它以一行文字
`duration: 10s` 的形式抵达模型，决定这次重写安排几个镜头、以及怎么给它们分配节奏。
片段最后到底有多长，是在下游、由采样它的那个东西定下来的。

**同一个插槽既收 `FLOAT` 也收 `INT`。** 小部件本身是浮点数，以十分之一秒为单位，
因为重写写出来的切点时间是精确到千分之一秒的 ——
`At 00:02.378, the camera cuts to` —— 而要 7.5 秒和要 8 秒一样平常。但大多数图里
带的长度是整数，所以这个插槽两种都收：一个 `INT` 原始节点、一个数帧数的节点，或者
从某个数学节点出来的 `FLOAT`，都能直接接上，中间不用转换器。两个 Universal 节点把
它画成滑块，其余的画成一个可以打字的数字框；无论哪一种，范围和菜单都一样。

**它的上限由你来挪。** 右键点节点 —— 在经典画布上，直接右键点这个小部件，那一项会
排在最前面 —— 就有一个 `duration` 子菜单，里面两项：

| 菜单项 | 它做什么 |
|---|---|
| **Default value** | 把小部件放回它一开始的那十秒。 |
| **Longest offered…** | 问你要一个数，并把它作为小部件的上限。它会被取整到十分之一秒，并保存在节点里，所以它跟着工作流走。 |

![经典画布上一个通用写作节点的右键菜单，标题是 MiniMaxH3UniversalWriter，`duration` 是它的第一项、高亮着，子菜单往右展开，里面两项 —— “Default value (10 s)” 和 “Longest offered (now 30 s)…”。高亮那一项下面是节点平常的菜单：Set、Get、两个灰掉的 rgthree 队列项、Run、Reload Node，以及 “Favorite Widget: duration”。菜单背后，节点显示着它的画面比例小方块，16:9 亮着，时长滑块读数是 10.0](docs/duration_widget.png)

*直接右键点这个小部件，`duration` 会排在菜单最上面，就像这里一样；右键点节点上的
别处，也能在靠下的位置够到同一个子菜单。子菜单里那两个数都是节点自己的，跟着工作流
一起保存。*

在你改它之前是三十秒，而服务端最多接受六百。这里是两个数而不是一个，因为一个小部件
的范围在节点被声明时就定死了，而没有哪一个范围适合所有的图：MiniMax 自己的指南是
围着几秒钟的片段写的，而社区搭出来的那些拉长了的流水线跑得远不止于此。所以服务端
收下任何讲得通的数，而小部件覆盖的是*你*实际在用的那一段 —— 一个能拉到十分钟的
滑块，是一个没法轻轻推到 9 的滑块。

那个数就是节点自己的 `max_duration` 属性，所以属性面板碰到的是同一个设置；而在这项
功能出现之前保存的工作流，直接拿到那个三十。

### 重复上一次的答案

每一个写作节点、重写节点和描述节点都有一个 `repeat_last` 开关。把它打开，节点就把
上一次产出的那个答案交回来，而不是再跑一遍模型 —— 手上还什么都没留着时它会跑一次，
把写出来的东西留住并把这件事说出来，此后只要开关还开着，不管你改了别的什么，它都
返回同样那段文字。它的意义正在于此：一次五十秒的重写，不该在你接它下游那一堆东西的
时候被付第二遍钱。

ComfyUI 自己的缓存做不到这件事。一个节点的缓存键是它的类、它的 `IS_CHANGED` 值，
以及它收到的每一个输入，所以改一下提示词，恰好就是把那条缓存扔掉 —— 而 `IS_CHANGED`
只能再添一份失效，从来遮不住失效。所以这些答案是存在本节点包自己的一个仓库里的，
一个节点一份。

那个仓库只活在这一次 ComfyUI 会话的内存里，别处哪儿都没有：不写硬盘，不跟着工作流
走，重启一次就空了。每一次运行都会在三个地方说明它的答案是从哪儿来的 —— 节点底下
那行说明、ComfyUI 的控制台，以及这个开关自己的工具提示（那里写着它是什么时候被留下
的、有多长，还有文字的开头）。当输入和当初产出它的那一套不再对得上时，这三处都会把
这件事说出来，而不是假装什么都没变。`bypass` 仍然压过它。

**而且留下来的东西可以编辑。** `Edit the last prompt` 会用提示词库那同一个编辑器
打开节点手上拿着的那个答案：提示词摆在一个你可以往里写的框里，一边打字，一边按当初
读这次运行时用的那套规则去读它。保存时会重新把字段从新文字里切出来，所以分节的那些
输出和这段文字保持同步；节点留下的、字段之外的一切属于那次运行而不属于这段文字，
因此保持原样。什么都不会写到硬盘上 —— 这是会话仓库，而一处值得留下的改动，事后是
要存进提示词库的。

下一次 `repeat_last` 要它交东西时，节点交出去的就是改过的那段文字。一次编辑不会改变
节点的任何输入，所以每个写作节点都把自己手上拿着的东西作为 `IS_CHANGED` 报出去，
和它报出自己被指向的那条库记录是一个道理；没有这一手，ComfyUI 会继续把编辑之前的
那个答案端上来。

**这处编辑到不到得了输出，是另一个问题**，而编辑器在你打字之前就把它回答了。一段
保存下来的提示词压过节点自己的答案，所以只要还选着一段，这处编辑就排在它后面等着
—— 要么把那个选择清掉，要么改去编辑那条记录。而 `repeat_last` 关着时，下一次运行
会直接写一个新答案盖掉这处编辑；在 greedy、输入又没变的情况下，那个新答案和旧的
逐字节一致，看上去和被忽略了一模一样 —— 所以在没有别的东西挡路时，保存会顺手把
`repeat_last` 打开，和选中一条库记录时的做法一样。

在两个描述节点上，被留下的是那段描述，而不是拼好的那个块：编号属于整条链，所以那
一行会围着 `previous` 上送来的东西重写一遍，而在这之后才接上来的素材会被真的描述
一遍。

默认情况下，这个开关交回来的是节点自己上一次的答案，而[提示词库](#提示词库)是你把
它指向别的东西的办法。无论哪一边都只有一个开关：`repeat_last` 决定一段留下来的提示词
到底交不交出去，库窗口决定交的是哪一段。

### 卸载留在内存里的模型

`keep_model_loaded` 会让写作模型在一次运行之后留在内存里。在你反复打磨提示词的时候，
这正是你要的；而等到文字已经满意、显卡要腾给视频的时候，它就成了挡路的东西。把这个
开关关掉，并不是把显卡要回来的省事办法：它是节点的一个输入，输入一变就是一次新的
运行，节点会先把同样那段文字再写一遍，然后才卸载。有三种办法可以不让模型再跑一遍就
把它放掉。

**`Unload model`** 是每个带这个开关的节点上的一个按钮。它把节点包手上拿着的东西立刻
全部卸载。如果在节点还在写的时候按下去，它会等这次运行结束，然后再卸载 —— 不管
`keep_model_loaded` 怎么说，因为这一下是后按的。

**ComfyUI 自己的 `Unload Models`** 也够得着节点包的模型。它们不是 ComfyUI 的模型，
ComfyUI 的模型管理并不认识它们，所以节点包把自己挂进了那个卸载动作。凡是要求 ComfyUI
卸载全部模型的东西也都一样：`/free` 接口、显存不足时的处理，以及别的节点包里的清理
节点 —— 在写作节点和采样器之间放一个，写作模型就会赶在视频模型加载之前离开。反过来
也成立：一个为自己清空显卡的采样器（有好几个节点包就是这么做的）会把留着的写作模型
连同其余的一起带走；而在 `--disable-smart-memory` 下，ComfyUI 每跑完一次工作流就卸载
一切，所以留着的模型只活到这一次运行结束，不会更久。

**`repeat_last` 开着而 `keep_model_loaded` 关着**，同样会卸载。一个把留下来的提示词
交回来的节点并不运行模型，但它照样会读这个开关：文字从节点自己的仓库里回来，模型被
卸载，什么都不会写两遍。

这三种办法放掉的都是节点包在 ComfyUI 进程里拿着的一切 —— 一个 Transformers 检查点，
或者一个跑在 `llama-cpp-python` 上的 GGUF —— 而不只是你按下按钮的那个节点的模型。由
llama.cpp 二进制程序运行的 GGUF 本来就没有被留下：模型随进程一起离开。

### 答案会被检查一遍

每一个新写出来的答案，在离开节点之前都会被拿回来、对着 MiniMax 自己那几份写作指南
的规则读一遍 —— 这就是自检，什么都没找到时它一声不吭。真找到东西时，它会在三个地方
说出来：ComfyUI 窗口里的一个弹出提示，带着最前面那几条发现 —— 因为从图的另一头看
过去，节点底下那行说明很容易被漏掉；说明本身，那里在答案上方放着完整的列表；以及
控制台，为那些没人盯着的运行准备的。它看的是：

- **镜头结构。** `[Shot 1]` 给描述起头，不带切点时间；后面的镜头按顺序编号，每一个
  都带一个 `At MM:SS.mmm` 时间戳，比前一个晚，并且落在要求的时长之内。
- **对白标记。** `<d>` 和 `</d>` 成对出现，而且每一个块都以它的 `[Language]` 标签
  开头。
- **参考素材标签，两个方向都查。** 一个当前任务扛不动的标签 —— 画面任务里的
  `<Video 1>`、T2VA 里的任何标签 —— 会被标出来；反过来也一样：一份到了节点面前却
  从未被引用的参考素材，生成器照样会收到它，却对它派什么用场没有任何发言权。看得见
  自己参考素材的那些节点，会把数量和实际接上的东西对一遍；纯文本的写作节点跳过这
  一半，而不是去猜。
- **全参考的账目。** `subject_definitions` 引入的每一个 `<Subject N>`，都欠
  `retention_analysis` 一行；而指南给 `detailed_description` 定的 350–500 词，没做到
  时会被记一笔。
- **对齐行。** I2VA、FL2VA 和 L2VA 以一句固定的话开头，告诉 H3 参考帧落在什么位置；
  它不在的时候，值得在渲染之前就知道，而不是让渲染去发现。
- **特效嵌入的 token。** 这里唯一一条不是 MiniMax 定的规则：一个大写的
  `Embedding:`、一个粘在前一个词上的 token，或者名字末尾的一个句号，都会让 ComfyUI
  无声地把这个 token 丢掉。它之所以在自检里，是因为这种失败和其余那些形状相同 ——
  输出里没有任何东西会提一句 —— 也因为[某个特效什么都没干](#minimax-h3-effect-embeddings-特效嵌入)
  的时候，你就是往这儿看。一个不在那十个之列的名字只记成提示，不算警告：那很可能是
  你自己的文本反演。

![一个标题为 “Self-check: MiniMax-H3 Universal Writer” 的警告弹窗，列着三条带感叹号的警告 —— 答案里缺了 5 个字段，点名 subject_definitions、summary、retention_analysis、overall_soundscape 和 non_diegetic_music，并建议把 temperature 调低或者换一个更大的写作模型；描述里没有 Shot 1，而镜头正是 H3 读结构的方式；Picture 1 和 Picture 2 接上了却从未被引用，模型照样收到它们，却对它们派什么用场没有发言权 —— 以及一条带短横的提示：detailed_description 有 1286 词，而指南建议 350–500 词](docs/self_check_alert.png)

*一次故意跑坏的运行，被回读了一遍。`!` 那几行是警告 —— H3 很可能会读错的东西；`-`
那一行是提示 —— 指南仅仅建议的东西。弹窗显示最前面几条发现；完整的列表待在节点底下
那行说明里，以及控制台里 —— 不管弹窗被给了多长时间，它在那儿都活得更久。*

拒绝是节点唯一真正强制执行的东西，而它也是用同样的方式说出来的：在异常抛上去之前，
那句话会作为弹出提示出现在 **Stopped** 底下，所以一次停下来的运行，不会看上去像是
无缘无故失败了。

发现只被说出来，从不被强制：答案原样发出，因为模型有时候把规则掰弯是对的，而这一次
是不是那种情况，只有你知道。警告（`!`）是 H3 很可能读错的东西；提示（`-`）是指南
仅仅建议的东西。从库里递出来的提示词不会被重新检查一遍 —— 它在被写出来的时候就检查
过了，那行说明留着的是那条记录的名字。

**说出多少**由选项节点上的 `self_check` 决定：`warnings and notes` 是全说；
`warnings only` 会把指南那些比较软的建议（比如 350–500 词）丢开，留下 H3 很可能会
读错的部分；`off` 什么都不说。读取本身永远都会做 —— 它是在已经躺在内存里的文本上跑
几个正则，不花什么钱 —— 所以这一项决定的是报出多少，不是看多少。一个读不懂的设置会
把所有东西都报出来：一个陈旧的工作流应该让检查比预期更响，绝不该让它变哑。

同一个弹窗也带着节点自己的警告，标题是 **Heads-up** 而不是 **Self-check**：一份当前
任务不会去读的参考素材、一段保存下来的提示词其参考素材已经和节点眼前看到的对不上、
一个描述模型对着接上的素材什么都没写回来。这些讲的是你的接线，而不是模型写的文字，
所以 `self_check` 管不着它们，它们永远都会被说出来。它们一直都在说明和控制台里；
如今它们也没法被漏掉了。

**其中一部分发生在运行之前。** 一个用不了眼前所接内容的任务，一直都是在任何权重挪动
之前就被拒绝的 —— 接了两张图片的 `I2VA`、接进画面任务的一段片段。如今素材条装的
东西超过 H3 收得下的量时，`Ref2VA` 也照这个方式拒绝：九张图片、三段片段、三段声音。
以前是先描述、事后再标出来，为了弄明白一件本可以事先数一数就知道的事，要付掉一整轮
描述外加一次重写。主体是和图片分开计数的，所以往往改一个标签就是全部的修法。

### 照着发现的问题动手

选项节点上的 `fix_once` 让节点可以对这些发现做点什么，而不只是把它们报出来。它
**默认是关的**，而关着是一个承诺：你拿到的答案就是模型写出来的那个，出门路上没有
任何东西被改过。

打开之后会发生两件事。**对齐行会被补回去**，在它缺席的地方 —— 那句话是固定文本，
节点在拼提示词的时候本来就会把它排好，所以一个把它丢掉的模型，并没有做出什么值得
尊重的判断。这一步不跑模型，也不可能把什么弄坏。然后，对那些机械性的发现 —— 超出
结尾的切点时间、不成对的 `<d>`、指向一份并不存在的参考素材的标签 ——
**写作模型会被再问一次**，那些发现会被折进提示词里，作为必须遵守的规则。

只一次，绝不成环。一个两次无视某条规则的模型，第三次照样会无视它，而每一次尝试都要
付掉一整次生成。有三件事保证这唯一一次重试是安全的：

- **在一个没救的答案上，它会被拒绝。** 一半字段都缺，意味着模型根本没端住这个格式，
  第二遍也改变不了这件事；节点会让你换一个更大的写作模型，而不是再花掉一分钟。
- **约束是走在提示词内部的**，而不是作为对话的第二轮。这保住了 LoRA 训练时用的那种
  单轮形状，所以重跑在一个训练过的适配器上，和在一个靠指南引导的模型上，表现是一样
  的。
- **第二个答案只有在更好的时候才被留下** —— 警告更少，或者警告一样多而总的发现更少。
  打平就留第一个。所以最坏的情况是花掉一分钟，绝不会换来一段更差的提示词。

它做了什么，都会在节点上说出来：补回了那一行、重跑了一次以及因此赚回多少、或者第一个
答案站住了。

### 提示词库

`repeat_last` 每个节点留一个答案，而 ComfyUI 一重启就把它忘掉。库是另外那一半：由你
命名并留下来的提示词，装在 ComfyUI 用户目录下的一个 JSON 文件里，对每一个工作流、
每一个节点都可用。

**保存。** 任何一个写作节点或重写节点上的 **Save the last prompt** 按钮，会打开一个
框，问你要一个名字、一段说明，以及任意多个分组 —— 文件里已经有的分组会以小标签的
形式给出来，而新建一个只需要写个词再按回车。被保存下来的是这次运行本身：文字、随后
那些分节、任务、模型、比例、时长、种子，以及节点见过的每一份参考素材的一张 50x50
缩略图。

那些缩略图是**在运行当中**取的，因为那是唯一取得到它们的时刻。等那个框打开的时候，
节点已经返回了字符串，张量早就没了，所以图片、片段的第一帧和那些度量，是在答案被
产出的同时抓下来、并跟着它一起走的。被量下来的都是真实存在的东西：一张图片有尺寸；
一段作为真正的 `VIDEO` 抵达的片段有帧数、秒数和帧率，而一段以图片批次形式抵达的
就只有帧数、别的什么都没有，因为一个批次没有容器可问；一段声音有时长、采样率和声道
数。参考素材是按位置标名的 —— `ref1-image`、`ref2-audio` —— 而不是按文件名，反正
节点也看不到文件名。

**使用。** **Prompt library** 按钮打开那份列表：先是这个节点自己上一次的答案，然后
是文件里的一切，新的在前。用那些小标签按分组过滤，或者在搜索框里打字，只留下名字、
说明、提示词、分组、参考素材或者设置里含有你所打内容的那些记录。**Use** 把节点指向
一条记录；**Copy** 只把提示词放进剪贴板，不把节点指向任何东西；**Edit** 打开它以供
修改；**Delete** 把它从文件里永久删掉，删之前会先问一句。这份列表也听键盘的：方向键
在列表里上下走，光标不用离开搜索框；回车用上亮着的那条记录，双击一张卡片也一样。
列表打开时会滚到这个节点当前指向的那条记录。

![提示词库窗口，标题 “Prompt library”，底下一行 “Hand a saved prompt straight to this node’s output. No model is loaded.”：顶上横着一个搜索框，写着 “Search names, descriptions, prompts, references”，旁边是文件下拉框，停在 `global`，再下面是四个分组小标签 —— bakery、dinosaur、fish、joke。最上面是这个节点自己的 Last Prompt 那一行，它的 Use 和 Copy 是灰的，底下写着 “Nothing kept yet. Run this node once and its answer is what gets saved.”。然后是两条保存下来的记录：“Дино и пеламида”，左边三张缩略图，上方一行 Ref2VA · 16:9 · 10s · 3 images + 1 audio · joke, dinosaur, fish，下面是它的说明和 subject_definitions 的开头；以及 “Дино”，一张缩略图和一行 I2VA · 16:9 · 15s · 1 image · joke, bakery, dinosaur，它的卡片被染成蓝色，因为这个节点当前指向的正是这条记录。每一行的右边都是 Use、Copy 和 Delete，而窗口底部是 Write a new one 和 Close](docs/prompt_library.png)

*卡片上的一切都能被搜到 —— 名字、说明、提示词、分组、设置和参考素材 —— 正是这一点
让一个装着一百条记录的文件仍然好用。名字底下那一行的结尾写着这条记录当初是为什么
而写的，`3 images + 1 audio`，用的是节点在警告“眼前所见已经对不上了”时用的同一套
说法；旁边的缩略图则说清那些图片究竟是哪几张。被染色的卡片是当前的选择。Last Prompt
那一行排在文件之上，因为它不在文件里 —— 它是这一次会话自己的答案，所以它没有
**Delete**。*

提示词库、编辑器、**Model list** 和内置提示词这几个窗口，都能像普通窗口一样改变
大小：拖边或者拖角，或者用右上角的按钮铺满整个浏览器标签页；每一种窗口各自记住
自己的尺寸。窗口底部写着文件里有多少条记录、一次搜索之后还剩多少条，以及上一次
编辑或删除做了什么。

被选中的提示词抵达 `rewritten_prompt` 和那些分节输出的方式，和一段新写的一模一样
—— 不加载模型，不生成任何东西，跑一次大约十分之一秒。由另一种节点写下的记录也能用：
类相同时，输出原样回来；否则那段文字会由正在读它的这个节点切成分节，和它切自己写出
来的答案是同一种切法。

**编辑。** **Edit** 会打开这条记录：它的名字、说明、分组，以及提示词本身 —— 摆在
一个你可以往里写的框里。*产出*这条记录的那些东西 —— 写作模型、设置、时长、参考素材
缩略图 —— 只给你看，不能编辑，也不会变。那一半是一次运行的账目，而一张把自己的来历
报错了的卡片，比没有卡片更糟。

提示词在你打字的同时被检查，用的是写作节点检查模型答案的同一套规则、出自同一个模块，
并且是对着这条记录自己的任务、时长和参考素材来检查的：一个已经漂到视频结尾之外的
切点时间，会在你还盯着它的时候就被点出来。这就是该在这里编辑、而不是去改那个 JSON
文件的理由 —— 那个文件什么都收。

Escape 和 **Cancel** 在丢掉任何已经打进去的内容之前都会先问一句 —— 这里如此，其他
要打字的窗口也一样：**Save the last prompt** 和 **Model list** 的条目表单。什么都
没打的时候，它们直接关掉。

![编辑窗口，标题 “Edit a saved prompt”，底下一行 “The prompt itself, and what the card says about it. What produced this record -- the writer, the settings, the references -- stays as it was.”：一个 Name 框写着 “Дино и пеламида (changed)”，下面是 Description，然后是四个分组小标签，joke、dinosaur 和 fish 亮着，bakery 没亮，一个空的 “New group, then Enter” 框，以及提示词本身 —— 摆在一个等宽字体的框里，正停在 subject_definitions 和 summary 上。它下面是自检 —— “Self-check: 2 warning(s), 1 note(s)” 之下两行琥珀色，一行说 &lt;Video 1&gt; 被引用了，可是没有任何视频到过这个节点，另一行说 &lt;Picture 2&gt; 和 &lt;Picture 3&gt; 接上了却从未被引用，还有一行灰色的说 detailed_description 只有 147 词，而指南建议 350-500 词。窗口底部是这条记录自己的账目，MiniMaxH3UniversalWriter · Ref2VA · 16:9 · 10s · 3 images + 1 audio · saved 29.08.2026, 15:07:10 · edited 01.09.2026, 11:07:04，底下是四张参考素材缩略图，角落里是 Cancel 和 Save changes](docs/prompt_library_edit.png)

*这些发现是从记录里读出来的，不是猜出来的：任务是 Ref2VA，时长十秒，参考素材就是
底下那四份 —— 它正是这么知道 `<Video 1>` 被写进了一段从来没有视频到过的提示词里，
以及接上来的图片里有两张从未被引用。琥珀色是警告，灰色是指南仅仅建议的提示。选项
节点上的 `self_check` 管的是节点在运行时说什么；编辑器永远两种都显示，因为你打开
这个框，本身就是在要它们。分割线以下的一切，是不可编辑的那一半。*

改动文字会把写作节点当初对那个答案做的字段切分丢掉，因为那次切分是照着文字原来的
样子做的。没有它，读这条记录的节点会自己去切那段文字 —— 而这正是它面对另一种节点
写出来的提示词时本来就在做的事 —— 所以分节的那些输出和你写下的东西保持同步。记录
的其余部分全都活着，而它的卡片会在保存时间旁边多出一个编辑时间。

一个已经指向这条记录的节点，会在它下一次运行时把新的文字递出去。ComfyUI 是从输入来
决定一个节点跑不跑的，而一次编辑一个输入都没改 —— 选中的仍然是同一个文件里的同一个
id —— 所以每个写作节点都把这条记录自身的内容作为 `IS_CHANGED` 报出去，于是这次运行
就发生了。缓存方面别的什么都没动：没有保存的提示词参与其中时，这个值是恒定的；而一条
只是被改了名、并没有被改写的记录，不会让每一个拿着它的节点重新跑一遍。

**一条带参考素材的记录，是被钉在那些素材上的。** 一段 T2VA 提示词是自足的，去哪儿
都行。而一段为某个画面任务、或者为 Ref2VA 写的提示词不是：它点了自己参考素材的名，
还描述了它们 —— `<Subject 1>` *就是*那只有着带纹理鳞片皮肤的蓝色恐龙 —— 所以给这段
提示词换一套素材，等于递给生成器一段关于并不在它眼前的东西的描述。卡片上的缩略图
是你看清一条记录当初为什么而写的办法，而节点也会去查：当它眼前所见的种类和数量，
和这条记录当初为之而写的那一套不再对得上时，它会在递出提示词之前，在节点上和控制台
里把这件事说出来。它是说出来，而不是拒绝 —— 把一段描述当模板复用是一件正当的事，
而你是不是这个意思，只有你知道。任何检查都抓不住的，是一张图片被换成了同一类的另一
张；缩略图就是为这件事准备的。

**一个开关决定递不递，窗口决定递哪一段。** 选中一条记录会把 `repeat_last` 打开，
因为把一段留下来的提示词递出去，靠的正是那个开关。把它关掉，节点就还给模型，而那个
选择原地等着；**Write a new one** 两件事都做，连那个选择也一并忘掉。按钮上写着当前
是哪一种 —— `Library: Storm at sea`，或者同样这句后面跟着 `(repeat_last is off)`
—— 而节点标题旁边的徽记写着 `REPEAT` 或者 `LIBRARY`。

这个选择本身住在一个隐藏的小部件里，所以它跟着工作流一起保存，也能抵达一次 API
运行：明天重新打开这张图，返回的还是同一段提示词。要是那条记录在这期间被删掉了，
运行会停下来并把这件事说出来，而不是悄悄写一段新的 —— 这张图要的是某一段特定的
提示词。

**成套。** 选项节点上的 `prompt_file` 指定接在它上面的那些节点往哪个文件里保存、
又列出哪个文件的内容，你不另说的话就是 `global`，而它旁边的 **New prompt file**
可以再建一个。一个文件就是一套工作集，这是把一个项目的提示词和别的一切分开的最省事
的办法。它们住在

```text
ComfyUI/user/minimax_h3_rewriter/prompts/
```

就是普通的 JSON，所以它们可以手工编辑、在机器之间复制，或者删掉。一条带一张缩略图和
一段完整长度提示词的记录，大约 11 KB。

### 写作指南是取下来的，不是随包附带的

那两份指南没有装在这个节点包里面。它们是 MiniMax 的文档，而 MiniMax H3 社区许可证
只在它的“适用地域”之内授予再分发这些材料的权利 —— 全球范围，但*不包括*欧盟、英国、
大韩民国和美国 —— 而且必须随附一份协议副本和一个 NOTICE 文件。一个挂在注册表上的
节点包没有办法遵守地域边界，所以每一次安装都在第一次使用时直接从 MiniMax 取自己的
那一份：16 KB 和 24 KB，取一次，归在和其他东西同一个 `auto_download` 开关底下。

它们落在

```text
ComfyUI/user/minimax_h3_rewriter/guides/
```

**Open guide folder** 按钮打开的就是这里。改动一份副本就改动了系统提示词，而取回操作
从不覆盖已经在那儿的文件 —— 把指南裁短，是让它装进一个小模型上下文的最省事的办法。

### 模型列表

节点只有一个模型字段时，点击 **Model list** 打开模型目录。有多个模型字段时，
点击对应字段的按钮。通过 **Add a model**、**Edit** 和 **Delete** 管理目录条目，
用 **Check it** 在下载模型前检查条目。目录保存在

```text
ComfyUI/user/minimax_h3_rewriter/models.json
```

编辑器和节点下拉列表会显示 **On disk**（已在磁盘上）、**Download needed**（需要下载）
或 **Files missing**（缺少文件）。多模态 GGUF 需要同时找到模型和对应的 `mmproj`，
才会显示 **On disk**。在 **Model list** 中可以搜索模型，或用 **On disk only** 只显示
本地模型。点击星标可将模型加入 **Favorites**，并在对应的节点下拉列表中优先显示。

点击 **Use this** 可在打开列表的节点中选用模型。节点有多个模型字段时，
**Captioner models** 和 **Writer models** 等按钮会打开对应的标签页，
**Use in writer_model** 会将模型应用到指定字段。目录标签显示模型所在位置或下载目标，
例如 `LLM/quant`。

开启 **Search nested folders** 可搜索已配置模型目录下的所有子目录。
该设置和收藏都保存在 ComfyUI 服务器上。复制模型后可点击 **Rescan folders** 刷新列表。

此文件在首次使用时由随包附带的副本生成，更新节点包会保留你的改动。
窗口底部的 **Open models.json** 仍然会用你桌面上的 JSON 编辑器打开那个文件 ——
是在跑 ComfyUI 的那台机器上，而那不一定就是正看着浏览器标签页的这一台 —— 用来做
窗口有意不碰的那两件事：`adapters` 那几节，以及指向网络共享的路径。

窗口只显示打开它的那个节点真正会读的那几份列表，因为几个适配器吃的是不同的架构，
一份列表里的条目在另一个节点上根本加载不起来。所以通用重写器打开时是三个标签页，
而描述节点只有一个。展开 **Model requirements** 可查看架构、模型尺寸和投影器要求。
**Found in model folders** 显示在 ComfyUI 模型目录和 Ollama 仓库中找到的其他模型，
这些模型也会出现在节点的下拉列表中。如果目录条目指向已找到的文件，两个列表都用
该条目的名称和 **On disk** 标记来显示这个文件。

![Model list 窗口的 Guided writers 标签页，已勾选 Search nested folders。已安装模型显示 On disk 标记、相对目录、Use in writer_model 按钮和收藏星标。](docs/model_list_dialog.png)

窗口底部数着当前这份列表里的条目，并说出上一次保存、删除或恢复做了什么 —— 包括
这张图里有多少个节点换到了条目的新标签上，或者还握着一个已经删掉的条目的名字。
这些节点会留着那个名字，好让运行时直接说这个条目没了，而不是替它们挑另一个模型。

**Check it** 会把在不挪动任何权重的前提下能回答的都回答掉。已经在这台机器上的文件会
被直接读出来，所以它报得出架构和形状、模型带没带聊天模板，也说得出那个投影器带没带
视觉和音频：

```text
- 'Qwen2.5-Omni-7B-Q4_K_M.gguf' is a 'qwen2vl' model, 28 blocks of width 3584. That fits.
- It carries its own chat template.
- 'mmproj-Qwen2.5-Omni-7B-Q8_0.gguf' carries the vision and audio encoder.
```

聊天模板比看上去要紧。27B 重写器和各个写作节点是按模型自己的模板来拼提示词的，所以
一个不带模板的 GGUF —— 通常是某个模型的基础版转换，而不是 instruct 版 —— 看上去是个
完全正常的模型，却会在第一次运行时就停下。在这些地方 Check it 会拒绝它，并且拿一条
系统消息和一条用户消息把模板渲染一遍，所以本节点包渲染不了的模板也一样会被拒绝。
描述节点以及 8B 和 Omni 重写器把模板交给 `llama-mtmd-cli`，没有模板时它会退回
ChatML，所以在那边缺了模板只是一条警告。

只存在于 Hugging Face 上的东西，则被问它的元数据能说出些什么：一个 transformers
仓库是照着它那 4 KB 的 `config.json` 来判断的，而一个 GGUF 仓库会被问一句你写的
那些文件在不在里面 —— 正是这一下抓住了那个拼写错误，否则它要等到一次下载跑了几分钟
之后才浮出水面；顺手还替你把 `download_gb` 填上。GGUF 的形状和聊天模板都在文件本身
里，所以对一个还在 Hub 上的文件，这两样要等到第一次运行 —— 或者等文件下好之后再点一次
Check it。

把 `file` 留空，**Check it** 回答的就是另一个问题：这个仓库里有什么？它把仓库里的
GGUF 文件列成一排按钮，从小到大。点一下，文件就填进表单；如果你还没有自己打名字，
还会按内置条目的写法填上一个 —— `Qwen2.5-Omni-7B Q4_K_M`。需要成对的列表会把投影器
单独列成一排，给 `mmproj` 用；仓库里只有一个投影器时它会自己填上，两半都齐了，检查
就会自动跑一遍，把下载大小加起来。拆成几部分的模型（`-00001-of-00003.gguf`）会被数
出来，但不列出：一个条目只写一个文件，也只下载那一个文件。`repo` 里写的若是这台
机器上的一个文件夹，也照样列出来；写了一个仓库里没有的文件名，同样会出现这些按钮。

有两件事这个窗口不做。它拒绝 `repo`、`file` 或者 `mmproj` 里的**网络路径**：这个
窗口是通过 ComfyUI 的 API 够得着的，而那个 API 没有 CSRF 令牌，还常常挂在
`--listen` 上；而光是去看一眼一个 UNC 路径，就已经是在对它点名的那台主机发起一次
身份验证了。手工敲进 `models.json` 的路径仍然不受这条限制 —— 那个文件不会跟着走。
还有，只要这个文件解析不了，它就干脆拒绝写入，因为覆盖着保存会拿随包附带的那份列表
把你自己的条目替掉；它会把解析错误显示出来，每一个会写文件的按钮都变成死的，只剩下
**Open models.json**。它在那种状态下列出来的是随包附带的那一份，因为在文件重新能被
解析之前，下拉框提供的也正是那一份。

改一个条目的下载大小、显存说明或者备注，会改变下拉框里读到的东西，而保存过的工作流
记住的正是那串字 —— 但条目是按名字找的，所以存着旧标签的工作流照样能运行它，并且在
被打开时迁到新标签上。改名则是另一回事，因为名字正是用来找它的：表单在提交之前会把
这件事说明白，你眼下开着的这张图会被替你迁过去，而别的工作流会丢掉这个选择。

删除一个条目时，用着它的节点保持原样：运行会停下并说明这个条目已经没了，而不是换一个
别的模型、开始一次谁也没要求过的下载。唯一的例外是文件已经在本机的条目 —— 它的
`on disk:` 那一行会回到下拉框里，节点会迁到那一行上。

这个文件里有五份列表，字段都一样。**`models`** 喂给 LoRA 重写器，必须是
Qwen3.6-27B：

```json
{
  "name": "Qwen3.6-27B FP8",
  "repo": "Qwen/Qwen3.6-27B-FP8",
  "download_gb": 28.8,
  "vram": "~29 GB, no extra package needed"
}
```

**`models_8b`** 喂给 8B 重写器，必须是 Qwen3-VL-8B-Instruct，而且模型旁边要有一个
`mmproj` —— 既然是多模态的，它就是两个文件：

```json
{
  "name": "Qwen3-VL-8B-Instruct GGUF Q4_K_M",
  "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
  "file": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
  "mmproj": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
  "format": "gguf",
  "download_gb": 5.4,
  "vram": "~9 GB with the adapter"
}
```

**`models_omni`** 喂给 Omni 重写器，必须是 Qwen2.5-Omni-7B，收的也是同样那两个文件
—— 区别在于它的投影器必须带着一个音频编码器，否则适配器就挂到了一个听不见的模型
上：

```json
{
  "name": "Qwen2.5-Omni-7B GGUF Q4_K_M",
  "repo": "ggml-org/Qwen2.5-Omni-7B-GGUF",
  "file": "Qwen2.5-Omni-7B-Q4_K_M.gguf",
  "mmproj": "mmproj-Qwen2.5-Omni-7B-Q8_0.gguf",
  "format": "gguf",
  "download_gb": 6.2,
  "vram": "~9 GB with the adapter and a 12k context"
}
```

这里是三份列表，而不是往 `models` 里多塞几条，因为它们里面任何一条到另外两个节点上
都加载不起来 —— 架构不同，适配器也不同。三个适配器分开住也是同一个理由，各自在
`adapters`、`adapters_8b` 和 `adapters_omni` 底下。

**`writers`** 喂给写作节点，什么都行，只要它是一个带聊天模板的 GGUF 语言模型：

```json
{
  "name": "Qwen3.5-4B",
  "repo": "unsloth/Qwen3.5-4B-GGUF",
  "file": "Qwen3.5-4B-Q4_K_M.gguf",
  "format": "gguf",
  "download_gb": 2.6,
  "vram": "~5 GB with the guide in context"
}
```

**`captioners`** 喂给参考素材描述节点，需要多一个字段 `mmproj` —— 一个多模态模型是
两个文件，而且两个都出自同一次转换：

```json
{
  "name": "Qwen2.5-Omni-3B",
  "repo": "ggml-org/Qwen2.5-Omni-3B-GGUF",
  "file": "Qwen2.5-Omni-3B-Q4_K_M.gguf",
  "mmproj": "mmproj-Qwen2.5-Omni-3B-Q8_0.gguf",
  "format": "gguf",
  "download_gb": 3.4
}
```

#### 把一个条目指向你已经有的文件

`repo` 不一定非得是一个 Hugging Face 的 id。给它**这台机器上的一个文件夹**，文件就
直接从那儿读出来，什么都不下载，也什么都不复制：

```json
{
  "name": "Gemma 4 26B",
  "repo": "X:/models/gemma-4-26B-A4B-it-UD-Q8_K_XL",
  "file": "gemma-4-26B.gguf",
  "format": "gguf"
}
```

或者把整条路径写进 `file`，把 `repo` 略掉 —— 两种写法都行，三节里都行，因为这两种
都是人们实际会打出来的样子。有两点要留神：

- **反斜杠在 JSON 里必须写成双份**，或者干脆写成正斜杠。`"X:\Programs\..."` 根本
  就不是合法的 JSON —— `\P` 不是一个转义 —— 而一个坏字符会让*整个文件*读不了，不
  只是那一个条目。Windows 到处都接受 `X:/Programs/...`，所以那是更省心的习惯。
- 一条并不存在的路径会**被报出来，而不是绕开它去下载**。节点会点名它没找到的那个
  文件，然后停下。

要是你在别处存着一整个文件夹的 GGUF，一条一条加条目是绕远路：在 ComfyUI 的
`extra_model_paths.yaml` 里用 `LLM` 这个键指向它，里面的每一个文件都会被自动提供
出来，名字前面带着 `on disk:`。如果那个文件夹排在最前面，新的下载就会进到它里面；
而已经在任何一个 `LLM` 文件夹里的模型，会在原地被找到，不会再下载一遍。

加一个条目，**刷新浏览器标签页** —— ComfyUI 不必重启 —— 它就在下拉框里了。让
`name` 保持稳定：保存过的工作流正是靠它找到条目的。一个存下来的选择已经不见了的
节点，会在任何东西运行之前停下，指名道姓地把这件事说出来，而不是默默换一个别的。

#### 当这份列表本身坏掉的时候

`models.json` 里的一个语法错误，从前是这个节点包里最安静的一种失败：解析抛出异常，
随包附带的默认值被端上来，而下拉框看上去平平无奇。一次根本没生效的编辑，和一次什么
都没改变的编辑，是分不出来的。

如今每一个模型下拉框的第一条就会把这件事说出来，带着行号和列号：

```text
!! models.json is not valid JSON — Invalid \escape: line 3 column 30 (char 46) — showing the packaged list instead
```

列表的其余部分还在，ComfyUI 也照跑；选中那第一条再按运行，它会把这句话再说一遍，
并点名该修的是哪个文件。修好、刷新标签页，它就消失了。

#### 更新会加进新条目，而不碰你自己的

你那一份永远不会被覆盖，但在你安装之后才加进节点包的模型**会被并进来** —— 否则
“我们不碰你的列表”就悄悄变成了“你永远看不到任何新东西”，而且哪儿都没有一句话说明
节点其实知道得更多。

这条规则是集合运算，不是版本检查。在那几份列表旁边，你的文件里记着 `seed_offered`：
随包附带的那份列表曾经摆到这次安装面前的每一个名字。于是一次更新加进来的是

```text
节点包里的名字  −  你文件里的名字  −  已经摆到你面前过的名字
```

于是被你删掉的条目就一直是删掉的，被你改了名的不会重复出现，而真正新的那个会到来。
从 `seed_offered` 里删掉一个名字，下次启动时那个条目就会被重新提供出来；而模型列表
窗口里的 **Restore the packaged entries**，就是这同一处编辑做成的一个按钮：它把某
一份列表的那条记录扔掉，于是下一次读取会把这份列表缺掉的每一个随包条目都带回来。
你自己写的条目，它一概不碰。

有一次例外，只有一次：在这套机制出现之前写下的文件，没有关于它被提供过什么的记录，
所以第一次更新时，缺掉的一切都会回来 —— 包括你手工删掉过的那些。原来的文件会以
`models.json.bak` 的名字留在它旁边，而从那以后，你的删除就作数了。那份备份只记录这
一次合并，别的什么都不记：你在模型列表窗口里做的编辑不会把它花掉，否则两下点击就
足以弄丢它本来要保住的那一份副本。

合并会按名字记进 ComfyUI 的控制台。一个节点解析不了的文件会被原封不动地留着，那一次
会话用随包附带的列表 —— 所以一个多余的逗号让你付出的是一次重启，不是你的改动。

### 你为 Ollama 拉下来的那些模型

如果你在跑 Ollama，那这个节点包想要的写作模型和描述模型早就在你硬盘上了，把同一个
量化再下一遍未免可笑。它们会出现在 `writer_model` 和 `caption_model` 下拉框里，
前缀是 `ollama:`，名字就是 `ollama list` 给它们起的那个：

```text
ollama: qwen3:8b [qwen3, 4.7 GB]
ollama: moondream:latest [+mmproj, vision, 1.7 GB]
```

什么都不复制、不转换、不下载，而且 Ollama 本身不必在跑：它拉下来的就是一个普通的
GGUF，llama.cpp 就地把它读出来。这里用的是硬盘上的那个文件，不是 Ollama 的 API
—— 那个服务可以停掉、禁用，甚至卸载，模型照样能用。

一个多模态模型会同时出现在两份列表里：作为写作模型时不带它的投影器，作为描述模型时
带着。这一对是从模型自己的 manifest 里得来的 —— 那两个文件在里面是被写在一起的 ——
所以它是确定的，而在一个文件夹里比对文件名做不到这一点。

会去看三个地方，而 Ollama 的布局在每个平台上都一样：

```text
$OLLAMA_MODELS                     # 如果你设了它
~/.ollama/models                   # 通常的安装位置，Windows 也在这里
/usr/share/ollama/.ollama/models   # Linux，作为服务安装时
```

存在别处的仓库 —— 在 WSL 里、在一个容器里、在一块服务器毫不知情的盘上 —— 要手工
点名，写在 `models.json` 里

```json
"ollama_stores": ["\\\\wsl$\\Ubuntu\\usr\\share\\ollama\\.ollama\\models"]
```

或者写在 `MINIMAX_H3_OLLAMA_MODELS` 里 —— 它收多条路径，分隔方式和 `PATH` 一样。
这些位置不被自动搜索是有意的：去够 `\\wsl$\...` 会**把一个停着的 WSL 发行版启动
起来**，而这些列表在每一次填下拉框的时候都要重建一遍。打开一个 ComfyUI 标签页，
不该顺手启动一台虚拟机。

`ollama rm` 把一个模型从下拉框里拿掉，和删掉任何别的文件是一样的。你的工作流保存的
是标签而不是路径，所以把同一个 tag 重新拉一遍，它照样能用 —— 那个仓库是按内容寻址
的，同样那些文件会落回原处。不过标签里带着大小，而不同的量化就是不同的大小，所以用
另一个量化重新拉一遍 `qwen3:8b`，回来时会变成一个你得重新挑一次的选项。

## 权重都落在哪儿

```text
ComfyUI/models/LLM/
├── Qwen3.6-27B/                          # 底座模型，约 52 GB
├── MiniMax-H3-Prompt-Rewriter-LoRA/      # 适配器，约 3.5 GB
├── Qwen3.5-9B-Q4_K_M.gguf                # 一个写作模型，一个文件
└── Qwen2.5-Omni-3B/                      # 一个描述模型：模型和 mmproj 在一起

ComfyUI/user/minimax_h3_rewriter/
├── models.json                           # 你的模型列表
├── guides/                               # 那两份写作指南，40 KB
└── runtime/                              # llama.cpp 的二进制，如果取过的话
```

**已经把 LoRA 下下来了？** 直接把 `adapter` 小部件指向那个文件夹（绝对路径是收的），
就什么都不会再取一遍。

## 用你已经有的模型

52 GB 是一笔不小的要求，所以节点会先去找底座模型，然后才谈下载。`model` 下拉框会列出
在下面这些地方找到的每一个 Qwen3.6-27B：

- ComfyUI 为 `LLM`、`transformers`、`diffusers`、`text_encoders` 和 `clip` 登记的
  所有目录 —— 包括通过 `extra_model_paths.yaml` 映射进来的任何东西，以及
- Hugging Face 的缓存（`HF_HOME` / `HF_HUB_CACHE` / `~/.cache/huggingface/hub`），

于是早先被任何别的工具拉下来的副本会被复用，而不是下第二遍。只有权重确实在场的目录
才会被列出来；一个除了 `config.json` 什么都没有的缓存条目不会被提供。

### 更小的重打包

你放进模型列表的任何一个仓库，都会在一个权重都还没挪动**之前**，先照着 4 KB 的
`config.json` 检查一遍。每一种重打包保留的都是同一个指纹（`model_type qwen3_5`、
hidden 5120、64 层、vocab 248320），所以真正决定结果的是量化的运行时：

| 构建 | 下载 | 运行时要的包 | LoRA 挂得上吗？ | 结论 |
|---|---|---|---|---|
| `quant_method: bitsandbytes`（nf4 重打包） | **约 17 GB** | `bitsandbytes`，本来就是必需的 | 能 | **开箱就能用 —— 低资源那条路** |
| `Qwen/Qwen3.6-27B`（bf16） | 52 GB | 不用；要 `nf4`/`int8` 时用 `bitsandbytes` | 能 | **开箱就能用** |
| `Qwen/Qwen3.6-27B-FP8` | 29 GB | `kernels`，外加一个从 Hub 上取的内核 | 能 | 仅限进阶；需要超过 29 GB 显存 |
| `quant_method: awq` | 约 19 GB | `autoawq` | 能 | 装上它之后可用 |
| `quant_method: gptq` | 约 19 GB | `gptqmodel` | 能 | 装上它之后可用 |
| `quant_method: compressed-tensors` | 约 19 GB | `compressed-tensors` | **不能** | 不支持 |
| `quant_method: modelopt`（NVFP4） | 约 20 GB | `nvidia-modelopt` | **不能** | 不支持 |

PEFT 为 bitsandbytes、AWQ、GPTQ、HQQ、EETQ、AQLM 和 torchao 这些层各自带了 LoRA 的
分派器，而普通的 `nn.Linear` 覆盖了 bf16/fp16 和 FP8。compressed-tensors 和
modelopt 没有分派器，所以适配器根本就挂不上去 —— 装什么包都修不好这件事。

> 一个仓库的**名字**不代表它的格式：`cyankiwi/Qwen3.6-27B-AWQ-INT4` 和
> `unsloth/Qwen3.6-27B-NVFP4` 两个都是 `compressed-tensors`，所以两个都吃不下这个
> LoRA。节点会在下载任何东西之前，报出真正的 `quant_method` 和确切的理由。

### 如果节点说缺了某个包

推荐的那几条路上没有哪一条需要它。bf16、nf4 重打包，以及 `nf4`/`int8` 这两个选项，
跑的都是本节点包已经声明过的包，所以“缺包”这句话只会为你自己特地去找来的某个构建
而出现。

它真出现的时候，会点名那个包，**以及给真正在跑 ComfyUI 的那个解释器用的命令** ——
在一个普通终端里敲 `pip install kernels`，装进去的是 PATH 上的那个 Python，而在
便携版安装里，那从来都不是对的那一个，于是节点会继续拒绝：

```
This base model cannot run the prompt-rewriter LoRA.
  - the 'fp8' checkpoint needs the 'kernels' package, which is not installed in
    this Python environment. Install it with:
      "…\python_embeded\python.exe" -m pip install kernels
    Note: installing it is only half of it: the FP8 matmul is a Triton kernel
    that transformers then downloads from 'kernels-community/finegrained-fp8'
    on the first generation, and that needs a build matching this torch and
    CUDA version.
```

照着打印出来的那一行跑，然后重启 ComfyUI。这个节点包不会自作主张往你的环境里装任何
东西，将来也不会：一个会偷偷 pip 安装的节点，是一个能在你盯着进度条的时候，把
ComfyUI 某个毫不相干的部分弄坏的节点。

## 什么都不用另装的最小下载量

Qwen3.6-27B 的一个 `bitsandbytes` nf4 重打包是**约 17 GB 的下载、约 16 GB 的显存**，
而且它要的东西，这个节点本来就都要。这就是该指给别人看的那条路：和官方检查点走同一
条代码路径，不添新依赖，下载量只有三分之一。

那些重打包是第三方的 —— 节点在取任何东西之前会照着 `config.json` 核验架构，那证明
的是形状对，而不是上传的人可信。这一点由你自己判断，或者从官方权重出发，自己做一次
重打包。

## GGUF —— 还要更小，而且什么都不用装

从模型列表里挑一个 `[gguf]` 条目，重写器就跑在 llama.cpp 底下，而不是 Transformers。
**这不牵涉任何 pip 安装。** 如果 ComfyUI 的环境里恰好有 `llama-cpp-python`，节点就
用它；没有的话，它就去跑这台机器上已经有的 llama.cpp；再没有，就把官方二进制取下来
（约 34 MB，或者在 `llama_backend` 解析到 CUDA 时约 511 MB —— 见下表）放进
`ComfyUI/user/minimax_h3_rewriter/runtime/`，并把 `llama-completion` 作为子进程跑
起来。和权重共用同一个下载开关：`auto_download`。

子进程每跑一次就重新加载一次模型，而这在实际中不花什么代价 —— 节点的默认值就是
`keep_model_loaded = False`，因为重写一结束，那块卡就要拿去生成视频了，而进程内的
那个后端每跑完一次也照样把模型卸掉。二进制后端真正做不到的，是遵守
`keep_model_loaded = True`。它还白送两样东西：显存是由操作系统还回来的，而不是由某个
释放器；以及 llama.cpp 崩溃时带走的是一个子进程，而不是 ComfyUI 和它的队列。

选项节点里有两个选项管这件事，而它们回答的是不同的问题。**`gguf_runtime`** 挑的是
*由谁来跑这个模型*：

| `gguf_runtime` | 含义 |
|---|---|
| `auto` | llama-cpp-python 能导入就用它，否则用二进制 |
| `llama-cpp-python` | 强制用 wheel；它不在时会带着一句清楚的话失败，而不是悄悄换成别的东西 |
| `llama.cpp` | 强制用二进制，哪怕装着一个 wheel —— 装着的那个 wheel 坏掉时，这就是出路 |

只有 `llama-cpp-python` 能遵守 `keep_model_loaded`；子进程一退出，二进制就把模型
交还给操作系统了。

而 **`llama_backend`** 挑的是*去取官方的哪一个构建*，只有在用二进制时才起作用：

| `llama_backend` | 下载 | 说明 |
|---|---|---|
| `auto` | 34 或 511 MB | **这台机器跑得动的最快那个构建，而不是最小的那个。** 在 Windows 上、显卡是算力 8.6、8.9、12.0 或 12.1 的 NVIDIA 时用 CUDA —— 那个压缩包恰好只带这几个的原生 SASS，没有可以回退的 PTX —— 其余一律 Vulkan |
| `vulkan` | 34 MB | NVIDIA、AMD 和 Intel 一视同仁；吞吐大约是 CUDA 的一半。在一块 `auto` 会送去 CUDA 的卡上想把下载量压小，就明确挑它 |
| `cuda` | 511 MB | NVIDIA 上快约 2 倍；**仅限 Windows** —— 上游不发布 Linux 的 CUDA 构建，所以在 Linux 上你自己编一个，节点去跑它 |
| `cpu` | 17 MB | 完全没有 GPU |

**你已经有的 llama.cpp 会被原样拿去跑。** 在取任何东西之前，节点会按顺序看四个地方：
`MINIMAX_H3_LLAMA_BIN` 里的路径、写在 `ComfyUI/user/minimax_h3_rewriter/llama_bin.txt`
里的路径、它**为当前所用的那个后端**已经解包好的那一份，然后是 `PATH`。所以你自己
编译出来的构建根本不需要任何设置 —— 把它的 `build/bin` 放到 `PATH` 上，或者干脆
点它的名：

```sh
export MINIMAX_H3_LLAMA_BIN=/opt/llama.cpp/build/bin   # 文件夹，或者里面那个可执行文件
```

**在服务器上，宁可用那个文件。** 一次 export 只够得着从同一个 shell 里启动起来的
服务，别的都够不着 —— 一个 systemd 单元、一个容器入口点或者一个启动脚本，交给进程
的是它自己的一套环境，而且永远不会去读你的 `~/.bashrc`。`llama_bin.txt` 里的一行是
节点自己去读的，它不在乎 ComfyUI 是谁启动的：

```sh
echo /opt/llama.cpp/build/bin > ~/comfy/ComfyUI/user/minimax_h3_rewriter/llama_bin.txt
```

这时 `llama_backend` 就不再要紧了：它挑的是下载哪一个压缩包，而不是一个已经存在的
二进制当初照着什么编出来。在 Linux 上弄到一个 CUDA 版 llama.cpp，走的就是这条路
—— 用 `-DGGML_CUDA=ON` 编一次，在这儿点它的名，`device = cuda:0` 就名副其实。描述
节点会在它旁边找 `llama-mtmd-cli`；一个没有编出那个目标的构建，会让节点为那一份活
回头去取压缩包。

**解包出来的那一份是按后端分开的**，在 `runtime/<release>-<backend>` 底下，所以一个
已经躺在 `runtime/b10310-vulkan` 里的压缩包，答不了一次解析到 CUDA 的运行 —— 节点
会把那一个也取下来。因此改 `llama_backend` 就是第二次下载，而手工解包一个构建，意思
是把它放进以你实际要跑的那个后端命名的文件夹里。

一条指向虚空的路径是一个错误，而不是一次悄无声息的下载，所以一个拼写错误会被说出来，
而不是花掉半个 GB。而当哪儿都找不到时，那句拒绝会把 ComfyUI **进程**手上实际拿着的
东西打印出来 —— 那个变量、那个文件、那个文件夹以及它的 `PATH` —— 而不是把你也许早就
在某个它从没见过的 shell 里照做过的建议再念一遍。

> **为什么不用 `llama-cpp-python` 的 CUDA wheel。** 现有那两个在普通消费级硬件上都
> 会挂，而且挂法是两个互不相干的：
>
> | wheel | 编译标志 | 会发生什么 |
> |---|---|---|
> | `v0.3.34-cu130` | `AVX512 = 1`、`ARCHS = 750..900` | 权重加载得进去，然后 `llama_init_from_model` 以 `0xC000001D` 死掉 —— 消费级的 Intel 12 到 14 代芯片没有一个有 AVX-512 |
> | `v0.3.34-cu132` | `AVX512 = 0`、`ARCHS = 750..900` | 走到第一个内核，然后是 `the provided PTX was compiled with an unsupported toolchain` —— 列表里没有 `sm_120`，所以一块 RTX 50 系的卡会退回去做 JIT，而比这个构建的工具包更旧的驱动会拒绝 |
> | `v0.3.34-vulkan` | `AVX512 = 0`，没有架构列表 | 能用，而且驱动提供 `NV_coopmat2` 时它会用上 |
>
> 官方的发布压缩包这两个问题都没有。它们带着 **14** 个 CPU 后端变体，在运行时挑
> 一个 —— 这就是为什么在 cu130 那个 wheel 会死掉的机器上，同一个模型在 `llama-cli`
> 底下跑得好好的。而它们的 CUDA 压缩包带的是原生 SASS，完全没有 PTX ——
> `cuobjdump --list-elf` 报出来的是 `sm_86 sm_89 sm_120a sm_121a` —— 所以驱动从来
> 不会被要求去编译什么。
>
> 要是你无论如何都想要进程内的那个后端，能用的是 Vulkan 那个 wheel：
> `pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-vulkan/llama_cpp_python-0.3.34-py3-none-win_amd64.whl`

| 底座量化 | 下载 | 带适配器的显存 |
|---|---|---|
| `Q4_K_M` | 15.7 GB | 约 19 GB |
| `IQ4_XS` | 14.4 GB | 约 18 GB |
| `UD-Q3_K_XL` | 13.5 GB | 约 17 GB |
| `UD-IQ2_M` | 10.1 GB | 约 13 GB，保真度明显下降 |

把选项节点里的 `gpu_layers` 调低，可以塞进一块更小的卡，代价是速度。`Q4_K_M` 完全
卸载到一块高端消费级 NVIDIA 卡上时，CUDA 带着适配器大约 **50 tok/s**，不带则是
78 tok/s —— 那约 35% 就是 llama.cpp 在算适配器那几次矩阵乘法 —— 而 Vulkan 大约是
CUDA 那个数字的一半。

> **一个更小的 Qwen3.5 不是替代品。** Qwen3.5-9B 的头部里写着同样的
> `general.architecture = qwen35`，所以它看上去是对的，模型列表也会把它显示出来 ——
> 但它是 32 个宽 4096 的块，而适配器要的是 64 个宽 5120 的。llama.cpp 会拒绝挂上
> 这个 LoRA（`tensor 'blk.0.attn_gate.weight' has incorrect shape`），于是这次运行
> 失败。节点会先查这两个头部数字，并在下载任何东西之前把这件事说出来，还会在下拉框
> 里给这类文件打上 `(wrong size for the adapter)` 的标签。要是你看见一个 9B 产出了
> 一段看上去像模像样的重写，那它跑的是**不带**适配器的：那个格式来自系统提示词，
> 不是来自 LoRA。这不是一条死路 —— 那恰恰正是[写作节点](#minimax-h3-prompt-writer-t2vai2vafl2val2va-提示词写作节点)
> 有意在做的事，只不过它们放进提示词的是完整的指南，而不是七行字。

> **更新的 Qwen 也不是替代品。** Qwen3.8-27B 同样是 `qwen35`，同样是 64 个宽 5120
> 的块，所以形状检查会放行它，llama.cpp 也会一声不响地挂上 LoRA —— 挂在它从没
> 训练过的权重上。这个适配器只在
> [Qwen3.6-27B](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA)
> 上训练过。节点会读取 GGUF 给自己写的名字，如果其中没有一个是 Qwen3.6-27B，就在
> 节点上和 Check it 里发出警告。它不会拒绝，因为这些名字是可选的，由量化这个文件
> 的人填写。Qwen3.6-27B 的 **MTP 版本**没有问题：它报告 65 个块，是因为带着一个
> 用于投机解码的草稿头，而节点不把这个头算进去 —— 在这里 llama.cpp 从不加载它，
> 适配器也没有给它的张量。

GGUF 这条路用的是**转换过的**适配器，不是 PEFT 那个 ——
[27B LoRA](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF) 和
[8B 那个](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-8B-GGUF)
的 F16 与 Q8_0。它是不问自取的；想用你自己的，就把那个 `.gguf` 丢进 `models/LLM`，
再从选项节点的 `adapter` 列表里挑它，或者在模型列表里设 `adapters.gguf.repo`。提示词
是照着这个 GGUF 自己的聊天模板、带着 `enable_thinking=False` 拼出来的，结果和
`transformers.apply_chat_template` 产出的逐字节一致 —— 模型看到的，正是这个 LoRA
当初训练时用的那段文字。

检查点自带量化时，`quantization` 小部件被忽略 —— bitsandbytes 不会叠在 AWQ 或者
FP8 上面。

### 节点上的进度

下载、权重加载和 token 生成，全都通过 ComfyUI 自己的进度通道报到节点上 —— 一个
进度条，外加一行说明，写着当前文件、已传输的大小、速度和预计剩余时间。这里不牵涉
任何自定义的前端扩展，所以 ComfyUI 前端更新时不会弄坏什么。

### 环境变量

| 变量 | 作用 |
|---|---|
| `HF_TOKEN` | 访问受限或私有仓库用的令牌 |
| `HF_ENDPOINT` | 用来代替 `huggingface.co` 的镜像 |
| `MINIMAX_H3_LLAMA_BIN` | 这台机器上已经有的 llama.cpp：那个可执行文件，或者装着它的文件夹 |
| `MINIMAX_H3_MTMD_SERVER` | `auto`、`never` 或 `always`：一次描述运行是让一个模型一直开着，还是每份参考素材加载一遍 |

### 语言

把 ComfyUI 设成俄语（Settings → Comfy → Locale），这些节点就以俄语出现：每个节点是
什么，以及每个小部件的工具提示 —— 那些长长的解释才是这份翻译的重点，因为你真正会读
的正是它们。

**小部件的名字有意保持英文。** `repeat_last`、`fix_once`、`prompt_file` 是标识符，
不是散文：它们是工作流 JSON 里、以及一次 API 调用里的键，是这份 README 和 issue 里
对这些东西的称呼，而工具提示也是用它们来点名小部件的。把标签翻译掉，会让上面每一处
引用都指向虚空 —— 所以一段俄语的工具提示里可以出现 `repeat_last`，而它和节点上写着
的那个对得上。

有一个工具提示是故意留成英文的 —— 选项节点上的 `device`，它是在运行时拼出来的，
结尾是这台机器自己的 GPU 列表。一个静态文件装不下这种东西，除非把某一个人的硬件发给
所有人。

翻译放在节点包根目录下的 `locales/<lang>/nodeDefs.json` 里，那正是 ComfyUI 会去找的
地方。加一种语言就是加一个文件夹；Python 那边不需要知道任何事。有两样东西帮着让一份
翻译保持诚实：

```text
python tools/locales.py report ru     缺了什么，以及什么已经过时了
python tools/locales.py fill ru       把缺掉的键以英文补进去，再往上翻译
```

`report` 是去问一个正在跑的 ComfyUI 这些节点究竟是什么，所以它不会像一份签进仓库的
副本那样和代码漂开；`fill` 从不覆盖一个已经有翻译的键。其余的由测试套件覆盖 ——
文件是不是合法的 JSON、它点名的是不是只有本节点包的节点，以及每一个键是不是 ComfyUI
真的会去读的那个，因为一个拼错的键会无声地失败，只是永远不出现。

**已经在画布上的节点，保留它被创建时的那个标题。** 切换语言不会给它改名，因为那个
标题是保存在工作流里的。想看到翻译过的名字，就重新添加一个节点。

## 杂记

- **速度。** Qwen3.6-27B 是一个混合模型：它 64 层里有 48 层用的是线性注意力。没装
  `flash-linear-attention` 和 `causal-conv1d` 时，Transformers 会退回一条更慢的纯
  PyTorch 路径，并在控制台里把这件事说出来。那条回退路径是正确的，只是更慢；这两个
  包都是可选的，而且在 Windows 上不好编。
- **确定性。** 开着 `greedy` 时，同样的提示词、分辨率、时长和种子产出同样的重写，
  而 ComfyUI 也照此缓存这个节点。
- **中断。** 取消一次运行会把正在进行的下载和生成一起停掉；下到一半的下载会在下一次
  运行时续传。
- **写作节点上的格式是被遵循的，不是被保证的。** 一个通用模型是在跟随指令，而不是在
  复现它训练时学到的某个分布。让 `greedy` 开着 —— 小模型一采样就会从格式里漂出去
  —— 而要是某个字段缺了，节点会把它确实拿到的东西全部返回，并在节点上点名那个缺口，
  而不是直接失败。
- **一个拒绝 system 角色的聊天模板照样能用。** Gemma 的模板碰上 system 会
  `raise_exception`；那时指南会被折进第一轮用户消息里，正如那些模型自己的说明卡所
  规定的。
- **列出你的 GGUF 模型几乎不花钱。** 建这个下拉框只需要每个文件头几 KB 里的六个值，
  可是 `gguf.GGUFReader` 一打开文件，就把整个头部实体化出来 —— 包括
  `tokenizer.ggml.tokens`，二十五万个字符串。一个模型文件夹里十个文件要花 31 秒，
  而这笔钱是在 ComfyUI 第一次回答 `/object_info` 的时候付掉的。现在改成直接走过那个
  头部再跳过去，同样这六个值只要 **0.4 秒**。一个下到一半的文件仍然会被拒绝 ——
  靠的是拿它的张量偏移量和它的大小对一对，而不是靠映射失败。
- **描述节点会把媒体写进一个临时文件夹** —— 一个 IMAGE 和一个 VIDEO 变成 PNG，
  一个 AUDIO 变成用标准库写出来的 16 位 WAV —— 事后再把那个文件夹删掉，子进程崩溃
  时也照删。
- **VIDEO 是在这里取样的，不是递给 `llama-mtmd-cli --video`。** 那个参数是通过
  *stdin* 把文件喂给 `ffprobe` 的，而当 MP4 把它的 `moov` 原子放在文件开头时 ——
  那正是“faststart”的意思，也是 ComfyUI、手机和网上大多数东西产出的样子 ——
  ffprobe 读几 KB 就拿到了它要的东西，然后退出，剩下的根本不读。llama.cpp 却还在往
  那根管子里写剩下的几十兆，于是就永远堵在那儿：没有输出，没有报错，也没有结尾。
  同一段片段，把 `moov` 挪到末尾，六秒就跑完了。所以帧改成在进程内解码，这也顺带让
  `max_frames` 对一个 VIDEO 有了意义。
- 重写可能会添上一段短提示词从没说过的细节。当身份、对白、时间点或者构图必须精确时，
  先审一遍再去生成。

## 致谢

模型方面的工作全都是 [LightX2V](https://github.com/ModelTC/LightX2V) 的 ——
这个仓库只是把他们的适配器接进 ComfyUI。你要是觉得有用，就去给
**[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V)** 点个星，MiniMax-H3 的
推理支持和将来的重写任务（FL2VA、Ref2VA）都在那儿维护。

| 组成部分 | 来源 |
|---|---|
| LoRA 适配器 | [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA) |
| LoRA 适配器，8B | [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B) |
| 底座语言模型 | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| 底座语言模型，8B | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| 视频与音频生成器 | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| 提示词写作指南 | [MiniMaxAI/MiniMax-H3 `docs/`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/docs) —— 运行时取回，见上文 |
| 推理框架 | [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) |
| 随包附带的提示词，以及它们描述的那些片段 | [ostris/minimax_h3_1k](https://huggingface.co/datasets/ostris/minimax_h3_1k) —— 经作者许可使用；提示词实际上等同于 MIT，片段以及从中截下来的帧是 MiniMax-H3 的输出，不得用于训练模型 |
| 它们的拍摄风格和主体标签 | [H3 Atlas](https://cohub.live/baize/video-altas/w/h3-atlas) |
| RefMod 格式，以及它的参考素材如何呈现 | [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)，作者 Luisa (luisacaotica)，MIT；为文本编码器保存的图片和“每四张取一张”的默认做法来自 [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder)。由本节点包自己的代码读取，没有复制它们的代码 |

`minimax_h3_rewriter/prompt_template.py` 和 `prompt_template_8b.py` 里的提示词模板，
是从它们各自的适配器仓库逐字节复制过来的；改动它们会让重写变差。

使用 MiniMax-H3 受[官方 MiniMax-H3 仓库](https://huggingface.co/MiniMaxAI/MiniMax-H3)
里的许可证和可接受使用条款约束。

感谢 [AxiomGraph](https://www.youtube.com/@AxiomGraph) 为这些节点做的第一期视频评测
—— [《Stop Struggling With MiniMax H3 Prompts. Do This Instead.》](https://www.youtube.com/watch?v=h3rZTIRB_G8)
—— 以及上文链接的那些社区工作流。

感谢 [ЭйАй Генератьон](https://www.youtube.com/@AyiTheDeer) 做的第二期、也更详细的
一期，用俄语，它把这些节点放在其他 MiniMax-H3 节点包之中来看，而不是孤立地看 ——
[《MiniMax H3 - как создать непрерывное длинное видео. Обзор наборов нод для ComfyUI + рерайтер промта》](https://www.youtube.com/watch?v=PZd9fWX15VA)。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。它只覆盖 ComfyUI 集成部分的代码；模型权重各有自己的
许可证。


---

## FILE: docs/api_writer.md (https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/blob/HEAD/docs/api_writer.md)

# API Universal Writer

[Русская инструкция](api_writer_RU.md) | [Security review notes](api_writer_security.md)

**MiniMax-H3 API Universal Writer** writes H3 prompts through a running
llama-server or an OpenAI-compatible remote API. This is an additional node;
the existing local writers work as before. It accepts images, video, audio and
decoded RefMods, with a reference strip, prompt library and `repeat_last`.
Video travels as sampled frames with timestamps. An explicit descriptions mode
lets a text-only model work from descriptions supplied by you or a RefMod.

The workflow stores a connection ID and an optional model ID. Configure the endpoint, models and key
in **API providers** beside the model selector, or through the local CLI. The pack does not start or stop your LLM
server. Restart ComfyUI after installing this code so it registers the new node.

Start with **11 - Write through an API** in the pack's workflow templates. It
contains API Writer, API Writer Options and core Preview Any nodes for the
prompt, findings and report. Configure a provider and choose it before queuing:
the template deliberately uses `configure-a-profile`, with no endpoint, key,
private model ID or personal media file.

## Provider support

This version implements the OpenAI-compatible **Chat Completions** protocol.
Provider names do not imply that every API or model from that provider works.

| Connection | Verification and boundary |
|---|---|
| llama.cpp / llama-server | Live local text, images, sampled video and WAV audio on suitable models; model selection and cancellation checked. The external process is managed separately. |
| DeepSeek ordinary API | Live custom text, H3 text, I2VA and Ref2VA with sampled images. Audio is not declared by the supplied profile. |
| Ollama, including WSL2 | Live `/v1` text, image and cancellation checks. Native `/api/chat`, model pulling and `keep_alive` controls are outside this adapter. |
| vLLM | Chat Completions contract, aliases, errors and limits checked with loopback test servers. No live vLLM engine was available for this release preparation. |
| Other compatible API | Configure exact endpoint, served model IDs, parameters and limits yourself; no live provider certification is implied. |
| Gemini native API / Anthropic Messages | Not implemented by this Chat Completions adapter. A separate protocol adapter is needed. |

GLM/Kimi coding subscriptions are not ordinary integration API credentials.
The [GLM Coding Plan tool list](https://docs.z.ai/devpack/tool/others) limits
subscription use to supported environments; this node is not on that list.
[Kimi's platform comparison](https://www.kimi.com/code/docs/en/kimi-code/faq.html#empty-model-list-during-login)
separates Code membership endpoints/keys from its pay-as-you-go platform.
These restrictions were rechecked on 2026-11-07. The pack does not impersonate a
coding client or silently switch a subscription to a billed API.

## Edit providers in ComfyUI

Click **API providers** beside **Refresh models** on the API Universal Writer.
The window uses the pack's resizable library-window controls. Select an existing
connection or choose a template and **Add provider**. Set its ID, display name,
Chat Completions base URL, network mode and streaming support.

Under **Allowed models**, add exact served IDs or aliases. Each model has its own
input modalities, output formats, supported parameters and limits. Set one as
the default. These declarations describe what the server can already do;
checking **audio** or **video** does not install support. Sampled video requires
**image** input; the **video** flag records a declaration for this adapter. Generation values still
come from the API Writer Options node.

Choose **Encrypted server storage** to enter a key. **Keep existing key /
configure later** leaves a saved key unchanged. **Enter or replace key** accepts
a new one; **Delete saved key** removes it. Environment mode stores only the
variable name and does not edit the server environment. A saved key is never
loaded into a password field or returned through an endpoint.

**Test connection** checks the current form, including unsaved providers, model
IDs, endpoints and new keys. It sends a fresh `GET /models` from the ComfyUI
server, with a 3-second connection limit and a 10-second total limit, and reports
whether the highlighted model was listed. Only locally allowed IDs are returned.
The test does not save settings/keys, populate the generation model cache, load
a model or generate text. It cannot verify declared modalities, formats or
generation parameters. A public `/models` endpoint may ignore credentials, so
success there does not prove a key is valid for generation. HTTP 401/403 and
connection/TLS/timeout errors are shown without returning provider response bodies.
A typed key uses the same one-use encrypted envelope as saving; it remains in
the input for **Save provider**, or is cleared when the window closes. Saved
keys cannot be sent to an edited recipient: re-enter a key for the new endpoint.
Tests run one at a time per editor, with a three-second cooldown. No automatic
retry or fallback request is made if `/models` is unavailable.

**Save provider** writes server configuration without calling a model. It updates
the profile/model choices on open API Writer nodes. **Use in this node** explicitly
selects the saved connection and the model highlighted in the window, including
when it is the profile's default model. It does not change the profile default.
The highlighted model is preserved after saving, and reopening the node's own
profile highlights its current explicit model. Save edited settings before
applying them to a node. Deleting a
provider leaves its ID in existing workflows, so they fail visibly rather than
switching to another recipient. Existing profile IDs are fixed; create a new
provider to use another ID.

Changing models on the same connection updates the encrypted key's allowed-model
binding during save. Changing its endpoint or credential source requires a new
key or explicit deletion of the old one. Concurrent changes from another window
or the CLI are rejected; reopen the editor before saving again.

Each editor request has a 20-second deadline, including response-body reading.
The status distinguishes settings validation, key encryption and saving. A
network error, invalid reply or lost commit response releases the waiting
state. If the save result is unknown, **Close** remains available and another
save is blocked until you reopen the window and check the stored settings.
Closing or timing out does not prove a server write was cancelled. The editor
never retries a save automatically.

Reference strips release discarded image/video downloads, playback and hover
timers when redrawn or removed. Old video previews no longer occupy browser
connections needed by the provider editor and other ComfyUI requests.

New keys travel as a one-use encrypted envelope: Web Crypto uses AES-256-GCM,
and wraps that random AES key with the server's temporary RSA-3072-OAEP/SHA-256
public key. The 90-second challenge fixes the profile operation on the server
and can be consumed only once. Private RSA keys stay in memory. The API key is
briefly plaintext in the browser input and server memory, where it is needed
to encrypt/store it and authorize requests. This does not protect against a
compromised browser extension, custom-node script or server account.

### Remote ComfyUI and Linux storage

Direct local access through `localhost` or a loopback IP needs no separate token
unless the operator enables one. Remote editing requires HTTPS and these server
settings, supplied before ComfyUI starts:

| Environment variable | Value |
|---|---|
| `MINIMAX_H3_API_ADMIN_ORIGIN` | Exact browser origin, for example `https://comfy.example.com`, including a nonstandard port if used, without a trailing slash |
| `MINIMAX_H3_API_ADMIN_TOKEN` | A separate random administrator token, 32-512 printable ASCII characters |
| `MINIMAX_H3_VAULT_KEY` | On Linux: base64 of 32 cryptographically random bytes, kept stable across restarts |

Use the hosting provider's secret manager or a protected service configuration.
For generating the two secrets yourself, these commands print them to your
terminal; keep their output private:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
python3 -c 'import base64,secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())'
```

The first value is the administrator token; the second is the Linux vault key.
Keep a separate secure backup of the vault key. Losing or simply replacing it
makes existing ciphertext unreadable; changing it is not a key-rotation procedure.
The pack never writes that master key beside the encrypted file or sends it to
the browser. On Windows it keeps using current-user DPAPI, without a vault-key
environment variable.

On Linux, `api_secrets.enc` uses AES-256-GCM with a fresh 96-bit nonce per write
and authenticated format metadata. New files use mode `0600`, with a `0700`
configuration subdirectory. Without a configured vault key, explicit environment
API credentials remain available and browser key entry is disabled.

The window asks for the administrator token when needed. It holds it only in
that window's memory and sends it in a request header; it is not saved in the
workflow or browser storage. Setting the token also protects local editing.
This authorization covers the editor, not all of ComfyUI: protect the rest of
an Internet-facing ComfyUI installation separately.

A TLS reverse proxy must preserve the public `Host`, set
`X-Forwarded-Proto: https`, and reach ComfyUI over loopback. Forwarded TLS headers
from remote peers are rejected. Direct ComfyUI TLS also works. An SSH tunnel to
localhost is an alternative; plain HTTP to a VDS/LAN address cannot administer
profiles even though the API-key envelope itself is encrypted. Web Crypto needs
a [secure context](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Secure_Contexts).

The [Registry standards](https://docs.comfy.org/registry/standards) prohibit code
obfuscation and runtime package installation; they do not prohibit credential
encryption. This implementation uses readable source and the declared
`cryptography` dependency, with no runtime installer. A Registry scan/review of
the actual release is still required.

## Configure connections

Run the following from the pack directory with a Python environment containing
its declared dependencies. Use the **same OS account as ComfyUI**. The examples
use the repository's development environment; a normal installation can use
its ComfyUI Python instead.

`--user-directory` is the existing directory ComfyUI uses. For a default
portable installation it is `<ComfyUI>/user`; if you launch ComfyUI with
`--user-directory`, pass that exact directory here too.

```powershell
.\.venv\Scripts\python.exe -m minimax_h3_rewriter.api_admin --user-directory 'D:\ComfyUI\user' init
```

For a new configuration, `init --include-local-runtimes` also creates anonymous,
text-only Ollama and vLLM templates. Replace their placeholder model IDs before
use. Both forms of `init` refuse to overwrite an existing configuration; the
ordinary `init` still creates only the original llama-server/DeepSeek pair.

If portable Python omits the current directory from its module search path,
run `python.exe minimax_h3_rewriter\api_admin.py` with the same arguments instead.
The administrator module ships in Registry installations too; the `tools`
directory is only a repository convenience.

This creates `<user-directory>/minimax_h3_rewriter/api_profiles.json` with
non-secret llama-server and DeepSeek templates. It never overwrites an existing
file. Edit that JSON locally:

- For `llama-server-local`, set `base_url` to the running server's `/v1` base
  and `model` to its served alias. The default is `http://127.0.0.1:8080/v1`
  with a placeholder `local-model`. The template is anonymous and text-only.
- For `deepseek`, the template uses the ordinary DeepSeek API and a vault record
  named `deepseek`. Coding-plan credentials for GLM or Kimi are not interchangeable
  with ordinary API keys. This stage does not integrate coding subscriptions.
- Declare only the formats, generation parameters and modalities the chosen
  model supports. The node does not infer support from a model name or silently
  fall back to another provider.
- Connections use streaming by default, including existing profiles without a
  `stream` field. The node collects the stream into one completed answer. This
  lets a forwarding proxy notice a cancelled connection during generation,
  instead of waiting for the entire response. For an API that only supports
  ordinary JSON responses, set `"stream": false` in its local profile. There is
  no automatic retry with different transport settings.

You can validate and save an edited document through the tool:

```powershell
.\.venv\Scripts\python.exe -m minimax_h3_rewriter.api_admin --user-directory 'D:\ComfyUI\user' import 'D:\my-profiles.json'
.\.venv\Scripts\python.exe -m minimax_h3_rewriter.api_admin --user-directory 'D:\ComfyUI\user' list
```

The profile dropdown is built from this file. Refresh node definitions or
reload ComfyUI's page after changing profile IDs. A malformed file produces a
configuration error when running the new node; other nodes still register.

## Local Ollama in WSL and vLLM

Ollama in WSL2 serves the same HTTP API as a local Linux installation. Start
your installed Ollama service in WSL, then check it from Windows:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 5
Invoke-RestMethod -Uri 'http://127.0.0.1:11434/v1/models' -TimeoutSec 5
```

[WSL supports Windows access to Linux services through localhost](https://learn.microsoft.com/en-us/windows/wsl/networking#accessing-linux-networking-apps-from-windows-localhost).
If that connection fails, check the running distribution, service and WSL
network configuration first. A direct private WSL address requires an explicit
`network: "lan"` profile. The node uses the configured HTTP recipient; WSL and
the LLM service remain under your control.

The optional templates use these settings:

| Profile ID | Base URL | Model to configure | Credential |
| --- | --- | --- | --- |
| `ollama-local` | `http://127.0.0.1:11434/v1` | Exact installed ID including its tag, replacing `local-model:tag` | Anonymous |
| `vllm-local` | `http://127.0.0.1:8000/v1` | Served model ID or alias, replacing `served-model` | Anonymous, or your explicit vault/environment reference |

Both use `chat_completions`, streaming, text output and the common
`max_tokens`, `temperature` and `seed` parameters. Each starts with text input
only. For an existing configuration, add the needed profile locally rather
than running `init` again. Declare its image/audio support, response formats
and token limits for the actual model and server. Extra `models` entries keep
their own conservative defaults as described below.

The [Ollama compatible API](https://docs.ollama.com/api/openai-compatibility)
accepts Chat Completions, image parts, streaming and model listing. The local
service does not require a dummy API key. Context size and model retention are
server settings for this connection: use your Ollama configuration, including
`OLLAMA_CONTEXT_LENGTH` and `OLLAMA_KEEP_ALIVE` where needed. A native `/api/chat`
adapter is not part of this client. Video remains timestamped JPEG frames;
audio requires explicit support for the writer's `input_audio` content parts.

For vLLM, the connection follows the
[OpenCode runtime pattern](https://opencode.ai/docs/providers/#runtimes): a base
URL, a readiness check and a model inventory, with capabilities configured
separately. The existing Python client sends generation requests to
`/v1/chat/completions`. The [vLLM server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
must provide a suitable chat template. Set limits to match its context and
multimodal configuration. Server defaults can also come from the model's
`generation_config.json`; omitting temperature uses those defaults.

Check a configured connection from the local administrator tool:

```powershell
.\.venv\Scripts\python.exe -m minimax_h3_rewriter.api_admin --user-directory 'D:\ComfyUI\user' probe ollama-local
.\.venv\Scripts\python.exe -m minimax_h3_rewriter.api_admin --user-directory 'D:\ComfyUI\user' probe vllm-local --health
```

`probe` reads the configured `/models` and prints only the allowed IDs and
public capabilities. With `--health`, it first checks `/health` on the same
recipient, preserving any proxy prefix before `/v1`. Health allows three
seconds and at most 8 KiB; model listing allows ten seconds and at most 1 MiB,
each also bounded by the profile's response limit. Redirects and retries are
disabled. A failed health check stops before model discovery.

The probe makes no generation or model-management requests and does not edit
profiles. Authenticated probes use the configured server-side credential;
normal legacy vault-binding migration rules still apply. The node's
**Refresh models** continues to query only `/models`, and the local model
allowlist still controls the choices. A successful probe confirms HTTP access
and listing, not generation quality or multimodal support.

## Save or remove a key on Windows

```powershell
.\.venv\Scripts\python.exe -m minimax_h3_rewriter.api_admin --user-directory 'D:\ComfyUI\user' key-set deepseek
.\.venv\Scripts\python.exe -m minimax_h3_rewriter.api_admin --user-directory 'D:\ComfyUI\user' key-delete deepseek
```

`key-set` prompts twice with hidden input in your local terminal. There is no
command-line key argument, node key widget or key-reading HTTP endpoint. Do not put
the key in a workflow, a prompt, a screenshot or a chat message.

Saved keys use Windows [current-user DPAPI](https://learn.microsoft.com/en-us/windows/win32/api/dpapi/nf-dpapi-cryptprotectdata)
in `api_secrets.enc` alongside the non-secret profiles file. Temporary writes
also contain ciphertext only; replacement is atomic and local writers use a
file lock. No master key is saved next to this file. An invalid file or failed
decryption stops the operation and preserves the original store. Moving to
another user or computer generally requires entering the keys again.

The encrypted record is bound to the profile ID, endpoint, protocol, network,
allowed model IDs and secret reference. Switching between these models, or
changing the default within the same allowed set, does not require another key.
After changing the recipient, profile ID or secret source, enter a key again in
the window or run `key-set` again. The same CLI commands work on Linux with the
server vault key configured.
Legacy records remain readable. Expanding a legacy profile with additional
models upgrades its encrypted binding on first credential use, only when its
original default model and complete destination still match. No plaintext
copy is written. After later changes to an already migrated model set, run:

```powershell
.\.venv\Scripts\python.exe -m minimax_h3_rewriter.api_admin --user-directory 'D:\ComfyUI\user' key-migrate deepseek
```

Credentials are read for each request; changing a saved key does not require
a ComfyUI restart. A `401` is a credential rejection. Connection failures can
instead identify DNS, proxy, operating-system access, refusal/reset or an
incomplete response. These details use fixed messages, never raw exceptions,
headers or proxy credentials. Failed requests are not retried automatically.

This preserves the key and verifies the same profile, recipient, protocol,
network and secret reference. It cannot migrate a key to another endpoint.
The tool's `list` command prints availability, never the key. It does not make
an API request or test billing/access.

## Choose a model within a connection

The **model** selector sits below `profile_id`. **From profile** (the default)
uses that profile's existing `model`, so old workflows retain their behavior.
An explicit choice saves the model ID in the workflow. Selecting another
profile preserves that choice and marks it if it is not allowed there; choose
**From profile** or an allowed ID deliberately. There is no automatic fallback.

The existing top-level `capabilities` and `limits` describe the default model.
Add an optional `models` array for other allowed IDs, each with its own support
and limits. For example, add this field to the DeepSeek profile, keeping its
existing default model and other connection fields:

```json
"models": [
  {
    "id": "deepseek-v4-pro",
    "capabilities": {
      "modalities": ["text"],
      "formats": ["text", "json_object"],
      "parameters": ["max_tokens", "temperature"]
    },
    "limits": {"output_tokens": 8192, "context_tokens": 1048576}
  }
]
```

This is a conservative declaration based on the
[DeepSeek model-list documentation](https://api-docs.deepseek.com/api/list-models/),
checked on 2026-11-07; it does not prove access for an account. Additional entries
default to text input/text output with conservative limits if fields are omitted;
they never inherit vision/audio support from the default model. Do not duplicate
the default ID in `models`. Up to 64 models including the default are supported.
After editing the file, reload the page to reread local choices.

**Refresh models** explicitly reads `<base_url>/models` on the ComfyUI server,
using that connection's credential. Opening or restoring a workflow only reads
local configuration/cache; it makes no provider call. The list is filtered to
locally allowed IDs. Unknown models are not automatically authorized. Only IDs,
availability and validated capabilities/limits reach the selector; router paths,
process arguments and credentials do not. Requests allow 10 seconds and at most
1 MiB (or the profile's smaller response limit), without redirects or retries.
Refreshes have a three-second cooldown; availability metadata is cached for
60 seconds. Network errors appear next to the selector and preserve the choice.

Available metadata can narrow locally declared modalities or token limits.
Missing metadata grants no extra support. An unavailable allowed model remains
visible, and a fresh cached server list stops its execution before HTTP or key
access. Without a fresh list, the admin declaration applies and the API can
still reject a model; there is no fallback. Repair pins the same model/settings.
Reports and saved prompt information record the model actually used. Repeating
a kept prompt reports its original model, even after the selector changes.

For a single-model llama-server, configure its served alias (`--alias`) and
keep the default; choosing another name cannot select a different GGUF. For a
[llama.cpp router](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#using-multiple-models),
add its configured model IDs to the local allowlist. A generation request sends
the chosen ID in `model`; the router may load it under its own configuration.
The node makes no load/unload/download requests and launches no processes.
An external proxy/launcher must expose a compatible model list and routing;
the node does not assume that it behaves like llama.cpp's router.

DPAPI protects the saved copy. A process running as the same OS user can access
that user's decrypted data; this is not isolation from arbitrary local Python
code. Python strings cannot be guaranteed erased from memory. The node keeps
credentials out of its outputs, reports, session memory and workflow data.

## Explicit environment credentials

For a headless system or an OS without this Windows backend, change a profile's
reference to:

```json
"secret": {"kind": "environment", "name": "MY_DEEPSEEK_API_KEY"}
```

Set that variable in the environment that launches ComfyUI. Only the explicitly
named variable is consulted. The pack does not persist its value or describe
this mode as encrypted storage. There is no automatic fallback from a vault
to environment variables or a plaintext file. macOS Keychain and Linux system
stores have an interface for later integration; they are not implemented yet.

## Write text

Choose `profile_id`, keep `task = T2VA`, and enter the raw request, duration and
aspect ratio. With an empty `system_prompt`, the node reads MiniMax's guide,
fetching it if missing and `auto_download_guide` is enabled. A custom system
prompt replaces the whole system message and skips guide access.

The reference strip is the first control, followed by the task buttons,
aspect-ratio picker and duration slider used by Universal Writer. Click
`+ instr` on a reference to edit its instructions and descriptions.

Connect **MiniMax-H3 API Writer Options** to `options` for these settings:

| Setting | Default |
|---|---|
| `max_tokens` | 4096 |
| `response_format` | text |
| `temperature` | -1 (server default) |
| `timeout_seconds` | 180 |
| `auto_download_guide` | true |
| `video_sample_seconds` | 1 |
| `video_max_seconds` | 15 |
| `batch_video_fps` | 24 |
| `audio_sample_rate` | 16000 |
| `repair_once` | false |

A connected Options node supplies all ten values. Without it, a new writer
uses these defaults. Existing saved API workflows retain their scalar values;
the frontend hides those legacy controls, and connecting Options overrides them.
Endpoint, model and credential settings remain in the local connection profile.

- `validation = h3` splits sections and runs the pack's H3 checks. Findings are
  advisory. Missing fields remain visible rather than being invented.
- `validation = custom` returns plain final text, skips H3 checks and leaves
  section outputs empty. Use it when your system message targets another format.
- `response_format = text` preserves the final answer as received. `json_object`
  or `json_schema` requires supported profile capabilities and exactly the H3
  string fields; the node assembles them into labelled H3 text. No silent format
  fallback occurs.
- `temperature = -1` and `seed = -1` omit those parameters. Otherwise the profile
  must explicitly support them. An API seed does not promise reproducibility.
  New API writers use `control after generate = fixed` to keep the default
  `seed = -1` across requests. For DeepSeek, keep both settings: `randomize`
  produces a positive seed that its profile rejects before sending a request.
  Existing workflows keep their saved control mode; change it to `fixed` and
  set `seed` back to `-1` if the profile does not support seed.
- `max_tokens` is per request and bounded by the profile. Some models count
  reasoning against this budget; increase it if generation ends before final
  text. Truncated, refused or tool-call answers are errors.
- `timeout_seconds` limits the complete HTTP request, including a slow body.
  ComfyUI cancellation closes our stream. A server that observes the disconnect
  can stop generation; a proxy may notice only on its next attempted write.
  During a long prompt prefill this can be delayed until output starts. Other
  services may still finish or bill the submitted request. Failed requests are
  not automatically retried. The pack never stops the whole server or other
  clients' work to cancel its own request.
- `repair_once` in API Writer Options is off by default. Enabling it permits **one extra, potentially
  paid request** for repairable H3 findings. It does not retry transport errors.

Outputs include the full prompt, H3 sections, separate reasoning, findings and
a JSON report with attempt counts, elapsed time and known token usage. Missing
usage stays unknown. Connect `prompt` to the H3 generator's prompt input.

## Use images

Connect an IMAGE to `ref_0`; another free slot appears, up to 12 references.
Each square in the strip represents a connected batch. Drag squares to change
their order, click an image label to cycle Picture, Subject and Video, or click the
image to disable it. The instructions button adds text sent with that reference.
Disabled references are neither encoded nor sent. T2VA ignores all references
and returns an empty `references` bundle, with a note in the report.

| Task | Enabled Picture references |
|---|---|
| I2VA | One, the first frame |
| FL2VA | Two, first and last in strip order |
| L2VA | One, the last frame |
| Ref2VA | At least one reference; Picture, Subject, Video and Audio have separate numbering |

The selected profile must declare `image` in `capabilities.modalities`, and its
model must actually support vision. The llama-server template stays text-only;
enable images only after loading a compatible model and projector yourself.
DeepSeek's `deepseek-flash` accepts inline images through its
[documented vision API](https://api-docs.deepseek.com/guides/vision/).

The node sends JPEG copies directly in the user message. `image_max_side`
defaults to 1024 pixels; resizing keeps aspect ratio, never crops or enlarges.
Copies use quality 90; alpha is composited onto white. Every frame in an IMAGE
batch is sent under that batch's single reference label. The ceiling is 64
images across all batches and 100 million source pixels per image. Exceeding a
limit fails explicitly, without dropping images or starting a partial request.
No local captioner, temporary image files, public image URLs or Files API uploads are used.

The JSON report's `media` lists original/sent sizes, frame positions, byte sizes
and resizing. The complete request, including base64 and prompts, must fit the
profile's `limits.request_bytes` (the templates default to 1 MiB). For several
images, reduce `image_max_side` or raise that limit in the local profile within
the server's own limits. Nothing is silently resized again to fit a request.

The final `references` output preserves the original objects. Connect it to
**MiniMax-H3 Reference Slots** or **Reference Encode** so H3 receives the same
numbered assets as the prompt. Subjects follow Pictures in the generator's
picture bundle. The first eleven text outputs retain their previous positions.
The node also shows the configured recipient and the data to be sent before a
run; viewing that information does not contact the LLM or resolve credentials.

## Use video and audio

The same `ref_0` and subsequent sockets accept VIDEO, AUDIO and RefMod outputs.
VIDEO is always Video; AUDIO is always Audio. An IMAGE batch can be assigned
the Video role. RefMod kinds come from their loader; only an image RefMod can
switch between Picture and Subject.

Video is sent as timestamped JPEG frames under one `Video N` label. It uses
the same `image_url` transport as pictures, so the profile needs `image`, not
a provider-specific video endpoint. This shows selected frames, not every
motion in the source. DeepSeek can receive these frames through its vision
API. A profile declaring only text will fail before any request.

| Control | Effect |
|---|---|
| `video_max_frames` | At most this many frames per clip, default 8; 64 images total across all references |
| `video_sample_seconds` | Requested interval, default 1 second; when the frame cap is reached, samples spread across the selected window |
| `video_max_seconds` | Window from the active trim start, bounded by this value, target duration and H3's 15-second reference limit |
| `batch_video_fps` | Clock for an IMAGE batch assigned Video, default 24; actual VIDEO inputs retain their presentation times |
| `image_max_side` | Maximum side of both picture and video JPEG copies, preserving aspect ratio |
| `audio_max_seconds` | Standalone audio/voice window, default 30 seconds; clip sound follows its video's selected window |
| `audio_sample_rate` | Rate of mono 16-bit PCM WAV copies, default 16000 Hz |

Click a connected VIDEO's label to include its sound: the label turns purple
and shows its additional Audio number. With sound off, **no audio attachment
is created**. The writer uploads neither the original video container nor its
muted/re-encoded equivalent, so an off soundtrack cannot travel inside it.
A silent clip gets no Audio number even with the switch on, and the report
explains that. A track that exists but cannot be decoded causes an error.

Clip sounds precede standalone sounds in Audio numbering, matching Reference
Slots and Reference Encode. The `references` bundle preserves the original
objects and carries the clip sound flags. Sampling, resizing, cropping an API
audio copy or resampling never modifies those originals.

Audio uses `input_audio` with inline base64 WAV. This form is documented by
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
and [vLLM](https://docs.vllm.ai/en/latest/features/multimodal_inputs/#audio-inputs).
The selected profile must declare `audio`, and its actual model, projector
and server must accept audio in Chat Completions. For example, llama.cpp lists
Qwen2.5-Omni as supporting audio and images. A server accepting images alone
does not qualify. The DeepSeek template declares no audio input: switch clip
sound off there, use descriptions mode, or select an audio-capable profile.
Adding a modality to a profile does not add that capability to its model.

Audio/video preparation uses PyAV (`av`), declared in the installation
manifests. Nothing is installed during node execution. Text/image-only use
does not import PyAV. File VIDEO inputs are streamed from their connected
local source with nested network protocols and playlist formats disabled;
remote URLs, UNC paths and raw filename strings are rejected. VIDEO inputs
already holding image components use that batch. File inputs have a 2 GiB
ceiling; decode loops have a 20000-frame ceiling. Trim large clips upstream.
ComfyUI accumulated/concatenated `VideoFromList` inputs must first be saved
and reloaded as a local clip; otherwise their component getter would decode
the whole concatenation before any frame limit could apply. Descriptions mode
with clip sound off also works without materializing such a video.
Audio requires one waveform batch; multiple batches are rejected rather than
silently selecting the first. Preparation checks cancellation between decoded
frames and audio chunks and leaves no detached decoder running.

The report adds `media_mode`, `preparation_seconds`, video sampling/window
metadata, actual frame times and audio rate/channel/duration/byte changes.
`elapsed_seconds` measures API writing; preparation is reported separately.
Request byte limits include prompts and every base64 copy. A limit error
does not send a partial request or start an automatic retry.

## Use RefMods and descriptions

Connect visual and/or audio outputs from **MiniMax-H3 RefMod**. In native mode,
visual media must already be decoded by that loader's `vae` input, and a voice
by `audio_vae`. The API writer does not load a VAE or a caption model. A missing
decode stops with the slot and loader input to connect. Visual RefMods use
their stored/decoded pictures and timestamps; photo stacks keep their saved
hint that these are separate photographs. Matching visual/voice halves keep
their saved descriptions and related slot IDs in the manifest. The generator
receives the original RefMods, including their conditioning and strength.

If a visual RefMod has no stored thumbnail, its chip gains a preview after
native reference preparation, before the LLM replies. Up to four thumbnails
come from the JPEG copies already prepared for the request, preserving the
Video label for clips and photo stacks. Hover to cycle their frames. This
does not decode the original reference again or load an extra model.

Choose `media_mode = descriptions` to send text descriptions instead of any
media. Open a square's instructions/describe button and fill its description;
for a VIDEO with sound on, also describe its sound. A RefMod can supply its
saved appearance/voice description without a VAE decode. Every enabled
manifest component must have a description. Instructions alone do not count,
and missing text never triggers a hidden captioner or a media fallback.
An existing captioner can produce descriptions upstream; provide the result
through the editor or a wired `reference_descriptions` JSON string:

```json
{
  "ref_0": {"text": "A woman walks down a wet street.", "sound": "Rain and footsteps."},
  "ref_1": "A low, soft voice."
}
```

In this mode, the model is explicitly told that no media are attached.
Description text also accompanies native media when supplied. Disabled slots
are ignored; their pixels, audio and RefMods are not decoded. The JSON fields
remain ordinary fallback widgets if the frontend extension is unavailable.
Saved widget values are restored by name or their recorded order. Workflows
from before the controls moved retain their values, references and links;
old image workflows default to native mode.

## Reuse and save prompts

`bypass` passes the original prompt and enabled original references through
without reading the guide, opening video/audio sources or making an API
request. Section, reasoning and findings outputs are empty; the report records
`source = bypass` and `requests = 0`. Bypass takes precedence over `repeat_last`
and leaves its kept answer intact. Both switches also have the same clickable
badges as Universal Writer.

After a successful run, `repeat_last` returns the kept prompt without calling
the API. With nothing kept yet it performs one normal run. Current enabled
references still pass through; changing those references does not rewrite the
kept text. The report identifies session/library reuse with `requests = 0`.

Use **Save the last prompt**, **Edit the last prompt** or **Prompt library** on
the node to keep or choose text. The existing library stores the result,
reference thumbnails and selected non-secret run settings under the ComfyUI
user directory. The API node does not add the key, credential source, endpoint,
raw request, system message or encoded transport images to that record. Session
memory lasts until ComfyUI restarts; saved library records persist. A selected
library record is used only while `repeat_last` is enabled.

The [contract](api_writer_contract.md) documents validation and transport limits.
Cloud tests require an ordinary API key configured locally by the user; the
test suite itself uses loopback servers and dummy credentials only.


---

## FILE: docs/api_writer_RU.md (https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/blob/HEAD/docs/api_writer_RU.md)

# API Universal Writer

**MiniMax-H3 API Universal Writer** пишет структурированный H3-промт через
запущенный llama-server, Ollama, vLLM или совместимый облачный Chat Completions
API. Входы принимают изображения, видео, звук и декодированные RefMod.
Системное руководство, исходный пользовательский текст и включенные референсы
передаются выбранной модели. Видео отправляется выбранными кадрами с временными
отметками. В режиме `descriptions` отправляются только заданные описания.

Подробный [контракт и инструкция на английском](api_writer.md) содержат все
лимиты и настройки. [Материалы для security-review](api_writer_security.md)
описывают операции с файлами, сетью и ключами.

## Первый запуск

1. Открой пример **11 - Write through an API** в шаблонах пака.
2. Нажми **API providers** возле выбора модели. Выбери шаблон подключения
   или **Custom connection**, затем **Add provider**.
3. Укажи адрес сервера и точный ID модели. Добавь разрешенные модели, выбери
   default, объяви реальные модальности, форматы, параметры и лимиты каждой.
   Галочка `audio` не добавляет серверу поддержку звука.
4. Для облака выбери **Encrypted server storage** и введи API-ключ в поле
   нового ключа. Сохраненный ключ в окно не возвращается. Локальный сервер без
   авторизации использует **No key**.
5. Нажми **Test connection**. Проверяется текущая форма через `GET /models`,
   включая еще не сохраненные адрес, модель и ключ. Настройки не записываются,
   генерация не запускается. Публичный список моделей может не проверять ключ;
   успешный тест не подтверждает доступ к генерации или поддержку медиа.
6. Нажми **Save provider**, затем **Use in this node**: в ноду попадут
   сохраненное подключение и выделенная в окне модель. Пустой выбор модели
   в самой ноде означает default из профиля.
7. Введи идею, выбери задачу, пропорции и длительность, запусти граф.
   Пример выводит готовый промт, findings и report штатными Preview Any.

Пример не содержит ключей, адреса облачного получателя или личных файлов.
Первоначальный ID `configure-a-profile` нужно заменить своим подключением.
Для проверки передачи изображений подключи Load Image к `ref_0` и переключи
задачу на I2VA. Для Ref2VA подключай нужные картинки, видео, звук и RefMod;
порядок и галочки на входах определяют, что увидит модель.

## Параметры и повтор

**MiniMax-H3 API Writer Options** подключается ко входу `options`. Здесь
задаются `max_tokens`, формат ответа, температура, таймаут, загрузка гайда,
частота и окно выборки видео, FPS IMAGE-batch, частота WAV и `repair_once`.
Последний разрешает один дополнительный запрос для исправления H3 и может
увеличить расходы. Без Options действуют значения по умолчанию; старые
workflow сохраняют свои прежние скалярные настройки.

Пустой `system_prompt` берет руководство MiniMax; непустой полностью заменяет
его. Режим `custom` возвращает обычный текст без H3-проверок. Seed `-1` с
фиксированным управлением не отправляется; используй seed только для модели,
которая поддерживает этот параметр.

`bypass` отдает исходный промт и включенные оригинальные референсы без API.
`repeat_last` возвращает сохраненный результат из сессии или библиотеки;
если результата еще нет, выполняется первый обычный запрос. Изменение
референсов не переписывает ранее сохраненный текст. Выход `references`
передает оригиналы в Reference Slots или Reference Encode с той же нумерацией.

## Где хранятся ключи

Профили находятся в `user/minimax_h3_rewriter/api_profiles.json` выбранного
ComfyUI, ключи - в соседнем `api_secrets.enc`. Учитывается `--user-directory`.
В workflow сохраняются ID подключения и модели, настройки и инструкции;
API-ключ, его источник и адрес подключения туда не записываются.

Windows использует DPAPI текущего пользователя. Linux использует AES-256-GCM
с отдельным мастер-ключом `MINIMAX_H3_VAULT_KEY`: base64 от 32 случайных байтов,
заданный оператором до запуска ComfyUI. Храни его отдельно и сохраняй между
перезапусками. Потеря мастер-ключа не восстанавливает сохраненные API-ключи.
Без него доступен явно настроенный источник из переменной окружения.

После замены API-ключа перезапуск не нужен: следующий запрос читает новое
значение. Уже отправленный запрос использует свою прежнюю авторизацию.
При ошибке сохранения с неизвестным результатом закрой и заново открой окно,
проверь сохраненное состояние, затем повторяй действие.

## Удаленный ComfyUI

Редактор на VDS работает через HTTPS и отдельный токен администратора.
Перед запуском сервера задай точный публичный origin без завершающего `/`
в `MINIMAX_H3_API_ADMIN_ORIGIN`, а случайный токен длиной 32-512 печатных
ASCII-символов - в `MINIMAX_H3_API_ADMIN_TOKEN`. Linux дополнительно требует
мастер-ключ для сохранения credentials. Настроенный токен защищает редактор
и при локальном доступе; в браузере он хранится только в памяти окна.

Reverse proxy должен подключаться к ComfyUI по loopback, сохранять публичный
Host и задавать `X-Forwarded-Proto: https`. Этот токен защищает маршруты
редактора, а не весь ComfyUI. Настройка с примерами приведена в разделе
[Remote ComfyUI and Linux storage](api_writer.md#remote-comfyui-and-linux-storage).

## Что проверено

Текст, изображения и выборка кадров проверены с обычным DeepSeek API;
смешанные изображения и звук - с локальным Qwen2.5-Omni. Ollama проверена
через WSL2. Для vLLM проверен HTTP-контракт на тестовом сервере; живой движок
в доступном окружении не установлен. Сведения о моделях задаются явно:
список `/models` не является описанием их мультимодальных возможностей.

Coding-планы GLM/Kimi не подменяют обычные API-ключи и не подключаются
автоматически. Собственные протоколы Gemini и Anthropic Messages требуют
отдельных адаптеров; текущий путь использует Chat Completions.


---

## FILE: docs/api_writer_contract.md (https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/blob/HEAD/docs/api_writer_contract.md)

# API Universal Writer: contracts and execution

`MiniMaxH3ApiUniversalWriter` is an additional ComfyUI node for all five H3 tasks,
with image/audio input, sampled video frames, decoded RefMods and text output.
It sends the guide or a complete custom system message and the raw user prompt
to an administrator's configured Chat Completions server. See
[setup and use](api_writer.md). Existing writers keep their execution paths.

The implementation separates these responsibilities:

| Module | Responsibility |
|---|---|
| `api_contract.py` | Versioned job, model capabilities, reference manifest and API result |
| `api_profiles.py` | Non-secret server configuration, endpoint validation and profile selection |
| `api_client.py` | Request validation, Chat Completions transport and safe error classification |
| `api_config.py` | Strict local profile loading, configured user directory and atomic writes |
| `api_admin.py` | Local CLI, shipped with the Registry package; the repository also has a `tools/api_profiles.py` wrapper |
| `api_secrets.py` | Current-user DPAPI, explicit environment resolution and system-store interface |
| `api_models.py` | Allowlisted discovery, bounded metadata cache and per-model settings |
| `api_model_routes.py` | Local/cached choices and an explicit read-only provider refresh |
| `api_management.py` | Revision-checked profile edits, one-use encrypted key transfer and coordinated vault/config writes |
| `api_management_routes.py` | Local-origin access or explicit HTTPS origin with administrator authorization |
| `api_text.py` | H3 section parsing, checks, reports and optional single repair |
| `api_media.py` | Strict image strip state and in-memory JPEG preparation |
| `api_multimedia.py` | Mixed reference metadata, bounded video/audio decoding, WAV copies and explicit descriptions mode |
| `api_options.py` | Immutable, validated generation and transport settings; separate API Writer Options node |
| `api_writer.py` | ComfyUI node, original reference bundle, library, reuse and cancellation |

Importing them does not read profiles, inspect credentials, download guides or
contact servers. `INPUT_TYPES()` reads non-secret profiles to build its dropdown.
`requests`, `aiohttp`, `av` and `cryptography` are declared in both installation manifests. PyAV
is imported only during audio/video preparation. The API
transport uses aiohttp for cancellation and a total deadline; it reuses
`download._SystemTrust.context()` with certifi added, as the requests adapter
normally does. Certificate and hostname verification stay enabled. Existing
download/model transports are unchanged.

## Profiles

[api_profiles.example.json](../minimax_h3_rewriter/api_profiles.example.json)
contains two templates and no keys:

- `llama-server-local`: `http://127.0.0.1:8080/v1`, model alias `local-model`.
  Replace the alias with the one served by an already running llama-server.
  Capabilities default to text, with optional JSON output and seed. A different
  model or server build may require editing those declarations. The schema
  envelope follows llama.cpp's [Chat Completions parser](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-common.cpp),
  checked on 2026-11-05.
- `deepseek`: `https://api.deepseek.com`, `deepseek-flash`, a vault record named
  `deepseek`. The record is a reference, not a credential. The template declares
  text/image input and text/JSON-object output from the
  [official model metadata](https://api-docs.deepseek.com/api/list-models/) and
  [Chat Completions documentation](https://api-docs.deepseek.com/api/create-chat-completion/),
  checked on 2026-11-05. Images use the standard inline `image_url` blocks.

`ProfileSet.from_dict()` accepts a version-1 document with a `profiles` list.
It rejects unknown fields, duplicate IDs, unsupported protocols and invalid
endpoints. `api_config.load_profiles()` loads the fixed `api_profiles.json`
under ComfyUI's configured user directory. The local CLI and the authorized
API providers window can edit this configuration. The read-only
`GET /minimax_h3_rewriter/api/profiles` returns ID, display name, recipient,
model, network mode and modalities for the node's disclosure. It omits the
secret reference and never calls a provider or resolves/decrypts credentials.

Each profile contains its endpoint, model, network mode, capabilities, limits,
the boolean `stream` (default `true`) and a `SecretRef`. Allowed credential kinds are `none`, `vault` and
`environment`; the latter two contain an identifier only. `SecretResolver`
implements the client's `authorize(profile)` callback. It consults only the
chosen reference, without credential discovery or fallback. Stored keys are
bound inside the encrypted payload to profile ID, protocol, recipient, network,
allowed model set and secret reference. Legacy single-model records remain
readable. When their original default and destination match, adding allowed
models upgrades that binding atomically on first use. Later model-set changes
use the local `key-migrate` command; recipient changes require key entry again.
Profile enumeration and cache stamps do not decrypt keys.

The editor's `POST /api/manage/{open,stage,commit,test,close}` uses fixed server-owned
files. Editing never calls a provider; only explicit `test` reads `/models`.
All operations require a matching, non-null
Origin and reject cross-site fetches. Direct loopback clients may edit locally.
Remote access additionally requires the exact HTTPS `MINIMAX_H3_API_ADMIN_ORIGIN`
and constant-time verification of `X-MiniMax-H3-Admin` against the server's
`MINIMAX_H3_API_ADMIN_TOKEN`. A configured token also protects local editing.
Only a loopback proxy with the configured public Host may forward HTTPS. These
checks do not authenticate the rest of ComfyUI and do not isolate mutually
untrusted users sharing its OS account or same-origin custom-node scripts.

The body limit is 128 KiB, uncompressed JSON with strict fields; responses are
`no-store`. At most eight in-memory editor sessions live for 15 minutes. A
session is bound to its Origin. Stage validates one profile operation against a
SHA-256 revision of the current file. Commit checks that revision again under
the same interprocess lock as CLI profile/key changes. No endpoint accepts a
path, arbitrary headers or a new reference to another profile's vault record.

For replacement, stage creates a temporary RSA-3072 keypair and a 90-second
one-use challenge. Web Crypto encrypts the API key with a fresh AES-256-GCM key
and 96-bit IV, then wraps the AES key using RSA-OAEP/SHA-256. Both OAEP's label
and GCM's additional authenticated data are `minimax-h3-api-key:v1:<challenge>`.
The staged operation stays on the server; commit contains only the session,
challenge and envelope. Consume the challenge before decryption or writing.
Reject malformed/truncated/tampered envelopes and replay without returning
crypto exceptions or plaintext. Saved keys are never returned to the browser.

Connection tests stage a separate `test` operation. Its challenge cannot be
committed as a save, and a save challenge cannot be consumed by `test`. Tests
read the exact current draft after the same validation and recipient-binding
checks, using a typed encrypted key, an explicit environment reference or the
previous vault binding. Reading a legacy vault record does not migrate it.
Nothing is persisted, and discovery/inference caches are not touched. Only
`GET /models` is sent, with 3-second connect/10-second total bounds, the existing
response limit capped at 1 MiB, no redirects or retries. Per-editor single-flight
and a three-second cooldown apply. Responses contain only allowed model IDs,
selected-model availability, elapsed time and whether a credential was sent.
Provider errors use the client's fixed safe messages and numeric HTTP status,
never raw response bodies or exception text. Success does not prove credentials
were checked by a public metadata endpoint, or validate generation/media support.

Windows keeps the existing DPAPI format. Linux/other headless servers can use
`ServerKeyStore` with a random 32-byte master key supplied in base64 through
`MINIMAX_H3_VAULT_KEY`. Its distinct versioned magic authenticates AES-256-GCM
storage with a fresh 96-bit nonce. The master key is not persisted by the pack;
there is no automatic generation, weak derivation, desktop-keyring requirement
or plaintext fallback. Missing/wrong keys preserve ciphertext. The two
administration secrets cannot be selected as model-credential environment
references. The file location, record bindings, size limits and locks are shared
with DPAPI. New Unix files/directories use 0600/0700.

Same-recipient model-set changes rebind only the matching old record. Recipient
or credential-source changes cannot reuse a saved key silently. Vault ciphertext
is replaced before the profile file; if the profile write fails, restore the previous
ciphertext (and report separately if restoration fails). The two-file update is
not crash-atomic: a crash between replacements may require key entry again,
but binding checks prevent sending an old key to a changed recipient. Plaintext
temporary files and transport retries are not used. See the
[user setup](api_writer.md#remote-comfyui-and-linux-storage) for bootstrap and
the limits of browser-side encryption.

The default two templates are unchanged. Local CLI
`init --include-local-runtimes` optionally adds anonymous, text-only Ollama
(`127.0.0.1:11434/v1`) and vLLM (`127.0.0.1:8000/v1`) templates. They contain
placeholder served IDs and require local configuration. WSL is another host
for the configured HTTP server; the pack never executes WSL or service commands.

Local CLI `probe <profile_id>` reads the configured model inventory.
`--health` first checks the same recipient's `/health`, preserving a prefix
before `/v1`, with a three-second deadline and an 8 KiB ceiling (or the smaller
profile limit). Empty 2xx health responses are valid; their body is never
returned. A failed health check prevents discovery. Both GETs use the normal
TLS/proxy/no-redirect policy and resolve the configured credential once.
The printed result is the existing filtered public model list plus a health
status. This command performs no inference or profile administration; normal
legacy encrypted-binding upgrades during credential resolution still apply.
No browser health route is added, and node metadata refresh remains `/models`
only. A successful probe does not establish media support or writing quality.

A job carries `profile_id` and optional `model` (empty uses the profile default).
It cannot supply an endpoint, headers, API key,
credential reference or extra request body. `ApiClient` resolves the ID through
its server-owned `ProfileSet`; there is no fallback to a different profile.

The optional profile `models` array contains other allowed `ModelSettings`.
Each declares its own capabilities/limits, with conservative defaults rather
than inheriting the default model's modalities. The default ID is not duplicated.
The effective selection determines request validation, media byte ceilings,
payload model and result. Authorization receives the original connection
profile. A client's first completion pins selected settings for its repair too.

`GET /api/models?profile_id=...` (under the pack route prefix) returns only
local declarations/fresh cached metadata. `POST /api/models/refresh` accepts
only a small JSON object with `profile_id`, then explicitly reads the configured
`/models` using the same TLS, proxy, redirect and credential policy as completions.
It accepts no credential values or profile edits. Discovery permits 10 seconds, at most 1 MiB,
1024 server entries and 64 locally allowed IDs. Cache entries expire after
60 seconds; refreshes have a three-second cooldown and the cache is bounded.
Only allowed IDs and validated known metadata are retained; paths, names,
router arguments and unknown fields are discarded. Metadata only narrows
local support/limits. Missing metadata never grants new support. Fresh absence
rejects the choice before completion/key access; without fresh metadata, the
administrator declaration applies. No inference/model-management call is made
by discovery. Workflow loading and graph definition reads never refresh it.

Network modes are explicit:

| Mode | Accepted recipient |
|---|---|
| `loopback` | Literal loopback IPv4/IPv6, HTTP or HTTPS |
| `lan` | Explicit private IP, HTTP or HTTPS; no link-local/unspecified address |
| `cloud` | HTTPS with a public address or administrator-managed DNS name |

Credentials in URLs, query strings, fragments, control characters, encoded
paths and path traversal are rejected. DNS and proxies for cloud connections
remain part of the administrator's configuration. Local connections bypass
ambient proxies. Automatic `.netrc` authentication is disabled for all profiles.
All HTTP redirects are rejected without following them or forwarding a key.

## Jobs and references

`make_job()` accepts the original prompt, task, duration, aspect ratio, profile
ID, explicit guide text or custom system prompt, generation settings and ordered
`ReferenceSpec` metadata. An empty system prompt selects the H3 guide. A custom
system prompt replaces it entirely and requires explicit `h3` or `custom`
validation. The original prompt and custom system text retain their whitespace.
There is no automatic guide download in the contract layer.

Job serialization has version 1. `WriterJob.from_dict()` checks the field set
and types; the endpoint and secret source are outside this schema. All input
is data, including prompt text resembling Python, shell commands or HTML.

References arrive after the caller has arranged the strip, removed disabled
slots, and checked whether clip sound really exists. Metadata retains the
slot ID, role, kind, description, instruction, RefMod origin and related slot
IDs. Original media/RefMods remain with the caller for the downstream reference
bundle; no local filename, tensor or file content is serialized by this layer.

The manifest uses the existing `references.turn_order()` convention. Clip sound
gets an `Audio N` before its `Video N`, and every clip sound is numbered before
standalone audio. Picture, Subject, Video and Audio have separate counters.
Subjects remain after pictures in the downstream picture bundle. The caller
must use this manifest for media preparation, H3 checking and conditioning.

All five H3 tasks are represented. Picture counts and H3 reference capacity are
checked independently of the API's limits. T2VA ignores connected references
and previous reference context and emits a notice. Other tasks carry previous
context as text; it neither creates attached media nor offsets current labels.

There are two explicit media modes:

- `native`: model capabilities and H3 counts are checked before preparation.
  JPEG copies are supplied separately to `ApiClient(images=...)`; the versioned
  job remains metadata-only. Every image/batch frame must match its manifest
  slot in order. Audio, video and RefMod transport still fail explicitly.
- `descriptions`: every manifest entry needs its own nonempty description,
  including a clip's sound. All descriptions, labels and instructions are sent
  in one request. No captioner is invoked and no fallback is automatic.

Generation settings are typed: `max_tokens`, optional `temperature` and optional
`seed`. Unsupported options fail before authorization or HTTP. Output tokens,
reference count, encoded request bytes and received response bytes have profile
limits. `context_tokens` records model metadata; it is not a tokenizer or a
guarantee that arbitrary input fits the model's context window.

Response formats are `text`, `json_object` and `json_schema`, checked against
profile capabilities. H3 JSON requests use the selected task's fields and an
`alignment` field for I2VA/FL2VA/L2VA. Custom validation uses plain text. Schema
to H3 conversion, semantic checks and the optional repair request are handled
by `api_text.py`. The node exposes all five tasks with mixed references.
Video is explicitly sampled to timestamped image parts in this adapter;
capability checks therefore require `image` for Video entries and `audio` for
each actual Audio entry. Opaque native video uploads are not implemented.

## Requests and results

Native images are embedded in the user message as base64 data URLs, preceded
by their reference labels. User and system text retain their original bytes.
The transport accepts encoded images from the in-memory preparation path, not
workflow paths or URLs. All batch frames are retained under one manifest entry;
frame and byte limits fail rather than truncate. Image copies never reach the
reference output or session record. The optional repair uses the same copies.

`api_multimedia.py` binds decoded RefMod pictures/voices and VIDEO/AUDIO inputs
to the same manifest. A sounding clip contributes an Audio component before
its Video component; all clip sounds precede standalone Audio numbering.
Absent tracks produce a notice and no phantom label. An undecodable track is
an error. RefMod saved appearance/voice text and paired slot IDs are retained;
latents and original filenames never enter the HTTP payload.

Each native audio component has exactly one `PreparedAudio`, encoded as mono
16-bit PCM WAV in an `input_audio` part. Each Video entry has a complete set
of `PreparedVideoFrame` copies with increasing actual presentation timestamps.
These are internal objects, never graph-supplied filenames or URLs. The client
rejects missing, duplicate, extra or mislabelled copies before authorization.
Repair retains the same image/audio copies. With explicit descriptions mode,
no attachments are allowed and each enabled component needs nonempty text.

File video decoding uses PyAV on a connected local file handle or BytesIO.
Nested protocols are disabled and accepted demuxers exclude playlists. UNC
paths, remote URLs and relative paths are rejected. Handles/containers close
on errors and cancellation; the caller's BytesIO cursor is restored. Decoded
file frames are resized and encoded before retention, keeping retained copies
within the request budget. Component VIDEO inputs already own their decoded
batch. The writer preserves trim, presentation times and ComfyUI file crop/
rotation. Video windows respect the target duration and H3 reference ceiling;
audio conversion reports its crop/downmix/resampling. Decode loops yield and
check interruption without subprocesses or detached worker threads.

The original scalar inputs remain declared for existing API workflows.
`reference_layout`, `reference_instructions`, `image_max_side`, `repeat_last`
and `library_pick` originally followed those inputs.
The `references` output is appended after the original eleven text outputs.
The image strip uses DOM widgets and the existing strip/preview helpers;
ordinary inputs remain available if the extension is unavailable. Other
writers' declarations and execution are unchanged. The existing badge helper
also recognizes the API writer.
The stage-5 `media_mode`, `reference_descriptions`, video sampling and audio
copy controls follow that existing prefix. Existing `ref_n` sockets broaden
to the IMAGE/VIDEO/AUDIO/RefMod union with their names and links preserved.

The frontend places the strip first and uses the shared task, ratio and duration
controls. `api_writer_migration.js` restores widget values by name or recorded
order, with the original positional order as a fallback. Existing durations
above the new slider's initial offered maximum retain their value and ceiling.
The ten generation/transport controls, including `repair_once`, appear in `MiniMaxH3ApiWriterOptions`.
Their legacy scalar inputs are hidden on the writer and remain effective until
typed `H3_API_WRITER_OPTIONS` is connected. Connected options override all ten
scalars together; they cannot contain endpoints, headers, models or key sources.
The repair control is appended to the Options widgets and optional in its input
schema, preserving saved positions and API prompts created with nine settings.
It defaults to false; without connected Options, the writer's saved repair flag
remains effective.

`bypass` is evaluated before profile loading, options resolution, guide access,
clip sound probing and media conversion. It returns the stripped original
prompt, empty section/reasoning/findings strings, a zero-request report and
the enabled original reference bundle. It takes precedence over library/session
reuse without replacing the kept answer. Normal request preparation still
validates both local options and the selected profile's capabilities/limits.

Requests use `stream: true` and `stream_options.include_usage: true` by default,
with one choice and no tools or function calls. SSE deltas are accumulated into
the same final text/reasoning/usage envelope; partial text is not published.
The parser accepts LF, CRLF and CR line endings, fragmented UTF-8, comments,
multiline data and a final usage-only event. It requires a completed stop and
`[DONE]`, rejects other choices, tool calls, refusal and malformed events, and
counts all decompressed response bytes against the profile limit. SSE pings do
not extend the total deadline. Missing final usage remains unknown.

An administrator can set `stream: false` for an API that requires a unary JSON
request; `stream_options` is then omitted. Existing version-1 profiles without
this field default to streaming. A complete JSON response from a server that
ignores streaming remains readable, but its upstream cancellation is not
guaranteed. No fallback request is made. Transport selection does not change
the credential's recipient binding.

The client makes one attempt. It does not retry after 429, server errors,
connection failures or ambiguous timeouts; a timed-out request may already have
been processed remotely.

Connection and read-idle timeouts default to 5 and 60 seconds in the core.
A separate total HTTP deadline defaults to 180 seconds, including body reading;
slowly arriving bytes do not extend it. The node's timeout sets the total and
read-idle budgets. A cancellation poll runs every 50 ms while awaiting headers
or content, cancels the request task and closes its connection. There is no
detached worker thread. The remote server may still process or bill a submitted
request after the client disconnects. Streaming lets a forwarding proxy observe
a failed downstream write before the entire generation finishes; a buffered
unary response can hide the disconnect until the end. A proxy without a
separate disconnect monitor can still delay cancellation during prefill until
the upstream sends output. This cancels one connection, never a server process
or all of its slots. Each explicitly enabled repair has its
own deadline. Guide retrieval precedes the API call and uses the existing guide
downloader's timeouts.

`ApiResult` carries final text, separate reasoning, token usage, elapsed time,
the configured model/profile IDs, manifest and notices. Missing usage remains
unknown. A result is a completed API answer, not proof of valid H3 output.
Neither a response body nor authorization headers are included in error text.
There is no request/response logger or provider conversation history. The node
keeps final prompt outputs and safe reference thumbnails for its existing
library/session integration, without raw/system prompts, transport copies or
credential details.

Empty final answers, refusal, tool/function calls, incomplete finish reasons,
invalid JSON envelopes and oversized responses raise `ApiError`. Vendor errors
are reduced to fixed error codes without their exception chains. Model text
remains text. A final answer or reasoning that echoes the actual authorization
token is rejected before it reaches node outputs.

## Reading an answer

Plain `text` output is retained as returned, with fields split by the existing
`fields` helper. Missing sections or H3 semantic problems produce advisory
findings via `checks.review()`, matching the pack's existing behavior.
`json_object` and `json_schema` output must contain exactly the requested
nonempty string fields; duplicates, extra keys, fences and non-finite numbers
are errors. These strings are compiled into ordinary labelled H3 text. Neither
mode executes content. `custom` validation returns final text unchanged and
leaves H3 section outputs empty.

Repair is disabled by default. Opting in permits at most one additional request
for fixable findings, skipped if the format is too incomplete. The first answer
is retained unless the second reduces the existing check score. Refusal,
transport errors, truncation or invalid JSON on the first call do not trigger
repair. Cancellation propagates. Failed repair attempts remain counted, and
aggregate usage stays unknown when a request has no usage result.

The JSON report contains configured profile/model IDs, task, validation/format,
attempt/completion counts, total elapsed time, known aggregate token counts,
manifest, findings and notices. It contains no endpoint, key, request headers,
system message or raw prompt. Preparation duration is separate from API elapsed
time. The report includes sampling windows/timestamps and audio-copy changes.
The separate API providers window does not add serialized node widgets or alter
the existing writers' paths.

The tests in `tests/test_api_writer.py`, `tests/test_api_stream.py`,
`tests/test_api_text.py`, `tests/test_api_images.py`, `tests/test_api_multimedia.py`
and `tests/test_api_secrets.py` exercise HTTP on loopback, image/audio identity,
real container decoding and trim, VFR timestamps, sound exclusion, RefMod pairs,
descriptions, byte limits, node execution,
parsing, reuse, deadlines, cancellation and dummy-key storage. They use no real API keys or
paid endpoints. Windows DPAPI tests require an ordinary current-user process;
a restricted sandbox may not have access to the OS master key.
`tests/test_api_management.py` also covers origin/authentication checks, real
test TLS, encrypted envelopes/replay, conflicts, rollback and server-key AEAD.
These use only dummy secrets and temporary directories. The local
security review and Registry snapshot are kept outside the repository as
required by stage 0; release artifacts still need their stage-6 checks.


---

## FILE: docs/qwen_rewriter.md (https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/blob/HEAD/docs/qwen_rewriter.md)

# Full Qwen rewriters

[Русская версия](qwen_rewriter_RU.md) · [中文版](qwen_rewriter_ZH.md)

**MiniMax-H3 Qwen Rewriter** loads LightX2V's full Qwen checkpoints. It reads the
reference images itself and writes the H3 sections in one generation. Connect
**MiniMax-H3 Qwen Rewriter Options** for sampling, thinking, memory and media settings.

| Checkpoint | Tasks | Original weights | GGUF Q5_K_M / Q4_K_M |
|---|---|---|---|
| Qwen3.5 9B FL2VA | T2VA, I2VA, L2VA, FL2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF) |
| Qwen3.5 9B Ref2VA | Ref2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA-GGUF) |
| Qwen3.8 27B FL2VA | T2VA, I2VA, L2VA, FL2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA-GGUF) |
| Qwen3.8 27B Ref2VA | Ref2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-GGUF) |

The model list has its own `qwen_rewriters` section. Original SFT directories
and GGUF model/projector pairs belong here. The older LoRA nodes keep their own
lists and settings. A full checkpoint does not need a base-model download plus
a separate adapter.

## Quick start

Open [12 - Write with the full Qwen model](../example_workflows/12%20-%20Write%20with%20the%20full%20Qwen%20model.json)
from the template browser. It connects Options, a 9B FL2VA Q5_K_M writer and a
text preview. The model and projector total 7.39 GB; first use can also fetch
the runtime and task guide. No MiniMax video weights are needed to write a prompt.

The example disables thinking and uses a 3000-token output budget for a short
T2VA run. Change the task and connect the required frames, or choose a Ref2VA
checkpoint for mixed references. The node's normal defaults still enable thinking.

![Qwen Options, the full Qwen writer and a completed T2VA prompt](qwen_rewriter_workflow.png)

## Connect references

Connect an IMAGE, VIDEO, AUDIO or decoded RefMod to `ref_0`. Another free input
appears when it is connected. Each input has a checkbox; disabling it keeps the
wire and excludes that reference from the request and the `references` output.
Disabled references are lazy inputs, so the writer does not request their
upstream execution.

The chips are the first controls in the node. Drag them to set the order, click
below the label to enable or disable one, and use **+ instr** for instructions
and descriptions. **Save instructions** applies the edit; **Cancel** discards
it. **Clear** empties the fields and waits for Save.

For the frame tasks, enable exactly the required Picture inputs:

| Task | Pictures, in chip order |
|---|---|
| T2VA | None; connected references are ignored |
| I2VA | Exact first frame |
| L2VA | Exact last frame |
| FL2VA | Exact first frame, then exact last frame |
| Ref2VA | Ordered Pictures, Subjects, Videos and Audio descriptions |

A batch assigned the Picture role contains several images, so it cannot stand
for one exact first/last frame. Select one frame upstream. In Ref2VA, an IMAGE
batch may instead have the Video role; `batch_video_fps` supplies its clock.

Videos reach the writer as sampled frames with source timestamps. The frame
cap, sampling interval, reference-window limit and target duration bound the
copies. RefMods need decoded pictures from their loader's H3 video VAE input.
Their original conditioning is preserved in `references`, ready for Reference
Encode. The report lists the actual frames and timestamps shown to the writer.

**These Qwen checkpoints have no audio encoder.** For AUDIO, fill in a sound
description through **+ instr**. If a Video's sound is enabled, fill in its clip
sound description too. Saved RefMod voice descriptions can supply that text.
An undescribed sound stops before inference. Waveforms are not sent to the
model; the original audio still passes through `references`.

Enabled AUDIO and video-sound chips show a red **!**. Hover over it for the
sound-description instructions, or click it to open the reference editor.
The badge remains after a description is supplied: Qwen reads that text instead
of hearing the recording. Turning off a video's sound removes the badge and
keeps the video enabled.

If a required description is empty, a **Stopped** notification identifies each
reference and its input slot, names the field to fill, and explains how to omit
the sound. This input notice also appears with `self_check=none`.

## Choose weights and settings

SFT uses the original processor and Transformers with `qwen3_5` support; 9B has
been run with Transformers 5.8.0. Keep the complete checkpoint directory,
including tokenizer, processor and all indexed weight shards. The defaults
load BF16 weights. The SFT-only `quantization` control also offers `float16`,
`nf4` and `int8`; the latter two require bitsandbytes.

GGUF uses a local `llama-server` owned by this node. Both the model and its
vision `mmproj` must come from the same checkpoint conversion. The runtime
search and backend selector are shared with the pack's other local nodes.
`gpu_layers` controls offloading; precision comes from the selected GGUF file.
No separately running API provider is needed for this path.

`keep_model_loaded` keeps the SFT model or the owned GGUF server between runs.
**Unload model** and ComfyUI's **Unload Models** release it. Cancellation closes
the owned server or waits for the SFT generation worker to finish interrupting.

The upstream sampling defaults are `temperature=0.7`, `top_p=0.8`, `top_k=20`
and a 16384-token output budget. Thinking is enabled by default. The output
budget includes reasoning: if it ends inside a thought, the node reports an
empty final answer instead of returning analysis as the video prompt. Increase
the budget or disable thinking for shorter runs. `context_size` must hold both
the request and output budget; the default is 32768.

The aspect-ratio picker and duration slider use the pack's shared controls.
A connected `aspect_ratio` overrides the picker. Duration accepts FLOAT and
INT connections; its context menu changes the slider ceiling.

An empty `system_prompt` uses LightX2V's task prefix and the official MiniMax
writing guide. Guides are downloaded into the ComfyUI user directory on first
use, with the same editable copies as the existing writers. A custom system
prompt replaces that entire message. Individual reference instructions remain
next to their images in either case.

## Outputs and reuse

`prompt` contains the complete result. The seven section outputs cover both
checkpoint families; sections not used by the current task are empty.
`reference_assets` records the labels, roles and instructions. `reasoning` is
separate, `findings` contains H3 checks, and `report` records the model, settings,
generation count and prepared media. Native completion counts are explicitly
labelled as re-encoded counts; the GGUF path uses server usage when supplied.

`bypass` returns the original prompt and enabled references without model work.
`repeat_last` reuses the last result or the prompt selected in the library.
The usual save, edit and browse buttons work here. Options selects the prompt
library file and can create another one.

`repair_once` is off by default. When enabled, repairable findings can trigger
one extra generation. The second result replaces the first only if the checks
improve; the report records both attempts.


---

## FILE: docs/qwen_rewriter_RU.md (https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/blob/HEAD/docs/qwen_rewriter_RU.md)

# Qwen Rewriter: полные модели

[English version](qwen_rewriter.md) · [中文版](qwen_rewriter_ZH.md)

**MiniMax-H3 Qwen Rewriter** загружает полные Qwen-чекпойнты LightX2V.
Модель сама читает референсные изображения и за одну генерацию пишет разделы
H3-промта. Подключите **MiniMax-H3 Qwen Rewriter Options**, чтобы настроить
сэмплирование, рассуждения (thinking), память и подготовку медиа.

| Чекпойнт | Задачи | Исходные веса | GGUF Q5_K_M / Q4_K_M |
|---|---|---|---|
| Qwen3.5 9B FL2VA | T2VA, I2VA, L2VA, FL2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF) |
| Qwen3.5 9B Ref2VA | Ref2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA-GGUF) |
| Qwen3.8 27B FL2VA | T2VA, I2VA, L2VA, FL2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA-GGUF) |
| Qwen3.8 27B Ref2VA | Ref2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-GGUF) |

В списке моделей для них выделен раздел `qwen_rewriters`. В него добавляются
каталоги исходных SFT и пары GGUF-модель/проектор. У прежних LoRA-нод свои
списки и настройки. Полный чекпойнт не требует отдельной загрузки базовой
модели и адаптера.

## Быстрый старт

Откройте [12 - Write with the full Qwen model](../example_workflows/12%20-%20Write%20with%20the%20full%20Qwen%20model.json)
в браузере шаблонов. В нём соединены Options, писатель с 9B FL2VA Q5_K_M и
просмотрщик текста. Модель с проектором занимают 7.39 ГБ; при первом запуске
могут также скачаться рантайм и гайд для выбранной задачи. Для написания
промта веса видеогенератора MiniMax не нужны.

В примере thinking выключен, а лимит вывода равен 3000 токенам для короткого
прогона T2VA. Выберите другую задачу и подключите нужные кадры либо возьмите
Ref2VA-чекпойнт для набора разных референсов. В обычных настройках ноды
thinking по умолчанию включён.

![Qwen Options, полный Qwen-рерайтер и готовый T2VA-промт](qwen_rewriter_workflow.png)

## Подключение референсов

Подключите IMAGE, VIDEO, AUDIO или декодированный RefMod к `ref_0`. Рядом
появится следующий свободный вход. У каждого входа есть checkbox: отключение
сохраняет провод, но исключает референс из запроса и выхода `references`.
Отключённые референсы используют ленивые входы: писатель не запрашивает
выполнение подключённых к ним нод.

Чипы расположены в самом верху среди элементов управления. Перетаскивайте их,
чтобы задать порядок, нажимайте под меткой для включения/отключения, а
**+ instr** используйте для инструкций и описаний. **Save instructions**
применяет изменения, **Cancel** отменяет их. **Clear** очищает поля; чтобы
применить очистку, затем нажмите Save.

Для кадровых задач включите ровно столько входов Picture, сколько требуется:

| Задача | Изображения в порядке чипов |
|---|---|
| T2VA | Не нужны; подключённые референсы игнорируются |
| I2VA | Точный первый кадр |
| L2VA | Точный последний кадр |
| FL2VA | Точный первый кадр, затем точный последний |
| Ref2VA | Упорядоченные Picture, Subject, Video и описания Audio |

Батч с ролью Picture содержит несколько изображений и не может считаться
одним точным первым/последним кадром. Выберите один кадр в предыдущей ноде.
В Ref2VA батчу IMAGE можно назначить роль Video; частоту его кадров задаёт
`batch_video_fps`.

Видео поступает писателю как выборка кадров с временными отметками исходника.
Объём выборки ограничен числом кадров, интервалом, допустимым окном референса
и длительностью целевого видео. RefMod нужны декодированные изображения:
подключите H3 video VAE к его загрузчику. Исходное кондиционирование
сохраняется в `references` для передачи в Reference Encode. В отчёте
перечислены фактические кадры и временные отметки, показанные писателю.

**У этих Qwen-чекпойнтов нет аудиоэнкодера.** Для AUDIO заполните описание
звука через **+ instr**. Если у Video включён звук, заполните и описание
звуковой дорожки клипа. Для RefMod может использоваться сохранённое описание
голоса. Звук без описания останавливает выполнение до инференса. Волновая
форма не передаётся модели; исходное аудио остаётся на выходе `references`.

У включённых AUDIO и видеочипов со звуком появляется красный **!**.
Наведите на него курсор для подсказки или нажмите, чтобы открыть редактор
референса. После заполнения описания значок остаётся: Qwen читает этот текст,
а не слушает запись. Если отключить звук видео, значок исчезнет, а само видео
останется включённым.

Если обязательное описание пусто, уведомление **Stopped** укажет каждый
референс, его вход, нужное поле и способ исключить звук из запроса. Такое
уведомление появляется и при `self_check=none`.

## Выбор весов и настроек

SFT использует исходный процессор и Transformers с поддержкой `qwen3_5`;
9B проверена с Transformers 5.8.0. Нужен полный каталог чекпойнта, включая
токенизатор, процессор и все шарды весов, перечисленные в индексе. По
умолчанию веса загружаются в BF16. Параметр `quantization` для SFT также
предлагает `float16`, `nf4` и `int8`; последние два требуют bitsandbytes.

Для GGUF нода запускает собственный локальный `llama-server`. Модель и
визуальный `mmproj` должны происходить из одной конверсии чекпойнта. Поиск
рантайма и выбор бэкенда общие с остальными локальными нодами пака.
`gpu_layers` задаёт число слоёв на GPU, а точность определяется выбранным
GGUF-файлом. Отдельно запущенный API-провайдер для этого пути не нужен.

`keep_model_loaded` сохраняет SFT-модель или собственный GGUF-сервер между
прогонами. **Unload model** и штатная **Unload Models** в ComfyUI выгружают
их. Отмена завершает собственный сервер либо дожидается остановки рабочего
потока генерации SFT.

Исходные настройки сэмплирования: `temperature=0.7`, `top_p=0.8`, `top_k=20`
и лимит вывода 16384 токена. Thinking по умолчанию включён. Его рассуждения
расходуют тот же лимит: если он закончится внутри рассуждения, нода сообщит
об отсутствии итогового промта. Анализ не выдаётся за промт для видео.
Увеличьте лимит либо отключите thinking для коротких прогонов.
`context_size` должен вмещать запрос и лимит вывода; по умолчанию это 32768.

Выбор соотношения сторон и ползунок длительности используют общие элементы
управления пака. Подключённый `aspect_ratio` имеет приоритет над выбором в
ноде. Длительность принимает соединения FLOAT и INT; её контекстное меню
позволяет изменить верхний предел ползунка.

Пустой `system_prompt` использует префикс задачи LightX2V и официальный гайд
MiniMax по написанию промтов. При первом использовании гайды скачиваются в
пользовательский каталог ComfyUI; это те же редактируемые копии, которые
используют остальные писатели. Собственный system prompt заменяет всё это
сообщение. Индивидуальные инструкции референсов в обоих случаях остаются
рядом с их изображениями.

## Выходы и повторное использование

`prompt` содержит полный результат. Семь выходов разделов охватывают оба
семейства чекпойнтов; неиспользуемые в текущей задаче разделы пусты.
`reference_assets` перечисляет метки, роли и инструкции. `reasoning` выдаёт
рассуждения отдельно, `findings` содержит результаты H3-проверок, а `report`
записывает модель, настройки, число генераций и подготовленные медиа. На пути
SFT число токенов завершения явно помечено как полученное повторной
токенизацией; GGUF использует статистику сервера, если тот её передал.

`bypass` возвращает исходный промт и включённые референсы без работы модели.
`repeat_last` повторно выдаёт последний результат или выбранный в библиотеке
промт. Обычные кнопки сохранения, редактирования и просмотра библиотеки
работают и здесь. В Options выбирается файл библиотеки; там же можно
создать новый.

`repair_once` по умолчанию выключен. Если его включить, исправимые замечания
могут вызвать одну дополнительную генерацию. Второй результат заменяет
первый только при улучшении проверок; отчёт учитывает обе попытки.


---

## FILE: docs/qwen_rewriter_ZH.md (https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/blob/HEAD/docs/qwen_rewriter_ZH.md)

# 完整 Qwen 重写模型

[English version](qwen_rewriter.md) · [Русская версия](qwen_rewriter_RU.md)

**MiniMax-H3 Qwen Rewriter** 加载 LightX2V 的完整 Qwen 检查点。
模型直接读取参考图像，在一次生成中写出 H3 提示词的各个部分。
连接 **MiniMax-H3 Qwen Rewriter Options**，即可设置采样、思考模式、
内存和参考素材处理参数。

| 检查点 | 任务 | 原始权重 | GGUF Q5_K_M / Q4_K_M |
|---|---|---|---|
| Qwen3.5 9B FL2VA | T2VA, I2VA, L2VA, FL2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA-GGUF) |
| Qwen3.5 9B Ref2VA | Ref2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA-GGUF) |
| Qwen3.8 27B FL2VA | T2VA, I2VA, L2VA, FL2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-FL2VA-GGUF) |
| Qwen3.8 27B Ref2VA | Ref2VA | [LightX2V SFT](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA) | [pytraveler GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-GGUF) |

模型列表为这些模型单独提供了 `qwen_rewriters` 分区。原始 SFT 目录和
GGUF 模型/投影器配对应放在这里。原有 LoRA 节点使用各自的列表和设置。
完整检查点不需要另行下载基础模型和适配器。

## 快速开始

在模板浏览器中打开 [12 - Write with the full Qwen model](../example_workflows/12%20-%20Write%20with%20the%20full%20Qwen%20model.json)。
模板连接了 Options、使用 9B FL2VA Q5_K_M 的写作节点和文本预览。
模型与投影器合计 7.39 GB；首次运行还可能下载运行时和所选任务的指南。
仅生成提示词时，无需 MiniMax 视频生成模型的权重。

为了快速完成一次简短的 T2VA 生成，示例关闭了 thinking，并把输出上限设为
3000 个 token。可以切换任务并连接所需帧，或选择 Ref2VA 检查点来处理多种
参考素材。节点本身的默认设置仍然开启 thinking。

![Qwen Options、完整 Qwen 重写模型节点和已生成的 T2VA 提示词](qwen_rewriter_workflow.png)

## 连接参考素材

将 IMAGE、VIDEO、AUDIO 或已解码的 RefMod 连接到 `ref_0`，随后会出现下一个
空闲输入。每个输入都有复选框；取消勾选会保留连线，但把该素材排除在请求和
`references` 输出之外。被禁用的参考素材使用惰性输入，写作节点不会请求执行
连接在这些输入上的上游节点。

参考芯片位于节点控件的最上方。拖动芯片可以调整顺序，点击标签下方可启用或
禁用素材，**+ instr** 用于填写指令和描述。**Save instructions** 应用修改，
**Cancel** 放弃修改。**Clear** 清空字段，之后仍需点击 Save 才会应用。

对于指定首尾帧的任务，只启用所需数量的 Picture 输入：

| 任务 | 按芯片顺序提供的图像 |
|---|---|
| T2VA | 不需要；忽略已连接的参考素材 |
| I2VA | 确切的首帧 |
| L2VA | 确切的尾帧 |
| FL2VA | 确切的首帧，然后是确切的尾帧 |
| Ref2VA | 按顺序提供 Picture、Subject、Video 和 Audio 描述 |

Picture 角色的图像批次包含多张图，不能当作一张确切的首帧或尾帧。
请先在上游选择一帧。在 Ref2VA 中，也可以把 IMAGE 批次设为 Video 角色；
`batch_video_fps` 用于指定它的帧率。

视频以采样帧的形式送入模型，每帧带有源视频的时间戳。帧数上限、采样间隔、
参考片段的时间窗口上限和目标视频时长共同限制采样量。RefMod 需要解码后的
图像，因此应在其加载节点上连接 H3 video VAE。原始条件数据保留在
`references` 输出中，可继续送入 Reference Encode。报告会列出模型实际看到
的帧和时间戳。

**这些 Qwen 检查点没有音频编码器。** 对于 AUDIO，请通过 **+ instr**
填写声音描述。若启用了 Video 的声音，还需要填写该片段的音轨描述。
RefMod 中保存的语音描述也可提供这段文字。缺少描述时，节点会在推理前停止。
音频波形不会发送给模型；原始音频仍通过 `references` 输出传递。

启用的 AUDIO 芯片和带声音的视频芯片会显示红色 **!**。悬停可查看声音描述的
填写说明，点击则打开参考素材编辑器。填写描述后，标记仍然保留：Qwen 读取
的是这段文字，并不聆听录音。关闭视频的声音后，标记会消失，视频本身仍保持启用。

如果必填描述为空，**Stopped** 通知会逐一指出相关素材、输入插槽、应填写的
字段，以及如何从请求中排除声音。即使设置了 `self_check=none`，这类输入
通知也会显示。

## 选择权重与设置

SFT 路径使用原始处理器和支持 `qwen3_5` 的 Transformers；9B 已在
Transformers 5.8.0 上运行验证。请保留完整检查点目录，包括 tokenizer、
processor 和索引中列出的所有权重分片。默认加载 BF16 权重。
仅用于 SFT 的 `quantization` 还可选择 `float16`、`nf4` 和 `int8`；
后两项需要 bitsandbytes。

GGUF 路径由节点启动自己的本地 `llama-server`。模型和视觉 `mmproj` 必须
来自同一检查点的同一次转换。运行时查找方式和后端选择与节点包中的其他本地
节点共用。`gpu_layers` 控制加载到 GPU 的层数，精度由所选 GGUF 文件决定。
此路径不需要另行运行 API 服务。

`keep_model_loaded` 会在两次运行之间保留 SFT 模型或节点自己的 GGUF 服务。
**Unload model** 和 ComfyUI 的 **Unload Models** 都可以将其卸载。
取消任务会关闭节点启动的服务，或等待 SFT 生成线程完成中断。

上游采样默认值为 `temperature=0.7`、`top_p=0.8`、`top_k=20`，输出上限为
16384 个 token，默认开启 thinking。思考过程也占用输出额度：如果额度在
思考结束前耗尽，节点会报告没有最终提示词，不会把分析过程当作视频提示词。
可以提高上限，或在短任务中关闭 thinking。`context_size` 必须同时容纳
输入和输出额度，默认值为 32768。

宽高比选择器和时长滑块使用节点包的共用控件。连接到 `aspect_ratio` 的值
优先于节点内的选择。时长接受 FLOAT 和 INT 连线，可通过其右键菜单调整
滑块上限。

`system_prompt` 为空时，节点使用 LightX2V 的任务前缀和 MiniMax 官方写作
指南。首次使用时，指南下载到 ComfyUI 用户目录，与其他写作节点使用同一份
可编辑副本。自定义 system prompt 会替换整条系统消息。无论采用哪种方式，
各参考素材的单独指令仍放在对应图像旁边。

## 输出与复用

`prompt` 包含完整结果。七个分段输出覆盖两个检查点系列；当前任务不用的
部分为空。`reference_assets` 记录标签、角色和指令。`reasoning` 单独输出
思考过程，`findings` 包含 H3 检查结果，`report` 则记录模型、设置、生成次数
和准备好的参考素材。SFT 路径中的输出 token 数明确标注为重新分词后的计数；
GGUF 路径在服务提供用量统计时采用该统计。

`bypass` 原样返回输入提示词和启用的参考素材，不运行模型。`repeat_last`
复用上一次结果或提示词库中选中的提示词。常用的保存、编辑和浏览按钮在这里
同样可用。Options 可以选择提示词库文件，也可以新建一个文件。

`repair_once` 默认关闭。开启后，可修复的检查问题可能触发一次额外生成。
只有检查结果改善时，第二次结果才会替换第一次；报告会记录两次尝试。


---

# ISSUES / PRs of pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI


### #24 [closed] ISSUE: Captioners: add a thinking-off toggle (the pinned llama-server already supports --reasoning off)
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/24  by LacklusterOpsec 2026-11-07T04:05:41Z

Version 0.33.0 on Windows (RTX 5060 Ti 16 GB), captioner Qwen3.8-27B
IQ3_XS GGUF + its mmproj. Its chat template is the stock Qwen3.8 one:
`reasoning_effort` defaults to `xhigh`.

The pack's GGUF caption path has no way to turn model thinking off, so the
captioner can spend most of its budget deliberating before it writes a caption.
The writer path already asks for the opposite: it renders the template itself
and passes `enable_thinking=False`. The CLIP caption route also passes
`thinking=False`; this issue is only about the GGUF route.

As noted in #8, `llama-mtmd-cli` has no switch for this. The server path does,
and the pack's pinned release is not the blocker. I downloaded the official
`llama-b10310-bin-win-cpu-x64.zip` and checked it directly:

    llama-server --reasoning off --version      # accepted, b10310
    llama-mtmd-cli --reasoning off --version    # error: invalid argument: --reasoning

The b10310 source shows `--reasoning off` setting
`default_template_kwargs["enable_thinking"] = "false"`, and the server passes
that into the chat template, so it disables the thought channel rather than just
being accepted. The same split holds on a local b12180 build.

The server exposes:

    -rea, --reasoning [on|off|auto]
        --reasoning-budget N
        --reasoning-effort LEVEL
        --chat-template-kwargs STRING

`server_engine.build_command` passes none of them, so the server keeps the
template default and the model thinks. The existing `NO_REASONING` line in the
caption prompt is only a prompt-level request; it does not disable the thought
channel.

Why it matters beyond untidiness: every GGUF caption call is hard-capped at 1024
tokens (`nodes.py:2584`, `multi_caption.py:488`, `universal.py:909`), and raising
`max_new_tokens` on the Options node does not lift it — the caption calls clamp
it themselves. If the thought does not close before the cap, there is no closing
think tag in the completion for `answer_only` to cut at. On the server path the
template puts the opening think tag in the prompt, not in the completion, so the
completion can be all deliberation, or empty after the server extracts it; either
way the caption is lost. Even when the thought closes, those tokens are spent
before the caption and can leave it truncated.

Could the caption path get a thinking toggle, or just handle it off for captions?
A suggested shape:

- Add `captioner_thinking` (`auto` / `off` / `on`) to the Options node, or a
  boolean on the caption nodes. `auto` could default to off for captions, since
  the caption prompt already ends with "Do not think out loud"; writers stay
  unchanged, and `on` lets people opt back in.
- On the server path (when `llama-server` is found beside the captioner), add
  `--reasoning off` when the binary supports it. The pinned b10310 already does;
  a one-time `--help` probe would keep older local builds working, where the
  flag is unknown and the server would otherwise exit. `--jinja` is enabled by
  default in b10310, so no extra flag is needed there.
- On the single-asset path, b10310's `llama-mtmd-cli` still rejects
  `--reasoning`. When thinking is off, either force the server
  (`MINIMAX_H3_MTMD_SERVER=always`, which already exists) or pass the flag if a
  future mtmd adds it.
- `--reasoning-format` is not a fix: it only changes where the thought is
  reported (or hides it); the tokens are still generated and the 1024-token
  budget is still spent.

Expected: with thinking off, the first token is the caption, the 1024 tokens are
available for the caption, and a strip stops paying for deliberation.
Non-thinking captioners are unaffected.


> **Comment by pytraveler (2026-11-07T22:22:01Z):**

Implemented for 0.34.0.

`captioner_thinking` is now in **MiniMax-H3 Rewriter Options**: `off` (default), `auto` (use the model template default), or `on`. It applies to GGUF captions in Reference Caption, Multi Reference Caption and Universal Writer.

Each executable's supported flags are read through `--help` and cached. The pack sends `--reasoning off/on` only when supported, with `--jinja` when available. If the CLI cannot apply the requested mode, a compatible `llama-server` beside it is used even for a single reference. `MINIMAX_H3_MTMD_SERVER=never` still takes precedence. Older builds retain the CLI fallback with a warning if the setting cannot be applied.

The 1024-token cap stays in place. Empty GGUF captions now raise an error rather than becoming empty reference descriptions. Writer generation and the CLIP caption route keep their existing thinking behaviour.

Verified on Windows with a local Qwen3.8-27B Q5_K_P build, including single/batched references and server cleanup on cancellation. Thanks for checking b10310.



### #23 [open] ISSUE: Is there anyway to prevent from the writer to offload my diffusion model from system ram?
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/23  by ran2me3 2026-11-02T22:12:53Z

I have 128 GB of RAM and plenty of available memory to accommodate both models simultaneously. I'm also using the keep-model-loaded option, but whenever the workflow starts, my diffusion model gets completely offloaded.

Is there any way to prevent this behavior and keep the diffusion model loaded?

> **Comment by pytraveler (2026-11-04T11:49:31Z):**

There isn't currently an option to keep the diffusion model in RAM while the writer frees the GPU for itself.

Before loading the writer on the same GPU, the pack calls ComfyUI's [unload_all_models()](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/blob/main/minimax_h3_rewriter/runner.py#L183-L210), regardless of how much system RAM is available. keep_model_loaded controls whether the writer itself stays loaded after generation; it doesn't prevent that initial unload.

Depending on your ComfyUI version and how the diffusion model is loaded, a full unload can also release its RAM buffers. In the current [dynamic model patcher](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_patcher.py), full unloading includes a separate RAM cleanup step. That could explain what you're seeing, but I'd need your version and logs to confirm.

This is a limitation of the pack's current memory handling. Running the writer through llama.cpp on GPU doesn't inherently require removing the diffusion model from RAM. Preserving those RAM buffers while freeing VRAM would require changing our unload path, and that hasn't been implemented or tested yet.

For now, connect MiniMax-H3 Rewriter Options and set device to cpu, or to a second GPU if you have one. Both choices skip the writer's unload call. CPU generation will be slower. Setting only gpu_layers=0 with device=auto won't skip that call.

If you're reusing an existing prompt, turn on repeat_last. Once the node has an answer stored, it returns that text without running the writer again.

Could you share your ComfyUI version, the writer node and model you're using, and the console output around the diffusion model's unload and reload?



### #22 [open] ISSUE: Ref2VA: Rewriter hallucinating references
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/22  by sixpt 2026-10-26T19:07:38Z

First, thanks, I've been having a lot of success using this to write/rewrite prompts in the correct structure!

I typically use Ref2VA. I do find that, much of the time, the MiniMax-H3 Prompt Writer (Ref2VA) node hallucinates references. If I provide a <Subject 1> description and describe <Picture 1> as the character reference in the reference_assets section, the final prompt output often includes references I DIDN'T specify, such as referencing "<Video 1>" and "<Audio 1>" for cut/pacing/sound when I didn't specify any video/audio inputs. Any advice?

> **Comment by pytraveler (2026-10-26T20:07:37Z):**

The model is copying the guide. The Ref2VA writer puts MiniMax's [full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) into the system prompt as it is, and the guide shows `retention_analysis` with one sample entry per kind of label. Two of those samples are exactly what you got:

```
<Video 1> (cut and pacing structure): weak_reference - ...
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.
```

The complete example at the end of the guide has a clip and a sound in it too. A writer model that takes all of this for part of the format fills those entries in, whether or not `reference_assets` lists a video or an audio. The system prompt does tell it "Do not invent assets that are not listed there", and not every model sticks to that.

The pack should have caught this. It didn't, and that part is a bug. The self-check compares the tags in an answer with the references the node was given, but the Ref2VA writer never told it what was in `reference_assets`. So `<Video 1>` looked like any other Ref2VA tag (the task takes up to three clips), and the node didn't flag it. In 0.32.2 the writer reads the labels in `reference_assets`, and an answer that cites `<Video 1>` or `<Audio 1>` without a matching line gets a warning on the node. With `fix_once` on in **MiniMax-H3 Rewriter Options**, a warning like that also makes the node write the prompt once more, with the finding added as a rule to follow.

Two things you can try now:

- Which writer model are you using? The node's `model` tooltip says "a 4B will hold the format but a 9B keeps the labels consistent across all six sections". If yours is 4B or smaller, a 9B is the first thing to try.
- Say it in `prompt` as well, e.g. "The only references are Subject 1 and Picture 1. There is no reference video or audio; all sound is generated." I haven't measured how much that helps, but it costs nothing.

Thanks for quoting what came back. "Cut/pacing" pointed straight at the guide.




### #21 [open] ISSUE: cannot install it from extensions because it is currently flagged
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/21  by axiomgraph 2026-10-24T05:58:50Z

<img width="432" height="367" alt="Image" src="https://github.com/user-attachments/assets/a236c0a0-eeb9-485d-b9c1-d135105d4a07" />

> **Comment by pytraveler (2026-10-24T10:53:29Z):**

I’m looking into it. Once I’m done, I’ll let you know what the issue is.


> **Comment by pytraveler (2026-10-24T15:09:51Z):**

It's the Comfy Registry's automated security review that flagged it, not anything on your side. Versions 0.27.1 to 0.30.0 carry the reason `rce-remote-code`.

The registry doesn't say which line it means, but one path in the pack fit, and the registry was right about it. The Options node had a `trust_remote_code` switch. It was off by default, and while it was off the pack refused any model that ships its own Python. But it was a widget, so a workflow could turn it on. And the pack's model list can be extended through ComfyUI's API, which has no login. On a ComfyUI reachable from the network, that was enough to make it download a model and run the code that came with it.

0.31.0, out today, removes the switch. The pack doesn't run code from a model at all now. None of the models in its lists needed it, so nothing you use changes. Workflows saved with the old Options node still load, including the two in your ComfyUIWorkflow repo; I opened both.

The registry still lists 0.31.0 as flagged, but not for that reason any more. What's left are 12 info-level scanner matches, such as starting the llama.cpp binaries and reading `HF_ENDPOINT`, which the pack can't work without. I've asked for a manual review: [registry-backend issue 302](https://github.com/Comfy-Org/registry-backend/issues/302). How long that takes is up to them.

Until then, ComfyUI-Manager only refuses a flagged version when ComfyUI listens on a network address. The terminal shows the two ways around it: listen on 127.0.0.1 only, or set `allow_flagged_nodepack_install = true` in Manager's `config.ini` on a network you trust.

Thanks for reporting it.




### #20 [closed] ISSUE: Does omni not support transformers 4.57.6?
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/20  by hben35096 2026-10-11T19:07:08Z

```
[ERROR] !!! Exception during processing !!! Unrecognized configuration class <class 'transformers.models.qwen2_5_omni.configuration_qwen2_5_omni.Qwen2_5OmniConfig'> for this kind of AutoModel: AutoModelForImageTextToText.
Model type should be one of AriaConfig, AyaVisionConfig, BlipConfig, Blip2Config, ChameleonConfig, Cohere2VisionConfig, DeepseekVLConfig, DeepseekVLHybridConfig, Emu3Config, EvollaConfig, Florence2Config, FuyuConfig, Gemma3Config, Gemma3nConfig, GitConfig, Glm4vConfig, Glm4vMoeConfig, GotOcr2Config, IdeficsConfig, Idefics2Config, Idefics3Config, InstructBlipConfig, InternVLConfig, JanusConfig, Kosmos2Config, Kosmos2_5Config, Lfm2VlConfig, Llama4Config, LlavaConfig, LlavaNextConfig, LlavaNextVideoConfig, LlavaOnevisionConfig, Mistral3Config, MllamaConfig, Ovis2Config, PaliGemmaConfig, PerceptionLMConfig, Pix2StructConfig, PixtralVisionConfig, Qwen2_5_VLConfig, Qwen2VLConfig, Qwen3VLConfig, Qwen3VLMoeConfig, ShieldGemma2Config, SmolVLMConfig, UdopConfig, VipLlavaConfig, VisionEncoderDecoderConfig.
[ERROR] Traceback (most recent call last):
File "/root/ComfyUI/execution.py", line 547, in execute
output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/root/ComfyUI/execution.py", line 352, in get_output_data
return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/root/ComfyUI/execution.py", line 326, in _async_map_node_over_list
await process_inputs(input_dict, i)
File "/root/ComfyUI/execution.py", line 314, in process_inputs
result = f(**inputs)
^^^^^^^^^^^
File "/root/ComfyUI/comfy_api/internal/__init__.py", line 149, in wrapped_func
return method(locked_class, **inputs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/root/ComfyUI/comfy_api/latest/_io.py", line 2208, in EXECUTE_NORMALIZED
to_return = cls.execute(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/root/ComfyUI/custom_nodes/MiniMax-H3-Prompt-Rewriter-ComfyUI/minimax_h3_rewriter/writer_omni.py", line 897, in execute
text = saved or rewrite_omni(
^^^^^^^^^^^^^
File "/root/ComfyUI/custom_nodes/MiniMax-H3-Prompt-Rewriter-ComfyUI/minimax_h3_rewriter/writer_omni.py", line 564, in rewrite_omni
return _with_transformers(
^^^^^^^^^^^^^^^^^^^
File "/root/ComfyUI/custom_nodes/MiniMax-H3-Prompt-Rewriter-ComfyUI/minimax_h3_rewriter/writer_omni.py", line 389, in _with_transformers
return engine.rewrite(
^^^^^^^^^^^^^^^
File "/root/ComfyUI/custom_nodes/MiniMax-H3-Prompt-Rewriter-ComfyUI/minimax_h3_rewriter/engine.py", line 734, in rewrite
tokenizer, model = load(
^^^^^
File "/root/ComfyUI/custom_nodes/MiniMax-H3-Prompt-Rewriter-ComfyUI/minimax_h3_rewriter/engine.py", line 342, in load
model = model_class.from_pretrained(base_dir, **model_kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/root/miniconda3/lib/python3.11/site-packages/transformers/models/auto/auto_factory.py", line 607, in from_pretrained
raise ValueError(
ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_omni.configuration_qwen2_5_omni.Qwen2_5OmniConfig'> for this kind of AutoModel: AutoModelForImageTextToText.
Model type should be one of AriaConfig, AyaVisionConfig, BlipConfig, Blip2Config, ChameleonConfig, Cohere2VisionConfig, DeepseekVLConfig, DeepseekVLHybridConfig, Emu3Config, EvollaConfig, Florence2Config, FuyuConfig, Gemma3Config, Gemma3nConfig, GitConfig, Glm4vConfig, Glm4vMoeConfig, GotOcr2Config, IdeficsConfig, Idefics2Config, Idefics3Config, InstructBlipConfig, InternVLConfig, JanusConfig, Kosmos2Config, Kosmos2_5Config, Lfm2VlConfig, Llama4Config, LlavaConfig, LlavaNextConfig, LlavaNextVideoConfig, LlavaOnevisionConfig, Mistral3Config, MllamaConfig, Ovis2Config, PaliGemmaConfig, PerceptionLMConfig, Pix2StructConfig, PixtralVisionConfig, Qwen2_5_VLConfig, Qwen2VLConfig, Qwen3VLConfig, Qwen3VLMoeConfig, ShieldGemma2Config, SmolVLMConfig, UdopConfig, VipLlavaConfig, VisionEncoderDecoderConfig.
```

transformers
Version: 4.57.6

But I had no issues when using lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni//infer.py
python infer.py \
    --task t2av \
    --prompt "A cinematic fox walks through a snowy forest while distant branches crack in the wind." \
    --duration 15 \
    --resolution 16:9 \
    --output outputs/t2av.json \
    --model-path ../Qwen2.5-Omni-7B/

> **Comment by pytraveler (2026-10-12T10:07:22Z):**

It does. 4.57.6 is supported, and the error was a bug in the node, fixed in 0.28.1. Nothing needs to change on the Transformers side.

Qwen2.5-Omni-7B is a wrapper around three models (thinker, talker and token2wav), and the node loads only the thinker, the part that writes text. It looked up the thinker's class in Transformers' image-text-to-text auto mapping. Transformers 5 lists `qwen2_5_omni_thinker` there, but 4.57 doesn't list it anywhere, so the node fell back to `AutoModelForImageTextToText` with the wrapper's config, and that's the error you got. `infer.py` imports `Qwen2_5OmniThinkerForConditionalGeneration` directly, which is why it worked for you. 4.57 has that class too. When no mapping names it, the node now finds it in the `qwen2_5_omni` module by the config it's built from.

Behind that was a second 4.57 problem, and you'd have hit it as soon as you connected a picture. In 4.57, `Qwen2_5OmniProcessor.apply_chat_template` expects the system message as a list of parts and fails on a plain string with `TypeError: string indices must be integers`. The processor now gets every message as a list of parts. The prompt it renders is the same as before.

I ran Qwen2.5-Omni-7B with the Omni LoRA on 4.57.6 and on 5.8, T2AV and I2AV. With the same seed, both versions wrote exactly the same text.

On 4.57 the log will still show `System prompt modified, audio output may not work as expected`. Transformers prints that whenever the system prompt isn't Qwen's default. It's about speech output, which the node doesn't use, so you can ignore it.

Thanks for trying `infer.py` side by side, that narrowed it down to how the class gets picked.



> **Comment by hben35096 (2026-10-13T13:14:20Z):**

Wow, thank you so much! The issue is finally solved!
Your node provides a fantastic user experience.
But there's something I've been thinking about. Many nodes that use models have a keep_model_loaded option to control whether the model stays loaded in memory, right?
That means once we've got the result we want, if we want to free up VRAM by unloading the model, we have to turn keep_model_loaded off. And here's the problem: doing so causes the text generation to run all over again, even with the same seed, producing exactly the same text as before, right?
So why hasn't anyone thought of this? Why not separate the model unloading functionality into its own dedicated node and place it after the text inference node?
That way, the text inference node would simply pass a variable identifying the model to the unloading node, and the unloading node would handle the straightforward task of unloading it.
Wouldn't that be a much more logical and efficient design?


> **Comment by pytraveler (2026-10-13T21:13:34Z):**

Yes, that's what happens, and 0.29.0 fixes it, though not with a new node.

`keep_model_loaded` is an input, so changing it changes the node's cache key and ComfyUI runs the node again. The model isn't reloaded, but the text is generated a second time, and only then is the model unloaded.

The node you describe already exists: KJNodes has `VRAM Debug` and Easy-Use has `easy cleanGpuUsed`. Both call ComfyUI's `unload_all_models`, and that only knows the models ComfyUI loaded itself. This pack's writer isn't one of them, so those nodes never touched it. Neither did ComfyUI's own Unload Models button.

In 0.29.0 the pack hooks into `unload_all_models`, so the button and those nodes free the writer too. That gives you your design with a node you may already have. Put `VRAM Debug` after the writer, and when you're happy with the text turn its `unload_all_models` on. Only that node runs. The writer's text comes from the cache.

There's also an `Unload model` button on every node that has `keep_model_loaded`. It unloads at once, with no run at all. If you press it while the node is still writing, it waits for that run to finish.

And `repeat_last` on with `keep_model_loaded` off now returns the kept text and unloads the model, so those two switches do what you were after without anything being written twice.

Two things to know. With `--disable-smart-memory` ComfyUI unloads everything after every run, and that now includes a kept writer. And none of this matters for a GGUF run through the llama.cpp binaries, which never kept a model in the first place.

I tested it on Qwen2.5-Omni-7B through Transformers (20.9 GB of VRAM in use, 1.8 GB after unloading) and on a GGUF through llama-cpp-python. ComfyUI needs a restart after you update.

Thanks for bringing it up, the hook wouldn't exist otherwise.



> **Comment by hben35096 (2026-10-14T03:27:18Z):**

I'm honestly amazed by how quickly you implemented this! This is such a pleasant surprise. You've done an outstanding job—better than I ever expected!


> **Comment by hben35096 (2026-10-14T06:15:15Z):**

By the way, is there any chance that a system_prompt input socket could be added in the future? For example, a system_prompt optional input for the Text Reasoning node, allowing users to decide whether to provide their own system prompt or use the default one.
I think this would greatly increase the flexibility of the models and give users much more freedom, without being restricted by the default system prompt.


> **Comment by pytraveler (2026-10-14T21:16:54Z):**

There's no node called Text Reasoning in this pack, so I'm not sure which one you mean, but the answer covers all of them.

The writer nodes already have it. Both `MiniMax-H3 Prompt Writer` nodes, the `MiniMax-H3 Universal Writer` and the `MiniMax-H3 Prompt Reducer` take an optional `system_prompt`. It's a text widget, so you can type into it or wire any STRING output to it. Empty means the default; anything in it replaces the default entirely. To start from the default instead of from nothing, `MiniMax-H3 Guide Prompt (any LLM)` outputs it for you to edit. The README covers this under [`system_prompt` — pointing the writer at another model](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI#system_prompt--pointing-the-writer-at-another-model).

The LoRA rewriters (27B, 8B, Omni, and the Universal Rewriter that runs them) don't have one, on purpose. Each adapter was trained against one fixed system prompt, copied byte for byte from the adapter's own repository, and the output format comes from that pairing. Some of it is quoted back verbatim: the alignment line that opens an I2AV or FL2AV answer, which MiniMax-H3 reads. With a different system prompt the LoRA runs on input it was never trained on. If you want your own instructions, a writer node is the way to do it.

I'm closing this one, since the 4.57.6 problem it was opened for is fixed. If something else comes up, please open a new issue for it.




### #19 [closed] ISSUE: Quick question on your MiniMax-H3 Universal Rewriter Node
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/19  by mostlyhuman 2026-09-29T19:11:52Z

Hello there, really enjoying your prompt nodes! question though, on the universal rewriter, if I choose Omni and then Ref2VA, why are there only options to connect a First and Last frame, instead of all the references similar to how the node titled Check It Against The H3 Rules allows you to connect all your references as inputs, or am I possibly doing something completely wrong? heh, I figured since the Omni model has vision it would make sense for it to visually analyze all ones references.  Thank you for any help/guidance, I am very grateful! :-)

> **Comment by pytraveler (2026-09-30T17:00:07Z):**

You're not doing anything wrong. That's simply as many as the Universal Rewriter takes. On the Omni tab with Ref2VA it reads four sockets, and all four go to the model as references, numbered in socket order. With all four connected:

- `first_frame` is `<Picture 1>`
- `last_frame` is `<Picture 2>`
- `reference_video` is `<Video 1>`
- `reference_audio` is `<Audio 1>`

On Ref2VA the two frame sockets are just pictures, and neither is pinned to the start or the end of the video. So this node tops out at two pictures, one clip and one sound.

The limit is deliberate. Labels are numbered by order, and with four fixed sockets the order can just be the socket order. Past that you'd want to set the order yourself, and that needs something you can drag.

That's what **MiniMax-H3 Prompt Rewriter Omni (sees and hears)** is for. It runs the same Omni LoRA and offers the same base models, so whatever you downloaded for the Omni tab gets reused. There the task is called `REF2AV`. Its `references` input adds a socket every time you connect one, up to 12, and each reference shows up as a square in the strip: drag the squares to change the numbering, click one to switch it off. At most 9 pictures, 3 clips and 3 sounds can be switched on at once, and an over-full strip is refused before anything loads. README: [Prompt Rewriter Omni](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI#minimax-h3-prompt-rewriter-omni-sees-and-hears).

Two nodes that go with it:

- [Reference Slots](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI#minimax-h3-reference-slots): wire the Omni node's `references` output into it, and its outputs into `MiniMaxH3ReferenceToVideo` in the same order. That way the generator gets each reference on the socket the prompt's number points to, even after you reorder the strip.
- [Reference Adapter](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI#minimax-h3-reference-adapter), if your images come in as one batch or a list. It splits them onto separate sockets the Omni node can take.

As for "Check it against the H3 rules" (the MiniMax-H3 Prompt Check node): it takes any number of references because no model ever looks at them. It only reads what kind each one is and how many there are, so it can tell you when the prompt cites a picture that isn't connected, or skips one that is.



> **Comment by mostlyhuman (2026-09-30T19:41:55Z):**

ahh okay sounds workable enough. thank you so much. I really appreciate it!
:-)

On Wed, Sep 30, 2026 at 12:00 PM pytraveler ***@***.***>
wrote:

> *pytraveler* left a comment
> (pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI#19)
> <https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/19#issuecomment-5915887921>
>
> You're not doing anything wrong. That's simply as many as the Universal
> Rewriter takes. On the Omni tab with Ref2VA it reads four sockets, and all
> four go to the model as references, numbered in socket order. With all four
> connected:
>
>    - first_frame is <Picture 1>
>    - last_frame is <Picture 2>
>    - reference_video is <Video 1>
>    - reference_audio is <Audio 1>
>
> On Ref2VA the two frame sockets are just pictures, and neither is pinned
> to the start or the end of the video. So this node tops out at two
> pictures, one clip and one sound.
>
> The limit is deliberate. Labels are numbered by order, and with four fixed
> sockets the order can just be the socket order. Past that you'd want to set
> the order yourself, and that needs something you can drag.
>
> That's what *MiniMax-H3 Prompt Rewriter Omni (sees and hears)* is for. It
> runs the same Omni LoRA and offers the same base models, so whatever you
> downloaded for the Omni tab gets reused. There the task is called REF2AV.
> Its references input adds a socket every time you connect one, up to 12,
> and each reference shows up as a square in the strip: drag the squares to
> change the numbering, click one to switch it off. At most 9 pictures, 3
> clips and 3 sounds can be switched on at once, and an over-full strip is
> refused before anything loads. README: Prompt Rewriter Omni
> <https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI#minimax-h3-prompt-rewriter-omni-sees-and-hears>
> .
>
> Two nodes that go with it:
>
>    - Reference Slots
>    <https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI#minimax-h3-reference-slots>:
>    wire the Omni node's references output into it, and its outputs into
>    MiniMaxH3ReferenceToVideo in the same order. That way the generator
>    gets each reference on the socket the prompt's number points to, even after
>    you reorder the strip.
>    - Reference Adapter
>    <https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI#minimax-h3-reference-adapter>,
>    if your images come in as one batch or a list. It splits them onto separate
>    sockets the Omni node can take.
>
> As for "Check it against the H3 rules" (the MiniMax-H3 Prompt Check node):
> it takes any number of references because no model ever looks at them. It
> only reads what kind each one is and how many there are, so it can tell you
> when the prompt cites a picture that isn't connected, or skips one that is.
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/19?email_source=notifications&email_token=AC7TKCW2L76GK7YWQE33RIT5RU325A5CNFSNUABFM5UWIORPF5TWS5BNNB2WEL2JONZXKZKDN5WW2ZLOOQXTKOJRGU4DQNZZGIY2M4TFMFZW63VGMF2XI2DPOKSWK5TFNZ2KYZTPN52GK4S7MNWGSY3L#issuecomment-5915887921>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AC7TKCWOESVPWPDPSNKZ52T5RU325AVCNFSNUABGKJSXA33TNF2G64TZHMYTGMRWGQ4TOMJVGY5US43TOVSTWNJWGM2TMNRYHE4TPILWAI>
> .
> You are receiving this because you authored the thread.Message ID:
> ***@***.***
> .com>
>




### #18 [closed] ISSUE: Support MTP (65-block) Qwen3.8-27B GGUFs, or document the incompatibility
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/18  by LacklusterOpsec 2026-09-28T22:10:42Z

Thanks for the pack — the writer nodes work great on my setup. I'd like to use the 27B LoRA route with the newer MTP/draft-head quants, and it's blocked by the shape check. Asking whether that check can be relaxed, or whether a 65-block variant is planned.

### Setup

- Pack version **0.27.0** (`minimax_h3_rewriter`), Windows, ComfyUI.
- Base models I have locally, all `qwen35`, width **5120**, **65 blocks**:
  - `Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf`
  - `Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf`
  - `RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_S-mtp.gguf`
  - `RVN-IQ3_S-multilingual-mtp.gguf`

### What happens

With `use_lora` on, `_resolve_adapter` / `gguf_problem()` refuses them:

```text
'…-IQ3_S-mtp.gguf' has 65 blocks of width 5120; the adapter was trained on 64
of 5120 (Qwen3.6-27B). It shares the 'qwen35' architecture but not the shape,
so llama.cpp cannot attach the LoRA to it — the model will run, but as a plain
one with no rewriter.
```

In the model list they show as `(wrong size for the adapter)`.

### Why I think it may be benign

The extra block is the MTP / NextN speculative-decoding head, not a transformer layer. Reading the tensors:

- `blk.0` … `blk.63` are the normal 64 layers (the same modules the adapter targets).
- `blk.64.*` is only the draft head: `blk.64.nextn.eh_proj.weight`, `blk.64.nextn.enorm.weight`, `blk.64.nextn.hnorm.weight`, `blk.64.nextn.shared_head_norm.weight`.
- These models are run with `--spec-type draft-mtp` in llama.cpp, which is where the 65th block comes from.

So all 64 adapter-targeted layers exist; the 65th has no LoRA targets either way.

### Asks

1. If llama.cpp maps LoRA tensors by name and ignores the extra block, can `GGUF_BLOCK_COUNT` accept `>= 64` (or match on the first N blocks) so MTP builds are allowed? If that attaches and rewrites correctly, it opens the whole GSQ-RCO / MTP family.
2. If it does **not** work — e.g. llama.cpp rejects the shape, or the merge is wrong — could the README say so explicitly? Right now "it shares the architecture but not the shape" reads like a hard mathematical mismatch, and it isn't obvious that the only difference is a draft head the adapter never touches.
3. If a purpose-converted 65-block adapter is the real answer, is one planned at `pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF`?

> **Comment by pytraveler (2026-09-29T13:31:12Z):**

In 0.27.1 your files pass the shape check. Any of them whose header names Qwen3.8 rather than Qwen3.6 also gets a warning on every run, and that part matters more.

**1. The draft head.** You're right about the mechanism. llama.cpp looks up each LoRA tensor by name and compares shapes one tensor at a time; the block count never comes into it. I checked b10310, the build the pack downloads: it reads `qwen35.nextn_predict_layers`, runs the 64-block trunk, and doesn't load block 64 at all unless you pass `--spec-type draft-mtp`. So the check now subtracts `nextn_predict_layers` from `block_count` instead of accepting `>= 64`, and a 65-block model without a draft head is still refused. One correction to your reading: `blk.64` is a full attention + FFN block plus the `nextn.*` tensors, not only the four you listed. That changes nothing, since the adapter has no tensors for it.

The pack never passes `--spec-type draft-mtp`, so an MTP build runs here exactly like a plain one: the head stays in the file and isn't loaded, and there's no speed-up from it. It can't just be switched on, either. The rewriter runs through the llama-cpp-python wheel or `llama-completion`; at b10310 `llama-completion` has no speculative decoding at all, only `llama-server` does, and the pack doesn't ask the wheel for it. I'm also not sure it would help much. The head guesses the base model's next token, it has no LoRA, and it has never seen H3 format, so I don't know how many of its guesses would be accepted. The text wouldn't change either way, only the speed.

**The base.** This is what the block count was hiding. The adapter was trained on Qwen3.6-27B: [lightx2v's model card](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA) gives `base_model: Qwen/Qwen3.6-27B`. A LoRA is a correction learned against one particular set of weights. Qwen3.8-27B has the same shape but different weights, and the abliterated and Heretic builds move them further still. The adapter wasn't trained on any of them, and I haven't checked what it writes there. The old check couldn't see this either: a 64-block Qwen3.8-27B without the draft head passed it in silence.

So 0.27.1 also reads the names a GGUF gives itself (`general.name`, `general.basename`, `general.base_model`), and when none of them is Qwen3.6-27B it says so on the node and in Check it. It warns rather than refuses, because those fields are optional and written by whoever quantised the file. A file whose header names nothing gets no warning at all.

**2.** The README now says this outright, next to the note about Qwen3.5-9B in [GGUF — smaller still, and nothing to install](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI#gguf--smaller-still-and-nothing-to-install).

**3.** No 65-block adapter is planned. It would be the same file, since the adapter has nothing to put in block 64. Your models would need an adapter trained on Qwen3.8-27B, and training one is up to lightx2v. [pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF](https://huggingface.co/pytraveler/MiniMax-H3-Prompt-Rewriter-LoRA-GGUF) is their safetensors converted to GGUF, and converting doesn't move an adapter to another base.

If you want the adapter on the weights it was trained on, that's a Qwen3.6-27B GGUF. MTP builds of it load now too, just without the draft head. For the Qwen3.8 builds the writer nodes are the better fit: they get the H3 format from MiniMax's guide in the system prompt rather than from a LoRA, so any base works.

Thanks for reading the tensors before asking. That's what turned up the MTP bug.



> **Comment by pytraveler (2026-10-12T10:09:13Z):**

Closing, since 0.27.1 covers all three points. If an MTP build still gets refused, or the base-model warning shows up on a file that does name Qwen3.6-27B, reopen this and paste the node's message.




### #17 [closed] ISSUE: Please add Rocm Support
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/17  by GWCoding 2026-09-17T21:03:22Z

For the AMD users, it would be nice to add Rocm support.
A Github Repository with pre-built llama is : https://github.com/lemonade-sdk/llamacpp-rocm

> **Comment by pytraveler (2026-09-18T18:24:35Z):**

Hi, thanks for the link!

This should already work, no code changes needed. The GGUF route doesn't care which backend llama.cpp was built for. Before downloading anything, the node looks for an existing llama.cpp build, so the lemonade ROCm build can be used as it is:

1. Unpack the lemonade release for your GPU somewhere, e.g. C:\llama-rocm or /opt/llama-rocm.

2. Point the node at that folder in one of these ways:
    - put the path on a single line in ComfyUI/user/minimax_h3_rewriter/llama_bin.txt (the most reliable option, because the node reads it itself no matter how ComfyUI was started)
    - or set MINIMAX_H3_LLAMA_BIN to that path
    - or just add the folder to PATH

3. In the options node, set gguf_runtime to llama.cpp so the binaries are used even if a llama-cpp-python wheel is installed.

Once a build is found, llama_backend doesn't matter anymore. That setting only decides which archive gets downloaded when nothing is found. The node needs llama-completion in that folder. The caption nodes also look for llama-mtmd-cli next to it.

All of this is covered in the README under [GGUF — smaller still, and nothing to install](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI#gguf--smaller-still-and-nothing-to-install), in the part starting with "An llama.cpp you already have is run as it is".

If you'd rather not deal with ROCm at all, llama_backend = vulkan works on AMD cards out of the box. It's a 34 MB download and the node fetches it by itself.

If the lemonade build doesn't get picked up, please post the node's error message. When no build is found, it prints what the ComfyUI process actually saw (the variable, the file and PATH), which is usually enough to see what went wrong.



### #16 [closed] ISSUE: This is more of a suggestiton. How about an option to manually download the files?
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/16  by shawnlau23-coder 2026-09-06T20:19:36Z

Downloading the files from the workflow can be very slow. 8mb  down to 4mb/s to download a 15gb file. An hour! Using the HF CLI, it would take less than 5 minutes.

> **Comment by pytraveler (2026-09-07T11:46:16Z):**

Thanks for the report — and your diagnosis was right, which made this easy to act on.

The reason isn't mysterious: this pack moved bytes over a single ranged HTTP connection at a time. The Comfy-Org repositories are Xet-backed, and `hf_xet` fetches the chunks of *one* file over many connections at once. That's the whole of the difference between your hour and the CLI's five minutes.

**In 0.22.0** there's a `downloader` option at the bottom of the MiniMax-H3 Rewriter Options node, with two values:

- `built-in` — the default, unchanged: the transfer that needs nothing installed.
- `huggingface_hub` — the same downloads routed through `huggingface_hub` / `hf_xet`.

For the second one, install into ComfyUI's own Python and restart:

```
python -m pip install huggingface_hub hf_xet
```

If you pick the option without them, the node prints that command with the correct interpreter path for your install — the portable build's embedded Python isn't on PATH, so a bare `pip install` tends to land in the wrong place.

Neither package is a dependency of this pack, and neither is going to become one. If they aren't importable, the built-in transfer runs and says so on the node — nothing fails. That matters because the setting is saved inside the workflow: a graph you share still runs for someone who hasn't installed them.

Four things worth knowing:

- **It covers Hugging Face repositories only** — base models, GGUF writers, LoRA adapters, a model with its projector. The llama.cpp binaries come from GitHub releases, and the two writing guides are one small request each; neither goes through this setting.
- **Half-finished files aren't interchangeable.** The built-in transfer parks bytes in `<name>.part` beside the file; `huggingface_hub` parks them under `.cache/huggingface` inside the destination folder. Switching in the middle of a download restarts that one file and leaves the other's leftovers behind.
- **A file already on disk at its full size is never handed to either backend**, so re-running a workflow costs nothing. This isn't only an optimisation: a finished multi-gigabyte file sitting there with no hub metadata beside it would otherwise make `huggingface_hub` read the whole thing back to hash it.
- **`hf_xet` keeps a chunk cache** under `~/.cache/huggingface/xet` — on `C:` whatever drive your models live on. The free-space check runs against the destination only.

On your literal request, a manual download: that already works [today](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/releases/tag/v0.22.0) and is worth knowing about. Turn `auto_download` off, and instead of fetching anything the node fails with the repository id and the exact file name it wanted — so you can pull it with the CLI into the folder it names and re-run.

One thing I'm deliberately not claiming: I've verified both backends produce byte-identical files, but I haven't timed them against a 15 GB checkpoint, so there's no speedup figure in the changelog. If you try it, I'd be glad to hear what you get.




### #15 [closed] ISSUE: Embeddings ??
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/15  by davoodkharmanzar 2026-09-06T19:31:57Z

Would it be great if implement a method to use **Embedding Keywords** automatically ??

https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/embeddings

> **Comment by pytraveler (2026-09-07T12:36:05Z):**

Done — it's in [0.23.0](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/releases/tag/v0.23.0), as a node called **MiniMax-H3 Effect Embeddings**.

<img width="450" alt="effect_embeddings_node" src="https://github.com/user-attachments/assets/f14d006c-a830-40ac-9e67-b7bb2398cf21" />

One thing worth saying first, because everything else follows from it: they aren't keywords. Each of those files is a single `[N, 5120]` tensor — fifty to a hundred and forty positions of prompt that have *already been through the text encoder*, saved as numbers. `embedding:minimaxh3_bullet_time` in the text isn't a request for bullet time in words; it tells ComfyUI's tokenizer to splice those positions into the sequence at that point. Three consequences, none of them obvious from the folder listing:

- An effect costs prompt **length**, not adjectives. `four_seasons` is 142 positions spent on the effect rather than on your scene. The node has a `tokens` output so you can watch that.
- It arrives at full strength or not at all. Weights don't work either — H3 tokenizes with weights disabled, so `(embedding:x:0.8)` is read as literal text.
- There's nothing to rephrase. It is the same fragment every time.

**What the node does.** You tick effects off a grid that shows what each one costs and which are already on disk, and a button fetches the missing ones (about 10 MB for all ten) into ComfyUI's own `models/embeddings` folder — the only place `embedding:` looks. The prompt passes through character for character apart from the tokens themselves, and running it twice doesn't stack them: unticking an effect removes it again.

**It needs ComfyUI 0.33.0 or newer.** That's the release where `embedding:` started reaching the MiniMax-H3 tokenizer. Below that the token is read as ordinary words. The node checks and says so.

**On "automatically".** The node won't pick effects for you, and I don't think it should. Choosing `dark_magic` for a prompt because it mentions a wizard would be a guess dressed up as a feature — these are not keywords a model can match against your words, and a wrong one costs a hundred positions of your prompt and changes the video. Ticking a box is one click and it is honest about what it did.

**A `placement` widget, which turned out to matter more than I expected.** ComfyUI's tokenizer joins everything that follows a token onto one line — the text after `embedding:name` goes through `' '.join(text.split())` before H3 sees it. So a token at the top of an H3 prompt flattens every blank line between the fields below it. `end of the prompt` is the placement where nothing follows the token and nothing is flattened; the default opens the description field, which is the position an effect was meant to modify. The node tells you on its `findings` output when line breaks are being flattened, so at least it isn't happening behind your back.

I haven't settled by experiment which placement H3 actually responds to best, and I'd rather say that than pretend. If you run the same seed both ways, I'd be glad to hear what you get.

**Every way this fails is silent**, which is the real reason it's a node and not a line you type. ComfyUI drops a token it can't use with one line in the console and nothing else — the video generates, just without the effect, looking exactly like a prompt that never asked for one. So the pack's self-check now reads them: a capital `Embedding:` (the test is case-sensitive), a token stuck to the word in front of it (tokens are only found after whitespace), a full stop on the end of a name. The node itself also says which chosen files aren't on disk.

Two things that would have bitten you were fixed on the way, both in this release:

- A prompt carrying several effects was read by the self-check as a model stuck in a repetition loop — `embedding:a embedding:b ...` is a fixed unit at a short period, which is exactly what that rule looks for. That wasn't only a stray warning: the same function is a stop condition for the generators, so the pack would have burned an extra generation "fixing" a prompt that was correct.
- The Prompt Reducer capitalised `embedding:` at the start of a sentence, silently killing the token.

Last honest note: I haven't verified that any two of these combine into something sensible. They're separate pieces of encoded prompt, and asking for a spiral ascent through four seasons is asking the model to reconcile them. The node allows any combination because there's no reason to forbid one — whether it's worth generating is yours to find out.

Thanks for the suggestion, and for the link — the folder was the whole of what I needed to start.




### #14 [closed] PR: feat: 'remote (llama.cpp server)' gguf_runtime — run every node against an already-running llama-server or llama-swap
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/pull/14  by LacklusterOpsec 2026-09-05T22:01:34Z

## Summary

Adds a fourth `gguf_runtime`, **`remote (llama.cpp server)`**, that sends the messages over HTTP instead of running any of the model locally: the model files, the LoRA, the KV cache and the GPU all belong to whatever answers `server_url`. Since `llama-server` and llama-swap both speak the OpenAI-compatible chat API, the natural target is llama-swap (its default `127.0.0.1:9090` is the default `server_url`), and a plain `llama-server` works the same way.

Nothing is downloaded and nothing is loaded in ComfyUI in this mode — no `llama-cpp-python`, no fetched llama.cpp binaries, no Transformers load, no VRAM. This is the way to use a model already resident in a llama.cpp server (e.g. pointing at an abliterated/heretic build loaded with the prompt-rewriter LoRA server-side).

## What it changes

**New module `minimax_h3_rewriter/remote_engine.py`** — a small OpenAI-compatible client:

- `rewrite()` — text completion used by the 27B rewriter, the guided writers, the universal writer and the reducer. Chat-template rendering and the context belong to the server, so `n_ctx` / `gpu_layers` / `device` / `keep_model_loaded` stop mattering and the remote branch is reached **before** any local model resolution.
- `caption()` — the multimodal route for the captioners and the frame-carrying 8B/Omni writers: pictures as base64 data URIs, sounds as `input_audio` content parts — the same wire shape `server_engine` already uses for a locally spawned `llama-server`. A `media_marker` in a written turn splices each attachment in next to the sentence naming it, matching how `llama-mtmd-cli` interleaves frames.
- Both stream the answer, drive the node progress bar, honour ComfyUI interrupts and run the same repetition stop as the local engines; a missing/refusing server raises a readable error.

**Options node** gains two widgets:
- `server_url` — default `http://127.0.0.1:9090`
- `server_model` — empty lets llama-swap serve its default; a name makes it swap to (and load) that model on demand

`use_lora`, the adapter field and the model dropdowns are ignored in this mode, since the served model carries its own adapter.

**Wiring** — `remote_target()` is the single decision point shared by the text nodes, the caption node, the multi/universal captioners and both frame-carrying writers; `mtmd_engine.describe()` accepts `remote_url`/`remote_model` and takes that path before any local binary or server session is considered. Local context widening is skipped when the server owns the context.

## Notable details found on a live llama-swap

Reasoning-capable served models (every `-heretic`/`-ablit` build answers as a thinker) spend the whole token ceiling on `reasoning_content` and return an **empty** `content` field unless the request carries `chat_template_kwargs: {"enable_thinking": false}`. The body always sends it — the same thinking suppression the pack applies to every local run.

## Verification

Live against llama-swap at `127.0.0.1:9090`:

- text rewrite through `Qwen3.5-4B-ablit` streams back correctly
- image caption through `joycaption-beta-one-llava` describes the picture
- a server refusing a request (vision model served without its `mmproj`) raises a readable error instead of dead-ending

All 391 existing tests pass. No docs/tests were added for the remote path — happy to follow the repo's conventions for them if wanted.

> **Comment by pytraveler (2026-09-06T10:00:39Z):**

Thanks for the PR — it's clearly a real use case, the code reads like it belongs to the pack, and the write-up is thorough. I'm not going to merge it, though, and I owe you the concrete reasons rather than a wave.

Two things are broken.

Saved workflows. server_url and server_model are inserted between gguf_runtime and device in the Options node. ComfyUI restores widgets_values by position, so every workflow saved since 0.17.2 comes back with device holding trust_remote_code's boolean and everything after it shifted by two. This is exactly issue #11 — web/js/options_migration.js exists because it happened once already. New widgets go at the end of the list, or ship with a migration; the PR touches nothing under web/.

The frame writers. mtmd_engine.describe() calls remote_engine.caption(..., media_marker=None) with the None hard-coded, and describe() never gained a media_marker parameter, so the splicing branch of content_parts is unreachable. In practice the literal <__media__> string goes to the server inside the prompt and all frames land at the front of the turn — for FL2VA the model has no way to tell the first frame from the last. That is the failure the marker exists to prevent (see the docstring on render() in writer_8b.py). It's also the behaviour the PR description lists as working; the live test was a single image caption, which wouldn't show it.

Smaller things: reachable() and models() are never called (models()'s docstring names a log line that doesn't exist); _chat() takes messages and model and uses neither; _delta, MEDIA_TYPE, REQUEST_SECONDS and content_parts are copies of server_engine's. The mmproj vision/audio checks are skipped in remote mode, so the pack's "this captioner cannot hear" refusal becomes a raw HTTP error. And gguf_runtime now overrides the Transformers path too, while its tooltip still says it's about GGUF.

The design reason, which would stand even with all of that fixed. This pack's job is a specific model with a specific LoRA and specific templates, and every node's output quality rests on that. In remote mode the model is whatever answers the port, and use_lora, merge_lora, the adapter field, both model dropdowns, gpu_layers, n_ctx, keep_model_loaded, auto_download and device all become silent no-ops with no sign of it in the UI. Every bug report from that mode is "the server gave me bad text", and nothing in this repository can fix it. The Ollama integration in 0.20.0 was built the opposite way on purpose — ollama_store.py opens with "Nothing here is a new backend": it indexes someone else's files and hands them to the engines that already exist. A fourth backend that bypasses model resolution entirely isn't a direction I want to maintain.

If you want to pursue the narrower version — server_engine gaining an "attach to a server I didn't start" mode instead of a parallel module, widgets appended at the end of Options, the marker actually threaded through, and the UI saying plainly what's disabled — open an issue first and let's agree on the shape before code. Closing this one.



### #13 [closed] ISSUE: It insists on downloading an llm fmodel that is already there.
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/13  by shawnlau23-coder 2026-09-05T21:09:03Z

Qwen2.5-Omni-7B-Q8_0.gguf is in models/LLM/Qwen2.5-Omni-7B-GGUF/  and it insists on writing over that file. Is it looking for a certain hash and not the name? Maybe you could point to the model so I could download it faster with HF CLI.


### #12 [closed] ISSUE: Please add Ollama support.
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/12  by razvanab 2026-09-04T17:23:47Z

Some of these models I have already downloaded for Ollama.
Thank you.

> **Comment by pytraveler (2026-09-05T13:21:59Z):**

Done — it will be in the next release (v0.20.0), and it turned out simpler than adding a
backend: the node reads your Ollama store directly instead of talking to its API.
What Ollama pulls is a plain GGUF, so llama.cpp loads it where it lies. Nothing
is copied, converted or downloaded.

Your models appear in the `writer_model` and `caption_model` dropdowns, prefixed
`ollama:` and named the way `ollama list` names them:

    ollama: qwen3:8b [qwen3, 4.7 GB]
    ollama: moondream:latest [+mmproj, vision, 1.7 GB]

Ollama itself does not have to be running — this reads the files, not the API. A
vision model reaches both lists: as a writer without its projector, as a
captioner with it, paired from the model's own manifest.

`OLLAMA_MODELS`, `~/.ollama/models` and `/usr/share/ollama/.ollama/models` are
found automatically, so on a normal install there is nothing to configure. A
store somewhere else (WSL, a container, another drive) is named by hand in
`ollama_stores` in `models.json`.

One limit worth stating: this is for the writers and the captioners. The
MiniMax-H3 rewriter nodes still need their own base model plus the LoRA, which
Ollama cannot apply.


> **Comment by razvanab (2026-09-05T18:06:10Z):**

I understand. 
Thank you.



### #11 [closed] ISSUE: Rewriter nodes can't find any models to use
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/11  by Huhtaluhta 2026-08-30T00:23:56Z

Rewiter Options and Rewriter Omni nodes can't find, nor scan any models that I already have in models/sub_folders. Including folders: text_encoders, clip, llm. So I can't choose any Rewriter Lora or LLM model to use. I've installed [MiniMax-H3 Prompt Rewriter] node pack succesfully and requirements.txt too and there is no red outlined nodes in the workflow. 

When I click to generate: Comfylog says:

[ERROR] Failed to validate prompt for output 22:
[ERROR] * MiniMaxH3RewriterOptions 66:
[ERROR]   - Value not in list: merge_lora: True not in ['auto', 'on', 'off']
[ERROR]   - Value not in list: llama_backend: False not in ['auto', 'vulkan', 'cuda', 'cpu']
[ERROR]   - Failed to convert an input value to a INT value: n_ctx, auto, invalid literal for int() with base 10: 'auto'
[ERROR]   - Value 8192 bigger than max of 999: gpu_layers
[ERROR] Output will be ignored
[ERROR] Failed to validate prompt for output 65:
[ERROR] Output will be ignored

Workflow works like any regular H3 workflow when I turn If/else to false and bypass Rewriter Options node. 

> **Comment by Huhtaluhta (2026-08-30T00:48:11Z):**

<img width="1040" height="822" alt="Image" src="https://github.com/user-attachments/assets/b32cc80f-abff-4c06-908a-8dda2420ec3c" />

<img width="1278" height="833" alt="Image" src="https://github.com/user-attachments/assets/7c1b4ef2-1232-4e17-8919-74c1f66db9a4" />

Rewiter Options node's auto download does not work either.


> **Comment by pytraveler (2026-08-30T09:29:58Z):**

Thanks for the detailed report - the log and the screenshots made this easy to pin down. There were two separate things going on, and 0.18.1 addresses both.

The validation errors. Your workflow was exported with a version of this pack older than 0.17.2 (the axiomgraph one you linked in #10). 0.17.2 added a merge_lora widget in the middle of the Options node, and ComfyUI restores saved widget values by position - so everything below use_lora landed one widget too low: merge_lora got auto_download's true, gpu_layers got n_ctx's 8192, n_ctx got the string auto (the NaN on your screenshot), llama_backend got trust_remote_code's false. That is exactly the four errors in your log. As of 0.18.1 the node recognizes the old layout when a workflow loads and puts every value back where it was written from - nothing to redo by hand.

The model dropdowns. Up to 0.18.0 the nodes only scanned models/LLM (plus transformers, diffusers, unet_gguf and the Hugging Face cache) - text_encoders and clip were not looked at, which is why your files were invisible. 0.18.1 scans those two as well, and the adapter list also checks loras. One thing to know: the rewriter dropdowns only list models that can actually carry the LoRA - the 8B/27B rewriters take qwen35/qwen3vl-architecture GGUFs and the Omni rewriter takes Qwen2.5-Omni-7B specifically, because an adapter only fits its own base model. A QwenVL that doesn't match shows up in the guided writer and captioner nodes instead, where any architecture works.

Auto download wasn't broken - it never got the chance to run: validation rejected the graph before any node executed, so nothing was ever asked to download. Once the workflow loads cleanly, the dropdown entries marked with a download size fetch themselves on first run while auto_download is on.

So: update the pack to 0.18.1, restart ComfyUI, refresh the browser tab, reload the workflow. Both the errors and the empty dropdowns should be gone. Thanks again - the widget-shift bug would have bitten everyone with a pre-0.17.2 workflow.



### #10 [closed] ISSUE: You could not spare ONE workflow to  get it off the ground?
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/10  by shawnlau23-coder 2026-08-29T17:54:06Z

Can you add a work flow?

> **Comment by Huhtaluhta (2026-08-30T00:54:31Z):**

I got my workflow from here: https://github.com/axiomgraph/ComfyUIWorkflow


> **Comment by pytraveler (2026-08-30T09:31:03Z):**

<img width="350" alt="workflow in comyui gallery now" src="https://github.com/user-attachments/assets/ae9ea52c-606a-4cb6-8cd2-2652662442e3" />

Did. Please update the nodes to version 0.18.1.


> **Comment by shawnlau23-coder (2026-08-31T22:52:03Z):**

Thanks!




### #9 [closed] ISSUE: If possible, I would appreciate it if the option to select 32:9 and 48:9 aspect ratios could be added.
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/9  by Geese586 2026-08-27T16:25:27Z

A very useful keyword optimizer!
However, if possible, it would be best to add options for 32:9 and 48:9 aspect ratios, as I currently use two 1080p monitors connected together for one video, and another for a three-screen setup similar to a racing game. My current workflow involves using Prompt-Rewriter for optimization and then manually adjusting the settings. Finally, have a nice day!

> **Comment by pytraveler (2026-08-28T08:33:45Z):**

Both are in **0.17.1**: `32:9` and `48:9` sit at the wide end of the ratio list,
before `21:9`, on every writer and rewriter node.

One thing worth knowing before you use them. The ratio reaches the model as a
line of text in the request, and none of the adapters was trained on those two
strings — so the model composes wider by reading the request, not by recognising
a shape it was taught. Worth a look at the shot layout before you generate.

The same release also adds an `aspect_ratio` input beside the picker, which reads
what other nodes write: `3840x1080` arrives as 32:9, `5760x1080` as 48:9, and a
labelled choice like `3:4 (Portrait Standard)` — what ComfyUI's own Resolution
Selector calls it — reads as 3:4. So the ratio can come from whatever already
decides your frame size, instead of being set in two places.

Thanks for the request.


> **Comment by Geese586 (2026-09-03T13:39:18Z):**

> 两者都在**0.17.1 版本**中：`32:9`并且`48:9`位于比率列表的宽端，在每个写入器和重写器节点上 位于 之前。`21:9`
> 
> 在使用它们之前，有一点需要注意。宽高比是以 请求中的一行文本的形式传递给模型的，而适配器并没有针对这两个 字符串进行训练——因此，模型是通过读取请求来构建更宽的画面，而不是通过识别 它已经学习过的形状。建议在生成画面之前先查看一下镜头布局。
> 
> 同一版本还在`aspect_ratio`选择器旁边添加了一个输入框，用于读取 其他节点写入的内容：`3840x1080`例如 32:9、48 `5760x1080`:9 以及一个 带有标签的选项`3:4 (Portrait Standard)`（ComfyUI 自带的分辨率 选择器称之为 3:4）。因此，屏幕比例可以由已经 决定帧大小的因素决定，而无需在两个地方分别设置。
> 
> 感谢您的请求。



> Both are in **0.17.1**: `32:9` and `48:9` sit at the wide end of the ratio list, before `21:9`, on every writer and rewriter node.
> 
> One thing worth knowing before you use them. The ratio reaches the model as a line of text in the request, and none of the adapters was trained on those two strings — so the model composes wider by reading the request, not by recognising a shape it was taught. Worth a look at the shot layout before you generate.
> 
> The same release also adds an `aspect_ratio` input beside the picker, which reads what other nodes write: `3840x1080` arrives as 32:9, `5760x1080` as 48:9, and a labelled choice like `3:4 (Portrait Standard)` — what ComfyUI's own Resolution Selector calls it — reads as 3:4. So the ratio can come from whatever already decides your frame size, instead of being set in two places.
> 
> Thanks for the request.

Tks for your support.



### #8 [closed] ISSUE: Best prompt nodepack! Thanks. Ideas: amended instruction for 'MiniMax-H3 Reference Caption' (and instruction field in 'Multi Reference Caption')
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/8  by 808charlie 2026-08-24T20:39:16Z

Now I've got things running, this really is great. Very configurable and flexible. Thank you!

I observed some models caption a picture well (albeit with an empty <think> </think> at the start of the description). 
Other models at times return a ton of thinking (particularly Qwen 3.5, particularly variants of original model). They get to a great description, but think first.

The prompt you use for the Reference Caption node is output to the cli, so I'd already amended the prompt using a text input to the 'instruction' field in MiniMax-H3 Reference Caption. I then found adding "return only the prompt not your thinking" fixed the problem with 'thinking' in every case! I daresay other similar instructions to 'return only...' would have a similar effect.

The 'instruction' field for custom instructions is therefore great.
Request 1: any chance of an instruction field in the Multi Reference Caption node? Appreciate it might need one of pic/aud/vid as the prompt is not the same.

Request 2: Even more usefully, is it possible the 'MiniMax-H3 Prompt Writer' could include an 'instruction' field. You provide the 'system prompt' (instruction) exactly through the 'minimx-H3 guide promt (and LLM) node. Users could then edit it if needed, but importantly edit it for other models.

### Why so much interest?
Best node pack I've found.

- Your nodes have significant advantage of others like 'ComfyUI-QwenVL' or 'ComfyUI_Qwen3VL-instruct' nodes, which require editing json files and changes to the node stop them trying to download models and to specify personally selected and pre-downloaded models. They use safetensors. 
- MiniCPM node has above problems, but was one which used GGUF, but it had other issues as I've mentioned before.
- pixaroma has some prompt nodes, but whilst some of his are great, I have real issues with the slick functionality and JS (Claude coded) and how non-native they are and fragile the interaction feels - it feels like the difference between 'windows' and 'linux' :D 
- Your nodes are faster than the native 'text-generate' that use safetensors and is much more fussy on models and much slower. Indeed, I've used safetensors models with yours, as you know, via clip-load, and for some reason it also takes WAY longer than loading a Q8 GGUF. Way longer. I suspect all the pinning and over-management of memory to be related. 

Whilst I'd not normally be a fan of GGUF within ComfyUI (on my hardware), it really hits the spot for quick calls to LLMs doesn't it. Really impressed so far thanks.




> **Comment by pytraveler (2026-08-24T21:23:30Z):**

It’s really great and nice how you describe my work. thank you very much. This is motivating.
Your requests have been accepted. I need a little time to make this more user‑friendly. 
I’ll definitely let you know when I’m done.


> **Comment by 808charlie (2026-08-24T21:33:15Z):**

Thanks so much. I'll leave this one open. Sorry to have raised so many issues - kind of ironic that my lack-of-knowledge of python and understanding of llama.cpp has somehow influenced improvements to someone's code.

Would be great if a youtuber like benjiAI or nerdyrodent pick up on your nodes - that would really increase the visibility. The lightx2v loras almost seem a red-herring here, it's great you implemented them, but the more general prompt/img nodes are so welcome.
Get a break, no rush with the requests! Thank again.





> **Comment by pytraveler (2026-08-25T18:01:10Z):**

Both are in 0.16.6, and so is the thing your workaround was working around.

## The empty `<think> </think>` was mine

Not a quirk of those models. The CLIP captioner has always cut the thought block
out of an answer; the GGUF one never did, so it went straight into the caption -
and `" ".join(caption.split())` is what turned `<think>\n\n</think>` into the
`<think> </think>` you were looking at. On a model that really reasons it is
worse than untidy: the whole deliberation lands in `reference_assets`, and from
there it travels on into the writer's prompt as if you had written it.

Both paths cut it now.

Cutting is only half of it, though, and the other half cannot be a flag. The
writer nodes render the chat template themselves and pass
`enable_thinking=False`. A caption cannot: `llama-mtmd-cli` applies the template
itself - that is what puts the media tokens in the right place in the turn - and
it publishes no switch for the thought channel. I checked the build here (b10599)
rather than guessing: no `--chat-template-kwargs`, no `--reasoning-budget`, only
`--jinja`. So the reasoning is still charged against `--predict`, which is how a
thinking model gives you a page of deliberation and half a sentence of caption.

Which is exactly what your sentence fixed. So it is now the last line of every
caption question, in the node rather than in your text field:

```
Do not think out loud: give the answer only, with no reasoning and no preamble.
```

It asks for no reasoning rather than for a description, which turned out to
matter: the first wording said "reply with the description only" and quietly
argued with every instruction that wanted something other than a description.

You found the right instrument. I have only bolted it on so nobody else has to.

## Request 1: per reference, not per node

You anticipated the problem in the same breath as the request - one node, four
kinds of asset, and no single question that suits a picture, a clip and a sound
at once. So the field is not on the node. It is on the reference.

The strip of coloured squares that the Universal Writer has always had is now on
Multi Reference Caption too, and under each square there is a narrow band in the
same colour. Dark while that reference is asked its role's usual question, lit
once it is asked something else. Click it to write the question, right-click to
take it back, hover to read it. The text rides in a hidden JSON widget, so it
travels with the workflow and through the API like anything else.

The band has two lit states, and testing is what put them there. A checkbox in
the little editor decides which, and the band then says which it was:

- **`+ instr`** - your line is *added* to the role's question.
- **`= instr`**, on a solid band - it is asked *instead* of it, and it takes over
  from `length` as well: whoever writes the question owns the shape of its
  answer.

Neither could be the only one. "Do not mention the window" is a rule, not a
question: asked on its own it leaves nothing wanting a description, and a 3B
model answers the rule - `Subject 1: No`. "Always answer 'blah blah blah'" is a
whole question, and leaving the role's in front of it hands the model a
contradiction, which it settles by describing anyway. Added is the default,
because a line typed into an empty box is usually the first kind.

Bringing the strip across had a second reason. The checkboxes on the input rows
were drawn straight onto the canvas, and *Modern Node Design (Nodes 2.0)* does
not draw the canvas - so that node quietly had no switches at all for anyone on
the new renderer. The squares are HTML and work in both.

## Request 2: `system_prompt` on the writers

Yours is the better statement of it, so I will use it: the guide *is* the whole
of what makes these H3 writers. Nothing else in the node knows the format. So
`system_prompt` replaces the assembled guide outright, on the T2VA writer, the
Ref2VA writer and the Universal Writer - and when it holds anything, the guide is
not even fetched. That last part is for you specifically: an offline machine
should not have to download 24 KB of MiniMax's document in order to ignore it.

The way in is the one you spotted. **MiniMax-H3 Guide Prompt (any LLM)** hands
you the stock system prompt on an output; take it, edit it for Krea or LTX,
connect it back into `system_prompt`. Editing what the node would have sent
anyway beats writing one from nothing.

Two things do not move:

- **The task message is never replaced.** It carries the prompt, the aspect ratio
  and the duration, which every writing guide needs whatever house style it asks
  for - and it is the half you have no reason to rewrite.
- **The answer is still split into the H3 sections.** A guide that replies with a
  paragraph fills `rewritten_prompt` and leaves the section outputs empty. Worth
  knowing rather than worth avoiding.

## One thing your report cost me, usefully

New widgets now go at the *end* of a node, always. ComfyUI restores a saved
node's widget values by position, so adding `reference_instructions` next to the
setting it belongs with - which is the obvious place - handed every widget below
it its neighbour's value. A workflow saved before the change came back with
`max_frames` holding what `context_size` had, and `context_size` holding a
boolean. Caught it here, nothing shipped in that state; but it is the sort of
thing that would have been very hard to report from the outside.

## On the rest of what you wrote

Thank you, genuinely. Two of your points are worth answering rather than just
accepting.

**Why the GGUF route is faster than a safetensors encoder here.** It is not the
node being clever. `CLIPLoader` loads a full-precision text encoder through
ComfyUI's own allocator, with all the residency management that implies; the
GGUF route memory-maps a quantised file and hands it to a process that exits
afterwards. Different job, different cost. The CLIP path still wins when you have
several references in one run - it loads once for the lot, where the GGUF
captioner starts a process and reads the weights again for every asset.

**The LoRAs are not a red herring, but you are right about the shape of it.**
They are the reason the pack exists and the smallest part of what it now does:
everything guide-driven runs on any instruction-following GGUF, and after this
release it does not even have to be MiniMax's guide. That was worth someone
saying out loud.

Thanks for four reports in three days, every one of them precise enough to act
on. Two of them found real bugs and this one found a third.



> **Comment by 808charlie (2026-08-25T23:58:54Z):**

Ah, you do the right thing! 
I've come across nodes where authors fiddled with widgets etc, and existing workflows didn't work, or field had 'NaN' etc. I tend to try and open new nodes rather than copy them around workflow to avoid it. 

> You found the right instrument. I have only bolted it on so nobody else has to.

Funnily enough it was the first thing I tried, without thinking, so that was lucky! I guess that's something to be said for natural language LLMs - you can speak to them in normal language without a lot of thought!

Thanks so much for all the changes. I'll grab the new release and try in a few days... need some non IT time... and have work to do now!

As it happens, I just recommended your node to someone who posted on ComfyUI issues with a llama cpp problem with a node they were using.

It might be worth creating a new custom node if the pack now extends beyond Minimax, as it's title might be broader?

Thanks again. 



> **Comment by pytraveler (2026-08-26T06:39:26Z):**

Good idea. And I’m just toying with the idea of reusing the nodes and, based on them, creating similar ones for other diffusion models.


> **Comment by 808charlie (2026-08-26T22:29:47Z):**

More to maintain that way? But equally a good idea to keep stuff split. Two times I'm frustrated by packs (repos) growing rather than separate nodes:

- RGthree - I had to fork the repo to make a 'PowerLTXLoraLoader' (the only item I have on github) to allow attenuation of 3 categories of audio layers (Audio, A2V and V2A) on LoRAs (kind of re-using Kijai's ideas but in the PowerLoraLoaderFormat). But I couldn't figure how to build it as stand alone node LIKE is PLL, as the node relies on lots of other pack functionality - so I had to just fork and add my crappy version. As it's a huge pack and he keeps making changes, I've not bothered to keep merging so I'll have to mark mine as stale - would have been much easier if PLL had been separate. I'm unlikely to bother with LTX given Minimax's superiority, so no interest in admin updating the fork. (I hope Kijai does a similar Lora node for minimax as he did for LTX). Seems odd to have a lora loader thrown in with critical things like mutter and context switches etc. That's a lot of work to pick apart if something happened like Nodes 2.0 got forced on us and RGthree stopped updating.

- Pixaroma - at an extreme, these nodes have some great ideas (like some of the cropping nodes etc) but has grown to include so much functionality, and numerous items which already exist, the latest being VRAM monitoring. Kijai had already bettered over the once-staple Crystools suite. Too much stuff in one repo.

So yea off that, perhaps it would be sensible to create other nodes for other models... but whether it's one huge repo with the lot or not will be interesting.

I'd best close this issue do you think? 
You've included the functionality - thanks again - which warrant closure I guess. And I see you've added some sensible controls over image size etc; I can hardly keep up!


> **Comment by pytraveler (2026-08-26T23:11:17Z):**

I completely agree with you. It’s better to share responsibility. Therefore, this project will be used exclusively for the purpose for which the code was developed. I will focus on usability and finding errors in the code. 
Thank you again for your feedbacks! It was very helpful.



### #7 [closed] ISSUE: Reference Image captioning work with CLIP, but insists on trying to download Vulkan build to use GGUF (can it use cuda build of llama-python-cpp)?
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/7  by 808charlie 2026-08-23T22:42:15Z

Prompt writer is working great for me now. Thanks for your earlier advice.

Reference image now poses a problem.

Running offline, I got:

`Gave up on 'llama-b10310-bin-ubuntu-vulkan-x64.tar.gz' after 5 attempts: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /ggml-org/llama.cpp/releases/download/b10310/llama-b10310-bin-ubuntu-vulkan-x64.tar.gz (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x788a52c6f560>: Failed to establish a new connection: [Errno -2] Name or service not known'))`

Odd, as I alread have llama-ccp-python. 

Next I tried connecting 'minimax h3 rewriter options' to 'minimax-h3 reference caption', and selection 'llama backend cuda'

then I got

`llama.cpp cuda: upstream publishes no CUDA build of llama.cpp for Linux; the Vulkan build runs on NVIDIA cards too, and one you compiled yourself is run as it is -- put its bin folder on PATH, or name it in MINIMAX_H3_LLAMA_BIN.`

then i tried connecting a qwen3-VL-8B model (as a safetensors file) via the native 'load clip' node. **Now that works!** But how to get GGUFs working?

So...
now i have a recent cuda llama.cpp version, **can I use that** instead of needing to download a vulkan build, and if so... could you possibly give me a better clue as to how to achieve that?

"put its bin folder on PATH" on what path, where?
"name it in MINIMAX_H3_LLAMA_BIN" - where is that, I can't find such a file in your custom node folder.

Sorry to be dumb!

Thanks in advance.

> **Comment by pytraveler (2026-08-24T07:56:46Z):**

You did exactly what the error told you to do, and it sent you looking for a file that does not exist - it was written for someone who already knew the part it left out. That missing part is the whole of your problem, and it is my bug.

## Two different things are called llama.cpp here

`llama-cpp-python` is llama.cpp **as a library**: a set of shared objects that Python loads through ctypes. Look inside your own build and you will see exactly that, and nothing else:

```
ls ~/ComfyUI/venv/lib/python3.*/site-packages/llama_cpp/lib
# libggml-base.so  libggml-cpu.so  libggml-cuda.so  libllama.so  libmtmd.so
```

No programs. The writer nodes are happy with that, because they call the library directly.

The caption nodes cannot be. Multimodal input in llama.cpp lives in `mtmd`, and `llama-cpp-python` publishes no binding for it - so a reference image goes to **`llama-mtmd-cli`, a program**, run as a subprocess. Your machine has no such program: compiling the wheel with `-DGGML_CUDA=on` built the CUDA library, not the CLI tools. That is why the node went looking for an archive even though a perfectly good CUDA llama.cpp is installed, and why there is nothing on your disk to point `MINIMAX_H3_LLAMA_BIN` at yet.

Two consequences worth naming while you are here:

- `gguf_runtime` steers the **writer** nodes only. The caption nodes have no wheel path at all, so that dropdown cannot help them.
- `llama_backend = cuda` on Linux refuses because upstream publishes no CUDA archive for Linux - only CPU, Vulkan, ROCm, SYCL and OpenVINO. Vulkan runs on NVIDIA perfectly well, at roughly half the tokens per second.

## Your two questions, answered plainly

**"put its bin folder on PATH" - on what PATH, where?** The `PATH` of the process that runs ComfyUI, which is your shell's `PATH` only if you start ComfyUI from that shell:

```sh
export PATH=$HOME/llama.cpp/build/bin:$PATH
python main.py
```

If ComfyUI runs as a service, a container or from a launcher script, an `export` in your terminal never reaches it - it is handed an environment of its own.

**"name it in MINIMAX_H3_LLAMA_BIN - where is that file?"** It is not a file, it is an environment variable, set in the shell that starts ComfyUI:

```sh
export MINIMAX_H3_LLAMA_BIN=$HOME/llama.cpp/build/bin   # the folder, or the binary in it
```

Since 0.16.1 there is a third way that needs no environment at all - one line in a file **you create yourself**, in ComfyUI's user directory, not in the custom node folder:

```sh
mkdir -p ~/ComfyUI/user/minimax_h3_rewriter
echo $HOME/llama.cpp/build/bin > ~/ComfyUI/user/minimax_h3_rewriter/llama_bin.txt
```

That one is read by the node itself and does not care who started ComfyUI. Update the pack to 0.16.3 before trying it.

## So, pick one

### Option A: build the CLI tools with CUDA - you already have the toolchain

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j --target llama-mtmd-cli llama-completion
```

`llama-mtmd-cli` is the target that matters; `llama-completion` in the same build lets the writer nodes use it too. Then name `~/llama.cpp/build/bin` in any of the three ways above.

Once a build is found, `llama_backend` stops mattering entirely - it picks which archive to download, not what an existing binary was compiled against. Set `device = cuda:0` and it does what it says. This is the only road to a CUDA llama.cpp on Linux.

### Option B: no compiling - install the official archive by hand

You are offline, so let the archive arrive the way your models do. On a machine with a connection:

```sh
wget https://github.com/ggml-org/llama.cpp/releases/download/b10310/llama-b10310-bin-ubuntu-vulkan-x64.tar.gz
```

On the ComfyUI machine:

```sh
mkdir -p ~/ComfyUI/user/minimax_h3_rewriter/runtime/b10310-vulkan
tar -xf llama-b10310-bin-ubuntu-vulkan-x64.tar.gz \
    -C ~/ComfyUI/user/minimax_h3_rewriter/runtime/b10310-vulkan
```

The folder name is exactly where the node looks, so nothing else is needed - leave the layout inside the archive alone, the binary is found at any depth. It carries `llama-mtmd-cli` beside `llama-completion`, so this covers both kinds of node, and `auto_download` can stay off afterwards. Use `tar` rather than a file manager: the archive ships its `.so` files as symlinks and the loader needs them.

32 MB, and it runs on your 5090 through the NVIDIA Vulkan driver.

### Option C: what you already have working

Qwen3-VL as safetensors through the native **Load CLIP** node is not a fallback - it is the other supported captioning path, and it touches no llama.cpp at all. If it does what you need, there is nothing here you have to fix. The GGUF captioner exists for people who would rather not keep a second copy of the weights, or who are short on VRAM.

## What I will change

- The Linux CUDA refusal will say that a `llama-cpp-python` build, CUDA or not, contains no `llama-mtmd-cli`, and will name `llama_bin.txt` alongside `PATH` and the variable.
- The caption nodes will say up front that the wheel cannot caption, so "but I already have llama-cpp-python" stops being a surprise at run time.
- A download that fails with no connection will print the manual unpack path and the exact URL, instead of five identical connection errors.

Thanks for writing it up this precisely - "I set the dropdown and got told to use a file I cannot find" is exactly the report that shows where the docs stop being docs.



> **Comment by pytraveler (2026-08-24T08:20:59Z):**

https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/releases/tag/v0.16.4


> **Comment by 808charlie (2026-08-24T13:32:46Z):**

Thanks again. I'll have to read the above a couple of times as it is both a solution and education which is great.

I am online, of course, I just tend to run updates online for this venv, then use it offline. I've tried so many custom nodes, I've no time to read through everyone's code nor the ability to understand much of it - so I make the assumption of giving it least necessary access in case of bad actors. Probably overly cautious. 

Right, I'll grab the new version of your node, thank you.

### Question re option A

(Feel free to tell me to Go Ask chatGPT or RTFM, I'm probably working my ticket!)

I'll do option B and see what performance is like, but I'd like to try option A. 

Now, I have llama.cpp as is use the server to run models at times. I can't afford to have that running with ComfyUI, hence the use of nodes which can run models without having llama server running appeals.

I installed with compile options I kind of grabbed from others, which worked:

`cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON`
build options:
`cmake --build llama.cpp/build --config Release -j %(nproc) --clean-first`
then I pop the binares in home folder:
`cp llama.cpp/buld/bin/llama-* llama.cpp/`

you have suggested;
```
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j --target llama-mtmd-cli llama-completion
```

So I understand that essentially **I just need to add the two options**:  `llama-mtmd-cli llama-completion` to my build process, then I'll be able to utilise the install from ComfyUI. It wouldn't effect my ability to run server etc?  Is that correct?

second Q: with the cli options installed, my question is how this would work: I presume the cli version runs when it is called then releases memory after. But would this run in a process outside of ComfyUI? Would I likely see memory challenges as it tries to grab say 20GB of GPU, which Comfy already thought it had free-reign over? Is it a non-ideal route?

I guess that is a positive to using clip a loader, Comfy handles it all. (irritatingly too much - Kijai's visualizer indicates pinned RAM equal to any GPU model for the last 3 releases, robbing my machine's natural file cache in RAM, but that another topic (https://github.com/Comfy-Org/ComfyUI/issues/15556) - I do wish the guys had time to respond to issues - thank you SO MUCH for your speedy responses.




> **Comment by 808charlie (2026-08-24T13:58:35Z):**

I'm being a bit slow here. My build already includes the two items:
/home/user/llama.cpp/build/bin/llama-mtmd-cli
/home/user/llama.cpp/build/bin/llama-completion

Haven't updated llama.cpp since June, so due a pull and rebuild, but it sounds like it is just a matter of setting paths... I'll try now....




> **Comment by 808charlie (2026-08-24T15:51:24Z):**

Good news. 

- I downloaded vulkan libraries. that works.
- i added` echo $HOME/llama.cpp/build/bin > ~/ComfyUI/user/minimax_h3_rewriter/llama_bin.txt. that works`

- I connected the 'minimax-h3 rewriter options' to 'minimax-h3 reference caption'
- i set 'gguf runtime' to llama.cpp. that works
- i set 'gguf runtime' to llama-cpp-python. that works.
- i tried setting 'llama_backend' to cuda
- i tried setting llama_backend to 'vulkan'. 
- i tried removing the vulkan runtime so I know it's definitely using cuda.
all working

qwen2.5-omni runs in 3.5 seconds. 
Qwen 3.5 4B/9B, around 4.5 seconds
out of interest I tried some finetunes/abliterated versions of above - they worked too (as per writer node).

What's odd - the time was the same speed under each of the above bullet conditions. Strikes me as odd; not slow, just odd. **is that expected?**

_What doesn't work_ (not that is is supposed to, just observation)

Qwen3-VL (4b/8b)
gives error:

```
llama-mtmd-cli exited with code 1.
0.00.017.544 I common_init_result: fitting params to device memory ...
0.00.017.550 I common_init_result: (for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)
0.00.407.101 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 999, abort
0.00.556.755 W load: control-looking token: 128247 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
0.01.321.442 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 36864.00 MiB on device 0: cudaMalloc failed: out of memory
0.01.321.449 E alloc_tensor_range: failed to allocate CUDA0 buffer of size 38654705664
0.01.321.923 E llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
0.01.321.929 E common_init_result: failed to create context with model '/home/user/ComfyUI/models/llm_gguf/Qwe3VL-8b/Qwen3-VL-8B-Instruct-Q4_K_M.gguf'
0.01.321.930 E common_init_from_params: failed to create context with model '/home/user/ComfyUI/models/llm_gguf/Qwe3VL-8b/Qwen3-VL-8B-Instruct-Q4_K_M.gguf'

Not every multimodal GGUF works here: llama.cpp's mtmd has to understand the projector format, and several current models abort while loading it. Pick an entry from the captioner list, which names the ones this pack has been run against.
```

It does work with 'clip loader'. odd. this model uniquely doesn't seem to work as GGUF.
Interested if you know why.

Suggestion:
less verbose output to cli:

Every single run reports on every GGUF which doesn't have a mmproj file. Oddly it does this 4 time I think. As I have many GGUF with no mmproj on my disk (e.g. all qwen 3 versions!) I get pages of output like this:


```
INFO] got prompt
[INFO] [minimax_h3_rewriter.scan_captioner_gguf] no obvious projector for /home/user/ComfyUI/models/llm_gguf/Qwen3-4B-AB-v2.Q8_0.gguf among ['qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.V2_jackr_jackr-mmproj-BF16.gguf', 'qwen3.5-27B-qwopus3.5-v3.5_jackr_jackr-mmproj-BF16.gguf', 'qwen3.5-27B_unsloth-mmproj-BF16.gguf', 'qwen3.6-27B_unsloth-mmproj-BF16.gguf', 'qwen3.6-35B-A3B_unsloth-mmproj-BF16.gguf']
... (loads more models) ...
[INFO] [minimax_h3_rewriter.scan_captioner_gguf] no obvious projector for /home/user/ComfyUI/models/llm_gguf/Qwen3-4B.Q8_0.gguf among ['qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.V2_jackr_jackr-mmproj-BF16.gguf', 'qwen3.5-27B-qwopus3.5-v3.5_jackr_jackr-mmproj-BF16.gguf', 'qwen3.5-27B_unsloth-mmproj-BF16.gguf', 'qwen3.6-27B_unsloth-mmproj-BF16.gguf', 'qwen3.6-35B-A3B_unsloth-mmproj-BF16.gguf']
Generating tokens:   7%|█████████▏                                                                                                                  | 76/1024 [00:03<00:40, 23.29it/s]
[INFO] Prompt executed in 3.80 seconds
```

Strangely enough the folder the folder  /home/user/ComfyUI/models/llm_gguf/  only contains some GGUFs. the above reporting complains about every model without a mmproj on every run in that folder, but within that folder is a symlink to an external drive with LOADS of models on it. The above output neither comments on any of the GGUFs without mmproj nor suggests other mmproj it finds on that disk (which is good as there possibly 100s). 
The node does FIND the models on the link and list them as 'ondisk' and i run them fine. But the above output doesn't ever reference them. Odd. Just thought I'd mention.





> **Comment by pytraveler (2026-08-24T20:08:55Z):**

Good - and thank you for testing all six combinations rather than one. Three of
your observations are answers in themselves, and the fourth is a bug of mine
with a number attached to it.

## The build: do not add those flags

You read it the natural way and it is the opposite of what I meant. `--target`
**narrows** a build, it does not extend it. I put it in because you had not
compiled yet and I did not want to make you build the whole tree for two
programs. Your own command builds everything, which is exactly why both binaries
were already sitting there:

```
/home/user/llama.cpp/build/bin/llama-mtmd-cli
/home/user/llama.cpp/build/bin/llama-completion
```

So: change nothing. If you *had* added `--target llama-mtmd-cli
llama-completion` to your usual line you would have stopped building
`llama-server`. Keep the build you have; a pull and a rebuild is all it ever
needs.

Nothing about the node affects `llama-server` either. They are separate
executables out of the same tree, and running one has never had an opinion about
the other. The only place they meet is the card, and only if both hold a model
at the same time.

## What the subprocess does to memory

Yes, it is a real process outside ComfyUI, and it is not left to fight for the
card. Before the child starts, the node asks ComfyUI to evict:

```python
mm.unload_all_models()
mm.soft_empty_cache(force=True)
```

but only when the caption is going to the same GPU ComfyUI is using. Send it to
a second card with `device` and the unload is skipped deliberately - freeing
VRAM the child will not touch buys nothing and costs a full reload of the
diffusion model afterwards.

The child exits when the caption is done and its memory goes back with it. So
the two never hold the card together, and the cost of that is a reload of
whatever ComfyUI had loaded, once, after the caption.

Two things that route does pay for and the CLIP loader does not:

- **One process and one read of the weights per asset.** Five references, five
  loads. `Load CLIP` loads once for the whole run and ComfyUI keeps it.
- **Nothing arbitrates.** ComfyUI's allocator does not know the child exists. If
  `llama-server` is up with a model resident, or anything else is on the card,
  the eviction above does not help - that is your 20 GB scenario, and it is
  real.

Non-ideal is fair. It is the trade: the GGUF route reaches models ComfyUI has no
encoder for and keeps nothing resident, the CLIP route is managed and cheaper
per run. Both produce the same shape of answer, and the nodes pick between them
by whether `clip` is connected. Yours has an encoder on disk that works - use it,
and keep the GGUF route for what it cannot load.

## Why every combination ran at the same speed

Because every one of them ran the same binary: yours.

`llama-mtmd-cli` is looked for in five places, in this order:

1. `MINIMAX_H3_LLAMA_BIN`
2. `llama_bin.txt` in the user directory
3. an archive this pack unpacked earlier
4. `PATH`
5. the download

You wrote `llama_bin.txt`, so it stops at 2 and never reaches a question about
backends. On top of that:

- `gguf_runtime` steers the **writer** nodes. The caption nodes have no wheel
  path at all, so both settings did the same nothing.
- `llama_backend` picks **which archive to fetch**, not what an existing binary
  was compiled against. Deleting the Vulkan runtime changed nothing for the same
  reason - it was not being used.

So the six bullets were one configuration tested six times, and identical
timings are the correct result. The switch that does change something there is
`device`.

## Qwen3-VL: not the projector, and not your fault

The projector loaded. Look at where it stopped - it got past `load:`, past the
vision tower, and died creating the context:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 36864.00 MiB on device 0
alloc_tensor_range: failed to allocate CUDA0 buffer of size 38654705664
llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
```

That is a KV cache, and here is where the number comes from. The caption node
sends `--ctx-size 0`, which tells llama.cpp to use the model's own training
context. For Qwen3-VL that is 262144 tokens. The model has 36 layers, 8 KV heads
of 128 dimensions, K and V, at 2 bytes:

```
36 * (1024 + 1024) * 2          = 147,456 bytes per token  (144 KiB)
147,456 * 262,144               = 38,654,705,664 bytes      (36 GiB)
```

which is the number in your log to the byte. Qwen2.5-Omni works because it asks
for 32k rather than 256k. Qwen3-VL-4B fails identically because it has the same
36 layers and the same KV width - the parameter count is in the feed-forward
part, which the cache does not touch. And `Load CLIP` works because transformers
grows the cache with the sequence instead of reserving the maximum up front.

The line above it is the other half:

```
common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 999, abort
```

b10310 would have sized the context to your card by itself. It refused because
the node pins `--n-gpu-layers 999`, which counts as "the user has decided".

**Right now:** set `context_size` on the Reference Caption node to `16384` and
Qwen3-VL-8B loads - that is 2.25 GiB of cache. `32768` if you are captioning a
video and want the frames to fit comfortably.

**What I will change:**

- `0` will mean "the model's context, capped to what will actually fit" rather
  than "whatever the header says". The node already reads GGUF headers to build
  the dropdown, so the context length costs nothing to add.
- Stop pinning the layer count when the device is left on auto, so `-fit` can do
  the job it was trying to do.
- The "not every multimodal GGUF works here" note stops being printed on every
  non-zero exit. It is true of some models and it was flatly wrong here, and it
  sent you looking at the model when the child had already told you it ran out
  of memory. That is the part that cost you the most time, and it is the worst
  bug in this report.
- Qwen3-VL joins the captioner list once it loads on a sane context.

## The log noise

Also mine, and narrower than it looks. That line prints for a model that sits in
a folder holding an mmproj it could not be paired with - your top-level
`llm_gguf` has five projectors beside every unrelated Qwen3, so each one gets
announced. Four times over, because four different nodes offer a captioner
dropdown and each rebuilds its list when ComfyUI asks for the node definitions.

Folders with no projector in them are silent, which is why your symlinked drive
never shows up in that output while its models still appear in the dropdown -
nothing there ever fails to pair, so there is nothing to report. Not the symlink.

One debug-level line per folder in the next release.



> **Comment by 808charlie (2026-08-24T20:57:06Z):**

Amusingly, all the info about missing mmproj for models I never intended on using has had me collecting them as I'd overlooked the download in some cases. So at least it made me start some housekeeping to my libraries for models !!!

Just had a look through the diff, thanks for all the work on this. I'd found Qwen3VL is a really nice little model for captioning without having to wheel out Qwen 3.5, so it's great there's potential try that  again.

Really interesting info re GGUF vs safetensor routes in terms of memory. I guess maybe once I narrow down favorite models and actually bother to use minimax much, the safetensor route will fit better for repeated runs. But your nodes offer both options and I've learned lots today!

Grabbed the new release - I'll go test Qwen3-VL now!



> **Comment by pytraveler (2026-08-24T21:14:22Z):**

I’m glad if I helped. Thank you for your research and reports. This has helped improve my work. It’s also very important for other users! 


> **Comment by 808charlie (2026-08-24T21:22:42Z):**

Great news. With current release, leaving context at 0.

All my Qwen3VL 4B/8B models work. Spot on!

setting to 32768 seems a good default anyhow.

Yes, the nodes will now work out-of-the-box for more users I suspect. Great work.

OK, I'll close this one too!


> **Comment by 808charlie (2026-08-24T21:23:45Z):**

Closing; author improved code to add functionality and offered advice. Thank you.



### #6 [closed] ISSUE: It would be better if it supported pasting images with Ctrl+V.
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/6  by Windowsex 2026-08-23T17:31:54Z

I cannot paste images directly inside the plugin panel.
My usual workflow is to prepare my image inside ComfyUI first. After I get the ready‑to‑use image in ComfyUI and switch to the plugin panel, I get overwhelmed by the long image list when trying to pick the correct one.

If I select an image from the plugin list instead, I have to copy the image from the plugin panel over to ComfyUI, then copy the prompt separately again. This workflow is quite tedious.

Things would be much smoother if Ctrl+V paste‑image was supported here. A one‑click paste feature would be even better.

> **Comment by Windowsex (2026-08-23T17:43:59Z):**

Sorry, posted to the wrong repository by mistake! Closing this now.


> **Comment by pytraveler (2026-08-24T07:43:48Z):**

No trouble at all - thanks for coming back to say so. Closing it here.

If you do end up wanting this in a ComfyUI graph: the reference caption nodes read from whatever feeds their image input, so a native Load Image node is the place a pasted image would land.



### #5 [closed] ISSUE: fallback to CPU
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/5  by siraxe 2026-08-23T14:47:49Z

Can't use GPU by default , qwen3.6-27B Q4 on 32gb vram
`MiniMax-H3 Prompt Rewriter` `MiniMax-H3 Universal Rewriter`
```
llama_context: n_ctx_seq (8192) < n_ctx_train (262144) -- the full capacity of the model will not be utilized
llama_adapter_lora_init_impl: lora for 'blk.0.ffn_gate.weight' cannot use buft 'CPU_REPACK', fallback to CPU
llama_adapter_lora_init_impl: lora for 'blk.0.ffn_up.weight' cannot use buft 'CPU_REPACK', fallback to CPU
```
Is there's some requirements/settings I need to apply/use ?

> **Comment by siraxe (2026-08-23T15:03:40Z):**

nvm needed to set this in options 

<img width="458" height="85" alt="Image" src="https://github.com/user-attachments/assets/bebd227c-5ad6-46fc-b0e5-cc0441fd2bf8" />



### #4 [closed] ISSUE: Writer runs with Qwen3-4B GGUF, but not Qwen3-VL-4B or Qwen3.5-9B etc
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/4  by 808charlie 2026-08-22T18:18:20Z

Thanks for creating those... but I'm having challenges getting the prompt writer to run.

Can you advise?
Happy to gather info as needed.

I've successfully Prompt Writer with GGUF quants of Qwen3-4B and Qwen3-8B. Unsloth and indeed abliterated versions work. Super speedy.

BUT...

It fails to utilise all other models:
Qwen3-VL-4B
Qwen3-VL-8B
Qwen3.5-9B-Q4_K_M.gguf  (the exact unsloth version listed in your models.json!)
Qwen3.5-9B from HauhauCS (the exact version listed in your models.json!)

I've checked requirements.txt; I have models exceeding the requirements, and running 
I tried editing the models.json rather than just putting GGUF in my LLM folder.

In all cases, the erros is the same:

```
[INFO] got prompt
[INFO] [minimax_h3_rewriter._guided_text] raising n_ctx from 8192 to 9216 for the T2VA guide
[ERROR] !!! Exception during processing !!! Failed to load model from file: /home/username/ComfyUI/models/llm_gguf/Qwen3.5-9B-Q4_K_M_unsloth.gguf
[ERROR] Traceback (most recent call last):
  File "/home/username/ComfyUI/execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/username/ComfyUI/execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/username/ComfyUI/execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "/home/username/ComfyUI/execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "/home/username/ComfyUI/custom_nodes/MiniMax-H3-Prompt-Rewriter-ComfyUI-main/minimax_h3_rewriter/nodes.py", line 1248, in write
    text = _guided_text(
           ^^^^^^^^^^^^^
  File "/home/username/ComfyUI/custom_nodes/MiniMax-H3-Prompt-Rewriter-ComfyUI-main/minimax_h3_rewriter/nodes.py", line 1088, in _guided_text
    return gguf_engine.rewrite(**common)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/username/ComfyUI/custom_nodes/MiniMax-H3-Prompt-Rewriter-ComfyUI-main/minimax_h3_rewriter/gguf_engine.py", line 399, in rewrite
    llama = load(model_path, adapter_path, gpu_layers, n_ctx, device, progress)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/username/ComfyUI/custom_nodes/MiniMax-H3-Prompt-Rewriter-ComfyUI-main/minimax_h3_rewriter/gguf_engine.py", line 269, in load
    llama = llama_cpp.Llama(**kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/username/ComfyUI/.venv/lib/python3.12/site-packages/llama_cpp/llama.py", line 374, in __init__
    internals.LlamaModel(
  File "/home/username/ComfyUI/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 58, in __init__
    raise ValueError(f"Failed to load model from file: {path_model}")
ValueError: Failed to load model from file: /home/username/ComfyUI/models/llm_gguf/Qwen3.5-9B-Q4_K_M_unsloth.gguf

[INFO] Prompt executed in 0.52 seconds
Exception ignored in: <function LlamaModel.__del__ at 0x703ef3154720>
Traceback (most recent call last):
  File "/home/username/ComfyUI/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 86, in __del__
    self.close()
  File "/home/username/ComfyUI/.venv/lib/python3.12/site-packages/llama_cpp/_internals.py", line 78, in close
    if self.sampler is not None:
       ^^^^^^^^^^^^
AttributeError: 'LlamaModel' object has no attribute 'sampler'
```

System details

```
5090/128GB RAM, linux, ComfyUI running in a venv (under UV)

[INFO] DynamicVRAM support detected and enabled
[INFO] Python version: 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]
[INFO] ComfyUI version: 0.33.1
[INFO] comfy-aimdo version: 0.4.13
[INFO] comfy-kitchen version: 0.2.31
[INFO] comfyui-frontend-package version: 1.48.7
[INFO] comfyui-workflow-templates version: 0.11.41
[INFO] comfyui-embedded-docs version: 0.5.9
[INFO] comfy-kitchen version: 0.2.31
[INFO] comfy-aimdo version: 0.4.13
```



> **Comment by pytraveler (2026-08-22T19:51:46Z):**

Thanks for the report, and for pasting the whole traceback - it had everything I needed.

Good news first: nothing is wrong with your setup, and your GGUF files are fine. The node was simply hiding the real reason from you.

## What is going on

`llama-cpp-python` is a Python wrapper with a copy of llama.cpp compiled inside it. That copy is frozen at whatever date the wheel was built, and it can only load model families that llama.cpp already supported on that date.

| Model | Supported in llama.cpp since |
| --- | --- |
| Qwen3 (the ones that work for you) | 9 April 2025 |
| Qwen3-VL | 30 October 2025 |
| Qwen3.5 | 10 February 2026 |

Judging by the line numbers in your traceback, your `llama-cpp-python` is somewhere in the 0.3.10-0.3.16 range. Version 0.3.16 has llama.cpp from **14 August 2025** inside it. So every model released before that date loads, and every model after it fails - which is exactly the split in your list.

llama.cpp does say what is wrong; it prints `unknown model architecture: 'qwen35'` to a low-level channel that the node keeps quiet, so all that reached you was `Failed to load model from file`. That part is my bug: the next release names the architecture and tells you the runtime is too old. The `'LlamaModel' object has no attribute 'sampler'` line at the very end is harmless noise from the same version.

## How to fix it - pick one

### Option A: change one dropdown, install nothing

Add a **MiniMax-H3 Rewriter Options** node if you do not have one already, connect its output to the `options` input of the Writer node, and set `gguf_runtime` to `llama.cpp`. Then run again.

The node then downloads the official llama.cpp binaries (about 30 MB, the Vulkan build on Linux) on first use and runs your model through those. They are build b10310 and know every model on your list.

One trade-off: with this runtime the model is unloaded after each run, because `keep_model_loaded` only works through llama-cpp-python. So the next run reloads it.

### Option B: update llama-cpp-python

In the same environment ComfyUI runs from (your `.venv`):

```
uv pip install --force-reinstall https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-vulkan/llama_cpp_python-0.3.34-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

Then restart ComfyUI. That wheel carries llama.cpp from 11 July 2026, which covers both Qwen3-VL and Qwen3.5.

To see which version you have, before or after:

```
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

0.3.34 or newer is what you want.

I suggest the Vulkan wheel even on a 5090: it runs on any GPU and does not have to match your driver's CUDA version. The CUDA wheels are faster when they fit - they are on the same releases page - but a CUDA wheel and a driver that disagree fail in ways that are much harder to read than this.

## One note about Qwen3-VL

The Writer node uses it as a plain text model - it will not look at images through it. It works, just as a text writer. Image and video input goes through the Reference Caption nodes instead.


> **Comment by 808charlie (2026-08-23T20:27:01Z):**

That has to be tied with rattus128 as the most helpful response I've had on github. Thank you **so much** for such a comprehensive and directive response.

I elected to try updating llama-cpp-python rather than using the options setting via your nodes, as the only thing I'm aware of using GGUF is minicpm. The other image describing or prompt improvement nodes I use make use of safetensors 

- native 'text generate'
- https://github.com/IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct
- https://github.com/1038lab/ComfyUI-QwenVL

(ironically, the last of those added GGUF support like 3 days ago)
 
Because gguf doesn't seem to be used except with one custom node https://github.com/1038lab/ComfyUI-MiniCPM I figured I'd just update it in the venv. 

Funnily enough, I had issues with llama-cpp-python a while back; it was my first github issue https://github.com/1038lab/ComfyUI-MiniCPM/issues/16 . I found the solution myself. I had llama installed without cuda, but the node considered requirements met, so I uninstalled so it would get reinstalled with `cmake.args="-DGGML_CUDA=on"`

As such I thought I'd try building as I know it has previously worked with cuda, so I'd prefer that over a pre-built vulkan wheel (ableit I might be setting myself up for hard-to-identify errors later). 

```
cd ~/ComfyUI
source ./venv/bin/activate
uv pip uninstall llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --no-cache-dir --no-binary llama-cpp-python "llama-cpp-python==0.3.34"
```

Your node now works with newer models. You were right! And everything is faster!!!

Funnily enough the same node i learned about updating llama-cpp-python for, miniCPM turned out to be the cause of huge memory pinning in ComfyUI (figured out between rattus128 and me - mainly him! https://github.com/Comfy-Org/ComfyUI/issues/14378). Turned out to be useful as rattus128 was able to fix ComfyUI to stop nodes causing this hog. I should probably post a bug report to node author, as he has many useful nodes, but he seemed to be awol for a while letting issues close without response so I never raised the issue. Ironically he now has Minimax prompting node, but I'm really keen to try working with yours instead as it is much more flexible.

### Little ideas for potential improvements

1. the 'on disk' dialog is rather messy. Is there any chance this could be displayed with the normal native display in ComfyUI? You know, when you go to load a model or a lora or suchlike, the folders are visible? I think it's the 'auto next subdirectories in menus' from RGthree's custom node. The presentation from your node is by contrast a list of GGUFs it found. I have a link out from LLM folder to a drive with like 80 GGUFs on it nicely folderised- but it is a massive list on your node (and a lot of splurge on the startup log too).

2. This node is labelled as minimax, but it is pretty universal. I was seeking this from the native 'text generate' node, but got no response as the guys are innundated https://github.com/Comfy-Org/ComfyUI/issues/14388 . 
I am guess your nodes just need different system prompts (not the long guide minimax uses) and they could refine prompts for Krea, LTX etc. Would it be beneficial to allow connection of a custom prompt instead of it using the text files from minimax, to expand its use?

3. Not an improvement, but can I say THANK YOU for creating a node that (save for guide) doesn't rely on internet connectivity and makes explicit what it will download. I like to be able to reinstall and run entirely offline. So many nodes seem to fail in that regard, or worse still - re-download the same damn thing every time, so you can restart ComfyUI, and the damn thing wants to go online again. Yours work really well.

Now I have prompts working, I'll have to test image description node. Hoping and qwen3VL can be used in place of Omni in that role?

Thank you again!




> **Comment by pytraveler (2026-08-24T07:35:33Z):**

You're welcome 🙏 


> **Comment by 808charlie (2026-08-24T13:40:30Z):**

Would you like me to close this, or leave open changing title to resolved but leave up for others to see? Probably the former. Your call!

Thanks!


> **Comment by pytraveler (2026-08-24T20:00:42Z):**

Close it, please - and thank you for asking rather than just doing it. A closed
issue is still searchable, the diagnosis is in the thread, and "Failed to load
model from file" is the phrase the next person will paste into the search box
either way. 


> **Comment by 808charlie (2026-08-24T20:45:28Z):**

Helpful advice provided by author. Running well. Closing with thanks.



### #3 [closed] PR: fix: don't refuse safe symlinks when unpacking llama.cpp release archives
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/pull/3  by ViolinKaine 2026-08-21T21:40:39Z

## Summary

`_safe_extract()` in `llamacpp.py` refused **every** tar symlink/hardlink outright (`member.issym() or member.islnk()`), as a zip-slip guard. But the official Linux/macOS llama.cpp release archives ship their shared libraries as ordinary SONAME symlinks (e.g. `libllama.so.0 -> libllama.so.0.0.0`), so extraction always failed with:

```
RuntimeError: refusing archive member outside the target: 'llama-b10310/libllama.so.0'
```

on every Linux Vulkan/CPU build and the macOS build — i.e. any non-Windows install fetching the runtime for the first time.

## Fix

Only refuse a symlink/hardlink whose **target** actually resolves outside the destination directory — that's the real zip-slip risk. A same-directory SONAME symlink (or anything else that stays inside the extraction root) is allowed through, same as a regular file.

## Test plan

- [x] Reproduced the original failure on Linux (Vulkan archive) before the fix
- [x] Confirmed the archive extracts cleanly and `llama-mtmd-cli`/`llama-completion` run after the fix
- [ ] Would appreciate a check on Windows/macOS archives too, since I only have Linux to test on

🤖 Generated with [Claude Code](https://claude.com/claude-code)

> **Comment by pytraveler (2026-08-22T13:21:41Z):**

Merged as `aebfbc2`. Thanks - good catch, and a clean diagnosis.

I reproduced it independently against release b10310 before merging, and the
numbers back you up:

| archive | members | symlinks | old guard |
|---|---|---|---|
| `ubuntu-vulkan-x64.tar.gz` | 64 | 10 | refused at `llama-b10310/libllama.so.0` |
| `ubuntu-x64.tar.gz` | 63 | 10 | refused at `llama-b10310/libllama.so.0` |
| `macos-arm64.tar.gz` | 62 | 18 | refused at `llama-b10310/libggml-rpc.0.dylib` |

To your open question about the other platforms:

- **Windows** was never affected. All three win32 assets are `.zip`, and the zip
  branch has a check of its own that never looked at links.
- **macOS** is hit exactly the way Linux is. The arm64 archive carries 18 SONAME
  symlinks and the old guard died on the first one; your fix clears it. I only
  ran the guard against that archive, not a full extraction on real hardware, so
  a macOS user confirming end-to-end is still welcome.

Worth adding, since it makes the fix load-bearing rather than cosmetic: the links
are not optional. `ldd llama-completion` resolves `libllama.so.0`,
`libggml.so.0` and `libggml-base.so.0` - the symlinks themselves - so even a
variant that kept the real files and skipped the links would unpack a runtime the
loader refuses to start.

One thing I changed in a follow-up commit: the resolution base for hardlinks.
CPython resolves a hardlink's `linkname` against the extraction root, not the
member's directory:

```python
# tarfile.py
if filtered.islnk():
    filtered._link_target = os.path.join(path, filtered.linkname)
```

Symlinks go the other way - `os.symlink(tarinfo.linkname, targetpath)` writes the
name verbatim, so it resolves from the link's own directory, which is exactly
what your patch does. Applying that same base to hardlinks counts the link's own
depth twice, so an escape shallower than the member's depth slips through:

```
llama/bin/link -> ../../secret.txt
  computed: normpath("llama/bin/" + "../../secret.txt") = "secret.txt"  -> inside, allowed
  actual:   dest/../../secret.txt                                       -> two levels above dest
```

Low practical risk here - the archives come from ggml-org over HTTPS, and
`os.link` only succeeds when the outside file already exists and grants no
permission the process does not already have - but it is a one-line difference:

```python
base = os.path.dirname(member.name) if member.issym() else ""
return inside(os.path.normpath(os.path.join(base, member.linkname)))
```

Verified with crafted archives: absolute and escaping symlinks, absolute and
escaping hardlinks (both deeper and shallower than the member) and a plain
`../evil.sh` are all refused; same-directory SONAME links, subdirectory-relative
symlinks and root-relative hardlinks are allowed; all three real archives still
pass, and `llama-completion --version` runs from the extracted tree.

Going out as 0.16.2. Thanks again.



### #2 [closed] ISSUE: LoRA silently not applied on llama-cpp-python 0.3.47 (+ crash on unload once it is), and a non-English output option
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/2  by ioritree 2026-08-17T11:37:20Z

# LoRA silently not applied on llama-cpp-python 0.3.47 (+ crash on unload once it is), and a non-English output option

Three items, kept in one report because they were found in one session and the
second only becomes reachable once the first is fixed. The third is a feature
request and can be ignored independently of the two bugs.

| | |
|---|---|
| Pack version | 0.12.0 (`b9b8116`) |
| llama-cpp-python | **0.3.47** |
| ComfyUI | portable, Windows 11, Python 3.12 |
| Base model | a 64-block / 5120-wide `qwen35` GGUF (`Q4_K_M`, 15.8 GB) |
| Adapter | `MiniMax-H3-Prompt-Rewriter-LoRA-F16.gguf` |
| Node | `MiniMax-H3 Prompt Rewriter`, `gguf_runtime: auto` → resolves to `llama-cpp-python` |

All line numbers below refer to `b9b8116`.

---

# Bug 1 — the adapter is never applied, and nothing says so

`gguf_engine.py:159` passes the adapter as a constructor argument:

```python
if adapter_path:
    kwargs["lora_path"] = adapter_path
```

llama-cpp-python **0.3.47 removed `lora_path` from `Llama.__init__`**. It was
replaced by a two-stage API — `load_lora(name, path)` at load time plus a per-call
`active_loras=[{"name": ..., "scale": ...}]`. The same release also added a
`**kwargs` to `Llama.__init__` (declared, never read), so the argument is
**accepted and discarded** instead of rejected:

```python
>>> import llama_cpp, inspect
>>> llama_cpp.__version__
'0.3.47'
>>> [p for p in inspect.signature(llama_cpp.Llama.__init__).parameters if 'lora' in p]
[]
>>> inspect.signature(llama_cpp.Llama.__init__).bind(None, model_path='x', lora_path='y')
<BoundArguments ...>          # no TypeError
```

That last line is why the guard at `gguf_engine.py:163-171` cannot help. Its
comment reads:

```python
except TypeError as error:
    # Older builds spell the adapter argument differently; retry without
    # it rather than silently dropping the rewriter.
```

It anticipates the argument being *rejected*. 0.3.47 does the opposite — it
swallows it. So the `TypeError` never fires and the node runs the plain base model
while logging `prompt-rewriter LoRA: ...` as if the adapter were attached.

## How it presents

Nothing in the log looks wrong, and `_resolve_adapter` reports success because it
only resolves a path — it runs before the load. The one visible symptom is that
`use_lora` makes no difference at all:

Same prompt, same seed, `use_lora` on vs off → **byte-identical output**. Both
produced range-style timestamps (`[Shot 1] 00:00-00:05`), never the trained
`00:08.500` cut-point form.

## Suggested fix

Detect the API by signature rather than by `try`/`except`, since the failure mode
is silence and not an exception:

```python
def _lora_api(llama_cpp) -> str:
    """'modern', 'legacy' or '' — which adapter API this build offers."""
    if hasattr(llama_cpp.Llama, "load_lora"):
        return "modern"
    parameters = inspect.signature(llama_cpp.Llama.__init__).parameters
    accepted = parameters.get("lora_path")
    if accepted is not None and accepted.kind is not accepted.VAR_KEYWORD:
        return "legacy"
    return ""
```

`legacy` keeps `kwargs["lora_path"]`. `modern` calls `llama.load_lora(name, path)`
after construction and adds `active_loras=[{"name": name, "scale": 1.0}]` to the
completion call. `""` should raise, pointing the user at `gguf_runtime: llama.cpp`.

Two details worth keeping:

- **Verify the registration.** `Llama.eval` looks the adapter up by name and, on a
  miss, skips it with a warning printed **only when `verbose`** — and this backend
  sets `verbose=False` (`gguf_engine.py:155`). An unchecked `load_lora` therefore
  rebuilds the same silence. `name in llama.list_loras()` closes it.
- **Never drop `active_loras` in a retry.** Falling back without it would run the
  base model as though the adapter were attached, which is bug 1 again.

## Workaround for users

Set `gguf_runtime` to `llama.cpp`. `cli_engine.py:112` passes `--lora` to
`llama-cli.exe` and was never affected. Note that `auto` picks the wheel whenever
llama-cpp-python imports (`nodes.py:834`), so `auto` lands on the broken path.

---

# Bug 2 — access violation on unload, once the adapter really is attached

With bug 1 fixed, the run completes correctly and then the worker thread dies:

```
[minimax_h3_rewriter.gguf_engine.load] adapter registered as '...' (1 loaded)
Windows fatal exception: access violation

Stack (most recent call first):
  File "llama_cpp\_internals.py", line 561 in free
  File "llama_cpp\_internals.py", line 455 in unload_lora
  File "llama_cpp\_internals.py", line 484 in unload_all_loras
  File "llama_cpp\_internals.py", line 100 in close
  File "llama_cpp\llama.py", line 891 in close
  File "minimax_h3_rewriter\gguf_engine.py", line 187 in unload
  File "minimax_h3_rewriter\gguf_engine.py", line 294 in rewrite
...
OSError: exception: access violation reading 0x0000000000000058
OSError: exception: access violation reading 0xFFFFFFFFFFFFFFFF
```

The cause is an ordering bug in `LlamaModel.close()` (`_internals.py` ~90-105):

```python
if getattr(self, "model", None) is not None:
    llama_cpp.llama_model_free(self.model)     # base model freed FIRST
    self.model = None
...
if hasattr(self, "_lora_registry") and self._lora_registry:
    self.unload_all_loras()                    # adapters freed AFTER
```

The adapters hold references into the model's own structures, so
`llama_adapter_lora_free` runs against memory that is already gone. It is not a
double free — `LlamaLoraAdapter.free` does guard on its own pointer — purely the
order.

This one is upstream in llama-cpp-python, but it is worth working around here,
because it takes the ComfyUI worker thread with it and it fires on every unload.

## Suggested fix

Release the adapters in `unload()` while the model is still alive, before `close()`:

```python
release = getattr(llama, "unload_all_loras", None)
if callable(release):
    try:
        release()
    except Exception:
        log.debug(...)
```

`close()` then finds an empty registry and skips the bad branch, and everything
downstream is safe to run twice: the `_exit_stack` callback and
`LlamaLoraAdapter.__del__` both become no-ops because `free()` guards on its own
pointer.

## Verified

Two consecutive runs after the fix — the second proving the teardown left no broken
state behind — with no access violation and no traceback:

```
17:10:48  adapter registered ... (1 loaded)  →  169.26 s
17:13:08  adapter registered ... (1 loaded)  →   77.99 s
```

And the adapter is demonstrably working now: cut points come back as
`[Shot 2] 00:08.500`, the trained form, instead of the `00:00-00:05` ranges the base
model produces unaided.

## Related: `verbose=False` hides the only clue

`gguf_engine.py:155` suppresses llama.cpp's loader output, which is where adapter
tensor counts and skip warnings would appear. Reasonable as a default, but it did
mean the terminal offered nothing for either bug. One `INFO` line after a successful
`load_lora` — the registry name and `loaded_lora_count` — makes both failure modes
visible without turning `verbose` back on.

---

# Feature request — an `output_language` option

## What it is

A `Combo` widget on the prompt nodes selecting the language of the **prose**, while
every piece of structure stays English. Running it locally for a while, the
non-obvious parts turned out to be these.

## The scaffolding must never be translated

`fields.py` splits a node's answer on field names, ASCII colons and `[Shot N]`
markers, so a model that helpfully translates `overall_soundscape` produces output
this pack cannot parse. The rule therefore has to enumerate what survives
untranslated: field names and their colons, `[Shot N]`, timestamps such as
`00:05.000`, the tags `<d>`, `<scenetrans>`, `<cutoff>`, speaker IDs such as `(S1)`,
and the alignment line. Ref2VA adds more: the relationship markers
(`fully_preserved` and friends), the square-bracketed task-type prefix, and the
`<Subject N>` / `<Picture N>` / `<Video N>` / `<Audio N>` labels with their
`(appears in [Shot N])` parentheses. ASCII punctuation only — a full-width colon
breaks the split just as thoroughly as a translated field name.

Separately, the `<d>[...]` tag is a *language* tag driving pronunciation, not
decoration. It has to track the language actually spoken, which is not necessarily
the prose language, so it needs its own mapping rather than reusing the prose one.

## Rebutting the guide takes naming both of its forms

Neither guide merely happens to be English; both say so, and they say it
differently. The base guide scatters the instruction through its sections ("Use 1-4
English sentences", "a natural English action", "350-500 English words") while the
ref guide opens with a single global mandate on line 5 ("Write all six rewrite
sections in English"). A rule that rebuts only the scattered form leaves the global
one standing, and it loses quietly — the answer comes back perfectly formatted and
entirely in English, which reads like the widget did nothing rather than like it was
outvoted. Quoting both shapes and stating which side wins fixed that.

## Some GGUF chat templates steal the primacy slot

Placing the rule at the top of the output contract assumes the pack owns the start
of the system message. A quantised community build can break that. One
Qwen3.8 ABLITERATED GGUF carries a ~1,100-character English persona inside its own
`tokenizer.chat_template` and the template prepends it to whatever system message it
is handed, so `ROLE` is no longer the first thing read and the language rule ends up
mid-message. Measured on that file: the first Chinese character moved from offset
1,152 to 10,038 in the rendered prompt depending on whether the demand was restated
before `ROLE`. Two lines ahead of `ROLE`, for non-English only, takes the slot back
and costs nothing on a clean template.

## Rules lose to demonstration — this was the important one

Even with the demand at offset 57, at the top of the contract, and restated as the
last line of the user message, one 27B build still wrote the whole answer in English.
The reason is in the guide itself:

```
Ref2VA guide: 22 worked example blocks, 6,198 of 23,551 characters
3 of them open with exactly "<Subject 1> is the ..."
```

That is verbatim how the model opened its answer. It was not deciding on a language,
it was **imitating the examples**, and the examples carry their language with them.
With `greedy: true` the argmax at the first token is English, deterministically;
turning greedy off only made it a coin toss.

What fixed it was giving imitation something else to copy — one opening line, in the
requested language, as the very last thing in the user message, immediately before
the assistant turn:

```
output_language: Traditional Chinese (繁體中文) — write the prose in it, not in English, ...
Your answer therefore begins like this, and continues in the same language throughout:
subject_definitions:
<Subject 1> 是 <Picture 1> 中的年輕女子，深棕色長髮，身穿藍色開襟外套。<|im_end|>
<|im_start|>assistant
```

Only the first line, and only the shape the requested mode actually starts with
(`subject_definitions:` + `<Subject 1>` for Ref2VA, `integrated_multimodal_description:`
+ `[Shot 1]` otherwise) — the language of the whole answer is settled at that one
token. The scaffolding inside the example stays English on purpose, so the example
models the split the rules describe rather than only the prose.

Result on the build that had been failing: **4/4 with sampling on random seeds, and
2/2 with `greedy` on**. Greedy going non-English is the meaningful part — it means
the argmax itself moved, so the output is deterministic rather than a coin toss.

## Two implementation notes

- **The widget belongs in `optional`, appended last.** ComfyUI stores
  `widgets_values` positionally, so inserting it anywhere else shifts every saved
  workflow. (0.12.0 adding `format` to `Guide Prompt` before an existing option
  demonstrated this from the other direction.)
- **On the LoRA path, leave `SYSTEM_PROMPT` byte-identical.** The adapter was
  trained on that exact text, so the language directive is best appended as one more
  bullet on the end of the existing Requirements list, and only when a non-English
  language is chosen. Expect it to hold less firmly there than on the guided writers,
  where the prompt can simply be rewritten instead of being fine-tuned in — although
  in practice the short LoRA system prompt held better than the writers did, since
  there are no 22 English examples competing with it.
- **The default must stay `English`.** Anything else silently changes behaviour for
  every existing user on upgrade, and breaks the property that selecting English
  sends the current prompts unchanged.

## Test coverage, honestly

- **Traditional Chinese** — 6/6 as described above, on the build that had failed,
  with no content bleed from the exemplar wording.
- **Simplified Chinese, Japanese, Korean** — exemplar lines written, **never run**.
  Those need someone to verify before they are trusted.

Captions are a smaller case: they land in `reference_material` and the writer rewrites
them, so a caption language option mostly decides what you read on the node — though
matching the writer's language does remove one pull back towards English. Asset labels
(`Picture 1:`) have to stay English either way.


> **Comment by pytraveler (2026-08-17T16:30:30Z):**

Thank you - this is an unusually good report. Every code reference lands, both bugs are real, and both are fixed on `main` in c6362cf, going out as 0.12.1.

## One correction, for anyone who finds this later

The version. abetlen's llama-cpp-python is at **0.3.35**, released the same day this was filed, and `lora_path` is alive and well there: declared at `llama.py:102`, applied through `llama_adapter_lora_init` + `llama_set_adapters_lora` at 436-458. There is no 0.3.47 upstream, and `_internals.py` upstream contains the string `lora` exactly zero times, so none of the traceback frames can come from it.

`0.3.47` is **JamePeng's CUDA fork** - tag `v0.3.47-cu131-win-20260815`. Naming the build would have saved a while, because read against upstream the whole report describes an API that does not exist. Worth a line in the version table next time.

Everything else checks out against that fork's `main`:

- `llama.py:183` - `**kwargs`, declared and never read in `__init__`; no `lora_path` anywhere in the file
- `llama.py:1016-1035` - `load_lora` / `unload_lora` / `list_loras` / `unload_all_loras`
- `llama.py:1272-1283` - `active_loras` routing in `eval`, and the registry-miss warning at 1280 is indeed `elif self.verbose:`
- `_internals.py:93` frees the base model, `:100` walks the adapter registry afterwards

## One thing is worse than you reported

The `**kwargs` is not new in the fork. Upstream 0.3.34 has it too - probed on the installed wheel:

```
version: 0.3.34
has VAR_KEYWORD: True
bogus kwarg accepted -> silently swallowed
```

So the `except TypeError` guard was dead code on every build that has ever existed. It simply never mattered upstream, where `lora_path` is a real parameter. That makes the case for a signature probe rather than a `try`/`except` stronger than the report puts it.

## The fix

Close to your sketch:

- `_lora_api()` reads the signature and returns `modern` / `legacy` / `""`. The empty case raises and points at `gguf_runtime: llama.cpp`, whose `--lora` was never affected.
- `load_lora` is read back with `list_loras()` and refused if the name did not register - exactly your point that the eval warning is invisible at `verbose=False`, so an unchecked call rebuilds the same silence one layer down.
- One `INFO` line on success: `adapter registered as 'MiniMax-H3-Prompt-Rewriter-LoRA-F16' (1 loaded)`.
- The `TypeError` retry in `generate()` drops `seed` and only `seed`; `active_loras` is never dropped, for the reason you gave.
- `unload_all_loras()` runs in `unload()` while the model is still alive, so `close()` finds an empty registry.

## Verified on the fork, not only reasoned about

cp313/cu131 wheel, RTX 5090, driver 591.74, Qwen3.6-27B Q5_K_M + `MiniMax-H3-Prompt-Rewriter-LoRA-F16.gguf`.

A negative control first, because a fix that is not load-bearing is worth knowing about. Calling `close()` with a populated registry - the pre-fix path - reproduces your crash exactly:

```
_internals.py, line 561 in free
_internals.py, line 455 in unload_lora
_internals.py, line 484 in unload_all_loras
_internals.py, line 100 in close
OSError: exception: access violation reading 0x0000000000000028
```

Releasing the adapters first: `survived close()`.

Then in ComfyUI itself, two consecutive runs with `keep_model_loaded` off, so the teardown ran both times:

```
17:36:35  adapter registered as 'MiniMax-H3-Prompt-Rewriter-LoRA-F16' (1 loaded)
17:37:07  Prompt executed in 43.23 seconds
17:37:53  adapter registered as 'MiniMax-H3-Prompt-Rewriter-LoRA-F16' (1 loaded)
17:38:24  Prompt executed in 42.62 seconds
```

No access violation, VRAM returned, and the adapter is demonstrably attached: the base model writes ranges like `(0s-4s)`, the adapted one writes `At 00:05.800`.

---

## On the `output_language` request

I ran it rather than reasoned about it - 21 generations on Qwen3.6-27B Q5_K_M with the adapter, greedy throughout so a language that moves means the argmax moved, one resident model, Simplified Chinese, T2VA and Ref2VA.

**Where the report holds up, and strongly.** Ref2VA really does resist. The language rule alone - quoting and rebutting the guide's line 5 by name, exactly as you describe - produces **0.000** CJK. Not degraded: zero. And demonstration really is the lever that moves it.

**Where I would refine it: the demonstration works per section, not per answer.** A single `subject_definitions` exemplar flipped that section to 0.685 and `retention_analysis`, immediately after it, to 0.462, and left the other four in English. Per-section CJK share across five conditions:

| section | control | rule | rule first | rule last | + exemplar |
|---|---|---|---|---|---|
| subject_definitions | 0.000 | 0.000 | 0.000 | 0.000 | 0.685 |
| summary | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| retention_analysis | 0.000 | 0.000 | 0.000 | 0.000 | 0.462 |
| detailed_description | 0.000 | 0.000 | 0.785 | 0.827 | 0.818 |
| overall_soundscape | 0.000 | 0.000 | 1.000 | 1.000 | absent |
| non_diegetic_music | 0.000 | 0.000 | 0.000 | 0.000 | absent |

`summary` never moves. `non_diegetic_music` degenerates to `N/A` - 3 characters against 166 in the control - in every language condition. And the single-exemplar run lost two sections outright.

Six exemplars, one short opening line per section, fixes that. All six sections present and all six carrying Chinese, on two seeds; `summary` finally moves (0.558 and 0.387) with `[reference generation]` intact.

**Where your model and mine disagree.** On T2VA the rule alone is enough here: 0.806 on the LoRA path, 0.748 on the guided writer, greedy, and the exemplar adds nothing measurable (0.806 -> 0.800). So the exemplar is a fix for some builds rather than a universal mechanism - which matters mostly for how the option would have to be validated.

**Two more findings.** Ordering swings the structure while barely touching the language, and the best order differs per node: rule-first keeps two shots and the cut point on the LoRA path, rule-last does the same on the guided writer, neither wins on both. And putting a structural restatement *after* the language rule flipped a T2VA answer wholly back to English (0.000, full structure intact) - your effect, from the other side: English text late outvotes the rule.

**Good news on the scaffolding worry.** Across seven Ref2VA runs: zero full-width punctuation, zero translated field names, `fully_preserved` / `partially_preserved` verbatim every time, 20-25 asset labels intact, the task-type prefix present in every run, and the pack's own splitter parsed all seven. The only full-width colon I saw anywhere was on T2VA, inside prose, where it is harmless.

**Two things still open, and they are why the widget is not in 0.12.1.**

*Length.* Ref2VA came back at 1557 characters against 3229 for the English control, `detailed_description` 521 against 1071 - and that is with an explicit "700-1000 Chinese characters" target in the prompt, which was ignored. Chinese density accounts for maybe 1.6-2x, not this. Nothing I phrased moved it, including targets counted in Chinese units, which made the answer *shorter*.

*Bleed.* The two shortest exemplars came back copied verbatim, in both seeds, trailing ellipsis included. A short exemplar reads as a template rather than a demonstration, and the two sections that copied it are barely longer than the line they were given.

So the shape of the feature is not one combo plus one prompt string: it is per-section exemplars, tuned per language and per mode, with an ordering that differs per node. If you want to take that on as a PR I would be glad to review it - the per-section exemplars and the per-node ordering are the parts that need deciding, and you have already done the hardest thinking on both.



> **Comment by ioritree (2026-08-17T17:23:42Z):**

Thank you for the correction, and you are right that it cost you time — the report describes an API that does not exist upstream, and naming the build would have made that obvious in one line.

For anyone else who trips on this: the fork's wheel metadata is almost entirely inherited, which is exactly why the version alone reads as upstream.

```
Name: llama_cpp_python
Version: 0.3.47
Author-Email: Andrei Betlen <abetlen@gmail.com>     <- inherited, gives nothing away
Project-URL: Homepage, https://github.com/JamePeng/llama-cpp-python
Tag: cp312-cp312-win_amd64
```

`Name` and `Author-Email` are unchanged from upstream. Only `Project-URL` identifies it. I checked `__version__` and the signatures and never read the `dist-info`, which is the actual lesson.

Your negative control — reproducing the crash by calling `close()` with a populated registry before checking that the fix survives it — is a better standard than I met. I only confirmed the crash stopped, which does not establish that the change is load-bearing.

## Data points from a different build

These are the parts you cannot reproduce on your side, so they may be worth having. Same fork wheel, cp312 rather than cp313, but a different model and a different language: `Qwen3.8-27B-ABLITERATED-UNCENSORED-PHILADELPHIA-CLASS` Q4_K_M — labelled 3.8 but 64 blocks / 5120 wide, so `gguf_problem()` passes it and the adapter attaches. Traditional Chinese, Ref2VA on the guided writer.

**No section loss with a single exemplar.** Your single-exemplar run dropped `overall_soundscape` and `non_diegetic_music` entirely. Mine did not: two separate outputs, both parsed with the pack's own `split_fields` against `REF_OUTPUT_FIELDS`, both `missing: []`, all six sections present and carrying Chinese — including `summary`, which you report never moves.

**No bleed in six runs.** 4 sampled on random seeds plus 2 greedy. Greedy going Chinese is the part I would emphasise: on this build the exemplar moved the argmax itself, so it is not a coin toss here. That is the opposite of your T2VA result where the rule alone was already sufficient, and it supports your conclusion that the exemplar is a fix for some builds rather than a universal mechanism.

One possibly relevant difference: my exemplar is a single complete sentence with no trailing ellipsis, e.g.

```
Your answer therefore begins like this, and continues in the same language throughout:
subject_definitions:
<Subject 1> 是 <Picture 1> 中的年輕女子，深棕色長髮，身穿藍色開襟外套。
```

You describe the copied ones as coming back "trailing ellipsis included". An ellipsis is a strong signal that the line is a template to be filled rather than an opening to be continued.

## A guess connecting your two open problems

I suspect bleed and length collapse are the same phenomenon rather than two.

The evidence is in your own sentence: the sections that copied the exemplar are "barely longer than the line they were given". A short exemplar does not only set a language prior, it sets a **length** prior — the model reads it as the target shape, and imitates the size along with the language. That would also explain why length targets are ignored no matter how they are phrased, including in Chinese units: an instruction is arguing with a demonstration, which is the same asymmetry that made the exemplar necessary in the first place.

If that holds, the lever is not a better length instruction but an exemplar that demonstrates density: a full-length opening sentence, explicitly framed as an opening rather than a template, and no ellipsis. Testable against the per-section CJK numbers you already have.

## One path your 21 runs did not cover

All of them were on the guide path. The LoRA path never receives an exemplar, and its user message carries no language line at all:

```python
{"role": "system", "content": system_prompt(language)},
{"role": "user", "content": (
    f"resolution: {resolution}\n"
    f"duration: {duration}s\n"
    f"original_prompt: {prompt}"
)},
```

So where the guided writers have four reinforcements, the Rewriter node has one: the bullets appended to `SYSTEM_PROMPT`. On one run there I saw the first and last fields come back in Chinese with the middle one in English — consistent with per-section decisions, but with the adapter's English training pulling against a single line of defence. That is one observation, not a measurement, and the position after `original_prompt` is the obvious place to test if you decide it is worth it.

## On the PR

I would rather not take it, and the reason is capability rather than willingness.

The hard part of that feature is not the code, it is the validation — per-section exemplars across four languages and two modes, with an ordering that differs per node, judged on effects too small to see by eye. You have the harness for that; I have "run it a few times and look". That method has already misled me twice in this work: a single successful greedy run nearly convinced me the language option was fixed when it was still a coin toss. Shipping configuration into your repo that I cannot measure would be a liability.

There is also a concrete instance of it: my Simplified Chinese, Japanese and Korean exemplar lines have never been run once. I would be putting unvalidated content in front of your users.

And the per-node ordering is your architectural call — which node gets rule-first and which gets rule-last says something about how you see those nodes, and that is not mine to fix.

What I can offer instead is a second platform. You have one machine, one model, one language; this side has a different model on the same fork and a different script, and it is already producing results that diverge from yours in ways that matter to how the feature would have to be validated. If you want specific conditions run — an ordering, an exemplar shape, a language — say which, and I will run them and report the numbers rather than an impression.



> **Comment by pytraveler (2026-08-17T19:08:59Z):**

Good point about the `dist-info`. It reproduces here exactly:

```
Name           llama_cpp_python
Version        0.3.47
Author-Email   Andrei Betlen <abetlen@gmail.com>
Project-URL    Homepage, https://github.com/JamePeng/llama-cpp-python
```

`Name` and `Author-Email` come straight from upstream. Only `Project-URL` shows it is the fork.

## One correction back

> All of them were on the guide path. The LoRA path never receives an exemplar, and its user message carries no language line at all

That is true for the pack as shipped, but not for my runs. 9 of the 21 went through the LoRA path, and in the exemplar condition both the language line and the opening line were appended to the **user** message.

The conclusion also goes the other way: the LoRA path holds the language *better*, not worse. On T2VA, with the rule alone: 0.806 CJK on the LoRA path, 0.748 on the guided writer.

You are right about one part: all seven Ref2VA runs were on the guide path. But that is unavoidable - Ref2VA only exists in the guided writer, so there is no LoRA + Ref2VA combination to test.

## Testing your ellipsis idea

I ran it with only one thing changed. Same model, same language, same prompt bullets, same seeds. Only the exemplar text is different:

- **baseline**: short lines ending in `...` (my previous run)
- **A**: the same lines, finished as normal sentences, no `...`
- **B**: long, detailed sentences, no `...`

| | baseline | A | B |
|---|---|---|---|
| copied verbatim (2 seeds) | 2 and 2 | **0 and 1** | 0 and 0 |
| total characters | 1557 / 1483 | 2154 / 2011 | 1421 / 1416 |
| CJK share | 0.626 / 0.592 | **0.351 / 0.328** | 0.637 / 0.623 |
| sections lost | 0 | 0 | **2 and 2** |

**You are right about the copying.** Remove the `...` and it stops. Your explanation fits: a cut-off line looks like a template to fill in.

**But the length part does not hold.** If the exemplar sets a length target, longer exemplars should give longer sections. The opposite happens:

| section | exemplar -> output, A | exemplar -> output, B |
|---|---|---|
| subject_definitions | 46 -> 224 | 90 -> 217 |
| summary | 67 -> 300 | 133 -> 180 |
| retention_analysis | 40 -> 720 | 71 -> 417 |
| detailed_description | 60 -> 681 | 119 -> 529 |
| overall_soundscape | 29 -> 72 | 64 -> **0** |
| non_diegetic_music | 25 -> 35 | 63 -> **0** |

Every section came out *shorter* with the longer exemplar, and the last two were not written at all. So the copying and the short answers are two separate problems, not one.

## Removing the ellipsis has a cost too

CJK share per section:

| section | baseline | A/42 | A/12345 | B/42 | B/12345 |
|---|---|---|---|---|---|
| subject_definitions | 0.750 | 0.773 | 0.772 | 0.764 | 0.764 |
| summary | 0.558 | **0.000** | **0.000** | 0.453 | 0.500 |
| retention_analysis | 0.459 | **0.000** | **0.000** | 0.557 | 0.454 |
| detailed_description | 0.771 | 0.660 | 0.800 | 0.793 | 0.823 |
| overall_soundscape | 1.000 | 1.000 | 1.000 | absent | absent |
| non_diegetic_music | 1.000 | 1.000 | 1.000 | absent | absent |

With shape A, `summary` and `retention_analysis` went back to English - the two weakest sections. That also explains why A looks longer: `retention_analysis` grew from 421 to 720+ characters *because it switched to English*, and English simply takes more characters. Most of A's extra length is just those two English sections.

So the `...` seems to matter more than I thought: a cut-off line says "continue in this language", a finished sentence says "here is an example, now write your own" - and the language does not carry over.

Small extra regression: baseline had zero full-width punctuation, but A/12345 and both B runs wrote the speaker id with full-width parentheses (U+FF08/FF09) instead of ASCII. Nothing failed to parse, but it goes against the guide.

## Where this leaves the option

On my model, no exemplar shape wins:

- with `...`: language holds, all 6 sections present, but copying + short answers
- without `...`: no copying, but 2 sections flip back to English
- long and dense: language holds, but the answer stops after 4 sections

Your 6 clean runs do not contradict this - you changed model, language and exemplar shape at the same time, so maybe the trade-off simply does not exist on your build. That is the one open question.

## The ask

If the offer to run conditions still stands: **the same three exemplar shapes on your build and your language**. Ref2VA, two seeds each, per-section CJK share and section lengths. Nothing else changed.

That settles one question: is the ellipsis trade-off a property of the prompt, or of the model? If the prompt - the option needs an exemplar shape nobody has found yet. If the model - it needs per-model defaults, which is a very different feature.

On the PR: understood, and I think declining for that reason is the right call. A second measuring platform is worth more here than a patch.



> **Comment by ioritree (2026-08-18T17:34:29Z):**

# Three exemplar shapes on a second build, across four languages

Running the conditions you asked for, then extending them. 42 runs: 36 sampled,
plus a 6-run greedy round in the appendix.

**Short version.** The shape barely matters on this build — but the run turned up
two things that matter a lot more, and one of them is a partial vindication of
your ellipsis claim:

1. **한국어 falls back to English in 8 of 18 runs**, and which runs fail is decided
   by the seed, not by the shape. The exemplar mechanism, in any shape, does not
   control it.
2. **The only exemplar bleed in 42 runs landed in a `...` condition** — one run,
   and it took three sections down with it. Rare here, but it happened where you
   said it would.
3. Shape effect on language retention, where nothing fails: smaller than the
   difference between two seeds of the same shape.

So the answer to your question is *mostly* "property of the model" — with the
caveat that your copying finding did reproduce, once, in exactly the predicted
place.

## Setup

| | |
|---|---|
| Writer | `Qwen3.8-27B-abliterated` (prithivMLmods) `Q4_K_M`, GGUF, llama-cpp-python 0.3.47 |
| Captioner | the same file + its `mmproj-f16`, via `llama-mtmd-cli` b10310 |
| Mode | Ref2VA, guided writer |
| Languages | 繁體中文, 简体中文, 日本語, 한국어 |
| Card | RTX 3090, 24 GB |
| Decoding | sampled: temp 0.7, top_p 0.8, top_k 20 (a greedy round is in the appendix) |
| Seeds | 42 and 12345 everywhere; 한국어 also 7, 1234, 2026, 99999 |
| Held constant | `n_ctx` 8192, `max_new_tokens` 2048, 16:9, 10 s |

Three real images — two character sheets, one street scene — and a prompt
meaning "the characters from picture 1 and picture 2 fighting in the location of
picture 3, very intense", written in each target language.

Two things this run had to get right:

- **The reference block is captioned per language.** A Chinese block pulls a
  Japanese answer toward Chinese on its own, which would confound language with
  shape. Each language got its own block from the same three images, which is
  also how the nodes are used in practice — `caption_language` matching
  `output_language`. Block script share came out 0.842 / 0.873 / 0.880, so no
  language started from a compromised block.
- **Script share is per language.** Counting CJK ideographs measures Chinese;
  Japanese is largely kana and Korean is Hangul, neither of which is in that
  range. Japanese counts kana + kanji, Korean counts Hangul syllables and Jamo.

`LANGUAGE_EXAMPLES` was swapped in memory per condition; nothing on disk changed.

### The three shapes

This build's status quo is already your **A**, not your baseline — the local
`output_language` patch ships a finished sentence in every language's Ref2VA slot.
So `baseline` is that same line cut off with `...`, and `B` is a long dense one.

| language | baseline | A | B |
|---|---|---|---|
| 繁體中文 | 50 chars | 48 | 101 |
| 简体中文 | 50 | 48 | 101 |
| 日本語 | 58 | 56 | 126 |
| 한국어 | 67 | 65 | 156 |

`B` is longer in Japanese and Korean than in Chinese on purpose: those languages'
shipped exemplars are already longer, so a `B` capped at the Chinese length would
not have applied the same pressure.

## Result 1 — 한국어 is a coin toss, and no shape changes the odds

Script share, 3 shapes x 6 seeds. `0.0` means literally zero Hangul characters:
the whole answer came back in English, opening
`<Subject 1> is the cosplayer in <Picture 1>, featuring silver hair, blue horns…`

| seed | baseline | A | B |
|---|---|---|---|
| 7 | 0.578 | 0.627 | 0.597 |
| 42 | **0.0** | **0.0** | **0.0** |
| 1234 | 0.613 | 0.593 | 0.607 |
| 2026 | **0.0** | **0.0** | **0.0** |
| 12345 | 0.511 | 0.618 | 0.616 |
| 99999 | **0.0** | 0.591 | **0.0** |

| shape | failed |
|---|---|
| baseline | 3 / 6 |
| A | 2 / 6 |
| B | 3 / 6 |
| **total** | **8 / 18** |

**The failure is seed-locked, not shape-locked.** Seeds 42 and 2026 fail in all
three shapes; seeds 7, 1234 and 12345 succeed in all three. Exactly one cell
breaks the pattern — seed 99999 survives under A — which is 1 run out of 18 and
not a pattern to build on.

The failing runs are also much longer: 2478–2706 characters against 1496–1801 for
the clean ones. English simply takes more characters for the same content, the
same effect you saw in your A condition.

This looks like the thing the exemplar was added to fix, still unfixed in one
language. Chinese and Japanese never failed once in 18 sampled runs, nor in the 6
greedy ones; Korean fails 44% of
the time and the exemplar's shape is not the lever.

## Result 2 — the one bleed, and it is in the ellipsis condition

Across all 42 runs exactly one answer contains the exemplar's own wording:
**`简体中文 / baseline / seed 42`**, which wrote

```
年轻女子，深棕色长发，身穿蓝色开襟外套
```

into a `retention_analysis` entry. The reference images are a silver-haired
character and a pink-haired one. There is no brown hair and no blue cardigan
anywhere in the block — that is the exemplar, copied.

That run then failed in a way worth spelling out, because it is a chain and not a
coincidence:

1. it copied the exemplar's subject wording,
2. it invented `<Subject 4>` and `<Subject 5>` to hold the extra material,
3. it spent the 2048-token budget on that,
4. `detailed_description`, `overall_soundscape` and `non_diegetic_music` were
   never written — 542 characters total against 1170 for the same shape on the
   other seed.

It is the only run in 36 that lost a section, and the only one that copied. Both
happened in a `...` condition. **Your explanation — a cut-off line reads as a
template to fill in — is supported by this run.** What does not carry over is the
rate: 1 in 12 ellipsis runs here against 2 of 2 on your build.

## Result 3 — where nothing fails, the shape is under the noise

Script share by language and shape, seed 42 / seed 12345:

| language | baseline | A | B |
|---|---|---|---|
| 繁體中文 | 0.487 / 0.534 | 0.474 / 0.508 | 0.492 / 0.472 |
| 简体中文 | 0.452 / 0.546 | 0.519 / 0.543 | 0.498 / 0.532 |
| 日本語 | 0.591 / 0.628 | 0.658 / 0.599 | 0.626 / 0.612 |
| 한국어 | — / 0.511 | — / 0.618 | — / 0.616 |

For 繁體中文, where I have the tightest data: within-shape spread across two seeds
is 0.047 / 0.034 / 0.020, while the spread between the three shape means is
**0.028** — smaller than the seed spread of the baseline condition alone.

Mean script share of non-failing runs, by language: 繁體中文 0.494, 简体中文 0.515,
日本語 0.619, 한국어 0.582. Japanese scores highest partly because kana inflate the
count, so read that column as within-language only.

## What reproduced and what did not

**Your A: `summary` and `retention_analysis` → 0.000 CJK.** Did not reproduce.
Across 18 sampled Chinese and Japanese runs no section came back in English. The lowest
`summary` was 0.450 and the lowest `retention_analysis` 0.262 — and that floor is
structural, see the limits below.

**Your baseline: copied verbatim, 2 and 2.** Reproduced once in twelve.

**Your B: 2 sections lost, both seeds.** Did not reproduce. B never lost a
section in any language. The only lost sections were in a `...` run.

**Your B: every section shorter than A.** Did not reproduce. B is the longest
condition in 简体中文 (1265 / 1339 against A's 1155 / 1225) and in 日本語
(1629 / 1567 against 1393 / 1217).

**Measurement note**, because it cost me a wrong number first: the longest run
this build shares with the exemplar is 28 characters, and it is
`<Subject 1> 是 <Picture 1> 中的` — the opening the output contract mandates, not
the exemplar's wording. Counting it reports bleed in every run including the
clean ones. Everything above is measured after cutting that structural prefix,
per language, off the front of the exemplar.

## Reading

Two builds, opposite results on shape. That is the "property of the model" answer
to your question, and it argues against spending the option's design budget on
finding a better exemplar wording — there may be no wording right for both builds.

A guess at why the shape does so little here, offered as a testable prediction
rather than a conclusion: this is the Qwen3.8-27B-ABLITERATED case, the build
whose own chat template prepends a ~1,100-character English persona and which
`language_headline()` was written for. With the headline in the primacy slot and
`language_rule` rebutting the guide, the language may already be decided before
the exemplar is read, which would make the exemplar's shape irrelevant — and make
the exemplar carry more weight on a build where the headline is doing less.

The Korean result cuts against a purely positional story, though. If the headline
were deciding the language, it should decide it for Korean too, and it does not —
it decides it 56% of the time. That looks less like a prompt-position problem and
more like the model having weaker Korean than Chinese or Japanese, with the whole
language mechanism sitting on top of that and unable to compensate. If that is
right, per-model defaults would not fix Korean either; per-*language* fallback
detection might — noticing the answer came back in the wrong script and retrying
with a different seed, which is a different feature from anything in the option
today.

## Limits

- One model, one prompt, one set of three reference images.
- The exemplar wordings are my constructions, not your exact strings — same three
  shapes, different sentences, so "shape" is the only thing intended to carry
  across builds.
- Script share is *target-script characters / non-whitespace characters* over the
  raw section text. It counts the mandatory English scaffolding — field names,
  `[Shot N]`, timestamps, `fully_preserved` — against the total, which is why
  `retention_analysis` sits near 0.27 in every Chinese condition. That floor is
  structural, not a language failure. If your numbers use a different denominator
  the absolute values will not line up; the differences between conditions are
  the comparable part.
- 6 runs per language for 繁體中文 / 简体中文 / 日本語, 18 for 한국어. Enough to say
  "no shape effect visible" and to put a rough rate on the Korean failure, not
  enough to bound a small effect.

## Appendix — greedy

繁體中文 only, before the sampled round. Worth keeping because with `greedy=True`
the two seeds produced **byte-identical** text, so this is three runs, not six —
and it makes the same point without sampling in the way.

| | baseline | A | B |
|---|---|---|---|
| total characters | 1066 | 1201 | 1211 |
| CJK share | 0.462 | 0.500 | 0.498 |
| copied | 0 | 0 | 0 |
| sections lost | 0 | 0 | `non_diegetic_music` written as `N/A` |

`non_diegetic_music` comes back as the literal string `N/A` in several runs across
all three shapes and both decoding modes. It tracks the seed, not the shape, and
reads as the model deciding the shot needs no score — a different event from the
section not being written.



> **Comment by ioritree (2026-08-18T18:12:51Z):**

## Correction: "did not reproduce" was only true for Ref2VA

My last comment said your `summary` / `retention_analysis` → English finding did
not reproduce. That conclusion does not survive contact with the writer node
**without** references. On T2VA it reproduces thoroughly, and my experiment had a
variable in it I had not noticed.

**What I had not controlled.** Every one of those 42 runs was Ref2VA, so every one
carried a `reference_assets` block **written in the target language** in the user
message — three captioned pictures, 1.1–1.4k characters of 繁體中文 sitting
immediately before generation. I had measured the exemplar's shape while a much
larger language anchor was present in every condition. That is why I saw a floor
nothing could break: I was only ever testing the half of the space that has one.

Prompted by my collaborator reporting exactly your symptom on the plain writer
node, I ran T2VA with `reference_assets` empty. Same model, same language, same
settings, 6 seeds x 2 prompt lengths.

### Chinese on T2VA fails 5 times in 12

| length | seed | output chars | description | soundscape | music | verdict |
|---|---|---|---|---|---|---|
| short | 42 | 1116 | 0.0 | 0.0 | 0.0 | **all English** |
| short | 12345 | 367 | 1.0 | 1.0 | 1.0 | ok |
| short | 7 | 290 | 1.0 | 1.0 | 1.0 | ok |
| short | 1234 | 585 | **0.327** | 1.0 | 1.0 | **middle drift** |
| short | 2026 | 421 | 0.948 | 1.0 | 1.0 | ok |
| short | 99999 | 342 | 1.0 | 1.0 | 1.0 | ok |
| long | 42 | 1978 | 0.0 | 0.0 | 0.0 | **all English** |
| long | 12345 | 3522 | **0.024** | 1.0 | 1.0 | **middle drift** |
| long | 7 | 2099 | 0.022 | 0.0 | 0.0 | **all English** |
| long | 1234 | 817 | 0.922 | 1.0 | `N/A` | ok |
| long | 2026 | 776 | 0.953 | 1.0 | 1.0 | ok |
| long | 99999 | 574 | 1.0 | 1.0 | 1.0 | ok |

Figures are CJK over prose, with the contract's mandated English — `[Shot N]`,
timestamps, field names — stripped out first.

**Same model, same language, same `output_language`: 0 failures in 12 Ref2VA runs,
5 in 12 T2VA runs.** The reference block was doing the work I attributed to the
rest of the mechanism.

### Two distinct failure modes, not one

**All English.** The whole answer, every section. It is not length-dependent — seed
42 does it to `一個女子在跳舞`, a seven-character prompt. This is the same
seed-locked coin toss I found in 한국어, now showing up in Chinese as soon as the
reference block is gone.

**Middle drift — this is your finding.** `long / 12345`:

```
[Shot 1]  電影感，16:9 畫幅，空鏡頭。晨光透過高窗斜射進入老舊排練廳…      ← Chinese
[Shot 2]  At 00:02.000, the shot cuts to a medium shot of a young woman…  ← switches
[Shot 3…16]  …English…
overall_soundscape: 空氣中瀰漫著極其安靜的氛圍…                          ← Chinese again
non_diegetic_music: 稀疏的鋼琴音符以緩慢的速度演奏…                       ← Chinese
```

`integrated_multimodal_description` scores 0.024 while both tail sections score
1.000. It starts in the target language, loses it one shot in, runs 15 shots in
English, and recovers the moment a new field label appears.

That is your `summary` / `retention_analysis` observation with a different field
layout. T2VA has one long field instead of six, so the drift lands inside it
rather than between sections — but it is the same shape: the parts furthest from
both the instruction and the end are the parts that go.

### Length raises the risk, it does not gate it

Clean runs: 290, 342, 367, 421, 574, 776, 817 characters.
Failed runs: 585, 1116, 1978, 2099, 3522.

Four of the five failures are the longest outputs in the set, and the longest
clean run (817) is shorter than four of the five failures. But 585 drifted while
817 held, so length is a risk factor rather than a threshold. It fits a "grip
weakens with distance from the instruction" story better than a token count.

Worth noting `long / 12345` was 3522 characters, produced with `max_new_tokens`
raised to 4096. At the default 2048 it would have been truncated instead — the
failure would have looked like a budget problem.

### What I think this means for the option

My previous comment guessed the language was already decided before the exemplar
was read, because `language_headline()` holds the primacy slot on this build. The
T2VA numbers say that is not sufficient: the headline is present in all 12 of
these runs and the language still fails 5 times. What was actually carrying it in
the Ref2VA runs was a large block of target-language text inside the user message.

If that is right, the lever is neither the exemplar's shape nor a per-model
default. It is **how much target-language text the user turn contains, and how far
generation gets from it**. Two things follow that are cheap to test:

- Repeat the exemplar mid-prompt, or make it a short block rather than one line,
  and see whether T2VA's failure rate drops toward Ref2VA's.
- Detect the wrong script in the answer and retry with another seed. That catches
  "all English" cleanly, and catches middle drift too if it checks per field
  rather than over the whole answer.

The second is unglamorous but it is the only one of the two that would have caught
every failure in this set.

### Caveats

- One model — `Qwen3.8-27B-abliterated Q4_K_M`, the build with the ~1,100-character
  English persona in its own chat template.
- 12 runs per configuration. Enough for "this happens often", not for a rate with
  a confidence interval on it.
- Sampled decoding throughout: temp 0.7, top_p 0.8, top_k 20.
- The `1234 / long` run wrote `non_diegetic_music: N/A` rather than a description.
  That happens across conditions and reads as a judgement about the shot, not a
  language failure — I did not count it as one.



> **Comment by pytraveler (2026-08-18T19:25:14Z):**

Thank you for the correction comment especially - finding the flaw in your own
experiment and saying so is worth more than the original 42 runs.

I ran the test that follows from it. Every run on my side had been greedy, so I had
never checked whether this build coin-tosses at all.

## The sampling test

T2VA, no references, sampled (temp 0.7, top_p 0.8, top_k 20), your six seeds, both
writer paths, each in its best configuration from the earlier rounds. Only decoding
and seed changed.

| path | mode | language failures | CJK range |
|---|---|---|---|
| LoRA | greedy, seed 42 | 0 / 1 | 0.823 |
| LoRA | sampled, 6 seeds | **0 / 6** | 0.820 - 0.880 |
| guided | greedy, seed 42 | 0 / 1 | 0.822 |
| guided | sampled, 6 seeds | **0 / 6** | 0.751 - 0.791 |

No all-English run, no drift, and a very narrow spread. Seeds 42 and 2026 - the two
that fail in all three shapes on your build - pass here on both paths.

**So it is a property of the model.** Your 5 in 12 against my 0 in 12, same mode,
same language, same seeds.

That answers the main question, and it decides the option too. If a build coin-tosses the language, no exemplar wording fixes it. If it does
not, none is needed. Prompt wording is not the lever, so there is no point spending
more of the design budget on finding a better one.

One residual on my side, different in kind: 2 of the 12 sampled runs did not write
`overall_soundscape`. Sections go missing here even when the language holds.

## Two corrections to the mechanism

**The reference block.** My Ref2VA block was in English for every run - all five
lines, `Subject 1: young woman, mid-30s, shoulder-length dark hair...`. The language
still held at 0.626. So a target-language block in the user turn is not necessary
for the anchoring you describe. On this build something else was carrying it.

**Distance from the instruction.** That does not fit my data: the two furthest
sections, `overall_soundscape` and `non_diegetic_music`, held best of all at 1.000,
while sections 2 and 3 failed. What does fit is how much mandatory English each
section is required to carry:

| section | forced English | CJK in condition A |
|---|---|---|
| subject_definitions | 15.5% | 0.773 |
| **summary** | **26.0%** | **0.000** |
| **retention_analysis** | **21.5%** | **0.000** |
| detailed_description | 3.0% | 0.660 |
| overall_soundscape | 0.0% | 1.000 |
| non_diegetic_music | 0.0% | 1.000 |

The two sections with the most forced English are exactly the two that flipped.
That also fits your drift example: Shot 1 carries no timestamp - the guide forbids
one - Shot 2 does, and Shot 2 is where the language goes.

Honest limit: this only holds between sections. Inside a section it does not
replicate - two seeds, one supports it, the other contradicts it. So treat it as an
observation, not a rule.

## Where this leaves the option

Not shipping on prompt wording. The evidence across both builds: on mine the
wording is not needed, on yours it does not help.

If `output_language` happens, it will be your second proposal - check the script of
the answer, per field, and retry on a different seed when it comes back wrong. That
catches the all-English case and the drift case, it needs no per-language exemplar
tuning, and it is the only thing in this thread that would have caught every failure
either of us produced.

Thank you for all of it - the report, the 42 runs, and the correction. This is the
most carefully argued thread the project has had.




### #1 [closed] ISSUE: Bypass Option ??
URL: https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/issues/1  by davoodkharmanzar 2026-08-14T11:59:21Z

it would be great if add an **Bypass** Option to **MiniMax-H3 Reference Caption** :]

<img width="562" height="717" alt="Image" src="https://github.com/user-attachments/assets/a57d1d3c-3d96-4c96-8f23-f16eba4c2ebf" />

> **Comment by pytraveler (2026-08-14T18:02:49Z):**

Yes, no problem.


> **Comment by davoodkharmanzar (2026-08-14T18:20:42Z):**

nice ...

can you add **Bypass Option** to **MiniMax-H3 Prompt Writer** too?


> **Comment by pytraveler (2026-08-14T18:23:16Z):**

Yep 😄 


> **Comment by pytraveler (2026-08-14T19:59:38Z):**

<img width="450" alt="bypass toggle" src="https://github.com/user-attachments/assets/104d992a-2d72-4057-af3e-23521df5bb20" />

[Ready - v0.9.5](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI/releases/tag/v0.9.5).



