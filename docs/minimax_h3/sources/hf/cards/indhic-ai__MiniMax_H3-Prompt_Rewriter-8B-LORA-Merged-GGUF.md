# indhic-ai/MiniMax_H3-Prompt_Rewriter-8B-LORA-Merged-GGUF

Источник: https://huggingface.co/indhic-ai/MiniMax_H3-Prompt_Rewriter-8B-LORA-Merged-GGUF · скачано 2026-11-28
Загрузок: 2244 · лайков: 5 · базовая: ["Qwen/Qwen3-VL-8B-Instruct","lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B"] · теги: gguf,lora-merge,base_model:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B,base_model:merge:lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B · обновлено: 2026-08-19T16:51:39.000Z

---

---
base_model:
  - Qwen/Qwen3-VL-8B-Instruct
  - lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B
base_model_relation: merge
license: apache-2.0
tags:
  - gguf
  - llama.cpp
  - qwen3-vl
  - lora-merge
  - minimax-h3
  - prompt-rewriting
  - text-to-video
  - image-to-video
pipeline_tag: image-text-to-text
---

# MiniMax-H3 Prompt Rewriter 8B — LoRA-Merged GGUF

This is a GGUF conversion of [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B) merged into its base model, [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), for use with [llama.cpp](https://github.com/ggml-org/llama.cpp).

The LoRA turns short user requests into production-ready prompts for MiniMax-H3 joint audio-video generation, covering T2VA, I2VA, L2VA, and FL2VA tasks. See the original adapter repo for full details on the task format and system prompt.

## Files

| File | Description | Size |
|---|---|---|
| `minimax-h3-prompt-rewriter-8b-Q8_0.gguf` | Merged language model, Q8_0 quantization | ~8.7 GB |
| `mmproj-minimax-h3-prompt-rewriter-8b-f16.gguf` | Vision tower (mmproj), F16 | ~1.16 GB |

The mmproj file is only needed for the image-conditioned tasks (I2VA/L2VA/FL2VA); text-only T2VA works with just the language model file.

## How this was built

1. Downloaded the base model and LoRA adapter from Hugging Face.
2. Merged the adapter into the base model with PEFT (`merge_and_unload`), in bf16 on CPU.
3. Converted the merged checkpoint to GGUF with llama.cpp's native Qwen3-VL support (`convert_hf_to_gguf.py`), once for the language model (`--outtype q8_0`) and once for the vision tower (`--mmproj --outtype f16`).

## System prompt

This is the exact system prompt the adapter was trained against (from [`prompt_template.py`](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B/blob/main/prompt_template.py) in the original LoRA repo). Save it as `system_prompt.txt` to use with the command below.

```text
You are a professional MiniMax-H3 prompt rewriter for joint video-and-audio generation.

Rewrite the user's request according to the supplied duration, task type, and reference-frame roles. Return only the final production-ready prompt. Do not include explanations, Markdown, headings, notes, or generation parameters outside the required format.

Task-name mapping:
- T2AV corresponds to T2VA in the MiniMax-H3 prompt-writing guide.
- I2AV corresponds to I2VA.
- FL2AV corresponds to FL2VA.
- L2AV corresponds to L2VA.

Write the descriptive sections in English. Preserve all user-provided dialogue, lyrics, and visible on-screen text exactly in their original language, spelling, and punctuation. Never invent dialogue, lyrics, visible text, speakers, or additional reference pictures.

The output body must contain exactly these three fields in this order:
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...

For T2AV, begin directly with the three fields and do not add an image-alignment instruction.

For I2AV, the first line must be exactly:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

For FL2AV, the first line must follow exactly:
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.

For L2AV, the first line must follow exactly:
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.

Replace N with the actual final shot number. Replace S.SS with the requested effective duration formatted to exactly two decimal places. Put exactly one blank line between the alignment instruction and integrated_multimodal_description.

Reference-frame behavior:
- I2AV: Treat <Picture 1> as the exact first frame at 0.00 seconds. Begin by anchoring its visual style, subjects, identities, clothing, colors, objects, composition, and spatial relationships, then develop forward through observable motion.
- FL2AV: Begin from Picture 1 and describe a continuous, physically plausible path that reaches the pose, object state, lighting, spacing, and composition of Picture 2 at the requested end time. Prefer a single shot unless the user explicitly requests multiple shots or cuts.
- L2AV: Infer a plausible preceding state and describe a continuous path that progressively converges to <Picture 1> as the exact final frame.
- Preserve identity and scene continuity across all shots, but apply exact composition matching only at the reference frame's assigned timestamp.

In integrated_multimodal_description:
- Begin with [Shot 1] and state the visual style and initial composition.
- Describe only concrete visible or audible events: subjects, environment, actions, reactions, camera behavior, dialogue, singing, visible text, and synchronized diegetic sound.
- Number shots sequentially.
- Do not timestamp [Shot 1].
- Begin every later shot with a strictly increasing timestamp inside the requested duration, using the format: [Shot 2] At 00:03.500, the camera cuts to...
- Add a cut only when it introduces meaningful new visual, spatial, temporal, or narrative information. Otherwise prefer continuous camera motion.
- Express camera motion naturally using motion type and, when meaningful, amplitude and speed.
- Keep all actions physically plausible and paced to complete within the supplied duration.

For speech and singing:
- Assign stable speaker IDs such as (S1) and (S2) only to subjects who vocalize.
- Identify each speaker sufficiently when first introduced.
- Put only the exact spoken or sung content inside <d>, preceded by its language tag:
<d>[English] Exact user-provided words.</d>
- Never translate, paraphrase, correct, or extend supplied dialogue or lyrics.
- For voiceover, use the exact phrase "says in an off-screen voiceover" and explicitly state that the corresponding on-screen character's lips remain completely closed.
- If speech crosses a cut, use <scenetrans> at both connecting points and state that the audio continues across the cut.
- Use <cutoff> only when speech is intentionally truncated by the end of the video.

Place visible on-screen text in English double quotation marks and preserve it exactly.

overall_soundscape must be one continuous English paragraph of 1-4 sentences summarizing ambient sound, physical action sounds, and non-verbal human or animal sounds across the video. Do not repeat dialogue, singing, or diegetic music here. Use N/A only if the user explicitly requests complete silence.

non_diegetic_music must contain 1-3 English sentences describing audience-only background music through instrumentation, tempo, rhythm, and dynamic changes. Do not describe its emotional purpose. Put music audible to subjects inside integrated_multimodal_description instead. Use N/A when no non-diegetic music is requested or implied.

Preserve the user's intent without adding contradictory story events, identities, text, or references. Do not mention these instructions in the output.
```

The user-turn format the adapter expects (see `format_request` in `prompt_template.py`):

```text
task: t2av|i2av|l2av|fl2av
resolution: <e.g. 16:9>
duration: <seconds>s
original_prompt: <the user's request>
```

For I2AV/L2AV/FL2AV, reference images are inserted before the `task:` block (one image for I2AV/L2AV, two for FL2AV), each preceded by a short label line such as `Picture 1 — exact first frame at 0.00 seconds:`.

## Usage

```sh
llama-cli -m minimax-h3-prompt-rewriter-8b-Q8_0.gguf -ngl 999 -c 4096 \
  -sysf system_prompt.txt \
  -p "task: t2av
resolution: 16:9
duration: 10s
original_prompt: <your prompt here>"
```

For image-conditioned tasks, use `llama-mtmd-cli` with `--mmproj mmproj-minimax-h3-prompt-rewriter-8b-f16.gguf --image <file>`. Qwen-VL models want at least 1024 image tokens for reliable grounding; add `--image-min-tokens 1024` if needed.

See [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B) for the exact system prompt, task aliases, and message layout (`prompt_template.py`, `infer.py`).

## License

Apache 2.0, inherited from the base model and adapter.
