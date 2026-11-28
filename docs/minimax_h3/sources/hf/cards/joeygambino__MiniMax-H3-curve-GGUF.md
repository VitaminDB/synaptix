# joeygambino/MiniMax-H3-curve-GGUF

Источник: https://huggingface.co/joeygambino/MiniMax-H3-curve-GGUF · скачано 2026-11-28
Загрузок: 11182 · лайков: 10 · базовая: MiniMaxAI/MiniMax-H3 · теги: gguf,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-14T11:57:37.000Z

---

---
license: other
license_name: minimax-h3-community
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3
tags:
- gguf
- video
- audio
- text-to-video
- image-to-video
- comfyui
- minimax-h3
---

# MiniMax-H3 GGUF - curve form (fl2va + ref2va)

GGUF quantizations of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)'s
33B video+audio DiTs, built from MiniMax's **pruned** checkpoints. Same model,
same quant tiers as the [original-form
repo](https://huggingface.co/joeygambino/MiniMax-H3-GGUF), **about 40% smaller**
- and a Q8_0 that fits a 24 GB card.

| File | Size | For |
|---|---|---|
| **fl2va curve Q8_0** | **21.5 GB** | best quality that still fits 24 GB |
| **fl2va curve Q5_1** | **15.2 GB** | the everyday choice, fully resident on 16 GB |
| **fl2va curve Q4_0** | **11.5 GB** | 12 GB cards |
| **ref2va curve Q8_0 / Q5_1 / Q4_0** | 21.5 / 15.2 / 11.5 GB | same tiers, reference workflows |

`fl2va` = text/first-last-frame to video+audio (T2V and I2V).
`ref2va` = reference-conditioned generation (identity from up to 9 images,
3 videos with soundtracks, 3 voice clips).

Runs in ComfyUI with [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
plus a one-line architecture patch (node pack:
[**ComfyUI-H3-Multishot**](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot); workflows:
[**MiniMax-H3-Multishot-Workflow**](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow)).

## Requires ComfyUI 0.30.0 or newer

On anything older these will **not load at all**, because the shape of the
modulation weights changed and older builds do not know how to read them. That
is the only catch - everything else is a drop-in swap: same node, same workflow,
same filenames apart from `-curve-`.

If you are pinned to an older ComfyUI, or you would simply rather run the larger
quants, the original-form files are still maintained at
[**joeygambino/MiniMax-H3-GGUF**](https://huggingface.co/joeygambino/MiniMax-H3-GGUF).
That repo is not going anywhere.

## Where the 40% went

About 40% of this model was never unique data. Each block carried a
`96768 x 2688` adaln modulation matrix, and MiniMax's pruned checkpoints show
those matrices are a smooth function of the timestep rather than 51 independent
tensors. The pruned form factorizes each one to `96768 x 8` and adds a single
shared `adaln_t_table` to reconstruct them. Same math, ~26 GB less file.

This is why Q8_0 exists here and not in the original repo. In the original form
it was pointless - it landed at roughly the size of the official int8 release.
Factored down, Q8_0 is **smaller than the Q5_1 in that repo** (21.5 GB vs
25.9 GB).

## Is it actually the same?

Checked by rendering, not by assuming. Every comparison below is the same graph
and the same seed, changing only the model file:

| | vs the original-form Q5_1 render |
|---|---|
| same model, re-run (noise floor) | **0.00** |
| original Q5_1 -> original Q4_0 (one quant tier) | 29.64 |
| **curve Q5_1** | **10.83** |
| **curve Q8_0** | 17.06 |
| **curve ref2va Q5_1** (vs its own baseline) | **10.98** |

Numbers are mean absolute pixel difference over all 124 frames. The noise floor
being exactly 0.00 is what makes the rest meaningful - sampling is deterministic
at a fixed seed, so these are real differences and not run-to-run variation.
Swapping to curve form moves the output about a third as far as dropping one
quant tier does, and per-frame motion stayed in family in every case (a frozen
render would otherwise score a deceptively good pixel distance).

The Q8_0 figure is a *distance from a Q5_1 baseline*, not an error measure - a
higher-fidelity file should sit further from Q5_1, not closer. Proving the
direction would need an F16 reference render, and the F16 curve file is 37.5 GB,
so that is untested rather than assumed.

## Speed

RTX 5090, 124 frames at 480x864, 20 steps:

    original Q5_1   3.7 min
    curve    Q5_1   3.1 min
    curve    Q8_0   3.1 min

The curve files are faster here for the obvious reason - less weight to move.

If a render is far slower than this, check **power draw** rather than
utilization. A card thrashing weights between system RAM and VRAM still reports
~98% utilization while drawing a fraction of its rated power; that is the signal
the model did not fit, and the utilization figure will not tell you.

## Why there is no Q6_K / Q5_K / Q4_K / Q3_K

**K-quants are architecturally impossible for this model.** H3's hidden width is
2688, and K-quants require weight rows divisible by 256 (2688 % 256 = 128), so
requesting one just quantizes something else with the wrong name on it. The
legal ladder is the classic family: **Q8_0**, **Q5_1**, Q5_0, Q4_1, **Q4_0**.
Q5_0 and Q4_1 are buildable and simply have not been - they would land between
files already here.

## Build notes

- `llama-quantize` **cannot** build these. The factored adaln rows are only 8
  wide, and no GGUF block type can represent a row of 8 (they all work in
  32-element blocks). It tags them with the requested type regardless, and the
  reference reader then refuses the file (`row size 8 is not a multiple of Q5_1
  block size 32`); Q4_0 fails outright, writing 0 bytes. These were built with a
  small Python writer that skips what it cannot legally quantize.
- Those 52 short-row tensors are stored **F32**, matching the reference
  converter. That costs ~155 MB and is not optional: storing them F16 produces a
  structurally valid file that dies at sampling with `expected dtype struct
  c10::Half for 'weight'`.
- Kept at full precision throughout: patch projections, condition projection,
  final layer, token refiner.

## You also need the text encoder

A GGUF here contains the transformer and **nothing else**. The DiT alone will
not generate anything.

* **Text encoder:**
  [joeygambino/MiniMax-H3-encoder-GGUF](https://huggingface.co/joeygambino/MiniMax-H3-encoder-GGUF)
  - Qwen3-VL-32B, Q4_K_M (19.8 GB) or Q5_K_M (23.2 GB). **Take the `mmproj`
  file with it.** It is required for image conditioning *and* for multi-shot
  chaining, which feeds the previous shot's last frame through the encoder's
  vision path. Keep both filenames as downloaded so the loader pairs them.
  Also on Civitai: [Text Encoder GGUF](https://civitai.com/models/2834385).
* **VAEs:** [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)

Note the encoder *can* have K-quants even though this DiT cannot - it is a stock
Qwen3-VL, so its rows are 256-divisible.

The encoder and the DiT together do not fit on a 32 GB card. If your workflow
holds both, evict the encoder once conditioning is computed - the node pack's
keyframe and multishot nodes do this for you, and it is worth roughly a 4x
difference in render time.

## Support

Everything I publish is free and stays free. If it saved you a night of
debugging, tips keep the 5090 warm:

* [Buy me a coffee on Ko-fi](https://ko-fi.com/joeygambino)
* [Sponsor on GitHub](https://github.com/sponsors/jlucasmcrell)
* [Liberapay](https://liberapay.com/joeygambino) (recurring)

## On Civitai

This set also lives on Civitai with the measurement tables: [curve-form GGUF](https://civitai.com/models/2835678/minimax-h3-curve-form-gguf-fl2va-ref2va-low-vram). The original-form set is [MiniMax-H3-GGUF](https://huggingface.co/joeygambino/MiniMax-H3-GGUF).
