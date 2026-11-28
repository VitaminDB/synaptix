# joeygambino/MiniMax-H3-GGUF

Источник: https://huggingface.co/joeygambino/MiniMax-H3-GGUF · скачано 2026-11-28
Загрузок: 20744 · лайков: 10 · базовая: MiniMaxAI/MiniMax-H3 · теги: gguf,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-14T11:57:36.000Z

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

# MiniMax-H3 GGUF (fl2va + ref2va)

GGUF quantizations of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)'s
33B video+audio DiTs, quantized from the
[Comfy-Org bf16 repackages](https://huggingface.co/Comfy-Org/MiniMax-H3).
Runs in ComfyUI with [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
plus a one-line architecture patch (node pack:
[**ComfyUI-H3-Multishot**](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot); workflows:
[**MiniMax-H3-Multishot-Workflow**](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow)).

> **Want smaller files?** There is a second set built from MiniMax's pruned
> checkpoints, ~40% smaller at the same quant tier, in
> [**MiniMax-H3-curve-GGUF**](https://huggingface.co/joeygambino/MiniMax-H3-curve-GGUF).
> They need **ComfyUI 0.30.0+**, and they are a different trade, not a straight
> replacement - so this repo stays exactly as it is. Stay here if you are on an
> older ComfyUI, or if you would rather run the larger quants.

| File | Size | Bits/weight | For |
|---|---|---|---|
| fl2va Q5_1 | 25.9 GB | 6.0 | 24-32 GB cards (near-Q6 quality) |
| fl2va Q4_0 | 19.9 GB | 4.5 | 16 GB cards (streams the overflow) |
| ref2va Q5_1 | 25.9 GB | 6.0 | reference/identity workflows |
| ref2va Q4_0 | 19.9 GB | 4.5 | same, low-VRAM |

`fl2va` = text/first-last-frame to video+audio (T2V and I2V).
`ref2va` = reference-conditioned generation (identity from up to 9 images,
3 videos with soundtracks, 3 voice clips).

## Troubleshooting: "Unexpected architecture type in GGUF file: 'minimax_h3'"

If ComfyUI throws this the moment it tries to load the file, **the quant is
fine** - ComfyUI-GGUF checks a GGUF's architecture against a fixed list and
rejects anything not on it before reading a single tensor, and upstream's list
does not yet include `minimax_h3`.

The fix is to install (or update to) **ComfyUI-H3-Multishot v1.5.2 or newer**:

    https://github.com/jlucasmcrell/ComfyUI-H3-Multishot

From v1.5.2 the pack teaches ComfyUI-GGUF that architecture automatically, in
memory, at every ComfyUI start - no files are modified and the fix survives
ComfyUI-GGUF updates. You should see this in the ComfyUI console on startup:

    [H3] taught ComfyUI-GGUF the 'minimax_h3' architecture

If you still hit the error, the pack is not being loaded. As a fallback you can
patch ComfyUI-GGUF on disk with `python apply_gguf_arch_patch.py` from the
pack's folder, then restart.

*(Earlier pack versions required that script as a manual install step. It was
too easy to miss, which is why several people reasonably concluded these quants
were broken. Sorry about that - it is automatic now.)*

## Sample

30 seconds, three chained shots, one script, rendered on the fl2va **Q5_1**
quant - same presenter, same voice, across both seams:

<video controls src="https://huggingface.co/joeygambino/MiniMax-H3-GGUF/resolve/main/samples/H3_multishot_presenter_demo.mp4"></video>

## Workflows

The multishot workflows (easy mode + expert) live in their own repo:
[**MiniMax-H3-Multishot-Workflow**](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow) - with the node pack at
[ComfyUI-H3-Multishot on GitHub](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot).

## Why there is no Q6_K / Q5_K / Q4_K / Q3_K

**K-quants are architecturally impossible for this model.** H3's hidden width
is 2688, and K-quants require weight rows divisible by 256
(2688 % 256 = 128), so requesting one just quantizes something else with
the wrong name on it.

The ladder for pure quants is the classic family: Q8_0, **Q5_1**, Q5_0,
**Q4_0**. Q8_0 is not built *for these original-form files* on **size** grounds
- it lands at about the size of MiniMax's own `int8_convrot` release, so it would
give you a similar download for no obvious reason to prefer it.

To be clear, since this has been read the other way: that is a statement about
file size, **not** a claim that Q8_0 is better or worse than the official int8.
They are closer than the names suggest. `int8_convrot` averages 8.22 bits per
parameter and leaves 272 tensors at BF16, while Q8_0 is 8.5 bits per weight - so
Q8_0 actually carries slightly *more* bits. Q8_0 has much finer granularity (a
scale every 32 weights, against per-tensor scales), while the rotation in
`convrot` spreads outliers before quantizing, which is exactly why an official
release can ship at 8 bits flat. Those pull in opposite directions and I have not
measured which wins.

That caveat is worth spelling out, because it stops being true in the pruned
form: factored down, a Q8_0 fits in 21.5 GB, which is *smaller* than the Q5_1
in this repo. If you want it, it is in
[MiniMax-H3-curve-GGUF](https://huggingface.co/joeygambino/MiniMax-H3-curve-GGUF).

**One useful exception**: H3's FFN down-projections have 256-divisible rows,
so a *mixed* file (real K-quants on those tensors, classic types on
everything else) can land below the pure-classic floor: **Q3mix, 17.3 GB** -
light streaming on 16 GB cards, fully resident with headroom on 20 GB+.
Publishing after it passes a quality gate against Q4_0.

## Quantization notes

- ~40% of this model is per-block adaln modulation (96768x2688 per block).
  It is quantized here, as it is in the official int8_convrot release.
- Kept at full precision: patch projections, time embedder, condition
  projection, final layer, token refiner (all genuinely small).
- Built with a patched llama.cpp adding a first-class `minimax_h3`
  architecture (no mislabeled arch strings in the metadata).

## VRAM guidance

The DiT does not need to fit entirely in VRAM - ComfyUI streams what does
not fit. Measured on an RTX 5090 (32 GB): Q5_1 runs with ~22 GB resident,
~10.3 min for a 243-frame 544x960 shot at 20 steps. On 16 GB cards use
Q4_0 and expect slower steps from streaming, not failure.

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

Note the encoder *can* have K-quants even though this DiT cannot - it is a
stock Qwen3-VL, so its rows are 256-divisible.


## Support

Everything I publish is free and stays free. If it saved you a night of
debugging, tips keep the 5090 warm:

* [Buy me a coffee on Ko-fi](https://ko-fi.com/joeygambino)
* [Sponsor on GitHub](https://github.com/sponsors/jlucasmcrell)
* [Liberapay](https://liberapay.com/joeygambino) (recurring)

## Curve-form variant

There is a second GGUF set cut from the curve-form (pruned) bf16 - smaller at the same quant level, needs ComfyUI 0.30+. [Hugging Face](https://huggingface.co/joeygambino/MiniMax-H3-curve-GGUF) | [Civitai](https://civitai.com/models/2835678/minimax-h3-curve-form-gguf-fl2va-ref2va-low-vram)
