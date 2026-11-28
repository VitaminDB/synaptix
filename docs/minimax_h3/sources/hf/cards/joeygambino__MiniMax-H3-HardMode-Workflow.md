# joeygambino/MiniMax-H3-HardMode-Workflow

Источник: https://huggingface.co/joeygambino/MiniMax-H3-HardMode-Workflow · скачано 2026-11-28
Загрузок: 0 · лайков: 5 · базовая: MiniMaxAI/MiniMax-H3 · теги: comfyui · обновлено: 2026-08-07T20:14:03.000Z

---

---
license: other
license_name: minimax-h3-community
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3
tags:
- comfyui
- workflow
- minimax-h3
- video
- audio
- reference-to-video
---

# MiniMax-H3 Hard Mode — reference to video

Identity from **reference images, video and audio** instead of a start frame.
Up to **9 images, 3 videos, 3 soundtracks and 3 standalone audio clips**.

This is the `ref2va` counterpart to the
[multishot workflows](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow).
Where those chain shots from a start frame, this one carries a person, a place or
a voice from reference material you supply.

Also on **[Civitai](https://civitai.com/models/2836967)**.

## v1.1 — chained shots, one master, with audio

By request: the Multishot release's chaining, driven by your references.
`H3_HardMode_Chained.json` takes one script — one prompt per shot, `---`
between them. Shot 1 renders in Hard Mode from your reference images (and
optional voice clip); every later shot continues image-to-video from the
previous shot's last frame, so the identity cast in shot 1 carries down the
chain in pixels; the segments join in-graph into one master with audio.
Your `<Picture N>` binding lines go in their own box and attach to shot 1
automatically. Needs the
[node pack](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot) at
**v1.5+**. At least 2 prompts; for a single shot use `H3_HardMode_R2V.json`.

## What is here

| file | |
|---|---|
| `H3_HardMode_R2V.json` | the reference-to-video graph, every node labelled |
| `workflows/` | the multishot AIO, long-form memory and keyframes graphs — **also in the [multishot pack](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow)**, relabelled here so nothing shows a raw class name |
| `reference_images/` | three views of one synthetic subject, so the graph works on first load |
| `HARD_MODE_GUIDE.html` | the full guide |
| `INSTALL.txt` | setup, and the traps |

## You need the node pack — and nothing else custom

Every node in the graph is either ComfyUI core or from
[**ComfyUI-H3-Multishot**](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot):
`H3ModelLoaderAny`, `H3ClipLoaderAny`, `H3ConditionStrength`,
`H3FreeTextEncoder`, `H3ReferenceAudio`. No other custom packs.

Install it, then run `python apply_gguf_arch_patch.py` from its folder and
restart. Without that patch, loading an H3 GGUF fails with
`ValueError: Unexpected architecture type in GGUF file: 'minimax_h3'` —
ComfyUI-GGUF keeps a hardcoded architecture list and this one is not on it.

You also need a **ref2va** DiT ([curve form](https://huggingface.co/joeygambino/MiniMax-H3-curve-GGUF)
or [original](https://huggingface.co/joeygambino/MiniMax-H3-GGUF)), the
[text encoder](https://huggingface.co/joeygambino/MiniMax-H3-encoder-GGUF) with its
`mmproj` sidecar, and the [VAEs](https://huggingface.co/Comfy-Org/MiniMax-H3).
`fl2va` files will not work in this mode.

Load the encoder with `H3ClipLoaderAny` at **type = minimax**, not with
"CLIPLoader (GGUF)" — that node has no minimax type, falls back silently, and
dies inside a tokenizer with an unrelated-looking JSON error.

## Naming your references in the prompt

The part that is not guessable. Labels are **1-based**, input slots are
**0-based**, so `ref_image_0` is `<Picture 1>`:

```
<Picture 1> is the woman. <Picture 2> is the room she is standing in.
She walks to the window in the style of <Video 1>.
```

Ordering is fixed by the node however you wire it: **all images, then videos,
then standalone audios**. A reference video *with a soundtrack* consumes an
`<Audio>` ordinal **before** your standalone clips — one ref video with sound
plus one `ref_audio_0` makes your voice clip `<Audio 2>`, not `<Audio 1>`. Get
that wrong and the wrong voice binds, with no error.

## References and keyframes cannot be combined

A hard limit in ComfyUI core, not a choice made here. `model_base.py` writes
`cond_video_latents` for keyframes, then *assigns* it again for references, so
keyframe latents are discarded while the keyframe layout rows survive. The packed
sequence desyncs and the sampler dies on a shape mismatch. Use the keyframes graph
when you need a specific opening or closing frame.

## Settings, measured

- **Leave the eviction node ON.** It frees the ~16.5 GB text encoder before the
  DiT loads. Images only: **3.4 min** with it against 4.5 without, and the output
  is **bit-identical** — every pairing scored 0.00 mean pixel distance. With a
  reference video it is 6.5 min against a 46-minute run that was killed. If you
  ever disable it use **Bypass**, never Mute: a muted node produces nothing and
  the graph stops validating.
- **`ref_image_size = match`, not `max`.** Same subject and seed: `max` cost 24%
  more time and rendered a facial scar as a vague smudge on the *wrong* eyebrow,
  with less fine detail.
- **Prefer more references over bigger ones.** A third reference image cost
  nothing measurable — 2.82 min against 2.87.
- **Reference audio must be STEREO.** The audio VAE encodes `[B, 2, L]`; a mono
  clip produces half the rows it reserved and dies deep in the model with no
  useful error. The stereo-guard node handles it. Audio costs ~14% wall clock and
  trims motion ~14–27%.
- **Reference video:** minimum 5 frames, truncated to the generation length then
  trimmed to `n % 17 == 5`, shown to the encoder at 2 fps.

## If a render crawls

Check **power draw**, not utilisation. A card thrashing weights between system
RAM and VRAM still reports ~98% while pulling a fraction of its rated watts.
Sample for 20–30 s though: the encoder phase at the start of every render is
legitimately quiet, around 100 W, and one unlucky reading there looks exactly like
thrashing.

## Licence

Workflows MIT. Weights under MiniMax's licence. The reference images are
synthetic, made for this release, and free to use.

## Support

Everything I publish is free and stays free.

* [Ko-fi](https://ko-fi.com/joeygambino)
* [GitHub Sponsors](https://github.com/sponsors/jlucasmcrell)
* [Liberapay](https://liberapay.com/joeygambino)
