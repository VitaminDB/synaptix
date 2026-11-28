# joeygambino/MiniMax-H3-comfy-native-fl2va

Источник: https://huggingface.co/joeygambino/MiniMax-H3-comfy-native-fl2va · скачано 2026-11-28
Загрузок: 13865 · лайков: 2 · базовая: MiniMaxAI/MiniMax-H3 · теги: comfyui,quantized,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-15T19:32:13.000Z

---

---
license: apache-2.0
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: quantized
pipeline_tag: text-to-video
library_name: diffusion-single-file
tags:
  - minimax-h3
  - comfyui
  - quantized
  - video
  - fl2va
---

# MiniMax-H3 fl2va — comfy-native quantisations

The `fl2va` half of the comfy-native set. If you want `ref2va`, it is
[over here](https://civitai.com/models/2857809) —
same formats, same method, different job.

**Stock ComfyUI 0.32+ loads these directly.** No ComfyUI-GGUF, no custom loader
node, no dequantise-on-the-fly. ComfyUI carries its own quantisation system
(`comfy/quant_ops.py` plus the `comfy_kitchen` kernels) and these files use it,
so the kernels run in the quantised domain instead of unpacking to bf16 first.

## Which one is fl2va, and why you would want it

`fl2va` is the first/last-frame variant. It has **no reference rows**, so voice
anchoring, the identity bank and reference images do nothing on it. In exchange
it does the one thing `ref2va` cannot:

**it lands on a supplied frame.**

Measured against the same target frame:

* `fl2va` — **26.35 dB**
* `ref2va` + keyframe, turbo 6 steps — 16.15 dB
* `ref2va` + keyframe, stock 20 steps — 16.81 dB

The stock-20-step control rules out the sampler. The checkpoints genuinely
differ, by about 10 dB. `ref2va` treats a keyframe as a suggestion; `fl2va`
treats it as the task.

It can also take a first **and** a last frame and plan a camera move between
them, which `ref2va` has no mechanism for at all.

So: **`ref2va` when a face or a voice has to persist across shots, `fl2va` when
a shot has to start exactly where the last one ended.** Stepping through a scene
one shot at a time — render, look at it, feed its last frame to the next — is
the case `fl2va` is built for.

Note this is **not** a size decision. fl2va and ref2va are byte-identical in
size at every quantisation. "Lighter" only ever meant fewer tokens per sampling
step, because there are no reference rows riding along.

## The files

```
MiniMax-H3-fl2va-pruned-comfy-nvfp4.safetensors    11.67 GB   4-bit
MiniMax-H3-fl2va-pruned-comfy-w4a8.safetensors     11.68 GB   4-bit
MiniMax-H3-fl2va-pruned-comfy-int8.safetensors     19.53 GB   8-bit
MiniMax-H3-fl2va-pruned-comfy-fp8.safetensors      19.52 GB   8-bit
MiniMax-H3-fl2va-pruned-comfy-mxfp8.safetensors    20.08 GB   8-bit
```

## Check this before you download 20 GB

A file that fits can still run **emulated**, and nothing in the UI tells you
which you got. ComfyUI prints it at startup — look for the line beginning
`Native ops:` and check your format is in it.

Emulation is hardware, not a ComfyUI version problem. NVFP4 and microscaling
FP8 are Blackwell tensor-core features; `float8_e4m3fn` wants Ada or newer. An
Ampere card emulates all three no matter how current ComfyUI is.

* **Blackwell, 50-series** — `nvfp4`. Fastest thing here.
* **Ada, 40-series** — `w4a8`. Measured by a user on a 4090 at 17.4 s/it
  against 31.0 for fp8 on the same settings — because fp8 at 21 GB does not fit
  beside the activation pool on 24 GB and streams weights over PCIe. Native ops
  you cannot reach are worth nothing.
* **Ampere, 30-series** — `int8`. The 4-bit formats will load and run slowly.

## Which format, and an honest correction

**Corrected 2026-08-15.** This section used to say the formats were all the same
quality and that you should take the smallest file your card runs natively. A
user measured otherwise and they were right, so here is the accurate version.

What we tested: on the ref2va set, the same scene through `w4a8` and `int8` at
704x1280 and 20 steps, both reviewed as video. They showed **the same defect
classes** - garbled text on instrument dials, reflective safety-tape edges that
warp during movement, limb distortion on fast motion, stiff lip sync. That part
still holds, and it is worth knowing: those are characteristics of the base
model at these shapes, not quantisation damage. If you see them, check against a
GGUF or the unquantised model before blaming the quant.

Where that went wrong: "the same defect classes" is not "the same quality", and
the review was done by an automated video reviewer that is demonstrably blind to
fine texture differences. It was the wrong instrument for the question.

What a user measured by eye, on a 5060 Ti at 0.5 MP, 12 steps,
res_multistep/simple:

* `int8_convrot` (8-bit, ~19.5 GB) - good quality, 193s
* 4-bit, this repo - moderate quality, 187s
* 4-bit, a different maintainer's build - moderate quality, 187s

Two independent 4-bit builds landing on the same verdict is a real result, and
~3% on wall clock is not a trade worth making for it.

**So: if an 8-bit build fits your card natively, take it.** Reach for 4-bit when
8-bit does not fit, when it would stream weights over PCIe, or when you are at a
large enough workload that the format actually shows up in the wall clock - at
0.5 MP over 5 seconds it does not, because load, text encode and VAE dominate.

The one quality line that was always real: 4-bit **activations** genuinely
damage this model, which is why there is no `w4a4` here. 4-bit *weights* are
usable - just not free.

## How these were verified

Honest answer, and it differs from the ref2va set: **these were forward-gated,
not render-tested.** Every build was run through a GPU forward pass on both an
attention layer and a feed-forward layer before publication, and all five
passed. That catches a format that cannot execute; it does not catch one that
executes and looks wrong.

What that inherits from the ref2va set, which *was* rendered at matched seed and
shape:

* the speed ordering — nvfp4 fastest, then int8, mxfp8, w4a8, fp8, all of them
  between 18% and 34% faster than the Q8_0 GGUF
* the quality equivalence above
* both cuts (see below)

Same architecture, same 200-layer set, same quantiser, same pruned lineage — so
those carry. If you render a fl2va build and it disagrees with any of this, I
want to hear about it.

## Two formats that are NOT here, and why

* **`convrot_w4a4`** — renders without error and the output is visibly degraded.
  It quantises **activations** to 4 bits; every file here keeps activations at 8.
  Instrument text becomes scribble, hi-vis banding dissolves, hands smear.
  Caught by render on the ref2va set at seed 4040 against four other builds. It
  saves 1.2 GB against `w4a8` and costs all of it.
* **`float8_e5m2`** — loads cleanly, then throws at the first matmul:
  *"Multiplication of two Float8_e5m2 matrices is not supported"*. Same on
  LTX-2.5. It would ship as a file that appears fine until you sample.

Neither was built for fl2va. Both failures are properties of the format rather
than of these particular weights, so building 32 GB to quarantine it would have
been busywork.

## Why there are no K-quants of this model anywhere

H3's hidden width is **2688**, and K-quants need weight rows divisible by 256.
2688 is not, so the whole K-ladder is unavailable for the DiT — which is why the
GGUFs of it are Q4_0/Q5_1/Q8_0 and not Q4_K_M. (The *encoder* is a stock
Qwen3-VL and its rows do divide, so
[that repo](https://huggingface.co/joeygambino/MiniMax-H3-encoder-GGUF) has real
K-quants. The two are easy to confuse.)

The comfy-native formats have no such constraint, which is part of why this set
exists.

## How these were made

The layer SET is mirrored from Comfy-Org's own
`minimax_h3_fl2va_pruned_int8_convrot` rather than guessed: exactly **200
Linears**, being `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1` and `mlp.fc2`
across 50 blocks. Everything else stays bf16/fp16 — adaLN, the timestep
embedders, every norm and bias, `token_refiner`, the patch projections and the
final layer. Those steer every block and rounding them is how a quantised DiT
dies.

Built from the **pruned** bf16 lineage. That matters arithmetically: on the
unpruned lineage adaLN is 38% of the file, and 4-bit lands at 19.9 GB — which
the existing GGUF Q4_0 already achieves. Pruned 4-bit lands near 11.7 GB.


## What else you need

* **Text encoder and VAEs** — `Comfy-Org/MiniMax-H3`. Two separate VAEs, video
  and audio.
* Nothing else. That is the point of these.

## Honest limits

* No render A/B on the fl2va weights specifically. The speed and quality
  findings are inherited from the ref2va set — identical architecture, layer
  set, quantiser and lineage — not measured here.
* `fl2va` ignores `voice_ref`, `self_anchor_voice` and the identity bank
  entirely. If your workflow depends on those, you want `ref2va`.
* Two formats are absent on purpose: `convrot_w4a4` (4-bit activations
  visibly degrade H3) and `float8_e5m2` (loads, then throws at the first
  matmul — e5m2 cannot matmul in PyTorch). Both were cut on `ref2va` evidence
  that is not weight-specific, so neither was built here.

## License

Apache-2.0, following the base model.
