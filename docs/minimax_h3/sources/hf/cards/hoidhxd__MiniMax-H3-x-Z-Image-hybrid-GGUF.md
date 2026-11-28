# hoidhxd/MiniMax-H3-x-Z-Image-hybrid-GGUF

Источник: https://huggingface.co/hoidhxd/MiniMax-H3-x-Z-Image-hybrid-GGUF · скачано 2026-11-28
Загрузок: 2700 · лайков: 5 · базовая: ["hoidhxd/MiniMax-H3-x-Z-Image-hybrid"] · теги: gguf,base_model:quantized:hoidhxd/MiniMax-H3-x-Z-Image-hybrid · обновлено: 2026-08-24T17:41:31.000Z

---

---
license: apache-2.0
base_model:
  - hoidhxd/MiniMax-H3-x-Z-Image-hybrid
library_name: gguf
tags:
- minimax
- minimax-h3
- z-image
- fl2va
- ref2va
- hybrid
- gguf
- q4_0
- video-generation
- image-to-video
---

# MiniMax-H3 x Z-Image Hybrid GGUF

GGUF conversion of the custom **MiniMax-H3 x Z-Image Hybrid** model.

The underlying hybrid architecture is:

```text
B00-B24  -> FL2VA
B25-B49  -> REF2VA
B50+     -> FL2VA
````

This model combines the FL2VA and REF2VA branches into a single hybrid
model.

## Files

### BF16 GGUF

```text
minimax_h3_hybrid_fl2va_ref2va_zs05_b25-49_BF16.gguf
```

This file was converted from:

```text
minimax_h3_hybrid_fl2va_ref2va_zs05_b25-49_bf16.safetensors
```

using the `molbal/ComfyUI-GGUF` conversion tools.

The BF16 GGUF is provided as the source GGUF representation before
additional quantization.

## Quantization

Additional quantized GGUF versions may be added later.

For example:

```text
Q4_0
Q5_0
Q8_0
```

will be added as separate files when available and verified.

## Original Hybrid Model

The original BF16/INT8 safetensor versions are available here:

[https://huggingface.co/hoidhxd/MiniMax-H3-x-Z-Image-hybrid](https://huggingface.co/hoidhxd/MiniMax-H3-x-Z-Image-hybrid)

## Native / Reference Repository

For the original/native MiniMax-H3 x Z-Image model and REF2VA files,
see:

[https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-native](https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-native)

## ComfyUI

This GGUF model is intended for use with:

[https://github.com/molbal/ComfyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF)

The conversion used the `molbal/ComfyUI-GGUF` implementation.

Make sure your ComfyUI-GGUF version supports the MiniMax-H3 architecture.

## Hybrid Layout

The model is intentionally constructed as:

```text
                    MiniMax-H3 Hybrid

        ┌───────────────────────────────────┐
        │ B00 - B24                         │
        │ FL2VA                             │
        ├───────────────────────────────────┤
        │ B25 - B49                         │
        │ REF2VA                            │
        ├───────────────────────────────────┤
        │ B50+                              │
        │ FL2VA                             │
        └───────────────────────────────────┘
```

## Verification

The original BF16 safetensor hybrid was verified at tensor level:

```text
Tensor count                 : 532
Expected source matches      : 532/532
Payload failures             : 0
Wrong-source tensors         : 0

B00-B24 : FL2VA
B25-B49 : REF2VA
B50+    : FL2VA

Tensor set       : PASS
dtype            : PASS
shape            : PASS
payload sizes    : PASS
Payload hashes   : PASS
```

The hybrid construction itself was performed by byte-level tensor
replacement. No BF16 numerical conversion was performed.

## Important

This is an experimental community-created hybrid.

It is **not an official MiniMax release**.

Use the native repository for the original model and official model
structure.
