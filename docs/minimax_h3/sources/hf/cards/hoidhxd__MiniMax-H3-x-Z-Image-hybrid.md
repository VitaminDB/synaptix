# hoidhxd/MiniMax-H3-x-Z-Image-hybrid

Источник: https://huggingface.co/hoidhxd/MiniMax-H3-x-Z-Image-hybrid · скачано 2026-11-28
Загрузок: 15172 · лайков: 27 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги:  · обновлено: 2026-08-24T17:14:31.000Z

---

---
license: apache-2.0
base_model:
  - MiniMaxAI/MiniMax-H3
library_name: diffusers
tags:
- minimax
- minimax-h3
- z-image
- fl2va
- ref2va
- hybrid
- bf16
- int8
- video-generation
- image-to-video
- text-to-video
---

# MiniMax-H3 x Z-Image Hybrid

A custom hybrid version of **MiniMax-H3** combining the **FL2VA** and
**REF2VA** branches of the model.

The model uses different branches for different transformer blocks:

```text
B00-B24  -> FL2VA
B25-B49  -> REF2VA
B50+     -> FL2VA
````

This hybrid layout was constructed at the raw safetensor payload level.
No tensor conversion was performed when constructing the BF16 hybrid.

## Files

### BF16

```text
minimax_h3_hybrid_fl2va_ref2va_zs05_b25-49_bf16.safetensors
```

Approximately 37.46 GiB.

This is the full-precision BF16 hybrid model.

The B25-B49 tensors were copied byte-for-byte from the REF2VA ZS05 BF16
model, while the remaining tensors come from the FL2VA ZS05 BF16 model.

### INT8

The repository also contains an INT8 version of the same hybrid model.

The INT8 model follows the same conceptual hybrid layout:

```text
B00-B24  -> FL2VA
B25-B49  -> REF2VA
B50+     -> FL2VA
```

## Verification

The BF16 hybrid was verified at tensor level.

Results:

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

All B25-B49 payloads were verified to be byte-exact copies from the
REF2VA source.

No tensor conversion was performed during hybrid construction.

## Why this hybrid exists

The purpose of this model is to experiment with a mixed FL2VA/REF2VA
architecture where only the middle transformer blocks use REF2VA.

The selected layout is:

```text
FL2VA
  |
  +-- B00-B24
  |
REF2VA
  |
  +-- B25-B49
  |
FL2VA
  |
  +-- B50+
```

This allows the model to be tested as a single hybrid model without
requiring separate model files for the two branches.

## Related model

For the original/native Z-Image implementation and the REF2VA BF16
model, see:

[https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-native](https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-native)

Please refer to the original repository for the native model structure,
configuration and usage information.

## GGUF

A GGUF conversion of the BF16 hybrid is provided separately:

[https://huggingface.co/hoidhxd/MiniMax-H3-x-Z-Image-hybrid-GGUF](https://huggingface.co/hoidhxd/MiniMax-H3-x-Z-Image-hybrid-GGUF)

The GGUF version is intended for use with GGUF-compatible ComfyUI
workflows and loaders.

## Disclaimer

This is an experimental custom hybrid model.

It is not an official MiniMax release.

The hybrid block selection was created for experimentation and testing.

 
