# hoidhxd/MiniMax-H3-Hybrid-b25-49-GGUF

Источник: https://huggingface.co/hoidhxd/MiniMax-H3-Hybrid-b25-49-GGUF · скачано 2026-11-28
Загрузок: 6021 · лайков: 5 · базовая: ["smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models"] · теги: gguf,comfyui,quantized,base_model:quantized:smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models · обновлено: 2026-08-13T13:19:47.000Z

---

---
base_model:
  - smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models
base_model_relation: quantized
library_name: gguf
tags:
  - gguf
  - minimax-h3
  - minimax
  - video-generation
  - image-to-video
  - text-to-video
  - audio-to-video
  - reference-to-video
  - fl2va
  - ref2va
  - comfyui
  - quantized
---

# MiniMax H3 Hybrid FL2VA + REF2VA — GGUF (Q4_0 & Q5_0)

Community GGUF Q4_0 and Q5_0 quantizations of the `b25-49` hybrid MiniMax H3 checkpoint.

These GGUFs are derived from:

**`smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models`**

Specifically:

**`minimax_h3_hybrid_fl2va_ref2va_b25-49.safetensors`**

The original hybrid checkpoint combines the FL2VA and REF2VA MiniMax H3 variants, using FL2VA as the base model while using REF2VA `adaln_proj` weights for transformer blocks 25–49.

This repository provides GGUF Q4_0 and Q5_0 versions intended for use with ComfyUI and other GGUF-compatible runtimes.

---

## Model Information

| Property | Value |
|---|---|
| Architecture | `minimax_h3` |
| Variant | `b25-49` |
| Quantization | `Q4_0` / `Q5_0` |
| GGUF version | `3` |
| Tensor count | `532` |
| Tensor elements | `20,111,438,744` |
| GGUF file size (Q4_0) | `10.6 GB` (`11,387,224,800 bytes`) |
| GGUF file size (Q5_0) | `12.9 GB` (`13,895,623,392 bytes`) |
| Format | GGUF |
| Primary runtime | ComfyUI + ComfyUI-GGUF |

---

## Source Model

Original hybrid model:

**`smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models`**

https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models

The original repository contains multiple FL2VA + REF2VA hybrid variants.

These GGUFs were created from the:

**`b25-49`**

variant.

---

## Quantization

The source checkpoint was converted to GGUF and quantized to:

**Q4_0** and **Q5_0**

using the tools from:

**`molbal/ComfyUI-GGUF`**

https://github.com/molbal/ComfyUI-GGUF

The resulting GGUFs contain:

- 532 tensors
- 20,111,438,744 tensor elements
- 213 quantized tensors (Q4_0 or Q5_0 depending on the file)
- 319 tensors retained in higher precision formats (F16/F32)
- GGUF architecture: `minimax_h3`
- GGUF version: 3

The large transformer weight matrices are quantized to Q4_0/Q5_0.

Smaller and/or sensitive tensors, including normalization, bias and modulation-related tensors, remain in F16/F32 to preserve quality.

These are quantizations of the original checkpoint. No additional training or fine-tuning was performed during the conversion.

---

## GGUF Files

Choose the file that best fits your VRAM and quality requirements:

**Q4_0 Version** (Smaller size, lower VRAM usage)
```text
minimax_h3_hybrid_fl2va_ref2va_b25-49-Q4_0.gguf
Size: 10.6 GB (11,387,224,800 bytes)

```

**Q5_0 Version** (Better precision, slightly larger file size)

```text
minimax_h3_hybrid_fl2va_ref2va_b25-49-Q5_0.gguf
Size: 12.9 GB (13,895,623,392 bytes)

```

---

## ComfyUI Compatibility

These GGUFs were created for use with:

**`molbal/ComfyUI-GGUF`**

[https://github.com/molbal/ComfyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF)

The models have been loaded in ComfyUI using the GGUF workflow.

---

## FL2VA Testing

**Q4_0: Tested successfully.**
**Q5_0: Untested.**

The **Q4_0** GGUF was loaded in ComfyUI and successfully generated video using an FL2VA workflow.
The **Q5_0** version is provided for users with higher VRAM capacity but remains untested in this environment.

---

## REF2VA Testing

**Q4_0: Tested successfully.**
**Q5_0: Untested.**

The **Q4_0** GGUF was loaded in ComfyUI and successfully generated video using a REF2VA reference-conditioning workflow. The REF2VA workflow completed successfully to 100% and produced a video output.

---

## Installation

Install `ComfyUI-GGUF`:

[https://github.com/molbal/ComfyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF)

Follow the installation instructions provided by the repository.

Place the downloaded GGUF model in the appropriate ComfyUI model directory used by your GGUF workflow.

For example:

```text
ComfyUI/
└── models/
    └── unet/
        ├── minimax_h3_hybrid_fl2va_ref2va_b25-49-Q4_0.gguf
        └── minimax_h3_hybrid_fl2va_ref2va_b25-49-Q5_0.gguf

```

The exact location may depend on the ComfyUI workflow and loader being used.

---

## Hardware

Actual VRAM requirements depend on the workflow and generation settings, including:

* resolution
* number of frames
* reference inputs
* text encoder
* VAE
* GPU offloading
* ComfyUI configuration
* CUDA/PyTorch configuration

The GGUF Q4_0/Q5_0 formats substantially reduce the model weight size compared with the original checkpoint.

**Testing Environment:**
The **Q4_0** version has been tested on a system equipped with:

* NVIDIA RTX 5060 Ti 16 GB

> ⚠️ **Important Note on Q5_0:**
> The **Q5_0** version has **not been tested** on this hardware. Due to its larger file size (12.9 GB), it may exceed the 16 GB VRAM limit during generation depending on your workflow settings, potentially resulting in Out Of Memory (OOM) errors.

Performance and VRAM requirements may vary significantly depending on the workflow.

---

## Original Model and Credits

### MiniMax

Original MiniMax H3 model and architecture.

### smhfacct

Creator of the FL2VA + REF2VA hybrid MiniMax H3 checkpoints.

Source:

[https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models)

### molbal

Creator of the ComfyUI-GGUF implementation used for GGUF support and conversion.

Repository:

[https://github.com/molbal/ComfyUI-GGUF](https://github.com/molbal/ComfyUI-GGUF)

---

## Disclaimer

This is a community-created GGUF quantization.

It is **not an official MiniMax release**.

The underlying weights originate from:

`smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models`

No additional training or fine-tuning was performed during the GGUF conversion.

Please refer to the original model and MiniMax licensing terms before using or redistributing this model.

---

## SHA256

```text
Q4_0: 6b14de64d26a3e67316c31b11198eb01bee03cabddbd16cb6f55b65c7b70943c
Q5_0: 21db216adde8836164ec75211f874c45e3d4df36b6a052e1e0b703dff6554aef

```

---

## Status

| Component | Status |
| --- | --- |
| Source checkpoint | ✅ `b25-49` |
| GGUF conversion | ✅ Complete |
| Q4_0 & Q5_0 quantization | ✅ Complete |
| GGUF header verification | ✅ Passed |
| Tensor inspection | ✅ Passed |
| FL2VA | ✅ Q4_0 Tested / ⚠️ Q5_0 Untested |
| REF2VA | ✅ Q4_0 Tested / ⚠️ Q5_0 Untested |
| Video output | ✅ Confirmed (Q4_0) |
