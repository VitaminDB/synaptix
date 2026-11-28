# joeygambino/MiniMax-H3-encoder-GGUF

Источник: https://huggingface.co/joeygambino/MiniMax-H3-encoder-GGUF · скачано 2026-11-28
Загрузок: 52129 · лайков: 15 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: gguf,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-07T11:07:22.000Z

---

---
library_name: gguf
tags:
  - gguf
  - text-encoder
  - minimax
  - minimax-h3
  - qwen3-vl
  - comfyui
  - video
  - text-to-video
  - image-to-video
base_model:
  - MiniMaxAI/MiniMax-H3
---

# MiniMax-H3 Text Encoder — GGUF

GGUF quantizations of the **Qwen3-VL-32B vision-language text encoder** used by
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) in ComfyUI.

The H3 DiT quants are here: **[joeygambino/MiniMax-H3-GGUF](https://huggingface.co/joeygambino/MiniMax-H3-GGUF)**.
You need **one file from each repo** to run H3 — the DiT alone will not generate anything.

---

## Troubleshooting: state_dict / vision mismatch with the `-mmproj` file

Load these encoders with **H3 Clip Loader (Any)** from
[ComfyUI-H3-Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot),
**not** the stock `CLIPLoaderGGUF` node.

The H3 text encoder is a *truncated* Qwen3-VL-32B - 50 layers, no final norm,
no `lm_head` - and its vision tower ships separately as the `-mmproj-F16.gguf`
sidecar. Stock ComfyUI-GGUF only merges an mmproj when the encoder's
architecture is `qwen2vl`; Qwen3-VL reports `qwen3vl`, so the sidecar is never
merged at all, and the resulting missing vision tensors surface as a state_dict
mismatch. Its mmproj key map is also qwen2vl-era: wrong merger keys, and no
rules for H3's deepstack mergers or split QKV, so even forcing the merge would
not line up.

The pack's loader does all three things stock cannot - truncates the text tower
to H3's shape, merges the sidecar explicitly, and renames the vision tensors to
H3's layout (including remapping deepstack mergers, which llama.cpp indexes by
vision layer 8/16/24 and H3 indexes by position 0/1/2).

Keep the `-mmproj-F16.gguf` in the **same folder** as the encoder and do not
rename either file: they are paired by filename stem.

**Why does a `.safetensors` encoder work without any of this?** Because a full
safetensors encoder (fp8, int8, NVFP4-AWQ, ...) is a complete, pre-shaped model
with its vision tower already inside. It needs no sidecar and no remapping.
That is a property of the container, not of the quantization - NVFP4 is not
doing anything special here.

## Files

| File | Size | Use |
|---|---|---|
| `MiniMax-H3-encoder-Q4_K_M.gguf` | 19.8 GB | **Recommended.** ~16.5 GB resident. |
| `MiniMax-H3-encoder-Q5_K_M.gguf` | 23.2 GB | Higher precision, more VRAM. |
| `MiniMax-H3-encoder-mmproj-F16.gguf` | 1.2 GB | **Multimodal projector — REQUIRED for image input.** |

### The mmproj is not optional if you use reference images

H3's reference-to-video (`ref2va`) and image-to-video paths feed **images** into the
text encoder. Without `mmproj`, the encoder is text-only: reference images are
ignored and reference/I2V workflows will not behave correctly.

`CLIPLoaderGGUF` (from [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF))
**auto-pairs the mmproj sidecar by filename**, so keep the names exactly as
downloaded and put the mmproj in the same folder as the encoder.

---

## Install

Place all files in:

```
ComfyUI/models/text_encoders/
├── MiniMax-H3-encoder-Q4_K_M.gguf
└── MiniMax-H3-encoder-mmproj-F16.gguf
```

Load with **`CLIPLoaderGGUF`** (ComfyUI-GGUF) — or `H3ClipLoaderAny` from
[ComfyUI-H3-Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot),
which accepts either `.safetensors` or `.gguf` and handles the mmproj pairing.

Set the CLIP **type** to `minimax`.

---

## VRAM: read this before you file a "it's so slow" issue

The encoder and the DiT **do not co-fit on a 32 GB card**:

| Component | Resident |
|---|---|
| Encoder (Q4_K_M) | ~16.5 GB |
| H3 DiT | ~25 GB |

If both are held at once, the DiT loads **partially** and streams the remainder from
system RAM every sampling step. Measured on an RTX 5090: **~60 minutes** for a clip
that takes **~15 minutes** when the DiT is fully resident. The tell in your log is:

```
loaded partially; 6423 MB usable, 5847 MB loaded, 19363 MB offloaded
```

**Fix:** evict the encoder after conditioning and before sampling. Conditioning is
already computed at that point, so the encoder weights are safe to drop:

```python
import comfy.model_management as mm
clip.patcher.model.to(mm.text_encoder_offload_device())
mm.free_memory(mm.get_total_memory(mm.get_torch_device()) * 0.9, mm.get_torch_device())
mm.soft_empty_cache()
```

The multishot sampler in
[ComfyUI-H3-Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot)
does this automatically between shots.

Note that a chained multi-shot workflow re-encodes per shot (each shot conditions on
the previous shot's last frame), so the encoder reloads each time — a few seconds per
shot, which is far cheaper than streaming ~19 GB every step.

---

## Notes on quantization

The encoder is **Qwen3-VL-32B**, whose tensor dimensions permit K-quants — hence
`Q4_K_M` / `Q5_K_M` here. The H3 **DiT** cannot use K-quants (its 2688-wide tensors
are not divisible by 256), which is why that repo ships `Q4_0` / `Q5_1` instead. If
you are mixing and matching, that difference is expected, not a packaging error.

---

## Also on Civitai

Same files, if you prefer downloading there: [MiniMax-H3 Text Encoder GGUF (Qwen3-VL)](https://civitai.com/models/2834385)

## Support

Everything here is free and stays free — the format spec, the nodes, the workflows,
the cartridges, the LoRAs. If it saved you a night of debugging (it contains several
hundred of mine), tips keep the 5090 warm:

* ☕ [Buy me a coffee on Ko-fi](https://ko-fi.com/joeygambino)
* 💛 [Sponsor on GitHub](https://github.com/sponsors/jlucasmcrell)
* 🔁 [Liberapay](https://liberapay.com/joeygambino) (recurring)

---

## Credits

- **MiniMax-H3** — [MiniMaxAI](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- **Qwen3-VL** — Alibaba / Qwen team
- **GGUF loader for ComfyUI** — [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)

Quantized and packaged by [joeygambino](https://huggingface.co/joeygambino).
