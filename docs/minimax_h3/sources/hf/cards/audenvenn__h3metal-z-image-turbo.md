# audenvenn/h3metal-z-image-turbo

Источник: https://huggingface.co/audenvenn/h3metal-z-image-turbo · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая: Tongyi-MAI/Z-Image-Turbo · теги: z-image-turbo,base_model:Tongyi-MAI/Z-Image-Turbo,base_model:quantized:Tongyi-MAI/Z-Image-Turbo · обновлено: 2026-11-13T06:32:48.000Z

---

---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
base_model_relation: quantized
pipeline_tag: text-to-image
library_name: h3metal
tags:
- apple-silicon
- metal
- mac
- int8
- z-image
- z-image-turbo
---

# Z-Image-Turbo for h3metal (int8)

[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Tongyi-MAI, converted for
[h3metal](https://github.com/audenvenn/h3metal), an open-source (Apache-2.0) inference engine for Apple
silicon that runs the model on the matrix units of M5- and M6-class GPUs. These files are h3metal's own
format: they do not load in diffusers or other libraries.

## Use

```sh
pip install https://github.com/audenvenn/h3metal/releases/download/v0.5.1/h3metal-0.5.1-py3-none-macosx_26_0_arm64.whl
h3metal download --model z-image-turbo      # these files, 10.7 GB, into ~/.h3metal/zimage
h3metal generate --model zimage -p "A red fox sitting in fresh snow at sunrise" -o fox.png
```

It needs h3metal 0.5.1 or later, a Mac with an M5- or M6-class GPU, macOS 26 and Python 3.10 or later. At 1024 x 1024 h3metal
uses about 8.4 GB of memory: the transformer's 6.1 GB file, which it maps instead of copying, and 2.2 GB
besides.

LoRA adapters for Z-Image-Turbo work with these files (`--lora FILE[:STRENGTH]`), including the AdaLN
modulation layers most of them adapt; h3metal applies them exactly, as low-rank terms inside its int8 products
([docs/lora.md](https://github.com/audenvenn/h3metal/blob/main/docs/lora.md)).

## Speed and quality

On a Mac mini (M6, 24 GB), a 1024 x 1024 image takes 17.4 s from a new prompt (eight steps, the whole run
from launch to PNG), 2.1 times faster than mflux 0.24 with its own 8-bit weights on the same machine, and a
2048 x 2048 image 97.1 s (2.2 times). With the same prompt and noise as diffusers' `ZImagePipeline` in
bfloat16, the image is within 31.3 dB PSNR of it (SSIM 0.974). The method and the part-by-part comparison
are in h3metal's [docs/benchmarks.md](https://github.com/audenvenn/h3metal/blob/main/docs/benchmarks.md)
and [docs/zimage.md](https://github.com/audenvenn/h3metal/blob/main/docs/zimage.md).

## Files

| File | Contents | Size |
|---|---|---:|
| `dit.h3m` | the transformer: int8 weights, activations quantized to int8 per row after a Hadamard rotation; the modulation's inputs for LoRA adapters | 6.1 GB |
| `te.h3m` | the text encoder (Qwen3-4B, the 35 layers the pipeline reads), int8 | 4.3 GB |
| `vae.h3m` | the VAE decoder, float32 | 0.2 GB |
| `tokenizer.json` | the original tokenizer | 11 MB |

They are what `h3metal convert` makes from the original checkpoint with its default settings; converting
yourself (`h3metal download --model z-image-turbo --original`, then `h3metal convert`) gives the same
bytes. The git tag `h3m-1` marks this layout; each h3metal version downloads the tag it reads.

## License

Apache-2.0, like the original weights; NOTICE lists the changes. Z-Image-Turbo is by Tongyi-MAI. Images
the model makes are AI-generated, and h3metal says so in every PNG it writes.
