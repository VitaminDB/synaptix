# woodfireind/MiniMax-H3-GGUF-MiniStack

Источник: https://huggingface.co/woodfireind/MiniMax-H3-GGUF-MiniStack · скачано 2026-11-28
Загрузок: 3603 · лайков: 5 · базовая: MiniMaxAI/MiniMax-H3 · теги: gguf,comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-07T10:57:06.000Z

---

---
language:
- en
library_name: diffusers
pipeline_tag: text-to-video
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3
tags:
- comfyui
- gguf
- video-generation
- minimax
- text-to-video
---

# MiniMax H3 GGUF MiniStack

A shrunk, GGUF-quantized MiniMax H3 **FL2VA** stack built for ComfyUI on consumer GPUs (developed on Intel Arc A770 16 GB). The full 33B stack has been reduced via AdaLN pruning of the DiT and distillation of the 32B text encoder down to a 4B student, then quantized to Q4_K_M. The audio VAE is **dropped** (see [Limitations](#limitations)).

**Provenance:** this is a derived, quantized repack of [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) as repackaged by [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3), plus locally trained adapter weights. It is released under the **MiniMax H3 Community License Agreement**.

## What's in the box

```
📂 ComfyUI/models/
├── 📂 diffusion_models/
│   └── MiniMax-H3-FL2VA-pruned-Q4_K_M.gguf       11.38 GB  pruned AdaLN DiT, Q4_K_M
├── 📂 text_encoders/
│   ├── qwen3vl-4b-h3student-Q4_K_M.gguf           2.50 GB  distilled TE, Q4_K_M (default)
│   ├── qwen3vl-4b-h3student-BF16.gguf             8.05 GB  distilled TE, BF16 (optional)
│   ├── te_adapter_v1.safetensors                    120 MB  student→H3 5120-d MLP adapter (required)
│   └── h3_tokenizer/                                  11 MB  H3 tokenizer (required)
└── 📂 vae/
    └── minimax_h3_video_vae_fp16.safetensors       4.85 GB  stock video VAE, fp16
```

Total download ≈ **27 GB** (Q4 default ≈ 19 GB if you skip the BF16 TE).

| File | SHA-256 |
|---|---|
| `diffusion_models/MiniMax-H3-FL2VA-pruned-Q4_K_M.gguf` | `51783047a6be43c9fcedc3c0a1746480dac867839ba1ad2e1c698da9d7058201` |
| `text_encoders/qwen3vl-4b-h3student-Q4_K_M.gguf` | `7ff66b1541723fc4c4c3f2cba75f72740fad43276ab1d6d564379e591a2e74e5` |
| `text_encoders/qwen3vl-4b-h3student-BF16.gguf` | `b6c55a51fe20aca2700fcedd228ad9af9b67f896a10ae5ff80092752c09145ba` |
| `text_encoders/te_adapter_v1.safetensors` | `ef2060e1a277820416d895812dd580234d3c2760af8e6d203b38e3df2e0952b0` |
| `text_encoders/h3_tokenizer/tokenizer.json` | `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7` |
| `text_encoders/h3_tokenizer/tokenizer_config.json` | `a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68` |
| `text_encoders/h3_tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `text_encoders/h3_tokenizer/merges.txt` | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `vae/minimax_h3_video_vae_fp16.safetensors` | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |

## Documented changes vs. stock

1. **Pruned DiT (FL2VA).** The 33B full-width DiT is replaced with the AdaLN-curve form from `Comfy-Org/MiniMax-H3` (`minimax_h3_fl2va_pruned_bf16`): AdaLN 13B→44M params via the `adaln_t_table` [1025,8] + per-block [96768,8] projections. Effective ≈ **18.7B params**.
2. **Q4_K_M quantization.** Converted with a two-pass streaming converter and quantized with a ComfyUI-GGUF-patched llama.cpp build (arch `"wan"`; stock `llama-quantize` rejects this arch). **11.38 GB** vs 14.5 GiB for the full Q3_K_M. GGUF: `general.architecture=wan`, 532 tensors, 53 fp32/fp16 hiprec islands (adaln*, t_embed, norm, patch_proj) kept as-is.
3. **Distilled student text encoder.** The 32B Qwen3-VL-32B teacher is replaced with a **Qwen3-VL-4B** student plus a trained MLP adapter (`2560→4096→5120`, fp32) that maps student language hidden states into H3's 5120-d multimodal token slots. Distilled from 2,500 teacher dumps; **val_cos 0.9310** (full-corpus v4 adapter). Q4_K_M GGUF is numerically near-identical to the BF16 student (**>0.99 cosine**); the gap to the teacher is adapter fidelity, not quant noise.
4. **Audio VAE dropped.** The `minimax_h3_audio_vae_fp32` component is removed from the workflow. Output videos are **silent**. The stock `minimax_h3_video_vae_fp16` is unchanged and included.
5. **Video VAE unchanged.** Stock fp16 video VAE; no pruning or requantization was applied.

## Requirements

- **ComfyUI** with MiniMax H3 support (`Comfy-Org/ComfyUI` PR #15224 / current main) and the **ComfyUI-GGUF** custom node.
- For the student TE you need the `h3_small_te` custom node (H3SmallTELoader / H3SmallTextEncoder) — it loads the GGUF via ComfyUI-GGUF's `gguf_clip_loader` into comfy-native `Llama2_(Qwen3VL_4BConfig)` with GGMLOps, keeping the H3 HF tokenizer + Identity final-norm + the adapter. The adapter and `h3_tokenizer/` are **required** — without them the student cannot encode.
- DiT and video VAE are loaded with the standard `UnetLoaderGGUF` / `VAELoader` nodes.

### Recommended settings (640×480 previz, Arc A770 16 GB)

- Sampling: 8 steps with the turbo LoRA (`larryvrh/MiniMax-H3-Turbo-Lora`), `--reserve-vram 3`. Note the turbo LoRA's `adaln_proj` targets dangle on the *pruned* checkpoint — apply it to the full checkpoint if you need those.
- Length: snap to `5 mod 17` frames (17k+5 grid). Example: `length = max(5, round(seconds*24)) + (5 - (max(5, round(seconds*24)) % 17)) % 17`.
- Default: 640×480 / 124 frames / 24 fps / 8 steps.

## Limitations

- **No audio.** The audio VAE is dropped from this stack — generated videos are silent and there is no voice/SFX/music output.
- **Student TE is text-only.** The 4B student adapter was distilled on pure-text teacher dumps; it has **no vision path**. For I2V / FL2V keyframe identity (which needs the teacher's vision tower + deepstack features), use the teacher 32B TE (`qwen3vl-32B-MiniMax-H3-*`); the student cannot consume image tokens.
- **Adapter fidelity ceiling.** val_cos ≈ 0.93 vs the teacher. Shot structure and cuts are preserved, but **cut timing can drift ±2 s** and fine prompt specifics (small props, signage details) can wash out. Timestamp-magnitude tuning (v3/v4) plateaued; this is accepted as a known tradeoff.
- **Turbo LoRA compatibility.** The preview turbo LoRA targets full-width `adaln_proj` weights that do not exist in the pruned curve-form checkpoint; using both together silently skips those targets.
- **GGUF arch.** The DiT is archived as `"wan"` for quantizer compatibility; use a ComfyUI-GGUF build that can load arch-`wan` GGUFs. The student GGUF is arch `qwen3vl`.
- **Frames-vs-seconds prompting.** H3 is trained on `mm:ss.mmm` timestamps; the 4× temporal VAE grid only quantizes where cuts can land — frame-based timestamps are off-distribution.
- **Not for 2K.** Local gen is 640×480 previz; 768P/2K regeneration requires the official MiniMax cloud API.

## Usage

1. Unzip files into the ComfyUI `models/` folders as shown above.
2. For **T2V** (text only): `UnetLoaderGGUF` → pruned Q4 GGUF; `H3SmallTELoader` → student Q4 GGUF + adapter + tokenizer dir; `VAELoader` → video VAE.
3. For **I2V / FL2V**: swap the student TE for the teacher 32B TE (vision path); everything else stays.
4. Encode with `MiniMaxH3ImageToVideo`, sample with `SamplerCustomAdvanced` (res_multistep / simple, 25 steps, or 8 with turbo LoRA).

## Eval notes (summary)

- Baseline (full Q3_K_M, 25 steps): 620 s wall @ 640×480/124f.
- Pruned Q4_K_M (25 steps): 610 s wall, output on-par with baseline (same composition/cut).
- Turbo 4/8-step: ~160 s / ~250 s wall; 8-step recommended.
- FL2V smoke (pruned Q4 + teacher TE): 295.5 s wall; keyframe lock first/last pixel-cos 0.9905/0.9987.
- Student vs teacher numeric A/B: mean cosine 0.847 (mid adapter) → **0.9310** (full-corpus v4). Video A/B: structure and cuts preserved; fine details degrade consistent with ~0.93 cos.

## License

Derived from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — **MiniMax H3 Community License Agreement** (see link in metadata header). The student adapter is a training-derived artifact of distillation against the teacher model and is distributed under the same agreement. The video VAE is the unmodified Comfy-Org repack.
