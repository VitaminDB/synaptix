# MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF

Источник: https://huggingface.co/MarxistLeninist/MiniMax-H3-FL2VA-Pruned-IQ1-GGUF · скачано 2026-11-28
Загрузок: 7695 · лайков: 0 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: gguf,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-10T14:27:21.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/6818f6c32d12b210915e44ad56a4228c2608f160/LICENSE
base_model:
- MiniMaxAI/MiniMax-H3
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-video
tags:
- stable-diffusion.cpp
- minimax-h3
- gguf
- iq1
- mixed-precision
- audio-video
---

# MiniMax H3 FL2VA Pruned — importance-guided mixed-IQ1 QF2 GGUF

Experimental mixed-precision GGUF quantizations of the approximately 20B-parameter pruned MiniMax H3 FL2VA diffusion denoiser. This is not the complete H3 system; its Qwen3-VL text encoder and video/audio VAEs are separate.

The repository owner confirms separate written MiniMax authorization covering this publication. That authorization is not sublicensed here; downstream users remain responsible for the official license and location-specific authorization.

## Artifact and quality status

The original uniform IQ1_S and IQ1_M files are **deprecated: perceptual FAIL**. They load and produce valid A/V containers, but a controlled 22-frame test produced no recognizable fox or snow.

QF2 is the final importance-guided mixed-precision policy. Earlier QF1 working files are superseded internal evidence, not final artifacts.

| File | Bytes | SHA-256 | Validated scope |
|---|---:|---|---|
| `minimax_h3_fl2va_pruned-UD-IQ1_M-QF.gguf` | 10,985,133,600 | `c7993a8a0eb202f63f30cb410817fc0a504f3c844ea4e8dbb38599d29b0d4bdc` | **Four-step website PASS**, including motorcycle holdout; eight-step visual PASS/audio FAIL |
| `minimax_h3_fl2va_pruned-UD-IQ1_S-QF.gguf` | 10,876,753,440 | `b12244f203e257f10c9cec56b8062aa65d627d9250c433ef633a244d610b292c` | **Four-step website PASS**, including motorcycle holdout; eight-step visual PASS/audio FAIL |
| `minimax_h3_fl2va_pruned-IQ1_S.gguf` | 4,062,866,976 | `76f1d0bcc9052978489814004b0aaefd1a60fc8b7b8223029749085915b2946f` | **DEPRECATED — perceptual FAIL** |
| `minimax_h3_fl2va_pruned-IQ1_M.gguf` | 4,532,514,336 | `b356d321476e7ceaf7848d0d0db7f7a82403f212313f0b7e69d1cd26c6799746` | **DEPRECATED — perceptual FAIL** |

The website four-step profile and eight-step full profile have separate verdicts. Do not describe either QF2 file as having passed full eight-step audiovisual generation.

## What QF2 means

The refreshed activation importance matrix showed a steep rise through later `fc1` blocks. QF2 therefore uses:

- BF16: `condition_proj.weight` and protected one-dimensional norm/gain tensors
- Q8_0: both token-refiner blocks and every eligible matrix in main blocks 46–49
- Q4_K: `mlp.fc1` in blocks 30–45, plus attention `qkv_proj`/`out_proj` and `mlp.fc2` in all non-Q8 main blocks
- IQ1_S or IQ1_M: only `mlp.fc1` in blocks 0–29
- inherited Q8_0: audio/video patch-projection weights whose shapes cannot be converted to IQ1/Q4_K
- source precision: biases, final/output layers, and other shapes excluded by converter safeguards

These are mixed-precision derivatives. IQ1 is the lowest precision and covers 4,624,220,160 parameters; the files are not uniform one-bit models.

## Controlled quick audit

Same fox prompt, seed 11, CPU RNG, `320x192`, 22 frames, 24 FPS, 4 steps, CFG 1.0:

| Denoiser | PSNR vs Q8 | SSIM vs Q8 | Luma SD | Laplacian | Phase-grid 16 | Adjacent MAD | Audio clipped | Human gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q8_0 control | reference | reference | 41.371961 | 4.104783 | 0.0180393 | 3.101225 | 0.00% | PASS |
| Original IQ1_S | 11.680717 dB | 0.502004 | 25.35 | 26.55 | 0.180 | not recorded | 4.50% | **FAIL: no fox or snow** |
| Original IQ1_M | 15.176109 dB | 0.448978 | 20.56 | 54.10 | 0.403 | not recorded | 3.60% | **FAIL: no fox or snow** |
| QF2 IQ1_M | 24.800133 dB | 0.876201 | 45.441886 | 3.198496 | 0.01557654 | 2.260789 | 0.00841864% | PASS: coherent fox, no grid |
| QF2 IQ1_S | 27.801705 dB | 0.895095 | 44.8127344 | 3.1386273 | 0.0148840364 | 1.78388595 | 0.00336746% | Low-resolution shape smeared, but no grid |

PSNR and SSIM compare decoded frames against Q8 at identical settings. LPIPS was not collected and is not claimed.

## Website and holdout evidence

- QF2 IQ1_M fox, 640x384/39 frames/4 steps: clear fox; audio RMS -15.6642 dBFS; clipped fraction 0; MP4 SHA-256 `e42ee46827018f86e86b67238e1806a6e9d20500a90e5e03710d3cee4fa02253`.
- QF2 IQ1_S fox, same website profile: clear walking fox; MAD 4.089718; audio RMS -16.70895 dBFS; clipped fraction 0; MP4 SHA-256 `23214d5008dc58edc036d0f89b9e4a1b442be5d4265e2218d03c4416095db24a`.
- QF2 IQ1_M motorcycle holdout, seed 123: human PASS; PSNR 22.117554 dB; SSIM 0.885048; MP4 SHA-256 `0b28a8c0a19ef8090fdc580a2c0f672ed1a341128ab0d2957f56b104a680c099`.
- QF2 IQ1_M motorcycle audio versus Q8: RMS -4.226/-4.380 dBFS, clipped 2.853%/2.530%, spectral correlation 0.8285/0.8136. Clipping occurs in the Q8 baseline too and is disclosed, not treated as clipping-free output.
- QF2 IQ1_S motorcycle holdout, seed 123: human PASS with a stable red motorcycle and rider on a coastal road at sunset; PSNR 21.096775 dB; SSIM 0.862191; MP4 SHA-256 `93ddcf76e9d98c78b0a51dd059a43d6bc478f451f06176ef8cda00a5a0a144f2`.
- QF2 IQ1_S motorcycle audio versus Q8: RMS -4.43595/-4.37951 dBFS; clipped 2.0843%/2.5304%; spectral correlation 0.84383/0.83414; time correlation 0.6931/0.7082. The prompt is loud and clipped in the Q8 baseline too; QF2 IQ1_S does not add excess clipping in this test.

## Review samples

| Profile | IQ1_M | IQ1_S |
|---|---|---|
| Quick diagnostic, 4 steps | [MP4](assets/qf2/iq1-m-fox-quick.mp4) · [contact sheet](assets/qf2/iq1-m-fox-quick-contact.png) | [MP4](assets/qf2/iq1-s-fox-quick.mp4) · [contact sheet](assets/qf2/iq1-s-fox-quick-contact.png) |
| Website fox, 4 steps | [MP4](assets/qf2/iq1-m-fox-site.mp4) · [contact sheet](assets/qf2/iq1-m-fox-site-contact.png) | [MP4](assets/qf2/iq1-s-fox-site.mp4) · [contact sheet](assets/qf2/iq1-s-fox-site-contact.png) |
| Independent motorcycle holdout, 4 steps | [MP4](assets/qf2/iq1-m-motorcycle-holdout-site.mp4) · [contact sheet](assets/qf2/iq1-m-motorcycle-holdout-site-contact.png) | [MP4](assets/qf2/iq1-s-motorcycle-holdout-site.mp4) · [contact sheet](assets/qf2/iq1-s-motorcycle-holdout-site-contact.png) |
| Full fox, 8 steps; visual PASS/audio FAIL | [MP4](assets/qf2/iq1-m-fox-full.mp4) · [contact sheet](assets/qf2/iq1-m-fox-full-contact.png) | [MP4](assets/qf2/iq1-s-fox-full.mp4) · [contact sheet](assets/qf2/iq1-s-fox-full-contact.png) |

Independent Q8 holdout control: [MP4](assets/qf2/q8-motorcycle-holdout-site.mp4) · [contact sheet](assets/qf2/q8-motorcycle-holdout-site-contact.png).

## Eight-step limitation

Both QF2 variants produce excellent articulated walking-fox video at 640x384/39 frames/8 steps, but both fail the full-profile audio comparison:

- IQ1_M: audio RMS -70.98 dBFS versus Q8 -50.40 dBFS; motion MAD 6.809 versus Q8 5.984.
- IQ1_S: audio RMS -73.11065 dBFS versus Q8 -50.4007 dBFS; motion MAD 6.23523; PSNR 16.112583 dB; SSIM 0.754325; MP4 SHA-256 `b8bc469b9b2f521a844b431afa750dcb5a89b9e7b88bf59123adabb0fda606af`.

The deployed four-step website profile passes; eight-step audio does not.

## Tensor inventory

Both actual QF2 files contain 532 tensors and 20,111,438,744 parameters. Their inventories are identical except for the IQ1 subtype.

| Type | Tensors | Parameters |
|---|---:|---:|
| BF16 | 212 | 28,113,664 |
| F16 | 102 | 43,642,368 |
| F32 | 8 | 707,224 |
| Q8_0 | 26 | 2,312,798,208 |
| Q4_K | 154 | 13,101,957,120 |
| IQ1_S or IQ1_M | 30 | 4,624,220,160 |

The Q8_0 count includes audio/video patch weights inherited from the Q8 conversion input. The BF16 count includes explicit `condition_proj.weight` plus the one-dimensional tensors protected by the patched converter.

## Required companion files

- `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` from [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)
- `minimax_h3_video_vae_fp16.safetensors` and `minimax_h3_audio_vae_fp32.safetensors` from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)

## Run the validated website profile

```bash
sd-cli --mode vid_gen \
  --diffusion-model minimax_h3_fl2va_pruned-UD-IQ1_M-QF.gguf \
  --llm qwen3vl_32b_minimax_h3-Q2_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 39 --steps 4 \
  --cfg-scale 1.0 --backend te=cpu --diffusion-fa \
  --output out.webm
```

`--mode vid_gen`, `--cfg-scale 1.0`, and `--backend te=cpu` are required for this H3 setup. Add `--offload-to-cpu` when needed for GPU memory.

## Provenance

- base model: `MiniMaxAI/MiniMax-H3` revision `6818f6c32d12b210915e44ad56a4228c2608f160`
- pruned FL2VA lineage: `Comfy-Org/MiniMax-H3` revision `014cd40f7e177756c6b2473c0d93b1c89a790dd2`
- QF2 conversion input: `unsloth/MiniMax-H3-GGUF` Q8_0 revision `9ee8213df85a2fcec53dec8c651a0fb1e821674a`
- converter/runtime: `unslothai/stable-diffusion.cpp` revision `13b9d92b5e9a1563536c9c980e700470f9ab6702`
- calibration matrix: 18,693,323 bytes; SHA-256 `6ce80c23f85d50cc170b1bfafa681811e5ef26590757f1c31540485b502c67a2`

Read `LICENSE`, `NOTICE`, and `QUALITY_REPORT.md` before use. These are unofficial Model Derivatives, not official MiniMax or Comfy-Org products, and neither organization endorses them.
