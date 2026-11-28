# Wesley1234/minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_T8

Источник: https://huggingface.co/Wesley1234/minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_T8 · скачано 2026-11-28
Загрузок: 1266 · лайков: 1 · базовая: ["MiniMaxAI/MiniMax-H3","Comfy-Org/MiniMax-H3"] · теги: lora,comfyui,base_model:Comfy-Org/MiniMax-H3,base_model:adapter:Comfy-Org/MiniMax-H3 · обновлено: 2026-08-27T16:30:17.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
pipeline_tag: text-to-video
base_model:
  - MiniMaxAI/MiniMax-H3
  - Comfy-Org/MiniMax-H3
tags:
  - minimax-h3
  - text-to-video
  - text-to-audio
  - audio-video
  - synchronized-audio-video
  - lora
  - comfyui
  - four-step
  - experimental
language:
  - zh
  - en
inference: false
---

# MiniMax H3 Turbo 4-Step — 10Eros Pruned Curve Compatibility LoRA (EXP v001)

An exact-checkpoint-specific, four-step ComfyUI acceleration LoRA for the local
`10Eros_Max_h3_fl2va_bf16_test4_pruned.safetensors` MiniMax H3 fine-tune.

This repository contains an **independent LoRA**, not a merged 40GB diffusion checkpoint. It keeps
the 208 already compatible attention/MLP adapters from a legacy Standard MiniMax H3 Turbo LoRA and
converts its 51 full-width AdaLN adapters to the target checkpoint's 8-dimensional pruned curve
basis.

> [!CAUTION]
> This is an Experimental, exact-base compatibility build. It is not an official MiniMax release,
> not a newly trained or re-distilled Turbo model, and not a generally compatible LoRA for arbitrary
> `pruned` checkpoints. Use it only with the exact main-model SHA-256 documented below.

> [!IMPORTANT]
> MiniMax H3 and Model Derivatives are governed by the
> [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
> That agreement defines excluded territories, redistribution conditions, commercial terms,
> downstream safeguards, an acceptable-use policy, and mandatory `LICENSE`/`NOTICE` handling.
> A public Hugging Face repository can be reachable from excluded territories; repository gating is
> not necessarily geographic access control. Confirm that you have the right to distribute both this
> derivative and the 10Eros fine-tuned-base-dependent adaptation. This model card is not legal advice.

## 中文说明

这是为以下**唯一指定主模型**制作的 MiniMax H3 四步加速 LoRA：

```text
10Eros_Max_h3_fl2va_bf16_test4_pruned.safetensors
SHA-256: f82cc3f723b080e7ae94a7c98f95aa989e387618d0bdc940133dfbd9f432c062
```

普通完整 MiniMax H3 Turbo LoRA 有 259 组 adapter，其中 208 组 attention/MLP 与该 pruned
主模型形状兼容，另外 51 组 AdaLN 的输入维度为 2688，而主模型只接受 8 维曲线坐标，直接用
Bypass 加载会发生维度错误。

本模型没有简单删除这 51 组 AdaLN，而是：

- 保留 208 组原兼容 adapter，416 个 tensor 与来源逐张量完全一致；
- 将 51 个 AdaLN-A 从 `[16,2688]` 投影为 `[16,8]` BF16；
- 51 个 AdaLN-B 保持逐张量完全一致；
- 增加 51 个 FP32 `.diff_b`，保留仿射投影中的常数响应；
- 最终为 259 组 A/B adapter + 51 个 bias diff，共 569 tensors。


## Model details

| Property | Value |
|---|---|
| Model type | MiniMax H3 ComfyUI acceleration LoRA |
| Target task | FL2VA-family joint video + synchronized stereo audio generation |
| Training | None for this conversion |
| Distillation source | Existing legacy Standard four-step Turbo LoRA |
| Target architecture | Exact 10Eros curve-pruned BF16 checkpoint |
| Recommended steps | 4, the only locally validated setting |
| Precision | BF16 A/B adapters; FP32 `.diff_b` |
| Adapter count | 259 A/B adapters + 51 bias patches |
| Tensor count | 569 |
| File size | 794,888,696 bytes, approximately 758.06 MiB |
| Status | Experimental exact-checkpoint compatibility candidate |

### Weight provenance

The acceleration source is the exact local file:

```text
File: minimax_h3_turbo_4步加速_comfyui.safetensors
SHA-256: 35946f9f2957c2766e28b627c88169535249dd07a3040ce3c2c8c99951fdbc7b
```

Its embedded conversion metadata points to local source SHA-256
`9344cd958f8d354da03dd00b7d462933eb5d0cbf11e56a25d8e9911bb971160e`. Independent provenance
inspection identifies it as a small BF16/T8 perturbation of the **initial/legacy non-EMA Standard
four-step line** from
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).

It is **not** v4-step600, ckpt500, ckpt850, or EMA. Do not rename or advertise this file as one of
those checkpoints. The current upstream repository may recommend newer weights; that recommendation
does not change this file's provenance.

### Exact compatible main model

```text
File: 10Eros_Max_h3_fl2va_bf16_test4_pruned.safetensors
Size: 40,225,724,112 bytes
SHA-256: f82cc3f723b080e7ae94a7c98f95aa989e387618d0bdc940133dfbd9f432c062
Structure: 532 tensors; adaln_t_table [1025,8]; 50 blocks + 2 token-refiner blocks
```

The main model is not included here. This card does not establish the authorship, license, or
redistribution rights of that local fine-tuned checkpoint. The publisher must verify those rights
separately.

## What was changed

The full H3 AdaLN path consumes the post-SiLU time embedding

```text
S(t) = SiLU(TimeEmbedder(t)) ∈ R^2688
```

while the curve-pruned checkpoint consumes a shared coordinate table
`C(t) ∈ R^8`. On the 1025-point grid `t_j=j/1024`, each source AdaLN adapter
`A ∈ R^(16×2688), B ∈ R^(d_out×16)` was converted with an affine least-squares fit:

```text
Y = S @ A.T
X = pinv([C, 1]) @ Y
A8 = X[0:8].T
c = X[8]
diff_b = B @ c
```

The stored pruned-path update is therefore:

```text
delta(x) = B @ (A8 @ x) + diff_b
```

The intercept and `.diff_b` are mandatory. A fit without the constant column loses approximately
94%–99.8% of the original AdaLN Turbo response in the local audit.

Only four FP32 time-embedder tensors were read from the full FL2VA reference checkpoint; its
quantized Transformer weights were not used. The 10Eros `adaln_t_table` raw SHA-256 is
`ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e`.

## Checkpoint integrity

### Final LoRA

```text
File: minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_exp_v001.safetensors
Size: 794,888,696 bytes
SHA-256: 6c2f38d45dfa3fc282a48de3171b6946a5e6d46e13f832c43b93734f6d12edf5
```

Structural verification:

- safetensors readback succeeded;
- 259 A/B adapters and 51 `.diff_b` tensors are present;
- current ComfyUI parsed 259 bypass adapters + 51 regular bias patches;
- 569/569 source keys were consumed;
- all 259 target weight shapes and all 51 bias shapes match the exact 10Eros main model;
- 416 direct adapter tensors are bit-identical to the source;
- all 51 projected-module B tensors are bit-identical to the source;
- no alpha tensors are present, preserving scale `1.0` semantics;
- all three input hashes were unchanged after conversion;
- no `.partial` file remained.

Detailed evidence:

- [`evidence/conversion_manifest.json`](evidence/conversion_manifest.json)
- [`evidence/validation.json`](evidence/validation.json)

## ComfyUI usage

### Required components

- the exact 10Eros pruned main model and SHA listed above;
- a current MiniMax H3-compatible ComfyUI build;
- a MiniMax H3 Qwen3-VL text encoder;
- `minimax_h3_video_vae_fp16.safetensors`;
- `minimax_h3_audio_vae_fp32.safetensors`;
- the `minimax-h3-audio-T8` dual-clock nodes used by the validated route.

Suggested layout:

```text
ComfyUI/
└── models/
    ├── diffusion_models/
    │   └── 10Eros_Max_h3_fl2va_bf16_test4_pruned.safetensors
    ├── loras/
    │   └── minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_exp_v001.safetensors
    ├── text_encoders/
    │   └── qwen3vl_32b_minimax_h3_*.safetensors
    └── vae/
        ├── minimax_h3_video_vae_fp16.safetensors
        └── minimax_h3_audio_vae_fp32.safetensors
```

### Required loading route

1. Load the exact 10Eros model with **Load Diffusion Model**.
2. Load this LoRA with **Load LoRA (Bypass, Model Only)** / `LoraLoaderBypassModelOnly`.
3. Start with LoRA strength `1.0`.
4. Use the MiniMax H3 stable dual-clock path with:
   - video steps `4`;
   - audio steps `4`;
   - video shift `12`;
   - audio shift `3`;
   - sampler `dual_clock_euler`;
   - scheduler `native_flow`.
5. Connect the same packed H3 AV latent to the dual-clock setup and
   `SamplerCustomAdvanced.latent_image`.

> [!WARNING]
> Do not use a separate custom node that automatically re-injects full-width time conditioning for
> pruned bases with this file. This LoRA has already converted the 51 AdaLN adapters and includes
> explicit `.diff_b` patches; a second pruned-base adaptation can double-apply or change the intended
> path. The locally validated route is the generic ComfyUI bypass loader plus the T8 dual-clock
> sampler.

Do not combine this route with an extra Sigma Shift node or an external scheduler. Settings above
are the validated baseline, not a guarantee that all prompts will look best at those values.

## Validation

### Static projection

| Metric | Relative error |
|---|---:|
| Stored 1025-point aggregate | `9.57255e-5` |
| Stored native-four aggregate | `1.58861e-4` |
| Worst native-four module | `5.85406e-4` |

The native-four grid includes video/audio clocks and visual/audio conditioning timestep classes.
These are response-preservation metrics, not perceptual-quality scores.

### Real four-step AV smoke

```text
ComfyUI: 0.31.0, commit cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44
GPU: NVIDIA GeForce RTX 4060 Ti 16GB
Attention: SageAttention in the live server
Task: T2VA
Resolution: 256 × 256
Frames: 124 at 24 FPS
Steps: 4 video / 4 audio
Sampler/scheduler: dual_clock_euler / native_flow
Shifts: 12 / 3
Seed: 2608104101
Strength: 1.0
```

The final LoRA and a `core208` AdaLN-free ablation both completed real joint H3 generation, dual VAE
decoding, and MP4 muxing.

| Treatment | Runtime | Video | Decoded audio | Local faster-whisper |
|---|---:|---|---|---|
| curveproj1025 | 357.84s | 124 coherent frames in six-frame screen | stereo 32kHz; 162,816 samples; RMS 0.01879; peak 0.15245; finite; 0% clipping | `Today, the rain finally stopped.` |
| core208 ablation | 288.58s | 124 coherent frames in six-frame screen | stereo 32kHz; 162,816 samples; RMS 0.01222; peak 0.12440; finite; 0% clipping | `Today the rain finally stopped` |

The pair's full-video RGB MAD was `5.615/255`, and decoded-audio correlation was `0.719`. This proves
the projected AdaLN route is active rather than behaviorally identical to dropping the 51 modules;
it does not prove that either treatment is perceptually superior.

![curveproj1025 versus core208 contact sheet](assets/curveproj1025_vs_core208_contact.png)

## Limitations

- This is not an official MiniMax, Comfy-Org, LarryVrh, or 10Eros release.
- It is tied to one exact 40GB fine-tuned pruned checkpoint SHA.
- It was not trained, fine-tuned, re-distilled, or calibrated against a broad prompt set.
- The structural projection repairs the pruned AdaLN width mismatch; it cannot guarantee that the
  fine-tuned main model still follows the original Turbo four-step semantic trajectory.
- Real evaluation covers one prompt, one seed, 256×256, 124 frames, and SageAttention.
- No 20-step no-LoRA quality anchor, four-step no-LoRA negative control, Stock-Attention comparison,
  high-resolution matrix, held-out prompt/seed matrix, blind listening, identity metric, or trained
  lip-sync metric has passed.
- The smoke proves intelligible speech and absence of numerical noise/clipping in one case; it does
  not guarantee universal audio quality.
- More steps, alternative samplers/schedulers, arbitrary strengths, other pruned checkpoints,
  INT8/FP8 target bases, Ref2VA, and third-party pruned reinjection nodes are unvalidated.
- If held-out four-step quality is unstable or not better than the core208 ablation, no-training
  static adaptation has reached its evidence boundary and re-distillation/training is required.

## Intended use

Intended for controlled local research and A/B evaluation of four-step MiniMax H3 generation on the
exact compatible 10Eros main model, subject to all upstream licenses and the acceptable-use policy.

Not intended for claims of universal acceleration quality, arbitrary pruned-base compatibility,
identity preservation, deterministic lip sync, medical/forensic use, deceptive impersonation,
bypassing safety systems, or any use prohibited by the MiniMax H3 license.

## License and attribution

MiniMax H3 and this Model Derivative are governed by the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
The upstream license, including territorial restrictions, redistribution conditions, commercial
terms, downstream safeguards and acceptable-use policy, takes precedence over this card.

The Turbo source repository declares Apache-2.0 metadata:
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).
That does not replace the MiniMax H3 base-model license or any separate rights attached to the local
10Eros fine-tuned checkpoint.

Before distributing this LoRA:

1. include an unmodified copy of the official MiniMax H3 `LICENSE`;
2. include the provided [`NOTICE`](NOTICE);
3. retain the prominent modification and exact-base notices;
4. verify rights to distribute the 10Eros-dependent derivative;
5. comply with the applicable-territory and downstream-user requirements;
6. verify any commercial authorization requirements directly with MiniMax.

Upstream projects:

- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
- [MiniMax H3 ComfyUI guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)

## Modification notice

`minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_exp_v001.safetensors` is an
independently modified Model Derivative. It was produced without training by preserving 208 source
adapters and converting 51 AdaLN adapters to the exact 10Eros pruned curve basis with an affine
1025-point projection and explicit bias deltas. It is not endorsed by MiniMax, Comfy-Org, LarryVrh,
or the author of the 10Eros fine-tuned checkpoint.
