# yitongl/minimax-h3-nvfp4-lora-recovery

Источник: https://huggingface.co/yitongl/minimax-h3-nvfp4-lora-recovery · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая:  · теги: quantization,lora,post-training-quantization · обновлено: 2026-08-10T04:36:50.000Z

---

---
license: other
tags:
  - minimax-h3
  - nvfp4
  - quantization
  - lora
  - post-training-quantization
---

# LoRA-only PTQ recovery for MiniMax-H3 NVFP4 — deployed on an RTX 5090

Freeze the 4-bit weights, train **only** the rank-128 LoRA to reproduce the BF16 layer output on
real teacher-forced rows, then run the result through the real NVFP4 kernel on sm_120.

```
y = Q4(x) @ Q4(W_r)^T  +  (x @ down) @ up  +  b
    |___ frozen ____|     |___ trained __|
```

`lambda = 1` — no activation preprocessing, nothing folded into the weights, so the whole
`1/lambda` path in the converter is the identity and the channel-order hazards that have bitten
this project elsewhere cannot arise here.

## The layer-wise result

300 block linears, held-out clips, rel-L2 against BF16:

| arm | median |
| --- | --- |
| rank 0 (no low-rank branch) | 0.1044 |
| truncated SVD | 0.0988 |
| **closed-form alternating fit** (the initialisation) | 0.0913 |
| **trained** | **0.0807** |

**299/300 layers improve, median +12.4%.** The opponent is the *alternating* fit, not plain SVD —
quoting a gain over SVD would be true and uninteresting.

## Data scaling saturates at ~16k rows/layer

Continuing the same checkpoint on 4x the data (65,536 rows/layer from 512 replayed steps, vs
16,384 from 128) moves the median trained error from **0.0807 to 0.0808** — nothing. Only 139/300
layers improve by more than 0.5%.

An earlier reading of the gain-percentage curve suggested it had not saturated. That was an
artifact of the metric: `gain` is measured against `closed_form`, and more data improves the
closed-form fit too, so the ratio keeps moving while the absolute error does not. **Read the
absolute error.** 16k rows/layer is the point of diminishing returns.

## Deployed on the real kernel

Both arms converted to NVFP4 and generated on one RTX 5090, 768x1344, 124 frames, 50 steps,
seed 1101. All 300 LoRAs loaded, none skipped.

**The two arms share their 4-bit weights bit for bit.** `W_r = Q(W − down_cf@up_cf)` and both
artifacts carry the same `down_cf`/`up_cf`, so only the 16-bit low-rank branch differs — no
re-quantization, no re-calibration, same seed, same scheduler. Any difference is attributable to
the training and to nothing else.

| arm | pixel PSNR vs BF16 | mean RGB | sharpness (lap var) |
| --- | --- | --- | --- |
| closed-form (untrained) | 14.31 | 37.51 | 63.1 |
| **trained** | **14.38** | 37.28 | **80.8** |
| BF16 reference | — | 36.27 | 40.3 |

**The training does not show up in this measurement, and that is a statement about the metric.**
0.07 dB is inside the noise: at 50 steps, PSNR-against-BF16 measures which basin the sample fell
into, not image quality. A companion study on this model showed the divergence saturates by
denoising evaluation ~8 and then does not move for the remaining 41 steps — so a 50-step pixel PSNR
cannot resolve a per-layer improvement of this size.

A latent-space measurement of the same LoRAs *did* show the gain: −8.1% video / −8.9% audio on x0
latents against the BF16 teacher, which is the same quantity the training optimises.

What both arms *do* show, consistently, is **more high-frequency energy than the reference**
(63.1 and 80.8 against 40.3), and training increases it. The LoRA is fitting each layer's BF16
output in L2, not matching a spectral profile, so it gets closer in the trained metric while
carrying more high-frequency quantization noise.

## Files

| file | |
| --- | --- |
| `sbs_recovery.mp4` | **BF16 \| closed-form \| trained**, side by side, labelled |
| `sheet.png` | 5 frames from each arm |
| `0_bf16.mp4`, `recov_closed_form.mp4`, `recov_trained.mp4` | each arm alone |
| `RESULT.md` | full method, cost measurements, verification, and the on/off-policy result |

## What is not established

- **The end-to-end comparison here is one prompt at 50 steps.** The step count where the metric
  still discriminates is ~6; that run has not been made on the real kernel.
- **On-policy vs off-policy disagree between measurements.** Layer-wise cross-evaluation puts them
  within 0.02%; a latent-space rollout put off-policy ~2x ahead. Teacher forcing pins the input
  distribution and a rollout does not. n=2 prompts — a signal to measure, not a settled effect.
- `blocks.0.ff.net.2` gains **+0.0%** at every data size tried while its training error falls
  72–75%. It is one layer, not a layer type; more data is the wrong intervention there.
