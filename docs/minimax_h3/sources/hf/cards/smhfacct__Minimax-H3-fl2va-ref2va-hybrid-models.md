# smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models

Источник: https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models · скачано 2026-11-28
Загрузок: 0 · лайков: 428 · базовая: ["MiniMax-H3-fl2va","MiniMax-H3-ref2va"] · теги:  · обновлено: 2026-08-16T22:53:44.000Z

---

---
license: other
tags:
  - video-generation
  - text-to-video
  - image-to-video
  - audio-video-generation
  - diffusion-transformer
  - merge
  - hybrid
base_model:
  - MiniMax-H3-fl2va
  - MiniMax-H3-ref2va
pipeline_tag: text-to-video
---

# MiniMax H3 Hybrid (fl2va base + ref2va reference pathway)

This is a **merged variant of MiniMax H3**, a joint audio+video diffusion transformer (DiT). It combines the two officially released MiniMax H3 checkpoints — `fl2va` and `ref2va` — into a single model that aims to keep the best qualities of each.

**NOTE:** All of the models in this repo (those noted below) were created from the pruned int8-convrot base models. 

## Why this merge exists

MiniMax released two H3 checkpoints with identical architecture and weight layout, but different training regimes:

- **`fl2va`** — trained on first/last-keyframe conditioning only. Produces noticeably higher visual and audio output quality.
- **`ref2va`** — additionally trained on multimodal reference conditioning (image, video, and audio references), enabling reference-driven generation. However, a known training-quality issue makes its raw output — even outside of the reference-conditioning use case — noticeably worse than `fl2va`.

This creates an awkward tradeoff: `ref2va` is the only checkpoint that supports reference conditioning at all, but it does so at a real cost to output quality — including on the exact same non-reference generation tasks where `fl2va` excels.

A tensor-by-tensor comparison of the two checkpoints shows that the overwhelming majority of weights — attention QKV/output projections, MLPs, RMSNorms, patch projections, rotary position embeddings, and the token refiner — are bit-identical or extremely close (cosine similarity ≥ 0.9997) between the two models. The meaningful differences are concentrated almost entirely in the per-block **`adaln_proj`** weights: the AdaLN (adaptive layer norm) modulation projections that route text, audio, video, and *reference* modality signals into the residual stream at each transformer block. The final AdaLN projection and the video/audio output heads differ to a smaller degree as well.

In other words: the part of the network primarily responsible for *processing* reference conditioning signal is fairly localized, while the parts responsible for general visual/audio fidelity are shared almost entirely between the two checkpoints. That made a targeted merge look promising rather than a lossy compromise.

## Available variants

Two variants are provided, differing only in *how many* of the later transformer blocks pull their `adaln_proj` weights from `ref2va`. Both use `fl2va` as the base for everything else.

| File | Blocks taken from `ref2va` | Tradeoff |
|---|---|---|
| `minimax_h3_hybrid_fl2va_ref2va_b30-49-int8.safetensors` | 30–49 (last 20 of 50) | Closer to `fl2va` — higher visual/audio quality, somewhat reduced reference capability. |
| `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors` | 25–49 (last 25 of 50) | Closer to `fl2va` — slight higher visual/audio quality, slightly reduced reference capability. |
| `minimax_h3_hybrid_fl2va_ref2va_b20-49-int8.safetensors` | 20–49 (last 30 of 50) | Closer to `ref2va` — slightly higher reference capability, slightly lower visual/audio quality. |
| `minimax_h3_hybrid_fl2va_ref2va_b15-49-int8.safetensors` | 15–49 (last 35 of 50) | Closer to `ref2va` — higher reference capability, somewhat lower visual/audio quality. |

Both are reasonable defaults depending on what you're optimizing for: pick **b30-49** if output fidelity matters more and reference conditioning just needs to work well; pick **b20-49** if faithfulness to the reference input is the priority and you can tolerate a small step down in raw quality. Neither variant is uniformly better than the other — the difference is a fairly narrow, gradual tradeoff rather than a clear win/loss.

I personally recommend trying the *b25-49* version first, and if you find you need stronger reference adherence (at the cost of output quality), try the *b20-49* or *b15-49* version.

## What this model is

Both variants use **`fl2va` as the base** — providing its higher-quality attention, MLP, normalization, token refiner, and output-head weights throughout the network — with the **per-block `adaln_proj` weights for a range of later transformer blocks taken from `ref2va`** instead (see the table above for the exact range per variant). Everything else, including the earlier-network `adaln_proj` weights and the final-layer AdaLN projection, remains on `fl2va` in both variants.

The intent is to preserve `ref2va`'s reference-conditioning pathway — which is expressed primarily through those later-block AdaLN modulation weights — while keeping the rest of the network on the higher-fidelity `fl2va` weights that govern overall visual and audio quality. This configuration was arrived at empirically, by comparing outputs across different block ranges and preset combinations, and represents the best subjective balance found between reference fidelity and output quality.

## Intended use

- Text/image/video/audio-to-video generation where you want to condition on a reference image, video, or audio clip (the capability unique to `ref2va`), while retaining output quality closer to `fl2va`.
- A drop-in replacement for `ref2va` in reference-conditioned workflows, for users who found `ref2va`'s raw output quality lacking.

This model is **not** expected to exceed `fl2va`'s quality on non-reference-conditioned generation, since the great majority of its weights are identical to `fl2va` to begin with; the goal is to close the gap on reference-conditioned generation without giving up `fl2va`'s output quality elsewhere in the network.

## How it was made

Both source checkpoints are unmodified, officially released MiniMax H3 weights. Each variant was produced by combining them at the tensor level: for each weight in the network, either the `fl2va` or `ref2va` version was selected according to that variant's block-range rule above, and the result was saved as a new, self-contained checkpoint with the same architecture and tensor layout as the source models. No additional training, fine-tuning, or gradient-based optimization was performed — this is a weight-selection merge, not a fine-tune.

## Limitations

- This is an experimental merge based on empirical, subjective comparison rather than a principled or exhaustively validated method. Results may vary by prompt, reference type, and generation settings.
- Because the merge boundary was chosen based on the *known* architectural role of `adaln_proj`, there may be additional latent quality/reference tradeoffs not fully captured by this particular block range.
- Inherits all licensing terms and usage restrictions of the original MiniMax H3 `fl2va` and `ref2va` checkpoints. See MiniMax's original model license for terms; this merge does not grant any rights beyond what the source checkpoints permit.

## Acknowledgements

Built entirely from MiniMax's original H3 `fl2va` and `ref2va` checkpoints. All credit for the underlying model architecture and training goes to MiniMax.
