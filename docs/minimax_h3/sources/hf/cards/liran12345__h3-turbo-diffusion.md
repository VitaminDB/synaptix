# liran12345/h3-turbo-diffusion

Источник: https://huggingface.co/liran12345/h3-turbo-diffusion · скачано 2026-11-28
Загрузок: 0 · лайков: 0 · базовая: MiniMaxAI/MiniMax-H3 · теги: turbodiffusion · обновлено: 2026-11-03T09:48:05.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
pipeline_tag: image-text-to-video
tags:
  - minimax-h3
  - image-to-video
  - audio-video-generation
  - dmd
  - bf16
  - safetensors
  - turbodiffusion
---

# H3 Turbo Diffusion

Full **BF16 (16-bit), four-update DMD transformer** derived from MiniMax H3
FL2VA for image-to-video generation with synchronized audio.

These are the complete, unquantized DMD DiT weights. The source DMD checkpoint
is uploaded without re-serialization or changes to its tensor values.
The original checkpoint stores 626 tensors in BF16 and 12 input/output
and time-embedding tensors in FP32 (about 66 MiB); these original precisions
are preserved. This is an independent derivative, not an official MiniMax release.

## Contents

- `transformer/`: all 34 original safetensors shards, config and index.
- `adaln/`: the DMD AdaLN table for the schedule below, plus its metadata.
- `manifest.json`: checkpoint tensor counts, file sizes and SHA256 checksums.
- `LICENSE` and `NOTICE`: upstream model terms and derivative attribution.

The transformer weights are approximately 61.73 GiB. The index metadata is
normalized to omit internal storage locations; its tensor-to-shard mapping
and every shard's bytes are unchanged. Text encoder, video/audio
VAEs, tokenizer and other original H3 components are **not bundled**. Obtain
those from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
Obsolete configurations for a separate inference implementation were removed;
use `configs/h3_i2va_w8a8.yaml` in the TurboDiffusion code repository.

## Download

From a TurboDiffusion checkout containing the H3 integration, use its bundled
downloader:

```bash
python -m turbodiffusion.inference.download_h3 \
  --dit-dir /models/h3-turbo-diffusion \
  --components-dir /models/MiniMax-H3
```

It calls `snapshot_download(allow_patterns=[...])`, pins both repositories to
immutable revisions, and verifies the selected DMD files against the pinned
manifest's sizes and SHA256 hashes. This avoids incompatible multi-pattern
`hf --include` parsing between Hugging Face Hub 0.36.x and 2.x.
Do not upgrade the H3 environment's Hub package independently of Transformers.
The publisher manifest verifies release consistency, not organizational
ownership or a cryptographic signature.

For downloading the weight files before obtaining the integrated code, the
Python API is also available:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "liran12345/h3-turbo-diffusion",
    revision="d02a14305ac620253c5da2278b1be0df865f969d",
    allow_patterns=["transformer/*", "adaln/*", "manifest.json", "LICENSE", "NOTICE"],
    local_dir="/models/h3-turbo-diffusion",
)
```

That immutable weight revision has the FP32-timestep AdaLN table and unchanged
transformer shards. Use the bundled downloader to verify its manifest.
Old local configuration files are not automatically removed by downloads.

## AdaLN Precision Update

On **2026-11-03**, the AdaLN table was replaced with the verified table generated
using the source checkpoint's original **FP32 timestep MLP**. The transformer
shards were not changed. The table uploaded on 2026-10-29 used a BF16 timestep
MLP; users with that version should refresh both the table and its JSON:

```bash
python -m turbodiffusion.inference.download_h3 \
  --dit-dir /models/h3-turbo-diffusion --adaln-only
```

The new JSON and embedded safetensors metadata record
`"timestep_mlp_dtype": "float32"`. The modulation table itself is still stored
in BF16, matching the source AdaLN projection computation; this does not mean
the timestep MLP ran in BF16. No local table regeneration is needed for the
published schedule. The loader rejects inconsistent embedded and JSON metadata.

## Inference Settings

The integrated H3 implementation is submitted to the official TurboDiffusion
repository in [PR #144](https://github.com/thu-ml/TurboDiffusion/pull/144).
Use that implementation and its `docs/h3_i2va.md`; no mutable personal code
branch is needed as a documentation link.
These are transformer weights, **not a standalone
`DiffusionPipeline.from_pretrained` bundle**. The `MiniMaxH3Transformer3DModel`
config does not imply that every Diffusers release includes that model class.

The verified schedule is `i2va / steps=5 / video_shift=12 / audio_shift=3`.
Five sigma points give **four denoising updates** in the H3 runtime.
The AdaLN table is specific to these source weights, the task and the shifts.
Do not reuse it for a different schedule.

## TurboDiffusion W8A8 Export

From a TurboDiffusion checkout containing the H3 integration, after building
the native CUDA extension:

```bash
CUDA_VISIBLE_DEVICES=0 python -m turbodiffusion.inference.export_h3_w8a8 \
  --model-path /models/h3-turbo-diffusion/transformer \
  --output /models/h3-turbo-diffusion/model_w8a8.safetensors
```

This uses TurboDiffusion's native W8A8 projections and preserves the original
FP32 input/output/timestep tensors and RoPE frequencies. INT8 checkpoints
exported before the FP32-preservation fix must be regenerated from the full
source weights. The published FP32-timestep AdaLN table can be reused.
To generate another copy, add
`--adaln-output /models/h3-turbo-diffusion/adaln/adaln_i2va_s5_v12_a3.safetensors`.

SpargeAttn is an external dependency. The integrated video decoder uses native
W8A8 projections directly from the public VAE weights at startup; no separate
VAE export is required. Audio VAE decoding remains FP32.
W8A8 conversion can change generated results; the uploaded transformer remains
unquantized. See the TurboDiffusion H3 documentation for installation and
the `fullopt` no-cache default or explicit `fullopt-dbcache` profile.

## Limitations

The full BF16 DMD model was used as the reference for five 768p I2VA test
cases. That is a small functional evaluation, not comprehensive quality,
safety or audio evaluation. This release does not claim lossless parity
with the original many-step FL2VA model or Wan-specific speedups.
The AdaLN precision update is a computation-correctness change, not evidence
of a measured video quality improvement.

## License

The weights are governed by the **MiniMax H3 Community License Agreement**,
not MIT or TurboDiffusion's code license. The license includes territorial,
use and commercial conditions; its excluded territories include the EU,
UK, Republic of Korea and USA. Review the full [LICENSE](LICENSE) and obtain
any required permissions before downloading, using or redistributing.
The upstream acceptable-use policy is included in that agreement.

The DMD transformer and derived AdaLN table are modifications of the
original H3 works. See [NOTICE](NOTICE) for attribution. Generated outputs
must be identified as AI-generated as required by the upstream terms.
