# WaveCut/MiniMax-H3-OrbitQuant-W4A4

Источник: https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4 · скачано 2026-11-28
Загрузок: 36 · лайков: 3 · базовая: MiniMaxAI/MiniMax-H3 · теги: orbitquant,comfyui · обновлено: 2026-10-02T02:22:37.000Z

---

---
pipeline_tag: image-text-to-video
library_name: diffusers
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
tags:
  - orbitquant
  - comfyui
  - w4
  - w4a4
  - native-w4a4-transformer-runtime
  - text-to-video
  - image-text-to-video
  - audio-video-generation
---

# MiniMax H3 — OrbitQuant W4A4 with source FP32 VAEs

OrbitQuant conversion of
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), pinned to
source revision
[`73372e6cf53e414edd3ab03e357717fb0602e758`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/73372e6cf53e414edd3ab03e357717fb0602e758).

Eligible linear weights in `transformer`, `transformer_ref`, and the Qwen3-VL
`text_encoder` are stored and executed through OrbitQuant's native packed W4A4
path. Non-quantized boundaries use BF16 compute. The visual `vae` and
`audio_vae` are byte-for-byte FP32 source copies and are never quantized.

`transformer` and `transformer_ref` are stored as fused blocks for OrbitQuant
0.11 (see [Fused transformers](#fused-transformers-orbitquant-011)); on an RTX
3090 a T2VA generation takes 115 s instead of 175 s. The per-projection layout
for OrbitQuant 0.9.2 is at revision
[`478202c3`](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/tree/478202c387a37b327843fd8556fd7983f6721ef1).

## Final 608×480 example

<video controls playsinline preload="metadata" width="608" aria-label="MiniMax H3 OrbitQuant balanced ComfyUI proof">
  <source src="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/examples/hevc/comfyui-balanced-t2va-608x480-crf10.mp4" type="video/mp4; codecs=&quot;hvc1&quot;">
  <source src="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/examples/h264/comfyui-balanced-t2va-608x480.mp4" type="video/mp4">
  <a href="https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4/resolve/main/comfyui/examples/h264/comfyui-balanced-t2va-608x480.mp4">MiniMax H3 OrbitQuant proof MP4</a>
</video>

[H.265 10-bit CRF 10](comfyui/examples/hevc/comfyui-balanced-t2va-608x480-crf10.mp4)
· [H.264 fallback](comfyui/examples/h264/comfyui-balanced-t2va-608x480.mp4)
· [CRF 1 yuv444p master](comfyui/examples/masters/comfyui-balanced-t2va-608x480-crf1.mp4)
· [16-frame overview](comfyui/examples/timelines/comfyui-balanced-t2va-608x480.jpg)
· [adjacent-frame review](comfyui/examples/review/comfyui-balanced-adjacent-triplets.jpg)
· [audio spectrum](comfyui/examples/review/comfyui-balanced-audio-spectrum.png)

This live ComfyUI run uses 608×480, 124 frames at 24 FPS, seed 42, and 24
sigma points / 23 denoiser forwards. All 300 eligible denoiser linears use the
native packed W4A4 path with no exact INT8 weight cache. Source FP32 tiled visual
decode produced the retained CRF 1 master; the HEVC card copy was derived from
that master at CRF 10. The output contains AAC stereo at 32 kHz.

Full-resolution frames and adjacent triplets were reviewed for face geometry,
eyes, lips, grid artifacts, ghosting, texture breakup, and abrupt section
redraw. The macro-to-face shot remains coherent. The audio spectrum is broadband
without a persistent narrow electronic whistle.

## ComfyUI workflow

Download the ready-to-import
[MiniMax H3 OrbitQuant T2VA workflow](comfyui/workflows/MiniMax-H3-OrbitQuant-T2VA.json).
It is based on Comfy-Org's bundled
[`video_minimax_h3_t2v.json`](https://github.com/Comfy-Org/workflow_templates/blob/7653f1cdef1d92394b6ef9946018c0a8aa4136b8/templates/video_minimax_h3_t2v.json)
and preserves the official preset's readable composition.

![MiniMax H3 OrbitQuant public ComfyUI workflow](comfyui/workflow-export.png)

The PNG above is a 3060×1310 ComfyUI Workflow Image Export, not a browser
screenshot. Its `tEXt` `workflow` chunk contains the same six-node graph with
`balanced`, 608×480, 124 frames, 24 steps, and the detailed example prompt.

Install [ComfyUI-OrbitQuant](https://github.com/iamwavecut/ComfyUI-OrbitQuant)
into `ComfyUI/custom_nodes`, restart ComfyUI, import the workflow, and set
`OrbitQuant Release Loader.model_path` to this downloaded model directory. The
graph uses only the generic public nodes `OrbitQuant Release Loader` and
`OrbitQuant Generate Video`; there are no MiniMax-specific public node classes.

On the RunPod ComfyUI image, launch ComfyUI with:

```bash
python main.py --listen 0.0.0.0 --port 8188 \
  --disable-cuda-malloc \
  --disable-dynamic-vram \
  --disable-async-offload
```

These supported flags let the OrbitQuant subprocess enforce its own allocator
cap instead of competing with ComfyUI's global DynamicVRAM and async-offload
layers.

## Fused transformers (OrbitQuant 0.11)

Every block of `transformer` and `transformer_ref` runs fused: one Q|K|V W4A4
GEMM whose activation-quantization prologue applies the RMSNorm and the AdaLN
modulation, read per row from the block's (timestep, modality) table; one Q/K
RMSNorm + rotate-half RoPE kernel; the attention output projection and the
SwiGLU feed-forward with their gated residual updates in the GEMM epilogues;
and attention with INT8 Q·Kᵀ and FP16 P·V in 47 of the 50 blocks (the three
blocks whose Q/K norm weights spread too far keep BF16 SDPA). The 300 W4A4
linears of each transformer form 200 fused groups of the same 4-bit codes, so
the files keep their size.

The runner streams fused blocks whole (block-level group offloading, also when
`leaf_level` is requested) without recorded streams, keeps the remaining
top-level modules on the GPU for the denoising loop, and computes the token
refiner (two transformer blocks over the text rows) once per prompt from its
own offloaded blocks. Ref2VA encodes the reference before the denoiser is
placed, with the VAEs on the GPU when they fit
(`--reference-vae-placement auto`). The decoder keeps the FP32 visual VAE
resident when 13 GiB are free (`--video-vae-placement auto`).

RTX 3090 24 GB, torch 2.10.0+cu128, Triton 3.6.0, Diffusers `abc5e9bf`,
608×480, 124 frames, 24 sigma points / 23 forwards, one run per cell. The 0.9.2
column ran the runner and per-projection layout of revision `478202c3` on the
same GPU. Peak is the GPU's NVML memory during generation; generation is the
pipeline call (conditioning and denoising) without loading and decoding.

| Profile | Task | 0.9.2: peak / denoise / generation | 0.11 fused: peak / denoise / generation |
| --- | --- | ---: | ---: |
| `speed` | T2VA | 19.70 GiB / 143.1 s / 175.3 s | 19.95 GiB / 92.3 s / 116.1 s |
| `balanced` (default) | T2VA | 5.33 GiB / 145.6 s / 174.5 s | 3.70 GiB / 95.1 s / 114.7 s |
| `minimum_vram` | T2VA | 5.31 GiB / 173.1 s / 222.5 s | 3.66 GiB / 130.1 s / 150.1 s |
| `speed` | Ref2VA | 21.94 GiB / 399.2 s / 703.8 s | 22.48 GiB / 260.5 s / 323.7 s |

The separate decode of the 124-frame video takes 15.9 s with the resident FP32
visual VAE (10.7 GiB peak allocated) instead of 177.5 s sequentially offloaded
(1.0 GiB). `minimum_vram` now peaks like `balanced` and differs only in
pinning less host memory (37 instead of 43 GiB resident set) at a slower
transfer rate.

The fused kernels round differently from the per-projection path, so outputs
at the same seed are not bit-identical. The T2VA balanced and speed latents of
this revision are identical to each other, and frames 0, 40, 80 and 120 of the
T2VA and Ref2VA examples keep face geometry, eyes, lips and motion coherent. In
Ref2VA a resident VAE encodes the reference untiled, as the model does without
memory limits; the 128-pixel tiled encode gives slightly different conditioning,
so the two placements produce different videos at the same seed.

### OrbitQuant 0.9.2 reference measurements

The measurements below describe the per-projection layout (revision
`478202c3`). All numbers use CUDA 13, 608×480, 124 frames, 24 sigma points /
23 forwards, native-auto Torch Flash SDPA, no weight cache, sequential CUDA text
conditioning, and source FP32 VAEs.

| Profile | GPU | Task | Placement | Process peak | Denoise | Generation |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `balanced` (default) | RTX PRO 6000 | T2VA | streamed leaf offload, 12 GiB cap | 6.36 GiB child; 6.90 GiB incl. idle ComfyUI | 46.68 s | — |
| `speed` | RTX PRO 6000 | T2VA | resident transformer | 21.14 GiB | 46.84 s | 51.10 s |
| `minimum_vram` | RTX 4090 | T2VA | low-CPU-memory streamed leaf offload, 8 GiB cap | 4.07 GiB | 154.25 s | 188.70 s |
| `speed` | RTX PRO 6000 | Ref2VA | resident `transformer_ref` | 24.06 GiB | 118.48 s | 155.42 s |

`balanced` is the recommended Pareto recipe. On the tested PRO 6000, streamed
weight movement overlaps denoising closely enough to match the resident path
while cutting the child process's physical CUDA peak by about 70%.
`minimum_vram` is the verified absolute-minimum endpoint. `speed` removes
transformer transfers when VRAM is available.

SageAttention2's available CUDA 13 binary did not include SM120 code for this
PRO 6000, and forced cuDNN attention was slower. Native-auto Torch Flash SDPA is
therefore the shipped supported attention path.

## Install

```bash
pip install "orbitquant[hf,kernels]>=0.11.0,<0.12"
pip install "diffusers @ git+https://github.com/huggingface/diffusers.git@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
pip install "transformers>=5.13,<6" accelerate av soundfile
```

Or install all pinned runtime requirements from this repository:

```bash
pip install -r runtime-requirements.txt
```

## Direct runner

The runner writes each scheduler checkpoint atomically and saves the latent
bundle before decode. The examples below keep the prompt in a file to avoid
shell quoting a multi-kilobyte description.

Balanced T2VA:

```bash
python scripts/run_quantized_example.py \
  --release . \
  --output balanced.mp4 \
  --save-latents balanced.latents.pt \
  --prompt "$(cat prompt.txt)" \
  --seed 42 --width 608 --height 480 --num-frames 124 --steps 24 \
  --manual-stage-offload \
  --text-encoder-sequential-offload \
  --transformer-group-offload-type leaf_level \
  --group-offload-use-stream \
  --cuda-memory-cap-gib 12 \
  --transformer-runtime-mode auto_fused \
  --checkpoint-dir checkpoints/balanced
```

Maximum-speed T2VA: remove the group-offload and allocator-cap options while
keeping `--manual-stage-offload --text-encoder-sequential-offload`.

Minimum-VRAM T2VA: use the balanced command with
`--group-offload-low-cpu-mem-usage --cuda-memory-cap-gib 8`.

Ref2VA speed:

```bash
python scripts/run_quantized_example.py \
  --release . \
  --output ref2va.mp4 \
  --save-latents ref2va.latents.pt \
  --prompt "$(cat prompt.txt)" \
  --task ref2va --reference reference.png \
  --seed 42 --width 608 --height 480 --num-frames 124 --steps 24 \
  --manual-stage-offload \
  --text-encoder-sequential-offload \
  --transformer-runtime-mode auto_fused \
  --checkpoint-dir checkpoints/ref2va
```

The reference VAEs enter the GPU for the reference encode when the device (or
the allocator cap) leaves 3 GiB beside their 10.3 GiB of weights. On smaller
GPUs add `--reference-vae-placement sequential_offload --reference-vae-tile-size 128`.

Decode only after the latent-producing process exits:

```bash
python scripts/decode_h3_latents.py \
  --latents balanced.latents.pt \
  --vae vae \
  --audio-vae audio_vae \
  --output balanced.master-crf1.mp4 \
  --preview-output balanced.mp4
```

The decoder always loads the release's untouched source FP32 visual and audio
VAEs. The tiled visual VAE stays resident on the GPU when 13 GiB are free and is
sequentially offloaded otherwise (`--video-vae-placement`); the audio VAE enters
GPU only for the audio stage.

## Component precision and size

| Component | Stored mode | Artifact GiB | Eligible linear coverage | OrbitQuant modules | AdaLN INT4 |
| --- | --- | ---: | ---: | ---: | ---: |
| `transformer` | W4A4 | 17.03 | 97.45% | 300 | 50 |
| `transformer_ref` | W4A4 | 17.03 | 97.45% | 300 | 50 |
| `text_encoder` | W4A4 | 18.55 | 95.80% | 448 | 0 |
| `vae` | source FP32 copy | 9.70 | exact source copy | 0 | 0 |
| `audio_vae` | source FP32 copy | 0.56 | exact source copy | 0 | 0 |

Input/output projections, time/context/refiner boundaries, embeddings, norms,
and language-head boundaries excluded by the pinned H3/Qwen policy remain in
source precision. “Four bit” describes eligible packed linear weights, not
every tensor in the architecture.

## Validation and provenance

- OrbitQuant 0.9.2 revision `cd58b4ecf77f22b8c4116b3d0b7d4af258e16ba3`
  (quantization).
- OrbitQuant 0.11.0 revision `74f0bc8be9c0329e58c7b7cc58ebdbe4aaaa3df1` (fused
  transformers, converted from revision `478202c3` with `orbitquant.fused`).
- Diffusers revision `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`.
- The live ComfyUI workflow reached terminal `pass` through `/prompt` and
  produced the standard `VIDEO` output.
- All 23 denoiser checkpoints and the final latent were persisted before
  source-FP32 decode.
- Visual/audio VAE weight SHA256 values match the pinned source revision; see
  [`validation/source_component_copy_audit.json`](validation/source_component_copy_audit.json).
- Exact artifact hashes are in [`SHA256SUMS`](SHA256SUMS).
- Full machine-readable profile and media evidence is in
  [`comfyui/report.json`](comfyui/report.json).

## License and modifications

The original
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/73372e6cf53e414edd3ab03e357717fb0602e758/LICENSE)
is copied as [`LICENSE`](LICENSE). See [`NOTICE`](NOTICE),
[`MODIFICATIONS.md`](MODIFICATIONS.md), and the upstream
[`QA-about-License`](docs/QA-about-License.md).
