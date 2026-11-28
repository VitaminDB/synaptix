# antirez__h3.c

Source: https://github.com/antirez/h3.c
Fetched: 2026-11-28

Description: MiniMax H3 inference engine for Mac computers
Stars: 3100  Updated: 2026-08-11T13:53:32Z



---

## FILE: README.md (https://github.com/antirez/h3.c/blob/HEAD/README.md)

# h3-metal

Native MiniMax-H3 inference for Apple Silicon. The project is being built as a
sequence of working vertical slices: deterministic host/model metadata first,
then portable Metal block parity, prompt encoding, prompt-to-video/audio, and
first/last-frame conditioning and then ordered references.

Prompt-to-video/audio, first/last-frame conditioning, and ordered Ref2VA
image/video/audio references work end to end. The current work is incremental
H3-specific Metal performance and memory optimization on M3 Max and M5 Max.

## Tutorial

### 1. Build and inspect the model

The examples assume that the Hugging Face snapshot is in `./MiniMax-H3` and
that FFmpeg and FFprobe are available on `PATH`.

```sh
make -j8
mkdir -p outputs
./h3 --info -d ./MiniMax-H3
```

`--info` checks the model layout and prints the selected Metal device without
mapping all weights or generating media. Run `./h3 --help` for the complete CLI
reference.

Without `-p`, the same binary starts an Iris-style interactive session:

```sh
./h3 -d ./MiniMax-H3 --width 512 --height 512 --steps 6
```

Type a prompt to generate a numbered video. The session keeps the exact BF16
prompt conditioning, prepared DiT, and video decoder in memory, so repeating a
prompt with another seed avoids loading and encoding them again. Useful commands
are `!status`, `!seed random`, `!seconds 2`, `!show`, `!save output.mp4`, and
`!cache`. Use `!help` for the full, short list.

First/last-frame conditioning is persistent in the session:

```text
h3> !first opening.png
h3> !last ending.png
h3> The camera moves slowly around the subject.
```

Use `!first clear` or `!last clear` to remove an anchor. Generated videos are
written to the session directory printed at startup.

For a general Ref2VA conditioning image, use `!ref-image PATH` instead. Images
are appended in order and exposed to the model as `<Picture 1>`, `<Picture 2>`,
and so on; filenames have no meaning to the model.

```text
h3> !ref-image person.png
h3> Make the person shown in Picture 1 wave to the camera.
```

`!refs` lists the current order, `!ref-remove N` removes one entry, and
`!refs clear` removes them all. Ref2VA references cannot be mixed with
`!first`/`!last` anchors.

### 2. Make a first fast video

Start with the validated balanced preset. It generates 22 frames at 24 fps
(about 0.92 seconds), displays the evolving middle-video frame after every
denoising transition in a supported graphical terminal, and prints phase
timings:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --show \
  -o outputs/fox-fast.mp4
```

This is deliberately not the most aggressive configuration:

- `--steps 20` performs the default 20 denoising passes.
- `--reuse 2` computes 11 fresh denoiser velocities instead of all 20 and
  extrapolates the skipped transitions.
- `--layers 45` runs 45 of the 50 transformer blocks, reducing both time and
  unified-memory use.
- `--show` is optional. It supports Kitty/Ghostty and
  iTerm2/WezTerm/Konsole graphical protocols. It loads a resident preview VAE,
  displays one representative middle-video frame after every Euler transition,
  and then displays all final frames. Display dimensions default to 2x so the
  image has its intended logical size on macOS Retina screens; use `--zoom 1`
  on a non-HiDPI display. This adds preview decode time and roughly 10 GiB of
  temporary model residency; runs without `--show` are unchanged.
- `--profile` is optional and does not select a different generation path.

The first process invocation also pays model loading and filesystem-cache
costs. Compare performance using repeated runs, and alternate variants when
the machines are warming up because this workload is sensitive to thermal
throttling.

For a very short iteration, request four denoising passes directly:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur." \
  --width 512 --height 512 --frames 22 \
  --steps 4 --layers 50 --reuse 1 \
  --show \
  -o outputs/fox-four-step.mp4
```

`--steps N` always means exactly N denoising passes. Four through seven passes
use the same schedule that won the low-budget comparison; increasing from 4
to 7 progressively improves detail and motion. Keep `--reuse 1` at such small
budgets so every requested pass runs the model. `--show` displays one preview
after each pass.

Several tail-heavy schedules were evaluated because most visible cleanup
happens late in a long run. They preserved too few early composition updates
and produced woven texture, weak motion, or clipped colors. The retained mode
uses the released linear base grid with one terminal point. On the 512-square,
22-frame fox test, the selected four-pass result had 0.556 full-video SSIM
against a 29-pass reference; an independent surfer test measured 0.547. The
four-pass denoise took about 3.5 seconds on M5 Max, versus 26.4 seconds for the
reference.

For a low-memory run, add `--ssd-streaming`:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 --ssd-streaming \
  -o outputs/fox-ssd.mp4
```

This uses the original BF16 checkpoint without conversion or quantization. It
keeps two DiT blocks in memory and reads the next block from SSD while the GPU
runs the current one. On M5 Max, tracked DiT storage fell from about 36.5 GiB to
2.0 GiB at 512 square and 2.1 GiB at 864x480. A warm 50-block forward measured
1.35 versus 2.49 seconds at 512 square (84% slower), and 2.14 versus 2.68
seconds at 864x480 (26% slower). These are comparisons against the same
full-residency BF16 path, and the results were byte-identical in both checks.

The 2.0--2.1 GiB figure is the DiT's tracked tensor storage, not total system
RAM. Prompt encoding and the two VAEs run in separate phases rather than adding
their full peaks to it; the OS, media buffers, and output resolution still need
headroom. `--show` keeps a preview VAE resident and adds roughly 10 GiB, so omit
it for the lowest-memory run.

SSD streaming is an explicit memory/speed tradeoff and is not the default. It
cannot be combined with `--use-int8-row-fc2`. In an interactive session, use
`!ssd-streaming on`.

### 3. Move toward reference quality

Change one control at a time when evaluating quality. First restore all layers,
then all denoiser evaluations, and finally raise the default 20-pass schedule
to the slower 50-pass reference:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 50 \
  --layers 50 --reuse 1 \
  -o outputs/fox-close.mp4
```

The defaults are `--steps 20 --layers 50 --reuse 1`; keep `--steps 50`
explicit for this close path. It performs 50 complete 50-block denoiser
forwards and is much more expensive than the default, but is the right oracle
when a fast mode changes the subject, anatomy, motion, or composition.
Numerical pixel identity with MLX is not expected because the random-number and
execution engines differ; the depicted content and motion should agree.

### 4. Choose a speed/quality preset

These controls are independent unless noted otherwise:

| Control | Slow reference | Default | Aggressive | Main impact |
|---|---:|---:|---:|---|
| Denoising passes | `--steps 50` | `--steps 20` | `--steps 4..7` | The number always names actual denoising passes. |
| Whole denoiser reuse | `--reuse 1` | `--reuse 2` | `--reuse 3` | At 20 steps: 20, 11, or 8 fresh DiT evaluations. |
| Active DiT blocks | `--layers 50` | `--layers 45` | `--layers 40` | Fewer blocks reduce compute and resident transformer weights. |
| Core residual reuse | `--core-reuse 1` | `--core-reuse 4` | `--core-reuse 6` | Refreshes patch/head work every step but runs the expensive core less often. |
| Token reduction | off | optional | `--token-reduction` | Pairs horizontal video tokens inside middle blocks; faster but may change composition. |
| Internal canvas | output size | `384x384` for 512 square output | `320x320` | Runs DiT/VAE smaller, then upscales with vImage. |

On M5, `--use-int8-row-fc2` uses one activation scale per FC2 row and a single
full-width TensorOps product. It is optional because it is less numerically
conservative than grouped int8. It reduced complete denoiser forwards by about
2.6% in reciprocal tests. Matched four-step fox and surfer videos kept the same
subjects, setting, and motion (full-video SSIM 0.919 and 0.828). In the
interactive session, use `!int8-row-fc2 on`.

`--reuse` and `--core-reuse` are mutually exclusive. Layer thinning can be
combined with either one.

To make the first command faster while keeping its output resolution, add
token reduction:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A surfer riding inside a sharp blue ocean wave, one rider and one white board, realistic spray." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --token-reduction \
  -o outputs/surfer-fast.mp4
```

At the validated 512 square shape, token reduction cut the `45 layers + reuse
2` denoise profile from 16.69 to 12.60 seconds on the IT M5 Max. Independent
fox and surfer renders stayed coherent, but composition can diverge more from
the close path.

For an aggressive preview, render internally at 320 square and upscale to the
requested 512 square output:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walking through snow, realistic, tracking shot." \
  --width 512 --height 512 \
  --render-width 320 --render-height 320 \
  --frames 22 --steps 20 --layers 40 --reuse 3 \
  -o outputs/fox-aggressive.mp4
```

This combination produced a clean, recognizable 22-frame fox in validation,
but loses fine detail and can change framing. Do **not** add `--token-reduction`
to both `--layers 40` and `--reuse 3`: that tested combination produced color
ringing, outlines, and ghosted limbs.

As an alternative to whole-velocity reuse, this keeps the timestep-dependent
patch and output heads fresh at every transition:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A surfer riding a blue ocean wave." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --core-reuse 4 \
  -o outputs/surfer-core-reuse.mp4
```

Use `--core-reuse 6` only as an aggressive preview. Values above 6 are not
exposed because validation lost subject fidelity.

### 5. Pick resolution and duration

Width and height must each be multiples of 32, at least 32, and their product
must not exceed `768 * 1344` pixels. Those are mechanical limits, not a promise
that every tiny canvas has good model quality. H3-Base is a 768p model.

| Canvas | Current guidance |
|---|---|
| `512x512` | Safest development size; repeatedly validated with multiple prompts. |
| `768x768` | Validated close-quality square output; substantially more expensive. |
| `1344x768`, `768x1344` | Released 768p-class landscape/portrait limit. |
| `1024x768`, `768x1024` | Valid 4:3 and 3:4 768p-class canvases. |
| `384x384` internal to `512x512` | Validated fast-quality scaling point. |
| `320x320` internal to `512x512` | Validated aggressive scaling point. |
| `256x256` | Native fast-preview canvas with automatic low-resolution RoPE adaptation. |

For a fast native 256-square preview:

```sh
./h3 -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 256 --height 256 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 \
  -o outputs/fox-256.mp4
```

At 256 square, H3 has only an `8x8` effective spatial-token grid, so it has less
room for fine detail and complex composition. H3 automatically halves spatial
RoPE coordinates at exactly 256 square. This removed repeating lattice
artifacts in long fox renders and stayed coherent on an independent portrait,
without adding tokens or runtime. Use `--use-reference-rope` to restore the
released/MLX coordinates for parity checks. Keep token reduction off at this
size. Native 128 square remains unsupported: its `4x4` token grid did not
recover a recognizable subject even with adjusted RoPE.

`--render-width` and `--render-height` must be set together, must have the same
aspect ratio as the output, and cannot exceed the output dimensions. The model
and VAE use the internal size; terminal frames and the encoded video retain the
requested output size.

H3 emits 24 fps and aligns frame requests upward to `5 + 17*n`:

Use `--seconds N` for a duration-oriented request, or `--frames N` for direct
frame control; the two options are mutually exclusive. Fractional seconds are
accepted. Seconds are converted at 24 fps and then rounded upward to the next
legal H3 temporal shape, so `--seconds 10` produces 243 frames (10.125 seconds).

| Frames | Approximate video duration |
|---:|---:|
| 22 | 0.917 seconds |
| 39 | 1.625 seconds |
| 56 | 2.333 seconds |
| 107 | 4.458 seconds |
| 243 | 10.125 seconds |
| 362 | 15.083 seconds |

Short clips are useful for development. The released workflow is intended for
roughly 4–15 second videos. A request such as `--frames 23` is rounded up to 39
frames rather than producing an arbitrary temporal shape.

### 6. Improve the prompt

A short prompt works, but the released system expects a Context-IR-like
description. State the subject, action, setting, camera, lighting/style, and
desired sound. For example:

```text
Scene: a single red fox in a snow-covered pine forest at dawn.
Action: the fox walks steadily left to right and looks toward the camera once.
Camera: medium-height lateral tracking shot, 50 mm lens, stable framing.
Look: photorealistic fur, cold blue ambient light, warm sunrise rim light.
Audio: soft footsteps in snow, light wind through pine branches, no music.
```

Keep identity and object counts explicit when they matter. `--seed N` controls
the native random stream; the default is 42. Compare options with the same
prompt, seed, resolution, frame count, and step count.

### 7. Preview frames and diagnose performance

- `--show` displays a representative frame after every denoising transition,
  followed by all frames from the completed video. Like Iris, it advertises 2x
  display dimensions by default for Retina terminals; `--zoom N` changes that
  factor without resizing the generated video or the encoded terminal image.
- `--frames-dir DIR` writes final callback frames as PPM files. Intermediate
  `--show` previews are not written there.
- `-o ''` disables MP4 encoding; combine it with `--frames-dir` when FFmpeg is
  unavailable.
- `--profile` reports phase wall time, Metal encoding/wait time, peak live
  tensor storage, cumulative allocation, and dispatch counts.

For example:

```sh
./h3 --profile -d ./MiniMax-H3 -p "A hummingbird hovering over red flowers." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --frames-dir outputs/hummingbird-frames \
  -o ''
```

### 8. Add image, video, and audio references

First/last-frame anchors select the FL2VA path:

```sh
./h3 -d ./MiniMax-H3 -p "The fox keeps walking through the snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --first-frame fox.png --last-frame fox-later.png \
  -o outputs/fox-anchored.mp4
```

Ordered references select the distinct Ref2VA checkpoint. Use the flag matching
the media semantics:

```sh
# One image reference.
./h3 -d ./MiniMax-H3 -p "Use the animal and setting in the reference." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-image fox.png -o outputs/fox-reference.mp4

# Continue a clip but ignore its soundtrack.
./h3 -d ./MiniMax-H3 -p "Continue the motion in this clip." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-silent-video fox.mp4 -o outputs/fox-video-reference.mp4

# Preserve the clip's embedded audio.
./h3 -d ./MiniMax-H3 -p "Continue this audiovisual scene." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video fox-with-audio.mp4 -o outputs/fox-video-audio.mp4

# Replace a video's soundtrack explicitly.
./h3 -d ./MiniMax-H3 -p "Continue the scene with the supplied music." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video-audio silent-fox.mp4 replacement.wav \
  -o outputs/fox-replaced-audio.mp4

# An ordered image plus standalone audio reference.
./h3 -d ./MiniMax-H3 -p "Use the animal and music from the references." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-image fox.png --ref-audio music.wav \
  -o outputs/fox-image-audio.mp4
```

Reference flags may be repeated and their command-line order is preserved.
Standalone audio must accompany an image or video reference. Audio references
must be 2–15 seconds; at most three audio inputs are accepted and their total
decoded duration is capped at 15 seconds.

## Tests and runtime requirements

```sh
make test
make parity
```

`make test` runs the deterministic host suite and, when the ignored MLX fixture
is installed under `misc/fixtures/`, compiles the Metal source at runtime and
checks a complete toy H3 block against named MLX outputs. Runtime compilation is
intentional: it follows Iris and does not require Xcode's optional offline Metal
toolchain. The test covers both an F32 diagnosis path and the production BF16
storage path; wide BF16 matrix products and SDPA use cached MPSGraph graphs, with
direct Metal correctness fallbacks. `make parity` runs only those Metal/MLX
checks.

FFmpeg and FFprobe must be available on `PATH` for media inputs and MP4 output
(`H3_FFMPEG` and `H3_FFPROBE` may select explicit executables). Generated RGB24 and
32 kHz stereo F32 PCM are fed through concurrent pipes; no intermediate
uncompressed media file is created.

## Implementation and performance notes

The remainder documents the implementation behind the tutorial presets and the
environment variables retained for exact A/B diagnosis.

### Sampler and DiT controls

The default sampler uses the released shifted video/audio schedule. `--steps`
always names the number of denoising passes, with terminal zero added after the
last pass. Whole-denoiser reuse evaluates the first and last pass plus every
requested interval, then extrapolates skipped video and audio velocities on
their independent schedules. With very small step counts, keep `--reuse 1`.

For the low-budget path, the released linear base grid won against
actual-video-sigma linear spacing,
quadratic and cubic warps, exact 30-point tail subsets, mild power warps,
zero-order held full-grid velocities, linear velocity extrapolation, and RES.
The more tail-heavy candidates often sharpened the subject but damaged motion
or left a repetitive woven background; sparse RES and long extrapolation
intervals failed much more visibly.

Layer thinning ranks the checkpoint's actual AdaLN gates while protecting
structurally important first and final blocks. Unused weights and schedule
tensors are not retained, so `--layers 45` and `--layers 40` reduce both
transformer time and unified-memory use. Core reuse holds the previous full
transformer residual while refreshing the patch projection and timestep-aware
head; it remains mutually exclusive with whole-velocity reuse.

### Exact DiT fusions

Every active DiT block fuses its attention residual gate with the following MLP
AdaLN. The rounded BF16 residual is still written exactly, but the same row is
kept in threadgroup memory for normalization, eliminating one dispatch and one
global reread. Away from token-reduction boundaries, the MLP residual gate also
produces the next block's attention AdaLN and carries that normalized state
across the loop. `H3_DISABLE_FUSED_GATE_ADALN=1` and
`H3_DISABLE_FUSED_CROSS_BLOCK_ADALN=1` restore the two-kernel oracles.
The final audio/video AdaLN kernels bind directly to offsets in the residual
stream, avoiding two slice blits and 18.8 MiB of scratch at 512x512 (29.4 MiB
at the 864-class benchmark shape).
`H3_DISABLE_FUSED_FINAL_SLICE=1` restores the copy-plus-AdaLN oracle at load.
The BF16 final heads then apply AdaLN while loading their 16x16 projection
tiles, preserving the standalone rounding and accumulation order while
removing another equally sized normalized activation. The two optimizations
together save 37.5/58.9 MiB. `H3_DISABLE_FUSED_FINAL_HEAD=1` restores the
offset-AdaLN-plus-linear oracle at load.

### Token-reduction internals

`--token-reduction` is an independent aggressive DiT mode. After block 3 it
pairs adjacent horizontal target-video tokens while leaving text, audio,
conditions, and reference tokens exact. The complete full-resolution state is
kept as a bypass. During the first ten noisy evaluations it restores before
block 40; subsequent detail-forming evaluations restore before block 30. Each
token returns as its original value plus the update learned by its pair, so
within-pair detail is not discarded.
The pooling kernel writes only true-pair baselines into a dense tail of the
already allocated attention scratch buffer; odd-width singleton tokens need no
baseline. The full bypass uses the oversized QKV tail when it fits, with a
guarded dedicated fallback only for reference-heavy layouts. Common text-only
canvases therefore add no activation arena at any token-grid width. Pooling
also snapshots both source tokens while their BF16 values are already in
registers, avoiding a separate full-hidden blit and redundant source read. The
same entry kernel keeps each pooled row in threadgroup memory and emits the
first reduced block's attention AdaLN, eliminating another global residual read.
At the restore boundary, the first full-resolution attention AdaLN is fused
into expansion: a 10.5 KiB threadgroup row avoids a global residual reread while
still writing the exact bypass needed by the following residual branch.
On a thermal-balanced 512x512x22, 19-forward IT M5 Max A/B this reduced denoise
time from 39.13 to 28.06 seconds (28.3%). Final video/audio latent relative L2
was 5.56%/15.14%. First/middle/last fox frames retained one clean muzzle,
coherent legs, and sharp fur; an independent surfer remained consistent with
one rider and board through the wave spray. It changes composition and is
therefore opt-in rather than the close-reference default.
`H3_TOKEN_REDUCTION_BLOCKS` can override the later `4:30` interval;
`H3_TOKEN_REDUCTION_EARLY=STEPS:END` overrides the early schedule and `0`
disables it. `H3_DISABLE_TOKEN_REDUCTION=1` provides an in-context exact oracle.
`H3_DISABLE_FUSED_TOKEN_POOL_ADALN=1` and
`H3_DISABLE_FUSED_TOKEN_ADALN=1` independently restore the two-kernel entry and
exit boundaries for diagnosis.
Token reduction composes cleanly with the validated `--layers 45 --reuse 2`
settings: on the same 512 benchmark it reduced that profile from 16.69 to
12.60 seconds (24.5% marginal), and independent fox and surfer renders stayed
coherent. Do not combine it with both `--layers 40` and `--reuse 3`; that
6.47-second experiment produced chromatic ringing and ghosted limbs despite
acceptable latent norms.

### Internal canvas and video VAE

`--render-width` and `--render-height` run the model and VAE on a lower
same-aspect internal canvas, then high-quality vImage-scale RGB frames to the
requested output size before callbacks, terminal display, and encoding. This is an
explicit quality/speed tradeoff: a measured 384-to-512 prompt render reduced
M5 DiT time by 33% and video-VAE time by 18% while retaining a clean,
recognizable photorealistic result. Both values must be multiples of 32; the
exact output canvas remains the default.
For square 512 output, 384 is the fast-quality point and 320 is the validated
aggressive point. The latter produced a coherent walking fox and repeated at
8.02 seconds of DiT versus about 15.82 seconds natively. Native 256 uses the
same-cost spatial-RoPE adaptation described above; it remains a fast composition
preview rather than a substitute for a 512- or 768-class final render.
The video VAE automatically chooses a 256-320 pixel spatial tile from the
requested canvas geometry, minimizing repeated overlap work while keeping peak
storage bounded. `H3_VAE_TILE_PIXELS=256` restores the original conservative
tile plan for close-reference diagnosis.

### Weight residency and streamed prompt encoding

On M5-class GPUs, persistent transformer weights are mapped directly from their
safetensor shards instead of copied into anonymous shared buffers. This keeps
the 37 GiB model file-backed/reclaimable and slightly improves total transformer
time; M3 uses the faster copied-buffer path. `H3_ZERO_COPY_WEIGHTS=0` disables
the M5 selection for diagnostics.
The streamed Qwen text encoder preallocates a small ring of future layer
buffers and fills them on eight I/O workers while Metal executes the current
layer. The default ring depth is two layers on M3/older hardware and three on
M5, where the target machine has 128 GiB. `H3_QWEN_PREFETCH=0` restores the
single-layer synchronous reference path; values 1-8 select the worker count,
and `H3_QWEN_PREFETCH_DEPTH=1` through `6` overrides the ring depth.

`--ssd-streaming` is a separate, more aggressive residency mode for the DiT.
Only its small per-block normalization weights remain resident. Two complete
BF16 matrix slots alternate while a background reader fills the next slot in
checkpoint-offset order; the current Metal command buffer runs concurrently.
Darwin uncached reads avoid retaining a second copy in the filesystem cache.
The first active block is prefetched again during the final block, so a cached
interactive DiT is ready for its next denoiser evaluation. Measurements reached
about 13--14.6 GiB/s from the internal SSD. `H3_PROFILE=1` reports total bytes,
read throughput, and the part of the read wait that was not hidden by GPU work.

### Metal 4 and TensorOps paths

M5 GPUs automatically use native BF16 Metal 4/TensorOps for the DiT QKV and
attention-output projections at sequence lengths up to 2,048. The compact
Morton schedule routes Q/K/V directly into head-major attention inputs, avoids
three MPSGraph input transposes, and is byte-identical to the portable path. It
improves a complete 512x512 50-block forward by about 2% across repeated IT/US
M5 Max runs. For 2,049-3,072 rows, including 864x480, two row-offset Morton
dispatches preserve the efficient tile geometry and improve the complete
forward by about 2% in balanced runs. Still larger sequences stay on MPSGraph.
`H3_NAX=0` disables TensorOps for exact A/B diagnosis. The selection is guarded
at runtime and falls back to the unchanged portable library if compilation is
unavailable.

`H3_NAX=1` forces the broader native BF16 linear path. It passes the complete
50-block MLX fixture, but remains opt-in: exact-shape microbenchmarks favor its
128-row tile while full DiT runs currently favor MPSGraph scheduling. This
keeps a working NAX integration available for later quantized/fused kernels
without making a benchmark regression the default.
`H3_NAX=mlp` selects a more specialized Metal 4 path: paired FC1 gate/up
TensorOps tiles apply SwiGLU in threadgroup memory and write only the
14,336-wide activated intermediate, then FC2 also stays on TensorOps.
`H3_DISABLE_NAX_MLP=1` keeps the MPSGraph MLP in a context created this way for
same-process A/B testing. The path is deliberately opt-in because scheduling
depends on the OS GPU stack: the primary macOS 26.5.2 M5 Max gained 1.3-2.0%
in isolated real-weight MLP runs but lost about 1-3% in a complete 50-block forward,
while an otherwise identical macOS 26.5 M5 Max gained 1.4% in a same-context
forward A/B. The resulting 50-block velocities were close (1.9% video and 2.4%
audio relative L2), but not byte-identical.

### Specialized projection kernels

The narrow DiT audio/video output heads convert their small released F32
weights to BF16 once and use the Iris-derived 16x16 tiled linear directly on
BF16 activations. At the production 320-render geometry, isolated paired-head
measurements are 2.30x faster on M3 Max and 1.83x faster on M5 Max, with
relative L2 `8.64e-4`; the absolute M5 saving is about 0.6 ms per evaluated
step. Full fox and surfer sequences remained clean and measured 29.9/38.4 dB
against the F32-head renders. `H3_DIT_F32_FINAL=1` restores the close-reference
head and its extra activation buffers.
The F32 `96->5376` video and `32->5376` audio patch projections use a dedicated
16x16 cooperative tile, retaining F32 weights, inputs and accumulation while
rounding the tile result directly to BF16.
Paired production-shape measurements are 1.77x faster on M3 and 1.62-1.78x
on M5; the complete generated RGB stream is byte-identical to the scalar path.
Fusing the final cast improves the 2835-row tile itself from 2.499 to 1.734 ms
on M3 and 1.555 to 1.186 ms on M5, and removes 38.27/59.66 MiB of F32 scratch
at 512/864-class geometry. `H3_DISABLE_FUSED_PATCH_CAST=1` restores the tiled
F32 output plus standalone cast; `H3_SCALAR_PATCH=1` selects the scalar
diagnostic path.
The same tile binds its output directly into the packed hidden stream, removing
the BF16 media staging buffers and their blits. This saves another 19.13/29.83
MiB and improves the 2835-row boundary from 1.847 to 1.730 ms on M3 and 1.282
to 1.184 ms on M5. Contiguous T2VA uses byte offsets; FL2VA/Ref2VA use compact
destination-row maps so each modality remains one large dispatch. A complete
six-segment Ref2VA M5 ABBA remained byte-identical and improved 5.067 to 5.033
seconds per measured forward pair. `H3_DISABLE_FUSED_PATCH_PACK=1` restores the
staging buffers and packing blits.

### Scheduling and activation memory

The DiT core is split into two ordered Metal command buffers so GPU execution
of the first part overlaps CPU encoding of the second. Thermal-balanced ABBA
measurements select a 60%-depth split on M5 (30/50, 27/45, and 24/40), with
roughly 0.5-1.8% wins; M3 automatically splits only the validated 30/50 case,
which measured 1.2% faster, because 24/40 regressed there. The operation order
and generated bytes are unchanged. `H3_DIT_COMMAND_BLOCKS=0` restores one
command buffer; values 1-50 override the split for further tuning.
DiT activation buffers also follow their actual intra-block lifetimes: the QKV
projection arena is reused first for attention heads and then for the normalized
MLP input, while the current attention-output arena becomes the MLP output after
its branch has been consumed. This removes 61.25 MiB at 512-class geometry and
99.63 MiB at 864-class geometry without changing dispatches or arithmetic.
`H3_DISABLE_DIT_ACTIVATION_ALIAS=1` restores separate diagnostic buffers.
MPSGraph tensor-data wrappers for immutable DiT weights and biases are retained
with their resident buffers. This avoids rebuilding the same binding metadata
for every block and denoiser evaluation without copying tensor storage; measured
ABBA gains were 1.6% on M3 Max and 0.4-1.1% on M5 Max. Activation wrappers stay
transient because retaining them regressed the M5. The outputs remain
byte-identical, and `H3_DISABLE_GRAPH_DATA_CACHE=1` restores transient wrappers
for all tensors.
On M3/older hardware, the four MPSGraph segments in each DiT block also reuse
one `MPSCommandBuffer` wrapper for their shared underlying Metal command buffer.
Repeated thermal-balanced runs measured 1.0-1.6% faster on M3 Max; M5 measured
neutral, so it retains fresh wrappers. `H3_REUSE_MPS_COMMAND=0` or `1` overrides
the automatic selection. Results are byte-identical.
On M5, the serving Euler sampler keeps its patch-packed F32 latents and cached
BF16 velocities in Metal buffers. Each selected denoiser refresh is completed
before the next is encoded, avoiding MPSGraph back-pressure while removing all
intermediate latent/velocity readbacks and repacking. Two warm eight-run A/B
sequences measured small 0.1% and 0.3% gains with byte-identical final latents;
the path also saves roughly 16 bytes of transient host state per video-latent
element (about 136 MB at the 768p shape). M3 and older GPUs retain the CPU
sampler by default. `H3_CPU_SAMPLER=1` restores it on M5;
`H3_GPU_SAMPLER=1` selects the GPU-state path explicitly, and
`H3_GPU_SAMPLER_WINDOW=0` enables the slower unbounded encode-ahead diagnostic.

### Checkpoint layout and media pipeline

The released checkpoint stores DiT QKV rows interleaved per attention head.
Native Metal consumes that layout directly in the fused QK-normalization/RoPE
kernel, avoiding a checkpoint transpose and extra RAM. The earlier identity
interpretation was the cause of the noisy diagnostic outputs.

The public generation path decodes the joint audio latent with a streamed native
BigVGAN/AudioVAE and writes synchronized H.264 plus 32 kHz stereo AAC. The native
waveform agrees with the corrected MLX oracle to relative L2 `6.94e-5`.
`--first-frame`, `--last-frame`, and their combination use the released visual
VAE encoder, Qwen3-VL vision tower and three-deepstack multimodal presentation,
0.999 condition augmentation, and fixed condition rows in the native DiT. The
first image is stretched to the target canvas; the last image is aspect-cover
scaled and center cropped, matching the reference implementation. `--ref-image`
selects the distinct Ref2VA transformer, preserves ordered `<Picture N>`
presentation, and uses the released down-only aspect-preserving reference canvas.
`--ref-silent-video` additionally performs bounded 24 fps decoding, the visual
VAE's causal `ceil(T/4)` compression, two-frame Qwen sampling, and timestamped
`<Video N>` presentation. `--ref-video` preserves an embedded soundtrack,
`--ref-video-audio VIDEO AUDIO` supplies an explicit replacement, and
`--ref-audio` appends an ordered standalone clip. Reference audio is decoded as
32 kHz stereo F32, encoded by the native AudioVAE posterior-mean path, mixed as
0.999 clean latent plus 0.001 seeded noise, pinned to the audio condition
timestep 1.0, and packed as width-32 rows on the same rotary timeline as visual
references. Audio inputs are 2-15 seconds, at most three are
accepted, their total decoded duration is capped at 15 seconds, and a standalone
audio reference must be combined with an image or video reference.

The native audio encoder matches the corrected MLX oracle at relative L2
`3.59e-6` on a real two-second stereo fixture. The correction is important: the
original MLX reshape interleaved left/right samples, whereas the official
PyTorch/SGLang path folds intact stereo channels into the batch dimension. On
the 128 GB M5 Max, clean end-to-end image+audio and embedded-video+audio renders
completed in 74.58 and 76.99 seconds respectively, each with about a 40.1 GB
peak physical footprint and zero swaps.

### Profiling and diagnostic paths

`--profile` reports each Metal-backed phase separately: wall time, CPU-side
command encoding, complete commit-to-fence wait, root-command GPU timestamps,
peak live tensor storage, cumulative allocation, and dispatch counts. The wait
measurement is the complete command turnaround; the root GPU timestamp alone
can omit child buffers scheduled internally by MPSGraph and is labeled
accordingly.

The DiT fast path evaluates each BF16 `fc1 -> SwiGLU -> fc2` block as one cached
graph, avoiding separate graph boundaries and persistent intermediate tensors.
Set `H3_DISABLE_FUSED_MLP=1` to retain the close-reference operation boundaries
for numerical diagnosis.

On supported M5 Metal 4 TensorOps hardware, the native int8 MLP engine is the
default. It dynamically quantizes activations, uses per-output-channel weight
scales, and gives the sensitive FC2 input one scale per 1,024 channels.
The selected FC2 kernel keeps scaled partial products in private cooperative
fragments instead of repeatedly spilling a 32 KiB threadgroup tile. A fixed
50-layer, 19-transition 512x512 render measured 36.30 seconds with BF16 MPS and
25.80 seconds with int8 on M5 Max. Beginning, middle, and final decoded frames
retained the same subject, composition, and motion; small edge and fur details
can differ. The current diagnostic implementation retains both BF16 and int8
MLP weights only when an A/B diagnostic requests them. Normal int8 loading
releases each block's BF16 FC1/FC2 buffers after their submitted quantization
finishes, reducing measured peak tensor storage to 25.9 GiB from the BF16
path's 36.4 GiB. Runtime weight quantization still adds startup time.

The fastest M5 path also quantizes each DiT QKV projection and writes its
Q/K/V tiles directly in head-major attention layout before the existing Q/K
normalization and RoPE kernel. In a fixed 50-layer, 19-transition 512x512
render this reduced denoising again, from 25.80 to 19.32 seconds. Sampled
beginning, middle, and final frames remained a coherent detailed fox walking
through snow; quantized attention can change framing and fine detail. Use
`--use-slower-bf16-qkv` for the close-reference BF16 projection. Normal int8
loading releases the redundant BF16 QKV weights after quantization.

The following attention-output projection is int8 as well on the default M5
path. Crossed same-model tests improve a complete forward by another 4.5-5.5%
at 512 and 864. A decoded fox render remained clean and closely matched the
int8-QKV-only composition; its thermally hot denoise measured 19.18 seconds.
Use `--use-slower-bf16-attention-output` to retain that projection in BF16.

On that int8 path, SDPA now leaves its result in native
`[head,row,dimension]` order. A specialized 256-thread kernel gathers and
quantizes each H3 row directly into the projection's row-major int8 buffer,
eliminating the intervening full-width BF16 transpose without changing any
output byte. Thermally controlled crossed runs improve complete 512 and 864
forwards by roughly 0.2-1.2%. Use
`--use-slower-row-major-attention-output` to restore the explicit BF16
row-major SDPA output and ordinary quantizer.

The M5 path also folds QKV and MLP activation quantization into the preceding
gated AdaLN kernel. This removes 99 standalone quantizer dispatches per
50-layer forward while preserving the previous output bytes, improving crossed
512/864 measurements by about 0.3-0.6%. Use
`--use-slower-unfused-int8-inputs` to restore the standalone quantizers.

The fused gated-AdaLN path loads its full 5,376-wide H3 rows as BF16x4 vectors
and writes int8x4. It stages the rounded values locally before computing the
original per-thread RMS sequence, so the reduction tree and every output byte
remain unchanged. Crossed measurements save roughly another 0.1-0.5%. The
existing `--use-slower-unfused-int8-inputs` option retains the portable scalar
and standalone-quantizer fallback.

Q/K RMS normalization and RoPE are performed inside the int8 QKV projection
tile as well. The fused epilogue is byte-identical and improves complete
forwards by 2.1-3.2% at 512 and 1.0-1.8% at 864 in crossed M5 measurements.
Use `--use-slower-unfused-qkv-rope` to restore the separate Q/K kernel.

That epilogue processes four adjacent Q/K dimensions per work item with
BF16x4 loads and stores. The per-element arithmetic and BF16 rounding order are
unchanged, while crossed cool-state measurements improve complete forwards by
about 0.4-1.0% at both 512 and 864. The same
`--use-slower-unfused-qkv-rope` option restores the scalar standalone path.

At up to 2,048 rows, the exact RMS loop uses BF16x4 loads followed by four
explicit ordered FMAs. This preserves every output bit and improves 512-class
forwards by another 0.5-0.6%; larger shapes retain scalar loads because the two
forms tie there. Use `--use-slower-scalar-qkv-rms` to force scalar loads.

The int8 attention-output projection caches its 128 row and column scales in
1 KiB of threadgroup memory instead of rereading them for every cooperative
fragment element. Above 2,048 rows the fused QKV kernel uses the same idea and
then recycles that storage for inverse RMS values; smaller QKV shapes retain
direct loads because the two forms tie there. Both are byte-identical and
improve complete forwards by about 0.2-0.7% where selected. Use
`--use-slower-uncached-int8-scales` to restore direct device-scale loads.

For sequences of at most 2,048 rows, the H3 attention-output projection also
compiles its 7,168-by-5,376 shape into the TensorOps kernel. The result remains
byte-identical while saving about 0.2-0.8% in crossed complete 512-forward
measurements. Larger sequences retain the dynamic-shape kernel because the
specialization regresses there. `--use-slower-uncached-int8-scales` restores
the general dynamic, direct-scale-load implementation.

FC1 also uses an H3-specialized, compile-time 5,376-wide TensorOps loop. It is
byte-identical to the generic loop and saves about 0.1-0.4% in crossed complete
forwards. Use `--use-slower-dynamic-fc1-k` to restore the runtime-bound loop.

```sh
./h3 --profile -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 -o outputs/fox-int8.mp4
```

Use `--use-slower-bf16-mlp` to force the portable close-reference MPS/BF16 MLP
path for numerical comparison. Older Metal hardware selects that path
automatically when the required native TensorOps kernels are unavailable.
For FC2 activation quantization, sequences of at most 2,048 rows use an exact
128-thread reduction. Each thread retains its eight BF16 input values while
computing the group maximum, avoiding a second device-memory read when it emits
the int8 values; crossed M5 measurements improved complete 512 forwards by
about 0.2-0.8% without changing any output byte. Larger sequences retain the
measured 256-thread kernel. `--use-slower-grouped-quantizer` forces the latter
at every size for A/B comparison.

The native baseline targets the original `FL2VA/` and `Ref2VA/` checkpoint
trees. Model phases are loaded and released separately so the 33B transformer,
Qwen encoder, and decoders never have to coexist in unified memory.


---

## FILE: THIRD_PARTY_NOTICES.md (https://github.com/antirez/h3.c/blob/HEAD/THIRD_PARTY_NOTICES.md)

# Third-party notices

The rectangular Morton decoder and the dynamic symmetric int8 quantization /
Metal 4 TensorOps scheduling design in `h3_shaders.metal` are adapted from
ccv's Metal FlashAttention `NAMatMulKernel` and `NAInt8MatMulKernel`,
distributed under the following BSD-3-Clause license:

Copyright (c) 2010, Liu Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the authors nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


---

# ISSUES / PRs of antirez/h3.c


### #100 [open] PR: Run the MPSGraph BF16 linear and SDPA graphs with FP16 operands when they are faster
URL: https://github.com/antirez/h3.c/pull/100  by FengGuanyun 2026-11-19T19:19:03Z

## Summary

On Apple Silicon GPUs that do not implement the BF16 matrix kernels in
hardware, MPSGraph runs the BF16 matmul and SDPA on an emulated path that is
much slower than the same graph with FP16 operands. This PR measures both
variants once per device, when the Metal context is created, and keeps the
faster one for the graphs whose operands fit in FP16's range. The cast lives
inside the MPSGraph, so there is no extra buffer, no host copy and no
duplicated weight set: the placeholders keep their BF16 type and the graph
casts them to FP16 before the matrix product.

Measured on an M2 Ultra (macOS 26, Metal 4 capable, no TensorOps), at the
DiT's real shapes:

| MPSGraph op | BF16 | FP16 operands | gain |
|---|---:|---:|---:|
| linear 2835x5376x21504 (QKV) | 12.59 TF/s | 21.87 TF/s | 1.74x |
| linear 2835x7168x5376 (attention output) | 12.20 TF/s | 21.11 TF/s | 1.73x |
| SDPA 2835 rows x 56 heads x 128 | 3.81 TF/s | 13.09 TF/s | 3.44x |
| fused MLP fc1 / fc2 (measured, not used) | 12.60 / 11.87 | 21.95 / 20.69 | 1.74x |

For reference, MPSGraph FP32 matmul measures 20.8 TF/s on the same machine,
which is the FP32 SIMD peak: neither BF16 nor FP16 reaches a tensor core path
here, and BF16 is the slowest of the three.

These graphs are the DiT hot path on every machine that is not an M5. The
int8 MLP, int8 QKV, int8 attention output and NAX MLP branches in `h3_dit.c`
are all gated on `h3_gpu_has_int8_mlp()` / `h3_gpu_has_nax_mlp()`, which return
`gpu.tensorOpsEnabled`, which `h3_gpu_create()` only sets when the device name
contains "M5". So on M1 through M4 the DiT falls through to
`h3_gpu_linear_bf16()` for QKV and the attention output, `h3_gpu_sdpa_bf16()`
for the attention itself and `h3_gpu_mlp_bf16()` for the MLP, which are exactly
the graph builders touched here.

The decision is taken by calibration, not by a hardware table: `h3_gpu_create`
encodes one linear graph and one SDPA graph in both variants, times them, and
keeps the faster one with a 5% margin. A GPU whose BF16 kernels are native
keeps the current path unchanged, including M5, where the TensorOps/int8 paths
are selected and the calibration is skipped.

## Why the fused MLP is excluded

An earlier revision of this branch applied the operand selection to the fused
MLP as well, because its fc1 and fc2 measure the same 1.74x. That revision
rendered a black video at 512x512. The reason is FP16's exponent range, not
its precision: the SwiGLU intermediate is the product of two unbounded
projections, and on the released checkpoint at 512x512, 20 steps, it reaches
9.7e6, with about a third of the elements of several layers above FP16's
+-65504 limit (roughly 3e8 elements out of range in one render). Those become
infinity, fc2 propagates them, and the decode is black. BF16's +-3.4e38 range
has no such cliff.

The linear and SDPA graphs do not have this problem, and this was measured
rather than assumed: reading back the real linear outputs of the same render
gives magnitudes below 128 (QKV and attention output), three orders of
magnitude inside the FP16 range, because their inputs are RMS-normalized
activations and attention outputs. The SDPA operands are the same normalized
quantities.

So the MLP keeps BF16 operands unconditionally, and `h3_mps_operands_test`
pins that: it builds one BF16 context and one FP16 context and requires the
fused MLP to produce bit identical results in both, which fails the moment the
MLP starts following the operand selection. The cost is the fc1/fc2 speedup;
the QKV, attention output and SDPA gains remain.

## Behavioural change

On a machine where the calibration picks FP16, a render with the same prompt
and seed is no longer byte identical to the render from the previous build:
the linear results are bit identical, but the attention output differs by
about 1.7e-3 relative, the same order as the BF16 rounding it replaces. The
fused MLP is untouched, so its contribution is exactly as before. On M5 and on
any device whose BF16 kernels win the calibration, nothing changes at all.
`H3_FP16_OPERANDS=0` restores the previous behaviour everywhere.

Memory is unaffected: the cast lives inside the graph, so there are no extra
weight copies, and the measured peak live tensor storage is identical. The
calibration costs about 70 ms once per process, on the first Metal context
only.

## Accuracy

BF16 carries eight significand bits and FP16 carries eleven, so every BF16
operand of the linear and SDPA graphs converts exactly, and both graphs
accumulate in FP32.

`make test` runs `h3_mps_operands_test`, which builds one Metal context with
BF16 operands and one with FP16 operands, feeds both the same synthetic data,
and compares them against each other and against a double precision reference:

```
linear: 0 of 262144 results differ, max delta 0.000e+00
linear with bias: 0 of 262144 results differ, max delta 0.000e+00
attention: relative L2 difference 1.474e-03
attention vs double reference: BF16 1.715e-03, FP16 1.754e-03
mlp: 0 of 65536 results differ, max delta 0.000e+00
mlp vs double reference: BF16 3.313e-03
```

The biased linear case is there because a graph that casts its input and
weight but adds a BF16 bias to an FP16 product does not compile at all
(MPSGraph rejects a mixed `mps.add`); the DiT passes no bias, but #80's video
VAE decoder does.

The causal grouped query attention of the text encoder keeps its BF16
operands: it is not a hot path, and the text embedding should not move.

## End to end

M2 Ultra, 512x512, 22 frames, `--steps 20 --layers 45 --reuse 2`, seed 42,
ABBA order, original BF16 FL2VA checkpoint (sha256-verified against the
Hugging Face oids). Two runs per variant, averaged:

| | BF16 | FP16 operands |
|---|---:|---:|
| DiT forwards, `h3_dit_bench` 512x512 | 51.41 s | 36.47 s (1.41x) |
| DiT Euler denoise in the render | 76.48 s | 53.91 s (1.42x) |
| end to end per render | 105.5 s | 83.5 s (1.26x) |
| video VAE decode (FP32, untouched) | 4.77 s | 4.77 s |
| peak live tensor storage | 32.84 GiB | 32.84 GiB |
| video SSIM against the BF16 render | - | 0.9472 |
| same-variant SSIM, run-to-run floor | 0.9593 | 0.9475 |

The cross-variant SSIM sits inside the fp16 run-to-run floor, so the format
change perturbs the video no more than repeating the same configuration does.
The DiT output hashes differ between the two modes, as expected from the
attention rounding: `8044e2fb18d9f20e` against `4d623718ede30b91`.

## Interaction with #80

#80 routes its BF16 video VAE decoder through the same entry points:
`vae_linear()` calls `h3_gpu_linear_bf16()` and `vae_sdpa()` calls
`h3_gpu_sdpa_bf16()`. #80 measures a 3.37x decode win on an M5 Pro, where BF16
reaches the tensor cores. Running #80's own `h3_video_vae_tests` on this M2
Ultra, three runs per configuration:

| latent | F32 | BF16 (#80) | BF16 with this PR |
|---|---:|---:|---:|
| 8x12 (128x192, 22 frames) | 0.260 s | 0.398 s (0.66x) | 0.251 s (0.95x) |
| 32x32 (512x512, 22 frames) | 3.055 s | 4.920 s (0.62x) | 2.853 s (1.06x) |

Rel-L2 against the FP32 decode is 0.00506 either way and the peak allocation
stays at the BF16 figure, so on hardware without BF16 matrix kernels this PR
is what keeps #80's decoder from regressing. The VAE decoder's SwiGLU runs in
its own kernel on BF16 buffers, outside the MPSGraph, which is why its
intermediate never enters an FP16 graph and the decoder can use the faster
operands in full.

## What changed

The branch is seven commits on top of `8974cc0` (current `main`) and touches
only `h3_gpu.m`, `h3_gpu.h`, `tests/test_mps_operands.c`, `Makefile` and
`README.md`, so it should cherry-pick onto a fork without dragging anything
else along. The only conflict seen so far is one line in the `test` target of
`Makefile` when stacked on top of #80.

* `h3_gpu_mps_operand()` inserts the cast; the linear and SDPA graph builders
  use it, the fused MLP deliberately does not.
* `h3_gpu_calibrate_operands()` measures both variants once per process and
  sets `fp16Linear` / `fp16SDPA`; `h3_gpu.h` exposes the selection.
* The SDPA graph casts its FP16 attention result back to BF16 so the caller's
  buffer keeps its own dtype.
* `tests/test_mps_operands.c`: the probe above, wired into `make test`.
* `README.md`: a new "MPSGraph operand formats" section.

`H3_FP16_OPERANDS=0` keeps the BF16 operands and `H3_FP16_OPERANDS=1` forces
the FP16 ones on the graphs that may use them. `H3_PROFILE=1` prints both
timings.

## Effect on the existing parity suites

`make parity` bounds the BF16 toy block against the MLX fixture at `1e-2`
relative (`tests/test_bf16.c`), and the F32 diagnosis path at `5e-3`
(`tests/test_metal.c`). The F32 path is untouched: `h3_gpu_mps_operand()`
returns the tensor unchanged for any dtype other than BF16. On the BF16 path
the FP16 operands sit at 1.7e-3 against the double precision reference, so
even against a BF16 reference the deviation stays around 5e-3, inside the
existing bound. I could not run `make parity` here because the MLX fixtures
are not installed; this is the reasoning plus the probe numbers, not a
measured parity run.

## Unrelated finding

While measuring, the causal GQA probe in this test kept differing between two
identical contexts. The cause is not this PR: `h3_gpu_gqa_causal_bf16` reads
`reductions[0]` for the softmax maximum and every thread, including thread 0,
then writes its partial sum into `reductions[tid]` without a barrier in
between, so a late SIMD group can read the partial sum as the maximum. #85
already fixes this, and #98 extends the same fix to
`h3_vae_encoder_group_norm_silu_f32`, which has the identical hazard. On an
idle M2 Ultra two consecutive runs of the same context differed in 254-507 of
262144 results (rel L2 ~8e-3) at sequence 256, 8 query heads, 2 KV heads, head
dim 128, which is more often than the under-load-only repro described there.

## Tests

* `make -j8` builds without warnings.
* `make test` passes; the fixture-dependent suites still skip.
* `./h3_mps_operands_test` passes (output above).
* Full 512x512 renders in both operand modes, ABBA, with SSIM below.



### #99 [open] PR: Optional offline INT8 DiT pack (h3_pack_int8) with page-aligned zero-copy loading
URL: https://github.com/antirez/h3.c/pull/99  by ain-soph 2026-11-17T23:29:14Z

## Summary

On the M5 TensorOps path the DiT block matrices run as INT8 with F32 row scales, and the loader computes them from the 62 GiB BF16 FL2VA checkpoint on every load (12–15 s here). This PR adds an **optional, offline INT8 pack**. Nothing changes unless a pack is present.

- **`h3_pack_int8`** (new tool, `make h3_pack_int8`) writes `<variant>/transformer-int8/` once from the BF16 DiT. The output is ordinary safetensors files, about 22 GiB per pipeline:
  - `blocks-NN.safetensors`: every block matrix as `qweight` (I8) + `qscale` (F32). They come from the same GPU quantizer the resident loader runs, so a packed load equals quantizing on load.
  - `adaln-table.safetensors`: the AdaLN outputs of every 2..`--max-steps` serving schedule (default 20), per kernel regime.
  - `base.safetensors`: all other tensors, unchanged.
  - `h3-int8-pack.json`: written last, marks a complete pack.

  `--verify` (at pack time) and `--verify-only` (later) rebuild those AdaLN tables from both checkpoints and compare them bit for bit.
- **`h3`** loads the pack when the marker exists. `H3_DIT_BF16=1` forces the BF16 checkpoint. Once a pack exists, the BF16 directory can be deleted. `--ssd-streaming` streams the INT8 matrices and their scales. Options that need BF16 block matrices (`--use-slower-bf16-*`, the `H3_DISABLE_INT8_*` / `H3_INT8_KEEP_BF16_*` / A/B knobs) are refused with a clear error, as are non-M5 GPUs, sequences under 128 tokens, and schedules longer than `--max-steps`.
- **`tools/align_int8_pack.py`** rewrites the pack so that every tensor whose size is a whole number of pages starts on a page. It compares every tensor byte for byte before it replaces a file. On M5 those tensors are then mapped zero-copy (`mmap` + `newBufferWithBytesNoCopy`), like the BF16 shards already are. The small scale and norm tensors are copied.
- **BF16 path unchanged.** `h3_gpu.m` gains an I8 loader and a dtype-generic file read/stream. The page-alignment rule for mapping applies only to `/transformer-int8/` paths, so BF16 shards map exactly as before. The README has a short section on the pack.

## Relation to #88

#88 (thanks @kelvincht) solves the same problem — repeated BF16 reads and quantization on every load — in a different way:

| | #88 `H3_DIT_INT8_CACHE` | this PR |
|---|---|---|
| Extra step | none: the first load writes the cache | run `h3_pack_int8` once |
| BF16 checkpoint after that | still required (the cache is keyed on the shards' size/mtime) | can be removed: about 62 → 22 GiB per pipeline |
| Format | custom cache file per pipeline and block mask | standard safetensors, plus a JSON marker |
| Loading | read into buffers | zero-copy page-aligned mapping on M5 |
| AdaLN | computed per run | precomputed tables for 2..N steps, verified bit for bit |
| Coverage | any GPU path that quantizes | the M5 INT8 path, ≤ `--max-steps` steps |

The two are not exclusive. #88 needs no user action. This PR trades a one-time pack step for less disk, no BF16 dependency, and a load that is mostly mapping.

## Validation (this branch, M5 Ultra 96 GB)

Passed:
- `make h3 h3_pack_int8` (no warnings), `make test` (all non-weight tests pass; the weight-dependent ones skip as on `main`).
- `./h3_pack_int8 FL2VA/transformer FL2VA/transformer-int8 --max-steps 20 --verify-only`: 76/76 schedules `identical`. The pack here was written by this same `h3_pack_int8.c` source, byte for byte, in my fork; I did not write a second 22 GiB copy for this run.
- T2V 512×512, 56 frames, 20 steps, `--layers 45 --reuse 2`, seed 42, same prompt. The MP4s are **byte-identical** (md5 `0bebc18fe918588167c2076b2f004f3e`) for:
  - `main` (8974cc0) on the BF16 DiT
  - this branch with `H3_DIT_BF16=1`
  - this branch on the aligned pack (each twice, alternating), plus three more pack runs.
- Times (`--profile`):

| | `main`, BF16 DiT | this PR, aligned pack |
|---|---:|---:|
| H3 DiT load | 12.5 / 14.9 s | 0.33–0.73 s |
| GPU Euler denoise | 13.7 s | 15.9 / 16.6 s right after a BF16 run; 13.7–15.7 s in back-to-back pack runs |
| Wall, whole render | 41–42 s | 31–32 s (27–29 s back to back) |

In the alternating runs, the BF16 renders read 62 GiB and evict the mapped pack from the page cache. The next pack render then faults its pages in during the first denoise steps, which accounts for the slower denoise there.

Not run:
- Ref2VA (no BF16 Ref2VA checkpoint here), 768p, or `--ssd-streaming` on this branch. In my fork, the same pack code at 1344×768 (124 frames) gave the same MP4 as resident BF16 quantization, and 365 s vs 367 s.
- Non-M5 hardware. The pack refuses to load there, and the BF16 checkpoint keeps working.

This was prepared with an AI coding agent (Claude Code); I reviewed the diff, ran the commands above and checked the outputs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #98 [open] PR: Fix two threadgroup reduction races (causal GQA softmax, VAE encoder group norm)
URL: https://github.com/antirez/h3.c/pull/98  by ain-soph 2026-11-17T16:31:27Z

## Summary

Adds a `threadgroup_barrier` after a value is read from the reduction scratch and before the scratch is reused, in two kernels:

- **`h3_gqa_causal_bf16`** (Qwen text encoder attention): every thread must read the row maximum from `reductions[0]` before thread 0 overwrites that slot with its partial sum. This is the same fix as #63 (thanks @quantz8a) for #52.
- **`h3_vae_encoder_group_norm_silu_f32`** (video VAE encoder): same pattern. The mean is read from `reductions[0]`, and `reductions[]` is then refilled for the variance reduction. #63 does not cover this one.

This PR includes both, so it supersedes #63.

## Why it matters for video

The race in the text encoder makes renders irreproducible whenever the Qwen prompt sequence is long. That covers every first-frame (FL2VA) and Ref2VA render, because the vision span makes the sequence long. A plain T2V render with a ~600-token prompt is affected too. On an M5 Ultra, identical parameters and seed gave a different video each run. Short-prompt T2V happened to be stable, which is why the problem is easy to miss.

We located it by hashing every pipeline stage. Conditioning pixels, VAE latents and vision rows matched across runs, and the text context first diverged after layer 0.

## Testing (M5 Ultra; tested in our fork, which carries other changes as well)

- First-frame, Ref2VA and a 600-token T2V prompt each render byte-identically across two runs.
- The short-prompt T2V output is unchanged byte for byte.
- End-to-end wall time is unchanged within noise: a 512² T2V takes 26 s, against 24–27 s before.

Fork commit: https://github.com/ain-soph/h3.c/commit/0fa3634323c8580eb5d0f27dafc6713d0a3e67ae (also adds an optional per-stage hash, `H3_DEBUG_HASH=1`, not included here). This PR's branch is upstream `main` plus the two barriers only. I did not run a separate render on this exact branch.

Fixes #52.

🤖 Generated with [Claude Code](https://claude.com/claude-code)


> **Comment by FengGuanyun (2026-11-19T19:19:17Z):**

Independent confirmation of the `h3_vae_encoder_group_norm_silu_f32` half of
this PR, on different hardware (M2 Ultra, macOS 26.2, no TensorOps).

I found the same race separately and had a branch ready before I saw this PR,
so rather than open a duplicate here is the reproduction data. It may be
useful since the testing note above says the renders were done in a fork that
carries other changes.

A probe links `libh3.a`, fills one input buffer with fixed pseudo-random
floats, and dispatches the kernel repeatedly into the same output buffer. The
reduction tree order is fixed, so a correct kernel is exactly reproducible and
any difference between two runs is the race:

```c
for (int iteration = 0; iteration < iterations; iteration++) {
    h3_gpu_begin(gpu);
    h3_gpu_vae_encoder_group_norm_silu_f32(gpu, out, in, w, b,
        1, depth, height, width, channels, /*groups=*/32, 1e-6f);
    h3_gpu_submit(gpu);
    h3_gpu_tensor_read_f32(out, current, count);
    /* memcmp current against the first run */
}
```

The window is widest when each thread does exactly one iteration of the
variance loop, i.e. when `elements == 256`, so the probe uses
`depth 2048, height 8, width 8, channels 128, groups 32` (65536 threadgroups
per dispatch, 4 dispatches per submit). GPU otherwise idle:

| | runs | nondeterministic | differing results | worst delta |
|---|---:|---:|---:|---:|
| before the barrier | 1999 | 10 | 2559 | 1.398e-01 |
| with the barrier | 1999 | 0 | 0 | - |

Two things worth noting:

* Every failing run differs in a multiple of 256 results, i.e. whole groups.
  That is what a corrupted mean produces: one thread reads thread 0's partial
  sum of squares instead of the mean, so the variance for that entire
  `(temporal_plane, group)` pair is wrong and every value in it moves. The
  worst delta, 0.14, is far outside rounding.
* The barrier changes no computed value. The FNV-1a hash of the full output
  buffer is `83c4045f69d5a450` both before and after, so this cannot shift any
  render that was not already hitting the race.

A shape sweep (`elements` of 4, 16, 32, 64, 128, 256) only ever tripped at
256. The encoder's real shapes have larger groups, which lengthens each
thread's variance loop and narrows the window a lot, so this is much rarer in
a real render than the GQA race is. It is still unordered, and unlike the GQA
case it changes the output rather than only the rounding: a wrong mean is not
cancelled out by the normalization.

For what it is worth, `h3_layer_norm_f32` and `h3_layer_norm_bf16` already
place a barrier at exactly this point, and the tiled matmul kernels place both
of theirs correctly. I scanned all 83 kernels in `h3_shaders.metal` for a read
of a reduction result followed by a write to the same scratch array with no
barrier in between; the two this PR fixes are the only hits.




### #97 [open] PR: SageAttention-style int8 attention for Sol-Attn: 21% faster sparse attention, 6% faster renders (opt-in, stacked on #86)
URL: https://github.com/antirez/h3.c/pull/97  by kelvincht 2026-11-17T06:04:43Z

Stacked on #86 (Sol-Attn block-sparse attention). Review that first; this PR's diff is only the commit on top of `sol-attn`.

## What

Opt-in int8 QKᵀ for the sparse flash kernel, similar to SageAttention. Off by default; with it off, the kernel and dispatch sequence are the same as #86.

Env knobs:
- `H3_SOL_QK_INT8=1` enables it. It only affects steps that already run the sparse kernel (`H3_SPARSE_ATTN=1`, inside the sparse sigma window). Dense steps are unchanged.
- `H3_SOL_QK_INT8_SKIP_BLOCKS` takes a comma-separated list of DiT blocks kept on the bf16 sparse kernel. `49` is the recommended value; see Quality.

With sparse attention on, the log prints `h3: sol int8 QK on, bf16 blocks mask=0x...` at start.

## How

A quantize kernel runs after routing. It turns Q and K into int8 per (64-token block, head), with one scale per block. K is smoothed first by subtracting its per-head mean, which leaves softmax unchanged. Virtual keys (unrouted block means) are quantized the same way. The sparse kernel computes QKᵀ with int8 inputs and int32 accumulation on the Metal 4 `matmul2d` path, then applies the scales. Softmax state stays fp32 and PV stays bf16.

## Speed

Sparse flash stage per DiT block, including the quantize kernel. Measured on Q/K/V captured from a real generation with heads repeated to 56 and video rows tiled to length. Apple M5 Max (40-core GPU):

| rows | bf16 sparse | int8 QK | saved |
|---|---|---|---|
| 25.3k | 203 ms | 164 ms | 19% |
| 107.2k | 2600 ms | 2053 ms | 21% |

End to end, on a 101-frame 864x1152 9-step clip, denoise went from 281 s to 263 s (−6.4%) with `H3_SOL_QK_INT8=1 H3_SOL_QK_INT8_SKIP_BLOCKS=49`, against the bf16 sparse kernel. Both runs used the same dense path (`H3_ATTN_TILED=1`) for steps outside the sparse window.

## Quality

- Per-head output against the bf16 sparse kernel, 2020 heads (blocks 0-34 at full head count, plus blocks 0/12/25/37/49 at three steps): every head is at cosine 0.9997 or above except block 49 head 1, at 0.961-0.992. That head has one outlier channel (19, in both Q and K, about 30-60x the median magnitude). One int8 scale per block leaves the other channels with 1-2 levels. A double-precision CPU emulation of the same scheme gives the same loss, so the kernel is not the cause. With block 49 skipped, every checked head passes 0.999.
- A per-channel K scale folded into Q was tried. It was worse on that head (0.955-0.978) because it moves the outlier into Q.
- The same end-to-end pair (int8 with block 49 skipped, against bf16 sparse, both with `H3_ATTN_TILED=1`): average PSNR 41.1 dB. That is about 48 dB on early frames, falling to about 30 dB by the last frames from slow motion divergence. No visible difference in a side-by-side.

## Tests

- `make h3_sol_qk8_check && ./h3_sol_qk8_check PREFIX...` (`tests/check_sol_qk8.c`): runs the int8 sparse kernel, the bf16 sparse kernel, the dense tiled kernel and MPSGraph SDPA on Q/K/V captured with `H3_ATTN_CAPTURE_DIR` from #86. It prints per-head cosine, rel L2 and max abs error, and exits 0 when the worst int8-vs-bf16 head cosine is at least 0.999. It needs a capture, so it is not part of the default run.
- On this branch, a synthetic capture (16384 rows, 4 heads, 512 sink rows, block-correlated Q/K with a nonzero key mean) gives a worst head cosine of 0.99969 (PASS). The same capture with a large outlier channel planted in one head drops that head to 0.943 (FAIL), which matches the failure mode above.
- `h3_sparse_attn_test` from #86 still ends with `ALL PASS` with the knob off.

## Limits / not run

- The speed and accuracy numbers come from one machine and one checkpoint. The outlier head and the skip list of `49` are specific to that checkpoint; other checkpoints may need a different list or none.
- Blocks 35-48 were checked at 4 heads and 3 steps, not at full head count.
- The stage timings came from a per-stage bench that is not in this PR.
- The measurements were taken on the tree this was developed in. On this branch, `h3` and the check tools were built and the synthetic check was run, but the end-to-end render was not re-run.
- Audio was not checked by ear.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #96 [open] PR: Key conditioning cache by input content so re-uploaded identical files hit the cache
URL: https://github.com/antirez/h3.c/pull/96  by kelvincht 2026-11-16T22:00:13Z

Key conditioning cache by input content so re-uploaded identical files hit the cache

Problem: the conditioning cache keys an input by path, size and mtime. A server that writes each upload to a new file misses on every request, even when the bytes are identical. Measured on the author's fork: 30.6 s cold versus 6.2 s when the same content is reused.

Fix: when the cache is enabled, the key is the SHA-256 of the file bytes, so identical content hits regardless of path or mtime. When the cache is disabled, the old stat key is used and no extra file reads happen.

Evidence: `make all` passes; `make test` passes 1768 checks.

Limits / Not run: no render on this branch. The 30.6 s and 6.2 s figures come from the author's fork, not from this branch.

Related: #88 #90 #93

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #95 [open] PR: Add tools to fold LoRAs into H3 weights, with lightx2v distill example
URL: https://github.com/antirez/h3.c/pull/95  by kelvincht 2026-11-16T20:45:18Z

## What

Adds `tools/fold_turbo_lora.py`, an offline script that bakes one or more LoRA adapters into the bf16 MiniMax-H3 transformer shards, and `tools/test_fold_turbo_lora.py`, its tests.

## Why

h3.c has no LoRA runtime, and it doesn't need one for step-distilled LoRAs. A LoRA is `W' = W + scale*(B@A)`, so folding it into the checkpoint once gives a model that samples at fewer steps with no extra per-step cost and no speed flags.

## How it works

- Validates every target tensor (BF16 dtype, shape) before anything is written. Output goes to `<out>.tmp` and is renamed to `<out>` only on success. On failure the tmp dir is removed. `<out>` must not already exist.
- Clones each original shard (copy-on-write via `/bin/cp -c` where available) and rewrites only the bytes of the tensors the adapter touches. Headers, tensor order and alignment stay byte-identical.
- Each targeted tensor is read once and written once with the sum of all deltas. After each write the tool re-reads the BF16 bytes from the output file and checks two things. (a) The bytes must equal the BF16 encoding of the float32 fold exactly. (b) The float32 fold must lie within 2^-20 of W + dW, computed independently in float64, where the bound scales with the sum of magnitudes |W| + sum(|scale|·|B|@|A|) rather than with the result. The check is absolute to that magnitude, so near-zero elements do not fail. A 4096x4096 test with a rank-64 LoRA covers this.
- BF16 rounding is round-to-nearest-even. NaN maps to a quiet NaN with the sign preserved (`0x7FC0`/`0xFFC0`).
- Accepts native, ComfyUI (`diffusion_model.*` with `.alpha`) and kohya (`lora_unet_*`) key naming. Effective scale is `strength * alpha / rank`.
- Time-curve projection pairs (`blocks.N.adaln_proj.linear` with an 8-dim A input, which means the adapter was trained against a low-rank-AdaLN base) cannot be folded into the weight. The tool aborts on them. Pass `--skip-curve-pairs` to fold the rest and leave those targets unchanged.
- Requires only numpy.

## Usage

Fold the public lightx2v MiniMax-H3 distill LoRA (ComfyUI naming) into the Ref2VA transformer. The adapter filename is a placeholder:

```sh
python3 tools/fold_turbo_lora.py \
    --checkpoint MiniMax-H3/Ref2VA/transformer \
    --lora distill_comfyui.safetensors:1.0 \
    --out MiniMax-H3-distill/Ref2VA/transformer
```

Add `--skip-curve-pairs` if the adapter contains adaln_proj curve pairs and you accept leaving them unfolded.

Point `h3` at a model directory whose `transformer` is the folded tree. Don't combine a distilled schedule with `--reuse` or `--core-reuse`.

## Evidence

- `python3 -I tools/test_fold_turbo_lora.py`: 7 tests pass. They cover:
  - an F32 target aborts before any output is created;
  - curve pairs abort without the flag, and are skipped with it, with the rest folded;
  - shape mismatch aborts;
  - kohya naming with alpha and no curve pairs;
  - NaN and sign handling in BF16 conversion;
  - a 4096x4096 rank-64 LoRA, where near-zero elements must pass.
  The tests write synthetic safetensors. Verification is two-part: the bytes read back from the output equal the written BF16 bytes exactly, and the float32 fold is within 2^-20·(|W| + Σ|scale|·|B|@|A|) of W+ΔW computed in float64.
- `python3 -I tools/fold_turbo_lora.py --help` prints the usage.

## Limits / Not run

- Not run against a real MiniMax-H3 checkpoint or the lightx2v adapter. The tests use synthetic tensors only, so there is no measured quality or speed result in this PR.
- Diffusers-layout adapters (`to_q`/`to_k`/`to_v`, `ff.net.*`) are rejected with an error. They need the ComfyUI conversion first.
- Re-reading each tensor and computing W + dW in float64 doubles the memory and I/O per target. This is acceptable for an offline tool but has not been measured on full-size shards.
- `<out>` must not already exist; the tool no longer reuses an existing directory.
- No README section yet. Usage is documented in the script docstring only.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #94 [open] PR: Add an opt-in x265 10-bit BT.709 MP4 encode that avoids x264 border residue (H3_MP4_HEVC10=1)
URL: https://github.com/antirez/h3.c/pull/94  by kelvincht 2026-11-16T20:17:18Z

Add an opt-in x265 10-bit, BT.709 MP4 encode (H3_MP4_HEVC10=1)

Off by default: with `H3_MP4_HEVC10` unset, both MP4 writers build the same x264 8-bit command line as before. With `H3_MP4_HEVC10=1` they encode x265 10-bit (`yuv420p10le`, CRF 20, tagged `hvc1`), convert RGB with the BT.709 matrix to limited range, and tag BT.709 in the bitstream.

Why: x264's 16x16 macroblocks sit on the decoder's 16-px patch grid and add a little patch-border residue on smooth areas.

Evidence, measured on the author's fork: on a 544x736x260 clip, the share of patch-border segments above 0.5/255 after decode was 16.1% for lossless, 20.9% through x264 CRF 18, and 16.7% through x265 CRF 20 10-bit. The x265 files were about 40% smaller at about the same encode time.

Limits / Not run: the default command line was checked by reading the code, not by diffing an encoded file against upstream; no playback check on other players for the 10-bit or tagged output.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #93 [open] PR: Add video VAE precision, weight cache and prefetch knobs, all off by default
URL: https://github.com/antirez/h3.c/pull/93  by kelvincht 2026-11-16T20:15:52Z

Add video VAE precision, weight cache and prefetch knobs (off by default)

Three opt-in knobs for the video VAE decoder. Default output is unchanged: with none of them set, the F32 path and load order run as before.

- `H3_VAE_PRECISION=bf16|int8` runs the four large block matmuls (QKV, output, MLP) and attention on BF16 or int8 TensorOps. The residual stream, norms and RoPE stay in F32. Unset or `f32` is the released path.
- `H3_VAE_CACHE=<dir>` (with bf16) keeps the lowered projections in `<dir>/video-vae-bf16-<fingerprint>.cache`, about 4.8 GiB, and reads them instead of about 9.7 GiB of F32 plus casts. The fingerprint covers the resolved weight directory and each shard's name, size and mtime, so a changed weight set rebuilds the file. It is written under a temporary name and renamed when complete. The cache applies to the resident decoder (prefetch included) and to `decode_chunked`, bf16 precision only; the single-tile streaming path does not use it.
- `H3_VAE_PREFETCH=1` loads the video VAE decoder on a second thread while the DiT denoises, then decodes with it. On any failure the serial load runs as before. Prefetch keeps the full video VAE resident alongside the DiT during denoise: about +9.7 GiB of F32 memory at the default f32 precision. With bf16 and the cache it uses less. Leave prefetch off on machines where that does not fit.

Also adds `H3_DUMP_VIDEO_LATENT` and `tools/vae_bench` to replay a VAE decode from a dumped latent (diagnostic only).

Evidence, measured on the author's fork (M5 Max): the F32 block projections took about 45 s of a 5.2 s 768x448 clip. With `H3_VAE_CACHE` and bf16, a 124-frame 768x448 render went from 100 s to 97 s end to end. With an int8-streamed DiT the prefetch saved about 8 s. The author reports output bit-identical with and without the cache or prefetch.

Limits / Not run: no quality comparison of bf16 or int8 against F32; no unit test for the cache fingerprint or the prefetch failure path; `make all` passes on this branch.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #92 [open] PR: Add a renoise sampler (H3_RENOISE=1) that re-injects fresh noise each step
URL: https://github.com/antirez/h3.c/pull/92  by kelvincht 2026-11-16T20:13:38Z

Add a renoise sampler (H3_RENOISE=1) that re-injects fresh noise each step

Off by default. With `H3_RENOISE=1` each step jumps to the clean estimate (`x0 = x - sigma * v`) and re-noises to the next sigma with fresh normal noise, on the CPU sampler path. The schedule uses the published knots 1000/750/500/250/100/40/0, shifted by the video and audio sigma shifts; other step counts resample that curve linearly.

Stopped previews keep the last clean estimate taken before the re-noise, and a resumed span re-seeds the re-noise stream from the request seed and skips the draws of the earlier steps. So a run stopped at step N and resumed matches an unbroken run.

Depends on #91 (stop/resume); this branch includes its commit until that merges.

Evidence: measured on the author's fork, 352x448x39, 6 steps, seed 7: stop at 3 and resume is bit-identical to the one-shot render (video and audio).

Limits / Not run: CPU re-noise path only; no GPU path, no quality comparison against the default sampler, not measured on other hardware.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #91 [open] PR: Stop denoising at a step and resume it later, so previews and final renders share one trajectory
URL: https://github.com/antirez/h3.c/pull/91  by kelvincht 2026-11-16T20:10:36Z

Adds the ability to stop denoising after N steps, write the latent state to a file, and continue from that file later. A preview rendered with fewer steps follows a different sigma grid than the full render, so the two clips drift apart even with the same seed. The full render also repeats the work the preview already did. Stopping and resuming keeps both on one trajectory.

**What it does.** `h3_params.stop_step` runs only the first N Euler steps of the full schedule and decodes the clean estimate: one Euler step from that sigma to 0 using the last velocity, with no extra model evaluation. `latent_save` writes the state at the stop. `latent_resume` continues from a saved state. The saved file is keyed by the prepared-DiT key, the seed and the transformer path, so a resume is refused if the prompt, input, canvas, frame count, step count or seed differ. CLI commands: `!stop [N]`, `!save-latent [PATH|clear]`, `!resume [PATH|clear]`.

**Why.** A low-step preview and the final render can share one trajectory, and the final render does not redo the steps the preview already computed.

**Env knobs.** None. The default `stop_step` of 0 runs all steps, so the sampling path is unchanged unless `!stop` or the resume options are used.

**Constraints.** A partial run needs `reuse 1` and core reuse 1, because velocity extrapolation and core reuse carry state between steps that a stopped or resumed run does not have. The run is rejected with an error otherwise.

**Port notes.** The change is applied to upstream `main` as it is. No private-only symbols were needed. The private branch's other changes to the same area (the VAE prefetch thread and a latent dump diagnostic) are not part of this PR.

**Dependencies.** Does not depend on #85, #86, #88 or #90.

Limits / Not run:
- `make all` builds cleanly on macOS (clang, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`, no warnings in the output).
- No unit test exists for the stop or resume path, and the `!stop` / `!save-latent` / `!resume` commands were not run on this branch. No video render was run, so the claim that a resumed clip matches a fresh full render is not verified here.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #90 [open] PR: Cache the AdaLN schedule on disk (H3_ADALN_CACHE_DIR) to skip recomputing it on every load
URL: https://github.com/antirez/h3.c/pull/90  by kelvincht 2026-11-16T20:08:59Z

Adds an opt-in on-disk cache for the AdaLN schedule.

## AdaLN schedule cache (`H3_ADALN_CACHE_DIR`)

Setting `H3_ADALN_CACHE_DIR=<dir>` makes the first load build the schedule and store it in that directory. Later loads with the same inputs read it back instead of recomputing it.

**What it does.** The schedule holds the per-step AdaLN modulation for every block and the final layer. It depends only on the checkpoint, the step count, the sigma schedule and the condition-row layout, so it is keyed on those, not on seed or prompt. Each file stores the full key, which is compared on load, so a file-name hash collision cannot return the wrong schedule. Writes go to a per-pid temporary file and are renamed into place, so an interrupted write is never read; a file with trailing bytes is rejected. The directory is capped at 10 GiB with least-recently-used eviction.

**Key contents.** The key includes the checkpoint directory, a schedule version constant, the model dimensions, and a weights identity for the checkpoint. The weights identity is a 64-bit stamp: an FNV-1a hash of each `.safetensors` file's name, size and mtime, summed over all shards. Summing makes the result independent of `readdir` order. Replacing a shard in place changes its size or mtime and so changes the stamp, which makes the entry miss. The stamp is a 64-bit hash, so a collision is unlikely but not impossible; the key is still compared in full on load, but the stamp itself is not a collision-free identity. If the directory or any shard cannot be read, the stamp is not computed and the cache is skipped.

**Path length.** If the cache directory or the temporary file name does not fit its fixed buffer, the cache is skipped and the schedule is computed as normal.

**Why.** Every load recomputes the same schedule for a given checkpoint and settings. Storing it once removes that work from repeat loads.

**Env knob.** `H3_ADALN_CACHE_DIR` is off by default. Without it, the load path is unchanged. Bump `ADALN_SCHEDULE_VERSION` when the schedule math changes.

**Removed from this PR.** The low-rank AdaLN store support and the `H3_ADALN_TABLE` table lookup have been removed. No public producer exists for low-rank stores, and `H3_ADALN_TABLE=1` made full-rank loads fail, so neither is part of this change. This PR is cache only.

## Evidence

- `make all` builds cleanly on macOS (clang, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`) on the branch tip.
- `make test` on the branch tip: `h3_tests` ok (1768 checks), `h3_audio_gpu_tests` ok, FFmpeg mux test ok. The fixture- and weight-dependent tests skipped because those files are not installed in the test checkout.

## Limits / Not run

- No unit test exists for the cache. The round trip, the key check and the eviction path were not exercised on this branch. No video render was run.
- The truncation guards on the cache path and temporary file name were not exercised with an over-long directory.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #89 [open] PR: Add !offload to free the DiT and video decoder between renders, keeping the prompt conditioning cache
URL: https://github.com/antirez/h3.c/pull/89  by kelvincht 2026-11-16T20:06:13Z

Adds `h3_offload()` and an interactive `!offload` command. Between renders it frees the DiT and the video decoder, so their memory is returned, and keeps the prompt conditioning cache, so the next render with the same prompt still skips the text encoder. The next render reloads the DiT.

**What it does.** `h3_offload(ctx)` calls `h3_dit_free`, frees the video decoder, and clears the keys that record what was loaded. It is a no-op on a NULL context. In the interactive CLI, `!offload` calls it and prints a confirmation. The help text and the README command list mention it.

**Why.** An interactive session holds the DiT and decoder resident between prompts. On memory-tight machines, a user who is pausing between renders can release them without restarting and still keep the cheap prompt-conditioning reuse.

**Env knobs.** None. This is a user command, so nothing changes unless `!offload` is typed.

**Port notes.** The private fork's version also released a retained DiT core (`h3_dit_release_core`). Upstream has no such block, so the port frees the DiT through the normal `h3_dit_free` path only. The README paragraph about on-disk DiT and AdaLN caches from the same private commit is left out; those features are not in this PR. This PR has no dependency on the int8 cache PR (#88).

Limits / Not run:
- `make all` builds cleanly on macOS (clang, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`, no warnings in the output).
- No unit test exists for this path. The interactive `!offload` command was not run, and no render was run. The memory release and the reload on the next render are unverified in this branch.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #88 [open] PR: Cache DiT int8 weights on disk behind H3_DIT_INT8_CACHE to skip BF16 reads and quantization on later loads
URL: https://github.com/antirez/h3.c/pull/88  by kelvincht 2026-11-16T20:04:43Z

Adds an opt-in on-disk cache for the DiT int8 core weights, so later loads skip the BF16 reads and int8 quantization.

**What it does.** Setting `H3_DIT_INT8_CACHE=<base>` makes the first load write the int8 weights and f32 scales of each block to a file. The file is named `<base>.<pipeline>.<active mask hex>`, where the pipeline is `ref2va` or `fl2va` taken from the transformer directory. Ref2VA, FL2VA and different block masks therefore never overwrite each other. Later loads read the file back directly.

The cache header stores the magic and version, the model dimensions, the active block mask, the weight directory and a fingerprint of the `.safetensors` shards. The fingerprint is a 64-bit sum over shards of an FNV-1a hash of each shard's name, size and mtime; the sum does not depend on `readdir` order. A file is used only when all header fields match, its size is exact, and it is marked complete. Writes go to a pid-unique temporary file that is renamed into place once complete, so an interrupted write is never reused.

**Failures never fail the load.** The cache is an optional side file.
- If the weight directory or a shard cannot be read, or the transformer directory is neither Ref2VA nor FL2VA, the cache is skipped.
- A read or validation failure is a miss. A block that fails to read from the cache is rebuilt uncached, and the bad file is removed so the next load rewrites it.
- A write-side failure (open, write, close or rename) closes the file, removes the temporary file, and continues the load uncached.

**Why.** Quantizing the BF16 checkpoint to int8 is deterministic for a given checkpoint, so the result can be stored once. Loads from the cache skip the BF16 reads and the quantization pass.

**Env knob.** `H3_DIT_INT8_CACHE` is off by default. Replacing the shards in place is detected through their size and mtime, so the cache file does not need deleting for that. The header does not hash file contents.

**Port notes.** Only the cache commit is included. It needed one small addition that upstream lacks: a `weight_directory` pointer on `h3_dit`, set in `load_dit` and read only while the cache is opened during load.

Limits / Not run:
- `make all` and `make test` pass on macOS (clang, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`). The test suite skips the weight-dependent tests, so the cache path itself was not exercised by the suite.
- No unit test exists for the cache. The failure fallbacks (unreadable file, unreadable shard, write failure) were not exercised with real weights. The "outputs match the no-cache path byte-for-byte" claim comes from the private branch's testing and was not re-run on this upstream-based branch. No video render was run here.
- The cache does not key on the int8 quantizer code. A change to the quantization math needs a manual delete of the cache file.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #87 [open] PR: Fix black video from unpadded safetensors, quilt artifacts from large VAE tiles, and shader load failures outside the source dir
URL: https://github.com/antirez/h3.c/pull/87  by kelvincht 2026-11-16T20:02:22Z

Fixes three user-visible failures. All three are fixes to existing behavior; none adds an opt-in knob.

**1. Unpadded safetensors headers render black video.** The GPU path maps tensor data in place and assumes the data section (8 + header bytes) starts 8-byte aligned. Writers that do not pad the JSON header (e.g. MLX's `save_safetensors`) produce spec-valid files that load without error but decode to garbage, giving silent black video with intact audio. The loader now refuses such files at open with an actionable message ("re-save with the JSON header padded to an 8-byte boundary"). Verified by round-tripping a checkpoint through such a writer: the unpadded file renders black, and the same file with the header padded renders correctly, tensor order unchanged.

**2. `h3` fails when launched outside the source directory.** A relative `h3_shaders.metal` that is absent from the current directory now falls back to the directory containing the binary, so `h3` can be launched from PATH installs, daemons, or other projects, instead of failing with "cannot compile h3_shaders.metal". The fallback only applies when the relative path does not exist in the current directory.

**3. Large VAE tiles produce a grid/quilt artifact.** The `H3_VAE_TILE_PIXELS` env override accepted up to 512, but tiles above 320 px show a visible grid across the whole frame (reproduced at 512 and 1088, 960x544 canvas). The override now uses the same 320 bound as the automatic search. Values above 320 now fall back to the automatic choice instead of being accepted.

Commits are independent of each other; no dependency on other open PRs.

Limits / Not run:
- `make all` builds cleanly on macOS (clang, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`, no warnings in the output).
- No unit test exists for these paths. No video render was run; the black-video and quilt results come from the reproductions described above, not from a render in this branch.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #86 [open] PR: Opt-in Sol-Attn block-sparse attention for the DiT (23-28% faster on 5 s clips)
URL: https://github.com/antirez/h3.c/pull/86  by kelvincht 2026-11-16T09:48:27Z

## What

Opt-in block-sparse attention for the DiT. Off by default; with it off, output is bit-identical to the base.

Env knobs:
- `H3_SPARSE_ATTN=1` enables it.
- `H3_SPARSE_TAU` (default 1.3) sets the routing threshold.
- `H3_SPARSE_START` / `H3_SPARSE_END` (0.2 / 0.9 of the schedule) bound the sparse sigma window.
- `H3_SPARSE_MIN_TOKENS` (default 20000): shorter sequences stay dense.
- `H3_ATTN_TILED` selects the dense tiled TensorOps kernel that the sparse kernel builds on.

## How

Block summaries (64-token blocks: fp32 query and key means, value sums), then per-head key statistics, then routing, then a sparse kernel. Unrouted blocks are folded in as one "virtual key" each (key mean, value sum). Rows before the target video (text, conditions, audio) are exact sinks.

## Speed

End to end on Apple Silicon (M-series Mac Studio), 6 steps, idle GPU, one configuration per row:

| config | dense | sparse tau 1.3 | saved |
|---|---|---|---|
| 768x1024x22 | 59.6 s | 58.9 s | 1.1% |
| 768x1344x22 | 68.8 s | 66.2 s | 3.7% |
| 512x768x124 | 118.0 s | 105.9 s | 10.3% |
| 768x1024x124 | 297.7 s | 229.4 s | 22.9% |
| 768x1344x124 | 451.4 s | 324.3 s | 28.2% |
| 512x768x362 | 519.5 s | 383.6 s | 26.2% |

tau 0.7 saves 17-22% at 124 frames. Break-even is about 7k tokens.

## Quality

- Sparse-off output is bit-identical to the base (framemd5 plus final latent) across 9 model configurations.
- On 1 s clips with sparse forced on: final-latent PSNR 20-33 dB against dense. That is comparable to the run-to-run noise between two dense runs. No block or grid artifacts appear; the differences are pose, expression and timing drift.
- Real-use 768x1024x124 run: 32.5 dB, visually near-identical.
- A small blind human A/B on 16 pairs rated the differences mild.

## Tests

- `make h3_sparse_attn_test && ./h3_sparse_attn_test h3_shaders.metal` (`tests/test_sparse_attn.c`): checks the routed kernel against the dense tiled kernel (bitwise when all blocks are routed), against MPSGraph SDPA, and against a CPU reference for routing and the approximation. Output ends with `ALL PASS`.
- `tests/check_sparse_attn.c` (`h3_sparse_attn_check`) checks captured Q/K/V dumps (`H3_ATTN_CAPTURE_DIR`); it needs a capture, so it is not part of the default run.
- `tools/parity_renders.sh` compares two binaries' renders per profile; it needs a `ROOT` with `models.json` and the server registry, so it is a local tool.

Note: main's text encoder has a separate nondeterminism race; see #85.

## Limits / not run

- Speed was measured on one configuration and one machine.
- Audio was not checked by ear.
- The long-clip visual check covers one configuration only.
- The base itself is not run-to-run deterministic: a missing barrier in `h3_gqa_causal_bf16` in `h3_shaders.metal`. A separate fix may follow; `H3_MPS_GQA=1` gives determinism for comparisons.
- The quality and speed numbers were measured on the fork this was developed in, before rebasing onto the current main. The rebased code was built and unit-tested here, but the end-to-end renders were not re-run on the rebased tree.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #85 [open] PR: Fix same seed giving different renders: missing barrier in h3_gqa_causal_bf16
URL: https://github.com/antirez/h3.c/pull/85  by kelvincht 2026-11-16T09:48:14Z

## metal: add missing barrier in h3_gqa_causal_bf16 reduction

### Problem

`h3_gqa_causal_bf16` is dispatched with up to 128 threads per threadgroup, which is four SIMD groups. After the max reduction, every thread reads `reductions[0]`. Right after that, every thread, thread 0 included, writes its partial softmax sum into `reductions[tid]`, with no barrier in between:

```metal
float maximum = reductions[0];
float local_sum = 0.0f;
for (...) { ... local_sum += exp(scores[key_row] - maximum); }
reductions[tid] = local_sum;   // thread 0 overwrites reductions[0]
```

A SIMD group that reaches the read late can get thread 0's partial sum in place of the max. The softmax stays mathematically correct, because the subtracted constant cancels, but the rounding changes. The text-encoder output can therefore differ between runs with the same seed. The MPSGraph path (`H3_MPS_GQA=1`) is not affected.

### Fix

This PR adds one `threadgroup_barrier(mem_flags::mem_threadgroup)` after the read of the maximum. The sum reduction that follows does not need one: after `inverse_sum` reads `reductions[0]`, nothing writes `reductions` again.

### Testing

- `make all` builds cleanly.
- A small probe links `libh3.a` and runs `h3_gpu_gqa_causal_bf16` 50 times in one process on fixed pseudo-random bf16 inputs (sequence 175, 64 query heads, 8 KV heads, head_dim 128).
  - With the fix, all 50 output hashes are identical, and they match the unfixed kernel's output bit for bit.
  - The race depends on timing. On an idle GPU the unfixed kernel also gave 50 identical hashes. Run-to-run differences were seen under GPU load. This change is a correctness fix under the Metal memory model, so it does not depend on reproducing the race.
- The cost of one extra barrier per threadgroup is negligible.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #84 [open] PR: Add MiniMax-H3-Image: text-to-image and reference-image editing
URL: https://github.com/antirez/h3.c/pull/84  by ronicayu 2026-11-13T04:10:10Z

## Summary

This adds a MiniMax-H3-Image path to h3.c: text-to-image and reference-image editing with the official BF16 weights, on the same Metal engine and in the same style as the video path. The video path is unchanged apart from shared refactors and one race fix.

- **Text-to-image:** `./h3 -d MiniMax-H3-Image -p "a red fox in snow" --size 1024x1024 -o fox.png`. Six steps, no CFG, the six per-step heads. Resident BF16 is the default; `--int8` and `--ssd-streaming` work as for video. Measured on an M5 Max with 128 GB: 4096×4096 in 163 s; canvases up to 8192×4096.
- **Reference-image editing:** `--ref-image PATH`, repeatable up to 8, with `<Picture N>` tags in the prompt. EXIF orientation is applied. Size and count limits are checked before any weight loads. A 512×512 edit with one reference takes 32 s.
- **Text-embedding disk cache:** repeated prompts with the same references skip the vision tower and the text encoder. The key covers the prompt, the decoded reference pixels, the model files, the GPU, the `H3_*` environment and a digest of the sources on the text path. `--no-cache` turns it off.
- **Race fix in `h3_gqa_causal_bf16`:** a missing `threadgroup_barrier` after the max reduction. A late thread could read thread 0's partial sum as the maximum, which gives wrong attention for that (row, head), not only run-to-run noise. This is the same fix as #63. `tests/test_gpu_determinism.c` repeats the dispatch and requires byte equality; it fails on current main and passes with the barrier.

## How it is built

- **New modules:** `h3_image_*` (canvas sizing, VAE decoder and encoder, pipeline, edit, cache, generation) and `h3_dit_family.[ch]` (the image model constants).
- **Shared code is parameterized, not forked:** DiT tensor-name tables, the AdaLN activation order, VAE tiling (`h3_vae_tiles.c`) and the QKV/FC1 weight loaders.
- **`h3_dit_tuning`:** one struct now carries the DiT tuning flags for both `h3_params` and `h3_image_params`.
- **One heavy model resident at a time:** text encoder, then DiT, then VAE.

## Testing

- **Unit and GPU tests:** `make test` passes. Real-weight tests skip when their weights or fixtures are absent.
- **Real-weight parity tests** against the diffusers reference (and an F32 truth where cheap), for text, block 0, every step, the VAE decoder and encoder, end to end, and edit text / edit DiT / edit end to end, in bf16, stream and int8. Stream and bf16 are bit-identical.
- **Fixtures:** generated by `tools/h3_image_reference.py`; see `docs/image-fixtures.md`.
- **Video encoder:** byte-identical before and after the shared encoder refactor (checked with a probe, since the video fixtures are not public).

## Notes

- **Size:** this is a large change (83 commits, about +16k lines including tests and the Python reference harness). I am happy to split it into a sequence of smaller PRs if you prefer, for example: shared refactors with no behaviour change, then PNG I/O, the image VAE, the image DiT, text-to-image, editing, and the cache last.
- **Not included:** model weights and fixtures.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #83 [open] ISSUE: Deadlock (not corruption) at 243 frames: `_MTLCommandBuffer waitUntilCompleted` never returns
URL: https://github.com/antirez/h3.c/issues/83  by ChunxueLi 2026-11-11T00:40:49Z

## Deadlock (not corruption) at 243 frames: `_MTLCommandBuffer waitUntilCompleted` never returns

### Environment

- Apple M4 Max (36 GiB unified memory), macOS 26.6.2 (Build 25G83)
- h3.c at `8974cc0` **plus PR #77 merged** (SDPA query-block splitting compiled in)
- Official original BF16 FL2VA shards, `--ssd-streaming`, no quantization
- Also reproduced on a **fresh reboot** with nothing else running

### Symptom

At `--frames 243` (1248x704, 7 or 4 steps), denoising stalls at step 0 or mid-chain and never progresses. `sample(1)` on the hung process shows the main thread parked in:

```
h3_generate → h3_dit_denoise_euler_preview
  -[_MTLCommandBuffer waitUntilCompleted]  (2526/2526 samples)
    __psynch_cvwait
```

CPU ~0%. The command buffer never completes. Killing and retrying the exact same command hangs at the same place. No error, no crash log, no MTLCompiler messages.

### What makes it confusing

- The **same prompt and anchors at 107 frames** (`latent_t=27`, sequence ≈ 23166 rows) completes fine — 20+ runs across days, night and day, clean reboots, zero failures.
- 243 frames (`latent_t=61`, sequence ≈ 52338 rows) hung **on a clean boot, GPU idle, single process, 48% memory free**.
- Attempts that did **not** help:
  - PR #77 SDPA query splitting (`H3_SDPA_MAX_QUERY_ROWS=2048`) — PROFILE confirms blocks active (`rows=64378 heads=56 query-block=2048`), still deadlocks
  - `H3_GPU_SAMPLER_WINDOW=1` + `H3_DIT_COMMAND_BLOCKS=4` (command-chain split) — got further (2/4 steps) but parked at the same `waitUntilCompleted`
  - Killing ToDesk / screen-capture software, freeing memory, fresh reboot
- Meanwhile a **107f run with the very same prompt and anchor** completed right after a 243f hang.

### Workaround in production

We generate at 107 frames per segment (4.5 s) and stitch with xfade. Works flawlessly for episodic production. Long takes are simply written as multiple short takes.

### Suspected layer

MPSGraph fused SDPA kernel / AGX driver on large non-causal sequences (M4 Max, macOS 26.6). Corrupting variant already reported in #16 / #64 / #77; this adds a **hang variant** at even larger sequences, where the buffer never completes instead of returning garbage.

Happy to provide full logs, the exact command lines, or run additional probes on this machine.

### Repro summary

```sh
# HANGS (denoise never progresses):
./h3 --ssd-streaming -d MODEL_DIR --first-frame anchor.png \
  -p "<any prompt>" --width 1248 --height 704 --frames 243 \
  --steps 4 --layers 50 --reuse 1 -o out.mp4

# COMPLETES (same everything, 107 frames):
./h3 --ssd-streaming -d MODEL_DIR --first-frame anchor.png \
  -p "<same prompt>" --width 1248 --height 704 --frames 107 \
  --steps 4 --layers 50 --reuse 1 -o out.mp4
```


> **Comment by ChunxueLi (2026-11-11T04:34:40Z):**

**Follow-up data point: a "hung" 107f run that self-recovered after 24h+**

One more observation that may help anyone else hitting this.

A 107-frame run using `--motion-context` (PR #76) appeared to hang exactly like the 243f runs above — stuck at `denoise 3/4` with the thread sleeping in `h3_dit_denoise_euler_preview → _MTLCommandBuffer waitUntilCompleted`. I left it running overnight, assuming it was dead.

~24 hours later it **completed on its own** (FFmpeg 107/107, valid 4.46s output). The completion happened roughly 75 minutes after all *other* GPU work on the machine stopped — during the entire stall, a second `h3` instance had been running a production batch concurrently on the same GPU.

Two practical takeaways:

1. **Concurrent h3 instances on M4 Max can stall each other's denoise for many hours.** Anyone debugging these hangs should first verify exactly one instance is running, and observe for >12h before declaring a hang permanent.
2. The 243f hang in my original report still reproduced on clean single-instance runs after a fresh reboot, so that case stands — but at least some "hangs" may be extreme stalls rather than true deadlocks.



> **Comment by ChunxueLi (2026-11-13T02:24:43Z):**

**RESOLVED — the hang is a macOS GPU-driver regression, fixed in macOS 26.7.3**

Closing the loop for anyone hitting this on Apple Silicon.

Timeline on one M4 Max 36GB machine (same hardware throughout):

| Period | macOS | 243f behavior |
|---|---|---|
| late Oct – Nov 2 | earlier 26.x | completed normally (our v4 pipeline was built on 243f takes) |
| Nov 8 – Nov 12 | 26.6.2 | hung 100% of the time: stuck at a `denoise` step in `_MTLCommandBuffer waitUntilCompleted`, 0% CPU, never recovers. ~10+ attempts, all workarounds failed (SDPA query chunking #77, command splitting, clean reboots, single instance). 107f never hung once in 20+ runs. |
| Nov 13 | **26.7.3 (25G342)** | **same probe completes cleanly**: 1248×704, 243 frames, 4 steps, `--ssd-streaming`, first-frame anchor, `H3_SDPA_MAX_QUERY_ROWS=2048` → exit 0 in 3780s, valid 10.125s output |

So the deadlock was introduced by a macOS 26.6.x GPU-stack change and fixed again in 26.7.3. Nothing to fix in h3.c — but one diagnostic note that cost us dearly: on a *healthy* M4 Max run, one 243f denoise step takes ~10–15 min (SSD streaming + 2× sequence length), and the progress bar shows 0% CPU the whole time because the thread sleeps in `waitUntilCompleted`. That is visually identical to the hang during the first minutes. Give any suspicious run up to ~30 minutes at one progress-bar cell before declaring a hang — and, as noted in our earlier comment, never run two h3 instances concurrently while debugging.

Thanks for the great project — closing out from our side.



> **Comment by ChunxueLi (2026-11-27T14:47:48Z):**

**Recurrence on 26.7.3 + a false-alarm pattern worth knowing**

Follow-up for anyone tracking this on 26.7.3 (25G342):

- **Nov 25:** one real recurrence on the same machine, 124 frames this time: parked at `denoise 2/7` inside `_MTLCommandBuffer waitUntilCompleted`, 0% CPU, never recovered (~10 h before we killed it). So while 26.7.3 made the 243f reproducible hang disappear, the stall can still strike at random — now at any frame count, not just long sequences.
- **False-alarm pattern:** the *first* denoise step after launch includes shader compilation plus SSD-streaming warm-up and can take ~2.5 min with the main thread parked in the exact same `waitUntilCompleted` stack at 0% CPU. In `sample` it looks byte-for-byte identical to the hang. Don't judge from the CPU state or a `sample` dump alone — watch the `denoise N/7` counter in the log; if the counter advances, it is alive.
- Production mitigation that works for us: wrap runs in a watchdog that kills the process when the step counter has not advanced for N minutes, then retries (a per-step resume knob like PR #91 would make retries much cheaper).
- We have not tested macOS 27 yet. The report that 27.2 fails fast on bad command buffers instead of hanging sounds promising — will report back after an upgrade.



### #82 [open] PR: Lock audio: use a reference clip as the exact soundtrack (--lock-audio)
URL: https://github.com/antirez/h3.c/pull/82  by Bambushu 2026-10-25T06:51:32Z

H3 always generates its own soundtrack. `--ref-audio` conditions the voice, but the
words and timing are denoised from noise with the picture, so there is no way to make
a take lip-sync to a vocal you already have (a song, a recorded line).

`--lock-audio` makes the first `--ref-audio` clip the target soundtrack itself:

- its clean AudioVAE latent becomes the target audio rows,
- the audio sigmas are zeroed, so those rows carry the condition timestep
  (1 - sigma = 1) at every step, the way reference rows do,
- the audio Euler update is skipped.

Only the video is denoised, attending to the given audio. This is what ComfyUI does
for the same job: the community NativeAudioLock node masks the audio stream, and
ComfyUI's H3 model pins masked rows at the condition timestep. Earlier fork work on
the idea: KaedeTai/h3.c (`H3_AUDIO_ANCHOR`) and lyj97/h3.c (`codex/a2va-locked`).

```sh
./h3 -d MODEL -p "$(cat prompt.txt)" --seconds 7.5 \
  --ref-image singer.png --ref-audio vocal.wav --lock-audio -o take.mp4
```

The clip must cover the take; a longer one is cut to it (a slice cut to the take's
duration often encodes to one latent more). The flag joins the prepared-DiT cache key,
since the zeroed audio schedule is baked into the loaded model. It runs on the host
Euler sampler.

Verified on this branch (M5 Pro 48 GB, FL2VA weights in the Ref2VA slot): a 7.5 s
singing performance, 960x544, 6 steps, one image reference plus the isolated vocal
stem, 950 s. The mouth follows the vocal (checked by eye; a mediapipe lip-aperture
score was not reliable on singing behind a microphone, so no number is claimed). The
output soundtrack is the AudioVAE round trip of the clip (envelope r 0.995 against
the source). Production use: the nine lip-synced performance shots of a music video
at 1344x768, 12 steps, on the integration fork.

`make h3_tests`: ok, 1768 checks. Two commits, 68 added lines in h3.c, h3.h,
h3_dit.c, h3_dit.h and main.c.

Not done here: the GPU sampler path (forced to the host sampler), and muxing the
original audio file instead of the VAE round trip.



### #81 [open] PR: Video VAE: pin the default spatial tile to the checkpoint's 256 pixels
URL: https://github.com/antirez/h3.c/pull/81  by jiangzhenguo 2026-10-25T01:09:31Z

## Summary

Pins the video VAE decoder's default spatial tile to the 256 pixels the released checkpoint was trained on, instead of auto-tuning it.

Stacked on #80 — the diff here is just the one function.

The released checkpoint trains the decoder on 256-pixel spatial tiles (`vae_tile_size` in `FL2VA/video_vae/config.json`), and the RoPE table is derived from the tile's own latent size (see `prepare_rope()`), so a different tile size decodes a slightly different function than the one the model saw in training. The auto-tuned default picks 304 pixels for 960×544, which is 19×19 latent cells against the checkpoint's 16×16.

## Measurements

960×544×124, same machine and checkpoint:

| | 304 px (auto-tuned) | 256 px (checkpoint) |
|---|---:|---:|
| decoded frames, PSNR (Y) | — | **41.62 dB** between the two |
| decoded frames, SSIM | — | **0.9856** between the two |
| decoder time, FP32 | 126.09 s | 159.45 s |
| decoder time, BF16 | 35.4 s | 58.1 s |

## This is a fidelity-over-speed trade, not a quality fix

The two tilings are not perceptibly different — 41.6 dB is far above the noise floor between two runs of the same decoder under different thermal states (20.86 dB measured on this machine).

What the change costs is time. 304 pixels uses 8 tiles against 15, so it decodes about 26% less in FP32. Under BF16 the gap widens to 64%, because the fixed per-tile dispatch cost does not shrink along with the matrix multiplies, and after the BF16 port the decoder is compute-light enough for that overhead to matter.

So the argument for this change is strict fidelity to the checkpoint's training configuration, not a visible quality gain. If you would rather keep the auto-tuning, `H3_VAE_TILE_PIXELS` already exposes the same knob and the tile size can be selected explicitly.



### #80 [open] PR: Video VAE: BF16 decoder on M5 tensor cores, FP32 by default elsewhere
URL: https://github.com/antirez/h3.c/pull/80  by jiangzhenguo 2026-10-25T01:08:52Z

## Summary

The video VAE decoder can run in BF16, which puts its matrix multiplies on the Apple tensor cores. The default is hardware-aware:

* On M5-class GPUs the decoder defaults to BF16. The check is `h3_gpu_is_m5()` (the device name contains `"M5"`), the same signal `h3_gpu_create()` uses before it also applies the `H3_NAX` switch.
* On earlier GPUs, which have no BF16 matrix kernels, the decoder defaults to FP32. MPS emulates BF16 there, and that path is slower than the FP32 SIMD path.
* `H3_VAE_PRECISION=bf16` or `H3_VAE_PRECISION=f32` (also `bfloat16`, `fp32`, and `float32`) overrides the default in either direction. BF16 still halves decoder weight residency on every device, so `=bf16` is how to take that saving where FP32 is the faster default.

The default follows `h3_gpu_is_m5()` rather than `h3_gpu_has_int8_mlp()` / `tensorOpsEnabled`. Those helpers are also false when `H3_NAX=0` or the TensorOps library fails to compile. The decoder uses MPS BF16 matmul and SDPA, not those kernels, so a DiT diagnostic flag should not change its precision.

The decoder is the same 36-layer / 2048-wide ViT here and in [VPIPE](https://github.com/tgo-app-dev/vpipe); the difference was precision. MPS's FP32 matmul is a plain FP32 SIMD path, so on an M5 Pro the decoder was running at 6.2 TFLOP/s — which is the FP32 SIMD peak (20 cores × 128 FLOP × 2 × 1.26 GHz), i.e. no tensor core was involved at all.

The separate MPS FP16-operand calibration on `FengGuanyun/h3.c` `feature/mps-fp16-operands` is left out. It times BF16 against FP16 graphs at startup and is independent of this default.

## Measurements

### M5 Pro

M5 Pro (MacBook Pro 16"), 960×544×124 frames, 6 steps, seed 42, `--layers 50 --reuse 1 --core-reuse 1`. Both runs start after a 10 minute cooldown and use the same tile layout, so precision is the only variable:

| | FP32 | BF16 |
|---|---:|---:|
| video VAE decode | 159.45 s | **47.33 s** (3.37×) |
| end to end | 409.95 s | **316.93 s** (1.29×) |
| video VAE peak allocation | 9.37 GiB | **4.68 GiB** |

A standalone micro-benchmark at the decoder's real shapes (2048→6144, 2048→16384, 8192→2048, and the SDPA) gives 6.2 TFLOP/s in FP32 against 26 TFLOP/s in BF16. The FP32 number is the SIMD peak, the BF16 number is the tensor core peak.

For reference, VPIPE decodes the same clip in 45.01 s in the same session.

### M2 Ultra

Measured by @FengGuanyun on an M2 Ultra (192 GB, macOS 26.2) with `h3_video_vae_tests`. Three runs per configuration; the spread is under 2%. This GPU has no BF16 matrix kernels, so an unconditional BF16 default is slower than FP32 here. The peaks match the M5 Pro table, and accuracy does too (rel-L2 about 0.00506, against 0.00512 on the M5 Pro micro-benchmark below).

| latent | frames | F32 decode | BF16 decode | speedup | F32 peak | BF16 peak |
|---|---|---:|---:|---:|---:|---:|
| 8x12 (128x192) | 22 | 0.260 s | 0.398 s | **0.66×** | 9.16 GiB | 4.58 GiB |
| 32x32 (512x512) | 22 | 3.055 s | 4.920 s | **0.62×** | 9.45 GiB | 4.73 GiB |

An MPSGraph matmul at these shapes measures 20.8 TF/s in FP32 and 12.59 TF/s in BF16 on that machine: the FP32 SIMD peak against an emulated BF16 path. The memory drop, from 9.45 GiB to 4.73 GiB, does not depend on arithmetic speed, so it holds on every device. The default on this hardware is now FP32. `H3_VAE_PRECISION=bf16` still selects the smaller residency.

## Accuracy

Relative error against the FP32 decoder is 0.0048 on the M5 Pro clip (VPIPE's own decoder test allows 0.02). On the decoded video, BF16 against FP32 gives 23.05 dB PSNR / 0.836 SSIM — closer than two runs of the unmodified decoder differ from each other under thermal drift alone (20.86 dB / 0.842). The RGB range stays within [0, 1]. On the M2 Ultra, @FengGuanyun measured rel-L2 about 0.00506.

I also checked that the refactor is numerically neutral on the FP32 path: forcing `H3_VAE_PRECISION=f32` and the old tile size reproduces the pre-change binary's output to 23.12 dB, and the profile counters are identical (`submissions=56 direct=12264 linear=8120 attention=2016`).

## What changed

* `h3_scale_add_bf16` and `h3_video_qkv_rope_bf16` in `h3_shaders.metal`, with C wrappers in `h3_gpu.m` and declarations in `h3_gpu.h`. They mirror their FP32 counterparts line for line, including the half-split RoPE sign handling and the QK RMS epsilon placement, and reuse the existing bf16↔f32 helpers. Both are added to the pipeline warm-up list.
* `h3_weight_load_f32_as_bf16()` in `h3_weights.c`. The released checkpoint stores all 560 decoder tensors as FP32, so they are converted on the host with round-to-nearest-even before upload. This costs 0.18 s for the full 9.7 GiB set against 112 s saved, and resident weights drop from 9.16 GiB to 4.58 GiB.
* The 11 decoder ops switch to their BF16 variants, and the activations to BF16 tensors, when that precision is selected. The MLP keeps its bias, so it stays split into linear → swiglu → linear; the fused fc1+SwiGLU NAX kernel does not take a bias.
* `vae_prefers_bf16()` returns BF16 by default only when `h3_gpu_is_m5()` is true. The decision is made after `h3_gpu_create()`, before weights are uploaded.

Everything else is untouched: the DiT, the audio VAE, and the spatial tile selection all behave exactly as before.

## Compatibility

With `H3_VAE_PRECISION` unset, M5-class GPUs use BF16 and every other GPU uses FP32. `H3_VAE_PRECISION=f32` forces the original FP32 path. `H3_VAE_PRECISION=bf16` forces BF16, including where FP32 is the default. The comparison test below sets the variable for both decodes, so it does not depend on the machine.

## Tests

`tests/test_video_vae.c` decodes the same pseudo-random latent in FP32 and in BF16 and reports timing, relative error, absolute error, RGB range, and the decoded frame count and resolution. It forces both precisions through `H3_VAE_PRECISION`. It is wired into `make test` and skips when the released weights are not installed.

`tests/test_gpu_bf16_ops.c` checks the two new primitives against their FP32 counterparts in three layers: accuracy against a BF16-truncated input, a bit-exactness check where both paths consume the same values, and a double-precision host reference. It also probes the RoPE rotation and the RMS epsilon placement directly. Twelve deliberately broken copies of the shaders were checked to be caught by the assertions.

The semantic and real-video VAE parity tests compare against MLX frames with a 0.05 relative bound. Both precisions stay inside that bound (measured rel-L2 is about 0.005), and the dispatch counts match on both paths, so those tests do not assume one unconditional default.

```
$ ./h3_video_vae_tests /path/to/MiniMax-H3
video VAE decode: 22 frames at 128x192
  F32 : load   0.694 s, decode   0.598 s, GPU   0.560 s, 145 MPS linears, 36 MPS SDPA, peak 9.16 GiB
  BF16: load   0.877 s, decode   0.199 s, GPU   0.172 s, 145 MPS linears, 36 MPS SDPA, peak 4.58 GiB
  BF16 decode speedup: 2.81x
  BF16 vs F32: rel-L2 0.005121, max |delta| 0.0332994, mean |delta| 0.00172416
  RGB range: F32 [0, 1], BF16 [0, 1]
ok: the BF16 video VAE matches the float32 decoder
```


> **Comment by FengGuanyun (2026-11-19T19:19:15Z):**

Measured this branch on an M2 Ultra (192 GB, macOS 26.2) with
`h3_video_vae_tests`, because the numbers above are from an M5 Pro and the two
do not agree in sign. Three runs per configuration; the spread is under 2%.

## The decode is slower here, not faster

`vae_prefers_bf16()` returns 1 unless `H3_VAE_PRECISION=f32` is set, so BF16 is
the default on every device. On an M5 Pro that is a win because BF16 reaches
the tensor cores. This GPU has no BF16 matrix kernels, so MPSGraph runs the
BF16 matmul on an emulated path that is slower than the FP32 SIMD path it
replaces.

| latent | frames | F32 decode | BF16 decode | speedup | F32 peak | BF16 peak |
|---|---|---:|---:|---:|---:|---:|
| 8x12 (128x192) | 22 | 0.260 s | 0.398 s | **0.66x** | 9.16 GiB | 4.58 GiB |
| 32x32 (512x512) | 22 | 3.055 s | 4.920 s | **0.62x** | 9.45 GiB | 4.73 GiB |

The peaks match the ones in the table above exactly, so the two measurements
are comparable. Accuracy does too: rel-L2 0.00506 against 0.00512 reported
there.

For the arithmetic behind it, an MPSGraph matmul at these shapes measures
20.8 TF/s in FP32 and 12.59 TF/s in BF16 on this machine, which is the FP32
SIMD peak against an emulated BF16 path.

## The memory result is a win regardless

9.45 -> 4.73 GiB peak does not depend on arithmetic speed, so it helps every
device, including this one. That part is unambiguously good.

## Suggestion

The problem is only that the default is hardware dependent. Two options:

* Cheapest: default to BF16 only where the tensor cores exist. The DiT
  already has this question answered, `h3_gpu_has_int8_mlp()` and
  `h3_gpu_has_nax_mlp()` both return `gpu.tensorOpsEnabled`, which
  `h3_gpu_create()` sets only when the device name contains "M5".
* More general: measure both once and keep the faster one, rather than naming
  the hardware.

I have a branch doing the second thing for the MPSGraph BF16 graphs
(`FengGuanyun/h3.c` `feature/mps-fp16-operands`): at context creation it
encodes one matmul and one SDPA in BF16 and in FP16, times them, and keeps the
faster one with a 5% margin. It is calibration, so it needs no device table.

It happens to interact with this PR, because `vae_linear()` calls
`h3_gpu_linear_bf16()` and `vae_sdpa()` calls `h3_gpu_sdpa_bf16()`, which are
the entry points that branch instruments. Stacked on top of this PR, same
machine and same three runs:

| latent | BF16 decode as-is | BF16 decode with FP16 operands | vs F32 |
|---|---:|---:|---:|
| 8x12 (128x192) | 0.398 s | 0.251 s | 0.95x |
| 32x32 (512x512) | 4.920 s | 2.853 s | **1.06x** |

So the 512x512 decode goes from 38% slower than FP32 to 6% faster, 1.72x
faster than the BF16 path alone, and rel-L2 is unchanged at 0.00506 because a
BF16 operand converts exactly into FP16 and both accumulate in FP32. The peak
memory stays at the BF16 numbers.

One thing to know if you ever stack something like that yourselves: the
decoder's MLP passes a bias, and a graph that casts its input and weight to
FP16 but adds the bias as BF16 will not compile, MPSGraph rejects a mixed
`mps.add`. The DiT never passes a bias to `h3_gpu_linear_bf16()`, so this only
shows up on the VAE path.

None of this argues against the change. The `H3_VAE_PRECISION` escape hatch is
the right shape and the memory saving is real everywhere; it is only the
unconditional default that does not hold on hardware without BF16 matrix
kernels.




### #78 [open] PR: Video VAE: keep auto tile search in the 256-304px stable band (34% less temporal flicker at 768-class, +8% decode score)
URL: https://github.com/antirez/h3.c/pull/78  by bluetorch 2026-10-23T16:44:17Z

## Problem

Rendering fine-textured, static-camera shots (woven bamboo/thatch interiors, oil-paint surfaces; both T2VA and FL2VA) at canvases near the 768p cap shows a temporal shimmer/boil (with no other distinguishing settings) — the whole textured frame wobbles frame-to-frame, the figure ghosts during motion, and a faint grid appears on smooth surfaces — *only* when the canvas is large. The same prompt at 480x864 is clean. A user-visible case: "sky-bright fine texture boils" on a static nipa-hut interior (768x1344, 10s); a second no-human case (1344x768 office FL2VA) reproduces it.

## Diagnosis

The video VAE tile is a *temporal-stability gain stage*, not purely a throughput shape. Methodology: static-shot renders with lossless PPM frames-dir; per-pixel temporal high-pass (residual after a 5-tap temporal smoother) measured over matched static content.

- Decoding **one fixed set of DiT latents** at 1344x768 (73 frames, identical latent stream) while sweeping the tile:

| tile px   | 240  | 256  | 272  | 288  | 304  | 320  | 352  | 384  |
|-----------|------|------|------|------|------|------|------|------|
| flicker   | 1.06 | 0.51 | 0.53 | 0.56 | 0.65 | 1.09 | 2.30 | 2.90 |

- Per-frame flicker time-series across those configs correlate **r = 0.93-0.99** (89% shared variance) while different-latent renders correlate r = 0.42: the decoder supplies the *gain*, the denoiser supplies the signal. The tile auto-search's score `tiles * px^2` unknowingly picked the hot 320px shape exactly at 768-class canvases, which is why only top-resolution renders boiled and 22/56-frame fox tests looked fine.
- Ruled out by controlled A/B at one factor: int8-vs-BF16 numerics (`--use-slower-*` set, all-BF16 keeps 1.06 vs 1.09), CPU Euler sampler (1.07), sampler/step count, and VAE chunk seams (flicker flat from chunk 0 across all 14 chunks; crossfade zones equal chunk midframes).
- Confirmed in a second content domain (1344x768 office FL2VA, no subject): static-region flicker 1.3-2.4x the 480-class render with blurry regions identical.

## Fix

`configured_tile_pixels()` searches tile shapes 256..320 by throughput score; the permit band is now 256..304. Small canvases keep their prior optimum (480x864: auto 272 unchanged), 768-class canvases now pick 304 instead of 320.

- `H3_VAE_TILE_PIXELS` remains fully parameterized, so the old 320px shape (or any other) stays reachable for close-reference diagnosis.
- Verbatim user-config A/B (768x1344 portrait, 10s = 243 frames, 50 steps, seed 42): wall-region flicker 0.60-0.93 -> 0.30-0.38, landing at/below the same-content 480x864 render (0.32-0.46); face/hands regions drop 1.2-3x.
- Cost: 15x320px -> 18x304px tiles at 768-class, +8% of the video-VAE score model; measured on a 73f/50-step A/B: 52 -> 55 min total render (VAE is a small share of the run).
- `make test` passes (deterministic host suite; MLX fixture-dependent rows skipped as not installed).

## Honest limits

- The latent-level per-frame noise remains: 768-class denoiser latents are measurably noisier than 480-class ones (the tile fix closes most of the visible gap at 768-class, but the small canvas keeps a small edge; checkerboard-on-skin in still frames is a separate low-amplitude static pattern).
- The out-of-band shapes (240/352/384) are still reachable via the env var and are documented in the README as a quality tradeoff.

## Why default rather than opt-in

This is not a new fast path: the old auto-search landed on the artifact-triggering
320px shape only at 768-class canvases, and no user could reach the clean shapes
without already knowing `H3_VAE_TILE_PIXELS` existed. The full old behavior
remains one variable away (`H3_VAE_TILE_PIXELS=320`, or pin any other shape), and
the README note keeps the tradeoff documented. If the preference is still to
gate it, the same clamp behind e.g. `H3_VAE_TILE_STABLE=1` is a two-line change.

## Repro

```
# static-texture probe, both cleans vs. opts identical latents:
./h3 -d ./MiniMax-H3 -p "<static woven-wall prompt>" --seconds 3 --steps 50 \\
  --width 1344 --height 768 --frames-dir out/H1 --frame-dir ...
H3_VAE_TILE_PIXELS=320 ./h3 -d ./MiniMax-H3 -p "$same" ... # flicker ~2x the default
```




### #77 [open] PR: Fix opt-in large SDPA corruption on M4 Max; add CPU-reference probe
URL: https://github.com/antirez/h3.c/pull/77  by flysheep-ai 2026-10-23T02:58:13Z

## Problem and change

On an Apple M4 Max (36 GiB, macOS 15.7.7), a native 1344x768 / 39-frame / 20-step FL2VA render completed successfully but produced full-frame colored noise. The same prompt, seed and steps at 512x512 / 39 frames produced a coherent fox. The failing high-resolution DiT uses 12,256 sequence rows.

This adds **opt-in non-causal SDPA query splitting** through `H3_SDPA_MAX_QUERY_ROWS=2048`, with a default split gate of 8192 rows. Each query block still attends to every key/value; no model layers, resolution or denoising steps are dropped. Unset/zero keeps existing behavior. Causal attention is unchanged. The effective block participates in the graph cache key, and `--profile` reports it.

Related to #16 and #64. Credit to the query-splitting investigation/workaround in those discussions. The additional evidence here is that this also reproduces on M4 Max/macOS 15, **below #64's default 12288-row gate**. This PR includes a weight-free CPU-reference probe so the failure and workaround can be checked independently of a full generation. The 8192 gate is a conservative choice for the observed machine, not a claimed universal hardware limit. If preferred, the reproducer and M4 findings can be incorporated into #64 instead.

## Reproduction and validation

Base: `8974cc055ea9c02fcd14cc27dfda3e1027c05153`. Official original BF16 FL2VA shards, SHA-verified; SSD streaming, no repacking or quantization.

```sh
make h3_sdpa_large_test
H3_SDPA_MAX_QUERY_ROWS=0 ./h3_sdpa_large_test 12256
H3_SDPA_MAX_QUERY_ROWS=2048 ./h3_sdpa_large_test 12256
```

The probe uses 56 heads × 128 channels, deterministic random BF16 Q/K/V, and a double-precision CPU reference using the same rounded input values. It checks all channels at first/middle/last rows and heads. It is deliberately excluded from `make test` because of memory requirements; these are sampled results, not an exhaustive attention tensor comparison.

| Rows | Query splitting | Max absolute error | Relative RMS error | Result |
| --- | --- | --- | --- | --- |
| 8192 | off (below gate) | 0.0001057682 | 0.002592436 | PASS |
| 12256 | off | 0.03598442 | 1.0 | FAIL, no GPU error |
| 12256 | 2048 | 0.00009574344 | 0.002576932 | PASS |

Earlier probes at 12200 rows also failed silently without splitting. At 12500 rows the unsplit path aborted with an MPS temporary-array-too-large assertion, while splitting passed (relative RMS 0.002578).

End-to-end command, run from the shader/repository directory:

```sh
H3_SDPA_MAX_QUERY_ROWS=2048 ./h3 -d /path/to/MiniMax-H3 \
  --ssd-streaming --profile --layers 50 --reuse 1 \
  --width 1344 --height 768 --frames 39 --steps 20 --seed 42 \
  -p "A red fox walks slowly through fresh snow in a pine forest. Natural daylight. Audio: soft footsteps and light wind, no speech, no music." \
  -o fox.mp4
```

With splitting, this produced a coherent fox in the first/middle/last inspected frames, replacing the full-frame noise. FFprobe confirmed 1344x768, 39 frames, 1.625 seconds; full video/audio decoding passed. Elapsed time was 1468.11 seconds and `/usr/bin/time -l` reported peak memory footprint 13,313,423,360 bytes. The previous unsplit run took 2179.11 seconds and produced noise. These are single runs, not a controlled performance benchmark. Audio listening quality and all possible visual artifacts were not evaluated.

The full render used the same query-splitting implementation before the environment parsing/cache-key cleanup; the final source was rebuilt and the 12256-row off/on and 8192-row probes above were rerun after that cleanup.

Other validation: `make -j8 h3 h3_tests h3_sdpa_large_test` completed without warnings; `./h3_tests` passed 1768 checks; `git diff --check` passed. Fixture-dependent parity suites were not run. Only the stated M4 Max machine was validated. Query splitting can increase memory/time on other configurations; it remains opt-in.



### #76 [open] PR: Motion context: continue a take from the previous one's latent tail (--save-context / --motion-context)
URL: https://github.com/antirez/h3.c/pull/76  by Bambushu 2026-10-21T12:24:06Z


This adds clip chaining to h3.c: a second take that continues the first one's motion and sound instead of restaging the scene.

**Usage**

```sh
./h3 -d ./MiniMax-H3 -p "$(cat beat1.txt)" --seconds 5 --save-context beat1.h3ctx -o beat1.mp4
./h3 -d ./MiniMax-H3 -p "$(cat beat2.txt)" --seconds 5 --motion-context beat1.h3ctx -o beat2.mp4
```

`--save-context` writes the take's final video and audio latents. `--motion-context` pins that file's last 22 frames (7 latent steps on the 1,4,4,4,4 grid) as visual condition rows on the new take's own first step times, and its last 0.917 s of sound as audio condition rows that end at the join. The pinned head is trimmed from the delivered clip, so a `--seconds 5` continuation delivers 22 frames less than requested and the two files concatenate. A finished MP4 is accepted too: its tail is decoded and re-encoded with the released VAEs in 17 + 5 frame clips, and its sound window is taken from the clip's own audio grid rather than the soundtrack, whose AAC padding runs a third of a step past the picture (that off-by-0.69-step window was the difference between a cut and a continuation).

**What changes in the engine**

- `h3_host.c`: the packed layout gains a head anchor and an audio keyframe (`head_steps`, `head_audio_t`, `head_audio_offset` in `h3_layout_spec`). Keyframes and the head may now sit alongside Ref2VA references, timed from the target origin after the reference blocks, as ComfyUI's `PackedLayout` does. The old "keyframes and references are mutually exclusive" rule goes; `--first-frame` is replaced by the pinned head, `--last-frame` and `--ref-*` ride along.
- `h3_motion_context.c` (new): context file format, latent slicing, the video route, the audio window arithmetic, the output trim.
- `h3_ffmpeg.c`: a tail reader that keeps the last N decoded frames of a stream in a ring.
- `h3.c`: the head rows lead the condition buffers, then anchors, then references; seeded augmentation covers the head like a reference video; `--save-context` writes before the audio decode; the delivery drops the head.
- `main.c`: the two flags. Everything else is unchanged; dense runs without the flags are bit-identical.

**Measurements** (M5 Pro 48 GB, 544x960, 124 frames, 8 steps; join scored as the frame-to-frame mean absolute difference inside the clips, sd about 1.5 levels, plus the correlation of A's last and B's first frame)

| Chain source | Join | r(A last, B first) |
|---|---|---:|
| context file (this branch as submitted) | +0.5 sd | 0.994 (adjacent frames inside a clip: 0.999) |
| the take's MP4 | -0.1 sd | 0.994 |
| context file, Ref2VA take with an identity image, waving arm and camera push | +3.2 sd (under the clip's own max) | 0.969 |
| for comparison, `--ref-video` of the previous take | +18.6 sd | 0.841, a visible cut |

The spoken line survives the chain verbatim. The head costs about a third more denoise (358 s for the first take, 483 s for the continuation here; 7 more steps' worth of rows in every attention). Keep links at 8 to 10 s: in ComfyUI the same 22-frame pin held a 10 s join and lost a 15 s one.

The first row was measured on this branch as submitted; the other three on my fork's `integration` branch, which carries the community integration on top (same layout, file format and trim). Host tests cover the layout, the context file round trip, the slicing, the audio window offsets (0 and -1 step cases), the trim and the tail reader (`H3_FFMPEG=... make test`).

The design follows NikoDemon80's [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context); the latent-tail pin, the audio window that ends at the join and the trim are his. Thanks to jerryfane for the community integration branch this was developed on.




### #75 [open] PR: Compile direct Metal pipelines on demand and cache per GPU context
URL: https://github.com/antirez/h3.c/pull/75  by jerryfane 2026-10-14T05:57:36Z

## Summary

Compile direct Metal compute pipelines only when an operation first needs them, caching successful states in the existing `H3GPU` context. This replaces the eager kernel-name list and pipeline-construction loop with an initially empty mutable dictionary. Shader-library compilation, dispatch callers, arithmetic, shaders, generation settings, and context teardown are unchanged.

On a cache miss, look up the function and compile its pipeline. Missing functions and compilation failures set the existing GPU operation error (`cannot build …`) and return `nil`; failed pipelines are not cached. Such failures occur at first use rather than during context creation. No global cache or concurrency contract is introduced.

This contribution changes only `h3_gpu.m` and the existing README. It is independent of parser PR #74 and includes no parser, compact-QKV, DiT, guardian, serving, or shader changes.

## Previously measured evidence — not a benchmark of this PR branch

The 2026-10-14 prospectively fixed study compared a private installed baseline with its on-demand-pipeline candidate. **Both measured variants retained the same existing parser and compact-QKV changes.** The only enrolled native-source delta was `h3_gpu.m`; other enrolled native sources/headers and shader bytes were unchanged. This PR applies only that pipeline change to upstream `8974cc055ea9c02fcd14cc27dfda3e1027c05153`, preserving unrelated upstream code. Do not interpret the measurements as runs of this upstream branch.

M5 Ultra; fresh process for every run; 864×480, 56 frames at 24 fps, 8 Euler steps, 50 layers, requested BF16 precision, seed 17, guidance 1, `--reuse 1`, no CLI quantization request or references, generated stereo 32 kHz audio. Prompt: “A geometric folded-paper sculpture slowly rotates on a plain light background. Static camera, soft studio lighting.” No negative prompt. Requested precision is not independent attestation of backend compute precision.

Each cohort ran **baseline warmup → measured baseline → candidate → unchanged baseline control** in fixed order. Three separate baseline warmups were retained but excluded from all measured medians.

| Cohort | Separate baseline warmup (ms) | Measured baseline (ms) | Candidate (ms) | Unchanged control (ms) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 56841 | 56645 | 49123 | 56907 |
| 2 | 56853 | 58429 | 57443 | 57909 |
| 3 | 57921 | 58079 | 49951 | 57821 |
| Measured median | excluded | 58079 | 49951 | 57821 |

Descriptive candidate median reduction: **13.99473131424439% versus baseline**, **13.61097179225541% versus unchanged control**. Candidate variability remains substantial: **49.123–57.443 seconds**. Timing is original guarded native-root host elapsed time, including codecs, excluding terminal root cleanup, subsequent full-media inspection, and Studio restoration. **It is not serving API latency.**

All **12 complete clips**, including warmups, had identical full decoded video and audio hashes:

- RGB24: 56 frames, 69,672,960 bytes, SHA-256 `93a2e78c0713eb112727e739822071fc05be12d0576077a02ab8ae4aa10c02ff`.
- PCM: 74,400 stereo sample frames at 32 kHz, 595,200 bytes, SHA-256 `7fe63516174e079822985139c966b38860347226ec470f62d5b620463038c2e4`.

This establishes decoded-output identity for this fixed workload, not general numerical or perceptual parity. No audio-listening claim is made.

### Limitations and provenance

Three fixed-order triplets, not randomized, not a universal or statistically established speedup. Cache policy was unspecified. oMLX was unmanaged/unverified; whole-host exclusivity and a whole-model-payload fingerprint were not established. Earlier failed studies were retained and excluded, not silently replaced with successful samples. The owner explicitly approved a 1 GiB filesystem-growth budget; the other guards, including the 10 GiB free-space floor and one-second authority-freshness rule, were unchanged. The surrounding harness full suite was not all green: its report retains 118 passes and one listener-accounting failure in the serialized drain-fix run. That is not an H3 upstream test result.

SHA-256 provenance:

- Measured baseline `h3_gpu.m`: `4a41d398bd169a651cce8ab7cba85497811d4adaf9ee6a8428e50924302941e8`.
- Measured candidate `h3_gpu.m`: `f0edc230f99cfdd0865808bfbc9f8bd13bcef21b976d0b725a63008f3a91ff70`.
- Original report `h3-lazy-pipeline-approved-disk-2026-10-14T044542870Z/reviewed/comparison-result.json`: `5d73a241a396cc48a2b096e41304a3ade0635d3883476e346996ac29176533c2`.
- Original fixed study plan: `cb9afcd8787d0bec9c2cecacab8b5d31fc3d1dbb3a4a11458f7fde283f22cb8d`.

Public, commit-pinned evidence:

- [All twelve original samples, complete RGB/PCM identities, source bindings and limits](https://github.com/jerryfane/local-llm-performance-lab/blob/21fa3c0a51688b4441f1e79b50e4cf14e37d61d0/runs/20261014T-h3-lazy-pipelines/evidence.json).
- [Separate actual-serving API, full-media and browser playback proof](https://github.com/jerryfane/local-llm-performance-lab/blob/21fa3c0a51688b4441f1e79b50e4cf14e37d61d0/runs/20261014T-h3-lazy-pipelines/deployment-proof.json).

These sanitized files remove private paths and process identifiers. The full original report remains in the submitter's local archive; its SHA-256 is bound into the public extraction. Serving proof is separate from the measured comparison.

## Validation status

The isolated upstream branch passed `make -j2 h3 libh3.a` on 2026-10-14 (exit 0). This compiled and linked the CLI and library; it did not run the full host/Metal test suite or a model on this upstream branch. No source-text/copy/wiring tests were added.

Separately, the exact measured GPU source was built into the existing local Studio installation while preserving its parser and compact-QKV changes. One real draft API job (`44fe84ed949048fb90b9a7c9291d1964`) completed: 56 frames at 864×480 / 24 fps, stereo 32 kHz audio, seed 17. Complete software RGB/PCM decoding, gallery publication, and advancing browser playback passed. No service restart or generation-setting change. Public Studio remains Cloudflare Access protected; verification exercised its actual local backend, not public authenticated playback. This serving smoke is not another benchmark sample and establishes no API-latency improvement.

## AI-assistance disclosure

An OpenAI coding assistant assisted with this implementation, evidence transcription, README, and PR draft. The benchmark rows and decoded-media identities above come from the retained measured report rather than generated estimates. This disclosure does not claim human code review or checks that have not occurred.



### #74 [open] PR: Bound safetensors JSON string allocations to encoded string length
URL: https://github.com/antirez/h3.c/pull/74  by jerryfane 2026-10-14T03:16:23Z

## Summary

`h3_json_string()` currently reserves the entire remaining header for each decoded string. Prescan to the next unescaped quote (or end of input), then allocate the encoded string length plus its terminator. Encoded bytes bound decoded size; the existing decoder still owns validation, error reporting, and cursor updates.

The production change is seven lines (+6/-1). Add 16 behavioral cases covering empty strings, escaped quotes/backslashes, Unicode escapes, control escapes, truncated input, invalid escapes, and raw control characters. Valid cases also verify a following tensor remains readable. Add a short measurement note; no inference kernels or runtime controls change.

## Previously exercised evidence (2026-10-13)

These measurements and checks were performed on the original local candidate, not newly run on this upstream-based branch:

- AddressSanitizer + UndefinedBehaviorSanitizer parser-only execution of `test_safetensors()` and `test_safetensors_strings()`: exit 0, `ok: 177 parser checks; no Metal probe invoked`.
- Across 29 shards / 3,240 tensors, retained tensor-name buffer capacity: 106,441,680 bytes baseline and reverted control -> 163,504 bytes candidate (99.846% reduction). Measured by summing `malloc_size(tensor->name)` while retaining all 29 parsed headers, including allocator rounding. This is not allocation-request size, peak RSS or total model memory.
- Three measured header-parser runs per variant, excluding warmups: median 3.144 ms baseline / 1.305 ms candidate / 3.112 ms reverted control. Parsed-metadata hashes match.
- One fixed native warmup followed by baseline/candidate/reverted-control runs: root-host wall time 57.308 / 58.861 / 57.456 seconds. No native inference speedup is demonstrated. Full decoded RGB and PCM hashes match for all four clips, including warmup. Earlier failed cohorts are excluded.
- Native evidence came from the lab runner, not The Grill; it is not evidence of a local service deployment.

## Verification on the submitted source (2026-10-14)

The exact parser and regression-test files in this branch passed a fresh parser-only run under AddressSanitizer and UndefinedBehaviorSanitizer: exit 0, `ok: 177 parser checks; no Metal probe invoked`. The focused entry point invokes `test_safetensors()` and `test_safetensors_strings()` from `tests/test_h3.c`; Clang uses `-O1 -g -fsanitize=address,undefined`.

The local deployment CLI also built successfully with the default optimized Makefile flags. Its parser source and regression-test hashes match this branch; the deployment additionally preserves the separately installed compact-QKV change, which is intentionally excluded from this PR.

The new cases are included in the normal `make h3_tests && ./h3_tests` host suite. The full host/Metal suites were not rerun for this parser-only change.

## Public evidence

- [Original component and native samples, full decoded RGB/PCM hashes, and source-report digests](https://github.com/jerryfane/local-llm-performance-lab/blob/319d614562851bfc832e5562cb1a818bc06dea98/runs/20261013T-h3-string-allocation/evidence.json)
- [Separate actual Studio API deployment smoke, binary/source hashes, and full media/browser verification](https://github.com/jerryfane/local-llm-performance-lab/blob/319d614562851bfc832e5562cb1a818bc06dea98/runs/20261013T-h3-string-allocation/deployment-proof.json)

Deployment preserves the previously installed compact-QKV change; that unrelated change is not included here. The serving smoke is not a before/after performance comparison.

Prepared with AI assistance. Existing copyright, license, and third-party notices are unchanged.



### #73 [open] PR: Video VAE: opt-in BF16/int8 decode, weight cache and prefetch (45s -> 13s on M5 Max)
URL: https://github.com/antirez/h3.c/pull/73  by justwjx 2026-10-07T07:37:51Z

The video VAE decoder is a 36-layer, 2048-wide transformer (about 2.4B parameters) whose released weights are F32, and every block projection and the attention currently run on the F32 path. On M5 Max this makes the VAE decode of a 768x448, 124-frame clip take about 45 seconds, roughly as long as the DiT denoise of the faster presets.

This PR adds three opt-in environment variables. The default stays F32, so output is unchanged unless they are set.

### `H3_VAE_PRECISION=bf16|int8`

The residual stream, norms and RoPE stay in F32. Only the four large projections (QKV, attention output, MLP in/out) and the attention move to BF16: `h3_gpu_linear_bf16` and `h3_gpu_sdpa_bf16`, i.e. TensorOps on M5. With `int8` the projections use the existing `h3_gpu_linear_int8_bf16` path instead; without TensorOps it falls back to bf16. One small kernel, `h3_bias_bf16_to_f32`, adds the F32 bias while widening back into the residual stream.

Same latent, replayed with the new `vae_bench` tool on M5 Max (768x448, 124 frames):

| VAE precision | Decode | PSNR vs F32 |
|---|---|---|
| f32 (default) | 43.3 s | — |
| bf16 | 13.2 s | 75.3 dB |
| int8 | 10.5 s | 51.6 dB |

After 8-bit conversion, bf16 changes 7–10% of pixels by at most 4–6 levels; that is below the rounding error of the 8-bit output itself, and I could not see a difference in frame crops (faces, fur). int8 also looks the same, but its difference images show faint 16-pixel patch structure, which is why bf16 is the one I'd suggest.

### `H3_VAE_CACHE=<dir>` and `H3_VAE_PREFETCH=1`

- **Cache**: with bf16, the lowered projections are stored once in `<dir>/video-vae-bf16-<fingerprint>.cache` (4.8 GiB). Later runs read that file instead of 9.7 GiB of F32 plus GPU casts. The fingerprint covers the resolved weight directory and each shard's name, size and mtime, so changed weights rebuild the cache. The file is written under a temporary name and renamed when complete.
- **Prefetch**: loads the VAE decoder on a second thread while the DiT denoises, so the weight read overlaps GPU work.

### Diagnostics

`H3_DUMP_VIDEO_LATENT=<file>` saves the final video latent, and `make vae_bench` builds `tools/vae_bench`, which replays only the VAE decode from that file and reports timing and PSNR against a reference.

### Verification

Tested on M5 Max (64 GB, macOS 27.0.1) with the original FL2VA weights, seed 42, on top of `8974cc0`:

- **No new variables set**: draft (512², 22 frames) and 768x448 124-frame outputs have the same video and audio MD5 as unpatched `8974cc0`.
- **End to end**, 768x448, 124 frames, 20 steps, `--layers 45 --core-reuse 4`:

| Configuration | End to end | VAE |
|---|---|---|
| `8974cc0` | 131 s | 46.2 s |
| `H3_VAE_PRECISION=bf16` | 100 s | 15.8 s |
| bf16 + cache + prefetch | 97 s | (overlapped) |

- Cache and prefetch produce the same video MD5 as plain bf16, with the cache missing and with it hit. The audio MD5 is identical across every configuration, since the audio VAE is untouched.
- Prefetch saves more when the DiT itself is streamed and there is more I/O to hide. With a folded Turbo LoRA plus the int8 attention cache from #55, 6 steps, it took a 768x448 124-frame run from 93 s to 85 s.

I have only tested on M5 Max. The bf16 path uses the generic BF16 linear/SDPA entry points, so it should also run on M1–M4, but I have not measured it there.

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #72 [open] ISSUE: Long clips (≥56 frames) render structured noise for some prompts at 480×864 portrait — robust to seed, canvas size, RoPE mode and fast presets; the same prompt renders cleanly at 22 frames
URL: https://github.com/antirez/h3.c/issues/72  by php-cpm 2026-10-04T11:57:00Z


## Environment

- MacBook Pro, Apple M2 Max, 32 GB unified memory, `applegpu_g14s`, macOS 26.6.2 (25G83), Metal 4
- h3-metal 0.1.0-dev, built from `main` (source downloaded 2026-10-03) with the stock `Makefile` (`make -j8`, Apple clang 21.0.0), no local modifications
- FFmpeg 8.1.2 (Homebrew)
- Checkpoint: official `MiniMaxAI/MiniMax-H3` snapshot, `FL2VA/` subtree only, sharded files used as-released (no conversion, no repacking). `./h3 --info -d ./MiniMax-H3`:

```
Device: Apple M2 Max (applegpu_g14s)
  physical memory       32.0 GiB
  recommended GPU set   25.0 GiB
  max Metal buffer      18.7 GiB
  Apple GPU family      8
  Metal 4               yes
Native checkpoint inventory (header-only):
  Qwen3-VL encoder   14 files  1058 tensors   62.133 GiB
  FL2VA DiT          13 files   535 tensors   61.728 GiB
  video VAE           1 files   560 tensors    9.700 GiB
  audio VAE           1 files  1087 tensors    0.564 GiB
```

## Summary

With one specific prompt ("pelican riding a bicycle"), every clip of **56 frames or longer** at 480×864 portrait decodes to full-frame structured noise (exit 0, no Metal errors), while the **same prompt at 22 frames** and a **different prompt at 124 frames** render cleanly with byte-identical command lines. The failure is deterministic per (prompt, shape, seed) — re-running an identical command reproduces the identical noise pattern — and survives every knob I tried: canvas size, seed, RoPE mode, `--layers/--reuse/--token-reduction`. A 480×864 × 124-frame run of the cat prompt is clean, so this does not look like a hard shape/token-count limit; it looks like a prompt-conditioning × long-horizon divergence.

## Reproduction

**Clean baseline (same prompt, 22 frames):**

```sh
./h3 -d ./MiniMax-H3 --width 480 --height 864 --frames 22 --steps 8 \
  --layers 50 --reuse 1 --ssd-streaming \
  -p "A large white pelican with a bright orange beak rides a vintage red bicycle along a sunlit seaside boardwalk path, webbed feet on the pedals, wings on the handlebars, ocean sparkling behind wooden railings, golden hour light, camera tracks smoothly alongside at wheel level, feathers ruffling in the breeze, cinematic, ultra detailed" \
  -o outputs/pelican_22f.mp4
```

→ clean pelican-on-bicycle video+audio, 0.925 s.

**Failing (same prompt, long clip — only `--frames` changed):**

```sh
./h3 -d ./MiniMax-H3 --width 480 --height 864 --frames 120 --steps 8 \
  --layers 50 --reuse 1 --ssd-streaming \
  -p "A large white pelican with a bright orange beak rides a vintage red bicycle along a sunlit seaside boardwalk path, webbed feet on the pedals, wings on the handlebars, ocean sparkling behind wooden railings, golden hour light, camera tracks smoothly alongside at wheel level, feathers ruffling in the breeze, cinematic, ultra detailed" \
  -o outputs/pelican_124f.mp4
```

→ structured noise on every frame (120 requested, aligned to 124). `GEN_EXIT=0`.

**Clean control (long clip, different prompt, identical flags):**

```sh
./h3 -d ./MiniMax-H3 --width 480 --height 864 --frames 120 --steps 8 \
  --layers 50 --reuse 1 --ssd-streaming \
  -p "An astronaut tabby cat wearing a clear glass space helmet swims freestyle through a crystal-clear pool under a starry night sky, nebula light shimmering on the water surface, tiny bubbles trailing from its paws, fur rippling underwater, cinematic tracking shot at water level, soft moonlight, slow motion, ultra detailed" \
  -o outputs/cat_124f.mp4
```

→ clean 124-frame video+audio.

**Full matrix** (all runs on the same machine/binary/checkpoint, sequential, default seed 42 unless noted; prompts: P1 = long pelican prompt above, P2 = "A white pelican rides a red bicycle along a sunny seaside boardwalk path by the ocean, golden afternoon light, smooth cinematic tracking shot, ultra detailed", C = astronaut cat prompt above):

| # | canvas     | frames | prompt | extra flags                              | result |
|---|------------|--------|--------|------------------------------------------|--------|
| 1 | 480×864    | 22     | P1     | —                                        | clean  |
| 2 | 480×864    | 124    | P1     | —                                        | noise  |
| 3 | 448×800    | 124    | P1     | —                                        | noise  |
| 4 | 480×864    | 124    | P2     | —                                        | noise  |
| 5 | 480×864    | 124    | P2     | `--seed 43`                              | noise  |
| 6 | 480×864    | 56     | P2     | —                                        | noise  |
| 7 | 480×864    | 124    | P2     | `--layers 40 --reuse 3 --token-reduction` | noise  |
| 8 | 480×864    | 124    | C      | —                                        | clean  |
| 9 | 512×512    | 124    | C      | `--steps 4`                              | clean  |
|10 | 704×1280   | 124    | C      | —                                        | noise  |
|11 | 704×1280   | 124    | C      | `--use-reference-rope`                   | noise  |
|12 | 704×1280   | 22     | C      | —                                        | clean  |

The exact template for every run (only the flagged values vary):

```sh
./h3 -d ./MiniMax-H3 --width <W> --height <H> --frames <F> --steps 8 \
  --layers <L> --reuse <R> [--ssd-streaming] [<extra flags>] \
  -p "<prompt>" -o outputs/out.mp4
```

## Observations

- The noise is present from the first frame and fully structured (regular grid patterns), not a partially degraded render. The MP4 muxes normally with its audio track; `GEN_EXIT=0`.
- Deterministic: the two 704×1280 runs (#10, #11) produced visually identical noise patterns for the same seed, and identical re-runs of #2 reproduce the same noise.
- `--profile` of a failing run (#2) shows nothing anomalous — `BF16 SSD stream 287.827 GiB read in 64.931s (4.433 GiB/s), unhidden wait 0.006s`; Qwen text encoder wall 9.7 s, DiT Euler denoise wall 1694.5 s over 8 passes, video VAE decoder wall 121.8 s (peak 9.4 GiB); RSS stayed at 1.5–4.4 GiB throughout.
- Not a capacity issue on this machine: the clean 124-frame cat run at the same canvas (#8) used the same memory path.
- FYI: while two h3 instances ran concurrently, one failed early with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` (AdaLN block 6) — separate from this report, mentioned only in case it helps.

## Workaround that works reliably

Conditioning a continuation clip on the last frame of a clean short clip breaks the divergence:

```sh
ffmpeg -sseof -0.05 -i pelican_22f.mp4 -frames:v 1 anchor.png

./h3 -d ./MiniMax-H3 --width 480 --height 864 --frames 56 --steps 8 \
  --layers 50 --reuse 1 --ssd-streaming \
  --first-frame anchor.png -p "A white pelican rides a red bicycle along a sunny seaside boardwalk path by the ocean, golden afternoon light, smooth cinematic tracking shot, ultra detailed" \
  -o outputs/pelican_clip2.mp4     # clean
```

Chaining 22f → 56f → 39f this way produced a clean 117-frame (4.9 s) portrait video, while a monolithic 120-frame run of the same prompt never did.


## Success Case
 — 480×864 portrait, 8 steps, English prompt, then lanczos upscale to 704×1280 (same canvas as the failing pelican runs):**

```sh
./h3 -d ./MiniMax-H3 --width 480 --height 864 --frames 120 --steps 8 \
  --layers 50 --reuse 1 --ssd-streaming \
  -p "An astronaut tabby cat wearing a clear glass space helmet swims freestyle through a crystal-clear pool under a starry night sky, nebula light shimmering on the water surface, tiny bubbles trailing from its paws, fur rippling underwater, cinematic tracking shot at water level, soft moonlight, slow motion, ultra detailed" \
  -o outputs/cat_480x864_124f.mp4

ffmpeg -i outputs/cat_480x864_124f.mp4 \
  -vf "scale=704:1280:flags=lanczos" \
  -c:v libx264 -preset slow -crf 17 -c:a copy -movflags +faststart \
  outputs/space_cat_swim_720p_5s.mp4
```

→ clean 124-frame (5.17 s) video+audio at 480×864, upscaled to 704×1280. Note this is row #8 in the matrix above: identical canvas, flags and seed as the failing pelican runs (#2, #4) — only the prompt text differs.

## Questions

- Is there a known practical limit on video rows / packed sequence length (or on the native 256 RoPE adaptation range) beyond which conditioning-dependent divergence is expected?
- Could the native 256 RoPE adaptation be interacting with conditioning content over long horizons? Happy to share noise frames, full `--profile` logs, or run any targeted experiment you suggest.


### #71 [open] ISSUE: Thank you: H3.c + Storyboard
URL: https://github.com/antirez/h3.c/issues/71  by crashtestoz 2026-09-27T07:34:02Z

Firstly thank you for putting together these libraries. As many of us I started by rendering images and video using command line prompts but that quickly got complex. So I decided to build a simple Storyboard based UI (FREE to use) and interfaced it with your H3.c library.

<img width="1672" height="941" alt="Image" src="https://github.com/user-attachments/assets/7a468c36-6be9-449b-93f1-758bf2d57c92" />

I ran a few tests on my Mac mini M4 Pro 20 GPU + 48GB RAM

960x544 8sec 8 Step clip
VPIPE -> 1hr 18min
H3.c -> 1hr 8min 13% faster!

Thank you!

Here is the Storyboard repo for any one that is interested [https://github.com/crashtestoz/storyboard](url)

> **Comment by Henninges (2026-09-27T10:32:05Z):**

there is no repo...


> **Comment by crashtestoz (2026-09-27T21:00:48Z):**

https://github.com/crashtestoz/storyboard this is the repo link



### #70 [open] ISSUE: Rectangular mosaic texture in dense fur at 1344x768 is h3.c-specific (independent impl renders clean strands)
URL: https://github.com/antirez/h3.c/issues/70  by dominicletz 2026-09-26T17:35:54Z

## Human Preamble

I've experienced a very heavy mosaic structure on the generated videos using h3.c on my hardware. I'm surprised that nobody else reported this yet, as I couldn't really find a work-around configuration wise. I wonder if this is related to my Apple hardware or my copy of the model? Every video has those. I've generated below report on this and attached a sample side-by-side of the issue.


## Summary

Dense fox-fur regions at 1344x768 render with a rectangular mosaic texture: squarish dark blotches with hard axis-aligned edges instead of coherent fur strands. It is present in the **raw pre-encode output** (`--frames-dir` PPM), so it is not x264. A second, independent Metal implementation (vpipe, own kernels, own VAE decode) renders the same prompt/geometry/seed with clean directional fur strands — so this looks h3.c-specific (DiT kernels or the video-VAE decode path), not model-inherent.

## Repro

- Commit: `8974cc0` (predates PR #63)
- Device: Apple M5 Pro, 48 GB, Metal 4 (`h3 --info`: applegpu_g17s, GPU family 10)
- Model: MiniMax-H3 FL2VA, `-d ~/models/MiniMax-H3`
- Must run with cwd at the shader dir (h3 loads `h3_shaders.metal` from cwd)

```sh
cd ~/src/h3.c && ./h3 --profile -d ~/models/MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 1344 --height 768 --frames 22 --seed 42 \
  --steps 20 --layers 50 --reuse 1 --use-slower-bf16-mlp \
  --frames-dir /tmp/raw-frames -o /tmp/fox.mp4
```

Look at the fox tail in any raw frame (e.g. frame 21): the mid-tail fur breaks into discrete dark squarish spots with visible block edges, roughly 15-30 px, instead of strands. Also visible (weaker) in `exact-default` and `balanced` configs.

## Ruled out

- **Denoising steps**: identical mosaic at 4 / 20 / 50 steps (50-step sample kept).
- **Precision**: int8 default and BF16-MLP (`--use-slower-bf16-mlp`) show identical structure.
- **Encoder**: raw PPM frames show it; the old x264 defaults (fast/CRF 18) added *extra* blocking on top, fixed separately by encoding slow/CRF 14 — the underlying mosaic remains.
- **VAE tile seams**: documented tiles are 256-320 px with 64 px overlap, an order of magnitude coarser than these blocks.
- **Model**: vpipe cross-check below.

## Cross-check: vpipe renders clean fur

vpipe (`tgo-app-dev/vpipe`, independent Metal kernels, own MiniMax-H3 loader/VAE decode) at the same prompt, 1344x768, 22 frames, seed 42, 16 steps, DiT + text encoder quantized to 8-bit group-64 **from the same local weights** (registered the diffusers dir directly, skipped the Comfy repack download). Its raw pre-encode PNGs show coherent directional fur strands with smooth tonal gradation in the same tail region — no blocks. Note vpipe runs *lower* effective precision (8-bit + lossy i8_gemm) than the h3.c BF16-MLP run that shows the mosaic, so precision cannot explain the difference either. (Poses differ between the two implementations, so this compares texture character, not pixels.)

## Suspects

Remaining candidates on the h3.c side: DiT attention/MLP Metal kernels (note this checkout predates the PR #63 GQA-barrier fix — I also observe same-seed output variance consistent with issue #52) or the video-VAE decoder path at 768p. The related texture artifacts already documented in the README (woven texture in rejected schedules, 256 lattice fixed via RoPE halving) suggest this could be another member of that family.

> **Comment by dominicletz (2026-09-26T17:43:41Z):**

Side-by-side, as promised — same tail region, raw pre-encode frames, full-res crop on top with 2x nearest-neighbor zoom below:

![h3.c (left, blocks) vs vpipe (right, strands), raw tail crops](https://files.catbox.moe/sudyr5.png)

Left: h3.c `bf16-mlp` frame 21 — mid-tail fur breaks into discrete squarish blotches with axis-aligned edges. Right: vpipe 8-bit frame 21 — coherent directional strands, smooth gradation. (Poses differ between implementations, so compare texture character, not pixels. Full frames + clips available on request.)


> **Comment by Eldirfar (2026-11-20T11:23:34Z):**

Additional data point, same symptom (16×16 patch mosaic, fixed to the frame grid, visible in raw frames before encoding), on M5 Max 128 GB, h3.c @ 8974cc0, weights MiniMaxAI/MiniMax-H3 @ 0919fb6 (all FL2VA/Ref2VA files match the Hub sha256).

**Measurement.** Lossless frames via `--frames-dir`. On smooth areas (|∂x|,|∂y| < 1.5 levels) I average the absolute neighbour difference per phase mod 16 (from the frame origin) and report the 15|0 boundary relative to the median phase. A clean image should sit near 0 %.

| Run (512×512, 56 frames, 8 steps, seed 7, FL2VA) | seam 15\|0, x / y |
|---|---|
| default | +57 % / +52 % |
| `H3_VAE_TILE_PIXELS=256` | +53 % / +54 % |
| `H3_VAE_TILE_PIXELS=512` (single tile) | +60 % / +64 % |
| `--use-slower-bf16-mlp --use-slower-bf16-qkv --use-slower-bf16-attention-output` | +54 % / +51 % |

768×1344, 158 frames, Ref2VA, 10 steps, `--layers 45 --reuse 2`, seed 101: +49 % / +38 % in the PPM frames; the libx264 crf 18 MP4 roughly doubles it (+94 % / +95 %) and adds an 8-px component.

**Ruled out so far**
- VAE tile size / tile blending (256, 288, 512 identical).
- int8 DiT paths (full BF16 gives the same seam).
- A static additive per-patch pattern: averaging the high-passed 16×16 patch over all 637k patches of a clip gives only ~0.34 levels RMS (peak ~1 level), so the seam is content-dependent, not a fixed offset.
- The video-VAE decoder vs. the reference `vae_vit.py` / `klvae.py` / `func.py`: normalized (t,h,w) RoPE with +0.5 offsets, 2π·100^(−k/8), rotate-half pairs (i, i+24) on the first 48 of 64 dims, QK RMS norm before RoPE, registers + zero cls token at position 0, `(c, pt, ph, pw)` unpatchify, ImageNet de-normalization, and tile blending all look equivalent to me.

The seams show up as per-patch shading on flat backgrounds and as small discontinuities across patch edges in detail (nose, fur), and stay put when the content moves. Happy to run more variants or share frames/scripts if useful.


> **Comment by Eldirfar (2026-11-20T15:31:37Z):**

Follow-up with a direct A/B against vpipe on identical weights.

**Setup**
- MiniMaxAI/MiniMax-H3 @ 0919fb6 (FL2VA, BF16 snapshot), M-series Mac, macOS 26.6
- h3.c @ 8974cc0, DiT forced to BF16: `--use-slower-bf16-mlp --use-slower-bf16-qkv --use-slower-bf16-attention-output`
- vpipe @ 4dc36b7 (`minimax-h3-text-to-video` pipeline, `i8_gemm: false`)
- Same prompt, seed 7, 56 frames, video shift 12 / audio shift 3
- Lossless output on both sides: h3.c `--frames-dir` (PPM), vpipe FFV1 yuv444p full range. No H.264 involved.

**Metric**
Mean absolute luma gradient on smooth pixels (|gx| and |gy| < 1.5/255), bucketed by position mod 16. The value is bucket 15 (the 15|0 patch boundary) relative to the median bucket. 0% = no grid. Averaged over frames 5, 15, 25, 35, 45, 54.

| run | h3.c x / y | vpipe x / y |
|---|---|---|
| 512x512, 8 steps | +53% / +48% | +34% / +35% |
| 768x1344, 20 steps | +39% / +42% | +23% / +25% |

All other buckets stay within about ±10% in both implementations. The boundary bucket is the only outlier.

**Takeaways**
1. The 16-px seam is present in vpipe too: on a high-passed background crop, both show the same 16x16 blocks. So part of it seems inherent to the patch-based video VAE decoder, and vpipe output is not fully clean.
2. h3.c consistently shows ~1.6-1.7x more seam energy than vpipe on the same weights and settings, in both runs. That excess looks implementation-specific.
3. Higher resolution and more steps reduce the seam in both implementations.

Caveat: the two runners use different RNGs, so the frames differ in content. This compares seam strength statistically, not pixel by pixel.

Happy to share the PPM frames or the measurement script if useful.



### #69 [open] ISSUE: [Community] Wild H3C — the first ComfyUI custom node for h3-metal (on the official Registry)
URL: https://github.com/antirez/h3.c/issues/69  by skaiy 2026-09-23T23:46:31Z

Hi Salvatore & community —

Following up on h3.c-studio (#65), I've now brought h3-metal into ComfyUI: **Wild H3C** — as far as I can tell the first ComfyUI custom node package built on h3.c, and the first h3-metal-based package on the official ComfyUI Registry.

🔗 Repo: https://github.com/skaiy/wild-h3c-comfyui
📦 Registry: https://registry.comfy.org/nodes/wild-h3c

**What it does**

- 🎬 **Two nodes** — *H3 Text to Video* (prompt, canvas size, 1–15 s, steps, speed presets, seed, optional first/last frame anchors) and *H3 Reference to Video* (up to 9 ordered Ref2VA images + one reference audio)
- 📈 **Live progress in the ComfyUI UI** — parsed from the CLI's stderr phase protocol (text encode → denoise → VAE → ffmpeg), and cancelling a run terminates the subprocess cleanly
- 🔌 **Standard `VIDEO` output** — plugs straight into SaveVideo and the rest of the node ecosystem
- ✅ **97 pytest cases** (preflight rules, argv building, progress parsing, cancellation) plus a real end-to-end run on an M-series MacBook Pro with 128 GB (512×512, 4 steps, ~32 s warm)

**Design choices**

- **Wraps the CLI with subprocess** instead of reimplementing anything: the engine stays antirez's, untouched, and upstream improvements drop in with a `git pull && make -j8`
- **No binaries, no weights in the package** — the README walks through building h3.c and offers two model-setup paths (point at an existing MiniMax-H3 snapshot, or download fresh), with the MiniMax community license called out prominently
- **MIT**, same as the engine

This also sits nicely next to the official MiniMax partner nodes: local h3.c for fast, free, private iteration, then the official API nodes for 2K finals — the same layering the ComfyUI team itself describes.

Thank you again for the engine — the predictable exit codes and machine-readable progress output made this integration honestly pleasant. If you'd ever consider a community-links section in the README, I'd be honored. Feedback, issues and PRs are very welcome. 🙇

---

**中文版**

继 h3.c-studio（#65）之后，我把 h3-metal 带进了 ComfyUI：**Wild H3C**——目前第一个基于 h3.c 的 ComfyUI 自定义节点插件，也是官方 Registry 上第一个 h3-metal 生态的包。

🔗 仓库：https://github.com/skaiy/wild-h3c-comfyui
📦 Registry：https://registry.comfy.org/nodes/wild-h3c

- 🎬 两个节点：文生视频（提示词/画幅/时长/步数/速度预设/seed/首尾帧）+ 参考生成（≤9 张有序参考图 + 1 条参考音频）
- 📈 ComfyUI 界面实时进度（解析 CLI stderr 的分阶段协议），取消运行会干净地终止子进程
- 🔌 标准 VIDEO 输出，直接接 SaveVideo 和整个节点生态
- ✅ 97 条 pytest 用例 + 真机端到端验证（M 系列 MBP 128GB，512×512 / 4 步 / 热缓存约 32 秒）

设计上用 subprocess 包 CLI 而不是重写任何逻辑——引擎始终是 antirez 的，上游更新 `git pull && make` 就能获得；包内不含二进制和权重，README 提供「已有权重直接指目录 / 全新下载」双路径，并显著标注 MiniMax 社区许可证。MIT 协议，与引擎一致。

和官方 MiniMax 节点是互补关系：本地 h3.c 做快速、免费、私密的迭代，定稿后接官方 API 节点出 2K 成片。

再次感谢这个引擎——可预期的退出码和机器可读的进度输出让集成非常愉快。如果 README 考虑加社区链接区，不胜荣幸。欢迎 issue / PR 🙇

> **Comment by 122786891-pixel (2026-10-29T15:22:38Z):**

我电脑是 Macbook Pro M2 MAX 64GB，Comfyui版本为0.30.0，测试了一下Ref2VA，token_reduction和SSD streaming都关闭，视频全部黑屏，同时打开token_reduction和SSD streaming，结果如下，

480*864   30步 Balanced   Match   黑屏
480*864   26步 Balanced   Match   有视频但镜头未遵从提示词，比例基本正常
480*864   20步 Balanced   Match   有视频但镜头未按提示词，且比例压扁
480*864   18步 Balanced   Match   有视频基本遵从提示词但比例压扁
480*864   16步 Balanced   Match   黑屏

480*864   30步 Precise   Match       黑屏
480*864   20步 Precise   Match       黑屏
480*864   16步 Precise   Match        黑屏

且生成的视频人物面部特征和参考图人物一致性也不足，请作者看看是哪里出的问题呢



### #68 [open] PR: Preserve Video VAE decoder cache across input changes
URL: https://github.com/antirez/h3.c/pull/68  by Siilent-Sea 2026-09-16T08:36:09Z

## Problem and scope

In the interactive session, the resident cache is fully cleared whenever the model
input changes. `h3_cli.c` calls `h3_cache_clear()` at five input-change sites:

- `clear_references()` — `!refs clear`
- `add_reference_image()` — `!ref-image`, and the other reference adders
- `remove_reference()` — `!ref-remove N`
- `set_anchor()` — `!first` / `!last`, both the `clear` branch and the set branch

`h3_cache_clear()` drops the prompt/reference conditioning, the prepared DiT **and**
the Video VAE decoder. The decoder, however, is keyed independently in
`h3_generate()` by `vae_path|latentWxH`. When that key is unchanged — which is the
common case for an anchor or reference edit, since neither changes the decoder model
path or the spatial latent shape — the decoder is still valid and is now needlessly
released. The next generation reloads it from scratch.

The change keeps the full clear as the explicit/default behavior and adds one narrower
invalidation for input changes.

## Why the change is general

- It is not Mac-, shape- or workflow-specific: the affected code is the shared CLI
  session path and the shared `h3_ctx` cache.
- It uses the decoder-key mechanism that already exists upstream
  (`h3_acquire_video_decoder()` and the key comparison in `h3_generate()`); no new
  residency policy or heuristic is introduced.
- The existing `!cache clear` command, `h3_cache_set_enabled(ctx, 0)` and
  `h3_free(ctx)` still perform full clears, so the user-visible explicit-clear contract
  is unchanged.
- Default behavior for one-shot callers is unchanged: the new function is additive and
  is only called from the five input-change sites.

## Public reproducer

Host-only and deterministic; no model weights, no GPU, no Metal device. See
`public-repro/test_cache_invalidate.c`.

## Implementation

- `h3.c`: add `h3_cache_invalidate_inputs()` — clears conditioning and the prepared
  DiT, leaves `video_decoder` / `video_decoder_key` untouched.
- `h3.h`: declare it with a contract comment.
- `h3_cli.c`: call it at the five input-change sites; keep `h3_cache_clear()` for
  `!cache clear`.

## Regression test

`tests/test_cache_invalidate.c`, 15 asserts:

- input-only invalidation clears conditioning (key, values, `conditioning_present`)
  and DiT (`dit`, `dit_key`) but keeps both the resident `video_decoder` pointer
  and `video_decoder_key` unchanged;
- `h3_cache_clear()` still drops the decoder and both keys;
- NULL-argument safety for both entry points.

## Numerical validation

Not applicable. The change alters cache lifetime, not arithmetic; no tensor path is
touched.

## Performance and memory validation

None claimed. The intended effect is to avoid a decoder reload across an input change.
No timing measurement accompanies this draft, and none is asserted.

## Platforms tested / not tested

- Tested: macOS 26.5.1 (build 25F80), Apple clang 21.0.0, arm64.
  `make -j8 libh3.a h3` builds with no warnings under the project's
  `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`; `h3_tests` reports 1768 checks OK;
  the new host test passes.
- Not tested: GPU/generation paths, non-Apple platforms, any model run.

## Risks and fallback

- Risk: a caller that relied on an input change also resetting the decoder keeps the
  decoder resident. That is intentional and matches the decoder's own key contract;
  `!cache clear` still offers a hard reset.
- Fallback: drop the `h3_cli.c` hunk; the new library function remains additive and
  harmless.

## Privacy and asset provenance

No private media, prompts, latents, hashes or paths. The patch is original project code
under the repository's existing MIT license; it adds no third-party code.

## Related issue or discussion

No existing Issue or PR found at 2026-09-16 for input-only cache invalidation
(`repo:antirez/h3.c` searches for "invalidation" and "cache invalidate" return no
match). PR #62 / Issue #25 concern non-TTY stdout flushing; PR #1 concerns VAE tile
size; PR #35 concerns decode progress. None owns this behavior.

## Contribution status

No `.github/` contribution policy, CLA or DCO requirement detected on
`antirez/h3.c`. The account holder submits; no separate sign-off action is known to be
required. Recheck immediately before submission.

## AI assistance disclosure

Submitted by the project owner with assistance from an owner-authorized AI engineering
agent. Reproduction, code and tests are included.



### #67 [open] ISSUE: [MPSNDArrayMatrixMultiplication encodeToCommandBuffer:] destination ... is too large for kernel on M3 Max 64 GB
URL: https://github.com/antirez/h3.c/issues/67  by rideapandafly 2026-09-15T08:25:19Z

**Reproduction**
Any generation where the combined video+audio token sequence grows past ~26k. Two confirmed failures, both at denoise 0/20:
A. 1024×576, 8 s (192 frames):
./h3 -d ./MiniMax-H3 -p "..." --width 1024 --height 576 --seconds 8 \
  --steps 20 --layers 45 --reuse 2 -o outputs/out.mp4

B. 896×512, 8 s (192 frames): same command with --width 896 --height 512.
Both abort with:
MPSNDArrayMultiaryKernel.mm:2644: failed assertion `[MPSNDArrayMatrixMultiplication encodeToCommandBuffer:] destination: [
<MPSTemporaryNDArray ...> Rank: 4, MPSDataTypeBFloat16, Device: Apple M3 Max
- dimensionLengths = [ 33654, 33654, 56, 1, ... ]   (case A)
- dimensionLengths = [ 26358, 26358, 56, 1, ... ]   (case B)
- strideElements   = [ 1, 33656, ... ]  /  [ 1, 26360, ... ]
] is too large for kernel.  Encode failed.'
zsh: abort

**Analysis**
The failing op is an explicitly materialized attention-score matrix: 56 heads × S×S BF16, i.e. 56 × S² × 2 bytes:

Canvas / duration | Sequence S | Score-matrix footprint
1024×576 × 8 s | 33,654 | ≈ 127 GB
896×512 × 8 s | 26,358 | ≈ 78 GB

where S ≈ (W/32)×(H/32)×⌈frames/4⌉ + audio tokens. Both footprints far exceed a 64 GB unified-memory machine. Development appears to happen on a 128 GB M5 Max, where the 78 GB case fits — which is presumably why this hasn't been caught. M3 Max also has no Metal 4/TensorOps attention path, so it falls into this MPS path even at shapes that work on M5.
Meanwhile small sequences work fine on the same machine: the README's 512×512 / 22-frame fox validation runs correctly.

**Expected behavior**
M3-class / ≤ 64 GB machines need a fallback that doesn't materialize the full S×S score matrix — e.g. a chunked/tiled SDPA (flash-attention style), or dispatching to an MPSGraph scaled-dot-product path when S exceeds a memory-derived threshold.

**Suggested immediate mitigation for users**
Keep 56 × S² × 2 ≤ ~22 GB of temp headroom (DiT at --layers 45 already resident at ~33 GB). On a 64 GB machine that means S ≲ 14,000, e.g. 864×480 × 5 s (S = 14,339) is the largest validated-geometry shape expected to work; longer videos must be produced as chained shorter segments via --ref-video.
Happy to run any diagnostics / patched builds on this machine.


> **Comment by rideapandafly (2026-09-15T09:53:09Z):**

**Update  — bisection complete, ceiling located:**

[table-1789465649688.csv](https://github.com/user-attachments/files/32235271/table-1789465649688.csv)

The cutoff lies between S = 9,122 (9.3 GB destination) and S = 12,362 (17.1 GB). Note the 17.1 GB failure cannot be explained by total-memory pressure either (17.1 GB temp + ~34 GiB resident DiT + OS overhead < 64 GB), and the 2^32 total-element hypothesis is dead (4.66B elements passed). This points to an internal MPS limit on the M3 fallback path — per-allocation quota or kernel bound — that M5 avoids by dispatching large sequences to MPSGraph.

**Impact:** on M3-class machines the practical ceiling is ~73 frames per segment at 864×480 (3 s of video). Longer clips require segment chaining. A chunked/tiled attention implementation for the non-Metal-4 path would remove the ceiling.


> **Comment by rideapandafly (2026-09-15T13:41:01Z):**

**Update 1 — silent latent corruption on M3, distinct from (and below) the MPS crash ceiling:**

[table-1789479554289.csv](https://github.com/user-attachments/files/32244507/table-1789479554289.csv)

864×480 × 73 frames corrupts silently at any step count, while the same frame count at 512×512 is clean — so the corruption threshold lies between S≈6,500 and S≈9,122, below the "too large for kernel" crash ceiling (S between 9,122 and 12,362). Both modalities corrupted → the DiT latent stream itself, not the VAE decoders. Suspect an indexing/aliasing bug in the M3 copied-buffer attention path (ring depth 2) that activates past a token threshold at the 27×15 spatial grid. Could you run 864×480 × 73f on M5 to confirm it's M3-path-specific? A cheap improvement regardless: a NaN/range sanity check on latents before VAE decode so corruption fails loudly instead of writing a broken MP4.



### #66 [open] ISSUE: Build fails on macOS 15 SDK: MTLGPUFamilyMetal4 undeclared
URL: https://github.com/antirez/h3.c/issues/66  by rideapandafly 2026-09-14T03:36:48Z

h3_metal.m:43:51: error: use of undeclared identifier 'MTLGPUFamilyMetal4'; did you mean 'MTLGPUFamilyMetal3'?
   43 |             info->metal4 = [device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
      |                                                   ^~~~~~~~~~~~~~~~~~
      |                                                   MTLGPUFamilyMetal3
/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers/MTLDevice.h:222:5: note: 'MTLGPUFamilyMetal3' declared here
  222 |     MTLGPUFamilyMetal3 API_AVAILABLE(macos(13.0), ios(16.0)) = 5001,
      |     ^
1 error generated.
make: *** [h3_metal.o] Error 1
make: *** Waiting for unfinished jobs....

Please modify lines 42-44 of h3.c/h3_metal.m as follows：
#if defined(__MAC_26_0)
        if (@available(macOS 26.0, *)) {
            info->metal4 = [device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
        }
#endif


### #65 [open] ISSUE: [Community] h3.c-studio — a bilingual web GUI for h3-metal
URL: https://github.com/antirez/h3.c/issues/65  by skaiy 2026-09-13T15:29:26Z

Hi Salvatore & community — first of all, thank you for h3-metal. The clean public API in `h3.h` (frame/progress callbacks, cache control, the full `h3_params` surface) made it a joy to build on.

I've put together **h3.c-studio**, the second web GUI for h3-metal, as a fork that adds a `studio/` directory without touching the engine itself:

🔗 https://github.com/skaiy/h3.c-studio

**What it does**

- 🎛 **Prompt studio** — Scene / Action / Camera / Look / Audio structured editing (Context-IR style), canvas sizes, 1–15s duration, speed/quality presets, seed control
- 🖼 **Conditioning UI** — first/last frame anchors and ordered Ref2VA reference images with drag upload
- 📈 **Live progress** — per-phase progress bar (text encode → denoise → VAE decode) with streaming logs, cancellable jobs, serial GPU queue
- 🎞 **Clip library** — auto-indexed generation history with in-browser playback
- 🔗 **Chain last frame** — one click extracts any clip's last frame into the next generation's `--first-frame`, which makes the multi-shot storyboard workflow (validated in the README tutorial) practical
- 🌐 **i18n** — 中文 / English switch

**Stack**: React + TypeScript + Vite + Tailwind frontend, FastAPI backend wrapping the CLI (moving to a direct `libh3.a` binding is on the roadmap, since the public API already exposes everything needed for in-process previews).

**Licensing**: engine stays MIT © antirez, untouched; everything in `studio/` is Apache-2.0. Model weights are not included and the README links to the MiniMax community license.

**What's next**: this is very much a living experiment. I plan to keep evolving the studio — direct `libh3.a` binding for in-process live previews, a multi-shot storyboard board, prompt templates, maybe a job queue with presets — I honestly don't know how far it will go, but I want to try, and I'll keep shipping in the open.

Feedback, issues and PRs are very welcome — and if you'd ever like to link it in the README, I'd be honored. 🙇

---

**中文版**

首先感谢 h3-metal —— `h3.h` 里干净的公共 API（帧/进度回调、缓存控制、完整的 `h3_params`）让二次开发非常愉快。

我做了 **h3.c-studio**——h3-metal 的第二个图形界面，以 fork 方式新增 `studio/` 目录，引擎本体零改动：

🔗 https://github.com/skaiy/h3.c-studio

- 🎛 提示词工作室：五段式结构编辑、画幅/时长/预设/seed
- 🖼 条件输入：首尾帧锚点 + Ref2VA 参考图拖拽上传
- 📈 实时进度：分阶段进度条 + 日志流 + 任务队列
- 🎞 作品库 + 🔗 末帧接力（一键用上一条末帧做下一条首帧，多镜头叙事工作流）
- 🌐 中英双语

协议：引擎保持 MIT 不动，`studio/` 为 Apache-2.0；仓库不含模型权重。

后续计划：这会是一个持续更新的实验——直连 `libh3.a` 做进程内实时预览、多镜头分镜板、提示词模板库……说实话不知道最终会做成什么样，但想试一试，会持续在开源中迭代。

欢迎 issue / PR，如果 README 愿意收录链接，不胜荣幸 🙇

> **Comment by skaiy (2026-09-14T01:38:08Z):**

**Update 2026-09-14 — Storyboard view, demo video, Turbo preset, resumable checkpoints / 更新：分镜板、演示视频、Turbo 预设、断点续跑**

Two weeks in, still shipping. What's new since the first post:

- 🎬 **Storyboard view** — multi-shot cards with per-shot prompt/canvas/seconds/seed, automatic last-frame chaining between shots, and one-click lossless concat. The workflow used to produce a 30-second three-shot noir piece (see demo).
- ⚡ **Turbo 6-step preset** — folded `larryvrh/MiniMax-H3-Turbo-Lora` into the checkpoint (via PR #14's `tools/fold_turbo_lora.py`), wired as a one-click preset and a per-shot toggle in the storyboard.
- ⏸ **Resumable checkpoints in the UI** (from PR #2) — pause a long render at step N for a sigma-zero draft, resume to finish if the direction looks right.
- 📹 **Demo video + screenshots in the README** — 28-second storyboard walkthrough, recorded against the real UI.

Also absorbed upstream community PRs #62, #35, #9, #31 (progress reporting & robustness), and published a comparison doc: [h3.c vs other stacks](https://github.com/skaiy/h3.c-studio/blob/main/docs/h3c-vs-other-stacks.md).

首次发布后的两周，持续更新中：

- 🎬 **分镜板**：多镜头卡片、自动末帧接力、一键无损拼接（演示视频就是用这个功能做的 30 秒三镜短片）
- ⚡ **Turbo 6 步预设**：Turbo LoRA 已折叠进 checkpoint，预设卡和分镜板镜头开关一键启用
- ⏸ **断点续跑界面化**：长片第 N 步暂停出草稿，方向对了再续跑
- 📹 **README 已附演示视频与界面截图**
- 同步吸收上游社区 PR #62/#35/#9/#31，并发布了[h3.c 与其他推理栈的对比分析](https://github.com/skaiy/h3.c-studio/blob/main/docs/h3c-vs-other-stacks.md)

🔗 https://github.com/skaiy/h3.c-studio



> **Comment by Henninges (2026-09-14T10:59:55Z):**

Hi @skaiy , welcome to the GUI club! 👋

Just a small correction: this isn't actually the first web GUI for h3.c — 
I published https://github.com/Henninges/h3-studio on August 16 (4 weeks ago). 
It's also bilingual (DE/EN) with Context-IR prompting, but I specialized it 
for the music-video workflow: lip-synced video from reference image + audio, 
support for 9 images / 3 videos / 3 audio refs, and Context-IR field defaults.

Nice to see the ecosystem growing! Your clip library and chain-last-frame 
features look interesting. Would be cool to compare notes on what works 
best for users.


> **Comment by skaiy (2026-09-14T14:33:19Z):**

Hi @Henninges — honestly just excited that anyone noticed this and took the time to reply, so thank you for that! 😄

And thanks for the correction — checked your repo's commit history, Aug 16 vs. my Sep 13, so this is fair and square the *second* GUI, not the first. Ha, guess I'll happily take "#2" then 😂 (going to fix the wording in my README/issue title accordingly).

Went and read through your h3-studio README — the Context-IR fields with smart defaults for empty inputs is a really nice idea, lowers the barrier for the "advanced mode" a lot instead of forcing users to fill in every field. Also love that you specialized hard into the music-video / lip-sync workflow (image + audio → synced video in one pass) with the pose-cue and Ref2VA/FL2VA-can't-mix notes — that's the kind of hard-won prompt-engineering knowledge you only get from actually using the thing a lot, and I don't have an audio-conditioning path in my studio at all yet, so there's real stuff for me to learn from your project.

Funny coincidence too — I'm also on an M-series MacBook Pro with 128GB unified memory, so we're probably fighting the same memory/cache tradeoffs. My side really started from a much simpler place: I just wanted to make it a little easier for people to actually sit down and play with h3.c without wrestling the CLI, so that's what the whole studio/ layer is for.

I honestly have no idea how far mine will end up going either — it's very much "ship what I find useful this week and see" — but I'd genuinely enjoy comparing notes with you, especially on the reference-conditioning UX and anything you've learned about progress-reporting/robustness against real usage. Happy to hop into an issue thread (or wherever's convenient for you) whenever you want to dig into specifics. Welcome to being #1 in the GUI club and looking forward to more people building on antirez's excellent engine 🙌



> **Comment by Henninges (2026-09-14T15:07:22Z):**

Haha, thank YOU for taking the correction so gracefully – "happily take #2" 
is exactly the sportsmanship that makes open source great. 😄 And respect for 
checking the commit history before answering, that's class.

Seriously: storyboard view, Turbo preset and resumable checkpoints in two 
weeks is an impressive shipping pace. Your studio and mine really are two 
halves of the same dream – yours makes multi-shot storytelling practical 
(FL2VA chaining), mine makes the music-video path practical (Ref2VA with 
audio conditioning). Neither can do the other's job today.

Since you mentioned the audio path: I'd genuinely love to share everything 
I've learned there, it might save you weeks of weird renders:
- Ref2VA audio constraints (2–15 s, always needs an image/video alongside, 
  Ref2VA and FL2VA can't mix in one run)
- prompt semantics: "sings the vocals" vs. "speaks the words" makes or breaks 
  lip-sync; speech in non-English languages is currently garbled (model limit)
- the `--token-reduction` ghosting/audio trap
- pose cues hiding in reference images (back-view pics turn your person around)

And one shared idea I've been circling: a **"song storyboard"** – cut a track 
into ≤15 s segments, render each segment with the same reference images plus 
its own audio slice (Ref2VA), then lossless-concat like your storyboard does. 
That would combine your chaining/concat engine with my audio workflow and 
finally break the 15-second ceiling for music videos. If you're interested, 
I'm happy to prototype the segment-cutting + concat side and compare notes 
in an issue thread.

Same 128 GB club over here – at least we're the two people who can render 
768p without swap anxiety. 😄

Let's keep shipping. And thanks for the kind words about the Context-IR 
defaults – they were born from exactly the frustration you describe: nobody 
wants to fill five fields just to test a prompt.


> **Comment by skaiy (2026-09-15T05:02:29Z):**

Hi @Henninges — this is genuinely some of the most useful feedback I've gotten on this project, thanks for taking the time to write it all out.

Reacting point by point on the audio notes so you know they didn't just scroll by:

- Ref2VA/FL2VA can't mix — good catch, and slightly embarrassing: my `refImages` label already *says* "exclusive w/ anchors" but I never actually wired up a guard. Backend will happily let someone pass both today and presumably eat an error at generation time. I'll add real validation for that, extended to `ref_audio` too.
- "sings the vocals" vs. "speaks the words", and non-English speech garbling — exactly the kind of thing you only learn by burning GPU-hours. Appreciate you saving me the trial and error.
- `--token-reduction` fighting with audio — noted, will flag/disable that combo once ref_audio is set.
- Pose cues leaking from reference images — makes sense given how strong identity conditioning is here. Good to know before I chase a "why did they turn around" bug thinking it's my code.

On the song-storyboard idea: I like it, and it maps cleanly onto what already exists — my board/shot/concat plumbing doesn't care what's feeding each shot, so "same reference set + per-segment audio slice" is a plausible new segment type rather than a rewrite. Two things I'd want nailed down before committing to a design:
1. Does identity actually hold across segments with a fixed reference set and no first/last-frame anchor between them (pure Ref2VA doing the work, no FL2VA chaining)?
2. What happens at the splice points once concatenated — natural audio cut points, or do we need a crossfade pass at the video level too?

Happy to open a tracking issue (my repo or yours, whichever you'd rather host it on) once you've got a segment-cutting prototype running. In the meantime I've got `--ref-audio` wired into my UI now — landed a few hours after your first comment, oddly good timing — so I'll actually run your prompt-semantics and token-reduction warnings against real hardware this week and report back what breaks.

Same 128GB club, cheers 🍻



> **Comment by Henninges (2026-09-15T13:42:44Z):**

@skaiy  — now *this* is how you do an engineering dialogue. Taking each 
point and actually wiring it up, not just nodding along. Respect. 🙌

To your two questions on the song storyboard:

**1. Identity across segments with fixed Ref2VA, no FL2VA chaining:**

Yes — and arguably better than FL2VA chaining. My Take 3 was a 10-second 
render with 4 reference angles (front / side / back / endframe), pure 
Ref2VA, no first/last-frame anchor at all. Consistency was rock solid 
throughout. FL2VA forces the model to start from a locked pixel state, 
while Ref2VA lets it choose the best pose freely — which often makes 
the result *more* stable, not less. So the plumbing should work: feed 
the same reference set to every segment, no chaining needed.

**2. Splice points — natural audio cuts or video crossfade?**

Neither, ideally. My plan for the prototype:
- **Audio:** pre-cut the song with ffmpeg at natural phrase boundaries 
  (e.g. between chorus lines), no crossfade — that just muddies things
- **Video:** FFmpeg concat with `-c copy` — lossless, frame-accurate, 
  no re-encode
- **Transition trick:** take the last frame of segment N, use it as 
  `--ref-image` for segment N+1 (NOT as `--first-frame`). That gives 
  a smooth visual handoff without the FL2VA lock-in, while the model 
  still has freedom to choose its starting pose in each segment.

Happy to host the tracking issue on my repo — I've got the music-video 
context already and it feels like the natural home. I'll open it this 
week and tag you. First milestone: cut a real song into 3-4 segments, 
render each with the same reference set + segment audio, concat, and 
see if the identity actually holds end-to-end.

And thank you for actually running my prompt-semantics and token-reduction 
notes against real hardware. That's the most useful validation anyone 
could give — especially from someone on the same 128 GB club. 🍻

Looking forward to the report.



### #64 [open] PR: Split long non-causal SDPA queries to avoid Ultra corruption
URL: https://github.com/antirez/h3.c/pull/64  by quantz8a 2026-09-12T15:51:06Z

## Summary
- Work around MPSGraph `scaledDotProductAttention` silent corruption on M2/M3 Ultra for long non-causal sequences (#16 / Apple FB24605554).
- Chunk query rows (default 2048) when `sequence > 12288`; leave smaller jobs on the fast single-op path.
- Tunables: `--sdpa-query-block N`, `H3_SDPA_MAX_QUERY_ROWS` (0 disables), `H3_SDPA_SPLIT_MIN_ROWS`.

Fixes #16
Related: #15 (same size class; assert vs silent garbage)

## Notes
- Non-causal only (DiT path). Causal GQA unchanged.
- Split can increase MPSGraph intermediate residency; combine with `--ssd-streaming` for long HQ jobs (as discussed in #16).

## Test plan
- [x] Compiles `h3_gpu.m` / `main.c` with stock flags
- [x] `make -j && ./h3_tests`
- [x] On M2/M3 Ultra: `896×512` or `1344×768`, ≥100 frames, steps≥4 previously produced static → now coherent scene/audio
- [x] Short job (`512×512`, 22–107f) unchanged wall-clock path (no split below gate)
- [x] Local M3 Ultra validation in progress (`896×512`, 107f, steps=4, `--ssd-streaming`)

> **Comment by quantz8a (2026-09-12T16:31:55Z):**

Local M3 Ultra (96GB) validation for this PR:

- `896×512`, `frames=90`, `steps=4`, `layers=45`, `--ssd-streaming`, `H3_DIT_COMMAND_BLOCKS=5`
- Completed `exit=0`; wrote a playable MP4 (`duration=3.75s`)
- Mid-frame luma contrast check: `mean≈17.5`, `std≈17.5` (not collapsed static)

This geometry is in the Ultra corruption zone discussed in #16; output is coherent rather than noise.



### #63 [open] PR: Barrier before reusing GQA reduction scratch
URL: https://github.com/antirez/h3.c/pull/63  by quantz8a 2026-09-12T15:49:42Z

## Summary
- Add a `threadgroup_barrier` after reading the reduced max in `h3_gqa_causal_bf16` before reusing `reductions[]` for the softmax sum.

Fixes #52

## Test plan
- [x] `make -j && ./h3_tests`
- [x] Repeat identical causal-GQA dispatches at long sequence (issue reports nondeterminism around seq ≥662); outputs should match bit-for-bit


### #62 [open] PR: Flush Done lines when stdout is a pipe
URL: https://github.com/antirez/h3.c/pull/62  by quantz8a 2026-09-12T15:49:40Z

## Summary
- Line-buffer stdout at startup so interactive `Done -> ...` completion lines flush immediately when stdout is a pipe (non-TTY).

Fixes #25

## Test plan
- [x] `make -j && ./h3_tests`
- [x] Drive interactive mode with piped stdin/stdout; confirm `Done ->` appears after FFmpeg completes (repro from #25)



### #61 [closed] PR: Preview v2
URL: https://github.com/antirez/h3.c/pull/61  by janishar 2026-09-11T21:52:26Z




### #60 [closed] PR: Add multi-frame denoising preview from the middle chunk
URL: https://github.com/antirez/h3.c/pull/60  by janishar 2026-09-11T18:26:59Z

h3_video_vae_decoder_decode_range decodes N consecutive frames from the middle temporal chunk per Euler step instead of just the middle frame, reusing the same VAE decode already run for the single-frame preview (unpacking more frames from the output tensor is CPU-only, so this costs nothing extra). Wired in as an opt-in path alongside the existing single-frame preview, selected via the new --preview-frames N flag / preview_frame_count param; default (1) keeps prior behavior untouched. Also fixes two bugs found while finishing this: the single-frame branch's cleanup jump was firing unconditionally instead of only on denoise failure, and the range decoder's frame-count clamp didn't account for the chunk's own frame budget.


### #59 [open] PR: Build against SDKs older than macOS 26
URL: https://github.com/antirez/h3.c/pull/59  by abhas9 2026-09-07T03:25:51Z

`make` fails with the macOS 15.2 SDK because of a missing standard header and an unguarded reference to a macOS 26 SDK enum. Three added lines, no behaviour change when building against the macOS 26 SDK.

## The failures

**1. `SSIZE_MAX` undeclared - `h3_ffmpeg.c`**

```text
h3_ffmpeg.c:60:49: error: use of undeclared identifier 'SSIZE_MAX'
h3_ffmpeg.c:591:46: error: use of undeclared identifier 'SSIZE_MAX'
```

The file uses `SSIZE_MAX` without including `<limits.h>`, its POSIX defining header. Compilation relies on transitive includes, which do not supply it in the tested macOS 15.2 toolchain. `SSIZE_MAX` itself is not a macOS 26 addition.

**2. `MTLGPUFamilyMetal4` undeclared - `h3_metal.m`**

```text
h3_metal.m:43:51: error: use of undeclared identifier 'MTLGPUFamilyMetal4';
                        did you mean 'MTLGPUFamilyMetal3'?
```

`MTLGPUFamilyMetal4` first appears in the macOS 26 SDK. The existing `if (@available(macOS 26.0, *))` guard is a runtime check, so it cannot hide a symbol the headers never declare.

## The fix

`h3_ffmpeg.c` includes `<limits.h>`.

`h3_metal.m` gates the probe on the SDK version in addition to the runtime check:

```c
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
        if (@available(macOS 26.0, *)) {
            info->metal4 = [device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
        }
#endif
```

No `#else` is needed: `h3_metal_probe` already does `memset(info, 0, sizeof(*info))`, so `info->metal4` stays `0` when the block is compiled out.

This is a conservative fallback: binaries built with an older SDK report `Metal 4 no` even when subsequently run on macOS 26 with capable hardware. Builds using SDK 26 retain the existing runtime capability check.

## Verification

Verified at commit `91452961c0af4f1f3efa920e3d831014aca80fa9`.

### macOS 15.2 / Command Line Tools SDK 15.2

```sh
make clean && make -j8
make test
```

Clean build with no errors or new warnings; `make test` exits 0. Native AudioVAE Metal primitives and the FFmpeg video/PCM pipe test pass. Fixture-dependent checks skip because the required fixtures are not installed.

Device inspection reports `Metal 4 no`, as expected for this OS/SDK combination, not as a limitation of M4 hardware.

End-to-end FL2VA generation with first-frame conditioning and `--ssd-streaming` succeeds at 576x1024.

### macOS 26.6.2 / Command Line Tools SDK 26.5

On Metal 4-capable Apple Silicon:

```sh
make -j8 all h3_tests h3_audio_gpu_tests h3_av_mux_test
```

The CLI, library, and selected test binaries build without warnings. The host suite passes 1,768 checks; native AudioVAE Metal primitives and FFmpeg A/V integration also pass.

A standalone assertion harness linked to the build's actual `h3_metal.o` confirms the positive branch against Apple's direct capability query:

```text
native_supports_Metal4=1 h3_metal_probe_metal4=1 expected=1
PASS
```

The preprocessed `h3_metal_probe` function is identical to the base revision with SDK 26.5.

### SDK and deployment-target coverage

| Configuration | Result |
|---|---|
| Both changed files compiled against installed SDK 15.4 | Pass with `-Werror` |
| SDK 15.4 probe run on macOS 26 | Returns `metal4 = 0`, confirming the compiled-out fallback |
| SDK 26.5 probe compiled with `-mmacosx-version-min=15.0` | Pass with `-Werror -Wunguarded-availability`; returns `metal4 = 1` when run on macOS 26 |

The original Metal enum compilation failure reproduces with SDK 15.4. The original `SSIZE_MAX` failure does not reproduce with SDK 15.4; it was observed with SDK 15.2.

The SDK 26 build has not been run on macOS 15. Fixture-dependent tests and end-to-end generation were not repeated on macOS 26.


## Note
The change was done by claude but has been manually reviewed by me. Happy to discuss this with fellow humans if needed (without any AI use in discussion) 


### #57 [closed] PR: Align windowed-attention slabs to 5-latent groups
URL: https://github.com/antirez/h3.c/pull/57  by 6ruibin 2026-09-04T14:38:03Z

## Summary
`H3_ATTN_WINDOW=W` previously sliced query frames in slabs of `W` latent frames. Video latents are grouped 1+4 (`h3_frame_per_token` spacing `{1,4,4,4,4}`), and the first latent of each group is the 1-frame anchor the other four depend on. Slabs that were not aligned to those groups left offset-3/4 latents without their anchor; the VAE decoded that as a ~4-frame dissolve every misaligned group (period 10 latents ≈ 1.4 s at `W=2`).

Default slab is now the 5-latent group. `H3_ATTN_SLAB=S` overrides. `H3_ATTN_DENSE_STEPS` / `H3_ATTN_DENSE_LAYER_STRIDE` run selected passes with full attention. `bench_sdpa` gains `--slab`.

## Verification
- Unit tests: `./h3_tests` → 1802 checks
- Probe at 1344×768 / 90 frames, same prompt/seed:
  - `W=2` slab=2 (old): 2 dissolves
  - `W=2` slab=5 (new default): 0 luma jumps
- Full 15 s Turbo rerun (`H3_ATTN_WINDOW=2`, 362 frames): 0 luma jumps vs 33 on the pre-fix clip; denoise ~22% slower (1518 s vs 1248 s)

## Env knobs
- `H3_ATTN_WINDOW=W` — unchanged
- `H3_ATTN_SLAB=S` — query frames per K/V span (default 5)
- `H3_ATTN_DENSE_STEPS=N` — first N denoise steps use full attention
- `H3_ATTN_DENSE_LAYER_STRIDE=K` — every K-th DiT block uses full attention

> **Comment by LRBin (2026-09-04T14:41:47Z):**

Opened against the wrong repository — this branch belongs on LRBin/h3c (it includes the private performance-lab stack, not just the slab fix). Closing.



### #56 [open] PR: Register the seven H3 control tokens the vocabulary omits
URL: https://github.com/antirez/h3.c/pull/56  by Tatlatat 2026-09-04T02:53:34Z

### Problem

MiniMax declares seven H3-specific control tokens in `tokenizer_config.json` under `additional_special_tokens`:

```
<d>  </d>  <|cutoff|>  <|lyrics_start|>  <|lyrics_end|>  <|caption_start|>  <|caption_end|>
```

They are in neither the BPE vocabulary nor `tokenizer.json`'s `added_tokens` array, which stops at 151668. `h3_tokenizer_load()` only reads the latter, so it never learns them and the BPE merges swallow the markup.

The damage reaches past the tag itself. Against the released `FL2VA/tokenizer`, `<d>[English] Hello there.</d>` currently encodes as nine pieces:

```
90707 30768 22574 60 21927 1052 3918 67 29
```

`90707` fuses `<d>[` into one token and `3918` fuses `.</`, so the dialogue marker and the `[English]` language tag beside it both reach the DiT as ordinary prose. `<|cutoff|>` becomes six pieces, `<|caption_start|>` six.

This matters because the official prompting guides ([base](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md), [ref](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)) build every dialogue line, cross-cut continuation and truncation on exactly these markers.

### Fix

Continue the identifiers from the highest declared added token, which is what the reference tokenizer does. For the released vocabulary that yields 151669-151675, comfortably inside the 151936 `TEXT_VOCAB` embedding bound. Tokens already present in `tokenizer.json` are skipped, so a future release that declares them properly keeps its own ids rather than getting duplicates.

### Verification

`transformers` 5.16.1 loading the same `FL2VA/tokenizer` directory:

```
<d> -> 151669   </d> -> 151670   <|cutoff|> -> 151671   <|lyrics_start|> -> 151672
<|lyrics_end|> -> 151673   <|caption_start|> -> 151674   <|caption_end|> -> 151675

'<d>[English] Hello there.</d>' -> [151669, 58, 22574, 60, 21927, 1052, 13, 151670]
```

`h3.c` after this patch, same input, same file:

```
<d>[English] Hello there.</d>      ->  8 ids: 151669 58 22574 60 21927 1052 13 151670
<|cutoff|>                         ->  1 ids: 151671
<|caption_start|>                  ->  1 ids: 151674
<|im_start|>                       ->  1 ids: 151644          (unchanged)
A red fox walking through snow     ->  6 ids: 32 2518 38835 11435 1526 11794   (unchanged)
```

`tests/test_tokenizer.c` gains the three cases above, including the decode round-trip that `check_case` already performs. `make h3_tokenizer_tests` passes against both the FL2VA and the Ref2VA tokenizer (83 checks, was 60), and `./h3_tests` still reports its 1768 checks. Builds clean under the existing `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`.

One thing I did **not** do is an A/B of actual generations. I checked the embedding rows at 151669-151675 and they are statistically indistinguishable from the unused padding rows above them (norm 0.505 +/- 0.006 against 0.503 +/- 0.005 for rows 151676-151935, while the 26 genuinely declared special tokens average 0.662 with a 1.34 maximum), so these embeddings look frozen at initialization rather than trained as language-model tokens. That does not make the mapping wrong, since the text encoder is frozen and the DiT trained on top of whatever fixed vectors it produced, and it is what the reference pipeline feeds. It does mean the argument here is "match the reference tokenizer", not "measured better video".



### #55 [open] PR: Add H3_ATTENTION_CACHE: stream QKV/attention-out (and optionally MLP)…
URL: https://github.com/antirez/h3.c/pull/55  by JunzoKamahara 2026-09-02T10:31:49Z

Parallels `--ssd-streaming` but for the int8-resident path: instead of loading BF16 and quantizing at startup, QKV/attention-output (and, with `H3_INT8_STREAM_MLP=1`, FC1/FC2 too) are streamed per block from a cache built once by the new `build_attention_cache` tool, using the same `h3_gpu_quantize_weight_int8` routine as the existing resident path.

**Usage:**
```sh
build_attention_cache MiniMax-H3/FL2VA dit_int8_v2.cache

H3_ATTENTION_CACHE=dit_int8_v2.cache ./h3 -d MiniMax-H3 -p "..."
# add H3_INT8_STREAM_MLP=1 to also stream FC1/FC2, for the lowest DiT peak
```

This reaches full 50/50-block quality (no `--layers` skipping) at a DiT peak close to or below today's degraded-quality resident-int8 runs, since double-buffered streaming needs only two small slots instead of full block residency. With `H3_INT8_STREAM_MLP` additionally streaming the MLP, DiT peak drops further (~1.5 GiB at 22 frames, ~5.4 GiB at 362 frames), which is what makes a 24GB machine comfortable at ~15s/362-frame lengths.

In practice, this also makes full-depth generation substantially more usable on memory-constrained Macs: on a 24GB M5 Mac, a 5-second video can now be generated in under 10 minutes while still using all 50 DiT blocks.

The trade-off is additional SSD usage: the pre-quantized int8 cache requires approximately 18 GB of extra disk space in addition to the original model files. The cache is generated once and can then be reused across subsequent runs, avoiding repeated BF16 loading and runtime quantization.

Verified bit-for-bit identical output (matching SHA-256) against the existing resident-int8 path at matched seed, for both attention-only streaming and full attention+MLP streaming.


> **Comment by monotected (2026-09-03T09:46:58Z):**

Hi,
It’s not very convenient to use when I need to generate a quick test video on the MFR M5 Pro (24GB).
I rebuilt it with your commit yesterday; everything works great for long videos, but for short ones, it takes significantly longer. It’s easier and faster to run them using the `--ssd-streaming` flag.
However, there’s a restriction in place, so I can't simply add that argument to bypass the cache; I’ve had to load the model by bypassing your implementation every time.
Could you please allow the use of the `--ssd-streaming` flag in automatic mode? (When this flag is set, the cache isn't used.)
Thanks!


> **Comment by JunzoKamahara (2026-09-04T03:10:58Z):**

Hi, thanks for the report — and sorry, the original PR description didn't actually show how to enable this feature at all (just pushed a fix for that too, see the updated description above): you need to explicitly build a cache and point `H3_ATTENTION_CACHE` at it:

```sh
build_attention_cache MiniMax-H3/FL2VA dit_int8_v2.cache # once
H3_ATTENTION_CACHE=dit_int8_v2.cache H3_INT8_STREAM_MLP=1 ./h3 \
  -d MiniMax-H3 -p "..." --width 512 --height 512 --frames 22 \
  --steps 4 --layers 50 --reuse 2 -o outputs/fox-quick-test.mp4
```

I've also pushed a separate fix (`3ec6ac6`) so `--ssd-streaming` now silently overrides `H3_ATTENTION_CACHE` if you ever want to bypass the cache entirely — but with `H3_INT8_STREAM_MLP=1` set, you likely won't need it for short clips anymore. Let me know how it goes.


> **Comment by monotected (2026-09-04T14:05:17Z):**

Thanks!!!
short tests on cold (1st run) 

so i made about 12 vids 6 H3_ATTENTION_CACHE +6 standart h3.c --ssd-streaming
H3_ATTENTION_CACHE really gives me 25 % speed  
 
 in t2v - no visible degrade and its fater 
 in ref2va i didnt understand why but :Unfortunately, there is a visible drop in quality with  H3_ATTENTION_CACHE  (all fine details are missing—I tried 0.3–0.5 MP).
ps 
also i have 2 backed lora over the main model (maybe that is the reason of degrade )
add pics of missing parts on same seed  
1pic normal legs + feet
2pic legs + transparent feet 

same on 0.3-0.5 mp 
<img width="239" height="181" alt="Снимок экрана — 2026-09-05 в 01 41 50" src="https://github.com/user-attachments/assets/c0a51744-ab9e-4630-a8be-09db62b48c54" />
<img width="178" height="169" alt="Снимок экрана — 2026-09-05 в 01 42 18" src="https://github.com/user-attachments/assets/226e05bc-7dae-4475-9a7e-0ef04a2e27f4" />






cli args :
H3_ATTENTION_CACHE=./fl2va_v2.cache H3_INT8_STREAM_MLP=1 H3_ATTN_DENSE_STEPS=4 ./h3 -d ./MiniMax-DoubleLoRA-Master -p \

**now  with test short video  with cash i got 1m 28sec ** (Memory lifespan is barely consumed.)
N/A" \
--ref-image /Users/icoidd/Desktop/minimax/start1.jpg \
--ref-image /Users/icoidd/Desktop/minimax/start2.jpg \
--ref-image /Users/icoidd/Desktop/minimax/start3.jpg \
--ref-image /Users/icoidd/Desktop/minimax/face.jpg \
--layers 50 \
--width 352 --height 576 \
--seconds 1 --steps 15 --reuse 3 \
--use-reference-rope \
--seed 948572 \
-o sad_mirror_close_011.mp4
tokenizer                    1/1
video VAE encoder            4/4
video VAE encoder            6/6
video VAE encoder            6/6
video VAE encoder            6/6
Qwen vision                 27/27
Qwen vision                 27/27
Qwen vision                 27/27
Qwen vision                 27/27
text encoder                50/50
refine text                  1/1
precompute AdaLN            50/50
load transformer core       50/50
denoise enqueue             15/15
denoise                     15/15
audio VAE                    7/7
video VAE load              36/36
FFmpeg                      39/39
h3: wrote sad_mirror_close_011.mp4


** and with --ssd-streaming. its 1m 50sec ** (Memory lifespan is on 15 to 18 gb in peak )
cli args :
./h3 -d ./MiniMax-DoubleLoRA-Master -p \

--ref-image /Users/icoidd/Desktop/minimax/start1.jpg \
--ref-image /Users/icoidd/Desktop/minimax/start2.jpg \
--ref-image /Users/icoidd/Desktop/minimax/start3.jpg \
--ref-image /Users/icoidd/Desktop/minimax/face.jpg \
--layers 50 \
--width 352 --height 576 \
--seconds 1 --steps 15 --reuse 3 \
--ssd-streaming \
--use-reference-rope \
--seed 948572 \
-o sad_mirror_close_011.mp4
tokenizer                    1/1
video VAE encoder            4/4
video VAE encoder            6/6
video VAE encoder            6/6
video VAE encoder            6/6
Qwen vision                 27/27
Qwen vision                 27/27
Qwen vision                 27/27
Qwen vision                 27/27
text encoder                50/50
refine text                  1/1
precompute AdaLN            50/50
load transformer core       50/50
denoise enqueue             15/15
denoise                     15/15
audio VAE                    7/7
video VAE load              36/36
FFmpeg                      39/39
h3: wrote sad_mirror_close_011.mp4


> **Comment by JunzoKamahara (2026-09-06T00:00:51Z):**

Thanks again — I found the cause of the Ref2VA degradation.

The cache file was valid structurally, but an FL2VA cache could also be accepted during Ref2VA generation because both transformers use the same tensor dimensions. That meant Ref2VA could silently run with FL2VA cached weights.

I changed the cache format and added automatic FL2VA / Ref2VA cache selection.

New recommended usage

Build both caches once from the model root:

./build_attention_cache \
  ./MiniMax-DoubleLoRA-Master \
  ./attention-cache

This creates:

attention-cache/
  fl2va.cache
  ref2va.cache

Then use the cache directory instead of selecting a cache file manually:

H3_ATTENTION_CACHE_DIR=./attention-cache \
H3_INT8_STREAM_MLP=1 \
./h3 -d ./MiniMax-DoubleLoRA-Master -p "..." ...

h3.c will now choose the correct cache automatically:

- normal T2V → "fl2va.cache"
- "--ref-image" / Ref2VA → "ref2va.cache"

So for your previous Ref2VA command you should no longer use:

H3_ATTENTION_CACHE=./fl2va_v2.cache

Instead, build the two caches from your "MiniMax-DoubleLoRA-Master" directory and use "H3_ATTENTION_CACHE_DIR".

This is especially important in your case because your two LoRAs are baked into the model: the caches should be generated from those exact baked transformer weights.

The old single-file form is still supported:

H3_ATTENTION_CACHE=./some.cache

but the cache now records whether it is FL2VA or Ref2VA and h3.c rejects the wrong type instead of silently using it.

The cache format has also changed from v2 to v3, so existing "fl2va_v2.cache" files need to be rebuilt.

I tested:

- FL2VA cache with T2V: OK
- Ref2VA cache with "--ref-image": OK
- FL2VA cache used for Ref2VA: correctly rejected
- "H3_ATTENTION_CACHE_DIR" automatic selection for both modes: OK

Could you rebuild the caches from "MiniMax-DoubleLoRA-Master" with the command above and try the same Ref2VA seed again? That should tell us whether the missing-detail issue was entirely caused by the FL2VA/Ref2VA cache mismatch.


> **Comment by monotected (2026-09-08T23:57:40Z):**

Hi - sorry for late  soo.... i rebuild h3.c  also  update and bake a new cashe 
now gen time for me takes  about 4-5 -7 min for 2sec 0.2mp video ( maybe new v3 code, streaming int8 matrices from the attention-cache folder creates a severe data bottleneck between the SSD and the GPU, or completely saturates the CPU cache [0.2]. The author's new patch simply failed to interface properly with your custom DoubleLoRA build architecture)

ps 
so sorry that i didt use common minimax weights.
pls see args 

icoidd@MacBook-Pro-icoidd h3.c % bash# 1. 
rm -rf ./attention-cache/*.cache

# 2. 
./build_attention_cache /Users/icoidd/h3.c/MiniMax-CustomLoRA-fl2va/FL2VA/transformer ./attention-cache/fl2va.cache
./build_attention_cache /Users/icoidd/h3.c/MiniMax-CustomLoRA-ref2va/Ref2VA/transformer ./attention-cache/ref2va.cache

# 3. 
H3_ATTENTION_CACHE_DIR=./attention-cache H3_INT8_STREAM_MLP=1 ./h3 -d ./MiniMax-DoubleLoRA-Master -p "non_diegetic_music: N/A" \
--ref-image /Users/icoidd/Desktop/minimax/1010.jpg \
--layers 50 \
--width 352 --height 576 \
--seconds 2 --steps 15 --reuse 3 \
--use-reference-rope \
--seed 719385 \
-o test_perfection_v3.mp4
zsh: command not found: bash#
zsh: command not found: #
h3: attention cache block  1/50
h3: attention cache block  2/50
h3: attention cache block  3/50
h3: attention cache block  4/50
h3: attention cache block  5/50
h3: attention cache block  6/50
h3: attention cache block  7/50
h3: attention cache block  8/50
h3: attention cache block  9/50
h3: attention cache block 10/50
h3: attention cache block 11/50
h3: attention cache block 12/50
h3: attention cache block 13/50
h3: attention cache block 14/50
h3: attention cache block 15/50
h3: attention cache block 16/50
h3: attention cache block 17/50
h3: attention cache block 18/50
h3: attention cache block 19/50
h3: attention cache block 20/50
h3: attention cache block 21/50
h3: attention cache block 22/50
h3: attention cache block 23/50
h3: attention cache block 24/50
h3: attention cache block 25/50
h3: attention cache block 26/50
h3: attention cache block 27/50
h3: attention cache block 28/50
h3: attention cache block 29/50
h3: attention cache block 30/50
h3: attention cache block 31/50
h3: attention cache block 32/50
h3: attention cache block 33/50
h3: attention cache block 34/50
h3: attention cache block 35/50
h3: attention cache block 36/50
h3: attention cache block 37/50
h3: attention cache block 38/50
h3: attention cache block 39/50
h3: attention cache block 40/50
h3: attention cache block 41/50
h3: attention cache block 42/50
h3: attention cache block 43/50
h3: attention cache block 44/50
h3: attention cache block 45/50
h3: attention cache block 46/50
h3: attention cache block 47/50
h3: attention cache block 48/50
h3: attention cache block 49/50
h3: attention cache block 50/50
h3: wrote attention cache to ./attention-cache/fl2va.cache
h3: attention cache block  1/50
h3: attention cache block  2/50
h3: attention cache block  3/50
h3: attention cache block  4/50
h3: attention cache block  5/50
h3: attention cache block  6/50
h3: attention cache block  7/50
h3: attention cache block  8/50
h3: attention cache block  9/50
h3: attention cache block 10/50
h3: attention cache block 11/50
h3: attention cache block 12/50
h3: attention cache block 13/50
h3: attention cache block 14/50
h3: attention cache block 15/50
h3: attention cache block 16/50
h3: attention cache block 17/50
h3: attention cache block 18/50
h3: attention cache block 19/50
h3: attention cache block 20/50
h3: attention cache block 21/50
h3: attention cache block 22/50
h3: attention cache block 23/50
h3: attention cache block 24/50
h3: attention cache block 25/50
h3: attention cache block 26/50
h3: attention cache block 27/50
h3: attention cache block 28/50
h3: attention cache block 29/50
h3: attention cache block 30/50
h3: attention cache block 31/50
h3: attention cache block 32/50
h3: attention cache block 33/50
h3: attention cache block 34/50
h3: attention cache block 35/50
h3: attention cache block 36/50
h3: attention cache block 37/50
h3: attention cache block 38/50
h3: attention cache block 39/50
h3: attention cache block 40/50
h3: attention cache block 41/50
h3: attention cache block 42/50
h3: attention cache block 43/50
h3: attention cache block 44/50
h3: attention cache block 45/50
h3: attention cache block 46/50
h3: attention cache block 47/50
h3: attention cache block 48/50
h3: attention cache block 49/50
h3: attention cache block 50/50
h3: wrote attention cache to ./attention-cache/ref2va.cache
zsh: command not found: #
tokenizer                    1/1
video VAE encoder            6/6
Qwen vision                 27/27
text encoder                50/50
refine text                  1/1
precompute AdaLN            50/50
load transformer core       50/50
denoise enqueue             15/15
denoise                     15/15
audio VAE                    7/7
video VAE load              36/36
FFmpeg                      56/56
h3: wrote test_perfection_v3.mp4.  = 7min vs --ssd-stream 1.4min


> **Comment by monotected (2026-09-09T00:12:01Z):**

i allready rebuild with original weights = same time 5-7 min  
(it also work with --ssd-streaming together  takes 1.28min but i didnt understand - does it use cache v3 or not )




> **Comment by JunzoKamahara (2026-09-09T09:00:27Z):**

Thanks for testing this.

A couple of notes for the benchmark:

- "build_attention_cache" only needs to be run once per model. I have already built the cache from the original model weights, so there is no need to rebuild it for each run.
- On my machine, using your prompt, the PR version took about 129 seconds, while "--ssd-streaming" took about 158 seconds.
- When measuring the total execution time, please prepend "/usr/bin/time" to the "./h3" command, for example:

"/usr/bin/time ./h3 ..."

That should make the comparison easier and more consistent.


> **Comment by monotected (2026-09-09T14:24:10Z):**

> Спасибо за тестирование.
> 
> Несколько замечаний по поводу теста производительности:
> 
> * Функцию "build_attention_cache" нужно запускать только один раз для каждой модели. Я уже создал кэш на основе исходных весов модели, поэтому нет необходимости перестраивать его при каждом запуске.
> * На моей машине при использовании вашей команды выполнение версии PR заняло около 129 секунд, а "--ssd-streaming" — около 158 секунд.
> * При измерении общего времени выполнения, пожалуйста, добавьте перед командой "./h3" префикс "/usr/bin/time", например:
> 
> "/usr/bin/time ./h3 ..."
> 
> Это должно упростить и сделать сравнение более последовательным.
Thanx later this nigh ill try 

.....some kind a bud on my side 
icoidd@MacBook-Pro-icoidd h3.c % H3_ATTENTION_CACHE_DIR=./attention-cache H3_INT8_STREAM_MLP=1 /usr/bin/time ./h3 -d ./MiniMax-DoubleLoRA-Master -p "non_diegetic_music: N/A" \
--ref-image /Users/icoidd/Desktop/minimax/1010.jpg \
--layers 50 \
--width 352 --height 576 \
--seconds 2 --steps 15 --reuse 3 \
--use-reference-rope \
--seed 719385 \
-o test_perfection_v3.mp4

tokenizer                    1/1
video VAE encoder            6/6
Qwen vision                 27/27
text encoder                50/50
refine text                  1/1
precompute AdaLN            50/50
load transformer core       50/50
denoise enqueue             15/15
denoise                     15/15
audio VAE                    7/7
video VAE load              36/36
FFmpeg                      56/56
h3: wrote test_perfection_v3.mp4
      505.26 real         1.31 user        78.63 sys


> **Comment by monotected (2026-09-09T21:05:38Z):**

thanx!!  /issue was on my syde  now its great no quality loss 
icoidd@MacBook-Pro-icoidd h3.c % H3_ATTENTION_CACHE_DIR=./attention-cache H3_INT8_STREAM_MLP=1 /usr/bin/time ./h3 -d ./MiniMax-DoubleLoRA-Master -p "non_diegetic_music: N/A" --ref-image /Users/icoidd/Desktop/minimax/1010.jpg --layers 50 --width 352 --height 576 --seconds 2 --steps 15 --reuse 3 --use-reference-rope --seed 719385 -o test_perfection_v3.mp4

tokenizer                    1/1
video VAE encoder            6/6
Qwen vision                 27/27
text encoder                50/50
h3: warning: attention cache ./attention-cache/ref2va.cache's model fingerprint does not match the loaded checkpoint (different weights, a LoRA baked in after the cache was built, or weights rewritten since) - results may be wrong; rebuild the cache if unsure
refine text                  1/1
precompute AdaLN            50/50
load transformer core       50/50
denoise enqueue             15/15
denoise                     15/15
audio VAE                    7/7
video VAE load              36/36
FFmpeg                      56/56
h3: wrote test_perfection_v3.mp4
       80.21 real         1.27 user        34.31 sys
icoidd@MacBook-Pro-icoidd h3.c %



### #54 [open] ISSUE: how to generate 2K videoes
URL: https://github.com/antirez/h3.c/issues/54  by ChristopherKai 2026-08-30T22:59:44Z

how to generate 2K videoes? thanks


### #53 [open] PR: docs: add Community integrations section
URL: https://github.com/antirez/h3.c/pull/53  by jszfly1989 2026-08-29T16:05:40Z

Adds a short Community integrations section linking h3-metal-server.


### #52 [open] ISSUE: A potential nondeterministic issue
URL: https://github.com/antirez/h3.c/issues/52  by yfyang86 2026-08-27T02:29:38Z

I am using this project to enhance other VLM (A Qwen2 variant) and found a potential nondeterministic issue, i.e. the causal-GQA kernel proved nondeterministic under repeated identical dispatches (29/30 runs differed at sequence 662). Fix this will greatly reduce the performance. In my case 47 ms vs 5.5 ms at sequence length of  2,013. 

If deterministic is required in some case, some one may find this short patch useful, but it needs a re-design to get better performance.

```diff
--- a/h3_shaders.metal	2026-08-20 17:01:10
+++ b/h3_shaders.metal	2026-08-21 10:54:11
@@ -4003,6 +4003,11 @@
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }
     float maximum = reductions[0];
+    /* fix: WAR hazard —  reuses reductions[]; without this
+     * barrier one thread can overwrite reductions[0] before another thread
+     * has read the max, making the softmax nondeterministic at long
+     * sequences (observed: 29/30 identical-input runs differed at seq 662). */
+    threadgroup_barrier(mem_flags::mem_threadgroup);
     float local_sum = 0.0f;
     for (uint key_row = tid; key_row < key_count; key_row += threads) {
         float probability = exp(scores[key_row] - maximum);
```

> **Comment by ain-soph (2026-11-17T16:08:10Z):**

We hit the same race independently and can add three data points.

**1. It affects end-to-end video renders, not only long text sequences.** On an M5 Ultra, every first-frame (FL2VA) and Ref2VA render differed run to run with identical parameters and seed. The vision span makes the Qwen prompt sequence long, so the race in `h3_gqa_causal_bf16` almost always triggers. A plain T2V render with a ~600-token prompt was also irreproducible. Short-prompt T2V happened to be stable, which is why it is easy to miss. We located it by hashing each pipeline stage: conditioning pixels, VAE latents and vision rows matched across runs, and the text context first diverged after layer 0.

**2. The video VAE encoder's group norm has the same read-then-overwrite pattern.** In `h3_vae_encoder_group_norm_silu_f32`, `mean` is read from `reductions[0]`, and the slot is then reused for the variance reduction without a barrier in between:

```diff
     float mean = reductions[0] / float(elements);
+    threadgroup_barrier(mem_flags::mem_threadgroup);   /* read before reuse */
     local = 0.0f;
```

**3. On our setup the barrier did not cost measurable speed.** With the barrier in both kernels (the `h3_gqa_causal_bf16` change is the same as the patch above), a 512² T2V render took 26 s against 24–27 s before. First-frame, Ref2VA and the 600-token T2V prompt now render byte-identically run to run, and the short-prompt T2V output is unchanged. This is end-to-end wall time, so I can't speak to the isolated 47 ms vs 5.5 ms kernel figure above. In a full render, the text encoder is a small part of the total.

Fix (both barriers, plus an `H3_DEBUG_HASH=1` per-stage hash that helped find it): https://github.com/ain-soph/h3.c/commit/0fa3634323c8580eb5d0f27dafc6713d0a3e67ae

I'm happy to open a PR with just the two barriers if that's useful.




### #51 [open] ISSUE: M4 Pro 24GB: fold Turbo LoRA (PR #14) benchmark — 124f is the sweet spot on both 480P and 768P
URL: https://github.com/antirez/h3.c/issues/51  by AideYu 2026-08-25T09:05:47Z

# M4 Pro 24GB: fold Turbo LoRA (PR #14) 4-spec benchmark — 124f is the sweet spot on both 480P and 768P

Machine: Mac mini M4 Pro, 24GB unified memory, macOS 26.6.2, `--ssd-streaming`.
Test: `MiniMax-H3-turbo` (fold Turbo v4), FL2VA first/last frames, seed 42, 6 steps,
ET5T shot 1 (strong-motion drift, gravel dust). Wall-clock = process birth → `h3: wrote mp4`.

| Resolution | Frames | Steps | Wall-clock |
|---|---|---|---|
| 832×480  | 124 | 6 | 19.4 min (01:12→01:31) |
| 832×480  | 243 | 6 | 46.0 min (07:31→08:17) |
| 1344×768 | 124 | 6 | 75.5 min (08:47→10:03) |
| 1344×768 | 243 | 6 | 206.9 min (~3.4 h) (10:50→14:17) |

Key observations:

1. **124 frames is the sweet spot for BOTH 480P and 768P.** Per-frame
   cost and wall-clock stay predictable at 124f; jumping to 243f is
   clearly superlinear on this memory budget. If you need ~10s, plan for
   a near-linear cost jump in time (and a healthy swap peak).

2. **Single-step linear extrapolation fails for long sequences.** Naively
   scaling 768P/124f (~11 min/step) by latent ×1.9 predicts ~21 min/step
   for 768P/243f; measured (incl. load) is ~34 min/step. ETA for folded
   long shots should not be extrapolated linearly — wait for step 1 and
   re-baseline.

3. **768P fold 6 steps runs fine on 24GB** — no OOM, no major swap
   (peak swap ~11.5 GB). Upscaling from 480P+super-res is still cheaper
   than native 768P/243f (~3.4 h) on this box, but native 768P is viable.

Optional quality note: on this strong-motion clip, Laplacian variance
was lower for fold (309.4) than 4-bit mlx-serve (604.0), yet human
subjective judgement favored fold's natural sharpness — the mlx output
reads as over-sharpened artificial halos. So Laplacian alone is not a
quality proxy; "natural" vs "over-sharp" must be judged by eye.


> **Comment by Henninges (2026-09-15T15:38:46Z):**

M4 Max 128 GB — folded Turbo LoRA (PR #14 tooling, larryvrh v4 adapter), Ref2VA image+audio:

| Mode | Steps / reuse | Frames | Time (warm) |
|------|---------------|--------|-------------|
| Standard | 20 / 2 | 56 | 4:23 |
| Turbo | 6 / 1 | 56 | 3:00 |

End-to-end ~1.46× speedup — matches guyz' 1.42× on M5 Max. Fixed costs
(text encode, VAE, FFmpeg) dominate at short clips, so the gain grows
with clip length.

Quality note: Turbo is excellent for iteration, but high-frequency detail
(eyes, fingers) shows slightly more artifacts at 6 steps than Standard.
For close-up faces / finals I stay on Standard. Confirms PR #14's
"comparable quality" holds broadly, with the caveat that faces reveal
the distillation.



### #50 [open] ISSUE: PLEASE, do *NOT* add cuda support!  ask author to fork the source, and make a new distribution, DON'T pollute this one, thanks!
URL: https://github.com/antirez/h3.c/issues/50  by buggz2k 2026-08-22T18:25:51Z

PLEASE, do *NOT* pollute this package with cuda crap.
We NEED MORE Apple Silicon things, not some compromise.

If you want cuda support, ask author if you can fork and call it something different, and of course, thank the author, and give author credit.

Thanks for your understanding.



> **Comment by monotected (2026-08-23T00:05:04Z):**

agree


> **Comment by Henninges (2026-08-23T04:52:32Z):**

AGREE!



### #49 [open] ISSUE: what is the practical use for this?  I don't see it...
URL: https://github.com/antirez/h3.c/issues/49  by buggz2k 2026-08-22T15:25:20Z

So, I have a 64GB M4 Mac Mini Pro.
I am trying to perform maximum quality output, and get kernel errors.
I don't get this with ComfyUI.

Is this distribution only for small output, to be able to get something for small machines?

I am not understanding, please educate me...


> **Comment by buggz2k (2026-08-22T18:21:56Z):**

now working - is this maximum quality ?

% ./h3 --profile \
  -d ./MiniMax-H3 \
  -p "Scene: a single red fox in a snow-covered pine forest at dawn.
Action: the fox walks steadily left to right and looks toward the camera once.
Camera: medium-height lateral tracking shot, 50 mm lens, stable framing.
Look: photorealistic fur, cold blue ambient light, warm sunrise rim light.
Audio: soft footsteps in snow, light wind through pine branches, no music." \
--width 864 \
--height 480 \
--frames 56 \
--steps 50 \
--layers 50 \
--reuse 1 \
--core-reuse 1 \
--profile \
-o outputs/h3-max-quality.mp4
tokenizer                    1/1   
text encoder                50/50  
h3 profile: Qwen text encoder        total          wall=   8.171s encode=  0.041s wait=   2.491s root-gpu=  0.002s peak=  2.738GiB alloc= 46.874GiB submissions=51 direct=451 linear=350 conv=0 attention=0
refine text                  1/1   
precompute AdaLN            50/50  
load transformer core       50/50  
h3 profile: H3 DiT                   load           wall=  15.109s encode=  0.040s wait=   1.678s root-gpu=  1.260s peak= 37.584GiB alloc= 63.445GiB submissions=55 direct=22 linear=62 conv=0 attention=2
denoise                     50/50  
h3 profile: H3 DiT                   Euler denoise  wall=2721.182s encode=2236.820s wait= 472.183s root-gpu=  0.464s peak= 37.584GiB alloc=  0.000GiB submissions=100 direct=7850 linear=10000 conv=0 attention=2500
h3 profile: H3 DiT                   total          wall=2737.447s encode=2236.860s wait= 473.861s root-gpu=  1.724s peak= 37.584GiB alloc= 63.445GiB submissions=155 direct=7872 linear=10062 conv=0 attention=2502
audio VAE                    7/7   
h3 profile: audio VAE decoder        total          wall=   0.929s encode=  0.656s wait=   0.181s root-gpu=  0.120s peak=  0.282GiB alloc=  0.632GiB submissions=16 direct=340 linear=0 conv=136 attention=0
h3: video VAE tiles 4x2 at 272 pixels
video VAE load              36/36  
h3 profile: video VAE decoder        total          wall=  51.556s encode=  0.130s wait=  49.460s root-gpu=  0.387s peak=  9.408GiB alloc=  9.412GiB submissions=24 direct=5256 linear=3480 conv=0 attention=864
FFmpeg                      56/56  
h3: wrote outputs/h3-max-quality.mp4







> **Comment by buggz2k (2026-08-22T21:36:20Z):**

so, this h3.c will only gen low res, small runtime videos, I can gen longer, and higher resolution videos in ComfyUI with out the hard crash limitation of h3.c



> **Comment by monotected (2026-08-23T00:07:02Z):**

u dont need this - u can use comfui




### #48 [open] PR: Reject overlapping safetensors data offsets
URL: https://github.com/antirez/h3.c/pull/48  by glasses666 2026-08-22T09:44:38Z

## Summary
- Validate the global safetensors payload layout after parsing the header.
- Sort tensors by payload offset before checking contiguity, so unordered JSON headers still load.
- Reject overlapping ranges and payload bytes that are not covered by tensor metadata.

## Why
The safetensors format defines data_offsets relative to the byte buffer and requires the byte buffer to be entirely indexed, with no holes, to prevent polyglot files: https://github.com/huggingface/safetensors#format

h3 already checked each tensor shape against its own byte range, but a header could still make two tensors overlap or leave payload bytes unindexed. This keeps that validation at the header parser boundary so all safetensors consumers share it.

## Tests
- make test
- clang -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE -I. -fsyntax-only h3_safetensors.c tests/test_h3.c
- ASan/UBSan h3_tests



### #47 [open] ISSUE: Qwen text encoder: sequence exceeds Metal threadgroup memory with reference video + ref images (≥150 frames)
URL: https://github.com/antirez/h3.c/issues/47  by cuiyuxin-gif 2026-08-22T03:12:39Z

## Error

```
h3: Qwen layer 0 causal GQA failed: causal attention sequence 10231 exceeds threadgroup memory
```

Generation aborted (exit 1) at the **very beginning** of the text encoder (layer 0, step 0/50), before denoising starts.

## Trigger condition

Combining a **reference video** (`--ref-silent-video`, ~7-8s, 1080×1920) with **multiple reference images** (`--ref-image` × 3-5) and **≥150 frames** (~6.25s at 24fps). Long prompt (~100 characters) further increases the sequence count.

The **dominant factor is the reference video** (~7500 tokens for 7.7s). Reference images contribute negligibly (~300 tokens total for 5 images). The apparent limit is **~7700 tokens** (32KB threadgroup / 4 bytes per float).

## Systematic single-variable experiments

| Combination | Frames | Result | Sequence |
|---|---|---|---|
| 5 images + video + long prompt | 22 | ✅ pass | — |
| 5 images + video + long prompt | 90 | ✅ pass | — |
| 5 images + video + long prompt | 120 | ❌ fail | 8199 |
| 5 images + video + long prompt | 160 | ❌ fail | 10231 |
| **video only** + long prompt | 160 | ❌ fail | 8184 |
| **5 images only** + long prompt | 160 | ✅ pass | — |
| 5 images + video + **short** prompt | 160 | ❌ fail | 8510 |

Downscaling all 5 images to 896px → sequence unchanged (10231). Vision token count from Qwen appears fixed-per-image, not resolution-dependent.

## MPS fallback is not safe

The existing `H3_MPS_GQA=1` fallback (`h3_gpu_gqa_mps`) fails differently: it **causes a system-wide watchdog panic** (machine freeze, 5 verified occurrences) during the `denoise` stage with these large sequences. This was tested on both M5 Max 128GB and M3 Ultra 96GB (macOS 26.4.1/26.5.2). I had to revert my MPS fallback patch because it was worse than the original error.

## Current workaround (in our webapp)

- Auto-cap frames to **90** when a `--ref-silent-video` is present (sequence stays under ~7700)
- Auto-downscale all `--ref-image` to 896px before passing to h3 (solves the jetsam memory explosion: original 2400px images × 5 caused h3 to consume 168GB unified memory on a 96GB machine)
- **Hard-block** the "reference video + reference images" combination: even with the above guards, the denoise stage triggers watchdog panics with ≥2 images + 1 video (4 verified panics, including after a fresh reboot)

90-frame cap (~3.75s at 24fps) is quite limiting — users needing longer videos must generate multiple short clips and stitch them.

## Suggested fix directions

The root cause is `h3_gpu_gqa_causal_bf16` storing the **entire causal attention score row** (sequence × 4 bytes) in threadgroup memory in a single dispatch:

1. **Tiled GQA**: split the score computation across multiple dispatches with intermediate global memory, requiring only `tile_size × sizeof(float)` of threadgroup memory per dispatch.

2. **Reference video token compression**: the Qwen encoder produces ~1000 tokens per second of video — pooling/down-sampling these before feeding them into the text encoder would directly reduce sequence length without losing semantic content.

---

*Note: PR [#44](https://github.com/antirez/h3.c/pull/44) (16-byte alignment fix) is orthogonal — it addresses a Metal validation-layer assertion, not this capacity limit.*

## Environment

- h3.c: latest `main` (commit `8b7d5b9`)
- macOS 26.5.2 / M5 Max 128GB; macOS 26.4.1 / Mac Studio M3 Ultra 96GB
- Model: `MiniMax-AI/MiniMax-H3` (FL2VA checkpoint)
- Metal: `maxThreadgroupMemoryLength` = 32768 bytes



### #46 [open] ISSUE: how to download the models in a batch file?  where are the instructions?
URL: https://github.com/antirez/h3.c/issues/46  by buggz2k 2026-08-21T18:07:00Z

Where are the instructions to download the the hugging face models?
You want use to download the entire repo ?
I assume you perform this in some batch file method, otherwise it will, and still will under batch mode, take forever.


> **Comment by buggz2k (2026-08-21T18:21:43Z):**

currently using a script that codex cli generated...



### #45 [open] PR: Add H3 Studio, an adaptive macOS GUI
URL: https://github.com/antirez/h3.c/pull/45  by PierpaoloV 2026-08-21T12:19:03Z

## Why

Local video generation should be approachable beyond users who are already comfortable assembling long terminal commands. This PR adds **H3 Studio**, a small native macOS GUI aimed at democratizing access to H3 while keeping the existing CLI workflow unchanged.

H3 Studio detects the Mac it is running on, reports its relevant hardware characteristics, and recommends a practical starting preset. Users can then work with the same generation controls exposed by the terminal through a more discoverable desktop interface.

## What it adds

- Automatic detection of Apple silicon, unified memory, architecture, and Metal support.
- Hardware-aware **Fast**, **Balanced**, and **Quality** presets, plus a **Custom** mode exposing the practical generation parameters.
- Controls for model location, prompt, output, aspect ratio, duration, seed, render canvas, steps, layers, reuse, core reuse, token reduction, SSD streaming, reference sizing, and supported M5 INT8 acceleration.
- Ordered multi-image conditioning for up to nine reference pictures, matching H3's `Picture 1`, `Picture 2`, … prompt semantics.
- Automatic HEIC/HEIF-to-PNG conversion for iPhone photos. Embedded orientation is materialized into the output pixels so Qt, FFmpeg, and H3 all see the same upright image; originals remain untouched.
- Separate previews for selected inputs and live denoising frames, with resize-aware rendering and clear preview failures.
- A cancellable generation runner with three progress stages—preparation, generation, and decode/export—and an ETA that handles repeated encoder phases and recovers from early underestimates.
- A responsive, scrollable English-only interface that adapts from wide to narrow windows.
- Session-only prompts and reference images, while useful technical choices are persisted locally.
- A `make gui-app` target that builds and signs a double-clickable `H3Studio.app`, bundling the H3 executable and Metal shader while keeping model weights external.

The GUI is optional: the existing CLI remains the primary engine interface and its behavior is preserved.

## Verification

- `make test` — 1,768 core checks passed; available Metal, FFmpeg, and audio primitive tests passed.
- `QT_QPA_PLATFORM=offscreen .venv-gui/bin/python -m unittest discover -s gui/tests -t .` — 33 GUI tests passed.
- `.venv-gui/bin/python -m mypy gui` — passed.
- `make gui-app` — application built, ad-hoc code signature verified, and packaged smoke test passed.
- Orientation handling was also verified against a real iPhone HEIC reference: the normalized PNG is upright, metadata-neutral, and reported consistently as 3024×4032 by Qt and FFmpeg.


> **Comment by PierpaoloV (2026-08-21T13:05:27Z):**

Hi Salvatore,

Following your goal of democratising AI access, as discussed in the latest Resistori Live, I thought a small GUI could make local image/video generation easier and more approachable.

![H3 Studio running with its hardware-aware Fast preset](https://raw.githubusercontent.com/PierpaoloV/h3.c/feature/h3-studio-gui/gui/assets/h3-studio-ui.png)

The interface is organised around a simple workflow:

- **Hardware-aware starting point:** the top panel detects the Mac, unified memory, architecture, and Metal support, then recommends a suitable preset.
- **Progressive control:** Fast, Balanced, and Quality cover common workflows, while Custom exposes the practical terminal parameters for users who want full control.
- **Reference workflow:** users can provide ordered image references, including iPhone HEIC/HEIF photos, and inspect them as Picture 1, Picture 2, and so on before generation.
- **Generation feedback:** the right panel switches between input previews and live denoising frames, while the three-part progress bar tracks preparation, generation, and decode/export with an ETA and cancellation support.
- **CLI compatibility:** the GUI remains optional and launches the existing H3 executable with explicit arguments; the terminal workflow is unchanged.

The aim is to preserve H3's flexibility while giving people who are less comfortable with long CLI commands a practical path into local generation.



> **Comment by Henninges (2026-08-22T08:30:24Z):**

Is it a coincidence that your GUI has the same name as mine from last week? ^^
( https://github.com/Henninges/h3-studio ) 


> **Comment by Henninges (2026-08-22T08:39:15Z):**

Great to see this! I actually published a web-based GUI last week with the same name (https://github.com/Henninges/h3-studio), but it's a different approach: I focus on the music-video workflow (lip-synced video from image + audio references, bilingual DE/EN, 9 images / 3 videos / 3 audio refs, Context-IR prompt builder), while your native macOS app looks like it targets a broader desktop use case with hardware-aware presets.

Two findings from my testing that might be useful for both projects:

1. **`--token-reduction` causes ghosting + audio artifacts** on my M4 Max 128GB. I tested this systematically: every render with `--token-reduction` produced double-image artifacts and garbled audio, while disabling it restored clean output. My GUI now disables it by default.

2. **Language limitation for speech**: The model handles singing well (clear lip-sync), but **German speech becomes garbled/nonsensical** while English speech works fine. This appears to be a model training limitation, not a code bug. Worth mentioning in docs to set expectations.

Happy to share more benchmarks (512x512 / 10s / 20 steps = 32min on M4 Max) or collaborate on documentation!


> **Comment by PierpaoloV (2026-08-22T08:40:59Z):**

@Henninges  Yes, i think is a coincidence. 
I haven't chosen the name myself and let GPT 5.6 Sol do it for me. On my side, my responsability not to doublecheck if someone else already used that name. 


> **Comment by PierpaoloV (2026-08-22T08:44:00Z):**

@Henninges Also, as you can see from my screen, my hardware is limited and I haven't been able to achieve great results for now, but I will do more configuration tests. 
I will also take into account your suggestions and of course will be happy to collaborate on documentation!
 


> **Comment by Henninges (2026-08-22T08:49:12Z):**

@PierpaoloV : Haha, no worries about the name – great minds (and LLMs; GPT 5.6 Sol vs Qwen3.8 Max) obviously think alike! 😄

And hey – if your hardware is limited, I'm happy to be your test bench: 
I can run benchmark/config tests for your PR on my M4 Max (128 GB). 
Just tell me which configurations you'd like validated (token-reduction, 
core-reuse, presets, ...). That's what community is for, right?

Looking forward to collaborating on the docs!





### #44 [open] PR: Align causal GQA threadgroup memory length to 16 bytes
URL: https://github.com/antirez/h3.c/pull/44  by jordanswilson 2026-08-21T02:18:52Z

`setThreadgroupMemoryLength` requires the length to be a multiple of 16 bytes, but `h3_gpu_gqa_causal_bf16` sizes the causal score buffer as `sequence * sizeof(float)` — out of spec whenever the prompt's token count is not a multiple of 4.

The driver tolerates this silently, so ordinary runs work. But with the Metal API validation layer active (`MTL_DEBUG_LAYER=1` — which any child process inherits from an Xcode-launched app with default scheme diagnostics), the first text encoder dispatch aborts:

```
-[MTLDebugComputeCommandEncoder setThreadgroupMemoryLength:atIndex:]:799:
failed assertion `length(140) must be a multiple of 16 bytes.'
```

(140 bytes = a 35-token prompt.)

## Fix

Round the threadgroup memory length up to the next 16-byte multiple. The kernel never reads the padding, and the rounding happens before the `maxThreadgroupMemoryLength` capacity check so that check stays accurate. Behavior for already-aligned sequence lengths is unchanged.

## Verification

The bundled `h3_text_tests` can't catch this — its fixture sequence length of 32 is already a multiple of 4. I verified with a minimal fixture-free repro calling `h3_gpu_gqa_causal_bf16` at sequence 35 (synthetic bf16 tensors, same head geometry as the Qwen text encoder):

- pre-fix, `MTL_DEBUG_LAYER=1`: aborts with the exact assertion above (SIGABRT)
- post-fix, `MTL_DEBUG_LAYER=1`: passes, all outputs finite
- post-fix, no validation: passes — no change for normal runs

Happy to add the repro as a test if you'd like it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)


### #43 [open] PR: Add CUDA backend for Linux/NVIDIA (GB10-tested) and h3c studio web UI
URL: https://github.com/antirez/h3.c/pull/43  by matrixfede 2026-08-20T07:18:37Z

Adds a native CUDA backend for Linux/NVIDIA plus **h3c studio**, a self-hosted web front end for the same binary.

Supersedes my earlier three-commit shape: the branch is now 20 commits (CUDA port → hardening → web UI → installer/docs/CI → branding). Related to #42. Independent of #45 — "H3 Studio" there is a different project; "h3c studio" here is the browser UI in this PR.

## What is in here

**1. CUDA backend (`h3_gpu.h` implementation)**
- cuBLASLt GEMMs, custom kernels for the H3 op set, optional cuDNN SDPA enabled at build time via `CUDNN_ROOT` / `CUDNN_FRONTEND_ROOT`, with a custom tiled attention fallback when cuDNN is absent.
- Build-time backend selection; the Metal path on macOS, the CLI semantics, the checkpoint formats, and all model logic are unchanged.

**2. Portable host code**
- C11 + ICU tokenizer replacing the Accelerate/Objective-C dependencies so the same sources build on Linux and macOS.

**3. h3c studio (web UI)**
- FastAPI backend + React front end driving the same `h3` binary: every CLI option is reachable from the browser, generation progress streams live with a weighted progress bar, uploads (photos/clips) live in a reusable library.
- Multi-user by design: one admin account bootstrapped from the environment, every other account created through a single-use invite; jobs and uploads are private per user. Binds to `127.0.0.1` by default; auth is mandatory in the shipped `docker-compose.yml`.
- Docs: [`docs/WEBUI.md`](docs/WEBUI.md), post-processing notes: [`docs/POSTPROCESSING.md`](docs/POSTPROCESSING.md).

**4. Installer**
- `install.sh` checks prerequisites, clones, writes `.env`; it downloads nothing (the ~465 GB checkpoint and the optional face-swap runtime are separate, declined-by-default questions) and is meant to be read, not piped.

**5. Tests and CI**
- C suite (`make test`, includes the Metal/MLX parity path), 182 backend tests, frontend test suite, Docker-compose smoke test; a CI workflow runs the JS/Python jobs on push (CPU-only runners — see caveat below).

## Measured on NVIDIA GB10 (DGX Spark, CUDA 13.0, driver 595.84)

Median of three runs, released checkpoint, max-quality preset (1024x576, 107 frames, 50 steps/layers) — reproducible details in [`docs/GB10_PROFILE.md`](docs/GB10_PROFILE.md):

- full render wall clock: 33:36.80 → 18:56.36 (**1.78x**)
- video VAE decode: 1112.96 s → 247.74 s (**4.49x**, tiled F32 attention kernel)
- DiT long-shape attention: custom tiled kernels; optional cuDNN SDPA BF16 path when the library is present at build time
- `--ssd-streaming`: exact (SSIM 1.0) low-memory mode, DiT peak 27.06 GB → 1.63 GB at +37.6% load+denoise time — kept opt-in, not default
- validation: CPU oracles max_abs ≤ 2e-5, Compute Sanitizer clean, matched renders SSIM 0.999303 / PSNR 55.45 dB (short gate), bit-identical rerenders

API coverage and thresholds: [`docs/CUDA_GPU_API_INVENTORY.md`](docs/CUDA_GPU_API_INVENTORY.md).

## Known limits (stated up front)

- `--use-int8-row-fc2` is a Metal/M5 specialization and a measured CUDA no-op today; there is no verified FP8 execution path.
- The web UI is validated end to end on GB10 (Docker and the no-Docker path) — not against the Metal binary.
- CI runners are CPU-only: they cover the backend/frontend/compose layers, not CUDA compilation; `./scripts/verify.sh all` on the GPU box remains the release gate.
- macOS keeps using the existing Metal backend; nothing in this PR changes its behavior.

## Suggested review path

1. `README.md` (hero + `h3c studio` section, then the CUDA notes in "Platforms and prerequisites")
2. `git log --oneline upstream/main..HEAD` for the commit grouping
3. `docs/GB10_PROFILE.md` for the measurement protocol
4. `webui/` + `docker-compose.yml` for the studio

Happy to split this into smaller PRs (e.g. CUDA backend vs. web UI) if a single 134-file review is too much — say the word.



### #42 [open] ISSUE: CUDA backend for Linux/NVIDIA — interest in upstreaming?
URL: https://github.com/antirez/h3.c/issues/42  by matrixfede 2026-08-20T07:18:21Z

I ported h3.c to Linux ARM64 with a native CUDA backend behind the existing `h3_gpu.h` API (CLI, checkpoint formats and the Metal path unchanged; backend selected at build time).

Measured on an NVIDIA GB10 (DGX Spark, CUDA 13.0), max-quality preset (1024x576, 107 frames, 50 steps/layers):
- full render: 33:37 -> **18:56 wall clock (1.78x)**
- video VAE decode: 1113 s -> 248 s via a tiled F32 attention kernel (4.49x)
- DiT attention: cuDNN SDPA BF16 (~84x on the long shape) with custom tiled fallback
- parity gates: CPU oracles (max_abs <= 2e-5), Compute Sanitizer clean, matched renders SSIM >= 0.999 on the short gate

The branch is ~6.8k lines across 3 commits (portable host/tokenizer, CUDA backend, integration + optimizations). Happy to open a PR if this is of interest, and to split or rework anything.


### #41 [open] ISSUE: Feature request: 4bit and 8bit support
URL: https://github.com/antirez/h3.c/issues/41  by faifung01 2026-08-19T13:22:48Z

bf16 is more than 80G. Can I get support with 4bit/8bit to solve ram and storage? Thank you!

> **Comment by monotected (2026-08-20T21:58:05Z):**

check forks - allready done 




### #40 [closed] PR: docs: fix grammar - "the name of the authors" should be "the names of the authors"
URL: https://github.com/antirez/h3.c/pull/40  by lunar-me 2026-08-17T16:05:49Z

## Description

This PR fixes a minor grammar error in the THIRD_PARTY_NOTICES.md file.

## Problem

On line 19 of `THIRD_PARTY_NOTICES.md`:

```markdown
Neither the name of the authors nor the names of its contributors may be
```

Singular "name" is used with plural "authors", creating a noun-number agreement error.

## Fix

Changed "the name of the authors" to "the names of the authors":
```markdown
Neither the names of the authors nor the names of its contributors may be
```

## Type of change

- [x] Documentation fix (grammar)


### #39 [closed] ISSUE: mosaic  output
URL: https://github.com/antirez/h3.c/issues/39  by monotected 2026-08-17T11:49:47Z

Good afternoon!
Thank you for the wonderful repository.

I’m having a small issue with the output mosaic image.
mac m5pro 24 gb  macos 26.5.2 

Trying with different args :

"A hummingbird hovering over red flowers." --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 --ssd-streaming
"A hummingbird hovering over red flowers." --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 1 --ssd-streaming
"A hummingbird hovering over red flowers." --width 512 --height 768  --frames 22 --steps 20 --layers 45 --reuse 2 

model downloaded from official repo :
......_HUB_ENABLE_HF_TRANSFER=1 hf download MiniMaxAI/MiniMax-H3 \
    --local-dir ./MiniMax-H3 --include "FL2VA/*" 
130+ gb

cli shows no errors 

https://github.com/user-attachments/assets/451cf2bd-b2da-46b4-9458-37b309a42287


> **Comment by monotected (2026-08-17T12:28:44Z):**

There was some kind of bug.
Everything started working after a clean reinstall.

