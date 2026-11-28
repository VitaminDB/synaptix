# sgl-project__sglang__cookbook-minimax-h3

Source: https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx
Fetched: 2026-11-28



---

## FILE: docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx (https://github.com/sgl-project/sglang/blob/HEAD/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)

---
title: MiniMax-H3
description: Run native MiniMax-H3 video-and-audio generation with SGLang Diffusion.
metatags:
    description: "Serve MiniMax-H3 with SGLang Diffusion for text-to-video-and-audio, first/last-frame conditioning, video-to-video, and multimodal reference conditioning."
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/MiniMaxAI/minimax-h3.jsx";

<DiffusionModelTags tags={["video + audio", "T2VA / FL2VA / Ref2VA", "multimodal references", "4–15 seconds", "768p"]} />

## 1. Quick start

Follow the [SGLang Diffusion installation guide](/docs/sglang-diffusion/installation),
then choose a verified recipe below. Docker is recommended where the picker offers it.
Setup changes the deployment; Server and Request expose
orthogonal startup and sampling choices.

<Deployment config={config} />

<Note>
The generated Server command already includes the recommended encoder policy.
Change a Server option only for a deliberate trade-off; Request options do not
reload the model.
</Note>

The Docker form uses the image's bundled diffusion dependencies without reinstalling
them at startup. For conditioned requests, set **Host media directory**
under **Variables**; the builder mounts it read-only at `/data/minimax-h3`.
AMD and Grace Blackwell currently offer the Python form; the other NVIDIA
recipes also offer Docker.

To use ModelScope through the same normal `sglang serve` path, prefix the copied
command with `SGLANG_USE_MODELSCOPE=true` and replace the model path with
`MiniMax/MiniMax-H3`. Keep the selected variant and topology flags unchanged.

For platform-specific installation details, see the
[SGLang Diffusion installation guide](/docs/sglang-diffusion/installation).

## 2. Model capabilities

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is a native joint video-and-audio model for text-to-video-and-audio, first/last-frame control, and multimodal reference conditioning. Its main strength is producing the picture and stereo soundtrack together, so speech, music, ambient sound, and visible events can stay aligned without a separate audio-generation pass.

Choose H3 when synchronized audiovisual output or reference-driven generation matters more than a lightweight deployment. The released recipe targets a 768-pixel short edge at 24 fps for 4–15 seconds, and its capabilities are split across two checkpoint partitions; serving every mode therefore requires separate FL2VA and Ref2VA deployments, or [one server that loads both transformers](#one-server-for-both-partitions).

| Task | `task` value | Conditioning |
| --- | --- | --- |
| Text to video and audio | `t2va` | Text prompt only |
| First/last frame to video and audio | `fl2va` | First frame, last frame, or both |
| Reference to video and audio | `ref2va` | Image, video, and audio references, optionally combined with first/last keyframes |

Video-to-video (V2V) is a supported `ref2va` use case, not a fourth task
value. Run the `Ref2VA` partition and provide a video reference in
`conditions`. A hybrid `ref2va` request may also include the same ordered
first/last keyframes accepted by `fl2va`, but it must still contain at least one
reference condition.

Use the selected Hub's root model ID: `MiniMaxAI/MiniMax-H3` on Hugging Face
or `MiniMax/MiniMax-H3` on ModelScope. Select the checkpoint variant with
`--model-variant`: `fl2va` serves both `t2va` and `fl2va`, while `ref2va`
serves reference-conditioned requests, and `fl2va+ref2va` loads both
transformers into one server. SGLang owns the checkpoint-directory
mapping; do not point `--model-path` at a manually downloaded subdirectory.
The same FL2VA / Ref2VA split applies in
[ComfyUI integrated mode](#comfyui-integrated-mode).

<Warning>
Review the license and usage terms in the MiniMax-H3 model card before production or commercial use. SGLang support does not grant additional model usage rights.
</Warning>

### Image checkpoints

For the separate six-step text-to-image and editing checkpoint, see
[MiniMax-H3-Image](/cookbook/diffusion/MiniMax/MiniMax-H3-Image).

## 3. Deployment details

The builder accepts legal custom GPU counts and topologies, marking them
**Unverified** until the exact recipe has completed end-to-end validation.
Static H3 head or partition violations disable Copy before they reach
`sglang serve`.

### Grace Blackwell

Select **GB300** or **GB200** in the builder. The default is one Linux ARM64
host with four GPUs, Ulysses4, and resident weights. This follows the
[four-GPU compute-tray layout](https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html);
an NVL72 rack is not a single 72-GPU process host.

For two four-GPU hosts, set **Nodes = 2** and **GPUs / node = 4**. The builder
emits Ulysses4 inside each host, Ring2 across hosts, and replicated encoders.
Set the same reachable **Head address** on both hosts and use node ranks 0 and
1. Run the generated command once per host, with the same SGLang revision and
dependencies. Let `sglang serve` resolve and download the selected weights.

GB200 is a **derived, unverified recipe**, not a measured GB300-equivalent
performance claim. Keep the native precision and platform-default attention
for the baseline; the B200/B300 quantization measurements and H200 Cache-DiT
quality audit do not establish coverage on Grace Blackwell.

GB300 end-to-end coverage currently includes single-host FL2VA text-to-video
with audio, native precision, eager execution, and `quality="exact"`.
Ref2VA and multi-host commands remain **Unverified**. See the
[GB300 measurements](#gb300-single-host) for the tested workload.

### Checkpoint and adapter formats

Start with the command emitted by the builder. Every row below is an overlay on
the same native SGLang pipeline; component repositories contribute their own
config and weights, while weight files retain the base component config. Storage
layout and inference behavior are separate contracts: for example, a PEFT file
may be either a normal style adapter or a timestep-distilled Turbo adapter.

| Scope | Format or variant | Add to the base command | Contract |
| --- | --- | --- | --- |
| Full model | Official mixed BF16/FP32, CFG-distilled | `--model-variant fl2va` or `--model-variant ref2va` | Lossless reference and consistency GT path. CFG distillation removes the negative branch; it is not the few-step timestep distillation used by Turbo releases. |
| DiT | Official Diffusers component layout | `--component-paths.transformer MiniMaxAI/MiniMax-H3/transformer` (`fl2va`) or `.../transformer_ref` (`ref2va`) | Loads the official component through the native SGLang graph; no Diffusers runtime fallback. |
| DiT | [AdaLN-pruned Diffusers component](https://huggingface.co/multimodalart/MiniMax-H3-Pruned) | `--component-paths.transformer multimodalart/MiniMax-H3-Pruned/transformer` or `.../transformer_ref` | Approximate curve-AdaLN architecture; its config and basis metadata are loaded natively. |
| DiT | [Full or AdaLN-pruned](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models), or [LoRA-merged/remixed](https://huggingface.co/FX-FeiHou/MiniMax-H3-Remix) BF16 safetensors | `--component-weights-paths.transformer OWNER/REPO/path/FILE.safetensors` | Weight-only override for a native full/pruned H3 layout. Match the FL2VA/Ref2VA partition; pruned, merged, or dtype-converted exports are approximate, and any author-specific sampler remains a separate requirement. |
| DiT | FL/Ref hybrid fine-tunes, including [Singularity v1.3](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity) (full or pruned INT8) | Replace `--model-variant` with `hybrid`; add `--component-weights-paths.transformer OWNER/REPO/FILE.safetensors` | Explicit merged-weight deployment serving `t2va`, `fl2va`, and `ref2va` on one native pipeline. Quantization and pruning are detected from the file; INT8 requires `comfy-kitchen`. See [Singularity](#singularity-hybrid-weights). |
| DiT | Comfy FP8 or self-describing MXFP8 safetensors | `--component-weights-paths.transformer OWNER/REPO/path/FILE.safetensors` | Per-layer metadata selects static/dynamic FP8 or MXFP8 automatically. |
| DiT | ConvRot INT8, W4A8, W4A4, or mixed W4A4+INT8 safetensors ([INT8](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models), [W4A8](https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI), [W4A4](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot)) | `--component-weights-paths.transformer OWNER/REPO/path/FILE.safetensors` | Auto-detected. INT8 layers run on SGLang's JIT-compiled ConvRot ops on CC 9.0, 10.0, 12.0 and 12.1 and on `comfy-kitchen` elsewhere; W4A8 and W4A4 layers require `comfy-kitchen`. TP must preserve each file's ConvRot group boundaries. |
| DiT | [NVFP4, optionally mixed with INT8 or FP8](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot) | `--component-weights-paths.transformer OWNER/REPO/path/FILE.safetensors` | Auto-detected; NVFP4 execution requires NVIDIA compute capability 10.0+. |
| DiT | [AutoRound W4A16 component](https://huggingface.co/Ar4ikov/MiniMax-H3-transformer-W4A16-RTN) | `--component-paths.transformer Ar4ikov/MiniMax-H3-transformer-W4A16-RTN` | Self-describing Diffusers component; SGLang reuses the SRT GPTQ/Marlin backend. The linked export is FL2VA. |
| DiT | GGUF, full or AdaLN-pruned ([full](https://huggingface.co/leejet/MiniMax-H3-GGUF), [pruned](https://huggingface.co/unsloth/MiniMax-H3-GGUF)) | `--component-weights-paths.transformer OWNER/REPO/FILE.gguf` | CUDA capacity path; aligned TP and layerwise offload are supported, FSDP and LoRA are not. |
| Text encoder | Architecture-compatible Qwen3-VL BF16 finetune ([Heretic example](https://huggingface.co/llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic)) | `--component-paths.text_encoder OWNER/REPO` | Reuses the native H3 extractor, including its vision tower and layer-50 hidden-state selection. Finetuning changes conditioning, not the sampling schedule. |
| Text encoder | [Serialized FP8 component](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) | `--component-paths.text_encoder Qwen/Qwen3-VL-32B-Instruct-FP8` | Only eligible language-model linears use FP8; embeddings, norms, and the vision tower keep their declared precision. |
| Text encoder | ConvRot INT8, W4A8, or W4A4 safetensors ([INT8](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders), [Heretic INT8](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot), [W4A8](https://huggingface.co/Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI), [W4A4](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot)) | `--component-weights-paths.text_encoder OWNER/REPO/path/FILE.safetensors` | Auto-detected. INT8 layers run on SGLang's JIT-compiled ConvRot ops on CC 9.0, 10.0, 12.0 and 12.1 and on `comfy-kitchen` elsewhere; W4A8 and W4A4 layers require `comfy-kitchen`. INT8/W4A8 files may include a scalar-scale INT8 embedding, which requires `comfy-kitchen`. Unmarked tensors retain their declared precision. |
| Text encoder | Comfy NVFP4 with dynamic activation quantization ([Heretic example](https://huggingface.co/sakamakismile/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4)) | `--component-weights-paths.text_encoder OWNER/REPO/path/FILE.safetensors` | Requires `comfy-kitchen` and NVIDIA compute capability 10.0+. Per-layer metadata selects native NVFP4 matmul; scalar/row-wise INT8 embeddings remain packed. |
| Text encoder | [NVFP4-AWQ](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) or [Quanto qint8](https://huggingface.co/DeepBeepMeep/MiniMax-H3/tree/main/Qwen3-VL-32B-Instruct) safetensors | `--component-weights-paths.text_encoder OWNER/REPO/path/FILE.safetensors` | Memory-oriented formats: compressed storage is restored, then each active matrix uses BF16/FP16 compute. |
| Text encoder | [GGUF Qwen3-VL](https://huggingface.co/DeepBeepMeep/MiniMax-H3/tree/main/Qwen3-VL-32B-Instruct) | `--component-weights-paths.text_encoder OWNER/REPO/FILE.gguf` | CUDA capacity path with encoder TP/layerwise support; encoder FSDP is not supported. |
| Text encoder | [Compact Qwen3-VL 4B/8B + ClipProj](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) | `--component-paths.text_encoder ENCODER_REPO --component-paths.conditioning_projection PROJECTION.safetensors` | Approximate conditioning replacement. A separate weight-only override may quantize the selected small encoder. |
| DiT or adapter | Timestep-distilled Turbo, as merged weights or LoRA ([Larry](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora), [LightX2V](https://huggingface.co/lightx2v/Minimax-h3-Turbo), [merged Ref2VA INT8 example](https://huggingface.co/PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot)) | Use `--component-weights-paths.transformer ...` for merged weights, or `--lora-path OWNER/REPO --lora-weight-name FILE --lora-merge-mode auto` for LoRA | Few-step semantic variant. Pin the exact FL2VA/Ref2VA file and its NFE/sigma schedule, scale, and alpha; storage-format detection does not infer sampling behavior. See [LoRA recipes](#5-lora-recipes). |
| Adapter | Style, subject, or behavior LoRA ([example](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA)) | `--lora-path OWNER/REPO [--lora-weight-name FILE] --lora-merge-mode auto` | Native fused and Diffusers/PEFT layouts are normalized at load time. Keep the base schedule unless the author specifies another one, and preserve any trigger phrase, scale, and alpha metadata. |

The rows compose rather than enumerate every cross-product. A Turbo-merged INT8
ConvRot checkpoint, for example, must satisfy both the Turbo sampling contract
and the ConvRot storage/backend contract.

H3's Qwen3-VL text encoder also handles image understanding; there is no separate
`image_encoder` component. Select the conditioning checkpoint, not an optional
generation-tail file for prompt rewriting. The `uncensored` or `Heretic` label
describes a weight modification, not a separate loader or a guarantee of output quality.

For H3, the registered component names are `transformer`, `text_encoder`,
`video_vae`, and `audio_vae`. The shorter `--transformer-weights-path` and
`--text-encoder-path` aliases remain supported. `conditioning_projection` is an
H3 text-encoder sidecar key, not a standalone model component. Plain
video/audio VAE safetensors can use `--component-weights-paths.video_vae` or
`--component-weights-paths.audio_vae`, but SGLang does not currently advertise
a native quantized H3 VAE format.

Pre-quantized files are self-describing: do not combine those rows with
`--quantization` or `--component-quantizations.*`. Only byte-identical official
full weights—whether loaded from the original model, Diffusers component, or a
weight-only layout—belong to the consistency GT. Pruned, quantized,
compact-encoder, and LoRA routes are outside that baseline and the audited
`quality="high"` contract. Packed and per-layer mixed formats reject FSDP unless
their row says otherwise; see
[Quantization](/docs/sglang-diffusion/quantization) for backend-wide constraints.

LoRA tensors alone do not make an execution-coupled release portable. Sparse-
attention/SLA adapters and causal-streaming adapters such as RAVEN also require
their matching attention or streaming pipeline; they are not standard H3 LoRA
overlays in SGLang. Likewise, a remixed checkpoint that prescribes a custom
sampler is only covered when that sampler contract can be reproduced—the fact
that its safetensors layout loads is not sufficient.

For a four-card H200 host, keep the full BF16/FP32 model resident by default.
The model fits without FSDP, so this path avoids the per-block parameter
all-gathers of the memory-oriented FSDP profile:

```bash 4×H200 resident
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --encoder-parallel auto \
  --performance-mode speed \
  --port 30010
```

Pure Ulysses4 is also the faster measured topology on H200, not just a
capacity default. The 4×H100 TP2 + Ulysses2 recipe below fits on 141 GB H200
cards, but it replaces the Ulysses all-to-all exchange with two per-block
tensor-parallel all-reduces and measured slower end-to-end, at about 30 GB
lower peak memory per GPU. See the **H200 topology comparison** in the
Benchmarks section for the measured numbers; treat TP2 + Ulysses2 on H200 as
a deliberate memory trade, not a latency default.

For 4×H100 80 GB, balance the large packed activation with resident weight
sharding. TP2 + Ulysses2 was the fastest measured lossless topology while the
Qwen encoder still folds across all four GPUs:

```bash 4×H100 fastest
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --tp-size 2 \
  --ulysses-degree 2 \
  --encoder-parallel auto \
  --performance-mode speed \
  --port 30010
```

Pure Ulysses4 could not keep the full pipeline resident on 80 GB H100s. Use
`--tp-size 4 --ulysses-degree 1` when lower resident memory matters more than
the last few percent of latency. FSDP remains a verified capacity option, but
its per-block weight all-gathers do not make it the H100 speed default:

```bash 4×H100 FSDP capacity
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --encoder-parallel auto \
  --performance-mode speed \
  --use-fsdp-inference true \
  --port 30010
```

For a two-card RTX 5090 host, use TP2 and keep 20 DiT blocks
resident. Layerwise placement is lossless: it changes parameter placement and
transfer scheduling, not the BF16/FP32 denoising or VAE math. This is the
fastest measured 32 GB operating point:

```bash 2×RTX 5090 fastest lossless
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 2 \
  --tp-size 2 \
  --ulysses-degree 1 \
  --encoder-parallel auto \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder,vae \
  --dit-offload-prefetch-size 1 \
  --dit-layerwise-resident-layers 20 \
  --enable-torch-compile false \
  --port 30010
```

The DiT residency and prefetch knobs apply only to the repeatedly executed DiT
blocks. The text encoder and the video VAE decoder blocks use one-layer
prefetch with zero resident layers. The video VAE encoder stays resident
because its indexed down blocks cannot host executable layerwise hooks; the
roughly 577 MiB audio VAE also stays resident because offloading it only adds
transfer overhead. This exact recipe was validated on
2× RTX 5090 (32 GB each) and a 377 GiB host; use a 384 GiB-class machine. The
latency and memory comparison is collected in the benchmark section below.

For a single 24 GB consumer card (RTX 4090), stream the DiT and text encoder
and quantize DiT linear layers online with `convrot_int8`. Keep `vae` out of
`--layerwise-offload-components`: putting the VAE decoder in layerwise
offload re-streams about 9 GiB on each of 167 decode tiles. Default
attention stays `fa` (exact). Approximate backends are opt-in; see
[Attention Backends](/docs/sglang-diffusion/attention_backends#sage-then-sol-hybrid).
On the RTX 4090 (CC 8.9) `convrot_int8` runs on `comfy-kitchen`, so install it
first (`pip install comfy-kitchen`); on CC 9.0, 10.0, 12.0 and 12.1 it uses
SGLang's JIT-compiled ConvRot ops instead (built once per machine on first use).

```bash 1×RTX 4090 24GB
sglang generate \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --quantization convrot_int8 \
  --attention-backend fa \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder \
  --dit-offload-prefetch-size 1 \
  --dit-layerwise-resident-layers 0 \
  --enable-torch-compile false \
  --prompt "A cat walking on a sunny beach, gentle waves." \
  --save-output
```

The same flags work on `sglang serve`. Drop `--quantization` for the BF16
baseline; everything else stays identical. GPU peak stays about 18 GB
either way because streaming offload is set by the offload buffers and VAE
decode, not the weight dtype.

To skip DiT steps on 4090 while keeping `fa`, add Spectrum to the generate
command above. Do not combine it with `quality="high"` or Cache-DiT.
On this card, 50-step BF16+FA (**1008.6 s**) drops to **285.3 s (3.53×)**
with exact attention, up to **219.0 s (4.61×)** with Sage→Sol hybrid +
Spectrum. See the 4090 table in Benchmarks.

```bash
  --enable-spectrum \
  --spectrum-warmup-steps 11 \
  --spectrum-window-size 5 \
  --spectrum-flex-window 1.0
```

The first launch resolves every selected source through the normal Hub path. If
a repository requires authentication, export a Hugging Face token in the
server environment; no manual pre-download is required.

For MiniMax-H3, `--performance-mode speed` deliberately keeps the DiT eager.
The current `torch.compile` path changes the model's numerical output, so no
recommended lossless preset enables it implicitly. An explicit
`--enable-torch-compile true` remains available for controlled experiments, but
do not use it to generate consistency ground truth.

### One server for both partitions

`--model-variant fl2va+ref2va` keeps the FL2VA and the Ref2VA transformer in
one server. The text encoder, both VAEs, the tokenizer and the processor are
the same files in the two partitions and load once, from `FL2VA/`; the Ref2VA
transformer loads beside the FL2VA one. Requests need no extra field: a
`ref2va` request runs on the Ref2VA transformer, `t2va` and `fl2va` requests
on the FL2VA one. Both partitions must declare the same sampling contract in
`model_index.json`; the server checks this at load.

```bash Command
sglang serve --model-path Pika-Labs/Pika-H3-Flex --model-variant fl2va+ref2va --port 30010
```

- **Memory.** Each GPU that holds the DiT holds both transformers (61.73 GiB
  of weights each) beside one text encoder and one pair of VAEs. The automatic
  memory policy places them at startup from the free memory of the least-free
  GPU:
  - from 181.7 GiB (the 120 GiB H3 keeps for one transformer and the rest of
    the pipeline, plus the second transformer), both stay resident, as on
    B300;
  - from 120 GiB, as on H200, both use component offload: the transformer a
    request runs moves onto the GPU and stays there, so only a request for the
    other partition's tasks moves weights, and it waits for the swap;
  - below 120 GiB the server does not start, and the error names the flags
    that place the transformers explicitly.

  Explicit placement (`--dit-cpu-offload`, `--component-residency`,
  `--layerwise-offload-components`) replaces this choice; with
  `--dit-cpu-offload true` every request moves its transformer onto the GPU
  and back. Validated on 2× B300 and 2× H200 (Ulysses 2, 4 s at a 480 short
  edge, MiniMax-H3 at 8 steps): on B300 a request after a switch denoises as
  fast as one without; on H200 the denoise stage took 7.7–8.1 s after a swap
  against 5.0–5.3 s without one, and 57.5 s and 46.6 s at the first swap to
  each transformer.
- **Adapters.** [Pika H3 Flex](#pika-h3-flex-six-step-adapters) applies each
  partition's bundled adapter to its own transformer. `--lora-path` applies an
  adapter to both; the [LoRA API](/docs/sglang-diffusion/api/openai_api#lora-management)
  addresses one with `target`: `transformer` (FL2VA) or `transformer_2` (Ref2VA).
- **Weights.** For `MiniMaxAI/MiniMax-H3` the server also downloads `Ref2VA/`
  of the same revision. The Hub cache keeps one copy of the files the two
  partitions share, so this adds the Ref2VA transformer (61.7 GiB). Pika H3
  Flex materializes both. Component and weight overrides are **Unverified**
  with this variant.
- **Limits.** The startup warm-up request is `t2va`, so the first `ref2va`
  request also warms the Ref2VA transformer. Cache-DiT, `quality: "high"` and
  `--minimax-h3-adaln-cache-path` need a single-partition server.

### ComfyUI integrated mode

The [ComfyUI plugin](/docs/sglang-diffusion/comfyui) can run H3 as a per-step
DiT backend. ComfyUI keeps CLIP, VAE, and the sampler loop. SGLang loads only
the transformer in ComfyUI integrated mode (the plugin sets `comfyui_mode`; there is no CLI flag).

Copy `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion` into ComfyUI
`custom_nodes/`. Load a shipped graph and set `SGLDOptions.model_type` to
`minimax_h3`:

| Graph | Task | DiT filename |
| --- | --- | --- |
| `minimax_h3_t2v_sgld.json` | T2V / I2VA / FL2VA | `minimax_h3_fl2va_*.safetensors` |
| `minimax_h3_r2v_sgld.json` | Reference-to-video | `minimax_h3_ref2va_*.safetensors` |
| `minimax_h3_t2v_sgld_upscaler.json` | Low-res then 3D ×2 | `fl2va` DiT |

Pruned BF16 / FP8 files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models)
are the usual loader inputs. GGUF goes in `SGLDOptions.transformer_weights_path`.
On a 24 GB card put the 32B CLIP on CPU and enable DiT layerwise offload; use
`convrot_int8` or GGUF rather than the full unpruned BF16 DiT.

### Singularity hybrid weights

[Singularity](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity) is an
FL/Ref fusion fine-tune, not a complete pipeline repository. Keep the official
model ID for the encoders and VAEs, and select one exact transformer file:

```bash 2×B300 hybrid checkpoint
pip install comfy-kitchen

WEIGHTS=WarmBloodAban/Minimax-h3_Singularity/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors
sglang serve --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant hybrid \
  --component-weights-paths.transformer "$WEIGHTS" \
  --num-gpus 2 --ulysses-degree 2 --performance-mode speed \
  --port 30000
```

For the full INT8 checkpoint, replace the filename with
`Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors`. SGLang downloads the
selected file automatically. Do not add an online quantization override.
`hybrid` requires an explicit weight override; it does not convert the official
Ref2VA weights into a multimode model. Request `task` and `conditions` retain
their [usual meanings](#4-generate-video-and-audio), including `conditions: []`
for `t2va` and keyframes for `fl2va`.

Start with `quality: "lossless"` and the standard 50-point schedule. Here
`exact` disables additional request-time approximations; it does not undo
INT8 quantization, pruning, or fine-tuning. The author's recommended Ref2V
Turbo LoRA is a separate optional adapter, not an automatically inferred
4-step schedule. Reference fidelity can differ from the official model; see
the [author's guidance](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/discussions/1)
before applying the [LoRA sampling contract](#5-lora-recipes).

### Advanced: precomputed AdaLN cache

The [model card](https://huggingface.co/MiniMaxAI/MiniMax-H3) notes that about
13B H3 parameters are AdaLN branches whose outputs can be precomputed for
inference. The public base checkpoint contains the original branches, not a
ready-to-use cache. SGLang therefore keeps the standard path as the default.

<Warning>
This is an experimental deployment path. It is intentionally disabled unless
you provide an explicitly generated cache; end-to-end numerical and peak-memory
validation remains required before using it in production.
</Warning>

When an inference-only deployment has a fixed sampling schedule, build a cache
from the already materialized transformer directory on CUDA, then pass it to
the usual `sglang serve` command. This does not alter the denoising formula:
the cache stores the BF16 outputs of the original AdaLN linears.

```bash Command
python -m sglang.multimodal_gen.tools.build_minimax_h3_adaln_cache \
  --transformer-path "$TRANSFORMER_PATH" \
  --model-variant fl2va \
  --mode t2va \
  --num-inference-steps 50 \
  --flow-shift 12 \
  --audio-flow-shift 3 \
  --output /models/minimax-h3-fl2va-adaln-50step.safetensors

sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --minimax-h3-adaln-cache-path /models/minimax-h3-fl2va-adaln-50step.safetensors \
  --num-gpus 4 \
  --tp-size 2 \
  --ulysses-degree 2 \
  --port 30010
```

`$TRANSFORMER_PATH` is the `FL2VA/transformer` or `Ref2VA/transformer`
directory in the normal SGLang/Hugging Face snapshot; the builder never
downloads a second copy. A cache only covers the scheduler settings used to
create it, including its mode, step count, flow shifts, and condition noise
values. SGLang rejects a request outside that coverage instead of silently
changing conditioning. Cache mode supports the matching unquantized checkpoint
only.

### Advanced: online AdaLN rebuild with a host cache

`--minimax-h3-adaln-online true` needs no prebuilt artifact: the server drops
the 24.2 GiB of `adaln_proj` weights from the GPU and computes each request's
AdaLN outputs from the checkpoint on demand, bit-exact with the resident-weight
path. Because a rebuild pass streams the whole 24.2 GiB, the first request of
every new `(task shape, num_inference_steps, flow_shift, audio_flow_shift)`
combination pays several seconds; after that the plans are served from a
64-slot GPU slab (LRU per plan) backed by a pinned host cache
(`--minimax-h3-adaln-host-cache-gb`, LRU per schedule, default 8 GB per
rank), so mixed-schedule serving does not re-read the checkpoint. Plan sets
that exceed the host budget are simply recomputed on their next occurrence.
Expert escape hatches live in environment variables:
`SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS` resizes the GPU slab (needed
only above 64 inference steps) and `SGLANG_DIFFUSION_MINIMAX_H3_ADALN_FP32`
computes the one-time projections in fp32 (experimental; not bit-comparable
to resident weights, validate end-to-end before production). LoRA adapters
that modify `adaln_proj` are rejected in both cache modes rather than
silently ignored.

Both cache modes hold values derived from `adaln_proj`, so a runtime weight
update is only accepted when the cache can follow it: online mode takes a disk
update whose target directory carries native `adaln_proj` safetensors, and
rejects everything else (tensor updates, and directories without those
tensors) before a single weight is written. A sidecar is built offline and
cannot be regenerated in the server, so weight updates are rejected outright;
rebuild the sidecar against the new weights and restart.

### Serve MiniMax-H3 on Ascend NPUs

For Ascend NPU, follow the
[NPU installation guide](/docs/hardware-platforms/ascend-npus/getting-started/installation)
before starting the server.

The Ascend commands below explicitly enable the Cache-DiT configuration used
for the reported performance measurements. Remove these `SGLANG_CACHE_DIT_*`
variables to use lossless denoising. See the **Ascend NPU topology comparison**
in the Benchmarks section for the measured eight- and four-NPU latency.

The measured latency configuration also passes `--dit-cpu-offload false` to
keep the transformer resident on the NPUs. Omit this flag when lower device
memory usage is more important than avoiding CPU-to-NPU transfer latency.

For an eight-NPU host, the validated topology is TP2 + SP4 with Laser
Attention. Use Ascend Flash Attention by replacing `laser_attn` with `fa`.

```bash 8-NPU
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=2 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.4 \
SGLANG_CACHE_DIT_MC=4 \
SGLANG_CACHE_DIT_TAYLORSEER=true \
SGLANG_CACHE_DIT_TS_ORDER=2 \
HCCL_BUFFSIZE=256 sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-type diffusion \
  --model-variant fl2va \
  --dit-cpu-offload false \
  --num-gpus 8 \
  --tp-size 2 \
  --sp-degree 4 \
  --attention-backend laser_attn \
  --port 30088 \
  --component-residency text_encoder=layerwise-offload
```

For a four-NPU host, use TP2 + SP2:

```bash 4-NPU
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=2 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.4 \
SGLANG_CACHE_DIT_MC=4 \
SGLANG_CACHE_DIT_TAYLORSEER=true \
SGLANG_CACHE_DIT_TS_ORDER=2 \
HCCL_BUFFSIZE=256 sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-type diffusion \
  --model-variant fl2va \
  --dit-cpu-offload false \
  --num-gpus 4 \
  --tp-size 2 \
  --sp-degree 2 \
  --attention-backend laser_attn \
  --port 30088 \
  --component-residency text_encoder=layerwise-offload
```

For optional MindIE-SD quantized attention on Ascend 950, follow the
[installation and launch guide](/docs/sglang-diffusion/attention_backends#ascend-quantized-attention-with-mindie-sd)
and use these attention options:

```bash
--attention-backend fa \
--attention-backend-config '{"precision":"mxfp8","head_chunk_size":4}'
```

Attention precision is independent of linear-weight quantization. Keep
`--quantization` unchanged when comparing attention backends. Use
`'{"precision":"bf16"}'` for an explicit native floating-point baseline.
Quantized attention applies to DiT self-attention; the refiner and text encoder
do not inherit the DiT quantization settings.

## 4. Generate video and audio

`num_inference_steps: N` executes **N denoising updates** over N+1 sigma
points, including the terminal zero. Reference implementations that count
sigma points need N+1 to reproduce this schedule. Older SGLang versions ran
N-1 updates for the same request; use N-1 here to preserve their sampling
trajectory. Changing the step count changes the trajectory, not just latency.

MiniMax-H3 uses the asynchronous OpenAI-compatible video endpoint. Choose a
generation mode below, submit a job, poll its status, and then download the
completed MP4.

<Tabs>

<Tab title="T2VA">

MiniMax-H3 supports output durations from 4 through 15 seconds, inclusive. The
following request keeps the verified 5-second profile at a 768-pixel short
edge. MiniMax-H3 resolves the aligned output canvas and frame count from
`target`.

```bash Command
video_id=$(
  curl -sS -X POST http://127.0.0.1:30010/v1/videos \
    -H "Content-Type: application/json" \
    -d '{
      "model": "MiniMaxAI/MiniMax-H3",
      "prompt": "At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny brass instruments, then abruptly file out.",
      "seconds": 5,
      "task": "t2va",
      "conditions": [],
      "target": {
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0
      },
      "num_outputs_per_prompt": 1,
      "num_inference_steps": 50,
      "flow_shift": 12.0,
      "audio_flow_shift": 3.0,
      "seed": 1101
    }' |
    jq -r '.id'
)

while true; do
  status=$(curl -sS "http://127.0.0.1:30010/v1/videos/${video_id}" | jq -r '.status')
  [ "$status" = "completed" ] && break
  [ "$status" = "failed" ] && exit 1
  sleep 1
done

curl -sS -L "http://127.0.0.1:30010/v1/videos/${video_id}/content" \
  -o minimax-h3-t2va.mp4
```

The output contract is an MP4 containing H.264 video at 24 fps and one AAC stereo audio stream at 32 kHz.

</Tab>

<Tab title="FL2VA">

For `fl2va`, provide one or two image conditions with role `keyframe`. The supported frame-index sets are `[0]`, `[-1]`, and `[0, -1]`.

The following request uses one server-local first frame. Use
`frame_index: -1` for a last frame, or include both entries for first-and-last
conditioning.

Choose FL2VA when the supplied image should be the actual first or last frame
of the generated clip. Use image-based Ref2VA instead when the image should
guide identity, style, or composition without being preserved as an endpoint;
Ref2VA may recompose or crop the reference.

```bash Command
curl -sS -X POST http://127.0.0.1:30010/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-H3",
    "prompt": "The supplied frame continues with calm, natural motion and synchronized ambient sound.",
    "seconds": 5,
    "task": "fl2va",
    "conditions": [
      {
        "type": "image",
        "uri": "file:///data/minimax-h3/first-frame.png",
        "role": "keyframe",
        "frame_index": 0
      }
    ],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "auto",
      "duration_seconds": 5.0
    },
    "num_outputs_per_prompt": 1,
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 2101
  }'
```

</Tab>

<Tab title="V2V">

V2V uses the reference-conditioning weights. Launch the server with
`--model-variant ref2va`, keep the request `task` set to `ref2va`, and provide a video
reference in `conditions`. There is no separate `v2v` task value.

Use `type: "video"` when the input may be silent. If the file has a soundtrack,
H3 also uses it as an audio reference. Use `type: "video_audio"` only when both
streams are required; that form rejects an input without audio. The prompt tag
for the visual stream is `<Video 1>`; an available soundtrack is exposed as
`<Audio 1>`.

<Note>
Ref2VA treats the input video as reference material, not as a pixel-aligned
edit source. It can resynthesize or reorder motion and cuts, and it does not
expose a denoising-strength control. Do not rely on it to preserve every source
frame or exact timing.
</Note>

Set `conditions[].start_time_seconds` to select a segment from a longer source.
The default is `0`. SGLang seeks the visual stream and soundtrack to the same
offset, then decodes at most the requested target duration in one pass; the
source is not re-encoded into an intermediate clip.

```bash Command
curl -sS -X POST http://127.0.0.1:30010/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-H3",
    "prompt": "Follow the motion and appearance of <Video 1>, changing the setting to a moonlit bedroom while preserving coherent timing.",
    "seconds": 5,
    "task": "ref2va",
    "conditions": [
      {
        "type": "video",
        "uri": "file:///data/minimax-h3/input.mp4",
        "role": "reference",
        "start_time_seconds": 35.0
      }
    ],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5.0
    },
    "num_outputs_per_prompt": 1,
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 4101
  }'
```

Use `conditions[].uri` for H3 V2V. The generic top-level `video_path`,
`video_url`, and `video_reference` upload fields are not lowered into H3
reference conditions.

</Tab>

<Tab title="Multimodal Ref2VA">

For `ref2va`, first launch the reference-conditioning capability with
`--model-variant ref2va`, then provide conditions with role `reference`.
Image, video, and audio references can be combined. Material tags in the
prompt use the one-based order for each modality.

An image condition here is semantic reference material rather than a
pixel-aligned first frame. Use the FL2VA tab when animating a screenshot from
that exact starting composition.

Set `reference_resize_mode` on the request to choose how reference images are
resized before encoding:

- `max` (default) only downsizes references whose short edge exceeds 2048 pixels.
- `diffusers` scales the short edge to 2048 pixels, including upscaling small
  inputs. Select this explicitly to reproduce the original preprocessing.
- `match` downsizes to the target canvas pixel area when needed, without
  upscaling smaller references. Use this with the
  [ModelTC Ref2VA Turbo LoRA](https://github.com/ModelTC/Minimax-H3-Turbo#note-on-reference-image-resizing),
  whose training uses this policy.

All modes scale without cropping, then round each dimension to the nearest
multiple of 32. This rounding can slightly change the aspect ratio. The option
affects Ref2VA image references, not FL2VA keyframes, video, or audio references.
Changing it changes the conditioning input, so keep it fixed for quality and
performance comparisons; the server does not infer it from the adapter name.
FastH3 Omni retains its previous `diffusers` default; this native H3 change does
not alter that distilled release's preprocessing. Pika H3 Flex also defaults to
`diffusers`, which is how its released pipeline resizes image references from
the 768 tier upwards.

```bash Command
curl -sS -X POST http://127.0.0.1:30010/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-H3",
    "prompt": "Use <Picture 1> as the visual subject and <Audio 1> as the sound reference, with coherent natural motion.",
    "seconds": 5,
    "task": "ref2va",
    "conditions": [
      {
        "type": "image",
        "uri": "file:///data/minimax-h3/reference.png",
        "role": "reference"
      },
      {
        "type": "audio",
        "uri": "file:///data/minimax-h3/reference.mp3",
        "role": "reference"
      }
    ],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "auto",
      "duration_seconds": 5.0
    },
    "num_outputs_per_prompt": 1,
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 3101
  }'
```

</Tab>

</Tabs>

Poll and download any conditioned request with the same job-status and
content endpoints used in the T2VA example. Server-local `file://` URIs must
refer to files visible inside the SGLang server environment.

## 5. LoRA recipes

H3 accepts native fused adapters, standard Diffusers/PEFT adapters, and Kohya
`lora_unet_blocks_*` exports from `networks.lora_minimax_h3`. Native adapters
target modules such as `blocks.*.attn.qkv_proj`; PEFT adapters may instead
provide separate `to_q`, `to_k`, and `to_v` projections and the `default`
adapter namespace. SGLang normalizes all three layouts.

The following pinned FL2VA adapters have distinct purposes:

| Recipe | Repository and pinned file | Request setting | Prompt requirement |
| --- | --- | --- | --- |
| Recommended speed/quality balance | [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora), `minimax_h3_turbo_v4_step600_ema.safetensors` | `num_inference_steps: 8` (8 denoiser evaluations), `lora_scale: 1.0` | None |
| Most aggressive speed preset (standard PEFT layout) | [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo), `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | `num_inference_steps: 4` (4 denoiser evaluations), `lora_scale: 1.0`, `lora_alpha: 8` | None |
| Realistic people style | [`fal/MiniMax-H3-Realism-People-LoRA`](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA), `h3-realism-people-t2v-i2v-r2v.safetensors` | Keep the normal `num_inference_steps: 50` schedule; start with `lora_scale: 0.7` | Include `r34l1sm` in the prompt |
| Spatial and physics | [`Jojocodex/minimax-h3-spatial-physics-lora`](https://huggingface.co/Jojocodex/minimax-h3-spatial-physics-lora), `wushu_spatial_physics_clean_3000_pruned.safetensors` | Keep the normal `num_inference_steps: 50` schedule; start with `lora_scale: 0.4` | None. Describe collisions, stacking, and gravity-consistent motion in the prompt |
| Wushu action | [`Jojocodex/minimax-h3-wushu-action-lora`](https://huggingface.co/Jojocodex/minimax-h3-wushu-action-lora), `wushu_action_h3_lora_v5_3000_pruned.safetensors` | Keep the normal `num_inference_steps: 50` schedule; start with `lora_scale: 0.8` | Put `wushu_action` at the start of the prompt |
| Camera motion | [`Jojocodex/minimax-h3-yunjing-lora`](https://huggingface.co/Jojocodex/minimax-h3-yunjing-lora), `camera_motion_h3_lora_v1_3000_pruned.safetensors` | Keep the normal `num_inference_steps: 50` schedule; start with `lora_scale: 0.8` | Put `camera motion` at the start of the prompt |
| Looping sketch anime | [`Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime`](https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime), `minimax_h3_looping_sketch_anime_v1.safetensors` | Keep the normal `num_inference_steps: 50` schedule; start with `lora_scale: 1.0` | Include the author's hand-drawn sketch style phrase in the prompt |
| 3D-to-real detail slider | [`siraxe/3d_to_real_detail_slider_H3`](https://huggingface.co/siraxe/3d_to_real_detail_slider_H3), `3d_to_real_H3_detail_slider_bf16.safetensors` | Keep the normal `num_inference_steps: 50` schedule; start with `lora_scale: 1.0` | None. Sweep the scale; this is a slider, not a fixed style |

The H3 request field controls the number of denoiser evaluations. An 8-step
adapter uses `8`, and a 4-step adapter uses `4`; the sigma schedule includes
one additional terminal zero.

Every recipe uses the same launch shape. Pinning the filename is required for
repositories that publish multiple revisions, and is also recommended for a
reproducible single-file recipe:

```bash Command
LORA_REPO=larryvrh/MiniMax-H3-Turbo-Lora
LORA_FILE=minimax_h3_turbo_v4_step600_ema.safetensors
LORA_NAME=h3-turbo-v4
LORA_SCALE=1.0
LORA_ALPHA_ARGS=()
# LightX2V only: LORA_ALPHA_ARGS=(--lora-alpha 8)

sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --lora-path "$LORA_REPO" \
  --lora-weight-name "$LORA_FILE" \
  --lora-nickname "$LORA_NAME" \
  --lora-scale "$LORA_SCALE" \
  "${LORA_ALPHA_ARGS[@]}" \
  --lora-merge-mode auto \
  --port 30010
```

`auto` merges adapters into ordinary resident weights to avoid per-step LoRA
matmuls. It keeps FSDP-sharded weights dynamic to avoid full-gather peaks, and
uses dynamic execution for supported packed weights that cannot merge in place
or adapter updates that would mostly round away in a BF16 merge. The same
policy applies when activating multiple adapters through the
[LoRA API](/docs/sglang-diffusion/api/openai_api#lora-management).
Distilled adapters such as the
Larry Turbo LoRA sit below a BF16 ulp of most weights, and a merge drops
77–97% of their update in typical layers. Use `dynamic` when one resident
server must switch repeatedly between base and LoRA output, and `merge` to
force a merge.

Use the filename, scale, and request schedule from the table together. The
4-evaluation LightX2V recipe is the more aggressive latency/quality tradeoff.
Its checkpoint has rank 128 but omits the training alpha from both the file and
repository metadata, so `--lora-alpha 8` is required to reproduce the author's
reference implementation. Start with the Larry 8-evaluation recipe when
preserving fine visual detail is more important than minimum latency.

The pinned files above were trained for the **FL2VA** partition and apply to
`t2va` or `fl2va` requests. Some repositories, including LightX2V, publish
separate files for Ref2VA/Ref2V; select one explicitly for a `ref2va` server
rather than reusing an FL2VA file. Those Ref2VA files are not yet a pinned,
validated recipe on this page. Also avoid stacking a distilled adapter with
`quality: "high"`: both alter denoising, and that combination has not been
quality-validated.

AdaLN-pruned Diffusers components that publish `adaln_basis` and `adaln_mean`
can also consume a LoRA trained against the released full-width AdaLN modules:
SGLang projects those adapter factors onto the pruned coordinates at load time.
A structurally modified checkpoint without that metadata still fails closed,
and packed GGUF weights remain incompatible with LoRA.

### Pika H3 Flex: six-step adapters

[Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) is
MiniMax-H3 plus two FP32 PEFT LoRA adapters (rank/alpha 128): `fl2va` for the
FL2VA partition (`t2va`, `fl2va`) and `ref2va` for the Ref2VA partition. The
adapters were post-trained for a fixed six-step sampler: knots
`1000/750/500/250/100/40`, shifted per modality (video 12, audio 3), with a
re-noising update. Each step predicts x0 and then re-noises it to the next
sigma with fresh noise. Pika notes that generic six-step Euler is not
equivalent, so do not serve these files through the `--lora-path` recipe
above.

Pass the repository directly to `--model-path`. Its registered model overlay
downloads only the two adapters (about 5.5 GB) and checks their sha256 against
Pika's manifests. It then links the base `FL2VA/` and `Ref2VA/` partitions
from `MiniMaxAI/MiniMax-H3` at the revision the adapters pin, so an existing
MiniMax-H3 download is reused. The matching adapter loads at startup, and
`model_index.json` turns on the six-step re-noising schedule. There,
`stochastic_sampling` requires `dmd_denoising_steps`, which carries the
knots. The adapter stays unmerged (`--lora-merge-mode dynamic`) unless a merge
mode is passed: a BF16 merge rounds away about a third of its update.

```bash Command
# t2va / fl2va
sglang serve --model-path Pika-Labs/Pika-H3-Flex --port 30010

# ref2va
sglang serve --model-path Pika-Labs/Pika-H3-Flex --model-variant ref2va --port 30010

# all three tasks from one server, see "One server for both partitions"
sglang serve --model-path Pika-Labs/Pika-H3-Flex --model-variant fl2va+ref2va --port 30010
```

Use the requests in [Generate video and audio](#4-generate-video-and-audio)
with `"model": "Pika-Labs/Pika-H3-Flex"` and `num_inference_steps: 6`. That is
the default, and any other value is rejected. Omit `flow_shift` and
`audio_flow_shift` to keep the trained 12 / 3. `--lora-path` cannot replace
the bundled adapter, and `quality: "high"` is not supported.

On 1× H200 with the default memory policy, 5-second requests took the
following times:

| Task | Canvas | Warm latency |
| --- | --- | --- |
| `t2va` | 1344×768 | 37 s |
| `fl2va` | 1344×768 | 42 s (cold) |
| `ref2va` (2 images + 1 video with audio) | 1344×768 | 260 s |

Most of the `ref2va` time goes to its long packed sequence, which includes
the reference video and two 2048-pixel reference images.

Pika H3 Flex is a Model Derivative of MiniMax H3 by Mellis, Inc., distributed
under the MiniMax H3 Community License Agreement.

### pi-H3: 2- and 4-step adapters, decoder-free video

[Lakonik/pi-H3](https://huggingface.co/Lakonik/pi-H3) distills the AdaLN-pruned
MiniMax-H3 DiT into 2- and 4-step samplers. Its DWD variants predict the frames
directly at the last step with a pixel head, so the video skips the VAE decoder
(audio still uses the audio VAE).

| `--model-variant` | Steps | Video output |
| --- | --- | --- |
| `4step` (default) | 4 | Pixel head |
| `2step` | 2 | Pixel head |
| `4step-vae`, `2step-vae` | 4, 2 | Video VAE |

Pass the repository to `--model-path`. Its model overlay assembles the pruned
DiT, the adapters and the MiniMax-H3 encoders and VAEs (about 51 GB under
`SGLANG_DIFFUSION_CACHE_ROOT`). Do not use `--lora-path` with these files: the
adapters also replace base tensors, add the pixel head and fix the sampling
schedule.

```bash Command
# 1x H200
sglang serve --model-path Lakonik/pi-H3 --port 30010

# 2x H200, fastest measured lossless: Ulysses 2
sglang serve --model-path Lakonik/pi-H3 --model-variant 2step \
  --num-gpus 2 --ulysses-degree 2 --port 30010
```

Requests use [Generate video and audio](#4-generate-video-and-audio) with
`"model": "Lakonik/pi-H3"` and `task` `t2va` or `fl2va`. Omit
`num_inference_steps`, `flow_shift` and `audio_flow_shift`: the variant fixes
them (Euler on the trained knots, video shift 12, audio shift 3). `ref2va` and
`quality: "high"` are not supported, and breakable CUDA graph stays off.

Warm `t2va` requests at 1344×768, 5 seconds, H200:

| Variant | 1 GPU | 2 GPUs, Ulysses 2 | 1 GPU, FP8 + Sol (approximate) |
| --- | --- | --- | --- |
| `2step` | 11.9 s | 6.8 s | 6.7 s |
| `4step` | 22.6 s | 12.3 s | 12.0 s |
| `2step-vae` | 14.7 s | | |

The pixel head decodes the clip in 0.6 s, the video VAE in 4.2 s. The
approximate column adds `--quantization fp8
--component-attention-backends text_encoder=torch_sdpa,transformer=sol_attn
--attention-backend-config '{"dense_steps": 0, "dense_layers": "0,1"}'`. For
80 GB cards, these offload flags keep `2step` on Ulysses 2 at 50 GiB per GPU
(7.2 to 7.9 s per request on H200):

```bash
--performance-mode memory --layerwise-offload-components dit,text_encoder,vae \
--layerwise-resident-layers video_vae=36 --dit-offload-prefetch-size 1 \
--dit-layerwise-resident-layers 20
```

The pi-H3 adapters are Model Derivatives of MiniMax H3, distributed under the
MiniMax H3 Community License Agreement.

### PDD acceleration LoRAs

[Alibaba PAI's MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)
use Parallel Decoding Distillation (PDD) for eight denoiser evaluations. These
checkpoints contain backbone LoRA weights and interval-specific output heads.
Merge the backbone weights offline, then fuse the output heads before serving.
The `--lora-path` recipe above does not perform these preparation steps.

Match the adapter and original MiniMax-H3 transformer weights to the server's
model variant:

| Model variant | Adapter file | Tasks |
| --- | --- | --- |
| `fl2va` | `MiniMax-H3-FL2VA-Acc-8Step.safetensors` | `t2va`, `fl2va` |
| `ref2va` | `MiniMax-H3-Ref2VA-Acc-8Step.safetensors` | `ref2va` |

`BASE_TRANSFORMER_DIR` is the directory containing the original model's
BF16/FP32 transformer weight files, not the model root or the LoRA directory.
If you downloaded `MiniMaxAI/MiniMax-H3` to `/path/to/MiniMax-H3`, use
`/path/to/MiniMax-H3/FL2VA/transformer` for T2VA/FL2VA or
`/path/to/MiniMax-H3/Ref2VA/transformer` for Ref2VA. This directory contains
`model-*.safetensors`, `model.safetensors.index.json`, and `config.json`.
The builder reads these files and writes the merged weights to
`$PDD_DIR/transformer`, leaving the original weights unchanged.

The following example prepares FL2VA
weights and serves them on eight NVIDIA CUDA GPUs with TP2 + Ulysses4:

```bash Command
BASE_TRANSFORMER_DIR=/path/to/MiniMax-H3/FL2VA/transformer
LORA_DIR=/path/to/h3-acc-loras
PDD_DIR=/path/to/h3-fl2va-pdd

hf download alibaba-pai/MiniMax-H3-Acc-LoRAs \
  MiniMax-H3-FL2VA-Acc-8Step.safetensors --local-dir "$LORA_DIR"

python3 -m sglang.multimodal_gen.tools.build_minimax_h3_pdd_weights \
  "$BASE_TRANSFORMER_DIR" \
  "$LORA_DIR/MiniMax-H3-FL2VA-Acc-8Step.safetensors" \
  "$PDD_DIR"

python3 -m sglang.multimodal_gen.tools.fuse_minimax_h3_pdd_heads \
  "$PDD_DIR" --video-shift 12.0 --audio-shift 3.0

SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS="$PDD_DIR/pdd_fused_heads.safetensors" \
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --transformer-weights-path "$PDD_DIR/transformer" \
  --num-gpus 8 \
  --tp-size 2 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --port 30010
```

Keep `pdd_config.json` beside `pdd_fused_heads.safetensors`, outside the
`transformer/` directory. For Ref2VA, use its adapter and base transformer,
a separate output directory, and `--model-variant ref2va`.

Use the requests in [Generate video and audio](#4-generate-video-and-audio)
with `num_inference_steps: 8`, `flow_shift: 12.0`, and `audio_flow_shift: 3.0`.
The nine sigma grid points give eight denoiser evaluations. These settings
must match the fused heads; the server rejects a different step count or
shift. Use `quality: "lossless"` in requests.

## 6. FastH3: 8-step distilled

[FastVideo/FastVideo-FastH3-8-Step-V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2)
is an 8-step DMD2 distillation of MiniMax-H3, trained data-free with Video
Sparse Attention (VSA) at 0.8 sparsity and 64-token tiles, with video / audio
scheduler shifts 10 / 3. Only the T2VA capability was distilled: requests must
use `task: "t2va"`, and `fl2va` / `ref2va` requests are rejected. The
checkpoint inherits the MiniMax-H3 Community License.

Pass the repository directly to `--model-path`. The flat native-Diffusers
upload is materialized into the base-H3 layout through a registered model
overlay; the only non-symlink step is a one-time re-serialization of the
roughly 10 GB video VAE on first launch.

```bash 4×GB300 VSA-H3
sglang serve \
  --model-path FastVideo/FastVideo-FastH3-8-Step-V2 \
  --num-gpus 4 \
  --component-attention-backends transformer=video_sparse_attn_h3 \
  --port 30010
```

Requests use the same asynchronous video endpoint as the base model, with
`task: "t2va"`, `conditions: []`, and a target such as
`{"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0}`. The
request default is `num_inference_steps: 8`: nine sigma grid points, i.e. the
eight distilled DiT evaluations on the checkpoint's trained rungs
`[999, 874, 749, 624, 500, 375, 250, 125]` (unshifted timesteps over 1000,
shifted once per modality, as FastVideo runs them). Any other step count is
rejected. The trained sparsity 0.8 is the default; `--attention-backend-config
'{"VSA_sparsity": ...}'` overrides it. FastH3 requests default to
`quality: "lossless"`, which fuses the VAE decoder's per-head QK RMSNorm and
RoPE and runs its attention on cuDNN SDPA; `quality: "exact"` keeps the
reference decoder kernels. Under `"lossless"` FastH3 decodes each GPU's
spatial VAE tiles in one call rather than one at a time (base MiniMax-H3
batches up to 64 per call), which changes GEMM shapes and so the output at
rounding level; `"exact"` decodes tile by tile on both.

`video_sparse_attn_h3` (VSA-H3) is the trained sparse policy: in-tree
block-sparse kernels (FastVideo's native tcgen05 forward on SM100 / SM103, a
Triton tile-64 kernel on SM90) over segment-pure prefix tiles and (4, 4, 4)
video tiles, driven by the checkpoint's trained
`to_gate_compress` compression branch. Select it for the transformer only, as
with cube sparse attention: the packed DiT blocks run sparse while the token
refiner falls back to dense FA inside the backend, and the text encoder and
VAEs keep their own backends. Ulysses sequence parallelism is supported. See
[Attention Backends](/docs/sglang-diffusion/attention_backends) for
`VSA_sparsity`, `vsa_mode`, `vsa_dense_first_n_steps`, and
`vsa_dense_layers`. On CUDA, FastH3 selects VSA-H3 for the transformer when no
transformer backend is given, unless torch.compile, the breakable CUDA graph or
ring attention is on (VSA-H3 serves none of them). Every dense backend that
runs on base H3 also runs on the FastH3 weights, without the compression
branch: select it with `--component-attention-backends transformer=fa` (or
`torch_sdpa`, ...). On 4× H200 dense FA is about 25% slower end to end than
VSA-H3 (12.6 s against 10.0 s). `sglang generate` takes the same flags as
`sglang serve`.

Measured latencies for the 4× GB300 recipe are in
[FastH3 on GB300](#fasth3-on-gb300).

FastH3 rejects deployment options that do not apply to the distilled
checkpoint instead of silently ignoring them: `--model-variant`,
`quality: "high"`, `fl2va` / `ref2va` requests, and, with VSA-H3,
`--ring-degree` greater than 1, `torch.compile`, and breakable CUDA graph
execution.

<Note>
Quality gaps versus base H3 on hard motion and fine detail are properties of
the released distillation, not of the SGLang port. Use base MiniMax-H3 when
output quality matters more than latency.
</Note>

### FastH3 Omni: Ref2VA, 8-step PDD

[FastVideo/FastVideo-FastH3-Omni-8-Step-V1](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1)
is a Parallel Decoding Distillation (PDD) student of the MiniMax-H3 Ref2VA
partition: eight denoiser evaluations over a 32-interval grid, with VSA at 0.9
sparsity on 128-token (4, 4, 8) tiles and each reference video keeping 0.1 of
its tiles. Only `task: "ref2va"` is served, with up to 9 images, 3 videos and 3
audio clips as ordered references (12 in total), at least one of them visual.

Pass the repository directly to `--model-path`. Its registered model overlay
maps `transformer_ref/` into the base-H3 layout, fuses the 32 interval output
heads into 8 per-step heads (the math of `tools/fuse_minimax_h3_pdd_heads.py`)
and links the text encoder and VAEs from the `MiniMaxAI/MiniMax-H3` Ref2VA
partition pinned by the release. The served model is the native H3 pipeline:
the release's `pdd_fused_heads` load through the same path as the PDD LoRA
recipe above, and the PDD grid is the 8-step shifted grid scaled to the
release's `sigma_max` of 0.999 (FastGen's `max_t`).

```bash 2×H200
sglang serve \
  --model-path FastVideo/FastVideo-FastH3-Omni-8-Step-V1 \
  --num-gpus 2 --ulysses-degree 2 \
  --port 30010
```

Everything stays resident on 141 GB and 180 GB cards. On a card that cannot
hold it, add `--dit-cpu-offload true --text-encoder-cpu-offload true
--vae-cpu-offload true`; on H200 and B200 the three offloads only cost time
(image reference, 768 short edge, 124 frames, median of two requests after
three warmups):

| GPUs | Resident | With the three offloads |
| --- | --- | --- |
| 2× H200, Ulysses 2 | 39.0 s | 43.8 s |
| 4× B200, Ulysses 4 | 10.3 s | 15.3 s |

```json Request
{
  "model": "FastVideo/FastVideo-FastH3-Omni-8-Step-V1",
  "prompt": "<Picture 1> is the appearance reference for <Subject 1>. [reference generation] ...",
  "task": "ref2va",
  "conditions": [{"type": "image", "role": "reference", "uri": "/path/to/subject.png"}],
  "target": {"short_edge": 480, "aspect_ratio": "16:9", "duration_seconds": 5.0},
  "seed": 0
}
```

Requests default to `num_inference_steps: 8` and clean references
(`imgvid_cond_noise_aug_for_inference` and `audio_cond_noise_aug_for_inference`
of 1.0, the release's target-only flow). On CUDA the transformer selects VSA-H3
when no backend is given; Ref2VA layouts tile every reference video as its own
region and take a per-region top-k, while text, reference images and audio
rows stay dense. `--attention-backend-config` keys `vsa_tile_size`,
`vsa_ref_keep_rate` and `VSA_sparsity` override the trained values.

Warm request latency on 2× H200 with the three CPU offloads (Ulysses 2; median
of three requests after one full warmup, 124 frames at 24 FPS with audio,
until the MP4 is saved):

| References | Short edge | Latency |
| --- | --- | --- |
| Image | 480 | 26.9 s |
| Image + audio | 480 | 27.8 s |
| Video with audio | 480 | 23.2 s |
| Image | 768 | 46.6 s |
| Video with audio | 768 | 37.1 s |

## 7. VDN-H3: hybrid attention, 8-step distill

[Video DeltaNet](https://openvdn.github.io/) (VDN) is a hybrid attention
architecture that approaches the speed of linear attention while maintaining
near-lossless quality. It splits the video-video attention into two
complementary branches: a sliding-window softmax branch for local frame pairs
(five consecutive latent frames form a chunk, and each chunk attends to
itself, the previous chunk, and the following chunk; the first and last
frames, text and audio stay dense) and a bidirectional linear branch, Video
Delta Attention, for long-range context.
[OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3)
applies it to MiniMax-H3: the released `stage-dmd-step-250` checkpoint adds a
4.3 GB linear branch and OpenVDN's own 8-step DMD2 LoRA on the unchanged H3
backbone. It is the FL2VA partition, so it serves `t2va` and `fl2va` (first,
last, or both keyframes); `ref2va` is rejected. The weights inherit the
MiniMax-H3 Community License.

Pass the repository directly to `--model-path`. On first launch a registered
overlay prefuses both adapters into the transformer weights (a 62 GB write)
and attaches the linear branch; the Qwen3-VL conditioner and the VAEs are
hard-linked from `MiniMaxAI/MiniMax-H3`. Keep `SGLANG_DIFFUSION_CACHE_ROOT` on
the same filesystem as the Hugging Face cache with at least 90 GB free.

```bash 8×B200
sglang serve \
  --model-path OpenVDN/vdn-minimax-h3 \
  --num-gpus 8 \
  --attention-backend hybrid_window_attn_h3 \
  --performance-mode speed \
  --warmup-num-frames 345 \
  --warmup-resolutions 1344x768 \
  --port 30010
```

`--num-gpus 4`, `2` or `1` gives the smaller Ulysses recipes in the benchmark
table; `sglang generate` takes the same flags. `--warmup-num-frames` /
`--warmup-resolutions` warm up at the clip length and canvas you will serve;
without them the first long request pays 2 to 3 s of allocator growth and
kernel setup.

Requests use the same video endpoint as the base model with `task: "t2va"`
and `conditions: []`, or `task: "fl2va"` with the
[keyframe conditions](#4-generate-video-and-audio), and a target such as
`{"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.375}` (the
345-frame paper workload). The default `num_inference_steps: 8` is the only
accepted value: nine sigma grid points, eight DiT forwards. Each keyframe adds
about 2,000 dense rows, so an `fl2va` forward costs a little more than `t2va`.

`hybrid_window_attn_h3` is required: a dense backend would silently skip the
linear branch and the gates and produce the wrong model. On Blackwell
(SM100 / SM103 / SM120) the transformer defaults to online MXFP8;
`--quantization bf16` opts out. Before SM100, `--quantization fp8` is base
H3's per-channel fp8 path, and SM80 runs the bf16 DiT (62 GB; budget for
`--layerwise-offload-components`). Ampere and Ada are enabled but not
benchmarked. Ulysses sequence parallelism is supported; `--model-variant`,
`quality: "high"`, `--ring-degree` greater than 1, `torch.compile`, and
breakable CUDA graph execution are rejected. The backend's
`--attention-backend-config` keys are listed in
[Attention Backends](/docs/sglang-diffusion/attention_backends).

Measured latencies for 1, 2, 4 and 8 B200s are in
[VDN-H3 on B200](#vdn-h3-on-b200); single-card and PCIe multi-card numbers
for the RTX PRO 6000 are in [VDN-H3 on RTX PRO 6000](#vdn-h3-on-rtx-pro-6000).

## 8. Sampling and output controls

MiniMax-H3 supports more than one output per prompt. The video API accepts
`num_outputs_per_prompt` (or OpenAI-compatible `n`) from 1 through 10. Offline
generation accepts `--num-outputs-per-prompt N`; `--num-outputs N` is the short
alias. A scalar seed is expanded deterministically as `seed + output_index`, so
the outputs do not reuse the same noise.

Same-prompt fan-out reuses text conditioning. On the verified 2× RTX 5090
recipe, a 5-step two-output request completed in 155.39 seconds versus 78.11
seconds for one output, while producing two distinct valid MP4 files. The
independent denoise and decode passes remain sequential on this 32 GB profile
to keep peak memory bounded; the grouped path adds essentially no orchestration
overhead. Use server replicas when lower wall-clock latency for many variants
matters more than per-server memory efficiency.

For example, set `"num_outputs_per_prompt": 2` in any request above. After the
job completes, download both outputs by selecting each zero-based variant:

```bash Command
video_id="<completed-job-id>"
for variant in 0 1; do
  curl -sS -L \
    "http://127.0.0.1:30010/v1/videos/${video_id}/content?variant=${variant}" \
    -o "minimax-h3-${variant}.mp4"
done
```

### Choose the quality level

`quality` is a cumulative request-scoped optimization parameter with three
levels:

- `"exact"`: the exact reference path. Output is bit-exact
  against the reference implementation and the CI ground truth. On SM100+
  the video VAE's temporal blend and output denormalization run as single
  kernels that reproduce the reference arithmetic bit for bit and verify that
  against the eager chain on first use.
- `"lossless"` (default): includes the global fusion-only tier but does not enable
  Cache-DiT or another approximate optimization. Each DiT block runs its
  RMSNorm and AdaLN modulation as one kernel that rounds once instead of after
  every op, and VAE decode fuses the ViT decoder's per-head QK RMSNorm and
  RoPE into one launch, runs its attention on cuDNN SDPA and, on CUDA, decodes
  several spatial tiles per decoder forward (rounding-level differences only).
- `"high"`: the audited accelerated path. Cache-DiT reuses the DiT stack's
  output on a fixed step schedule: the first 15 of the 50 steps always run,
  then every second step up to step 25, then every eighth. Quality is
  guaranteed (the audited schedule measures SSIM 0.911 / PSNR 29.09 dB against
  `lossless`), but output is no longer bit-identical to the reference.

One resident server serves all three levels; a `quality: "high"` request
mounts its audited Cache-DiT policy at the batch boundary, and a later
`quality: "exact"` or `quality: "lossless"` request removes the hooks
before denoising.

Start the validated server once:

```bash Command
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --tp-size 1 \
  --sp-degree 4 \
  --ulysses-degree 4 \
  --ring-degree 1 \
  --encoder-parallel auto \
  --performance-mode speed \
  --use-fsdp-inference false \
  --enable-torch-compile false \
  --port 30010
```

Then choose a request level:

<Tabs>

<Tab title="exact">

Native denoising and VAE decode with no feature-cache approximation or
quality-gated VAE fusions.

```json Request field
{
  "quality": "exact"
}
```

</Tab>

<Tab title="lossless (default)">

The global fusion-only tier. It does not enable MiniMax-H3 Cache-DiT. It
differs from `exact` only in rounding: the DiT blocks' fused RMSNorm + AdaLN
modulation, and the VAE decoder's fused QK RMSNorm + RoPE and cuDNN SDPA
attention.
Omitting the field selects this level.

```json Request field
{
  "quality": "lossless"
}
```

</Tab>

<Tab title="high">

The audited accelerated path. Use it when you can trade bit-exactness for
latency while keeping output closest to the same-seed exact trajectory.

```json Request field
{
  "quality": "high"
}
```

</Tab>

</Tabs>

The measured trade-off is:

| `quality` | Mean <br />inference <br />latency | Speedup | SSIM vs <br />lossless | PSNR vs <br />lossless | Expected <br />trade-off |
| --- | ---: | ---: | ---: | ---: | --- |
| `exact` | 71.84 s | 0.99× | 0.660 | 17.85 dB | Native reference path; a different sample from `lossless` |
| `lossless` | 71.17 s | 1.00× | 1.000 | identical | Default; fusion-only tier (rounding-level differences in each operation) |
| `high` | 35.80 s | 1.99× | 0.911 | 29.09 dB | Stays close to the same-seed `lossless` sample |

These numbers use 1344×768, 124-frame, 24 fps T2VA with 50 inference steps,
video flow shift 12, audio flow shift 3, and three fixed prompt/seed pairs on
4×H200. The prompts cover a quiet detailed scene, fast multi-subject action,
and a moving close-up portrait. `inference_time_s` is averaged across the three
prompts.

SSIM and PSNR compare decoded, frame-aligned output with the `lossless`
result for the same prompt and seed. They measure trajectory deviation, not
absolute perceptual quality: the `high` path can produce a different but
still plausible realization. It also changes the joint audio-video denoise
trajectory, while these two metrics cover video only. `exact` and `lossless`
differ only by rounding inside each operation, but 50 denoising steps grow that
into a different sample, so `exact` is not a usable reference for `high`.

`quality: "high"` currently accepts only the exact workload and 4×H200
deployment above; other hardware, task modes, request shapes, step counts, or
flow shifts fail before denoising. Offline generation uses the same level
name, for example `sglang generate --quality high`.

<Note>
`quality` selects a model sampling level and can change generated content.
`output_quality` controls only output-file compression; it is a separate field.
</Note>

For manually tuned Cache-DiT experiments outside that validated path, omit
the request `quality` field and set `--enable-cache-dit` or the
process-wide `SGLANG_CACHE_DIT_*` defaults. Any explicit `quality`, including
`"exact"` and `"lossless"`, takes H3 off the generic Cache-DiT path. The
24 GB layerwise recipe above can use the same switch; skipped blocks are
not streamed.

```bash Command
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=1 \
SGLANG_CACHE_DIT_BN=0 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.12 \
SGLANG_CACHE_DIT_MC=2 \
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 8 \
  --ulysses-degree 8 \
  --encoder-parallel auto \
  --performance-mode speed \
  --port 30010
```

<Warning>
Cache-DiT skips selected block computation and is approximate. It cannot
be combined with FSDP inference. DiT layerwise offload is compatible:
skipped blocks are not streamed, and the first layer after a skip may
sync-load. Breakable CUDA graph execution takes precedence and leaves
Cache-DiT disabled. Tune the cache thresholds only after comparing both
video and audio quality on the target task profile. A real B200 request
has completed, but the `quality: "high"` path above remains fail-closed
to the audited 4×H200 workload.
</Warning>

## 9. Feature contracts and advanced recipes

The generated command already contains the recommended topology and encoder
setting. Use the detailed reference below only when applying an optional
override or checking its installation, topology limits, and validation evidence.

<Tabs>

<Tab title="Lossless runtime">

The recommended `speed` launch already combines resident components with
Ulysses sequence parallelism. Validation status below applies only to the
listed hardware and topology; it is not inherited by a similar GPU family.

| Feature | Validation status | Notes |
| --- | --- | --- |
| Ulysses sequence parallelism | Verified: 8× B200, 4× H200, 4× H100, and Ulysses1/2/4/8 on MI300X and MI355X | Use `--ulysses-degree`. Combine with Ring for cross-node scaling; see the next row. |
| Ring sequence parallelism (cross-node) | Verified: 2 nodes of 8× H200 each (Ulysses8 × Ring2) | Use `--ring-degree` together with `--nnodes`/`--node-rank`/`--dist-init-addr`. Ring shards the sequence across nodes while Ulysses shards heads within a node; H3's packed multi-segment attention only supports Ring across the node boundary, not within a single node's Ulysses group. Requires `--encoder-parallel replicate` — `auto`'s fold decision is not node-boundary aware. See the benchmark section below. |
| Tensor parallelism | Verified: B200 TP2 + Ulysses4; H100 TP2 + Ulysses2 and TP4 + Ulysses1 | `--tp-size` may be combined with Ulysses when the TP-local head count remains divisible by the Ulysses degree. On 4×H100, TP2 + Ulysses2 is the measured speed default. |
| FSDP inference | Verified: 4× B200 and 4× H100 + Ulysses4 | Preserves H3's mixed BF16/FP32 parameter policy. B200 completed the exact eager comparison; H100 completed consecutive real requests at about 57 GB peak memory per GPU. |
| Resident components | Verified: B200, H200, 4×H100 with TP, and 1/2/4/8× MI300X and MI355X | This is the recommended single-request latency path when the complete workload fits. |
| CPU and layerwise offload | Verified: 2× RTX 5090 TP2; 1× RTX 4090 24 GB | The 5090 lossless recipe keeps 20 DiT blocks plus both VAE encoders resident, streams the remaining DiT blocks, text encoder, and video VAE decoder blocks, and leaves the small audio VAE resident. The 4090 recipe streams DiT and the text encoder with zero resident DiT layers and **omits `vae`** from `--layerwise-offload-components`. Compatible with Cache-DiT; skipped blocks are not streamed. |
| Breakable CUDA graph | Verified: B200 Ref2VA, opt-in | Matching eager output was observed for the captured signature, without a measured speedup. Re-capture for other shapes and reference sets. |
| `torch.compile` | Measured: H200, opt-in | Steady-state benefit was below measurement noise, while startup increased and numerical output changed. Do not use it for consistency ground truth. |

The verified parallel, placement, and matching-signature BCG paths keep the
BF16/FP32 weights and denoising math. `torch.compile` is the exception called
out above. Always use the eager BF16/FP32 launch when producing CI consistency
ground truth.

For the validated 1344×768 Ref2VA profile, use a 5504-row text bucket so both
the server warmup and reference-conditioned requests share the captured
signature:

```bash Command
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 8 \
  --ulysses-degree 8 \
  --encoder-parallel auto \
  --performance-mode speed \
  --enable-breakable-cuda-graph true \
  --warmup-resolutions 1344x768 \
  --bcg-text-buckets 5504 \
  --port 30010
```

BCG is lossless for a matching captured signature, but capture reserves extra
GPU memory. Re-measure the live H3 text length before reusing this bucket for a
different task profile, reference set, resolution, or prompt template.

</Tab>

<Tab title="Attention backends">

Leave `--attention-backend` unset for the platform default. Use
`--attention-backend fa` only for an explicit FlashAttention comparison.

SageAttention uses quantized attention math and is not a consistency mode. To
select H3's native packed-varlen Sage path, install the dependency and add
`--attention-backend sage_attn`. On Hopper, install the upstream SM90 binding
fix rather than the PyPI 2.2.0 build:

```bash Command
pip install --force-reinstall \
  git+https://github.com/thu-ml/SageAttention.git@d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4 \
  --no-build-isolation
```

The backend is a server-wide default. Use
`--component-attention-backends` only when a measured component needs a
different kernel, and keep the platform default for every component not named
in the override.

</Tab>

<Tab title="Online quantization">

On the verified 8× B200 topology, quantize the BF16 transformer at server load:

```bash Command
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 8 \
  --ulysses-degree 8 \
  --encoder-parallel auto \
  --performance-mode speed \
  --quantization fp8 \
  --port 30010
```

H3 automatically keeps its video/audio patch projections, timestep MLP, and
final video/audio heads in FP32. All other linear layers have stable full
module prefixes, so additional layers can be kept unquantized:

```bash Command
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 8 \
  --ulysses-degree 8 \
  --encoder-parallel auto \
  --quantization fp8 \
  --quantization-ignored-layers blocks.0.attn token_refiner \
  --port 30010
```

<Warning>
Online FP8 is approximate and is not a consistency ground-truth mode. It can
be combined with Cache-DiT, but the two approximations compound. Validate
visual quality, audio quality, memory use, and latency on the target workload.
This recipe is limited to the resident B200 and B300 topologies used for real
H3 validation runs.
</Warning>

On a single 24 GB card, use `convrot_int8` instead of FP8. It quantizes the
four GEMMs per DiT block online from the Hub BF16 weights (data-free, no
calibration) and runs them on `comfy_kitchen.int8_linear` on the 4090 (CC 8.9)
or on SGLang's JIT-compiled ConvRot ops on CC 9.0, 10.0, 12.0 and 12.1
(see [Quantization](/docs/sglang-diffusion/quantization#convrot-int8)).
Quantization happens after H3's grouped `qkv` reorder, so do not load an
externally pre-quantized INT8 checkpoint here.

```bash Command
sglang generate \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --quantization convrot_int8 \
  --attention-backend fa \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder \
  --dit-offload-prefetch-size 1 \
  --dit-layerwise-resident-layers 0 \
  --enable-torch-compile false \
  --prompt "A cat walking on a sunny beach, gentle waves." \
  --save-output
```

`fa` keeps exact attention. For a faster, approximate DiT path, use
`--attention-backend sol_attn` with
`--attention-backend-config dense_backend=sage_attn,dense_steps=10` and
`--component-attention-backends text_encoder=torch_sdpa,transformer=sol_attn`.
See [Quantization](/docs/sglang-diffusion/quantization#convrot-int8)
and [Attention Backends](/docs/sglang-diffusion/attention_backends#sage-then-sol-hybrid).

<Warning>
`convrot_int8` changes Linear numerics. `sol_attn` / `sage_attn` also change
the attention algorithm. Neither is a consistency ground-truth mode. The
BF16 path never depends on `comfy-kitchen`; `convrot_int8` needs it only on
GPUs the JIT ops do not cover (any CC other than 9.0, 10.0, 12.0 and 12.1) or
when no CUDA toolchain is available to build them.
</Warning>

Pre-quantized and compact H3 text encoders are listed once in
[Checkpoint and adapter formats](#checkpoint-and-adapter-formats). They use
component-local paths and never inherit the DiT's `--quantization` setting.
SGLang reads their metadata before constructing the native Qwen3-VL encoder and
fails closed when the selected format, projection, or topology is incompatible.

</Tab>

<Tab title="Cube sparse attention">

Cube sparse attention applies TopK sparsity only to H3's 3D visual streams.
Text, audio, standalone reference images, and the text-only token refiner stay
dense. It runs on pure PyTorch plus FlexAttention, so it has no third-party
kernel dependency.

Select it for the H3 transformer with
`--component-attention-backends transformer=cube_sparse_attn` and pass
`--attention-backend-config` with both `local_cube_size` and
`topk_ratio_list`. Scoping the backend leaves the text encoder on its native
backend:

```bash Command
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 8 \
  --ulysses-degree 8 \
  --performance-mode speed \
  --component-attention-backends transformer=cube_sparse_attn \
  --attention-backend-config '{"local_cube_size": [4, 4, 4], "topk_ratio_list": [1.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5]}' \
  --port 30010
```

- `local_cube_size` is the `(T, H, W)` cube that groups neighboring latent
  tokens into one attention block. It must have exactly three entries.
- `topk_ratio_list` sets the per-step keep ratio and must have exactly one
  entry per denoise step, each in `(0, 1]`. A ratio of `1.0` keeps a step
  on H3's native dense attention path; smaller values select the FlexAttention
  sparse path and drop more blocks. The example above matches a request with
  `num_inference_steps: 7`, whose endpoint-inclusive sigma schedule has seven
  denoise updates.

Cube labeling is coordinate-driven. FL2VA keyframes share the target video's
position grid, so a keyframe token and a target token at the same `(T, H, W)`
coordinate receive the same semantic cube label. Duplicate coordinates do not
extend the temporal grid; a semantic cube can therefore span multiple physical
attention blocks. In Ref2VA, standalone reference images remain dense, while
reference videos and the target video contribute to one global TopK candidate
pool rather than receiving separate per-stream quotas.

<Warning>
Cube sparse attention is an approximate backend and is not a consistency
ground-truth mode. `topk_ratio_list` length must equal the denoise step count
or the server rejects the request. Cube sparse attention does not support Ring
parallelism; use `--ulysses-degree` without `--ring-degree`. FlexAttention's
routing overhead can outweigh sparse-kernel savings on short sequences, so
benchmark latency as well as visual and audio quality on the target workload.
</Warning>

</Tab>

<Tab title="SubBlock sparse attention">

SubBlock sparse attention applies training-free block sparsity to MiniMax-H3's
long, non-causal DiT self-attention. The BF16 path supports CUDA SM90, SM100,
and SM120. Other compute capabilities, including SM103/B300, are rejected.
Ulysses sequence parallelism is supported; Ring parallelism is not.

The global backend selects the H3 DiT lazily, so keep the Qwen text encoder on
its compatible dense backend with a component override:

```bash Command
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 8 \
  --ulysses-degree 8 \
  --attention-backend subblock_sparse_attn \
  --component-attention-backends text_encoder=fa \
  --attention-backend-config '{"sparsity": 0.75, "skip_first_steps": 10}' \
  --port 30010
```

Use `text_encoder=torch_sdpa` on SM120. The default configuration drops at
most 75% of key blocks after the first 10 denoise forwards; short sequences,
cross-attention, and unsupported shapes continue on the dense path. Tune
`sparsity` and `skip_first_steps` together with visual and audio quality checks.

On H100/H200 (SM90), `compute_mode=sage_fp8` switches the sparse kernel to
online INT8 Q/K and FP8 P/V compute. Install the optional kernel and add the
compute mode:

```bash Command
pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation

sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 8 \
  --ulysses-degree 8 \
  --attention-backend subblock_sparse_attn \
  --component-attention-backends text_encoder=fa \
  --attention-backend-config '{"compute_mode": "sage_fp8", "sparsity": 0.75}' \
  --port 30010
```

<Warning>
Both SubBlock routing and `sage_fp8` are approximate. Use dense BF16 attention
for consistency ground truth, and validate the selected sparsity on the target
resolution, duration, task, and checkpoint.
</Warning>

</Tab>

<Tab title="Encoder scheduling">

The picker explicitly writes `--encoder-parallel auto` in every single-node
recipe. At the default request batch size of one, H100/H200/B200/B300 servers
with peer-to-peer access fold the Qwen encoder across otherwise idle Ulysses
ranks. A pure-TP recipe keeps the encoder inside its TP group, while a
PCIe-only host can avoid an expensive world fold. Keep `auto` unless one of
the cases below applies.

Encoder DP is a throughput policy for compatible request batches. It requires
TP1 and DiT DP1, replicates the encoder weights, and does not improve a batch
of one:

```bash Overlay
--encoder-parallel dp \
--batching-max-size 2
```

The cross-node picker recipe already uses replication because the automatic
fold decision is not node-boundary aware:

```bash Overlay
--encoder-parallel replicate
```

</Tab>

</Tabs>

## 10. Configuration notes

- MiniMax-H3 produces the canonical 24 fps output; request duration is expressed through `target.duration_seconds`.
- `target.duration_seconds` must be between 4 and 15 seconds, inclusive. The command picker defaults to the verified 5-second profile.
- Use a 768-pixel short edge for the released quality recipe. The aligned output dimensions are derived from `target.aspect_ratio`.
- `flow_shift` controls video diffusion and `audio_flow_shift` controls audio diffusion.
- V2V uses `task: "ref2va"` with a `video` or `video_audio` reference; it is served by the `Ref2VA` partition and is not a separate public task value.
- `conditions[].start_time_seconds` selects a non-negative offset for a video reference. Its visual and audio streams are always sought together.
- Ref2VA condition order is semantic and must match the one-based material tags in the prompt. For Ref2VA, `target.aspect_ratio: "auto"` resolves to the model's 16:9 fallback rather than inheriting a reference asset's geometry.
- The distilled pipeline uses a single denoising branch, so CFG parallelism does not apply. Keep it disabled: a CFG degree greater than 1 is accepted for request compatibility, but the extra ranks only recompute the same branch and do not accelerate the request. Explicitly disabling CFG, or setting its size to 1, remains a valid no-op.
- The released visual VAE quality recipe uses overlapping tiled decode. SGLang keeps that recipe by default and distributes complete tiles across the decode group; this changes scheduling, not the computation inside each tile.
- H3 rejects `--vae-config.parallel-decode-mode spatial` and `spatial_shard`: validation found output mismatches. Use the default released tiled recipe.
- On CUDA, `quality: "lossless"` (the default) decodes up to `vae_config.decoder_tile_batch` (64) spatial tiles and `vae_config.window_batch` (1) temporal windows per video VAE decoder forward, next to the fused Q/K norm+RoPE and cuDNN SDPA; `exact` decodes one tile and one window at a time, `--performance-mode memory` does so at every level, and other platforms keep one tile per forward. On SM100+ the temporal blend and output denormalization run as bit-exact kernels at every quality level.
- Keep the default `--encoder-parallel auto`. With the server’s default `batching_max_size` of 1, single-node H100/H200/B200/B300 recipes with peer-to-peer access fold the Qwen text encoder over otherwise idle Ulysses ranks. This is separate from DiT tensor parallelism. A pure-TP recipe already shards the encoder over its TP group and does not add a world fold.
- Single-node Ulysses recipes pipeline each block's all-to-all against dense attention (FlashAttention on Hopper, cuDNN SDPA on Blackwell) over the copy engine: one head group's q/k/v moves while the previous group attends, every copy (the merge into the output included) runs on the copy engine, and the output is bit-identical to the sequential exchange. It needs peer-to-peer access and stream memory operations on every rank and about 6 × (rows per rank) × heads × head_dim extra activations per rank (2.1 GB for Ref2VA at Ulysses 2), one set shared by every resolution; when that does not fit on some rank, every rank keeps the sequential exchange. `SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS=0` restores the sequential exchange.
- For throughput-oriented serving, select **DP (batched throughput)**. The picker pairs `--encoder-parallel dp` with an editable `--batching-max-size` greater than 1. Encoder DP stays inside each DiT replica and composes with encoder TP: the H100 TP2 + Ulysses2 recipe has two TP-sharded encoder copies that can split a batch, while the RTX 5090 pure-TP2 recipe has one encoder copy and therefore no additional batch-DP degree. It provides no benefit for a batch of one and is not bitwise-identical to the unsplit deployment.
- Use explicit **Fold** to prioritize single-request latency and encoder memory on a measured high-bandwidth single-node topology. Use **Replicate** as the compatibility path when folding or encoder DP is unsuitable.
- `--use-fsdp-inference true` shards only the DiT. MiniMax-H3 preserves the original FP32 dtype of its patch, time, and output projections during FSDP all-gather, so this path does not trade numerical correctness for memory. On 4×H100, prefer TP2 + Ulysses2 for speed; use FSDP as an explicit capacity policy rather than assuming it is faster.
- `speed` keeps model components resident, while `auto` applies the model-aware 120 GiB residency threshold. `memory` prioritizes avoiding OOM and includes the executable VAE decoder in its default layerwise set. A measured recipe with sufficient headroom can opt into `--component-residency vae=resident`; the 2×H100 CI recipe does this because the VAE's 4.8 GiB/GPU cost avoids repeated decoder transfers during tiled decode. DiT residency and prefetch knobs remain scoped to the DiT. Use `speed` only after confirming that the complete target workload fits.
- Breakable CUDA graph execution is an explicit opt-in, not part of the recommended `speed` preset. It requires `--enable-breakable-cuda-graph`, every served size in `--warmup-resolutions`, and `--bcg-text-buckets` that cover the live H3 condition sequence. The validated 1344×768 Ref2VA recipe uses 5504; other task profiles and reference sets may need a different value. It preserves eager output for matching captured signatures, but graph capture consumes additional GPU memory and may provide little latency benefit when Ulysses attention and collectives dominate, so benchmark it on the target topology before enabling it.
- `--enable-spectrum` skips selected DiT steps. On H3 it cannot combine with Cache-DiT / `quality="high"` or SP &gt; 1. 50-step 4090 recipe is in §3.

## 11. Benchmarks

The picker exposes resident and FSDP profiles on NVIDIA datacenter GPUs. GPU
counts are properties of the selected recipes, not a claim that every platform
requires that many GPUs. The detailed tables below report performance only for
the configurations with collected measurements:

| Hardware | Default resident recipe | Other profile or topology |
| --- | --- | --- |
| B300 | 8× Ulysses8 resident | 8× FSDP + Ulysses8; the 8-GPU sweep is not a minimum-GPU claim. |
| B200 | 8× Ulysses8 resident | 4× FSDP + Ulysses4 |
| GB300 | 4× Ulysses4 resident (FL2VA T2VA) | 2 nodes × 4× Ulysses4×Ring2 is unverified. |
| GB200 | 4× Ulysses4 resident, derived/unverified | No performance measurements. |
| H200 | 4× Ulysses4 resident | 4× FSDP + Ulysses4; 4× TP2 + Ulysses2; 2 nodes × 8× Ulysses8×Ring2 cross-node |
| H100 | 4× TP2 + Ulysses2 resident | 4× TP4 + Ulysses1; 4× FSDP + Ulysses4 |
| Ascend NPU | 8 NPUs, TP2 + SP4, Laser Attention | 4 NPUs, TP2 + SP2, Laser Attention |
| MI300X / MI355X | 8× Ulysses8 resident | 1×, 2×, and 4× scaling runs |
| RTX 5090 | 1× layerwise offload, measured on a physical desktop | 2× TP2 + layerwise offload |
| RTX 4090 24 GB | 1× layerwise offload + `convrot_int8` | Spectrum and approximate attention backends are opt-in |

### GB300 single-host

Four requests on **1 host × 4 NVIDIA GB300**, using Linux ARM64, SGLang main
[`15d2cbcc90fc`](https://github.com/sgl-project/sglang/commit/15d2cbcc90fc66b4d08745994cc707a45b067ffc),
PyTorch 2.13.0+cu130, and NCCL 2.29.7. The source checkout was installed with
`python3 -m pip install -e "python[diffusion]" --upgrade` inside
`lmsysorg/sglang:latest`; this is a source-validation environment, not a claim
that the image's bundled release contains that revision.

The command matches the GB300 default in the builder: Ulysses4, resident
weights, automatic encoder folding, native BF16/FP32, platform-default
Dynamic cuDNN/FA attention, and eager execution. The model was downloaded by
`sglang serve` from `MiniMaxAI/MiniMax-H3`.

The workload uses the builder's cat-band prompt, FL2VA T2VA, a 5-second request
at a 768-pixel short edge, 50 inference steps, `quality="exact"`, one output,
and seed 1101. Every MP4 contained video and audio and passed a full decode check.

| First full request after server warmup | Following three requests | Warm median |
| ---: | --- | ---: |
| 35.16 s | 33.11 / 33.10 / 33.10 s | **33.10 s** |

These are client-observed completion times, including up to one second of
status-polling delay. Download, model loading, and server warmup are excluded.
This is a deployment smoke/latency measurement, not a cross-framework quality
audit, a minimum-GPU claim, or an extrapolation to GB200/NVL72 scaling.

### Ascend NPU topology comparison

Both topologies used Laser Attention and the explicit Cache-DiT configuration
from the Ascend launch commands, with `--dit-cpu-offload false` keeping the DiT
resident. The measured workload was one 5-second T2VA request at 1344×768,
124 frames, 24 fps, and 50 inference steps.

| NPU count | Topology | End-to-end latency |
| ---: | --- | ---: |
| 8 | TP2 + SP4 | **55.21 s** |
| 4 | TP2 + SP2 | **96.48 s** |

These are individual end-to-end measurements for each topology, not averages.
The eight-NPU topology had 42.8% lower end-to-end latency than the four-NPU
topology.

### B300 precision and encoder placement

A 12-configuration sweep on a single 8× B300 host, covering both checkpoint
partitions, both transformer precisions, and all three text-encoder
placements. It answers one question — *how long does one request take, and how
much memory does it need*.

### What was measured

**Hardware.** 8× NVIDIA B300 SXM6, single node.

**Model.** `MiniMaxAI/MiniMax-H3`, both released weight partitions.

**Serve command.** Exactly the recipe the picker emits for B300, plus the one
or two overlay flags under test:

```bash Command
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 8 \
  --ulysses-degree 8 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30010
```

The swept axes are `--model-variant` (`fl2va` / `ref2va`), `--quantization`
(unset for BF16 / `fp8`), and `--encoder-parallel` (`auto` / `fold` /
`replicate`). Nothing else differs between the 12 servers.

This is a single-request latency sweep (`batching_max_size: 1`), so encoder DP
is intentionally excluded: it cannot distribute a batch of one. Use the
**DP for a request batch** setting above for a compatible multi-request
deployment; the table below does not claim a measured H3 DP speedup.

**Driver.**

```bash Command
python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
  --host 127.0.0.1 --port 30010 \
  --model MiniMaxAI/MiniMax-H3 \
  --dataset vbench --task text-to-video \
  --num-prompts 1 --max-concurrency 1 \
  --warmup-requests 1 --warmup-inference-steps 50 \
  --extra-body '{"task":"t2va","conditions":[],"target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5.0},"seconds":5,"flow_shift":12.0,"audio_flow_shift":3.0}'
```

**Workload**

| Property | Value |
| --- | --- |
| Output duration | 5.167 s |
| Resolution | 1344×768 |
| Frames | 124 @ 24 fps |
| Denoising steps | 50 |
| `flow_shift` / `audio_flow_shift` | 12.0 / 3.0 |
| Requests in flight | 1 (`--max-concurrency 1`, server at `batching_max_size: 1`) |
| Requests measured | 1 per cell, after 1 warmup request |

### Results

| Weights | Precision | Encoder | Load | Warmup | Latency | Peak/GPU |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FL2VA | BF16 | auto | 118.1 s | 29.65 s | **19.04 s** | 83,578 MB |
| FL2VA | BF16 | fold | 114.0 s | 28.72 s | **19.04 s** | 83,578 MB |
| FL2VA | BF16 | replicate | 116.0 s | 28.33 s | **19.04 s** | 124,158 MB |
| FL2VA | FP8 | auto | 116.0 s | 27.16 s | **18.03 s** | 51,926 MB |
| FL2VA | FP8 | fold | 116.0 s | 25.99 s | **18.04 s** | 51,926 MB |
| FL2VA | FP8 | replicate | 118.0 s | 27.97 s | **18.04 s** | 92,506 MB |
| Ref2VA | BF16 | auto | 114.0 s | 38.69 s | **29.12 s** | 83,968 MB |
| Ref2VA | BF16 | fold | 118.0 s | 36.58 s | **29.13 s** | 83,968 MB |
| Ref2VA | BF16 | replicate | 116.0 s | 35.17 s | **29.13 s** | 124,490 MB |
| Ref2VA | FP8 | auto | 124.0 s | 34.30 s | **27.12 s** | 52,816 MB |
| Ref2VA | FP8 | fold | 112.0 s | 34.44 s | **27.12 s** | 52,816 MB |
| Ref2VA | FP8 | replicate | 116.0 s | 33.42 s | **27.12 s** | 93,396 MB |

### FastH3 on GB300

A 4× GB300 host served [FastH3](#6-fasth3-8-step-distilled) with the VSA-H3
recipe above (1344×768 at 24 fps with audio, `task: "t2va"`,
eight DiT forwards, the default `quality: "extra-high"` and sparsity 0.8,
eager BF16, Ulysses4, `NCCL_NVLS_ENABLE=0`), measured with the MP4 encoded
while the VAE decodes
([sgl-project/sglang#41819](https://github.com/sgl-project/sglang/pull/41819)).
The workload is 72 requests at
concurrency 1 after one warm request per shape: six scene families, 1,000- and
10,000-token prompts, 5 / 10 / 15 s, two seeds, every prompt unique so no
conditioning cache hits. E2E is the client wall clock from submission to the
last byte of the MP4; the stage columns are the server timings of the same
requests. Each cell is the median of the 12 requests of that shape:

| Prompt / requested / aligned | Text encoder | Denoise (8 forwards) | Decode (video + audio) | E2E | Peak/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K / 5 s / 124 | 0.10 s | 3.84 s | 0.56 s | **5.20 s** | 94,468 MB |
| 1K / 10 s / 243 | 0.10 s | 8.45 s | 1.12 s | **10.74 s** | 97,628 MB |
| 1K / 15 s / 362 | 0.10 s | 14.85 s | 1.72 s | **18.24 s** | 100,246 MB |
| 10K / 5 s / 124 | 0.18 s | 6.31 s | 0.55 s | **7.90 s** | 94,510 MB |
| 10K / 10 s / 243 | 0.18 s | 12.35 s | 1.12 s | **14.98 s** | 98,128 MB |
| 10K / 15 s / 362 | 0.18 s | 20.28 s | 1.67 s | **24.01 s** | 100,408 MB |

Over all 72 requests the E2E p50 / p90 / p95 is 12.94 / 23.94 / 24.08 s, and
every MP4 carries 1344×768 H.264 at 24 fps with a 32 kHz stereo AAC track.

### VDN-H3 on B200

An 8× B200 (SM100, 183 GB) host served [VDN-H3](#7-vdn-h3-hybrid-attention-8-step-distill)
on the paper workload: 1344×768 at 24 fps with audio for 14.375 s (345 frames,
102 latent frames, about 104k packed rows), `task: "t2va"`,
nine sigma grid points (8 DiT forwards), seed 1000, `hybrid_window_attn_h3`,
eager, and the warmup run at the served clip shape
(`--warmup-num-frames 345 --warmup-resolutions 1344x768`), so every forward
of the served request is steady state; without those flags the first forward
pays 2 to 3 s of allocator growth and kernel setup. "Steady
s/NFE" is the mean of forwards 2 to 8. The OpenVDN reference rows ran the released
inference stack (`8nfe_tuned_fp8.yaml`, `infer_ulysses.py`) on the same host
and the same clip length, with its Ulysses layout (`parallel.softmax_ranks`)
swept at every GPU count rather than assumed. The SGLang rows were re-measured
in one campaign at `f1a2ee763d`, one arm at a time on an otherwise idle node,
each with the [8×B200 recipe](#7-vdn-h3-hybrid-attention-8-step-distill) and
only `--num-gpus` and `--quantization` changed; the OpenVDN reference rows come
from an earlier campaign on the same host type. A repeated arm moves by about
3% across nodes, so treat smaller differences as noise:

| Config | Steady s/NFE | Denoise (8 forwards) | Decode | Peak/GPU |
| --- | ---: | ---: | ---: | ---: |
| OpenVDN published, FP8, 8× B200 Ulysses 5+3 ([openvdn.github.io](https://openvdn.github.io/)) | 1.40 | 11.2 s | — | — |
| OpenVDN reference, BF16 (`8nfe_tuned.yaml`), 1× B200 | 7.66 | 61.2 s | — | — |
| OpenVDN reference, FP8, 1× B200 | 6.25 | 50.0 s | — | — |
| OpenVDN reference, FP8, 2× B200 Ulysses (standard / 1+1 branch-parallel) | 4.87 / 5.04 | 39.0 / 40.3 s | — | — |
| OpenVDN reference, FP8, 4× B200 Ulysses (3+1 branch-parallel) | 2.59 | 20.7 s | — | — |
| OpenVDN reference, FP8, 8× B200 Ulysses (5+3 / 4+4 / standard) | 1.40 / 1.47 / 1.52 | 11.2 / 11.7 / 12.2 s | — | — |
| SGLang, `--quantization bf16`, 1× B200, `--performance-mode speed`, served-shape warmup | 7.27 | 58.2 s | 7.0 s | 156,298 MB |
| SGLang, `--quantization fp8` (→ online `mxfp8` on SM100+), 1× B200, `--performance-mode speed`, served-shape warmup | 5.93 | 47.4 s | 7.0 s | 128,420 MB |
| SGLang default (online `mxfp8`), 2× B200 Ulysses2, `--performance-mode speed`, served-shape warmup | **3.25** | **26.0 s** | 3.7 s | 102,664 MB |
| SGLang, `--quantization bf16`, 4× B200 Ulysses4, `--performance-mode speed`, served-shape warmup | 2.05 | 16.4 s | 2.1 s | 104,358 MB |
| SGLang default (online `mxfp8`), 4× B200 Ulysses4, `--performance-mode speed`, served-shape warmup | **1.67** | **13.3 s** | 2.1 s | 81,168 MB |
| SGLang default (online `mxfp8`), 8× B200 Ulysses8, `--performance-mode speed`, served-shape warmup | **0.85** | **6.8 s** | 1.4 s | 76,918 MB |
| SGLang, per-channel fp8 weight scales (the online `fp8` path of other models; on SM100+ `--quantization fp8` resolves to `mxfp8` and no flag selects the per-channel path, so this arm forces it), 8× B200 Ulysses8 | 0.91 | 7.3 s | 1.4 s | 76,450 MB |

On 8× B200 the SGLang Ulysses8 path runs the paper workload at 0.85 s/NFE
(6.8 GPU-seconds per NFE) against the published 1.40 s/NFE of OpenVDN's 5+3
branch-parallel layout; the whole request (text encoding, 8 forwards, joint
decode) completes in about 8.2 s after warmup. At this GPU count the step is
launch- and copy-bound in the linear branch rather than FLOP-bound, and the
transformer runs online `mxfp8` by default (e4m3 with one E8M0 scale per 32
elements into cuBLASLt's block-scaled GEMM, with the activation quant fused
into the adaLN and SwiGLU producers).

Against the published 8× B200 headline (1.40 s/NFE, 11.2
GPU-seconds per NFE), the SGLang 4× B200 run spends 6.7 GPU-seconds per
NFE, with half the GPUs and half the all-to-all fan-out. At equal GPU count
SGLang is 1.50× faster at two cards (3.25 vs 4.87 s/NFE), 1.56× at four
(1.67 vs 2.59) and 1.66× at eight (0.85 vs 1.40), while on one card the two
stacks are within 5% of each other (5.93 vs 6.25 s/NFE quantized, 7.27 vs 7.66
in BF16). The advantage is therefore parallel scaling rather than per-GPU
kernel work: SGLang holds 88-91% parallel efficiency from 2 to 8 cards against
the reference stack's 56-64%, and sweeping the reference stack's own layout
does not close it (its best split is 8% better than standard Ulysses at eight
cards, 1.40 against 1.52 s/NFE).

On one B200 both single-GPU rows run with the DiT resident under the auto
memory policy (128 GB peak quantized, 156 GB in BF16, of the card's 183 GB),
so no `--layerwise-offload-components` flag is needed at this clip length and
the single-GPU rows are compute-bound.

### VDN-H3 on RTX PRO 6000

The same paper workload (1344×768, 345 frames, 8 DiT forwards, seed 1000,
`hybrid_window_attn_h3`, `--performance-mode speed`, served-shape warmup) on
RTX PRO 6000 Blackwell Server Edition cards (SM120, 96 GB, PCIe, no NVLink).
The window runs on FA4's sm120 kernel and the transformer defaults to online
`mxfp8` (a cutlass sm120 block-scaled GEMM); `--quantization fp8` maps to it.
One card needs `--layerwise-offload-components text_encoder`: the fp8 DiT
(33 GB) and the bf16 text encoder (48 GB) do not both fit, and the whole-module
`--text-encoder-cpu-offload` re-homes the encoder in one piece and runs out of
memory when it is used. Multi-card Ulysses goes over PCIe; the all-to-all is
the bulk of the step there (54% of GPU time at 8 cards), so the per-card
efficiency drops with count. A 2-card run should use a pair on one PCIe
switch (`nvidia-smi topo -m` shows `PIX`): a pair across the CPU root complex
measured 15.7 to 17.0 s/NFE on a shared host.

| Config | Steady s/NFE | Denoise (8 forwards) | Decode | Peak/GPU |
| --- | ---: | ---: | ---: | ---: |
| 1× RTX PRO 6000, `--layerwise-offload-components text_encoder` (default `mxfp8`) | **20.27** | **160.0 s** | 20.6 s | 95,790 MB |
| 1× RTX PRO 6000, same, per-channel `fp8` path (before this mapping) | 22.85 | 180.3 s | 20.7 s | 95,160 MB |
| 2× RTX PRO 6000 Ulysses2, one PCIe switch (default `mxfp8`) | 13.61 | 107.4 s | 10.6 s | 93,846 MB |
| 2× RTX PRO 6000 Ulysses2, one PCIe switch, per-channel `fp8` | 14.97 | 118.1 s | 10.6 s | 94,700 MB |
| 4× RTX PRO 6000 Ulysses4 (default `mxfp8`) | 7.98 | 63.0 s | 5.6 s | 82,178 MB |
| 4× RTX PRO 6000 Ulysses4, per-channel `fp8` | 8.23 | 65.1 s | 5.6 s | 82,422 MB |
| 8× RTX PRO 6000 Ulysses8 (default `mxfp8`) | 5.43 | 43.1 s | 3.5 s | 77,704 MB |
| 8× RTX PRO 6000 Ulysses8, per-channel `fp8` | 5.47 | 43.4 s | 3.5 s | 77,236 MB |

For comparison the same code on B200 measures 6.03 / 3.30 / 1.67 / 0.88 s/NFE
at 1 / 2 / 4 / 8 cards (NVLink, 86% parallel efficiency at 8).

### H200 topology comparison

The same four-card H200 host completed both lossless resident placements with
the standard 1344×768, 5-second, 50-step T2VA request (fixed prompt and seed,
eager BF16/FP32, back-to-back runs on an otherwise idle host). Latency is the
warmed-up request; the first pair uses the default warmup request, the second
pair adds `--warmup-resolutions 1344x768` so warmup already covers the served
resolution:

| Topology | Warmup | Denoise | Decode | E2E | Peak/GPU |
| --- | --- | ---: | ---: | ---: | ---: |
| Ulysses4 | default | 79.04 s | 3.77 s | **84.14 s** | 94,288 MB |
| TP2 + Ulysses2 | default | 81.17 s | 2.97 s | 85.51 s | 63,490 MB |
| Ulysses4 | `--warmup-resolutions 1344x768` | 71.73 s | 1.32 s | **74.38 s** | 94,290 MB |
| TP2 + Ulysses2 | `--warmup-resolutions 1344x768` | 75.52 s | 1.29 s | 78.33 s | 63,490 MB |

Ulysses4 stays the H200 latency default: 5.0 % faster end-to-end than
TP2 + Ulysses2 once warmup covers the served resolution (1.6 % with the
default warmup, where first-request cold start masks the topology gap).
TP2 + Ulysses2 shards the DiT weights and holds peak memory about 30 GB per
GPU lower, which is why it remains the 80 GB H100 recipe. Matching the warmup
request to the served resolution removes the cold first-request cost on both
topologies (about 10 s end-to-end on this workload).

### H200 cross-node scaling

Long references and long durations grow the packed sequence length, and
Ulysses alone cannot scale sequence parallelism past the GPU count of one
node without either violating head-count divisibility or exposing
all-to-all traffic across the slower inter-node link. H3 combines
node-local Ulysses with cross-node Ring: Ring's point-to-point KV rotation
is designed to overlap with attention compute, which fits a slower
cross-node link better than an all-to-all does.

**Hardware.** 2 nodes × 8× NVIDIA H200 SXM, same cluster, InfiniBand
between nodes.

**Serve command.** The cross-node cell the picker emits for H200, run
identically on both nodes with `--node-rank` set to 0 and 1:

```bash Command
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 16 \
  --nnodes 2 \
  --node-rank {{NODE_RANK}} \
  --dist-init-addr {{NODE0_IP}}:20000 \
  --sp-degree 16 \
  --ulysses-degree 8 \
  --ring-degree 2 \
  --encoder-parallel replicate \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30010
```

**What was measured.** A controlled denoise-stage comparison on identical
hardware: 8× H200 single-node (Ulysses8, no Ring) versus the same 16-GPU
cross-node command above (Ulysses8 × Ring2), holding prompt, seed, and
step count fixed:

| Task | Single-node (Ulysses8) | Cross-node (Ulysses8 × Ring2) | Change |
| --- | ---: | ---: | ---: |
| T2VA denoise/step | 0.749 s | 0.477 s | −36.3% |
| Ref2VA/V2V denoise/step | 2.572 s | 1.494 s | −41.9% |

The gain grows with sequence length because Ring's per-hop communication
cost stays roughly constant while attention compute grows quadratically
with sequence length, so V2V's longer packed sequence benefits more than
T2VA's shorter one. With the point-to-point KV rotation pipelined against
attention compute, one V2V request's full denoise stage completed in
68.1–68.3 seconds versus 128.6 seconds on the single-node 8-GPU baseline
(−47.0%), with byte-identical output to the unpipelined cross-node path.

Cross-node determinism was confirmed separately: the same request run
twice against the same cross-node deployment produced byte-identical
output. A cross-node run's output is not expected to bit-match a
single-node run of the same prompt and seed — Ring's online-softmax merge
across hops accumulates floating-point operations in a different order
than single-node attention, which is an expected source of bit-level
difference, not a correctness regression.

<Warning>
`--encoder-parallel auto`'s fold decision is not yet node-boundary aware
and attempts to fold the text encoder across nodes, which crashes the
Ref2VA reference-conditioned encoder. Always pass
`--encoder-parallel replicate` explicitly for cross-node H3 deployments.
</Warning>

### H100 topology comparison

The same four-card H100 host completed three lossless placements. TP2 with
Ulysses2 was the fastest; TP4 used the least memory:

| Topology | Pipeline latency | Peak/GPU |
| --- | ---: | ---: |
| TP2 + Ulysses2 | 13.25 s | 66.04 GB |
| FSDP + Ulysses4 | 13.36 s | 57.01 GB |
| TP4 + Ulysses1 | 13.86 s | 49.80 GB |

### RTX 5090 capacity run (two cards)

The single-card recipe is a different one — see *RTX 5090 single card, physical
desktop* below; the twenty resident layers here are half a layer per GPU under
TP2 and do not transfer to one card. The verified two-card RTX 5090 host used TP2 with layerwise offload. The full
50-step, 1344×768, 5-second request completed in 559.67 seconds: 525.05
seconds of denoising and 33.61 seconds of decoding, with a 26.3 GiB sampled
peak per GPU.

| DiT settings | 5-step denoise | Inference | Peak/GPU | Result |
| --- | ---: | ---: | ---: | --- |
| prefetch 1, resident 20 | 43.48 s | 78.11 s | 26.3 GiB | Selected recipe |
| prefetch 2, resident 20 | 43.37 s | 78.06 s | 27.5 GiB | No measurable gain |
| Ulysses2, prefetch 2, resident 10 | Did not reach warmup | — | — | Rejected |

### Consumer GPU tuning

On consumer hardware the binding question is not which card you have but how much
host RAM sits behind it. H3's weights are about 108 GB — 61.73 GB of DiT and
46.18 GB of text encoder — so no consumer configuration holds them all, and where
the shortfall lands decides the throughput.

**The command** — most consumer machines need exactly one flag beyond the model:

```bash consumer single GPU, lossless
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --layerwise-offload-components dit,text_encoder,vae
```

With 16 GB of VRAM or more, add `--layerwise-resident-layers video_vae=36` for
the 13 s decode; with ~96 GB of host RAM and 16 GB+ of VRAM, add
`--dit-layerwise-resident-layers 4` for the 6 s step. That is the whole flag
surface. The [builder at the top of this page](#1-quick-start) has consumer
cards and a Host RAM selector: pick your budget and it emits this command with
your tier's measured expectations attached as comments. The table below is the
same data in one view.

**Unified-memory machines are the exception**: on a DGX Spark (GB10, 128 GB
shared between CPU and GPU) launch with **no flags at all**. The deployment
still exceeds the pool, automatic offload engages on its own, and its placement
beat the explicit recipe above by 2.1× on the denoise (12.1 vs 25.8 s/it
measured on the same box) — the VRAM/host split that justifies every flag in
this section does not exist there. Expect ~12 min of load, a ~5.5 min text
encoding stage per request (steady-state compute on this chip, not a stall),
and ~12 min per warm 480P request.

**Two budgets, and what each one buys**

| | 12 GB VRAM + 32 GB host | host free, VRAM 16 GB |
| --- | ---: | ---: |
| Recipe | A | B |
| Peak VRAM | ≤ 12 GiB | ≤ 16 GiB (OOMs at 12) |
| Host anonymous (must fit) | 24.5 GiB | 116.7 GB pinned |
| Denoise, 864×480 / 124 frames / 20 NFE | 16.8 - 18.7 s/it | **6.01 s/it** |
| Runs at all | yes | yes |

The left column is one configuration measured twice, at 318.94 s and 356.37 s;
the 12% spread tracked host load on a shared machine, so treat smaller
differences than that as unresolved. The right column is 120.92 s at a 16 GiB
allocator cap. Four resident DiT layers is what Recipe B buys its speed with,
and it is also why 12 GiB is not enough for it.

Read the host row carefully, because the two numbers are not the same kind of
memory. *Anonymous* host memory — pinned buffers and pageable copies — has to fit,
and the kernel cannot reclaim it. Page cache backing a file mapping is
*droppable*, so it does not count against the budget even though it shows up in
`VmRSS`; use `RssAnon` from `/proc/<pid>/status` when checking. Likewise measure
VRAM with `torch.cuda.set_per_process_memory_fraction` and let the allocator fail,
rather than reading `nvidia-smi`, which reports the caching allocator's reserved
pool and overstates the requirement.

Inside 32 GB the weights cannot be pinned, so each denoise step copies about
60 GiB from the checkpoint mapping, and a mapped source is synchronous however
the copy is requested: the driver stages it through its own buffer, so the
transfer neither overlaps compute nor runs at pinned bandwidth. That is where
the step goes, and giving the host room to pin the weights instead is what takes
it to 6.01 s.

Two caveats on the constrained number, both from instrumenting the run rather
than from arithmetic. The machine it was measured on has 2 TB of host memory, so
the kernel kept all 107.7 GiB of mapped checkpoint pages resident: major faults
across a whole request were 6, and `read_bytes` was zero. Nothing was read from
disk. A real 32 GB host cannot cache 107.7 GiB, so it will fault and re-read,
and should be expected to be slower than the figures here rather than equal to
them — an NVMe is a requirement, not a recommendation. Measure your own machine
with major faults (`/proc/<pid>/stat`) on the worker process, not on the
launcher, which holds no weights.

**Recipe A — fits 12 GB VRAM + 32 GB host**

```bash 12 GB + 32 GB, lossless
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder,vae \
  --layerwise-resident-layers video_vae=36
```

**Recipe B — host memory is free (the fast path)**

```bash unconstrained host, lossless
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder,vae \
  --dit-layerwise-resident-layers 4 \
  --layerwise-resident-layers video_vae=36
```

Recipe B pins ~112 GB of host memory (DiT 61.56 GB, text encoder 46.18 GB, VAE
~4.5 GB in its decode dtype). Do not reach for it on a 32 GB machine.

**What not to change, and why**

- `video_vae=36` holds every decoder block for the decode only — residency
  arms at the decoder's first block and releases when it finishes, so the
  denoise still runs on an empty card. It fits 12 GB because decoder weights
  are held in their decode compute dtype (fp16) from load, which halves them
  to ~4.9 GiB; the rounding was already part of every output (the decode
  computes in fp16 autocast), so the result is bit-identical, and the decode
  drops from 60 s streamed (or 209 s on a busy host) to ~10 s. The
  `expandable_segments` line stays: the decode sits close enough to the cap
  that fragmentation otherwise tips it over.
- Leave `--enable-torch-compile` off, as elsewhere on this page. Layerwise offload
  rebinds `param.data` on every layer, so compiled graphs do not get the benefit
  they would on resident weights.
- Recipe A's flags are what the automatic policy should choose on its own. Until
  the model declares its own placement, `--performance-mode memory` plus the
  explicit component list is what makes it happen; pass them.

**Reading the startup log**

The server prints the memory decisions it made; checking three lines against
your budget catches a mis-set machine in the first minute instead of the first
request.

- `Layerwise offload: host memory available: N GiB` — what the runtime sees
  after loading, not your DIMM size. On a 32 GB host expect single digits here;
  a much larger number means another process's memory accounting (or a
  container limit) is in play.
- `leaving N GiB of weights on the checkpoint mapping` — the expected line on a
  32 GB host: the DiT streams from the checkpoint file. If instead the log
  reports pinned weights, the runtime decided your host has room — which is
  faster, and means the 32 GB figures above do not apply to you.
- `Loaded video_vae: ... host mmap` vs `host pageable` — where the VAE landed
  (decoder weights are ~4.9 GiB once held in their decode dtype).
  `Loaded <component>` lines carry the same buckets for every component.

If a request dies after the denoise finishes, it is the decode colliding with
the cap: keep the `expandable_segments` line, and if it persists drop to
`video_vae=24` and take the partially streamed decode.

**Against ComfyUI, on the same weights**

Same unpruned bf16 checkpoints, same card, same sampler settings (cfg 1.0,
euler_ancestral, sigma shift 12.0/3.0, seed 1101), 864×480 / 124 frames / 20 NFE:

When host memory is free, the engines are close and sglang is ahead:

| | denoise | host anonymous | peak VRAM |
| --- | ---: | ---: | ---: |
| sglang, Recipe B | **6.01 s/it** | 116.7 GB pinned | ≤ 16 GiB |
| ComfyUI KSampler | 6.58–6.59 s/it | 116.5 GiB | 13048 MiB |

Inside 12 GB, both engines run these weights, and one measurement convention
matters on each side. ComfyUI's memory manager reads system RAM and adapts, so
the rows below patch `psutil` to a pretend host size — the same convention the
sglang rows use. Its `--reserve-vram` is also soft: told to keep 12 GiB free it
still peaked at 13.5 GiB, a figure a real 12 GB card cannot give it, so both
engines here run under the same hard allocator cap
(`set_per_process_memory_fraction`), where its peak stays at 12.1–12.3 GiB.
Under that cap, Recipe A wins the whole request at every host size:

| 12 GB VRAM, both engines hard-capped | sglang Recipe A (TE + denoise + decode) | ComfyUI, bf16 (warm) |
| --- | ---: | ---: |
| 32 GB host | 12.4 + 212.8 + 9.4 ≈ **235 s** | 276–302 s |
| 48 GB host | 15.8 + 192.1 + 10.0 ≈ **218 s** | 246–267 s |
| 64 GB host | 7.5 + 162.4 + 10.3 ≈ **180 s** | 194–195 s |

Same GPU, same load window, unpruned bf16 checkpoints, outputs verified. All
figures are anchored at 480P — activations grow with the pixel count, so at
768P drop the resident DiT layers to 0 first, then `video_vae` to 24 if the
decode still collides.

One convention behind that table is the one a home machine does not share: the
host sizes above are `psutil` patches on a 2 TB lab machine, so the kernel kept
the whole checkpoint in page cache and nothing was read from disk. A physical
host of that size re-reads the streamed weights every step, and what that costs
is computable — `bytes per step = (50 - resident DiT layers - layers the host
pin budget covered) x 1.23 GiB`. The courier overlaps that read with compute, so
the drive stops being the denoise clock once its sequential rate exceeds those
bytes divided by the card's step time. A 32 GB host covers roughly 13 layers,
which puts a 24 GB card at ~38 GiB/step and a 12/16 GB card at ~46 GiB/step: a
PCIe 4.0 NVMe (6–7 GiB/s) clears both, a PCIe 3.0 drive (~3.2 GiB/s) becomes the
clock, and a SATA SSD is not usable for this model. Resident DiT layers remove
1.23 GiB/step each, but only while the streamed set is larger than the pin
budget — once the read fits under the step's compute time, more of them buy
nothing and only cost VRAM.

An earlier revision of this page recorded 38 s/step on a physical 4090 with a
990 Pro, reading 52.9 GB per step — about 1.4 GB/s, and faulted sequentially, so
almost none of it showed in `majflt` (measure `read_bytes`, not major faults).
That rate was the old path, not the drive: the pinned-store pool rounded every
block up to a power of two, so the pin budget over-committed and the machine had
to run with pinning off, and the layers that stayed mapped faulted in 4 KiB at a
time. Exact-size pinning and O_DIRECT reads through the courier removed both,
and the physical 5090 run below shows the drive fully hidden behind compute. A
4090 on a 32 GB host is derived to return to its compute wall (~8.5 s/step) on a
PCIe 4.0 drive; that machine has not been re-measured. The
VRAM axis holds too: capped at 16 GiB the same recipe wins ~250 vs 292–301 s,
and at 24 GiB (with `--dit-layerwise-resident-layers 6` — measured at a
22 GiB cap so a desktop's own allocations fit; a headless card can raise it
to 10 for under 1% more) ~8.5 s/step vs ComfyUI's 249–260 s requests. Four changes carry it: the VAE staying on its checkpoint mapping (#35862, root fix
#35946), per-layer pinning with net-cost accounting (#35867), the courier
thread that ships still-mapped layers through pinned slots (#35882), and
decoder weights held in their decode dtype from load (#35967) — which is what
lets `video_vae=36` fit and turns the decode from the slowest stage (54–96 s
streamed) into the fastest (~10 s, faster than ComfyUI's own 15–25 s). Output
equivalence is bit-level: the fp16-held decode reproduced the fp32-store run's
video byte for byte, and the audio stream is bit-identical.

Stage by stage under the cap: text encoding is even (both stream the same
48 GB Qwen3VL), the denoise leads at 32–48 GB hosts and sits within
run-to-run variance of ComfyUI at 64 GB (162 vs 159 s), and the decode leads
everywhere. Two ComfyUI notes that still matter: `--fast-disk` measured no
faster than its default on this lab host — though on a physical 60 GB desktop
its default is OOM-killed while loading (55.6 GB of anonymous memory) and
`--fast-disk` is what makes it run at all — and stacking
`--novram --cache-none --disable-pinned-memory` made things strictly worse
(69.1 GiB anonymous, 750 s requests) — the adaptive default is the right
configuration on a small host.

The path ComfyUI ships for 12 GB cards uses
`minimax_h3_fl2va_pruned_int8_convrot` and `qwen3vl_32b_minimax_h3_nvfp4_awq`,
i.e. an int8 DiT and an NVFP4 text encoder, and its pruned bf16 file is 40.2 GB
against the unpruned 66.3 GB. Those are different weights, so it is not a
like-for-like comparison with the recipes above.

### RTX 5090 single card, physical desktop

Every consumer figure above this point was measured with at least one axis
simulated: a hard allocator cap standing in for VRAM, a `psutil` patch standing
in for host RAM, and a lab machine whose page cache held the whole checkpoint.
This run had none of that — an RTX 5090 (32 GB) in a Ryzen 9 9950X desktop with
60 GB of RAM, an 8 GB swap file and a Samsung 9100 PRO on PCIe 5.0, on Ubuntu
24.04. The workload matches the comparison above: FL2VA T2VA, 864×480, 124
frames, 20 steps, cfg 1.0, euler_ancestral, sigma shift 12.0/3.0, same seeds and
keyframes on both engines, unpruned bf16 weights on both sides.

```bash 32 GB VRAM + 60 GB host, lossless
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder,vae \
  --dit-layerwise-resident-layers 14 \
  --layerwise-resident-layers video_vae=36 \
  --warmup-resolutions 864x480 --warmup-num-frames 124
```

| | text encoding | denoise step | decode | request |
| --- | ---: | ---: | ---: | ---: |
| sglang | **4.2 s** | 5.14–5.17 s | **6.4 s** | **112.2 / 112.1 s** |
| ComfyUI `--fast-disk --cache-none` | 11.3 s | 5.17–5.30 s | 7.2 s | 145.9 / 145.6 / 140.9 s |

Three things a capped run cannot settle:

**The denoise is at the card's compute wall, not the drive's.** The two engines
are within noise of each other per step, and prefetch depth 2 and cuDNN SDPA
each measured as no change. The 51.3 GiB the DiT reads from the drive per
request is fully overlapped — across a whole request the compute thread waited
0.9 s for the courier.

**Host RAM decides how much traffic there is to hide.** At 60 GB the pin budget
covers 34 of the 50 DiT layers, so only about two layers per step come off the
disk. The 46 GiB text encoder is never pinned — streamed layers take the budget
first — and is read whole, once per request, at the drive's sequential rate:
that is the 4.2 s stage, against 12.1–12.9 s when the same bytes go through the
page cache.

**Neither engine's default command survives this machine.** ComfyUI's is
OOM-killed during load (55.6 GB of anonymous memory) and needs `--fast-disk`;
sglang's was too, until pinned stores were allocated at their exact size instead
of the allocator's next power of two. Even now the automatic placement keeps no
resident DiT layers and streams the VAE on every decode, so the flags above are
what reach the numbers in the table.

Two caveats to carry into your own machine. The run finished with its 8 GB swap
file full, because the pin budget takes 95% of the memory available after
loading and there is no smaller setting yet — on a machine that is also your
desktop, expect that pressure and give the card more resident layers, since a
resident layer is one the host no longer has to pin. And these are 480P figures:
at 768P drop the resident DiT layers first, then `video_vae` to 24.

### RTX 4090 24 GB single-GPU run

One RTX 4090 D 24 GB completed the 1344×768, 107-frame, 20-NFE T2VA
workload (euler, `torch.compile` and step caching disabled) with DiT and
text-encoder layerwise offload. Same process: load → warmup (seed 0) →
timed (seed 42); only the timed pass is reported. GPU peak stayed about
18 GB.

| Config | Timed e2e | Denoise | vs BF16 | PSNR vs BF16 |
| --- | ---: | ---: | ---: | ---: |
| BF16 + FlashAttention | 405.6 s | 370.2 s | 1.00× | — |
| `convrot_int8` + FA | 303.3 s | 273.7 s | 1.34× | 24.81 dB |
| `convrot_int8` + `sol_attn` | 223.9 s | 203.9 s | 1.81× | 24.44 dB |
| `convrot_int8` + `sage_attn` | 174.9 s | 154.2 s | 2.32× | 23.51 dB |
| `convrot_int8` + Sage→Sol hybrid | 163.8 s | 143.1 s | 2.48× | 23.04 dB |

`convrot_int8` + FA changes Linear numerics only. The `sol_attn` /
`sage_attn` / hybrid rows also change the attention algorithm, so speed
and pixel fidelity rank in opposite orders there. Default remains
`convrot_int8` + `fa`. Cache-DiT can share this layerwise recipe; omit
`quality` and see the quality-level section.

Same card, 50-step run. Speedup is vs BF16+FA. PSNR is vs same-tree INT8+FA (Spectrum off).

| Config | e2e | vs BF16+FA | PSNR mean / min |
| --- | ---: | ---: | ---: |
| BF16 + FlashAttention | 1008.6 s | 1.00× | — |
| `convrot_int8` + FA | 718.5 s | 1.40× | — |
| `convrot_int8` + FA + Spectrum `11 / 5 / 1.0` | **285.3 s** | **3.53×** | **27.4 / 25.3** |
| `convrot_int8` + `sage_attn` + Spectrum `15 / 5 / 0` | 245.3 s | 4.11× | 23.6 / 21.0 |
| `convrot_int8` + Sage→Sol hybrid + Spectrum `15 / 5 / 0.5` | **219.0 s** | **4.61×** | 20.6 / 17.8 |

Prefer FA + `11 / 5 / 1.0` when the bar is FA-like. Sage / hybrid + Spectrum is faster; quality then follows the attention backend, not the skip schedule.

### AMD Instinct task and scaling runs

The AMD recipes keep the released BF16/FP32 precision policy and use AITER
packed attention. The picker emits the fastest measured topology, 8 GPUs with
Ulysses degree 8. All runs below completed full H.264/AAC decoding and
representative-frame inspection.

| Hardware | Task | Denoise | Decode | Peak/GPU |
| --- | --- | ---: | ---: | ---: |
| MI355X | T2VA | 55.2907 s | 9.5344 s | 97,444 MB |
| MI355X | FL2VA | 53.7978 s | 9.4477 s | 96,922 MB |
| MI355X | Ref2VA | 41.3812 s | 6.8247 s | 94,518 MB |
| MI300X | T2VA | 167.4878 s | 25.3244 s | 97,272 MB |
| MI300X | FL2VA | 150.2311 s | 12.5684 s | 96,750 MB |
| MI300X | Ref2VA | 107.6232 s | 11.3768 s | 94,268 MB |

The task matrix used 8 GPUs and 50 denoising steps. The scaling matrix uses
one 1344×768, 209-frame T2VA request and changes only the GPU count and
matching Ulysses degree:

| Hardware | GPUs | Denoise | Decode | Peak/GPU |
| --- | ---: | ---: | ---: | ---: |
| MI355X | 8 | 55.2907 s | 9.5344 s | 97,444 MB |
| MI355X | 4 | 104.2294 s | 11.1824 s | 103,350 MB |
| MI355X | 2 | 223.0246 s | 15.5330 s | 115,250 MB |
| MI355X | 1 | 288.7968 s | 24.0472 s | 137,676 MB |
| MI300X | 8 | 167.4878 s | 25.3244 s | 97,272 MB |
| MI300X | 4 | 297.3727 s | 26.5067 s | 103,436 MB |
| MI300X | 2 | 585.5401 s | 29.4909 s | 115,010 MB |
| MI300X | 1 | 978.0886 s | 36.0142 s | 137,626 MB |

For a measured lower-count AMD deployment, set both `--num-gpus` and
`--ulysses-degree` to 4, 2, or 1. AITER packed attention matched segment-wise
BF16 SDPA at cosine similarity `0.9999991655` on MI355X and `0.9999991059` on
MI300X.

## 10. Run in ComfyUI

import { ComfyUISupport } from '/src/snippets/diffusion/comfyui-support.jsx';

<ComfyUISupport model="minimax-h3" />


---

## FILE: docs/src/snippets/configs/MiniMaxAI/minimax-h3.jsx (https://github.com/sgl-project/sglang/blob/HEAD/docs/src/snippets/configs/MiniMaxAI/minimax-h3.jsx)

```
// MiniMax-H3 diffusion deployment matrix. Consumed by _deployment.jsx.
//
// H3 is the first opt-in user of the scoped commandBuilder renderer. Topology,
// checkpoint, server overlays, and request fields share one semantic selection,
// while the UI presents them by lifecycle and composes them through the existing
// deployment command engine.


// Single-GPU consumer cards run H3 lossless through layerwise offload. The
// flags carry only what differs from the defaults; what changes with the
// machine is the expectation, which the hints spell out per budget. Measured
// on one RTX 4090 (denoise medians across interleaved runs, outputs verified
// end to end); 48-64 GB hosts sit between the measured points.
export const config = (() => {
// One recipe per VRAM tier, measured under a hard allocator cap of that
// size: the figures were taken at 12/16/24 GiB caps, so every card of a
// tier shares them. 30-series cards run the same recipe; their step times
// land above the measured 40/50-series figures.
const CONSUMER_12G = ["rtx4070", "rtx5070", "rtx3060"];
const CONSUMER_16G = ["rtx4080", "rtx5080", "rtx5070ti", "rtx4060ti"];
const CONSUMER_24G = ["rtx4090", "rtx3090"];
// The only tier measured on a physical desktop rather than under an allocator
// cap: an RTX 5090 with 60 GB of host RAM and a PCIe 5.0 NVMe, where the VRAM,
// the host RAM and the drive are all real at once.
const CONSUMER_32G = ["rtx5090"];
// Workstation cards a home builder can actually buy. No hard-cap anchor was
// measured for these sizes (the lab card is 24 GB and caps only shrink), so
// their recipes are derived from the tier logic, not verified runs.
const WORKSTATION_48G = ["rtx6000ada"];
const WORKSTATION_96G = ["rtxpro6000"];
// GB10 unified memory: 128 GB shared between CPU and GPU (121.7 GB visible
// to torch), so the VRAM/host split that shapes every tier above does not
// exist. The 134 GiB deployment still exceeds the pool, and the loader's
// automatic placement handles that split better than any explicit flag set:
// verified on DGX Spark, see unified128Flags().
const UNIFIED_128G = ["dgx-spark"];
const CONSUMER_SINGLE = [
  ...CONSUMER_12G,
  ...CONSUMER_16G,
  ...CONSUMER_24G,
  ...CONSUMER_32G,
  ...WORKSTATION_48G,
  ...WORKSTATION_96G,
  ...UNIFIED_128G,
];
const CONSUMER_VRAM_16_PLUS = [...CONSUMER_16G, ...CONSUMER_24G, ...CONSUMER_32G];
const CONSUMER_AMPERE = ["rtx3060", "rtx3090"];

// The consumer recipes below are single-card. The 5090 is the one consumer
// card with a verified two-card recipe (TP2, twenty resident layers -- half a
// layer per GPU), so a two-card selection keeps the generic offload path.
function consumerSingleCard(s) {
  return (
    CONSUMER_SINGLE.includes(s.hw)
    && !(CONSUMER_32G.includes(s.hw) && Number(s.gpus_per_node) > 1)
  );
}

function consumerFlags(s) {
  if (UNIFIED_128G.includes(s.hw)) return unified128Flags();
  if (WORKSTATION_96G.includes(s.hw)) return workstation96Flags();
  // The whole video decoder held for the decode only: residency arms at the
  // decoder's first block and releases when it finishes, so the denoise still
  // runs on an empty card. All 36 blocks fit 12 GB because decoder weights are
  // held in their decode compute dtype (fp16, ~4.9 GiB) from load -- the
  // rounding was already in every output, so the result is bit-identical --
  // and the decode drops from 60 s streamed to ~10 s.
  const flags = [
    "--performance-mode memory",
    "--layerwise-offload-components dit,text_encoder,vae",
    "--layerwise-resident-layers video_vae=36",
  ];
  if (CONSUMER_VRAM_16_PLUS.includes(s.hw) && s.host_ram === "ram96") {
    flags.push("--dit-layerwise-resident-layers 4");
  }
  // A 24 GB card has headroom for resident DiT layers, but their benefit
  // flattened once the decoder went fp16 and the courier overlapped the
  // streaming: measured at a 22 GiB cap (2 GiB desktop headroom), r10/r6/r4
  // land at 8.41/8.48/8.51 s/step. Six layers keep ~2.4 GiB more free than
  // ten for under 1% of speed -- the desktop-safe point. A 16 GB card keeps
  // the plain recipe; even four resident layers measured slower there.
  if (CONSUMER_24G.includes(s.hw) && s.host_ram === "ram32") {
    flags.push("--dit-layerwise-resident-layers 6");
  }
  // Fourteen layers (~17 GiB) leave the decode its room on a 32 GB card and
  // shrink the streamed set the host has to pin. Measured at a 60 GB host;
  // a smaller host does not change what fits on the card, so the count holds
  // there too and matters more, since fewer streamed layers get pinned.
  if (CONSUMER_32G.includes(s.hw) && s.host_ram !== "ram96") {
    flags.push("--dit-layerwise-resident-layers 14");
  }
  if (WORKSTATION_48G.includes(s.hw)) {
    flags.push("--dit-layerwise-resident-layers 40");
  }
  return flags;
}

function unified128Flags() {
  // No flags: the deployment (134 GiB) exceeds the pool, automatic offload
  // engages on its own and pins 42 of 50 DiT layers. Measured on DGX Spark,
  // the explicit discrete-GPU recipe (--performance-mode memory + offload
  // components + video_vae=36) ran the same denoise 2.1x slower (25.8 vs
  // 12.1 s/it) -- do not carry discrete-card flags onto unified memory.
  return [];
}

function workstation96Flags() {
  // 96 GB holds the whole 61.7 GB DiT; only the encoders and VAEs step aside.
  return [
    "--performance-mode memory",
    "--layerwise-offload-components text_encoder,vae",
    "--layerwise-resident-layers video_vae=36",
  ];
}

function consumerHints(s) {
  const hints = [];
  const bigHost = s.host_ram === "ram96";
  const midHost = s.host_ram === "ram64";
  if (bigHost) {
    if (CONSUMER_VRAM_16_PLUS.includes(s.hw)) {
      hints.push(CONSUMER_32G.includes(s.hw)
        ? "with the DiT pinned the denoise runs at this card's compute wall, measured at 5.14-5.17 s per step on a physical 5090; the decode holds all 36 blocks in their fp16 decode dtype and takes ~6.4 s"
        : "verified end to end: ~6 s per denoise step, 13 s decode");
      hints.push("fewer resident layers than the 32 GB rows is not a typo: with the DiT pinned in a big host, streamed layers arrive at pinned-copy speed and GPU residency buys little; on a 32 GB host the stream is the bottleneck residency cuts");
    } else {
      hints.push("~6 s per step once the host pins the DiT; the decode holds all 36 blocks in their fp16 decode dtype and takes ~10 s");
    }
    return hints;
  }
  if (CONSUMER_32G.includes(s.hw)) {
    if (midHost) {
      hints.push("measured end to end on a physical desktop (RTX 5090, 60 GB host, PCIe 5.0 NVMe -- VRAM, host RAM and drive all real): 112.2 / 112.1 s per request at 864x480 / 124 frames / 20 steps, against ComfyUI's 140.9-145.9 s on the same weights and sampler settings. Text encoding 4.2 s, denoise 98.0-98.5 s (5.14-5.17 s/step), decode 6.4 s");
      hints.push("at this host size the pin budget covers 34 of the 50 DiT layers, so only ~2 layers per step and the 46 GiB text encoder come off the drive -- 96 GiB per request, read with O_DIRECT and fully hidden behind compute (the compute thread waited 0.9 s per request)");
    } else {
      hints.push("derived, not yet measured at this host size: the card holds the same fourteen resident layers, but the pin budget covers only ~13 of the 36 streamed layers, so ~28 GiB per step comes off the drive. That stays hidden behind the 5.14 s/step compute wall on a drive delivering ~5.5 GiB/s or better (a PCIe 4.0 NVMe); a PCIe 3.0 drive becomes the clock at ~8.8 s/step");
    }
  } else if (CONSUMER_24G.includes(s.hw)) {
    hints.push("measured at 32 GB host under a 22 GiB cap (desktop headroom): ~8.5 s per denoise step with six resident layers, ~9.6 s decode -- ahead of ComfyUI (249-260 s at the 24 GiB cap); a headless card can raise to ten layers for under 1% more");
  } else if (CONSUMER_16G.includes(s.hw)) {
    hints.push("measured at 32 GB host: ~11.9 s per denoise step, ~11 s decode, ~250 s per request -- ahead of ComfyUI (292-301 s) under the same hard 16 GiB cap");
  } else {
    hints.push("measured at 32 GB host: ~10.6 s per denoise step, ~9.4 s decode, ~235 s per request -- ahead of ComfyUI (276-302 s) on the same weights under the same hard 12 GiB cap, output bit-identical");
  }
  if (CONSUMER_AMPERE.includes(s.hw)) {
    hints.push("the recipe and its memory behavior are tier-exact for this card; the step times above were measured on 40-series compute, and Ampere lands above them");
  }
  if (UNIFIED_128G.includes(s.hw)) {
    return [
      "verified on DGX Spark at 480P: ~12.1 s per denoise step steady-state, ~40 s decode, ~12 min per warm request -- with no flags at all; adding the discrete-GPU offload flags measured 2.1x slower on the same box",
      "the text encoder runs ~5.5 min per request and does not warm up: it is steady-state compute on this chip, not a stall -- budget for it",
      "expect ~12 min of server load before the first request; the first request itself runs at full speed (no JIT tax was measured)",
      "step times sit above the discrete-GPU rows because the GB10's ~273 GB/s memory bandwidth is the denoise ceiling, not the placement",
    ];
  }
  if (WORKSTATION_96G.includes(s.hw)) {
    hints.push("derived recipe, not yet verified: 96 GB holds the whole 61.7 GB DiT resident, so only the text encoder and VAEs stream -- expect near-datacenter step times rather than the offload figures above");
  }
  if (WORKSTATION_48G.includes(s.hw)) {
    hints.push("derived recipe, not yet verified: 48 GB holds forty of the fifty DiT layers; the figures above are the 24 GB tier's and this card should land well under them");
  }
  hints.push("run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- the decode sits close enough to the cap that fragmentation otherwise tips it over");
  if (midHost) {
    hints.push("measured on a 12 GB card at a 48 GB host: ~9.6 s/step, ~218 s per request (ComfyUI 246-267 s); at 64 GB: ~8.1 s/step, ~180 s (ComfyUI 194-195 s); larger cards land at or below these");
  } else {
    hints.push("a 32 GB host cannot cache the 108 GB checkpoint, so the drive is in the loop on every step -- see the rate it has to clear, below. A physical desktop that clears it reaches the quoted step times rather than falling short of them");
  }
  hints.push('the startup log should say "leaving ... GiB of weights on the checkpoint mapping" -- if it does not, the host is not the constraint you set');
  hints.push("every figure here is anchored at 480P: activations grow with the pixel count, so at 768P drop the resident DiT layers to 0 first, then video_vae to 24 if the decode still collides -- the flags trade speed for headroom in that order");
  hints.push("what the drive has to deliver is computable: each step re-reads (50 - resident DiT layers - the layers the host pin budget covered) x 1.23 GiB, and the courier overlaps that read with compute, so the drive stops being the denoise clock once its sequential rate exceeds those bytes divided by this card's step time. A 32 GB host covers ~13 layers, which puts a 24 GB card at ~38 GiB/step and a 12/16 GB card at ~46 GiB/step: a PCIe 4.0 NVMe (6-7 GiB/s) clears both, a PCIe 3.0 drive (~3.2 GiB/s) becomes the clock, and a SATA SSD is not usable for this model");
  hints.push("the 38 s/step a physical 4090 with a 990 Pro once measured (52.9 GB per step at ~1.4 GB/s) was the old path: the pinned-store pool rounded every block up to a power of two, so the pin budget over-committed and the machine had to run with pinning off, and the layers that stayed mapped faulted in 4 KiB at a time. Exact-size pins and O_DIRECT reads through the courier removed both -- on the physical 5090 the drive is now fully hidden behind compute. A 4090 on a 32 GB host is derived to return to its compute wall (~8.5 s/step) on a PCIe 4.0 drive; that machine has not been re-measured");
  hints.push("resident DiT layers still remove 1.23 GiB/step of drive traffic each, but only while the streamed set is larger than the pin budget; once the read fits under the step's compute time, more resident layers buy nothing and only cost VRAM");
  hints.push("warm at the shape you will serve (--warmup-resolutions WxH --warmup-num-frames N): the default warmup is 1344x768 x 124 frames, which cost 7.6 min of startup on the physical 5090 and sizes the residency plan for a request you are not going to make");
  hints.push("the pin budget takes 95% of the memory available after loading, so on a machine you also use as a desktop expect swap pressure: the 5090 run above finished with its 8 GB swap file full. There is no supported knob for a smaller budget yet; giving the card more resident layers is the lever that exists, since a resident layer is one the host no longer has to pin");
  hints.push("on Windows run under WSL2, and keep the checkpoint inside the ext4 side (under ~), never on /mnt/c -- the NTFS bridge reads an order of magnitude slower and multiplies the disk clock");
  return hints;
}

return {
  modelName: "MiniMax-H3",

  supportedHardware: [
    "b200",
    "b300",
    "gb300",
    "gb200",
    "h200",
    "h100",
    "xeon",
    "mi300x",
    "mi355x",
    "rtxpro6000",
    "rtx6000ada",
    "dgx-spark",
    "rtx5090",
    "rtx4090",
    "rtx3090",
    "rtx5080",
    "rtx5070ti",
    "rtx4080",
    "rtx4060ti",
    "rtx5070",
    "rtx4070",
    "rtx3060",
  ],
  hardware: [
    { id: "xeon", label: "XEON", vram: "CPU", vendor: "cpu" },
    { id: "rtxpro6000", label: "RTX PRO 6000", vram: "96GB", vendor: "consumer" },
    { id: "rtx6000ada", label: "RTX 6000 Ada", vram: "48GB", vendor: "consumer" },
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "consumer" },
    { id: "rtx4090", label: "RTX 4090", vram: "24GB", vendor: "consumer" },
    { id: "rtx3090", label: "RTX 3090", vram: "24GB", vendor: "consumer" },
    { id: "rtx5080", label: "RTX 5080", vram: "16GB", vendor: "consumer" },
    { id: "rtx5070ti", label: "RTX 5070 Ti", vram: "16GB", vendor: "consumer" },
    { id: "rtx4080", label: "RTX 4080", vram: "16GB", vendor: "consumer" },
    { id: "rtx4060ti", label: "RTX 4060 Ti", vram: "16GB", vendor: "consumer" },
    { id: "rtx5070", label: "RTX 5070", vram: "12GB", vendor: "consumer" },
    { id: "rtx4070", label: "RTX 4070", vram: "12GB", vendor: "consumer" },
    { id: "rtx3060", label: "RTX 3060", vram: "12GB", vendor: "consumer" },
  ],
  groupHardware: false,

  matchDims: [],

  overlayDims: [
    {
      id: "host_ram",
      title: "Host RAM",
      scope: "serve",
      description: "System memory decides where the DiT weights wait between steps: pinned when they fit, on the checkpoint mapping when they do not.",
      default: "ram32",
      showWhen: (s) =>
        CONSUMER_SINGLE.includes(s.hw) && !UNIFIED_128G.includes(s.hw),
      options: [
        { id: "ram32", label: "32 GB" },
        { id: "ram64", label: "48-64 GB" },
        { id: "ram96", label: "96 GB+" },
      ],
    },
    {
      id: "weights",
      title: "Checkpoint Weights",
      scope: "base",
      description: "Choose the checkpoint partition required by the request mode.",
      default: "fl2va",
      options: [
        {
          id: "fl2va",
          label: "FL2VA",
          subtitle: "(First-and-Last-Frame-to-Video-and-Audio)",
          flags: ["--model-variant fl2va"],
        },
        {
          id: "ref2va",
          label: "Ref2VA",
          subtitle: "(Reference-to-Video-and-Audio)",
          flags: ["--model-variant ref2va"],
        },
      ],
    },
    {
      id: "mode",
      title: "Request Mode",
      scope: "base",
      description: "The visible modes follow the selected checkpoint.",
      default: "t2va",
      options: [
        {
          id: "t2va",
          label: "Text only",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "i2va",
          label: "First frame",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "l2va",
          label: "Last frame",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "fl2va",
          label: "First + last frames",
          showWhen: (s) => s.weights === "fl2va",
        },
        {
          id: "ref_image",
          label: "Image reference",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "ref_image_audio",
          label: "Image + audio",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "v2v",
          label: "Video reference",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "video_audio",
          label: "Video + soundtrack",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "audio_only",
          label: "Audio reference",
          showWhen: (s) => s.weights === "ref2va",
        },
        {
          id: "mixed_ref",
          label: "Mixed references",
          showWhen: (s) => s.weights === "ref2va",
        },
      ],
    },
    {
      id: "placement",
      title: "Placement",
      scope: "serve",
      docsHref: "/docs/sglang-diffusion/api/cli#component-residency",
      description: "Keep weights resident for latency; shard or offload only when capacity requires it.",
      quality: "Memory policy",
      learnMore: "#9-feature-contracts-and-advanced-recipes",
      default: "resident",
      options: [
        {
          id: "auto",
          label: "Auto",
          flags: (s) => {
            if (consumerSingleCard(s)) return consumerFlags(s);
            const recipe = config.commandBuilder.resource.verifiedRecipes.find((entry) =>
              entry.hw === s.hw && entry.nodes === Number(s.nodes)
              && entry.gpus_per_node === Number(s.gpus_per_node));
            const placement = recipe?.placement || (s.hw === "rtx5090" ? "offload" : "resident");
            if (placement === "fsdp") return ["--performance-mode speed", "--use-fsdp-inference true"];
            return placement === "offload" ? [
                "--performance-mode memory",
                "--layerwise-offload-components dit,text_encoder,vae",
                "--dit-layerwise-resident-layers 20",
              ] : ["--performance-mode speed"];
          },
          hints: (s) => (consumerSingleCard(s) ? consumerHints(s) : []),
          description: "Use the recommended placement for the selected hardware and resource shape.",
        },
        {
          id: "resident",
          label: "Resident",
          flags: ["--performance-mode speed"],
          disabled: (s) => CONSUMER_SINGLE.includes(s.hw),
          disableReason: "The 61.7 GB DiT cannot be resident on a single consumer card.",
          recommendedWhen: (s) => s.hw !== "rtx5090" && !CONSUMER_SINGLE.includes(s.hw),
          description: "Lowest-latency path when the full pipeline fits in aggregate GPU memory.",
        },
        {
          id: "fsdp",
          label: "FSDP",
          flags: ["--performance-mode speed", "--use-fsdp-inference true"],
          soft: (s) => !["b200", "b300", "h200", "h100"].includes(s.hw) || s.nodes > 1,
          softReason: "Verified on single-node B200/B300/H200/H100. Other hardware and multi-node runs take the same flags but have not been through a verification round.",
          description: "Reduces resident DiT memory but adds parameter collectives on every block.",
        },
        {
          id: "offload",
          label: "Layerwise offload",
          flags: (s) => {
            if (consumerSingleCard(s)) return consumerFlags(s);
            return [
              "--performance-mode memory",
              "--layerwise-offload-components dit,text_encoder,vae",
              "--dit-layerwise-resident-layers 20",
            ];
          },
          hints: (s) => (consumerSingleCard(s) ? consumerHints(s) : []),
          soft: (s) => s.hw !== "rtx5090" && !CONSUMER_SINGLE.includes(s.hw),
          softReason: "Tuned and verified on the consumer cards. It runs on the datacenter GPUs too, where a resident recipe is simply faster.",
          recommendedWhen: (s) => s.hw === "rtx5090" || CONSUMER_SINGLE.includes(s.hw),
          description: "Capacity-first PCIe path. It is substantially slower than a resident datacenter recipe.",
        },
      ],
    },
    {
      id: "attention",
      title: "Attention",
      scope: "serve",
      docsHref: "/docs/sglang-diffusion/attention_backends",
      description: "Select the packed-attention kernel used by H3 transformer modules.",
      quality: "Kernel policy",
      learnMore: "#9-feature-contracts-and-advanced-recipes",
      default: "platform",
      options: [
        {
          id: "platform",
          label: "Automatic",
          flags: (s) => ["mi300x", "mi355x"].includes(s.hw) ? ["--attention-backend aiter"] : [],
          env: (s) => ["mi300x", "mi355x"].includes(s.hw) ? ["SGLANG_USE_AITER=1"] : [],
          recommended: true,
          description: "Applies the verified backend policy for the selected hardware.",
        },
        {
          id: "fa",
          label: "FlashAttention",
          flags: ["--attention-backend fa"],
          disabled: (s) => ["mi300x", "mi355x"].includes(s.hw),
          disableReason: "Use the verified AITER platform default on AMD.",
          description: "An explicit native-dtype CUDA comparison path; reduction ordering may still differ.",
        },
        {
          id: "sage",
          label: "SageAttention",
          flags: ["--attention-backend sage_attn"],
          disabled: (s) => ["mi300x", "mi355x"].includes(s.hw),
          disableReason: "SageAttention is not exposed for the AMD recipes.",
          description: "Approximate attention math. Install its packed-varlen dependency and inspect video and audio quality.",
        },
      ],
    },
    {
      id: "precision",
      title: "Precision",
      scope: "serve",
      docsHref: "/docs/sglang-diffusion/quantization",
      description: "Choose native mixed precision or a validated online weight quantization path.",
      quality: "Weight precision",
      learnMore: "#9-feature-contracts-and-advanced-recipes",
      default: "native",
      options: [
        {
          id: "native",
          label: "BF16 / FP32",
          recommended: true,
          description: "Reference mixed precision: BF16 transformer weights with required projections retained in FP32.",
        },
        {
          id: "fp8",
          label: "Online FP8",
          flags: ["--quantization fp8"],
          soft: (s) => !["b200", "b300"].includes(s.hw)
            || !["auto", "resident"].includes(s.placement)
            || s.nodes !== 1,
          softReason: "Verified for resident single-node B200/B300. Other hardware and placements take the same flag, but those recipes have not been verified yet.",
          description: "Approximate transformer weight quantization with the required H3 projections protected.",
        },
      ],
    },
    {
      id: "encoder",
      title: "Encoder",
      scope: "serve",
      docsHref: "/docs/sglang-diffusion/encoder_parallel",
      description: "Control text-encoder work placement independently from DiT topology.",
      quality: "Parallel policy",
      learnMore: "#9-feature-contracts-and-advanced-recipes",
      default: "auto",
      options: [
        {
          id: "auto",
          label: "Auto",
          flags: (s) => s.hw === "xeon" ? ["--encoder-parallel auto"] : (s.nodes > 1 ? ["--encoder-parallel replicate"] : []),
          recommended: true,
          description: "Folds on verified single-host P2P systems and resolves to replicate across nodes.",
        },
        {
          id: "dp",
          label: "Data parallel",
          flags: ["--encoder-parallel dp"],
          disabled: (s) => s.hw === "xeon" || CONSUMER_SINGLE.includes(s.hw) || (s.topology_mode === "manual"
            ? Number(s.tp_size)
            : config.commandBuilder.resource.autoTopology(s).tp_size) > 1,
          disableReason: "Encoder DP requires TP1 and a multi-GPU DP group; TP > 1 and the single-card consumer recipes do not qualify.",
          soft: (s) => s.nodes > 1,
          softReason: "Runs across nodes, but the measured 1.9× encode speedup comes from a single-node 2× H100 run; cross-node encoder DP is unverified.",
          description: "Useful for a real request batch; it is not bitwise-identical to fold scheduling.",
        },
        {
          id: "fold",
          label: "Fold",
          flags: ["--encoder-parallel fold"],
          disabled: (s) => s.hw === "xeon" || s.nodes > 1,
          disableReason: "Fold assumes fast node-local peer-to-peer access.",
          description: "Uses one folded encoder copy across a node-local group and preserves native weights.",
        },
        {
          id: "replicate",
          label: "Replicate",
          flags: ["--encoder-parallel replicate"],
          disabled: (s) => s.hw === "xeon",
          disableReason: "Xeon always runs the auto encoder policy.",
          recommendedWhen: (s) => s.nodes > 1,
          description: "The safe cross-node default because encoder auto is not node-boundary aware.",
        },
      ],
    },
    {
      id: "execution",
      title: "Execution",
      scope: "serve",
      description: "Choose eager execution or the measured breakable CUDA graph path.",
      quality: "Graph policy",
      learnMore: "#9-feature-contracts-and-advanced-recipes",
      default: "eager",
      options: [
        {
          id: "eager",
          label: "Eager",
          recommended: true,
          description: "Reference execution and the consistency baseline.",
        },
        {
          id: "bcg",
          label: "Compatible BCG",
          flags: [
            "--enable-breakable-cuda-graph true",
            "--warmup-resolutions 1344x768",
            "--bcg-text-buckets 5504",
          ],
          soft: (s) => !(["b200", "h200"].includes(s.hw) && s.weights === "ref2va"),
          softReason: "Verified for B200/H200 Ref2VA; BCG runs on the other recipes but they have not been through a verification round yet.",
          description: "Reuses matching execution signatures and reserves capture memory; it takes precedence over Cache-DiT.",
        },
      ],
    },
    {
      id: "quality",
      title: "Quality",
      scope: "request",
      docsHref: "/docs/sglang-diffusion/cache_dit",
      description: "Cumulative reference, fusion-only, or audited Cache-DiT execution.",
      quality: "Sampling policy",
      learnMore: "#choose-the-quality-level",
      default: "exact",
      options: [
        {
          id: "exact",
          label: "Exact",
          recommended: true,
          description: "Reference-exact denoising without Cache-DiT approximation.",
        },
        {
          id: "lossless",
          label: "Lossless",
          description: "Includes fusion-only request paths but not Cache-DiT; MiniMax-H3 currently follows its exact denoise path at this tier.",
        },
        {
          id: "high",
          label: "Audited high",
          disabled: (s) => s.execution !== "eager",
          disableReason: "BCG supersedes Cache-DiT, so the preset would have no effect — switch Execution to Eager to use it.",
          soft: (s) => !(s.hw === "h200" && s.nodes === 1 && s.gpus_per_node === 4
            && ["auto", "resident"].includes(s.placement)),
          softReason: "The 1.99× / SSIM 0.911 audit covers the resident eager 4× H200 workload; elsewhere the preset runs but its quality figures are unaudited.",
          description: "Measured 1.99× with SSIM 0.911 and PSNR 29.09 dB against lossless on the audited workload.",
        },
      ],
    },
    {
      id: "outputs",
      title: "Outputs",
      scope: "request",
      description: "Generate independent variants from one request.",
      quality: "1–10",
      kind: "number",
      min: 1,
      max: 10,
      unit: "outputs per prompt",
      default: 1,
      options: [],
    },
    {
      id: "reference_resize_mode",
      title: "Reference image resize",
      scope: "request",
      showWhen: (s) => ["ref_image", "ref_image_audio", "mixed_ref"].includes(s.mode),
      description: "Resize Ref2VA image references before encoding; audio, video, and FL2VA keyframes are unaffected.",
      quality: "Preprocessing policy",
      default: "max",
      options: [
        {
          id: "max",
          label: "Preserve reference resolution",
          recommended: true,
          description: "Keep smaller inputs at source scale and downscale only when the short edge exceeds 2048 pixels.",
        },
        {
          id: "diffusers",
          label: "Diffusers-compatible",
          description: "Reproduce the original preprocessing by scaling the reference short edge to 2048 pixels, including upscaling.",
        },
        {
          id: "match",
          label: "Match target area",
          description: "Match the target canvas pixel area without upscaling; required by ModelTC Ref2VA Turbo.",
        },
      ],
    },
  ],

  commandBuilder: {
    defaultSelection: {
      hw: "b200",
      nodes: 1,
      gpus_per_node: 8,
      topology_mode: "auto",
      tp_size: 1,
      ulysses_degree: 8,
      ring_degree: 1,
    },
    resource: {
      limits: {
        nodes: { min: 1, max: 8, allowedValues: (s) => s.hw === "xeon" ? [1, 2, 4, 8] : [] },
        gpus_per_node: { min: 1, max: 8, disabledWhen: (s) => s.hw === "xeon", disabledValue: 0 },
      },
      verifiedRecipes: [
        { id: "b200-resident-8", hw: "b200", nodes: 1, gpus_per_node: 8, placement: "resident", tp_size: 1, ulysses_degree: 8, ring_degree: 1, encoder: "auto", default: true },
        // Auto takes the first entry for a GPU count. 4x B200 holds the DiT
        // resident, which beats FSDP by 4-7% (FL2VA 209f 64.9 vs 67.6 s,
        // T2VA 124f 36.0 vs 38.7 s); FSDP stays available as a placement.
        { id: "b200-resident-4", hw: "b200", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "b200-fsdp-4", hw: "b200", nodes: 1, gpus_per_node: 4, placement: "fsdp", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "b300-resident-8", hw: "b300", nodes: 1, gpus_per_node: 8, placement: "resident", tp_size: 1, ulysses_degree: 8, ring_degree: 1, encoder: "auto", default: true },
        { id: "b300-fsdp-8", hw: "b300", nodes: 1, gpus_per_node: 8, placement: "fsdp", tp_size: 1, ulysses_degree: 8, ring_degree: 1, encoder: "auto" },
        { id: "gb300-resident-4", hw: "gb300", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto", default: true },
        { id: "gb200-resident-4", hw: "gb200", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto", default: true, unverified: true },
        { id: "h200-resident-4", hw: "h200", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto", default: true },
        { id: "h200-fsdp-4", hw: "h200", nodes: 1, gpus_per_node: 4, placement: "fsdp", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "h200-cross-node-16", hw: "h200", nodes: 2, gpus_per_node: 8, placement: "resident", tp_size: 1, ulysses_degree: 8, ring_degree: 2, encoder: "replicate" },
        { id: "h100-resident-4", hw: "h100", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 2, ulysses_degree: 2, ring_degree: 1, encoder: "auto", default: true },
        { id: "h100-fsdp-4", hw: "h100", nodes: 1, gpus_per_node: 4, placement: "fsdp", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "mi300x-resident-1", hw: "mi300x", nodes: 1, gpus_per_node: 1, placement: "resident", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto" },
        { id: "mi300x-resident-2", hw: "mi300x", nodes: 1, gpus_per_node: 2, placement: "resident", tp_size: 1, ulysses_degree: 2, ring_degree: 1, encoder: "auto" },
        { id: "mi300x-resident-4", hw: "mi300x", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "mi300x-resident-8", hw: "mi300x", nodes: 1, gpus_per_node: 8, placement: "resident", tp_size: 1, ulysses_degree: 8, ring_degree: 1, encoder: "auto", default: true },
        { id: "mi355x-resident-1", hw: "mi355x", nodes: 1, gpus_per_node: 1, placement: "resident", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto" },
        { id: "mi355x-resident-2", hw: "mi355x", nodes: 1, gpus_per_node: 2, placement: "resident", tp_size: 1, ulysses_degree: 2, ring_degree: 1, encoder: "auto" },
        { id: "mi355x-resident-4", hw: "mi355x", nodes: 1, gpus_per_node: 4, placement: "resident", tp_size: 1, ulysses_degree: 4, ring_degree: 1, encoder: "auto" },
        { id: "mi355x-resident-8", hw: "mi355x", nodes: 1, gpus_per_node: 8, placement: "resident", tp_size: 1, ulysses_degree: 8, ring_degree: 1, encoder: "auto", default: true },
        { id: "dgx-spark-offload-1", hw: "dgx-spark", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true, unverified: true },
        { id: "rtxpro6000-offload-1", hw: "rtxpro6000", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true, unverified: true },
        { id: "rtx6000ada-offload-1", hw: "rtx6000ada", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true, unverified: true },
        { id: "rtx5090-offload-1", hw: "rtx5090", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
        { id: "rtx5090-offload-2", hw: "rtx5090", nodes: 1, gpus_per_node: 2, placement: "offload", tp_size: 2, ulysses_degree: 1, ring_degree: 1, encoder: "auto" },
        { id: "rtx4090-offload-1", hw: "rtx4090", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
        { id: "rtx4080-offload-1", hw: "rtx4080", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
        { id: "rtx3090-offload-1", hw: "rtx3090", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
        { id: "rtx5080-offload-1", hw: "rtx5080", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
        { id: "rtx5070ti-offload-1", hw: "rtx5070ti", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
        { id: "rtx4060ti-offload-1", hw: "rtx4060ti", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
        { id: "rtx5070-offload-1", hw: "rtx5070", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
        { id: "rtx3060-offload-1", hw: "rtx3060", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
        { id: "rtx4070-offload-1", hw: "rtx4070", nodes: 1, gpus_per_node: 1, placement: "offload", tp_size: 1, ulysses_degree: 1, ring_degree: 1, encoder: "auto", default: true },
      ],
      autoTopology: (s) => {
        const recipes = config.commandBuilder.resource.verifiedRecipes;
        const exact = recipes.find((recipe) => recipe.hw === s.hw
          && recipe.nodes === Number(s.nodes)
          && recipe.gpus_per_node === Number(s.gpus_per_node)
          && (s.placement === "auto" || recipe.placement === s.placement));
        if (exact) {
          return {
            tp_size: exact.tp_size,
            ulysses_degree: exact.ulysses_degree,
            ring_degree: exact.ring_degree,
          };
        }
        return {
          tp_size: 1,
          ulysses_degree: Number(s.gpus_per_node),
          ring_degree: Number(s.nodes),
        };
      },
      validateTopology: (s, topology) => {
        const errors = [];
        const nodes = Number(s.nodes);
        const perNode = Number(s.gpus_per_node);
        const world = nodes * perNode;
        const tp = Number(topology.tp_size);
        const ulysses = Number(topology.ulysses_degree);
        const ring = Number(topology.ring_degree);
        if (!Number.isInteger(nodes) || nodes < 1 || nodes > 8) errors.push("H3 supports 1–8 nodes.");
        if (!Number.isInteger(perNode) || perNode < 1 || perNode > 8) errors.push("H3 supports 1–8 GPUs per node.");
        if (![1, 2, 4, 8].includes(tp)) errors.push("Tensor parallel size must be one of 1, 2, 4, or 8.");
        if (world !== tp * ulysses * ring) errors.push(`World size ${world} must equal TP × Ulysses × Ring (${tp * ulysses * ring}).`);
        if (56 % tp !== 0 || (56 / tp) % ulysses !== 0) errors.push("H3's 56 attention heads must divide evenly across TP and Ulysses.");
        if (64 % (ulysses * ring) !== 0) errors.push("Ulysses × Ring must divide the 64 packed sequence partitions.");
        return errors;
      },
    },
    resolveDeployment: (s) => {
      if (s.hw === "xeon") {
        return {
          match: { hw: s.hw },
          nnodes: 1,
          verified: false,
          verificationStatus: "unverified",
          env: ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cpu"],
          flags: [
            `--model-path MiniMaxAI/MiniMax-H3 \\\n  --tp-size ${s.nodes}`,
            "--host {{HOST_IP}}",
            "--port {{PORT}}",
          ],
          builder: {
            topologySummary: "CPU",
            errors: [],
            warnings: ["Xeon CPU deployment is supported."],
            verification: { serve: "unverified", request: "unverified" },
            resolvedSettings: {},
          },
        };
      }
      const resource = config.commandBuilder.resource;
      const topology = s.topology_mode === "manual"
        ? {
            tp_size: Number(s.tp_size),
            ulysses_degree: Number(s.ulysses_degree),
            ring_degree: Number(s.ring_degree),
          }
        : resource.autoTopology(s);
      const errors = resource.validateTopology(s, topology);
      const automaticRecipe = resource.verifiedRecipes.find((entry) => entry.hw === s.hw
        && entry.nodes === Number(s.nodes)
        && entry.gpus_per_node === Number(s.gpus_per_node)
        && entry.tp_size === topology.tp_size
        && entry.ulysses_degree === topology.ulysses_degree
        && entry.ring_degree === topology.ring_degree);
      const resolvedPlacement = s.placement === "auto"
        ? (automaticRecipe?.placement
          || (s.hw === "rtx5090" || CONSUMER_SINGLE.includes(s.hw) ? "offload" : "resident"))
        : s.placement;
      const coverageWarnings = [];
      if (resolvedPlacement === "offload" && s.hw !== "rtx5090"
        && !CONSUMER_SINGLE.includes(s.hw)) {
        coverageWarnings.push("Layerwise offload is tuned and verified on consumer cards; on this hardware it runs unverified and a resident recipe is faster.");
      }
      if (CONSUMER_SINGLE.includes(s.hw) && s.host_ram === "ram64") {
        coverageWarnings.push("48-64 GB hosts sit between the measured 32 GB and 96 GB points and have not been through their own verification round.");
      }
      if (resolvedPlacement === "fsdp" && (s.nodes !== 1 || !["b200", "b300", "h200", "h100"].includes(s.hw))) {
        coverageWarnings.push("FSDP outside the single-node NVIDIA recipes runs unverified.");
      }
      if (s.precision === "fp8" && (! ["b200", "b300"].includes(s.hw)
        || resolvedPlacement !== "resident" || s.nodes !== 1)) {
        coverageWarnings.push("Online FP8 outside resident single-node B200/B300 runs unverified.");
      }
      const highAudited = s.hw === "h200" && s.nodes === 1
        && s.gpus_per_node === 4 && resolvedPlacement === "resident";
      if (s.quality === "high" && s.execution !== "eager") {
        coverageWarnings.push("BCG supersedes Cache-DiT, so the high preset has no effect under this execution mode.");
      } else if (s.quality === "high" && !highAudited) {
        coverageWarnings.push("The high preset's 1.99× / SSIM 0.911 figures were audited on resident eager 4× H200; this workload is unaudited.");
      }

      const recipe = resource.verifiedRecipes.find((entry) => entry.hw === s.hw
        && entry.nodes === Number(s.nodes)
        && entry.gpus_per_node === Number(s.gpus_per_node)
        && entry.placement === resolvedPlacement
        && entry.tp_size === topology.tp_size
        && entry.ulysses_degree === topology.ulysses_degree
        && entry.ring_degree === topology.ring_degree);
      const topologyVerified = !!recipe && !recipe.unverified && errors.length === 0;
      const encoderVerified = s.encoder === "auto"
        || s.encoder === recipe?.encoder
        || (s.nodes > 1 && s.encoder === "replicate");
      const attentionVerified = s.attention === "platform";
      const precisionVerified = s.precision === "native"
        || (s.precision === "fp8" && ["b200", "b300"].includes(s.hw));
      const executionVerified = s.execution === "eager"
        || (s.execution === "bcg" && ["b200", "h200"].includes(s.hw) && s.weights === "ref2va");
      const checkpointVerified = s.hw !== "gb300" || s.weights === "fl2va";
      const serveVerified = topologyVerified && encoderVerified && attentionVerified
        && precisionVerified && executionVerified && checkpointVerified;
      const requestCovered = s.hw !== "gb300" || (serveVerified && s.weights === "fl2va"
        && s.mode === "t2va" && s.quality === "exact" && Number(s.outputs) === 1);
      const requestVerified = topologyVerified && requestCovered && (["exact", "lossless"].includes(s.quality)
        || (s.quality === "high" && highAudited && s.execution === "eager"));

      const topologyParts = [];
      if (topology.tp_size > 1) topologyParts.push(`TP ${topology.tp_size}`);
      if (Number(s.nodes) > 1) {
        topologyParts.push(`Ulysses ${topology.ulysses_degree} inside each node`);
        topologyParts.push(`Ring ${topology.ring_degree} across nodes`);
      } else {
        topologyParts.push(`Ulysses ${topology.ulysses_degree}`);
      }
      topologyParts.push({ resident: "Resident", fsdp: "FSDP", offload: "Layerwise offload" }[resolvedPlacement]);
      topologyParts.push(Number(s.nodes) > 1 ? `${s.nodes} nodes` : "Single node");

      const world = Number(s.nodes) * Number(s.gpus_per_node);
      const flags = ["--model-path {{MODEL_NAME}}"];
      if (world > 1) flags.push(`--num-gpus ${world}`);
      if (topology.ring_degree > 1) flags.push(`--sp-degree ${world}`);
      if (topology.tp_size > 1) flags.push(`--tp-size ${topology.tp_size}`);
      if (topology.ulysses_degree > 1) flags.push(`--ulysses-degree ${topology.ulysses_degree}`);
      if (topology.ring_degree > 1) flags.push(`--ring-degree ${topology.ring_degree}`);
      flags.push("--host {{HOST_IP}}", "--port {{PORT}}");

      const warnings = [...coverageWarnings];
      if (!topologyVerified && errors.length === 0) {
        warnings.push("This topology satisfies H3's static constraints but has not completed an exact end-to-end verification run.");
      }
      if (resolvedPlacement === "fsdp") {
        warnings.push("FSDP lowers resident DiT memory but adds per-block parameter collectives; prefer Resident when the pipeline fits.");
      }
      if (s.hw === "rtx5090" && Number(s.gpus_per_node) === 2) {
        warnings.push("The 2× RTX 5090 path requires a 384 GiB-class host and prioritizes capacity over latency.");
      }

      let automaticAttention = "FlashAttention (auto)";
      if (["mi300x", "mi355x"].includes(s.hw)) {
        automaticAttention = "AITER (auto)";
      } else if (topology.ring_degree === 1 && ["b200", "b300", "gb200", "gb300"].includes(s.hw)) {
        automaticAttention = "Dynamic cuDNN / FA (auto)";
      } else if (topology.ring_degree === 1 && ["rtx5090", "rtx4090"].includes(s.hw)) {
        automaticAttention = "Torch SDPA (auto)";
      }

      return {
        match: { hw: s.hw },
        nnodes: Number(s.nodes),
        verified: serveVerified,
        verificationStatus: serveVerified ? "verified" : "unverified",
        flags,
        builder: {
          topology,
          topologySummary: topologyParts.filter(Boolean).join(" · "),
          errors,
          warnings,
          verification: {
            serve: errors.length ? "error" : (serveVerified ? "verified" : "unverified"),
            request: errors.length ? "error" : (requestVerified ? "verified" : "unverified"),
          },
          resolvedSettings: {
            placement: { resident: "Resident", fsdp: "FSDP", offload: "Layerwise offload" }[resolvedPlacement],
            attention: s.attention === "platform" ? automaticAttention : undefined,
            encoder: s.encoder === "auto" ? (s.nodes > 1 ? "Replicate (auto)" : "Auto") : undefined,
          },
        },
      };
    },
  },

  modelNames: {
    default: "MiniMaxAI/MiniMax-H3",
  },

  placeholders: {
    HOST_IP: {
      target: "command",
      label: "Bind host",
      default: "0.0.0.0",
    },
    PORT: {
      target: "command",
      label: "Bind port",
      default: "30010",
    },
    HF_TOKEN: {
      target: "command",
      label: "HF token (Docker)",
      default: "<your-hf-token>",
    },
    MEDIA_DIR: {
      target: "command",
      label: "Host media directory (Docker)",
      default: "/data/minimax-h3",
    },
    CURL_HOST: {
      target: "curl",
      label: "Server host",
      default: "localhost",
    },
    CURL_PORT: {
      target: "curl",
      label: "Server port",
      default: "30010",
    },
    DURATION_SECONDS: {
      target: "curl",
      label: "Duration (seconds, 4-15)",
      default: "5",
    },
    FIRST_FRAME: {
      target: "curl",
      label: "FL2VA first frame URI",
      default: "file:///data/minimax-h3/first-frame.png",
    },
    LAST_FRAME: {
      target: "curl",
      label: "FL2VA last frame URI",
      default: "file:///data/minimax-h3/last-frame.png",
    },
    INPUT_VIDEO: {
      target: "curl",
      label: "First video URI",
      default: "file:///data/minimax-h3/video-1.mp4",
    },
    INPUT_VIDEO_START_SECONDS: {
      target: "curl",
      label: "First video start (seconds)",
      default: "0",
    },
    SECOND_INPUT_VIDEO: {
      target: "curl",
      label: "Second video URI (mixed ref)",
      default: "file:///data/minimax-h3/video-2.mp4",
    },
    SECOND_INPUT_VIDEO_START_SECONDS: {
      target: "curl",
      label: "Second video start (seconds)",
      default: "0",
    },
    REFERENCE_IMAGE: {
      target: "curl",
      label: "First reference image URI",
      default: "file:///data/minimax-h3/reference-1.png",
    },
    SECOND_REFERENCE_IMAGE: {
      target: "curl",
      label: "Second reference image URI",
      default: "file:///data/minimax-h3/reference-2.png",
    },
    REFERENCE_AUDIO: {
      target: "curl",
      label: "First reference audio URI",
      default: "file:///data/minimax-h3/reference-1.mp3",
    },
    SECOND_REFERENCE_AUDIO: {
      target: "curl",
      label: "Second reference audio URI",
      default: "file:///data/minimax-h3/reference-2.mp3",
    },
  },

  curl: (s) => {
    const request = {
      model: "{{MODEL_NAME}}",
      prompt:
        "Night-vision bedroom footage: while the owner sleeps, three cats burst in playing tiny brass instruments at full volume, freeze, then march out as if nothing happened.",
      seconds: "{{DURATION_SECONDS}}",
      task: "t2va",
      conditions: [],
      target: {
        short_edge: 768,
        aspect_ratio: "16:9",
        duration_seconds: "{{DURATION_SECONDS}}",
      },
      quality: s.quality,
      num_outputs_per_prompt: Number(s.outputs),
      num_inference_steps: 50,
      flow_shift: 12.0,
      audio_flow_shift: 3.0,
      seed: 1101,
    };
    const imageReference = (uri) => ({
      type: "image",
      uri,
      role: "reference",
    });
    const audioReference = (uri) => ({
      type: "audio",
      uri,
      role: "reference",
    });
    const videoReference = (uri, start, type = "video") => ({
      type,
      uri,
      role: "reference",
      start_time_seconds: start,
    });

    if (["i2va", "l2va", "fl2va"].includes(s.mode)) {
      request.task = "fl2va";
      request.prompt =
        "Continue naturally between the supplied endpoint frame or frames, with synchronized ambient sound.";
      request.target.aspect_ratio = "auto";
      request.seed = 2101;
      request.conditions = [];
      if (s.mode !== "l2va") {
        request.conditions.push({
          type: "image",
          uri: "{{FIRST_FRAME}}",
          role: "keyframe",
          frame_index: 0,
        });
      }
      if (s.mode !== "i2va") {
        request.conditions.push({
          type: "image",
          uri: "{{LAST_FRAME}}",
          role: "keyframe",
          frame_index: -1,
        });
      }
    } else if (s.mode === "ref_image") {
      request.task = "ref2va";
      request.prompt = "Use <Picture 1> as the visual subject and style reference.";
      request.target.aspect_ratio = "auto";
      request.conditions = [imageReference("{{REFERENCE_IMAGE}}")];
      request.seed = 3101;
    } else if (s.mode === "ref_image_audio") {
      request.task = "ref2va";
      request.prompt =
        "Use <Picture 1> as the visual subject and <Audio 1> as the sound reference.";
      request.target.aspect_ratio = "auto";
      request.conditions = [
        imageReference("{{REFERENCE_IMAGE}}"),
        audioReference("{{REFERENCE_AUDIO}}"),
      ];
      request.seed = 3102;
    } else if (s.mode === "v2v" || s.mode === "video_audio") {
      request.task = "ref2va";
      request.prompt =
        s.mode === "video_audio"
          ? "Follow <Video 1> and its required <Audio 1> soundtrack with coherent synchronized motion."
          : "Follow the appearance and motion of <Video 1>; use its soundtrack when present.";
      request.conditions = [
        videoReference(
          "{{INPUT_VIDEO}}",
          "{{INPUT_VIDEO_START_SECONDS}}",
          s.mode === "video_audio" ? "video_audio" : "video",
        ),
      ];
      request.seed = s.mode === "video_audio" ? 4102 : 4101;
    } else if (s.mode === "audio_only") {
      request.task = "ref2va";
      request.prompt = "Build a coherent visual scene around <Audio 1>.";
      request.conditions = [audioReference("{{REFERENCE_AUDIO}}")];
      request.seed = 3103;
    } else if (s.mode === "mixed_ref") {
      request.task = "ref2va";
      request.prompt =
        "Combine <Picture 1>, <Picture 2>, <Audio 1>, <Audio 2>, <Video 1>, and <Video 2> in their one-based modality order.";
      request.conditions = [
        imageReference("{{REFERENCE_IMAGE}}"),
        imageReference("{{SECOND_REFERENCE_IMAGE}}"),
        audioReference("{{REFERENCE_AUDIO}}"),
        audioReference("{{SECOND_REFERENCE_AUDIO}}"),
        videoReference("{{INPUT_VIDEO}}", "{{INPUT_VIDEO_START_SECONDS}}"),
        videoReference(
          "{{SECOND_INPUT_VIDEO}}",
          "{{SECOND_INPUT_VIDEO_START_SECONDS}}",
        ),
      ];
      request.seed = 3104;
    }

    if (["ref_image", "ref_image_audio", "mixed_ref"].includes(s.mode)) {
      request.reference_resize_mode = s.reference_resize_mode;
    }

    const body = JSON.stringify(request, null, 2).replace(
      /"{{(DURATION_SECONDS|INPUT_VIDEO_START_SECONDS|SECOND_INPUT_VIDEO_START_SECONDS)}}"/g,
      "{{$1}}",
    );
    return `curl -sS -X POST http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos \\
  -H 'Content-Type: application/json' \\
  -d '${body}'`;
  },

  dockerMounts: ["{{MEDIA_DIR}}:/data/minimax-h3:ro"],

  // Publish Docker only after the platform's H3 image/command has been validated.
  runModes: (s) =>
    ["mi300x", "mi355x", "gb200", "gb300", "xeon"].includes(s.hw)
      ? ["python"]
      : ["docker", "python"],

  dockerImages: {
    b200: "lmsysorg/sglang:dev",
    b300: "lmsysorg/sglang:dev",
    h200: "lmsysorg/sglang:dev",
    h100: "lmsysorg/sglang:dev",
  },

  showPlaygroundLink: false,

  cells: [],
};
})();

```


---

## FILE: docs/cookbook/diffusion/MiniMax/MiniMax-H3-Image.mdx (https://github.com/sgl-project/sglang/blob/HEAD/docs/cookbook/diffusion/MiniMax/MiniMax-H3-Image.mdx)

---
title: MiniMax-H3-Image
description: "Run native six-step MiniMax-H3-Image generation and image editing with SGLang Diffusion."
tag: NEW
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/MiniMaxAI/minimax-h3-image.jsx";

<DiffusionModelTags tags={["text to image", "image editing", "multiple references", "6-step PDD", "BF16"]} />

## 1. Quick start

Use the [SGLang main branch](https://github.com/sgl-project/sglang/tree/main)
with the [diffusion installation guide](/docs/sglang-diffusion/installation),
then install its diffusion dependencies from the repository root:

```bash
python3 -m pip install -e 'python[diffusion]'
```

The public checkpoints are `MiniMaxAI/MiniMax-H3-Image` and its
`MiniMaxAI/MiniMax-H3` text encoder.
Run the commands from the same checkout and Python environment. The picker
saves the returned image as `generated.png`. These recipes use Linux CUDA and
a source installation; DGX Spark uses the [ARM64 environment below](#dgx-spark-128-gb),
and AMD Instinct GPUs use the [ROCm environment below](#amd-instinct-mi300x-mi325x-mi350x-mi355x).

<Deployment config={config} />

For the lowest measured latency, see the [recipe summary](#measured-text-to-image-summary):
two B300s generate a 4K image in 2.654 s at native precision or 2.065 s with
opt-in MXFP8. These are warm pipeline times; quantization can change image quality.

Server warmup runs a valid prefix of the PDD schedule; real requests always
run all six steps. With online AdaLN and enough cache slots, warmup prepares
the full grids for both declared T2I and editing tasks without extra denoising
steps. Smaller custom caches prepare only the warmup task. New shapes can
still initialize kernels. When every plan fits on the GPU, Auto omits the
pinned host cache; an explicit `--minimax-h3-adaln-host-cache-gb` overrides this.
`--warmup-mode off` skips this preparation. For CLI/API options, see the
[CLI reference](/docs/sglang-diffusion/api/cli) and
[OpenAI API guide](/docs/sglang-diffusion/api/openai_api).

## 2. Model capabilities

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) uses one
native checkpoint for text-to-image generation and image editing. An edit's
multipart `image` fields preserve reference order; repeat the field to supply
multiple references. The prompt refers to them as `<Picture 1>`, `<Picture 2>`,
and so on. The root checkpoint selects the image pipeline automatically;
`--model-variant` is not needed.

The DiT has about 33.1 billion parameters. Its separate Qwen3-VL encoder
checkpoint has about 33.4 billion; downloading both checkpoints and the VAE
requires about 135 GB. The pipeline loads only the encoder layers needed by
this release. Runtime memory also depends on placement, projection reuse and
the generation canvas.

Prompt text uses no chat template or system prompt and is truncated after
4096 tokens. Use a complete visual description for generation or editing.

The checkpoint has six fused PDD output heads and an explicit sigma
grid. Omit `num_inference_steps`, or set it to 6. It runs one positive branch:
omit guidance parameters and `negative_prompt`. It returns one image per
request. Audio and video inputs and outputs are not supported.

| Task | Endpoint | Input |
| --- | --- | --- |
| Text to image | `/v1/images/generations` | JSON prompt |
| Image editing | `/v1/images/edits` | Multipart prompt and ordered image files |

### Resolution and references

Recommended output sizes are:

| Aspect ratio | Landscape | Portrait |
| --- | --- | --- |
| 1:1 | 4096 × 4096 | — |
| 4:3 | 4736 × 3552 | 3552 × 4736 |
| 3:2 | 5024 × 3360 | 3360 × 5024 |
| 16:9 | 5472 × 3072 | 3072 × 5472 |
| 21:9 | 6272 × 2688 | — |

The default minimum generation area is 2048 × 2048 pixels. Smaller requests
are generated on a larger canvas with the same aspect ratio, aligned to 32
pixels, then resized to the requested output size with LANCZOS. A 1024 × 1024
output therefore has the generation cost of a 2048 × 2048 canvas.

`min_generation_area` is a request-level integer pixel area between 1024 and
16777216. Reducing it changes the generation workload and potentially image
quality; it is not a lossless acceleration of the default recipe. Set
`min_generation_area` to `null` in a pipeline configuration file to disable
canvas enlargement; output dimensions must then be multiples of 32.

Edit references are oriented using EXIF and resized separately for each
encoder. Qwen uses a 1024-pixel short edge; the VAE keeps the short edge within
1024–2048 pixels. Reference order, dimensions, generator device, model revision
and precision all matter when reproducing an edit.

## 3. Deployment options

DiT and text-encoder weights use BF16. The VAE checkpoint uses FP32; its
decoder weights are prepared in BF16 for BF16 autocast. The DiT uses dense
FlashAttention. Keep these precision and attention settings when reproducing
the release. ROCm has no FA3 build, so on AMD GPUs the DiT runs the same dense
packed attention through AITER. It is selected automatically and `fa` is
rejected there. Ring parallelism requires FA; use Ulysses or TP on AMD.

On discrete GPUs, Auto keeps all components resident when each GPU has at least
130 GiB available at startup, as tested on H200, B300, MI325X and MI350X. Otherwise it retains
component offload. GB10 uses the shared-memory policy in the Spark recipe below.
Explicit placement overrides take precedence. Resident mode requires
space for all weights and request activations; sequence parallelism still
replicates weights. TP shards the DiT projections. Ulysses and Ring split the
packed sequence; choose degrees compatible with the 56 attention heads.
The picker marks unmeasured hardware and option combinations as unverified.
For two H200s or B300s, select encoder fold. Ulysses2 has the
lowest measured server pipeline latency; TP2 is also close and uses less memory.

On H200, B300, GB10 and RTX PRO 6000 Blackwell with standard weights, Auto precomputes the native AdaLN timestep
projections and releases their weights. This retains the original precision
and outputs, reducing measured peak reserved memory by about 24.6–24.7 GiB on one
GPU. Text-to-image and editing use separate plans;
keep the checkpoint files available for uncached plans. The bounded GPU and
host caches reuse these exact results. Select **Keep weights** in the picker,
or pass `--minimax-h3-adaln-online false`, to avoid the rebuild cost. Other
GPUs keep projection weights by default; explicit online precomputation is
available through **Precompute**.

`quality=lossless` is the default. It enables shared DiT and VAE QK-norm/RoPE
fusion, uses cuDNN SDPA in the VAE with a backend fallback, and batches existing
overlapping VAE decode tiles, with at most 128 samples per tile batch.
With multiple GPUs, complete tiles run across the decode group before assembly;
tile geometry and overlap stay unchanged. Select **Local tiles** in the picker
or pass `--vae-config.use-parallel-tiling false` to run all tiles on each GPU.
`quality=exact` restores the release's fusion and sequential tile path. Neither
changes the six-step schedule or canvas. Floating-point reordering can change
pixels; lossless does not mean byte-identical output across paths or topologies.
Text refinement and rotary tables are prepared once per request for both paths.
The shared conditioning cache defaults to 512 MiB per worker. Matching prompts
and ordered reference content can reuse text conditioning; matching VAE inputs
can reuse posterior moments, including with multiple GPUs. Sampling still uses
the request's generator. Pass `--disable-conditioning-cache` to disable reuse
across requests.
These recipes retain the release checkout's library numerics defaults, including
cuDNN TF32 and BF16 reduced-precision reduction. The stricter overrides
`--allow-cudnn-tf32 false --allow-bf16-reduced-precision-reduction false` are
available but are not covered by these measurements.

**Layerwise offload** streams the DiT, text encoder and VAE decoder blocks from
host RAM while preserving native precision. Use **Precompute** with it to avoid
reloading the large AdaLN projection weights. This is the capacity option for
smaller GPUs; resident serving remains faster when the model fits.

**Online FP8 / MXFP8** quantizes the DiT, optionally with the text encoder, from
the original checkpoint. Select it separately in the picker; native BF16 remains
the default. These are approximate modes even with `quality=exact` or `lossless`:
quality controls kernel execution within the selected precision. Quantization
can change composition and reference identity, so compare representative edits
before choosing it. AdaLN
precomputation and the VAE retain native precision. MXFP8 uses block scaling on
Blackwell and the FP8 fallback on Hopper; use FP8 on older supported GPUs.

**FP8 attention** separately quantizes attention inputs on SM120/SM121 GPUs.
Select it explicitly in the picker, or pass
`--component-attention-backends.transformer fp8_fa_sm120`. Native FlashAttention
remains the default. The measured Spark combination uses **MXFP8 DiT + encoder**
and **FP8 attention** together; both change numerical precision. Ring parallelism
requires native FlashAttention.

**Breakable CUDA graph** captures the picker's task, size and reference layout
during startup. Its warmup prompt matches the generated request; editing also
needs the reference files on the server. Matching signatures replay the graph,
while other signatures run eager. **Regional compile** compiles repeated DiT
blocks with PyTorch. Both keep all six steps; compile can change floating-point
rounding and increases cold startup time. Graphs also work with layerwise
offload; capture retains device buffers and needs more GPU memory than eager
streaming. Eager remains the default.

### Hardware recommendations

The picker provides measured recipes for H200/B300, AMD Instinct, single-card RTX PRO 6000 and DGX Spark,
and **Derived** recipes for other listed GPUs. A derived recipe estimates placement and
parallel layout for the selected GPU count from checkpoint sizes and measured
H200 memory use. It has no hardware latency measurement or capacity guarantee.
Apply **Use derived recipe** to set its placement, encoder fold and online
AdaLN together. Start with 2048 × 2048.

| GPU class | Default shape | Recommended placement and layout |
| --- | --- | --- |
| B200, GB200, GB300 | 1 GPU | Auto resident; Ulysses as GPU count grows |
| MI355X / MI350X, MI325X / MI300X | 1 GPU; Ulysses as GPU count grows | Measured Auto resident recipes on 1, 2, 4 and 8 GPUs; Ulysses is faster, TP uses less memory |
| H100 / H800, 80 GB | 1 GPU, or 2 for resident serving | Component offload on one; resident Ulysses2 with encoder fold on two |
| RTX PRO 6000 Blackwell, 96 GB | 1 GPU | Measured resident recipe with AdaLN precomputation; component offload saves memory |
| DGX Spark, 128 GB shared | 1 GB10 | Measured Auto resident recipe with native AdaLN precomputation; requires at least 106 GiB available |
| RTX 6000 Ada, RTX 5090 / 4090 and listed 12–48 GB cards without NVLink | 1 GPU | Layerwise offload with native AdaLN precomputation |
| RTX 3090, 24 GB | 1 GPU; optionally a [two-card NVLink pair](https://www.nvidia.com/content/geforce-gtx/GEFORCE_RTX_3090_USER_GUIDE_v02.pdf) | Layerwise offload with native AdaLN precomputation |

Multi-GPU recommendations assume an NVLink-connected machine. Cards without
NVLink use single-card recipes by default. Offload recipes assume a 256 GB-class
host and local NVMe. Layerwise offload preserves the six steps, native precision
and minimum generation canvas. The capacity validation uses a constrained B300
allocator: native 2K/4K generation and editing, including two-reference 4K,
passed with an 8 GiB allocator limit and a 10.01 GiB measured whole-GPU peak.
It is not a latency measurement on the listed consumer GPUs.

### AMD Instinct (MI300X, MI325X, MI350X, MI355X)

Select an AMD GPU in the picker for single-node recipes on 1, 2, 4 and 8 GPUs.
MI355X and MI350X (gfx950) share one recipe; MI325X and MI300X (gfx942) share
another. The recipes were run on MI350X and MI325X. No attention flag is
needed: the DiT selects AITER on ROCm. Auto placement keeps every component
resident, and AdaLN projections keep their weights. Ring is not available; use
Ulysses or TP. At each GPU count, Ulysses has the lowest measured latency and
TP the lowest memory per GPU.

Install this checkout over the SGLang in an SGLang ROCm image for your GPU:

| GPU | Tested image | Python, PyTorch |
| --- | --- | --- |
| MI355X, MI350X | `lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261109` | 3.10, 2.9.1+rocm7.2.0 |
| MI325X, MI300X | `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi30x-20261107` | 3.12, 2.11.0+rocm10.0.0 |

```bash
docker run -it --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -v "$PWD":/workspace/sglang -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e HF_TOKEN \
  lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261109 bash

# Inside the container, from the SGLang checkout:
cd /workspace/sglang
python3 -m pip install -e python --no-deps --no-build-isolation
# On the Python 3.10 MI35x image, add --ignore-requires-python.
```

The image already provides the ROCm builds of PyTorch, AITER and the
diffusion dependencies, so `--no-deps` keeps them in place. Then run the
picker's serving and request commands in this environment.

### DGX Spark (128 GB)

Select **DGX Spark** for a verified single-node, single-GB10 recipe at 2K and the measured 4K sizes. Use an
Arm64 CUDA environment with GB10 support before installing this checkout.
At `539fea720b86` and later, Auto enables native AdaLN precomputation and keeps
components resident when at least 106 GiB is available in the shared pool.
The picker therefore needs no placement or AdaLN override. Resident GB10
rejects **Keep weights**, which would retain the large projection tensors.
Start with a 2048 × 2048 output.

The [128 GB configuration](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
has one [shared CPU/GPU memory pool](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#guidance-for-reporting-memory-resources-with-unified-memory-architecture).
Component offload does not provide another 128 GB of memory. Whole-component transfers can temporarily hold both CPU and GPU copies in
that pool. Use a bounded
container with swap disabled for the process, such as `--memory 104g
--memory-swap 104g --shm-size 8g`, and download weights inside that container
so its file cache can be reclaimed within the limit. The picker marks option
combinations outside its measured scope **Unverified**.
The container limit does not cap all CUDA allocations; keep at least 106 GiB
available in the shared pool before starting the verified Auto recipe.

The tested ARM64 base is `lmsysorg/sglang@sha256:d4a56f6f9356539957974bb4008161b4c08d14a712805ff63eae526f36dbda71`,
with driver 580.173.02, PyTorch 2.14.1+cu130 and FlashInfer 0.7.3.
Mount this checkout and a task-owned cache directory, then install the source
and matching diffusion dependencies inside the container:

```bash
H3_IMAGE_CACHE="$HOME/.cache/h3-image-spark"
umask 077
mkdir -p "$H3_IMAGE_CACHE"
docker run --platform linux/arm64 --gpus all \
  --memory 104g --memory-swap 104g --shm-size 8g --cpus 12 \
  -v "$PWD":/workspace/sglang -v "$H3_IMAGE_CACHE":/root/.cache/huggingface \
  -e HF_TOKEN -it \
  lmsysorg/sglang@sha256:d4a56f6f9356539957974bb4008161b4c08d14a712805ff63eae526f36dbda71 bash

# Inside the container, from the SGLang checkout:
cd /workspace/sglang
export CARGO_TARGET_DIR=/tmp/h3-image-spark-build
python3 -m pip install setuptools-rust setuptools-scm wheel
python3 -m pip install --no-build-isolation -e 'python[diffusion]'
python3 -m pip uninstall -y flashinfer-jit-cache-sm80 flashinfer-jit-cache-sm89 \
  flashinfer-jit-cache-sm90a flashinfer-jit-cache-sm100a flashinfer-jit-cache-sm103a \
  flashinfer-jit-cache-sm110a flashinfer-jit-cache-sm120f flashinfer-jit-cache-sm121a
python3 -m pip install --no-deps flash-attn-4==4.0.0b38
python3 -m pip install --no-deps flashinfer-cubin==0.7.3 \
  --index-url https://flashinfer.ai/whl/
python3 -m pip install --no-deps flashinfer-jit-cache==0.7.3+cu130 \
  --index-url https://flashinfer.ai/whl/cu130
```

This replaces the base image's incompatible optional kernel providers while
retaining FlashInfer's version checks. Download weights and run the picker's
serving/request commands in this environment. The measured container uses 12
CPU cores; the host's 128 GB is a shared pool, not 128 GB of additional VRAM.

## 4. Measured recipes

### Measured text-to-image summary

The native rows keep the checkpoint's release precision. FP8 and MXFP8 quantize
the DiT and text encoder; they can change image details and reference identity.
Here, 2K is 2048 × 2048 and 4K is 4736 × 3552.

| Hardware | Precision | 2K | 4K |
| --- | --- | --- | --- |
| 2 × B300 | Native | 0.604 s | 2.654 s |
| 2 × B300 | MXFP8 | 0.451 s | 2.065 s |
| 2 × H200 | Native | 1.123 s | 5.465 s |
| 1 × H200 | Native | 1.934 s | 10.133 s |
| 1 × RTX PRO 6000 Blackwell | Native | 3.434 s | 18.544 s |
| 1 × RTX PRO 6000 Blackwell | FP8 | 2.338 s | 14.445 s |
| 1 × DGX Spark | Native | 16.21 s | 84.96 s |
| 8 × MI350X | Native | 0.490 s | 2.110 s |
| 4 × MI325X | Native | 0.890 s | 3.680 s |
| 1 × MI325X | Native | 2.190 s | 10.750 s |

In the picker, use **Ulysses2** and encoder **Fold** for two GPUs. Choose **Auto**
placement for native two-GPU, Spark and AMD recipes, or **Resident** for the other rows.
The MI350X row uses TP8 and the four-MI325X row Ulysses4; MI355X and MI300X use
the MI350X and MI325X recipes.
Select **FP8 DiT + encoder** or **MXFP8 DiT + encoder** only for the quantized rows.

All rows run the full six-step schedule with lossless quality, eager execution,
native AdaLN precomputation and warm conditioning caches. Times are medians of
warm server pipeline requests, excluding startup, HTTP transport, PNG encoding
and client saving; they are not first-request or end-to-end latency.
See the sections below for source revisions, checkpoints, sample counts and
environments.

### DGX Spark at 2K and 4K

One GB10 with Auto resident placement and native AdaLN precomputation, at
`539fea720b86`, using the pinned checkpoints and the same ARM64 base with
FlashInfer 0.7.2:

| Workload | Pipeline median | Observed warm range |
| --- | --- | --- |
| T2I, 2048 × 2048 | 16.21 s | 15.65–16.80 s |
| Editing, 2048 × 2048 | 25.05 s | 24.58–25.25 s |
| T2I, 4736 × 3552 | 84.96 s | 81.51–87.97 s |
| Editing, 4096 × 4096 | 98.83 s | 93.73–105.62 s |
| T2I, 6272 × 2688 | 81.04 s | 79.36–85.95 s |

Each cell uses 10 warm lossless requests across two fresh servers, excluding the
first lossless request per task in each. All six steps run at native precision;
conditioning is warm. T2I uses seed 42; 2K editing uses seed 42 and 4K editing
uses seed 1234, with one reference. These are pipeline times, excluding HTTP,
PNG encoding and client saving. Maximum CUDA reserved memory over the validated
workflows was about 91 GiB; other runtime and host allocations also share the
same pool. The native offline API also passed 2K generation and editing at
exact and lossless quality with matching decoded pixels.

Startup takes several minutes in the 12-core container, even with cached
checkpoints. Keep the server running to reuse weight loading and warmup; both
are excluded from the warm-request table.

### DGX Spark with MXFP8 and FP8 attention

Select **MXFP8 DiT + encoder** and **FP8 attention** for the accelerated Spark
recipe. Native precision remains the default; these options can change image
composition and details even with `quality=lossless`.

| Workload | Pipeline median | Observed warm range |
| --- | --- | --- |
| T2I, 2048 × 2048 | 10.99 s | 10.86–11.14 s |
| T2I, 4736 × 3552 | 56.31 s | 54.85–56.55 s |
| Editing, 4096 × 4096 | 62.78 s | 62.75–62.82 s |

Measured at `7e9078906472` and `78567c4042a0` with FlashInfer 0.7.3 and the ARM64
environment above. 4K generation has five measured requests across two fresh
servers (three plus two); the other rows have three each. The first request per
task in each server is excluded as warmup. All six steps
run with resident components, eager execution and warm conditioning. T2I uses
seed 42; editing uses seed 1234 and one 1024 × 1024 reference. Times exclude
HTTP, PNG encoding and client saving. Peak CUDA reserved memory is 51.32 GiB
in the shared pool. The native baseline at `1b51a11129bb` on the same environment measured
82.45 s for 4736 × 3552 (81.73–83.18 s, two warm requests).

### B300 at 2K

Unquantized checkpoints at native precision, 2048 × 2048 generation canvas,
six steps, seed 42 and a CPU generator; one node of B300 GPUs. The image revision is
`7b3d20b4659f5d4b3ad7205daa008e47d6b1e868`, with H3 text encoder revision
`42ed227ee7df40d41602854ae760620d6eb651fe`. Source installation was tested at
`7ccff196eb3a` for the single-GPU row and `539fea720b86` for all two-GPU rows,
with PyTorch 2.14.1+cu130, FlashInfer 0.7.2 and Transformers 5.19.0.
The two-GPU comparison uses an NV18-connected pair, parallel VAE tiles and warm
exact conditioning reuse.

| GPUs and layout | T2I pipeline | Editing pipeline |
| --- | --- | --- |
| 1 × B300, Auto resident | 0.994 s | 1.570 s |
| 2 × B300, TP2, encoder fold | 0.621 s | 0.978 s |
| 2 × B300, Ulysses2, encoder fold | 0.604 s | 0.931 s |
| 2 × B300, Ring2, encoder fold | 0.692 s | 1.093 s |

All rows use lossless quality, resident placement and native AdaLN
precomputation. The single-GPU row is the median of six warm requests per
task across two server lifecycles, after two excluded requests per task in each.
Each two-GPU row uses 10 warm requests across two fresh servers, excluding the
first lossless request per task in each. Layout order is balanced across the
six server starts. Cold plan construction is excluded. All use the checkpoint's
example prompts and one edit reference. These are server-reported pipeline
latencies, excluding HTTP transport, PNG encoding and client saving. They are
not throughput measurements or results for other GPUs. H100 and B200 recipes
remain unmeasured.

### B300 with MXFP8

Select **MXFP8 DiT + encoder**, **Resident**, two B300s and Ulysses2 with
encoder fold. Online quantization changes numerics; native precision remains
the default. At `ff5dede83b0f`, a balanced native → MXFP8 → MXFP8 → native
comparison measured:

| Workload | Native pipeline | MXFP8 pipeline | Reduction |
| --- | --- | --- | --- |
| T2I, 2048 × 2048 | 0.604 s | 0.451 s | 25.4% |
| T2I, 4736 × 3552 | 2.653 s | 2.065 s | 22.2% |
| Editing, 4096 × 4096 | 3.146 s | 2.490 s | 20.9% |

Each cell has 10 warm requests across two fresh servers, excluding the first
lossless request per task in each. Both paths use all six steps, warm exact
conditioning reuse, native AdaLN precomputation and parallel VAE tiles on the
same NV18 pair, checkpoints and dependencies. T2I uses seed 42; editing uses
seed 1234 and one reference. Times exclude HTTP, PNG encoding and client saving.
Maximum server-reported CUDA reserved memory across these workloads fell from
69.2 GiB to 40.0 GiB; this is not total host or combined multi-GPU memory.

MXFP8 DiT + encoder also passed 2K/4K generation, editing and ordered multi-image
requests on one DGX Spark. Choose that weight option in the Spark picker;
AdaLN and VAE keep native precision.

### B300 at 4K

Two B300s at `539fea720b86`, with encoder fold, Auto resident placement and
native AdaLN precomputation, using the same pinned checkpoints and dependencies:

| Layout | T2I, 4736 × 3552 | Editing, 4096 × 4096 | T2I, 6272 × 2688 |
| --- | --- | --- | --- |
| TP2 | 2.790 s | 3.314 s | 2.795 s |
| Ulysses2 | 2.654 s | 3.146 s | 2.662 s |
| Ring2 | 3.032 s | 3.568 s | 3.026 s |

Each cell uses 10 warm lossless requests across two fresh servers, excluding the
first lossless request per task in each. The same balanced layout comparison
covers 2K and 4K. All six steps run at native precision; conditioning is warm.
T2I uses seed 42; editing uses seed 1234 and one reference. These are pipeline
times, excluding HTTP, PNG encoding and client saving. Ulysses2 is the fastest
measured layout for these workloads; TP2 lowers maximum reserved memory across
ranks and workflow phases to about 49.6 GiB, versus 69.2 GiB for Ulysses2 and
69.6 GiB for Ring2. These allocator measurements exclude additional CUDA process
memory.

### RTX PRO 6000 Blackwell, 96 GB

On one RTX PRO 6000 Blackwell Server Edition, resident placement is faster than
component offload for the tested workloads. Both use native precision, FA4,
all six steps and AdaLN precomputation; select **Resident** in the picker.

| Workload | Resident pipeline | Component-offload pipeline |
| --- | --- | --- |
| T2I, 2048 × 2048 | 3.434 s | 5.094 s |
| Editing, 2048 × 2048 | 5.603 s | 7.259 s |
| T2I, 4736 × 3552 | 18.544 s | 20.193 s |
| Editing, 4096 × 4096 | 21.971 s | 23.615 s |
| T2I, 6272 × 2688 | 18.634 s | 20.281 s |

Source `a9d721f36e16`, PyTorch 2.14.1+cu130, FlashInfer 0.7.2 and FA4
4.0.0b35; the same pinned checkpoints and example inputs as above. Each cell
uses 10 warm requests across two fresh servers in a resident/offload ABBA
comparison, excluding the first lossless request per task in each. Conditioning
is warm; HTTP transport, PNG encoding and client saving are excluded.
Maximum reserved memory over the 4K workflows was about 92.2 GiB resident or
66.1 GiB with offload, before additional CUDA process memory. Resident has
limited headroom for other workloads; Auto retains conservative component
offload on this 96 GB device. The picker provides the single-card recipe for
this GPU, which has no NVLink.

For a smaller working set, select **FP8 DiT + encoder** with **Resident**.
At `5c7ce652b563`, a separate native → FP8 → FP8 → native comparison on the
same single GPU measured:

| Workload | Native pipeline | FP8 pipeline |
| --- | --- | --- |
| T2I, 2048 × 2048 | 3.432 s | 2.338 s |
| T2I, 4736 × 3552 | 18.557 s | 14.445 s |
| Editing, 4096 × 4096 | 21.988 s | 17.336 s |

Each cell has 10 warm requests across two fresh servers, excluding the first
lossless request per task in each. All six steps, warm conditioning, example
inputs, seeds and the pipeline timing boundary match the B300 quantization
comparison. Maximum CUDA reserved memory across these workloads fell from
91.6 GiB to 50.4 GiB. FP8 is approximate and remains an explicit option.

### H200 at 2K

H200 measurements use the same checkpoints, prompts, precision, six-step
schedule and canvas. Single-GPU T2I uses `851f8cb71469` with explicit resident
placement and online AdaLN; these become Auto defaults at `9979ece836fb`.
Single-GPU editing uses `be38e34be99d`; Ulysses2 uses `862194eeaedb` with
parallel VAE tiles. Both use Auto placement and exact conditioning reuse.
TP2 and Ring2 use `9979ece836fb`.
All two-GPU rows use encoder fold.

| GPUs and layout | T2I pipeline | Editing pipeline |
| --- | --- | --- |
| 1 × H200, resident | 1.934 s | 3.116 s |
| 2 × H200, TP2, encoder fold | 1.169 s | 2.041 s |
| 2 × H200, Ulysses2, encoder fold | 1.123 s | 1.774 s |
| 2 × H200, Ring2, encoder fold | 1.224 s | 2.146 s |

Single-GPU T2I uses 20 warm requests across four fresh servers, excluding the
first request in each. Single-GPU editing uses eight across two servers,
excluding the first two in each. Each two-GPU row uses 10 across two servers,
excluding the first request per task in each. Peak reserved memory was about
90 GiB on one H200 and, per GPU, 48 GiB for TP2 or 67 GiB for Ulysses2.
These are latency measurements, not throughput results.

For the single-H200 editing workload, the picker defaults to exact conditioning
reuse. Disabling it with `--disable-conditioning-cache` increased the warm
pipeline median from 3.116 s to 3.446 s. Both settings use eight measured
requests across two fresh servers in an ABBA comparison, with the first two
requests excluded per server. All 24 output PNGs matched byte for byte.
Changing the prompt or references can miss the cache; these are repeated-input
measurements, not uncached editing latency.

### H200 at 4K

The Ulysses2 row uses `862194eeaedb` with parallel VAE tiles and exact conditioning reuse; the other
rows use `f1af1685f6c3`. All use the same checkpoint revisions and dependencies
above, native precision, lossless quality and all six steps.
Auto placement and AdaLN precomputation are enabled; two-GPU rows use encoder
fold. Each cell is the median of 10 warm requests across two fresh servers,
excluding the first lossless request for each task in each server.

| GPUs and layout | T2I, 4736 × 3552 | Editing, 4096 × 4096 | T2I, 6272 × 2688 |
| --- | --- | --- | --- |
| 1 × H200 | 10.133 s | 11.839 s | 9.948 s |
| 2 × H200, TP2 | 5.850 s | 6.992 s | 5.730 s |
| 2 × H200, Ulysses2 | 5.465 s | 6.379 s | 5.402 s |
| 2 × H200, Ring2 | 6.023 s | 7.145 s | 5.954 s |

Conditioning caches are warm. T2I uses seed 42; editing uses seed 1234 and
one reference image. These are server pipeline times, excluding HTTP transfer,
PNG encoding and client saving. Ulysses2 gives the lowest measured latency;
TP2 uses less memory. Measured peak reserved memory over these workflows was
about 93 GiB on one H200, or a maximum across ranks of 50 GiB for TP2,
70 GiB for Ulysses2 and 70 GiB for Ring2. These allocator measurements do not
include all CUDA process memory. Other 4K sizes, multiple-reference 4K edits
remain unmeasured on H200.

### AMD Instinct at 2K and 4K

MI325X (gfx942) and MI350X (gfx950), one node each, in the images listed
[above](#amd-instinct-mi300x-mi325x-mi350x-mi355x), at `2dcc93b624` with the
ROCm attention selection described in [Deployment options](#3-deployment-options).
The image revision is `7b3d20b4659f5d4b3ad7205daa008e47d6b1e868`, with
text-encoder weights identical to revision `42ed227ee7df40d41602854ae760620d6eb651fe`.
The server ran plain `sglang serve` with startup warmup, Auto placement and no
attention flag; requests used lossless quality.

| GPUs and layout | T2I, 4736 × 3552 | Editing, 4096 × 4096 | T2I, 6272 × 2688 | T2I, 2048 × 2048 | Editing, 2048 × 2048 | Peak GiB/GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 1 × MI325X | 10.750 s | 12.700 s | 10.730 s | 2.190 s | 3.360 s | 117 |
| 2 × MI325X, TP2 | 8.680 s | 10.060 s | 8.690 s | 1.890 s | 2.940 s | 62 |
| 2 × MI325X, Ulysses2 | 7.570 s | 8.460 s | 7.640 s | 1.710 s | 2.460 s | 93 |
| 4 × MI325X, TP4 | 4.960 s | 5.620 s | 4.960 s | 1.080 s | 1.600 s | 33 |
| 4 × MI325X, Ulysses4 | 3.680 s | 4.620 s | 3.610 s | 0.890 s | 1.350 s | 80 |
| 8 × MI350X, TP8 | 2.110 s | 2.290 s | 2.050 s | 0.490 s | 0.740 s | 19 |

Each cell is the median of five warm requests on one server, excluding the
first request per task. T2I uses seed 42; editing uses seed 1234 and one
reference image, with warm conditioning caches. These
are server pipeline times, excluding HTTP transfer, PNG encoding and client
saving. All 180 requests completed without errors. The MI350X GPUs were shared
with an idle process, so that server passed `--component-residency all=resident`
to keep the resident placement Auto selects on a free 288 GB GPU.

The 1-GPU, TP2, Ulysses2, Ulysses8 and TP8 recipes on both GPUs also served the
three 4K workloads without errors at the earlier day-0 revision `e1ab33200` with
the same attention change. Their latency at `2dcc93b624` is not yet measured.
The guide's T2I and editing examples match the release reference images
visually; outputs are not byte-identical to CUDA or across layouts. MI355X and
MI300X use the same gfx950 and gfx942 recipes and were not measured separately.
Multiple-reference, exact-quality and quantized AMD workloads remain unmeasured.

### PNG encoding at 4K

Select **No compression** under Request options, or send `output_compression=0`,
to trade response size for less PNG encoding and decoding work. The default
compression stays unchanged. On two-H200 and two-B300 Ulysses2 recipes at
`862194eeaedb`, all decoded pixels matched for both formats:

| GPUs | Workload | Default PNG, saved-request time | Uncompressed PNG, saved-request time | PNG bytes, default → uncompressed |
| --- | --- | --- | --- | --- |
| 2 × H200 | T2I, 4736 × 3552 | 8.140 s | 7.211 s | 31.7 → 50.5 MB |
| 2 × H200 | Editing, 4096 × 4096 | 8.838 s | 8.258 s | 22.7 → 50.4 MB |
| 2 × H200 | T2I, 6272 × 2688 | 8.009 s | 7.159 s | 31.0 → 50.6 MB |
| 2 × B300 | T2I, 4736 × 3552 | 5.259 s | 4.367 s | 31.7 → 50.5 MB |
| 2 × B300 | Editing, 4096 × 4096 | 5.577 s | 5.012 s | 22.8 → 50.4 MB |
| 2 × B300 | T2I, 6272 × 2688 | 5.225 s | 4.436 s | 31.1 → 50.6 MB |

Each format uses eight measured requests per workload across two fresh servers,
with balanced request order and one excluded warmup per format in each server.
Timing runs from the loopback HTTP request through PNG saving and pixel validation.
Pipeline time stayed approximately unchanged; these are codec/client-path savings.
Larger payloads may take longer over a slower network. Both formats retain native
precision, all six steps and the same generation canvas.

### File URL delivery

**Base64** remains the default and uses temporary output files. Select **File URL**
under Setup to add `--output-path` to the server command and save each response
through a separate image download. Restart the server after switching. Files
remain in the selected output directory until you remove them; the URL records
last only for that server process.

On two NVLink-connected B300s with Ulysses2 at `539fea720b86`, File URL shortened
the measured save path further when using **No compression**:

| Workload | Base64, uncompressed PNG | File URL, uncompressed PNG |
| --- | --- | --- |
| T2I, 4736 × 3552 | 4.380 s | 3.990 s |
| Editing, 4096 × 4096 | 4.982 s | 4.620 s |
| T2I, 6272 × 2688 | 4.397 s | 4.028 s |

Both columns use the same persistent output directory, native precision, six
steps and warm conditioning. Each cell uses eight measured requests across two
fresh servers, excluding the first request per format/server, with balanced
format and compression order. Timing includes the POST, image download, PNG
saving and pixel validation. Default and uncompressed PNG both retained identical
decoded pixels across response formats. Pipeline latency stayed unchanged.
The extra HTTP request and larger PNGs can cost more on a slower connection.
Other hardware and workloads remain unverified for File URL delivery.

### B300 startup warmup

Peak reserved CUDA memory was about 91 GiB on one B300. These results include
native fusion, VAE batching and exact projection reuse, without reducing steps
or generation area.

One-step server warmup at `f3c0a6f09c6b` prepares both task grids when the
online cache holds at least 12 plans and omits the unnecessary host copy.
Compared with `ea0cba698a08`, on the
same cached checkpoints and one resident B300:

| First-request measurement | Before | Current default |
| --- | --- | --- |
| First editing request, pipeline | 4.041 s | 2.065 s |
| Process start to first editing HTTP response | 38.3 s | 35.4 s |
| Process start to first T2I HTTP response | 36.4 s | 34.0 s |
| Worker CPU RSS after first editing request | 14.52 GiB | 6.52 GiB |

Each first-request scenario uses four fresh servers per path. Both-task
precomputation moves work into startup; avoiding the unused host tier saves
about 8 GiB of worker RSS and removes its allocation cost. The warm-request
table above uses a different timing boundary.
