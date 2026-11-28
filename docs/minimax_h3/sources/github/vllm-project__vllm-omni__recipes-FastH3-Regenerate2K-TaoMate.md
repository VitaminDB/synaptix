# vllm-project__vllm-omni__recipes-FastH3-Regenerate2K-TaoMate

Source: https://github.com/vllm-project/vllm-omni/tree/main/recipes/FastVideo
Fetched: 2026-11-28



---

## FILE: recipes/FastVideo/FastH3.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/FastVideo/FastH3.md)

# FastH3 — four-step Preview V1

[FastH3 family](README.md) · [FastH3 V2](FastH3-V2.md) · [FastH3 Omni](FastH3-Omni.md)

## Summary

- Task: T2VA, text to synchronized video and audio.
- Release: a load-time adapter for `MiniMaxAI/MiniMax-H3`.
- Platform: CUDA with a compatible dense or tile64 VSA backend.
- Mode: existing `/v1/videos` API.

## Hardware and backend selection

The adapter is not tied to a GPU architecture. Select a backend for the
adapter variant and the installed hardware:

| Variant | Backend | Hardware requirements |
| --- | --- | --- |
| Dense | `FLASH_ATTN`, or another H3-compatible dense backend | Follow that backend's GPU and build requirements |
| VSA | `FASTVIDEO_VSA`, BF16 | FastVideo tile64 on compatible CUDA GPUs; automatic FlashInfer selection on SM120/SM121 when available |

The command below uses eight GPUs. GPU memory, interconnect, and output size
determine suitable parallel settings.

## Model and installation

[FastH3](https://haoailab.com/blogs/fasth3-preview/) generates video and audio
in four steps using an adapter for MiniMax-H3. Load one adapter variant when
starting the server.

Install the matching vLLM/vLLM-Omni source environment using the
[CUDA installation guide](../../docs/getting_started/installation/gpu.md).
`ffmpeg` and `ffprobe` are required for video output. Authenticate with
`hf auth login` when the base model or adapter requires access.

Download the Dense / Data-Free adapter and pass its file path to `--lora-path`:

```bash
export FASTH3_DIR=/path/to/fasth3
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors --local-dir "${FASTH3_DIR}"
export FASTH3_LORA="${FASTH3_DIR}/dense-datafree/adapter_model.safetensors"
```

## Serve the Dense / Data-Free adapter

```bash
export FASTH3_DENSE_BACKEND=FLASH_ATTN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve MiniMaxAI/MiniMax-H3 --omni \
  --host 127.0.0.1 --port 8094 --trust-remote-code \
  --task-type fl2va --served-model-name FastH3 \
  --lora-path "${FASTH3_LORA}" \
  --num-gpus 8 --tensor-parallel-size 1 --usp 8 --ring 1 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-config.default.backend "${FASTH3_DENSE_BACKEND}"
```

T2VA uses H3's FL2VA partition, so `--task-type fl2va` is intentional.
The adapter is fused at load time; `--lora-backend` and per-request `lora=`
selection do not apply.

## Generate video and audio

```bash
curl --fail-with-body -sS http://127.0.0.1:8094/v1/videos/sync \
  -F 'model=FastH3' \
  -F 'prompt=A red fox runs through fresh snow at dawn, with pawsteps, winter wind and distant birds.' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F 'num_inference_steps=4' -F 'guidance_scale=1' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":4.4}' \
  -o fasth3.mp4

ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json fasth3.mp4
```

Use `num_inference_steps=4` and `task=t2va`. Keep the default noise schedule
and flow shifts. Choose the adapter at startup; per-request adapter
switching and CPU/layer offload are unsupported.

## VSA adapter variants

For a VSA adapter variant, download that variant's adapter file, install the
`vllm-omni[vsa]` extra, and replace the dense attention backend in the serve
command with these flags. The provider is selected for the actual GPU;
SM90 and SM100/SM103 use FastVideo's tile64 route, while compatible SM120/SM121
installations use FlashInfer BF16. The video request API is unchanged:

```bash
--diffusion-attention-config.default.backend FASTVIDEO_VSA \
--diffusion-attention-config.default.fastvideo_vsa_topk 64
```

See
[provider selection](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md#automatic-provider-selection)
for hardware requirements and provider overrides.

For multi-GPU VSA, use Ulysses sequence parallelism. Ring and all-gather
sequence parallelism are unsupported. The Dense / Data-Free variant uses
a dense attention backend and does not require the VSA extra.


---

## FILE: recipes/FastVideo/FastH3-V2.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/FastVideo/FastH3-V2.md)

# FastH3 V2

[FastH3 family](README.md) · [FastH3 Omni](FastH3-Omni.md)

Generate video and audio from a text prompt with
[FastH3 V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2).
No reference image is required. This eight-GPU deployment is experimental.
For reference-image generation, use [FastH3 Omni](FastH3-Omni.md).

## Install

Follow the [CUDA installation guide](../../docs/getting_started/installation/gpu.md)
and install `ffmpeg`. From the vLLM-Omni repository root:

```bash
uv pip install -e '.[cudnn-bsa]'
```

If Hugging Face requests access to MiniMax-H3 components, accept the model's
access terms and run `hf auth login` once.

## Start the server

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastVideo-FastH3-8-Step-V2 --omni \
  --config recipes/FastVideo/fasth3-8step-8gpu.yaml
```

The configuration handles eight-GPU placement, eight inference steps, and
video decoding. Wait for `Application startup complete` before sending a
request.

## Generate video and audio

In a second terminal:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=A red fox runs through snow at dawn, with pawsteps, wind and distant birds.' \
  -F 'size=1344x768' -F 'seconds=5' \
  -o output.mp4
```

Open `output.mp4` to play the video with audio. For a smaller output, use
`size=832x480`. Keep the configuration's sampling defaults; this model is
trained for eight steps.


---

## FILE: recipes/FastVideo/FastH3-Omni.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/FastVideo/FastH3-Omni.md)

# FastH3 Omni

[FastH3 family](README.md) · [Single-card guide](<FastH3 RefOmni Single Card.md>)

Run [FastH3 Omni](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1)
on eight GPUs to generate video and audio from a prompt and reference image.
For RTX 4090, RTX 5090, DGX Spark, or RTX PRO 6000 Blackwell single-card
serving, use the [single-card guide](<FastH3 RefOmni Single Card.md>).

## Install

Follow the [CUDA installation guide](../../docs/getting_started/installation/gpu.md)
and install `ffmpeg`. From the vLLM-Omni repository root, install the
attention kernels used by this configuration:

```bash
uv pip install -e '.[cudnn-bsa]'
```

If Hugging Face requests access, accept the model's access terms and run
`hf auth login` once. The server downloads the required model components.

## Start the server

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
  --config recipes/FastVideo/fasth3-8step-8gpu.yaml
```

The configuration handles eight-GPU placement, eight inference steps, and
video decoding. Wait for `Application startup complete` before sending a
request. To change the default port, add `--port 8001`.

## Request-level LoRA adapters

For a distributed-layerwise-offload deployment, add `--lora-backend peft` to enable ordinary H3 adapters, including Character Swap. Keep the checkpoint's eight-step sampling settings and disable Turbo adapters.

In ComfyUI, chain existing LoRA nodes through their `lora` input and set each node's strength independently. The HTTP API accepts the corresponding `lora` list; see [stacking multiple LoRAs](../../docs/user_guide/diffusion/lora.md#peft-backend-stacking-multiple-loras-in-one-request). This support applies to FastH3 Omni, not FastH3 Preview or V2.

## Generate video and audio

Save your reference image as `reference.png` and run in a second terminal:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=The person in the reference image smiles and waves at the camera. Birds chirp in the background.' \
  -F 'input_reference=@reference.png' \
  -F 'size=1344x768' -F 'seconds=5' \
  -o output.mp4
```

Open `output.mp4` to play the video with stereo audio. A five-second request
produces 124 frames at 24 fps. For a smaller output, use `size=832x480`.
For multiple images, video, or audio references, see the
[H3 reference API](../MiniMaxAI/MiniMax-H3.md).

## Editing, continuation, timeline guides and ControlNet

FastH3 Omni serves Ref2VA only, so every request below sends `"task":"ref2va"`
and at least one reference image (a `continue_from` request may omit it). The
eight-step schedule, `guidance_scale=1` and the flow shifts stay pinned.

- **Latent-mask editing:** send `source_video` and `video_noise_mask` as in
  [latent-mask editing](../../docs/serving/videos_api.md#minimax-h3-latent-mask-editing).
- **Continuation:** use `continue_from` with a completed job's id
  ([continuation across requests](../../docs/serving/videos_api.md#minimax-h3-continuation-across-requests)),
  or `"long_video":true` for in-request latent-tail continuation. As with the base
  Ref2VA checkpoint, the hidden tail is followed loosely: expect a mild jump at the
  join, and later windows may drift back toward the reference composition.
- **Timeline guides:** send `timeline_guides` and `guide_files` as in
  [timeline guides](../../docs/serving/videos_api.md#minimax-h3-timeline-guides),
  with `quality=lossless`.

### ControlNet

The [Fun ControlNet Union](../MiniMaxAI/MiniMax-H3.md#fun-controlnet-union-ctrl-01)
branch runs on FastH3 Omni with tensor parallelism instead of Ulysses and HSDP.
Copy `fasth3-8step-8gpu.yaml`, set `usp: 1`, delete `use-hsdp: true`, and add
`tensor-parallel-size: 8`, then start the server with
`--controlnet-model-path /path/to/MiniMax-H3-Fun-Controlnet-Union.safetensors`.
Control requests are Ref2VA requests with a reference image:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=A snowy winter evening at the school gate.' \
  -F 'input_reference=@reference.png' \
  -F 'control_type=canny' -F 'control_reference=@canny.mp4' \
  -F 'extra_params={"task":"ref2va","canny":{"control_context_scale":1.0}}' \
  -F 'size=1344x768' -F 'num_frames=124' -F 'quality=lossless' \
  -o controlled.mp4
```

Control describes only the target video; reference rows receive no control
hint. Control cannot be combined with `continue_from`, long-video continuation,
timeline guides or latent-mask editing, and needs the dense BF16 checkpoint (the
NVFP4 QAD export is rejected). The control weights were trained against the base FL2VA transformer, so
control strength and fidelity on the distilled checkpoint may differ from base H3.

### Measured on 8x H200

One 1344x768, 124-frame request per feature, seed 20261122, vLLM 0.34.0.
Editing and continuation ran on the Ulysses 8 + HSDP configuration above; timeline
guides and ControlNet ran on the TP 8 control configuration:

| Feature | Latency | Check |
| --- | --- | --- |
| Ref2VA baseline | 12-13 s warm | - |
| Latent-mask edit, right half regenerated | 18.1 s | kept half PSNR 33.2 dB vs source; regenerated half 20.5 dB |
| `continue_from` (119 new frames) | 12.0 s | SSIM across the join 0.62; adjacent frames within a clip average 0.81 |
| 20 s `long_video` continuation (481 frames) | 40.3 s | SSIM across the window boundary 0.91 |
| Timeline guide at frame 96 | 21.3 s | SSIM to the guide image 0.85 at frame 96 (0.25 without guides); best-matching frame 96 |
| ControlNet canny, strength 1.0 | 27.6 s | edge F1 vs control 0.85 (0.44 at strength 0) |

## Optional NVFP4 video decoder

On Blackwell GPUs with FlashInfer installed, add this to the deployment
YAML to use the faster 8-layer decoder:

```yaml
additional-config:
  video_vae_decoder:
    model: FastVideo/FastH3-NVFP4-VAE-Decoder
    subfolder: fast-8L
```

Change `fast-8L` to `quality-36L` for the deeper decoder, then restart.
See the [decoder model card](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder)
for the available variants. For FP8 or NVFP4 QAD checkpoints, follow the
[single-card guide](<FastH3 RefOmni Single Card.md>).

## Benchmark: 8× NVIDIA RTX PRO 5000

Measured at 1344×768 with BF16 on eight 72 GB Blackwell GPUs, one request
at a time. Timing includes MP4 encoding and excludes model loading:

| Output | Generation time | Reported allocator peak |
| --- | ---: | ---: |
| 124 frames, about 5 seconds | 41.04 s median of 3 runs | 35.15 GiB |
| 345 frames, about 14 seconds | 95.93 s, 1 run | 38.94 GiB |

These are offline measurements, not HTTP response times. See the
[benchmark details](https://github.com/lishunyang12/vllm-omni/pull/63)
for the measured environment and configuration.


---

## FILE: recipes/FastVideo/FastH3%20RefOmni%20Single%20Card.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/FastVideo/FastH3%20RefOmni%20Single%20Card.md)

# FastH3 RefOmni Single Card

[FastH3 family](README.md) · [Multi-GPU serving](FastH3-Omni.md)

Turn a reference image and a text prompt into a video with stereo audio.
Choose the command for your GPU below; the configuration handles memory
placement, attention, eight inference steps, and video decoding.

| GPU | Default checkpoint | Configuration |
| --- | --- | --- |
| RTX 5090 (32 GB), DGX Spark (128 GB), RTX PRO 6000 Blackwell (96 GB) | NVFP4 QAD | [Blackwell YAML](fasth3-omni-quantized.yaml) |
| RTX 4090 (24 GB) | BF16 with online FP8 | [RTX 4090 YAML](fasth3-omni-rtx4090.yaml) |

## 1. Install

Use Linux and Python 3.12. Install vLLM and PyTorch following the
[CUDA installation guide](../../docs/getting_started/installation/gpu.md),
then run from the vLLM-Omni repository root (system packages shown for Ubuntu):

```bash
uv pip install -e .
sudo apt-get update
sudo apt-get install -y ffmpeg
```

On DGX Spark, use the native `aarch64` environment. Blackwell uses the
FlashInfer package supplied by the matching vLLM installation.
If Hugging Face requests access to the MiniMax-H3 components, accept the
model's access terms and run `hf auth login` once.

## 2. Start the server

**RTX 5090, DGX Spark, or RTX PRO 6000 Blackwell:**

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastH3-Omni-NVFP4-QAD --omni \
  --config recipes/FastVideo/fasth3-omni-quantized.yaml
```

This loads the smaller prequantized checkpoint and uses the fast 8-layer
video decoder. Layer offload reduces GPU memory use.

**RTX 4090:**

```bash
uv pip install -e '.[vsa]'

CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
  --config recipes/FastVideo/fasth3-omni-rtx4090.yaml
```

The first start downloads weights; the 4090 also converts eligible weights
to FP8 while loading. Subsequent starts reuse the download cache.
Keep the server terminal open and wait for `Application startup complete`.

## 3. Generate video and audio

In a second terminal, save your image as `reference.png` and run:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=The person in the reference image smiles and waves at the camera. The camera slowly moves closer. Birds chirp as the person says, "Welcome to the garden!"' \
  -F 'input_reference=@reference.png' \
  -F 'size=832x480' \
  -F 'seconds=5' \
  -F 'seed=1234' \
  -o output.mp4
```

Open `output.mp4`: it contains 124 frames at 24 fps and 32 kHz stereo audio.
For 768P, change `size=832x480` to `size=1344x768`.
For multiple images, video, or audio references, see the
[H3 reference API](../MiniMaxAI/MiniMax-H3.md).

## Optional settings

- **FP8 checkpoint:** in the Blackwell command, replace the model ID with
  `FastVideo/FastH3-Omni-FP8`. A local snapshot directory also works.
  The loader detects the format; this export's NVFP4 text encoder also
  requires Blackwell.
- **36-layer decoder:** in the Blackwell YAML, change
  `additional-config.video_vae_decoder.subfolder` from `fast-8L` to
  `quality-36L`, then restart. Both come from
  [FastH3-NVFP4-VAE-Decoder](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder).
- **DGX Spark:** remove `diffusion-offload-config` from the Blackwell YAML
  to keep the model resident in unified memory and avoid offload transfers.
- **Memory is tight:** start at 480P, keep the default decoder and tiled
  decoding, and close other GPU workloads.


---

## FILE: recipes/Efficient-Large-Model/Open-H3-Regenerate-2K-H200.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/Efficient-Large-Model/Open-H3-Regenerate-2K-H200.md)

# Open-H3-Regenerate-2K — H200

> Regenerate a 768P MiniMax-H3 draft at 2560x1440 in three denoising steps

## Summary

- Vendor: NVlabs (`Efficient-Large-Model`)
- Model: [`Efficient-Large-Model/Open-H3-Regenerate-2K`](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)
  over the `MiniMaxAI/MiniMax-H3` FL2VA partition
- Task: video-to-video refinement of an H3 draft (`t2va` layout)
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Hardware: 2x NVIDIA H200 141GB
- Maintainer: Community

## When to use this recipe

Generate a clip cheaply at 1344x768 with MiniMax-H3, then re-render the chosen
draft at 2560x1440. The release is an open replacement for the closed
H3-Regenerate-2K API: the draft latent is upscaled by the 3D-conv latent
upscaler, put back on the flow at sigma 0.909375 and denoised in three Euler
steps by MiniMax-H3 carrying a merged LoRA (Open-H3 refiner + LightX2V FL2VA
Turbo + LMS Sharp), then decoded.

## Supported model contract

| Item | Contract |
| ---- | -------- |
| Request | `/v1/videos` or `/v1/videos/sync` with `source_video` (the draft) and `extra_params` `{"task": "t2va", "regenerate": true}`; no noise masks |
| Draft canvas | `width`, `height`, `num_frames` / `fps` describe the draft. The draft is resampled to 24 fps, scaled to cover the canvas and centre-cropped, then cut or padded (last frame held) to the frame count. Validated: 1344x768, 124 frames |
| Prompt | The draft's prompt; it conditions the refine pass |
| Output size | `latent_upscale` in `extra_params`, default `{"width": 2560, "height": 1440, "align": 32}`; the same request formats as [latent super-resolution](../MiniMaxAI/MiniMax-H3.md#latent-super-resolution) |
| Steps | `num_inference_steps` 3 (default, sigmas 0.909375, 6/7, 2/3) or 2 (sigmas 0.7, 0.494, from the release's shift-6 rule) |
| Sampler | Euler, CFG 1, video flow shift 6; `flow_shift`, `sampler`, `latent_refine` and per-request `lora` are rejected |
| Audio | The refiner conditions on encoded silence, as the release does. The output carries the draft's soundtrack after an H3 audio-VAE round trip, or silence if the draft has none |

Every request on this server is a regenerate request; serve plain MiniMax-H3
from a separate server.

## References

- Release: <https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K>
  at revision `5809c33730c18b6b1c827b40974855a4288641b3`, and its reference code in
  [NVlabs/Sana `models/Open-H3-Regenerate-2K`](https://github.com/NVlabs/Sana/tree/6e4a44002149418a547434b91d7280a8bbdcace3/models/Open-H3-Regenerate-2K)
  at commit `6e4a44002149418a547434b91d7280a8bbdcace3`
- Base model: <https://huggingface.co/MiniMaxAI/MiniMax-H3> at revision
  `48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc` (`FL2VA/`)
- Base model and request API: [MiniMax H3 recipe](../MiniMaxAI/MiniMax-H3.md),
  [Videos API](../../docs/serving/videos_api.md#minimax-h3-regenerate)

## Checkpoints

`vllm-omni` serves the native MiniMax-H3 FL2VA checkpoint and needs only the
adapter files and the latent upscaler from the release (about 14 GB); the
release's `base_model/` copy is not used.

```bash
export REGEN_DIR=/path/to/Open-H3-Regenerate-2K
hf download Efficient-Large-Model/Open-H3-Regenerate-2K \
  --include 'lora/*' 'latent_upsampler/*' --local-dir "${REGEN_DIR}"
```

`lora/open_h3_regenerate_2k_lora.safetensors` is fused into the transformer
while the checkpoint loads. The LMS Sharp AdaLN curve branches in
`lora/sharp_curve_branches.safetensors` have all-zero up factors in this
release, so they contribute nothing; the loader verifies that and does not need
the release's separately fetched `adaln_t_table`.

The fusion runs on the accelerator, one parameter at a time, on top of the
weights already loaded. The largest parameter needs up to 1.72 GiB at that
moment; the server logs the figure when it claims the bundle and reports an
out-of-memory error during the fusion as such.

## Hardware

- Accelerator model and per-device memory: NVIDIA H200, 141 GB
- Number of devices: 2 (Ulysses SP2)
- Device interconnect: NVLink
- Host memory: the Qwen3-VL text encoder is layer-offloaded to host memory
- Qualification scope: online serving of one 1344x768 -> 2560x1440, 124-frame request

## Software environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.15
- Driver / runtime: NVIDIA 590.48.01, CUDA 13.2, PyTorch 2.13.0
- vLLM version: 0.32.0
- vLLM-Omni version or commit: `8d4b2789a` plus this change

## Command

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "${MODEL_ROOT}/FL2VA" --omni \
  --served-model-name H3-Regenerate --task-type fl2va --trust-remote-code \
  --lora-path "${REGEN_DIR}" \
  --num-gpus 2 --usp 2 --ring 1 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}' \
  --additional-config '{"latent_refine_max_tokens_per_rank":70000}' \
  --enforce-eager --diffusion-attention-backend FLASH_ATTN \
  --port 8091
```

- `--lora-path` names the release directory; the LoRA is fused at load time,
  so the DiT cannot be offloaded. Offloading the text encoder alone is
  supported.
- A 2560x1440, 124-frame refine packs about 133,000 video rows, 66,600 per
  rank at SP2, above the default 65,536-row refine guard; the
  `latent_refine_max_tokens_per_rank` override admits it.
- The release's latent upscaler is used unless `latent_upscaler_path` names
  another checkpoint.

## Verification

```bash
curl -sS http://127.0.0.1:8091/v1/videos/sync \
  -F "prompt=The prompt the draft was generated from" \
  -F width=1344 -F height=768 -F aspect_ratio=16:9 -F num_frames=124 -F fps=24 \
  -F seed=1766580932 \
  -F 'extra_params={"task":"t2va","regenerate":true}' \
  -F "source_video=@draft.mp4;type=video/mp4" \
  -o regenerated.mp4
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames -of csv=p=0 regenerated.mp4
```

Expected: `h264,2560,1440,124` and an `aac` stream.

## Notes

Qualification evidence on this hardware (single bounded runs, not benchmarks):

- **DiT parity.** One transformer forward on real weights: vLLM-Omni (native
  FL2VA checkpoint at the base revision above, LoRA fused at load) against
  Diffusers 0.40.0 `MiniMaxH3Transformer3DModel` (the release's `base_model/`
  at the release revision above, LoRA merged as `cast(W_fp32 + B @ A)` like
  the release). Fixed random 7x16x32 video latent, 48 text rows and audio rows
  locked at timestep 1, all drawn from seed 1234, at the first and last
  sigmas of the default 3-step refine. Each run needs one GPU with about 80 GB
  free:

  ```bash
  T=tools/minimax_h3/regenerate_dit_parity.py
  PYTHONPATH=. python $T reference ref.pt
  PYTHONPATH=. python $T reference --no-adapter ref-base.pt
  PYTHONPATH=. python $T vllm-omni omni.pt
  PYTHONPATH=. python $T vllm-omni --no-adapter omni-base.pt
  PYTHONPATH=. python $T compare ref.pt omni.pt            # column 1
  PYTHONPATH=. python $T compare ref-base.pt omni-base.pt  # column 2
  PYTHONPATH=. python $T compare ref.pt ref-base.pt        # column 3
  ```

  | sigma | vLLM-Omni vs reference, with adapter | same, base H3 without adapter | adapter vs base (reference) |
  | ----- | ------------------------------------ | ----------------------------- | --------------------------- |
  | 0.909375 | rel. L2 1.2e-2, cosine 0.99993 | rel. L2 2.2e-2, cosine 0.99976 | rel. L2 0.44 |
  | 0.666667 | rel. L2 2.8e-2, cosine 0.9996 | rel. L2 1.8e-2, cosine 0.99984 | rel. L2 0.38 |

  The adapted outputs agree as closely as the two base implementations do
  (BF16, FlashAttention vs cuDNN SDPA), while the adapter itself moves the
  velocity by about 40%. The script downloads both models at the pinned
  revisions (`--release-dir` and `--base-dir` take local copies instead) and
  skips the Sharp curve branches after asserting they are zero.
- **End to end.** A MiniMax-H3 T2VA draft (1344x768, 124 frames, 50 steps)
  regenerated with the command above, first request after startup:

  | Steps | Denoise | Request (encode, text, upscale, refine, decode, MP4) | Output |
  | ----- | ------- | ---------------------------------------------------- | ------ |
  | 3 | 77 s (25.8 s/step) | 112 s | 2560x1440, 124 frames, H.264 + AAC |
  | 2 | 51 s | 77 s | 2560x1440, 124 frames, H.264 + AAC |

  Peak process memory was 87.3 GB per DiT rank (nvidia-smi, 2 s sampling).
  Composition and motion follow the draft; the mean squared Laplacian
  response of the output is 15.8x (range 14.4-19.3, 11 frames) that of the
  draft upscaled bicubically to the same size.
- **Known limitations.** One request at a time. Only the default 2560x1440
  output was run end to end; other `latent_upscale` sizes follow the upscaler's
  limits but are unqualified. The release's accelerated `sol_engine_fwd`
  (Sol-Attn, MXFP8, CUDA graphs) and its fast VAE decoder are not ported.

## Supported features

| Feature | Status |
| ------- | ------ |
| Ulysses sequence parallelism | Supported (SP2 validated) |
| Text-encoder layerwise offload | Supported |
| DiT offload | Not supported (the LoRA is fused while the checkpoint streams in) |
| Step execution, continuation, latent-mask editing, timeline guides | Not supported |
| Cache-DiT / TeaCache | Not supported (`quality=lossless` only) |
| Separate encoder stage | Not supported (the DiT stage encodes the silence the refiner conditions on) |


---

## FILE: recipes/TaoLiveAIGC/TaoMate-H3.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/TaoLiveAIGC/TaoMate-H3.md)

# TaoMate-H3: ordinary inference and realtime streaming

[TaoLiveAIGC/TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3) is a rank-128 LoRA
for the [MiniMax-H3 FL2VA checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-H3).
It generates a continuous presenter video with audio and accepts prompt changes between
five-second generation windows. vLLM-Omni serves it through the AR-Diffusion engine and
`WS /v1/realtime/video`.

## Execution and cache ownership

A window contains four causal phases of 34/34/34/17 frames. The first window produces
124 frames including the decoder's five-frame tail; subsequent windows add 119 frames.
The timeline is fixed at 24 fps with 32 kHz stereo audio.

| Operation | Model behavior |
| --- | --- |
| Prompt encode | Qwen3-VL encodes the prompt at startup and on an interaction. One prompt conditions all four phases of a window. |
| Audio teacher | The base DiT, with LoRA disabled, runs nine audio-only forwards once per window. It retains the previous window's last 40 audio latents per channel. |
| Student denoise | Three LoRA forwards per phase consume the current document and clean history. The teacher's states 3, 6 and 9 replace the corresponding student audio states. |
| Clean commit | A separate sigma-zero forward recomputes clean K/V. Only a successful complete forward publishes the phase into history. |
| Decode | Incremental video VAE windows retain their temporal overlap; the audio VAE decodes the matching sample interval. The final phase flushes the held video tail. |

The AR-Diffusion runtime owns session KV allocation, capacity accounting, transactions
and release. TaoMate supplies audio/video token spans and retention rules: keep the
first phase's video sink and two most recent phases, and clear audio history every
12 windows by default. Each Ulysses rank stores its own attention-head shard. Denoising
scratch never becomes committed history; the last student step is still noisy and
cannot replace the clean-commit forward.

The shared step scheduler handles runnable and prompt-waiting requests, interaction
wakeups, deadlines and cancellation. TaoMate defines the window boundary and optional
fallback prompt. A waiting request does not repeatedly execute a model step.

Teacher/student math, global audio/video positions and incremental VAE state remain
model-specific. By default the LoRA uses BF16 sidecars on the resident base weights;
the worker-owned `DiffusionLoRAManager` temporarily suspends the adapter for the teacher.
The checkpoint uses ordinary low-rank A/B pairs in a native serialization format; its
fused QKV factors already match the loaded projection order. The manager also controls
the optional merged student bank, which uses a second weight set and has different
memory and numerical behavior.

## Requirements and serve commands

- MiniMax-H3 `FL2VA/` weights and the TaoMate adapter directory containing
  `adapter_model.safetensors` and `config.json` (or `adapter_config.json`). Point the
  model path at `FL2VA/`, not the snapshot's modular Diffusers index.
- Hopper or newer GPUs with FlashAttention-3 (`fa3_fwd_interface`), or FA4 on Blackwell.
  Reference deployments were exercised on H200-class 141 GB GPUs; budget for the DiT,
  text encoder, VAEs, LoRA, engine KV, scratch buffers and graph allocations.
- Pure Ulysses sequence parallelism with DiT tensor parallelism fixed at 1.
  Both supplied configurations admit one session (`max_num_seqs: 1`).

Four-GPU BF16 configuration:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/taomate_h3_usp4_realtime.yaml \
  --lora-path /path/to/TaoMate-H3 --port 8000
```

Two-GPU configuration with per-channel FP8 DiT linears and teacher/text CUDA graphs:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/taomate_h3_usp2_realtime.yaml \
  --lora-path /path/to/TaoMate-H3 --port 8000 \
  --stage-init-timeout 1800 --init-timeout 2100
```

The two-GPU config leaves the text encoder, token refiner, patch/condition projections,
time embeddings, final layer and AdaLN projections in BF16. It uses the checkpoint's
256 px video decoder tiles with 32 px minimum overlap and batched tile decoding.
Do not increase the decoder tile size: 384/480 px tiles previously produced a visible
16 px lattice. Keep `taomate_h3_lora_merge: false`: merging into FP8 student weights
previously erased much of the small adapter delta and blurred the video.

The engine performs a throwaway rollout before serving, including in eager mode when
`ar_diffusion_kv_config.warmup_cudagraph: true`. Four windows cover the teacher's initial
audio length and the later 198/198/199-latent cycle. Capturing teacher and encoder graphs
for a wide prompt-length range can take several minutes; size the startup timeouts and
capture range for the deployment.

## Ordinary full-request inference

Use `taomate_h3_usp4_full_request.yaml` to return a complete video with stereo audio
through the ordinary `/v1/videos` job API:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/taomate_h3_usp4_full_request.yaml \
  --lora-path /path/to/TaoMate-H3 --port 8000
```

This configuration uses four GPUs, eager BF16, Ulysses 4, text encoder TP4 and
tile-parallel VAE decoding. It disables step execution and streaming output while
retaining the AR-Diffusion engine that owns the session's KV. The pipeline runs all
requested phases, joins their audio and video, and returns one final MP4. The prompt
is fixed for the request; the default `close_session: true` releases its history and
decoder state after completion or cancellation. Requests are served one at a time.

Create a job:

```bash
curl --fail-with-body --silent --show-error http://localhost:8000/v1/videos \
  -F 'prompt=A friendly presenter in a quiet studio explains how to grow a small garden, speaking clearly with natural gestures.' \
  -F 'size=480x864' -F 'num_frames=124' -F 'fps=24' \
  -F 'seed=8301' -F 'num_inference_steps=3'
```

Copy the returned `id` into `VIDEO_ID`. Poll the job until its status is `completed`,
then download the content:

```bash
VIDEO_ID='video_gen_replace_with_returned_id'
curl --fail-with-body --silent --show-error "http://localhost:8000/v1/videos/$VIDEO_ID"
# After the job reports completed:
curl --fail-with-body --silent --show-error \
  "http://localhost:8000/v1/videos/$VIDEO_ID/content" --output taomate.mp4
```

Use native frame budgets `124 + 119 * (n - 1)`, such as 124 or 243. The current
pipeline expands other budgets to complete internal requests, so the actual result
can be longer than the requested frame budget. At 24 fps, 124 frames are about
5.167 seconds and 243 frames are 10.125 seconds.

To cancel a job that is `in_progress`, use `DELETE /v1/videos/{id}`. Cancellation is
checked at student denoise-step boundaries; it does not preempt a kernel already
running. A subsequent request starts with fresh request-owned state.

### Measure ordinary request latency

Use the diffusion serving benchmark to measure serial video jobs through receipt
of the complete MP4. Set a 50 ms polling interval for this comparison; job cleanup
and media decoding are outside the reported request latency. The custom request
also uses 124 frames for warmup, matching the original experiment: the JSONL
`num_frames` field takes precedence over the benchmark's one-frame warmup override.

```bash
cat > requests.jsonl <<'JSONL'
{"prompt":"A friendly presenter in a quiet studio explains how to grow a small garden, speaking clearly with natural gestures.","size":"480x864","num_frames":124,"fps":24,"seed":8301,"num_inference_steps":3}
JSONL
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --endpoint /v1/videos \
  --dataset custom --dataset-path requests.jsonl --task t2v \
  --num-prompts 34 --max-concurrency 1 --warmup-requests 1 \
  --warmup-num-inference-steps 3 --video-poll-interval 0.05 \
  --output-file block-a.json
```

For an optimization comparison, keep the checkpoint, adapter, devices, inputs and
configuration fixed. Enable `taomate_h3_decode_overlap` and
`taomate_h3_teacher_cuda_graph` in both arms; enable `taomate_h3_vae_cuda_graph`
only in the candidate. Restart and warm each arm, and use AB/BA/AB blocks rather
than comparing one first request. This measures serial latency; it does not
establish concurrent throughput or tail-latency improvements.

## Client and prompt updates

```bash
python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --port 8000 --model /path/to/MiniMax-H3/FL2VA --size 480x864 --fps 24 --num-frames 362 \
  --seed 8301 --no-helios-distilled-preset --prompt "..." \
  --prompt-updates '[{"at": 6.0, "prompt": "..."}]'
```

A `session.start` specifies `prompt`, `width`, `height`, `fps: 24`, `seed` and
`num_frames`. A budget of `124 + 119 * (n - 1)` frames requests `n` windows.
Binary messages carry fragmented MP4; `video.chunk_metadata` describes each chunk.
The WebSocket fMP4 carries aligned video and audio tracks. Saving the stream as
MP4 preserves both tracks and their timestamps.

Send prompt updates using the interaction envelope:

```json
{
  "type": "session.interaction",
  "interaction": {
    "event_id": "prompt-2",
    "event": {"prompt": "The presenter introduces the next topic."},
    "transition_chunks": 0
  }
}
```

The update is processed at a chunk boundary and conditions the next generation window;
it never changes the teacher or remaining phases of a window already in progress.
The `started_event_ids`, `active_event_ids`, and `completed_event_ids` markers
acknowledge installation of the prompt embeddings in session state. With TaoMate's
immediate transition, these markers can appear before the window that uses the new
prompt: a window snapshots conditioning when its first phase starts. They do not
identify the first visibly changed frame; VAE temporal context and overlap blending
can delay or spread that change across frame boundaries.
Different prompt lengths are supported. Multiple pending updates use the latest prompt.
Send `session.stop` to cancel, and `session.ping` to keep a deliberately held connection
alive. A bounded hold may use the previous prompt or the configured fallback prompt;
an update arriving after that window starts applies to the following window.

## Deployment controls

These keys live under the stage's `model_config`. Defaults below are pipeline defaults;
the supplied YAML files enable additional optimizations.

| Key | Default | Meaning |
| --- | --- | --- |
| `taomate_h3_width` / `taomate_h3_height` | 480 / 864 | Fixed canvas; both dimensions must be 32-aligned, with short edge 480, 768 or 1088. |
| `taomate_h3_seed` | 8301 | Default seed. Window `k` uses `seed + k` for audio and `seed + k * 1000003` for video. |
| `taomate_h3_audio_kv_reset_requests` | 12 | Clear retained audio K/V every N windows while keeping the video sink and recent video. |
| `taomate_h3_allow_no_lora` | false | Run the base checkpoint without the adapter for debugging. |
| `taomate_h3_pad_text_tokens` | unset | Bound packed text shapes using 64-token buckets, capped by this budget. Longer prompts run unpinned with a warning. |
| `taomate_h3_hold_for_prompt` | false | Starting with window 1, wait for a prompt update received since the previous window started. |
| `taomate_h3_hold_max_seconds` | 0 | Maximum hold duration; 0 waits indefinitely. On timeout start a window with the previous or fallback prompt. |
| `taomate_h3_hold_fallback_prompt` | unset | Prompt used on hold timeout; encoded once on first use. Unset retains the previous prompt. |
| `taomate_h3_hold_poll_seconds` | 0.02 | Compatibility setting from worker polling; prompt waits now use shared scheduler wakeups. |
| `taomate_h3_teacher_cuda_graph` | false | Capture/replay teacher forwards by document shape. Capture failure disables graphs and falls back to eager execution consistently across Ulysses ranks. |
| `taomate_h3_cuda_graph_max_entries` | 16 | Teacher graph capacity; unpinned entries are evicted in LRU order. |
| `taomate_h3_teacher_graph_text_lengths` | unset | Inclusive token-count range such as `"360-470"` to capture at startup. Requires a text budget and at least three teacher graph entries per length; these entries stay pinned. |
| `taomate_h3_text_encoder_cuda_graph` | false | Capture/replay text-only encoding by token count, including the configured startup range. Offloaded encoders keep eager execution. |
| `taomate_h3_warmup_requests` | 4 | Number of windows in the startup rollout; four cover the initial and continuation teacher geometries. |
| `taomate_h3_adaln_cache` | true | Cache exact AdaLN projections. Disable to recompute small projections and avoid the host digest synchronization. |
| `taomate_h3_lora_merge` | false | Experimental shadow student weights with merged LoRA; restores base weights for the teacher. Leave off for the validated FP8 configuration. |
| `taomate_h3_decode_overlap` | false | Queue video decode on a second CUDA stream alongside clean commit. |
| `taomate_h3_vae_cuda_graph` | false | Capture and replay fixed-shape video VAE tile decode; tile gather collectives remain outside the graph. |
| `taomate_h3_cudnn_benchmark` | false | Enable cuDNN autotuning for fixed-shape VAE convolutions. |
| `taomate_h3_vae_decoder_tile_size` | checkpoint: 256 | Decoder tile edge, a positive multiple of 16. Preserve 256 for the released checkpoint; too few tiles for the VAE ranks trigger a slower fallback. |
| `taomate_h3_vae_decoder_tile_overlap_min` | checkpoint: 64 | Minimum tile overlap, a nonnegative multiple of 16. The USP2 config uses 32. |
| `taomate_h3_vae_stack_tiling` | false | Decode each rank's tiles as a batch to reduce peak memory. |
| `taomate_h3_fp8_quant_cuda_op` | true | Use vLLM's CUDA activation-quantization kernel for FP8 linears. |
| `taomate_h3_log_timings` | false | Log phase/stage durations; adds device synchronizations. |
| `taomate_h3_profile_chunk` / `taomate_h3_profile_dir` | unset / `/tmp/taomate_h3_profile` | Profile the selected chunk and write per-rank kernel tables and Chrome traces. |

`ar_diffusion_kv_config` controls the engine pool and warmup. Its memory fraction now
budgets real persistent K/V; it is not a bookkeeping-only allocation. Include graph
and model-owned decoder/teacher state when checking available capacity.

Both configurations default to `enforce_eager: true`. Regional compilation remains an
option with `enforce_eager: false`, a suitable `taomate_h3_pad_text_tokens` budget and
`VLLM_OMNI_TORCH_DYNAMO_RECOMPILE_LIMIT=64`. Validate it with the chosen graph and LoRA
settings; combining compilation and merged weights previously failed graph capture.

## Validation and limits

- One session per server; this integration does not add concurrent-session batching.
- Text-only T2VA with a fixed deployment canvas. Reference media, another canvas, or
  explicit `fps`/`frame_rate` values other than 24 are rejected before generation.
- Video is not bit-reproducible between runs, including in the release runtime.
- Teacher graphs distinguish valid prompt lengths. A previously unseen length outside
  the startup range can pause the stream while capturing; size the range using actual
  token counts, not characters.

Validate the deployment with prompt changes, at least 13 windows to cross audio-history
reset, bounded/unbounded prompt holds, cancellation followed by another session, and
complete MP4 decoding. Compare both per-window latency and peak GPU memory with the
same checkpoint, prompts, precision, graph settings and host load.
