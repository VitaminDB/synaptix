# vllm-project__vllm-omni__recipe-MiniMax-H3

Source: https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md
Fetched: 2026-11-28



---

## FILE: recipes/MiniMaxAI/MiniMax-H3.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/MiniMaxAI/MiniMax-H3.md)

# MiniMax H3

> Joint video and audio generation with text, first/last keyframes, and
> mixed image/video/audio references

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Maintainer: Community

MiniMax H3 is a CFG-distilled joint video/audio diffusion transformer. Its
checkpoint has two task-specific DiT partitions:

- `FL2VA`: text-to-video+audio (`t2va`) and first-frame-to-video+audio
  (`fl2va`)
- `Ref2VA`: up to 9 images, 3 videos, and 3 audio references in supported
  image/video/audio combinations (`ref2va`; audio-only is rejected)

One vLLM-Omni diffusion stage can load both DiTs while instantiating the
tokenizer, processor, Qwen3-VL text encoder, video VAE, and audio VAE only
once. Requests select the DiT with `extra_params.task`.

This single-stage recipe uses the default `model_loaded.text_encoder: true` and
`model_loaded.vae_encoder: true`. Precomputed text conditioning can replace the
local Qwen call. Setting `model_loaded.text_encoder: false` while leaving
`model_loaded.vae_encoder: true` requires precomputed text conditioning;
reference media is still encoded locally.

The generated MP4 contains H.264 video and synchronized stereo audio.

## Optional LiteVAE video decoder

To use the standard non-INT8 H3 Lite decoder, add:

```bash
--additional-config '{"lightvae_decoder_path":"/weights/lightvae-lite-h3-decoder.safetensors"}'
```

LiteVAE changes the video decoder while preserving the video encoder and audio
VAE. Omit `video_vae_decoder` when selecting LiteVAE. See the
[LiteVAE guide](../../docs/user_guide/diffusion/light_vae.md) for checkpoint
requirements, decoder selection, and quality review.

## Prerequisites

The checkpoint requires Hugging Face access approval. Authenticate once;
`vllm serve` downloads the required components automatically:

```bash
hf auth login
export MODEL=MiniMaxAI/MiniMax-H3
```

By default, the pipeline downloads `FL2VA/**`, `Ref2VA/model_index.json`, and
`Ref2VA/transformer/**`. It does not download or load the diffusers-format
`transformer`, `transformer_ref`, or `vae` weights at the repository root, nor
duplicate Ref2VA copies of shared components.

Install vLLM-Omni from the checkout containing MiniMax H3 support. The
two-GPU RTX 5090/4090 profiles use cuDNN attention and do not need
FlashAttention-4. Install the optional dependency only for the four-GPU
B300/GB200 `FLASH_ATTN` profile:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

To keep FA4 available as an explicit option on Blackwell, install the
FlashAttention-4 extra:

```bash
uv pip install -e '.[fa4]'
```

On AMD ROCm, install without the `[fa4]` extra (FA4 is CUDA-only) and use
`--diffusion-attention-backend FLASH_ATTN`; see
[AMD ROCm (gfx942 / gfx950)](#amd-rocm-gfx942--gfx950).

`ffmpeg` and `ffprobe` must be available on `PATH`. They are used for
reference-video preparation and MP4 output.

## CPU MP4 response encoding

For CUDA and ROCm deployments, non-streaming MP4 responses are encoded on the
host CPU through PyAV/libx264 after generation. The response encoder selects the
path automatically at runtime. The server-owned parallel converter accepts
supported frame shapes and dtypes with either per-channel-contiguous or strided
RGB planes, including interleaved arrays materialized by output transport.
Standalone callers without a parallel converter retain the legacy fallback for
strided planes. No CLI flag, model declaration, or user configuration is
required. Streaming fMP4 output is unchanged.

A community benchmark on 2x Xeon 8480C reported the following comparison
between the legacy and direct planar paths
([full result](https://github.com/vllm-project/vllm-omni/pull/6288#issuecomment-5337546499)):

| Metric | Legacy | Direct planar | Change |
| --- | ---: | ---: | ---: |
| Median wall time | 1.805 s | 1.394 s | -22.8% |
| Median process CPU time | 3.613 s | 3.207 s | -11.2% |
| Peak RSS | 3182 MiB | 2794 MiB | -387 MiB (-12.2%) |

Across the 1.0-8.7 s sweep, wall-time improvement was approximately 21.8-22.6%;
outputs were byte-identical and full decode passed. This is evidence for host
CPU response encoding. Actual gains depend on the CPU and runtime, and should
not be interpreted as GPU, DiT, or stage 0 speedups.

## Start a server

Pass the repository ID directly. The pipeline uses `FL2VA` for model discovery
and shared components, and loads the second DiT from `Ref2VA/transformer`.

### Memory and storage requirements

Treat GPU HBM, host RAM, and checkpoint storage as separate requirements. Each
H3 checkpoint partition (`FL2VA` or `Ref2VA`) contains about **134 GiB** of
BF16 safetensors (about **135 GiB** on disk). Keeping both partitions
locally therefore needs roughly **270 GiB** of model storage. A combined
service downloads both; `--task-type fl2va` or `--task-type ref2va` downloads
only the selected partition.

CPU offload and distributed layerwise offload reduce GPU residency; they do
not make the model weights disappear. With `--dlo-no-use-allgather`, each
worker retains its standard-loader rank-local weights in host memory, including
pinned CPU buffers used for H2D streaming. Use at least **200 GiB available
system RAM** before starting the two-GPU recipe; a **384 GiB host is
recommended** to leave room for the OS, CUDA/PyTorch allocations, request
inputs, and filesystem cache. Do not run the FL2VA and Ref2VA servers at the
same time on a host sized for this minimum.

The consumer-GPU profiles below are HBM budgets only. They still require the
host-RAM budget above.

### Single GPU: blockwise capacity path

Use ordinary layerwise offload when one GPU cannot keep the Qwen3-VL encoder
and DiT resident together. The component list below streams the active DiT,
the Qwen vision blocks, and the first 50 Qwen text layers. Encoder blocks stay
rank-local; the video/audio VAEs remain resident.

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 1 \
  --diffusion-offload-config \
  '{"mode":"layer","components":["dit","text_encoder"]}' \
  --enforce-eager \
  --diffusion-attention-backend FLASH_ATTN
```

This is a capacity profile, not a latency profile: every denoising step streams
DiT blocks over the host link, while encoder blocks are streamed only during
the short conditioning phase. It requires host memory for the complete
checkpoint plus pinned transfer buffers. Re-measure peak HBM on the target
shape; this command does not claim a particular GPU model as validated.

A one-B300 correctness smoke selected only `text_encoder` (keeping the DiT and
VAEs resident) and completed a 384x672, 5-second T2VA request with two denoise
steps. It installed 77 encoder hooks across the vision and text stacks, used a
77,728 MiB worker peak, and measured 2.803 seconds encode, 2.706 seconds
denoise, and 4.503 seconds decode. These reduced-step numbers validate the
execution path; they are not a quality or production-latency benchmark.

To use whole-component behavior, change the config to
`{"mode":"module","components":["dit","text_encoder"]}`. Module
offload swaps the complete encoder and DiT and therefore has a higher
encode-phase peak than encoder blockwise offload.

### Two 24/32 GB GPUs: TP2 distributed layerwise offload

For two PCIe consumer GPUs, combine TP2 with distributed layerwise offload
(DLO). The standard loader first creates the rank-local TP shard. DLO keeps
that shard in pinned host memory and streams the 30 tail DiT blocks through a
shared two-buffer window without a DP AllGather. The first 20 DiT blocks are
copied to the GPUs once per denoise stage, reused by every sampling step, and
released before VAE decode so the decoder can reuse their HBM.

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --dlo-resident-layers 20 \
  --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

Use the profile that matches the per-GPU memory capacity:

| Profile | GPUs | Starting shape | Resident DiT blocks | Attention | Execution | Status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `rtx5090` | 2 x 32 GB | 1344x768 | 20 | cuDNN attention | eager | Target-hardware validated |
| `rtx4090` | 2 x 24 GB | 1024x576 | 12 | cuDNN attention | eager | Capacity-proxy starting point |

This topology uses all available parallel capacity: TP2 shards both the DiT
and text encoder, `--dlo-no-use-allgather` streams each rank's local TP shard
without reconstructing full blocks, and VAE patch parallelism splits tiled
decode across both GPUs. cuDNN attention is selected explicitly for the RTX
consumer path; the server stays eager to avoid an unqualified compile path.

The resident count changes placement and transfer frequency only; it does not
quantize or change the BF16/FP32 denoise math. Re-measure peak memory before
increasing it on a different request shape.

### RTX 5090 target-hardware validation

At vLLM-Omni commit `ae6577ea`, one full 50-step T2VA request completed on
2 x RTX 5090 without OOM:

| Shape    | Frames        | Client E2E | Sampled peak/GPU       | Output validation                                            |
| -------: | ------------: | ---------: | ---------------------: | -----------------------------------------------------------: |
| 1344x768 | 124 at 24 FPS | 8 min 38 s | approximately 22.6 GiB | H.264 video + 32 kHz stereo AAC; full `ffmpeg` decode passed |

This is a single end-to-end validation run, not a warmed multi-run latency
benchmark. The sampled `nvidia-smi` peak is also not a CUDA allocator
high-water mark. The environment used vLLM 0.26.0, vLLM-Omni
`0.26.1.dev14+gae6577ea`, and PyTorch 2.11.0+cu130. The
[run record](https://github.com/lishunyang12/vllm-omni-rankings/blob/dcd06d7e83cb069842535918c0169ee9f3f29ba0/scripts/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20260805000034_86_237.png)
captures the environment, output contract, elapsed time, and sampled peak.

Before the target run, both profiles were exercised on two B300 ranks as an
allocation and correctness proxy. At 1344x768, 124 frames, and 50 steps, the
20-layer profile peaked at 27,726 MiB per rank. At 1024x576, the 12-layer
profile peaked at 18,888 MiB per rank in a 5-step capacity run. The resident
and fully streamed placements produced identical decoded video-frame and audio
hashes for the same shape, step count, prompt, and seed. The B300 result does
not establish RTX 4090 PCIe latency; treat the 4090 profile as a conservative
starting point until it is measured on that GPU.

To run T2VA, FL2VA, image+audio Ref2VA, and two-video Ref2VA in order, validate
every MP4's H.264/AAC streams, and retain live server and GPU-memory logs:

```bash
RUN_ROOT=/path/to/run-root \
MODEL_ROOT=/path/to/MiniMax-H3 \
GPU_IDS=0,1 \
PROFILE=rtx5090 \
bash examples/offline_inference/minimax_h3/run_h3_2gpu_all_tasks.sh
```

The script selects 20 resident layers for `PROFILE=rtx5090` and 12 for
`PROFILE=rtx4090`; `DLO_RESIDENT_LAYERS=N` overrides either default.

### Two 80 GB GPUs (A100): TP2 with text-encoder layerwise offload

On two 80 GB GPUs (such as 2x NVIDIA A100-SXM4-80GB), keeping the TP2-sharded
DiT (~30 GiB/GPU), video/audio VAEs, and the TP2-sharded Qwen3-VL text encoder
(~25.75 GiB/GPU) simultaneously resident leaves insufficient headroom for
`Ref2VA` multi-reference conditioning and request-scoped Cache-DiT
(`quality="high"`). Offloading only the `text_encoder` component in `layer`
mode keeps the DiT and VAEs permanently resident on GPU while streaming the
Qwen3-VL decoder layers from pinned host memory through a double-buffered
CUDA stream during conditioning:

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600 \
VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=3600 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type ref2va \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}'
```

Validated on 2x NVIDIA A100-SXM4-80GB (`Ref2VA`, 1344x768, 10.0 s at 24 FPS,
25 steps, `quality="high"`):

| Workload | Quiescent VRAM/GPU | Allocator peak/GPU (`x-peak-memory-mb`) | Sampled `nvidia-smi` peak/GPU | Warm client E2E |
| --- | ---: | ---: | ---: | ---: |
| 2 reference images (2726x1536) | 41.69 GiB | 54.63 GiB | 57.46 GiB | 756.3 s |
| 2 reference images + 2 reference videos | 41.69 GiB | — | 67.92 GiB | — |

### Four GPUs: throughput-oriented combined service

For a combined service on four high-memory GPUs, use:

- no CPU or layerwise offload;
- Ulysses sequence parallelism degree 4;
- native tiled VAE patch parallelism degree 4;
- regional `torch.compile` for the repeated DiT blocks;
- dense BF16 `TRTLLM_ATTN`, with Ring and TP left at 1.

Both DiTs remain resident, except that supported SM120 deployments offload
AdaLN projection weights by default. Other GPU architectures keep those weights
resident unless explicitly opted in. If the models do not fit, use model-level
CPU offload.

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling
```

Do not add `--enforce-eager` to this performance configuration. The first
request includes regional compilation; warm the server once before measuring
steady-state latency. H3 is CFG-distilled, so `--cfg-parallel-size` must remain
`1`. The H3 VAE supports its native `tile` mode, not
`spatial_shard_height` or `spatial_shard_width`.

### Attention Backends

On supported datacenter Blackwell systems, MiniMax H3 defaults to dense BF16
`TRTLLM_ATTN`; no attention backend flag is required. To select it explicitly,
use:

```bash
--diffusion-attention-backend TRTLLM_ATTN
```

Stable measurements with the four-GPU profile above put dense `TRTLLM_ATTN`
and FA4 within 2% of each other. `TRTLLM_ATTN` remains the datacenter Blackwell
default and enables the optional optimizations below. Confirm the server log
contains `Defaulting to diffusion attention backend TRTLLM_ATTN` before
recording measurements when using the default selection.

FA4 remains available by explicitly selecting the `FLASH_ATTN` backend:

```bash
--diffusion-attention-backend FLASH_ATTN
```

With the optional `[fa4]` dependency installed, `FLASH_ATTN` prefers FA4 on
Blackwell. Confirm the server log contains `Using CuTe FlashAttention-4 on
Blackwell` before recording FA4 measurements.

`TRTLLM_ATTN` additionally offers two **lossy** optimizations for the long main
DiT attention sequence: SAGE attention quantization and Skip-Softmax sparse
attention. Both work under the pure Ulysses parallelism of the profile above
(`--usp 4 --ring 1`). The example below enables both:

- SAGE with `fp8_e4m3` Q/K; P and V are always FP8 in this kernel. B200
  additionally supports `int8` Q/K, which preserves accuracy better than FP8.
- Skip-Softmax with a direct `threshold=0.05` (the calibrated
  `target_sparsity` control needs ModelOpt metadata that the official H3
  checkpoint does not include). Together with the cutoff below this is a
  **conservative** setting: a low threshold skips only clearly negligible tiles,
  and a cutoff close to `1.0` leaves a substantial dense prefix. Raise
  `threshold` or lower `disabled_until_timestep` for more speedup once quality
  is verified.
- `disabled_until_timestep=0.97` keeps the early high-noise steps dense. The
  gate compares against the video sigma, which H3's default flow shift of 12
  keeps high for much of the run: at 50 steps, `0.99`, `0.97`, and `0.95`
  leave the first 6, 14, and 19 of 49 denoiser forwards dense. See the
  [Skip-Softmax design](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/skip_softmax.md#timestep-gating)
  for how the cutoff maps to steps.
- A `per_role` entry that keeps the token refiner, a short attention path,
  dense. A per-role spec does not inherit `quant` or `skip_softmax` from
  `default`.

```bash
--diffusion-attention-config '{
  "default": {
    "backend": "TRTLLM_ATTN",
    "quant": {
      "dtype_qk": "fp8_e4m3",
      "q_block_size": 1,
      "k_block_size": 16
    },
    "skip_softmax": {
      "threshold": 0.05,
      "disabled_until_timestep": 0.97
    }
  },
  "per_role": {
    "minimax_h3.token_refiner": {
      "backend": "TRTLLM_ATTN"
    }
  }
}'
```

Both optimizations trade fidelity for speed and their effects compound.
Compare against dense output on the same prompt and seed before adopting them.
For the full key reference, see
[Skip-Softmax](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends/trtllm.md#skip-softmax)
and
[SAGE quantization](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends/trtllm.md#sage-quantization)
in the TRTLLM attention guide.

### Text encoder tensor parallelism

The Qwen3-VL text encoder (~51.5 GB in BF16 for the retained 50 layers) is by
default fully resident on the DiT main rank.  On multi-GPU no-offload runs that
rank becomes the peak-memory hotspot.  Add `--text-encoder-tp-size N` to shard
the encoder across the first `N` DiT ranks (the encoder is implemented with
vLLM-style tensor-parallel layers and runs with distributed collectives over
its own encoder process group):

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling
```

`N` must divide the Qwen3-VL head counts (64 attention heads / 8 KV heads), so
valid values on a 4-GPU server are 1, 2, and 4 (1, 2, 4, 8 on 8 GPUs).  The
encoder TP rank set is the first `N` DiT ranks; on 4 GPUs
`--text-encoder-tp-size 4` shards the encoder 4-way, dropping the DiT main
rank's no-offload peak by roughly `(N-1)/N` of the ~51.5 GB encoder while the
other ranks each gain `~51.5/N` GB.  The encoder output remains identical to
the reference path within bf16 rounding: every encoder rank all-reduces the
row-parallel projections, so the full `[seq, 5120]` layer-50 hidden state is
replicated on every rank.

No restart is needed: `task=fl2va` routes to `FL2VA/transformer`, while
`task=ref2va` routes to
`Ref2VA/transformer`. T2VA uses the FL2VA DiT.

### Step execution and continuous batching

H3 implements the step-wise execution contract, so the scheduler can admit and
retire requests between denoise steps instead of running one request end to
end. Add the feature gate, then raise `--max-num-seqs` to co-batch:

```bash
--step-execution --max-num-seqs 4
```

Co-batched requests are packed into a single sequence that keeps one attention
document per request, so a batch costs one DiT forward. That packing requires
`--diffusion-attention-backend FLASH_ATTN`; other backends fall back to one
forward per request. `--max-num-seqs 1` keeps the conservative single-request
step path. Cache acceleration (`--cache-backend`) is not available in step mode.

!!! warning "Co-batching does not improve H3 throughput for large simultaneous requests"
    Keep `--max-num-seqs 1` unless you specifically need scheduler-level control
    (admitting and retiring requests between denoise steps). Measured on two
    H100s (TP2, BF16, 672x384, 209 frames, 30 steps, 4 requests at concurrency
    4 with `--request-rate inf`, i.e. all four submitted at time 0; one packed
    request is 16384 rows):

    | Configuration | Wall time | Mean latency | Peak memory |
    |---------------|-----------|--------------|-------------|
    | request mode | 174.8 s | 111.5 s | 72.4 GB |
    | `--step-execution --max-num-seqs 1` | 179.0 s | 113.8 s | 72.4 GB |
    | `--step-execution --max-num-seqs 4` | 182.1 s | 175.7 s | 78.3 GB |

    A single H3 denoise step is a compute-bound dense GEMM over an already long
    packed sequence, so fusing N requests costs N times the FLOPs and buys almost
    no amortization — unlike LLM decoding, which is memory-bandwidth bound.
    Going from one request per step to four cuts the per-request denoise cost
    only from 1.323 s to 1.291 s (2.4%), which the step-mode bookkeeping then
    spends. Mean latency degrades further because co-batched requests finish
    together instead of staggered. Quantization moves the absolute numbers
    without changing this: with online `int8` the same workload runs in 153.3 s
    at 56.9 GB (request mode), and `--max-num-seqs 4` is still 5.0% slower than
    request mode.

    Two workloads outside this table are unmeasured and may behave differently:

    - **Staggered arrivals (admission latency).** A single 16384-row H3 request
      already saturates the GPUs, so submitting all requests at time 0 is the
      one arrival pattern where co-batching cannot win: it can only bunch
      completions. In request mode a new request queues behind the whole
      in-flight generation (~45 s at this size); step mode admits at the next
      denoise-step boundary (~1.3 s). Whether that shows up as a wall-clock
      benefit under Poisson arrivals is not yet measured for H3. To reproduce,
      run request mode vs `--step-execution --max-num-seqs 4` with
      `diffusion_benchmark_serving.py --request-rate 0.05` (roughly one request
      every 20 s) and report mean / p95 latency, which then includes queueing.
    - **Small requests.** The numbers above are drawn from 672x384 / 209-frame
      requests. A short clip at lower resolution packs a few thousand rows and
      may not saturate the hardware; co-batching may amortize better there.
      This is also unmeasured.

### Online FP8 quantization

MiniMax H3 supports online FP8 quantization of both the DiT and the Qwen3-VL
text decoder. The checkpoint remains BF16 on disk; vLLM-Omni creates FP8
weights at runtime and uses dynamic activation scaling during inference. By
default, `--quantization fp8` quantizes eligible attention and MLP linears in
the text decoder, token refiner, and main DiT blocks, as well as the DiT
condition and AdaLN projections. The Qwen vision tower, embeddings, norms,
RoPE, both VAEs, and the model's FP32 patch, timestep, and output projections
keep checkpoint precision.

To select a component, use `--diffusion-quantization-config` with
`{"transformer":{"method":"fp8"}}` for DiT-only FP8 or
`{"text_encoder":{"method":"fp8"}}` for text-decoder-only FP8. The two
entries can be combined. The shorthand below enables both components.

Add this option to an existing H3 server command:

```bash
--quantization fp8
```

#### Single 96 GB GPU, no-offload capacity check

Use the FL2VA-only partition for this capacity test. Loading the combined
service would also load the Ref2VA DiT and would test a different memory
budget. A no-offload capacity check should omit
`--diffusion-offload-config` and all legacy `--enable-*-offload` aliases. VAE
tiling changes decode placement but does not offload model weights to the CPU.

The run passes the capacity check when the server initializes, the request
finishes without CUDA OOM or Xid errors, `peak_used_mib` remains below the
card's reported `memory_total_mib`, and `ffprobe` reports H.264 video plus
32 kHz stereo AAC audio. Report the measured headroom rather than assuming
that every nominal 96 GB SKU exposes the same MiB total.

As a capacity proxy only, the same five-second case on one B300 measured a
92,946 MiB whole-device first-request peak and a 92,146 MiB worker peak. Its
encode, diffuse, decode, and client wall times were 8.664 s, 134.504 s,
6.016 s, and 151.583 s. These numbers suggest that a 96 GB card may fit, but
they are not an RTX PRO 6000 validation: kernels, allocator behavior, and
reported device capacity differ across GPUs.

Use `ignored_layers` to keep any otherwise eligible linear in BF16. H3
resolves the `transformer` component before constructing the DiT, so names do
not start with `transformer.`. Entries are exact runtime linear prefixes; a
parent name such as `blocks.0.attn` does not exclude its children.

Eligible names are the `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and
`mlp.fc2` children under `token_refiner.blocks.<0-1>` or `blocks.<0-49>`.
The other eligible names are `condition_proj`,
`blocks.<0-49>.adaln_proj.linear`, and `final_layer.adaln_proj.linear`.
For example, keep the first main block's attention projections in BF16 with:

```bash
--diffusion-quantization-config \
  '{"transformer":{"method":"fp8","ignored_layers":["blocks.0.attn.qkv_proj","blocks.0.attn.out_proj"]}}'
```

The structured option replaces `--quantization fp8`. Online FP8 can be used
with H3 layerwise offload and with either DiT DLO transfer. DiT `allgather`
uses the ordinary loader to finalize FP8 weights and scales before sharding
them across ranks. DiT `rank-local` instead retains complete loader-produced
tensors and avoids the synchronized request-wave contract. H3's TP-sharded
text encoder uses `rank-local`.

## AMD ROCm (gfx942 / gfx950)

The sections above describe the NVIDIA CUDA path. This section covers the AMD
ROCm path; use it instead of the CUDA commands when running on AMD Instinct
GPUs.

MiniMax H3 runs on AMD Instinct GPUs (gfx942 / gfx950) in BF16. Use
`--diffusion-attention-backend FLASH_ATTN`, which resolves to AITER packed varlen
attention on both architectures.

Select the device with `HIP_VISIBLE_DEVICES` (not `CUDA_VISIBLE_DEVICES`), drop the
CUDA-only `FLASHINFER_DISABLE_VERSION_CHECK`, and install without the `[fa4]` extra
(FA4 is CUDA-only). The VAE uses AITER GroupNorm on ROCm.

For this source checkout, build the image with the matching vLLM 0.34.0
ROCm runtime from the Omni repository root:

```bash
docker build -f docker/Dockerfile.rocm -t vllm-omni-rocm:v034-source .
```

See the [ROCm installation guide](../../docs/getting_started/installation/gpu/rocm.inc.md)
for source/runtime details. Container and native H3 validation on ROCm remain
pending for this checkout; an older ROCm wheel cannot provide its 0.34 APIs.

Historical prebuilt image: `vllm/vllm-omni-rocm:minimax-h3`. That image bundles TorchCodec (for image+audio Ref2VA) and `ffmpeg` (for
video-reference Ref2VA).

### ROCm single GPU

Single GPU with model-level CPU offload keeps the Qwen3-VL encoder and DiT from
being co-resident:

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

HIP_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni --host 0.0.0.0 --port "${PORT}" --trust-remote-code \
  --num-gpus 1 --enable-cpu-offload \
  --diffusion-attention-backend FLASH_ATTN
```

This validated ROCm capacity recipe intentionally retains the compatibility
full-topology alias because MiniMax-H3 also stages its VAEs on that path. The
compact API in this release selects only `dit` and `text_encoder`, so replacing
the flag would change residency rather than perform a mechanical migration.
No removal deadline is assigned until the compact API offers equivalent
component coverage.

### ROCm four GPUs

The best-practice CUDA four-GPU configuration works on ROCm with the changes above.
It mirrors the CUDA command including Ulysses sequence parallelism, VAE patch
parallelism, and text-encoder tensor parallelism, with no CPU offload when the model
shards across the GPUs:

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

HIP_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

As on CUDA, H3 is CFG-distilled, so keep `--cfg-parallel-size` at 1, and the H3 VAE
supports only its native `tile` mode. `--text-encoder-tp-size` is validated on
gfx942; on gfx950 it was not exercised.

### Validated ROCm evidence

vLLM-Omni with MiniMax H3 support, BF16. gfx942 rows measured with the
`vllm/vllm-omni-rocm:minimax-h3` image; gfx950 rows measured with the
`0.26.0+rocm723` wheel (HIP 7.2).

| Workload | Configuration | Observed result |
| ---------- | --------------- | ----------------- |
| T2VA, 1344x768, 209 frames, 50 steps | 4x gfx942 (MI300X), FLASH_ATTN, USP4, text-enc TP4, VAE PP4 tile | encode 0.09 s, denoise 244.04 s, decode 4.15 s, 267.42 s client E2E; H.264 24 FPS + 32 kHz stereo AAC |
| FL2VA, 1344x768, 209 frames, 50 steps | 4x gfx942 (MI300X), FLASH_ATTN, USP4, text-enc TP4, VAE PP4 tile | encode 13.98 s, denoise 257.58 s, decode 4.11 s, 287.07 s client E2E; H.264 24 FPS + 32 kHz stereo AAC |
| T2VA, 832x480, ~4 s, 40 steps | 1x gfx950 (MI350), FLASH_ATTN, CPU offload | valid MP4 (H.264 + synced audio); ~0.73 s/denoise-step (~1.37 it/s), ~55 s client E2E incl. warmup |

gfx942 figures are the mean of three requests after one excluded warmup
(external evidence: vllm-project/recipes#732). gfx950 figures are
functional-correctness validations, not tuned throughput; the first request
includes lazy regional compilation. MI325X (gfx942) and other MI355X SKUs are not
listed until their own evidence is added.

## HTTP API examples

The following requests use the synchronous endpoint so the returned body can
be saved directly as an MP4. The asynchronous `POST /v1/videos` endpoint can
also be used when job polling is preferred.

All four tasks use 24 FPS, 50 sigma points, seed values from the validated
workloads, and the checkpoint-reference video/audio flow shifts of 12 and 3.
Decimal durations are passed through `extra_params`.

Set the endpoint once:

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
```

### 1. T2VA: text to video and audio

Run this request against the combined service:

```bash
curl -sS -X POST "${API_URL}" \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

### 2. FL2VA: first frame to video and audio

Run this request against the combined service. When width and height are
omitted, H3 preserves the first-frame aspect ratio and uses a 768-pixel short
edge.

```bash
export FIRST_FRAME=/path/to/fl2va_first_frame.png

curl -sS -X POST "${API_URL}" \
  -F 'prompt=A man stands beside a yellow car at night. The car drives away; he follows it with his eyes and begins singing sadly, with synchronized voice and city ambience.' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=2101' \
  -F 'extra_params={"task":"fl2va","duration":8.7,"audio_flow_shift":3.0}' \
  -F "input_reference=@${FIRST_FRAME};type=image/png" \
  -o fl2va.mp4
```

Use the same `FL2VA` partition for the official tail-keyframe forms. A single
image with `frame_indices=[-1]` conditions the last frame; two ordered images
with `frame_indices=[0,-1]` condition the first and last frames:

```bash
export LAST_FRAME=/path/to/fl2va_last_frame.png
export FIRST_FRAME=/path/to/fl2va_first_frame.png

curl -sS -X POST "${API_URL}" \
  -F 'prompt=The subject moves naturally from the first image to the last image.' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=2102' \
  -F 'extra_params={"task":"fl2va","duration":8.7,"frame_indices":[0,-1],"audio_flow_shift":3.0}' \
  -F "input_references=@${FIRST_FRAME};type=image/png" \
  -F "input_references=@${LAST_FRAME};type=image/png" \
  -o fl2va_first_last.mp4
```

### 3. Ref2VA: image-only, image/audio, or mixed references

Run these requests against the combined service or a Ref2VA-only service.
Reference images keep their original resolution, with dimensions aligned to
32 pixels. No additional request or server setting is needed.

Image-only Ref2VA omits `audio_reference`; adding one or more audio references
is optional. The typed fields accept one object or an ordered JSON list.
`audio_reference` accepts an HTTP(S) URL or a `data:` URL. In one terminal,
expose the local reference assets to the serving host:

```bash
python -m http.server 8092 \
  --bind 127.0.0.1 \
  --directory /path/to/reference_assets
```

Then submit an image-only request from another terminal:

```bash
export REF_IMAGE=/path/to/reference_assets/ref2va_image.png

curl -sS -X POST "${API_URL}" \
  -F 'prompt=A white cat sits on a beige couch and slowly looks toward the camera.' \
  -F 'aspect_ratio=adaptive' \
  -F 'short_edge=768' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3100' \
  -F 'extra_params={"task":"ref2va","duration":8.0,"audio_flow_shift":3.0}' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -o ref2va_image_only.mp4
```

An image-plus-audio request is:

```bash
export REF_IMAGE=/path/to/reference_assets/ref2va_image.png
export AUDIO_URL=http://127.0.0.1:8092/ref2va_audio.mp3

curl -sS -X POST "${API_URL}" \
  -F 'prompt=A white cat with black mustache and eyebrow markings sits on a beige couch, lip-syncing precisely to the complete reference audio before shifting from confusion to deadpan speechlessness.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3101' \
  -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -F "audio_reference={\"audio_url\":\"${AUDIO_URL}\"}" \
  -o ref2va_image_audio.mp4
```

The requested duration should cover the complete audio. If `duration` is
shorter, the reference soundtrack is truncated to the generated clip.

### 4. Ref2VA: video, separate audio, and mixed references

Run this request against the combined service. Repeat the
`input_references` multipart field once per source video. H3 consumes the
videos in form order and preserves their original soundtracks during
conditioning.
Small reference videos retain their original size; larger videos are scaled
down to the reference canvas. Dimensions are aligned to 32 pixels. No
additional request or server setting is needed.

```bash
export SUBJECT_VIDEO=/path/to/green_screen_subject.mp4
export BACKGROUND_VIDEO=/path/to/fairytale_background.mov

curl -sS -X POST "${API_URL}" \
  -F 'prompt=Remove the green screen background of Video 1 and replace it with the fairytale environment from Video 2. Match the background motion to the character actions and relight the character to fit the scene.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3101' \
  -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
  -F "input_references=@${SUBJECT_VIDEO};type=video/mp4" \
  -F "input_references=@${BACKGROUND_VIDEO};type=video/quicktime" \
  -o ref2va_video_video.mp4
```

The server stores uploaded references only for the lifetime of the request and
deletes temporary files after generation. A video may use its embedded
soundtrack, a separate `audio_reference`, or both. To send a mixed multipart
request, repeat `input_references` for each image, video, or audio file; the
server classifies them by MIME type and preserves the per-type order.

Reference videos must be MP4/MOV with H.264/H.265 video, optional AAC/MP3
audio, 2–15 seconds each, and at most 15 seconds combined. They may still be
longer than the generated clip. Use `start_time_seconds` to select a
synchronized segment; for multiple typed video references, pass one value per
video in `extra_params.start_time_seconds`.

Reference images accept JPG/JPEG, PNG, or WEBP up to 30 MiB. Standalone
audio references accept WAV or MP3 up to 15 MiB, with 2–15 seconds per file and
at most 15 seconds combined.

## Fun ControlNet Union (CTRL-01)

The original [Alibaba PAI Fun ControlNet Union checkpoint](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union)
adds one control branch to the server's transformer. It accepts prepared
Canny, Depth, HED, MLSD or Pose videos and mask-based video inpainting. The
control checkpoint contains only branch weights; keep the base FL2VA components
available. ComfyUI-repacked, pruned and quantized control checkpoints are not
supported by this loading path.

Set local checkpoint paths and start the control-enabled server:

```bash
export MODEL=/path/to/MiniMax-H3/FL2VA
export CONTROL_MODEL=/path/to/MiniMax-H3-Fun-Controlnet-Union.safetensors
CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "$MODEL" --omni --task-type fl2va --trust-remote-code \
  --served-model-name MiniMaxAI/MiniMax-H3 --host 127.0.0.1 --port 8092 \
  --controlnet-model-path "$CONTROL_MODEL" \
  --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --usp 1 --ring 1 --enforce-eager --diffusion-attention-backend TORCH_SDPA \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling
```

The initial control path uses resident BF16 weights and tensor parallelism.
Keyframe conditioning, visual timeline guides, continuation (`continue_from` and long-video), sequence parallelism,
approximate cache acceleration and quantized control execution require separate support; do not combine them with
this configuration. The branch hints the task its DiT serves: an FL2VA server (`--task-type fl2va`) takes control on
T2VA requests without references, and a Ref2VA server (`--task-type ref2va`) takes it on Ref2VA requests with their
reference images, as the FastH3 Omni checkpoint does; see [FastH3 Omni control](../FastVideo/FastH3-Omni.md#controlnet).
Reference rows receive no hint. A combined server has two DiTs and one branch, so it is refused at startup, and so
is reference-KV offload (`kv_offload_config`), which compacts reference rows during the loop.
Control/Turbo combinations need their own validation and are not established by ordinary H3 Turbo results.

Active control conditioning cannot be reused in a `latent_refine` pass at a
larger canvas after `latent_upscale`: its hint/source/mask rows still describe
the first-pass resolution. That combination is rejected before denoising.
Upscaling without refinement, same-canvas refinement and zero-strength requests
retain their existing execution paths; control quality for these combinations
has not been validated.

Upload a prepared Canny video through the existing control API:

```bash
curl --fail-with-body http://localhost:8092/v1/videos/sync \
  -F 'model=MiniMaxAI/MiniMax-H3' \
  -F 'prompt=A forest stream with flowing water and birdsong.' \
  -F 'control_type=canny' \
  -F 'control_reference=@canny.mkv;type=video/x-matroska' \
  -F 'extra_params={"task":"t2va","canny":{"control_context_scale":1.0},"audio_flow_shift":3.0}' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' \
  -F 'num_frames=124' -F 'fps=24' -F 'num_inference_steps=40' \
  -F 'guidance_scale=1.0' -F 'flow_shift=12.0' -F 'seed=43' \
  -o control-output.mp4
```

Use `depth`, `hed`, `mlsd` or `pose` for the corresponding prepared hint, and
use that same key in `extra_params`. Selecting a type does not run a
preprocessor. Strength is the model's `control_context_scale`: default 1.0,
finite and nonnegative; 0 disables the control branch.

For inpainting, source and mask have explicit roles:

```bash
curl --fail-with-body http://localhost:8092/v1/videos/sync \
  -F 'model=MiniMaxAI/MiniMax-H3' \
  -F 'prompt=A small wooden bridge crosses the forest stream.' \
  -F 'control_type=inpaint' \
  -F 'source_reference=@source.mp4;type=video/mp4' \
  -F 'mask_reference=@mask.png;type=image/png' \
  -F 'extra_params={"task":"t2va","inpaint":{"control_context_scale":1.0},"audio_flow_shift":3.0}' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' \
  -F 'num_frames=124' -F 'fps=24' -F 'num_inference_steps=40' \
  -F 'guidance_scale=1.0' -F 'flow_shift=12.0' -F 'seed=43' \
  -o inpaint-output.mp4
```

White/1 in the mask means regenerate; black/0 supplies preserved source
conditioning. This is learned conditioning, not a promise of pixel-exact
compositing outside the mask. A static image mask repeats over the target
frames; a temporal mask uses a video. Prefer lossless mask encodings. Keep the
source, control and mask aligned to the same canvas and timeline. The model
fits conditioning to the requested frame count by truncating longer inputs or
holding the final frame of shorter inputs.

A source requires a mask. A mask without a source is supported, using zeroed
source pixels; it does not preserve an unavailable original video. A mask can
also accompany a prepared control video. Do not send the inpainting source as
`video_reference`: that field has Ref2VA semantics.

Both `/v1/videos` and `/v1/videos/sync` share these inputs. Async requests retain
the existing job polling/content interface. Unsupported controls and invalid
combinations fail instead of silently falling back to ordinary generation.
See the [video API](../../docs/serving/videos_api.md) for upload rules and errors.

### Validation status

Local integration checks used two H20-3e GPUs, driver 550.127.08, PyTorch
2.13.0+cu129, vLLM 0.29.0+cu129, Diffusers 0.40.0 and Transformers 5.14.1.
The configuration above used resident weights and TORCH_SDPA. Canny and
inpainting also completed with the native
`minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` adapter: 5 sigma
positions / 4 denoiser forwards, video/audio shifts 6/3, guidance 1, adapter
scale 1, seed 1101, 1344x768, 124 frames and 24 FPS.

These checks predate the step-count alignment: the recorded Base requests used
40 sigma positions / 39 denoiser forwards. The current examples above use
40 forwards; use `num_inference_steps=39` to match that earlier Base comparison.

To reproduce that Turbo comparison, add `--lora-backend peft --lora-path "$TURBO_LORA"`
at startup and activate the same artifact in the request, following the
[LoRA contract](#lora). Change the request to `num_inference_steps=4`,
`flow_shift=6`, and add:

```bash
-F "lora={\"name\":\"h3-turbo-v1.0\",\"path\":\"${TURBO_LORA}\",\"scale\":1.0}"
```

The Canny check used an FFV1 edge video derived from a generated forest clip;
the inpainting check used that source clip and a static PNG mask. Canny
followed the source layout; inpainting introduced a bridge in the masked
region while retaining the surrounding scene. Both output audio and video
fully decoded, but the Turbo control samples had much quieter audio than
the same-seed control-disabled sample. This remained after matching the
reference timestep precision and correcting container timestamp rounding.
A Base 40-step Canny comparison restored the audio signal level, but a
final-code Base 40-step inpainting sample was also nearly silent. Base
inpainting therefore is not established as a workaround. A same-input comparison
with the pinned VideoX-Fun reference runtime also produced nearly silent Base
inpainting audio and no requested footbridge in the inspected frames. These
results are not unique to the Omni API path, but do not isolate a checkpoint,
training, or integration defect. Audio quality validation remains incomplete.
These checks establish those two request paths,
not an exhaustive quality evaluation of all hint types or Turbo variants. The
40-forward examples above use the current API step-count convention. This does not
claim exhaustive coverage of all Base or Turbo conditioning combinations.

Four additional real-model API smokes used synthetic analytic Depth, HED,
MLSD and Pose hints with the same Turbo configuration above (4 forwards).
All returned HTTP 200 with 1344x768 H264 video, 124 frames at 24 FPS, and
32 kHz stereo AAC; full decoding passed. Their audio RMS values were:

| Hint | Audio RMS |
| --- | ---: |
| Depth | 0.02181784 |
| HED | 0.00042317 |
| MLSD | 0.00017059 |
| Pose | 0.00990357 |

These hints were analytic geometry, edges, segments and skeletons, without
learned preprocessing. They establish API/inference smoke coverage, not
real-video fidelity. Prompts and inputs differed from the forest checks;
stronger Depth/Pose signal levels do not isolate an effect of the mode name.
HED/MLSD audio remained weak. Listening and dialogue quality were not verified,
and these results do not resolve the earlier audio findings.

### ComfyUI client boundary

CTRL-01 provides serving; the remote conditioning node and full workflow belong
to CTRL-02 and WF-06. The client integration should extend the existing
Generate Video node with the explicit multipart roles above. Pose preprocessing belongs in the client workflow:

```text
Load Video -> Get Video Components -> SDPose Keypoint Extractor
           -> SDPose Draw Keypoints -> Create Video (24 FPS) -> control video
```

Use the detector and model loader required by the pose extractor. Already
prepared pose videos can skip preprocessing. The server does not load SDPose
or a person detector merely because `control_type=pose` was selected.

## Official input matrix and limits

| Task | Supported references | Limits |
| ------ | ---------------------- | -------- |
| T2VA | text only | prompt must be non-empty |
| FL2VA | first image, last image, or ordered first+last images | at most 2 images; `frame_indices` is `[0]`, `[-1]`, or `[0,-1]` |
| Ref2VA | image-only, image+image, image+video, video+audio, and mixed image/video/audio | images ≤9, videos ≤3, audios ≤3, total references ≤12; audio requires a visual reference |

The default H3 output contract is 4–15 seconds at 24 FPS, stereo 32 kHz audio, and a
32-pixel canvas multiple. T2VA requires one named output ratio from `21:9`,
`16:9`, `4:3`, `1:1`, `3:4`, or `9:16`. FL2VA always follows the first input
image's ratio and ignores a generic `aspect_ratio` override. Ref2VA defaults to
`16:9`; `adaptive` and SGLang's `auto` spelling are accepted aliases for that
default. `short_edge` controls the 768-pixel canvas and must be `768`.
`num_outputs_per_prompt` accepts 1–10 and derives each output seed as
`seed + output_index`. The asynchronous endpoint returns all
outputs; the synchronous raw-MP4 endpoint returns the first output when more
than one is requested.

## Long video and driving audio

Set `extra_params.long_video=true` to opt into durations above 15 seconds.
Ref2VA long-video requests now default to latent continuation with global
temporal RoPE positions (A+B), described below. Set `long_video_mode=full`
explicitly to sample the entire target latent at once for comparison. Both modes
retain the native `17n+5` video-frame grid, `5n+2` video-latent grid, and 40 Hz
audio latents. A request for 75 seconds becomes 1,807 frames (75.292 seconds at
24 FPS). In full mode, attention cost and activation memory grow with the full
sequence length. Reference-video and reference-audio limits remain unchanged.
T2VA and FL2VA keep full mode unless the request sets
`long_video_mode=continuation`. Requests are limited to 30 seconds
in full mode and 300 seconds in continuation mode; without `long_video=true`,
the limit remains 15 seconds. The limits cover duration aliases and explicit
`num_frames`, and externally encoded requests are checked again before diffusion.
Native frame-grid alignment can round the output slightly above the requested
limit. These are request sanity bounds, not a guarantee that a given resolution
fits GPU memory. Full mode still allocates the whole sequence, and continuation
retains cumulative latents and reference inputs.

Use `audio_mode=lock_source` with one `audio_reference` to drive generation
with a soundtrack, rather than treating it as a short reference. The encoder
places that waveform in the target audio latent, crops or zero-pads its two
channels independently to the output duration, and keeps those rows clean
(timestep 1) throughout sampling. This mode does not insert an `<Audio 1>`
reference tag. It supports request and step execution; the default `native`
mode continues generating audio normally. Returned audio is the audio VAE's
reconstruction, not a byte-for-byte copy of the input recording.

Against a Ref2VA server, using the same asset-server setup as above:

```bash
curl --fail-with-body -sS -X POST "${API_URL}" \
  -F 'prompt=<prompt.txt' \
  -F 'width=960' -F 'height=544' -F 'fps=24' \
  -F 'num_inference_steps=50' -F 'flow_shift=12' -F 'seed=19960422' \
  -F 'extra_params={"task":"ref2va","duration":75,"long_video":true,"audio_mode":"lock_source","audio_flow_shift":3,"preencode_mp4":true,"preencode_batch_frames":17}' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -F "audio_reference={\"audio_url\":\"${AUDIO_URL}\"}" \
  -o long-video.mp4
```

`preencode_mp4` decodes and muxes temporal chunks on the worker, avoiding a
full decoded long video in the API process. It does not reduce the denoiser's
sequence length. For Turbo adapters, use the matching adapter's step count
and flow shift from the Turbo table below.

H3 also starts AAC encoding of the decoded audio early, overlapping it with video
VAE decoding.

The [RunningHub workflow](https://www.runninghub.cn/post/2100530217365364737/)
uses T8's `MiniMaxH3AudioConditioningT8`, a full-length latent, and
`MiniMaxH3DualClockSamplerT8`; its `remix_source` strength is zero, equivalent
to source locking. This differs from rolling-overlap continuation nodes.
See [T8 conditioning](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)
and [ComfyUI H3 masking](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py).
The workflow's custom Dasiwa weights, Sol attention, and T8-specific LoRA are
not supplied by this feature.

### Latent-tail continuation with global temporal positions (A+B)

To continue a finished job in a separate request, see
[MiniMax H3 Continuation Across Requests](../../docs/serving/videos_api.md#minimax-h3-continuation-across-requests).

`long_video=true` selects this mode by default for Ref2VA. It can also be selected
explicitly with `long_video_mode=continuation`, which is how T2VA and FL2VA
requests opt in. This experimental mode supports request execution with
uncached denoising (`quality=lossless`); step execution is rejected. Use `long_video_mode=full` explicitly for whole-target
sampling, including requests up to 30 seconds that require step execution.
Continuation currently rejects latent-mask editing (`video_noise_mask` or
`audio_noise_mask` with retained source regions). Full mode supports editing;
`audio_mode=lock_source` cannot be combined with audio latent-mask editing.

```json
{
  "task": "ref2va",
  "duration": 75,
  "long_video": true,
  "long_video_mode": "continuation",
  "continuation_window_frames": 277,
  "continuation_overlap_frames": 22,
  "audio_mode": "lock_source",
  "audio_flow_shift": 3,
  "preencode_mp4": true,
  "preencode_batch_frames": 17
}
```

These defaults sample seven windows for a 1,807-frame output. Each continuation
uses the preceding AV latent tail as fixed condition rows at the new target's
time origin, samples a fresh unmasked target, discards its hidden overlap, and
appends only the new suffix. The reference image and full prompt remain present
in every window. Both window and overlap use the `17n+5` frame grid; the window
must be 107..345 frames and larger than the overlap. A final window may be shorter.
Audio boundaries are rounded on the cumulative 24 FPS / 40 Hz timeline to avoid
drift; a locked driving track is sliced separately for each stereo channel.

Before RoPE evaluation, every window adds `start_frame * 40 / 24` to the temporal
position IDs of its target audio/video, reference audio/video, and latent-tail
guides. The start includes the overlap (not just the newly appended frames).
Text, static image references, spatial coordinates, and padding remain unchanged.
Offsets retain fractional precision independently of rounded audio slice indices.
For the default windows starting at frames 0, 255, 510, the added positions are
0, 425, 850. This follows the media-only global-offset convention in
[LongMedia temporal positioning](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia/blob/main/temporal_positioning.py).
The shifted layout is constructed once per window and then used by all denoising
steps and sequence-parallel ranks; no RoPE-frequency scaling is applied.

For a narrative with different scenes, pass `continuation_prompts` as a JSON
list containing exactly one non-empty prompt string per planned window (seven
strings for the default 75-second request). These prompts are independently
encoded with the same reference assets and selected in window order. Shot times
inside each prompt describe that local window, including its hidden overlap.
This option requires a local text encoder; external-encoder stage execution is
not supported. Ref2VA media positions use one shared origin after the longest
encoded text prefix, so different prompt lengths do not shift the global AV clock;
T2VA and FL2VA windows each follow their own prompt, as separate `continue_from`
requests do. Without this list, the request's single prompt is reused as before.

T2VA and FL2VA requests with more than one window prompt are refused while the
text encoder uses distributed layerwise offload
(`--enable-distributed-layerwise-offload`): their prompts are text-only encodes
that run one after another, which that backend does not handle yet. A single
prompt for all windows works there, and so do Ref2VA window prompts.

T2VA and FL2VA windows carry the preceding AV latent tail as a frame-0 timeline
guide, the layout [`continue_from`](../../docs/serving/videos_api.md#minimax-h3-continuation-across-requests)
uses across requests, instead of a Ref2VA `latent_guide` reference. Each window
gets the sampler inputs the matching request of a `continue_from` chain would.
An FL2VA first frame conditions the first window only: later windows drop the
keyframe and take the prompt encoded without the image, as a `continue_from`
request without an image does, which requires a local text encoder. Any other
keyframe is rejected; use `long_video_mode=full` for those. Nothing re-anchors
the subject after the first window. When a reference has to hold for the whole
clip, use Ref2VA, which keeps its references in every window.

FL2VA windows default to 141 frames instead of 277. Later windows no longer
carry the first frame, and in Ascend runs at 1344x768 with the Turbo 4-step
adapter its look held across joins for 141- and 209-frame windows and was lost at
the first join of a 277-frame window, in a `continue_from` chain as well. Base
weights with the full schedule also held at 277; set `continuation_window_frames`
to use longer windows there.

How well a join holds also depends on the prompt. A prompt that describes a
continuing scene joined cleanly throughout; one that describes a one-off action
can restart it in a later window, for T2VA and FL2VA alike. Describe a scene that
can continue, or give each window its own prompt.

```bash
curl --fail-with-body -sS -X POST "${API_URL}" \
  --form-string 'prompt=A red fox walks slowly through a snowy pine forest at dawn, sniffing the ground, with soft pawsteps, steady winter wind and distant birds.' \
  -F 'width=1344' -F 'height=768' -F 'fps=24' \
  -F 'num_inference_steps=50' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":30,"long_video":true,"long_video_mode":"continuation","audio_flow_shift":3}' \
  -o long-t2va.mp4
```

The cumulative latent is decoded once after all windows. Denoising memory is
bounded by the window, while cumulative latent storage still grows with duration.
This follows the [ComfyUI latent-tail continuation algorithm](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation).
Global positions can still exceed the model's trained temporal range. This
mode does not guarantee seamless cuts or adherence to absolute shot timestamps in
a repeated prompt. For comparison, keep the model, seed, prompt, source media,
steps, shifts, and output size fixed, changing only `long_video_mode`.

For a reproducible approximately 20-second comparison, use a request-mode
Ref2VA server with caching disabled and the same `prompt.txt` and `REF_IMAGE`
for both requests. Use a single prompt without `continuation_prompts` so that
both modes receive identical text. The commands below generate native audio:

```bash
for mode in full continuation; do
  curl --fail-with-body -sS --max-time 14400 "${API_URL}" \
    -F 'prompt=<prompt.txt' \
    -F "input_reference=@${REF_IMAGE};type=image/png" \
    -F 'width=960' -F 'height=544' -F 'fps=24' \
    -F 'num_inference_steps=50' -F 'flow_shift=12' -F 'seed=19960422' \
    -F 'quality=lossless' \
    -F "extra_params={\"task\":\"ref2va\",\"duration\":20,\"long_video\":true,\"long_video_mode\":\"${mode}\",\"audio_flow_shift\":3,\"preencode_mp4\":true}" \
    -o "h3-20s-${mode}.mp4" || break
done
```

Both outputs should contain 481 frames (20.042 seconds at 24 FPS).
Continuation uses windows `[0,277)` and `[255,481)`; inspect frames 276/277
(about 11.54 seconds) for visual and audio discontinuities. Compare the complete
clips for motion, identity, prompt adherence, and sound, not just frame similarity:
the trajectories differ by design. Report the tested commit, checkpoint, hardware,
settings, and both videos with any comparison; the command alone is not quality
evidence. A matching seed does not imply matching noise over differently sized
full and windowed latents.

## Request-scoped sampler

H3 supports `euler` (the default), `res_multistep`, `heun`, and `dpmpp_2m`.
Select the sampler per request; it applies to both video and audio, using their
separate sigma schedules. For the HTTP examples above, set
`num_inference_steps=20` and add `"sampler":"res_multistep"` to the existing
`extra_params` object. For example, the T2VA fields become:

```bash
-F 'num_inference_steps=20' \
-F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0,"sampler":"res_multistep"}'
```

For the offline Python API, use `extra_args`:

```python
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

sampling_params = OmniDiffusionSamplingParams(
    num_inference_steps=20,
    seed=42,
    extra_args={"task": "t2va", "sampler": "res_multistep"},
)
```

`euler`, `res_multistep`, and `dpmpp_2m` use one DiT evaluation per step in
request mode and step execution. `heun` evaluates the DiT a second time at the
Euler proposal and averages the two derivatives, so an N-step schedule ending at
zero uses `2N - 1` evaluations; the final step to zero is a plain Euler update.
Heun needs uncached model runs, so it is rejected with `quality=high` or a
server-wide cache backend. `res_multistep` and `dpmpp_2m` retain the previous
denoised prediction independently for each request and each stream; their first
and final updates use Euler. `dpmpp_2m` extrapolates the prediction linearly in
`-log(sigma)`. Omitting `sampler` or selecting `euler` preserves the existing
Euler update. Reducing the step count reduces model evaluations, but quality
depends on the prompt and configuration; 20 RES steps are a candidate to compare
against an Euler-50 reference.

Requests using a fixed distilled `base_schedule` from a checkpoint or adapter
must keep `sampler="euler"`. RES, Heun, and DPM++ 2M sampling with those
schedules have not been validated and are rejected with `ValueError`.

## Request-scoped quality

Add one of these fields to any HTTP request above. No Cache-DiT startup option
is required; H3 installs its conservative profile when a `high` request
arrives and removes it for a `lossless` request.

```bash
-F 'quality=lossless'  # Native reference path for this request.
-F 'quality=high'      # H3's conservative Cache-DiT profile.
```

Omitting `quality` preserves the startup default: it uses the reference path
normally, or the server-configured profile when the server was started with
`--cache-backend cache_dit`.

Turbo is independent of this quality switch: `quality` selects a Cache-DiT
policy, while Turbo changes the active LoRA weights and sampling schedule. See
[LoRA](#lora) below.

The following result was measured on 4× NVIDIA H200 with SP4, text-encoder
TP4, 1344×768, 124 frames, 24 FPS, and 50 inference steps. One full
`lossless` warmup was excluded, followed by three fixed prompt/seed pairs in
balanced switch order.

| `quality` | Median inference latency | Speedup | SSIM vs `lossless` | PSNR vs `lossless` | Expected trade-off |
| --- | ---: | ---: | ---: | ---: | --- |
| `lossless` | 85.49 s | 1.00× | 1.0000 | exact | Native reference path |
| `high` | 63.36 s | 1.35× | 0.9709 | 34.98 dB | Faster with measured same-seed deviation |

> [!NOTE]
> `high` selects a fixed Cache-DiT profile, but its cache hit rate and
> resulting latency/quality trade-off may vary by hardware, topology, and
> workload. The values above apply to this deployment and are not universal
> guarantees. `lossless` remains the exact reference path.

## Cache-DiT and TaylorSeer

A server-level Cache-DiT profile is selected with `--cache-backend cache_dit`
plus `--cache-config '{...}'`. The optional TaylorSeer calibrator extrapolates
the steps Cache-DiT skips; its state grows with the extrapolation order and
with the number of consecutive cached steps.

The validated profile is `taylorseer_order=2` with
`max_continuous_cached_steps=4` (the MiniMax-H3-NPU recipe's cache config):
an independent controlled run on 910B3 (v0.30.0) kept it clean at 5/10/15 s
outputs (SSIM 0.848/0.861/0.781), while raising `max_continuous_cached_steps`
to 6 with the same order produced the artifact (SSIM 0.798), and a probe of
the same calibrator measured order-2 extrapolation error growing
quadratically with consecutive cached steps
([comment](https://github.com/vllm-project/vllm-omni/issues/8805#issuecomment-6340466984)).
On Ref2VA videos of 10 s and longer this combination shows up as a localized
translucent dark artifact on the subject (same seed, prompt, references, and
dense attention; `main@9778e52bd`, 910C): with
`Fn_compute_blocks=1, Bn_compute_blocks=0, max_warmup_steps=4,
residual_diff_threshold=0.4, max_continuous_cached_steps=6`, the order-2 run
showed a dark shoulder-region smear and a second-half mean-luminance drop
(78.1 → 73.2), while the identical run with `enable_taylorseer=false` stayed
clean and brightened (78.8 → 81.5). The server logs a warning when a profile
leaves this envelope — order 2 with `max_continuous_cached_steps` above 4, or
any order above 2, which has no validated window; for such profiles disable
`enable_taylorseer`. The same report measured no
meaningful end-to-end speedup from TaylorSeer on H3 (122.0 s vs 121.9 s per
video), so treat it as optional. See #8805 for the reproduction data.

On 64 GB-class NPU dies with distributed layerwise offload, Cache-DiT and DLO
together leave under 1 GiB of device headroom (the offload weight pool holds
the full model at sequence-parallel `tp=1`), so cached requests can fail with
an NPU OOM at an arbitrary step. Prefer the uncached path on such
deployments, or reduce `max_continuous_cached_steps`, and track #7618 for the
request-boundary memory release work.

## Exact AdaLN reuse

H3 enables exact AdaLN projection reuse by default, independently of the task,
hardware, sampling schedule, quantization setting, and request `quality`.
This applies to both original H3 (including its default 50-point schedule) and
FastH3; no distilled adapter or explicit few-step ladder is required.
The first occurrence of an input runs the existing projection; later identical
time embeddings reuse its output only while the projection weights and numerical
settings remain unchanged. This does not approximate neighboring timesteps or
change attention, precision, sampling, or the generated audio/video contract.

Each DiT keeps at most 256 MiB of runtime projection outputs. If a long schedule
exceeds that budget, it retains a reusable subset and computes other entries
normally, avoiding cyclic eviction on repeated original-H3 requests. All original weights
remain loaded so new schedules and adapters can compute normally. Adapter changes,
weight reloads, and model device moves invalidate the cache; parameter versions
also guard individual entries. TP ranks coordinate hits before skipping a
projection collective. Gradient-enabled execution, custom linear hooks, and
tensors without weight version counters use the original computation. Both
eager and compiled execution support projection reuse.
Offload paths that replace weight storage can therefore reduce the hit rate.

The runtime cache uses the shared `ExactProjectionCache` implementation; H3 owns
its sidecar adaptation and projection weight offload. Other models can use the same
[projection cache interface](../../docs/design/module/diffusion/diffusion_model_integration.md#exact-conditioning-projection-reuse).
This cache retains projection outputs, not offloaded weights, and does not skip
block weight prefetch.

For an A/B comparison, disable only this reuse at server startup:

```bash
--cache-config '{"minimax_h3_adaln_cache": false}'
```

### Default AdaLN weight offload

AdaLN weight offload is enabled automatically on **SM120** for unquantized BF16
CUDA inference with the default exact result cache. The default uses each
worker's device capability; other GPU architectures keep their existing weight
placement. It reduces GPU memory usage by
keeping projection weights in host memory and reusing exact cached results.
Both compiled and eager execution are supported; no extra enable flag is needed.
Allow additional host RAM for the offloaded weights and pinned staging copies.
Cache misses, including a cold request or a new timestep schedule, transfer
weights to the GPU and synchronize staging; warmed cache hits avoid that work.
Measure both cold latency and warmed throughput for the intended workload.

Other compatible CUDA deployments can opt in explicitly:

```bash
--cache-config '{"minimax_h3_adaln_offload": true}'
```

The automatic default is limited to the tested SM120 path and does not depend
on the attention provider or require FlashInfer.

To keep AdaLN weights on the GPU while retaining result caching:

```bash
--cache-config '{"minimax_h3_adaln_offload": false}'
```

Non-CUDA platforms, quantized models, DiT-wide offload and HSDP retain their
existing weight placement. Disabling the result cache also disables automatic
AdaLN offload. Explicitly enabling it in an incompatible configuration raises
an error.

### Optional offline sidecar

An offline sidecar can seed the first projection results; it is not required to
enable the default cache. Sidecars require native BF16 TP1 math and the same
numerical environment as the builder. Both eager and compiled H3 execution
consult sidecars through the host cache boundary. Uncovered or incompatible
inputs use the original projection instead.
From the repository root, for a fixed FastH3 adapter and its own four-step schedule:

```bash
PYTHONPATH=. python tools/minimax_h3/build_adaln_cache.py \
  --transformer-path "${MODEL_ROOT}/FL2VA/transformer" \
  --model-variant fl2va --mode t2va \
  --fasth3-adapter "${FASTH3_ADAPTER}" \
  --num-inference-steps 4 --flow-shift 12 --audio-flow-shift 3 \
  --device cuda --output /path/to/h3-adaln.safetensors
```

Omit `--fasth3-adapter` for base weights and use the serving step count and shifts.
The builder accepts a native transformer directory with `config.json` and indexed
or single-file safetensors. It streams the required inputs and refuses to overwrite
an existing output. It does not instantiate the full DiT.

Pass the resulting local artifact at server startup:

```bash
--cache-config '{"minimax_h3_adaln_cache_path": "/path/to/h3-adaln.safetensors"}'
```

Combined servers can also use `minimax_h3_ref_adaln_cache_path` for their Ref2VA
transformer. Each artifact binds the model architecture, task, schedule, shifts,
fixed adapter file, effective time-embedding/AdaLN weights, payload checksums, and
numerical environment. A missing, corrupt, incompatible, or outdated sidecar is
rejected with a warning; the model still loads every weight and computes through
the runtime path. A request with different settings similarly falls back.
Build sidecars from trusted local inputs: checksums verify identity and integrity,
not the correctness of an untrusted generator.

Caching and sidecars alone save repeated projection work; GPU residency changes
with weight offload, which is enabled by default on supported SM120 configurations. Gains depend on the workload,
offload behavior, embedding fingerprint cost, and TP coordination overhead.

## Ref2VA dialogue prompts

Ref2VA generates speech from the prompt text. The audio branch follows the
official H3 prompt contract
([h3-prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing))
rather than a serving-layer parser: `vllm_omni` passes the raw prompt to the
H3 text encoder with no `<d>`-mark preprocessing, and the official Ref2VA
processor contains no dialogue parsing either. Prompts that deviate from the
contract surface as audio artifacts (#8905): narrative text outside `<d>`
spoken verbatim, dialogue lines repeated, and filler syllables around the
clip. #8905 reports these artifacts on NPU across the attention and cache
variants it tried, and a separate GPU report shows a bare English prompt
spoken as well; `vllm_omni` forwards the raw prompt to the text encoder
unchanged on every stack, so rewrite the prompt into the official format
first, before changing serve flags.

The contract rules that matter for Ref2VA dialogue:

- Write the description sections in English; keep only dialogue and lyrics in
  their original language, wrapped as `<d>[Language] ...</d>`. Narration in
  other languages outside `<d>` can be spoken aloud.
- Dialogue appears only inside `<d>`. Do not repeat the spoken lines in other
  sections; text outside `<d>` is stage direction the audio branch may
  vocalize.
- Assign one stable speaker ID per vocal source and write
  `<Subject N> (Sx) says, <d>[Language] ...</d>`; do not pack multiple
  speakers into one `<d>` span.
- Keep dialogue punctuation simple (`,`, `.`, `?`, `!`); drop tildes, emoji,
  bullets, and decorative punctuation.

The official skills directory provides rewrite agents that convert free-form
storyboard prompts into this six-section format.

## LoRA

### Turbo LoRA

The nine Diffusers-layout LightX2V Turbo artifacts below are supported. The
filename records the contract, so the server reads the step count, task family
and flow shift from it and validates each request against the artifact that is
loaded. It does not rewrite request or deploy-config sampling values: the
request must carry that artifact's own settings, listed here, or it is
rejected.

| Artifact | Task | Forwards | `num_inference_steps` | `flow_shift` | declared `alpha` |
| --- | --- | ---: | ---: | ---: | ---: |
| `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | T2VA / FL2VA | 4 | 4 | 12 | none -> 8 |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` | T2VA / FL2VA | 4 | 4 | 6 | 128 |
| `minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors` | T2VA / FL2VA | 4 | 4 | 6 | 128 |
| `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` | T2VA / FL2VA | 4 | 4 | 6 | 8 |
| `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` | T2VA / FL2VA | 8 | 8 | 12 | 8 |
| `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` | T2VA / FL2VA | 8 | 8 | 6 | 8 |
| `minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors` | T2VA / FL2VA | 8 | 8 | 12 | 128 |
| `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` | Ref2VA | 4 | 4 | 12 | 8 |
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | Ref2VA | 8 | 8 | 6 | 8 |

`audio_flow_shift` is `3.0` across the family. Each row is the complete
published filename; use it verbatim as `TURBO_FILE` below.

Alpha needs no manual compensation: the server reads it from the artifact's
metadata, falling back to 8 with a warning for
`minimax_h3_fl2v_turbo_4step_v0.1`, the one artifact that declares none. The
request-level `scale` is a further multiplier on top of it.

> [!NOTE]
> Every artifact is rank 128 and the delta is applied at `scale * alpha /
> rank`, so the `alpha=8` rows drive at 1/16 the strength of the `alpha=128`
> rows. 8 is the default of LightX2V's reference script, which never reads the
> metadata; its documented `v0.1` command does not override that default.
>
> `8step_v1.2` is the one artifact that also adapts the timestep modulation:
> it adds a rank-16 pair on the AdaLN projection of each of the 50 DiT blocks
> and of the output layer, and stores an alpha tensor next to every pair. Those
> 51 pairs are applied at their own `scale * alpha / rank` (16 / 16), not at
> the rank-128 pairs' value. It is also the one `_768p` artifact that runs at
> `flow_shift` 12: LightX2V's reference configs use 12 for it and 6 for the
> other `_768p` files.

Every artifact except `minimax_h3_fl2v_turbo_4step_v0.1` also ships a
`_comfyui_` export of the same weights. Those fuse Q/K/V into one projection
and are **not** supported; downloading one is refused by name. Take the
Diffusers file. The filename is the contract, so do not rename an artifact
either -- a renamed file is rejected rather than served on a guess.

FL2VA artifacts serve `t2va` and `fl2va` on any FL2VA or combined server.
Ref2VA artifacts require
`--task-type ref2va`: a combined server serves `ref2va` from a second DiT that
the adapter cannot bind to, so loading one there is refused rather than silently
running an undistilled model on the few-step schedule.

Download the artifact you want:

```bash
export TURBO_DIR=/path/to/minimax-h3-turbo
export TURBO_FILE=minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
hf download lightx2v/Minimax-h3-Turbo "${TURBO_FILE}" --local-dir "${TURBO_DIR}"
export TURBO_LORA="${TURBO_DIR}/${TURBO_FILE}"
```

`--lora-path` accepts one artifact, or a directory holding exactly one.

> [!IMPORTANT]
> This changes earlier behaviour. `--lora-path /path/to/minimax-h3-turbo`
> pointing at a full clone of the Turbo repository used to select the v1.0 768p
> file implicitly; a directory holding several recognized artifacts is now
> rejected as ambiguous. Name the artifact instead:
> `--lora-path /path/to/minimax-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`.

Start from a non-offloaded or DLO FL2VA server command and add
`--task-type fl2va --lora-backend peft --lora-path "${TURBO_LORA}"`.
`--lora-path` preloads the adapter; each request still activates it and must
carry that artifact's sampling settings:

```bash
-F 'num_inference_steps=4' \
-F 'flow_shift=6' \
-F 'extra_params={"task":"t2va","duration":4.4,"audio_flow_shift":3.0}' \
-F "lora={\"name\":\"h3-turbo-v1.0\",\"path\":\"${TURBO_LORA}\",\"scale\":1.0}"
```

Switching to another FL2VA artifact means repointing `TURBO_FILE`, which moves
both `--lora-path` and the request's `lora.path`, and carrying that row's
`num_inference_steps` and `flow_shift`: `8` and `6` for `8step_v1.0_768p`, `8`
and `12` for the 544p `8step_v1.0`. A request that does not match the loaded
artifact is rejected, so a mismatch cannot silently degrade output.

The two `ref2v` rows are not served by this FL2VA command. Start a
`--task-type ref2va` server, take a request from
[Ref2VA](#3-ref2va-image-only-imageaudio-or-mixed-references) and override
`num_inference_steps` and `flow_shift` with that row's values; those examples
already send `audio_flow_shift=3.0`.

For FL2VA, change `task` and add `input_reference` as shown above. This
integration is dynamic-only and does not support prefusion. DLO is
supported by keeping the request-switchable LoRA A/B buffers resident on the
accelerator while DLO streams only the base blocks; budget for this additional
fixed HBM usage. Model-level and standard layerwise offload remain unsupported.
The requested sigma points always number one more than the artifact's denoiser
evaluations.

### FlashGen native LoRA

FlashGen native adapters require `--lora-application runtime` (the default).
`--lora-application merged` is rejected during loading because the adapter's
schedule and request validation require runtime activation. An active native
adapter cannot be stacked with other adapters.

The FlashGen 4-step T2VA artifact uses the native MiniMax-H3 module layout and
declares its distilled sigma schedule in safetensors metadata. It is published on
[ModelScope](https://modelscope.cn/models/FlashGen/Minimax-H3-4step-lora-flashgen):

```text
FlashGen/Minimax-H3-4step-lora-flashgen/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
```

Download only that file:

```bash
python -m pip install modelscope
export FLASHGEN_DIR=/path/to/minimax-h3-flashgen-lora
export FLASHGEN_FILE=minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
modelscope download FlashGen/Minimax-H3-4step-lora-flashgen \
  --local_dir "${FLASHGEN_DIR}" \
  --include "${FLASHGEN_FILE}"
export FLASHGEN_LORA="${FLASHGEN_DIR}/${FLASHGEN_FILE}"
```

Start from a non-offloaded or DLO FL2VA server command and add
`--task-type fl2va --lora-backend peft --lora-path "${FLASHGEN_LORA}"`.
Each request must use T2VA and the distilled interval-count contract:

```bash
-F 'num_inference_steps=4' \
-F 'extra_params={"task":"t2va","duration":5.2}' \
-F "lora={\"name\":\"h3-flashgen-v1.0\",\"path\":\"${FLASHGEN_LORA}\",\"scale\":1.0}"
```

The T2VA v1 artifact rejects Ref2VA and checkpoints that already pin `base_schedule` in
`model_index.json`. The adapter metadata carries
`base_schedule=1.0,0.7,0.4,0.15,0.0`, so `num_inference_steps=4` means four
denoiser evaluations, not five sigma points. Request-mode generation may omit
the field and take the count from the adapter schedule; `--step-execution`
requires it explicitly, because the step scheduler reads the total step count
off the request at admission, before the adapter schedule is known.

DLO is supported in request-mode generation on the same terms as the Turbo
adapter: the request-switchable LoRA A/B buffers stay resident on the
accelerator while DLO streams only the base blocks, so budget for that
additional fixed HBM usage. The native artifact is rank 64 over 259 target
modules, and its packed `qkv_proj` and `fc1` layers reuse the full-input A
tensor per slice while B carries slice-local output rows, so the resident
footprint exceeds the on-disk payload; measure it for your parallel layout
rather than assuming the checkpoint size. Pure Ulysses replicates the adapter
on every rank, while DiT tensor parallelism shards the B buffers. Model-level
and standard layerwise offload remain unsupported, and `--step-execution`
cannot be combined with `--enable-distributed-layerwise-offload`.

To validate a deployment, post the same fixed-seed T2VA request twice with the
adapter and twice without it, then compare the four output digests. The adapter
is bound and deterministic when each pair matches internally and the two pairs
differ from each other.

#### FlashGen native LoRA v2 for I2VA and Ref2VA

Use the published v2 safetensors file exported for the matching official base
partition. For I2VA, start a FL2VA server with `--task-type fl2va` and send a
single first frame in `input_reference`. Set `task=fl2va` and
`frame_indices=[0]` in `extra_params`. For Ref2VA, start a dedicated Ref2VA
server with `--task-type ref2va`; set `task=ref2va` and send the reference
image in `input_reference`. Both services use `--lora-backend peft` and may
preload the published file with `--lora-path`. Select it in the request with
the usual `lora` field and `scale=1.0`. FL2VA exports also load on a
`--task-type combined` server; requests using that adapter must still select
`task=fl2va`.

The [four-A5 conditioning validation](https://github.com/vllm-project/vllm-omni/pull/10596#discussion_r4623182377)
also completed four-step FL2VA requests with a last frame, first and last
frames, and `continue_from`. First/last output frames matched their input
images with SSIM 0.982/0.979 in the tested two-frame case. Three keyframes
and timeline guides were rejected by existing validation. Ref2VA requests
with a video, image plus audio, and image plus video also completed with
fully decodable video and audio. Reference audio/video adherence remains
unverified; these results establish completion for the tested inputs.

The artifact supplies a four-interval base sigma schedule. Omit `sampler`
to use the H3 default Euler sampler. The adapter requires `flow_shift=12`
and `audio_flow_shift=3`. If the base model defaults differ, pass these
values explicitly in `extra_params`. Send `num_inference_steps=4` when using
step execution.

Ref2VA uses rank 384 for packed QKV and rank 128 for the other projections.
The runtime LoRA manager uses a shared buffer rank across wrapped layers. It
rounds 384 up to the next supported rank, 512, so rank-128 projections reserve
four times the LoRA buffer capacity they would need at rank 128. The actual
per-device memory overhead depends on tensor parallelism and the backend.
The exported factors already carry their effective scale. An active v2
adapter requires request `scale=1.0`; other nonzero scales are rejected.
`scale=0` disables the adapter and its sampling constraints. Legacy T2VA v1
request scaling is unchanged.

### FastH3 adapter

The four-step adapter recipe has moved to the [FastH3 section](../FastVideo/README.md):
[FastH3](../FastVideo/FastH3.md).

### Open-H3-Regenerate-2K

[Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)
re-renders a 768P draft from this model at 2560x1440 in three steps. It is a
load-time adapter served as its own deployment; see
[Open-H3-Regenerate-2K — H200](../Efficient-Large-Model/Open-H3-Regenerate-2K-H200.md).

### FastH3 8-Step V2 full checkpoint

See [FastH3 V2 — T2VA](../FastVideo/FastH3-V2.md) in the
[FastH3 section](../FastVideo/README.md) for the eight-forward full checkpoint.

### VDN-H3 checkpoint

[VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) is an 8-step DMD student of
the FL2VA partition. Each DiT block replaces dense self-attention with an exact
softmax over a frame window plus a Video DeltaNet linear branch for the frames
outside it. The `VDNH3_ATTN` backend runs the window; see
[VDN-H3 Hybrid Attention](../../docs/user_guide/diffusion/attention_backends/vdnh3_attn.md)
for the window layout. The release's two LoRA adapters are fused into the H3
weights as they stream in, and the linear-branch weights are attached to the
DiT blocks.

Pass the release as `--lora-path`. Only its `stage-dmd-step-250/` directory
(about 5 GB) is downloaded; a local copy of the release or of that directory
also works:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve MiniMaxAI/MiniMax-H3 --omni \
  --trust-remote-code \
  --task-type fl2va \
  --lora-path OpenVDN/vdn-minimax-h3 \
  --diffusion-attention-backend VDNH3_ATTN \
  --tensor-parallel-size 2 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2
```

Requests use 8 steps and task `t2va` or `fl2va`:

```bash
-F 'num_inference_steps=8' \
-F 'extra_params={"task":"t2va","duration":5}'
```

FL2VA takes a first frame, a last frame, or both, with the same
`frame_indices` forms as
[FL2VA for base H3](#2-fl2va-first-frame-to-video-and-audio).

The student was distilled at 8 steps with H3's default video/audio shifts of
12/3 and the Euler sampler. A request with another step count, a shift
override, a non-Euler `sampler`, or a `lora=` field is rejected. Add
`--quantization fp8` to run the DiT and text-encoder linears in FP8, including
the linear branch's output projection.

`VDNH3_ATTN` and the checkpoint must be selected together, and the server must
use `--task-type fl2va`. VDN-H3 scales with tensor parallelism only: `--usp`
and `--ring` are rejected. Distributed layerwise offload is rejected because it
installs the DiT without the load-time fusion. The startup log confirms the
fusion:

```text
VDN-H3 stage-dmd-step-250: fused 259 LoRA targets, loaded 800 branch tensors
```

Tensor-parallel sizes 1 (with `--enable-cpu-offload`), 2, and 8 were run end to
end on NVIDIA H200 (141 GB, NVLink), with T2VA and with all three FL2VA
keyframe forms at TP2. Cache acceleration, step execution, and
latent upscale/refine have not been run with VDN-H3.

Measured on H200 at 1344x768 and 14.375 s (345 frames, about 104k tokens) for
one T2VA prompt with seed 1000, using the offline API and
`--vae-patch-parallel-size` equal to the TP size. Compilation is on, the first
request is excluded as warm-up, and one request was recorded per row. DiT time
per step is the mean of steps 2-8:

| GPUs | Configuration | DiT time per step | End-to-end |
| ---: | --- | ---: | ---: |
| 2 | VDN-H3, BF16 | 8.2 s | 74.6 s |
| 2 | VDN-H3, FP8 | 7.2 s | 67.9 s |
| 2 | Base H3, `FLASH_ATTN`, 8 steps, BF16 | 15.4 s | 132.4 s |
| 8 | VDN-H3, BF16 | 2.9 s | 28.8 s |
| 8 | VDN-H3, FP8 | 2.8 s | 27.9 s |
| 8 | Base H3, `FLASH_ATTN`, 8 steps, BF16 | 4.4 s | 41.0 s |

The base H3 rows only isolate the attention cost; base H3 is not distilled for
8 steps. For correctness, the first eager DiT forward was replayed through the
OpenVDN reference implementation on identical inputs. The relative L2 error
was 1.6e-2 for video and 1.0e-2 for audio, against 1.2e-2 and 1.4e-2 for base
H3 replayed the same way. Software: Ubuntu 22.04, driver 580.178.04, Python
3.12, PyTorch 2.13.0+cu130, vLLM 0.30.0.

### FastH3 Omni 8-Step V1 (experimental)

See [FastH3 Omni — Ref2VA](../FastVideo/FastH3-Omni.md) in the
[FastH3 section](../FastVideo/README.md) for reference-conditioned generation.

### ComfyUI-style LoRA adapters

Use `--lora-backend peft` to load ComfyUI-style single-file A/B LoRA adapters using the native H3 parameter layout. Provide a server-local `.safetensors` path in the request's `lora` object.

Use a `lora` array to combine multiple adapters, each with its own `scale`. These adapters can also be combined with the supported Turbo LoRA. Start with `scale=1.0` and adjust it to control adapter strength.

Select a specific file when a directory contains multiple adapters. General Diffusers checkpoint conversion is not supported.

### Pika Flex Turbo LoRA

[Pika-Labs/Pika-H3-Flex-Turbo-LoRA](https://huggingface.co/Pika-Labs/Pika-H3-Flex-Turbo-LoRA) fuses Pika's Flex
adapter with a few-step distill into one native-layout LoRA, so it loads through the ComfyUI-style path above. The
files carry no metadata; as with Turbo, the filename is the contract and must not be renamed:

| Artifact | Server | `num_inference_steps` | `flow_shift` |
| --- | --- | ---: | ---: |
| `pika_h3_flex_turbo_<n>step[_draft\|_hd]_ref2va[_lite_r32].safetensors` | `--task-type ref2va` | `n` | 12 |
| `pika_h3_flex_turbo_<n>step[_hd]_fl2va[_lite_r32].safetensors` | `--task-type fl2va` | `n` | 6 |

`audio_flow_shift` is `3.0`. A request with another step count or shift is rejected, whether `lora.path` names the
file or a directory holding only that file. The full files are rank 768 on
the packed QKV projection, which sets the per-layer LoRA buffer rank to 768 for the whole server; the `_lite_r32`
files stay at rank 96.

### PDD acceleration LoRA

[alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs) applies Parallel
Decoding Distillation: a rank-64 LoRA plus 32 copies of both output heads, of which four fuse into the head of each
of eight forwards. The server fuses the heads when it loads the adapter and binds them for the requests that
activate it, so it is selected like any other adapter:

```bash
hf download alibaba-pai/MiniMax-H3-Acc-LoRAs MiniMax-H3-Ref2VA-Acc-8Step.safetensors --local-dir "${PDD_DIR}"
```

Start a `--task-type ref2va` server with `--lora-backend peft` (`--task-type fl2va` for
`MiniMax-H3-FL2VA-Acc-8Step.safetensors`) and send `num_inference_steps=8` with the base model's shifts
(`flow_shift=12`, `audio_flow_shift=3`) and the adapter in `lora` at `scale=1.0`. Each head is the mean velocity
of one block of that schedule, so a request with another scale, a sampler other than `euler`, or `latent_refine` is
rejected. The adapter is refused at load on a combined server and with step execution, `--lora-application merged`
or `--cache-backend tea_cache`: the heads belong to one DiT and are bound for one request at a time. It cannot be
stacked with other adapters in a `lora` array.

Keep the released filename. It names the DiT the adapter was distilled from, and the server refuses a file whose
name does not match its `--task-type`. The file's metadata does not record the DiT, so a renamed or differently
named export (`pdd_ema.safetensors`) is bound to whichever DiT the server runs, without that check.

### Conditioning features with adapters

ControlNet, `continue_from` and long-video continuation, latent-mask editing and timeline guides do not depend on
how a request is sampled, so they are available with every adapter above on its own schedule. On one H200 node
(TP 8, 480x832, 124 frames, seed 20261124, `quality=lossless`), each cell is one request.

Ref2VA server (`--task-type ref2va`), one reference image:

| Weights | Depth control | Guide at frame 96, SSIM to the guide (unguided) | Edit, kept half PSNR | `continue_from`, SSIM across the join (inside a clip) | 20 s `long_video` |
| --- | --- | --- | --- | --- | --- |
| Base Ref2VA, 20 steps | follows the pose | 0.930 (0.777) | 38.2 dB | 0.860 (0.825) | 481 frames |
| LightX2V `ref2v_turbo_8step_v1.0_768p` | follows the pose | 0.914 (0.725) | 37.6 dB | 0.821 (0.838) | 481 frames |
| Pika `8step_hd_ref2va` | follows the pose | 0.936 (0.724) | 38.4 dB | 0.907 (0.865) | 481 frames |
| Pika `4step_ref2va_lite_r32` | follows the pose | 0.879 (0.723) | 36.6 dB | 0.796 (0.806) | 481 frames |
| PAI `Ref2VA-Acc-8Step` | follows the pose | 0.932 (0.747) | 37.9 dB | 0.824 (0.794) | 481 frames |

FL2VA server (`--task-type fl2va`), T2VA requests with `aspect_ratio=9:16`:

| Weights | Depth control | Guide at frame 96, SSIM to the guide (unguided) | Edit, kept half PSNR | `continue_from`, SSIM across the join (inside a clip) | 20 s `long_video` |
| --- | --- | --- | --- | --- | --- |
| Base FL2VA, 20 steps | follows the pose | 0.932 (0.619) | 35.7 dB | 0.858 (0.834) | 481 frames |
| LightX2V `fl2v_turbo_8step_v1.2_768p` | follows the pose | 0.928 (0.600) | 35.4 dB | 0.800 (0.773) | 481 frames |
| Pika `8step_hd_fl2va` | follows the pose | 0.913 (0.565) | 35.9 dB | 0.755 (0.700) | 481 frames |
| PAI `FL2VA-Acc-8Step` | follows the pose | 0.923 (0.625) | 35.6 dB | 0.829 (0.795) | 481 frames |

Depth control was judged by eye against the control clip; the other columns are measured on the returned MP4s. The
guide image is a frame of the base Ref2VA row's unguided clip, which raises that row's unguided figure. These runs
establish that each combination completes and takes effect, not a quality ranking: one prompt, one seed.

## Key parameters

| Parameter | Recommended value | Notes |
| ----------- | ------------------- | ------- |
| `quality` | omitted or `lossless` | Request-level quality intent; `high` dynamically installs H3's conservative Cache-DiT profile |
| `extra_params.force_refresh_step_hint` | omitted | Optional positive 1-based denoising-step hint for an active Cache-DiT request; pair with `extra_params.force_refresh_step_policy`=`once` or `repeat` |
| `task` | `t2va`, `fl2va`, or `ref2va` | Passed in `extra_params`; selects the task-specific DiT |
| `duration` | Workload-specific | Decimal seconds in `extra_params`; converted to H3-compatible frame count |
| `fps` | `24` | H3 output FPS is fixed |
| `num_inference_steps` | `50` | Matches the reference accuracy workloads |
| `extra_params.sampler` | `euler` | Per-request `euler`, `res_multistep`, `heun`, or `dpmpp_2m` for both streams; fixed distilled schedules require Euler; Heun requires uncached runs |
| `flow_shift` | `12` | Video sigma shift |
| `audio_flow_shift` | `3` | Audio sigma shift, passed in `extra_params` |
| `seed` | Task-specific | Use a fixed value for reproducibility |
| `aspect_ratio` | Task-specific | T2VA requires a named ratio; FL2VA follows the input image; Ref2VA defaults to `16:9` |
| `short_edge` | `768` | H3 shape policy requires exactly 768 when `width`/`height` are omitted |
| `num_outputs_per_prompt` | `1` | 1–10; async API returns every output |
| `start_time_seconds` | `0` | Reference-video segment start; use a list in `extra_params` for multiple videos |
| `width`, `height` | Multiples of 32 | Output aspect ratio must be between 1:4 and 4:1 |

## ComfyUI Frontend

Users can also use a ComfyUI frontend to interact with a hosted MiniMax-H3 service. The ComfyUI frontend can run in a separate environment or machine. Refer to [vLLM-Omni ComfyUI Integration](../../docs/features/comfyui.md) for details.

## Validated four-GPU evidence

The four-GPU recommendation was measured on four NVIDIA B300 GPUs with one
excluded warmup followed by three requests.

| Workload                               | Configuration                               | Observed result                      |
| -------------------------------------- | ------------------------------------------- | ------------------------------------ |
| FL2VA, 209 frames, 1248x768            | no offload, U4, VPP4 tile, regional compile | 86.964 s mean HTTP client latency    |
| Two-video Ref2VA, 362 frames, 1344x768 | no offload, U4, VPP4 tile, regional compile | 784.394 s accounted model-stage mean |

These measurements describe the validated shapes rather than a general
throughput guarantee. Multi-video Ref2VA is much slower because the two
reference videos expand both the Qwen3-VL vision sequence and the packed DiT
attention sequence.

## Validated FP8 evidence

With eager DiT/text-encoder TP2 and VAE tiling, the 384x672, 107-frame,
10-step quality case measured LPIPS 0.1156 (limit 0.20), PSNR 23.6316 dB,
audio spectral cosine 0.9589 (minimum 0.80), and audio RMS ratio 0.9342. The
resident per-GPU peak was 68.52 GiB for BF16 and 53.51 GiB for FP8, a 22%
reduction.

For direct human inspection, see the external
[BF16 versus global-FP8 comparison](https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_global_fp8_vs_bf16/),
which includes matched five-second T2VA, I2VA, and Ref2VA videos. The
[comparison sources](https://github.com/lishunyang12/vllm-omni-rankings/tree/main/scripts/minimax_h3_global_fp8_vs_bf16)
also record per-task fidelity metrics and provenance without storing generated
media in this repository.

## TeaCache acceleration

TeaCache reuses DiT block residuals across denoising steps when consecutive
timestep embeddings are similar. MiniMax-H3 TeaCache is currently calibrated
only for the FL2VA partition. In combined serving, FL2VA requests use TeaCache
while Ref2VA requests run uncached; Ref2VA-only serving rejects TeaCache.

TeaCache and Cache-DiT are mutually exclusive; pick one cache backend per server.

The model-specific default and examples use `rel_l1_thresh=0.17`, which
provided the best conservative speed/quality balance in the validated 107-frame
T2VA workload. Lower values
may produce few or no cache hits, while higher values can improve performance
at the cost of output quality. Validate the threshold on representative
prompts and generation settings before changing it.

### Offline (Python API)

Export the directory containing the `FL2VA` and `Ref2VA` subdirectories before
running the Python example, for example `export MODEL_ROOT=/models/MiniMax-H3`.

```python
import os

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model=os.path.join(os.environ["MODEL_ROOT"], "FL2VA"),
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.17},
    trust_remote_code=True,
    enable_cpu_offload=True,
)
outputs = omni.generate(
    "A quiet cinematic night scene with matching ambient sound.",
    OmniDiffusionSamplingParams(
        height=256,
        width=448,
        num_frames=29,
        fps=24,
        num_inference_steps=50,
        seed=42,
        extra_args={
            "task": "t2va",
            "duration": 4.0,
            "aspect_ratio": "16:9",
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
    ),
)
```

### Online serving

```bash
vllm serve "${MODEL_ROOT}/FL2VA" \
  --omni \
  --trust-remote-code \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh":0.17}'
```

## Latent super-resolution

H3's video VAE is a ~5B-parameter model, so the usual way to raise output
resolution -- decode, upscale in pixel space, re-encode -- pays for that VAE
twice. The community
[Minimax H3 latent upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
is a learned 3D-convolutional resizer trained on H3's 24-channel latents: it
runs between the denoise loop and the VAE, so the clip is decoded once, at the
larger size. Unlike bilinear or bicubic interpolation of a latent, it does not
introduce ghosting or double-image artifacts.

The network runs one normalization below the pipeline latent: an H3 latent in
vLLM-Omni is already normalized, and the upscaler was trained on that tensor
normalized again by the VAE's `latents_mean` / `latents_std`. The stage applies
and undoes that itself, so nothing about a request changes -- but a port that
skips it inflates the latent about 5x and decodes as a magenta grid at one tile
per latent cell.

The checkpoint is optional and is not part of the H3 release. Download one file
from that repository (`minimax_h3_latent_upscaler_3d_bf16.safetensors`, 691 MB)
and name it at startup:

```bash
vllm serve "${MODEL_ROOT}/FL2VA" \
  --omni \
  --trust-remote-code \
  --additional-config '{"latent_upscaler_path":"/models/minimax_h3_latent_upscaler_3d_bf16.safetensors"}'
```

A request then asks for a size. The three modes are mutually exclusive and all
resolve to the same 16-pixel latent grid, so the three requests below are the
same 960x544 -> 1920x1088 upscale:

```bash
curl -sS -X POST "${API_URL}" \
  -F 'prompt=...' \
  -F 'width=960' -F 'height=544' -F 'fps=24' \
  -F 'extra_params={"task":"t2va","duration":8.7,"latent_upscale":2.0}' \
  -o t2va_1080p.mp4

# or name the output size in pixels
#   "latent_upscale":{"width":1920,"height":1088}
# or name a pixel budget and keep the aspect ratio
#   "latent_upscale":{"megapixels":2.0}
```

Over HTTP the field is `extra_params`: the videos request model declares that
name and nothing else, so an `extra_args` form field is dropped without an
error and the request silently returns the un-upscaled size. Offline, the same
value goes in `extra_args`:

```python
omni = Omni(
    model=os.path.join(os.environ["MODEL_ROOT"], "FL2VA"),
    trust_remote_code=True,
    additional_config={"latent_upscaler_path": "/models/minimax_h3_latent_upscaler_3d_bf16.safetensors"},
)
outputs = omni.generate(
    "A quiet cinematic night scene with matching ambient sound.",
    OmniDiffusionSamplingParams(
        height=544,
        width=960,
        fps=24,
        num_inference_steps=50,
        seed=42,
        extra_args={"task": "t2va", "duration": 8.7, "latent_upscale": 2.0},
    ),
)
```

### The hi-res route: generate small, refine large

Upscaling alone decodes the enlarged latent directly, which is fast but can
only interpolate detail the first pass never generated. Adding `latent_refine`
resumes the denoise schedule at the new size, so the DiT puts real detail back:

1. generate at a cheap size -- far fewer DiT tokens,
2. upscale the latent,
3. re-noise it and run the tail of the schedule at the target size.

```bash
curl -sS -X POST "${API_URL}" \
  -F 'prompt=...' \
  -F 'width=960' -F 'height=544' -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'extra_params={"task":"t2va","duration":8.7,"latent_upscale":2.0,"latent_refine":0.4}' \
  -o t2va_1080p.mp4
```

`latent_refine` is the fraction of the request's steps the second pass re-runs,
the img2img convention, so it sets both how much the pass may change and what
it costs: `0.4` of `num_inference_steps=50` is 50 cheap steps followed by 20
expensive ones. Low values stay close to the first pass and mostly sharpen;
high values re-generate and may drift from it. `1.0` re-runs the whole schedule
from noise, which discards the first pass entirely.

Detail does not keep climbing with strength. Measured on a 15s 2688x1536 clip,
six frames apart, against the same seed and prompt: 0.5 returned 0.98x the
high-frequency energy of 0.35 and 0.65 returned 0.92x, while the mean per-pixel
drift from the 0.35 result rose from 6.3 to 12.6 (0-255). What caps detail is
what the DiT can resolve at the target size, not how noisy a start it is given;
past that, a higher strength only erases more of the first pass and re-imagines
it from less. On this checkpoint 0.35 was the best of the three -- treat higher
values as "re-generate", not as "sharpen".

Why this is faster than generating at the target size: the DiT cost follows the
video token count, which is quadratic in the latent area for attention. The
example above is 31,620 video tokens at 960x544 against 126,480 at 1920x1088 --
4x the tokens, so roughly 4x the linear cost and up to 16x the attention cost
per step. Paying that on 20 steps instead of 50 is where the saving comes from.
The gain therefore depends on the resolutions, the strength, and how
attention-bound the shape is; measure it on your own workload.

Audio is re-noised and refined alongside the video at the same schedule
position. The two modalities are shifted apart (`flow_shift` 12.0 against
`audio_flow_shift` 3.0), so each resumes at its own sigma for that position --
the state the first pass actually passed through, which is what keeps the pass
in distribution.

`latent_refine` works without `latent_upscale` too, as a plain detail pass at
the generated size. For FL2VA, the keyframe condition is pinned to the output
latent size, so a refine after an upscale re-encodes the keyframes from the
originals at the new size; REF2VA references carry their own shapes and are
reused as they are.

### Options

| `--additional-config` key | Default | Meaning |
| --- | --- | --- |
| `latent_upscaler_path` | unset | Checkpoint file, or a directory holding exactly one. Unset disables the stage and loads no extra weights. |
| `latent_upscaler_dtype` | the engine dtype | `bf16`, `fp16` or `fp32`. |
| `latent_upscaler_chunk_frames` | `0` | `0` runs the whole clip in one pass; a positive value bounds activation memory with approximate temporal chunking. |
| `latent_upscaler_resident` | `false` | Keep the ~700 MB of weights on the GPU between requests instead of in host memory. |
| `latent_refine_max_tokens_per_rank` | `65536` | Reject larger refine layouts before denoising. Set `0` only after validating a larger deployment. |
| `latent_upscale` | unset | A default target applied to every request, in the request's own format. A request overrides it, and `false` opts out. |
| `latent_refine` | unset | A default refine strength applied to every request. A request overrides it, and `false` opts out. |
| `latent_cache_clips` | `16` | Finished clips whose latents each DiT worker keeps for `continue_from`. `0` disables continuation. |

The `latent_upscale` request value is a multiplier (`2.0`), `false`, or an
object naming one mode: `{"scale": 2.0}`, `{"width": …, "height": …}` or
`{"megapixels": …}`, each accepting an `align` (default `32` pixels). The
checkpoint was trained between 1.0x and 4.0x and only upscales; a smaller
target is rejected, and `scale` 1.0 is a no-op that skips the stage. The
`latent_refine` value is a strength in `(0, 1]`, `false`, or
`{"strength": 0.4}`; it needs no checkpoint of its own.

When `latent_refine` is enabled, both resolved output dimensions must be
divisible by 32 pixels for the DiT's spatial patches. A custom `align` that
produces an incompatible target is rejected before denoising; upscale-only
requests can still use the VAE's 16-pixel grid. Numeric upscale fields must be
positive and finite, and `width`, `height`, and `align` must be integers.

Temporal chunking can bound activation memory on long clips -- a 15-second clip is
102 latent frames, more than three 32-frame chunks. It is an approximation rather than a
partition: the network's GroupNorms pool statistics over whatever clip they are
handed, so a chunked pass can differ from a single pass across the whole output
and not only at the chunk seams. How much that costs in practice was not
measured here; the default single pass buys the exact result for the
memory a full-length activation volume takes (10.1 GiB against 5.6 GiB at
2688x1536 for 15 seconds, measured on one B300). Set
`latent_upscaler_chunk_frames: 32` explicitly if that memory cost is too high.

## Known limitations

- TeaCache is calibrated for FL2VA only; Ref2VA requests run uncached.
- The latent upscaler is a community checkpoint, not part of the H3 release,
  and is unavailable unless `latent_upscaler_path` names one. It runs
  replicated on every DiT rank rather than sharded.
- `latent_refine` is request-mode only. It is a second denoise loop with its
  own resolution and its own truncated schedule, and the step contract carries
  one schedule per request, so step execution rejects it. `latent_upscale`
  without `latent_refine` works in both modes.
- Non-empty attention step ranges reject `latent_refine` in request mode too;
  disable them (`attention_schedule=[]`) or refinement (`latent_refine=false`).
  Sigma windows work with `latent_refine`.
- Large refine layouts can abort workers with `CUDA error: an illegal memory
  access was encountered`. On B300, observed video token counts of 54,144 per
  rank complete while 108,288 and 109,360 abort. The preflight limit of 65,536
  counts the full padded layout, including text, audio, and visual references,
  per Ulysses rank. It rejects the observed failing sizes before the base pass.
  This is an empirical B300 guard, not a proven kernel limit on every device;
  deployments with different hardware can adjust the limit after validation.
- Combined serving requires sibling `FL2VA` and `Ref2VA` directories, loads
  both task-specific DiTs, and loads shared components once from `FL2VA`.
- Request mode executes one generation request per diffusion batch. Use
  `--step-execution` with `--max-num-seqs N` to admit several requests at once
  (see
  [Execution modes](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/execution_modes.md));
  step
  mode does not support `cache_backend`.
- The first regional-compile request is a warmup and should not be included in
  steady-state performance measurements.
- Ref2VA accepts up to 9 images, 3 video clips, and 3 audio clips, with at most
  12 references in total. At least one image or video is required; audio-only
  requests are rejected.
- The 768 px short-edge mode is available for T2VA and FL2VA; 1344x768 is the
  documented 16:9 request shape.
- `--cfg-parallel-size > 1` is rejected by design (CFG-distilled, no negative branch).
- VAE patch parallelism requires size 1 or the full DiT group size and supports the
  H3 native `tile` mode only.
- Ulysses x Ring hybrid attention supports H3's single-request contiguous suffix
  padding. Arbitrary attention masks and multi-request packed batches remain
  unsupported on the Ring path.
- Online FP8 with DLO AllGather temporarily materializes the complete FP8 model
  in host memory on every rank during startup before retaining only each rank's
  shard. Size startup host memory for that transient peak.
- TeaCache and Cache-DiT cannot be enabled on the same server.
- Pure Ulysses still replicates the full DiT on every rank, so smaller-memory GPUs
  cannot use `--usp N --tp 1` as a resident capacity path. Use DiT tensor parallelism
  or model-level CPU offload; text-encoder TP alone is not sufficient.

## Additional resources

- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
- [Diffusion parallelism](../../docs/user_guide/diffusion/parallelism/overview.md)


---

## FILE: recipes/models/MiniMaxAI/MiniMax-H3.yaml (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/models/MiniMaxAI/MiniMax-H3.yaml)

```
meta:
  title: "MiniMax-H3"
  slug: "minimax-h3"
  provider: "MiniMax"
  description: "Open-weight general-purpose multimodal generation model — jointly generates 24 FPS video with native stereo audio from text, image, video, and audio references, served via vLLM-Omni"
  date_added: 2026-08-02
  date_updated: 2026-10-21
  tasks:
    - omni
  performance_headline: "FastH3 VSA generates 15 s of 1344×768 video with synchronized stereo audio in 10.8 s server-side on 8×B300"
  default_hardware: b300
  related_recipes:
    - Wan-AI/Wan2.2-T2V-A14B-Diffusers
    - nvidia/Cosmos3-Super-Image2Video
  hardware:
    gb200: verified
    b300: verified
    dgx_spark_gb10: verified
    rtx_4090_2x: verified
    rtx_pro_5000_4x: verified
    rtx_5090_2x: verified
    rtx_pro_6000_2x: verified
    mi300x: verified
    mi325x: verified
    mi355x: verified
    ascend_950pr: verified

model:
  model_id: "MiniMaxAI/MiniMax-H3"
  min_vllm_version: "0.26.0"
  docker_image:
    nvidia: "vllm/vllm-omni:minimax-h3"
    amd: "vllm/vllm-omni-rocm:minimax-h3"
  # The published H3 images predate the modular pipeline merged in #5720.
  # Docker binds current source read-only and prepends it to PYTHONPATH.
  docker_source:
    mount_path: "/path/to/vllm-omni"
  install:
    docker:
      note: "The H3 images bundle CUDA/ROCm and media dependencies, but predate the merged modular MiniMax-H3 pipeline. Clone current vLLM-Omni main; Docker mounts it read-only and prepends it to PYTHONPATH."
    pip:
      command: |
        uv venv
        source .venv/bin/activate
        uv pip install --torch-backend=cu130 vllm==0.33.0
      note: "H3 also needs current vLLM-Omni main for the modular FL2VA/Ref2VA pipeline — see the extra install block below."
  architecture: dense
  parameter_count: "64B"
  active_parameters: "64B"
  # Diffusion pipeline — there is no autoregressive KV context. The Qwen3-VL
  # presentation ranges from ~1.8k tokens (text-only) to ~16.3k (two video refs).
  context_length: 0
  # The validated four-GPU best-practice profile uses no offload, Ulysses SP 4,
  # native tiled VAE patch parallelism 4, Ring/TP 1, and the datacenter
  # Blackwell default TRTLLM_ATTN. Keep regional torch.compile enabled.
  base_args:
    - "--trust-remote-code"
    - "--host"
    - "0.0.0.0"
    - "--port"
    - "8000"
    - "--num-gpus"
    - "4"
    - "--usp"
    - "4"
    - "--ring"
    - "1"
    - "--vae-patch-parallel-size"
    - "4"
    - "--vae-parallel-mode"
    - "tile"
    - "--vae-use-tiling"
  base_env:
    VLLM_WORKER_MULTIPROC_METHOD: "spawn"
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT: "1800"

omni:
  port: 8000
  task_label: "Request Mode"
  task_hint: "Select a ready-to-run MiniMax-H3 request preset. High-memory hardware keeps both DiTs loaded; memory-constrained hardware automatically starts the matching FL2VA-only or Ref2VA-only service."
  # The modular pipeline serves the checkpoint root and shares the encoder and
  # VAEs across FL2VA/Ref2VA. Memory-constrained hardware overrides select one
  # DiT with --task-type while retaining the same root model path.
  tasks:
    - id: t2v
      key: t2va
      label: "T2VA - Text"
      model_id: "/path/to/MiniMax-H3"
      vram_minimum_gb: 202
      endpoint: "/v1/videos/sync"
      description: "Text prompt to joint video and synchronized audio through the FL2VA DiT."
      benchmark: |
        python3 /path/to/vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py \
          --model /path/to/MiniMax-H3 \
          --host localhost \
          --port 8000 \
          --endpoint /v1/videos \
          --dataset random \
          --task t2v \
          --num-prompts 3 \
          --max-concurrency 1 \
          --warmup-requests 1 \
          --warmup-concurrency 1 \
          --seed 42 \
          --random-request-config '[{"width":1344,"height":768,"prompt":"Cinematic drone shot pushing through misty mountain ridges at sunrise, golden light catching the clouds, slow and smooth.","num_inference_steps":8,"num_frames":209,"fps":24,"weight":1}]' \
          --extra-body '{"flow_shift":12.0,"extra_params":"{\"task\":\"t2va\",\"duration\":8.7,\"aspect_ratio\":\"16:9\",\"audio_flow_shift\":3.0}"}'
      hardware_overrides:
        rtx_pro_5000_4x:
          vram_minimum_gb: 288
          extra_args: ["--task-type", "fl2va"]
        rtx_pro_6000_2x:
          vram_minimum_gb: 192
          extra_args: ["--task-type", "fl2va"]
        rtx_5090_2x:
          vram_minimum_gb: 64
          extra_args: ["--task-type", "fl2va"]
        rtx_4090_2x:
          vram_minimum_gb: 48
          extra_args: ["--task-type", "fl2va"]
          curl: |
            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'aspect_ratio=16:9' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=1101' \
              -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
              -o t2va.mp4
        dgx_spark_gb10:
          vram_minimum_gb: 128
          extra_args: ["--task-type", "fl2va"]
          curl: |
            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'aspect_ratio=16:9' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=1101' \
              -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
              -o t2va.mp4
        ascend_950pr:
          vram_minimum_gb: 128
          extra_args: ["--task-type", "fl2va"]
          curl: |
            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'aspect_ratio=16:9' \
              -F 'fps=24' \
              -F 'num_inference_steps=60' \
              -F 'flow_shift=12' \
              -F 'seed=1101' \
              -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
              -o t2va.mp4
      curl: |
        curl -sS -X POST http://localhost:8000/v1/videos/sync \
          -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
          -F 'width=1344' \
          -F 'height=768' \
          -F 'aspect_ratio=16:9' \
          -F 'fps=24' \
          -F 'num_inference_steps=50' \
          -F 'flow_shift=12' \
          -F 'seed=1101' \
          -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
          -o t2va.mp4

    - id: i2v
      key: fl2va_first
      label: "FL2VA - First Frame"
      model_id: "/path/to/MiniMax-H3"
      vram_minimum_gb: 202
      endpoint: "/v1/videos/sync"
      description: "First frame and prompt to video and audio. Omitting width and height keeps the input aspect ratio at a 768 px short edge."
      benchmark: &minimax_h3_fl2va_benchmark |
        python3 /path/to/vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py \
          --model /path/to/MiniMax-H3 \
          --host localhost \
          --port 8000 \
          --endpoint /v1/videos \
          --dataset random \
          --task ti2v \
          --num-prompts 3 \
          --max-concurrency 1 \
          --warmup-requests 1 \
          --warmup-concurrency 1 \
          --num-input-images 1 \
          --seed 42 \
          --random-request-config '[{"width":1344,"height":768,"prompt":"Continue the scene: the camera slowly pushes in on the subject as the background city lights drift and blur, with the actor singing softly and emotionally.","num_inference_steps":8,"num_frames":209,"fps":24,"weight":1}]' \
          --extra-body '{"flow_shift":12.0,"extra_params":"{\"task\":\"fl2va\",\"duration\":8.7,\"audio_flow_shift\":3.0}"}'
      hardware_overrides:
        rtx_pro_5000_4x:
          vram_minimum_gb: 288
          extra_args: ["--task-type", "fl2va"]
        rtx_pro_6000_2x:
          vram_minimum_gb: 192
          extra_args: ["--task-type", "fl2va"]
        rtx_5090_2x:
          vram_minimum_gb: 64
          extra_args: ["--task-type", "fl2va"]
        rtx_4090_2x:
          vram_minimum_gb: 48
          extra_args: ["--task-type", "fl2va"]
          curl: |
            export FIRST_FRAME=/path/to/fl2va_first_frame.png

            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=A man stands beside a yellow car at night. The car drives away; he follows it with his eyes and begins singing sadly, with synchronized voice and city ambience.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=2101' \
              -F 'extra_params={"task":"fl2va","duration":5.0,"audio_flow_shift":3.0}' \
              -F "input_reference=@${FIRST_FRAME};type=image/png" \
              -o fl2va_first.mp4
        dgx_spark_gb10:
          vram_minimum_gb: 128
          extra_args: ["--task-type", "fl2va"]
          curl: |
            export FIRST_FRAME=/path/to/fl2va_first_frame.png

            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=A man stands beside a yellow car at night. The car drives away; he follows it with his eyes and begins singing sadly, with synchronized voice and city ambience.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=2101' \
              -F 'extra_params={"task":"fl2va","duration":5.0,"audio_flow_shift":3.0}' \
              -F "input_reference=@${FIRST_FRAME};type=image/png" \
              -o fl2va_first.mp4
      curl: |
        export FIRST_FRAME=/path/to/fl2va_first_frame.png

        curl -sS -X POST http://localhost:8000/v1/videos/sync \
          -F 'prompt=A man stands beside a yellow car at night. The car drives away; he follows it with his eyes and begins singing sadly, with synchronized voice and city ambience.' \
          -F 'fps=24' \
          -F 'num_inference_steps=50' \
          -F 'flow_shift=12' \
          -F 'seed=2101' \
          -F 'extra_params={"task":"fl2va","duration":8.7,"audio_flow_shift":3.0}' \
          -F "input_reference=@${FIRST_FRAME};type=image/png" \
          -o fl2va_first.mp4

    - id: i2v
      key: fl2va_first_last
      label: "FL2VA - First + Last"
      model_id: "/path/to/MiniMax-H3"
      vram_minimum_gb: 202
      endpoint: "/v1/videos/sync"
      description: "Ordered first and last keyframes plus prompt to video and audio through the FL2VA DiT."
      benchmark: *minimax_h3_fl2va_benchmark
      hardware_overrides:
        rtx_pro_5000_4x:
          vram_minimum_gb: 288
          extra_args: ["--task-type", "fl2va"]
        rtx_pro_6000_2x:
          vram_minimum_gb: 192
          extra_args: ["--task-type", "fl2va"]
        rtx_5090_2x:
          vram_minimum_gb: 64
          extra_args: ["--task-type", "fl2va"]
        rtx_4090_2x:
          vram_minimum_gb: 48
          extra_args: ["--task-type", "fl2va"]
          curl: |
            export FIRST_FRAME=/path/to/fl2va_first_frame.png
            export LAST_FRAME=/path/to/fl2va_last_frame.png

            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=The subject moves naturally from the first image to the last image, with synchronized ambience and sound.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=2102' \
              -F 'extra_params={"task":"fl2va","duration":5.0,"frame_indices":[0,-1],"audio_flow_shift":3.0}' \
              -F "input_references=@${FIRST_FRAME};type=image/png" \
              -F "input_references=@${LAST_FRAME};type=image/png" \
              -o fl2va_first_last.mp4
        dgx_spark_gb10:
          vram_minimum_gb: 128
          extra_args: ["--task-type", "fl2va"]
          curl: |
            export FIRST_FRAME=/path/to/fl2va_first_frame.png
            export LAST_FRAME=/path/to/fl2va_last_frame.png

            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=The subject moves naturally from the first image to the last image, with synchronized ambience and sound.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=2102' \
              -F 'extra_params={"task":"fl2va","duration":5.0,"frame_indices":[0,-1],"audio_flow_shift":3.0}' \
              -F "input_references=@${FIRST_FRAME};type=image/png" \
              -F "input_references=@${LAST_FRAME};type=image/png" \
              -o fl2va_first_last.mp4
      curl: |
        export FIRST_FRAME=/path/to/fl2va_first_frame.png
        export LAST_FRAME=/path/to/fl2va_last_frame.png

        curl -sS -X POST http://localhost:8000/v1/videos/sync \
          -F 'prompt=The subject moves naturally from the first image to the last image, with synchronized ambience and sound.' \
          -F 'fps=24' \
          -F 'num_inference_steps=50' \
          -F 'flow_shift=12' \
          -F 'seed=2102' \
          -F 'extra_params={"task":"fl2va","duration":8.7,"frame_indices":[0,-1],"audio_flow_shift":3.0}' \
          -F "input_references=@${FIRST_FRAME};type=image/png" \
          -F "input_references=@${LAST_FRAME};type=image/png" \
          -o fl2va_first_last.mp4

    - id: ti2v
      key: ref_image
      label: "Ref2VA - Image"
      model_id: "/path/to/MiniMax-H3"
      vram_minimum_gb: 202
      endpoint: "/v1/videos/sync"
      description: "Reference image and prompt to video and audio through the Ref2VA DiT."
      benchmark: &minimax_h3_ref2va_image_benchmark |
        python3 /path/to/vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py \
          --model /path/to/MiniMax-H3 \
          --host localhost \
          --port 8000 \
          --endpoint /v1/videos \
          --dataset random \
          --task ti2v \
          --num-prompts 3 \
          --max-concurrency 1 \
          --warmup-requests 1 \
          --warmup-concurrency 1 \
          --num-input-images 1 \
          --seed 42 \
          --random-request-config '[{"width":1344,"height":768,"prompt":"A white cat sits on a beige couch and slowly looks toward the camera as quiet room ambience continues.","num_inference_steps":8,"num_frames":209,"fps":24,"weight":1}]' \
          --extra-body '{"flow_shift":12.0,"extra_params":"{\"task\":\"ref2va\",\"duration\":8.7,\"audio_flow_shift\":3.0}"}'
      hardware_overrides:
        rtx_pro_5000_4x:
          vram_minimum_gb: 288
          extra_args: ["--task-type", "ref2va"]
        rtx_pro_6000_2x:
          vram_minimum_gb: 192
          extra_args: ["--task-type", "ref2va"]
        rtx_5090_2x:
          vram_minimum_gb: 64
          extra_args: ["--task-type", "ref2va"]
        rtx_4090_2x:
          vram_minimum_gb: 48
          extra_args: ["--task-type", "ref2va"]
          curl: |
            export REF_IMAGE=/path/to/reference_assets/ref2va_image.png

            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=A white cat sits on a beige couch and slowly looks toward the camera as quiet room ambience continues.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=3100' \
              -F 'extra_params={"task":"ref2va","duration":5.0,"audio_flow_shift":3.0}' \
              -F "input_reference=@${REF_IMAGE};type=image/png" \
              -o ref2va_image.mp4
        dgx_spark_gb10:
          vram_minimum_gb: 128
          extra_args: ["--task-type", "ref2va"]
          curl: |
            export REF_IMAGE=/path/to/reference_assets/ref2va_image.png

            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=A white cat sits on a beige couch and slowly looks toward the camera as quiet room ambience continues.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=3100' \
              -F 'extra_params={"task":"ref2va","duration":5.0,"audio_flow_shift":3.0}' \
              -F "input_reference=@${REF_IMAGE};type=image/png" \
              -o ref2va_image.mp4
      curl: |
        export REF_IMAGE=/path/to/reference_assets/ref2va_image.png

        curl -sS -X POST http://localhost:8000/v1/videos/sync \
          -F 'prompt=A white cat sits on a beige couch and slowly looks toward the camera as quiet room ambience continues.' \
          -F 'aspect_ratio=adaptive' \
          -F 'short_edge=768' \
          -F 'fps=24' \
          -F 'num_inference_steps=50' \
          -F 'flow_shift=12' \
          -F 'seed=3100' \
          -F 'extra_params={"task":"ref2va","duration":8.0,"audio_flow_shift":3.0}' \
          -F "input_reference=@${REF_IMAGE};type=image/png" \
          -o ref2va_image.mp4

    - id: ti2v
      key: ref_image_audio
      label: "Ref2VA - Image + Audio"
      model_id: "/path/to/MiniMax-H3"
      vram_minimum_gb: 202
      endpoint: "/v1/videos/sync"
      description: "Reference image and separate audio plus prompt to video and audio through the Ref2VA DiT."
      benchmark: |
        # Serve /path/to/reference_assets on port 8092 before running this benchmark.
        python3 /path/to/vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py \
          --model /path/to/MiniMax-H3 \
          --host localhost \
          --port 8000 \
          --endpoint /v1/videos \
          --dataset random \
          --task ti2v \
          --num-prompts 3 \
          --max-concurrency 1 \
          --warmup-requests 1 \
          --warmup-concurrency 1 \
          --num-input-images 1 \
          --seed 42 \
          --random-request-config '[{"width":1344,"height":768,"prompt":"A white cat with black mustache and eyebrow markings sits on a beige couch, lip-syncing precisely to the complete reference audio.","num_inference_steps":8,"num_frames":209,"fps":24,"weight":1}]' \
          --extra-body '{"flow_shift":12.0,"audio_reference":"{\"audio_url\":\"http://localhost:8092/ref2va_audio.mp3\"}","extra_params":"{\"task\":\"ref2va\",\"duration\":8.7,\"audio_flow_shift\":3.0}"}'
      hardware_overrides:
        rtx_pro_5000_4x:
          vram_minimum_gb: 288
          extra_args: ["--task-type", "ref2va"]
        rtx_pro_6000_2x:
          vram_minimum_gb: 192
          extra_args: ["--task-type", "ref2va"]
        rtx_5090_2x:
          vram_minimum_gb: 64
          extra_args: ["--task-type", "ref2va"]
        rtx_4090_2x:
          vram_minimum_gb: 48
          extra_args: ["--task-type", "ref2va"]
          curl: |
            # audio_reference takes an HTTP(S) or data: URL. Serve local assets first:
            # python -m http.server 8092 --bind 127.0.0.1 --directory /path/to/reference_assets
            export REF_IMAGE=/path/to/reference_assets/ref2va_image.png
            export AUDIO_URL=http://127.0.0.1:8092/ref2va_audio.mp3

            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=A white cat with black mustache and eyebrow markings sits on a beige couch, lip-syncing precisely to the complete reference audio.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=3101' \
              -F 'extra_params={"task":"ref2va","duration":5.0,"audio_flow_shift":3.0}' \
              -F "input_reference=@${REF_IMAGE};type=image/png" \
              -F "audio_reference={\"audio_url\":\"${AUDIO_URL}\"}" \
              -o ref2va_image_audio.mp4
        dgx_spark_gb10:
          vram_minimum_gb: 128
          extra_args: ["--task-type", "ref2va"]
          curl: |
            # audio_reference takes an HTTP(S) or data: URL. Serve local assets first:
            # python -m http.server 8092 --bind 127.0.0.1 --directory /path/to/reference_assets
            export REF_IMAGE=/path/to/reference_assets/ref2va_image.png
            export AUDIO_URL=http://127.0.0.1:8092/ref2va_audio.mp3

            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=A white cat with black mustache and eyebrow markings sits on a beige couch, lip-syncing precisely to the complete reference audio.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=3101' \
              -F 'extra_params={"task":"ref2va","duration":5.0,"audio_flow_shift":3.0}' \
              -F "input_reference=@${REF_IMAGE};type=image/png" \
              -F "audio_reference={\"audio_url\":\"${AUDIO_URL}\"}" \
              -o ref2va_image_audio.mp4
      curl: |
        # audio_reference takes an HTTP(S) or data: URL. Serve local assets first:
        # python -m http.server 8092 --bind 127.0.0.1 --directory /path/to/reference_assets
        export REF_IMAGE=/path/to/reference_assets/ref2va_image.png
        export AUDIO_URL=http://127.0.0.1:8092/ref2va_audio.mp3

        curl -sS -X POST http://localhost:8000/v1/videos/sync \
          -F 'prompt=A white cat with black mustache and eyebrow markings sits on a beige couch, lip-syncing precisely to the complete reference audio before shifting from confusion to deadpan speechlessness.' \
          -F 'width=1344' \
          -F 'height=768' \
          -F 'fps=24' \
          -F 'num_inference_steps=50' \
          -F 'flow_shift=12' \
          -F 'seed=3101' \
          -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
          -F "input_reference=@${REF_IMAGE};type=image/png" \
          -F "audio_reference={\"audio_url\":\"${AUDIO_URL}\"}" \
          -o ref2va_image_audio.mp4

    - id: ti2v
      key: ref_video_mixed
      label: "Ref2VA - Video / Mixed"
      model_id: "/path/to/MiniMax-H3"
      vram_minimum_gb: 202
      endpoint: "/v1/videos/sync"
      description: "One or more ordered video or mixed media references plus prompt to video and audio through the Ref2VA DiT."
      benchmark: |
        python3 /path/to/vllm-omni/benchmarks/diffusion/diffusion_benchmark_serving.py \
          --model /path/to/MiniMax-H3 \
          --host localhost \
          --port 8000 \
          --endpoint /v1/videos \
          --dataset random \
          --task v2v \
          --num-prompts 3 \
          --max-concurrency 1 \
          --warmup-requests 1 \
          --warmup-concurrency 1 \
          --seed 42 \
          --random-request-config '[{"width":1344,"height":768,"prompt":"Restyle the reference footage into a cinematic night scene with neon reflections, keeping the same camera motion and subject.","num_inference_steps":8,"num_frames":209,"fps":24,"weight":1}]' \
          --extra-body '{"flow_shift":12.0,"extra_params":"{\"task\":\"ref2va\",\"duration\":8.7,\"aspect_ratio\":\"16:9\",\"audio_flow_shift\":3.0}"}'
      hardware_overrides:
        rtx_pro_5000_4x:
          vram_minimum_gb: 288
          extra_args: ["--task-type", "ref2va"]
        rtx_pro_6000_2x:
          vram_minimum_gb: 192
          extra_args: ["--task-type", "ref2va"]
        rtx_5090_2x:
          vram_minimum_gb: 64
          extra_args: ["--task-type", "ref2va"]
        rtx_4090_2x:
          vram_minimum_gb: 48
          extra_args: ["--task-type", "ref2va"]
          curl: |
            export SUBJECT_VIDEO=/path/to/green_screen_subject.mp4
            export BACKGROUND_VIDEO=/path/to/fairytale_background.mov

            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=Remove the green screen background of Video 1 and replace it with the fairytale environment from Video 2. Match motion and relight the character to fit the scene.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=3102' \
              -F 'extra_params={"task":"ref2va","duration":5.0,"audio_flow_shift":3.0}' \
              -F "input_references=@${SUBJECT_VIDEO};type=video/mp4" \
              -F "input_references=@${BACKGROUND_VIDEO};type=video/quicktime" \
              -o ref2va_video_mixed.mp4
        dgx_spark_gb10:
          vram_minimum_gb: 128
          extra_args: ["--task-type", "ref2va"]
          curl: |
            export SUBJECT_VIDEO=/path/to/green_screen_subject.mp4
            export BACKGROUND_VIDEO=/path/to/fairytale_background.mov

            curl -sS -X POST http://localhost:8000/v1/videos/sync \
              -F 'prompt=Remove the green screen background of Video 1 and replace it with the fairytale environment from Video 2. Match the background motion to the character actions and relight the character to fit the scene.' \
              -F 'width=1024' \
              -F 'height=576' \
              -F 'fps=24' \
              -F 'num_inference_steps=50' \
              -F 'flow_shift=12' \
              -F 'seed=3102' \
              -F 'extra_params={"task":"ref2va","duration":5.0,"audio_flow_shift":3.0}' \
              -F "input_references=@${SUBJECT_VIDEO};type=video/mp4" \
              -F "input_references=@${BACKGROUND_VIDEO};type=video/quicktime" \
              -o ref2va_video_mixed.mp4
      curl: |
        export SUBJECT_VIDEO=/path/to/green_screen_subject.mp4
        export BACKGROUND_VIDEO=/path/to/fairytale_background.mov

        curl -sS -X POST http://localhost:8000/v1/videos/sync \
          -F 'prompt=Remove the green screen background of Video 1 and replace it with the fairytale environment from Video 2. Match the background motion to the character actions and relight the character to fit the scene.' \
          -F 'width=1344' \
          -F 'height=768' \
          -F 'fps=24' \
          -F 'num_inference_steps=50' \
          -F 'flow_shift=12' \
          -F 'seed=3102' \
          -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
          -F "input_references=@${SUBJECT_VIDEO};type=video/mp4" \
          -F "input_references=@${BACKGROUND_VIDEO};type=video/quicktime" \
          -o ref2va_video_mixed.mp4

dependencies:
  # Host-side prerequisite, not a pip install — the weights have to be on disk
  # before the container can bind-mount them, so this one shows under Docker too.
  - note: "Gated checkpoint — request access on the model card, then download it locally. Serve the root directory: both DiTs remain in subdirectories while the modular service shares the encoder and VAEs."
    command: "hf download MiniMaxAI/MiniMax-H3 --local-dir /path/to/MiniMax-H3"
    install_modes: [pip, docker]
  - note: "Ascend pip path: build vLLM v0.26.0 with the NPU target and vLLM Ascend from the matching release branch. This replaces the CUDA wheel shown in the pip tab above."
    command: |
      git clone --depth 1 --branch v0.26.0 https://github.com/vllm-project/vllm.git
      cd vllm && VLLM_TARGET_DEVICE=empty pip install -e . && cd ..
      git clone --depth 1 --branch releases/v0.26.0rc https://github.com/vllm-project/vllm-ascend.git
      cd vllm-ascend && git submodule update --init --recursive && pip install -e . && cd ..
    install_modes: [pip]
    brand: Huawei
  - note: "pip path only: install current vLLM-Omni main for the modular MiniMax-H3 pipeline."
    command: "git clone --depth 1 https://github.com/vllm-project/vllm-omni.git /path/to/vllm-omni && cd /path/to/vllm-omni && uv pip install -e ."
    install_modes: [pip]
  - note: "Optional FastH3 adapters: download the Dense / Data-Free baseline and the recommended VSA / Data-Free artifact under the already-mounted H3 root. FastH3 preview v1 serves T2VA only and is fused at startup."
    command: |
      hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
        dense-datafree/adapter_model.safetensors \
        vsa-datafree/adapter_model.safetensors \
        --local-dir /path/to/MiniMax-H3/FastH3-LoRA
    install_modes: [pip, docker]
    optional: true
  - note: "Optional LightX2V Turbo few-step LoRA: download one Diffusers-layout artifact under the already-mounted H3 root. The filename carries the sampling contract, so do not rename it; the _comfyui_ exports are not supported."
    command: |
      hf download lightx2v/Minimax-h3-Turbo \
        minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors \
        --local-dir /path/to/MiniMax-H3/Turbo-LoRA
    install_modes: [pip, docker]
    optional: true
  - note: "Optional CUDA-only FastH3 VSA kernel for the pip path. --no-deps preserves the PyTorch/CUDA stack managed by vLLM. The tested B300/SM103 route uses the Triton fallback with FASTVIDEO_VSA_SM100A=0."
    command: "uv pip install --no-deps fastvideo-kernel==0.3.5"
    install_modes: [pip]
    brand: NVIDIA
    optional: true
  - note: "Optional pip extra for explicitly comparing FA4 on datacenter Blackwell. The generated B300/GB200 route uses the TRTLLM_ATTN default; RTX PRO 5000 and the two-GPU RTX routes use CUDNN_ATTN."
    command: "cd /path/to/vllm-omni && uv pip install -e '.[fa4]'"
    install_modes: [pip]
    brand: NVIDIA
    optional: true
  - note: "Docker path only: clone current vLLM-Omni main to the path mounted by the generated command so it overrides the older package in the image."
    command: "git clone --depth 1 https://github.com/vllm-project/vllm-omni.git /path/to/vllm-omni"
    install_modes: [docker]
  - note: "pip path only: ffmpeg and ffprobe must be on PATH for reference-video preparation and MP4 video+audio muxing. The official Docker images already bundle them."
    command: "sudo apt-get install -y ffmpeg"

features: {}

opt_in_features: []

variants:
  default:
    precision: bf16
    vram_minimum_gb: 202
    description: "BF16 modular service — two 66.3 GB DiTs share the 51.5 GB Qwen3-VL encoder and ~10.6 GB VAEs. Use --task-type fl2va or ref2va on memory-constrained profiles to load one DiT."
    hardware_overrides:
      dgx_spark_gb10:
        extra_args:
          - "--num-gpus"
          - "1"
          - "--usp"
          - "1"
          - "--ring"
          - "1"
          - "--vae-patch-parallel-size"
          - "1"
          - "--quantization"
          - "fp8"
          - "--init-timeout"
          - "3600"
          - "--enforce-eager"
          - "--diffusion-attention-backend"
          - "CUDNN_ATTN"
        extra_env:
          VLLM_OMNI_VIDEO_SYNC_TIMEOUT: "7200"
      ascend_950pr:
        docker_image: "quay.io/ascend/vllm-omni:minimax-h3"
        extra_args:
          - "--num-gpus"
          - "1"
          - "--tensor-parallel-size"
          - "1"
          - "--usp"
          - "1"
          - "--ring"
          - "1"
          - "--text-encoder-tp-size"
          - "1"
          - "--vae-patch-parallel-size"
          - "1"
          - "--quantization"
          - "int8"
          - "--init-timeout"
          - "14400"
          - "--stage-init-timeout"
          - "14400"
        extra_env:
          VLLM_OMNI_VIDEO_SYNC_TIMEOUT: "14400"
      rtx_pro_5000_4x:
        extra_args:
          - "--num-gpus"
          - "4"
          - "--tensor-parallel-size"
          - "2"
          - "--usp"
          - "2"
          - "--ring"
          - "1"
          - "--text-encoder-tp-size"
          - "4"
          - "--vae-patch-parallel-size"
          - "4"
          - "--diffusion-attention-backend"
          - "CUDNN_ATTN"
      rtx_pro_6000_2x:
        extra_args:
          - "--num-gpus"
          - "2"
          - "--tensor-parallel-size"
          - "2"
          - "--usp"
          - "1"
          - "--ring"
          - "1"
          - "--text-encoder-tp-size"
          - "2"
          - "--vae-patch-parallel-size"
          - "2"
      rtx_5090_2x:
        extra_args:
          - "--num-gpus"
          - "2"
          - "--tensor-parallel-size"
          - "2"
          - "--usp"
          - "1"
          - "--ring"
          - "1"
          - "--text-encoder-tp-size"
          - "2"
          - "--vae-patch-parallel-size"
          - "2"
          - "--diffusion-attention-backend"
          - "CUDNN_ATTN"
          - "--enable-distributed-layerwise-offload"
          - "--dlo-no-use-allgather"
          - "--dlo-resident-layers"
          - "20"
          - "--enforce-eager"
        extra_env:
          VLLM_OMNI_VIDEO_SYNC_TIMEOUT: "14400"
      rtx_4090_2x:
        extra_args:
          - "--num-gpus"
          - "2"
          - "--tensor-parallel-size"
          - "2"
          - "--usp"
          - "1"
          - "--ring"
          - "1"
          - "--text-encoder-tp-size"
          - "2"
          - "--vae-patch-parallel-size"
          - "2"
          - "--diffusion-attention-backend"
          - "CUDNN_ATTN"
          - "--enable-distributed-layerwise-offload"
          - "--dlo-no-use-allgather"
          - "--dlo-resident-layers"
          - "12"
          - "--enforce-eager"
        extra_env:
          VLLM_OMNI_VIDEO_SYNC_TIMEOUT: "14400"


# Omni recipes are served through `vllm serve --omni`, which has its own
# diffusion-parallelism flags (--usp / --ring / --vae-patch-parallel-size).
# The serving-strategy row does not apply.
compatible_strategies: []

hardware_overrides:
  blackwell:
    extra_env:
      FLASHINFER_DISABLE_VERSION_CHECK: "1"
  amd:
    extra_args:
      - "--text-encoder-tp-size"
      - "4"
    extra_env:
      VLLM_ROCM_USE_AITER: "1"

strategy_overrides: {}

guide: |
  ## Overview

  MiniMax H3 is an **open-weight, general-purpose multimodal generation model**. Rather than
  being confined to one specialized task — generate, edit, or reference — H3 reads a
  multimodal context that mixes text, images, video, and audio together, interprets the
  creative intent as a whole, and produces coherent audio-visual output end to end.

  Architecturally it is a **CFG-distilled joint video/audio diffusion transformer** served
  through vLLM-Omni's OpenAI-compatible `/v1/videos` API. Every request returns a single MP4
  containing H.264 video and **native stereo audio** — the audio is generated jointly with
  the video by the same DiT, not dubbed on afterwards.

  Three capability areas the model targets:

  - **Commercial-grade generation** — film and entertainment, advertising and branding,
    e-commerce, gaming; dynamic typography, VFX, product showcases, UI/UX motion design.
  - **Native multimodal understanding + generation** — interprets characters, motion, sound,
    emotion, cinematography, and visual style across mixed references, then combines them.
  - **Precise multimodal editing and control** — iterative refinement of characters, objects,
    scenes, sound, and rhythm with strong instruction following.

  The checkpoint contains two DiT backbones that share the encoder and VAEs:

  | DiT | Tasks | Conditioning |
  |---|---|---|
  | `FL2VA` | `t2va`, `fl2va` | text only, or text + first/last frame |
  | `Ref2VA` | `ref2va` | text + omni references (images, video clips, audio clips) |

  Since vLLM-Omni [#5720](https://github.com/vllm-project/vllm-omni/pull/5720), serving
  `/path/to/MiniMax-H3` initializes both DiTs by default, while the Qwen3-VL encoder and
  video/audio VAEs stay shared. One server can therefore handle all three task types.
  Add `--task-type fl2va` or `--task-type ref2va` only when the hardware profile needs to
  load one DiT to save GPU or host memory.

  Components (BF16): two 52-block joint video/audio DiTs (66.3 GB each), a shared
  Qwen3-VL layer-50 text/vision encoder (51.5 GB), a video VAE (~10 GB), and an audio
  VAE (~0.6 GB).

  ## Model specifications

  | | |
  |---|---|
  | Output duration | 4–15 seconds |
  | Frame rate | 24 FPS (fixed) |
  | Audio | Native stereo, on every generation |
  | Resolution — 1440p (2K) | Short edge 1440 px for 16:9 … 9:16; wider formats ≈3.7 MP (e.g. 2976×1248 at 21:9) |
  | Resolution — 768p | Short edge 768 px for 16:9 … 9:16; wider formats ≈1 MP (e.g. 1536×672 at 21:9). Upscalable to 1440p. *(mode coming soon)* |
  | Aspect ratio | First/Last Frame follows the uploaded image. Text-to-Video and Omni Reference select from 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 — Omni Reference also offers Auto |
  | Prompt length | Up to 7,000 characters |

  ### Input modes and limits

  **First/Last Frame** — 0, 1, or 2 images. Edge lengths in `[256, 5760]`, aspect ratio 5:2
  to 2:5. With no image, H3 falls back to Text-to-Video.

  **Omni Reference** — up to 9 images; up to 3 video clips (2–15 s each, 15 s total); up to
  3 audio clips (2–15 s each, 15 s total). Edge lengths in `[256, 5760]`, video aspect ratio
  5:2 to 2:5. **Audio must be paired with at least one image or video** — it cannot be the
  sole reference. Mixed inputs cap at 12 files total; with none, H3 falls back to
  Text-to-Video.

  | | Formats | Per-file size |
  |---|---|---|
  | Video | H.264/AVC, H.265/HEVC (embedded audio AAC or MP3) | 50 MB |
  | Image | JPG, JPEG, PNG, WEBP, HEIC, HEIF | 30 MB |
  | Audio | WAV, MP3 | 15 MB |

  Limits are per file, not per combined upload. Keep the whole request body under 64 MB —
  prefer URL-based media over inlined uploads for anything large.

  ## Prerequisites

  H3 can be served directly from the gated HF ID or from a local root directory. The local root contains both DiT
  subdirectories plus the shared encoder and VAEs. Request access on the gated model card,
  then download the complete repository:

  ```bash
  hf download MiniMaxAI/MiniMax-H3 --local-dir /path/to/MiniMax-H3
  ```

  That yields `/path/to/MiniMax-H3/FL2VA`, `/path/to/MiniMax-H3/Ref2VA`, and the
  shared component directories. Serve `/path/to/MiniMax-H3` itself and substitute your own
  `--local-dir` throughout the commands below.

  `ffmpeg` and `ffprobe` must be on `PATH` — they are used for reference-video preparation
  and MP4 muxing.

  ## Installation

  ### Docker (recommended)

  The image bundles H3 handlers and media dependencies. The published tag predates the
  modular pipeline merged in vLLM-Omni #5720, so clone current `main` before starting:

  ```bash
  docker pull vllm/vllm-omni:minimax-h3
  git clone --depth 1 https://github.com/vllm-project/vllm-omni.git /path/to/vllm-omni
  ```

  The command builder's Docker mode binds both the root checkpoint and source checkout,
  then prepends the checkout to `PYTHONPATH`:

  ```bash
  -v /path/to/MiniMax-H3:/path/to/MiniMax-H3:ro
  -v /path/to/vllm-omni:/path/to/vllm-omni:ro
  -e PYTHONPATH=/path/to/vllm-omni
  ```

  ### pip

  H3 support ships in vLLM-Omni rather than the `vllm` wheel, so the pip path needs a
  current source checkout:

  ```bash
  uv venv
  source .venv/bin/activate
  uv pip install --torch-backend=cu130 vllm==0.33.0
  git clone --depth 1 https://github.com/vllm-project/vllm-omni.git /path/to/vllm-omni
  cd /path/to/vllm-omni
  uv pip install -e .
  ```

  The generated datacenter Blackwell route uses the default `TRTLLM_ATTN`; RTX PRO 5000 and the two-GPU RTX profiles use `CUDNN_ATTN`. These routes need no
  FlashAttention-4 install. Install the optional extra only when explicitly comparing FA4:
  `uv pip install -e '.[fa4]'`.

  ## Launch — two 24/32 GB GPUs with distributed layerwise offload

  vLLM-Omni includes a low-memory TP2 path for H3. Install the current vLLM-Omni
  source checkout as shown above.

  The two-GPU profile combines TP2 with distributed layerwise offload (DLO). Each rank
  keeps its rank-local TP shard in pinned host memory; `--dlo-no-use-allgather` streams
  those shards directly instead of reconstructing full DiT blocks on every rank. A
  configurable prefix of DiT blocks stays resident for the complete denoise stage, then
  releases its HBM before VAE decode.

  Select the profile first. RTX 5090 uses 20 resident DiT layers at the released
  1344×768 shape; RTX 4090 keeps more HBM headroom with 12 resident layers and a
  1024×576 starting shape:

  ```bash
  # 2× RTX 5090 (32 GB)
  export DLO_RESIDENT_LAYERS=20 WIDTH=1344 HEIGHT=768

  # Or, 2× RTX 4090 (24 GB)
  # export DLO_RESIDENT_LAYERS=12 WIDTH=1024 HEIGHT=576
  ```

  Load only the FL2VA DiT on these memory-constrained profiles:

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
  vllm serve /path/to/MiniMax-H3 \
    --omni \
    --task-type fl2va \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --num-gpus 2 \
    --tensor-parallel-size 2 \
    --usp 1 \
    --ring 1 \
    --text-encoder-tp-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-parallel-mode tile \
    --vae-use-tiling \
    --enable-distributed-layerwise-offload \
    --dlo-no-use-allgather \
    --dlo-resident-layers "${DLO_RESIDENT_LAYERS}" \
    --enforce-eager \
    --diffusion-attention-backend CUDNN_ATTN
  ```

  The resident count changes placement and transfer frequency only; it does not quantize
  or change the BF16/FP32 denoise math. Use a **384 GiB-class host** for the pinned model
  shards and offload buffers.

  Wait for `Application startup complete`, then verify health and run one full 50-step
  T2VA request:

  ```bash
  curl --fail http://127.0.0.1:8000/health

  curl --fail-with-body -sS \
    -X POST http://127.0.0.1:8000/v1/videos/sync \
    -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
    -F "width=${WIDTH}" \
    -F "height=${HEIGHT}" \
    -F 'fps=24' \
    -F 'num_inference_steps=50' \
    -F 'flow_shift=12' \
    -F 'seed=1101' \
    -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
    -o minimax-h3-t2va.mp4

  ffprobe -v error -show_entries \
    stream=index,codec_name,width,height,r_frame_rate,sample_rate,channels \
    -of json minimax-h3-t2va.mp4
  ffmpeg -v error -i minimax-h3-t2va.mp4 \
    -map 0:v:0 -map 0:a:0 -f null -
  ```
  **Note:** If using 2× RTX 4090, the curl request must include -F 'aspect_ratio=16:9' to succeed.

  A successful output has H.264 video at 24 FPS and stereo AAC audio at 32 kHz, and the
  final `ffmpeg` command exits with status 0. This capacity profile intentionally loads one
  DiT. Restart the same root-model command with `--task-type ref2va` for Ref2VA; remove
  `--task-type` only after confirming that the combined modular service fits the target.

  The same root path is used for both task families; only `--task-type` changes on these
  capacity profiles.

  ## Launch — four GPUs (validated best practice)

  On B300/GB200, the no-offload root service uses Ulysses SP4, tiled VAE patch
  parallelism 4, regional `torch.compile`, dense BF16 `TRTLLM_ATTN`, and Ring/TP1. Both DiTs load
  once while the encoder and VAEs remain shared.

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  vllm serve /path/to/MiniMax-H3 \
    --omni \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --num-gpus 4 \
    --usp 4 \
    --ring 1 \
    --vae-patch-parallel-size 4 \
    --vae-parallel-mode tile \
    --vae-use-tiling
  ```

  Omitting the attention flag is intentional: current datacenter Blackwell builds default to
  dense BF16 `TRTLLM_ATTN`. Install `[fa4]` and add `--diffusion-attention-backend FLASH_ATTN`
  only for an explicit FA4 comparison.

  ### Optional lossy attention: SAGE + Skip-Softmax

  `TRTLLM_ATTN` offers two **lossy** optimizations for the long main DiT attention sequence.
  Both work under the pure Ulysses parallelism of the four-GPU profile (`--usp 4 --ring 1`) and
  are datacenter-Blackwell only. Append the following to the four-GPU command above to enable
  both:

  ```bash
  --diffusion-attention-config '{
    "default": {
      "backend": "TRTLLM_ATTN",
      "quant": {"dtype_qk": "fp8_e4m3", "q_block_size": 1, "k_block_size": 16},
      "skip_softmax": {"threshold": 0.05, "disabled_until_timestep": 0.97}
    },
    "per_role": {
      "minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}
    }
  }'
  ```

  - **SAGE** quantizes Q/K to `fp8_e4m3` for `QK^T`; P and V are always FP8 in this kernel.
    B200 additionally supports `int8` Q/K, which preserves accuracy better than FP8.
  - **Skip-Softmax** skips the Softmax and `PV` work of KV tiles whose scores fall far below
    the running row maximum. `threshold=0.05` with `disabled_until_timestep=0.97` is a
    **conservative** setting: a low threshold skips only clearly negligible tiles, and the
    cutoff keeps the early high-noise steps dense — H3's default video flow shift of 12 keeps
    the normalized timestep above `0.97` for the first 14 of 49 denoiser forwards at 50 steps.
    Raise `threshold` or lower `disabled_until_timestep` for more speedup once quality is
    verified.
  - The `per_role` entry keeps the short token-refiner attention dense; a per-role spec does
    not inherit `quant` or `skip_softmax` from `default`.

  Both optimizations trade fidelity for speed, and their effects compound. The measured B300
  speedup and LPIPS for each mode and for the combination are in the vLLM blog's
  [Quantized and Sparse Attention](https://vllm.ai/blog/2026-09-01-minimax-h3-production-serving#quantized-and-sparse-attention-in-trtllm_attn)
  section as a reference; tune the settings according to the accuracy and speed required. See
  the
  [TRTLLM attention guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/attention_backends/trtllm/)
  for the full configuration reference.

  No restart is required: `extra_params.task` routes T2VA/FL2VA requests to the FL2VA
  DiT and Ref2VA requests to the Ref2VA DiT.

  ### FastH3 model family

  FastH3 is a family of post-trained models based on MiniMax-H3. It generates
  video with stereo audio in four or eight steps. Choose a
  starting point for your input and hardware:

  | Input | Hardware | Model |
  |---|---|---|
  | Prompt and reference image | RTX 5090, DGX Spark, RTX PRO 6000 Blackwell | [Omni NVFP4 QAD](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD), single-card guide below |
  | Prompt and reference image | RTX 4090 | [Omni BF16](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1) with FP8 enabled, single-card guide below |
  | Prompt and reference image | Eight GPUs | Omni BF16, eight-GPU guide below |
  | Prompt only | Eight GPUs | [V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) (experimental) or [four-step adapter](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA) |

  The command builder above serves native MiniMax-H3. Use the FastH3 setup
  and commands below for these models. Select one model when starting the server.
  The single-card profiles are configuration guidance; this documentation change
  does not independently qualify them with GPU inference.

  #### Install FastH3

  Use Linux, Python 3.12, a CUDA 13-compatible NVIDIA driver, and
  [uv](https://docs.astral.sh/uv/getting-started/installation/). The eight-step
  models currently need the source branch from
  [vLLM-Omni #9167](https://github.com/vllm-project/vllm-omni/pull/9167):

  This external fork's instructions retain its vLLM 0.32.0 runtime. They do
  not install or qualify this checkout's final vLLM 0.33.0 alignment; follow
  the [source installation guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/#build-wheel-from-source)
  for this checkout. FastH3 on the 0.33.0 runtime remains unverified.

  ```bash
  sudo apt-get update
  sudo apt-get install -y git ffmpeg

  git clone --branch pr/fasth3-omni-upstream https://github.com/lishunyang12/vllm-omni.git vllm-omni-fasth3
  cd vllm-omni-fasth3
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install vllm==0.32.0 --torch-backend=cu130
  uv pip install -e .
  ```

  On DGX Spark, run in the native `aarch64` environment. If Hugging Face
  requests access, accept the model's access terms and run `hf auth login`.
  Run the following server commands from the `vllm-omni-fasth3` directory.

  #### FastH3 RefOmni Single Card

  **RTX 5090 (32 GB), DGX Spark (128 GB), or RTX PRO 6000 Blackwell (96 GB):**

  ```bash
  CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
  vllm serve FastVideo/FastH3-Omni-NVFP4-QAD --omni \
    --config recipes/FastVideo/fasth3-omni-quantized.yaml
  ```

  This uses the smaller NVFP4 model and fast 8-layer video decoder. Layer
  offload reduces GPU memory use.

  **RTX 4090 (24 GB):**

  ```bash
  uv pip install -e '.[vsa]'

  CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
  vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
    --config recipes/FastVideo/fasth3-omni-rtx4090.yaml
  ```

  The first start downloads the model; the 4090 also converts eligible
  weights to FP8 while loading. Later starts reuse the download cache.
  Keep the terminal open and wait for `Application startup complete`.

  **Generate.** Save your image as `reference.png`. In a second terminal:

  ```bash
  curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
    -F 'prompt=The person in the reference image smiles and waves at the camera. Birds chirp as the person says, "Welcome to the garden!"' \
    -F 'input_reference=@reference.png' \
    -F 'size=832x480' -F 'seconds=5' -F 'seed=1234' \
    -o output.mp4
  ```

  Open `output.mp4` to play the video with stereo audio. For 768P, change
  `size=832x480` to `size=1344x768`. If memory is tight, start at 480P and close
  other GPU workloads. For multiple images, video, or audio references, see
  the [H3 reference API](https://github.com/lishunyang12/vllm-omni/blob/pr/fasth3-omni-upstream/recipes/MiniMaxAI/MiniMax-H3.md).

  **Optional settings:**

  - **FP8 model:** if you have access to [FastH3-Omni-FP8](https://huggingface.co/FastVideo/FastH3-Omni-FP8),
    replace the model ID in the Blackwell command with `FastVideo/FastH3-Omni-FP8`.
    A local model directory also works. This release also requires Blackwell
    because its text encoder uses NVFP4.
  - **36-layer decoder:** in `recipes/FastVideo/fasth3-omni-quantized.yaml`, change
    `additional-config.video_vae_decoder.subfolder` from `fast-8L` to
    `quality-36L`, then restart. Both decoders come from
    [FastH3-NVFP4-VAE-Decoder](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder).
  - **DGX Spark:** remove `diffusion-offload-config` from the Blackwell YAML to
    keep the model resident in unified memory and avoid offload transfers.

  #### FastH3 Omni on eight GPUs

  Install the attention kernels, then start the server:

  ```bash
  uv pip install -e '.[cudnn-bsa]'

  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
  vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
    --config recipes/FastVideo/fasth3-8step-8gpu.yaml
  ```

  Wait for `Application startup complete`, then use the image-and-prompt request
  above. The configuration sets eight inference steps and handles GPU placement.

  For an NVFP4 video decoder on Blackwell, add this to the deployment YAML:

  ```yaml
  additional-config:
    video_vae_decoder:
      model: FastVideo/FastH3-NVFP4-VAE-Decoder
      subfolder: fast-8L
  ```

  Use `quality-36L` for the deeper decoder, then restart the server.

  **Performance reference: eight RTX PRO 5000 Blackwell 72 GB GPUs.**
  BF16, 1344×768, one request at a time; timing includes MP4 encoding and
  excludes model loading:

  | Output | Generation time | Reported allocator peak |
  |---|---:|---:|
  | 124 frames, about 5 seconds | 41.04 s, median of 3 runs | 35.15 GiB |
  | 345 frames, about 14 seconds | 95.93 s, 1 run | 38.94 GiB |

  These are offline measurements. See the
  [benchmark details](https://github.com/lishunyang12/vllm-omni/pull/63)
  for the measured environment and configuration.

  #### FastH3 V2: text to video and audio

  This eight-GPU deployment is experimental. Install the `cudnn-bsa` extra
  above, then start V2 in place of the Omni server:

  ```bash
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
  vllm serve FastVideo/FastVideo-FastH3-8-Step-V2 --omni \
    --config recipes/FastVideo/fasth3-8step-8gpu.yaml
  ```

  Wait for startup, then send a text-only request in a second terminal:

  ```bash
  curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
    -F 'prompt=A train crosses a stone bridge at sunrise, with rhythmic wheels and distant birdsong.' \
    -F 'size=1344x768' -F 'seconds=5' \
    -o fasth3-v2.mp4
  ```

  Keep the model's default eight steps and sampling settings.

  ### FastH3 four-step Dense and VSA adapters

  Use the four-step adapter for text-to-video generation on eight GPUs.
  Choose the adapter at startup; per-request switching and CPU/layer offload
  are unsupported. Download the Dense and VSA adapters:

  ```bash
  hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
    dense-datafree/adapter_model.safetensors \
    vsa-datafree/adapter_model.safetensors \
    --local-dir ./FastH3-LoRA
  ```

  Start with the Dense adapter:

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  vllm serve MiniMaxAI/MiniMax-H3 --omni \
    --host 127.0.0.1 --port 8095 --trust-remote-code \
    --task-type fl2va --served-model-name FastH3 \
    --num-gpus 8 --tensor-parallel-size 1 --usp 8 --ring 1 \
    --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
    --diffusion-attention-config.default.backend FLASH_ATTN \
    --lora-path ./FastH3-LoRA/dense-datafree/adapter_model.safetensors
  ```

  For the VSA adapter, install `uv pip install -e '.[vsa]'`, change `--lora-path`
  to `./FastH3-LoRA/vsa-datafree/adapter_model.safetensors`, and replace the
  attention option with:

  ```bash
  --diffusion-attention-config.default.backend FASTVIDEO_VSA \
  --diffusion-attention-config.default.fastvideo_vsa_topk 64
  ```

  Keep pure Ulysses parallelism (`--usp 8 --ring 1`) for VSA. See
  [backend selection](https://github.com/lishunyang12/vllm-omni/blob/pr/fasth3-omni-upstream/recipes/FastVideo/FastH3.md#hardware-and-backend-selection)
  for hardware requirements.

  After the server starts, generate a video in a second terminal:

  ```bash
  curl --fail-with-body -sS http://127.0.0.1:8095/v1/videos/sync \
    -F 'model=FastH3' \
    -F 'prompt=A cinematic starship glides through a luminous nebula while its engines hum and distant radio chatter fills the cabin.' \
    -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
    -F 'num_inference_steps=4' -F 'guidance_scale=1' -F 'seed=1101' \
    -F 'extra_params={"task":"t2va","duration":10.0}' \
    -o fasth3.mp4
  ```

  Use four steps and `task=t2va`; keep the default sampling settings.

  **Performance reference: eight B300 GPUs.** At 1344×768 and 24 FPS,
  server generation time including MP4 encoding was:

  | Output duration | Dense (TRTLLM) | VSA |
  |---|---:|---:|
  | 10 seconds | 9.84 s | 7.28 s |
  | 15 seconds | 14.20 s | 10.80 s |

  See [benchmark details](https://github.com/vllm-project/vllm-omni/pull/6909)
  for the measured environment and configuration.

  ### LightX2V Turbo few-step LoRA

  [LightX2V Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) is a family of
  few-step LoRA artifacts for H3. It reaches the same goal as FastH3 by the opposite
  route: instead of fusing full-rank deltas at startup, Turbo stays a standard LoRA that
  each request activates by name. One server can therefore hold several artifacts and let
  requests choose between them, at the cost of the LoRA matmuls on every denoise step.

  Eight Diffusers-layout artifacts are supported. The filename is the contract — the server
  parses the task family, step count and resolution out of it, so a renamed file is rejected
  rather than served on a guess. `audio_flow_shift` is `3.0` across the family.

  | Artifact | Tasks | `num_inference_steps` | `flow_shift` | alpha |
  |---|---|---:|---:|---:|
  | `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | T2VA / FL2VA | 5 | 12 | none → 8 |
  | `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` | T2VA / FL2VA | 5 | 6 | 128 |
  | `minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors` | T2VA / FL2VA | 5 | 6 | 128 |
  | `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` | T2VA / FL2VA | 5 | 6 | 8 |
  | `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` | T2VA / FL2VA | 9 | 12 | 8 |
  | `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` | T2VA / FL2VA | 9 | 6 | 8 |
  | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` | Ref2VA | 5 | 12 | 8 |
  | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | Ref2VA | 9 | 6 | 8 |

  Requested sigma points are always one more than the artifact's forwards, which is why a
  four-step Turbo artifact takes `num_inference_steps=5` where FastH3 takes `4`.

  Every artifact is rank 128 and the delta is applied at `scale * alpha / 128`, with alpha
  read from the artifact's metadata — only `4step_v0.1` declares none and falls back to 8
  with a warning.

  Download one artifact under the H3 root so the existing bind mount includes it. Each
  artifact except `4step_v0.1` also ships a `_comfyui_` export that fuses Q/K/V into a single
  projection; those are **not** supported and are refused by name.

  ```bash
  hf download lightx2v/Minimax-h3-Turbo \
    minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors \
    --local-dir /path/to/MiniMax-H3/Turbo-LoRA
  ```

  Serve it on the same eight-GPU pure-Ulysses topology. `--lora-path` preloads the adapter;
  it accepts one artifact, or a directory holding exactly one — a directory with several
  recognized artifacts is rejected as ambiguous.

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  vllm serve /path/to/MiniMax-H3 \
    --omni \
    --host 127.0.0.1 \
    --port 8095 \
    --trust-remote-code \
    --task-type fl2va \
    --served-model-name MiniMaxAI/MiniMax-H3 \
    --num-gpus 8 \
    --usp 8 \
    --ring 1 \
    --vae-patch-parallel-size 8 \
    --vae-parallel-mode tile \
    --vae-use-tiling \
    --diffusion-attention-backend TRTLLM_ATTN \
    --lora-backend peft \
    --lora-path /path/to/MiniMax-H3/Turbo-LoRA/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors
  ```

  Unlike FastH3, the request must activate the adapter and carry that row's sampling
  contract. A request whose steps or shift do not match the loaded artifact is rejected, so
  a mismatch cannot silently sample the distilled model at noise levels it never saw.

  ```bash
  curl --fail-with-body -sS -X POST http://127.0.0.1:8095/v1/videos/sync \
    -F 'prompt=A cinematic starship glides through a luminous nebula while its engines hum and distant radio chatter fills the cabin.' \
    -F 'width=1344' \
    -F 'height=768' \
    -F 'aspect_ratio=16:9' \
    -F 'fps=24' \
    -F 'num_inference_steps=5' \
    -F 'flow_shift=6' \
    -F 'seed=1101' \
    -F 'extra_params={"task":"t2va","duration":10.0,"audio_flow_shift":3.0}' \
    -F 'lora={"name":"h3-turbo-v1.2","path":"/path/to/MiniMax-H3/Turbo-LoRA/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors","scale":1.0}' \
    -o turbo.mp4
  ```

  FL2VA artifacts serve `t2va` and `fl2va` on an FL2VA or combined server. The two `ref2v`
  rows need `--task-type ref2va`: a combined server answers `ref2va` from a second DiT the
  adapter cannot bind to, so loading one there is refused rather than running an undistilled
  model on a few-step ladder. The integration is dynamic-only — no prefusion, no LoRA
  composition. DLO works by keeping the request-switchable LoRA A/B buffers resident on the
  accelerator while DLO streams the base blocks; model-level and standard layerwise offload
  remain unsupported.

  Measured on 8×B300 with the command above, at 1344×768, 24 FPS, a 15 s FL2VA request with
  first and last keyframes (362 frames), fixed seed, one discarded warmup and three recorded
  runs. The baseline is the same server without an adapter at `num_inference_steps=50` and
  `flow_shift=12`:

  | Metric | Base H3, 49 forwards | Turbo v1.2, 4 forwards | Speedup |
  |---|---:|---:|---:|
  | Denoise loop | ~105 s | ~9 s | 11.7× |
  | Per denoise step | 2.14–2.16 s/it | 2.20–2.42 s/it | — |
  | Diffusion engine | 111.1–111.3 s | 15.0–15.1 s | 7.4× |
  | Server E2E incl. MP4 | 112.5–112.9 s | ~16.5 s | 6.8× |

  The ~10% higher per-step cost is the LoRA matmul the fused FastH3 route does not pay —
  irrelevant against a 12× cut in forwards, but the reason a single-artifact deployment can
  still prefer a fused student.

  These figures are for a server serving Turbo requests back to back. Alternating between
  Turbo and adapter-free requests on one server additionally pays a LoRA rebind on each
  switch — measured at about 24 s per switch in the same run (39.2 s diffusion engine against
  15.1 s steady state), because every one of H3's 208 LoRA layers is torn down and re-uploaded.
  Give Turbo its own process if the workload mixes, or track vLLM-Omni #7195, which removes
  that rebind.

  ### RTX PRO 5000 Blackwell — four PCIe GPUs

  Select the `RTX PRO 5000 Blackwell 4x` hardware profile for the validated BF16,
  no-offload topology: TP2 × Ulysses2, Ring1, text-encoder TP4, VAE patch parallel 4
  in tile mode, and `CUDNN_ATTN`. On the measured dual-socket host,
  `CUDA_VISIBLE_DEVICES=0,2,1,3` kept the TP pairs on local PXB paths; check
  `nvidia-smi topo -m` and adapt the order on other systems.

  This 72 GiB/GPU profile serves the root path but selects one DiT with `--task-type`.
  The task picker adds `fl2va` for T2VA/FL2VA and `ref2va` for Ref2VA; loading both DiTs
  together is not qualified on this memory-constrained profile.

  A five-step topology screen after warmup produced decodable H.264 video with stereo AAC:

  | Workload | Client E2E | Encode | Denoise | VAE decode | Peak/GPU |
  |---|---:|---:|---:|---:|---:|
  | T2VA, 124 frames, 1344×768 | 32.993 s | 0.034 s | 27.915 s | 4.307 s | 67.6 GiB |
  | FL2VA first frame, same output shape | 30.130 s | 0.276 s | 24.396 s | 4.339 s | 67.6 GiB |

  These are topology-screening results, not 50-step throughput claims. Nsight Systems
  aggregation for the same run was:

  | Workload | AllGather | Send/Recv | NCCL other | NCCL total | Dense FMHA | max-min/mean |
  |---|---:|---:|---:|---:|---:|---:|
  | T2VA | 0.93% | 8.53% | 14.63% | 24.09% | 31.15% | 1.11% |
  | FL2VA | 0.89% | 7.26% | 15.76% | 23.91% | 32.53% | 0.13% |

  `NCCL other` covers NCCL kernels outside recognized AllGather and point-to-point
  Send/Recv, including AllReduce, ReduceScatter, or Broadcast when present.

  ### AMD ROCm — four MI300X GPUs

  Verified on 4× AMD Instinct MI300X (gfx942) with the ROCm H3 image, AITER packed
  variable-length attention, Ulysses 4, text-encoder TP4, and native tiled VAE patch
  parallelism 4:

  ```bash
  docker pull vllm/vllm-omni-rocm:minimax-h3

  HIP_VISIBLE_DEVICES=0,1,2,3 \
  VLLM_ROCM_USE_AITER=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  vllm serve /path/to/MiniMax-H3 \
    --omni \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --num-gpus 4 \
    --usp 4 \
    --ring 1 \
    --text-encoder-tp-size 4 \
    --vae-patch-parallel-size 4 \
    --vae-parallel-mode tile \
    --vae-use-tiling \
    --diffusion-attention-backend FLASH_ATTN
  ```

  On ROCm, `FLASH_ATTN` resolves to AITER on supported Instinct architectures. Do not
  install the CUDA-only `[fa4]` extra or set `FLASHINFER_DISABLE_VERSION_CHECK`.

  The refreshed `vllm/vllm-omni-rocm:minimax-h3` tag includes the Ref2VA soundfile
  fallback, TorchCodec, and FFmpeg, so T2VA, FL2VA, and the complete Ref2VA path use the
  same official image.

  ### AMD ROCm — four MI355X GPUs

  Verified on 4× AMD Instinct MI355X. The official ROCm image includes the
  MiniMax H3 Ref2VA fixes, TorchCodec, and FFmpeg. We tested both Ulysses 4 with Ring 1
  and Ulysses 1 with Ring 4. The command below shows the Ulysses 4 / Ring 1 configuration
  with tiled VAE patch parallelism degree 4, AITER, and CPU offload:
  ```bash
  docker pull vllm/vllm-omni-rocm:minimax-h3

  HIP_VISIBLE_DEVICES=0,1,2,3 \
  VLLM_ROCM_USE_AITER=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  vllm serve /path/to/MiniMax-H3 \
    --omni \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --num-gpus 4 \
    --usp 4 \
    --ring 1 \
    --vae-patch-parallel-size 4 \
    --vae-parallel-mode tile \
    --vae-use-tiling \
    --enable-cpu-offload \
    --text-encoder-tp-size 4
  ```

  This root-model command accepts T2VA, FL2VA, and Ref2VA requests without restarting.

  Three constraints on this configuration:

  - **Do not add `--enforce-eager`.** Regional compile is the default
    (`--diffusion-compile-granularity regional`) and is worth ~9.5%. The first request
    includes compilation — warm the server once before measuring steady state.
  - **`--cfg-parallel-size` must stay 1.** H3 is CFG-distilled and has no negative branch;
    anything greater is rejected with an explicit error rather than silently duplicating a
    branch that does not exist.
  - **The VAE supports its native `tile` mode only** — not `spatial_shard_width` or
    `spatial_shard_height` (those belong to the distributed Wan autoencoder). Patch
    parallel size must be 1 or the full DiT group size.

  ## Launch — single GPU (accuracy and memory first)

  Model-level CPU offload keeps the Qwen3-VL encoder and the DiT from being resident
  simultaneously. This is the accuracy-qualified reference path; it trades PCIe/NVLink
  transfer latency for a much smaller GPU footprint, and needs enough system RAM for the
  offloaded components.

  ```bash
  CUDA_VISIBLE_DEVICES=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  vllm serve /path/to/MiniMax-H3 \
    --omni \
    --task-type fl2va \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --num-gpus 1 \
    --enable-cpu-offload \
    --diffusion-attention-backend FLASH_ATTN
  ```

  ## Launch — DGX Spark (GB10, 128 GB unified memory)

  GB10 is a **unified-memory** platform: the 128 GB LPDDR5X (≈121 GiB usable) is shared by
  CPU and GPU, so both offload strategies used elsewhere in this recipe stop being capacity
  levers. `--enable-cpu-offload` and distributed layerwise offload move weights from "VRAM"
  to "host RAM", which is the same physical pool here — nothing is freed. DLO is actively
  harmful: staging rank-local shards into pinned host memory creates a second copy
  alongside the resident one, and the process is killed by the OOM killer (`Exit code: -9`)
  right after `Enabling offloader backend`.

  One partition on disk is 135 GiB in BF16:

  | Component | Size | Quantized by `--quantization fp8` |
  |---|---:|---|
  | `text_encoder` (Qwen3-VL) | 63 GiB | no |
  | `transformer` (DiT) | 62 GiB | yes |
  | `video_vae` | 9.8 GiB | no |
  | `audio_vae` | 0.6 GiB | no |

  135 GiB does not fit in 121 GiB, so **FP8 is mandatory** and only one DiT can be
  resident — always pass `--task-type fl2va` or `--task-type ref2va`. Online FP8 touches
  the DiT only, bringing the partition to roughly 104 GiB before activations.

  Weight quantization runs at load time and is slow on this platform; `--init-timeout 3600`
  keeps startup from being cut short.

  ```bash
  CUDA_VISIBLE_DEVICES=0 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
  vllm serve /path/to/MiniMax-H3 \
    --omni \
    --task-type fl2va \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --init-timeout 3600 \
    --num-gpus 1 \
    --usp 1 \
    --ring 1 \
    --quantization fp8 \
    --vae-patch-parallel-size 1 \
    --vae-parallel-mode tile \
    --vae-use-tiling \
    --enforce-eager \
    --diffusion-attention-backend CUDNN_ATTN
  ```

  Restart with `--task-type ref2va` to serve the Ref2VA family. Keep requests at a 576 px
  short edge (1024×576) and 5 s duration as the starting shape; GB10 has one GPU and no
  parallelism headroom, so wall-clock per request is far higher than the multi-GPU profiles
  above.

  ## Launch — single Ascend 950PR NPU (INT8)

  vLLM-Omni serves H3 on Ascend through the `vllm-ascend` platform plugin, on a
  CPU-only aarch64 PyTorch build with `torch_npu` — no CUDA runtime is present.
  One 950PR holds a single DiT partition, so this profile serves the `FL2VA`
  directory and uses online INT8 in place of the sharding levers the multi-GPU
  profiles rely on.

  Three Ascend-specific constraints come before the command:

  - **Attention backend.** `CUDNN_ATTN`, `FLASHINFER_ATTN`, `TRTLLM_ATTN`, and the
  SageAttention backends are CUDA-only; selecting one on NPU fails with a
  `NotImplementedError` out of `forward_npu`. The NPU platform default resolves
  to `FLASH_ATTN` when MindIE-SD is installed and `TORCH_SDPA` otherwise, and
  `FLASH_ATTN` is where H3's NPU memory work lives: a mask-free packed varlen
  path that never materializes the quadratic padding mask, plus K/V prefix
  slicing. Install MindIE-SD *before* passing
  `--diffusion-attention-backend FLASH_ATTN` — the backend imports `mindiesd`
  lazily and otherwise raises `ImportError` at the first denoise step. Once
  installed the platform default picks it up on its own.
  - **Shared memory.** Decoded video crosses the diffusion-worker process boundary
  through POSIX shared memory — roughly 0.9 GiB for a 5 s 1024×576 clip. The
  64 MiB Docker default is large enough for the control-plane queues, so the
  server starts and the denoise loop reaches 100% before the write into
  `/dev/shm` raises SIGBUS and kills the worker with `Bus error (core dumped)`.
  Check `df -h /dev/shm` and start the container with `--shm-size=8g` or
  `--ipc=host`. It cannot be changed on a running container, and an in-container
  `mount -o remount` needs `CAP_SYS_ADMIN`.
  - **No `--enforce-eager`.** It disables regional compilation and only slows the
  steady state; the first request is a warmup either way. `--cfg-parallel-size`
  must stay 1, as everywhere else in this recipe.

  ```bash
  ASCEND_RT_VISIBLE_DEVICES=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
  vllm serve /path/to/MiniMax-H3/FL2VA \
    --omni \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --num-gpus 1 \
    --tensor-parallel-size 1 \
    --usp 1 \
    --ring 1 \
    --text-encoder-tp-size 1 \
    --vae-patch-parallel-size 1 \
    --vae-parallel-mode tile \
    --vae-use-tiling \
    --init-timeout 14400 \
    --stage-init-timeout 14400 \
    --quantization int8
  ```

  ## Text-encoder tensor parallelism

  By default the whole Qwen3-VL encoder is resident on the DiT main rank, making it the
  peak-memory hotspot on multi-GPU no-offload runs. `--text-encoder-tp-size N` shards it
  across the first `N` DiT ranks using vLLM-style tensor-parallel layers on a dedicated
  encoder process group:

  ```bash
  vllm serve /path/to/MiniMax-H3 \
    --omni --trust-remote-code \
    --task-type fl2va \
    --num-gpus 4 --usp 4 --ring 1 \
    --text-encoder-tp-size 4 \
    --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling
  ```

  - `N` must divide the Qwen3-VL head counts (64 attention heads / 8 KV heads) → 1, 2, 4 on
    a 4-GPU server; 1, 2, 4, 8 on 8 GPUs.
  - On 4 GPUs, `N=4` drops the per-rank engine peak from **133 GB to 103 GB** (~35.7 GB) at
    no measurable throughput cost, while each non-main rank gains ~51.5/N GB.
  - Numerically: `N=1` reproduces the HF reference path bit-exactly (`max_abs = 0`); `N=4`
    introduces bounded BF16 rounding only (row-parallel all-reduce in FP32), giving
    31.11 dB PSNR / 0.9566 SSIM end-to-end against `N=1` with identical structure.
  - Combining `N>1` with regional compile used to produce a `ValueError: v must be finite`
    on the *second* request. That is fixed — the token refiner's attention runs on
    replicated rows before `sp_prepare` and was wrongly getting a Ulysses SP all-to-all.

  ## Key request parameters

  | Parameter | Recommended | Notes |
  |---|---|---|
  | `task` | `t2va`, `fl2va`, `ref2va` | Passed inside `extra_params`; must be compatible with `--task-type` when one is set |
  | `duration` | 4–15 s | Decimal seconds in `extra_params`; snapped to H3's legal `17n+5` frame count (15 s → 362 frames / 15.083 s) |
  | `fps` | `24` | Output FPS is fixed |
  | `num_inference_steps` | `50` | Matches the reference accuracy workloads |
  | `flow_shift` | `12` | Video sigma shift |
  | `audio_flow_shift` | `3` | Audio sigma shift, in `extra_params` |
  | `width`, `height` | see below | Server-side validation: multiples of 32, aspect ratio 1:4 to 4:1 |
  | `seed` | fixed value | Output is deterministic per request at a fixed seed |

  On `width`/`height`, the serving-side check is broader than the documented product modes —
  pick a **short edge of 768 or 1440** in one of the supported ratios (21:9, 16:9, 4:3, 1:1,
  3:4, 9:16) to stay on shapes the model was trained and validated for. FL2VA can omit both
  and inherit the first frame's aspect ratio at a 768 px short edge. Prompts cap at 7,000
  characters.

  Use `POST /v1/videos/sync` when you want the MP4 in the response body; `POST /v1/videos`
  is the async job-polling variant.

  ## Validated performance (4×B300)

  These measurements used task-specific single-DiT processes before the modular
  combined-service merge; they remain the validated per-task operating points.

  | Workload | Config | Result |
  |---|---|---|
  | FL2VA, 209 frames, 1248×768 (8.7 s) | no offload, U4, VPP4 tile, regional compile | **86.96 s** client E2E (±0.23 s) |
  | Two-video Ref2VA, 362 frames, 1344×768 (15 s) | same | **784.4 s** accounted model-stage mean |

  Stage split for the FL2VA case: text encoder 0.21 s, visual encoder 0.20 s, DiT 79.1 s
  (**88%** of the request), video+audio VAE decode 2.40 s. VAE patch parallelism is the
  cheapest win available — it cuts decode 3.4–3.5× (8.24 s → 2.40 s).

  ### Validated performance (4×MI300X)

  Measured with `vllm/vllm-omni-rocm:minimax-h3`
  (`sha256:29d1946af9c69e3e0a7128c247bdc8c82437ead43fbc366fed443533cf6ce9e8`),
  BF16, AITER, U4, text-encoder TP4, VPP4 tile, regional compile, one warmup, and
  the synchronous Video API:

  | Workload | Model stages | Client E2E |
  |---|---:|---:|
  | T2VA, 209 frames, 1344×768 (8.7 s) | encode 0.09 s, denoise 244.04 s, decode 4.15 s | **267.42 s** |
  | FL2VA, 209 frames, 1344×768 (8.7 s) | encode 13.98 s, denoise 257.58 s, decode 4.11 s | **287.07 s** |

  Both outputs were validated as 209-frame H.264 at 24 FPS with 32 kHz stereo AAC.
  A v0.26.0 source build additionally completed image+audio and video-reference Ref2VA,
  plus a 2048×1088 T2VA run. These measurements describe tested shapes, not a general
  latency guarantee.

  A `with_stack` profile of the two-video Ref2VA case shows the pipeline is
  **GPU-attention-bound, not host-starved**: FlashAttention-4 is ~76% of `diffuse` device
  time, Ulysses send/recv is secondary, GPU utilization is 92.8–94.3% and the CPU-idle
  union inside the transformer forwards is 3.15%. Reducing DiT time further needs a change
  in attention cost, which is not lossless;
  [SAGE + Skip-Softmax](#optional-lossy-attention-sage--skip-softmax) above is the supported
  way to trade attention precision and sparsity for speed on datacenter Blackwell.

  Against the checkpoint's own reference implementation, the accuracy-qualified path scores
  SSIM 0.9873–0.9896, PSNR 39–42 dB, pixel cosine 0.9996+, and audio log-mel cosine
  0.977–0.996 on T2VA/FL2VA. The two-video Ref2VA composite scores lower on raw pixels
  (SSIM 0.628) largely because of H.264 bitrate differences, while CLIP cosine is 0.9816 and
  audio log-mel cosine 0.9869 — content and motion stay aligned.

  ### Cache-DiT

  Opt-in only. A 50-step T2VA probe cut denoise time **30.2%** (121.01 s → 84.47 s) but
  dropped SSIM to **0.831** against the uncached output. H3's video trajectory is sensitive
  to cached denoise steps — choose the cache budget against your own quality target.

  ### Offload and accuracy

  On a **single GPU**, model-level CPU offload is the accuracy-qualified path. On four GPUs,
  no-offload is both faster and accuracy-clean once the encoder stays resident: an earlier
  no-offload build performed a ~63 GiB DiT GPU→CPU→GPU round trip inside `encode_prompt`,
  which was pure overhead (encoder 30.38 s → 0.207 s once removed, ~146×).

  ## Known limitations

  - The modular root service loads both DiTs by default. Capacity profiles use
    `--task-type fl2va` or `--task-type ref2va` and expose only that task family.
  - H3 executes **one generation request per diffusion batch** today.
  - The first request is a compile warmup only on the resident regional-compile profiles.
    The two-GPU DLO recipes use `--enforce-eager` and do not compile the DiT.
  - **The serving path accepts fewer references than the model supports.** H3 documents up
    to 9 images, 3 video clips, and 3 audio clips (12 files) per Omni Reference request;
    the current vLLM-Omni path takes exactly one image plus one audio reference, **or** one
    or more videos with **no** separate `audio_reference` (it uses the source soundtracks).
  - The 768 px short-edge mode is available for T2VA and FL2VA; 1344×768 is the documented
    16:9 request shape. The 24 GB DLO recipe uses 1024×576 as its lower-memory starting
    point.
  - `--cfg-parallel-size > 1` is rejected by design (CFG-distilled, no negative branch).
  - The VAE supports the native `tile` parallel mode only.
  - A `U2 × Ring2` hybrid currently fails with an attention-mask length mismatch; use pure
    Ulysses.
  - FastH3 preview v1 serves T2VA only, is fused at startup, and cannot use model-level,
    layerwise, or distributed layerwise offload. Its VSA variants additionally require CUDA,
    `fastvideo-kernel`, `FASTVIDEO_VSA`, and pure local/Ulysses attention; Ring and AllGather
    sequence parallelism are rejected.
  - Online FP8 quantizes the DiT only — the Qwen3-VL text encoder and both VAEs stay
    BF16. It is incompatible with layerwise offload, whose weight stride is rejected by
    the Cutlass FP8 kernel.
  - On unified-memory platforms (DGX Spark / GB10) CPU offload and DLO give no capacity
    benefit, because host RAM and VRAM are one pool; DLO additionally doubles peak usage
    while staging. Use resident FP8 with a single `--task-type` partition instead.
  - Pure Ulysses still replicates the 66.3 GB DiT on every rank, so 64 GB GPUs cannot use
    `--usp N --tp 1` as a resident capacity path. Use DiT TP plus DLO (the two-GPU recipes
    above), or model-level CPU offload; text-encoder TP alone is not sufficient.
  - DLO is a capacity path, not a free speedup: it needs substantial host RAM and streams
    weights over PCIe on every denoise step. Increase `--dlo-resident-layers` only after
    checking peak HBM on the exact request shape.

  ## References

  - [Model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  - [Modular FL2VA/Ref2VA pipeline (vLLM-Omni #5720)](https://github.com/vllm-project/vllm-omni/pull/5720)
  - [vLLM-Omni Video API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/videos_api/)
  - [vLLM-Omni diffusion parallelism](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/parallelism/overview/)
  - [vLLM-Omni diffusion attention backends](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/attention_backends/)
  - [FastVideo VSA attention guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/attention_backends/fastvideo_vsa/)
  - [FastH3 preview adapters](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA)
  - [FastH3 family and deployment guides](https://github.com/lishunyang12/vllm-omni/blob/pr/fasth3-omni-upstream/recipes/FastVideo/README.md)
  - [FastH3 Omni and V2 integration (vLLM-Omni #9167)](https://github.com/vllm-project/vllm-omni/pull/9167)
  - [MiniMax-H3 VSA and Ulysses support (vLLM-Omni #6909)](https://github.com/vllm-project/vllm-omni/pull/6909)
  - [vLLM-Omni TRTLLM attention guide (SAGE, Skip-Softmax)](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/attention_backends/trtllm/)
  - [MiniMax-H3 production serving on vLLM-Omni (blog)](https://vllm.ai/blog/2026-09-01-minimax-h3-production-serving)
  - [vLLM-Omni supported models](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/)
  - [vLLM-Omni GPU installation](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/)
  - [ROCm H3 tracking issue](https://github.com/vllm-project/vllm-omni/issues/5697)

```


---

## FILE: recipes/MiniMaxAI/minimax-h3-ref2va-8-rank-resident.yaml (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/MiniMaxAI/minimax-h3-ref2va-8-rank-resident.yaml)

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Single-stage Ref2VA: text TP8, DiT USP8, resident video/audio VAEs.
# Parameters from the 8x Ascend 910B1 (64 GiB/device) experiment.
# Use the existing serving container's Ascend runtime environment.
# vllm serve "$MODEL" --omni --config recipes/MiniMaxAI/minimax-h3-ref2va-8-rank-resident.yaml

host: "127.0.0.1"
port: 19100
trust-remote-code: true
task-type: ref2va
num-gpus: 8
usp: 8
ring: 1
text-encoder-tp-size: 8
vae-parallel-mode: tile
vae-use-tiling: true
vae-patch-parallel-size: 8
enable-distributed-layerwise-offload: true
diffusion-attention-backend: FLASH_ATTN

# The CLI accepts this argument as JSON. Selecting only these two components
# keeps video/audio VAEs resident and preserves rank-local text weight transfer.
diffusion-offload-config: >-
  {"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"rank-local"}}}

```
