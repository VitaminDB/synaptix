# vllm-project__vllm-omni__recipes-consumer-gpus

Source: https://github.com/vllm-project/vllm-omni/tree/main/recipes/MiniMaxAI
Fetched: 2026-11-28



---

## FILE: recipes/MiniMaxAI/MiniMax-H3-4090.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/MiniMaxAI/MiniMax-H3-4090.md)

# MiniMax-H3 on RTX 4090

This recipe uses BF16 weights, tiled VAE decode, tensor parallelism, optional
Ulysses sequence parallelism, and distributed layerwise offload (DLO). It is the
24 GiB sibling of [MiniMax-H3-5090.md](MiniMax-H3-5090.md): the two-GPU topology
matches that recipe, but the resident DiT-block count is lower so the per-rank
peak fits in 24 GiB. Lower resident counts reduce HBM use and increase
CPU-to-GPU transfer time.

## Capacity requirements

| Resource | Two RTX 4090s | Four RTX 4090s |
| --- | ---: | ---: |
| GPU HBM | 24 GiB per GPU | 24 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | 200 GiB minimum | 200 GiB minimum |
| Recommended system RAM | 384 GiB | 384 GiB |

`FL2VA` and `Ref2VA` are separate 135 GiB checkpoint partitions. Start one
server at a time. DLO keeps rank-local weights in pinned host memory; increasing
`--dlo-resident-layers` improves latency but does **not** reduce host RAM in the
current implementation because resident layers retain pinned CPU master copies.

A single RTX 4090 is not covered here. The one-GPU DLO profile in the RTX 5090
recipe peaked at 26.50 GiB with 12 resident layers, which exceeds 24 GiB. On one
4090, use the model-level CPU offload command in
[MiniMax-H3.md](MiniMax-H3.md#single-gpu-accuracy-and-memory-first) instead, or
lower the resident count and re-measure before trusting it.

> **Modular H3:** after #5720 lands, preserve this recipe's one-partition
> behavior with `--task-type fl2va` or `--task-type ref2va`.

## Two RTX 4090s: 1024x576, 5 seconds

Use TP2 and 12 resident DiT layers. This is the `rtx4090` profile from the main
recipe. On two RTX 4090s it peaked at 15,620 MiB per rank for T2VA and
14,918 MiB per rank for Ref2VA, leaving roughly 9 GiB of headroom per card. See
[Target-hardware validation](#target-hardware-validation) for the full numbers.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400

CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --usp 1 --ring 1 --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 12 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

For Ref2VA, stop the FL2VA server and restart the same command with
`/path/to/MiniMax-H3/Ref2VA`. Ref2VA reference image count and prompt length can
increase activation memory; begin with one request at a time.

`VLLM_OMNI_VIDEO_SYNC_TIMEOUT` matters more here than on larger cards. DLO
streams non-resident DiT blocks over PCIe on every denoising step, so a 60-step
request takes several minutes and would otherwise hit the 600-second default.

## Four RTX 4090s: 1024x576, 5 seconds

Keep TP2 and raise Ulysses sequence parallel to 2 so the world size is
`TP × USP = 4`. Text-encoder TP and VAE patch parallel follow the GPU count.
Resident layers stay at 12: adding USP does not shard DiT weights further, so
per-GPU peak HBM stays close to the two-GPU profile while wall-clock improves.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 4 --tensor-parallel-size 2 --text-encoder-tp-size 4 \
  --usp 2 --ring 1 --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 12 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

For Ref2VA, stop the FL2VA server and restart the same command with
`/path/to/MiniMax-H3/Ref2VA`. Pin four dedicated GPUs; if the launcher exposes
fewer than four devices, startup fails with
`Stage 0 requires 4 device(s) based on parallel_config`.

## Target-hardware validation

Measured on RTX 4090 (24,564 MiB each, driver 580.126.09) with the serve
commands above, at vLLM-Omni `0.26.1.dev55+g81b48e83e`, vLLM `0.26.0`, and
PyTorch `2.11.0+cu130`. GPUs used by each server were dedicated. `Client E2E`
and `Peak per GPU` come from the `/v1/videos/sync` response headers
`x-inference-time-s` and `x-peak-memory-mb`; the latter is the rank-0 CUDA
reserved high-water mark, so it is a per-GPU figure.

| GPUs | Topology | Task | Shape | Frames | Steps | Client E2E | Peak per GPU | Output validation |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | TP2 × USP1 | T2VA (`FL2VA`) | 1024x576 | 124 at 24 FPS | 60 | 7 min 9 s | 15.3 GiB | H.264 video + 32 kHz stereo AAC; full `ffmpeg` decode passed |
| 2 | TP2 × USP1 | Ref2VA | 1024x576 | 124 at 24 FPS | 60 | 14 min 52 s | 14.6 GiB | H.264 video + 32 kHz stereo AAC; full `ffmpeg` decode passed |
| 4 | TP2 × USP2 | T2VA (`FL2VA`) | 1024x576 | 124 at 24 FPS | 60 | 4 min 29 s | 15.2 GiB | H.264 video + 32 kHz stereo AAC; full `ffmpeg` decode passed |
| 4 | TP2 × USP2 | Ref2VA | 1024x576 | 124 at 24 FPS | 60 | 9 min 5 s | 16.1 GiB | H.264 video + 32 kHz stereo AAC; successful `/v1/videos/sync` 200 |

All rows use `seed=1101`, `flow_shift=12`, and `audio_flow_shift=3.0`.

Two-GPU repeats: T2VA measured 434.9 s and 429.3 s; Ref2VA measured 891.9 s and
892.1 s. Peak memory was identical across repeats for both tasks, and the two
outputs of each pair agreed to within 115 bytes.

Four-GPU T2VA repeats: 273.8 s and 269.5 s, peaking at 15,560 MiB and
15,612 MiB. Four-GPU Ref2VA is a single successful end-to-end run (545.2 s,
16,448 MiB). Follow-up Ref2VA requests on this build completed diffusion but
then hit the engine's async output wait during post-compute D2H, so they are
not reported as repeats. That wait was a hardcoded 30 s when these numbers were
taken; it is now `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` (default 600 s), so a rerun on
current main should not lose these repeats to the bound.

Relative to the two-GPU baseline on the same shape, four-GPU TP2×USP2 was about
1.6× faster for both tasks, while per-GPU peak HBM stayed in the mid-15 GiB
range because USP does not further shard resident DiT weights. These are
single-request validation runs, not concurrent throughput benchmarks.

## Request examples

`t2va` requires an explicit `aspect_ratio`. Without it the request fails with
`t2va requires an explicit aspect_ratio`, even when `width` and `height` are set.

Text to video and audio, against the `FL2VA` server:

```bash
curl -sS -D headers_t2va.txt -o out_t2va.mp4 --max-time 14400 \
  -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=傍晚的小厨房，夕阳余晖从窗边洒进来，旧木桌上放着洗到一半的马克杯和起雾的玻璃瓶，悬挂的抹布轻轻晃动。画面带有手持拍摄的轻微晃动和逆光曝光波动，生活感十足。环境音是安静的厨房底噪。' \
  -F 'aspect_ratio=16:9' \
  -F 'width=1024' -F 'height=576' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5,"audio_flow_shift":3.0}'
```

Reference to video and audio, against the `Ref2VA` server:

```bash
curl -sS -D headers_ref2va.txt -o out_ref2va.mp4 --max-time 14400 \
  -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=傍晚的小厨房，夕阳余晖从窗边洒进来，旧木桌上放着洗到一半的马克杯和起雾的玻璃瓶，悬挂的抹布轻轻晃动。画面带有手持拍摄的轻微晃动和逆光曝光波动，生活感十足。环境音是安静的厨房底噪。' \
  -F 'input_reference=@/path/to/reference.jpg;type=image/jpeg' \
  -F 'audio_reference=</path/to/audio_reference.json' \
  -F 'width=1024' -F 'height=576' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"ref2va","duration":5,"audio_flow_shift":3.0}'
```

`audio_reference` is a JSON object, which is why the example loads it from a file
with curl's `<` syntax. Its `audio_url` must be an `http(s)` URL or a data URL;
a bare filesystem path is rejected with `Invalid audio_reference.audio_url`.

```json
{"audio_url": "https://example.com/reference.wav"}
```

Reference images are validated before inference: the short edge must be at least
256 pixels, the long edge at most 5760, and the aspect ratio must fall between
0.4 and 2.5.

The response headers carry the measurements used in the table above:

```bash
cat headers_t2va.txt
ffmpeg -v error -i out_t2va.mp4 -f null - && echo DECODE_OK
```


---

## FILE: recipes/MiniMaxAI/MiniMax-H3-5090.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/MiniMaxAI/MiniMax-H3-5090.md)

# MiniMax-H3 on RTX 5090

This recipe uses BF16 weights, tiled VAE decode, tensor parallelism where a
second GPU is available, and distributed layerwise offload (DLO). It is a
memory-first serving configuration; lower resident counts reduce HBM use and
increase CPU-to-GPU transfer time.

## Capacity requirements

| Resource | One RTX 5090 | Two RTX 5090s |
| --- | ---: | ---: |
| GPU HBM | 32 GiB | 32 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | 200 GiB minimum | 200 GiB minimum |
| Recommended system RAM | 384 GiB | 384 GiB |

`FL2VA` and `Ref2VA` are separate 135 GiB checkpoint partitions. Start one
server at a time. DLO keeps rank-local weights in pinned host memory; increasing
`--dlo-resident-layers` improves latency but does **not** reduce host RAM in the
current implementation because resident layers retain pinned CPU master copies.

> **Modular H3:** after #5720 lands, preserve this recipe's one-partition
> behavior with `--task-type fl2va` or `--task-type ref2va`.

## One RTX 5090: 1344x768, 5 seconds

Use 12 resident DiT layers. A 50-step B300 allocation test with this exact
single-rank topology peaked at 26.50 GiB; re-measure peak HBM on the target
card before increasing the resident count.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
  --usp 1 --ring 1 --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 12 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

## Two RTX 5090s: 1344x768, 5 seconds

Use TP2 and 20 resident DiT layers. The two-rank B300 capacity run peaked at
27,726 MiB per rank for this shape and 50 steps. This is a memory/correctness
proxy, not a consumer-GPU latency claim.

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --usp 1 --ring 1 --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 20 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

For Ref2VA, stop the FL2VA server and restart the same command with
`/path/to/MiniMax-H3/Ref2VA`. Ref2VA reference video count and prompt length
can increase activation memory; begin with one request at a time.


---

## FILE: recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-5000.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-5000.md)

# MiniMax-H3 on RTX PRO 5000 Blackwell GPUs

This recipe runs MiniMax-H3 in BF16 on 72 GiB RTX PRO 5000 Blackwell GPUs. It
contains the validated two-GPU DLO configuration and the recommended resident
configurations: TP1 x Ulysses2 with 20 resident layers on two GPUs, TP2 x
Ulysses2 on four GPUs, and TP4 x Ulysses2 on eight GPUs. The four- and
eight-GPU routes do not require offload.

## Capacity requirements

| Resource | Requirement |
| --- | ---: |
| GPUs | 2, 4, or 8 x RTX PRO 5000 Blackwell |
| GPU HBM | 72 GiB per GPU |
| Checkpoint storage | 135 GiB per partition |
| Available system RAM | 200 GiB minimum |
| Recommended system RAM | 384 GiB |

`FL2VA` and `Ref2VA` are separate checkpoint partitions. Start one server at a
time on a host sized for the minimum system-memory requirement.

## PCIe topology and GPU order

RTX PRO 5000 does not provide NVLink. Before starting the server, identify the
closest PCIe pairs and their NUMA affinity:

```bash
nvidia-smi topo -m
nvidia-smi nvlink -s
```

For two GPUs, select the closest PCIe pair on one NUMA node. For four GPUs,
select all cards from one NUMA node. On the validated host,
physical GPUs `(0,1)` and `(2,3)` are the two `PXB` pairs on NUMA node 0. The
order `CUDA_VISIBLE_DEVICES=0,2,1,3` maps the Ulysses groups to those local
pairs. For eight GPUs, the validated order is `0,4,1,5,2,6,3,7`, with host
memory interleaved across NUMA nodes 0 and 1. Do not copy these IDs blindly:
reproduce the same PCIe and NUMA relationships on the target host.

## Recommended serving configurations

### Two GPUs

Two 72 GiB cards require distributed layerwise offload. The validated route
uses TP1 x Ulysses2, keeps 20 leading DiT layers resident, and streams
rank-local weights without AllGather. Eager execution avoids regional-compile
instability on this offload path.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
numactl --cpunodebind=0 --membind=0 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 2 \
  --tensor-parallel-size 1 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --dlo-resident-layers 20 \
  --enforce-eager
```

### Four GPUs

The validated baseline uses TP2 x Ulysses2, text-encoder TP4, VAE patch
parallelism 4, and explicit cuDNN BF16 attention. Selecting the backend
explicitly keeps the recipe independent of platform-default backend changes.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,2,1,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
numactl --cpunodebind=0 --membind=0 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --tensor-parallel-size 2 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN
```

### Eight GPUs

The recommended eight-GPU route uses TP4 x Ulysses2, text-encoder TP8, VAE
patch parallelism 8, and host-memory interleaving across both NUMA nodes.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,4,1,5,2,6,3,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
numactl --interleave=0,1 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --tensor-parallel-size 4 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN
```

For the four- and eight-GPU resident routes, do not add `--enforce-eager`.
Warm the server once before measuring so regional compilation is outside the
measured request. The two-GPU DLO route intentionally remains eager.

For Ref2VA, stop the FL2VA server and restart the same command with
`MODEL="${MODEL_ROOT}/Ref2VA"`.

## Target-hardware validation

All three configurations were exercised on a PCIe-only, dual-socket host with
eight RTX PRO 5000 GPUs. The run used PyTorch 2.11.0+cu130, CUDA 13.0, driver
580.95.05, 1344x768 output, 124 frames, and two warmups.

The four-GPU route also received a five-step profiling validation for GPU
balance:

| Measurement | Result |
| --- | ---: |
| Maximum externally sampled peak | 69,219 MiB (67.60 GiB) per GPU |
| T2VA worker-reported peak | 65,314 MiB |
| First-frame FL2VA worker-reported peak | 66,780 MiB |
| T2VA maximum GPU kernel-time deviation | 0.56% |
| First-frame FL2VA maximum GPU kernel-time deviation | 0.09% |

The recommended routes produced the following 50-step results. MiniMax-H3
requested 50 denoise steps and executed 49 denoise updates, so per-step latency
is `denoise / 49`.

| GPUs | Workload | Parallelism | E2E (s) | Text encode (s) | Denoise (s) | VAE decode (s) | Per step (ms) | Peak memory (GiB) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | T2VA | TP1 x Ulysses2, DLO resident 20 | 515.57 | 1.19 | 504.69 | 8.79 | 10,300 | 36.38 |
| 2 | FL2VA first-frame I2VA | TP1 x Ulysses2, DLO resident 20 | 553.45 | 1.27 | 541.90 | 8.76 | 11,059 | 36.38 |
| 4 | T2VA | TP2 x Ulysses2 | 284.01 | 0.03 | 278.73 | 4.36 | 5,688 | 67.58 |
| 4 | FL2VA first-frame I2VA | TP2 x Ulysses2 | 305.72 | 0.28 | 299.85 | 4.36 | 6,119 | 67.58 |
| 8 | T2VA | TP4 x Ulysses2 | 163.20 | 0.04 | 159.40 | 2.58 | 3,253 | 45.83 |
| 8 | FL2VA first-frame I2VA | TP4 x Ulysses2 | 171.62 | 0.27 | 167.25 | 2.57 | 3,413 | 45.83 |

Peak memory is the maximum per-GPU value sampled externally with
`nvidia-smi`. The two-GPU route leaves about 34.7 GiB below the reported
73,415 MiB device capacity, the four-GPU route leaves about 4.1 GiB, and the
eight-GPU route leaves about 25.9 GiB. Re-measure memory for longer reference
inputs, concurrency greater than one, or a different output shape.

## T2VA request example

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"

curl -sS --max-time 1800 -X POST "${API_URL}" \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```


---

## FILE: recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-6000.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-6000.md)

# MiniMax-H3 on RTX PRO 6000 Blackwell GPUs

This recipe runs MiniMax-H3 in BF16 on 96 GiB RTX PRO 6000 Blackwell GPUs.
The additional HBM over the RTX PRO 5000 profile removes the four-GPU
minimum: two GPUs are enough to keep the model resident with TP2. Four and
eight GPUs raise the Ulysses degree to shard the attention sequence further.
CPU offload and distributed layerwise offload are not required in any of
these configurations.

Validated on:
- Host: YLX Y762
- GPUs: 8 × RTX PRO 6000 Blackwell (96 GiB)
- Device order: default (`CUDA_VISIBLE_DEVICES` not set)
- Driver / CUDA: 580.105.08 / 13.0
- Container image: `vllm/vllm-omni:minimax-h3`
- Workload: T2VA, 1344×768, `duration=5.0`, 50 steps, `flow_shift=12`, `seed=1101`

## Capacity requirements

| Resource | Two GPUs | Four GPUs | Eight GPUs |
| --- | ---: | ---: | ---: |
| GPU HBM | 96 GiB per GPU | 96 GiB per GPU | 96 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | 200 GiB minimum | 200 GiB minimum | 200 GiB minimum |
| Recommended system RAM | 384 GiB | 384 GiB | 384 GiB |
| Measured peak HBM per GPU | 77.49 GiB | 66.44 GiB | 61.07 GiB |

`FL2VA` and `Ref2VA` are separate checkpoint partitions. Start one server at
a time on a host sized for the minimum system-memory requirement, or pass
`--task-type fl2va` or `--task-type ref2va` to download and load a single
partition. The two-server layout described in the eight-GPU section doubles
both the storage and the host-memory requirement.

## Parallelism summary

| Configuration | TP | Ulysses | Text-encoder TP | VAE patch parallel |
| --- | ---: | ---: | ---: | ---: |
| Two GPUs | 2 | 1 | 2 | 2 |
| Four GPUs | 2 | 2 | 4 | 4 |
| Eight GPUs | 2 | 4 | 8 | 8 |
| Eight GPUs, headroom variant | 4 | 2 | 8 | 8 |

Per-GPU weight residency is set by the tensor-parallel degree alone. Raising
the Ulysses degree lowers activation memory and per-step latency but leaves
weight residency unchanged. Ring parallelism stays at 1, and H3 is
CFG-distilled, so `--cfg-parallel-size` must also remain 1.

## PCIe topology and GPU order

RTX PRO 6000 Blackwell does not provide NVLink. Before starting the server,
inspect the host topology:

```bash
nvidia-smi topo -m
nvidia-smi nvlink -s
```

Tensor-parallel groups are consecutive ranks. Ulysses groups are strided by
the tensor-parallel degree. Confirm this grouping against the server log
before tuning the device order on a new host.

For the two-GPU configuration, pick a single `PXB` pair on one NUMA node.
For the four-GPU configuration, pick two `PXB` pairs on the same node and
order them so that each Ulysses group lands inside one pair. If physical
GPUs `(0,1)` and `(2,3)` are the two local pairs, the order is
`CUDA_VISIBLE_DEVICES=0,2,1,3`. Do not copy these IDs blindly: reproduce the
same relationship on the target host.

Device-order pinning and NUMA binding are tuning knobs, not prerequisites.
The measurements in this recipe were taken without either, so any gain from
them is additional to the numbers reported below and is host-specific.

## Two-GPU serving configuration

Two 96 GiB GPUs hold the BF16 model with TP2 alone. There is no Ulysses
group, so the DiT attention sequence is not sharded and activation memory is
higher than in the four-GPU profile at the same output shape. Re-measure
peak memory before increasing the output shape, the reference-input length,
or concurrency.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8000

VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN
```

## Four-GPU serving configuration

Four GPUs keep TP2 for the weight shard and add Ulysses2, which shards the
attention sequence and lowers per-GPU activation memory. Text-encoder TP4
and VAE patch parallelism 4 spread the Qwen3-VL encoder and tiled decode
across all four GPUs. TP1 is not an option on this card: an unsharded BF16
partition does not fit in 96 GiB.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8000

VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --tensor-parallel-size 2 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN
```

## Eight-GPU serving configuration

Eight GPUs extend the Ulysses degree to 4 while keeping TP2. This does not
lower per-GPU weight residency relative to the four-GPU profile; it lowers
activation memory and per-step latency by sharding the attention sequence
four ways.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8000

VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --tensor-parallel-size 2 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN
```

### Rank ordering across two sockets

With TP2 and Ulysses4 the two Ulysses groups are ranks `(0,2,4,6)` and
`(1,3,5,7)`, and the four tensor-parallel pairs are `(0,1)`, `(2,3)`,
`(4,5)`, and `(6,7)`. On a dual-socket host these two collectives cannot
both be socket-local. If you choose to pin the device order, the trade-off
is:

| Device order | Socket-local collective | Crosses sockets |
| --- | --- | --- |
| `0,4,1,5,2,6,3,7` | Ulysses all-to-all | TP all-reduce |
| `0,1,2,3,4,5,6,7` | TP all-reduce | Ulysses all-to-all |

The first order assumes physical GPUs `0-3` are on NUMA node 0 and `4-7` on
node 1. Reproduce that relationship on the target host rather than copying
the IDs. The reported measurements use the default order and no NUMA
binding; the two orders were not compared on this host.

### Headroom variant: TP4 with Ulysses2

Raising the tensor-parallel degree to 4 halves per-GPU weight residency and
leaves room for long Ref2VA references, larger output shapes, or
concurrency greater than one. It also adds a four-way all-reduce per DiT
block over a link without NVLink. Use it when the TP2 profile runs out of
headroom, not as the default. Replace the parallel flags above with:

```bash
  --tensor-parallel-size 4 \
  --usp 2 \
```

### Two independent four-GPU servers

An eight-GPU host can instead run one FL2VA server and one Ref2VA server
side by side, each pinned to its own NUMA node with the four-GPU
configuration. This removes every cross-socket collective and lifts the
one-server-at-a-time restriction, at the cost of doubling the requirements
to 400 GiB available system RAM and 270 GiB of model storage.

Use `CUDA_VISIBLE_DEVICES=0,2,1,3`, `--cpunodebind=0 --membind=0`, and
`--port 8091` for the FL2VA server. Use `CUDA_VISIBLE_DEVICES=4,6,5,7`,
`--cpunodebind=1 --membind=1`, and `--port 8092` for the Ref2VA server.
This layout is described for completeness and was not part of the measured
runs.

## Shared serving notes

Do not add `--enforce-eager` for a performance run. Warm the server before
measuring so regional compilation is outside the measured request. On this
host the first request after startup ran 19% slower than the steady state
(107.58 s versus 90.48 s on eight GPUs); two warmup requests were enough to
converge.

For Ref2VA on a single-server layout, stop the FL2VA server and restart the
same command with `MODEL="${MODEL_ROOT}/Ref2VA"`.

## Attention backend

RTX PRO 6000 Blackwell is an SM120 part, so the datacenter-Blackwell
`TRTLLM_ATTN` default does not apply here. Every configuration above selects
cuDNN BF16 attention explicitly. This keeps the recipe independent of
platform-default backend changes and does not depend on the experimental
SM120 kernel.

## Optional: online FP8

`--quantization fp8` quantizes eligible DiT linears at load time and is
compatible with tensor parallelism and VAE tiling. Use it when the resident
BF16 peak leaves too little headroom for long Ref2VA references, a larger
output shape, or concurrency greater than one. It cannot be combined with
layerwise offload.

## Target-hardware validation

T2VA, 1344×768, `duration=5.0`, `fps=24`, 50 steps, `flow_shift=12`,
`seed=1101`, BF16, `CUDNN_ATTN`, tiled VAE, one request at a time. Servers
were started with the commands above — default device order, no NUMA
binding — plus `--enable-diffusion-pipeline-profiler`. Two warmup requests
preceded each measured request.

| Measurement | Two GPUs | Four GPUs | Eight GPUs |
| --- | ---: | ---: | ---: |
| Topology | TP2 × USP1 | TP2 × USP2 | TP2 × USP4 |
| Text encode | 0.040 s | 0.044 s | 0.035 s |
| Denoise, 50 steps | 278.55 s | 168.73 s | 87.90 s |
| Per step | 5.571 s | 3.375 s | 1.758 s |
| VAE decode | 5.396 s | 2.791 s | 1.798 s |
| Client E2E | 284.76 s | 172.32 s | 90.48 s |
| Peak HBM per GPU | 77.49 GiB | 66.44 GiB | 61.07 GiB |
| Headroom below 96 GiB | 18.5 GiB | 29.6 GiB | 34.9 GiB |

Stage times are read from the `X-Stage-Durations` response header of
`/v1/videos/sync`. Peak memory is the maximum of `nvidia-smi
--query-gpu=memory.used` sampled at 1 Hz across every device for the
duration of the measured request. Per step is denoise wall time divided by
the 50 requested steps. Stage times sum to roughly 0.75 s less than
end-to-end in all three configurations; queueing, result transfer, and MP4
muxing sit outside the profiled stages.

### Scaling behaviour

Denoise speedup is not uniform across the sweep: 1.65× from two to four
GPUs (83% efficiency) but 1.92× from four to eight (96%). Two GPUs run
Ulysses1, so there is no sequence parallelism at all; going to four GPUs
pays the one-time cost of introducing the all-to-all, while going from four
to eight only widens an all-to-all that already exists. Expect the second
doubling to pay off better than the first on this topology.

VAE decode behaves in the opposite direction — 1.93× from two to four GPUs
but only 1.55× from four to eight — as patch parallelism produces smaller
tiles and tile-boundary overhead grows.

Text encoding is 0.04 s in every configuration and is not worth optimising;
denoise is 96-98% of end-to-end throughout.

### Per-GPU memory model

Peak memory falls by 11.05 GiB from two to four GPUs and by a further
5.37 GiB from four to eight — halving each time the Ulysses degree doubles,
because all three configurations shard the DiT with TP2 and hold identical
weights per GPU. Fitting the first two points gives:

```
peak HBM per GPU ≈ 55.4 GiB + 22.1 GiB / ulysses_degree
```

That fit predicts 60.9 GiB at Ulysses4 against 61.07 GiB measured, a 0.25%
error. The practical consequence is a floor near 55 GiB per GPU: adding
GPUs past eight will not reduce the per-GPU requirement further. To serve
on smaller cards, raise the tensor-parallel degree instead.

Re-measure memory for longer reference inputs, concurrency greater than
one, or a different output shape.

### Not yet measured

Ref2VA latency and memory have not been measured on this host. Ref2VA
drives the Qwen3-VL presentation from tens of tokens for text-only prompts
to several thousand with image or video references, so T2VA numbers do not
transfer to it. The eight-GPU device-order comparison in the section above
was also not run.

## T2VA request example

`t2va` requires an explicit `aspect_ratio` even when `width` and `height`
are supplied.

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"

curl -sS --max-time 1800 -X POST "${API_URL}" \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

To collect the stage breakdown and peak memory for a request, add
`--enable-diffusion-pipeline-profiler` to the server command and `-D
headers.txt` to the curl invocation, then read `X-Stage-Durations` and
`X-Peak-Memory-MB` from the saved headers.

## Ref2VA request example

`ref2va` defaults to a 16:9 aspect ratio, so `aspect_ratio` is optional
here.

```bash
curl -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  --fail-with-body -w '\nHTTP %{http_code}\n' \
  --max-time 1200 \
  -F "input_reference=@/root/hand.jpg;type=image/jpeg" \
  -F "audio_reference=</root/audio_ref.json" \
  -F "prompt=2D动画融合在一起的影像。夕阳余晖残留在窗边，生活感十足的小厨房里有旧木桌、洗到一半的马克杯、起雾的玻璃瓶、悬挂的抹布。画面带有智能手机单手拍摄的手抖、近距离对焦的犹豫、逆光曝光波动。要像在家中慌忙拍下某个不可思议事件的自然质感，不要广告影像的精心整理。声音只用厨房环境声与手绘生物柔和的电子音、小小的叫声。" \
  -F 'width=1344' -F 'height=768' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"ref2va","duration":8,"audio_flow_shift":3.0}' \
  -o /root/out_ref2va.mp4
```


---

## FILE: recipes/MiniMaxAI/MiniMax-H3-Spark-GB10.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/MiniMaxAI/MiniMax-H3-Spark-GB10.md)

# MiniMax-H3 on DGX Spark (GB10)

This recipe uses online FP8 weight quantization, tiled VAE decode, and a single
resident partition. GB10 is a unified-memory platform, so unlike the discrete-GPU
recipes it uses **no offload of any kind** — see the capacity note below.

Validated on:

- Host: DGX Spark (GB10), aarch64
- GPUs: 1, or 2 hosts with one unified-memory GPU each
- Driver: 580.173.02 for the two-host qualification
- vLLM: 0.26.0
- vLLM-Omni: `main` at `e1aa6eae75c460cd1893bc320546e81e66973831`
- One-host workloads: T2VA and Ref2VA, 960x576, `duration=8.0`, 50 steps
- Two-host workload: T2VA, 1344x768, `duration=5.0` and `duration=10.0`,
  50 steps, text-encoder TP2, and DiT USP2
- Common sampling settings: `flow_shift=12`, `seed=1101`

The commands and measurements below describe this pinned revision. Newer
versions have changed the scope of `--quantization fp8`; see
[Online FP8 quantization](MiniMax-H3.md#online-fp8-quantization) for the current
component selection and offload support. Re-qualify memory, output, and latency
on GB10 before comparing results across revisions.

## Capacity requirements

| Resource | DGX Spark (GB10) |
| --- | ---: |
| Unified memory | 128 GB LPDDR5X (~121 GiB usable) |
| GPUs | 1 |
| Checkpoint storage | 135 GiB per partition |
| Measured peak allocator high-water, T2VA (FP8) | 97.7 GiB |
| Measured peak allocator high-water, Ref2VA (FP8) | 102.8 GiB |
| Measured peak per rank, two-host T2VA 5 s / 10 s (FP8) | 75.45 / 87.17 GiB |

`FL2VA` and `Ref2VA` are separate 135 GiB checkpoint partitions. Start one server
at a time.

On GB10 the CPU and GPU share one physical memory pool, which changes the capacity
math relative to the RTX and datacenter recipes:

- **Do not use `--enable-distributed-layerwise-offload`.** DLO stages rank-local
  weights into pinned host memory, which on GB10 is the same pool the weights are
  already in. Peak usage roughly doubles and the process is killed by the OOM
  killer (`Exit code: -9`) shortly after `Enabling offloader backend`.
- **Do not use `--enable-cpu-offload`.** Moving weights from "VRAM" to "host RAM"
  frees nothing here.
- **`--quantization fp8` is mandatory.** A BF16 partition is 135 GiB and does not
  fit in 121 GiB. At the validated revision, online FP8 quantizes the DiT only
  (62 GiB to ~31 GiB); the Qwen3-VL text encoder and both VAEs stay BF16.
  Current versions also quantize eligible text-decoder linears by default, so
  the same flag does not imply the same quantization scope across revisions.

With ~97.7 GiB of the pool held by the allocator at peak, T2VA leaves roughly
23 GiB of headroom for the OS, page cache, and any other process on the box.
Ref2VA is tighter on both counts. The allocator high-water rises to 102.8 GiB,
and the allocator figure understates true pool usage.

Sampling `/proc/meminfo` at 1 Hz across a 50-step Ref2VA request on a host with
`MemTotal: 127600524 kB` (121.7 GiB), `MemAvailable` started at 8.1 GiB, fell to
a trough of **7.4 GiB**, and recovered to 8.9 GiB. At the trough the pool held
roughly **114.3 GiB** — 11.4 GiB more than the 102.8 GiB the allocator reports,
which is the CUDA context, non-PyTorch allocations, and page cache the header
does not count.

Treat this platform as strictly single-tenant for Ref2VA. There is no room for a
second process, and adding reference images, reference videos, or a larger output
shape will run into the OOM killer.

## One GB10: 960x576, 8 seconds

```bash
CUDA_VISIBLE_DEVICES=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --init-timeout 3600 \
  --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
  --usp 1 --ring 1 --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile --vae-use-tiling \
  --quantization fp8 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

Wait for `Application startup complete`, then run a short smoke test before
committing to a full 50-step request.

`t2va` requires an explicit `aspect_ratio` even when `width` and `height` are
supplied; omitting it fails the request with
`OmniClientError: t2va requires an explicit aspect_ratio`.

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=960' \
  -F 'height=576' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=10' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":4.0,"audio_flow_shift":3.0}' \
  -o h3-gb10-smoke.mp4
```

The full-quality request uses 50 steps:

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=960' \
  -F 'height=576' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.0,"audio_flow_shift":3.0}' \
  -o h3-gb10-t2va.mp4

ffprobe -v error -show_entries \
  stream=index,codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json h3-gb10-t2va.mp4
```

Both requests return `200 OK` with an MP4 body: H.264 video at the requested
geometry plus a 32 kHz stereo AAC track.

## One GB10: Ref2VA, 960x576, 8 seconds

`FL2VA` and `Ref2VA` are separate 135 GiB partitions and only one fits at a time.
Stop the FL2VA server before starting this one.

```bash
CUDA_VISIBLE_DEVICES=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve /path/to/MiniMax-H3/Ref2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --init-timeout 3600 \
  --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
  --usp 1 --ring 1 --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile --vae-use-tiling \
  --quantization fp8 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

The 7200 s timeout used for T2VA is too short here: a 50-step Ref2VA request
takes over an hour on this platform. Use 14400 s.

Unlike `t2va`, `ref2va` defaults to a 16:9 output ratio, so `aspect_ratio` can be
omitted when `width` and `height` are supplied.

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8000/v1/videos/sync \
  -D ref2va-headers.txt \
  -w 'client_e2e_s=%{time_total}\n' \
  --max-time 14400 \
  -F 'input_reference=@/path/to/reference.jpg;type=image/jpeg' \
  -F 'prompt=A hand rests on a wooden table in warm evening light, then slowly opens and turns toward the camera, with quiet room ambience.' \
  -F 'width=960' \
  -F 'height=576' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"ref2va","duration":8.0,"audio_flow_shift":3.0}' \
  -o h3-gb10-ref2va.mp4
```

The response is `200 OK` with an MP4 body: H.264 at the requested geometry plus a
32 kHz stereo AAC track. `ffprobe` reports 8.032 s for `duration=8.0`.

## Two GB10 hosts: T2VA at 1344x768

The two-host qualification uses one GB10 GPU per DGX Spark, connected through a
single ConnectX-7 RoCE interface. The FL2VA checkpoint and software revision are
identical on both hosts. The distributed configuration is:

- world size 2;
- text-encoder tensor parallel size 2;
- DiT Ulysses sequence parallel size 2 and ring degree 1;
- VAE patch parallel size 1;
- online FP8, eager execution, cuDNN attention, and tiled VAE decode.

The current diffusion multiprocess executor starts all ranks on one host and sets
`MASTER_ADDR=localhost`. These measurements therefore used an external `env://`
launcher with one rank on each host and a benchmark-only adaptation that maps
both hosts' local worker to `cuda:0`. This qualifies the two-rank MiniMax-H3
pipeline and its output, but it is not yet a supported cross-host `vllm serve`
command. Do not infer that the single-host launch command above works unchanged
across two machines.

One 5-second, 2-step request warmed both workers before the measured runs. The
reported pipeline elapsed time covers distributed model execution through GPU
completion; MP4 encoding happened afterward and is not included.

| Duration | Steps | Text encode | Denoise | Per step | VAE decode | Pipeline elapsed | Max peak per rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 s | 50 | 0.21 s | 1570.19 s | 31.40 s | 79.45 s | 1649.87 s (27 min 30 s) | 75.45 GiB |
| 10 s | 50 | 0.15 s | 4448.12 s | 88.96 s | 160.71 s | 4609.00 s (76 min 49 s) | 87.17 GiB |

Both ranks reported stage times and allocator peaks within measurement noise.
The generated outputs were H.264 at 1344x768 and 24 FPS with 32 kHz stereo AAC:
124 frames (5.167 s) and 243 frames (10.125 s). Both files decoded successfully,
and neither run encountered an OOM or NCCL error. These are single qualification
runs (`n=1`), not throughput guarantees.

## Measured latency

Single GB10, FP8, eager, cuDNN attention, tiled VAE, one request at a time:

| Workload | Steps | Text encode | Denoise | Per step | VAE decode | MP4 mux | Client E2E | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2VA, 4 s, smoke shape | 10 | — | — | 8.68 s | — | 0.49 s | 95.9 s | — |
| T2VA, 8 s, 960x576 | 50 | 0.25 s | 2088 s | 42.61 s | 70 s | 4.84 s | 2169.4 s (36 min 9 s) | 97.7 GiB |
| Ref2VA, 8 s, 960x576, one image (3 runs) | 10 | 8.43-10.94 s | 665.5-748.8 s | 66.55-74.88 s | 67.9-69.8 s | 3.63 s | 770.3-861.1 s | 102.8 GiB |
| Ref2VA, 8 s, 960x576, one image | 50 | 9.66 s | 4039.6 s | 80.79 s | 69.4 s | 3.63 s | 4157.0 s (69 min 17 s) | 102.8 GiB |

How these were measured:

- Stage times come from `--enable-diffusion-pipeline-profiler`, read from the
  `X-Stage-Durations` response header of `/v1/videos/sync`. The keys are
  `MiniMaxH3Pipeline.encode_prompt` (text encode),
  `MiniMaxH3Pipeline.diffuse` (denoise), and `MiniMaxH3Pipeline.decode`
  (VAE decode).
- MP4 mux is not a header field. It is the
  `Video response encoding (MP4 bytes)` line in the server log.
- Peak memory is the `X-Peak-Memory-MB` header, which reports the CUDA allocator
  high-water mark (`torch.cuda.max_memory_reserved`) for the request. It excludes
  the CUDA context and non-PyTorch allocations, so true pool usage is somewhat
  higher. `nvidia-smi --query-gpu=memory.used` is not usable on GB10 — unified
  memory has no discrete VRAM to report and the field returns `[N/A]`. See the
  capacity section for a `/proc/meminfo` cross-check.
- For T2VA, text encode and VAE decode were measured on a separate 10-step run at
  the same 960x576 / 8 s geometry. Both are independent of step count, so they
  transfer to the 50-step row unchanged. The Ref2VA MP4 mux figure transfers the
  same way, from the 10-step run at identical output geometry and duration.
- Ref2VA peak memory is byte-identical between the 10-step and 50-step runs
  (`X-Peak-Memory-MB: 105318.000` in both), confirming that peak is set by output
  geometry and reference input rather than step count.
- Stage times do not sum exactly to end-to-end. Queueing, result transfer, and
  engine bookkeeping sit outside the profiled stages.

Denoise dominates at 96% of end-to-end, so step count and geometry are the only
knobs that materially change the wall clock on this platform.

Ref2VA costs 1.92x the wall clock of T2VA at the same geometry and step count:
4157.0 s versus 2169.4 s. Text encoding shows the most dramatic ratio — 9.66 s
versus 0.25 s, 39x — but at 0.2% of end-to-end it is not what makes Ref2VA slow.
The cost is in denoise. A single reference image expands the Qwen3-VL
presentation to 6883 tokens (logged as `MiniMax H3 ref2va Qwen presentation:
6883 tokens, 1 reference images`), and those tokens sit in the attention context
of every denoise step. Per-step time rises from 42.61 s to 80.79 s, and that
1.90x multiplies across all 50 steps. VAE decode is unchanged at ~69 s, as
expected for an identical output shape.

Repeated identical 10-step T2VA requests varied by up to 23% in denoise time
(397 s to 490 s) on a machine that had already been under sustained load. Treat
these as single-run measurements on one machine, not a throughput guarantee, and
expect the slower end of that range for back-to-back jobs.

Ref2VA per-step time is load-dependent in the same way. On an idle machine the
first 10-step request ran at 66.55 s per step. The 50-step request that followed
occupied the box for 69 consecutive minutes and averaged 80.79 s per step. Two
further 10-step requests taken immediately afterwards settled at 74.88 s and
74.31 s — 12.5% slower than the idle run, and within 0.8% of each other, so this
is a stable thermal state rather than measurement noise.

Plan for 81 s per step when sizing a full 50-step Ref2VA job; 66 s per step is
only reachable on a cold machine and does not hold for the length of one request.

Ref2VA text encode converged downward across the four runs (10.94 s, 9.66 s,
8.52 s, 8.43 s) as first-touch costs fell away; treat ~8.5 s as steady state.
VAE decode was stable at 67.9-69.8 s throughout.

Set `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` well above the expected request time — the
default 1800 s is shorter than a 50-step run on this platform, and 7200 s is
shorter than a 50-step Ref2VA run.

## Known limitations

- With one GPU there is no Ulysses, ring, TP, or VAE patch parallelism. Pass
  `--usp 1 --ring 1 --vae-patch-parallel-size 1` explicitly if a profile or
  script would otherwise supply higher degrees.
- The two-host results require the external launch adaptation described above;
  cross-host diffusion worker launch is not exposed by the documented server CLI.
- `--enforce-eager` is used because regional compile adds startup time without a
  parallelism win on a single rank.
- `t2va` rejects requests without an explicit `aspect_ratio`. When `width` and
  `height` are both supplied they take precedence and `aspect_ratio` only has to
  name one of the supported ratios.
- Sustained load degrades throughput. See the variance notes above.
- Only one partition is resident. For Ref2VA, stop the FL2VA server and restart
  with `/path/to/MiniMax-H3/Ref2VA`. Ref2VA with one reference image is measured
  above; reference videos and additional images push the token count higher and
  were not measured. With `MemAvailable` bottoming out at 7.4 GiB during a
  single-image request, there is very little room to grow — re-measure before
  adding references or raising the output shape.
- Ref2VA requires a vLLM-Omni build newer than `v0.26.0`. See the version note
  under **Validated on**: FP8 weight loading and image-only Ref2VA are both
  broken on `release/v0.26.0`. The suggested version is `v0.26.1`.
- At the validated revision, online FP8 with layerwise offload produced a weight
  stride the Cutlass FP8 kernel rejected. Current H3 supports online FP8 with
  layerwise offload and both DLO transfer paths; see the linked FP8 section
  above. This does not change the GB10 capacity guidance: offload uses the same
  unified-memory pool and is not enabled in this recipe.


---

## FILE: recipes/MiniMaxAI/MiniMax-H3-Disaggregated.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/recipes/MiniMaxAI/MiniMax-H3-Disaggregated.md)

# MiniMax H3 disaggregated encoder stage

This opt-in topology runs the Qwen3-VL text encoder, video VAE encoder, and
audio WVAE encoder together in a vLLM stage. It sends their conditioning
payload over shared memory by default, or opt-in NIXL, to an encoder-free
diffusion stage. The standard MiniMax H3 recipes
remain single-stage and continue to run all components inside the diffusion
pipeline.

## Full encoder payload

Both the standard and Turbo builtin deployments use `SharedMemoryConnector`
for the same-host stage 0 → stage 1 edge. The corresponding `_nixl.yaml`
overlays opt into `NixlConnector` with the UCX backend. Stage 0 is the sender and stage 1
is the receiver. The producer hook `encoder2diffusion_full_payload` validates
the complete conditioning and packages it under `encoder_output`; the diffusion
stage declares `stage_input_payload_keys: [encoder_output]`.

This is not a text-only handoff. The payload includes:

- `hidden_states.output`: BF16 text embeddings.
- `meta.token_role_ids`: INT64 token roles aligned with the text embeddings.
- `embed.embedding`: FP32 visual conditioning latents when present.
- `embed.speech_feat`: FP32 audio conditioning latents when present.
- `kv_metadata.minimax_h3_encoder_layout`: packed layout metadata describing
  the task, dimensions, condition shapes/lengths, keyframes, and reference order.

Text-only requests still carry the full encoder schema, with empty optional
media tensors. The H3 layout field is model conditioning metadata, **not**
native paged-KV transfer. Stage 1 does not request native KV-cache reception.
All participating encoder ranks complete their component collectives before
the producer sends. The diffusion group leader receives the payload and the
generic stage-payload path distributes it to the participating diffusion ranks.
After a successful connector send, the orchestrator carries no duplicate inline
conditioning; original image/video/audio inputs and temporary encoder media
buffers are removed from the diffusion prompt. Stage 1 consumes the received
`encoder_output` without re-encoding text or media.

## Start the server

Choose the topology explicitly and load its deployment defaults:

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated.yaml
```

The default deployment assigns stage 0 to GPUs 0-1 with tensor parallel size
2 and `max_num_seqs: 1`. Stage 1 uses GPUs 2-5 with diffusion tensor parallel
size 1, Ulysses degree 4, and VAE patch parallel size 4. Adjust the
`devices`, `tensor_parallel_size`, and stage 1 `parallel_config` values in a
deployment override for the available hardware. Diffusion quantization,
layerwise offload, distributed layerwise offload, VAE parallelism, and USP
settings use the same stage 1 options documented in [MiniMax-H3.md](MiniMax-H3.md).

Stage 0's `tensor_parallel_size` defines one shared rank group for all three
encoder roles; it does not choose each role's execution strategy. The required
`hf_overrides.minimax_h3_encoder_components` mapping declares those strategies
explicitly: `text_encoder` uses tensor parallelism, `video_vae` uses patch
parallelism, and `audio_vae` runs on the group leader. All three role entries
are mandatory. They share the stage's `max_num_seqs` scheduler and batch limit;
the configuration does not create independent role schedulers or world sizes.

For a six-GPU TP2 encoder → TP4 diffusion topology, retain the builtin
deployment and override only stage 1's TP and Ulysses degrees. VAE patch
parallelism stays at 4. `--stage-overrides` keeps parallelism scoped to its
owning stage rather than broadcasting an override to both:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated.yaml \
  --stage-overrides '{"1":{"tensor_parallel_size":4,"ulysses_degree":1}}'
```

The same override is supported with the Turbo builtin deployment. Device IDs
in the deployment are relative to `CUDA_VISIBLE_DEVICES`; choose visible
devices and per-stage `devices` overrides for the available hardware. These
commands describe configuration, not a claim of hardware validation for this
integrated encoder implementation.

For memory-constrained deployments, start from the CPU-offload or distributed
layerwise-offload profiles in [MiniMax-H3.md](MiniMax-H3.md). Apply memory and
quantization options only to Stage 1 with `--stage-overrides`; retain the
encoder's BF16 configuration and the video/audio VAEs' FP32 precision. Select
one offload strategy per deployment:

```bash
# Stage 1 full-topology model offload, including MiniMax-H3's VAEs. Keep the
# compatibility alias because the compact selector covers only DiT/text encoder.
--stage-overrides '{"1":{"enable_cpu_offload":true}}'

# Stage 1 distributed layerwise offload. Tune resident layers for available RAM.
--stage-overrides '{"1":{"enable_distributed_layerwise_offload":true,"dlo_use_allgather":false,"dlo_resident_layers":20}}'

# Stage 1 online FP8 quantization of the DiT only.
--stage-overrides '{"1":{"diffusion_quantization_config":"{\"transformer\":{\"method\":\"fp8\"}}"}}'
```

The Stage 1 VAE patch-parallel options remain independent of offload and
quantization. See [MiniMax-H3.md](MiniMax-H3.md) for memory requirements and
hardware-qualified profiles before combining these options.

Stage 1 sets `model_loaded.text_encoder: false` and
`model_loaded.vae_encoder: false`. It loads only the VAE decoders and skips
tokenizer, processor, and text-encoder downloads. It requires complete text and
media conditioning from Stage 0, with no fallback to local encoding.

This H3 topology explicitly keeps its single-replica
diffusion stage inline, avoiding serialization of decoded video through a
subprocess. The builtin placement describes a single-host deployment. NIXL's
transport design can support remote endpoints, but **cross-node H3 execution
has not been validated** here; these examples are not a tested multi-node
launch procedure. CPU wiring and payload-contract tests do not establish native
NIXL, GPU, Turbo-weight, or cross-node execution success.

## Optional NIXL configuration

The base deployments above use `SharedMemoryConnector` and require same-host
shared-memory access. To use NIXL instead, select the standard overlay:

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated_nixl.yaml
```

For Turbo, use
`--deploy-config vllm_omni/deploy/minimax_h3_disaggregated_turbo_nixl.yaml`
with the LoRA options shown below. Both overlays require NIXL and a working
UCX backend; see [NIXL installation](../../docs/design/feature/omni_connectors/nixl_connector.md#installation).
Initialization failures do not automatically fall back to shared memory.

Each overlay inherits its base deployment and switches **both** edge
references to NIXL. The encoder roles, `model_loaded` flags, hooks, and full
`encoder_output` schema stay unchanged. The processor also retains support for
an inline full-conditioning handoff when supplied; stage 1 still does not fall
back to local encoders. SharedMemory requires same-host shared-memory access
and is not a cross-node transport.

The `/v1/videos` request schema and `extra_params.task` values (`t2va`,
`fl2va`, and `ref2va`) are unchanged from the single-stage recipe.

## Turbo LoRA

The deploy config below carries the four-step 768p contract -- five sigma
points, `flow_shift=6`, `audio_flow_shift=3` -- so requests that omit sampling
controls do not inherit the 50-step base schedule:

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --lora-path /path/to/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated_turbo.yaml
```

`--lora-path` must name one artifact, or a directory holding exactly one; the
Turbo repository holds several and a directory of them is rejected as
ambiguous. Serving a different artifact means overriding those defaults with
that artifact's own contract -- see the
[Turbo LoRA table](MiniMax-H3.md#turbo-lora) for every published row.

The base deployment intentionally retains 50 inference steps for non-LoRA
quality. The `fl2v` artifacts serve T2VA and FL2VA; the `ref2v` artifacts need
a `--task-type ref2va` server. Standard and Turbo configuration coverage does
not imply that Turbo weights have been exercised on this integrated topology.
