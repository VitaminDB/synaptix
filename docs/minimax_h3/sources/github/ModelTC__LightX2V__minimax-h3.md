# ModelTC__LightX2V__minimax-h3

Source: https://github.com/ModelTC/LightX2V/tree/main/scripts/minimax_h3
Fetched: 2026-11-28



---

## FILE: scripts/minimax_h3/README.md (https://github.com/ModelTC/LightX2V/blob/HEAD/scripts/minimax_h3/README.md)

# MiniMax-H3

[English](README.md) | [简体中文](README_zh.md)

Before running, set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in the selected script.

## Weights and tasks

Download Diffusers-format weights from the official Hugging Face repository [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and set `model_path` to the local model directory.

| Task | Request inputs | Transformer |
| --- | --- | --- |
| `t2av` | `prompt` | `transformer` |
| `i2av` | `prompt`, `image_path` (first frame) | `transformer` |
| `l2av` | `prompt`, `last_frame_path` | `transformer` |
| `fl2av` | `prompt`, `image_path`, `last_frame_path` | `transformer` |
| `ref2av` | `prompt`, reference images and/or videos, optional reference audio | `transformer_ref` |

Set `--model-variant` at startup: `fl2av` supports the first four tasks in the table; `ref2av` supports reference generation. Select the request's `task`; switching base tasks requires no reload. Both variants can share an inference JSON, but LoRA and quantized weights must match the variant.

## AdaLN cache

The AdaLN cache stores reusable computation results to avoid repeating work during inference. **Example configs already enable caching. Generate the cache before your first inference run or server startup; skip this step if a matching cache exists.**

1. **Check the cache settings.** Keep these fields in your inference JSON (already included in the example configs):

   ```json
   {
     "use_adaln_cache": true,
     "adaln_cache_dir": "~/.cache/lightx2v/adaln"
   }
   ```

2. **Edit the [cache generation script](../../tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh).** Set the following parameters:

   | Parameter | Setting |
   | --- | --- |
   | `lightx2v_path`, `model_path` | Use the same repository and model paths as inference. |
   | `CUDA_VISIBLE_DEVICES` | Select a GPU, e.g. `0`; cache generation only needs one GPU. |
   | `--config_json` | Use the inference JSON. |
   | `--model-variant` | Use `fl2av` for `t2av`, `i2av`, `l2av`, and `fl2av`; use `ref2av` for reference generation. |

3. **Generate the cache.** Run from the repository root:

   ```bash
   bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
   ```

   The log message `AdaLN cache saved to` confirms completion. You can then run offline inference or start a server as described below. Files are stored in a subdirectory of `adaln_cache_dir` and loaded automatically during inference; no cache file path is needed.

> **Cache reuse**: regenerate when changing the base variant/checkpoint, timestep grid, or any LoRA affecting AdaLN/final norm (including strength, alpha, mode or `LORA_DTYPE`). Export and inference must share these settings. LoRAs without modulation updates can share the base cache.

## Offline inference

The five task scripts share `configs/minimax_h3/minimax_h3.json`: one GPU, BF16 weights, and 124 frames at `[height, width] = [544, 960]`.

```bash
bash scripts/minimax_h3/run_minimax_h3_t2av.sh
bash scripts/minimax_h3/run_minimax_h3_i2av.sh
bash scripts/minimax_h3/run_minimax_h3_l2av.sh
bash scripts/minimax_h3/run_minimax_h3_fl2av.sh
bash scripts/minimax_h3/run_minimax_h3_ref2av.sh
```

The ordinary config uses SageAttention2 and SGL kernels. Select a config whose attention, quantization, and parallel settings match your installed kernels and devices.

Select a config through `--config_json`; the table lists representative configs under `configs/minimax_h3/`. For one GPU, use the matching task script; for SP, TP, or mixed parallelism, use `run_minimax_h3_t2av_parallel.sh` as a reference and adjust the variant, task, and inputs for other tasks.

| Configuration under `configs/minimax_h3/` | Mode |
| --- | --- |
| `minimax_h3.json` | Single-GPU BF16 |
| `minimax_h3_compile.json` | Compile with startup warmup |
| `minimax_h3_block_offload.json` | Single-GPU BF16 block offload |
| `minimax_h3_sp.json` | SP4 |
| `minimax_h3_tp.json` | TP2 |
| `minimax_h3_tp_sp.json` | TP2 × SP2 |
| `fp8/minimax_h3.json` | Single-GPU DiT FP8 |
| `distill/4steps/minimax_h3_bf16_4step.json` | Single-GPU BF16, 4-step LoRA |
| `distill/4steps/minimax_h3_fp8_4step.json` | SP4, FP8 + 4-step LoRA |
| `distill/4steps/minimax_h3_ref2av_4step.json` | SP8, reference-task 4-step LoRA |

The multi-GPU script defaults to SP4. When switching configs, adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` to match the configured parallel size.

Use `minimax_h3_compile.json` for compilation, enabling `use_compile` and `warmup` and setting `rope_type: "torch_real_rope"`. Service launchers can use it too.

Add `video_codec_options` to the inference JSON to configure MP4 encoding, for example:

```json
{
  "video_codec_options": {
    "preset": "ultrafast",
    "crf": "18"
  }
}
```

## Python API

The [four-step DMD T2AV example](inference/distill/minimax_h3_t2av_dmd.py) uses
`configs/minimax_h3/distill/4steps/minimax_h3_bf16_4step.json`. After installation, set
`MODEL_PATH` and `OUTPUT_PATH` in the script and the local LoRA path in that JSON,
then run from the repository root:

```bash
python scripts/minimax_h3/inference/distill/minimax_h3_t2av_dmd.py
```

See the [Python API guide](../../docs/EN/source/getting_started/python_api.md)
for generator setup and configuration precedence.

## Low-cost candidate generation

For low-resolution drafts, candidate selection, and higher-resolution regeneration, see the [MiniMax-H3 Roll documentation](../minimax_h3_roll/README.md).

## LoRA

Set `lora_mode` in the inference JSON (`merge` by default). All `distill/` configs select `consumer`:

```json
{
  "lora_mode": "consumer"
}
```

Precision remains an environment setting for **both cache export and inference**:

```bash
export LORA_DTYPE=BF16  # BF16, FP16, FP32; unset follows DTYPE
```

| DiT weights | `lora_mode` | Behavior |
| --- | --- | --- |
| Original BF16 | `merge` (default), `separate`, `consumer` | Merge at load time, compute independent branches, or fuse branch addition into H3's activation/residual consumers. |
| Quantized FP8 or INT8, including ConvRot | `separate`, `consumer` | Retain the existing quantized GEMM and a floating-point LoRA branch; merging into quantized DiT weights is unsupported. |

`consumer` is implemented only for `minimax_h3` on CUDA. It supports eager and
`use_compile: true`, but not tensor parallelism or GEMM calibration; sequence parallelism
and model/block offload remain available. Unsupported combinations raise an
error rather than switching modes. The fused kernels retain intermediate dtype
rounding, including BF16 projections with FP32 gates/residuals. Numerical parity
and speed must be checked on the chosen backend; no universal speedup is implied.
Compilation can change rounding elsewhere in the block, so eager/compiled outputs
are not guaranteed to be bitwise equal. Quantized compilation also requires the
selected GEMM/attention backend to support tracing; consumer does not replace it.

`separate` and `consumer` retain request-time adapter switching. This is distinct
from the execution mode: both leave base weights unchanged.

Set `export LORA_DTYPE=FP32` (or `BF16` / `FP16`) before launching. Unset means
follow `DTYPE`, not `SENSITIVE_LAYER_DTYPE`. This controls LoRA factors and
computation in all three modes. Merge restores the stored base weight dtype;
separate/consumer restore the base projection output dtype **after** adding the
LoRA result. Base precision and quantized GEMMs are unchanged. FP32 merge still
has final weight rounding and is not equivalent to separate/consumer.
The old JSON `lora_merge_force_fp32` is removed; no JSON `lora_dtype` is needed.

Base-task 4-step v1.0 LoRA example: add or update these fields in `minimax_h3.json`, replacing `path` with your local LoRA file:

```json
{
  "lora_mode": "consumer",
  "infer_steps": 4,
  "video_flow_shift": 6.0,
  "audio_flow_shift": 3.0,
  "scheduler_type": "euler",
  "lora_configs": [
    {
      "path": "/path/to/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
      "strength": 1.0,
      "alpha": 128
    }
  ]
}
```

Native (non-ComfyUI) H3 Turbo LoRAs use per-layer alpha tensors first, file
metadata alpha second, and `lora_configs[].alpha` only if neither is embedded.
The loader logs each alpha's source, value and layer count. Rank comes from the
factor shapes; AdaLN and `norm_out.linear` need no special configuration.
All BF16, FP8 and INT8 ConvRot configs under `configs/minimax_h3/distill/8steps/`
use `minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors`: eight Euler steps,
shifts 12/3 and strength=1. Replace the LoRA path with your local file. Alpha is
read per layer from the checkpoint, without a global JSON override.
These configs set `lora_mode: "consumer"`; `separate` is also supported. Quantized bases do not support merge.
The adapter remains BF16; FP8/INT8 refers only to the base weights. Before the
first run, export the matching AdaLN cache as below, using the same config and
`lora_mode` config and `LORA_DTYPE` environment for export and inference.
The earliest FL2V v0.1 has no embedded alpha and requires an explicit configuration.
The repository's FP8 full transformer checkpoint is not a LoRA.

### Scheduler and LoRA-aware AdaLN cache

`scheduler_type` selects `euler`, `lcm`, or the base model's default
`reference_blend`. No legacy field aliases are accepted. LCM uses a clean prediction
followed by re-noising, matching Pika H3's sampler; it is not Euler with fewer steps.
For a matching 6-step LCM adapter, set:

```json
{
  "scheduler_type": "lcm",
  "infer_steps": 6,
  "distilled_sigma_values": [1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0],
  "video_flow_shift": 12.0,
  "audio_flow_shift": 3.0
}
```

These are **unshifted** sigma knots, including both endpoints. There must be
`infer_steps + 1` values strictly descending from 1 to 0. Each modality applies
its own shift once. Do not change an Euler-trained adapter to LCM arbitrarily.

Generate the cache with the **same inference JSON (including `lora_mode`) and `LORA_DTYPE` environment**, including LoRA and scheduler:

```bash
PYTHONPATH=. python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path /path/to/MiniMax-H3 --model-variant fl2av \
  --config_json configs/minimax_h3/distill/8steps/minimax_h3_bf16_8step.json
```

Use `--model-variant ref2av` for the reference transformer. Cache identity includes
the base checkpoint, timestep grid, affected LoRA projections, strength, and
merge arithmetic. `separate`/`consumer` share modulation arithmetic and caches;
`merge` has its own. Existing H3 caches need regeneration once. Cached modulation
LoRAs are not applied a second time online. Before switching/removing a LoRA,
export its corresponding cache; a missing cache fails before changing the active
adapter. LoRAs without modulation updates can share the base cache.

### Pika-H3-Flex six-step LoRAs

The [official implementation](https://github.com/Pika-Labs/Pika-H3-Flex) releases separate
FL2VA and Ref2VA adapters: rank/alpha=128/128, strength=1, six-step LCM with shifts 12/3.
The files have no embedded alpha, so these configs explicitly set `alpha: 128`.
No checkpoint conversion is needed.

- [FL2AV config](../../configs/minimax_h3/distill/6steps/minimax_h3_pika_fl2av_6step.json): `transformer/`, for T2AV/I2AV/L2AV/FL2AV.
- [Ref2AV config](../../configs/minimax_h3/distill/6steps/minimax_h3_pika_ref2av_6step.json): `transformer_ref/`; use the matching adapter.

Download the adapters and update `lora_configs[0].path` in each config. Use the MiniMax-H3 base:

```bash
hf download Pika-Labs/Pika-H3-Flex fl2va/adapter.safetensors ref2va/adapter.safetensors --local-dir /path/to/pika-h3-flex

export DTYPE=BF16
export LORA_DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
export PYTHONPATH=.
model_path=/path/to/MiniMax-H3
config_path=configs/minimax_h3/distill/6steps/minimax_h3_pika_fl2av_6step.json

python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$model_path" --model-variant fl2av --config_json "$config_path"
python -m lightx2v.infer \
  --model_cls minimax_h3 --model-variant fl2av --task t2av \
  --model_path "$model_path" --config_json "$config_path" \
  --prompt 'integrated_multimodal_description: A fox walks through a snowy forest. overall_soundscape: Wind and crunching snow. non_diegetic_music: N/A' \
  --seed 42 --save_result_path output_pika_t2av.mp4
```

Change `--task` and add `--image_path` for I2AV, `--last_frame_path` for L2AV,
or both for FL2AV. For Ref2AV, use its own config and cache:

```bash
config_path=configs/minimax_h3/distill/6steps/minimax_h3_pika_ref2av_6step.json
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$model_path" --model-variant ref2av --config_json "$config_path"
python -m lightx2v.infer \
  --model_cls minimax_h3 --model-variant ref2av --task ref2av \
  --model_path "$model_path" --config_json "$config_path" \
  --image_path /path/to/reference.png \
  --prompt 'Animate the subject in <Picture 1>, preserving its appearance and setting. Natural motion and ambient sound, no music.' \
  --seed 42 --save_result_path output_pika_ref2av.mp4
```

Examples use 768×1344, 124 frames, single-GPU block offload and unquantized weights.
Set `LORA_DTYPE=BF16` to match the official adapter compute dtype. `lora_mode: "separate"`
is also supported; merging introduces different weight rounding.
The Ref config uses `reference_image_resize_mode: "match"`: cap reference image
area to the output canvas while preserving aspect ratio, without upscaling.
This is our selected preprocessing policy, not the official 768p tier's reference
short edge of 2048. Attention/kernel differences may also change numerical results.

## Server and POST

The base service starts with `--model-variant fl2av` and no startup `--task`. Select a launcher by GPU count, then submit requests from another terminal after the model loads.

Single GPU:

```bash
bash scripts/minimax_h3/server/start_server.sh
```

Four-GPU SP4 uses `minimax_h3_sp.json`:

```bash
bash scripts/minimax_h3/server/start_server_parallel.sh
```

Both services use the following POST clients:

```bash
python scripts/minimax_h3/server/post_t2av.py
python scripts/minimax_h3/server/post_i2av.py
python scripts/minimax_h3/server/post_l2av.py
python scripts/minimax_h3/server/post_fl2av.py
```

Each base request must specify `task`; see the task table for inputs. The image examples encode client-local files as Base64; server-accessible absolute paths also work.

The reference service uses `--model-variant ref2av`:

```bash
bash scripts/minimax_h3/server/start_server_ref2av.sh
```

Set the reference image path in the client, then send a request from another terminal:

```bash
python scripts/minimax_h3/server/post_ref2av.py
```

## Request and startup settings

- **Startup and requests**: set model paths, variant, weights/LoRA, kernels, offload, parallelism, compile, and warmup at startup. Pass `prompt`, media paths, `seed`, and the output path through Python, CLI, or POST.
- **Output settings**: The JSON `request_defaults` object provides defaults for `num_frames` and `size`; requests can override them. `size` is `[height, width]`, with both dimensions divisible by 32. Frame counts round up to `17*n+5` (e.g. 125 → 141). The aligned count must be 22–362 frames. Output is MP4 with 24 FPS video and 32 kHz stereo audio.
- **Random seed**: an omitted or `null` `seed` uses 42. You can specify any non-negative integer, including 0.
- **Saving results**: CLI skips saving when no output path is supplied. For service requests, prefer a relative `save_result_path` such as `./minimax_h3_t2av.mp4`, resolved under the service output directory. Omitted or `null` paths skip saving; saved results can be downloaded.


---

## FILE: lightx2v_train/docs/minimax_h3_distill.md (https://github.com/ModelTC/LightX2V/blob/HEAD/lightx2v_train/docs/minimax_h3_distill.md)

# MiniMax-H3 dCM and PDMD training

Use a converted Diffusers H3 model root with `transformer/` for FL2AV and `transformer_ref/` for Ref2AV. The training environment must provide MiniMax-H3 in Diffusers and the `_flash_3_hub` attention backend.

| Task | 4-step PDMD | 8-step PDMD | dCM video/audio shift | PDMD video/audio shift |
| --- | --- | --- | --- | --- |
| FL2AV | [Config](../configs/train/pdmd/minimax_h3_fl2av_pdmd4_gan_fsdp32.yaml) | [Config](../configs/train/pdmd/minimax_h3_fl2av_pdmd8_gan_fsdp32.yaml) | 6/3 | 6/3 for updates 1–1,500, 12/3 for updates 1,501–3,000 |
| Ref2AV | [Config](../configs/train/pdmd/minimax_h3_ref2av_pdmd4_gan_fsdp32.yaml) | [Config](../configs/train/pdmd/minimax_h3_ref2av_pdmd8_gan_fsdp32.yaml) | 12/3 | 12/3 |

Both tasks train dCM for 1,500 student updates and use the final student EMA to initialize PDMD. PDMD runs five fake updates followed by one student update. Each role update counts as one iteration, so 3,000 iterations contain 2,500 fake and 500 student updates. Checkpoints are saved every 500 dCM updates and 100 PDMD updates, keeping two per stage.

PDMD uses student/fake learning rates of `5e-5`/`4e-6`, AdamW betas `(0, 0.9)`, and zero weight decay. The GAN head uses the fake optimizer settings. Generator and discriminator loss weights are `0.005` and `0.01`. The student uses rank/alpha 128/128, with AdaLN and the output normalization projection at 16/16.

## Data

Use `data.train.name: cache_dataset` and point `data.train.data_path` to a `meta.json` file. The supplied training configs read the path from `H3_TRAIN_META`. The file contains a JSON array of cached samples.

### FL2AV

Set `model.name: minimax_h3_fl2av` for `t2av`, `i2av`, or `fl2av`. Each record carries one task. The condition cache contains the prompt embeddings and, for I2AV or FL2AV, the encoded first frame or first/last frames.

```json
[
  {
    "task": "t2av",
    "prompt": "A boat crosses the lake.",
    "training_cache": "conditions/boat.pt",
    "num_frames": 124,
    "target_height": 768,
    "target_width": 1344,
    "video_latent_path": "video_latents/boat.pt",
    "audio_latent_path": "audio_latents/boat.pt"
  },
  {
    "task": "i2av",
    "prompt": "The camera follows the boat.",
    "training_cache": "conditions/follow_boat.pt",
    "num_frames": 124,
    "target_height": 768,
    "target_width": 1344
  },
  {
    "task": "fl2av",
    "prompt": "The boat pulls into the dock.",
    "training_cache": "conditions/boat_at_dock.pt",
    "num_frames": 124,
    "target_height": 768,
    "target_width": 1344
  }
]
```

Each global batch covers every active task across the data-parallel ranks. Each rank loads one sample. With three tasks on 32 ranks, the sampler assigns 10, 11, and 11 samples, rotating the extra slots across batches. Smaller task pools are reused as they run out. The default uses all FL2AV-compatible tasks in the file. Use `H3_TASKS='[t2av]'` to select one task from a mixed file.

### Ref2AV

Set `model.name: minimax_h3_ref2av`. `ref_image_count` gives the number of encoded reference images in the cache. This example uses two images.

```json
{
  "task": "ref2av",
  "prompt": "The two referenced people walk through a garden.",
  "training_cache": "conditions/garden.pt",
  "ref_image_count": 2,
  "num_frames": 124,
  "target_height": 1344,
  "target_width": 768,
  "video_latent_path": "video_latents/garden.pt",
  "audio_latent_path": "audio_latents/garden.pt"
}
```

Use `H3_REF_IMAGE_COUNTS='[2]'` for a two-image run or the default `[1,2,3,4,5,6]` for mixed counts. The sampler uses `ref_image_count` to spread each count across global batches while preserving its frequency in the data as closely as full batches allow. The shipped Ref configs drop the final incomplete batch.

### Cache paths and training methods

Paths can be absolute or relative to `meta.json`. For existing H3 condition caches, set `training_cache` to the old `condition_path`. Use the prompt that was encoded for that cache. The target geometry describes the rollout. Visual conditions must match it. The H3 cache uses 24 fps.

- **dCM** selects samples with paired target latents. Provide both `video_latent_path` and `audio_latent_path` for external targets. The video latents use H3 VAE normalization. For a full training cache that already contains paired `inputs.video_latents` and `inputs.audio_latents`, mark the record with `"has_targets": true`.
- **DMD and PDMD** can also sample condition-only records.
- **GAN** is enabled with `training.dmd.gan.enabled: true` in either algorithm. Each rollout geometry needs a paired target for the discriminator. The positive sample carries its own conditioning.

## Launch with torchrun

These configurations use 32 GPUs. The example uses four nodes with eight GPUs each. On every node, activate your training environment and run the following from the repository root. Set `MASTER_ADDR` to the hostname or IP address of node 0, reachable from every node, and set `NODE_RANK` to `0`, `1`, `2`, or `3` on the corresponding node. Use shared paths for the data and output directories.

```bash
cd lightx2v_train
export PLATFORM=nvidia
export PYTHONPATH="$PWD/..${PYTHONPATH:+:$PYTHONPATH}"

export H3_MODEL_PATH=/path/to/MiniMax-H3
export H3_TRAIN_META=/path/to/data/meta.json
export H3_DCM_OUTPUT=/path/to/output/dcm

MASTER_ADDR=node0.example.com
NODE_RANK=0
TASK=ref2av          # Use fl2av for T2AV, I2AV, or FL2AV training.
PDMD_STEPS=8         # Use 4 for four-step distillation.
export H3_PDMD_OUTPUT="/path/to/output/pdmd${PDMD_STEPS}"
```

### dCM

```bash
torchrun --nnodes=4 --nproc-per-node=8 \
    --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
    train.py --config "configs/train/consistency/minimax_h3_${TASK}_dcm_fsdp32.yaml"
```

Set `H3_DCM_TEACHER_PATH` to use a separate teacher model directory.

### PDMD with GAN

After dCM finishes, point `H3_STUDENT_INIT` to its final EMA and start PDMD on every node.

```bash
printf -v dcm_step '%09d' "${H3_DCM_TRAIN_ITERS:-1500}"
export H3_STUDENT_INIT="$H3_DCM_OUTPUT/checkpoint-$dcm_step/student_ema"

torchrun --nnodes=4 --nproc-per-node=8 \
    --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29501 \
    train.py --config "configs/train/pdmd/minimax_h3_${TASK}_pdmd${PDMD_STEPS}_gan_fsdp32.yaml"
```

Each stage writes `train.log` and checkpoints to its output directory.

The [FL2AV DMD config](../configs/train/dmd/minimax_h3_fl2av_dmd_lora.yaml) uses the same `cache_dataset` loader. Set `training.dmd.gan.enabled` to `true` and specify the fake-model block in `training.dmd.gan.feature_block`. GAN training requires paired targets.


---

## FILE: scripts/minimax_h3/inference/offload_layout/README.md (https://github.com/ModelTC/LightX2V/blob/HEAD/scripts/minimax_h3/inference/offload_layout/README.md)

# MiniMax-H3 CPU Offload

English | [简体中文](README_CN.md)

Run MiniMax-H3 text-to-audio/video on one device to compare DiT CPU block offload layouts: `baseline` copies weights tensor by tensor; `contiguous` stores each block in contiguous pinned memory and copies it as a whole.

Prepare the full BF16 Diffusers components described in the [model instructions](../../README.md). Set `lightx2v_path` and `model_path` in both scripts. Generate the matching [AdaLN cache](../../README.md#adaln-cache) first; baseline and contiguous share a cache for the same weights and schedule.

## NVIDIA GPU

Set `CUDA_VISIBLE_DEVICES` in both scripts. In the cache-generation script, select `PLATFORM=cuda`, `--model-variant fl2av`, and `configs/minimax_h3/layout/baseline.json`.

```bash
bash scripts/minimax_h3/inference/offload_layout/run_baseline.sh
bash scripts/minimax_h3/inference/offload_layout/run_contiguous.sh
```

## Ascend NPU

Set `ASCEND_RT_VISIBLE_DEVICES` in both NPU scripts. In the cache-generation script, select `PLATFORM=ascend_npu`, `--model-variant fl2av`, and `configs/minimax_h3/layout/baseline_npu.json`.

```bash
bash scripts/minimax_h3/inference/offload_layout/run_baseline_npu.sh
bash scripts/minimax_h3/inference/offload_layout/run_contiguous_npu.sh
```

Configs are in `configs/minimax_h3/layout/`. Set the prompt through `--prompt` in each script. Outputs are saved to `save_results/minimax_h3_layout/`; rerunning a script overwrites its output.

`cpu_offload_block_contiguous` is `false` in baseline configs and `true` in contiguous configs.


---

## FILE: configs/minimax_h3/minimax_h3.json (https://github.com/ModelTC/LightX2V/blob/HEAD/configs/minimax_h3/minimax_h3.json)

```
{
  "infer_steps": 29,
  "fps": 24,
  "enable_cfg": false,
  "cpu_offload": true,
  "offload_granularity": "model",
  "use_adaln_cache": true,
  "adaln_cache_dir": "~/.cache/lightx2v/adaln",
  "text_encoder_cpu_offload": true,
  "vae_cpu_offload": true,
  "attn_type": "sage_attn2",
  "rms_type": "sgl-kernel",
  "rope_type": "minimax_h3_triton_rope",
  "feature_caching": "NoCaching",
  "use_compile": false,
  "video_flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "vae_spatial_scale_factor": 16,
  "request_defaults": {
    "num_frames": 124,
    "size": [544, 960]
  }
}

```


---

## FILE: configs/minimax_h3/distill/4steps/minimax_h3_bf16_4step.json (https://github.com/ModelTC/LightX2V/blob/HEAD/configs/minimax_h3/distill/4steps/minimax_h3_bf16_4step.json)

```
{
  "lora_mode": "consumer",
  "infer_steps": 4,
  "fps": 24,
  "enable_cfg": false,
  "cpu_offload": true,
  "offload_granularity": "block",
  "use_adaln_cache": true,
  "adaln_cache_dir": "~/.cache/lightx2v/adaln",
  "text_encoder_cpu_offload": true,
  "text_encoder_offload_granularity": "block",
  "vae_cpu_offload": true,
  "attn_type": "flash_attn3",
  "rms_type": "torch_native",
  "rope_type": "torch_real_rope",
  "feature_caching": "NoCaching",
  "use_compile": false,
  "video_flow_shift": 6.0,
  "audio_flow_shift": 3.0,
  "scheduler_type": "euler",
  "vae_spatial_scale_factor": 16,
  "lora_configs": [
    {
      "path": "/path/to/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
      "strength": 1.0,
      "alpha": 128
    }
  ],
  "request_defaults": {
    "num_frames": 362,
    "size": [768, 1344]
  }
}

```


---

## FILE: configs/minimax_h3/distill/4steps/minimax_h3_ref2av_4step.json (https://github.com/ModelTC/LightX2V/blob/HEAD/configs/minimax_h3/distill/4steps/minimax_h3_ref2av_4step.json)

```
{
  "lora_mode": "consumer",
  "infer_steps": 4,
  "fps": 24,
  "enable_cfg": false,
  "cpu_offload": true,
  "reference_image_resize_mode": "match",
  "offload_granularity": "model",
  "use_adaln_cache": true,
  "adaln_cache_dir": "~/.cache/lightx2v/adaln",
  "text_encoder_cpu_offload": true,
  "vae_cpu_offload": true,
  "attn_type": "sage_attn2",
  "rms_type": "sgl-kernel",
  "rope_type": "minimax_h3_triton_rope",
  "feature_caching": "NoCaching",
  "use_compile": false,
  "video_flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "scheduler_type": "euler",
  "vae_spatial_scale_factor": 16,
  "text_encoder_tensor_parallel": true,
  "lora_configs": [
    {
      "path": "/path/to/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
      "strength": 1.0,
      "alpha": 8
    }
  ],
  "parallel": {
    "seq_p_size": 8,
    "seq_p_attn_type": "ulysses"
  },
  "request_defaults": {
    "num_frames": 362,
    "size": [768, 1344]
  }
}

```


---

## FILE: configs/minimax_h3/distill/8steps/minimax_h3_bf16_8step.json (https://github.com/ModelTC/LightX2V/blob/HEAD/configs/minimax_h3/distill/8steps/minimax_h3_bf16_8step.json)

```
{
  "lora_mode": "consumer",
  "infer_steps": 8,
  "fps": 24,
  "enable_cfg": false,
  "cpu_offload": true,
  "offload_granularity": "block",
  "use_adaln_cache": true,
  "adaln_cache_dir": "~/.cache/lightx2v/adaln",
  "text_encoder_cpu_offload": true,
  "text_encoder_offload_granularity": "block",
  "vae_cpu_offload": true,
  "attn_type": "flash_attn3",
  "rms_type": "torch_native",
  "rope_type": "torch_real_rope",
  "feature_caching": "NoCaching",
  "use_compile": true,
  "video_flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "scheduler_type": "euler",
  "vae_spatial_scale_factor": 16,
  "lora_configs": [
    {
      "path": "/path/to/minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors",
      "strength": 1.0
    }
  ],
  "request_defaults": {
    "num_frames": 362,
    "size": [768, 1344]
  }
}

```


---

## FILE: configs/minimax_h3/distill/4steps/minimax_h3_fp8_4step_5090.json (https://github.com/ModelTC/LightX2V/blob/HEAD/configs/minimax_h3/distill/4steps/minimax_h3_fp8_4step_5090.json)

```
{
  "lora_mode": "consumer",
  "infer_steps": 4,
  "fps": 24,
  "enable_cfg": false,
  "cpu_offload": true,
  "offload_granularity": "block",
  "use_adaln_cache": true,
  "adaln_cache_dir": "~/.cache/lightx2v/adaln",
  "text_encoder_cpu_offload": true,
  "text_encoder_offload_granularity": "block",
  "vae_cpu_offload": true,
  "vae_decode_parallel": true,
  "attn_type": "sage_attn2",
  "rms_type": "sgl-kernel",
  "rope_type": "minimax_h3_triton_rope",
  "feature_caching": "NoCaching",
  "use_compile": true,
  "warmup": true,
  "video_flow_shift": 6.0,
  "audio_flow_shift": 3.0,
  "scheduler_type": "euler",
  "vae_spatial_scale_factor": 16,
  "dit_quantized": true,
  "dit_quant_scheme": "fp8-sgl",
  "dit_quantized_ckpt": "/path/to/minimax_h3_fp8.safetensors",
  "lora_configs": [
    {
      "path": "/path/to/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
      "strength": 1.0,
      "alpha": 128
    }
  ],
  "parallel": {
    "seq_p_size": 8,
    "seq_p_attn_type": "ulysses",
    "seq_p_prepost_backend": "triton",
    "seq_p_tensor_fusion": true,
    "seq_p_head_parallel": true,
    "seq_p_quant_scheme": "fp8"
  },
  "request_defaults": {
    "num_frames": 362,
    "size": [768, 1344]
  }
}

```


---

## FILE: configs/minimax_h3/minimax_h3_block_offload.json (https://github.com/ModelTC/LightX2V/blob/HEAD/configs/minimax_h3/minimax_h3_block_offload.json)

```
{
  "infer_steps": 29,
  "fps": 24,
  "enable_cfg": false,
  "cpu_offload": true,
  "offload_granularity": "block",
  "use_adaln_cache": true,
  "adaln_cache_dir": "~/.cache/lightx2v/adaln",
  "text_encoder_cpu_offload": true,
  "text_encoder_offload_granularity": "block",
  "vae_cpu_offload": true,
  "attn_type": "sage_attn2",
  "rms_type": "sgl-kernel",
  "rope_type": "minimax_h3_triton_rope",
  "feature_caching": "NoCaching",
  "use_compile": false,
  "video_flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "vae_spatial_scale_factor": 16,
  "request_defaults": {
    "num_frames": 124,
    "size": [768, 1344]
  }
}

```


---

## FILE: configs/minimax_h3_causal/h100/refa2av_realtime_768p_sp8.json (https://github.com/ModelTC/LightX2V/blob/HEAD/configs/minimax_h3_causal/h100/refa2av_realtime_768p_sp8.json)

```
{
  "infer_steps": 4,
  "fps": 24,
  "enable_cfg": false,
  "use_adaln_cache": true,
  "adaln_cache_dir": "~/.cache/lightx2v/adaln",
  "use_compile": true,
  "vae_use_compile": true,
  "vae_attn_type": "flash_attn3",
  "text_encoder_tensor_parallel": true,
  "vae_encoder_cpu_offload": true,
  "vae_stitch_preallocate": true,
  "vae_decode_parallel": true,
  "vae_decode_overlap": true,
  "audio_vae_decode_overlap": true,
  "vae_rgb_pinned_memory": true,
  "video_streaming_encode": true,
  "reference_image_resize_mode": "crop",
  "reference_condition_cache": true,
  "attn_type": "flash_attn3",
  "rms_type": "torch_native",
  "rope_type": "torch_real_rope",
  "rope_compute_dtype": "bf16",
  "feature_caching": "NoCaching",
  "video_flow_shift": 5.0,
  "audio_flow_shift": 3.0,
  "scheduler_type": "renoise",
  "dit_original_ckpt": "/data/zhangbilang/models/MiniMaxAI/minimax_h3_causal/SekoTalk-H3-v0.1.0.pt",
  "ar_config": {
    "latest_chunk_num": 2,
    "audio_latest_chunk_num": 9,
    "kv_cache_max_t_rope": 9,
    "sink_text_condition": true,
    "sink_ref_condition": true,
    "motion_prompt_origin": 20480,
    "media_origin": 30000
  },
  "parallel": {
    "seq_p_size": 8,
    "seq_p_attn_type": "ulysses"
  },
  "use_fused_qkv": true,
  "request_defaults": {
    "num_frames": 957,
    "size": [1344, 768]
  }
}

```


---

## FILE: lightx2v_train/configs/train/flow/minimax_h3_fl2av_lora.yaml (https://github.com/ModelTC/LightX2V/blob/HEAD/lightx2v_train/configs/train/flow/minimax_h3_fl2av_lora.yaml)

```
model:
    name: minimax_h3_fl2av
    # Converted Diffusers model root containing transformer/config.json.
    pretrained_model_name_or_path: /path/to/models/MiniMax-H3
    running_dtype: bf16
    # H3 diffusion states are kept in fp32; text embeddings remain bf16.
    latent_dtype: fp32
    local_files_only: true
    patch_size: [1, 2, 2]
    video_latent_channels: 24
    audio_latent_channels: 32
    vae_spatial_scale_factor: 16
    video_fps: 24
    audio_sampling_rate: 32000
    audio_latents_per_second: 40
    # Cache construction activates the large frozen encoders one at a time.
    cache_encoder_cpu_offload: true
    attention_backend: _flash_3_hub
    capabilities:
        flow_matching:
            video_flow_shift: 6.0
            audio_flow_shift: 3.0
            video_loss_weight: 1.0
            audio_loss_weight: 1.0

distributed:
    backend: nccl
    timeout_minutes: 60
    sequence_parallel:
        enabled: false
        size: 1
    fsdp2:
        enabled: true
        sharded_load: true
        size: ${oc.env:WORLD_SIZE,1}
        reshard_after_forward:
            root_reshard: false
            block_reshard: true
        mixed_precision:
            param_dtype: null
            reduce_dtype: fp32
            output_dtype: null
            cast_forward_inputs: false

data:
    train:
        name: video_dataset
        data_path:
            - /path/to/datasets/minimax_h3/train.jsonl
        video_column: video
        audio_column: audio
        prompt_column: prompt
        height: 768
        width: 1344
        # H3 requires F = 17*n+5.
        num_frames: 124
        frame_rate: 24
        fix_frame_rate: true
        random_start: false
        skip_missing: true
        decode_retries: 3
        num_workers: 4
        pin_memory: true
        shuffle: true
        drop_last: true

scheduler:
    num_train_timesteps: 1000
    timestep_distribution: uniform
    min_sigma: 0.001
    max_sigma: 1.0
    # The H3 capability applies separate rational shifts to video and audio.
    time_shift_settings:
        do_time_shift: false

training:
    method: flow_matching
    train_type: lora
    max_train_iters: 3000
    gradient_accumulation_iters: 1
    gradient_checkpointing: true
    max_grad_norm: 1.0
    lr_scheduler: constant
    lr_warmup_iters: 10
    save_every_iters: 100
    save_total_limit: 10
    lora:
        rank: 128
        alpha: 128
        target_modules:
            - to_q
            - to_k
            - to_v
            - to_out.0
            - ff.net.0.proj
            - ff.net.2
    optimizer:
        learning_rate: 0.0001
        adam_beta1: 0.9
        adam_beta2: 0.999
        weight_decay: 0.01
        adam_epsilon: 0.00000001
    output_dir: ./output_train/minimax_h3_fl2av_lora

inference:
    method: none
    infer_every_iters: null

logging:
    rank_zero_only: true
    train_log_every_iters: 10

resume:
    auto_resume: false

```


---

## FILE: lightx2v_train/configs/train/dmd/minimax_h3_fl2av_dmd_lora.yaml (https://github.com/ModelTC/LightX2V/blob/HEAD/lightx2v_train/configs/train/dmd/minimax_h3_fl2av_dmd_lora.yaml)

```
model:
    name: minimax_h3_fl2av
    # This must be the converted Diffusers model root containing
    # transformer/config.json (MiniMaxH3Transformer3DModel).
    pretrained_model_name_or_path: ${oc.env:H3_MODEL_PATH}
    running_dtype: bf16
    local_files_only: true
    patch_size: [1, 2, 2]
    video_latent_channels: 24
    audio_latent_channels: 32
    vae_spatial_scale_factor: 16
    video_fps: 24
    audio_sampling_rate: 32000
    audio_latents_per_second: 40
    # Keep frozen cache encoders off GPU outside their forward passes.
    cache_encoder_cpu_offload: true
    # H3 keeps its diffusion latents in fp32; prompt embeddings remain bf16.
    # Full SDPA is impractical at H3 sequence lengths; this matches run_h3.py.
    attention_backend: _flash_3_hub
    capabilities:
        distribution_matching:
            # The H3 denoiser applies separate shifts to one shared base sigma.
            video_flow_shift: 6.0
            audio_flow_shift: 3.0
            video_loss_weight: 1.0
            audio_loss_weight: 1.0

distributed:
    backend: nccl
    timeout_minutes: 60
    sequence_parallel:
        enabled: false
        size: 1
    fsdp2:
        enabled: true
        size: ${oc.env:WORLD_SIZE,1}
        reshard_after_forward:
            root_reshard: false
            block_reshard: true
        # H3's checkpoint intentionally mixes fp32 projections with bf16 blocks.
        mixed_precision:
            param_dtype: null
            reduce_dtype: fp32
            output_dtype: null
            cast_forward_inputs: false

data:
    train:
        name: cache_dataset
        data_path: ${oc.env:H3_TRAIN_META}
        tasks: ${oc.decode:${oc.env:H3_TASKS,'[]'}}
        num_workers: 2
        pin_memory: true
        shuffle: true
        drop_last: true

scheduler:
    min_sigma: 0.02
    max_sigma: 0.98
    num_train_timesteps: 1000
    timestep_distribution: uniform
    time_shift_settings:
        # Joint audio/video latents apply their modality shifts in the model.
        do_time_shift: false

training:
    method: dmd
    max_train_iters: 1000
    gradient_accumulation_iters: 1
    gradient_checkpointing: true
    max_grad_norm: 1.0
    lr_scheduler: constant
    lr_warmup_iters: 0
    save_every_iters: 100
    save_total_limit: 5
    dmd:
        # Four student model evaluations; the terminal zero sigma is added internally.
        num_inference_steps: 4
        fake_update_ratio: 5
        gan:
            enabled: false
            feature_block: 49
        score_sampling:
            type: continuous_uniform
        # Do not apply the generic scheduler shift before H3's modality shifts.
    student:
        train_type: lora
        lora:
            rank: 128
            alpha: 8
            target_modules:
                - to_q
                - to_k
                - to_v
                - to_out.0
                - ff.net.0.proj
                - ff.net.2
        optimizer:
            learning_rate: 0.00005
            adam_beta1: 0.0
            adam_beta2: 0.999
            weight_decay: 0.01
            adam_epsilon: 0.00000001
    fake:
        train_type: full
        optimizer:
            learning_rate: 0.0000004
            adam_beta1: 0.0
            adam_beta2: 0.999
            weight_decay: 0.01
            adam_epsilon: 0.00000001
    teacher:
        guidance_scale: 1.0
        cfg_norm: none
    output_dir: ./output_train/minimax_h3_fl2av_dmd

inference:
    method: none
    infer_every_iters: null

logging:
    rank_zero_only: true
    train_log_every_iters: 10

resume:
    auto_resume: true

```


---

# ISSUES / PRs of ModelTC/LightX2V


---

# ISSUES in ModelTC/LightX2V matching "H3" (top 15 by comments)


### #1321 [closed] [Model Request]新增对MiniMax H3的视频模型的4或8 steps LoRA支持
URL: https://github.com/ModelTC/LightX2V/issues/1321  by Goan114 2026-08-03T05:07:12Z  comments: 7

MiniMax H3视频生成模型于近日发布。模型性能十分优秀，但是20 steps需要的计算时间真的太长......
可否新增对MiniMax H3视频模型的支持？

> **Comment by KarinMiku (2026-08-03T06:23:05Z):**

indeed, this is the only thing that minimax h3 needs to be perfect! the text enconder and vae are really fast already


> **Comment by Goan114 (2026-08-03T11:12:05Z):**

> indeed, this is the only thing that minimax h3 needs to be perfect! the text enconder and vae are really fast already

Yes, the model inference is really fast too! I have 11s/it on my RTX 5080 (16G VRAM) laptop, almost 2-3x faster than Wan 2.2's single step!
However the model still needs 20+ steps to generate, that's problem.
If they can release a 4 step lora, or inference framework that can increase inference speed, that's better!


> **Comment by kingp0dd (2026-08-03T13:51:52Z):**

I'm surprised with how fast h3 is. If it can be distilled, it can match or even be faster than ltx in my case


> **Comment by overallbit (2026-08-04T00:24:26Z):**

> > indeed, this is the only thing that minimax h3 needs to be perfect! the text enconder and vae are really fast already
> 
> Yes, the model inference is really fast too! I have 11s/it on my RTX 5080 (16G VRAM) laptop, almost 2-3x faster than Wan 2.2's single step! However the model still needs 20+ steps to generate, that's problem. If they can release a 4 step lora, or inference framework that can increase inference speed, that's better!

what resolution is that? 1366x768?


> **Comment by Goan114 (2026-08-04T01:50:18Z):**

> > > indeed, this is the only thing that minimax h3 needs to be perfect! the text enconder and vae are really fast already
> > 
> > Yes, the model inference is really fast too! I have 11s/it on my RTX 5080 (16G VRAM) laptop, almost 2-3x faster than Wan 2.2's single step! However the model still needs 20+ steps to generate, that's problem. If they can release a 4 step lora, or inference framework that can increase inference speed, that's better!
> 
> what resolution is that? 1366x768?

864×480. Same resolution I used in WAN 2.2.


> **Comment by truss-frame (2026-08-05T05:55:20Z):**

+1


> **Comment by Goan114 (2026-08-07T16:01:18Z):**

由于LightX2V在 #1326 中加入了推理框架对MiniMax H3的支持，并将 #1329 标记为Closed as Completed，现将标题修改为“[Model Request]新增对MiniMax H3的视频模型的4或8 steps LoRA支持”



### #1728 [closed] MiniMax-H3: allow warmup together with DPCache (compile warmup and step caching are independent)
URL: https://github.com/ModelTC/LightX2V/issues/1728  by ZhengXinwei-F 2026-11-18T05:56:21Z  comments: 2

## Summary

For MiniMax-H3 with `use_compile: true`, `warmup: true` and `feature_caching: "DPCache"` cannot be enabled together: `DefaultRunner.warmup()` raises `NotImplementedError("Warmup does not support feature caching")` for any caching mode other than `NoCaching`. A DPCache deployment therefore has to run with `warmup: false`, and every fresh worker pays the `torch.compile` cost on live requests.

The two features operate at different levels and do not depend on each other:

- DPCache operates **across denoising steps**. For each step it decides whether to run the transformer blocks or predict the output with a Taylor expansion.
- `torch.compile` operates **inside a step**, one compiled callable per transformer block.

Steps that DPCache computes run the same compiled blocks that warmup would compile. The restriction is a generic guard in the shared runner, not a real conflict between the two features. This issue proposes letting the H3 warmup run with DPCache enabled by bypassing the caching logic only while warming up.

Code references below are against `fc646ec`.

## Where the restriction comes from

[`lightx2v/models/runners/default_runner.py#L152-L162`](https://github.com/ModelTC/LightX2V/blob/fc646ec128a3170e0ce3282fe9578695691407ca/lightx2v/models/runners/default_runner.py#L152-L162)

```python
def warmup(self):
    if not self.config.get("warmup", False):
        return
    ...
    if self.config.get("feature_caching", "NoCaching") != "NoCaching":
        raise NotImplementedError("Warmup does not support feature caching")
    self.run_warmup()
```

The guard applies to every model and every caching mode. For H3, `MiniMaxH3Model._init_infer_class` only accepts `NoCaching` and `DPCache` ([`model.py#L527-L544`](https://github.com/ModelTC/LightX2V/blob/fc646ec128a3170e0ce3282fe9578695691407ca/lightx2v/models/networks/minimax_h3/model.py#L527-L544)), so the guard's only effect on H3 is to rule out warmup whenever DPCache is on.

## Why DPCache and compile warmup are independent

1. **Compilation is per block.** `BaseTransformerInfer.get_compiled_block` wraps each block in `torch.compile(block_runner, dynamic=None)` and caches the result in `self.compiled_blocks` by block index ([`common/transformer_infer/transformer_infer.py#L29-L45`](https://github.com/ModelTC/LightX2V/blob/fc646ec128a3170e0ce3282fe9578695691407ca/lightx2v/common/transformer_infer/transformer_infer.py#L29-L45)).
2. **DPCache sits above the block loop.** `MiniMaxH3TransformerInferDPCaching` subclasses `MiniMaxH3TransformerInfer`. On a computed step, its `infer` calls the inherited `self.infer_func(block_weights.blocks, ...)`, which is the same block loop and the same `compiled_blocks`. On a predicted step it runs no blocks at all ([`dpcache/transformer_infer.py#L90-L128`](https://github.com/ModelTC/LightX2V/blob/fc646ec128a3170e0ce3282fe9578695691407ca/lightx2v/models/networks/minimax_h3/infer/feature_caching/dpcache/transformer_infer.py#L90-L128)). Both paths use the same compiled graphs.
3. **What H3 warmup needs from the transformer** is only that the blocks are traced and compiled for two different shapes ([`minimax_h3_runner.py#L101-L104`](https://github.com/ModelTC/LightX2V/blob/fc646ec128a3170e0ce3282fe9578695691407ca/lightx2v/models/runners/minimax_h3/minimax_h3_runner.py#L101-L104): `480x480x158` and `544x960x124`). With `dynamic=None`, the second shape makes Dynamo recompile with the changed dimensions marked dynamic. After that, later resolutions reuse the dynamic graph instead of recompiling. Step skipping plays no part in this.

## Why simply deleting the guard is not enough

Today `run_warmup` reaches the transformer through `self.model.infer(self.inputs)` ([`minimax_h3_runner.py#L147`](https://github.com/ModelTC/LightX2V/blob/fc646ec128a3170e0ce3282fe9578695691407ca/lightx2v/models/runners/minimax_h3/minimax_h3_runner.py#L147)). With DPCache on, that call goes through `MiniMaxH3TransformerInferDPCaching.infer`:

- Warmup runs only `_WARMUP_STEP_COUNT = 2` steps, but DPCache assumes it sees the full calibrated schedule from step 0. Steps 0 and 1 happen to be computed steps (`dpcache_first_steps >= order + 1 >= 2`), so the run does not fail. Still, warmup then becomes part of DPCache's step bookkeeping, which it should not be.
- With `dpcache_calibration: false`, the warmup steps fill `self.factors` with warmup-sized feature tensors. They stay referenced after warmup, because `clear_warmup_state` ([`#L190-L194`](https://github.com/ModelTC/LightX2V/blob/fc646ec128a3170e0ce3282fe9578695691407ca/lightx2v/models/runners/minimax_h3/minimax_h3_runner.py#L190-L194)) does not call `transformer_infer.clear()`. They are only released at step 0 of the next real request.
- With `dpcache_calibration: true`, the warmup steps are appended to `self.history` through `_record` and stay there in the same way.
- The log shows `H3 DPCache selected steps ...` and `H3 DPCache step N: compute` lines for warmup inputs, which is misleading.

None of this affects correctness of later requests, because step 0 resets the state. But warmup should not interact with the cache at all.

## Impact of running without warmup

When DPCache is on and warmup is therefore off, with `use_compile: true`:

- The first live request pays the initial compilation of every block.
- The first live request with a **second** resolution pays a recompile. On our TP8 deployment, the first denoising step of that request took about 6 s longer than its later steps.
- Until a second shape arrives, the process runs the static-shape graph. Its host-side behaviour differs from steady state. For the same `768x1344` request we measured the following per-step times:

  | | Host enqueue | Wait at end of step | Total |
  |-|-|-|-|
  | Static graph (fresh process) | 4.0 s | 13.9 s | 17.9 s |
  | Dynamic graph (after another shape) | 17.5 s | 0.4 s | 17.9 s |

  With the static graph the host enqueues the whole step ahead of the device. After the recompile it is paced block by block. Total step time is the same, so this is not a throughput problem. It does mean a fresh worker does not behave like a warm one, so latency, timing logs and anything that synchronizes at step boundaries look different until the worker has seen a second resolution.

Warmup exists precisely to move this cost and this transition out of live traffic, and DPCache deployments currently cannot use it.

## Proposed change

1. **Bypass DPCache while warming up.** Give `MiniMaxH3TransformerInferDPCaching` a way to run the plain inherited path. For example, a flag or context manager that makes `infer` delegate to `super().infer(...)`, plus a `clear()` call when it exits:

   ```python
   # MiniMaxH3TransformerInferDPCaching
   @contextmanager
   def bypass(self):
       self._bypass = True
       try:
           yield
       finally:
           self._bypass = False
           self.clear()

   def infer(self, block_weights, pre_infer_out):
       if self._bypass:
           return super().infer(block_weights, pre_infer_out)
       ...
   ```

   `MiniMaxH3Runner.run_warmup` would then wrap its denoising loop in `bypass()` when the transformer infer object provides it. The compiled blocks are shared, so the compilation done during warmup carries over to cached inference unchanged.

2. **Make the guard opt-out per runner** instead of global. For example, `DefaultRunner.warmup()` could check a class attribute such as `warmup_supports_feature_caching = False`, which `MiniMaxH3Runner` sets to `True` once (1) is in place. Other models and caching modes keep today's behaviour until someone checks them.

## Suggested checks

- `warmup: true`, `use_compile: true`, `feature_caching: "DPCache"` starts without error, both with `dpcache_calibration: true` and with `false`.
- After warmup, the first request at a resolution not used in warmup triggers no recompile (`TORCH_LOGS=recompiles` stays silent).
- With a fixed seed, outputs are identical whether warmup was on or off. The selected steps and the saved cost file are identical too.
- No `H3 DPCache ...` log lines appear during warmup, and `transformer_infer.factors` and `history` are empty after warmup.


> **Comment by zhangbilang (2026-11-18T11:55:01Z):**

@ZhengXinwei-F Thanks for the detailed report! MiniMax-H3 now supports warmup: true together with feature_caching: "DPCache" on main. Warmup bypasses DPCache, so only real requests update its state. Fixed in [#1729](https://github.com/ModelTC/LightX2V/pull/1729).


> **Comment by ZhengXinwei-F (2026-11-19T03:10:27Z):**

@zhangbilang Thank you so much for the quick and thoughtful fix, and to @llmc-reviewer for reviewing and merging it so fast. #1729 landed within hours of the report, which is really appreciated.

The design is cleaner than what I had proposed: letting each runner declare which caching modes its warmup supports keeps the shared guard intact for other models, and marking warmup steps on the scheduler so DPCache falls through to the plain path means the synthetic warmup inputs never touch its calibration state. Only real requests update it.

We'll pick up #1729 in our deployment and enable `warmup: true` together with DPCache, so fresh workers no longer pay the compile cost on live requests. Thanks again to the whole LightX2V team!




### #1678 [open] MiniMax-H3 slows down over uptime: torch 2.9 torch.compile list leak + forced gc.collect() per request
URL: https://github.com/ModelTC/LightX2V/issues/1678  by ZhengXinwei-F 2026-11-07T17:16:16Z  comments: 1

## Summary

With torch 2.9.0/2.9.1 and `use_compile` / `vae_use_compile` enabled, MiniMax-H3 serving gets slower the longer the process runs. Per-step DiT time and VAE decode time stay constant; all of the growth is in the `Offload DiT` stage, specifically the forced full `gc.collect()` it runs at the end of every request.

Two things combine:

1. **torch 2.9.x leaks two small lists per `torch.compile`'d call** (an empty `[]` and a `[[]]`, 120 bytes in total). They survive a full collection and have no Python referrers. This matches the upstream regression [pytorch/pytorch#174468](https://github.com/pytorch/pytorch/issues/174468) (2.9.0/2.9.1 affected, 2.8 not, fixed from 2.10).
2. **`_offload_transformer` forces a full GC on every request**: `maybe_empty_cache(force=True, collect_garbage=True)` ([minimax_h3_runner.py](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/runners/minimax_h3/minimax_h3_runner.py#L722)) → `gc.collect()` ([default_runner.py](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/runners/default_runner.py#L279)). A full collection walks every tracked object, so each request re-walks everything leaked by all previous requests.

Either alone is harmless: the leak is ~1 MB per request, and a full GC over a stable heap takes a few ms. Together, GC time grows linearly with the number of requests served.

## Observed

- About 13.7k leaked lists per 768p / 120-frame request: roughly 8k from the compiled VAE decoder blocks, the rest from the compiled DiT blocks around the flash-attn call that graph-breaks (located with `tracemalloc.get_object_traceback`).
- Forced GC grows by about 1.6–3.6 ms per request, depending on the task. For a long-running server that is roughly +2.9 s per request per day of uptime: after one day a ~108 s request takes ~111 s, after three days ~117 s.
- Restarting the process resets it.

## Minimal reproduction (independent of LightX2V)

```python
import gc, torch
f = torch.compile(lambda x: x + 1, backend="eager")   # same result with "inductor"
x = torch.randn(9, 3840, device="cuda")
[f(x) for _ in range(50)]
gc.collect(); a = sum(type(o) is list for o in gc.get_objects())
[f(x) for _ in range(2000)]
gc.collect(); b = sum(type(o) is list for o in gc.get_objects())
print(torch.__version__, b - a)
```

| torch | lists leaked per 2000 calls (eager and inductor) |
|---|---|
| 2.9.1 | 4000 |
| 2.11.0 | 0 |

The eager backend leaks too, so this is in Dynamo's frame handling, not in generated kernels.

## Suggested fixes

- **Avoid rescanning survivors.** After the full collection, call `gc.freeze()` so the next collection only walks new objects, and unfreeze plus collect once every N calls to reclaim cycles among frozen objects. In our test this keeps the forced GC at ~2 ms per request instead of growing (0.14 s after 85 requests without it). I can open a PR with this behind config switches (default on, can be disabled).
- **Document or constrain the torch version.** `torch>=2.1` allows 2.9.x. Upgrading to torch ≥ 2.10 removes the leak itself.


## Related: a frozen runner cannot be released (latent since #1188)

Since #1188, `BaseRunner._maybe_freeze_gc()` runs `gc.collect()` + `gc.freeze()` once after a runner's first `run_pipeline` and never unfreezes. A runner holds itself in reference cycles (`self.run_input_encoder = self._run_input_encoder_local_*` is one of its own bound methods), so once frozen, dropping or replacing it in the same process does not free it: `gc.collect()` skips the permanent generation, `__del__` never runs, and its models stay in memory.

The bug is latent in LightX2V's own entry points: the server worker builds one runner per process, and `LightX2VPipeline.create_generator` refuses to create a second one. It shows up when external code replaces a runner in a long-lived process (notebooks, model-switching scripts), and repeated replacement can run out of memory. `lazy_load` / `unload_modules` are not affected because `_maybe_freeze_gc()` skips them.

```python
import gc, weakref

class Runner:
    def __init__(self):
        self.run_input_encoder = self._enc   # self -> bound method -> self
        self.weights = bytearray(10**8)
    def _enc(self):
        pass

r = Runner(); gc.collect(); gc.freeze()       # what _maybe_freeze_gc() does
w = weakref.ref(r); del r; gc.collect()
print(w() is not None)                         # True: still alive
gc.unfreeze(); gc.collect(); print(w() is None)  # True: released
```

Addressed in #1680: the periodic full-pass counter is process-wide, so the next periodic pass releases a dropped runner, and `base_runner.release_frozen_gc()` releases it immediately.


> **Comment by ZhengXinwei-F (2026-11-09T06:13:18Z):**

/assign



### #1379 [closed] Observations on MiniMax H3 Step Distillation
URL: https://github.com/ModelTC/LightX2V/issues/1379  by Hammingbo 2026-08-13T13:47:49Z  comments: 1

I’ve recently been experimenting with step distillation for MiniMax. I’ve found that the base model seems to retain fairly strong performance even with very few inference steps (8 or 4). Have you observed something similar? Is it possible that, alongside CFG distillation, the official model also uses some form of few-step distillation?

> **Comment by Hammingbo (2026-08-13T13:55:44Z):**

https://github.com/user-attachments/assets/02706ff3-18e1-4b84-92df-fd88d2cf1f8c

https://github.com/user-attachments/assets/197d0a2e-cdac-4ca3-836a-53948b5da2b2

https://github.com/user-attachments/assets/c7ed12da-7e8f-4a81-b651-b86c6b99beb2  2step-4step-8step



### #1329 [closed] MiniMax-H3 runs end-to-end on Intel XPU (incl. a custom INT8 path), but output is wrong (blocky video, garbled audio) — requesting native model support
URL: https://github.com/ModelTC/LightX2V/issues/1329  by hanyan009 2026-08-04T02:58:42Z  comments: 1

## Problem Description

MiniMax-H3 has no native support in LightX2V. We did a from-scratch native port (DiT, T2VA packing, dual video/audio rectified-flow scheduler, both VAEs) by reading the reference Python implementation in [`ComfyUI_RH_MinMaxH3`](https://github.com/HM-RunningHub/ComfyUI_RH_MinMaxH3), the only existing implementation of this architecture. Note that repo already ships an INT8-quantized checkpoint plus layerwise DiT offload / AdaLN weight-release tricks, and claims to run MiniMax-H3 on a single **24GB** GPU — so the model is already known to be runnable at this memory budget, this isn't a from-scratch feasibility question.

The pipeline runs **end-to-end without runtime errors** on Intel XPU (1× Intel Arc B65, 32GB discrete VRAM each) — real text prompt → Qwen3-VL encode → 50-block DiT denoise loop → VAE decode → real `.mp4`/`.wav` output. But the generated content is wrong: the video shows visible block/tile artifacts, and the audio is garbled.

https://github.com/user-attachments/assets/fd15bde1-9bc2-4684-926a-8ffc1dcc22b9

## Confirmed working / Observed issue

- No crashes, no NaN/Inf anywhere; correct output shapes/dtypes; sane memory (GPU memory flat during block-offloaded inference, ~14.5GB peak)
- But: video has clear block/patch tiling artifacts (not a coherent scene), audio is garbled — reproducible


## Request

Requesting official native support for MiniMax-H3. Given a working quantized+offloaded reference already exists (ComfyUI_RH_MinMaxH3, 24GB), this looks achievable on LightX2V too — but a review/implementation from maintainers familiar with the codebase's conventions would catch the remaining bugs far faster than we can by guessing against reference source alone. Happy to share our current implementation as a starting point.


> **Comment by gushiqiao (2026-08-04T11:08:15Z):**

We already have an initial implementation in PR https://github.com/ModelTC/LightX2V/pull/1326 , and we are continuing to optimize both inference latency and GPU memory usage.

In the current version, enabling block offload allows the BF16 model to run with less than 24 GB of GPU memory. If you would like to use INT8 quantization, you can convert the model with the following command:

python tools/convert/converter.py
--source /path/to/MiniMax-H3/transformer
--output /path/to/MiniMax-H3/quantized/int8
--output_name minimax_h3_int8
--model_type h3
--quantized
--linear_type int8
--device cuda
--single_file



### #1445 [open] Missing FP8 quantized MiniMax-H3 models referenced in configs
URL: https://github.com/ModelTC/LightX2V/issues/1445  by nanwanj 2026-08-26T17:59:40Z  comments: 0

Hi,

While going through the source code and config folder, we noticed that several H3 configs (e.g., [configs/minimax_h3/fp8](https://github.com/ModelTC/LightX2V/tree/main/configs/minimax_h3/fp8)) reference a MiniMax-H3_quantized/fp8 model. However, we couldn't find this model in your Hugging Face repo.

Additionally, the model conversion/quantization tool ([tools/convert](https://github.com/ModelTC/LightX2V/blob/main/tools/convert/readme_zh.md)) doesn't appear to support H3 models, so we're unable to generate the quantized weights ourselves.

Could you let us know where the FP8 quantized H3 models can be downloaded, or whether there are plans to release them (or add H3 support to the conversion tool)?

Thanks!


### #1593 [open] [Bug] PLATFORM=amd_rocm ignores expandable_segments:True, MiniMax-H3 OOMs on 32 GB GPUs
URL: https://github.com/ModelTC/LightX2V/issues/1593  by Shan2L 2026-10-10T15:29:31Z  comments: 0

### Description
`lightx2v/pipeline.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with `os.environ.setdefault`.
With `PLATFORM=amd_rocm` the setting never takes effect. `lightx2v/__init__.py` imports
`lightx2v_platform.set_ai_device` before `pipeline.py`, and on `amd_rocm` that import pulls in `aiter`
(`lightx2v_platform/ops/attn/amd_rocm/flash_attn.py` and `AmdRocmDevice.init_device_env()`). `aiter`
queries the device at import time, which initializes the CUDA/HIP caching allocator. PyTorch then never
reads the variable that `pipeline.py` sets afterwards.

Without expandable segments, PyTorch reserves about 1.5 GiB more for the same tensors. On a 32 GB GPU
that is enough for MiniMax-H3 ref2av to OOM at DiT step 1.

### Steps to Reproduce
One GPU, no model needed:
```bash
for p in cuda amd_rocm; do PLATFORM=$p python -c "
import os, torch, lightx2v
x = torch.empty(64 << 20, dtype=torch.uint8, device='cuda')
print(os.environ['PLATFORM'], os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
      'is_expandable =', torch.cuda.memory_snapshot()[0]['is_expandable'])"; done
```

### Expected Result
`is_expandable = True` on both platforms, because `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set.

### Actual Result
```
cuda expandable_segments:True is_expandable = True
amd_rocm expandable_segments:True is_expandable = False
```
End-to-end impact (MiniMax-H3 ref2av, 768x1344, 362 frames, 3 steps, TP2, DiT block offload, 2x R9700 32 GB):

| | result |
|---|---|
| `PLATFORM=cuda` | completes, ~101 s/step |
| `PLATFORM=amd_rocm` | `torch.OutOfMemoryError` in `transformer_infer._ff` (`F.silu`) at step 1 |
| `PLATFORM=amd_rocm` + `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | completes, ~101 s/step |

### Environment Information
- Operating System: Ubuntu 24.04, image `rocm/pytorch:rocm10.0_ubuntu24.04_py3.13_pytorch_release_2.11.0` (torch 2.11.0+rocm10.0.0)
- Commit ID: `main` 4fe984c0 (also `0.5.0` 95e9b86b); aiter main
- GPU: AMD Radeon AI PRO R9700 (gfx1201, 32 GB)

### Log Information
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.71 GiB. GPU 0 has a total capacity of 31.86 GiB
of which 1.46 GiB is free. Of the allocated memory 17.00 GiB is allocated by PyTorch, and 2.91 GiB is reserved
by PyTorch but unallocated.
```
The large "non-PyTorch" share implied by this message is transient. Synchronized per-block probes show
0.8 GiB non-PyTorch memory on both platforms. After attention in block 0, allocated memory is the same
(13.57 vs 13.54 GiB) and only reserved memory differs (26.11 vs 24.28 GiB).

### Additional Information
Ruled out by experiment: cuDNN on/off, the `sgl_kernel` injection, process-group options, and
`set_device` order. Keeping `aiter` from being imported before `pipeline.py` fixes the OOM.
Proposed fix: run the same `setdefault` in `lightx2v/__init__.py` before `import lightx2v_platform.set_ai_device`.
A user-exported `PYTORCH_CUDA_ALLOC_CONF` still takes precedence. PR: #1595 



### #1594 [open] [Bug] [ROCm] MiniMax-H3 output differs run to run: hipBLASLt picks a per-process kernel for the FP32 proj_out GEMM
URL: https://github.com/ModelTC/LightX2V/issues/1594  by Shan2L 2026-10-10T15:33:23Z  comments: 0

### Description
On AMD ROCm (gfx1201, Radeon AI PRO R9700), two runs of the same MiniMax-H3 ref2av job with the same seed and
inputs produce different videos and audio. The first divergence is the FP32 `proj_out` GEMM in
`post_infer` (`Default-ForceFp32` -> `torch.addmm`). hipBLASLt selects a different GEMM kernel per process for
this shape, so the step-0 velocity differs in the last bits, and the sampler amplifies the difference over the
following steps.

### Steps to Reproduce
Run the same MiniMax-H3 ref2av job twice (same seed and inputs, `PLATFORM=amd_rocm`, TP2) on two GPU pairs and
compare the decoded video/audio tensors.

### Expected Result
Identical outputs for identical inputs and seed.

### Actual Result
Video and audio differ between runs (see table under Additional Information).

### Environment Information
- Image `rocm/pytorch:rocm10.0_ubuntu24.04_py3.13_pytorch_release_2.11.0` (torch 2.11.0+rocm10.0.0)
- LightX2V `0.5.0` (95e9b86b); also checked on `main` (4fe984c0) - see below
- MiniMaxAI/MiniMax-H3 (revision 42ed227e), ref2av, 768x1344, 362 frames, seed 42, 3 steps, TP2, DiT block offload + AdaLN cache
- 2x R9700 (gfx1201)

### Log Information
- Q/K/V of all 50 DiT blocks at step 0 are bitwise identical between runs.
- The input of `post_infer` (`h`) is bitwise identical; its FP32 `proj_out` output differs in ~94% of
  elements (ulp level). From here on the latents differ.
- Kernel trace of `proj_out` in one run: rank 0 used a `MT16x8x64` hipBLASLt kernel, rank 1 a
  `MT128x128x16` one, for the same shape. Other runs pick other combinations.
- `torch.use_deterministic_algorithms(True, warn_only=True)` emits no warning.
- A standalone script with the same GEMM shapes (M 109256..128358, K 5376, N 96/32) is deterministic, so
  the selection seems to depend on process state (heuristics / previously used shapes), not only the shape.

### Additional Information
Workaround:
Run FP32 `mm`/`addmm`/`linear`/`conv*` with `torch.backends.cuda.preferred_blas_library("cublas")`
(rocBLAS on ROCm) and `ROCBLAS_USE_HIPBLASLT=0`. BF16 GEMMs stay on hipBLASLt. With this, two runs on
different GPU pairs give bitwise identical video and audio, with no measurable change in step time.
On `main` (4fe984c0, with the allocator fix from #1593 so that `PLATFORM=amd_rocm` does not OOM), 4 runs on 4 GPU pairs:

| | video / audio, run A vs run B | DiT s/step |
|---|---|---|
| default | differ | 100.5-101.8 |
| FP32 GEMM/conv on rocBLAS | bitwise identical | 100.0-100.9 |

PR: #1596  (opt-in `ROCM_DETERMINISTIC_FP32_BLAS=1` in `AmdRocmDevice.init_device_env()`).



### #1736 [closed] MiniMax-H3: one undecodable packet in a reference audio/video fails the request (e.g. concatenated MP3)
URL: https://github.com/ModelTC/LightX2V/issues/1736  by ZhengXinwei-F 2026-11-19T03:08:00Z  comments: 0

## Problem

MiniMax-H3 reference decoding (`lightx2v/models/networks/minimax_h3/packing_ref2av.py`, at 0b6805e7) iterates `container.decode(stream)` in `_decode_reference_soundtrack` and `decode_reference_video`. PyAV raises on the first packet the decoder rejects, so a single bad packet fails the whole request, while ffmpeg and ordinary players log the packet and keep decoding.

A common real-world case is an MP3 reference that is two MP3 files joined byte for byte (`cat a.mp3 b.mp3`). The second file's ID3v2 tag sits in the middle of the stream, the demuxer hands it to the decoder as a packet, and the request fails with:

```
av.error.InvalidDataError: [Errno 1094995529] Invalid data found when processing input: 'avcodec_send_packet()'; last error log: [mp3float] Header missing
```

The same file plays normally, `ffmpeg -i joined.mp3 -f null -` reports the error and still decodes all of the audio, and the hosted MiniMax-H3 API accepts it as `reference_audio` and generates video that follows the audio.

## Reproduce

```bash
ffmpeg -v error -f lavfi -i "sine=f=440:d=2" -ar 24000 -ac 1 a.mp3
ffmpeg -v error -f lavfi -i "sine=f=660:d=0.5" -ar 24000 -ac 1 b.mp3
cat a.mp3 b.mp3 > joined.mp3
ffmpeg -v error -i joined.mp3 -f null -        # logs "Header missing", decodes ~2.5 s
python -c "from lightx2v.models.networks.minimax_h3.packing_ref2av import decode_reference_audio as d; d('joined.mp3')"  # raises InvalidDataError
```

## Proposal

Decode packet by packet, skip packets that raise `av.error.InvalidDataError`, and log how many were skipped. When nothing decodes at all, the existing "contains no samples" / "No video frames" errors still fire. A PR follows.



### #1664 [closed] MiniMax-H3: resident Qwen3-VL text encoder keeps its full host copy (~47 GiB per process); text_encoder_host_pinned=false does not make it reclaimable
URL: https://github.com/ModelTC/LightX2V/issues/1664  by ZhengXinwei-F 2026-11-02T13:18:15Z  comments: 0

## Summary

When the MiniMax-H3 Qwen3-VL text encoder is resident on the accelerator (`text_encoder_cpu_offload: false`), its host copy is never released after `to_cuda()`. It costs about 47–55 GiB of host memory per process that nothing reads again.

`text_encoder_host_pinned: false` (#1424) does not help on our platform. The pageable copy maps the checkpoint privately (`rw-p`), and every page ends up `Private_Dirty`/`Anonymous` after the host-to-device copy. So the memory is not reclaimable either, contrary to the comment on `_adopt_pageable_weights`.

## Environment

- LightX2V: MiniMax-H3 path as on `main` (`lightx2v/models/input_encoders/hf/minimax_h3/qwen3vl.py`; `_adopt_pageable_weights` and `load_text_encoder` are unchanged there).
- Hardware: Moore Threads MTT S5000 (80 GB), driver 3.3.8, `PLATFORM=musa`.
- Checkpoint on a FUSE filesystem.
- Single-card fl2av, 8-step turbo LoRA (dynamic), BF16.
- Config: `cpu_offload: true`, `offload_granularity: block`, `text_encoder_cpu_offload: false`, `parallel: {}`.

## Observed

`/proc/<pid>/smaps` of the server process after warmup, grouped by mapping:

| mapping | Rss |
|---|---|
| `/dev/dri` (driver-pinned host memory: DiT block-offload copy) | 68.5 GiB |
| checkpoint files under the model directory (`rw-p`) | 46.9 GiB, all of it `Private_Dirty` / `Anonymous` |
| other anonymous | 4.4 GiB |

The cgroup reports `anon` ≈ 51.6 GiB and `file` ≈ 0. With `text_encoder_host_pinned: true` the same weights are pinned instead, about 120 GiB total either way.

Eight single-card workers on a 1 TiB, 8-GPU node therefore need about 960 GiB. Nodes ran with 2–3 GiB available and one hit a system OOM.

## Expected

A resident text encoder should not keep a host copy: after `to_cuda()` it is never read again. `to_cpu()` can copy back from the device if it is ever needed.

## Proposed fix

After `text_encoder.to_cuda()` in the resident branch of `load_text_encoder`, drop every `pin_*` tensor of the backbone's weight modules. The change is small:

```python
elif not self.cpu_offload:
    text_encoder.to_cuda()
    released = text_encoder.release_host_weights()  # drops storage.pin_* tensors, gc.collect()
```

Measured with the patch on the same setup:

- the load log reports `Released 46.9 GiB of host text-encoder weights`;
- after warmup the checkpoint mappings are gone and process Rss drops from ~120 GiB to 75.7 GiB (`/dev/dri` 68.5 + anonymous 4.4 + other 2.7);
- load time is unchanged (136.9 s vs 133.6 s); a 5 s 768P request runs in 106.1 s vs 106.6 s before, GPU peak unchanged.

The peak during loading is still ~118 GiB (`VmHWM`), because the host copy and the device copy coexist until the release. Loading layer by layer straight to the device would also remove that peak.

Separately, the assumption behind `_adopt_pageable_weights` ("retaining safetensors-backed storage keeps those pages reclaimable") does not hold here. The mapping is private, and the device copy writes its pages (most likely the driver pins them for write during the transfer). This matters for offloaded configurations, which do read the host copy, so it may be worth a look on its own.



### #1336 [closed] [Bug] WAN 2.2 generates black output when TP=2 and SP=2 are enabled simultaneously on XPU
URL: https://github.com/ModelTC/LightX2V/issues/1336  by qiuxin2012 2026-08-07T01:17:57Z  comments: 0

### Description
I tested [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) on Intel XPU(B70) with multi-GPU execution. Both Tensor Parallelism (TP=2) and Ulysses Sequence Parallelism (SP=2) work correctly when enabled separately. However, when TP=2 and SP=2 are used together, the generated result is entirely black.

### Steps to Reproduce
Infer script:
```
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/llm/models/Wan2.2-TI2V-5B}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/dist_infer/wan22_ti2v_t2v_tp_ulysses.json}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/output_wan22_ti2v_t2v_tp2_ulysses2.mp4}
num_processes=${NUM_PROCESSES:-4}

# Use the normal rank-to-device mapping. This overrides the 0/1 swap used by
# the MiniMax-H3 physical-device diagnostic script.
export PLATFORM=intel_xpu
export LIGHTX2V_XPU_DEVICE_MAP=${LIGHTX2V_XPU_DEVICE_MAP:-0,1,2,3}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONPATH=${PYTHONPATH:-}

source "${lightx2v_path}/scripts/base/base.sh"
mkdir -p "$(dirname -- "${output_path}")"

torchrun --standalone --nproc_per_node="${num_processes}" -m lightx2v.infer \
  --model_cls wan2.2 \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${PROMPT:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage}" \
  --negative_prompt "${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，低质量，JPEG压缩残留，畸形，多余的手指，杂乱的背景}" \
  --seed "${SEED:-42}" \
  --save_result_path "${output_path}"
```

Config:
```
{
    "infer_steps": 50,
    "text_len": 512,
    "target_video_length": 120,
    "target_height": 704,
    "target_width": 1280,
    "cpu_offload": true,
    "offload_granularity": "phase",
    "t5_cpu_offload": true,
    "vae_cpu_offload": true,
    "num_channels_latents": 48,
    "vae_stride": [
        4,
        16,
        16
    ],
    "self_attn_1_type": "torch_sdpa",
    "cross_attn_1_type": "torch_sdpa",
    "cross_attn_2_type": "torch_sdpa",
    "sample_guide_scale": 5.0,
    "sample_shift": 5.0,
    "enable_cfg": true,
    "fps": 24,
    "rope_type": "torch_real_rope",
    "parallel": {
        "seq_p_size": 2,
        "seq_p_attn_type": "ulysses",
	"tensor_p_size": 2,
        "cfg_p_size": 1
    }
}

```

### Expected Result
A normal video

### Actual Result
A black video

### Environment Information
- Operating System: [Ubuntu 24.04.4]
- Python Env: [python 3.12.3, torch 2.10.0+xpu, torchaudio 2.10.0+xpu, torchvision 0.25.0+xpu]
- Commit ID: [cb8d588864d9132b593c3cb96faf35b1f6b83b5a] + https://github.com/qiuxin2012/LightX2V/commit/877a73ee36558ad49b1a528be2d2920d78fdcb67

### Log Information

### Additional Information


