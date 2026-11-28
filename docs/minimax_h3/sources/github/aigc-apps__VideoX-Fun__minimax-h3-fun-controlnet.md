# aigc-apps__VideoX-Fun__minimax-h3-fun-controlnet

Source: https://github.com/aigc-apps/VideoX-Fun
Fetched: 2026-11-28



---

## FILE: scripts/minimax_h3_fun/README_TRAIN.md (https://github.com/aigc-apps/VideoX-Fun/blob/HEAD/scripts/minimax_h3_fun/README_TRAIN.md)

# MiniMax-H3 Control Training Guide

This document provides a complete workflow for MiniMax-H3 control (VACE-style controllable video generation) training, including environment setup, data preparation, various distributed training strategies, CFG distillation, and inference testing.

> **Note**: MiniMax-H3 is an audio-visual generative video model that simultaneously generates video and corresponding audio. Control training adds a zero-initialised control side branch (`MiniMaxH3ControlTransformer3DModel`) on top of the base packed-sequence transformer: a paired control video (pose / depth / canny ...) is patchified and injected as per-layer skips, so a freshly initialised model is numerically identical to the base MiniMax-H3 model and only the control parameters (`--trainable_modules control`) need training. The joint video + audio flow-matching loss of `scripts/minimax_h3/train.py` is kept, so the training data requires video, control video **and** audio files.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Control Branch Training](#3-control-branch-training)
  - [3.1 Download Pre-trained Model](#31-download-pre-trained-model)
  - [3.2 Control Branch YAML Configuration](#32-control-branch-yaml-configuration)
  - [3.3 Quick Start (FSDP)](#33-quick-start-fsdp)
  - [3.4 Control-specific Parameter Reference](#34-control-specific-parameter-reference)
  - [3.5 Training Validation](#35-training-validation)
  - [3.6 Training with FSDP2](#36-training-with-fsdp2)
  - [3.7 Other Backends](#37-other-backends)
    - [3.7.1 Training with DeepSpeed-Zero-2](#371-training-with-deepspeed-zero-2)
    - [3.7.2 Training without DeepSpeed or FSDP](#372-training-without-deepspeed-or-fsdp)
  - [3.8 Multi-node Distributed Training](#38-multi-node-distributed-training)
  - [3.9 CFG Distillation of the Control Branch](#39-cfg-distillation-of-the-control-branch)
  - [3.10 Extracting Control-only Weights](#310-extracting-control-only-weights)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameter Reference](#41-inference-parameter-reference)
  - [4.2 V2V Control Inference](#42-v2v-control-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. More Resources](#5-more-resources)

---

## 1. Environment Setup

**Option 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Option 2: Manual Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**Option 3: Using Docker**

When using Docker, please ensure that the GPU driver and CUDA environment are correctly installed on the machine, then execute the following commands:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset with control signals containing some training data.

```bash
# Download the official example dataset (with control signals)
modelscope download --dataset PAI/X-Fun-Videos-Controls-Demo --local_dir ./datasets/X-Fun-Videos-Controls-Demo
```

After downloading, the dataset contains the following metadata files:
- `metadata.json`: basic format (control video paths only)
- `metadata_add_width_height.json`: with width/height info
- `metadata_add_width_height_add_wav.json`: with width/height info + audio paths (recommended for MiniMax-H3 control training)

> 💡 The demo's control signal lives in `canny/` and the paired audio tracks live in `wav/` (referenced by `metadata_add_width_height_add_wav.json`); metadata files without `audio_path` still work, since the training dataset decodes the audio track from each video container in that case.

### 2.2 Dataset Structure

Control training datasets require original videos with corresponding control signal videos (e.g., pose videos, depth videos, canny edge videos, etc.) **and** paired audio tracks, since MiniMax-H3 keeps the joint video + audio training loss.

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/                    # Original training videos
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 control/                  # Control signal videos (e.g., pose / depth / canny)
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 wav/                      # Paired audio tracks
│   │   ├── 📄 audio001.wav
│   │   ├── 📄 audio002.wav
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

> **Note**:
> - `train/` directory stores original videos
> - `control/` (or `pose/`, `depth/`, `canny/`, etc.) directory stores control signal videos that correspond one-to-one with the original videos. The directory name is customizable, as long as `control_file_path` in `metadata.json` correctly points to it
> - `wav/` directory stores the paired waveforms; they are resampled to the audio VAE's sample rate (32 kHz) during training

### 2.3 metadata.json Format

> ⚠️ **Important**: unlike regular video training, control training of MiniMax-H3 requires **the `control_file_path` field** in `metadata.json` — `VideoSpeechControlDataset` reads it on top of the usual ones. The `audio_path` field is optional; when it is absent, the audio track is decoded from the video container itself.

**Relative path format** (example):
```json
[
  {
    "file_path": "train/video001.mp4",
    "control_file_path": "control/video001.mp4",
    "audio_path": "wav/audio001.wav",
    "text": "A brown dog barks on a sofa, sitting on a light-colored couch in a cozy room",
    "type": "video",
    "width": 768,
    "height": 512
  }
]
```

**Absolute path format**:
```json
[
  {
    "file_path": "/mnt/data/videos/dog.mp4",
    "control_file_path": "/mnt/data/control/dog.mp4",
    "audio_path": "/mnt/data/wavs/dog.wav",
    "text": "A brown dog barks on a sofa",
    "type": "video",
    "width": 768,
    "height": 512
  }
]
```

**Key field descriptions**:
- `file_path`: Original video path (relative or absolute)
- `control_file_path`: Control signal video path (**required for control training**)
- `audio_path`: Audio file path (**MiniMax-H3 specific, optional**). Audio files are typically in `.wav` format; the path should correspond to `file_path`. When it is absent, the audio track is decoded from the video container itself
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `width` / `height`: Video dimensions (**recommended to provide**, used for bucket training. If not provided, they will be read automatically during training, which may affect training speed when data is stored on slower systems like OSS).
  - Use `scripts/process_json_add_width_and_height.py` to extract width and height for JSON files without these fields, supporting both images and videos.

### 2.4 Relative vs Absolute Path Usage

**Relative paths**:

If data paths are relative, set in the training script:

```bash
export DATASET_NAME="datasets/my_dataset/"
export DATASET_META_NAME="datasets/my_dataset/metadata.json"
```

**Absolute paths**:

If data paths are absolute, set in the training script:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **Recommendation**: Use relative paths for small local datasets; use absolute paths for external storage (NAS, OSS) or shared storage across multiple machines.

---

## 3. Control Branch Training

### 3.1 Download Pre-trained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download MiniMax-H3 official weights
hf download MiniMaxAI/MiniMax-H3 --local-dir models/Diffusion_Transformer/MiniMax-H3

# Download the pretrained control branch (Controlnet-Union) weights
modelscope download --model PAI/MiniMax-H3-Fun-Controlnet-Union --local_dir models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union
```

> 💡 The loader accepts either the converted diffusers layout above or an *original* MiniMax-H3 partition (e.g. `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, with no intermediate copy on disk. The control branch is **not** part of the base MiniMax-H3 weights: `from_pretrained` fills it in, every control block is initialised from the main block it is attached to and `control_proj_in` from `proj_in`, with `before_proj` / `after_proj` zeroed, so a freshly loaded model is numerically identical to the base MiniMax-H3 model. To train or infer with a control signal, load the released control checkpoint `models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors` through `--transformer_path` (training hot start) or `transformer_path` (inference).

### 3.2 Control Branch YAML Configuration

The control branch layout is driven by a YAML config passed through `--config_path`, whose `transformer_additional_kwargs` is spread into `from_pretrained` the same way at training and inference time:

```yaml
# config/minimax_h3/minimax_h3_control.yaml (inpaint variant)
format: diffusers
pipeline: minimax-h3
transformer_additional_kwargs:
    control_blocks_places: [0, 10, 20, 30, 40]
    control_in_dim: 49
```

| Key | Description | Default |
|-----|-------------|---------|
| `control_blocks_places` | The transformer layer indices the zero-initialised control blocks attach to | - |
| `control_in_dim` | The channels the control rows carry. `49` = control latent (24) + visibility map (1) + masked-video latents (24), used with `--enable_inpaint`; use `24` for a mask-less branch (`config/minimax_h3/minimax_h3_control_only.yaml`) | - |
| `control_apply_audio` | Whether the control skips reach the audio rows. `False` zeros the audio rows out of every skip before injection, so the control video guides the video rows alone while the soundtrack stays on the base model's path | `True` (default) |

> ⚠️ **Important**: `control_in_dim` must be pinned by the YAML — it is never inferred in code. `control_in_dim=49` is only valid together with `--enable_inpaint`; without the flag use `control_in_dim=24`. A checkpoint trained with one layout cannot be loaded into a model built with the other (`control_proj_in.weight` shape mismatch), and inference must load the **same** YAML the checkpoint was trained with.

### 3.3 Quick Start (FSDP)

FSDP is recommended for training MiniMax-H3 control: the transformer alone is about 62 GB in bfloat16 and the Qwen3-VL conditioner is another 62 GB, so the model weights must be sharded across GPUs — which FSDP (`FULL_SHARD`) does but DeepSpeed-Zero-2 does not. Only the control branch is trainable (`--trainable_modules control`), the rest of the model stays frozen.

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
      scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=311 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_control_inpaint" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --enable_inpaint \
  --low_vram \
  --validation_paths "asset/pose.mp4" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --uniform_sampling \
  --trainable_modules "control" \
  --resume_from_checkpoint=latest
```

> 💡 The launch script `scripts/minimax_h3_fun/train_control.sh` can be used as a starting point. `VIDEOX_OFFLOAD_VACE_LATENTS=True` offloads the encoded control/video latents to CPU between steps.

### 3.4 Control-specific Parameter Reference

**Key Control Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Control branch YAML config (pins `control_blocks_places` / `control_in_dim`, see 3.2) | `config/minimax_h3/minimax_h3_control.yaml` |
| `--enable_inpaint` | Feed a random inpaint mask through the control branch alongside the control video: the control rows carry the visibility map + masked-video latents on top of `in_channels`. The YAML must pin the matching widened `control_in_dim` (49) | - |
| `--trainable_modules` | Trainable modules (`control` trains only the control side branch, the default recipe) | `"control"` |
| `--trainable_modules_low_learning_rate` | Modules trained with lower learning rate | `[]` |
| `--transformer_path` | Load transformer weights from another checkpoint (e.g. a hot start) | None |
| `--pretrained_model_name_or_path` | Pre-trained MiniMax-H3 model path | `models/Diffusion_Transformer/MiniMax-H3` |
| `--train_data_dir` | Training data directory (empty when the meta uses absolute paths) | `""` |
| `--train_data_meta` | Training data metadata file | `/mnt/data/datasets/my_dataset/metadata.json` |
| `--train_batch_size` | Batch size per device | 1 |
| `--video_sample_size` | Max video training resolution | 960 |
| `--token_sample_size` | Resolution corresponding to the max token length when `training_with_video_token_length` is on | 960 |
| `--video_sample_stride` | Frame sampling stride (MiniMax-H3 is 24 fps) | 1 |
| `--video_sample_n_frames` | Number of frames to sample, must follow the `17*n+5` form of the video VAE (duration stays between 0 and 16 seconds) | 311 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--dataloader_num_workers` | DataLoader subprocess count | 8 |
| `--num_train_epochs` | Training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate (recommended for control branch training) | 2e-05 |
| `--lr_scheduler` | LR scheduler: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup` | `constant_with_warmup` |
| `--lr_warmup_steps` | LR warmup steps | 100 |
| `--seed` | Random seed for reproducible training | 42 |
| `--output_dir` | Output directory | `output_dir_minimax_h3_control_inpaint` |
| `--gradient_checkpointing` | Enable gradient checkpointing to save memory | - |
| `--gradient_checkpointing_save_on_cpu` | Offload the activations saved for backward of the transformer blocks to CPU memory | - |
| `--mixed_precision` | Mixed precision: `no`, `fp16`, `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | VAE encoding mini batch size | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training, no cropping of videos, train by resolution grouping | - |
| `--random_hw_adapt` | Auto-scale videos to random sizes within `[min_size, max_size]` range | - |
| `--training_with_video_token_length` | Train by token length for arbitrary resolution | - |
| `--uniform_sampling` | Uniform timestep sampling (recommended) | - |
| `--low_vram` | Keep VAE and conditioner on CPU, move to GPU only while encoding | - |
| `--offload_every_step` | Move the transformer through CPU between steps (cards far below 62 GB) | - |
| `--resume_from_checkpoint` | Resume training path, use `"latest"` for auto-selection | `latest` |
| `--validation_steps` | Validate every N steps | 100 |
| `--validation_epochs` | Validate every N epochs | 500 |
| `--validation_prompts` | Validation prompts | `"1girl, ..."` |
| `--validation_paths` | Validation control video paths | `"asset/pose.mp4"` |
| `--validation_sampling_steps` | Number of denoising steps of the validation sampling loop | 50 |
| `--use_8bit_adam` | Use 8-bit Adam optimizer to save memory | - |

**Training recipe notes**:
- 10% of the batches zero the control latents, keeping the unconditional path trainable (CFG)
- The audio stream keeps the joint video + audio flow-matching loss of `scripts/minimax_h3/train.py`
- Checkpoints are serialized in the diffusers layout: `<output_dir>/checkpoint-x/transformer/diffusion_pytorch_model.safetensors` plus `config.json`, so the predict scripts load them with `from_pretrained(..., subfolder="transformer")`

### 3.5 Training Validation

You can configure validation parameters to periodically generate test videos during training to monitor training progress and model quality.

**Validation Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--validation_steps` | Validate every N steps | 2000 |
| `--validation_epochs` | Validate every N epochs | 5 |
| `--validation_prompts` | Prompts for validation video generation | None |
| `--validation_paths` | Control video paths for validation | None |
| `--validation_sampling_steps` | Denoising steps of the validation sampling loop | 50 |

**Validation Example**:

```bash
  --validation_paths "asset/pose.mp4" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"
```

**Notes**:
- Validation videos are saved to the `output_dir/sample/` directory
- Multi-prompt format: `--validation_prompts "prompt1" "prompt2" "prompt3"`
- `validation_paths` should correspond one-to-one with `validation_prompts`, pointing to control video files

### 3.6 Training with FSDP2

The same `train_control.py` runs under FSDP2 by adding `--fsdp_version 2` to the accelerate launch command. Things to know before choosing it:

- `--fsdp_sharding_strategy`, `--fsdp_backward_prefetch` and `--fsdp_use_orig_params` are FSDP1-only; under FSDP2 the reshard behaviour is `--fsdp_reshard_after_forward`, which must be a boolean
- FSDP2 reshards a fully frozen unit through an autograd function on the unit's forward inputs, so the frozen main stack is released during the backward pass without the workarounds FSDP1 needs
- accelerate's FSDP2 path upcasts the model to float32 whenever mixed precision is on, so the frozen base keeps fp32 master weights: expect roughly twice the sharded parameter memory of the FSDP1 run
- `fully_shard` moves a CPU-resident model wholesale to the GPU before sharding, so a 60GB+ model needs `--fsdp_cpu_ram_efficient_loading True` (meta init + rank-0 broadcast) to avoid a per-rank VRAM OOM

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" \
    --use_fsdp --fsdp_version 2 \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_reshard_after_forward true \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_cpu_ram_efficient_loading True \
      scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # the same train_control.py arguments as the Quick Start
```

### 3.7 Other Backends

#### 3.7.1 Training with DeepSpeed-Zero-2

> ⚠️ **Warning**: DeepSpeed-Zero-2 only partitions optimizer states and gradients, **not the model weights**. The MiniMax-H3 transformer is about 62 GB, so each GPU still holds a full weight replica and this setup usually runs out of memory. Prefer FSDP (**3.3**) for MiniMax-H3 control training; the command below is provided for reference only.

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # the same train_control.py arguments as the Quick Start
```

#### 3.7.2 Training without DeepSpeed or FSDP

**This approach is not recommended as it lacks memory-saving backends and can easily cause out-of-memory issues**. With MiniMax-H3's ~62 GB transformer and ~62 GB Qwen3-VL conditioner replicated on every GPU, it will almost certainly run out of memory. This is provided for reference only.

```bash
accelerate launch --mixed_precision="bf16" scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # the same train_control.py arguments as the Quick Start
```

### 3.8 Multi-node Distributed Training

**Suitable for**: Ultra-large datasets, faster training speed

#### 3.8.1 Environment Configuration

Assuming 2 machines, each with 8 GPUs:

**Machine 0 (Master)**:
```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total machines
export NUM_PROCESS=16                # Total processes = machines x 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
      scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # the same train_control.py arguments as the Quick Start
```

**Machine 1 (Worker)**:
```bash
export RANK=1  # Note: this is 1
# Other environment variables identical to Machine 0

# Use the same accelerate launch command as Machine 0
```

#### 3.8.2 Multi-node Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Sync**: All machines must access the same data path (NFS/shared storage)

### 3.9 CFG Distillation of the Control Branch

`scripts/minimax_h3_fun/train_control_distill.py` distills classifier-free guidance into the control branch (mirroring `scripts/flux2_fun/train_control_distill.py`): a frozen teacher copy of the control transformer runs two forward passes per step — one on the prompt, one on the empty negative prompt — and the two predictions combine with `--real_guidance_scale` into the target the trainable student regresses onto with an MSE loss over the video and audio rows alike. Both copies load the same trained control branch, so point `--transformer_path` at it — either the whole transformer safetensors saved by `train_control.py` or the control-only file extracted in **3.10** (the loader is `strict=False` on missing keys).

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
      scripts/minimax_h3_fun/train_control_distill.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=311 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_control_distill" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --enable_inpaint \
  --uniform_sampling \
  --trainable_modules "control" \
  --transformer_path="output_dir_minimax_h3_control_inpaint/checkpoint-xxx/transformer/diffusion_pytorch_model_control.safetensors" \
  --real_guidance_scale=3.5 \
  --resume_from_checkpoint=latest
```

**Key distillation parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | The same control branch YAML the source checkpoint was trained with (see 3.2) | `config/minimax_h3/minimax_h3_control.yaml` |
| `--transformer_path` | Trained control branch checkpoint loaded by both the student and the frozen teacher; accepts the whole transformer safetensors or the control-only file of **3.10** | `output_dir_minimax_h3_control_inpaint/checkpoint-xxx/transformer/diffusion_pytorch_model_control.safetensors` |
| `--enable_inpaint` | Keep the flag of the source checkpoint: the distill recipe feeds the same mask channels through student and teacher | - |
| `--real_guidance_scale` | CFG scale of the teacher's two-pass score | 3.5 |
| `--learning_rate` | A lower learning rate than control training is recommended | 2e-06 |

> 💡 The launch script `scripts/minimax_h3_fun/train_control_distill.sh` can be used as a starting point. The released MiniMax-H3 checkpoint is guidance-distilled, so the student takes no guidance input; the guidance enters only through the teacher's two CFG forward passes. `--low_vram` of control training is supported here too, but the frozen teacher then moves through the GPU on every step, so the launch script leaves it off.

### 3.10 Extracting Control-only Weights

`train_control.py` saves the whole transformer (main branch + control branch). To ship only the control branch (`control_blocks.*` + `control_proj_in.*`) as a standalone safetensors file:

```bash
python scripts/minimax_h3_fun/extract_control_weights.py \
  --model_path output_dir_minimax_h3_control_inpaint/checkpoint-1000/transformer/diffusion_pytorch_model.safetensors \
  --output_path output_dir_minimax_h3_control_inpaint/checkpoint-1000/transformer/diffusion_pytorch_model_control.safetensors
```

> 💡 `--model_path` also accepts the `<checkpoint>/transformer` directory itself (every shard is merged, and `control_blocks_places` / `control_in_dim` are carried over from its `config.json` as safetensors metadata).

The extracted file can be re-applied onto a fresh base model with `MiniMaxH3ControlTransformer3DModel.materialize_missing_control_params(...)`, and is directly loadable through `--transformer_path` of `train_control.py` / `train_control_distill.py` and `transformer_path` of the predict scripts (`strict=False` on missing keys).

---

## 4. Inference Testing

### 4.1 Inference Parameter Reference

**Key Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `GPU_memory_mode` | GPU memory management mode, see table below | `model_group_offload` |
| `ulysses_degree` | Head dimension parallelism, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelism, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP on Transformer (main + control blocks) for multi-GPU inference | `False` |
| `fsdp_text_encoder` | Use FSDP on the Qwen3-VL text encoder (~62 GB); wraps the inner `text_encoder.model` decoder layers | `True` |
| `compile_dit` | Compile Transformer for faster inference (fixed resolution) | `False` |
| `model_name` | Base MiniMax-H3 model path | `models/Diffusion_Transformer/MiniMax-H3` |
| `config_path` | Control branch YAML, **must match the one `train_control.py` ran with** (`control_in_dim` 49 for an inpaint checkpoint). `None` builds the default 24-channel branch, which cannot load an inpaint checkpoint | `config/minimax_h3/minimax_h3_control.yaml` |
| `transformer_path` | Control checkpoint path: the released `MiniMax-H3-Fun-Controlnet-Union` weights or a trained control checkpoint. Hand it a `.safetensors` file (the whole transformer or the control-only file of **3.10** load too, `strict=False` on missing keys); a checkpoint's `transformer` folder is **not** accepted here | `models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors` |
| `vae_path` | Trained VAE weight path | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated video resolution `[height, width]`; height/width must be multiples of 32. Control inference fits the control video onto this canvas with the training's resize + crop geometry, so it **cannot be None** | `[704, 1280]` |
| `video_length` | Upper bound of the generation length, capped at the video VAE's `17*n+5` grid (duration stays under 15 seconds): generation follows the control video's actual length, snapped **down** to the largest valid `17*n+5` so a short control video is never padded | 243 |
| `fps` | Frames per second (MiniMax-H3 generates at a fixed 24 fps) | 24 |
| `control_context_scale` | Scale applied to every control skip before it is added to the main branch. 0.0 switches the control branch off; values below 1.0 weaken the control guidance | 1.00 |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 | `torch.bfloat16` |
| `control_video` | Control signal video path | `"asset/pose.mp4"` |
| `inpaint_video` | Source video behind the mask (only read by `--enable_inpaint` checkpoints); requires `inpaint_video_mask` | `None` |
| `inpaint_video_mask` | Mask video marking the regions to regenerate, binarized at 0.5 — white = repaint, black = keep (only for inpaint checkpoints) | `None` |
| `prompt` | Positive prompt describing content | `"视频中，一位年轻女性站在阳光洒满的沙滩上..."` |
| `negative_prompt` | Negative prompt, only used when `guidance_scale` > 1 | `"色调艳丽，过曝，静态..."` |
| `guidance_scale` | Guidance strength. The released checkpoint is guidance-distilled: keep it at 1 to run one forward pass per step with no CFG — the distill checkpoints of `train_control_distill.py` already bake the teacher's CFG target into the weights, so any value above 1 applies guidance twice and degrades the output | 1.0 |
| `num_inference_steps` | Number of denoising steps | 40 |
| `flow_shift` | Exponential sigma shift of the video schedule, `None` keeps the one of the checkpoint (12.0) | `None` |
| `audio_flow_shift` | Exponential sigma shift of the audio schedule, `None` keeps the one of the checkpoint (3.0) | `None` |
| `seed` | Random seed for reproducibility | 43 |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Generated video save path | `samples/minimax-h3-videos-v2v-control` |

**GPU Memory Management Modes**:

| Mode | Description | Memory Usage |
|------|-------------|--------------|
| `model_full_load` | Entire model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

> 💡 The transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner is another 62.1 GB, so a single 80 GB card needs `model_cpu_offload` or `model_group_offload`.

### 4.2 V2V Control Inference

Run single-GPU inference:

```bash
python examples/minimax_h3_fun/predict_v2v_control.py
```

Modify `examples/minimax_h3_fun/predict_v2v_control.py` as needed. For first-time inference, focus on the following parameters. For other parameters, see the Inference Parameter Reference above.

```python
# Choose based on GPU memory
GPU_memory_mode = "model_group_offload"
# Set to your actual model path
model_name = "models/Diffusion_Transformer/MiniMax-H3"
# Control branch layout: must match the yaml train_control.py ran with
# (minimax_h3_control.yaml for an --enable_inpaint checkpoint, minimax_h3_control_only.yaml otherwise)
config_path = "config/minimax_h3/minimax_h3_control.yaml"
# Control checkpoint (.safetensors file): the released MiniMax-H3-Fun-Controlnet-Union weights or a checkpoint
# trained by train_control.py (the control-only file of extract_control_weights.py loads too)
transformer_path = "models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors"
# Control signal video (e.g., pose video); control inference fits it onto sample_size
# with the training's resize + crop geometry
control_video = "asset/pose.mp4"
# Generated video resolution [height, width], multiples of 32, cannot be None here
sample_size = [704, 1280]
# Strength of the control guidance, 1.00 applies every control skip at full strength
control_context_scale = 1.00
# Inpaint inputs, only read by checkpoints trained with --enable_inpaint
inpaint_video = None
inpaint_video_mask = None
# Write according to generation content
prompt = "..."
# ...
```

**Notes**:
- The control branch is not part of the base MiniMax-H3 weights: point `transformer_path` at the released `MiniMax-H3-Fun-Controlnet-Union` checkpoint (or one trained by `train_control.py`) for the control video to take effect; a base `model_name` without `transformer_path` starts the side branch as an identity (`after_proj` is zero) and the control video has no effect
- Generation follows the control video's actual length, snapped down to the largest `17*n+5` the video VAE can decode and capped by `video_length` — a short control video is never padded (below 5 frames it is raised to 5). With an inpaint checkpoint but no inpaint inputs given, the pipeline zero-pads the mask channels and the run degrades to pure generation; a mask-less checkpoint rejects inpaint inputs outright
- The mask video is binarized at 0.5 before use, so grayscale masks work directly (white = repaint, black = keep)
- The generated output carries audio: videos are saved with `save_videos_with_audio_grid`

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, faster inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser yunchang
```

#### Configure Parallel Strategy

Edit `examples/minimax_h3_fun/predict_v2v_control.py`:

```python
# Ensure ulysses_degree x ring_degree = number of GPUs used
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallel
ring_degree = 1     # Sequence dimension parallel
```

**Configuration Principles**:
- `ulysses_degree` must divide the model's head count
- `ring_degree` splits along sequence dimension and affects communication overhead; avoid using if heads can divide evenly
- Multi-GPU runs through the xfuser sequence-parallel path and is **incompatible with the `*cpu_offload*` memory modes** (accelerate offload hooks own a single device); use `model_full_load` / `model_full_load_and_qfloat8` across GPUs there, with `fsdp_dit` / `fsdp_text_encoder` to save memory

**Configuration Examples**:

| GPUs | ulysses_degree | ring_degree | Note |
|------|----------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallel |
| 8 | 8 | 1 | Head parallel |
| 8 | 2 | 4 | Mixed parallel |

#### Run Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 examples/minimax_h3_fun/predict_v2v_control.py
```

---

## 5. More Resources

- **MiniMax-H3 Official GitHub**: https://github.com/MiniMax-AI/MiniMax-H3
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun


---

## FILE: scripts/minimax_h3/README_TRAIN.md (https://github.com/aigc-apps/VideoX-Fun/blob/HEAD/scripts/minimax_h3/README_TRAIN.md)

# MiniMax-H3 Full Parameter Training Guide

This document provides a complete workflow for full parameter training of MiniMax-H3, including environment configuration, data preparation, distributed training, and inference testing.

> **Note**: MiniMax-H3 is an audio-visual generative video model that can simultaneously generate video and corresponding audio. The training data requires both video and audio files.

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Full Parameter Training](#3-full-parameter-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (FSDP)](#32-quick-start-fsdp)
  - [3.3 Common Training Parameters](#33-common-training-parameters)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with DeepSpeed-Zero-2](#35-training-with-deepspeed-zero-2)
  - [3.6 Training Without DeepSpeed or FSDP](#36-training-without-deepspeed-or-fsdp)
  - [3.7 Multi-Machine Distributed Training](#37-multi-machine-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters](#41-inference-parameters)
  - [4.2 Single GPU Inference](#42-single-gpu-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. Additional Resources](#5-additional-resources)

---

## 1. Environment Configuration

**Method 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Method 2: Manual Dependency Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**Method 3: Using Docker**

When using Docker, please ensure that the GPU driver and CUDA environment are correctly installed on your machine, then execute the following commands:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset containing several audio-video training samples.

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Audios-Demo --local_dir ./datasets/X-Fun-Videos-Audios-Demo
```

### 2.2 Dataset Structure

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 wav/
│   │   ├── 📄 audio001.wav
│   │   ├── 📄 audio002.wav
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json Format

> ⚠️ **Important**: MiniMax-H3 is an audio-visual generative model. Unlike regular video training, you **must provide the `audio_path` field in metadata.json**. The paired waveform is resampled to the audio VAE's sample rate (32 kHz) during training.

**Relative Path Format** (example):
```json
[
  {
    "file_path": "train/video001.mp4",
    "audio_path": "wav/audio001.wav",
    "text": "A brown dog barks on a sofa, sitting on a light-colored couch in a cozy room",
    "type": "video",
    "width": 768,
    "height": 512
  },
  {
    "file_path": "train/video002.mp4",
    "audio_path": "wav/audio002.wav",
    "text": "A group of young men in suits and sunglasses are walking down a city street",
    "type": "video",
    "width": 640,
    "height": 640
  }
]
```

**Absolute Path Format**:
```json
[
  {
    "file_path": "/mnt/data/videos/dog.mp4",
    "audio_path": "/mnt/data/wavs/dog.wav",
    "text": "A brown dog barks on a sofa",
    "type": "video",
    "width": 768,
    "height": 512
  }
]
```

**Key Fields Description**:
- `file_path`: Video file path (relative or absolute)
- `audio_path`: Audio file path (**MiniMax-H3 specific and required**, main difference from regular video training)
  - Audio files are typically in `.wav` format
  - Path should correspond to `file_path`, e.g., `train/video001.mp4` corresponds to `wav/audio001.wav`
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `width` / `height`: Video dimensions (**recommended** to provide for bucket training; if not provided, they will be automatically read during training, which may slow down training when data is stored on slow systems like OSS)
  - You can use `scripts/process_json_add_width_and_height.py` to add width and height fields to JSON files without these fields, supporting both images and videos
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Audios-Demo/metadata.json --output_file datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json`

**Dataset Comparison: MiniMax-H3 vs Regular Video Training**:

| Model Type | Required Fields | Audio Field |
|---------|---------|---------|
| Regular Video (WAN, CogVideoX, etc.) | `file_path`, `text`, `type` | ❌ Not needed |
| **MiniMax-H3 (Audio-Visual Generation)** | `file_path`, `audio_path`, `text`, `type` | ✅ **Required** |

### 2.4 Relative vs Absolute Path Usage

**Relative Paths**:

If your data uses relative paths, configure the training script as follows:

```bash
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
```

**Absolute Paths**:

If your data uses absolute paths, configure the training script as follows:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata_add_width_height.json"
```

> 💡 **Recommendation**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. Full Parameter Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download MiniMax-H3 official weights
hf download MiniMaxAI/MiniMax-H3 --local-dir models/Diffusion_Transformer/MiniMax-H3
```

> 💡 The loader accepts either the converted diffusers layout above or an *original* MiniMax-H3 partition (e.g. `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, with no intermediate copy on disk.

### 3.2 Quick Start (FSDP)

If you have downloaded the data as per **2.1 Quick Test Dataset** and the weights as per **3.1 Download Pretrained Model**, you can directly copy and run the quick start command.

FSDP is recommended for training MiniMax-H3. The transformer alone is about 62 GB in bfloat16 and the Qwen3-VL conditioner is another 62 GB, so the model weights must be sharded across GPUs — which FSDP (`FULL_SHARD`) does but DeepSpeed-Zero-2 does not.

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2va"
```

### 3.3 Common Training Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--pretrained_model_name_or_path` | Path to pretrained model | `models/Diffusion_Transformer/MiniMax-H3` |
| `--train_data_dir` | Training data directory | `datasets/X-Fun-Videos-Audios-Demo/` |
| `--train_data_meta` | Training data metadata file | `datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json` |
| `--train_batch_size` | Samples per batch | 1 |
| `--video_sample_size` | Maximum video resolution for training | 960 |
| `--token_sample_size` | Token length sampling size | 960 |
| `--video_sample_stride` | Frame sampling stride (MiniMax-H3 is 24 fps) | 1 |
| `--video_sample_n_frames` | Number of frames to sample, must follow the `17*n+5` form of the video VAE (duration stays between 0 and 16 seconds) | 124 |
| `--video_repeat` | Number of times each video is repeated per epoch | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (equivalent to larger batch) | 1 |
| `--dataloader_num_workers` | DataLoader subprocesses | 4 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 500 |
| `--learning_rate` | Initial learning rate | 1e-5 |
| `--lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_minimax_h3` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--gradient_checkpointing_save_on_cpu` | Offload the activations saved for backward of the transformer blocks to CPU memory | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | Mini-batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 1.0 |
| `--random_hw_adapt` | Auto-scale videos to random size in range `[512, video_sample_size]` | - |
| `--training_with_video_token_length` | Train based on token length instead of fixed resolution | - |
| `--enable_bucket` | Enable bucket training: trains entire videos grouped by resolution without center cropping | - |
| `--uniform_sampling` | Uniform timestep sampling | - |
| `--low_vram` | Keep VAE and conditioner on CPU, move to GPU only while encoding | - |
| `--train_mode` | `t2v` (text only) or `fl2va` (first-frame keyframe conditioning, the keyframe taken from the training sample itself) | `fl2va` |
| `--t2v_ratio` | Under `--train_mode=fl2va`, the fraction of steps that drop the keyframe and train t2v instead, so one run keeps both conditionings. Must be in [0, 1] and only applies to fl2va; 0 trains fl2va only | 0.25 |
| `--resume_from_checkpoint` | Resume training from checkpoint path, use `"latest"` to auto-select latest | None |
| `--trainable_modules` | Trainable modules (`"."` means all modules) | `"."` |
| `--validation_steps` | Execute validation every N steps | 2000 |
| `--validation_epochs` | Execute validation every N epochs | 5 |
| `--validation_prompts` | Prompts used during validation | `"A man in a blue blazer..."` |


### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training, allowing you to monitor training progress and model quality.

**Validation Parameters**:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Execute validation every N steps | 2000 |
| `--validation_epochs` | Execute validation every N epochs | 5 |
| `--validation_prompts` | Prompt for validation video generation. Use multiple space-separated prompt strings | Space-separated prompt strings |

**Example**:

```bash
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks."
```

**Notes**:
- Validation videos will be saved to the `output_dir/sample/` directory
- For multi-prompt validation, use: `--validation_prompts "prompt1" "prompt2" "prompt3"`

### 3.5 Training with DeepSpeed-Zero-2

> ⚠️ **Warning**: DeepSpeed-Zero-2 only partitions optimizer states and gradients, **not the model weights**. The MiniMax-H3 transformer is about 62 GB, so each GPU still holds a full weight replica and this setup usually runs out of memory. Prefer FSDP (**3.2**) for MiniMax-H3; the command below is provided for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2va"
```

### 3.6 Training Without DeepSpeed or FSDP

**This approach is not recommended as it lacks VRAM-saving backends**. With MiniMax-H3's ~62 GB transformer and ~62 GB Qwen3-VL conditioner replicated on every GPU, it will almost certainly run out of memory. This is provided for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2va"
```

### 3.7 Multi-Machine Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2va"
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Same as Master
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # Note this is 1
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# Use the same accelerate launch command as Machine 0
```

#### 3.7.2 Multi-Machine Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Synchronization**: All machines must be able to access the same data paths (NFS/shared storage)

## 4. Inference Testing

### 4.1 Inference Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | GPU memory mode, see table below for options | `model_cpu_offload` |
| `ulysses_degree` | Head dimension parallelization degree, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelization degree, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer in multi-GPU inference to save VRAM | `False` |
| `fsdp_text_encoder` | Use FSDP for the Qwen3-VL text encoder in multi-GPU inference to save VRAM | `False` |
| `compile_dit` | Compile Transformer to accelerate inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/MiniMax-H3` |
| `transformer_path` | Path to trained Transformer weights | `None` |
| `vae_path` | Path to trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated video resolution `[height, width]`; height/width must be multiples of 32. `None` uses MiniMax-H3's own 16:9 canvas (768x1344) | `[704, 1280]` |
| `video_length` | Number of frames to generate, snapped up to the next `17*n+5` the video VAE can decode (duration stays between 0 and 16 seconds) | 124 |
| `fps` | Frames per second (MiniMax-H3 generates at a fixed 24 fps) | 24 |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `prompt` | Positive prompt describing the content to generate | `"A red fox trotting..."` |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Number of denoising steps, i.e. of model evaluations; the sigma grid it is built from holds one more point, the terminal 0 | 40 |
| `guidance_scale` | Guidance strength. The released checkpoint is guidance-distilled: keep it at 1 to run one forward pass per step with no CFG; a value above 1 enables classifier-free guidance with two passes | 1 |
| `flow_shift` | Exponential sigma shift of the video schedule, `None` keeps the one of the checkpoint (12.0) | `None` |
| `audio_flow_shift` | Exponential sigma shift of the audio schedule, `None` keeps the one of the checkpoint (3.0) | `None` |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Generated video save path | `samples/minimax-h3-videos-t2v` |

**GPU Memory Mode Description**:

| Mode | Description | VRAM Usage |
|------|------|---------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer group offload between CPU/CUDA | Low |
| `sequential_cpu_offload` | Offload each layer individually (slowest) | Lowest |

> 💡 The transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner is another 62.1 GB, so a single 80 GB card needs `model_cpu_offload` or `model_group_offload`.

### 4.2 Single GPU Inference

Run single GPU inference with:

```bash
python examples/minimax_h3/predict_t2v.py
```

Edit `examples/minimax_h3/predict_t2v.py` according to your needs. For first-time inference, focus on these parameters. For other parameters, see the Inference Parameters section above.

```python
# Choose based on your GPU VRAM
GPU_memory_mode = "model_cpu_offload"
# Your actual model path
model_name = "models/Diffusion_Transformer/MiniMax-H3"  
# Trained weights path, e.g. "output_dir_minimax_h3/checkpoint-xxx/transformer"
transformer_path = None  
# Write based on content to generate
prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"  
# ...
```

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser yunchang
```

#### Configure Parallel Strategy

Edit `examples/minimax_h3/predict_t2v.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelization
ring_degree = 1     # Sequence dimension parallelization
```

**Configuration Principles**:
- `ulysses_degree` must evenly divide the model's number of heads
- `ring_degree` splits on sequence dimension, affecting communication overhead; avoid using it when heads can be divided
- Multi-GPU runs through the xfuser sequence-parallel path and is **incompatible with the `*cpu_offload*` memory modes** (accelerate offload hooks own a single device); use `model_full_load` / `model_full_load_and_qfloat8` across GPUs there, with `fsdp_dit` / `fsdp_text_encoder` to save memory

**Example Configurations**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelization |
| 8 | 2 | 4 | Hybrid parallelization |
| 8 | 8 | 1 | Head parallelization |

#### Run Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 examples/minimax_h3/predict_t2v.py
```

## 5. Additional Resources

- **MiniMax-H3 Official GitHub**: https://github.com/MiniMax-AI/MiniMax-H3
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun


---

## FILE: scripts/minimax_h3/README_TRAIN_LORA.md (https://github.com/aigc-apps/VideoX-Fun/blob/HEAD/scripts/minimax_h3/README_TRAIN_LORA.md)

# MiniMax-H3 LoRA Training Guide

This document provides a complete workflow for LoRA (Low-Rank Adaptation) training of MiniMax-H3, including environment configuration, data preparation, distributed training, and inference testing.

> **Note**: MiniMax-H3 is an audio-visual generative video model that can simultaneously generate video and corresponding audio. The training data requires both video and audio files.

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. LoRA Training](#3-lora-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (FSDP)](#32-quick-start-fsdp)
  - [3.3 LoRA Training Parameters](#33-lora-training-parameters)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with DeepSpeed-Zero-2](#35-training-with-deepspeed-zero-2)
  - [3.6 Training Without DeepSpeed or FSDP](#36-training-without-deepspeed-or-fsdp)
  - [3.7 Multi-Machine Distributed Training](#37-multi-machine-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters](#41-inference-parameters)
  - [4.2 Single GPU Inference](#42-single-gpu-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. Additional Resources](#5-additional-resources)

---

## 1. Environment Configuration

**Method 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Method 2: Manual Dependency Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**Method 3: Using Docker**

When using Docker, please ensure that the GPU driver and CUDA environment are correctly installed on your machine, then execute the following commands:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset containing several audio-video training samples.

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Audios-Demo --local_dir ./datasets/X-Fun-Videos-Audios-Demo
```

### 2.2 Dataset Structure

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 wav/
│   │   ├── 📄 audio001.wav
│   │   ├── 📄 audio002.wav
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json Format

> ⚠️ **Important**: MiniMax-H3 is an audio-visual generative model. Unlike regular video training, you **must provide the `audio_path` field in metadata.json**. The paired waveform is resampled to the audio VAE's sample rate (32 kHz) during training.

**Relative Path Format** (example):
```json
[
  {
    "file_path": "train/video001.mp4",
    "audio_path": "wav/audio001.wav",
    "text": "A brown dog barks on a sofa, sitting on a light-colored couch in a cozy room",
    "type": "video",
    "width": 768,
    "height": 512
  },
  {
    "file_path": "train/video002.mp4",
    "audio_path": "wav/audio002.wav",
    "text": "A group of young men in suits and sunglasses are walking down a city street",
    "type": "video",
    "width": 640,
    "height": 640
  }
]
```

**Absolute Path Format**:
```json
[
  {
    "file_path": "/mnt/data/videos/dog.mp4",
    "audio_path": "/mnt/data/wavs/dog.wav",
    "text": "A brown dog barks on a sofa",
    "type": "video",
    "width": 768,
    "height": 512
  }
]
```

**Key Fields Description**:
- `file_path`: Video file path (relative or absolute)
- `audio_path`: Audio file path (**MiniMax-H3 specific and required**, main difference from regular video training)
  - Audio files are typically in `.wav` format
  - Path should correspond to `file_path`, e.g., `train/video001.mp4` corresponds to `wav/audio001.wav`
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `width` / `height`: Video dimensions (**recommended** to provide for bucket training; if not provided, they will be automatically read during training, which may slow down training when data is stored on slow systems like OSS)
  - You can use `scripts/process_json_add_width_and_height.py` to add width and height fields to JSON files without these fields, supporting both images and videos
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Audios-Demo/metadata.json --output_file datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json`

**Dataset Comparison: MiniMax-H3 vs Regular Video Training**:

| Model Type | Required Fields | Audio Field |
|---------|---------|---------|
| Regular Video (WAN, CogVideoX, etc.) | `file_path`, `text`, `type` | ❌ Not needed |
| **MiniMax-H3 (Audio-Visual Generation)** | `file_path`, `audio_path`, `text`, `type` | ✅ **Required** |

### 2.4 Relative vs Absolute Path Usage

**Relative Paths**:

If your data uses relative paths, configure the training script as follows:

```bash
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
```

**Absolute Paths**:

If your data uses absolute paths, configure the training script as follows:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata_add_width_height.json"
```

> 💡 **Recommendation**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. LoRA Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download MiniMax-H3 official weights
hf download MiniMaxAI/MiniMax-H3 --local-dir models/Diffusion_Transformer/MiniMax-H3
```

> 💡 The loader accepts either the converted diffusers layout above or an *original* MiniMax-H3 partition (e.g. `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, with no intermediate copy on disk.

### 3.2 Quick Start (FSDP)

If you have downloaded the data as per **2.1 Quick Test Dataset** and the weights as per **3.1 Download Pretrained Model**, you can directly copy and run the quick start command.

FSDP is recommended for training MiniMax-H3. Even with LoRA, the frozen transformer (about 62 GB in bfloat16) and the Qwen3-VL conditioner (another 62 GB) still need to be sharded across GPUs — which FSDP (`FULL_SHARD`) does but DeepSpeed-Zero-2 does not.

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/minimax_h3/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --rank=64 \
  --network_alpha=32 \
  --low_vram \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,proj_in,audio_proj_in,context_embedder" \
  --t2v_ratio=0.25 \
  --train_mode="fl2va"
```

### 3.3 LoRA Training Parameters

**LoRA-Specific Parameters**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--rank` | Dimension of LoRA update matrices | 64 |
| `--network_alpha` | Scale of LoRA update matrices | 32 |
| `--target_name` | Components/modules to apply LoRA (comma-separated) | `to_q,to_k,to_v,ff.0,ff.2,proj_in,audio_proj_in,context_embedder` |
| `--use_peft_lora` | Use PEFT module for adding LoRA (more memory-efficient, optional) | - |

**Common Training Parameters**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--pretrained_model_name_or_path` | Path to pretrained model | `models/Diffusion_Transformer/MiniMax-H3` |
| `--train_data_dir` | Training data directory | `datasets/X-Fun-Videos-Audios-Demo/` |
| `--train_data_meta` | Training data metadata file | `datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json` |
| `--train_batch_size` | Samples per batch | 1 |
| `--video_sample_size` | Maximum video resolution for training | 960 |
| `--fix_sample_size` | Fixed `[height, width]` canvas (both multiples of 32) overriding `--video_sample_size`; requires `--enable_bucket` and turns off `--random_hw_adapt` / `--training_with_video_token_length` | None |
| `--token_sample_size` | Token length sampling size | 960 |
| `--video_sample_stride` | Frame sampling stride (MiniMax-H3 is 24 fps) | 1 |
| `--video_sample_n_frames` | Number of frames to sample, must follow the `17*n+5` form of the video VAE (duration stays between 0 and 16 seconds) | 124 |
| `--video_repeat` | Number of times each video is repeated per epoch | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (equivalent to larger batch) | 1 |
| `--dataloader_num_workers` | DataLoader subprocesses | 4 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 500 |
| `--learning_rate` | Initial learning rate (LoRA typically uses higher LR than full training) | 1e-4 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_minimax_h3_lora` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--gradient_checkpointing_save_on_cpu` | Offload the activations saved for backward of the transformer blocks to CPU memory | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | Mini-batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 1.0 |
| `--random_hw_adapt` | Auto-scale videos to random size in range `[512, video_sample_size]` | - |
| `--training_with_video_token_length` | Train based on token length instead of fixed resolution | - |
| `--enable_bucket` | Enable bucket training: trains entire videos grouped by resolution without center cropping | - |
| `--uniform_sampling` | Uniform timestep sampling | - |
| `--low_vram` | Keep VAE and conditioner on CPU, move to GPU only while encoding | - |
| `--train_mode` | `t2v` (text only) or `fl2va` (first-frame keyframe conditioning, the keyframe taken from the training sample itself) | `fl2va` |
| `--t2v_ratio` | Under `--train_mode=fl2va`, the fraction of steps that drop the keyframe and train t2v instead, so one run keeps both conditionings. Must be in [0, 1] and only applies to fl2va; 0 trains fl2va only | 0.25 |
| `--resume_from_checkpoint` | Resume training from checkpoint path, use `"latest"` to auto-select latest | None |
| `--validation_steps` | Execute validation every N steps | 2000 |
| `--validation_epochs` | Execute validation every N epochs | 5 |
| `--validation_prompts` | Prompts used during validation | `"A man in a blue blazer..."` |


### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training, allowing you to monitor training progress and model quality.

**Validation Parameters**:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Execute validation every N steps | 2000 |
| `--validation_epochs` | Execute validation every N epochs | 5 |
| `--validation_prompts` | Prompt for validation video generation. Use multiple space-separated prompt strings | Space-separated prompt strings |

**Example**:

```bash
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks."
```

**Notes**:
- Validation videos will be saved to the `output_dir/sample/` directory
- For multi-prompt validation, use: `--validation_prompts "prompt1" "prompt2" "prompt3"`

### 3.5 Training with DeepSpeed-Zero-2

> ⚠️ **Warning**: DeepSpeed-Zero-2 only partitions optimizer states and gradients, **not the model weights**. The MiniMax-H3 transformer is about 62 GB, so each GPU still holds a full weight replica and this setup usually runs out of memory. Prefer FSDP (**3.2**) for MiniMax-H3; the command below is provided for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --rank=64 \
  --network_alpha=32 \
  --low_vram \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,proj_in,audio_proj_in,context_embedder" \
  --t2v_ratio=0.25 \
  --train_mode="fl2va"
```

### 3.6 Training Without DeepSpeed or FSDP

**This approach is not recommended as it lacks VRAM-saving backends**. With MiniMax-H3's ~62 GB transformer and ~62 GB Qwen3-VL conditioner replicated on every GPU, it will almost certainly run out of memory. This is provided for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/minimax_h3/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --rank=64 \
  --network_alpha=32 \
  --low_vram \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,proj_in,audio_proj_in,context_embedder" \
  --t2v_ratio=0.25 \
  --train_mode="fl2va"
```

### 3.7 Multi-Machine Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/minimax_h3/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --rank=64 \
  --network_alpha=32 \
  --low_vram \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,proj_in,audio_proj_in,context_embedder" \
  --t2v_ratio=0.25 \
  --train_mode="fl2va"
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Same as Master
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # Note this is 1
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# Use the same accelerate launch command as Machine 0
```

#### 3.7.2 Multi-Machine Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Synchronization**: All machines must be able to access the same data paths (NFS/shared storage)

## 4. Inference Testing

### 4.1 Inference Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | GPU memory mode, see table below for options | `model_cpu_offload` |
| `ulysses_degree` | Head dimension parallelization degree, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelization degree, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer in multi-GPU inference to save VRAM | `False` |
| `fsdp_text_encoder` | Use FSDP for the Qwen3-VL text encoder in multi-GPU inference to save VRAM | `False` |
| `compile_dit` | Compile Transformer to accelerate inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/MiniMax-H3` |
| `transformer_path` | Path to trained Transformer weights | `None` |
| `vae_path` | Path to trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `output_dir_minimax_h3_lora/checkpoint-xxx/lora_diffusion_pytorch_model.safetensors` |
| `sample_size` | Generated video resolution `[height, width]`; height/width must be multiples of 32. `None` uses MiniMax-H3's own 16:9 canvas (768x1344) | `[704, 1280]` |
| `video_length` | Number of frames to generate, snapped up to the next `17*n+5` the video VAE can decode (duration stays between 0 and 16 seconds) | 124 |
| `fps` | Frames per second (MiniMax-H3 generates at a fixed 24 fps) | 24 |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `prompt` | Positive prompt describing the content to generate | `"A red fox trotting..."` |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Number of denoising steps, i.e. of model evaluations; the sigma grid it is built from holds one more point, the terminal 0 | 40 |
| `guidance_scale` | Guidance strength. The released checkpoint is guidance-distilled: keep it at 1 to run one forward pass per step with no CFG; a value above 1 enables classifier-free guidance with two passes | 1 |
| `flow_shift` | Exponential sigma shift of the video schedule, `None` keeps the one of the checkpoint (12.0) | `None` |
| `audio_flow_shift` | Exponential sigma shift of the audio schedule, `None` keeps the one of the checkpoint (3.0) | `None` |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Generated video save path | `samples/minimax-h3-videos-t2v` |

**GPU Memory Mode Description**:

| Mode | Description | VRAM Usage |
|------|------|---------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer group offload between CPU/CUDA | Low |
| `sequential_cpu_offload` | Offload each layer individually (slowest) | Lowest |

> 💡 The transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner is another 62.1 GB, so a single 80 GB card needs `model_cpu_offload` or `model_group_offload`.

### 4.2 Single GPU Inference

Run single GPU inference with:

```bash
python examples/minimax_h3/predict_t2v.py
```

Edit `examples/minimax_h3/predict_t2v.py` according to your needs. For LoRA inference, focus on these parameters:

```python
# Choose based on your GPU VRAM
GPU_memory_mode = "model_cpu_offload"
# Your actual model path
model_name = "models/Diffusion_Transformer/MiniMax-H3"  
# LoRA weights path, e.g. "output_dir_minimax_h3_lora/checkpoint-xxx/lora_diffusion_pytorch_model.safetensors"
lora_path = None
# LoRA weight strength
lora_weight = 0.55  
# Write based on content to generate
prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"  
# ...
```

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser yunchang
```

#### Configure Parallel Strategy

Edit `examples/minimax_h3/predict_t2v.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelization
ring_degree = 1     # Sequence dimension parallelization
```

**Configuration Principles**:
- `ulysses_degree` must evenly divide the model's number of heads
- `ring_degree` splits on sequence dimension, affecting communication overhead; avoid using it when heads can be divided
- Multi-GPU runs through the xfuser sequence-parallel path and is **incompatible with the `*cpu_offload*` memory modes** (accelerate offload hooks own a single device); use `model_full_load` / `model_full_load_and_qfloat8` across GPUs there, with `fsdp_dit` / `fsdp_text_encoder` to save memory

**Example Configurations**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelization |
| 8 | 2 | 4 | Hybrid parallelization |
| 8 | 8 | 1 | Head parallelization |

#### Run Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 examples/minimax_h3/predict_t2v.py
```

## 5. Additional Resources

- **MiniMax-H3 PDD LoRA training**: `scripts/minimax_h3/README_TRAIN_PDD_LORA.md`
- **MiniMax-H3 Official GitHub**: https://github.com/MiniMax-AI/MiniMax-H3
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun


---

## FILE: scripts/minimax_h3/README_TRAIN_PDD_LORA.md (https://github.com/aigc-apps/VideoX-Fun/blob/HEAD/scripts/minimax_h3/README_TRAIN_PDD_LORA.md)

# MiniMax-H3 PDD LoRA Training Guide

This document provides a complete workflow for Parallel Decoding Distillation (PDD, [arXiv 2607.26004](https://arxiv.org/abs/2607.26004)) LoRA training of MiniMax-H3, including environment configuration, conditioning-cache preparation, distributed training, and inference testing.

> **Note**: MiniMax-H3 is an audio-visual generative video model that can simultaneously generate video and corresponding audio. PDD training is **data-free**: it never reads target videos. Each rank carries one trajectory, rolls it forward with the student's own predictions, and is supervised by a frozen teacher on the same backbone. Only cached Qwen3-VL conditioning is needed, which keeps the ~62 GB text encoder out of the training run.

PDD turns the pre-trained flow model into a *parallel decoder*. The sampling interval is discretized into `N` intervals grouped into blocks of size `L`; one network evaluation predicts the mean velocity of every interval of the next block, so generation advances `L` intervals per evaluation (`NFE = N / L`). The default recipe is `N = 32`, `L = 4` (8 NFE). The student is the teacher's own transformer with the two final heads (`proj_out` / `audio_proj_out`) repeated `N` times; switching LoRA off is still the teacher, so there is no second copy of the 33 B backbone.

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Data-free Conditioning](#21-data-free-conditioning)
  - [2.2 Cache Structure](#22-cache-structure)
  - [2.3 Generating the Cache](#23-generating-the-cache)
  - [2.4 Annotation JSON Format](#24-annotation-json-format)
  - [2.5 Ref2VA Request Cache](#25-ref2va-request-cache)
- [3. PDD LoRA Training](#3-pdd-lora-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (FSDP)](#32-quick-start-fsdp)
    - [3.2.1 Ref2VA Training](#321-ref2va-training)
  - [3.3 PDD Training Parameters](#33-pdd-training-parameters)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Checkpoint Layout](#35-checkpoint-layout)
  - [3.6 Training with DeepSpeed-Zero-2](#36-training-with-deepspeed-zero-2)
  - [3.7 Training Without DeepSpeed or FSDP](#37-training-without-deepspeed-or-fsdp)
  - [3.8 Multi-Machine Distributed Training](#38-multi-machine-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters](#41-inference-parameters)
  - [4.2 Single GPU Inference](#42-single-gpu-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. Additional Resources](#5-additional-resources)

---

## 1. Environment Configuration

**Method 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Method 2: Manual Dependency Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**Method 3: Using Docker**

When using Docker, please ensure that the GPU driver and CUDA environment are correctly installed on your machine, then execute the following commands:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

PDD is **data-free**: the student trajectory is sampled from noise, so no target video/audio media is ever read for the loss. Training consumes only *conditioning* — the prompt for `fl2va`, and the prompt plus reference media for `ref2va`. Both `--train_mode`s support both conditioning routes, selected by `--enable_preprocess_training`:

| `--train_mode` | Conditions on | **Cache route** (`--enable_preprocess_training`) | **Direct-load route** (flag omitted) |
|----------------|---------------|--------------------------------------------------|--------------------------------------|
| `fl2va` | prompt | Pre-encode with `generate_prompt_cache.py`, read the `outputs.json` | Read a `{"text": ...}` annotation (`TextDataset`), encode on the fly |
| `ref2va` | prompt + reference media | Pre-encode with `generate_ref2va_request_cache.py`, read the `outputs.json` | Read a request annotation (`load_requests`), encode the prompt + reference latents on the fly |

- **Cache route** (recommended for long / repeated runs): the Qwen3-VL embeddings — and, for `ref2va`, the VAE-encoded reference latents — are pre-encoded to safetensors **once**, so the ~62 GB conditioner never loads during training.
- **Direct-load route** (recommended to start, or for a small request set): there is no separate preprocessing step; the run loads the ~62 GB conditioner (and, for `ref2va`, the two VAEs) and encodes each entry on the fly. Keep `--low_vram` on so they move onto the GPU only while encoding.

> 💡 Either route feeds **both** the train trajectories and the validation renders — data-free PDD has no train/val split, so `--train_data_meta` and `--val_data_meta` usually point at the same annotation (e.g. the official `datasets/X-Fun-Videos-Audios-Demo`, the standard test data — never an ad-hoc prompt set).

### 2.1 Data-free Conditioning

`--train_mode=fl2va` (the default recipe, FL2VA / t2va packed layout) needs the prompt conditioning; `--train_mode=ref2va` additionally needs reference media (images / videos / audio) and loads `transformer_ref` by default. On the **cache route**, generate the cache **once** with `scripts/minimax_h3/generate_prompt_cache.sh` (`fl2va`) or `scripts/minimax_h3/generate_ref2va_request_cache.sh` (`ref2va`); both run multi-GPU under `accelerate launch`, and the ~62 GB Qwen3-VL conditioner is then not loaded during PDD training. On the **direct-load route** there is no preprocessing step: point `--train_data_meta` straight at the annotation and the run encodes it on the fly (see **3.2.1** for both `ref2va` routes end to end).

### 2.2 Cache Structure

```
📦 datasets/
├── 📂 X-Fun-Videos-Audios-Demo/          # official demo dataset (source of the `text` captions)
│   └── 📄 metadata_add_width_height.json
└── 📂 minimax_h3_pdd_prompt_cache/       # generated once; feeds both train and validation
    ├── 📄 outputs.json
    ├── 📄 00000.safetensors
    ├── 📄 00001.safetensors
    └── 📄 ...
```

`outputs.json` is a list of `{"file_path": ".../00000.safetensors"}` records that `ImageVideoSafetensorsDataset` reads. `--train_data_dir` is the optional root prepended to each `file_path`; leave it empty when `outputs.json` already stores repo-relative (or absolute) paths, as the generators do. Each `fl2va` `.safetensors` holds:

| Field | Description |
|-------|-------------|
| `prompt_embeds` | Qwen3-VL hidden states at the MiniMax-H3 text-encoder layer (bfloat16) |
| `text_token_tags` | Per-token tags for the packed sequence (int64) |

### 2.3 Generating the Cache

This step is only for the **cache route**; the direct-load route (**3.2.1 Route A**) reads the annotation directly and skips it.

```bash
# fl2va: cache the prompt conditioning of the official demo dataset (multi-GPU)
bash scripts/minimax_h3/generate_prompt_cache.sh

# ref2va: cache the request conditioning (prompt embeds + reference latents)
bash scripts/minimax_h3/generate_ref2va_request_cache.sh
```

Each launcher runs `accelerate launch ... generate_*_cache.py` once: every rank walks an interleaved slice of the annotation, `.safetensors` that already exist are skipped (resume), and rank0 finally writes `outputs.json`. Edit the `MODEL_NAME` / `DATASET_META` / `CACHE_ROOT` variables at the top of each `.sh` first.

> 💡 `--pretrained_model_name_or_path` may be either the converted diffusers layout or an original MiniMax-H3 partition (e.g. `MiniMax-H3/FL2VA`). The tokenizer is read from `tokenizer/`, the processor from `processor/`, and the conditioner from `text_encoder/`.

### 2.4 Annotation JSON Format

The `fl2va` generator reads the official demo dataset's annotation JSON — a list of records whose `text` caption is the only field PDD uses (the `file_path` / `audio_path` / `control_file_path` / `width` / `height` fields the audio-visual dataset carries are ignored). A bare list of strings, or `{"prompt": ...}` / `{"examples": [...]}` jobs, are accepted too:

```json
[
  {"file_path": "train/00000001.mp4", "text": "A young woman gently turns her head to the right ...", "audio_path": "wav/00000001.wav", "type": "video"}
]
```

The `ref2va` route — both `generate_ref2va_request_cache.py` (cache) and direct-load training — reads a list of request records instead. Either an explicit `{"prompt": ..., "references": [...]}` request, where each reference is an `"image=..."` / `"video=..."` / `"audio=..."` string (the `predict_ref2va.py` schema, in the order the model reads them):

```json
[
  {"prompt": "The character turns and waves", "references": ["image=ref/face.png", "audio=ref/voice.wav"]}
]
```

or — the default — the *same* audio-visual demo annotation as `fl2va` above: `load_requests` derives the `video=<file_path>` + `audio=<audio_path>` references from each record (relative media paths resolve against the annotation's directory) and uses its `text` as the prompt.

Only the conditioning is encoded. Resolution and duration are training/inference flags (`--video_sample_size` / `--fix_sample_size` / `--video_sample_n_frames`), not cache fields.

### 2.5 Ref2VA Request Cache

`--train_mode=ref2va` loads `transformer_ref` by default. On the **cache route** it reads the request cache written by `generate_ref2va_request_cache.py`: besides `prompt_embeds` / `text_token_tags`, each request `.safetensors` also carries the reference latents for the Ref2VA packed layout, flattened into tensors — `reference_kind_ids` / `reference_has_audio` (per-reference kind and has-audio flag), `num_condition_latents` / `num_audio_condition_latents`, and the indexed `condition_latents_{i}` / `audio_condition_latents_{i}`. On the **direct-load route** the very same latents are produced on the fly instead — `train_pdd_lora.py` reads the request annotation with `load_requests`, encodes the prompt with Qwen3-VL and the references with the two VAEs — so no `.safetensors` cache is needed (`--train_mode=ref2va` without `--enable_preprocess_training`).

---

## 3. PDD LoRA Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download MiniMax-H3 official weights
hf download MiniMax-AI/MiniMax-H3 --local-dir models/Diffusion_Transformer/MiniMax-H3
```

> 💡 The loader accepts either the converted diffusers layout above or an *original* MiniMax-H3 partition (e.g. `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, with no intermediate copy on disk.

### 3.2 Quick Start (FSDP)

If you have generated a conditioning cache as per **2.3** and downloaded the weights as per **3.1**, you can copy and run the command below. `scripts/minimax_h3/train_pdd_lora.sh` is the same launch.

FSDP is recommended: even though PDD does not load Qwen3-VL, the frozen transformer is still about 62 GB in bfloat16 and must be sharded — which FSDP (`FULL_SHARD`) does but DeepSpeed-Zero-2 does not.

`--mixed_precision=no` is required. The released checkpoint already pins `proj_out` / `audio_proj_out` in float32 (`_keep_in_fp32_modules`); the parallel heads built from them stay float32 master weights over a bfloat16 backbone, and the run does not use autocast.

**fl2va (from a prompt cache — the default recipe):**

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  --video_sample_n_frames=124 \
  --fix_sample_size 768 1344 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-5 \
  --lora_learning_rate=1e-4 \
  --seed=43 \
  --output_dir="output_dir_minimax_h3_pdd_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="no" \
  --adam_weight_decay=0.0 \
  --max_grad_norm=1.0 \
  --rank=64 \
  --network_alpha=64 \
  --low_vram \
  --target_name="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear" \
  --train_mode="fl2va" \
  --pdd_num_steps=32 \
  --pdd_block_size=4 \
  --validation_steps=200 \
  --resume_from_checkpoint=latest
```

#### 3.2.1 Ref2VA Training

`--train_mode=ref2va` conditions on reference media (images / videos / audio) in addition to the prompt, and loads `transformer_ref` by default. It runs on either conditioning route; both commands below are the FSDP Quick Start with only the mode / data flags changed.

**Route A — direct load (no request cache).** Point `--train_data_meta` straight at a request annotation: either the explicit `{"prompt", "references"}` schema or the official `X-Fun-Videos-Audios-Demo` annotation, whose own video + audio become the references (see **2.4**). The run loads the Qwen3-VL conditioner and the two VAEs and encodes every request on the fly, so keep `--low_vram` — they are onloaded only while encoding.

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export REQUEST_META="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_mode="ref2va" \
  --train_data_meta=$REQUEST_META \
  --val_data_meta=$REQUEST_META \
  --video_sample_n_frames=124 \
  --fix_sample_size 768 1344 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-5 \
  --lora_learning_rate=1e-4 \
  --seed=43 \
  --output_dir="output_dir_minimax_h3_pdd_ref2va_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="no" \
  --adam_weight_decay=0.0 \
  --max_grad_norm=1.0 \
  --rank=64 \
  --network_alpha=64 \
  --low_vram \
  --target_name="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear" \
  --pdd_num_steps=32 \
  --pdd_block_size=4 \
  --validation_steps=200 \
  --resume_from_checkpoint=latest
```

> ⚠️ Route A keeps the ~62 GB Qwen3-VL conditioner resident in the run (sharded by FSDP alongside the transformer) and VAE-encodes each request's references on every trajectory reset. For long or repeated runs, pre-encode once with Route B so training never loads the conditioner.

**Route B — request cache (recommended for long / repeated runs).** Pre-encode the requests once with `generate_ref2va_request_cache.sh` (**2.3**), then train with `--enable_preprocess_training`; the ~62 GB conditioner stays out of the run:

```bash
# 1) Pre-encode the ref2va requests once (multi-GPU): prompt embeds + reference latents → safetensors
bash scripts/minimax_h3/generate_ref2va_request_cache.sh

# 2) Train off the cache — Route A with these flags changed
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export REQUEST_CACHE_META="datasets/minimax_h3_pdd_request_cache/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_mode="ref2va" \
  --enable_preprocess_training \
  --train_data_meta=$REQUEST_CACHE_META \
  --val_data_meta=$REQUEST_CACHE_META \
  ... # the remaining arguments identical to Route A
```

### 3.3 PDD Training Parameters

**PDD / LoRA parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|----------------|
| `--pdd_num_steps` | Grid size `N` | 32 |
| `--pdd_block_size` | `L_min`: intervals the carried state advances by (`NFE = N / L`) | 4 |
| `--pdd_max_block_size` | `L_max`: widest block a loss target is drawn from. Defaults to `--pdd_block_size` | 4 |
| `--pdd_solver` | Runge-Kutta method for the teacher's mean velocity: `euler` or `midpoint` | `midpoint` |
| `--pdd_num_targets` | How many intra-block indices `k` one student evaluation is supervised at | 2 |
| `--rank` | Dimension of LoRA update matrices | 64 |
| `--network_alpha` | Scale of LoRA update matrices | 64 |
| `--target_name` | Modules to apply LoRA (comma-separated) | `to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear` |
| `--learning_rate` | Learning rate of the parallel heads | 1e-5 |
| `--lora_learning_rate` | Learning rate of the low-rank updates | 1e-4 |
| `--use_ema` | Keep an EMA of the trainable set; validation and `pdd_ema.safetensors` use it | off |
| `--ema_decay` | EMA decay | 0.99 |

**Common training parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|----------------|
| `--pretrained_model_name_or_path` | Path to pretrained model | `models/Diffusion_Transformer/MiniMax-H3` |
| `--enable_preprocess_training` | Train on the pre-processed safetensors cache instead of encoding the conditioning on the fly; both `fl2va` and `ref2va` support either route | on |
| `--train_data_dir` | Optional root prepended to each `file_path` of `--train_data_meta`; empty when it stores repo-relative/absolute paths | `""` |
| `--train_data_meta` | With the flag: the cache `outputs.json`. Without it: the on-the-fly annotation (`fl2va`: `{"text": ...}`; `ref2va`: the request list) | `datasets/minimax_h3_pdd_prompt_cache/outputs.json` |
| `--val_data_meta` | Mirrors `--train_data_meta`: with the flag the cache `outputs.json`, without it the on-the-fly annotation (`fl2va`: `{"text": ...}`; `ref2va`: the request list). Skipped when empty | `datasets/minimax_h3_pdd_prompt_cache/outputs.json` |
| `--train_mode` | `fl2va` (t2va layout) or `ref2va` (`transformer_ref` + reference media); both run on the cache or the direct-load route | `fl2va` |
| `--transformer_subfolder` | Transformer subfolder. Default: `transformer_ref` for `ref2va`, else `transformer` | None |
| `--train_batch_size` | Must be 1: each rank carries one trajectory | 1 |
| `--num_train_epochs` | Training epochs when `--max_train_steps` is omitted. One epoch is one pass through the conditioning set | 100 |
| `--max_train_steps` | Total optimization steps. If set, overrides `--num_train_epochs` | 3000 |
| `--video_sample_n_frames` | Number of frames, must follow the `17*n+5` form of the video VAE (duration stays between 0 and 16 seconds) | 124 |
| `--video_sample_size` | Square canvas size (height = width); must be a multiple of 32 | 1280 |
| `--fix_sample_size` | Fixed `[height, width]` overriding `--video_sample_size` for a non-square canvas; both must be multiples of 32 | 768 1344 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--checkpointing_steps` | Save checkpoint every N steps | 200 |
| `--seed` | Random seed | 43 |
| `--output_dir` | Output directory | `output_dir_minimax_h3_pdd_lora` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--gradient_checkpointing_save_on_cpu` | Offload the activations saved for backward of the transformer blocks to CPU memory | - |
| `--mixed_precision` | Use `no`. The parallel heads stay float32 over a bfloat16 backbone | `no` |
| `--adam_weight_decay` | AdamW weight decay | 0.0 |
| `--max_grad_norm` | Gradient clipping threshold | 1.0 |
| `--low_vram` | Keep the VAEs on CPU; they move to GPU only inside validation decode | - |
| `--resume_from_checkpoint` | Resume from a checkpoint path, or `"latest"` to auto-select | `latest` |
| `--validation_steps` | Run validation every N steps | 200 |
| `--validation_nfe` | Student NFE during validation; must divide `--pdd_num_steps` | 8 |
| `--video_loss_weight` / `--audio_loss_weight` | Weights of the joint video + audio MSE | 0.5 / 0.5 |

### 3.4 Training Validation

Validation does **not** take `--validation_prompts`. It generates every entry of `--val_data_meta`, sharded across ranks, at `--validation_nfe`: the cache `outputs.json` under `--enable_preprocess_training`, or the on-the-fly annotation without it — mirroring the training route for both `fl2va` (prompts) and `ref2va` (requests). Validation is skipped when `--val_data_meta` is empty.

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Validate every N steps | 200 |
| `--validation_nfe` | Student evaluations per clip (`N / NFE` must be an integer) | 8 |

Videos are saved under `output_dir/sample/` as `sample-{step}-prompt{index}-{train_mode}-nfe{nfe}.mp4` (with audio).

### 3.5 Checkpoint Layout

Each `checkpoint-{step}/` holds:

| File | Role |
|------|------|
| `pdd.safetensors` | Live (non-EMA) gathered trainable tensors (parallel heads + LoRA). Used to resume DDP |
| `pdd_ema.safetensors` | EMA export when `--use_ema` is on; this is the inference file |
| `pdd_config.json` | Rank / alpha / targets / grid, read by `examples/minimax_h3/predict_t2v.py` |
| `optimizer.pt` / `scheduler.pt` / `scaler.pt` / `ema.pt` | DDP trainer state (`optimizer.bin` / `scheduler.bin` from Accelerate `--save_state` are also accepted on resume) |

FSDP stage 3 / ZeRO-3 also write `accelerator.save_state` into the same folder (auto `--save_state`) and still export a gathered `pdd.safetensors` (live) plus `pdd_ema.safetensors` when EMA is on. Checkpoints written before the rename stored live weights in `pdd_live.safetensors`; DDP resume still loads that file when it is present.

### 3.6 Training with DeepSpeed-Zero-2

> ⚠️ **Warning**: DeepSpeed-Zero-2 only partitions optimizer states and gradients, **not the model weights**. The MiniMax-H3 transformer is about 62 GB, so each GPU still holds a full weight replica and this setup usually runs out of memory. Prefer FSDP (**3.2**); the command below is provided for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  ... # the same train_pdd_lora.py arguments as the Quick Start
```

### 3.7 Training Without DeepSpeed or FSDP

**This approach is not recommended on 80 GB cards**: every GPU keeps a full ~62 GB transformer replica. PDD does not load Qwen3-VL, so DDP is lighter than `scripts/minimax_h3/train_lora.py`, but FSDP (**3.2**) is still the default. Drop `--use_fsdp` and the FSDP wrap flags from the Quick Start command; DDP resume reads `pdd.safetensors` plus `optimizer.pt` / `optimizer.bin`.

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  ... # the same train_pdd_lora.py arguments as the Quick Start
```

### 3.8 Multi-Machine Distributed Training

**Suitable for**: more GPUs, faster training

#### 3.8.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  ... # the same train_pdd_lora.py arguments as the Quick Start
```

**Machine 1 (Worker)**:
```bash
export RANK=1  # Note this is 1
# Other environment variables identical to Machine 0

# Use the same accelerate launch command as Machine 0
```

#### 3.8.2 Multi-Machine Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Synchronization**: All machines must be able to access the same conditioning cache and model paths (NFS/shared storage)

## 4. Inference Testing

PDD inference attaches the parallel heads and LoRA from `pdd_ema.safetensors` (falling back to `pdd.safetensors` when EMA was not saved), then samples at `num_inference_steps` NFE. Use `examples/minimax_h3/predict_t2v.py`; do not set `lora_path` at the same time (`lora_path` and `pdd_lora_path` cannot be used together).

The default recipe (`N = 32`, `L = 4`) runs at **8** inference steps. `num_inference_steps` must divide `pdd_num_steps` from `pdd_config.json`. If it is left at the teacher default of 40, the script snaps it to `N / L`.

### 4.1 Inference Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | GPU memory mode, see table below for options | `model_cpu_offload` |
| `ulysses_degree` | Head dimension parallelization degree, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelization degree, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer in multi-GPU inference to save VRAM | `False` |
| `fsdp_text_encoder` | Use FSDP for the Qwen3-VL text encoder in multi-GPU inference to save VRAM | `False` |
| `compile_dit` | Compile Transformer to accelerate inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/MiniMax-H3` |
| `transformer_path` | Path to trained Transformer weights | `None` |
| `vae_path` | Path to trained VAE weights | `None` |
| `pdd_lora_path` | PDD checkpoint directory (loads `pdd_ema.safetensors` if present, else `pdd.safetensors`, plus `pdd_config.json`) or a `.safetensors` file | `output_dir_minimax_h3_pdd_lora/checkpoint-3000` |
| `lora_path` | Turbo/PEFT LoRA; **cannot** be combined with `pdd_lora_path` | `None` |
| `sample_size` | Generated video resolution `[height, width]`; height/width must be multiples of 32. `None` uses MiniMax-H3's own 16:9 canvas (768x1344) | `[768, 1344]` |
| `video_length` | Number of frames to generate, snapped up to the next `17*n+5` the video VAE can decode (duration stays between 0 and 16 seconds) | 124 |
| `fps` | Frames per second (MiniMax-H3 generates at a fixed 24 fps) | 24 |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `prompt` | Positive prompt describing the content to generate | `"A red fox trotting..."` |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Student NFE. Default PDD recipe uses 8 (not the teacher's 40) | 8 |
| `guidance_scale` | Guidance strength. The released checkpoint is guidance-distilled: keep it at 1 to run one forward pass per step with no CFG | 1 |
| `flow_shift` | Exponential sigma shift of the video schedule, `None` keeps the one of the checkpoint (12.0) | `None` |
| `audio_flow_shift` | Exponential sigma shift of the audio schedule, `None` keeps the one of the checkpoint (3.0) | `None` |
| `save_path` | Generated video save path | `samples/minimax-h3-videos-t2v` |

**GPU Memory Mode Description**:

| Mode | Description | VRAM Usage |
|------|------|---------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer group offload between CPU/CUDA | Low |
| `sequential_cpu_offload` | Offload each layer individually (slowest) | Lowest |

> 💡 The transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner is another 62.1 GB, so a single 80 GB card needs `model_cpu_offload` or `model_group_offload`. Inference *does* load the text encoder; training does not.

### 4.2 Single GPU Inference

Run single GPU inference with:

```bash
python examples/minimax_h3/predict_t2v.py
```

Edit `examples/minimax_h3/predict_t2v.py` according to your needs. For PDD inference, focus on these parameters:

```python
# Choose based on your GPU VRAM
GPU_memory_mode = "model_cpu_offload"
# Your actual model path
model_name = "models/Diffusion_Transformer/MiniMax-H3"
# PDD checkpoint directory or weights file; a directory prefers pdd_ema.safetensors. Rank / alpha / targets / grid are read from pdd_config.json
pdd_lora_path = "output_dir_minimax_h3_pdd_lora/checkpoint-3000"
# Must stay None when pdd_lora_path is set
lora_path = None
# Student NFE; 8 for the default N=32 / L=4 recipe. Left at 40, the script snaps to N / L
num_inference_steps = 8
# Write based on content to generate
prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
# ...
```

Image-to-video and Ref2VA use the same `pdd_lora_path` field in `examples/minimax_h3/predict_i2v.py` and `examples/minimax_h3/predict_ref2va.py`. Ref2VA needs a checkpoint trained with `--train_mode=ref2va`.

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser yunchang
```

#### Configure Parallel Strategy

Edit `examples/minimax_h3/predict_t2v.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelization
ring_degree = 1     # Sequence dimension parallelization
```

**Configuration Principles**:
- `ulysses_degree` must evenly divide the model's number of heads
- `ring_degree` splits on sequence dimension, affecting communication overhead; avoid using it when heads can be divided
- Multi-GPU runs through the xfuser sequence-parallel path and is **incompatible with the `*cpu_offload*` memory modes** (accelerate offload hooks own a single device); use `model_full_load` / `model_full_load_and_qfloat8` across GPUs there, with `fsdp_dit` / `fsdp_text_encoder` to save memory
- Sequence parallel needs a working FlashAttention. Without it, run independent single-GPU jobs (`CUDA_VISIBLE_DEVICES=i`, `ulysses_degree = 1`, `ring_degree = 1`) instead

**Example Configurations**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelization |
| 8 | 2 | 4 | Hybrid parallelization |
| 8 | 8 | 1 | Head parallelization |

#### Run Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 examples/minimax_h3/predict_t2v.py
```

## 5. Additional Resources

- **PDD paper**: https://arxiv.org/abs/2607.26004
- **MiniMax-H3 Official GitHub**: https://github.com/MiniMax-AI/MiniMax-H3
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
- **Base MiniMax-H3 LoRA training**: `scripts/minimax_h3/README_TRAIN_LORA.md`
- **MiniMax-H3 Fun control training**: `scripts/minimax_h3_fun/README_TRAIN.md`


---

## FILE: examples/minimax_h3_fun/predict_v2v_control.py (https://github.com/aigc-apps/VideoX-Fun/blob/HEAD/examples/minimax_h3_fun/predict_v2v_control.py)

```
import os
import sys

import torch

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3ControlTransformer3DModel,
                               Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline import MiniMaxH3ControlPipeline
from videox_fun.utils import (MiniMaxH3Scheduler, apply_gpu_memory_mode,
                              convert_model_weight_to_float8,
                              get_video_to_video_latent, merge_lora,
                              save_videos_with_audio_grid, unmerge_lora)


# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, model_group_offload, sequential_cpu_offload, text_encoder_offload, text_encoder_offload_and_qfloat8].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_group_offload transfers internal layer groups between CPU/CUDA, 
# balancing memory efficiency and speed between full-module and leaf-level offloading methods.
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
# 
# text_encoder_offload keeps only the text encoder in CPU RAM, streaming it to the GPU module by module for
# its single encode pass, so the transformer stays resident (add `_and_qfloat8` for float8 storage).
GPU_memory_mode     = "model_group_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
# Multi-GPU runs through the xfuser sequence-parallel path and must be launched with torchrun, e.g.
# `torchrun --nproc_per_node=2 examples/minimax_h3_fun/predict_v2v_control.py` for ulysses_degree=2, ring_degree=1.
# It is incompatible with the *cpu_offload* memory modes (accelerate offload hooks own a single device);
# use model_full_load / model_full_load_and_qfloat8 there, with fsdp_dit to save memory.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus. The Qwen3-VL conditioner is ~62 GB, so with fsdp_dit alone every
# rank still replicates it; fsdp_text_encoder shards it too. Note it must wrap the inner `text_encoder.model`
# (Qwen3VLModel): encode_prompt calls that submodule directly, so a wrap on the top-level module would never fire.
fsdp_dit            = True
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/MiniMax-H3"
# Control branch layout, must match the yaml `train_control.py` ran with: `control_blocks_places` selects the
# layers the control blocks attach to and `control_in_dim` the channels the control rows carry (49 for an
# `--enable_inpaint` checkpoint, whose `control_proj_in` is widened with the mask channels). Leaving it None
# builds the default 24-channel branch, which cannot load an inpaint checkpoint.
config_path         = "config/minimax_h3/minimax_h3_control_inpaint_post_norm.yaml"

# Load pretrained model if need. The control branch is not part of the released MiniMax-H3 weights, so a base
# `model_name` starts the side branch as an identity (`after_proj` is zero) and the c ontrol video has no effect;
# point `transformer_path` at a control checkpoint trained by `scripts/minimax_h3_fun/train_control.py`.
transformer_path    = "models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union-2.0/MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors"
vae_path            = None
lora_path           = None

# Other params
# MiniMax-H3 generates at a fixed 24 fps, only accepts multiples of 32 as height / width, and the generation
# follows the control video's actual length — snapped down to the largest 17 * n + 5 the video VAE can decode so
# a short control video is never padded (the duration has to stay under 15 seconds), capped by video_length.
# Control inference fits the control video onto this canvas with the training's resize + crop geometry, so
# sample_size must be set (it cannot be None).
sample_size         = [1280, 704]
video_length        = 243
fps                 = 24
# Scale applied to every control skip before it is added to the main branch. 0.0 switches the control branch off,
# values below 1.0 weaken the guidance of the control video.
control_context_scale = 1.00

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
control_video       = "asset/pose.mp4"
# Inpaint inputs, only read by checkpoints trained with `--enable_inpaint` (control_in_dim widened, e.g. 49):
# `inpaint_video` is the source video behind the mask and `inpaint_video_mask` marks the regions to regenerate
# (white = repaint, black = keep). With an inpaint checkpoint but no inpaint inputs given, the pipeline zero-pads
# the mask channels and the run degrades to pure generation; a mask-less checkpoint rejects them outright.
inpaint_video       = None
inpaint_video_mask  = None
prompt              = "视频中，一位年轻女性站在阳光洒满的沙滩上，背景是无垠碧蓝的大海与澄澈如洗的天空，构成一幅充满夏日度假氛围的画面。她身穿一件深海军蓝吊带泳衣，线条简约贴身，凸显健康匀称的身材曲线；外搭一条纯白色背带短裙，裙摆轻盈飘逸，随风微微扬起，增添了几分俏皮与少女感。她的长发柔顺披肩，发梢微卷，在阳光下泛着自然光泽，耳畔垂挂着一对小巧精致的珍珠吊坠耳环，为整体造型注入一丝温柔优雅的气息。她面带甜美笑容，嘴角上扬，露出整齐洁白的牙齿，眼神清澈明亮，直视镜头时流露出真诚与自信，仿佛在与观众分享此刻的快乐。"
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
seed                = 43
# Number of denoising steps, i.e. of model evaluations: num_inference_steps = 40 runs 40 of them.
num_inference_steps = 40
# The released checkpoint is guidance-distilled: leave guidance_scale at 1 to run one forward pass per step
# with no CFG — the distill checkpoints of train_control_distill.py already bake the teacher's CFG target into
# the weights, so any value above 1 applies guidance twice and degrades the output. A value above 1 enables
# classifier-free guidance with a negative_prompt, running two passes.
guidance_scale      = 1.0
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
# The exponential sigma shifts of the two schedules. None keeps the ones of the checkpoint (12.0 video, 3.0 audio).
flow_shift          = None
audio_flow_shift    = None
lora_weight         = 0.55
save_path           = "samples/minimax-h3-videos-v2v-control"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# The yaml pins the control branch layout exactly as in training (scripts/minimax_h3_fun/train_control.py), where
# `transformer_additional_kwargs` is spread into `from_pretrained` the same way.
transformer_load_kwargs = {}
if config_path is not None:
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    transformer_load_kwargs.update(
        OmegaConf.to_container(config["transformer_additional_kwargs"], resolve=True)
    )

# `model_name` may point either at a converted diffusers layout or at an *original* MiniMax-H3 partition (e.g.
# `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, no intermediate copy on disk.
# Transformer. `from_pretrained` fills the control branch the released checkpoint does not carry: every control
# block is initialised from the main block it is attached to and `control_proj_in` from `proj_in`, with
# before_proj / after_proj zeroed, so a freshly loaded model is numerically identical to the base MiniMax-H3 model.
transformer = MiniMaxH3ControlTransformer3DModel.from_pretrained(
    model_name,
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
    **transformer_load_kwargs,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Video VAE. The released weights are float32 and the decode runs under float16 autocast, so the VAE is not
# downcast even when the rest of the pipeline is bfloat16.
vae = AutoencoderKLMiniMaxH3.from_pretrained(
    model_name,
    subfolder="vae",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Audio VAE, waveform in / waveform out: MiniMax-H3 has no separate vocoder.
audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
    model_name,
    subfolder="audio_vae",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Get Tokenizer and Processor
tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(model_name, "tokenizer"))
processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_name, "processor"))

# Get Text encoder. MiniMax-H3 reads the unnormalized hidden state after the 50th decoder layer of Qwen3-VL.
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(model_name, "text_encoder"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Schedulers. MiniMax-H3 steps the video and the audio latents down two schedules inside one transformer call.
scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="scheduler")
audio_scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="audio_scheduler")

pipeline = MiniMaxH3ControlPipeline(
    vae=vae,
    audio_vae=audio_vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    processor=processor,
    transformer=transformer,
    scheduler=scheduler,
    audio_scheduler=audio_scheduler,
)

# The float32 modules of the mixed-precision checkpoint stay untouched by the float8 quantization. The `proj_in`
# entry also covers the control patch projection `control_proj_in`, which shares the video patch projection's dtype.
fp8_exclude_module_name = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
use_qfloat8 = "qfloat8" in GPU_memory_mode
if use_qfloat8:
    convert_model_weight_to_float8(transformer, exclude_module_name=fp8_exclude_module_name, device=device)

# The pure *cpu_offload modes (qfloat8 variants excluded, unsupported under FSDP). Under sequence parallelism the
# sharded modules offload through FSDP's own CPUOffload, enabled here at wrap time; the pipeline then routes the
# whole-model offload call to the non-FSDP VAE / audio VAE automatically, so no per-mode branch is needed below.
wants_offload = GPU_memory_mode in ("model_cpu_offload", "model_group_offload", "sequential_cpu_offload")
# text_encoder_offload keeps the transformer resident and offloads only the conditioner, so under FSDP the text
# encoder is wrapped with CPUOffload while the DiT stays on the card; match the base to cover `_and_qfloat8`.
te_wants_offload = wants_offload or GPU_memory_mode.startswith("text_encoder_offload")

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        fp32_modules = [m for m in transformer.modules()
                        if any(p.dtype == torch.float32 for p in m.parameters(recurse=False))]
        shard_fn = partial(shard_model, device_id=device, param_dtype=None, cast_dtype=False,
                          module_to_wrapper=list(transformer.transformer_blocks) + list(transformer.control_blocks),
                          ignored_modules=fp32_modules, offload_to_cpu=wants_offload)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype,
                          module_to_wrapper=list(text_encoder.model.language_model.layers),
                          offload_to_cpu=te_wants_offload)
        pipeline.text_encoder.model = shard_fn(pipeline.text_encoder.model)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
    print("Add Compile")

# Quantize (when the mode carries an "_and_<quant>" suffix) and then place the pipeline.
# The order lives inside the helper: quantization has to happen before the offload hooks
# are installed.
# The FP8 conversion above has already run (before the FSDP sharding, on purpose); only the dequant
# wrapper and the memory placement are left, which is what the preconverted tag installs.
apply_gpu_memory_mode(pipeline, GPU_memory_mode, device, weight_dtype,
                      quant_tag="qfloat8_preconverted" if GPU_memory_mode.endswith("_and_qfloat8") else None,
                      exclude_module_name=[])

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def snap_num_frames(actual_num_frames, max_num_frames):
    """
    Pick the generation length from the control video instead of padding a short one: the largest `17 * n + 5`
    the video VAE can decode that does not exceed the frames actually read (capped by `max_num_frames`), snapping
    down so no tail frame is ever repeated. A control video below 5 frames is raised to 5, the smallest count
    the video VAE can encode.
    """
    num_frames = min(actual_num_frames, max_num_frames)
    num_frames = (num_frames - 5) // 17 * 17 + 5
    return max(num_frames, 5)

with torch.no_grad():
    control_video, _, _, _ = get_video_to_video_latent(control_video, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None, keep_aspect_ratio=True)

    # Generate at the control video's actual length, never padding; only control videos below the 5 frames the
    # video VAE can encode are raised to 5.
    num_frames = snap_num_frames(control_video.shape[2], video_length)
    if num_frames != video_length:
        print(f"[{os.environ.get('RANK', '0')}] control video holds {control_video.shape[2]} frames, generating "
              f"{num_frames} instead of {video_length}", flush=True)

    mask_video = None
    if inpaint_video is not None:
        if inpaint_video_mask is None:
            raise ValueError("inpaint_video_mask is required when inpaint_video is provided")
        inpaint_video, _, _, _ = get_video_to_video_latent(inpaint_video, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None, keep_aspect_ratio=True)
        inpaint_video_mask, _, _, _ = get_video_to_video_latent(inpaint_video_mask, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None, keep_aspect_ratio=True)
        # Binarize the grayscale mask onto one channel: 1 marks the regions to regenerate, mirroring the training
        # `get_random_mask` convention the visibility map `1 - mask` is built from.
        mask_video = (inpaint_video_mask[:, :1] > 0.5).to(inpaint_video_mask.dtype)

    output = pipeline(
        prompt=prompt,
        control_video=control_video,
        control_context_scale=control_context_scale,
        mask_video=mask_video,
        inpaint_video=inpaint_video,
        height=None if sample_size is None else sample_size[0],
        width=None if sample_size is None else sample_size[1],
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        generator=generator,
        output_type="pt",
    )
print(f"[{os.environ.get('RANK', '0')}] generation done, decoding", flush=True)

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

sample = output.videos
audio = output.audio
audio_sample_rate = output.sampling_rate

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_videos_with_audio_grid(sample, audio, video_path, fps=fps, audio_sample_rate=audio_sample_rate)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
    # Keep every rank alive until the saving rank finishes; an early exit of one rank makes the elastic launcher
    # terminate the others.
    dist.barrier()
else:
    save_results()

```


---

## FILE: examples/minimax_h3/predict_t2v.py (https://github.com/aigc-apps/VideoX-Fun/blob/HEAD/examples/minimax_h3/predict_t2v.py)

```
import os
import sys

import torch

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3Transformer3DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline import MiniMaxH3Pipeline
from videox_fun.utils import (MiniMaxH3Scheduler, apply_gpu_memory_mode,
                              convert_model_weight_to_float8, merge_lora,
                              save_videos_with_audio_grid, unmerge_lora)

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, model_group_offload, sequential_cpu_offload, text_encoder_offload, text_encoder_offload_and_qfloat8].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_group_offload transfers internal layer groups between CPU/CUDA, 
# balancing memory efficiency and speed between full-module and leaf-level offloading methods.
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
# 
# text_encoder_offload keeps only the text encoder in CPU RAM, streaming it to the GPU module by module for
# its single encode pass, so the transformer stays resident (add `_and_qfloat8` for float8 storage). 
GPU_memory_mode     = "model_group_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus. The Qwen3-VL conditioner is ~62 GB, so with fsdp_dit alone every
# rank still replicates it; fsdp_text_encoder shards it too. Note it must wrap the inner `text_encoder.model`
# (Qwen3VLModel): encode_prompt calls that submodule directly, so a wrap on the top-level module would never fire.
fsdp_dit            = True
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/MiniMax-H3"

# Load pretrained model if need
# A full finetune goes in `transformer_path`, either as the `transformer` folder a training checkpoint writes
# (`output_dir_minimax_h3/checkpoint-N/transformer`, config.json included) or as a single safetensors file. A LoRA
# goes in `lora_path`: handed to `transformer_path` it would match no key at all and load nothing. A PDD LoRA
# (parallel decoder) goes in `pdd_lora_path` and cannot be combined with `lora_path`.
transformer_path    = None
vae_path            = None
lora_path           = None
pdd_lora_path       = None

# Other params
# MiniMax-H3 generates at a fixed 24 fps, only accepts multiples of 32 as height / width, and snaps video_length up
# to the next 17 * n + 5 the video VAE can decode (the duration has to stay between 0 and 16 seconds).
# Leave sample_size as None to use MiniMax-H3's own 16:9 canvas (768x1344).
sample_size         = [704, 1280]
video_length        = 124
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
seed                = 43
# Number of denoising steps, i.e. of model evaluations: num_inference_steps = 40 runs 40 of them.
num_inference_steps = 40
# The released checkpoint is guidance-distilled: leave guidance_scale at 1 to run one forward pass per step
# with no CFG. A value above 1 enables classifier-free guidance with a negative_prompt, running two passes.
guidance_scale      = 1
# The exponential sigma shifts of the two schedules. None keeps the ones of the checkpoint (12.0 video, 3.0 audio).
flow_shift          = None
audio_flow_shift    = None
lora_weight         = 0.55
save_path           = "samples/minimax-h3-videos-t2v"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# `model_name` may point either at a converted diffusers layout or at an *original* MiniMax-H3 partition (e.g.
# `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, no intermediate copy on disk.
# Transformer
transformer = MiniMaxH3Transformer3DModel.from_pretrained(
    model_name,
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if os.path.isdir(transformer_path):
        # A training checkpoint's `transformer` folder carries its own config.json, so the loader restores the
        # mixed-precision contract of the checkpoint (`_keep_in_fp32_modules`) by itself.
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            transformer_path,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        # `strict=False` accepts a file whose keys belong to another model — a LoRA checkpoint, say — by loading
        # nothing at all and silently generating with the base weights, so an unexpected key is a hard error.
        assert len(u) == 0, (
            f"{transformer_path} holds {len(u)} key(s) the transformer does not have, e.g. {u[:3]}. A LoRA "
            "checkpoint belongs in `lora_path`, not `transformer_path`."
        )

pdd_config = None
if pdd_lora_path is not None:
    if lora_path is not None:
        raise ValueError("`lora_path` and `pdd_lora_path` cannot be used together.")
    from videox_fun.models.minimax_h3_pdd import (load_pdd_lora,
                                                  pdd_num_inference_steps,
                                                  pdd_step_callback)
    pdd_config = load_pdd_lora(transformer, pdd_lora_path)
    num_inference_steps = pdd_num_inference_steps(pdd_config, num_inference_steps, teacher_default=40)

# Video VAE. The released weights are float32 and the decode runs under float16 autocast, so the VAE is not
# downcast even when the rest of the pipeline is bfloat16 (this is also how the training scripts load it).
vae = AutoencoderKLMiniMaxH3.from_pretrained(
    model_name,
    subfolder="vae",
    low_cpu_mem_usage=True,
)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Audio VAE, waveform in / waveform out: MiniMax-H3 has no separate vocoder. Float32 as released, like the video VAE.
audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
    model_name,
    subfolder="audio_vae",
    low_cpu_mem_usage=True,
)

# Get Tokenizer and Processor
tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(model_name, "tokenizer"))
processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_name, "processor"))

# Get Text encoder. MiniMax-H3 reads the unnormalized hidden state after the 50th decoder layer of Qwen3-VL.
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(model_name, "text_encoder"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Schedulers. MiniMax-H3 steps the video and the audio latents down two schedules inside one transformer call.
scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="scheduler")
audio_scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="audio_scheduler")

pipeline = MiniMaxH3Pipeline(
    vae=vae,
    audio_vae=audio_vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    processor=processor,
    transformer=transformer,
    scheduler=scheduler,
    audio_scheduler=audio_scheduler,
)

# The float32 modules of the mixed-precision checkpoint stay untouched by the float8 quantization.
fp8_exclude_module_name = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
use_qfloat8 = "qfloat8" in GPU_memory_mode
if use_qfloat8:
    convert_model_weight_to_float8(transformer, exclude_module_name=fp8_exclude_module_name, device=device)

# The pure *cpu_offload modes (qfloat8 variants excluded, unsupported under FSDP). Under sequence parallelism the
# sharded modules offload through FSDP's own CPUOffload, enabled here at wrap time; the pipeline then routes the
# whole-model offload call to the non-FSDP VAE / audio VAE automatically, so no per-mode branch is needed below.
wants_offload = GPU_memory_mode in ("model_cpu_offload", "model_group_offload", "sequential_cpu_offload")
# text_encoder_offload keeps the transformer resident and offloads only the conditioner, so under FSDP the text
# encoder is wrapped with CPUOffload while the DiT stays on the card; match the base to cover `_and_qfloat8`.
te_wants_offload = wants_offload or GPU_memory_mode.startswith("text_encoder_offload")

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        fp32_modules = [m for m in transformer.modules()
                        if any(p.dtype == torch.float32 for p in m.parameters(recurse=False))]
        shard_fn = partial(shard_model, device_id=device, param_dtype=None, cast_dtype=False,
                          module_to_wrapper=list(transformer.transformer_blocks),
                          ignored_modules=fp32_modules, offload_to_cpu=wants_offload)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype,
                          module_to_wrapper=list(text_encoder.model.language_model.layers),
                          offload_to_cpu=te_wants_offload)
        pipeline.text_encoder.model = shard_fn(pipeline.text_encoder.model)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
    print("Add Compile")

# Quantize (when the mode carries an "_and_<quant>" suffix) and then place the pipeline.
# The order lives inside the helper: quantization has to happen before the offload hooks
# are installed.
# The FP8 conversion above has already run (before the FSDP sharding, on purpose); only the dequant
# wrapper and the memory placement are left, which is what the preconverted tag installs.
apply_gpu_memory_mode(pipeline, GPU_memory_mode, device, weight_dtype,
                      quant_tag="qfloat8_preconverted" if GPU_memory_mode.endswith("_and_qfloat8") else None,
                      exclude_module_name=[])

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

pdd_callback = None if pdd_config is None else pdd_step_callback(
    transformer, scheduler, audio_scheduler, pdd_config, num_inference_steps
)

with torch.no_grad():
    output = pipeline(
        prompt=prompt,
        height=None if sample_size is None else sample_size[0],
        width=None if sample_size is None else sample_size[1],
        num_frames=video_length,
        num_inference_steps=num_inference_steps,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
        callback_on_step_end=pdd_callback,
    )
print(f"[{os.environ.get('RANK', '0')}] generation done, decoding", flush=True)

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

sample = output.videos
audio = output.audio
audio_sample_rate = output.sampling_rate

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_videos_with_audio_grid(sample, audio, video_path, fps=fps, audio_sample_rate=audio_sample_rate)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
    # Keep every rank alive until the saving rank finishes; an early exit of one rank makes the elastic launcher
    # terminate the others.
    dist.barrier()
else:
    save_results()

```


---

## FILE: examples/minimax_h3/predict_i2v.py (https://github.com/aigc-apps/VideoX-Fun/blob/HEAD/examples/minimax_h3/predict_i2v.py)

```
import os
import sys

import torch
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3Transformer3DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline import MiniMaxH3Pipeline
from videox_fun.utils import (MiniMaxH3Scheduler, apply_gpu_memory_mode,
                              convert_model_weight_to_float8, merge_lora,
                              save_videos_with_audio_grid, unmerge_lora)

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, model_group_offload, sequential_cpu_offload, text_encoder_offload, text_encoder_offload_and_qfloat8].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_group_offload transfers internal layer groups between CPU/CUDA, 
# balancing memory efficiency and speed between full-module and leaf-level offloading methods.
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
# 
# text_encoder_offload keeps only the text encoder in CPU RAM, streaming it to the GPU module by module for
# its single encode pass, so the transformer stays resident (add `_and_qfloat8` for float8 storage).
GPU_memory_mode     = "model_group_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus. The Qwen3-VL conditioner is ~62 GB, so with fsdp_dit alone every
# rank still replicates it; fsdp_text_encoder shards it too. Note it must wrap the inner `text_encoder.model`
# (Qwen3VLModel): encode_prompt calls that submodule directly, so a wrap on the top-level module would never fire.
fsdp_dit            = True
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/MiniMax-H3"

# Load pretrained model if need
# A full finetune goes in `transformer_path`, either as the `transformer` folder a training checkpoint writes
# (`output_dir_minimax_h3/checkpoint-N/transformer`, config.json included) or as a single safetensors file. A LoRA
# goes in `lora_path`: handed to `transformer_path` it would match no key at all and load nothing. A PDD LoRA
# (parallel decoder) goes in `pdd_lora_path` and cannot be combined with `lora_path`.
transformer_path    = None
vae_path            = None
lora_path           = None
pdd_lora_path       = None

# Other params
# MiniMax-H3 generates at a fixed 24 fps, only accepts multiples of 32 as height / width, and snaps video_length up
# to the next 17 * n + 5 the video VAE can decode (the duration has to stay between 0 and 16 seconds).
# Leave sample_size as None to follow the aspect ratio of the first keyframe, which is what the model was released
# for; the first keyframe is stretched onto that canvas.
sample_size         = [704, 1280]
video_length        = 124
fps                 = 24

# The keyframe the video starts from, and the one it ends on. Either can be left as None: only an end frame generates
# *up to* that frame, and neither of them is a plain text-to-video request (see predict_t2v.py).
validation_image_start  = "asset/1.png"
validation_image_end    = None

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "一只棕色的狗摇着头，坐在舒适房间里的浅色沙发上。在狗的后面，架子上有一幅镶框的画，周围是粉红色的花朵。房间里柔和温暖的灯光营造出舒适的氛围。"
seed                = 43
# Number of denoising steps, i.e. of model evaluations: num_inference_steps = 40 runs 40 of them.
num_inference_steps = 40
# The released checkpoint is guidance-distilled: leave guidance_scale at 1 to run one forward pass per step
# with no CFG. A value above 1 enables classifier-free guidance with a negative_prompt, running two passes.
guidance_scale      = 1
# The exponential sigma shifts of the two schedules. None keeps the ones of the checkpoint (12.0 video, 3.0 audio).
flow_shift          = None
audio_flow_shift    = None
lora_weight         = 0.55
save_path           = "samples/minimax-h3-videos-i2v"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# `model_name` may point either at a converted diffusers layout or at an *original* MiniMax-H3 partition (e.g.
# `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, no intermediate copy on disk.
# Transformer
transformer = MiniMaxH3Transformer3DModel.from_pretrained(
    model_name,
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if os.path.isdir(transformer_path):
        # A training checkpoint's `transformer` folder carries its own config.json, so the loader restores the
        # mixed-precision contract of the checkpoint (`_keep_in_fp32_modules`) by itself.
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            transformer_path,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        # `strict=False` accepts a file whose keys belong to another model — a LoRA checkpoint, say — by loading
        # nothing at all and silently generating with the base weights, so an unexpected key is a hard error.
        assert len(u) == 0, (
            f"{transformer_path} holds {len(u)} key(s) the transformer does not have, e.g. {u[:3]}. A LoRA "
            "checkpoint belongs in `lora_path`, not `transformer_path`."
        )

pdd_config = None
if pdd_lora_path is not None:
    if lora_path is not None:
        raise ValueError("`lora_path` and `pdd_lora_path` cannot be used together.")
    from videox_fun.models.minimax_h3_pdd import (load_pdd_lora,
                                                  pdd_num_inference_steps,
                                                  pdd_step_callback)
    pdd_config = load_pdd_lora(transformer, pdd_lora_path)
    num_inference_steps = pdd_num_inference_steps(pdd_config, num_inference_steps, teacher_default=40)

# Video VAE. The released weights are float32 and the decode runs under float16 autocast, so the VAE is not
# downcast even when the rest of the pipeline is bfloat16 (this is also how the training scripts load it).
vae = AutoencoderKLMiniMaxH3.from_pretrained(
    model_name,
    subfolder="vae",
    low_cpu_mem_usage=True,
)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Audio VAE, waveform in / waveform out: MiniMax-H3 has no separate vocoder. Float32 as released, like the video VAE.
audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
    model_name,
    subfolder="audio_vae",
    low_cpu_mem_usage=True,
)

# Get Tokenizer and Processor
tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(model_name, "tokenizer"))
processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_name, "processor"))

# Get Text encoder. MiniMax-H3 reads the unnormalized hidden state after the 50th decoder layer of Qwen3-VL.
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(model_name, "text_encoder"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Schedulers. MiniMax-H3 steps the video and the audio latents down two schedules inside one transformer call.
scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="scheduler")
audio_scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="audio_scheduler")

pipeline = MiniMaxH3Pipeline(
    vae=vae,
    audio_vae=audio_vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    processor=processor,
    transformer=transformer,
    scheduler=scheduler,
    audio_scheduler=audio_scheduler,
)

# The float32 modules of the mixed-precision checkpoint stay untouched by the float8 quantization.
fp8_exclude_module_name = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
use_qfloat8 = "qfloat8" in GPU_memory_mode
if use_qfloat8:
    convert_model_weight_to_float8(transformer, exclude_module_name=fp8_exclude_module_name, device=device)

# The pure *cpu_offload modes (qfloat8 variants excluded, unsupported under FSDP). Under sequence parallelism the
# sharded modules offload through FSDP's own CPUOffload, enabled here at wrap time; the pipeline then routes the
# whole-model offload call to the non-FSDP VAE / audio VAE automatically, so no per-mode branch is needed below.
wants_offload = GPU_memory_mode in ("model_cpu_offload", "model_group_offload", "sequential_cpu_offload")
# text_encoder_offload keeps the transformer resident and offloads only the conditioner, so under FSDP the text
# encoder is wrapped with CPUOffload while the DiT stays on the card; match the base to cover `_and_qfloat8`.
te_wants_offload = wants_offload or GPU_memory_mode.startswith("text_encoder_offload")

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        fp32_modules = [m for m in transformer.modules()
                        if any(p.dtype == torch.float32 for p in m.parameters(recurse=False))]
        shard_fn = partial(shard_model, device_id=device, param_dtype=None, cast_dtype=False,
                          module_to_wrapper=list(transformer.transformer_blocks),
                          ignored_modules=fp32_modules, offload_to_cpu=wants_offload)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype,
                          module_to_wrapper=list(text_encoder.model.language_model.layers),
                          offload_to_cpu=te_wants_offload)
        pipeline.text_encoder.model = shard_fn(pipeline.text_encoder.model)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
    print("Add Compile")

# Quantize (when the mode carries an "_and_<quant>" suffix) and then place the pipeline.
# The order lives inside the helper: quantization has to happen before the offload hooks
# are installed.
# The FP8 conversion above has already run (before the FSDP sharding, on purpose); only the dequant
# wrapper and the memory placement are left, which is what the preconverted tag installs.
apply_gpu_memory_mode(pipeline, GPU_memory_mode, device, weight_dtype,
                      quant_tag="qfloat8_preconverted" if GPU_memory_mode.endswith("_and_qfloat8") else None,
                      exclude_module_name=[])

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

image_start = None if validation_image_start is None else Image.open(validation_image_start)
image_end = None if validation_image_end is None else Image.open(validation_image_end)

pdd_callback = None if pdd_config is None else pdd_step_callback(
    transformer, scheduler, audio_scheduler, pdd_config, num_inference_steps
)

with torch.no_grad():
    output = pipeline(
        prompt=prompt,
        image=image_start,
        last_image=image_end,
        height=None if sample_size is None else sample_size[0],
        width=None if sample_size is None else sample_size[1],
        num_frames=video_length,
        num_inference_steps=num_inference_steps,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
        callback_on_step_end=pdd_callback,
    )
print(f"[{os.environ.get('RANK', '0')}] generation done, decoding", flush=True)

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

sample = output.videos
audio = output.audio
audio_sample_rate = output.sampling_rate

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_videos_with_audio_grid(sample, audio, video_path, fps=fps, audio_sample_rate=audio_sample_rate)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
    # Keep every rank alive until the saving rank finishes; an early exit of one rank makes the elastic launcher
    # terminate the others.
    dist.barrier()
else:
    save_results()

```


---

## FILE: examples/minimax_h3/predict_ref2va.py (https://github.com/aigc-apps/VideoX-Fun/blob/HEAD/examples/minimax_h3/predict_ref2va.py)

```
import os
import sys

import torch

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3Transformer3DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline import (MiniMaxH3AudioReference,
                                 MiniMaxH3ImageReference, MiniMaxH3Pipeline,
                                 MiniMaxH3VideoReference)
from videox_fun.utils import (MiniMaxH3Scheduler, apply_gpu_memory_mode,
                              convert_model_weight_to_float8, merge_lora,
                              save_videos_with_audio_grid, unmerge_lora)

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, model_group_offload, sequential_cpu_offload, text_encoder_offload, text_encoder_offload_and_qfloat8].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_group_offload transfers internal layer groups between CPU/CUDA, 
# balancing memory efficiency and speed between full-module and leaf-level offloading methods.
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
# 
# text_encoder_offload keeps only the text encoder in CPU RAM, streaming it to the GPU module by module for
# its single encode pass, so the transformer stays resident (add `_and_qfloat8` for float8 storage).
GPU_memory_mode     = "model_group_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus. The Qwen3-VL conditioner is ~62 GB, so with fsdp_dit alone every
# rank still replicates it; fsdp_text_encoder shards it too. Note it must wrap the inner `text_encoder.model`
# (Qwen3VLModel): encode_prompt calls that submodule directly, so a wrap on the top-level module would never fire.
fsdp_dit            = True
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/MiniMax-H3"

# Load pretrained model if need
# The `ref2va` weights ship in their own subfolder, same architecture as the base transformer. A full finetune goes
# in `transformer_path`, either as the `transformer` folder a training checkpoint writes (config.json included) or
# as a single safetensors file, overriding the `transformer_ref` subfolder. A LoRA goes in `lora_path`: handed to
# `transformer_path` it would match no key at all and load nothing. A PDD LoRA (parallel decoder) goes in
# `pdd_lora_path` and cannot be combined with `lora_path`; use a checkpoint trained with `--train_mode=ref2va`.
transformer_subfolder = "transformer_ref"
transformer_path    = None
vae_path            = None
lora_path           = None
pdd_lora_path       = None

# Other params
# MiniMax-H3 generates at a fixed 24 fps, only accepts multiples of 32 as height / width, and snaps video_length up
# to the next 17 * n + 5 the video VAE can decode (the duration has to stay between 0 and 16 seconds). References
# never bind the generated geometry: leaving height / width unset resolves MiniMax-H3's own 16:9 canvas.
sample_size         = [1280, 704]
video_length        = 124
fps                 = 24

# The references to condition on, **in the order the model should read them**: the order labels them in the prompt
# presentation and lays them out on the shared rotary clock. One entry per reference, `image=path`, `video=path` or
# `audio=path`; a video's own soundtrack is conditioned on with it. Budgets of the released checkpoint: at most 9
# images, 3 videos, 3 audios and 12 references in total, and an audio reference cannot stand alone.
references          = [
    "video=asset/ref2va_video.mp4",
    "audio=asset/ref2va_audio.wav",
]

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "参考视频中的角色与场景，生成一段动作连贯、镜头流畅的续写视频，环境音与画面同步。"
seed                = 43
# Number of denoising steps, i.e. of model evaluations: num_inference_steps = 50 runs 50 of them.
num_inference_steps = 50
# The released `ref2va` checkpoint is guidance-distilled with no unconditional branch, so `references` runs one
# forward pass per step and needs guidance_scale of 1 — the pipeline raises on anything above.
guidance_scale      = 1.0
# The exponential sigma shifts of the two schedules. None keeps the ones of the checkpoint (12.0 video, 3.0 audio).
flow_shift          = None
audio_flow_shift    = None
lora_weight         = 0.55
save_path           = "samples/minimax-h3-videos-ref2va"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# `model_name` may point either at a converted diffusers layout or at an *original* MiniMax-H3 partition; the
# original shards are converted on the fly while loading, no intermediate copy on disk. The transformer comes from
# the `transformer_ref` subfolder — the released `ref2va` weights, same architecture as the base model.
transformer = MiniMaxH3Transformer3DModel.from_pretrained(
    model_name,
    subfolder=transformer_subfolder,
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if os.path.isdir(transformer_path):
        # A training checkpoint's `transformer` folder carries its own config.json, so the loader restores the
        # mixed-precision contract of the checkpoint (`_keep_in_fp32_modules`) by itself.
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            transformer_path,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        # `strict=False` accepts a file whose keys belong to another model — a LoRA checkpoint, say — by loading
        # nothing at all and silently generating with the base weights, so an unexpected key is a hard error.
        assert len(u) == 0, (
            f"{transformer_path} holds {len(u)} key(s) the transformer does not have, e.g. {u[:3]}. A LoRA "
            "checkpoint belongs in `lora_path`, not `transformer_path`."
        )

pdd_config = None
if pdd_lora_path is not None:
    if lora_path is not None:
        raise ValueError("`lora_path` and `pdd_lora_path` cannot be used together.")
    from videox_fun.models.minimax_h3_pdd import (load_pdd_lora,
                                                  pdd_num_inference_steps,
                                                  pdd_step_callback)
    pdd_config = load_pdd_lora(transformer, pdd_lora_path)
    num_inference_steps = pdd_num_inference_steps(pdd_config, num_inference_steps, teacher_default=50)

# Video VAE. The released weights are float32 and the decode runs under float16 autocast, so the VAE is not
# downcast even when the rest of the pipeline is bfloat16 (this is also how the training scripts load it).
vae = AutoencoderKLMiniMaxH3.from_pretrained(
    model_name,
    subfolder="vae",
    low_cpu_mem_usage=True,
)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Audio VAE, waveform in / waveform out: MiniMax-H3 has no separate vocoder. Float32 as released, like the video VAE.
audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
    model_name,
    subfolder="audio_vae",
    low_cpu_mem_usage=True,
)

# Get Tokenizer and Processor
tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(model_name, "tokenizer"))
processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_name, "processor"))

# Get Text encoder. MiniMax-H3 reads the unnormalized hidden state after the 50th decoder layer of Qwen3-VL.
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(model_name, "text_encoder"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Schedulers. MiniMax-H3 steps the video and the audio latents down two schedules inside one transformer call.
scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="scheduler")
audio_scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="audio_scheduler")

pipeline = MiniMaxH3Pipeline(
    vae=vae,
    audio_vae=audio_vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    processor=processor,
    transformer=transformer,
    scheduler=scheduler,
    audio_scheduler=audio_scheduler,
)

# The float32 modules of the mixed-precision checkpoint stay untouched by the float8 quantization.
fp8_exclude_module_name = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
use_qfloat8 = "qfloat8" in GPU_memory_mode
if use_qfloat8:
    convert_model_weight_to_float8(transformer, exclude_module_name=fp8_exclude_module_name, device=device)

# The pure *cpu_offload modes (qfloat8 variants excluded, unsupported under FSDP). Under sequence parallelism the
# sharded modules offload through FSDP's own CPUOffload, enabled here at wrap time; the pipeline then routes the
# whole-model offload call to the non-FSDP VAE / audio VAE automatically, so no per-mode branch is needed below.
wants_offload = GPU_memory_mode in ("model_cpu_offload", "model_group_offload", "sequential_cpu_offload")
# text_encoder_offload keeps the transformer resident and offloads only the conditioner, so under FSDP the text
# encoder is wrapped with CPUOffload while the DiT stays on the card; match the base to cover `_and_qfloat8`.
te_wants_offload = wants_offload or GPU_memory_mode.startswith("text_encoder_offload")

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        fp32_modules = [m for m in transformer.modules()
                        if any(p.dtype == torch.float32 for p in m.parameters(recurse=False))]
        shard_fn = partial(shard_model, device_id=device, param_dtype=None, cast_dtype=False,
                          module_to_wrapper=list(transformer.transformer_blocks),
                          ignored_modules=fp32_modules, offload_to_cpu=wants_offload)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype,
                          module_to_wrapper=list(text_encoder.model.language_model.layers),
                          offload_to_cpu=te_wants_offload)
        pipeline.text_encoder.model = shard_fn(pipeline.text_encoder.model)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
    print("Add Compile")

# Quantize (when the mode carries an "_and_<quant>" suffix) and then place the pipeline.
# The order lives inside the helper: quantization has to happen before the offload hooks
# are installed.
# The FP8 conversion above has already run (before the FSDP sharding, on purpose); only the dequant
# wrapper and the memory placement are left, which is what the preconverted tag installs.
apply_gpu_memory_mode(pipeline, GPU_memory_mode, device, weight_dtype,
                      quant_tag="qfloat8_preconverted" if GPU_memory_mode.endswith("_and_qfloat8") else None,
                      exclude_module_name=[])

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)


def parse_reference(entry: str):
    kind, _, media = entry.partition("=")
    kind, media = kind.strip().lower(), media.strip()
    if not media:
        raise ValueError(f"A reference entry must be `image=path`, `video=path` or `audio=path`, got {entry!r}.")
    if kind == "image":
        return MiniMaxH3ImageReference.from_file(media)
    if kind == "video":
        return MiniMaxH3VideoReference.from_file(media)
    if kind == "audio":
        return MiniMaxH3AudioReference.from_file(media)
    raise ValueError(f"A reference entry must start with `image=`, `video=` or `audio=`, got {entry!r}.")


# Decode every reference at the rate its container carries, which the pipeline's setup resamples onto MiniMax-H3's
# own 24 fps and the audio VAE's sample rate.
parsed_references = [parse_reference(entry) for entry in references]

pdd_callback = None if pdd_config is None else pdd_step_callback(
    transformer, scheduler, audio_scheduler, pdd_config, num_inference_steps
)

with torch.no_grad():
    output = pipeline(
        prompt=prompt,
        references=parsed_references,
        height=None if sample_size is None else sample_size[0],
        width=None if sample_size is None else sample_size[1],
        num_frames=video_length,
        num_inference_steps=num_inference_steps,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
        callback_on_step_end=pdd_callback,
    )
print(f"[{os.environ.get('RANK', '0')}] generation done, decoding", flush=True)

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

sample = output.videos
audio = output.audio
audio_sample_rate = output.sampling_rate

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_videos_with_audio_grid(sample, audio, video_path, fps=fps, audio_sample_rate=audio_sample_rate)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
    # Keep every rank alive until the saving rank finishes; an early exit of one rank makes the elastic launcher
    # terminate the others.
    dist.barrier()
else:
    save_results()

```


---

## FILE: config/minimax_h3/minimax_h3_control.yaml (https://github.com/aigc-apps/VideoX-Fun/blob/HEAD/config/minimax_h3/minimax_h3_control.yaml)

```
format: diffusers
pipeline: minimax-h3
transformer_additional_kwargs:
    control_blocks_places: [0, 10, 20, 30, 40]
    control_in_dim: 49
    control_apply_audio: false
```


---

# ISSUES in aigc-apps/VideoX-Fun matching "H3" (top 10 by comments)


### #507 [open] h3 fun control net  comfyui workflow please
URL: https://github.com/aigc-apps/VideoX-Fun/issues/507  by sreekar 2026-08-24T10:15:15Z  comments: 1



> **Comment by bubbliiiing (2026-08-26T07:17:52Z):**

https://github.com/Comfy-Org/ComfyUI/issues/15850
Sorry, we’re looking forward to official support, since our ComfyUI nodes aren’t very well done…



### #510 [open] Requests for comfyui example workflow for brand new MinMax H3 Controlnet
URL: https://github.com/aigc-apps/VideoX-Fun/issues/510  by irrikrlla 2026-08-25T04:03:42Z  comments: 1

Incredible work on the H3 ControlNet-Union! We at Unsounded Lab are currently building a high-end AI-native cinematic pipeline, and this model is exactly the game-changer we need for digital human choreography and complex physical interactions.
However, it seems the current VideoX-Fun node doesn't fully support this H3 checkpoint yet. Could you kindly share an official ComfyUI example workflow, or advise if an update to the node is on the roadmap?
Integrating this into the ComfyUI ecosystem would massively accelerate high-end visual storytelling. Looking forward to your insights!

> **Comment by bubbliiiing (2026-08-26T07:13:25Z):**

https://github.com/Comfy-Org/ComfyUI/issues/15850
Actually, I’d like official support, because our ComfyUI nodes aren’t very well done…



### #509 [open] Where is examples/minimax_h3_fun/predict_v2v_control_inpaint.py ?
URL: https://github.com/aigc-apps/VideoX-Fun/issues/509  by rodjjo 2026-08-25T01:22:09Z  comments: 1

Hello, Nice work!


You mentioned it at https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union

Inpainting is supported: the control input is widened to control_in_dim = 49 (latent + masked latent + mask channels); use examples/minimax_h3_fun/predict_v2v_control_inpaint.py.

> **Comment by bubbliiiing (2026-08-26T07:14:38Z):**

This version’s inpainting isn’t very good… so I’m still training a new version and haven’t provided the corresponding .py file.





### #518 [closed] minimax-h3 train error
URL: https://github.com/aigc-apps/VideoX-Fun/issues/518  by stihuangyuan 2026-09-23T04:41:18Z  comments: 0

[rank1]:   File "VideoX-Fun-0916/scripts/minimax_h3/train.py", line 292, in encode_prompt
[rank1]:     outputs = text_encoder.model(**encoder_kwargs)
[rank1]:               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 851, in forward
[rank1]:     output = self._fsdp_wrapped_module(*args, **kwargs)
[rank1]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/transformers/utils/generic.py", line 939, in wrapper
[rank1]:     output = func(self, *args, **kwargs)
[rank1]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 1180, in forward
[rank1]:     image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(
[rank1]:                                                           ^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/transformers/utils/generic.py", line 993, in wrapper
[rank1]:     return func(*args, **translated)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/transformers/utils/generic.py", line 939, in wrapper
[rank1]:     output = func(self, *args, **kwargs)
[rank1]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 1053, in get_image_features
[rank1]:     pixel_values = pixel_values.type(self.visual.dtype)
[rank1]:                                      ^^^^^^^^^^^^^^^^^
[rank1]:   File "/opt/conda/lib/python3.11/site-packages/transformers/modeling_utils.py", line 918, in dtype
[rank1]:     return next(param.dtype for param in self.parameters() if param.is_floating_point())
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: StopIteration
[rank2]: Traceback (most recent call last):
[rank2]:   File "VideoX-Fun-0916/scripts/minimax_h3/train.py", line 2130, in <module>
[rank2]:     
[rank2]:   File "VideoX-Fun-0916/scripts/minimax_h3/train.py", line 1757, in main
[rank2]:     )
[rank2]:      
[rank2]:   File "VideoX-Fun-0916/scripts/minimax_h3/train.py", line 292, in encode_prompt
[rank2]:     outputs = text_encoder.model(**encoder_kwargs)
[rank2]:               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
[rank2]:     return self._call_impl(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
[rank2]:     return forward_call(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 851, in forward
[rank2]:     output = self._fsdp_wrapped_module(*args, **kwargs)
[rank2]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
[rank2]:     return self._call_impl(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
[rank2]:     return forward_call(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/transformers/utils/generic.py", line 939, in wrapper
[rank2]:     output = func(self, *args, **kwargs)
[rank2]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 1180, in forward
[rank2]:     image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(
[rank2]:                                                           ^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/transformers/utils/generic.py", line 993, in wrapper
[rank2]:     return func(*args, **translated)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/transformers/utils/generic.py", line 939, in wrapper
[rank2]:     output = func(self, *args, **kwargs)
[rank2]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py", line 1053, in get_image_features
[rank2]:     pixel_values = pixel_values.type(self.visual.dtype)
[rank2]:                                      ^^^^^^^^^^^^^^^^^
[rank2]:   File "/opt/conda/lib/python3.11/site-packages/transformers/modeling_utils.py", line 918, in dtype
[rank2]:     return next(param.dtype for param in self.parameters() if param.is_floating_point())


### #516 [open] minimax h3 我想请问下训练lora 会不会跟diffsynstudio 一样的问题，训练的效果很差，比如cfg dis 的问题影响，有验证过效果吗
URL: https://github.com/aigc-apps/VideoX-Fun/issues/516  by 01-111162-173 2026-09-04T16:25:38Z  comments: 0




---

## README.md excerpts mentioning H3 (https://github.com/aigc-apps/VideoX-Fun/blob/main/README.md)

329-| CogVideoX-Fun V1.0 | Video | Legacy weights trained at 49 frames 8fps, superseded by V1.1/V1.5 | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-2b-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-2b-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-2b-InP">🤖</a></td><td valign="top" style="padding:2px 0;">Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second.</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">CogVideoX-Fun-5b-InP</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/CogVideoX-Fun-5b-InP">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/CogVideoX-Fun-5b-InP">🤖</a></td><td valign="top" style="padding:2px 0;">Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second.</td></tr></table> |
330-| HunyuanVideo | Video | Official diffusers-format weights; this project directly supports inference and LoRA training | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">HunyuanVideo</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/hunyuanvideo-community/HunyuanVideo">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo">🤖</a></td><td valign="top" style="padding:2px 0;">文生视频</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">HunyuanVideo-I2V</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo-I2V">🤖</a></td><td valign="top" style="padding:2px 0;">图生视频</td></tr></table> |
331:| MiniMax-H3 | Video | Official video generation weights and the ControlNet trained by this project | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">MiniMax-H3</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/MiniMax/MiniMax-H3">🤖</a></td><td valign="top" style="padding:2px 0;">Official MiniMax-H3 T2V/I2V weights</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">MiniMax-H3-Fun-Controlnet-Union</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/MiniMax-H3-Fun-Controlnet-Union">🤖</a></td><td valign="top" style="padding:2px 0;">ControlNet trained by this project, supports multiple control conditions and trajectory control</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">MiniMax-H3-Fun-Controlnet-Union-2.0</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/PAI/MiniMax-H3-Fun-Controlnet-Union-2.0">🤖</a></td><td valign="top" style="padding:2px 0;">ControlNet trained by this project (2.0), supporting multiple control conditions, trajectory control, and inpaint checkpoints</td></tr></table> |
332:| TaoMate-H3 | Video+Audio | Official streaming audio-video generation adapter built on MiniMax-H3 | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">TaoMate-H3-Adapter</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/TaoLiveAIGC/TaoMate-H3">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/TaoLiveAIGC/TaoMate-H3">🤖</a></td><td valign="top" style="padding:2px 0;">Official rank-128 adapter (step-3000 EMA) with a built-in 3-step distilled schedule for streaming speech-driven generation; requires the MiniMax-H3 base weights</td></tr></table> |
333-| LTX-2 | Video+Audio | Official DiT audio-video joint generation weights | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">LTX-2</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/Lightricks/LTX-2">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/Lightricks/LTX-2">🤖</a></td><td valign="top" style="padding:2px 0;">Official audio-video joint generation weights; repo contains multiple precision files</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">LTX-2.3-Diffusers</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/dg845/LTX-2.3-Diffusers">🤗</a></td><td valign="top" style="padding:2px 8px;">-</td><td valign="top" style="padding:2px 0;">v2.3 requires community-converted diffusers weights; see Lightricks/LTX-2.3 for official weights</td></tr></table> |
334-| LongCat-Video | Video | Official long-video generation weights; supports LoRA training | <table style="width:100%;border-collapse:collapse;"><tr><td valign="top" style="padding:2px 8px 2px 0;">LongCat-Video</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/meituan-longcat/LongCat-Video">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/meituan-longcat/LongCat-Video">🤖</a></td><td valign="top" style="padding:2px 0;">Official LongCat-Video T2V weights</td></tr><tr><td valign="top" style="padding:2px 8px 2px 0;">LongCat-Video-Avatar</td><td valign="top" style="padding:2px 8px;"><a href="https://huggingface.co/meituan-longcat/LongCat-Video-Avatar">🤗</a></td><td valign="top" style="padding:2px 8px;"><a href="https://modelscope.cn/models/meituan-longcat/LongCat-Video-Avatar">🤖</a></td><td valign="top" style="padding:2px 0;">Official LongCat-Video avatar/digital-human weights</td></tr></table> |
--
350-
351-> Notes:
352:> - Audio-driven and reference models (FantasyTalking, InfiniteTalk, Phantom, TaoMate-H3) are incremental weights and must be used together with the corresponding base video weights and audio encoder.
353-> - The TurboWan weights released by the TurboDiffusion scheme are listed above; other distillation schemes such as Flex-Forcing and PDD have no publicly released weights — train them following `scripts/{model_name}/README_TRAIN*.md` and then fill the resulting path into `transformer_path`.
354-> - Weight names map one-to-one to folder names under `models/Diffusion_Transformer/`. Weights within the same family are not interchangeable; choose according to the inference task. If a weight is not listed here, it is either produced by this project or should be obtained from the upstream official repository.
