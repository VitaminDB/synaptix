# OpenVDN__vdn-minimax-h3

Source: https://github.com/OpenVDN/vdn-minimax-h3
Fetched: 2026-11-28

Description: VideoDeltaNet-H3: Live T2VA / I2VA / FL2VA / Ref2VA-like generation based on Minimax H3.
Stars: 700  Updated: 2026-10-05T04:45:20Z



---

## FILE: README.md (https://github.com/OpenVDN/vdn-minimax-h3/blob/HEAD/README.md)

# Video DeltaNet: Hybrid Attention to Speed Up Video Models with Near-Lossless Quality

[[`📄 Paper`](https://arxiv.org/abs/2609.20744)] [[`Blog`](https://openvdn.github.io/)] [[`Code`](https://github.com/OpenVDN/vdn-minimax-h3)] [[`🤗 Weights`](https://huggingface.co/OpenVDN/vdn-minimax-h3)] [[`ModelScope`](https://www.modelscope.ai/models/OpenVDN/vdn-minimax-h3)] [[`Local Inference`](https://github.com/FlashML-org/FreeVideo)] [[`License`](#license)]

We release **VDN-Minimax-H3** (**VDN-H3**), a hybrid-attention model that generates video faster than it plays, powered by [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). It offers these key features:

- **Fast inference:** With SGLang Diffusion on 8×B200 GPUs, VDN-H3 generates a 14.4-second clip in about **9.0 seconds end-to-end**, including **6.9 seconds for denoising**, using 8 denoising steps.
- **Hybrid Architecture:** We propose a hybrid-attention architecture: one frame-wise linear attention branch that is highly efficient, and a softmax branch that maintains the backbone's visual quality and consistency.
- **Plug-and-Play:** The checkpoint adds a separate linear attention branch and two small LoRA adapters that can be merged into the backbone during inference without touching the backbone weights.
- **Fully open-source:** We don't just open-source the weights. The optimized inference stack and its corresponding training code are released together.

We present some samples of generated videos here:

<table>
<tr>
<td width="33%"><video src="https://github.com/user-attachments/assets/343d03e6-8d88-444d-b66c-19d9fb252b37" controls muted></video></td>
<td width="33%"><video src="https://github.com/user-attachments/assets/9de4ffbc-1a38-4467-8c9e-c975d2b84fa1" controls muted></video></td>
<td width="33%"><video src="https://github.com/user-attachments/assets/f0ec47d4-96ad-4fbb-8bc9-1b9d5eeb0b94" controls muted></video></td>
</tr>
<tr>
<td width="33%"><video src="https://github.com/user-attachments/assets/bf849387-03c6-45ef-988b-38164f27e88a" controls muted></video></td>
<td width="33%"><video src="https://github.com/user-attachments/assets/2cc62303-1ebb-4d3f-a367-3c0125b11c3d" controls muted></video></td>
<td width="33%"><video src="https://github.com/user-attachments/assets/2e0f43d3-4919-44c4-9f64-42d90d9e08b1" controls muted></video></td>
</tr>
</table>

## News

- **October 2, 2026:** [FreeVideo](https://github.com/FlashML-org/FreeVideo), built on VDN-H3, runs MiniMax H3 on consumer GPUs with as little as 8 GB of VRAM.
- **September 17, 2026:** The [Video DeltaNet paper](https://arxiv.org/abs/2609.20744) is available on arXiv.
- **September 17, 2026:** We support the [Ref2VA-like](#supporting-ref2va-like-task) task now with the same checkpoint: reference images through the FL2VA weights.
- **September 14, 2026:** [SGLang Diffusion](https://github.com/sgl-project/sglang) now supports VDN-H3, achieving its fastest reported performance: 6.9 seconds for denoising and about 9.0 seconds end-to-end for a 14.4-second video on 8×B200 GPUs. See the [MiniMax-H3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx#7-vdn-h3-hybrid-attention-8-step-distill) for more details.
- **September 8, 2026:** We support I2VA, FL2VA and L2VA now with the same checkpoint.
- **September 6, 2026:** We released the [VDN-H3 blog](https://openvdn.github.io/), [training and inference code](https://github.com/OpenVDN/vdn-minimax-h3), and [model weights](https://huggingface.co/OpenVDN/vdn-minimax-h3).

## Set up the environment

**Before you begin**, please read the [license](#license) before downloading or running VDN-H3.

1. Clone the VDN-H3 repository from GitHub.

```bash
git clone https://github.com/OpenVDN/vdn-minimax-h3.git
cd vdn-minimax-h3
```

2. Create the environment. We recommend PyTorch 2.13 (`torch.__version__` = `2.13.0+cu129`) and installing FlashAttention 4: on Hopper and data-center Blackwell the window softmax runs on its kernels, through [FlexAttention's Flash backend](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) or directly. On Ampere, Ada and consumer Blackwell (RTX 50 series), or without it, the window softmax runs on FlexAttention's Triton kernel and on PyTorch's own varlen attention.

```bash
conda create -n vdn python=3.12 -y
conda activate vdn
pip install uv

uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
```

3. Install the other packages shown in `pyproject.toml`, `flash-attn-4` included (`--prerelease=allow` is needed for its pre-release `nvidia-cutlass-dsl` dependency); on Ampere, Ada and consumer Blackwell neither it nor the regular `flash-attn` is used, as attention there runs on PyTorch's own kernels.

```bash
uv pip install --prerelease=allow -e .
```

4. Install the patched Diffusers. The setup script handles everything:

```bash
bash scripts/setup_diffusers.sh
```

5. Read [H3-Context-IR](https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir) and the official [prompt-writing skills](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills). Rewriting your prompt with them before encoding it can greatly improve the generated video quality for VDN-H3 inference.

## Quick Start — Generate your own video

### Inference with Diffusers

The quickest way to a first render is using diffusers, as we already release the checkpoints as modular diffusers components:

```python
import torch
from accelerate import cpu_offload_with_hook
from diffusers import ModularPipeline
from diffusers.hooks import apply_group_offloading

pipe = ModularPipeline.from_pretrained("OpenVDN/vdn-minimax-h3", workflow="t2va")
pipe.load_components(trust_remote_code=True, torch_dtype=torch.bfloat16)

apply_group_offloading(pipe.text_encoder, onload_device="cuda", offload_type="leaf_level",
                       use_stream=True)
_, vae = cpu_offload_with_hook(pipe.vae, execution_device="cuda")
cpu_offload_with_hook(pipe.audio_vae, execution_device="cuda", prev_module_hook=vae)
pipe.transformer.to("cuda")

out = pipe(prompt="a prompt", num_frames=345, num_inference_steps=9,
           output=["videos", "audio", "sampling_rate"])
```

`num_inference_steps` counts sigma grid points, so 9 of them is 8 model evaluations.

Or as a script, keyframes included:

```bash
python src/inference/infer_diffusers.py "a prompt" --out results/diffusers.mp4
python src/inference/infer_diffusers.py "a prompt" \
    --first prompts/image/first.png --last prompts/image/last.png
```

On a 24 GB card, stream the transformer in one block at a time: swap `pipe.transformer.to("cuda")` for the line below, or add `--offload_dit` to the script. 345 frames then peak at 22 GB, 20 in fp8.

```python
apply_group_offloading(pipe.transformer, onload_device="cuda", offload_type="block_level",
                       num_blocks_per_group=1, use_stream=True)
```

Offloading granularity, the window-softmax backend, fp8 and your own quantization config are described in [docs/diffusers.md](docs/diffusers.md).

### Inference with SGLang

[SGLang Diffusion](https://github.com/sgl-project/sglang) provides native VDN-H3 serving for T2VA and FL2VA. Install SGLang and launch the 8×B200 configuration below. Set `--num-gpus` to 1, 2, or 4 for the corresponding smaller configuration:

```bash
uv pip install "sglang[diffusion]" --prerelease=allow

sglang serve \
  --model-path OpenVDN/vdn-minimax-h3 \
  --num-gpus 8 \
  --quantization fp8 \
  --attention-backend hybrid_window_attn_h3 \
  --performance-mode speed \
  --warmup-num-frames 345 \
  --warmup-resolutions 1344x768 \
  --port 30010
```

With MXFP8, denoising takes 47.6, 25.9, 13.1, and 6.9 seconds on 1, 2, 4, and 8 B200 GPUs, respectively. The 8-GPU configuration returns the finished 14.4-second video in about 9.0 seconds end-to-end after warm-up. See the [MiniMax-H3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx#7-vdn-h3-hybrid-attention-8-step-distill) for more details.

### Inference with our repository

The configs, kernels, fp8, the eight-GPU layout and the prompt caches are described in [docs/inference.md](docs/inference.md).

#### Download the weights

To render through this repository's own stack instead -- fp8, the tuned kernels, and Ulysses across eight GPUs, which is where the numbers in [Results](#results) come from -- download everything (about 82 GB) from [Hugging Face](https://huggingface.co/OpenVDN/vdn-minimax-h3) into `ckpts/` using

```bash
hf download OpenVDN/vdn-minimax-h3 --local-dir ckpts
```

or from [ModelScope](https://www.modelscope.ai/models/OpenVDN/vdn-minimax-h3) with

```bash
modelscope download --model OpenVDN/vdn-minimax-h3 --local_dir ckpts
```

The layout will look like

```text
ckpts/
  h3-base/             the released MiniMax H3: transformer, video and audio VAEs, schedulers · 72 GB
  stage-b-step-2000/   VDN-H3-50-step: linear_branch/ + adapters/default/ LoRA · 4.3 GB
  stage-dmd-step-250/  VDN-H3-8-step: the above + adapters/turbo/ · 5.1 GB
```

#### Render a video

Then, run the following script:

```bash
bash scripts/inference/8nfe_tuned_fp8.sh
```

Note that the first run needs to compile all of the kernels, which might take several minutes. Later runs can reuse the cache.

#### Use your own prompt

We provide [three examples](prompts/README.md) and encode them using the Qwen3-VL-32B VLM. For your own prompt, you should first encode it using the VLM, then render it through the main diffusion model:

```bash
python src/inference/encode_prompt.py --prompt "..." --out prompts/mine.pt

python src/inference/infer.py \
  --config configs/inference/8nfe_tuned_fp8.yaml \
  checkpoint=ckpts/stage-dmd-step-250 \
  render.prompt_file=prompts/mine.pt \
  render.out=results/mine.mp4
```

#### Supporting FL2VA, I2VA, and L2VA

The same checkpoints also generate from keyframes. We provide an FL2VA example in [prompts/image/](prompts/image/):

<table>
<tr>
<td width="50%"><img src="prompts/image/first.png" alt="first keyframe"></td>
<td width="50%"><img src="prompts/image/last.png" alt="last keyframe"></td>
</tr>
<tr>
<td align="center"><code>prompts/image/first.png</code></td>
<td align="center"><code>prompts/image/last.png</code></td>
</tr>
</table>

Render it with:

```bash
python src/inference/infer.py \
  --config configs/inference/8nfe_tuned_fp8.yaml \
  checkpoint=ckpts/stage-dmd-step-250 \
  render.prompt_file=prompts/image/example_fl2va.pt \
  render.out=results/example_fl2va.mp4
```

and you should get something like this:

<video src="https://github.com/user-attachments/assets/56728e17-9081-4f5a-b707-54de1cd8c166" controls muted></video>

For your own keyframes, encode the prompt together with the images first:

```bash
python src/inference/encode_keyframes.py --prompt "..." \
  --first first.png --last last.png --out prompts/image/mine.pt
```

`--first` alone is I2VA, `--last` alone is L2VA, both is FL2VA. Each mode wants its own instruction as the prompt's first line, given by MiniMax-H3's [prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).

The multi-GPU entrypoint takes the same prompt file:

```bash
torchrun --standalone --nproc_per_node=8 src/inference/infer_ulysses.py \
  --config configs/inference/8nfe_tuned_fp8_ulysses_h200.yaml \
  checkpoint=ckpts/stage-dmd-step-250 \
  render.prompt_file=prompts/image/example_fl2va.pt \
  render.out=results/example_fl2va.mp4
```

#### Supporting Ref2VA-Like Task

The MiniMax-H3 FL2VA checkpoints have zero-shot ability to take reference images and perform the way the MiniMax-H3 Ref2VA checkpoints generate videos. Each reference image appears in the prompt as a `<Picture i>` entry and is accompanied by its own block of conditioning rows placed before the video tokens.

Generation still uses the FL2VA weights rather than MiniMax-H3's separate Ref2VA transformer, which is why we describe this mode as Ref2VA-like rather than Ref2VA proper.

An example is provided in [prompts/reference/](prompts/reference/), using six reference images that cover two characters, a Samoyed, and a cafe:

<table>
<tr>
<td width="16%"><img src="prompts/reference/ref-1.png" alt="Picture 1"></td>
<td width="16%"><img src="prompts/reference/ref-2.png" alt="Picture 2"></td>
<td width="16%"><img src="prompts/reference/ref-3.png" alt="Picture 3"></td>
<td width="16%"><img src="prompts/reference/ref-4.png" alt="Picture 4"></td>
<td width="16%"><img src="prompts/reference/ref-5.png" alt="Picture 5"></td>
<td width="16%"><img src="prompts/reference/ref-6.png" alt="Picture 6"></td>
</tr>
<tr>
<td align="center"><code>ref-1.png</code></td>
<td align="center"><code>ref-2.png</code></td>
<td align="center"><code>ref-3.png</code></td>
<td align="center"><code>ref-4.png</code></td>
<td align="center"><code>ref-5.png</code></td>
<td align="center"><code>ref-6.png</code></td>
</tr>
</table>

Render it with:

```bash
python src/inference/infer.py \
  --config configs/inference/8nfe_tuned_fp8.yaml \
  checkpoint=ckpts/stage-dmd-step-250 \
  render.prompt_file=prompts/reference/example_ref2va.pt \
  render.out=results/example_ref2va.mp4
```

and you should get something like this:

<video src="https://github.com/user-attachments/assets/99b02319-4f20-49c6-86e7-bb0ed8a48255" controls muted></video>

For your own references, encode the prompt together with the images, in the order the prompt numbers them:

```bash
python src/inference/encode_keyframes.py --prompt "..." \
  --refs ref-1.png ref-2.png ref-3.png --out prompts/reference/mine.pt
```

Each reference is put on a 768-pixel short edge (`--ref_size`). The prompt defines its subjects by picture, in the format of MiniMax-H3's [reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

#### Choosing an inference configuration

We support both single-GPU and multi-GPU inference for the released model. Single-GPU scripts auto-detect the best kernels for your GPU. Multi-GPU scripts vary for different hardware (H200, B200) to achieve the best performance.

```bash
bash scripts/inference/8nfe_tuned_fp8.sh                # one GPU
bash scripts/inference/8nfe_tuned_fp8_ulysses_h200.sh   # eight H200s, one node
bash scripts/inference/8nfe_tuned_fp8_ulysses_b200.sh   # eight B200s, one node
```

## Results

Our fastest reported result uses SGLang Diffusion on 8×B200 GPUs: 6.9 seconds for denoising and about 9.0 seconds end-to-end for a 768p, 14.4-second video. The tables below compare its steady-state denoising speed with our reference inference pipeline on H200s and B200s:

**H200:**

| Configuration | GPUs | Seconds/NFE | 50 NFE (VDN-H3-50-step) | 8 NFE (VDN-H3-8-step) |
|---|---:|---:|---:|---:|
| dense MiniMax H3 | 1 | 32.7 | 27.3 min | 4.4 min |
| VDN-H3 FP8 | 1 | 11.2 | 9.4 min | 90.5 s |
| VDN-H3 FP8 Distributed | 8 | 2.29 | 1.9 min | 18.3 s |

**B200:**

| Configuration | GPUs | Seconds/NFE | 50 NFE (VDN-H3-50-step) | 8 NFE (VDN-H3-8-step) |
|---|---:|---:|---:|---:|
| dense MiniMax H3 (cuDNN) | 1 | 16.74 | 13.95 min | 2.23 min |
| VDN-H3 FP8 | 1 | 6.41 | 5.3 min | 51 s |
| VDN-H3 FP8 Distributed | 8 | 1.40 | 1.2 min | 11.23 s |
| SGLang Diffusion, VDN-H3 FP8 Distributed | 8 | **0.88** | **44.0 s** | **6.9 s** |

We exclude model loading, warm-up, VAE decoding, and MP4 encoding. For a live setup, we recommend running the text prompt rewriter, VAE decoding, and MP4 conversion on separate machines, so the eight GPUs only denoise.

## Training Recipe

VDN-H3 is trained in three stages based on the frozen dense model, each starting from the previous stage's final checkpoint. We additionally include a DMD training stage to align with the community [few-step distillation LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora). The training scripts are located in `src/training/` and `scripts/training/`, and all training configurations can be found under `configs/training/`.

| Stage | What it trains | Steps |
|---|---|---:|
| A1 | the new linear-attention branch, aligned per layer to the dense model | 200 |
| A2 | the same parameters, end-to-end | 500 |
| B | a LoRA on the QKV and O projections plus the linear branch | 2000 |
| DMD | the 8-step `turbo` LoRA, by DMD2 (no GAN) | 250 |

To reproduce the training process, run:

```bash
bash scripts/training/stage_a1.sh       data.index_file=/path/to/video_index.jsonl
bash scripts/training/stage_a2.sh       data.index_file=/path/to/video_index.jsonl
bash scripts/training/stage_b.sh        data.index_file=/path/to/video_index.jsonl
bash scripts/training/stage_dmd_vdn.sh  data.index_file=/path/to/video_index.jsonl
```

### Data Preprocess

The trainers read pre-encoded video latents, audio latents, and text latents, following the H3 standard pipeline. Captions should be written in the same format inference expects.

The preprocessed data should be placed as follows, with a `video_index.jsonl` carrying the metadata:

```
<root>/
├── video_index.jsonl       one JSON row per clip, carrying its "latent_path"
├── video/
│   ├── 00000.pt            (24, 102, 48, 84) bf16 — video VAE, normalized space
│   └── ...
├── audio/
│   ├── 00000.pt            (2, 32, 575) bf16 — audio VAE, stereo, 40 latents/s
│   └── ...
└── text/
    ├── 00000.pt            {"prompt_embeds": (L, 5120) bf16,
    │                        "text_token_tags": (L,) int64}
    └── ...
```

`data.index_file` names the jsonl. Only `latent_path` is read from a row; the audio and text sidecars are found by path arithmetic — same file name, sibling directory — so `latent_path` must end in `video/<name>.pt`. The reader is `src/training/dataset_h3_latents.py`.

### Stage-DMD

Stage-DMD is data-free, only requiring the text rows. Its `turbo` LoRA adapter is initialized from [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora), which the config expects in `ckpts/external/`:

```bash
hf download larryvrh/MiniMax-H3-Turbo-Lora \
  minimax_h3_turbo_v4_step600_ema.safetensors --local-dir ckpts/external
```

## Acknowledgement

VDN-H3 is built on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and starts from its released transformer weights. We also thank [Diffusers](https://github.com/huggingface/diffusers), [FlashAttention](https://github.com/Dao-AILab/flash-attention), and [Triton](https://github.com/triton-lang/triton), on which the optimized inference path is built. We thank [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents) for kernel design support. We also thank [Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention) and [FlexAttention](https://pytorch.org/docs/stable/nn.attention.flex_attention.html) for their open-source attention implementations.

## BibTeX

```bibtex
@misc{xi2026videodeltanet,
  title  = {VideoDeltaNet on MiniMax H3},
  author = {Haocheng Xi and Yiming Xie and Hexu Zhao and Yiwen Zhang and Michael Liu and Thomas Creavin and Kurt Keutzer and Xiuyu Li and Zhaoyang Lv and Chenfeng Xu and Haiwen Feng},
  year   = {2026},
  url    = {https://openvdn.github.io/}
}
```

---

*VDN-Minimax-H3 · Independent architecture study · 2026*

## License

This repository contains the VDN-H3 training and inference code, which is licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 the VDN authors.

**The model weights are not in this repository and are not covered by that license.** VDN-H3 is a derivative of MiniMax H3, and its weights are distributed separately at [huggingface.co/OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3) under the [MiniMax H3 Community License Agreement](licenses/MiniMax-H3-Community-License-Agreement.txt).


---

## FILE: assets/Logo/README.md (https://github.com/OpenVDN/vdn-minimax-h3/blob/HEAD/assets/Logo/README.md)

# Video DeltaNet logo exports

## Variants

- `mark`: the VDN symbol only, without the `VIDEO DELTA NET` line.
- `lockup`: the VDN symbol and the fixed `VIDEO DELTA NET` wordmark.

## Canvas formats

- `square/`: original 1254 × 1254 exports, retained for social thumbnails and source compatibility.
- `cropped/`: production-ready rectangular exports, tightly cropped with a consistent 48 px safety margin.
- `source/`: the HTML source used to generate the logo family.
- `legacy/`: the previous flat exports, preserved under their original filenames.

Each `mark` and `lockup` folder contains both `black-on-white` and `white-on-black` versions.

The older flat files are retained in `legacy/` for compatibility. In particular, `vdn-logo-black-on-white.png` is identical to the black-on-white lockup, and `vdn-logo-white-on-black.png` is identical to the white-on-black lockup.


---

## FILE: docs/diffusers.md (https://github.com/OpenVDN/vdn-minimax-h3/blob/HEAD/docs/diffusers.md)

# The diffusers component

Each checkpoint on the Hub carries a `diffusers/` directory, so the VDN-H3 transformer loads as a component of diffusers' `ModularPipeline`. This page covers its options and the matching flags of [src/inference/infer_diffusers.py](../src/inference/infer_diffusers.py). The basic usage is in the [README](../README.md#inference-with-diffusers); this repository's own, faster stack is in [inference.md](inference.md).

## What gets loaded

`diffusers/` holds a `config.json` and one file of remote code, `modeling_vdn_h3.py`, and no weights. Loading assembles three things that live elsewhere in the same Hub repository: the 66 GB MiniMax-H3 transformer under `h3-base/`, the checkpoint's linear branch, and its LoRA adapters, which are merged into the base weights. The base loads in bf16.

The pipeline index names the 8-step model. The 50-step model is selected per component:

```python
pipe.load_components(trust_remote_code=True, torch_dtype=torch.bfloat16,
                     subfolder={"transformer": "stage-b-step-2000/diffusers"})
out = pipe(prompt="a prompt", num_frames=345, num_inference_steps=51, ...)
```

`load_components` turns a component's load error into a warning and leaves `pipe.transformer` as `None`. The traceback printed with that warning names the cause.

## Steps and frames

`num_inference_steps` counts sigma grid points, the terminal 0 included, so it is one more than the number of model evaluations: 9 for the 8-step model, 51 for the 50-step one. The script's `--steps` takes model evaluations and adds the one.

`num_frames` runs from 120 to 360, 5 to 15 seconds at 24 fps, and is snapped up to the next 17n+5. 345 frames are the 14.4 seconds every reported number uses.

## Offloading

The README snippet offloads everything but the transformer. The 62 GB Qwen3-VL text encoder streams onto the GPU one layer at a time (`leaf_level`), and each decoder comes in whole through accelerate's hook while it runs. The transformer stays on the GPU, or streams in one block at a time:

```python
apply_group_offloading(pipe.transformer, onload_device="cuda", offload_type="block_level",
                       num_blocks_per_group=1, use_stream=True)
```

The transformer can be offloaded per model or per block, but not per leaf: its fused kernels read a child module's weights without calling the child, so a leaf's hook never fires.

Measured on an H200 at 345 frames and 8 model evaluations, the streamed rows with the CUDA allocator capped at 24 GB:

| Transformer | Seconds per evaluation | Peak GPU memory |
|---|---:|---:|
| bf16, on the GPU | 14.3 | 85 to 103 GB, by prompt length |
| fp8, on the GPU | 12.1 | 65 GB |
| bf16, streamed per block | 16.2 | 22 GB |
| fp8, streamed per block | 13.1 | 20 GB |

The decode phase peaks at 19 GB: the 9.8 GB video VAE plus the 345 decoded frames.

## Window softmax backend

The component's window softmax runs on FlexAttention, which is what fits a 24 GB card. `softmax_backend` selects the decomposition this repository's own scripts use instead. It is faster on B200, and its transformer peak when streamed is 28 to 30 GB.

```python
pipe.load_components(trust_remote_code=True, torch_dtype=torch.bfloat16,
                     softmax_backend={"transformer": "decomposed"})
```

Both backends pick their kernel by the card:

| Card | `flex` | `decomposed` |
|---|---|---|
| Hopper, data-center Blackwell (sm90, sm100, sm110) | FlexAttention's Flash backend (FA4) | FA4's varlen kernel, cuDNN SDPA |
| Ampere, Ada, consumer Blackwell (sm8x, sm120) | FlexAttention's Triton kernel | PyTorch's `varlen_attn`, SDPA's own kernels |

## Quantization

Quantization is torchao's (`pip install torchao`) and runs last, after the LoRA merge, because a quantized weight cannot take the merge. `quantization_config` therefore accepts a `TorchAoConfig` only. Any other backend raises within a second, before the 66 GB load, and so does `fp8=True` together with a config.

### fp8

```python
pipe.load_components(trust_remote_code=True, torch_dtype=torch.bfloat16,
                     fp8={"transformer": True})
```

`fp8`, or `--fp8` for the script, is a preset `TorchAoConfig`: every Linear at least 4096 wide on both sides goes to fp8 e4m3 with dynamic activation scales. That is 363 Linears: the qkv and output projections, the feed-forward pair and the linear branch's readout. Narrow Linears stay in bf16. Scales are per row on sm90 and per tensor on sm100 and above, which is where each card's fast GEMM is. The transformer's weights drop from 62 GB to 45.

Each quantized Linear calls one shared `torch.compile`d `F.linear`, so the first evaluation compiles. The quantized weights are ordinary parameters, and diffusers' group offloading streams them as they are.

fp8 needs compute capability 9.0 or above. It changes the sample: the same seed gives a different video than bf16, of the same quality.

### A config of your own

```python
from diffusers import TorchAoConfig
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

pipe.load_components(
    trust_remote_code=True, torch_dtype=torch.bfloat16,
    quantization_config={"transformer": TorchAoConfig(
        Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))},
)
```

A config without a `modules_to_not_convert` of its own leaves the same narrow Linears in bf16. A list of your own is applied as given, plus the modules the model keeps in fp32. A Float8 dynamic-activation config that sets no `activation_value_lb` gets the preset's floor on the activation amax. The linear branch skips the anchor frames, so rows of exact zeros reach its readout, and an unfloored per-row scale quantizes them to NaN.

### int8

`Int8DynamicActivationInt8WeightConfig` runs from Ampere up. On an H200 it renders at bf16 speed with fp8's memory: 14.7 to 15.0 seconds per evaluation and 67 to 77 GB with the transformer on the GPU. Its first evaluation compiles for 100 to 180 seconds.

`Int8WeightOnlyConfig` loads and renders, an order of magnitude slower: torchao's weight-only int8 has no fast path under the compiled `F.linear`.

## The script

`infer_diffusers.py` runs the README snippet and shares no code with the repository's own stack.

| Flag | Meaning |
|---|---|
| `prompt` | the text; defaults to the text `prompts/example_0.pt` was encoded from |
| `--out` | the mp4, default `results/diffusers.mp4` |
| `--steps` | model evaluations, default 8 |
| `--frames` | default 345 |
| `--transformer` | a checkpoint's `diffusers/` subfolder, e.g. `stage-b-step-2000/diffusers` with `--steps 50` |
| `--first`, `--last` | keyframes: `--first` alone is I2VA, `--last` alone is L2VA, both is FL2VA |
| `--fp8` | the fp8 preset |
| `--offload_dit` | stream the transformer per block, for a 24 GB card |
| `--seed`, `--device` | default 0 and `cuda` |


---

## FILE: docs/inference.md (https://github.com/OpenVDN/vdn-minimax-h3/blob/HEAD/docs/inference.md)

# Inference with this repository

How the repository's own stack renders: the entry points, the configs, the kernels behind them, fp8, the eight-GPU layout, memory, and what a prompt cache carries. The commands to run are in the [README](../README.md#inference-with-our-repository); the portable diffusers path is in [diffusers.md](diffusers.md).

## Entry points

| Script | What it does |
|---|---|
| [src/inference/infer.py](../src/inference/infer.py) | one GPU |
| [src/inference/infer_ulysses.py](../src/inference/infer_ulysses.py) | eight GPUs under `torchrun`, sequence parallel |
| [src/inference/encode_prompt.py](../src/inference/encode_prompt.py) | a text prompt to a prompt cache (`.pt`) |
| [src/inference/encode_keyframes.py](../src/inference/encode_keyframes.py) | a prompt plus keyframes (`--first`, `--last`) or reference images (`--refs`) to a prompt cache |
| [src/inference/infer_diffusers.py](../src/inference/infer_diffusers.py) | the published diffusers component; shares no code with the two above |

Encoding is separate from rendering because the Qwen3-VL text encoder is 63 GB: a prompt is encoded once, and every render reuses the cache without loading the encoder.

## Configs

A config file says how to compute, and the command line says what: the checkpoint, the prompt cache, the output. Every knob is a field of [src/config/inference.py](../src/config/inference.py), set in the YAML or as a `key=value` override. Nothing on the inference path reads an environment variable.

The files in [configs/inference/](../configs/inference/) form a ladder, once for 8 and once for 50 model evaluations:

| File | Adds |
|---|---|
| `8nfe.yaml` | the released arithmetic in bf16; with `checkpoint=null`, the dense MiniMax-H3 |
| `8nfe_tuned.yaml` | the inference kernels; the same sample up to kernel rounding |
| `8nfe_tuned_fp8.yaml` | fp8 on every wide Linear, 2 warm-up evaluations; the top single-GPU tier |
| `8nfe_tuned_fp8_ulysses_h200.yaml`, `_b200.yaml` | the eight-GPU layout tuned for that card |

The fields:

| Field | Default | Meaning |
|---|---|---|
| `checkpoint` | `null` | a checkpoint directory; `null` renders the dense base model |
| `base_source`, `vae_source` | the release copy | where the base transformer and the two decoders load from |
| `external_loras` | `[]` | community adapters, see [Weights](#weights) |
| `kernels.inference_kernels` | `true` | the forward-only kernel set |
| `kernels.softmax_backend` | `auto` | `auto`, `flex`, `decomposed` or `ref` |
| `precision.fp8.enabled` | `false` | fp8 on the wide Linears |
| `precision.fp8.skip_end_blocks` | `4` | first and last N blocks left in bf16; the shipped fp8 configs use 0 |
| `parallel.softmax_ranks` | `6` | `infer_ulysses.py` only: ranks given to the softmax heads; 0 is standard Ulysses |
| `render.prompt_file`, `render.out` | required | the prompt cache and the mp4 |
| `render.num_frames` | `345` | aligned to 17n+5; 345 frames are 14.4 seconds at 24 fps |
| `render.num_steps` | `50` | model evaluations |
| `render.warmup_steps` | `0` | evaluations run and discarded first |
| `render.seed` | `42` | |
| `render.video_shift`, `render.audio_shift` | `12.0`, `3.0` | scheduler shifts; a distilled adapter is valid only at the shifts it was trained at |
| `render.record` | `false` | also write `<out>.inference.json` |
| `render.save_latents` | `false` | also save the video and audio latents under `render.latents_root` |

`render.num_steps` is the number of model evaluations. The scheduler counts sigma grid points, one more than that, and the render passes the extra one itself.

## Assembly

Both entry points build the model through [src/inference/utils/assemble.py](../src/inference/utils/assemble.py), in this order: read the checkpoint's spec, load the base, apply the hybrid transform, load the linear branch, merge the checkpoint's LoRAs, merge the external LoRAs, select the inference kernels and the window-softmax backend, then convert to fp8. The record written by `render.record=true` states what actually ran: the checkpoint's identity, the resolved config, the resolved backend, the number of fp8 Linears, and the time of every evaluation.

## Inference kernels

`kernels.inference_kernels: true` swaps in forward-only kernels with the same arithmetic:

- **Attention:** QK-norm and RoPE fused in one pass over views of the packed q/k/v buffer, and the softmax gate fused with the output repack.
- **Linear branch:** one kernel for the frame statistics, a Triton 5-tap temporal convolution, a compiled RMSNorm and output gate, one triangular solve for the inverse, and `baddbmm` scan updates.
- **Block:** RMSNorm with the AdaLN affine, gate and residual as two compiled kernels, and SwiGLU as one.

The first run compiles them, which takes several minutes; later runs reuse the cache.

## Window softmax

Three implementations of the same windowed attention:

- `flex`: FlexAttention with a block mask. Training uses it.
- `decomposed`: the mask as a union of dense attentions, with no mask machinery. Global rows attend to everything in one dense call, and the frames that share a window go through one variable-length call.
- `ref`: the eager reference, for parity checks.

`auto` is `decomposed` on every CUDA device. It is the faster kernel on B200 and matches `flex` on H200. The two differ by bf16 reduction order, and by their transient memory, which [Memory](#memory) measures. Each picks its kernel by the card:

| Card | `flex` | `decomposed` |
|---|---|---|
| Hopper, data-center Blackwell (sm90, sm100, sm110) | FlexAttention's Flash backend (FA4) | FA4's varlen kernel, cuDNN SDPA |
| Ampere, Ada, consumer Blackwell (sm8x, sm120) | FlexAttention's Triton kernel | PyTorch's `varlen_attn`, SDPA's own kernels |

Nothing degrades to dense attention. A missing package raises its own import error: `flash-attn-4` on the first row, PyTorch 2.13 on the second.

## fp8

`precision.fp8.enabled: true` replaces every Linear at least 4096 wide on both sides with [Fp8Linear](../src/models/ops/fp8_linear.py): the qkv and output projections, the feed-forward pair and the linear branch's readout, 363 Linears with `skip_end_blocks: 0`. Narrow Linears stay in bf16.

- The quantizer is a Triton kernel, one program per row. The attention quantizes its input once for the three projections, and the feed-forward fuses SwiGLU with the quantization of its output.
- Scales are per row on sm90 and per tensor on sm100 and above, which is where each card's fast GEMM is. The granularity follows the card and is not a knob.
- The swap is one way: the bf16 weight is released as each Linear is replaced, so the model is roughly half its weight afterwards.
- fp8 needs compute capability 9.0 or above.

fp8 changes the sample: one evaluation stays close to bf16 (cosine 0.998), and the trajectory settles into a different video of the same quality. A seed reproduces a render only within one precision.

## Eight GPUs

`infer_ulysses.py` shards the packed sequence by row across the ranks. Inside attention one all-to-all turns sequence-sharded, all-heads q/k/v into all-sequence, head-sharded q/k/v, and a second one restores the layout, so the window softmax and the linear branch both see the whole sequence. There is no KV cache, token dropping or approximate communication.

The layout is branch-parallel: `parallel.softmax_ranks` ranks run the softmax heads and the rest run the linear branch. The shipped configs state the split and the kernel each card was tuned with:

| Config | Split | Window kernel |
|---|---|---|
| `_ulysses_h200.yaml` | 6 + 2 | `flex` |
| `_ulysses_b200.yaml` | 5 + 3 | `decomposed` |

Only rank 0 loads the decoders and writes the mp4. T2VA, keyframe and reference caches all run here: conditioning rows sit outside the video span, so they shard and attend as ordinary global rows.

Setup is dominated by eight ranks reading 66 GB each, so keep `ckpts/` on a local disk. The last log line splits the time into setup, denoise, and decode plus encode. Eight-GPU renders are not bit-reproducible from run to run; one-GPU renders are.

## Memory

The fp8 configs fit an 80 GB card, on one GPU and on eight. Peak GPU memory in GiB at 345 frames and 8 evaluations, on the T2VA example, measured on an H200 with the CUDA allocator capped at 78 GiB:

| Phase | One GPU | Eight GPUs, per rank |
|---|---:|---:|
| Load: the bf16 transformer | 61.7 | 61.7 |
| Assembly, through the fp8 conversion | 65.9 | 65.9 |
| Denoise | 71.4 | 55.9 to 60.4 |
| Decode | 67.8 | 68.5, rank 0 only |

The decoders wait on the CPU and move to the GPU when denoising is over, and the fp8 conversion releases each bf16 weight as it goes, so the model never sits on the card twice. Assembly then returns the freed memory to the driver, where NCCL allocates its buffers at the first collective. On one GPU the bf16 configs need more than 80 GB: 66 GiB of weights plus 26 GiB of transients.

The two encoders fit the same card: `encode_prompt.py` peaks at 63.2 GiB, and `encode_keyframes.py` with six references at 69.2.

Conditioning rows raise the denoise peak, and the window kernel decides by how much. `decomposed` gathers the global rows once per window group, and `flex` has no gather. One GPU, the same cap:

| Request | Window kernel | Denoise peak | Seconds per evaluation |
|---|---|---:|---:|
| T2VA | `decomposed` | 71.4 | 11.8 |
| T2VA | `flex` | 62.7 | 11.8 |
| FL2VA example | `decomposed` | 74.4 | 14.2 |
| Ref2VA-like example | `decomposed` | out of memory | 19.1 without the cap |
| Ref2VA-like example | `flex` | 63.8 | 17.8 |

`auto` stays `decomposed`. On an 80 GB card, render a reference cache with `kernels.softmax_backend=flex`:

```bash
python src/inference/infer.py --config configs/inference/8nfe_tuned_fp8.yaml \
  checkpoint=ckpts/stage-dmd-step-250 kernels.softmax_backend=flex \
  render.prompt_file=prompts/reference/example_ref2va.pt render.out=results/example_ref2va.mp4
```

## Prompt caches and conditioning

A prompt cache is one `.pt` holding the prompt text, the Qwen3-VL hidden state after layer 50 as `(L, 5120)` bf16, and a tag per row. A keyframe or reference cache adds the anchors and one VAE latent per image. Each token costs 10 KB, and an image costs about one token per 32×32 pixels, so a cache with images runs to tens of megabytes.

| Request | Encoder flags | Packed sequence |
|---|---|---|
| T2VA | `encode_prompt.py --prompt` | `[text, audio, video]` |
| I2VA, L2VA, FL2VA | `encode_keyframes.py --first` and/or `--last` | `[text, keyframes, audio, video]` |
| Ref2VA-like | `encode_keyframes.py --refs a.png b.png ...` | `[text, one block per reference, audio, video]` |

Keyframes go onto the video canvas: the first is stretched to it and the last is cover-cropped. References keep their own aspect ratio on a 768-pixel short edge (`--ref_size`), and each takes its own rotary slot ahead of the video. Every image appears twice: as a `<Picture i>` vision block in the text, and as conditioning rows that are noised to t = 0.999 and held there while only the generated rows are stepped. On the hybrid model the conditioning rows are global tokens, attended densely in both directions by the window softmax and absent from the linear scan.

Conditioning rows lengthen the sequence. A T2VA render is about 104k audio and video rows plus the text. The Samoyed reference example adds 6,948 text and 5,664 reference rows, and runs at 19.1 seconds per evaluation on one H200 in fp8 and 3.4 on eight.

## Frames, canvas and seeds

The canvas is 1344×768. `render.num_frames` is aligned to 17n+5, and 345 frames become 102 latent frames.

One-GPU renders are bit-reproducible: the same config, seed and cache give the same mp4 bytes on the same GPU model. The initial noise differs between GPU models.

## Output

The mp4 is 24 fps with the audio muxed in. It is encoded to a sibling `.partial.mp4` and renamed into place, so a file under the final name is always complete.

## Weights

A path written as `ckpts/<name>` that does not exist locally is downloaded from the Hub into the Hugging Face cache and used from there, so every command in the README also runs without a local `ckpts/`.

`external_loras` merges community adapters for the dense MiniMax-H3, safetensors with peft names, after the checkpoint's own LoRA:

```bash
python src/inference/infer.py --config configs/inference/8nfe.yaml \
  checkpoint=null "external_loras=[{path: turbo.safetensors}]" \
  render.prompt_file=prompts/example_0.pt render.out=results/turbo.mp4
```

`alpha` defaults to the value in the file's metadata, and to the rank when the file has none.

## Timing a render

Set `render.warmup_steps: 2`, as the fp8 configs do. The first evaluation compiles, and the second re-specializes for the next timestep. The timing line reports every evaluation after that, and `render.record=true` keeps them in the JSON. The numbers in the README's [Results](../README.md#results) exclude model loading, warm-up, VAE decoding and mp4 encoding.


---

## FILE: prompts/README.md (https://github.com/OpenVDN/vdn-minimax-h3/blob/HEAD/prompts/README.md)

# Example prompts

Pre-encoded text embeddings: three T2VA prompts (`example_N.pt`), one FL2VA prompt
(`image/example_fl2va.pt`) and one Ref2VA-Like prompt (`reference/example_ref2va.pt`). The
text of each:

## example_0.pt

**integrated_multimodal_description:**

[Shot 1] 2D-animated, a wide shot transitioning to a medium shot from a low camera angle as the camera performs an arc shot at slow speed. An infinite black void is filled with thick, rolling grey smoke, drifting black ink haze, and glowing red ember ash mixed with floating crimson thread particles. A female ronin assassin (S1) stands motionless in the center. She has pale skin, long flowing black hair, and wears layered black silk robes with oversized sleeves, tightly cinched with vibrant red waist cords. A sleek black katana rests in a black scabbard at her hip. Cold white volumetric lighting cuts through the dark void, heavily accented by the deep red ambient glow of the floating embers. She remains perfectly still in a restrained stance, her right hand resting lightly on the katana's hilt, as the smoke rolls smoothly across the unseen floor beneath her feet.

[Shot 2] At 00:03.500, the camera cuts to a close-up, pushing in at fast speed toward the assassin (S1). Her eyes narrow sharply. In a sudden blur of motion, she executes a rapid quickdraw slash, pulling the sleek black katana from its scabbard. A glowing crimson slash trail erupts from the metallic blade, slicing the rolling smoke in half. She drops into a low stance and dashes forward aggressively. Her layered black silk robes and oversized sleeves snap backward, demonstrating heavy, delayed fabric physics, while her long black hair whips fiercely behind her head.

[Shot 3] At 00:06.200, the shot transitions to a medium wide shot, panning right at fast speed to follow the action. The assassin (S1) leaps into the air, performing rapid airborne rotational strikes. The sleek black blade reflects the cold white light, leaving overlapping ink-like energy arcs and brilliant crimson thread-shaped distortions in its wake. The glowing sword impacts the surrounding black ink haze, scattering bright red embers violently outward in a spray. Her body twists powerfully in mid-air, the red waist cords and black robes trailing dramatically with the cinematic motion.

[Shot 4] At 00:09.800, the camera cuts to a wide shot, zooming in at fast speed toward the assassin (S1). She lands on the unseen floor and unleashes a massive, devastating crimson-infused finishing slash straight ahead. A giant, jagged crimson energy arc erupts from the blade, visually fracturing the infinite black void into shattered geometric shards of dark matter and blinding white light. The rolling smoke, flying debris, scattered ember particles, and shattered crimson thread energy suddenly freeze in extreme cinematic slow motion. The assassin (S1) is captured in a striking dark silhouette against the glowing red and white fracture lines, holding her extended katana follow-through pose motionless as the glowing threads remain perfectly suspended in the air around her.

**overall_soundscape:**

A low, rumbling wind and a soft hiss of swirling smoke mix with the subtle crackle of burning embers. This ambient tension is shattered by a loud, metallic scrape as a blade is quickly drawn, followed immediately by an aggressive, tearing whoosh and the rapid, heavy thud of quick footsteps. Multiple sharp, high-pitched swishes and crisp, clearly heard fabric snaps dominate the foreground as the spinning blade slices the air. A deafening, explosive boom and a deep, resonant bass crash mark the massive final impact, abruptly transitioning into a sustained, high-frequency ringing that stretches out over the frozen slow-motion sequence.

**non_diegetic_music:**

A fast-paced score featuring heavy, driving taiko drums and a rapid, solo shamisen melody, steadily increasing in rhythmic density and dynamic volume, before abruptly cutting off into a sudden, sustained low-frequency synthetic sub-bass tone with no additional instrumentation.

## example_1.pt

**integrated_multimodal_description:**

[Shot 1] 3D CG fantasy animation, cinematic photorealism, a wide shot frames ancient jungle ruins at dawn as the camera slowly pushes in toward a moss-covered stone bridge. Warm, volumetric sunbeams pierce through the dense, vibrant green tropical foliage, casting long shadows. Crystal-clear water flows smoothly beneath the bridge, while a thick, luminous mist drifts steadily through the serene environment.

[Shot 2] At 00:01.916, the camera cuts to a medium close-up of a young female elf with pointed ears and long, flowing blue hair, wearing an emerald-green linen tunic and leather bracers. Crouched on the mossy bridge, she extends her right hand toward the stream below. A glowing, translucent fox formed entirely of flowing water and light begins to emerge from the surface beside her, sending sparkling water particles into the air. The camera shakes slightly, adding subtle intimacy to the framing.

[Shot 3] At 00:03.833, the shot transitions to a closer medium close-up as the camera slowly pushes in. The fully materialized water-fox from Shot 2 playfully circles the elf's extended hand, nuzzling her fingers with its fluid snout. The elf smiles softly, her shoulders relaxing. Magical, luminescent ripples spread outward across the stream's surface below them, reflecting the teal light.

[Shot 4] At 00:05.750, the camera cuts to a medium shot, executing a smooth tracking shot to the right as the elf and the water-fox move together across the ancient stone bridge. The fox bounds gracefully between mossy stones, leaving a vibrant trail of glowing, suspended water droplets in its wake. Dense jungle foliage heavily frames the foreground and background of the stone path.

[Shot 5] At 00:07.666, the scene shifts to a medium wide shot where the camera performs a dynamic arc shot around the pair. The elf sweeps her arms upward in a graceful gesture. Luminous, fluid streams of teal water rise from the stream, spiraling like living ribbons around her and the magical companion. The water-fox darts joyfully through the swirling aquatic ribbons, its body blending seamlessly with the magical currents.

[Shot 6] At 00:09.583, the camera cuts to a medium shot and pans right to capture their fast but smooth motion. The water-fox leaps energetically through floating rings of glowing water, sending sparkling droplets splashing playfully around the elf. She throws her head back in a visual laugh, reaching both hands toward the creature as the suspended magical droplets catch the warm, volumetric sunlight.

[Shot 7] At 00:11.500, the camera cuts to a cinematic wide shot and gently pulls out. The elf and the water-fox sit side-by-side at the far edge of the stone bridge, their backs partially turned as they overlook the glowing jungle stream. The fox slowly settles its fluid body onto the stone beside her hip. Warm yellow sunlight breaks through the dense canopy above, illuminating the drifting mist and suspended water particles as the environment fades into a soft, hazy light.

**overall_soundscape:**

A continuous, loud flow of rushing water dominates the environment, accompanied by the distinct rustle of leaves in the breeze. As the fox materializes, pronounced, resonant magical chimes and clearly heard liquid splashes ring out. Audible, rhythmic splashing and the fluid sloshing of water follow as the creature bounds across the stones, mingling with distinct, glassy tinkling sounds of water droplets popping in the air and a faint but clearly audible creature trill.

**non_diegetic_music:**

An orchestral fantasy score plays at a slow tempo, featuring continuous string melodies, sustained low cellos, and light woodwind flourishes.

## example_2.pt

**integrated_multimodal_description:**

[Shot 1] Cinematic, wide establishing shot with the camera shaking slightly, revealing the Rhine square at golden hour. A beautiful young woman in her mid-20s with long, wavy dark brown hair stands small in the foreground. She wears a cream linen blouse, beige wide-leg trousers, brown walking shoes, and a light beige scarf. Her natural, glowing skin is flushed by the wind, which gently blows her hair and scarf. She gazes up reverently at the massive twin spires and flying buttresses of Cologne Cathedral dominating the background. Warm golden sunlight hits the Gothic stone, casting long, deep teal shadows across the square. An off-screen adult male narrator with a deep, calm, echoing voice (S1) begins a voiceover, <d>[English] For over 750 years, Cologne Cathedral has pierced the German sky like a stone prayer.</d> while the young woman's lips remain completely closed.

[Shot 2] At 00:02.100, the camera cuts to a low-angle tracking shot that follows the woman from Shot 1 as she walks toward the western portal. The camera glides forward from below her waist level, emphasizing the overwhelming scale and intricate carvings of the ancient stone facade. A warm golden rim light catches her silhouette against the teal-shadowed stone. She reaches out and touches the realistic, rough texture of the facade with her fingertips. The off-screen narrator (S1) continues his voiceover, <d>[English] ...a masterpiece of Gothic ambition, built by generations of hands and faith.</d> while the woman's lips remain entirely closed.

[Shot 3] At 00:03.800, the shot transitions to a medium static shot at the main portal. The woman from Shot 1 tilts her head back, looking up at thousands of sculpted stone figures and gargoyles surrounding the doorway. Golden light sculpts every deep crevice in the stonework. Her fingers slowly trace the cold stone surface, and her shoulders relax in awe. The deep-voiced off-screen narrator (S1) states, <d>[English] Every stone tells a story — of devotion, power, and time itself.</d> with the woman's mouth remaining entirely still.

[Shot 4] At 00:05.500, the camera cuts to a push-in shot that advances through the grand open wooden doors and into the vast cathedral nave. The framing captures the towering 43-meter ribbed vaults and glowing stained-glass windows in the deep background. Dramatic, volumetric god rays cut through the air, illuminating floating dust particles. The woman from Shot 1 walks forward slowly into the interior, dropping her arms to her sides in awe. The narrator (S1) says, <d>[English] Inside these walls, light itself becomes sacred.</d> with the woman's mouth remaining completely still.

[Shot 5] At 00:07.400, the shot changes to a close-up static shot of the woman's face. Jewel-toned stained-glass light washes over her wind-flushed skin and dark hair in patches of red, blue, and gold. Soft bokeh blurs the stone arches behind her, and bright colorful reflections gleam in her wide, unblinking eyes. She stares mesmerized as a subtle draft shifts the stray strands of her hair. The off-screen narrator (S1) intones softly, <d>[English] Each window a kaleidoscope of history, faith and divine artistry.</d> while her lips remain firmly closed.

[Shot 6] At 00:09.100, the camera cuts to a medium close-up static shot at the gilded Shrine of the Three Kings. Warm candlelight flickers, casting colored reflections across the highly detailed gold surface of the shrine and onto the face of the woman from Shot 1. She stands in silent contemplation, gently clasping her hands together at her waist and leaning forward slightly. The deep narrator (S1) continues, <d>[English] Here lie the relics of the Three Magi — a pilgrimage site for centuries.</d> with her mouth remaining completely still.

[Shot 7] At 00:10.900, the shot transitions to a pull-out shot inside the cathedral nave. The camera moves backward down the center aisle, making the woman from Shot 1 appear tiny against the massive stone columns and towering arched ceilings. The vast space balances deep teal shadows with warm golden highlights. She stands perfectly still in the center of the aisle, turning her head slightly to the right to observe the scale. The narrator (S1) reflects, <d>[English] In its vastness, the soul feels both humbled and elevated.</d> while the woman's lips remain closed.

[Shot 8] At 00:12.700, the camera cuts to a pedestal up shot that rises from the exterior stone steps. The upward motion reveals the entire Gothic masterpiece of the cathedral, silhouetted against a brilliant golden sunset reflecting off the distant Rhine river. The silhouetted figure of the woman from Shot 1 stands on the steps, her head tilted up toward the spires. She holds her position without moving. The off-screen narrator (S1) concludes, <d>[English] Cologne Cathedral — a living testament to the human spirit.</d> as her mouth stays completely still.

**overall_soundscape:**

A pronounced, continuous ambient breeze rushes through the open square, accompanied by the loud, rhythmic clack of hard walking shoes on stone cobblestones. As the scene transitions indoors, the outdoor wind fades instantly, replaced by a dense, cavernous room tone. The sharp scrape and heavy thud of slow footsteps resonate and echo widely throughout the vast stone interior. A clear rustle of thick linen fabric is clearly heard as the woman shifts her weight, followed by the faint crackle of burning candles in the background.

**non_diegetic_music:**

Solo cello playing long, sustained, mournful notes at a very slow tempo, underscored by a deep, resonant contrabass drone that creates a majestic, cinematic atmosphere without percussion or sudden dynamic swells.

## image/example_fl2va.pt

Keyframes `image/first.png` and `image/last.png`.

How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 14.30-second mark of the target video.

**integrated_multimodal_description:**

[Shot 1] This is a live-action, cinematic shot with a golden-hour fantasy aesthetic, capturing a continuous, unbroken journey. Exactly one uninterrupted 14.3-second take is recorded by the same continuously moving camera with a fixed 20mm lens. No hard cuts, jump cuts, dissolves, match cuts, reverse-angle edits, camera resets, teleportation, or hidden edits occur anywhere. Every timestamp below is an action beat inside this single Shot 1. Camera position, orientation, velocity, parallax, and scene geometry evolve continuously along one connected physical route. Cloth, sails, darkness, flashes, motion blur, pillars, and bird wings never conceal an edit or replace the background. Spatially connected entrances become visible before the camera passes through them. In the foreground, a young man in a blue robe holds a fluffy white cat wrapped in a burgundy cloth, looking out a wooden window at a sprawling canyon city with a central canal and a distant pagoda. At 00:00.500 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, the white cat leans outward, extending a front paw, prompting the man to shift his weight to stabilize it as a sudden gust of wind flutters the burgundy cloth and his hair. The camera pushes in, accelerating past their right side and crossing the wooden window frame. At 00:01.500 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, the camera pedestals down and pushes in sharply toward the canal below, dropping close to the water's surface. It pushes in past a billowing ochre boat sail, then trucks left to glide through the open doorway of a riverside fabric workshop. At 00:03.500 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, a loud, rhythmic fabric snap is heard as an artisan violently shakes out a massive sheet of off-white fabric, sending a heavy wave through the material. The camera pushes in directly underneath the flying cloth, briefly brushing the lower edge, to reveal a large, spinning wooden water wheel. The cloth only partially occludes the view; part of the doorway, floor, or water trough remains visible throughout. The same camera trajectory and continuous background parallax persist before, during, and after the cloth passes, with no occlusion cut. At 00:04.800 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, the camera pedestals down to dodge the splashing water from the wooden spokes and pushes in into an open stone aqueduct. At 00:06.000 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, the camera pushes in rapidly just above the rushing water, rolling slightly to bank through two consecutive curves in the narrow stone channel. At 00:07.500 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, the channel plunges into a short, shadowy rock tunnel; the camera physically travels through the same connected tunnel while the illuminated exit remains continuously visible and expands ahead. Water and stone-wall details remain readable; the image never becomes fully black or fully white. The same channel and camera trajectory continue through the exit into golden sunlight with smoothly adapting exposure, without a hidden cut or a viewpoint jump. At 00:09.000 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, a swordsman in flowing white robes enters the frame from the front, riding a slender flying sword. He bends his knees, leaning his body to execute a sharp turn, and ascends steeply along the vertical cliff face. At 00:10.500 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, a pronounced, powerful swoosh is heard as the camera tilts up and pushes in aggressively to chase him through a wide gap in the rocky canyon walls, the man's white sleeves whipping furiously in the wind. At 00:12.000 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, the swordsman arcs gracefully over the curved eaves of a high pavilion; the camera pushes in over the roofline and tilts down to level out, its speed slowing to reveal the sweeping panorama of the canyon city. At 00:13.000 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, a colossal blue and green mythical bird sweeps into the frame from the lower left foreground, its massive feathered wings beating heavily as it banks toward the central canal. Its body and wings occupy only part of the left foreground; the central canal and distant pagoda remain visible. There is no full-frame occlusion, wing-wipe edit, or replacement by the end image: the final composition is reached by continuous camera motion. At 00:14.000 within this same uninterrupted shot, with no cut, hidden edit, or camera reset, the swordsman continues flying toward the right side of the city, perfectly aligning with the final frame's grand composition as the continuous forward flight carries on over the sunlit metropolis.

**overall_soundscape:**

The soundscape begins with the gentle ambient rushing of the distant canal and a soft breeze rustling the man's hair and clothing. As the camera dives out the window, the wind noise intensifies into a loud, rushing roar. A loud, rhythmic fabric snap is clearly heard the moment the artisan shakes out the massive sheet of cloth, followed immediately by the heavy, splashing water and wooden creaking of the spinning water wheel. As the camera enters the aqueduct, the sound of rushing water becomes a loud, enclosed echo bouncing off the stone walls of the short tunnel. A pronounced, powerful swoosh cuts through the air as the swordsman on the flying sword rapidly ascends the cliff face, accompanied by the aggressive, distinct flapping of his wind-whipped robes. Toward the end, as the panorama opens up, a series of deep, resonant wing beats dominates the foreground as the giant blue bird sweeps into view, blending with the expansive, echoing ambiance of the bustling canyon city below.

**non_diegetic_music:**

An epic, soaring orchestral score plays throughout, starting with a steady, rhythmic pulse that builds in tempo and intensity as the camera accelerates through the narrow environments. The music swells into a majestic, triumphant crescendo with soaring strings and booming brass the moment the camera crests the roofline, perfectly matching the reveal of the grand city panorama.

## reference/example_ref2va.pt

References `reference/ref-1.png` to `reference/ref-6.png`, in that order.

**subject_definitions:**

&lt;Subject 1&gt; is the blonde woman wearing a light pink button-up shirt in &lt;Picture 1&gt;.

&lt;Subject 2&gt; is the young man with messy brown hair wearing a dark gray hoodie in &lt;Picture 2&gt;.

&lt;Subject 3&gt; is the white Samoyed dog defined by the multi-view reference board consisting of &lt;Picture 3&gt;, &lt;Picture 4&gt;, and &lt;Picture 5&gt;, featuring a thick, fluffy pure white coat, dark eyes, and a pink tongue.

&lt;Subject 4&gt; is the cafe lounge environment in &lt;Picture 6&gt;, featuring a prominent tufted orange sofa, a rustic wooden coffee table, exposed brick walls, and a glowing neon Central Perk sign.

**summary:**

[reference generation] The target video takes place in &lt;Subject 4&gt;, where &lt;Subject 1&gt; is sitting on the orange sofa eating a cookie. &lt;Subject 2&gt; walks into the cafe leading &lt;Subject 3&gt;, a fluffy white Samoyed, on a leash. The dog eagerly reaches for the cookie, prompting a playful exchange between the characters. &lt;Subject 2&gt; sits down next to her on the sofa, excusing his dog's behavior, and &lt;Subject 1&gt; responds with an amused, warm remark.

**retention_analysis:**

&lt;Subject 1&gt; (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the target video features the same blonde woman in the light pink button-up shirt.

&lt;Subject 2&gt; (appears in [Shot 1], [Shot 2]): fully_preserved - the target video features the same young man with messy brown hair wearing the dark gray hoodie.

&lt;Subject 3&gt; (appears in [Shot 1], [Shot 2]): fully_preserved - the target video features the same thick, fluffy pure white Samoyed dog.

&lt;Subject 4&gt; (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the target video takes place in the same cafe lounge environment, featuring the tufted orange sofa, rustic coffee table, and brick walls.

**detailed_description:**

The target video is in a realistic live-action comedy style with warm, natural indoor lighting that gently illuminates the scene.

[Shot 1] The video opens with a medium-wide shot in &lt;Subject 4&gt;, a cozy cafe lounge featuring a tufted orange sofa, a rustic wooden coffee table in the foreground, exposed brick walls, and a glowing neon Central Perk sign in the background. &lt;Subject 1&gt;, a blonde woman wearing a light pink button-up shirt (S1, presenting a clear, playful female voice), sits comfortably on the right side of the orange sofa, holding a half-eaten cookie in her right hand. Quiet, natural cafe room tone and the subtle metallic jingle of a dog collar are audible. &lt;Subject 2&gt;, a young man with messy brown hair wearing a dark gray hoodie (S2, presenting a smooth, apologetic male voice), walks into the frame from the left, holding a slack leather leash. He is walking &lt;Subject 3&gt;, a thick, fluffy pure white Samoyed dog with dark eyes, a bright pink tongue, and a highly textured fur coat. As they approach the seating area, &lt;Subject 3&gt; eagerly stretches its head forward, aiming its nose directly at the cookie in &lt;Subject 1&gt;'s hand. &lt;Subject 1&gt; defensively pulls her arm back, bringing the cookie safely toward her chest to protect the treat. Looking up at the dog and its owner, &lt;Subject 1&gt; (S1) says, &lt;d&gt;[English] Hey! Watch your dog!&lt;/d&gt; After she speaks, her mouth closes in mild surprise while she keeps her hand pulled back securely, and the quiet cafe room tone continues to play underneath the ambient scene.

[Shot 2] At 00:05.200, the shot cuts to a closer medium two-shot framing both characters on the sofa. &lt;Subject 2&gt; sits down on the left side of the orange sofa next to &lt;Subject 1&gt;. He gently gathers the slack leash in his left hand, pulling it closer to steady &lt;Subject 3&gt; so the fluffy dog sits obediently beside his leg. The ambient cafe room tone remains the only background sound, emphasizing the quiet interaction. &lt;Subject 2&gt; turns his head to look directly at &lt;Subject 1&gt;, offering a soft, apologetic smile to defuse the tension. Keeping his relaxed posture, &lt;Subject 2&gt; (S2) says, &lt;d&gt;[English] He just likes cookies more than me.&lt;/d&gt; Following his spoken line, the room tone continues through a brief, deliberate comic pause; &lt;Subject 2&gt; maintains his charming, sheepish smile while calmly petting the dog's thick coat, and &lt;Subject 1&gt; processes his charming excuse.

[Shot 3] At 00:10.100, the shot cuts to an intimate close-up of &lt;Subject 1&gt;. The ambient cafe room tone persists in the background. She looks off-screen to the left toward &lt;Subject 2&gt; and &lt;Subject 3&gt;, her initial surprise melting as her expression softens into a warm, amused smile. Easing her posture, &lt;Subject 1&gt; (S1) replies, &lt;d&gt;[English] Well, he has good taste at least.&lt;/d&gt; Immediately after she finishes speaking, her lips meet in a relaxed, closed-mouth smile. She slightly tilts her head to the side and lowers her cookie hand down out of frame, shifting her body weight comfortably against the sofa cushion. She holds this lingering, amused gaze steadily as the physical movement concludes. The quiet cafe room tone remains audible until the final frame ends.

**overall_soundscape:**

The video features a natural diegetic soundscape consisting of quiet cafe room tone, the subtle jingles of a dog collar, and clear, synchronized English dialogue between the two characters.

**non_diegetic_music:**

N/A


---

## FILE: diffusers_patches/BASE.txt (https://github.com/OpenVDN/vdn-minimax-h3/blob/HEAD/diffusers_patches/BASE.txt)

upstream base: 3a2f35d (huggingface/diffusers main, 2026-08-12)


---

# ISSUES / PRs of OpenVDN/vdn-minimax-h3


### #14 [open] ISSUE: VDN-H3 8-step DMD fl2va: mid-lower 16px-pattern mottle on 10s (243-frame) clips
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/14  by funnyfunny 2026-11-18T02:23:45Z

Environment
- Checkpoint: stage-dmd-step-250 (DMD 8 NFE; adapters default+turbo, rank 64; AdaLN/norm_out rank 16)
- Task: fl2va, 768x1024, 243 frames (10.125s @24fps), seed 42, fixed prompt + first-frame image
- Reference precision: bf16 (we run the same weights in fp16 on Ascend 910A; engine casts bf16->fp16)
- Topology: denoise-only transformer + external keyframe rows cache + remote VAE decode

Symptom
- A 16px-periodic "mottle"/dot pattern in flat lower-middle regions, strongest mid-clip.
- Same prompt/seed/cache at 141f (5.875s) is clean; the dense H3 base at 243f is clean.
- Metric (our local proxy, FFT 16px bin on a flat crop): VDN 3.56-3.63 (mid windows 4.2-4.7)
  vs dense base 2.18-2.64.

Localization (all measured on the same seed/cache)
- Zeroing the linear branch output gate (all 50 blocks) -> 2.42, mid-clip spike gone.
- Forcing the delta-rule decay to reset every frame (A_log=+30, alpha~0) -> 2.29.
- Bypassing scan backend / ATB-FA / frame_mean / NS / prepack / online-chunk / window radius: no change.
- Full-sequence GE vs torch VAE decode on the same latent: 3.5637 vs 3.5631 (rmse 0.186/255).
- Merged checkpoint vs runtime LoRA fill: 3.627 vs 3.621.
- Scan state magnitude (operator-returned prefix/suffix, fp32): max 53,464 / 51,600,
  injections 12,004, transitions 1.0005; zero Inf/NaN and zero elements beyond the fp16 hi/lo
  exact domain (+-131,008). So this is not an fp16 range/saturation issue.

Related: #12 (`Why does disabling the linear attention branch have little effect on video quality?`). Our long-clip data adds the missing signal: at 243 frames the branch is **not** quality-neutral — removing it removes the mid-lower 16px mottle, while 141f is clean either way. The effect is length-dependent.

Ask
1. Does the official 8-step DMD fl2va reproduce this at 243 frames in bf16 on GPU?
2. Is long-sequence linear-branch state drift a known limitation of this DMD student?
3. Recommended mitigation that preserves the official semantics (e.g., segment state reset,
   decay floor, or a different checkpoint/stage) and the expected quality trade-off?
4. We can share latents/prompts/params and the exact reproduction commands.



### #13 [open] ISSUE: ref2va模式支持参考音色输入吗
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/13  by qingmeizhujiu 2026-10-19T07:22:22Z

ref2va模式支持参考音色输入吗


### #12 [open] ISSUE: Why does disabling the linear attention branch have little effect on video quality?
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/12  by lynne2k 2026-10-10T02:45:32Z

Hi, thanks for releasing the model and training code!

We tested `stage-dmd-step-250` with the linear attention branch enabled and disabled, using the same prompts and initial noise.

Across two scenes and three paired runs, disabling it did not noticeably degrade overall visual quality, although motion and shot timing changed. We verified that the branch was actually disabled, while keeping window softmax and boundary anchors unchanged.

Why might the model remain this effective without the linear branch? What benefits should we expect it to provide, and under what conditions would those become clearly visible?

Paired videos attached: left ON, right OFF.

https://github.com/user-attachments/assets/065be075-c1fb-4f01-a906-c13b0e785977

https://github.com/user-attachments/assets/ddaead64-d979-4758-b96a-15c4b49942d3

> **Comment by funnyfunny (2026-11-18T02:23:59Z):**

Related data for this question (OpenVDN/vdn-minimax-h3, stage-dmd-step-250, 8 NFE, fl2va, 768x1024, seed 42): on a 10s / 243-frame clip, disabling the linear branch does not just change motion/timing — it removes a 16px-periodic mottle in flat lower-middle regions (FFT 16px-bin metric on a flat crop: 3.63 ON -> 2.29 OFF; dense base 2.64). On a 5.9s / 141f clip both ON/OFF are clean. So the branch's quality effect is length-dependent: it is nearly neutral at short lengths but becomes visible as a long-context artifact at ~10s. Forcing the delta-rule decay to reset every frame (A_log -> large positive) gives the same clean result as removing the branch. We also verified the scan state stays in range (max 53k < 65504 and < the 131,008 fp16 hi/lo exact domain), so this is not a fp16 range/saturation issue. Full evidence and a minimal reproduction are in our new issue (long-sequence mottle).

Filed as #14: https://github.com/OpenVDN/vdn-minimax-h3/issues/14



### #11 [open] ISSUE: Please add support for MacOS
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/11  by Emasoft 2026-10-05T16:40:20Z

Please add support for MacOS


### #10 [closed] ISSUE: Why does OpenVDN use the teacher model with decomposed attention rather than the dense version during DMD training?
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/10  by splinter21 2026-09-26T15:59:30Z



> **Comment by haochengxi (2026-09-28T06:54:10Z):**

The real score (teacher) in our DMD stage is the Stage-B VDN itself. The generator is VDN plus the `turbo` LoRA, so with VDN as the teacher, `turbo` only learns 50 steps to 8 steps, and the VDN-vs-dense gap is not part of what it has to close. The ceiling is VDN's own 50-step output. A dense teacher is supported with `real_score: dense`.




### #9 [closed] ISSUE: Will the training data be open-sourced?
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/9  by kunkun0w0 2026-09-17T05:17:32Z

Thanks for releasing this amazing work! Will you also open-source the training data?

> **Comment by haochengxi (2026-09-17T05:48:16Z):**

Hi, we don't plan to release the training data for now. You can curate the training dataset on your own by gathering prompts from the internet.



### #8 [open] ISSUE: Feature request: multi-reference image conditioning (Ref2VA-style) — training plan or validated workaround
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/8  by ft54482 2026-09-16T12:46:40Z

### Background

We run MiniMax-H3 in commercial production (a video-generation platform). **Multi-reference generation (Ref2VA-style: 1–9 reference images, often combined with video references and audio references) is the dominant task type in our traffic**, and it is also the most compute-heavy per request (many refs × 345-frame 768p).

Issue #5 confirmed: I2VA / FL2VA are natively supported, but for Ref2VA the team is *"not training a dedicated model"*, with a community-workaround suggestion of feeding references into the FL2VA base model. We fully understand the training-compute tradeoff — filing this as a formal feature request because this workload is exactly where VDN-H3's architectural speedup (2.6–2.9× per step, step-count independent) would matter most in production.

### What we are asking

1. **Roadmap**: is there any plan to train the VDN linear-attention branch + adapters (the open-sourced Stage A1/A2/B recipe) on **reference-conditioned data** (FL2VA-style keyframe conditioning extended to multi-image, or Ref2VA conditioning)? If you can share the data/compute recipe for a community effort, we would be interested in contributing.

2. **Near term**: has anyone validated the workaround mentioned in #5 — merging the VDN adapters (stage-b or stage-dmd) into the FL2VA base and feeding reference images through the community path? Specifically:
   - does adapter-merged inference remain stable with keyframe/reference conditioning present?
   - does the 8-step DMD tier hold acceptable quality with references, or does it need the 50-step tier?
   - any recommended config (e.g. which stages to merge, conditioning placement) that could be published as a "known-working with refs" setup?

3. **Related**: video-reference (reuse a reference clip's motion/audio) and audio-reference conditioning are part of the same commercial workload — any thought given to those in the roadmap?

### What we can offer

We can run **same-seed A/B benchmarks at production scale** (multi-image refs, video refs, 362-frame 768p / 2K) on our fleet and publish detailed results (latency per NFE, peak VRAM, quality impressions with side-by-side samples).

### Secondary note (hardware)

Our production fleet is Ada/SM89 + consumer Blackwell (RTX 4090 / L40S / 5090). The current FA4 + FP8 (sm90/sm100) requirement blocks direct adoption for us. If a `--softmax-backend sdpa`-style fallback and an sm89 fp8/triton path are ever on the table, that would widen the usable hardware range considerably — happy to test any such branch.


> **Comment by haochengxi (2026-09-17T04:41:56Z):**

https://github.com/OpenVDN/vdn-minimax-h3#supporting-ref2va-like-task

Ref2VA-like generation is supported and tested. As long as the image reference resolution is controlled (<=768p), VDN offers ~2.5x speedup for each denoising step. Combined with few-step distillation, this still leads to a significant speedup ratio.


> **Comment by haochengxi (2026-09-17T05:52:33Z):**

Thanks for the note. On current `main` the window softmax runs on RTX 4090 / L40S / RTX 5090 without FA4: on Ampere, Ada and consumer Blackwell, it uses FlexAttention's Triton kernel and PyTorch's own `varlen_attn` + SDPA, selected automatically by the card. We have not tested these cards ourselves, so reports from your fleet are welcome.

FP8 is not supported on sm89 today (our path needs rowwise `_scaled_mm`, sm90+), and it is untested on sm120. We don't plan to make the FP8 part more complex for now, but PRs are welcome.




### #7 [closed] ISSUE: Question: Support for RTX A6000 (48GB) and availability of a non-FP8 / quantized version
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/7  by ckcfcc 2026-09-15T18:28:52Z

Hi, I have 2× RTX A6000 (48GB). The docs mention FP8 — since the A6000 (Ampere) has no native FP8 support, is this hardware supported? If not, could you provide a quantized (INT8/INT4 or FP16/BF16) version? Thanks!

> **Comment by haochengxi (2026-09-16T20:53:54Z):**

Hi, the A6000 is supported — you just won’t be able to use the FP8 path since Ampere doesn’t have native FP8 tensor cores.

The model weights on the Hub are already BF16, so you can simply load and run them in BF16 without any quantization. There’s no need for a separate checkpoint.

If you want to reduce memory usage further, you can pass a custom TorchAO quantization config, for example `TorchAoConfig(Int8DynamicActivationInt8WeightConfig())`, to use W8A8 INT8. That works on sm80+ GPUs, including the A6000.

I haven’t tested INT4 on this model yet, though.



> **Comment by gengcong940126 (2026-09-21T07:02:45Z):**

Hi, I have 8 H100 GPUs, but when I run bash scripts/inference/8nfe_tuned_fp8.sh or bash scripts/inference/8nfe_tuned_fp8_ulysses_h200.sh, I still run out of memory because the transformer is too large to fit on the GPUs.


> **Comment by haochengxi (2026-09-21T21:58:51Z):**

Thanks for the report. This is fixed on `main`: please pull and run the same two scripts again.

The out-of-memory happened while the model was being assembled, before the first step: the two decoders sat on the GPU next to the bf16 transformer, and the fp8 conversion kept every bf16 weight alive until it finished. The peak is now 66 GiB during assembly and 71 GiB while denoising on one GPU.

Memory details are in [docs/inference.md](https://github.com/OpenVDN/vdn-minimax-h3/blob/main/docs/inference.md#memory).



> **Comment by gengcong940126 (2026-09-23T09:51:38Z):**

Sorry to bother you again. I try to run bash scripts/inference/8nfe_tuned_fp8.sh for one GPU, it's okay. But when I run scripts/inference/8nfe_tuned_fp8_ulysses_h200.sh for 8 GPUs, it still run out of memory. It seems the failure occurs during the first warm-up forward pass, specifically in Ulysses’ all_to_all_single operation.
At this point, NCCL attempts to allocate communication buffers, but there is insufficient available GPU memory on each device, resulting in a CUDA out-of-memory error.
Could you please help to fix it?


> **Comment by haochengxi (2026-09-24T20:55:23Z):**

Thanks for the details. The eight-GPU out-of-memory is fixed on `main`: please pull and run `scripts/inference/8nfe_tuned_fp8_ulysses_h200.sh` again.

We reproduced your error on GPUs limited to 80 GB. With the fix, it should able to run now.



### #6 [closed] ISSUE: Test
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/6  by haochengxi 2026-09-08T03:45:32Z

https://github.com/user-attachments/assets/343d03e6-8d88-444d-b66c-19d9fb252b37
https://github.com/user-attachments/assets/9de4ffbc-1a38-4467-8c9e-c975d2b84fa1
https://github.com/user-attachments/assets/f0ec47d4-96ad-4fbb-8bc9-1b9d5eeb0b94
https://github.com/user-attachments/assets/bf849387-03c6-45ef-988b-38164f27e88a
https://github.com/user-attachments/assets/2cc62303-1ebb-4d3f-a367-3c0125b11c3d
https://github.com/user-attachments/assets/2e0f43d3-4919-44c4-9f64-42d90d9e08b1

> **Comment by haochengxi (2026-09-09T05:49:09Z):**

https://github.com/user-attachments/assets/56728e17-9081-4f5a-b707-54de1cd8c166


> **Comment by haochengxi (2026-09-17T02:37:01Z):**

https://github.com/user-attachments/assets/99b02319-4f20-49c6-86e7-bb0ed8a48255



### #5 [closed] ISSUE: how to perform FL2VA and Ref2VA?
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/5  by joystreamer 2026-09-08T02:52:59Z

Thank you for your great release!  How to modify the codes to support FL2VA and Ref2VA?

> **Comment by haochengxi (2026-09-08T16:56:03Z):**

Hi, we’ve tested the model and confirmed that it natively supports I2VA and FL2VA.

For Ref2VA, we are not training a dedicated model. However, based on this implementation: https://github.com/T8mars/comfyui-minimax-h3-audio-T8, it may be possible to feed the reference directly into the FL2VA base model and achieve some degree of controllability over the reference.


> **Comment by joystreamer (2026-09-09T03:39:45Z):**

Thank you so much for your effort. The performance of FL2VA is pretty good. Could you further support multi-GPU inference like T2AV？


> **Comment by haochengxi (2026-09-09T04:38:54Z):**

Hi, we've already supported Multi-GPU inference. Please check https://github.com/OpenVDN/vdn-minimax-h3/blob/main/scripts/inference/50nfe_tuned_fp8_ulysses_b200.sh


> **Comment by joystreamer (2026-09-09T05:20:29Z):**

I want to perform multi-GPU inference in FL2VA mode. It raises ValueError("keyframe conditioning (i2va / fl2va) is only supported by infer.py; the Ulysses path would have silently ignored it")


> **Comment by haochengxi (2026-09-09T07:32:39Z):**

Should be good now, you can take a look.


> **Comment by joystreamer (2026-09-09T12:49:56Z):**

Thank you very much.



### #4 [closed] ISSUE: videos are not visible
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/4  by makovez 2026-09-06T19:59:32Z

could you make the videos visible ? and do you think same speed of inference can be achieved on a GB200 ?

> **Comment by haochengxi (2026-09-06T20:39:37Z):**

Hi, which video is not visible right now? Could you send me a screenshot so we can investigate and fix it?

For GB200, we don’t currently have access to test it, but I don’t see any obvious bottlenecks or compatibility issues that would prevent it from working.



> **Comment by makovez (2026-09-07T08:53:47Z):**

<img width="1970" height="1454" alt="Image" src="https://github.com/user-attachments/assets/f1e8c359-edbb-4ab1-bcd4-151ee53dae3b" />




> **Comment by haochengxi (2026-09-08T03:51:27Z):**

Thanks for telling us that! We've fixed the problem now.



### #3 [closed] PR: Add persistent Ulysses request worker with low-memory decode
URL: https://github.com/OpenVDN/vdn-minimax-h3/pull/3  by YaoleWu 2026-09-04T10:26:10Z

## Problem

The one-shot Ulysses entrypoint assembles the model and prepares LoRA again for every clip. Add a finite JSONL request queue so multiple clips can reuse one process.

## Change

A `WorkerConfig` (`requests_file`, `decoder_cpu_offload`, `transformer_cpu_offload_blocks`) is added to `InferenceConfig`, and `infer_ulysses.py` parses the queue with `load_requests()` before any rank enters a collective: each rank reads the same file, every row overrides `render.*` fields, and unknown fields, a missing `prompt_file` / `out`, duplicate outputs, or an empty file are rejected. The model is assembled once, then a per-request loop reloads conditioning, samples, decodes, and writes one record per output; `worker.*` is rejected on the single-process path. `load_decoder_models()` is factored out of `utils/assemble.py` so the decoders can be loaded separately for the offload path. The README documents the queue, the `examples/requests.jsonl` sample, and the rank-0 decoder swap.

Each request loads its own text/keyframe prompt cache. A text-only request after a keyframe request clears the previous conditioning. Outputs have explicit paths, and duplicate outputs or unknown fields are rejected.

## Behaviour and compatibility

Optional rank-0 decoder swapping offloads the whole transformer or trailing blocks during decoding; other ranks retain their replicas. Per-request records separate initial setup from rendering. Native `render.num_steps` continues to count model evaluations.

The worker reuses upstream prompt-cache and model-assembly utilities. The upstream one-way FP8 conversion is retained without adding a second conversion mechanism.

## Testing

At `c99485d4`, an offline CPU check executes the actual worker, assembly and prompt-cache code with model/CUDA calls simulated. It covers per-request steps, keyframe → text reset, one-time setup records and decoder/transformer restoration. Changed Python parses and the diff check passes.

Current-head distributed execution, GPU capacity and same-process multi-request rendering remain unverified. Historical GPU files used different offload settings and recorded no commit, so they do not establish a matched consecutive-request benchmark or a current block-count recommendation.

AI assistance: Codex assisted with implementation, review, validation and this description.


> **Comment by YaoleWu (2026-09-14T07:06:15Z):**

I read back the retained inference records on the original 8 x RTX PRO 5000 72GB machine on 2026-09-14. Here are selected machine-generated fields behind the 25-block row, with machine paths omitted:

```json
{
  "fp8_linears": 363,
  "merged_lora_pairs": 571,
  "parallel": {
    "kind": "ulysses_branch_parallel",
    "world_size": 8,
    "softmax_ranks": 5,
    "persistent_worker": true,
    "decoder_cpu_offload": true,
    "transformer_cpu_offload_blocks": 25,
    "warmup_steps": 0
  },
  "timings": {
    "request_index": 1,
    "model_setup_seconds": 0.0,
    "request_seconds": 179.7034954300616,
    "denoise_seconds": 91.29142212599982,
    "decode_and_encode_seconds": 88.40701913600788,
    "decoder_swap_out_seconds": 12.758040287066251,
    "decoder_swap_back_seconds": 16.08773321902845
  }
}
```

Timing interpretation: `decode_and_encode_seconds` already includes the swap operations; adding the two swap fields to it would double-count them. Denoising plus that combined phase is approximately the reported request total. Zero setup is the benefit for this subsequent request, not a claim that cold startup is free.

Two provenance limits worth preserving when reproducing:

- The retained `first.mp4.inference.json` is now the separate **20-block** boundary run (170.4852 s, request index 0, setup 279.7259 s), while `second.mp4.inference.json` is the **25-block** record above. Their filenames alone must not be treated as a matched consecutive pair.
- These historical records have an empty `git_commit` field. This readback supports the timing/configuration data, but does not independently prove byte-for-byte identity to the current PR head. It was not a fresh GPU benchmark.

The conservative recommendation remains 25 trailing blocks; the 20-block number remains a single boundary sample, as stated in the PR.



> **Comment by YaoleWu (2026-11-20T19:02:55Z):**

Moving this to draft. Reusing loaded weights across clips is a real workflow need, but the current head is only exercised by an offline worker/assembly fixture with simulated model and CUDA calls. The distributed two-request path and the decoder/transformer offload capacity have not been validated at this head. I'll keep the implementation available without requesting review until a matching isolated multi-GPU run establishes these core behaviors.


> **Comment by YaoleWu (2026-11-22T15:53:11Z):**

Closing this draft for now. The finite worker queue has CPU-simulated coverage, but the current revision has not completed a real distributed, consecutive-request render on the required topology. The historical runs used different offload settings and are not current-head evidence. The branch is retained; a follow-up should first establish same-process multi-request correctness and memory recovery on an isolated fleet.



### #2 [closed] ISSUE: Training data scale for reproducing VDN-H3
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/2  by shuangtianxiaoye 2026-09-04T05:57:01Z

Hi, thank you for releasing VDN-H3 and the full training pipeline!

We are currently trying to reproduce the VDN-H3 training process, especially Stages A1, A2, and B. From the released training configurations, we can estimate the total number of sample exposures from the number of GPUs and training steps, but we could not find information about the actual number of unique training video clips used.

Could you please share some details about the training data scale?

Specifically:

Approximately how many unique video clips were used for training Stages A1/A2/B?
If the exact dataset cannot be disclosed, what dataset scale would you recommend for reproducing the reported VDN-H3 results (e.g., 10K, 30K, 100K unique clips)?

This information would be very helpful for estimating the data and compute requirements for reproducing the attention-surgery stages.

Thanks again for the great work and for making the training code available!

> **Comment by haochengxi (2026-09-04T17:37:26Z):**

We would suggest at least 10k, but more is always better. We use the same dataset for A1/A2/B stages.


> **Comment by shuangtianxiaoye (2026-09-04T17:41:59Z):**

Thank you very much for the clarification! This is very helpful for our reproduction. We really appreciate your response and the open-source work.


> **Comment by ChdDongyang (2026-10-09T12:56:05Z):**

@haochengxi  Thank you for your excellent work. I would like to ask about the ratio of t2va to fl2va data within the 10,000-record dataset. During testing, I observed that fl2va performed relatively poorly compared to t2va; is this because fl2va data accounts for a smaller proportion of the training set?



### #1 [open] ISSUE: Invitation to integrate VDN-H3 into FastVideo
URL: https://github.com/OpenVDN/vdn-minimax-h3/issues/1  by Davids048 2026-09-02T23:48:33Z


Hi @haochengxi and the OpenVDN team,

Congratulations on releasing VDN-H3! We are glad to see you compare with [FastH3](https://haoailab.com/blogs/fasth3-preview/).

We would love to invite you to open a PR that integrates the VDN-H3 method and checkpoints into [FastVideo](https://github.com/hao-ai-lab/FastVideo). The FastVideo team is happy to help with the integration.

Congratulations again on the release!

