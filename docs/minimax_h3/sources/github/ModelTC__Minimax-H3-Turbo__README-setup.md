# ModelTC__Minimax-H3-Turbo__README-setup

Source: https://github.com/ModelTC/Minimax-H3-Turbo
Fetched: 2026-11-28



---

## FILE: README.md (https://github.com/ModelTC/Minimax-H3-Turbo/blob/HEAD/README.md)

# [Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)

Minimax-H3-Turbo provides MiniMax-H3 Turbo LoRA checkpoints, plus Diffusers batch inference and ComfyUI workflows.

- **Online Studio:** [x2v.light-ai.top](https://x2v.light-ai.top/)
- **API Documentation:** [x2v.light-ai.top/api-docs](https://x2v.light-ai.top/api-docs)
- **Model Weights:** [Hugging Face](https://huggingface.co/lightx2v/Minimax-h3-Turbo)

## 1. Model specs

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th align="center">Tasks</th>
      <th align="center">Training<br>resolution</th>
      <th align="center">Training shifts<br>(video / audio)</th>
      <th align="center">Distillation<br>steps (NFE)</th>
      <th align="center">Recommended inference<br>steps (NFE)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <strong>FL2VA Turbo 4-step v0.1</strong><br>
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors">Diffusers</a> ·
        <a href="https://huggingface.co/Kijai/MiniMax-H3_comfy/blob/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors">ComfyUI</a>
      </td>
      <td align="center">FL2VA / T2VA</td>
      <td align="center">544p<br><sub>mixed aspect ratio</sub></td>
      <td align="center">12 / 3</td>
      <td align="center">4</td>
      <td align="center">4</td>
    </tr>
    <tr>
      <td>
        <strong>FL2VA Turbo 8-step v1.0</strong><br>
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors">Diffusers</a> ·
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors">ComfyUI</a>
      </td>
      <td align="center">FL2VA / T2VA</td>
      <td align="center">544p<br><sub>mixed aspect ratio</sub></td>
      <td align="center">12 / 3</td>
      <td align="center">8</td>
      <td align="center">8 / 4</td>
    </tr>
    <tr>
      <td>
        <strong>FL2VA Turbo 4-step v1.0 768p</strong><br>
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors">Diffusers</a> ·
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors">ComfyUI</a>
      </td>
      <td align="center">FL2VA / T2VA</td>
      <td align="center">768p<br><sub>1344×768</sub></td>
      <td align="center">6 / 3</td>
      <td align="center">4</td>
      <td align="center">4</td>
    </tr>
    <tr>
      <td>
        <strong>FL2VA Turbo 8-step v1.0 768p</strong><br>
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors">Diffusers</a> ·
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors">ComfyUI</a>
      </td>
      <td align="center">FL2VA / T2VA</td>
      <td align="center">768p<br><sub>1344×768</sub></td>
      <td align="center">6 / 3</td>
      <td align="center">8</td>
      <td align="center">8</td>
    </tr>
    <tr>
      <td>
        <strong>Ref2VA Turbo 4-step v0.1</strong><br>
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors">Diffusers</a> ·
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors">ComfyUI</a>
      </td>
      <td align="center">Ref2VA</td>
      <td align="center">544p<br><sub>mixed aspect ratio</sub></td>
      <td align="center">12 / 3</td>
      <td align="center">4</td>
      <td align="center">4</td>
    </tr>
  </tbody>
</table>

### Note on shift

For `NFE = N`, define the N transformer evaluation points on the unshifted grid as
`q_i = (N - i) / N`, where `i = 0, 1, ..., N - 1`.

For example, with `NFE = 4`, `video shift = 12`, and `audio shift = 3`, the shared
grid is `q = [1, 0.75, 0.5, 0.25]`, giving video sigma
`[1, 0.9730, 0.9231, 0.8000] -> 0` and audio sigma
`[1, 0.9000, 0.7500, 0.5000] -> 0`; each list therefore uses exactly four NFEs.

### Note on reference-image resizing

The three reference-image resizing policies used by our workflows are based on
the `ref_image_size` implementation described in [ComfyUI's MiniMax H3 R2V reference-image sizing guidance](https://docs.comfy.org/tutorials/video/minimax/minimax-h3#prompting-tips-3):

<table>
  <thead>
    <tr>
      <th align="center">Mode</th>
      <th>Behavior</th>
      <th align="center">Scale factor<br><sub>before 32-pixel rounding</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><code>match</code></td>
      <td>Matches the reference pixel area to the target canvas while preserving the reference aspect ratio. It never upscales a smaller reference.</td>
      <td align="center"><code>min(1, sqrt(target_area / ref_area))</code></td>
    </tr>
    <tr>
      <td align="center"><code>max</code></td>
      <td>Preserves the reference aspect ratio and only scales down references whose short edge exceeds 2048 pixels.</td>
      <td align="center"><code>min(1, 2048 / ref_short_edge)</code></td>
    </tr>
    <tr>
      <td align="center"><code>diffusers</code></td>
      <td>Preserves the reference aspect ratio and forces the short edge to 2048 pixels, matching the original Diffusers behavior.</td>
      <td align="center"><code>2048 / ref_short_edge</code></td>
    </tr>
  </tbody>
</table>

All three policies keep the reference aspect ratio, use the H3 resolution grid
(dimensions rounded to multiples of 32), and avoid cropping the reference
content. In our distillation training, we use `match`, so the reference-image
pixel budget follows the target training resolution.

The Ref2VA inference entry point in this repository exposes the same three
policies through `--reference-resize-mode` and defaults to `match`. Passing
`--reference-resize-mode diffusers` restores the original Diffusers behavior
(the fixed 2048-pixel short edge). For our distilled models, **we recommend
selecting `match`** so inference uses the same resizing policy as training.

## 2. Online Studio and API

### Online App

Try the MiniMax-H3 Turbo LoRA directly in [LightX2V Studio](https://x2v.light-ai.top/).

The Studio currently uses the [FL2VA Turbo 8-step v1.0 768p LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors).

- **Online Studio:** [https://x2v.light-ai.top/](https://x2v.light-ai.top/)
- **Current model:** FL2VA Turbo 8-step v1.0 768p
- **Recommended settings:** 8 NFE, video shift 6, audio shift 3

> The model version deployed in the Studio may be updated over time.

### Studio Preview

<!-- Drag and drop the LightX2V Studio screenshot below this line. -->


### Online API

Integrate MiniMax-H3 Turbo into your application through the LightX2V API:

- **API Documentation:** [https://x2v.light-ai.top/api-docs](https://x2v.light-ai.top/api-docs)

## 3. Diffusers setup and inference

See [DIFFUSERS_SETUP_AND_INFERENCE.md](DIFFUSERS_SETUP_AND_INFERENCE.md) for
environment setup, checkpoint downloads, test JSON files, and single- or
multi-GPU inference commands.

## 4. ComfyUI inference

See [COMFYUI_SETUP_AND_INFERENCE.md](COMFYUI_SETUP_AND_INFERENCE.md) for
ComfyUI requirements, model installation, inputs, prompts, and run instructions.

### Example workflows

Ready-to-import graphs are in [example_workflows](example_workflows/). The T2VA
and I2VA graphs default to **FL2VA Turbo 8-step v1.0**; the Ref2VA graph uses
**Ref2VA Turbo 4-step v0.1**.

| Workflow | Task | Default resolution |
|---|---|---|
| [video_minimax_h3_t2v_lightx2v_turbo.json](example_workflows/video_minimax_h3_t2v_lightx2v_turbo.json) | T2VA (text-to-video + audio) | 960×544 (`16:9`, `0.5` MP) |
| [video_minimax_h3_i2v_lightx2v_turbo.json](example_workflows/video_minimax_h3_i2v_lightx2v_turbo.json) | I2VA / FL2VA (image-to-video + audio) | 864×480 (`16:9`, `0.4` MP) |
| [video_minimax_h3_ref2v_lightx2v_turbo.json](example_workflows/video_minimax_h3_ref2v_lightx2v_turbo.json) | Ref2VA (reference-to-video + audio) | 960×544 (`16:9`, `0.5` MP) |

All graphs wrap the same MiniMax-H3 subgraph. T2VA leaves `first_frame` /
`last_frame` unconnected; I2VA connects a `LoadImage` to `first_frame`, with
`last_frame` optional for first/last-frame interpolation. Ref2VA connects one
or more reference images through the reference-input branch.

For detailed workflow inputs and execution steps, see
[COMFYUI_SETUP_AND_INFERENCE.md](COMFYUI_SETUP_AND_INFERENCE.md).

## 5. Roadmap

1. Improve the visual quality and consistency of Ref2VA and FL2VA Turbo.

## 6. Acknowledgements

Some Ref2VA test cases and reference assets are adapted from public showcases on
the [Hailuo website](https://hailuoai.video/) and from the [MiniMax-H3 discussion
on Hugging Face](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/29).
We thank the community contributors for sharing their test assets and prompts.

Special thanks to the
[MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) project and the
MiniMax team for open-sourcing the MiniMax-H3 model.


---

## FILE: COMFYUI_SETUP_AND_INFERENCE.md (https://github.com/ModelTC/Minimax-H3-Turbo/blob/HEAD/COMFYUI_SETUP_AND_INFERENCE.md)

# ComfyUI Setup and Inference

## Environment

**ComfyUI 0.31.0 or newer is required.** The example graphs use ComfyUI core
nodes only, including the built-in MiniMax-H3 subgraph
(`MiniMaxH3ImageToVideo`).

## Required models

Download the base MiniMax-H3 components from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main)
and place them in the corresponding ComfyUI model directories.

Download the ComfyUI LoRAs from
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
into `ComfyUI/models/loras/`:

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
  minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors \
  --local-dir ComfyUI/models/loras
```

```text
ComfyUI/
└── models/
    └── loras/
        ├── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
        ├── minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
        └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
```

## Example workflows

Ready-to-import graphs are listed in the main [README](README.md). The T2VA
and I2VA graphs default to **FL2VA Turbo 8-step v1.0**; the Ref2VA graph uses
**Ref2VA Turbo 4-step v0.1**.

## Inputs

On the subgraph, set `lora_name`, `steps`, `shift_video`, and `shift_audio`
together to match the model table. Sampler defaults to `euler`.

**Resolution.** Width and height must be multiples of 32. Change the aspect
ratio or megapixel value on **Resolution Selector** when another size is needed.
The I2VA graph also includes an unused **Use Image Size** group
(`ImageScaleToTotalPixels` → `GetImageSize`); wire those outputs to the
subgraph `width` / `height` if you want the video canvas to follow the input
image instead of the selector.

**Duration.** Defaults to `5` seconds. MiniMax-H3 runs at 24 FPS and the frame
count must be `17 * n + 5`, so a 5-second request is snapped to 124 frames
(about 5.17 seconds).

**Prompts.** Use three sections:

```text
integrated_multimodal_description: [Shot 1] Describe the visual style, subject, action, camera, lighting, and dialogue.

overall_soundscape: Describe dialogue, ambient sound, and synchronized sound effects.

non_diegetic_music: Describe the background score, or use N/A for no score.
```

Dialogue can be written as `<d>[English] Dialogue text.</d>`.

For I2VA / FL2VA, keep first-frame identity with `<Picture 1>` (and
`<Picture 2>` if `last_frame` is connected):

```text
integrated_multimodal_description: For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

[Shot 1] Preserve the subject, clothing, and scene from <Picture 1>, then describe the motion.
```

## Run

Drag a workflow JSON onto the ComfyUI canvas, set the inputs above, and queue
it. Output is written under `video/MiniMax_H3`.


---

## FILE: DIFFUSERS_SETUP_AND_INFERENCE.md (https://github.com/ModelTC/Minimax-H3-Turbo/blob/HEAD/DIFFUSERS_SETUP_AND_INFERENCE.md)

# Diffusers Setup and Inference

## Environment

Follow the [Diffusers MiniMax-H3 documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3)
to prepare a compatible Diffusers version and Python environment. The inference
script uses `MiniMaxAI/MiniMax-H3` as the base model by default.

## Download LoRA checkpoints

Download the Diffusers checkpoints from
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
to the repository root:

```bash
python -m pip install -U huggingface_hub
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --local-dir .
```

Jobs live in `examples/`:

- T2VA: [`examples/prompts_t2va_test.json`](examples/prompts_t2va_test.json)
- I2VA: [`examples/prompts_i2va_test.json`](examples/prompts_i2va_test.json)
- Ref2VA: [`examples/prompts_ref2va_test.json`](examples/prompts_ref2va_test.json)

Each test example specifies `duration`, `megapixels`, and `aspect_ratio`.
The inference script resolves the final width and height from these fields
using [`resolution_util.py`](resolution_util.py) and rounds both dimensions to
multiples of 32. Supported aspect ratios are `21:9`, `16:9`, `4:3`, `1:1`, and
their transposes.

## Multi-GPU inference with FSDP2

FSDP2 shards the text encoder and the active transformer across GPUs, so CPU
offload is disabled. Launch one process per GPU with `torchrun` (PyTorch >= 2.6
and an NCCL-capable environment are required):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc-per-node=8 \
  inference_minimax_h3.py \
  --fsdp2 \
  --jobs-json examples/prompts_t2va_test.json \
  --lora-path minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  --inference-steps 8 \
  --output-dir outputs/lora_8nfe_fsdp2
```

To run the base model with FSDP2, omit `--lora-path`. FSDP2 shards only the
text encoder and the active transformer; the remaining pipeline components are
replicated on every rank.

## Single-GPU inference

LoRA, 4 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_t2va_test.json \
  --lora-path minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  --inference-steps 4 \
  --output-dir outputs/lora_4nfe
```

LoRA, 8 NFE (v1.0):

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_t2va_test.json \
  --lora-path minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  --inference-steps 8 \
  --output-dir outputs/lora_8nfe
```

I2VA, 8 NFE (v1.0):

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_i2va_test.json \
  --lora-path minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  --inference-steps 8 \
  --output-dir outputs/i2va_lora_8nfe
```

Ref2VA, 4 NFE (v0.1):

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_ref2va_test.json \
  --lora-path minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --inference-steps 4 \
  --output-dir outputs/ref2va_lora_4nfe
```

Ref2VA base model, 50 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_ref2va_test.json \
  --inference-steps 50 \
  --output-dir outputs/ref2va_base_50nfe
```

The Ref2VA jobs JSON must contain only `ref2va` examples. The script selects
`transformer_ref` automatically. Do not use an FL2VA LoRA checkpoint for this
command unless it was specifically trained for `transformer_ref`.

LoRA, 4 NFE, 768p (v1.0):

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_t2va_test.json \
  --lora-path minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --inference-steps 4 \
  --video-shift 6 \
  --lora-alpha 128 \
  --megapixels 1.0 \
  --aspect-ratio 16:9 \
  --output-dir outputs/lora_4nfe_768p
```

Base model, 50 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_t2va_test.json \
  --inference-steps 50 \
  --output-dir outputs/base_50nfe
```

To fuse LoRA weights into the model before inference, add `--fuse-lora`.


---

## FILE: examples/prompts_t2va_test.json (https://github.com/ModelTC/Minimax-H3-Turbo/blob/HEAD/examples/prompts_t2va_test.json)

```
{
  "examples": [
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a low-angle medium-wide tracking shot follows two athletic martial artists sparring on a rain-slick rooftop at blue hour, city lights glowing behind them. The camera arcs right at fast speed as the fighter in a red jacket ducks beneath a spinning kick, plants one hand on the wet concrete, sweeps the other fighter's legs, and both recover into guarded stances without injury. Water sprays from each footfall and a loose rooftop cable whips in the wind.\n\noverall_soundscape: Strong rooftop wind, rapid shoe scrapes, fabric snaps, controlled grunts, and three crisp padded impacts echo between nearby buildings. Distant traffic and a low thunder roll remain underneath.\n\nnon_diegetic_music: Fast taiko drums and short low-string ostinatos accelerate, then stop sharply on the final stance.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action sports documentary style, a handheld ringside close shot frames two female boxers wearing headguards and gloves during controlled sparring in a small neighborhood gym. The camera tracks left at fast speed as the boxer in blue slips a jab, answers with a rapid body-head combination against the pads, and pivots away while sweat droplets catch the overhead light in brief slow motion. Her coach with a firm, energetic voice (S1) calls: <d>[English] Move, breathe, reset!</d>\n\noverall_soundscape: Gloves thud against padded targets, shoes squeak on canvas, the round timer beeps once, and both athletes breathe sharply. A jump rope and muffled training activity continue in the background.\n\nnon_diegetic_music: A tight electronic beat at a fast tempo with clipped bass pulses, fading under the final breath.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, energetic urban realism, a wide tracking shot follows a young parkour runner in a gray hoodie sprinting across a sunlit concrete plaza. The camera trucks right with large amplitude at fast speed as he vaults a handrail, lands on a narrow wall, takes two balanced steps, and leaps across a short gap. [Shot 2] At 00:03.600, the camera cuts to a low close shot of his shoes absorbing the landing before he rolls forward and rises into a run.\n\noverall_soundscape: Fast footsteps slap concrete, palms brush metal, clothing flutters, and a landing produces a deep shoe impact followed by a quick roll. Pigeons flap away as distant city traffic continues.\n\nnon_diegetic_music: Syncopated hand percussion and muted electric bass at a fast tempo, ending with one sustained bass note.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, cinematic portrait, an extreme close-up frames an elderly East Asian woman's face beside a softly lit window at dawn, revealing fine skin texture, silver eyelashes, and a warm reflection in her dark eyes. The camera pushes in with small amplitude at slow speed as she looks just past the lens, inhales, and forms a restrained smile while one tear gathers and travels naturally down her cheek. Her lips remain closed and the pale curtain moves gently behind her.\n\noverall_soundscape: Quiet room tone, a slow breath, faint fabric movement, and distant morning birds are audible.\n\nnon_diegetic_music: Three widely spaced felt-piano notes with a soft sustained cello harmonic, gradually decreasing in volume.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, naturalistic, an over-the-shoulder medium close shot frames two old friends seated across from each other in a nearly empty late-night café, warm pendant lamps reflected in the dark window. The camera trucks left with small amplitude at slow speed as the tired man with a low, hesitant voice (S1) places a key on the table and says: <d>[English] I kept it all these years.</d> [Shot 2] At 00:03.200, the camera cuts to the woman with a soft, steady voice (S2) in close-up; she looks at the key, meets his eyes, and replies: <d>[English] Then open the door.</d>\n\noverall_soundscape: Rain taps the windows, a ceramic cup settles onto a saucer, and a refrigerator hum fills the quiet café.\n\nnon_diegetic_music: Sparse upright-piano chords at a slow tempo with a very soft brushed-cymbal swell.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action action-camera footage in first-person POV, the viewer races down a narrow forest mountain-bike trail in bright afternoon light. Gloved hands grip the handlebars as the camera shakes slightly at fast speed, weaving between pine trunks, dropping over a rocky ledge, splashing through a shallow stream, and banking hard around a muddy turn while the front suspension compresses visibly. Sunbeams flicker rapidly through the canopy and droplets strike the lens.\n\noverall_soundscape: The freewheel buzzes, tires grind over gravel, suspension knocks, water splashes, and the rider breathes hard while wind rush grows louder on the descent.\n\nnon_diegetic_music: N/A",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Photorealistic 3D CG, first-person POV from the pilot seat of a compact spacecraft waiting inside a dark orbital hangar. Instrument panels cast blue light across gloved hands as warning indicators pulse and the hangar doors separate to reveal a planet's bright curved horizon. The cockpit shakes strongly while the pilot pushes both throttles forward; engines ignite, the craft accelerates through drifting vapor, and the star field expands beyond the opening. A calm synthetic computer voice (S1) announces: <d>[English] Launch corridor clear.</d>\n\noverall_soundscape: Cooling fans hum, switches click, warning tones pulse twice, engines rise from a low vibration to a powerful roar, and the pilot's harness creaks under acceleration.\n\nnon_diegetic_music: Low brass, granular electronic pulses, and a rising string glissando build quickly before cutting to silence.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, cinematic landscape, an ultra-wide aerial view glides above a mirror-still alpine lake before sunrise, with dark fir trees in the foreground and snow-covered peaks filling the horizon. The camera pedestals up with large amplitude at slow speed as the first orange sunlight reaches the summit, travels down the ridges, and turns thin valley mist gold; a small flock of birds crosses the reflection and subtle ripples spread from the shore.\n\noverall_soundscape: A light mountain breeze moves through pine needles, distant birds call, and gentle water laps against stones in the otherwise open valley.\n\nnon_diegetic_music: Long French-horn tones and slowly expanding string chords at a very slow tempo, ending in a soft sustained note.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, dramatic maritime cinematography, a wide static shot frames a solitary lighthouse on black cliffs during a violent night storm. A towering wave rises, crashes across the rocks, and sends white spray past the lantern room as its rotating beam sweeps through rain. Lightning briefly reveals the entire coastline. [Shot 2] At 00:03.700, the camera cuts to a close shot of seawater streaming down the lighthouse glass while the warm lamp continues rotating behind it.\n\noverall_soundscape: Heavy rain lashes stone and glass, gale-force wind howls around the tower, surf crashes with deep impacts, and one thunderclap rolls across the coast.\n\nnon_diegetic_music: Low contrabass drones with slow metallic percussion strikes, rising in volume until the lightning flash.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action wildlife documentary style, a telephoto tracking shot follows a cheetah sprinting across dry golden grass in late-afternoon light while a gazelle changes direction ahead of it. The camera pans right with large amplitude at fast speed, keeping both animals sharp as dust trails behind their feet; the gazelle bounds over a fallen branch, the cheetah brakes and turns, and both continue safely out of frame without contact.\n\noverall_soundscape: Rapid paw impacts, grass brushing against bodies, short animal breaths, and wind over the open savanna dominate the scene. A few distant bird calls remain audible.\n\nnon_diegetic_music: Fast hand drums and plucked low strings repeat in an uneven rhythm, then diminish as the animals leave frame.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action macro cinematography, an extreme close-up shows a bumblebee clinging to a purple lavender flower just after rain, individual hairs and transparent wing veins sharply visible. The camera holds a static shot as a water droplet slides down the stem, the flower bends under its weight, and the bee opens its wings in slow motion. The droplet falls past the lens at the instant the bee lifts off, scattering tiny sparkling beads against a creamy green background.\n\noverall_soundscape: Close wing buzz, delicate leaf rustle, individual droplets touching petals, and soft garden ambience fill the scene.\n\nnon_diegetic_music: A sparse marimba pattern with high glass harmonics at a slow tempo, fading after the bee departs.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, high-contrast automotive commercial, a low tracking shot follows a red sports coupe entering a rain-soaked neon intersection at night. The camera arcs left at fast speed as the driver initiates a controlled drift, the rear tires sweep through standing water, and reflections of cyan and magenta signs stretch across the rotating bodywork. The car straightens precisely and accelerates toward a tunnel while vapor trails behind it.\n\noverall_soundscape: The engine revs rapidly, tires hiss and squeal on wet asphalt, water sprays against the wheel wells, and the exhaust cracks once as the car accelerates away.\n\nnon_diegetic_music: Fast electronic drums, distorted bass pulses, and a rising synthesizer arpeggio cut off at the tunnel entrance.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action food cinematography, a close overhead shot frames a chef's hands cooking noodles in a black steel wok over a blue gas flame in a busy open kitchen. The camera pushes in with small amplitude at normal speed as the chef tosses noodles and vegetables into the air, catches them cleanly, pours in sauce that flashes into steam, and finishes with chopped scallions. The middle-aged chef with a cheerful, resonant voice (S1) says: <d>[English] High heat, quick hands.</d>\n\noverall_soundscape: Oil sizzles continuously, the burner roars, metal utensils strike the wok, vegetables crackle, and nearby cooks call indistinctly over ventilation noise.\n\nnon_diegetic_music: A brisk pizzicato-string rhythm with light woodblock accents, ending on the final garnish.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Photorealistic industrial documentary style, a symmetrical wide shot frames an automated electric-car assembly line under cool white factory lights. The camera trucks right at slow speed beside an unfinished chassis as six orange robotic arms rotate in coordinated sequence, position a roof panel, clamp it, and fire a precise series of blue-white welding sparks. An autonomous cart crosses behind the chassis while safety lights change from amber to green.\n\noverall_soundscape: Servo motors whine, pneumatic clamps snap shut, conveyor rollers hum, warning chimes sound briefly, and welding crackles in short synchronized bursts.\n\nnon_diegetic_music: Minimal electronic percussion built from metallic clicks at a steady moderate tempo, with a low synthesizer tone rising slightly at the end.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action underwater cinematography, a wide tracking shot follows a lone scuba diver swimming above a bright coral reef in clear tropical water. The camera pulls out with large amplitude at slow speed as a giant manta ray emerges from blue haze, glides directly over the diver, and banks gracefully toward a column of sunbeams; the diver turns upward and releases a stream of silver bubbles while schools of small fish divide around them.\n\noverall_soundscape: Muffled regulator breaths and bubbles pulse rhythmically, with soft underwater current noise and distant reef clicks surrounding the diver.\n\nnon_diegetic_music: Sustained bowed-vibraphone tones and slow airy synthesizer chords, increasing gently as the manta ray passes overhead.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action survival drama, a medium-wide handheld shot frames a mountain rescuer in a red shell jacket struggling through a whiteout snowstorm with a search dog on a rope. The camera tracks backward at slow speed as the dog suddenly pulls left, digs beside a half-buried backpack, and barks; the rescuer drops to one knee, brushes away snow, and raises a radio. The wind-driven snow repeatedly obscures and reveals both figures. The rescuer with a breathless, urgent voice (S1) reports: <d>[English] I found the marker!</d>\n\noverall_soundscape: Fierce wind, crunching boots, strained breathing, rope hardware clinking, rapid digging, two sharp barks, and radio static fill the storm.\n\nnon_diegetic_music: Repeating low piano notes and sustained viola tones gradually increase in speed and volume.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Hand-drawn 2D animation with watercolor backgrounds, a wide shot frames a young mage standing on a cliff above a floating island city at sunset as a long blue dragon dives from the clouds. The camera tilts up with large amplitude at fast speed while the mage spins a staff, draws a glowing circular shield, and redirects the dragon's harmless stream of golden fire into a ribbon that spirals around both characters. The dragon folds its wings and lands beside the smiling mage.\n\noverall_soundscape: High wind crosses the cliff, wings beat with deep rhythmic force, the staff rings like glass, magical energy crackles, and the dragon releases a low friendly rumble.\n\nnon_diegetic_music: Rapid orchestral strings, two brass swells, and bright celesta notes resolve into one held chord.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Claymation stop-motion style, a miniature flour-dusted kitchen is seen in a medium-wide static shot as a thumb-sized baker made of red clay struggles to carry an oversized strawberry toward a tiny cake. The baker slips, the strawberry rolls in a circle around the mixing bowl, and three clay mice push it together until it lands perfectly on top of the cake; all four characters freeze, then cheer with raised arms. The tiny baker with a high, squeaky voice (S1) exclaims: <d>[English] Perfect teamwork!</d>\n\noverall_soundscape: Small clay footsteps make soft taps, the strawberry bumps against wooden utensils, the bowl rattles, and the mice squeak before a brief group cheer.\n\nnon_diegetic_music: Playful bassoon and pizzicato violin at a bouncy moderate tempo, ending with a short triangle note.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, vintage 16 mm film look, a medium-wide shot frames a four-piece jazz group performing under warm string lights in a narrow city alley at night while pedestrians gather nearby. The camera arcs right at slow speed as the saxophonist plays a bright solo, the upright bassist plucks steadily, the drummer brushes the snare, and two onlookers begin dancing. The live saxophone, bass, and drums are heard by everyone in the scene and remain synchronized with the musicians' fingers and breathing.\n\noverall_soundscape: The diegetic jazz performance fills the alley with saxophone phrases, walking bass, and brushed drums, mixed with quiet applause, shoe taps, and distant traffic.\n\nnon_diegetic_music: N/A",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action psychological horror, a first-person POV moves down a dim abandoned hospital corridor illuminated by a flickering fluorescent tube. The camera pushes in at slow speed with slight handheld shake as a wheelchair at the far end rolls sideways by itself, revealing an empty doorway; the light cuts out for one second, then returns with the wheelchair suddenly close to the camera. A gloved hand enters frame and stops against the wall while the camera remains fixed on the motionless chair.\n\noverall_soundscape: Fluorescent electricity buzzes and stutters, slow footsteps echo, a loose wheel squeaks, distant pipes knock once, and the unseen viewer's breathing becomes faster in the final second.\n\nnon_diegetic_music: A nearly inaudible low-frequency drone rises continuously, followed by one sharp reversed-piano strike when the light returns.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium close-up in warm daylight frames a skilled chef in a crisp white chef's coat and black pants standing at a wooden cutting board in a cluttered working kitchen. Strong side light streams through a large window, casting a soft glow across the chef's focused expression and the vibrant array of vegetables, while pots hang from overhead racks and neatly arranged ingredients line the shelves behind. The camera trucks right with small amplitude at slow speed as the chef swiftly chops the vegetables in precise, rapid strokes, gathers the cut pieces with the flat of the knife, and resumes the rhythmic knife work. Steam curls continuously from a simmering pot on the stove, drifting through the sunbeam as the chef's hands and blade move in a controlled, dance-like rhythm against the board.\n\noverall_soundscape: Rapid, even knife strikes tap against the wooden cutting board beneath a low kitchen room tone. Vegetables rustle and scrape as they are gathered, the simmering pot bubbles softly, steam hisses faintly, and hanging cookware gives an occasional light metallic clink.\n\nnon_diegetic_music: A brisk acoustic-percussion rhythm at a moderate-fast tempo, joined by light plucked strings and subtle pizzicato accents that follow the chopping cadence before ending with a short clean cadence.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide aerial shot looks down over a large city intersection at sunrise in clear daylight, rendered in warm golden colors. The streets and surrounding sidewalks are crowded with rush-hour traffic and pedestrians. The camera slowly descends with small amplitude while arcing around the intersection, keeping the entire crossing in view as streams of cars turn, merge, and weave through their lanes in organized chaos. Pedestrians hurry along the sidewalks, gather at the corners, and cross in dense groups when traffic pauses, their long shadows stretching across the pavement. Sunlight glints from brightly colored vehicle roofs and glass building facades while soft clouds drift lazily across the sky in the background. Engines surge as traffic begins moving again, several horns sound briefly, and rapid footsteps accompany the crowd crossing below.\n\noverall_soundscape: Layered rush-hour traffic fills the scene with engine hum, tire noise, occasional brief horns, and the muted hiss of vehicles accelerating through the intersection. Dense footsteps, faint crowd chatter, and intermittent crosswalk signals rise from the sidewalks beneath a light breeze high above the streets.\n\nnon_diegetic_music: A steady mid-tempo pulse of muted electronic percussion and low bass is joined by sustained string chords, gradually increasing in volume as the intersection fills with motion.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, cinematic, daytime, a balanced medium shot frames two athletic fencers dueling on a well-lit indoor training mat, one in a white fencing uniform and blue mesh mask and the other in a similar white uniform with a red mesh mask. Strong sunlight streams through large side windows, casting sharp lateral shadows that sweep across the floor as the camera tracks alongside the pair at fast speed. They advance and retreat in quick, controlled footwork, feint with rapid blade movements, parry in close succession, and lunge past each other as their swords meet with precise metallic clashes. The blue-masked fencer pivots away from a thrust and immediately counters; the red-masked fencer blocks, slides backward, then drives forward with another swift lunge. Their tense posture, fixed gaze through the masks, and decisive movements convey intense focus and determination, while spectators in the background lean forward and watch every exchange intently. The shot holds on both fencers as their blades lock briefly at center frame before they spring apart and reset their stances.\n\noverall_soundscape: Rapid shoe squeaks and stamping footwork move across the training mat beneath repeated sharp metallic blade clashes, brief scrapes, and the swish of fast lunges. The indoor hall carries light reverberation, controlled breathing from the fencers, and subdued reactions and shifting from the attentive spectators.\n\nnon_diegetic_music: Fast staccato strings over a tight, driving percussion rhythm, building in volume through the exchange and ending on a short sustained note as the fencers reset.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "t2va",
      "prompt": "integrated_multimodal_description: [Shot 1] Live-action, cinematic, daytime with warm sunny side lighting, a medium shot frames a determined man standing braced on a weathered wooden fishing boat beneath a clear blue sky with slowly drifting white clouds. Sweat streaks his face as he grips a taut fishing rod and pulls hard against the line, his shoulders and arms visibly tense. A large fish surges beside the boat and bursts out of the sunlit water, twisting through the air as bright droplets scatter over the highlighted waves. The fish plunges back into the sea, jerking the line and making the boat rock gently beneath the man; he widens his stance, leans back, and reels with controlled force while keeping his eyes fixed on the churning water. The camera tracks the fish's leap with small amplitude at fast speed, then settles back into the medium framing as the man continues the struggle against the open horizon.\n\noverall_soundscape: Small waves slap against the wooden hull beneath steady open-sea ambience and a light coastal breeze. The fishing reel clicks and whirs under strain, the line hums taut, wood creaks as the boat rocks, and the fish breaks the surface with a heavy splash while the man breathes hard and grunts with exertion.\n\nnon_diegetic_music: Low hand-drum pulses at a moderate tempo with short plucked-string accents, building in volume during the fish's leap and easing slightly as the struggle continues.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    }
  ]
}

```


---

## FILE: examples/prompts_i2va_test.json (https://github.com/ModelTC/Minimax-H3-Turbo/blob/HEAD/examples/prompts_i2va_test.json)

```
{
  "examples": [
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_1_chef.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, a warm daylight medium close-up preserves the adult male chef's short brown hair, trimmed beard, focused expression, white double-breasted chef's coat with black piping and buttons, black trousers, and his position at the wooden cutting board established in <Picture 1>. The crowded kitchen remains unchanged, with copper pans hanging behind him, jars and ingredients on the shelves, vivid vegetables and herbs filling the right foreground, and a metal pot steaming on the stove. Beginning with the knife resting against the cucumber, he curls the fingertips of his guiding hand inward and drives the broad blade through it in a quick sequence of clean, even downward cuts. The camera pushes in with small amplitude at slow speed toward his hands as thin cucumber rounds accumulate beside the chopped carrots and other vegetables. He finishes the final slice, turns the blade sideways, and uses its flat face to gather the fresh pieces into a neat pile while steam continues to billow behind him and sunlight glints along the moving knife.\n\noverall_soundscape: The knife strikes the wooden board in a fast, regular rhythm beneath the low hiss and bubbling of the simmering pot. Vegetables scrape softly across the board, fabric shifts with the chef's arm movements, and quiet kitchen room tone continues in the background.\n\nnon_diegetic_music: N/A",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_2_child_running.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, a low handheld medium shot preserves the blond young child's identity, white short-sleeved T-shirt, blue shorts, centered position, and the tall sunlit field established in <Picture 1>. Warm dawn backlight outlines the child's hair as the camera tracks backward at a lively pace, maintaining the frontal composition while the child runs toward it along the narrow path. The child's steps quicken, both arms gradually spread wide, and an open-mouthed laugh continues as grass heads brush against the child's legs. The child bends both knees, springs forward over a dense tuft of grass with arms fully outstretched, then lands and keeps running toward the retreating camera. Tall grass sways in the breeze around the path, drifting clouds move slowly across the pale blue sky, and the rising sun remains low behind the child, casting long golden highlights across the field. Gentle handheld movement responds naturally to each stride without losing the child's face or clothing details.\n\noverall_soundscape: Quick footfalls press through grass beneath the child's bright laughter and light breathing. Tall stems rustle and brush against clothing while a gentle breeze moves across the open field, with a soft thump as the child lands from the jump.\n\nnon_diegetic_music: N/A",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_3_guitar.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, <Picture 1> anchors a medium-wide, center-composed view of the same young guitarist on an outdoor stage, preserving their shoulder-length dark brown hair, black short-sleeved T-shirt, dark trousers, red guitar strap, wide stance, and downward gaze. They hold the same sharply angular white electric guitar, which blazes with intense golden-white light beneath the low sun at upper left; the stage truss remains at frame left, with dark storm clouds, silhouetted trees, and distant buildings behind them. The camera pushes in with small amplitude at slow speed as the guitarist begins strumming vigorously with the right hand while the left hand moves rapidly along the fretboard, producing a loud, distorted electric-guitar riff synchronized precisely to each stroke. The guitar’s existing glow pulses brighter around the body, strings, and neck; restrained golden electrical arcs crackle along its edges, and subtle heat-like pressure ripples spread through the nearby air with the strongest chords. The guitarist leans deeper into the performance and nods sharply to the rhythm, causing their hair to whip across their face, while the shirt, strap, and loose strands flutter in the energized breeze. The riff builds into a sustained final chord as the guitarist holds the low, forceful stance; the guitar flares once, the electrical arcs race toward the headstock, and the visible air ripples expand before fading against the dramatic clouded sky.\n\noverall_soundscape: Wind moves across the open stage, stirring hair and clothing while the guitar strap creaks faintly. Electrical crackles snap around the instrument, and each forceful chord carries a deep, thunder-like resonance through the surrounding air.\n\nnon_diegetic_music: N/A",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_4_hamster.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, <Picture 1> establishes a warm daytime medium-wide shot inside a pale wooden hamster enclosure: the same small golden-brown, black, and white hamster stands alert at the lower right beside the slatted wooden bridge, while the large empty wooden exercise wheel dominates the left side. Pale wood shavings and small colorful toys cover the floor; sunlight filters through the window, casting soft shadows and bright highlights across the enclosure, with a leafy green plant and bookshelf blurred in the background. The camera holds nearly static as the hamster twitches its whiskers, turns toward the wheel, and scurries left across the bedding. It climbs into the wheel and begins running, first cautiously and then swiftly, its tiny paws gripping the inner wooden track as the wheel accelerates into a smooth spin. Its patterned fur gleams in the sunlight, the bedding shifts beneath the wheel, and the hamster settles into a steady rapid stride while the spokes circle continuously behind it.\n\noverall_soundscape: Quiet indoor room tone and faint leaf rustling continue beneath the soft scratch of paws crossing wood shavings. The wheel begins with a few gentle wooden creaks, then produces a quick, steady rolling rhythm synchronized with the hamster’s rapid footfalls.\n\nnon_diegetic_music: A light pizzicato-string pattern at a brisk tempo with sparse wooden percussion, gradually becoming quicker and fuller as the wheel accelerates.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_5_helicopter.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, a low-angle wide shot explicitly begins from <Picture 1>: the same dark navy-black helicopter, with its rounded glazed cockpit, roof-mounted engine housing, long right-pointing tail boom, enclosed tail rotor, landing skids, and seated pilot silhouette, holds a very low hover above the paved landing area. Its scale, right-facing profile, central composition, warm backlit edges, and deep shadow remain consistent. The large sunrise glows behind the helicopter on the left, silhouetting the distant tree line beneath a pale blue-and-gold sky; small birds remain high in the distance, dark grass fills the foreground, and suspended golden dust drifts across the runway. The main and tail rotors accelerate into broader motion blur as their synchronized whir intensifies. Downwash flattens the nearby grass and drives dust and loose debris outward in expanding circular waves, glowing as they cross the sunbeams. The helicopter then rises smoothly from its low hover, its skids gaining clear separation from the ground while its moving shadow softens beneath it. It pitches forward only slightly and begins a controlled rightward acceleration while continuing to climb. The camera holds the restrained low viewpoint and tilts up with small amplitude at slow speed to keep the aircraft centered, allowing the sunlit dust cloud to spread below as the helicopter finishes several meters above the landing area.\n\noverall_soundscape: The helicopter engine builds into a strong sustained turbine roar while the main rotor produces increasingly rapid, rhythmic blade beats and the tail rotor adds a higher mechanical whine. Rotor wash rushes across the ground in sync with the lift, whipping grass and scattering dust and light debris as the aircraft climbs.\n\nnon_diegetic_music: N/A",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_6_rocket.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, <Picture 1> establishes a low-angle wide shot of the same slender white rocket with black and red bands standing upright at the center of its launch pad, flanked by the lattice mast on the left and the dark service tower on the right. The paved launch complex, low white buildings, sunlit grass, sweeping wispy clouds, and white contrails retain their exact positions beneath the vivid blue sky. The rising sun remains directly behind the rocket’s base, producing a brilliant golden backlight, long foreground shadows, and warm orange highlights across the enormous gray-brown exhaust cloud already billowing to the right. The engine plume intensifies beneath the rocket, glowing white-orange as turbulent smoke surges outward, curls over itself, and spreads low across the right side of the pad. The rocket begins to lift vertically, slowly clearing its supports before accelerating upward in one continuous ascent. The camera tilts up with small amplitude at slow speed to keep the climbing rocket centered while the launch tower and mast sink lower in the frame. A bright narrowing exhaust column connects the ascending rocket to the pad, and a dense smoke trail lengthens beneath it as the vehicle rises into the cloud-streaked morning sky; the ground-level smoke continues expanding and glowing along its sunlit edges.\n\noverall_soundscape: A deep engine rumble swells into a sustained thunderous roar as the plume brightens and the rocket lifts from the pad. Pressurized exhaust crackles sharply beneath the vehicle while turbulent smoke produces a broad rushing hiss, with the roar gradually becoming more distant as the rocket climbs.\n\nnon_diegetic_music: N/A",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_7_train_passenger.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, nighttime. <Picture 1> establishes a medium-wide view inside a high-speed train: the same young East Asian woman sits on the right in three-quarter profile, preserving her identity, long straight black hair, pearl stud earrings, black coat over a white crew neckline, serene half-smile, and gaze directed through the window. Her face and upper body remain sharply framed against the glass while her darker reflection occupies the center-left; overhead fluorescent fixtures, metal seat rails, layered window reflections, cool blue shadows, and softly blurred urban lights retain their exact opening composition. The camera holds nearly static with only subtle train vibration. As the train accelerates through the nighttime Japanese city, blue-white lamps and indistinct illuminated streets stream rapidly sideways beyond the window, their light briefly sweeping across her cheeks and hair. Her reflection glides continuously over the glass, stretches across passing bands of light, and momentarily softens into motion blur before becoming clearer again. She blinks once, slightly lifts her chin, and follows the passing scenery with a small, smooth movement of her eyes while her restrained smile remains. The shot ends with her calmly gazing outward as another bright cluster of city lights rushes past and her reflected profile slips farther across the window.\n\noverall_soundscape: A steady low train rumble and rhythmic wheel noise continue beneath the soft hum of the carriage ventilation and fluorescent fixtures. Passing track joints create gentle synchronized vibrations, with occasional muted metallic rattles from the seat rails as the exterior rush deepens during acceleration.\n\nnon_diegetic_music: N/A",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_8_flame_wisp.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Cinematic fantasy 3D animation with photorealistic textures, explicitly beginning from the exact wide composition in <Picture 1>: a small feminine blue flame wisp with a bright cyan face, swept flame-shaped hair, luminous translucent limbs, and a short ragged dark skirt hovers left of center above a produce stall, her brilliant blue-white flame trail extending behind her. Preserve the bustling wooden market, fruit and bread displays, barrels, hanging baskets, canvas awnings and pennants, wet cobblestones, leafy trees, warm lanterns, the orange cooking fire at the right stall, the nearby vendors and customers, and the curly-haired man seen from behind in the right foreground. Cool blue nighttime illumination fills the lane while lanterns and the cooking fire cast soft amber highlights and flickering shadows. The wisp glances toward the open aisle, leans forward, and accelerates with a brief flare of blue fire. The camera pushes in with small amplitude at slow speed and pans gently right to follow as she darts into the central lane, dips above the cobblestones, then rises and curves between the produce stall and the orange-lit counter without touching either. Her flame hair and trailing plume stream backward, brighten during each burst of speed, and ripple across nearby awnings, faces, and stones with moving cyan light. Vendors continue serving their stalls and call out indistinctly while customers stroll, turn their heads, and step slightly aside as the wisp passes. She sweeps around the edge of the cooking fire, where blue and orange light briefly mingle, then slows near the center-right of the market, turns in midair to look back at the watching crowd, and settles into a buoyant hover as her long flame trail contracts into a softly flickering glow.\n\noverall_soundscape: Layered market chatter and indistinct vendor calls mingle with footsteps on damp cobblestones, soft fabric movement from the awnings, wooden stall creaks, and occasional clinks from goods being handled. Lantern flames and the cooking fire crackle steadily; the blue wisp produces a light airy hiss while hovering and synchronized rushing whooshes whenever she accelerates or changes direction.\n\nnon_diegetic_music: N/A",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_9_skateboarder.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, a warm daytime medium-wide shot begins from the exact side-on composition in <Picture 1>: the long-haired skateboarder in a charcoal-gray T-shirt, black pants, black sneakers, and a red wristband rides from the left foreground along the broad concrete sidewalk beside the busy city intersection. Golden side light silhouettes the skateboarder's profile and streams through the roadside trees while cars and distant pedestrians continue moving through the background. A low tracking shot keeps pace beside the skateboarder as they compress into the existing crouch, snap the tail against the pavement, and lift into a clean ollie over a transverse sidewalk seam. Their hair and shirt move in the slipstream as the board levels beneath both feet; they land smoothly with knees absorbing the impact, carve gently toward the open center of the sidewalk, then build speed and execute a controlled frontside 180, turning shoulders, hips, and board together. The wheels reconnect with the pavement in alignment, and the skateboarder rides away toward the intersection while the camera continues tracking at the same warm, low angle.\n\noverall_soundscape: Skateboard wheels roll steadily over concrete beneath light city traffic and scattered pedestrian ambience. The tail cracks against the pavement for each takeoff, followed by brief airborne silence, firm wheel landings, and soft clothing movement in the passing air.\n\nnon_diegetic_music: A brisk instrumental groove led by muted electric guitar, dry kick and snare, and a steady bass line, with a short cymbal lift under the second jump and a clean continuation through the ride-away.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_10_lava_skater.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Cinematic, photorealistic, a wide rear view explicitly begins from <Picture 1>, preserving the solitary centered skater’s identity and silhouette, black jacket, dark fitted pants, orange-red cap and exposed orange waistband, along with the exact low mountain horizon, star-filled navy sky, snow-rimmed frozen lava field, glossy black ice plates separated by glowing orange-red fissures, and harsh blue-white overhead spotlight. Initially balanced mid-glide with one leg trailing and both arms slightly extended, the skater lowers their stance and presses the trailing skate outward. The blade scrapes the glossy crust as they accelerate away from the camera, then transfer weight smoothly from side to side in a shallow weaving path. Each carve leaves a thin pale score across the black surface; small cracks briefly branch near the blade before stopping at the existing molten seams, releasing short synchronized wisps of steam. Orange lava light flickers across the skater’s lower body and ripples in the wet-looking reflections while the hard spotlight keeps the upper silhouette sharply outlined and throws a long shifting shadow toward the foreground. The camera tracks forward at slow speed with small amplitude, maintaining the centered wide composition as the skater completes one broad curve, straightens, and continues gliding steadily toward the misty horizon beneath the fixed stars.\n\noverall_soundscape: Skate blades produce crisp, continuous scrapes that sharpen during each push and curve. A low volcanic rumble persists beneath the faint hiss of steam, occasional delicate ice cracking, light wind across the open field, and subtle clothing flutter synchronized with the skater’s motion.\n\nnon_diegetic_music: Sparse low synthesizer pulses at a slow tempo under sustained glassy high tones, gradually increasing in volume as the skater accelerates and easing back as the glide steadies.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_11_rally_car.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, <Picture 1> anchors a sunlit daytime wide shot of an empty tan gravel racetrack sweeping through a sharp bend, preserving the balanced composition, dark metal guardrails, grassy verges, dense green trees, overhanging leaves, blue sky, long side-lit shadows, existing tire marks, and pale dust plume rising at the far-left background. The camera initially holds static as the dust billows and drifts through the trees; an approaching rally engine rapidly grows louder before a rally car bursts from the left background through the existing dust. The car accelerates into the bend, countersteers through a controlled slide, and leans slightly as its tires bite into the loose gravel, flinging a low spray of stones and extending the dust trail behind it. The camera pans left with small amplitude at fast speed to keep the car in the wide composition as it sweeps across the curve toward the near-left foreground and exits frame. The expanding dust cloud rolls over the track in its wake while sunlit leaves and fine roadside grass continue trembling in the breeze.\n\noverall_soundscape: Wind rustles the tree canopy and roadside grass beneath the rapidly approaching growl of a rally engine. Gravel rattles against the underbody and guardrail as the tires scrape and hiss through the turn, followed by a sharp acceleration note that recedes after the car exits while airborne dust and small stones settle.\n\nnon_diegetic_music: N/A",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "i2va",
      "image": "i2va_testset/i2va_test_12_meteor.jpg",
      "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Cinematic, high-detail 3D CG, the wide orbital composition established by <Picture 1> is preserved: a dark rocky meteor descends diagonally from the upper left toward the blue, cloud-covered planet filling the lower half of the frame, trailing a bright orange-white flame and a thick blue-gray smoke plume against a dense black star field. Hard sunlight outlines the planet’s curved horizon with a cool blue atmospheric rim, while the meteor casts the only intense warm light. The flame flickers and sheds small incandescent fragments as the meteor accelerates downward; its smoke trail lengthens, curls, and disperses behind it. The camera tracks the meteor with small amplitude at slow speed while keeping the curved horizon and star field stable. As the meteor penetrates the atmosphere, its fire envelope widens and glows nearly white, briefly lighting the nearby cloud tops orange. It strikes a distant point on the visible surface in a concentrated flash, followed by a rapidly expanding fireball and a circular shock front spreading across the cloud layer. Glowing debris arcs outward from the impact zone as dense dark ash clouds billow upward around the flames. The fireball reaches its maximum size, leaving a radiant impact scar surrounded by churning smoke and scattered molten fragments.\n\noverall_soundscape: A low atmospheric roar rises with the meteor’s descent, joined by the sharp hiss and crackle of burning fragments. The impact produces a deep, heavy boom followed by a sustained rumble as debris erupts and the shock front expands.\n\nnon_diegetic_music: Sustained low brass and sub-bass drones build gradually beneath slow percussion pulses, swelling into a single powerful orchestral strike at impact before fading under the lingering rumble.",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    }
  ]
}

```


---

## FILE: examples/prompts_ref2va_test.json (https://github.com/ModelTC/Minimax-H3-Turbo/blob/HEAD/examples/prompts_ref2va_test.json)

```
{
  "examples": [
    {
      "task": "ref2va",
      "references": [
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_1_1.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_1_2.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_1_3.jpg"
        }
      ],
      "prompt": "subject_definitions:\n<Subject 1> is the long-haired East Asian female hunter in <Picture 1>, preserving her face, long wet black hair, black leather combat suit, tattered black cloak, black gloves, and single silver short blade.\n<Subject 2> is the East Asian female tech hunter in <Picture 2>, preserving her face, short black hair, white fox mask with red markings, black trench coat with cyan circuit lines, black boots, and single blue energy dagger.\n<Subject 3> is the rainy cyberpunk intersection in <Picture 3>, preserving its skyscrapers, giant blue-purple holographic billboard, neon signs, wet reflective asphalt, rain, and drifting fog.\n\nsummary:\n[reference generation] In one continuous 5.1667-second cinematic shot, <Subject 1> and <Subject 2> begin in a tense standoff inside <Subject 3>, charge across the wet street, exchange one fast blade combination, and end locked blade-to-blade in a burst of blue energy.\n\nretention_analysis:\n<Subject 1> (appears throughout): fully_preserved - retain the same face, long hair, black combat suit, cloak, body proportions, and one silver blade.\n<Subject 2> (appears throughout): fully_preserved - retain the same face, short hair, fox mask, cyan-lit trench coat, body proportions, and one energy dagger.\n<Subject 3> (appears throughout): fully_preserved - retain the wet intersection, billboard, neon palette, rain, fog, and reflective pavement.\n\ndetailed_description:\nPhotorealistic live-action cyberpunk action cinema, 16:9, high contrast purple-and-cyan night lighting, heavy rain, physically coherent motion, and stable identities. One continuous shot with no cuts, no duplicate people, no duplicate weapons, no costume changes, no gore, and no readable text.\n\n[Shot 1, 00:00.000-00:01.200] A low wide camera frames both fighters five meters apart on the rain-soaked street. <Subject 1> crouches on frame left with her silver blade held low; <Subject 2> stands on frame right with the blue energy dagger raised. Rain streaks through the holographic light and fog crosses the reflective road.\n\n[Shot 1, 00:01.200-00:03.600] Both women burst forward. The camera tracks sideways at waist height as <Subject 1>'s cloak opens behind her. She makes one upward diagonal slash; <Subject 2> sidesteps and blocks with the energy dagger. A compact blue-white spark and a ring of displaced droplets mark the impact. <Subject 2> answers with one horizontal counter, which <Subject 1> deflects with the flat of her blade.\n\n[Shot 1, 00:03.600-00:05.166] They rotate once around their joined weapons and stop in a close blade lock, faces and costumes still distinct. Cyan electricity crawls briefly across the crossed blades while rain runs from the fox mask and black cloak. The camera settles into a medium two-shot as the billboard flickers behind them and both hold the unresolved confrontation.\n\noverall_soundscape:\nHeavy rain, wet footfalls, two sharp blade impacts, controlled fabric movement, electrical crackle, distant traffic, and a low thunder roll. No dialogue.\n\nnon_diegetic_music:\nA dark electronic pulse rises through the charge and stops on the final blade lock.\n",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "ref2va",
      "references": [
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_2_1.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_2_2.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_2_3.jpg"
        }
      ],
      "prompt": "subject_definitions:\n<Subject 1> is the fictional Victorian detective in <Picture 3>, preserving his angular clean-shaven face, combed-back dark hair, long dark overcoat, brown waistcoat, cream cravat, watch chain, and briar pipe.\n<Subject 2> is the fictional Victorian doctor in <Picture 2>, preserving his sturdy build, cropped brown hair, heavy moustache, brown tweed suit, dark tie, and wooden cane.\n<Subject 3> is the Baker Street study in <Picture 1>, preserving the wood paneling, lit fireplace, bookshelves, rain-streaked windows, leather armchair, desk, chandelier, rug, and warm-firelight-versus-cool-window-light composition.\n\nsummary:\n[reference generation] In one continuous 5.1667-second locked-off Victorian dialogue shot, <Subject 2> asks whether <Subject 1> could have become a criminal; <Subject 1> answers with dry confidence while both remain naturally alive in <Subject 3>.\n\nretention_analysis:\n<Subject 1> (appears throughout): fully_preserved - retain his face, hair, clothing, pipe, slim proportions, and fireplace-side placement.\n<Subject 2> (appears throughout): fully_preserved - retain his face, moustache, tweed suit, cane, sturdy proportions, and armchair placement.\n<Subject 3> (appears throughout): fully_preserved - retain the room layout, fireplace, shelves, windows, desk, furniture, rain, and mixed warm-cool lighting.\n\ndetailed_description:\nPhotorealistic live-action Victorian feature-film style, restrained 35 mm texture, natural skin, subtle performance, and consistent eyelines. One continuous medium-wide two-shot from a fully static locked-off camera. No cuts, camera movement, identity swapping, wardrobe changes, extra people, readable text, or exaggerated gestures.\n\n[Shot 1, 00:00.000-00:01.300] <Subject 1> stands beside the fireplace on frame left, his pipe held near his chest, while <Subject 2> sits in the leather armchair on frame right with one hand resting on his cane. Wind-driven rain traces the cool windows behind them. A coal shifts in the grate, briefly sharpening warm firelight across <Subject 1>'s angular profile and <Subject 2>'s moustached face.\n\n[Shot 1, 00:01.300-00:03.200] <Subject 2> leans forward just enough to make the leather chair creak, studies the detective, and asks with familiar curiosity, <d>[English] What if you'd chosen crime?</d> His fingers tighten once around the cane handle. <Subject 1> remains still for a beat, draws quietly from the pipe, and turns only his eyes toward the doctor as smoke curls through the warm light.\n\n[Shot 1, 00:03.200-00:05.166] <Subject 1> meets his eyeline and replies in a low, dry voice, <d>[English] I'd have been remarkably successful.</d> <Subject 2>'s amused expression tightens into thoughtful concern. <Subject 1> allows the faintest private smile as the fire gives one crisp crackle and a distant lightning reflection passes across the rain-streaked glass. Both settle into the charged silence without changing position.\n\noverall_soundscape:\nSteady rain on glass, close coal-fire crackle, faint room tone, one subtle leather creak, and distant carriage wheels. Dialogue stays clear and intimate.\n\nnon_diegetic_music:\nN/A\n",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "ref2va",
      "references": [
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_3_1.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_3_2.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_3_3.jpg"
        }
      ],
      "prompt": "subject_definitions:\n<Subject 1> is the abandoned warehouse in <Picture 1>, preserving the open doorway, long wooden table, two chairs, wet reflective floor, storage racks, crates, industrial architecture, sparse overhead lamps, cold rain light, and dim warm practical light.\n<Subject 2> is the younger East Asian man in <Picture 2>, preserving his face, lean build, short messy black hair, dark layered hooded jacket, black cargo pants, work boots, and one plain black backpack.\n<Subject 3> is the middle-aged East Asian man in <Picture 3>, preserving his face, stocky build, slicked-back black hair, black patterned shirt, black trousers, belt, and leather shoes.\n\nsummary:\n[reference generation] In one continuous 5.1667-second crime-thriller shot, <Subject 2> crosses the final steps from the rainy doorway to the table in <Subject 1>, removes his backpack, and places it between himself and the seated <Subject 3>, ending in a compressed silent confrontation.\n\nretention_analysis:\n<Subject 1> (appears throughout): fully_preserved - retain the warehouse geometry, doorway, table, chairs, wet floor, racks, rain, and mixed cold-warm lighting.\n<Subject 2> (appears throughout): fully_preserved - retain his face, hair, lean proportions, dark clothing, wet appearance, and exactly one backpack.\n<Subject 3> (appears throughout): fully_preserved - retain his face, hair, stocky proportions, black clothing, seated posture, and position across the table.\n\ndetailed_description:\nPhotorealistic live-action crime thriller, 16:9, restrained contrast, cold blue rain light and dim amber practical light. One continuous shot with a slow small push-in. Preserve both identities, the spatial axis, and one physically consistent backpack. No dialogue, bodyguards, extra people, duplicate bags, identity swapping, wardrobe changes, readable text, or weapons.\n\n[Shot 1, 00:00.000-00:01.400] From low beside the long table, the camera looks directly toward the rain-filled doorway. <Subject 3> is a dark motionless silhouette at the far end. <Subject 2> strides the final few steps from the doorway with the backpack secured over both shoulders; cold backlight outlines his wet hood and shoulders while his boots kick shallow water across the reflective concrete. The camera pushes rapidly but smoothly toward the table.\n\n[Shot 1, 00:01.400-00:03.450] <Subject 2> reaches the near chair without stopping. In one continuous motion he hooks one strap off his shoulder, rotates his torso, slides the second strap free, and swings the same wet backpack onto the tabletop. It lands firmly between the two men with a heavy impact; water droplets scatter across the wood and the hanging lamp trembles. <Subject 3> does not flinch, following the bag only with his eyes.\n\n[Shot 1, 00:03.450-00:05.166] <Subject 2> drags the near chair back with one sharp scrape and drops into it opposite <Subject 3>. He plants both hands on the wet table without touching the backpack. A lightning flash turns both faces stark for an instant as the camera completes its push-in: the single bag remains centered between them and they lock eyes across it while rain lashes the doorway.\n\noverall_soundscape:\nHeavy rain on metal, distant thunder, measured wet footsteps, soaked fabric and backpack straps, one muted bag impact on wood, and low warehouse room tone. No dialogue.\n\nnon_diegetic_music:\nA restrained low-frequency drone rises slightly as the backpack reaches the table and holds through the final stare.\n",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "ref2va",
      "references": [
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_4_1.jpg"
        }
      ],
      "prompt": "subject_definitions:\n<Subject 1> is a tan-skinned adult man with close-cropped dark hair, light stubble, a sand-colored field jacket, and a burgundy scarf, as shown in <Picture 1>.\n<Subject 2> is a vintage brass telescope on a dark tripod, as shown in <Picture 1>.\n<Subject 3> is a small red-glowing metal lantern on a weathered table, as shown in <Picture 1>.\n<Subject 4> is a quiet desert observatory at dusk with soft mountain silhouettes, as shown in <Picture 1>.\n\nsummary:\n[reference generation] In one continuous 5.1667-second shot inside <Subject 4>, <Subject 1> notices a distant flash, turns toward <Subject 2>, adjusts its focus ring once, and reaches the eyepiece as <Subject 3> pulses red.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - Preserve the same adult identity, facial structure, hair, stubble, jacket, scarf, and body proportions.\n<Subject 2> (appears in [Shot 1]): fully_preserved - Preserve the same brass tube, black fittings, tripod geometry, scale, and position.\n<Subject 3> (appears in [Shot 1]): fully_preserved - Preserve the same metal housing and red light.\n<Subject 4> (appears in [Shot 1]): fully_preserved - Preserve the dusk palette, mountain horizon, and sparse desert setting.\n\ndetailed_description:\nVisual style: cinematic naturalism with warm lantern light against a cool violet dusk. One locked medium shot preserves the composition from <Picture 1>. The camera stays fixed with no pan, tilt, zoom, or dolly.\n\n[Shot 1, 00:00.000-00:01.300] <Subject 1> stands beside <Subject 2> as the final dusk light outlines the brass tube and distant mountain ridge. A faint cold flash crosses his eyes from outside frame. He stops breathing for a beat and turns his gaze sharply toward the telescope while <Subject 3> holds a steady red glow on the table.\n\n[Shot 1, 00:01.300-00:03.600] He reaches across the brass tube, grips the focus ring, and makes exactly one deliberate adjustment. The mechanism gives a precise metal click. Reflected violet sky slides across the brass surface, his scarf lifts in the desert wind, and <Subject 3> flickers once brighter without moving.\n\n[Shot 1, 00:03.600-00:05.166] <Subject 1> leans quickly but naturally to the eyepiece, steadies the telescope with his free hand, and peers through it with sudden concentration. The lantern throws a brief red edge across his cheek and telescope fittings. He holds the tense viewing pose as the distant flash fades behind the mountains.\n\noverall_soundscape:\nA faint desert breeze, a soft metal adjustment click, fabric rustle, and a quiet lantern flame.\n\nnon_diegetic_music:\nN/A\n",
      "duration": 5.1666666667,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "ref2va",
      "references": [
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_5_1.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_5_2.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_5_3.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_5_4.jpg"
        }
      ],
      "prompt": "subject_definitions:\n<Subject 1> is the East Asian woman sword fighter shown in <Picture 1>, preserving her facial identity, black hair tied in a high bun, focused expression, cream martial-arts jacket, and black-handled katana.\n<Subject 2> is the pale-skinned woman fighter shown in <Picture 2>, preserving her facial identity, short layered black hair with bangs, dark eye makeup, black lipstick, black leather jacket, and black clothing. The motorcycle is not present in the target video.\n<Subject 3> is the outdoor competition swimming pool shown in <Picture 3>, preserving the bright turquoise water, blue-and-white lane dividers, pale pool deck, and strong daylight.\n<Subject 4> is the underwater pool environment shown in <Picture 4>, preserving the cyan water, tiled pool floor, dark lane marking, moving caustic light, suspended bubbles, and refracted surface light.\n\nsummary:\n[reference generation] In one continuous 10-second action shot, <Subject 1> and <Subject 2> sprint into a violent exchange beside <Subject 3>. A rolling dodge leads into a left-fist versus right-hand-sword collision, a jumping counter, and a connected two-body tumble across the wet deck. Their momentum throws both women into the pool, where the confrontation finishes underwater inside <Subject 4>.\n\nretention_analysis:\n<Subject 1> (appears throughout): fully_preserved - retain the same face, high bun, cream jacket, body proportions, and single black-handled katana before and after entering the water.\n<Subject 2> (appears throughout): fully_preserved - retain the same face, short black hair, dark makeup, black lipstick, black leather jacket, black clothing, and body proportions before and after entering the water.\n<Subject 3> (appears before the underwater transition): fully_preserved - retain the competition-pool geometry, turquoise water, blue-and-white lane ropes, sunlit deck, and bright daylight.\n<Subject 4> (appears after the underwater transition): fully_preserved - retain the tiled floor, dark lane marking, cyan water, caustic illumination, bubbles, and refracted sunlight.\n\ndetailed_description:\nPhotorealistic live-action action cinema, 16:9 composition, hard bright daylight above water and luminous cyan light below water. Use one continuous aggressively tracking shot with no cuts: low lateral sprint tracking, one fast camera roll during the deck tumble, then a seamless plunge through the waterline. Keep both adult women clearly distinguishable by face, hairstyle, clothing color, and silhouette. <Subject 1> holds exactly one katana in her right hand throughout. <Subject 2> remains unarmed. Any fist-to-sword contact strikes the flat near the guard, never the sharpened edge. No blood, wounds, severed contact, duplicate people, duplicate weapons, motorcycle, readable text, logos, or costume changes.\n\n[Shot 1, 00:00.000-00:02.000] A low handheld camera races parallel to the wet edge of <Subject 3>. <Subject 1> in the cream jacket and <Subject 2> in black leather sprint toward each other from opposite ends of the pool deck. <Subject 1> carries the single katana low in her right hand, blade trailing safely behind. She plants one foot on a low starting block, leaps forward, and brings the sword into a fast diagonal arc. <Subject 2> drops beneath it into a forward shoulder roll, slides across the wet tiles, and rises inside the sword's range as water sprays from both fighters' shoes.\n\n[Shot 1, 00:02.000-00:04.200] <Subject 2> drives upward from the roll with one explosive left punch. <Subject 1> snaps the katana across in her right hand so the punch collides with the flat of the blade immediately beside the guard. The impact produces one sharp metallic crack, a compact spray of sparks, and a ring of displaced droplets. Their faces pass within centimeters. <Subject 2> hooks the sword wrist with her right hand and whips a low leg sweep; <Subject 1> jumps over it, twists in the air, and lands one step from the pool edge without releasing the sword.\n\n[Shot 1, 00:04.200-00:06.800] <Subject 2> immediately tackles <Subject 1> around the waist. <Subject 1> braces her left forearm across <Subject 2>'s shoulder and extends her right sword arm away from both bodies. Locked together, they crash onto the wet deck and complete two fast connected shoulder-over-shoulder rolls. The camera rolls with them at tile level as sky, lane ropes, faces, black leather, and cream fabric whip across frame. <Subject 1> hooks <Subject 2>'s ankle during the second rotation; their combined momentum carries both bodies over the pool lip.\n\n[Shot 1, 00:06.800-00:07.900] Both women rotate once in open air and strike the turquoise surface together in an enormous splash. The camera plunges between them without a cut, crossing the violently folding waterline as daylight becomes refracted cyan and a dense wall of bubbles engulfs the frame.\n\n[Shot 1, 00:07.900-00:10.000] Inside <Subject 4>, their clothes and hair surge in opposite currents while both identities remain clear. <Subject 2> throws one slowed left punch through the bubbles. <Subject 1>, still gripping the single katana in her right hand, blocks with the hilt and her left forearm while keeping the blade pointed safely away. The impact rolls both women apart above the dark lane marking. They kick backward, regain facing guard positions, and end suspended on opposite sides of a rising column of bubbles as caustic light flashes across their faces and the tiled floor.\n\noverall_soundscape:\nRapid wet footfalls, starting-block impact, sword whoosh, rolling bodies, one metallic fist-to-blade crack, sparks, tile scrapes, a heavy synchronized splash, dense bubbles, muffled underwater impact, and filtered pool ambience.\n\nnon_diegetic_music:\nFast taiko-like percussion accelerates through the sprint and deck rolls, cuts completely at the splash, then returns underwater as two muffled bass pulses ending on one suspended metallic note.\n",
      "duration": 10.0,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "ref2va",
      "references": [
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_6_1.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_6_2.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_6_3.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_6_4.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_6_5.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_6_6.jpg"
        }
      ],
      "prompt": "subject_definitions:\n<Subject 1> is the large football stadium in <Picture 1>, preserving its green striped pitch, white field markings, blue seating tiers, open roofline, and broad daylight scale.\n<Subject 2> is the football trophy in <Picture 2>, preserving its gold supports, red central column, black base, and fixed black-and-white football at the top. The football remains permanently attached to the trophy and never becomes the object being kicked.\n<Subject 3> is the purple cosmic energy phenomenon in <Picture 3>, preserving its black star field, luminous magenta core, violet plasma layers, and sparkling particles. It defines the high-altitude explosion after the meteor is kicked back into the sky.\n<Subject 4> is the adult woman in the orange astronaut suit in <Picture 4>, preserving her facial identity, brown hair, orange flight suit with black trim, body proportions, white boots, and black helmet. She is the football champion standing on the pitch; the helmet rests on the grass beside the trophy rather than covering her face.\n<Subject 5> is the severe storm sky in <Picture 5>, preserving the dense charcoal clouds, layered turbulent cloud texture, dim gray light, and sudden threatening weather.\n<Subject 6> is the meteor approaching Earth in <Picture 6>, preserving the curved blue Earth horizon, black space, incandescent orange meteor core, fragmented fiery tail, smoke plume, and atmospheric entry direction. It defines the single incoming meteor before and during the kick.\n\nsummary:\n[reference generation] In one continuous 10-second science-fiction sports shot inside <Subject 1>, <Subject 4> has just received <Subject 2>. She notices <Subject 6>'s fiery meteor descending from above Earth toward the pitch, sets the trophy safely beside her, and kicks the meteor back into the sky like a football. It explodes high above the stadium as <Subject 3>'s purple cosmic energy, and the bright weather instantly transforms into <Subject 5>'s dark storm sky.\n\nretention_analysis:\n<Subject 1> (appears throughout): fully_preserved - retain the pitch geometry, white markings, blue seats, stadium scale, and recognizable bowl-shaped architecture through the weather transition.\n<Subject 2> (appears throughout): fully_preserved - retain exactly one intact gold-red-black trophy with its football permanently attached, first in the champion's hands and then upright on the grass.\n<Subject 3> (appears after the kick): fully_preserved - retain the magenta core, violet plasma layers, black-space contrast, and particles in the distant high-altitude explosion.\n<Subject 4> (appears throughout): fully_preserved - retain the same adult woman's face, hair, orange astronaut suit, black trim, proportions, and white boots during the full-body kick.\n<Subject 5> (appears after the explosion): attribute_transfer - the charcoal cloud mass, turbulent texture, and dim storm lighting replace the original clear daylight without changing the stadium or people.\n<Subject 6> (appears before and during the kick): partially_preserved - retain the incandescent orange core, fragmented fiery tail, smoke plume, and atmospheric-entry direction while reducing the meteor to a kickable football-sized object near the pitch.\n\ndetailed_description:\nPhotorealistic live-action science-fiction sports cinema, 16:9, large stadium scale, physically readable staging, and one continuous camera move with no cuts. Keep exactly one woman, one intact trophy, one black helmet, and one meteor. The trophy's football never detaches. No spectators, additional players, duplicate objects, readable text, logos, injuries, ground impact, stadium damage, or identity changes.\n\n[Shot 1, 00:00.000-00:02.500] A low medium-wide camera faces <Subject 4> near the center circle of <Subject 1>. She stands proudly in the orange astronaut suit, holding <Subject 2> at chest height; her uncovered face is clearly visible, and the black helmet rests on the grass at her left. The stadium is empty under bright daylight. A hot orange reflection suddenly crosses the trophy and her face. She looks upward as one football-sized meteor derived from <Subject 6>, with an incandescent orange core, fragmented fiery tail, and narrow smoke plume, descends steeply toward the pitch behind her. For a brief instant, the high sky reveals the curved blue Earth-horizon composition from <Picture 6> before the camera returns to stadium scale.\n\n[Shot 1, 00:02.500-00:05.500] <Subject 4> places <Subject 2> upright on the grass beside the helmet without dropping or deforming it. She takes two quick steps toward the descending meteor. The camera tracks sideways and lowers to keep her entire body, the stationary trophy, and the meteor visible. Just before the meteor reaches knee height, she plants her left foot and strikes it once with the instep of her right boot in a powerful football kick. Her leg anatomy and balance remain coherent. The meteor compresses slightly at contact, then reverses direction and rockets upward, retaining <Subject 6>'s orange core, fiery fragments, and smoke tail. The trophy remains intact and stationary behind her.\n\n[Shot 1, 00:05.500-00:10.000] The camera whip-tilts upward following the meteor. High above the open stadium, it expands into <Subject 3>'s broad magenta-violet cosmic explosion without touching the ground. A deep shockwave rolls across the clouds. Bright daylight collapses into <Subject 5>'s dense charcoal storm sky within one visible atmospheric transition; turbulent clouds spiral around the fading purple core and the stadium light becomes cold and dim. The camera tilts down slightly for the final wide composition: <Subject 4> stands in a completed kicking pose on the pitch, looking upward, while the intact trophy and helmet remain beside her and the darkened stadium is silhouetted beneath the distant explosion.\n\noverall_soundscape:\nOpen-stadium wind, a low descending cosmic rumble, trophy contact with grass, two quick footsteps, one heavy boot impact, a sharp upward plasma whoosh, a distant high-altitude explosion, rolling thunder, and sudden storm wind. No dialogue or crowd.\n\nnon_diegetic_music:\nA brief heroic brass pulse rises during the approach, stops at the kick, and returns as a low ominous chord when the storm clouds cover the stadium.\n",
      "duration": 10.0,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    },
    {
      "task": "ref2va",
      "prompt": "subject_definitions:\n<Picture 1> defines the Martian surface below, preserving the layered ochre ridges, wind-shaped ground, barren geology, and immense planetary scale. Do not reproduce panorama seams, borders, rover hardware, vegetation, or buildings.\n<Picture 2> defines <Subject 1>, the sole human defense operator. Preserve her face, skin tone, short natural hair, age, and body proportions. Ignore her studio clothing and backdrop.\n<Picture 3> defines <Suit 1>, the operator's dark retro-futuristic pressure suit, helmet, padded torso, gloves, hoses, and life-support details. Transfer it onto <Subject 1>; ignore the original wearer and cabin. Keep <Subject 1>'s frightened face visible through the visor.\n<Picture 4> defines <Weapon 1>, one massive white-and-graphite orbital rail cannon with a long rectangular barrel, rotating armored base, exposed joints, and cyan energy accents.\n<Picture 5> defines <Craft 1>, one dark graphite alien flying saucer with a low disc body, segmented rim, cyan underside lights, swept fins, and transparent central dome.\n<Picture 6> defines <Subject 2>, the sole alien pilot. Preserve its blue-gray skin, elongated head, large dark eyes, slender proportions, and fitted black suit with cyan seams.\n<Picture 7> defines <Vessel 1>, the human defense spacecraft and its exposed armored weapons deck. Preserve the white-and-graphite plates, cyan guide lights, rails, barriers, and broad cannon platform. The entire deck belongs to one falling spacecraft rather than a fixed station.\n<Picture 8> defines <Console 1>, one rugged physical cannon-control console with guarded cyan controls, toggles, cables, and a prominent mechanical firing lever.\n<Picture 9> defines <Cockpit 1>, the saucer cockpit beneath the transparent dome. Preserve its curved graphite ribs, alien pilot seat, sculpted controls, cyan seams, and canopy geometry. It contains only <Subject 2>.\n\nsummary:\n[reference generation] <Vessel 1> and <Craft 1> are both falling out of control toward the Martian surface. As the ground rushes up, a terrified <Subject 1> pulls the firing lever on <Console 1>. <Weapon 1> fires at point-blank range, knocks <Craft 1> away from the human vessel, and sends the saucer crashing into Mars while <Vessel 1> barely pulls out of its dive.\n\nretention_analysis:\nPicture 1 supplies only the rapidly approaching Martian surface. Picture 2 supplies only the human identity. Picture 3 supplies only the pressure suit. Picture 4 supplies only the rail cannon. Picture 5 supplies only the saucer exterior. Picture 6 supplies only the alien identity. Picture 7 supplies only the falling human spacecraft and weapons-deck layout. Picture 8 supplies only the firing console. Picture 9 supplies only the alien cockpit. Keep one human, one alien, one human spacecraft, one cannon, one console, and one saucer. Do not add other spacecraft, weapons, people, aliens, text, logos, or Earth scenery.\n\ndetailed_description:\nPhotorealistic live-action science-fiction disaster cinema, cinematic 16:9, one continuous ten-second shot with violent but physically coherent motion. The camera is anchored low on <Vessel 1>'s shaking armored deck. <Weapon 1> dominates frame left; <Console 1> and <Subject 1> in <Suit 1> occupy frame right. The vessel is pitched steeply downward. Mars expands rapidly across the entire lower background, while red-orange atmospheric plasma and loose fragments streak upward past the deck. <Craft 1> falls beside the human vessel at dangerously close range, scraping through the plasma. Through its transparent dome, <Subject 2> is clearly visible fighting the controls inside <Cockpit 1>.\n\n[00:00.000-00:01.400] Both spacecraft tumble toward Mars. The horizon rotates, the deck bucks violently, warning lights pulse, and the Martian ridges grow larger every instant. <Craft 1> slides closer until its rim nearly clips the human deck. <Subject 1> looks from the approaching ground to the saucer with visible panic, braces both boots, and reaches desperately for the firing lever.\n\n[00:01.400-00:03.500] <Subject 1> grips the lever with both gloved hands and yanks it fully downward. Cyan electromagnetic energy races along <Weapon 1> in three rapid stages. The cannon rotates a few degrees toward the nearby saucer and fires once with a brutal recoil that shakes the whole deck. A concentrated blue-white projectile crosses the narrow gap.\n\n[00:03.500-00:05.700] The projectile tears through the outer rim of <Craft 1>, away from the transparent dome. A violent burst of sparks, molten fragments, and cyan electrical arcs erupts between the two falling spacecraft. The saucer spins away from <Vessel 1>; <Subject 2> is thrown against the cockpit controls as the intact dome flashes past the camera. The human cannon settles crookedly from recoil while <Subject 1> remains locked on the lever.\n\n[00:05.700-00:10.000] <Craft 1> tumbles below the human vessel, strikes the Martian surface, and produces a huge expanding fireball and towering rust-colored dust plume. The blast illuminates the underside of <Vessel 1>. At the final possible moment, the human spacecraft's engines flare bright cyan and pull the deck sharply above a rocky ridge. The camera shudders as the vessel skims past the terrain, ending on <Subject 1>'s terrified face, the recoiling cannon, and the enormous impact cloud behind them. No cuts, duplicate subjects, planetary destruction, gore, readable text, or unexplained object transformation.\n\noverall_soundscape:\nViolent hull vibration, rushing upper-atmosphere wind, suit breathing turning into a gasp, warning alarms, a heavy lever slam, escalating electromagnetic whine, one concussive rail-cannon crack, tearing metal, alien engine failure, a delayed planetary impact, and a final engine roar. No dialogue.\n\nnon_diegetic_music:\nRapid low percussion and rising brass pressure that cuts out at the cannon shot, followed by one massive bass impact and a sustained note during the last-second pull-up.\n",
      "references": [
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_7_1.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_7_2.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_7_3.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_7_4.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_7_5.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_7_6.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_7_7.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_7_8.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_7_9.jpg"
        }
      ],
      "duration": 10.0,
      "megapixels": 0.5,
      "aspect_ratio": "16:9"
    },
    {
      "task": "ref2va",
      "references": [
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_8_1.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_8_2.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_8_3.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_8_4.jpg"
        },
        {
          "type": "image",
          "path": "ref2va_testset/ref2va_test_8_5.jpg"
        }
      ],
      "prompt": "subject_definitions:\n<Picture 1> defines <Environment 1>, the gamer and computer room. Preserve the seated operator on frame right, desk, monitor, keyboard, dim interior, curtains, and solid exterior wall. Replace all existing monitor content; do not preserve game graphics, text, logos, or interface elements.\n<Picture 2> defines <Object 1>, the sole blue sport motorcycle. Preserve its blue-and-black bodywork, aggressive aerodynamic silhouette, wheels, headlight geometry, suspension, and mechanical proportions. Remove all manufacturer marks, model text, racing decals, and license plates. The motorcycle remains riderless.\n<Picture 3> defines <Motion 1>, the away-from-camera acceleration, low rear viewpoint, strong centered vanishing point, and tree-lined paved-road destination. Preserve only the motion direction, perspective depth, and leafy corridor. Ignore the source cyclist, bicycle, backpack, bicycle-lane markings, traffic signs, distant vehicles, and readable text. In the target video, widen the path into a normal road suitable for <Object 1>.\n<Picture 4> defines <Environment 2>, the cyan-and-magenta virtual light tunnel visible only inside the monitor.\n<Picture 5> defines <Effect 1>, the rectangular particle-grid portal and luminous digital fragments used at the monitor boundary.\n\nsummary:\n[reference generation] <Object 1> races out of <Environment 2>, punches through <Effect 1> at the monitor, becomes full size inside <Environment 1>, then accelerates directly through the room's exterior wall. Bricks, concrete, plaster, and dust explode outward as the riderless motorcycle bursts onto the road and speeds away using <Motion 1>.\n\nretention_analysis:\nPicture 1 supplies only the operator, room, desk, monitor, and breakable exterior wall. Picture 2 supplies the only motorcycle identity. Picture 3 supplies only the final rear-view motion, vanishing-point geometry, and tree-lined road direction. Picture 4 supplies only the virtual tunnel. Picture 5 supplies only the digital portal. Keep one operator and one riderless motorcycle. Do not copy the original game screen, cyclist, bicycle, backpack, bicycle-lane markings, traffic signs, distant vehicles, readable text, logos, license plates, or additional vehicles. The wall breaks only when struck by <Object 1>; the operator and desk remain safely to the side.\n\ndetailed_description:\nPhotorealistic live-action science-fiction action commercial, cinematic 16:9, one continuous ten-second shot with escalating scale and strong physical impact. Start from a low three-quarter view that keeps the monitor and desk on frame left, the startled operator on frame right, and a solid exterior wall visible beyond the motorcycle's escape line.\n\n[00:00.000-00:01.200] The monitor abruptly fills with <Environment 2>. Cyan-magenta light floods the dark room as a tiny <Object 1> accelerates from deep inside the tunnel directly toward the viewer. <Effect 1> expands beyond the monitor frame, bending loose papers and curtain fabric toward the screen. The operator turns in alarm and pushes back from the desk.\n\n[00:01.200-00:02.600] The front tire of <Object 1> punches through the monitor plane first. The screen surface bursts into thousands of luminous square particles rather than glass. The same motorcycle scales smoothly to full physical size in midair and lands hard on the floor. Its suspension compresses, both tires bite, sparks rip backward beneath the chassis, and the headlight flashes across the operator's frightened face. The motorcycle misses the operator and desk by a narrow but clear margin.\n\n[00:02.600-00:05.100] Without slowing, <Object 1> aims directly at the solid exterior wall. The front wheel and fairing strike it with enormous force. A circular fracture races across the plaster and masonry before the whole wall erupts outward. Large bricks, concrete chunks, splintered framing, plaster fragments, insulation, and a dense pressure wave blast into the open air. Dust rolls back through the room while smaller fragments spin past the camera. The motorcycle remains coherent and punches completely through the new opening.\n\n[00:05.100-00:10.000] The camera whip-pans through the collapsing hole and drops into the low rear chase composition from <Motion 1>. <Object 1> lands on the outdoor road amid bouncing masonry, fishtails once, regains traction, and accelerates rapidly toward the vanishing point. Behind it, the broken wall continues collapsing in a huge dust cloud while the cyan portal flickers inside the exposed room. No cuts, rider, duplicated motorcycles, collision with the operator, rain, fireball, readable text, brand marks, or impossible wheel deformation.\n\noverall_soundscape:\nComputer fan hum, rising digital tunnel roar, particle crackle, a heavy tire landing, motor surge, floor sparks, one massive wall impact, exploding masonry, falling debris, dust pressure, tire squeal, and accelerating road sound. No speech.\n\nnon_diegetic_music:\nA fast electronic build that peaks at the wall impact, followed by heavy percussion and a final accelerating synth pulse.\n",
      "duration": 10.0,
      "aspect_ratio": "16:9",
      "megapixels": 0.5
    }
  ]
}

```


---

# ISSUES / PRs of ModelTC/Minimax-H3-Turbo


### #15 [closed] ISSUE: 这个r2v lora蒸馏模型缺少video shift信息
URL: https://github.com/ModelTC/Minimax-H3-Turbo/issues/15  by splinter21 2026-09-13T16:47:56Z

https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors
推理用12还是6？

> **Comment by qinweicheng33-create (2026-09-16T01:58:01Z):**

这个是视频6，音频3；



### #14 [closed] PR: Update README.md
URL: https://github.com/ModelTC/Minimax-H3-Turbo/pull/14  by gushiqiao 2026-08-27T09:22:27Z




### #13 [closed] ISSUE: I2VA ERROR
URL: https://github.com/ModelTC/Minimax-H3-Turbo/issues/13  by eeewhe 2026-08-20T14:44:05Z

The i2va won't run, and the H200 is experiencing memory OOM (Out of Memory) errors. Why?


### #12 [open] ISSUE: _resolve_lora_path() strips .safetensors suffix via .resolve() on default HF cache symlinks, crashes into torch.load() mmap path
URL: https://github.com/ModelTC/Minimax-H3-Turbo/issues/12  by keeganmccallum 2026-08-18T06:29:16Z

## `_resolve_lora_path()`'s `.resolve()` strips `.safetensors` via the default HF Hub cache's symlink layout, misrouting into the `torch.load()` fallback and crashing

`inference_minimax_h3.py`'s `_resolve_lora_path()` (and the same helper as
vendored standalone, e.g. by downstream consumers who extract just the LoRA-
loading slice) does:

```python
def _resolve_lora_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {path}")
    return path
```

`Path.resolve()` follows symlinks. The **default, standard** `huggingface_hub`
cache (i.e. whatever `hf_hub_download(repo_id, filename)` returns with no
`local_dir=` override -- the normal way most callers will obtain the LoRA
checkpoint path) is laid out as:

```
~/.cache/huggingface/hub/models--lightx2v--Minimax-h3-Turbo/
  snapshots/<rev>/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors  # symlink
  blobs/<sha256>                                                     # real file, NO extension
```

`hf_hub_download()` returns the `snapshots/.../*.safetensors` symlink path.
Calling `.resolve()` on it follows the symlink through to
`blobs/<sha256>` -- which has **no extension**. The returned path's
`.suffix` is then `""`, not `".safetensors"`.

`_load_lora_state_dict()` branches on exactly that suffix:

```python
def _load_lora_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        checkpoint = load_safetensors_file(path, device="cpu")
    else:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
```

With the suffix stripped, it silently falls into the `torch.load(..., mmap=True)`
branch on a file that IS valid safetensors (not a torch-pickled checkpoint),
and crashes:

```
RuntimeError: mmap can only be used with files saved with
`torch.save(_use_new_zipfile_serialization=True)`, please torch.save your
checkpoint with this option in order to use mmap.
```

This is 100% reproducible any time the LoRA path comes from
`hf_hub_download()` (or any other call that returns the default HF cache's
symlink layout) rather than a manually-placed, non-symlinked local file --
i.e. the most common way a caller would obtain this checkpoint.

### Fix

Drop `.resolve()`; keep `.expanduser()`. `hf_hub_download()` paths are
already absolute, so `.resolve()`'s only remaining effect here is following
the symlink and destroying the suffix. `Path.is_file()` and any subsequent
file read (`safetensors.torch.load_file`, `open()`, `torch.load()`) already
transparently follow symlinks at the OS level, so nothing is lost by not
calling `.resolve()`:

```diff
 def _resolve_lora_path(path: Path) -> Path:
-    path = path.expanduser().resolve()
+    path = path.expanduser()
     if not path.is_file():
         raise FileNotFoundError(f"LoRA checkpoint does not exist: {path}")
     return path
```

### Repro

```python
from pathlib import Path
from huggingface_hub import hf_hub_download

p = Path(hf_hub_download("lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"))
print(p.suffix)              # '.safetensors'
print(p.resolve().suffix)    # ''  <-- the bug
```

Hit live while running the 8-step Turbo LoRA against
`MiniMaxAI/MiniMax-H3` via the documented `add_adapter`/PEFT mechanism,
2026-08-18.



### #11 [open] ISSUE: High resolution support?
URL: https://github.com/ModelTC/Minimax-H3-Turbo/issues/11  by SamTyurenkov 2026-08-17T18:48:39Z

My hardware allows me to generate 2048x1024 videos. 

Do you think turbo loras should be compatible? So far I couldn't setup them to work. 
I also tried 1344x768 but results also looked distorted. either video goes into distortion or there is no motion.

> **Comment by FlowForgeLabAi (2026-10-08T02:00:01Z):**

Worth separating two things here: "the turbo LoRA can't do high resolution" vs
"nothing in H3 can".

**1. The released turbo LoRAs are 768p.** The file itself is named
`minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`, and #3 is literally
"Add 768p MiniMax-H3 LoRA inference docs". So 2048x1024 is far outside what these
specific weights were distilled for — even if the base model could go there, the
LoRA is a separate object with its own distribution.

**2. The base model's own canvas is capped at ~1.03 MP**, not by opinion but by the
documented native range: short side <= 768, area <= 768x1344, both dimensions
multiples of 32. That gives:

| target | area | vs 768x1344 cap |
|---|---|---|
| 1344x768 | 1,032,192 | exactly at the cap |
| 2048x1024 | 2,097,152 | **~2.03x over** |

So 2048x1024 is over the base envelope *and* ~2.6x the LoRA's training resolution.
1344x768 sits exactly on the cap but is a different aspect than 768p — it is the
more interesting of your two cases, because if that one still distorts, the LoRA's
aspect/size conditioning is the suspect rather than the base model's canvas.

**3. "Either distortion or no motion" is the classic out-of-envelope signature** on H3
in my experience — it is not a sampling or CFG problem, so no amount of step/CFG
tuning fixes it.

**Suggested isolation, cheapest first:**
1. Run the same workflow/seed at **768x1024** with the turbo LoRA. If that is clean,
   the LoRA is fine and you are looking at a size/aspect sensitivity.
2. Then **768x1344** (same area class as your 1344x768, but the short side the weights
   were distilled for). If this is clean and 1344x768 is not, it is aspect, not area.
3. Only then go to 2048x1024.

**If you are driving this through ComfyUI:** check what the `ResolutionSelector`
"megapixels" dial actually resolves to before blaming the LoRA. It rounds to a
32-multiple *after* scaling to the target MP, so 3:4 at 1.0 MP silently becomes
**896x1184** — short side 896, over the 768 cap. Setting the dial to **0.75** gives
768x1024 instead. Worth confirming your resolution before changing anything else.




### #10 [open] ISSUE: metallic audio?
URL: https://github.com/ModelTC/Minimax-H3-Turbo/issues/10  by FezVrasta 2026-08-16T09:52:04Z

Hi, I always get slightly metallic audio, do you have any suggestion?

> **Comment by chongsoo67-blip (2026-08-16T15:19:02Z):**

I second this. Extensively been playing around with this Lora for the past 24 hours and the result is promising.

Although there are two issues that are deal breakers in my opinion. Above 0.5 megapixels the visuals fall apart and the voices sound metallic no matter what you do on reference models. 

Edit: using the ref2v Lora.


> **Comment by CS6700 (2026-08-16T23:13:59Z):**

Just a heads up. Had those metallic voices too and couldn't get rid of them.
Finally found a workaround by using the checkpoint minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors + turbo lora.

Make sure you load the Lora in a standard comfyui Lora loader and NOT a custom one, it's important.
Hope this helps.


> **Comment by FezVrasta (2026-08-17T11:59:09Z):**

the hybrid model does not support real fl2va thought right? The node does not allow it.



### #9 [closed] PR: feat(examples): ref2va inference test cases
URL: https://github.com/ModelTC/Minimax-H3-Turbo/pull/9  by HamQiuZ 2026-08-14T13:35:14Z




### #8 [closed] PR: feat: ref2va
URL: https://github.com/ModelTC/Minimax-H3-Turbo/pull/8  by X-niper 2026-08-13T15:31:56Z

the test files are not uploaded yet


### #7 [open] ISSUE: Pruned compact-curve models: ComfyUI conversions must drop all AdaLN adapters — request curve-compatible turbo
URL: https://github.com/ModelTC/Minimax-H3-Turbo/issues/7  by AlexanderIstomin 2026-08-13T14:33:09Z

First: thank you for the turbo distillation — merged into the pruned INT8 transformer it produces the best visual fidelity we have seen from local H3 at 6–8 steps.

## The problem

The Comfy-Org pruned INT8 ConvRot package (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`) uses the compact time-curve AdaLN parameterization: per-block AdaLN input dimension **8** (a precomputed `adaln_t_table` plus small per-block linears) instead of the full model's **2688**.

Because of that, the community ComfyUI conversions of the turbo LoRA for pruned models (e.g. `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`) have to **drop all 51 AdaLN adapter pairs** — the file's own metadata says:

> "All AdaLN LoRA adapters were removed because the source targets input dimension 2688 while the pruned model targets dimension 8. Four-step distillation behaviour may be degraded or broken."

## What we observe with the AdaLN pairs missing

Merging the remaining 208 pairs (attention + MLP, rank 64) at strength 1.0 into the pruned INT8 transformer and sampling at 6–8 steps (euler, beta sigmas):

- **Fidelity is excellent** — sharp, photoreal detail well beyond what the base pruned model reaches at comparable step counts.
- **Prompt adherence degrades in a characteristic way**: scene/verb structure survives while subjects drift toward dataset priors. Concrete failures at seed 42, 256²–512², 22 frames: a "red fox in snow" prompt renders a photoreal **dog** close-up; "a surfer riding a wave" renders a **chat-app screenshot with garbled text**; subtitle/watermark-like glyphs appear frequently. Frames and exact repro commands available on request.

This matches what missing timestep conditioning would do to a step-distilled model, exactly as the conversion metadata warns.

## Request

Would you consider either:

1. publishing turbo AdaLN adapters (or a full turbo variant) **compatible with the compact-curve pruned models** — e.g. distilled against the pruned parameterization, or with a documented mapping from the 2688-dim adapters onto the 8-dim curve inputs so converters can stop dropping them; or
2. documenting the intended path for pruned-model users (is the 2688→table recomputation feasible from the released adapters, i.e. re-evaluating the LoRA-adapted time-embedding stack over the t-grid to rebuild `adaln_t_table`)?

The pruned INT8 package is the main low-memory way to run H3 locally (Apple-silicon consumers included), so a curve-compatible turbo would land the distillation exactly where memory-constrained users need it most.


> **Comment by AlexanderIstomin (2026-08-13T20:12:23Z):**

Correction to my report above: the observed-degradation section is not
supported by my own later testing and I withdraw it.

When I re-ran every failing prompt against the **unmodified base package
at identical settings, seed, and canvas**, the base model produced the
same failures — the fox prompt rendered a dog at 512²/8 steps with and
without the adapter, and the surfer prompt produced the same text-card
artifact at 256²/8 steps with no adapter loaded. My original comparisons
also varied the sigma schedule at the same time, so they did not isolate
the adapter at all. The failures appear to be base-model behaviour at low
step counts on the pruned INT8 package, not a consequence of the merge.

The structural question stands and is what I should have asked in the
first place: the pruned conversions must drop all 51 AdaLN pairs because
of the 2688 vs 8 dimension mismatch, so compact-curve users are running a
partial adapter. Is a curve-compatible variant feasible, or is there a
documented mapping for rebuilding `adaln_t_table` from the released
adapters? Apologies for the noise in the original evidence.




### #6 [closed] PR: feat: add i2v testcases
URL: https://github.com/ModelTC/Minimax-H3-Turbo/pull/6  by HamQiuZ 2026-08-12T03:02:15Z




### #5 [closed] PR: Add ComfyUI example workflows
URL: https://github.com/ModelTC/Minimax-H3-Turbo/pull/5  by GACLove 2026-08-11T16:41:44Z

## Summary
- add ComfyUI T2V / I2V example workflows
- document usage in README


### #4 [closed] PR: feat(examples): add i2v testcases
URL: https://github.com/ModelTC/Minimax-H3-Turbo/pull/4  by HamQiuZ 2026-08-11T16:16:37Z




### #3 [closed] PR: Add 768p MiniMax-H3 LoRA inference docs
URL: https://github.com/ModelTC/Minimax-H3-Turbo/pull/3  by gushiqiao 2026-08-11T14:16:39Z




### #2 [closed] PR: feat: remove preview
URL: https://github.com/ModelTC/Minimax-H3-Turbo/pull/2  by X-niper 2026-08-11T13:36:39Z




### #1 [closed] PR: feat: 8step v1.0, multi-gpu inference
URL: https://github.com/ModelTC/Minimax-H3-Turbo/pull/1  by X-niper 2026-08-11T13:33:16Z


