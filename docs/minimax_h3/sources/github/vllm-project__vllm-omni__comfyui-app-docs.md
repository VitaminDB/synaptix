# vllm-project__vllm-omni__comfyui-app-docs

Source: https://github.com/vllm-project/vllm-omni/tree/main/apps/ComfyUI-vLLM-Omni/docs
Fetched: 2026-11-28



---

## FILE: apps/ComfyUI-vLLM-Omni/docs/minimax-h3-t2v.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/apps/ComfyUI-vLLM-Omni/docs/minimax-h3-t2v.md)

# MiniMax H3 Text to Video (WF-01)

Import [MiniMax_H3_Text_to_Video.json](../example_workflows/MiniMax_H3_Text_to_Video.json)
into ComfyUI, or select it under **Templates → ComfyUI-vLLM-Omni**.
The template connects existing nodes to a remote H3 service and saves the
returned video with audio. ComfyUI does not load H3 weights.

## Setup

Install the [ComfyUI extension](../README.md#installation) and follow the
[H3 deployment recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for your hardware.
For a two-GPU FL2VA server with sufficient memory:

```bash
export MODEL=/path/to/MiniMax-H3/FL2VA
CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve "$MODEL" --omni --host 127.0.0.1 --port 8000 \
  --served-model-name MiniMaxAI/MiniMax-H3 --trust-remote-code \
  --task-type fl2va --num-gpus 2 --tensor-parallel-size 2 \
  --text-encoder-tp-size 2 --enforce-eager
```

Set Generate Video's **url** to the address reachable from ComfyUI, including
`/v1`, and match its **model** to the served name. The defaults are
`http://localhost:8000/v1` and `MiniMaxAI/MiniMax-H3`.

## Base and Turbo

Base presets are connected by default. Write the scene and its sound in the
same **prompt**; leave **frame**, **references**, and **fast_h3** disconnected.
The template uses 1344×768, 24 FPS, seed 1101 and **duration** 5.167 seconds
(124 frames, satisfying H3's `17k+5` constraint). The existing client infers
`16:9` from these dimensions. For portrait output, use 768×1344.

| Setting | Base | Turbo v1.0 768p |
| --- | --- | --- |
| Inference steps | 50 | 5 (4 forwards) |
| Video flow shift | 12 | 6 |
| Audio flow shift | 3 | 3 |
| Remote LoRA | Disconnected | Connected, scale 1 |

For Turbo, use the **Diffusers-layout** artifact from
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo):

```text
minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
```

Add `--lora-backend peft --lora-path "$TURBO_LORA"` to the server command,
where `TURBO_LORA` is the full server-side path to that file. Enter the same
path in Remote LoRA's **local_path**; retain name `h3-turbo-v1.0-768p`, scale
`1.0`, and **int_id** `0` so the existing API derives the ID from the path.
The template leaves the path blank for users to configure. The `_comfyui_`
export is not supported by this backend.

Connect **Turbo sampling**, **Turbo H3 params**, and **Remote LoRA** to Generate
Video's `sampling_params`, `model_params`, and `lora` inputs, replacing the
Base links. To return to Base, restore both Base links and disconnect LoRA.
Other Turbo artifacts may need different settings; see the
[artifact table](../../../recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora).

If importing an older template with a frame-count widget, reimport this JSON
or set **duration** to **5.167**; do not interpret 124 frames as 124 seconds.

## Validation

Queue the configured workflow and inspect the MP4 saved under
`ComfyUI/output/video/MiniMax_H3_WF01_*.mp4`:

```bash
ffprobe -v error -show_streams -show_format -of json "$OUTPUT"
ffmpeg -v error -i "$OUTPUT" -f null -
```

Check the requested dimensions, frame count, 24 FPS and both video/audio
streams, then watch and listen to the saved file. The current upstream
extension includes the shared audio-preserving decoder fix.

On September 16, the narrowed workflow was validated through ComfyUI using
existing upstream Python code at `4a8ac297`, two H20-3e GPUs, vLLM 0.29.0,
PyTorch 2.13.0+cu129 and ComfyUI `1d48d9cf`. The run used the template's
forest-stream prompt, explicit server-side Turbo LoRA path, and settings
above, with `TORCH_SDPA`, `--usp 1 --ring 1`, and tiled VAE parallelism on two
GPUs. ComfyUI execution took 131.55 seconds. SaveVideo produced 124 H.264
frames at 1344×768 and 24 FPS, plus 32 kHz stereo AAC audio. Video/audio
durations were 5.166667/5.167000 seconds; full decoding passed and audio was
nonzero. This is model-generated output, not a ground-truth reference or a
subjective audio-quality assessment. Base graph validation passed with Turbo
nodes disconnected; full 50-step Base quality was not evaluated.

Based on the official
[ComfyUI H3 T2V template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json),
using remote inference instead of local model loaders.


---

## FILE: apps/ComfyUI-vLLM-Omni/docs/h3-03-music-video.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/apps/ComfyUI-vLLM-Omni/docs/h3-03-music-video.md)

# H3-03: MiniMax-H3 music video from a supplied track

Turn a music track you supply into a multi-shot music video. Each shot is a
MiniMax-H3 Ref2VA request that uses a shared character or style image and its
own slice of the track as **driving audio**, so the visuals follow that part of
the music. ComfyUI then joins the shots and puts the original recording on the
final MP4. All inference runs on a remote vLLM-Omni service; ComfyUI does not
load H3 weights.

This is the supplied-audio branch of H3-03. The YuE2-generated-music branch
depends on the YuE2 serving API and ComfyUI music controls (CUI-02).

## Requirements

- ComfyUI **v0.36.0 or later**, for the core **Concatenate Video** node and its
  `complete_audio` input. The graph also uses the core Load Audio, Trim Audio
  Duration, Primitive, and Save Video nodes.
- ComfyUI-vLLM-Omni with the `audio_mode` option on **MiniMax-H3 Video Params**.
- A Ref2VA-capable H3 service: the combined service (repository ID, no
  `--task-type`) or one started with `--task-type ref2va`. Follow the
  [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for your hardware.

## How it works

| Stage | Nodes | What happens |
| --- | --- | --- |
| Inputs | Load Audio, Load Image | The supplied track (WAV or MP3) and one character/style reference |
| Shot slices | Trim Audio Duration (one per shot) | Cuts consecutive 124-frame (5.1667 s) slices of the track |
| Shots | Video References → Generate Video (one per shot) | Ref2VA with `image_1` = the shared reference and `audio_1` = the shot's slice |
| Sync | MiniMax-H3 Video Params, `audio_mode=lock_source` | H3 uses the slice as the shot's soundtrack and keeps it fixed while sampling |
| Assembly | Trim Audio Duration, Concatenate Video, Save Video | Joins the shots in order and replaces their audio with the original track |

With `audio_mode=lock_source`, the server places the single driving audio in
the target audio latent, crops or zero-pads it to the shot duration, and keeps
it clean throughout sampling. The driving audio is not a `<Audio 1>` reference
tag; it is the shot's soundtrack. The node sends `audio_mode` only for
`lock_source`, so requests from other workflows are unchanged. The server
rejects `lock_source` unless exactly one audio is connected to Video
References.

## Use

Open **vLLM-Omni MiniMax-H3 Music Video.json** from **Templates →
ComfyUI-vLLM-Omni**, or drag the JSON onto the canvas.

1. Set **Server URL** and **Served model name** for your H3 service. Defaults:
   `http://localhost:8000/v1` and `MiniMaxAI/MiniMax-H3`.
2. Select your track in **Supplied music track** and your subject in
   **Character / style reference**. The template's filenames are placeholders
   relative to ComfyUI's input directory; no sample assets are bundled.
3. Write one prompt per shot. Name the subject `<Picture 1>` and describe the
   action, camera, and how the shot should follow the music, such as
   lip-syncing for vocals or movement on the beat.
4. Run the workflow. Save Video writes `output/video/MiniMax-H3-MV_*.mp4`.

### Shot timing

Each shot is 124 frames at 24 FPS (H3 requires `17k+5` frames), which is
124/24 = 5.1667 seconds. The shared **Shot duration** primitive feeds every
Generate Video and every slice length. Slice starts are set on each shot's
Trim Audio Duration node:

| Shot | Slice start (s) | Video time in the final MP4 (s) |
| --- | --- | --- |
| 1 | 0.00 | 0.00 – 5.17 |
| 2 | 5.17 | 5.17 – 10.33 |
| 3 | 10.33 | 10.33 – 15.50 |

The final soundtrack is trimmed to 3 × 5.1667 = 15.5 seconds from the same
start. Trim widgets step by 0.01 s, so a slice start is at most 0.005 s from
its exact boundary, well under one frame (0.042 s).

To start later in the song, add the same offset to every slice start and to
the full-soundtrack start. To add shots, duplicate a shot group, set its slice
start to the next boundary (`start + i × 5.1667`), connect its Generate Video to
the next free Concatenate Video input, and extend the full-soundtrack duration
by 5.1667 s per shot. To change the shot length, keep it on the `17k+5` frame
grid (for example 107, 124, 209 frames) and recompute every start and the
total duration.

### Replace or keep the shot audio

- **Replace (default):** Concatenate Video's `complete_audio` puts the
  original recording on the final video, so the output carries the exact
  source audio and has no seams between shots.
- **Keep shot audio:** disconnect `complete_audio`. Each shot then keeps the
  audio H3 returned: the audio VAE's reconstruction of its slice, not a
  byte-for-byte copy, which can differ slightly at shot boundaries.

`lock_source` keeps the driving audio fixed, so H3 does not add sound effects
or vocals on top of the track. To mix another recording with the music, combine
it with the full soundtrack using the core **Audio Merge** node before
`complete_audio`.

## Limits

- Each shot is a separate request. Identity comes from the shared reference
  image and the prompt; there is no latent continuity between shots, so
  expect cuts rather than continuous camera motion.
- A shot is 4–15 seconds (Ref2VA). Longer shots can use the server's
  `long_video` mode, which this workflow does not expose.
- The workflow has no prompt-assistance stage; H3 does not supply a prompt
  LLM.
- The graph's reference topology is not yet audited against the linked
  community example, so this template does not claim parity with it.

## Validation

Check the template's wiring and the client changes locally:

```bash
python -m pytest tests/e2e/features/comfyui/test_h3_music_video_workflow.py \
  tests/e2e/features/comfyui/test_video_references.py -q
```

For a real-model check, run the workflow against a served H3 model and inspect
the saved file:

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,r_frame_rate,nb_frames,sample_rate,channels \
  -show_entries format=duration -of json "$OUTPUT_MP4"
```

Expect 372 frames (3 × 124) of 24 FPS H.264 video and an audio stream whose
duration matches the video (15.5 s). Watch and listen for lip-sync or beat
alignment within each shot, subject consistency across shots, and no audio
gaps at shot boundaries. Record the server commit, model checkpoint, ComfyUI
version, graph revision, seed, inputs, runtime, and output.


---

## FILE: apps/ComfyUI-vLLM-Omni/docs/h3-06-all-in-one.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/apps/ComfyUI-vLLM-Omni/docs/h3-06-all-in-one.md)

# H3-06: MiniMax-H3 all-in-one workflow

One entry graph for the three MiniMax-H3 generation modes: text-to-video+audio
(T2VA), first/last-frame (FL2VA), and mixed-reference (Ref2VA). It uses only
the existing **Generate Video**, **Video References**, sampling, and H3 params
nodes plus core ComfyUI nodes; no extra custom nodes are required. ComfyUI does
not load H3 weights. Every mode runs on a remote vLLM-Omni service.

## Requirements

- ComfyUI with ComfyUI-vLLM-Omni installed. The graph uses the core
  `PrimitiveString` and `PrimitiveFloat` nodes for shared settings.
- An H3 service that serves the modes you plan to run. Follow the
  [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for your hardware.
  - A **combined** service (repository ID, no `--task-type`) serves all three
    modes without a restart: `t2va` and `fl2va` use the FL2VA DiT, and
    `ref2va` uses the Ref2VA DiT. See the recipe's
    [four-GPU combined service](../../../recipes/MiniMaxAI/MiniMax-H3.md#four-gpus-throughput-oriented-combined-service)
    for a starting point. It needs both checkpoint partitions.
  - A service started with `--task-type fl2va` serves T2VA and FL2VA only, and
    one started with `--task-type ref2va` serves Ref2VA only.

## Graph layout

| Group | Contents |
| --- | --- |
| Shared settings - all modes | Server URL, served model name, duration, Base sampling (50 steps, seed 42), H3 flow shifts (video 12, audio 3), usage note |
| Mode 1: T2VA | Generate Video with no image/reference input, Save Video |
| Mode 2: FL2VA | First frame and last frame Load Image nodes, Generate Video, Save Video |
| Mode 3: Ref2VA | Load Image, Video References, Generate Video, Save Video |

The shared nodes feed the `url`, `model`, `duration`, `sampling_params`, and
`model_params` inputs of all three Generate Video nodes, so you set them once.
Each mode keeps its own prompt because the modes need different wording:
Ref2VA prompts name their references, and FL2VA prompts describe motion
between keyframes. Generate Video selects the server task from which inputs
are connected:

| Mode | Connected inputs | Server task |
| --- | --- | --- |
| T2VA | none | `t2va` (aspect ratio derived from width/height) |
| FL2VA | `first_frame`, `last_frame`, or both | `fl2va` (`frame_indices` `[0]`, `[-1]`, or `[0, -1]`) |
| Ref2VA | `references` | `ref2va` |

## Use

Open **vLLM-Omni MiniMax-H3 All-in-One.json** from **Templates →
ComfyUI-vLLM-Omni**, or drag the JSON onto the canvas.

1. Set **Server URL** to the service's `/v1` address as seen from ComfyUI and
   **Served model name** to its `--served-model-name` (the repository ID when
   none is given). Defaults: `http://localhost:8000/v1` and
   `MiniMaxAI/MiniMax-H3`.
2. Choose a mode. Only the T2VA group is active when the template opens; the
   FL2VA and Ref2VA groups are muted. Right-click a group title and choose
   **Set Group Nodes to Always** to enable it, and **Set Group Nodes to Never**
   on the groups you are not using. Selecting nodes and pressing **Ctrl+M** also
   toggles muting. Several active groups run together and send one request per
   mode.
3. Fill in the active mode's inputs:
   - **T2VA**: write the scene and its sound in the prompt. Width and height
     select the nearest supported aspect ratio; 1344×768 maps to `16:9`, and
     768×1344 to `9:16`.
   - **FL2VA**: select the first and last frame images. Disconnect one loader
     for first-frame-only or last-frame-only generation. The output ratio
     follows the input image.
   - **Ref2VA**: select a reference image. Add more image, video, or audio
     loaders to **Video References**: up to 9 images, 3 videos, and 3 audio
     clips, 12 inputs in total, with at least one image or video. Name the
     references in the prompt as `<Picture 1>`, `<Video 1>`, and `<Audio 1>`,
     counting each media type in slot order.

   The template's image filenames are placeholders relative to ComfyUI's input
   directory. No sample assets or model weights are bundled.
4. Run the workflow. The active mode's Save Video node writes an MP4 with the
   generated audio under `output/video/MiniMax-H3-AIO-<mode>_*.mp4`.

The default duration is 5.167 seconds, which converts to 124 frames at 24 FPS
and satisfies H3's `17k+5` frame constraint. Keep durations within the 4–15
second output range; 107, 124, 209, and 345 frames are valid examples.

### Turbo

The graph is configured for Base H3. Turbo adapters are task-specific (the
FL2VA and Ref2VA artifacts differ) and must be preloaded by the server, so they
do not fit one shared request setting. To use Turbo for one mode, start a server
with that mode's adapter, add a **Remote LoRA** node to that mode's Generate
Video, and change the shared sampling and flow-shift values as described in the
[Turbo LoRA recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora). The
[text-to-video](minimax-h3-t2v.md) and
[reference-to-video](../README.md#minimax-h3-reference-to-video) guides give
per-artifact settings.

## Prompt assistance

This graph does not include a prompt-assistance stage. H3 itself does not
supply a prompt LLM, so each mode's prompt is used as written. Prompt
expansion will be added only after the source graph's assistant model and
optional dependencies are identified. It would then be a separate,
optional stage, for example a **Multimodality Understanding** node calling its
own vLLM-Omni service, rather than part of the H3 request.

## Validation

Check the template's wiring and defaults locally:

```bash
python -m pytest tests/e2e/features/comfyui/test_h3_all_in_one_workflow.py -q
```

For a real-model check, run each mode against a served H3 model and inspect
the saved file:

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,r_frame_rate,nb_frames,sample_rate,channels \
  -show_entries format=duration -of json "$OUTPUT_MP4"
```

Expect 24 FPS H.264 video, a stereo AAC audio stream, and aligned video/audio
durations. Watch and listen to each result for keyframe adherence (FL2VA),
reference conditioning (Ref2VA), and audio/video synchronization. Record the
server commit, model checkpoint, ComfyUI version, graph revision, seed, inputs,
runtime, and output. Local schema checks do not establish generation quality.


---

## FILE: apps/ComfyUI-vLLM-Omni/docs/wf05-h3-latent-editing.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/apps/ComfyUI-vLLM-Omni/docs/wf05-h3-latent-editing.md)

# WF-05: MiniMax-H3 latent-mask editing

Edit a source video with MiniMax-H3 using spatial or temporal masks. The workflow includes object removal, inpainting, continuation, and extension, with a mask preview for each case.

## Requirements

- ComfyUI with ComfyUI-vLLM-Omni installed.
- An H3 service using the FL2VA checkpoint partition with latent-mask editing support. Follow the [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for server setup.

## Use

Open **vLLM-Omni MiniMax-H3 Latent Mask Editing.json** from the example workflows and set the Generate Video URL/model for the case you want to run.

1. Upload your source video in Load Video.
2. Adjust the case's mask. For removal and inpainting, set Mask Canvas to the video dimensions, Mask Rectangle to the region size, and Combine Masks x/y to its position, or replace that chain with Load Image (as Mask) for an arbitrary-shape mask. For continuation and extension, set MiniMax-H3 Temporal Mask's mode and duration; continuation also uses `preserve_fraction`. Match the duration in Generate Video, and use a duration longer than the source for extension. Mask values are `0` to preserve and `1` to regenerate.
3. Adjust the prompt to describe the desired content. Set the output dimensions and duration in Generate Video. If editing audio, update the sound description and audio mask (`0` preserves, `1` regenerates the whole clip).
4. Select the case's Mask Preview Save Video node and choose **Execute to selected output nodes** to preview the mask as a red overlay without inference, or select its Result Save Video node to generate the edited video. Running the entire workflow generates all four cases.


---

## FILE: apps/ComfyUI-vLLM-Omni/docs/wf07-h3-upscale.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/apps/ComfyUI-vLLM-Omni/docs/wf07-h3-upscale.md)

# WF-07: H3 video upscale with SeedVR2

Generate video and audio through vLLM-Omni H3, upscale the decoded frames with SeedVR2, and save both the original and upscaled videos. The generated audio and FPS connect directly to Create Video.

## Requirements

- ComfyUI with ComfyUI-vLLM-Omni and [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler#-installation) installed.
- An H3 service using the FL2VA checkpoint partition. Follow the [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for server setup.
- `seedvr2_ema_3b_fp16.safetensors` and `ema_vae_fp16.safetensors` from [SeedVR2's ComfyUI weights](https://huggingface.co/numz/SeedVR2_comfyUI/tree/main), placed in `ComfyUI/models/SEEDVR2`. The SeedVR2 nodes can also download them on first use.

## Use

Open **vLLM-Omni MiniMax H3 Video Upscale.json** from the example workflows. An API-format version is provided alongside it. Set the Generate Video URL/model and the LoRA path as seen by the H3 server.

The preset uses the LightX2V v1.0 768p Turbo adapter in Diffusers format: `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`, five sigma points, video/audio flow shifts 6/3, and LoRA scale 1. Preload the same adapter on the server using the [Turbo LoRA recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora). For base H3, disconnect LoRA, set inference steps to 50, and set video flow shift to 12.

The H3 preset requests 1344 x 768 at 24 FPS for 5.167 seconds, which gives 124 frames. SeedVR2's `resolution=1536` sets the target short edge, producing 2688 x 1536 for this input. Update this value when changing the input size; it is an output resolution, not a fixed 2x multiplier.

The SeedVR2 preset uses 3B FP16, five-frame batches, one overlapping frame, uniform batch padding, and LAB color correction. The DiT loader uses CPU offload with 32 swapped blocks; the VAE uses tiled encoding and decoding. Select the local GPU in both model loaders. See the [SeedVR2 settings](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler#-usage) when adjusting memory use or batch size.

Run the workflow. The two Save Video nodes write MP4s under `WF07/generated` and `WF07/upscaled` in ComfyUI's output directory.


---

## FILE: apps/ComfyUI-vLLM-Omni/README.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/apps/ComfyUI-vLLM-Omni/README.md)

# vLLM-Omni

vLLM-Omni offers a ComfyUI integration on top of its online serving API.
It can send model inference requests to either a locally running vLLM-Omni service or a remote one.

## Requirement

- Python 3.12 or above
- [ComfyUI installed](https://docs.comfy.org/installation/system_requirements)
- [vLLM-Omni installed](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/) on either the same device or another device discoverable via the internet.
- No need to install additional packages apart from those already required by ComfyUI.

> [!TIP]
> If you run both ComfyUI and vLLM-Omni on the same device, you can create separate virtual environments and use different Python versions for them.

## Installation

Copy this folder to the `custom_nodes` subfolder of your ComfyUI installation. Your directory should look like `ComfyUI/custom_nodes/ComfyUI-vLLM-Omni`.

If you are running ComfyUI during copying, you should restart ComfyUI to load this extension.

> [!TIP]
> You can use utility websites such as <https://download-directory.github.io/> to download a subdirectory of a repo. Also checkout community discussions (e.g., <https://stackoverflow.com/questions/7106012/download-a-single-folder-or-directory-from-a-github-repository>) for more info.

On the device and virtual environment you run ComfyUI, launch ComfyUI with

```bash
cd ComfyUI

# The regular way
python main.py

# If you are mainly using this node, launch it faster with
python main.py --cpu
```

On the device and virtual environment you run vLLM-Omni, start a model service with

```bash
vllm serve The_Model_ID_to_Serve --omni --port 8000
```

Check **ComfyUI's sidebar -> Node Library**. There should be a new folder named **vLLM-Omni**.
If no, check your shell running the ComfyUI process. There may be some error messages before the line `Import times for custom nodes:` and the line `To see the GUI go to: http://127.0.0.1:8188`.

## Quickstart

This extension offers the following nodes based on the output modalities (at **ComfyUI sidebar -> Node Library**):

- **Generate Image** for text-to-image and image-to-image tasks
- **Generate Video** for text-to-video, first-frame/image-to-video, and reference-conditioned video
- **Restore Video (SeedVR2)** for remote VIDEO restoration with source FPS and audio;
  see the [standalone and H3 chaining guide](docs/seedvr2-restoration.md).
- **Latent Mask Editing** for MiniMax-H3 latent-mask editing (source media plus video/audio noise masks)
- **MiniMax H3 Add Guide** for anchoring images, clips, and audio at chosen frames of a MiniMax-H3 video (timeline guides)
- **MiniMax-H3 Control** for conditioning MiniMax-H3 on a prepared control video with Fun ControlNet Union (canny, depth, hed, mlsd, pose, inpaint)
- **FastH3 Deployment** for routing text-to-video requests to a MiniMax-H3 server with FastH3 fused at startup
- **Multimodality Understanding** for multimodality-to-text and multimodality-to-audio tasks
- **TTS** and **TTS Voice Clone** for TTS tasks
- **Generate Music** for lyrics-and-description-to-music generation

This extension also offers example workflows (at **ComfyUI sidebar -> Templates -> vLLM-Omni**)

> [!NOTE]
> The node UI and feature designs are intended to match vLLM-Omni online serving interfaces. It cannot offer more than what the interfaces support.

Every node carries the vLLM-Omni mark in its title bar and is tinted by what it outputs, so a graph is readable at a glance:

| Colour | Nodes | What they produce |
| --- | --- | --- |
| Blue | Generate Image, Generate Video, Restore Video, Multimodality Understanding, TTS, TTS Voice Clone | A generated or restored image, video, audio, or text. These are the only nodes that reach a server. |
| Amber | AR / Diffusion / Multi-Stage Sampling Params | Sampling parameters that apply to any model |
| Purple | Qwen TTS Params, Wan Video Params, MiniMax-H3 Video Params | Parameters that only one model family accepts |
| Red | LoRA, FastH3 Deployment | Which weights the server is expected to have loaded |
| Teal | Video References, Latent Mask Editing | Reference media |

Recolouring a node by hand (right click -> Colors) overrides its tint, and the choice is kept.

The `model` field of every node that talks to a server doubles as a dropdown of what the server actually serves: the list is fetched through the ComfyUI backend (so `localhost` resolves to the host running ComfyUI, not to the machine the browser is on), refetched when the `url` field changes or the **↻ refresh models** button is clicked, and falls back to plain free text whenever the list cannot be fetched -- a server that is down must never make the node unfillable. A value saved in a workflow is always kept, even when it is not in the fetched list.

**Generate Video** takes a clip length in seconds (`duration`), not a frame count. Frames stay the wire unit and are derived with the node's `fps`, so the length is always measured against the rate that is actually served; models that accept only certain frame counts still round to their own lattice server-side. Graphs saved before this widget existed stored `num_frames` in its place and are converted on load, using the fps recorded alongside it -- the browser console names every node it rewrites.

To build a simple workflow yourself,

- Drag a generation node onto the canvas.
- Depending on your need, grab built-in multimedia file loader nodes, such as **image->Load Image**, **image->video->Load Video**, **audio->Load Audio**
- Depending on your need, grab built-in multimedia file preview nodes, such as **image->Preview Image**, **image->video->Save Video**, **audio->Preview Audio**, **utils->Preview as Text**.
- If you want to tune sampling parameters, grab corresponding nodes from **vLLM-Omni-> Sampling Params**.
    - For multi-stage models, you can connect multiple **AR Sampling Params** and **Diffusion Sampling Params** nodes to a **Multi-Stage Sampling Params List** node, and connect this node to the generation node.
    - For some multi-stage models like BAGEL, [only one stage's sampling parameters are exposed and tunable via vLLM-Omni's online serving API](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/bagel/). Thus, these models are treated as single-stage ones. Please check the vLLM-Omni documentation on how to correctly set each model's sampling parameters.
    - For multi-stage models where all stages are either autoregression or diffusion, you can also connect only a single Sampling Params node, indicating that this set of sampling parameters will be used for all stages.

## Screenshots and Examples

### Multimodal understanding (e.g., Qwen Omni series, BAGEL)

(Also available at **ComfyUI sidebar->Template->vLLM-Omni->vLLM-Omni Multimodal Understanding**)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-understanding.jpg">
    <img alt="vLLM-Omni multimodal understanding" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-understanding.jpg" width=55%>
  </picture>
</p>

> [!TIP]
> Although this node enables all-modality input, you should check whether the specific model you host and request for supports the modalities you connect to the node.

You can configure per-stage sampling parameters for multi-stage models.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-multi-stage.jpg">
    <img alt="vLLM-Omni multiple stages" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-multi-stage.jpg" width=55%>
  </picture>
</p>

### Text-to-image and image-to-image generation (e.g., Z-Image-Turbo, Qwen-Image-Edit, BAGEL)

(Also available at **ComfyUI sidebar->Template->vLLM-Omni->vLLM-Omni Image Generation**)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-image-generation.jpg">
    <img alt="vLLM-Omni image generation" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-image-generation.jpg" width=55%>
  </picture>
</p>

> [!TIP]
> The node automatically choose text-to-image or image-to-image API endpoints depending on whether you connect an image input or not.

### Text-to-video and image-to-video generation (e.g., Wan, MiniMax-H3)

(Also available at **ComfyUI sidebar->Template->vLLM-Omni->vLLM-Omni Video Generation**)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-video-generation.jpg">
    <img alt="vLLM-Omni video generation" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-video-generation.jpg" width=55%>
  </picture>
</p>

> [!TIP]
> Connect **frame** for backward-compatible first-frame / image-to-video behavior. For MiniMax-H3 FL2VA,
> connect **first_frame**, **last_frame**, or both to condition the start frame, end frame, or both.
>
> For reference-conditioned generation (MiniMax-H3 Ref2VA), connect a **Video References** node instead.
>
> Do not combine `frame` with `first_frame` or `last_frame`, and do not combine any frame input with
> `references`. Task routing is automatic from which inputs you connect.
>
> For MiniMax-H3 Fun ControlNet Union, connect a **MiniMax-H3 Control** node. It cannot be combined with `frame`, `first_frame`/`last_frame`, `references`, or `latent_edit` either.

#### MiniMax-H3 Fun ControlNet Union

The **MiniMax-H3 Control** node conditions generation on a prepared control video, and supports `canny`, `depth`, `hed`, `mlsd`, `pose` and `inpaint`. Connect its `control` output to **Generate Video**.

| Input | Type | Notes |
| --- | --- | --- |
| `control_type` | enum | Which conditioning mode to request |
| `strength` | `FLOAT` | Conditioning strength for the selected mode, 0-10, default 1 |
| `control_video` | `VIDEO` | Required for the five structure modes, optional for `inpaint` |
| `source_video` | `VIDEO` | Optional original video; requires a mask |
| `mask` | `MASK` | Static mask for the whole clip; white/1 marks what to regenerate |
| `mask_video` | `VIDEO` | Per-frame mask; mutually exclusive with `mask` |

The server must be started with the Fun ControlNet checkpoint (`--controlnet-model-path`); otherwise it rejects control requests.

Control videos are re-encoded as lossless H.264 (CRF 0) and uploaded whole, so each must stay under the server's 512 MiB limit; trim long clips to the generated length first. The encode needs a ComfyUI whose video `save_to` accepts `crf`; this was tested on ComfyUI 0.35.0.

> [!TIP]
> The node does not run any preprocessor. Build the control video in the workflow, for example `SDPose Keypoint Extractor` -> `SDPose Draw Keypoints` -> `Create Video` for `pose`, and pass an `IMAGE` batch through **Create Video** before connecting it.
>
> `inpaint` needs a mask. `source_video` without a mask, a structure mode without `control_video`, or `mask` together with `mask_video` are rejected before the request is sent.
>
> Control requires a MiniMax-H3 model. With a custom `--served-model-name`, connect **MiniMax-H3 Video Params** so the request is recognised as MiniMax-H3; this is not needed when **FastH3 Deployment** is connected.

Load [MiniMax-H3 ControlNet Canny](example_workflows/vLLM-Omni%20MiniMax-H3%20ControlNet%20Canny.json) for a complete `canny` example. It builds the control video inside the workflow: **Load Video** -> fit to 1344x768 -> first 124 frames -> **Canny** -> **Create Video** at 24 FPS. It also saves the control video so you can check the edges. The **Load Video** filename is a placeholder; put your source clip in `ComfyUI/input`.

For MiniMax-H3 Ref2VA, **Video References** accepts up to 9 images (`image_1`–`image_9`),
3 videos (`video_1`–`video_3`), and 3 audio clips (`audio_1`–`audio_3`), with at most
12 connected inputs in total. Any mixture containing at least one image or video
is supported; empty and audio-only references are rejected. For example, you can
combine 6 images, 3 videos, and 3 audio clips in one request.

Within each media type, references follow slot-number order, skipping unconnected
slots. For example, connecting `image_2` and `image_9` sends `image_2` as the first
image and `image_9` as the second. Each image slot uses the first image in its batch.
Existing connections to `image_1`, `image_2`, `audio_1`, `audio_2`, `video_1`, and
`video_2` remain valid in saved workflows.

#### MiniMax-H3 Reference to Video

Load [MiniMax-H3 Reference to Video](example_workflows/vLLM-Omni%20MiniMax-H3%20Reference%20to%20Video.json)
from **Templates → ComfyUI-vLLM-Omni**, or drag the JSON onto the canvas.
It adapts the [official ComfyUI R2V workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json)
to remote vLLM-Omni execution. Model loading, sampling, and VAE decoding run on
the server; ComfyUI loads references and saves the returned video.

Use an extension version containing the audio-preserving output fix from
[#7456](https://github.com/vllm-project/vllm-omni/pull/7456). Older versions can
drop the server's audio while decoding the response.

1. Start a Ref2VA-capable service using the [MiniMax-H3 recipe](../../recipes/MiniMaxAI/MiniMax-H3.md).
   Set **Generate Video** to its `/v1` URL and served model name.
2. Select your own image in **Load Image**. The template's filenames are placeholders
   relative to ComfyUI's input directory; no sample assets or model weights are bundled.
3. Connect **Load Video** and **Load Audio** when needed. Duplicate the loaders to
   fill more reference slots, or disconnect the image when using video-only input.
   Keep the references connected to **Generate Video** and leave `frame` disconnected.
4. Match prompt tags such as `<Picture 1>`, `<Video 1>`, and `<Audio 1>` to the
   connected references, counting each media type separately and skipping empty slots.
   To drive the clip with a soundtrack instead, connect exactly one audio and set
   **MiniMax-H3 Video Params → audio_mode** to `lock_source`; that audio is not
   counted as an `<Audio N>` reference.
5. Run the workflow. **Save Video** writes an MP4 under `output/video/` and preserves
   the generated audio when the audio-output prerequisite is installed.

The default is **1344×768, 24 FPS, 124 frames** (about 5.17 seconds), 50 sampling
points, video flow shift 12, audio flow shift 3, and seed 42. The duration widget
is set to 5.167 seconds, which converts to 124 frames at 24 FPS. Other H3 canvas presets
are 1024×768, 768×768, 768×1024, and 768×1344. Keep frame counts at `17k+5` within
the 4–15 second output range; examples are 107, 124, 209, and 345 frames.
Reference video/audio clips must each be 2–15 seconds, with at most 15 seconds of
reference video and at most 15 seconds of standalone reference audio per request.

For **Ref2VA Turbo**, use the same template. Configure a Ref2VA-only server with
`--lora-backend peft --lora-path "$TURBO_LORA" --task-type ref2va`, following the
recipe's hardware setup. For `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`,
set **LoRA** `local_path` to that file's path on the server and connect it to
**Generate Video**. Use `num_inference_steps=5`, `flow_shift=12`,
`audio_flow_shift=3`, and LoRA `scale=1`. The API counts sigma points, so five
points produce four denoiser evaluations. Use the Diffusers artifact, not its
`_comfyui_` export. To return to base mode, disconnect LoRA and restore 50 steps.
The 8-step v1.0 Ref2VA adapter instead requires nine sampling points and video
flow shift 6. FastH3 is a separate startup-fused T2VA deployment.

Validate the template's wiring and defaults locally with:

```bash
python -m pytest tests/e2e/features/comfyui/test_h3_reference_workflow.py -q
```

For a real-model check, run the workflow against H3 and inspect the saved file:

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,r_frame_rate,sample_rate,channels \
  -show_entries format=duration -of json "$OUTPUT_MP4"
```

Check for 24 FPS video, stereo audio, the requested canvas, and aligned duration;
also play the result to assess reference conditioning and audio/video synchronization.
Record the server commit, model/adapter, input assets, prompt, settings, and output
alongside the result. Local schema or mocked-server checks do not establish H3
generation quality or replace this real-model validation.

#### MiniMax H3 Character Replacement

Download [Character Swap V1.5](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA) on the machine running vLLM-Omni:

```bash
hf download akatz-ai/MiniMax-H3-Character-Swap-LoRA h3_character_swap_v1.5.safetensors --local-dir /models
```

In **LoRA(Character Swap)**, set `local_path` to `/models/h3_character_swap_v1.5.safetensors`, enable the adapter and start with `scale=1.0`. Leave `int_id=0` for automatic identification. Start the service with `--lora-backend peft`; for Swap + Turbo composition, use `--max-cpu-loras 3`. The target image supplies appearance, the video supplies motion and scene structure, and the prompt identifies which person to replace. FastH3-Omni + Swap requires a server that supports request-level LoRA for that checkpoint.

Load [MiniMax-H3 Character Replacement](example_workflows/vLLM-Omni%20MiniMax-H3%20Character%20Replacement.json),
select your own target image and motion video, and replace both `/models/...`
adapter paths with server-local files. This template uses reference conditioning,
not pixel-locked editing, and saves generated audio. Supply the complete reference
video and target duration. Enable long_video and select continuation for the existing
latent-tail continuation mode. Adjust continuation_window_frames and
continuation_overlap_frames to set the window length and overlap. Reference video
and audio are reused across windows, rather than sliced to each source interval.
Inspect continuity and source action timing; output may include frame-alignment padding.

### MiniMax H3 Text to Video

Import [the H3 text-to-video template](example_workflows/MiniMax_H3_Text_to_Video.json)
for remote video generation with audio. It includes connected base presets and
optional Turbo sampling, H3 parameters, and Remote LoRA nodes. See the
[workflow guide](docs/minimax-h3-t2v.md) for server setup, artifact-specific Turbo
settings, and saved-video/audio validation. Set Remote LoRA’s `local_path` to
the downloaded Turbo artifact on your server before enabling Turbo.

#### H3 all-in-one (H3-06)

The **vLLM-Omni MiniMax-H3 All-in-One** template puts T2VA, first/last-frame FL2VA, and mixed-reference Ref2VA in one graph. The server URL, served model name, duration, sampling, and flow shifts are set once and shared by all modes; enable the group for the mode you want to run. It does not include a prompt LLM. See [workflow setup](docs/h3-06-all-in-one.md).

#### H3 music video from a supplied track (H3-03)

The **vLLM-Omni MiniMax-H3 Music Video** template cuts a supplied music track into 124-frame slices, generates one Ref2VA shot per slice with a shared character/style image, and joins the shots with the original track as the soundtrack. Each shot sets **MiniMax-H3 Video Params → audio_mode** to `lock_source`, so its slice drives generation instead of acting as a short reference. Requires ComfyUI v0.36.0+ for Concatenate Video. See [workflow setup](docs/h3-03-music-video.md).

#### H3 video upscale (WF-07)

The **vLLM-Omni MiniMax H3 Video Upscale** template generates video remotely, upscales it with SeedVR2, and saves the original and upscaled videos with the generated audio and FPS. See [workflow setup](docs/wf07-h3-upscale.md).

#### FastH3 text-to-video

FastH3 is fused into MiniMax-H3 when the vLLM-Omni server starts; it is not a request-switchable LoRA. Download one adapter variant and start a non-offloaded FL2VA server. For the Dense / Data-Free profile:

```bash
export FASTH3_DIR=/path/to/fasth3
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors --local-dir "${FASTH3_DIR}"

vllm serve /path/to/MiniMax-H3 \
  --omni \
  --task-type fl2va \
  --lora-path "${FASTH3_DIR}/dense-datafree/adapter_model.safetensors" \
  --port 8000
```

For the VSA / Data-Free profile, download `vsa-datafree/adapter_model.safetensors`, use that file as `--lora-path`, and add:

```bash
--diffusion-attention-backend FASTVIDEO_VSA \
--fastvideo-vsa-topk 64
```

The VSA profile also requires a compatible `fastvideo-kernel` installation. See the [MiniMax-H3 FastH3 recipe](../../recipes/MiniMaxAI/MiniMax-H3.md#fasth3-adapter) for the full serving contract, supported parallel layouts, and kernel requirements.

Open the **vLLM-Omni FastH3 Text to Video** template, then:

- Set the server URL and served model name on **FastH3 Deployment**.
- Connect its output to **Generate Video → fast_h3**. When connected, the deployment node's URL and model take precedence over the corresponding Generate Video widgets.
- Keep `frame`, `references`, **LoRA**, and **MiniMax-H3 Video Params** disconnected. FastH3 Preview v1 supports T2VA only, is already fused, and owns both flow shifts.
- A connected **Diffusion Sampling Params** node may set seed and other ordinary sampling options. The integration always enforces four denoising steps and 24 FPS for FastH3.

The node records which server the workflow targets; it does not start one, nor switch adapters or attention backends on a running server.

#### Latent-mask editing (MiniMax-H3)

The [WF-05 template](example_workflows/vLLM-Omni%20MiniMax-H3%20Latent%20Mask%20Editing.json) contains inpainting, object removal, continuation, and extension examples in one graph. See the [workflow guide](docs/wf05-h3-latent-editing.md) for inputs, mask settings, dependencies, and preview limitations.

Connect a **Latent Mask Editing** node to **Generate Video → latent_edit** to edit a source clip instead of generating from scratch. It uploads the source media and serializes the video/audio noise masks the MiniMax H3 API accepts:

- `source_video` / `source_audio` — the media to edit.
- `video_mask` — a ComfyUI mask image; `0` preserves a region, `1` regenerates it, fractional values blend. A 2D mask `[H, W]` is applied to every frame; a 3D mask `[T, H, W]` is treated as a temporal mask (one slice per frame) for continuation or extension. The client area-downsamples it by 16 to keep the upload small, and the server resizes it to the video latent grid.
- `audio_mask` — a scalar in `[0, 1]`; `0` keeps the source audio, `1` regenerates it, fractional values blend.

The node requires at least one mask; the server then enforces the cross-field rules (a source requires a matching mask, and a non-trivial mask requires its source). This node only forwards inputs to the server; the served model must declare latent-mask editing support (MiniMax-H3).

#### Timeline guides (MiniMax-H3)

Timeline guides pin a still image or a clip, optionally with audio, to a chosen
frame of the generated video. Chain **MiniMax H3 Add Guide** nodes
(`VLLMOmniMiniMaxH3Guide`) and connect the last one to
**Generate Video → timeline_guides**:

```text
Load Image ─────image──► Add Guide (frame_index=36) ─┐
Load Image ─────image──► Add Guide (frame_index=72) ◄┘ ─┐
Load Video ─────video──► Add Guide (frame_index=-22) ◄──┘ ──timeline_guides──► Generate Video ──► Save Video
Load Audio ─────audio──┘
```

The [MiniMax-H3 Timeline Guides](example_workflows/vLLM-Omni%20MiniMax-H3%20Timeline%20Guides.json)
template (**Templates → ComfyUI-vLLM-Omni**) wires this chain, guiding the closing
clip with its own soundtrack through **Get Video Components → audio**. Its loader
filenames are placeholders; select your own files.

| Socket | Type | Notes |
| --- | --- | --- |
| `frame_index` (widget) | `INT` | Output frame where the guide starts, on the 24 FPS timeline. Negative values count from the end: `-1` is the last frame, `-22` starts a clip that ends on it. |
| `image` | `IMAGE` | One frame. A batch of several images is rejected; turn it into a clip with **Create Video** at 24 FPS and connect `video` instead. |
| `video` | `VIDEO` | A clip, decoded by the server at 24 FPS whatever its own frame rate. Its soundtrack is **not** used; connect **Get Video Components → audio** to `audio` to guide with it. |
| `audio` | `AUDIO` | One mono or stereo clip, starting at the same frame. Audio-only guides are allowed. |
| `timeline_guides` (input/output) | `H3_TIMELINE_GUIDES` | The chain. Each node appends one guide to a copy, so branching a chain never changes the other branch. |

Every node adds exactly one guide that carries media: `image`, `video`, `audio`,
`image` + `audio`, or `video` + `audio`. `image` and `video` cannot be combined.
Guides keep chain order; they are not sorted, deduplicated, or checked for
overlap, so a repeated image is sent as its own guide.

The request carries the guides as a `timeline_guides` JSON manifest plus one
repeated `guide_files` upload per medium (PNG; MP4, with MKV/WebM sources
converted; FLAC), numbered in chain order with the visual input before the audio,
as described in the [Videos API](../../docs/serving/videos_api.md#minimax-h3-timeline-guides).
Limits are 8 guides, 16 uploads, and 30 MiB per image, 50 MiB per clip, 15 MiB
per audio after encoding; larger inputs fail before anything is uploaded.

The server, not the node, fits guides to the video. The output length is
aligned up to `17k+5` frames, a negative index resolves to `frames + index`,
and a clip is shortened to `17k+5` frames (or 1) and must fit between its start
and the end of the video. Errors such as an index that does not fit are shown
in ComfyUI as returned by the server.

A guided request:

- sets **fps** to 24 (other values are rejected rather than changed) and sends
  `quality=lossless`; requests without guides are sent exactly as before;
- keeps the usual routing: guides alone are T2VA, with `first_frame` /
  `last_frame` FL2VA, with **Video References** Ref2VA. References keep their
  `<Picture N>` / `<Video N>` / `<Audio N>` numbering;
- requires base H3 weights and rejects FastH3 (Preview and Omni) or active request-level LoRA before upload; set every LoRA scale to zero or disconnect it. Server-side restrictions remain authoritative.

Use a vLLM-Omni server that includes timeline guides
([#8838](https://github.com/vllm-project/vllm-omni/pull/8838)). Older servers
may silently ignore the unknown form fields, and `/v1/models` does not report
guide support, so the node cannot detect this for you.

The served model name must match the H3 model spec (for example, start the
server with `--served-model-name MiniMaxAI/MiniMax-H3`); otherwise the node
refuses to send guides, for example to a server named only by its model path.

The [Agentic Continuation](example_workflows/vLLM-Omni%20MiniMax-H3%20Character%20Replacement%20Continuation%20(Agentic).json) example uses two boundary images at frames 0 and 21, ordinary Generate Video requests, and native crop/concatenation nodes. **Reverse Image Batch** reverses the frame order of any `IMAGE` batch; it has no model, VAE, request, or queue logic. The supplied segment lengths reproduce a 27-second example; another source needs matching contexts, prompts, and cut values. Loader filenames are placeholders. Guided segments require base H3 weights with all request-level LoRAs disabled; FastH3 + Swap + guides is unsupported.

Validate the node and the request contract locally with:

```bash
python -m pytest -q tests/e2e/features/comfyui/test_timeline_guides.py
python -m pytest -q tests/e2e/features/comfyui/test_comfyui_integration.py -k timeline_guides
```

The second command runs the real `/v1/videos` multipart parser against a mocked
model. Neither replaces a real-model run: generate against H3, save with
**Save Video**, then check the file with
`ffprobe -v error -count_frames -show_streams -show_format -of json "$OUTPUT"`
and `ffmpeg -v error -i "$OUTPUT" -f null -`, and watch and listen to it.

### TTS (e.g., Qwen TTS series)

(Also available at **ComfyUI sidebar->Template->vLLM-Omni->vLLM-Omni TTS**)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-tts.jpg">
    <img alt="vLLM-Omni TTS" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-tts.jpg" width=55%>
  </picture>
</p>

> [!TIP]
> There is a dedicated node for VoiceClone tasks with reference audio input. Other simple text-to-speech tasks should use the regular TTS node.

### Text to Music (e.g., MiniMax Music 3/YuE2)

Open **ComfyUI sidebar->Template->vLLM-Omni** and select:

- **vLLM-Omni Text to Music(MiniMax Music3)** — the existing MiniMax Music 3 template.
- **vLLM-Omni Text to Music(YuE2)** — uses YuE2 Params with `cot=off` by default.

Set Generate Music's URL and model to the running service. Caption and lyrics
map to `instructions` and `input`. For YuE2, connect YuE2 Params to the optional
`model_specific_params` input. Leave ABC empty for `off`; `melody/full` require
ABC score text, pasted or connected as STRING, not a file path.

See the [YuE2 recipe](../../recipes/m-a-p/YuE2-3B.md) for deployment details.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-text-to-music.jpg">
    <img alt="vLLM-Omni Text to Music(MiniMax Music3)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-text-to-music.jpg" width=55%>
  </picture>
</p>

### Music Cover (e.g. YuE2)

Open **ComfyUI sidebar->Template->vLLM-Omni->vLLM-Omni Music Cover(YuE2)**.

The template uses native Load Audio, Audio Encoder Loader and SheetSage2 Audio
to ABC nodes locally, followed by remote YuE2 generation. Download
[sheetsage2_bf16.safetensors](https://huggingface.co/Comfy-Org/YuE2/resolve/main/audio_encoders/sheetsage2_bf16.safetensors)
into `models/audio_encoders/`, select it and upload your source audio.

Keep SheetSage2 mode and YuE2 Params cot matched; both default to `melody`.
Provide lyrics separately. The remote service receives ABC text rather than the
source recording, and returns audio without generated ABC or truncation metadata.
Save Audio previews and saves the result.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-music-cover.jpg">
    <img alt="vLLM-Omni Music Cover(YuE2)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-music-cover.jpg" width=55%>
  </picture>
</p>

### Chaining multiple model services

(Also available at **ComfyUI sidebar->Template->vLLM-Omni->vLLM-Omni Chaining Services**)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-chaining-services.jpg">
    <img alt="vLLM-Omni TTS" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/apps/ComfyUI-vLLM-Omni/docs/images/comfyui-chaining-services.jpg" width=55%>
  </picture>
</p>

## Develop

Follow the [development convention and rules of vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/).

Node tints and the title-bar mark are applied in `web/main.js`, keyed off each node's declared output types rather than a list of node names. A new node that returns an existing type is themed with no front-end change; a new output type needs one entry in `FAMILY_BY_OUTPUT` there.

## Limitation and Non-Goals

- Single server mode only. No automatic load balancing or failover.
- Control conditioning is prepared client-side. The nodes upload prepared control, source and mask media; they neither run preprocessors nor load ControlNet weights in the ComfyUI process.
- MiniMax-H3 control cannot be combined with image or clip timeline guides. The client does not check this; with a nonzero control strength the server rejects the job when generation starts. Audio-only guides work with control.
- Features set is bounded to vLLM-Omni's online service capability, including
    - The types of models supported in online mode,
    - The types of sampling parameters supported in the online mode,
    - The ways to send files (primarily through full-length base64 in JSON payload),
    - Figuring out errors in the payload (such as unsupported fields by a specific model) if the endpoint does not explicitly return an error,
    - (The lack of) Authentication
    - (The lack of) Progress indicator

## Support

If you are new to ComfyUI, please check out [its documentation](https://docs.comfy.org/) for usage instructions.

If you are new to vLLM-Omni, please also check out [its documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/) for usage instructions.

Whenever you find an issue or problem, please

- First find out if this is an upstream limitation of vLLM-Omni's online serving mode, by [checking their documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/examples/).
- [Open an issue](https://github.com/vllm-project/vllm-omni/issues) that clearly describes this ComfyUI or online service problem.

## Acknowledgements

Features

- <https://github.com/dougbtv/comfyui-vllm-omni/> The official reference implementation for ComfyUI integration with vLLM-Omni's DALL-E compatible image generation API.
- <https://github.com/Comfy-Org/ComfyUI/tree/master/comfy_extras> ComfyUI's built-in node implementations.

UI/UX design references

- <https://github.com/sgl-project/sglang/pull/15271> SGLang Diffusion's official ComfyUI integration for image and video generation.
- <https://github.com/SXQBW/ComfyUI-Qwen-Omni> A third party ComfyUI integration for Qwen Omni series.
- <https://github.com/flybirdxx/ComfyUI-Qwen-TTS> <https://github.com/DarioFT/ComfyUI-Qwen3-TTS> Third  party ComfyUI integrations for Qwen TTS series.
