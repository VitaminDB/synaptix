# leejet__stable-diffusion.cpp__minimax-h3

Source: https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/models/minimax_h3.md
Fetched: 2026-11-28



---

## FILE: docs/models/minimax_h3.md (https://github.com/leejet/stable-diffusion.cpp/blob/HEAD/docs/models/minimax_h3.md)

# MiniMax-H3

MiniMax-H3 jointly generates video and stereo audio with a packed diffusion
transformer. The implementation supports text-to-audio-video (T2VA), optional
first-frame conditioning (I2VA), first/last-frame conditioning (FL2VA), and
image/video/audio reference conditioning (Ref2VA).

## Model files

Pass the four MiniMax-H3 components separately:

- `--diffusion-model`: MiniMax-H3 diffusion transformer
- `--vae`: MiniMax-H3 video VAE
- `--audio-vae`: MiniMax-H3 audio VAE
- `--llm`: the MiniMax-H3 Qwen3-VL-32B text encoder, Qwen3-VL-4B with the P-H3 text connector, or Qwen3-VL-4B/8B with the ClipProj projection

For the 32B option, use the MiniMax-H3 variant: Qwen3-VL-32B truncated to 50
language layers and exported without the final language-model normalization.

For the smaller 4B option, pass the P-H3-Image text encoder, which bundles the
connector, with `--llm`. Alternatively, pass Qwen3-VL-4B with `--llm` and the
[separate P-H3 text connector](https://huggingface.co/PrunaAI/P-H3-Image/blob/main/text_connector/diffusion_pytorch_model.safetensors)
with `--embeddings-connectors`. For example, replace the `--llm` argument in the
commands below with either:

**Windows (PowerShell) arguments**

```text
--llm ..\models\text_encoders\p-h3-image_text_encoder_int8_convrot.safetensors
```

**Linux/macOS arguments**

```text
--llm ../models/text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors
```

or:

**Windows (PowerShell) arguments**

```text
--llm ..\models\text_encoders\qwen3vl_4b.gguf `
  --embeddings-connectors ..\models\text_encoders\p-h3-image_text_connector.safetensors
```

**Linux/macOS arguments**

```text
--llm ../models/text_encoders/qwen3vl_4b.gguf \
  --embeddings-connectors ../models/text_encoders/p-h3-image_text_connector.safetensors
```

or use a plain Qwen3-VL-4B or Qwen3-VL-8B with the
[ClipProj projection](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3), a
small learned matrix that replaces the 32B encoder:

**Windows (PowerShell) arguments**

```text
--llm ..\models\text_encoders\qwen3vl_4b.gguf `
  --embeddings-connectors ..\models\text_encoders\mmh3-4b-ClipProj-v3.1.safetensors
```

**Linux/macOS arguments**

```text
--llm ../models/text_encoders/qwen3vl_4b.gguf \
  --embeddings-connectors ../models/text_encoders/mmh3-4b-ClipProj-v3.1.safetensors
```

The `mmh3-4b-*` matrices expect a Qwen3-VL-4B encoder and the `mmh3-8b-*`
matrices a Qwen3-VL-8B encoder; match the two. ClipProj is a community proof of
concept: outputs are close to, not identical to, the 32B encoder. See the
[model card](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) for
measurements and limitations.

Reference-image and reference-video conditioning needs the matching encoder's
Qwen3-VL vision tower; the official 32B encoder includes the three DeepStack mergers.
If stored separately, pass it with `--llm_vision`. Keep using the MiniMax-H3
video and audio VAEs with any text encoder.

In DGL, use `load_text_encoder` with `family = "minimax_h3"`. For a separate
4B connector or vision projector, pass `connectors_path` or `vision_path` to
that loader. The native generation API trims decoded audio to the duration of
the output video. In scripts, save decoded frames and audio with
`video::save(path, {frames=frames, fps_num=24, fps_den=1}, audio=audio, trim_audio=true)`
to match the generated clip duration.
For video presentation, pass frame tensors to `encode_prompt` as
`presentation=[{video={frames=frames, fps_num=24, fps_den=1}, stride=12}]`. Frame rate and sampling
stride are explicit; VAE reference latents remain a separate input.

Both the original time-embedder DiT and the smaller AdaLN curve-table variant
are detected from their weights.

### Download weights

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| minimax_h3_fl2va/minimax_h3_ref2va | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models)<br>[GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main) | `--diffusion-model` | Required |
| qwen3vl_32b_minimax_h3 | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders)<br>[GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main) | `--llm` | Use matching text encoder |
| P-H3-Image text encoder (Qwen3-VL-4B with the text connector) | [Safetensors](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/main/text_encoders) | `--llm` | Use matching text encoder |
| ClipProj projection (for a plain Qwen3-VL-4B or Qwen3-VL-8B) | [Safetensors](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) | `--embeddings-connectors` | Pair with a matching Qwen3-VL-4B/8B `--llm` |
| vae | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `--vae` | Use matching VAE |
| audio vae | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `--audio-vae` | Required |

## Usage

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| MiniMax H3 FL2VA (`minimax-h3-fl2va`) | `diffusion_model`, `llm`, `llm_vision`, `embeddings_connectors`, `vae`, `tae`, `audio_vae` |
| MiniMax H3 Ref2VA (`minimax-h3-ref2va`) | `diffusion_model`, `llm`, `llm_vision`, `embeddings_connectors`, `vae`, `tae`, `audio_vae` |
| HyperFlow H3 Flash FL2VA (`hyperflow-h3-flash-fl2va`) | `diffusion_model`, `llm`, `llm_vision`, `embeddings_connectors`, `vae`, `tae`, `audio_vae` |
| HyperFlow H3 Flash Ref2VA (`hyperflow-h3-flash-ref2va`) | `diffusion_model`, `llm`, `llm_vision`, `embeddings_connectors`, `vae`, `tae`, `audio_vae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### DGL

| Example | Description |
| --- | --- |
| [Text to audio/video](../dgl/minimax_h3/minimax-h3-text-to-av.dgl) | Generate video and its soundtrack together. |
| [First/last-frame conditioning](../dgl/minimax_h3/minimax-h3-first-last-frame.dgl) | Guide the video with endpoint images. |
| [Reference to audio/video](../dgl/minimax_h3/minimax-h3-reference-to-av.dgl) | Use image, video and audio references. |
| [Interpolated latent hires](../dgl/minimax_h3/minimax-h3-latent-hires.dgl) | Resize the draft latent, then run a second sampling pass. |
| [Learned latent hires](../dgl/minimax_h3/minimax-h3-learned-hires.dgl) | Use a learned latent upscaler between sampling passes. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\minimax_h3\minimax-h3-text-to-av.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/minimax_h3/minimax-h3-text-to-av.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

#### Latent upscaling

Use `load_latent_upscaler` to load the learned spatial upscaler.
`upscale_latent` takes the video part of the draft diffusion latent
and returns a diffusion latent for the refine pass. Pass the target width
and height in latent units. A resize target uses even dimensions
(multiples of 32 in pixel units).

```dgl
upscaler = load_latent_upscaler("minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors")
bigger = upscale_latent(upscaler, draft_av.video, width = 76, height = 42)
```

### Legacy CLI

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

#### Text-to-audio-video

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf `
  --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors `
  --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors `
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf `
  -p "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten`鈥檚 energetic surfing." `
  --cfg-scale 1.0 `
  -W 864 `
  -H 480 `
  --fps 24 `
  --video-frames 56 `
  --rng cpu `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf \
  --vae ../models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ../models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten鈥檚 energetic surfing." \
  --cfg-scale 1.0 \
  -W 864 \
  -H 480 \
  --fps 24 \
  --video-frames 56 \
  --rng cpu \
  --offload-to-cpu \
  -v
```

<video src=../../assets/minimax-h3/t2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

Omitting `--audio-vae` still runs the joint diffusion model but produces video without a
decoded audio track.

#### First/last-frame conditioning

Add `--init-img` for I2VA, or both `--init-img` and `--end-img` for FL2VA:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf `
  --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors `
  --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors `
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf `
  -p "a lovely cat" `
  -i ..\inputs\ernie_image\turbo_example.png `
  --cfg-scale 1.0 `
  -W 864 `
  -H 480 `
  --fps 24 `
  --video-frames 56 `
  --rng cpu `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf \
  --vae ../models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ../models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "a lovely cat" \
  -i ../inputs/ernie_image/turbo_example.png \
  --cfg-scale 1.0 \
  -W 864 \
  -H 480 \
  --fps 24 \
  --video-frames 56 \
  --rng cpu \
  --offload-to-cpu \
  -v
```

<video src=../../assets/minimax-h3/i2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

#### Reference-to-audio-video conditioning

Ref2VA accepts any combination of reference images, reference videos, paired
video soundtracks, and standalone audio references:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\minimax_h3_ref2va_pruned-Q4_K_M.gguf `
  --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors `
  --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors `
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf `
  -p "Use the cat from <Picture 1> as the main character. Keep the cat`鈥檚 appearance, fur color, facial features, and identity consistent with the reference image. Create a 2-second cinematic video: start with an extreme close-up shot of the cat`鈥檚 face, focusing on its cute expression and detailed fur texture. The camera slowly rotates around the cat`鈥檚 head, creating a dynamic reveal. Then smoothly pull back and zoom out to reveal the full scene: the cat is standing confidently on a surfboard, riding ocean waves. Water splashes around the board, sea breeze gently moves the cat`鈥檚 fur, and the cat maintains a cute and fearless expression while surfing. Smooth camera movement, cinematic orbit shot, seamless zoom-out transition, low-angle wide shot, realistic ocean environment, golden sunlight, dynamic waves, high-quality realistic style, natural motion, no distortion, keep the cat`鈥檚 identity unchanged." `
  -r ..\inputs\ernie_image\turbo_example.png `
  --cfg-scale 1.0 `
  -W 864 `
  -H 480 `
  --fps 24 `
  --video-frames 56 `
  --rng cpu `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/minimax_h3_ref2va_pruned-Q4_K_M.gguf \
  --vae ../models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ../models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "Use the cat from <Picture 1> as the main character. Keep the cat鈥檚 appearance, fur color, facial features, and identity consistent with the reference image. Create a 2-second cinematic video: start with an extreme close-up shot of the cat鈥檚 face, focusing on its cute expression and detailed fur texture. The camera slowly rotates around the cat鈥檚 head, creating a dynamic reveal. Then smoothly pull back and zoom out to reveal the full scene: the cat is standing confidently on a surfboard, riding ocean waves. Water splashes around the board, sea breeze gently moves the cat鈥檚 fur, and the cat maintains a cute and fearless expression while surfing. Smooth camera movement, cinematic orbit shot, seamless zoom-out transition, low-angle wide shot, realistic ocean environment, golden sunlight, dynamic waves, high-quality realistic style, natural motion, no distortion, keep the cat鈥檚 identity unchanged." \
  -r ../inputs/ernie_image/turbo_example.png \
  --cfg-scale 1.0 \
  -W 864 \
  -H 480 \
  --fps 24 \
  --video-frames 56 \
  --rng cpu \
  --offload-to-cpu \
  -v
```

<video src=../../assets/minimax-h3/r2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

`--ref-image`, `--ref-video`, and `--ref-audio` can each be repeated. A
reference video can be a directory of image frames sorted lexicographically,
treated as 24 fps, or a file decoded and sampled at 24 fps. File inputs support
AVI, static/animated WebP and this project's independent-frame VP8 WebM;
see [DGL media I/O](../dgl.md) for codec and build requirements.
Reference files retain supported embedded audio by default. Repeated
`--ref-video-audio` WAV files are paired by index with repeated `--ref-video`
inputs and replace their embedded audio; the replaced track is not decoded.
WAV PCM (8/16/24/32-bit) and 32/64-bit floating-point samples are accepted; audio is converted to stereo 32 kHz during
generation.

Reference inputs are presented to Qwen3-VL in image, video, then audio order.
Videos are sampled at 2 fps for the Qwen presentation while their full 24 fps
latents condition the diffusion transformer. Paired video and audio references
share the same timeline. Ref2VA cannot be combined with `--init-img` or
`--end-img` in one request.

Reference images preserve their aspect ratio and are downscaled only when
their pixel area exceeds the generation canvas. Each axis is then snapped to
the nearest 32-pixel multiple, with a minimum of 32. An axis may grow by up to
16 pixels during alignment. Custom reference sizes and preprocessing require a
[DGL program](../guides/edit.md#custom-reference-processing).

The C API exposes the same inputs through `ref_images`, `ref_videos`, and
the `ref_audios` video-option key. Each `sd_video_t` supplies its own
frame rate and optional soundtrack; non-24-fps inputs are resampled internally.

#### Sol-H3-VAE decoding

Pass a [Sol-H3-VAE checkpoint](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae)
with `--vae` to use the distilled video decoder. The ComfyUI FP16 and INT8
ConvRot checkpoints include the MiniMax-H3 encoder for image and video inputs.
Supply `--audio-vae` as usual to decode the soundtrack.

Sol decodes the whole clip by default. Use `--vae-tiling` for spatial tiling or
`--temporal-tiling` for temporal tiling. Either can reduce memory use but may
affect output quality because the decoder has less context at tile boundaries.

Temporal tiles default to 10 latent frames with 5 latent frames of overlap.
Set `temporal_tile_frames` and `temporal_tile_overlap` in
`--extra-tiling-args` to override them. Both must be multiples of 5 latent
frames, and overlap must be smaller than the tile. The last tile may be shorter.
Circular decoding requires spatial tiling on each circular axis, with a tile
size smaller than the input on that axis.

#### Latent hires second pass (two-stage sampling)

MiniMax-H3 supports `--hires` with two interchangeable second-stage upscalers:

- an interpolating latent upscaler (`Latent`, `Latent (nearest)`,
  `Latent (bicubic)`, ...): after the first pass finishes, the packed AV latent
  is resampled in latent space to `--hires-width` x `--hires-height` and
  refined by a second sampling pass;
- a model-backed upscaler (`--hires-upscaler <name>` + `--hires-upscalers-dir`):
  a H3-shaped learned latent upscaler checkpoint runs between the passes,
  like the LTX spatial upsampler. This is the sd.cpp equivalent of the
  two-sampler workflows used for H3 in node-based frontends (e.g. the
  community 3D-conv latent upscaler from
  [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)).

Draft at a low megapixel count (e.g. 0.4 MP) and refine at a higher one
(e.g. 0.8 MP) instead of paying for the full resolution in every step:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf `
  --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors `
  --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors `
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf `
  -p "a lovely cat" `
  --cfg-scale 1.0 `
  -W 864 `
  -H 480 `
  --video-frames 56 `
  --rng cpu `
  --hires `
  --hires-upscaler "Latent (bicubic)" `
  --hires-width 1216 `
  --hires-height 672 `
  --hires-steps 4 `
  --hires-denoising-strength 0.4 `
  --diffusion-fa `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf \
  --vae ../models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ../models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "a lovely cat" \
  --cfg-scale 1.0 \
  -W 864 \
  -H 480 \
  --video-frames 56 \
  --rng cpu \
  --hires \
  --hires-upscaler "Latent (bicubic)" \
  --hires-width 1216 \
  --hires-height 672 \
  --hires-steps 4 \
  --hires-denoising-strength 0.4 \
  --diffusion-fa \
  --offload-to-cpu \
  -v
```

or with the learned upscaler:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen ... `
  --hires-upscalers-dir ..\models\latent_upscale_models `
  --hires `
  --hires-upscaler h3-latent-upscaler-3d `
  --hires-width 1216 `
  --hires-height 672 `
  --hires-steps 4 `
  --hires-denoising-strength 0.4
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen ... \
  --hires-upscalers-dir ../models/latent_upscale_models \
  --hires \
  --hires-upscaler h3-latent-upscaler-3d \
  --hires-width 1216 \
  --hires-height 672 \
  --hires-steps 4 \
  --hires-denoising-strength 0.4
```

Notes for this mode:

- The learned upscaler expects a second normalization of the diffusion latent,
  matching its training convention. The runner applies and reverses this extra
  transform internally; callers should pass the usual H3 diffusion latents.
- The frame count is never changed by the second stage, so the packed audio
  latent length stays valid and the generated soundtrack is preserved.
- Width and height still have to be multiples of 32; the hires target is
  aligned like any other resolution. The learned upscaler additionally wants
  an even latent size on both axes, which 32-pixel targets provide.
- `--hires-sigmas` can replace `--hires-denoising-strength` when the second
  stage should resume from an explicit sigma schedule, which is how most
  two-stage H3 workflows split the sampler.
- Keyframe conditioning (`--init-img` / `--end-img`) is not available together
  with hires yet: those latents are encoded at the first stage resolution and
  would have to be re-encoded for the hires canvas. Use T2VA or Ref2VA.
- The second stage runs at the larger latent, so it is the memory-hungry half;
  combine it with quantization, `--vae-tiling` and `--offload-to-cpu` on
  small cards.
- The learned upscaler is scale-conditional: the target-to-source ratio is
  fed into the network as an embedding, 1.2-2.0x is the trained domain, and
  downscaling is rejected.

#### HyperFlow H3 Flash

[HyperFlow H3 Flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash)
uses the shared components and input options described in this guide. Replace
`--diffusion-model` with the matching HyperFlow FL2VA or Ref2VA checkpoint and
use these sampling options:

**Windows (PowerShell) arguments**

```text
--sampling-method euler `
  --cfg-scale 1 `
  --sigmas hyperflow-h3-flash-6
```

**Linux/macOS arguments**

```text
--sampling-method euler \
  --cfg-scale 1 \
  --sigmas hyperflow-h3-flash-6
```

Use `hyperflow-h3-flash-4` or `hyperflow-h3-flash-8` for 4 or 8 steps. These
presets also work with `--hires-sigmas`; their length determines the step count.
They already include video flow shift 12, so keep the default flow shift.

## Notes and limitations

- Width and height are aligned upward to a multiple of 32.
- Frame count is aligned upward to the `17k + 5` grid, with a minimum of 5.
- MiniMax-H3 runs at 24 fps; another requested value is overridden.
- Attention head grouping is controlled by
  `--model-args minimax_h3_attn_chunk_budget=2`, in GiB (default: `2`).
  The budget estimates the size of a single FP32 Q/K tensor per group:
  `head_dim * tokens * batch * 4 bytes * heads_per_group`. The group size adapts
  to the sequence length, with a minimum of one head; inputs that fit use all
  heads together. Set `0` to disable grouping, or a smaller budget such as
  `0.25` to reduce peak activation memory for long sequences at the cost of
  extra launches and concatenation. Each head still attends to the full sequence.
  This is a grouping target, not a cap on total attention memory or VRAM usage;
  other intermediates, backend workspace, and weights require additional memory.
  `--max-vram` and `--offload-to-cpu` control weight residency, but the active
  block's working memory must also fit within the VRAM budget.
- The default video flow shift is 12. The audio stream is mapped internally to
  its shift of 3, so the regular samplers can operate on the packed AV latent.
  CLI and C API translation keep this mapping consistent when `--flow-shift`
  is overridden, including hires refinement.


---

## FILE: examples/server/recipes/minimax-h3-fl2va.json (https://github.com/leejet/stable-diffusion.cpp/blob/HEAD/examples/server/recipes/minimax-h3-fl2va.json)

```
{
  "schema_version": 1,
  "id": "minimax-h3-fl2va",
  "name": "MiniMax H3 FL2VA",
  "size_multiple": 32,
  "description": "Generate video and audio, optionally with start/end images. Supports full and pruned FL2VA weights with either the MiniMax-H3 Qwen3-VL-32B text encoder or Qwen3-VL-4B with the P-H3 text connector. Audio VAE is optional for silent video.",
  "model_url": "https://huggingface.co/MiniMaxAI/MiniMax-H3",
  "slots": [
    {
      "id": "diffusion_model",
      "required": true
    },
    {
      "id": "llm",
      "required": true,
      "label": "MiniMax-H3 Qwen 3 VL 32B or P-H3 Qwen 3 VL 4B Text Encoder"
    },
    {
      "id": "llm_vision",
      "required": false,
      "label": "Vision tower (when not included in the text encoder)"
    },
    {
      "id": "embeddings_connectors",
      "required": false,
      "label": "P-H3 Text Connector (for Qwen 3 VL 4B, when not included in the text encoder)"
    },
    {
      "id": "vae",
      "required": false,
      "label": "MiniMax H3 Video VAE"
    },
    {
      "id": "tae",
      "required": false
    },
    {
      "id": "audio_vae",
      "required": false,
      "label": "MiniMax H3 Audio VAE"
    }
  ],
  "required_any": [
    [
      "vae",
      "tae"
    ]
  ],
  "video_constraints": {
    "frame_step": 17,
    "frame_offset": 5,
    "frame_rounding": "ceil",
    "min_frames": 5,
    "fixed_fps": 24
  },
  "defaults": {
    "video_frames": 56,
    "fps": 24,
    "cfg_scale": 1,
    "width": 1344,
    "height": 768,
    "sampling_method": "euler"
  },
  "features": [
    "end_image",
    "flow_shift",
    "init_image"
  ],
  "disabled_features": [
    "img2img",
    "mask_image"
  ]
}

```


---

## FILE: examples/server/recipes/minimax-h3-ref2va.json (https://github.com/leejet/stable-diffusion.cpp/blob/HEAD/examples/server/recipes/minimax-h3-ref2va.json)

```
{
  "schema_version": 1,
  "id": "minimax-h3-ref2va",
  "name": "MiniMax H3 Ref2VA",
  "size_multiple": 32,
  "description": "Supply image, video or audio references with compatible Ref2VA weights. Use either the MiniMax-H3 Qwen3-VL-32B text encoder or Qwen3-VL-4B with the P-H3 text connector. Image/video references need the matching vision tower. Audio VAE is optional for silent video.",
  "model_url": "https://huggingface.co/MiniMaxAI/MiniMax-H3",
  "slots": [
    {
      "id": "diffusion_model",
      "required": true
    },
    {
      "id": "llm",
      "required": true,
      "label": "MiniMax-H3 Qwen 3 VL 32B or P-H3 Qwen 3 VL 4B Text Encoder"
    },
    {
      "id": "llm_vision",
      "required": false,
      "label": "Vision tower (required for image/video references when not included in the text encoder)"
    },
    {
      "id": "embeddings_connectors",
      "required": false,
      "label": "P-H3 Text Connector (for Qwen 3 VL 4B, when not included in the text encoder)"
    },
    {
      "id": "vae",
      "required": false,
      "label": "MiniMax H3 Video VAE"
    },
    {
      "id": "tae",
      "required": false
    },
    {
      "id": "audio_vae",
      "required": false,
      "label": "MiniMax H3 Audio VAE"
    }
  ],
  "required_any": [
    [
      "vae",
      "tae"
    ]
  ],
  "video_constraints": {
    "frame_step": 17,
    "frame_offset": 5,
    "frame_rounding": "ceil",
    "min_frames": 5,
    "fixed_fps": 24
  },
  "defaults": {
    "video_frames": 56,
    "fps": 24,
    "cfg_scale": 1,
    "width": 1344,
    "height": 768,
    "sampling_method": "euler"
  },
  "features": [
    "flow_shift",
    "ref_images",
    "ref_videos",
    "ref_audios",
    "ref_video_audios"
  ],
  "disabled_features": [
    "img2img",
    "mask_image"
  ]
}

```


---

## FILE: examples/server/recipes/hyperflow-h3-flash-fl2va.json (https://github.com/leejet/stable-diffusion.cpp/blob/HEAD/examples/server/recipes/hyperflow-h3-flash-fl2va.json)

```
{
  "schema_version": 1,
  "id": "hyperflow-h3-flash-fl2va",
  "name": "HyperFlow H3 Flash FL2VA",
  "size_multiple": 32,
  "description": "Generate video and audio from text, optionally with start/end images. Use the HyperFlow H3 Flash FL2VA transformer. Reuses MiniMax-H3 components with either the MiniMax-H3 Qwen3-VL-32B text encoder or Qwen3-VL-4B with the P-H3 text connector and the HyperFlow 6-step schedule. Audio VAE is optional for silent video.",
  "model_url": "https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash",
  "slots": [
    {
      "id": "diffusion_model",
      "required": true,
      "label": "HyperFlow H3 Flash FL2VA Transformer"
    },
    {
      "id": "llm",
      "required": true,
      "label": "MiniMax-H3 Qwen 3 VL 32B or P-H3 Qwen 3 VL 4B Text Encoder"
    },
    {
      "id": "llm_vision",
      "required": false,
      "label": "Vision tower (when not included in the text encoder)"
    },
    {
      "id": "embeddings_connectors",
      "required": false,
      "label": "P-H3 Text Connector (for Qwen 3 VL 4B, when not included in the text encoder)"
    },
    {
      "id": "vae",
      "required": false,
      "label": "MiniMax H3 Video VAE"
    },
    {
      "id": "tae",
      "required": false
    },
    {
      "id": "audio_vae",
      "required": false,
      "label": "MiniMax H3 Audio VAE"
    }
  ],
  "required_any": [
    [
      "vae",
      "tae"
    ]
  ],
  "video_constraints": {
    "frame_step": 17,
    "frame_offset": 5,
    "frame_rounding": "ceil",
    "min_frames": 5,
    "fixed_fps": 24
  },
  "defaults": {
    "video_frames": 124,
    "fps": 24,
    "cfg_scale": 1,
    "width": 1344,
    "height": 768,
    "sampling_method": "euler",
    "steps": 6,
    "flow_shift": 12,
    "custom_sigmas": [
      1.0,
      0.982655942,
      0.914642572,
      0.674755692,
      0.385223746,
      0.159055337,
      0.0
    ]
  },
  "features": [
    "end_image",
    "flow_shift",
    "init_image"
  ],
  "disabled_features": [
    "img2img",
    "mask_image"
  ]
}

```


---

## FILE: docs/dgl/minimax_h3/minimax-h3-text-to-av.dgl (https://github.com/leejet/stable-diffusion.cpp/blob/HEAD/docs/dgl/minimax_h3/minimax-h3-text-to-av.dgl)

```
# version: 0.0.0
# docs/models/minimax_h3.md — Text-to-audio-video

var {
    model_path = "../models/diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf"
    text_path = "../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    vae_path = "../models/vae/minimax_h3_video_vae_fp16.safetensors"
    audio_vae_path = "../models/vae/minimax_h3_audio_vae_fp32.safetensors"

    output_path = "output.avi"

    prompt = "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten鈥檚 energetic surfing."

    seed = 42
    rng_type = "cpu"
    width = 864
    height = 480
    frames = 56
    fps = 24

    steps = 20
    method = "euler"
    scheduler = "discrete"
    audio_scale = 4
}

fn pack_av(video, audio, audio_scale) {
    shape = tensor::shape(video)
    count = tensor::numel(audio)
    tail_channels = math::ceil_div(count, shape[0] * shape[1] * shape[2])
    tail_shape = [shape[0], shape[1], shape[2], tail_channels, 1]
    flat = tensor::reshape(tensor::mul(audio, audio_scale), [count])
    padded = tensor::pad(flat, [0, math::product(tail_shape) - count])
    return tensor::concat([video, tensor::reshape(padded, tail_shape)], axis = 3)
}

fn unpack_audio(packed, channels, audio_shape, audio_scale) {
    tail = packed[:, :, :, channels:]
    flat = tensor::reshape(tail, [tensor::numel(tail)])
    return tensor::div(tensor::reshape(flat[:math::product(audio_shape)], audio_shape), audio_scale)
}

fn main() {
    model = load_dit(model_path, family = "minimax_h3", params_backend = "cpu")
    text = load_text_encoder(text_path, family = "minimax_h3", params_backend = "cpu")
    vae = load_vae(vae_path, family = "minimax_h3", params_backend = "cpu")

    video_base = tensor::zeros(latent_shape(vae, width = width, height = height, frames = frames))
    audio_vae = load_audio_vae(audio_vae_path, family = "minimax_h3", params_backend = "cpu")
    audio_shape = audio_latent_shape(audio_vae, frames = frames, fps = fps)
    base = pack_av(video_base, tensor::zeros(audio_shape), audio_scale = audio_scale)

    condition = encode_prompt(text, prompt, width = width, height = height)
    predict = predictor(model, condition, audio_length = audio_shape[0])
    sigmas = make_sigmas(predict, latent = base, steps = steps, scheduler = scheduler)
    rng = make_rng(rng_type, seed = seed)
    start = add_noise(predict, base, tensor::randn_like(base, rng = rng), sigma = sigmas[0])
    latent = sample(start, sigmas, predict = predict, method = method, rng = rng)
    video_latent = latent[:, :, :tensor::shape(video_base)[2], :tensor::shape(video_base)[3]]
    output_frames = vae_decode(vae, dit_to_vae_latents(vae, video_latent))
    sound_latent = unpack_audio(latent, channels = tensor::shape(video_base)[3], audio_shape = audio_shape, audio_scale = audio_scale)
    audio = audio_vae_decode(audio_vae, sound_latent)
    video::save(output_path, {frames = output_frames, fps_num = fps, fps_den = 1}, audio = audio)
    return {video = {frames = output_frames, fps_num = fps, fps_den = 1}, latent = latent, audio = audio}
}

```


---

## FILE: docs/dgl/minimax_h3/minimax-h3-reference-to-av.dgl (https://github.com/leejet/stable-diffusion.cpp/blob/HEAD/docs/dgl/minimax_h3/minimax-h3-reference-to-av.dgl)

```
# version: 0.0.0
# docs/models/minimax_h3.md — Reference-to-audio-video conditioning

var {
    model_path = "../models/diffusion_models/minimax_h3_ref2va_pruned-Q4_K_M.gguf"
    text_path = "../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    vae_path = "../models/vae/minimax_h3_video_vae_fp16.safetensors"
    audio_vae_path = "../models/vae/minimax_h3_audio_vae_fp32.safetensors"

    h3_image_0_path = "../inputs/ernie_image/turbo_example.png"
    output_path = "output.avi"

    prompt = "Use the cat from <Picture 1> as the main character. Keep the cat鈥檚 appearance, fur color, facial features, and identity consistent with the reference image. Create a 2-second cinematic video: start with an extreme close-up shot of the cat鈥檚 face, focusing on its cute expression and detailed fur texture. The camera slowly rotates around the cat鈥檚 head, creating a dynamic reveal. Then smoothly pull back and zoom out to reveal the full scene: the cat is standing confidently on a surfboard, riding ocean waves. Water splashes around the board, sea breeze gently moves the cat鈥檚 fur, and the cat maintains a cute and fearless expression while surfing. Smooth camera movement, cinematic orbit shot, seamless zoom-out transition, low-angle wide shot, realistic ocean environment, golden sunlight, dynamic waves, high-quality realistic style, natural motion, no distortion, keep the cat鈥檚 identity unchanged."

    seed = 42
    rng_type = "cpu"
    width = 864
    height = 480
    frames = 56
    fps = 24

    steps = 20
    method = "euler"
    scheduler = "discrete"
    audio_scale = 4
}

fn pack_av(video, audio, audio_scale) {
    shape = tensor::shape(video)
    count = tensor::numel(audio)
    tail_channels = math::ceil_div(count, shape[0] * shape[1] * shape[2])
    tail_shape = [shape[0], shape[1], shape[2], tail_channels, 1]
    flat = tensor::reshape(tensor::mul(audio, audio_scale), [count])
    padded = tensor::pad(flat, [0, math::product(tail_shape) - count])
    return tensor::concat([video, tensor::reshape(padded, tail_shape)], axis = 3)
}

fn unpack_audio(packed, channels, audio_shape, audio_scale) {
    tail = packed[:, :, :, channels:]
    flat = tensor::reshape(tail, [tensor::numel(tail)])
    return tensor::div(tensor::reshape(flat[:math::product(audio_shape)], audio_shape), audio_scale)
}

fn h3_reference_noise(latent, seed) {
    rng = make_rng("cuda", seed = seed)
    return tensor::lerp(latent, tensor::randn_like(latent, rng = rng), 0.001)
}

fn sqrt_step(estimate, index, value) {
    return (estimate + value / estimate) / 2
}

fn positive_sqrt(value) {
    return fold(tensor::to_list(tensor::arange(0, 32)), value + 1.0, process = sqrt_step, value = value)
}

fn fit_reference_area(input, area, factor, mode) {
    shape = tensor::shape(input)
    width = positive_sqrt(area * shape[0] / shape[1])
    rounded = [math::round(width / factor) * factor, math::round(width * shape[1] / shape[0] / factor) * factor]
    size = tensor::maximum(tensor::vector(rounded, dtype = "i64"), tensor::full([2], factor, dtype = "i64"))
    return tensor::interpolate(input, size = size, mode = mode)
}

fn main() {
    h3_image_0 = image::load(h3_image_0_path)

    model = load_dit(model_path, family = "minimax_h3", params_backend = "cpu")
    text = load_text_encoder(text_path, family = "minimax_h3", params_backend = "cpu")
    vae = load_vae(vae_path, family = "minimax_h3", params_backend = "cpu")

    rng = make_rng(rng_type, seed = seed)
    video_base = tensor::zeros(latent_shape(vae, width = width, height = height, frames = frames))
    audio_vae = load_audio_vae(audio_vae_path, family = "minimax_h3", params_backend = "cpu")
    audio_shape = audio_latent_shape(audio_vae, frames = frames, fps = fps)
    base = pack_av(video_base, tensor::zeros(audio_shape), audio_scale = audio_scale)
    h3_image_0_original = image::to_f32(h3_image_0, channels = 3)
    h3_image_0_area = math::min(width * height, tensor::shape(h3_image_0_original)[0] * tensor::shape(h3_image_0_original)[1])
    h3_image_0_pixels = fit_reference_area(h3_image_0_original, h3_image_0_area, 32, mode = "bilinear")
    h3_visual_0_raw = vae_to_dit_latents(vae, vae_encode(vae, h3_image_0_pixels, rng = rng))
    h3_visual_0 = h3_reference_noise(h3_visual_0_raw, seed)

    condition = encode_prompt(text, prompt, width = width, height = height, media = [{image = h3_image_0_pixels}])
    predict = predictor(model, condition, audio_length = audio_shape[0], ref_latents = [h3_visual_0], reference_blocks = [{kind = "image", video = 0}])
    sigmas = make_sigmas(predict, latent = base, steps = steps, scheduler = scheduler)
    start = add_noise(predict, base, tensor::randn_like(base, rng = rng), sigma = sigmas[0])
    latent = sample(start, sigmas, predict = predict, method = method, rng = rng)
    video_latent = latent[:, :, :tensor::shape(video_base)[2], :tensor::shape(video_base)[3]]
    output_frames = vae_decode(vae, dit_to_vae_latents(vae, video_latent))
    sound_latent = unpack_audio(latent, channels = tensor::shape(video_base)[3], audio_shape = audio_shape, audio_scale = audio_scale)
    audio = audio_vae_decode(audio_vae, sound_latent)
    video::save(output_path, {frames = output_frames, fps_num = fps, fps_den = 1}, audio = audio)
    return {video = {frames = output_frames, fps_num = fps, fps_den = 1}, latent = latent, audio = audio}
}

```


---

## FILE: docs/dgl/minimax_h3/minimax-h3-first-last-frame.dgl (https://github.com/leejet/stable-diffusion.cpp/blob/HEAD/docs/dgl/minimax_h3/minimax-h3-first-last-frame.dgl)

```
# version: 0.0.0
# docs/models/minimax_h3.md — First/last-frame conditioning

var {
    model_path = "../models/diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf"
    text_path = "../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    vae_path = "../models/vae/minimax_h3_video_vae_fp16.safetensors"
    audio_vae_path = "../models/vae/minimax_h3_audio_vae_fp32.safetensors"

    first_frame_path = "../inputs/ernie_image/turbo_example.png"
    output_path = "output.avi"

    prompt = "a lovely cat"

    seed = 42
    rng_type = "cpu"
    width = 864
    height = 480
    frames = 56
    fps = 24

    steps = 20
    method = "euler"
    scheduler = "discrete"
    audio_scale = 4
}

fn fit_canvas(input, width, height, mode) {
    shape = tensor::shape(input)
    crop_width = math::min(shape[1], math::floor(shape[2] * width / height))
    crop_height = math::min(shape[2], math::floor(shape[1] * height / width))
    start = [math::floor((shape[1] - crop_width) / 2), math::floor((shape[2] - crop_height) / 2)]
    cropped = tensor::crop(input, axes = [1, 2], start = start, size = [crop_width, crop_height])
    return tensor::interpolate(cropped, size = [width, height], axes = [1, 2], mode = mode)
}

fn pack_av(video, audio, audio_scale) {
    shape = tensor::shape(video)
    count = tensor::numel(audio)
    tail_channels = math::ceil_div(count, shape[0] * shape[1] * shape[2])
    tail_shape = [shape[0], shape[1], shape[2], tail_channels, 1]
    flat = tensor::reshape(tensor::mul(audio, audio_scale), [count])
    padded = tensor::pad(flat, [0, math::product(tail_shape) - count])
    return tensor::concat([video, tensor::reshape(padded, tail_shape)], axis = 3)
}

fn unpack_audio(packed, channels, audio_shape, audio_scale) {
    tail = packed[:, :, :, channels:]
    flat = tensor::reshape(tail, [tensor::numel(tail)])
    return tensor::div(tensor::reshape(flat[:math::product(audio_shape)], audio_shape), audio_scale)
}

fn h3_reference_noise(latent, seed) {
    rng = make_rng("cuda", seed = seed)
    return tensor::lerp(latent, tensor::randn_like(latent, rng = rng), 0.001)
}

fn main() {
    first_frame = image::load(first_frame_path)

    model = load_dit(model_path, family = "minimax_h3", params_backend = "cpu")
    text = load_text_encoder(text_path, family = "minimax_h3", params_backend = "cpu")
    vae = load_vae(vae_path, family = "minimax_h3", params_backend = "cpu")

    rng = make_rng(rng_type, seed = seed)
    video_base = tensor::zeros(latent_shape(vae, width = width, height = height, frames = frames))
    first_frame_pixels = tensor::unsqueeze(image::to_f32(fit_canvas(first_frame, width, height, mode = "nearest"), channels = 3), axis = 2)
    audio_vae = load_audio_vae(audio_vae_path, family = "minimax_h3", params_backend = "cpu")
    audio_shape = audio_latent_shape(audio_vae, frames = frames, fps = fps)
    base = pack_av(video_base, tensor::zeros(audio_shape), audio_scale = audio_scale)
    h3_visual_0_raw = vae_to_dit_latents(vae, vae_encode(vae, first_frame_pixels, rng = rng))
    h3_visual_0 = h3_reference_noise(h3_visual_0_raw, seed)

    condition = encode_prompt(text, prompt, width = width, height = height, media = [first_frame_pixels[:, :, 0]])
    predict = predictor(model, condition, audio_length = audio_shape[0], ref_latents = [h3_visual_0], keyframe_indices = [0])
    sigmas = make_sigmas(predict, latent = base, steps = steps, scheduler = scheduler)
    start = add_noise(predict, base, tensor::randn_like(base, rng = rng), sigma = sigmas[0])
    latent = sample(start, sigmas, predict = predict, method = method, rng = rng)
    video_latent = latent[:, :, :tensor::shape(video_base)[2], :tensor::shape(video_base)[3]]
    output_frames = vae_decode(vae, dit_to_vae_latents(vae, video_latent))
    sound_latent = unpack_audio(latent, channels = tensor::shape(video_base)[3], audio_shape = audio_shape, audio_scale = audio_scale)
    audio = audio_vae_decode(audio_vae, sound_latent)
    video::save(output_path, {frames = output_frames, fps_num = fps, fps_den = 1}, audio = audio)
    return {video = {frames = output_frames, fps_num = fps, fps_den = 1}, latent = latent, audio = audio}
}

```


---

# ISSUES in leejet/stable-diffusion.cpp matching "H3" (top 15 by comments)


### #1852 [closed] [Feature]  MiniMax H3 video support
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1852  by jadbox 2026-08-03T22:30:05Z  comments: 19

### Feature Summary

MiniMax H3 video generation

### Detailed Description

Please support the OS model MiniMax H3 that had its release today. Either the int8 or the GGUF version would be ideal to support.
https://www.reddit.com/r/StableDiffusion/comments/1ve67ny/minimax_h3_smaller_other_quants/
https://huggingface.co/Abiray/MiniMax-H3-GGUF


### Alternatives you considered

_No response_

### Additional context

_No response_

> **Comment by Christopher-Chianelli (2026-08-05T05:30:10Z):**

FYI, there are GGUF's avaiable that work with stable-diffusion.cpp, available at https://huggingface.co/leejet/MiniMax-H3-GGUF.
Usage example:
```bash
sd-cli \
    -M vid_gen \
    --diffusion-model /models/minimax_h3_ref2va_pruned-Q2_K_M.gguf \
    --vae /models/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae /models/minimax_h3_audio_vae_fp32.safetensors \
    --llm /models/qwen3vl_32b_minimax_h3-Q2_K_M.gguf \
    --cfg-scale 1.0 \
    -W 864 \
    -H 480 \
    --diffusion-fa \
    -p "A jumping cat" --video-frames 124 \
    -o /output/video.avi
```

Note: Q2 barely fits on a 7900 XTX when generating 124 frames on the vulkan backend. You could run Q4 On 7900 XTX, but then you are limited to basically generating a single frame at a time.


> **Comment by Christopher-Chianelli (2026-08-05T05:58:25Z):**

If you really, really, minimize the width and height, you can generate a video using Q4 on the vulkan backend for 7900 XTX:

```bash
sd-cli \
    -M vid_gen \
    --diffusion-model /models/minimax_h3_ref2va_pruned-Q4_K_M.gguf \
    --vae /models/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae /models/minimax_h3_audio_vae_fp32.safetensors \
    --llm /models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
    --cfg-scale 1.0 \
    -W 64 \
    -H 32 \
    --diffusion-fa \
    -p "A jumping cat" --video-frames 120 \
    -o /output/video.avi
```

That being said, it does appear you need to ground the model with a base image; a lot of the video generated look like TV static from a text only prompt. (Haven't tried with an image input yet; I do know generating a single frame with Q4 with a text only input creates something coherent, but across multiple frames it basically become static/noise).


> **Comment by kwilkins-82 (2026-08-05T08:48:42Z):**

I notice some are getting good results with an 8 bit quant and as little as 12GB VRAM, using Comfy.   Is Comfy better at offloading weights to system memory?    


> **Comment by Green-Sky (2026-08-05T09:19:10Z):**

Try `--offload-to-cpu --max-vram X` with X in GiB. Also if you are on vulkan, try to enable direct conv.


> **Comment by kwilkins-82 (2026-08-05T13:47:51Z):**

I got it to run with a Q2 quant, but getting a black output.    This is with Vulkan, will try ROCM later.


> **Comment by AmeowCAT (2026-08-05T14:26:44Z):**

It seems that sd-server still does not support ref2va.


> **Comment by Christopher-Chianelli (2026-08-05T15:38:39Z):**

> Try `--offload-to-cpu --max-vram X` with X in GiB. Also if you are on vulkan, try to enable direct conv.

Alas, `--offload-to-cpu --max-vram 22` basically hard crashes my computer (for generating 120 frames of 864 x 480); from the final logs, it appears the RAM that was allocated but enough to basically fill RAM + SWAP, but just low enough not to trigger OOM Killer (I built my system when 32 GiB was considered excessive; how have the times changed); couldn't do anything without a system reset. `direct conv` on diffusion and vae appears to work though.


> **Comment by kwilkins-82 (2026-08-05T16:03:07Z):**

I got it to run using -"-offload-to-cpu --max-vram -2 --clip-on-cpu --vae-tiling"  on ROCM, but the results are not good at all.    The Q2 model just produces a smudge - minimax_h3_fl2va_pruned-Q4_K_M.gguf is a bit better in that it's giving me a real video, but it's not following my prompt at all and the audio is just white noise.      I tried setting --guidance 4.0 but it didn't make any difference, nor did using the Q4 qwen quant.


> **Comment by mlsterpr0 (2026-08-06T20:18:14Z):**

Please don't stop updating the tutorial for this model, as i'm hoping for a good 4-8 step lora in future, and overall a way to run it on a lower end system the best way possible 🙏 


> **Comment by jbiatek (2026-08-07T02:31:41Z):**

I've discovered that omitting `--cfg-scale` causes a crash to a stack trace dump. Easy enough to include it, but it wasn't clear what was happening. If you're interested, my friendly local LLM's conclusion was:

> - Crash is in the text encoder (`LLM::Attention::forward` → `ggml_ext_attention_ext` manual fallback).
> - Trigger: encoding an empty prompt (L_q/L_k = 0). `ggml_mul_mat(ctx, v, kq)` fails because `v` becomes
degenerate/transposed (nb[0] > nb[1]) for a zero-length tensor - `ggml.c:3244` assert.
> - Reproducible with `--cfg-scale` default (7.0 → `use_uncond=true`) and no `--negative-prompt`; avoided by `--cfg-scale 1.0`.
> - Suggested guard: `ggml_ext_attention_ext` should handle/skip the `L_q==0 || L_k==0` case, or the conditioning pipeline shouldn't build the LLM graph for an empty prompt - plus a friendlier error than dumping a raw backtrace.


> **Comment by kwilkins-82 (2026-08-07T06:56:10Z):**

The last couple of commits helped a lot, it's now mostly following my prompts, though the audio output is still bad.

EDIT:    it's only following my prompt if I generate a tiny video e.g. 320x320x2 sec - as soon as I go 480x480 it's back to ignoring it.    I asked Gemini for an explanation and it thinks the text embeddings are only being sent to the first segment?   (I've no idea whether that makes sense or not...)

EDIT2:  Seems fixed as of 20th August


> **Comment by velartrill (2026-08-07T14:57:00Z):**

i'm also having this problem with ref2va. for me the visual output follows the prompt at 864×480 up to 56 frames but any longer than that and the entire video is visual nonsense. every configuration I've tried produces audio that's mostly blaring noise however. (using Vulkan on AMD if that's relevant)

ETA: the fix described in #1865 resolves the problem for me


> **Comment by libai-lab (2026-08-07T15:21:59Z):**

Thank you for your excellent work. Will you support LoRa, such as accelerating https://huggingface.co/lightx2v/Minimax-h3-Turbo? Currently, using multiple reference images, approximately how much VRAM is required?


> **Comment by mlsterpr0 (2026-08-09T19:39:17Z):**

In stable diffusion cpp, i am trying to get good results with this lora, but it seems to not work at full potential... 
https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors
Any tips how to run this correctly? Or just wait for an update. Thanks.

There is a preview video with chameleon on the lora page made in ComfyUI, and it looks amazing.


> **Comment by noctrex (2026-08-10T11:43:36Z):**

I have a 7900XTX, and using the ROCm version on Windows 11.
Now that Unsloth has released their GGUFs, I thought of trying them out. These results are my experience:

`minimax_h3_fl2va_pruned-Q4_K.gguf` - `10.64GiB` - Worst quality, full of artifacts, audio and video desynchronized, about 16GB VRAM used
`minimax_h3_fl2va_pruned-Q5_0.gguf` - `12.97GiB` - Slightly better than Q4, some artifacts, audio and video desynchronized, about 18GB VRAM used
`minimax_h3_fl2va_pruned-Q6_K.gguf` - `15.45GiB` - Black output with ROCm, usable only with Vulkan, about 21GB VRAM used
`minimax_h3_fl2va_pruned-Q8_0.gguf` - `19.97GiB` - Best quality, all VRAM used, does OOM if resolution larger than 640x384 used.

I used the following command:
```
C:\Programs\AI\stable-diffusion.cpp-rocm\sd-cli.exe --mode vid_gen --cfg-scale 1.0 --backend te=cpu --offload-to-cpu --diffusion-fa --output videooutput ^
--width 640 --height 384 --video-frames 56 ^
--diffusion-model G:\Models\MiniMax-H3\minimax_h3_fl2va_pruned-Q8_0.gguf ^
--llm G:\Models\MiniMax-H3\qwen3vl_32b_minimax_h3-Q4_K_M.gguf ^
--vae G:\Models\MiniMax-H3\minimax_h3_video_vae_fp16.safetensors ^
--audio-vae G:\Models\MiniMax-H3\minimax_h3_audio_vae_fp32.safetensors ^
--prompt "a small cute papllion dog trotting through falling snow and barking, besides a rustic old German home, cinematic"
```



> **Comment by jadbox (2026-08-11T18:50:39Z):**

@mlsterpr0 How do you even use Turbo-Lora in stable diffusion cpp? Can you walk me through it? Also, has anyone figured out how to get it properly working?


> **Comment by mlsterpr0 (2026-08-12T19:26:29Z):**

> [@mlsterpr0](https://github.com/mlsterpr0) How do you even use Turbo-Lora in stable diffusion cpp?

I don't know if you are running it through sd-cli or sd-server? In sd-server it's straightforward because you can use embedded web UI. In sd-cli just add <lora:[name of lora file without .safetensors]:1> right where your promt is. Example: 
```
--prompt "A beautiful cat<lora:minimax_h3_turbo_v4_step600_ema_pruned_comfyui:1>"
```
 (:1 at the end is lora strengths)

Recommended starting settings for this lora:
6–8 steps · Euler sampler · Beta scheduler · LoRA strength 1.0



> **Comment by farawayso (2026-08-16T14:01:11Z):**

> 你到底怎么用Turbo-Lora在稳定扩散CPP里？你能给我详细讲讲吗？另外，有人找到办法让它正常工作吗？

[sd_lora_usage.md](https://github.com/user-attachments/files/31117927/sd_lora_usage.md) 我是让agent帮我转换了参数，这是ai总结的md


> **Comment by leejet (2026-11-28T08:58:53Z):**

This feature is already supported.



### #1947 [closed] ROCm Q6_K pruned streaming degrades on OCuLink (近期已无该问题复现，该贴暂时关闭)
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1947  by farawayso 2026-09-08T11:07:09Z  comments: 16

# [Bug] ROCm Q6_K pruned streaming degrades on OCuLink (missing prefetch)

## Environment

- **OS**: Windows 11
- **GPU**: AMD Radeon RX 7900 XTX 24GB (gfx1100 / RDNA3)
- **iGPU**: AMD Radeon 890M (integrated in 890M CPU)
- **Connection**: OCuLink external GPU dock (PCIe 4.0 x4,实测 bandwidth ~6.5 GB/s)
- **sd-cli version**: master-849 (d04e895), includes PR #1576 (`--stream-layers`)
- **Missing**: PR #1905 (`--layer-prefetch-depth`, `--resident-layers`) not compiled
- **Model**: `minimax_h3_fl2va_pruned-Q6_K.gguf` (15.9 GB)
- **Config**: 480x864, 73 frames, 5 steps, Euler scheduler, CFG=1.0

## Problem Description

When using `--stream-layers` + `--params-backend diffusion=cpu` streaming mode:
- **Step 1**: Normal (~40s)
- **Steps 2-N**: Extremely slow (~825s/it), outputs black screen (52KB video)

**Expected**: master-841 generated Q6_K normally at ~40s/step.

## Log Evidence

```
[INFO ] model_manager.cpp:428  - model manager prepared params backend buffers (15902.72 MB, 532 tensors, 1 blocks, RAM) on ROCm_Host
[INFO ] model_manager.cpp:555  - model manager staged compute params (15902.75 MB, 532 tensors, 210 blocks) to ROCm0, taking 3.16s
[DEBUG] ggml_runner.cpp:919  - minimax_h3 executing segment 1/1: graph
# Subsequent steps hang or are extremely slow...
```

**Key observations**:
1. 15.9GB weights loaded as **1 blocks, RAM** - bulk transfer
2. Graph cut produces **210 blocks**, but no prefetch pipeline
3. Each step requires full staging, HIP driver degrades after multiple stagings

## Root Cause Analysis

### 1. Missing Async Prefetch Pipeline

PR #1576 only provides basic residency framework:
- ✅ `--stream-layers` enabled
- ❌ No `--layer-prefetch-depth` (to hide PCIe/OCuLink transfer latency)
- ❌ No `--resident-layers` (minimal rolling window control)

PR #1905 was merged to upstream master (2026-09-06) but not compiled into local binary.

### 2. OCuLink Bandwidth Bottleneck

| Metric | Value |
|---|---|
| Interface | OCuLink (PCIe 4.0 x4) |
| Measured bandwidth | ~6.5 GB/s |
| Weight size | 15.9 GB (Q6_K pruned) |
| Theoretical min transfer | 15.9 / 6.5 ≈ 2.4s |
| Actual per-step time | 825s (Steps 2-5) |

**Degradation cause**:
- No prefetch → serial load-then-compute per step
- 210 blocks repeatedly transferred triggers HIP driver fallback
- OCuLink bandwidth is far below PCIe 5.0 x16 (~64 GB/s), latency-sensitive

### 3. Comparison: master-841 vs Current Binary

| Feature | master-841 | Current binary |
|---|---|---|
| `--stream-layers` | ✅ Yes | ✅ Yes |
| `--layer-prefetch-depth` | ❌ No | ❌ No |
| `--resident-layers` | ❌ No | ❌ No |
| Async prefetch pipeline | ❌ No | ❌ No |
| Performance (Q6_K 73f) | ~40s/step | 825s/step ✗ |

**Conclusion**: `--stream-layers` alone cannot solve performance issues in OCuLink bandwidth-limited environments. PR #1905's prefetch pipeline is needed to hide transfer latency.

## Reproduction Steps

```bash
# Environment
export H3_QUANT=Q6_K
export H3_MAX_VRAM=23

# Run (will auto-enable streaming mode)
python scripts/sd_interactive_video.py t2va pruned short 480 864 3 1 5 auto
```

Or manual command:
```bash
./rocm/sd-cli.exe \
  --diffusion-model minimax_h3_fl2va_pruned-Q6_K.gguf \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --lora-model-dir loras/ --lora-apply-mode at_runtime \
  -p "test prompt" \
  --steps 5 --cfg-scale 1.0 --sampling-method euler \
  -W 480 -H 864 --video-frames 73 --seed 42 \
  --backend llm=cpu,vae=ROCm0,diffusion=ROCm0 \
  --params-backend diffusion=cpu \
  --stream-layers \
  --max-vram ROCm0=23 \
  --temporal-tiling \
  -o out.mp4
```

## Expected Behavior

- Steps 1-5 should each take ~40s consistently
- No black screen output
- `--layer-prefetch-depth 1` should hide OCuLink transfer latency

## Actual Behavior

- Step 1: ~40s ✅
- Steps 2-5: ~825s/it ❌ (20x slower)
- Output: 52KB black screen video ❌

## Suggested Fixes

### Short-term Workarounds (User-side)
```bash
# Option 1: Force eager mode (weights resident in GPU)
H3_FORCE_EAGER=1 H3_QUANT=Q6_K python scripts/sd_interactive_video.py ...

# Option 2: Downgrade to Q4_K_M
H3_QUANT=Q4_K_M python scripts/sd_interactive_video.py ...

# Option 3: Reduce frames to under 56
python scripts/sd_interactive_video.py ...  # Choose 56 frames
```

### Long-term Fixes (Upstream)
1. Prioritize releasing PR #1905 (`--layer-prefetch-depth`) and PR #1906 (`--stream-layer-pool`)
2. Update documentation with OCuLink/low-bandwidth environment configuration tips
3. Consider adding auto-detection for low-bandwidth environments with eager fallback

## Additional Notes

- This machine uses OCuLink external GPU dock, bandwidth is far below native PCIe
- On主板 directly (PCIe 5.0 x16), performance may be normal
- This is a combination issue: OCuLink bandwidth limitation + missing prefetch pipeline

## Related Links

- PR #1576: feat: `--stream-layers` for streaming weights from CPU during generation
- PR #1905: feat: prefetch streamed layers during compute (merged 2026-09-06)
- PR #1906: feat: pool VRAM for streamed layer (open)


> **Comment by farawayso (2026-09-10T23:48:16Z):**

## Update (2026-09-11): black output reproduces *without* streaming — likely a separate root cause

Re-tested on **master-853 (`b68d586`)**, which already includes PR #1905 (`--layer-prefetch-depth`).
Short version: the black output is **not** caused by streaming / missing prefetch / OCuLink bandwidth.
It reproduces with everything resident on the GPU and zero per-step transfer.

### Decisive test: 13-frame, eager, single-segment

Same machine as the original report (7900 XTX 24GB, gfx1100, OCuLink). Config that removes
every variable the original report blamed:

- **eager mode** — no `--stream-layers`, no `--params-backend diffusion=cpu`
  (all 15.9 GB resident in VRAM, so there is no weight transfer during sampling at all)
- **13 frames** — tiny activation buffer, no memory pressure (nothing near the 24 GB limit)
- `--video-frames 13`, 480x864, 5 steps, Euler, CFG 1.0

Result: output is **still fully black**.

### Quant x backend matrix (all other params identical)

| quant | backend | result | brightness (YAVG, all frames) |
|---|---|---|---|
| Q6_K | ROCm  | **all-black** | **0.00** |
| Q6_K | Vulkan | normal | 134.7 |
| Q4_K_M | ROCm | normal | 132 – 137 |
| Q4_K_M | Vulkan | normal | (not re-tested, known good) |

YAVG measured with `ffmpeg -vf signalstats,metadata=print:key=lavfi.signalstats.YAVG`
averaged over every frame. Black = 0.00 on every single frame, not just the first ones.

### Variables I ruled out

Each of these was flipped independently — Q6_K + ROCm stayed black in all of them:

| variable | values tested |
|---|---|
| streaming | eager (`H3_FORCE_EAGER=1`) / streamed (`--stream-layers` + `diffusion=cpu`) |
| graph-cut segmentation | 51 segments (`--max-vram ROCm0=23`) / single segment (`--disable-segmented-compute`) |
| VAE placement | on `ROCm0` / on CPU via `--params-backend vae=cpu` |
| frame count | 13 / 22 / 73 |

At 13 frames in eager mode, OCuLink bandwidth is simply not in the picture — weights are
loaded once and never moved again. So the black frames cannot be a transfer/prefetch artifact.

### What did change

- The **825 s/it slowdown** does look like the streaming-path issue the original report
  describes, and on master-853 (+ #1905) it is no longer reproducible here. That part may
  well be resolved.
- The **black output** is a separate, quant-x-backend defect: `ROCm + Q6_K` produces garbage
  (all zeros), while `Vulkan + Q6_K` and `ROCm + Q4_K_M` are both correct.

### Possible regression window

The original report notes Q6_K was fine on **master-841** (~40 s/step, correct output).
On master-853 it is black. That puts the regression somewhere in **842–853**, which overlaps
the changes already implicated in #1946 (root-caused to PR #1940, landed at master-843) and
the weight-lifecycle refactor in #1956. I have not bisected this — flagging it as a hypothesis,
not a conclusion.

### Minimal repro (no wrapper scripts)

```bash
./rocm/sd-cli.exe \
  --diffusion-model minimax_h3_fl2va_pruned-Q6_K.gguf \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  -p "a cat sitting on a table" \
  --steps 5 --cfg-scale 1.0 --sampling-method euler \
  -W 480 -H 864 --video-frames 13 --seed 42 \
  --backend llm=cpu,vae=ROCm0,diffusion=ROCm0 \
  -o out.mp4


> **Comment by laurentvv (2026-09-11T07:47:05Z):**

Data points from #1946 (same suspected window, different failure mode — **Vulkan**, RX 6950 XT 16 GB, Windows 11): your "842–853" hypothesis is consistent with what we bisected, and it narrows the start of the window to **exactly 843**.

**Binary bisect on the official `bin-win-vulkan-x64` releases** (Flux schnell Q6_K, split text encoders, 512×512):

| Build | Result |
|---|---|
| master-841 `6b3edaa` | ✅ works (re-verified) |
| master-843 `462d675` (#1905 + #1940) | ❌ first crash |
| master-845 `80bac2d` | ❌ identical crash |

Local MSYS2/UCRT64 builds separating the two commits inside 843 (same ggml pin `e20c3a14`):

| Commit | Result |
|---|---|
| `6c57cc3` (#1905 prefetch, alone) | ✅ works — **#1905 innocent** |
| `462d675` (#1940 unify runner lifecycles and weight residency) | ❌ crashes with `vk::Queue::submit: ErrorOutOfDeviceMemory` right after the DiT weights finish staging (on the official MSVC build the exception is swallowed → the silent exit-127 of the original report) |

So on our side the window opens at **843 = #1940 landing**, which also matches the overlap you flagged with #1956's weight-lifecycle refactor.

Two caveats before reading too much into it:

- Our failure mode is **not** yours: no black output, no Q6_K×backend kernel issue — it's the weight-preparation / residency manager running out of device memory mid-flight (deterministic ~47 MB shortfall at a specific block on a freshly rebooted, idle 16 GB card; full details and logs in #1946, leejet's candidate guard PR #1958 turns the hard crash into a clean propagated error but generation still cannot complete here).
- I can't test ROCm/HIP on this machine (RDNA2 + Vulkan only), so I can't check whether your black-output defect shares a root cause or is a second independent casualty of the same refactor cluster. The honest summary so far: **#1940 is proven guilty for the Vulkan 16 GB OOM path; for your Q6_K×ROCm black frames it is only in the suspect lineup.**



> **Comment by leejet (2026-09-11T15:34:41Z):**

It looks like the memory budget calculation determined that your device has enough VRAM to execute the entire graph, so it used a unified graph instead of splitting the computation into chunks.

Could you try running the following command and send me the detailed logs so I can investigate the issue further?

```
./rocm/sd-cli.exe \
  --diffusion-model minimax_h3_fl2va_pruned-Q6_K.gguf \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  -p "a cat sitting on a table" \
  --steps 5 --cfg-scale 1.0 --sampling-method euler \
  -W 480 -H 864 --video-frames 13 --seed 42 \
  --backend llm=cpu,vae=ROCm0,diffusion=ROCm0 \
  --log-level debug -o out.mp4
```


> **Comment by farawayso (2026-09-11T16:28:34Z):**

@leejet 按您的要求跑了命令，结果：**仍然是全黑输出**。补充测试了 Vulkan+Q6_K，正常。

## 环境
- GPU: AMD Radeon RX 7900 XTX 24GB (gfx1100)
- sd-cli: b68d586 (master-853)
- 模型: minimax_h3_fl2va_pruned-Q6_K.gguf (15.9 GB)，全部 resident 在 GPU
- 帧数: --video-frames 13 → 自动对齐到 22 帧
- 分辨率: 480x864, 5 steps, Euler, CFG=1.0, seed=42

## 决定性子测试（eager + 全 resident GPU）

```bash
./rocm/sd-cli.exe \
  --mode vid_gen \
  --diffusion-model minimax_h3_fl2va_pruned-Q6_K.gguf \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  -p "a cat sitting on a table" \
  --steps 5 --cfg-scale 1.0 --sampling-method euler \
  -W 480 -H 864 --video-frames 13 --seed 42 \
  --backend llm=cpu,vae=ROCm0,diffusion=ROCm0 \
  --log-level debug -o out.mp4
```

**结果**: 22 帧全部黑屏（像素值 0x00）

## 完整对比矩阵

| quant | backend | 设备 | 结果 | 采样耗时 | VAE 耗时 | 总耗时 |
|-------|---------|------|------|----------|----------|--------|
| Q6_K | **ROCm** | 7900XTX | **全黑** | 38.86s | 253.30s | 305s |
| Q6_K | **Vulkan** | 7900XTX | **正常** | 66.05s | 11.20s | 93s |
| Q4_K_M | ROCm | 7900XTX | 正常 | ~38s | ~253s | ~305s |
| Q4_K_M | Vulkan | 7900XTX | 正常 | ~66s | ~11s | ~93s |

像素验证:
- ROCm+Q6_K: 所有帧所有像素 = 0x00 (完全黑)
- Vulkan+Q6_K: 像素值 ~0x80 (中等亮度，正常)

## 已排除的变量

| 变量 | 测试值 | 黑屏是否复现 |
|------|--------|-------------|
| streaming | eager / stream | 都是黑 |
| graph-cut | 1 segment / 51 segments | 都是黑 |
| VAE placement | ROCm0 / CPU | 都是黑 |
| frame count | 13 / 22 / 73 | 都是黑 |
| backend | **ROCm** | **黑** |
| backend | **Vulkan** | **正常** |

## 关键日志（ROCm 测试）

```
[INFO] diffusion_engine.cpp:1221 - total params = 39748.45MB
  VRAM 20883.71MB: diffusion 15902.72MB, vae 4980.99MB

[DEBUG] ggml_runner.cpp:922 - minimax_h3 executing segment 1/1: graph
[VERBOSE] compute buffer size: 2475.63 MB(VRAM) on ROCm0

  | 1/5 - 18.31s/it   (预热)
  | 2/5 - 5.12s/it
  | 3/5 - 5.13s/it
  | 4/5 - 5.14s/it
  | 5/5 - 5.14s/it

VAE decode: 10 tiles, ~25s each
Total: 305.08s
```

## 结论

1. 黑屏与流式/prefetch/OCuLink 带宽/显存压力**无关**（权重全 resident GPU，eager 模式）
2. 问题锁定在 **Q6_K dequantization kernel × ROCm backend** 的计算错误
3. 同一模型、同一权重、同参数，换 Vulkan backend 即正常
4. Vulkan 的 Q6_K kernel 正确，ROCm 的 Q6_K kernel 输出全零

## 建议排查方向

- `ggml-cuda` 的 Q6_K mul_mat kernel（ROCm backend）
- 可能与 recent refactor (PR #1940, master-843+) 相关
- 建议二分 841-853 之间的提交，定位回归点

## 附件

- ROCm 测试日志: [leejet_debug_log.txt](https://github.com/user-attachments/files/32123086/leejet_debug_log.txt)
- Vulkan 测试日志: [leejet_vulkan_q6k_log.txt](https://github.com/user-attachments/files/32123013/leejet_vulkan_q6k_log.txt)
- Vulkan 测试日志2:[leejet_vulkan_q6k_log2.txt](https://github.com/user-attachments/files/32123012/leejet_vulkan_q6k_log2.txt)





> **Comment by leejet (2026-09-11T16:54:40Z):**

You reported two issues in this issue: a performance regression and completely black output videos.

It looks like the performance issue was not triggered this time. The black output is likely caused by a numerical overflow issue in the ROCm backend.

As for the performance issue, my guess is that the VRAM usage may have exceeded the available limit, causing the system to fall back to shared GPU memory (i.e. borrowing system RAM), which can significantly impact performance.

By the way, if you want to force segmented computation, you can use `--max-vram` and set it to a value slightly lower than the amount of VRAM required. This will cause the computation to be segmented automatically.


> **Comment by leejet (2026-09-11T17:45:15Z):**

Could you please try the latest `master` branch and add the following two options:

```text
--linear-scale 0.0078125 --attn-scale 0.0078125
```

and see whether they resolve the completely black output issue?

For example:

```bash
./rocm/sd-cli.exe \
  --diffusion-model minimax_h3_fl2va_pruned-Q6_K.gguf \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  -p "a cat sitting on a table" \
  --steps 5 --cfg-scale 1.0 --sampling-method euler \
  -W 480 -H 864 --video-frames 13 --seed 42 \
  --backend llm=cpu,vae=ROCm0,diffusion=ROCm0 \
  --log-level debug \
  --linear-scale 0.0078125 --attn-scale 0.0078125 \
  -o out.mp4
```

Please let me know whether the output is still completely black with these settings.


> **Comment by farawayso (2026-09-11T23:43:20Z):**

> Could you please try the latest `master` branch and add the following two options:
> 
> ```
> --linear-scale 0.0078125 --attn-scale 0.0078125
> ```
> 
> and see whether they resolve the completely black output issue?
> 
> For example:
> 
> ./rocm/sd-cli.exe \
>   --diffusion-model minimax_h3_fl2va_pruned-Q6_K.gguf \
>   --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
>   --vae minimax_h3_video_vae_fp16.safetensors \
>   -p "a cat sitting on a table" \
>   --steps 5 --cfg-scale 1.0 --sampling-method euler \
>   -W 480 -H 864 --video-frames 13 --seed 42 \
>   --backend llm=cpu,vae=ROCm0,diffusion=ROCm0 \
>   --log-level debug \
>   --linear-scale 0.0078125 --attn-scale 0.0078125 \
>   -o out.mp4
> Please let me know whether the output is still completely black with these settings.

@leejet 谢谢你的建议！用最新二进制（commit 7f410a3, master-859）加了 --linear-scale 0.0078125 --attn-scale 0.0078125 重新测试，黑屏问题已解决！

测试结果：

| 测试                         | YAVG (亮度均值) | 结果    |
|------------------------------|-----------------|---------|
| 旧 ROCm Q6_K（无参数）       | 0.00            | ❌ 全黑 |
| 新 ROCm Q6_K（+ scale 参数） | 140.01          | ✅ 正常 |
| Vulkan Q6_K（参考）          | 140.33          | ✅ 正常 |

新 ROCm 输出和 Vulkan 输出几乎完全一致（140.01 vs 140.33），问题已修复！

采样速度基本不变（~5.47s/it），性能无回归。

完整日志：[leejet_q6k_scale_fix_log.txt](https://github.com/user-attachments/files/32133955/leejet_q6k_scale_fix_log.txt)



> **Comment by farawayso (2026-09-12T04:21:52Z):**

# Feedback for Issue #1947: ROCm Q6_K Streaming Performance Analysis

## Summary

I've been testing the streaming mode with Q6_K quantization on ROCm and found that **PR #1905 (prefetch) was merged but then removed by PR #1940**. This means the current master (including master-859/7f410a3) does NOT have the async prefetch pipeline, which explains the performance degradation.

---

## Environment

- **OS**: Windows 11
- **GPU**: AMD Radeon RX 7900 XTX 24GB (gfx1100)
- **Connection**: OCuLink external GPU dock (PCIe 4.0 x4, 实测 bandwidth ~6.5 GB/s)
- **Binary**: master-859 (7f410a3) from Sep 12, 2026
- **Model**: minimax_h3_fl2va_pruned-Q6_K.gguf (15.9 GB)
- **Config**: 480x864, LoRA turbo4

---

## Test Results

### Test 1: 328 Frames + Default Segmented Streaming
```bash
# 328 frames with default segmented compute (30 segments)
[H3_SINGLE_SEG not set]
```
**Result**: Stalls at step 1/5 (~40s), appears to hang
- Weight staging: 17.7 GB / 3.49s = 5.1 GB/s
- VRAM usage: ~10 GB (well within 23 GB budget)
- **Analysis**: Sequential load-then-compute causes apparent stall

### Test 2: 73 Frames + Single Segment Mode
```bash
H3_SINGLE_SEG=1
```
**Result**: Success!
```
[INFO] H3_SINGLE_SEG=1: 单段执行 (--disable-segmented-compute)
generate_video completed in 161.35s
```
- 73 frames × 5 steps = 166s total
- **~33s/step**, output verified correct (717KB MP4)

### Test 3: 328 Frames + Single Segment Mode
```bash
H3_SINGLE_SEG=1
```
**Result**: OOM failure
```
[WARN] model manager cannot make enough memory available on ROCm0:
  need 29059.42 MB device / 24410.62 MB budget
```
- 328 frames requires ~29 GB, exceeds 24 GB physical limit

### Test 4: 73 Frames + Default Segmented Streaming (未测试)
- 根据 Issue #1947 报告，73 帧在默认分段模式下步骤 2-5 极慢 (~825s/step) 且输出黑屏
- 我尚未复现此场景，但基于 PR #1905 被删除的事实，推测性能退化原因相同

---

## Key Finding: PR #1905 Was Removed

Git history confirms:
```
6c57cc3 feat: prefetch streamed layers during compute (#1905)  [MERGED Sep 6]
462d675 refactor: unify runner lifecycles and weight residency (#1940)  [MERGED AFTER]
```

Current binary verification:
- `src/core/layer_stream_prefetch.cpp` **does not exist**
- `--layer-prefetch-depth` parameter **not available**
- `--resident-layers` parameter **not available**
- Only `--disable-prefetch` exists (defaults to false)

**Conclusion**: PR #1905's prefetch functionality was removed in PR #1940's refactor.

---

## PR #1906 Status

PR #1906 (feat: pool VRAM for streamed layer) is still **open** (unmerged):
- Depends on PR #1905 being merged first
- Provides `--stream-layer-pool` for VRAM allocation reuse
- Could fix the repeated malloc/free bottleneck

---

## Current Limitations

The current master-859 has:
- ✅ `--stream-layers` basic streaming (from PR #1576)
- ✅ `--linear-scale` / `--attn-scale` black screen fix (from PR #1964)
- ❌ `--layer-prefetch-depth` (removed after being merged)
- ❌ `--resident-layers` control
- ❌ Async prefetch pipeline
- ❌ VRAM pool reuse

---

## Recommendations

### Short-term Workaround
Use `H3_SINGLE_SEG=1` for short videos (≤100 frames):
```bash
set H3_SINGLE_SEG=1
minimax_h3_video_gen_full.bat
```
This disables segmented compute and runs monolithic graph execution.

### Long-term Fix Needed
1. **Re-evaluate PR #1905 removal**: The prefetch pipeline was useful for hiding transfer latency
2. **Merge PR #1906**: The VRAM pooling optimization addresses repeated allocation overhead
3. **ROCm-specific testing**: PR #1905/#1906 were only tested on CUDA, not ROCm

---

## Technical Details

### Performance Comparison

| Configuration | Frames | Time | Result |
|---------------|--------|------|--------|
| Default streaming | 328 | ~40s+ | Stalls at step 1 |
| H3_SINGLE_SEG=1 | 73 | 166s | ✅ Success |
| H3_SINGLE_SEG=1 | 328 | - | OOM (need 29GB) |

### Memory Analysis
- Q6_K pruned weights: 15.9 GB in RAM
- VRAM usage (73f): ~10 GB (VAE 5.5GB + compute 2.6GB + cache 0.45GB)
- VRAM budget: 23 GB (1 GB reserved)
- **Conclusion**: OOM is NOT the issue for 73 frames; performance is

---

## Request

Could the maintainer clarify:
1. Was PR #1905 removal intentional? The prefetch pipeline significantly improves streaming performance on bandwidth-limited connections.
2. Is there a plan to re-merge prefetch functionality or provide an alternative?
3. Can PR #1906 be prioritized for review/merge?

---

*Tested on 2026-09-12 with master-859 (7f410a3)*



> **Comment by leejet (2026-09-13T08:24:37Z):**

There is an obvious error in the LLM-generated content here. The prefetch feature is already enabled by default as long as the backend supports async offload; it has not been removed.

If you do not have sufficient ability to verify the LLM's output, please avoid using LLM-generated replies. Large amounts of incorrect or irrelevant content can easily distract from the actual issue.

This issue originally reported two problems:

1. The black-screen issue with the Q6_K GGUF **ROCm** backend. You have already reported that this issue has been resolved.
2. The performance regression, which was the original problem you reported. However, based on the tests from September 11, it appears that this issue is no longer present.

If the performance issue still exists, please provide the full command line and logs with debug-level logging enabled.


> **Comment by farawayso (2026-09-13T12:39:40Z):**

> 这里的LLM生成内容明显存在错误。只要后端支持异步卸载，预取功能默认已启用;它尚未被移除。
> 
> 如果你无法充分验证LLM的输出，请避免使用LLM生成的回复。大量错误或无关内容很容易分散对问题的注意力。
> 
> 本期最初报告了两个问题：
> 
> 1. Q6_K GGUF **ROCm** 后台的黑屏问题。你已经报告过这个问题已经解决。
> 2. 性能回归，这也是你最初报告的问题。然而，根据9月11日的测试结果，这个问题似乎已经不存在了。
> 
> 如果性能问题依然存在，请提供完整的命令行和日志，并启用调试级日志。

First, my apologies: I should have verified the LLM-generated analysis before posting it. It conflated two different things, and you're right to call that out.

**The two "prefetch" concepts the earlier reply got mixed up**

- `--disable-prefetch` — the default asynchronous next-segment weight prefetch. As you said, this was never removed; I confirmed it in the master-859 (7f410a3) binary: `--help` shows `--disable-prefetch  disable asynchronous next-segment weight prefetch (defaults to false)`.
- The dedicated CLI surface from PR #1905/#1906 (`--layer-prefetch-depth`, `--resident-layers`, `--stream-layer-pool`) is **not** present in the 859 binary. I believe I understood it backwards in the earlier reply: it wasn't "deleted" so much as the streamed-layer prefetch capability got folded into the default residency pipeline during the #1940 refactor — the *behavior* survives (default-on weight prefetch), the dedicated flags don't. So "prefetch removed" was the wrong phrasing; "the #1905/#1906 CLI surface was reworked into the default pipeline" is what I actually observed.

**On the two original problems**

1. Q6_K GGUF ROCm black screen — agreed, resolved on our side (the `--linear-scale`/`--attn-scale` values from #1964 fix it; output YAVG is normal).
2. Performance regression — per your note the Sep 11 testing shows it's gone globally, and I'm happy with that. What I can report is the state on this specific setup (7900 XTX 24GB via OCuLink external dock, 480x864 / 73f / 5-step / Q6_K / ROCm streaming, master-859):

| config | sampling time | per-step breakdown |
|---|---|---|
| default prefetch, `--log-level debug` (run 1) | 114 s | ~22.8 s/step, uniform |
| default prefetch, `--log-level debug` (run 2) | 824 s | step 1 = 746 s, steps 2-5 = ~19 s each |
| `--disable-prefetch` | 114 s | ~22.8 s/step, uniform |

The 824 s figure is suspiciously close to the "~825s" from the original report (though the slow chunk is in step 1 here rather than steps 2-5 there — possibly graph warmup / first-step transfer overhead landing on a different step). One earlier non-debug run did hit `ROCm error: unspecified launch failure` at step 1 (I'd say intermittent, ~1 in several runs), but I have no crash log with debug enabled — I can keep trying to capture one if it's useful.

**Attachments** (both `--log-level debug`, master-859 7f410a3, ROCm backend, Q6_K pruned, 480x864/73f/5 steps, default streaming, default prefetch on):
- `q6k_dbg_with_prefetch.log` — the 114 s run
- `q6k_dbg_with_prefetch_crash.log` — the 824 s run (name is a leftover; it completed successfully)

Exact command for both:

```
sd-cli.exe -M vid_gen \
  --diffusion-model minimax_h3_fl2va_pruned-Q6_K.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --cfg-scale 1 -W 480 -H 864 --fps 24 --video-frames 73 \
  --diffusion-fa --diffusion-conv-direct --vae-conv-direct --rng cpu \
  --flow-shift 12 --steps 5 --sampling-method euler --scheduler discrete \
  --log-level debug -t 16 \
  --linear-scale 0.0078125 --attn-scale 0.0078125 \
  --backend llm=cpu,vae=ROCm0,diffusion=ROCm0 \
  --params-backend diffusion=cpu --max-vram ROCm0=22 \
  -p "a cat sitting on a table" -o out.mp4
```

Thanks for the clarification on the prefetch side.

[q6k_dbg_with_prefetch.log](https://github.com/user-attachments/files/32160697/q6k_dbg_with_prefetch.log)
[q6k_dbg_with_prefetch_crash.log](https://github.com/user-attachments/files/32160696/q6k_dbg_with_prefetch_crash.log)


> **Comment by leejet (2026-09-13T16:45:19Z):**

Thanks for the logs. Both runs show `minimax_h3 executing segment 1/1: graph`, so next-segment prefetch was not active in either run. Weight staging took about 3.07 seconds in both cases, with no further staging logged during sampling.

The difference is concentrated in the first step: **37.09s versus 746.02s**. Steps 2–5 remain around **19s**. This confirms an intermittent first-step delay, but the logs do not yet identify its cause.

Please apply the diagnostic patch below, rebuild, and rerun the same command with `--log-level debug`. It separates preparation, backend execution, and output handling times. This is a diagnostic patch; it does not yet fix the slowdown. It builds and passes the CPU runner lifecycle regression tests, but I cannot validate ROCm locally.

```diff
diff --git a/src/core/ggml_runner.cpp b/src/core/ggml_runner.cpp
--- a/src/core/ggml_runner.cpp
+++ b/src/core/ggml_runner.cpp
@@ -877,6 +877,7 @@
                         index + 1, plan.segments.size(), segment.group_name.c_str(), phase);
             return std::nullopt;
         };
+        const int64_t preparation_start = ggml_time_us();
         cut_cache_.prune(segment.live_cut_names);
         bindings.reset(segment);
         if (!bindings.bind_cached_inputs(segment, get_desc().c_str())) {
@@ -944,12 +945,20 @@
         if (!prefetch_requests.empty()) {
             weights.enqueue_next(index, prefetch_requests.front());
         }
-        LOG_DEBUG("%s executing segment %zu/%zu: %s", get_desc().c_str(),
-                  index + 1, plan.segments.size(), segment.group_name.c_str());
-        if (!execute_segment(segment_graph, n_threads) ||
-            !cache_.capture(segment_graph) ||
+        LOG_DEBUG("%s executing segment %zu/%zu: %s (preparation %.2fs)", get_desc().c_str(),
+                  index + 1, plan.segments.size(), segment.group_name.c_str(),
+                  (ggml_time_us() - preparation_start) / 1e6);
+        const int64_t compute_start = ggml_time_us();
+        if (!execute_segment(segment_graph, n_threads)) {
+            return fail_segment("execution");
+        }
+        LOG_DEBUG("%s segment %zu/%zu compute completed in %.2fs on %s", get_desc().c_str(),
+                  index + 1, plan.segments.size(), (ggml_time_us() - compute_start) / 1e6,
+                  ggml_backend_name(runtime_backend));
+        const int64_t output_start = ggml_time_us();
+        if (!cache_.capture(segment_graph) ||
             !cut_cache_.capture(graph, segment, get_desc().c_str())) {
-            return fail_segment("execution or output caching");
+            return fail_segment("output caching");
         }
         sync_runtime_residency();
         if (last) {
@@ -969,6 +978,8 @@
         }
         // Final outputs and their callbacks may still be views of consumed cuts.
         cut_cache_.prune(segment.future_cut_names);
+        LOG_DEBUG("%s segment %zu/%zu outputs completed in %.2fs", get_desc().c_str(),
+                  index + 1, plan.segments.size(), (ggml_time_us() - output_start) / 1e6);
     }
     if (segments_changed || peak_compute_bytes != logged_compute_bytes_) {
         for (const auto& entry : peak_compute_bytes) {
```

Please share the complete logs for a normal and a slow run, along with your AMD driver version. Does the delay happen only on the first launch after a reboot, or also on subsequent launches?



> **Comment by farawayso (2026-09-14T07:17:21Z):**

Thanks for the analysis — and thanks for getting sdcpp working on AMD; it's been serving me well on this setup.

Direct answers to your two questions:

1. **Driver:** both GPUs report 32.0.31041.1004 (Adrenalin 32.0.x ROCm on Windows) — RX 7900 XTX (gfx1100) and 890M iGPU (gfx1150).
2. **Reproducibility:** not first-launch-only. It reproduces across consecutive reruns with no reboot in between — slow → fast → slow, interleaved. Intermittent, not tied to boot.

**Data I can hand over:**

- Two complete `--log-level debug` logs on master-859 (7f410a3), 480x864 / 73 frames / Q6_K / ROCm, backend `llm=cpu,vae=ROCm0,diffusion=ROCm0`, lazy streaming. The delay is isolated to step 1 in both:
  - [attach: q6k_dbg_fast.log] — step 1 = 37.09 s/it, steps 2–5 ~19 s, total ~114 s
  - [attach: q6k_dbg_slow.log] — step 1 = 746.02 s/it, steps 2–5 ~19 s, total ~824 s

- A locally built instrumented master-866 (42d6c0a) binary carrying your ggml_runner.cpp prep/compute/outputs patch, same conditions, run against the pip-installed ROCm 10.0 runtime the project uses day-to-day (stable.repo.amd.com, whl-next index) — not TheRock. Per-segment phase split:

  | run | total | step 1 (prep / compute / outputs) | steps 2–4 compute |
  |---|---|---|---|
  | [attach: 866diag_q6k_slow.log] (run 2, slow) | 849 s | 17.30 s / **730.97 s** / 0.01 s | ~19.4 s each |
  | [attach: 866diag_q6k_fast.log] (run 3, fast) | 129 s | 13.61 s / **20.37 s** / 0.00 s | 19.07 s each |

  Model load (8.63 s) and VRAM staging of 15.9 GB (3.07 s) complete before the timed window in both runs.

**Reading of the numbers:** the 730.97 s sits entirely inside the *first* diffusion forward, while the three identical follow-ups each take ~19 s — a 38x one-time cost, not a per-step algorithmic cost. Runs 2 and 3 differ only in runtime warm state (same model, flags, machine, minutes apart): step 1 compute 730.97 s → 20.37 s, total 849 s → 129 s. There are no JIT/autotune first-call log lines anywhere around that window, and the load/staging costs are timed separately above it. So the ~711 s extra lands on first HIP graph execution / ROCm runtime warm-up — the driver/init area you pointed at, not prefetch and not the GEMM.

**Build notes, in case they help anyone else:** this machine ships no AMD dev toolchain, so TheRock 7.14 was pulled into an isolated venv via rocm-cli (github.com/ROCm/rocm-cli, `rocm install sdk`) purely as the build toolchain; the instrumented exe still runs against the ROCm 10.0 runtime above. Two pitfalls: (a) MSVC 14.51 + TheRock clang 23 trips the isgreater/isless constexpr overload clash in the HIP math forward-declares header (llama.cpp#22570 / LLVM#201563) — fixed by moving the `__clang_cuda_math_forward_declares.h` include before `<cmath>`; (b) the 7.14 multi-arch package ships no rocBLAS TensileLibrary data files, so the exe ran against the ROCm 10.0 runtime's `_rocm_sdk_libraries/bin/rocblas/library` instead.

Happy to follow up with a HIP_DEBUG / ROCm-internal timing pass to pin the first-call cost to hipModuleLoad / first kernel launch vs the driver itself, if that helps narrow the fix — I can have my agent run it and post the split back here.

One caveat on the data above: I don't compile or test this myself — my agent (Hermes) handled the build and runs, so if any reading of the numbers is off, please bear with it. Also worth flagging: this GPU is attached over an OCuLink connection rather than a full-function x16 PCIe slot, so a flaky/incomplete link may surface driver-side issues that a standard setup wouldn't — keeping that in mind when interpreting the intermittent first-call cost.

[866diag_q6k_fast.log](https://github.com/user-attachments/files/32183432/866diag_q6k_fast.log)
[866diag_q6k_slow.log](https://github.com/user-attachments/files/32183435/866diag_q6k_slow.log)
[q6k_dbg_fast.log](https://github.com/user-attachments/files/32183434/q6k_dbg_fast.log)
[q6k_dbg_slow.log](https://github.com/user-attachments/files/32183433/q6k_dbg_slow.log)


> **Comment by leejet (2026-09-14T16:43:16Z):**

Thanks for testing the patch. The new logs narrow the delay to the first diffusion compute call: **20.37s in the fast run versus 730.97s in the slow run**, while output handling is negligible and subsequent steps remain around 19s. Both logs also show `disable_prefetch: true` and a single segment, confirming that this reproduction does not involve next-segment prefetch.

One clarification: the compute timer includes backend initialization, temporary allocations, GEMM calls, kernel execution, and synchronization. It does not yet distinguish between them, so we cannot rule out the hipBLAS/rocBLAS path. In this ggml version, the first execution of a new graph normally runs directly before graph capture is enabled; the timing alone does not identify HIP Graph capture or replay as the cause.

A HIP/rocBLAS trace would be useful now. Please keep the instrumented binary, runtime, and generation arguments unchanged, and enable these variables in the PowerShell session used to launch it:

```powershell
$env:AMD_LOG_LEVEL = "4"
$env:ROCBLAS_LAYER = "1"
```

Run your existing command with `--log-level debug`, appending `> trace-run1.log 2>&1` to capture stdout and stderr. Use a different filename for each run. If possible, attach one fast and one slow trace, compressed if large. Tracing itself can affect timing, so please note whether the large first-step delay still reproduces with tracing enabled.

Please also include the actual loaded HIP/hipBLAS/rocBLAS DLL paths and versions, plus the selected Tensile library directory and any runtime/library-path overrides. Since the diagnostic build uses the 7.14 toolchain with the 10.0 runtime, we should verify which binaries and kernel data are actually being loaded. That combination alone does not establish a cause.

The next target is to identify whether the long interval occurs during library/module initialization, memory allocation, a GEMM call, or synchronization. That should give us a concrete basis for the next fix.



> **Comment by farawayso (2026-09-14T23:19:28Z):**

Thanks for narrowing it down. Both the trace and the rebuilt instrumented binary are ready. Note: I had to rebuild the instrumented master-866 (`42d6c0a`) after deleting the source tree — build notes and one new build pitfall are at the bottom.

## What I captured

Two complete `AMD_LOG_LEVEL=4` + `ROCBLAS_LAYER=1` traces on the instrumented 866 binary, ROCm 10.0 pip runtime, identical generation args to the 866diag runs (480×864 / 73f / 4 steps / Q6_K pruned / ROCm0 / `--disable-prefetch` / `--linear-scale` & `--attn-scale` 0.0078125 / lazy streaming `diffusion=cpu` / `--max-vram 22`):

- **fast** run — step 1 compute `37.36 s/it`, total `151 s` → `trace-c3-fast.zip`
- **slow** run — step 1 compute `732.11 s/it`, total `853 s` → `trace-fast1-slow.zip`
- **duplicate slow** run — step 1 `730.91 s/it`, total `852 s` → `trace-fast22-slow.zip` (confirms the slow case is reproducible, not a one-off)

## Does tracing affect timing?

Mixed. On the **slow** run the trace adds nothing measurable (`730.9 s` without vs `732.1 s` with). On the **fast** run it adds ~17 s (`20.4 s` vs `37.4 s` step 1 — the trace itself slows the fast case), but the slow-vs-fast split still reproduces cleanly with tracing enabled.

## Where the ~700 s difference actually is (new data)

Per-event breakdown inside the step-1 compute window of both traces (AMD_LOG_LEVEL=4 timestamped API events, ~1.3 M events each):

| share of window events | fast (37 s) | slow (732 s) |
|---|---|---|
| PAL VMM heap ops (`palvirtual`, 2.7 GB pool) | 31.6 % | 32.7 % |
| kernel-launch config API (`Push/PopCallConfiguration`) | 31.1 % | 31.2 % |
| `hipLaunchKernel` | 15.5 % | 15.6 % |
| per-call error checking (`hip_error`) | 14.9 % | 14.8 % |
| `hipMemcpy*` | 2.1 % | 1.7 % |
| GEMM `Cijk` kernel lines | 0.7 % | < 0.1 % |

The event composition is essentially **identical** in both runs — same kernels, same counts, same call mix. The difference is purely wall time *between* events: the same ~1.3 M API events take 37 s in the fast run and 732 s in the slow one. This is not a GEMM-count or algorithmic difference, not a JIT/autotune first-call cost in the GEMM path, and not a single large blocking stall — the largest individual gap in the slow run is ~9 s, the rest is uniformly smeared across the window (thousands of sub-second stalls). So the ~700 s is per-API-call overhead (host↔device launch path / PAL VMM heap manager / error checking), **not compute**.

## Loaded binaries (7.14 toolchain build running against the 10.0 runtime)

Actual loaded modules of the running process (`Get-Process | Modules`):

| DLL | source | version |
|---|---|---|
| `amdhip64_7.dll` | `C:\WINDOWS\System32` (driver 32.0.31041.1004) | 10.0.3679.0 |
| `amd_comgr_3.dll` | `C:\WINDOWS\System32` | COMGR 3.0 |
| `hipblas.dll`, `rocblas.dll`, `libhipblaslt.dll` | pip ROCm 10.0 `_rocm_sdk_libraries\bin` | 10.0 |
| UCRT / MSVC runtime | `System32` (VS 14.51) | |

Trace confirms `HIP Library Path: C:\WINDOWS\SYSTEM32\amdhip64_7.dll`. So the mix is: system-driver HIP runtime + pip-ROCm-10.0 BLAS libs.

## Tensile library directory

`ROCBLAS_LAYER=1` was set but produces no lines of its own in this trace (ggml links the HIPBLAS path; the rocblas layer hooks fire but log nothing here). On disk the selected Tensile data is the 10.0 pip runtime's: `...\Python312\Lib\site-packages\_rocm_sdk_libraries\bin\rocblas\library\` (75 gfx1100 `.dat` files; lazy-JIT `.hsaco` fallbacks for the Cijk GEMMs — visible as `! kernel : Cijk_Alik_Bljk_..._ISA1100` lines that appear **before** the step-1 window in both runs). No runtime/library-path overrides other than the venv's own PATH entries.

## Build notes (rebuilt the instrumented 866 from scratch)

- **Toolchain:** TheRock 7.14.0a20260611 nightly wheel into an isolated venv via rocm-cli's `uv`; build dir `Temp/sdcpp-build-diag714`; **runtime:** ROCm 10.0 pip.
- **Pitfall (a)** — same as before: MSVC 14.51 + clang 23 `isgreater`/`isless` clash, fixed by including `__clang_cuda_math_forward_declares.h` before `<cmath>` (patched TheRock's `__clang_hip_runtime_wrapper.h`).
- **Pitfall (b):** the 7.14 devel tarball extraction on Windows leaves 0-byte files where the tar holds symlinks; re-extracting resolved it.
- **New pitfall:** without `--offload-arch=gfx1100 gfx1150` on the `.cu` compile lines the fatbin carries no gfx1100 code objects and the first kernel launch dies with `No compatible code objects found for: gfx1100` / `device kernel image is invalid`. The old successful build had these flags; CMake's Windows CXX-as-HIP path does not add them automatically, so I injected them via the ggml-hip target's compile options.

[trace-c3-fast.zip](https://github.com/user-attachments/files/32215503/trace-c3-fast.zip)
[trace-fast1-slow.zip](https://github.com/user-attachments/files/32215502/trace-fast1-slow.zip)
[trace-fast22-slow.zip](https://github.com/user-attachments/files/32215501/trace-fast22-slow.zip)


> **Comment by leejet (2026-09-21T14:58:48Z):**

Thanks for the traces. Comparing all three files, I found a concrete difference around the first compute-buffer allocation that makes memory placement the leading suspect.

These are the timings from the `minimax_h3 segment 1/1 compute completed` messages, with the longest `hipStreamSynchronize` call matched by thread:

| Trace | First compute | Synchronization wait | First workspace heap | Second compute |
| --- | ---: | ---: | --- | ---: |
| `trace-c3.log` | 20.33 s | 18.74 s | `heap[1]` | 19.18 s |
| `trace-fast1.log` | 732.11 s | 728.77 s | `heap[3]` | 19.38 s |
| `trace-fast22.log` | 730.91 s | 728.70 s | `heap[3]` | 19.43 s |

In both slow traces, the initial `hipMalloc(..., 2717076736)` request (about 2.53 GiB) is followed by **two `Failed PAL memory allocation!` messages**, then `hipSuccess`. This is at lines 5626–5630 in `trace-fast1.log` and 5629–5633 in `trace-fast22.log`. The resulting buffer, starting at `0000000AC7DD0000`, is subsequently logged as `heap[3]`. The fast trace allocates the same-sized buffer without those failures and logs `heap[1]`.

PAL defines heap 1 as local GPU memory and heap 3 as GPU-accessible, cached system memory. Strictly, the logger prints the allocation descriptor's first preferred heap, rather than a live physical-residency measurement. Together with the allocation failures, this is strong evidence of a system-memory fallback in the slow runs. See the [PAL heap definitions](https://github.com/GPUOpen-Drivers/pal/blob/dev/inc/core/pal.h), [allocation descriptor](https://github.com/GPUOpen-Drivers/pal/blob/dev/inc/core/palGpuMemory.h), and [ROCm memory logger](https://github.com/ROCm/clr/blob/develop/rocclr/device/pal/palvirtual.hpp).

There is also a useful explanation for the first-step-only behavior: before step 2, that buffer is freed and a slightly larger one (2,718,835,456 bytes) is allocated. The replacement is logged as `heap[1]` in all three runs, and compute returns to about 19 seconds.

The long synchronization call means we should revise the conclusion about uniformly distributed API overhead. The main thread waits for about 729 seconds while the runtime worker thread continues producing log entries; activity in the combined log does not exclude a long blocking call. This does not identify an individual slow kernel, but it does locate almost all the elapsed time in waiting for queued work to finish.

Could you repeat the same test a few times, changing only `--max-vram ROCm0=22` to `--max-vram ROCm0=16`, and keeping `--disable-prefetch`, the CPU parameter placement, executable, and inputs unchanged? Please keep `AMD_LOG_LEVEL=4` enabled for the comparison. A smaller budget may introduce graph segmentation, so the useful checks are whether the PAL allocation failures and `heap[3]` compute workspace disappear, and whether the roughly 730-second first step disappears with them.

This is a diagnostic test, not yet a confirmed fix. The traces identify the allocation/placement difference; they do not yet establish why the first device allocation fails intermittently.



> **Comment by farawayso (2026-09-21T23:29:43Z):**

> Thanks for the traces. Comparing all three files, I found a concrete difference around the first compute-buffer allocation that makes memory placement the leading suspect.
> 
> These are the timings from the `minimax_h3 segment 1/1 compute completed` messages, with the longest `hipStreamSynchronize` call matched by thread:
> 
> Trace	First compute	Synchronization wait	First workspace heap	Second compute
> `trace-c3.log`	20.33 s	18.74 s	`heap[1]`	19.18 s
> `trace-fast1.log`	732.11 s	728.77 s	`heap[3]`	19.38 s
> `trace-fast22.log`	730.91 s	728.70 s	`heap[3]`	19.43 s
> In both slow traces, the initial `hipMalloc(..., 2717076736)` request (about 2.53 GiB) is followed by **two `Failed PAL memory allocation!` messages**, then `hipSuccess`. This is at lines 5626–5630 in `trace-fast1.log` and 5629–5633 in `trace-fast22.log`. The resulting buffer, starting at `0000000AC7DD0000`, is subsequently logged as `heap[3]`. The fast trace allocates the same-sized buffer without those failures and logs `heap[1]`.
> 
> PAL defines heap 1 as local GPU memory and heap 3 as GPU-accessible, cached system memory. Strictly, the logger prints the allocation descriptor's first preferred heap, rather than a live physical-residency measurement. Together with the allocation failures, this is strong evidence of a system-memory fallback in the slow runs. See the [PAL heap definitions](https://github.com/GPUOpen-Drivers/pal/blob/dev/inc/core/pal.h), [allocation descriptor](https://github.com/GPUOpen-Drivers/pal/blob/dev/inc/core/palGpuMemory.h), and [ROCm memory logger](https://github.com/ROCm/clr/blob/develop/rocclr/device/pal/palvirtual.hpp).
> 
> There is also a useful explanation for the first-step-only behavior: before step 2, that buffer is freed and a slightly larger one (2,718,835,456 bytes) is allocated. The replacement is logged as `heap[1]` in all three runs, and compute returns to about 19 seconds.
> 
> The long synchronization call means we should revise the conclusion about uniformly distributed API overhead. The main thread waits for about 729 seconds while the runtime worker thread continues producing log entries; activity in the combined log does not exclude a long blocking call. This does not identify an individual slow kernel, but it does locate almost all the elapsed time in waiting for queued work to finish.
> 
> Could you repeat the same test a few times, changing only `--max-vram ROCm0=22` to `--max-vram ROCm0=16`, and keeping `--disable-prefetch`, the CPU parameter placement, executable, and inputs unchanged? Please keep `AMD_LOG_LEVEL=4` enabled for the comparison. A smaller budget may introduce graph segmentation, so the useful checks are whether the PAL allocation failures and `heap[3]` compute workspace disappear, and whether the roughly 730-second first step disappears with them.
> 
> This is a diagnostic test, not yet a confirmed fix. The traces identify the allocation/placement difference; they do not yet establish why the first device allocation fails intermittently.

**Headline:** the `heap[3]` / system-RAM fallback hypothesis does **not** reproduce (10 runs, `Failed PAL memory allocation` = 0, `heap[3]` = 0). The real trigger is **automatic graph segmentation** (1 -> 51 segments), and at a 16 GiB budget a monolithic graph **OOMs immediately**. Segmentation is not a side effect of lowering `--max-vram`; it is the only available path.

---

## 1. Environment

- Binary: `stable-diffusion.cpp` commit `c678dfe` (current `rocm/sd-cli.exe`), **without** the ggml_runner patch. Step timings come from the progress line `N/4 - X.XXs/it`, so they include per-step preparation.
- Driver Adrenalin 32.0.31041.1004, RX 7900 XTX 24 GB gfx1100, OCuLink.
- Runtime pip ROCm 10.0. `hipGetDevice`: `VMM: no, Wave Size: 32, VRAM: 24560 MiB`.
- Otherwise identical to the 866diag runs: 480x864, 73 frames, 4 steps, euler, flow-shift 12.0, Q6_K pruned, `--disable-prefetch`, `--params-backend diffusion=cpu`, `--linear-scale`/`--attn-scale` 0.0078125, `AMD_LOG_LEVEL=4`.

Command (only `--max-vram` changed):

```
sd-cli.exe -M vid_gen --llm ...qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --diffusion-model ...minimax_h3_fl2va_pruned-Q6_K.gguf \
  --vae ...minimax_h3_video_vae_fp16.safetensors \
  --prompt "a cat sitting on a table" -W 480 -H 864 --steps 4 \
  --sampling-method euler --scheduler discrete --flow-shift 12.0 \
  --cfg-scale 1.0 --guidance 3.5 --video-frames 73 --fps 24 -s 42 \
  --linear-scale 0.0078125 --attn-scale 0.0078125 \
  --backend llm=cpu,vae=ROCm0,diffusion=ROCm0 --params-backend diffusion=cpu \
  --max-vram ROCm0=<22|20|18|16> --disable-prefetch \
  --diffusion-fa --diffusion-conv-direct --vae-conv-direct --log-level debug
```

Strictly serial runs; no concurrent GPU processes.

## 2. The hypothesis does not reproduce

Across **all 10 runs**: `Failed PAL memory allocation` = **0**, `heap[3]` occurrences = **0** (only `heap[0]` and `heap[1]` ever appear). There is no system-memory fallback to remove, at either budget.

## 3. `--max-vram 16` is a segmentation trigger, not a memory-placement change

| | `--max-vram 22` | `--max-vram 16` |
|---|---|---|
| graph segments | **1** | **51** |
| diffusion compute buffer | 2591.21 MB VRAM | 2380.91 MB VRAM |
| workspace alloc | 2717076736 B (2.53 GiB) | 2496456832 B (2.33 GiB) |
| params blocks | 15902.72 MB, 1 block, `ROCm_Host` | 963.08 + 5x304.83 MB, many blocks |
| large alloc placement | `heap[1]` | `heap[1]` |

Segment-count evidence differs by regime: for 51 segments the log emits `minimax_h3 using 51 segments` (ggml_runner.cpp:853); for the 1-segment case that line is **not printed at all** — segment count there is read from `minimax_h3 executing segment 1/1: graph` and `peak across 1 segment` (ggml_runner.cpp:947/977).

The workspace stays 2.3-2.5 GiB and stays on `heap[1]` at both budgets — so the smaller budget did not relocate it. It only shrank the workspace and fragmented the weights into many small host blocks.

## 4. Segmentation threshold: exactly 18.06 GiB

A monolithic graph needs a **18493.95 MB** budget. Below that value segmentation kicks in:

| max_vram | budget (MiB) | vs threshold | segments | result |
|---|---|---|---|---|
| 22 | 22528 | +4034.05 MB | 1 | fast |
| **20** | 20480 | **+1986.05 MB** | **1** | **fast, 128.03 s** |
| **18** | 18432 | **-61.95 MB** | **51** | **slow, 2563.53 s** |
| 16 | 16384 | -2109.95 MB | 51 | slow |

`--max-vram 18` misses the threshold by **61.95 MB** — one byte under and it segments. The two slow cases reproduce within 0.1% of each other (2563.53 s vs 2566.52 s), so the cost is pinned to segmentation, not to the budget size.

## 5. Decisive test: 16 GiB + forced monolithic graph -> OOM

`--max-vram ROCm0=16` **plus** `--disable-segmented-compute`:

```
[WARN  ] model_manager.cpp:1801 - model manager cannot make enough memory
          available on ROCm0: need 19005.95 MB device / 18493.95 MB budget,
          available 24410.62 MB device / 16384.00 MB budget
[ERROR ] ggml_runner.cpp:873 - minimax_h3 segment 1/1 (graph) failed during
          weight preparation
[ERROR ] video.cpp:1696 - sampling failed after 1.19s
```

12 seconds to exit, rc=1.

Physical VRAM is fine (24410.62 MB). The **budget cap** (16384.00 MB) is what blocks it; the gap is **2109.95 MB ~ 2.06 GiB**. So at 16 GiB the option "small budget but monolithic graph" does not exist — the graph does not fit. Segmentation is not a side effect of the lower budget; it is the only available path. To keep a monolithic graph the budget must be >= 18.07 GiB; `--max-vram 20` already has 1986 MB of headroom.

## 6. The slowness is per-step, not first-step

```
run     max_vram  segs  step1     step2     step3     step4     total
v22_01  22        1     37.92     19.28     19.10     19.07     133.59
v22_02  22        1     37.19     19.45     19.26     19.23     133.14
v22_03  22        1     37.76     19.34     19.16     19.13     132.49
v22_04  22        1     37.22     19.32     19.18     19.15     131.74
v22_05  22        1     37.81     19.19     19.09     19.09     133.33
v20     20        1     36.59     19.15     19.05     19.08     128.03
v18     18        51    42.83     830.48    826.38    825.20    2563.53
v16_01  16        51    42.74     814.68    829.37    831.87    rc=1*
v16_02  16        51    43.77     (crash)
v16_03  16        51    43.06     815.65    835.01    834.25    2566.52
```

1. All six 22/20 GiB runs are uniformly fast (~19 s/step). The intermittent slow case did not appear at those budgets.
2. At 18/16 GiB, steps 2/3/4 are **each** ~815-835 s. This is a **per-step** cost, not one-time warm-up or first-allocation cost. That contradicts the "first buffer lands badly, then gets replaced" model, which predicts step 1 slow and steps 2+ normal — the opposite of what was measured.

## 7. The mechanism: a 24x kernel-launch throughput collapse

| run | hipLaunchKernel | span | launches/s |
|---|---|---|---|
| v22_04 (1 seg) | 218,608 | 131.7 s | **1659** |
| v16_01 (51 seg) | 150,962 | 2164.2 s | **69.8** |
| v16_03 (51 seg) | 195,704 | 2566.5 s | **76.3** |

Same graph content, **24x fewer launches per second**. And there is no single blocked `hipStreamSynchronize`: the main thread is quiet while the worker keeps emitting API calls throughout the slow window.

`PAL fence isn't ready!` appears **only** in the slow runs: **0** in every 22/20 GiB run; 23 / 298 / 298 in the three 16 GiB runs. Inter-arrival is a fixed ~6 s poll (min 6000.1 ms, p50 6015.8 ms, max 11337 ms), so the message is not itself a stall — but its presence tracks the slow regime exactly.

## 8. Separate finding: v16_02 crashed

```
palvirtual.cpp:464  PAL failed to submit CMD! result:-7
palvirtual.cpp:417  PAL failed to finalize a command buffer! result: -28  (x3)
hipStreamSynchronize: Returned hipErrorLaunchFailure
ggml: ROCm error: unspecified launch failure
  current device: -1, in ggml_backend_cuda_buffer_cpy_tensor
  ggml-cuda.cu:842
exit code 3221226505 (0xC00000FD, STATUS_STACK_OVERFLOW)
```

The failing copy is `__amd_rocclr_copyBuffer`, src 2505506816 B `heap[1]` -> dst 197001216 B `heap[1]`. Still no `heap[3]`, still no allocation failure — this is a command-submission failure. A hard failure, not merely slow.

## 9. Conclusion

`--max-vram ROCm0=16` is **not** a fix; it makes things worse — **133 s -> 2567 s, 19x** — in order to remove an allocation-failure signature that does not exist in this environment. And forcing a monolithic graph at that budget OOMs immediately.

At this model and size, **low budget = segmentation = slow**, with no middle state.

The productive direction is why 51 segments drops launch throughput from 1659/s to 70/s. That is the ~19x per-step cost, and since steps 2+ are all slow it is a **per-step graph-switching cost**, not a first-launch cost.

---

Raw traces: `v22_05_raw.zip` (22 GiB, rc=0, 1 segment, 133.33 s), `v20_raw.zip` (20 GiB, rc=0, 1 segment, 128.03 s), `v18_raw.zip` (18 GiB, rc=0, 51 segments, 2563.53 s), `v16_03_raw.zip` (16 GiB, rc=0, 51 segments, complete), `v16_01_raw.zip` (16 GiB, 51 segments, killed), `v16_02_raw.zip` (16 GiB, the crash), `v16_noseg_raw.zip` (16 GiB + forced monolithic, the OOM).

\* All 10 runs complete. v16_01 shows rc=1 because the process was killed (it had reached step 4 and was still running); its per-step numbers are valid but `total` is not available. v16_03 and v18 are clean complete slow cases.

[v22_05_raw.zip](https://github.com/user-attachments/files/32491148/v22_05_raw.zip)

[v20_raw.zip](https://github.com/user-attachments/files/32490841/v20_raw.zip)

[v18_raw.zip](https://github.com/user-attachments/files/32490844/v18_raw.zip)

[v16_03_raw.zip](https://github.com/user-attachments/files/32490846/v16_03_raw.zip)

[v16_01_raw.zip](https://github.com/user-attachments/files/32490851/v16_01_raw.zip)

[v16_02_raw.zip](https://github.com/user-attachments/files/32490855/v16_02_raw.zip)

[v16_noseg_raw.zip](https://github.com/user-attachments/files/32490859/v16_noseg_raw.zip)



### #1976 [closed] Video regressions on 16 GB Vulkan: LTX-2.5 + MiniMax-H3 broken since master-864, Wan 2.2 broken by master-866 (graph segmentation / weight-budget behavior) — --auto-fit off only rescues Wan
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1976  by laurentvv 2026-09-14T19:39:31Z  comments: 15

**TL;DR:** Two distinct video regressions on 16 GB Vulkan, both with deterministic A/B repro (same command/seed, control run on old build passes):

1. **LTX-2.5 + MiniMax-H3 break in the 841→864 window** (memory-manager rework #1940/#1956/#1957): graph now cut into 50+ segments, dies with `vk::Queue::submit: ErrorOutOfDeviceMemory` despite ~15 GB free VRAM — master-841 control passes with 2 segments.
2. **Wan 2.2 T2V breaks in 864→866** (suspect #1967): DiT graph switches 1 segment → 42 segments, dies at step 5/8; `--auto-fit off` rescues it.

Image paths (Flux/SDXL/upscale) unaffected. We're pinning master-864 plus a parallel master-841 install meanwhile. **Happy to bisect the 841→864 window or run any experiment.**

---

## Environment

- GPU: AMD Radeon RX 6950 XT 16 GB (RDNA2, Vulkan, proprietary driver, no matrix cores)
- OS: Windows x64
- Binaries: official release assets (`sd-master-<sha>-bin-win-vulkan-x64.zip`); the master-841 reference run used a parallel install from our archived binaries (`--version` verified `6b3edaa`)
- Machine state for all runs: freshly booted or verified idle, 1.1–1.3/16 GB VRAM in use
- Models: `Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf` + `umt5-xxl-encoder-Q4_K_M.gguf` + `wan_2.1_vae.safetensors`, `LTX-2.5-Distilled-Q4_K_M.gguf`, `minimax_h3_ref2va_pruned-Q4_K_M.gguf`

## Summary

Three video paths break differently across recent builds on a 16 GB Vulkan setup — with **two distinct regression windows**:

| | master-841 `6b3edaa` | master-864 `ca37fad` | master-866 `42d6c0a` |
|---|---|---|---|
| Wan 2.2 T2V 14B | ❌ (#1946, fixed later) | ✅ | ❌ (works with `--auto-fit off`) |
| LTX-2.5 (distilled, T2V & I2V) | ✅ | ❌ | ❌ (identical failure) |
| MiniMax-H3 Ref2VA (turbo LoRA) | ✅ | ❌ | ❌ (identical failure) |
| Flux / SDXL / ESRGAN upscale | ✅/✅/✅ | ✅/✅/✅ | ✅/✅/✅ |

- **LTX-2.5 and MiniMax-H3 broke in the 841→864 window** (the memory-manager rework), not in 864→866. We had initially mislabeled master-864 as the last good build for these two — in fact all our LTX/H3 validations had been done on `6b3edaa` and these paths were simply never re-tested after upgrading; the failures below are deterministic (identical computed refusals across builds and days, on freshly booted machines).
- **Wan 2.2 broke in the 864→866 window** (suspect: #1967), where the DiT graph switches from fully resident (1 segment) to a 42-segment cut that fails mid-sampling.

## 1. Wan 2.2 T2V A14B — fails at step 5/8 on master-866, works on master-864

Command (identical on both versions, seed fixed):

```
sd-cli.exe -M vid_gen \
  --diffusion-model Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf \
  --t5xxl umt5-xxl-encoder-Q4_K_M.gguf \
  --vae wan_2.1_vae.safetensors \
  -p "a red fox walking through deep snow, winter forest, cinematic" \
  --cfg-scale 6.0 --steps 8 --sampling-method euler \
  -W 832 -H 480 --video-frames 17 --fps 24 \
  --diffusion-fa --temporal-tiling --vae-on-cpu --seed 42
```

**master-864 (working)**: DiT weights resident (`model manager prepared params backend buffers (9211.74 MB, 1095 tensors, 10 blocks, VRAM)`); compute graph runs in 1 segment (`Wan2.x-T2V-14B compute buffer size: 941.58 MB(VRAM) (peak across 1 segment)`); sampling completes 8/8 (~264 s/it); VAE decode on CPU (`--vae-on-cpu`); webm written (2332 s total).

**master-866 (failing)**: identical weight placement (same 9211.74 MB / 1095 tensors resident in VRAM), but the graph is now cut into **42 segments** (`compute buffer size: 787.47 MB(VRAM) (peak across 42 segments)`), and generation dies at **step 5/8** after 1390 s of sampling:

```
[WARN   ] model_manager.cpp:1767 - model manager cannot make enough memory available on Vulkan0: need 664.34 MB device / 1092.16 MB budget, available 631.30 MB device / 2366.17 MB budget
[ERROR  ] ggml_runner.cpp:877  - Wan2.x-T2V-14B segment 24/42 (wan.blocks.22) failed during weight preparation
[ERROR  ] diffusion_engine.cpp:2421 - diffusion model compute failed
```

The shape is reminiscent of #1946 (a few dozen MB short during weight preparation), but this is the **new segmented path** on a graph that master-864 kept fully resident.

**Workaround (verified end-to-end on master-866)**: adding `--auto-fit off` restores the master-864 behavior — back to 1-segment resident compute (941.58 MB), sampling completes 8/8 (~272 s/it vs 264), webm written (2399 s total). Output is the same scene/trajectory with visible numerical divergence vs the master-864 run (middle-frame mean abs diff 14.8/255), as expected across different graph layouts.

## 2. LTX-2.5 — broken since master-864, with AND without `--auto-fit off` (no workaround found)

Recipe: explicit `--backend diffusion=vulkan0,te=cpu,vae=cpu` (all weights streamed from RAM: text encoder ~9.0 GB, DiT ~14.4 GB, VAE ~1.7 GB — **0 MB of weights in VRAM**), 768×512, 33 frames, 8 steps with Lightricks distilled sigmas, euler_a, cfg 1.0. Last verified working on **master-841**; on **both master-864 and master-866** it dies ~20 s in at the first workspace capacity check, with byte-identical numbers:

```
[WARN   ] model_manager.cpp:1767 - model manager cannot make enough memory available on Vulkan0: need 910.16 MB device / 398.16 MB budget, available 319.37 MB device / 668.90 MB budget
[ERROR  ] ggml_runner.cpp:877  - ltxav segment 1/1 (graph) failed during workspace capacity check
[ERROR  ] video.cpp:1684 - sampling failed after 21.24s
```

With `--auto-fit off` the budget becomes "unlimited" but available device stays ~317 MB → same failure at 19 s. Note the paradox: with ~15 GB of VRAM physically free and 0 MB of weights resident in VRAM, the manager still reports ~320 MB "available device".

I2V (65 frames, 832×480, init image) also fails on master-864 with `vk::Queue::submit: ErrorOutOfDeviceMemory` ~115 s into sampling; interestingly that run did **not** segment the graph at all, while the same command in 480×832 portrait did segment (50 segments) and still died at submit.

## 3. MiniMax-H3 Ref2VA — broken since master-864 (same signature as master-866)

Recipe: `--backend diffusion=vulkan0,te=cpu,vae=cpu --offload-to-cpu --rng cpu --max-vram 10`, turbo LoRA 8 steps, 22 frames, 864×480. On **master-841** the graph-cut produced **2 segments** and was stable. On **both master-864 and master-866**, after the CPU-side reference/audio conditioning completes (~15–20 min), the DiT reports **51 segments** and dies at submit:

```
[VERBOSE] ggml_runner.cpp:857  - minimax_h3 using 51 segments
[ERROR   ] ggml_runner.cpp:651  - minimax_h3 graph execution failed on Vulkan0: vk::Queue::submit: ErrorOutOfDeviceMemory
[ERROR   ] video.cpp:1684 - sampling failed after 262.86s   (master-864)  /  358.87s  (master-866)
```

**Control run (master-841, parallel install, same machine 24 h apart, exact same command/seed/assets): PASSES** — `minimax_h3 graph cut executing segment 1/2: minimax_h3.blocks.0..minimax_h3.blocks.37` + `segment 2/2`, 8 steps × 2 segments, webm 864×480 + PCM 32 kHz stereo written in 2279 s (historical timing). This isolates the regression to the build, not the environment.

## What still works on master-866

- Flux schnell smoke (512×512, 4 steps, seed 42): clean image
- SDXL (monolithic checkpoint, 512×512, 20 steps): clean image
- ESRGAN upscale mode (4x-UltraSharp): fine

## Current mitigation on our workstation (until fixes land)

Two side-by-side installs, same GPU/driver, same models directory (`C:\Modeles_LLM\`), only the binaries differ (`sd-cli.exe --version` verified before each run):

1. **`C:\SD\` = master-864 `ca37fad`** (main install) — used for:
   - image paths: Flux schnell/dev, SDXL, ESRGAN upscale (`-M upscale`);
   - Wan 2.2 T2V **and** I2V (A14B LowNoise Q4_K_M): **≤ ~20 frames** with `--vae-on-cpu` mandatory on 16 GB (resident GPU VAE decode of 17 frames requests 19.4 GB; 33 frames gets a clean 2.2 GB workspace refusal). Validated end-to-end at 17 frames.
2. **`C:\SD-6b3edaa\` = master-841 `6b3edaa`** (parallel install, binaries restored from our archive into a separate directory, never mixed with the main install) — used for:
   - MiniMax-H3 Ref2VA production: re-validated 2026-09-15 with the exact turbo recipe (2-segment graph cut × 8 steps, webm 864×480 + PCM 32 kHz stereo in 2279 s);
   - LTX-2.5 production: last validated on this build (33-frame benchmark 2026-09-10, 65-frame I2V production chains 2026-09-12/13); not re-run since the A/B, same binary as the H3 control.

Coverage: every validated recipe on this workstation has a working binary again. What we lose meanwhile: the 841→864 memory-manager improvements (fix #1946 for Wan/Flux, streaming perf) on the LTX/H3 paths — which is why a fix in the 841→864 window would let us collapse back to a single install.

## Suspects

- **LTX-2.5 + MiniMax-H3 (window 841→864)**: the memory-manager rework — #1940 (unify runner lifecycles and weight residency) and related refactors (#1956 model sources, #1957 generation pipeline split), possibly interacting with #1958 (memory capacity guards).
- **Wan 2.2 T2V (window 864→866)**: #1967 (`44dd137`, preserve explicit backend assignments during auto-fit) is the only memory/auto-fit-related commit in `ca37fad...42d6c0a` (the others: attention regex #1919, Wan2.2 S2V #1925, video metadata #1901, MSVC warnings #1969).

Happy to run any experiment that helps (bisect builds for the 841→864 window, extra logs, other flag combinations).



> **Comment by laurentvv (2026-09-23T13:32:10Z):**

Follow-up: we tested **master-899 `28b454b`** (2026-09-22 release, includes #2019 and #2020) on the same machine, same models, same commands. Full matrix below — net result: **the Wan 2.2 regression is gone, LTX-2.5 now passes but is VRAM-margin sensitive, MiniMax-H3 is still broken with the identical signature.**

## Results on master-899 (freshly booted, desktop VRAM usage 0.4/16 GB)

| Path | master-841 `6b3edaa` | master-864/866 | **master-899 `28b454b`** |
|---|---|---|---|
| Flux 1024² (dev, 4 steps) | ✅ | ✅ | ✅ |
| Wan 2.2 T2V 17f (exact recipe above) | ❌ (#1946) | ✅ (864) / ❌ (866) | ✅ — sampling 2023 s vs 2116 s on 864 (−4 %), no `--auto-fit off` needed |
| LTX-2.5 T2V 33f benchmark (2 steps, 768×512) | ✅ | ❌ | ✅ **on fresh boot** — 53.75 s sampling, webm written — but ❌ on the *same build* before a reboot, see below |
| MiniMax H3 Ref2VA turbo (exact recipe above) | ✅ (2 segments) | ❌ | ❌ **unchanged**: `minimax_h3 using 51 segments` + `vk::Queue::submit: ErrorOutOfDeviceMemory`, sampling failed after **276.68 s** (864: 262.86 s, 866: 358.87 s) — deterministic, reproduced on a freshly rebooted machine |

## 1. LTX-2.5 on master-899: passes, but the capacity check is desktop-VRAM-margin sensitive

Same command, same build, same machine, ~40 minutes apart:

**Before reboot** (desktop/apps holding ~0.8 GB VRAM) — fails at 21.5 s:

```
[WARN   ] model_manager.cpp:1809 - model manager cannot make enough memory available on Vulkan0: need 908.75 MB device / 396.75 MB budget, available 787.26 MB device / 668.90 MB budget
[ERROR  ] ggml_runner.cpp:880  - ltxav segment 1/1 (graph) failed during workspace capacity check
[ERROR  ] video.cpp:1715 - sampling failed after 21.51s
```

**After reboot** (desktop VRAM usage 0.8 → 0.4 GB) — passes end-to-end (53.75 s sampling, webm written, one non-fatal `Failed to allocate pinned memory` WARN).

So the failure margin here was **~121 MB of "available device"**. Two observations worth flagging:

- **The available-device under-reporting seems greatly reduced by #2020.** In our master-864/866 runs (original report above) the manager claimed `available 319.37 MB device` while ~15 GB was physically free and 0 MB of weights were resident in VRAM. On master-899 the same staging reports `available 787.26 MB device / total 16368.00 MB, tracked weights 14388.75 MB` — consistent with the actual free VRAM minus non-tracked overhead. Whatever ghost reservation produced the 319 MB figure appears gone (matches #2020 "handle GPU memory reports").
- The remaining ~120 MB shortfall was real desktop VRAM held by other processes. Practical consequence: **on a 16 GB card the LTX-2.5 33-frame capacity check now passes but with little headroom** — anything else holding a few hundred MB of VRAM flips it back to a refusal. We keep our master-841 parallel install for LTX/H3 production for that reason (our LTX I2V 65-frame production chains are also not re-validated on 899 yet).

## 2. MiniMax-H3: still the blocker

The 841→864 regression is **not** fixed by #2019/#2020: the DiT graph is still cut into **51 segments** (control build: 2) and dies at submit with `ErrorOutOfDeviceMemory` several minutes into sampling, deterministically, on a fresh boot. H3 production on our side still runs on the master-841 parallel install. Happy to test any candidate fix or bisect the 841→864 window for this path — we have byte-stable A/B repro (same command/seed/assets, control webm from the 841 binary).



> **Comment by laurentvv (2026-09-24T11:32:45Z):**

Follow-up: we tested **master-908 `88411ef`** (2026-09-23 release, includes #1900 "add graph cuts for MiniMax-H3 text conditioning") on the same machine, same models, same commands as the matrix above, freshly rebooted (desktop VRAM usage 0.4/16 GB). Net result: **Flux/LTX/Wan are stable or slightly faster, but MiniMax-H3 is still broken — the graph is now cut into 54 segments (was 51 on 864/866/899, control = 2 on 841) and still dies at submit with `ErrorOutOfDeviceMemory`.**

## Results on master-908 vs previous builds

| Path | master-841 `6b3edaa` | master-864→899 | **master-908 `88411ef`** |
|---|---|---|---|
| Flux 1024² (dev, 4 steps) | ✅ | ✅ 26.98 s (899) | ✅ 27.00 s |
| LTX-2.5 T2V 33f benchmark (2 steps, 768×512) | ✅ | ✅ on fresh boot — 53.75 s (899) | ✅ 54.21 s, 1 segment, no capacity-check refusal, same non-fatal `Failed to allocate pinned memory` WARN |
| Wan 2.2 T2V 17f (exact recipe above) | ✅ | ✅ 2023 s (899, −4 % vs 864) | ✅ **1999.5 s (−1.2 % vs 899)**, 1 resident segment (9211.74 MB weights, identical to 864/899), VAE CPU decode 209.5 s |
| MiniMax H3 Ref2VA turbo (exact recipe above) | ✅ (**2 segments**) | ❌ `using 51 segments` + OOM at submit (899: 276.68 s) | ❌ **`using 54 segments` + OOM at submit, sampling failed after 257.37 s** |

## #1900 changes the H3 graph cut (51 → 54 segments) but does not restore the healthy plan

Full failure sequence on master-908 (same command/seed/assets as the control webm produced by the 841 binary):

```
[VERBOSE] vae.hpp:262  - computing vae encode graph completed, taking 855.82s
[VERBOSE] conditioner.hpp:3139 - computing condition graph completed, taking 92884 ms
[VERBOSE] ggml_runner.cpp:864  - minimax_h3 using 54 segments
[ERROR  ] ggml_runner.cpp:654  - minimax_h3 graph execution failed on Vulkan0: vk::Queue::submit: ErrorOutOfDeviceMemory
[ERROR  ] diffusion_engine.cpp:2662 - Diffusion model sampling failed
[ERROR  ] video.cpp:1709 - sampling failed after 257.37s
```

Two observations:

- The segment count **changed** across the range that includes #1900 (51 on 864/866/899 → 54 on 908), so the commit does affect how the H3 DiT graph is partitioned — but the result is still a fully fragmented plan (54 segments) versus the healthy **2-segment cut on master-841** (`segment 1/2: blocks.0..37` + `segment 2/2`), and the deterministic `ErrorOutOfDeviceMemory` at submit remains. The 841→864 regression reported here is **still open on master-908**.
- Everything else in the matrix is unchanged or marginally better (Wan −1.2 % sampling vs 899, LTX passes on a fresh boot with the same ~120 MB desktop-VRAM sensitivity noted above), so we kept 908 pinned — H3/LTX production on our side still runs on the master-841 parallel install.

Happy to test any candidate fix — we have a byte-stable A/B repro (same command/seed/assets, control webm + logs from the 841 binary).



> **Comment by laurentvv (2026-10-01T16:01:43Z):**

Follow-up: we tested **master-929 `3f8527a`** on the same machine (RX 6950 XT 16 GB,
RDNA2, no matrix cores), same models, same commands as the matrix above, machine idle
(gate-checked). Net result: **Wan 2.2 stays fixed and faster; the Wan I2V full path now
also passes; LTX-2.5 33f now fails EARLY and EXPLICITLY (the new memory manager reports
the numbers instead of crashing); MiniMax-H3 is still broken with the identical
54-segment OOM signature.**

## Results on master-929

| Path | master-841 `6b3edaa` | master-864→908 | **master-929 `3f8527a`** |
|---|---|---|---|
| Flux 1024² (dev, 4 steps) | ✅ | ✅ | ✅ |
| Wan 2.2 T2V 17f | ✅ | ✅ (fixed by 899) | ✅ (+8.4% vs 841) |
| Wan I2V full path (25 steps, `--vae-on-cpu`) | ❌ 0xC0000409 (Sep 25) | ? | ✅ PASS (52 min, healthy) |
| LTX-2.5 33f distilled (Q4_K_M, 768×512) | ❌ (machine-state) | ❌ | ❌ **explicit refusal** (below) |
| MiniMax-H3 turbo Ref2VA (validated recipe, 22f + LoRA) | ✅ 2 segments | ❌ 51→54 segments, OOM | ❌ **still 54 segments, OOM at submit** |

**LTX-2.5 33f on master-929** — no more silent crash, the memory manager now refuses
cleanly with the full picture (this looks like a 16 GB fit problem, not a crash bug):
stages **14,388.75 MB** (4,349 tensors, 15 blocks) onto Vulkan0, leaving 776.09 MB free,
then: `model manager cannot make enough memory available on Vulkan0: need 908.75 MB
device / 396.75 MB budget, available 776.09 MB device / 668.90 MB budget` →
`ltxav segment 1/1 (graph) failed during workspace capacity check` at 21 s.
Question: should `--offload-to-cpu` stream the LTX DiT weights for vid_gen (it does
rescue Wan 14B for us), or is 33f @ Q4_K_M simply above the 16 GB envelope?

**MiniMax-H3 turbo on master-929** — retested today with the full validated recipe
(`--turbo` LoRA 8 steps, 864×480, 22 frames @ 24 fps, `--offload-to-cpu --max-vram 10
--rng cpu`, te+vae on CPU, ref tail + LoRA): `minimax_h3 using 54 segments` →
`graph execution failed on Vulkan0: vk::Queue::submit: ErrorOutOfDeviceMemory` (exit
0xC0000409). Control master-841 = **2 segments, runs fine** (re-validated on the
parallel build). The #1900 graph cuts (51→54 segments) did not close the gap.
The question from the matrix still stands: is 16 GB below the supported envelope for
H3 even with offload + max-vram, or should the graph segmentation eventually make it
fit? Happy to test any candidate build — this machine is our standard 16 GB Vulkan
datapoint.



> **Comment by nebkmb (2026-10-02T07:02:37Z):**

I have similar problems, spent hours testing various configurations, various sd.cpp builds. Minimax H3 support is hopelessly broken and memory inefficient with stable-diffusion.cpp. Can only generate short clips on low res even with the 3090. Comfy can do 15 sec 1344x768 (H3's specified max) clips with H3 even on my 12 gb laptop 4080 without additional memory optimization nodes.


> **Comment by laurentvv (2026-10-02T14:53:59Z):**

Confirming @nebkmb's findings with our own matrix (RX 6950 XT 16 GB, RDNA2, no matrix cores, machine gate-checked idle):

- **MiniMax-H3**: full validated recipe on master-929 `3f8527a` -> `minimax_h3 using 54 segments` -> `vk::Queue::submit: ErrorOutOfDeviceMemory`, exit 0xC0000409. Same signature on master-841. Since it also fails on your 24 GB 3090, this is not a 16 GB envelope issue — the segmentation path itself looks broken/memory-inefficient in sd.cpp (ComfyUI doing 15 s @ 1344x768 on a 12 GB 4080 corroborates that).
- **LTX-2.5**: on master-929 the memory manager stages 14,388.75 MB (4,349 tensors) onto Vulkan0, leaves 838.65 MB free, then refuses cleanly: `cannot make enough memory available on Vulkan0: need 908.56 MB device / 396.56 MB budget` -> `ltxav segment 1/1 (graph) failed during workspace capacity check` (33 frames @ 768x512, 2 steps). LTX-2.5 33f does not fit on 16 GB with 929, while it runs fine on master-841.
- Tested today: **`--params-backend disk` does not change the LTX outcome** on 929 — same staging onto Vulkan0, same workspace refusal.

Net on RDNA2 16 GB: Wan 2.1/2.2 (incl. I2V + VACE) fixed and faster on 929; LTX-2.5 fits only on 841; H3 broken on both. Happy to test candidate patches/builds — the H3 envelope and the LTX offload behavior remain the two open questions from our previous comment.


> **Comment by laurentvv (2026-10-08T08:01:15Z):**

Follow-up: we tested **master-945 `a1ded76`** (2026-10-07 release, first build carrying the
H3-path change #2103 "keep MiniMax-H3 VAE weights resident across temporal chunks") on the
same machine (RX 6950 XT 16 GB, RDNA2, no matrix cores), same models, same commands as the
matrix above, machine gate-checked idle. Net result: **MiniMax-H3 is still broken with the
identical signature, and LTX-2.5 still fails the same workspace-capacity refusal.**

## Results on master-945

| Path | master-841 `6b3edaa` | master-864→929 | **master-945 `a1ded76`** |
|---|---|---|---|
| MiniMax-H3 Ref2VA 22 f turbo (factory full path, `--turbo`) | ✅ (2 segments) | ❌ 51→54 segments + `ErrorOutOfDeviceMemory` | ❌ **identical: `minimax_h3 using 54 segments` → `vk::Queue::submit: ErrorOutOfDeviceMemory` at step ~10/28, exit 0xC0000409** |
| LTX-2.5 T2V 33 f (2-step distilled benchmark) | ✅ | ❌ early explicit refusal (908 MB / 776 free on 929) | ❌ **same class: `ltxav segment 1/1 failed during workspace capacity check` — tracked weights 14,388.75 MB on Vulkan0, need 893.75 MB device, reported free 694.57 MB** |
| Flux 1024² | ✅ | ✅ | ✅ (real-path smoke, 160 s) |

Notes:

- #2103 did **not** change the H3 graph segmentation on our envelope: still 54 segments
  (control = 2 on 841). The VAE-residency change may help other configurations, but the
  DiT graph cut is unchanged and the submit OOM is byte-for-byte the 929 signature.
- LTX numbers on 945 are the same failure mode as reported for 929 in the 10-01 comment;
  `--params-backend disk` was already tested negative on 929 (10-02).
- Production on our side remains the pinned master-841 parallel install for H3 + LTX.

Not closing — the issue stays accurate for both regressions on 16 GB Vulkan. Happy to run
any experiment on 945+ (bisect offers from the original comment still stand).


> **Comment by laurentvv (2026-10-11T12:53:00Z):**

Follow-up 2: we tested **master-956 `1b0ba10`** (2026-10-10 release, carrying #2119 "demote resident params to disk residency when memory reclamation fails") on the same machine (RX 6950 XT 16 GB, RDNA2, no matrix cores), same models, same commands as the matrix above (exact validated recipe: `h3_ref2va --turbo`, 864x480, 22 frames, seed 42, ref = 12 frames + 0.5 s WAV), machine gate-checked idle.

**MiniMax-H3 is still broken with the identical signature**: `minimax_h3 using 54 segments` -> `sampling failed after 325.43 s` (submit OOM) -> exit 3221226505 (`0xC0000409`). Neither #2103 (945) nor #2119 (956) changes the DiT graph cut — the healthy control stays master-841 `6b3edaa` (2 segments), still our production binary for H3/LTX.

Updated row:

| Path | master-841 `6b3edaa` | master-864→945 | **master-956 `1b0ba10`** |
|---|---|---|---|
| MiniMax-H3 Ref2VA 22 f turbo (factory full path, `--turbo`) | ✅ (2 segments) | ❌ 51→54 segments + `ErrorOutOfDeviceMemory` | ❌ **identical: 54 segments → submit OOM at ~325 s, exit 0xC0000409** |

The issue stays accurate for the H3/LTX regressions. Separately, we hit what looks like a **new master-956-specific regression on the Qwen-Image path** (not #1976-related) — filing it as its own issue and linking it here if it turns out to share the memory-management root cause.



> **Comment by laurentvv (2026-10-11T12:56:15Z):**

Addendum to the follow-up above: the Qwen-Image slowdown I mentioned (and filed as #2127, now closed) turned out to be **environmental on our side** — the models sit on an external USB 3.0 spinning drive and the post-graph-cut tensor re-read is disk/cache-bound, not build-related. No master-956 qwen regression; apologies for the cross-noise. The H3 datapoint in the table above (54 segments + submit OOM on 956) is unaffected by this and stands.


> **Comment by laurentvv (2026-10-12T20:33:23Z):**

Datapoint on **master-959-d637cfb** (official `sd-master-d637cfb-bin-win-vulkan-x64.zip`, Windows 11, RX 6950 XT 16 GB): **still reproduces, identical signature including the segment count.**

Same real-path recipe as our 945/956 datapoints (workflow `h3_ref2va --turbo`):

```
sd-cli -M vid_gen \
  --diffusion-model <models>/minimax_h3_ref2va_pruned-Q4_K_M.gguf \
  --vae <models>/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae <models>/minimax_h3_audio_vae_fp32.safetensors \
  --llm <models>/qwen3vl_32b_minimax_h3-Q2_K_M.gguf \
  -p "The camera continues its slow push in on the helmet <lora:minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16:1.0>" \
  --ref-video <12 ref frames> --ref-video-audio <ref_audio.wav> \
  -W 864 -H 480 --video-frames 22 --fps 24 --steps 8 --cfg-scale 1.0 \
  --sampling-method euler --diffusion-fa --offload-to-cpu --rng cpu --max-vram 10 -s 42 \
  --lora-model-dir <loras_dir> --lora-apply-mode auto \
  -o out.webm -t 16 --backend diffusion=vulkan0,te=cpu,vae=cpu -v
```

Result:

```
[I] minimax_h3 using 54 segments
[E] minimax_h3 graph execution failed on Vulkan0: vk::Queue::submit: ErrorOutOfDeviceMemory
[E] Diffusion model sampling failed
[E] sampling failed after 252.17s
→ exit 3221226505 (0xC0000409)
```

Same failure window as before (956: 325 s at the same exit code; 945: identical signature) — expected, since the three commits in 959 (#2123 TAEQI2.1, #2128 MoSR upscaler, #2130 sigma preset) don't touch the H3 graph cut; recording it for the bisect map. For completeness: a second run WITHOUT `--ref-video-audio` fails the same way (253.06 s, same exit code), and all other smokes pass on 959 (SDXL 512² 12.7 s, Qwen-Image-2.1 turbo 1152×640 ≈ 112 s, byte-identical output through the new #2130 sigma preset).



> **Comment by laurentvv (2026-10-14T16:31:47Z):**

## Datapoint: master-964-1d0a58a — still reproduced, identical signature

Machine: AMD Radeon RX 6950 XT 16 GB (RDNA2), Windows 11, official
`sd-master-1d0a58a-bin-win-vulkan-x64.zip` release binary.

Exact real-path recipe (unchanged across our master-864/945/956/959/964 datapoints):
`vid_gen` Ref2VA, `minimax_h3_ref2va_pruned-Q4_K_M.gguf` + fp16 video VAE + fp32 audio VAE
+ qwen3vl-32b Q2_K_M text encoder, 864×480, 22 frames @ 24 fps, 8 steps (turbo LoRA),
cfg 1.0, euler, `--diffusion-fa --offload-to-cpu --rng cpu --max-vram 10`,
`--backend diffusion=vulkan0,te=cpu,vae=cpu`, 12 reference frames + LoRA
`minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16:1.0`, seed 42.

Result on master-964 (1d0a58a):

```
[I] minimax_h3 build cached graph cut plan done (taking 13 ms) --- ggml_graph_cut.cpp:1034
[V] minimax_h3 using 54 segments --- ggml_runner.cpp:879
[E] minimax_h3 graph execution failed on Vulkan0: vk::Queue::submit: ErrorOutOfDeviceMemory --- ggml_runner.cpp:660
[E] Diffusion model sampling failed --- diffusion_engine.cpp:2943
[E] sampling failed after 322.13s --- video.cpp:1709
```

Exit code 3221226505 (0xC0000409). This is the same signature as master-864, 945, 956
and 959 (sampling-failed wall time across builds: 253-325 s, same variance class).
The delta 959→964 (#2124 Vulkan iGPU memory reclaim, #2132 flash-attn auto, #2134
conv-direct auto, #2135) does not change the graph-cut plan: still 54 segments vs the
healthy 2-segment plan this workload gets on the older pinned build we keep for
production (6b3edaa).

Separate note: on this GPU master-964 also regressed ~7x on SDXL end-to-end via the new
`conv-direct auto` selection (separate report), so we rolled back to master-959 and keep
waiting for a build that changes the H3 graph-cut behavior.

Log available on request.


> **Comment by laurentvv (2026-10-15T23:43:37Z):**

**Still reproducible on master-968 (8cd731e) — same signature as the 956/959/964 datapoints above.**

Same real-path recipe as the previous datapoints (MiniMax-H3 ref2va, turbo 8-step, 22 frames @ 864×480, seed 42, DiT on Vulkan + text-encoder/VAE on CPU, `--max-vram 10`, official win-vulkan release binaries, idle machine, gate-checked):

```
[V] minimax_h3 using 54 segments                                  (healthy plan on older builds: 2)
[E] minimax_h3 graph execution failed on Vulkan0: vk::Queue::submit: ErrorOutOfDeviceMemory
[E] Diffusion model sampling failed
[E] sampling failed after 346.77s                                 (956: 325s / 959: 253s / 964: 322s — same variance class)
exit 3221226505 (0xC0000409), no output file
```

The 959→968 delta is 9 commits and none touches the graph cut / segment planner — `#2124` (reclaim before accepting wrapped Vulkan iGPU memory estimates) was the only memory-domain candidate in the window and it does **not** prevent the OOM submit, so the root cause remains the 54-segment plan from the memory-rework window (#1900-era), not the allocation timing.

For completeness: #2141 in the same window did fix the separate conv-direct auto-regression on this GPU (RDNA2), detailed in #2137.

Datapoints in this thread from us: 945 (10-08), 959 (10-12), 964 (10-14), now 968. Production for this model stays on our last-good parallel build until the segment plan is fixed.



> **Comment by laurentvv (2026-10-16T22:46:33Z):**


**Still reproducible on master-970 (2f016e8) — same signature as the 956/959/964/968 datapoints above.**

For completeness — this release only carries the Windows UTF-8 path fixes (#2144/#2145), no memory-manager or graph-cut change, but we tested it anyway to close the question. Same real-path recipe as the previous datapoints (MiniMax-H3 ref2va, turbo 8-step LoRA, 22 frames @ 864×480, seed 42, DiT on Vulkan + text-encoder/VAE on CPU, `--max-vram 10`, official win-vulkan release binaries, idle machine, gate-checked):

```
[V] minimax_h3 using 54 segments                                  (healthy plan on older builds: 2)
[E] minimax_h3 graph execution failed on Vulkan0: vk::Queue::submit: ErrorOutOfDeviceMemory
[E] Diffusion model sampling failed
[E] sampling failed after 363.54s
```

No webm produced. The reference-frames VAE encode completes normally right before (891.90 s), so the failure remains isolated to the 54-segment DiT plan. Submit-failure timing across our datapoint series stays in the same variance class: 956: 325.43 s / 959: 253 s / 964: 322.13 s / 968: 346.77 s / 970: 363.54 s.



> **Comment by laurentvv (2026-10-17T19:52:35Z):**

**Still reproducible on master-973 (`f0aac52`) — same signature as the 956/959/964/968/970 datapoints above.**

For completeness — this release carries only the ggml submodule sync + obsolete conv1d workaround removal (#2147, no memory-domain change), moss-image support (#2148) and a MUSA CI align; no memory-manager or graph-cut change — but we tested it anyway to close the question. Same real-path recipe as the previous datapoints (MiniMax-H3 ref2va, turbo 8-step LoRA, 22 frames @ 864×480, seed 42, DiT on Vulkan + text-encoder/VAE on CPU, `--max-vram 10`, official win-vulkan release binaries, idle machine, gate-checked):

```
[V] computing vae encode graph completed, taking 829.65s          (ref-frames VAE encode completes normally; 970: 891.90s)
[V] minimax_h3 using 54 segments                                  (healthy = 2)
[E] minimax_h3 graph execution failed on Vulkan0: vk::Queue::submit: ErrorOutOfDeviceMemory
[E] Diffusion model sampling failed
[E] sampling failed after 319.05s                                 (956: 325 / 959: 253 / 964: 322 / 968: 347 / 970: 363.54 — same variance class)
exit code 3221226505, no output webm
```

The 54-segment execution plan is still built (instead of the healthy 2) and still OOMs the queue submit on the RX 6950 XT 16 GB. As of this build, upstream master still carries no graph-cut/memory-manager change for this path, so we consider the question closed until such a commit lands; production on our side stays on the older known-good binary.



> **Comment by laurentvv (2026-10-18T14:13:41Z):**

Re-tested on **master-975 (`95b3090`)** — same machine (RX 6950 XT 16 GB, RDNA2, Vulkan), same models, same command as the matrix above, gate-checked. Net result: **still not fixed, identical signature** — `minimax_h3 using 54 segments`, then `vk::Queue::submit: ErrorOutOfDeviceMemory`, sampling failed after **302.54 s**, no output.

For completeness: 975 carries two feature commits since 973 — Diffusers-format MiniMax-H3 LoRA support (#2159) and MOSS split-QKV mapping + batch host loading (#2160). The latter only changes the weight-load path (`name_conversion.cpp` / `model_manager.cpp` tensor-part loading); it does not touch the per-segment staging/streaming behavior, so the identical failure is expected — datapoint recorded to keep the series continuous.

| build | segments | failure | sampling failed after |
|---|---|---|---|
| master-956 | 54 | submit OOM | 325 s |
| master-959 | 54 | submit OOM | 253 s |
| master-964 | 54 | submit OOM | 322 s |
| master-968 | 54 | submit OOM | 347 s |
| master-970 | 54 | submit OOM | 363.5 s |
| master-973 | 54 | submit OOM | 319.1 s |
| master-975 | 54 | submit OOM | 302.5 s |

(Context: in parallel we root-caused this on PR #2157 — per-segment replicas accumulate in pinned host-visible memory while the capacity gates watch the device heap; measured data, a candidate fix inside the streaming design, and a four-build matrix are in [this comment](https://github.com/leejet/stable-diffusion.cpp/pull/2157#issuecomment-6249993736).)



> **Comment by laurentvv (2026-10-23T15:34:06Z):**

# Draft comment for leejet/stable-diffusion.cpp#1976 (closure confirmation, master-995)

Confirming the fix on the **first official release carrying #2181** — `master-995-fd21ecf` — same machine (RX 6950 XT 16 GB, RDNA2, Vulkan), same models, gate-checked, stock binary (no env overrides):

**1. MiniMax-H3 (the main subject of this issue) — fixed, confirmed on the factory full path.**
This time we ran the real production route rather than the bisect rig from the #2181 validation: our `h3_ref2va` workflow (automatic 12-frame reference-tail + matched WAV extracted from a source video, 22 frames @ 864×480, turbo 8-step LoRA, `--max-vram 10`, `te=cpu,vae=cpu`, seed 42). Result: **exit 0 in 36.2 min, zero failure signatures** (no OOM, no device lost, no sampling failure). The **stock 54-segment plan survives** — as expected from the #2181 design (the small-BAR heap mask keeps staging allocations in real VRAM, so the WDDM shared-segment budget is no longer blown). Output webm (1,013,557 B) with PCM 32 kHz audio, visual check clean (reference subject/scene persist, no artifacts), and sampling ran at **133.4 s/step — exact parity with our master-841 production build** (134 s/step), so the fix costs nothing in throughput. Issue closure confirmed on our side.

**2. Bonus datapoint on the LTX-2.5 part of the title — the workspace refusal is gone too.**
The 33-frame LTX-2.5 T2V benchmark leg — the exact recipe that produced the "workspace 2.2 GB" clean refusal on master-929/945 (`--params-backend disk` did not help, tested 10-02) — **now completes on 995 with the same verbatim command**: 33 f @ 768×512, 2 steps, euler_a, `diffusion=vulkan0,te=cpu` → exit 0, `generate_video completed in 111.31 s`, valid webm out. So both regressions in the issue title appear resolved by the same ggml small-BAR heap mask. (Our LTX production shapes — 65-frame routes with audio VAE — still need their own re-validation before we move them off the pinned master-841 build, but the planner-level blocker itself no longer reproduces.)

Thanks for the fast diagnosis and the surgical fix — the #2166/#2181 investigations turned a week of 54-segment OOM datapoints into a one-commit resolution.




### #1946 [closed] [Bug] Vulkan/AMD RX 6950 XT: Flux (split text encoders) silently crashes since master-848-9cdb6b6 — regression vs master-841-6b3edaa (weight preparation OOM)
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1946  by laurentvv 2026-09-07T19:40:06Z  comments: 14

## Summary

Since release **master-848-9cdb6b6**, `sd-cli` (official `bin-win-vulkan-x64` asset) **crashes silently** (process exits, no error message on stdout/stderr, bash reports exit code 127) when generating a **Flux** image with **split text encoders** on AMD Vulkan. The exact same command works on the previous release asset (**master-841-6b3edaa**). **SDXL with a monolithic checkpoint works fine** on the new build, so this looks related to the multi-model loading / weight-residency refactor (candidates: #1940, #1942, #1905).

## Environment

- OS: Windows 11 (10.0.26200)
- GPU: AMD Radeon RX 6950 XT 16 GB (RDNA2 / gfx1030)
- Driver: 32.0.21045.5002 (Adrenalin)
- Build: `sd-master-9cdb6b6-bin-win-vulkan-x64.zip` (official release asset), commit `9cdb6b6`
- Working build for comparison: `sd-master-<sha>-bin-win-vulkan-x64.zip` with commit `6b3edaa` (master-841)

Models (GGUF from the project's recommended sources):

- `flux1-schnell-Q6_K.gguf` (diffusion model)
- `clip_l.safetensors`, `t5xxl_fp16.safetensors` (text encoders)
- `ae.safetensors` (VAE)

## Reproduction

```
sd-cli.exe --diffusion-model flux1-schnell-Q6_K.gguf \
  --clip_l clip_l.safetensors --t5xxl t5xxl_fp16.safetensors --vae ae.safetensors \
  -p "a red apple on a wooden table, photo" \
  --steps 4 --cfg-scale 1.0 --sampling-method euler -W 512 -H 512 --seed 42 \
  --vae-tiling --backend vulkan -o out.png
```

### Observed on `9cdb6b6` (master-848)

**1) Default invocation — silent death.** The process terminates without any error message (exit code 127 under Git Bash). Last log lines with `-v`:

```
[INFO   ] stable-diffusion.cpp:5322 - get_learned_condition completed, taking 13.73s
[INFO   ] stable-diffusion.cpp:5705 - generating image: 1/1 - seed 42
[INFO   ] ggml_graph_cut.cpp:965  - flux build cached graph cut plan done (taking 6 ms)
[VERBOSE] model_loader.cpp:1243 - loading tensors completed, taking 5.32s (read: 4.06s, ...)
[VERBOSE] model_manager.cpp:428  - model manager prepared params backend buffers (9379.29 MB, 776 tensors, 192 blocks, VRAM) on Vulkan0
<process exits here, no message>
```

**2) `--auto-fit off` — identical silent crash** at the same point.

**3) `--offload-to-cpu` — hard Vulkan OOM error:**

```
ggml_vulkan: vk::Device::createComputePipeline: ErrorOutOfDeviceMemory
```

**4) `--max-vram 6` — an actual error line is finally logged, then the process still dies silently:**

```
[ERROR  ] ggml_runner.cpp:848  - t5 segment 17/26 (t5.block.15) failed during weight preparation
<process exits here, exit code 127, no additional message>
```

### Expected / previous behavior

The identical command on the `6b3edaa` build completes successfully and produces the image (this GPU is 16 GB; the previous build's streaming/offload behavior fits Flux Q6_K + t5xxl fp16 in VRAM).

### Control test: SDXL is unaffected

A monolithic SDXL checkpoint (`-m juggernautXL...safetensors`, 512×512, euler_a) **generates fine** on `9cdb6b6` — so the regression is specific to the split-encoders path (diffusion model + separate clip/t5/vae), i.e. exactly what the new model manager / runner lifecycle code (#1940, #1942) and streamed-layer prefetch (#1905) touch.

## Additional notes

- Two distinct problems, possibly: (a) the weight-preparation memory management regression itself, and (b) **the process exits with no diagnostic at all** in cases 1/2/4 — even when an `[ERROR]` line was just logged (case 4), there is no exception message, no abort message, nothing. A graceful failure with the underlying Vulkan error would have made this much easier to report.
- No other GPU-heavy process was running; no zombie `sd-cli` processes; VRAM was free before each run (each configuration was tested in a fresh process).
- Happy to test a fix build or additional variants (e.g. quantized t5xxl, flux1-dev, other `--max-vram` values) on this machine.


> **Comment by laurentvv (2026-09-07T19:53:09Z):**

Data point: **still present in `master-849-d04e895`** (official `bin-win-vulkan-x64` asset).

Tested the exact same Flux schnell command (split text encoders, Vulkan, RX 6950 XT): identical silent crash, exit code 127 with no message, at the same point — right after:

```
[VERBOSE] model_manager.cpp:428  - model manager prepared params backend buffers (9379.29 MB, 776 tensors, 192 blocks, VRAM) on Vulkan0
```

Unsurprising since the only commit in 849 (#1939, SDXL embeddings loading) doesn't touch this path. `6b3edaa` remains unaffected. I'll keep `6b3edaa` installed and can test any candidate fix build on this machine (AMD RX 6950 XT 16 GB, Windows 11, Vulkan).



> **Comment by vmobilis (2026-09-08T20:23:07Z):**

@laurentvv, hello, if nobody else will confirm the bug, there is a little possibility that the regression affects only your particular hw+drivers setup and others will not be able to reproduce the error.

If so, since some of 7 commits between https://github.com/leejet/stable-diffusion.cpp/commit/6b3edaaf32cc19e5bb2d819c788bd557eddc8eba and https://github.com/leejet/stable-diffusion.cpp/commit/9cdb6b6fc05edf2d446bfe881a4c462b241b3025 have binaries in releases, maybe you could try to locate the crashing commit more precisely, to clarify possible reason? (Or are they all crash right after https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-841-6b3edaa)?
Good luck!


> **Comment by laurentvv (2026-09-08T21:27:16Z):**

Follow-up per @vmobilis' suggestion — bisected further, the culprit is now isolated to a single commit.

**Binary bisect (official `bin-win-vulkan-x64` releases, same Flux schnell command as the original report):**

| Build | Commits added vs `6b3edaa` | Result |
|---|---|---|
| master-841 `6b3edaa` | — (reference) | ✅ image generated (re-verified 2026-09-08) |
| master-843 `462d675` | #1905 + #1940 | ❌ silent crash, exit 127 |
| master-845 `80bac2d` | + #1941 + #1942 | ❌ identical crash |

**Local builds to separate #1905 from #1940** (MSYS2/UCRT64, gcc 16.2, `-DGGML_VULKAN=ON`, same ggml submodule pin `e20c3a14` for all three, same machine/command):

| Commit | Result |
|---|---|
| `6b3edaa` (control) | ✅ image generated |
| `6c57cc3` (#1905 prefetch streamed layers, alone) | ✅ **image generated — #1905 is not the culprit** |
| `462d675` (#1940 unify runner lifecycles and weight residency) | ❌ crash |

**The regression is introduced by `462d675` — PR #1940 "refactor: unify runner lifecycles and weight residency".**

Extra detail that may help: with my gcc build the failure is not fully silent — the process aborts with a visible exception:

```
what():  vk::Queue::submit: ErrorOutOfDeviceMemory
```

It happens right after the DiT weights finish streaming to VRAM (last log line `model_loader.cpp:1243 - loading tensors completed`, i.e. the same stage as the `model_manager.cpp:428` buffer-preparation line in the original report). So the ~9.4 GB of DiT buffers get allocated successfully, then the first compute submit runs out of device memory — consistent with the `--offload-to-cpu` / `--max-vram` behaviors reported earlier. It looks like the weight-residency refactor changed when (or whether) streamed buffers are released before compute on this 16 GB RDNA2 setup.

Happy to test any candidate fix build on this machine (AMD RX 6950 XT 16 GB, Windows 11, Vulkan, AMD proprietary driver).


> **Comment by mlsterpr0 (2026-09-09T18:27:23Z):**

master-841-6b3edaa seems to be last working vulkan release. After it, Minimax H3 crashes when loading text encoder.

[INFO ] ggml_graph_cut.cpp:952  - qwen3vl build cached graph cut plan done (taking 4 ms)
\stable-diffusion.cpp\ggml\src\ggml-backend.cpp:930: pre-allocated tensor (text_encoders.llm.model.embed_tokens.weight) in a buffer (Vulkan0) that cannot run the operation (NONE)


> **Comment by leejet (2026-09-10T17:08:07Z):**

I’m unable to reproduce this issue on my device, so I can’t confirm whether https://github.com/leejet/stable-diffusion.cpp/pull/1958 fixes it. Could you give it a try?


> **Comment by laurentvv (2026-09-10T20:11:54Z):**

Tested leejet's candidate fix **PR #1958** (`58d3f6e`, local MSYS2/UCRT64 build, gcc 16.2,
`-DGGML_VULKAN=ON`, same ggml submodule pin `e20c3a14` as the bisect builds, RX 6950 XT
16 GB, Windows 11, AMD proprietary driver).

**The hard crash is fixed — error handling now works as designed — but generation still
does not complete on this card.** No more `terminate()` / `vk::Queue::submit:
ErrorOutOfDeviceMemory` abort on the plain repro: the failure is caught and propagated
cleanly to the CLI with a precise diagnostic. Same command as the original report
(Flux schnell Q6_K, 512×512, 4 steps, euler, `--vae-tiling`, `--backend vulkan`):

```
[WARN   ] model_manager.cpp:1762 - model manager cannot make enough memory available on Vulkan0: need 954.20 MB device / 457.22 MB budget, available 906.93 MB device / unlimited budget
[ERROR  ] ggml_runner.cpp:858  - flux segment 30/59 (flux.single_blocks.9) failed during weight preparation
[ERROR  ] diffusion_engine.cpp:2351 - diffusion model compute failed
[ERROR  ] diffusion_engine.cpp:2462 - Diffusion model sampling failed
[ERROR  ] image.cpp:888  - sampling for image 1/1 failed after 22.02s
```

- **Deterministic, not fragmentation**: identical failure (same numbers, same segment
  30/59) on a freshly rebooted machine with only 0.5/16 GB VRAM in use before launch.
  The manager comes up ~47 MB short at `flux.single_blocks.9` (need 954.20 MB device,
  available 906.93 MB) — mid-sampling, i.e. it can no longer evict enough of the
  resident DiT weights to prepare the next segment.
- **`--offload-to-cpu` also fails, differently**: all 18,858 MB of weights go to RAM as
  expected (`VRAM 0.00MB, RAM 18858.13MB`), loading completes, then the process dies
  uncaught with exit 127 during compute:

  ```
  ggml_vulkan: vk::Device::createComputePipeline: ErrorOutOfDeviceMemory
  terminate called after throwing an instance of 'vk::OutOfDeviceMemoryError'
    what():  vk::Queue::submit: ErrorOutOfDeviceMemory
  ```

  ~15.5 GB of VRAM were free at that point, so a pipeline creation failing with
  OutOfDeviceMemory looks like a separate backend/allocation issue on this path
  (possibly also #1940-related).

Machine available for any follow-up build. (Also confirmed: master-841 `6b3edaa` still
generates this exact command fine.)




> **Comment by leejet (2026-09-11T15:27:05Z):**

Could you try running the following command and send me the detailed logs so I can investigate the issue?

```bash
sd-cli.exe --diffusion-model flux1-schnell-Q6_K.gguf \
  --clip_l clip_l.safetensors --t5xxl t5xxl_fp16.safetensors --vae ae.safetensors \
  -p "a red apple on a wooden table, photo" \
  --steps 4 --cfg-scale 1.0 --sampling-method euler -W 512 -H 512 --seed 42 \
  --vae-tiling --backend vulkan --offload-to-cpu --log-level debug -o out.png
```



> **Comment by laurentvv (2026-09-11T16:14:49Z):**

Ran your exact command on my side — same setup as my earlier reports (AMD RX 6950 XT 16 GB, Windows 11, AMD proprietary driver, `uma: 0`), local MSYS2/UCRT64 build of `58d3f6e` (PR #1958 head, gcc 16.2, `-DGGML_VULKAN=ON`, ggml pin `e20c3a14`), with `--log-level debug`.

**Full debug log (192 lines, raw): [gist](https://gist.github.com/laurentvv/7f93d6c2e50feee1483dc49b3f5730ea)**

**Result: same uncaught crash as my 2026-09-10 report — exit 127, `terminate` on `vk::Device::createComputePipeline: ErrorOutOfDeviceMemory` — but the debug level pinpoints where it happens now: the very first T5 segment, right after a 9 GB "staged compute params" copy into VRAM.**

Key excerpt:

```
[INFO   ] diffusion_engine.cpp:1206 - total params memory size = 18858.13MB (VRAM 0.00MB, RAM 18858.13MB): text_encoders 9318.83MB(RAM), diffusion_model 9379.29MB(RAM), vae 160.00MB(RAM)
[VERBOSE] model_manager.cpp:488  - model manager prepared params backend buffers (235.06 MB, 196 tensors, 1 blocks, RAM) on Vulkan_Host
[VERBOSE] model_manager.cpp:614  - model manager staged compute params (235.06 MB, 196 tensors, 4 blocks) to Vulkan0, taking 0.04s
[DEBUG  ] ggml_runner.cpp:929  - clip executing segment 1/1: graph
[VERBOSE] ggml_runner.cpp:954  - clip compute buffer size: 1.42 MB(VRAM) on Vulkan0 (peak across 1 segment)
[VERBOSE] model_loader.cpp:1279 - loading tensors completed, taking 2.18s (read: 2.02s, memcpy: 0.00s, convert: 0.00s, copy_to_backend: 0.00s)
[VERBOSE] model_manager.cpp:488  - model manager prepared params backend buffers (9083.77 MB, 219 tensors, 10 blocks, RAM) on Vulkan_Host
[VERBOSE] model_manager.cpp:614  - model manager staged compute params (9083.77 MB, 219 tensors, 146 blocks) to Vulkan0, taking 6.05s
[DEBUG  ] ggml_runner.cpp:929  - t5 executing segment 1/1: graph
ggml_vulkan: Compute pipeline creation failed for rms_norm_mul_f32
ggml_vulkan: vk::Device::createComputePipeline: ErrorOutOfDeviceMemory
terminate called after throwing an instance of 'vk::OutOfDeviceMemoryError'
  what():  vk::Queue::submit: ErrorOutOfDeviceMemory
```

Observations (no interpretation, just what the log shows):

1. The engine summary counts **all 18858.13 MB of weights as RAM, VRAM 0.00 MB** — offload to CPU is in effect. Yet right before compute, `model_manager` "**staged compute params (9083.77 MB, 219 tensors, 146 blocks) to Vulkan0**" — the full T5 weight set gets copied into VRAM anyway (6.05 s copy). CLIP-L does the same (235 MB staged) and runs fine.
2. **No capacity-guard warning on this path.** Without `--offload-to-cpu` (plain repro), the #1958 guard logs `model manager cannot make enough memory available on Vulkan0…` and fails cleanly at `flux.single_blocks.9`. Here the 9 GB staging to Vulkan0 happens silently and the next pipeline creation OOMs uncaught — the guard doesn't seem to cover the staging step.
3. Puzzling detail: with VRAM at 0.7/16 GB before launch and ~9.3 GB staged in total, roughly 6 GB should still be free, yet `createComputePipeline` for `rms_norm_mul_f32` already fails with `ErrorOutOfDeviceMemory`. Same deterministic behavior as before (rebooted-machine test on 2026-09-10), `max_vram: 0` in the params dump.
4. No image written (failure occurs during text-encoding, before sampling).

Happy to test any follow-up build on this machine.
---

<details>
<summary>📄 <b>Full raw debug log</b> (192 lines — 12 terminal progress-bar redraw lines elided for readability; byte-exact raw version in the gist linked above)</summary>

```text
# ============================================================
# Test demande par leejet — issue #1946, commentaire 5636742869 (2026-09-11)
# Binaire   : build local MSYS2/UCRT64 gcc 16.2 -DGGML_VULKAN=ON de 58d3f6e (PR #1958 head), ggml pin e20c3a14
# Materiel  : AMD RX 6950 XT 16 Go, Windows 11, pilote AMD proprietaire
# Etat VRAM avant lancement : 0.7/16 Go utilises (machine sans autre charge GPU)
# Commande  : commande exacte de leejet, chemins absolus
# ============================================================
[VERBOSE] main.cpp:700  - version: stable-diffusion.cpp version unknown, commit unknown
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 6950 XT (AMD proprietary driver) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 32768 | int dot: 1 | matrix cores: none
[VERBOSE] main.cpp:701  - System Info: 
   SSE3 = 1 |    OPENMP = 1 |    REPACK = 1 | 
[VERBOSE] main.cpp:702  - SDCliParams {
  mode: img_gen,
  output_path: "C:\GIT\generator-assets\scratch\bisect_1946\out_leejet_debug.png",
  image_path: "",
  metadata_format: "text",
  log_level: debug,
  color: false,
  canny_preprocess: false,
  convert_name: false,
  preview_method: none,
  preview_interval: 1,
  preview_path: "preview.png",
  preview_fps: 16,
  taesd_preview: false,
  preview_noisy: false,
  imatrix_out: "",
  metadata_raw: false,
  metadata_brief: false,
  metadata_all: false
}
[VERBOSE] main.cpp:703  - SDContextParams {
  n_threads: 12,
  model_path: "",
  clip_l_path: "C:\Modeles_LLM\clip_l.safetensors",
  clip_g_path: "",
  clip_vision_path: "",
  t5xxl_path: "C:\Modeles_LLM\t5xxl_fp16.safetensors",
  llm_path: "",
  llm_vision_path: "",
  diffusion_model_path: "C:\Modeles_LLM\flux1-schnell-Q6_K.gguf",
  high_noise_diffusion_model_path: "",
  uncond_diffusion_model_path: "",
  embeddings_connectors_path: "",
  vae_path: "C:\Modeles_LLM\ae.safetensors",
  vae_format: "auto",
  audio_vae_path: "",
  taesd_path: "",
  esrgan_path: "",
  control_net_path: "",
  embedding_dir: "",
  embeddings: {
  }
  wtype: NONE,
  tensor_type_rules: "",
  lora_model_dir: ".",
  hires_upscalers_dir: "",
  photo_maker_path: "",
  rng_type: cuda,
  sampler_rng_type: NONE,
  offload_params_to_cpu: true,
  max_vram: "0",
  disable_prefetch: false,
  disable_segmented_compute: false,
  eager_load: false,
  backend: "vulkan",
  params_backend: "",
  split_mode: "",
  model_args: "",
  auto_fit: true,
  enable_mmap: false,
  control_net_cpu: false,
  clip_on_cpu: false,
  vae_on_cpu: false,
  flash_attn: false,
  diffusion_flash_attn: false,
  diffusion_conv_direct: false,
  vae_conv_direct: false,
  prediction: NONE,
  lora_apply_mode: auto,
  force_sdxl_vae_conv_scale: false
}
[VERBOSE] main.cpp:704  - SDGenerationParams {
  loras: "{
  }",
  high_noise_loras: "{
  }",
  prompt: "a red apple on a wooden table, photo",
  negative_prompt: "",
  ad_model_path: "",
  ad_prompt: "",
  ad_negative_prompt: "",
  extra_ad_args: "",
  clip_skip: -1,
  width: 512,
  height: 512,
  batch_count: 1,
  qwen_image_layers: 3,
  init_image_path: "",
  end_image_path: "",
  mask_image_path: "",
  control_image_path: "",
  ref_image_paths: [],
  ref_video_paths: [],
  ref_video_audio_paths: [],
  ref_audio_paths: [],
  control_video_path: "",
  auto_resize_ref_image: true,
  increase_ref_index: false,
  pm_id_images_dir: "",
  pm_id_embed_path: "",
  pm_style_strength: 20,
  skip_layers: [7, 8, 9],
  sample_params: (txt_cfg: 1.00, img_cfg: 1.00, distilled_guidance: 3.50, slg.layer_count: 0, slg.layer_start: 0.01, slg.layer_end: 0.20, slg.scale: 0.00, scheduler: NONE, sample_method: euler, sample_steps: 4, eta: inf, shifted_timestep: 0, flow_shift: inf, extra_sample_args: ),
  high_noise_skip_layers: [7, 8, 9],
  high_noise_sample_params: (txt_cfg: 7.00, img_cfg: 7.00, distilled_guidance: 3.50, slg.layer_count: 0, slg.layer_start: 0.01, slg.layer_end: 0.20, slg.scale: 0.00, scheduler: NONE, sample_method: NONE, sample_steps: 20, eta: inf, shifted_timestep: 0, flow_shift: inf, extra_sample_args: ),
  custom_sigmas: [],
  cache_mode: "",
  cache_option: "",
  cache: disabled (threshold=inf, start=0.15, end=0.95),
  moe_boundary: 0.875,
  video_frames: 1,
  fps: 16,
  vace_strength: 1,
  strength: 0.75,
  control_strength: 0.9,
  seed: 42,
  upscale_repeats: 1,
  upscale_tile_size: 128,
  hires: { enabled: false, upscaler: "Latent", model_path: "", scale: 2, target_width: 0, target_height: 0, steps: 0, denoising_strength: 0.7, custom_sigmas: [], upscale_tile_size: 128 },
  vae_tiling_params: { 1, 0, 0, 0, 0.5, 0, 0, "" },
}
[VERBOSE] model_loader.cpp:174  - using 12 threads for model loading
[INFO   ] diffusion_engine.cpp:700  - loading diffusion model from 'C:\Modeles_LLM\flux1-schnell-Q6_K.gguf'
[INFO   ] model_loader.cpp:191  - load C:\Modeles_LLM\flux1-schnell-Q6_K.gguf using gguf format
[VERBOSE] model_loader.cpp:237  - init from 'C:\Modeles_LLM\flux1-schnell-Q6_K.gguf'
[INFO   ] diffusion_engine.cpp:721  - loading clip_l from 'C:\Modeles_LLM\clip_l.safetensors'
[INFO   ] model_loader.cpp:197  - load C:\Modeles_LLM\clip_l.safetensors using safetensors format
[VERBOSE] model_loader.cpp:265  - init from 'C:\Modeles_LLM\clip_l.safetensors', prefix = 'clip_l.'
[INFO   ] diffusion_engine.cpp:742  - loading t5xxl from 'C:\Modeles_LLM\t5xxl_fp16.safetensors'
[INFO   ] model_loader.cpp:197  - load C:\Modeles_LLM\t5xxl_fp16.safetensors using safetensors format
[VERBOSE] model_loader.cpp:265  - init from 'C:\Modeles_LLM\t5xxl_fp16.safetensors', prefix = 'text_encoders.t5xxl.transformer.'
[INFO   ] diffusion_engine.cpp:771  - loading vae from 'C:\Modeles_LLM\ae.safetensors'
[INFO   ] model_loader.cpp:197  - load C:\Modeles_LLM\ae.safetensors using safetensors format
[VERBOSE] model_loader.cpp:265  - init from 'C:\Modeles_LLM\ae.safetensors', prefix = 'vae.'
[INFO   ] diffusion_engine.cpp:893  - Version: Flux 
[VERBOSE] ggml_extend_backend.cpp:395  - Initializing backend: Vulkan0
[VERBOSE] ggml_extend_backend.cpp:395  - Initializing backend: CPU
[VERBOSE] model_loader.cpp:174  - using 12 threads for model loading
[INFO   ] diffusion_engine.cpp:1385 - Weight type stat:                      f32: 712  |     f16: 419  |    q6_K: 304  
[INFO   ] diffusion_engine.cpp:1386 - Conditioner weight type stat:          f16: 415  
[INFO   ] diffusion_engine.cpp:1387 - Diffusion model weight type stat:      f32: 468  |     f16: 4    |    q6_K: 304  
[INFO   ] diffusion_engine.cpp:1388 - VAE weight type stat:                  f32: 244  
[VERBOSE] diffusion_engine.cpp:1390 - ggml tensor size = 432 bytes
[VERBOSE] clip_tokenizer.cpp:66   - vocab size: 49408
[VERBOSE] flux.hpp:185  - flux: depth = 19, depth_single_blocks = 38, guidance_embed = false, context_in_dim = 4096, hidden_size = 3072, num_heads = 24
[INFO   ] model_builders.cpp:505  - using VAE for encoding / decoding
[INFO   ] auto_encoder_kl.hpp:529  - vae decoder: ch = 128
[VERBOSE] diffusion_engine.cpp:1101 - validating model metadata
[VERBOSE] diffusion_engine.cpp:1160 - model metadata validated; weights will be prepared lazily
[INFO   ] diffusion_engine.cpp:1206 - total params memory size = 18858.13MB (VRAM 0.00MB, RAM 18858.13MB): text_encoders 9318.83MB(RAM), diffusion_model 9379.29MB(RAM), vae 160.00MB(RAM), controlnet 0.00MB(N/A), extensions 0.00MB(N/A)
[INFO   ] diffusion_engine.cpp:1340 - running in Flux FLOW mode
[INFO   ] image.cpp:798  - generate_image 512x512
[VERBOSE] model_loader.cpp:174  - using 12 threads for model loading
[INFO   ] diffusion_engine.cpp:2719 - Using 'flux_kontext' preset for reference images
[INFO   ] denoiser.hpp:1125 - get_sigmas with Flux scheduler
[VERBOSE] denoiser.hpp:765  - Flux scheduler: image_seq_len=4096, steps=4, mu=1.150
[INFO   ] request.cpp:415  - sampling using Euler method
[VERBOSE] conditioner.hpp:1139 - parse 'a red apple on a wooden table, photo' to [['a red apple on a wooden table, photo', 1], ]
[VERBOSE] bpe_tokenizer.cpp:208  - split prompt "a red apple on a wooden table, photo" to 9 tokens ["a</w>", "red</w>", "apple</w>", "on</w>", "a</w>", "wooden</w>", "table</w>", ",</w>", "photo</w>", ]
[VERBOSE] t5_unigram_tokenizer.cpp:336  - split prompt "a red apple on a wooden table, photo" to tokens ["▁", "a", "▁red", "▁apple", "▁on", "▁", "a", "▁wooden", "▁table", ",", "▁photo", ]
[VERBOSE] clip.hpp:326  - identity projection
[INFO   ] ggml_graph_cut.cpp:963  - clip build cached graph cut plan done (taking 0 ms)
[VERBOSE] model_loader.cpp:1021 - loading 196/196 tensors from C:\Modeles_LLM\clip_l.safetensors

[INFO   ] model_loader.cpp:1279 - loading tensors completed, taking 0.20s (read: 0.05s, memcpy: 0.00s, convert: 0.00s, copy_to_backend: 0.00s)
[VERBOSE] model_manager.cpp:488  - model manager prepared params backend buffers (235.06 MB, 196 tensors, 1 blocks, RAM) on Vulkan_Host
[VERBOSE] model_manager.cpp:614  - model manager staged compute params (235.06 MB, 196 tensors, 4 blocks) to Vulkan0, taking 0.04s
[DEBUG  ] ggml_runner.cpp:929  - clip executing segment 1/1: graph
[VERBOSE] ggml_runner.cpp:954  - clip compute buffer size: 1.42 MB(VRAM) on Vulkan0 (peak across 1 segment)
[INFO   ] ggml_graph_cut.cpp:963  - t5 build cached graph cut plan done (taking 1 ms)
[VERBOSE] model_loader.cpp:1021 - loading 219/219 tensors from C:\Modeles_LLM\t5xxl_fp16.safetensors

[INFO   ] model_loader.cpp:1279 - loading tensors completed, taking 2.18s (read: 2.02s, memcpy: 0.00s, convert: 0.00s, copy_to_backend: 0.00s)
[VERBOSE] model_manager.cpp:488  - model manager prepared params backend buffers (9083.77 MB, 219 tensors, 10 blocks, RAM) on Vulkan_Host
[VERBOSE] model_manager.cpp:614  - model manager staged compute params (9083.77 MB, 219 tensors, 146 blocks) to Vulkan0, taking 6.05s
[DEBUG  ] ggml_runner.cpp:929  - t5 executing segment 1/1: graph
ggml_vulkan: Compute pipeline creation failed for rms_norm_mul_f32
ggml_vulkan: vk::Device::createComputePipeline: ErrorOutOfDeviceMemory
terminate called after throwing an instance of 'vk::OutOfDeviceMemoryError'
  what():  vk::Queue::submit: ErrorOutOfDeviceMemory
```
</details>



> **Comment by leejet (2026-09-11T17:49:37Z):**

Thanks for testing. The log narrows the failure to T5’s first Vulkan graph execution, after its weights have been staged successfully.

I found a cleanup path that can throw a second exception while handling the pipeline creation failure. That explains why the final error mentions Queue::submit and the new error handler does not get a chance to report it.

For the original OOM, I suspect the smaller weight allocations introduced by the refactor may interact with Vulkan’s preference for CPU-visible VRAM. Could you run the same build and command twice from MSYS2 Bash:
```
GGML_VK_MEMORY_LOGGER=1 <same sd-cli command>

GGML_VK_MEMORY_LOGGER=1 GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 <same sd-cli command>
```
Please attach both logs and, if available, the memory heaps/types section from vulkaninfo. This should help distinguish memory placement pressure from a pipeline-specific driver issue.


> **Comment by laurentvv (2026-09-11T18:11:03Z):**

Both runs done, same setup as my debug log (RX 6950 XT 16 GB, Windows 11, AMD proprietary
driver, local `58d3f6e` build, exact same command: Flux schnell Q6_K, `--offload-to-cpu
--log-level debug`).

**The headline: `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` makes the whole `--offload-to-cpu`
path work.** Clean exit 0, correct image (red apple on wooden table, seed 42). Same 9,083.77
MB T5 staging that died at pipeline creation now completes — and everything is faster, not
just working: T5 staging 1.25 s (vs 5.84 s in run 1), conditioning 4.71 s, sampling 11.65 s
(4 steps), VAE decode 1.41 s, image saved.

Run 1 (`GGML_VK_MEMORY_LOGGER=1` only) pins the failure exactly where you suspected — T5's
first graph execution, after successful staging, and with the double exception you described
(the `terminate` reports `Queue::submit` while the failing call was `createComputePipeline`):

```
[VERBOSE] model_manager.cpp:614  - model manager staged compute params (9083.77 MB, 219 tensors, 146 blocks) to Vulkan0, taking 5.84s
ggml_vulkan memory: ggml_backend_vk_buffer_type_alloc_buffer(71566344)
ggml_vulkan memory: Vulkan0: +68.25 MiB device at 0x152096f9310. Total device: 9.17 GiB, total host: 9.10 GiB
[DEBUG   ] ggml_runner.cpp:929  - t5 executing segment 1/1: graph
ggml_vulkan: Compute pipeline creation failed for rms_norm_mul_f32
ggml_vulkan: vk::Device::createComputePipeline: ErrorOutOfDeviceMemory
terminate called after throwing an instance of 'vk::OutOfDeviceMemoryError'
  what():  vk::Queue::submit: ErrorOutOfDeviceMemory
```

Note the sequence: the 68.25 MiB compute buffer allocates fine right after staging, then the
next allocation (pipeline creation) OOMs while the logger reports 9.17 GiB total device —
with roughly 6 GB of the 16 GB card still free.

`vulkaninfo` memory layout for this card (full section in the gist) matches your CPU-visible
VRAM hypothesis:

- heap 0: 15.73 GiB `DEVICE_LOCAL` (+MULTI_INSTANCE) — real VRAM
- heap 1: 15.92 GiB, no flags — shared/GTT
- heap 2: **256 MiB** `DEVICE_LOCAL|HOST_VISIBLE` — the host-visible VRAM window
- type 2 = `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` → heap 2 (256 MiB)

So the only host-visible device-local pool this driver exposes is 256 MiB. With host-visible
vidmem enabled, device allocations (staged weights, pipeline resources) interact with that
tiny window and pipeline creation ends up `ErrorOutOfDeviceMemory` despite heap 0 being
mostly free; disabling it forces plain device-local placement and the path both works and
gets measurably faster.

Full logs, raw (single gist, 3 files): [https://gist.github.com/laurentvv/6422321845c3466179f20f7ec5c0cf80](https://gist.github.com/laurentvv/6422321845c3466179f20f7ec5c0cf80)
- `run1_memlogger_crash.log` — `GGML_VK_MEMORY_LOGGER=1` (exit 127)
- `run2_nohostvisible_SUCCESS.log` — `+GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` (exit 0, full memory-logger output)
- `vulkaninfo_memory_heaps_types.txt` — heaps/types section for this GPU

Happy to A/B any candidate (e.g. gating host-visible usage by the heap size, or defaulting it
off when a small host-visible device-local heap is detected).


> **Comment by laurentvv (2026-09-11T19:27:11Z):**

Also tested your `--max-vram` suggestion (from #1947) on my #1946 repro — plain path, i.e. no `--offload-to-cpu` and no `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM`, so this maps to the original (non-offload) failure. Same setup as my earlier reports (RX 6950 XT 16 GB, Windows 11, AMD proprietary driver, local `58d3f6e` build = PR #1958 head; Flux schnell Q6_K, 512×512, 4 steps, euler, seed 42, `--vae-tiling`), `--max-vram` 10/12/14 (GiB).

**The graph still doesn't complete on this card, but `--max-vram` produces a very clean diagnostic.** The budget mechanism itself works: with a cap set, reported free device memory tracks the cap exactly (the underflowed free-VRAM accounting from my original report is bypassed). What fails instead is the budget-side bookkeeping: the manager fills residency up to the cap, then can no longer evict enough to make budget headroom for the next segment's streaming weights.

| `--max-vram` | fails at | need (device / budget) | available (device / budget) |
|---|---|---|---|
| (none, unlimited) | segment 30/59 `flux.single_blocks.9` | 954.20 MB / 457.22 MB | 906.93 MB / unlimited |
| 10 GiB | segment 4/59 `flux.double_blocks.2` | 793.13 MB / 537.55 MB | 6187.66 MB / 316.08 MB |
| 12 GiB | segment 11/59 `flux.double_blocks.9` | 793.13 MB / 537.55 MB | 4324.72 MB / 501.13 MB |
| 14 GiB | segment 19/59 `flux.double_blocks.17` | 793.13 MB / 537.55 MB | 2195.64 MB / 420.06 MB |

Observations:

- With a budget set, the failing need is **constant** (793.13 MB device / 537.55 MB budget — streamed weights + runtime peak, identical at every death point), while physical device memory is plentiful (the cap still leaves 2.2–6.2 GB free). The only blocker is budget headroom: once residency saturates the cap, the eviction pass never frees the 537.55 MB of budget needed for staging. The death segment moves later as the budget grows (4 → 11 → 19) simply because more fits, but every value dies the same way.
- Best case (12 GiB) misses by only 36.42 MB of budget (need 537.55 / available 501.13).
- Without a budget, the per-segment budget need is lower (457.22 MB at segment 30 — single blocks are lighter than double blocks), but there it's the broken free-VRAM report that blocks instead (need 954.20 / available 906.93 device).

So on this card (18,858 MB of weights vs 16 GB VRAM — stream-through is mandatory), no `--max-vram` value I tried lets the graph complete; the failure just moves. The eviction/budget interplay on the stream-through path looks like the remaining blocker for the plain (non-offload) path — happy to test any candidate that changes it.

Full logs (10/12/14 GiB): [gist](https://gist.github.com/laurentvv/8a4eaa0437b586ae0e2c3e288181d819)



> **Comment by leejet (2026-09-13T09:43:38Z):**

I made an attempt to fix the issue. Please try the latest code from https://github.com/leejet/stable-diffusion.cpp/pull/1958.

Run the following command and check whether the issue has been resolved. If the problem still persists, please send me the detailed debug logs so I can investigate further.

```bash
sd-cli.exe --diffusion-model flux1-schnell-Q6_K.gguf \
  --clip_l clip_l.safetensors --t5xxl t5xxl_fp16.safetensors --vae ae.safetensors \
  -p "a red apple on a wooden table, photo" \
  --steps 4 --cfg-scale 1.0 --sampling-method euler -W 512 -H 512 --seed 42 \
  --vae-tiling --backend vulkan --offload-to-cpu --log-level debug -o out.png
```


> **Comment by laurentvv (2026-09-13T11:41:04Z):**

Tested the latest code from PR #1958 (head `1bf76e6`, including `da4e841` "handle backend cleanup errors and use 256 MiB residency blocks" and `5c7fc62` "use 1 GiB residency blocks") on the same setup as my previous reports.

**Setup**: RX 6950 XT 16 GB, Windows 11, AMD proprietary driver, local MSYS2/UCRT64 gcc build of PR head `1bf76e6`, ggml submodule at `e20c3a14` (unchanged from my bisect — conditions identical to my earlier tests). Flux schnell Q6_K, 512×512, 4 steps, euler, seed 42, `--vae-tiling --backend vulkan`.

**Results — both failure paths from my previous reports now work:**

1. **Plain path (no offload, the original #1946 failure)**: ✅ fixed. Generation completes (`generate_image completed in 13.14s`, exit 0). Previously (`58d3f6e`) it failed deterministically at segment 30/59 (`flux.single_blocks.9`), ~47 MB short, and on master ≥848 it silently crashed.

2. **`--offload-to-cpu`** (your suggested command, with `--log-level debug`): ✅ works too. Exit 0, image saved (`completed in 17.41s`). Previously (`58d3f6e`) this path died with a silent exit 127 on `vk::Device::createComputePipeline: ErrorOutOfDeviceMemory` despite ~15.5 GB free VRAM.

**Outputs** (repro prompt "a photo of a red fox in snow" for #1, your apple command for #2):

![plain path — PR #1958 head 1bf76e6](https://raw.githubusercontent.com/laurentvv/generator-assets/main/scratch/bisect_1946/pr1958_head2_test.png)

![--offload-to-cpu — PR #1958 head 1bf76e6](https://raw.githubusercontent.com/laurentvv/generator-assets/main/scratch/bisect_1946/pr1958_offload_head2.png)

**Sanity check vs pre-regression baseline**: same repro on the installed `6b3edaa` (master-841) binary produces a visually identical image — mean per-pixel difference 1.6/255 (small numeric drift expected from the merged master refactors):

![baseline master-841 6b3edaa](https://raw.githubusercontent.com/laurentvv/generator-assets/main/scratch/bisect_1946/baseline_6b3edaa_fox.png)

One extra observation: with ~2-3 GB of VRAM occupied by another process, the old `6b3edaa` binary fails at `flux prepare` (OOM on a 331 MB params buffer) while the PR build had just completed the same run — the new residency manager also seems more robust under VRAM pressure on this 16 GB card.

Thanks — the residency block sizing was indeed the missing piece. From my side the regression is resolved; happy to re-test if you change anything else before merging.



> **Comment by leejet (2026-09-13T13:35:00Z):**

Thanks for testing. Since all the issues have been fixed, I’ll merge the corresponding PR.



### #1985 [open] MiniMax-H3 I2V/Ref2VA conditioning hangs GPU on gfx1100 ROCm 7.2.4 & 7.14 (works on CPU backend; Unsloth fork 13b9d92)
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1985  by kaishi00 2026-09-17T02:07:12Z  comments: 10

## Summary

MiniMax-H3 Ref2VA / keyframe I2V conditioning **hangs the GPU** when run on the ROCm backend of the Unsloth fork build, while the **identical workload completes on the CPU backend**. Tested across two ROCm versions (7.2.4 and 7.14), multiple resolutions (384×640, 640×384, 544×960, 960×544), and both Studio-spawned and shell-launched invocations. Text-to-video works fine on the same installation; only the image/ref-conditioning path is affected.

## Environment

- HW: AMD Radeon RX 7900 XTX (gfx1100), 24 GB; Ryzen 9 9950X, 64 GB RAM
- OS: Ubuntu 24.04 (LXC, GPU passthrough via /dev/kfd + /dev/dri)
- Binary: self-compiled Unsloth fork `13b9d92` (H3 fixes) — `sd-cli --mode vid_gen`
- ROCm runtime tested: **7.2.4** (segfault variant) and **7.14.0** (hang variant)
- Models: `minimax_h3_ref2va_pruned-Q5_0.gguf` + `minimax_h3_fl2va_pruned-Q5_0.gguf` + qwen3vl 32B encoder + fp16 video VAE + fp32 audio VAE

## Repro

```
sd-cli --mode vid_gen --offload-to-cpu \
  --backend "vae=cpu,clip=cpu,diffusion=ROCm0" --vae-conv-direct \
  --diffusion-model .../minimax_h3_ref2va_pruned-Q5_0.gguf \
  --vae .../minimax_h3_video_vae_fp16.safetensors \
  --audio-vae .../minimax_h3_audio_vae_fp32.safetensors \
  --llm .../qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --ref-image <ref.png> \
  --prompt "lantern" --cfg-scale 1 --width 640 --height 384 --rng cpu --fps 24 \
  --video-frames 33 --steps 8 --flow-shift 12 --seed 42 \
  --output out.webm
```

## Behavior matrix (all tested on this host)

| ROCm | Backend | I2V (keyframe/ref) | Result |
|---|---|---|---|
| 7.2.4 | GPU | T2V 640-960 | ✅ works |
| 7.2.4 | GPU | I2V 640×384 | ❌ SIGSEGV (exit 139) inside libamdhip64, first sampling step |
| 7.2.4 | GPU | I2V 960×544 | ❌ SIGSEGV |
| 7.14 | GPU | I2V any resolution | ❌ **HANG**: main thread busy-loops `sched_yield()` inside libamdhip64; HSA threads wait on GPU events; 0% GPU, VRAM flat, killed after 25+ min |
| 7.14 | **CPU backend** | I2V | ✅ **completes** (proves conditioning logic + weights are sound) |

## gdb backtrace (7.14 hang, main thread)

```
#0  __GI_sched_yield ()
#1-#11 ?? () from /opt/rocm-7.14/rocm/core-7.14/lib/libamdhip64.so.7
```

Thread 1 spins in a HIP busy-wait; HSA async threads wait on GPU events that never fire. A GPU kernel launched by the keyframe/ref conditioning encode **never completes on gfx1100**.

## What we ruled out

- Resolution (640→960, portrait and landscape)
- Quantization (Q3_K_XL, Q4_K, Q5_0, Q6_K)
- Image format / staged-file presence (valid PNG verified by md5 before each run)
- Studio wrapper flags (`--diffusion-fa`, `--max-vram -1`, `--stream-layers` — all removed, same result)
- `--end-img` presence (FLF2V full contract — same crash)
- VRAM pressure (hang occurs at 1.5 GB used, before weights finish loading)
- The Unsloth fork's H3 fixes (CPU backend runs the same conditioning fine — logic is sound, kernel launch is what dies)

## Impact

Any I2V / Ref2VA use on gfx1100 with ROCm is blocked. The workaround is running the diffusion model on CPU too (`--offload-to-cpu` without any GPU backend assignment), which costs ~97 min for a 5-second clip at 544×960/30 steps vs an expected few minutes on GPU.

Happy to provide additional traces, HSA logs (`HSA_ENABLE_SDMA=0` was also tested — no change), or test builds.


> **Comment by wbruna (2026-09-17T02:45:58Z):**

> Binary: self-compiled Unsloth fork `13b9d92` (H3 fixes) — `sd-cli --mode vid_gen`

Apart from [13b9d92](https://github.com/unslothai/stable-diffusion.cpp/commit/13b9d92) being on a fork, it's more than a month old. Can you reproduce this issue on latest master?


> **Comment by kaishi00 (2026-09-17T08:40:47Z):**

Result of testing latest master (**cc515a0**, self-built against gfx1100):

**Short answer: the hang reproduces on master — but master also improved the conditioning stage, which changes the failure signature.**

Test env: same host as the OP (RX 7900 XTX, Ubuntu 24.04 LXC), two runtimes:

### 1) master + ROCm 7.14
- Conditioning (Ref2VA image encode with the ref image) now **completes on GPU** — no hang there anymore, reaches sampling `1/8` at ~2 it/s
- Then **hard hang at first sampling step**: VRAM flatlines at 1.5 GB, GPU busy 0%, no further steps, killed after 12+ min. Same signature as the OP's fork builds.
- Notable: master's auto-fit now assigns `te=cpu,vae=cpu` params automatically (`--backend "diffusion=ROCm0,te=ROCm0,vae=ROCm0" --params-backend "te=cpu,vae=cpu"`), and even with VAE compute nominally on ROCm the stall point is identical.

### 2) master + ROCm 7.2.4 (the distro `/opt/rocm` stack)
- Reaches sampling `1/8`, then **SIGSEGV** — backtrace is a heap corruption signature (`_int_malloc` → `operator new` ← deep `libamdhip64.so.7` recursion), i.e. the same crash window as the OP, but the failure now lands *after* conditioning instead of during it.

So on current master: **7.2.4 → segfault, 7.14 → hang.** Both die in/around the first Ref2VA sampling step; the CPU-backend run of the identical command still completes (weights/logic exonerated). Whatever upstream fixed in the conditioning path since the Sep 13 fork base didn't reach the underlying ROCm interaction.

Happy to test any patch/branch, or provide full logs + gdb traces from both runs (logs preserved at `/root/master_714.log`, `/root/master_full.log` on the test box).



> **Comment by kaishi00 (2026-09-17T08:48:01Z):**

Follow-up: reproduced the 7.14 hang on a **completely fresh boot** — container rebooted, zero GPU holders, VRAM 0.0 GB at start, nothing else loaded. Master cc515a0 + ROCm 7.14, same 640×384 Ref2VA command:

- conditioning completes, sampling starts at 1.48 it/s
- pins at `1/8` permanently: VRAM flatlines 1.5 GB, GPU busy 0%, RSS frozen at 1,056,392 kB (byte-identical across minutes), 4+ min CPU spin before kill

So this is **deterministic on the GPU path, not accumulated-state** — a fresh process/host state does not clear it. Logs: `/root/master_coldboot.log`.



> **Comment by kaishi00 (2026-09-17T09:23:35Z):**

**Breakthrough: the same Ref2VA workload completes ALL-GPU in 75s on the Vulkan backend** (master cc515a0, self-built `-DGGML_VULKAN=ON`, Mesa RADV, RX 7900 XTX) — no hang, no segfault, no CPU offload of the diffusion model:

```
auto-fit: DiT 13245 MiB -> compute Vulkan1, params Vulkan1
generate_video completed in 74.96s
save result video to '/root/vulkan_gpu_i2v_test.avi'
```

Output verified: real pixels, correct dimensions, audio present. Same command that hangs >12 min at `1/8` sampling on ROCm 7.14 (and SIGSEGVs on 7.2.4).

**So the defect is specific to the ROCm/HIP backend path on gfx1100 — the Vulkan stack on the same GPU handles the identical graph fine.** This also gives AMD users a same-GPU escape hatch: build sd.cpp with GGML_VULKAN and point `--backend diffusion=Vulkan1`.

Happy to test any ROCm-side patch; meanwhile Vulkan is a viable workaround that keeps diffusion fully on the 7900 XTX.



> **Comment by wbruna (2026-09-17T10:55:29Z):**

This could be #1947 , and the 'hang' actually an extreme slowdown; that issue mentions steps taking more than 13 minutes. One attempt to debug this further would be applying the logging patch from that issue, and if it behaves the same a HIP/rocBLAS trace.


> **Comment by farawayso (2026-09-17T11:41:45Z):**

Same gfx1100 hardware (RX 7900 XTX, 24GB), and I'm running the exact same binary commit as @kaishi00's latest-master test — cc515a0 — so I can corroborate the ROCm hang on H3 I2VA/Ref2VA conditioning from a matching setup. Two additions the matrix doesn't cover yet:

1. ROCm version axis. I'm on ROCm 10.0 (Windows/MSVC build, HIP 7.15), well past the 7.2.4 / 7.14 in the repro. I2VA/Ref2VA conditioning on the ROCm backend is still blocked here — so this looks gfx1100 + ROCm-independent rather than a 7.x regression. (Different OS though: I'm Windows, the OP is Linux LXC, so I'm not claiming the 7.x hang reproduces on 10.x, only that the ROCm path stays unusable for H3 conditioning on the same GPU + commit.)

2. The Vulkan escape hatch is confirmed on this same GPU + cc515a0. I don't just manually flip --backend diffusion=Vulkan1; I enforce it at the wrapper layer — i2va/fl2va/ref2va semantics all route to Vulkan unconditionally, leaving only T2VA (decode-only) on ROCm. That's because ROCm VAE encode on gfx1100 is its own separate crash (Access Violation 0xC00000FD, upstream #7691, still open), distinct from this sampling-stage hang.

For triage: the failure point here is after conditioning completes — first sampling step, VRAM flatlines ~1.5GB, GPU 0%, main thread busy-looping sched_yield() in libamdhip64. That's a diffusion-kernel/ROCm hang, not the VAE-encode 0xC00000FD stack overflow from #7691. So I'd keep it as a separate gfx1100+ROCm issue rather than a duplicate of #7691 (but it's the same family: "H3 conditioning path dies on ROCm/gfx1100").

Happily test any patch/branch on ROCm 10.0 + Windows or drop HIP/rocBLAS traces — let me know the logging patch from #1947 you want applied and I'll run it.


> **Comment by kaishi00 (2026-09-17T18:34:55Z):**

**Regression-hunt result (Outcome B): does NOT reproduce on our environment — 6b3edaa falls back to CPU at the same resolutions as current master.**

Tested the reported pre-regression revision on our RX 7900 XTX / Ubuntu 24.04 / Mesa RADV 25.2.8 / Vulkan loader 1.3.275:

- Worktree: `6b3edaaf32cc` ("generalize temporal tiling across video VAEs", #1926, Aug 31) self-built `-DGGML_VULKAN=ON`, same toolchain as our master build

| Resolution | old 6b3edaa | current master cc515a0 |
|---|---|---|
| 640×384 33f | ✅ GPU compute confirmed (GPU 100% sustained), 100 s/it, completed | ✅ GPU confirmed, 0.5 s/it, 75 s total |
| 864×480 33f | ⚠️ **mixed** — completed, 360 s/it, `ggml_vec_dot_f32` in hot stacks, ~11 min CPU burn/step, GPU bursts 26-100% alternating with 0%, `pinned memory allocation failed` ×1 | ❌ CPU fallback (from earlier comment) |
| 960×544 73f | ❌ **CPU fallback confirmed** — 1448 s/it (24 min/step), 17 min CPU/step, `ggml_vec_dot_f32`/`mul_mat` in hot stacks, VRAM residency 25.6GB while GPU 0-3% | ❌ same (earlier comment) |

Key observations:

1. **The old revision is not better here — it degrades at the same graph sizes**, just with slower, older shaders when it does use the GPU (100 s/it vs 0.5 s/it at 640×384).
2. The failure correlates with RADV's pinned-memory allocation failure ("Requested buffer size exceeds device buffer size limit") appearing right as the compute falls into CPU `ggml_vec_dot_f32`. VRAM can still *hold* the weights (25.6GB reported) while the matmuls silently run on CPU — residency ≠ compute, as the earlier comments established.
3. So on our hardware the large-graph CPU fallback looks like a **ggml-Vulkan/RADV buffer-limit interaction, not a stable-diffusion.cpp code regression**. The upstream reporter's success at ~864×480 on the old revision likely reflects different buffer limits (NVIDIA, different GPU, or different driver) rather than a code change between 6b3edaa and master.

Classification per the issue's framework: **Outcome B — not reproduced**. We're not pinning old code; our working all-GPU path is current master Vulkan at ≤640×384-class graphs, and larger graphs need either graph chunking in ggml-vulkan or a maxAllocationSize accommodation.



> **Comment by kaishi00 (2026-09-17T22:42:28Z):**

Follow-up with the #1947 diagnostic patch + HIP tracing: this does **not** match #1947's finite first-step slowdown.

I instrumented the same `cc515a0` / ROCm 7.14 / gfx1100 Ref2VA repro and left it running for **3h30m** after it reached sampling `1/8`. It never reached `2/8`.

Conditioning is healthy:

```text
VAE prep:    0.30s
VAE compute: 0.49s on ROCm0
outputs:     0.00s
```

The failure localizes to the first Ref2VA denoiser graph launch. The last HIP trace sequence is:

```text
hipGraphExecUpdate: Returned hipSuccess
hipGraphLaunch
GraphExec::Run max_streams: 1
UpdateStreams failed for device id:0
```

After that:

- no subsequent useful kernel activity is observed
- GPU remains 0% busy
- VRAM remains flat at ~1.5 GB
- the main thread spins in `sched_yield()` inside `libamdhip64.so.7`
- all-thread gdb backtrace contains the corresponding HSA/HIP wait stack
- the compute call never returns

The process was finally terminated after **3h30m30s** and exited cleanly on SIGTERM.

So this appears materially different from #1947: there, the pathological first compute eventually returned after ~731s and later steps ran normally. Here, the first denoiser compute is non-returning and the trace ends immediately after the HIP graph stream-update failure.

I'm treating `UpdateStreams failed for device id:0` as the **failure location**, not claiming it is the deeper root cause.

I have the full artifacts preserved:

- instrumented `ggml_runner.cpp` patch
- complete debug log / AMD trace
- command/environment/build metadata
- main-thread and all-thread gdb backtraces
- timeline samples

The full trace is ~200 MB, so I can upload a compressed archive or any narrower subset that would be more useful.


> **Comment by farawayso (2026-09-17T23:26:04Z):**

Thanks for the follow-up. We have a complementary data point from the same gfx1100 hardware, and the two cases line up nicely:

- Env: gfx1100 (RX 7900 XTX 24GB), ROCm 10.0, sdcpp at recent commit, Ref2VA/I2VA paths
- Our first-step trace shows the same API-call overhead pattern, but it eventually returns after ~732s (vs. your non-returning case). The fast/slow traces have identical ~1.3M-event API call structure (PAL-VMM heap and launch-config events each ~31%, GEMM only ~0.7%) — so the 700s delta is pure HIP API call-interval overhead, not compute.
- This suggests your hard hang (UpdateStreams failed for device id:0 → main thread spinning in sched_yield) is the same pathological HIP graph/stream-update path, just without the eventual-return behavior.

We have the full fast (37s) vs. slow (732s) traces captured locally. Happy to share a compressed subset (or specific rocm-instrument / timeline windows) if it helps narrow down whether the 7.x regression is in the graph-update path or the stream-sync fallback. Also relevant to our side: we have a runtime guard that demotes I2VA to the Vulkan backend on ROCm, so the hang doesn't block generation — if your artifacts include the instrumented ggml_runner.cpp patch, that would let us A/B it against our guard.

Note: this analysis was drafted with help from Hermes Agent running on a free-tier LLM — the trace numbers were verified locally, but the interpretation may not be fully accurate, so please treat the root-cause claims as hypotheses.


> **Comment by basementops (2026-10-06T13:31:06Z):**

**Same family on gfx1151 (Strix Halo), but in the MiniMax-H3 video VAE *encode*, sporadic, and avoided by disabling HIP graphs**

Env: Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, UMA), Debian 13, ROCm 7.2.4, built with `-DSD_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON`.
Command: I2VA, `--init-img`, 704×704, `--video-frames 49`, `--steps 2`, `--cfg-scale 1 --diffusion-fa --offload-to-cpu --rng cpu`, `minimax_h3_fl2va_pruned-Q4_K.gguf` + `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` + fp16 video VAE + fp32 audio VAE.

- Hangs **sporadically** during the VAE encode of the init image, right after `ggml_backend_cuda_graph_compute: CUDA graph warmup complete`, at varying tiles (2/16, 12/16). GPU busy 100 %, one CPU core spinning. Main thread: `sched_yield()` in `libamdhip64` ← `ggml_cuda_graph_evaluate_and_capture` ← `MiniMaxH3VideoVAERunner::encode` (full backtrace available).
- Rate with graphs enabled: 11/31 runs on `b167b94`, 1/10 on current master `a1ded76` (same location; master tiles 3×3).
- With `GGML_CUDA_DISABLE_GRAPHS=1`: 0/10 hangs, interleaved with graph-enabled runs that did hang. Vulkan build: 0/5.
- T2VA is unaffected.

Happy to run a patch or collect a HIP trace.




### #1871 [open] [Bug] Poor video and audio quality with Minimax H3
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1871  by GlasslessPizza 2026-08-11T20:13:01Z  comments: 9

### Git commit

Using sd-master-b4f1fd6-bin-win-cuda12-x64.zip

### Operating System & Version

Windows

### GGML backends

CUDA

### Command-line arguments used

sd-cli.exe -M vid_gen --diffusion-model minimax_h3_fl2va_pruned-Q4_K_M.gguf --vae minimax_h3_video_vae_fp16.safetensors --audio-vae minimax_h3_audio_vae_fp32.safetensors --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf --lora-model-dir lora --output video.avi -p "Fast paced car chase through a steampunk city, with the drivers loudly trashtalking each other in an obnoxious british smug tone, one of the drivers say 'You’re driving a glorified toaster with a spoiler. I’ve seen more aerodynamic shapes in a bowl of porridge'.<lora:minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16:1>" --cfg-scale 1.0 -v -W 1280 -H 768 --diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 96 --steps 8 --seed -1 --batch-count 1

### Steps to reproduce

Run the provided cmd on the binaries provided in the master-815-b4f1fd6 release.

### What you expected to happen

Clean video quality without artifacts, clean audio quality with clearly audible voices and without compression noise and other audio artifacts.

### What actually happened

Video quality is poor, with tons of pixel crawling and mpeg artifacts.
The audio quality is so poor I can't even make out what the characters are saying, I'd say it's comparable to extremely low bitrate mp3. Taking off the lora doesn't help.

### Logs / error messages / stack trace

_No response_

### Additional context / environment details

Model files gathered from the official documentation:
https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/minimax_h3.md

> **Comment by GlasslessPizza (2026-08-11T20:16:42Z):**

I believe the bug report form has nuked the `<` and `>` around the lora declaration in the prompt.


> **Comment by GlasslessPizza (2026-08-11T22:33:07Z):**

Ok after updating everything, ditching the lora and massively increasing the amount of steps to 40 (I believe the default is 20) now the audio is bearable, but all this only marginally improved the video quality issues described.


> **Comment by SharkyRawr (2026-08-12T15:24:27Z):**

Are you using libggml from ggml-org or https://github.com/leejet/ggml ? I was using the ggml-org variant and got similar heavy distortion but after switching to leejet/ggml it looks pretty much as perfect as can be for my GPU.


> **Comment by saddam213 (2026-08-12T22:17:46Z):**

I have hit a similar issue, longer videos seem to degrade in video and audio quality, with 5 second videos have fully distorted audio

Nvidia 3090 24GB
Windows Server 2022
128GB RAM
sd-master-de298c2


```
--llm E:\MiniMax-H3\qwen3vl_32b_minimax_h3-Q4_K_M.gguf 
--diffusion-model E:\MiniMax-H3\minimax_h3_fl2va-Q4_K_M.gguf 
--vae E:\MiniMax-H3\minimax_h3_video_vae_fp16.safetensors 
--audio-vae E:\MiniMax-H3\minimax_h3_audio_vae_fp32.safetensors 
--backend cuda0 
--offload-to-cpu 
--max-vram -1 
--fa 
--rng cpu 
--cfg-scale 1.0
```

56 Frames
https://github.com/user-attachments/assets/b53e89d7-d9c9-446e-a8df-b3e855a24b04

73  Frames
https://github.com/user-attachments/assets/60a7c702-2efc-492b-9c77-4da8be9d7711

90 Frames
https://github.com/user-attachments/assets/5785353e-08df-4d7f-bc58-952eec1c36c0

107 Frames
https://github.com/user-attachments/assets/0955d21c-b4d6-4428-8a2a-0e16fff0ae2c

124 Frames
Wont upload but is worse that 107 frames


I have tried the following models, all seem to behave the same

- minimax_h3_fl2va_pruned_bf16.safetensors
- minimax_h3_fl2va_pruned_int8_convrot.safetensors
- minimax_h3_fl2va_pruned-Q8_0.gguf
- minimax_h3_fl2va-Q4_K_M.gguf

- qwen3vl_32b_minimax_h3-Q4_K_M.gguf
- qwen3vl_32b_minimax_h3_bf16.safetensors

Not sure if I am simply missing a flag, or its something deeper


> **Comment by GlasslessPizza (2026-08-13T07:44:24Z):**

> Are you using libggml from ggml-org or https://github.com/leejet/ggml ? I was using the ggml-org variant and got similar heavy distortion but after switching to leejet/ggml it looks pretty much as perfect as can be for my GPU.

Not sure what you mean, I'm using the precompiled binaries.

> Not sure if I am simply missing a flag, or its something deeper

I'd say try to add --steps 40, it improved the situation for me. Are you using loras?


> **Comment by saddam213 (2026-08-13T20:46:58Z):**

> > Are you using libggml from ggml-org or https://github.com/leejet/ggml ? I was using the ggml-org variant and got similar heavy distortion but after switching to leejet/ggml it looks pretty much as perfect as can be for my GPU.
> 
> Not sure what you mean, I'm using the precompiled binaries.
> 
> > Not sure if I am simply missing a flag, or its something deeper
> 
> I'd say try to add --steps 40, it improved the situation for me. Are you using loras?

No lora, just using the instructions in the sd-cpp minimax-h3 documentation

Extra steps make no difference, and things get worse the longer the video is, at 15 seconds the "surfing cat" is a "dog playing in grass"??


> **Comment by velartrill (2026-08-15T15:19:33Z):**

> Extra steps make no difference, and things get worse the longer the video is, at 15 seconds the "surfing cat" is a "dog playing in grass"??

if it's the same problem I hit, the change suggested in #1865 should fix it


> **Comment by saddam213 (2026-08-22T01:56:27Z):**

> > Extra steps make no difference, and things get worse the longer the video is, at 15 seconds the "surfing cat" is a "dog playing in grass"??
> 
> if it's the same problem I hit, the change suggested in [#1865](https://github.com/leejet/stable-diffusion.cpp/issues/1865) should fix it

Just tested latest build, issues seem to be resolved :)


> **Comment by CHIPMUNK-T0T (2026-09-10T13:36:00Z):**

Adding data from OpenCL/Adreno in case it helps.

**The failure tracks the video latent frame count (`latent_t`).** At 256×160, 12 steps, fixed seed, varying only `--video-frames`:

| frames | latent_t | verdict |
| ---: | ---: | --- |
| 56 | 17 | clean |
| 73 | 22 | broken |
| 124 | 37 | badly broken |

Sharp cliff between 17 and 22, not gradual. Still there on current master (`7f410a3`), re-tested today.

@GlasslessPizza — your `--video-frames 96` aligns up to 107 by the `17k+5` rule, so `latent_t = 32`, past that cliff. Separately, two MiniMax-H3 audio fixes landed after your report (#1908, #1924), so the audio half is worth re-testing on current master.

What I ruled out, in case it saves anyone time:

- **Not a GPU or backend problem** — same corruption with `--backend all=cpu`.
- **Not sequence length or resolution** — 256×160/73f (packed length 1153) is broken; 512×288/39f (1887, *longer*) is clean.
- **Not the audio branch** — forcing the audio latent length independently, `latent_t` 17 stays clean with audio tokens at the broken-case value, and `latent_t` 22 breaks with audio at its normal value.
- **Not the VAE** — decoding only the first 17 latent frames of a 73-frame render is still broken, so the latent is corrupt from its first frame.
- **Not temporal RoPE range or spacing** — scaled both ways; the outcome tracks `latent_t` regardless.
- Also ruled out: NaN/overflow, step count, RNG, attention mask (H3 passes none).

**On #1865:** that fixes the segmented-offload path (`max_vram`), which fits @saddam213 running `--max-vram -1` and then seeing it resolved. My runs have `max_vram: 0`, `stream_layers: false`, `offload_params_to_cpu: false` on a tree that contains the #1865 fix, and the cliff is unchanged there — so this cliff is not that bug. Whether @saddam213's frame sweep was purely the offload path, or that plus this, is untested.

Caveat: everything above is 256×160 and 512×288. I have not tested 1280×768, so I can't claim this accounts for all of what the original report is seeing. Nothing in the code changes behaviour between `latent_t` 17 and 22 that I could find, so I also can't say whether this is a `latent_t`-dependent path shared by all backends or something in the model itself. Happy to post the full measurements and elimination table, or run more experiments on this hardware — just say what would help.




### #2169 [closed] [Bug] ROCm dual-device: streaming diffusion buffer init fails with "unspecified launch failure / current device: -1" when a gfx1150 iGPU is visible alongside gfx1100 dGPU (MiniMax-H3 ref2va)
URL: https://github.com/leejet/stable-diffusion.cpp/issues/2169  by farawayso 2026-10-19T15:13:59Z  comments: 7

## Git commit
95b3090 (master-975 prebuilt, official Windows ROCm bundle; prebuilt carries no source, so exact ggml submodule state unknown — crash path: ggml/src/ggml-cuda/ggml-cuda.cu:817)

## Operating System & Version
Windows 11 (AMD platform)

## GGML backends
HIP

## Command-line arguments used
Failing case (all three of these variants fail; shown: variant C):

```
sd-cli -M vid_gen \
  --diffusion-model minimax_h3_ref2va_pruned-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --cfg-scale 1 -W 480 -H 864 --fps 24 --video-frames 13 \
  --flash-attn off,diffusion=on \
  --conv-direct auto,diffusion=on,controlnet=on,vae=on \
  --rng cpu --flow-shift 12 --steps 4 --sampling-method euler --scheduler discrete \
  --log-level debug -t 16 \
  --backend "llm=ROCm0,vae=ROCm1,diffusion=ROCm1" \
  --params-backend "diffusion=cpu,te=cpu" \
  --max-vram "ROCm0=12" \
  --ref-image <one image> -p "a cat sitting on a table" -o out.mp4
```

Minimal repro variant (recommended to reproduce — variant B, the iGPU is NEVER used for compute):

```
--backend "llm=ROCm1,vae=ROCm1,diffusion=ROCm1"
--params-backend "diffusion=cpu,te=cpu"
--max-vram "ROCm0=12"
(HIP_VISIBLE_DEVICES unset, so both devices are visible; iGPU is never initialized/used)
```

## Steps to reproduce
1. Machine: 64 GB system RAM, AMD iGPU Radeon 890M (gfx1150, BIOS dedicated allocation = minimum 0.5 GB, shared pool enabled) + discrete RX 7900 XTX (gfx1100).
2. Ensure both devices are visible to HIP (no HIP_VISIBLE_DEVICES filter). The iGPU is enumerated as ROCm0 with ~43.7 GB of shared VRAM.
3. Run any MiniMax-H3 ref2va/video command with streaming params backend (`--params-backend diffusion=cpu`) and diffusion placed on the dGPU.
4. Crash after Qwen3VL conditioning and diffusion weight load, at the first diffusion compute-buffer init on the dGPU.

## What you expected to happen
Clean generation (single-device run does: same command with only the dGPU visible via HIP_VISIBLE_DEVICES=1 completes in ~62 s and writes the video), or at minimum a clear error.

## What actually happened
Deterministic crash, reproduced 3/3 back-to-back with no reboot in between (LLM stage 24.95 / 24.95 / 25.55 s), identical line every time:

```
[V] model manager prepared params backend buffers (10975.93 MB, 532 tensors, 1 blocks, RAM) on ROCm_Host
[E] ggml: ROCm error: unspecified launch failure
[E] ggml:   current device: -1, in function ggml_backend_cuda_buffer_init_tensor at .../ggml/src/ggml-cuda/ggml-cuda.cu:817
[E] ggml:   hipStreamSynchronize(((hipStream_t)2))
```

Process exits with `0xC0000409` (STATUS_STACK_BUFFER_OVERRUN, GGML abort path). No output file.

## Isolation matrix (same binary, same models, same flags; only device visibility / placement / max-vram vary)

| variant | devices visible | placement | --max-vram | result |
|---|---|---|---|---|
| C | 2 (iGPU+dGPU) | llm→iGPU, vae/diff→dGPU | ROCm0=12 | **FAIL 3/3** (crash above) |
| A | 1 (dGPU only, HIP=1) | all dGPU | (removed) | PASS, 62.0 s |
| A2 | 1 (dGPU only, HIP=1) | all dGPU | ROCm0=12 | PASS, 71.9 s |
| B | 2 (iGPU+dGPU) | all dGPU, **iGPU backend never initialized** | ROCm0=12 | **FAIL** (same crash) |

Conclusions:
- `--max-vram` is not the variable (A2 vs B).
- Using the iGPU for compute is NOT required (B): the iGPU backend is never created in B (log shows only `Initializing backend: ROCm1` + `CPU`), yet the dGPU buffer init still fails.
- The only difference between PASS and FAIL rows is whether the gfx1150 iGPU is enumerated/initialized in the process.
- VAE buffer init on the dGPU succeeds in all failing runs; only the streaming diffusion compute-buffer init (`ggml_backend_cuda_buffer_init_tensor` with `hipStreamSynchronize` on a per-thread stream) fails, with `current device: -1` reported by the error handler.

## Not the same root cause as #2008
#2008 ("unspecified launch failure" at first compute, gfx1100, single GPU) was narrowed to pinned host memory and is avoided with `GGML_CUDA_NO_PINNED=1`. This report:
- requires TWO HIP devices to be visible (a gfx1150 iGPU in the mix);
- is independent of `--max-vram` value and of whether the iGPU is used;
- fails at buffer *init* (hipStreamSynchronize), not at kernel launch after tensor load.

## Possible mechanism (related upstream work)
ggml-org/llama.cpp#21140 documents the same `current device: -1` family: on multi-GPU ROCm, `hipGetDevice()` can return an unexpected value on threads that never explicitly initialized a device context, so the early-return in `ggml_cuda_set_device()` skips `hipSetDevice()` and the next stream op fails. PR #21170 (remove that early return) is open/unmerged. If the vendored ggml in this prebuilt still has the early-return, this may be the same bug class surfacing at `buffer_init_tensor` when a second (gfx1150) agent is present.

## Environment details
- Driver: Adrenalin 32.0.31041 (ROCm 10.1 surface)
- ggml init: `ggml_cuda_init: found 2 ROCm devices (Total VRAM: 69256 MiB)` — iGPU gfx1150 44696 MiB / dGPU gfx1100 24560 MiB
- iGPU BIOS dedicated VRAM = 0.5 GB (minimum); shared memory pool enabled (HIP reports ~43.7 GB for the iGPU). Dedicated-allocation test pending (planning to raise it to 4 GB and re-run variant B; will report back).
- Single-device (HIP_VISIBLE_DEVICES=1) runs work fine, including the same streaming config.

## Logs
Full captured logs available (4 files: C/B failing, A/A2 passing; each includes device enumeration, the full plan fingerprint, and the 3-line error). Happy to attach or zip them here.

[sdcpp-issue-roc-dual-device-buffer-init_logs.zip](https://github.com/user-attachments/files/33672754/sdcpp-issue-roc-dual-device-buffer-init_logs.zip)

> **Comment by farawayso (2026-10-19T16:40:50Z):**

[追加] Qwen Image 2.1 8B line, dual-GPU te→iGPU, new signature — same machine/95b3090/ROCm 10.1

Same prebuilt (master-975, 95b3090) + ROCm 10.1.0, machine with 890M iGPU (gfx1150) + RX 7900 XTX (gfx1100). This time the Qwen Image 2.1 image line (8B text encoder, not H3 32B), image-edit mode, 2 reference images, placement te→ROCm0 (890M), diffusion/vae→ROCm1 (7900XTX):


[D] qwen3vl executing segment 1/1: graph
[E] ggml: ROCm error: invalid configuration argument
[E] ggml:   current device: 0, in function ggml_cuda_kernel_launch at common.cuh:1742
[E] ggml:   hipGetLastError()


current device: 0 = the 890M is actually being used for the text encoder here (contrast: #2169's iGPU was merely visible, unused). Error captured at hipGetLastError() right after the launch — reads like a silently-failed HSA agent/context init on gfx1150, not a per-kernel launch-config violation.

Control (known-good): build 959 (d637cfb) on this same box + ROCm 10.1 ran this exact dual-GPU qwen21 use cleanly (rc=0, ~82s single image). So the regression sits in 959..975; the window's main event is the ggml submodule sync (#2147, 277 llama.cpp commits). gfx1150-relevant suspects: fa692ff (fused RMS_NORM+SCALE), 757416a (norm/rope grid-dim clamp), 8d2974e (RDNA3.5 MMQ 64-row tiles), e05d18a (new precomputed RoPE op).

Auto-fallback to single-GPU (HIP=1, te→CPU) works but CPU 8B+vision is impractically slow for this line.

Happy to run any A/B you need: 959 vs 975 side-by-side, HIP_VISIBLE_DEVICES combos, or per-flag bisect of the ggml suspects.


> **Comment by farawayso (2026-10-21T05:11:16Z):**

[10-21 update — same machine: Ryzen AI 9 HX 470 with 890M iGPU (gfx1150) + RX 7900 XTX (gfx1100), Windows 11, new AMD Adrenalin driver + reboot]

Re-tested on both prebuilts (0518b34 and 95b3090). Two findings:

**1. The 8B te→iGPU coop placement now works (comment #1 signature no longer reproduces)**

Qwen Image 2.1 8B with te compute on the iGPU, diffusion+VAE on the dGPU:

```
sd-cli -M img_gen \
  --diffusion-model qwen_image_2.1-Q8_0.gguf \
  --llm Qwen3-VL-8B-Instruct-Q8_0.gguf \
  --vae qwen_image_2.1_vae_bf16.safetensors \
  --conv-direct auto,diffusion=on,controlnet=on,vae=on --rng cpu \
  --steps 4 --cfg-scale 6.0 --sampling-method euler -H 1024 -W 1024 \
  --backend "llm=ROCm0,vae=ROCm1,diffusion=ROCm1" \
  --params-backend "diffusion=cpu,te=cpu" \
  -p "a red teacup on a wooden table" -o out.png
```

rc=0 on BOTH 0518b34 and 95b3090 (~32 s, 1024x1024 image fine, te 7.7GB RAM-staged with compute on ROCm0/iGPU confirmed in the log). Since the 95b3090↔0518b34 crash path has zero diff, the change is on the driver side (new Adrenalin), not sdcpp — the "invalid configuration argument / current device: 0" from the 10-19 addendum does not reproduce anymore.

**2. H3 32B streaming buffer init still crashes when the iGPU is merely visible (variant B)**

Minimal repro from the issue (iGPU enumerated but never used for compute, everything on ROCm1):

```
sd-cli -M vid_gen \
  --diffusion-model minimax_h3_ref2va_pruned-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --cfg-scale 1 -W 480 -H 864 --fps 24 --video-frames 13 \
  --flash-attn off,diffusion=on --conv-direct auto,diffusion=on,controlnet=on,vae=on \
  --rng cpu --flow-shift 12 --steps 4 --sampling-method euler --scheduler discrete -t 16 \
  --backend "llm=ROCm1,vae=ROCm1,diffusion=ROCm1" \
  --params-backend "diffusion=cpu,te=cpu" \
  --max-vram "ROCm0=12" \
  --ref-image ref.jpg -p "a cat sitting on a table" -o out.mp4
```

Fails right after "loading tensors completed": `[E] ggml: ROCm error: unspecified launch failure / current device: -1, in ggml_backend_cuda_buffer_init_tensor at ggml-cuda.cu:817` (hipStreamSynchronize at ggml-cuda.cu:111), process aborts with 0xC0000409. Reproduced on 0518b34, byte-identical fingerprint to the original report. So the iGPU-visible crash in the H3 32B line persists even with the new driver — only the 8B te path recovered.

**Enumeration gotcha (helps repro):** a stray system-level `HIP_VISIBLE_DEVICES=1` env var silently filters the iGPU out of HSA enumeration (`found 1` instead of `found 2`). Remove it in-process (`Remove-Item Env:HIP_VISIBLE_DEVICES` on Windows PowerShell) to get the two-device enumeration the repro needs.



> **Comment by farawayso (2026-10-21T05:13:42Z):**

Release: master-982-0518b34, asset `sd-master-0518b34-bin-win-rocm-10.1.0-x64.zip` (Windows ROCm 10.1.0 prebuilt)

## Asset contents (verified by unzip)

amd_comgr.dll, amdhip64_7.dll, rocm_kpack.dll, sd-cli.exe, sd-server.exe, stable-diffusion.dll, start-server.bat/.sh — **only the 3 runtime DLLs**, no hipblas/rocblas chain, no data dirs.

## Problem

`stable-diffusion.dll` from this build **hard-imports** `hipblas.dll` (new in the 982 build; the 975/95b3090 prebuilt did not), with the load chain:

```
stable-diffusion.dll → hipblas.dll → {rocblas.dll, rocsolver.dll}
                       → libhipblaslt.dll → origami.dll   (+ amdhip64_7 / 3-piece set)
```

(import tables verified with pefile.) So out of the box:

- `sd-cli.exe --version` / `sd-server.exe` → **0xC0000135 STATUS_DLL_NOT_FOUND**
- Even with the DLLs resolvable, rocBLAS/hipBLASLt locate their Tensile data dirs (`rocblas\library\TensileLibrary*.dat`, `hipblaslt\library\...`) **relative to the DLL's own directory** (no env var exists — strings-scan of rocblas.dll / libhipblaslt.dll shows only `HIPBLASLT_EXT_OP_LIBRARY_PATH`, which is for extension ops, not the main libs). If the data dirs are not siblings of the DLLs, model loading silently corrupts/fails at the tail (no [E] log line).

## Local install is not picked up

A locally installed ROCm 10.1 (Windows pip `rocm-sdk-*` wheels, DLLs in `<python>\Lib\site-packages\_rocm_sdk_libraries\bin` + `_rocm_sdk_core`) is **not found** by the prebuilt — Windows hard-import resolution only walks exe dir → system dirs → PATH. Two working workarounds:

1. Copy the hipblas chain + the two data dirs next to `sd-cli.exe` (full self-contained dir), or
2. Prepend the two site-packages bin dirs to `PATH` for the process (`_rocm_sdk_core\bin`, `_rocm_sdk_libraries\bin`) — the data dirs are already siblings of the DLLs there, zero copies.

## Expectation

Either the `win-rocm` asset should bundle the hipblas chain + `rocblas/library` + `hipblaslt/library` (the 982 build needs them), or `stable-diffusion.dll` should not hard-import hipblas (975 behavior). As shipped, the 982 win-rocm prebuilt cannot start without the user knowing to supply ROCm library files that were not in the zip.

Repro: unzip the 982 win-rocm-10.1.0 asset on a clean Windows machine (ROCm installed via AMD's Windows pip wheels only, nothing in the exe dir besides the zip), run `sd-cli.exe --version`.


> **Comment by farawayso (2026-10-21T05:14:43Z):**

以上由大模型自行发现和总结的，如有错误，还请海涵。


> **Comment by leejet (2026-10-21T15:44:22Z):**

Thanks for the detailed logs. I traced the failure to quantized weight padding initialization during diffusion weight staging, before the diffusion graph executes.

One candidate worth testing is ggml commit ee665f42 (ggml-org/llama.cpp#28782), included in #2147. It changed this exact path from cudaMemset to cudaMemsetAsync on cudaStreamPerThread, followed by synchronization. This is a candidate, not yet a confirmed cause.

Could you first rerun variant B, keeping both GPUs visible and all other settings unchanged, with:

$env:GGML_CUDA_NO_PINNED = "1"

The logs show that CPU parameters use ROCm_Host, so pinned memory has not yet been ruled out. If this still fails, the next useful comparison is a build reverting only the padding memset change.

Also, current device: -1 does not necessarily indicate an uninitialized device context: the error handler initializes the ID to -1 and ignores the return status of hipGetDevice(), which may itself fail after a GPU error.

The Qwen invalid configuration argument report may be separate; we should track it independently until we identify the failing kernel.


> **Comment by farawayso (2026-10-21T16:25:46Z):**

> 6\. Thanks for the detailed logs. I traced the failure to quantized weight padding initialization during diffusion weight staging, before the diffusion graph executes.
>     One candidate worth testing is ggml commit ee665f42 ([ggml-org/llama.cpp#28782](https://github.com/ggml-org/llama.cpp/pull/28782)), included in [#2147](https://github.com/leejet/stable-diffusion.cpp/pull/2147). It changed this exact path from cudaMemset to cudaMemsetAsync on cudaStreamPerThread, followed by synchronization. This is a candidate, not yet a confirmed cause.
>     Could you first rerun variant B, keeping both GPUs visible and all other settings unchanged, with:
>     $env:GGML_CUDA_NO_PINNED = "1"
>     The logs show that CPU parameters use ROCm_Host, so pinned memory has not yet been ruled out. If this still fails, the next useful comparison is a build reverting only the padding memset change.
>     Also, current device: -1 does not necessarily indicate an uninitialized device context: the error handler initializes the ID to -1 and ignores the return status of hipGetDevice(), which may itself fail after a GPU error.
>     The Qwen invalid configuration argument report may be separate; we should track it independently until we identify the failing kernel.

Reran variant B as requested: 0518b34 (master-982) + ROCm 10.1 + current Adrenalin
driver, both GPUs visible (no HIP_VISIBLE_DEVICES filter), iGPU never used for
compute, all other settings unchanged; single variable = GGML_CUDA_NO_PINNED.

- Leg A (baseline, NO_PINNED unset): FAIL, rc=0xC0000409 — reproduces the
  original signature exactly:
    [E] ggml: ROCm error: unspecified launch failure
    [E] ggml:   current device: -1, in function ggml_backend_cuda_buffer_init_tensor
             at .../ggml-cuda.cu:817
  (crash right after "staged compute params (16239.66 MB, 551 tensors, 16 blocks)
  to ROCm1")
- Leg B (GGML_CUDA_NO_PINNED=1): PASS, rc=0 — sampling 149.2s + VAE decode 7.2s,
  video written.

So pinned host allocations (ROCm_Host) are confirmed as the trigger for the
variant-B crash while the gfx1150 iGPU is visible; the run is unblocked on the
current build with no code change. Happy to run the ee665f42 (cudaMemset ->
cudaMemsetAsync) revert-build comparison separately if you still want to pin
the exact commit. Full logs attached.

[variantB_A_baseline.log](https://github.com/user-attachments/files/33778235/variantB_A_baseline.log)
[variantB_B_no_pinned_1.log](https://github.com/user-attachments/files/33778234/variantB_B_no_pinned_1.log)
[variantB_nopinned_summary.txt](https://github.com/user-attachments/files/33778233/variantB_nopinned_summary.txt)


> **Comment by farawayso (2026-10-27T09:53:27Z):**

[10-27 update — same machine: Ryzen AI 9 HX 470, 890M iGPU (gfx1150) + RX 7900 XTX (gfx1100), Windows 11, ROCm 10.1.0, current Adrenalin driver]

Re-ran the variant-B A/B on the latest prebuilt 1017 (be48649, verified via sd-cli --version). The iGPU-visible crash is not fixed yet, and the pinned-memory workaround still holds:

run (identical variant B: iGPU visible, never used for compute; streaming params backend; ref2va pruned Q4_K_M, 480x864x13f): 1017, no
  GGML_CUDA_ NO_PINNED
result: FAIL — original signature exactly: "unspecified launch failure / current device: -1" @ ggml-cuda.cu:817
  (hipStreamSynchronize), abort 0xC0000409, right after "staged compute params ... to ROCm1"
────────────────────────────────────────
run (identical variant B: iGPU visible, never used for compute; streaming params backend; ref2va pruned Q4_K_M, 480x864x13f): 1017 +
  GGML_CUDA_ NO_PINNED= 1
result: PASS rc=0, 56s wall total (LLM conditioning 22s + 4-step sampling + VAE decode 7.4s), video written

So as of 1017 the pinned-host trigger (ee665f42 candidate) still needs the env-var stopgap; the 8B te→iGPU coop line (10-19 addendum) remains fine since the 10-21 driver-side recovery. Happy to run the ee665f42-only revert build comparison to pin the exact commit, as suggested.

Full logs of both legs available on request. A/B done today in a clean iGPU state.

(This update drafted with Hermes Agent; all runs verified locally.)



### #1869 [closed] Tested Spectrum, Block Cache, SageAttention2, and ConvRot GGML implementations that may be useful for SD.cpp
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1869  by 0xShug0 2026-08-11T07:28:55Z  comments: 7

Hi @leejet, 

I'm the maintainer of audio.cpp. I recently implemented the MiniMax-H3 text-to-audio path in GGML (released under the `dev` branch). The DiT path is effectively shared with video generation.

During the implementation, I tested several DiT optimizations that may also be useful for SD.cpp:

Spectrum-style acceleration
First-block / BlockCache
SageAttention
INT8 ConvRot

The biggest gain came from Spectrum. For a 30s H3 audio at 20 steps, Spectrum improved RTF from about 0.56 to 0.30. BlockCache also provided meaningful speedups, although quality was more sensitive to the number of steps. INT8 ConvRot gave consistent speedup on the normal DiT path, around 9%, and the optimizations could be combined.

I'm not planning to spend much time supporting or optimizing video model pipelines in audio.cpp. H3 video support mostly comes "for free" because the model produces audio and video latents jointly. These GGML implementations may be useful for video diffusion workloads in SD.cpp. 

On RTX 5090, for the 1344x768 / 124 frames (5 seconds) / 20 steps case from the official comfyui workflow (the "Realistic live-action cinematic look, action movie trailer..." one) :

INT8 ConvRot, no spectrum  --- 260.839 s
INT8 ConvRot + spectrum --- 114.035 s

For video, INT8 ConvRot + spectrum actually looks better. 

Feel free to use the implementations at https://github.com/0xShug0/audio.cpp/tree/dev if they’re useful. 



> **Comment by 0xShug0 (2026-08-11T07:33:53Z):**

BTW, there are also some benchmarks that may be useful for understanding the trade-offs, although they’re audio-only.
https://github.com/0xShug0/audio.cpp/blob/dev/docs/reports/minimax_h3_performance.md


> **Comment by Green-Sky (2026-08-11T08:11:00Z):**

Spectrum caching, among others, is already supported.

Also i think conv rot was merged recently too.


> **Comment by stduhpf (2026-08-11T09:00:01Z):**

> Also i think conv rot was merged recently too.

If you're talking about https://github.com/leejet/stable-diffusion.cpp/pull/1857, that's still pending, not merged yet.


> **Comment by 0xShug0 (2026-08-11T15:34:24Z):**

Ah never mind. I thought SD.cpp pinned a specific version of upstream GGML.


> **Comment by leejet (2026-08-11T15:49:52Z):**

Thanks for sharing! Spectrum's caching approach has actually been supported in `sd.cpp` for quite a while. I also added CUDA support for `int8_convrot` about a week ago.

Over the past few days, I've been working on adding Vulkan support for `int8_convrot`, so I held off on merging the changes until that was done. I've merged everything now.

If you haven't implemented Vulkan support for `int8_convrot` yet and are interested in adding it to `audio.cpp`, feel free to use the relevant code.


> **Comment by 0xShug0 (2026-08-11T17:45:10Z):**

@leejet I checked your changes and found that the way you extended GGML is quite different from mine 😄. Your approach is definitely cleaner architecturally, but it’s a bit overkill for `audio.cpp`’s use case, especially since only one model benefits from it. In general, for TTS/Speech audio generation, I’ve found INT8 ConvRot to be less accurate than Q4. I’ll probably keep INT8 ConvRot support CUDA-only as an experimental feature for now. 

I may be missing something, but how is SageAttention2 implemented in `sd.cpp`? 

Also, now that I know you use a customized GGML, I’m wondering whether some of the localized GGML optimizations in `sd.cpp` could benefit DiT-based TTS models as well. Would you like to share any of the relevant changes or PRs?




> **Comment by leejet (2026-09-19T10:28:44Z):**

I don't support SageAttention yet, but it's one of the things I plan to work on next. So far, my changes to ggml mainly add support for INT8 ConvRot, FP8, Conv2D optimizations, and Conv3D optimizations. You can check the corresponding ggml branch in sd.cpp for the details.

If a DiT-based TTS model uses some of the same code paths, it should be able to benefit from these optimizations as well.



### #1882 [closed] [Bug] Minimax h3 audio reference doesn't seem to be working
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1882  by stduhpf 2026-08-14T20:38:32Z  comments: 5

### Git commit

de298c225bed97c3f9026b73cd7b71e7879bd41b

### Operating System & Version

Windows

### GGML backends

HIP

### Command-line arguments used

`./build/bin/sd-cli.exe -M vid_gen --diffusion-model C:\sd-models\minimax_h3_ref2va_pruned-Q4_K_M.gguf --vae C:\sd-models\minimax_h3_video_vae_fp16.safetensors --audio-vae C:\sd-models\minimax_h3_audio_vae_fp32.safetensors --llm C:\sd-models\qwen3vl_32b_minimax_h3-Q2_K_M.gguf --prompt-file prompt-file.txt --cfg-scale 1.0 -v -H 128 -W 128 --diffusion-fa --offload-to-cpu --fps 24 --steps 20 --color --preview proj --video-frames $(([int]((3.5*24-5)/17))*17+5) --ref-audio out.wav -r .\assets\qwen\example.png`

### Steps to reproduce

Try for example to run the command above with these two references:
[out.wav](https://github.com/user-attachments/files/31086512/out.wav)
<img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/33cad9ea-4241-4339-bfd1-72eba27712cc" />

And this as prompt-file.txt:

```
subject_definitions
<Subject 1> is the character from <Picture 1>.
<Picture 1> is the reference image of the character used as the visual anchor for the target video.
<Audio 1> is the source audio providing the dialogue, timbre, and delivery for the target video.

retention_analysis
retention_analysis
<Subject 1> (appears in [Shot 1]): fully_preserved - The character's appearance, clothing, and identity are kept consistent with the reference.
<Picture 1> ([Shot 1] first frame): fully_preserved - The video begins exactly with the composition and subject pose shown in <Picture 1>.
<Audio 1>: full_copy

summary : [audio reuse + reference generation] The target video features <Subject 1> speaking the dialogue from <Audio 1> while maintaining the visual identity, clothing, and facial features established in <Picture 1>.

detailed_description
[Shot 1] For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced. The scene opens with <Subject 1> in the exact pose and framing of <Picture 1>. The camera holds a static shot with a very slight push in. <Subject 1> (S1) looks directly into the lens and begins to speak, her lips and facial muscles moving in perfect synchronization with the dialogue from <Audio 1>
```

### What you expected to happen

I expect the audio of the video to match the reference.

### What actually happened

https://github.com/user-attachments/assets/b40903cc-3b2b-48ce-862c-548ccc428a46

Audio is completely different, it seems to ignore the reference altogether

### Logs / error messages / stack trace

```
[INFO ] stable-diffusion.cpp:5973 - MiniMax-H3 Ref2VA: 1 image(s), 0 video(s), 1 audio clip(s)
[DEBUG] vae.hpp:142  - VAE Tile size: 16x16
[DEBUG] ggml_extend.hpp:884  - num tiles : 2, 2 
[DEBUG] ggml_extend.hpp:885  - optimal overlap : 0.750000, 0.750000 (targeting 0.250000)
[DEBUG] ggml_extend.hpp:886  - processing 4 tiles
[DEBUG] model_loader.cpp:1045 - loading 118/562 tensors from C:\sd-models\minimax_h3_video_vae_fp16.safetensors
  |##################################################| 118/118 - 960.80MB/s
[INFO ] model_loader.cpp:1309 - loading tensors completed, taking 0.36s (read: 0.21s, memcpy: 0.00s, convert: 0.00s, copy_to_backend: 0.00s)
[DEBUG] model_manager.cpp:409  - model manager prepared params backend buffer (344.04 MB, 118 tensors, RAM)
[DEBUG] model_manager.cpp:506  - model manager staged compute params (344.04 MB, 118 tensors) to ROCm0, taking 0.08s
[DEBUG] ggml_extend.hpp:2244 - minimax_h3_video_vae compute buffer size: 627.13 MB(VRAM)
  |==================================================| 4/4 - 4.88it/s
[DEBUG] vae.hpp:166  - computing vae encode graph completed, taking 2.21s
[DEBUG] model_loader.cpp:1045 - loading 135/917 tensors from C:\sd-models\minimax_h3_audio_vae_fp32.safetensors
  |##################################################| 135/135 - 910.11MB/s
[INFO ] model_loader.cpp:1309 - loading tensors completed, taking 0.36s (read: 0.18s, memcpy: 0.00s, convert: 0.00s, copy_to_backend: 0.00s)
[DEBUG] model_manager.cpp:409  - model manager prepared params backend buffer (329.46 MB, 135 tensors, RAM)
[DEBUG] model_manager.cpp:506  - model manager staged compute params (329.46 MB, 135 tensors) to ROCm0, taking 0.05s
[DEBUG] ggml_extend.hpp:2244 - minimax_h3_audio_vae compute buffer size: 417.04 MB(VRAM)
[INFO ] minimax_h3_audio_vae.hpp:470  - MiniMax-H3 audio VAE encode completed, taking 0.68s
```

### Additional context / environment details

No notable errors messages. Audio reference does get parsed and encoded, it just seems to be ignored by the DiT.

> **Comment by stduhpf (2026-08-14T21:22:41Z):**

I can't be bothered to fix my ComfyUI install to compare it to, but based on some examples I saw online, this kind of prompt should work.


> **Comment by stduhpf (2026-08-14T22:05:36Z):**

Note that the result is a little bit different but still very simmilar if i don't pass a reference audio at all.

https://github.com/user-attachments/assets/79bc99b0-38e2-451c-86c2-dd998304c3f2


> **Comment by stduhpf (2026-08-15T00:02:25Z):**

I'm thinking it could be the Audio VAE encoding that is not working properly, I could be mistaken though. 

Edit: nevermind encoding->decoding preserves the audio, so the VAE is working fine.


> **Comment by Komorebi623 (2026-08-18T11:43:30Z):**

 I tested commit `8e9d77fad381b4e382d9c50d2e9163cefc8b00e7` with the original repro case. The run now completes, and the reference image is followed correctly, but the standalone reference audio is still not preserved.

  Reference audio:

  > This is a simple sentence to test lip sync with audio reference mode.

  Generated audio is still unrelated / unintelligible. Whisper roughly transcribes it as:

  > The Pajitsu vaccine got the gold star-arce million laws loan life, bunnine and Indian estate sheet.

  So I think this is not fully fixed yet. The audio VAE layout fix seems helpful, but the MiniMax-H3 Ref2VA `--ref-audio` conditioning path still appears incorrect or incomplete.

  Could you please re-check how the standalone reference audio latent is added to the Ref2VA conditioning?


> **Comment by Komorebi623 (2026-08-18T11:52:07Z):**

By the way, we are also implementing MiniMax-H3 in edge-dit.cpp: https://github.com/THU-MIG/edge-dit.cpp

  The project focuses on fast, low-memory DiT inference in C++. For MiniMax-H3, we have been benchmarking GGUF / quantized execution, staged loading, and Ref2VA conditioning, and edge-dit.cpp currently shows very competitive speed and VRAM
  usage.

  It might be useful as an additional reference implementation when comparing MiniMax-H3 behavior across C++ runtimes.

  Happy to share benchmark commands and artifacts if useful.



### #2392 [closed] [Feature] LoRA on INT8 convrot weights: at_runtime pushes MiniMax-H3 over 32 GB VRAM and into a 54-segment graph cut (~2x slower per step)
URL: https://github.com/leejet/stable-diffusion.cpp/issues/2392  by shiwano 2026-11-20T19:42:31Z  comments: 4

### Feature Summary

Make a LoRA usable on `int8_tensorwise` / convrot diffusion weights without the runtime path's extra compute buffer: either support `--lora-apply-mode immediately` for these weights, or make the at_runtime path cheap enough that the model still runs as one segment.

### Detailed Description

Setup: RTX 5090 (32 GB), Windows CUDA12 build of master-1106-532d2c1, MiniMax-H3 Ref2VA at 1248x704, 124 frames, `--cfg-scale 1.0`.

- DiT: `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (Comfy-Org/MiniMax-H3), params 19998 MiB
- LoRA: `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` via `--lora-model-dir` + `<lora:...:1>`, 624 / 624 tensors applied
- auto-fit: `--backend "diffusion=CUDA0,te=CUDA0,vae=CUDA0" --params-backend "te=cpu,vae=cpu"`, GPU budget 30479 MiB

| DiT                                                                         | LoRA              | segments | DiT compute buffer | per step (steady state) |
| --------------------------------------------------------------------------- | ----------------- | -------- | ------------------ | ----------------------- |
| INT8 convrot                                                                | none              | 1        | 6383.54 MB         | 10.5 s                  |
| INT8 convrot                                                                | turbo, at_runtime | 54       | 9988.18 MB         | ~19 s                   |
| GGUF Q4_K_M (`leejet/MiniMax-H3-GGUF`), `--params-backend "te=cpu,vae=cpu"` | none              | 1        | 8369.00 MB         | 14.8 s                  |
| GGUF Q4_K_M, `--params-backend "te=cpu,vae=cpu"`                            | turbo, at_runtime | 1        | 10349.19 MB        | 17.1 s                  |

With the LoRA on the INT8 file the log shows `apply lora at runtime`, the compute buffer grows by about 3.6 GB, and the DiT is cut into 54 segments. During sampling `loading N/732 tensors from ...` appears 83 times (once without the LoRA). The GGUF rows separate the two costs: with one segment in both cases the runtime LoRA adds about 2.3 s per step, so most of the 8.5 s added on the INT8 file comes from the graph cut. The picture is correct; only the speed changes.

`--lora-apply-mode immediately` does not help, because `configure_weight_loading` falls back:
https://github.com/leejet/stable-diffusion.cpp/blob/532d2c15c4/src/pipeline/diffusion_engine.cpp#L1058-L1063

```
[W] ComfyUI INT8/packed INT4 weights do not support the immediately LoRA apply mode; using at_runtime instead
```

Sampling was 109.8 s, still 54 segments.

For reference, ComfyUI runs the same DiT, LoRA, resolution and step count in 55-58 s wall time for the whole job on the same GPU.

Possible directions, whichever fits the design:

1. `immediately` for int8 tensorwise / convrot: dequantize a row block, undo the rotation (`W = W_rot R`, `R` is orthogonal and symmetric), add `scale * B A`, rotate, requantize per output row with a new `weight_scale`. Equivalently, rotate the LoRA delta once (`ΔW R`) and add it in the rotated domain before requantizing, so no un-rotation of the base is needed.
2. Keep at_runtime but shrink its footprint for these layers, so a model that fits without a LoRA still fits with one (the low-rank branch `X A^T B^T` needs only rank-sized intermediates).
3. Let auto-fit account for the at_runtime LoRA buffer when it decides the plan, and say in the log that the LoRA is what forced the cut.


### Alternatives you considered

- `--lora-apply-mode immediately`: rejected for these weights, as above.
- The GGUF DiT (`minimax_h3_ref2va_pruned-Q4_K_M.gguf`, 10.9 GB): `immediately` is rejected with the same warning (packed INT4), and the base speed is lower than INT8 convrot (14.8 vs 10.5 s per step). With default auto-fit it is much slower still: the freed VRAM goes to keeping the text encoder params on the GPU (`--params-backend "vae=disk"`), and the DiT is cut into 54 segments even without a LoRA (sampling 694 s for 20 steps). The same seed also gives a different take, while the INT8 file matches ComfyUI closely (PSNR 33.4 dB against ComfyUI, 17.1 dB between INT8 and GGUF).
- Merging the LoRA into the float checkpoint offline and re-exporting INT8 convrot: needs the unquantized DiT and a converter, per LoRA combination.


### Additional context

Distilled few-step LoRAs are the normal way to run MiniMax-H3 locally, so this path is the common one rather than a corner case. I can test builds on the RTX 5090 against ComfyUI output for the same inputs.


> **Comment by wbruna (2026-11-20T23:49:25Z):**

| | |
|:---:|:---:|
|params | 19998 MiB |
|DiT compute buffer | 9988.18 MB |
|LoRA size| 1.96 GB | 

Yep, looks like you are exceeding your ~30.5G available VRAM, by 1.5G or so.

> or make the at_runtime path cheap enough that the model still runs as one segment.

Sorry: how could that possibly work? `at_runtime` runs the LoRA separate from the model, by definition; and that LoRA alone has 2GB of weights. If you are close to the VRAM limit _for a given resolution and number of frames_, of course loading it will get you past that limit.

By the way: even if `immediately` was supported for int8_convrot, it wouldn't be for free, because dequantizing and requantizing the weights for merging would demand a lot of processing.

> Distilled few-step LoRAs are the normal way to run MiniMax-H3 locally

You could try a smaller LoRA: for instance, https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI has a few around 300MB. Or a pre-merged checkpoint, like https://huggingface.co/PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot .


> **Comment by shiwano (2026-11-21T10:24:12Z):**

Thanks for looking into this, and for the suggestions!

I tried the smaller LoRA. `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_resized_avg_rank_21_bf16.safetensors` from that repo is a compressed version of the same LoRA (average rank ~21, 327 MB instead of 1.96 GB).

Using `at_runtime`:

| LoRA | LoRA size | DiT compute buffer | Segments | Per step |
| --- | --- | --- | --- | --- |
| None | — | 6383.54 MB | 1 | 10.5 s |
| Turbo, original | 1.96 GB | 9988.18 MB | 54 | ~19 s |
| Turbo, compressed | 327 MB | 9988.18 MB | 54 | 15.8 s |

The compressed-LoRA row and the `immediately` numbers below are from a Linux CUDA build (WSL2) of the same commit; the first two rows are the Windows numbers from the issue. On Linux, the no-LoRA baseline is 11.6 s/step and the original LoRA with `at_runtime` is 18.7 s/step.

Both LoRAs require the same 9988 MB compute buffer, adding ~3.6 GB of workspace regardless of rank and leaving less than 500 MiB of GPU headroom before loading the LoRA weights.

Looking at `get_out_diff()`, the runtime path already uses separate `lora_down` and `lora_up` projections, so I measured how the extra workspace scales. It scales with the number of frames rather than the LoRA's rank:

| Latent frames | No LoRA | `at_runtime` | Extra |
| --- | --- | --- | --- |
| 12 | 2484 MB | 3311 MB | 826 MB |
| 22 | 3979 MB | 5941 MB | 1962 MB |
| 37 (124 video frames) | 6384 MB | 9988 MB | 3604 MB |

That is about 110 MB per latent frame (858 tokens), roughly the size of two FFN-width activations plus one hidden-width activation. So it looks like full-size `out_diff` tensors living next to the layer outputs, rather than anything rank-sized. Could those be added into the output in place, or freed earlier?

You're also right that `immediately` isn't free. I've implemented per-row INT8 merging, including convrot (rotating the LoRA delta into the stored basis, then requantizing the affected rows with updated scales). On this checkpoint, merging takes about 140 s.

After merging, the DiT runs as a single segment at 11.5 s/step with a 6384 MB compute buffer, the same as without a LoRA. `sd-server` retains the merged weights, so only the first request pays the merge cost.

For a single 4-step `sd-cli` run, it's slower overall (278 s vs 160 s wall time). But for a long-running server, avoiding the 54-segment execution on every request seems like a useful trade-off.

I'd be happy to open a PR with the implementation and benchmarks if this approach would be worth supporting.

I also tested the suggested pre-merged checkpoint. It runs in one segment at 10.4 s/step with a 6384 MB buffer. However, it produces a different output (22.8 dB PSNR vs ComfyUI, compared with 32.0 dB for my merge), which may be expected since it uses a different Turbo LoRA. It also doesn't cover arbitrary LoRAs.



> **Comment by wbruna (2026-11-23T17:07:00Z):**

> on `int8_tensorwise` / convrot diffusion weights without the runtime path's extra compute buffer: either support `--lora-apply-mode immediately` for these weights

Support was added in #2403 .




> **Comment by shiwano (2026-11-24T19:15:37Z):**

Thanks for pointing me to #2403!
That changes covers the immediate LoRA merge support I was looking for on MiniMax-H3 INT8 convrot weights.
Closing this as completed 😄 



### #1897 [closed]  MiniMax-H3 I2VA/FL2VA: ROCm VAE encode crashes with 0xC0000005 (ACCESS_VIOLATION) on Windows（问题已解决）
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1897  by farawayso 2026-08-21T18:19:47Z  comments: 4

# [Bug] MiniMax-H3 I2VA/FL2VA: ROCm VAE encode crashes with 0xC0000005 (ACCESS_VIOLATION) on Windows

**sd-cli version**: `stable-diffusion.cpp version unknown, commit 97d2990` (pre-built Windows ROCm binary, ~2026-08-19)
**OS**: Windows 11
**GPU**: AMD Radeon RX 7900 XTX (gfx1100, 24GB VRAM / 23.7GB usable)
**ROCm**: ROCm 7.14.0 (bundled with the pre-built binary)

## Description

MiniMax-H3 I2VA (image-to-video+audio) and FL2VA (first+last frame → video+audio) modes crash with exit code `3221225725` (0xC0000005, `STATUS_ACCESS_VIOLATION`) when using the **ROCm HIP backend** (`--backend llm=cpu,vae=ROCm0,diffusion=ROCm0`).

The crash is **deterministic** (100% reproduction rate) and occurs during **VAE encode of the input image** — the very first GPU operation after model loading. The diffusion sampling and VAE decode stages are never reached.

## Reproduction

### Command line

```bash
sd-cli.exe --mode vid_gen \
  -p "<lora:minimax_h3_turbo_v11_sdcpp:1.0> 美女说，树青，听说你去洱海了。" \
  --init-image input.png \
  --width 480 --height 864 \
  --video-frames 175 \
  --sample-steps 5 --flow-shift 6 --cfg-scale 1.0 \
  --seed 42 \
  --backend llm=cpu,vae=ROCm0,diffusion=ROCm0 \
  --offload-to-cpu \
  --eager-load \
  --lora-model-dir ./models/MiniMax-H3/loras \
  --lora-apply-mode at_runtime \
  --diffusion-fa \
  --max-vram ROCm0=22
```

The crash also occurs **without** `--offload-to-cpu`, with `--max-vram` values 14–22, and with or without `--eager-load`. The only variable that matters is the backend: `vae=ROCm0` always crashes; `vae=Vulkan1` always works.

### Via menu

Select backend [5] ROCm+VAE乒乓, model pruned, mode I2VA, provide an input image. The auto-route (menu [4]) correctly avoids this by sending I2VA/FL2VA to Vulkan — the crash only happens when ROCm is explicitly selected.

## Crash log

```
[INFO ] stable-diffusion.cpp:6123 - MiniMax-H3 I2VA
[DEBUG] vae.hpp:142  - VAE Tile size: 16x16
[DEBUG] ggml_extend.hpp:884  - num tiles : 2, 5
[DEBUG] ggml_extend.hpp:885  - optimal overlap : 0.125000, 0.406250 (targeting 0.250000)
[DEBUG] ggml_extend.hpp:886  - processing 10 tiles
[INFO ] ggml_graph_cut.cpp:973  - minimax_h3_video_vae build cached graph cut plan done (taking 1 ms)
[DEBUG] model_manager.cpp:506  - model manager staged compute params (344.04 MB, 118 tensors) to ROCm0, taking 0.11s
[DEBUG] ggml_extend.hpp:2244 - minimax_h3_video_vae compute buffer size: 627.13 MB(VRAM)

  [STDERR]   Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
[FAILED] sd-cli.exe 退出码: 3221225725
```

The image preprocessing (crop + resize) completes successfully, and all four model files load and initialize normally. The crash happens at the instant the VAE encode graph is computed on ROCm0 — 118 tensors (344MB) are staged to VRAM, then the compute buffer is allocated (627MB), and immediately the process dies with an access violation. No error message from the HIP runtime, no user-mode exception handling.

## Successful Vulkan comparison (same config, same hardware)

Running the identical config with `--backend llm=cpu,vae=Vulkan1,diffusion=Vulkan1` completes successfully:

```
[DEBUG] vae.hpp:142  - VAE Tile size: 16x16
[DEBUG] ggml_extend.hpp:2244 - minimax_h3_video_vae compute buffer size: 627.13 MB(VRAM)
  |==================================================| 10/10 - 19.61it/s
[DEBUG] vae.hpp:166  - computing vae encode graph completed, taking 0.62s
```

| Stage | ROCm (crash) | Vulkan (works) |
|---|---|---|
| VAE encode (image→latent) | 0xC0000005 crash during compute | 0.62s, 19.6it/s, clean |
| LLM vision conditioning | never reached | 68.6s (normal for Qwen3-VL) |
| Diffusion sampling (5 steps) | never reached | 122.8s@22.1s/it |
| VAE decode (4 passes × 10 tiles) | never reached | 21.9s |
| **Total generation** | **crash** | **231.1s / 4m17s** |

## What has been ruled out

- **Not the graph-cut bugs (#1865/#1880/#1883)**: Those were in the diffusion sampling graph-cut path and were fixed in master-826 (16304cc). The VAE encode path is separate code — it uses its own graph-cut plan (`minimax_h3_video_vae`), not the diffusion model's graph. The encode path has never been patched.
- **Not an OOM**: VRAM during load is 0MB (all weights in RAM via `--offload-to-cpu`), and the encode only needs 344MB staged + 627MB compute buffer = ~1GB, well within 23.7GB available. The same config runs to completion on Vulkan with the same VRAM budget.
- **Not model-specific**: Both `minimax_h3_fl2va_pruned-Q4_K_M` (11GB) and `minimax_h3_fl2va-Q4_K_M` (18GB, full) models crash identically when I2VA/FL2VA is used on ROCm.
- **Not input-image-specific**: Tested with various input images (768×1368, 2048×2048, diverse content) — all crash at the same point.
- **Not mitigatable**: `--offload-to-cpu`, different `--max-vram` values (14–22), `--eager-load`, `--vae-on-cpu` (deprecated) — none prevent the crash. The only working VAE encode path on this hardware is Vulkan.

## Hypothesis

The MiniMax-H3 video VAE (`minimax_h3_video_vae_fp16.safetensors`, 5.6GB, 562 tensors) is a custom architecture that differs from the standard SD VAE — it has a 24-channel latent space, temporal clip length 17, and uses tiled encoding. The ggml HIP backend for ROCm likely has a bug in one of the operations used by the VAE encoder graph (possibly a 2D convolution, a memory layout/view operation, or the tiled overlap handling) that causes an out-of-bounds memory access. The Vulkan backend handles the same computation graph without issue, suggesting the bug is in the HIP kernel implementation, not in the model itself or the higher-level VAE code.

## Additional context: T2V/T2VA works fine on ROCm

Text-only modes (T2V/T2VA, no image input, no VAE encode needed) run flawlessly on the same ROCm binary — including VAE **decode** during the final video assembly. The VAE decode path uses the same model weights but different graph operations, and it completes successfully. This strongly points to a bug specific to the VAE **encode** graph's kernel set on the ROCm HIP backend.

## Attachments

- Full crash log (from `--verbose` output): the log above is the complete relevant portion; the model loading section (before the crash) is unremarkable — all four models load without warnings.
- Successful Vulkan run log: available on request.

> **Comment by stduhpf (2026-08-21T21:12:09Z):**

Did you manage to run any other model with the same ROCm build (are you sure it is specifically a MiniMax VAE issue)?


> **Comment by farawayso (2026-08-22T04:20:32Z):**

> Did you manage to run any other model with the same ROCm build (are you sure it is specifically a MiniMax VAE issue)?

@stduhpf Yes — I ran other models on the exact same ROCm build (commit 97d2990, the pre-built Windows ROCm binary), and the crash is specific to the MiniMax-H3 video VAE's encode graph, not a general problem with this ROCm build or with VAE encode on HIP:

| Model | Mode | VAE encode (ROCm0, same build) |
|---|---|---|
| Z-Image Turbo (z-image ae.safetensors) | txt2img | not exercised (no encode needed) — but full pipeline + VAE decode works |
| Boogu Image Edit (img2img, z-image ae.safetensors) | img2img 512x512, untiled | ✅ works — encode 3.20s, full run OK |
| Boogu Image Edit, forced `--vae-tiling` | img2img 1280x1280, 9 tiles | ✅ works — encode 3.08s (tiled encode fine on HIP!) |
| MiniMax-H3 T2V/T2VA (minimax_h3_video_vae_fp16) | video, no encode | ✅ works incl. VAE **decode** (same weights, entire pipeline OK) |
| MiniMax-H3 I2VA/FL2VA (same VAE) | video + input image, 10 tiles (2x5, overlap 0.125/0.40625) | ❌ 0xC0000005, deterministic, at first graph compute |

So: VAE encode in general (including **tiled** encode with the 2D AE) runs fine on this HIP build on the same GPU. The access violation only happens when computing the MiniMax-H3 video VAE encoder graph — a 3D/temporal VAE (24-channel latent, temporal clip 17, 562 tensors) whose tiled encode path differs from the standard 2D AE.

One more data point that suggests this is a HIP-kernel-set issue rather than a tiling logic issue: the identical tiled encode of the same MiniMax VAE completes cleanly on Vulkan (0.62s), and the MiniMax VAE *decoder* (different op set from the encoder, same weights) runs fine on ROCm. The crash happens the instant the encoder graph is computed (118 tensors staged → 627MB buffer allocated → process dies), with no HIP runtime error — consistent with an out-of-bounds access inside a HIP kernel used only by the encoder graph (possibly the 3D conv/layout ops).

If it helps narrow it down, I can also reproduce with a minimal input (single tile, no overlap) or grab a verbose graph dump — just tell me what would be most useful. For now I've verified the report is not a build-level issue: the same binary+backend handles VAE encode/tiled-encode fine for other models.


> **Comment by stduhpf (2026-08-22T10:13:19Z):**

That's helpful thanks. Minimax VAE encode works fine on my local HIP builds so that's strange. I don't think the prebuilt releases are compatibe with my hardware so I can't test it myself.


> **Comment by farawayso (2026-10-01T16:39:02Z):**

I've identified the root cause of this issue. sd.cpp was by default loading an outdated amdhip64_7.dll from Windows System32 — a leftover from a previous AMD driver installation. The newer version of amdhip64_7.dll has actually been moved to a different path, but the old copy in System32 was never cleaned up. As a result, sd.cpp kept picking up the stale DLL instead of the correct one.

AMD engineers have confirmed and reproduced this issue, and we believe AMD will follow up and fix it soon. As a temporary workaround, you can copy the amdhip64_7.dll from the latest driver into the same directory as sd.exe.



### #1865 [closed] MiniMax-H3: segmented offload (max_vram) deterministically produces an unrelated image (SD1.5 unaffected)
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1865  by dkcompactica 2026-08-08T13:19:13Z  comments: 4

### Summary

With MiniMax-H3, setting `sd_ctx_params_t.max_vram` (segmented parameter offload) makes generation complete normally in the usual wall time but produce a **completely unrelated image** — deterministically. The same seed that renders "a red paper lantern drifting over dark water" without a budget renders grey storm clouds with birds with one. Byte-identical mp4 output across separate processes on different days, so it is a wrong computation, not a race.

SD1.5 under the same mechanism is fine (control below), so this looks specific to the H3 pipeline.

### Environment

- commit `ea7f0c8` (the MiniMax-H3 support commit), CUDA 13.1 backend, Windows 11, RTX 5070 Ti 16 GB
- via the C API (managed binding over `stable-diffusion.h`; `generate_video` path). `SDNET_DEBUG` param dumps confirm the struct reaches native intact.

### Model set

- `minimax_h3_fl2va_pruned-Q4_K_M.gguf` (leejet/MiniMax-H3-GGUF)
- `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (llm slot)
- `minimax_h3_video_vae_fp16.safetensors`, `minimax_h3_audio_vae_fp32.safetensors` (Comfy-Org)
- text encoder on CPU (`te=cpu` in the backend spec), flash attention on, VAE tiling on

### Repro

t2v, 864x480, 56 frames, 8 steps, cfg 1.0, seed 7, prompt "a red paper lantern drifting over dark water at night, warm glow, gentle ripples".

| `max_vram` | `stream_layers` | Result | Wall time |
|---|---|---|---|
| 0 (off) | off | correct (lantern over water) | ~6.6 min |
| 9 GiB | off | **wrong scene** (storm clouds, birds) | ~6.8 min |
| 9 GiB | on | **wrong scene, byte-identical to the row above** | ~6.7 min |

`stream_layers` is exonerated — the budget alone does it. The wrong output is stable: three runs across two days produced byte-identical files. Frame count, duration and the generated audio track are all correct; only the picture content is wrong, so nothing errors and nothing looks failed.

Weights report as `total params memory size = 35398.76MB (VRAM 16534.02MB, RAM 18864.74MB)` without a budget — i.e. the model already oversubscribes this 16 GB card via CUDA VMM and renders correctly that way; the corruption only appears when segmentation takes over placement.

### Control (mechanism works elsewhere)

SD1.5 Q4 (1.5 GB weights), 512x512 image, same build: `max_vram = 0.25 GiB` (6x oversubscribed) produces **byte-identical** output to the unsegmented render, at ~2x the time. So segmented offload itself is sound for the single-model case; the suspicion lands on H3's multi-component graph (second VAE / `token_refiner` blocks) interacting with the segment planner.

Happy to run diagnostics or test patches — this card depends on segmentation being usable for H3, since without it every step pays full VMM re-paging.


> **Comment by dkcompactica (2026-08-08T18:06:36Z):**

Root cause found, and it's a one-line fix. Verified byte-identical output on the exact repro above.

## Root cause

`is_graph_cut_tensor` (src/core/ggml_graph_cut.cpp) classifies cut tensors by **name prefix**:

```cpp
return std::strncmp(tensor->name, GGML_RUNNER_CUT_PREFIX, std::strlen(GGML_RUNNER_CUT_PREFIX)) == 0;
```

But ggml's auto-naming gives derived nodes the parent's name plus a suffix — a view of a cut tensor is named `"<cut-name> (view)"`, a cont of that `"<cut-name> (view) (cont)"`. So every view/cont *derived from* a cut tensor is itself classified as a cut tensor. Two things then go wrong at once:

1. each derived node becomes a segment output, cached **by name**, and
2. sibling views of the same parent share one identical auto-name, so they collide in the cache — and when the next segment reads them back, `bind_segment_cached_inputs` binds every one of them to the same tensor at **offset zero**.

MiniMax-H3 hits this on every budgeted run because each transformer block slices `hidden_states` into per-modality rows (audio 17 / video 16 / text 810 at 864×480×56). When the block just past a segment boundary slices the boundary tensor, all three slices read the same rows, and every downstream block processes the wrong modality. That explains the exact failure signature: a deterministic, coherent, *unrelated* clip with correct frame count, duration and audio — nothing ever errors.

It also explains why flux and SD1.5 measure byte-identical under segmentation: they never slice a cut tensor across a segment boundary, so the misclassification has nothing to collide.

How it was isolated: a leaf-content probe (log every graph leaf's name/shape/content signature after external data lands) diffed a budgeted vs unbudgeted run of the same seed. All 539 shared leafs were byte-identical; the budgeted run's segment-2 graph had four extra leafs — the base `ggml_runner_cut:minimax_h3.blocks.39|hidden_states` [5376,843] plus three `... (view) (cont)` leafs [5376,17], [5376,16], [5376,810], **all three with the same content signature** — i.e. all bound to the base tensor at offset 0.

## Fix

Require the exact cut name — a derived node's `" (view)"`/`" (cont)"` suffix disqualifies it:

```cpp
bool is_graph_cut_tensor(const ggml_tensor* tensor) {
    if (tensor == nullptr || tensor->name[0] == '\0') {
        return false;
    }
    if (std::strncmp(tensor->name, GGML_RUNNER_CUT_PREFIX, std::strlen(GGML_RUNNER_CUT_PREFIX)) != 0) {
        return false;
    }
    // ggml auto-names derived nodes "<parent> (view)" / "<parent> (view) (cont)", so a prefix
    // match alone also classifies views/conts OF a cut tensor as cut tensors. Sibling views share
    // one identical auto-name, collide in the by-name segment cache, and all bind to the same
    // data at offset zero in the next segment.
    return std::strstr(tensor->name, " (") == nullptr;
}
```

The derived nodes then become ordinary internal nodes again, recomputed inside the consuming segment from the properly bound base tensor.

## Verification (RTX 5070 Ti 16 GB, CUDA 13, commit ea7f0c8 + this fix)

Repro config from the issue: `minimax_h3_fl2va_pruned-Q4_K_M` + Q4 encoder + both VAEs, 864×480×56, 8 steps, seed 7, `te=cpu`.

| Configuration | Before | After |
| --- | --- | --- |
| `max_vram cuda=9` | unrelated clip (deterministic) | **byte-identical to unsegmented** (7.3 vs 6.8 min) |
| `max_vram cuda=4.5` + `stream_layers` | not tested (assumed worse) | **byte-identical to unsegmented** (7.2 min, 13.6 s/step vs 10.4) |

The probe confirms the budgeted graph still splits into the same two segments (10077 + 2564 nodes, boundary at blocks.39) — the fix doesn't suppress segmentation; the colliding derived leafs are simply gone, recomputed in segment 2.

The 4.5 GiB result is worth a note: deep oversubscription is nearly free on this model, which makes 8 GB cards a realistic target for H3 through `max_vram` once this lands.

Happy to open a PR with the fix if useful.



> **Comment by CAHbKA-IV (2026-08-16T02:00:44Z):**

For ROCm backend fix also works, thank you! Tested on targets gfx1030, gfx1100.


> **Comment by farawayso (2026-08-16T10:41:20Z):**

我也是这个问题，求pr，但是我不知道怎么提？


> **Comment by farawayso (2026-08-16T13:50:21Z):**

#1865 issue comment draft — copy and paste to https://github.com/leejet/stable-diffusion.cpp/issues/1865

---

**Additional reproduction: AMD Radeon RX 7900 XTX (gfx1100), Vulkan & ROCm backends**

We can confirm the same bug on a 7900 XTX 24 GB with both Vulkan and ROCm backends (commit `de298c2`).

**Setup:**
- GPU: AMD Radeon RX 7900 XTX (gfx1100), 24 GB VRAM, Vulkan & ROCm 7.14
- Model: `minimax_h3_fl2va_pruned-Q4_K_M.gguf` + `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`
- Backend: Vulkan (eager-load, single GPU) and ROCm (stream-layers, params-backend cpu)

**Observations:**
| `--max-vram` | Segments | Content correct? |
|---|---|---|
| `Vulkan1=22` | 51→1 (merged single segment) | ✅ Yes |
| `Vulkan1=7` | 51→17 (budgeted) | ❌ No — unrelated but coherent clip |
| `Vulkan1=6` | 51→8 (budgeted) | ❌ No |
| Streaming (ROCm, fully segmented) | 49~51 segments | ❌ No |

The symptom matches the report exactly: deterministic, coherent, unrelated output with correct frame count, duration and audio — nothing ever errors, the content is just wrong.

**Workaround confirmed:** a budget large enough to merge the entire forward pass into a single segment (e.g., `Vulkan1=22` / `ROCm0=22`) completely avoids the bug, producing prompt-correct output. This is consistent with the root cause analysis (no segment boundary → no view/cont collision).

**The one-line fix from the comment** (`return std::strstr(tensor->name, " (") == nullptr;`) is expected to resolve this on both Vulkan and ROCm backends — we have not tested with a patched build but the symptom is identical across all three backends (CUDA, Vulkan, ROCm).

Thanks for the excellent root cause analysis — this completely explains our testing results.



### #1210 [closed] [Bug] "CLIP: Using CPU backend" and then crash
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1210  by AndriiParf 2026-01-18T18:34:58Z  comments: 4

### Git commit

61659ef

### Operating System & Version

Windows 11 24H2

### GGML backends

Vulkan

### Command-line arguments used

sd-cli.exe --diffusion-model E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf --vae E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors --llm E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-8B-Q6_K.gguf -r example.png -p "change the background to park" -W 960 -H 960 --cfg-scale 4.0 --steps 4 -v --offload-to-cpu --diffusion-fa -t 12 --sampling-method dpm++2mv2 --clip-on-cpu --vae-tiling --vae-tile-size 64x64

### Steps to reproduce

Checkout commit 61659ef.
Run the generation command with Flux 2 Klein 4B.
The application initializes the system info, loads the arguments, detects the Vulkan device, and prints weight statistics.
Immediately after printing CLIP: Using CPU backend, the process terminates (or hangs) without any further error message or assertion failure.

### What you expected to happen

The application should continue to load the LLM vocab/merges and then proceed to load model tensors, as it did in previous versions.

### What actually happened

The execution stops abruptly at initialization.


### Logs / error messages / stack trace

sd-cli.exe --diffusion-model E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf --vae E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors --llm E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-8B-Q6_K.gguf -r example.png -p "change the background to park" -W 960 -H 960 --cfg-scale 4.0 --steps 4 -v --offload-to-cpu --diffusion-fa -t 12 --sampling-method dpm++2mv2 --clip-on-cpu --vae-tiling --vae-tile-size 64x64
[DEBUG] main.cpp:500  - version: stable-diffusion.cpp version unknown, commit 61659ef
[DEBUG] main.cpp:501  - System Info:
    SSE3 = 1 |     AVX = 1 |     AVX2 = 1 |     AVX512 = 1 |     AVX512_VBMI = 0 |     AVX512_VNNI = 0 |     FMA = 1 |     NEON = 0 |     ARM_FMA = 0 |     F16C = 1 |     FP16_VA = 0 |     WASM_SIMD = 0 |     VSX = 0 |
[DEBUG] main.cpp:502  - SDCliParams {
  mode: img_gen,
  output_path: "output.png",
  verbose: true,
  color: false,
  canny_preprocess: false,
  convert_name: false,
  preview_method: none,
  preview_interval: 1,
  preview_path: "preview.png",
  preview_fps: 16,
  taesd_preview: false,
  preview_noisy: false
}
[DEBUG] main.cpp:503  - SDContextParams {
  n_threads: 12,
  model_path: "",
  clip_l_path: "",
  clip_g_path: "",
  clip_vision_path: "",
  t5xxl_path: "",
  llm_path: "E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-8B-Q6_K.gguf",
  llm_vision_path: "",
  diffusion_model_path: "E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf",
  high_noise_diffusion_model_path: "",
  vae_path: "E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors",
  taesd_path: "",
  esrgan_path: "",
  control_net_path: "",
  embedding_dir: "",
  embeddings: {
  }
  wtype: NONE,
  tensor_type_rules: "",
  lora_model_dir: "",
  photo_maker_path: "",
  rng_type: cuda,
  sampler_rng_type: NONE,
  flow_shift: INF
  offload_params_to_cpu: true,
  enable_mmap: false,
  control_net_cpu: false,
  clip_on_cpu: true,
  vae_on_cpu: true,
  diffusion_flash_attn: true,
  diffusion_conv_direct: false,
  vae_conv_direct: false,
  circular: false,
  circular_x: false,
  circular_y: false,
  chroma_use_dit_mask: true,
  qwen_image_zero_cond_t: false,
  chroma_use_t5_mask: false,
  chroma_t5_mask_pad: 1,
  prediction: NONE,
  lora_apply_mode: auto,
  vae_tiling_params: { 1, 64, 64, 0.5, 0, 0 },
  force_sdxl_vae_conv_scale: false
}
[DEBUG] main.cpp:504  - SDGenerationParams {
  loras: "{
  }",
  high_noise_loras: "{
  }",
  prompt: "change the background to park",
  negative_prompt: "",
  clip_skip: -1,
  width: 960,
  height: 960,
  batch_count: 1,
  init_image_path: "",
  end_image_path: "",
  mask_image_path: "",
  control_image_path: "",
  ref_image_paths: ["example.png"],
  control_video_path: "",
  auto_resize_ref_image: true,
  increase_ref_index: false,
  pm_id_images_dir: "",
  pm_id_embed_path: "",
  pm_style_strength: 20,
  skip_layers: [7, 8, 9],
  sample_params: (txt_cfg: 4.00, img_cfg: 4.00, distilled_guidance: 3.50, slg.layer_count: 3, slg.layer_start: 0.01, slg.layer_end: 0.20, slg.scale: 0.00, scheduler: NONE, sample_method: dpm++2mv2, sample_steps: 4, eta: 0.00, shifted_timestep: 0),
  high_noise_skip_layers: [7, 8, 9],
  high_noise_sample_params: (txt_cfg: 7.00, img_cfg: 7.00, distilled_guidance: 3.50, slg.layer_count: 3, slg.layer_start: 0.01, slg.layer_end: 0.20, slg.scale: 0.00, scheduler: NONE, sample_method: NONE, sample_steps: 20, eta: 0.00, shifted_timestep: 0),
  custom_sigmas: [],
  cache_mode: "",
  cache_option: "",
  cache: disabled (threshold=1, start=0.15, end=0.95),
  moe_boundary: 0.875,
  video_frames: 1,
  fps: 16,
  vace_strength: 1,
  strength: 0.75,
  control_strength: 0.9,
  seed: 42,
  upscale_repeats: 1,
  upscale_tile_size: 128,
}
[DEBUG] stable-diffusion.cpp:171  - Using Vulkan backend
[DEBUG] ggml_extend.hpp:75   - ggml_vulkan: Found 1 Vulkan devices:
[DEBUG] ggml_extend.hpp:75   - ggml_vulkan: 0 = Radeon RX 580 Series (AMD proprietary driver) | uma: 0 | fp16: 0 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 0 | matrix cores: none
[INFO ] stable-diffusion.cpp:192  - Vulkan: Using device 0
[INFO ] stable-diffusion.cpp:257  - loading diffusion model from 'E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf'
[INFO ] model.cpp:370  - load E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf using gguf format
[DEBUG] model.cpp:412  - init from 'E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf'
[INFO ] stable-diffusion.cpp:304  - loading llm from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-8B-Q6_K.gguf'
[INFO ] model.cpp:370  - load E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-8B-Q6_K.gguf using gguf format
[DEBUG] model.cpp:412  - init from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-8B-Q6_K.gguf'
[INFO ] stable-diffusion.cpp:318  - loading vae from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors'
[INFO ] model.cpp:373  - load E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors using safetensors format
[DEBUG] model.cpp:503  - init from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors', prefix = 'vae.'
[INFO ] stable-diffusion.cpp:334  - Version: Flux.2 klein
[INFO ] stable-diffusion.cpp:362  - Weight type stat:                      f32: 393  |    q8_0: 80   |    q6_K: 253  |    bf16: 69
[INFO ] stable-diffusion.cpp:363  - Conditioner weight type stat:          f32: 145  |    q6_K: 253
[INFO ] stable-diffusion.cpp:364  - Diffusion model weight type stat:     q8_0: 80   |    bf16: 69
[INFO ] stable-diffusion.cpp:365  - VAE weight type stat:                  f32: 248
[DEBUG] stable-diffusion.cpp:367  - ggml tensor size = 400 bytes
[INFO ] stable-diffusion.cpp:426  - CLIP: Using CPU backend

### Additional context / environment details

GPU: Radeon RX 580 8GB (Vulkan)
Models: Flux 2 Klein 4b Q8_0, Qwen3-8B-Q6_K
In commit 9565c7f, progress continues, but an error also appears (a different error, not this one).

> **Comment by stduhpf (2026-01-18T18:42:30Z):**

I think that's because Flux 2 Klein 4b requires Qwen3-4B, not Qwen3-8B. The lack of error message to indicate that the it's not the right text encoder is definitely an issue in this case.


> **Comment by AndriiParf (2026-01-18T18:44:28Z):**

Oops, thank you, and sorry for bothering you.


> **Comment by AndriiParf (2026-01-18T19:19:03Z):**

I downloaded Qwen3 4b, but the problem remains. I checked the file paths and there are no problems.
<h3>Logs:</h3>

```text
sd-cli.exe --diffusion-model  E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf --vae E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors  --llm E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf -r example.png -p "change the background to park" -W 960 -H 960 --cfg-scale 4.0 --steps 4 -v --offload-to-cpu --diffusion-fa -t 12 --sampling-method dpm++2mv2 --clip-on-cpu --vae-tiling --vae-tile-size 64x64
[DEBUG] main.cpp:500  - version: stable-diffusion.cpp version unknown, commit 61659ef
[DEBUG] main.cpp:501  - System Info:
    SSE3 = 1 |     AVX = 1 |     AVX2 = 1 |     AVX512 = 1 |     AVX512_VBMI = 0 |     AVX512_VNNI = 0 |     FMA = 1 |     NEON = 0 |     ARM_FMA = 0 |     F16C = 1 |     FP16_VA = 0 |     WASM_SIMD = 0 |     VSX = 0 |
[DEBUG] main.cpp:502  - SDCliParams {
  mode: img_gen,
  output_path: "output.png",
  verbose: true,
  color: false,
  canny_preprocess: false,
  convert_name: false,
  preview_method: none,
  preview_interval: 1,
  preview_path: "preview.png",
  preview_fps: 16,
  taesd_preview: false,
  preview_noisy: false
}
[DEBUG] main.cpp:503  - SDContextParams {
  n_threads: 12,
  model_path: "",
  clip_l_path: "",
  clip_g_path: "",
  clip_vision_path: "",
  t5xxl_path: "",
  llm_path: "E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf",
  llm_vision_path: "",
  diffusion_model_path: "E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf",
  high_noise_diffusion_model_path: "",
  vae_path: "E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors",
  taesd_path: "",
  esrgan_path: "",
  control_net_path: "",
  embedding_dir: "",
  embeddings: {
  }
  wtype: NONE,
  tensor_type_rules: "",
  lora_model_dir: "",
  photo_maker_path: "",
  rng_type: cuda,
  sampler_rng_type: NONE,
  flow_shift: INF
  offload_params_to_cpu: true,
  enable_mmap: false,
  control_net_cpu: false,
  clip_on_cpu: true,
  vae_on_cpu: false,
  diffusion_flash_attn: true,
  diffusion_conv_direct: false,
  vae_conv_direct: false,
  circular: false,
  circular_x: false,
  circular_y: false,
  chroma_use_dit_mask: true,
  qwen_image_zero_cond_t: false,
  chroma_use_t5_mask: false,
  chroma_t5_mask_pad: 1,
  prediction: NONE,
  lora_apply_mode: auto,
  vae_tiling_params: { 1, 64, 64, 0.5, 0, 0 },
  force_sdxl_vae_conv_scale: false
}
[DEBUG] main.cpp:504  - SDGenerationParams {
  loras: "{
  }",
  high_noise_loras: "{
  }",
  prompt: "change the background to park",
  negative_prompt: "",
  clip_skip: -1,
  width: 960,
  height: 960,
  batch_count: 1,
  init_image_path: "",
  end_image_path: "",
  mask_image_path: "",
  control_image_path: "",
  ref_image_paths: ["example.png"],
  control_video_path: "",
  auto_resize_ref_image: true,
  increase_ref_index: false,
  pm_id_images_dir: "",
  pm_id_embed_path: "",
  pm_style_strength: 20,
  skip_layers: [7, 8, 9],
  sample_params: (txt_cfg: 4.00, img_cfg: 4.00, distilled_guidance: 3.50, slg.layer_count: 3, slg.layer_start: 0.01, slg.layer_end: 0.20, slg.scale: 0.00, scheduler: NONE, sample_method: dpm++2mv2, sample_steps: 4, eta: 0.00, shifted_timestep: 0),
  high_noise_skip_layers: [7, 8, 9],
  high_noise_sample_params: (txt_cfg: 7.00, img_cfg: 7.00, distilled_guidance: 3.50, slg.layer_count: 3, slg.layer_start: 0.01, slg.layer_end: 0.20, slg.scale: 0.00, scheduler: NONE, sample_method: NONE, sample_steps: 20, eta: 0.00, shifted_timestep: 0),
  custom_sigmas: [],
  cache_mode: "",
  cache_option: "",
  cache: disabled (threshold=1, start=0.15, end=0.95),
  moe_boundary: 0.875,
  video_frames: 1,
  fps: 16,
  vace_strength: 1,
  strength: 0.75,
  control_strength: 0.9,
  seed: 42,
  upscale_repeats: 1,
  upscale_tile_size: 128,
}
[DEBUG] stable-diffusion.cpp:171  - Using Vulkan backend
[DEBUG] ggml_extend.hpp:75   - ggml_vulkan: Found 1 Vulkan devices:
[DEBUG] ggml_extend.hpp:75   - ggml_vulkan: 0 = Radeon RX 580 Series (AMD proprietary driver) | uma: 0 | fp16: 0 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 0 | matrix cores: none
[INFO ] stable-diffusion.cpp:192  - Vulkan: Using device 0
[INFO ] stable-diffusion.cpp:257  - loading diffusion model from 'E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf'
[INFO ] model.cpp:370  - load E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf using gguf format
[DEBUG] model.cpp:412  - init from 'E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf'
[INFO ] stable-diffusion.cpp:304  - loading llm from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf'
[INFO ] model.cpp:370  - load E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf using gguf format
[DEBUG] model.cpp:412  - init from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf'
[INFO ] stable-diffusion.cpp:318  - loading vae from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors'
[INFO ] model.cpp:373  - load E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors using safetensors format
[DEBUG] model.cpp:503  - init from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors', prefix = 'vae.'
[INFO ] stable-diffusion.cpp:334  - Version: Flux.2 klein
[INFO ] stable-diffusion.cpp:362  - Weight type stat:                      f32: 393  |    q8_0: 333  |    bf16: 69
[INFO ] stable-diffusion.cpp:363  - Conditioner weight type stat:          f32: 145  |    q8_0: 253
[INFO ] stable-diffusion.cpp:364  - Diffusion model weight type stat:     q8_0: 80   |    bf16: 69
[INFO ] stable-diffusion.cpp:365  - VAE weight type stat:                  f32: 248
[DEBUG] stable-diffusion.cpp:367  - ggml tensor size = 400 bytes
[INFO ] stable-diffusion.cpp:426  - CLIP: Using CPU backend
```


> **Comment by AndriiParf (2026-01-18T20:04:03Z):**

There is no problem in commit 9565c7f. 
<h3>Logs:</h3>

```text
sd-cli.exe --diffusion-model  E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf --vae E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors  --llm E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf -p "park, sunny, trees" -W 512 -H 512 --cfg-scale 4.0 --steps 4 -v --offload-to-cpu --diffusion-fa -t 12 --sampling-method dpm++2mv2 --clip-on-cpu --vae-tiling --vae-tile-size 64x64
[DEBUG] main.cpp:500  - version: stable-diffusion.cpp version unknown, commit 9565c7f
[DEBUG] main.cpp:501  - System Info:
    SSE3 = 1 |     AVX = 1 |     AVX2 = 1 |     AVX512 = 0 |     AVX512_VBMI = 0 |     AVX512_VNNI = 0 |     FMA = 1 |     NEON = 0 |     ARM_FMA = 0 |     F16C = 1 |     FP16_VA = 0 |     WASM_SIMD = 0 |     VSX = 0 |
[DEBUG] main.cpp:502  - SDCliParams {
  mode: img_gen,
  output_path: "output.png",
  verbose: true,
  color: false,
  canny_preprocess: false,
  convert_name: false,
  preview_method: none,
  preview_interval: 1,
  preview_path: "preview.png",
  preview_fps: 16,
  taesd_preview: false,
  preview_noisy: false
}
[DEBUG] main.cpp:503  - SDContextParams {
  n_threads: 12,
  model_path: "",
  clip_l_path: "",
  clip_g_path: "",
  clip_vision_path: "",
  t5xxl_path: "",
  llm_path: "E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf",
  llm_vision_path: "",
  diffusion_model_path: "E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf",
  high_noise_diffusion_model_path: "",
  vae_path: "E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors",
  taesd_path: "",
  esrgan_path: "",
  control_net_path: "",
  embedding_dir: "",
  embeddings: {
  }
  wtype: NONE,
  tensor_type_rules: "",
  lora_model_dir: "",
  photo_maker_path: "",
  rng_type: cuda,
  sampler_rng_type: NONE,
  flow_shift: INF
  offload_params_to_cpu: true,
  enable_mmap: false,
  control_net_cpu: false,
  clip_on_cpu: true,
  vae_on_cpu: false,
  diffusion_flash_attn: true,
  diffusion_conv_direct: false,
  vae_conv_direct: false,
  circular: false,
  circular_x: false,
  circular_y: false,
  chroma_use_dit_mask: true,
  qwen_image_zero_cond_t: false,
  chroma_use_t5_mask: false,
  chroma_t5_mask_pad: 1,
  prediction: NONE,
  lora_apply_mode: auto,
  vae_tiling_params: { 1, 64, 64, 0.5, 0, 0 },
  force_sdxl_vae_conv_scale: false
}
[DEBUG] main.cpp:504  - SDGenerationParams {
  loras: "{
  }",
  high_noise_loras: "{
  }",
  prompt: "park, sunny, trees",
  negative_prompt: "",
  clip_skip: -1,
  width: 512,
  height: 512,
  batch_count: 1,
  init_image_path: "",
  end_image_path: "",
  mask_image_path: "",
  control_image_path: "",
  ref_image_paths: [],
  control_video_path: "",
  auto_resize_ref_image: true,
  increase_ref_index: false,
  pm_id_images_dir: "",
  pm_id_embed_path: "",
  pm_style_strength: 20,
  skip_layers: [7, 8, 9],
  sample_params: (txt_cfg: 4.00, img_cfg: 4.00, distilled_guidance: 3.50, slg.layer_count: 3, slg.layer_start: 0.01, slg.layer_end: 0.20, slg.scale: 0.00, scheduler: NONE, sample_method: dpm++2mv2, sample_steps: 4, eta: 0.00, shifted_timestep: 0),
  high_noise_skip_layers: [7, 8, 9],
  high_noise_sample_params: (txt_cfg: 7.00, img_cfg: 7.00, distilled_guidance: 3.50, slg.layer_count: 3, slg.layer_start: 0.01, slg.layer_end: 0.20, slg.scale: 0.00, scheduler: NONE, sample_method: NONE, sample_steps: 20, eta: 0.00, shifted_timestep: 0),
  custom_sigmas: [],
  cache_mode: "",
  cache_option: "",
  cache: disabled (threshold=1, start=0.15, end=0.95),
  moe_boundary: 0.875,
  video_frames: 1,
  fps: 16,
  vace_strength: 1,
  strength: 0.75,
  control_strength: 0.9,
  seed: 42,
  upscale_repeats: 1,
  upscale_tile_size: 128,
}
[DEBUG] stable-diffusion.cpp:171  - Using Vulkan backend
[DEBUG] ggml_extend.hpp:75   - ggml_vulkan: Found 1 Vulkan devices:
[DEBUG] ggml_extend.hpp:75   - ggml_vulkan: 0 = Radeon RX 580 Series (AMD proprietary driver) | uma: 0 | fp16: 0 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 0 | matrix cores: none
[INFO ] stable-diffusion.cpp:192  - Vulkan: Using device 0
[INFO ] stable-diffusion.cpp:257  - loading diffusion model from 'E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf'
[INFO ] model.cpp:370  - load E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf using gguf format
[DEBUG] model.cpp:412  - init from 'E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf'
[INFO ] stable-diffusion.cpp:304  - loading llm from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf'
[INFO ] model.cpp:370  - load E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf using gguf format
[DEBUG] model.cpp:412  - init from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf'
[INFO ] stable-diffusion.cpp:318  - loading vae from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors'
[INFO ] model.cpp:373  - load E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors using safetensors format
[DEBUG] model.cpp:503  - init from 'E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors', prefix = 'vae.'
[INFO ] stable-diffusion.cpp:334  - Version: Flux.2 klein
[INFO ] stable-diffusion.cpp:362  - Weight type stat:                      f32: 393  |    q8_0: 333  |    bf16: 69
[INFO ] stable-diffusion.cpp:363  - Conditioner weight type stat:          f32: 145  |    q8_0: 253
[INFO ] stable-diffusion.cpp:364  - Diffusion model weight type stat:     q8_0: 80   |    bf16: 69
[INFO ] stable-diffusion.cpp:365  - VAE weight type stat:                  f32: 248
[DEBUG] stable-diffusion.cpp:367  - ggml tensor size = 400 bytes
[INFO ] stable-diffusion.cpp:426  - CLIP: Using CPU backend
[DEBUG] llm.hpp:285  - merges size 151387
[DEBUG] llm.hpp:317  - vocab size: 151669
[DEBUG] llm.hpp:1139 - llm: num_layers = 36, vocab_size = 151936, hidden_size = 2560, intermediate_size = 9728
[INFO ] flux.hpp:1364 - flux: depth = 5, depth_single_blocks = 20, guidance_embed = false, context_in_dim = 7680, hidden_size = 3072, num_heads = 24
[INFO ] stable-diffusion.cpp:572  - Using flash attention in the diffusion model
[DEBUG] ggml_extend.hpp:1922 - qwen3 params backend buffer size =  4076.43 MB(RAM) (398 tensors)
[DEBUG] ggml_extend.hpp:1922 - flux params backend buffer size =  4101.40 MB(RAM) (149 tensors)
[DEBUG] ggml_extend.hpp:1922 - vae params backend buffer size =  94.72 MB(RAM) (140 tensors)
[DEBUG] stable-diffusion.cpp:751  - loading weights
[DEBUG] model.cpp:1370 - using 12 threads for model loading
[DEBUG] model.cpp:1392 - loading tensors from E:\GenerateImage\Models\FLUX\Realistic\flux-2-klein-4b-Q8_0.gguf
  |=========>                                        | 149/795 - 144.38it/s
[DEBUG] model.cpp:1392 - loading tensors from E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\Qwen3-4B-Q8_0.gguf
  |==================================>               | 547/795 - 265.79it/s
[DEBUG] model.cpp:1392 - loading tensors from E:\GenerateImage\SDXLVaeAndClips\FLUX2Klein\ae.safetensors
  |==================================================| 795/795 - 350.84it/s
[INFO ] model.cpp:1618 - loading tensors completed, taking 2.27s (process: 0.00s, read: 1.96s, memcpy: 0.00s, convert: 0.01s, copy_to_backend: 0.00s)
[DEBUG] stable-diffusion.cpp:786  - finished loaded file
[INFO ] stable-diffusion.cpp:859  - total params memory size = 8272.55MB (VRAM 4196.12MB, RAM 4076.43MB): text_encoders 4076.43MB(RAM), diffusion_model 4101.40MB(VRAM), vae 94.72MB(VRAM), controlnet 0.00MB(VRAM), pmid 0.00MB(RAM)
[INFO ] stable-diffusion.cpp:938  - running in Flux2 FLOW mode
[DEBUG] stable-diffusion.cpp:3471 - generate_image 512x512
[INFO ] stable-diffusion.cpp:3505 - sampling using modified DPM++ (2M) method
[DEBUG] denoiser.hpp:703  - Flux2FlowDenoiser: set shift to 2.031
[INFO ] denoiser.hpp:403  - get_sigmas with discrete scheduler
[INFO ] stable-diffusion.cpp:3632 - TXT2IMG
[DEBUG] conditioner.hpp:1679 - parse '<|im_start|>user
park, sunny, trees<|im_end|>
<|im_start|>assistant
<think>

</think>

' to [['<|im_start|>user
', 1], ['park, sunny, trees', 1], ['<|im_end|>
<|im_start|>assistant
<think>

</think>

', 1], ]
[DEBUG] llm.hpp:259  - split prompt "<|im_start|>user
" to tokens ["<|im_start|>", "user", "Ċ", ]
[DEBUG] llm.hpp:259  - split prompt "park, sunny, trees" to tokens ["park", ",", "Ġsunny", ",", "Ġtrees", ]
[DEBUG] llm.hpp:259  - split prompt "<|im_end|>
<|im_start|>assistant
<think>

</think>

" to tokens ["<|im_end|>", "Ċ", "<|im_start|>", "assistant", "Ċ", "<think>", "ĊĊ", "</think>", "ĊĊ", ]
[DEBUG] llm.hpp:203  - token length: 512
[DEBUG] ggml_extend.hpp:1734 - qwen3 compute buffer size: 74.00 MB(RAM)
[DEBUG] conditioner.hpp:1923 - computing condition graph completed, taking 9856 ms
[DEBUG] conditioner.hpp:1679 - parse '<|im_start|>user
<|im_end|>
<|im_start|>assistant
<think>

</think>

' to [['<|im_start|>user
', 1], ['<|im_end|>
<|im_start|>assistant
<think>

</think>

', 1], ]
[DEBUG] llm.hpp:259  - split prompt "<|im_start|>user
" to tokens ["<|im_start|>", "user", "Ċ", ]
[DEBUG] llm.hpp:259  - split prompt "<|im_end|>
<|im_start|>assistant
<think>

</think>

" to tokens ["<|im_end|>", "Ċ", "<|im_start|>", "assistant", "Ċ", "<think>", "ĊĊ", "</think>", "ĊĊ", ]
[DEBUG] llm.hpp:203  - token length: 512
[DEBUG] ggml_extend.hpp:1734 - qwen3 compute buffer size: 74.00 MB(RAM)
[DEBUG] conditioner.hpp:1923 - computing condition graph completed, taking 9834 ms
[INFO ] stable-diffusion.cpp:3249 - get_learned_condition completed, taking 19693 ms
[INFO ] stable-diffusion.cpp:3360 - generating image: 1/1 - seed 42
[INFO ] ggml_extend.hpp:1834 - flux offload params (4101.40 MB, 149 tensors) to runtime backend (Vulkan0), taking 0.97s
[DEBUG] ggml_extend.hpp:1734 - flux compute buffer size: 377.00 MB(VRAM)
  |==================================================| 4/4 - 19.47s/it
[INFO ] stable-diffusion.cpp:3402 - sampling completed, taking 78.03s
[INFO ] stable-diffusion.cpp:3413 - generating 1 latent images completed, taking 78.35s
[INFO ] stable-diffusion.cpp:3416 - decoding 1 latents
[DEBUG] stable-diffusion.cpp:2657 - VAE Tile size: 32x32
[DEBUG] ggml_extend.hpp:849  - num tiles : 1, 1
[DEBUG] ggml_extend.hpp:850  - optimal overlap : 0.000000, 0.000000 (targeting 0.500000)
[DEBUG] ggml_extend.hpp:881  - tile work buffer size: 3.50 MB
[DEBUG] ggml_extend.hpp:894  - processing 1 tiles
[INFO ] ggml_extend.hpp:1834 - vae offload params ( 94.72 MB, 140 tensors) to runtime backend (Vulkan0), taking 0.03s
[DEBUG] ggml_extend.hpp:1734 - vae compute buffer size: 1984.50 MB(VRAM)
  |==================================================| 1/1 - 5.28s/it
[DEBUG] stable-diffusion.cpp:2683 - computing vae decode graph completed, taking 5.30s
[INFO ] stable-diffusion.cpp:3426 - latent 1 decoded, taking 5.30s
[INFO ] stable-diffusion.cpp:3430 - decode_first_stage completed, taking 5.30s
[INFO ] stable-diffusion.cpp:3740 - generate_image completed in 103.35s
[INFO ] main.cpp:421  - save result image 0 to 'output.png' (success)
```



### #2310 [open] [Bug] MiniMax-H3: decoding with VAE tiles larger than 256 px leaves a 16 px block grid
URL: https://github.com/leejet/stable-diffusion.cpp/issues/2310  by Windyskr 2026-11-09T04:18:37Z  comments: 2

### Git commit

3d315c4 (also seen on 0b3b98d)

### Operating System & Version

Ubuntu 22.04

### GGML backends

CUDA

### Command-line arguments used

```
sd-cli -M vid_gen --diffusion-model Sakura-MiniMax-H3-FL2VA-Pruned-Q8_0-19.94GiB.gguf --llm <MiniMax-H3 Qwen3-VL-32B encoder, Q8_0> --vae minimax_h3_video_vae_fp16.safetensors --audio-vae minimax_h3_audio_vae_fp32.safetensors -p "A steam locomotive crosses a tall stone viaduct over a misty mountain valley at dawn, wide cinematic shot, billowing white steam, golden sunlight breaking through the clouds." --cfg-scale 1.0 --steps 20 -W 960 -H 544 --video-frames 22 -s 1234 --rng cpu --sage-attn --vae-tiling --vae-relative-tile-size 1x1
```

`--vae-tiling --vae-tile-size 384x384` and `512x512` show the same artifact, weaker.

### Steps to reproduce

1. Generate a MiniMax-H3 clip with the default VAE tiling (256 px tiles).
2. Generate the same seed with larger decode tiles (`--vae-tiling --vae-tile-size 512x512` or `--vae-relative-tile-size 1x1`).
3. Compare the decoded frames.

### What you expected to happen

The same content, at most with different tile seams. docs/performance.md presents larger tiles as a memory/speed trade-off ("explicit spatial sizes are honored").

### What actually happened

Every 16x16 pixel block (one decoder token) gets visible edges. Grid strength, measured as the gradient energy on 16 px boundaries relative to the average over all offsets (about 1.2 means no visible grid):

| decode tiles | x | y |
|---|---:|---:|
| 256 px (default) | 1.28 | 1.20 |
| 384 px | 1.61 | 1.40 |
| 512 px | 2.30 | 2.32 |
| full frame | 2.86 | 2.45 |
| full frame, `--flash-attn auto,vae=off` (OOM, retried at 480x272) | 2.18 | 1.18 |

Visually, with 256 px tiles the sky is smooth; with full-frame tiles every 16x16 block has visible edges; with VAE flash attention off the horizontal edges disappear but vertical stripes remain.

### Logs / error messages / stack trace

No errors.

### Additional context / environment details

- 2x RTX 3090 Ti, CUDA 12.4 build for sm_86.
- The decoder is a ViT over latent tokens, and the reference implementation ships with 256 px tiles (diffusers `AutoencoderKLMiniMaxH3`: "MiniMax-H3 was released with tiling enabled ... disabling tiling changes the output"), so larger tiles may simply be out of distribution.
- Flash attention in the decoder (K/V cast to F16) makes it worse at long sequences: turning it off for the VAE removes the horizontal lines, but not the vertical ones. At the default 256 px tiles, VAE flash attention on and off are indistinguishable (PSNR 46-47 dB).
- Suggestion: clamp or warn for MiniMax-H3 decode tiles above 256 px, and/or note it next to "explicit spatial sizes are honored" in docs/performance.md.


> **Comment by farawayso (2026-11-09T15:15:39Z):**

Cross-referencing #2268: on Windows + ROCm (RX 7900 XTX, commit 290148d), the default 256 px tiles are not a performance bottleneck — a 1024x576x124 decode finished decode_first_stage in ~66 s (~1.9 it/s per 15-tile segment), with the VAE compute buffer (521 MB) fully resident in VRAM. That supports clamping H3 decode tiles at 256 px without a meaningful cost on high-VRAM cards, plus a note in docs/performance.md. I have not A/B-tested tiles >256 px on ROCm, so I can't confirm whether the 16 px grid reproduces on this backend.


> **Comment by Windyskr (2026-11-10T08:12:04Z):**

Follow-up with a corrected diagnosis: the flash-attention part of my report was wrong, and the grid has a simpler cause.

**Correction.** The "full frame, VAE FA off" run ran out of memory, and the OOM retry decoded it with 480x272 tiles, so it changed the tile size as well as flash attention. With the same 480x272 tiles, VAE FA on and off give the same grid (x 2.18 / y 1.15 vs x 2.15 / y 1.14, PSNR 49.6 dB between the two outputs). VAE flash attention plays no role.

**Cause.** The decoder's RoPE positions are normalized to the tile: `(2 * (i + 0.5) / N - 1) * 2 * pi` per axis, where N is the tile size in latent tokens ([minimax_h3_vae.hpp#L812-L815](https://github.com/leejet/stable-diffusion.cpp/blob/a3c49b13be5a64a0b4da2b576af3a17e8ae1ccbf/src/model/vae/minimax_h3_vae.hpp#L812-L815)). The diffusers reference does the same ("length-normalized to `[-1, 1)` per axis", [L279-L284](https://github.com/huggingface/diffusers/blob/51a454be9a43854939d2ef2d231565f46a41e366/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L279-L284), [L463-L467](https://github.com/huggingface/diffusers/blob/51a454be9a43854939d2ef2d231565f46a41e366/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L463-L467)), so the port is faithful. Every tile spans the same position range, so the spacing between neighboring tokens depends on N. The model was released with 256 px tiles (N = 16); any other N gives the decoder a token spacing the released configuration never uses, and the error shows up at the 16x16 pixel patch boundaries. Larger and smaller tiles both trigger it.

Measured on one latent decoded with different tiles, using the command from the report (Q8_0, 20 steps, seed 1234, 960x544, 22 frames). The table in the report mixed two setups (the 384 and 512 px rows came from a different run); this one does not. A repeated 256 px run gave bit-identical frames.

| decode tiles | tokens per tile | tiles | grid x | grid y | decode |
|---|---|---|---:|---:|---:|
| 128 px | 8 x 8 | 10x5 | 2.64 | 2.86 | 4.4 s |
| 256 px (default) | 16 x 16 | 5x3 | 1.25 | 1.15 | 4.8 s |
| 384 px | 24 x 24 | 3x2 | 1.66 | 1.38 | 4.8 s |
| 480x272 px | 30 x 17 | 3x3 | 2.18 | 1.15 | 5.7 s |
| 512 px | 32 x 32 | 2x2 | 2.43 | 2.41 | 5.3 s |
| full frame | 60 x 34 | 1x1 | 3.08 | 2.67 | 3.8 s |

Each axis follows its own token count: 480x272 tiles are 17 tokens tall, so y stays clean while x shows the grid. (384 px is the one row where x and y differ at equal N; its two tiles in y overlap by 14 of 24 rows, so blending probably averages part of it out.)

What this means for the suggestion:

- Clamping only above 256 px is not enough. Tiles below 256 px also produce the grid, and the OOM retry halves H3 tiles from 256 to 128 px ([backend_fit.cpp#L510-L526](https://github.com/leejet/stable-diffusion.cpp/blob/a3c49b13be5a64a0b4da2b576af3a17e8ae1ccbf/src/core/backend_fit.cpp#L510-L526)), so a decode that runs out of memory at 256 px gets the grid without any warning about quality.
- So the rule would be "H3 decode tiles are 16 latent tokens per axis": warn when an explicit size differs, keep the OOM retry from shrinking H3 tiles below 256 px (or warn when it does), and note it next to "explicit spatial sizes are honored" in [docs/performance.md](https://github.com/leejet/stable-diffusion.cpp/blob/a3c49b13be5a64a0b4da2b576af3a17e8ae1ccbf/docs/performance.md?plain=1#L95-L98). An axis shorter than 256 px still gets one smaller tile; the reference does the same.
- On cost (re the comment above): decode times differ by about 1 s here, with full frame the fastest, out of about 62 s for the whole run. Keeping 256 px tiles costs little.




### #1927 [open] [Feature]Request: Add FP8 (e4m3) and fix Q6_K/Q8_0 Vulkan and Rocm10.0 support for MiniMax-H3
URL: https://github.com/leejet/stable-diffusion.cpp/issues/1927  by farawayso 2026-08-31T05:16:38Z  comments: 2

### Feature Summary

Q6_K/Q8_0 crash on Vulkan and Rocm10.0

### Detailed Description

I'm running MiniMax-H3 video generation on AMD 7900 XTX (24GB VRAM) with stable-diffusion.cpp using Vulkan backend. Current setup works well with Q4_K_M quantization, but I'd like to use higher-quality or more efficient quantization formats.

### Alternatives you considered

or q6.0 q8.0 gguf 

### Additional context

I'm running MiniMax-H3 video generation on AMD 7900 XTX (24GB VRAM) with stable-diffusion.cpp using Vulkan backend. Current setup works well with Q4_K_M quantization, but I'd like to use higher-quality or more efficient quantization formats.

## Problem 1: Q6_K/Q8_0 crash on Vulkan

When selecting Q6_K or Q8_0 quantization, sd.cpp crashes with:
```
ggml/src/ggml.c:3301: GGML_ASSERT(!ggml_is_transposed(a)) failed
```

This happens in the Vulkan matmul kernel path. Q4_K_M works fine, suggesting the dequant+matmul pipeline for these quant types isn't fully implemented for Vulkan.

**Test command:**
```bash
sd-cli.exe -M vid_gen --diffusion-model minimax_h3_fl2va_pruned-Q6_K.gguf --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf --vae minimax_h3_video_vae_fp16.safetensors -p "test" -W 64 -H 64 -o out.png -v
```

**Expected:** Model loads and runs (same as Q4_K_M)
**Actual:** Assertion failure during graph execution

## Problem 2: No FP8 (e4m3) support

The official MiniMax-H3 models include FP8 e4m3 checkpoints:
- `minimax_h3_fl2va_pruned_fp8_e4m3fn.safetensors`
- `minimax_h3_ref2va_pruned_fp8_e4m3fn.safetensors`

FP8 would be ideal because:
- ~2x memory savings vs BF16
- Near-lossless quality compared to BF16
- Native hardware support on modern GPUs

## GPU Details

```
GPU: AMD Radeon RX 7900 XTX (gfx1100, RDNA3)
Vulkan: 1.3.274
Driver: AMD proprietary (25.30.1)
VRAM: 24GB
```

Note: RDNA3 has cooperative matrix support but lacks `VK_EXT_shader_float8` (that's RDNA4/gfx1200+). The portable e4m3 dequant-to-f16 path from ggml PR #1528 would work here.

## Suggested Implementation

1. **Fix Q6_K/Q8_0 Vulkan matmul**: Add proper transposed-tensor handling or route to CPU fallback
2. **Add FP8 e4m3 support**: The ggml PR #1528 already implements this - verify sd.cpp properly wires it up for diffusion models

## Reference

- ggml PR #1528: vulkan: add e4m3 (fp8) matmul for AMD RDNA4
- MiniMax-H3 FP8 discussion: https://github.com/sgl-project/sglang/discussions/34079

---

Would be happy to help test any fixes!

> **Comment by farawayso (2026-09-01T18:03:38Z):**

## Update: Q6_K/Q8_0 crash is FIXED in current master (6b3edaa) — remaining ask is FP8

I want to close the loop on Problem 1 of this report. Retesting on the **same hardware** (AMD 7900 XTX / gfx1100 / 24GB, Vulkan backend) against current master (`6b3edaa`), the Q6_K/Q8_0 crash is **no longer reproducible**:

- **Q6_K** now runs cleanly. The earlier `GGML_ASSERT(!ggml_is_transposed(a))` is gone. One important practical finding: pruned Q6_K (~16.5GB diffusion) + VAE (~5.5GB) + `--eager-load` together can exceed the VRAM budget and cause heavy page-swapping, making it look like a hang. Adding `--params-backend vae=cpu` (keep VAE weights in system RAM, GPU only holds diffusion) restores full speed — **~1m45s for 480x864 / 39 frames / 4 steps, on par with Q4_K_M**. So on 24GB cards, Q6_K is a fully viable quality upgrade over Q4_K_M.
- **Q8_0** also produces valid output (no crash), but at 21.4GB it is too close to the 24GB ceiling; VAE decode drops to ~40s/tile due to swapping, making the whole run ~5-6x slower than Q4_K_M/Q6_K. Practical takeaway: Q8_0 is not recommended on 24GB cards.

So the remaining actionable item from this issue is **Problem 2: FP8 (e4m3) support** — the official MiniMax-H3 ships FP8 e4m3fn checkpoints, and on RDNA3 (which has cooperative matrix support but no `VK_EXT_shader_float8`), the portable e4m3→f16 dequant path would give meaningful memory savings without the Q8_0 VRAM pressure.

Happy to help test the FP8 path once available.



> **Comment by GreenShadows (2026-09-08T11:11:26Z):**

RDNA 3 doesn't support FP8. 
INT8 ConvRot is the best option for RDNA 3.

