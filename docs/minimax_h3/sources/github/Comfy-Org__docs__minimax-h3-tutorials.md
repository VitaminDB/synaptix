# Comfy-Org__docs__minimax-h3-tutorials

Source: https://github.com/Comfy-Org/docs/blob/main/tutorials/video/minimax/ (docs.comfy.org/tutorials/video/minimax/minimax-h3)
Fetched: 2026-11-28



---

## FILE: tutorials/video/minimax/minimax-h3.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/tutorials/video/minimax/minimax-h3.mdx)

---
title: "ComfyUI MiniMax H3 Video Generation Guide"
description: "Run open-weights MiniMax H3 in ComfyUI: text-to-video, image-to-video, and reference-to-video Workflows with native stereo audio."
sidebarTitle: "Overview"
---

[MiniMax H3](https://www.minimax.io/blog/minimax-h3) is MiniMax's open-weights omni-modal model. It understands text, images, video, and audio in one context and generates video with **native stereo audio** (voice, SFX, and music in a single pass). Open weights target about 1 megapixel (768px short edge) at 24fps for up to ~15 seconds. 2K is a hosted-service output and needs a separate upscale pass in ComfyUI.

<Note>
H3's open weights let you run the model locally. Commercial use of locally generated outputs requires a [MiniMax commercial license](https://comfy.org/minimax/license), available through Comfy, the only official reseller. Generations on Comfy Cloud already include commercial rights.
</Note>

## Key features

- **Native stereo audio**: Dialogue, SFX, and music in one MP4
- **Multimodal context**: Combine text, image, video, and audio references
- **Reference-driven generation**: Lock identity, style, motion, camera, or voice from references
- **Instruction following**: Describe reference-to-shot relationships in natural language
- **Accurate text rendering**: Clean spelled-out text and brand elements
- **Open weights**: Full local control in ComfyUI

## Choosing a template Workflow

Pick a Workflow from **Template Library** > **Video** and download models from the pop-up. Weights: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

Five `video_minimax_h3_*` template Workflows are available (examples, not an exhaustive list): `t2v`, `i2v`, `r2v`, `multiframe_reference`, and `fun_controlnet_union`.

<CardGroup cols={2}>
  <Card title="Text to Video (T2V)" icon="type" href="/tutorials/video/minimax/minimax-h3-native#minimax-h3-text-to-video-t2v">
    Generate videos from text prompts with native stereo audio. ComfyUI **0.30.0+**.
  </Card>
  <Card title="Image to Video (I2V)" icon="image" href="/tutorials/video/minimax/minimax-h3-native#minimax-h3-image-to-video-i2v">
    Generate videos from an input image, with optional first/last-frame control. ComfyUI **0.30.0+**.
  </Card>
  <Card title="Reference to Video (R2V)" icon="link" href="/tutorials/video/minimax/minimax-h3-native#minimax-h3-reference-to-video-r2v">
    Lock in a character, style, motion, camera move, or voice from reference images, videos, and audio. ComfyUI **0.30.0+**.
  </Card>
  <Card title="Multiframe Reference" icon="film" href="/tutorials/video/minimax/minimax-h3-multiframe">
    Anchor reference frames at specific points along the output timeline with chained Add Guide Nodes. ComfyUI **0.34.0+**.
  </Card>
  <Card title="Fun ControlNet Union" icon="person" href="/tutorials/video/minimax/minimax-h3-fun-controlnet">
    Drive H3 with a Canny, Depth, HED, MLSD, or Pose control video, or run video inpainting with a mask. ComfyUI **0.35.0+** (Union 2.0: **0.38.0+**). Sparse attention and `Model Attention Backend` also need **0.35.0+**.
  </Card>
  <Card title="Prompt guide" icon="book-open" href="/tutorials/video/minimax/minimax-h3-prompt-guide">
    Official prompt guides, tips, and embeddings for H3
  </Card>
  <Card title="FastH3" icon="bolt" href="/tutorials/video/minimax/minimax-h3-fastvideo">
    FastVideo sparse-attention checkpoints and speed-focused Workflows. ComfyUI **0.36.0+**.
  </Card>
</CardGroup>

Two templates cover several modes:

- `MiniMax H3 Image to Video` covers T2V and first/last-frame I2V (`t2va` / `fl2va`).
- `MiniMax H3 Reference to Video` covers R2V (`ref2va`).

## Commercial licensing

Commercial use of locally generated outputs requires a license from Comfy. Comfy Cloud generations already include commercial rights.

<CardGroup cols={2}>
  <Card title="MiniMax commercial license" icon="scale" href="https://comfy.org/minimax/license">
    When you need a license, community vs Professional and Enterprise tiers, and how to request a quote.
  </Card>
</CardGroup>


---

## FILE: tutorials/video/minimax/minimax-h3-native.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/tutorials/video/minimax/minimax-h3-native.mdx)

---
title: "ComfyUI MiniMax H3 Text-to-Video, Image-to-Video, and Reference-to-Video Workflows"
description: "Step-by-step MiniMax H3 native workflow examples in ComfyUI, including guide anchoring and latent noise masks for finer control over video generation."
sidebarTitle: "Native workflows"
---

This page covers the three base MiniMax H3 templates plus advanced native-node techniques. For capabilities, tuning, and speedups, see the [MiniMax H3 overview](/tutorials/video/minimax/minimax-h3).

## MiniMax H3 Text to Video (T2V)

Generate videos from text prompts with native stereo audio.

<video controls className="w-full aspect-video" src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/output/video_minimax_h3_t2v.mp4"></video>

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=video_minimax_h3_t2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json">
    Download JSON or search "MiniMax H3 T2V" in Template Library
  </Card>
</CardGroup>

### Model downloads

<CardGroup cols={2}>
  <Card title="Diffusion Model: minimax_h3_fl2va_pruned_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/diffusion_models/</code>
  </Card>
  <Card title="Text Encoder: qwen3vl_32b_minimax_h3_nvfp4_awq" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors">
    Place in <code>ComfyUI/models/text_encoders/</code>
  </Card>
  <Card title="VAE: minimax_h3_video_vae_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="VAE: minimax_h3_audio_vae_fp32" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="LoRA: minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16" icon="download" href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors">
    Required by the workflow's model scan; powers the optional Lightning LoRA turbo mode. Place in <code>ComfyUI/models/loras/</code>
  </Card>
</CardGroup>

### Model storage

```
ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── minimax_h3_fl2va_pruned_int8_convrot.safetensors
│   ├── 📂 text_encoders/
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
│   │   └── minimax_h3_audio_vae_fp32.safetensors
│   ├── 📂 loras/
│   │   └── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
│   └── 📂 embeddings/
│       └── minimaxh3_art_is_explosion.safetensors
```

### Prompting tips

1. **Scene first**: Overall scene, then timed shots
2. **One block**: Shots, camera, dialogue, SFX, and music together
3. **Resolution**: Native 16:9 is **1344×768** (768px short edge); round to multiples of **32**
4. **Duration**: Snaps to **17k+5** frames at 24fps
5. **First/last frame**: Wire images to `first_frame` / `last_frame` on `MiniMaxH3ImageToVideo` for FL2VA
6. **Turbo**: Default **20** steps; enable `turbo_mode` for **8**-step Lightning LoRA (faster, slightly softer motion/audio). Raise steps (e.g. **25**) for motion quality

Official base guide: [prompt guide](/tutorials/video/minimax/minimax-h3-prompt-guide).

## MiniMax H3 Image to Video (I2V)

Generate videos from an input image, with optional first/last-frame keyframes.

<video controls className="w-full aspect-video" src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/output/video_minimax_h3_i2v.mp4"></video>

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=video_minimax_h3_i2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json">
    Download JSON or search "MiniMax H3 I2V" in Template Library
  </Card>
</CardGroup>

<CardGroup cols={1}>
  <Card title="Input Image: transparent_rgb_gaming_mouse.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/transparent_rgb_gaming_mouse.png">
    Download the default input image, or use your own image.
  </Card>
</CardGroup>

### Model downloads

<CardGroup cols={2}>
  <Card title="Diffusion Model: minimax_h3_fl2va_pruned_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/diffusion_models/</code>
  </Card>
  <Card title="Text Encoder: qwen3vl_32b_minimax_h3_nvfp4_awq" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors">
    Place in <code>ComfyUI/models/text_encoders/</code>
  </Card>
  <Card title="VAE: minimax_h3_video_vae_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="VAE: minimax_h3_audio_vae_fp32" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
</CardGroup>

<CardGroup cols={1}>
  <Card title="Turbo LoRA: minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16" icon="download" href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors">
    Place in <code>ComfyUI/models/loras/</code>. Used when <code>turbo_mode</code> is enabled.
  </Card>
</CardGroup>

### Model storage

```
ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── minimax_h3_fl2va_pruned_int8_convrot.safetensors
│   ├── 📂 text_encoders/
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 loras/
│   │   └── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
│   │   └── minimax_h3_audio_vae_fp32.safetensors
│   └── 📂 embeddings/
│       └── minimaxh3_art_is_explosion.safetensors
```

### Prompting tips

1. **Keyframes**: Optional `first_frame` / `last_frame`; motion is generated between them
2. **Prompt**: Shots, motion, and audio in one block
3. **Resolution / duration**: the same rules as T2V (**1344×768** native target, **17k+5** grid)
4. **Turbo**: `turbo_mode` uses `turbo_steps` (default **8**) and `turbo_model_strength` (default **1.0**) instead of 20 steps

`first_frame` is resized to the canvas without keeping its aspect ratio, so an off-aspect image comes out stretched. `last_frame` is center-cropped instead. The I2V template opens at **1:1** and **0.4** MP (**640×640**), so set the aspect ratio to your image and raise Megapixels for a final render.

Official base guide: [prompt guide](/tutorials/video/minimax/minimax-h3-prompt-guide).

## MiniMax H3 Reference to Video (R2V)

Lock character, style, motion, camera, or voice from reference images, videos, and audio.

<video controls className="w-full aspect-video" src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/output/video_minimax_h3_r2v.mp4"></video>

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=video_minimax_h3_r2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json">
    Download JSON or search "MiniMax H3 R2V" in Template Library
  </Card>
</CardGroup>

<CardGroup cols={2}>
  <Card title="Reference Image: red_superboy_on_city_roof.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/red_superboy_on_city_roof.png">
    Character reference for the workflow, or use your own images.
  </Card>
  <Card title="Reference Image: mecha_dragon_lightning.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/mecha_dragon_lightning.png">
    Style and subject reference for the workflow, or use your own images.
  </Card>
</CardGroup>

### Model downloads

<CardGroup cols={2}>
  <Card title="Diffusion Model: minimax_h3_ref2va_pruned_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/diffusion_models/</code>
  </Card>
  <Card title="Text Encoder: qwen3vl_32b_minimax_h3_nvfp4_awq" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors">
    Place in <code>ComfyUI/models/text_encoders/</code>
  </Card>
  <Card title="VAE: minimax_h3_video_vae_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="VAE: minimax_h3_audio_vae_fp32" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="LoRA: minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors">
    Required by the workflow's model scan; powers the optional Lightning LoRA turbo mode. Place in <code>ComfyUI/models/loras/</code>
  </Card>
</CardGroup>

### Model storage

```
ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   ├── 📂 text_encoders/
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
│   │   └── minimax_h3_audio_vae_fp32.safetensors
│   ├── 📂 loras/
│   │   └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
│   └── 📂 embeddings/
│       └── minimaxh3_art_is_explosion.safetensors
```

### Prompting tips

1. **Tags**: Reference inputs as `<Picture 1>`, `<Video 1>`, `<Audio 1>` (`ref_image_0` → `<Picture 1>`). Node order: images, then videos, then audio
2. **Assign roles**: Say which reference drives identity, style, motion, camera, or voice
3. **Limits**: Up to **9** images, **3** videos, **3** standalone audio clips
4. **Video FPS**: Reference video is read as **24fps** with no metadata. Convert other frame rates to 24fps before load
5. **Video sound**: Picture and soundtrack are separate references. Wire `ref_video_audio_N` to match `ref_video_N` or the file's audio is not used as `<Audio N>`
6. **Audio numbering**: Video soundtracks get `<Audio N>` first (video order), then standalone clips. Video and picture indices are independent
7. **VAE wiring**: Video soundtrack needs `vae` + `audio_vae`; standalone audio needs `audio_vae` only
8. **Audio roles**: e.g. `use <Audio 1> exactly as it is` vs `follow the timbre of <Audio 2>`
9. **Voice**: Audio carries timbre, not words. Specify delivery; one audio ref per speaker when assignment must be exact
10. **`ref_image_size`**: Default **`match`** (fit to canvas). **`max`** keeps 2048px short edge and is much slower; it does not upscale small files (round sides to 32). Pre-scale to 2048 short edge when needed
11. **Weights**: R2V uses **`ref2va`**, not the T2V/I2V **`fl2va`** model
12. **Turbo**: Optional **4**-step Lightning LoRA (faster, weaker reference/audio). Prefer **20–25** steps when references must hold
13. **Style LoRAs**: Compete with references; use low strength (~**0.5**) on base schedule when adherence matters

Official ref guide: [prompt guide](/tutorials/video/minimax/minimax-h3-prompt-guide).

## Advanced workflows with the native nodes

Requires a recent ComfyUI build for the features below.

### Anchoring guides at any frame (#15439)

[`MiniMaxH3AddGuide`](https://github.com/Comfy-Org/ComfyUI/pull/15439) anchors a guide at any `frame_idx` (negative = from end), not only first/last frame.

Wire `positive` and latent from an H3 node into `MiniMaxH3AddGuide`. Inputs:

- **image**: Still or clip (multi-frame batches crop to valid **17k+5** lengths; &lt;5 frames uses first image only)
- **audio**: Soundtrack at the same frame index, cropped to remaining duration
- **frame_idx**: Anchor position

Connect `vae` for image guides and `audio_vae` for audio. Chain nodes for multiple anchors.

Example: first **22** frames of a clip at frame **0** continues both streams; a still at frame **60** forces the shot through that pose.

This is conditioning, not video-to-video: sampling still starts from the workflow empty latent. `denoise` does not scale guide strength like V2V. To restyle a whole clip, use R2V; to edit regions, use [Fun ControlNet Union](/tutorials/video/minimax/minimax-h3-fun-controlnet) masks.

See the [Multiframe Reference](/tutorials/video/minimax/minimax-h3-multiframe) template.

### Inpainting and extension with latent noise masks (#15375)

[Per-token noise masks](https://github.com/Comfy-Org/ComfyUI/pull/15375) on video and audio latents let you regenerate part of a clip.

Connect a mask to the sampler `denoise_mask`: **0** = keep, **1** = regenerate. Video masks snap to the 2×2 latent patch grid; audio masks to whole latent frames. Use for inpainting, removal, or extension while holding the rest stable.


---

## FILE: tutorials/video/minimax/minimax-h3-prompt-guide.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/tutorials/video/minimax/minimax-h3-prompt-guide.mdx)

---
title: "ComfyUI MiniMax H3 Prompt Guide: Official Guides, Tips, and Prompt Embeddings"
description: "Write better MiniMax H3 prompts in ComfyUI: MiniMax's official prompt guides, general prompting tips, dialogue and speaker rules, and style embeddings."
sidebarTitle: "Prompt guide"
---

H3 expects shots, camera, dialogue, and SFX in one multimodal prompt. This page collects official guides, cross-workflow tips, shot/dialogue rules, and style embeddings.

## Official guides

- [Base modes](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) (T2VA, I2VA, FL2VA, L2VA): timed shots, camera, audio
- [Full-reference mode](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) (R2V): rewrite structure, reference labels, retention markers

Installable [H3 skills](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills) package these for agents.

Applied per template: [T2V/I2V](/tutorials/video/minimax/minimax-h3-native), [R2V](/tutorials/video/minimax/minimax-h3-native#minimax-h3-reference-to-video-r2v), [Multiframe](/tutorials/video/minimax/minimax-h3-multiframe#prompt-writing-guide), [Fun ControlNet](/tutorials/video/minimax/minimax-h3-fun-controlnet#prompting-tips).

## General tips

I2VA opens with a fixed image-alignment line, blank line, then body fields. FL2VA/L2VA use their own alignment line; T2VA starts at the fields directly:

```text
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
```

Write prompts in **English**. Dialogue/lyrics in `<d>` tags; on-screen text verbatim in its original language.

1. **Scene first**: Overall scene, then timed shots inside `[Shot 1]` of `integrated_multimodal_description`
2. **One block**: Shots, camera, and audio together
3. **Resolution**: Native **1344×768** at 16:9; multiples of **32**
4. **Length**: `length` is frames at 24fps (default **124** ≈ 5s), snapped to **17k+5**
5. **On-screen text**: English double quotes per string, verbatim (e.g. `A red neon sign reading "营业中"`). Spoken lines use `<d>`, not quotes
6. **R2V audio markers**: In retention analysis, mark each audio ref as `fully_copy`, `partially_copy`, `reference`, or `weak_reference` to match how the clip is used (see official ref guide table)
7. **Audio fields**: `overall_soundscape` = ambient/SFX/non-verbal (1–4 sentences; `N/A` only if truly silent). Dialogue and diegetic music stay in shots. `non_diegetic_music` = score only (1–3 sentences; `N/A` if none). If unwanted speech appears, fill both fields explicitly
8. **No negatives**: `BasicGuider` runs at `cfg` 1 with no negative branch. Ban lists add unwanted concepts. State what should appear (or use R2V retention fields: `fully_preserved`, `partially_preserved`, etc.)
9. **Reference sheets (R2V)**: One image can hold multiple panels. Give the sheet its own `<Picture N>` (not only inside `<Subject N>`), describe panels, and point shots at a panel (e.g. full front view of `<Picture 2>`). Label panels clearly

Workflow-specific tags and controls live on each workflow page.

## Shots and cuts

Number shots `[Shot 1]`, `[Shot 2]`, … `[Shot 1]` starts at 0.00s. Later shots open with a cut time:

```text
[Shot 2] At 00:03.500, the camera cuts to...
```

- Cut times increase and stay within clip duration (`length` at 24fps)
- Use a single timestamp (`At 00:05.000`), not a range like `[5s-8s]`
- Phrase cuts as actions: `the camera cuts to`, `the shot transitions to`, etc. Dissolves/fades/wipes when needed

New shot when subject, space, state, viewpoint, or time changes. Small reframes stay one shot with camera motion.

## Dialogue and speakers

- **Speaker IDs**: `(S1)`, `(S2)`, … for every voice (including off-screen/singing). IDs index speakers, not character numbers. Duet: `(S1,S2)`
- **`<d>` content**: ID, action, and delivery outside tags; inside tags, language tag + spoken words verbatim, in the shot paragraph where spoken
- **Punctuation in `<d>`**: Keep `, . ? !`; drop decorative repeats/emoji; end statements with `.` `?` `!` before `</d>`; use `[unclear]` when unsure. Ellipses often hurt intelligibility
- **Voiceover**: Use `says in an off-screen voiceover` and state lips stay closed after `</d>`. Cross-cut dialogue: `<scenetrans>` at both sides plus audio carries over
- **Multi-speaker + audio refs**: Wrong-speaker assignment can happen. Tie lines to visible events, not timecodes. If needed, generate per-speaker audio externally and reference with the right marker (`fully_copy` vs `partially_copy`)

## Prompt embeddings

[ComfyUI#15697](https://github.com/Comfy-Org/ComfyUI/pull/15697) supports `embedding:name` in H3 prompts. Place files in `ComfyUI/models/embeddings/`.

[Comfy-Org/MiniMax-H3/embeddings](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/embeddings) lists **10** community style embeddings ([silveroxides](https://huggingface.co/silveroxides), [discussion #50](https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/50); unofficial). Trigger word = filename without extension:

| Embedding | Effect (by name) |
|---|---|
| `minimaxh3_art_is_explosion` | Explosive art composition |
| `minimaxh3_blooming_flowers` | Flowers blooming |
| `minimaxh3_bullet_time` | Bullet-time effect |
| `minimaxh3_dark_magic` | Dark magic atmosphere |
| `minimaxh3_fire_breath` | Fire-breath effect |
| `minimaxh3_four_seasons` | Four seasons transition |
| `minimaxh3_kiss_camera` | Kiss scene camera move |
| `minimaxh3_spiral_ascent` | Spiraling ascent |
| `minimaxh3_storm_magic` | Storm magic |
| `minimaxh3_truman_show` | Truman Show style |


---

## FILE: tutorials/video/minimax/minimax-h3-multiframe.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/tutorials/video/minimax/minimax-h3-multiframe.mdx)

---
title: "MiniMax H3 Multiframe: Video Continuations and Multi-Shot Scenes"
description: "Build MiniMax H3 video continuations and multi-shot scenes in ComfyUI by pinning reference frames to set points on the timeline with Add Guide nodes."
sidebarTitle: "Multiframe Reference"
---

Generate video continuations and multi-shot scenes by anchoring reference frames at specific points along the output timeline. The workflow combines the Reference to Video node with chained Add Guide nodes: each Add Guide pins a still image (plus optional audio) to a frame index, so the video passes through that exact composition at that moment.

![MiniMax H3 Multiframe Reference workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_multiframe_reference-1.webp)

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=video_minimax_h3_multiframe_reference&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_multiframe_reference.json">
    Download JSON or search "MiniMax H3: Multiframe Reference" in Template Library
  </Card>
</CardGroup>

<CardGroup cols={2}>
  <Card title="Frame Reference 1: h3_frame_ref_1.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/h3_frame_ref_1.png">
    Identity reference and first frame of Shot 1, or use your own images.
  </Card>
  <Card title="Frame Reference 2: h3_frame_ref_2.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/h3_frame_ref_2.png">
    Keyframe anchored at 1.5s (Shot 2), or use your own images.
  </Card>
  <Card title="Frame Reference 3: h3_frame_ref_3.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/h3_frame_ref_3.png">
    Keyframe anchored at 3.0s (Shot 3), or use your own images.
  </Card>
  <Card title="Frame Reference 4: h3_frame_ref_4.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/h3_frame_ref_4.png">
    First frame anchored at 5.0s (Shot 4), or use your own images.
  </Card>
</CardGroup>

## Model downloads

<CardGroup cols={2}>
  <Card title="Diffusion Model: minimax_h3_ref2va_pruned_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/diffusion_models/</code>
  </Card>
  <Card title="Text Encoder: qwen3vl_32b_minimax_h3_nvfp4_awq" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors">
    Place in <code>ComfyUI/models/text_encoders/</code>
  </Card>
  <Card title="VAE: minimax_h3_video_vae_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="VAE: minimax_h3_audio_vae_fp32" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="LoRA: minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors">
    Powers the optional Lightning LoRA turbo mode. Place in <code>ComfyUI/models/loras/</code>
  </Card>
</CardGroup>

## Model storage

```
ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   ├── 📂 text_encoders/
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
│   │   └── minimax_h3_audio_vae_fp32.safetensors
│   └── 📂 loras/
│       └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
```

## Prompting tips

1. **Reference by tag**: `<Picture 1>` is the image connected to the first `ref_images` slot, `<Picture 2>` the second, and so on. In this workflow, `<Picture 1>` is both the identity lock and the first frame of the video
2. **Timeline anchors vs references**: Pictures connected only to Add Guide nodes pin compositions at specific frames but are not visible to the text encoder. Plug them into extra `ref_images` slots too if the prompt should see them as `<Picture N>`
3. **Frame index**: Each Add Guide takes `frame_idx` in frames at 24fps; the template computes it from seconds with `round(seconds * 24)`. Negative values count from the end of the video. The index plus the guide length must stay inside the generated duration
4. **Add more keyframes**: Copy one guide block (Load Image + Frame Index + Math Expression + Add Guide for MiniMax H3) and keep the `positive` outputs chained in series into the Basic Guider
5. **Turbo mode (optional)**: Enable the **Lightning LoRA** checkbox to use the 4-step turbo LoRA for much faster generation, with slightly lower audio and motion quality

## Prompt writing guide

The template prompt follows the [full-reference mode prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md): it defines each subject, maps each `<Picture N>` to a shot's first frame or keyframe, and cuts between shots at explicit timestamps. Use the same structure when writing your own multi-shot prompts. The guide is summarized in the [prompt guide](/tutorials/video/minimax/minimax-h3-prompt-guide).


---

## FILE: tutorials/video/minimax/minimax-h3-fastvideo.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/tutorials/video/minimax/minimax-h3-fastvideo.mdx)

---
title: "FastVideo FastH3: ComfyUI Workflow Examples"
description: "FastVideo FastH3 generates MiniMax H3 video with synchronized audio in 8 sampling steps in ComfyUI, for text-to-video and first/last-frame workflows."
sidebarTitle: "FastVideo FastH3"
---

[FastVideo FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) is a DMD2-distilled MiniMax H3 checkpoint from the FastVideo team that generates video with synchronized native audio in **8 sampling steps** instead of the base model's full schedule. It reuses the H3 text encoder, video VAE, and audio VAE, so output matches the base model's setup while sampling finishes in a fraction of the steps.

The FastH3 preview work was developed in collaboration with Nuva Lab and the NVIDIA FastGen team; read the [FastH3 announcement](https://haoailab.com/blogs/fasth3-preview/) for distillation details. ComfyUI ships with the 8-step V2 checkpoint, repacked by Comfy-Org in the [FastVideo-FastH3-Comfy](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy) repository.

FastH3 trades some motion and audio fidelity for speed. Use it for drafts, iteration, and fast turnaround; use the base MiniMax H3 workflows when maximum quality matters.

<UpdateReminder/>

<Note>
The distilled checkpoint supports text-to-video and first/last-frame image-to-video only. Ref2VA (multi-reference conditioning) was not distilled; use the base [MiniMax H3 workflows](/tutorials/video/minimax/minimax-h3-native) for reference-based generation.
</Note>

## Requirements

- ComfyUI 0.36.0 or later
- The MiniMax H3 text encoder and VAEs (shared with the base model, listed below)

## FastH3 Text to Video

Generate synchronized video and audio from a text prompt in 8 steps.

<video controls className="w-full aspect-video" src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/output/video_fastvideo_fasth3_t2v.mp4"></video>

{/* TODO: Enable Cloud template when it is available in Comfy Cloud */}
{/*<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=video_fastvideo_fasth3_t2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3">
    Open in Comfy Cloud
  </Card>
</CardGroup>*/}

<CardGroup cols={2}>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_fastvideo_fasth3_t2v.json">
    Download JSON or search "FastVideo FastH3: Text to Video" in Template Library
  </Card>
</CardGroup>

### Prompting tips

1. **Describe the whole scene**: State the overall scene first (location, character, what is happening), then break it into timed shots
2. **Shots, camera, and audio**: Describe the shots, camera moves, and the accompanying audio (dialogue, SFX, music) in one prompt block
3. **Resolution**: H3's native canvas has a 768px short edge and a 768×1344 area, and resolutions are rounded to a multiple of 32. The node accepts larger sizes, but they render slower without adding detail
4. **Duration**: The duration input snaps to the model's 17-frame-per-block (17k+5) grid at 24fps
5. **Steps are fixed at 8**: The distilled checkpoint is trained for exactly 8 steps; changing the step count in the scheduler degrades quality

For prompt writing resources, see the [prompt guide](/tutorials/video/minimax/minimax-h3-prompt-guide).

## FastH3 Image to Video

Animate a still image with synchronized audio, with optional first/last-frame control.

<video controls className="w-full aspect-video" src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/output/video_fastvideo_fasth3_i2v.mp4"></video>

{/* TODO: Enable Cloud template when it is available in Comfy Cloud */}
{/*<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=video_fastvideo_fasth3_i2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3">
    Open in Comfy Cloud
  </Card>
</CardGroup>*/}

<CardGroup cols={2}>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_fastvideo_fasth3_i2v.json">
    Download JSON or search "FastVideo FastH3: Image to Video" in Template Library
  </Card>
</CardGroup>

**Input material**

Upload this file to the matching `LoadImage` node, or use your own image:

<CardGroup cols={2}>
  <Card title="red_line_barrier.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/red_line_barrier.png">
    Example first frame for the Image to Video workflow
  </Card>
</CardGroup>

### First/last-frame mode

Connect images to `first_frame` and/or `last_frame` on the `MiniMaxH3ImageToVideo` node to generate the motion between two keyframes. Leave both disconnected for text-to-video.

### Prompting tips

1. **Describe the motion and audio**: With an image providing the look, the prompt should focus on motion, camera, and the accompanying audio
2. **Duration**: The duration input snaps to the model's 17-frame-per-block (17k+5) grid at 24fps
3. **Steps are fixed at 8**: Keep the scheduler at 8 steps; the checkpoint is distilled for that schedule

## Model downloads

The diffusion model comes from the FastVideo repository; the text encoder and VAEs are shared with the base MiniMax H3 model.

<CardGroup cols={2}>
  <Card title="Diffusion Model: fastvideo_fasth3_8step_v2_pruned_int8_convrot" icon="download" href="https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/diffusion_models/</code>
  </Card>
  <Card title="Text Encoder: qwen3vl_32b_minimax_h3_nvfp4_awq" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors">
    Place in <code>ComfyUI/models/text_encoders/</code>
  </Card>
  <Card title="VAE: minimax_h3_video_vae_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="VAE: minimax_h3_audio_vae_fp32" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
</CardGroup>

### Model storage

```
ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   └── fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors
│   ├── 📂 text_encoders/
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   └── 📂 vae/
│       ├── minimax_h3_video_vae_int8_convrot.safetensors
│       └── minimax_h3_audio_vae_fp32.safetensors
```

## Workflow structure notes

Both FastH3 workflows share these nodes with the base H3 setup, with distillation-specific settings:

- **BlockSparseAttention**: Runs Video Sparse Attention (VSA) with `keep_percent` 10 starting at 20% of the schedule, reducing attention cost while preserving quality
- **MiniMaxH3SigmaShift**: Applies the H3 sigma shift (10 for video, 3 for audio) that matches the distilled schedule
- **ComfyMathExpression**: Converts the duration input into a valid frame `length` on the 17k+5 grid at 24fps
- **Sampler**: `res_multistep` sampler with the `simple` scheduler at 8 steps

The FastH3 weights include `to_gate_compress` gate layers, which VSA reads for its coarse branch. Base H3 checkpoints do not have them. To compare methods, set `method` to `sol-attn` on the **Model Sparse Attention** node and keep the same seed. `sol-attn` needs no matching weights and runs on any H3 checkpoint at the `tau` you set.


---

## FILE: tutorials/video/minimax/minimax-h3-fun-controlnet.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/tutorials/video/minimax/minimax-h3-fun-controlnet.mdx)

---
title: "MiniMax H3 Fun ControlNet Union: Pose, Depth, and Canny Control"
description: "Guide MiniMax H3 video motion in ComfyUI with Canny, Depth, HED, MLSD, or Pose control videos, plus mask-based inpainting with the Fun ControlNet Union patch."
sidebarTitle: "Fun ControlNet Union"
---

Control the motion of a MiniMax H3 video with a control video instead of text alone. The Fun ControlNet Union patch ([alibaba-pai/MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union), packaged in [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)) is a single checkpoint that conditions H3 on Canny, Depth, HED, MLSD, or Pose control videos, and also runs video inpainting with a mask. It works with both the `fl2va` and `ref2va` transformer files.

![MiniMax H3 Fun ControlNet Union workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_fun_controlnet_union-1.webp)

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=video_minimax_h3_fun_controlnet_union&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_fun_controlnet_union.json">
    Download JSON or search "MiniMax H3 Fun ControlNet Union" in Template Library
  </Card>
</CardGroup>

**Input materials**

Upload this file to the matching `Load Video` node:

<CardGroup cols={2}>
  <Card title="dancer_field_pose.mp4" icon="video" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/dancer_field_pose.mp4">
    Pose control video used by the example workflow, or use your own.
  </Card>
</CardGroup>

This template requires ComfyUI 0.35.0 or later. The example uses a pose control video: a built-in SDPose subgraph extracts the pose skeleton from the input video, so the output reproduces the dancer's motion. You can also connect any preprocessed control video directly to the `Apply MiniMax H3 Fun ControlNet` node's `control_video` input.

## Model downloads

<CardGroup cols={2}>
  <Card title="Diffusion Model: minimax_h3_ref2va_pruned_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/diffusion_models/</code>
  </Card>
  <Card title="ControlNet Patch: minimax_h3_fun_controlnet_union_pruned_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/model_patches/</code>
  </Card>
  <Card title="Text Encoder: qwen3vl_32b_minimax_h3_nvfp4_awq" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors">
    Place in <code>ComfyUI/models/text_encoders/</code>
  </Card>
  <Card title="VAE: minimax_h3_video_vae_int8_convrot" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="VAE: minimax_h3_audio_vae_fp32" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors">
    Place in <code>ComfyUI/models/vae/</code>
  </Card>
  <Card title="LoRA: minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16" icon="download" href="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors">
    Powers the optional Lightning LoRA turbo mode. Place in <code>ComfyUI/models/loras/</code>
  </Card>
  <Card title="Checkpoint: sdpose_wholebody_fp16" icon="download" href="https://huggingface.co/Comfy-Org/SDPose/blob/main/checkpoints/sdpose_wholebody_fp16.safetensors">
    Pose extractor used by the built-in SDPose subgraph. Place in <code>ComfyUI/models/checkpoints/</code>
  </Card>
  <Card title="Detector: rt_detr_v4-x-hgnet_fp16" icon="download" href="https://huggingface.co/Comfy-Org/SDPose/blob/main/diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors">
    Person detector used by the built-in SDPose subgraph. Place in <code>ComfyUI/models/diffusion_models/</code>
  </Card>
</CardGroup>

## Model storage

```
ComfyUI/
├── 📂 models/
│   ├── 📂 diffusion_models/
│   │   ├── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   │   └── rt_detr_v4-x-hgnet_fp16.safetensors
│   ├── 📂 text_encoders/
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── 📂 vae/
│   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
│   │   └── minimax_h3_audio_vae_fp32.safetensors
│   ├── 📂 loras/
│   │   └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
│   ├── 📂 model_patches/
│   │   └── minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors
│   └── 📂 checkpoints/
│       └── sdpose_wholebody_fp16.safetensors
```

## Prompting tips

1. **Video length**: The target video runs 5 to 15 seconds on the 17n+5 frame grid at 24fps (124 frames = 5 seconds). A control video longer than the target is trimmed to its first frames; a shorter one holds its last frame. To match the control video's length exactly, link the example's computed `batch_size` output to the length input of the `MiniMax H3 Reference to Video` node
2. **Guidance and strength**: Keep `guidance_scale` at 1.0. Raise the patch `strength` above 1 only if the output drifts away from the control
3. **Video inpainting**: Connect a mask to the `mask` input: regions marked `1` are regenerated under control of the optional `source_video` input, while the rest of the frame stays fixed
4. **Turbo mode (optional)**: Enable the **Lightning LoRA** checkbox to use the 4-step turbo LoRA for much faster generation, with slightly lower audio and motion quality
5. **Preparing control videos**: ComfyUI ships with built-in preprocessing nodes for each control type. See the [Detect Edges (Canny) node](https://docs.comfy.org/built-in-nodes/Canny) for edges and the [Depth Anything 3 tutorial](https://docs.comfy.org/tutorials/utility/depth-anything-3) for depth maps
