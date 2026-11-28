# Comfy-Org__docs__minimax-h3-node-reference

Source: https://github.com/Comfy-Org/docs/blob/main/built-in-nodes/
Fetched: 2026-11-28



---

## FILE: built-in-nodes/EmptyMiniMaxH3LatentAV.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/EmptyMiniMaxH3LatentAV.mdx)

---
title: "Empty MiniMax H3 AV Latent - ComfyUI Built-in Node Documentation"
description: "This node creates an empty latent that combines both video and audio information for the MiniMax H3 model."
sidebarTitle: "Empty MiniMax H3 AV Latent"
icon: "circle"
mode: wide
---

This node creates an empty latent that combines both video and audio information for the MiniMax H3 model. You define the width, height, and length of the content, and the node produces a blank latent that the model can use as a starting point for generation. The duration (length) is automatically adjusted to fit the model's required frame grid of 17k+5 frames at 24 fps.

## Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `width` | The width of the latent in pixels. Values must be multiples of 32. Default: 1344. | INT | Yes | 32 - MAX_RESOLUTION (step 32) |
| `height` | The height of the latent in pixels. Values must be multiples of 32. Default: 768. | INT | Yes | 32 - MAX_RESOLUTION (step 32) |
| `length` | Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested). Default: 124. | INT | Yes | 5 - 3600 (step 17) |

Note: The `length` value is snapped up to the next frame count that fits the model's 17k+5 grid (17 x k + 5 frames, such as 5, 22, 39, 56, 73, 90, 107, 124, and so on). The `width` and `height` values must be multiples of 32. The maximum resolution is the system-defined value in ComfyUI.

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `latent` | The generated empty joint video+audio latent for MiniMax H3, sized according to the input width, height, and length. | LATENT |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/EmptyMiniMaxH3LatentAV/en.md)

---
**Source fingerprint (SHA-256):** `ee24f4ac630858d87b9b98bb402689a5790e0ed882ec47dffe7b497216e37a5c`


---

## FILE: built-in-nodes/MiniMaxH3AddGuide.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MiniMaxH3AddGuide.mdx)

---
title: "Add Guide for MiniMax H3 - ComfyUI Built-in Node Documentation"
description: "This node anchors an image, a short clip, audio, or a clip with its soundtrack at any frame of a MiniMax H3 video."
sidebarTitle: "Add Guide for MiniMax H3"
icon: "circle"
mode: wide
---

This node anchors an image, a short clip, audio, or a clip with its soundtrack at any frame of a MiniMax H3 video. It adds a guide keyframe to the conditioning at the chosen frame index, and you can chain several of these nodes to anchor several frames in the same video.

## Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `positive` | The conditioning to which the guide keyframe is attached. | CONDITIONING | Yes | - |
| `vae` | Video VAE, needed when an image is connected. | VAE | No | - |
| `audio_vae` | Audio VAE, needed when an audio is connected. | VAE | No | - |
| `latent` | The MiniMax H3 audio-video latent that defines the target video. Must be a MiniMax H3 AV latent (nested, with two 5D tensors, the video tensor having 24 channels). | LATENT | Yes | - |
| `image` | Image or video frames to anchor. Multi-frame batches are anchored as a clip and cropped down to the model's valid clip lengths: 5, 22, 39... (17k + 5) frames. Batches shorter than 5 frames use only the first image. | IMAGE | No | - |
| `audio` | Soundtrack to anchor starting at the same frame index, cropped to the video's remaining duration. | AUDIO | No | - |
| `frame_idx` | Frame index to anchor the image or the clip's first frame at. Negative values are counted from the end of the video. (default: 0) | INT | Yes | -9999 to 9999 |

**Constraints:**
- At least one of `image` or `audio` must be provided; otherwise the node raises an error.
- `vae` is required when `image` is connected.
- `audio_vae` is required when `audio` is connected.
- `image` batches with fewer than 5 frames use only the first image; batches of 5 or more frames are cropped down to a valid clip length (5, 22, 39, etc.).
- `frame_idx` must place the guide within the video's frame range, and a multi-frame clip must fit entirely in the video; otherwise the node raises an error.
- When audio is connected, the frame index must not be past the end of the video's audio track.

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `positive` | The conditioning with the guide keyframe added, containing the resolved frame index and, when provided, the encoded image or audio latents. | CONDITIONING |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3AddGuide/en.md)

---
**Source fingerprint (SHA-256):** `7a2f742421cc2655bd9c914258801e4538f1554a7c5e2b0836b2df1577f5a104`


---

## FILE: built-in-nodes/MiniMaxH3ContextWindows.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MiniMaxH3ContextWindows.mdx)

---
title: "MiniMax H3 Context Windows - ComfyUI Built-in Node Documentation"
description: "Splits sampling into overlapping context windows so a MiniMax H3 video can be longer than the model's single-pass window."
sidebarTitle: "MiniMax H3 Context Windows"
icon: "circle"
mode: wide
---

Splits sampling into overlapping context windows so a MiniMax H3 video can be longer than the model's single-pass window. The windows are placed on the model's 17-frame chunks, which keeps the generated video and audio aligned across window boundaries. Connect the returned model to the sampler in place of the original model.

## Inputs

| Parameter | Description | Data Type | Required | Range |
| --- | --- | --- | --- | --- |
| `model` | The model to apply context windows to during sampling. | MODEL | Yes | - |
| `context_length` | The length of the context window in real frames. Must be 17*n + 5 (default: 124). | INT | Yes | 22 to 16384 (MAX_RESOLUTION), step 17 |
| `context_overlap` | The overlap of the context window in real frames. Must be 17*n + 5 (default: 39). | INT | Yes | 5 or higher, step 17 |
| `fuse_method` | The method used to fuse the context windows (default: `"pyramid"`). | COMBO | Yes | `"pyramid"`<br />`"relative"`<br />`"flat"`<br />`"overlap-linear"` |
| `freenoise` | Whether to apply FreeNoise noise shuffling, which improves window blending (default: True, advanced parameter). | BOOLEAN | Yes | true<br />false |
| `split_conds_to_windows` | Whether to split multiple conditionings (created by ConditionCombine) to each window based on region index (default: False, advanced parameter). | BOOLEAN | Yes | true<br />false |

Both lengths are given in real frames and converted to latents before sampling: `context_length` becomes 2 latents at 5 frames or fewer, otherwise `((frames - 5) / 17) * 5 + 2`. `context_overlap` is converted the same way and then capped at `context_length` minus 5, so the overlap always leaves a full chunk to advance. The node always uses the static schedule with a stride of 1, applies the windows on the latent time dimension, and does not add the causal fix frame, because every window starts on a chunk boundary. `split_conds_to_windows` expects multiple conditionings created by the ConditionCombine node.

## Outputs

| Output Name | Description | Data Type |
| --- | --- | --- |
| `model` | The model with the applied context window configuration. | MODEL |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3ContextWindows/en.md)

---
**Source fingerprint (SHA-256):** `e9e019d0bc870abfe226ef1aafb2607f55c0cc1b297884db35fe98c465cc0d91`


---

## FILE: built-in-nodes/MiniMaxH3FunControlNetApply.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MiniMaxH3FunControlNetApply.mdx)

---
title: "Apply MiniMax H3 Fun ControlNet - ComfyUI Built-in Node Documentation"
description: "This node applies a MiniMax H3 Fun ControlNet to a text-to-video model as a model patch."
sidebarTitle: "Apply MiniMax H3 Fun ControlNet"
icon: "circle"
mode: wide
---

This node applies a MiniMax H3 Fun ControlNet to a text-to-video model as a model patch. It can use an optional control video and an optional mask to steer generation, and it returns a patched copy of the model for further sampling. When the strength is set to 0, or when neither a control video nor a mask is provided, the input model is returned unchanged.

## Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `model` | The diffusion model to which the MiniMax H3 Fun ControlNet patch is applied. | MODEL | Yes | N/A |
| `model_patch` | The MiniMax H3 Fun ControlNet patch whose control signals are injected into the model; it must be compatible with the given `model`. | MODEL_PATCH | Yes | N/A |
| `vae` | VAE used to encode the control and source video frames into the latent space expected by the model. | VAE | Yes | N/A |
| `strength` | Overall strength of the ControlNet effect. When set to 0, the node does nothing and returns the input model unchanged. (default: 1.0) | FLOAT | Yes | min 0.0, max 10.0, step 0.01 |
| `start_percent` | Start of the sampling range, expressed as a percentage of the sampling schedule, during which the ControlNet is active. It is converted internally to the equivalent sigma value. This is an advanced setting. (default: 0.0) | FLOAT | Yes | min 0.0, max 1.0, step 0.001 |
| `end_percent` | End of the sampling range, expressed as a percentage of the sampling schedule, during which the ControlNet is active. It is converted internally to the equivalent sigma value. This is an advanced setting. (default: 1.0) | FLOAT | Yes | min 0.0, max 1.0, step 0.001 |
| `control_video` | Optional video frames used as the ControlNet visual hint. The frames are resized to match the generated video and encoded with the `vae`. | IMAGE | No | N/A |
| `mask` | 1 marks the regions to regenerate. Mask values above 0.5 are treated as marked regions. | MASK | No | N/A |
| `source_video` | Video behind the mask; only read when a mask is given. | IMAGE | No | N/A |

Note: For the patch to have an effect, `strength` must be greater than 0 and at least one of `control_video` or `mask` must be supplied. `source_video` is ignored unless `mask` is provided; if `mask` is given without `source_video`, the content behind the masked regions is treated as black. When both `control_video` and `mask` are provided, the control hint combines the control video features with the masked source content.

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `model` | A patched clone of the input model with the MiniMax H3 Fun ControlNet applied. If `strength` is 0, or no control video or mask is supplied, the original model is returned unchanged. | MODEL |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3FunControlNetApply/en.md)

---
**Source fingerprint (SHA-256):** `e907fb8e5ae60663d1d10b315985695ee5d49397fef6bd76b0e723637457a74a`


---

## FILE: built-in-nodes/MiniMaxH3ImageConditioning.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MiniMaxH3ImageConditioning.mdx)

---
title: "MiniMax H3 Image Conditioning - ComfyUI Built-in Node Documentation"
description: "Turns a prompt into conditioning and an empty latent for a still image with MiniMax H3 Image."
sidebarTitle: "MiniMax H3 Image Conditioning"
icon: "circle"
mode: wide
---

Turns a prompt into conditioning and an empty latent for a still image with MiniMax H3 Image. Without images it is text to image; with images connected it edits them, and the prompt refers to them as `<Picture 1>`, `<Picture 2>`, and so on.

## Inputs

| Parameter | Description | Data Type | Required | Range |
| --- | --- | --- | --- | --- |
| `clip` | The MiniMax H3 Image CLIP model, which tokenizes the prompt with any reference images and encodes the conditioning. | CLIP | Yes | - |
| `vae` | The MiniMax H3 Image VAE. With it the pictures are also anchored as reference latents; without it they only condition the text encoder. | VAE | No | - |
| `prompt` | What to generate, or how to edit the reference pictures. | STRING | Yes | Multiline text |
| `width` | Output width in pixels. 0 uses the width of `image_1` after its reference scaling, or 2048 without images. Default: 0. | INT | Yes | 0 to MAX_RESOLUTION (step 32) |
| `height` | Output height in pixels. 0 uses the height of `image_1` after its reference scaling, or 2048 without images. Default: 0. | INT | Yes | 0 to MAX_RESOLUTION (step 32) |
| `images` | Growable group of reference pictures, `image_1` upward; `image_1` is the picture to edit. The prompt addresses them as `<Picture n>`. | AUTOGROW | No | 0 to 100 images |

## Outputs

| Output Name | Description | Data Type |
| --- | --- | --- |
| `positive` | Conditioning for the MiniMax H3 Image sampler, with the reference latents attached when images are connected. | CONDITIONING |
| `latent` | Empty latent for one image, 24 channels at a sixteenth of the output size. | LATENT |

Note: The model is tuned for 2048x2048 pixels of area, so leave `width` and `height` at 0 unless a different canvas is needed.

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3ImageConditioning/en.md)

---
**Source fingerprint (SHA-256):** `b0927bbc44ac82e8a23b1edb5470c0cc8945844c326a352d75495ed801d11b1d`


---

## FILE: built-in-nodes/MiniMaxH3ImageToVideo.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MiniMaxH3ImageToVideo.mdx)

---
title: "MiniMax H3 Image to Video - ComfyUI Built-in Node Documentation"
description: "This node prepares the conditioning and empty latent needed to generate a video with the MiniMax H3 model."
sidebarTitle: "MiniMax H3 Image to Video"
icon: "circle"
mode: wide
---

This node prepares the conditioning and empty latent needed to generate a video with the MiniMax H3 model. It takes a text prompt and, optionally, images for the first and/or last frame of the video, and converts them into model inputs. Keyframe images are resized, encoded, and attached to the conditioning at the start and end of the video.

## Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `clip` | CLIP model used to tokenize the prompt and encode the keyframe images into conditioning. | CLIP | Yes |  |
| `vae` | VAE model used to encode the keyframe images into latent space when keyframe images are provided. | VAE | Yes |  |
| `prompt` | Text prompt describing the video to generate. Supports multiple lines and dynamic prompts. | STRING | Yes |  |
| `width` | Width of the video in pixels (default: 1344). | INT | Yes | 32 to MAX_RESOLUTION (step 32) |
| `height` | Height of the video in pixels (default: 768). | INT | Yes | 32 to MAX_RESOLUTION (step 32) |
| `length` | Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested) (default: 124). | INT | Yes | 5 to 3600 (step 17) |
| `first_frame` | Optional image used as the first frame of the video. It is stretched to the full canvas size, so its aspect ratio is not preserved. Only the first image of the input batch is used. | IMAGE | No |  |
| `last_frame` | Optional image used as the last frame of the video. It is cropped to cover the canvas while preserving its aspect ratio. Only the first image of the input batch is used. | IMAGE | No |  |

When `first_frame` and/or `last_frame` are provided, the keyframe images are encoded with the VAE and attached to the conditioning at frame 0 and at the final frame, respectively. When neither is provided, the node works from the prompt alone. The requested `length` is snapped up to the nearest valid frame count (17k + 5), so the effective frame count may be slightly higher than requested.

The audio-video latent is created as an empty pair matching the requested `width`, `height`, and the snapped frame count. The audio portion is sized from the same frame count at 40 audio frames per second.

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `positive` | Conditioning containing the encoded prompt and, when keyframe images are provided, the encoded keyframes and frame count for the MiniMax H3 model. | CONDITIONING |
| `latent` | Empty audio-video latent representing the content to be generated, with the requested width, height, and frame count. | LATENT |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3ImageToVideo/en.md)

---
**Source fingerprint (SHA-256):** `d1bdf3f8c66ef20ff11c35203d2c266a88dcf8cc00c65dbb0aea2b1dd16befd6`


---

## FILE: built-in-nodes/MiniMaxH3LatentUpscaleWithModel.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MiniMaxH3LatentUpscaleWithModel.mdx)

---
title: "MiniMax H3 Latent Upscale With Model - ComfyUI Built-in Node Documentation"
description: "MiniMax H3 Latent Upscale With Model resizes the video part of a MiniMax H3 latent to a target width and height with a latent upscale model."
sidebarTitle: "MiniMax H3 Latent Upscale With Model"
icon: "circle"
mode: wide
---

MiniMax H3 Latent Upscale With Model resizes the video part of a MiniMax H3 latent to a target width and height with a latent upscale model. The audio latent rides along untouched, so the clip keeps its sound, and no re-sampling is needed. Use it after MiniMax H3 sampling and before the H3 VAE decode when the clip should come out larger.

## Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `model` | Latent upscale model, usually loaded with Load Latent Upscale Model. It has to be a MiniMax H3 3D-conv latent upscaler. | LATENT_UPSCALE_MODEL | Yes | - |
| `samples` | MiniMax H3 latent to resize. A nested latent keeps its audio part as it is. | LATENT | Yes | - |
| `width` | Target video width in pixels. The latent is resized to `width / 16` columns, so the value is rounded down to a multiple of 16. (default: 2560) | INT | Yes | 32 to 16384 (step 32) |
| `height` | Target video height in pixels, also rounded down to a multiple of 16. (default: 1440) | INT | Yes | 32 to 16384 (step 32) |

When the requested size already matches the latent, the input latent is returned unchanged. A resized latent carries no `noise_mask`, so a partially masked latent loses its mask. The network runs its 3D convolutions in temporal slabs, which keeps memory use on long clips close to that of a single slab.

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `LATENT` | The resized latent, ready for the H3 VAE decode. | LATENT |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3LatentUpscaleWithModel/en.md)

---
**Source fingerprint (SHA-256):** `cd659baace5a21620033dd3470b854cccbada01d09545a2f61273b949c3a004b`


---

## FILE: built-in-nodes/MiniMaxH3ReferenceToVideo.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MiniMaxH3ReferenceToVideo.mdx)

---
title: "MiniMax H3 Reference to Video - ComfyUI Built-in Node Documentation"
description: "MiniMax H3 Reference to Video creates the text conditioning and the empty audio-video latent needed for MiniMax H3 reference-to-video generation."
sidebarTitle: "MiniMax H3 Reference to Video"
icon: "circle"
mode: wide
---

MiniMax H3 Reference to Video creates the text conditioning and the empty audio-video latent needed for MiniMax H3 reference-to-video generation. You provide a prompt plus optional reference images, videos, and audio clips, and the node encodes these references into conditioning the model can use while generating. The prompt refers to the references with `<Picture i>`, `<Video k>`, and `<Audio j>` tags.

## Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `clip` | CLIP model used to tokenize the prompt and encode the reference media into conditioning tokens. | CLIP | Yes | |
| `vae` | Video VAE. Without it reference images/videos only condition the text encoder. | VAE | No | |
| `audio_vae` | Audio VAE. Without it reference audio only conditions the text encoder. | VAE | No | |
| `prompt` | Text prompt for the video. Reference media can be addressed with `<Picture i>`, `<Video k>`, and `<Audio j>` tags (1-based per type). Supports multiline and dynamic prompts. | STRING | Yes | |
| `width` | Width of the generated video in pixels (default: 1344). | INT | Yes | 32 to 16384 (step 32) |
| `height` | Height of the generated video in pixels (default: 768). | INT | Yes | 32 to 16384 (step 32) |
| `length` | Frame count at 24 fps, (124 = ~5s, trained range is ~124-362) (default: 124). | INT | Yes | 5 to 3600 (step 17) |
| `ref_image_size` | Reference image sizing. `match` scales each ref (down only, keeping aspect) to the generation's pixel area; `max` uses the reference pipeline's 2048px short edge for best identity fidelity. Reference tokens ride through every sampling step, so `max` can be several times slower (default: `match`). | COMBO | Yes | `"match"`<br />`"max"` |
| `ref_images` | Growable slot: connect up to 9 reference images (`ref_image_1` ... `ref_image_9`). Reference image (downscaled to 2048 short edge if larger, never upscaled). | IMAGE | No | 0 to 9 |
| `ref_videos` | Growable slot: connect up to 3 reference videos (`ref_video_1` ... `ref_video_3`). Reference video frames at 24 fps (2-15s). | IMAGE | No | 0 to 3 |
| `ref_video_audios` | Growable slot: connect up to 3 soundtracks (`ref_video_audio_1` ... `ref_video_audio_3`). Soundtrack of the same-numbered reference video. | AUDIO | No | 0 to 3 |
| `ref_audios` | Growable slot: connect up to 3 standalone reference audio clips (`ref_audio_1` ... `ref_audio_3`). Standalone reference audio. | AUDIO | No | 0 to 3 |

Notes:

- The prompt refers to reference media with 1-based tags per type: `<Picture i>` for images, `<Video k>` for videos, and `<Audio j>` for audio. References are presented to the model in a fixed order: images, then videos (with each soundtrack's `<Audio j>` label right before its `<Video k>`), then standalone audio.
- A soundtrack connected to `ref_video_audio_N` is used with the reference video connected to `ref_video_N`.
- Reference videos must contain at least 5 frames (~0.2 seconds at 24 fps), otherwise the node raises an error. Frames beyond the requested `length` are trimmed, and the remaining frame count is adjusted to a value supported by the model.
- The requested `length` is aligned to a supported frame count before the latent is created.
- Without `vae`, reference images and videos only condition the text encoder (no reference latents are produced). Without `audio_vae`, reference audio only conditions the text encoder.

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `positive` | Conditioning containing the encoded prompt. When reference media and the relevant VAEs are provided, it also contains the encoded reference image, video, and audio content used by the MiniMax H3 model. | CONDITIONING |
| `latent` | Empty audio-video latent at the requested `width`, `height`, and `length` (frame count), including the aligned video latent and audio latent. | LATENT |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3ReferenceToVideo/en.md)

---
**Source fingerprint (SHA-256):** `47df0d6d13cb02aa4f69b50a7f8d0f6c1639c1fb5e0f69bf8fc57dd4cb752db8`


---

## FILE: built-in-nodes/MiniMaxH3SigmaShift.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MiniMaxH3SigmaShift.mdx)

---
title: "ModelSamplingMiniMaxH3 - ComfyUI Built-in Node Documentation"
description: "Sets the video and audio flow shift values for a MiniMax H3 model."
sidebarTitle: "ModelSamplingMiniMaxH3"
icon: "circle"
mode: wide
---

Sets the video and audio flow shift values for a MiniMax H3 model. The video shift controls the sampler's sigma schedule, and both shift values are passed to the model's internal transformer, which uses them to derive the audio schedule from the shared base grid.

## Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `model` | The model to apply the sigma shift patch to. The node clones the model, so the original remains unchanged. | MODEL | Yes | - |
| `shift_video` | The video flow shift value. It drives the sampler's sigma schedule. Default: 12.0. | FLOAT | Yes | 0.01 to 100.0 |
| `shift_audio` | The audio flow shift value. It is used by the model to derive the audio schedule. Default: 3.0. | FLOAT | Yes | 0.01 to 100.0 |

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `MODEL` | The cloned model with the video and audio sigma shift settings applied. | MODEL |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3SigmaShift/en.md)

---
**Source fingerprint (SHA-256):** `0f731585cc1a9c87a3e54341757c4cf4e490d1d4718ecf458bd2b9f4378af63f`


---

## FILE: built-in-nodes/ModelMergeMiniMaxH3.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/ModelMergeMiniMaxH3.mdx)

---
title: "Model Merge MiniMax H3 - ComfyUI Built-in Node Documentation"
description: "Merge two compatible MiniMax H3 models by component, with a separate ratio for each part of the architecture."
sidebarTitle: "Model Merge MiniMax H3"
icon: "circle"
mode: wide
---

Merge two compatible MiniMax H3 models by component, with a separate ratio for each part of the architecture. Each ratio is the share of `model1`, so you can keep the patch projections and refinement blocks of one model while replacing the transformer blocks with another.

## Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `model1` | First MiniMax H3 model. Every component starts from it, and each ratio decides how much of that component comes from `model2` instead. | MODEL | Yes | - |
| `model2` | Second MiniMax H3 model that supplies the components you blend in. | MODEL | Yes | - |
| `video_patch_proj.` | Share of `model1` for the video patch projection (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `audio_patch_proj.` | Share of `model1` for the audio patch projection (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `condition_proj.` | Share of `model1` for the conditioning projection (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `time_embedder.` | Share of `model1` for the time embedder (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `token_refiner.blocks.0.` | Share of `model1` for the first token refiner block (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `token_refiner.blocks.1.` | Share of `model1` for the second token refiner block (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `token_refiner.final_norm.` | Share of `model1` for the token refiner final normalization (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `blocks.0.` through `blocks.49.` | Share of `model1` for each of the 50 transformer blocks, one widget per block (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `final_layer.norm.` | Share of `model1` for the final layer normalization (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `final_layer.adaln_proj.` | Share of `model1` for the final adaptive normalization projection (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `final_layer.video_out.` | Share of `model1` for the video output layer (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |
| `final_layer.audio_out.` | Share of `model1` for the audio output layer (default: 1.0). | FLOAT | Yes | 0.0 to 1.0 (step: 0.01) |

A ratio of 1.0 (the default) keeps that component entirely from `model1` and 0.0 takes it entirely from `model2`; values in between blend the two. Tensors that carry no ratio, such as normalization statistics, always come from `model1`. Both inputs must be MiniMax H3 models with the same architecture, otherwise the merge produces a model that does not run.

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `MODEL` | The merged MiniMax H3 model. | MODEL |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/ModelMergeMiniMaxH3/en.md)

---
**Source fingerprint (SHA-256):** `4fea4e588d16a8396c68aa071a59e671a2a0c645734606be6d7e221986ba88bd`


---

## FILE: built-in-nodes/MinimaxHailuo03ContextIRNode.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MinimaxHailuo03ContextIRNode.mdx)

---
title: "MiniMax H3 Context IR (Prompt Enhancer) - ComfyUI Built-in Node Documentation"
description: "This node uses MiniMax H3 Context IR to analyze your text description and any attached media, then produces a stronger, structured video prompt."
sidebarTitle: "MiniMax H3 Context IR (Prompt Enhancer)"
icon: "circle"
mode: wide
---

## Overview

This node uses MiniMax H3 Context IR to analyze your text description and any attached media, then produces a stronger, structured video prompt. The returned prompt is designed to be connected to the prompt input of a MiniMax H3 video node; if you attach media there, attach the same media in the same order, because the enhanced prompt refers to media by position.

## Inputs

### Common Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `model` | Model to use for prompt enhancement. | DYNAMIC_COMBO | Yes | `"MiniMax H3"` |
| `first_frame` | First frame of the video you intend to generate. Cannot be combined with reference media. | IMAGE | No | Single image |
| `last_frame` | Last frame of the video you intend to generate. Cannot be combined with reference media. | IMAGE | No | Single image |

### MiniMax H3 Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `prompt` | Description of the video you intend to generate. Cannot be empty. (default: `""`) | STRING | Yes | Any text (cannot be empty) |
| `duration` | Duration of the video you intend to generate, in seconds (4-15). (default: 5) | INT | Yes | 4 to 15 |
| `ratio` | Aspect ratio of the video you intend to generate. `"adaptive"` requires at least one image, video, or audio input. (default: `"adaptive"`) | COMBO | Yes | `"adaptive"`<br />`"16:9"`<br />`"4:3"`<br />`"1:1"`<br />`"3:4"`<br />`"9:16"`<br />`"21:9"` |

### Reference Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `reference_images` | Subject or style reference images, referred to in the prompt as "Image 1".."Image 9" in connection order. Up to 9 images. Growable slot: connect `image_1`...`image_9`. | IMAGE | No | 0 to 9 images |
| `reference_videos` | Motion or scene reference videos, referred to in the prompt as "Video 1".."Video 3" in connection order. Up to 3 videos, 2-15 seconds each, 15 seconds in total. Growable slot: connect `video_1`...`video_3`. | VIDEO | No | 0 to 3 videos |
| `reference_audios` | Audio references, referred to in the prompt as "Audio 1".."Audio 3" in connection order. Up to 3 clips, 2-15 seconds each, 15 seconds in total. Cannot be used without a reference image or video. Growable slot: connect `audio_1`...`audio_3`. | AUDIO | No | 0 to 3 clips |

### Parameter constraints

- The `prompt`, `duration`, `ratio`, `reference_images`, `reference_videos`, and `reference_audios` inputs are part of the `model` option group and appear when "MiniMax H3" is selected.
- `first_frame` and `last_frame` cannot be combined with any reference media.
- `reference_audios` cannot be used unless at least one `reference_image` or `reference_video` is also connected.
- When no frames and no reference media are connected, `ratio` cannot be set to `"adaptive"`.
- Reference videos must be approximately 2-15 seconds each, with a total duration of no more than 15 seconds. Their frame rate must be between 23.9 and 60.5 FPS.
- Reference audios must be approximately 2-15 seconds each, with a total duration of no more than 15 seconds.
- `first_frame`, `last_frame`, and every reference image must be at least 256x256 pixels and have an aspect ratio between 0.4 and 2.5.

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `STRING` | The enhanced, structured video prompt generated by MiniMax H3 Context IR. It can be connected to the prompt input of a MiniMax H3 video generation node. | STRING |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MinimaxHailuo03ContextIRNode/en.md)

---
**Source fingerprint (SHA-256):** `73015517f9c0f55f0aceeef935508a372e0d95668e4733d1c8100b53e4afa7e2`


---

## FILE: built-in-nodes/MinimaxHailuo03ReferenceNode.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MinimaxHailuo03ReferenceNode.mdx)

---
title: "MiniMax H3 Reference to Video - ComfyUI Built-in Node Documentation"
description: "This node generates a video using the MiniMax H3 models, conditioned on reference images, videos, and audio."
sidebarTitle: "MiniMax H3 Reference to Video"
icon: "circle"
mode: wide
---

This node generates a video using the MiniMax H3 models, conditioned on reference images, videos, and audio. References are referred to in the prompt by their connection order: "@Image1", "@Image2", "@Video1", "@Audio1", and so on. Two models are available: "MiniMax H3" and "MiniMax H3 Max".

## Inputs

### Common Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `model` | Model to use for video generation (default: "MiniMax H3"). Selecting "MiniMax H3" provides the MiniMax H3 generation and reference inputs below. Selecting "MiniMax H3 Max" provides the MiniMax H3 Max generation and reference inputs below. | DYNAMIC_COMBO | Yes | "MiniMax H3"<br />"MiniMax H3 Max" |
| `seed` | Random seed. The same request with the same seed gives similar, but not guaranteed identical, results (default: 42). | INT | Yes | 0 to 4294967295 |
| `watermark` | Whether to add an AIGC watermark to the video (default: false). Only supported by the MiniMax H3 model. | BOOLEAN | No | true<br />false |

### MiniMax H3 Inputs

These inputs are available when "MiniMax H3" is selected as the model.

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `prompt` | Text prompt for video generation. Reference media can be referred to by their order, for example "@Image1", "@Image2", "@Video1", or "@Audio1". | STRING | Yes | Minimum 1 character |
| `resolution` | Resolution of the output video (default: "768P"). | COMBO | Yes | "768P"<br />"2K" |
| `ratio` | Aspect ratio of the output video (default: "adaptive"). | COMBO | Yes | "adaptive"<br />"16:9"<br />"4:3"<br />"1:1"<br />"3:4"<br />"9:16"<br />"21:9" |
| `duration` | Duration of the output video in seconds (default: 5). | INT | Yes | 4 to 15 |

### MiniMax H3 Max Inputs

These inputs are available when "MiniMax H3 Max" is selected as the model.

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `prompt` | Text prompt for video generation. Reference media can be referred to by their order, for example "@Image1", "@Image2", "@Video1", or "@Audio1". | STRING | Yes | 1 to 50000 characters |
| `resolution` | Resolution of the output video (default: "768P"). | COMBO | Yes | "480P"<br />"768P" |
| `ratio` | Aspect ratio of the output video (default: "adaptive"). | COMBO | Yes | "adaptive"<br />"16:9"<br />"4:3"<br />"1:1"<br />"3:4"<br />"9:16"<br />"21:9" |
| `duration` | Duration of the output video in seconds (default: 5). | INT | Yes | 5 to 15 |
| `prompt_expansion_mode` | How much effort is spent rewriting the prompt before generation (default: "balanced"). | COMBO | Yes | "balanced"<br />"quality" |
| `reference_detail` | Detail level at which reference images are sent. "high" sends them at the largest size the model uses (up to a 2048 pixel short side); "standard" downsizes them to at most 2048x1024 to reduce the reference cost (default: "standard"). | COMBO | Yes | "high"<br />"standard" |

### Reference Inputs

These reference inputs are shared by both models. Each is a growable slot.

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `reference_images` | Growable slot: connect up to 9 items (`image_1`...`image_9`). Subject or style reference images, referred to in the prompt as "@Image1".."@Image9" in connection order. Up to 9 images. | IMAGE | No | 0 to 9 images |
| `reference_videos` | Growable slot: connect up to 3 items (`video_1`...`video_3`). Motion or scene reference videos, referred to in the prompt as "@Video1".."@Video3" in connection order. Up to 3 videos, 2-15 seconds each, 15 seconds in total. | VIDEO | No | 0 to 3 videos |
| `reference_audios` | Growable slot: connect up to 3 items (`audio_1`...`audio_3`). Audio references, referred to in the prompt as "@Audio1".."@Audio3" in connection order. Up to 3 clips, 2-15 seconds each, 15 seconds in total. Cannot be used without a reference image or video. | AUDIO | No | 0 to 3 clips |

### Parameter Constraints

- At least one reference image or one reference video is required. Reference audio alone is not accepted.
- Each reference image must have an aspect ratio between approximately 0.4 and 2.5 (2:5 to 5:2) and a minimum width and height of 256 pixels.
- Each reference video must be between 2 and 15 seconds long with a frame rate between 23.976 and 60 FPS. The total duration of all reference videos cannot exceed 15 seconds.
- Each reference audio clip must be between 2 and 15 seconds long. The total duration of all reference audio clips cannot exceed 15 seconds.
- When "MiniMax H3 Max" is selected, the `watermark` setting must be disabled.
- When "MiniMax H3 Max" is selected, the total number of reference files (images, videos, and audio combined) cannot exceed 12.

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `video` | The generated video. | VIDEO |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MinimaxHailuo03ReferenceNode/en.md)

---
**Source fingerprint (SHA-256):** `4c0dfb349809b51393b8ec8afd235977f1822c0c8ae6dc42a2c264271bb50b53`


---

## FILE: built-in-nodes/MinimaxHailuo03RegenerateNode.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/built-in-nodes/MinimaxHailuo03RegenerateNode.mdx)

---
title: "MiniMax H3 Regenerate to 2K - ComfyUI Built-in Node Documentation"
description: "This node re-renders a MiniMax H3 768P video output at 2K resolution."
sidebarTitle: "MiniMax H3 Regenerate to 2K"
icon: "circle"
mode: wide
---

This node re-renders a MiniMax H3 768P video output at 2K resolution. It uploads the unmodified 768P video and the exact prompt used to generate it, starts a MiniMax H3 regeneration job, and returns the re-rendered 2K video. If the original generation used first or last frames or reference media, attach the same inputs.

## Inputs

### Common Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `model` | The model to use for video regeneration. Selecting "MiniMax H3" reveals the prompt, resolution, and reference media settings. | DYNAMIC_COMBO | Yes | "MiniMax H3" |
| `video` | The MiniMax H3 768P output video to re-render. Connect the unmodified output of a MiniMax H3 video node (24 FPS, 4-15 seconds). 2K outputs cannot be used. | VIDEO | Yes | 24 FPS, 4-15 seconds |
| `first_frame` | First frame image from the original generation, if one was used. | IMAGE | No | Image |
| `last_frame` | Last frame image from the original generation, if one was used. | IMAGE | No | Image |
| `watermark` | Whether to add an AIGC watermark to the video. Default is false. | BOOLEAN | Yes | false / true |

### MiniMax H3 Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `prompt` | The exact prompt used to generate the source video. Must not be empty. | STRING | Yes | Text (multiline) |
| `resolution` | Resolution to re-render the source video at. | COMBO | Yes | "2K" |

### Reference Inputs

| Parameter | Description | Data Type | Required | Range |
|-----------|-------------|-----------|----------|-------|
| `reference_images` | Growable slot: connect `image_1` through `image_9` (up to 9 images). Reference images from the original generation, in the same order. | IMAGE | No | 0-9 images |
| `reference_videos` | Growable slot: connect `video_1` through `video_3` (up to 3 videos). Reference videos from the original generation, in the same order. | VIDEO | No | 0-3 videos |
| `reference_audios` | Growable slot: connect `audio_1` through `audio_3` (up to 3 clips). Audio references from the original generation, in the same order. Cannot be used without a reference image or video. | AUDIO | No | 0-3 clips |

### Constraints

- The `prompt` must not be empty.
- The source `video` must be an unmodified MiniMax H3 768P output: 24 FPS, width and height divisible by 32, at most 1,032,192 total pixels, and 107 to 362 frames in steps of 17 (4 to 15 seconds at 24 FPS). 2K outputs cannot be used as the source.
- `first_frame` and `last_frame` are mutually exclusive with reference media (`reference_images`, `reference_videos`, `reference_audios`). Use frames for an image-to-video prompt, or reference media for a reference-to-video prompt.
- `reference_audios` requires at least one `reference_images` or `reference_videos` input.
- `first_frame`, `last_frame`, and each `reference_image` must have an aspect ratio between 0.4 and 2.5 and be at least 256x256 pixels.
- `reference_videos`: each video must be 23.976 to 60 FPS and 2-15 seconds long; total duration cannot exceed 15 seconds.
- `reference_audios`: each clip must be 2-15 seconds long; total duration cannot exceed 15 seconds.

## Outputs

| Output Name | Description | Data Type |
|-------------|-------------|-----------|
| `video` | The re-rendered MiniMax H3 video at 2K resolution. | VIDEO |

> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MinimaxHailuo03RegenerateNode/en.md)

---
**Source fingerprint (SHA-256):** `4b5aa6dee12364cf6f44e7ee78b984c3568529b97051637a6ac62db9761d3a77`


---

## FILE: tutorials/partner-nodes/minimax/minimax-h3.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/tutorials/partner-nodes/minimax/minimax-h3.mdx)

---
title: "MiniMax H3 API Video Generation in ComfyUI"
description: "Use the MiniMax H3 API nodes in ComfyUI for text-to-video, first/last-frame, and reference-to-video cloud workflows with native stereo audio."
sidebarTitle: "Minimax H3"
---
import ReqHint from "/snippets/tutorials/partner-nodes/req-hint.mdx";
import UpdateReminder from "/snippets/tutorials/update-reminder.mdx";


[MiniMax H3](https://www.minimax.io/blog/minimax-h3) is MiniMax's general-purpose, omni-modal generation model, available in ComfyUI through the MiniMax H3 API nodes. The API workflows run generation on MiniMax's servers, so no model downloads or local GPU is required, and each second of generated video is billed to your Comfy API account.

MiniMax H3 generates video with **native stereo audio**: voice, sound effects, and music are modeled together in a single forward pass instead of being layered on afterward. Output is up to 2K resolution at 5-15 seconds per clip.

**MiniMax H3 Max** is a variant of MiniMax H3 post-trained by fal for high-speed generation, added to the same API nodes in ComfyUI 0.34.2. It generates faster than real time, accepts a prompt of up to 50,000 characters, and offers 480P and 768P resolutions with an optional two-stage graph built from nodes ComfyUI already supports: prompt enhancement through the MiniMax H3 Context IR node, then regeneration to 2K with the MiniMax H3 Regenerate to 2K node.

**MiniMax H3 Max Turbo** is a faster, cheaper H3 Max option added to the Text to Video and First-Last-Frame nodes in ComfyUI 0.34.4. It renders at 480P or 768P and drops resolution ratio controls on the First-Last-Frame node, where the output ratio follows the input frames.

<Note>
These API workflows run on MiniMax's servers through your Comfy account. To run H3 locally on your own hardware for commercial work, you need a [MiniMax commercial license](https://comfy.org/minimax/license), available through Comfy, the only official reseller.
</Note>

## What MiniMax H3 is good at

- **Native stereo audio**: Voice, sound effects, and music are modeled together with the video in a single forward pass instead of being layered on afterward
- **High-resolution output**: Up to 2K resolution at 5-15 seconds per clip
- **Text-to-video**: Generates videos, with audio, from text prompts
- **First-last-frame video**: Generates the motion between a first frame and an optional last frame image
- **Reference-conditioned generation**: Generates videos conditioned on up to 9 reference images, 3 reference videos, and 3 reference audio clips
- **MiniMax H3 Max**: A faster variant with 480P/768P output, long prompts, and an optional 2K upscale stage. The Reference node accepts up to 12 reference files on H3 Max
- **MiniMax H3 Max Turbo**: The fastest H3 Max option for the Text to Video and First-Last-Frame nodes, at roughly half the Max per-second rate

## Example outputs

Text-to-video generation from a single prompt, with native stereo audio:

<video controls className="w-full aspect-video" src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/output/api_minimax_h3_t2v.mp4"></video>

First-last-frame generation, with the model creating the motion between two frames:

<video controls className="w-full aspect-video" src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/output/api_minimax_h3_flf2v.mp4"></video>

<ReqHint/>
<UpdateReminder/>

## MiniMax H3 API workflows

The template library ships with three API example workflows for the base H3 model, four for H3 Max, and two for H3 Max Turbo:

- **Text to Video**: Generate videos from text prompts
- **First-Last-Frame to Video**: Generate a video between a first and last frame image
- **Reference to Video**: Generate videos conditioned on reference images, videos, and audio
- **H3 Max T2V / I2V / FLF2V / R2V**: The same modes through the faster MiniMax H3 Max model, with optional prompt enhancement and 2K upscaling stages built into each workflow
- **H3 Max Turbo T2V / I2V**: Text-to-video and first/last-frame video through the fastest MiniMax H3 Max Turbo model, with the same optional prompt enhancement and 2K upscaling stages

### MiniMax H3 Max

The four `api_minimax_h3_max_*` templates run on the MiniMax H3 Max model through the same API nodes: the Text to Video and First-Last-Frame nodes gained an H3 Max option in ComfyUI 0.34.2, and the Reference to Video node in 0.34.3. Update ComfyUI to at least 0.34.3 to run all of them.

H3 Max generates faster than real time and bills at lower rates than the base model: 15.09 credits/sec at 480P and 24.14 credits/sec at 768P (see the [Minimax pricing table](/tutorials/partner-nodes/pricing)). Each Max template adds two optional stages behind a Switch node, both off by default:

1. **Prompt enhance**: Rewrites the prompt with the MiniMax H3 Context IR node before generation. Turn on the `Prompt Enhance` boolean to enable it
2. **Upscale to 2K**: Runs the MiniMax H3 Regenerate to 2K node on the first-pass video. Turn on the `Upscale to 2K` boolean to enable it

Keep the following in mind when wiring a Max template:

- **Generate at 768P if you plan to upscale**: The Regenerate to 2K node only accepts 768P source video
- **Match durations across nodes**: Generation and prompt enhance must use the same duration, or the stages drift out of sync. The Regenerate to 2K node has no duration input; it re-renders the submitted 768P video at its original duration
- **Duplicate extra inputs**: The three nodes do not share extra context automatically. If you add a reference video, audio, or another image, plug the same wires into both the Context IR and Regenerate nodes

### MiniMax H3 Text to Video

Generate videos from text prompts through the MiniMax H3 API, with native stereo audio.

<video controls className="w-full aspect-video" src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/output/api_minimax_h3_t2v.mp4"></video>

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=api_minimax_h3_t2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3-api">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_t2v.json">
    Download JSON or search "MiniMax H3 T2V" in Template Library
  </Card>
</CardGroup>

#### Prompting tips

1. **Prompt**: Describe the whole scene and the accompanying audio (dialogue, SFX, music) in one block
2. **Duration**: The duration input supports 5-15 seconds; the workflow ships with a fast 5-second preview
3. **Resolution and ratio**: The API renders at 2K; choose an aspect ratio preset such as `16:9`, `9:16`, or `1:1`
4. **Watermark**: The AIGC watermark is off by default; enable it in the node's advanced settings if required

#### Prompt writing guide

MiniMax publishes an official [video prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) for the base generation modes: T2VA, I2VA, FL2VA, and L2VA. The guide explains how to structure a prompt into timed shots with camera movement and audio (dialogue, SFX, music), and includes examples for each mode.

#### MiniMax H3 Max: Text to Video

Generate videos from text prompts on the faster H3 Max model. The template renders at 768P by default, with optional prompt enhancement and 2K upscaling stages.

![MiniMax H3 Max Text to Video workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/api_minimax_h3_max_t2v-1.webp)

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=api_minimax_h3_max_t2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3-api">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_max_t2v.json">
    Download JSON or search "MiniMax H3 Max: Text to Video" in Template Library
  </Card>
</CardGroup>

#### MiniMax H3 Max Turbo: Text to Video

Generate videos from text prompts on the fastest H3 Max Turbo model. The template renders at 768P and 1:1 by default, with the same optional prompt enhancement and 2K upscaling stages as the Max templates. Requires ComfyUI 0.34.4 or later.

![MiniMax H3 Max Turbo Text to Video workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/api_minimax_h3_max_turbo_t2v-1.webp)

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=api_minimax_h3_max_turbo_t2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3-api">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_max_turbo_t2v.json">
    Download JSON or search "MiniMax H3 Turbo: Text to Video" in Template Library
  </Card>
</CardGroup>

### MiniMax H3 First-Last-Frame to Video

Generate a video between a first frame and an optional last frame image through the MiniMax H3 API. The aspect ratio of the output follows the input images.

<video controls className="w-full aspect-video" src="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/output/api_minimax_h3_flf2v.mp4"></video>

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=api_minimax_h3_flf2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3-api">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_flf2v.json">
    Download JSON or search "MiniMax H3 FLF2V" in Template Library
  </Card>
</CardGroup>

<CardGroup cols={2}>
  <Card title="First Frame: angel-warrior-demon-battle-start.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/angel-warrior-demon-battle-start.png">
    First frame for the workflow, or use your own image.
  </Card>
  <Card title="Last Frame: angel-warrior-demon-battle-end.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/angel-warrior-demon-battle-end.png">
    Last frame for the workflow, or use your own image.
  </Card>
</CardGroup>

#### Prompting tips

1. **Frames**: The `first_frame` input is required; `last_frame` is optional. The model generates the motion between them
2. **Aspect ratio**: The output follows the input images, so keep both frames at the same aspect ratio
3. **Image constraints**: Each frame's width and height must be between 256 and 5760 pixels, with an aspect ratio between 2:5 and 5:2
4. **Prompt**: Describe the transition between the frames, and the audio (dialogue, SFX, music) you want

#### Prompt writing guide

MiniMax publishes an official [video prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) for the base generation modes: T2VA, I2VA, FL2VA, and L2VA. The guide explains how to structure a prompt into timed shots with camera movement and audio (dialogue, SFX, music), and includes examples for each mode.

#### MiniMax H3 Max: Image to Video

Animate a still image on the faster H3 Max model. The template feeds a first frame into the First-Last-Frame node (the last frame stays optional) and ships with optional prompt enhancement and 2K upscaling stages.

![MiniMax H3 Max Image to Video workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/api_minimax_h3_max_i2v-1.webp)

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=api_minimax_h3_max_i2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3-api">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_max_i2v.json">
    Download JSON or search "MiniMax H3 Max: Image to Video" in Template Library
  </Card>
</CardGroup>

<CardGroup cols={1}>
  <Card title="Input Image: sneaker_black.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/sneaker_black.png">
    Input image for the workflow, or use your own.
  </Card>
</CardGroup>

#### MiniMax H3 Max Turbo: Image to Video

Animate a still image on the fastest H3 Max Turbo model. The template feeds a first frame into the First-Last-Frame node and ships with optional prompt enhancement and 2K upscaling stages. Requires ComfyUI 0.34.4 or later.

![MiniMax H3 Max Turbo Image to Video workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/api_minimax_h3_max_turbo_i2v-1.webp)

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=api_minimax_h3_max_turbo_i2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3-api">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_max_turbo_i2v.json">
    Download JSON or search "MiniMax H3 Max Turbo: Image to Video" in Template Library
  </Card>
</CardGroup>

### MiniMax H3 Reference to Video

Generate videos conditioned on reference images, videos, and audio through the MiniMax H3 API.

![MiniMax H3 Reference to Video workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/api_minimax_h3_r2v-1.webp)

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=api_minimax_h3_r2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3-api">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_r2v.json">
    Download JSON or search "MiniMax H3 R2V" in Template Library
  </Card>
</CardGroup>

<CardGroup cols={1}>
  <Card title="Reference Image: 9panel_storyboard_golden_hour_clay_court.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/9panel_storyboard_golden_hour_clay_court.png">
    Storyboard reference for the workflow, or use your own images.
  </Card>
</CardGroup>

#### Prompting tips

1. **Reference by tag**: Refer to each input by its order in the prompt, for example `Image 1`, `Video 1`, or `Audio 1`
2. **Limits**: Up to 9 reference images, 3 reference videos, and 3 reference audio clips. Each video or audio clip must be 2-15 seconds; total reference video and audio is capped at 15 seconds each
3. **Audio needs a visual anchor**: Reference audio cannot be used without at least one reference image or video
4. **Video requirements**: Reference videos must be 23.976-60 FPS
5. **Prompt**: Assign each reference a job (identity, style, motion, voice) and describe the target shot; explicit assignments tend to work much better

#### Prompt writing guide

MiniMax publishes an official [full-reference mode prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md) for reference-driven generation (R2V). The guide covers the rewrite output structure, including subject definitions, reference labels, and retention analysis, and how to assign each reference a role in the target shot.

#### MiniMax H3 Max: First-Last-Frame to Video

Generate a video between a first and last frame image on the faster H3 Max model, with optional prompt enhancement and 2K upscaling stages.

![MiniMax H3 Max First-Last-Frame to Video workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/api_minimax_h3_max_flf2v-1.webp)

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=api_minimax_h3_max_flf2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3-api">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_max_flf2v.json">
    Download JSON or search "MiniMax H3 Max: FLF2V" in Template Library
  </Card>
</CardGroup>

<CardGroup cols={2}>
  <Card title="First Frame: dream_face_first.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/dream_face_first.png">
    First frame for the workflow, or use your own image.
  </Card>
  <Card title="Last Frame: dream_face_last.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/dream_face_last.png">
    Last frame for the workflow, or use your own image.
  </Card>
</CardGroup>

#### MiniMax H3 Max: Reference to Video

Generate videos conditioned on reference media on the faster H3 Max model. The Reference node accepts up to 12 reference files in total on H3 Max, with a `reference_detail` option that trades reference image resolution for cost.

![MiniMax H3 Max Reference to Video workflow preview](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/api_minimax_h3_max_r2v-1.webp)

<CardGroup cols={2}>
  <Card title="Run on Comfy Cloud" icon="cloud" href="https://cloud.comfy.org/?template=api_minimax_h3_max_r2v&utm_source=docs&utm_medium=referral&utm_campaign=minimax-h3-api">
    Open in Comfy Cloud
  </Card>
  <Card title="Download Workflow" icon="download" href="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/api_minimax_h3_max_r2v.json">
    Download JSON or search "MiniMax H3 Max: Reference to Video" in Template Library
  </Card>
</CardGroup>

<CardGroup cols={1}>
  <Card title="Reference Image: burdened_warrior_in_dreamscape.png" icon="image" href="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/burdened_warrior_in_dreamscape.png">
    Reference image for the workflow, or use your own images.
  </Card>
</CardGroup>


---

## FILE: catalog/models/minimax/minimax-h3.mdx (https://github.com/Comfy-Org/docs/blob/HEAD/catalog/models/minimax/minimax-h3.mdx)

---
title: "MiniMax H3 video generation and audio generation model (minimax/minimax-h3): where to run it and pricing"
description: "MiniMax H3 is a video generation and audio generation model from MiniMax. Run it with Comfy Router, ComfyUI locally and Comfy Cloud; templates, files..."
---

{/* GENERATED by .github/scripts/catalog/gen-catalog.mjs. Do not edit; edit catalog-src/ or the generator. */}

Model ID: minimax/minimax-h3
Provider: minimax
Product: router, local, cloud, workflows, nodes
Input: text, image
Output: video, audio
License: open-weights

MiniMax H3 is a video generation and audio generation model from MiniMax. It takes text and image as input and produces video and audio. It has open weights, so you can download the files and run it locally in ComfyUI, or call the hosted version through Comfy Router.

## Where you can run it

| Where | Page | Use it when |
| --- | --- | --- |
| Comfy Router (API) | [MiniMax H3 with Comfy Router](/development/comfy-router/models/minimax/minimax-h3/code) | You call `minimax/minimax-h3` from your own code with `COMFY_API_KEY` |
| ComfyUI (local or Comfy Cloud) | [P-H3-Image in ComfyUI: 8-Step Text to Image and Edit](/tutorials/image/pruna/p-h3-image) | You follow a step-by-step workflow tutorial |
| ComfyUI (local or Comfy Cloud) | [ComfyUI MiniMax H3 Video Generation Guide](/tutorials/video/minimax/minimax-h3) | You follow a step-by-step workflow tutorial |
| ComfyUI (local or Comfy Cloud) | [FastVideo FastH3: ComfyUI Workflow Examples](/tutorials/video/minimax/minimax-h3-fastvideo) | You follow a step-by-step workflow tutorial |
| ComfyUI (local or Comfy Cloud) | [ComfyUI MiniMax H3 Text-to-Video, Image-to-Video, and Reference-to-Video Workflows](/tutorials/video/minimax/minimax-h3-native) | You follow a step-by-step workflow tutorial |
| ComfyUI (local or Comfy Cloud) | [MiniMax H3 Multiframe: Video Continuations and Multi-Shot Scenes](/tutorials/video/minimax/minimax-h3-multiframe) | You follow a step-by-step workflow tutorial |
| ComfyUI (local or Comfy Cloud) | [MiniMax H3 Fun ControlNet Union: Pose, Depth, and Canny Control](/tutorials/video/minimax/minimax-h3-fun-controlnet) | You follow a step-by-step workflow tutorial |
| ComfyUI (local or Comfy Cloud) | [ComfyUI MiniMax H3 Prompt Guide: Official Guides, Tips, and Prompt Embeddings](/tutorials/video/minimax/minimax-h3-prompt-guide) | You follow a step-by-step workflow tutorial |
| ComfyUI (local or Comfy Cloud) | [Pika H3 Flex ComfyUI Workflows](/tutorials/video/pika/pika-h3-flex) | You follow a step-by-step workflow tutorial |
| Partner Nodes (ComfyUI and Comfy Cloud) | [MiniMax H3 API Video Generation in ComfyUI](/tutorials/partner-nodes/minimax/minimax-h3) | You build a workflow with the hosted model |
| Comfy Cloud | [Open `video_minimax_h3_i2v` on Comfy Cloud](https://cloud.comfy.org/?template=video_minimax_h3_i2v) | You want to run a ready-made workflow without installing anything |

## Call it with Comfy Router

- `POST https://api.comfy.org/v2/models/minimax/minimax-h3`. Live request schema: `GET /v2/models/minimax/minimax-h3/openapi.json`.

## Model files for running it locally

Put each file in the matching folder under `ComfyUI/models/`. These come from the workflow templates below (workflow_templates at `3c2199b9`).

| File | Folder | Download |
| --- | --- | --- |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `models/loras/` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) |
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | `models/loras/` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors) |
| `minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` | `models/model_patches/` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors) |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae/` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| `sdpose_wholebody_fp16.safetensors` | `models/checkpoints/` | [Download](https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/sdpose_wholebody_fp16.safetensors) |
| `fasth3_omniref_8step_pruned_int8_convrot.safetensors` | `models/diffusion_models/` | [Download](https://huggingface.co/FastVideo/FastH3-OmniRef-Comfy/resolve/main/diffusion_models/fasth3_omniref_8step_pruned_int8_convrot.safetensors) |
| `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` | `models/diffusion_models/` | [Download](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors) |
| `rt_detr_v4-x-hgnet_fp16.safetensors` | `models/diffusion_models/` | [Download](https://huggingface.co/Comfy-Org/SDPose/resolve/main/diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |

## Workflow templates

| Template | Title | Added |
| --- | --- | --- |
| `image_minimax_h3_t2i` | MiniMax H3: Text to Image | 2026-10-27 |
| `video_fasth3_ominiref_r2v` | FastH3 OmniRef: Reference to Video | 2026-10-20 |
| `video_fastvideo_fasth3_i2v` | FastVideo FastH3: Image to Video | 2026-09-15 |
| `video_fastvideo_fasth3_t2v` | FastVideo FastH3: Text to Video | 2026-09-15 |
| `video_minimax_h3_fun_controlnet_union` | MiniMax H3 Fun ControlNet Union | 2026-09-07 |
| `video_minimax_h3_multiframe_reference` | MiniMax H3: Multiframe Reference | 2026-09-04 |
| `api_minimax_h3_max_turbo_i2v` | MiniMax H3 Max Turbo: Image to Video | 2026-09-04 |
| `api_minimax_h3_max_turbo_t2v` | MiniMax H3 Turbo: Text to Video | 2026-09-04 |
| `api_minimax_h3_max_r2v` | MiniMax H3 Max: Reference to Video | 2026-09-02 |
| `api_minimax_h3_max_i2v` | MiniMax H3 Max: Image to Video | 2026-09-01 |
| `api_minimax_h3_max_t2v` | MiniMax H3 Max: Text to Video | 2026-09-01 |
| `api_minimax_h3_max_flf2v` | MiniMax H3 Max: FLF2V | 2026-09-01 |
| `video_minimax_h3_i2v_continuation` | Image to Video | 2026-08-24 |
| `video_minimax_h3_i2v` | MiniMax H3: Image to Video | 2026-08-02 |
| `video_minimax_h3_r2v` | MiniMax H3: Reference to Video | 2026-08-02 |

4 more templates use this model.

## Nodes

- [MiniMaxH3AddGuide](/built-in-nodes/MiniMaxH3AddGuide)
- [MiniMaxH3FunControlNetApply](/built-in-nodes/MiniMaxH3FunControlNetApply)
- [MiniMaxH3ImageToVideo](/built-in-nodes/MiniMaxH3ImageToVideo)
- [MiniMaxH3ReferenceToVideo](/built-in-nodes/MiniMaxH3ReferenceToVideo)
- [MiniMaxH3SigmaShift](/built-in-nodes/MiniMaxH3SigmaShift)
- [MinimaxHailuo03ContextIRNode](/built-in-nodes/MinimaxHailuo03ContextIRNode)
- [MinimaxHailuo03FirstLastFrameNode](/built-in-nodes/MinimaxHailuo03FirstLastFrameNode)
- [MinimaxHailuo03ReferenceNode](/built-in-nodes/MinimaxHailuo03ReferenceNode)
- [MinimaxHailuo03RegenerateNode](/built-in-nodes/MinimaxHailuo03RegenerateNode)
- [MinimaxHailuo03TextToVideoNode](/built-in-nodes/MinimaxHailuo03TextToVideoNode)

## Pricing

Comfy Router charges the model maker's list price, billed from your workspace credits (211 credits = $1 USD). Current per-model prices are on [router.comfy.org](https://router.comfy.org).

- Price a call before you run it: `POST https://api.comfy.org/v2/models/minimax/minimax-h3/estimate`.
- Billing facts, such as whether a refused request is charged: the `billing` field from `GET https://api.comfy.org/v2/models/minimax/minimax-h3`.

See [Comfy Router pricing](/development/comfy-router/pricing) and [Billing](/development/comfy-router/billing#model-billing-facts).
