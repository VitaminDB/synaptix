# Comfy-Org__workflow_templates__minimax-h3-templates

Source: https://github.com/Comfy-Org/workflow_templates/tree/main/templates
Fetched: 2026-11-28

Extraction method: for each template JSON, every node (including nodes inside subgraph definitions) is listed as `type [title]: widgets_values` (the node's saved parameter values, in widget order). Note/MarkdownNote texts and long prompt strings are then repeated unescaped below each template. Raw JSON omitted (very large); see URLs.


---

## TEMPLATE: video_minimax_h3_t2v (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)

```
SaveVideo []: ["video/MiniMax_H3","auto","auto"]
ResolutionSelector []: ["16:9 (Widescreen)",0.4,32]
MarkdownNote [Note: MiniMax H3]: ["## MiniMax H3\n\n[MiniMax H3](https://www.minimax.io/blog/minimax-h3) is MiniMax's general-purpose, omni-modal generation model. It jointly understands text, image, video, and audio, and generates video with **native stereo audio**: voice, sound effects, and music are modeled jointly in a single forward pass, not layered on afterward. Output is up to 2K resolution, 24fps, and up to about 15 seconds.\n\n## ComfyUI links\n- [ComfyUI#15224](https://github.com/Comf
MarkdownNote [Note: Model Links]: ["Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)\n\n## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n\n### Alternative Repos\n\n- [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)\n\n**vae**\n\n- [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.
MarkdownNote [Note: Size Settings Reference]: ["| megapixels | Aspect | Output (multiple=32) |\n|---|---|---|\n| 0.2 | 16:9 | 608 x 352 |\n| 0.3 | 16:9 | 736 x 416 |\n| 0.4 | 16:9 | 864 x 480 |\n| 0.5 | 16:9 | 960 x 544 |\n| 0.6 | 16:9 | 1056 x 608 |\n| 0.7 | 16:9 | 1152 x 640 |\n| 0.8 | 16:9 | 1216 x 672 |\n| 0.9 | 16:9 | 1280 x 736 |\n| 0.98 | 16:9 | 1344 x 768 (official 768p) |\n"]
79dd8a95-ce9d-4c14-b264-2162e8bec5ce []: ["Realistic live-action cinematic look, action movie trailer: practical film photography style, a post-rain dusk metropolis, anamorphic lens, shallow depth of field, film grain, city volumetric fog, flying-car traffic between the towers, restrained grading for a premium feel, powerful natural movement.\n\nScene overview: at dusk on a cluster of skyscrapers, the protagonist is being chased, sprinting and leaping across rooftops, jumping from one building's
VAELoader []: ["minimax_h3_video_vae_int8_convrot.safetensors"]
VAELoader []: ["minimax_h3_audio_vae_fp32.safetensors"]
KSamplerSelect []: ["res_multistep"]
BasicScheduler []: ["simple",4,1]
UNETLoader []: ["minimax_h3_fl2va_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
RandomNoise []: [1,"fixed"]
CreateVideo []: [24,8]
MiniMaxH3ImageToVideo []: ["Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.\n\nTimeline:\n[0s-1s] VHS static opens the frame, the title \"COMFYUI\" appears with RGB split and a slight horizontal jitter.\n[1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.\n[2.5s-4s] Clean \"STARRING\" credits appear,
ComfyMathExpression []: ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"]
PrimitiveFloat [Float (duration)]: [2]
LoraLoaderModelOnly []: ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",1]
ComfySwitchNode [If/Else Switch (Model)]: [false]
ComfySwitchNode [If/Else Switch (Steps)]: [false]
PrimitiveInt []: [20,"fixed"]
PrimitiveInt []: [6,"fixed"]
PrimitiveBoolean [Boolean (Enable Lightning LoRA)]: [false]
```

### Notes and long text widgets (unescaped)

    ## MiniMax H3
    
    [MiniMax H3](https://www.minimax.io/blog/minimax-h3) is MiniMax's general-purpose, omni-modal generation model. It jointly understands text, image, video, and audio, and generates video with **native stereo audio**: voice, sound effects, and music are modeled jointly in a single forward pass, not layered on afterward. Output is up to 2K resolution, 24fps, and up to about 15 seconds.
    ## ComfyUI links
    - [ComfyUI#15224](https://github.com/Comfy-Org/ComfyUI/pull/15224)
    - [🤗 Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ## About this workflow
    **Key inputs**
    - **prompt**: describe the shots, camera moves, and the accompanying audio (dialogue, SFX, music) in one block
    - **width / height**: set via Resolution Selector. H3's native canvas is a 768px short edge, capped at 768x1344 pixels, rounded to a multiple of 32
    - **duration (seconds)**: converted to a valid frame `length` by the Math Expression node, snapping up to the model's 17-frame-per-block (17k+5) grid at 24fps
    Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    **vae**
    - [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.62 GB)
    - [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) (577.2 MB)
    **diffusion_models**
    - [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) (19.53 GB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (14.61 GB)
    **loras**
    - 8 steps:  [minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) (1.82 GB)
    -  4 steps: [minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors) (1.82 GB)
    **embeddings**
    - [10 style embeddings](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/embeddings) (`minimaxh3_*.safetensors`), for example `minimaxh3_art_is_explosion`
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 vae/
    │   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
    │   │   └── minimax_h3_audio_vae_fp32.safetensors
    │   ├── 📂 diffusion_models/
    │   │   └── minimax_h3_fl2va_pruned_int8_convrot.safetensors
    │   ├── 📂 text_encoders/
    │   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    │   ├── 📂 loras/
    │   │   └── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
    │   └── 📂 embeddings/
    │       └── minimaxh3_art_is_explosion.safetensors
    ## Using Prompt Embeddings
    Prompt embeddings (supported since [ComfyUI#15697](https://github.com/Comfy-Org/ComfyUI/pull/15697)) add reusable style tokens to your prompt. Download one or more embeddings from the folder above into `ComfyUI/models/embeddings/`, then reference one in the prompt text with the `embedding:` prefix and the filename without its extension, for example:
    `embedding:minimaxh3_art_is_explosion`
    Using an embedding is optional and the template runs fine without one.
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    | megapixels | Aspect | Output (multiple=32) |
    |---|---|---|
    | 0.2 | 16:9 | 608 x 352 |
    | 0.3 | 16:9 | 736 x 416 |
    | 0.4 | 16:9 | 864 x 480 |
    | 0.5 | 16:9 | 960 x 544 |
    | 0.6 | 16:9 | 1056 x 608 |
    | 0.7 | 16:9 | 1152 x 640 |
    | 0.8 | 16:9 | 1216 x 672 |
    | 0.9 | 16:9 | 1280 x 736 |
    | 0.98 | 16:9 | 1344 x 768 (official 768p) |
    Realistic live-action cinematic look, action movie trailer: practical film photography style, a post-rain dusk metropolis, anamorphic lens, shallow depth of field, film grain, city volumetric fog, flying-car traffic between the towers, restrained grading for a premium feel, powerful natural movement.
    Scene overview: at dusk on a cluster of skyscrapers, the protagonist is being chased, sprinting and leaping across rooftops, jumping from one building's roof to the next with pursuers closing in behind. This is the escape sequence of an action movie trailer: every leap is life-or-death, thrilling and fluid.
    Storyboard (each shot a separate scene, rapid cuts, all landing on the musical beats):
    [0s-1.5s] Shot 1: high side angle: the protagonist sprinting at the roof edge, pursuers appearing in the rooftop doorway behind him, wind catching his coat.
    [1s-2.5s] Shot 2: the protagonist leaps across the gap between buildings, body stretching mid-air, towers and flying-car light trails behind him, a slight slow-motion feel.
    [2.5s-4s] Shot 3: he lands, rolls and rises, low-angle shot, tower shadows and fog behind him, he keeps running.
    [4s-5s] Shot 4: freeze: the instant he hits the edge of the next roof and launches into the jump, silhouette, holding.
    Camera: each shot its own angle, cuts clean and hard, no dissolves, a slight frame jitter on the jumps.
    Audio: wind, rapid footsteps, city ambience, low score underneath, an accent hit on each leap, the score bursting at 4s, closing the last 1s.
    No text, subtitles, logos or watermarks of any kind, no animation or cartoon rendering, no overly-CG look, keep the live-action texture.
    Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.
    Timeline:
    [0s-1s] VHS static opens the frame, the title "COMFYUI" appears with RGB split and a slight horizontal jitter.
    [1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.
    [2.5s-4s] Clean "STARRING" credits appear, "LATENT" and "CONTROLNET" each shown exactly once.
    [4s-5s] Final card "DIRECTED BY COMFYUI" holds, one VHS tracking glitch settling into stability.
    Hard cuts only, transitions landing with tape jumps, no push-ins, no dissolves.
    Audio: lo-fi vaporwave score, slow drum machine with soft bass, VHS tape-noise sample joins at 2.5s, melody fading for the last 1s.
    All text must be clearly legible, do not misspell English, no Chinese characters, do not repeat names or job titles, no soft dissolves, no subtitle bars.


---

## TEMPLATE: video_minimax_h3_i2v (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json)

```
SaveVideo []: ["video/MiniMax_H3","auto","auto"]
LoadImage []: ["transparent_rgb_gaming_mouse.png","image"]
ResolutionSelector []: ["1:1 (Square)",0.4,32]
4c314f31-ecda-4b08-ae98-faaba1bf613f []: ["Editorial tech product film. The transparent gaming mouse from <Picture 1> in its original scene: a pitch-black studio void with a dark, subtle reflective surface, lit by dramatic duotone vibrant blue and warm neon orange rim lighting, deep soft shadow falloff into pure black. Monochromatic dark palette with electric blue and amber accents. Material motif: glowing internal metallic micro-components and glossy acrylic refractions. The environment is cons
MarkdownNote [Note: MiniMax H3]: ["## MiniMax H3\n\n[MiniMax H3](https://www.minimax.io/blog/minimax-h3) is MiniMax's general-purpose, omni-modal generation model. It jointly understands text, image, video, and audio, and generates video with **native stereo audio**: voice, sound effects, and music are modeled jointly in a single forward pass, not layered on afterward. Output is up to 2K resolution, 24fps, and up to about 15 seconds.\n\n## About this workflow\n\nThis template runs the **Image to
MarkdownNote [Note: Model Links]: ["Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)\n\n## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n\n### Alternative Repos\n\n- [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)\n\n**Input Assets**\n- [transparent_rgb_gaming_mouse.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/transparent_rgb_gaming_mouse.png
MarkdownNote [Note: Size Settings Reference]: ["| megapixels | Aspect | Output (multiple=32) |\n|---|---|---|\n| 0.2 | 16:9 | 608 x 352 |\n| 0.3 | 16:9 | 736 x 416 |\n| 0.4 | 16:9 | 864 x 480 |\n| 0.5 | 16:9 | 960 x 544 |\n| 0.6 | 16:9 | 1056 x 608 |\n| 0.7 | 16:9 | 1152 x 640 |\n| 0.8 | 16:9 | 1216 x 672 |\n| 0.9 | 16:9 | 1280 x 736 |\n| 0.98 | 16:9 | 1344 x 768 (official 768p) |"]
ImageScaleToTotalPixels []: ["nearest-exact",0.9,32]
VAELoader []: ["minimax_h3_video_vae_int8_convrot.safetensors"]
VAELoader []: ["minimax_h3_audio_vae_fp32.safetensors"]
KSamplerSelect []: ["res_multistep"]
BasicScheduler []: ["simple",4,1]
UNETLoader []: ["minimax_h3_fl2va_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
RandomNoise []: [1,"fixed"]
CreateVideo []: [24,8]
MiniMaxH3ImageToVideo []: ["Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.\n\nTimeline:\n[0s-1s] VHS static opens the frame, the title \"COMFYUI\" appears with RGB split and a slight horizontal jitter.\n[1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.\n[2.5s-4s] Clean \"STARRING\" credits appear,
ComfyMathExpression []: ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"]
PrimitiveFloat [Float (duration)]: [2]
LoraLoaderModelOnly []: ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",1]
ComfySwitchNode [If/Else Switch (Model)]: [false]
ComfySwitchNode [If/Else Switch (Steps)]: [false]
PrimitiveInt []: [20,"fixed"]
PrimitiveInt []: [6,"fixed"]
PrimitiveBoolean [Boolean (Enable Lightning LoRA)]: [false]
```

### Notes and long text widgets (unescaped)

    Editorial tech product film. The transparent gaming mouse from <Picture 1> in its original scene: a pitch-black studio void with a dark, subtle reflective surface, lit by dramatic duotone vibrant blue and warm neon orange rim lighting, deep soft shadow falloff into pure black. Monochromatic dark palette with electric blue and amber accents. Material motif: glowing internal metallic micro-components and glossy acrylic refractions. The environment is constant throughout.
    SHOT 1: The scene opens exactly on image 1, the mouse resting confidently on the dark surface; the blue and orange lights slowly pulse brighter, refracting deeply through the transparent acrylic shell as the camera executes a slow, deliberate push-in to reveal the intricate circuitry.
    SHOT 2: Cut to an extreme macro profile of the ridged scroll wheel and layered internal micro-components; the camera glides slowly along the side as a sharp beam of warm orange light sweeps across the metallic textures, contrasting perfectly against the deep blue ambient glow.
    SHOT 3: Cut to a low-angle beauty shot: the mouse levitates weightlessly a few centimeters above the dark reflective surface, rotating in a slow, precise orbit; the duotone lighting flares gently along the glassy transparent edges before fading slowly into a sleek silhouette.
    Audio: deep pulsing sub-bass room tone, sharp tactile mechanical clicks, a sweeping glassy whoosh on cuts, and a rising electronic swell that resolves to near-silence on the final fade. 
    ## MiniMax H3
    
    [MiniMax H3](https://www.minimax.io/blog/minimax-h3) is MiniMax's general-purpose, omni-modal generation model. It jointly understands text, image, video, and audio, and generates video with **native stereo audio**: voice, sound effects, and music are modeled jointly in a single forward pass, not layered on afterward. Output is up to 2K resolution, 24fps, and up to about 15 seconds.
    ## About this workflow
    This template runs the **Image to Video** task (`MiniMaxH3ImageToVideo` node), which covers both:
    - **t2va** (text-to-video), when no images are connected
    - **fl2va** (first/last-frame image-to-video), when `first_frame` and/or `last_frame` are connected
    **Key inputs**
    - **first_frame / last_frame**: optional keyframes; the model generates the motion between them
    - **prompt**: describe the shots, motion, and the accompanying audio (dialogue, SFX, music) in one block
    - **width / height**: set via Resolution Selector. H3's native canvas is a 768px short edge, capped at 768x1344 pixels, rounded to a multiple of 32
    - **duration (seconds)**: converted to a valid frame `length` by the Math Expression node, snapping up to the model's 17-frame-per-block (17k+5) grid at 24fps
    Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    **Input Assets**
    - [transparent_rgb_gaming_mouse.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/transparent_rgb_gaming_mouse.png) — required workflow input (`input/transparent_rgb_gaming_mouse.png`)
    **vae**
    - [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.62 GB)
    - [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) (577.2 MB)
    **diffusion_models**
    - [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) (19.53 GB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (14.61 GB)
    **loras**
    - 8 steps:  [minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) (1.82 GB)
    -  4 steps: [minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors) (1.82 GB)
    **embeddings**
    - [10 style embeddings](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/embeddings) (`minimaxh3_*.safetensors`), for example `minimaxh3_art_is_explosion`
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 vae/
    │   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
    │   │   └── minimax_h3_audio_vae_fp32.safetensors
    │   ├── 📂 diffusion_models/
    │   │   └── minimax_h3_fl2va_pruned_int8_convrot.safetensors
    │   ├── 📂 text_encoders/
    │   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    │   ├── 📂 loras/
    │   │   └── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
    │   └── 📂 embeddings/
    │       └── minimaxh3_art_is_explosion.safetensors
    ## Using Prompt Embeddings
    Prompt embeddings (supported since [ComfyUI#15697](https://github.com/Comfy-Org/ComfyUI/pull/15697)) add reusable style tokens to your prompt. Download one or more embeddings from the folder above into `ComfyUI/models/embeddings/`, then reference one in the prompt text with the `embedding:` prefix and the filename without its extension, for example:
    `embedding:minimaxh3_art_is_explosion`
    Using an embedding is optional and the template runs fine without one.
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    | megapixels | Aspect | Output (multiple=32) |
    |---|---|---|
    | 0.2 | 16:9 | 608 x 352 |
    | 0.3 | 16:9 | 736 x 416 |
    | 0.4 | 16:9 | 864 x 480 |
    | 0.5 | 16:9 | 960 x 544 |
    | 0.6 | 16:9 | 1056 x 608 |
    | 0.7 | 16:9 | 1152 x 640 |
    | 0.8 | 16:9 | 1216 x 672 |
    | 0.9 | 16:9 | 1280 x 736 |
    | 0.98 | 16:9 | 1344 x 768 (official 768p) |
    Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.
    Timeline:
    [0s-1s] VHS static opens the frame, the title "COMFYUI" appears with RGB split and a slight horizontal jitter.
    [1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.
    [2.5s-4s] Clean "STARRING" credits appear, "LATENT" and "CONTROLNET" each shown exactly once.
    [4s-5s] Final card "DIRECTED BY COMFYUI" holds, one VHS tracking glitch settling into stability.
    Hard cuts only, transitions landing with tape jumps, no push-ins, no dissolves.
    Audio: lo-fi vaporwave score, slow drum machine with soft bass, VHS tape-noise sample joins at 2.5s, melody fading for the last 1s.
    All text must be clearly legible, do not misspell English, no Chinese characters, do not repeat names or job titles, no soft dissolves, no subtitle bars.


---

## TEMPLATE: video_minimax_h3_r2v (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json)

```
SaveVideo []: ["video/MiniMax_H3","auto","auto"]
ResolutionSelector [Resolution Selector (Size)]: ["16:9 (Widescreen)",0.4,32]
MarkdownNote [Note: MiniMax H3]: ["## MiniMax H3\n\n**Input Assets**\n- [red_superboy_on_city_roof.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/red_superboy_on_city_roof.png) — required workflow input (`input/red_superboy_on_city_roof.png`)\n- [mecha_dragon_lightning.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/mecha_dragon_lightning.png) — required workflow input (`input/mecha_dragon_lightning.png`)\n\n[Mi
MarkdownNote [Note: Model Links]: ["## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n\n### Alternative Repos\n\n- [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)\n\n**vae**\n\n- [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.62 GB)\n- [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/
VAELoader []: ["minimax_h3_video_vae_int8_convrot.safetensors"]
VAELoader []: ["minimax_h3_audio_vae_fp32.safetensors"]
KSamplerSelect []: ["res_multistep"]
BasicScheduler []: ["simple",20,1]
UNETLoader []: ["minimax_h3_ref2va_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
RandomNoise []: [261662374822964,"randomize"]
CreateVideo []: [24,8]
ComfyMathExpression []: ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"]
PrimitiveFloat [Float (Duration)]: [5]
MiniMaxH3ReferenceToVideo []: ["",1344,768,124,"match"]
LoadImage []: ["red_superboy_on_city_roof.png","image"]
PrimitiveStringMultiline [Input Text (Prompt)]: ["Bold comic-book ink style, heavy linework, red and blue-black palette, night city. Use <Picture 2> and <Picture 1> as reference frames\nCUT 1: top-down view of the little boy superhero on the rooftop — red cape fluttering in the wind, hands planted on his hips, freckles and a cocky grin as he looks straight up into the camera. The camera slowly descends toward him as he delivers his line — as he speaks, comic-book graphic overlay text word by wor
LoadImage []: ["mecha_dragon_lightning.png","image"]
MarkdownNote [Note: Size Settings Reference]: ["| megapixels | Aspect | Output (multiple=32) |\n|---|---|---|\n| 0.2 | 16:9 | 608 x 352 |\n| 0.3 | 16:9 | 736 x 416 |\n| 0.4 | 16:9 | 864 x 480 |\n| 0.5 | 16:9 | 960 x 544 |\n| 0.6 | 16:9 | 1056 x 608 |\n| 0.7 | 16:9 | 1152 x 640 |\n| 0.8 | 16:9 | 1216 x 672 |\n| 0.9 | 16:9 | 1280 x 736 |\n| 0.98 | 16:9 | 1344 x 768 (official 768p) |"]
ComfySwitchNode [If/Else Switch (model)]: [false]
ComfySwitchNode [If/Else Switch (Steps)]: [false]
PrimitiveInt [Int (Full)]: [20,"fixed"]
PrimitiveInt [Int (Lightning LoRA)]: [4,"fixed"]
LoraLoaderModelOnly []: ["minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",1]
PrimitiveBoolean [Boolean (Enable Lightning LoRA)]: [false]
```

### Notes and long text widgets (unescaped)

    ## MiniMax H3
    
    **Input Assets**
    - [red_superboy_on_city_roof.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/red_superboy_on_city_roof.png) — required workflow input (`input/red_superboy_on_city_roof.png`)
    - [mecha_dragon_lightning.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/mecha_dragon_lightning.png) — required workflow input (`input/mecha_dragon_lightning.png`)
    [MiniMax H3](https://www.minimax.io/blog/minimax-h3) is MiniMax's general-purpose, omni-modal generation model. It jointly understands text, image, video, and audio, and generates video with **native stereo audio**: voice, sound effects, and music are modeled jointly in a single forward pass, not layered on afterward. Output is up to 2K resolution, 24fps, and up to about 15 seconds.
    ## ComfyUI links
    - [ComfyUI#15224](https://github.com/Comfy-Org/ComfyUI/pull/15224)
    - [🤗 Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ## About this workflow
    This template runs the **reference-to-video (ref2va)** task using the `MiniMaxH3ReferenceToVideo` node. It takes any mix of reference images, videos, and standalone audio, and weaves them into the generation to lock in a character's identity, a style, a motion, a camera move, or a voice.
    **Key inputs**
    - **ref_images / ref_videos / ref_video_audios / ref_audios**: up to 9 reference images, 3 reference videos (each may carry its own paired soundtrack), and 3 standalone reference audio clips
    - **prompt**: reference the inputs by tag, in the exact order they were connected, for example `<Picture 1>`, `<Video 1>`, `<Audio 1>`, then describe the target scene, motion, and audio
    - **ref_image_size**: `match` scales references down to the generation's resolution (faster); `max` keeps up to a 2048px short edge for stronger identity fidelity, at the cost of speed since reference tokens ride along every sampling step
    - **width / height**: set via Resolution Selector.
    - **duration (seconds)**: converted to a valid frame `length` by the Math Expression node
    **Sampling and decode**
    - Sampler: `res_multistep`. `beta` or `normal` scheduler tends to outperform `simple` for reference-heavy prompts like this one
    - The sampler's joint audio+video `LATENT` output feeds directly into both `VAEDecode` (video, `minimax_h3_video_vae_int8_convrot`) and `VAEDecodeAudio` (audio, `minimax_h3_audio_vae_fp32`); each decode node automatically pulls its own half out of the packed latent. `CreateVideo` then muxes the two into a single MP4 with synced sound
    - The diffusion model here is `minimax_h3_ref2va_pruned_int8_convrot.safetensors`, a different set of weights from the `fl2va` model used by the t2v/i2v templates
    Ref2va's output is very sensitive to prompt wording; matching the reference tags precisely and being explicit about which reference drives which part of the shot tends to work best.
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    **vae**
    - [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.62 GB)
    - [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) (577.2 MB)
    **diffusion_models**
    - [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) (19.53 GB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (14.61 GB)
    **loras**
    - 4 steps: [minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors) (1.82 GB)
    **embeddings**
    - [10 style embeddings](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/embeddings) (`minimaxh3_*.safetensors`), for example `minimaxh3_art_is_explosion`
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 vae/
    │   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
    │   │   └── minimax_h3_audio_vae_fp32.safetensors
    │   ├── 📂 diffusion_models/
    │   │   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
    │   ├── 📂 text_encoders/
    │   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    │   ├── 📂 loras/
    │   │   └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
    │   └── 📂 embeddings/
    │       └── minimaxh3_art_is_explosion.safetensors
    ## Using Prompt Embeddings
    Prompt embeddings (supported since [ComfyUI#15697](https://github.com/Comfy-Org/ComfyUI/pull/15697)) add reusable style tokens to your prompt. Download one or more embeddings from the folder above into `ComfyUI/models/embeddings/`, then reference one in the prompt text with the `embedding:` prefix and the filename without its extension, for example:
    `embedding:minimaxh3_art_is_explosion`
    Using an embedding is optional and the template runs fine without one.
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    Bold comic-book ink style, heavy linework, red and blue-black palette, night city. Use <Picture 2> and <Picture 1> as reference frames
    CUT 1: top-down view of the little boy superhero on the rooftop — red cape fluttering in the wind, hands planted on his hips, freckles and a cocky grin as he looks straight up into the camera. The camera slowly descends toward him as he delivers his line — as he speaks, comic-book graphic overlay text word by word in sync with his voice: "GET READY TO" - "MEET" — "YOUR" — "MAKER" — huge jagged comic lettering, white with heavy black outlines and red drop shadows, tilted at scrappy angles, until the three words hang stacked in the air above him between his face and the lens.
    TRANSITION: a violent WHIP PAN off the rooftop that SMEARS the floating words away with it, motion-streaked —
    CUT 2: low hero angle on the colossal black mech-kaiju towering over the skyline as it rears back and unleashes a GIANT terrifying ROAR — jaws wide with fangs, red eyes and chest-core flaring blinding bright, blue lightning arcing off its head, the roar's shockwave rippling dust and rattling windows down the buildings, comic-style speed-lines and ink splatter bursting from the impact of the sound. It leans INTO the camera as the roar peaks. Hold on the roar.
    | megapixels | Aspect | Output (multiple=32) |
    |---|---|---|
    | 0.2 | 16:9 | 608 x 352 |
    | 0.3 | 16:9 | 736 x 416 |
    | 0.4 | 16:9 | 864 x 480 |
    | 0.5 | 16:9 | 960 x 544 |
    | 0.6 | 16:9 | 1056 x 608 |
    | 0.7 | 16:9 | 1152 x 640 |
    | 0.8 | 16:9 | 1216 x 672 |
    | 0.9 | 16:9 | 1280 x 736 |
    | 0.98 | 16:9 | 1344 x 768 (official 768p) |


---

## TEMPLATE: video_minimax_h3_i2v_continuation (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v_continuation.json)

```
SaveVideo []: ["video/MiniMax_H3","auto","auto"]
LoadImage []: ["transparent_rgb_gaming_mouse.png","image"]
ResolutionSelector []: ["1:1 (Square)",0.4,32]
4c314f31-ecda-4b08-ae98-faaba1bf613f []: ["",1344,768,5,757358688076805,"minimax_h3_fl2va_pruned_int8_convrot.safetensors","qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax_h3_video_vae_int8_convrot.safetensors","minimax_h3_audio_vae_fp32.safetensors",false,"minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",1,8]
MarkdownNote [Note: Model Links]: ["Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)\n\n**Input Asset**\n- [transparent_rgb_gaming_mouse.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/transparent_rgb_gaming_mouse.png) / or your input\n\n## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n\n### Alternative Repos\n\n- [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy
MarkdownNote [How to use]: ["1. If you are running it locally, please make sure you have downloaded all the models listed under `Note: Model Links`, and that they are all in the correct model folder\n2. Upload the input image in the `Load Image` node\n3. Update the prompt to describe how you want the input image to be animated\n4. Click the `Run` button to run it"]
VAELoader []: ["minimax_h3_video_vae_int8_convrot.safetensors"]
VAELoader []: ["minimax_h3_audio_vae_fp32.safetensors"]
KSamplerSelect []: ["res_multistep"]
BasicScheduler []: ["simple",4,1]
UNETLoader []: ["minimax_h3_fl2va_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
RandomNoise []: [1,"fixed"]
CreateVideo []: [24,8]
MiniMaxH3ImageToVideo []: ["Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.\n\nTimeline:\n[0s-1s] VHS static opens the frame, the title \"COMFYUI\" appears with RGB split and a slight horizontal jitter.\n[1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.\n[2.5s-4s] Clean \"STARRING\" credits appear,
ComfyMathExpression []: ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"]
PrimitiveFloat [Float (duration)]: [2]
LoraLoaderModelOnly []: ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",1]
ComfySwitchNode [If/Else Switch (Model)]: [false]
ComfySwitchNode [If/Else Switch (Steps)]: [false]
PrimitiveInt []: [20,"fixed"]
PrimitiveInt []: [6,"fixed"]
PrimitiveBoolean [Boolean (Enable Lightning LoRA)]: [false]
```

### Notes and long text widgets (unescaped)

    Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)
    
    **Input Asset**
    - [transparent_rgb_gaming_mouse.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/transparent_rgb_gaming_mouse.png) / or your input
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    **vae**
    - [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.62 GB)
    - [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) (577.2 MB)
    **diffusion_models**
    - [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) (19.53 GB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (14.61 GB)
    **loras**
    - 8 steps:  [minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) (1.82 GB)
    -  4 steps: [minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors) (1.82 GB)
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 vae/
    │   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
    │   │   └── minimax_h3_audio_vae_fp32.safetensors
    │   ├── 📂 diffusion_models/
    │   │   └── minimax_h3_fl2va_pruned_int8_convrot.safetensors
    │   ├── 📂 text_encoders/
    │   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    │   └── 📂 loras/
    │       └── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    1. If you are running it locally, please make sure you have downloaded all the models listed under `Note: Model Links`, and that they are all in the correct model folder
    2. Upload the input image in the `Load Image` node
    3. Update the prompt to describe how you want the input image to be animated
    4. Click the `Run` button to run it
    Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.
    Timeline:
    [0s-1s] VHS static opens the frame, the title "COMFYUI" appears with RGB split and a slight horizontal jitter.
    [1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.
    [2.5s-4s] Clean "STARRING" credits appear, "LATENT" and "CONTROLNET" each shown exactly once.
    [4s-5s] Final card "DIRECTED BY COMFYUI" holds, one VHS tracking glitch settling into stability.
    Hard cuts only, transitions landing with tape jumps, no push-ins, no dissolves.
    Audio: lo-fi vaporwave score, slow drum machine with soft bass, VHS tape-noise sample joins at 2.5s, melody fading for the last 1s.
    All text must be clearly legible, do not misspell English, no Chinese characters, do not repeat names or job titles, no soft dissolves, no subtitle bars.


---

## TEMPLATE: video_minimax_h3_multiframe_reference (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_multiframe_reference.json)

```
SaveVideo []: ["video/MiniMax_H3","auto","auto","auto"]
ResolutionSelector [Resolution Selector (Size)]: ["16:9 (Widescreen)",0.4,32]
MarkdownNote [Note: MiniMax H3]: ["## MiniMax H3\n\n[MiniMax H3](https://www.minimax.io/blog/minimax-h3) is MiniMax's general-purpose, omni-modal generation model. It jointly understands text, image, video, and audio, and generates video with **native stereo audio**: voice, sound effects, and music are modeled jointly in a single forward pass, not layered on afterward. Output is up to 2K resolution, 24fps, and up to about 15 seconds.\n\n## ComfyUI links\n- [ComfyUI#15224](https://github.com/Comf
MarkdownNote [Note: Model Links]: ["## Input Assets\n- [h3_frame_ref_1.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_frame_ref_1.png)\n- [h3_frame_ref_2.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_frame_ref_2.png)\n- [h3_frame_ref_3.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_frame_ref_3.png)\n- [h3_frame_ref_4.png](https://raw.githubusercontent.com/Com
VAELoader []: ["minimax_h3_video_vae_int8_convrot.safetensors"]
VAELoader []: ["minimax_h3_audio_vae_fp32.safetensors"]
KSamplerSelect []: ["res_multistep"]
BasicScheduler []: ["simple",20,1]
UNETLoader []: ["minimax_h3_ref2va_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
RandomNoise []: [148096032077131,"randomize"]
CreateVideo []: [24,"auto","sRGB"]
ComfyMathExpression []: ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"]
PrimitiveFloat [Float (Duration)]: [5]
MiniMaxH3ReferenceToVideo []: ["",1344,768,124,"match"]
PrimitiveStringMultiline [Input Text (Prompt)]: ["subject_definitions:\n<Subject 1> is the woman in <Picture 1>, featuring pale skin, bright red lipstick, wearing a white textured swim cap covered in raised bumps and a blue sleeveless collared shirt with white polka dots.\n<Picture 1> is the first frame of [Shot 1], showing <Subject 1> holding a green apple with a leaf over her right eye against a solid blue background.\n<Picture 2> is the keyframe of [Shot 2], showing the woman wearing a smooth
MarkdownNote [Note: Video Settings]: ["## Resolution Selector\nDelete the **Resolution Selector** node if you want to customize video size manually\n| megapixels | Aspect | Output (multiple=32) |\n|---|---|---|\n| 0.2 | 16:9 | 608 x 352 |\n| 0.3 | 16:9 | 736 x 416 |\n| 0.4 | 16:9 | 864 x 480 |\n| 0.5 | 16:9 | 960 x 544 |\n| 0.6 | 16:9 | 1056 x 608 |\n| 0.7 | 16:9 | 1152 x 640 |\n| 0.8 | 16:9 | 1216 x 672 |\n| 0.9 | 16:9 | 1280 x 736 |\n| 0.98 | 16:9 | 1344 x 768 (official 768p) |\n\n---\n\n## Li
ComfySwitchNode [If/Else Switch (model)]: [false]
ComfySwitchNode [If/Else Switch (Steps)]: [false]
PrimitiveInt [Int (Full)]: [20,"fixed"]
PrimitiveInt [Int (Lightning LoRA)]: [4,"fixed"]
LoraLoaderModelOnly []: ["minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",1]
PrimitiveBoolean [Boolean (Enable Lightning LoRA)]: [false]
MiniMaxH3AddGuide []: [0]
ComfyMathExpression []: ["round (a * 24)"]
PrimitiveFloat [Frame Index (Second)]: [1.5]
LoadImage []: ["h3_frame_ref_2.png","image"]
LoadImage []: ["h3_frame_ref_3.png","image"]
PrimitiveFloat [Frame Index (Second)]: [3]
LoadImage []: ["h3_frame_ref_1.png","image"]
LoadImage []: ["h3_frame_ref_4.png","image"]
PrimitiveFloat [Frame Index (Second)]: [5]
MarkdownNote [Note]: ["Follow the same pattern to add more video frame guidance"]
```

### Notes and long text widgets (unescaped)

    ## MiniMax H3
    
    [MiniMax H3](https://www.minimax.io/blog/minimax-h3) is MiniMax's general-purpose, omni-modal generation model. It jointly understands text, image, video, and audio, and generates video with **native stereo audio**: voice, sound effects, and music are modeled jointly in a single forward pass, not layered on afterward. Output is up to 2K resolution, 24fps, and up to about 15 seconds.
    ## ComfyUI links
    - [ComfyUI#15224](https://github.com/Comfy-Org/ComfyUI/pull/15224)
    - [Add Guide for MiniMax H3](https://github.com/Comfy-Org/ComfyUI/pull/15439)
    - [🤗 Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ## About this workflow
    This template runs the **reference-to-video (ref2va)** task using the **MiniMax H3 Reference to Video** node. It takes any mix of reference images, videos, and standalone audio, and weaves them into the generation to lock in a character's identity, a style, a motion, a camera move, or a voice.
    On top of that, chained **Add Guide for MiniMax H3** nodes pin extra stills (or short clips, and optional audio) onto specific frames of the *output* timeline. The model can address guides anywhere on a continuous time axis, not only the first and last frame. That is the multi-frame / keyframe path in this graph.
    **Key inputs**
    - **ref_images / ref_videos / ref_video_audios / ref_audios** on **MiniMax H3 Reference to Video**: up to 9 reference images, 3 reference videos (each may carry its own paired soundtrack), and 3 standalone reference audio clips. These are semantic references. Name them in connection order as `<Picture 1>`, `<Video 1>`, `<Audio 1>`. They do not sit on a timeline by themselves.
    - **prompt** (**Input Text (Prompt)**): reference the inputs by tag, in the exact order they were connected, then describe the target scene, motion, and audio. In this template, also name each later picture as a shot's first frame or keyframe, and cut to it at the matching timestamp in `detailed_description`.
    - **ref_image_size**: `match` scales references down to the generation's resolution (faster); `max` keeps up to a 2048px short edge for stronger identity fidelity, at the cost of speed since reference tokens ride along every sampling step
    - **width / height**: set via **Resolution Selector (Size)**
    - **duration (seconds)** (**Float (Duration)**): converted to a valid frame `length` by the **Math Expression** node
    **Multi-frame guides (Add Guide for MiniMax H3)**
    Each **Add Guide for MiniMax H3** anchors a guide at `frame_idx` (negative values count from the end of the video):
    - a still image
    - a clip (image batches snap down to the model's valid clip lengths: 5, 22, 39... frames; batches shorter than 5 frames use only the first image)
    - audio, starting at the same position
    - any combination. Chain several nodes (`positive` in to `positive` out) for several guides
    `frame_idx` is in frames at 24fps. This template sets it from **Frame Index (Second)** through **Math Expression** `round(a * 24)` (1.5s → frame 36). The index plus the guide length must stay inside the generated `length`.
    This graph wires four pictures as follows:
    - **Load Image** `h3_frame_ref_1.png` → first `ref_images` input on **MiniMax H3 Reference to Video**. This is `<Picture 1>`, the identity lock and the first frame of Shot 1. There is no Add Guide at 0s; the prompt tells the model the video starts from `<Picture 1>`.
    - **Load Image** `h3_frame_ref_2.png` → first **Add Guide for MiniMax H3** at **1.5s** (Shot 2 keyframe)
    - **Load Image** `h3_frame_ref_3.png` → second Add Guide at **3.0s** (Shot 3 keyframe)
    - **Load Image** `h3_frame_ref_4.png` → third Add Guide at **5.0s** (Shot 4 first frame)
    To add more keyframes, copy the same block: **Load Image** + **Frame Index (Second)** + **Math Expression** + **Add Guide for MiniMax H3**. Keep `positive` in series (last guide → **Basic Guider**). Reuse the same video VAE, audio VAE, and `LATENT` from **MiniMax H3 Reference to Video**. Leave `audio` empty unless you also want a soundtrack anchored at that time.
    `<Picture N>` tags only bind to images actually connected on `ref_images`, in that order. Pictures 2-4 here are timeline anchors only. Also plug them into extra `ref_images` slots if you want the text encoder to see them as `<Picture 2>` / `<Picture 3>` / `<Picture 4>`.
    Other useful patterns from the same node: feed the first 22 frames of an existing video plus its audio into one Add Guide at `frame_idx` 0 to continue both streams, or drop a still at a mid-clip frame (for example frame 60 of 124) to force the shot to hit that frame.
    ## Input Assets
    - [h3_frame_ref_1.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_frame_ref_1.png)
    - [h3_frame_ref_2.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_frame_ref_2.png)
    - [h3_frame_ref_3.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_frame_ref_3.png)
    - [h3_frame_ref_4.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_frame_ref_4.png)
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    **vae**
    - [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.62 GB)
    - [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) (577.2 MB)
    **diffusion_models**
    - [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) (19.53 GB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (14.61 GB)
    **loras**
    - 4 steps: [minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors) (1.82 GB)
    **embeddings**
    - [10 style embeddings](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/embeddings) (`minimaxh3_*.safetensors`), for example `minimaxh3_art_is_explosion`
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 vae/
    │   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
    │   │   └── minimax_h3_audio_vae_fp32.safetensors
    │   ├── 📂 diffusion_models/
    │   │   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
    │   ├── 📂 text_encoders/
    │   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    │   ├── 📂 loras/
    │   │   └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
    │   └── 📂 embeddings/
    │       └── minimaxh3_art_is_explosion.safetensors
    ## Using Prompt Embeddings
    Prompt embeddings (supported since [ComfyUI#15697](https://github.com/Comfy-Org/ComfyUI/pull/15697)) add reusable style tokens to your prompt. Download one or more embeddings from the folder above into `ComfyUI/models/embeddings/`, then reference one in the prompt text with the `embedding:` prefix and the filename without its extension, for example:
    `embedding:minimaxh3_art_is_explosion`
    Using an embedding is optional and the template runs fine without one.
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    subject_definitions:
    <Subject 1> is the woman in <Picture 1>, featuring pale skin, bright red lipstick, wearing a white textured swim cap covered in raised bumps and a blue sleeveless collared shirt with white polka dots.
    <Picture 1> is the first frame of [Shot 1], showing <Subject 1> holding a green apple with a leaf over her right eye against a solid blue background.
    <Picture 2> is the keyframe of [Shot 2], showing the woman wearing a smooth teal green swim cap and a red swimsuit top, resting her chin on a red ball stacked on a soccer ball, while holding a red tennis racket that casts a grid shadow across her face.
    <Picture 3> is the keyframe of [Shot 3], showing the woman in a black geometric textured swim cap and a sleek black swimsuit leaning over a vivid blue table towards a yellow apple with a matchstick stuck in it on a red plate, with a mirror reflecting the arrangement against an ocean background.
    <Picture 4> is the first frame of [Shot 4], showing a blurry, ethereal white humanoid head and neck in profile against a soft blue-gray gradient background.
    summary:
    [keyframe completion + reference generation] The target video is a dynamic fashion photography collection featuring <Subject 1> in various surreal, highly stylized vignettes. It begins with <Picture 1> as <Subject 1> dynamically poses with a green apple against a solid blue background. The video then progresses through rapid stylistic cuts: first matching <Picture 2> where she balances her chin on stacked balls while a racket casts a shifting grid shadow on her face, and then matching <Picture 3> where she leans over a table towards a yellow apple with a matchstick, set against a rippling ocean backdrop. The video concludes by cutting to the abstract, blurry white humanoid profile from <Picture 4>, which slightly vibrates before freezing to end the sequence.
    retention_analysis:
    <Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): partially_preserved - <Subject 1>'s facial identity, pale skin, and distinct red lipstick are maintained across the first three shots, but her swim cap and clothing are explicitly changed in [Shot 2] and [Shot 3] to match the distinct fashion styles of those keyframes.
    <Picture 1> ([Shot 1] first frame): fully_preserved - the video starts exactly from <Picture 1>'s framing, subject state, and solid blue background.
    <Picture 2> ([Shot 2] keyframe): fully_preserved - the specific surreal arrangement of the red ball, soccer ball, tennis racket, and the resulting grid shadow on the face is accurately recreated in [Shot 2].
    <Picture 3> ([Shot 3] keyframe): fully_preserved - the precise composition of the black-clad model, the yellow apple with a matchstick on a red plate, the mirror reflection, and the outdoor ocean background is accurately established in [Shot 3].
    <Picture 4> ([Shot 4] first frame): fully_preserved - the blurry, abstract white humanoid profile is directly used as the static ending composition of [Shot 4].
    detailed_description:
    The target video is a dynamic fashion photography film collection presented in a highly stylized, cinematic live-action medium. The visuals are characterized by bold, contrasting primary colors, crisp lighting, and surreal, editorial compositions.
    [Shot 1] The shot begins from <Picture 1>. <Subject 1> is positioned centrally against a vibrant, solid blue background. She wears a striking white textured swim cap covered in small raised bumps, complementing her blue sleeveless collared shirt adorned with white polka dots. Her pale skin and bright red lipstick stand out vividly. She holds a bright green apple with a single, crisp green leaf directly over her right eye, gripping it lightly with her slender fingers. To fulfill the dynamic fashion intent, the camera pushes in sharply. As the framing tightens, <Subject 1> slowly lowers the green apple away from her face, revealing her intense, unblinking gaze. The crisp studio lighting highlights the textures of her cap and the smooth surface of the apple, establishing a high-fashion, editorial atmosphere.
    [Shot 2] At 00:01.500, the shot cuts to the next fashion vignette, whose keyframe corresponds to <Picture 2>. <Subject 1> is now seen in profile against a two-toned background of solid blue above and a vibrant red surface below. She wears a smooth teal green swim cap and a red swimsuit top. Her pose is sculptural: she rests her chin gracefully atop a small red ball, which is precariously stacked on a classic black-and-white soccer ball. In her hand, she holds a bright red tennis racket vertically near her face. The strong directional sunlight casts a distinct, sharp grid shadow from the racket's strings across her closed eyes and pale skin. The camera executes a smooth, continuous arc shot around her. This movement causes the harsh grid shadow to dynamically glide and warp across her facial contours, while she subtly tilts her head, maintaining her perfectly still, mannequin-like pose amidst the camera motion.
    [Shot 3] At 00:03.000, the shot cuts to an outdoor surreal composition, whose keyframe corresponds to <Picture 3>. The setting features a bright blue ocean and a clear sky in the background. <Subject 1>, now wearing a black textured swim cap featuring geometric triangular patterns and a sleek black swimsuit, leans dramatically forward over a vivid blue table. In front of her, a glossy red plate holds a perfectly smooth yellow apple, which has a single unlit matchstick pierced upright into its top. A round vanity mirror on a silver stand sits just behind the plate, perfectly reflecting the back of the apple, the matchstick, and a sliver of the ocean. The camera tracks slowly to the right as <Subject 1> continues to lean closer to the apple, her lips slightly parted. The natural sunlight creates sharp, defined shadows on the blue table, while the ocean waves in the background gently ripple and shift, adding environmental dynamism to the editorial shot.
    [Shot 4] At 00:05.000, the shot cuts to the concluding sequence, anchored by <Picture 4>. The scene shifts abruptly to a highly abstract, ethereal presentation. A heavily blurred, glowing white humanoid head and neck in profile are silhouetted against a soft blue-gray gradient background. The figure appears ghost-like, devoid of distinct facial features. The camera holds a static frame as the white silhouette exhibits a subtle, vibrating motion, its edges smearing gently into the blue background. After this brief, haunting movement, the figure completely freezes in place, providing a stark, arresting final image that concludes the dynamic fashion showcase.
    overall_soundscape:
    The video features a stylized, rhythmic soundscape with subtle, crisp Foley effects like fabric swishes, ocean ripples, and gentle air movements synchronized to the camera motions, enhancing the high-end editorial aesthetic.
    non_diegetic_music:
    A punchy, dynamic electronic beat with a fast tempo, ticking hi-hats, and heavy bass serves as the driving soundtrack, matching the high-energy, rapid-fire cuts of the fashion vignettes.
    ## Resolution Selector
    Delete the **Resolution Selector** node if you want to customize video size manually
    | megapixels | Aspect | Output (multiple=32) |
    |---|---|---|
    | 0.2 | 16:9 | 608 x 352 |
    | 0.3 | 16:9 | 736 x 416 |
    | 0.4 | 16:9 | 864 x 480 |
    | 0.5 | 16:9 | 960 x 544 |
    | 0.6 | 16:9 | 1056 x 608 |
    | 0.7 | 16:9 | 1152 x 640 |
    | 0.8 | 16:9 | 1216 x 672 |
    | 0.9 | 16:9 | 1280 x 736 |
    | 0.98 | 16:9 | 1344 x 768 (official 768p) |
    ---
    ## Lightning LoRA
    For fast video generation, but scarce video quality
    ## Duration
    Up to 15s


---

## TEMPLATE: video_minimax_h3_fun_controlnet_union (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_fun_controlnet_union.json)

```
ResolutionSelector [Resolution Selector (Size)]: ["16:9 (Widescreen)",0.4,32]
MarkdownNote [Note: MiniMax H3 Fun ControlNet]: ["## MiniMax H3 Fun ControlNet-Union\n\n[MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) is a single ControlNet-Union branch trained with VideoX-Fun. One checkpoint conditions MiniMax H3 on **Canny, Depth, HED, MLSD, or Pose** control video, and also runs **video inpainting**. Load it on top of a MiniMax H3 transformer (`fl2va` or `ref2va`).\n\nPackaged weights: [🤗 Comfy-Org/MiniMax-H3](https://h
MarkdownNote [Note: Model Links]: ["## Input Asset\n\n- [dancer_field_pose.mp4](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/dancer_field_pose.mp4)\n\n## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n- [Hugging Face:Comfy-Org/SDPose](https://huggingface.co/Comfy-Org/SDPose)\n\n### Alternative Repos\n\n- [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)\n- [ModelScope:Co
VAELoader []: ["minimax_h3_video_vae_int8_convrot.safetensors"]
VAELoader []: ["minimax_h3_audio_vae_fp32.safetensors"]
KSamplerSelect []: ["res_multistep"]
BasicScheduler []: ["simple",20,1]
UNETLoader []: ["minimax_h3_ref2va_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
RandomNoise []: [1086599926715797,"randomize"]
CreateVideo []: [24,"auto","sRGB"]
ComfyMathExpression []: ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"]
PrimitiveFloat [Float (Duration)]: [5]
MiniMaxH3ReferenceToVideo []: ["",1344,768,124,"match"]
PrimitiveStringMultiline [Input Text (Prompt)]: ["Scene: Hyperreal desert-storm aesthetic. A humanoid figure made of packed sand dances in a barren wasteland, each move shedding dust, the wind eroding and reshaping its body.\n[0s-2s] Pale desert, a sand humanoid begins to move from stillness, surface grains trickling off and dragging mist-like trails in the wind.\n[2s-4s] Spins and arm swings keep dissolving and rebuilding its edges, raised sand columns glowing gold in hard light under a searing
MarkdownNote [Note: Video Settings]: ["## Resolution Selector\nDelete the **Resolution Selector** node if you want to customize video size manually\n| megapixels | Aspect | Output (multiple=32) |\n|---|---|---|\n| 0.2 | 16:9 | 608 x 352 |\n| 0.3 | 16:9 | 736 x 416 |\n| 0.4 | 16:9 | 864 x 480 |\n| 0.5 | 16:9 | 960 x 544 |\n| 0.6 | 16:9 | 1056 x 608 |\n| 0.7 | 16:9 | 1152 x 640 |\n| 0.8 | 16:9 | 1216 x 672 |\n| 0.9 | 16:9 | 1280 x 736 |\n| 0.98 | 16:9 | 1344 x 768 (official 768p) |\n\n---\n\n## Li
ComfySwitchNode [If/Else Switch (model)]: [false]
ComfySwitchNode [If/Else Switch (Steps)]: [false]
PrimitiveInt [Int (Full)]: [20,"fixed"]
PrimitiveInt [Int (Lightning LoRA)]: [4,"fixed"]
LoraLoaderModelOnly []: ["minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",1]
PrimitiveBoolean [Boolean (Enable Lightning LoRA)]: [false]
MiniMaxH3FunControlNetApply []: [1,0,1]
ModelPatchLoader []: ["minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"]
LoadVideo [Load Video  (Control Video)]: ["dancer_field_pose.mp4","image"]
622cceeb-7ba3-4dd9-b0e8-00066c5222a4 []: [1024,"lanczos",true,true,true,true,4,2,0.51,0.5,"person",2,"sdpose_wholebody_fp16.safetensors","rt_detr_v4-x-hgnet_fp16.safetensors"]
Video Slice []: [0,0,false]
SaveVideo []: ["video/MiniMax_H3","auto","auto","auto"]
MarkdownNote []: ["Fun ControlNet: one checkpoint adds Canny / Depth / HED / MLSD / Pose control video plus inpainting to MiniMax H3.\n\nVideo length: 5-15s (frames snap to the 17n+5 grid at 24fps, e.g. 124 = 5s). Control videos longer than the target are trimmed to the first frames; shorter ones hold their last frame.\n\nWorks with both fl2va and ref2va model files. Keep guidance_scale = 1.0. Raise strength above 1 only if the control drifts."]
MarkdownNote [Note: Use Video length]: ["If you want the final video to match the control video length, you can use the group above and link the `batch_size` output to the `MiniMax H3 Reference to Video` node's length input"]
MarkdownNote [Note:  Preparing the control video]: ["## Preparing the control video with built-in preprocessing\n\nThe Fun ControlNet follows the control video you feed it, so the quality of that video's edges, depth or pose decides the quality of the control. ComfyUI ships with built-in preprocessing nodes, and each control type has an official tutorial:\n\n- **Edges (Canny)**: [Detect Edges (Canny) node](https://docs.comfy.org/built-in-nodes/Canny)\n- **Depth**: [Depth Anything 3 tutorial](htt
SDPoseKeypointExtractor []: [16]
ResizeImageMaskNode []: ["scale longer dimension",1024,"lanczos"]
SDPoseDrawKeypoints []: [true,true,true,true,4,2,0.5,true]
CheckpointLoaderSimple []: ["sdpose_wholebody_fp16.safetensors"]
UNETLoader []: ["rt_detr_v4-x-hgnet_fp16.safetensors","default"]
RTDETR_detect []: [0.5,"person",2]
```

### Notes and long text widgets (unescaped)

    ## MiniMax H3 Fun ControlNet-Union
    
    [MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) is a single ControlNet-Union branch trained with VideoX-Fun. One checkpoint conditions MiniMax H3 on **Canny, Depth, HED, MLSD, or Pose** control video, and also runs **video inpainting**. Load it on top of a MiniMax H3 transformer (`fl2va` or `ref2va`).
    Packaged weights: [🤗 Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (`minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors`).
    ## ComfyUI links
    - [ComfyUI#15224](https://github.com/Comfy-Org/ComfyUI/pull/15224) (MiniMax H3)
    - [ComfyUI#15975](https://github.com/Comfy-Org/ComfyUI/pull/15975) (Fun ControlNet as model patch)
    - [🤗 alibaba-pai/MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union)
    - [🤗 Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ## About this workflow
    This graph is a **Pose control** example only. The Union checkpoint can take Canny / Depth / HED / MLSD or an inpaint mask as well, but those paths are not wired here.
    **What this graph does**
    1. **Load Video (Control Video)** reads `dancer_field_pose.mp4`. Optional **Video Slice** trims the clip.
    2. **SDPose Video Pose Extractor [Multi-Person]** detects people and draws whole-body skeletons (body, hands, face, feet). That skeleton sequence is the control video, not the raw RGB footage.
    3. **Preview Image** shows the pose frames, then feeds them into **Apply MiniMax H3 Fun ControlNet** as `control_video`.
    4. **MiniMax H3 Reference to Video** builds text conditioning and the joint audio+video latent. In this demo the `ref_*` slots are empty, so the run is text-to-video locked to the pose. You can still attach reference images, videos, or audio if you want identity or voice lock on top of the control.
    5. Sampling and decode stay the same as other H3 templates: joint latent → video VAE + audio VAE → muxed MP4.
    **Apply MiniMax H3 Fun ControlNet**
    - **model_patch**: Fun ControlNet-Union from **ModelPatchLoader**
    - **vae**: MiniMax H3 **video** VAE (used to encode the control frames)
    - **control_video**: preprocessed frames. This template uses SDPose skeletons
    - **mask** / **source_video**: leave empty here. `mask` (1 = regenerate) plus optional `source_video` is the inpaint path
    - **strength**: how hard the control skip is added (same role as `control_context_scale` on the original model). `1.0` is strongest and is what this graph uses. Values below `1.0` loosen the pose. `0` turns the branch off
    - **start_percent** / **end_percent**: which part of sampling the control is active (`0` → `1` here)
    **Length and size**
    - Frames snap to the video VAE grid `17n + 5` at 24fps. This template uses **5s → 124 frames**. Cap is about 15 seconds
    - A control clip longer than `length` is taken from the start. A shorter clip holds its last frame
    - To match the control video length, use the bypassed group and wire `batch_size` into **MiniMax H3 Reference to Video** `length`
    **Prompt**
    [Guide here](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
    **Using the other Union modes**
    Keep **Apply MiniMax H3 Fun ControlNet** and swap only the preprocessor that writes `control_video`:
    - **Canny**: Detect Edges
    - **Depth**: Depth Anything 3 or MoGe
    - **Pose**: this graph (SDPose), or SAM3D Body
    - **HED / MLSD**: no built-in preprocessor; use a community pack if you need them
    - **Inpaint**: connect `mask` and optional `source_video`. `control_video` can stay empty or add extra structure
    See the **Preparing the control video** note for official preprocessor tutorials.
    ## Input Asset
    - [dancer_field_pose.mp4](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/dancer_field_pose.mp4)
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    - [Hugging Face:Comfy-Org/SDPose](https://huggingface.co/Comfy-Org/SDPose)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    - [ModelScope:Comfy-Org/SDPose](https://modelscope.cn/models/Comfy-Org/SDPose)
    **vae**
    - [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.62 GB)
    - [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) (577.2 MB)
    **diffusion_models**
    - [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) (19.53 GB)
    - [rt_detr_v4-x-hgnet_fp16.safetensors](https://huggingface.co/Comfy-Org/SDPose/resolve/main/diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors) (118.2 MB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (14.61 GB)
    **loras**
    - [minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors) (1.82 GB)
    **model_patches**
    - [minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors) (2.14 GB)
    **checkpoints**
    - [sdpose_wholebody_fp16.safetensors](https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/sdpose_wholebody_fp16.safetensors) (1.79 GB)
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 vae/
    │   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
    │   │   └── minimax_h3_audio_vae_fp32.safetensors
    │   ├── 📂 diffusion_models/
    │   │   ├── minimax_h3_ref2va_pruned_int8_convrot.safetensors
    │   │   └── rt_detr_v4-x-hgnet_fp16.safetensors
    │   ├── 📂 text_encoders/
    │   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    │   ├── 📂 loras/
    │   │   └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
    │   ├── 📂 model_patches/
    │   │   └── minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors
    │   └── 📂 checkpoints/
    │       └── sdpose_wholebody_fp16.safetensors
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    Scene: Hyperreal desert-storm aesthetic. A humanoid figure made of packed sand dances in a barren wasteland, each move shedding dust, the wind eroding and reshaping its body.
    [0s-2s] Pale desert, a sand humanoid begins to move from stillness, surface grains trickling off and dragging mist-like trails in the wind.
    [2s-4s] Spins and arm swings keep dissolving and rebuilding its edges, raised sand columns glowing gold in hard light under a searing white sky.
    [4s-5s] On the final pose, half the body has already been carried away by wind as a hanging line of sand, not yet fallen.
    Camera: medium fixed shot, focus pull from sand-grain close-up to full body.
    Audio: fine sand-scrape texture throughout, one distant desert wind call at 2s, no music.
    Negative: no dissolve transitions / no text overlays / no human face / no camera shake
    ## Resolution Selector
    Delete the **Resolution Selector** node if you want to customize video size manually
    | megapixels | Aspect | Output (multiple=32) |
    |---|---|---|
    | 0.2 | 16:9 | 608 x 352 |
    | 0.3 | 16:9 | 736 x 416 |
    | 0.4 | 16:9 | 864 x 480 |
    | 0.5 | 16:9 | 960 x 544 |
    | 0.6 | 16:9 | 1056 x 608 |
    | 0.7 | 16:9 | 1152 x 640 |
    | 0.8 | 16:9 | 1216 x 672 |
    | 0.9 | 16:9 | 1280 x 736 |
    | 0.98 | 16:9 | 1344 x 768 (official 768p) |
    ---
    ## Lightning LoRA
    For fast video generation, but scarce video quality
    ## Duration
    Up to 15s
    Fun ControlNet: one checkpoint adds Canny / Depth / HED / MLSD / Pose control video plus inpainting to MiniMax H3.
    Video length: 5-15s (frames snap to the 17n+5 grid at 24fps, e.g. 124 = 5s). Control videos longer than the target are trimmed to the first frames; shorter ones hold their last frame.
    Works with both fl2va and ref2va model files. Keep guidance_scale = 1.0. Raise strength above 1 only if the control drifts.
    ## Preparing the control video with built-in preprocessing
    The Fun ControlNet follows the control video you feed it, so the quality of that video's edges, depth or pose decides the quality of the control. ComfyUI ships with built-in preprocessing nodes, and each control type has an official tutorial:
    - **Edges (Canny)**: [Detect Edges (Canny) node](https://docs.comfy.org/built-in-nodes/Canny)
    - **Depth**: [Depth Anything 3 tutorial](https://docs.comfy.org/tutorials/utility/depth-anything-3) / [MoGe nodes](https://docs.comfy.org/built-in-nodes/MoGeInference)
    - **Pose**: [SAM3D Body tutorial](https://docs.comfy.org/tutorials/utility/sam3d-body) / [SDPose Keypoint Extractor](https://docs.comfy.org/built-in-nodes/SDPoseKeypointExtractor)
    - **All preprocessors overview**: [ComfyUI preprocessor workflows](https://docs.comfy.org/tutorials/utility/preprocessors)
    **Mask / segmentation (for inpainting mode)**:
    - [SAM3 video segmentation tutorial](https://docs.comfy.org/tutorials/utility/video-segment-sam3)
    Note: HED and MLSD have no built-in preprocessing nodes. Canny, Depth and Pose cover most use cases; install a community pack if you need the others.


---

## TEMPLATE: video_fastvideo_fasth3_t2v (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_fastvideo_fasth3_t2v.json)

```
SaveVideo []: ["video/MiniMax_H3","auto","auto","auto"]
4c314f31-ecda-4b08-ae98-faaba1bf613f []: ["integrated_multimodal_description: [Shot 1] Cinematic, a wide-angle shot filmed from an extreme low-angle, worm's-eye perspective. In the extreme foreground of the square frame, a massive tan suede work boot features a thick, chunky brown rubber sole and tightly laced brown shoelaces, rendered slightly out of focus. Above the boot, a young adult man wears loose-fitting, olive green cargo pants, an unbuttoned dark navy blue heavy-cotton overshirt, and a 
MarkdownNote [Note: Fast Video FastH3]: ["## About this workflow\n\nThis template runs the **Text to Video** task (`MiniMaxH3ImageToVideo` node with no images connected) using the [FastVideo FastH3 8-Step V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) checkpoint, a DMD2-distilled model that generates synchronized video and audio in **8 sampling steps** instead of the base model's full schedule.\n\n**Checkpoint repo**: [FastVideo/FastVideo-FastH3-Comfy](https://huggingface.co/Fa
MarkdownNote [Note: Model Links]: ["Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)\n\n## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n\n### Alternative Repos\n\n- [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)\n\n**vae**\n\n- [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.
ResolutionSelector []: ["1:1 (Square)",0.4,32]
VAELoader []: ["minimax_h3_video_vae_int8_convrot.safetensors"]
VAELoader []: ["minimax_h3_audio_vae_fp32.safetensors"]
KSamplerSelect []: ["res_multistep"]
BasicScheduler []: ["simple",8,1]
UNETLoader []: ["fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
RandomNoise []: [1,"randomize"]
CreateVideo []: [24,"auto","sRGB","none"]
MiniMaxH3ImageToVideo []: ["Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.\n\nTimeline:\n[0s-1s] VHS static opens the frame, the title \"COMFYUI\" appears with RGB split and a slight horizontal jitter.\n[1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.\n[2.5s-4s] Clean \"STARRING\" credits appear,
ComfyMathExpression []: ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"]
PrimitiveFloat [Float (duration)]: [2]
BlockSparseAttention []: ["vsa",10,0.2,1,"",12288,256,"exact_kv_and_rows",false]
ModelAttentionBackend []: ["comfy kitchen attention"]
MiniMaxH3SigmaShift []: [10,3]
```

### Notes and long text widgets (unescaped)

    integrated_multimodal_description: [Shot 1] Cinematic, a wide-angle shot filmed from an extreme low-angle, worm's-eye perspective. In the extreme foreground of the square frame, a massive tan suede work boot features a thick, chunky brown rubber sole and tightly laced brown shoelaces, rendered slightly out of focus. Above the boot, a young adult man wears loose-fitting, olive green cargo pants, an unbuttoned dark navy blue heavy-cotton overshirt, and a tan baseball cap. He has a serious expression, looking directly down into the lens. Behind him, a modern skyscraper made of reflective glass panels stretches into an overcast, pale grey sky. Large, bold, white typography reading "STREETWEAR" bursts into the frame, positioned seamlessly between the man's back and the skyscraper, creating a three-dimensional interlaced title sequence effect. The man lowers his raised foot, and the massive tan boot descends rapidly, striking the pavement. He then raises both hands to adjust his cap's brim. The camera pushes in slowly, intensifying the visual scale of the figure and the floating text. [Shot 2] At 00:02.100, the camera cuts to a medium shot from a slightly elevated low-angle vantage point. The man from Shot 1 now stands fully planted on the ground against the sheer glass facade of the skyscraper. Fast-paced, neon-yellow text reading "VOL. 1" flashes rapidly in the extreme foreground, momentarily overlapping his navy jacket. He drops his hands and executes a sharp, forceful turn of his shoulders to the right, striking a dynamic, angular pose. His jacket flares out, revealing a neutral-toned t-shirt underneath. His gaze shifts away from the camera, looking off into the distance. The camera arcs right at a medium speed, tracking his rotation and capturing the shifting reflections of the city on the glass building behind him. [Shot 3] At 00:03.600, the camera cuts to a wide tracking shot positioned just above the pavement level. The man from Shot 1 strides purposefully forward along the grey concrete sidewalk. Clean, white typography reading "AVAILABLE NOW" scrolls vertically down the right side of the frame, weaving behind his swinging right arm to maintain the interlaced text aesthetic. The focus is directed toward the lower half of his body, highlighting the movement of his olive green cargo pants as they wrinkle dynamically. His heavy tan boots stomp rhythmically on the concrete. The camera pulls out at a fast speed, maintaining a constant distance just ahead of his advancing boots while the vertical grid lines of the skyscraper blur in the background.
    overall_soundscape: A faint, continuous urban hum of distant city traffic provides a subtle ambient baseline. A loud, resonant thud is heard prominently in the foreground as the heavy tan boot strikes the concrete pavement, followed by the crisp, audible rustle of thick cotton and denim as the man adjusts his cap. When the shot changes, a sharp, distinct swish of heavy fabric accompanies his sudden shoulder rotation. Finally, a rhythmic series of heavy, pronounced footsteps—thick rubber soles slapping firmly against the concrete—dominates the foreground, accompanied by the continuous, rough rustling of cargo pants.
    non_diegetic_music: Electronic bass music, fast tempo, featuring a heavy, rhythmic drum machine beat, deep synthesizer drones, and sudden aggressive bass drops with no melodic swell.
    ## About this workflow
    
    This template runs the **Text to Video** task (`MiniMaxH3ImageToVideo` node with no images connected) using the [FastVideo FastH3 8-Step V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) checkpoint, a DMD2-distilled model that generates synchronized video and audio in **8 sampling steps** instead of the base model's full schedule.
    **Checkpoint repo**: [FastVideo/FastVideo-FastH3-Comfy](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy)
    **Scope**: this distilled checkpoint supports t2va only. FL2VA (first/last-frame) and Ref2VA (multi-reference conditioning) were not distilled; use the base MiniMax H3 model for image-conditioned tasks.
    **Key inputs**
    - **prompt**: describe the shots, motion, and the accompanying audio (dialogue, SFX, music) in one block
    - **width / height**: set via Resolution Selector. H3's native canvas is a 768px short edge, capped at 768x1344 pixels, rounded to a multiple of 32
    - **duration (seconds)**: converted to a valid frame `length` by the Math Expression node, snapping up to the model's 17-frame-per-block (17k+5) grid at 24fps
    Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    **vae**
    - [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.62 GB)
    - [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) (577.2 MB)
    **diffusion_models**
    - [fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors) (20.61 GB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (14.61 GB)
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 vae/
    │   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
    │   │   └── minimax_h3_audio_vae_fp32.safetensors
    │   ├── 📂 diffusion_models/
    │   │   └── fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors
    │   └── 📂 text_encoders/
    │       └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.
    Timeline:
    [0s-1s] VHS static opens the frame, the title "COMFYUI" appears with RGB split and a slight horizontal jitter.
    [1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.
    [2.5s-4s] Clean "STARRING" credits appear, "LATENT" and "CONTROLNET" each shown exactly once.
    [4s-5s] Final card "DIRECTED BY COMFYUI" holds, one VHS tracking glitch settling into stability.
    Hard cuts only, transitions landing with tape jumps, no push-ins, no dissolves.
    Audio: lo-fi vaporwave score, slow drum machine with soft bass, VHS tape-noise sample joins at 2.5s, melody fading for the last 1s.
    All text must be clearly legible, do not misspell English, no Chinese characters, do not repeat names or job titles, no soft dissolves, no subtitle bars.


---

## TEMPLATE: video_fastvideo_fasth3_i2v (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_fastvideo_fasth3_i2v.json)

```
SaveVideo []: ["video/MiniMax_H3","auto","auto","auto"]
4c314f31-ecda-4b08-ae98-faaba1bf613f []: ["For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\nintegrated_multimodal_description: [Shot 1] This is a live-action, highly stylized fashion video with a stark, high-contrast aesthetic and digital glitch overlays. The figure stands dead center against a plain off-white background that has a subtle, paper-like texture. The outfit is a layered, monochrome streetwear ensemble. The upper body is e
LoadImage []: ["red_line_barrier.png","image"]
MarkdownNote [Note: Fast Video FastH3]: ["\nThis template runs the **Image to Video** task (`MiniMaxH3ImageToVideo` node) with the [FastVideo FastH3 8-Step V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) checkpoint, a DMD2-distilled model that generates synchronized video and audio in **8 sampling steps** instead of the base model's full schedule. It covers both:\n\n- **t2va** (text-to-video), when no images are connected\n- **fl2va** (first/last-frame image-to-video), when `fir
MarkdownNote [Note: Model Links]: ["Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)\n\n## Input Asset\n\n- [red_line_barrier.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/red_line_barrier.png)\n\n## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n\n### Alternative Repos\n\n- [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)\n\n**vae**\n\n- [minim
ImageScaleToTotalPixels []: ["nearest-exact",0.4,32]
VAELoader []: ["minimax_h3_video_vae_int8_convrot.safetensors"]
VAELoader []: ["minimax_h3_audio_vae_fp32.safetensors"]
KSamplerSelect []: ["res_multistep"]
BasicScheduler []: ["simple",8,1]
UNETLoader []: ["fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
RandomNoise []: [1,"fixed"]
CreateVideo []: [24,"auto","sRGB","none"]
MiniMaxH3ImageToVideo []: ["Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.\n\nTimeline:\n[0s-1s] VHS static opens the frame, the title \"COMFYUI\" appears with RGB split and a slight horizontal jitter.\n[1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.\n[2.5s-4s] Clean \"STARRING\" credits appear,
ComfyMathExpression []: ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"]
PrimitiveFloat [Float (duration)]: [2]
BlockSparseAttention []: ["vsa",10,0.2,1,"",12288,256,"exact_kv_and_rows",false]
ModelAttentionBackend []: ["comfy kitchen attention"]
MiniMaxH3SigmaShift []: [10,3]
```

### Notes and long text widgets (unescaped)

    For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
    integrated_multimodal_description: [Shot 1] This is a live-action, highly stylized fashion video with a stark, high-contrast aesthetic and digital glitch overlays. The figure stands dead center against a plain off-white background that has a subtle, paper-like texture. The outfit is a layered, monochrome streetwear ensemble. The upper body is enveloped in a bulky, matte black puffer jacket with thick horizontal quilting and a central zipper, worn over a dark inner layer. The hands are tucked casually into the lower side pockets of the jacket. The lower body features loose-fitting, baggy black cargo pants with large side pockets, prominent seams, and a long, thin black fabric strap dangling freely from the left thigh area. The feet are clad in chunky, textured grey sneakers with intricate mesh detailing, thick soles, and dark grey laces tied neatly. The figure's head is completely concealed, creating an anonymous, edgy vibe. A black, textured hood is pulled up tightly, framing a black fabric face covering that reaches up to the bridge of the nose. Over the eyes, a pair of large, prominent round goggles with thick black frames and dark, opaque lenses completes the look. Cutting horizontally across the figure's midsection is a striking digital glitch effect. It consists of a thick, vibrant, solid red horizontal band, flanked by jagged, horizontal black streaks that blur and distort the fabric of the jacket behind it. The camera slowly pushes in on the static figure. As the camera moves, a low, synthesized bass drone is heard, and the thick red glitch band flickers violently, its edges stuttering horizontally, while the figure maintains a rigid, confident, and imposing stance, entirely motionless except for the digital distortion. [Shot 2] At 01.200, the camera cuts to a tight, dynamic close-up of the figure's head and upper shoulders, framed from a slightly low angle to emphasize dominance. The intricate textures of the headgear are now highly visible: the coarse weave of the black hood, the fine ribbing of the black face covering, and the smooth, matte finish of the thick black goggle frames. The dark, opaque lenses of the round goggles reflect a faint, unseen rectangular light source, adding depth. The camera executes a fast tilt-up motion. During this rapid upward movement, the figure slowly tilts their head downward, creating a contrasting motion. Simultaneously, a sharp, high-frequency digital stutter is heard as a secondary, localized digital glitch—a rapid sequence of thin, horizontal red and black scanlines—flashes across the reflective surface of the goggles for a split second. The bulky collar of the black puffer jacket frames the lower edge of the shot, remaining completely still. [Shot 3] At 02.500, the camera cuts to a medium close-up focusing specifically on the figure's lower torso, the baggy black cargo pants, and the hanging strap. The camera trucks right at a fast speed, creating a sense of lateral momentum. In response to the camera's movement, the figure shifts their weight from the left leg to the right leg. A heavy fabric rustle is heard as this subtle action causes the stiff fabric of the cargo pants to buckle and crease, creating deep shadows in the folds, while the long black fabric strap on the left thigh swings like a pendulum. As the strap swings, a new, jagged digital glitch effect abruptly tears across the left side of the frame. This glitch consists of sharp, alternating black and bright white horizontal lines that momentarily obscure the swinging strap and the side pocket of the pants, mimicking a corrupted video feed. The plain off-white background slides smoothly past in the background due to the trucking motion. [Shot 4] At 03.800, the camera cuts to a medium-wide shot, returning to a frontal perspective similar to the initial framing, but from a slightly closer vantage point. The camera zooms in quickly with a sudden, aggressive burst of speed. Reacting to the escalating energy, the figure abruptly pulls both hands out of the puffy black jacket pockets in a sharp, synchronized motion, and crosses their arms tightly over their chest. The fabric of the puffer jacket rustles and compresses. A loud, aggressive digital distortion crash is heard the moment the arms cross, and the primary thick red glitch band from Shot 1 reappears across the midsection. This time, the red band pulsates intensely, expanding vertically and horizontally, its edges fraying into chaotic, pixelated red and black blocks that briefly consume the figure's crossed arms and chest. The stark off-white background remains perfectly undisturbed, providing a clean canvas that highlights the aggressive digital distortion and the figure's final, striking pose as the clip concludes.
    
    overall_soundscape: The soundscape begins with a continuous, low-frequency synthesized bass drone that establishes a tense, modern atmosphere. As the first digital glitch flickers, a series of rapid, electrical crackling sounds and low-pitched digital stutters become clearly heard in the foreground. The moment the camera cuts to the close-up of the head, a sharp, high-frequency digital stutter cuts through the mix, synchronized with the visual flash on the goggles. Followed by this, a distinct, heavy fabric rustle is distinctly audible as the figure shifts their weight, the stiff material of the cargo pants rubbing together. Toward the end of the clip, as the figure sharply crosses their arms, a loud, aggressive digital distortion crash dominates the foreground, resembling a corrupted audio file tearing apart. This crash transitions into a final, heavy synthesized sub-bass impact that lingers as the video ends. A faint, rhythmic electronic clicking, like a fast metronome, plays consistently in the background throughout the entire sequence, tying the dynamic cuts together.
    non_diegetic_music: A fast-paced, aggressive cyberpunk-style electronic track drives the sequence, featuring a heavy, distorted 808 bassline and rapid, syncopated hi-hat trills. The music maintains a high-energy, rhythmic pulse that perfectly matches the dynamic, multi-shot fashion edit and the digital glitch aesthetic.
    This template runs the **Image to Video** task (`MiniMaxH3ImageToVideo` node) with the [FastVideo FastH3 8-Step V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) checkpoint, a DMD2-distilled model that generates synchronized video and audio in **8 sampling steps** instead of the base model's full schedule. It covers both:
    - **t2va** (text-to-video), when no images are connected
    - **fl2va** (first/last-frame image-to-video), when `first_frame` and/or `last_frame` are connected
    **Checkpoint repo**: [FastVideo/FastVideo-FastH3-Comfy](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy)
    **Scope**: this distilled checkpoint supports t2va and fl2va only. Ref2VA (multi-reference conditioning) was not distilled; use the base MiniMax H3 model for reference-based tasks.
    **Key inputs**
    - **first_frame / last_frame**: optional keyframes; the model generates the motion between them
    - **prompt**: describe the shots, motion, and the accompanying audio (dialogue, SFX, music) in one block
    - **width / height**: set via Resolution Selector. H3's native canvas is a 768px short edge, capped at 768x1344 pixels, rounded to a multiple of 32
    - **duration (seconds)**: converted to a valid frame `length` by the Math Expression node, snapping up to the model's 17-frame-per-block (17k+5) grid at 24fps
    Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)
    ## Input Asset
    - [red_line_barrier.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/red_line_barrier.png)
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    **vae**
    - [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.62 GB)
    - [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) (577.2 MB)
    **diffusion_models**
    - [fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors) (20.61 GB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (14.61 GB)
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 vae/
    │   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
    │   │   └── minimax_h3_audio_vae_fp32.safetensors
    │   ├── 📂 diffusion_models/
    │   │   └── fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors
    │   └── 📂 text_encoders/
    │       └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.
    Timeline:
    [0s-1s] VHS static opens the frame, the title "COMFYUI" appears with RGB split and a slight horizontal jitter.
    [1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.
    [2.5s-4s] Clean "STARRING" credits appear, "LATENT" and "CONTROLNET" each shown exactly once.
    [4s-5s] Final card "DIRECTED BY COMFYUI" holds, one VHS tracking glitch settling into stability.
    Hard cuts only, transitions landing with tape jumps, no push-ins, no dissolves.
    Audio: lo-fi vaporwave score, slow drum machine with soft bass, VHS tape-noise sample joins at 2.5s, melody fading for the last 1s.
    All text must be clearly legible, do not misspell English, no Chinese characters, do not repeat names or job titles, no soft dissolves, no subtitle bars.


---

## TEMPLATE: video_fasth3_ominiref_r2v (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_fasth3_ominiref_r2v.json)

```
SaveVideo []: ["video/FastH3_OmniRef","auto","auto","auto"]
ResolutionSelector [Resolution Selector (Size)]: ["16:9 (Widescreen)",1,32]
MarkdownNote [Note: Fast Video FastH3]: ["## FastVideo FastH3 OmniRef\n\n[FastH3 OmniRef](https://huggingface.co/FastVideo/FastH3-OmniRef-Comfy) is FastVideo's reference-to-video checkpoint on top of MiniMax H3: the distilled student (step 3500) trained with trajectory-based PDD and a joint DMD loss, so it renders synchronized video and audio in **8 sampling steps**. VSA is trained on 128-token tiles; ComfyUI runs 64-token tiles and the difference is not visible, since the larger tile only speed
MarkdownNote [Note: Model Links]: ["## Input Asset\n\n- [yellow_plane_ref.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/yellow_plane_ref.png)\n\n## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n- [Hugging Face:FastVideo/FastH3-OmniRef-Comfy](https://huggingface.co/FastVideo/FastH3-OmniRef-Comfy)\n\n### Alternative Repos\n\n- [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/Min
VAELoader []: ["minimax_h3_video_vae_int8_convrot.safetensors"]
VAELoader []: ["minimax_h3_audio_vae_fp32.safetensors"]
KSamplerSelect []: ["euler"]
BasicScheduler []: ["simple",8,1]
UNETLoader []: ["fasth3_omniref_8step_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
RandomNoise []: [771204598,"fixed"]
CreateVideo []: [24,8,"sRGB","none"]
ComfyMathExpression []: ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"]
PrimitiveFloat [Float (Duration)]: [5]
MiniMaxH3ReferenceToVideo []: ["",1344,768,124,"match"]
LoadImage []: ["yellow_plane_ref.png","image"]
PrimitiveStringMultiline [Input Text (Prompt)]: ["subject_definitions:\n<Subject 1> is the plane in <Picture 1>: a yellow propeller plane climbing vertically with a smoke trail.\n<Subject 2> is the setting in <Picture 1>: a towering cumulus cloud against a near-black sky.\n\nsummary:\n[reference generation] The target video is a 5-second, multi-cut cinematic aviation film. The plane climbs vertically, stalls at the top and tumbles into a dramatic dive.\n\nretention_analysis:\n<Subject 1> (appear
MiniMaxH3SigmaShift []: [12,3]
ModelAttentionBackend []: ["comfy kitchen attention"]
BlockSparseAttention []: ["vsa",10,0,1,"",12288,256,"exact_kv_and_rows",false]
MarkdownNote [Note]: ["At full `1344x768` (around 1MP) and `362` frames (15s), VSA sparse attention is about 2x faster than the int8 attention backend; at shorter, smaller clips it offers little speedup. This template uses `120 frames` (about `5s`) just to avoid OOM on some low-VRAM GPUs."]
```

### Notes and long text widgets (unescaped)

    ## FastVideo FastH3 OmniRef
    
    [FastH3 OmniRef](https://huggingface.co/FastVideo/FastH3-OmniRef-Comfy) is FastVideo's reference-to-video checkpoint on top of MiniMax H3: the distilled student (step 3500) trained with trajectory-based PDD and a joint DMD loss, so it renders synchronized video and audio in **8 sampling steps**. VSA is trained on 128-token tiles; ComfyUI runs 64-token tiles and the difference is not visible, since the larger tile only speeds up the native kernel on datacenter GPUs.
    ## About this workflow
    This template runs the **reference-to-video (ref2va)** task with the `MiniMaxH3ReferenceToVideo` node. Reference images are woven into the packed sequence, so a character, a product or a location can be reused across shots while the model still generates the performance and the audio.
    **Key inputs**
    - **ref_images**: up to 9 reference images; two are wired here (`<Picture 1>` identity, `<Picture 2>` location). Add more slots on the node if you need them.
    - **ref_videos / ref_audios**: optional, up to 3 each. A reference video can carry its own soundtrack.
    - **prompt**: address the inputs by tag in connection order (`<Picture 1>`, `<Picture 2>`, …), then describe the shot, the motion and one line of dialogue; the audio follows the description.
    - **ref_image_size**: `match` downscales references to the output resolution (faster), `max` keeps up to a 2048px short edge for stronger identity fidelity at the cost of speed.
    - **Duration → length**: the `Float (Duration)` value is converted to the 17k+5 frame grid the model expects (5 s = 124 frames at 24fps).
    ## Required settings
    - **Model Sparse Attention** must stay on method `vsa` with `keep_percent` 10: FastH3 checkpoints are trained with VSA and must never run plain block-sparse attention. `sol-attn` and `sla` are not compatible.
    - **Model Attention Backend** only picks the dense fallback used before `start_percent` and under `min_tokens`; `comfy kitchen attention` is the int8 kernel.
    - **MiniMaxH3SigmaShift** 12 / 3 is this checkpoint's video / audio shift. The FastH3 V2 text-to-video recipe (video shift 10) is not an OmniRef recipe, so do not copy it here.
    ## Checkpoints
    Two diffusion files are published for this model: the int8 convrot export used here (22.2 GB) and a bf16 export (44.2 GB). Load exactly one of them; the text encoders and VAEs are shared with the FastH3 V2 packaging. The NVFP4 AWQ option in the text encoder folder is a text encoder, not an NVFP4 export of this diffusion model.
    ## Scope
    This checkpoint is distilled for reference-to-video. Plain text-to-video works but is not its strength; use **FastVideo FastH3: Text to Video** or **Image to Video** for those tasks.
    ## Input Asset
    - [yellow_plane_ref.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/yellow_plane_ref.png)
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    - [Hugging Face:FastVideo/FastH3-OmniRef-Comfy](https://huggingface.co/FastVideo/FastH3-OmniRef-Comfy)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    **vae**
    - [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) (2.62 GB)
    - [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) (577.2 MB)
    **diffusion_models**
    - [fasth3_omniref_8step_pruned_int8_convrot.safetensors](https://huggingface.co/FastVideo/FastH3-OmniRef-Comfy/resolve/main/diffusion_models/fasth3_omniref_8step_pruned_int8_convrot.safetensors)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (14.61 GB)
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 vae/
    │   │   ├── minimax_h3_video_vae_int8_convrot.safetensors
    │   │   └── minimax_h3_audio_vae_fp32.safetensors
    │   ├── 📂 diffusion_models/
    │   │   └── fasth3_omniref_8step_pruned_int8_convrot.safetensors
    │   └── 📂 text_encoders/
    │       └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    subject_definitions:
    <Subject 1> is the plane in <Picture 1>: a yellow propeller plane climbing vertically with a smoke trail.
    <Subject 2> is the setting in <Picture 1>: a towering cumulus cloud against a near-black sky.
    summary:
    [reference generation] The target video is a 5-second, multi-cut cinematic aviation film. The plane climbs vertically, stalls at the top and tumbles into a dramatic dive.
    retention_analysis:
    <Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - the yellow plane and white smoke trail are retained.
    <Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the towering cumulus and dark sky are retained.
    detailed_description:
    The target video is a striking, high-contrast aerial film with bright sunlit clouds, deep dark sky and dynamic camera moves.
    [Shot 1] A wide shot matches <Picture 1>: <Subject 1> climbs straight up above <Subject 2>, smoke trail curling, and the camera tilts up with it.
    [Shot 2] At 00:02.000, the shot cuts to a beautiful close-up of the propeller and yellow nose as it slows at the top; the plane stalls, flips over and the camera follows it diving toward the bright cloud.
    overall_soundscape:
    Straining engine, wind, a sudden silence at the stall and a roaring dive.
    non_diegetic_music:
    Tense rising strings that break into a thrilling crescendo.
    At full `1344x768` (around 1MP) and `362` frames (15s), VSA sparse attention is about 2x faster than the int8 attention backend; at shorter, smaller clips it offers little speedup. This template uses `120 frames` (about `5s`) just to avoid OOM on some low-VRAM GPUs.


---

## TEMPLATE: image_minimax_h3_t2i (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_t2i.json)

```
SaveImage []: ["image_minimax_h3_t2i"]
26ae0bb9-8bee-49c8-bca0-2897ee957e33 []: ["Epic sci-fi concept art, extreme low-angle wide shot of a colossal spacecraft hovering above a sandstorm-ravaged desert. The open hangar glows furnace-orange through dense blowing dust and volumetric light shafts. A long boarding ramp descends to the dunes; a lone human climbs it, wind-whipped sand swirling around their ankles, tiny against the massive hull. Low amber sun behind the craft casts a hard silhouette and long god-rays across ochre and rust d
ResolutionSelector []: ["16:9 (Widescreen)",0.9,8]
MarkdownNote [Note: Steps]: ["**Steps**\n\n- `30` is the default for this checkpoint.\n- Raise it to `50` for the most detail: H3 keeps improving with more steps.\n- Lower step counts lose fine texture and skin detail first."]
MarkdownNote []: ["## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n\n### Alternative Repos\n\n- [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)\n\n**diffusion_models**\n\n- [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) - 20.97 GB\n- [minimax_h3_fl2va_pruned_bf16.safetensors](http
UNETLoader []: ["minimax_h3_fl2va_pruned_int8_convrot.safetensors","default"]
LoraLoaderModelOnly [Turbo LoRA (8 step) - bypass for base 30 steps]: ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",1]
CLIPTextEncode []: ["photograph of a red panda with a black bird sitting on it's head, background has a waterfall and zen garden"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
VAELoader []: ["minimax_h3_video_vae_fp16.safetensors"]
EmptyLatentImage []: [1024,1024,1]
ComfySwitchNode [If/Else Switch (Model)]: [false]
RandomNoise []: [0,"randomize"]
KSamplerSelect []: ["res_multistep"]
BasicScheduler []: ["simple",8,1]
ModelAttentionBackend []: ["comfy kitchen attention"]
```

### Notes and long text widgets (unescaped)

    Epic sci-fi concept art, extreme low-angle wide shot of a colossal spacecraft hovering above a sandstorm-ravaged desert. The open hangar glows furnace-orange through dense blowing dust and volumetric light shafts. A long boarding ramp descends to the dunes; a lone human climbs it, wind-whipped sand swirling around their ankles, tiny against the massive hull. Low amber sun behind the craft casts a hard silhouette and long god-rays across ochre and rust dunes. Distant convoy lights glimmer on the horizon. Monumental scale, cinematic digital illustration, ultra-detailed metal plating and dust particles, desert palette of ochre, rust, and ember orange, dramatic rim light, sharp focus, no text, no watermark.
    ## Model Links
    
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    **diffusion_models**
    - [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) - 20.97 GB
    - [minimax_h3_fl2va_pruned_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors) - 40.23 GB, full precision
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) - 15.69 GB
    - [qwen3vl_32b_minimax_h3_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) - 19.4 GB, drop-in swap
    **vae**
    - [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) - 5.21 GB (the audio VAE is not needed for stills)
    **loras**
    - [minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) - 1.96 GB, the 8-step turbo this graph loads
    - [minimax_h3_turbo_4step_ema_ckpt850.safetensors](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ema_ckpt850.safetensors) - 780 MB, optional 4-step community LoRA ([larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)); rename to `MiniMaxH3_fl2va_pruned_turbo_lora_larryvrh_4step_ema_ckpt850.safetensors`
    - [minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors) - 1.96 GB, 4-step, tuned for 768p frames
    **Disk footprint**: about **43.8 GB** with the files this graph loads (20.97 + 15.69 + 5.21 + 1.96).
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 diffusion_models/
    │   │   └── minimax_h3_fl2va_pruned_int8_convrot.safetensors   20.97 GB
    │   ├── 📂 text_encoders/
    │   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors       15.69 GB
    │   ├── 📂 vae/
    │   │   └── minimax_h3_video_vae_fp16.safetensors               5.21 GB
    │   └── 📂 loras/
    │       └── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors  1.96 GB
    Single-image decoding needs the ComfyUI build that ships PR #16881 (merged 2026-10-10).
    ## Report Issue
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)


---

## TEMPLATE: image_minimax_h3_image_t2i (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json)

```
SaveImage []: ["minimax_h3_image_t2i"]
ResolutionSelector []: ["1:1 (Square)",16,32]
b350fbfc-0ff8-4af1-8d09-09f50d84fcaa []: ["The image is a cheerful toy-packaging poster built around a handmade plush fox floating above a summer meadow, drawn in a warm, fairly saturated palette: clear sky blue deepening toward the top, rich burnt-sienna and terracotta plush for the fox, warm cream, fresh sage green in the meadow, cornflower and cherry-red accents on the small props, and buttercup yellow in the lettering, with bright afternoon sun from the upper left and crisp short shadows. Th
MarkdownNote [Note: size]: ["MiniMax H3 Image is native at about 16 MP, so this workflow defaults to 16 MP (4096x4096 at 1:1). Effective range is about 4 MP to 20 MP: lower `megapixels` to 4 MP on a small GPU or for faster runs, going higher gains nothing.\n\n**Settings** - 6 steps, CFG 1.0, `euler` + `simple`, fixed by the CFG-distilled checkpoint. The negative is a `Conditioning Zero Out` and is never read. `switch` on the subgraph is **on**, so `Generate Text` (Qwen3.5 9B) rewrites your promp
MarkdownNote [Model Links]: ["Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)\n\n## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n- [Hugging Face:Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5)\n\n**diffusion_models**\n\n- [minimax_h3_image_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H
UNETLoader []: ["minimax_h3_image_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_int8_convrot.safetensors","minimax","default"]
VAELoader []: ["minimax_h3_image_vae_fp16.safetensors"]
MiniMaxH3ImageConditioning []: ["",3104,3104]
TextGenerateV2 []: ["",8192,"on",1,20,0.95,0,1,0,0,true,true,"auto"]
CLIPLoader []: ["qwen3.5_9b_w4a8_gptq.safetensors","minimax","default"]
PrimitiveStringMultiline [Text (system_prompt)]: ["Reason in the thinking section first. The visible final content must be exactly one JSON object with exactly two keys: expanded_caption and aspect_ratio. Every ASCII double quotation mark used as caption content inside expanded_caption must be serialized as the JSON escape sequence \\\". Do not output text outside that object."]
PreviewAny []: []
KSampler []: [42,"randomize",6,1,"euler","simple",1]
PrimitiveStringMultiline [Text (task_prompt)]: ["You are a caption-expansion and canvas-selection model for still-image\ntext-to-image generation. Rewrite input_text into a natural English detailed\ncaption and choose one aspect ratio for the requested final image. After private\nreasoning, serialize the complete caption and aspect ratio together as the one\nrequired two-key JSON object; no visible analysis, markdown, or self-talk. The\ncaption may be long when the requested image is a complex i
StringReplace []: ["","{input text}",""]
ComfySwitchNode []: [false]
PrimitiveStringMultiline [Text (user_prompt)]: [""]
JsonExtractString []: ["","expanded_caption"]
```

### Notes and long text widgets (unescaped)

    The image is a cheerful toy-packaging poster built around a handmade plush fox floating above a summer meadow, drawn in a warm, fairly saturated palette: clear sky blue deepening toward the top, rich burnt-sienna and terracotta plush for the fox, warm cream, fresh sage green in the meadow, cornflower and cherry-red accents on the small props, and buttercup yellow in the lettering, with bright afternoon sun from the upper left and crisp short shadows. The fox is the hero, sewn from long-staple mohair-type plush in warm burnt sienna and hand-stuffed so its silhouette is slightly lumpy; the pile is long and irregular, single fibres catching the light along its outline as a bright hairy halo against the sky, matted in patches where it has been handled, with cream muzzle, chest bib and sock paws, small pointed ears with soft cream inner felt, matte dark eyes with a dull highlight, a stitched smile, brushed felt blush patches, a small black felt nose, and a brush tail whose long pile fades to cream at the tip. It floats tilted in the air, hind legs tucked, one foreleg raised holding a small bunch of felt daisies in off-white and dusty rose, a thin felt collar with a small dull brass bell around its neck. Around and behind it the scene is populated rather than empty: a small plush mushroom friend sits on the grass at the lower right, its rounded cap in cherry red felt with cream spots and a stitched face on a cream stalk; a tiny round plush bird in cornflower blue perches on the fox's tail, its body made of short dense pile with a small felt beak; a second, much smaller fox plush lies on its back on the grass at the lower left, legs in the air; three felt daisies on thin wire stems stand in the meadow, and a scatter of loose felt petals and two small acorns lie among the lit blades; a few soft felt cloud puffs hang on thin white strings from the upper right, and a folded paper plane with a dashed trail glides across the sky toward the fox. Thin white line doodles are drawn flat over the blue: four small four-point stars, one larger outlined star at the left, a loose squiggle beside the raised paw, and a dotted vertical line at the right, all in a loose dry hand-drawn stroke. The typography is set in soft matte material rather than glossy plastic: across the top of the frame the word "MINIMAX" is inflatable and rounded but reads as thick felted wool, in warm buttercup-cream with a faint fibre texture, edges slightly soft and imperfect, lit gently from above with a shaded underside and no specular shine, filling the full width and slightly arched; directly beneath, smaller and centred, "H3" sits between two small dots, and below it a second small bouncy line reads "IMAGE MODEL" in the same felted cream. Along the bottom, on the white cloud band, a heavy bold sans-serif logotype reads "TEXT TO IMAGE" in saturated cornflower blue, tracked wide across almost the full width, its ink slightly uneven as if printed on soft paper. Just above the right end of that line a small right-aligned block of the same cornflower is stacked in four short lines reading "ONE CHECKPOINT", "TWO TASKS", "SIX STEPS", "NO CFG". All lettering is English only, with no Chinese characters anywhere on the poster, and the light is bright and even with soft shadows under every plush object.
    MiniMax H3 Image is native at about 16 MP, so this workflow defaults to 16 MP (4096x4096 at 1:1). Effective range is about 4 MP to 20 MP: lower `megapixels` to 4 MP on a small GPU or for faster runs, going higher gains nothing.
    
    **Settings** - 6 steps, CFG 1.0, `euler` + `simple`, fixed by the CFG-distilled checkpoint. The negative is a `Conditioning Zero Out` and is never read. `switch` on the subgraph is **on**, so `Generate Text` (Qwen3.5 9B) rewrites your prompt first.
    ## Recommended sizes (16MP)
    | Aspect ratio | Size (width x height) |
    | --- | --- |
    | 1:1 | 4096x4096 |
    | 4:3 / 3:4 | 4736x3552 / 3552x4736 |
    | 3:2 / 2:3 | 5024x3360 / 3360x5024 |
    | 16:9 / 9:16 | 5472x3072 / 3072x5472 |
    | 21:9 | 6272x2688 |
    Guide: [Subgraph](https://docs.comfy.org/interface/features/subgraph)
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    - [Hugging Face:Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5)
    **diffusion_models**
    - [minimax_h3_image_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors) (21 GB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) (25.28 GB)
    - [qwen3.5_9b_w4a8_gptq.safetensors](https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors) (6.84 GB)
    **vae**
    - [minimax_h3_image_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors) (1.17 GB)
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 diffusion_models/
    │   │   └── minimax_h3_image_pruned_int8_convrot.safetensors
    │   ├── 📂 text_encoders/
    │   │   ├── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
    │   │   └── qwen3.5_9b_w4a8_gptq.safetensors
    │   └── 📂 vae/
    │       └── minimax_h3_image_vae_fp16.safetensors
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    Reason in the thinking section first. The visible final content must be exactly one JSON object with exactly two keys: expanded_caption and aspect_ratio. Every ASCII double quotation mark used as caption content inside expanded_caption must be serialized as the JSON escape sequence \". Do not output text outside that object.
    You are a caption-expansion and canvas-selection model for still-image
    text-to-image generation. Rewrite input_text into a natural English detailed
    caption and choose one aspect ratio for the requested final image. After private
    reasoning, serialize the complete caption and aspect ratio together as the one
    required two-key JSON object; no visible analysis, markdown, or self-talk. The
    caption may be long when the requested image is a complex information or design artifact.
    Role: act like an expert visual designer and technical illustrator, not a word
    expander. Preserve every hard fact from the input, then complete the image with
    visible, inspectable content. The goal is not more words; the goal is that a
    viewer can inspect each important part of the image and find meaningful subjects,
    labels, relationships, materials, data, and hierarchy.
    First choose which kind of image this is:
    - A depicted scene or photograph: prioritize subject identity, pose, contact,
    scale, environment, lighting, lens/medium, and physical plausibility.
    - A communication artifact such as poster, character card, product-design board,
    webpage, UI, chart, dashboard, scientific/medical infographic, technical diagram,
    printed page, packaging, cover, or multi-panel composite: build a finished
    information architecture, not just a scene containing the requested objects.
    Aspect-ratio decision policy:
    Choose exactly one of these allowed ratios: 1:1, 5:4, 4:3, 3:2, 16:9, 21:9,
    4:5, 3:4, 2:3, or 9:16. Decide in the following order; earlier evidence
    overrides later evidence.
    1. Explicit final canvas
    - Preserve an explicit final-output ratio. Map explicit output pixel dimensions
      by width divided by height to the closest allowed ratio.
    - Do not confuse the final canvas with a source image, reference asset, object,
      inset, collage cell, or embedded screen.
    2. Requested deliverable
    - Use an explicitly requested medium or display surface. Treat each carousel
      page, card, or slide as its own canvas unless a contact sheet, storyboard,
      continuous strip, or combined overview is explicitly requested.
    - Prefer square or near-square for individual social posts, icons, avatars,
      compact logos, or compact product assets; vertical for covers, posters, phone
      stories, or mobile wallpapers; horizontal for slides, headers, banners, or
      explicitly wide displays. Do not invent a platform or deliverable.
    3. Primary composition
    - The main subject, action, or information structure outweighs optional context.
      A generic room, park, forest, sky, or cinematic background must not turn a
      focused subject horizontal.
    - Prefer vertical for dominant upright or full-body subjects, tall structures,
      stacked information, rising or falling motion, upward reach, or vertical
      negative space. Prefer horizontal for genuinely side-by-side structure,
      lateral action, broadly distributed subjects, a requested broad environment,
      horizontal sequences, or lateral negative space. Prefer square or near-square
      for compact, centered, radial, symmetrical, or directionally neutral content.
    4. Minimal completion and ratio strength
    - If the input is underspecified, form the smallest natural, content-preserving
      composition needed to choose a ratio. Lock it before adding detailed scenery
      or styling; do not invent a panorama, broad environment, or multi-panel layout
      to justify it, and do not revise it because later expansion adds detail.
    - Words such as "portrait," "cinematic," "video," "UE5," "Weta,"
      "photorealistic," "dramatic," and "wide-angle" do not determine orientation.
    - Use 5:4 or 4:5 for mild direction, 4:3 or 3:4 for moderate direction, and
      3:2 or 2:3 for clear photographic direction. Reserve 16:9 and 9:16 for
      strongly directional or mobile-first compositions, and 21:9 for genuine
      banners, panoramas, timelines, or explicitly continuous wide layouts. When
      evidence remains weak, prefer 1:1 or a moderate ratio.
    Choose and lock the ratio from the requested canvas, deliverable, and smallest
    natural primary composition before enriching the caption. Later details added by
    the caption-expansion rules must not retroactively change the ratio.
    Universal hard rules:
    1. Preserve exact requested text, named entities, counts, formulas, data values,
    component names, style, medium, orientation, and left/right/top/bottom order.
    2. Required content must be visible, not implied. If text or a label is requested,
    place it on the surface and describe where it sits.
    3. For active bodies, solve scene geometry before pose detail. Preserve any
    domain-native layout first: a sport court, game grid, path, lane, stair, room,
    or interface should keep its recognizable structure unless the input changes
    it. Do not move the target or redesign the ground merely to obtain a preferred
    body view. Choose one camera only after the scene layout is fixed, then place
    the source and target in that camera's image frame using screen-left/screen-
    right, upper/lower image position, and nearer/farther depth. Silently trace the
    source-to-target vector in this fixed view. Derive the actor's camera-relative
    orientation from that vector and state it explicitly as front-facing, back-
    facing, left-facing profile, right-facing profile, front three-quarter, or rear
    three-quarter, together with a frame-absolute motion direction such as leftward,
    rightward, toward the camera/foreground, or away from the camera/deeper into the
    scene. Never use only a target-relative phrase such as "facing square 2" or
    "looking toward the door". The mapping follows the chosen geometry rather than
    a fixed default: a target at screen-right implies a right-facing or rightward
    body; a target at screen-left implies a left-facing or leftward body; a target
    farther into the scene implies a back-facing or rear-three-quarter body moving
    away; and a target nearer the camera implies a front-facing or front-three-
    quarter body moving toward the viewer. These are coordinate mappings, not
    instructions to relocate every target. Verify the visible body surfaces before
    writing: movement away must show the back, back of the head, or rear shoulders
    and must not show a frontal chest or full face; movement toward the camera must
    show the face and chest; lateral movement must read as the corresponding side
    profile. If the stated source/target positions and visible body surfaces imply
    opposite directions, revise the body orientation instead of changing the scene
    layout. Keep gaze, chest, torso lean, leading limb, support foot or hand, and
    motion consistent with the derived orientation and with one frozen phase. A
    limb cannot be simultaneously ahead and behind, planted and airborne, or
    assigned to two contacts. For actions on marked ground, use an elevated high-angle oblique or near-
    overhead three-quarter camera, never a low ground-level view. Center the camera on the complete ground map and make it the dominant, fully
    contained region of the image: show its outer boundary, every cell or segment,
    the source, target, and full travel path. After deriving the source-to-target
    vector, place the actor's torso outside the map on the ray extending backward
    from the source in the direction opposite the target. Do not move the actor to
    an arbitrary side margin if that would change the apparent travel direction.
    Only the support foot may enter the near edge of the source cell, leaving its
    label visible; place the leading foot in the source-target direction near their
    gap. Keep torso, limbs, hair, clothing, and cast shadow outside every later cell
    and label. Widen or elevate the camera framing rather than rotating the actor or
    relocating the target. This placement must preserve the previously derived front,
    back, left, right, or three-quarter camera-relative orientation. Preserve conventional topology and perspective, keep every label upright and
    large enough to read, and show each requested label exactly once in order. If
    the input does not specify label colors or materials, use one neutral, uniform,
    high-contrast marking style for every label; unspecified styling is not permission
    to assign a different color to each item. Keep the actor's hands empty unless the
    input names a held object. Do not add loose markers, stones, chalk pieces, buckets,
    ropes, toys, shadows, limbs, or cropping over any label or grid line.
    Preserve a plausible silhouette and avoid broken foreshortening, hidden critical
    joints, mirrored layouts, alternative camera choices, and blended before-and-
    after poses. For emotion-driven photo or video frames, keep facial anatomy
    natural and consistent with the selected viewpoint.
    4. For hands, grips, linked fingers, hugs, carrying, pointing, tools, weapons, or
    body-object contact, choose the framing that makes the interaction readable. Use
    a medium shot or waist-up view when hand/facial detail is central; do not force a
    full-body view if it makes the required contact or expression tiny. State which
    hand or limb touches what, how the fingers or support points align, and keep
    nonessential limbs simple.
    5. Use world knowledge as visible evidence only: cultural motifs, scientific
    structures, medical anatomy, interface conventions, manufacturing details,
    professional tools, materials, units, labels, measurement marks, clothing, props,
    architecture, or environment. Do not add lore that cannot be seen.
    6. Respect requested subject count. If the prompt names event types, two
    characters, or one primary subject, keep the number of primary figures to that
    minimum unless extra people are requested. Background crowds, spectators, or
    staff should read as small silhouettes or graphic texture, not extra competing
    main subjects.
    For communication artifacts, silently make a content map before writing. Divide
    the canvas in reading order into major slots: header/title, subtitle/tagline,
    hero visual, left panel, right sidebar, card row, icon group, diagram body,
    process path, chart area, formula box, exploded view, inset, specification area,
    swatch strip, instruction cards, footer, screen region, or background context.
    Each major slot should normally include at least three of these: position, visible
    text or label, concrete visual subject, function or relationship, material/data
    treatment, and hierarchy/style treatment. Do not use vague slot fillers such as
    "icons," "details," "information," "supporting graphics," or "annotations"
    without saying what they show.
    Make the artifact feel complete. If the prompt asks for a card, board, poster,
    infographic, webpage, or product sheet, add a few plausible supporting elements
    that belong to that artifact when they help completeness: subtitle, role/profile,
    about note, quote, ratings, legend, color key, units, variable definitions,
    dimension labels, back/side view, material stack, use-case icons, footer tips,
    QR/logo placeholder, warning, or compact specification table. Keep these as short
    readable on-image labels or one-line captions, not prose blocks.
    Correspondence and legibility rules:
    - Multi-view outputs such as floor plan plus interior render, blueprint plus
    product render, map plus scene, or before/after panels need a shared inventory
    and coordinate frame. Reuse the same walls, openings, furniture/components,
    materials, labels, and relative positions in every panel. Name the render camera
    origin or viewing direction from the plan when useful, and do not invent major
    objects that are absent from the paired view.
    - Dense flowcharts, medical diagrams, route maps, and process infographics should
    be simplified into a few large numbered nodes, separated lanes, or a clean
    U-shaped loop. Arrows must be thick, directional, non-crossing, and attached to
    named start/end nodes. Prefer legible topology and large labels over many small
    leader lines, tiny arrows, or decorative clutter.
    Contract affordances:
    - Scientific/medical/educational infographic: include named section headers;
    main diagram with attached component labels; arrows with correct direction and
    start/end nodes; legends or color keys; short functional captions; key-point
    cards with icon plus meaning; definitions and units; formula box with variable
    definitions; and process/system context such as body, lungs, vessels, lab setup,
    molecule, device, or environment when implied.
    - Charts, graphs, dashboards, and data screens: include chart title when useful,
    axis labels, tick labels or scale endpoints, units, legend entries, colors, data
    marks, selected states, and values or qualitative endpoints placed consistently
    with the visual geometry. Even if the chart appears on a photographed stage
    screen, the screen content still needs readable chart semantics.
    - Product/cultural design board: include title/subtitle/tagline; inspiration or
    use-case cues; large hero render; motif-specific details; front/back/side or
    scale insets when useful; exploded layers; manufacturable joints, pins, clasps,
    fasteners, seams, ribs, holes, tolerances, or contact points; material stack;
    finish behavior; dimension labels; palette swatches; specification and craft
    icons.
    - Character card/mascot sheet: include identity/title; name tag; role/profile or
    about area; full-body readable proportions; attribute rows with icon, label, and
    brief meaning; props tied to the role; material swatches or callouts; optional
    rating, quote, badge, instruction row, or contextual background so it reads as a
    designed card/poster rather than a plain character portrait.
    - UI/web/dashboard/data graphic: include real page/screen sections, navigation,
    controls, fields, buttons, cards, selected states, labels, values, icons, and
    intentional empty states only when requested.
    - Public poster/cover/packaging/display graphic: include typography hierarchy,
    headline/subtitle/supporting copy, image slots, instruction cards, icons, badges,
    borders/rules, QR/barcode/logo placeholders if requested, palette, and intentional
    negative space. When a poster features sports, products, or people, use one clear
    hero subject per requested category unless the prompt asks for a group.
    Style comes after content. Choose clean isolation for icons/logos/cutouts and
    technical assets; layout-rich pages or boards for design communication; believable
    environment, lighting, lens, and depth for photographs. Replace abstract mood and
    generic adjectives with specific visible cues.
    Final self-check: every input requirement is visible; every major slot has real
    content; required text/formulas/values are exact; labels attach to correct
    structures; arrows and data geometry are coherent; body orientation is explicit relative to the camera and agrees with the preserved scene layout, source, target, depth, gaze, support, and motion; paired panels correspond; subject count has not grown by accident;
    world knowledge appears as visible evidence; no important surface is blank.
    Final output contract:
    Use the private thinking section to plan the caption and ratio. After thinking,
    the visible final answer must be exactly one JSON object with exactly these two
    keys:
    {{"expanded_caption":"ONE_DETAILED_ENGLISH_CAPTION","aspect_ratio":"ALLOWED_RATIO"}}
    For example, serialize a caption containing the title "你好" as:
    {{"expanded_caption":"The title reads \"你好\".","aspect_ratio":"16:9"}}
    Replace ONE_DETAILED_ENGLISH_CAPTION with one fluent English caption paragraph
    that follows every rule above, and ALLOWED_RATIO with exactly one allowed ratio
    string. Keep expanded_caption as one valid JSON string. Every ASCII double quotation mark used as caption content inside expanded_caption must be serialized as the JSON escape sequence \". Escape all other characters as valid JSON requires. Do not add or omit keys. Do not put thinking or think tags inside
    the object. Do not output a Markdown fence, alternatives, or any text outside
    the final JSON object.
    input_text: {input text}
    output_text:


---

## TEMPLATE: image_minimax_h3_image_edit (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json)

```
SaveImage []: ["image_minimax_h3_image_edit"]
MarkdownNote [Note: size and Referencen Images]: ["MiniMax H3 Image is native at about 16 MP, so this workflow edits at 16 MP: the megapixel value sets the canvas, the output size and the reference latent, and the input picture is upscaled to it first. Effective range is about 4 MP to 20 MP; drop `scale total pixels` to 4 MP for a faster, smaller run.\n\n**Settings** - 6 steps, CFG 1.0, `euler` + `simple`, fixed by the CFG-distilled checkpoint. The negative is a `Conditioning Zero Out` and is ne
MarkdownNote [Model Links]: ["## Input Asset\n\n- [vintage_yellow_taxi.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/vintage_yellow_taxi.png)\n\n## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)\n- [Hugging Face:Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5)\n\n**diffusion_models**\n\n- [minima
UNETLoader []: ["minimax_h3_image_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_int8_convrot.safetensors","minimax","default"]
VAELoader []: ["minimax_h3_image_vae_fp16.safetensors"]
MiniMaxH3ImageConditioning []: ["",3104,3104]
TextGenerateV2 []: ["",8196,"on",1,20,0.95,0,1,0,0,true,true,"auto"]
CLIPLoader []: ["qwen3.5_9b_w4a8_gptq.safetensors","minimax","default"]
PrimitiveStringMultiline [Text (system_prompt)]: ["Reason in the thinking section first. The visible final content must be exactly one JSON object with exactly two keys: expanded_caption and aspect_ratio. Every ASCII double quotation mark used as caption content inside expanded_caption must be serialized as the JSON escape sequence \\\". Do not output text outside that object."]
PreviewAny []: []
KSampler []: [422033620208127,"randomize",6,1,"euler","simple",1]
PrimitiveStringMultiline [Text (task_prompt)]: ["You are an expert at enhancing a short image-editing instruction into a rich, vivid, reference-conditioned description of the intended finished target image.\n\n## What you are given\n- Input image(s), labeled sequentially starting from <Picture 1>. These are the user's source and reference images.\n- One short editing instruction. It may be terse, vague, informal, or phrased as a desired result rather than an imperative. The final text part suppl
StringReplace []: ["","{input text}",""]
ComfySwitchNode []: [true]
PrimitiveStringMultiline [Text (user_prompt)]: ["Replace the vintage yellow taxi in <Picture 1> with a pastel pink ice-cream truck with a scalloped awning and a glowing service window, sitting in the same side profile at the curb."]
JsonExtractString []: ["","expanded_caption"]
LoadImage []: ["vintage_yellow_taxi.png","image"]
ResizeImageMaskNode []: ["scale to multiple",32,"area"]
MarkdownNote [Note: PE model]: ["For MiniMax H3 Image's PE model it’s not tied to any specific model\n\n[MiniMax H3 Image PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) repo is using `Qwen3.8-27B, FP4` and we are using Qwen3.5 9B in this workflow"]
ResizeImageMaskNode []: ["scale total pixels",16,"area"]
MarkdownNote [Note]: ["Hover on the `prompt output` or `right click -> show link` to restore the link"]
PrimitiveBoolean [Boolean (Enable PE?)]: [true]
ImageCompare []: []
```

### Notes and long text widgets (unescaped)

    MiniMax H3 Image is native at about 16 MP, so this workflow edits at 16 MP: the megapixel value sets the canvas, the output size and the reference latent, and the input picture is upscaled to it first. Effective range is about 4 MP to 20 MP; drop `scale total pixels` to 4 MP for a faster, smaller run.
    
    **Settings** - 6 steps, CFG 1.0, `euler` + `simple`, fixed by the CFG-distilled checkpoint. The negative is a `Conditioning Zero Out` and is never read. `Boolean (Enable PE?)` is **on**, so `Generate Text` (Qwen3.5 9B) rewrites your instruction first. For more references, add `Load Image` nodes and name them `<Picture 2>`, `<Picture 3>`, ...
    ## Recommended sizes (16MP)
    | Aspect ratio | Size (width x height) |
    | --- | --- |
    | 1:1 | 4096x4096 |
    | 4:3 / 3:4 | 4736x3552 / 3552x4736 |
    | 3:2 / 2:3 | 5024x3360 / 3360x5024 |
    | 16:9 / 9:16 | 5472x3072 / 3072x5472 |
    | 21:9 | 6272x2688 |
    ## Input Asset
    - [vintage_yellow_taxi.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/vintage_yellow_taxi.png)
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    - [Hugging Face:Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5)
    **diffusion_models**
    - [minimax_h3_image_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors) (21 GB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) (25.28 GB)
    - [qwen3.5_9b_w4a8_gptq.safetensors](https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors) (6.84 GB)
    **vae**
    - [minimax_h3_image_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors) (1.17 GB)
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 diffusion_models/
    │   │   └── minimax_h3_image_pruned_int8_convrot.safetensors
    │   ├── 📂 text_encoders/
    │   │   ├── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
    │   │   └── qwen3.5_9b_w4a8_gptq.safetensors
    │   └── 📂 vae/
    │       └── minimax_h3_image_vae_fp16.safetensors
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)
    Reason in the thinking section first. The visible final content must be exactly one JSON object with exactly two keys: expanded_caption and aspect_ratio. Every ASCII double quotation mark used as caption content inside expanded_caption must be serialized as the JSON escape sequence \". Do not output text outside that object.
    You are an expert at enhancing a short image-editing instruction into a rich, vivid, reference-conditioned description of the intended finished target image.
    ## What you are given
    - Input image(s), labeled sequentially starting from <Picture 1>. These are the user's source and reference images.
    - One short editing instruction. It may be terse, vague, informal, or phrased as a desired result rather than an imperative. The final text part supplies it as User Editing Instruction (JSON string).
    - User Aspect Ratio Selection: any concrete positive-integer W:H ratio, or adaptive, supplied in the final text part.
    - Each input image is immediately preceded by a Reference Media text part containing its <Picture N> label and aspect_ratio W:H. The caller has already reduced the image's original display width and height to their exact lowest integer ratio, after EXIF orientation and before preview resizing or padding. No pixel dimensions are supplied in this metadata. The ratio applies only to the immediately following image. The supplied aspect_ratio is the reference image's exact ratio, not necessarily the final target ratio. Use it as given; do not estimate the source ratio from the preview. Determine the final target ratio using the rules below. If the original dimensions are unknown, the metadata says aspect_ratio unavailable.
    - There is no target image. Construct the intended target from the visible source evidence and the instruction; never pretend to have observed a hidden result.
    ## Response format
    Return exactly one valid JSON object with exactly these two keys, both with string values:
    {"expanded_caption":"COMPLETE_DESCRIPTION","target_aspect_ratio":"W:H"}
    Resolve target_aspect_ratio using the rules below. Put the complete description in expanded_caption as one continuous paragraph with no actual line breaks. All narration, language, Subject, literal-text, and three-part rules below apply to expanded_caption, not to the outer JSON syntax or target_aspect_ratio. Use valid JSON escaping for quotes and backslashes without changing the decoded literal content. Do not output reasoning, a code fence, extra keys, or text outside this JSON object.
    ## Target aspect ratio
    Determine target_aspect_ratio before writing expanded_caption. Treat it as a constraint on the intended target image, adapting framing and composition while respecting the requested edit and reference-preservation requirements. Describe any necessary changes under the changed/new-content rules. User Aspect Ratio Selection is part of the user's request. A concrete selection governs conflicting canvas wording in the editing instruction; continue to honor all compatible content constraints. Do not mechanically copy reference metadata or routing explanations into expanded_caption, and do not treat metadata as lettering inside an image.
    Reading aspect_ratio metadata alone must not create a <Subject N> or change a preservation relationship. However, when source framing, aspect ratio, or composition is independently preserved, changed, or borrowed, identify that source-derived scope under the rules below. If realizing the selected canvas actually changes framing, visible content, placement, or layout within a Subject's defined scope, reflect that change in its relationship and describe only the changed/new consequences in Part 3. Do not call the entire source composition fully preserved while changing it; do not automatically mark every Subject partially preserved merely because the canvas ratio differs. Limit any completion or reframing to what the user's edit and selected canvas require; preserve unrelated content.
    Read User Aspect Ratio Selection from the final input part.
    Specified ratio:
    If a concrete ratio is supplied, copy that exact string to target_aspect_ratio. Any positive-integer W:H ratio is allowed; it is not limited to a preset list. Neither the editing instruction nor a reference image may override it.
    Adaptive mode:
    Only when the selection is adaptive, determine the aspect ratio of the intended finished target image, which may differ from the main reference image. Apply these rules in order:
    1. Follow an explicit final target ratio or target dimensions in the editing instruction. Use the requested final geometry rather than source or earlier geometry. Exact numeric geometry takes precedence over generic orientation. When only a final orientation is specified, use 21:9 for ultra-wide, 16:9 for landscape, 1:1 for square, and 9:16 for portrait.
    2. Otherwise, select the main reference that supplies the edited base canvas or overall target composition and has a usable aspect_ratio. Prioritize an explicitly designated canvas/composition reference, then the edited base, then an overall composition anchor. Break ties by Picture order and skip missing, invalid, or unavailable ratios. Images used only for identity, objects, color, or style do not determine the canvas.
       Use the main reference's aspect_ratio as the starting ratio. Copy it exactly unless the requested edit requires a different canvas shape. Proportional resizing or changing a subject's size alone does not require changing the ratio.
       When expansion or cropping changes the canvas shape, determine the ratio of the resulting target. If the available information unambiguously determines it, calculate it exactly and reduce it. For relative changes:
       target_ratio = reference_ratio × width_scale / height_scale,
       where both scales are relative to the original canvas.
       If the requested shape change does not determine a unique ratio, choose one that fits the intended result while preserving unrelated composition. Adjust it only as needed for the requested edit, not merely for aesthetics.
    3. If neither an explicit target requirement nor a usable main reference determines the ratio, output 16:9.
    Apply these rules to the whole target canvas. Grid row/column counts, individual panels, objects inside the image, and literal text are not whole-canvas ratio requests. A portrait subject or wide-angle lens alone is not a target-orientation request. Do not use a reference board's ratio as the ratio of one panel.
    Serialize target_aspect_ratio as two positive integers in W:H form, with no spaces; never output adaptive. Preserve concrete user selections, explicit positive-integer target ratios in the instruction, and unchanged reference ratios exactly. Convert other explicit numeric ratios or target dimensions to their exact reduced integer ratio; reduce newly calculated or chosen ratios to their lowest integer form. Do not snap to preset ratios or clamp to a supported range.
    Later descriptive additions must not change the chosen ratio. Pixel-size resolution, 32 alignment, and downstream size fallback are outside this task.
    ## Interpret the User's Intent Flexibly
    Read the instruction together with the input images and any context it supplies to understand the result the user actually wants. Terse wording, colloquial phrasing, shorthand, and requests such as 'complete this' may require reasonable interpretation and substantive completion. Do not mechanically echo the request, insist that the user prewrite every detail, or treat an illustrative example or category list as the only supported case.
    Choose the interpretation best supported by the visible evidence, the request's purpose, and ordinary context. Supply the visual or textual details needed to make that intended result complete and coherent, including directly supporting consequences. Be flexible in how you realize the request, but do not override explicit constraints, change specified literal wording without permission, invent source facts, or expand unrelated content. Keep the boundaries between preserved source content, requested changes, newly authored text, and actual image lettering clear.
    ## Source-Grounded Content and New Target Content
    Treat source-derived content and instruction-introduced content differently. Source facts describe what is actually visible in the inputs; target changes describe the requested result. Source fidelity does not prohibit an explicitly requested transformation. Preservation is the default only outside requested changes and their necessary consequences.
    A) Descriptions of source facts and properties retained in the target must be strictly faithful. Ground identity, facial features, body shape, clothing, objects, colors, materials, text, data, count, pose, layout, scene, camera, lighting, and spatial relationships in the specific input picture where they are visible. Do not invent a source attribute, accessory, logo, word, number, object, or relationship. If something is genuinely illegible or ambiguous, describe that fact neutrally rather than guessing. When an unfamiliar source element lacks a reliable name, identify it using its location and distinctive visible cues rather than guessing its identity or function.
    B) Content newly introduced or substantially transformed by the instruction should still be imagined richly, concretely, and coherently. Within the changed or new scope, supply plausible visual specifics that make the requested result recognizable and renderable: materials, shapes, arrangement, depth, perspective, lighting direction, shadows, reflections, contact, overlap, occlusion, atmosphere, and surrounding details that naturally belong to the requested scene. A newly requested background, environment, design, or global style may be described as a complete vivid visual scope because that scope itself is new or changed. Do not make the answer skeletal merely to satisfy a delta format.
    Whenever the intended target needs visual content not already determined by a preserved Subject or an explicitly referenced property, reasonably imagine and concretely describe that new content. This applies to any new scene, entity, action, design, composition, or newly visible region, not only viewpoint changes. Treat unspecified details within that new scope as room for coherent creation, not a reason to leave it vague. Keep the invention consistent with the user's intent and remaining reference constraints; do not redesign preserved Subjects or alter properties that the request requires you to match. If a source Subject gains newly visible parts, keep its identity and Subject number; describe the plausible completion in Part 3, never as an observed source fact.
    Creative latitude stays inside the requested or directly supporting scope. Do not redesign unrelated source content or import incidental attributes from a donor picture merely to enrich the prose. Ground factual claims in the available reference and instruction; do not invent unsupported specifics and present them as real. Explicitly requested fictional names or illustrative sample data may be authored as fictional or illustrative target content, never as recovered source facts or verified real-world information. Omit unsupported specifics, not the requested writing itself, and do not add unrelated text beyond the requested scope.
    Complete every textual deliverable the user asks you to write, supplement, finish, expand, rewrite, or otherwise provide. If exact wording is not supplied, compose the complete final wording yourself; missing wording is not a reason to leave the content unspecified. This applies to any requested textual deliverable, not limited to particular genres or to text inside an image. Requested wording is the deliverable itself, not forbidden restatement of the user's instruction. Do not delegate completion to the downstream model.
    ## Output contract
    Write expanded_caption as one continuous plain-text paragraph. Do not add structural headings, labels, bullets, markdown, preamble, reasoning, notes, or meta-commentary to the narration. Literal content may contain such words or symbols when they belong to source text or requested writing. The paragraph has three logical parts, in this order:
    1. natural definitions of the relevant source-derived <Subject N> elements;
    2. one target overview followed by a separate preservation/change relationship clause for every defined Subject;
    3. Changed and Newly Introduced Target Details Only, together with the complete wording of any textual deliverable the user asks you to provide.
    Determine each requested text's destination from the instruction. If it belongs in the image, include its literal wording and describe placement as needed. If it is requested only as a textual response, provide it as text alongside the visual delta; do not turn it into a sign, caption, label, or other image element. Quoting a textual deliverable does not mean that it should be rendered in the image.
    Complete all Subject definitions before the target overview. Do not expose structural commands such as 'Define', 'Let', 'Identify', 'Section 1', 'Section 2', or 'Section 3' as narration. Write natural element-first definitions, for example, 'The woman in a red coat in <Picture 1> is <Subject 1>.' Refer to the intended visual result only as 'the target image'. Narrate the result, not the user's instruction or an imperative such as 'generate' or 'please create'; this does not prohibit exact supplied text or requested authored text from containing those words. If no source-derived element is retained or referenced, omit Subject definitions and relationships and begin with the target overview; do not invent a Subject to fill the structure.
    ## Part 1: Natural Reference Element Identification
    Assign <Subject N> only to source-derived elements that remain visible or traceable in the intended target, or that provide a visible transferred or borrowed property. Every definition must name its source with an explicit <Picture N> and concise visible cues sufficient to uniquely identify the referenced object, region, or property. Add detail only to resolve ambiguity or clarify the reference/edit scope, not to restate its full appearance. Keep the full text/data transcription required below; concise identification must not omit readable content. List source pictures individually; never use ranges or vague phrases such as 'the reference images', 'the previous images', 'all inputs', or 'these pictures'. Use one number for the same convincingly identified person or element across pictures, and keep clearly different identities separate.
    When only a source property is needed, define that property rather than the entire source object; use the source object or region to locate the property. Give the source object its own Subject only if it is independently used in the target.
    Choose Subject granularity by relationship scope, not by a quota. An element that is independently moved, recolored, restyled, reposed, recounted, replaced while remaining traceable, or used as a donor should be independently addressable. Split a part, pose, layout, palette, lighting setup, text block, garment, accessory, held object, or background only when it needs its own preservation/change judgment. Do not mechanically split an ordinary person into face, hair, hands, every garment, and every accessory when those parts move together and have no independent role. Do not force one Subject per picture, and do not define incidental objects merely to make the inventory longer.
    Judge preservation over the scope actually defined. A parent Subject cannot be fully preserved if a changed child remains inside its scope, even when the child has a separate number. A genuinely unchanged narrower region or identity may be defined explicitly, but do not call the whole person, object, or scene unchanged while editing part of it. Do not transcribe the same source field again through an overlapping parent or child Subject.
    The identifiable types include the following when independently relevant:
    - Entity type: a person, animal, object, creature, garment, accessory, prop, vehicle, furniture item, or other concrete entity.
    - Scene/background type: a room, landscape, street, stage, tabletop, architecture, or other environment preserved, changed, or used by the target.
    - Attribute/style type: a source-derived texture, material language, visual medium, design style, typography, border treatment, or other transferable visible quality.
    - Shared-element type: one convincingly identical visual element shown across several pictures; define it once and list every source picture.
    - Multi-form reference type: one source element used in several intended target forms, such as a physical figure, package artwork, screen render, or line drawing; define it once and distinguish the forms later.
    - Data-bearing chart/table/plot/document type: do not collapse readable data into one undifferentiated Subject. Identify the source-derived series, bars, lines, slices, columns, labeled groups, axes, titles, legends, headers, fields, and readable cells needed to preserve or edit the visible data faithfully.
    - Dense/multi-line text type: identify every clearly readable source-derived line or field in a document, slide, poster, menu, search interface, caption block, subtitle group, or UI region. Either define lines separately or define one coherent block while transcribing its lines literally and in order; do not replace readable content with a vague 'text' Subject.
    - Text/watermark type: identify an isolated watermark, brand text, subtitle, label, sign, logo text, or other short readable element, and state its clearly readable literal content once in Part 1. Later use its Subject token unless that text is intended to change.
    - Part/region type: hair, a sleeve, a wheel, a screen area, a poster corner, a body region, a panel, or another local part that needs its own relationship judgment.
    - Spatial-layout/composition type: framing, aspect ratio, grid or split-screen geometry, panel order, placement scheme, image-text hierarchy, or whitespace when the arrangement itself is preserved, changed, or borrowed.
    - Color/palette/lighting type: a source color, palette, color grade, illumination direction, warmth, contrast, shadow design, or lighting setup when color or light itself is independently referenced. Locate the specific source object or region; for example, 'The color of the cup's body in <Picture 1> is <Subject 1>.'
    - Pose/action type: a source-derived posture, gesture, gaze, body orientation, action, or interaction that is preserved, changed, or transferred.
    - Logo/symbol/icon type: a graphical emblem, symbol, UI icon, or pictographic mark; handle any readable words under the text rules.
    - Embedded/thumbnail type: a complete or cropped source picture embedded in the target as a panel or thumbnail.
    - Partial-crop type: a target crop or zoom whose visible content comes from a specific region of a source picture.
    These fifteen types form the complete supported taxonomy, not a per-example checklist or quota. Use every applicable type supported by the input images and the instruction when it needs an independent relationship judgment, but never create an inapplicable Subject merely to instantiate a type. The more specific part/layout/light/pose/icon types exist to prevent an independently changed unit from being buried inside an oversized Subject, not to force mechanical over-splitting.
    Brand-new target-only content has no <Subject N>. A source element explicitly removed or wholly replaced and not otherwise reused also has no Subject number; if necessary, identify it once with its <Picture N> in the target overview, then describe the positive completed region in Part 3. Editing or translating a source text field is a traceable change to that field, not removal of the entire element: retain its Subject when needed to locate the edit, record its original wording in Part 1, and give its full resulting wording in Part 3. Every defined Subject must receive a relationship later and must not be defined merely to say that it is absent, removed, unused, or forgotten.
    Readable source text and data require special care. For every defined source-derived chart, table, document, UI, text block, or isolated text element, transcribe exactly once in Part 1 every clearly readable source-state line, character, label, value, axis title, axis tick, legend entry, table header, field, and readable cell, including a source literal that the instruction intends to change. Preserve the original language, script, grouping, and reading order. Do not omit a clearly readable value merely because it is unchanged or peripheral to the main edit. Never guess hidden, cropped-away, or illegible source text.
    For grouped or clustered bars, radar charts, multi-panel plots, and many-point plots, enumerate every clearly readable value rather than summarizing the display. Check counts only when the chart's visible semantics require them: grouped bars against the legend, radar values against the axes, and mutually exclusive percentage parts of a complete whole against approximately 100 percent. Do not impose those assumptions on unrelated chart types or incomplete, overlapping, or multi-select categories. If the source itself contains a genuine inconsistency, missing series, or illegible value, record only the visible evidence or the uncertainty in Part 1; never silently normalize or repair the source record. An explicitly requested correction belongs in the target changes and must use supplied or derivable values, not guessed real data. Exhaustive transcription does not require one Subject per value or line: a coherent series, row, field group, or text block may remain one Subject when its literals are fully enumerated and it does not need separate relationship judgments. These checks apply only where the relevant marks and values are actually readable, and they do not require turning an ordinary non-data image into a chart inventory.
    ## Part 2: Target Overview and Reference Relationships
    After the definitions, write one natural sentence beginning with 'The target image'. Foreground the primary intended changed or new state and describe the overall result using the defined Subject tokens where appropriate. Target-only new content remains unnumbered. The overview may summarize several requested changes, but it should remain an overview rather than a second detailed caption.
    Then give every defined Subject exactly one whole-subject relationship in its own independent clause. Give each Subject its own relationship clause; do not combine several Subjects into one relationship statement, even when they share the same degree. Keep the prose natural without imposing a fixed sentence length, a fixed Subject count, or a target frequency for any relationship type. Use these four relationships:
    - Fully preserved: use only when the whole Subject is unchanged in identity, appearance, count, pose or state, position, topology, and intended target role. Write only the short standalone clause '<Subject N> is fully preserved.' Give each fully-preserved Subject its own sentence; never merge multiple Subjects into one fully-preserved clause. Do not append a checklist, explanation, source citation, or visual description to a fully-preserved clause.
    - Partially preserved: use when the Subject remains traceable but at least one relevant property changes. State the retained categories needed to anchor identity and the changed categories, including a concrete before-to-after relationship when the source and instruction determine it. Do not exhaustively re-render retained appearance.
    - Attribute transfer: use when a specific source-derived property is directly reused or matched on a target carrier. Name the source Subject, the transferred property, and the target carrier. Transfer only the assigned property, not unrelated donor attributes.
    - Weak reference: use when the source guides a general style, palette, atmosphere, composition principle, or similar quality without requiring a specific property to be matched exactly. Name the borrowed quality and make clear what the new design may vary. Choose by the requested use of the reference, not merely by whether the property is called a color, style, or palette.
    Attribute-transfer example: the user wants the coat in <Picture 2> to match the cup's body color in <Picture 1>. Part 1 defines that color as <Subject 1> and the coat as <Subject 2>. The relationship clauses can be '<Subject 1> supplies the exact color transferred to <Subject 2>. <Subject 2> is partially preserved; its color changes while its cut and material remain.' The cup's shape and material are not transferred.
    Weak-reference example: the user wants a new abstract composition inspired by a landscape's palette, without copying it exactly. Part 1 defines the landscape's palette as <Subject 1>. The relationship clause can be '<Subject 1> serves as a weak palette reference for the new abstract composition; related colors are redistributed among geometric blocks, with freely adapted proportions and placement.' The new composition has no Subject number. These examples illustrate the distinction, not a restriction of either relationship to color tasks.
    For a Subject used in several target forms or instances, judge the intended uses together. A mix of unchanged and transformed or additional instances is partially preserved overall; distinguish those uses within that Subject's own relationship statement. Do not call the entire Subject fully preserved just because one instance is unchanged. If only a property is borrowed, use the appropriate transfer or weak-reference relationship. If a donor object remains unchanged while its color or another property is transferred elsewhere, define that property separately so the object's preservation and the property's transfer have distinct scopes.
    The relationship must agree with the described target. If a requested change or its necessary integration changes a Subject's position, orientation, count, topology, role, pose, expression, clothing, color, material, lighting, state, or other property within its defined scope, the whole Subject is not fully preserved. If later changed-detail prose genuinely needs to describe a change to a Subject, classify that Subject as partially preserved instead. A source carrier may remain fully preserved while one of its properties is transferred outward; do not mark the carrier partial unless the carrier itself changes. Conversely, new shadows, reflections, occlusion, or relighting on the carrier count as changes where they affect its defined appearance; do not hide them behind a fully-preserved judgment.
    For partially preserved chart, table, document, UI, and text-block Subjects, state which categories are retained and which change. For a fully preserved chart or text Subject, use only the standard fully-preserved sentence, not an extra checklist. Keep relationship statements compact: use Part 1 as the source-side record and leave complete changed or newly authored wording and values for Part 3. Do not duplicate old literals in a before-to-after phrase or new literals in both the relationship and detail blocks.
    Treat unmentioned source-derived content as stable by default, but do not let that conservative baseline suppress the requested imagination. The changed/new target scope may still be rich. Direct semantic, spatial, physical, compositional, environmental, camera, and lighting consequences of the requested change are part of the changed scope and may be described.
    ## Part 3: Changed and Newly Introduced Target Details Only
    This final logical block is a richly rendered visual delta, not a standalone T2I re-caption of the whole target image. Also supply the requested textual deliverables. For visual description, cover only:
    - the concrete final appearance of source-derived attributes that actually change;
    - target-only people, objects, text regions, environments, panels, charts, layouts, effects, or other content introduced by the instruction;
    - the newly exposed or positively completed surface or space left by a removal, or the positive new occupant of a replacement;
    - plausible supporting details inside the changed/new scope that make it visually recognizable and coherent;
    - direct integration consequences involving scale, orientation, contact, overlap, occlusion, perspective, depth, shadow, reflection, lighting, material response, anatomy, support, camera framing, or composition.
    This changed/new-only rule does not impose terse writing. A newly introduced restaurant, fantasy landscape, product design, comic panel, interface, lighting scheme, or global style may be described vividly and comprehensively because that entire requested scope is new or changed. Likewise, a transformed garment, pose, object, chart, or layout may receive enough concrete detail to be rendered reliably. Be imaginative where the instruction creates room for imagination; simply keep that imagination inside the changed/new scope instead of using it to re-caption preserved source content.
    Strict non-repetition rules:
    - After a Subject is classified as fully preserved, never visually describe that Subject again. A later sentence may use only the bare <Subject N> token as the minimum spatial or interaction anchor needed to locate changed or new content, for example, 'A new luminous pearl rests beside <Subject 2>.' Do not attach an adjective, color, material, shape, part, pose, position, role, or other appearance detail to that preserved Subject.
    - For a partially preserved Subject, Part 3 describes only its changed attributes and their concrete intended final appearance. Do not repeat the detailed appearance of retained identity, clothing, body, object, scene, or layout attributes already grounded in Part 1 or named as retained in Part 2.
    - For attribute transfer or weak reference, describe only how the transferred or borrowed quality is realized on the intended target carrier. Do not import or re-render unrelated donor properties.
    - Do not repeat fully preserved source text or data as visual description; use the Part 1 record and bare <Subject N> references. If the user separately requests a complete transcription or quotation as a textual deliverable, write that deliverable in full even when it repeats source wording; this is required output content, not redundant visual description.
    - Write every changed or target-only text/data field and requested textual deliverable in full. Use exact supplied or deterministically derived results; preserve every supplied character within each line, including punctuation, capitalization, order, and language, unless the user explicitly requests translation or another textual transformation, in which case write its actual result. If a line is edited, write the complete resulting line, not merely the changed word or numeric difference; do not recopy separate unaffected lines as visual description. Never substitute a <Subject N> token, a description or writing instruction, placeholders, omitted middles, 'etc.', 'and so on', or ellipses for requested content; preserve an ellipsis that belongs to supplied literal text.
    - Do not describe an unchanged background, person, object, pose, camera, light, style, or image-quality property merely to make the paragraph resemble a complete T2I caption.
    - If no changed or newly introduced detail remains beyond the overview and relationship statements and no requested textual deliverable remains to be supplied, stop. Never pad the output with preserved appearance.
    For example, for a reference showing a lakeside path and hills, a request to write a scenic introduction requires actual copy such as '沿着湖畔步道缓缓前行，欣赏湖水与远山相映的景色，在悠闲漫步中感受旅途的宁静。', not 'a paragraph introducing the attraction'. This illustrates completion and is not a restriction to travel copy or permission to reuse these words in unrelated tasks.
    Correct pattern: 'The black horse in <Picture 1> is <Subject 1>. The target image shows <Subject 1> beside a newly added luminous pearl. <Subject 1> is fully preserved. The pearl emits a cool glow onto the ground beside <Subject 1>.' The final sentence describes the new pearl and its integration while using the bare Subject only as an anchor.
    Incorrect pattern: '<Subject 1> is fully preserved. <Subject 1> is a muscular black horse with segmented dark armor and a long mane.' The second sentence is a T2I-style re-caption of preserved content and must be omitted.
    Correct partial pattern: '<Subject 1> is partially preserved; its identity and body armor remain, while its bare head changes to an ornate helmet. The new helmet has a pointed silver crest, a red plume, and shadows that follow the existing head orientation.' Do not subsequently re-describe the retained face, body, or armor.
    For exact reference-color or palette matching, use the defined color or palette <Subject N> to specify where it is applied on the target. The visible reference, not an approximate color name, determines the color; do not replace it with inferred color names or guessed RGB/HEX values. Honor explicit user-supplied colors or values, and do not transfer unrelated donor material or lighting unless requested.
    When describing a grid, multi-panel image, storyboard, table-like arrangement, or other row-and-column layout, state the row count and column count explicitly, either in words such as '2 rows and 3 columns' or with labeled multiplication notation such as '2 x 3 (rows x columns)' or '3 x 2 (columns x rows)'. Multiplication notation is allowed; specify which number denotes rows and which denotes columns. Do not use an unlabeled '2 x 3' or '2×3' alone. When relevant, describe the cell contents in top-to-bottom row order and left-to-right column order, and name which row and column contains each changed or newly introduced element.
    For uneven rows, merged cells, or a freeform composition, describe the actual arrangement directly, such as 'the top row has three panels and the bottom row has two', with positions and spans as needed; do not impose a rectangular grid that is not intended. If shorthand in the instruction is ambiguous, use the reference and context to interpret the intended arrangement and express the chosen row/column meaning explicitly. Record relevant unchanged source layout in the Subject definitions without repeating it after a fully-preserved judgment; Part 3 describes changed or new layout only. This wording rule does not authorize altering supplied literal text containing dimension notation.
    ## Final language and self-check
    Write all descriptive and relationship prose in fluent English, regardless of the language of the user instruction. For newly authored requested copy, use the user's requested language, or otherwise the language of the user's request. Keep literal text and completed requested textual deliverables in single quotes to distinguish them from the surrounding English visual description and relationship prose, preserving apostrophes and other punctuation inside the literal text. Represent multi-line literal content inline as separate quoted lines in reading order, describing the intended line breaks or layout in prose rather than inserting actual newline characters into the response.
    Use objective, concrete visual language. Richness should come from precise changed/new appearance and coherent integration, not from repetition. Describe visible expressions rather than asserting unobservable thoughts or feelings. Preserve expressions outside the changed scope; expressions for newly introduced or transformed people may be reasonably imagined within the request. Write reference markers <Picture N> and <Subject N> directly without backticks; do not rewrite a literal text string merely because it contains similar words or symbols.
    Before answering, verify:
    - every source visual fact is visible in a named <Picture N>, and every intended change is supported by the editing instruction or User Aspect Ratio Selection and their necessary consequences; reference aspect ratios come only from the supplied metadata;
    - the output still imagines instruction-introduced content richly enough to be recognizable and renderable;
    - every Subject definition is concise but unambiguous about its source and reference scope, without unnecessary appearance detail or missing required text/data;
    - every Subject is naturally defined before use and has its own independent relationship clause, never merged with another Subject;
    - every fully preserved Subject has its own exact '<Subject N> is fully preserved.' sentence and receives no later visual re-description;
    - no fully preserved parent contains an edited child, and mixed target forms or integration effects are reflected in the relationship;
    - every partially preserved Subject repeats no detailed retained appearance in Part 3;
    - donor attributes do not leak beyond the property requested;
    - target-only and removed-only content has no Subject number;
    - every applicable reference-element type is supported without treating the taxonomy as a per-example quota;
    - source text/data is recorded once in Part 1 without guessing, repairing, or translating the source record;
    - all requested target text and textual deliverables are complete and literal, with source repetition only for an explicitly requested transcription or quotation;
    - each textual deliverable has the requested image or text-only destination;
    - Part 3's visual description contains changed/new content and direct integration only, with no whole-frame T2I padding;
    - all non-literal descriptive and relationship prose is English, while quoted source, user-specified, or newly authored requested text uses its required language and script;
    - any grid layout described makes the row count and column count unambiguous, either in words or with explicitly labeled multiplication notation, with positions or uneven-row structure clarified where needed;
    - expanded_caption is one fluent, complete paragraph with no structural meta-language in the narration; multi-line literals retain their wording, reading order, and intended line boundaries through inline quoted lines;
    - target_aspect_ratio copies any concrete user selection exactly; in adaptive mode, it preserves the main reference ratio when no canvas-shape change is required and reflects any requested expansion or cropping; expanded_caption and its preservation relationships agree with the final canvas;
    - the final answer is a valid JSON object containing only the two string fields target_aspect_ratio and expanded_caption.
    Output only the required two-field JSON object.
    User input: {input text}
    output_text:
    For MiniMax H3 Image's PE model it’s not tied to any specific model
    [MiniMax H3 Image PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) repo is using `Qwen3.8-27B, FP4` and we are using Qwen3.5 9B in this workflow


---

## TEMPLATE: image_minimax_h3_image_character_swap (https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_character_swap.json)

```
SaveImage []: ["image_minimax_h3_image_character_swap"]
MarkdownNote [Note: size and reference images]: ["Two pictures go in: **Load Image (Source Image)** is the picture to edit and reads as `<Picture 1>`; **Load Image (Reference Character)** is the character to carry over and reads as `<Picture 2>`.\n\nThe source is resized to **1 MP** (`scale total pixels`) and then to a multiple of 32, and that size becomes the canvas and the output size, 864 × 1152 with the example images. 1 MP is what the adapter's author tested and recommends. This workflow al
MarkdownNote [Model Links]: ["## Input Assets\n\n- [h3_char_swap_source.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_char_swap_source.png)\n- [h3_char_swap_reference.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_char_swap_reference.png)\n\n## Model Links\n\n- [Hugging Face:Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)\n- [Hugging Face:Comfy-Org/MiniMax-H3](https://huggi
UNETLoader []: ["minimax_h3_image_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_int8_convrot.safetensors","minimax","default"]
VAELoader []: ["minimax_h3_image_vae_fp16.safetensors"]
KSampler []: [142232601287349,"randomize",6,1,"euler","simple",1]
PrimitiveStringMultiline [Text (user_prompt)]: ["Swap the human in <Picture 1> with human <Picture 2>, also update the human in the mirror"]
LoadImage [Load Image (Source Image)]: ["h3_char_swap_source.png","image"]
ResizeImageMaskNode []: ["scale to multiple",32,"area"]
ResizeImageMaskNode []: ["scale total pixels",1,"area"]
ImageCompare []: []
LoadLoraModel []: [1,"h3_image_character_swap_v1.safetensors",1,true]
LoadImage [Load Image (Reference Character)]: ["h3_char_swap_reference.png","image"]
MiniMaxH3ImageConditioning []: ["",0,0]
```

### Notes and long text widgets (unescaped)

    Two pictures go in: **Load Image (Source Image)** is the picture to edit and reads as `<Picture 1>`; **Load Image (Reference Character)** is the character to carry over and reads as `<Picture 2>`.
    
    The source is resized to **1 MP** (`scale total pixels`) and then to a multiple of 32, and that size becomes the canvas and the output size, 864 × 1152 with the example images. 1 MP is what the adapter's author tested and recommends. This workflow also holds up at 16 MP (`scale total pixels` = 16), at more VRAM and more time. The reference picture is scaled internally, so its own resolution only needs to be reasonable.
    **No prompt enhancer**: the adapter was trained on short editing instructions, so write the instruction yourself and keep the `<Picture n>` references. The author's own prompt, shortened:
    > Swap only the red-haired woman in the dark high-collared jacket in the close foreground in `<Picture 1>` with the character from `<Picture 2>`. Keep the original scene, framing, other people, and the target’s placement and action.
    For more references, add `Load Image` nodes, connect them to the conditioning node and name them `<Picture 3>`, `<Picture 4>`, ...
    ## Input Assets
    - [h3_char_swap_source.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_char_swap_source.png)
    - [h3_char_swap_reference.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/h3_char_swap_reference.png)
    ## Model Links
    - [Hugging Face:Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)
    - [Hugging Face:Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
    ### Alternative Repos
    - [ModelScope:Comfy-Org/MiniMax-H3-Image](https://modelscope.cn/models/Comfy-Org/MiniMax-H3-Image)
    - [ModelScope:Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)
    **diffusion_models**
    - [minimax_h3_image_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors) (21 GB)
    **text_encoders**
    - [qwen3vl_32b_minimax_h3_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) (25.28 GB)
    **vae**
    - [minimax_h3_image_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors) (1.17 GB)
    **loras**
    - [h3_image_character_swap_v1.safetensors](https://huggingface.co/akatz-ai/MiniMax-H3-Image-Character-Swap-LoRA/resolve/main/h3_image_character_swap_v1.safetensors) (155 MB)
    ## Adapter
    The character-swap adapter is published in its original repo, [akatz-ai/MiniMax-H3-Image-Character-Swap-LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Image-Character-Swap-LoRA) (Akatz). It is a rank-16 LoRA trained with AI Toolkit on 140 paired edits and evaluated on the INT8 ConvRot checkpoint above; that repo holds the still-image examples, their prompts and the adapter license. Adapter use follows the MiniMax H3 Image Community License Agreement.
    ## Model Storage Location
    ```
    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 diffusion_models/
    │   │   └── minimax_h3_image_pruned_int8_convrot.safetensors
    │   ├── 📂 loras/
    │   │   └── h3_image_character_swap_v1.safetensors
    │   ├── 📂 text_encoders/
    │   │   └── qwen3vl_32b_minimax_h3_int8_convrot.safetensors
    │   └── 📂 vae/
    │       └── minimax_h3_image_vae_fp16.safetensors
    ## Report Issue
    Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models. Desktop/Cloud updates follow stable releases, so some nightly-supported models may not be available yet.
    - Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
    - UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
    - Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)


---

## BLUEPRINT (ComfyUI core): blueprints/Image to Video (MiniMax H3).json (https://github.com/Comfy-Org/ComfyUI/blob/master/blueprints/)

```
d70595de-536a-4124-9bf3-d5c9ee60c978 [Image to Video (MiniMax H3)]: ["",1344,768,5,757358688076805,"minimax_h3_fl2va_pruned_int8_convrot.safetensors","qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax_h3_video_vae_fp16.safetensors","minimax_h3_audio_vae_fp32.safetensors",false,"minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",1,8]
VAELoader []: ["minimax_h3_video_vae_fp16.safetensors"]
VAELoader []: ["minimax_h3_audio_vae_fp32.safetensors"]
KSamplerSelect []: ["res_multistep"]
BasicScheduler []: ["simple",4,1]
UNETLoader []: ["minimax_h3_fl2va_pruned_int8_convrot.safetensors","default"]
CLIPLoader []: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax","default"]
RandomNoise []: [1,"fixed"]
CreateVideo []: [24,8,"sRGB"]
MiniMaxH3ImageToVideo []: ["Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.\n\nTimeline:\n[0s-1s] VHS static opens the frame, the title \"COMFYUI\" appears with RGB split and a slight horizontal jitter.\n[1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.\n[2.5s-4s] Clean \"STARRING\" credits appear,
ComfyMathExpression []: ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"]
PrimitiveFloat [Float (duration)]: [2]
LoraLoaderModelOnly []: ["minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",1]
ComfySwitchNode [If/Else Switch (Model)]: [false]
ComfySwitchNode [If/Else Switch (Steps)]: [false]
PrimitiveInt []: [20,"fixed"]
PrimitiveInt []: [6,"fixed"]
PrimitiveBoolean [Boolean (Enable Lightning LoRA)]: [false]
```
    Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, mood languid and nostalgic.
    
    Timeline:
    [0s-1s] VHS static opens the frame, the title "COMFYUI" appears with RGB split and a slight horizontal jitter.
    [1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.
    [2.5s-4s] Clean "STARRING" credits appear, "LATENT" and "CONTROLNET" each shown exactly once.
    [4s-5s] Final card "DIRECTED BY COMFYUI" holds, one VHS tracking glitch settling into stability.
    Hard cuts only, transitions landing with tape jumps, no push-ins, no dissolves.
    Audio: lo-fi vaporwave score, slow drum machine with soft bass, VHS tape-noise sample joins at 2.5s, melody fading for the last 1s.
    All text must be clearly legible, do not misspell English, no Chinese characters, do not repeat names or job titles, no soft dissolves, no subtitle bars.
