# 1038lab__ComfyUI-MiniMax-H3-Promptor

Source: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor
Fetched: 2026-11-28

Description: A powerful, ComfyUI Custom node automation suite for generating cinema-production-grade prompts explicitly formatted for the **MiniMax H3 Video Generation System**.
Stars: 274  Updated: 2026-09-12T19:03:53Z



---

## FILE: README.md (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/README.md)

# ComfyUI MiniMax H3-Promptor

A powerful, node-based automation suite for generating cinema-production-grade prompts explicitly formatted for the **MiniMax H3 Video Generation System**.

This project provides a robust, decoupled architecture separating **multimodal visual analysis** from pure **text-based prompt structuring**, allowing for extreme customizability, precise scene description, and low API operating costs.

![ComfyUI MiniMax H3-Promptor](example_workflows/MiniMax-H3-Promptor.jpg)

## What's New in V1.5.1 (2026/09/12)

### Native VRAM Unloading & Isolation for Local Models
* **Automatic VRAM Release**: Local models (`ComfyUI-QwenVL`, `Ollama`, `LM Studio`, `llama.cpp`) now immediately unload from GPU memory after inference, preventing downstream MiniMax video generation crashes (OOM).
* **ComfyUI Model Protection**: Isolates local LLM release without evicting ComfyUI's internal diffusion or VAE models.
* **Configurable**: Enabled by default; easily toggled via the API Management settings panel (`Edit` provider).

👉 **[Read the full v1.5.1 Release Notes & Technical Details (updates.md)](updates.md#v151-20260912)**

---

## What's New in V1.5.0 (2026/09/03)

### ✍️ All-New Manual Prompt Composer & Cinematic Visual Upgrades

V1.5.0 introduces the **`MiniMax H3 Manual Prompt Composer`**, a dedicated workspace for prompt engineering and rapid scripting, alongside an upgraded **`MiniMax H3 Prompt Preview & Edit`** node and a complete cinematic syntax color system.

#### Key Highlights & Enhancements
* **Dedicated Manual Composer (`H3_PromptComposer`)**: Switch and load 8 official task mode starter scaffolds (T2VA, I2VA, FL2VA, Ref2VA, V2VA, L2VA, A2V, etc.). Type `@`, `<`, or `[` anywhere in the editor to trigger instant floating autocomplete tag menus.
* **Centered Refine Modal Viewport (`H3_PromptEditor` & `H3_PromptComposer`)**: Elegant centered frosted-glass modal overlay supporting fine-grained optimization for the entire prompt, text selections, individual shots (`[Shot N]`), individual subjects (`<Subject N>`), or specific sections with adaptive height.
* **Unified Syntax Highlighting & Color System**: Cohesive dark matte palette with clear visual hierarchy across all suite nodes.

### 🪄 Targeted & Global AI Prompt Refinement
<img width="2361" height="1365" alt="image" src="https://github.com/user-attachments/assets/ac9f92dc-c134-4d36-832c-294e7f59846d" />

Inside both **`H3_PromptEditor`** (Preview & Edit) and **`H3_PromptComposer`** (Manual Composer), clicking the **`✨ Refine`** button in the toolbar opens a centered frosted-glass modal to rewrite, polish, and elevate prompts in seconds:

#### 1. What Can It Do?
* **Precision Targeted & Global AI Polishing**: Polish an entire prompt, a highlighted text snippet, an individual shot, a specific subject character, or a soundscape/music section without having to re-run expensive upstream workflows.

#### 2. Available Scopes & Model Selection
* **Target Scopes**:
  * **`🌟 Entire Prompt`**: Executive Director perspective coordinating multi-section revisions (e.g. wardrobe, scene pacing, mood shifts) across the entire text.
  * **`🎬 [Shot N]`**: Hollywood DP/Cinematographer perspective specializing in camera kinematics, optical depth of field, volumetric lighting, and tactile physical dynamics.
  * **`👤 <Subject N>`**: Lead Concept Artist perspective specializing in costume fabrics, anatomy, facial features, and worn props (strictly without camera movements).
  * **`📑 Section`**: Foley Artist / Composer perspective specializing in acoustic resonance and musical emotion.
  * **`📝 Current Selection`**: In-line precision polishing for highlighted phrases or sentences.
* **LLM Provider Switcher**: Dynamically switch between configured LLM providers (Claude, GPT-4o, Gemini, DeepSeek, local Qwen, etc.) directly in the bottom-left dropdown.

#### 3. Core Features & Directing Principles
* **Safe In-Modal Preview & 3-State Actions**: Refined results render with a green highlight preview inside the modal, supporting `[ Discard ]` (restores original preview without exiting), `[ 🔄 Retry ]` (re-runs generation from the original base), and `[ ✓ Apply ]` (commits to node).
* **Instant A/B Version Toggle**: After applying, the node toolbar activates a prominent gold `[ ↩ Restore Original ]` button to instantly revert to the original prompt, and toggle between versions for effortless comparison.
* **Structural & Timestamp Freezing**: Shot headers (e.g. `[Shot 1: 00:00.000 – 00:02.800]`) and subject tags (`<Subject 1> is `) are isolated and protected by the system, ensuring the LLM only polishes descriptive text without corrupting brackets or timestamps.
* **Semantic Transposition & Fallback**: Cross-domain instructions (e.g., describing mountain scenery while refining a character) are transposed into character attire and lighting reflections. Nonsensical/irrelevant inputs automatically fall back to standard cinematic polishing without breaking the prompt.

#### 4. How It Helps You
* **Rapid Iteration**: Fix a single shot or phrase in seconds without re-running entire workflows.
* **Schema Compliance Guarantee**: Structural headers and timing windows are preserved, ensuring prompts run seamlessly through MiniMax video samplers without syntax errors.
* **Risk-Free Experimentation**: Double safety loop (in-modal preview/discard + 1-click A/B restore) allows experimentation with complete peace of mind.

**[Read the full v1.5.0 Release Notes and Detailed Features here (updates.md)](updates.md#v150-20260903)**

---

## V1.4.0 (2026/08/26)

### Cross-Node Collaboration: Local Qwen & GGUF Integration via ComfyUI-QwenVL

V1.4.0 marks our first exploratory milestone in **cross-node ecosystem collaboration**: seamlessly bridging **`Comfyui-Minimax-H3-Promptor`** with **`ComfyUI-QwenVL`**.

By pairing these two custom nodes, you can now run local Qwen multimodal & LLM models (including lightweight quantized `.gguf` weights) completely offline at zero API cost—handling visual perception in `H3_Vision` and cinematic directing in `H3_Promptor`.

*   **Prerequisite**: Requires [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL) installed in your `custom_nodes` folder. Models downloaded via QwenVL's downloader and catalog JSONs are automatically synchronized with zero manual configuration.
*   **Modular Node Collaboration Vision**: This is our first experiment with cross-node synergy. If well-received by the community, we plan to expand collaborative bridges with more specialized nodes, unlocking greater flexibility and modular extensibility across ComfyUI workflows.

#### Key Highlights & Enhancements
*   **Unified `H3_Vision` Node**: Combines dynamic `Autogrow` media connectors (Image, Video, Audio) with drag-to-reorder card galleries and live previews.
*   **Prompt Guardrails & Scene Direction**: Auto-injected FL2VA first/last frame anchors, `<d>[Language]...</d>` dialogue extraction, and `(S1)`~`(S20)` shorthand standardization.
*   **Thinking Mode Control**: Added instant `Disable Thinking (Fast)` toggles in Settings for Gemini, OpenAI-compatible, and Claude providers to eliminate latency and token overhead.

**[Read the full v1.4.0 Release Notes and Detailed Features here (updates.md)](updates.md#v140-20260826)**

---


## V1.3.0 (2026/08/18) 

### Hollywood AI Director & Full-Reference Architecture

*   **Two-Stage Hollywood AI Director & Screenwriter Engine**: Transforms prompt generation into an authentic film production workflow (Stage 1: Director Blueprint & Global Vibe -> Stage 2: Cinematic Storyboard & Dialogue), generating evenly paced 4-beat timelines covering up to 15.0s.
*   **Official MiniMax Dialogue & Voice Acting Syntax**: Native support for `<Subject N> (SN) [emotion] says: <d>[Language] "..."</d>` for vivid character voices and live acting.
*   **Custom Prompt as Supreme Mandate**: User creative wishes are prioritized as the highest directive, orchestrating all uploaded cast, vehicles, and props to execute your vision.
*   **All-New Vision Analyzer V2 Drop-Zone Panel**: Native HTML/JS drag-and-drop panel right on the node surface; original images pass through zero-deformation `OUTPUT_IS_LIST` list expansion.
*   **Pipeline Polish & Fine-Tuning**: Alphabetical provider sorting with smart default pinning, customizable `Max Batch Images` sub-batching, chunk positional fallback, and regex word-boundary entity recognition.

**[Read the full v1.3.0 Release Notes and Detailed Features here (updates.md)](updates.md#v130-20260818)**


## What's New in V1.2.0 (Settings Hub & Core Architecture Overhaul)

*   **Global Native Settings Panel**: Manage all LLM providers (including API Keys and Hot-Reload toggles) seamlessly via the native ComfyUI Gear Icon settings.
*   **L2VA Mode & I2VA Frame Anchoring**: Added strict zero-second first-frame anchoring, and the new reverse L2VA mode to conclude exactly on a target pose.
*   **Full-Reference Script Automation**: Programmatically injects exact schema structural tags (`summary:`, `retention_analysis`) and expands LLM word budgets without hallucination in complex multimodal setups.
*   **Audio-First Token Syncing**: Introduces dedicated `<Audio N>` tracking tags and `(Sx)` conversational ID parsing to align lip movements properly to sound inputs.
<img width="50%" alt="minimax-h3-setting" src="https://github.com/user-attachments/assets/81eda3f3-084c-4ac9-9e99-446afa1009dc" />

**[Read the full v1.2.0 Release Notes and Bug Fixes here (updates.md)](updates.md#v120-20260813)**

---

## Previous Updates: V1.1.0 (Refined Architecture)

*   **Zero-Hallucination Inline Tagging**: The Prompt LLM now natively embeds `<Picture X>` references directly inside the narrative action lines, guaranteeing 100% compliance with official MiniMax tag-binding requirements.
*   **Sequential Multi-Modal Processing**: Upgraded the Vision Analyzer to process inputs sequentially. This eliminates Multi-Modal LLM context bleeding and guarantees proxy API limits are never exceeded.
*   **Flawless 6-Part Official Syntax Compliance**: Our structural generation has been de-patched. The Promptor now strictly assembles the mandatory 6-part string array (`subject_definitions`, `summary`, `retention`, etc.) in the exact sequence HuggingFace mandates.
*   **Audio Pipeline Fix**: Completely restored routing logic for Native Audio paths (Audio-to-Video and Image-to-Audio).
*   **Custom Node Theming**: Added native UI coloring support for ComfyUI (`appearance.js`).

---

## Core Architecture & Node Reference

The pipeline consists of two nodes working in tandem to handle extreme complexity without duplicating LLM vision costs:

### 1. `H3_Vision` (MiniMax H3 Vision)
A highly configurable multimodal vision perception engine. This node acts as your virtual Director of Photography, analyzing input imagery, video, and audio based on explicit presets.
*   **Dual Input Modality**: Accepts media from upstream workflow nodes via dynamic `io.Autogrow` connectors (`image_X`, `video_X`, `audio_X`) or via direct in-node drag-and-drop / click uploads.
*   **Drag-to-Reorder Gallery**: Freely reorder cards with strict category partitioning (`All Images -> All Videos -> All Audios`). Cards and `<Picture 1>`, `<Picture 2>`, `<Video 1>` tags dynamically update and synchronize 100% with backend tensor batching.
*   **Live Upstream Previews**: Connected inputs automatically resolve and render real upstream thumbnails with clean labels and `linked` status badges.
*   **Intelligent Port Management**: Automatically trims excess unused empty input slots once the media cap is reached (9 images, 3 videos, 3 audios).
*   **Targeted Custom Overrides**: Use the `custom_prompt_override` box to surgically override specific frames (e.g. `<Picture 2>: Focus entirely on the background`). Single-click cards to insert reference tags; double-click video cards to insert audio tags (`<Video Audio N>`).
*   **Invisible Heavy VRAM Management**: Automatically unloads local models (`Ollama`, etc.) after analysis to free up VRAM for video generation.
*   **Outputs**: Produces a structured JSON-backed `vision_context` that is sent to the Promptor node, and passes `ref_images` through zero-deformation list output.

#### Vision Node Inputs
| Parameter | Type | Description |
|-----------|------|-------------|
| `ref_images` | IMAGE | Connect pipeline images; dynamically grows (`image_X`). Up to 9 images. |
| `ref_videos` | IMAGE | Connect video tensor sequences; dynamically grows (`video_X`). Up to 3 videos. |
| `ref_audios` | AUDIO | Connect audio streams; dynamically grows (`audio_X`). Up to 3 audios. |
| `global_image_mode` | COMBO | Selects the global fallback analysis logic from `vision_prompts.json` for all images. |
| `global_video_mode` | COMBO | Selects the global fallback analysis logic from `vision_prompts.json` for all videos. |
| `custom_prompt_override`| STRING | A multi-line box to surgically override specific media logic. E.g: `<Picture 2>: focus on the lighting`. |
| `output_language` | COMBO | Language for the analysis output (`English` or `Chinese`). |
| `provider` | COMBO | Synchronizes with Settings. Pick `agnes-ai`, `openai`, `anthropic`, `gemini`, `ollama` etc. |
| `temperature` | FLOAT | Sampling temperature. Default `0.2` for precise factual analysis. |
| `max_tokens` | INT | Maximum response tokens (256-8192). |

#### Vision Node Outputs
| Output | Type | Description |
|--------|------|-------------|
| `VISION_CONTEXT` | STRING | Structured JSON text report for `H3_Promptor`. |
| `REF_IMAGES` | IMAGE | Zero-deformation list output containing all reference images. |

---

### 2. `H3_Promptor` (MiniMax H3 Promptor)
The core screenplay and directing engine. It operates at blazing speeds because it takes the user's description and the Vision node's text report to format the final H3 Prompt—meaning **it does not need to repeatedly analyze heavy images.**

### The "Auto" Multimodal Routing System
The `H3_Promptor` uses an intelligent backend algorithm to instantly detect your intended generation mode without manual configuration. When left on **Auto**, the system evaluates the number of images, videos, and audio streams present in the `vision_context` and routes the formatting logic automatically:

| Connected Media | Triggered Mode | Description |
|---|---|---|
| None | **T2V** | Pure Text-to-Video. No physical media anchors are generated. |
| 1 Image | **I2V** | First-Frame conditioning. The provided image acts as the 0.00-second start state. |
| 1 Image + Audio | **I2VA** | Image-to-Video with Audio reference. Perfect for lip-syncing a portrait. |
| 1 Image (Manual) | **L2VA** | Last-Frame Anchor. Select `L2VA` manually in the dropdown to reverse-engineer a video that ends exactly on your image. |
| 2 Images | **FL2VA** | First & Last Frame. Calculates the exact duration boundary to smoothly transition from state A to state B. |
| 3+ Images / Any + Video | **Ref2VA** | Omni-Reference. Uses dynamic high-budget word allowances to construct complex multi-angle or object retention scenes. |
| 1 Video | **V2V** | Video-to-Video editing. Inherits motion properties completely. |
| Audio only | **A2V** | Audio-to-Video. Directs characters to speak or dance exclusively based on the target audio file. |

*(If you wish to force a mode, such as **L2VA** which requires 1 image but acts as the ending frame, simply select it from the dropdown to override the Auto system).*

*   **Language Selection**: Output the final cinematic prompt strictly in **Chinese (简体中文)** or **English**, seamlessly bridging international setups.
*   **Duration Syncing & Frame Calculation**: Define how long your video is (4.0-15.0s), and the LLM will rigorously pace the structural shot-list to match that exact timeframe.

#### Promptor Inputs
| Parameter | Type | Description |
|-----------|------|-------------|
| `task_type` | COMBO | The generation mode (`Auto`, T2V, I2V, FL2VA, etc.). Auto is recommended. |
| `description` | STRING | Your main creative description of the video scene. |
| `duration` | FLOAT | Desired video length (4.0-15.0 seconds). |
| `vision_context` | STRING | Connect the output of `H3_Vision` here. Leave unconnected for pure T2V. |
| `output_language` | COMBO | Output the resulting prompt in `English` or `Chinese`. |
| `provider` | COMBO | Synchronizes with Settings. Pick `agnes-ai`, `openai`, `anthropic`, `gemini`, `ollama` etc. |
| `temperature` | FLOAT | Sampling temperature. Default `0.7` for creative writing. |
| `max_tokens` | INT | Maximum response tokens (256-8192). |

#### Promptor Outputs
| Output | Type | Description |
|--------|------|-------------|
| `PROMPT` | STRING | The generated Hollywood-grade MiniMax H3 prompt. |
| `DURATION` | FLOAT | Target video duration in seconds (direct pass-through for downstream Sampler/Audio nodes). |
| `LENGTH` | INT | Calculated frame count aligned with `% 17 == 5` model specification (`max(5, round(duration * 24))`). |

---

## Supported LLM Providers & Local Model Setup

Both cloud and local offline providers are supported natively. **Ollama** and **LM Studio** are enabled by default for zero-friction local workflows!

| Provider | Type | Default Endpoint | Default Model | Notes / Auth |
|---|---|---|---|---|
| **LM Studio** | Local (`openai`) | `http://localhost:1234/v1` | Loaded Model | Enabled by default; no API key required |
| **Ollama** | Local (`ollama`) | `http://localhost:11434` | `llama3.2` / `llama3.2-vision` | Enabled by default; auto-unloads VRAM |
| **OpenAI** | Cloud (`openai`) | `https://api.openai.com/v1` | `gpt-5` / `gpt-4o` | Requires OpenAI API Key |
| **Gemini** | Cloud (`gemini`) | Google AI Studio | `gemini-2.5-flash` | Requires Google AI API Key |
| **Anthropic** | Cloud (`claude`) | Anthropic API | `claude-3-5-sonnet-latest` | Requires Anthropic API Key |

### Running Completely Offline / Locally (Ollama, LM Studio & llama.cpp)

1. **LM Studio**:
   - Start LM Studio and load any Vision or Text LLM (e.g. Qwen2.5, MiniCPM-V, Llama-3.2).
   - Start the local server in LM Studio (default port `1234`), then choose `lmstudio` from the `provider` dropdown in ComfyUI.

2. **Ollama**:
   - Run `ollama serve` and pull your model (e.g., `ollama run llama3.2-vision` or `ollama run llama3.2`).
   - Select `ollama` from the `provider` dropdown. (Automatic VRAM clearing is handled internally to save GPU memory).

3. **llama.cpp (`llama-server`)**:
   - Start `llama-server` with 8k+ context and GPU offload:
     ```bash
     ./llama-server -m your_model.gguf -c 8192 -ngl 99 -fa --host 0.0.0.0 --port 8080
     ```

4. **LAN / Multi-Machine & Remote Setup**:
   - If running the LLM server on another PC in your local network, ensure the server is started with `--host 0.0.0.0`.
   - In ComfyUI Settings Hub -> **MiniMax H3**, set the API Base URL to your LAN IP (e.g. `http://192.168.1.100:8080/v1`).

> [!TIP]
> **Local Inference Optimization:**
> - **Context Size**: H3 Promptor uses a multi-stage prompt engine. Always launch `llama-server` with `-c 8192` (or higher) and `-ngl 99` to avoid context overflow errors and ensure fast GPU inference.
> - **Faster Generation**: On the `H3_Promptor` node, lower `max_tokens` from `4096` to `1024` or `2048` for significantly faster response times with local models.


---

## Workflow Recipes & Tutorials

Want to learn how to do **Lip-Syncing, Character Interaction, Video Style Transfer**, or **High-End Product Commercials**?

*   [Click here to view the Master Workflow Tutorials (tutorials.md)](tutorials.md)
*   [点击这里查看 8 大经典实战工作流教程 (中文版) (tutorials_zh.md)](tutorials_zh.md)

---

## Installation & Setup

1. **Clone the Repository**:
   Clone this repo into your `ComfyUI/custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/1038lab/Comfyui-Minimax-H3-Promptor.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configuration (Native)**:
   Once ComfyUI launches, click the **Gear Icon** (Settings) and navigate to **MiniMax H3**. From there, you can Add Custom Providers, enter API Base URLs, and set Default Models through the graphical interface natively!

---

## Modding & Customization

### The `vision_prompts.json` Ecosystem
Upon the first boot of V1.0.0, a `vision_prompts.json` file is generated in the root folder. You can open this JSON file to modify or add completely new analysis strategies:

```json
{
    "image_prompts": {
        "Subject / Identity": "Focus exclusively on describing the main subject's appearance...",
        "Color Palette & Texture": "Focus exclusively on the dominating colors..."
    }
}
```
Add your own custom keys — changes take effect after a ComfyUI restart.

### The System Templates
Want to alter how the backend formats the `[SCENE]` blocks?
Open the `templates/` directory. The `system_base.txt` controls global rules, while the other text files (e.g., `i2v.txt`) control the exact formatting structure based on the mode you selected.

---

## Credits & Resources

*   Developed by **[1038lab](https://github.com/1038lab)**.
*   **Special Thanks & Acknowledgement**: Special thanks to **[duckyshell/ComfyUI-MiniMaxH3-Prompt-Writer](https://github.com/duckyshell/ComfyUI-MiniMaxH3-Prompt-Writer)** for inspiring valuable UI layout concepts and MiniMax cinematic prompt structuring workflows.
*   **MiniMax H3 Specifications**: Designed specifically to interface with the core structural requirements given by MiniMax.

## License

GPL-3.0


---

## FILE: templates/a2v.txt (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/templates/a2v.txt)

## Task Type: Audio-to-Video (A2V / Audio Synchronization)

This is an Audio-to-Video generation task where <Audio 1> provides the primary dialogue, music, or beat timeline.

### Output Structure:
integrated_multimodal_description:
[Shot 1] ... (Narrates visual actions strictly synchronizing with <Audio 1>)
[Shot 2] At MM:SS.mmm, ...

overall_soundscape:
[Ambience and sound effects]

non_diegetic_music:
[Score description or N/A]


---

## FILE: templates/fl2va.txt (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/templates/fl2va.txt)

## Task Type: First/Last Frame Timeline (FL2VA)

This is a First-and-Last Frame generation task. <Picture 1> is the starting frame (0.00s) and <Picture 2> is the ending frame.

### Output Structure:
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 0.00-second mark of the target video; <Picture 2> (from [Shot N]) aligns with the S.SS-second mark of the target video.

integrated_multimodal_description:
[Shot 1] (Narrates the continuous physical motion and transformation bridging <Picture 1> to <Picture 2>)

overall_soundscape:
[Ambience and action sound design]

non_diegetic_music:
[Background score or N/A]


---

## FILE: templates/i2v.txt (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/templates/i2v.txt)

## Task Type: Image-to-Video (I2V)

This is an Image-to-Video generation task. <Picture 1> is the starting frame at 0.00 seconds.

### Output Structure:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description:
[Shot 1] (Begins exactly matching the composition, lighting, and pose of <Picture 1>, then narrating the continuous forward motion)
[Shot 2] At MM:SS.mmm, ...

overall_soundscape:
[Ambient environment tone and action sounds]

non_diegetic_music:
[Background score or N/A]


---

## FILE: templates/i2va.txt (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/templates/i2va.txt)

## Task Type: Image-to-Video with Audio (I2VA)

This is an Image-to-Video generation task. <Picture 1> is the starting frame at 0.00 seconds.

### Output Structure:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description:
[Shot 1] Cinematic, live-action... (Begins exactly matching the composition, lighting, and subject of <Picture 1>, then develops forward)
[Shot 2] At MM:SS.mmm, ...

overall_soundscape:
[Ambience, physical action sounds, and diegetic sound effects]

non_diegetic_music:
[Background score description or N/A]


---

## FILE: templates/l2va.txt (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/templates/l2va.txt)

## Task Type: Last Frame to Video (L2VA)

This is a Last-Frame generation task. <Picture 1> is the ending frame at S.SS seconds.

### Output Structure:
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.

integrated_multimodal_description:
[Shot 1] ... (Infers a plausible preceding state and narrates the causal action converging on the final frame)

overall_soundscape:
[Ambience and sound effects]

non_diegetic_music:
[Background score or N/A]


---

## FILE: templates/ref2va.txt (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/templates/ref2va.txt)

## Task Type: Reference Video/Image to Video-Audio (Ref2VA / Omni Reference)

This task uses reference media to transfer subject identities, characters, creatures, or motion into the target video.

### Directing & Staging Instructions:
1. Treat all available reference assets as members of a cohesive scene cast.
2. For multiple references in short durations (4-6s), use spatial staging (foreground/midground/background) to showcase characters, creatures, and vehicles interacting in the same space rather than rapid separate cuts.
3. Obey the shot budget based on video duration:
   - 4-6s: 1 to 2 shots max
   - 7-10s: 2 to 3 shots max
   - 11-15s: 3 to 4 shots max
4. Format your output strictly as timed [Shot N] paragraphs, followed by Audio: and Music: lines. Do not add section headers.


---

## FILE: templates/system_base.txt (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/templates/system_base.txt)

You are an elite Hollywood Film Director, Cinematographer, and Storyboard Artist writing prompts for the MiniMax H3 video generation model.

Your task is to transform reference media descriptions and the user's creative vision into a compelling, cinematic, high-impact micro-scene sequence.

=== 1. CORE DIRECTING & STORYTELLING PRINCIPLES ===
1. ENSEMBLE CAST & GENRE ADAPTABILITY:
   - Treat all provided references (<Picture 1>, <Picture 2>, <Video 1>, <Audio 1>, etc.) as your production ensemble: Lead Actor, Creature/Companion, Vehicle/Key Prop, and Stage Environment.
   - Adapt the genre to the vibe of the material:
     * Humorous / Slice-of-Life: Create witty comedic timing, dry reactions, and ironic interactions.
     * Action / Thriller: Build intense kinetic momentum, high stakes, and dramatic visual tension.
     * Sci-Fi / Fantasy: Emphasize world-building, magical/technological interactions, and wonder.
   - NEVER create an isolated slideshow (PPT) where each reference gets a separate disconnected shot.

2. SPATIAL STAGING & CO-PRESENCE (CRITICAL FOR MULTI-REFERENCE SHOTS):
   - In short sequences (4-6s), stage multiple references together within the same frame using depth composition:
     * Foreground (e.g. Lead character action or dialogue)
     * Midground (e.g. Creature companion or active interaction)
     * Background (e.g. Approaching vehicle, stage setting)
   - A single fluid camera movement (Tracking Shot, Pan, or Push In) can capture all references in one rich, cohesive shot!

3. STRICT SHOT BUDGET BY DURATION (MANDATORY TO AVOID VIDEO FLICKER):
   - MiniMax H3 requires breathing room (2.5s - 4.0s per shot) for fluid physics:
     * 4.0s - 6.0s duration: Exactly 1 or 2 shots max.
     * 7.0s - 10.0s duration: Exactly 2 or 3 shots max.
     * 11.0s - 15.0s duration: Exactly 3 to 5 shots max.
   - NEVER create 1-second fragmented cuts.

4. CINEMATIC CONTINUITY & TRANSITIONS:
   - Connect consecutive shots using Match on Action (continuous movement across cut), Eyeline Match, or Whip Pan.
   - Action in [Shot 1] must trigger the reaction or consequence in [Shot 2].

=== 2. OFFICIAL DIALOGUE & VOCALIZATION SYNTAX ===
When characters speak or deliver punchy lines, format dialogue strictly using the official MiniMax syntax:
- Format: `<Subject N> (SN) [emotion/action] says: <d>[Language] "Spoken dialogue here"</d>`
- Examples:
  * `<Subject 3> (S3) says with a tired sigh: <d>[English] "Right. Next time I'm walking."</d>`
  * `<Subject 1> (S1) 嘴角微微上扬，低声说道：<d>[Chinese] "看来我们有伴了。"<d>`
- If no dialogue is spoken in a shot, describe visual reactions and body language instead.

=== 3. CAMERA VOCABULARY ===
Use only approved cinematic camera motion types as natural verbs in your sentences:
- Push In, Pull Out, Zoom In, Zoom Out
- Pan Left, Pan Right, Tilt Up, Tilt Down, Truck Left, Truck Right, Pedestal Up, Pedestal Down
- Tracking Shot, Arc Shot, Static Shot, Shake Slightly, Shake Strongly, POV
- Add speed/amplitude only when meaningful (e.g. "The camera pushes in at slow speed with small amplitude toward...").

=== 4. AUDIO & MUSIC SPECIFICATION ===
End your response with exactly two lines:
Audio: [1-3 sentences describing natural ambience, physical action sounds, footsteps, engine roar, impacts, breathing, chuckles, or sighs. Do not repeat the speech text here.]
Music: [1-2 sentences describing non-diegetic score: instrumentation, tempo/BPM, dynamic rise or resolve across shots.]

=== 5. STRICT OUTPUT RULES ===
- Output ONLY the timed shot narrative paragraphs ([Shot 1], [Shot 2] At MM:SS.mmm) followed by Audio: and Music: lines.
- NEVER write section headers like "subject_definitions:", "summary:", "retention_analysis:", or "detailed_description:" (an automated pipeline adds them).
- NEVER invent custom tags like <Environment 1> or <Location 1>. Reference images only as <Subject N> or <Picture N>.
- Output in the requested language ({output_language}).


---

## FILE: templates/t2v.txt (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/templates/t2v.txt)

## Task Type: Text-to-Video (T2V / T2VA)

This is a Text-to-Video generation task building the complete audiovisual timeline from text.

### Output Structure:
integrated_multimodal_description:
[Shot 1] Cinematic, live-action...
[Shot 2] At MM:SS.mmm, ...

overall_soundscape:
[Ambience and sound effects]

non_diegetic_music:
[Background score or N/A]


---

## FILE: templates/v2v.txt (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/templates/v2v.txt)

## Task Type: Video-to-Video (V2V)

Your job is to write a cinematic timeline description based on <Video 1>.

### Storyboarding Rules:
1. Divide your narrative into timed shots ([Shot 1], [Shot 2] At MM:SS.mmm).
2. Follow the camera movement, rhythm, and shot structure of <Video 1>, applying the requested visual changes.
3. Format your output strictly as a series of [Shot N] paragraphs with timestamps, followed by the Audio: and Music: lines. Do not add section headers.


---

## FILE: templates/v2va.txt (https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/HEAD/templates/v2va.txt)

## Task Type: Video-to-Video with Audio (V2VA)

Your job is to write a cinematic timeline description based on <Video 1> and <Audio 1>.

### Storyboarding Rules:
1. Divide your narrative into timed shots ([Shot 1], [Shot 2] At MM:SS.mmm).
2. Synchronize visible actions and speech with <Audio 1>, while maintaining the visual framework of <Video 1>.
3. Format your output strictly as a series of [Shot N] paragraphs with timestamps, followed by the Audio: and Music: lines. Do not add section headers.


---

# ISSUES / PRs of 1038lab/ComfyUI-MiniMax-H3-Promptor


### #20 [open] ISSUE: ollama not found
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/20  by BblackmilkK 2026-10-06T14:26:29Z

{
    "<Picture 1>": "API Error: Model 'llama3.2' not found in Ollama. Run: ollama pull llama3.2",
    "<Video 1>": "API Error: Model 'llama3.2' not found in Ollama. Run: ollama pull llama3.2",
    "_media_keys": [
        "<Picture 1>",
        "<Video 1>"
    ]
}

llama3.2 / llama3.2-vision / llama3.2-vision:11b I tried one by one and couldn't find any


### #19 [open] ISSUE: Several recurring issues: regular fails on first node, and cuts off results when it does work, incorrect info in image reads
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/19  by mdkberry 2026-10-06T00:06:05Z

Couple more things to check that are recurring issues. I probably get this custom node working 1 time out of every 5 times I try, and the rest fail in first or second Promptor nodes. Here are some of the issues that are recurring.

- If it isnt API errors, then it does stuff like the below in the first promptor node, where the API has connected fine but it doesnt produce a result in the first node.

> {
>     "<Picture 1>": "User Safety: safe",
>     "_media_keys": [
>         "<Picture 1>"
>     ]
> }

- when it does work in the first node, sometimes it still fails in the second node with:

> [H3-Promptor Stage 1 Error] Unexpected error: object of type 'NoneType' has no len()

- when I get it working, the detailed subject definitions it provides in first node will not carry over in full to the second node, and cut off after about 15 words per subject. This problem happens every time, I dont think I have seen it provide the full details from a subject yet without it cutting something off in the final output of the second promptor node.

anyway, hope this helps you improve it as I like it when it works, its pretty handy.




### #18 [open] ISSUE: issues when upstream doesnt connect
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/18  by mdkberry 2026-09-30T21:38:08Z

I really like this custom node when it works, but its got issues when upstream (openrouter) fails for whatever reason.

> [H3-Promptor] Analyzer calling prov_1790726904661 for IMAGES (3 items) in BATCH mode...
> [H3-Promptor] OpenAI request → https://openrouter.ai/api/v1/chat/completions | model=openrouter/free | temp=0.2
> [H3-Promptor ERROR] API Error in Image Batch 1: Rate limited: Provider returned error
> [H3-Promptor] OpenAI request → https://openrouter.ai/api/v1/chat/completions | model=openrouter/free | temp=0.7
> [H3-Promptor] OpenAI response ← 4407 chars | tokens: 2117
> [H3-Promptor] OpenAI request → https://openrouter.ai/api/v1/chat/completions | model=openrouter/free | temp=0.7

in the above case it jumps onto the second prompter node without finishing the first one as the first fails connection (rate limited)
but it cant cancel, or takes ages, so I either have to restart comfyui or wait

but then the next run, instead of it running the first prompter node again properly, it skips over it, because it thinks its done, but it isnt. so I get caught in the same "hung" situation again where it eventually  fails as the second prompter didnt get passed valid info and its refusing to run the first prompter node again despite it containing the below result:

> {
>     "<Picture 1>": "API Error: Rate limited: Provider returned error",
>     "<Picture 2>": "API Error: Rate limited: Provider returned error",
>     "<Picture 3>": "API Error: Rate limited: Provider returned error",
>     "_media_keys": [
>         "<Picture 1>",
>         "<Picture 2>",
>         "<Picture 3>"
>     ]
> }

1. it needs a way to tell it that the first prompter node should run again and not just skip over it. 
2. a way to kill the process without having to restart comfyui or wait ages for it to finally give up.

it would also be great to see more info in the command window (I am on windows 10) so I could know what is happening in the communication with upstream services. VSCOde cline and roocode are pretty good with that you get full info. would be cool to have more info here.

otherwise, when this works its fantastic.


> **Comment by mdkberry (2026-09-30T21:46:54Z):**

FYI, the only way I could get it to run the first prompter again was to change the seed. So that is a work around for that bit.



### #17 [open] ISSUE: issue with API with LM Studio locally
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/17  by mdkberry 2026-09-30T09:08:14Z

This is working for me with openrouter, but I cant get it working with LM Studio locally. I've checked the settings against VSCOde where I have LM Studio working.

<img width="708" height="402" alt="Image" src="https://github.com/user-attachments/assets/6db5f9de-c896-454b-96bd-3f04640a2eb9" />

It tests okay but in the LM Studio I get

> 2026-09-30 19:05:49 [DEBUG]
>  Received request: POST to /chat/completions with body  {
>   "model": "gguf",
>   "messages": [
>     {
>       "role": "system",
>       "content": "You are a master Hollywood Director, Showrunner, a... <Truncated in logs> ...rences, camera movement, and emotional punchline.\n"
>     },
>     {
>       "role": "user",
>       "content": "Task: Construct the Director's Blueprint & Global ... <Truncated in logs> ...00:00.000 to 15.000 seconds with zero empty gaps.\n"
>     }
>   ],
>   "temperature": 0.7,
>   "max_tokens": 4096
> }
> 2026-09-30 19:05:49 [ERROR]
>  Unexpected endpoint or method. (POST /chat/completions). Returning 200 anyway
> 2026-09-30 19:05:51 [DEBUG]
>  Received request: POST to /api/v0/models/unload with body  {
>   "model": "gguf"
> }
> 2026-09-30 19:05:51 [ERROR]
>  Unexpected endpoint or method. (POST /api/v0/models/unload). Returning 200 anyway

stuff like the above

and when I run it in the comfyui workflow I  get

<img width="1156" height="535" alt="Image" src="https://github.com/user-attachments/assets/2316ca34-da8b-441e-9e58-01ac110a0383" />

I've tried various api changes but nothing seems to work. 

> **Comment by gackt2 (2026-10-02T19:47:08Z):**

For me LmStudio won't let me change the local host URL .. its always giving me an IP address. how did you change it to http://localhost:1234?


> **Comment by mdkberry (2026-10-02T22:05:31Z):**

> For me LmStudio won't let me change the local host URL .. its always giving me an IP address. how did you change it to http://localhost:1234?

its exactly the same thing as http://127.0.0.1:1234 so you dont need to change it.

In my LM Studio it actually says `http://127.0.0.1:1234` so it just changes it in comfuui for whatever reason.

but its still not working for me because of that API key, and I havent set one up in LM Studio and neither do I want to coz its local. Let me know if you get yours working without one.



### #16 [open] ISSUE: Can no longer select a specific ollama vison model
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/16  by smithian123 2026-09-23T02:50:47Z

Hi,
I used to be able to select 'ollama' as the provider and then a specific model. Now it just offers llama3.2 which doesn't have vision capabilities. The error is;

[INFO] got prompt
[H3-Promptor] Analyzer calling prov_1720000004 for SEQUENTIAL image processing...
[H3-Promptor] Ollama request → http://localhost:11434/api/chat | model=llama3.2 | temp=0.2
[H3-Promptor ERROR] API Error in Image 1: Model 'llama3.2' not found in Ollama. Run: ollama pull llama3.2
[H3-Promptor] Ollama request → http://localhost:11434/api/chat | model=llama3.2 | temp=0.2
[H3-Promptor ERROR] API Error in Image 2: Model 'llama3.2' not found in Ollama. Run: ollama pull llama3.2
[H3-Promptor] Ollama request → http://localhost:11434/api/chat | model=llama3.2 | temp=0.2
[H3-Promptor ERROR] API Error in Image 3: Model 'llama3.2' not found in Ollama. Run: ollama pull llama3.2
[H3-Promptor] Ollama request → http://localhost:11434/api/chat | model=llama3.2 | temp=0.2
[H3-Promptor ERROR] API Error in Image 4: Model 'llama3.2' not found in Ollama. Run: ollama pull llama3.2
[H3-Promptor] Ollama model 'llama3.2' unloaded.
[H3-Promptor] Ollama request → http://localhost:11434/api/chat | model=llama3.2 | temp=0.7
[H3-Promptor ERROR] [H3-Promptor Stage 1 Error] Model 'llama3.2' not found in Ollama. Run: ollama pull llama3.2
[H3-Promptor] Ollama model 'llama3.2' unloaded.
[INFO] Prompt executed in 14.54 seconds

Thanks for taking a look. - Ian

<img width="932" height="394" alt="Image" src="https://github.com/user-attachments/assets/782e308f-298e-45d2-862f-5d597124c316" />

> **Comment by Delerium (2026-10-29T07:30:27Z):**

in the comfyui settings, scroll down to the Minimax H3 tab, and you can define whatever models you want to appear in that list, or modify the existing ones.



### #15 [open] ISSUE: [H3-Promptor ERROR] [Local LLM Error] Failed to initialize QwenVLEngine runtime.
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/15  by raffetazarius 2026-09-21T23:04:18Z

I installed llama-cpp-python correctly using:
pip install "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.4.0-cu130-win-20260919/llama_cpp_python-0.4.0+cu130-cp313-cp313-win_amd64.whl" --upgrade --force-reinstall

For:
.venv\Scripts\python.exe -c "from llama_cpp.llama_chat_format import Qwen3VLChatHandler, Qwen25VLChatHandler; print('handlers OK')"

I get the response:
[llama-cpp-python] loaded bundled OpenMP runtime: C:\Users\user\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI-Sep2026-v3\ComfyUI\.venv\Lib\site-packages\llama_cpp\lib\libomp140.x86_64.dll

[llama-cpp-python].find_library: loaded library from C:\Users\user\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI-Sep2026-v3\ComfyUI\.venv\Lib\site-packages\llama_cpp\lib\ggml-base.dll

[llama-cpp-python].find_library: loaded library from C:\Users\user\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI-Sep2026-v3\ComfyUI\.venv\Lib\site-packages\llama_cpp\lib\ggml.dll

[llama-cpp-python].find_library: loaded library from C:\Users\user\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI-Sep2026-v3\ComfyUI\.venv\Lib\site-packages\llama_cpp\lib\llama.dll

handlers OK

But, attempting to run [this workflow ](https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/main/example_workflows/Minimax-H3-Promptor_vision_V1.4.0.json) I keep getting the following error:

[INFO] got prompt
[H3-Promptor] Analyzer calling prov_1720000007 for IMAGES (2 items) in BATCH mode...
[H3-Promptor ERROR] [Local LLM Error] Failed to initialize QwenVLEngine runtime.
[H3-Promptor ERROR] API Error in Image Batch 1: [Local LLM Error] Failed to initialize QwenVLEngine runtime.
[H3-Promptor] Local LLM (QwenVL GGUF) unloaded successfully.
[INFO] Prompt executed in 0.97 seconds

The mmproj file is  definitely downloaded and installed for the Qwen 3 VL 32B model I'm attempting to use.
<img width="653" height="165" alt="Image" src="https://github.com/user-attachments/assets/75020c0b-dfda-4338-9146-e414e5a1db05" />

Furthermore, if I try to "Test" the mode in the Settings > "MiniMax H3 Prtomptor API Settings", I constantly get this error:
<img width="785" height="562" alt="Image" src="https://github.com/user-attachments/assets/df87be71-9e42-465c-8c27-192b29e75134" />

Why can't the QwenVLEngine runtime initialize?  How do I get this working?


### #14 [closed] ISSUE: Reference mode is giving error.
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/14  by sudeep333 2026-09-14T05:34:27Z

This is the error when i am executing the node in reference mode. I recently have updated the comfyui and node broke.
Here is the error.

```
  File "E:\ComfyUI-9\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\ComfyUI-9\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "E:\ComfyUI-9\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\ComfyUI-9\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\comfy_api\latest\_io.py", line 2046, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\ComfyUI-9\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Minimax-H3-Prompt-Engineer\combined_node.py", line 640, in execute
    generated = comfy_h3.MiniMaxH3ReferenceToVideo.execute(
        clip,
    ...<10 lines>...
        bundle.ref_audios,
    )
  File "E:\ComfyUI-9\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\comfy_extras\nodes_minimax_h3.py", line 290, in execute
    latent, frame_count = _empty_av_latent(width, height, length)
                          ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI-9\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\comfy_extras\nodes_minimax_h3.py", line 85, in _empty_av_latent
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                                                   ~~~~~~~^^~~~
TypeError: unsupported operand type(s) for //: 'str' and 'int'
```

while searching for the solution i have found this solution. But i dont know how to implement it in your node.
link : https://github.com/yusu-02/ComfyUI-Yusu-MiniMaxH3-Unified/commit/e47b4668db94e7af007c52e2573d0fbcb9a3f1f2

Thank you for the node, waiting for the fix.


> **Comment by sudeep333 (2026-09-14T06:03:04Z):**

[combined_node.py](https://github.com/user-attachments/files/32181467/combined_node.py)

i have tried to fix it and its working now. 
You can crosscheck it.
i have attached the file. replace the old file with this one inside the node folder.


EARLIER: 

generated = comfy_h3.MiniMaxH3ReferenceToVideo.execute(
    clip,
    vae,
    audio_vae,
    formatted,
    width,
    height,
    length,
    ref_image_size,
    bundle.ref_images,
    bundle.ref_videos,
    bundle.ref_video_audios,
    bundle.ref_audios,
)

THE FIX:

generated = comfy_h3.MiniMaxH3ReferenceToVideo.execute(
    clip,
    formatted,
    width,
    height,
    length,
    ref_image_size,
    vae=vae,
    audio_vae=audio_vae,
    ref_images=bundle.ref_images,
    ref_videos=bundle.ref_videos,
    ref_video_audios=bundle.ref_video_audios,
    ref_audios=bundle.ref_audios,
)


> **Comment by 1038lab (2026-09-20T17:51:57Z):**

This error originates from `ComfyUI-Minimax-H3-Prompt-Engineer` (`combined_node.py`), not this repository (`ComfyUI-MiniMax-H3-Promptor`).
Please submit this issue to their repository instead. Closing as not applicable.



### #13 [open] ISSUE: VRAM is not cleared after using 'refine' in MiniMax H3 Prompt Composer
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/13  by AaronPorts 2026-09-11T15:34:09Z

VRAM is not released after running the refine function in MiniMax H3 Prompt Composer. I use Ollama & ComfyUI in Docker.

> **Comment by 1038lab (2026-09-12T19:00:01Z):**

Update [V1.5.1](https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/main/updates.md#v151-20260912) fixed you issues



### #12 [closed] ISSUE: No option for local LM Studio ?
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/12  by Lumabrik 2026-09-09T19:47:19Z

In the MiniMax H3 Vision Analyzer node there is no option for local LM Studio as mentioned in the documentation., the Openai option just tries to connect to their API ?

> **Comment by 1038lab (2026-09-10T02:46:10Z):**

LM Studio uses the OpenAI-compatible API format. Just set API Type to OpenAI, then enter your local LM Studio IP address and port in the Host field.

<img width="785" height="433" alt="Image" src="https://github.com/user-attachments/assets/56990abb-6960-4f39-ac09-ae86bf8e348d" />



### #11 [closed] ISSUE: qwenvl provider type not unloading model after run
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/11  by chompie-xyz 2026-09-04T04:13:22Z

When using a local qwenvl provider via ComfyUI-QwenVL node support with a GGUF model, the model is not unloaded afterwards. Attempting to 'unload models and execution cache' via comfy-ui admin panel does not free the allocated vram either, requiring a full reboot of comfy to reclaim this vram before the generated prompt can be used in a H3 video generation.

Based on previous release notes, it looks like "Invisible VRAM Unloading & Management" was added in v1.1.0, so it would be great to have the same functionality for qwenvl provider type.

> **Comment by 1038lab (2026-09-10T02:54:36Z):**

By default, our node will unload the QwenVL model after each run. We don't keep the model alive because most users need the VRAM for the MiniMax model afterward.


> **Comment by chompie-xyz (2026-09-12T14:52:07Z):**

Yes, that was what I expected. What I'm saying is that this unload does not appear to be happening. The model remains in VRAM after using the QwenVL provider.


> **Comment by 1038lab (2026-09-12T19:00:22Z):**

Update [V1.5.1](https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/main/updates.md#v151-20260912) fixed you issues



### #10 [closed] ISSUE: v1.2.0 - Promptor does not follow instructions / Duration is wrong
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/10  by RhapsodyinBlack 2026-08-18T17:56:58Z

Hi AILab

I really enjoyed using your custom node (v1.1.0). I did not have any problems and it worked great. However v1.2.0 has issues which makes it unusable (at least for me).

Issue 1 - Promptor does not follow instructions
Promptor ignores anything I put in the description box. It will generate whatever prompt it likes. Occasionally it will generate a blank prompt.

Issue 2 - Promptor does not respect Duration.
It will randomly pick a duration. In the examples below, the duration is set to 5s.

The first image is of v1.1.0. Everything is working ok. The prompt instructs the model to dance and it respects the duration of 5s.
<img width="1128" height="1060" alt="Image" src="https://github.com/user-attachments/assets/6464eb78-c44b-45e3-a389-cb00ee2d426e" />

This second image is of v1.2.0. Instead of dancing, the prompt describes a couple walking on the beach in the sunset. Duration is set to 5s but prompt has instructions up to 15s. Unfortunately, I did not screenshot the full 15s prompt.

<img width="1139" height="1078" alt="Image" src="https://github.com/user-attachments/assets/60de4b38-fbbd-49d0-b33b-0c5dd89a9e64" />

The third image is also of v.1.2.0.  Instead of a 5s dancing prompt, I get a 45s prompt of the subject exploring a building.

<img width="1120" height="1050" alt="Image" src="https://github.com/user-attachments/assets/465136cf-33c6-4f7a-a3d4-38722765a74f" />

The last 2 images are also of v.1.2.0. I changed the provider to gemma4:12b (still tested OK in config) and I get a [H3 Promptor Error - None]. The ollama window shows tokens were used but comfyui window show no prompt was returned. 

<img width="1137" height="1067" alt="Image" src="https://github.com/user-attachments/assets/7728d035-ea9c-43be-ad27-9707c3f1d968" />
<img width="1101" height="850" alt="Image" src="https://github.com/user-attachments/assets/5706498c-639a-4a1d-81cf-ccd2c49ccf41" />

Things I have tried:
1. Reinstalling v.1.2.0 (directly from git URL and also by upgrading from v1.1.0) via ComfyUI Manager)
2. Restarting ComfyUI between jobs.
3. Used Ollama gemma4:12b model as the provider for the promptor.
4.. Deleted and recreated the providers in the Comfyui settings (they test OK).
5. Reloaded/deleted/used new nodes in workflow after upgrading to v1.2.0.

Thank you very much - I hope this information is useful. I have since downgraded back to v1.1.0 which is still working well.

> **Comment by 1038lab (2026-08-19T17:43:17Z):**

update V1.3.0 will fix your issue


> **Comment by RhapsodyinBlack (2026-08-20T04:28:44Z):**

Thank you very much, I updated to v1.3.0. The duration and direction issues appear to have been fixed - thank you. 

Unfortunately with the new drag-and-drop vision analyzer I have trouble when I connect a Load Image node directly to Image0 input in the Vision node. If I don't have any image loaded directly through the node, it throws the following Execution Failed error. I get a similar error when connecting to the Image1 input.

Error log
# ComfyUI Error Report
## Error Details
- **Node ID:** 2
- **Node Type:** H3_Vision_Analyzer
- **Exception Type:** TypeError
- **Exception Message:** TypeError: H3_Vision_Analyzer.execute() got an unexpected keyword argument 'ref_images.image_0'

It works if I upload or drag-drop my images directly onto the node, but this means I have to run your custom node standalone and cannot incorporate it into an existing workflow. Is it possible to fix so it still takes inputs externally from the node?

Thank you.



### #9 [closed] ISSUE: Description is completely ignored
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/9  by FalconBob 2026-08-18T17:12:12Z

Whatever I right in the description of the MiniMax H3 Promptor node, it is completely ignored. It just gives me random output which has nothing to do with my initial description.

> **Comment by 1038lab (2026-08-19T17:43:31Z):**

update V1.3.0 will fix your issue



### #8 [closed] ISSUE: I put duration of 10 seconds into the MiniMax H3 Promptor node but it outputs shots up to 00:15.12 seconds
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/8  by abline11 2026-08-16T22:58:30Z

I put duration of 10 seconds into the MiniMax H3 Promptor node but it outputs shots up to 00:15.12 seconds

> **Comment by sudeep333 (2026-08-17T13:25:56Z):**

Yes, i noticed that too. it is not honoring the duration parameter. could be a bug.

Also it is giving random prompts not based on the input image. 


> **Comment by 1038lab (2026-08-19T17:44:39Z):**

update V1.3.0 will fix your issue


> **Comment by abline11 (2026-08-19T20:52:08Z):**

It is padding otherwise the end of the generation tend to get static or breaks down. 



### #7 [closed] ISSUE: Changing model in ollama does not reflects in the workflow
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/7  by ritonV 2026-08-15T08:32:27Z

When changing the settings to add a new model with Ollama

<img width="1296" height="966" alt="Image" src="https://github.com/user-attachments/assets/e41f4856-f70b-4aae-ac01-36fe8a3fbdd4" />


Testing it's OK.
But It does not change the LLM input in the workflow. No other choice in the menu.
No change when refreshing.

<img width="1146" height="885" alt="Image" src="https://github.com/user-attachments/assets/f70ef7b9-5186-4d63-9d4a-940cf7aa77c6" />

> **Comment by sudeep333 (2026-08-17T13:27:57Z):**

I faced that too.
I deleted the Ollama default connection and created a new one with these settings.  Hope it will help.

<img width="665" height="691" alt="Image" src="https://github.com/user-attachments/assets/de46126a-f402-455d-b0f5-0969fd92d12a" />


> **Comment by 1038lab (2026-08-18T16:18:51Z):**

There are two ways to update the node model list:
- Force refresh your browser (Ctrl+Shift+R or F5).
- Restart ComfyUI.



### #6 [closed] ISSUE: Error when i am giving more than 3 images in the ref2av model.
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/6  by sudeep333 2026-08-10T08:21:26Z

When i am trying to upload more than 3 images it is throwing error. I am using ollama locally with a custom model.

with 3 images it is working fine. I am using ref2va workflow.

I have tried increasing the max_token but it is not accepting. Can you please tell me whats happening.

<img width="718" height="532" alt="Image" src="https://github.com/user-attachments/assets/a896cc83-5624-4fee-ad53-37b17910e4fc" />

> **Comment by sudeep333 (2026-08-10T08:24:46Z):**

Ok i have found the issue. I had to increase the Context length in Ollama from 4000 to 8000 and it worked. I guess it will help someone with similar issue.

<img width="673" height="659" alt="Image" src="https://github.com/user-attachments/assets/e686f2f3-7733-46c2-8a48-ab6a90ea0c66" />


> **Comment by NeverOccurs (2026-08-13T09:32:49Z):**

Global vibe is "failed to synthesize," mine too. Is this a cause for cocern? Or it can be safely ignored?



### #5 [closed] ISSUE: Generating outputs based on the Video Prompt Writing Guide
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/5  by pxc052 2026-08-08T08:31:48Z

Could you please share how to generate output format based on the writing guide?
https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md

> **Comment by 1038lab (2026-08-14T16:05:21Z):**

Our node has fully supported these features since V1.0.0, and we have further optimized them in the latest V1.2.0 release.



### #4 [closed] ISSUE: Support llama.cpp?
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/4  by x5nder 2026-08-07T09:43:30Z

Can you add llama.cpp support?
So basically, support for /v1/chat at specified endpoints... 
Would need a server_url (eg https://llama-server-url.com:1234/), optional api_key, and model name.

> **Comment by 1038lab (2026-08-07T22:05:29Z):**

Yes! You can actually do this right now. Since llama.cpp provides an OpenAI-compatible API, you can just open 
config.json
 and change the openai provider's api_base to your llama.cpp server URL (e.g., http://localhost:1234/v1). Then just select openai in the node, and it will route to your local model automatically!



### #3 [closed] ISSUE: api_key widget is saved into workflow JSON and exported image metadata
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/3  by roadmaus 2026-08-06T10:11:00Z

Both nodes expose `api_key` as a plain `STRING` widget:

- `py/h3_promptor.py:91`
- `py/h3_vision_analyzer.py:131`

Anything a user types there is a normal widget value, so ComfyUI serializes it into the workflow `.json` and embeds it in the PNG metadata of every image the graph produces. Sharing a workflow, attaching a `.json` to an issue, or posting a generated image hands over the key with it.

Nothing in the UI signals this. The tooltip reads "API key override", which sounds local to the node.

### Reproducing

1. Enter a key in `api_key` on either node.
2. Save the workflow, or generate an image through the graph.
3. The key is in the saved `.json`, and in the exported PNG's embedded workflow metadata.

### Scope

The repo itself ships no credentials — the README's `sk-...` / `AIza...` are placeholders and the bundled example workflow has empty `api_key` values. So there's nothing to revoke here; the risk is to users who used the field and shared the results. Worth a note in the release announcement so people can check their own shared workflows and rotate if needed.

### Suggested fix

Store a credential *name* rather than the secret. `config.json` is already the source of truth and is outside the graph, so the widget can be a combo of names defined there, with the key never entering the node's inputs.

That also allows several keys per provider:

```json
"credentials": {
  "openai · work":     { "provider": "openai", "api_key": "sk-..." },
  "openai · personal": { "provider": "openai", "api_key": "sk-..." }
}
```

Two smaller things worth doing regardless:

- If the field stays, sanitize it on save and document that keys belong in `config.json`.
- Confirm `config.json` is gitignored — it holds real keys and is auto-created on first run.

> **Comment by 1038lab (2026-08-07T22:02:48Z):**

 you can manually update your API key in `config.json` for now. This ensures the key isn't stored within your generated images. We have a more robust API management system coming in the next update!



### #2 [closed] ISSUE: Unload Model Toggle
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/issues/2  by soulreaper7633 2026-08-05T22:55:26Z

Hello,

Great node, i look forward to using it. 
As it stands, my Vram fills up and then start overflowing once minimax starts its generation. Is it possible to add a toggle to the prompt generation node that would send an Unload command to Llama.cpp so that the Vram is cleaned before the image generation begins? 

Thanks!

> **Comment by 1038lab (2026-08-07T22:37:46Z):**

You can update ro V1.1.0
https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/blob/main/updates.md#invisible-vram-unloading--management



### #1 [closed] PR: Add LM Studio provider integration and update documentation
URL: https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor/pull/1  by mockcomic 2026-08-05T22:33:30Z



> **Comment by 1038lab (2026-08-06T00:04:25Z):**

Thanks for your contribution! Both LMStudio and llama.cpp utilize an OpenAI-compatible API, so our node supports them natively without additional code. To get LMStudio running, just edit the config.json file in the node directory and point the api_base to your local server (e.g., "api_base": "http://localhost:1234/v1"). You can use any dummy string for the API key, and it should work seamlessly.


> **Comment by NeoAnthropocene (2026-08-06T12:30:16Z):**

@1038lab  Are you planning to merge this LM Studio update? It would make it easier for users to use LM Studio instead of Ollama as the local provider.

The @mockcomic implementation is pretty neat. If you are not going to merge, I will fork and use that version :)


> **Comment by 1038lab (2026-08-19T17:48:56Z):**

We migrated the API Setting to the ComfyUI settings panel in v1.2.0.



