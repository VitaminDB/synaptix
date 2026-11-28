# tritant__ComfyUI_MiniMax_H3_Extender

Source: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender
Fetched: 2026-11-28

Description: ComfyUI node for MiniMax H3 that chains multiple video clips with motion context, disk caching, dynamic image references, audio reference support, and seamless final video/audio decoding.
Stars: 389  Updated: 2026-11-22T10:13:35Z



---

## FILE: README.md (https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/blob/HEAD/README.md)

# ComfyUI MiniMax H3 Extender

A ComfyUI custom node for **MiniMax H3** designed to generate long, continuous video sequences from multiple clips while preserving motion, visual continuity, and audio continuity between generations.

The node combines **Ref2VA conditioning, Motion Context, disk caching, multi-clip generation, image references, audio references, and final video/audio decoding** into a much simpler workflow.

---

### Project status

This is the **final feature release** of MiniMax H3 Extender.

Apart from possible bug fixes or compatibility corrections, I do not currently plan to continue active development of this project.

---

### 🚀 New in v3.1.0 — RelayVSR ×2 Upscaling

MiniMax H3 Extender now includes **RelayVSR ×2**, directly integrated into Final Decode!

- **2× video upscaling** for sharper, more detailed videos.
- **Adjustable Detail strength** (0–100%).
- **Automatic upscaling** after the final video is generated.
- **Automatic model download on first use** if the required RelayVSR models are not already installed.
- **Original / Upscale preview toggle** to compare both versions.
- **Both videos are preserved**, including audio.
- **Save/Load Project and Auto Save support**, including upscaled videos.
- **Smart caching** to avoid unnecessary processing.
- Works with **all three generation modes**.

Simply enable **RelayVSR ×2** in Final Decode. No additional nodes required!

---

### 🚀 New: v3.0.5 — Latent Refine / Upscale

A new optional **Latent Refine / Upscale** mode is now available, inspired by the latent refine work introduced in **PR #83**.

When enabled, each clip is generated normally, then its latent is upscaled and passed through a second H3 refine pass before continuing through the Extender’s existing cache, Motion Context, preview and export pipeline.

There is no separate draft/refine workflow: the refined result becomes the clip’s final result directly.

Refine settings include:
- **Scale Factor**
- **Refine Steps**
- **Refine Denoise**
- automatic display of the resulting final video resolution

The same H3 model is reused for both passes, and the required latent upscaler model is downloaded automatically on first use.

---

## 🆕 v2.9.0 — Continue an Existing Video

MiniMax H3 Extender can now continue from an existing video.

- Connect a video to the new optional `continue_existing_video` input.
- The source becomes a locked **Clip 0** and is automatically included at the beginning of the preview and final export.
- Clip 0 can be resized directly from its card using **Original**, **Auto / MP**, or **Manual** working resolution.
- The source video is normalized to the Extender working format without modifying the original file.
- The final frames of Clip 0 are used as the visual context for the first generated clip.
- **Choose Ref** lets you pick any frame from the source video and use it as a normal `<Picture N>` reference, useful for preserving character identity when the end of the source video does not show the subject clearly.
- Existing Motion Context behavior between generated clips remains unchanged.

This makes it possible to take an existing video and continue building it directly inside the Extender.

---

## 🆕 v2.8.0 — Optional individual Full Batch clip export

Full Batch can now optionally save each final clip as a separate video file using the new **Save Individual Clips** option in **Final Decode / Preview**.

The option is disabled by default, so the existing Full Batch workflow remains unchanged unless explicitly enabled.

When enabled, individual clips are exported only after all normal processing is complete. They use the same final video segments as the assembled output, including seam correction, per-clip color adjustments and FL2VA visible-frame trimming/handoffs.

Audio is captured during the same final PCM assembly pass used for the complete video, so no additional audio reconstruction pass is required.

No additional sampling or VideoVAE decode is introduced. Individual video streams are reused with stream-copy and saved in a dedicated folder beside the final assembled video.

Supported in:
- Ref2VA Motion ON
- Ref2VA Motion OFF
- FL2VA

---

## 🆕 v2.7.7 — Full Batch Autosave

- Optional automatic `.ext` project save after a successful **Full Batch**
- Supports **Ref2VA Motion ON/OFF** and **FL2VA**
- Preserves the **actual generation seed** and project settings
- Clip-by-Clip behavior remains unchanged

---

### 🆕 v2.7.5 — New Project button

Added a new **New Project** button to quickly start from a completely clean state.

It clears all current project data and cached references, removes prompts, resets the timeline to a single empty clip, and wipes previous computed/validated state — while keeping all global node settings unchanged.

Useful when switching to a completely different project and helps prevent old references or cached project data from carrying over into the next generation.

---

### 🆕 Motion Context OFF

Ref2VA now includes a **Motion Context ON/OFF** option.

With **Motion Context OFF**, clips are fully independent from each other. This means you can **insert, rerender, or invalidate any clip without invalidating the clips before or after it**, making random-access editing much more flexible.

The trade-off is that there is **no continuity between clips**: each clip is generated as a separate shot, so characters, framing, camera position, motion, and scene composition are not automatically carried over from one clip to the next.

Use **Motion Context ON** when you want continuous multi-clip motion and seamless progression, and **Motion Context OFF** when you prefer independent shots and easier editing.

---

### 🆕 Per-Clip Local References

Ref2VA clips can now use their own **local Picture, Video and Audio references** in addition to the existing global references.

Each clip gets a compact **Refs** panel where local references can be added, removed and previewed without cluttering the main card UI.

Local references use the normal H3 slot numbering and share the same limits as global references:

- up to **9 Pictures**
- up to **3 Videos**
- up to **3 Audio references**

Global references remain global exactly as before. Local references simply use the next available slots for that clip.

If a slot is already used locally, the matching global slot is automatically reserved to avoid conflicts. References coming from a Ref Pack are also handled safely without remapping or blocking generation.

Local Pictures can be edited with the same image editor as global references, and local Video/Audio references include compact preview players directly inside the Refs panel.

All local references are fully included in **Project Save/Load**, so they are restored automatically with the project.

---

### 🆕 Full Batch: Interrupt, Save & Resume

Full Batch workflows are now much safer and easier to manage.

You can interrupt a long Full Batch, keep the clips that are already computed, save the project, close ComfyUI, reload it later, and resume from where you stopped without losing completed work.

This also works with project Save/Load, so long generations can now be split across multiple sessions instead of needing to finish in one run.

Available for both **Ref2VA** and **FL2VA**.

---

## 🆕 News — Dynamic FL2VA Guides & Native VIDEO Output

### 🎯 Dynamic Image Guides for FL2VA
FL2VA clips can now use up to **3 independent image guides**, each placed at an exact `frame_idx`.

This is based on ComfyUI's native **MiniMaxH3AddGuide** support introduced in PR [#15439](https://github.com/Comfy-Org/ComfyUI/pull/15439).

You can now build a real visual timeline inside a single generation:

`First → Guide 1 → Guide 2 → Guide 3 → Last`

MiniMax H3 automatically generates the transitions between these visual anchors, opening the door to controlled transformations, pose changes, camera evolution, action staging and much more.

### 🎬 Native VIDEO Output
The **Final Decode / Preview** node now exposes a native ComfyUI `VIDEO` output.

This makes it possible to connect the final result directly to compatible video nodes such as upscalers, post-processing pipelines and video encoders, without having to reload the generated MP4 manually.

---

## 🚀 MiniMax H3 Extender 2.0.0

Version **2.0.0** is now available!

This release introduces **FL2VA support**

It also includes major **speed and memory optimizations**, with reduced RAM/VRAM usage, faster continuity handling, smoother preview behavior, and improved performance on longer projects.

Project save/load, preview restoration, clip continuity, and overall stability have also been improved.

**MiniMax H3 Extender 2.0.0** is now the new base for the project.

Thanks to everyone testing, reporting issues, and helping improve the extension ❤️

---

## 🎛️ New — Per-Clip LoRAs

Each clip card can now use its own LoRA.

- choose a LoRA directly inside the clip card
- set its strength per clip
- add multiple LoRAs to the same clip
- LoRA rows grow automatically as needed
- remove a LoRA and the list compacts automatically
- clips without LoRAs continue to use the incoming model unchanged
- global LoRAs applied before the Extender still work as before
- per-clip LoRA choices are saved with the project

This makes it possible to change style, motion or behavior from one clip to another while keeping the Extender’s normal continuity workflow.

---
## 🎬 New — Video & Audio References for MiniMax H3

The Extender now supports **MiniMax H3 video references** directly inside the workflow.

You can now use external video frames as `<Video N>` references, combine them with image references, and optionally attach the matching video audio for motion, timing, camera behavior and lip-sync guidance.

### What’s new

- up to **3 video references**
- up to **3 matching video-audio references**
- up to **3 standalone audio references**
- `<Video 1>`, `<Video 2>`, `<Video 3>` support in prompts
- stable logical video slots with no automatic remapping
- dynamic AV inputs in the node UI
- optional paired `ref_video_audio_N` inputs
- long standalone audio references are automatically sliced along the clip timeline
- video soundtracks are cropped to the effective reference-video duration
- reference-video inputs are automatically aligned to H3’s required `17k+5` frame structure
- external Prompt Pack and Reference Pack sockets remain at the bottom of the node

### Automatic FPS correction

MiniMax H3 expects reference-video frames at **24 fps**, but real source videos can be 23.976, 25, 30, 45, 50, 60 fps, etc.

The Extender now accepts the original source FPS through:

- `ref_video_fps_1`
- `ref_video_fps_2`
- `ref_video_fps_3`

The reference-video frame batch is then automatically **resampled to 24 fps while preserving the original duration** before it is sent to H3.

This fixes cases where the end of a reference video appeared to be missing because a non-24-fps image batch was previously interpreted as if it were already 24 fps.

Typical workflow:

~~~text
Load Video
   ↓
Get Video Components
   ├── images → ref_video_1
   ├── fps    → ref_video_fps_1
   └── audio  → ref_video_audio_1
~~~

### Character replacement

Video references can be combined with image references for workflows such as:

- character replacement
- motion transfer
- pose and body-performance transfer
- camera-motion transfer
- timing preservation
- lip-sync guidance from the original video audio

For example:

- `<Picture 1>` defines the new character identity
- `<Video 1>` provides the original performance, timing and camera behavior
- `ref_video_audio_1` provides the matching audio reference

This keeps the Extender’s existing internal image-reference system fully intact while adding proper H3 Ref2VA video/audio support.


**Added support for an external prompt pack through the new MiniMax H3 Prompt Pack Bridge node.**

## Reference Pack Bridge

A new node **MiniMax H3 Reference Pack Bridge** has been added.

This allows external ComfyUI `IMAGE` outputs to be injected directly into the Extender’s existing internal reference slots, while keeping the current internal reference system fully intact.

You can now mix both approaches in the same workflow:

- internal references loaded with the Extender
- external references coming from `Load Image`, crop/resize nodes, video frames, Kontext-style preprocessing, or any other `IMAGE` source
- each external input maps directly to its matching `<Picture N>` slot
- empty external slots leave the existing internal reference untouched
- reference indices are never compacted or remapped
- once imported, the external image becomes a normal internal reference with thumbnail, project persistence and Save/Load compatibility
- disconnecting the external input does not remove the imported internal reference
- unchanged external images are not rewritten on every run

The Bridge is optional and does not replace the current reference workflow. It simply adds a flexible external entry point for advanced ComfyUI pipelines.

### 🎬 Save Preview

The **Final Decode / Preview** node now includes a dedicated **Save Preview** button.

It saves the currently assembled Extender preview exactly as shown, including:

- seam handling
- assembled audio
- per-clip color corrections
- ComfyUI workflow metadata
- ComfyUI prompt metadata

The saved MP4 can be dragged back into ComfyUI to restore the workflow.

### 🎨 Real-Time Per-Clip Color Editor

Each generated clip now has its own **color editor**, accessible directly from the clip card with the `🎨` button.

You can adjust:

- Saturation
- Contrast
- Brightness

The editor provides a **live looping preview around the selected clip**, including a short part of the previous and next clips, making it much easier to visually match colors between continuations.

Corrections are:

- stored independently for every clip
- editable again at any time
- non-destructive
- preserved in `.ext` Save/Load projects
- automatically included in Preview, Full Batch and Save Preview

Color correction is applied only to the decoded video and **does not modify Motion Context, latent data or clip validation**.

A small `✓` next to the palette indicates that a clip has an active color correction.

### ⚡ Full Batch Integration

Full Batch correctly preserves and applies all previously configured clip corrections during final assembly.

You can therefore:

1. Generate several clips
2. Color-correct individual clips
3. Continue generating additional clips
4. Run Full Batch
5. Obtain the complete sequence with all previous adjustments preserved

### 💾 Save / Load Project

Per-clip color adjustments are stored inside `.ext` projects.

Loading a project restores the clip settings and color correction state, allowing you to continue exactly where you stopped.

- **Internal reference manager**

  - Add thumb editor.
  - Up to 9 image references can now be loaded directly inside the Extender.
  - No external `Load Image` nodes are required anymore.
  - References are shown as thumbnails directly in the node.
  - Reference slots stay fixed, so prompt numbering remains stable.
  - Double-click a thumbnail to view it larger.

- **Portable `.ext` projects**
  - `Save Project` now embeds the actual reference images inside the project archive.
  - `Load Project` restores prompts, refs, validation state, cache, preview and resolution settings.
  - A saved project can therefore be reopened on another machine without needing the original image files.

- **Named clip cards**
  - Each clip card can now have its own optional name, making long sequences much easier to organize.

- **Improved audio joins** (work in progress)
  - Audio is now rebuilt from PCM and encoded only once at the end instead of concatenating separate AAC streams.
  - Additional gain matching, declicking and smooth entry ramps reduce audible bumps between clips.

---

https://github.com/user-attachments/assets/a18cc6a5-2340-474e-9d3b-b784cd41584a

<img width="987" height="922" alt="Capture d&#39;écran 2026-08-31 010245" src="https://github.com/user-attachments/assets/62cf39ce-5ec4-4a69-b472-3fbafa18cb81" />


<img width="1836" height="945" alt="Capture d&#39;écran 2026-08-29 141350" src="https://github.com/user-attachments/assets/490497ef-93a0-4442-89be-fc5589491c2f" />


<img width="2307" height="1028" alt="image" src="https://github.com/user-attachments/assets/c1126ae8-2d4b-416c-a8c2-b839cd4c6b15" />


<img width="2376" height="1163" alt="Capture d&#39;écran 2026-08-20 062155" src="https://github.com/user-attachments/assets/072885ca-8faa-4d58-b525-4b12d57fe75c" />


<img width="1860" height="691" alt="image" src="https://github.com/user-attachments/assets/9f2354a4-d8b7-485d-904f-76481e8fba15" />


<img width="2048" height="1103" alt="image" src="https://github.com/user-attachments/assets/f67ba34a-5b8a-4d3b-9c18-d6314db2c873" />


<img width="2091" height="948" alt="image" src="https://github.com/user-attachments/assets/905dcd01-09f8-4dfe-9d20-2f949637f938" />


<img width="2557" height="1212" alt="Capture d&#39;écran 2026-08-15 083401" src="https://github.com/user-attachments/assets/99ca1fc4-d8b9-4662-a869-1fa06e5e58e1" />

## Keeping References Consistent Across Clips

When using image references, it is strongly recommended to place a `subject_definitions` block at the **beginning of every clip prompt**.

This helps MiniMax H3 keep the same subjects, identities, clothing, visual roles and even the environment associated with a reference image from one clip to the next.

### Example

> **subject_definitions:**  
> `<Picture 1>` is the reference image defining the exact visual appearance, identity, face, hairstyle, body proportions, clothing, accessories, and overall look of `<Subject 1>`, as well as the established environment and visual context of the scene.  
> `<Picture 2>` is the reference image defining the exact visual appearance, identity, face, hairstyle, body proportions, clothing, accessories, and overall look of `<Subject 2>`.  
> `<Subject 1>` is the exact same woman shown in `<Picture 1>`.  
> `<Subject 2>` is her friend, the exact same woman shown in `<Picture 2>`.  
> `<Subject 3>` is the same street environment and scene context established in `<Picture 1>`.

Place this `subject_definitions` block at the **very beginning of every clip prompt**.

The important point is that a reference image does not have to represent only a character.

A reference can also define the **environment, location or visual context** of the sequence.

In the example above:

- `<Picture 1>` defines the appearance of `<Subject 1>`
- `<Picture 2>` defines the appearance of `<Subject 2>`
- `<Picture 1>` also defines the street environment used as `<Subject 3>`

So the same reference image can be used both to preserve a character **and** to preserve the environment established by that image.

Repeating these definitions at the beginning of each clip prompt helps MiniMax H3 maintain the intended reference roles throughout the entire sequence.

---

## Features

- Multi-clip MiniMax H3 generation
- Continuous Motion Context between clips
- Video and audio latent continuity
- Disk-based latent cache
- Clip-by-clip generation workflow
- Full batch generation mode
- Per-clip prompt
- Per-clip seed
- Seed modes:
  - Randomize
  - Fixed
  - Increment
  - Decrement
- Per-clip duration
- Clip validation system
- Dynamic image reference inputs
- Up to **9 image references**
- Optional **audio reference**
- Shared references automatically applied to all clips
- Native MiniMax H3 Ref2VA conditioning
- Native ComfyUI sampling progress
- Final video preview
- Final video + audio export
- Seam correction between generated clips
- Audio seam correction / declick
- H.264, H.265 / HEVC and FFV1 export
- Persistent disk cache allowing generation to be resumed

---
# Clip Validation

Each clip has a **Validated** checkbox.

The validation system allows you to progressively build a long sequence without regenerating clips that have already been accepted.

When a clip is validated:

- its generated latent remains stored in the disk cache;
- the clip is locked and reused directly;
- it becomes the Motion Context source for the next clip;
- it will not be sampled again while it remains valid.

A typical workflow is:

    Clip 1 → Generate
    Clip 1 → Validate

    Clip 2 → Generate
    Clip 2 → Retry if needed
    Clip 2 → Validate

    Clip 3 → Generate
    Clip 3 → Retry if needed
    Clip 3 → Validate

This makes it possible to build a sequence one clip at a time while preserving all previously accepted generations.

Validation always forms a continuous chain from the beginning of the sequence.

For example:

    Clip 1 ✅
    Clip 2 ✅
    Clip 3 ⬜
    Clip 4 ⬜

If an earlier clip is unvalidated, every clip after it is automatically unvalidated as well, because each continuation depends on the previous generated clip.

Changing a generation parameter also invalidates the sequence when necessary. This includes changes to:

- prompt
- seed
- duration
- model
- sampling settings
- image references
- audio reference
- Motion Context settings

The affected clip and all dependent clips after it must then be regenerated.

In **clip_by_clip** mode, the Extender works on the first unvalidated clip in the sequence.

The intended workflow is therefore:

    Generate → Preview → Retry if needed → Validate → Continue

This allows long MiniMax H3 sequences to be created progressively without repeatedly regenerating clips that are already approved.

---

# Installation

## ComfyUI Manager

Search for:

    MiniMax H3 Extender

and install it directly from ComfyUI Manager.

## Manual Installation

Open a terminal in:

    ComfyUI/custom_nodes/

Then run:

    git clone https://github.com/tritant/ComfyUI_MiniMax_H3_Extender.git

Restart ComfyUI after installation.

### ℹ️ About the old Disk Join nodes

The old low-level **Motion Context Disk Join** workflow is now considered **deprecated** and is no longer actively maintained.

New workflows should use the main **MiniMax H3 Extender** node, which now handles cache management, references, trimming, validation, preview, seam correction, project Save/Load, color correction and final assembly internally.

---

Thanks again to everyone testing the node and reporting edge cases.

The Extender is becoming much more comfortable to use for long H3 sequences.


---

## FILE: THIRD_PARTY_NOTICE.md (https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/blob/HEAD/THIRD_PARTY_NOTICE.md)

Motion-context temporal anchor / payload patch logic in this build is adapted
from NikoDemon80/ComfyUI-H3-Motion-Context:
https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context

Upstream license: GPL-3.0.
The adaptation removes disk Save/Load and connects the previous sampled H3 latent
directly in RAM inside one ComfyUI DAG.

MiniMax H3 latent-upscaler architecture/normalization used by the optional
Latent Refine mode is compatible with the community model published by
LBH-123-AI:
https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler

Upstream model license: Apache-2.0.
The Extender downloads only the fixed BF16 SafeTensors checkpoint on first use
and verifies its SHA256 before loading it.

RelayVSR video super-resolution support (optional x2 post-processing) is based
on the native RelayVSR implementation in ComfyUI, using model checkpoints
repackaged by Comfy-Org:
https://github.com/kopperx/RelayVSR
https://huggingface.co/Comfy-Org/RelayVSR

RelayVSR was developed by Xijun Wang, Xin Li, Zirui Lang, Suhang Yao,
Haoran Li, and Zhibo Chen. The native x2 variant is credited to LightX2V:
https://huggingface.co/lightx2v/RelayVSR_X2

The original RelayVSR project and Comfy-Org model repository declare
Apache-2.0 licensing. Refer to the corresponding model repositories for the
terms applicable to the individual model checkpoints.

The Extender does not bundle RelayVSR weights or duplicate the RelayVSR
inference engine; it calls ComfyUI's native implementation and downloads the
Comfy-Org x2 model and FlashVAE checkpoints on demand if missing.


---

# ISSUES / PRs of tritant/ComfyUI_MiniMax_H3_Extender


### #116 [closed] ISSUE: Asset management?
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/116  by samsonsite1 2026-10-22T03:59:58Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

improve asset management.

### Problem or limitation

changing assets per clip is cumbersome.

### Proposed solution

_No response_

### Example use case

I understand the feature set is closed, so I'm looking for advice on asset management.

The problem I'm having is that my list of assets may change for each 10-second clip.

And there is only one "MiniMax H3 Reference Pack Bridge" node which is used for all clips.

Since my assets change each clip, what I really need are multiple "MiniMax H3 Reference Pack Bridge" nodes, each having their own list of reference images.

However, again, the problem is, there is only one connection.  Ideally, each clip needs its own exclusive set of ref images.  And they need to follow the current prompt number.

The second problem with this, is that the ref_audio and ref_video connections are coming off the main node.
These are assets too.  I think these need to moved into their own "Pack Bridges", so they can be setup as per-clip connections too.

Otherwise, this project is only suitable for creating videos where the assets are fixed for all clips,  and do not change.

### Additional context

Creating long videos can be frustrating due to the randonmess for the smallest prompt change, so I frequently need to insert a keyframe at the beginning of each clip to keep Minimax on track. And that keyframe is only used for that particular clip, and no where else.  So, my list of assets per clip frequently change.

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-10-22T04:48:26Z):**

Hi!

Have you noticed the **Global References** and **Local References** features already available in the Extender?

- **Global References** are shared across all clips.
- **Local References** are accessible through the **Ref** button on each clip card, allowing you to assign different image, video, and audio references to individual clips.

I'm also curious: **why are you using the Reference Pack Bridge instead of the Extender's built-in Global References panel?**

Is there a specific reason or workflow requirement that makes the Bridge more convenient for you?


> **Comment by samsonsite1 (2026-10-22T15:22:49Z):**

Thanks for the reply.
I didn't notice the **Refs** button on each clip until now.  After clicking on it, I see I can add a **Picture**, **Video**, or **Audio**.

However, I'm a bit confused on how the video works.  I need both Video and Video Audio.
When I add a video to the clip, I see that the global **ref_video_1** connection disappears.

Is this how it's suppose to work?  Are both video and video audio wired for the clip after adding a video?

>>why are you using the Reference Pack Bridge instead of the Extender's built-in Global References panel?

I'm just following the provided default workflow, and haven't explored that feature.



> **Comment by tritant (2026-10-22T19:35:04Z):**

You're correct. Local References support both video and audio, but they don't have the equivalent of `ref_video_audio_N` to associate an audio track with a specific video reference.

Also, it's perfectly normal for the corresponding global reference slot to disappear when you add a local reference. The Extender supports a maximum of **9 reference slots**, shared between global and local references, so local references take priority over their corresponding global slots.

Unfortunately, the local video/audio pairing won't be added, as development of the Extender is now complete.

Thanks for your understanding!



### #115 [closed] PR: Archive the Manual fallback as one canvas in the project autosave
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/pull/115  by JingHao-Leon 2026-10-19T17:43:17Z

## What breaks

`Auto Save Project` (Full Batch) can write a resolution into the `.ext` archive
that was never requested and never rendered.

The Manual fallback is one canvas, but the archive validated and applied it one
axis at a time. When only one axis of the fallback satisfied H3's canvas rules,
that axis was overwritten with the Manual value while the other axis kept the
auto-canvas value of the run:

```json
"settings": { "width": 1024, "height": 576 },
"resolution": {
  "manual_width": 1024,
  "manual_height": 576,
  "resolved_width": 896,
  "resolved_height": 576
}
```

`1024x576` is neither the Manual fallback the user typed (`1024x500`) nor the
canvas that was rendered (`896x576`). The project loader restores
`resolution.manual_width` / `manual_height` as the Manual fallback
(`web/extender.js`, `savedManualW` / `savedManualH`), so reloading the project
loses the typed fallback and restores the mixed pair instead.

The frames of the run that produced the archive are not affected, and no
exception is raised — the wrong canvas is written silently.

## Root cause

`extender.py`, `_capture_full_batch_project()` — the Manual fallback override
that captured the project settings (previously `extender.py:4878-4886`):

```python
manual = submitted.get("project_manual_resolution", {})
for dimension in ("width", "height"):
    value = manual.get(dimension)
    if type(value) in (int, float) and math.isfinite(value) and value == int(value) and 32 <= value <= 4096 and int(value) % 32 == 0:
        settings[dimension] = int(value)
```

Each axis was accepted on its own, so a pair that only satisfied the rule on
one axis was stored half-applied.

`extender.py:266` now validates both axes together and
`extender.py:4909` applies the pair at once, so an archived project always
stores either the Manual fallback or the canvas that was rendered.

## How it was verified

`python -m unittest discover -s tests -p "test_*.py"`

Before the fix, with the new regression test in
`tests/test_project_autosave.py::Autosave::test_manual_fallback_is_accepted_or_rejected_as_one_canvas`:

```
FAIL: test_manual_fallback_is_accepted_or_rejected_as_one_canvas (test_project_autosave.Autosave.test_manual_fallback_is_accepted_or_rejected_as_one_canvas)
----------------------------------------------------------------------
Traceback (most recent call last):
  File ".../tests/test_project_autosave.py", line 143, in test_manual_fallback_is_accepted_or_rejected_as_one_canvas
    self.assertEqual(
AssertionError: Tuples differ: (1024, 576) != (896, 576)

First differing element 0:
1024
896

- (1024, 576)
+ (896, 576) : {'width': 1024, 'height': 500}
```

After the fix:

```
test_manual_fallback_is_accepted_or_rejected_as_one_canvas (test_project_autosave.Autosave.test_manual_fallback_is_accepted_or_rejected_as_one_canvas) ... ok
```

Full suite, clean tree versus this branch:

```
clean tree : Ran 12 tests in 0.910s   FAILED (failures=1)
this branch: Ran 13 tests in 1.011s   FAILED (failures=1)
```

The one failure is identical in both runs and is not caused by this change:
`test_option_default_false_appended_and_export_hooks_after_video` asserts that
`auto_save_project` is the last required input of the Final Decode node, but
`motion_context_disk.py:5630` added `save_individual_clips` after it, so that
assertion is already outdated on `main`. This branch adds no new failure.

The JavaScript suite is unaffected and stays green:

```
node --test tests/*.test.mjs
ℹ tests 22
ℹ pass 22
ℹ fail 0
```

## Notes on the local environment

`python -m pytest tests/` cannot be used in a standalone checkout of this
repository: the repository root contains `__init__.py`, so pytest imports the
root package and fails on `import node_helpers` (ComfyUI core is not installed
here). The unittest discovery command above is used instead, which is also the
only runner that does not need the ComfyUI runtime.

The regression test is therefore a unit test of the archive-capture path; I
could not run a full ComfyUI render locally, because ComfyUI core is not
installed in this environment.

Fixes #114

This change was prepared with the assistance of an AI coding agent.


> **Comment by tritant (2026-10-20T11:17:31Z):**

Thanks for the fix!

I reviewed the Manual Save and Auto Save paths more closely.

The native Width and Height widgets are already configured with:

```python
"default": 896,
"min": 32,
"max": 4096,
"step": 32,
```

This means Manual resolutions entered through the normal UI are already constrained to the required 32-pixel grid.

The Manual Save path preserves the stored Manual width and height together, whereas the Auto Save path previously validated and replaced them independently.

Your fix addresses that inconsistency correctly by validating the entire Manual canvas before applying either dimension.

The `1024 × 500` example is useful for testing malformed or externally supplied project data, but it is not a resolution the standard UI is designed to produce.

Therefore, I agree with keeping the 32-pixel alignment validation in `_validated_manual_canvas()`.

No additional changes are needed for this issue. The fix looks good to merge.

Thanks for the contribution!



### #114 [closed] ISSUE: [Bug] Auto Save Project can archive a half-applied Manual resolution (mixed manual/auto canvas)
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/114  by JingHao-Leon 2026-10-19T17:41:43Z

## Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

## Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

## MiniMax H3 Extender version

3.0.12

## ComfyUI frontend

Both

## Generation mode

Ref2VA - Motion Context ON

## Run mode

Full Batch

## Describe the issue

Saving a Full Batch project with **Auto Save Project** enabled can archive a
resolution that was never requested and was never rendered.

The Manual fallback is a single canvas, but the archive validates and applies it
**one axis at a time**. When only one axis of the fallback satisfies the H3
canvas rules, that axis is overwritten with the Manual value while the other
axis keeps the auto-canvas value that was actually rendered.

Concrete case: the rendered canvas is the auto canvas `896x576`, and the Manual
fallback is `1024x500` (1024 is on H3's 32-pixel grid, 500 is not).

The produced `.ext` project then contains:

```json
"settings": { "width": 1024, "height": 576 },
"resolution": {
  "manual_width": 1024,
  "manual_height": 576,
  "resolved_width": 896,
  "resolved_height": 576
}
```

`1024x576` is neither the Manual fallback the user typed nor the canvas that
was rendered — it is a mix of the two.

What I expected to happen instead: the archived `width`/`height` pair should
always be one real canvas, i.e. either the Manual fallback or the canvas that
was rendered. A fallback that is not usable on one axis should not be archived
half-applied.

The same half-applied pair is reachable in the mirror direction (valid height,
invalid width) and when one axis is missing entirely.

## Steps to reproduce

1. Put the Extender in Auto resolution mode with a reference image, so the
   auto canvas resolves to something that is not the Manual fallback
   (e.g. `896x576`).
2. Type a Manual fallback whose width is on the 32-pixel grid and whose height
   is not, e.g. `1024` x `500` (`web/extender.js` sends it with the queued job as
   `project_manual_resolution`).
3. Run a Full Batch with **Auto Save Project** enabled.
4. Open the produced `.ext` archive and read
   `extender.settings.width` / `extender.settings.height` and
   `extender.resolution.manual_width` / `extender.resolution.manual_height`.
5. Reload that project in the Extender: the Manual fallback is now the mixed
   `1024x576` pair, and the `500` that was typed is gone.

## Error message / console log

No error is raised — the wrong canvas is written silently into the archive.

## Root cause

`extender.py`, `_capture_full_batch_project()` — the Manual fallback override
applied to the captured project settings:

```python
manual = submitted.get("project_manual_resolution", {})
for dimension in ("width", "height"):
    value = manual.get(dimension)
    if type(value) in (int, float) and math.isfinite(value) and value == int(value) and 32 <= value <= 4096 and int(value) % 32 == 0:
        settings[dimension] = int(value)
```

Each axis is accepted on its own, so a pair that only satisfies the rule on one
axis is stored half-applied: the accepted axis is replaced by the Manual value
and the rejected axis silently keeps the auto-canvas value.

`settings["width"]` / `settings["height"]` are what end up in
`resolution.manual_width` / `resolution.manual_height`, and those are the
values the project loader restores as the Manual fallback
(`web/extender.js`, `savedManualW` / `savedManualH`).

## Workflow / additional information

This only affects the metadata written into the `.ext` project archive. The
frames of the run that produced the archive are not affected, and no
exception is raised.


> **Comment by tritant (2026-10-20T08:11:37Z):**

Thanks for the fix!

I reviewed the Manual Save and Auto Save paths more closely.

The native Width and Height widgets are already configured with:

```python
"default": 896,
"min": 32,
"max": 4096,
"step": 32,
```

This means Manual resolutions entered through the normal UI are already constrained to the required 32-pixel grid.

The Manual Save path preserves the stored Manual width and height together, whereas the Auto Save path previously validated and replaced them independently.

Your fix addresses that inconsistency correctly by validating the entire Manual canvas before applying either dimension.

The `1024 × 500` example is useful for testing malformed or externally supplied project data, but it is not a resolution the standard UI is designed to produce.

Therefore, I agree with keeping the 32-pixel alignment validation in `_validated_manual_canvas()`.

No additional changes are needed for this issue. The fix looks good to merge.

Thanks for the contribution!



### #113 [closed] ISSUE: [Feature]
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/113  by guangliu 2026-10-17T01:26:01Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

1. In "clip-by-clip" mode, is it possible to avoid automatically merging the preceding content into the subsequent segments? While the generation process could still reference earlier content, there is no need to combine everything into a single file. This would improve efficiency, and users could easily merge the segments into a final video themselves later—what do you think?
2. In "ref2va" mode, could the prompt input field support referencing source images, audio, or video files directly using an "@" symbol?
3. The current prompt input interface is somewhat cumbersome; managing more than five prompts becomes quite difficult. Could an index be added above the prompt fields? Clicking an index item would allow for the editing of the corresponding prompt, which would significantly boost editing efficiency.
4. In "fl2va" mode, an error seems to occur during the final step—the synthesis and export of the video. Could you please look into this when you have some time?
Once again, thank you very much for all your hard work.

### Problem or limitation

none

### Proposed solution

_No response_

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-10-17T06:20:41Z):**

Thanks for the feedback.

1. Clip independence and merging: Both options already exist. You can disable Motion Context using Motion OFF to generate independent clips, and use Save Individual Clips in Final Decode / Preview to export them separately.

2. References: Global and per-clip local references are already supported, including image, video, and audio references.

3. Prompt navigation: Each clip already has its own numbered card and editable prompt.

4. FL2VA error: You mentioned that an error "seems to occur" during final export. Have you actually encountered an error? If so, please provide the exact error message, console logs, and steps to reproduce it. I can't investigate a suspected bug without a confirmed issue.
 
As stated in the README, active feature development has ended. I'm still open to fixing reproducible bugs, but I need concrete information to investigate them.

Please take the time to explore the existing features and the readme before submitting feature requests. Most of what you're asking for is already implemented.



### #112 [closed] ISSUE: [Bug] Save Project no active
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/112  by Vini76170 2026-10-15T15:17:26Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

last

### ComfyUI version

_No response_

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Clip-by-Clip

### Describe the issue

Hello, I’m using this node—which I find quite nice and likely very comprehensive—on my ComfyUI desktop platform. When I try to save a project, the interface doesn’t show any window or prompt for a filename—nothing at all! I have no way of knowing if my project has actually been saved. When I try to load a project, a window opens to my "Downloads" folder, but there are no saved JSON files to be found... Could someone please help?

### Steps to reproduce

No

### Error message / console log

```shell
Aucun
```

### Workflow / additional information

_No response_

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-10-15T16:59:49Z):**

Hi! Thanks for the feedback!

When you click **Save Project**, a browser dialog should appear asking you to enter a filename. Once confirmed, the project is saved as a JSON file.

If nothing appears, your browser may be blocking or suppressing JavaScript dialogs. Please check your browser settings or try another browser.

If the filename dialog never appears, the project isn't being saved, which would explain why you can't find any JSON files in your Downloads folder.

Could you also check your browser's developer console (F12 → Console) and see if any errors appear when you click **Save Project**?



> **Comment by Vini76170 (2026-10-16T14:05:56Z):**

Thank you for your response.
I am very interested in your work and am considering making your node
central to my project, but I need to test it thoroughly first. I find it
efficient and visually appealing. In this regard, is there any
documentation, a link, or a PDF available covering its usage and
configuration?
As for my issue, I will look into my browser; it is quite curious that the
browser window opens correctly when loading, but not when saving. It is
possible that the desktop version is blocking something.
Please let me know if I can continue to use this channel to keep you
updated... Thank you very much.

Le jeu. 15 oct. 2026, 19:00, tritant ***@***.***> a écrit :

> Closed #112
> <https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/112> as
> completed.
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/112?email_source=notifications&email_token=BU7FIKZ6HLO6RMLBAQPKTPD5UD7DDA5CNFSNUABQM5UWIORPF5TWS5BNNB2WEL2JONZXKZKFOZSW45CON52GSZTJMNQXI2LPNYXTGMZTGU3DQNBZGQ2DTJTSMVQXG33OUZQXK5DIN5ZKKZLWMVXHJLDGN5XXIZLSL5RWY2LDNM#event-33356849449>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/BU7FIKZ5ACMXW45VI2IEHXD5UD7DDAVCNFSNUABGKJSXA33TNF2G64TZHMYTGMZUHAYTGMZZHA5US43TOVSTWNJYGY4DQNBZGQZTTILWAI>
> .
> Triage notifications, keep track of coding agent tasks and review pull
> requests on the go with GitHub Mobile for iOS
> <https://github.com/notifications/mobile/ios/BU7FIK6O4O6DQZJK46ADQJL5UD7DDA5CNFSNUABQM5UWIORPF5TWS5BNNB2WEL2JONZXKZKFOZSW45CON52GSZTJMNQXI2LPNYXTGMZTGU3DQNBZGQ2DTJTSMVQXG33OUZQXK5DIN5ZKKZLWMVXHJKTGN5XXIZLSL5UW64Y>
> and Android
> <https://github.com/notifications/mobile/android/BU7FIK3MBSAWMFQESD6YPD35UD7DDA5CNFSNUABQM5UWIORPF5TWS5BNNB2WEL2JONZXKZKFOZSW45CON52GSZTJMNQXI2LPNYXTGMZTGU3DQNBZGQ2DTJTSMVQXG33OUZQXK5DIN5ZKKZLWMVXHJLTGN5XXIZLSL5QW4ZDSN5UWI>.
> Download it today!
> You are receiving this because you authored the thread.Message ID:
> <tritant/ComfyUI_MiniMax_H3_Extender/issue/112/issue_event/33356849449@
> github.com>
>



> **Comment by tritant (2026-10-16T14:16:30Z):**

Apart from the README file—which is worth reading as it summarizes quite a few features—there is no documentation. I cannot provide an answer regarding the desktop version, as I have never tried it; however, what is certain is that there is a significant difference between loading data and writing to disk—write restrictions are much stricter!


> **Comment by OnMYLai (2026-10-26T15:56:11Z):**

I am also using the Windows Desktop version of ComfyUI, and the save project button never works as well.
I'd like to know, is it possible to manually backup the project?
Json can be saved into the workflow, and I see there's a cache folder under minimax H3 extender, I wonder if I can just copy and paste it?


> **Comment by tritant (2026-10-26T17:53:47Z):**

Thanks for confirming the issue!

I think I've found a possible explanation for why Save Project doesn't work in ComfyUI Desktop.

ComfyUI Desktop uses Electron, which doesn't support standard JavaScript dialogs such as window.prompt(), window.confirm(), and window.alert().

Fortunately, ComfyUI provides its own dialog API, compatible with both Desktop and regular browsers.

For example, the fix could be as simple as replacing:

const filename = window.prompt("Enter project name");

With:

const filename = await app.extensionManager.dialog.prompt({
    title: "Save Project",
    message: "Enter a project name"
});

The same principle applies to confirmation dialogs:

// Old
const confirmed = window.confirm("Are you sure?");

// New
const confirmed = await app.extensionManager.dialog.confirm({
    title: "Confirmation",
    message: "Are you sure?"
});

These are illustrative changes, not a tested patch yet.

I'll investigate and test the necessary modifications when I have some free time. This looks promising, but I can't guarantee it will resolve the issue until I've tested it.


> **Comment by tritant (2026-10-26T18:21:03Z):**

@OnMYLai 

I've pushed a fix for the **Save Project popup issue in ComfyUI Desktop**.

The fix now uses ComfyUI's native dialog API instead of `window.prompt()`, which isn't supported properly in the Desktop version.

Could you please update the Extender, test the Save Project button, and let me know if it works correctly on your end?

Thanks!


> **Comment by OnMYLai (2026-10-27T00:09:11Z):**

@tritant 
The update didnt appear in ComfyUI's extension manager yet, so I downloaded and replaced the extender.js you've updated, and now it works pefectly! Thanks alot!


> **Comment by tritant (2026-10-27T07:08:10Z):**

Nice, since it works, I’ll update the manager.
Have fun !



### #111 [closed] ISSUE: [Feature] Add support for MiniMax H3 Mod
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/111  by ssuukk 2026-10-14T10:18:44Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

MiniMax H3 Mod plugin allows creating tensors from multiple reference inputs of a single subject and then connecting them to a single reference input, this makes referencing subjects much more robust and saves inputs! All it requires is to add conditioning input to your Minimax H3 Extender node (if possible), as this is the output the H3 Mod produces.

### Problem or limitation

No conditioning input.

### Proposed solution

_No response_

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-10-14T10:41:17Z):**

This has already been discussed in previous issues. Please check the existing discussions before opening a new feature request.

MiniMax H3 already supports character sheets, allowing multiple views of the same character to be used within a single image reference. Personally, I don't see enough additional benefit in RefMod to justify integrating it into the Extender.

Also, feature development for MiniMax H3 Extender has ended. The project is now in maintenance mode, with only bug fixes planned.

I'll close this as not planned. Thanks for understanding!



### #110 [closed] ISSUE: [Feature] Cache folder management?
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/110  by samsonsite1 2026-10-12T21:54:24Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

enhancement to cache folder management


### Problem or limitation

extending usefulness of cache folder

### Proposed solution

I would very much like the cache folder to be written to Comfy's default output folder (as a default path), 
instead of the custom_nodes folder.  custom_nodes shouldn't be writing cache data to the custom_nodes folder, in any case:

motion_context_disk.py:

from pathlib import Path
import folder_paths

output_dir = folder_paths.get_output_directory()
_CACHE_ROOT = Path(output_dir) / "cache"


It would also be useful to have the actual name of the cache folder listed as a "MiniMax H3 Extender" node parameter,
like:
cache_folder_filename -> "cache"

which would be configurable.

So, someone could have multiple cache folders for multiple extender projects.  Because right now, the cache folder is 
overwritten whenever a new project is started, and all older cache data is lost.  Thanks.

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-10-12T22:40:48Z):**

The cache folder is intended as temporary working storage for the active project.

If you want to preserve your generated clips and their caches, you should use **Save Project**. You can then restore everything later using **Load Project**.

Moving the cache folder to ComfyUI's output directory wouldn't change how this works. Starting a new project would still replace the active cache.

The project save/load system already handles multiple projects, so there's no need to manually manage different cache folder names.

Also, keeping cache files on their own is pointless. Each cache is tied to the specific generation parameters used when it was created. Without the corresponding project configuration, those cached files cannot simply be reused with another project.

I don't plan to add a configurable cache directory, as it would duplicate functionality already provided by the project management system.



### #109 [closed] PR: Experimental/refine memory
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/pull/109  by tritant 2026-10-09T16:21:45Z




### #108 [closed] ISSUE: Can the activation and deactivation of motion context mode be made flexible?
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/108  by AA-mobP 2026-10-09T12:37:27Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

Can the activation and deactivation of motion context mode be made flexible?

### Problem or limitation

I am currently creating a long video. For some clips, I need to disable the motion context mode as it isn't required, but for others within the same video, I need to enable it to ensure motion/spatial consistency. The issue arose when I had already finished two clips with the motion mode turned off; when I subsequently enabled it to maintain consistency between the second and third clips, the validation indicator disappeared from the first two clips. When I tried to mark them as valid again, I received the following error message: "comfy desktop> local reference invalidation failed: invalid clip index."

### Proposed solution

_No response_

### Example use case

_No response_

### Additional context

Sometimes, disabling the motion context mode is important for me because enabling it causes the new clip to overlap with part of the previous one; this leads to the loss of certain moments and confuses the model. Therefore, turning it off is crucial in those cases. However, within the same video, I sometimes need to enable the mode to maintain spatial/motion consistency. I would be very grateful if you could add this feature.

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-10-09T14:19:23Z):**

Thanks for the suggestion.

This is the intended behavior. Switching from Motion OFF to Motion ON invalidates existing clips because they were generated independently, without the continuity information required by Motion Context.

Mixing Motion ON and Motion OFF clips within the same project would require a fundamental redesign of the dependency, caching, and invalidation systems, with significant risks of regressions.

**This feature will never be implemented.**

As previously announced, MiniMax H3 Extender is now feature-complete. Development is limited to fixing major bugs, and no new features will be added.

Motion Context will remain a global setting, and its current behavior will not change.



### #107 [closed] ISSUE: [Feature]  Feature/Question (Auto validating videos)
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/107  by TheDarkKnightBG 2026-10-08T00:38:18Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

Hi. I hope my question/suggestion doesn't sound stupid. I see that you no longer plan to implement new features so if you refuse I understand. 

I need my 4090 during the day for my work, I can only generate during the night. With full-batch mode I can only make a few videos on decent resolution (1.0 + 1.2X on the refinement stage) before running out of VRAM. Meanwhile when it's clip-by-clip it can basically go forever.

Can you add a button that when clicked would auto validate the currently generated video by the end of it's run so we can hit RUN in comfy multiple times and that way QUE all the videos for the night.

OR alternatively, can you give me some suggestion of how I can reach that result (in whatever ways this can happen) on my end without you adding this as a feature. 

I use open source software quite a lot, I'm not rich but I like to spend 5-10$ to projects I use and I'm really happy  with the results from the workflow and I'm regretful for the work you have put into it. So if you have some type of donation account on other platforms let me know.

### Problem or limitation

Auto validating videos.

### Proposed solution

_No response_

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-10-08T07:07:54Z):**

Thanks for the report. Before looking at the workflow itself, I’d first like to understand why Full Batch is running out of VRAM in your case.
The actual H3 generation path is essentially the same between Full Batch and Clip-by-Clip. The main difference is that Full Batch processes several clips within the same ComfyUI execution, while Clip-by-Clip ends the execution after each clip and therefore gives ComfyUI an opportunity to perform its normal cleanup/offload before the next queued run.
On my side, with a 16 GB GPU, I’ve already rendered sequences of around 32 clips in Full Batch without running into a progressive VRAM problem, so I’m not yet convinced there is a general Full Batch memory leak.
Could you tell me:
- which mode you are using: Ref2VA Motion ON, Motion OFF or FL2VA;
- exactly how many clips Full Batch completes before the OOM;
- the duration of each clip;
- the Motion Context length;
- model precision / quantization and any LoRAs being used;
- and ideally paste the complete OOM traceback/log.
The most useful information in the log is where the OOM actually occurs: base sampling, Refine sampling, latent upscale, or the per-clip VideoVAE decode/cache.
My first suspicion is simply that the workload may be too close to the VRAM limit for a long uninterrupted batch. Clip duration has a large impact here — for example, if you are generating 15-second clips close to the VRAM limit, dropping them to 10 seconds may already be enough to make a long Full Batch stable.
Once I know how many clips you are generating, their duration, and where the OOM happens, it should be much easier to tell whether this is an actual accumulation between clips or simply the workload exceeding the available VRAM margin.


> **Comment by TheDarkKnightBG (2026-10-08T11:22:42Z):**

- Ref2VA Motion ON
- 5S long (each video, no exceptions)
- context length 22
- Lora: MiniMax H3 NSFW Unlocked (strenght 0.5)
No tubo loras
Models:
minimax_h3_ref2va_pruned_int8_convrot
minimax_h3_fl2va_pruned_int8_convrot

A bit more context from the testing from this week. I was able to 1.0MP + 1.4X on the refinement when doing clip by clip. I thought it will be fine so I kept that value, switched to full batch and it failed, so I lowered it to 1.3 and it failed, same result with 1.2.  I know that full batch works more than fine when it's a low res so say 0.2MP + 2X.  I haven't tested at 1MP + lower than 1.2X since clip per clip can do decent amount higher numbers. I decided not to test further down than 1.2 and instead open the issue here in hopes there is a way around this so I can keep the higher res of clip by clip.

"exactly how many clips Full Batch completes before the OOM"

To be sure I tested again. Already had 15 videos generate on clip-to-clip, so I switched to full batch and it failed on the 2nd video, after 33 minutes, a single video takes around 20 min, so it must have failed on the refinement stage of the 2nd video.

Log:




Not enough GPU memory. Try reducing image resolution or batch size and run again.


# ComfyUI Error Report
## Error Details
- **Node ID:** 61
- **Node Type:** MiniMaxH3Extender
- **Exception Type:** torch.OutOfMemoryError
- **Exception Message:** torch.OutOfMemoryError: Allocation on device 
This error means you ran out of memory on your GPU.

TIPS: If the workflow worked before you might have accidentally set the batch_size to a large number.
## Stack Trace
```
  File "C:\Users\PC\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 547, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 352, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 171, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 326, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "C:\Users\PC\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 314, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 4769, in wrapped
    result = function(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 6526, in extend
    sampled, refine_trim, _refine_w, _refine_h = _refine_ref2va_sample(
                                                 ^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 2668, in _refine_ref2va_sample
    up_latent, out_w, out_h, first_audio = upscale_for_refine(
                                           ^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\latent_refine_engine.py", line 13, in upscale_for_refine
    up_video, width, height = upscale_video_latent(video, scale, progress=progress)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\utils\_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\latent_upscaler.py", line 290, in upscale_video_latent
    out = model(x, float(scale), (t, h2, w2), enable_chunking=True)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\latent_upscaler.py", line 206, in forward
    seg_out = self._segment(seg, scale, (hi - lo, target_size[-2], target_size[-1]))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\latent_upscaler.py", line 177, in _segment
    x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\latent_upscaler.py", line 134, in forward
    return self.skip(x) + self.out_layers(h)
                          ^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\container.py", line 250, in forward
    input = module(input)
            ^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\conv.py", line 717, in forward
    return self._conv_forward(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\conv.py", line 712, in _conv_forward
    return F.conv3d(
           ^^^^^^^^^

```
## System Information
- **ComfyUI Version:** 0.38.2
- **Arguments:** ComfyUI\main.py --feature-flag show_signin_button=true --base-directory C:\Users\PC\Documents\ComfyUI --user-directory C:\Users\PC\Documents\ComfyUI\user --database-url sqlite:///C:\Users\PC\Documents\ComfyUI\user\comfyui.db --port 8000 --enable-manager --extra-model-paths-config C:\Users\PC\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1781127060470.yaml --input-directory C:\Users\PC\Documents\ComfyUI\input --output-directory C:\Users\PC\Documents\ComfyUI\output
- **OS:** win32
- **Python Version:** 3.12.9 (main, Feb 12 2025, 14:52:31) [MSC v.1942 64 bit (AMD64)]
- **Embedded Python:** false
- **PyTorch Version:** 2.9.1+cu130
## Devices

- **Name:** cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync
  - **Type:** cuda
  - **VRAM Total:** 25756696576
  - **VRAM Free:** 24064688128
  - **Torch VRAM Total:** 33554432
  - **Torch VRAM Free:** 25034752

## Logs
```
2026-10-08T12:37:21.754707 -   File "C:\Users\PC\AppData\Roaming\uv\python\cpython-3.12.9-windows-x86_64-none\Lib\socket.py", line 978, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 - socket.gaierror: [Errno 11004] getaddrinfo failed
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 - 
The above exception was the direct cause of the following exception:

2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 - Traceback (most recent call last):
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 -   File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\comfyui_manager\glob\manager_server.py", line 2131, in get_cache
    json_obj = await manager_util.get_data(uri, True)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 -   File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\comfyui_manager\common\manager_util.py", line 197, in get_data
    async with session.get(uri, headers=headers) as resp:
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 -   File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\aiohttp\client.py", line 1425, in __aenter__
    self._resp: _RetType = await self._coro
                           ^^^^^^^^^^^^^^^^
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 -   File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\aiohttp\client.py", line 703, in _request
    conn = await self._connector.connect(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 -   File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\aiohttp\connector.py", line 548, in connect
    proto = await self._create_connection(req, traces, timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 -   File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\aiohttp\connector.py", line 1056, in _create_connection
    _, proto = await self._create_direct_connection(req, traces, timeout)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 -   File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\aiohttp\connector.py", line 1357, in _create_direct_connection
    raise ClientConnectorDNSError(req.connection_key, exc) from exc
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:21.754707 - aiohttp.client_exceptions.ClientConnectorDNSError: Cannot connect to host raw.githubusercontent.com:443 ssl:default [getaddrinfo failed]
2026-10-08T12:37:21.754707 - 2026-10-08T12:37:22.114406 - ^[[1m^[[33m[WARNING]^[[0m Cannot connect to comfyregistry.
2026-10-08T12:37:22.114406 - FETCH DATA from: C:\Users\PC\Documents\ComfyUI\user\__manager\cache\1514988643_custom-node-list.json2026-10-08T12:37:22.114406 - 2026-10-08T12:37:22.147725 -  [DONE]2026-10-08T12:37:22.147725 - 
2026-10-08T12:37:22.189118 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] All startup tasks have been completed.
2026-10-08T12:37:22.931509 - ^[[32m[INFO]^[[0m No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
2026-10-08T12:37:24.081369 - Failed to find comfy root automatically, please copy the folder C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyLiterals\web manually in the web/extensions folder of ComfyUI2026-10-08T12:37:24.081369 - 
2026-10-08T12:37:29.580596 - ^[[1m^[[31m[ERROR]^[[0m bitsandbytes library load error: Configured CUDA binary not found at C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\bitsandbytes\libbitsandbytes_cuda130.dll
 If you are using Intel CPU/XPU, please install intel_extension_for_pytorch to enable required ops
Traceback (most recent call last):
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\bitsandbytes\cextension.py", line 318, in <module>
    lib = get_native_library()
          ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\bitsandbytes\cextension.py", line 282, in get_native_library
    raise RuntimeError(f"Configured {BNB_BACKEND} binary not found at {cuda_binary_path}")
RuntimeError: Configured CUDA binary not found at C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\bitsandbytes\libbitsandbytes_cuda130.dll
2026-10-08T12:37:31.864275 - ^[[34m[ComfyUI-Easy-Use] server: ^[[0mv1.3.6 ^[[92mLoaded^[[0m2026-10-08T12:37:31.864275 - 
2026-10-08T12:37:31.864275 - ^[[34m[ComfyUI-Easy-Use] web root: ^[[0mC:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-easy-use\web_version/v2 ^[[92mLoaded^[[0m2026-10-08T12:37:31.864275 - 
2026-10-08T12:37:31.895519 - ^[[32m[INFO]^[[0m ComfyUI-GGUF: Allowing full torch compile
2026-10-08T12:37:31.931139 - ^[[32m[INFO]^[[0m [LoRA-Manager] Found LoRA roots:
 - C:/Users/PC/ComfyUI-Shared/models/loras
 - C:/Users/PC/Documents/ComfyUI/models/loras
2026-10-08T12:37:31.931139 - ^[[32m[INFO]^[[0m [LoRA-Manager] Found checkpoint roots:
 - C:/Users/PC/ComfyUI-Shared/models/checkpoints
 - C:/Users/PC/ComfyUI-Shared/models/diffusion_models
 - C:/Users/PC/ComfyUI-Shared/models/unet
 - C:/Users/PC/Documents/ComfyUI/models/checkpoints
 - C:/Users/PC/Documents/ComfyUI/models/diffusion_models
 - C:/Users/PC/Documents/ComfyUI/models/unet
2026-10-08T12:37:31.931139 - ^[[32m[INFO]^[[0m [LoRA-Manager] Found embedding roots:
 - C:/Users/PC/ComfyUI-Shared/models/embeddings
 - C:/Users/PC/Documents/ComfyUI/models/embeddings
2026-10-08T12:37:31.931139 - ^[[32m[INFO]^[[0m [LoRA-Manager] Symlink cache loaded with 0 mappings
2026-10-08T12:37:31.931139 - ^[[32m[INFO]^[[0m [LoRA-Manager] Symlink mappings restored from cache in 5.32 ms
2026-10-08T12:37:31.972717 - ^[[32m[INFO]^[[0m [LoRA-Manager] Updated 'comfyui' library with current folder paths
2026-10-08T12:37:32.230892 - ^[[32m[INFO]^[[0m [LoRA-Manager] Detected async ComfyUI execution, installing async metadata hooks
2026-10-08T12:37:32.230892 - ^[[32m[INFO]^[[0m [LoRA-Manager] Metadata collection hooks installed for runtime values
2026-10-08T12:37:32.230892 - ^[[32m[INFO]^[[0m [LoRA-Manager] ComfyUI Metadata Collector initialized
2026-10-08T12:37:32.230892 - ^[[32m[INFO]^[[0m [LoRA-Manager] Example images path: 
2026-10-08T12:37:32.230892 - ^[[32m[INFO]^[[0m [LoRA-Manager] Added static route for locales: /locales -> C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-lora-manager\locales
2026-10-08T12:37:32.330934 - Failed to auto update `Quality of Life Suit` 2026-10-08T12:37:32.330934 - 
2026-10-08T12:37:32.330934 - ^[[33mQualityOfLifeSuit_Omar92_DIR:^[[0m C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI-QualityOfLifeSuit_Omar922026-10-08T12:37:32.330934 - 
2026-10-08T12:37:32.531038 - [Smart Merge] 成功将模型下载和缓存目录重定向至: C:\Users\PC\Documents\ComfyUI\models\elementEasy2026-10-08T12:37:32.531038 - 
2026-10-08T12:37:32.563830 - ^[[92m[PromptModels Studio] 🚀 Loading custom nodes for ComfyUI...^[[0m2026-10-08T12:37:32.563830 - 
2026-10-08T12:37:32.563830 - [ComfyDeploy] ✅ get_last_frame importado correctamente2026-10-08T12:37:32.563830 - 
2026-10-08T12:37:32.563830 - ^[[92m[GetLastFrame] ✅ Loaded^[[0m2026-10-08T12:37:32.563830 - 
2026-10-08T12:37:32.563830 - ^[[92m[TextPromptBlocker] ✅ Loaded^[[0m2026-10-08T12:37:32.563830 - 
2026-10-08T12:37:32.563830 - ^[[92m[DivisorDePrompts] ✅ Loaded^[[0m2026-10-08T12:37:32.563830 - 
2026-10-08T12:37:32.563830 - ✓ GETSETNODE_PRO v1.0.0 loaded2026-10-08T12:37:32.563830 - 
2026-10-08T12:37:32.563830 - ^[[92m[GETSETNODE_PRO] ✅ Loaded^[[0m2026-10-08T12:37:32.563830 - 
2026-10-08T12:37:32.563830 - ^[[32m[INFO]^[[0m [GoogleAI] Ruta registrada: /google-ai/health
2026-10-08T12:37:32.563830 - 
=================================================================
  ComfyUI_GoogleAI V2.5.1 -- 12 nodos
  Texto  | Vision (5 imagenes)
  Imagen: Nano Banana 2/Pro + Imagen 4 (14 aspect ratios)
  Veo 3.1 + Audio ffmpeg | Diagnostico (5 nodos)
  Video: transcodificacion H.264 + diagnostico automatico
  Error Explainer -> ComfyUI_UniversalErrorExplainer
=================================================================
2026-10-08T12:37:32.563830 - 
2026-10-08T12:37:32.563830 - ^[[92m[ComfyUI-GoogleAI] ✅ Loaded (V2.5.0 — 12 Nodos: NB2/Pro/Imagen4/Veo3.1/Diag)^[[0m2026-10-08T12:37:32.563830 - 
2026-10-08T12:37:32.580467 - ^[[92m[ComfyUI-GrokAI] ✅ Loaded (xAI Integration)^[[0m2026-10-08T12:37:32.580467 - 
2026-10-08T12:37:32.583907 - ✅ ComfyUI Selectores Pro cargado (4 nodos)2026-10-08T12:37:32.583907 - 
2026-10-08T12:37:32.583907 - ^[[92m[Selectores Pro] ✅ Loaded^[[0m2026-10-08T12:37:32.583907 - 
2026-10-08T12:37:32.583907 - ^[[93m[PromptModels Studio] 📦 Total nodes loaded: 38^[[0m2026-10-08T12:37:32.583907 - 
2026-10-08T12:37:32.583907 - ^[[92m[PromptModels Studio] ✅ Ready!^[[0m2026-10-08T12:37:32.583907 - 
2026-10-08T12:37:32.621986 - 
2026-10-08T12:37:32.621986 - ^[[92m[rgthree-comfy] Loaded 48 epic nodes. 🎉^[[0m2026-10-08T12:37:32.621986 - 
2026-10-08T12:37:32.621986 - 
2026-10-08T12:37:32.621986 - ^[[33m[rgthree-comfy] ComfyUI's new Node 2.0 rendering may be incompatible with some rgthree-comfy nodes and features, breaking some rendering as well as losing the ability to access a node's properties (a vital part of many nodes). It also appears to run MUCH more slowly spiking CPU usage and causing jankiness and unresponsiveness, especially with large workflows. Personally I am not planning to use the new Nodes 2.0 and, unfortunately, am not able to invest the time to investigate and overhaul rgthree-comfy where needed. If you have issues when Nodes 2.0 is enabled, I'd urge you to switch it off as well and join me in hoping ComfyUI is not planning to deprecate the existing, stable canvas rendering all together.
^[[0m2026-10-08T12:37:32.621986 - 
2026-10-08T12:37:32.630504 - ⚠️  SeedVR2 optimizations check: SageAttention ❌ | Flash Attention ❌ | Triton ❌2026-10-08T12:37:32.630504 - 
2026-10-08T12:37:32.630504 - 💡 For best performance: pip install sageattention flash-attn triton2026-10-08T12:37:32.630504 - 
2026-10-08T12:37:32.630504 - 🔧 Conv3d workaround active: PyTorch 2.9.1, cuDNN 91200 (fixing VAE 3x memory bug)2026-10-08T12:37:32.630504 - 
2026-10-08T12:37:32.697951 - 📊 Initial CUDA memory: 22.39GB free / 23.99GB total2026-10-08T12:37:32.697951 - 
2026-10-08T12:37:33.530938 - ^[[34mWAS Node Suite: ^[[0mOpenCV Python FFMPEG support is enabled^[[0m2026-10-08T12:37:33.530938 - 
2026-10-08T12:37:33.530938 - ^[[34mWAS Node Suite ^[[93mWarning: ^[[0m`ffmpeg_bin_path` is not set in `C:\Users\PC\Documents\ComfyUI\custom_nodes\was-ns\was_suite_config.json` config file. Will attempt to use system ffmpeg binaries if available.^[[0m2026-10-08T12:37:33.530938 - 
2026-10-08T12:37:34.133207 - ^[[34mWAS Node Suite: ^[[0mFinished.^[[0m ^[[32mLoaded^[[0m ^[[0m220^[[0m ^[[32mnodes successfully.^[[0m2026-10-08T12:37:34.133207 - 
2026-10-08T12:37:34.133207 - 
	^[[3m^[[93m"Success is not just about making money. It's about making a difference."^[[0m^[[3m - Unknown^[[0m
2026-10-08T12:37:34.133207 - 
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m 
Import times for custom nodes:
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\wanblockswap
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-mxtoolkit
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyLiterals
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI-GGUF
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfy-plasma
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui_essentials
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-minimax-h3-latent-upscaler
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI-QualityOfLifeSuit_Omar92
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_Element_easy
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-custom-scripts
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-frame-interpolation
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\promptmodels
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-kjnodes
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-videohelpersuite
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.0 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\rgthree-comfy
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.1 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI-MediaMixer
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.1 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-various
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.3 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\seedvr2_videoupscaler
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    0.3 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-lora-manager
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    1.2 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\was-ns
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m    7.8 seconds: C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-easy-use
2026-10-08T12:37:34.133207 - ^[[32m[INFO]^[[0m 
2026-10-08T12:37:34.153242 - ^[[32m[INFO]^[[0m Context impl SQLiteImpl.
2026-10-08T12:37:34.153242 - ^[[32m[INFO]^[[0m Will assume non-transactional DDL.
2026-10-08T12:37:34.428734 - ^[[32m[INFO]^[[0m Using RAM pressure cache.
2026-10-08T12:37:34.673846 - ^[[32m[INFO]^[[0m Starting server

2026-10-08T12:37:34.673846 - ^[[32m[INFO]^[[0m [LoRA-Manager] Lora cache hydrated from persisted snapshot with 52 models
2026-10-08T12:37:34.673846 - ^[[32m[INFO]^[[0m To see the GUI go to: http://127.0.0.1:8000
2026-10-08T12:37:34.673846 - ^[[32m[INFO]^[[0m [LoRA-Manager] Checkpoint cache hydrated from persisted snapshot with 5 models
2026-10-08T12:37:34.673846 - ^[[32m[INFO]^[[0m [LoRA-Manager] Embedding cache initialized in 0.00 seconds. Found 0 models
2026-10-08T12:37:34.673846 - ^[[32m[INFO]^[[0m [LoRA-Manager] Recipe cache miss: performing full directory scan
2026-10-08T12:37:34.673846 - ^[[32m[INFO]^[[0m [LoRA-Manager] Recipe cache initialized in 0.00 seconds. Found 0 recipes
2026-10-08T12:37:34.697111 - ^[[32m[INFO]^[[0m [LoRA-Manager] FTS index validated, reusing existing index with 0 recipes
2026-10-08T12:37:36.630349 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] The ComfyRegistry cache update is still in progress, so an outdated cache is being used.
2026-10-08T12:37:36.730360 - FETCH DATA from: C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\comfyui_manager\custom-node-list.json2026-10-08T12:37:36.730360 - 2026-10-08T12:37:36.763684 -  [DONE]2026-10-08T12:37:36.763684 - 
2026-10-08T12:37:37.980299 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-10-08T12:37:37.980299 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /extensions/core/widgetInputs.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-10-08T12:37:38.446811 - ^[[33mQualityOfLifeSuit_Omar92:^[[0m:NSP ready2026-10-08T12:37:38.446811 - 
2026-10-08T12:37:38.913440 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui/components/button.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-10-08T12:39:19.895660 - ^[[32m[INFO]^[[0m got prompt
2026-10-08T12:39:19.953653 - ^[[32m[INFO]^[[0m Model storage policy: fast_disk=True paths=['C:\\Users\\PC\\ComfyUI-Shared\\models\\vae\\minimax_h3_audio_vae_fp32.safetensors']
2026-10-08T12:39:19.996613 - ^[[32m[INFO]^[[0m VAE load device: cuda:0, offload device: cpu, dtype: torch.float32
2026-10-08T12:39:20.013292 - ^[[32m[INFO]^[[0m Model storage policy: fast_disk=True paths=['C:\\Users\\PC\\ComfyUI-Shared\\models\\vae\\minimax_h3_video_vae_fp16.safetensors']
2026-10-08T12:39:20.137937 - ^[[32m[INFO]^[[0m VAE load device: cuda:0, offload device: cpu, dtype: torch.float16
2026-10-08T12:39:20.541751 - [VideoHelperSuite] - ^[[0;33mWARNING^[[0m - Failed to calculate available memory. Memory load limit has been disabled
2026-10-08T12:39:23.528576 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-10-08T12:39:23.528576 - ^[[32m[INFO]^[[0m Using MixedPrecisionOps for text encoder
2026-10-08T12:39:23.727127 - ^[[32m[INFO]^[[0m Model storage policy: fast_disk=True paths=['C:\\Users\\PC\\ComfyUI-Shared\\models\\text_encoders\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors']
2026-10-08T12:39:23.939857 - ^[[32m[INFO]^[[0m CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
2026-10-08T12:39:23.959903 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-10-08T12:39:23.959903 - ^[[32m[INFO]^[[0m Detected mixed precision quantization
2026-10-08T12:39:23.959903 - ^[[32m[INFO]^[[0m Using mixed precision operations
2026-10-08T12:39:23.959903 - ^[[32m[INFO]^[[0m Native ops: float8_e5m2, int8_tensorwise, convrot_w4a4, asym_w4a8_int8, float8_e4m3fn, w6a8_int8 , emulated ops: mxfp8, nvfp4
2026-10-08T12:39:24.218655 - ^[[32m[INFO]^[[0m model weight dtype torch.bfloat16, manual cast: torch.bfloat16
2026-10-08T12:39:24.220666 - ^[[32m[INFO]^[[0m model_type FLOW_AV
2026-10-08T12:39:24.220666 - ^[[32m[INFO]^[[0m Model storage policy: fast_disk=True paths=['C:\\Users\\PC\\Documents\\ComfyUI\\models\\diffusion_models\\minimax_h3_ref2va_pruned_int8_convrot.safetensors']
2026-10-08T12:39:24.371047 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
2026-10-08T12:39:24.379063 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=1 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.379063 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 2 exists on disk; sampler forbidden
2026-10-08T12:39:24.381073 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=2 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.381073 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 3 exists on disk; sampler forbidden
2026-10-08T12:39:24.381073 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=3 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.381073 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 4 exists on disk; sampler forbidden
2026-10-08T12:39:24.381073 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=4 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.381073 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 5 exists on disk; sampler forbidden
2026-10-08T12:39:24.381073 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=5 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.389083 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 6 exists on disk; sampler forbidden
2026-10-08T12:39:24.389083 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=6 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.391092 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 7 exists on disk; sampler forbidden
2026-10-08T12:39:24.391092 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=7 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.391092 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 8 exists on disk; sampler forbidden
2026-10-08T12:39:24.391092 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=8 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.391092 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 9 exists on disk; sampler forbidden
2026-10-08T12:39:24.391092 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=9 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.391092 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 10 exists on disk; sampler forbidden
2026-10-08T12:39:24.399107 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=10 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.399107 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 11 exists on disk; sampler forbidden
2026-10-08T12:39:24.401117 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=11 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.401117 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 12 exists on disk; sampler forbidden
2026-10-08T12:39:24.401117 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=12 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.401117 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 13 exists on disk; sampler forbidden
2026-10-08T12:39:24.401117 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=13 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.401117 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 14 exists on disk; sampler forbidden
2026-10-08T12:39:24.401117 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=14 validated=True stop=False clips=15 frames=1552 cache=655.7 MB
2026-10-08T12:39:24.539470 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-08T12:39:24.581569 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:39:25.654130 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:39:26.004925 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:39:26.616251 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:39:31.869687 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:39:44.247289 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3AudioVAE
2026-10-08T12:39:44.317423 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-08T12:39:44.836606 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-08T12:39:44.938877 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-08T12:40:17.734891 - ^[[32m[INFO]^[[0m MiniMax H3 Motion Context RAM: reasserted Extender payload patch over comfy.model_base.extra_conds
2026-10-08T12:40:17.734891 - ^[[32m[INFO]^[[0m MiniMax H3 Motion Context RAM: 22 video frames -> 7 latent conditioning blocks, 37 audio latent steps, trim=22
2026-10-08T12:40:17.835144 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-08T12:40:17.866393 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-08T12:40:17.930638 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 200 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-08T12:53:08.274980 - 
100%|██████████| 20/20 [12:37<00:00, 37.94s/it]2026-10-08T12:53:08.274980 - 
100%|██████████| 20/20 [12:37<00:00, 37.89s/it]2026-10-08T12:53:08.274980 - 
2026-10-08T12:53:16.255054 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-08T12:53:39.707361 - ^[[32m[INFO]^[[0m MiniMax H3 Motion Context RAM: 22 video frames -> 7 latent conditioning blocks, 37 audio latent steps, trim=22
2026-10-08T12:53:39.769858 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-08T12:53:39.823234 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 200 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-08T12:57:19.006478 - 
100%|██████████| 4/4 [03:32<00:00, 54.57s/it]2026-10-08T12:57:19.006478 - 
100%|██████████| 4/4 [03:32<00:00, 53.03s/it]2026-10-08T12:57:19.006478 - 
2026-10-08T12:57:19.056593 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=clip_by_clip clip=15 validated=False stop=True clips=15 frames=1552 cache=473.9 MB
2026-10-08T12:57:19.504406 - ^[[32m[INFO]^[[0m Comfy model compiler graph breaks: 6, rogues: 0
2026-10-08T12:57:22.866418 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:58:11.618590 - ^[[32m[INFO]^[[0m MiniMax H3 Motion Context Join: auto seam shift=-1 frame(s); nominal jump ratio=1.502, scores={-2: 0.16149, -1: 0.06152, 0: 0.11316}
2026-10-08T12:58:11.913665 - ^[[32m[INFO]^[[0m H3 Final Decode: H.264 NVENC available; using hardware encoder by default.
2026-10-08T12:58:13.918756 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-08T12:58:16.947427 - ^[[32m[INFO]^[[0m H3 v12.5 FULL preview: cached validated clips=14, candidate=15, shift=-1; VAE decoded only pair 14->15
2026-10-08T12:58:17.077610 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 00:18:57^[[0m
2026-10-08T12:59:18.192479 - ^[[32m[INFO]^[[0m got prompt
2026-10-08T12:59:18.249264 - ^[[32m[INFO]^[[0m H3 Full Batch Ref2VA start: validated=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False] cached_segments=15
2026-10-08T12:59:18.250507 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
2026-10-08T12:59:18.252941 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=1 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.253942 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 2 exists on disk; sampler forbidden
2026-10-08T12:59:18.255942 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=2 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.256942 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 3 exists on disk; sampler forbidden
2026-10-08T12:59:18.258943 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=3 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.258943 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 4 exists on disk; sampler forbidden
2026-10-08T12:59:18.260943 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=4 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.261943 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 5 exists on disk; sampler forbidden
2026-10-08T12:59:18.263944 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=5 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.263944 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 6 exists on disk; sampler forbidden
2026-10-08T12:59:18.265944 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=6 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.265944 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 7 exists on disk; sampler forbidden
2026-10-08T12:59:18.267598 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=7 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.268594 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 8 exists on disk; sampler forbidden
2026-10-08T12:59:18.270594 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=8 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.270594 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 9 exists on disk; sampler forbidden
2026-10-08T12:59:18.272595 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=9 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.273594 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 10 exists on disk; sampler forbidden
2026-10-08T12:59:18.274594 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=10 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.275594 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 11 exists on disk; sampler forbidden
2026-10-08T12:59:18.276595 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=11 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.277595 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 12 exists on disk; sampler forbidden
2026-10-08T12:59:18.279595 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=12 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.279595 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 13 exists on disk; sampler forbidden
2026-10-08T12:59:18.281596 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=13 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.281596 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 14 exists on disk; sampler forbidden
2026-10-08T12:59:18.283596 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=14 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.284283 - ^[[32m[INFO]^[[0m H3 validated hard-skip: Ref2VA clip 15 exists on disk; sampler forbidden
2026-10-08T12:59:18.285279 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=15 validated=True stop=False clips=15 frames=1552 cache=653.2 MB
2026-10-08T12:59:18.450916 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:59:19.413669 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:59:19.784278 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:59:20.388893 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:59:25.181447 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T12:59:37.777931 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-08T12:59:38.343966 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-08T13:00:04.446975 - ^[[32m[INFO]^[[0m MiniMax H3 Motion Context RAM: 22 video frames -> 7 latent conditioning blocks, 37 audio latent steps, trim=22
2026-10-08T13:00:04.517101 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-08T13:00:04.577217 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 200 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-08T13:13:38.976429 - 
100%|██████████| 20/20 [13:35<00:00, 41.57s/it]2026-10-08T13:13:38.976429 - 
100%|██████████| 20/20 [13:35<00:00, 40.77s/it]2026-10-08T13:13:38.976429 - 
2026-10-08T13:13:52.229592 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-08T13:14:15.891358 - ^[[32m[INFO]^[[0m MiniMax H3 Motion Context RAM: 22 video frames -> 7 latent conditioning blocks, 37 audio latent steps, trim=22
2026-10-08T13:14:15.964021 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-08T13:14:16.018653 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 200 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-08T13:18:00.180633 - 
100%|██████████| 4/4 [03:43<00:00, 57.08s/it]2026-10-08T13:18:00.180633 - 
100%|██████████| 4/4 [03:43<00:00, 55.81s/it]2026-10-08T13:18:00.180633 - 
2026-10-08T13:18:00.214270 - ^[[32m[INFO]^[[0m H3 Disk Join: mode=full_batch clip=16 validated=False stop=False clips=16 frames=1654 cache=671.9 MB
2026-10-08T13:18:00.278145 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-08T13:18:00.317712 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-08T13:18:44.900316 - ^[[32m[INFO]^[[0m MiniMax H3 Motion Context Join: auto seam shift=-1 frame(s); nominal jump ratio=1.623, scores={-2: 0.17141, -1: 0.06423, 0: 0.12856}
2026-10-08T13:18:46.971190 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-08T13:18:47.265745 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-08T13:19:11.186007 - ^[[32m[INFO]^[[0m MiniMax H3 Motion Context RAM: 22 video frames -> 7 latent conditioning blocks, 37 audio latent steps, trim=22
2026-10-08T13:19:11.257834 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-08T13:19:11.307761 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 200 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-08T13:32:15.874796 - 
100%|██████████| 20/20 [13:02<00:00, 40.59s/it]2026-10-08T13:32:15.874796 - 
100%|██████████| 20/20 [13:02<00:00, 39.10s/it]2026-10-08T13:32:15.874796 - 
2026-10-08T13:32:18.706226 - ^[[32m[INFO]^[[0m Comfy model compiler graph breaks: 9, rogues: 0
2026-10-08T13:32:18.870686 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! Allocation on device 
2026-10-08T13:32:18.948870 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "C:\Users\PC\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 547, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 352, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 171, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 326, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "C:\Users\PC\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 314, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 4769, in wrapped
    result = function(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 6526, in extend
    sampled, refine_trim, _refine_w, _refine_h = _refine_ref2va_sample(
                                                 ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 2668, in _refine_ref2va_sample
    up_latent, out_w, out_h, first_audio = upscale_for_refine(
                                           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\latent_refine_engine.py", line 13, in upscale_for_refine
    up_video, width, height = upscale_video_latent(video, scale, progress=progress)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\utils\_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\latent_upscaler.py", line 290, in upscale_video_latent
    out = model(x, float(scale), (t, h2, w2), enable_chunking=True)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\latent_upscaler.py", line 206, in forward
    seg_out = self._segment(seg, scale, (hi - lo, target_size[-2], target_size[-1]))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\latent_upscaler.py", line 177, in _segment
    x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\latent_upscaler.py", line 134, in forward
    return self.skip(x) + self.out_layers(h)
                          ^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\container.py", line 250, in forward
    input = module(input)
            ^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\conv.py", line 717, in forward
    return self._conv_forward(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\conv.py", line 712, in _conv_forward
    return F.conv3d(
           ^^^^^^^^^
torch.OutOfMemoryError: Allocation on device 

2026-10-08T13:32:18.948870 - ^[[32m[INFO]^[[0m Memory summary:
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |   1391 MiB |   7712 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Active memory         |   1391 MiB |   7712 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Requested memory      |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| GPU reserved memory   |   1472 MiB |  13888 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Non-releasable memory |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Allocations           |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Active allocs         |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| GPU reserved segments |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|

2026-10-08T13:32:18.948870 - ^[[1m^[[31m[ERROR]^[[0m Got an OOM, unloading all loaded models.
2026-10-08T13:32:20.041654 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 00:33:01^[[0m

```
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
```
Workflow too large. Please manually upload the workflow from local file system.
```

## Additional Context
(Please add any additional context or steps to reproduce the error here)









> **Comment by tritant (2026-10-08T11:49:00Z):**

Thanks for the detailed testing and logs!

The stack trace confirms that the OOM occurs inside the latent upscaler's 3D convolution, rather than during H3 sampling.

The fact that 1.0 MP + 1.4x works in Clip by Clip but Full Batch fails on the second newly generated clip, even at 1.2x, strongly suggests a VRAM management issue rather than simply excessive resolution.

I've made some changes in **v3.0.8**, which I'll be releasing shortly:

- The latent upscaler now uses ComfyUI's native memory management (`ModelPatcher` / `load_models_gpu()`) instead of managing CUDA directly.
- ComfyUI can now manage VRAM allocation between H3 and the upscaler, unloading models when necessary.
- The upscaler is properly released after each pass, including when an exception or OOM occurs.
- Memory requirements are estimated in advance to help ComfyUI free sufficient VRAM before starting the upscale.

The actual Refine process and upscale calculations remain unchanged.

**I can't guarantee this will completely resolve your OOM**, since the 3D upscaler itself can still require significant VRAM, but it addresses a memory management weakness that could explain the difference between Clip by Clip and Full Batch.

Once v3.0.8 is available, could you try your original **1.0 MP + 1.4x Refine settings in Full Batch** again and let me know how it behaves?

Thanks again for taking the time to investigate this!


> **Comment by TheDarkKnightBG (2026-10-08T12:31:44Z):**

Oh wow that was quick! Yes I will be more than happy to test it out at 1.4 (and lower if it fails at 1.4). I will be traveling early tomorrow morning so if not today i will be able to test it Sunday and write the results here.

"The actual Refine process and upscale calculations remain unchanged." Side note on this, the refinement works amazingly! Without it after 12~ videos the degradation in the quality is quite visible, with it the quality is consistent. 



> **Comment by liberateme77 (2026-10-08T20:37:25Z):**

Hi there, i have exactly the same request / problem. Also 4090 here. I can go clip by clip the whole sequence. But when i switch to full_batch it sometimes freezes between clips. When i cancel run (red button) in comfy. And run it again. It recovers the video clip and goes on another one. But it will stale again after few clips. If you need logs or anything i can provide them as well. I am basically running into this consistently. edit: btw i love this extender ! 


> **Comment by tritant (2026-10-08T21:01:24Z):**

Try version 3.0.8 and let me know if it improves things.


> **Comment by liberateme77 (2026-10-09T10:10:23Z):**

I have some results from testing the latest version. Clip-by-clip generation still works fine (up to 11s clips), but full batch mode still hangs (even on 6s clips).  

GPU utilization drops, and it takes forever to finish (Clip 4, low gpu util picture i canceled the run) —if it finishes at all. I stopped waiting after two hours (maybe spilled to RAM from VRAM -> RAM is 15x slower in my case).
**This happens right after chain_extender_61.json is updated in the cache folder and just before the new video file is generated at cache\chain_extender_61.final.video.** (in gpu usage it's the part where red arrow is pointing from second picture - correct behaviour)

I then cancel the run using the red button in ComfyUI and start it again. After restarting, the full batch run successfully finishes writing the video file :-) and continues to the next clip.

Eventualy it failed on OOM during generation of Clip 7 (marked 0006 in cache).

IMHO, full batch mode is not freeing memory correctly compared to clip-by-clip mode. Something seems to be retaining memory between clips, causing memory usage to accumulate as the batch progresses.
The obvious workaround is the one suggested above: use clip-by-clip mode and queue 100 runs in ComfyUI. However, this would require automatically advancing to the next clip. As I understand it, the current workflow requires each clip to be validated before proceeding, so it would generate the same clip repeatedly.
Another workaround I considered is disabling system RAM fallback when VRAM is exhausted in the NVIDIA Control Panel. My expectation is that this would make it fail quickly, allowing me to queue 100 runs in full batch mode and potentially resume progress after each failure.
The best solution would be to identify and fix the root cause of the increasing memory usage between clips in full batch mode.


**I’d be happy to run additional tests or provide any further information you need.**

how it looks when it fails between clips
<img width="562" height="551" alt="Image" src="https://github.com/user-attachments/assets/c846baa5-aeea-4a41-8d66-3f4d55bad662" />

**correct behaviour with critical section marked - notice video encode part**
<img width="562" height="551" alt="Image" src="https://github.com/user-attachments/assets/00b7b937-36f2-4350-b36b-6c65095caf9e" />

LOG file for this run with new version:


[INFO] got prompt
[INFO] H3 Full Batch Ref2VA start: validated=[True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] cached_segments=3
[INFO] H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=1 validated=True stop=False clips=3 frames=481 cache=246.3 MB
[INFO] H3 validated hard-skip: Ref2VA clip 2 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=2 validated=True stop=False clips=3 frames=481 cache=246.3 MB
[INFO] H3 validated hard-skip: Ref2VA clip 3 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=3 validated=True stop=False clips=3 frames=481 cache=246.3 MB
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [04:02<00:00, 30.29s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [04:17<00:00, 64.31s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=4 validated=False stop=False clips=4 frames=719 cache=300.1 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Interrupting prompt 8429bf6e-e163-4509-beba-efff3aa6c032
[INFO] Comfy model compiler graph breaks: 11, rogues: 0
[INFO] Processing interrupted
[INFO] Prompt executed in 00:13:09
[INFO] got prompt
[INFO] H3 Full Batch Ref2VA start: validated=[True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] cached_segments=4
[INFO] H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=1 validated=True stop=False clips=4 frames=719 cache=300.1 MB
[INFO] H3 validated hard-skip: Ref2VA clip 2 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=2 validated=True stop=False clips=4 frames=719 cache=300.1 MB
[INFO] H3 validated hard-skip: Ref2VA clip 3 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=3 validated=True stop=False clips=4 frames=719 cache=300.1 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 4 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=4 validated=False stop=False clips=4 frames=719 cache=300.1 MB
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [02:25<00:00, 18.19s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [02:28<00:00, 37.16s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=5 validated=False stop=False clips=5 frames=872 cache=378.3 MB
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [02:47<00:00, 20.97s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [02:27<00:00, 36.77s/it]                                    
[INFO] H3 Disk Join: mode=full_batch clip=6 validated=False stop=False clips=6 frames=1025 cache=439.5 MB
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [02:40<00:00, 20.09s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
 25%|██▌       | 1/4 [00:45<02:17, 45.96s/it,  Model Initialization complete!  ]
[INFO] Comfy model compiler graph breaks: 31, rogues: 12
[ERROR] !!! Exception during processing !!! Allocation on device 0 would exceed allowed memory. (out of memory)
Currently allocated     : 3.77 GiB
Requested               : 6.28 GiB
Device limit            : 23.99 GiB
Free (according to CUDA): 389.34 MiB
PyTorch limit (set by user-supplied memory fraction)
                        : 17179869184.00 GiB
[ERROR] Traceback (most recent call last):
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 548, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 353, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 327, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 315, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 4769, in wrapped
    result = function(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 6526, in extend
    sampled, refine_trim, _refine_w, _refine_h = _refine_ref2va_sample(
                                                 ~~~~~~~~~~~~~~~~~~~~~^
        owner=owner, clip_index=i, clip_count=len(clips), sampled=first_pass,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<9 lines>...
        context_length=context_length, audio_context_length=audio_context_length,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 2687, in _refine_ref2va_sample
    refined = _sample_h3(
        clip_model, refine_positive, up_latent, seed, sampler_name, scheduler,
        int(refine_steps), float(refine_denoise), sigmas=refine_sigmas,
    )
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 2589, in _sample_h3
    samples = guider.sample(
        noise,
    ...<6 lines>...
        seed=int(seed),
    )
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\k_diffusion\sampling.py", line 205, in sample_euler
    denoised = model(x, sigma_hat * s_in, **extra_args)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_base.py", line 210, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_base.py", line 254, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 621, in forward
    graph_out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload,
              denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 765, in _forward
    h = blocks_replace[("double_block", i)](
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         "layout": layout, "transformer_options": transformer_options},
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        {"original_block": block_wrap})["img"]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_extras\nodes_sparse_attention.py", line 318, in block_patch
    return extra["original_block"](args)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 763, in block_wrap
    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                   ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         transformer_options=args["transformer_options"], attention=args.get("attention"))}
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 298, in forward
    return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
                                  ~~~~~~~~^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 211, in forward
    return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")
                                                ~~~~~~~~^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ops.py", line 1527, in forward
    output = self.forward_comfy_cast_weights(
        input,
    ...<2 lines>...
        weight_only_quant=weight_only_quant,
    )
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ops.py", line 1471, in forward_comfy_cast_weights
    return self._forward(input, weight, bias)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ops.py", line 1439, in _forward
    return torch.nn.functional.linear(input, weight, bias)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\comfy_kitchen\tensor\base.py", line 362, in __torch_dispatch__
    return op_handlers[parent_cls](qt, args, kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\comfy_kitchen\tensor\int8.py", line 287, in _handle_int8_linear_tensorwise
    return torch.ops.comfy_kitchen.int8_linear(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        input_tensor.contiguous(),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        convrot_groupsize,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\_ops.py", line 1275, in __call__
    return self._op(*args, **kwargs)
           ~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\_library\custom_ops.py", line 375, in backend_impl
    result = self._backend_fns[device_type](*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\_compile.py", line 54, in inner
    return disable_fn(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\_dynamo\eval_frame.py", line 1298, in _fn
    return fn(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\_library\custom_ops.py", line 410, in wrapped_fn
    return fn(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\comfy_kitchen\backends\eager\quantization.py", line 1261, in _op_int8_linear
    return impl(**kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\comfy_kitchen\backends\cuda\__init__.py", line 2190, in int8_linear
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
torch.OutOfMemoryError: Allocation on device 0 would exceed allowed memory. (out of memory)
Currently allocated     : 3.77 GiB
Requested               : 6.28 GiB
Device limit            : 23.99 GiB
Free (according to CUDA): 389.34 MiB
PyTorch limit (set by user-supplied memory fraction)
                        : 17179869184.00 GiB

[INFO] Memory summary:
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |    783 MiB |   5126 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Active memory         |    783 MiB |   5126 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Requested memory      |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| GPU reserved memory   |    896 MiB |   6592 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Non-releasable memory |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Allocations           |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Active allocs         |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| GPU reserved segments |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|

[ERROR] Got an OOM, unloading all loaded models.
[INFO] Prompt executed in 00:25:59


> **Comment by tritant (2026-10-09T10:43:06Z):**

Thanks for the detailed testing and logs!

Your latest results are particularly interesting. The previous OOM occurred inside the latent upscaler, but this time the crash happens during **H3 Refine sampling**, specifically in `comfy_kitchen.int8_linear`, after the upscale has completed.

I've been investigating similar reports, and I found two potentially related issues:

- [[ComfyUI #16175](https://github.com/Comfy-Org/ComfyUI/issues/16175)](https://github.com/Comfy-Org/ComfyUI/issues/16175) — An OOM with MiniMax H3 INT8 ConvRot and the official Model Sparse Attention node, also on an RTX 4090. Interestingly, the failed allocation was approximately 6.24 GiB, compared to 6.28 GiB in your log.
- [[comfy-kitchen #196](https://github.com/Comfy-Org/comfy-kitchen/issues/196)](https://github.com/Comfy-Org/comfy-kitchen/issues/196) — Reports stalls and memory problems with INT8 ConvRot during successive model executions.

I also noticed that your stack trace passes through `comfy_extras/nodes_sparse_attention.py`, so I'd like to eliminate that possibility before making further changes to the Extender's memory management.

**Could you try one Full Batch run with the official Model Sparse Attention node disabled?**

Ideally, keep everything else identical: resolution, Refine scale, clip duration, INT8 ConvRot model, and all other settings.

This would help us determine whether Sparse Attention is contributing to the memory pressure or whether we need to investigate model unloading and memory retention between clips more deeply.

I agree that the difference between Clip by Clip and Full Batch strongly suggests something related to memory management across successive clips. However, the current logs don't yet prove an actual memory leak in the Extender.

I'd rather identify the root cause than introduce aggressive CUDA cleanup that might interfere with ComfyUI's own memory management.

Thanks again for your patience and all the testing. It's extremely helpful!


> **Comment by liberateme77 (2026-10-09T10:57:25Z):**

**This run i did before your response. So i will now disable Model Sparse Attention and give you result in next post ;-)** 

i did some more tests. Basically first clip the full batch processes is as good on memory s clip by clip, then something happens between clips (more memory usage). I will get no video encode spike (as before in bad run - consistent just observation), nothing happens for long time (slow in system ram) or right out OOM. Never happens on first clip in full batch mode.

if failed on clip 8 (refva_0007.mp4), then i executed it again and it finished the same clip without problem. (cancel and rerun, i hit interupt because i wanted cached file with audio).

this second log is direct continuation of first log. 

[ERROR] Got an OOM, unloading all loaded models.
[INFO] Prompt executed in 00:25:59
[INFO] got prompt
[INFO] H3 Full Batch Ref2VA start: validated=[True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] cached_segments=6
[INFO] H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=1 validated=True stop=False clips=6 frames=1025 cache=467.2 MB
[INFO] H3 validated hard-skip: Ref2VA clip 2 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=2 validated=True stop=False clips=6 frames=1025 cache=467.2 MB
[INFO] H3 validated hard-skip: Ref2VA clip 3 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=3 validated=True stop=False clips=6 frames=1025 cache=467.2 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 4 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=4 validated=False stop=False clips=6 frames=1025 cache=467.2 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 5 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=5 validated=False stop=False clips=6 frames=1025 cache=467.2 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 6 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=6 validated=False stop=False clips=6 frames=1025 cache=467.2 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] Requested to load MiniMaxH3
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [02:23<00:00, 17.88s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [02:21<00:00, 35.35s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=7 validated=False stop=False clips=7 frames=1178 cache=502.4 MB
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [04:03<00:00, 30.43s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
  0%|          | 0/4 [00:00<?, ?it/s,   Model Initializing ...  ]
[INFO] Comfy model compiler graph breaks: 18, rogues: 15
[ERROR] !!! Exception during processing !!! Allocation on device 0 would exceed allowed memory. (out of memory)
Currently allocated     : 15.73 GiB
Requested               : 1.69 GiB
Device limit            : 23.99 GiB
Free (according to CUDA): 2.63 GiB
PyTorch limit (set by user-supplied memory fraction)
                        : 17179869184.00 GiB
[ERROR] Traceback (most recent call last):
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 548, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 353, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 327, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 315, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 4769, in wrapped
    result = function(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 6526, in extend
    sampled, refine_trim, _refine_w, _refine_h = _refine_ref2va_sample(
                                                 ~~~~~~~~~~~~~~~~~~~~~^
        owner=owner, clip_index=i, clip_count=len(clips), sampled=first_pass,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<9 lines>...
        context_length=context_length, audio_context_length=audio_context_length,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 2687, in _refine_ref2va_sample
    refined = _sample_h3(
        clip_model, refine_positive, up_latent, seed, sampler_name, scheduler,
        int(refine_steps), float(refine_denoise), sigmas=refine_sigmas,
    )
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 2589, in _sample_h3
    samples = guider.sample(
        noise,
    ...<6 lines>...
        seed=int(seed),
    )
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\k_diffusion\sampling.py", line 205, in sample_euler
    denoised = model(x, sigma_hat * s_in, **extra_args)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_base.py", line 210, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_base.py", line 254, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 621, in forward
    graph_out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload,
              denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 765, in _forward
    h = blocks_replace[("double_block", i)](
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         "layout": layout, "transformer_options": transformer_options},
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        {"original_block": block_wrap})["img"]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_extras\nodes_sparse_attention.py", line 318, in block_patch
    return extra["original_block"](args)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 763, in block_wrap
    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                   ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         transformer_options=args["transformer_options"], attention=args.get("attention"))}
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 298, in forward
    return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
                                  ~~~~~~~~^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 211, in forward
    return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")
           ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ops.py", line 1033, in linear_input_act
    return linear_input_act_(x, weight, bias, input_act, act_weight, act_eps,
                             residual, residual_scale, fp16_accumulation=not quantized)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ops.py", line 1071, in linear_input_act_
    return quant_ops.ck.int8_linear(
           ~~~~~~~~~~~~~~~~~~~~~~~~^
        x, qdata, scale, bias, x.dtype,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        residual_scale=residual_scale,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\comfy_kitchen\__init__.py", line 1054, in int8_linear
    return impl(**kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\comfy_kitchen\backends\cuda\__init__.py", line 2190, in int8_linear
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
torch.OutOfMemoryError: Allocation on device 0 would exceed allowed memory. (out of memory)
Currently allocated     : 15.73 GiB
Requested               : 1.69 GiB
Device limit            : 23.99 GiB
Free (according to CUDA): 2.63 GiB
PyTorch limit (set by user-supplied memory fraction)
                        : 17179869184.00 GiB

[INFO] Memory summary:
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |    922 MiB |   5126 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Active memory         |    922 MiB |   5126 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Requested memory      |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| GPU reserved memory   |   1056 MiB |   6592 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Non-releasable memory |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Allocations           |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Active allocs         |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| GPU reserved segments |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|

[ERROR] Got an OOM, unloading all loaded models.
[INFO] Prompt executed in 00:13:43
[INFO] got prompt
[INFO] H3 Full Batch Ref2VA start: validated=[True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] cached_segments=7
[INFO] H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=1 validated=True stop=False clips=7 frames=1178 cache=526.9 MB
[INFO] H3 validated hard-skip: Ref2VA clip 2 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=2 validated=True stop=False clips=7 frames=1178 cache=526.9 MB
[INFO] H3 validated hard-skip: Ref2VA clip 3 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=3 validated=True stop=False clips=7 frames=1178 cache=526.9 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 4 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=4 validated=False stop=False clips=7 frames=1178 cache=526.9 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 5 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=5 validated=False stop=False clips=7 frames=1178 cache=526.9 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 6 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=6 validated=False stop=False clips=7 frames=1178 cache=526.9 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 7 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=7 validated=False stop=False clips=7 frames=1178 cache=526.9 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] Requested to load MiniMaxH3
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [03:56<00:00, 29.57s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] 1 models unloaded.
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [04:03<00:00, 60.82s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=8 validated=False stop=False clips=8 frames=1416 cache=580.8 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Interrupting prompt a7ec03cb-7502-43f2-ad37-64b8646d6706
[INFO] Comfy model compiler graph breaks: 11, rogues: 0
[INFO] Processing interrupted
[INFO] Prompt executed in 00:14:15
[INFO] got prompt
[INFO] H3 Full Batch Ref2VA start: validated=[True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] cached_segments=8
[INFO] H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=1 validated=True stop=False clips=8 frames=1416 cache=580.8 MB
[INFO] H3 validated hard-skip: Ref2VA clip 2 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=2 validated=True stop=False clips=8 frames=1416 cache=580.8 MB
[INFO] H3 validated hard-skip: Ref2VA clip 3 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=3 validated=True stop=False clips=8 frames=1416 cache=580.8 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 4 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=4 validated=False stop=False clips=8 frames=1416 cache=580.8 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 5 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=5 validated=False stop=False clips=8 frames=1416 cache=580.8 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 6 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=6 validated=False stop=False clips=8 frames=1416 cache=580.8 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 7 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=7 validated=False stop=False clips=8 frames=1416 cache=580.8 MB
[INFO] H3 COMPUTED hard-reuse: Ref2VA clip 8 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=8 validated=False stop=False clips=8 frames=1416 cache=580.8 MB
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] H3 incremental Full Decode: clips=8 frames=1416 interrupted=True video=exact_segment_stream_copy output=E:\Comfy-Desktop\ComfyUI-Shared\output\MiniMax_H3_cached_00017.mp4
[INFO] Prompt executed in 117.63 seconds




> **Comment by liberateme77 (2026-10-09T11:30:45Z):**

As requested -> Could you try one Full Batch run with the official Model Sparse Attention node disabled?

Ideally, keep everything else identical: resolution, Refine scale, clip duration, INT8 ConvRot model, and all other settings.

the only thing that changed was the sparse attention was disabled for this run.
results below:

[INFO] got prompt
[INFO] H3 Full Batch Ref2VA start: validated=[True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] cached_segments=7
[INFO] H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=1 validated=True stop=False clips=7 frames=1178 cache=621.1 MB
[INFO] H3 validated hard-skip: Ref2VA clip 2 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=2 validated=True stop=False clips=7 frames=1178 cache=621.1 MB
[INFO] H3 validated hard-skip: Ref2VA clip 3 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=3 validated=True stop=False clips=7 frames=1178 cache=621.1 MB
[INFO] H3 validated hard-skip: Ref2VA clip 4 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=4 validated=True stop=False clips=7 frames=1178 cache=621.1 MB
[INFO] H3 validated hard-skip: Ref2VA clip 5 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=5 validated=True stop=False clips=7 frames=1178 cache=621.1 MB
[INFO] H3 validated hard-skip: Ref2VA clip 6 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=6 validated=True stop=False clips=7 frames=1178 cache=621.1 MB
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [02:57<00:00, 22.16s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [03:46<00:00, 56.71s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=7 validated=False stop=False clips=7 frames=1178 cache=450.1 MB
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [05:10<00:00, 38.81s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
 25%|██▌       | 1/4 [01:58<05:56, 118.96s/it,  Model Initialization complete!  ]
[INFO] Comfy model compiler graph breaks: 9, rogues: 15
[ERROR] !!! Exception during processing !!! Allocation on device 0 would exceed allowed memory. (out of memory)
Currently allocated     : 15.88 GiB
Requested               : 1.69 GiB
Device limit            : 23.99 GiB
Free (according to CUDA): 286.55 MiB
PyTorch limit (set by user-supplied memory fraction)
                        : 17179869184.00 GiB
[ERROR] Traceback (most recent call last):
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 548, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 353, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 327, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 315, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 4769, in wrapped
    result = function(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 6526, in extend
    sampled, refine_trim, _refine_w, _refine_h = _refine_ref2va_sample(
                                                 ~~~~~~~~~~~~~~~~~~~~~^
        owner=owner, clip_index=i, clip_count=len(clips), sampled=first_pass,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<9 lines>...
        context_length=context_length, audio_context_length=audio_context_length,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 2687, in _refine_ref2va_sample
    refined = _sample_h3(
        clip_model, refine_positive, up_latent, seed, sampler_name, scheduler,
        int(refine_steps), float(refine_denoise), sigmas=refine_sigmas,
    )
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 2589, in _sample_h3
    samples = guider.sample(
        noise,
    ...<6 lines>...
        seed=int(seed),
    )
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\k_diffusion\sampling.py", line 205, in sample_euler
    denoised = model(x, sigma_hat * s_in, **extra_args)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_base.py", line 210, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_base.py", line 254, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 621, in forward
    graph_out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload,
              denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 770, in _forward
    h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 298, in forward
    return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
                                  ~~~~~~~~^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 211, in forward
    return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")
           ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ops.py", line 1033, in linear_input_act
    return linear_input_act_(x, weight, bias, input_act, act_weight, act_eps,
                             residual, residual_scale, fp16_accumulation=not quantized)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ops.py", line 1071, in linear_input_act_
    return quant_ops.ck.int8_linear(
           ~~~~~~~~~~~~~~~~~~~~~~~~^
        x, qdata, scale, bias, x.dtype,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        residual_scale=residual_scale,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\comfy_kitchen\__init__.py", line 1054, in int8_linear
    return impl(**kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\comfy_kitchen\backends\cuda\__init__.py", line 2190, in int8_linear
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
torch.OutOfMemoryError: Allocation on device 0 would exceed allowed memory. (out of memory)
Currently allocated     : 15.88 GiB
Requested               : 1.69 GiB
Device limit            : 23.99 GiB
Free (according to CUDA): 286.55 MiB
PyTorch limit (set by user-supplied memory fraction)
                        : 17179869184.00 GiB

[INFO] Memory summary:
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |   1083 MiB |   5126 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Active memory         |   1083 MiB |   5126 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Requested memory      |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| GPU reserved memory   |   1216 MiB |   6592 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Non-releasable memory |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Allocations           |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Active allocs         |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| GPU reserved segments |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|

[ERROR] Got an OOM, unloading all loaded models.
[INFO] Prompt executed in 00:18:46




> **Comment by tritant (2026-10-09T12:27:00Z):**

Thanks for testing without Model Sparse Attention!

Since the OOM still occurs during H3 Refine sampling, we can rule out Sparse Attention as the sole cause.

I've prepared a new **experimental ** with an additional memory management improvement.

Before starting H3 Refine sampling, the Extender now asks ComfyUI to selectively unload models that are no longer needed at that stage, particularly the VideoVAE, AudioVAE, and text encoder.

The H3 model itself is preserved, along with its normal DynamicVRAM management. There are no changes to the sampling, Refine calculations, Motion Context, or Full Batch logic.

The goal is to free additional VRAM before H3 Refine begins, rather than waiting for an allocation failure.

**Experimental branch:**  
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/tree/experimental/refine-memory

Could you test this version with the same Full Batch settings that previously caused the OOM?

I can't guarantee this will solve the problem, but it's a targeted improvement worth testing before considering more substantial changes.


Thanks again for all your testing and detailed feedback!


> **Comment by liberateme77 (2026-10-09T12:31:51Z):**

Sure i will test it now.


> **Comment by liberateme77 (2026-10-09T13:29:45Z):**

**It went better this time, but we still ended up OOM.**
**log from v3.0.11 experimental refine memory branch**

[INFO] got prompt
[INFO] Model storage policy: fast_disk=True paths=['E:\\Comfy-Desktop\\ComfyUI-Shared\\models\\vae\\minimax_h3_audio_vae_fp32.safetensors']
[INFO] VAE load device: cuda:0, offload device: cpu, dtype: torch.float32
[INFO] Model storage policy: fast_disk=True paths=['E:\\Comfy-Desktop\\ComfyUI-Shared\\models\\vae\\minimax_h3_video_vae_fp16.safetensors']
[INFO] VAE load device: cuda:0, offload device: cpu, dtype: torch.float16
[INFO] Found quantization metadata version 1
[INFO] Using MixedPrecisionOps for text encoder
[INFO] Model storage policy: fast_disk=True paths=['E:\\Comfy-Desktop\\ComfyUI-Shared\\models\\text_encoders\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors']
[INFO] CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
[INFO] Found quantization metadata version 1
[INFO] Detected mixed precision quantization
[INFO] Using mixed precision operations
[INFO] Native ops: float8_e5m2, asym_w4a8_int8, w6a8_int8, int8_tensorwise, float8_e4m3fn, convrot_w4a4 , emulated ops: mxfp8, nvfp4
[INFO] model weight dtype torch.bfloat16, manual cast: torch.bfloat16
[INFO] model_type FLOW_AV
[INFO] Model storage policy: fast_disk=True paths=['E:\\Comfy-Desktop\\ComfyUI-Shared\\models\\diffusion_models\\Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors']
[INFO] H3 Full Batch Ref2VA start: validated=[True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] cached_segments=10
[INFO] H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=1 validated=True stop=False clips=10 frames=1773 cache=978.6 MB
[INFO] H3 validated hard-skip: Ref2VA clip 2 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=2 validated=True stop=False clips=10 frames=1773 cache=978.6 MB
[INFO] H3 validated hard-skip: Ref2VA clip 3 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=3 validated=True stop=False clips=10 frames=1773 cache=978.6 MB
[INFO] H3 validated hard-skip: Ref2VA clip 4 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=4 validated=True stop=False clips=10 frames=1773 cache=978.6 MB
[INFO] H3 validated hard-skip: Ref2VA clip 5 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=5 validated=True stop=False clips=10 frames=1773 cache=978.6 MB
[INFO] H3 validated hard-skip: Ref2VA clip 6 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=6 validated=True stop=False clips=10 frames=1773 cache=978.6 MB
[INFO] H3 validated hard-skip: Ref2VA clip 7 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=7 validated=True stop=False clips=10 frames=1773 cache=978.6 MB
[INFO] H3 validated hard-skip: Ref2VA clip 8 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=8 validated=True stop=False clips=10 frames=1773 cache=978.6 MB
[INFO] H3 validated hard-skip: Ref2VA clip 9 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=9 validated=True stop=False clips=10 frames=1773 cache=978.6 MB
[INFO] H3 validated hard-skip: Ref2VA clip 10 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=10 validated=True stop=False clips=10 frames=1773 cache=978.6 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: reasserted Extender payload patch over comfy.model_base.extra_conds
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] Requested to load MiniMaxH3
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
 12%|█▎        | 1/8 [01:13<08:37, 73.98s/it,  Model Initialization complete!  ][INFO] FETCH ComfyRegistry Data [DONE]
[INFO] [ComfyUI-Manager] default cache updated: https://api.comfy.org/nodes
FETCH DATA from: E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\__manager\cache\1514988643_custom-node-list.json [DONE]
[INFO] [ComfyUI-Manager] All startup tasks have been completed.
100%|██████████| 8/8 [03:24<00:00, 25.53s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] 1 models unloaded.
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [04:41<00:00, 70.45s/it]                                    
[INFO] H3 Disk Join: mode=full_batch clip=11 validated=False stop=False clips=11 frames=1943 cache=1017.5 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] H3 Final Decode: H.264 NVENC available; using hardware encoder by default.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [02:57<00:00, 22.14s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [03:50<00:00, 57.68s/it]                                    
[INFO] H3 Disk Join: mode=full_batch clip=12 validated=False stop=False clips=12 frames=2096 cache=1081.7 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [04:41<00:00, 35.24s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [06:34<00:00, 98.56s/it]                                    
[INFO] H3 Disk Join: mode=full_batch clip=13 validated=False stop=False clips=13 frames=2317 cache=1158.0 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] Comfy model compiler graph breaks: 15, rogues: 0
[ERROR] !!! Exception during processing !!! Allocation on device 0 would exceed allowed memory. (out of memory)
Currently allocated     : 3.48 GiB
Requested               : 867.68 MiB
Device limit            : 23.99 GiB
Free (according to CUDA): 0 bytes
PyTorch limit (set by user-supplied memory fraction)
                        : 17179869184.00 GiB
[ERROR] Traceback (most recent call last):
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 548, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 353, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 327, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 315, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 4825, in wrapped
    result = function(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 6493, in extend
    positive, latent = _make_ref2va_conditioning(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~^
        clip_text_encoder,
        ^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        audio_native_offset=audio_native_offset,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 2483, in _make_ref2va_conditioning
    cond = clip.encode_from_tokens_scheduled(tokens)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\sd.py", line 346, in encode_from_tokens_scheduled
    pooled_dict = self.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=True)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\sd.py", line 417, in encode_from_tokens
    o = self.cond_stage_model.encode_token_weights(tokens)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\sd1_clip.py", line 743, in encode_token_weights
    out = getattr(self, self.clip).encode_token_weights(token_weight_pairs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\text_encoders\minimax.py", line 114, in encode_token_weights
    out = super().encode_token_weights(token_weight_pairs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\sd1_clip.py", line 45, in encode_token_weights
    o = self.encode(to_encode)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\sd1_clip.py", line 306, in encode
    return self(tokens)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\sd1_clip.py", line 279, in forward
    outputs = self.transformer(None, attention_mask_model, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=self.layer_norm_hidden_state, dtype=torch.float32, embeds_info=embeds_info)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\text_encoders\minimax.py", line 100, in forward
    return super().forward(input_ids, attention_mask=attention_mask, embeds=embeds,
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                           num_tokens=num_tokens, intermediate_output=intermediate_output,
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                           final_layer_norm_intermediate=final_layer_norm_intermediate,
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                           dtype=dtype, embeds_info=embeds_info, **kwargs)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\text_encoders\qwen3vl.py", line 100, in forward
    return self.model(
           ~~~~~~~~~~^
        input_ids,
        ^^^^^^^^^^
    ...<10 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\text_encoders\llama.py", line 1033, in forward
    comfy.model_prefetch.prefetch_queue_pop(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        prefetch_queue, x.device, layer, x.dtype, core=core, enable_graph=enable_graph,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        malloc_scope="block"
        ^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_prefetch.py", line 184, in prefetch_queue_pop
    core()
    ~~~~^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\text_encoders\llama.py", line 1019, in core
    output, current_kv = layer(
                         ~~~~~^
        x=x,
        ^^^^
    ...<3 lines>...
        past_key_value=past_kv,
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\text_encoders\llama.py", line 794, in forward
    x = self.mlp(x)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\text_encoders\llama.py", line 760, in forward
    return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~
torch.OutOfMemoryError: Allocation on device 0 would exceed allowed memory. (out of memory)
Currently allocated     : 3.48 GiB
Requested               : 867.68 MiB
Device limit            : 23.99 GiB
Free (according to CUDA): 0 bytes
PyTorch limit (set by user-supplied memory fraction)
                        : 17179869184.00 GiB

[INFO] Memory summary:
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |   1827 MiB |   4438 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Active memory         |   1827 MiB |   4438 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Requested memory      |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| GPU reserved memory   |   2304 MiB |   6272 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Non-releasable memory |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Allocations           |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Active allocs         |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| GPU reserved segments |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|

[ERROR] Got an OOM, unloading all loaded models.
[INFO] Prompt executed in 00:45:03


> **Comment by tritant (2026-10-09T13:54:25Z):**

Thanks for the detailed logs and for testing v3.0.11!

The results are encouraging. You managed to generate three additional clips before the OOM, and the pre-Refine model offloading is working as intended.

Interestingly, the new OOM occurs during **Qwen3VL text encoding**, rather than H3 Refine. The traceback goes through `comfy.model_prefetch.prefetch_queue_pop()`, which points us toward a different potential cause.

I've found a very similar issue in MiniMax H3 Director, addressed by [[PR #207](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/207)](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/207). It concerns ComfyUI's prefetch resources not being cleaned up between segments generated within a single node execution.

This could explain why **Clip by Clip works while Full Batch progressively slows down and eventually runs out of VRAM**.

I've now prepared **experimental v3.0.12**, which retains the v3.0.11 improvements and adds three ComfyUI cleanup operations between Full Batch clips:

- `cleanup_prefetch_queues()`
- `reset_cast_buffers()`
- `vbars_reset_watermark_limits()`

These are the same cleanup operations ComfyUI normally performs at the end of a node execution.


Could you update the experimental branch and test v3.0.12 with the same model and Full Batch settings as your latest run?

The important things to check are whether the sampling speed remains consistent across clips and whether the OOM is finally resolved.

I can't guarantee the memory issue is completely fixed yet, but this looks like a much more targeted solution than changing the sampling or introducing automatic validation.

Thanks again for your patience and all the testing!


> **Comment by liberateme77 (2026-10-09T14:04:33Z):**

Sure, I'll test it right away. You've already done a great job—it's running much better. However, each subsequent test will probably take a bit more time :-). I'm getting right on it, though. I'm not changing any settings; I'll just verify I have the right version and run it. I'll post the results (logs) here once I have them. Thanks a lot.


> **Comment by tritant (2026-10-09T14:24:20Z):**

No rush at all! Take your time, and thanks again for all your testing and feedback.

I think v3.0.12 is probably our last real chance to solve this memory issue. We've already addressed model offloading, and now we're cleaning up ComfyUI's memory management resources between Full Batch clips.

If this still doesn't solve the OOM, I'm honestly not sure what else we could improve on the Extender side without making major changes.

Hopefully, this will finally do the trick! Fingers crossed! :)


> **Comment by liberateme77 (2026-10-09T15:57:50Z):**

**I love version v3.0.12!** 
I’ve already generated 9 clips, and the 10th is running. VRAM allocation is stable, not climbing (23,0/24,0 GB) Momentary peak of 23,1 but it drops to 23,0 each time. 
I don't want to jinx it, but I wanted to let you know. This full-batch version is just as good as the clip-by-clip method.

I don't see any issues. It looks like the problem is resolved. I'm continuing with the generation, but I don't think it will crash.

**Thank you very much Tritant and TheDarkKnightBG as well. I am keeping this version ;-)** 

I've attached the log in case you are interested. 


[INFO] got prompt
[INFO] Model storage policy: fast_disk=True paths=['E:\\Comfy-Desktop\\ComfyUI-Shared\\models\\vae\\minimax_h3_audio_vae_fp32.safetensors']
[INFO] VAE load device: cuda:0, offload device: cpu, dtype: torch.float32
[INFO] Model storage policy: fast_disk=True paths=['E:\\Comfy-Desktop\\ComfyUI-Shared\\models\\vae\\minimax_h3_video_vae_fp16.safetensors']
[INFO] VAE load device: cuda:0, offload device: cpu, dtype: torch.float16
[INFO] Found quantization metadata version 1
[INFO] Using MixedPrecisionOps for text encoder
[INFO] Model storage policy: fast_disk=True paths=['E:\\Comfy-Desktop\\ComfyUI-Shared\\models\\text_encoders\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors']
[INFO] CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
[INFO] Found quantization metadata version 1
[INFO] Detected mixed precision quantization
[INFO] Using mixed precision operations
[INFO] Native ops: float8_e5m2, convrot_w4a4, asym_w4a8_int8, w6a8_int8, float8_e4m3fn, int8_tensorwise , emulated ops: mxfp8, nvfp4
[INFO] model weight dtype torch.bfloat16, manual cast: torch.bfloat16
[INFO] model_type FLOW_AV
[INFO] Model storage policy: fast_disk=True paths=['E:\\Comfy-Desktop\\ComfyUI-Shared\\models\\diffusion_models\\Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors']
[INFO] H3 Full Batch Ref2VA start: validated=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] cached_segments=16
[INFO] H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=1 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 2 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=2 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 3 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=3 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 4 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=4 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 5 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=5 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 6 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=6 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 7 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=7 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 8 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=8 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 9 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=9 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 10 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=10 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 11 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=11 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 12 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=12 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 13 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=13 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 14 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=14 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 15 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=15 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] H3 validated hard-skip: Ref2VA clip 16 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=full_batch clip=16 validated=True stop=False clips=16 frames=2861 cache=1602.7 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: reasserted Extender payload patch over comfy.model_base.extra_conds
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] Requested to load MiniMaxH3
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
 38%|███▊      | 3/8 [01:15<02:00, 24.15s/it][INFO] FETCH ComfyRegistry Data [DONE]
[INFO] [ComfyUI-Manager] default cache updated: https://api.comfy.org/nodes
FETCH DATA from: E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\__manager\cache\1514988643_custom-node-list.json [DONE]
[INFO] [ComfyUI-Manager] All startup tasks have been completed.
100%|██████████| 8/8 [03:21<00:00, 25.22s/it]
[INFO] Requested to load LatentResizer3D
[INFO] 1 models unloaded.
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [04:25<00:00, 66.29s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=17 validated=False stop=False clips=17 frames=3031 cache=1641.6 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] H3 Final Decode: H.264 NVENC available; using hardware encoder by default.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Comfy model compiler graph breaks: 5, rogues: 0
[INFO] H3 Full Batch Ref2VA Motion ON: AIMDO boundary cleanup after clip 17
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [03:06<00:00, 23.36s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [03:55<00:00, 58.84s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=18 validated=False stop=False clips=18 frames=3184 cache=1704.2 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Comfy model compiler graph breaks: 5, rogues: 0
[INFO] H3 Full Batch Ref2VA Motion ON: AIMDO boundary cleanup after clip 18
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [02:57<00:00, 22.14s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [03:45<00:00, 56.38s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=19 validated=False stop=False clips=19 frames=3337 cache=1767.3 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Comfy model compiler graph breaks: 6, rogues: 0
[INFO] H3 Full Batch Ref2VA Motion ON: AIMDO boundary cleanup after clip 19
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [03:21<00:00, 25.13s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [04:24<00:00, 66.25s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=20 validated=False stop=False clips=20 frames=3507 cache=1833.0 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Comfy model compiler graph breaks: 5, rogues: 0
[INFO] H3 Full Batch Ref2VA Motion ON: AIMDO boundary cleanup after clip 20
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [02:59<00:00, 22.39s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [03:46<00:00, 56.72s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=21 validated=False stop=False clips=21 frames=3660 cache=1899.3 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Comfy model compiler graph breaks: 5, rogues: 0
[INFO] H3 Full Batch Ref2VA Motion ON: AIMDO boundary cleanup after clip 21
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [02:58<00:00, 22.32s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [03:48<00:00, 57.23s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=22 validated=False stop=False clips=22 frames=3813 cache=1961.7 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Comfy model compiler graph breaks: 5, rogues: 0
[INFO] H3 Full Batch Ref2VA Motion ON: AIMDO boundary cleanup after clip 22
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [04:39<00:00, 34.97s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [06:29<00:00, 97.35s/it]                                    
[INFO] H3 Disk Join: mode=full_batch clip=23 validated=False stop=False clips=23 frames=4034 cache=2037.0 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Comfy model compiler graph breaks: 5, rogues: 0
[INFO] H3 Full Batch Ref2VA Motion ON: AIMDO boundary cleanup after clip 23
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [03:58<00:00, 29.76s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3VideoVAE, MiniMaxH3AudioVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [05:15<00:00, 78.77s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=24 validated=False stop=False clips=24 frames=4221 cache=2116.0 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Comfy model compiler graph breaks: 5, rogues: 0
[INFO] H3 Full Batch Ref2VA Motion ON: AIMDO boundary cleanup after clip 24
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [03:28<00:00, 26.06s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3VideoVAE, MiniMaxH3AudioVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 4/4 [04:18<00:00, 64.71s/it]                                   
[INFO] H3 Disk Join: mode=full_batch clip=25 validated=False stop=False clips=25 frames=4391 cache=2185.2 MB
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Comfy model compiler graph breaks: 5, rogues: 0
[INFO] H3 Full Batch Ref2VA Motion ON: AIMDO boundary cleanup after clip 25
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|██████████| 8/8 [02:57<00:00, 22.22s/it]                                   
[INFO] Requested to load LatentResizer3D
[INFO] loaded completely;  658.57 MB loaded, full load: True
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
 25%|██▌       | 1/4 [01:02<03:08, 62.71s/it,  Model Initialization complete!  ]


> **Comment by tritant (2026-10-09T16:38:31Z):**

Bingo! I think we've finally cracked it! 🎉

This is excellent news! Nine additional clips generated in Full Batch with Refine enabled, VRAM remaining stable at around 23.0/24 GB, and no progressive slowdown. That's exactly what we were hoping to see!

It strongly suggests that the missing ComfyUI/AIMDO cleanup between clips was the main cause of the problem.

The solution turned out to be much simpler than expected: reproducing ComfyUI's three memory cleanup operations between Full Batch clips, inspired by [MiniMax H3 Director PR #207](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/207).

And the best part is that we didn't have to change the sampling, Motion Context, or introduce Auto Validate. Full Batch can finally behave like Clip by Clip in terms of memory management.

I'll keep v3.0.12 as the fix for this issue.

A huge thank you for your patience, detailed logs, and all those hours of testing! Your feedback was essential in tracking this down.

And thanks to the contributors behind PR #207 for pointing us in the right direction!

Enjoy your uninterrupted Full Batch generations! 😄


> **Comment by liberateme77 (2026-10-10T08:01:06Z):**

Update: I ran a six-hour overnight clip generation session without any problems. I generated around 40 clips.


> **Comment by tritant (2026-10-10T08:09:46Z):**

Great, I’ve merged the experimental branch into main.
Now I’m getting back to the current project. A little exclusive for you: did you notice the new output on the decode/preview?

<img width="2448" height="937" alt="Image" src="https://github.com/user-attachments/assets/c1e27076-24f0-4157-aa6f-f03434982e26" />

If you want to support the project, you can make a donation via my small free editing software.
https://minimediaedit.eu/


> **Comment by liberateme77 (2026-10-10T08:49:02Z):**

No, I hadn't noticed that, but it looks great in the image. I'll definitely give it a try once I finish my current project.  (74 clips)

I've only been working with video for three weeks, so everything is new to me. I am using capcut currently. That's why i wanted 1080p output. So i can cut it and upscale afterwards (24-60 fps interpolation).

By the way, I managed to change the number of steps and the denoise setting in "Refine," and it deleted all the generated clips. A warning message would be nice :-)

I used to think my graphics card couldn't handle 1080p in full batch mode, but it turned out the issue was actually memory handling between clips. 

Now I can run it overnight—or just walk away—and deal with the editing aftermath in CapCut later :-).

It's a good thing TheDarkKnightBG  opened a support ticket; comparing "clip-by-clip" versus "full batch" hadn't immediately occurred to me.

I'm thrilled to be able to generate Full HD now. Faces that were further away from the camera—and thus small in the frame—used to get really distorted, and even upscaling didn't save them. Now I just need a cheap new graphics card with HBM memory that's 10x faster :-D.


I think the best approach is to generate the native resolution based on the shorter edge—plus a slight correction on the longer edge to get closer to a 16:9 ratio (like 1376x768)—followed by a latent upscale to 1080p (which saves time and is perfect for editing), and finally an upscale to 4K (exactly 2x) with frame rate interpolation.

I'm really happy that I can run this overnight now. Thanks a lot for this workflow and the support.


> **Comment by tritant (2026-10-10T09:04:56Z):**

Yes, everyone has their own workflow; personally, I don't use the "refine" feature—I almost didn't implement it at all, so I never would have detected the problem. Being able to edit directly within ComfyUI strikes me as a good idea; the clips are automatically positioned on the timeline along with their audio. The video editor is currently under development and isn't available on GitHub yet.


> **Comment by TheDarkKnightBG (2026-10-12T18:16:20Z):**

"It's a good thing TheDarkKnightBG opened a support ticket; comparing "clip-by-clip" versus "full batch" hadn't immediately occurred to me."

Thank you for helping Tritant with testing! And yeah I'm glad I decided to ask VS just going "it is what it is".

"If you want to support the project, you can make a donation via my small free editing software.
https://minimediaedit.eu/" 

Just sent 10 euro your way, thank you again for making this workflow happen, and doing the optimization work!

I did some testing yesterday and today, and can confirm that batch generation is pretty stable! Even with opened brave and a video playing with VLC, it did 1MP + 1.5X without a hitch (30 videos done successfully in one batch). I closed everything beside comfy and it managed to push it to 1.8X, on the 5th video I opened brave and soon after that it fails so for an overnight run 1.8X should be fine (though i will test 1.9 and 2 because why the fuck not :D) . And all that compared to batch failing on 1MP + 1.2X on the older version (with nothing opened beside comfy) before you started tweaking it.

One thing I'm noticing is that SOME videos won't have perfect transition, it's not fucked movement but more like degradation in quality (fuzines) usually for a second and then it disappears but sometimes it doesn't fix itself perfectly and goes on to be fuzzy. I doubt this was caused by the latest update, more like i am stumbling in it now because it's very dependant on the prompt, the input video/images or both. I will keep testing and if i find enough correlating data showing why/what scenarios this is happening with i will then open another issue. 


> **Comment by tritant (2026-10-12T18:26:24Z):**

I've noticed those minor alignment issues before, but as you say, there are so many factors involved that it's impossible to pinpoint the exact cause; I think focusing on Turbo LoRAs is the best approach.
Thanks for the donation.
Have fun.


> **Comment by liberateme77 (2026-10-12T18:28:29Z):**

You have the denoise setting too low for the refiner. Look up the correct
settings for denoise and steps. It depends on your initial step count, so
be careful—it matters whether you're using a 4-step or 8-step Turbo LoRA,
or a base 20-step model. I use a denoise value of 0.42 with 4 refiner steps
and an 8-step Turbo LoRA, because I was getting that same softness issue
you're experiencing. However, you shouldn't set the denoise too high, or it
will start messing up faces and fingers (though you can fix that by
increasing the refiner steps—which, in turn, increases generation time).

po 12. 10. 2026 o 20:26 tritant ***@***.***> napísal(a):

> *tritant* left a comment (tritant/ComfyUI_MiniMax_H3_Extender#107)
> <https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/107#issuecomment-6130160868>
>
> I've noticed those minor alignment issues before, but as you say, there
> are so many factors involved that it's impossible to pinpoint the exact
> cause; I think focusing on Turbo LoRAs is the best approach.
> Thanks for the donation.
> Have fun.
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/107?email_source=notifications&email_token=AT5YANQRHOIKQJQ5GJE7V3L5TUO6RA5CNFSNUABFM5UWIORPF5TWS5BNNB2WEL2JONZXKZKDN5WW2ZLOOQXTMMJTGAYTMMBYGY4KM4TFMFZW63VHMNXW23LFNZ2KKZLWMVXHJLDGN5XXIZLSL5RWY2LDNM#issuecomment-6130160868>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AT5YANXEDD3JG3LLJLC23EL5TUO6RAVCNFSNUABGKJSXA33TNF2G64TZHMYTGMZUHAYTGMZZHA5US43TOVSTWNJXGUZTSMZWGM3DLILWAI>
> .
> Triage notifications, keep track of coding agent tasks and review pull
> requests on the go with GitHub Mobile for iOS
> <https://github.com/notifications/mobile/ios/AT5YANV6NE7YVK3KQ37RZI35TUO6RA5CNFSNUABFM5UWIORPF5TWS5BNNB2WEL2JONZXKZKDN5WW2ZLOOQXTMMJTGAYTMMBYGY4KM4TFMFZW63VHMNXW23LFNZ2KKZLWMVXHJKTGN5XXIZLSL5UW64Y>
> and Android
> <https://github.com/notifications/mobile/android/AT5YANRR6NWCIN5EZBWD5PL5TUO6RA5CNFSNUABFM5UWIORPF5TWS5BNNB2WEL2JONZXKZKDN5WW2ZLOOQXTMMJTGAYTMMBYGY4KM4TFMFZW63VHMNXW23LFNZ2KKZLWMVXHJLTGN5XXIZLSL5QW4ZDSN5UWI>.
> Download it today!
> You are receiving this because you commented.Message ID:
> ***@***.***>
>



> **Comment by TheDarkKnightBG (2026-10-12T18:42:47Z):**

I don't use turbo loras. Just the int8 and one NSFW lora set on 0.5 strength

Steps are 20

and on refinement steps are 4 and denoise on 0.3. 

I will play around with the denoise amount but yeah, there are many factors that may affect this.



> **Comment by tritant (2026-10-12T18:47:51Z):**

I think that without a Turbo LoRA, 4 steps is a bit low, even for a refinement pass.


> **Comment by TheDarkKnightBG (2026-10-12T19:17:15Z):**

Mkay, In that case I will see which prompt + images/videos combo with the current settings leads to the worse "transition effect" and then keep the prompt and images/videos and start pushing steps and denoise up in small steps to see the difference.


> **Comment by tritant (2026-10-13T04:42:21Z):**

By the way, since you work with high resolutions, you should give Veda a try—it's very efficient.
My latest node might also interest you; it upscales the final result to 4K at lightning speed.

https://github.com/tritant/ComfyUI_NVIDIA_RTX_VSR_Long_Video

<img width="1167" height="305" alt="Image" src="https://github.com/user-attachments/assets/a87ff58d-6554-4a50-94a2-32389e469eca" />



### #106 [closed] ISSUE: [Bug]  Latent refine (upscale) creates a visible seam between clips — sharpness step and color shift
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/106  by segatecm 2026-10-06T16:05:05Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

v3.0.6

### ComfyUI version

0.39

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Full Batch

### Describe the issue

## Environment
- ComfyUI 0.39.0, ComfyUI_MiniMax_H3_Extender v3.0.6
- 2 clips, ref2va + motion_context, full_batch
- 480x640 base → refine_scale 1.5 → 704x960, steps 6, refine_steps 4, refine_denoise 0.3
- Output: 226 frames @ 24fps, clip 1 = 124 frames, no seam shift applied

## Problem
With `refine_enabled=true`, clip 1 and clip 2 do not match at the seam (frame 124).
Two artifacts are clearly visible without any measurement:

1. **Sharpness drop** — clip 2 is noticeably softer than clip 1.
2. **Color temperature shift** — clip 2 is warmer/cooler than clip 1, with a
   visible jump right after the seam.

Setting `refine_enabled=false` removes both. So the issue is introduced by the
refine pass.

## Measurements (clip 1 tail frames 100-123 vs clip 2 head 124-147)

| Metric | clip 1 | clip 2 | Change |
|---|---|---|---|
| Sharpness (luma Laplacian var) | 28.95 | 17.88 | **-38.2%** |
| Color temperature (R/B, center crop) | 1.0779 | 1.0950 | +1.59% |
| Brightness (mean Y) | 56.11 | 53.36 | -4.9% |

The sharpness step at the seam is the largest frame-to-frame change in the whole
226-frame video (5.08 and 6.07 for frames 123→124 and 124→125); the next largest
natural motion change is only 2.50. Clip 2 never recovers.

## Likely causes

1. **No detail matching exists.** The only seam corrector,
   `_correct_current_segment` (motion_context_disk.py:1329), works purely on
   luma. It cannot correct a sharpness difference.

2. **The correction is luma-only.** Statistics use `_luma_map`
   (motion_context_ram.py:307), and the output delta is a scalar added to all
   three channels (motion_context_disk.py:1394-1398) — hue/saturation are never
   corrected. So chroma differences across the seam survive.

3. **The early-return test is luma-only** (motion_context_disk.py:1352), so
   pairs with matching brightness but mismatched color are skipped entirely.

4. **The two refine passes are asymmetric.** Clip 1's refine gets
   `previous_refined_proxy=None` (extender.py:6530) and therefore skips motion
   context (extender.py:2677); clip 2's refine is constrained. Each clip is
   refined by a separate unconstrained resample (extender.py:2668, :2686).

5. Nothing caps or matches high-frequency energy across the seam, so the
   upscaler's content-dependent output statistics (latent_upscaler.py:151) show
   up as a permanent step.

## Suggested direction

- Extend `_correct_current_segment` to match chroma (Cb/Cr mean/std), not just luma.
- Include chroma and high-frequency energy in the early-return test at
  motion_context_disk.py:1352.
- Add a detail/high-frequency energy match across the seam.
- Document the risk in the `refine_enabled` tooltip (extender.py:4941), which
  currently only mentions cache invalidation.

### Steps to reproduce

REF = "/path/to/reference.png"

SUBJ = ("subject_definitions:\n"
        "<Subject 1> is a woman. Her facial features, hairstyle, body proportions, "
        "clothing and overall appearance come entirely from <Picture 1>.\n\n")

PROMPTS = [
    SUBJ + "Scene: <Subject 1> stands by the seaside in bright sunlight, the sea breeze "
           "moves her hair, she smiles at the camera, camera slowly pushes in. "
           "Cinematic, natural light.",
    SUBJ + "Scene: <Subject 1> turns and walks toward the waves, the camera slowly pulls "
           "back to a wide shot, the sunset turns the sea golden. "
           "Cinematic, natural light.",
]

### Error message / console log

```shell

```

### Workflow / additional information

_No response_

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-10-06T18:15:57Z):**

Thanks for the detailed report and the Refine ON/OFF comparison.
I’ve now retested the current v3.0.6 with a 3-clip Ref2VA Motion ON sequence using Refine ON (10s / 5s / 12s). In that test I could not reproduce a clear loss of sharpness or a major seam issue between clips.
I also checked the actual visible transitions frame by frame. The joins remain visually coherent, and the sharpness measurements are not consistent enough to support a general “Refine ON causes softness” conclusion. Depending on the metric, the transition can even appear equally sharp or slightly sharper. This is important because simple Laplacian-variance measurements can vary strongly with framing, motion, depth of field, subject scale, etc.
I also confirmed that the current seam correction is luma-based, but since Refine OFF already gives good seams and I cannot reproduce the reported degradation reliably, I do not want to change the existing seam correction, add global chroma matching, or add sharpening. That could easily damage cases that already work correctly.
There is one Refine-specific implementation detail that I am still keeping an eye on: the previous refined latent is currently resized to provide Motion Context to the next base pass. It is not ideal conceptually, but at the moment I do not have evidence that it is the cause of a visible regression in normal use.
So for now I’m going to keep v3.0.6 unchanged in this area.

If possible, could you upload the saved .ext project somewhere and share a download link? That would be the most useful thing for me, because I could load exactly the same project, references, prompts, clip durations, seeds and Refine settings and try to reproduce the issue directly.

Thanks again for taking the time to document this so precisely.


> **Comment by gilly2020 (2026-11-16T23:08:20Z):**

[h3_refine_seam_fix.patch](https://github.com/user-attachments/files/34942069/h3_refine_seam_fix.patch)

using claude, we found this. 

basically the refiner passes latent tail from clip 1 is downsized to be used as the next clips low pass start, this introduces noise. i was having this problem and replicated the sharpeness degredation error that compounds over every clip. the fix was to cache the low pass latent tail from the previous clip alongside the refiner passes upscaled tail, then use the lowpass tail for the low pass of clip 2, and the high pass tail for the high pass of clip 2. 

if that makes sense. 

anyway, credit to claude. i've implemented the fixes that claude suggested and now i get no seam softening. 

next is to fix that tiny bit of brightness burst at the start of a new clip with refiner on. 

 


> **Comment by tritant (2026-11-17T02:55:14Z):**

Thanks for sharing the patch and your findings!

We actually discussed this approach when issue #106 was first opened. My main concern is that the Refine pass can modify details, including faces and textures. Using the unrefined latent tail for the next clip's low-resolution pass, while using the refined tail for its high-resolution pass, could introduce inconsistencies between the two stages.

That's why we decided to keep using the refined latent as the reference for both passes. We couldn't reproduce the progressive sharpness degradation in our tests, but it's interesting that you've managed to reproduce and fix it on your side.

Regarding the brightness burst at the beginning of new clips, **we already have an experimental version that significantly reduces the first-frame flash**. It's not completely eliminated yet, and this fix hasn't been pushed to GitHub. We're still testing it.



### #104 [closed] ISSUE: [Bug]
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/104  by tedtninety 2026-10-04T14:26:27Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

3.01

### ComfyUI version

_No response_

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Clip-by-Clip

### Describe the issue

No an issue

### Steps to reproduce

NO steps

### Error message / console log

```shell
nil
```

### Workflow / additional information

I didn't know where else to reach you to thank you for all your amazing work.
I found it a game changer.
Sad to see it's reached its end.
May I ask why and if you have something else you may be working on?
Thank you.
Regards.

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-10-04T14:48:38Z):**

Thanks for using the extender.
I’m delighted it was able to help you.
There’s no specific reason; I simply feel I’ve taken it as far as I can. Of course, I have other projects in mind—time will tell.

It has also become quite a large project to maintain, and every new feature now carries a much higher risk of breaking something that already works well.
So I prefer to stop adding major features rather than keep expanding it indefinitely and eventually make it less stable.
I will still fix important bugs or compatibility issues if they appear, but I do not currently plan any major new development for MiniMax H3 Extender.


> **Comment by tedtninety (2026-10-04T20:28:00Z):**

First of all, I want to apologize for using this medium to reach out to you. However, I wanted to express my deepest thanks to you—not just for your creation, but for who you are as a person and your integrity in the way you develop. It is a breath of fresh air to see such high capacity. I want to wish you the absolute best on whatever journey you decide to embark on. Thank you.



### #103 [closed] ISSUE: [Feature] Please document reference numbering logic
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/103  by honeyphoria 2026-10-02T20:40:05Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

The Extender node needs some human-written documentation.

### Problem or limitation

Hi, great nodes. I know it says you won't work on it anymore, but I'll try my luck anyways.

It feels impossible to me to untangle the intended behaviour and use of all the different ways that the extender node allows providing picture, video, and audio references.

I tried asking ChatGPT to explain it to me, and so it did, in 13 pages of text, graphs, and code:

https://github.com/user-attachments/assets/ccb1ce35-edaf-436c-b2ce-e734ec4a7d68

Which I tried reading faithfully to no avail. The only documentation that exists is release notes in the readme, seemingly bottom to top, of all these new features getting stacked on top of each other, possibly making each other redundant, or somehow working together?

### Proposed solution

The Extender node needs some human-written documentation.

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-10-02T20:56:34Z):**

Thanks for the feedback — I agree the reference system has grown over time and the README is now more of a history of features than a simple user guide.

The easiest way to think about references is this:

- **Global references** are available to every clip.
- **Local references** belong only to one clip.
- Pictures, Videos and Audio each have their own separate slot numbers.

For example:

- `<Picture 1>` is Picture slot 1
- `<Picture 2>` is Picture slot 2
- `<Video 1>` is Video slot 1

`<Picture 1>` and `<Video 1>` are completely different references; the numbering is separate for each type.

Global reference slots stay fixed. If Picture 2 is removed, Picture 3 does not suddenly become Picture 2. This is intentional so prompts do not change meaning unexpectedly.

Local references simply use the available slots for that clip, in addition to the global references.

A simple example:

Global refs:

- Picture 1 = Alice
- Picture 2 = Bob

Clip 1 has no local refs, so it sees:

- `<Picture 1>` = Alice
- `<Picture 2>` = Bob

Clip 2 adds one local Picture reference, for example a car. It then gets the next available Picture slot:

- `<Picture 1>` = Alice
- `<Picture 2>` = Bob
- `<Picture 3>` = car

That `<Picture 3>` exists only for Clip 2.

The same principle applies to local Video and Audio references: they are only added to the clip where you put them.

The important distinction is really just:

**Global = shared by all clips**  
**Local = only for this clip**

For images and videos, use the exact `<Picture N>` or `<Video N>` number shown by the Extender in the prompt.

Audio is slightly different: it is passed as an audio reference to H3 rather than something you normally need to name in the prompt.

A few practical rules:

- Use a **global reference** when you want the same character, place, video performance or voice reference available throughout the sequence.
- Use a **local reference** when it is only relevant to one specific clip.
- You can mix global and local references.
- Picture numbering is never silently compacted or reordered.
- The total limits are 9 Pictures, 3 Videos and 3 Audio references per clip.

I agree that this deserves a proper short “How references work” section instead of making users reconstruct it from release notes.


> **Comment by honeyphoria (2026-10-03T10:35:40Z):**

Thank you, you've answered a lot of my questions but maybe not quite all:
- I think the concept of local references is clear to me: Each clip has a "Refs" button, this is the one and only thing that is referred to as a "local reference", it's unambiguous. (Unless I'm wrong!)
- Global references however I see two different systems for: The sockets, and the UI buttons. I noticed when I plug an image node into a socket, that image actually gets fed into the UI button. But there are more UI slots than there are sockets. And what if I specify an image with the UI slot first, and then plug one in? Does it get overridden? Is that intended?
- `Audio is slightly different: it is passed as an audio reference to H3 rather than something you normally need to name in the prompt.` - I'm a bit surprised by this statement because I thought the MMH3 model is able to understand instructions for different uses of audio clips. Eg., one audio ref can be a reference for how a character sounds ("timbre", I've seen this called), while another audio ref could be a reference of what exact dialogue they might be saying. Do I have that wrong? :o 

Thank you for the response.


> **Comment by tritant (2026-10-03T11:12:36Z):**

Yes — your understanding of **local references** is correct.

The **Refs** button on each clip card manages references that belong only to that clip. That is what I mean by “local references”.

For **global Picture references**, the UI slots and the external IMAGE sockets are not two separate reference systems. They are two ways of filling the same global Picture slots.

The nine slots you see in the Extender UI are the actual global Picture slots:

- Ref 1 = `<Picture 1>`
- Ref 2 = `<Picture 2>`
- ...
- Ref 9 = `<Picture 9>`

The Reference Pack sockets are simply another way to feed images into those same slots from other ComfyUI nodes.

The sockets are dynamic, which may be why it looks like there are fewer of them: they grow as you connect them, up to the same 9 Picture slots.

For example, if you manually load an image into Ref 2 in the UI and then connect an external image to Ref 2 through the Reference Pack, the connected image will replace the content of global Ref 2 when the workflow runs. That is intentional — both refer to the same `<Picture 2>` slot.

It never shifts the other references around. Ref 3 always remains Ref 3, etc.

One potentially important detail: once an external image has been imported into that UI slot, disconnecting the socket does not restore whatever image was there before. The imported image has become the current content of that global slot, and you can then replace or remove it normally from the UI.

And yes, you are also right to question what I said about Audio. I oversimplified that part.

`<Audio N>` **is meaningful**, especially when several standalone Audio references are connected.

For example:

- `<Audio 1>` can select Audio reference 1
- `<Audio 2>` can select Audio reference 2
- several `<Audio N>` references can also be used in the same clip

So you can absolutely have, for example, one audio reference representing a voice/timbre and another containing speech or another audio example, and describe in the prompt how you want H3 to use them.

The Extender itself does not label an Audio slot as specifically “timbre”, “dialogue”, etc. It supplies the selected audio reference(s) to H3, and your prompt/reference content determines how H3 interprets them.

The reason I said the Audio tag is not always required is that when there is only **one** standalone Audio reference connected, the Extender automatically uses that reference. With multiple Audio references, using `<Audio 1>`, `<Audio 2>`, etc. is the proper way to explicitly select which one(s) a clip should use.

So the simple mental model is:

**Global Picture UI + Reference Pack sockets = the same Picture slots**  
**Clip Refs button = local references for that clip only**  
**`<Picture N>`, `<Video N>` and `<Audio N>` identify the corresponding reference in the prompt**


> **Comment by honeyphoria (2026-10-03T11:30:36Z):**

That does clear things up, thank you! (Even though it looks a bit AI-assisted, but I assume human-reviewed.)

Maybe one final question:

While exploring the codebase with Claude, it claimed that if an audio reference is longer than 5s, the node will somehow treat it differently compared to if it was shorter than 5s. I don't remember exactly what it said. Can you verify if this is a thing? Can you also speak as to a recommended audio duration for the two use cases?
1. For copying dialogue, of course the audio length should match the desired video length. No questions here, unless you have something to add.
2. But for voice timbre, I would want to be able to use a longer audio ref, to teach the model the desired voice as well as possible - presumably as wide a range of sounds made by the desired voice as possible, within as short a period of time as possible, just like when training a LoRA (few images, high variation). But can this voice timbre reference be much longer in duration than the output video? Say, a 30-60s audio clip, even if the target video duration is just 10s long?

Maybe I'll keep being annoying and asking questions, I'm not sure. But for now, that's all. Thank you once more!


> **Comment by tritant (2026-10-03T11:52:56Z):**

Claude did find a real thing there, although “longer than 5 seconds” is only part of the rule.

The 5-second value is **not an H3 quality threshold**. It is an Extender rule used to decide whether a **global audio reference** should behave like a reusable reference or like a continuous audio timeline.

For a global Audio reference:

- if it is short, it starts again from the beginning when used by another clip;
- if it is longer than both **5 seconds and the current clip duration**, the Extender assumes it may be a continuous source and advances through it across the clips that use it.

So for a 10-second clip, a 6-second reference is still just a normal reusable reference. A 30-second reference, on the other hand, is treated as a timeline: the first clip gets the beginning, the next clip continues farther into the source, etc.

**Local Audio references are deliberately different.** A local Audio reference always starts at the beginning and the Extender gives H3 the complete reference regardless of the generated clip duration, up to H3's native audio-reference limit.

That distinction is particularly relevant to your two examples.

For **dialogue/audio reuse**, matching the audio duration to the intended video span is indeed the simplest case. For a sequence of clips, a longer global Audio reference can also be useful because the Extender can progress through it across consecutive clips.

For **voice timbre**, I would recommend using a **local Audio reference** if you want a voice sample that is longer than the generated clip. For example, a 10-second video can perfectly well use a 15-second local voice reference; it will not be cut down to 10 seconds by the Extender.

However, 30–60 seconds cannot be given to H3 as one timbre reference. H3 accepts audio references from **2 to 15 seconds**, with **15 seconds maximum total audio-reference duration per generation**.

So if you have 60 seconds of good material from a voice, I would not feed the whole recording. I would select the best roughly **10–15 seconds**: clean speech, little or no background noise/music, and preferably enough variation in vowels, consonants, pitch and delivery to represent the voice well.

MiniMax's own H3 example explicitly uses one audio reference as reused audio and another separately as a **voice timbre reference**, so those are indeed intended to be distinct use cases.

In short:

**Dialogue / continuous audio:** global reference can make sense.  
**Voice timbre:** local reference, ideally a good 10–15 second sample.  
**30–60 second timbre sample:** choose the best 15 seconds rather than using the entire recording.



### #102 [closed] ISSUE: [Bug] The cache of previous clip doesn't use in the next clip generation. Even if the motion feature is enabled.
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/102  by volnodumcev 2026-10-01T12:22:00Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

2.9.0

### ComfyUI version

0.37.4

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Full Batch

### Describe the issue

The cache of previous clip doesn't use in the next clip generation. Even if the motion context feature is enabled.

### Steps to reproduce

1. Add 3 clips with 10, 5 and 12 secs of resulting duration
2. Add 3 reference photos
3. Use all 3 reference photos in different prompts of all 3 clips
4. Select 22 frames of context
5. Start generation
6. 3rd clip will start from one of reference photos, not from the last context of 2nd clip

### Error message / console log

```shell

```

### Workflow / additional information

Model: ref2va_int8_convrot
Adapter: h3_turbo_v4_step600

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-10-01T13:30:50Z):**

Thanks for the report.

In Ref2VA with Motion Context ON, it is technically not possible for Clip 3 to be sampled without the tail of Clip 2 being injected into its conditioning.

For every generated clip after the first one, the Extender requires the previous cached latent and applies Motion Context before sampling. If the previous latent is unavailable, generation stops with an error rather than silently continuing without context.

So if Clip 3 was actually generated in Motion ON, the end of Clip 2 was injected.

Could you please provide:

- the exact prompts used for Clips 1, 2 and 3
- which image references are used in each prompt
- whether those references are **global references** or **local references**
- ideally, which `<Picture N>` each clip refers to

The fact that Clip 3 visually resembles one of the reference images does not by itself mean Motion Context was missing, because image references and Motion Context are both present in the conditioning and can influence the result at the same time.


> **Comment by tritant (2026-10-01T14:29:13Z):**

There is no issue with the motion context; download this sample project, click "Load Project" in the Extender, load the file you just downloaded, and run it!

https://drive.google.com/file/d/1h1WfXCwFHD0qoMvgF1rtBnYCBOMKXeVD/view?usp=sharing

Your LoRA is very bad; try this one.
https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors?download=true


> **Comment by volnodumcev (2026-10-01T16:47:55Z):**

Your LoRA is very bad; 
How do you know this?


> **Comment by tritant (2026-10-01T16:56:08Z):**

Because I tried it—I tried them all.


> **Comment by volnodumcev (2026-10-01T17:13:26Z):**

And so what? In my experience this Lora is giving the best quality on 0.6MP with ref2va_int8_convrot


> **Comment by tritant (2026-10-01T17:17:48Z):**

We don't have the same experience, then.


> **Comment by volnodumcev (2026-10-01T17:18:53Z):**

It's not important in this case anyway. I'm just trying to delete all extender cache and try again.


> **Comment by tritant (2026-10-01T17:36:58Z):**

use the new project button

Yet it is important; the one that produces the best visual output isn't necessarily the best—you also have to evaluate prompt adherence and, above all, how non-destructive it is when handling references...


> **Comment by volnodumcev (2026-10-02T08:04:05Z):**

Yeap, the cause was the bad prompt. You can close the issue



### #101 [closed] ISSUE: [Bug] Cache files not being detected after stopping a partial "full-batch" run
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/101  by SyntaxError1871 2026-09-30T17:24:38Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

2.9.2

### ComfyUI version

0.33

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Both

### Describe the issue

Canceling a Full Batch workflow to switch over to clip-by-clip does not appear to work. I was running a long 12-part full-batch job overnight and had to terminate it early after clip 7 was rendered with the intention of switching to Clip to Clip mode for the remaining five clips, but the interface is not allowing me to select past Clip 3, claiming that the cache for Clip 3 does not exist, even though it does.

<img width="2934" height="144" alt="Image" src="https://github.com/user-attachments/assets/c1e1ac29-8db7-4eec-80ea-40ec0c88a367" />

<img width="701" height="83" alt="Image" src="https://github.com/user-attachments/assets/9598f28b-201c-4c19-bf7b-48df890ea2ba" />

But checking the cache folder in the directory shows all the cache files exist as expected, so the UI is refusing to allow me to just select "Validated" up to clip 8 to resume clip-by-clip generation.

I remember the ability to select "Validated" no matter the state was allowed until a recent update, so whatever check the plugin is using to check for cache may be bugged since the cache DOES exist, and I do not want to waste hours of generation time because of this bug and start from Clip 3 again when I have all the cache up to Clip 7.

<img width="1186" height="611" alt="Image" src="https://github.com/user-attachments/assets/474fb6a1-55ea-413c-a805-6bb67c8d9d0f" />

---

**Proposed solution:** restore the ability to just click "validated" without checks. 

A few major versions ago this function was available and was useful for queuing long chains of clips in clip-by-clip mode even without cache existing, so this additional check for cache seems to be causing more trouble than its worth. If cache doesn't exist, it should error out during generation rather than just preventing you from queueing up multiple clip-by-clip jobs, like for example queuing up Clip 1 -> Hit Run to queue Clip 1 -> Mark Clip 1 Validated -> Hit Run to queue Clip 2 -> Mark Clip 2 Validated -> Hit Run to queue Clip 3 and so on. Right now the plugin does not allow you to click "validated" unless the actual cache exists in the cache and I think that is an arbitrary limitation.

### Steps to reproduce

1. Run "Full Batch" job, cancel partly
2. Switch to "Clip to Clip" mode and attempt to select "validated" up to the latest generated clip to resume partway
3. H3 Extender plugin is not allowing me to select "validated" past clip 2, claiming Clip 3 cache does not exist when it does.

### Error message / console log

```shell

```

### Workflow / additional information

_No response_

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-09-30T19:09:54Z):**

Thanks for the report.

I’m currently unable to reproduce this behavior on my side.

From your description, I suspect the Full Batch may have been stopped using ComfyUI’s global Kill/Interrupt instead of the Extender’s own Interrupt button.

For Full Batch, please use the Interrupt button inside the Extender. It is designed to finish the current clip cleanly, save the resumable checkpoint, and keep the generated clip cache/manifest in a consistent state.

Using ComfyUI’s global Kill/Interrupt can stop execution at an arbitrary point and may leave files on disk that were not fully committed into the Extender’s cache manifest. In that situation, the files may physically exist, but the Extender cannot safely treat those clips as cached/validated.

I do not want to remove the cache validation safeguard, because allowing a clip to be marked as Validated without a confirmed physical cache can break the Motion Context chain.

<img width="692" height="75" alt="Image" src="https://github.com/user-attachments/assets/3906a7ad-e215-4a07-8e53-8e2306a7e666" />


> **Comment by SyntaxError1871 (2026-09-30T19:41:33Z):**

I would still like the option to ignore the cache validation safeguard. Didn't realize the "Interrupt" button was there for this purpose, so I suppose that's my bad, but I still liked the ability to queue individual clips that have not been generated into cache as a way to queue up long runs of jobs in the past and was sad to see that feature go away.


> **Comment by tritant (2026-09-30T19:52:18Z):**

Just to clarify one point: the behavior you describe was never an intended Extender feature.

Marking a clip as Validated has always meant that the clip is considered complete and backed by a valid physical cache. It was not designed as a way to queue future, not-yet-generated clips.

If an older version allowed a clip to remain or become Validated without a corresponding cache, that was an inconsistent state rather than a supported queueing workflow.

The current safeguard is intentional because, especially in Ref2VA Motion ON, allowing “Validated” clips without real caches can break the causal Motion Context chain and make the project state unreliable.

So I don’t plan to add an option to bypass cache validation.


> **Comment by tritant (2026-09-30T19:59:39Z):**

I think I understand what you mean now.

What you are describing is already possible in **Ref2VA Motion OFF**.

In Motion OFF, clips are independent, so you can leave specific clips unvalidated and generate/re-generate them individually without needing a contiguous cache chain.

The restriction is intentional in **Motion ON**, because clips are causally linked through Motion Context. In that mode, a later clip depends on the real cache of the previous clip, so allowing a clip to be marked as Validated without a physical cache would create an invalid chain.

So if your goal is to queue or work on individual clips independently, Motion OFF is the mode designed for that.


> **Comment by volnodumcev (2026-10-01T09:47:49Z):**

I have the same problem even with Motion ON


> **Comment by tritant (2026-10-01T09:59:21Z):**

Same answer, then.

For Full Batch, please use the Interrupt button inside the Extender. It is designed to finish the current clip cleanly, save the resumable checkpoint, and keep the generated clip cache/manifest in a consistent state.

Using ComfyUI’s global Kill/Interrupt can stop execution at an arbitrary point and may leave files on disk that were not fully committed into the Extender’s cache manifest. In that situation, the files may physically exist, but the Extender cannot safely treat those clips as cached/validated.



### #100 [closed] ISSUE: [Feature] Keep the accent of a speech ?
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/100  by jdorpitan-code 2026-09-29T10:30:31Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

Hello,
first of all : thank you for this great extension. To my opinion, it's much simplier than others "director" extensions.

I have a case where I have an audio/mp3 (a speech) of a person who speak french with a strong english accent.
When using this mp3, the output video gives a person that speak perfectly french (no accent).

When I use a standard minimax workflow, with the same prompt, and, of course the same assets, the output video gives a persons that speak french with a strong english accent.

Is there a way to ask this extension to keep the original flaw / bad french speech ?


### Problem or limitation

currently I have to use a standard minimax workflow, but I can't generate video of more than 15 seconds.

### Proposed solution

_No response_

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-09-29T11:40:34Z):**

Thanks for the report.

Could you give me a few more details about your test?

- How long is the reference audio file?
- How long is each generated clip?
- Are you generating only one clip, or several clips in the Extender?
- If several clips are generated, how many?

The Extender handles standalone reference audio a little differently from a normal MiniMax H3 workflow because it also has to support multi-clip generation.

In a standard H3 workflow, the reference audio can be provided as one complete reference to a single generation.

In the Extender, when the reference audio is longer than an individual generated clip, it can be treated as a timeline and split across the generated clips. For example, with a 15-second audio reference and three 5-second clips, the Extender may use approximately 0–5s for Clip 1, 5–10s for Clip 2, and 10–15s for Clip 3.

Because of that, the model may receive a shorter portion of the voice reference for each clip than it does in your standard workflow. This could potentially affect characteristics such as accent, pronunciation, rhythm, or voice style.

Your audio duration, clip duration, and number of clips will help me confirm whether this is what is happening in your case.


> **Comment by tritant (2026-09-30T12:08:12Z):**

Thanks, I’ve made a change for this.

Local audio references are now handled differently from global audio references.

A local audio reference is no longer truncated to the generated clip duration. The full local reference is now passed to H3, up to H3’s native 15-second reference limit.

So for example:

- 5-second generated clip
- 12-second local audio reference

H3 will now receive the full 12-second audio reference instead of only the first 5 seconds.

Important: for this behavior, you must use a **local audio reference on the clip card**.

Global audio references still keep the Extender’s existing timeline behavior, because they are designed to work across multiple clips and may be split between them.

So if the goal is to preserve a specific voice, accent, pronunciation, rhythm, or speaking style as closely as possible for one clip, please use the audio as a **local reference**, not as a global reference.

Update to 2.9.2


> **Comment by jdorpitan-code (2026-10-01T13:01:44Z):**

Hi Tritant,
thank you very much for your consideration.
Originally my mp3 was a 50 second speech. With your remark I have retried with a 10 seconds mp3 but I still have the "lost accent" problem.
I have also tried your new version 2.9.2 (delete + git clone of your repo) but this version also have this "lost accent" pb.


> **Comment by tritant (2026-10-01T14:10:57Z):**

Did you correctly use the local references? Don't connect the audio reference to the node; use the "ref" button in the clip map.

<img width="276" height="212" alt="Image" src="https://github.com/user-attachments/assets/e4f12fbf-18e3-46a0-97d7-e02150a80d30" />


> **Comment by jdorpitan-code (2026-10-03T12:42:03Z):**

yes I did both : the local references, with a 10 second mp3, and also by adding an audio on the general ref_audio_1 entry of the "minimax h3 extender". For all the cases, the strong english accent disappear on the generated video.


> **Comment by tritant (2026-10-03T13:43:02Z):**

I checked this more closely against the current official ComfyUI MiniMax H3 implementation.

At this point, for a **local audio reference**, I can no longer find a meaningful difference in the way the Extender prepares and sends the audio reference to H3 compared with the official implementation.

The local reference starts from the beginning of the source, is not shortened to the duration of the generated clip, and is encoded/passed to H3 through the same type of audio-reference conditioning.

So if the same audio produces noticeably different voice characteristics in the official workflow and in the Extender, I would now be very interested in seeing the exact prompt used in both cases.

One important point is that a timbre reference is not a special audio type in the node. `<Audio 1>` is still a normal audio reference. Its role as a **voice-timbre reference** is defined by the prompt itself.

For example, this is the kind of complete H3 prompt I mean:

```text
subject_definitions:
<Subject 1> is the woman in <Picture 1>; preserve her facial identity, hairstyle, clothing, and general appearance.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1), containing an English-accented female voice.

summary:
[reference generation + audio reference] Generate a realistic video of <Subject 1> speaking French, while preserving her visual identity from <Picture 1> and using <Audio 1> as the reference for her voice timbre, English accent, and delivery.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - preserve the identity, face, hairstyle, clothing, and overall appearance defined by <Picture 1>.
<Audio 1>: reference - use its voice timbre, English accent, rhythm, and delivery for <Subject 1> without copying the original spoken words or audio signal.

detailed_description:
Realistic live-action cinematography with natural lighting and restrained camera movement.

[Shot 1] A medium close-up frames <Subject 1> looking naturally toward the camera. <Subject 1> (S1) speaks fluent French using the same voice timbre, clearly recognizable English accent, vocal character, rhythm, and delivery referenced from <Audio 1>. She says: <d>[French] Bonjour, je suis très heureuse de vous retrouver aujourd'hui.</d>

overall_soundscape:
Clean natural room ambience. <Subject 1>'s spoken voice is clear and dominant, with the voice timbre, English accent, and delivery referenced from <Audio 1>.

non_diegetic_music:
N/A
```

That is different from simply mentioning `<Audio 1>` somewhere in the prompt. In the official H3 prompt format, the role of the audio reference is explicitly established in `subject_definitions`, then declared as `reference` in `retention_analysis`, and finally applied to the speaker in `detailed_description`.

So for a timbre test, I would recommend using a **local Audio reference** together with this kind of explicit prompt structure.

There is one other variable worth checking: **Turbo LoRAs**.

A Turbo LoRA changes the model's sampling behaviour quite significantly, usually with far fewer steps. This can sometimes reduce how faithfully some conditioning is followed, including audio references, voice characteristics, timing or other subtle audio details.

So if you are comparing the Extender with a normal MiniMax H3 workflow, please make sure both tests are actually using the **same model, the same Turbo LoRA (if any), the same strength, the same sigmas/sampler configuration and the same number of steps**.

In particular: are you using exactly the same Turbo LoRA in the normal workflow where the English accent is preserved?

If the normal workflow is using standard H3 sampling while the Extender test is using a 4-step Turbo LoRA, then that would be an important difference to eliminate before assuming the audio-reference handling itself is responsible.

If you can provide the exact prompt where the official MiniMax H3 workflow preserves the English accent but the Extender does not, together with the model/LoRA/sampling settings used in both workflows, I can compare them directly. At the moment, I cannot find an obvious difference in the current local-audio reference path that would explain the behaviour.



### #99 [closed] ISSUE: [Feature] Optional external SIGMAS input for Extender
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/99  by Kray76 2026-09-27T07:09:23Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

I would like MiniMax H3 Extender to support an optional external SIGMAS input.
Currently, the Extender generates its sigma schedule internally from scheduler, steps, and denoise. This prevents using sampling methods that require their own fixed or custom sigma schedules, such as HyperFlow and potentially other H3 distillation methods.
Proposed behavior:
- Add an optional SIGMAS input to the Extender.
- If external sigmas are connected, use the supplied sigma schedule directly for sampling.
- If no external sigmas are connected, keep the current internal sigma generation unchanged.
- Existing workflows should remain fully backward-compatible.
- When external SIGMAS are connected, scheduler, steps, and denoise should not regenerate or modify the supplied schedule.

This would make the Extender compatible with HyperFlow and other sampling/distillation methods that provide their own sigma schedule, without changing the behavior of existing workflows.

### Problem or limitation

The capability to not be able to connect nodes with their own sigmas.

### Proposed solution

Simple Input connection for sigmas and the switch logic inside your node, if there is a sigma connected.

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-09-27T08:49:24Z):**

see https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/pull/83
And try the branch test-pr83


> **Comment by tritant (2026-09-27T11:05:24Z):**

added to main



### #98 [closed] PR: Add a shared local image library for per-clip references
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/pull/98  by Localsmile 2026-09-26T10:53:31Z

Adds drag-and-drop uploads for reusable local images and per-clip selection. Global images are numbered first, followed by selected local images, up to nine Pictures per clip.

This is an optional workflow improvement based on personal preference, not a required fix.

image upload areas
<img width="614" height="323" alt="1" src="https://github.com/user-attachments/assets/1fd799a8-5187-467d-859b-7959e1b4471c" />


per-clip reference selection
<img width="612" height="462" alt="2" src="https://github.com/user-attachments/assets/bd59ec26-575f-4d60-99d9-546f70035019" />


> **Comment by tritant (2026-09-26T12:42:13Z):**

Thanks for the PR. I spent some time going through the implementation in detail.

I like the general idea of having a shared local image library, especially because it makes it possible to keep more than 9 local images available and select only the ones needed for each clip.

However, I don't think I can merge the PR in its current form because there is one architectural point that conflicts with how the Extender currently handles references: **`<Picture N>` slot identities must remain stable.**

At the moment, the new implementation compacts the active references for each clip.

For example, if the project contains:

- global `<Picture 1>`
- empty global slot 2
- global `<Picture 3>`
- one selected local image

the backend effectively sends them as Picture 1, Picture 2 and Picture 3.

The JavaScript then tries to compensate for that by automatically rewriting `<Picture N>` references inside prompts.

This is something I deliberately want to avoid. In the current Extender, a reference slot is canonical: `<Picture 3>` must always mean slot 3. Empty slots must remain empty and must never cause later references to shift.

There are several reasons why this matters.

One important case is the Prompt Pack. The imported prompt replaces the clip prompt directly, while the references can then be compacted separately. Unlike the Ref Pack handling, the Prompt Pack path does not remap the `<Picture N>` values afterward.

That means a prompt containing `<Picture 3>` can potentially point to a different image on the first Queue if there is a hole in the global slots and a local-library image is selected.

I also noticed that the Ref Pack allocation is currently based on the maximum number of local images selected by any clip. This makes the restriction global to the project.

For example, if Clip 1 uses no local references but Clip 2 uses 8, the Ref Pack can effectively be reduced to a single global reference for the whole project, even though Clip 1 would otherwise have room for all 9.

Another point is project size. The entire local library is embedded into the `.ext` project, including images that are not used by any clip. With a small library that is fine, but a library containing dozens of large images could make project files unnecessarily large.

The new drag-and-drop behavior also changes the existing reference workflow quite significantly. That isn't necessarily a problem by itself, but I would prefer to preserve the current explicit slot behavior rather than introduce another reference system that internally renumbers pictures.

That said, there are several parts of the implementation that I do like:

- adding an image to the library without using it does not invalidate clips;
- selecting or editing an image correctly invalidates the affected clips;
- legacy local-reference slots remain compatible;
- library image replacement correctly propagates to clips using that image;
- the feature comes with dedicated Python and JavaScript tests.

So I think the **shared library concept itself is good**, but I would like the reference assignment mechanism to follow the existing Extender rule:

**global picture slots keep their exact numbers, and local-library images are assigned to actual free `<Picture N>` slots for that clip.**

For example:

- global Picture 1 remains `<Picture 1>`
- global Picture 3 remains `<Picture 3>`
- if slot 2 is free, a selected local image can become `<Picture 2>`

No automatic prompt rewriting would then be necessary, Prompt Pack and Ref Pack would continue to use the same canonical numbering system, and inserting/removing/selecting local images would never silently change what an existing `<Picture N>` refers to.

So I would not merge the PR as-is, but I think the library idea is worth keeping if the slot/remapping architecture can be changed in that direction.


> **Comment by tritant (2026-09-26T13:14:18Z):**

Thinking about it a bit more, my preferred direction would actually be slightly different.

I would keep the current reference system exactly as it is today, with the existing fixed `<Picture N>` slots and no remapping at all.

Instead of integrating the library directly into the slot numbering logic, I would rather make the image library a separate tool with its own `Open Library` / `Save Library` / `Load Library` controls.

Then, when the user clicks to add or replace a reference in a slot, they would simply get two choices:

- load the image from disk
- choose the image from the library

Once an image is selected from the library, it should behave exactly like an image loaded from disk and become a normal reference assigned to that specific `<Picture N>` slot.

For example, if the user selects a library image for slot 4, it simply becomes `<Picture 4>`. From that point on, the existing system handles everything normally: preview, editing, project save/load, invalidation, Prompt Pack, Ref Pack, Motion Context, etc.

The library itself could contain any number of images, but it would remain completely separate from the project. The project should only store the references that are actually assigned to slots, exactly as it does now, rather than embedding the entire library.

I also think this would keep project portability much cleaner: once a library image is assigned to a slot, that image becomes part of the project/reference data, so loading the project elsewhere would not require the original external library.

This approach would preserve the current guarantees:

- `<Picture N>` always refers to the same slot
- empty slots stay empty
- no automatic prompt rewriting
- no reference compaction
- no changes to Prompt Pack or Ref Pack behavior
- no interaction between unused library images and clip invalidation

So in my view, the library should really be a convenient **source of reusable images**, not a new reference-management layer inside the H3 pipeline.

That would be much closer to the architecture I would like to keep for the Extender.


> **Comment by tritant (2026-09-26T13:54:30Z):**

Thanks again for the work on this. After thinking through the UX and architecture a bit more, this is the direction I would prefer for an image reference library.

# Image Reference Library — Proposed Specification

## 1. Goal

Add an **independent reusable image library** that allows users to keep and reuse a large number of reference images without changing the current MiniMax H3 Extender reference system.

The Library should only be an **additional image source**.

It should not become a new H3 reference-management system.

The current reference behavior should remain unchanged:

- maximum of 9 active references;
- `<Picture N>` always refers to the same slot;
- no remapping;
- no slot compaction;
- no automatic prompt rewriting.

---

## 2. Core principle

The Library and the references currently used by a project should remain completely separate concepts.

### Library

A personal collection of reusable images.

It can contain far more than 9 images:

- 20;
- 50;
- 100;
- more if needed.

An image in the Library is not considered an H3 reference until the user explicitly assigns it to an actual reference slot.

### Project references

The existing reference system remains unchanged.

An image becomes an H3 reference only when it is assigned to one of the real slots:

`<Picture 1>` through `<Picture 9>`.

Once an image comes from the Library and is assigned to a slot, it should behave **exactly like an image loaded from disk**.

---

## 3. Library UI

Add a separate collapsible panel:

**Reference Library**

At minimum it should contain:

- `Add Images`
- `Save Library`
- `Load Library`
- a thumbnail grid showing the images currently stored in the Library.

Optionally, each thumbnail could also have a short name or label.

The Library itself should not depend on the current generation mode:

- Ref2VA Motion Context ON;
- Ref2VA Motion Context OFF;
- FL2VA.

It should simply be a generic reusable image source.

---

## 4. Add Images

`Add Images` should open the system file picker.

The user should be able to select **multiple images at once**.

Those images are simply added to the Library.

This operation should:

- create no `<Picture N>`;
- modify no clip;
- modify no prompt;
- trigger no invalidation;
- modify no generation cache.

Adding an image to the Library should therefore be a completely passive operation.

---

## 5. Add an existing reference to the Library

An image already loaded as a project reference should have an action such as:

**Add to Library**

For example, the user could load an image normally from disk into `<Picture 3>`, then later decide that it is useful enough to keep in the reusable Library.

This should never happen automatically.

Loading an image into a normal reference slot must **not automatically add it to the Library**.

---

## 6. Using the Library for global references

When the user adds or replaces a global reference, the selector should offer two sources:

**From Disk**

or

**From Library**

### From Disk

Keep the current behavior exactly as it is today.

### From Library

Display the Library thumbnails.

The user selects an image.

That image is then assigned directly to the selected slot.

For example, if the user is editing `<Picture 4>` and selects an image from the Library, that image simply becomes:

`<Picture 4>`

There should be no concept of a "Library image number" inside the H3 reference system.

---

## 7. Drag & Drop — global references

Library thumbnails should also be draggable directly onto global reference slots.

Example:

**Library thumbnail → `<Picture 5>`**

Result:

the image becomes the global `<Picture 5>` reference.

If the slot is empty:

→ add it.

If the slot is already occupied:

→ replace it using exactly the same logic as the current Replace action.

While dragging, only valid reference slots should visually highlight as drop targets.

---

## 8. Drag & Drop — local references

Local references need slightly different behavior.

Currently, local references are managed through the **Ref** button on each clip card. Opening that interface covers the content behind it, so dragging directly into the expanded local-reference panel would not be practical.

### Proposed behavior

The **Ref button itself becomes a drop target**.

Example:

**Library thumbnail → Clip 4 `Ref` button**

Result:

the image is immediately added as a **new local reference for Clip 4**.

The Ref panel does not need to open.

The existing local-reference logic should decide which canonical `<Picture N>` slot is available, exactly as it does today when adding a local reference normally.

### Important

**Only the physical `Ref` button area should accept this drop.**

The entire clip card should not become a drop target.

This avoids ambiguity with:

- prompts;
- card selection;
- scrolling;
- other interactions;
- existing drag-and-drop behavior.

While dragging over a valid Ref button, it could visually highlight and show something like:

`+ Add local reference`

---

## 9. Existing local-reference UI

The current local-reference management system should remain fully available.

Clicking **Ref** should continue to open the existing local-reference interface.

The Library does not replace that UI.

It only adds a faster path:

**drag Library thumbnail → Ref button**

to create a new local reference.

If all 9 H3 reference slots are already occupied for that clip, the drop should simply be rejected cleanly without modifying the clip.

---

## 10. Absolute `<Picture N>` rule

The Library must never change slot identity.

Example:

- global `<Picture 1>` occupied;
- `<Picture 2>` empty;
- global `<Picture 3>` occupied.

`<Picture 3>` must always remain `<Picture 3>`.

If a new local reference can use slot 2, then it becomes `<Picture 2>`.

The system must never transform:

`Picture 1 + Picture 3`

into:

`Picture 1 + Picture 2`

through compaction.

So the rule should remain:

**no remapping, no renumbering, no automatic prompt rewriting.**

---

## 11. Prompt Pack and Ref Pack

No behavior change.

The Library should be transparent to:

- Prompt Pack;
- Ref Pack;
- manually entered prompts;
- global references;
- local references.

A prompt containing:

`<Picture 5>`

must continue to refer exactly to slot 5.

The Library system should never inspect or rewrite `<Picture N>` references inside prompts.

---

## 12. Saving the Library

The Library should have its own independent save system.

Button:

**Save Library**

This could create a dedicated file, for example:

`MyLibrary.h3lib`

The file should contain the images required to restore the Library, and optionally their names/labels.

The Library should therefore not be tied to a ComfyUI workflow or `.ext` project.

---

## 13. Loading the Library

Button:

**Load Library**

This reloads a previously saved Library.

Loading a Library should:

- modify no active reference slot;
- modify no clip;
- trigger no invalidation;
- modify no prompt.

It should only populate the Library panel.

The exact UX for **Replace Library / Merge Library** can be decided later without affecting the core architecture.

---

## 14. `.ext` project saving

This is important:

**the complete Library should never be embedded into every project.**

If the Library contains 100 images and the current project uses only 3 of them, the `.ext` file should not embed all 100 images.

When a Library image is assigned to a real reference slot, it becomes a normal project reference.

From that point on, it should be stored inside the project in exactly the same way as an image loaded from disk.

This preserves project portability.

Example:

local Library contains:

`Gertrude.png`

It is assigned to `<Picture 4>`.

The project is saved.

That project should be transferable to another computer and still restore `<Picture 4>` even if the original `.h3lib` file is not available there.

---

## 15. Invalidation behavior

The Library itself should never trigger generation invalidation.

The following operations should not change clip generation state:

- Add Images;
- Load Library;
- Save Library;
- removing an unused image from the Library;
- reorganizing the Library.

Invalidation should only happen when an image is actually assigned to or replaces a reference slot in the project.

At that point, the Extender should simply use the **existing invalidation behavior**.

The Library should have no separate invalidation logic of its own.

---

## 16. Motion Context and caches

No changes.

The Library should not interfere with:

- Motion Context ON;
- Motion Context OFF;
- disk caches;
- clip validation;
- Final Decode;
- Audio Context;
- seeds;
- Full Batch;
- Clip-by-Clip.

Once an image is assigned to a reference slot, the existing Extender pipeline should take over completely.

---

## 17. Image editor

Once an image from the Library is assigned to a reference slot, it becomes a normal reference.

The existing reference-image editor should therefore continue to work normally:

- saturation;
- color adjustments;
- Reset Original;
- thumbnail refresh;
- etc.

The Library does not need its own separate image editor.

---

## 18. Compatibility

This feature should be built **around the existing stable reference system**, not replace it.

It must preserve compatibility with:

- existing workflows;
- existing `.ext` projects;
- global reference slots;
- local reference slots;
- Ref2VA;
- FL2VA;
- Legacy frontend;
- Node 2.0 frontend.

A workflow that never uses the Library should behave exactly as it does today.

---

## 19. Things the Library should not do

The Library should not:

- become a second H3 reference list;
- change the 9-reference model limit;
- compact reference slots;
- renumber `<Picture N>`;
- rewrite prompts;
- reserve slots globally;
- modify Ref Pack behavior;
- modify Prompt Pack behavior;
- embed the entire Library into every project;
- invalidate clips simply because the Library changes;
- automatically update project references when the corresponding Library image changes.

---

## 20. Final user workflow

### Normal usage

`Add Reference → From Disk`

Exactly as today.

### Using the Library

`Add Reference → From Library → choose thumbnail`

### Fast global assignment

`drag Library thumbnail → global reference slot`

### Fast local assignment

`drag Library thumbnail → clip Ref button`

### Keeping a useful current reference

`current reference → Add to Library`

---

## Architectural summary

**The Library stores reusable images.**

**The `<Picture N>` slots store the references actually used by H3.**

The only connection between them is an explicit user action:

**"Take this image from my Library and put it into this reference slot."**

From that moment on, **the Library is no longer involved**.

That is the direction I would prefer because it gives us a genuinely useful reusable image library while keeping the stable reference architecture completely intact.


> **Comment by tritant (2026-09-26T14:25:56Z):**

Merged to the branch test-pr98


> **Comment by Localsmile (2026-09-26T17:03:18Z):**

@tritant Thanks for the detailed specification and for merging the current work into test-pr98. I realize my initial design did not sufficiently account for the Extender’s fixed-slot reference architecture.

I’ll revise this PR so the library is a separate source of reusable imageㄴ, assigned through the existing global and local reference slots. This will preserve the current reference behavior and keep .ext projects limited to assigned images. I’ll update the PR once the implementation and tests are ready.



### #97 [closed] ISSUE: [Bug] minimax-h3_fl2va_4stepsv0.1_int8_convrot.safetensors Wher ei sit
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/97  by OcBuddy 2026-09-24T08:21:46Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

NA

### ComfyUI version

_No response_

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Clip-by-Clip

### Describe the issue

Your minimax_extender.json example workflow references:
minimax-h3_fl2va_4stepsv0.1_int8_convrot.safetensors

I can't find it anywhere. Can you add a notecard in the workflow so people know where it can be found?

### Steps to reproduce

NA

### Error message / console log

```shell
NA
```

### Workflow / additional information

NA

### System information

NA

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-09-24T09:02:20Z):**

You won't be able to find it; it's a version of fl2va that I merged with a 4-step LoRA. This avoids the need to use the "Load LoRA" node, saving a few seconds. It isn't available for download.



### #96 [closed] ISSUE: [Feature]
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/96  by Frisby572 2026-09-22T22:24:23Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

**Inserting new clip into the flow**

Suppose one wants to do either of  (1) prepend a clip to the current sequence, or (2) insert a clip between two consecutive clips. Afterwards to re-number the prompt sequence and reconstruct the video.

Given that each clip prompt has links to associated numbered reference files, it's not obvious how a rearrangement can be done.

Alternatively, suppose one instead is obliged to create a new workflow, which prompt related components of the existing workflow may be copied across in a simple fashion?

In other words, does one have to start again from scratch when inserting new clip prompts among the body of already prepared prompts?

### Problem or limitation

Enhance flexibility of prompt inputs.

### Proposed solution

_No response_

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-09-23T05:36:04Z):**

This functionality already exists.

In **Ref2VA with Motion OFF**, and in **FL2VA**, each clip card has a `+` button that inserts a new clip directly after it. Clips can therefore be inserted anywhere in the sequence without starting a new workflow.

Also, `<Picture N>` does **not** refer to Clip N. Reference slots have stable identities and are never renumbered or remapped when clips are inserted or removed. For example, `<Picture 3>` remains `<Picture 3>` regardless of the clip's position in the sequence.

With **Motion Context ON**, clips are intentionally sequential: each clip depends on the generated context of the previous one. Therefore, changing the sequence in the middle necessarily invalidates the dependent clips that follow.

This behavior is described in the README, particularly in the **Motion Context OFF** .



### #95 [closed] ISSUE: [Feature] music
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/95  by drel-ai 2026-09-22T13:56:35Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

I can add a some music to the first clip, but for the next clip it will not take the continuation of that music piece. (shape mismatch: value tensor of shape [810, 32] cannot be broadcast to indexing result of shape [884, 32])

Can we get something that take the whole music for all clips? Of course I can add music later, but this will not work for interacting to the music like dancing.

Or am I doing something wrong? 

### Problem or limitation

it may solve the bad minimax music

### Proposed solution

_No response_

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-09-22T16:14:04Z):**

This should already work. A single long audio reference is automatically sliced across the Extender timeline, so Clip 2 should receive the continuation after Clip 1. I tested this locally and it works here.

The 810 -> 884 shape mismatch therefore looks specific to your setup or audio/clip durations rather than a missing feature.

Could you please provide:

Extender version
ComfyUI version/commit
Ref2VA Motion Context ON or OFF
duration of Clip 1 and Clip 2
exact duration of the music file
whether you use only ref_audio_1, or also video/audio references
ideally the workflow JSON

In particular, I'd like to know whether the music file ends during the second clip.


> **Comment by drel-ai (2026-09-22T17:33:02Z):**

Extender version:   2.8.4
ComfyUI version/commit    v0.37.0 , potable
Ref2VA Motion Context ON or OFF:   on
duration of Clip 1 and Clip 2:   clip1-clip 7 all 10seconds
exact duration of the music file: 70sec
whether you use only ref_audio_1, or also video/audio references: only ref_audio_1
ideally the workflow JSON: its you "MiniMax_Extender_external_refs_and_external_prompts.json" workflow coming with the node

its clip by clip, with audio_context_length: 22

I tried with the pixaroma node :"2. Upload Audio and adjust duration"  and the native audio loader 

Isn't there a difference between an audio file used as a reference, e.g for a certain voice or dialect and an audio track you want to 
play 1:1 without any change? 

After your post I tried different config, other nodes and other audio files but always get the error below:

--------------------------------------------------------------------------------------------------------------------

[ERROR] !!! Exception during processing !!! shape mismatch: value tensor of shape [810, 32] cannot be broadcast to indexing result of shape [884, 32]
[ERROR] Traceback (most recent call last):
  File "P:\DECODE\COMFY_EASY\ComfyUI\execution.py", line 547, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\execution.py", line 352, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\execution.py", line 326, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "P:\DECODE\COMFY_EASY\ComfyUI\execution.py", line 314, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 3852, in wrapped
    result = function(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 5273, in extend
    sampled = _sample_h3(
              ^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 1857, in _sample_h3
    samples = guider.sample(
              ^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\custom_nodes\ComfyUI-TiledDiffusion\utils.py", line 34, in KSAMPLER_sample
    return orig_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\k_diffusion\sampling.py", line 205, in sample_euler
    denoised = model(x, sigma_hat * s_in, **extra_args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\model_base.py", line 210, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\model_base.py", line 254, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\ldm\minimax\model.py", line 577, in forward
    graph_out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "P:\DECODE\COMFY_EASY\ComfyUI\comfy\ldm\minimax\model.py", line 711, in _forward
    all_audio_rows[~audio_update] = cond_audio_rows
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
RuntimeError: shape mismatch: value tensor of shape [810, 32] cannot be broadcast to indexing result of shape [884, 32]




> **Comment by tritant (2026-09-22T18:41:15Z):**

Thanks for the detailed information.

Before investigating this as a possible ComfyUI 0.37.0 compatibility issue, could you please check whether you have **ComfyUI-H3-Multishot** installed in your `custom_nodes` folder?

We have already had several issues caused by H3-Multishot interfering with the Extender, even when its nodes were not used in the workflow.

If it is installed, please temporarily disable/remove the entire `ComfyUI-H3-Multishot` folder, restart ComfyUI completely, and try the exact same workflow again.

The same setup works correctly here on ComfyUI 0.36.0 with Motion Context ON and a long `ref_audio_1`, so I would like to rule out H3-Multishot first before looking into a possible 0.37.0 regression.

See:
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/65


> **Comment by drel-ai (2026-09-22T20:09:45Z):**

ComfyUI-H3-Multishot was the bad guy. I dont even use it, but it has that one very little node to connect/discount pictures with a button press. without Multishot it takes the audio though all clips. thx for helping :)

Maybe at a little script that scans if ComfyUI-H3-Multishot is installed and throw out a warning? 

EDIT: ok, as said before, it takes the music for all clips, but now in every clip the track start over and does not run all the way through from start to finish.

EDIT 2: I will try if your advice in #94 (better prompt) will help, but for my understanding a pure music track should run through without special prompting

One unrelated question, I read you stopped working at the test-pr83 because of the latent-upscaler. Will you you work at another solution for a second refine pass with upscaling in pixel-space? maybe with a blue output so ppl can use what ever they want to upscale like a sharpener, or upscale with model, or add a little bit of noise, or RTX Video Super Resolution and then reinject it for a second pass? 

the advantage to do the 2nd refine step in H3-Extender is that you can use the exact prompt you used to create the video. Atm when I upscale with the help of a "Temporal Split" node, I just add a generic line like "Enhance this video with sharp, crisp details while preserving a natural photorealistic appearance."  I do not know, but I think using the original prompt might be much better.


> **Comment by tritant (2026-09-22T20:33:43Z):**

You must use this type of prompt in every clip.

subject_definitions:
<Audio 1> is the source sound track.

summary:
[reference generation + audio reuse] The target video uses <Audio 1> as its continuous sound track.

retention_analysis:
<Audio 1>: fully_copy - preserve the supplied audio segment as faithfully as possible.

detailed_description:
The visual action follows the timing and beat structure of <Audio 1>.
a woman walking in the street.

overall_soundscape:
N/A

non_diegetic_music:
<Audio 1> is directly reused as the complete sound track for this clip.

A second pass is the worst thing for a video aiming to maintain the best possible consistency between clips; even with light denoising, it reinvents the scene each time, altering numerous details, and so on.


> **Comment by drel-ai (2026-09-22T21:32:35Z):**

1) ok, i did some tests with audio lines added to the prompt. With audio_context_length 0 and with 22. Its is working better, the track does not start at the very beginning but there are gaps/jumps between every clip and from clip 3 on there were small modifications to the track. 

is there no possibility to add how strong the track is be influenced by the model? to harden it, or do some cutting magic to split the track so its length fits with the length of the track?  As I said, I can add a track after the video is created, but I want the video be influenced by the track, but the track should not be influenced in any way. I realy wonder why it is working fpr you and not for me :(  

2) Sry I hve t ask again, because upscaling is a big topic.  You said:" even with light denoising, it reinvents the scene each time, altering numerous details, and so on." but isnt this what upscaling should do, fixing faces, fingers, background ppl and small details by inventing new structures? If I only want a higher resolution with a sharper picture I can use "upscale with model" or "RTX Video Super Resolution" but that will not fix faces or anything else. 


> **Comment by tritant (2026-09-23T05:14:48Z):**

Don't forget that, originally, Minimax can only generate a single clip lasting a maximum of 15 seconds; everything else—such as maintaining motion continuity or splitting audio into multiple clips—relies on workarounds that come with their own flaws. Unfortunately, the Minimax documentation offers no way to prevent the model from altering the audio. The best way to achieve the result you're looking for is to manually split the audio into seven segments and use each piece as a local reference for the clips.

There’s no need to dwell on latent upscaling—you’ve clearly grasped its value—but the problem is that it never recreates the exact same details! A house might have three windows one time and five the next; its biggest issue is that it consistently breaks any text present in a reference image.



### #94 [closed] ISSUE: [Bug] Voice cloning not working properly
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/94  by oliacym 2026-09-22T12:19:21Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

2.8.4

### ComfyUI version

0.36.0

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Clip-by-Clip

### Describe the issue

On using any audio as a ref, the voice does not sound as accurate as when doing them in on a regular ref2va workflow. The problem is even worse on Clip 2, where the second voice sounds very little like the ref.

### Steps to reproduce

1. Add clip with audio ref using ref2va model
2. Generate clip
3. Resulting voice clone is inferior to regular workflow not using Minimax H3 Extender
4. Clip 2 is even less accurate

### Error message / console log

```shell

```

### Workflow / additional information

[MiniMax_H3_preview_20260921_144933_5f7c.json](https://github.com/user-attachments/files/32517425/MiniMax_H3_preview_20260921_144933_5f7c.json)

### System information

OS: Windows 11
GPU: RTX 5080 16GB

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-09-22T16:59:44Z):**

I think it's because your prompt isn't complete; you absolutely must ask Minimax to keep the reference intact, otherwise it might reinterpret it in its own way.
Here is an example of a prompt designed to keep the reference as intact as possible.

subject_definitions:
<Audio 1> is the source sound track. 

summary:
[reference generation + audio reuse] The target video uses <Audio 1> as its continuous sound track.

retention_analysis:
<Audio 1>: fully_copy - preserve the supplied audio segment as faithfully as possible.

detailed_description:
The visual action follows the timing and beat structure of <Audio 1>.
a woman walking in the street.

overall_soundscape:
N/A

non_diegetic_music:
<Audio 1> is directly reused as the complete sound track for this clip.

Otherwise, there is no way for me to improve the audio quality, which depends heavily on the models, turbo LoRAs, steps, etc.


> **Comment by oliacym (2026-09-22T17:55:50Z):**

Sorry but this isn't to use an audio file in it's entirety - this is for voice cloning specifically. That's why my prompt says in the subject definition (just the relevant parts of my layout):

subject_definitions:
<Audio 1> is the voice-timbre reference for the red-haired woman, <Subject 1> (S1).

summary:
[reference generation] The target video is an animated scene that takes place in <Subject 3>. In it, <Subject 1> and <Subject 2> are sitting across from each other in a restaurant. <Subject 1> has a line of dialogue.

retention_analysis:
<Audio 1>: reference - its vocal timbre guides the dialogue delivery of <Subject 1> (S1) without copying the original signal.

(Code ends)

Is this not enough specifics for Minimax? Like I say, it works fine outside of this node in a regular workflow. But I can't seem to get the latest version to sound like the voice reference provided. Thank you for your help!


> **Comment by tritant (2026-09-22T18:06:41Z):**

Your prompt is correct; however, I do not have a solution to improve this aspect.



### #93 [closed] ISSUE: [Bug] ref_audio voice cloning works in ref2va but is ignored in fl2va mode
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/93  by AaronPorts 2026-09-22T00:00:15Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

2.8.4

### ComfyUI version

_No response_

### ComfyUI frontend

Nodes 2.0

### Generation mode

FL2VA

### Run mode

Clip-by-Clip

### Describe the issue

Voice cloning via ref_audio does not work when using the fl2va mode in MiniMax H3 Extender. The generated video does not use the cloned voice from the provided MP3 file. However, the exact same audio file clones perfectly when switched to ref2va mode.

### Steps to reproduce

Add the MiniMax H3 generation node.
Connect a valid .mp3 voice recording to the ref_audio input.
Set the mode/preset to fl2va and queue the prompt -> Voice is not cloned.
Change the mode to ref2va using the same audio file and queue the prompt -> Voice clones successfully.

### Error message / console log

```shell

```

### Workflow / additional information

_No response_

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-09-22T05:24:16Z):**

This is expected behavior.

The original FL2VA model does not support audio references like Ref2VA does.

Some people are currently experimenting with ways to add voice/audio reference support to FL2VA, but those approaches are still experimental.

If a reliable implementation appears and works properly, we can look at integrating it into the Extender.



### #92 [closed] PR: Follow-up fixes for PR #83
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/pull/92  by tritant 2026-09-21T21:42:40Z




### #91 [closed] PR: Sync test-pr83 with main 2.8.4
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/pull/91  by tritant 2026-09-21T21:37:52Z




### #90 [closed] PR: Fix cache invalidation with linked resolution inputs
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/pull/90  by Localsmile 2026-09-21T17:13:47Z

When width or height is connected, the hidden widgets can retain stale fallback values. Restoring the node after switching tabs then treats the cached resolution as changed and clears validated clips.

Skip frontend resolution invalidation for linked inputs. Resolved values are still validated during backend execution, and direct width or height changes continue invalidating the cache.

Tests: `node --test tests/*.test.mjs` (22 passed)

> **Comment by tritant (2026-09-21T21:18:00Z):**

Thanks for the fix.

I checked it against the current 2.8.4 code and confirmed the issue is real: when `width` / `height` are provided through links, the hidden native widget values can be stale, and the frontend can incorrectly think the project resolution changed after a tab switch or rebuild.

Your fix makes sense — if either dimension is linked, the frontend should not try to infer the effective resolution from the hidden widget values. The backend will still validate the real connected resolution on the next run.

I also added the regression tests locally and they pass.

Merged, thanks!



### #89 [closed] PR: Follow-up fixes for PR #83
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/pull/89  by tritant 2026-09-21T04:42:02Z




### #88 [closed] ISSUE: [Bug] Cannot uncheck "Validated" when the cache manifest has no segments ("Invalid clip index")
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/88  by zpengcom 2026-09-21T01:20:56Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

Latest

### ComfyUI version

Latest

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Clip-by-Clip

### Describe the issue

**Describe the bug**
In Ref2VA with Motion Context ON, if clip 1 is marked Validated but its cache is missing,
running raises:

RuntimeError: MiniMax H3 Extender: Ref2VA clip 1 is marked Validated but its disk cache is missing.
Refusing to rerender a validated clip. Uncheck Validated explicitly if you want this clip rendered again.

But unchecking Validated as instructed fails with:

Local reference invalidation failed: Invalid clip index.

So the node asks the user to uncheck the box, and then refuses the uncheck.

**Cause**
The manifest file exists but has an empty list (cache/chain_extender_1.json contains "segments": []).
In the /h3_extender/local_ref_invalidate route (motion_context_disk.py), the causal Ref2VA branch does:

    if clip_index < 0 or clip_index >= len(segments):
        return web.json_response({"ok": False, "error": "Invalid clip index."}, status=400)

With 0 segments, clip_index 0 is out of range, and this check also runs when the request is
an un-validation (requested_validated == False). The ref2va_independent and fl2va branches
silently ignore an out-of-range index, so only this branch is affected.

**Suggested fix**
When `requested_validated` is False and the index is out of range, there is nothing to
invalidate on disk, so return {"ok": True, "found": False} (same as when the manifest is missing)
instead of a 400. Keep the error only for the validate=True case.

**Workaround**
Move cache/chain_extender_1.* out of the cache folder, refresh the page, uncheck Validated.

**Environment**
Windows, ComfyUI portable, node build motion-context-disk-v2.8.2

### Steps to reproduce

up

### Error message / console log

```shell

```

### Workflow / additional information

_No response_

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-09-21T07:55:14Z):**

Thanks for the detailed report — I was able to confirm the issue.

The problem happens when a clip is still marked as `Validated` in the workflow state, but its physical cache segment no longer exists. In that situation, the backend was rejecting the unvalidation request with `Invalid clip index` before allowing the UI to clear the validated state.

This has been fixed in **v2.8.3**.

The new behavior is:

- `Validated = false` + missing cache segment → allowed, so the user can recover from the inconsistent state
- `Validated = true` + missing cache segment → still rejected

So the safety rule remains intact: a validated clip cannot silently regenerate without its cache, but it is now possible to properly unvalidate it when the cache has disappeared.

Thanks again for catching this.



### #87 [closed] ISSUE: [Bug] ref_video_audio_N rejects VHS Load Video audio ("not a valid AUDIO payload"
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/87  by zpengcom 2026-09-21T01:18:41Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

Latest

### ComfyUI version

Latest

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Clip-by-Clip

### Describe the issue

**Describe the bug**
Connecting the `audio` output of "Load Video (Upload) 🎥🅥🅗🅢" (VideoHelperSuite) to `ref_video_audio_1`
fails with:

ValueError: MiniMax H3 Extender: ref_video_audio_1 is not a valid AUDIO payload.
  File "extender.py", line 1679, in _prepare_shared_refs
    soundtrack = _slice_ref_audio(
  File "extender.py", line 1248, in _slice_ref_audio
    raise ValueError(f"MiniMax H3 Extender: {label} is not a valid AUDIO payload.")

**Cause**
VHS returns its audio as a `LazyAudioMap`, which subclasses `collections.abc.Mapping`
(see videohelpersuite/utils.py). It supports `audio["waveform"]` and `audio.get("sample_rate")`
like a dict, but it is not a `dict` instance, so the `isinstance(audio, dict)` checks fail.

**Suggested fix**
Accept any Mapping in the two checks:
- `_audio_duration_seconds`
- `_slice_ref_audio`

    from collections.abc import Mapping
    if not isinstance(audio, Mapping) or "waveform" not in audio:
        ...

`_slice_ref_audio` already returns a plain dict afterwards, so nothing else needs to change.
I applied this locally and it lets the VHS audio through.

**Environment**
- Windows, ComfyUI portable
- Node build: motion-context-disk-v2.8.2
- VideoHelperSuite: <fill in version/commit>
- Mode: Ref2VA

### Steps to reproduce

Connecting the audio output of "Load Video (Upload) 🎥🅥🅗🅢" (VideoHelperSuite) to ref_video_audio_1
fails with:

### Error message / console log

```shell

```

### Workflow / additional information

_No response_

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-09-21T08:11:12Z):**

Thanks for the detailed report — this is fixed in **v2.8.4**.

The issue was that the Extender was checking whether the incoming `AUDIO` object was specifically a Python `dict`.

Some video loader nodes now return dict-compatible audio objects such as `LazyAudioMap`, which expose the same `waveform` / `sample_rate` interface but are not actual `dict` instances.

The check now accepts any `Mapping`, so both cases work:

- standard ComfyUI `AUDIO`
- lazy/dict-like audio objects such as `LazyAudioMap`

The existing validation of `waveform` and `sample_rate` remains unchanged.

This keeps compatibility with the native ComfyUI video/audio nodes while fixing support for the newer lazy audio payloads.

Thanks again for catching it.



### #86 [closed] ISSUE: [Feature Request] Improved preview window that updates periodically
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/86  by SyntaxError1871 2026-09-20T21:49:16Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

Hello,

A feature I think that would be useful especailly for long 20-30 step clip generations for quality is to have the preview window periodically update or have slightly more fidelity in order to sanity check if a clip is actually generating correctly or if it's gone off the rails or a image/video reference takes over the video generation and ruins it. Things like that are pretty obvious in a low resolution low fps preview window and would be good to have that option to just abort video generation if the preview image shows it to be ruined and not waste time/GPU processing time for it to finish before being able to tell if a clip has successfully generated or not.

An example ideal preview window would be something that is featured on the DaSiWa Minimax H3 workflow which features a node that connects to the "model" and "vae" endpoints and can display a low resolution 8fps preview of the video as it is generating real-time in order to sanity check an output.

<img width="998" height="1457" alt="Image" src="https://github.com/user-attachments/assets/1f9ec78f-e6d5-4c0c-87f6-3c7affad51c1" />

### Problem or limitation

Current preview window is very small and also can be hidden beneath the clip windows if there are a lot of clips added to the node. It also does not update at all, only showing presumably the first rendered frame at very low resolution.

### Proposed solution

Utilize a preview window solution similar to the one showcased in the DaSiWa Minimax H3 workflow here: https://civitai.red/models/2831978/dasiwa-minimax-h3-workflows-or-t2va-or-fl2va-or-ref2va, or at least something where the preview window is slightly higher quality and low fps to see more than just the first rendered frame.

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by SyntaxError1871 (2026-09-20T22:52:54Z):**

I was able to get the preview function to work with the "Model Override Preview" node in between the model and VAE loader, so I guess this feature request can be closed. Having a more detailed preview window is great for when doing longer gens with lots of steps and can tell early on if the video will be good or ruined early on without having to wait for the whole thing to finish.

<img width="1873" height="1241" alt="Image" src="https://github.com/user-attachments/assets/6480e593-7f3f-412f-a268-de7e8101d8f0" />

<img width="940" height="669" alt="Image" src="https://github.com/user-attachments/assets/b7e73cc8-768d-4814-b119-f7d6bc059301" />


> **Comment by Purplegopher (2026-09-21T06:08:05Z):**

Thank you so much for sharing that screenshot, I never knew that node existed. Many thanks!



### #85 [closed] ISSUE: [Feature] Context Windows
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/85  by drel-ai 2026-09-20T21:39:34Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

I played around with the new test-pr83 build. 

For the new refine pass, can we have some kind of "MiniMax H3 Context Windows" that split the frames into blocks? e.g https://github.com/ckinpdx/ComfyUI-MMH3Tools . 

### Problem or limitation

this is useful for people with not that much vram. thx in advance. 

### Proposed solution

_No response_

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-09-21T05:32:45Z):**

Thanks for the suggestion.

I had a look at the Context Windows implementation in MMH3Tools, and this is actually quite interesting for the new refine pass.

The important part is that it does not simply generate independent chunks. It keeps one full latent at the same noise level and samples overlapping temporal windows, then fuses the results. That makes much more sense for continuity, especially for a low-denoise refine/upscale pass.

It could potentially reduce the peak attention/compute VRAM required by the H3 model during a high-resolution refine, although the full latent/noise still has to exist.

For now I would prefer to keep this separate from the current PR #83 work and first stabilize the refine/PDD/upscale integration.

Since #83 is the branch that introduces the new refine pipeline, I think this would also be better handled by the author of that work if they are interested in extending it further. They already have the full context of the new sampling/refine architecture, so it would be the safest place to experiment with Context Windows without duplicating or conflicting with the current implementation.

Once the current refine path is stable, this is definitely something worth exploring.



### #84 [closed] ISSUE: MiniMax H3 Extender Node blocks mouse messages
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/84  by samsonsite1 2026-09-20T13:00:01Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

2.7.5

### ComfyUI version

0.34

### ComfyUI frontend

Not sure

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Clip-by-Clip

### Describe the issue

I use the mouse wheel on my mouse to zoom and pan around in the Comfyui interface.

When moving the mouse over the MiniMax H3 Extender Node, the bottom part of the node (where the clips are displayed) seem to block mouse messages, and the mouse wheel no longer responds.  So, if I'm zoomed into this node, I'm trapped, because I can't zoom out or pan away from the node.

Other Comfyui nodes behave normally, and don't block mouse messages, so I think something is wrong there.
Thanks.



### Steps to reproduce

1. Move mouse over over bottom half of node. Can't zoom out or pan away.

### Error message / console log

```shell

```

### Workflow / additional information

_No response_

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-09-20T13:54:45Z):**

Thanks for the report, but this is expected behavior.

When the mouse cursor is over the Extender, the mouse wheel is intentionally captured by the Extender UI so the clip cards and prompt areas can be scrolled.

This is necessary because the node contains its own scrollable interface. When the cursor is moved outside the Extender, the mouse wheel goes back to the normal ComfyUI canvas behavior.

If you zoom in so far that the Extender fills the screen and you can no longer easily move the cursor outside the node, you can still use ComfyUI’s small canvas toolbar to zoom back out and regain access to the surrounding canvas.

So this is working as intended.

Update Extender to 2.8.2 and comfyui to 0.36.0

<img width="332" height="77" alt="Image" src="https://github.com/user-attachments/assets/57ea0d36-1638-4a1b-ae01-5be755d014b1" />



### #83 [closed] PR: Optional Alibaba PDD Acc LoRA , Latent upscale + refine pass ,Seamless stitch onto a finished video
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/pull/83  by saffmila 2026-09-20T10:15:21Z

## Summary
Fork sync of our Extender work on top of current `main` (2.8.1), offering a few optional and backwards-compatible features we use daily.

The branch is fully synced with your latest `main` and merges cleanly.

### 1. Optional Alibaba PDD Acc LoRA
- Optional Acc LoRAs from `models/pdd_acc/` via [ComfyUI-MiniMax-H3-PDD-Acc](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc)
- When set: trained PDD sigma grid (`pdd_nfe` 8/4); normal `steps` / `scheduler` ignored
- `None` = unchanged sampling
- UI warns and clears Acc if FL2VA/Ref2VA file doesn’t match `generation_mode` (backend still hard-checks on Queue)

### 2. Latent upscale + refine pass
- Optional H3 latent upscaler (`models/latent_upscale_models/`)
- `run_refine`: keep draft cache → neural latent upscale → resample (`refine_denoise` / `refine_steps` / `refine_megapixels`)
- Honors `run_mode` + validated prefix like draft
- Aimed at **Ref2VA + Motion Context**

### 3. Seamless stitch onto a finished video
- Final Decode optional `original_images` (e.g. Load Video → IMAGE)
- **Disconnected:** unchanged Extender path (zero extra dependencies)
- **Connected:** soft-calls Seamless Video Stitcher (RIFE) from Comfyui-MinimaxUtils + Frame-Interpolation → `original[:cut] + bridge + AI`
- Video-only for now; Ref2VA full-batch path

### 4. Additional plumbing
- Prompt Pack Merge + Director `clear_empty` plumbing so replace packs don’t leave ghost refs (passive support code, doesn't affect standard nodes).

## Notes
- Defaults stay off / previous behavior when features are unused.
- Stitch works end-to-end, but the RIFE seam can still be slightly visible. We have not found a cleaner approach yet — this is a first usable integration, not a fully polished transition. Feedback / better seam ideas welcome.
- Short `ref_video` crops (e.g. last 20 frames) are padded to H3’s 2s @ 24 fps minimum by repeating the last frame (warning in log).
- Glad to rebase or split into 3 focused PRs (PDD / refine-upscale / stitch) if you prefer a smaller scope to review!

> **Comment by saffmila (2026-09-20T10:18:04Z):**

## Quick guide

### PDD Acc
1. Install [ComfyUI-MiniMax-H3-PDD-Acc](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc) and put Acc LoRAs in `ComfyUI/models/pdd_acc/`.
2. On the Extender, open **PDD Acc** and pick a LoRA matching the mode (**Ref2VA Acc** with Ref2VA, **FL2VA Acc** with FL2VA).
3. Set `pdd_nfe` to `8` (default) or `4` (faster). Leave LoRA/head strength at `1.0` unless you know you need otherwise.
4. When a PDD file is selected, `steps` / `scheduler` are ignored (trained sigma grid). Prefer sampler **euler**. Style/character LoRAs on clip cards still stack on top.
5. Wrong Acc vs mode → UI alert + reset to `None` (Queue would hard-fail anyway).

### Latent upscale + refine
1. Put H3 latent upscaler weights in `ComfyUI/models/latent_upscale_models/`.
2. Generate / validate **draft** as usual (Ref2VA + Motion Context).
3. Open **Latent refine** on the Extender: enable **Run refine pass**, pick upscale model, set `refine_megapixels` / `refine_denoise` / `refine_steps`.
4. Queue again — draft cache is kept; refine runs a second pass with the same `run_mode` + Validated rules as draft.
5. Final Decode: use layer `auto` / `draft` / `refine` (or the preview layer toggle) to check output. Refine does not delete draft.

### Seamless stitch (onto a finished video)
1. Optional deps only when used: [Comfyui-MinimaxUtils](https://github.com/) Seamless Video Stitcher + [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) (RIFE weights).
2. Generate the Extender continuation as usual (Ref2VA full batch path).
3. On **Final Decode**, connect the original clip as `original_images` (Load Video → IMAGE batch). Leave disconnected for normal export.
4. Typical knobs: `ref_frames_offset` ≈ overlap with the end of the original (default **20**, not 0), `rife_multiplier` (default **2**), `ai_skip_first` if the AI head is frozen.
5. Queue Final Decode → export is `original[:cut] + RIFE bridge + AI`. Video-only for now (original track silent; AI audio after the bridge).
6. Seam quality is usable but not perfect yet — RIFE transition can still be slightly visible.

### Short video refs
Last-N crops into `ref_video_*` (e.g. Get Image from Batch length 20) are OK: if under H3’s 2s @ 24 fps minimum, the Extender pads by repeating the last frame (warning in the log).


> **Comment by tritant (2026-09-20T13:07:32Z):**

Thanks a lot for the hard work — there’s quite a lot in this PR.

I’ve merged it into a dedicated test branch so I can take the time to properly test everything without touching `main`.

I’ll go through the new features and the existing workflows carefully, especially the areas around Motion Context, caching, final decode/assembly and the new refine/stitch path, then I’ll report back once I’ve had enough time to test it properly.

Really appreciate the effort you put into this.


> **Comment by tritant (2026-09-20T13:37:28Z):**

Thanks again for the work on this PR. I merged it into my test branch and went through the integration against the current 2.8.2 base.

The overall refine approach looks very interesting, especially the fact that the refined chain rebuilds the conditioning and Motion Context instead of simply upscaling the final decoded clips. Before I can consider merging this into `main`, I found a few points that I think need to be corrected:

1. **Final Decode widget order / backward compatibility**

   In 2.8.2, the end of the existing widget list is:

   `autoplay -> auto_save_project -> save_individual_clips`

   The PR currently inserts the new latent-upscale widgets between `autoplay` and the two existing widgets.

   This can break positional widget values when loading older workflows, especially with the newer ComfyUI frontend/store-backed topology.

   I think all new Final Decode widgets should be appended **after `save_individual_clips`**, without moving any existing widget position.

2. **Default Auto resolution changed from 0.40 MP to 0.70 MP**

   I found this change in both backend/frontend behavior.

   This is unrelated to the new optional features and changes the normal Extender behavior even when PDD/refine/upscale are disabled.

   I would like to keep the existing **0.40 MP** default.

3. **Final Decode latent upscale should default to `None`**

   `default_upscale_model()` currently automatically selects the bf16 model, or the first available checkpoint.

   That means simply installing the upscale weights can silently enable latent upscale in Final Decode.

   I would prefer the Final Decode `latent_upscale_model` default to always be **`None`**, so this path is explicitly enabled by the user.

   Auto-selecting a model for the refine path is less problematic because `run_refine` itself is disabled by default.

4. **PDD + Refine currently looks inconsistent**

   The draft sampling correctly uses the PDD patch/sigma grid, but the refine sampling appears to reuse the patched model while calling the regular sampler/scheduler path without the PDD sigmas.

   Also, the PDD sigmas calculated for the draft would not necessarily match `refine_denoise`.

   So for now I think we need either:
   - proper PDD handling inside the refine pass, including its own sigma calculation, or
   - explicitly prevent PDD + Refine from being enabled together until that is implemented.

5. **Refined cache/project persistence**

   The refine system creates sidecars such as:

   `*.refine.h3cache`
   `*.refine.json`

   but I could not find them being included in the `.ext` project archive.

   Since the Extender project system is intended to preserve computed work, I think the refined cache/metadata should also be saved/restored, otherwise a project reload may keep the draft cache but lose the refined chain.

On the positive side, the existing regression tests I ran against the merged branch for global drag/drop, FL2VA First/Last handling, input-link ordering and auto-resolution fallback are still passing.

So overall I like the direction of the PR a lot, but I would prefer to fix these compatibility/state points before doing the deeper GPU testing and considering it for `main`.


> **Comment by tritant (2026-09-20T14:33:41Z):**

One additional architectural point about the PDD integration:

For PDD specifically, I think I would prefer a more external / modular approach rather than having the whole PDD application happen inside the Extender.

Right now, once PDD is enabled in the Extender, the PDD patching and sigma handling happen internally, which means there is no longer any insertion point between “apply PDD” and “sample”. In practice, this makes the workflow less flexible, because it prevents users from placing other model-processing nodes in that part of the chain, such as a Model Attention Backend node.

What I would prefer is something closer to the standard ComfyUI philosophy:

- apply the PDD patch **outside** the Extender with a dedicated PDD node,
- let that node output both the patched `MODEL` and the corresponding `SIGMAS`,
- then feed the patched `MODEL` into the Extender,
- and give the Extender an **optional external sigmas input** for this case.

So the flow would be more like:

`Load Model -> PDD Apply -> optional external model nodes (attention backend, compatible patches, etc.) -> Extender`

with the PDD sigmas also passed into the Extender on a dedicated optional input.

From my point of view, this would have a few advantages:

1. **better workflow flexibility**  
   Users can still insert nodes such as Model Attention Backend in a clean way.

2. **less monolithic logic inside the Extender**  
   The Extender would not need to own all the PDD application logic itself.

3. **better separation of responsibilities**  
   The dedicated PDD node remains responsible for patching the model and generating the correct sigma schedule, while the Extender only consumes them.

4. **easier future maintenance**  
   If the PDD implementation evolves (different NFE modes, sigma logic, etc.), that can remain in the dedicated PDD node without duplicating more internal logic in the Extender.

So for PDD, my preference would be:
- keep the model patching external,
- add an optional external `SIGMAS` input to the Extender,
- and when that input is connected, let the Extender use those sigmas directly instead of its normal internal steps/scheduler path.

I think that would make the integration cleaner and more composable.


> **Comment by saffmila (2026-09-20T15:31:48Z):**

Thanks for taking the time to test this and for the thorough feedback! Really appreciate it.

All your points make total sense. We’re working on:
- Fixing the widget order (moving them after `save_individual_clips`)
- Reverting defaults back to upstream standards (0.40 MP and explicit `None` for latent upscale)
- Addressing the PDD + Refine conflict and looking into the external sigmas design

We'll push the updates to the branch once we have them tested.


> **Comment by saffmila (2026-09-20T21:09:41Z):**

Hi Tritant — thanks again for the review and for testing on your side.

I have addressed your feedback and pushed the updates to this branch:

1. Final Decode latent widgets moved after `save_individual_clips`
2. Megapixels default restored to `0.40`
3. Final Decode upscale default set to explicit `None`
4. Refine sidecars included in portable `.ext` save/load
5. **PDD Acc redesigned as you suggested** — no internal Extender PDD widgets anymore. Flow is now:
   `MiniMaxH3PDDAccApply` → patched `MODEL` → Extender, with optional `SIGMAS` → Extender
6. **PDD + refine** now share the same external SIGMAS (trimmed by `refine_denoise`), so draft/refine stay consistent
7. Also added migrations/coercion for older workflows whose `widgets_values` still had the removed PDD slots

I re-tested locally (including external PDD Acc + latent refine + Final Decode). Everything looks good on our side.

Whenever you’re happy with it, feel free to merge — happy to tweak anything else if needed. Thanks!


> **Comment by tritant (2026-09-21T04:47:57Z):**

Thanks — I’ve merged your latest fixes into the `test-pr83` branch.

I’ll go through the changes and test them properly when I have a bit more time.

Thanks again for the follow-up and the quick updates.


> **Comment by tritant (2026-09-21T05:20:39Z):**

Thanks — I checked the updated branch after the follow-up fixes.

The changes you mentioned are now there and the overall direction looks much better. In particular, I like the external PDD design much more: patched `MODEL` + optional external `SIGMAS` into the Extender is exactly the kind of separation I was looking for.

I found a few remaining points I would still like to fix before considering this ready for `main`:

1. **`SIGMAS` input position / socket order**

   The new optional `sigmas` input is currently inserted between `fl2va_model` and `audio_vae`.

   That changes the positional index of every existing input after `fl2va_model`.

   This is risky for older workflows, especially with the recent ComfyUI frontend/store-backed link topology issues we have already seen.

   I would strongly prefer `sigmas` to be appended **after all existing inputs**, so no previously existing socket changes index.

   Also, `normalizeDynamicReferenceInputOrder()` should preserve that final position and not treat `sigmas` as a normal static input that can be moved ahead of the dynamic reference inputs.

2. **Refine cache should identify the actual SIGMAS schedule**

   The refine manifest currently stores whether external sigmas were used, but `_refine_settings_match()` does not actually compare that flag.

   More importantly, two different external sigma grids with the same number of steps could still be treated as the same cached refine configuration.

   I think the refine cache should include:
   - whether external SIGMAS were used,
   - and ideally a small signature/hash of the actual sigma values.

   Changing from one PDD schedule to another, or from PDD to normal sampling, should invalidate the refined cache.

3. **Final Decode `VALIDATE_INPUTS` is too broad**

   The current `VALIDATE_INPUTS(..., **_kwargs)` effectively accepts every unknown constant input.

   In current ComfyUI this can bypass the normal validation path for the other widget values as well.

   I would prefer explicitly listing only the new compatibility fields that need special handling, rather than accepting arbitrary `**kwargs`.

Apart from that, the fixes look good on my side:

- 0.40 MP default restored
- Final Decode upscale default is now explicit `None`
- Final Decode widget ordering is fixed
- refine sidecars are included in portable project save/load
- external PDD + SIGMAS design looks correct
- PDD/refine now share the external sigma schedule

I also re-ran the important existing regression tests from the 2.8.2 base and the main ones are still passing.

So I think we are very close now — the main thing I really want to avoid is introducing another input-order compatibility issue.


> **Comment by tritant (2026-09-21T05:35:23Z):**

Hey, when you have a chance, could you also take a look at issue #85?

It’s about using a Context Windows / windowed sampling approach for the new refine path to reduce VRAM usage on high-resolution refines.

Since #83 introduces the whole refine/upscale pipeline, I think you’re in the best position to judge whether that can be integrated cleanly without breaking continuity, Motion Context, audio/video alignment or the current cache logic.

No rush — I’d prefer to get the current #83 work stable first — but #85 looks relevant enough that it’s probably worth keeping in mind for the next step.


> **Comment by tritant (2026-09-21T06:08:08Z):**

One more UX point about the refine workflow:

Right now, if `Run refine pass` is enabled on a brand-new project with no draft cache yet, the Extender stops and asks the user to generate the draft first with refine disabled.

I think the more intuitive behavior would be:

- if `Run refine pass = ON` and no draft exists yet:
  - generate the draft first,
  - then automatically continue into the refine pass in the same queue;

- if the draft already exists:
  - skip draft generation,
  - run only the refine pass.

That would make the option behave more naturally for new projects without changing the current workflow for users who prefer to review/validate their drafts first.

Users who want the current two-step workflow can simply leave refine OFF for the first run, validate the drafts, then enable refine afterwards.

So ideally:

`new project + refine ON` → `draft → refine`

`existing draft + refine ON` → `refine only`

I think this would make the feature much easier to understand and use.


> **Comment by saffmila (2026-09-21T11:59:16Z):**

Thanks for the follow-up! Just wanted to let you know I’ve seen this — currently at work, but I’ll go through everything and push the fixes this evening.


> **Comment by tritant (2026-09-21T12:22:35Z):**

I finally had a bit of time to actually run the current #83 branch in ComfyUI.

The backend/refine work is interesting, but there are still quite a few UI/state-management bugs and inconsistencies that make the branch feel unstable at the moment.

So far I found at least these:

1. **Final Decode / Preview buttons are clipped in Legacy**
   The `lowres / refined / SAVE PREVIEW` buttons at the bottom are visibly cut off.
   This looks like a CSS/layout issue around the toolbar height / overflow.

2. **Seamless Stitch section does not preserve its collapsed state**
   If I collapse the RIFE section, then refresh or switch tabs and come back, it is open again.

3. **Canvas Size megapixels value is not restored correctly in the custom UI**
   Example:
   - set Canvas Size megapixels to `0.50`
   - refresh or switch tabs
   - the UI shows `0.40` again

   The interesting part is that if I then enable `Run refine pass`, the UI suddenly shows the previously entered `0.50` again.

   This strongly suggests that the native ComfyUI widget value is actually being restored correctly, but the custom DOM is rebuilt with defaults and is not re-synced after widget hydration.

4. **Refine section collapsed/open state is not preserved**
   If the Refine section is open, after refresh/tab switch it comes back closed.

5. **`Run refine pass` state is not preserved correctly**
   If Refine is enabled, after refresh/tab switch it appears disabled again.

   More importantly, this also happens after saving a project with Refine enabled and loading it again: the section comes back closed and `Run refine pass` comes back disabled.

   This one is more serious than a visual issue because the stale custom UI can potentially overwrite the correctly restored native widget state later.

I suspect these are not isolated bugs, but symptoms of the same underlying issue: the new custom UI is maintaining its own state separately from the native ComfyUI widgets and is not always re-synchronizing correctly after workflow restoration, tab changes, DOM rebuilds or project load.

My preferred fix would be to make the native ComfyUI widgets the single source of truth:

- user interaction updates the native widget immediately
- workflow/project restore restores the native widget values
- after any rebuild / `onConfigure` / tab restore / project load, the custom DOM should be rebuilt from the native widget values
- the custom DOM should never become the authoritative source again at Queue time
- section open/collapsed states should be stored separately as node UI properties and restored explicitly

So ideally the flow should always be:

`restore native widgets -> sync custom UI from native widgets -> render`

rather than keeping independent UI values that can drift apart.

I think it would be worth reviewing all the new bound controls with this in mind, not only the ones above, because there may be more fields affected by the same synchronization problem.

I’ll keep testing and report anything else I find.


> **Comment by saffmila (2026-09-21T13:02:23Z):**

Thanks a lot for the thorough test and pinpointing the root cause! 

The "native widgets as the single source of truth" approach makes complete sense. We're currently keeping custom DOM state separate, which clearly breaks hydration on reload/tab switch.

I'm at work right now, but this evening we’ll rework the UI lifecycle (proper syncing from native widgets on configure/render, storing collapsed states in `node.properties`, and fixing the button overflow CSS).

Will keep you posted!


> **Comment by tritant (2026-09-21T13:14:17Z):**

Okay, one thing is certain: the refine function is very effective. I just tested a draft at 0.2 MP, followed by a refine at 1.2 MP—it's wonderful.


> **Comment by saffmila (2026-09-21T18:55:54Z):**

Thanks again for the thorough review and architectural suggestions! I've addressed all the feedback and pushed the updates:

### Fixes & Changes
- **Single Source of Truth UI Sync:** Custom DOM now strictly hydrates from native ComfyUI widgets on load, tab switch, and project restore (fixing megapixels, `run_refine`, and other controls drifting).

- **Persistent Section State:** Refine and Seamless Stitch collapse/expand states are now stored in `node.properties` and persist properly across sessions.

- **Legacy Decode Toolbar Clipping:** Fixed CSS layout/padding for the bottom toolbar (lowres / refined / SAVE PREVIEW buttons are now fully visible; requires a hard refresh `Ctrl+F5` to clear cached CSS).

- **Socket Ordering:** Moved the optional `SIGMAS` input to the very end of Extender sockets and preserved it at the end of `normalizeDynamicReferenceInputOrder()`.

- **Refine Logic & Cache:** 
  - Refine cache now tracks `external_sigmas` via a schedule hash.
  - Turning Refine ON without a draft now automatically runs draft → refine in a single pass; existing draft triggers refine-only as expected.
  
- **Validation:** Cleaned up `Final Decode` `VALIDATE_INPUTS` to only declare the new compatibility fields without `**kwargs`.

- **Sync:** Rebased/merged latest upstream `main` (2.8.4).

Everything is tested and merging cleanly. Let me know how it looks on your end!


> **Comment by tritant (2026-09-21T21:46:44Z):**

Thanks — I’ve merged your latest changes into the `test-pr83` branch.

I’ll test the updated UI/state handling and refine workflow properly as soon as I get some time.

Thanks again for the follow-up and for addressing all the points.


> **Comment by tritant (2026-09-21T22:40:39Z):**

After testing and thinking more about the actual workflow, I think I pushed you in the wrong direction with the automatic `draft → refine` behavior — sorry about that.

The more I use the refine system, the clearer it is that the intended workflow should stay split into two distinct phases:

1. **Draft phase**
   - generate the project normally
   - review/rerun clips
   - adjust prompts/refs
   - validate the draft chain

2. **Refine phase**
   - once the draft project is good, enable Refine
   - refine the existing draft clips only

So I think the cleaner behavior is actually the original one:

- `Run refine pass = OFF` → normal draft generation
- `Run refine pass = ON` → refine existing drafts only
- if no draft exists yet → stop with a clear message asking the user to generate the draft first

The automatic draft generation I suggested adds unnecessary complexity and creates awkward edge cases with `clip_by_clip`, partial chains and interrupted batches.

The main strength of Refine is precisely that we can work very quickly at low resolution, validate the project, and only then spend the time on the high-resolution pass.

So I think it would be better to revert that part and keep the Draft and Refine phases clearly separated.

Sorry for steering you toward the more complicated behavior there.


> **Comment by tritant (2026-09-22T05:10:27Z):**

Given the number of issues and regressions we found during testing, we preferred to go ahead and fix this branch ourselves rather than keep sending individual reports back and forth.

I kept the overall PR83 architecture intact — especially the Refine pipeline and external PDD/SIGMAS approach — and focused on restoring the stable behavior around it.

The main fixes are:

- restored the original stable seed handling across queued runs
- fixed Draft/Refine validation so it is tied to real cache state and persists correctly
- removed automatic Draft generation from Refine mode
- restored the intended workflow: generate/validate Draft first, then run Refine
- fixed custom UI state synchronization and workflow persistence
- removed stale DOM state being pushed back into native widgets at Queue time
- improved persistence of the new Refine / Final Decode / Stitch settings
- fixed Refine/Stitch collapse state handling
- fixed the Motion OFF Stitch case so it no longer silently ignores the input
- updated the regression tests and added coverage for Refine validation

The existing PR83 Refine sampling logic, Motion Context refine chain, PDD integration and SIGMAS handling were kept unchanged.

All the main stable regression tests are passing again on the corrected branch.

I’ll now continue testing this corrected version in real Legacy and Node 2.0 workflows before considering it ready for `main`.

See https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/tree/test-pr83


> **Comment by saffmila (2026-09-22T15:02:45Z):**

Thank you so much for taking the time to dive in and stabilize the branch directly — that saved a lot of back-and-forth, and I really appreciate the hands-on effort!

The clean separation between Draft and Refine makes total sense. Keeping them strictly split avoids those messy edge cases with seeds, queuing, and partial batches, and it fits the real-world workflow much better (nailing down the timing and motion at low res before committing to the heavy upscale pass).

I'm really glad the core Refine pipeline, Motion Context chain, and PDD/SIGMAS handling held up well. Looking forward to seeing it land in `main` once your testing wraps up. Thanks again for the great collaboration!


> **Comment by tritant (2026-09-22T17:31:32Z):**

I’m not a fan of latent upscaling, especially after testing your implementation more thoroughly. While the results can be spectacular, there is one major drawback: even with low denoising, the second pass reinterprets the image too heavily. This effect compounds from clip to clip, which makes it unsuitable for videos where maximum visual consistency between shots is required.

That is why, at the risk of disappointing you, this branch will not be merged into the main build. I intend to stick to my original plan and use a standard image upscaler instead, while still incorporating some of your ideas regarding the interface and related improvements.

I created the test branch because I know quite a few users like the latent upscaler, so it gives them access to an alternative version, and also because the amount of work you put into it deserves to be seen and used.

Personally, I won’t be working on that branch anymore.



> **Comment by saffmila (2026-09-22T19:52:12Z):**

Thanks for the honest verdict and for taking the time to test it so thoroughly!

Regarding the drift you observed: jumping from 0.2 MP to 1.2 MP is a 6x resolution increase, so the latent model is forced to hallucinate a massive amount of new detail from very few guide pixels, which naturally leads to compounding drift across clips. In my daily workflow, I always draft at 0.4–0.6 MP minimum. At that resolution, geometry, features, and textures are already firmly anchored, so the refine pass acts purely as a sharpener/cleanup rather than inventing new details.

That said, I completely respect your design philosophy for `main`. For zero-risk, strictly deterministic continuity across long sequences, standard pixel upscaling is definitely the safer baseline for a general release.

I really appreciate that you kept the `test-pr83` branch open for community members who prefer the diffusion/latent route, and huge thanks for fixing up the UI synchronization, seeds, and persistence along the way. 

Thanks again for the great collaboration and for maintaining such a cool project!


> **Comment by tritant (2026-09-22T20:54:38Z):**

Yes, but even with your workflow, try animating a branded product with text—it's a disaster...



### #82 [closed] ISSUE: [Bug] After the task is interrupted and stops running, it is impossible to re-select the clips that were successfully generated previously.
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/82  by lsp798732443-del 2026-09-20T09:53:19Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

2.8.1

### ComfyUI version

0.36.0

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Clip-by-Clip

### Describe the issue

After executing the run command in ComfyUI, I was not satisfied with the video preview and thus terminated the run. Then, when I tried to re-select the previously generated video clips, I found that I couldn't check them anymore. Furthermore, there are no error messages or warning log entries.

And there is another issue. Could we add an option to remember the clips that have already been selected last time? Or, which clip did the memory last run to? Because every time there is a interruption, one has to check each item one by one again. When the quantity becomes large, it becomes a bit troublesome.

### Steps to reproduce

1. First, generate a video clip and check the verification option.
2. Proceed with the generation of the next video clip and then interrupt the running process.

### Error message / console log

```shell

```

### Workflow / additional information

_No response_

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-09-20T11:42:38Z):**

Thanks for the detailed report.

We were able to identify a reproducible case that could trigger this validation/cache-state issue, and a fix has been included in **v2.8.2**.

Please update to **v2.8.2** and retest the same workflow on your side.

If you can still reproduce the issue after updating, let me know with the exact steps.


> **Comment by lsp798732443-del (2026-09-21T09:28:18Z):**

> Thanks for the detailed report.
> 
> We were able to identify a reproducible case that could trigger this validation/cache-state issue, and a fix has been included in **v2.8.2**.
> 
> Please update to **v2.8.2** and retest the same workflow on your side.
> 
> If you can still reproduce the issue after updating, let me know with the exact steps.

There are still issues. Now it's not a guaranteed occurrence but rather a problem that may occasionally arise. And I have already updated to version 2.8.2.



### #81 [closed] ISSUE: Reloading a workflow moves the ref_pack / prompt_pack cables onto `height` / `ref_image_size` (ComfyUI frontend 1.5x)
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/81  by zhukovsdo-star 2026-09-19T18:01:55Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

2.8.0

### ComfyUI version

0.36.0 (v0.36.0-15-g3c80da7f8), comfyui-frontend-package 1.53.6

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Full Batch

### Describe the issue

**Node version:** 2.7.x – 2.8.0 (every version since `ad8f134` "fix input order", 2026-08-21)
**ComfyUI:** 0.36.0, comfyui-frontend-package 1.53.6 (any frontend build with the store-backed link topology)
**OS / GPU:** Windows 11 portable build, RTX 5090

## Symptom

A workflow that uses `MiniMaxH3ReferencePackBridge` -> `ref_pack` and `MiniMaxH3PromptPackBridge` -> `prompt_pack` works while it is being built, but after saving and re-opening it every Queue fails with:

```
Failed to validate prompt for output 2:
* MiniMaxH3Extender 1:
  - Return type mismatch between linked nodes: ref_image_size, received_type(H3_PROMPT_PACK) mismatch input_type(['match', 'max'])
  - Return type mismatch between linked nodes: height, received_type(H3_REF_PACK) mismatch input_type(INT)
```

The same happens to a `GetImageSize -> width/height` wiring (cables end up on `sampler_name` / `scheduler`), and to the pack cables whenever a video/audio reference socket is added after the packs were connected.

## Cause

`normalizeDynamicReferenceInputOrder()` in `web/extender.js` reorders `node.inputs` with `splice()` and then tries to re-point the cables:

```js
node.inputs.splice(0, node.inputs.length, ...desired);
for (let slot = 0; slot < node.inputs.length; slot++) {
    const input = node.inputs[slot];
    if (!inputConnected(input)) continue;          // reads input.link
    const link = graphLinkById(node.graph, input.link);
    if (link && String(link.target_id) === String(node.id)) link.target_slot = slot;
}
```

On current frontends `input.link` is no longer a stored field: it is a deprecated read-only view resolved **from the link store by the input's current index** (`NodeInputSlot.link -> inputLink(graph, node.id, indexOf(slot))`). After the splice the store still owns the old numeric slots, so the loop "sees" the `ref_pack` cable on whatever input now sits at that index (`height`), keeps it there, and finds `ref_pack` itself unconnected.

Load sequence for a saved workflow: fresh node = 20 sockets then 16 widget inputs -> `ComfyNode.configure()` merges by name -> `realignInputLinkSlots()` puts the cables on `ref_pack`(18)/`prompt_pack`(19) -> the deferred dynamic AV sync removes 11 unconnected sockets -> cables are at slots 7/8 -> the reorder moves the widget rows up, so slots 7/8 are now `height`/`ref_image_size` -> the prompt is built from the store -> backend error above.

## Fix

Capture the cables **before** the splice through the public slot API (`node.getInputLink(slot)`, available on both legacy LiteGraph and the store-backed frontend), then re-point them in two phases (park on free out-of-range slots, then place) so the store's `occupied-target` validation can never reject a move onto a slot still owned by another moving cable. The patch below applies cleanly to every version since `ad8f134`; tested against a simulation of both the legacy and the store-backed frontend.

```diff
--- a/web/extender.js
+++ b/web/extender.js
@@ -545,15 +545,58 @@
         && desired.every((input, slot) => node.inputs[slot] === input);
     if (alreadyOrdered) return false;
 
+    // Capture every cable BEFORE the reorder, keyed by the input object.
+    //
+    // ComfyUI frontend >= 1.5x keeps link topology in a link store keyed by
+    // (node id, slot index); `input.link` is only a read-only view derived from
+    // the input's CURRENT index. Reading it after the splice would therefore
+    // report whatever cable the store still has at that numeric slot — e.g. the
+    // `ref_pack` cable showing up on `height` once the widget rows moved up —
+    // and the loop below would happily "keep" that wrong placement. That is
+    // exactly what produced the backend error
+    //   "ref_image_size, received_type(H3_PROMPT_PACK) mismatch input_type(['match','max'])"
+    //   "height, received_type(H3_REF_PACK) mismatch input_type(INT)"
+    // when a saved workflow with connected pack sockets was reloaded.
+    // node.getInputLink(slot) is the public slot API on both legacy LiteGraph
+    // and the store-backed frontend, so it is used first; graphLinkById() stays
+    // as the fallback for very old builds.
+    const cableByInput = new Map();
+    const hasSlotApi = typeof node.getInputLink === "function";
+    for (let slot = 0; slot < node.inputs.length; slot++) {
+        const input = node.inputs[slot];
+        let link = null;
+        if (hasSlotApi) {
+            try {
+                link = node.getInputLink(slot) || null;
+            } catch (_) {
+                link = null;
+            }
+        } else {
+            link = graphLinkById(node.graph, input?.link);
+        }
+        if (link && String(link.target_id) === String(node.id)) cableByInput.set(input, link);
+    }
+
     node.inputs.splice(0, node.inputs.length, ...desired);
 
-    // LiteGraph stores target sockets as numeric indices. Re-point every linked
-    // input after the visual reorder so existing workflows keep all cables.
+    // Re-point every cable to its input's new slot. The moves are applied in two
+    // phases (park on free out-of-range slots, then place) because the link
+    // store validates endpoint updates one by one and rejects a move onto a slot
+    // that is still owned by another cable which has not moved yet
+    // ("occupied-target"); that cable would then silently stay on the wrong
+    // input. Legacy LiteGraph treats target_slot as a plain number, where the
+    // two-phase write is harmless.
+    const moves = [];
     for (let slot = 0; slot < node.inputs.length; slot++) {
-        const input = node.inputs[slot];
-        if (!inputConnected(input)) continue;
-        const link = graphLinkById(node.graph, input.link);
-        if (link && String(link.target_id) === String(node.id)) {
+        const link = cableByInput.get(node.inputs[slot]);
+        if (link && Number(link.target_slot) !== slot) moves.push({ link, slot });
+    }
+    if (moves.length) {
+        const parkBase = node.inputs.length + 1024;
+        moves.forEach(({ link }, index) => {
+            link.target_slot = parkBase + index;
+        });
+        for (const { link, slot } of moves) {
             link.target_slot = slot;
         }
     }
```


### Steps to reproduce

1. Open the repo's own example Workflow/MiniMax_Extender_external_refs_and_external_prompts.json on ComfyUI frontend 1.53.6 (ReferencePackBridge -> ref_pack and PromptPackBridge -> prompt_pack are connected there).
2. Save the workflow.
3. Restart ComfyUI (or just reload the browser page) and open the saved JSON again.
4. Queue -> "Return type mismatch" validation error; the two pack cables are now attached to the height / ref_image_size widget rows of the Extender instead of ref_pack / prompt_pack.

### Error message / console log

```shell
got prompt
Failed to validate prompt for output 2:
* MiniMaxH3Extender 1:
  - Return type mismatch between linked nodes: ref_image_size, received_type(H3_PROMPT_PACK) mismatch input_type(['match', 'max'])
  - Return type mismatch between linked nodes: height, received_type(H3_REF_PACK) mismatch input_type(INT)
Output will be ignored
invalid prompt: {'type': 'prompt_outputs_failed_validation', 'message': 'Prompt outputs failed validation', 'details': '', 'extra_info': {}}
```

### Workflow / additional information

Reproduces with the repo's own example Workflow/MiniMax_Extender_external_refs_and_external_prompts.json (both pack bridges are connected there): open it on frontend 1.53.6, save, restart ComfyUI (or just reload the page), open the saved JSON again, Queue -> the validation error above. After the reload the cables are visibly attached to the height / ref_image_size widget rows of the Extender. Same thing with GetImageSize -> width/height (cables land on sampler_name / scheduler). A patch for web/extender.js is included in the description.

### System information

OS: Windows 11, ComfyUI portable build
GPU: NVIDIA GeForce RTX 5090 32 GB, driver 32.0.16.1692 (2026-09-04)
RAM: 192 GB
Python: 3.13.12 (python_embeded)
PyTorch: 2.11.0+cu130
ComfyUI: 0.36.0 (v0.36.0-15-g3c80da7f8), comfyui-frontend-package 1.53.6, comfy-aimdo 0.5.5, comfy-kitchen 0.2.35
Launch flags: --windows-standalone-build --use-sage-attention --cache-ram 60

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-09-19T19:48:15Z):**

Thanks for the detailed report and investigation.

At the moment, with the current stable ComfyUI release:

- ComfyUI: `0.36.0`
- comfyui-frontend-package: `1.52.7`

I cannot reproduce the issue. The `ref_pack` / `prompt_pack` connections remain on the correct inputs after save, browser reload, and reopening the workflow.

Your environment is using a newer post-release ComfyUI build and frontend `1.53.6`, so the behavior currently appears specific to that newer development/pre-release frontend.

For now, I prefer not to change the input/link handling based on behavior that is not reproducible on the current stable release, especially since this part of the code needs to remain compatible with both Legacy and Nodes 2.0.

When the ComfyUI/frontend versions you are using become stable, if the issue is still present, we can revisit it and adapt the Extender accordingly.


> **Comment by tritant (2026-09-20T04:18:52Z):**

reproducted, try to fix


> **Comment by tritant (2026-09-20T04:55:57Z):**

Thanks again for the detailed report.

This should now be fixed in **v2.8.1**.

The dynamic input reordering logic has been changed so existing connections are preserved correctly when the Extender rebuilds/reorders its inputs.

This should prevent `ref_pack` / `prompt_pack` cables, as well as other connected inputs such as `width` / `height`, from being moved onto the wrong sockets after a refresh, restart, tab switch, or other frontend rebuild.

Please update to **v2.8.1** and let me know if you can still reproduce the issue.



### #80 [closed] ISSUE: [Bug] Validation checkbox loses physical cache after switching browser window/tab
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/80  by sfarandola 2026-09-19T13:53:06Z

### Existing issues

- [x] I have searched the existing open and closed issues and did not find the same problem.

### Latest version

- [x] I have tested this issue with the latest available version of MiniMax H3 Extender.

### MiniMax H3 Extender version

2.7.13+ (commit 6ff0f8f)

### ComfyUI version

0.36.0

### ComfyUI frontend

Legacy

### Generation mode

Ref2VA - Motion Context ON

### Run mode

Clip-by-Clip

### Describe the issue

Since updating to the latest main (commit 6ff0f8f) from 2.7.13, the Validated checkbox stops working after switching browser window or tab.

Steps:
1. Generate clip 1 successfully - the video plays correctly in the preview.
2. Tick Validated on clip 1: it works.
3. Switch to another browser window/tab, then return to the ComfyUI tab.
4. Validation is reset, and re-ticking Validated no longer works. The status bar shows: "Clip X cannot be marked Validated because its cache does not exist yet."

The cache files are still on disk and the generated video still plays in the preview, but after the window switch the node reports the physical cache as missing. clipHasPhysicalCache appears to fail to re-detect the existing cache when the node is rebuilt on tab focus.

On 2.7.13 this did not happen: after a validation reset, re-ticking the checkbox worked fine.

This breaks the workflow of generating clips while working in another window in parallel, because every window switch permanently invalidates the clips with no way to re-validate.

### Steps to reproduce

1. Generate clip 1 successfully (the video plays correctly in the preview player).
2. Tick the Validated checkbox on clip 1 - it works.
3. Switch to another browser window or tab, then return to the ComfyUI tab.
4. Validation is reset, and re-ticking Validated no longer works: the status bar shows "cache does not exist yet". The cache files are still on disk and the video still plays in the preview.

### Error message / console log

```shell
Clip 1 cannot be marked Validated because its cache does not exist yet.
```

### Workflow / additional information

_No response_

### System information

_No response_

### Final confirmation

- [x] I have provided enough information to reproduce or investigate the issue.

> **Comment by tritant (2026-09-19T15:17:57Z):**

Thanks for the detailed report. I tried reproducing the exact sequence locally, but so far I can't reproduce the issue.

Could you please provide a few additional details?

- Browser name and version.
- Whether you switch to another tab in the same browser, another browser window, or another application.
- Whether the ComfyUI tab appears to reload/redraw when you return to it.
- Whether it happens every time or only after the tab has been in the background for some time.
- Whether you have more than one Extender tab open.
- The workflow JSON, if possible.
- The browser console output immediately after returning to the ComfyUI tab and before trying to tick `Validated` again.

If possible, could you also try a clean reinstall of MiniMax H3 Extender itself?

Please completely remove the existing `ComfyUI_MiniMax_H3_Extender` folder, then install/clone it again from the current main branch and restart ComfyUI.

This would help rule out stale frontend files, leftover files from an older version, or an incomplete update.

For now I don't want to patch the cache/validation path blindly until I can identify exactly which state is being lost.


> **Comment by drel-ai (2026-09-19T23:28:22Z):**

I do have the same problem, settings are gone after switching between workflows in the same browser tab (firefox) or after a run was cancelled. My workaround is to save the project after each run. when the checkboxes are not clickable mostly because I cancelled a run, I reload the project and than I can click the box again.   


> **Comment by tritant (2026-09-20T02:38:06Z):**

> I do have the same problem, settings are gone after switching between workflows in the same browser tab (firefox) or after a run was cancelled. My workaround is to save the project after each run. when the checkboxes are not clickable mostly because I cancelled a run, I reload the project and than I can click the box again.

Please completely remove the existing ComfyUI_MiniMax_H3_Extender folder, then install/clone it again from the current main branch and restart ComfyUI.


> **Comment by tritant (2026-09-20T03:16:44Z):**

Can you give me the information that appears in the console when comfyui starts?
example:
[INFO] Checkpoint files will always be loaded safely.
[INFO] Total VRAM 16380 MB, total RAM 65303 MB
[INFO] pytorch version: 2.10.0+cu130
[INFO] xformers version: 0.0.34
[INFO] Set vram state to: NORMAL_VRAM
[INFO] Device: cuda:0 NVIDIA GeForce RTX 4060 Ti: cudaMallocAsync
[INFO] Using async weight offloading with 2 streams
[INFO] Enabled pinned memory 26121.0
[INFO] working around nvidia conv3d memory bug.
[INFO] Using wise attention
[INFO] aimdo: src-win/cuda-detour.c:38:INFO:aimdo_setup_hooks: installing 6 hooks
[INFO] aimdo: src/control.c:269:INFO:comfy-aimdo NVML pressure enabled
[INFO] aimdo: src-win/shmem-detect.c:80:INFO:comfy-aimdo WDDM adapter match: NVIDIA GeForce RTX 4060 Ti runtime_luid=00000000:00009b5f dxgi_luid=00000000:00009b5f
[INFO] aimdo: src/control.c:284:INFO:comfy-aimdo initiated for GPU: NVIDIA GeForce RTX 4060 Ti (VRAM: 16379 MB)
[INFO] DynamicVRAM support detected and enabled
[INFO] Python version: 3.12.9 (tags/v3.12.9:fdb8142, Feb 4 2025, 3:27:58 PM) [MSC v.1942 64 bit (AMD64)]
[INFO] ComfyUI version: 0.36.0
[INFO] comfy-aimdo version: 0.5.3
[INFO] comfy-kitchen version: 0.2.34
[INFO] comfyui-frontend-package version: 1.52.7
[INFO] comfyui-workflow-templates version: 0.11.62
[INFO] comfyui-embedded-docs version: 0.5.11
[INFO] comfy-kitchen version: 0.2.34
[INFO] comfy-aimdo version: 0.5.3


> **Comment by tritant (2026-09-20T11:34:28Z):**

We were able to identify a reproducible case that could trigger this validation/cache-state issue, and a fix has been included in **v2.8.2**.

Please update to **v2.8.2** and retest the same workflow on your side.

If you can still reproduce the issue after updating, let me know with the exact steps.


> **Comment by sfarandola (2026-09-20T13:10:08Z):**

Thank you very much for the quick fix. I've updated to v2.8.2 and the validation now holds correctly after switching browser tab - the problem is gone. Really appreciate your fast and thorough work on this, and thanks for the great node.

Best regards.



### #77 [closed] ISSUE: [Feature]
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/77  by alexlml1 2026-09-17T17:01:29Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

Hi, would it be possible to add support for H3 RefMods like the ones used by ComfyUI-H3RefModPicker? It would be really useful for applying character.safetensors RefMods with the H3 Extender. Thanks!

### Problem or limitation

ㅤ

### Proposed solution

 

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-09-17T18:12:24Z):**

Thanks for the suggestion.

There is already an existing issue discussing H3 RefMod support.

At the moment, I don’t think integration is worth the risk yet:

the regular/full RefMod mode mainly saves the pre-encoded reference and is useful for building a reusable reference library, but it does not provide a meaningful advantage over the Extender’s existing native reference system
the compressed/training RefMod mode is more interesting because it can reduce reference token cost, but it is still quite experimental and can introduce fidelity/identity trade-offs

Given how sensitive the Extender is to reference ordering, Motion Context, cache invalidation and project save/load, adding another reference path right now would introduce a non-trivial regression risk for relatively limited benefit.

So for now I prefer to keep the native H3 reference pipeline stable and revisit RefMod support later if the compressed mode becomes more mature and predictable.



### #76 [closed] ISSUE: Add secondary sampling
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/76  by chenfazhan8 2026-09-16T06:44:04Z

### Existing requests

- [x] I have searched the existing open and closed issues and did not find the same request.

### Feature description

In my tests, secondary sampling can greatly reduce generation time.
Using shallow space for first-stage sampling, then secondary sampling can get much better clarity under the same VRAM usage and same generation time.
Hope to add this feature, keep the workflow: finish first sampling then run secondary sampling. Thanks.

### Problem or limitation

Produce higher-definition video with shorter generation time and lower VRAM consumption.

### Proposed solution

_No response_

### Example use case

_No response_

### Additional context

_No response_

### Final confirmation

- [x] This request is about MiniMax H3 Extender and is not a bug report.

> **Comment by tritant (2026-09-16T07:15:58Z):**

How does it differ from latent upscaling?
From my own testing, using two sampling steps—even with a very low denoising strength—is the best way to lose the original reference, especially any text that might be present.


> **Comment by chenfazhan8 (2026-09-16T07:47:25Z):**

Sorry, it may be a translation issue. What I mean is latent upscaling. You're absolutely right. When using latent upscaling with the same prompts, there may indeed be degradation in text quality for text that is not specified in the prompt. However, the quality of text that is included in the prompt will be better (my test conditions: denoising strength 0.2, 3 steps).


> **Comment by tritant (2026-09-16T07:58:35Z):**

Yes, but you won't manage to put them back in the right places. For a brand-name product, that is unacceptable.




### #75 [closed] ISSUE: Feature: Add a combined A/V latent for full batch on Extender node
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/75  by alexey-ivai 2026-09-15T20:59:15Z

Would be great to have access to the H3 latent for further upscaling to avoid VAE encode/decode step

> **Comment by tritant (2026-09-15T21:29:05Z):**

Please read the other issues before opening a new one.
see 
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/74



### #74 [closed] ISSUE: [Feature Request] : The ability of making a final pass of the whole clip sequence with Minimax H3 Latent Upscaling
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/74  by eljuani 2026-09-15T18:34:33Z

This is an AMAZING node. Very useful for working with clips. Projects functionality is perfect! But the inclusion of the upscaling would take it to next level, as you could first do a pass at, say, 0.2 mp, for the purpose of working fast and getting a rough cut, and once you're ok you can do a second pass of latent upscaling, getting more details and resolution.

> **Comment by tritant (2026-09-15T19:16:16Z):**

See
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/67
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/57
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/40
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/7


> **Comment by eljuani (2026-09-15T19:56:16Z):**

What I mean is being able to include the upscaler and regenerate from the start all clips, thus including the seaming and processes where the previous clip is needed. 


> **Comment by tritant (2026-09-15T20:19:16Z):**

You haven't read the issues; the latent upscaler won't be integrated, and there won't be a latent output—which wouldn't reflect the final seams anyway. If you want to upscale, use the video output with Ultimate Upscale, SeedVR2, or similar tools.



### #73 [closed] ISSUE: Preview restore fails when ComfyUI temp is a symlink to another drive (WinError 17)
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/73  by biomehanikamasashi-ai 2026-09-15T08:44:12Z

H3 restore preview on load failed: [WinError 17] appears on startup when ComfyUI's temp directory is symlinked to a different drive. os.replace() cannot move files across volumes on Windows. motion_context_disk.py line ~4068 os.replace(temp_preview, preview_path) moves from the node's cache dir to temp, so it fails. shutil.move() handles both cases. Workaround: junction the cache dir onto the same drive as temp.

> **Comment by tritant (2026-09-15T11:14:49Z):**

Thanks for the report — your diagnosis was correct.

The preview restore path was using:

os.replace(temp_preview, preview_path)

which fails on Windows with WinError 17 when the H3 cache and ComfyUI temp directory are on different drives/volumes.

This has been fixed in v2.7.11 by using shutil.move() for that restore step instead.

Thanks for tracking down the exact cause.

Let me know if you are still having problems.



### #72 [closed] ISSUE: Scrolling glitches
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/72  by smdisposable 2026-09-14T19:53:10Z

First: this node pack is awesome. The only one that provides a compact workflow for multi-clip extension of Rev2VA with automatic video and audio seam glitch mitigations. My previous manual workflow where I had to manage the .latent files myself and use a video editor and an audio editor to do brightness/color matching and de-clicking is greatly streamlined by this, thanks!

The main issue I keep coming up against with the main "MiniMax H3 Extender" node is that scroll resets constantly. If I'm typing into a prompt and need to click outside the node to look at something else, or if a running workflow ends, when I come back to the node the prompt textbox is always scrolled back to the top and I have to scroll back to where I was working. Certain actions also cause the horizontal clip list to reset to the beginning, so if I'm working on e.g.. Clip 7, I will sometimes find the list has scrolled back to bring Clip 1 into view, and if I don't notice this happening, I might accidentally start typing into the prompt textbox for Clip 3 or 4 because I didn't notice that the textbox I was working on (positionally within the node window) is no longer Clip 7's textbox.

Outside of these UI glitches with the node, and some other quality-of-life nitpicks around the UI (like the inability to drag/drop references and having to load references via a file picker), this is currently my go-to node pack for doing extended video sequences, so I hope the development on it continues!

> **Comment by tritant (2026-09-14T21:01:14Z):**

Yes, interface issues aren't really my area of ​​expertise. It is very difficult to juggle between legacy mode and Node 2.0 mode, which is still very poorly documented.
I am aware of all these flaws—they bother me too—and if I can fix them, I certainly will.
Thank you for your feedback.


> **Comment by tritant (2026-09-14T21:40:56Z):**

Update to 2.7.10, i have fixed the scrolling of the prompt


> **Comment by smdisposable (2026-09-22T23:46:03Z):**

Thank you!



### #71 [closed] ISSUE: Audio cache not getting updated?
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/71  by samsonsite1 2026-09-14T18:08:59Z

Hi,
I don't think the audio cache is being updated (audio.h3cache) whenever it changes. I changed the source to use a different audio source file, and I think the node was still using the old cached file.  I could tell by just looking at the file size, which didn't change.

Is there any way to manually refresh all caches (audio,video,images), other than deleting everything in the cache folder by hand?


> **Comment by tritant (2026-09-14T18:21:21Z):**

Yes the new project button



### #70 [closed] ISSUE: Local reference invalidation failed: Invalid clip index.
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/70  by skwrls 2026-09-14T14:37:54Z

I'm using clip-by-clip mode and adding clips to my project as I go.
After generating the first 2 clips, I can't get any further.
If I do anything with any of the clips beyond the second (try to add local references, click the validate checkbox) it gives the error: Local reference invalidation failed: Invalid clip index.
I tried starting over with New Project and was able to add local references to Clip 3, but after generating the first 2 clips I ran into the same problem.

> **Comment by tritant (2026-09-14T15:51:36Z):**

Thanks for the detailed report — I was able to reproduce it.

Clips that had been added in the UI but had not been generated yet do not exist in the disk manifest. However, adding/removing a local reference or trying to validate one of those future clips was still triggering a disk invalidation request, so the backend received a clip index that did not exist yet and returned:

Local reference invalidation failed: Invalid clip index.

This has been fixed in v2.7.8.

Now:

adding/removing local references on a not-yet-generated clip no longer tries to invalidate a non-existent disk segment
validating a clip that has not been generated is handled correctly
already-generated clips keep the normal Motion Context invalidation behavior

Thanks for catching this.

Update to 2.7.8


> **Comment by skwrls (2026-09-14T16:01:51Z):**

Confirmed it's working, thanks!



### #69 [closed] ISSUE: Feature Request: Automatically save project state after each Full Batch job and after each Clip-by-Clip job
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/69  by SyntaxError1871 2026-09-13T17:56:19Z

I think this function would be useful if doing a full batch of jobs to automatically save the project state (with the name to match the name of the output file) upon completion of a full batch job, so that it's possible to go back and re-run specific clips if a specific clip came out badly by loading the project cache and switching to clip to clip mode, for example. 

I think a setting that automatically saves cache after each clip would also be useful if there can be a unique file name per saved project file to easily identify which project file is what for being able to go back and pick whichever seed is best without having to regenerate the seed again to continue with the next clip.

> **Comment by tritant (2026-09-13T23:25:17Z):**

Sorry, but I don't understand the point of this request. If you want to save a project, you just need to click the "Save Project" button.


> **Comment by SyntaxError1871 (2026-09-14T07:38:27Z):**

It's for the purposes of batching a long queue of jobs overnight, so when I review the clips afterwards if there's a particular seed that did not render as expected or desired, I could load the project file for that particular full batch and not have to re-render everything. 


> **Comment by tritant (2026-09-14T09:17:42Z):**

Thanks for clarifying — I understand the overnight queue use case now.

I would consider an optional automatic project save after each completed Full Batch job, using the same name as the output video. For selectively regenerating individual clips afterwards, I see this being useful in Ref2VA with Motion Context OFF, or in FL2VA.

With Ref2VA Motion Context ON, regenerating a clip invalidates all following clips because they depend on its motion context. You would therefore need to regenerate the rest of the sequence from that point.

For Clip-by-Clip, I don't think automatically saving a complete project after every attempt is worthwhile. Each `.ext` archive includes the project's caches, so this would repeatedly copy large amounts of data, adding disk I/O, saving time, and significant disk usage just to preserve individual variations. The manual “Save Project” button lets you keep the versions you actually want.

So an optional Full Batch autosave for Ref2VA Motion Context OFF and FL2VA is something I can consider.



> **Comment by tritant (2026-09-14T09:43:07Z):**

working on


> **Comment by tritant (2026-09-14T10:44:33Z):**

added



### #68 [closed] ISSUE: Projects bleeding into each others
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/68  by The-Anomaly-be 2026-09-13T05:42:34Z

First of all, thank you for your excellent work, I created a few videos with it and they're amazing being able to produce high quality 1 minute videos on my own hardware. I only have an issue, it's that sometimes the projects seem to bleed into each other. If I simply load new refs and change the text prompts (or if I ask my agent to do it), it may still use one of the previous reference image and I can see characters or actions from the previous projects.

It's still easily fixed, I have to delete the node and recreate it to prevent that from happening. So maybe, if it's too hard to fix, this may be added to the custom node instructions. 

Kind regards, keep up the good work, this custom node is, by far, the best when it comes to creating long videos with Minimax H3.

> **Comment by tritant (2026-09-13T07:38:46Z):**

Thank you for the detailed report and for the kind words.

I had a look at the current state handling, and there are a couple of possible causes here, but I need a bit more detail to identify which one you are hitting.

Could you please tell me exactly how you replace the references when starting a new project?

Do you use the Extender’s own Load Ref / Remove Ref controls?
Or do you feed references through a Reference Pack / external node / agent?
Do any of the clips use Local Refs / Local Videos?
When the bleed happens, are the old characters/actions appearing in all clips, or only specific clips?

The distinction matters because external Reference Pack slots that are left empty currently do not automatically clear an older internal reference slot, and Local Refs are stored per clip and can override the global refs for that clip.

So if, for example, an older project had 3 refs and the new project only provides 2 through an external pack, the third old ref may still remain active unless it is explicitly cleared.

If you can describe one exact reproduction case — old refs/prompt, new refs/prompt, whether you use local refs, and which clip shows the bleed — I should be able to pinpoint the exact state that is carrying over.

I think that in your case, it is not necessary to reinstall the entire node; deleting the contents of the cache folder should work.

Alternatively, the simplest way to fix this is probably to add a "New Project" button that resets everything to zero.


> **Comment by tritant (2026-09-13T09:50:02Z):**

I took the easy way out and added a "New Project" button.
Thanks again for the report.
Update to 2.7.5


> **Comment by The-Anomaly-be (2026-09-13T14:26:57Z):**

Thank you, sorry for the delay in my answer:
- I only used the load ref 1 from the node to replace the image (global reference only).
- The issue happens randomly, in the last project, it used the characters of a prior generation in the second clip, all other clips were fine.
- It seemed to be a cache issue, I've seen that creating a new node leads to using another cache directory: chain_extender_xx.final.video, that cache directory was empty when I started working on the new project. The problem may be that the node uses the old chain_extender_xx.h3cache, chain_extender_xx.audio.h3cache and json from a processed project even if you replace everything directly in the node (reference and texts). If your "New project" button deals with those files, I think it will be fine. I'll test it tonight and give you feedback. 

Thanks again.



### #67 [closed] ISSUE: [Feature Request] add a latent upscaler, integrated into the node
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/67  by anto262carro-tech 2026-09-12T21:26:47Z

Could a feature be added to the node that would generate an initial pass of the clip at a lower resolution, and - if the preview is deemed satisfactory - it could perform a second pass of the clip using an integrated latent upscaler to increase the final resolution?

BTW, Thank you for your wonderful work!

> **Comment by tritant (2026-09-12T21:43:57Z):**

see this issue:
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/57

I tested several methods, such as Ultimate Upscale, Seedvr2, and the latent upscaler.
The latent upscaler yields good results, but it is the least faithful to the reference images—especially if the reference contains text.
The latent upscaler will not be included in the extender.



### #66 [closed] ISSUE: Clip editing is undoing all edits when run
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/66  by guzuligo 2026-09-12T14:37:58Z

When you press RUN, and then you make edits to the clip text as the render works, you lose all your edits after the run finishes. It feels like the undo button is auto-pressing. 

This prevents you to change anything while it is running and you have to either edit external notes and paste them later, or just wait for the run to finish. But, when I am running clip 1 and want to edit clip 2 or 3, this prevents me.

> **Comment by tritant (2026-09-12T17:13:51Z):**

Thanks for the report.

This is the expected behavior and will remain unchanged.

When a run starts, ComfyUI queues a snapshot of the Extender state. Any edits made to the node while that execution is still running are outside that queued state and may be overwritten when the run completes.

The Extender is therefore intended to be treated as read-only while a run is in progress. If you want to prepare later clip prompts during generation, please use an external text editor and paste them back once the run has finished.


> **Comment by guzuligo (2026-09-13T17:22:28Z):**

Here I wish the clips could optionally be separate nodes 😅



### #65 [closed] ISSUE: Crash - shape mismatch on 2nd clip in a chain
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/65  by Crisr 2026-09-12T03:17:03Z

I have a crash occurring on the 2nd clip in a chain (first clip renders without problems). I have 2 audio references and 3 image references in my workflow. 
I vibe-fixed the crash (I had the same problem with older versions of the node) on the 2.7.4 version and made a fork at https://github.com/Crisr/ComfyUI_MiniMax_H3_Extender.git
I don't know if the crash is something that's occurring because of my setup, could be, but I get it every time I update the node so I have to patch it locally.
Here's the crash log:

[INFO] got prompt
[INFO] H3 validated hard-skip: Ref2VA clip 1 exists on disk; sampler forbidden
[INFO] H3 Disk Join: mode=clip_by_clip clip=1 validated=True stop=False clips=1 frames=226 cache=27.4 MB
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 2677MB Staged. 0 patches attached. Force pre-loaded 128 weights: 692 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 2677MB Staged. 0 patches attached. Force pre-loaded 128 weights: 692 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 2677MB Staged. 0 patches attached. Force pre-loaded 128 weights: 692 KB.
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
[INFO] MiniMax H3 Motion Context RAM: reasserted Extender payload patch over G:\comfy-cu133\ComfyUI\custom_nodes\comfyui-h3-multishot.h3_avbank_probe.extra_conds
[INFO] MiniMax H3 Motion Context RAM: 5 video frames -> 2 latent conditioning blocks, 8 audio latent steps, trim=5
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 11956MB Staged. 204 patches attached. Force pre-loaded 210 weights: 1175 KB.
  0%|                                                                | 0/6 [00:00<?, ?it/s,   Model Initializing ...  ]
[INFO] Comfy model compiler graph breaks: 0, rogues: 10
[ERROR] !!! Exception during processing !!! shape mismatch: value tensor of shape [400, 32] cannot be broadcast to indexing result of shape [416, 32]
[ERROR] Traceback (most recent call last):
  File "G:\comfy-cu133\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "G:\comfy-cu133\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 5060, in extend
    sampled = _sample_h3(
              ^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 1795, in _sample_h3
    samples = guider.sample(
              ^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\env\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\k_diffusion\sampling.py", line 1528, in sample_res_multistep
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0., cfg_pp=False)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\env\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\k_diffusion\sampling.py", line 1486, in res_multistep
    denoised = model(x, sigmas[i] * s_in, **extra_args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\model_base.py", line 208, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\model_base.py", line 252, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\env\Lib\site-packages\torch\nn\modules\module.py", line 1783, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\env\Lib\site-packages\torch\nn\modules\module.py", line 1794, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\ldm\minimax\model.py", line 577, in forward
    graph_out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\comfy-cu133\ComfyUI\comfy\ldm\minimax\model.py", line 711, in _forward
    all_audio_rows[~audio_update] = cond_audio_rows
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
RuntimeError: shape mismatch: value tensor of shape [400, 32] cannot be broadcast to indexing result of shape [416, 32]

[INFO] Prompt executed in 37.03 seconds

> **Comment by tritant (2026-09-12T07:53:09Z):**

Thanks for the report.

Your log shows comfyui-h3-multishot patching the same internal MiniMax H3 extra_conds path used by the Extender. We have already seen similar audio-row mismatches from multiple users running Multishot alongside the Extender.

Compatibility with Multishot is not currently supported, because both extensions modify the same H3 internals and their payload/layout patches can conflict.

see issue:
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/54
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/18



### #64 [closed] PR: Fix per-clip seed advancement when batch queueing
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/pull/64  by JaRail 2026-09-11T17:55:43Z

Queuing several runs before the Extender finishes serializes identical `clips_json` seeds, allowing later runs to return cached duplicates. This advances candidate seeds in the native widget's `afterQueued` hook, so a three-item batch starting at increment seed 10 submits 10, 11, 12.

Fixed and validated clips retain their seeds. The hook uses the existing seed helper, hidden-state persistence, and rendering; serialization synchronizes the active `mode_clips` entry while keeping the inactive timeline independent.

Completion merges preserve the next candidate seeds by clip ID within the result's generation mode, preventing older queued results from rewinding or double-advancing them. Backend validation remains authoritative, backend-imported clips retain the completion-time fallback, and checkpoint resume handling is preserved.

Closes #49. Supersedes #50, which remains closed.

### Validation

Base: upstream `main` at [`08e5daa`](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/commit/08e5daa371faefef314a80441bb85ace65591836) (Extender 2.7.3).

- `node tests/batch-seeds.test.mjs`: **11 passed**.
- The same suite against unmodified upstream `08e5daa`: **9 failed, 2 passed**. Increment/decrement seeds repeat across all eight submissions; randomize produces only one distinct seed.
- Coverage exercises the actual widget and execution hooks with mocked DOM/ComfyUI services: eight-item batches in Ref2VA with Motion Context ON/OFF and FL2VA, randomization and collisions, wraparound, fixed/validated clips, interleaved completions, interrupted results, backend-imported clips, repeated UI construction, replacement state after hydration, and mode switching.
- The fixture includes the current UI's required `motion_context` widget and DOM `dataset`.
- `node --input-type=module --check < web/extender.js`, `node --check tests/batch-seeds.test.mjs`, and `git diff --check origin/main`: passed.

No end-to-end GPU generation or real-browser layout test was run. The original reported reproduction environment is Windows / Firefox / ComfyUI 0.30.0 / frontend 1.49.6; the queue lifecycle was checked against frontend tag v1.49.6.


> **Comment by tritant (2026-09-11T18:57:03Z):**

ok thx, merged.




### #63 [closed] ISSUE: Clip scrolling isn't working
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/63  by guzuligo 2026-09-11T17:19:03Z

When adding a clip, usually it takes writing multiple shots. But, when I try to scroll the text to edit it, the comfy zoom in effect activates instead, which makes editing so challenging. Also, the spelling error check isn't activating.

Though, I suggest making clips as separate nodes instead of being built in.

> **Comment by tritant (2026-09-11T18:35:14Z):**

Thanks, confirmed.

This is specific to Nodes 2.0: ComfyUI captures the mouse wheel for graph zoom before the custom prompt textarea can use it. Legacy does not have this problem.

I’ve fixed it by marking only the prompt textarea as a wheel-capturing control. The mouse wheel now scrolls long prompts normally in Nodes 2.0, while wheel behavior everywhere else on the card remains unchanged.
Update to v2.7.4.



### #62 [closed] ISSUE: ref_audio time error
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/62  by lsp798732443-del 2026-09-11T03:31:26Z

Just now, I encountered a duration error when using the audio reference. I used a 5-second audio reference to generate a 10-second video, but I was only referring to the sound quality, not the actual duration. There was no problem before, but today it suddenly occurred.

# ComfyUI Error Report
## Error Details
- **Node ID:** 67
- **Node Type:** MiniMaxH3Extender
- **Exception Type:** ValueError
- **Exception Message:** ValueError: MiniMax H3 Extender: ref_audio_1 is too short for the current clip. Source duration is 5.224s but this clip needs the window 0.000s -> 10.125s.

## Stack Trace
```
  File "H:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "H:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "H:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Lora-Manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^

  File "H:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "H:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)

  File "H:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 4993, in extend
    audio_items, audio_blocks = _prepare_standalone_audio_refs(
                                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        audio_vae,
        ^^^^^^^^^^
    ...<3 lines>...
        cache=standalone_audio_cache,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^

  File "H:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 1286, in _prepare_standalone_audio_refs
    sliced = _slice_ref_audio(audio, start, duration, label, require_full=True)

  File "H:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender\extender.py", line 1212, in _slice_ref_audio
    raise ValueError(
    ...<3 lines>...
    )


> **Comment by tritant (2026-09-11T08:47:05Z):**

Thanks for the report — confirmed.

The issue came from the new standalone audio timeline behavior: because your 5.224s audio is slightly above the 5s threshold, it was treated as a timeline reference and the Extender incorrectly required it to cover the full 10.125s clip.

I’ve changed this so timeline audio can now end naturally inside a clip. In your case, the available 5.224s will simply be used as the audio reference for the 10.125s clip instead of raising an error.

Please update to 2.7.3 and retry



### #61 [closed] ISSUE: Nodes that bundle every single function together like this are a real headache.
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/61  by DarkNightMeteor 2026-09-11T00:08:50Z

How are users supposed to work with them if they can't even define the resolution themselves? Can you set it to 1920x864? How do you output 1080x1920? Nodes like this look like half-baked products ,they show a complete lack of regard for the user.

> **Comment by tritant (2026-09-11T00:20:12Z):**

So don't use it.



### #60 [closed] ISSUE: Ref Audio for two Clips Fix not working
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/60  by stephenrt42 2026-09-10T20:41:39Z

Hey, I opened this ticket again because I tried release 2.6.2 and the fix for the global audio over two clips didn't work.

Here's my prompt

```bash

subject_definitions:

<Video 1>: the immediately preceding video segment. It provides the visual, temporal, conversational, and performance state from which the target video must continue.

<Picture 1>: reference image of the same man. His identity, facial features, hairstyle, clothing, body proportions, and overall visual appearance must remain consistent with <Video 1> and <Picture 1>.

<Audio 1>: the new audio segment that continues the man's speech from the preceding video. This audio is the exact spoken content for the target video and must be reused directly.

summary:

[video continuation + audio reuse + reference generation] Continue directly from the end of <Video 1>. The same man continues speaking to the camera using <Audio 1> as the exact spoken audio for the continuation. The target video must begin from the man's existing visual and conversational state at the end of <Video 1>, with no restart or reset. <Audio 1> begins exactly where the previous spoken audio ends.

retention_analysis:

<Video 1>: fully_preserved - Preserve the man's identity, appearance, camera relationship, conversational state, and visual continuity from the end of the source video. The target begins as a direct continuation of the source video's final state.

<Picture 1>: fully_preserved - Preserve the man's facial identity, facial structure, hairstyle, clothing, body proportions, and overall appearance.

<Audio 1>: fully_copy - Use <Audio 1> directly as the target video's complete spoken audio track, preserving its exact words, timing, pauses, cadence, pronunciation, and vocal characteristics.

detailed_description:

The target video begins immediately after the final moment of <Video 1>. The same man remains in the same ongoing conversation with the camera. Continue naturally from the exact visual and conversational state established at the end of <Video 1>.

[Shot 1] The shot opens from the ending state of <Video 1>. Preserve the man's identity, clothing, hairstyle, framing, camera relationship, and natural conversational posture. The man continues speaking directly to the camera without restarting his performance.

<scenetrans>

The speech continues seamlessly from the preceding video into the target video. There is no conversational reset, repeated phrase, new introduction, greeting, or artificial beginning. <Audio 1> represents the next portion of the same continuous speech.

The man (S1) continues speaking according to <Audio 1>. His mouth, jaw, cheeks, tongue, and facial muscles precisely reproduce the articulation required by the supplied audio.

<Audio 1> is the authoritative source for the spoken dialogue. Reuse its exact spoken content and timing. Do not invent, paraphrase, translate, summarize, restart, or replace any portion of the dialogue.

The first audible sound of <Audio 1> occurs immediately as the continuation of the preceding speech. Do not repeat the final word, syllable, breath, or phrase from <Video 1>. Do not insert a new opening phrase or additional dialogue before <Audio 1>.

The man's facial expression, mouth position, eye contact, head position, and conversational behavior should naturally continue from the ending state of <Video 1>. Avoid resetting his face to a neutral starting pose.

During pauses contained within <Audio 1>, maintain natural conversational micro-movements, blinking, breathing, subtle facial expressions, and restrained head movement.

The visual performance must remain continuous with <Video 1>. The man should appear to have been speaking continuously even though the target video begins at a new segment.

<Picture 1> reinforces the man's exact identity and appearance. Do not change his face, hairstyle, clothing, apparent age, body proportions, or distinctive features.

The target video's duration exactly follows the duration of <Audio 1>. The performance begins with the first audible sound of <Audio 1> and ends naturally with its final audible sound.

overall_soundscape:

The spoken audio is provided directly by <Audio 1>. Preserve its exact timing, pauses, cadence, vocal characteristics, and continuity with the preceding speech. No additional dialogue, music, narration, sound effects, or replacement audio is introduced.

non_diegetic_music:
```

<img width="1401" height="694" alt="Image" src="https://github.com/user-attachments/assets/066679bb-88ac-47d1-9191-89a2506df7b9" />

<img width="783" height="791" alt="Image" src="https://github.com/user-attachments/assets/a5e4aff0-d330-49a8-97b9-76957863c7b3" />


> **Comment by tritant (2026-09-10T21:43:29Z):**

Version 2.6.2 has been out for quite some time (5day); it was already the current version when we spoke.
It therefore doesn't include the fix; that will arrive soon with version 2.7.1.




### #59 [closed] ISSUE: Feature Request: RefMods
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/59  by spaha12 2026-09-10T18:15:32Z

Instead of feeding a lot of ref pictures into the workflow, is it possible to add 
https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod
?

The concept is great, it's easy to use and the safetensors are really small. 

TIA and thanks for your work!

> **Comment by tritant (2026-09-10T18:55:42Z):**

The only part I find really interesting there is the reference compression.

Full RefMod doesn’t bring much to my Extender because I already have a working reference/cache system, so pre-encoding the refs alone isn’t a big enough benefit.

The compressed reference mode is more interesting because it could reduce the token/VRAM cost of using many refs. But according to the project’s own README, aggressive compression can noticeably lose identity, fine details, clothing information, motion/timing, etc., and there is no reliable universal setting yet.

So for now it looks much too experimental for me to integrate into the Extender. I’d rather keep the native H3 reference path, which is predictable and already works well, than add a compression layer that may save resources at the cost of reference fidelity.



### #58 [closed] ISSUE: Feature Request - The ability to add the h3 upscaler pass
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/58  by merecesarchviz 2026-09-10T11:19:07Z

@tritant is not possible to add the sampler latent output for we can add the minimax h3 upscaler pass?

> **Comment by tritant (2026-09-10T11:22:15Z):**

see this issue
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/57




### #57 [closed] ISSUE: [feature request] Second pass.
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/57  by logoth80 2026-09-10T07:48:04Z

I'm in love with the node. already did 3 long lip-sync music videos, but having it do latent upscale 50%-66% noise second pass would make the node truly shine.
atm going above .4MP with 14.2s parts is the safety zone, but with upscaling you could go as high as the VRAM allows.

> **Comment by tritant (2026-09-10T08:04:13Z):**

see this issue
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/40
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/7


> **Comment by logoth80 (2026-09-10T08:17:11Z):**

Ok, but just to add to what I'm suggesting exactly.
inside the node you could have only one latent saved. just for each step it would:
downscale last frames (or last segment) by 0.35->pass once->upscale to original size-> add noise->use full latent for the calculations again and do 2/3rd  of the steps. 
I have no idea how complicated it becomes, but result should be high quality.


> **Comment by tritant (2026-09-10T08:36:07Z):**

As you say, that complicates things.
The primary goal of the extender is to generate the greatest number of clips (maximum duration and resolution) in a single pass, thereby maximizing VRAM efficiency. Latent upscaling would result in additional VRAM usage, which is something I want to avoid.
Personally, I think you'll get better results by connecting the video output to, for example, a SeedVR2.


> **Comment by logoth80 (2026-09-10T08:50:41Z):**

I prefer the look of not upscaled gens tbh. but why would latent upscale cost VRAM? it's done between passes and VRAM is exactly the same as doing one high res pass directly.
sorry for harping on, I do accept that it's not something you are working on ;p


> **Comment by tritant (2026-09-10T09:02:24Z):**

You're right about the VRAM point. The latent resize itself is cheap, and if the first-pass tensors are properly released before the second pass, peak VRAM should be roughly the same as a direct generation at the final resolution. The extra cost is mainly compute time. Higher VRAM would only come from implementation details such as keeping first-pass caches/tensors alive during the second pass.

That said, there won’t be any latent upscale in the Extender anyway.


> **Comment by merecesarchviz (2026-09-10T11:38:05Z):**

we can reach the another level having the possibility to add the h3 latent upscaler pass, will be great you can add the latent output for we can add it


> **Comment by tritant (2026-09-10T12:07:29Z):**

> we can reach the another level having the possibility to add the h3 latent upscaler pass, will be great you can add the latent output for we can add it

Please read all the issues!
Latent outputs do not benefit from the latest automatic corrections applied by the decode/preview node, so the seams in your result will not match what the extender produces.



### #56 [closed] ISSUE: [Feauture Request] Improvements to more flexible usage
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/56  by Ultracelestial 2026-09-10T07:36:50Z

First of all, thanks for such an excellent node - it’s very convenient, but there are a few things that could be added.

Is it possible to add a function to use a custom sampler or modify other nodes within the extender module?
Also. i think it would be nice if boxes with prompts stacks verticaly inside extender node, that would help to edit them.
And it would be good to add global prompts boxes with desriptions of characters, locations, soundscape and other things. Right now, I have to copy everything from one prompt to another.

> **Comment by tritant (2026-09-10T08:24:54Z):**

Thanks for your suggestions.

Using a different sampler isn't an option; Extender relies on the ComfyUI API—and thus the official sampler—so that part is hard-coded.

As for vertical prompts, that’s a matter of preference; I personally prefer them horizontal. In any case, making that change would take hours of work, with no guarantee against introducing regressions into an interface I’ve only just managed to stabilize.

Regarding global prompts, I feel that would unnecessarily clutter an interface that is already quite busy.



### #55 [closed] ISSUE: Running Ref Audio over two or more clips
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/55  by stephenrt42 2026-09-09T00:55:54Z

When I add two clips and try to have the source audio file (10s of audio) used across both clips; when it gets to the 2nd clip, the audio resets from the start instead of continuing; is this normal? I don't know if its my prompt or not. Here's an example of my 2nd clip prompt.

```bash
subject_definitions:

<Video 1>: the immediately preceding video segment. It provides the visual, temporal, conversational, and performance state from which the target video must continue.

<Picture 1>: reference image of the same man. His identity, facial features, hairstyle, clothing, body proportions, and overall visual appearance must remain consistent with <Video 1> and <Picture 1>.

<Audio 1>: the new audio segment that continues the man's speech from the preceding video. This audio is the exact spoken content for the target video and must be reused directly.

summary:

[video continuation + audio reuse + reference generation] Continue directly from the end of <Video 1>. The same man continues speaking to the camera using <Audio 1> as the exact spoken audio for the continuation. The target video must begin from the man's existing visual and conversational state at the end of <Video 1>, with no restart or reset. <Audio 1> begins exactly where the previous spoken audio ends.

retention_analysis:

<Video 1>: fully_preserved - Preserve the man's identity, appearance, camera relationship, conversational state, and visual continuity from the end of the source video. The target begins as a direct continuation of the source video's final state.

<Picture 1>: fully_preserved - Preserve the man's facial identity, facial structure, hairstyle, clothing, body proportions, and overall appearance.

<Audio 1>: fully_copy - Use <Audio 1> directly as the target video's complete spoken audio track, preserving its exact words, timing, pauses, cadence, pronunciation, and vocal characteristics.

detailed_description:

The target video begins immediately after the final moment of <Video 1>. The same man remains in the same ongoing conversation with the camera. Continue naturally from the exact visual and conversational state established at the end of <Video 1>.

[Shot 1] The shot opens from the ending state of <Video 1>. Preserve the man's identity, clothing, hairstyle, framing, camera relationship, and natural conversational posture. The man continues speaking directly to the camera without restarting his performance.

<scenetrans>

The speech continues seamlessly from the preceding video into the target video. There is no conversational reset, repeated phrase, new introduction, greeting, or artificial beginning. <Audio 1> represents the next portion of the same continuous speech.

The man (S1) continues speaking according to <Audio 1>. His mouth, jaw, cheeks, tongue, and facial muscles precisely reproduce the articulation required by the supplied audio.

<Audio 1> is the authoritative source for the spoken dialogue. Reuse its exact spoken content and timing. Do not invent, paraphrase, translate, summarize, restart, or replace any portion of the dialogue.

The first audible sound of <Audio 1> occurs immediately as the continuation of the preceding speech. Do not repeat the final word, syllable, breath, or phrase from <Video 1>. Do not insert a new opening phrase or additional dialogue before <Audio 1>.

The man's facial expression, mouth position, eye contact, head position, and conversational behavior should naturally continue from the ending state of <Video 1>. Avoid resetting his face to a neutral starting pose.

During pauses contained within <Audio 1>, maintain natural conversational micro-movements, blinking, breathing, subtle facial expressions, and restrained head movement.

The visual performance must remain continuous with <Video 1>. The man should appear to have been speaking continuously even though the target video begins at a new segment.

<Picture 1> reinforces the man's exact identity and appearance. Do not change his face, hairstyle, clothing, apparent age, body proportions, or distinctive features.

The target video's duration exactly follows the duration of <Audio 1>. The performance begins with the first audible sound of <Audio 1> and ends naturally with its final audible sound.

overall_soundscape:

The spoken audio is provided directly by <Audio 1>. Preserve its exact timing, pauses, cadence, vocal characteristics, and continuity with the preceding speech. No additional dialogue, music, narration, sound effects, or replacement audio is introduced.

non_diegetic_music:
```
N/A


> **Comment by tritant (2026-09-09T04:26:12Z):**

Thanks for the report.

In ref2va the global standalone Audio reference is designed to be automatically sliced along the clip timeline.

For example, with one 10-second global audio file and two 5-second clips:

- Clip 1 should receive approximately 0–5s
- Clip 2 should receive approximately 5–10s

However, if the same audio file is added separately as a **local per-clip Audio reference**, each local reference starts from the beginning of its own file, so both clips would start at 0s.

Could you please confirm:

1. Is the audio connected once as a global Audio reference, or added separately through the local `Refs` panel?
2. What are the durations of Clip 1 and Clip 2?
3. Is the same 10-second file supposed to cover the combined duration of both clips?




> **Comment by stephenrt42 (2026-09-09T11:25:05Z):**


Is the audio connected once as a global Audio reference, or added separately through the local Refs panel?
* Yes, I use Ref_audio_1

What are the durations of Clip 1 and Clip 2?
* Each clip is 5s long

Is the same 10-second file supposed to cover the combined duration of both clips?
* Yes, single 10s audio file is suppose to extend over the two clips

Setup part - Node connections
<img width="1214" height="695" alt="Image" src="https://github.com/user-attachments/assets/eff5d572-bd66-4ad0-a752-64d5a4dc151d" />


Setup part  - Clip duration
<img width="1092" height="790" alt="Image" src="https://github.com/user-attachments/assets/d78beba0-c616-415c-be77-f86b478f5bc0" />




> **Comment by tritant (2026-09-09T11:51:21Z):**

Thanks, that clarifies it.

I checked the Extender code, and I found the issue.

There is currently a condition that only treats a global audio reference as a sequential timeline when the source audio is longer than 15 seconds.

For audio files 15 seconds or shorter, each clip currently starts reading the audio again from 0s.

So with your setup:

Ref_audio_1 = 10 seconds
Clip 1 = 5 seconds
Clip 2 = 5 seconds

the current behavior is effectively:

Clip 1 → 0–5s
Clip 2 → 0–5s again

instead of the expected:

Clip 1 → 0–5s
Clip 2 → 5–10s

So yes, this is a real bug in the global audio splitting logic.

The 15-second limit should apply to the maximum audio segment used by a single clip, not decide whether the source audio is treated as a continuous timeline across clips.

I’ll fix it for the upcoming version. You have two options: a single audio clip longer than 15 seconds, or two short audio clips used as local references.


> **Comment by stephenrt42 (2026-09-09T11:56:16Z):**

Excellent news, happy to contribute to your project.

I mentioned your repo and how good it was in a Reddit chat; its being received very well.

https://www.reddit.com/r/StableDiffusion/comments/1wanmko/comment/p8pw8nh/?context=1&screen_view_count=9&ext-referrer=DIRECT

Thank you again! 

S.


> **Comment by stephenrt42 (2026-09-10T20:39:00Z):**

Hey, I downloaded yesterday's release (2.6.2) placed it in my custom_nodes, grab a workflow from its example_workflows and ran it again; it did the exact same thing as last time.

My settings are the same; 1 audio mp3 trimmed to 10 seconds; added two clips at 5seconds each, and used this prompt for each clip:

```bash
subject_definitions:

<Video 1>: the immediately preceding video segment. It provides the visual, temporal, conversational, and performance state from which the target video must continue.

<Picture 1>: reference image of the same man. His identity, facial features, hairstyle, clothing, body proportions, and overall visual appearance must remain consistent with <Video 1> and <Picture 1>.

<Audio 1>: the new audio segment that continues the man's speech from the preceding video. This audio is the exact spoken content for the target video and must be reused directly.

summary:

[video continuation + audio reuse + reference generation] Continue directly from the end of <Video 1>. The same man continues speaking to the camera using <Audio 1> as the exact spoken audio for the continuation. The target video must begin from the man's existing visual and conversational state at the end of <Video 1>, with no restart or reset. <Audio 1> begins exactly where the previous spoken audio ends.

retention_analysis:

<Video 1>: fully_preserved - Preserve the man's identity, appearance, camera relationship, conversational state, and visual continuity from the end of the source video. The target begins as a direct continuation of the source video's final state.

<Picture 1>: fully_preserved - Preserve the man's facial identity, facial structure, hairstyle, clothing, body proportions, and overall appearance.

<Audio 1>: fully_copy - Use <Audio 1> directly as the target video's complete spoken audio track, preserving its exact words, timing, pauses, cadence, pronunciation, and vocal characteristics.

detailed_description:

The target video begins immediately after the final moment of <Video 1>. The same man remains in the same ongoing conversation with the camera. Continue naturally from the exact visual and conversational state established at the end of <Video 1>.

[Shot 1] The shot opens from the ending state of <Video 1>. Preserve the man's identity, clothing, hairstyle, framing, camera relationship, and natural conversational posture. The man continues speaking directly to the camera without restarting his performance.

<scenetrans>

The speech continues seamlessly from the preceding video into the target video. There is no conversational reset, repeated phrase, new introduction, greeting, or artificial beginning. <Audio 1> represents the next portion of the same continuous speech.

The man (S1) continues speaking according to <Audio 1>. His mouth, jaw, cheeks, tongue, and facial muscles precisely reproduce the articulation required by the supplied audio.

<Audio 1> is the authoritative source for the spoken dialogue. Reuse its exact spoken content and timing. Do not invent, paraphrase, translate, summarize, restart, or replace any portion of the dialogue.

The first audible sound of <Audio 1> occurs immediately as the continuation of the preceding speech. Do not repeat the final word, syllable, breath, or phrase from <Video 1>. Do not insert a new opening phrase or additional dialogue before <Audio 1>.

The man's facial expression, mouth position, eye contact, head position, and conversational behavior should naturally continue from the ending state of <Video 1>. Avoid resetting his face to a neutral starting pose.

During pauses contained within <Audio 1>, maintain natural conversational micro-movements, blinking, breathing, subtle facial expressions, and restrained head movement.

The visual performance must remain continuous with <Video 1>. The man should appear to have been speaking continuously even though the target video begins at a new segment.

<Picture 1> reinforces the man's exact identity and appearance. Do not change his face, hairstyle, clothing, apparent age, body proportions, or distinctive features.

The target video's duration exactly follows the duration of <Audio 1>. The performance begins with the first audible sound of <Audio 1> and ends naturally with its final audible sound.

overall_soundscape:

The spoken audio is provided directly by <Audio 1>. Preserve its exact timing, pauses, cadence, vocal characteristics, and continuity with the preceding speech. No additional dialogue, music, narration, sound effects, or replacement audio is introduced.

non_diegetic_music:
```

<img width="695" height="646" alt="Image" src="https://github.com/user-attachments/assets/55bb691f-6d87-4aae-9897-3297460403e3" />

<img width="1401" height="694" alt="Image" src="https://github.com/user-attachments/assets/166e9723-4d32-4864-bd8d-90f28e0d6f4a" />


> **Comment by tritant (2026-09-10T20:44:49Z):**

Version 2.6.2 has been out for quite some time; it was already the current version when we spoke.
It therefore doesn't include the fix; that will arrive soon with version 2.7.1.


> **Comment by stephenrt42 (2026-09-10T20:47:10Z):**

oh ok. My bad, apologies. 



### #54 [closed] ISSUE: Subject: Shape mismatch crash from Clip 2 onward when using an Audio Reference (worked fine ~3 weeks ago)
URL: https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/54  by tomookazaki87-glitch 2026-09-07T20:17:21Z

Hi,

I'm running into a reproducible crash with the Extender as soon as an Audio Reference is used in a multi-clip sequence.

What works:

Everything runs fine without an Audio Reference (long chains, multiple clips, Motion Context, Disk Join — all stable).
Clip 1 always renders correctly, even with an Audio Reference attached.

What breaks:

As soon as Clip 2 (or any following clip) starts sampling, it crashes with a shape mismatch, every time, 100% reproducible.
I tested this with different Audio Reference lengths (5 seconds and 10 seconds) — same exact error, same exact shapes, regardless of audio length. So it doesn't seem to be about the reference being "too short" for the timeline.
About 3 weeks ago, on an older version of the workflow/node, this exact same setup (short voice reference, multi-clip chain) worked without any issue.

Error:

[ERROR] !!! Exception during processing !!! shape mismatch: value tensor of shape [336, 32] cannot be broadcast to indexing result of shape [410, 32]
...
File "...\ComfyUI\comfy\ldm\minimax\model.py", line 659, in _forward
    all_audio_rows[~audio_update] = cond_audio_rows
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
RuntimeError: shape mismatch: value tensor of shape [336, 32] cannot be broadcast to indexing result of shape [410, 32]

The crash happens inside comfy/ldm/minimax/model.py, called from ComfyUI_MiniMax_H3_Extender/extender.py (extend → _sample_h3 → guider.sample).

My guess: it looks like the audio conditioning built for clip N+ doesn't get resized to account for the Motion Context frames carried over from the previous clip, so the audio tensor stays sized for a "bare" single clip (336 rows) while the model expects it sized for clip + motion context (410 rows). Since it happens identically at any tested audio length, it seems unrelated to the reference duration itself and more likely tied to how the standalone audio reference is combined with Motion Context continuation.

Setup:

ComfyUI (Windows, ComfyUI-Easy-Install)
ComfyUI_MiniMax_H3_Extender, latest version
Using a single standalone Audio Reference (voice reference, not a video-audio reference)
Multi-clip mode with Motion Context continuation

Happy to provide the full workflow / more logs if useful. Thanks for the great node!

> **Comment by tritant (2026-09-07T21:37:16Z):**

Thanks for the detailed report.

The tensor mismatch is interesting:

`410 - 336 = 74`

This is the same 74-row difference we have already seen when the Motion Context audio payload is modified or partially lost from Clip 2 onward.

This does **not necessarily point to one specific custom node**. Several H3-related extensions can patch or wrap the same MiniMax H3 conditioning / `extra_conds` code, and a conflict between those patches can produce this kind of issue.

Could you please test the same workflow with only:

- standard ComfyUI MiniMax H3 nodes
- MiniMax H3 Extender

and temporarily disable other H3-related custom nodes, loaders, accelerators, Turbo patches, Multishot extensions, etc.?

Please fully restart ComfyUI after disabling them.

If the workflow works in that minimal configuration, re-enable the H3 extensions one by one to identify which one introduces the conflict.

If it still fails with only the standard H3 implementation + Extender, please post:

- your exact ComfyUI version
- your exact Extender build (latest is v2.6.2)
- the list of installed H3-related custom nodes
- the relevant startup and generation log around the Motion Context / H3 payload setup

The identical 74-row mismatch strongly suggests that the Motion Context audio conditioning is being altered somewhere, but I don't want to assume which extension is responsible before isolating the environment.
See this issue
https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/issues/18


> **Comment by tomookazaki87-glitch (2026-09-08T11:41:37Z):**

Confirmed: comfyui-h3-multishot is the culprit.

I went through the H3-related custom nodes I had installed one by one (disable → full restart → test):

ComfyUI-MiniMaxH3-Contex-Loop
ComfyUI-H3-Motion-Context-MultiRef-main
ComfyUI-H3-NativeAudioLock
comfyui-h3-multishot
Comfyui_Minimax_h3_latent_Upscaler
minimax-h3-motioncache
ComfyUI-MiniMaxRefPack-main
ComfyUI-ALLinONE-MinimaxH3
comfyui-minimax-h3-turbo
ComfyUI-SolAttn_triton-main

With comfyui-h3-multishot disabled, everything works perfectly — Audio Reference + multi-clip chain runs through all clips without any shape mismatch. All other H3-related nodes (including ComfyUI-H3-Motion-Context-MultiRef-main, which I actively use for lip-sync) stayed enabled and caused no issues at all.

Re-enabling only comfyui-h3-multishot reproduces the exact same crash every time (336 vs 410 shape mismatch, model.py:659).

So it's confirmed isolated to comfyui-h3-multishot specifically — not a general "multiple H3 extensions" conflict. Thanks for the guidance on the isolation test!


> **Comment by tritant (2026-09-08T11:50:28Z):**

Nice test, have fun !

