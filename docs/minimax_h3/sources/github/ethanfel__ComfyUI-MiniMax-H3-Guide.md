# ethanfel__ComfyUI-MiniMax-H3-Guide

Source: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide
Fetched: 2026-11-28

Description: Guided MiniMax H3 prompt preparation node for ComfyUI
Stars: 361  Updated: 2026-08-11T17:23:56Z



---

## FILE: README.md (https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/blob/HEAD/README.md)

# ComfyUI MiniMax H3 Guide

A dependency-free ComfyUI node pack for building MiniMax H3 prompts from
explicit references, subjects, and shots. It compiles those choices into valid
H3 prose and routes the media to ComfyUI's native H3 nodes.

- Typed Plan v2 workflow instead of one large prompt form
- Explicit identity, keyframe, motion, edit-source, voice, and soundtrack roles
- Deterministic labels, timing, validation, and native media routing
- Optional Qwen enhancement with compiler-owned structure kept locked
- Reusable image/audio Reference Sheets and a locked-picture Foley mode
- H3-native AV split/repack with frame-locked audio length

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide
```

Restart ComfyUI, then look under **MiniMax H3 → Plan v2**. The pack has no
additional Python dependencies.

This release requires a ComfyUI build with native MiniMax H3 support. Update
ComfyUI if **MiniMax H3 Image to Video**, **MiniMax H3 Reference to Video**, or
the MiniMax tokenizer is missing.

## Start here

```text
Project Setup
  → optional Image / Video / Audio References
  → one Shot per real cut
  → Prompt Merge
  → Apply Reference Plan
  → sampler
```

1. **Project Setup** selects the output size and native frame count.
2. **Reference nodes** declare what each file means. The compiler assigns H3
   labels and routes automatically.
3. **Shot nodes** define the timeline. Attach keyframes, motion, and dialogue to
   the Shot that uses them.
4. **Prompt Merge** validates and compiles the plan.
5. **Apply Reference Plan** hands the matching prompt, media, and AV latent to
   ComfyUI's native H3 implementation.

For the complete node-by-node workflow, see the
[Plan v2 guide](docs/PLAN_V2.md).

## Pick roles by intent

| Intent | Use |
| --- | --- |
| Exact opening or ending composition | Image Reference as an endpoint |
| Preserve identity, wardrobe, object, scene, or style | Image Reference as reusable content |
| Transfer pose, action, or body mechanics | Video Reference + Attach Motion |
| Modify an existing video | Video Reference as an edit source |
| Continue an existing video | Video Reference as a continuation source |
| Keep or copy source audio | Paired Audio Reference with continuity/copy role |
| Borrow voice, beat, music style, or sound texture | Audio Reference with the matching guidance role |

The same file type can take different routes. Its intended job—not its
extension—decides the H3 mode.

## Example workflows

Examples are available in [`example_workflows/`](example_workflows/) and
ComfyUI's workflow template browser.

| Workflow | Status | Purpose |
| --- | --- | --- |
| Animate Keyframe with Motion Reference | Ready | Retarget motion onto a supplied opening keyframe |
| Prompt Builder App | WIP | Prompt-only Plan v2 interface |
| Identity and Voice | WIP | Identity, voice, dialogue, and Ref2VA generation |
| First and Last Frames | WIP | Exact endpoint composition |
| Five Shot Keyframe Composition | WIP | Per-shot keyframes and motion transfer |
| Character Replacement | WIP | Replace one performer in an edit-source video |
| Video Extension with Audio Continuity | WIP | Character transfer plus continuation |
| Video to Audio Foley | WIP | Locked picture track with generated audio |
| Foley with Sound Reference | WIP | Foley guided by a sound texture |
| Foley with Multiple Sound and Voice References | WIP | Multi-reference Foley and dialogue |

The ready motion-reference workflow is the reference implementation. WIP
templates expose their intended graph but still need task-specific media and
default refinement.

## Limits worth knowing

| Constraint | H3 limit |
| --- | --- |
| Output | 4–15 seconds on the native `17k+5` frame grid at 24 FPS |
| Images | 9 |
| Videos | 3 |
| Audio clips | 3 |
| Mixed reference files | 12 total |
| Each reference video or audio clip | 2–15 seconds |
| Cumulative video references | 15 seconds |
| Cumulative audio references | 15 seconds |

The 362-frame endpoint plays for 15.083 seconds. One complete-copy or
continuity soundtrack paired with a matching 362-frame video may share that
padded boundary; it does not increase the general audio limit.

## Guides

- [Plan v2 workflow and node reference](docs/PLAN_V2.md)
- [Video-to-audio Foley](docs/FOLEY.md)
- [Qwen enhancement and prompt review](docs/QWEN_ENHANCER.md)
- [Persistent Reference Sheets](docs/REFERENCE_SHEETS.md)
- [H3 AV latent split and frame-locked repack](docs/AV_LATENTS.md)
- [Preparing reference media](REFERENCE_MEDIA_PRACTICAL_GUIDE.md)
- [Migrating legacy workflows](MIGRATION_TO_PLAN_V2.md)
- [Audit and compatibility decisions](AUDIT_REPORT.md)

Legacy nodes remain registered so existing workflows load. New projects should
use Plan v2.

## Development

```bash
pytest -q
```

## Credits

The interaction direction was inspired by
[ComfyUI-MiniMaxH3-Easy](https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy),
used under its MIT license. This project retains an independent Plan v2 data
model, validator, router, and compiler.

The H3 AV timing nodes were cross-checked against ComfyUI's native H3 grid
formulas and the independent findings in
[ComfyUI-MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools).

Prompt structure follows MiniMax's official
[base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md),
[reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md),
and [model card](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## License

GNU General Public License version 3 only
([GPL-3.0-only](LICENSE)).


---

## FILE: REFERENCE_MEDIA_PRACTICAL_GUIDE.md (https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/blob/HEAD/REFERENCE_MEDIA_PRACTICAL_GUIDE.md)

# MiniMax H3 Practical Reference Media Guide

This document records practical guidance for preparing scene-composition and
motion references for MiniMax H3. It complements MiniMax's formal prompt format
guide; it is not a claim that rough references guarantee exact geometry or
motion.

## 1. Choose the reference role by intent

| Desired result | Reference role | Typical retention |
| --- | --- | --- |
| Broad layout, blocking, viewpoint, or shot order | `Storyboard or shot planning` | `weak_reference` |
| A composition that should be followed closely | `Concrete Ref2VA keyframe or composition anchor` | `fully_preserved` |
| The literal opening or ending frame | `Exact first frame` or `Exact last frame` | `fully_preserved` |
| Reuse a person's or object's appearance | `Identity or appearance` or `Object, prop, clothing, interface, or effect` | Usually `fully_preserved` |
| Transfer an action, trajectory, pose change, or timing | `Motion or action` | `attribute_transfer` when applied to another subject |

MiniMax's reference format distinguishes reusable visible content (`<Subject
N>`) from a concrete picture used as a keyframe, composition anchor, or
storyboard (`<Picture N>`). Do not use an exact endpoint role merely because an
image happens to resemble the intended shot.

## 2. Minimum useful scene-composition reference

A reference does not have to be photorealistic. A sketch, collage, blocked 3D
viewport, or simple render can work when it communicates these relationships
clearly:

- Target aspect ratio.
- Camera viewpoint, height, and approximate lens/framing.
- Number of visible subjects.
- Each subject's screen position, size, and facing direction.
- Foreground, middle-ground, and background separation.
- Horizon and major perspective lines.
- Important occlusion relationships and intentional empty space.

A rough image controls broad composition more reliably than identity, anatomy,
lighting, texture, or fine object details. Use separate identity/item reference
images when those details matter.

### Practical image preparation

- Use the target video's aspect ratio whenever possible.
- A short edge around 768 pixels is a useful practical baseline. This is not an
  official minimum, but very small references are not enlarged by ComfyUI's H3
  reference pipeline.
- Keep silhouettes and major shapes visually distinct.
- Avoid visible arrows, labels, grids, or bounding boxes unless their possible
  appearance in the result is acceptable.
- Put mapping and ignore instructions in prompt metadata rather than drawing
  them into the image.
- Use one composition image per shot when exact shot-to-image mapping matters.

Example role notes for a rough layout:

```text
Use only the viewpoint, subject placement, relative scale, depth layers, and
camera height. Ignore the sketch style, placeholder colors, unfinished anatomy,
and missing surface details.
```

## 3. Designating character positions

Define character identity separately from the composition image:

```text
Character image 1
  role: Identity or appearance
  content_group: woman

Character image 2
  role: Identity or appearance
  content_group: man

Rough composition image
  role: Storyboard or shot planning
  retention: weak_reference
  shot_scope: 3
  notes: Map the left foreground placeholder to woman and the right background
         placeholder to man. Preserve blocking, relative scale, and eyelines.
         Ignore placeholder appearance and color.
```

The Shot description should reinforce the intended blocking using the viewer's
screen coordinates:

```text
The woman occupies the left third in a tight eye-level close-up, centered around
25% of the frame width. The man occupies the right third in a lower medium
close-up, centered around 75% of the frame width. Both remain visible
simultaneously.
```

Useful position vocabulary:

- `screen-left`, `screen-right`, and `center`.
- `left third`, `center third`, and `right third`.
- Approximate percentages of frame width or height.
- `foreground`, `middle ground`, and `background`.
- `extreme close-up`, `head-and-shoulders`, `waist-up`, and `full body`.
- Facing direction, eyeline, relative scale, and occlusion.

For this node pack, compact shot scopes are:

```text
3       one shot
3,4     selected shots
3-5     an inclusive range
all     every shot
```

If the composition source is an animatic video rather than a still, use
`Camera, cuts, or rhythm` for its whole-video editing language and timing. Use
separate identity references for the final characters.

## 4. Minimum useful motion reference

The simplest adequate proxy depends on what must transfer:

| Proxy | Information it can communicate reliably |
| --- | --- |
| Moving point | Screen-space position, path, timing, speed, and easing |
| Oriented point, arrow, or circle with a direction line | Position plus facing direction or simple rotation |
| Rigid head/object proxy | Translation, yaw, pitch, roll, and scale change |
| Stick figure or simple rig | Joint articulation and recognizable body poses |
| Rough rendered animatic | Articulation, contacts, camera relationship, depth, and timing |

A moving point can guide the center of a head from left to right, but it cannot
by itself distinguish among:

- The whole head translating sideways.
- The head turning left-to-right around the neck (yaw).
- The head tilting sideways (roll).
- The head nodding up and down (pitch).
- The eyes merely tracking an off-screen target.

Use an oriented proxy whenever rotation matters. For a left-to-right head turn,
a circle with a nose/direction line is a reasonable minimum:

```text
Start: face points screen-left.
Middle: face points toward camera.
End: face points screen-right.
The head center and body remain stationary.
```

Every motion reference should make these properties explicit:

- Start and end states.
- Screen-space or 3D path.
- Speed, acceleration, pauses, and easing.
- Translation versus rotation versus deformation.
- The body part or object receiving the motion.
- What remains stationary.
- Whether the source camera moves.
- Contacts or constraints, such as a hand remaining on a table.

### Motion-transfer binding

```text
Target identity reference
  role: Identity or appearance
  content_group: woman
  retention: fully_preserved

Motion video
  role: Motion or action
  retention: attribute_transfer
  content_group: head_motion
  transfer_target: woman
  shot_scope: 3
  notes: Transfer only the head motion. Keep the body and camera stationary.
         Ignore the proxy's shape, color, identity, and background.
```

Example for a moving-point trajectory:

```text
Transfer only the point's screen-space path and timing to the center of the
woman's head. Her head translates smoothly from screen-left to screen-right
without rotating. Keep her body stationary and do not include the point.
```

Use a 2-15 second reference video for the normal H3 Ref2VA workflow. A short
motion can be padded with clear stationary holds before and after the action so
the initial and final states remain legible.

## 5. Reliability checklist

Before generating, verify:

- The selected role describes the intended relationship, not merely the file
  type.
- Every identity or reusable item has a stable `content_group`.
- An `attribute_transfer` source names a different visible target group.
- The role's `shot_scope` refers to an existing Shot.
- Rough reference colors and shapes are explicitly marked as disposable when
  they should not transfer.
- The prompt distinguishes position from orientation and object motion from
  camera motion.
- The native H3 inputs receive the same reference media described by the Guide
  and Enhancer context.

## Sources

- [MiniMax H3 full-reference prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [ComfyUI MiniMax H3 native node implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)



---

## FILE: docs/AV_LATENTS.md (https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/blob/HEAD/docs/AV_LATENTS.md)

# H3 AV latent split and repack

Use these nodes when a workflow keeps the video stream from one MiniMax H3
latent and the audio stream from another—for example, retaining first-pass
audio after a video-only refinement pass.

```text
first-pass H3 AV ── MiniMax H3 Split AV Latent ── audio ─┐
                                                        ├─ MiniMax H3 Pack AV Latent (Frame-Locked)
refined H3 AV ───── MiniMax H3 Split AV Latent ── video ┘
```

Replace `LTXVSeparateAVLatent` and `LTXVConcatAVLatent` on this route. The LTX
nodes can nest the H3 tensors, but after separation the video input is plain,
so the joiner does not necessarily reconcile the audio length.

## Why the lengths differ

H3 uses separate clocks:

| Stream | Shape | Time rule |
| --- | --- | --- |
| Video | `[B, 24, Tv, H, W]` | `Tv = 5k+2` represents `17k+5` frames at 24 FPS |
| Audio | `[B, 32, 2, Ta]` | 40 latent steps per second |

The frame-locked pack node calculates the output duration from the video
latent, then sets audio to `round(frames / 24 × 40)`. Extra audio is trimmed
from the tail; short audio is silence-padded at the tail. The origin remains
at time zero, so lip-sync does not receive a compensating offset.

If either input carries a noise mask, the node constructs a matching nested
mask. A silence-padded audio tail receives mask value `1`, allowing the model
to generate that missing tail instead of pinning silence.

## Validation

The nodes reject:

- non-H3 channel layouts;
- video lengths outside the `5k+2` latent grid;
- mismatched video/audio batch sizes;
- masks whose temporal length no longer matches their source latent.

The pack output keeps metadata from the video latent. Audio metadata cannot
silently overwrite the video stream's batch or sampling metadata.


---

## FILE: docs/FOLEY.md (https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/blob/HEAD/docs/FOLEY.md)

# Video-to-audio Foley

Foley mode keeps the source picture track and asks H3 to generate a new
synchronized audio stream.

## Graph

```text
Project Setup
  → Foley Target: decoded source frames and real source FPS
  → optional Audio References
  → one Shot per source cut
  → Prompt Merge
  → optional Structured Enhancer / Review Gate
  → Apply Reference Plan
  → sampler
```

Foley Target carries the source frames as target media, not as a Ref2VA Video
Reference. Presenting the same video through Video Reference adds an expensive
reference stream without helping the locked latent.

## Latent behavior

Apply Reference Plan obtains positive conditioning from the normal H3 prompt
node, replaces its empty video stream with the encoded source frames, and
constructs:

```text
video latent: source video, noise mask 0 → preserve
audio latent: empty target, noise mask 1 → generate
```

The generated audio replaces the complete soundtrack. Optional Audio
References guide qualities such as timbre, material, texture, voice, or beat;
they do not preserve the source signal.

## Writing the Foley timeline

Create one Shot per real source cut. Write visible timing anchor → audible
result in chronological order.

```text
initial_prompt: Generate realistic production Foley synchronized to the locked source video. No dialogue and no music.

Shot 1: A person crosses the tiled room. Each visible heel and sole contact produces a short dry indoor footstep at the exact contact frame; clothing produces light rustle during each stride.

Shot 2: At the cut, a hand sets a glass on a wooden table. A brief glass-on-wood contact occurs when the base touches the surface, followed by a faint settling tick.

overall_soundscape: Low continuous indoor room tone remains beneath the synchronized footsteps, cloth movement, and object contacts.
non_diegetic_music: N/A
```

Guidelines:

- Use actions visibly supported by the locked video.
- Keep continuous ambience in `overall_soundscape`.
- Keep discrete events inside the Shot where they occur.
- Set `non_diegetic_music` to `N/A` unless an audience-only score is intended.
- Avoid visual/camera requests; the picture track is latent-locked.

When visual analysis is enabled, the enhancer samples timestamped source
frames as evidence. It may infer ordinary physical sounds from visible action,
but cannot add unseen events, speech, or music.

## Sound references

For a footstep texture reference, register an Audio Reference as
**Sound-effect texture**, then place its assigned `<Audio N>` tag in the Shot
sentence at the exact contact event. Source timing remains controlled by the
video; the reference supplies sonic character.

Complete-copy and partial-copy audio relationships are rejected for Foley
because the target audio stream is fully regenerated. Preserving only part of
an original soundtrack requires a separate audio-time mask workflow.

## Requirements

Foley requires MiniMax H3 per-token mask support from
[ComfyUI PR #15375](https://github.com/Comfy-Org/ComfyUI/pull/15375) or an
equivalent compatibility patch, such as a compatible per-row mask patch on the
MODEL branch.

Match Apply Reference Plan's width and height to the source aspect ratio. The
adapter resizes pixels and pads the temporal length to the H3 frame grid.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Picture changes | Ensure the video is connected only through Foley Target |
| Audio is noisy or muted | Test deterministic/ODE sampling or `eta=0` |
| Events drift after cuts | Create one Shot per real source cut |
| Reference texture appears at the wrong time | Put `<Audio N>` in the exact Shot sentence |
| Source audio disappears | Expected: Foley regenerates the complete audio stream |


---

## FILE: docs/PLAN_V2.md (https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/blob/HEAD/docs/PLAN_V2.md)

# Plan v2 workflow and node reference

Plan v2 represents an H3 project as one ordered `MINIMAX_H3_PLAN_V2` value.
Each node adds explicit semantic information; Prompt Merge assigns final labels
and compiles the native route.

## Recommended graph

```text
Project Setup
  → optional Foley Target
  → global Image / Video / Audio References
  → optional Subject Binding or Character Replacement
  → Shot 1
  → optional Attach Keyframe / Attach Motion / Dialogue Event
  → Shot 2 → its attachments
  → Prompt Merge
  → optional Structured Prompt Enhancer
  → optional Apply Structured Prose
  → optional Prompt Review Gate or Inline Prompt Override
  → Apply Reference Plan
  → sampler
```

Global references must be registered before the first Shot. Shot attachments
consume the stable handle returned by their Shot, so media never needs a
manually typed numeric scope.

## Project Setup

Project Setup defines target width, height, frame count, creative intent,
style, soundscape, and music. The frame selector advances through H3's native
`17k+5` grid at 24 FPS. The node displays the exact playback duration as
`frame_count / 24`, including 362 frames / 15.083 seconds.

## Reference nodes

### Image Reference

Image Reference distinguishes:

- reusable identity, wardrobe, object, scene, or style evidence;
- exact first and last endpoint frames;
- concrete keyframes;
- storyboard/planning frames.

Reusable visible content creates a Subject. Endpoint frames stay in the native
I2VA/L2VA/FL2VA route. A keyframe becomes a Picture attached to a specific
Shot.

### Video Reference

Video Reference distinguishes:

- a source video to edit;
- a source video to continue;
- reusable visible content;
- motion or action transfer;
- camera, cut, rhythm, or temporal structure.

Motion clips keep a native Video route while their reusable action is compiled
as a separate Subject. This lets one Subject receive another clip's pose or
action without asking H3 to reproduce the whole reference video.

### Audio Reference

Audio roles include voice, music style, beat, sound-effect texture, exact
dialogue/lyrics, continuity, complete copying, partial copying, and broad
inspiration.

- Voice requires a target Subject.
- Exact content requires a language and transcript.
- A paired-video handle marks audio as that Video Reference's soundtrack.
- Complete-copy and continuity audio must match the paired video interval
  within one 24 FPS frame.

The compiler places a tagged nonverbal audio reference at the exact sentence
location authored in Shot prose. Untagged texture/style references retain a
global soundscape fallback.

## Subject Binding

Subject aliases are human names such as `woman`, `watch`, or `truck`. Final
`<Subject N>` numbers are compiler-owned.

Use Subject Binding to:

- combine multiple references as evidence for one Subject; or
- transfer an attribute, pose, expression, or action to an existing Subject.

## Character Replacement

Character Replacement maps one precisely described performer in an edit-source
Video to one upstream identity Subject.

```text
Image Reference: identity
  → Video Reference: source video to edit
  → Character Replacement
       source character: the woman in the red jacket
       replacement: upstream identity Subject
  → Shots
```

The appearance policy controls identity, body, and wardrobe independently.
Separate preservation switches retain source performance and surroundings.

## Shots and attachments

Each Shot declares one cut time. The next cut or Project duration determines
its end. The chain rejects gaps, overlaps, reversed ranges, and a first Shot
that does not begin at `0.000`.

### Attach Keyframe

Registers one image as the Shot's opening, internal, or ending composition
anchor. The image receives a native Picture route and is cited only in that
Shot.

### Attach Motion

Transfers a Video's motion/action Subject to an upstream target Subject only
for the selected Shot. The compiler writes the relationship and assigns the
native labels.

### Dialogue Event

Dialogue attaches to the most recently opened Shot. Optional
`start_offset_seconds` uses:

- `-1`: automatic placement;
- `0`: Shot opening;
- positive value: seconds after the Shot opening.

Continuity can mark a line complete, split across the next cut, carried from
the previous cut, or interrupted by the video ending. Cross-cut lines receive
the required `<scenetrans>` markers; final interruptions receive `<cutoff>`.
Dialogue words and punctuation remain exact.

## Prompt Merge

Prompt Merge assigns Subject/Picture/Video/Audio numbers, validates scopes and
timing, canonicalizes native media order, and returns:

| Output | Purpose |
| --- | --- |
| `h3_prompt` | Deterministic H3 prompt |
| `rewrite_request` | Locked prose-enhancement contract |
| `plan_context` | Typed compiled plan |
| `problems_report` | Readiness, timing, inventory, routes, and warnings |
| `h3_length` | Native output frame count |

## Prompt refinement

The structured enhancer may expand editable prose, but Python restores and
checks all compiler-owned structure. Apply Structured Prose recompiles edited
JSON without rerunning Qwen. Prompt Review Gate provides an interactive final
approval. Inline Prompt Override provides a non-pausing A/B path.

See [Qwen enhancement and prompt review](QWEN_ENHANCER.md).

## Apply Reference Plan

Connect a prompt and `plan_context` from the same Prompt Merge, enhancer,
structured-prose, review, or override result. Apply Reference Plan verifies the
pair, routes stored media, and delegates conditioning to ComfyUI's installed
`MiniMaxH3ImageToVideo` or `MiniMaxH3ReferenceToVideo` implementation.

The node returns native positive conditioning and the joint audio/video latent.
Reference audio requires the audio VAE. Text-only, endpoint, and reference-free
Foley plans do not.

## Route selection

| Goal | Mode | Checkpoint family |
| --- | --- | --- |
| Text generation | T2VA | H3-Base-FL2VA |
| Exact first frame | I2VA | H3-Base-FL2VA |
| Exact final frame | L2VA | H3-Base-FL2VA |
| Exact first and final frames | FL2VA | H3-Base-FL2VA |
| Appearance, motion, editing, continuation, or audio references | Ref2VA | H3-Base-Ref2VA |

## Limits and native ordering

- Up to 9 images, 3 videos, 3 audio clips, and 12 mixed files.
- Each reference video/audio clip must be 2–15 seconds.
- Video references total at most 15 seconds.
- Audio references total at most 15 seconds.
- Reference audio cannot be the only Ref2VA media input.
- One complete-copy/continuity soundtrack paired to a matching 362-frame video
  may share the native 15.083-second boundary.

Native ordering is Pictures first, then each paired soundtrack immediately
before its Video, then standalone Audio. Picture, Video, and Audio labels are
numbered independently.


---

## FILE: docs/QWEN_ENHANCER.md (https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/blob/HEAD/docs/QWEN_ENHANCER.md)

# Qwen enhancement and prompt review

The optional structured enhancer expands Plan v2 prose without allowing a
language model to redefine the compiled plan.

## Data flow

```text
Prompt Merge
  → Structured Prompt Enhancer
  → optional Apply Structured Prose
  → optional Prompt Review Gate or Inline Prompt Override
  → Apply Reference Plan
```

The enhancer receives the compiled scene, references, roles, Shots, timing,
audio metadata, native routes, compiler report, and optional visual evidence
in one request.

## Enhancement modes

### Intent-locked expansion

This is the default. Original prose stays intact. Qwen may add short compatible
Shot, camera, or soundscape details. Attempts to replace global intent, style,
music, timing, or compiler-owned content are ignored and reported.

### Creative expansion

Qwen may rewrite editable prose and introduce additional presentation choices.
Labels, roles, timing, dialogue, Shot order, and native routing remain locked.

## Structured output

Qwen returns a versioned JSON object. Python validates it, reconstructs the H3
prompt, and checks:

- reference labels and roles;
- Subject retention and transfer relationships;
- native media routes;
- Shot order and cut times;
- speaker order and exact dialogue;
- required H3 structure.

Invalid, incomplete, or collapsed output falls back to the deterministic
compiler draft. `editable_prose` is always the final validated JSON. Use
**Apply Structured Prose** after a text editor to refine and recompile that JSON
without running Qwen again.

## Visual and audio context

The enhancer may analyze selected images and timestamped video samples. Audio
waveforms are not presented as understandable model input; audio meaning comes
from explicit Audio Reference metadata and transcripts.

## Model options

The `CLIP` input accepts:

1. a complete generation-capable Qwen3-VL or Qwen3.5 model; or
2. MiniMax H3's normal 50-layer conditioning CLIP plus the optional 50–63
   generation tail.

For option 2, use **MiniMax H3 Generation Tail Loader** with
`qwen3vl_32b_minimax_h3_generation_tail_50_63_int8_convrot.safetensors`.
Download it from
[`ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot`](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot)
and place it under:

```text
ComfyUI/models/text_encoders/MiniMax-H3/
```

The tail loader itself holds only a descriptor. During generation, the
enhancer reuses the connected embedding, vision tower, and language layers
0–49, then loads layers 50–63, final norm, and LM head. The tail is released
after decoding; the connected CLIP remains under ComfyUI's normal model manager.

DynamicVRAM can stream the model, but sub-32-GB operation is hardware-dependent
and may be slow. Keep DynamicVRAM enabled. `--highvram` and `--gpu-only` defeat
the low-VRAM path.

## Prompt Review Gate

The Review Gate pauses the queued job and opens the full H3 prompt in an editor.
It supports approval, input restoration, revision history, rejection, and an
optional timeout.

Editable content includes scene, camera, ambience, and music prose. The gate
rejects changes to compiler-owned sections, labels, retention, Shot markers,
cut timestamps, dialogue, replacement mappings, and media routes.

History is stored under ComfyUI user data and contains prompt text and hashes,
not reference media or tensors. Pass-through mode leaves the gate in the graph
without pausing.

## Inline Prompt Override

Inline Prompt Override provides the same plan-bound descriptive validation
without the interactive pause. Paste a complete experimental prompt, connect
the matching `plan_context`, and toggle `use_override`. Clear or disable it to
bypass the experiment.


---

## FILE: docs/REFERENCE_SHEETS.md (https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/blob/HEAD/docs/REFERENCE_SHEETS.md)

# Persistent Reference Sheets

A Reference Sheet is a reusable image/audio library entry. It can describe a
person, object, outfit, location, style, voice, sound, or a mixed project
collection.

## Storage

Sheets live outside the custom-node repository:

```text
ComfyUI/user/default/minimax_h3/reference_sheets/
└── <sheet-name>--<short-id>/
    ├── manifest.json
    ├── images/
    └── audio/
```

Each versioned manifest contains a UUID, descriptions, tags, relative paths,
and SHA-256 checksums. Connected `IMAGE` tensors are saved as PNG and connected
`AUDIO` tensors as WAV, so a sheet does not depend on the original workflow or
input filename.

Set `MINIMAX_H3_REFERENCE_SHEETS_DIR` before starting ComfyUI to use a custom
library root.

## Create a sheet

```text
Load Image.IMAGE → Reference Sheet.image_1
Load Image.IMAGE → Reference Sheet.image_2  (optional)
Load Audio.AUDIO → Reference Sheet.audio_1  (optional)
```

Choose **Create new**, provide a display name and metadata, then queue once.
The node accepts up to four image sources and three audio sources. An image
batch expands into separate saved pictures up to H3's nine-image limit.

Create never overwrites another sheet.

## Load and select media

Choose **Load existing**, select a sheet, then choose a thumbnail or audio item
from the embedded gallery. Selection is carried in the node output; users do
not type internal filenames or asset keys.

Audio uses a non-destructive trim window:

- `audio_start_seconds` selects the offset;
- `audio_duration_seconds` selects a 2–15 second output segment.

The stored WAV and player remain complete. Duplicate the Reference Sheet node
to use several assets or different segments from one saved clip in a workflow.

## Update behavior

Update requires confirmation and is atomic.

- Metadata-only updates preserve every media asset.
- Connected non-duplicate media is appended.
- Byte-identical media is skipped.
- Existing assets are not treated as a replaceable input list.

## Use with Plan v2

```text
Reference Sheet.selected_image → Image Reference.image
Reference Sheet.selected_audio → Audio Reference.audio
```

The Plan v2 reference node declares the current workflow's role, retention,
scope, speaker/layer binding, and native route. The sheet itself remains a
role-neutral media library.

The structured Qwen enhancer can analyze a selected image after it enters the
Plan v2 inventory. It does not analyze the audio waveform; audio meaning comes
from Audio Reference metadata. Video is handled by Video Reference rather than
Reference Sheet.


---

## FILE: AUDIT_REPORT.md (https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/blob/HEAD/AUDIT_REPORT.md)

# MiniMax H3 workflow audit

Audit date: 2026-08-04
Branch: `main`

## Sources and scope

This audit reviews the public nodes, their current context payload contracts,
and their documented workflow against:

- MiniMax's official [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- MiniMax's official [full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- The official [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- ComfyUI `14b05228` (2026-08-02), especially `nodes_minimax_h3.py`,
  `text_encoders/minimax.py`, and `text_encoders/qwen3vl.py`. Native MiniMax H3
  support first appeared in ComfyUI commit `57500fc5bc92`.

The review covers prompt families, task prefixes, semantic labels, retention
markers, shots and timestamps, native frame alignment, reference-media order,
Qwen visual presentation, the optional generation tail, node contracts,
tooltips, compatibility behavior, and regression coverage.

## Native ComfyUI contract used by this pack

- Target video is 24 FPS. Native `length` is rounded upward to a frame count
  satisfying `frames % 17 == 5`.
- Base image-to-video accepts a first frame, a last frame, or both. The first
  frame is stretched to the target canvas; the last frame is center-cover
  cropped. Composition is exact only when source and target aspect ratios agree.
- Full-reference presentation order is all pictures first, then each video with
  its enabled soundtrack audio immediately before it, followed by standalone
  audio. Prompt labels are one-based; native sockets are zero-based.
- Reference videos are interpreted as 24 FPS, truncated to target length,
  rounded downward to the same `17k+5` grid, and shown to MiniMax's Qwen
  tokenizer at fixed 2 FPS.
- MiniMax's tokenizer prepends raw reference blocks rather than applying a chat
  template. Every two video samples form one temporal patch; audio is not visual
  input to Qwen.
- The bundled H3 Qwen3-VL-32B CLIP contains 50 language layers and no final norm
  or LM head. It is a conditioning encoder, not a complete text generator. The
  optional side tail supplies layers 50-63, final norm, and LM head temporarily.

## Findings and resolutions

| Priority | Finding | Resolution |
| --- | --- | --- |
| Critical | Unassigned inventory files silently became appearance/content references and could create unsupported Subjects. | Bare Guide inventory remains descriptive. New Visual Reference nodes default to `Unassigned - choose a reference role` and require either an explicit compatibility role or a role-binding chain before execution. Once selected, that role intentionally supplies semantic evidence. |
| Critical | Reusable reference collections had no safe persistence contract, and the first persistence UI exposed filename/key bookkeeping instead of behaving like a media sheet. | One integrated Reference Sheet node now saves connected `IMAGE`/`AUDIO` values into a self-contained ComfyUI user library, displays saved thumbnails/audio, and carries the clicked selection to role-aware use nodes without typed paths or asset keys. Relative paths, checksums, create-without-overwrite, and confirmed atomic updates remain enforced; the old Image Asset, Audio Asset, and Library builders are disabled. |
| Critical | Adding connected media during `Update existing` rebuilt the manifest only from those connections, silently deleting every previously saved asset from the sheet. | Update is now an atomic append-only merge: all verified saved files and metadata are staged first, new media receives non-colliding keys, byte-identical connections are skipped, and capacity or write failures leave the original sheet untouched. The report gives preserved, appended, and duplicate counts. |
| High | Reference Sheet always emitted the complete selected audio file, so two otherwise useful eight-second references could not fit H3's 15-second cumulative audio limit without destructively editing source files elsewhere. | The integrated sheet now provides exact numeric start-offset and 2–15 second duration fields. It slices only the runtime `selected_audio`/legacy output, preserves the complete saved WAV, reports the exact applied range, and carries the same trim through legacy sheet-audio routing. Plan v2 cumulative-limit errors now list the total and every active clip duration. |
| Critical | A single asset role could not express multiple official relationships, and multiple assets could not jointly define one Subject. | Visual roles are repeatable bindings with a content group, retention relationship, optional shot scope, and notes. A shared group deterministically combines those assets into one Guide `<Subject N>`; separate groups remain separate Subjects. Direct keyframe/storyboard/edit/continuation/structure roles stay Picture/Video rows instead of becoming generic Subjects. The enhancer receives the same authoritative context for visual expansion, while user wiring controls native file order. |
| Critical | Storyboards and concrete keyframes shared one ambiguous role. | They are separate roles. Concrete frames produce keyframe-completion semantics; storyboards produce reference-generation semantics. The old value remains readable for saved 0.6.x workflows. |
| Critical | Endpoint images were forced through Ref2VA when chained for enhancement, preventing two-frame FL2VA analysis. | First-frame and last-frame bindings form a base endpoint context, preserve I2VA, L2VA, or FL2VA analysis, and report `first_frame` / `last_frame` as the intended native sockets. The user still makes those links. Base endpoint and Ref2VA roles cannot share one context. |
| Critical | A hand-written multi-shot plan was embedded as prose inside Shot 1, while final-frame syntax could cite a Shot that did not exist. | The legacy text plan is parsed into real sequential shot markers and validated. A connected Shot chain remains authoritative. Final-image alignment always cites an emitted final Shot. |
| Critical | Prompt duration could disagree with native H3 output. | The Guide exposes `h3_length`, rounds requested time to native `17k+5` frames, uses effective time in the prompt, and extends the final described shot through the effective end. |
| Critical | Enhancer video samples did not match MiniMax's temporal-pair tokenizer and could describe footage native H3 discarded. | Reference video is normalized to 24 FPS and native effective length. Generic Qwen keeps configurable sparse samples; MiniMax receives a distinct fixed-2-FPS sample sequence from the effective clip. Visual evidence excludes a discarded tail, although free-form draft text can still mention it and requires review. |
| Critical | Empty or punctuation-only model output could be reported as successful after falling back to the manual prompt. | Collapse checks now inspect raw decoded output, including empty text, punctuation, and incomplete thinking blocks. Fallback is explicit in `enhancer_report`. |
| Critical | Structured Qwen could copy compiler-owned `<d>` dialogue markup into editable Shot prose, causing a valid enhancement to fall back unchanged. | Dialogue lines are masked in the read-only compiled context. If a model still leaks a tagged dialogue segment into otherwise valid JSON, the enhancer removes only that segment, validates the remaining prose, and lets the locked compiler restore the exact dialogue once. Manual Apply Structured Prose remains strict. |
| Critical | A structurally valid Qwen rewrite could soften the source's creative emphasis and replace it with invented lighting, atmosphere, camera, or emotional interpretation. | The enhancer now defaults to `Intent-locked expansion`: Python retains every original prose field outside the model response and composes only compatible Shot/camera/sound addenda. Global replacements and camera additions where the source left camera blank are ignored and reported. The previous full-rewrite behavior remains explicitly available as `Creative expansion`. |
| High | Old workflows serialized obsolete built-in system prompts indefinitely. | Exact known historical defaults are migrated to the current built-in prompt by hash; genuinely customized prompts are preserved verbatim. |
| High | Global fidelity could assign `attribute_transfer` to endpoints, editing sources, or structural references. | Retention is validated per role. Exact endpoints require `fully_preserved`; transfer is limited to reusable visible content with an identifiable grouping/target. |
| High | Retention locations and task types were inferred from file type or hard-coded to Shot 1. | With `reference_context` connected, the Guide derives task types and one retention row per tracked item directly from validated bindings. Explicit shot scopes are preserved and an unknown Subject location stays unspecified instead of being fabricated as Shot 1. The enhancer still performs the detailed visual rewrite, which requires review because generated prose cannot be semantically proven correct. |
| High | Labels could be zero, gapped, duplicated, or inconsistent with native socket order. | Context labels are canonical. Guide inventory rejects zero, duplicates, and active gaps, and reports order mismatches. A selected role may intentionally synthesize a required placeholder label, including a partially missing endpoint; `mode_report` identifies every such placeholder as an expected native connection, not an attached file. |
| High | The enhancer prompt omitted several official formatting rules and accepted malformed output without diagnostics. | The built-in instructions cover section order, style placement, first-use labels, one retention row per tracked item, endpoint alignment, speaker IDs, voiceover, `<scenetrans>`, `<cutoff>`, and audio continuity. Post-generation validation diagnoses missing or empty sections, section order, basic shot markers, style opening, task-type and retention-marker vocabulary, definition/retention row cardinality, supplied visual labels, full-copy audio-section guards, and endpoint-instruction presence. Other semantic rules remain review requirements. |
| High | `Camera direction:` was emitted as a metadata label contrary to the base guide. | Camera movement is rendered as natural English inside each shot. |
| High | Full-copy audio could conflict with newly invented ambience/music. | Guide-generated audio sections state that a complete copy permits no additional audible layer and cite the copied track; the Guide warns when its dialogue/text field could imply a new signal. Enhancer instructions prohibit additions, and structural checks require applicable audio sections to cite the copied label and state exclusivity. Reference Sheet audio now supplies validated labels, descriptions, duration, Shot scope, and standalone native routing; Qwen still does not inspect the waveform. |
| Medium | UI route labels failed through reroutes/bypassed reference nodes. | Frontend traversal follows reroutes and skips bypassed chain nodes. Backend `routing_report` is authoritative for the intended mapping, but the user must make and verify every native connection. |
| Medium | The tail's chunked LM head assumed row-indexable INT8 scales and could mishandle tensor-wise scalar scales or other quantized layouts. | Scalar and per-row INT8 scales are handled explicitly. Unsupported storage/layout/ConvRot combinations fail before logits are calculated. |
| Medium | The real generation tail produced text but appeared not to offload after completion. | The generation loop now owns KV caches, embeddings, hidden states, and logits in a short-lived frame that is cleared before managed tail unload, including interruption/error tracebacks. Tail unload targets every device, then forces a final CUDA cache flush. Connected complete Qwen and MiniMax base CLIPs are never synchronously unloaded by the enhancer; ComfyUI manages their residency. |
| Medium | `llm_prompt` was described as the complete model input although visual tensors are injected separately. | Its tooltip now calls it the serialized text portion, explains MiniMax's prepended Picture/Video blocks for endpoint and Ref2VA media, and warns that an external multimodal LLM needs the pixels in its own visual-token format. |

## Input limits and routing decisions

- Target duration remains 4-15 seconds.
- H3 Ref2VA policy remains limited to 9 pictures, 3 videos, 3 audio items, and
  12 files total. Each video/audio item is 2-15 seconds and each media type
  totals at most 15 seconds. Native ComfyUI's executor is less strict.
- The visual chain validates picture/video counts and source-video durations.
  A connected Reference Sheet audio context validates standalone-audio count,
  per-item duration, total duration, labels, and routes. The Guide diagnoses the
  mixed 12-file count. Legacy manually wired audio remains outside those checks.
- H3 policy forbids audio as the sole Ref2VA input. The Guide rejects a Reference
  Sheet audio context unless a visual reference context is also connected.
- Picture and video numbering is independent and follows chain order within
  each media category. The report recommends native sockets; actual wiring is
  user-controlled.
- The native H3 node remains responsible for AV/keyframe/reference latent
  creation. This pack deliberately returns prompt text rather than incomplete
  `CONDITIONING`.
- Reference Sheet audio decodes and routes standalone `ref_audio_N` assets. It
  does not analyze waveforms with Qwen or create video-paired soundtrack inputs.

## Backward compatibility

- Serialized 0.6.x single-role visual-reference nodes remain accepted and are
  adapted into one role binding.
- Context version 1 remains readable. Newly produced contexts use a validated
  version 2 payload.
- The historical `Storyboard or keyframe` value remains accepted but is treated
  conservatively and reported as legacy/ambiguous.
- Existing custom enhancer system prompts are never rewritten; only exact known
  shipped defaults migrate.
- The new unassigned role is only the default for newly created nodes. Saved
  workflows with an explicit serialized `Identity or appearance` role keep that
  role.

## Remaining limitations

- Reference Sheets currently store images and standalone audio, not video frame
  batches or paired reference-video soundtracks. Sheet Audio Reference chains
  require one shared audio relationship because the Guide's audio sections still
  resolve one relationship family per workflow.
- Qwen3-VL does not inspect saved audio waveforms. The sheet's written audio
  description guides prompt enhancement while the actual `AUDIO` output goes to
  native H3.
- `enhanced_prompt` is a cleaned model candidate, not a guaranteed-valid prompt.
  Structural checks do not prove semantic first-use placement, speaker ordering,
  voice-over wording, dialogue continuity markers, audio continuity, reference
  meaning, or every timestamp against the declared duration.
- Role groups and task evidence constrain the enhancer instructions, but final
  Subject numbering/grouping and summary task types remain probabilistic Qwen
  output.
- Full-copy audio has cross-section citation/exclusivity checks, but this is not
  semantic proof: a free-form target description or custom instruction can still
  conflict and requires review.
- Selected Guide roles may create placeholder media labels. They keep a draft
  structurally complete but do not represent attached files; every placeholder
  must be matched to a real native connection.
- The project owner has smoke-tested the real multi-gigabyte INT8 tail artifact
  and confirmed successful text generation. This is user-reported hardware
  validation rather than a reproducible CI fixture. ComfyUI DynamicVRAM and
  legacy partial loading are designed to support sub-32-GB cards, but the full
  artifact has not been verified on that hardware tier. Practical success and
  speed still depend on the largest active layer, KV/vision working set, system
  RAM, and startup flags; `--highvram` / `--gpu-only` can defeat offloading.
- Visual-reference output labels in the canvas are convenience hints. When a
  role-binding chain makes the route ambiguous to the browser, use the backend
  `routing_report` as the source of truth for intended sockets, then verify the
  native links manually.
- Native first-frame stretching and last-frame center cropping cannot preserve
  an unmatched source aspect ratio. Match endpoint images to the target canvas
  when exact composition matters.

## Validation

The integrated fix branch passed:

| Check | Command/result |
| --- | --- |
| Complete Python suite | `pytest -q -ra` — 153 collected; 153 passed with CUDA available (152 passed and 1 CUDA-only skip on CPU-only runners) |
| Lint | `ruff check .` — passed |
| Byte-code compilation | `python -m compileall -q .` — passed |
| Frontend syntax | `node --check web/visual_reference_routes.js` — passed |
| Patch hygiene | `git diff --check` — passed |
| Dynamic mapping/schema and representative workflow smoke | 10/10 public nodes imported; callable schemas matched; integrated Reference Sheet create/load/update and gallery selection, checksum rejection, visual/audio routing, reversed FL2VA endpoints, Ref2VA transfer/routes, 24-FPS alignment, and legacy manual cuts validated — passed |
| Real INT8 generation tail | Project-owner hardware smoke test — text generation passed; post-generation residency issue reproduced and cleanup corrected in this release |


---

## FILE: MIGRATION_TO_PLAN_V2.md (https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/blob/HEAD/MIGRATION_TO_PLAN_V2.md)

# Migrating existing workflows to Plan v2

Existing node IDs remain registered, so old workflows still load. Their canvas
names now include **Legacy** because the old Guide/context path cannot express
per-reference roles as precisely as Plan v2. Migration is manual: positional
widgets from the large Guide do not have a safe one-to-one semantic conversion.

## Recommended replacement graph

```text
Project Setup
  -> Image / Video / Audio Reference nodes
  -> optional Subject Binding nodes
  -> Shot nodes (+ optional Dialogue Event nodes)
  -> Prompt Merge
  -> optional Structured Prompt Enhancer
  -> optional Prompt Review Gate
  -> Apply Reference Plan
  -> sampler
```

Connect `h3_prompt` and `plan_context` from the same Merge, Structured Prompt
Enhancer, Apply Structured Prose, or Prompt Review Gate node to Apply Reference
Plan. The adapter rejects mismatched pairs instead of silently using stale
labels or routes. Prompt Review Gate pauses only for prompt text; reference
media remains allocated in `plan_context` and needs no chooser.

## Node-by-node map

| Existing node or field | Plan v2 replacement |
| --- | --- |
| Prompt Guide target description | Project Setup `initial_prompt` |
| Prompt Guide duration, style, ambience, music | Project Setup fields |
| Target Timing | Project Setup `h3_length`; Prompt Merge and Apply Reference Plan carry the same authoritative length |
| Legacy Shot start/end text | One Shot node per segment; `cut_at` is the start and the next cut/global duration computes the end |
| Prompt Guide dialogue field | Dialogue Event after the Shot it belongs to |
| Visual Reference Role + Enhancer Visual Reference | One exact Image or Video Reference; add Subject Binding only when the same physical media defines another reusable Subject/role |
| Generic audio context | One exact Audio Reference per clip, including its speaker, layer/event, transcript, copied range, and optional paired-video handle |
| Legacy free-form Prompt Enhancer | Structured Prompt Enhancer (Plan v2), which returns prose JSON and a matching recompiled plan |
| Manual final-prompt text gate | Prompt Review Gate (Plan v2), which validates compiler locks and forwards the same media-bearing plan |
| Manual native reference sockets | Apply Reference Plan, or keep the official native node and follow Prompt Merge's route report |
| Generation Tail Loader | Keep it; both enhancer generations continue to support this side node |

## Reference Sheet migration

Reference Sheet itself is not legacy. It remains the reusable gallery and now
returns both `selected_image` and `selected_audio`:

```text
Reference Sheet.selected_image -> Image Reference.image
Reference Sheet.selected_audio -> Audio Reference.audio
```

Declare the workflow-specific role on the Plan v2 reference node. The old
Reference Sheet Visual/Audio Reference context nodes remain loadable only for
existing Guide workflows and are marked Legacy.

Duplicate Reference Sheet when a generation needs several independently
selected saved assets. Their order in the Plan v2 setup chain determines their
provisional inventory; Prompt Merge then assigns canonical native labels and
routes.

## Practical migration sequence

1. Leave the existing workflow intact and build the Plan v2 spine beside it.
2. Copy only the global creative intent, duration, style, ambience, and music
   into Project Setup.
3. Connect each real media output to its own exact reference node. Do not copy
   a generic media-type role.
4. Replace each old timing row with a Shot start. Add Dialogue Events for exact
   words, voice references, or multiple speakers.
5. Queue Prompt Merge and resolve every reported scope, label, timing, or media
   limit error.
6. Optionally enhance the compiled scene. For structured reusable edits, edit
   `editable_prose` and run Apply Structured Prose. For a final full-prompt
   check, connect both outputs through Prompt Review Gate.
7. Connect the matching prompt/context pair (or both approved gate outputs) to Apply Reference Plan. Load the
   checkpoint family named in `adapter_report`.
8. After a successful generation, remove or bypass the old branch.

There is intentionally no automatic conversion of broad options such as
"voice, music, beat, or sound." Choosing the exact new relationship is the
part that makes the migration correct.


---

## FILE: UX_REDESIGN_RESEARCH.md (https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/blob/HEAD/UX_REDESIGN_RESEARCH.md)

# MiniMax H3 prompt workflow UX redesign

Research date: 2026-08-04
Development branches: `dev/reference-workflow-ux`,
`dev/reference-workflow-ux-phase2`, `dev/reference-workflow-ux-phase3`

Implementation status: Phase 1 provides the isolated `MINIMAX_H3_PLAN_V2`
compiler and eight Plan v2 planning nodes. Phase 2 adds the interaction
extension, hard-locked full-context structured enhancer, and manual Apply
Structured Prose node. Phase 3 adds the native Apply Reference Plan adapter,
workflow onboarding assets, direct Reference Sheet audio output, and an
explicit legacy migration path.

## Recommendation

Replace the current large Prompt Guide form with a small, ordered planning
pipeline:

```text
H3 Project Setup
    -> H3 Image / Video / Audio Reference nodes
    -> H3 Shot nodes (+ optional structured Dialogue events)
    -> H3 Prompt Merge
    -> H3 Prompt Enhancer (optional Qwen rewrite)
```

The chain should be the source of truth. The merger should assign all
`<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`, and native socket
routes deterministically. The LLM should expand the creative description, not
decide what a reference means.

The existing Reference Sheet should remain a reusable media library and
gallery. Selecting or loading media is a different responsibility from
declaring how that media is used in one generation.

The recommended graph is a **semantic spine**, not merely a string-building
chain:

```text
Loaders / Reference Sheet outputs
            |
            v
H3 Project Setup
  -> H3 Image Reference
  -> H3 Subject Binding                 (advanced, optional)
  -> H3 Video Reference
  -> H3 Audio Reference
  -> H3 Shot 1
  -> H3 Dialogue Event                  (optional; attaches to Shot 1)
  -> H3 Shot 2
  -> H3 Prompt Merge
       |-> immediately usable H3 prompt
       |-> enhancer request + locked plan
       `-> native route plan
              |
              v
       H3 Prompt Enhancer                (optional)
              |
              v
       H3 Apply Reference Plan           (optional native adapter)
```

References intentionally precede shots. This gives every Shot node a complete
catalog for autocomplete and validation. A Shot opens the current timeline
segment; any following Dialogue Event attaches to that Shot until the next
Shot opens. The merger closes the final shot at the global duration.

The chain should carry one versioned custom type, `MINIMAX_H3_PLAN_V2`. Media
nodes append structured facts to this plan instead of appending prose. The
workflow JSON stores the nodes and links as usual; runtime media tensors are
passed by reference inside the plan and are not serialized into the workflow.

## Why the current workflow is unreliable

### One audio option represents several incompatible intentions

The current value `Reference voice, music, beat, or sound` becomes a generic
definition and generic audio-section text. Those alternatives do not have the
same prompt behavior:

- A voice reference must identify its target speaker and use that speaker's
  stable `(Sx)` ID.
- A music-style reference belongs in the audible music layer it controls.
- A beat reference controls timing or rhythm without necessarily controlling
  instrumentation.
- A sound-effect reference belongs to a particular event or sound layer.
- Dialogue or lyric content requires exact words and the original language.
- Audio continuity is a temporal relationship, not a voice or music style.

The official guide allows different audio assets in the same workflow to have
different relationships. The current audio context rejects mixed relationships,
so it cannot express the official example of partially copied background music
plus a separate voice-timbre reference.

### The large Guide form mixes unrelated levels of work

The current Prompt Guide simultaneously asks for:

- the global creative intent;
- mode selection;
- media inventory;
- global media roles;
- retention strength;
- timing and shots;
- camera instructions;
- dialogue and visible text;
- ambience and music;
- legacy and connected-context behavior.

Several of these fields are ignored when a reference context is connected.
The user must therefore understand implementation precedence before they can
predict the result. The form also exposes broad media-type roles even though
the official format tracks the purpose of each individual asset.

### Labels are shown too late

The user writes Shot prose before having a visible, local list of the labels
created by the reference setup. They must remember whether the woman is
`<Subject 1>` or `<Subject 2>`, whether a saved image creates a Subject or a
standalone Picture, and which Audio number belongs to a voice. This is a
recognition-versus-recall failure and makes detached instructions such as
"apply Subject 1" likely.

### A model rewrite cannot repair missing semantics reliably

The Qwen enhancer can inspect connected image/video pixels, but it does not
hear standalone audio waveforms. It cannot reliably infer that an audio clip is
a particular woman's voice, a score reference, a beat source, or a copied
layer. Text metadata and explicit bindings must establish those facts before
generation.

MiniMax also describes H3 Context-IR as a multi-stage preprocessing and
orchestration system. A single free-form local LLM rewrite is useful for prose
expansion, but it is not a substitute for typed relationship setup.

## Proposed v2 nodes

### 1. MiniMax H3 Project Setup

Purpose: define only properties of the target video as a whole.

Inputs:

- `initial_prompt`: the user's global creative intent.
- `duration_seconds`: authoritative target duration, 4-15 seconds.
- `visual_style`: optional explicit override; blank derives it from intent and
  references.
- `overall_soundscape`: optional global ambience and physical sounds.
- `non_diegetic_music`: optional audience-only score; `N/A` means none.

Outputs:

- `h3_plan`: the chain context used by every following v2 node.
- `h3_length`: the native `17k+5` frame count, available early enough for
  reference-video preparation and the native conditioning node.
- `project_preview`: concise target duration and global choices.

The global duration remains authoritative. Shot cuts must fall inside it, and
the final shot implicitly continues to the effective native duration. This
removes the current timing/reference feedback-cycle problem.

### 2. MiniMax H3 Image Reference

Purpose: append one actual image and one explicit primary relationship.

Suggested top-level uses:

- `Define reusable visible content`
- `Exact first frame`
- `Exact last frame`
- `Concrete keyframe / composition anchor`
- `Storyboard / shot planning`

When defining reusable content, expose a second exact content type:

- identity or appearance;
- object, prop, clothing, interface, or visual effect;
- scene or environment;
- visual style;
- pose, expression, action, or motion.

Use a human-readable `subject_name`, such as `woman`, `truck`, or `Arizona
road`, rather than asking users to invent `content_group` identifiers. The
merger maps stable subject names to `<Subject N>`. Several assets with the same
subject name jointly define one Subject. Different names remain different
Subjects.

For the common case, these fields define one role directly on the Image
Reference node. An optional advanced `H3 Subject Binding` chain should let one
image define several separately named Subjects, or contribute several distinct
roles, without registering or connecting the native image more than once.

### 3. MiniMax H3 Video Reference

Purpose: append one actual video and its exact relationship.

Suggested uses:

- `Source video to edit`
- `Source video to continue`
- `Define reusable visible content`
- `Motion or action reference`
- `Camera, cuts, rhythm, or temporal-structure reference`

The common form defines one relationship. It accepts the same optional Subject
Binding chain when one video supplies several separately tracked people,
objects, environments, actions, or styles.

An optional paired soundtrack input should be declared explicitly. The native
route is `ref_video_audio_N`, and its `<Audio N>` label is inserted immediately
before the corresponding `<Video N>` in the native presentation order.

### 4. MiniMax H3 Audio Reference

Purpose: append one actual audio clip with one exact relationship. Do not use
an option containing "voice, music, beat, or sound".

Suggested exact uses and required metadata:

| Use | Required binding | Deterministic definition pattern |
| --- | --- | --- |
| Voice timbre and delivery | target `subject_name`, or a stable nonvisual speaker name | `<Audio N> is the voice-timbre and delivery reference for <Subject M> (Sx).` |
| Background-music style | diegetic or non-diegetic destination | `<Audio N> is the background-music style reference, guiding ... without copying the signal.` |
| Beat or rhythm | target action, edit, or music layer | `<Audio N> is the beat-and-rhythm reference for ...` |
| Sound-effect texture | named event or sound layer | `<Audio N> is the sound-effect texture reference for ...` |
| Dialogue or lyric content | exact transcript and language | `<Audio N> provides the referenced spoken/lyric content ...` |
| Audio continuity | source/target phase or continuation | `<Audio N> is the audio-continuity reference for ...` |
| Copy complete signal | no competing new audio | `<Audio N> is reused as the complete final audio track.` |
| Copy selected part/layers | exact range/layer instructions | `<Audio N> provides the copied ... layer/range.` |
| Broad inspiration | described category/atmosphere | `<Audio N> is a weak ... reference.` |

Conditional widgets should appear only for the selected use. A voice clip
should ask who owns the voice. A music reference should not show a speaker
field. Direct dialogue/lyric reuse should require a transcript because Qwen
does not inspect the waveform and the official format forbids guessing unclear
words.

Mixed audio relationships must be valid in one chain. Each `<Audio N>` keeps
its own retention marker and section placement.

The merger must canonicalize native presentation order even if users arrange
setup nodes differently: all pictures first, then each reference video's
enabled soundtrack immediately before that video, then standalone audio.
Picture, Video, and Audio numbering remains independent.

### 5. MiniMax H3 Shot

Purpose: append one shot in playback order.

The node receives `h3_plan`, so it already knows every upstream reference. It
does not ask the user to type `[Shot N]`; numbering follows chain order.

Inputs:

- `cut_at`: hidden or fixed at `0.000` for the first shot; later shots use one
  float cut time. The next cut or global duration determines the shot's end.
- `description`: only the visible/audible events in this shot.
- `camera_direction`: optional natural camera motion.
- `transition`: direct cut by default, with explicit dissolve/fade/wipe options.

Using one cut time per shot matches the official output format and removes the
redundant start/end pair that currently creates gaps and overlaps.

#### `<` reference autocomplete

When the user types `<` in the Shot description, a popup should list the labels
already available from the upstream setup chain, for example:

```text
<Subject 1>  Woman - identity from Picture 1
<Subject 2>  Truck - object from Picture 2
<Video 1>    Source video to edit
<Audio 1>    Woman's voice - bound to Subject 1
```

The menu should filter while typing, support arrow keys, Enter/Tab, and Escape,
and insert only the selected H3 label. It may also offer an explicit speaker
snippet such as `<Subject 1> (S1)` after speaker IDs have been resolved.

The frontend can implement this through ComfyUI's JavaScript extension hooks
and the multiline STRING widget's input element. It can traverse the connected
upstream chain, as the existing route-label extension already does. Backend
validation must remain authoritative so API-mode workflows and manually typed
labels are equally safe.

### 6. MiniMax H3 Dialogue Event (optional)

Purpose: preserve exact words and assign speaker IDs from actual playback order
instead of guessing them from prose.

One or more Dialogue Event nodes may feed a Shot. Each event contains:

- `speaker`: an upstream `subject_name` or stable nonvisual speaker name;
- `language`;
- `exact_text`;
- `delivery`;
- `voice_mode`: on-screen speech, off-screen speech, or voiceover;
- optional timing/continuity choices for `<scenetrans>` and `<cutoff>`.

The merger assigns `(S1)`, `(S2)`, and later IDs by the order of these actual
vocal events across the completed Shot chain. It then resolves any Audio
Reference bound to that speaker and writes the same `(Sx)` in the Audio
definition. This is necessary because the official guide explicitly forbids
assigning speaker IDs independently in audio definitions.

The free-form Shot description remains available for visual action and simple
expert-authored dialogue, but structured events are the recommended route when
voice references, multiple speakers, voiceover, dialogue across cuts, or exact
word preservation matter.

### 7. MiniMax H3 Prompt Merge

Purpose: consume the completed plan and produce the H3 prompt package.

Outputs:

- `h3_prompt`: deterministic pre-LLM prompt with the correct three- or
  six-section structure.
- `rewrite_request`: self-contained text for an external LLM.
- `plan_context`: authoritative structured data for the bundled enhancer.
- `problems_report`: only actionable conflicts, limits, missing bindings, and
  native routes.
- `h3_length`: the same native frame count created by Project Setup.

The merger automatically chooses T2VA, I2VA, FL2VA, L2VA, or Ref2VA from the
actual reference relationships. It should never create a generic Subject just
because Ref2VA is active.

For Ref2VA it deterministically creates:

1. `subject_definitions`
2. `summary`
3. `retention_analysis`
4. `detailed_description`
5. `overall_soundscape`
6. `non_diegetic_music`

The enhancer may expand descriptions, but labels, subject grouping, speaker
bindings, task types, retention markers, routes, and shot cut times are locked.

### 8. MiniMax H3 Prompt Enhancer

Purpose: improve only the descriptive fields that benefit from a language or
vision-language model.

The enhancer should receive both the draft prompt and the structured locked
plan. Its editable fields are limited to visual description, action clarity,
camera prose, transitions, ambience, and music description. It must not invent
or renumber references, change retention, assign speakers, change exact
dialogue, or move cut times.

The safest contract is structured output rather than asking the model to
rewrite the complete six-section prompt. The model returns enhanced global and
per-shot prose; the node validates that response and reconstructs the final H3
prompt from the locked plan. If parsing or semantic validation fails, the node
returns the valid pre-LLM prompt and an actionable warning instead of emitting
a corrupted prompt.

Inputs and outputs should preserve the existing editing workflow:

- optional `system_prompt` override input;
- optional side-node tail input;
- `base_system_prompt` output, so the built-in contract can be inspected or
  edited;
- `system_prompt_used` output, including the applied override/tail;
- `enhanced_h3_prompt` and `enhancer_warnings` outputs.

Qwen3-VL can use connected images and sampled video frames to improve visible
details and temporal descriptions. It is not an audio-understanding path in
this workflow. Voice ownership, transcript, language, copied layers, timing,
and other audio semantics must therefore arrive as text from the typed Audio
Reference and Dialogue nodes.

### 9. MiniMax H3 Apply Reference Plan (optional native adapter)

Purpose: eliminate the remaining manual synchronization between prompt labels
and the native H3 conditioning sockets.

The adapter accepts the compiled plan, final prompt, CLIP, VAE, optional audio
VAE, and target size. It presents media to H3 in the native canonical order and
returns conditioning and the empty latent. It should use a narrow compatibility
layer around ComfyUI's native H3 implementation and fail with a version-specific
message if that API changes.

This node is important for the best end-to-end UX, but it is not required for
the first compiler milestone. Until it is implemented, each Reference node can
provide a media pass-through output named with its resolved route and Prompt
Merge can expose the same route table for connection to the native node.

Keeping the adapter separate from Prompt Merge has three benefits:

- prompt planning and validation still work without H3 weights loaded;
- users can keep the official native conditioning node when they prefer it;
- compatibility changes in ComfyUI do not destabilize the semantic compiler.

## Example: woman with a voice reference

Setup chain:

```text
Project Setup
-> Image Reference
     use: Define reusable visible content
     content type: Identity or appearance
     subject name: woman
-> Audio Reference
     use: Voice timbre and delivery
     target subject: woman
-> Dialogue Event (connected to Shot 1)
     speaker: woman
     language: English
     exact text: Exact user-provided dialogue.
-> Shot 1
     description: As the truck moves, <Subject 1> sits beside the driver...
-> Prompt Merge
```

The merger should produce the exact relationship:

```text
<Subject 1> is the woman whose identity and appearance come from <Picture 1>.
<Audio 1> is the voice-timbre and delivery reference for <Subject 1> (S1).
```

When she speaks in a shot:

```text
<Subject 1> (S1) speaks using the voice timbre and delivery referenced from
<Audio 1>: <d>[English] Exact user-provided dialogue.</d>
```

No wording about music, beat, generic sound, or copying the source words should
appear for this setup.

## Validation and error prevention

The v2 merger should reject or clearly diagnose:

- audio as the only Ref2VA input;
- missing voice target;
- a voice target that has no matching vocal event, or a vocal event whose
  speaker has no resolvable subject/nonvisual speaker identity;
- dialogue/lyric-content reference without transcript and language;
- partial copy without a declared time range or layer;
- exact endpoint roles mixed into a Ref2VA native route;
- unknown labels typed in a Shot;
- a referenced label that never appears where its scoped role is needed;
- cut times that are not increasing or fall outside the global duration;
- unsupported media counts/durations or total file count;
- native media order that differs from the assigned prompt label order;
- complete audio copy combined with newly requested dialogue, effects, ambience,
  or music.

The report should state the fix in user language, for example: `Audio 1 is set
to Voice, but no target subject is selected. Choose Woman or another speaker.`

## Compatibility strategy

Build the v2 chain under new internal node IDs so existing workflows continue
to load. Mark the current Prompt Guide, Target Timing, Visual Reference Role,
and context-builder nodes as legacy after the v2 path reaches feature parity.
Do not make the old large form the recommended entry point.

Keep Reference Sheet persistence compatible. Its job is to save, load, preview,
and output selected media; the new Image/Audio Reference nodes assign the
workflow-specific semantics after loading.

## Architectural alternatives

### Plan A - ordered semantic spine (recommended)

All project, reference, shot, and dialogue nodes consume and return one plan.
References must be configured before the first Shot. Prompt Merge is a pure,
deterministic compiler, and an optional adapter performs native conditioning.

Strengths:

- lowest practical wire count without hiding the workflow;
- a complete upstream catalog is available in every Shot;
- order of shots and vocal events is visible directly on the canvas;
- works in API mode because the backend plan is authoritative;
- supports simple cases with one field per node and advanced bindings only when
  requested;
- one validation pass can check prompt labels, scopes, timing, and native routes
  together.

Tradeoffs:

- adding a new reference requires inserting it before the first Shot;
- changing an early node invalidates downstream execution cache, as expected
  for a compiled plan;
- the optional native adapter needs a maintained ComfyUI compatibility layer.

### Plan B - parallel specialist chains feeding a catalog hub

```text
Image Reference chain --\
Video Reference chain --- H3 Reference Catalog -> H3 Shot chain -> Prompt Merge
Audio Reference chain --/                         ^
Project Setup ------------------------------------|
```

This keeps media categories independent and is the most conservative extension
of the current code. It is a reasonable fallback if one-plan chaining proves
incompatible with third-party nodes.

Strengths:

- category-specific chains can be edited and tested independently;
- less media is carried through unrelated setup nodes;
- the compiler remains headless and deterministic.

Tradeoffs:

- more wires and collector nodes;
- audio-to-subject bindings cross chain boundaries;
- the Shot chain needs both project and reference-catalog context;
- precedence and partial-connection states are harder for users to understand;
- it trends toward the current collection of context sockets rather than the
  requested setup flow.

### Plan C - one structured H3 Composer UI

A custom node or side panel would contain tabs for Project, References, Shots,
Dialogue, and Validation, with sortable media cards and timeline rows. This can
be the smoothest novice interface when it is complete.

Strengths:

- almost no canvas wiring;
- drag-and-drop ordering, inline previews, and immediate validation are easy to
  present in one surface;
- arbitrary references and shots can be managed like a small editor.

Tradeoffs:

- highest implementation and maintenance cost;
- duplicates ComfyUI graph, widget, serialization, and media-management
  behavior inside one custom UI;
- weaker API/headless ergonomics unless a separate schema API is maintained;
- repeats the monolithic-node problem the redesign is intended to remove;
- a frontend failure can make the complete workflow difficult to edit.

It should not be the first implementation. It can later be built as an App-mode
or composer view over exactly the same `MINIMAX_H3_PLAN_V2` schema, so it does
not create a second prompt engine.

### Comparison

| Criterion | Plan A: semantic spine | Plan B: parallel hub | Plan C: composer UI |
| --- | --- | --- | --- |
| Semantic safety | High | High | High if schema is duplicated correctly |
| Canvas clarity | High | Medium-low | Very high |
| Wire count | Low | High | Very low |
| Arbitrary refs/shots | High | High | High |
| API/headless use | High | High | Low-medium |
| Inspectability | High | High | Medium |
| Frontend complexity | Medium | Low-medium | Very high |
| Native-route automation | High with adapter | Medium | Medium-high |
| Fit with requested workflow | Best | Acceptable fallback | Poor first step |

### Distribution layer - presets, subgraphs, and App mode

This complements Plan A rather than replacing it. Ship tested templates or
subgraph blueprints for common tasks:

- text-only;
- one identity plus one voice;
- an object appearing only in later shots;
- source-video edit with copied music and a new voice reference;
- motion transfer;
- exact first/last-frame completion.

ComfyUI can package reusable subgraph blueprints, and App mode can expose only
selected controls for a fixed template. These are useful onboarding layers,
but arbitrary reference counts and shot chains should remain editable as normal
nodes.

## Plan schema and invariants

The plan should use stable internal IDs and human aliases. H3 labels are derived
only during compilation so inserting a new reference cannot silently retarget a
stored numeric label.

```text
project
  initial_prompt, duration, style, soundscape, music
assets[]
  asset_id, media_kind, media_value, source_name, primary_relationship
bindings[]
  binding_id, asset_id, subject_alias, exact_role, retention, shot_scope
audio_relationships[]
  asset_id, exact_use, target_alias/layer/event, transcript, language, retention
shots[]
  shot_id, cut_at, description, transition, referenced_binding_ids
dialogue_events[]
  shot_id, sequence, speaker_alias, language, exact_text, delivery, voice_mode
```

Required invariants:

1. Project is first and Prompt Merge is last.
2. All media and bindings precede the first Shot.
3. A Dialogue Event follows the Shot it belongs to.
4. Subject and speaker references use internal IDs/aliases, never stored
   `<Subject N>` numbers.
5. Native labels are assigned in canonical category order at compile time.
6. Speaker IDs are assigned from actual vocal-event order at compile time.
7. A numeric `shot_scope`, such as `3,4` or `3-4`, never asks the user to
   rewrite "Shot 3" in prose. The Shot editor's reference picker inserts the
   corresponding label at the intended entity/action; the merger validates
   that scoped references are actually used.
8. The compiler never creates a Subject merely because Ref2VA is selected.
9. The enhancer cannot mutate any of these invariants.

## Concrete interaction details

- Each reference node displays a live badge such as `Picture 2 - truck` or
  `Audio 1 - woman's voice -> woman`. Backend output is authoritative; the
  badge is a preview.
- Selecting an audio role changes the visible fields. `Voice` shows speaker and
  delivery; `Music style` shows destination/layer; `Partial copy` shows range
  and layer. No field or tooltip contains an ambiguous list joined by "or".
- Subject selectors and `<` completion use upstream aliases, so users choose
  `Woman - Subject 1` rather than remembering a number.
- Unknown or out-of-scope references are errors before sampling, with a direct
  fix rather than a large diagnostic dump.
- The default report is a short readiness summary. Full generated definitions,
  route order, and retention analysis remain available on advanced outputs.
- Bypass behavior is explicit: a bypassed reference disappears from the catalog
  and labels are recomputed; stale labels in Shot prose are then reported.
- The legacy nodes remain loadable but are clearly labeled `Legacy`; there is
  no automatic positional widget migration into semantically different v2
  nodes.

## Why standard ComfyUI nodes plus a small frontend extension are enough

Use the V3 node schema for all new nodes. `DynamicCombo` is suitable for the
role-specific Image, Video, and Audio fields. Standard repeatable inputs are
not suitable for a compound Dialogue row because an Autogrow template cannot
itself be a dynamic compound input in the current API. Separate chainable
Dialogue Event nodes therefore give better serialization and headless behavior
than a custom repeated form.

Custom JavaScript is limited to conveniences that cannot be expressed by the
backend schema:

- upstream alias discovery and `<` completion;
- live label/route badges;
- first-Shot cut-time hiding and computed range preview;
- recursive traversal through reroutes, bypassed nodes, and subgraphs.

Every frontend-derived result is independently recomputed and validated by the
Python compiler.

## Phased implementation plan

### Phase 1 - deterministic compiler MVP

1. Define `MINIMAX_H3_PLAN_V2` as plain versioned data plus pure builder,
   compiler, and validator functions.
2. Add Project Setup, exact Image/Video/Audio Reference nodes, Shot, Dialogue
   Event, and Prompt Merge using ComfyUI's V3 schema.
3. Implement canonical media numbering, speaker ordering, mode selection, six-
   section generation, strict validation, and concise previews.
4. Cover the official task families and limits with golden tests before adding
   model inference.

This phase is already useful: it produces a valid prompt and route table with
no LLM and leaves all current nodes operational.

### Phase 2 - interaction and enhancement

1. Add conditional role widgets, upstream subject selectors, live badges,
   numeric shot-scope parsing, and `<` autocomplete.
2. Adapt Prompt Enhancer to consume the locked plan and return structured
   editable prose; reconstruct and validate the final prompt afterward.
3. Keep the side-node tail input and expose both base and effective system
   prompts.
4. Add image/video analysis selectively; describe audio only from explicit
   metadata and transcripts.

Implemented on `dev/reference-workflow-ux-phase2`. Qwen receives the complete
compiled scene and reference inventory in one request, optionally accompanied
by visual evidence. Its response is restricted to a versioned prose JSON
surface; Python recompiles the H3 document and compares all semantic locks
before accepting it. Invalid responses fall back to the deterministic draft.

### Phase 3 - end-to-end integration and onboarding

1. Add the optional H3 Apply Reference Plan adapter and compatibility tests
   against the current native H3 node.
2. Ship templates/subgraphs for the acceptance cases and an App-mode preset for
   the simplest workflows.
3. Mark the existing monolithic/context nodes as legacy in the node library and
   document a manual migration map.
4. Consider the Composer UI only after the plan schema and adapter have proven
   stable in real workflows.

Implemented on `dev/reference-workflow-ux-phase3`. Apply Reference Plan verifies
that the final prompt and compiled context are the same package, translates the
compiler's canonical routes into the installed native H3 call, and delegates to
ComfyUI's own conditioning implementation. The compatibility layer checks both
native signatures without loading weights. Workflow templates, an APP-mode
prompt preset, direct saved-audio selection, Legacy library labels, and a manual
migration map complete the onboarding layer. A Composer UI remains deliberately
deferred until this graph workflow has real-world use.

## Acceptance cases

The implementation is ready only when golden tests cover at least:

- text-only generation;
- exact first frame, exact last frame, and first-plus-last endpoints;
- one image used only for a person identity, with no standalone Picture row;
- one item reference appearing only in Shots 3 and 4;
- one image defining multiple visible subjects without duplicate native media;
- several images jointly defining one Subject;
- a voice clip bound to a referenced woman, with no generic "or" wording;
- mixed partially copied music and separate voice timbre, matching MiniMax's
  published Ref2VA example;
- music-style, beat, sound-effect, continuity, transcript, full-copy, and
  partial-copy audio paths;
- video editing, continuation, motion transfer, and temporal-structure use;
- Shot autocomplete through normal links, reroutes, and bypassed setup nodes;
- API-mode backend validation without relying on the browser extension;
- every official media-count, duration, and ordering constraint.

## Research sources

- [MiniMax H3 official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  and its explanation of H3 Context-IR and media limits.
- [MiniMax H3 full-reference prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
- [MiniMax H3 base prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).
- [ComfyUI's native MiniMax H3 implementation](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
  and its fixed media-presentation order.
- [Official ComfyUI MiniMax H3 Ref2VA template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json).
- [ComfyUI V3 node schema](https://docs.comfy.org/custom-nodes/v3_migration),
  [JavaScript extensions](https://docs.comfy.org/custom-nodes/js/javascript_overview),
  [subgraphs](https://docs.comfy.org/interface/features/subgraph), and
  [subgraph blueprints](https://docs.comfy.org/custom-nodes/subgraph_blueprints).
- [Qwen3-VL-32B-Instruct official model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
  for image/video comprehension scope.
- [Nielsen Norman Group's usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/),
  especially visibility of status, match with user language, error prevention,
  recognition rather than recall, and minimalist design.


---

# ISSUES / PRs of ethanfel/ComfyUI-MiniMax-H3-Guide


### #12 [open] ISSUE: getting stutter frames in the overlap portion
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/12  by DalvoGalbo 2026-08-30T16:04:37Z

while continuing the video from previous generation i'm feeding 22 or 39 frames to the next cycle but i get stutter and flicker frames for that overlap portion in the beginning of the video and then the rest of the video is fine, it probably affects a couple of more frames around that portion too and it prevents from getting a seamless transition. otherwise this is really a helpful node, must have for long videos. is there any possible fix for this issue?

> **Comment by bingothreed (2026-10-14T05:12:49Z):**

Met the same issue



### #11 [open] ISSUE: # Prompt Review Gate rejects the compiler's own output when prompt_style is "Compact low-token prompt" (character replacement)
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/11  by andy91121313 2026-08-22T14:42:15Z

**Version:** 0.15.0
**ComfyUI:** v0.33.0-40-g783545f6
**Python:** 3.13.14 / torch 2.13.0+cu130 / Windows 11

## Summary

With a Character Replacement in the plan and `Prompt Merge -> prompt_style = "Compact low-token prompt"`,
`MiniMaxH3PlanV2PromptReview` always fails with:

```
ValueError: [Shot 1] must keep the compiler-managed character-replacement instruction unchanged.
```

This happens even when nothing is edited — the timeout path validates the compiler's own output
against itself.

## Root cause

There are two builders for the per-shot replacement instruction in `plan_v2.py`:

| builder | used by |
| --- | --- |
| `_replacement_shot_instruction()` (line 1684) | `_detailed_description()` (line 2685) → **Full structured prompt** |
| `_compact_replacement_shot_instruction()` (line 1759) | `_compact_detailed_description()` (line 2767) → **Compact low-token prompt** |

`compile_h3_plan` selects between them at `plan_v2.py:3248-3250`:

```python
_compact_detailed_description(validated, catalog)
if compact
else _detailed_description(validated, catalog)
```

But the validator in `prompt_review.py:_validate_detailed_section` (line ~285) only ever checks for
the **verbose** variant:

```python
for replacement in plan["character_replacements"]:
    if number not in _replacement_scope(replacement, plan):
        continue
    locked = _replacement_shot_instruction(replacement, plan, catalog)   # <-- always verbose
    if locked not in candidate:
        raise ValueError(
            f"[Shot {number}] must keep the compiler-managed character-replacement "
            "instruction unchanged."
        )
```

So in compact mode the shot text contains the compact string, the validator looks for the verbose
string, and the check can never pass.

## Why it is clearly a self-inconsistency, not user error

`pass_through_after_timeout` (`prompt_review.py:979`) calls:

```python
approve_reviewed_prompt(plan_context, source, source, approval_source="Prompt review timeout")
```

i.e. it passes the canonical prompt as **both** the canonical and the reviewed text. The preceding
guard (`if source != _normalized_prompt(canonical)`) already confirmed the pair matches. The
compiler's own untouched output therefore fails its own validator.

## Reproduce

1. Open the shipped example `MiniMax H3 Plan v2 - Character Replacement.json`
   (it ships with `prompt_style = "Compact low-token prompt"`).
2. Connect any image + any 2-15s video, fill `source_character_description`.
3. Queue. Let the Prompt Review Gate time out (or approve without editing).
4. → `[Shot 1] must keep the compiler-managed character-replacement instruction unchanged.`

## Suggested fix

Have the validator pick the same builder the compiler used, e.g. read the style off the plan:

```python
builder = (
    _compact_replacement_shot_instruction
    if plan.get("prompt_style") == PROMPT_STYLE_COMPACT
    else _replacement_shot_instruction
)
locked = builder(replacement, plan, catalog)
```

or accept either variant:

```python
if not any(
    b(replacement, plan, catalog) in candidate
    for b in (_replacement_shot_instruction, _compact_replacement_shot_instruction)
):
    raise ValueError(...)
```

The same divergence is worth auditing for `_shot_scoped_reference_instructions`, which is shared by
both description builders but may be embedded differently.

## Workaround

Set `Prompt Merge -> prompt_style = "Full structured prompt"`, or bypass the Prompt Review Gate.



### #10 [open] ISSUE: how bypas this
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/10  by peepeepeepoopoopoo 2026-08-19T18:59:11Z

workflow: [workflow.json](https://github.com/user-attachments/files/31235767/workflow.json)

<img width="549" height="411" alt="Image" src="https://github.com/user-attachments/assets/1e7caf8e-e628-4703-b9e9-700f7ab87b90" />




### #9 [open] ISSUE: Prompt Review Gate rejects its own canonical prompt when prompt_style is "Compact low-token prompt" and the plan has a character replacement
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/9  by gregowahoo 2026-08-16T12:28:17Z

### Summary

When `MiniMax H3 Prompt Merge` is set to `prompt_style = "Compact low-token prompt"` and the plan contains a character replacement, `MiniMaxH3PlanV2PromptReview` can never be satisfied. It rejects the **unedited** canonical prompt it was handed, so the workflow cannot proceed by approving, by letting the review time out, or by switching the gate to `Pass through without pausing`.

```
ValueError: [Shot 1] must keep the compiler-managed character-replacement instruction unchanged.
```

Version 0.15.0 (`ae0789c`), ComfyUI 0.33.0, Python 3.13.

### Cause

The compiler emits two different wordings for the locked replacement instruction:

- `_detailed_description` → `_replacement_shot_instruction` (`plan_v2.py:2729`)
  *"Using `<Video 1>` as the source timeline, replace only the source performer described as …"*
- `_compact_detailed_description` → `_compact_replacement_shot_instruction` (`plan_v2.py:2806`)
  *"Use `<Video 1>` frame-for-frame and replace only …"*

The review gate always regenerates the **full** wording and requires it as a literal substring:

```python
# prompt_review.py:285
locked = _replacement_shot_instruction(replacement, plan, catalog)
if locked not in candidate:
    raise ValueError(...)
```

In compact style that string is never present, so the check cannot pass. `pass_through_after_timeout` and the `Pass through without pausing` branch both route through `approve_reviewed_prompt(plan_context, source, source)`, which runs the same validator — hence there is no way past the node short of changing `prompt_style` back to `Full structured prompt`.

### Reproduction

Using the shipped `example_workflows/MiniMax H3 Plan v2 - Character Replacement.json` widget values, compiling once per style and approving the untouched prompt:

```
'Full structured prompt': OK (unedited canonical prompt approved)
'Compact low-token prompt': FAIL -> [Shot 1] must keep the compiler-managed character-replacement instruction unchanged.
```

### Suggested fix

Lock whichever instruction the plan actually compiled. `compile_h3_plan` already stores `prompt_style` on the compiled plan, so it is available in the validator:

```python
locked = (
    _compact_replacement_shot_instruction(replacement, plan, catalog)
    if plan.get("prompt_style") == PROMPT_STYLE_COMPACT
    else _replacement_shot_instruction(replacement, plan, catalog)
)
```

plus `PROMPT_STYLE_COMPACT` and `_compact_replacement_shot_instruction` added to the two `plan_v2` import blocks at the top of `prompt_review.py`. With that change both styles approve the unedited prompt, and both still reject edits to the replacement instruction.

Worth noting the test suite has no compact-style coverage for the gate — `tests/test_prompt_review.py` builds every fixture at the default `Full structured prompt`, which is why this slipped through.


### #8 [open] ISSUE: Character replacement Issue
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/8  by gayrardjean34-dot 2026-08-13T12:16:57Z

Hi, I was really hiped by your character replacement workflow and nodes. I tried several times and i can't get the desired results.

The main problem is that your workflow use the clothes from the reference picture, and even if i give it a close up portrait without clothes visibles, it will hallucinate new clothes, or different ones. is there a way to jsut keep the clothes from original video, and only edit the face + hair, or full body ? 
I can't get it to work perfectly, sometimes, on a lucky try, it works, but then the identity looks more merged than transfered. The identity is also a problem, which is not consistent from what i Tried. But I'm pretty sure it's possible to get it good, because i was trying to rebuild myself a workflow with your nodes, and saw things like " targetsubject for motion transfer " in the h3 prompt guide legacy, which is usingthe identity perfectly, with a face swapped picture of the first frame, but then the motion is not respected and start hallucinate and droft things from original.

If you figured out how to get a consistent way to do motion control on h3, I would be very grateful man


### #7 [closed] ISSUE: prompt for image to video???
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/7  by kiwiqq-jpg 2026-08-10T03:33:24Z

How I would connect this on the default comfyui template for IMAGE TO VIDEO?
complicated and with no documentation on how to prompt.... 
All of this actually makes something that should be simple unnecessarily complicated... Since the official guide itself demonstrates how easy it is, I thought the node would generate the format based on a sentence, but it’s actually something completely different... I think it’s better to just copy the official guide paste it into any LLM and call it a day.


> **Comment by ethanfel (2026-08-10T04:14:41Z):**

yes do that



### #6 [closed] ISSUE: Licence declaration?
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/6  by c-chienads 2026-08-09T11:24:48Z

Thanks for this node — it is genuinely useful.

I noticed the repository does not currently include a `LICENSE` file. Without an explicit licence, the default position is that all rights are reserved, which makes it hard for teams to adopt it even for internal use.

Would you consider adding a licence declaration (e.g. Apache-2.0, MIT, or GPL-3.0 — whichever fits your intent)? Even a one-line statement in the README would help a lot.

No pressure either way, and thanks again for sharing the work.

> **Comment by ethanfel (2026-08-09T19:16:58Z):**

Hello, you're right. Added GPL-3.0 license



### #5 [open] ISSUE: Request: First/Last + Referencce Audio and Identity
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/5  by moonwhaler 2026-08-09T09:18:53Z

I really like your approach and the provided workflows help a lot! One thing missing seems to be the possibility to include a first/last frame to the identity / ref audio as I get indentity drift quite often using only first/last. At least I am not able to do that.

> **Comment by ethanfel (2026-08-09T19:16:05Z):**

Hi, I didn't work on this yet, too much to do and explore with h3 :D



### #4 [closed] ISSUE: get_file_handle问题的幽灵又出现了
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/4  by tim3385 2026-08-07T15:25:18Z

我的comfyui是v0.30.2 + 15 commits(15989f8), PyTorch v2.10.0+cu130，使用最新的工作流见附件。

[video_minimax_h3_i2v_uncensored_enhancer.json](https://github.com/user-attachments/files/30834964/video_minimax_h3_i2v_uncensored_enhancer.json)

模型也安装好，使用你作为custom nodes，comfy-aimdo使用得是8月6日的最新版0.4.13。点运行后就报错

# ComfyUI Error Report
## Error Details
- **Node ID:** 105:13
- **Node Type:** CLIPLoader
- **Exception Type:** AttributeError
- **Exception Message:** AttributeError: 'ModelMMAP' object has no attribute 'get_file_handle'

## Stack Trace
```
  File "C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)

  File "C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)

  File "C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)

  File "C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\nodes.py", line 1024, in load_clip
    clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)

  File "C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\sd.py", line 1458, in load_clip
    sd, metadata = comfy.utils.load_torch_file(p, safe_load=True, return_metadata=True)

  File "C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\utils.py", line 149, in load_torch_file
    raise e

  File "C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\utils.py", line 129, in load_torch_file
    sd, metadata = load_safetensors(ckpt)

  File "C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\utils.py", line 90, in load_safetensors
    f = model_mmap.get_file_handle()

```
## System Information
- **ComfyUI Version:** 0.30.0

> **Comment by ethanfel (2026-08-07T21:56:47Z):**

你好，我检查了这个堆栈和对应版本。这个异常发生在 ComfyUI 核心的 `CLIPLoader -> comfy.utils.load_safetensors()` 中，还没有执行到本项目的自定义节点。

你使用的 ComfyUI 提交 `15989f8` 在 `requirements.txt` 中要求 `comfy-aimdo==0.4.13`，而官方 v0.4.13 的 `ModelMMAP` 确实包含 `get_file_handle()`。因此当前进程实际加载的很可能是旧的或重复安装的 `comfy_aimdo` 文件；常见原因是使用了另一个 Python 执行 pip，或者升级后没有完全重启。

请先完全关闭 ComfyUI，然后使用启动日志里 **Python executable** 显示的那个 Python 路径执行：

```powershell
& "<COMFY_PYTHON>" -c "import comfy_aimdo, comfy_aimdo.model_mmap as m, importlib.metadata as md; print(md.version('comfy-aimdo')); print(comfy_aimdo.__file__); print(m.__file__); print(hasattr(m.ModelMMAP, 'get_file_handle'))"
```

最后一行预期应为：

```text
True
```

如果显示 `False`，请仍然使用同一个 Python 强制重装：

```powershell
& "<COMFY_PYTHON>" -m pip install --no-cache-dir --force-reinstall comfy-aimdo==0.4.13
```

然后完全重启 ComfyUI 再测试。

如果仍然报错，请把上面诊断命令的完整输出，以及启动日志中的 **Python executable** 一并贴出来。输出中的模块路径可以直接确认是否存在多个 Python 环境或残留包。如果下一步出现 `undefined symbol: model_mmap_get_file_handle`，那也是 Python 文件与 AIMDO DLL 版本不一致，同样需要在正确的 ComfyUI Python 环境中强制重装。



> **Comment by tim3385 (2026-08-09T00:24:45Z):**

是的，我后来才注意到comfyui有自己的python venv，包更新到正确的venv里就解决了。



### #3 [closed] ISSUE: Sample Workflow?
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/3  by davoodkharmanzar 2026-08-04T16:34:26Z

it would be great if add some sample ComfyUI workflows :]

> **Comment by ethanfel (2026-08-05T05:50:16Z):**

it's a WIP, I didn't expect that node to get any attention but because of the minimax guy a lot of people has seen the ablit TE and this repo



### #2 [closed] ISSUE: 能分享下完整的工作流吗!
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/2  by Chadli2022 2026-08-04T16:05:10Z

能分享下完整的工作流吗!这么多节点，要去理解，自己搭建比较困难。

> **Comment by ethanfel (2026-08-05T05:53:46Z):**

it's a WIP, node will change completely soon. It's not even in comfyui manager,



### #1 [closed] ISSUE: MiniMax H3 Generation Tail Loader - creepingly slow on any system - regardless of ressources
URL: https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide/issues/1  by entropy010 2026-08-03T23:58:44Z

Hello ethanfel,

First and foremost. Thank you for this cool project. This is immensely powerful.
That beeing said, I guess you already know this, with the tail loader active - computation takes a very long time (~20 minutes on 80-100 words input in the manual promt).

Tested on:
4090 24Gb Vram 98% allocated / 64Gb Ram 75% allocated / Win 11 / torch 2.9.1+cu128 / python 3.10
5090 reported the same issue.
Node settings were left as is and were not changed.

Since there is no true error, hopefully the specs help identifying the performance hit.

Thank you again,
Cheers!

> **Comment by MasterToycode (2026-08-06T09:29:41Z):**

my own machine—a 5O70ti （16GB）, though a 15-second 640×832 video takes about 20 minutes.


> **Comment by ethanfel (2026-08-07T03:10:11Z):**

I checked several time and Ican't find anything to do. The code used is the native llm code of comfyui and it's just slow. 

