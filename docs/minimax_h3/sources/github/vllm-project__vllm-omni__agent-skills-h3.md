# vllm-project__vllm-omni__agent-skills-h3

Source: https://github.com/vllm-project/vllm-omni/tree/main/.agents/skills
Fetched: 2026-11-28



---

## FILE: .agents/skills/H3-SOURCES.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/.agents/skills/H3-SOURCES.md)

# MiniMax H3 skill provenance

Upstream: https://github.com/MiniMax-AI/MiniMax-H3

Source snapshot: `d21241f0a4b3acbb34c97dae47fa417b7065e438`, retrieved 2026-09-19.

Source directory:
https://github.com/MiniMax-AI/MiniMax-H3/tree/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills

The nine upstream skill names are retained. Each local skill corresponds to the
same-named upstream `skills/<name>/` directory. Credit for the original creative
workflows and templates belongs to MiniMax-AI and the upstream contributors.

## Files and adaptation

- `h3-prompt-writing/references/base-format.md` and `ref-format.md` are new,
  condensed descriptions of the public prompt interface. They preserve necessary
  field names and notation but do not copy the upstream guides or their examples.
  The entrypoint adds a portable execution reference and clarifies clip/window
  duration for verified long-video backends.
- The eight style-specific `SKILL.md` files are rewritten portable adaptations of
  the upstream workflows. Their reference documents retain the relevant creative
  methods while removing host-specific execution assumptions.
- The two `co-op-game-intro-generator/references/` templates retain the upstream
  creative structure with an added portable contract and explicit user overrides.
  Their instructions are normalized to English, and the video template expresses
  visible UI copy as localized variables with English defaults. It also removes
  an unrequested gender restriction and a fixed yellow/black environment
  transformation.
- The five 3D animation references are condensed adaptations of their upstream
  counterparts: shot table, storyboards, model selection, fallback, and review.
- Product, papercraft, collage, and music references extract reusable direction
  from their corresponding upstream skill. The shared portable workflow, catalog,
  and UI metadata are additions for this repository.

No upstream `LICENSE`, `COPYING`, or `NOTICE` file was present in the Git tree at
the pinned snapshot. Its README links to the MiniMax H3 Community License on the
model release. Because that license has redistribution conditions, this package
does not include the upstream prompt guides verbatim. This provenance record
preserves attribution and does not assert that this repository's code license
relicenses upstream material.

## Runtime changes

Canvas documents become local files and asset manifests; choice-card UI becomes
the host's normal clarification mechanism when a decision is actually missing.
Available generation/editing tools replace proprietary dispatch calls. Previous
user choices and authorization persist across steps. Providers, prices, resolution
support, and long-video features are checked rather than hardcoded as guarantees.

User intent overrides style presets, including instrumental/no-text work, selected
language, final morph shape, no-cut sequences, and custom player designs. Native
generation and external processing remain separately identified in delivery.


---

## FILE: .agents/skills/h3-prompt-writing/SKILL.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/.agents/skills/h3-prompt-writing/SKILL.md)

---
name: h3-prompt-writing
description: Write MiniMax H3 video generation prompts for T2VA, I2VA, FL2VA, L2VA, and Ref2VA. Use when rewriting multimodal requests into H3 prompt structures, composing integrated_multimodal_description, overall_soundscape, and non_diegetic_music, aligning keyframes, or defining reference labels for images, videos, and audio.
metadata:
  compatibility: Portable to any agent that can read local files — no external API calls, MiniMax Hub tools, or proprietary runtime required. The agents/openai.yaml file only adds optional ChatGPT/Codex UI metadata; it does not restrict the skill to OpenAI agents.
---

# H3 Prompt Writing

## Workflow

0. Read [portable-workflow.md](references/portable-workflow.md) for scope, available tools, durable artifacts, and generation handoff. For prompt-only work, no generation tools are required.
1. Identify the input mode: T2VA, I2VA, FL2VA, L2VA, or full-reference Ref2VA.
2. For base text/keyframe modes, read `references/base-format.md` and follow its final prompt structure.
3. For full-reference mode, read `references/ref-format.md` and follow its six-section rewrite format.
4. Preserve the exact field names, section order, labels, and timing notation from the selected guide.

## Base Modes

- T2VA: build the full audiovisual timeline from text.
- I2VA: start from the first frame and develop forward from it.
- FL2VA: describe the continuous path between the first and last frames.
- L2VA: infer a plausible opening and converge to the supplied last frame.

Use `integrated_multimodal_description`, `overall_soundscape`, and `non_diegetic_music` in the order shown in `references/base-format.md`.

## Full-Reference Mode

Ref2VA rewrites use `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and `non_diegetic_music` in that order. Reference labels stay consistent across all sections.

Read `references/ref-format.md` for label rules and retention analysis.

## Output Rules

- Write rewrite sections in English; preserve dialogue, lyrics, and visible scene text in their original language.
- Describe each shot by composition, subjects, environment, actions, camera, sound, and the exact point where referenced content appears.
- Avoid plot summaries, unresolved reference labels, and timing that does not match the requested duration.
## Tips for Better Results
- Match the description to the effective duration of the current clip or generation window. The upstream guides describe 4–15-second clips; longer work requires a verified backend extension or an explicit assembly plan, as described in the portable workflow.
- Keep reference labels consistent (e.g. `<Picture 1>`, `<Video 1>`, `<Audio 1>`) across every section.
- Prefer concrete visual and audio details over abstract words like "cinematic" or "beautiful".
- When using keyframes (I2VA / FL2VA / L2VA), clearly state how the first and/or last frame connects to the timeline.


---

## FILE: .agents/skills/h3-prompt-writing/references/base-format.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/.agents/skills/h3-prompt-writing/references/base-format.md)

# H3 base-mode prompt format

Use this reference for T2VA, I2VA, FL2VA, and L2VA prompt writing. It is a
compact, repository-authored description of the required output contract, not a
copy of an upstream prompt guide.

## Select the anchor mode

- T2VA has no image anchor. Build the audiovisual sequence from the brief.
- I2VA binds `<Picture 1>` to the first frame at 0.00 seconds.
- FL2VA binds `<Picture 1>` to the first frame and `<Picture 2>` to the final
  frame. Describe the observable path between them.
- L2VA binds `<Picture 1>` to the final frame. Infer a compatible opening and
  converge to the supplied image.

For an image-anchored mode, put a plain-English alignment statement before the
structured fields. State each picture label, its shot, and its exact target time.
The ending time must match the effective clip duration, formatted to two decimal
places. Do not invent a picture label when no image was supplied.

## Emit three fields in order

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

`integrated_multimodal_description` is the playback-ordered visual and diegetic
audio plan. Establish the initial style, framing, subjects, environment, and
spatial relationships, then describe actions and state changes. For keyframe
modes, explicitly preserve or approach the relevant composition, identity,
clothing, colors, objects, and geometry.

`overall_soundscape` summarizes ambience, physical effects, and non-verbal human
sound. Do not repeat dialogue or music already described elsewhere. Use `N/A`
only for explicitly silent output.

`non_diegetic_music` describes music heard by the audience but not the characters.
Specify audible properties such as instrumentation, tempo, rhythm, dynamics, and
ending behavior. Put source-visible music, singing, radio, or instrument playing
in the main description. Use `N/A` when no score is wanted.

## Timeline notation

- Start the opening with `[Shot 1]` and no timestamp.
- Mark each later cut with a sequential label and a strictly increasing time,
  for example `[Shot 2] At 00:03.500, ...`.
- Use a new shot for an actual edit or meaningful viewpoint/time change. Keep a
  continuous action or morph in one shot and describe its internal timing.
- Describe camera motion as an action: type, direction, useful amplitude, and
  speed. Do not append a disconnected list of camera keywords.
- Keep every cut and event within the requested clip duration.

## Speech, text, and sound

Give each speaking or singing source a stable ID such as `(S1)` or `(S2)` and
reuse it across shots. Put only the language tag and literal words inside the
dialogue wrapper:

```text
The courier (S1) says softly: <d>[English] The gate is open.</d>
```

Preserve supplied dialogue, lyrics, punctuation, and visible text verbatim. State
when a voice is off-screen and keep the visible character's lips closed when that
is required. If a line spans a cut, describe the audio continuity on both sides
of the cut. Place visible signs, labels, and interface copy in double quotes.

Tie diegetic sounds to the event that produces them. Avoid describing the same
sound as both diegetic and non-diegetic.

## Anchor-specific checks

- I2VA: begin from the supplied frame rather than merely borrowing its style.
- FL2VA: describe intermediate motion; do not provide two unrelated still-image
  descriptions. Reach Picture 2 at the actual end time.
- L2VA: keep the final picture assigned to the last shot, not automatically to
  Shot 1, and make the preceding action physically plausible.
- All modes: reference labels must map to supplied assets and remain stable.

Before delivery, verify the field order, shot numbering, timestamps, reference
labels, dialogue fidelity, requested language, duration, and absence of unresolved
template variables.


---

## FILE: .agents/skills/h3-prompt-writing/references/portable-workflow.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/.agents/skills/h3-prompt-writing/references/portable-workflow.md)

# Portable H3 production workflow

This is the runtime contract for the nine H3 skills in this repository. Read it
alongside the selected creative skill. The bundle uses local files and available
tools; no proprietary canvas, choice-card widget, planner, or executor is required.

## Scope and decisions

- Resolve the requested deliverable first: prompt, storyboard, assets, or finished
  video. A prompt-only request ends with the prompt; it does not start generation.
- Reuse the user's duration, language, style, model, audio policy, supplied assets,
  and existing authorization. State reasonable defaults for optional gaps. Ask
  only about missing inputs or choices that materially affect the requested work.
- Creative checkpoints are reviewable artifacts, not mandatory repeated approvals.
  Honor an explicit request to approve a preview before rendering. When the user
  has authorized the complete workflow or delegated creative choices, show the
  plan and continue within that scope without requiring a special UI control.
- A skill never grants permission to publish, send messages, spend outside an
  authorized budget, or change models against the user's choice. User constraints
  override a preset's style, language, shot count, timing, or audio defaults.

## Tools and durable artifacts

1. Inspect the actual available tools, existing scripts, endpoint documentation,
   and supplied media. Do not invent a tool name, model parameter, or returned path.
2. Write plans and prompts to the user-selected project folder. If none is given,
   use a new named folder under `outputs/`, outside the skill package. Keep useful
   artifacts such as `brief.md`, `storyboard.md`, `prompts/`, `references/`,
   `clips/`, and `final/`; create only those needed for this deliverable.
3. Record each reference's role and path: identity, environment, style, typography,
   first/last frame, or audio. Tie generated clips to the prompt revision, input
   assets, model, settings, and output paths. Update downstream references when an
   asset changes; retain previous outputs instead of overwriting user files.
4. Use available image generation/editing tools for requested raster assets. Use
   an available video endpoint or local service for clips, and available audio
   tooling for requested standalone speech/music. For assembly and inspection,
   use an installed editor or tools such as FFmpeg, ffprobe, and OpenCV. Follow the
   host's tool policies, including its preferred image generation mechanism.
5. Independent tool jobs may run concurrently when resources permit; this skill
   does not require separate agents or delegated generation.
6. If a required capability is missing, finish the useful in-scope preparation:
   complete prompts, asset-role manifest, timeline, and verified request example
   where possible. State the missing capability and unfinished media step. Do not
   silently substitute another model, claim a render happened, or call a written
   plan a completed video. Resume generation when the capability becomes available.

## Compile the creative plan into H3 prompts

Read [base-format.md](base-format.md) for T2VA / I2VA / FL2VA / L2VA, or
[ref-format.md](ref-format.md) for Ref2VA. These are writing modes, not guaranteed API
task names. Check the selected backend's input mapping before sending a request.

- Base prompts use `integrated_multimodal_description`, `overall_soundscape`,
  `non_diegetic_music`, in that order, with the appropriate keyframe preamble.
- Ref2VA uses `subject_definitions`, `summary`, `retention_analysis`,
  `detailed_description`, `overall_soundscape`, `non_diegetic_music`, in that order.
- Every reference label must resolve to a supplied asset, in the same order used
  by the backend. Omit nonexistent optional references and their dependencies.
- Keep the creative brief in the user's language. Structured H3 rendering prompts
  use English prose and exact field labels, while dialogue, lyrics, and visible
  text retain their requested language and wording. If the user explicitly wants
  only a same-language creative prompt, deliver that without silently replacing it
  with English. Compile a separate rendering prompt when generation is requested.
- `[Shot 1]` has no initial timestamp. Later shot labels mark actual cuts. For a
  continuous morph, describe timed actions inside one shot rather than turning
  each form into a new shot. Express camera direction, speed, and useful amplitude.
- Separate image style from character identity. A style reference must not inject
  its people, scenery, layout, typography, texture, or lighting into unrelated
  shots. For literal text fidelity, inspect the rendered result; a prompt is not
  proof that the exact lettering or beat synchronization was achieved.

## Longer videos and audio

Use the actual backend's supported duration, dimensions, frame grid, reference
count, audio modes, and model partition. Inspect this checkout's H3 implementation
or serving documentation before building requests. Do not assume that a feature
branch's long-video extension exists on main or on a hosted service.

For a verified vLLM-Omni Ref2VA continuation backend, `long_video=true` with
`long_video_mode=continuation` can accept one `continuation_prompts` entry per
planned window only when the Ref2VA request is executed with its local text
encoder. Step execution and stages that use an external text encoder reject
`continuation_prompts`; omit the field and use the supported per-step inputs in
those configurations. Compute the window count from the aligned total frames,
window size, and overlap. Each prompt uses local window time; track global
shot/music time separately. Repeated overlap is guidance, not additional output
duration. Do not use this mode on a task/partition that does not support it.

Otherwise split into supported clips and assemble them. Same-scene clips may use
tail/head keyframe continuity when supported. Scene changes should have explicit
new scene references; a previous scene's latent tail can preserve the wrong
setting. Choose cuts, morphing, and transitions from the brief rather than imposing
hard cuts whenever a work exceeds 15 seconds. Generation windows and visual shots
are different: one window may contain cuts, and one shot may span multiple windows.

For supplied music, use the selected excerpt as the master timeline unless the
user requests replacement. For native audio, write the musical/SFX direction in
the prompt and verify that the backend actually returns audio. Do not invent a
generic `generate_audio` parameter. If available, locked-source audio latents guide
generation but the audio-VAE reconstruction is not bit-identical to the source.
Use explicit final audio stream-copy when an exact original track is required.
Prompted BPM is a target, not measured timing. Avoid automatically adding lyrics,
voiceover, subtitles, or multiple competing scores to an instrumental request.

## Verification and bounded recovery

Inspect reference images before using them. After rendering, inspect representative
frames, transitions, opening/ending states, text, identities, and scene continuity.
Check audio by listening when possible; report any limit on audio review. Check
duration, dimensions, fps, frame count, audio presence, and decode validity with
media tools. If preserving audio exactly, compare extracted audio stream hashes.

Correct a demonstrated failure with a targeted change to the affected prompt,
reference, or edit. Keep raw generation alongside edited/upscaled versions and
label the difference. Use an agreed retry/resource budget; absent one, allow one
targeted rerender, then report persistent defects and concrete options rather than
entering an unbounded generation loop. Do not switch providers automatically.

Deliver actual artifact paths, the prompts/settings needed to repeat the work,
and material visual/audio limitations. Distinguish requested behavior, observed
model output, and post-production repairs.


---

## FILE: .agents/skills/h3-prompt-writing/references/ref-format.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/.agents/skills/h3-prompt-writing/references/ref-format.md)

# H3 full-reference prompt format

Use this reference for Ref2VA prompt writing. It is a compact,
repository-authored description of the output contract, not a copy of an
upstream prompt guide.

## Emit six fields in order

```text
subject_definitions: ...

summary: ...

retention_analysis: ...

detailed_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

Write the structural prose in English. Preserve the requested language and exact
wording for dialogue, lyrics, and visible text.

## Reference labels

Assign labels only to supplied content and keep every label stable across all six
fields:

- `<Subject N>` identifies reusable visible content such as a person, object,
  environment, costume, interface, action, or visual treatment.
- `<Picture N>` identifies an image used as a concrete frame, composition, or
  storyboard anchor. If an image only defines a subject, cite it in that subject's
  definition instead of inventing a separate target-frame role.
- `<Video N>` identifies a source video being edited or continued, or a video
  whose temporal structure is referenced.
- `<Audio N>` identifies a supplied audio signal that is copied or referenced.
  Audio and video label numbers are independent even when they came from one file.

In `subject_definitions`, give each tracked item a separate line. State its source,
role, and the features that later sections must preserve or transform. Do not
create labels for assets or roles that do not exist.

## Summary and retention

`summary` is one short paragraph describing the target and the principal reference
relationships. Use only labels already defined. Distinguish these operations:

- a concrete image frame anchor;
- generation guided by identity, style, action, camera, or audio characteristics;
- direct editing of a source video;
- continuation from a source video;
- copying an audio signal;
- referencing audio properties without copying the signal.

`retention_analysis` has one line per label and states where it is used and what
changes. Use consistent machine-readable relationship values:

- Visual: `fully_preserved`, `partially_preserved`, `attribute_transfer`, or
  `weak_reference`.
- Audio: `fully_copy`, `partially_copy`, `reference`, or `weak_reference`.

Explain the concrete retained and changed traits after the relationship value.
Do not claim full preservation when identity, geometry, timing, or signal content
is intentionally altered. Do not count newly requested target action as an
unintended loss of reference fidelity.

## Detailed description

`detailed_description` is the playback-ordered rendering plan. For each shot,
cover the current composition, subject appearance and position, environment,
lighting, actions and state changes, camera behavior, diegetic sound, and the
point at which each reference becomes relevant.

- Start with `[Shot 1]` and no timestamp.
- Give later cuts sequential labels with strictly increasing times, such as
  `[Shot 2] At 00:03.500, ...`.
- Place relevant reference labels next to the described subject, frame, motion,
  or sound; do not collect detached labels at the end.
- Use stable `(S1)`, `(S2)`, and subsequent IDs for vocal sources. Put literal
  speech or lyrics inside `<d>[Language] ...</d>`.
- Preserve supplied visible text verbatim and place it in double quotes.
- Describe continuous transformations inside one shot unless an actual cut is
  intended. State camera direction, speed, and amplitude when useful.

For video editing, explicitly identify what stays from the source timeline and
what is replaced. For continuation, distinguish the inherited opening state from
new content. For style reference, transfer visual properties without accidentally
copying people, scenery, layout, or typography.

## Audio fields

`overall_soundscape` summarizes ambience, physical action sounds, and non-verbal
human sounds. `non_diegetic_music` describes score audible only to the audience.
Keep dialogue and diegetic music in `detailed_description`. Use `N/A` only when
the corresponding layer is intentionally absent.

If audio is copied, state whether the whole signal or only a range/layer is used.
If it is referenced, describe the specific property—such as voice timbre, tempo,
rhythm, or texture—without claiming signal reuse.

Before delivery, verify the six-field order, one-to-one label definitions,
retention coverage, shot timing, dialogue fidelity, requested language, and
absence of unresolved template variables.


---

## FILE: .agents/skills/vllm-omni-h3-character-replacement/SKILL.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/.agents/skills/vllm-omni-h3-character-replacement/SKILL.md)

---
name: vllm-omni-h3-character-replacement
description: Plan MiniMax-H3 character replacement, write per-segment Ref2VA prompts and configure reusable H3-04 ComfyUI workflows for generation and review.
---

# vLLM-Omni H3 Character Replacement

Make character replacement easy to use: the user supplies source media and a replacement request; the agent derives the segment plan, prompts and workflow parameters. Keep the input boundary as original images, source video and explicit prompts. Work only on the requested deliverable; planning or prompt edits do not start generation, restart services or introduce preprocessing.

For a simple one-person edit without segmented continuation, reuse the [H3-04 Character Replacement workflow](<../../../apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Character Replacement.json>). For planned multi-segment work, use the Agentic Continuation template linked below; its example-specific settings must be recalculated for new footage.

1. Create a 24-FPS constant-frame-rate working copy of the source video before visual inspection, segmentation and frame-index calculations, unless the user explicitly requests another FPS. Preserve the original file and playback timing. Read [video-segment.md](video-segment.md) first. Use a vision-capable model to inspect the source video, establish shot boundaries and track when each selected person appears, interacts, becomes occluded and leaves. Produce the segment plan before writing new segment prompts.
2. Read [prompt-writing.md](prompt-writing.md), then write each segment's prompt from its inspected action and explicit source-person-to-picture mappings.
3. For a single-person, single-segment edit, configure the simple workflow linked above. For segmented or multi-person work, configure a task-specific copy of the [H3-04 Agentic Continuation workflow](<../../../apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Character Replacement Continuation (Agentic).json>), reusing Global Control, existing nodes and one shared segment-subgraph definition. Follow [workflow.md](workflow.md) for configuration, execution and review.
4. When generation is requested, submit the executable graph through ComfyUI, monitor results, inspect segment previews and deliver the assembled video with remaining quality defects.

Prompt-format dependency: read the official [MiniMax H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing). It can be read directly from the repository; local installation is optional. Before execution, establish the ComfyUI URL, installed node availability, effective Omni deployment settings and task output directory as described in [workflow.md](workflow.md).

Resolve repository-relative references from this skill directory after resolving any global skill symlink, not from the agent's current working directory.

Execution boundary: agent → ComfyUI execution API → existing vLLM-Omni nodes → vLLM-Omni video API. The nodes own model requests; the agent does not bypass ComfyUI with a separate generation client.

For a prompt-only correction to an existing segment, inspect its relevant footage and existing plan; revise the plan only if the observed defect exposes an incorrect boundary or mapping. Do not invent a new segmentation for every wording edit.

Preserve the user's model, resolution, FPS and fixed seed unless asked to change them. Report written plans and prompts separately from generation and visual acceptance. Keep Markdown paragraphs on single source lines and usage notes brief.


---

## FILE: .agents/skills/vllm-omni-h3-character-replacement/prompt-writing.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/.agents/skills/vllm-omni-h3-character-replacement/prompt-writing.md)

# Prompt Writing

Produce an explicit appearance-to-source-person mapping. Keep the user's images, source video and prompt as the input boundary. Prompt-only work ends with the prompt or requested workflow edit; it does not start generation or restart a service.

Prompt-format dependency: read the official [MiniMax H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing) and follow its Ref2VA instructions and referenced files. Local installation is optional; read the repository files directly when the skill is not installed. The upstream reference is currently `references/ref-en.txt`; `ref-format.md` belongs to the local adaptation, not the upstream skill. Use English for the rendering prompt and the user's language for the explanation.

## Planning prerequisite

For new footage or a new multi-segment workflow, complete the vision-based shot/person timelines and segment plan in [video-segment.md](video-segment.md) before writing prompts. Use each planned range's actual visible people and actions, including later entrants. For an existing segment correction, inspect its footage and preserve the plan unless a boundary or mapping needs revision.

## Establish the mapping

Inspect the supplied images, source video and actual reference connections before assigning people. If the media cannot be inspected, state that limitation and use only the user's explicit mapping; ask when the selected source person is ambiguous.

- Number `<Picture N>` by actual request input order, not canvas position or filename.
- Give each target appearance and each selected source-person track separate `<Subject N>` definitions. For two replacements, targets 1/2 and source tracks 3/4 are sufficient. Define only supplied references and selected source people.
- Identify source people with visible features, roles and actions. Use initial position as an aid, then preserve the same person's track as they move. Avoid celebrity or fictional-character names as identity cues.
- Write an explicit one-to-one mapping. A replacement count means selected source people, not the total people visible in the scene. Preserve unselected people.
- Replace example-specific descriptions when the source changes. “Central performer wearing a pale suit” and “right foreground person” describe one example, not a universal person-selection rule. First-appearance order alone is often ambiguous.

## Compile the prompt

Keep the six sections in order:

1. `subject_definitions`: targets from pictures, source-person tracks from video, and the source video's role.
2. `summary`: edit the source video, state the replacement count and each mapping.
3. `retention_analysis`: target identity/appearance `fully_preserved`; source tracks `attribute_transfer` for motion, pose, expression and visibility; video `partially_preserved` for scene, camera, interactions and timing.
4. `detailed_description`: `[Shot 1]` followed by playback-ordered actions and identity constraints. Describe real source cuts only when known.
5. `overall_soundscape`: retain or reference the requested source sound properties.
6. `non_diegetic_music`: follow the user's music policy.

Use a compact shared constraint paragraph rather than repeating the same rule in all sections:

> Each target appearance belongs to exactly one assigned source-person track. Keep each person's own face, hair, clothing, materials and worn accessories throughout turns, crossings and occlusion. Preserve both the presence and absence of these features in the assigned picture. When a person leaves the view, their replacement leaves with them; on re-entry, restore the same assigned identity. Every unassigned person retains their original appearance. Render only people visible in the source at each moment. Pictures supply appearance; the source video supplies actions, handheld props, scene, camera, lighting and timing.

Adapt this paragraph to the observed issue. Avoid accumulating redundant negative instructions or naming an unwanted accessory that is absent from the inputs. Preserve source interactions and natural movement rather than freezing a reference pose. Exact soundtrack copying requires actual audio handling, not prompt wording.

## Single and multiple people

Single-person mapping skeleton (replace the bracketed description before delivery):

```text
<Subject 1>: The target person in <Picture 1>, defining their appearance.
<Subject 2>: [Unambiguous visible description of the selected source person] in <Video 1>.
Mapping: replace <Subject 2> with <Subject 1>; preserve all other people.
```

For two people, define two target appearances and two source tracks, then state `<Subject 3> → <Subject 1>` and `<Subject 4> → <Subject 2>`. Scale to the actual reference count and backend limit, not a fixed eight- or ten-subject template. Keep faces, outfits and accessories separate even during close interaction. Multi-view images depict one target person; they do not add people to the scene.

## Segment prompts

For segmented work, use the plan from [video-segment.md](video-segment.md). Inspect each segment's actual inputs independently: define only its supplied references and selected source tracks. Appearance references define identity; continuation guides establish inherited composition and pose. State what remains visible and what exits when that affects identity.

## Diagnose and verify

- **Mixed or duplicated identity:** check image order and effective request inputs first; clarify the two source tracks and exclusive mappings before adding prose.
- **Accessory leakage:** distinguish target appearance from source motion and props; preserve the assigned reference's features, including their absence. A bad frame suggests leakage but does not by itself establish its cause.
- **Late identity drift:** inspect departures, occlusion, cuts and continuation boundaries; bind identity to the person rather than their current screen position.
- **Identity retained in occluded close-ups:** if generated close-ups retain the source identity under hair, hand or lighting occlusion, the segment-level mapping may need shot-specific appearance cues. In `detailed_description`, name the assigned target at that shot and state which picture features must be visible there (the target's own hair length and color, face, glasses or their absence), e.g. "the extreme close-up shows <Subject 1> with the short light-brown hair and face from <Picture 1>". Keep it a positive description of the assigned appearance; do not enumerate the source features being removed.
- **Scope:** background removal, face anonymization and automatic appearance description are separate features. Do not introduce them to repair a prompt unless the user requests them.

Check all labels resolve, mappings are one-to-one, and no placeholders or unused references remain. When editing a workflow, keep positional and named prompt values consistent where both exist; preserve unrelated fields and user layout. Usage Guide notes should only say which media, mappings and controls to change.

For an authorized comparison, keep inputs, model, seed, LoRA and sampling fixed while changing the prompt. Inspect both identities together, occlusion and the reported failure interval; inspect frame-by-frame around the defect when needed. Report prompt edits separately from observed generation results. Health checks, valid JSON and successful decoding do not establish character-replacement quality.


---

## FILE: .agents/skills/vllm-omni-h3-character-replacement/video-segment.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/.agents/skills/vllm-omni-h3-character-replacement/video-segment.md)

# Video Segment

Use this guide when dividing source video into character-replacement clips or revising a continuation workflow. Derive the plan from the actual video and current backend; the existing Case03 is a scene-specific example, not a general-purpose five-segment recipe.

## Visual inspection comes first

First probe the source and create a 24-FPS constant-frame-rate working copy unless the user explicitly requests another FPS. Use frame-rate conversion (for example, FFmpeg `fps=24`), not playback-speed changes. Preserve the original media and soundtrack; verify duration and audio alignment. Inspect this working copy and use its frame timeline consistently for source cuts, segment boundaries and guide/trim arithmetic. If already confirmed 24-FPS CFR, reuse it without another conversion.

Use a vision-capable model to examine the source footage before choosing segments. Inspect the reference images too, then distinguish their target appearances from the source people. If direct video viewing is unavailable, extract timestamped frames or contact sheets with a local media tool and inspect the images visually. Metadata and scene-change scores help navigation; they do not establish person identity or action continuity.

Survey the full video, not only the first frame. Inspect more densely around shot changes, entrances, exits, crossings, heavy occlusion and interactions involving multiple people. Play or inspect consecutive frames around a proposed boundary; sparse thumbnails alone can miss a person entering briefly or exchanging positions. Record uncertain observations and inspect additional frames instead of guessing. If visual inspection is unavailable, report the blocked planning step rather than presenting a visually verified plan.

Create two linked records before selecting boundaries:

- Shot timeline: source time ranges, actual cuts, camera/composition changes and the action within each shot.
- Person visibility timeline: stable source-track IDs, distinguishing visual cues, first appearance, visible intervals, occlusion, re-entry, departure and interactions with other tracks. Track IDs remain stable across position changes and across cuts only when identity is supported by the footage; record ambiguity explicitly.

Inspect duration, FPS and frame count. For variable-frame-rate media, use timestamps or explicitly normalize the working clip; do not assume frame index equals time multiplied by nominal FPS. Reference numbering follows actual input order; source-track IDs are independent of reference numbering and screen position.

## Choose segments for multiple replacements

Observed character-replacement runs have been more stable with segments around 5 seconds; longer segments are more prone to identity drift and visual breakdown. Use roughly 5 seconds as an initial planning target, then adjust boundaries to preserve coherent actions and source cuts. This is a quality heuristic, not a model duration limit or a reason to split every clip into equal lengths. Validate longer segments explicitly before retaining them.

Choose boundaries using the shot timeline and the changing set of selected people. Prefer the fewest segments that preserve coherent action, distinguishable identities and backend duration limits. Equal-duration slicing and splitting at every entrance or exit are not defaults.

| Observed source event | Planning decision |
|---|---|
| Several selected people remain identifiable during one continuous action | Keep them in one segment when supported; give each a separate mapping. |
| A selected person enters later | Include enough source context to identify the entrant. Use an entry-focused segment or a later anchor if needed; do not require every target in the first frame. |
| Selected people leave and the scene continues with fewer people | Consider a boundary after the departure stabilizes. The next segment maps the people visible anywhere within that segment; departed identities do not migrate to the remaining person. |
| People cross, overlap or exchange props | Keep the interaction and identity cues together where possible. Avoid a boundary at maximal occlusion or during contact; if a split is unavoidable, carry context on both sides and inspect the join. |
| A cut changes composition within the same scene | Check person identity on both sides. Split if the new composition needs a different mapping or the backend cannot retain it reliably; otherwise preserve the existing cut in the segment. |
| A cut introduces a new scene or a discontinuous cast | Prefer an independent segment and anchor; avoid imposing the previous scene through continuation guides. |
| A person is temporarily occluded and later reappears | Keep the same track if supported by the footage. Occlusion alone does not define a new person or require a split. |

For each candidate segment, list every selected person visible at any point in the range, their local appearance/exit times and the reason for the chosen boundary. A later entrant still needs a reference and mapping. Keep unselected people unchanged. If the selected cast exceeds the running service's reference limit, revise the plan or report the limitation; do not silently omit a target or assume separate edits can be composited without affecting other people.

An identity anchor should make the assigned people identifiable. A later multi-person view may be a better anchor than an opening where people have not entered. Generate outward from the chosen anchor when the workflow supports it. Backward continuation is an optional strategy: deliberately reverse the source, choose guides in that direction and reverse the generated frames back for assembly. Keep generation dependency order separate from chronological assembly order. A later anchor is not proof that reversal will improve quality.

## Produce a concrete plan

Record one row per segment:

| Segment | Source range / shot | Visible selected tracks → pictures | Boundary reason | Direction / dependency | Guide source/range | Generated duration/frames | Keep range | Assembly order |
|---|---|---|---|---|---|---|---|---|

Use half-open ranges `[start, end)` in the plan; translate them to each node/API's actual inclusive or length-based convention. Distinguish source frames, generated frames, previous-output guide indices and retained assembly frames. Identify the coordinate system for every frame number.

- Compute source duration from the selected source timestamps. For confirmed constant FPS, frame count divided by FPS also gives duration.
- Calculate required output frames from duration and output FPS, then apply the running model's frame-alignment and minimum-duration constraints. The current `_align_frame_count` helper aligns positive counts to `5 + 17n`; it does not impose a 107-frame minimum or choose FPS. Treat the example's 107-frame minimum and 22-frame guide window (positions 0 and 21) as configuration to verify, not universal model constraints.
- Mark overlap frames as guidance and retain the intended source action once. Derive trim/keep ranges from output FPS and the chosen overlap; do not copy frame constants from another clip.
- Select guide indices from the actual preceding frame batch, distinguishing raw generated frames from already trimmed assembly frames. Verify indices stay in bounds and correspond to the intended poses.
- Account for generated alignment tails explicitly. FPS changes require recalculating frame-based durations, overlaps, guide indices and trims; exposing a shared FPS widget alone does not perform that conversion.
- Keep the plan's user controls in seconds or source ranges where possible. Derive internal frame arithmetic rather than asking users to guess it.

Case03's B/A/C/D/E order, reverse generation, five-person mappings and 24-FPS frame settings belong to its supplied 27-second scene. New footage needs a new plan; do not reuse its cigarette-lighting, pointing or departure actions as defaults.

## Write and connect segments

Only after the visual timelines and segment plan are complete, write each segment's prompt using [prompt-writing.md](prompt-writing.md). Include segment-local entrance/exit timing only when supported by inspection; map it from global source time and account for reversed generation where applicable. Define only references connected for that segment and maintain the same real-person assignment across segments even if local reference indices change. Preserve the same reference appearance through continuation; guide frames supply inherited pose and composition.

## Handoff to execution

The plan is ready when source ranges cover the requested edit, each selected person has a supported reference mapping, guide indices are in bounds, and trim/keep ranges remove overlap exactly once. Record frame coordinate systems and separate generation dependencies from assembly order. Follow [workflow.md](workflow.md) to prepare inputs, configure/export the ComfyUI graph, execute and review results.


---

## FILE: .agents/skills/vllm-omni-h3-character-replacement/workflow.md (https://github.com/vllm-project/vllm-omni/blob/HEAD/.agents/skills/vllm-omni-h3-character-replacement/workflow.md)

# Workflow

Execute one character-replacement task through ComfyUI. Use [video-segment.md](video-segment.md) for visual analysis and frame planning, and [prompt-writing.md](prompt-writing.md) for prompts. A plan-only or prompt-only request stops before submission.

## 1. Discover the environment

Resolve service addresses in order: explicit user/task configuration, the selected workflow and Global Control for the Omni endpoint, then the ComfyUI launch configuration or active application connection for the ComfyUI endpoint. Ask for unresolved required values instead of scanning arbitrary ports. Do not treat localhost addresses from another task as defaults. Resolve the ComfyUI base URL from task configuration and check `GET /system_stats`. Read `GET /object_info` for the nodes required by the selected template; guide and concatenation nodes are needed only when used. Read effective Global Control settings, including mode, service URL/model, LoRA configuration, resolution, FPS, steps and seed. Preserve the user's choices; server-local LoRA paths are not agent-local paths. Confirm the selected deployment supports the requested combination. For FastH3 + Swap, record the effective model and server revision, inspect whether the serialized request contains a nonzero LoRA, and check adapter activation evidence. A past 400 or later successful job alone cannot establish current support or prove that the adapter took effect; never assume it was silently ignored.

Create a task output directory. Keep agent-local files, ComfyUI input/output paths and Omni server paths distinct. If checking the Omni model listing, resolve `/v1/models` from the configured API base without appending `/v1` twice; generation still goes through ComfyUI nodes.

## 2. Reuse relevant history when available

Look for earlier runs of the same task. Check their inputs, settings and outputs before reuse; successful execution does not establish visual acceptance. Record observed defects as the next run's fix list. A missing history entry does not block a first run.

A previous API prompt can be a baseline when its topology matches the new plan. On versions exposing the standard history tuple, it is stored at `history[prompt_id]["prompt"][2]`; inspect the actual response before using that field. Retrieve reported output files with `GET /view`, passing their filename, subfolder and type. Avoid downloading or reviewing unrelated task outputs.

## 3. Inspect and plan

Probe source resolution, duration, frame count and nominal/average FPS. Before inspecting and planning, normalize a working copy to 24-FPS CFR unless the user explicitly chose another FPS; preserve the original, playback timing and soundtrack. Verify the working copy's duration and frame rate, and use that timeline for all segment calculations. Start with timestamped contact sheets (for example, 1 FPS), then inspect more densely around entrances, occlusion, interactions and cuts. Scene-change detection can suggest candidates; thresholds such as 0.3 or 0.15 are adjustable starting points, not required settings.

Complete the shot/person timelines and segment plan in [video-segment.md](video-segment.md). Derive generated and retained frame ranges from actual FPS, model alignment, overlap and generation direction. Check complete source coverage, including the tail. Use independent anchors across real cuts when appropriate; derive continuation guide indices from the specific raw or trimmed frame batch connected as input, excluding unwanted alignment tails.

## 4. Prepare inputs and configure the graph

Create a task-specific copy of the simple or Agentic template linked in `SKILL.md`. Preserve the original and reuse Global Control, existing nodes and one shared segment-subgraph definition. Read actual node/subgraph input definitions and widget values rather than hardcoding their positions.

Prepare frame-exact source clips at the chosen FPS and a reference size accepted by the deployment. Keep aspect ratio and verify any scaling or padding. Re-probe clips and inspect first/last frames against the plan. Keep the source soundtrack separately if the final edit must retain it. Check reference-duration limits before cutting; extend undersized clips within the same shot where possible and account for added context in trimming.

For remote ComfyUI, confirm the installed media-upload contract, including MP4/M4A acceptance and input/output retrieval through `/view`. Record the tested server version and minimal upload/retrieval example in task artifacts so later runs can reuse verified behavior. The standard `POST /upload/image` accepts a multipart `image` field and returns `name`, `subfolder` and `type`; use the returned input-relative path in loader widgets. For shared local storage, resolve the configured input directory before copying. Version filenames rather than overwriting inputs or results.

Write each segment's prompt using [prompt-writing.md](prompt-writing.md), following the user's replacement scope. Apply the user's explicit appearance scope. The bundled template replaces full appearance; retaining source costumes while changing face/hair is a different policy. If the request leaves this choice materially ambiguous, ask one focused question rather than choosing from the scene or copying an old prompt without checking intent. Difficult close-ups or small figures may benefit from explicit positive appearance cues, which must be checked in generated results.

Set references, duration, guides, direction and trim/keep ranges per instance. Connect continuation dependencies explicitly and keep generated previews before concatenation. Reuse media nodes for source-order assembly and requested audio. Check socket types, including INT generation FPS versus FLOAT Create Video FPS. Keep named and positional widget values consistent where both exist; determine their actual export precedence from the installed frontend or exported API graph rather than guessing. Read subgraph instance/inner-node mappings from export; IDs such as `10078:98` are examples, not stable identifiers. `raw_frames` feeds a batch-length selection: a value such as 4096 is a finite upper bound, not a special “all frames” sentinel. Ensure it covers the planned frame count.

## 5. Export, submit and monitor

Save editable workflow JSON and export an executable API prompt through a verified facility supported by the installed ComfyUI frontend. Check native subgraphs are expanded and all required output dependencies are present. A matching history prompt is an optional baseline, not a substitute for validating a changed topology. If no verified export is accessible, provide the prepared workflow and request a one-time API-format export; do not invent a conversion API or bypass ComfyUI with direct Omni generation requests.

Submit `POST /prompt` with `{"prompt": <api graph>}`. Store the returned `prompt_id` and inspect HTTP errors and `node_errors`. Follow `/history/{prompt_id}`, `/queue` and execution events as available. Missing history alone cannot distinguish queued, running, unaccepted or cleared work. Confirm acceptance before retrying a lost submission response, and confirm terminal status before treating outputs as complete. Existing Omni nodes own downstream request serialization and polling.

When changing an open workflow, preserve current edits and verify the loaded canvas as well as the saved file; an unsaved canvas can overwrite a disk-only change.

## 6. Check and deliver

Retrieve output files reported by history. Check frame count, duration, FPS, expected audio and full decoding. Visually review segment starts/ends, joins, selected and unselected people, close-ups, wide shots, clothing/accessories and soundtrack alignment. Inspect reported defect intervals frame-by-frame where needed.

Deliver the playable result and remaining defects. Save the input mapping, segment plan, prompts, editable workflow, executable graph, execution records and raw outputs for inspection and resume. Successful execution and decoding are separate from visual acceptance.

## 7. Iterate on the affected dependency chain

Change one variable per comparison and retain previous outputs. Editing a segment can invalidate that segment and every downstream segment using its guides, plus assembly. Independent unchanged branches may reuse cache only if it is available and their effective inputs are unchanged. Inspect the actual rerun scope; do not promise that only one segment will regenerate. When necessary, reconnect saved successful outputs explicitly.

Use versioned output prefixes only on affected outputs where possible; changing every prefix can invalidate otherwise reusable save nodes. Retry within the agreed budget and report persistent defects rather than silently changing the model, replacement scope or reference images.

## First-run qualification and history backup

Validate the execution path once with empty history and new inputs. A workflow-to-API exporter/helper must handle native subgraphs, changed instance counts and current widget values without a matching prior run. A documented manual export handoff is a remaining automation limitation, not a completed one-sentence flow.

Before an authorized history clear, wait for an idle queue, save the complete `GET /history` response and verify it can be parsed with the same prompt IDs. Preserve referenced media separately if it must survive file cleanup; history JSON contains metadata and graphs, not media bytes. On versions supporting it, `POST /history` with `{"clear": true}` clears execution history; verify `GET /history` is empty afterward. This does not clear execution cache or delete source/output media. There is no assumed public history-import API: archived graphs may be inspected or resubmitted, but resubmission is a new execution, not restoration of history.
