# benjiyaya__Minimax-H3-Prompt-AgentSkill

Source: https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill
Fetched: 2026-11-28

Description: Use when making MiniMax H3 video prompts from media + ideas. Agent will do an formatted prompt for you.
Stars: 167  Updated: 2026-08-06T05:25:56Z



---

## FILE: README.md (https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill/blob/HEAD/README.md)

# 🎬 MiniMax H3 Video Prompt Agent Skill

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) skill that transforms rough video ideas + reference images/audio into production-grade **MiniMax H3** video generation prompts.

## What It Does

When you attach images, video, or audio and describe a video concept, this skill:

1. **Detects the H3 mode** automatically (Ref2VA, T2VA, I2VA, FL2VA, L2VA)
2. **Creatively enhances** your idea across 7 cinematic dimensions
3. **Outputs the exact H3 format** — ready to paste into ComfyUI

## H3 Modes Supported

| Mode | Use Case | Output Format |
|---|---|---|
| **Ref2VA** | Character/style/voice reference assets | 6-section structured prompt |
| **T2VA** | Text-to-video (no images) | Timed multi-shot timeline |
| **I2VA** | 1 image as first frame | Instruction line + timeline |
| **FL2VA** | 2 images (first + last frame) | Instruction line + timeline |
| **L2VA** | 1 image as last frame | Instruction line + timeline |

## Creative Enhancement Dimensions

Every prompt is enriched across:

- 🎥 **Camera Identity** — Physical camera type, imperfections, format aesthetic
- 🎨 **Visual Texture** — Grain, color palette, lighting design, exposure behavior
- ⚡ **Pacing Arc** — Energy progression, cut rhythm, build patterns
- 👤 **Character Detail** — Physical features, wardrobe, visual signature colors
- 🧭 **Spatial Geography** — Screen directions, action vectors, environmental layout
- 🔄 **Continuity Progression** — State tracking, damage accumulation, emotional arcs
- 🔊 **Sound Design** — Ambience, SFX, diegetic/non-diegetic music mapping

## Installation

### For Hermes Agent Users

Copy the skill folder to your Hermes skills directory:

```bash
# Linux/macOS
cp -r h3-video-prompt-enhancer ~/.hermes/skills/creative/

# Windows
xcopy h3-video-prompt-enhancer %LOCALAPPDATA%\hermes\skills\creative\ /E /I
```

Restart Hermes Agent. The skill auto-triggers when you attach media and describe a video idea.

### For Other Agent Frameworks (Claude, Codex, etc.)

The `SKILL.md` and reference files are standard Markdown — load them as system context or project rules in any agent that supports file-based instructions.

## File Structure

```
h3-video-prompt-enhancer/
├── README.md                              # You are here
├── SKILL.md                               # Main skill — workflow, rules, verification
└── references/
    ├── ref2va-format.md                   # Full Ref2VA 6-section format spec
    ├── base-multishot-format.md           # T2VA / I2VA / FL2VA / L2VA format spec
    └── creative-showcase.md               # Advanced long-form prompt patterns & benchmarks
```

## Usage Examples

### Ref2VA (Character References)

```
📎 Attach: character_sheet.png, voice_sample.mp3
💬 "A woman matching this character walks through a neon-lit Tokyo street at night, 
    stops at a ramen stand, says hi to the cook"
```

→ Outputs a 6-section structured prompt with `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`.

### I2VA (Image-to-Video)

```
📎 Attach: first_frame.png
💬 "A baker opening his shop before sunrise, proud of the first loaf"
```

→ Outputs a timed multi-shot timeline anchored to the first-frame image.

### FL2VA (First + Last Frame)

```
📎 Attach: first_frame.png, last_frame.png
💬 "Day to night transition over a city skyline"
```

→ Outputs a continuous interpolation prompt from first to last frame.

## Key Format Rules

- Output is **raw structured text** — no markdown fences, no preamble
- All timestamps use `MM:SS.mmm` format, strictly increasing
- One dominant action per shot — never cram multiple actions
- Camera motion written as natural English (e.g., *"The camera pushes in with small amplitude at slow speed"*)
- Dialogue preserved verbatim inside `<d>[Language] ...</d>` tags
- Reference labels (`<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`) stay consistent across all sections

## ComfyUI Integration

```
LLM Node (with system prompt) → text output → H3 Ref2VA / H3 Base node prompt input
```

Keep the ASSETS list order identical to the physical wiring order of the reference inputs — labels map positionally (Image 1 → first image socket, etc.).

## Provenance

Built for the [MiniMax H3](https://github.com/MiniMax-AI) omni-modal video model. The Ref2VA and Base MultiShot format specs are derived from MiniMax's official H3-Context-IR preprocessor documentation.

## License

MIT




---

## FILE: SKILL.md (https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill/blob/HEAD/SKILL.md)

---
name: h3-video-prompt-enhancer
description: "Use when making MiniMax H3 video prompts from media + ideas."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [video, prompt-engineering, minimax-h3, comfyui, text-to-video, image-to-video, creative]
    related_skills: [comfyui]
---

# H3 Video Prompt Enhancer

## Overview

Transform a user's rough video idea + attached assets into a production-grade MiniMax H3 video generation prompt. The skill handles two H3 generation modes:

- **Ref2VA** — when the user provides reference assets (character sheets, style images, reference videos, voice samples). Outputs a 6-section structured prompt. Load `references/ref2va-format.md` for the full spec.
- **Base MultiShot** (T2VA / I2VA / FL2VA / L2VA) — for text-to-video or image-anchored generation. Outputs a timed multi-shot prompt. Load `references/base-multishot-format.md` for the full spec.

The skill's distinctive value: it doesn't just format-comply — it **creatively enhances** the brief with professional-grade cinematic detail (camera aesthetics, visual texture, lighting design, pacing arcs, spatial choreography, continuity tracking) before mapping everything into the exact H3 output format. Load `references/creative-showcase.md` for quality benchmarks and pattern examples from advanced long-form prompts.

## When to Use

**Trigger when the user:**
- Attaches 1+ images/videos/audio files and describes a video they want to create
- Says "make a video prompt" / "enhance this for H3" / "write a Ref2VA prompt" / "img2vid" / "txt2vid"
- Provides a video concept and wants it structured for MiniMax H3 generation
- Mentions H3, MiniMax, Ref2VA, ComfyUI video generation
- Pastes a creative brief (CAMERA/LOOK/STYLE format) and wants it converted to H3 format

**Don't use for:**
- Non-H3 video models (Sora, Runway, Kling, etc.) — the output format is H3-specific
- Pure image generation prompts
- Video editing tasks that don't involve new generation

## Step 0: Classify the Mode

Determine the H3 mode from what the user attached and stated:

| User provides | Mode | Format reference |
|---|---|---|
| Reference images/videos/audio (character sheets, style refs, voice clips) | **Ref2VA** | `references/ref2va-format.md` |
| Nothing — just a text idea | **T2VA** (text-to-video) | `references/base-multishot-format.md` |
| 1 image as the first frame | **I2VA** (image-to-video) | `references/base-multishot-format.md` |
| 2 images (first frame + last frame) | **FL2VA** (first-last-to-video) | `references/base-multishot-format.md` |
| 1 image as the last frame only | **L2VA** (last-frame-to-video) | `references/base-multishot-format.md` |

**Key distinction:** An image used as a *first/last frame anchor* = I2VA/FL2VA/L2VA. An image used as a *character/style reference* (not a frame position) = Ref2VA. When ambiguous, ask the user: "Is this image a frame anchor (first/last frame of the video) or a reference (character/style/scene)?"

## Step 1: Gather Parameters

Confirm these before enhancing (ask if missing, but proceed if the idea is clear enough):

- **duration_s**: 4–15 seconds (integer). Default to 8 if unspecified.
- **aspect ratio**: 16:9, 9:16, 1:1, 4:3, 21:9. Default to 16:9.
- **shot count**: Let the planner decide, or respect user's explicit count. Budget: 4–6s → 1–2 shots; 7–10s → 2–3 shots; 11–15s → 3–5 shots.
- **asset inventory**: What each attached file is and its role. Numbering per modality defines labels: Image k → `<Picture k>` or `<Subject N>`; Video k → `<Video k>`; Audio k → `<Audio k>`.

## Step 2: Creative Enhancement

This is the core value-add. Take the user's idea and enrich it across seven dimensions. The goal: produce a prompt with the depth and cinematic intelligence of a professional storyboard. Consult `references/creative-showcase.md` for full examples of each pattern.

### Enhancement Dimensions

**1. Camera Identity** — Assign a distinct camera aesthetic that matches the scene's tone:
- Physical type: handheld, tripod, drone, steadicam, dolly, security cam, dashcam, POV, arc shot
- Imperfections to preserve (when stylistically appropriate): hand tremor, autofocus hunting, exposure fluctuation, lens flare, motion blur, awkward zoom
- Format/aesthetic hint: 16mm film, DV tape, digital clean, anamorphic, vintage camcorder, broadcast

**2. Visual Texture (LOOK)** — Define the image quality and color science:
- Grain/noise: film grain, electronic noise, clean digital, VHS tracking, subtle blur
- Color palette: warm/cool, saturated/desaturated, high/low contrast, natural skin tones
- Lighting design: natural, studio, neon, golden hour, mixed sources, practical lights
- Lighting transitions: if locations change across shots, describe how the light shifts

**3. Pacing Arc** — Plan the energy progression across the full duration:
- Build patterns: quiet→energetic, tense→release, slow build→explosive peak→settle, steady rhythm
- Cut rhythm: accelerating cuts toward climax, slow contemplative holds, musical cutting on beats
- Match the pacing arc to the emotional intent of the brief

**4. Character Detail** — Flesh out every on-screen person with specificity:
- Physical: age range, build, hair color/style, skin tone, distinctive features (scars, freckles, heterochromia)
- Wardrobe: specific garments with colors, materials, textures, accessories — note changes across locations or time
- Visual signature: a recurring color or visual element that makes the character instantly recognizable in every shot (e.g., "electric purple energy trails", "glossy teal jacket reflections", "always wears a red scarf")
- Coverage note: describe outfits fully — avoid implying revealing clothing unless the user explicitly requests it

**5. Spatial Geography** — For action sequences or multi-location videos:
- Screen direction: who enters from where, movement vectors (Left→Right, Deep→Front, foreground↔background)
- Key action moments: the 2–3 critical motion beats that define the sequence
- Environmental layout: what's in the space, how it's lit, reflective surfaces, depth

**6. Continuity Progression** — Track what changes across shots so the video feels coherent:
- Physical state: damage accumulates, hair gets messier, clothes get wet/torn/dusty
- Environmental: props move, lights flicker, weather shifts, debris scatters
- Emotional: expressions and body language evolve naturally across the timeline

**7. Sound Design Plan** — Map the full audio landscape:
- Ambience: room tone, environmental atmosphere, background chatter, traffic, wind
- Physical action sounds: footsteps, impacts, fabric rustling, door creaks, liquid pouring
- Non-diegetic score: instrumentation, tempo, rhythm, dynamic changes (goes in `non_diegetic_music` field)
- Diegetic music: source-visible music — radio, speaker, live performance (goes in shot description)
- Dialogue vs. voiceover: clearly mark which is spoken on-camera vs. off-screen narration

### Per-Shot Quality Bar

Every shot in the storyboard must specify:
- **Composition**: framing (wide, medium, close-up, extreme close-up, macro), angle (eye-level, low, high, overhead, Dutch)
- **Camera motion**: type + amplitude + speed in natural English (e.g., "The camera pushes in with small amplitude at slow speed")
- **Subject action**: exactly ONE dominant action per shot — never cram multiple actions
- **Environment/lighting**: what's visible, how it's lit, time of day cues
- **Sound cue**: what's audible in this specific moment
- **Reference labels** (Ref2VA only): where referenced content appears or takes effect

## Step 3: Format and Output

Load the appropriate format reference and produce the final H3 prompt.

**For Ref2VA** → Load `references/ref2va-format.md`. Output exactly 6 sections in order:
1. `subject_definitions:` — one line per tracked item
2. `summary:` — task-type prefix + one paragraph
3. `retention_analysis:` — fidelity markers per label
4. `detailed_description:` — 350–500 words, opens with style, then `[Shot N]` timeline
5. `overall_soundscape:` — 1–4 sentences
6. `non_diegetic_music:` — 1–3 sentences or N/A

**For Base MultiShot** → Load `references/base-multishot-format.md`. Output:
1. Instruction line (I2VA/FL2VA/L2VA only — exact template from reference)
2. `integrated_multimodal_description:` — timed multi-shot timeline
3. `overall_soundscape:` — 1–4 sentences
4. `non_diegetic_music:` — 1–3 sentences or N/A

### Critical Format Rules (both modes)

- Output ONLY the specified fields — no preamble, no explanation, no markdown fences, no commentary
- Write everything in English. Exceptions: dialogue/lyrics inside `<d>` tags and visible on-screen text keep their original language verbatim
- Timestamps: `[Shot 1]` has NO timestamp and opens with style + initial composition. Later shots: `[Shot N] At MM:SS.mmm, the camera cuts to ...` with strictly increasing times within duration
- A cut must add NEW information (subject, space, state, viewpoint, time). If only distance/angle changes, use camera motion instead of a cut
- Camera motion = motion type + amplitude + speed as natural English action, not stacked labels
- Dialogue format: identifying phrase + speaker ID + delivery OUTSIDE `<d>`; inside `<d>` ONLY language tag + exact words: `<d>[English] Wait for us!</d>`
- Preserve the user's dialogue verbatim — never translate, rewrite, or paraphrase
- Voiceover: exact phrase "says in an off-screen voiceover" + immediately state "while his/her lips remain completely closed"
- On-screen text (signs, neon, subtitles): English double quotation marks, verbatim, no translation
- Avoid named third-party IP, real celebrities, trademarked characters — describe them generically

### Camera Motion Vocabulary

Types: Zoom In/Out, Push In/Pull Out, Pan L/R, Truck L/R, Tilt Up/Down, Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly, POV, Roll CW/CCW.
Amplitude: "with small amplitude" / "with large amplitude" (omit when medium).
Speed: "at slow speed" / "at fast speed" (omit when normal).

## Step 4: Present to User

After generating the H3 prompt:
1. Present the full prompt in a code block so it can be copied directly
2. State which mode was detected and why
3. Flag any assumptions made (e.g., "Assumed 8s duration and 16:9 — adjust if needed")
4. Offer to refine specific aspects: camera aesthetic, pacing, character detail, shot count, sound design

## Creative Patterns from Showcase

The user maintains advanced long-form prompt examples representing the quality bar. Load `references/creative-showcase.md` for full examples. Key transferable patterns:

**Long-form Storytelling** (montage, day-in-the-life, narrative):
- Explicit CAMERA/LOOK/STYLE blocks defining the aesthetic before the storyboard
- Outfit and location progression across time, with lighting transitions
- Voiceover carrying emotional narrative over non-synchronized visuals
- Pace building from quiet opening to high-energy finale
- Per-shot storyboard with timing, action, and VO lines

**Action Choreography** (combat, chase, sports):
- Per-character color lock — each character has a signature color for their effects/trails/reflections
- Explicit spatial layout with screen directions (Deep→Front, Left→Right)
- Action vectors — the key motion moment that defines each shot
- Progressive continuity — damage accumulates, hair gets windblown, dust/cracks appear
- Strict shot count enforcement — state the count and hold it
- Environmental reactivity — holograms flicker, floors reflect, walls crack in response to action

## Common Pitfalls

1. **Treating Ref2VA reference images as frame anchors.** A character sheet or style reference is a `<Subject>`, not a `<Picture>`. Only use `<Picture N>` standalone when the image IS a concrete frame position (first frame, keyframe, last frame). When in doubt, cite the image inside the relevant `<Subject N>` line.

2. **Cramming multiple actions into one shot.** One dominant action per shot — this is a hard H3 constraint. If the brief describes sequential actions, split them across multiple shots with cuts.

3. **Forgetting camera motion.** Every shot needs an explicit camera movement specification — even "Static Shot" if the camera doesn't move. Omitting it leaves the model guessing.

4. **Inconsistent character identity across shots.** Repeat identity anchors (hair, clothing, key props) in every shot, phrased freshly but consistently. If hair gets messy in shot 3, it stays messy in shot 4.

5. **Wrong shot count for duration.** Respect the budget: 4–6s → 1–2 shots; 7–10s → 2–3 shots; 11–15s → 3–5 shots. Don't plan 5 shots for a 5-second video.

6. **Mixing diegetic and non-diegetic music.** Music audible to characters (radio, live performance, phone speaker) goes in the shot description. Background score the characters can't hear goes in `non_diegetic_music`. Never put the same music in both.

7. **Inventing reference labels (Ref2VA).** Never create labels beyond those defined in `subject_definitions`. `<Subject 3>` means the same thing in every section where it appears. Different indices for different modalities are numbered independently.

8. **Translating or rewriting dialogue.** Preserve the user's exact words inside `<d>` tags — including punctuation, hesitations, and language. Never translate to English if the user wrote in another language.

9. **Skipping the style opener.** The `detailed_description` (Ref2VA) or `integrated_multimodal_description` (Base) MUST open with 1–2 sentences of overall style BEFORE `[Shot 1]`. Don't jump straight into the first shot.

10. **Flat timestamps.** `[Shot 1]` never has a timestamp. Every subsequent shot needs `At MM:SS.mmm` with strictly increasing times. Convert duration to seconds with proper formatting (8s → 8.00 for the instruction line).

## Verification Checklist

- [ ] Mode correctly detected from attached assets and user intent
- [ ] All seven enhancement dimensions applied (camera, look, pacing, character, spatial, continuity, sound)
- [ ] Output contains ONLY the required H3 fields — no preamble, no markdown fences, no extra commentary
- [ ] Timestamps are strictly increasing and fall within duration_s
- [ ] Exactly ONE dominant action per shot
- [ ] Camera motion specified for every shot (type + amplitude + speed)
- [ ] Character identity consistent across all shots (repeat anchors every shot)
- [ ] Dialogue preserved verbatim in `<d>[Language] ...</d>` format
- [ ] Style opener present before `[Shot 1]`
- [ ] Reference labels (Ref2VA only) used consistently and never invented beyond subject_definitions
- [ ] Duration (4–15s) and aspect ratio specified or assumed-and-flagged
- [ ] No named IP, celebrities, or trademarked character names
- [ ] Diegetic music in shot descriptions, non-diegetic score in `non_diegetic_music`

---

## FILE: references/base-multishot-format.md (https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill/blob/HEAD/references/base-multishot-format.md)

# H3 Base-Mode Multi-Shot Format Reference

Complete specification for MiniMax H3-Base video model (checkpoint FL2VA), covering T2VA / I2VA / FL2VA / L2VA modes. Load this when generating Base-mode prompts.

---

## Role

You are producing a structured, multi-shot prompt that H3-Base can consume directly. You replicate MiniMax's H3-Context-IR preprocessor for base modes: take a free-form brief + target parameters and output ONE structured, multi-shot prompt. Your specialty: decomposing a brief into a clean, timed multi-shot timeline.

## Input You Receive

- **BRIEF**: free-form intent (story, action, scene, dialogue, style). Any language.
- **MODE**: `t2va` | `i2va` | `fl2va` | `l2va` (default `t2va` if absent).
- **KEYFRAMES** (i2va/fl2va/l2va only): short descriptions of the first-frame and/or last-frame images actually attached to the generation node.
- **TARGET**: duration_s (integer 4–15), aspect ratio, optionally desired shot count or shot plan.

## Output Contract (Absolute)

1. Output ONLY the fields below, in order, with exact lowercase field names followed by a colon. No preamble, no explanation, no markdown fences.
2. For i2va / fl2va / l2va, the FIRST line is the instruction line (exact templates below), then ONE blank line, then the three core fields. For t2va there is no instruction line.
3. Write everything in English. EXCEPTIONS: dialogue/lyrics inside `<d>` and visible on-screen text keep their original language verbatim.
4. All timestamps use MM:SS.mmm, are strictly increasing, and fall within duration_s. Convert duration to seconds with two decimals wherever S.SS appears (e.g., 8 s → 8.00).

---

## Instruction Line (first line, keyframe modes only — copy the template exactly)

### i2va
```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

### fl2va
```
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.
```

### l2va
```
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.
```

(N = index of the final shot; S.SS = effective duration to two decimals.)

---

## Core Fields

### `integrated_multimodal_description`
The timed multi-shot timeline. See Multi-Shot Planning Rules below.

### `overall_soundscape`
1–4 English sentences, one paragraph: ambience + physical action sounds + non-verbal human sounds across the FULL video. No dialogue/singing/diegetic music here. N/A only for explicit total silence.

### `non_diegetic_music`
1–3 English sentences: score the CHARACTERS CANNOT hear — instrumentation, tempo, rhythm, dynamics only; no mood words. Music audible to characters is diegetic → belongs in the shot description. N/A when no score.

---

## Multi-Shot Planning Rules

### Shot Budgeting
| Duration | Shot Count |
|---|---|
| 4–6 s | 1–2 shots |
| 7–10 s | 2–3 shots |
| 11–15 s | 3–5 shots |

Respect an explicit user shot count/plan first. Each shot needs at least ~1.5–2.0s to breathe. Never cram more than ONE dominant action into a shot. Distribute detail by information load — a single-shot video still deserves a full description.

### Timeline Syntax
- `[Shot 1]` has NO timestamp and MUST open with the overall style + initial composition.
- Styles: Cinematic, live-action, 2D-animated, 3D CG, claymation, watercolor, vintage film. For keyframe modes, derive style from the reference image; for t2va, from the brief.
- Later shots: `[Shot N] At MM:SS.mmm, the camera cuts to ...` — strictly increasing cut times.
- Cut verbs: "the camera cuts to", "the shot cuts to", "the shot transitions to", "the shot changes to", "the shot switches to". Cross-dissolve/fade/wipe ONLY when the user explicitly asked.

### Cut Logic (when to cut vs. move the camera)
- A cut must introduce NEW information: new subject, new space, new state, new viewpoint, or new time.
- If only the framing distance or a slight angle changes → use camera motion inside the current shot, NOT a cut.
- End each shot on a beat the next shot can pick up (action mid-motion, a glance, a sound cue).

### Continuity Across Shots (multi-shot discipline)
- **Repeat identity anchors EVERY shot**: the subject's appearance, clothing, key props — phrase them freshly but consistently.
- **Track state changes**: what got wet/opened/taken/broken in shot N stays that way in shot N+1.
- **Spatial logic**: preserve screen direction and relative positions across cuts unless a shot deliberately re-establishes geography.
- **Audio continuity**: ambience flows across cuts; a sound or line crossing a cut uses the rules below.

---

## Shared Grammar

### Camera Motion = type + amplitude + speed (natural English, not stacked labels)
- Types: Zoom In/Zoom Out, Push In/Pull Out, Pan Left/Pan Right, Truck Left/Truck Right, Tilt Up/Tilt Down, Pedestal Up/Pedestal Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Shake Strongly, POV, Roll Clockwise/Roll Counterclockwise.
- Amplitude: "with small amplitude" / "with large amplitude" (omit if medium).
- Speed: "at slow speed" / "at fast speed" (omit if normal).

Example: `The camera pushes in with small amplitude at slow speed toward the folded letter in her hands.`

### Speakers, Dialogue, Singing
- Vocal sources get stable IDs (S1), (S2)... kept across ALL shots; group speech (S1,S2); characters who never vocalize get NO ID.
- First appearance: identity anchors (type, age, gender, on/off-screen, pitch, timbre, rate, accent).
- Format: identifying phrase + ID + delivery OUTSIDE `<d>`; inside `<d>` ONLY the language tag and the exact words:
  > `The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>`
  — preserve the user's words and punctuation verbatim; never translate or rewrite.
- Voiceover: exact phrase "says in an off-screen voiceover" + immediately state the on-screen lips stay closed: "...while his lips remain completely closed."
- Dialogue crossing a cut: `<scenetrans>` at the connecting points in both parts + explicit continuity ("continues seamlessly across the cut", "carries over from the previous shot"). Speech cut off by the video end: `<cutoff>`.

### On-screen Text
Visible banners/signs/labels/subtitles/neon in English double quotation marks, verbatim, no translation:
> A red neon sign reading "营业中" glows above the doorway.

---

## Per-Mode Body Strategy

### t2va (text-to-video)
Build the full timeline from the brief; you may add consistent scene/character/sound details.

### i2va (image-to-video)
`[Shot 1]` anchors on the first frame — establish the image's style, subjects, composition, scene anchors, preserving identity/clothing/colors/key objects/spatial relationships — then develop forward:
> first-frame anchor → action onset → continuous development → result/reaction

Direct MOTION; do not re-describe static frame content.

### fl2va (first-last-to-video)
**DEFAULT TO A SINGLE SHOT** so the model interpolates continuously first→last; use multiple shots ONLY when the user explicitly specifies them. Body structure:
> first-frame state → observable intermediate changes → progressively narrowing differences → last-frame state, landed by the final `[Shot N]`

### l2va (last-frame-to-video)
> infer a plausible preceding state → explicit action/transition path → gradual convergence in the final shot → exact landing on the last-frame image (arrangement, position, camera angle, lighting, composition)

---

## Fidelity Rules

- Stay on the user's intent; supplement missing details only when consistent with the brief.
- Conform silently to hard constraints (duration 4–15s, shot budget); never explain.
- Avoid named third-party IP, real celebrities, trademarked characters — describe generically.
- Default dialogue language when unspecified: `<d>[English]`.

---

## Worked Example (t2va, 8s, multi-shot)

**Input:**
```
BRIEF: "A baker opens his street bakery before sunrise, proud of the first loaf; one warm line."
MODE: t2va
TARGET: duration_s=8, 16:9, 3 shots
```

**Output:**
```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a small street bakery before sunrise, cool blue dawn light outside and warm tungsten light inside. A middle-aged baker in a flour-dusted apron opens the wooden shutters. The camera pushes in with small amplitude at slow speed as morning light spills across the counter. [Shot 2] At 00:03.500, the camera cuts to a medium close-up of the baker placing a fresh loaf on the wooden counter, steam rising from the crust. The baker with a calm, slightly raspy voice (S1) says: <d>[English] First batch of the morning.</d> [Shot 3] At 00:05.800, the camera cuts to a close-up of steam rising from the sliced bread, the crumb glowing in the warm light, while the satisfaction of the baker's final words carries over from the previous shot.

overall_soundscape: Wooden shutters scrape open over a quiet street while trays clink softly inside the bakery. A doorbell rings once, followed by light footsteps and the crisp sound of bread being sliced.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and a gentle fade at the end.
```

---

## Suggested User-Message Template (for ComfyUI LLM nodes)

```
BRIEF:
<free-form story/scene intent, any language>

MODE: t2va          # t2va | i2va | fl2va | l2va

KEYFRAMES:          # only for i2va / fl2va / l2va
first_frame: <short description of the attached image>
last_frame: <short description, fl2va/l2va only>

TARGET:
duration_s: 8
ratio: 16:9
shots: 3            # optional; omit to let the planner decide
```

**Wiring tips:** LLM output string → H3 node's prompt input. For t2va runs, omit KEYFRAMES entirely. For fl2va, remember the single-shot default — request multiple shots explicitly in BRIEF if you want them. If the node truncates context, drop the EXAMPLE section first.


---

## FILE: references/creative-showcase.md (https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill/blob/HEAD/references/creative-showcase.md)

# H3 Creative Showcase — Advanced Long-Form Prompt Patterns

Quality benchmarks and transferable patterns from the user's advanced H3 prompts. These represent the depth of cinematic thinking the enhancer should bring to every prompt, regardless of mode. Study these to calibrate your creative enhancement quality.

---

## Pattern 1: Long-Form Narrative Storytelling (15s Multi-Shot Montage)

Use for: day-in-the-life, vlog aesthetic, montage, narrative progression across locations.

### Structural Blueprint

A long-form storytelling prompt opens with three explicit aesthetic blocks before the storyboard, then lays out per-shot detail with voiceover integration.

**CAMERA block** — defines the physical camera identity:
```text
CAMERA: Handheld DV 16mm camcorder aesthetic. CHASE records herself throughout every location, mostly holding the camera directly and occasionally setting it down for brief hands-free shots. Preserve natural hand tremors, crooked composition, slow autofocus correction, awkward zooms, accidental face cropping, and moments where the framing briefly loses her. The physical camcorder is never visible.
```

Transferable elements:
- Physical camera type tied to a narrative reason (she's filming herself)
- Named imperfections to preserve: hand tremors, autofocus hunting, exposure shifts, awkward zooms, framing loss
- What's NOT shown (the camera itself is never visible)

**LOOK block** — defines image quality and color science:
```text
LOOK: Authentic soft tape-image texture with mild blur, subtle electronic grain, glowing highlights, small automatic-exposure fluctuations, restrained contrast, and natural skin tones. The lighting changes realistically between locations: warm morning light in the apartment → cooler daylight inside the car → intense colorful lighting backstage and onstage.
```

Transferable elements:
- Texture vocabulary: tape-image, mild blur, electronic grain, glowing highlights
- Named exposure/contrast behavior
- Lighting transitions mapped to each location change with arrow notation

**STYLE block** — defines pacing, energy, and editing rhythm:
```text
STYLE: A fast, energetic day-in-the-life montage with sharp cuts between locations and slightly accelerated transitional movement. Reflective voiceover plays over the footage instead of synchronized dialogue. The pace gradually builds from a quiet sleepy morning into a high-energy performance finale.
```

Transferable elements:
- Pace described as an arc: quiet→energetic, with a named turning point
- Voiceover vs. synchronized audio distinction
- Editing rhythm: sharp cuts, accelerated transitions

### Character Description Pattern

```text
CHASE — an exceptionally beautiful Instagram influencer and rising pop performer in her 20s. Long glossy dark-brown hair, striking symmetrical features, large expressive eyes, flawless glowing skin, and soft pink lips. Slim, toned figure. Her outfit changes with each location: cozy loungewear at home, stylish casual clothing during the ride, and a glamorous performance outfit for the final scene. Each outfit fully covers her arms and torso.
```

Transferable elements:
- Age range + role/occupation for instant identity
- Physical features with specificity (hair color+texture, eye description, skin, lips)
- Wardrobe progression mapped to each location/shot
- Coverage note: "Each outfit fully covers her arms and torso" — always describe the level of coverage explicitly

### Setting Progression

```text
Setting Progression
Stylish apartment bedroom in the morning → luxury van interior during the day → busy backstage area and concert stage at night.
```

Transferable: Arrow notation mapping time + space progression across the full video.

### Per-Shot Storyboard with Voiceover

```text
Storyboard — 15 seconds, 5 cuts

(~3s, bedroom, camera resting on a table, soft morning light) She slowly sits up, stretches, fixes her messy hair, and hurriedly packs a small bag. VOICEOVER (CHASE): "Some mornings begin before I'm even fully awake."

(~3s, van interior, handheld selfie footage, natural window light) She sits beside the window as sunlight moves across her face, casually checking messages on her phone. VOICEOVER (CHASE): "I spend so much time on the road, this place almost feels like home."

(~3s, van interior, macro detail shot) Close-up of her fingers choosing a song from a playlist while passing sunlight flickers across the screen and her hand. No voiceover, only natural road noise.

(~3s, backstage, fast handheld montage) Quick glimpses of makeup brushes, hair styling, costume adjustments, staff rushing past, and her taking one focused breath before going onstage. VOICEOVER (CHASE): "And then everything suddenly starts moving at once."

(~3s, stage, wide shot moving into a close-up, energetic finale) Bright stage lights ignite as she steps through the curtains. The camera catches her silhouette, then a brief confident smile before the image cuts to black. VOICEOVER (CHASE): "This moment is why I keep doing it."
```

Transferable elements:
- Each shot: `(duration, location, camera position, lighting)` header
- ONE dominant action per shot (sits up + packs / checks phone / fingers choosing song / backstage montage / steps through curtains)
- VOICEOVER lines placed at end of shot with attribution
- Silent shots noted explicitly ("No voiceover, only natural road noise")
- Macro/detail shots included for rhythm variety
- Pacing builds: sleepy morning → quiet ride → intimate detail → kinetic backstage → explosive stage finale

---

## Pattern 2: Action Choreography (Ref2VA, 7 Shots, 15s)

Use for: combat, chase, sports, any sequence with spatial movement and character interaction.

### Character Color Lock System

Each character gets a signature color that governs their visual effects, trails, reflections, and environmental impact throughout the entire sequence. This is the single most powerful continuity tool for action prompts.

```text
TSUBAME (Image 1) — Young woman, sharp jawline, shoulder-length brown bob; wearing a glossy purple latex crop-top vest with silver zipper and black leather harness straps; sleek, agile build. Her movement leaves trailing neon-purple afterimages and light streaks.
GARAN (Image 2) — Tall, imposing figure in a metallic teal/seafoam green high-collar jacket over a white ribbed bodysuit; calm but dangerous expression. His heavy impacts generate teal shockwaves and digital distortion ripples.
```

**Color Lock rules:**
- Assign each character a distinct color at definition (Tsubame = Electric Purple/Magenta; Garan = Teal/Cyan)
- That color appears on: movement trails, glints/reflections, impact effects, environmental bounce light
- State the color lock explicitly: `Color Lock: Tsubame = Electric Purple/Magenta (trails, glints); Garan = Teal/Cyan (jacket sheen, shockwaves)`
- The environment reflects the character colors: "purple bounce on walls from her, teal reflections from him"

### Style Block for Action

```text
Style: High-fidelity 3D CGI animation, stylized realism (like Arcane or Spider-Verse). Glossy material rendering (latex/metal), volumetric neon lighting, motion blur on fast objects. Color Lock: Tsubame = Electric Purple/Magenta (trails, glints); Garan = Teal/Cyan (jacket sheen, shockwaves). World: Sleek, futuristic sci-fi corridor with circular archways and reflective floors. Lighting is driven by their suits—purple bounce on walls from her, teal reflections from him. Camera: Dynamic action cam—dolly zooms, whip pans, low angles, speed-ramping.
```

Transferable elements:
- Style reference points (named similar media for instant tonal calibration)
- Material rendering directives: "Glossy material rendering (latex/metal)"
- Lighting source tied to character colors: "Lighting is driven by their suits"
- Camera personality: "Dynamic action cam" with technique list

### Spatial Layout System

For action sequences, explicitly define the geography so movement directions are unambiguous across all shots:

```text
SPATIAL LAYOUT (MAIN VIEW = Corridor Axis): Tsubame starts at the far end (Screen Center-Deep) running toward camera. Garan intercepts from Screen Right. Action flows Deep → Front, then Left → Right.
```

**Action Vectors** — the 2–3 critical motion moments:
```text
ACTION VECTORS:
(V1, SHOT 2) Garan slams fist into floor → Teal shockwave travels Front→Back; Tsubame jumps over the wave.
(V2, SHOT 4) Tsubame slides under Garan's sweeping arm → Purple trail marks the slide path.
(V3, SHOT 6) Close combat lock → Purple glove vs Teal sleeve, sparks fly at contact point.
```

### Progressive Continuity Tracking

State what changes across the sequence so damage and wear accumulate visibly:

```text
Continuity: Tsubame's hair gets messier/windblown progressively. Garan's jacket shows scuff marks/dust after Shot 5.
```

### Per-Shot Choreography Pattern

Each shot in an action sequence needs: shot name + duration + camera + composition + ONE key action + visual effect + emotional beat.

```text
[SHOT 1] · The Stare Down (~2s): Static wide shot, symmetrical composition. Long futuristic corridor with glowing arches. Tsubame stands center-frame, distant, fists clenched, purple vest gleaming under overhead lights. Garan steps into frame from Right foreground, back to camera, teal jacket reflecting the hallway lights. He cracks his knuckles. Tense atmosphere.

[SHOT 2] · The Shockwave (~2s): Low angle, dynamic tracking. Garan stomps forward. A visible teal energy ripple explodes outward from his boot across the reflective floor. Tsubame sprints forward and leaps vertically, clearing the wave. Her purple trails hang in the air like neon ribbons. Debris floats in zero-G for a split second.

[SHOT 3] · The Interception (~2.5s): Handheld shaky cam. Tsubame lands and dashes Screen Left → Screen Right. Garan swings a massive backhand. She ducks under it with fluid grace, her brown bob whipping around. The teal jacket blurs as it passes inches above her head. Speed lines emphasize velocity.

[SHOT 4] · The Slide (~2.5s): Extreme low angle, floor-level. Tsubame goes into a baseball slide across the polished floor, passing directly between Garan's legs. Her purple latex vest reflects the floor lights intensely. Garan tries to stomp down, but she's already past, kicking up a spray of digital sparks (purple/teal mix).

[SHOT 5] · The Wall Run (~1.5s): Vertical pan up. Tsubame runs horizontally along the curved corridor wall (Screen Right), defying gravity. Garan punches the wall where she was a fraction of a second ago—cracks spiderweb out in teal light. She pushes off the wall, launching herself back toward the center.

[SHOT 6] · The Clash (~1s): Extreme close-up, slow-motion impact frame. Tsubame's purple-gloved fist meets Garan's teal-sleeved forearm. The collision point blooms with white-hot light and particle sparks. Their faces are inches apart—her fierce determination vs his stoic focus. Background is a blur of motion streaks.

[SHOT 7] · The Aftermath (~3.5s): Wide shot, camera pulls back rapidly. Tsubame lands in a superhero crouch (Screen Left), sliding to a halt, purple trails fading. Garan stands firm (Screen Right), adjusting his teal collar, unbothered but acknowledging her speed. Dust motes dance in the neon light. The corridor behind them is slightly damaged (cracks, scorch marks). Both breathe heavily. Fade out.
```

Transferable per-shot elements:
- **Named shots**: "The Stare Down", "The Shockwave", "The Clash" — gives each beat a clear identity
- **Camera per shot**: static wide / low angle tracking / handheld shaky / extreme low angle / vertical pan / extreme close-up slow-motion / wide pull-back
- **Color effects in every shot**: purple trails, teal shockwave, teal jacket blur, purple vest reflection, teal spiderweb cracks, white-hot bloom, fading purple trails
- **One key action per shot**: stare → shockwave jump → dodge → slide → wall run → clash → aftermath
- **Emotional contrast**: "her fierce determination vs his stoic focus"

### Environmental Reactivity & Technical Footer

```text
Environmental activity: Holographic ads flicker on walls reacting to their movement. Floor is highly reflective (mirror-like).
Audio: Heavy bass synth track. SFX: Electric hums, heavy thuds, glass-like shattering sounds, neon buzz.
Technical: 15 seconds. 16:9 aspect ratio. No text overlays. Consistent character models (purple vest/black straps for her, teal jacket/white undersuit for him).
```

---

## Transferable Quality Checklist

When enhancing ANY prompt, check if you've applied these patterns where relevant:

- [ ] **Camera identity** — not just "handheld" but WHY handheld and what imperfections to preserve
- [ ] **Visual texture vocabulary** — specific grain/blur/exposure language, not just "cinematic"
- [ ] **Lighting transitions** — if locations change, describe the light shift at each transition
- [ ] **Pacing arc** — named build pattern (quiet→energetic, tense→release) with turning point
- [ ] **Character visual signature** — a recurring color or visual element for instant recognition
- [ ] **Wardrobe progression** — outfits mapped to locations/time with materials described
- [ ] **Spatial geography** — screen directions stated for multi-character or action sequences
- [ ] **Color lock** (action) — each character's effects/trails/reflections have a consistent color
- [ ] **Progressive continuity** — what accumulates (damage, mess, dust, exhaustion) across shots
- [ ] **Environmental reactivity** — the world responds to character action (flicker, crack, reflect)
- [ ] **Named shots** — each shot has a memorable identity beyond "Shot 3"
- [ ] **Emotional beats** — expressions and body language described, not just physical action
- [ ] **Sound design layers** — ambience, physical SFX, diegetic music, non-diegetic score all mapped
- [ ] **Voiceover discipline** — VO clearly marked, lips-closed noted, emotional narration over visuals

---

## Converting Showcase Patterns to H3 Format

The showcase examples above use a creative-brief style (CAMERA/LOOK/STYLE/Storyboard blocks). When converting to actual H3 format output:

1. **Camera identity** → weave into the style opener sentence and each shot's camera motion line
2. **LOOK/visual texture** → style opener + environment/lighting descriptions per shot
3. **STYLE/pacing** → reflected in shot count, timing distribution, and cut rhythm
4. **Character descriptions** → `subject_definitions` entries (Ref2VA) or identity anchors in each shot (Base)
5. **Storyboard shots** → `[Shot N]` entries with timestamps, camera motion, one dominant action each
6. **Color lock** → embedded in character descriptions and per-shot visual effect descriptions
7. **Spatial layout / action vectors** → woven into shot descriptions with screen direction language
8. **Continuity progression** → tracked across shots in the detailed description
9. **Environmental reactivity** → described within relevant shots
10. **Voiceover** → formatted as `"says in an off-screen voiceover"` + `<d>[Language] ...</d>` + lips-closed note
11. **Sound design** → split between `overall_soundscape` (ambience/SFX) and `non_diegetic_music` (score)

---

## Pattern 3: Action Choreography — Wing Chun Street Fight (T2VA, 15s, 5 shots)

A grounded martial-arts fight emphasizing realistic technique, era-specific atmosphere, and progressive combat exhaustion. Shows how to choreograph a one-versus-many fight within the 5-shot budget.

### Key Patterns
- **Era-specific styling**: 1950s New York noir — cobblestone, fire escapes, fedoras, wool overcoats, film grain
- **Technique-specific choreography**: Named Wing Chun techniques (pak sao, tan sao, chain punches) ground the action in realism
- **One-dominant-action-per-shot discipline**: Each shot handles exactly one attacker exchange
- **Progressive exhaustion**: Yi Man calm throughout; attackers accumulate on the ground shot by shot
- **Environmental sound design**: Rain drumming, cobblestone splashes, cloth impacts — no music during combat, score cuts to silence on the final takedown

### Output

```
integrated_multimodal_description: [Shot 1] Cinematic live-action, 1950s film-noir aesthetic with high-contrast chiaroscuro lighting, shallow depth of field, and visible 35mm film grain. A wide static shot establishes a narrow New York back alley at night in heavy rain — wet cobblestone ground reflecting a single overhead streetlamp, brick walls on both sides dripping with water, iron fire escapes climbing the left wall, and a chain-link fence partially visible at the far end. YI MAN, a lean Chinese man in his late thirties with a calm composed expression, short black hair slicked from the rain, wearing a dark charcoal traditional Chinese cotton jacket with cloth buttons, dark trousers, and thin-soled cloth shoes, stands alone in the center of the alley under the streetlamp glow, hands at his sides. Four men in 1950s street clothing — a heavyset man in a brown wool overcoat and fedora, a wiry man in a leather jacket and flat cap, a broad-shouldered man in a dark-grey raincoat, and a younger man in a white undershirt and suspenders — spread out in a semicircle facing him from Screen Right to Screen Left, approximately five meters away. Rain pours steadily. The camera pushes in with small amplitude at slow speed toward Yi Man, isolating him against the dark alley, then holds.
[Shot 2] At 00:03.000, the shot cuts to a medium tracking shot at fast speed moving with Yi Man as he closes the distance toward the wiry man in the leather jacket and flat cap on Screen Right. Yi Man's hands rise into a relaxed Wing Chun guard — elbows tucked, hands open at centerline. The wiry man in the leather jacket throws a wide right hook. Yi Man slips inside the arc with a compact lateral step, his left hand executing a pak sao (slapping block) that deflects the hook outward, and immediately fires a chain of three straight punches into the man's chest and jaw using vertical-fist Wing Chun technique. The impacts are fast, precise, and audible over the rain. The wiry man stumbles backward into the brown-overcoat man behind him. Rain sprays off their shoulders with each impact. The camera shakes slightly during the exchange.
[Shot 3] At 00:06.000, the shot cuts to a low-angle medium shot from ground level, tilted up at Yi Man as the broad-shouldered man in the dark-grey raincoat and the younger man in the white undershirt and suspenders rush him simultaneously from Screen Left. Yi Man pivots on the balls of his feet, his thin-soled cloth shoes gripping the wet cobblestone. He intercepts the younger man's lunging grab with a tan sao (palm-up blocking arm) that redirects the momentum past his right side, then strikes the younger man's solar plexus with a sharp vertical punch. The broad-shouldered man in the raincoat swings a heavy haymaker. Yi Man ducks under it with a slight knee bend and delivers a low Wing Chun front kick to the raincoat man's lead knee, buckling his stance. Both attackers stagger. Water splashes from puddles beneath their feet. The camera arcs right with medium amplitude at fast speed, tracking the redirection of force.
[Shot 4] At 00:09.000, the shot cuts to a close-up handheld shot with strong shake, framing Yi Man from chest to head as the heavyset man in the brown wool overcoat and fedora — the last one standing — grabs Yi Man from behind in a bear hug, locking his arms at his sides. Yi Man's expression stays focused, not panicked. He drops his weight low, stomps his heel onto the heavyset man's instep, and as the grip loosens, throws a sharp backward elbow strike to the man's ribs, followed by a spinning backfist that connects with the heavyset man's jaw. The fedora flies off into the rain. The heavyset man's eyes glaze and he drops to his knees on the wet cobblestone, then falls forward. Rain streaks across the frame. Yi Man's dark charcoal jacket is now darkened with rain across the shoulders and forearms. The camera pulls back with small amplitude to reveal two other men on the ground behind Yi Man and one slumped against the brick wall.
[Shot 5] At 00:12.000, the shot cuts to a wide static shot from the far end of the alley, behind the fallen attackers. All four men lie on the wet cobblestone — two face-down in puddles, one slumped against the brick wall holding his ribs, and the heavyset man in the brown overcoat face-up with his fedora floating in a nearby puddle. Yi Man, the lean Chinese man in his dark charcoal Chinese cotton jacket, stands alone in the center of the alley under the streetlamp, his breathing slightly elevated but his expression composed and still. He lowers his hands slowly to his sides, rolls his shoulders once, and turns to walk toward the far end of the alley into the rain and shadow. The wet cobblestone reflects his retreating silhouette and the dim glow of the streetlamp. Rain continues to fall steadily. The camera holds on the empty alley as the last of the fallen men groans faintly.

overall_soundscape: Heavy rain pours continuously throughout the scene, striking cobblestone, brick walls, metal fire escapes, and puddles with a relentless drumming hiss. Rapid footsteps shuffle and slap on wet stone as fighters reposition. Fists strike cloth and flesh with dull, muffled thuds, and a heavy body hits the wet ground with a wet slap. A faint groan of pain is audible from a fallen attacker in the final shot.

non_diegetic_music: A brooding tension-building score begins with a low sustained double bass drone and sparse, dissonant piano notes that hang in the air during the standoff. As the fighting erupts, a driving rhythmic pulse enters — staccato strings and a muted taiko drum pattern at a fast tempo that accelerates through the combat shots, then cuts abruptly to silence when the last attacker falls, leaving only the rain.
```

---

## Pattern 4: Samurai vs Ninja — Chambara Sword Duel (T2VA, 15s, 5 shots)

A Japanese period-piece duel emphasizing tension-building pacing, stealth choreography, and the chambara film tradition of stylized violence. Shows how to handle a one-on-one fight that ends in assassination rather than victory.

### Key Patterns
- **Chambara aesthetic**: Black-and-yellow moonlit tones, volumetric fog, stone lanterns, temple courtyard
- **Pacing arc**: Frozen tension → rapid exchanges → sudden stealth → intimate kill → cold aftermath
- **Camera-as-character**: Lateral tracking matches dodge rhythm; camera roll matches ninja spin; slow push to impact point for the kill
- **Single turning point**: One taiko drum boom at the tanto draw divides the entire video into suspense and consequence
- **Stylized violence**: Blood appears near-black in moonlight (chambara tradition), one arc spray, not gratuitous
- **Environmental shift**: Fog thickens across shots; rain absent until the kill, then begins lightly — atmospheric punctuation

### Output

```
integrated_multimodal_description: [Shot 1] Cinematic live-action, Japanese chambara film aesthetic with high-contrast black-and-yellow moonlit tones, volumetric fog drifting between dark wooden architecture, shallow depth of field, and fine 35mm film grain. A wide static shot establishes a narrow temple courtyard at night — wet stone tiles reflecting cold moonlight, a stone lantern glowing faintly on the left, wooden temple walls with sliding screen doors on both sides, and low hanging mist clinging to the ground. THE SAMURAI, a tall Japanese man in his forties with a stern weathered face, topknot hairstyle, wearing dark indigo armor plates laced with black silk cord over a layered grey kimono, dark hakama trousers, and straw sandals, stands in the center of the courtyard gripping a katana in a two-handed middle guard, blade raised at eye level pointing forward. Opposite him, ten meters away, THE NINJA — a shorter lean figure wrapped entirely in dark charcoal-black cloth with only a narrow slit for the eyes, black tabi boots, a short ninjato sword strapped across his back — crouches low in a side stance, motionless. Fog drifts between them. The camera holds on this standoff for a beat, then pushes in with small amplitude at slow speed toward the gap between them. Moonlight glints off the samurai's katana blade.
[Shot 2] At 00:03.000, the shot cuts to a medium tracking shot moving at fast speed laterally from Screen Left to Screen Right, following the exchange. The samurai — in his dark indigo armor and topknot — steps forward and swings his katana in a powerful downward diagonal cut. The ninja — wrapped in black cloth, only eyes visible — pivots left, the blade missing his shoulder by inches, moonlight flashing on the steel as it slices through fog. The samurai immediately reverses into a horizontal side cut. The ninja drops low and ducks right, the katana whistling over his head, his black tabi boots sliding on the wet stone tiles. A third overhead strike — the ninja sidesteps left again with fluid precision, each evasion minimal and efficient. Sparks fly where the katana tip grazes a stone tile. The camera tracks the ninja's lateral dodging path, panning left then right then left with medium amplitude at fast speed, matching his evasion rhythm. The samurai's grey kimono sleeves snap with each swing.
[Shot 3] At 00:06.500, the shot cuts to a low-angle close-up from ground level tilted up at the ninja as he executes a sudden spinning side-turn — his body drops almost parallel to the wet stone tiles, black cloth rippling, as he rotates beneath the samurai's reaching guard in a tight spiral. The camera rolls clockwise with small amplitude at fast speed to match the spin, capturing the ninja's body sweeping past the samurai's armored legs in a blur of dark fabric. Wet stone tiles and mist fill the foreground. The ninja's tabi boots push off the ground with a muffled scrape. In a fraction of a second he has passed entirely behind the samurai, who is still mid-recovery from his missed strike, his katana extended forward into empty air. The ninja rises silently behind the samurai's right shoulder, his movement invisible to his opponent.
[Shot 4] At 00:09.000, the shot cuts to a close-up framed on the ninja's hands from behind the samurai's right shoulder. The ninja's right hand, wrapped in black cloth, draws a tanto — a short dagger with a plain dark wooden handle and a thin gleaming steel blade — from a concealed sheath at his lower back in one swift upward pull. The blade catches a sliver of moonlight. In a single fluid motion, the ninja drives the tanto forward into the exposed side of the samurai's neck, just above the indigo armor plates and below the topknot. The blade sinks in to the hilt. The samurai's eyes widen — his stern composure breaks into shock, his mouth opening slightly, his katana grip going slack. His fingers loosen on the katana handle. The camera pushes in with small amplitude at slow speed toward the point of impact, the tanto handle visible against the samurai's grey kimono collar and dark indigo armor lacing.
[Shot 5] At 00:11.500, the shot cuts to a medium shot, slightly low angle, framing both figures from the front as the ninja — still behind the samurai's right shoulder — pulls the tanto blade out in a sharp horizontal draw. Blood sprays from the samurai's neck wound in an arc that catches the cold moonlight, the droplets appearing almost black against the lit stone tiles and mist. The samurai's knees buckle, his dark indigo armor plates clacking as his posture collapses. His katana slips from his fingers and clatters onto the wet stone. The ninja steps back two paces into the fog, tanto held at his side, blood on the blade glinting faintly. The samurai — tall, topknot, grey kimono, indigo armor — sinks to his knees, one hand rising weakly toward his neck, then falls forward onto the wet courtyard tiles. Fog rolls over his fallen body. The camera holds static on the scene as the ninja turns and dissolves into the dark shadow between the temple walls, leaving only the stone lantern glow and the fallen samurai in frame. Rain begins to fall lightly on the blood-spattered stone.

overall_soundscape: A heavy silence dominates the temple courtyard, broken only by the faint scrape of straw sandals and tabi boots shifting on wet stone tiles. The katana cuts through cold air with a sharp metallic whistle on each swing, and stone sparks ping faintly when the blade grazes tile. A muffled cloth rustle signals the tanto being drawn from its hidden sheath, followed by a wet, subdued impact on the neck strike. The katana clatters on stone as it drops, and armor plates clack sharply when the samurai's body collapses. Light rain begins tapping the courtyard in the final seconds.

non_diegetic_music: A single sustained high note on a bamboo flute holds through the standoff and sword exchanges, tense and airy. When the tanto is drawn, a deep taiko drum strikes once — a single boom that resonates and decays through the neck strike. As the samurai falls, a sparse, mournful shamisen pluck enters at a very slow tempo, its dissonant tone hanging unresolved as rain begins to fall and the scene fades.
```

---

## Pattern 5: Anime Fight — SeeDance to H3 Conversion (T2VA, 15s, 5 shots)

A 2D anime-style fight converted from a SeeDance-format prompt. Shows how to consolidate 7 timestamped segments into the H3 5-shot budget without losing any choreography beat, and how to convert external prompt formats into H3 structure.

### Key Patterns
- **Segment consolidation**: 7 SeeDance segments → 5 H3 shots by merging the orbit-into-clinch and snap-impact sequences
- **Character color lock**: BULLDOG = burnt orange/rust/gold; VIPER = olive green/black/white tape — each character's palette consistent across every shot
- **Anime aesthetic specification**: 2D semi-realistic concept art, cel shading, clean lineart, 24fps motion feel, subtle film grain
- **Camera style preservation**: Alternating low-angle wides with violent snap-in extreme close-ups, whip pans, handheld energy — no smooth glides
- **Slow-motion beat**: Brief slow-mo at the apex of the dual lunge, then hard snap back to full speed
- **Audio architecture**: Percussive drums building intensity synced to strikes, bass drop on final impact, then cut to near-silence with only breathing

### Output

```
integrated_multimodal_description: [Shot 1] 2D semi-realistic anime concept art, clean confident lineart, painterly cel shading with soft gradients, subtle film grain, 24fps motion feel. A muted palette of burnt orange, olive green, warm gray concrete, and dusty sunset gold. A low-angle wide full shot frames a gritty back-street lot at sunset — cracked concrete ground, chain-link fence topped with razor wire, graffiti-covered walls, palm trees, and parked cars. Warm orange dusk light casts long hard shadows across the lot. BULLDOG — a heavyset, powerful woman with broad shoulders, deep tan skin, buzzed rust-orange fade hair, gold hoop earrings, a nose ring, wearing a burnt-orange cropped tank top, a thin gold chain, baggy gray sashed pants, blood-stained white hand wraps, gold knuckle rings, and gray high-top sneakers — and VIPER — a lean, athletic woman with deep tan skin, long black ponytail with blunt bangs, green eyes, a small black shoulder tattoo, wearing an olive sports bra, black shorts, black fingerless gloves over white taped wrists, tan ankle braces, and white sneakers — face each other at mid-distance in profile, guards raised, circling clockwise. Sneakers scuff on gravel. The camera tracks right at slow speed, orbiting with their circle, keeping both fighters fully in frame against the hazy sunset backdrop.
[Shot 2] At 00:03.000, the shot cuts to a tight over-the-shoulder shot from behind BULLDOG, still in her burnt-orange tank top and gold knuckle rings, looking past her wrapped fist toward VIPER. BULLDOG fires a heavy straight punch — her blood-stained white hand wrap and gold knuckle ring passing close by the lens in motion blur. VIPER, her green eyes tracking the fist, slips it with a sharp head snap to the side, her long black ponytail whipping across the frame. The punch misses by inches. The camera shakes slightly with handheld energy, emphasizing the miss. Dust motes float in the warm sunset light between them.
[Shot 3] At 00:06.000, the shot whip-pans to a medium full shot from the opposite side. VIPER counters immediately — her white-taped wrists and black fingerless gloves firing a fast one-two straight combination into BULLDOG's guard, followed by a sharp rising knee. BULLDOG, in her burnt-orange tank top and gold chain, absorbs the strikes on crossed forearms, her heavyset frame driven back two heavy steps, gray high-top sneakers scraping cracked concrete, dust kicking up from the impact. The camera tracks backward at fast speed with them, low horizon line, both fighters' bodies fully visible, long sunset shadows stretching behind them. BULLDOG's rust-orange fade and gold hoop earrings catch the warm light as she steadies.
[Shot 4] At 00:09.000, the shot cuts to a dynamic low wide shot. Both fighters — BULLDOG in burnt-orange and gray, VIPER in olive and black — lunge at each other simultaneously, bodies stretched long, one foot off the ground each, silhouetted against the bright hazy sunset sky. At the apex of the lunge the motion shifts to brief slow motion, sunset light rim-lighting both figures, then snaps hard back to full speed as their fists collide against each other's guards. The impact sends dust and gravel spraying. The camera then orbits tightly around the resulting clinch — chain-link fence and graffiti wall sweeping through the background as BULLDOG and VIPER trade short hooks and elbows at close range, shoulders grinding, feet shuffling on cracked concrete, ponytail and rust-orange fade visible in the tight orbit.
[Shot 5] At 00:11.500, the shot cuts to an extreme close-up as VIPER's taped fist — white tape over black fingerless glove — drives directly toward the lens, filling the frame in heavy motion blur. The punch lands. The camera snaps instantly to BULLDOG's face in extreme close-up — her head torquing violently sideways, sweat flying from her buzzed rust-orange fade, gold hoop earring swinging, eyes squeezed shut, the frame shaking hard on contact. At 00:13.500 the camera pulls back fast to a low-angle wide shot. BULLDOG drops to one knee on the cracked pavement, head down, shoulders heaving in her burnt-orange tank top, gold chain dangling. VIPER stands a few paces away in her olive sports bra and black shorts, guard lowering, chest rising and falling, black ponytail settling against her back. Both fighters are backlit against the glowing sunset haze. The camera slows to a held final frame on this aftermath.

overall_soundscape: Sneakers scuff and grind on gravel and cracked concrete throughout, with dust-shifting crunches on every heavy step. Fists strike guards and flesh with sharp percussive thwacks, and fabric rustles sharply during each exchange. Heavy breathing from both fighters grows more labored as the fight progresses. Distant city traffic hum provides a low ambient bed beneath the action.

non_diegetic_music: Gritty low percussive drums build steadily in intensity from the opening standoff, layering in sharper hi-hat patterns and deeper kick hits that sync to each landed strike. The rhythm tightens and accelerates through the clinch, peaks with a heavy bass drop on the final punch impact, then cuts abruptly to near-silence — leaving only the fighters' breathing over the final held frame.
```

---

## Pattern 6: Ninja Combo Indoor — Estate Infiltration (T2VA, 15s, 5 shots)

A fast-paced indoor combat sequence in a traditional Japanese estate. Shows how to choreograph one-vs-many combat in an enclosed environment, using the room's architecture (shoji screens, low tables, ceiling beams) as both obstacle and weapon.

### Key Patterns
- **Environmental combat**: Shoji screen smashed, ceramic scattered, ceiling beam entry point
- **Multi-target choreography**: Each shot handles a distinct engagement — entry → first takedown → weapon disarm → final clear
- **Reverse-grip ninjato**: Weapon specificity grounds the action (reverse grip = close-quarters assassin style)
- **Weather as punctuation**: Rain gusts through the broken screen mark the room's breach
- **Overhead final shot**: Ceiling-beam perspective re-establishes geography and shows aftermath

### Output

```
integrated_multimodal_description: [Shot 1] Ultra-realistic cinematic live-action with shallow depth of field, volumetric lantern lighting, and rich shadow contrast. A wide static shot establishes the interior of a traditional Japanese estate at night — a spacious tatami room with sliding shoji screen doors, dark wooden ceiling beams, iron wall sconces holding flickering oil lamps, a hanging scroll on the far wall, and a low wooden table. Three SAMURAI GUARDS in dark indigo kimono and hakama, straw sandals, each carrying a sheathed wakizashi short sword at the hip, stand alert at different points — one by the shoji door at Screen Left, one center, one by the table at Screen Right. Paper lantern light casts warm amber pools across the tatami, leaving deep black shadows in the corners. THE NINJA — a lean figure in matte charcoal-black cloth wrapping the entire body and face with only a narrow slit for sharp dark eyes, black tabi boots, a ninjoto short sword strapped across his back — drops silently from the wooden ceiling beams into a crouch at Screen Left-Center between the guards. The camera pushes in with medium amplitude at slow speed toward the ninja's crouched landing. The three samurai guards turn sharply toward the movement.
[Shot 2] At 00:03.000, the shot cuts to a medium tracking shot at fast speed moving Screen Left to Right with the ninja as he explodes from the crouch into the center guard. The ninja's right hand strikes the center samurai's throat with a knife-hand blow before the guard can draw his wakizashi — the man's eyes bulge, his hands clutch his neck, and he staggers backward into the low table, scattering a ceramic cup that shatters on the tatami. The left-door guard draws his wakizashi and slashes horizontally. The ninja ducks under the blade, his black tabi boots sliding on the tatami surface, and delivers a spinning back kick to the guard's sternum that launches him through the paper shoji screen — the thin wooden lattice splinters, white paper tears, and the guard crashes into the adjoining garden walkway. Rain gusts in through the broken screen. The camera whip-pans right following the flying guard, then snaps back to the ninja.
[Shot 3] At 00:06.500, the shot cuts to a low-angle medium close-up from floor level tilted up at the ninja as the third guard — the one from Screen Right by the table — charges with his wakizashi raised overhead in a two-handed grip. The ninja draws his ninjato from his back in a single reverse-grip pull, the short black-steel blade catching lantern light. The guard brings the wakizashi down in a powerful overhead strike. The ninja steps diagonally forward-left, parries the blade aside with the ninjato's flat, and in the same motion drives the ninjato's pommel into the guard's wrist — the wakizashi spins free and clatters across the tatami. The ninja then strikes the disarmed guard with three rapid elbow-punch-elbow combinations to the jaw and solar plexus, each impact snapping the guard's head and torso in a different direction. Oil lamp light flickers wildly across the walls from the concussive air displacement. The camera arcs right with medium amplitude at fast speed, tracking around the combination, tatami and scattered debris in the foreground.
[Shot 4] At 00:10.000, the shot cuts to a tight over-the-shoulder shot from behind the ninja looking toward the broken shoji screen. The first guard — throat-struck, recovered slightly — lunges back through the torn paper screen from the garden walkway, rain streaming behind him, gripping a short tanto dagger in his left hand. The ninja, without turning fully, sweeps his ninjato backward in a blind reverse cut that deflects the tanto thrust. He then spins 180 degrees, seizes the guard's tanto wrist with his free left hand, twists it inward until the guard drops to his knees, and delivers a precise knee strike to the man's temple. The guard collapses face-first onto the tatami. Rain and wind from the broken screen blow paper fragments and the guard's dark indigo kimono across the frame. The camera tracks the spin with a fast handheld arc, lantern light strobing as bodies move past the wall sconces.
[Shot 5] At 00:12.500, the shot cuts to a high-angle wide shot from above — mounted on the wooden ceiling beam looking down into the room. All three samurai guards lie motionless on the tatami — one sprawled across the broken table amid ceramic shards, one face-down near the shattered shoji screen with rain blowing over him, one curled near the far wall where the scroll hangs undisturbed. The ninja, still crouched in his reverse-grip ninjato stance at the room's center, sheathes the blade across his back in one clean motion. He rises, pulls a torn piece of shoji paper from his shoulder, and moves toward the dark corridor at the far end of the room — vanishing into the deeper darkness between two extinguished lanterns. The camera holds on the aftermath from above — warm lantern glow illuminating the scattered bodies, rain mist drifting through the broken screen, and the empty corridor swallowing the ninja's silhouette.

overall_soundscape: The interior is dominated by the quiet patter of rain on the roof tiles above, muffled through wooden beams. Bare feet and tabi boots scuff and slide on woven tatami with sharp friction hisses. A ceramic cup shatters on impact. The shoji screen explodes with a sharp paper-tearing rip followed by wood-lattice splintering. Fists and elbows strike flesh and bone with dull wet thuds, and a metal blade clatters across the tatami when the guard is disarmed. Wind gusts through the broken screen, sending loose paper rustling across the room.

non_diegetic_music: A sparse, nerve-taut score opens with a single low cello drone and the faint metallic ring of a singing bowl, building barely-perceptible tension. As combat erupts, a tight, rhythmic ostinato on a koto — fast plucked strings in a minor pentatonic — drives the pace of each combination strike. A deep taiko strike punctuates each guard's takedown. After the final guard falls, all instruments cut to silence, leaving only rain on the roof and wind through the broken screen.
```

---

## Pattern 7: Ninja Silent Kill — Castle Assassination (T2VA, 15s, 5 shots)

A suffocatingly quiet infiltration and assassination sequence. Shows how to build tension through near-total absence of sound, patient camera movement, and three sequential kills that escalate in intimacy — from distant guard to the target himself.

### Key Patterns
- **Absence as a tool**: `non_diegetic_music: N/A` — no score, no ambiance, total silence is the tension
- **Escalating intimacy of kills**: Guard 1 (behind, jaw) → Guard 2 (base of skull, blood drop) → Daimyo (hand over eyes, neck)
- **Candle/lantern as lighting discipline**: Each light source is a small island; everything between is deep black
- **Camera patience**: Slow pushes, floor-level tilts, held frames — speed would betray the silence
- **The ink stroke**: Daimyo's brush freezes then drags a final stroke as he dies — visual poetry marking the kill

### Output

```
integrated_multimodal_description: [Shot 1] Ultra-realistic cinematic live-action with extreme shallow depth of field, chiaroscuro candlelight, and near-total darkness dominant. A slow tracking shot moves through a dark wooden corridor of a Japanese castle at night — narrow plank flooring, wooden lattice walls, widely spaced iron candle brackets each holding a single dim flame. The corridor is deep black except for small amber pools around each candle. THE NINJA — a lean figure wrapped entirely in matte charcoal-black cloth, only a narrow slit revealing sharp dark eyes, black tabi boots, a short ninjato and a tanto dagger concealed at his lower back — moves down the corridor in absolute silence, his weight on the balls of his feet, back flat against the wooden wall, passing between candlelight pools without disturbing the flames. Ahead, a heavy sliding door of dark lacquered wood glows faintly at its edges from interior light. The camera tracks forward at slow speed behind the ninja's shoulder, candlelight briefly catching the texture of his black cloth wrapping with each pass. His breathing is invisible, controlled.
[Shot 2] At 00:03.000, the shot cuts to a close-up of the ninja's black-gloved hand sliding the lacquered wooden door open one inch — just enough to see through. The camera pushes in with small amplitude at slow speed toward the gap. Through the narrow opening: THE DAIMYO, a heavyset Japanese man in his fifties with a close-cropped grey beard, wearing a deep crimson silk inner robe, sits cross-legged at a low writing desk lit by a single paper lantern, brush in hand, writing on a scroll. Two SAMURAI GUARDS in dark indigo kimono, wakizashi at their hips, stand flanking the door on the interior — one at Screen Left, one at Screen Right, both facing forward, backs partially to the door. The room is spacious with tatami flooring, a weapon rack displaying a katana on the far wall, and a single paper lantern casting warm but limited light, leaving the corners in deep shadow.
[Shot 3] At 00:06.000, the shot cuts to a low-angle extreme close-up from floor level as the ninja slides through the opened door in a single fluid motion, his body staying below the guards' sightlines, tabi boots whispering on the tatami without sound. The camera tilts up slowly, tracking the ninja's body as he flows across the tatami in a low crouch toward the Screen-Left guard's blind spot. The guard at Screen Left shifts his weight slightly — his straw sandal scraping the tatami — but does not turn. The ninja rises behind the guard, one black-gloved hand clamping over the guard's mouth and nose from behind, the other hand drawing the tanto from his own lower back. The tanto blade — short, plain wooden handle, thin steel — catches a sliver of lantern light. In one compressed motion the ninja drives the tanto upward beneath the guard's jaw, holds for one second as the man's body stiffens, then lowers him silently to the tatami. The Screen-Right guard has not turned. The camera holds on the controlled descent, the lantern flame steady.
[Shot 4] At 00:09.000, the shot cuts to a medium shot from the Screen-Right guard's perspective, looking across the room. The Screen-Left guard appears to still be standing — but as the camera holds, the ninja's black-wrapped form separates from the shadow behind the fallen guard's body, already moving. The Screen-Right guard turns his head at the faintest sound — a single thread of the dead guard's indigo kimono settling. The ninja is already there, materializing from the dark corner like poured ink. The guard opens his mouth to shout but the ninja's left hand seals it shut while his right — still holding the tanto, its blade now dark — drives it into the base of the guard's skull from behind. The guard's eyes widen, then glaze. His knees buckle. The ninja guides the body down in absolute silence, one hand on the man's collar, the other withdrawing the tanto with a controlled pull. A single drop of dark blood falls from the blade onto the tatami. The camera pushes in with small amplitude at slow speed toward that falling drop.
[Shot 5] At 00:11.500, the shot cuts to a wide shot from the doorway looking into the room. THE DAIMYO, in his deep crimson silk robe, is still writing, unaware. The two guards lie motionless on the tatami in deep shadow, their positions unseen from the writing desk. The paper lantern casts its warm pool around the daimyo, leaving the approach path in darkness. The ninja, a silhouette of black cloth against black shadow, crosses the remaining distance in three silent strides — each step timed to the daimyo's brush strokes. On the third brush lift, the ninja arrives directly behind the daimyo. The ninja's left hand clamps the daimyo's forehead, pulling the head back to expose the neck. The tanto, dark with the blood of the two guards, comes around from the right. The daimyo's brush freezes mid-stroke. His eyes — wide, reflecting the lantern flame — are the last things visible before the ninja's black-gloved hand obscures them. The camera holds on this tableau for two seconds — the lantern flame flickering once — then the ninja releases the body and the daimyo slumps forward onto his writing desk, his brush hand dragging a final ink stroke across the scroll. The ninja sheathes the tanto, steps back into the darkness beyond the lantern's reach, and is gone. The camera holds on the lit room — the daimyo slumped, the scroll, the single flickering lantern, and the two unseen bodies in the shadows. Silence.

overall_soundscape: Near-total silence dominates the entire sequence. The only sounds are the faint creak of wooden planks under careful weight, the soft whisper of cloth on cloth as the ninja moves, and the single scrape of a straw sandal when a guard shifts. The lacquered door slides open with a barely audible wooden groan. A brush scratches lightly on paper. On each kill, a single wet, subdued sound — barely more than a breath — is followed by the muffled rustle of a body being lowered to tatami. A single drop of liquid — blood — taps the tatami with a faint tick. No breathing from the ninja is ever audible.

non_diegetic_music: N/A
```

---

## Pattern 8: Ninja Combo Out on Wooden — Temple Storm Chase (T2VA, 15s, 5 shots)

An exterior chase-and-combat sequence across raised wooden walkways and bridges of a temple complex during a thunderstorm. Shows how to use architecture as both arena and weapon, with weather as a dynamic participant in every shot.

### Key Patterns
- **Architecture-as-arena**: Raised walkways, railing edges, pagoda balconies, narrow bridge over a drop — each location offers unique combat geography
- **Weather as a combatant**: Rain makes surfaces slick (guards lose footing), wind swings lanterns (dynamic lighting), thunder masks sound
- **Vertical geography**: Chase moves UP through the structure — walkway level → railing → balcony → bridge → aerial reveal
- **Escalating weapon stakes**: Ninjato vs katana → ninjato vs two katana → ninjato vs nodachi (longsword) — each fight bigger than the last
- **Lightning as a lighting tool**: Strobe-flashes freeze frames, coincide with bass drum hits in the score, and silhouette the architecture

### Output

```
integrated_multimodal_description: [Shot 1] Ultra-realistic cinematic live-action with dramatic backlighting, volumetric rain and mist, and a rich amber-and-teak color palette. A high wide shot establishes an ancient Japanese wooden temple complex during a heavy thunderstorm at night — a multi-tiered pagoda with dark cypress-bark roofing, interconnected raised wooden walkways spanning between buildings, rain streaming off curved eaves, paper lanterns swinging wildly in the wind along the walkway railings. Lightning flashes silhouettes the structure against black sky. THE NINJA — lean, wrapped entirely in charcoal-black cloth, only narrow eye slit visible, black tabi boots, ninjato strapped across his back — sprints along a raised wooden walkway, rain hammering his black wrappings. Behind and above, FOUR SAMURAI GUARDS in dark indigo armor and straw sandals give chase across the wet wooden planks, their sandals slapping the surface. The camera tracks the ninja from behind at fast speed, tilting down as he reaches a T-intersection where the walkway branches left toward a higher pagoda balcony and right toward a bridge over a courtyard gap. The ninja leaps onto the wooden railing and runs along its narrow edge toward the left branch. Rain sheets across the frame.
[Shot 2] At 00:03.000, the shot cuts to a dynamic low-angle tracking shot from walkway level, tilted up at the ninja as he runs along the railing edge — wet dark wood beneath his tabi boots, the pagoda's massive curved roof and swinging paper lanterns filling the upper frame. The lead samurai guard — a tall man in dark indigo armor with a katana drawn — vaults onto the railing behind him and closes the gap. The guard lunges with a thrusting katana strike aimed at the ninja's back. Without breaking stride, the ninja drops into a slide on the wet railing, the katana blade passing inches above his head, rain spraying from his boots. He grabs a wooden support post at the railing's end, swings his body around it in a tight arc, and launches a spinning back kick into the guard's chest as the man's momentum carries him forward. The guard flies off the railing and crashes through a paper screen on the pagoda balcony. The camera arcs right with large amplitude at fast speed, sweeping around the post with the ninja's swing, the storm-lit courtyard visible far below between the wooden structures.
[Shot 3] At 00:06.500, the shot cuts to a medium tracking shot on the pagoda balcony as the ninja lands on the wet wooden planks. Two more samurai guards — one from each end of the balcony — converge on him simultaneously, katana drawn. The ninja draws his ninjato in a reverse grip and meets the first guard's downward katana cut with an upward diagonal block, steel sparking steel with a sharp ring that echoes through the storm. He redirects the force leftward, opening the guard's center, and drives a front kick into the man's armored stomach, sending him stumbling into the paper screen debris where the first guard already lies. The second guard swings a horizontal cut. The ninja drops to one knee on the wet wood, the katana whistling over his head, and slashes the ninjato across the back of the guard's lead knee — the man's straw sandal loses grip on the wet plank and he collapses. The ninja rises and delivers a sharp downward pommel strike to the fallen guard's temple. Rain pours across the balcony, washing blood from the ninjato blade. The camera pushes in with small amplitude at fast speed, handheld with slight shake, tracking each exchange.
[Shot 4] At 00:10.000, the shot cuts to a wide shot from across the courtyard gap as the ninja — black cloth soaked and clinging to his frame — sprints across a narrow wooden bridge connecting the pagoda balcony to the adjacent building. The bridge is barely two planks wide with a low rope railing, slick with rain, suspended over a five-story drop. The fourth and final samurai guard — a massive broad-shouldered man in full dark indigo armor, a nodachi longsword gripped two-handed — crashes onto the bridge behind him. The bridge sways and groans under their combined weight. The guard swings the nodachi in a massive overhead arc. The ninja parries with the ninjato held two-handed — the impact nearly buckles his knees, wood splintering from the bridge railing where the nodachi glances off. The ninja kicks the guard's lead foot, destabilizing him on the slick surface, then spins and slashes the ninjato across the guard's sword arm. The nodachi drops from the guard's grip and tumbles off the bridge into the dark courtyard below. The camera is positioned at bridge-end, capturing the fight in profile against the storm-lit sky, lightning flashing behind the figures, rain streaking horizontally across the frame. The bridge creaks and sways with each movement.
[Shot 5] At 00:12.500, the shot cuts to a low-angle close-up on the swaying bridge as the ninja — rain streaming down his black cloth wrapping — delivers a final spinning heel kick to the disarmed guard's chest. The guard staggers backward, his armored boots losing traction on the slick wet plank, and topples backward off the bridge railing — his dark indigo armor disappearing into the rain-filled darkness below, the sound of his fall swallowed by the storm. The ninja stands alone on the center of the swaying bridge, ninjato in reverse grip at his side, breathing hard — the only visible motion on the bridge. He sheathes the blade across his back. The camera pulls back and pedestals up with large amplitude at slow speed, rising above the bridge to reveal the full temple complex — tiered pagoda roofs, raised wooden walkways, swaying paper lanterns — all battered by the thunderstorm, the ninja a single black figure on a narrow bridge in the center of the vast wooden architecture. Lightning flashes one final time, freezing the entire complex in white light. The camera holds.

overall_soundscape: Heavy thunderstorm rain hammers wooden rooftops, planks, railings, and paper screens throughout, producing a constant percussive roar. Bare tabi boots and straw sandals slap, skid, and slide on wet wood with sharp friction sounds. Steel clashes against steel with bright ringing impacts that reverberate through the open structures. Wood splinters and cracks when blades and bodies strike railings and screens. The bridge groans and creaks under shifting weight. A heavy armored body falls from the bridge with a long, fading sound swallowed by wind and rain. Distant thunder rumbles beneath the action.

non_diegetic_music: A driving, percussive taiko-and-shamisen score matches the chase rhythm — rapid taiko beats at a fast tempo driving each footstep and strike, layered with sharp shamisen string strikes that punctuate each blade clash and impact. Lightning flashes coincide with deep bass-drum hits. As the final guard falls and the camera rises above the temple complex, the rhythm decelerates into a single sustained low taiko rumble that fades beneath the storm, leaving only rain and thunder.
```

---

## Pattern 9: Contemporary Dance × Supernatural Blade Combat — Seedance to H3 Conversion (T2VA, 15s, 5 shots, 9:16)

A high-end 3D cinematic sequence fusing contemporary dance choreography with supernatural katana combat against shadow wolves. Converted from a Seedance 2.5 prompt. Shows how to adapt a dense single-paragraph action prompt into H3's multi-shot structure while building a progressive escalation system for supernatural enemies.

### Key Patterns
- **Dance-as-combat choreography**: Every combat strike originates from a named dance technique — backbend, spin, floor slide, aerial contraction, whipping turn, plié
- **Supernatural enemy system**: Shadow wolves (black smoke-matter, glowing red eyes) disintegrate into ink-dark particles + crimson fragment trails — consistent visual language for every kill
- **Blade energy signature**: Every katana arc tears crimson fabric-like energy through the air — a persistent visual marker tying all strikes together
- **Progressive escalation**: 1 wolf → 3 wolves → 1 aerial → 5 wolves → 1 giant final wolf — the pack grows then peaks on a boss
- **Environmental consistency**: Wet reflective floor + cold volumetric haze in every shot — anchors the dark void setting
- **Seedance-to-H3 conversion**: Single paragraph → 5 timed shots; general camera language → specific type+amplitude+speed per shot; @[character ref] → full inline description (T2VA)

### Output

```
integrated_multimodal_description: [Shot 1] High-end cinematic 3D realism with feature-film rendering quality, controlled motion blur, realistic cloth dynamics, and a dark atmospheric palette dominated by deep blacks, cold steel highlights, crimson energy, and ink-dark particle effects. Volumetric haze fills the space. A medium tracking shot follows THE DANCER — a beautiful woman in her mid-twenties with a lean athletic dancer's build, long flowing dark hair that whips with every movement, fair porcelain skin, sharp striking features, wearing a form-fitting dark charcoal sleeveless performance top, bare-armed, loose dark leggings that ripple with fabric dynamics, and bare feet — as she stands center-frame in a dark void-like space with a wet reflective floor surface and cold volumetric fog drifting at ankle height. A SHADOW WOLF — a massive predatory silhouette formed entirely of dense black smoke-matter with no solid texture, only glowing dim red eyes and a lunging open maw of collapsing dark particles — leaps through the fog directly at her from Screen Right. The dancer folds beneath the attack in a violent backbend, her spine curving deeply, long dark hair sweeping the wet floor, as she draws a single katana — a sleek Japanese longsword with a dark handle and gleaming steel blade — from her hip in a fluid underhand arc that carves upward through the wolf's body. Crimson fabric-like energy tears through the air along the blade's path. The shadow wolf splits apart, its smoky form disintegrating into ink-dark particles and crimson fragments that scatter and dissolve in the haze. The camera orbits around her recovery arc with large amplitude at fast speed, capturing her body unfolding from the backbend into a combat-ready crouch.
[Shot 2] At 00:03.000, the shot cuts to a low-angle tracking shot from floor level tilted up at the dancer as she explodes from the crouch into a rapid spinning sequence — a contemporary dance spin that accelerates into a whirlwind of movement, her dark hair fanning outward, loose leggings rippling with realistic cloth dynamics, bare feet pivoting on the wet reflective surface. THREE MORE SHADOW WOLVES — each a dense predatory silhouette of black smoke-matter with glowing red eyes — circle her from different angles, lunging in from Screen Left, Screen Right, and Front-Center. The dancer's spin becomes a sweeping katana arc that catches the first wolf mid-lunge across its flank — the blade slices through smoke-matter, sending a ribbon of crimson fabric-like energy trailing behind it as the wolf splits and collapses into ink-dark particles. She drops instantly from the spin into a deep floor slide on the wet surface, her body extending horizontal beneath the second wolf's leaping attack, katana held outward so the blade catches the wolf's underbelly as it passes over her — the wolf shears apart above her into cascading black vapor. The camera tracks the floor slide at fast speed, low angle capturing the wet floor reflections and trailing particle effects. The dancer rolls out of the slide back onto her feet in one fluid recovery.
[Shot 3] At 00:06.000, the shot cuts to a dynamic medium shot with aggressive lateral camera acceleration as the dancer launches into an aerial contraction — a powerful upward leap from the wet floor, both knees tucking to her chest, her body coiling tight at the apex. The third shadow wolf lunges upward toward her. At the peak of her jump, the dancer whips her katana in a tight horizontal arc — a single clean strike that shears through the wolf's smoky form at neck height. The wolf's head separates from its body, both halves erupting into cascading ink-dark particles and crimson fragment trails that hang suspended in the volumetric haze around her airborne silhouette. As she descends, her body extends into a dramatic deep backbend mid-air — one arm reaching back, the katana trailing a ribbon of crimson energy behind it, long dark hair fanning upward. She lands on the wet floor in a controlled one-footed landing, the impact sending a circular ripple across the reflective surface. The camera pushes in with extreme foreshortening toward her landing point, distorting the sense of depth and speed. Cold fog swirls around her bare feet.
[Shot 4] At 00:09.000, the shot cuts to a spiraling orbit shot that circles the dancer at fast speed with large amplitude as a FINAL WAVE of FIVE SHADOW WOLVES — the largest pack yet, each wolf bigger and denser with brighter red eyes — converges on her simultaneously from all directions. The dancer accelerates through a continuous sequence of whipping turns, each rotation flowing directly into a katana strike: the first turn is a rising diagonal slash that bisects a wolf lunging from Screen Left — the second is a descending overhead cut that splits a wolf from above — the third is a horizontal spinning slash that cleaves through two wolves simultaneously as they leap from opposite sides. Each blade arc tears a long ribbon of crimson fabric-like energy through the air, and each disintegrating wolf bursts into a cloud of ink-dark particles and collapsing black shockwaves that radiate outward from the point of impact. The dancer's movements are graceful and feral simultaneously — contemporary dance technique expressed as lethal combat choreography. Her dark performance top and bare arms glisten with a fine layer of moisture from the cold haze. The camera spirals through the pack with her, sometimes passing between disintegrating wolves, particles streaming past the lens in motion blur.
[Shot 5] At 00:12.000, the shot cuts to a low-angle wide shot as the last remaining shadow wolf — the largest, twice the size of the others, its smoke-matter form denser and more defined — rears back and lunges directly at the dancer from Screen Right. She meets the charge head-on, launching off the wet floor into a powerful rotating slash — her entire body spinning horizontal at chest height, katana extended, long dark hair whipping in a full arc, loose leggings flaring outward. The blade meets the wolf's center mass. For a frozen instant — a brief slow-motion frame — the katana is embedded in the wolf's smoky form, crimson energy crackling along the blade, the wolf's red eyes flickering. Then the dancer completes the rotation, pulling the blade through in a clean horizontal draw. The wolf splits apart in an eruption of ink-dark particles, crimson fragment trails, and a collapsing black shockwave that ripples outward across the wet floor, disturbing the fog layer in an expanding ring. The dancer lands softly on the wet surface in a deep dancer's plié, katana held extended to the side, its blade still trailing wisps of crimson energy that slowly dissipate. Long dark hair settles around her shoulders. She straightens slowly, her breathing visible in the cold air. The camera holds on her silhouette — a single figure in a vast dark space, the wet floor reflecting her form and the fading crimson glow, surrounded by drifting ink-dark particles that are slowly dissolving into nothing. Volumetric haze rolls back in. The camera holds.

overall_soundscape: Bare feet strike and slide on a wet reflective surface with sharp percussive slaps and fluid friction sounds. The katana cuts through dense air and smoke-matter with a resonant steel ring that sustains and decays after each arc. Shadow wolves emit a low, barely-audible harmonic growl that modulates as they disintegrate into a rush of displaced air. Deep, full-body impacts produce muffled concussive thuds as collapsing black shockwaves radiate outward. The dancer's controlled breathing — rhythmic, deliberate — is audible between strikes.

non_diegetic_music: A hybrid electronic-orchestral score drives the entire sequence — a pulsing analog synth bassline at a fast tempo layered with rapid, irregular percussion that matches the dancer's rhythmic footwork. Sustained string swells rise during each aerial movement and peak on each blade impact, resolving into sharp decays. A distorted, processed vocal hum enters during the final wave, building in intensity and cutting abruptly to a single sustained low cello note as the last wolf disintegrates, decaying into silence over the final held frame.
```


---

## FILE: references/ref2va-format.md (https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill/blob/HEAD/references/ref2va-format.md)

# H3 Ref2VA Format Reference

Complete specification for MiniMax H3-Base-Ref2VA omni-modal video prompt format. Load this when generating Ref2VA prompts.

---

## Role

You are producing the structured output that H3-Base-Ref2VA consumes directly. You replicate the "full-reference mode" behavior of MiniMax's H3-Context-IR preprocessor: take a free-form brief + inventory of reference assets, output ONE structured prompt.

## Input You Receive

- **BRIEF**: free-form intent (story, action, scene, dialogue, style). Any language.
- **ASSETS**: numbered reference files in upload order — Images, Videos, Audios — with optional descriptions and durations. Numbering per modality defines labels: Image k → `<Picture k>`, Video k → `<Video k>`, Audio k → `<Audio k>`.
- **TARGET**: duration_s (integer 4–15), aspect ratio, optional shot plan.

## Output Contract (Absolute)

1. Output ONLY the six sections below, in this exact order, with these exact lowercase field names followed by a colon. No preamble, no explanation, no markdown fences, no commentary.
2. Write every section in English. EXCEPTIONS: preserve original language verbatim for dialogue/lyrics inside `<d>` and for text visibly present in the scene.
3. Never invent reference labels beyond those defined in subject_definitions. A label keeps one fixed meaning across all sections.

---

## SECTION 1 — subject_definitions

One line per referenced item that must be tracked. Four label types:

### `<Subject N>` — reusable VISIBLE content
People, animals, objects, scenes/environments, clothing, props, styles, actions, expressions, poses. State what it is, which asset(s) it comes from, and the concrete features to preserve (face, hairstyle, garments, accessories, palette). One subject may combine assets:
> `<Subject 1> is the woman whose appearance comes from <Picture 1> and whose walking motion comes from <Video 1>.`

### `<Picture N>` — standalone frame/composition anchor
Use ONLY when the image itself is a concrete frame/composition anchor (first frame, keyframe, last frame, storyboard). If an image merely defines a character/scene/style, cite it inside the `<Subject N>` line instead — no separate picture entry.

### `<Video N>` — whole-video relationships
Edit source, continuation source, or provider of camera movement/cuts/rhythm/temporal structure. Visible content reused from a video still belongs under `<Subject N>`.

### `<Audio N>` — audio asset roles
Copy of a signal, background-music style, voice-timbre reference, dialogue/lyric/sound-effect source, beat/rhythm/continuity. When bound to a target speaker, reuse the target's global speaker ID:
> `<Audio 1> is the voice-timbre reference for <Subject 1> (S1).`

`<Video N>` and `<Audio N>` are numbered independently; different indices may still come from the same source file. A reference video does NOT create an `<Audio N>` just because it contains sound.

---

## SECTION 2 — summary

One short paragraph. MUST begin with a square-bracketed task-type prefix, combined with `+` when several apply (never repeat a type):

| Prefix | Meaning |
|---|---|
| `[keyframe completion]` | an image is a concrete frame anchor (first/keyframe/last/edited keyframe) |
| `[reference generation]` | assets guide generation (character, scene, style, action, camera, storyboard) without being a frame anchor or edited/continued source video |
| `[video editing]` | an existing source video is directly modified |
| `[video continuation]` | new content continues/extends an existing source video |
| `[audio reuse]` | the same audio signal is reused in full or in part |
| `[audio reference]` | only music style, timbre, dialogue/lyric content, SFX texture, beat, or continuity is referenced (signal not copied) |

A video that only provides camera movement or rhythm = `reference generation`, NOT `video editing`/`video continuation`. For video-editing tasks, begin after the prefix with: `"The target video is an edited version of <Video 1>."` Use only previously defined labels.

---

## SECTION 3 — retention_analysis

One line per label from subject_definitions, using ONLY these fixed markers:

**Visual** (`<Subject N>`, `<Picture N>`, `<Video N>`):
- `fully_preserved` | `partially_preserved` | `attribute_transfer` | `weak_reference`

**Audio** (`<Audio N>`):
- `fully_copy` | `partially_copy` | `reference` | `weak_reference`

Formats:
```
<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - <what exactly is retained>.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - <features retained>.
<Video 1> (cut and pacing structure): weak_reference - <what aspect is referenced>.
<Audio 1>: reference - the target speaker follows <Audio 1>'s voice timbre and measured delivery without copying the original signal.
```

Newly added actions, backgrounds, or plot are NOT losses of reference fidelity. Never write `(Sx)` speaker IDs in this section.

---

## SECTION 4 — detailed_description (main body)

### Length
350–500 English words for generation tasks. Dialogue-dense content prioritizes fitting the complete spoken timeline over word count.

### Opening
Open with the overall style in 1–2 English sentences BEFORE `[Shot 1]`:
> "The target video uses a cinematic live-action style with soft lighting and a desaturated palette."

Do NOT put the style statement inside `[Shot 1]`.

### Shots
- `[Shot 1]` has NO timestamp.
- Later shots: `[Shot N] At MM:SS.mmm, ...` with strictly increasing cut times inside the target duration.
- Cut verbs: "the camera cuts to", "the shot cuts/transitions/changes/switches to".
- Use cross-dissolve/fade/wipe ONLY if the user explicitly asked.
- A cut must add NEW information (subject, space, state, viewpoint, time). If only distance or angle changes, use camera motion instead.

### Camera Motion
Motion type + amplitude + speed, written as natural English action (not stacked labels).
- Types: Zoom In/Zoom Out, Push In/Pull Out, Pan Left/Pan Right, Truck Left/Truck Right, Tilt Up/Tilt Down, Pedestal Up/Pedestal Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Shake Strongly, POV, Roll Clockwise/Roll Counterclockwise.
- Amplitude: "with small amplitude" / "with large amplitude" (omit when medium).
- Speed: "at slow speed" / "at fast speed" (omit when normal).

### Reference Labels
Insert at each label's first appearance and wherever its role applies; reuse without redefining. Natural frame-anchor phrasing:
> "the shot begins from `<Picture 1>`", "the shot ends on `<Picture 3>`"

### Speakers
Stable IDs (S1), (S2)... assigned in order of actual vocal events and reused at every later vocal event; group speech (S1,S2). Characters who never vocalize get NO ID. At first appearance give identity anchors (type, age, gender, on/off-screen, pitch, timbre, rate, accent). When a referenced subject speaks, keep both labels:
> `<Subject 2> (S1) turns and says, <d>[English] ...</d>`

Off-screen speech keeps the form and is marked "off-screen".

### Dialogue/Lyrics Format
- Identifying phrase + ID + delivery OUTSIDE `<d>`
- Inside `<d>` ONLY the language tag and the exact spoken words: `<d>[English] Wait for us!</d>`
- Preserve the user's words and punctuation verbatim — never translate or rewrite.
- For reused audio, write `[unclear]` for unintelligible spans; strip emoji/tilde/decorative punctuation; end statements with . ? or ! before `</d>`.

### Voiceover
Use the exact phrase "says in an off-screen voiceover" and immediately after the `<d>` block state the on-screen lips stay closed:
> "...while his lips remain completely closed."

### Dialogue Crossing a Cut
`<scenetrans>` at the connecting points in both parts plus an explicit continuity statement ("continues seamlessly across the cut", "carries over from the previous shot"). Speech truncated by the video end: `<cutoff>`.

### Audio Edge Cases
- When verbal content exists only inside a directly reused BGM/soundtrack and no concrete person produces it, cite `<Audio N>` as the source and do NOT invent an `(Sx)`.
- Timbre-only reference: do NOT carry the reference audio's original words into the target.

### On-screen Text
Any visible banner/sign/label/subtitle/neon text goes in English double quotation marks, verbatim, no translation:
> A red neon sign reading "营业中" glows above the doorway.

### Per-Shot Content
Describe only what is visible or audible. Per shot establish: composition, subject appearance and position, environment and lighting, actions and state changes, camera movement, current sound, and where referenced content appears or takes effect. Never reduce to a plot summary or a list of reference relationships. Keep ONE dominant action per shot.

---

## SECTION 5 — overall_soundscape

1–4 English sentences, one paragraph: ambience, physical action sounds, non-verbal human sounds across the FULL video. Do not repeat dialogue, singing, or shot-synced sound events here. Use N/A only if the user explicitly requests complete silence. If an audio asset provides this layer, state the relationship:
> "The copied ambience layer from `<Audio 1>` continues throughout the target video."

---

## SECTION 6 — non_diegetic_music

1–3 English sentences about score the CHARACTERS CANNOT hear: instrumentation, tempo, rhythm, dynamic changes only — no abstract mood words, no emotional-function explanations. Music audible to characters (radio, singing, phone) is diegetic → belongs in `detailed_description`. N/A when no score. If an audio asset is reused as score:
> "`<Audio 2>` is directly reused as the complete audience-only score."

Never repeat dialogue/lyrics in sections 5–6.

---

## Fidelity Rules

- Stay on the user's intent; supplement missing or underspecified semantic details (scene, props, sound) only when consistent with the brief.
- When the brief conflicts with hard constraints (duration 4–15s, asset counts), silently conform to the constraint closest to the intent — do not explain.
- Avoid named third-party IP, real celebrity identities, and trademarked character names; describe them generically instead.
- When BRIEF omits dialogue language, default to `<d>[English]`.

---

## Worked Example

**Input:**
```
BRIEF: "A blonde woman in a coffee shop is startled when a man's Samoyed lunges at her cookie; they trade two funny lines; canned laugh at the end."
ASSETS:
  Image 1 = coffee-shop interior photo
  Images 2–4 = white Samoyed photos
  Video 1 = blonde woman clip
  Video 2 = young man clip
  Audio 1 = woman's voice sample
TARGET: duration_s=7, 16:9
```

**Output:**

```
subject_definitions:
<Subject 1> is the coffee-shop environment in <Picture 1>, featuring an exposed brick wall, an orange tufted sofa with patterned pillows, a neon sign, and a wooden coffee table.
<Subject 2> is the fluffy white Samoyed in <Picture 2>, <Picture 3>, and <Picture 4>, with thick white fur, pointed ears, a dark nose, and a curved tail.
<Subject 3> is the young blonde woman in <Video 1>, with long blonde hair and a light-pink button-down shirt with rolled-up sleeves.
<Subject 4> is the young man in <Video 2>, with short wavy brown hair and a dark-grey hoodie with drawstrings.
<Audio 1> is the voice-timbre reference for <Subject 3> (S1), containing a spoken English vocal layer.

summary:
[reference generation + audio reference] The target video shows <Subject 3> eating a cookie in <Subject 1>. <Subject 4> enters with <Subject 2>, which lunges toward the cookie. The three-shot exchange uses <Audio 1> as the voice-timbre reference for <Subject 3> and ends with a canned audience laugh.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the Samoyed's thick white fur, pointed ears, dark nose, and curved tail are retained.
<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the blonde woman's identity, long hair, and light-pink shirt are retained.
<Subject 4> (appears in [Shot 1], [Shot 2]): fully_preserved - the young man's short wavy brown hair and dark-grey hoodie are retained.
<Audio 1>: reference - its vocal timbre guides the dialogue delivery of <Subject 3> without copying the original signal.

detailed_description:
The target video uses a realistic multi-camera sitcom style with warm indoor lighting.
[Shot 1] A medium shot establishes <Subject 1>, the coffee shop with its exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table. <Subject 3> (S1), the young woman with long blonde hair and a light-pink button-down shirt with rolled-up sleeves, sits on the sofa holding a chocolate-chip cookie. From the left, <Subject 4>, the young man with short wavy brown hair and a dark-grey hoodie with drawstrings, enters holding the leash of <Subject 2>, the thick-furred white Samoyed with pointed ears, a dark nose, and a curved tail. The dog lunges toward the cookie and pulls the leash taut. <Subject 3> (S1) jerks her hand back and, using the clear youthful voice timbre referenced from <Audio 1>, exclaims with light annoyance, <d>[English] Hey! Watch your dog!</d> She closes her lips and guards the cookie while <Subject 4> pulls the dog back.
[Shot 2] At 00:03.000, the shot cuts to a close-up of <Subject 4> (S2), the young man in the dark-grey hoodie from Shot 1, sitting beside <Subject 3> on the sofa and holding <Subject 2> securely in his arms. <Subject 4> (S2) says in a casual young male voice with a playful tone and an easy conversational pace, <d>[English] He just likes cookies more than me.</d> He closes his mouth into an apologetic smile and strokes the dog's thick white fur.
[Shot 3] At 00:05.000, the shot cuts to a close-up of <Subject 3> (S1), the blonde woman in the light-pink shirt from Shot 1. Her annoyance softens as she looks toward the Samoyed. <Subject 3> (S1) replies in the same clear youthful voice referenced from <Audio 1> with an amused cadence, <d>[English] Well, he has good taste at least.</d> She smiles and raises the cookie in a small toast-like gesture. A classic canned audience laugh begins immediately after the line and continues through the final frame.

overall_soundscape:
Soft indoor coffee-shop room tone continues throughout the scene.

non_diegetic_music:
N/A
```

---

## Suggested User-Message Template (for ComfyUI LLM nodes)

```
BRIEF:
<your free-form story/scene intent here, any language>

ASSETS:
Images:
1. <description of image 1, e.g. "character sheet: Mei, front/profile/full-body">
Videos:
1. <description, duration s>
Audios:
1. <description, duration s, e.g. "Mei's voice sample, 8 s">

TARGET:
duration_s: 8
ratio: 9:16
shot_plan: <optional, e.g. "hook → turn → payoff">
```

**Wiring tips for ComfyUI:** LLM node output string → feed directly into the H3 Ref2VA node's prompt input. Keep the ASSETS list order identical to the physical wiring order of the reference inputs — the labels map positionally (Image 1 → first image socket, etc.). If your LLM node truncates, remove the EXAMPLE section from the system prompt first (it's the largest block); the rules alone are sufficient.


---

# ISSUES / PRs of benjiyaya/Minimax-H3-Prompt-AgentSkill


### #1 [closed] ISSUE: Not an Issue :)
URL: https://github.com/benjiyaya/Minimax-H3-Prompt-AgentSkill/issues/1  by drel-ai 2026-08-06T17:25:48Z

can you please make one single system-prompt to use with nodes like qwen-vl or other nodes that can not handle agents, references, or multiple text inputs. thanks in advanced :)  
