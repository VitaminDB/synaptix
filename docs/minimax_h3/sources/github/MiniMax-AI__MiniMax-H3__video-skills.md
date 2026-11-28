# MiniMax-AI__MiniMax-H3__video-skills

Source: https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills
Fetched: 2026-11-28



---

## FILE: skills/3d-animation-short-generator/SKILL.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/3d-animation-short-generator/SKILL.md)

---
name: 3d-animation-short-generator
description: |
  Create complete stylized 3D animated shorts from a story idea through an ordered production workflow covering project brief, story outline, character and environment cards, standardized shot planning, text or optional pencil storyboards, video-model selection, single-shot generation, assembly, BGM matching, and final review. Use when the user wants an end-to-end narrative animation workflow with strong character consistency, scene continuity, timing, camera, performance, and audio control. Not for single images, simple edits, photorealistic live action, or one standalone clip.
compatibility: Requires the MiniMax Hub agent (canvas workspace, choice cards, and hub_generate_image/hub_generate_video tools); not portable to generic agent harnesses.
---

# 3D Animation Short Generator

Use this Skill when the user wants a complete story-first animated short workflow, from one-line idea to final edited video. The workflow must place every major artifact on the canvas in production order and pause at creative gates with user choice cards before expensive or high-impact steps.

Core rule: **story first, ask screen size and total duration with choice cards immediately after user intake, ordered canvas artifacts, every required confirmation via choice card, fixed order after character/scene cards: six-column standardized shot table with per-second directives + audio cues + spatial anchor chain → shot-table self-check gate → one single text storyboards document with one section per shot (multi-panel pencil image only when user explicitly opts into visualization mode; any shot flagged for heavy iteration is extracted to a standalone text storyboard node) → video-model choice card (H3 default, Seedance 2.0 fallback) + resolution choice card → single-shot clips rendered by the chosen video model → full-film assembly with BGM; final video must remove all storyboard artifacts**.

## Global Visual Style Lock

Unless the user explicitly requests another visual style, all character cards, scene cards, shot tables, text storyboards, optional pencil storyboards, single-shot video clips (regardless of which video model is selected), assembled videos, and final composites must use this visual style:

- Rendering style: Pixar-inspired 3D cartoon rendering, C4D + Octane renderer look, high-end animated feature quality.
- Character design: exaggerated geometric simplification balanced with excellent material detail. Avoid 100% realistic human anatomy; use high-level shape language, large readable silhouettes, and Q-version proportions when appropriate.
- Proportion language: friendly stylized proportions, often 2.5–3 head-tall for cute or childlike characters, with big heads, compact bodies, clear silhouettes, and high recognizability.
- Hair / fur: combine strong sculpted clumps and clean block shapes with fine edge flyaway hairs or fuzzy rim details, so hair/fur feels designed but tactile under light.
- Skin / material: warm subsurface scattering skin quality, soft translucent reddish light through ears, cheeks, nose, and fingertips; avoid hard plastic skin.
- Acting style: exaggerated, lively Disney/Pixar-style character animation performance with squash and stretch, strong brows, eye corners, pupils, lips, and cheek shape changes.
- Motion style: high-energy poses, clear line of action, forward lean, strong anticipation, fast but readable timing, elastic body mechanics, and vivid micro-expressions.
- Emotional range: balance cuteness and explosive expressiveness; intense emotions may use dramatic facial deformation while preserving character appeal.

Negative style constraints: no photorealistic live-action, no flat 2D anime, no plastic toy skin, no stiff mannequin posing, no realistic anatomical stiffness, no lifeless facial expressions.

## STEP 0: Intake and Canvas Plan

Capture:

- One-line idea or rough premise
- Desired output: blueprint only, assets only, standardized shot table with per-second directives, single text storyboards document (default) + extracted single-shot text storyboard nodes for heavy-iteration shots + multi-panel pencil storyboards (opt-in), single-shot video clips (with video model chosen in Step 7), assembled main video, or final BGM-composited film
- Approximate length, if the user already stated it
- Screen size / aspect ratio, if the user already stated it
- Visual tone: default warm stylized 3D animation
- Dialogue requirement: whether the film has dialogue, voiceover, or no speech
- Dialogue language only if the user explicitly states it; do not default to English dialogue

Immediately after capturing the user's input, before Project Brief or any other next step, show choice cards to confirm production format:

Screen size / aspect ratio card:

- 16:9 landscape (recommended for cinematic short)
- 9:16 vertical short
- 1:1 square
- 4:5 social portrait
- Custom size / aspect ratio

Total duration card:

- 30–60 seconds (recommended)
- 15–30 seconds
- 60–90 seconds
- 90–180 seconds
- Custom duration

Only proceed after the user chooses both screen size/aspect ratio and total duration, or explicitly supplies custom values. Store the approved screen size/aspect ratio and duration in the Project Brief and reuse them in shot timing, transition continuity, the standardized shot table with per-second directives, single-shot storyboards (text by default, pencil image when user opts in), single-shot video clips, assembly, BGM matching, and final composite settings.

Create or later update canvas artifacts in this order:

1. Project Brief text node
2. Story Outline text node
3. Labeled Character Card image nodes
4. Environment-only Scene Card image nodes
5. Standardized Shot Information Table node with six columns; each row must include per-second directives inside `Shot Description`
6. **Single text storyboards document** — one canvas text node named `<title> text storyboards` containing one section per shot (mirrors the half-narrated-drama storyboard structure). When the user flags a shot for heavy iteration, that section is extracted to a standalone text node and a `(extracted)` marker is left in the document. Pencil image storyboards, if opted in, are separate image nodes.
7. Single-Shot Video Clip nodes (rendered by the video model selected in Step 7 — H3 default, Seedance 2.0 fallback)
8. Assembled Main Video node
9. Matched BGM audio node and Final BGM-Composited Video node

Do not dump long production content only in chat. Put durable outputs on canvas as text, image, video, or audio nodes.

## STEP 1: Project Brief

Produce a concise project brief and write it to a canvas text node named with the project title or `项目简报`.

Include:

- Working title
- One-line What-if
- Emotional premise
- Target audience feeling
- Main deliverables planned
- Approved screen size / aspect ratio
- Approved total duration
- Dialogue mode and language: only use a specific language when the user explicitly requested it; otherwise write `language not specified` and keep dialogue minimal or ask later when needed
- Initial risks
- Dialogue intent when present

Then show a user choice card:

- Continue with this direction (recommended)
- Regenerate premise options
- Revise emotional premise
- Refine dialogue direction

Only proceed after the user chooses or explicitly says to continue.

## STEP 2: Story Outline and Gates

Create a story outline and write it to a canvas text node named `故事大纲` or `story-outline`.

Include:

- Protagonist Want / Need / flaw
- Core world rule
- 8-beat causal story spine
- Emotional anchor and payoff
- Dialogue beats if the user requested dialogue
- Red-line checks

Gate checks:

- Protagonist is active
- Crisis is intensified by protagonist flaw
- Coincidence never solves the problem
- Ending reuses an earlier emotional anchor
- Antagonistic pressure is not a flat villain
- Dialogue reveals relationship change instead of explaining the theme

Then show a user choice card:

- Approve story and continue (recommended)
- Revise beats
- Revise emotion curve
- Revise dialogue beats
- Return to premise

## STEP 3: Character Cards

Generate character reference cards and place each image on canvas. Recommended order:

1. Protagonist card
2. Contrast / pressure character card
3. Optional supporting character card

Each character card should be a 16:9 production reference sheet when possible. Unlike final rendered video, character cards should include clear readable labels so downstream generation can bind the correct person and props:

- Character name label in English and/or the project language
- Role label, such as protagonist, grandma, thief, sidekick, pressure character
- Main 3/4 view
- Front / side / back views
- Expressions
- Material / costume / prop details
- Important prop labels, such as handbag, wallet, skateboard, apple basket, scarf, shoes, glasses
- Identity lock repeated in the prompt
- A short visual-ID note listing age range, body type, hairstyle, outfit colors, signature props, and do-not-change traits

For stylized 3D animation, keep the character soft, readable, and consistent across later images and videos.

After the main character cards are generated, show a user choice card:

- Lock character designs and continue (recommended)
- Regenerate protagonist card
- Adjust specific visual details
- Add another character card

Warn the user that changing locked character designs later may require regenerating the shot table, single-shot storyboards, single-shot video clips, assembled main video, and final composite.

## STEP 4: Scene Cards

Generate scene reference cards and place them on canvas after character cards. Scene cards must show environments only: do not include characters, people, crowd figures, silhouettes, hands, faces, or character cameos. Character action belongs in the shot table, single-shot multi-panel pencil storyboards, and single-shot video clips, not scene cards.

Include:

- Main environment overview
- Key light states, such as day / night
- Emotional sub-spaces
- Continuity landmarks (fixed objects whose screen position must persist across shots in the same scene, e.g. kitchen island, sofa, door frame, tree, mailbox)
- Important props in the environment

Then show a user choice card:

- Lock scene design and continue (recommended)
- Regenerate scene card
- Add another scene angle
- Adjust lighting or layout

## STEP 5: Standardized Shot Table Video Prompts (Six Columns)

After character cards and scene cards are locked, output standardized video prompts as a shot information table. This step is mandatory and cannot be swapped with storyboard or video generation.

Required reference: read and follow `references/shot-table-spec.md` for the exact six-column schema, per-second directive requirements, table-wide rules, user approval card, and mandatory Step 5.5 self-check gate.

Minimum runtime contract:

- Create a canvas table node named `标准镜头信息表` or `standard-shot-table`.
- Use exactly six columns: `Shot ID & Duration`, `Continuity Handoff`, `Reference Anchors (Spatial + Identity)`, `Hook Type`, `Shot Description (Per-Second Directives)`, `Audio & Dialogue Track`.
- Every row must include complete per-second directives, continuity handoff, reference anchors, hook type, and audio/dialogue timing.
- Run the Step 5.5 self-check from `references/shot-table-spec.md` before storyboarding. Do not enter Step 6 until the self-check passes.

Then show the table approval/self-check choice cards defined in `references/shot-table-spec.md`.

## STEP 6: Text Storyboards Document (Default) + Pencil Image Storyboards (Opt-in)

After the Step 5.5 self-check passes, show a storyboard-mode choice card before producing any storyboard artifact.

Required reference: read and follow `references/storyboard-guidelines.md` for the default single text storyboards document, optional multi-panel pencil storyboards, shot-level extraction/re-integration, storyboard approval cards, and visualization fallback rules.

Minimum runtime contract:

- Default mode is one authoritative text storyboards document containing one section per shot.
- Pencil storyboard images are opt-in visualization artifacts only; they never override the text storyboard.
- Extract a shot into a standalone text node only when the user flags that shot for heavy iteration.
- Step 7 must read the matching text storyboard section or extracted standalone node, not the pencil image.

After all storyboards are approved, proceed to the video-model choice card.

## STEP 7: Video-Model Choice Card + Single-Shot Video Clips

Before any clip is rendered, show the video-model choice card and resolution choice card.

Required references:

- Read `references/model-selection.md` for the H3 default, Seedance 2.0 fallback, per-shot mixed mode, resolution choices, and model-specific prompt shaping.
- Read `references/fallback-policy.md` for per-model retry ladders, drift handling, and escalation choices.

Minimum runtime contract:

- H3 is the recommended default model.
- Seedance 2.0 is the fallback for high-stakes animation performance or repeated H3 failure.
- Per-shot mixed mode is allowed only when the shot table marks the model per row; unmarked rows default to H3.
- Strip all storyboard-only labels before video render.
- Bind each clip to the approved text storyboard section, exact character cards, and exact scene card.
- If a clip drifts from the approved `Reference Anchors`, follow `references/fallback-policy.md`; do not silently assemble incorrect clips.

After all clips render, place them on canvas in shot order, group them as `<title> shot clips`, and show the clip approval card defined in `references/model-selection.md`.

## STEP 8: Full Film Assembly, BGM Match, and Final Output

After all single-shot clips are approved, assemble the complete main video, match or generate one continuous BGM track, and produce the final composited video.

Required reference: read `references/qc-checklist.md` for assembly rules, BGM rules, final review checks, canvas ordering/grouping discipline, user choice-card discipline, and regeneration/latest-asset discipline.

Minimum runtime contract:

- Preserve the exact shot order from the approved table.
- Use only approved latest assets.
- Duck BGM under dialogue, reactions, and important SFX.
- Do not add subtitles or text unless explicitly requested.
- Final video must contain no storyboard traces, labels, arrows, timing marks, panel borders, or double-binding labels.

Then run the final review checks from `references/qc-checklist.md` and deliver the final approved asset.

## Boundaries

Do not use this Skill for a single image, a simple edit, a single clip animation, logo design, or pure prompt consultation. If the user only wants a prompt, use a video prompt workflow instead. If the user only wants a character card, use a character breakdown workflow instead.


---

## FILE: skills/3d-animation-short-generator/meta.yaml (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/3d-animation-short-generator/meta.yaml)

display-name-zh: 3D动画短片生成器
version: 0.5.4
tag-en: Animation
tag-cn: 动画
complete-tags-en:
- Animation / Planning
- Animation / Creative Generation
- Animation / Post-production
complete-tags-cn:
- 动画 / 计划制定
- 动画 / 创作生成
- 动画 / 后期制作
summary-en: Turn a story idea into a complete stylized 3D animated short.
summary-cn: 根据故事创意，完成人物与场景设定、镜头规划、分镜生成和视频合成，输出风格统一的3D动画短片。
desc-en: Designed for creators who want to turn a story idea into a complete stylized 3D animated short. Users provide a one-line concept or basic plot and confirm aspect ratio, duration, dialogue needs, and visual direction. The Skill creates a project brief and story outline, builds labeled character and environment-only scene cards, plans a shot table with per-second action, camera, audio, and continuity requirements, produces text storyboards or optional pencil boards, selects a video model, generates individual shots, assembles the film, matches BGM, and performs a final review. It outputs a coherent 3D animated short with consistent characters, continuous scenes, and controlled pacing. Best for narrative animation, birthday keepsakes, brand stories, and social shorts; not for single images, simple retouching, photorealistic live action, or one standalone clip.
desc-cn: 面向希望将故事灵感制作成完整3D动画短片的创作者。用户需提供一句故事创意或基础剧情，并确认画面比例、总时长、对白需求和目标风格。Skill 会依次完成项目简报与故事大纲，生成角色卡和无人物场景卡，规划带每秒动作、镜头、音频与连续性要求的镜头表，制作文本分镜或可选铅笔分镜，再选择视频模型生成单镜头片段，完成全片拼接、BGM匹配和成片复查。最终输出角色一致、场景连续、节奏完整的3D动画短片。适用于剧情动画、生日纪念、品牌故事和社交媒体短片，不适用于单张图片、简单修图、真人写实视频或仅生成一个独立镜头。
author-en: MiniMax Hub User
author-cn: MiniMax Hub 用户
source: community


---

## FILE: skills/3d-animation-short-generator/references/fallback-policy.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/3d-animation-short-generator/references/fallback-policy.md)

# Generation Failure and Drift Fallback Policy

### Failure fallback per video model

H3 fallback ladder when a clip fails or drifts:

1. First retry: regenerate the same clip with the H3 prompt prefix strengthened by quoting the exact `Reference Anchors` block from the table.
2. Second retry: shorten the shot to ≤6s and split the dropped seconds into a new adjacent row in Step 5; re-run Step 5.5 self-check, then re-render.
3. Third retry: switch the failing shot to Seedance 2.0 (this is exactly why the mixed mode exists — performance fallback is one click, not a re-architecture).
4. After three failed attempts on the same shot: pause and ask the user with a choice card:
   - Switch just this shot to Seedance 2.0.
   - Loosen the request (drop a prop, simplify the action, reduce the hook).
   - Skip the shot and add a `placeholder: missing clip` note for downstream review.
   - Manually supply a reference video to bind instead of generating.

Seedance 2.0 fallback ladder (used when the user has explicitly chosen Seedance as the global model, or after H3 already failed three times):

1. First retry: regenerate with a strengthened prompt that quotes the `Reference Anchors` block.
2. Second retry: drop reference images and use text-only generation.
3. Third retry: shorten the shot to ≤6s and split the dropped seconds into a new row.
4. After three failed attempts: ask the user — switch to H3 for this shot, loosen the request, skip the shot, or supply a reference video.

### After all clips are rendered

Place the rendered clips on canvas in shot order, group them as `<title> shot clips` (no longer hard-coded to a single model name), and show a user choice card:

- Approve clips and composite full film (recommended)
- Re-render selected clip (with the same or a different video model)
- Fix character mismatch
- Fix scene mismatch
- Strengthen storyboard cleanup
- Fix spatial anchor drift across clips

If a rendered clip drifts from the approved `Reference Anchors` (e.g. door-frame lands on the wrong side, character exits from the wrong edge, lighting flipped), re-render with a strengthened prompt that quotes the exact `Reference Anchors` block from the table. If the drift persists, switch that one shot to the other video model in the mixed-mode path; do not silently mix corrected and uncorrected clips into assembly.

## Storyboard Visualization Fallback

### Storyboard generation failure fallback (visualization mode only)

If a pencil image storyboard cannot be produced at the required quality (e.g. layout collapses, labels illegible, panels merged, character inconsistency), apply the following escalation before asking the user:

1. **First retry**: regenerate the same shot storyboard with a tightened prompt that explicitly mentions the four-quadrant layout, the `[char:…] [scene:…] [shot:…]` labels, and the per-panel content rules.
2. **Second retry**: drop the bottom-right audio/anchor quadrant text (keep it as a blank cell with a tiny `♪` mark) to reduce text load; this usually fixes illegible labels without losing the visual beat.
3. **Third retry**: reduce panel count by one (e.g. 6 panels → 5 panels by merging the two least-actionable seconds) and simplify camera icons to single arrows.
4. **After three failed attempts on the same shot**: pause and ask the user with a choice card:
   - Switch to a block-color storyboard (gray boxes for poses, no pencil lines) for the failing shot only.
   - Drop the pencil image for the failing shot and rely on the text storyboards document alone for that row.
   - Split the failing shot into two shorter shots in Step 5 and re-run Step 5.5.
   - Manually supply a reference image to bind instead of generating.

In default text mode this whole fallback is unnecessary — text storyboards fail only when the model cannot produce coherent structured text, in which case return to Step 5 to revise the table row.



---

## FILE: skills/3d-animation-short-generator/references/model-selection.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/3d-animation-short-generator/references/model-selection.md)

# Video Model Selection and Prompt Shaping

## STEP 7: Video-Model Choice Card + Single-Shot Video Clips

### Video-model choice card (mandatory before any clip render)

Before any clip is rendered, show the video-model choice card. The choice is stored in the Project Brief and reused for every clip in this project unless the user later changes it.

Video model card:

- **H3 (recommended default)** — strong on visual packaging, motion graphics, text/UI clarity, multi-modal context understanding, and cost efficiency (about 1/3 the price of comparable flagship models at 2K, 1/2 at 768P). Native dual-channel audio. Up to 15s per clip at 2K. Best for: stylized 3D animated shorts with strong design language, text overlays, motion-graphic moments, packaging-style transitions, and dialogue-driven beats where the audio is part of the deliverable.
- **Seedance 2.0 (fallback for high-stakes animation performance)** — strong on cinematic camera, complex shots, elastic Pixar-style performance, and tension-driven action. Best for: chase sequences, slapstick beats, climax shots where the selling point is the animation itself rather than the packaging.
- **Per-shot mixed (advanced)** — let the user mark `video_model: H3` or `video_model: Seedance2` in the `Shot Description` column of individual rows. Use this when the project has both packaging-heavy and performance-heavy shots. The default for unmarked rows is H3.

### Resolution choice card (after video model)

Once the video model is locked, show the resolution choice card:

- 768P (recommended for H3 first pass; cost-efficient)
- 2K (H3 default quality; higher cost, sharper final render)
- 1080p (recommended for Seedance 2.0)
- 720p (Seedance 2.0 draft; lowest cost)
- Match project / custom resolution

The user must confirm a resolution before the first clip renders. Resolution can be changed per clip later if the user wants a hero shot at higher detail.

### Single-shot clip rendering

For each approved table row, call the chosen video model to generate the corresponding independent video clip. Each clip must use exactly the matching section from the text storyboards document (extracted standalone node if that section was extracted, otherwise the in-document section), character card(s), and scene card from that row.

Per-shot rules common to all video models:

- Use the text storyboards document as the authoritative per-shot reference for narrative, composition, camera movement, action staging, per-second timing, and shot number. For shots that have been extracted to a standalone node, read the extracted node instead. If a pencil image storyboard also exists, use it only for human-side pose / silhouette pre-check; do not let it override the text storyboard.
- Use character cards as the authoritative identity source.
- Use scene cards as the authoritative environment source.
- **Strip all storyboard double-binding labels** (`[char:…]`, `[scene:…]`, `[shot:…]`, `[dur:…]`, `[hook:…]`) before video render — these labels are storyboard-only reference markers and must NOT appear in the final clip. Pencil image storyboards additionally have their own shot numbers, camera icons, arrows, and notes that must be removed at render time.
- The rendered clip must contain only clean full-color Pixar-inspired 3D animation content.
- No storyboard line art, no hand-drawn sketch texture, no labels, no subtitles unless requested, no watermarks.
- Maintain the approved screen size / aspect ratio and the approved video resolution from the resolution choice card.

### Model-specific prompt shaping

The text storyboards document (or the extracted standalone node for that shot) feeds both models, but the prompt prefix around the storyboard differs:

- **H3 prompt prefix** (default): emphasize packaging keywords, design language, motion clarity, text/UI presence when relevant, and dual-channel audio intent. H3 is strong at instruction following, so the per-second directive can be sent almost verbatim. Add: `Pixar-inspired 3D cartoon rendering, C4D + Octane look, stylized Q-version proportions, warm SSS skin, designed-with-detail hair, strong character design language, clean motion, on-brand color palette`.
- **Seedance 2.0 prompt prefix** (performance fallback): emphasize cinematic camera language, elastic squash-and-stretch, anticipation, follow-through, lighting drama, and lens choice. Add: `cinematic Pixar-quality 3D animation, elastic squash-and-stretch performance, Disney-style anticipation and overshoot, dramatic key lighting, lens-specific depth of field`.

When the user picked `per-shot mixed`, apply the prefix that matches the row’s `video_model` field.



---

## FILE: skills/3d-animation-short-generator/references/qc-checklist.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/3d-animation-short-generator/references/qc-checklist.md)

# Assembly, Final Review, and Asset Discipline

## STEP 8: Full Film Assembly, BGM Match, and Final Output

After all single-shot clips are approved, concatenate them in table order into the complete main video. Then generate one continuous BGM track that matches the story mood and embed it into the assembled video.

Assembly and BGM rules:

- Preserve the exact shot order from the approved table.
- Match BGM to the assembled video’s actual pacing, emotional arc, comedy beats, chase rhythm, and ending tone.
- Duck BGM under dialogue, non-language reactions, and important SFX.
- Preserve existing clip audio and SFX unless the user asks to replace them.
- Do not generate BGM per shot.
- Do not add subtitles or text unless the user explicitly asks.
- Output the final video with clean animation visuals and no storyboard traces, including no `[char:…] [scene:…] [shot:…]` labels.

Then show a user choice card:

- Approve final film (recommended)
- Regenerate BGM
- Adjust BGM mix
- Re-render selected clip

## STEP 9: Final Review

Create a short final review text node if the user asks for diagnosis or if there are visible risks.

Check:

- Character consistency
- Scene continuity (verify against the `Reference Anchors` column — did every landmark land where the table said it would?)
- Emotional anchor payoff
- Shot purpose clarity
- Dialogue intelligibility
- Foley/SFX sync (against the `Audio & Dialogue Track` column)
- BGM balance
- No storyboard artifacts in final video: no panel borders, sketch lines, arrows, labels, handwritten notes, timing marks, pose ghosts, storyboard text, AND no double-binding labels (`[char:…]`, `[scene:…]`, `[shot:…]`, `[dur:…]`, `[hook:…]`)
- Missing or weak clips
- Any asset that may need regeneration

## Canvas Ordering and Grouping Discipline

Whenever a durable artifact is created, write it to the canvas immediately in the sequence defined in STEP 0. If a generation tool automatically adds outputs to canvas, do not duplicate them.

Group every production section on canvas:

- Group project brief and story outline as `<title> story planning` when both exist.
- Group character cards as `<title> character cards`.
- Group scene cards as `<title> scene cards`.
- Group the standardized shot table as `<title> shot table`.
- Group the text storyboards document as `<title> text storyboards` (default mode, one document). Extracted standalone text storyboard nodes and pencil storyboards (visualization mode) are separate groups: `<title> extracted text storyboards` and `<title> multi-panel pencil storyboards`. Keep all storyboard groups separate from rendered clips because storyboards contain double-binding labels, shot numbers, camera icons, arrows, and sketch lines.
- Keep storyboard groups separate from rendered clips because storyboards contain double-binding labels, shot numbers, camera icons, arrows, and sketch lines.
- Group single-shot video clips as `<title> shot clips` (the label no longer hard-codes a specific model name, since projects can mix H3 and Seedance 2.0).
- Group assembled main video, matched BGM, and final composited video as `<title> final delivery` when they exist.

If a generation round produces two or more outputs, group recent outputs immediately with a clear title. If the renderer has not flushed positions and grouping fails, tell the user to click/drag any canvas node once, then retry grouping before continuing to the next costly stage when possible.

## User Choice Card Discipline

Use a choice card for every place that requires user confirmation. Do not replace these confirmations with plain chat questions or prose. Required choice-card gates:

- Immediately after intake, before any next step, to choose screen size / aspect ratio
- Immediately after intake, before any next step, to choose total duration
- After project brief
- After story outline
- After character cards
- After scene cards
- After standardized shot table (gate 1: approve table before self-check)
- After shot-table self-check passes (gate 2: approve self-check, then immediately choose storyboard mode: text only / text + pencil image)
- After single-shot storyboards (text storyboards document by default with optional extracted standalone nodes; pencil images if the user opted in)
- Before single-shot video-clip rendering, to choose the video model (H3 default, Seedance 2.0 fallback, per-shot mixed)
- Before single-shot video-clip rendering, to choose video resolution
- After single-shot video clips
- After full-film assembly, BGM match, and final composite

Default recommended option should be first. Always allow custom user input. If the user says “continue,” treat it as choosing the recommended option.

## Regeneration and Latest-Asset Discipline

When the user regenerates or revises any artifact, all downstream steps must use the newest approved artifact, not the older one.

Rules:

- If a character card is regenerated, future shot tables, the text storyboards document (plus any extracted standalone text storyboard nodes), pencil storyboards (if visualization mode is on), single-shot video clips, assembled videos, and final composites must reference the regenerated labeled character card by exact character name.
- If a scene card is regenerated, future shot tables, the text storyboards document (plus any extracted standalone text storyboard nodes), pencil storyboards (if visualization mode is on), single-shot video clips, assembled videos, and final composites must reference the regenerated scene card by exact scene name.
- If the shot table is revised, future text storyboards document (or extracted nodes), single-shot video clips, and assembly must use the revised table. The self-check in Step 5.5 must be re-run before storyboarding resumes.
- If a section in the text storyboards document is revised, the matching single-shot video clip must use the revised section. If that section has been extracted to a standalone node, that node is the source of truth; otherwise the document section is.
- If a standalone extracted text storyboard is revised, after the user is satisfied, re-integrate it back into the text storyboards document (replace the placeholder with the latest content) and archive the standalone node.
- In visualization mode, if a pencil storyboard is redrawn, the matching video clip is still bound to the text storyboards document (or extracted node); the pencil image is human-review-only and may be redrawn without forcing a video re-render.
- If a single-shot video clip is re-rendered, assembly, BGM matching, and final composite must use the new clip. If the clip was rendered with H3 and the user wants to re-render with Seedance 2.0 (or vice versa), treat that as a model switch — record the new model in the Project Brief and re-check per-shot rules.
- If BGM is regenerated, the final composite must use the regenerated BGM.

After any regeneration:

1. Mark the regenerated artifact as the current approved version in the next text output or reply.
2. Prefer the regenerated file path / node over previous versions in all subsequent prompts.
3. If there are multiple versions on canvas, identify the chosen current version by filename or node name before continuing.
4. Do not silently mix old and new assets in final assembly.



---

## FILE: skills/3d-animation-short-generator/references/shot-table-spec.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/3d-animation-short-generator/references/shot-table-spec.md)

# Standardized Shot Table Specification

## STEP 5: Standardized Shot Table Video Prompts (Six Columns)

After character cards and scene cards are locked, output standardized video prompts as a shot information table. This step is mandatory and cannot be swapped with storyboard or video generation. Create a canvas table node or markdown table named `标准镜头信息表` or `standard-shot-table`.

The table must have exactly six columns in this order:

| Shot ID & Duration | Continuity Handoff | Reference Anchors (Spatial + Identity) | Hook Type | Shot Description (Per-Second Directives) | Audio & Dialogue Track |
|---|---|---|---|---|---|

Column rules:

- **Shot ID & Duration**: shot number plus planned duration, e.g. `S03 / 6s`.
- **Continuity Handoff**: how this shot naturally continues from the previous shot’s ending image, prop position, eyeline, character posture, sound bridge, or emotional state, AND how it sets up the next shot’s opening. This is the cross-shot continuity spine.
- **Reference Anchors (Spatial + Identity)**: four sub-fields, all mandatory.
  - `Fixed Landmarks` — exact named landmarks from the scene card, with their screen-relative positions (e.g. `door-frame: right third`, `kitchen-island: center bottom`).
  - `Character Positions (camera view)` — for every character in the shot, screen-relative position (left/center/right, top/mid/bottom, foreground/midground/background), facing direction, and initial pose.
  - `Exited Character Status` — for any character who was in the previous shot but is not in this shot, state their off-screen position and reason (e.g. `Mia — exited frame-left, last seen holding apple basket at door-frame`).
  - `Lighting Baseline` — inherited key/fill/rim direction from the scene card, plus any per-shot modifier (e.g. `key: warm overhead, fill: cool bounce right, modifier: window-backlit silhouette`).
  - Plus identity bindings: exact approved character card names and exact approved scene card name.
- **Hook Type**: one short label from a controlled vocabulary, e.g. `visual-joke`, `reversal`, `suspense`, `tender`, `chase`, `reveal`, `callback`, `expression-beat`. Used for the per-episode hook distribution self-check.
- **Shot Description**: shot size, camera movement, Dutch-angle design, performance style, SFX, negative prompt, **video-model generation notes** (the model chosen in Step 7 — H3 or Seedance 2.0 — receives slightly different prompt shapes; for H3 emphasize packaging keywords and text/UI/motion-graphics clarity, for Seedance 2.0 emphasize cinematic camera and elastic performance), and a required `Per-Second Directives` subsection. The subsection must break the shot into second-by-second instructions such as `0–1s`, `1–2s`, `2–3s`; for sub-second critical beats, use `2.0–2.5s` style markers. Each per-second directive MUST cover all five required elements:
  1. Action / pose / expression (squash-and-stretch, anticipation, overshoot, follow-through where applicable)
  2. Camera movement (push / pull / pan / tilt / handheld-shake / locked / orbit)
  3. Spatial position (where the character is, what they hold, what landmark is in frame)
  4. Audio cue (narration / dialogue / SFX / breath / silence — or `silent` if intentional)
  5. Handoff to the next second or next shot (what state this second locks in for the next one)
- **Audio & Dialogue Track**: full audio script for the shot in time order, separate from per-second cues. Fields:
  - `Narration` — voiceover text with time range (omit if no narration).
  - `Dialogue` — line, speaker, tone, time range.
  - `SFX` — keyed sound effects, time-anchored.
  - `Performance Note` — when the protagonist is narrating off-screen, mark `narrator-mouth-closed: true`; describe expression path during narration; describe concrete eye-line and body-action changes for each dialogue line.

Table-wide rules:

- Each shot must naturally inherit the previous shot’s image state through the `Continuity Handoff` column and set up the next shot in the same column.
- Each shot row must include per-second directives that cover the entire shot duration from first frame to last frame, including action, pose, expression, camera movement, spatial position, sound cue, and continuity handoff.
- Per-second directives must be specific enough to generate storyboard panels directly; avoid vague timing such as “continues moving” without body, camera, or prop detail.
- Performance must be exaggerated and elastic, matching Disney-style squash-and-stretch, anticipation, overshoot, follow-through, overlap, arcs, fast pose changes, and clear comedic silhouettes.
- Shot sizes must alternate close-up / extreme close-up with other necessary framing; avoid repetitive framing.
- Dutch-angle tilted compositions must be designed into chase, imbalance, surprise, or slapstick beats.
- Dialogue language only if explicitly requested by the user; otherwise use minimal non-language-specific reactions or mark dialogue as optional / pending confirmation.
- For shots containing narration or dialogue, every second where the character is speaking must record whether the mouth is open or closed; the default is closed for off-screen narrator, open for on-screen dialogue.
- A character who left the frame must still be tracked in `Exited Character Status` for at least one shot, then dropped after they are explicitly off-stage for two consecutive shots.

Then show a user choice card:

- Approve table and run self-check (recommended)
- Adjust shot continuity
- Make animation more exaggerated
- Adjust close-up / extreme-close-up rhythm
- Adjust Dutch-angle design

## STEP 5.5: Shot Table Self-Check Gate (Mandatory)

Before moving to pencil storyboards, run a hard self-check on the approved shot table. If any check fails, revise the table and re-run before asking the user to approve storyboarding.

Six required checks:

1. **Hook density**: every shot has a `Hook Type`; at least one of every three consecutive shots uses a `reveal`/`reversal`/`callback`; the opening shot and the closing shot each carry a strong hook (`visual-joke` / `reversal` / `reveal` / `suspense` / `tender`).
2. **Single-shot duration**: no shot exceeds 15 seconds. If a beat needs more, split it.
3. **Character count per shot**: no shot contains more than three important characters (defined as characters with on-screen action or dialogue).
4. **Spatial anchor inheritance**: for every interior scene with two or more shots, the `Fixed Landmarks` and `Lighting Baseline` of the next shot must either match the previous shot or include an explicit continuity note (e.g. `door-frame moves from right third to center as camera orbits left`).
5. **Per-second directive coverage**: every second from `0s` to the shot duration is covered by a `Per-Second Directives` entry, and each entry contains all five required elements (action/pose/expression, camera, spatial, audio cue, handoff). Sub-second beats like `2.0–2.5s` are allowed but must not leave any time gap.
6. **Cross-shot continuity**: reading the `Continuity Handoff` column row by row produces a continuous chain — no shot starts from a state that contradicts the previous shot’s ending. Any shot that flips eyeline, character position, prop state, or lighting must mark the flip explicitly (e.g. `HARD CUT — time skip: 2h`).

If all six pass, place a `shot-table self-check: passed` stamp at the top of the canvas table node and show the user choice card:

- Approve self-check and draw shot storyboards (recommended)
- Show self-check details
- Revise failed checks
- Re-run self-check

If any check fails, do not enter Step 6. Return to Step 5, list the failed rows, and only re-show the storyboard approval card after the table is fixed and the self-check passes.


---

## FILE: skills/3d-animation-short-generator/references/storyboard-guidelines.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/3d-animation-short-generator/references/storyboard-guidelines.md)

# Text and Pencil Storyboard Guidelines

## STEP 6: Text Storyboards Document (Default) + Pencil Image Storyboards (Opt-in)

After the Step 5.5 self-check passes, show a storyboard-mode choice card before producing any storyboard artifact:

- **Text storyboards document only (default, recommended)** — one canvas text node containing all shot storyboards as in-document sections. Mirrors the half-narrated-drama storyboard structure: per-shot fields (title / hook / scene / characters / spatial anchors / continuity / performance) plus Pixar's per-panel four-quadrant content + optional ASCII layout. Carries the full quality-control payload at near-zero cost. The video model selected in Step 7 reads this directly as the per-shot rendering reference.
- **Text storyboards document + multi-panel pencil image (visualization mode, opt-in)** — the text storyboards document is still produced as the authoritative artifact, AND one multi-panel pencil image is generated per shot for human review. Higher cost, useful when the user wants a visual preview before committing to video generation, or when squash-and-stretch / pose silhouette is the main risk and the user wants to pre-check it visually.

Store the chosen storyboard mode in the Project Brief and reuse it in Step 7, Step 9, and the Regeneration discipline.

### Default path: single text storyboards document

Generate one canvas text node named `<title> text storyboards` (one document for the whole short). This document is the authoritative rendering reference for Step 7 even when pencil images are also produced. The structure mirrors the half-narrated-drama storyboard — every shot is a section in the same document, so the user can read cross-shot continuity without node-hopping.

Document top matter (header block at the top of the document):

- Project title, approved video model, approved resolution, storyboard mode, and self-check status (e.g. `shot-table self-check: passed at <timestamp>`).
- A short table-of-contents listing every shot, its hook, and its section anchor (`S01`, `S02`, …) so the user can jump.

Per-shot section structure (one `##` heading per shot, in shot order). Every section is mandatory to contain these fields, in this order — direct adaptation of the half-narrated-drama storyboard:

1. **Shot title & duration** — short human-readable title for the shot, plus `S<N> / <duration>s` (e.g. `S03 / 6s`).
2. **Hook type** — one of the controlled vocabulary: `setup` / `visual-joke` / `reversal` / `reveal` / `callback` / `suspense` / `tender` / `chase` / `expression-beat` / `climax`. Used by the per-episode hook distribution self-check.
3. **Scene & characters** — exact scene card name and exact on-screen character names (binding to character cards).
4. **Spatial anchor card** (mandatory, four sub-fields — directly adapted from half-narrated):
   - `Fixed landmarks` — named landmarks and their screen-relative positions (e.g. `door-frame: right third`, `kitchen-island: center bottom`).
   - `Character positions (camera view)` — for every on-screen character, screen-relative position, facing direction, and initial pose.
   - `Exited character status` — characters who were on screen in the previous shot but not in this one, with their off-screen position and reason.
   - `Lighting baseline` — inherited key/fill/rim direction from the scene card, plus per-shot modifier.
5. **Continuity** (mirrors half-narrated's handoff fields):
   - `Continuity from S(N-1)` — one or two sentences referencing the previous shot’s ending state.
   - `Continuity to S(N+1)` — one sentence setting up the next shot’s opening.
6. **Double-binding** — `[char:角色名-01] [char:角色名-02] ... [scene:场景名] [hook: visual-joke]` — exact character card names, scene card name, hook type. These are storyboard-only reference markers; the video model strips them at render time.
7. **Per-panel four-quadrant content** (one block per panel, in time order; this is the Pixar per-second directive, kept verbatim from the table row):
   - `Timecode` — e.g. `0–1s`.
   - `Pose + Expression` — concrete body posture, silhouette, key prop grip, eye-line, facial expression path; for elastic beats, explicitly call out squash / stretch / anticipation / overshoot. This is the largest section per panel and is what the video model reads as the visual beat.
   - `Camera` — shot size, camera movement (push / pull / pan / tilt / handheld-shake / locked / orbit), Dutch angle note when applicable.
   - `Audio + Anchor` — audio cue (`♪ narration: ...` / `dialogue: ...` / `SFX: ...` / `silent`) and spatial anchor note (`door-frame: right third` / `Mia: center midground facing camera`).
   - Performance notes (mirrors half-narrated): for narration seconds, mark `narrator-mouth-closed: true`; for on-screen dialogue, mark `mouth-open: speaker` and describe expression path / eye-line / body-action changes.
8. **Layout rules** (apply per shot):
   - 3-second shot → 3 panels (1 per second).
   - 4-second shot → 4 panels.
   - 5-second shot → 5 panels.
   - 6-second shot → 6 panels.
   - 7+ second shot → one panel per second; for sub-second critical beats, add an extra mini-panel such as `2.0–2.5s` only when the beat is the hook of the shot.
   - Panels must cover the full shot duration from first frame to last frame with no time gaps.
9. **Per-panel binding**:
   - Bind the exact character cards listed in the table row to lock appearance, face, hairstyle, body proportions, costume, signature props, and role identity. Use the same character names as the table.
   - Bind the exact scene card listed in the row to preserve environment, props, landmarks, movement paths, and spatial logic.
10. **Optional ASCII layout block (highly recommended, free)**:
    - Append a small ASCII sketch per panel (or one combined sketch for the whole shot) so the user can scan the spatial layout in seconds without rendering an image. Example:
      ```
      [0-1s]  Mia (L, mid)         door-frame (R)
              ──kneels, hands on apple basket──
              cam: low push-in, locked
              audio: silent | anchor: basket center-bottom
      [1-2s]  ...
      ```
    - The ASCII block is informational only; the video model reads the structured `Per-panel four-quadrant content` above, not the ASCII.
11. **Storyboard-only markers**:
    - When a beat is critical, append `[BEAT]` after the panel timecode.
    - When a panel must handoff a specific state to the next panel or the next shot, append `[HANDOFF → ...]` with a short label such as `[HANDOFF → S04 opening]`.

Per-shot section template (copy-paste skeleton, valid for any shot):

```markdown
## S03 / 6s — Title: 奶奶把苹果筐递给 Mia

- **Hook type**: reveal
- **Scene & characters**: scene:kitchen | char:Mia, char:Grandma
- **Spatial anchor card**:
  - Fixed landmarks: door-frame (right third), kitchen-island (center bottom)
  - Character positions: Mia (L, midground, facing camera) | Grandma (R, foreground, facing Mia)
  - Exited character status: —
  - Lighting baseline: warm overhead key + cool bounce right
- **Continuity from S02**: 奶奶弯下腰从中岛拿起苹果筐
- **Continuity to S04**: Mia 接住筐转身，门铃响起
- **Double-binding**: [char:Mia] [char:Grandma] [scene:kitchen] [hook:reveal]

### Per-panel four-quadrant content

#### 0–1s
- Pose + Expression: 奶奶弯腰双手持筐；Mia 左侧站姿，眼神好奇
- Camera: locked medium shot, eye-level
- Audio + Anchor: silent | Mia: L midground | basket: center bottom
- Performance: [BEAT]

#### 1–2s
- Pose + Expression: 奶奶手臂伸向 Mia，筐倾斜；Mia 双手前伸准备接
- Camera: locked medium shot, eye-level
- Audio + Anchor: ♪ SFX: basket rustle | anchor: door-frame: right third
- Performance: [HANDOFF → S04 opening]

#### 2–3s
...

### ASCII layout (optional)
[0-1s]  Grandma (R, fg)        door-frame (R, bg)
        ──lifts basket──        Mia (L, mid)
        cam: locked | silent
[1-2s]  ...
```

After all sections are written, place the document on canvas and move directly to Step 7. Do not call any image generation model in default mode.

### Shot-level extraction (heavy-iteration mode)

The default single-document form is optimized for reading and cross-shot continuity. When the user flags a specific shot for heavy iteration (typically climax / chase / slapstick beats where the per-panel content needs many rounds of revision), extract that section into a standalone text node so iteration is localized:

- User signal: at any time after Step 6, the user says things like "let me focus on S05", "S05 needs rework", "extract S05", or selects a shot during the storyboard approval choice card.
- Extraction mechanics:
  1. Create a new canvas text node named `<title> S05 text storyboard (extracted)`.
  2. Move the full content of the `## S05` section from the document into the new node.
  3. In the document, replace the `## S05` section with a one-line placeholder: `> S05 — extracted to standalone node (see `<title> S05 text storyboard (extracted)`)`.
  4. Step 7 reads from the extracted node for S05; all other shots still read from the document.
- Re-integration: when the user is satisfied, the standalone node is folded back into the document (replace the placeholder with the latest content) and the standalone node is archived.
- Multiple extracted shots: each shot gets its own standalone node; the document tracks them with placeholders.

The extraction mechanism exists because independent nodes are best used by need, not by default — but they remain available whenever iteration pressure is high on a specific shot.

### Opt-in path: multi-panel pencil image storyboards (visualization mode)

If the user picked the visualization mode in the storyboard-mode choice card, ALSO produce one multi-panel pencil storyboard image per table row on top of the text storyboards document. The text storyboards document remains the authoritative rendering reference; the pencil images are human-review-only.

For each pencil image storyboard:

- **Double-binding labels (top-right corner, mandatory on image)**:
  - `[char:角色名-01] [char:角色名-02] ...` — exact character card names used in this row.
  - `[scene:场景名]` — exact scene card name.
  - `[shot: S03] [dur: 6s] [hook: visual-joke]` — shot ID, duration, and hook type.
  - These labels are storyboard-only reference markers; they are stripped at video render time.
- Bind the exact character cards listed in that row to lock character appearance, face, hairstyle, body proportions, costume, signature props, and role identity.
- Bind the exact scene card listed in that row to preserve environment, props, landmarks, movement paths, and spatial logic.
- Convert every per-second directive in the row into one storyboard panel or one clearly labeled beat panel; for a 4-second shot, normally create 4 panels; for a 6-second shot, normally create 6 panels; for sub-second critical beats, add extra mini-panels only when needed.
- **Panel physical layout (mandatory)**:
  - 3-second shot → 1×3 strip.
  - 4-second shot → 2×2 grid.
  - 5-second shot → top row 3 + bottom row 2.
  - 6-second shot → 2×3 grid.
  - 7+ second shot → 3 rows, balanced panels.
  - Each panel occupies the same canvas area; do not let one panel dominate.
- **Per-panel four-quadrant content (mandatory)**:
  - Top-left: timecode (e.g. `0–1s`).
  - Top-right: pose + expression sketch (the largest area; the actual visual beat).
  - Bottom-left: camera icon + movement arrow (push/pull/pan/orbit/locked) and a tiny note for Dutch angle.
  - Bottom-right: audio cue (e.g. `♪ narration: "I knew it."` / `SFX: door creak` / `silent`) and anchor note (e.g. `door-frame: right third`).
- Arrange panels in reading order inside the same single-shot storyboard image; do not merge multiple different shots into one image.
- Each panel must mark its timecode, such as `0–1s`, `1–2s`, and show the corresponding pose, expression, action, camera movement, prop position, SFX cue, and continuity handoff.
- Output pure black-and-white pencil line-art only: no color, no final-render lighting, no polished 3D render.
- Mark the storyboard image with the shot number and include camera-movement icon / marker per panel when useful.
- Include storyboard-only marks when useful: pencil construction lines, action arrows, camera-path icon, timing marks, and small notes.
- Keep the draft as a video-render reference asset only, not final art.

### Storyboard approval (both modes)

After all text storyboard sections (and pencil images, if visualization mode is on) are produced, place them on canvas in shot order, group them as:
- `<title> text storyboards` (default mode, single document), OR
- `<title> text storyboards + multi-panel pencil storyboards` (visualization mode, group the text document and the pencil images separately because pencil images contain double-binding labels, ASCII labels, and shot numbers that the text document does not).

Show a user choice card:

- Approve storyboards and render shot videos (recommended)
- Extract / re-integrate a shot (move section between document and standalone node)
- Redraw selected pencil storyboard (visualization mode)
- Fix character consistency
- Fix scene logic
- Fix camera marker
- Fix audio/anchor markers

### Storyboard generation failure fallback (visualization mode only)

If a pencil image storyboard cannot be produced at the required quality (e.g. layout collapses, labels illegible, panels merged, character inconsistency), apply the following escalation before asking the user:

1. **First retry**: regenerate the same shot storyboard with a tightened prompt that explicitly mentions the four-quadrant layout, the `[char:…] [scene:…] [shot:…]` labels, and the per-panel content rules.
2. **Second retry**: drop the bottom-right audio/anchor quadrant text (keep it as a blank cell with a tiny `♪` mark) to reduce text load; this usually fixes illegible labels without losing the visual beat.
3. **Third retry**: reduce panel count by one (e.g. 6 panels → 5 panels by merging the two least-actionable seconds) and simplify camera icons to single arrows.
4. **After three failed attempts on the same shot**: pause and ask the user with a choice card:
   - Switch to a block-color storyboard (gray boxes for poses, no pencil lines) for the failing shot only.
   - Drop the pencil image for the failing shot and rely on the text storyboards document alone for that row.
   - Split the failing shot into two shorter shots in Step 5 and re-run Step 5.5.
   - Manually supply a reference image to bind instead of generating.

In default text mode this whole fallback is unnecessary — text storyboards fail only when the model cannot produce coherent structured text, in which case return to Step 5 to revise the table row.



---

## FILE: skills/README.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/README.md)

# MiniMax H3 Skills

This directory contains the skills bundled with [MiniMax H3](../README.md): **1 prompt writing skill** and **8 style-specific video generation skills**. Each skill lives in its own folder with an installable `SKILL.md` (plus a `SKILL.cn.md` Chinese version for the style skills) and any reference materials it needs.

## Status

The skills are actively maintained and still evolving. The 8 style skills ship with bilingual `SKILL.md`/`SKILL.cn.md`; `h3-prompt-writing` is currently English-only.

## Installation

Install skills with the [skills CLI](https://github.com/vercel-labs/skills):

```bash
# List all skills available in this repository
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --list

# Install all skills
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill '*'

# Install a single skill
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
```

## Skills

### h3-prompt-writing

[SKILL.md](h3-prompt-writing/SKILL.md)

Write structured MiniMax H3 video generation prompts for all five generation modes: T2VA, I2VA, FL2VA, L2VA, and Ref2VA. The skill rewrites multimodal requests into H3's prompt structure — `integrated_multimodal_description`, `overall_soundscape`, and `non_diegetic_music` — aligns keyframes, and defines reference labels for images, videos, and audio. It ships with two prompt guides under [`references/`](h3-prompt-writing/references/):

- [`base-en.txt`](h3-prompt-writing/references/base-en.txt) — base text/keyframe modes
- [`ref-en.txt`](h3-prompt-writing/references/ref-en.txt) — full-reference (Ref2VA) mode

### minimalist-product-ad-generator

<p align="center">
  <img src="../assets/minimalist-product-ad-generator.gif" alt="minimalist-product-ad-generator" width="240">
</p>

Turn product images and ad requirements into clean, minimalist product ad shorts for e-commerce promotion and product launches. The skill confirms format and product variants, extracts selling points, writes concise English ad copy, plans beat-synced typography and storyboards, and generates a premium product film with polished camera language. Not for KOC talking-head ads, general editing, or complex screen demos.

[SKILL.md](minimalist-product-ad-generator/SKILL.md) · [SKILL.cn.md](minimalist-product-ad-generator/SKILL.cn.md)

### 3d-animation-short-generator

<p align="center">
  <img src="../assets/3d-animation-short-generator.gif" alt="3d-animation-short-generator" width="240">
</p>

Create complete stylized 3D animated shorts from a story idea through an ordered production workflow: project brief, story outline, character and environment cards, standardized shot planning, text or optional pencil storyboards, video-model selection, single-shot generation, assembly, BGM matching, and final review. Built for end-to-end narrative animation with strong character consistency, scene continuity, timing, camera, performance, and audio control. Not for single images, simple edits, photorealistic live action, or one standalone clip.

[SKILL.md](3d-animation-short-generator/SKILL.md) · [SKILL.cn.md](3d-animation-short-generator/SKILL.cn.md)

### papercraft-stop-motion-explainer

<p align="center">
  <img src="../assets/papercraft-stop-motion-explainer.gif" alt="papercraft-stop-motion-explainer" width="240">
</p>

Explain science, education, or general knowledge through tactile handmade papercraft visuals. The skill extracts the learning goal and visual metaphor, proposes creative directions, designs paper characters, layered diorama sets, and props, creates preview concepts plus image and video prompts, and plans storyboards, camera movement, transitions, and sound with staged approvals and review checklists. It outputs a production-ready papercraft stop-motion explainer package, or selected assets such as still prompts, image-series prompts, short-video prompts, or storyboards. Best for cut-paper, pop-up-book, layered diorama, and miniature stop-motion explainers.

[SKILL.md](papercraft-stop-motion-explainer/SKILL.md) · [SKILL.cn.md](papercraft-stop-motion-explainer/SKILL.cn.md)

### brand-promo-video-generator

<p align="center">
  <img src="../assets/brand-promo-video-generator.gif" alt="brand-promo-video-generator" width="240">
</p>

For marketers and creators producing promotional content for brands, products, websites, apps, shops, or personal projects. The skill organizes brand facts and asset provenance, selects a narrative direction, plans precise beats and shots, generates needed imagery, video, voiceover, or music, and completes assembly and pre-delivery review. It outputs a promotional short that highlights product capabilities, use cases, and a call to action. Best for launches, website showcases, and social promotion; not for imitating real brand marks without authorized assets or inventing product claims.

[SKILL.md](brand-promo-video-generator/SKILL.md) · [SKILL.cn.md](brand-promo-video-generator/SKILL.cn.md)

### music-video-subtitle-generator

<p align="center">
  <img src="../assets/music-video-subtitle-generator.gif" alt="music-video-subtitle-generator" width="240">
</p>

For musicians, video creators, and social-media editors producing AI music videos or emotional short films with lyric typography. The skill analyzes beat and vocal timing, separates character, scene, and text references, designs beat-reactive spatial typography, decomposes long works into connected shots, audits prompts, and routes generation for H3 or other video tools. It outputs MV concepts, shot prompts, lyric text plans, and stitching guidance. Best for stylized MVs and subtitle-driven music visuals.

[SKILL.md](music-video-subtitle-generator/SKILL.md) · [SKILL.cn.md](music-video-subtitle-generator/SKILL.cn.md)

### co-op-game-intro-generator

<p align="center">
  <img src="../assets/co-op-game-intro-generator.gif" alt="co-op-game-intro-generator" width="240">
</p>

Create a two-player co-op game menu or opening animation. The skill locks identity cues, generates an approval image from a fixed menu framework with coordinated color, buttons, icons, and typography, then uses the approved result to rebuild character, UI-copy, and event timing instructions for the final video. It outputs a co-op game intro featuring two characters, player cards, and menu interaction motion. Best for game concepts, character-led menus, and social content.

[SKILL.md](co-op-game-intro-generator/SKILL.md) · [SKILL.cn.md](co-op-game-intro-generator/SKILL.cn.md)

### paper-collage-explainer-generator

<p align="center">
  <img src="../assets/paper-collage-explainer-generator.gif" alt="paper-collage-explainer-generator" width="240">
</p>

Give narration, knowledge points, opinions, or abstract topics a tactile paper-collage language. The skill extracts meaning, proposes visual metaphors, prepares a production plan and storyboard, generates approved halftone collage stills, then creates stop-motion clips with paper movement and tactile sound effects, with optional final assembly. By default it keeps collage SFX and does not add BGM, voiceover, or subtitles unless requested. Best for explainers, viewpoints, story visuals, and social B-roll.

[SKILL.md](paper-collage-explainer-generator/SKILL.md) · [SKILL.cn.md](paper-collage-explainer-generator/SKILL.cn.md)

### handdrawn-live-video-generator

<p align="center">
  <img src="../assets/handdrawn-live-video-generator.gif" alt="handdrawn-live-video-generator" width="240">
</p>

Create surreal short videos that blend rough glowing hand-drawn animation with live-action spaces. The skill clarifies the physical contact, designs continuous morphing, an escape route, and a delayed handheld chase movement, then writes a reusable 15-second 16:9 video prompt in the user's language. After confirmation it recommends MiniMax H3 generation and checks contact realism, camera delay, rough glowing stroke texture, and non-horror tone. Best for single-scene creative clips, not polished CG, horror jump scares, plush characters, or multi-scene cuts.

[SKILL.md](handdrawn-live-video-generator/SKILL.md) · [SKILL.cn.md](handdrawn-live-video-generator/SKILL.cn.md)

## Contribute

These skills are still being improved, and community contributions are encouraged. If you optimize an existing skill or add a new one, open a PR — contributing or optimizing skills comes with API credit rewards.


---

## FILE: skills/brand-promo-video-generator/SKILL.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/brand-promo-video-generator/SKILL.md)

---
name: brand-promo-video-generator
description: For marketers and creators producing promotional content for brands, products, websites, apps, shops, or personal projects. Users provide logos, product images, interface screenshots, official links, or other verifiable assets and confirm duration, aspect ratio, audience, and campaign focus. The Skill organizes brand facts and asset provenance, selects a narrative direction, plans precise beats and shots, generates needed imagery, video, voiceover, or music, and completes assembly and pre-delivery review. It outputs a promotional short that highlights product capabilities, use cases, and a call to action. Best for launches, website showcases, and social promotion; not for imitating real brand marks without authorized assets, inventing product claims, or producing long-form narrative films.
compatibility: Requires the MiniMax Hub agent (canvas workspace and the hub_* tools listed below); not portable to generic agent harnesses.
allowed-tools:
- webfetch
- hub_image_search
- hub_analyse_media
- hub_canvas_get_node
- hub_canvas_group_recent_outputs
- hub_generate_image
- hub_generate_video
- hub_generate_audio
- hub_generate_music
- hub_synthesize_speech
- hub_video_edit
- hub_audio_meta
- task
---

# Brand Promo Video Generator

Create a polished short promo video for a brand, product, website, app, shop, or personal project. Use this Skill when the user has a logo, product images, screenshots, a website link, or just a clear idea and wants the agent to turn those materials into a clean brand reel.

This Hub adaptation replaces third-party Vibe Motion / Remotion implementation details with Hub-native orchestration: source research, asset verification, story planning, image/video generation, optional speech or music, and editing assembly. Do not initialize external projects or depend on npm validators during normal execution.

## Tool Coverage Rule

The `allowed-tools` list must cover the full production promise in this Skill: source lookup, image preparation/generation, video clip generation, optional speech/music/audio generation, final editing/assembly, media inspection, and canvas grouping. If a runtime does not provide one of the listed generation or editing tools, downgrade the deliverable explicitly to a pre-production package instead of claiming that a final promo video can be generated.

## STEP 1: Intake assets and resolve the brief

Before any story planning or generation, run a required user intake. Ask the user to upload or provide links to the elements that must be verified:

- Logo files or official logo source pages
- Font files, font names, typography guidance, or official website pages that show brand typography
- Brand colors, color system, style guide, or pages that clearly show official colors
- Product images, UI screenshots, packaging, renders, footage, or other brand imagery
- Product information: official product name, feature list, launch focus, claims, CTA, disclaimers, and target audience
- Company/product official URL or official source package

Treat user-uploaded assets as usable by default for pre-production and concept planning. Do not repeatedly ask a standalone rights/permission question during the first intake unless there is a concrete risk signal, such as a visible third-party watermark, obviously scraped marketplace imagery, contradictory user wording, legal/medical/financial compliance claims, or the user asks for commercial publication. Record the source as "user-provided" and surface rights caveats in the source summary instead of blocking the flow.

In the same opening intake, ask the user to choose:

- Target duration, normally 15-30 seconds; recommend 15 seconds when the user wants a fast launch film
- Aspect ratio; offer common choices such as 16:9, 9:16, 1:1, 4:3, 3:4, or match a supplied reference

Also identify campaign focus, distribution channel, narration language, on-screen copy language, and visible copy needs when they are not already clear. Do not proceed to creative direction until the user has supplied the usable materials or explicitly confirms which elements are unavailable.

Language rule for promo content: choose narration and on-screen copy language from the brand materials, target audience, and platform context, not mechanically from the chat language. If the brand assets and visible source copy are primarily English or global corporate English, default narration and on-screen copy to English unless the user explicitly asks for Chinese localization. If the user is Chinese but says they are testing as a new user, keep chat replies in Chinese, but plan the actual video copy in the language that best fits the brand campaign.

If a logo, product UI, person, mascot, packaging, font, color system, or other identity-bearing asset cannot be authenticated, stop and ask for an authorized original instead of generating a plausible substitute.

## STEP 2: Build the brand truth sheet

Research or inspect the strongest available sources and summarize the brand truth sheet before creative production:

1. User-provided original exports
2. Official company website, static bundle, newsroom, brand portal, media kit, press kit, or official repository
3. Company-controlled media library
4. Licensed stock or authorized partner kit

Extract:

- Exact logo variants, clear space, and usage constraints
- Official fonts or visible typographic behavior
- Primary, secondary, and dynamic brand colors
- Brand tone, principles, visual motifs, and interaction language
- Current product names, features, scenarios, metrics, slogans, CTA, and disclaimers
- Official photography, renders, UI screenshots, footage, press assets, and media kit material

Do not use logo aggregation sites, search thumbnails, fan recreations, Pinterest reposts, or AI-generated substitutes as identity-bearing sources. Procedural graphics are allowed only as non-representational motion layers: masks, gradients, color fields, grids, glows, particles, trails, typography, verified-data charts, and transition geometry.

## STEP 3: Create a provenance manifest

Record every identity-bearing asset in a compact manifest that can be delivered to the user. The manifest may be a text node or table and should include:

- Stable asset ID and role
- Local path or canvas node reference when available
- Exact source URL or user-provided source note
- Source type, such as official website, media kit, user-provided original, or licensed stock
- Verification target used for comparison
- Rights or publication note
- Authenticity status: verified, user-supplied, licensed, or blocked

The manifest does not grant publication rights by itself. If authorization is unclear, label the video as an unofficial concept and tell the user commercial publication requires permission.

## STEP 4: Choose the story spine

Present 2-3 concise creative directions when the user has not already chosen one, recommend one, and continue after confirmation. Use the product category to pick a spine:

- AI / SaaS: user intent -> thinking or planning -> capabilities -> execution -> useful output -> proof -> logo
- Physical product: hero reveal -> interaction -> feature macro -> usage context -> result -> logo
- Service / company: context -> process -> evidence -> outcome -> promise -> logo
- Image-led brand: authentic imagery -> visual motif -> benefit -> emotional payoff -> logo

Keep the story product-specific. Show actual features, interactions, scenarios, outputs, and proof instead of hiding the story behind abstract effects.

## STEP 5: Plan exact beats

Plan a frame-aware timeline before generation. Use 30fps as the planning convention unless the output pipeline requires otherwise.

For a 15-second film, target 5-8 major beats. For a 30-second film, target 8-12 major beats. Each beat should define:

- Start and end time or frame range
- Visual owner and authentic asset IDs
- Primary action
- Product or brand proof shown in the shot
- Copy and readable hold
- Color state
- Incoming and outgoing transition
- Motion intent: setup, anticipation, commitment, impact, brake, settle

A useful 15-second pattern is: brand hook, user intent or setup, product mechanism, capabilities or scenarios, output or proof, product payoff, final logo and CTA. Use 6-12 frame overlaps when outgoing motion naturally supplies the next shot.

## STEP 6: Direct the motion language

Build intensity with control:

- Let product motion, cursor paths, UI flow, light, scrolling content, object edges, or matched geometry drive transitions
- Use 2-5 deliberate color states tied to meaning
- Keep one primary action per beat; delay secondary layers slightly
- Establish 2-3 high-energy peaks and quieter braking moments
- Preserve readable silhouettes, copy, and logo clear space
- Avoid fake HUDs, arbitrary glass cards, decorative text walls, unverified metrics, and identical easing everywhere

For AI products, include at least one readable chain such as: prompt -> planning -> parallel capabilities -> generated result -> proof. For physical or service products, show cause and effect from user action to concrete outcome.

## STEP 7: Hard confirmation before generation

Before generating any video, image sequence, speech, music, or final edit, stop and show the user the completed pre-production package:

- Provenance manifest or source summary
- Brand truth sheet
- Chosen creative direction
- Exact beat / shot plan
- Visible copy, CTA, narration, and audio plan
- Known authenticity, rights, or placeholder caveats

Use a concise confirmation step before generation. If the user clearly expresses approval or intent to proceed after seeing the pre-production package — for example "confirm", "generate", "go ahead", "continue", "next", "可以", "继续", "下一步", or similar — treat it as permission to generate, unless the message also asks for changes. If the user asks to skip the process, still provide a compact source summary, brand truth sheet, and beat plan first, then proceed when they indicate approval. Offer revision choices only when the user's reply is ambiguous or requests changes.

## STEP 8: Produce Hub assets

Use Hub-native generation and editing only after the hard confirmation gate has passed. Keep each dispatch self-contained with the chosen model, aspect ratio, authentic reference paths, and original request.

Typical production flow:

1. Generate or prepare verified still frames, UI plates, product hero frames, or motion-ready story images.
2. Generate video clips from those frames or from precise text prompts, preserving the same ratio and brand assets.
3. Default audio policy for native brand reels: when the user asks for BGM, music, soundtrack, ambient sound, or says nothing beyond needing a finished promo video, prefer video-native audio from the selected video model instead of generating a separate music track. For the default MiniMax H3 route, set native audio on (`generate_audio=true`) and prompt for brand-safe instrumental music / UI sound design inside the video prompt.
4. Generate separate speech or music only when the user explicitly needs controllable narration, voiceover, dialogue, replaceable standalone BGM, exact music duration independent of the video, post-production remixing, or when the selected video model cannot generate suitable audio. Do not duplicate a soundtrack between video-native audio and separate audio generation.
5. Assemble clips, any explicitly separate audio, and final brand lockup in editing. Add subtitles only when the user explicitly asks for subtitles.

Do not redraw or approximate logos, wordmarks, product UI, packaging, mascot, person, or brand scene. Use generated material only for abstract motion, atmosphere, transition geometry, or clearly conceptual scenes that do not impersonate official product evidence.

## STEP 9: Verify before delivery

Before final response, check:

- The logo and identity-bearing assets came from verified or user-authorized sources
- Product names, feature wording, claims, metrics, slogans, and CTA match official sources or are clearly marked as concept copy
- The video duration, aspect ratio, and language match the brief
- Copy is readable and not overcrowded
- The final logo is not stretched, cropped, or rebuilt
- Motion has clear visual ownership and does not obscure the product
- The output is on the canvas and multi-asset outputs are grouped

If the output fails an authenticity check, replace the questionable asset with an official/user-authorized source or stop and ask for the asset. Never improve an imitation.

## STEP 10: Deliver

Provide:

- Final video path or canvas output
- Duration, aspect ratio, and language
- Short creative summary
- Provenance manifest or source summary
- Rights/disclaimer note when needed
- Specific suggestions for the next iteration, such as pacing, claim clarity, CTA, audio, or platform crop

## Failure recovery

- Wrong or approximate logo: remove it, locate the current official file or ask for the user's original, then regenerate or re-edit.
- Fake-looking product/UI: replace with official, user-supplied, or licensed media. Do not polish the imitation.
- Beautiful but generic: add a complete product interaction, verified claim, or real output.
- Fast but chaotic: reduce simultaneous actions, assign a visual owner, and preserve matched motion across cuts.
- Smooth but slow: shorten holds, overlap transitions, and brake only around key messages.
- Asset unavailable: ask for an authorized original; never guess.


---

## FILE: skills/brand-promo-video-generator/meta.yaml (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/brand-promo-video-generator/meta.yaml)

display-name-zh: 品牌宣传短片生成器
version: 0.1.9
tag-en: Commercial Ad
tag-cn: 商业广告
complete-tags-en:
- Commercial Ad / Planning
- Commercial Ad / Creative Generation
- Commercial Ad / Post-production
complete-tags-cn:
- 商业广告 / 计划制定
- 商业广告 / 创作生成
- 商业广告 / 后期处理
summary-en: Turn verified brand assets and campaign goals into a polished promotional short video.
summary-cn: 基于品牌素材与推广目标，完成事实核验、创意方向、分镜和音画合成，输出可展示的品牌宣传短片。
desc-en: For marketers and creators producing promotional content for brands, products, websites, apps, shops, or personal projects. Users provide logos, product images, interface screenshots, official links, or other verifiable assets and confirm duration, aspect ratio, audience, and campaign focus. The Skill organizes brand facts and asset provenance, selects a narrative direction, plans precise beats and shots, generates needed imagery, video, voiceover, or music, and completes assembly and pre-delivery review. It outputs a promotional short that highlights product capabilities, use cases, and a call to action. Best for launches, website showcases, and social promotion; not for imitating real brand marks without authorized assets, inventing product claims, or producing long-form narrative films.
desc-cn: 面向需要为品牌、产品、网站、App、小店或个人项目制作宣传内容的运营与创作者。用户需提供 LOGO、产品图、界面截图、官网链接或其他可核验素材，并确认时长、画幅、受众和推广重点。Skill 会整理品牌事实与素材来源，选择叙事方向，规划精确节拍和镜头，生成所需图像、视频、旁白或音乐，并完成合成与交付前检查。最终输出一条突出产品功能、使用场景和行动号召的品牌宣传短片。适用于新品发布、官网展示和社交媒体推广，不适用于缺少授权素材时仿造真实品牌标识、虚构产品功能或制作长篇剧情影片。
author-en: MiniMax Hub
author-cn: MiniMax Hub
source: official-featured


---

## FILE: skills/co-op-game-intro-generator/SKILL.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/co-op-game-intro-generator/SKILL.md)

---
name: co-op-game-intro-generator
description: For users creating a two-player co-op game menu or opening animation. Users provide two player names, a game title, a target visual style, and optional character reference images. The Skill locks identity cues, generates an approval image from a fixed menu framework with coordinated color, buttons, icons, and typography, then uses the approved result to rebuild the character, UI-copy, and event timing instructions for the final video. It outputs a co-op game intro featuring two characters, player cards, and menu interaction motion. Best for game concepts, character-led menus, and social content; not for playable game development, complex multi-page UI, exact brand-logo replication, or generic character-free title sequences.
compatibility: Requires the MiniMax Hub agent (canvas workspace and MiniMax H3 generation); not portable to generic agent harnesses.
---

# Co-op Game Intro Generator

Use this Skill when the user wants a co-op game intro video and wants to confirm the visual direction with one image before generating the final H3 video. The workflow collects style, player names, game title, and optional character refs, then creates a framework-preserving confirmation image before generating the H3 video.

## Required References

These two templates are mandatory runtime inputs, not optional background notes:

- Use `references/h3-confirmation-image-template.md` when building the confirmation-image prompt in STEP 3 and generating the first confirmation image in STEP 4. Fill the template fields in order and do not skip framework, palette, UI, character, typography, layout, or negative-constraint fields.
- Use `references/h3-video-prompt-template.md` when refilling the final Minimax H3 video prompt in STEP 6. Fill the template from the approved confirmation image, player/game data, final UI copy, event timing, motion directions, and negative constraints.

If either template is unavailable, stop and report that the Skill package is incomplete instead of improvising a different prompt structure.

## STEP 1: Ask for visual style
Ask the user to choose a preset style or enter a custom style. This style has top priority and controls supplemental style language, palette language, background texture, character rendering, expression, outfit direction, UI colors, button/icon style, and typography texture.

## STEP 2: Collect player and game info
Collect PLAYER 1 name, PLAYER 2 name, and game title. If the user provides character images, use PLAYER 1 and PLAYER 2 refs only for identity mapping: recognizable face silhouette, hairstyle, glasses, relative facial proportions, and distinctive traits. Do not inherit photographic realism, skin texture, real-world lighting, camera quality, or the original image style; redesign the face into the selected visual style while preserving identity anchors.

## STEP 3: Build the GPT confirmation-image prompt
Load and follow `references/h3-confirmation-image-template.md` as the required prompt skeleton. Use a fixed framework + dynamic style fill + palette-linked prompt structure:

1. **Overall Style**: always preserve game main menu UI, high-quality game promo poster, deep UI-character integration, modern commercial game UI design, strong visual impact, clean composition, and avoid over-decoration. Add other style terms from the selected style.
2. **Color Palette**: always follow: xx as main color, xx as UI body color, xx as text color, xx as functional accent color, red as danger accent, palette within five colors, high-contrast color blocking, fresh modern look, and xx-style color language. All later UI/button/icon/type colors must match this palette.
3. **Composition**: preserve 16:9 landscape, full-frame background, x centered characters, UI around rather than blocking them, upper-left player card, right-side vertical menu, bottom caution tape, a few corner graffiti accents, Z reading path, enough negative space, clear hierarchy, and Continue as the visual focus.
4. **Background**: follow pure xx background, slight xx texture, large solid-color blank space, texture only as detail, avoid heavy dirt/dense scratches/too much ink splash/large paint splatter, a little black spray-paint edge detail, clean modern premium UI-first background. Derive xx and additions from the selected style.
5. **Character Style**: strictly expand according to the selected style. Use source text only as dimension guidance, not fixed style.
6. **Character A**: facial features from character image 1; expression, rendering, and outfit follow selected style; optional compatible outfit dimensions include khaki short jacket, black inner layer, black cargo pants, brown boots, adventurer outfit. Fixed action: cross-legged, hands on floor, slight backward lean, looking up.
7. **Character B**: facial features from character image 2; expression, rendering, and outfit follow selected style; optional compatible outfit dimensions include green thick jacket, burgundy padded lining, white inner shirt, black pants, thick boots. Fixed action: cross-legged, hands in front of legs, slight forward lean, looking at camera.
8. **Lighting**: preserve top main light, upper-left warm/cool fill, soft bottom ambient reflection, soft shadows, contact shadows, rim light, high-quality GI, and natural character-background integration. Choose warm/cool from style.
9. **Game UI**: preserve console game menu, unified button size, rounded rectangles, slight tilt, minimal spray/drip elements, modern clean sticker design, readability, and no over-decoration. Button body, outline, and glow colors must match the palette.
10. **Layout Rules**: preserve horizontal long buttons, unified width/height/radius, width adapted to text, single-line titles only, no wrapping, no two-line titles, centered text, consistent margins and spacing.
11. **Buttons**: adapt colors from palette and icons from selected style. Continue is the large visual-center highlighted button with hover/click state. Start New Game is above Continue. Settings remains visually consistent. Exit Game uses danger/exit cue with red accents allowed.
12. **Player Cards**: preserve upper-left x-player info cards, irregular rectangular card, outline, slight edge damage, left logo/icon, right three-level info (PLAYER label, nickname, READY), bold sans-serif, industrial sticker design. Card colors, outline, READY color, and logo style follow palette and selected style.
13. **Icon System**: derive icon style from selected style, colors from palette, and keep one-row-only, no wrapping, no stacking, at most one row per UI block, minimal quantity, unified size, never stealing focus.
14. **Typography**: preserve bold sans-serif, all caps, Anton/Impact/Burbank/Tungsten-like weight, tight tracking, heavy strokes, clear hierarchy, single-line typography, no wrapping, button width adapts to text, never two-line menu titles. Colors and texture follow palette and selected style.

## STEP 4: Generate the first confirmation image
Generate only one confirmation image from the filled `references/h3-confirmation-image-template.md`. Preserve the framework structure, while making the selected style visibly dominant.

## STEP 5: Wait for approval
Do not generate video until the user approves the image. If the user changes style, names, game title, identity, or image direction, return to the image prompt step.

## STEP 6: Refill video prompt and generate with Minimax H3
After approval, load `references/h3-video-prompt-template.md` and refill the final video prompt with confirmed style, character refs, player names, game title, UI text, event timing, motion directions, and negative constraints. Generate the final video with Minimax H3.

## STEP 7: Repair common failures
If text is unreadable, reduce on-screen text. If identities swap, strengthen names, positions, and colors. If faces drift, reuse uploaded refs and explicitly preserve identity anchors, hairstyle, and outfit anchors while keeping the face rendered in the selected visual style. If the selected style is weak, rewrite Overall Style, Color Palette, Character Style, Background, Game UI, Buttons, Icons, and Typography instead of changing layout framework.


---

## FILE: skills/co-op-game-intro-generator/meta.yaml (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/co-op-game-intro-generator/meta.yaml)

display-name-zh: 双人游戏开场视频生成器
version: 0.1.5
tag-en: Creative & Experimental
tag-cn: 创意实验
complete-tags-en:
- Creative & Experimental / Creative Generation
complete-tags-cn:
- 创意实验 / 创作生成
summary-en: Create a co-op game intro from player details, character references, and an approved menu style.
summary-cn: 根据玩家信息、角色参考图和视觉风格，完成首图确认与界面动效设计，输出双人游戏开场视频。
desc-en: For users creating a two-player co-op game menu or opening animation. Users provide two player names, a game title, a target visual style, and optional character reference images. The Skill locks identity cues, generates an approval image from a fixed menu framework with coordinated color, buttons, icons, and typography, then uses the approved result to rebuild the character, UI-copy, and event timing instructions for the final video. It outputs a co-op game intro featuring two characters, player cards, and menu interaction motion. Best for game concepts, character-led menus, and social content; not for playable game development, complex multi-page UI, exact brand-logo replication, or generic character-free title sequences.
desc-cn: 面向想制作双人合作游戏主菜单或开场动画的用户。用户需提供两位玩家名称、游戏名称、目标视觉风格，并可上传角色参考图。Skill 会先锁定人物身份特征，基于固定菜单布局生成与色彩、按钮、图标和字体联动的确认首图；用户确认后，回填角色、界面文案和事件节奏，生成完整视频提示词并制作最终开场视频。最终输出一条包含双人角色、玩家信息卡和主菜单交互动效的游戏开场视频。适用于合作游戏概念展示、角色化菜单和社交内容，不适用于完整可玩游戏开发、复杂多页面 UI、精确品牌标识复刻或无角色的通用片头。
author-en: MiniMax Hub
author-cn: MiniMax Hub
source: official


---

## FILE: skills/co-op-game-intro-generator/references/h3-confirmation-image-template.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/co-op-game-intro-generator/references/h3-confirmation-image-template.md)

# H3 Confirmation Image Template

## Core Rule
Generate a polished 16:9 landscape confirmation image for a cooperative game main menu.
The image consists of two independent systems:
• UI Framework (fixed)
• Visual Style (dynamic)
The UI Framework defines the layout, composition, information hierarchy, interaction logic, reading flow, menu structure, and readability. These elements must remain unchanged.
{visual_style} only controls the artistic appearance, including rendering medium, materials, color language, lighting mood, environment style, character design, costume language, UI materials, icon language, typography appearance, decorative motifs, and overall atmosphere.
Changing {visual_style} should feel like applying a different art direction to the same professionally designed game menu, never redesigning the interface itself.
If any default description conflicts with {visual_style}, prioritize {visual_style} while preserving the UI framework.

---

## Design Goal
Create a premium commercial console game main menu mockup in 16:9 landscape format. The image should combine the quality of a polished game promotional artwork with the clarity of a professional game UI. Characters and interface must feel naturally integrated, with strong visual hierarchy, excellent readability, clean composition, and a clear interaction focus.

---

## Overall Style
Follow the user-selected visual style: {visual_style}.
The selected style controls the rendering medium, illustration style, materials, palette language, environment, character appearance, outfit design, UI appearance, icon language, typography treatment, decorative elements, and atmosphere.
All visual elements must belong to one unified artistic language.
Do not mix incompatible visual styles within the same image.
The UI framework, layout, hierarchy, composition, and interaction logic must remain consistent regardless of style.

---

## Color Palette
Derive the entire color system from {visual_style}.
Use:
• {main_background_color} as the primary background color.
• {ui_body_color} as the main UI color.
• {text_color} as the primary text color.
• {functional_accent_color} as the interaction highlight color.
• Red only for warning, danger, or exit actions.
Limit the overall palette to five primary colors.
All buttons, player cards, icons, borders, highlights, decorative elements, typography, and visual effects must follow the same color system.
Avoid introducing unrelated colors outside the selected palette.

---

## Composition
Use a fixed 16:9 landscape layout.
Place {character_count} playable characters in the center as the visual subject.
The UI surrounds the characters without blocking them.
Place the player information cards in the upper-left corner.
Place the main menu vertically on the right.
Place a decorative horizontal strip along the bottom, adapting naturally to {visual_style} (such as ribbon, warning strip, wooden plank, stone slab, energy bar, scroll, etc.).
Keep a clear Z-shaped reading flow:
Player Cards → Characters → Menu → Continue Button.
The Continue button must always remain the primary visual focus.
Maintain generous negative space and excellent readability.

---

## Background
Generate a background naturally derived from {visual_style}.
The background supports the interface rather than competing with it.
Keep enough clean negative space behind all important UI elements.
Use subtle textures, environmental details, and decorative motifs only where appropriate for the selected style.
Avoid excessive visual noise, dense decoration, heavy dirt, scratches, paint splatter, or distracting textures.
The overall background should feel polished, premium, modern, and suitable for a commercial game menu.

---

## Character Style
All characters strictly follow {visual_style}.
The selected style determines:
• rendering medium
• body proportion
• facial stylization
• costume language
• material treatment
• lighting response
• animation style
• overall character quality
Only the character identity anchors and pose remain fixed. Facial rendering, eye design, nose simplification, mouth shape, skin treatment, expression style, and head proportion must be fully reinterpreted in {visual_style}.
No default character style may override {visual_style}.

---

## Character A / PLAYER 1
### Identity
Use character reference image 1 only for identity mapping.
The reference image donates identity anchors only: recognizable face silhouette, hairstyle, glasses if present, relative facial proportions, and distinctive personal traits.
It does not donate photographic realism, real skin texture, real-world lighting, camera quality, or the original image style.
Do not lose the character's identity anchors, but the face must be redesigned into {visual_style}.
All facial rendering, eye design, nose simplification, mouth shape, skin treatment, expression style, and head proportion must follow {visual_style}.
Do not swap identity with other characters.

### Appearance
Character rendering, expression, outfit, materials, and accessories are fully derived from {visual_style} while maintaining a recognizable playable-game-character silhouette.

### Action
Character A sits cross-legged on the ground.
Both hands support the body on the floor.
The upper body leans slightly backward.
Character A looks upward toward the camera.

---

## Character B / PLAYER 2
### Identity
Use character reference image 2 only for identity mapping.
The reference image donates identity anchors only: recognizable face silhouette, hairstyle, glasses if present, relative facial proportions, and distinctive personal traits.
It does not donate photographic realism, real skin texture, real-world lighting, camera quality, or the original image style.
Do not lose the character's identity anchors, but the face must be redesigned into {visual_style}.
All facial rendering, eye design, nose simplification, mouth shape, skin treatment, expression style, and head proportion must follow {visual_style}.
Do not swap identity with other characters.

### Appearance
Character rendering, expression, outfit, materials, and accessories are fully derived from {visual_style} while maintaining a recognizable playable-game-character silhouette.

### Action
Character B sits cross-legged on the ground.
Both hands rest naturally in front of the legs.
The upper body leans slightly forward.
Character B looks directly toward the camera.

---

## Lighting
Lighting follows {visual_style} while preserving clear readability.
Use a primary top light, secondary fill light, soft ambient bounce, contact shadows, gentle rim lighting, and high-quality global illumination where appropriate.
Faces should remain clearly visible.
Characters must integrate naturally into the environment while maintaining strong separation from the background.

---

## Game UI
Create a professional commercial console game main menu.
The menu structure, interaction hierarchy, spacing, composition, reading flow, and information architecture remain fixed.
{visual_style} only changes the artistic appearance of the interface.
The selected style may affect:
• UI materials
• surface textures
• border treatment
• corner shape
• decorative motifs
• color rendering
• lighting treatment
• icon language
• typography appearance
• interaction effects
It must NOT change:
• menu hierarchy
• button order
• button placement
• player-card placement
• composition
• interaction logic
• information architecture
• readability
The entire interface should feel like one professionally designed game UI system rather than separate elements assembled together.

---

## UI Visual Language
Derive the complete UI appearance from {visual_style}.
The selected style determines:
• button materials
• border language
• panel materials
• decorative details
• edge treatment
• visual weight
• highlight style
• interaction feedback
• shadow treatment
• texture language
Possible materials include wood, stone, paper, leather, cloth, glass, crystal, hologram, neon, metal, ceramic, carved surfaces, painted surfaces, magical energy, pixel blocks, watercolor paper, ink brush, comic graphics or any other material naturally belonging to {visual_style}.
Every UI element must share the same material language and artistic style.

---

## Button Layout Rules
Place all menu buttons vertically on the right side.
Maintain a consistent vertical rhythm.
Every button uses the same design family.
Button width automatically adapts to text length.
Keep all menu titles on a single line.
No automatic wrapping.
No stacked text.
Text remains horizontally centered.
Keep consistent padding.
Maintain equal spacing between buttons.
Avoid visual clutter.

---

## Continue Button
The Continue button is always the primary visual focus.
Make it visually dominant using the emphasis method naturally belonging to {visual_style}.
Possible emphasis methods include:
• larger size
• stronger contrast
• brighter color
• unique material
• distinctive border
• lighting
• glow
• engraving
• magical energy
• holographic effect
• embossed frame
• animated visual cue
Do not force glow or neon if they conflict with {visual_style}.
The Continue button should immediately attract the player's attention.

---

## Start New Game Button
Place above Continue.
Use the standard secondary button appearance.
Keep the visual language consistent with the selected style.
Add one small style-appropriate icon if suitable.
Maintain single-line text.

---

## Settings Button
Use the standard secondary button appearance.
Use one settings icon naturally derived from {visual_style}.
Maintain consistent proportions, spacing, and hierarchy.

---

## Exit Game Button
Use the standard secondary button appearance.
Use red only as the danger indicator.
Red should remain an accent rather than the dominant color.
Add one style-appropriate exit icon if suitable.

---

## Player Cards
Create an upper-left player information system for {character_count} players.
The appearance follows {visual_style}.
Card materials, borders, corners, textures, shadows, decorations, and icons are automatically derived from the selected style.
Each player card contains:
• Player label
• Player nickname
• READY status
Player names:
PLAYER 1
{player1_name}
PLAYER 2
{player2_name}
READY always uses the functional accent color.
The cards should remain highly readable and visually connected to the rest of the interface.

---

## Icon System
Derive all icons from {visual_style}.
Use one unified icon family throughout the entire interface.
Possible icon languages include:
• minimal outline
• solid filled
• geometric
• comic
• pixel
• engraved
• carved
• painted
• brush
• rune
• glyph
• holographic
• crystal
• paper-cut
• origami
• embossed
• low-poly
• flat design
• stylized 3D
All icons must share:
• the same artistic style
• the same material language
• the same perspective
• the same lighting treatment
• the same stroke language
• the same color hierarchy
Icons should remain simple, recognizable, and functional.
Each interface block may contain at most one horizontal row of icons.
No wrapping.
No vertical stacking.
Icons should support the interface rather than dominate it.

---

## Typography
Typography hierarchy remains fixed.
Typography appearance follows {visual_style}.
The selected style controls:
• font family
• decorative treatment
• texture
• material
• outline
• lighting
• shadow
• wear pattern
Maintain:
• excellent readability
• strong visual hierarchy
• single-line menu buttons
• consistent spacing
• high contrast
Button width should always adapt to text length.
Never allow menu titles to wrap onto multiple lines.

---

## Game Title
Display the game title:
{game_title}
The title should become the primary branding element of the menu.
Its material, decorative style, typography, and rendering all follow {visual_style}.
Avoid imitating existing commercial game logos.
Create an original title treatment inspired by the selected style.
Keep the title highly readable.
Avoid random symbols or meaningless letters.

---

## Interaction States
Represent the interface as an active game menu.
Buttons may display appropriate interaction feedback naturally belonging to {visual_style}.
Possible interaction feedback includes:
• hover
• pressed state
• magical glow
• energy pulse
• holographic highlight
• carved illumination
• ink spread
• brush bloom
• metallic reflection
• paper emboss
• animated border
• pixel flashing
Only use interaction effects that naturally belong to the selected style.
Avoid forcing modern UI effects into incompatible artistic styles.

---

## Rendering
Generate a polished AAA-quality commercial game menu artwork.
Characters, environment, UI, typography, icons, and decorative elements should feel like one cohesive production.
Maintain premium rendering quality.
Strong visual hierarchy.
Professional commercial presentation.
High readability.
Balanced composition.
Clean finish.

---

## Negative Constraints
No split screen.
No duplicated characters.
No identity swap.
No loss of identity anchors. No photorealistic face unless {visual_style} explicitly requires photorealism. No realistic photo face pasted onto a stylized body. No mismatched face style.
No hairstyle changes.
No unreadable UI.
No distorted buttons.
No overlapping interface.
No random text.
No random symbols.
No watermark.
No copied game logo.
No branded interface.
No inconsistent icon styles.
No mixed UI materials.
No excessive decoration.
No overwhelming background.
No visual clutter.
No broken layout.
No low readability.

---

## Medium Lock
Render as a polished commercial game key art combined with a professional console game main menu mockup.
The interface remains fixed.
Only the artistic appearance changes according to {visual_style}.


---

## FILE: skills/co-op-game-intro-generator/references/h3-video-prompt-template.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/co-op-game-intro-generator/references/h3-video-prompt-template.md)

# H3 Video Prompt Template

## Prompt principle
Use the same method as the GPT confirmation-image prompt:

**Fixed video event framework + dynamic user-style fill + locked character identity + palette-linked UI system.**

The video prompt must not blindly preserve the source prompt's default style. It must preserve the timeline, UI events, player positions, equipment logic, and text structure, while dynamically rewriting visual treatment, palette, character rendering, lighting, UI surface, icons, and city-world style according to the user's selected style.

## Priority order
1. User-selected style: {visual_style}
2. Confirmed image / UI reference: {ui_ref}
3. PLAYER 1 identity reference: {player1_ref}
4. PLAYER 2 identity reference: {player2_ref}
5. Height/body comparison reference: {height_ref}
6. Fixed source video event framework

## Reference roles
- {ui_ref}: confirmed first image. Use it to lock UI layout, menu hierarchy, color system, typography scale, button structure, character-game integration, and overall composition logic.
- {player1_ref}: PLAYER 1 identity anchor. Lock exact face, hairstyle, glasses if present, facial proportions, body identity, and nickname mapping to {player1_name}.
- {player2_ref}: PLAYER 2 identity anchor. Lock exact face, hairstyle, facial proportions, body identity, and nickname mapping to {player2_name}.
- {height_ref}: body comparison anchor. Lock the visible contrast between the two players and prevent identical body proportions.

## Global style baseline
整体画风必须优先服从用户选择：{visual_style}。
保留固定基准：游戏主菜单 UI、高品质游戏宣传片质感、UI 与角色深度融合、现代商业游戏 UI 设计、极强视觉冲击力、画面简洁干净、避免过度装饰。
其余视觉层面根据 {visual_style} 动态拓展：角色画风、表情气质、服装语言、色彩系统、灯光冷暖、UI材质、按钮图标、字体质感、城市加载后的世界风格。

## Palette system
根据 {visual_style} 推导视频全程色彩，但必须保持 UI 色彩联动：
- xx 色作为主体背景 / 世界主色
- xx 色作为 UI 主体颜色
- xx 色作为文字颜色
- xx 色作为功能强调色
- 红色作为危险/退出/警示提示色
- 整体颜色控制在 5 种以内
- 高对比撞色、鲜明现代、符合 {visual_style} 的色彩语言

视频中的菜单、装备面板、玩家卡、按钮、HUD、加载条、图标和文字都必须沿用同一色彩系统，不得随机新增无关颜色。

## Character identity and style lock
PLAYER 1：五官信息参考 {player1_ref}，必须保留脸、人脸比例、发型、眼镜（如有）、个人身份和 {player1_name} 昵称对应关系。表情、画风、服装和机械装备的视觉处理根据 {visual_style} 动态优化。PLAYER 1 始终位于左侧，偏高挑/修长/敏捷，装备色为功能强调色，机械爪轻量、纤细、灵活。

PLAYER 2：五官信息参考 {player2_ref}，必须保留脸、人脸比例、发型、个人身份和 {player2_name} 昵称对应关系。表情、画风、服装和机械装备的视觉处理根据 {visual_style} 动态优化。PLAYER 2 始终位于右侧，偏矮壮/宽厚/力量型，装备色为琥珀红或与危险/力量提示一致的暖色，机械拳厚重、宽大、有重量。

禁止角色身份交换、脸部互相融合、昵称交换、两人体型趋同。

## Fixed timeline framework

### [0秒–2秒] — 双人主菜单
景别/机位：高角度俯拍大全景，延续 {ui_ref} 的构图逻辑。镜头从上方轻微下压并缓慢推进。

画面内容：PLAYER 1（{player1_name}）和 PLAYER 2（{player2_name}）并排坐在画面中央，PLAYER 1 在左，PLAYER 2 在右，抬头看向摄像机。两人只有自然呼吸、眨眼和轻微身体动作。

UI结构：左上角玩家资料卡准确显示：
“PLAYER 1”
“{player1_name}”
“READY”
顶部中央或左上双卡系统中的第二张资料卡准确显示：
“PLAYER 2”
“{player2_name}”
“READY”
右侧纵向菜单准确显示：
“START NEW GAME”
“CONTINUE”
“SETTINGS”
“EXIT GAME”
“CONTINUE”是视觉中心和主高亮按钮。

动态风格填充：主菜单背景、地面纹理、按钮形态、图标、字体、边框、辉光和贴纸质感全部根据 {visual_style} 优化，但保留 {ui_ref} 的布局和层级。

声音：菜单环境音、轻微 UI hover 声、点击前的低频电子氛围。若生成无音频，则忽略声音执行。

### [2秒–4秒] — PLAYER 1 右臂配置
景别/机位：中景，镜头从主菜单平滑推近 PLAYER 1 的右臂，PLAYER 2 仍在背景中可见，不消失、不变形。

UI结构：右侧主菜单收缩滑出；带功能强调色识别线的 UI 面板从左侧滑入，准确显示：“PLAYER 1”“RIGHT ARM EQUIPMENT”。装备列表中先高亮“PHANTOM GRIP”，随后选区移动至“CHRONOS CLAW”。

动作：PLAYER 1 的右袖口自动打开，轻量机械结构从前臂下方展开。手指分开，修长爪状指节滑入并逐一锁定，内部短暂露出精细线路、微型活塞和金属连接件。配置完成后，功能强调色 LED 依次亮起。

动态风格填充：机械结构、UI面板材质、图标、线条、锁定动画和光效根据 {visual_style} 优化，但必须轻量、精密、灵活，符合 PLAYER 1 身形，不改变脸、发型和服装主体。

声音：精密机械展开声、轻快 UI 切换音、细小锁定声。

### [4秒–7秒] — PLAYER 2 重型手臂配置
景别/机位：中景，摄影机沿两人之间平滑横移并绕向 PLAYER 2 左侧。PLAYER 1 留在背景中，轻轻观察自己完成配置的机械手。

UI结构：新的暖色/琥珀红识别 UI 滑入，准确显示：“PLAYER 2”“ARMAMENT CUSTOMIZATION”。面板以网格形式展示：
“HAND”
“FOREARM”
“ELBOW”
“UPPER ARM”
选区快速但清晰地在四个组件之间切换。

动作：PLAYER 2 左臂外套袖口分段打开，厚重前臂护板向外弹开，旧组件脱离，新型装甲沿导轨滑入；肘关节替换为厚实机械轴承，宽大的机械手重新组合并锁定。更换过程中短暂露出粗壮线路、液压活塞和深色金属骨架。每个部件锁定时亮起低调暖色指示灯。

动态风格填充：重型机械臂、UI面板、组件图标、材质、光效根据 {visual_style} 优化，但必须厚重、宽大、有力量感，与 PLAYER 1 的轻量机械爪形成清晰对比。

声音：低沉电机声、重型机械扣合声、厚重锁定反馈。

### [7秒–8.5秒] — 双人确认配置
景别/机位：中景拉回，镜头平滑回到双人构图，PLAYER 1 左侧，PLAYER 2 右侧。

UI结构：两组装备面板向画面中央汇合，形成共享按钮，准确显示：“CONFIRM CONFIG”。按钮边框、辉光、图标和贴纸质感根据 {visual_style} 优化，但层级清晰、文字可读。

动作：光标点击按钮，功能强调色能量脉冲流过 PLAYER 1 的机械爪，暖色/琥珀红能量脉冲流过 PLAYER 2 的机械拳。所有 UI 面板快速向内收缩并消失。两人同时解开交叉的双腿并调整坐姿：PLAYER 1 轻盈抬起单膝，修长机械爪依次活动手指；PLAYER 2 一只脚稳稳踩地，厚重机械拳缓慢握紧。

声音：确认提示音、双色能量脉冲、UI 收缩声。

### [8.5秒–10秒] — 双人世界加载
景别/机位：全景，底部共享加载条出现。

UI结构：加载条准确显示：“LOADING”。进度从 0% 快速填充至 100%。左半段使用 PLAYER 1 的功能强调色，右半段使用 PLAYER 2 的暖色/力量色。HUD 和加载条的形态、边框、纹理、字体根据 {visual_style} 优化，但必须清晰可读。

环境转化：黄色平面环境连续转化为游戏世界。警戒线变成真实街道道路标记和施工围挡；墨迹/纹理化为潮湿路面反光或符合 {visual_style} 的地面光影；平面涂鸦扩展成建筑墙面的喷绘；黑色背景区域形成城市阴影和深巷入口。

关键约束：转化必须连续自然，不使用硬切，不用烟雾遮挡，不改变两位角色身份。

声音：加载上升音、环境从菜单氛围过渡到城市氛围。

### [10秒–15秒] — 双人进入游戏世界
景别/机位：大全景转第三人称跟拍。加载 100% 的瞬间，两名角色同时起身。摄影机平滑下降并绕到两人身后，变成稳定第三人称双人合作视角。

世界风格：完整游戏世界根据 {visual_style} 动态生成。保留游戏开场进入世界的结构：密集建筑、道路、标志灯、招牌、人群或动态背景、快速经过的载具、电线、工业管道、远处城市天际线。具体世界材质、建筑形态、招牌设计、光影和动效必须服从 {visual_style}，不要让源默认赛博朋克风格压过用户选择，除非用户选择的是赛博朋克。

角色关系：清楚展示两人的背影和体型差异：PLAYER 1 左侧，高挑修长，轻量机械爪自然垂下；PLAYER 2 右侧，矮壮宽厚，重型机械拳微微抬起。PLAYER 1 率先迈步，PLAYER 2 紧随其后，两人并肩进入街道。

HUD结构：HUD 淡入。右上角出现小地图；左下角出现两组独立状态栏：“{player1_name}”“{player2_name}”。{player1_name} 状态栏使用 PLAYER 1 功能强调色，{player2_name} 状态栏使用 PLAYER 2 暖色/力量色。前方街道中央出现共享任务标记。

声音：城市环境氛围、远处载具声、脚步声、HUD 淡入提示音。

## Negative constraints
No third player, no female character unless explicitly requested by user, no duplicated character, no character swapping, no username swapping, no merged bodies, no identical body proportions, no changing faces, no changing hairstyles, no changing identity, no missing PLAYER 1, no missing PLAYER 2, no split screen, no hard cuts, no random camera shake, no floating body parts, no gruesome dismemberment, no weapon, no excessive neon unless selected by user style, no purple background unless selected by user style, no unreadable UI, no random letters, no misspelled usernames, no extra menu options, no official game logo, no copied branded interface, no watermark.


---

## FILE: skills/handdrawn-live-video-generator/SKILL.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/handdrawn-live-video-generator/SKILL.md)

---
name: handdrawn-live-video-generator
description: |
  For creators making surreal short videos that blend rough glowing hand-drawn animation with live-action spaces. Users provide a scene idea, contact object or hand, desired mood, and optional language or style constraints. The Skill clarifies the physical contact, designs continuous morphing, escape route, and delayed handheld chase movement, then writes a reusable 15-second 16:9 video prompt in the user's language. After user confirmation it recommends MiniMax H3 generation and checks contact realism, camera delay, rough glowing stroke texture, and non-horror tone. Best for single-scene creative clips, not polished CG, horror jump scares, plush characters, or multi-scene cuts.
compatibility: Requires the MiniMax Hub agent (canvas workspace and MiniMax H3 generation); not portable to generic agent harnesses.
trigger-words: [手绘发光动画实拍融合, 15秒变形追逐视频提示词, Seedance视频prompt, H3生成视频, 视频prompt, 手绘动画接触真实物体, 蜡笔粉笔质感, 多语言视频提示词]
---

# Handdrawn Live-Action Fusion Video Generator

Use this Skill when the user wants a **finished 15-second, 16:9 live-action-and-hand-drawn fusion video**. The output must preserve the structure: a flat hand-drawn luminous animation appears in a real space, clearly contacts live-action hands or objects in the first 0-3 seconds, continuously morphs as one single entity, escapes, and a handheld phone camera follows slightly late.

This Skill **organizes the prompt in the user input language and recommends MiniMax H3 as the confirmed generation step**. Do not generate video until the user explicitly confirms H3 generation. Do not route to planner or executor for prompt-writing. The final prompt language must follow the dominant language of the user input: Chinese input produces Chinese, English input produces English, Japanese input produces Japanese; for mixed input, use the dominant language; when unclear, use the current conversation language. Only user-required proper nouns, model names, or literal parameters may remain unchanged.

## Step 1: Understand the user intent as same-language constraints

When the user provides any language, understand it as the following workflow requirements and express the final prompt in the dominant language of the user input:

- 基于参考 prompt 创作一个全新的 15 秒视频生成 prompt，不要表面模仿原句。
- 必须保留影像结构：实拍空间中出现平面的手绘发光动画；动画与真实手或真实物体接触；同一个存在连续变形并逃跑；相机总是慢半拍追赶。
- 手绘动画质感必须像蜡笔、粉笔、彩色铅笔、粉彩、粗糙笔刷；线条轻微抖动，有涂抹不均、毛边和逐帧重画感。
- 禁止 3DCG、毛绒玩具感、均匀矢量线、平滑霓虹、恐怖怪物、巨大眼睛、裂口、牙齿、威吓、扑咬、突然黑屏、跳吓。
- 0-3 秒必须出现实拍手与手绘动画的清晰接触，例如缠住手指、落在掌心、被抓时逃跑、从指尖诞生。
- 动画必须作为同一个实体连续变形，可在“线条、生物、记号、植物、交通工具、生活小物”等形态之间变化，并保留前一形态的痕迹。
- 不允许突然出现另一个全新角色。
- 全片在同一空间或相邻范围内连续展开，不用剪辑跳到另一个地点，像拍摄者真的边走边追。
- 每个区间 0-3、3-6、6-10、10-13、13-15 秒都要有新的变形、移动、接触、发现、恶作剧或惊喜。
- 拍摄者也必须参与：伸手、抓、追、打开门或盒子、接住、后退、被恶作剧等。
- 调性是可爱、生活感、怀旧、温柔、略带切感，不是恐怖喜剧。
- 13-15 秒必须有空间级变形：之前的线扩散到墙、地板、天花板、窗户、水槽或通道，变成巨大花、星空、夕阳、云、丝带、涂鸦小镇等；结尾要有感动余韵和一点可爱笑点。

## Step 2: Invent all creative content fresh

For every run, create a new combination. Do not reuse the reference prompt's dark room, PC, fridge, stars, hearts, blue vortex, butterfly, snake, octopus, hamburger, giant eye, teeth, or dark ending.

Choose new values for all of these:

1. 实拍空间，但仍是生活化且可连续追踪的相邻范围；
2. 手绘实体或初始形态；
3. 中心色彩；
4. 连续变形链路；
5. 与真实手或实物的接触方式；
6. 在空间中的追逐路线；
7. 拍摄者反应；
8. 最后的空间级变形与可爱笑点。

Good example directions include: 雨天厨房水槽、旧阳台晾衣角、清晨玄关、小书店走廊、火车窗边小桌、浴室镜柜、手作桌、温室通道、自助洗衣店长椅、旧餐桌. Use only one or adjacent connected areas.

## Step 3: Required output format

The final prompt must begin with a sentence pattern matching the dominant language of the user input. For Chinese input, use this pattern:

`15秒，16:9横版视频。将实拍的〇〇与手绘发光动画融合的影像。`

Replace “〇〇” with the newly invented live-action space or everyday scene. For non-Chinese input, use an equivalent opening in the same language and include duration, 16:9 landscape format, live-action space, and hand-drawn glowing animation fusion.

Do not add titles, auxiliary headings such as 舞台设置 or 色味, or explanatory notes. Keep this paragraph order:

1. 开头句；
2. 实拍空间与手机拍摄质感；
3. 0-3 秒；
4. 3-6 秒；
5. 6-10 秒；
6. 10-13 秒；
7. 13-15 秒；
8. 手绘质感；
9. 相机追随方式；
10. 禁止事项；
11. 环境音。

## Step 4: Prompt writing rules

- Keep the prompt executable as a video-generation prompt, not an essay.
- Unless the user asks for explanation, the final answer should contain the prompt text in the dominant language of the user input first, followed by one short next-step recommendation in the same language.
- The recommendation must invite the user, in the same language, to use MiniMax H3 to generate a 15-second 16:9 video from this prompt. Chinese example: `下一步建议：如果你确认这个 prompt，我可以继续用 H3 模型生成 15 秒 16:9 视频。`
- Do not generate a video, image, audio, storyboard, or intermediate asset until the user explicitly confirms the H3 generation step.
- Mention Seedance only as a target usage context when the user asks; do not add model parameters inside the creative prompt unless requested.
- The first 0-3 seconds must make real/hand-drawn fusion obvious through contact.
- The camera must not neatly center the animation. It should lag behind, pan/tilt/advance after the entity already leaves the frame edge.
- The entity should remain traceable: each new form preserves a line, tail, color smear, body curve, or motif from the previous form.
- Use soft, emotional, comic, cute beats: tiny stumbles, shy gestures, a dot falling behind, a petal sticking to lens, a star dropping into a palm, confetti sneeze, one small creature lagging behind.
- Avoid any horror-coded anatomy or threatening motion.
- The final prompt must not randomly mix in vocabulary, transition sentences, or writing systems from outside the user input language; when the user input language is Japanese, natural Japanese is allowed.

## Step 5: Delivery

Output the final prompt in the dominant language of the user input first. After the prompt, add exactly one short same-language recommendation line inviting the user to continue with MiniMax H3 video generation. Do not include a title, checklist, model call summary, filename, or canvas delivery note. If the user later confirms H3 generation, generate the 15-second 16:9 video with MiniMax H3 and check that the result preserves contact, continuous morphing, delayed camera chase, and non-horror hand-drawn texture.


---

## FILE: skills/handdrawn-live-video-generator/meta.yaml (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/handdrawn-live-video-generator/meta.yaml)

display-name-zh: 手绘实拍融合视频生成器
version: 1.0.2
tag-en: "Creative & Experimental"
tag-cn: "创意实验"
complete-tags-en:
  - "Creative & Experimental / Creative Generation"
  - "Animation / Creative Generation"
complete-tags-cn:
  - "创意实验 / 创作生成"
  - "动画 / 创作生成"
summary-en: "Create H3-ready prompts for hand-drawn animation interacting with live-action scenes."
summary-cn: "基于场景创意，生成手绘动画与实拍接触变形的15秒视频，适用于创意短片制作。"
desc-en: "For creators making surreal short videos that blend rough glowing hand-drawn animation with real-world footage. Users provide a scene idea, contact object or hand, desired mood, and optional language or style constraints. The Skill clarifies the interaction, designs continuous morphing and chase action, writes a reusable 15-second 16:9 video prompt in the user's language, recommends MiniMax H3 after confirmation, and checks contact realism, camera delay, texture, and non-horror tone. Best for single-scene creative clips, not polished CG, horror jump scares, plush characters, or multi-scene editing."
desc-cn: "面向希望制作手绘动画与实拍空间融合短片的创作者。用户提供场景创意、接触对象或手部动作、情绪氛围和可选风格限制。Skill 会明确触碰关系，设计连续变形、逃跑路线和慢半拍手持追拍，整理为用户输入语言的15秒16:9视频提示词；确认后建议使用 MiniMax H3 生成，并检查接触真实感、镜头延迟、笔触质感和非恐怖调性。适合单场景创意视频，不适合精致CG、恐怖跳吓、毛绒角色或多场景剪辑。"
author-en: MiniMax Hub User
author-cn: MiniMax Hub 用户
source: official-featured


---

## FILE: skills/minimalist-product-ad-generator/SKILL.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/minimalist-product-ad-generator/SKILL.md)

---
name: minimalist-product-ad-generator
description: |
  Turn product images and ad requirements into minimalist product ad shorts for e-commerce promotion and product launches. The Skill confirms format and product variants, extracts selling points, writes concise English ad copy, builds product anchors, plans beat-synced typography/storyboards, and generates a clean product film with premium camera language. Not for KOC talking-head ads, general editing, or complex screen demos.
compatibility: Requires the MiniMax Hub agent (canvas workspace and MiniMax H3 generation); not portable to generic agent harnesses.
metadata:
  trigger-words: [minimalist product ad, premium product ad, minimalist product film, 极简产品广告, 高质感产品广告, 产品广告片, 电商产品视频, 新品发布广告]
---

# Minimalist Product Ad Generator

Use this Skill to guide non-professional users through a flow-style workflow that creates a minimalist product advertising video for e-commerce promotion and product launches. The target users are e-commerce sellers, small brand owners, indie creators, and individual sellers; the user should provide at least one product image or related asset.

The core principle is: **confirm assets and brief first, build product facts and a product narrative spine, lock product visuals with independent anchor photos, control video with a precise beat storyboard, then finish the film with native audio or music-based editing.** This Skill no longer defaults to a 4-panel anchor sheet, because video models may reproduce the panel layout. By default, it uses three separate anchor photos plus one precise beat text storyboard as the video control system.

## Start Gate

Every time this Skill is triggered, run a complete but lightweight start gate before any analysis, copywriting, anchor generation, or video generation. Confirm only the information required for production. Do not ask about music here; music is handled later in the music step.

The start gate must confirm all of the following in one pass:

1. **Product images and related materials**
   - Ask the user to upload product images and related materials, such as product photos, packaging images, renders, detail shots, existing video assets, or visual references.
   - If the current message or current canvas / session already contains materials, state that materials have been detected. Do not ask the user to upload again; only confirm whether to use the current materials.
   - If there are multiple candidate materials and their roles are unclear, ask which one should be the main product image.

2. **Product variant status**
   - Confirm whether this is a single variant or multiple variants / colors / models.
   - If there are multiple variants, confirm which one is the main variant.
   - If the user does not specify, the agent may recommend one based on the product image, but the later storyboard must still write the concrete color or variant name.

3. **Target duration**
   - 5s / 10s / 15s, with 10s as the recommended default.

4. **Aspect ratio**
   - 16:9 / 9:16 / 1:1 / 4:3 / 3:4 / match reference image.
   - Once selected, the anchor sheet, video, and final delivery must strictly keep this ratio.

5. **Apple-style template**
   - White-tech / dark rim light / brand color field / light lifestyle scene.

6. **In-frame copy**
   - User-provided copy, or agent-generated Apple-style English copy.

7. **Video model strategy**
   - Default model is MiniMax-H3, because it often produces cleaner Apple-like product close-ups, negative space, and premium product-film camera language.
   - Do not ask the user to choose a video model during the start gate. Show “default model: MiniMax-H3” as part of the adopted start-gate result.
   - If the user explicitly asks for another model, follow the user's model choice after capability checking.
   - Seedance is only a backup when the user explicitly wants stronger multi-shot execution, asks for Seedance, or accepts a fallback after H3 fails or cannot satisfy the requested structure.

Start gate is mandatory. If the user has already uploaded product material, only skip the upload request; do **not** skip the rest of the start gate. The agent must still confirm product variant status, target duration, aspect ratio, Apple-style template, and in-frame copy mode in one pass before analysis, anchor generation, storyboard, or video generation. The video model is not a start-gate question: use MiniMax-H3 by default and display it in the completed start-gate result. If the user says “you decide / just do it,” the agent may choose recommended defaults for the remaining fields, but must still display the adopted choices as the completed start-gate result before continuing.

## Operating Principles

1. **Independent anchor photos are the default visual control system**
   - Generate three separate anchor photos by default, not one 4-panel sheet, because video models may copy panel layouts into the final film.
   - The three anchor photos usually cover: hero / striking main view, material or functional detail, and final copy composition.
   - Do not create a separate structure-action anchor when it overlaps with the hero view. Merge structure / action cues into the material-detail anchor unless the product truly needs a distinct mechanism reference.
   - Each anchor photo must be a complete standalone product photo. Do not create grid layouts, split screens, collage boards, framed panels, product walls, or storyboard sheets.
   - The precise beat storyboard determines the actual shot order, action, and text behavior. Do not use anchor photos as direct shot replacements.

2. **In-frame copy must appear as integrated video motion, not a subtitle fallback**
   - Text in the copy anchor photo is only a reference for font, position, single-line layout, and two-part color treatment. It does not mean the whole video can use only that one copy line.
   - The precise beat storyboard may define multiple short English copy lines across shots. Each line must be 3-5 English words, single-line, readable, and follow the two-part color rule: in white-tech style the first half uses black or dark gray, in dark rim-light style the first half may use white, and the second half always uses the specific product color. Do not use isolated 1-2 word feature labels.
   - Video generation must include the storyboard copy inside the frame.
   - The copy does not need to appear in every shot. If the video model tends to make typography mistakes, reduce copy frequency proactively. For a 10-second film, usually keep only one mid-film copy moment plus one final copy moment; do not put text in every shot. The final shot must include one stable single-line copy.
   - By default, ask the video model to generate in-frame typography and its timing directly. Do not silently degrade text into post-production subtitles or reserve empty space for later overlay. Use post-production fallback only when the user explicitly asks for it or when integrated video typography fails and the user accepts a fallback.

3. **Product body color is a hard fidelity constraint**
   - Apple-style means clean composition, premium light, restrained motion, and negative space; it does **not** mean turning the product itself white, silver, or AirPods-like.
   - The product's original body color, accent color, material tint, and visible finish from the user's image must be preserved in all anchor photos and videos.
   - Only the background, lighting environment, and composition may shift toward the selected Apple-style template. Never recolor the product body unless the user explicitly asks.
   - The final copy emphasis color must come from the real product main color, not from a generic silver / white Apple palette.

4. **Every product needs a product-specific narrative**
   - Do not reuse a fixed earbud template or write vague phrases like “product state changes.”
   - The storyboard must be based on product category, form, material, structure, and visible actions.
   - Product motion must be visual and concrete, such as “the lid opens 30 degrees and inner reflection appears,” or “the crown rotates while a highlight slides along the edge.”

4. **Progress must be visible**
   - After each step, show what was produced and what decision is needed next.
   - Do not auto-run the entire pipeline without user confirmation.

## Typography Rules

- In-frame advertising copy must be English.
- If the user provides Chinese copy, translate it into concise English or ask for approval of the English version.
- Visible in-frame copy must be 3-5 English words, preferably no more than 32 English characters including spaces. Do not write isolated 1-2 word feature labels.
- Copy should feel like an Apple product film: sensory, benefit-led, material-led, or a light value proposition; avoid promotional slogans and ecommerce feature-tag wording.
- Font reference: SF Pro Display / SF Pro Text. Prefer `SF Pro Display Semibold` in prompts.
- Use no more than two text colors in a shot.
- In white-tech style, the first half of the text must use black or dark gray; white text is forbidden on white backgrounds. In dark rim-light style, the first half may use white. The second half always uses the specific product color.
- Text must stay on one line. Do not wrap, stack, or split into multiple lines.
- At any moment, only one single-line English copy line may appear in the frame. Do not show two rows of text, two-line titles, title + subtitle pairs, or multiple text blocks at the same time. Two-part typography must continue or replace within the same line, never split into upper and lower rows.
- Do not place text in a lower subtitle position. Prefer the vertically centered visual zone, left-aligned or right-aligned as part of the composition. Text should be slightly larger and feel like a product-film visual element, not an explanatory subtitle.
- Approved mid-film pattern: clean negative space, product motion or product close-up leads the frame, the frame stays uncrowded, and only one single-line English copy appears. Text may sit near the product edge, product surface, or close-up highlight area so it feels integrated into the product image rather than floating like a subtitle. Do not damage negative space or add a second text block just to add copy.
- Default two-part text motion must remain visible: the first half fades in or slides in subtly first, then the second half fades in or slides in subtly later. When the second half appears, the first half shifts gently within the same line to make room.
- The shift should be subtle, around 10-15px or 8-12% of text width. Motion should be smooth and restrained; avoid bouncy, neon, or flashy UI effects.
- Do not make typography completely static just to avoid two-row text. The correct solution is same-line two-part entrance, subtle shift, fade-in, or gentle slide-in; the wrong solution is upper/lower line splitting, two-line titles, or removing text motion.

## STEP 1: Asset Check and Product Fact Summary

If no product material is available, ask the user to upload it. If material is already available, use it and analyze:

- Product category.
- Subject position and frame occupancy.
- Image quality: sharpness, lighting, composition, and visible details.
- Product main color and possible main variant.
- Showable structure: opening, rotating, lifting, snapping, folding, popping, glowing, screen, texture, etc.
- Real appearance features usable for the storyboard: material, edge, button, port, packaging, transparent part, screen, or distinctive silhouette.

If the image quality is not usable, stop and give concrete reshoot advice.

Output a short product fact summary:

- Material used.
- Product category.
- Main color candidates and main variant suggestion.
- Quality pass / fail.
- Showable structure / feature.

## STEP 2: Confirm the Production Brief

Turn the start gate answers into an executable production brief. Do not ask again for parameters that have already been confirmed. The brief drives the narrative spine, copy, anchor sheet, and storyboard.

The production brief must include:

- Product material used.
- Main variant / product main color.
- Aspect ratio.
- Target duration.
- Selected Apple-style template.
- Single-variant / multi-variant strategy.
- In-frame copy mode: user-provided copy, or agent-generated Apple-style English copy.

Multi-variant product rules:

- The user selects a style; do not automatically remove all other variants.
- Every multi-variant project must decide one main variant first.
- Other variants should not all appear at the very beginning; they appear only in process beats, transitions, or the final full set.
- If the user does not specify a main variant, the agent may choose the most visually stable one and use the others for rhythm and layering.
- Avoid ecommerce matrices, nine-grid displays, stacking, scattered layouts, all colors laid flat, or full-screen product walls.

## STEP 3: Choose a Product Narrative Spine

Before copywriting and anchor generation, choose a lightweight product narrative spine. Do not use a broad corporate brand-film structure; this Skill focuses on short Apple-style films for physical products.

If the user has not selected a direction, offer 2-3 concise directions, recommend one, and continue after confirmation. If the user says “you decide,” use the recommended one.

Recommended spines:

1. **Product Launch** (default recommendation)
   - Negative-space opening → main variant establishes hero view → material / structure detail → natural product action → variants or color relation → full-copy closing.
   - Good for earbuds, watches, digital accessories, small appliances, fragrance devices, etc.

2. **Feature Touch**
   - Product stillness → interaction trigger → feature action → detail magnification → result / feeling → closing frame.
   - Good for products with clear opening, rotation, magnetic snap, light change, screen, folding, mist, or similar actions.

3. **Color Family**
   - Main variant appears alone → supporting variants slide in lightly → color order forms → material and structure stay unified → full set closes → copy lands.
   - Good for multi-color, multi-variant, or multi-model products.

Output one sentence describing the selected narrative spine, for example:

> This film uses the Color Family spine: the purple main variant establishes the hero view, other colors enter as supporting layers, and the full product set closes with the English copy.

## STEP 4: Direct the Motion Language

Before copywriting and anchor generation, define the motion language of the film. This step does not generate assets; it defines motion intensity, transition logic, and rhythm peaks for the later storyboard.

Rules:

1. **Transitions are driven by real product or visual elements**
   - Prefer product edges, material highlights, opening / rotation / snapping / sliding actions, geometric changes in variant arrangement, color entry and exit, matched camera direction, or matched product silhouette.
   - Do not use meaningless white flashes, abstract particles, random light effects, or random cuts to fake premium motion.

2. **One main action per beat**
   - Secondary elements should appear with slight delay and should not compete for attention at the same time.
   - Examples: product enters first, then text appears; main variant stabilizes first, then supporting variants slide in; material highlight sweeps first, then text motion begins.

3. **Set strong and quiet moments**
   - 5s film: 1 small peak + 1 stable closing.
   - 10s film: 1-2 peaks + 1-2 braking moments.
   - 15s film: 2-3 peaks + 2 quiet braking moments.
   - Peaks may be product action completion, variant entrance, second-half copy reveal, or full-product closing.
   - Braking moments may be material-detail pause, readable copy hold, or final frame hold.

4. **Keep safe space clear**
   - Product silhouette remains clear, single-line copy stays readable, and the main variant is not blocked.
   - Add a logo only when the user provides one. Do not generate a fake logo.

5. **Avoid fake tech decoration, mirrored white stages, and empty openings**
   - Avoid fake technology interfaces, meaningless glass cards, decorative text walls, unconfirmed metrics, random particles and stacked light effects, one identical easing style across the whole film, and full-screen product walls.
   - White-tech style is not a mirrored white stage and not dead flat white lighting. Do not place the product on reflective floors, glass tables, or cheap studio mirror surfaces.
   - Apple-style white space should be a simple white background; impact comes from striking product angles, real product actions, camera rhythm, and negative-space composition, not from mirror reflections or waiting on empty frames.
   - The opening must not be empty dead time. It should quickly reveal an attractive product action or angle, such as the product rotating smoothly out of its own structure, emerging from an opening mechanism, sliding along its silhouette, or being revealed by an edge highlight. The exact action must be inferred from the current product form, not hardcoded to earbuds.

Output one short “motion language” statement, for example:

> This film uses product-edge highlights and the purple main variant sliding motion to drive transitions. The main peaks are variant entrance and full-copy landing, with a stable final hold for the product and copy.

## STEP 5: Generate or Confirm Apple-style English Copy

Before generating the text anchor and storyboard, obtain a final English copy line for the frame.

Rules:

1. Analyze product category, usage feeling, and narrative spine.
2. Generate 2-3 Apple-style English options, each 3-5 words.
3. Do not use fixed templates or fixed slogans; infer new copy for every product.
4. Avoid promotional language.
5. If the user already has copy, prioritize it. Chinese copy must be localized into concise English.

After the user confirms the copy, proceed to anchor generation. If the user says “you decide,” choose the recommended copy and continue.

## STEP 6: Generate Three Independent Product Anchor Photos

Generate **three independent anchor photos** in the same aspect ratio and resolution as the user-selected video setting. They must be three separate image outputs, not one combined 3-panel or 4-panel sheet. Background follows the selected style, and all three photos must share unified light, shadow, grade, and background language while remaining standalone product photos.

Anchor photo roles:

- Photo 1: Hero / striking main-view anchor, locking the most attractive product view, silhouette, and opening visual direction. Avoid ugly extreme material close-ups as the opening reference.
- Photo 2: Material / functional detail anchor, locking surface material, grain, reflection, tactile quality, and the most important visible mechanism or structure. If the product needs opening, docking, folding, clasping, button, port, lid, screen, or touch cues, merge those cues into this photo.
- Photo 3: Final copy anchor, locking final product composition, font feeling, text position, single-line layout, and two-part color treatment. The copy is a format reference and does not limit the later video to only this one line.

Multi-variant handling:

- The anchor photos must inherit the selected narrative spine and style strategy.
- Photos 1/2 prioritize building the visual system around the main variant.
- Other variants only support rhythm; do not lay everything flat or fill the frame.
- Photo 3 may show the main variant plus a few supporting variants, or a full-set closing composition, but must preserve negative space and order.

Final copy anchor typography rules:

- Must include the Step 5 confirmed copy; do not use placeholder copy.
- Text must stay on one line.
- Font: SF Pro Display Semibold.
- Text color must be split explicitly into two parts: the first half of the copy uses black or white, and the second half uses the specific product main color. If the main color is purple, write “first half black / white, second half purple.” Do not write only “product color + black or white.”
- Photo 3 may contain only this format-reference copy line and no other text; however, later video copy is controlled by the precise beat storyboard and may include multiple short English copy lines.

Show the three independent anchor photos and ask the user to approve, edit, or regenerate a specific photo before proceeding to the precise beat storyboard.

## STEP 7: Precise Beat and Text Storyboard Table

Before generating any video clip, create and read back a precise beat text storyboard table. It is not artwork; it is the execution table for the video model.

The storyboard table must be organized as “User Choice Statement → Beat Content → Principles.”

### User Choice Statement

Write the confirmed style, aspect ratio, duration, narrative spine, main variant, product main color, variant strategy, and copy. Example:

> White-tech style, 16:9, 10 seconds, Color Family spine, purple as the main variant, other colors as supporting layers, copy: Color Meets Sound.

### Beat Content Table

Recommended columns:

| Time range | Shot | Shot purpose | Visual lead | Visual / camera move | Variant state | Copy | Text color | Text effect | Transition / continuity | Rhythm intent |
|---|---|---|---|---|---|---|---|---|---|---|

Table rules:

- 5s suggests 3-4 beats.
- 10s suggests 5-7 beats.
- 15s suggests 6-9 beats.
- Use seconds as time ranges; do not require frame numbers.
- Keep one main action per beat; secondary layers appear with slight delay.
- Clearly define when the main variant appears alone, when supporting variants enter, and when the full set closes.
- Copy splitting may only be described in plain language as first half / second half. Do not write arrows, slashes, plus signs, or separators, because the model may render them as in-frame copy.
- Text color must use the specific product color name and must be split by style: in white-tech style write “first half black or dark gray, second half the specific product color”; only in dark rim-light style may the first half be white. For example, in white-tech style with purple as the main color, write “first half black or dark gray, second half purple.” Do not write “product color + black or white.”
- If typography is error-prone, reduce text beats first: for a 10-second film, usually use only one mid-film copy moment and one final copy moment, with all other beats marked as no text. Do not put text in every shot.
- The final beat must be a stable single full-frame product closing + full single-line copy. Never end on four panels, split screens, storyboard boards, anchor-sheet layout, product windows, or framed grids. Text still belongs in the vertically centered composition zone, left-aligned or right-aligned as part of the frame, not in a lower subtitle position. Add a logo only when the user provides one.

Rhythm intent vocabulary: setup, establish, prepare, impact, brake, settle.

### Principles

1. Keep core text on a single line.
2. At any moment, only one single-line English copy line may appear; forbid two text rows, upper/lower line splits, title + subtitle pairs, or multiple text blocks at the same time.
3. Use exactly two text colors: black or white + confirmed specific product color.
4. The text must appear inside the video; do not rely only on post-production fallback.
5. Shot continuity = element continuity + motion continuity + form continuity.
6. Smoothness = easing + no awkward timing + seamless entrances and exits.
7. Total pacing follows the user-selected duration.
8. Text motion must be visible: the first half fades in or slides in subtly first, then the second half fades in or slides in subtly later. When the second half appears, the first half shifts subtly within the same line to make room. Do not make the text completely static, and do not split it into upper and lower rows.
9. If the style is confirmed, the table must strictly follow that style.

After the user confirms the storyboard table, proceed to video generation.

## STEP 8: Video Generation

Generate video directly from the confirmed three independent anchor photos + precise beat text storyboard table. Video style, background, lighting, mood, and pacing follow the selected style. Aspect ratio and size must strictly follow the user-selected setting; if the user chose 16:9, generate 16:9 and do not auto-change to another ratio.

Video reference rules:

- During video generation, do not pass the original product image to the video model.
- The original product image is used only for product analysis and anchor generation.
- Use the three independent anchor photos to preserve product consistency, hero view, material / functional detail, final composition, and typography layout.
- Creative control comes from the precise beat storyboard: shot order, product action, copy, text motion, continuity, and transitions.
- Anchor photos are only product anchors used to protect identity, body color, material, and layout. Never treat them as the storyboard itself, and never turn the three photos into three sequential video segments.
- The video output must be one continuous full-frame ad film. Do not show four panels, split screens, collage layouts, frames, storyboard boards, grid layouts, or any shot that reproduces an anchor-sheet layout inside the video. This is especially critical for the ending: never return to a grid, storyboard board, small-window montage, or product wall.
- Default to one H3 full-frame product film, not multiple first-frame clips, unless the user explicitly asks for a different construction.
- Video generation defaults to MiniMax-H3 native audio unless the user explicitly asks for another model or silence. Use the later Apple-style tech BGM direction as the audio prompt so picture rhythm and sound are generated together.

The video prompt must include:

- The three independent anchor photos and the role of each photo.
- The full precise beat storyboard table; read every beat and rewrite every beat into the video prompt. Do not summarize only early beats or omit later typography, variants, or closing requirements.
- The exact copy lines. The video prompt must spell out every in-frame copy line verbatim, its time window, first-half / second-half entry timing, colors, and single-line behavior. Do not rely on the storyboard table alone, and do not summarize this as “text beat sync” or “leave space for copy.”
- Prompt completeness check: if the final video-generation prompt does not literally contain every intended English copy line, the task is not ready to dispatch. Rewrite the prompt before video generation.
- Single-line typography hard constraint: at any moment, only one single-line English copy line may appear in the frame; forbid two rows, two-line titles, title + subtitle pairs, or multiple text blocks at the same time. If typography is error-prone, reduce copy frequency instead of putting text in every shot.
- Two-part text motion rules: the first half and second half must continue or replace within the same line, never split into upper and lower rows; motion must be visible, with the first half fading in or sliding in subtly first, then the second half fading in or sliding in subtly later, while the first half shifts gently within the same line. Do not make the typography completely static.
- Shot continuity hard constraints.
- MiniMax-H3 native audio setting and music prompt; native audio is on by default unless the user explicitly asks for silence or another model.
- Do not pass the original product image.

## STEP 9: music-2.6 Tech BGM

Video generation defaults to MiniMax-H3 native audio first, and the Apple-style tech BGM direction below should be written into the video prompt. If H3's native music is unpleasant, too loud, too weak, out of sync, or the user asks for new music, use `music-2.6` to generate a longer standalone instrumental BGM and replace the video audio track. Do not default to ElevenLabs; use ElevenLabs only when the user explicitly requires a strict target duration, and follow the fee-confirmation rule.

Default music direction:

```text
Around 100 BPM, fast tech feeling, block chords, pluck and airy noise bed, kick + sub-bass + sine sweep, wooden percussion, sudden cut-off, within 0.5s everything stops, leaving only the pluck tail to decay.
```

Extended interpretation:

- Techy, Apple-style, product launch feeling.
- Rhythmic drive is allowed, but it must not become cheap EDM.
- Pluck should be crisp, pleasant, and forward; airy noise should feel premium.
- Kick and sub-bass serve edit points; avoid boomy bass.
- Wooden percussion adds tactile texture, not a complex drum loop.
- No vocals, no synthwave, no retro, no gaming, no cheap corporate stock music.

## STEP 10: Music Analysis, Beat-synced Editing, and Delivery Verification

Before final delivery, edit or assemble based on music analysis. Do not simply lay music under the video.

Music analysis:

- Detect rhythm, strong beats, energy sections, and clear single-note hits.
- Find an 8-12 second segment with continuous sound and suitable energy for product display.
- Do not blindly use the first 10 seconds.
- Prefer a segment with a clear entry point, no silence or collapse in the second half, and enough stable time for the closing frame.

Audio assembly rules:

- Keep the video image, cut a continuous music segment, and replace the video audio track.
- Mute the original video track; keep only the new music.
- Keep music loudness restrained: do not maximize loudness, do not use aggressive limiting, and do not raise music too much; but also do not push it so low that the pluck, rhythm points, and product display lose clarity.
- Do not use long fade-outs. Only apply a 0.1-0.3s tail fade when needed.
- After assembly, check for long silence or audible dropouts.

Pre-delivery verification:

- Video aspect ratio, duration, and language match the brief.
- Product appearance, main variant, and variant timing match the anchor sheet and storyboard.
- Copy is readable, single-line, not crowded, and contains no extra arrows or symbols; no two text rows, upper/lower line splits, title + subtitle pairs, or multiple text blocks appear at the same time.
- The final frame has a stable product + copy closing.
- Music is not too loud and not too low.
- Output is on canvas, and multi-asset outputs are grouped.

Delivery includes:

- Final video path or canvas output.
- Duration, aspect ratio, and language.
- Short creative summary.
- Copy used and music segment used.
- Whether beat-synced editing was applied.
- Specific next improvements, such as pacing, copy, main variant, audio, or platform crop.

## Failure Handling

- Product image too weak: stop, give reshoot guidance, and wait for a better image.
- Copy too long: compress to 3-5 English words and ask for approval.
- Text anchor is wrong: regenerate the text anchor before video generation.
- Anchor sheet fails product fidelity: regenerate with stronger product-preservation instructions.
- Storyboard is templated: return to the precise beat table and rewrite based on product form and narrative spine.
- Copy does not appear in video: rerun the video prompt, explicitly requiring copy in at least the middle and final beats.
- Video typography deforms: regenerate the text anchor or degrade the text shot to subtle anchor-based motion.
- Aspect ratio is wrong: immediately rerun with the user-selected ratio.
- Music quality is poor: retry with `music-2.6`, do not force duration, and cut the best segment from the longer track.
- Final assembly has no sound in the second half: re-analyze music, select a continuous audible segment, and use direct audio replacement.

## Trigger Examples

Use this Skill for requests like:

- “帮我的产品做一个苹果味儿广告片”
- “用这张产品图做 Apple 风电商广告”
- “做一个极简高级的产品发布视频”
- “Make an Apple-style product ad from this product photo”
- “Create a premium minimalist ecommerce product film”

Do not use this Skill for general video editing, documentary explainers, KOC talking-head ads, or complex UI/screen-text demos unless the user explicitly wants the Apple-style product ad workflow.


---

## FILE: skills/minimalist-product-ad-generator/meta.yaml (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/minimalist-product-ad-generator/meta.yaml)

display-name-zh: 极简产品广告生成器
version: 0.5.6
tag-en: "Commercial Ad"
tag-cn: "商业广告"
complete-tags-en:
  - "Commercial Ad / Planning"
  - "Commercial Ad / Creative Generation"
  - "E-Commerce / Creative Generation"
complete-tags-cn:
  - "商业广告 / 策划"
  - "商业广告 / 创作生成"
  - "电商 / 创作生成"
summary-en: "Turn product images and ad requirements into minimalist product ad shorts for e-commerce promotion and product launches."
summary-cn: "基于产品图片和广告需求，完成卖点提炼、英文文案与节拍分镜设计，输出极简产品广告短片，适用于电商推广与新品发布。"
desc-en: "Based on product images and advertising requirements, extract selling points, write concise English ad copy, design product anchors and beat-synced storyboards, then generate a minimalist product advertising short for e-commerce promotion, product launches, small brands, food, toys, gadgets, accessories, and other physical goods. The Skill focuses on clean product presentation, readable typography rhythm, premium camera language, and final video delivery."
desc-cn: "基于产品图片和广告需求，完成卖点提炼、英文文案与节拍分镜设计，输出极简产品广告短片。适用于电商推广、新品发布、小品牌上新、食品、潮玩、数码配件和其他实体产品。Skill 会确认时长与比例，整理产品事实和主推卖点，生成简洁英文广告文案，制作产品锚定图，规划文字卡点分镜，并生成干净、有节奏、产品质感突出的成片。"
author-en: "User"
author-cn: "用户"
source: community


---

## FILE: skills/music-video-subtitle-generator/SKILL.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/music-video-subtitle-generator/SKILL.md)

---
name: music-video-subtitle-generator
description: |
  For musicians, video creators, and social-media editors producing AI music videos or emotional short films with lyric typography. Users provide music, lyrics, references, characters, typography direction, mood, or target platform. The Skill analyzes beat and vocal timing, separates character, scene, and text references, designs beat-reactive spatial typography, decomposes long works into connected shots, audits prompts, and routes generation for H3 or other video tools. It outputs MV concepts, shot prompts, lyric text plans, and stitching guidance. Best for stylized MVs and subtitle-driven music visuals, not ordinary caption cleanup, licensed IP copying, or fully manual post-production editing.
compatibility: Requires the MiniMax Hub agent (canvas workspace and Hub generation/routing tools); not portable to generic agent harnesses.
trigger-words: [MV, music video, lyric typography, on-screen text, prompt audit, Trap MV, Gospel hip-hop, Dark-pop, Cyber-grunge, MV提示词, 歌词文字, 字幕MV, 贴字MV, 卡点MV, 多镜头拼接]
---

# Music Aesthetics MV

## Purpose

Use this Skill when the user wants to create, revise, audit, or generate music-video prompts or emotional short-film prompts where music, lyrics, typography, references, rhythm, performance, and camera language must be designed together. The workflow adapts MV prompt rules into Hub execution: key creative decisions are confirmed, locked prompts are written to canvas text nodes, and media creation is delegated to Hub agents.

Do not use it for ordinary subtitle burn-in, generic video editing, non-music product ads, or simple single-image / single-clip requests without MV structure.

## Hub compatibility rules

- Do not assume third-party tools, shell scripts, browser plugins, or external generation APIs.
- Write locked prompts and revisions to Hub canvas text nodes.
- Delegate character cards, scene cards, typography cards, video clips, BGM, and final assembly to Hub image, video, music, and editing agents.
- Do not hardcode output paths; always use Hub-returned paths.
- If the user says “skip confirmation / just do it”, apply it only to the current run unless the outer orchestrator has active session authorization.
- If the user only wants a prompt, stop after delivery. If the user wants a finished MV, continue to generation and assembly after confirmation.

## Core principles

1. Understand the work before choosing a structure. Templates organize, but are not mandatory forms.
2. Use the latest confirmed creative intent.
3. Omit irrelevant fields instead of filling placeholders.
4. Do not mechanically add performance, typography, transitions, camera moves, or character actions just because a preset includes them.
5. On-screen text is a designed visual layer, not ordinary subtitles, unless the user explicitly requests subtitles.
6. If the user uploads a real song or beat, it is the master music bed unless they request replacement.
7. **Natural multi-shot stitching for >15s videos**: when duration exceeds the model single-generation limit, use “multi-shot storyboard breakdown + tail/head frame continuation + beat hard cuts + global master-audio alignment”. Every segment must preserve the same groove, tempo feel, aspect ratio, character logic, visual preset, lighting language, and typography motion rules.
8. If the user wants a finished music-aesthetic MV without lyrics, generate and lock original lyrics first. The final MV must reference character, typography packaging, and scene cards together. The typography packaging card controls only text packaging style, font texture, graphic design, layout ratio, and motion language.

## STEP 1: Pre-flight lock

Before writing the final production prompt, confirm the minimum foundation with concise recommended choices.

### 1.1 Video format

Offer fixed options:

| Use case | Aspect ratio | Resolution |
| :--- | ---: | ---: |
| TikTok / Reels / Shorts vertical | 9:16 | 1080×1920 |
| YouTube / Bilibili horizontal | 16:9 | 1920×1080 |
| Feed square | 1:1 | 1080×1080 |
| Dense vertical ad test | 9:16 | 720×1280 |
| Cinematic widescreen MV | 21:9 | 2560×1080 |

The selected ratio must be reused across character cards, scene cards, typography references, shot prompts, video clips, and final assembly.

### 1.2 Target duration and multi-shot structure

Before prompt writing or video generation, confirm target MV duration. Never silently use the model default as user intent.

Offer a compact duration card:

1. **10-second test**: one or two shots to verify character, scene, typography, and audio style.
2. **15-second hook**: 2–4 short shots for a more complete chorus / hook / performance phrase.
3. **30-second or longer complete MV (multi-shot stitching)**: because many models have a 15-second single-generation ceiling, automatically use a Multi-Shot stitching workflow. First generate or lock one continuous song / BGM, split 30 seconds into 4–8 dynamic 2–5 second shots, then stitch via beat sync and head/tail frame continuity.
4. **Custom duration**: user provides duration; plan shot count and stitching based on the selected model’s single-clip limit.

If uploaded music exists, the confirmed duration is the music window length.

### 1.3 Music window

If uploaded or specified music is longer than the confirmed duration, lock a segment before writing prompts:

1. Recommended window: assistant identifies the strongest chorus / hook / emotional turn and requests confirmation.
2. User timestamps: user provides start and end seconds; verify both are inside the audio.
3. Multi-variant: only for ad-hook testing or multiple creative directions.

If music is shorter than the target duration, use the full audio unless the user explicitly asks to stretch or extend it.

### 1.3.0 Lyrics lock and lyrics-first fallback

Before final video prompts or generation, determine lyric ownership:

1. If the user provides lyrics, those are locked lyrics. Do not generate, add, rewrite, expand, translate, paraphrase, or replace them unless explicitly requested.
2. If the user wants a finished music-aesthetic MV without lyrics, generate short original lyrics matching the music style, vocal mode, duration, emotion, and visual preset, then lock them.
3. Locked lyrics are the only source text for rap/singing performance and visible typography.
4. Split locked lyrics into each shot’s `Rap line:`, `Vocal line:`, `Soft vocal line:`, or `Spoken line:`.
5. Every `Typography:` field must come from the same locked lyrics. During vocal performance, visible text must word-for-word match the performed phrase.
6. In the final typography MV, the performer must rap or sing the locked lyrics with visible lip shapes, jaw motion, breath, facial accents, nods, and hand accents following phrasing and rhythm.

### 1.3.1 Default multi-shot stitching for >15s videos

For >15s MVs such as 30 seconds, the global multi-shot prompt script is the authoritative source:

1. **Lock complete Master Audio**: lock one continuous song / BGM track with the full vocal and groove as the only audio baseline.
2. **Build a Shotlist Timeline**: split 30 seconds into 4–8 short shots of 2–5 seconds, mapped precisely to lyric timestamps and beats such as snare, 808, and drop.
3. **Head/tail frame and scene continuity**: if Shot B continues the same scene, use Shot A’s tail frame as Shot B’s head frame. If it is a hard cut, preserve the same character card, wardrobe, lighting prompt, and use same-direction camera motion or match-cut visual elements.
4. **Generation and edit assembly**: the video agent generates each shot from the Shotlist. The editing agent aligns all clips to the global Master Audio beat grid, trimming, speed-ramping, and stitching for natural lip-sync, beat sync, and transitions.

## STEP 2: Creative Contract

Before final prompts, create a concise creative contract:

- Music genre, instrumentation, tempo/BPM feel, vocal mode, and emotional temperature.
- Lyric source and locked lyrics.
- Target duration and multi-shot breakdown, e.g. 30s split into 6 short shots.
- Reference image roles: character card, scene card, typography packaging card.
- Camera language, shot sizes, focus, beat-cut density, and transition logic.
- Exclusions such as no fades, no glossy AI beauty face, no single-shot overstretching that causes deformation.

## STEP 3: Reference roles and scene sampling

Assign each reference image one narrow job:

- **Typography reference card**: controls only text packaging style, font texture, graphic design, layout ratio, and motion language. It must not contribute people or scenes.
- **Character reference card**: controls character identity, facial aura, hairstyle, clothing silhouette, proportions, posture, and presence.
- **Scene reference card**: controls scene style, spatial atmosphere, image texture, background depth, and lighting mood.

### 3.1 Close-up scene switching and multi-shot sampling

For Trap, Dark-pop, Cyber-grunge, and other fast-paced MVs, switch quickly among multiple close-up spaces:

- Keep scene mood unified, but each shot’s space must differ clearly, such as local background, tunnel, wall, light strip, or reflective floor.
- Scene switches must be triggered by musical impacts: bass hit, 808 drop, snare, vocal accent, or typography smash.
- Shot switches are “visual samples” on the trap beat: hard cuts, jump cuts, scan-glitch cuts, flash cuts, or action match cuts.

## STEP 4: Preset grammar, Dark-pop / Cyber-grunge and Trap example

### Visual and editing language

- Realistic high-fashion texture and film-magazine texture, referencing late-90s / early-00s indie magazines, photocopy paper, film scans, and zine collage.
- Heavy grain, slight film jitter, halftone dots, rough print edges, scan offsets.
- **Editing must use hard cuts only**. No fades, dissolves, or soft transitions.
- Image and typography strongly respond to beat cues: hi-hat roll creates micro-shake / frame skips, snare triggers scale-up / hard cut / shoulder drop, 808 bass hit creates low-frequency compression / stretch / offset.

### Typography packaging rules

- Typography is a dynamic graphic subject in space, not a subtitle bar. It may sit in foreground, midground, background, or be occluded by shoulders / hands.
- Text must never cover eyes or main facial expression; avoid the mouth during critical lip-sync.
- With vocals, visible words must exactly match the performed lyrics. Each shot has only one main typography event.

## STEP 5: Prompt structure template

A multi-shot script must be modular by shot, with accurate duration and audio mapping:

```text
[Global Aesthetic & Character Lock]: (global character and aesthetic anchor prompt)

Shot 1 (0.0s - 3.5s)
Vocal Line: "..."
Typography: "..."
Visual & Action: ...
Camera & Motion: ...
Transition Out: hard cut to Shot 2 on bass hit / same-direction whip pan

Shot 2 (3.5s - 7.0s)
Vocal Line: "..."
Typography: "..."
Visual & Action: ...
Camera & Motion: ...
Transition Out: ...
```

## STEP 6: BGM continuity and natural multi-shot stitching system

### 6.1 Global audio continuity

The entire MV must bind to one Master Audio track. During segmented video generation, do not use independent disconnected clip audio. All video clips are aligned to their corresponding timestamps on the Master Audio timeline during edit assembly.

### 6.2 Natural Stitching Protocol

For >15s stitching, enforce five continuity locks:

**Vocal & Lip Continuity**: Cut points must land on lyric pauses, breaths, snare, or drop. Do not hard cut inside an active vowel or lyric unless the next shot is an extreme close-up with continuous mouth shape.

**Rhythm & Beat Matching**: Cut points must hit the 1/4 or 1/8 beat grid. Use speed ramping to align head nods, hand gestures, and blinks to beats.

**Aesthetic & Color Grading**: Every cross-shot generation carries the same aesthetic header prompt: grain level, LUT direction, and light direction. Final assembly uses unified 35mm film grain and LUT to hide batch color differences.

**Transition & Motion Continuity**: For long-shot continuation, use the previous tail frame as the next head frame and prompt continuation of action. For hard scene switches, use kinetic continuity such as same-direction pan or hand-occlusion match cut.

**Typography Motion Stitching**: Text in the previous shot should shatter, sweep out, or smash-screen on a bass hit; next-shot text smashes in or unfolds on the accent, transferring visual momentum.

## STEP 7: Checklist

Silently verify before delivery:

- [ ] If target duration exceeds 15s, has Multi-Shot breakdown been used automatically?
- [ ] Is one global Master Audio locked, and are cut points beat-aligned?
- [ ] Do cuts avoid mid-lyric vocal breaks, with natural lip and breath continuity?
- [ ] Do adjacent shots have clear stitching logic such as tail/head continuation, same-direction motion, match cut, or beat hard cut?
- [ ] Is the edit style strictly hard-cut, with no fades or soft transitions?
- [ ] Are character / scene / typography cards isolated without cross-contamination?
- [ ] Is typography a spatial layer that never blocks eyes or main facial expression?
- [ ] Are image quality, grain, color, and lighting consistent globally?

## STEP 8: Canvas delivery

After the complete MV prompt script is locked, write it to a dedicated canvas text node named `Complete MV Prompt` or `完整MV Prompt`. The node must contain the full multi-shot prompt script and stitching notes. Later revisions update the same node.

## STEP 9: Final MV generation and assembly workflow

1. **Prompt lock and canvas write**: confirm the full multi-shot prompt script with timestamps, lyric mapping, and transition logic is written to canvas.
2. **Parallel shot generation**: the video agent generates each 2–5s shot from the Shotlist. For continuation shots, extract the prior shot’s tail frame as the next I2V head frame.
3. **Multi-track editing and stitching**: the editing agent imports the global Master Audio, aligns all shot videos by timestamp, trims and speed-ramps on the beat grid, and checks seams with same-direction motion or flash-cut masking.
4. **Finishing**: apply unified Movie LUT and 35mm film grain overlay to reduce AI plasticity and lock visual consistency.
5. **Delivery**: output the seamless complete MV video file.


---

## FILE: skills/music-video-subtitle-generator/meta.yaml (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/music-video-subtitle-generator/meta.yaml)

display-name-zh: 音乐MV动态字幕生成器
version: 0.6.6
tag-en: "Sound & Music"
tag-cn: "音频音乐"
complete-tags-en:
  - "Sound & Music / Planning"
  - "Sound & Music / Creative Generation"
  - "Creative & Experimental / Creative Generation"
complete-tags-cn:
  - "音频音乐 / 计划制定"
  - "音频音乐 / 创作生成"
  - "创意实验 / 创作生成"
summary-en: "Create beat-synced MV prompts and lyric typography plans from music and lyrics."
summary-cn: "基于音乐和歌词，完成节奏拆解、镜头设计与动态字幕方案，适用于AI MV创作。"
desc-en: "For musicians, video creators, and social-media editors producing AI music videos or emotional short films with lyric typography. Users provide music, lyrics, references, characters, typography direction, mood, or target platform. The Skill analyzes beat and vocal timing, separates character, scene, and text references, designs beat-reactive spatial typography, decomposes long works into connected shots, audits prompts, and routes generation for H3 or other video tools. It outputs MV concepts, shot prompts, lyric text plans, and stitching guidance. Best for stylized MVs and subtitle-driven music visuals, not ordinary caption cleanup, licensed IP copying, or fully manual post-production editing."
desc-cn: "面向音乐人、视频创作者和社交媒体剪辑者，用于制作带歌词贴字的 AI MV 或情绪短片。用户提供音乐、歌词、参考图、角色、字体方向、情绪意图或发布平台。Skill 会分析节拍与人声时序，区分人物、场景和文字参考，设计随节奏变化的空间字幕，拆解长视频为可衔接镜头，审查提示词并路由 H3 等视频生成工具。最终输出 MV 概念、分镜提示词、歌词文字方案和拼接建议。适合风格化音乐视觉，不适合普通字幕校对、照搬已有 IP 或完全手工后期剪辑。"
author-en: "MiniMax Hub User"
author-cn: "MiniMax Hub 用户"
source: official-featured


---

## FILE: skills/paper-collage-explainer-generator/SKILL.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/paper-collage-explainer-generator/SKILL.md)

---
name: paper-collage-explainer-generator
description: |
  For creators, educators, and social-video editors who need a tactile paper-collage language for narration, knowledge points, opinions, or abstract topics. Users provide source copy, story beats, or a core concept and may specify aspect ratio, duration, palette, and audio needs. The Skill extracts meaning, proposes visual metaphors, prepares a production plan and storyboard, generates approved halftone collage stills, then creates stop-motion clips with paper movement and tactile sound effects, with optional final assembly. By default it keeps collage SFX and does not add BGM, voiceover, or subtitles unless requested. Best for explainers, viewpoints, story visuals, and social B-roll; not for presenter ads, editable layers, complex typography, or prompt-only tasks.
compatibility: Requires the MiniMax Hub agent (canvas workspace and Hub-native image/video/audio tools); not portable to generic agent harnesses.
trigger-words: [paper collage explainer, paper-collage animation, halftone collage, collage explainer, 纸拼贴, 拼贴科普, 定格拼贴, 拼贴动画]
exported-by: MiniMax-hub
---

# Paper Collage Explainer Generator

Turn a short narration line, story topic, viewpoint sentence, or abstract idea into a cohesive editorial paper-collage animation sequence. The visual language is premium halftone paper collage: flat bold color fields, black-and-white photographic cut-outs, selective colored cardstock accents, warm cream keylines, soft paper shadows, tactile stop-motion assembly, and crisp collage sound effects.

This Hub-adapted Skill uses Hub-native image, video, audio, and optional postprocess capabilities. It prioritizes style continuity, color harmony, controlled paper texture, stop-motion collage rhythm, and an audio policy that keeps tactile collage SFX by default while explicitly not adding BGM, voiceover, or subtitles unless the user requests them.

## When to Use

Use this Skill when the user wants:

- A short script line turned into a visual-metaphor collage animation
- A simple story or literary topic explained through collage B-roll
- Editorial paper-collage animation for narration or social video
- Halftone collage animation with objects assembling from an empty color field
- A batch of short abstract sentences or story beats converted into separate visual-metaphor animation clips
- A tactile knowledge explainer that may include paper clicks, pops, slides, presses, and rustles but not automatic music or voiceover

Do not use this Skill when the user needs:

- A realistic product ad or presenter-led video
- Precise editable layers, timeline keyframes, or transparent cut-out assets
- Exact logo placement or readable typography
- Only a written video prompt without generation

## Default Creative Targets

Unless the user specifies otherwise:

- Output ratio: 16:9 landscape
- Clip length: about 4 seconds per segment
- Audio default: keep or generate tactile paper-collage sound effects only, such as paper slide, pop, press, light rustle, and soft tap sounds
- Default: do not add BGM. You may ask whether the user wants BGM when it may help, but add music only after explicit user confirmation
- Default: do not add voiceover, spoken narration, or presenter narration. You may ask whether the user wants narration/口播 when the project may benefit, but write, synthesize, or add spoken audio only after explicit user confirmation
- Default: do not add subtitles. You may ask whether the user wants subtitles, but create or burn in subtitles only after explicit user confirmation
- Visual style: premium editorial halftone paper collage
- Motion style: tactile stop-motion assembly, not slow zoom, generic drifting, or smooth digital layer movement
- Default video generation model: `MiniMax-H3`, unless the user explicitly specifies another model, the model is unavailable, or a hard capability requirement excludes it
- Text in image/video: avoid readable letters, numerals, UI, subtitles, watermark, and logos
- Image quality and depth: prioritize visually attractive, layered 16:9 compositions with clear foreground, midground, and background depth, strong subject hierarchy, rich but readable scene design, and controlled negative space

## Audio Policy

This Skill's default delivery is **with collage SFX, and without BGM, voiceover, or subtitles**.

1. During the first production-plan confirmation, state the default media approach explicitly: tactile paper-collage sound effects are kept or generated; BGM, voiceover/口播, and subtitles are not added by default.
2. It is acceptable to ask the user whether they want **voiceover narration/口播**, **BGM**, or **subtitles**, especially for explainers, but present all three as optional add-ons, not defaults.
3. Do not infer that an explainer automatically needs spoken narration. If the user does not choose narration, create visual story beats rather than a voiceover script.
4. Do not infer that a social video automatically needs BGM or subtitles. If the user does not choose them, keep the clip SFX only.
5. When generating video clips, request synchronized tactile collage SFX if the selected video model supports audio: paper pieces sliding, popping, pressing flat, soft taps, and light paper rustles.
6. During final assembly, preserve the original clip audio tracks when they contain collage SFX. Do not drop audio by default.
7. Remove or replace audio only when the user asked for silence, when the generated audio contains unwanted speech/music, or when the user requests a separate music/narration mix. Create subtitles only when the user explicitly asks for them.

## Global Style Rules

Apply these rules to every still and video in the project:

1. **Unify the overall visual style.** Every segment should feel like part of the same editorial paper-collage series: halftone cut-outs, flat color fields, warm cream keylines, soft physical shadows, clean composition, tactile paper material, and coordinated collage SFX.
2. **Control the paper texture intensity.** Paper must not look perfectly flat, but it also must not become over-aged, dirty, wrinkled, or brown unless the user explicitly asks. Prefer clean, refined hand-made texture: subtle fiber, slightly irregular torn edges, light deckled fibers, layered seams, and soft shadows.
3. **Unify the color tone.** Do not introduce kraft-paper, brown, yellowed, or distressed base papers when they clash with the approved still or the batch palette. Start each clip from a paper field that matches the approved final still's main color direction.
4. **Make motion read as stop-motion collage.** Use clear paper-piece actions: appear piece by piece, slide or pop into position, lightly bounce, press flat, pause, then lock into the final composition. Avoid fast spinning, excessive flipping, chaotic object flight, global fades, smooth digital panning, zooming, or generic drifting.
5. **Keep segments coordinated.** Once one or two clips establish the approved batch style and SFX cadence, later clips and revisions should reference that cadence and tone so the whole sequence feels coherent.
6. **Default to polished 16:9 layered scenes.** Unless the user explicitly requests another platform frame, plan and generate in 16:9 landscape. Use the wider canvas to build attractive foreground / midground / background separation, richer environment props, clear subject hierarchy, and cinematic lateral composition without turning the frame into clutter.
7. **Emphasize paper-collage craft.** Every still and clip should make the paper-collage method visible: separable paper groups, halftone photographic cut-outs, colored cardstock accents, tactile shadows, torn edges, layered seams, and stop-motion actions such as pop-in, slide-in, light bounce, press-flat, pause, and lock.

## STEP 1: Parse the Input

For each line, concept, story, or topic, extract:

- Core meaning: what the viewer should understand
- Emotion: calm, urgent, ironic, surprising, absurd, clarifying, reflective, mysterious, or playful
- Action verb: open, connect, leak, archive, compress, split, illuminate, bind, assemble, reveal, fall, chase, transform, collide
- Visual metaphor: a concrete image that expresses the idea without writing the script on screen
- Key objects: three to six large readable paper groups
- Audio implication: whether the beat benefits from paper slide, pop, press, tap, rustle, or snap SFX

For a story topic, split the story into three to six concise beats unless the user specifies a count. Similar beats may share the same design language, but each should have a distinct metaphor, object set, color field, and SFX rhythm.

## STEP 2: Gate 1 — Production Plan Document Approval

Before generating any media, create a concise production plan document and stop for user approval. Do not generate stills or videos before the user confirms this document.

The production plan must include:

### Brief

- Topic or source line
- Intended audience / use context when known
- Aspect ratio and duration assumptions
- Tone and pacing
- Visual style summary
- Media approach, stated as: default collage SFX are kept or generated; BGM, voiceover/口播, and subtitles are not added unless the user explicitly requests them
- Optional add-on question when useful: ask whether the user wants voiceover narration/口播, BGM, or subtitles, but keep them optional

### Visual Metaphors

For each planned segment:

- Core meaning
- Emotion
- One-sentence visual proposition
- Three to six key objects
- Suggested background color and accent colors
- Expected assembly order
- Expected collage SFX moments, such as slide, pop, press, tap, rustle, or snap

### Script / Visual Beat Track

If the user explicitly requests narration, write a concise voiceover script. If the user has not explicitly requested narration, do **not** write a voiceover script; instead write a silent visual beat track explaining what the audience understands from the sequence.

If the user only needs B-roll for an existing line, preserve the original line as context instead of inventing narration.

### Storyboard

For each segment, include:

- Segment title
- Final-frame description
- Motion idea
- Approximate duration
- Collage SFX idea
- Notes on style continuity and color harmony

After presenting the production plan document, wait for the user to approve, reject, or revise. If the user approves only some numbered items, move only those items forward and revise the rest.

## STEP 3: Build Still-Frame Specifications

After the production plan is approved, write a compact visual specification for each approved segment. The specification should be self-contained and suitable for image generation.

Include:

- Script meaning or story beat
- Visual metaphor
- Aspect ratio
- Background color field
- Accent colors
- Key object groups and their roles
- Composition, foreground / midground / background depth, and negative space
- Final frame relationship
- Style continuity notes
- Avoid list for this specific still

Use this style signature:

```text
flat bold color field, black-and-white halftone photographic cut-outs, selective colored cardstock accents, warm cream keylines, soft paper shadows, fine uncoated-paper grain, premium editorial paper collage, clean refined hand-torn paper edges, subtle fibrous edges, layered paper seams
```

Color guidance:

- Burnt orange or red: labor, time pressure, urgency
- Mustard yellow: tools, warning, accumulated errors
- Ink green: cognition, reset, judgment, surreal calm
- Deep purple: memory, structure, mystery, dream logic
- Teal: collaboration, execution, system flow
- Rose red: absurdity, ceremony, theatrical tension

Do not make cobalt blue the automatic default. Keep the batch visually unified through paper texture, halftone treatment, keylines, shadows, framing, and a controlled palette. Vary the color field only when the story beat benefits from contrast.

## STEP 4: Gate 2 — Generate and Approve Still Frames

Generate one final still frame per approved segment. The still must look like the completed last frame of the future animation.

Still-frame requirements:

- Use the user-specified aspect ratio; otherwise default to 16:9 landscape
- Flat bold paper background with subtle fiber texture
- Three to six large separable object groups, unless the approved storyboard requires slightly more
- Clear central subject, strong visual hierarchy, generous clean negative space, and visible foreground / midground / background layering
- Black-and-white halftone photographic cut-outs as the main structure, combined with rich but readable scene props that support the story beat
- Selective colored cardstock accents only where they clarify hierarchy
- Warm cream keylines, soft physical paper shadows, refined torn edges, and subtle layered paper seams
- No readable text, fake letters, numerals, subtitles, UI, watermark, or logos unless explicitly requested

Show the generated still frames to the user and stop for approval before video generation. If a still looks too busy, has mismatched color, too many paper layers, too-flat digital edges, or too much aged/brown paper texture, revise the still before video generation.

## STEP 5: Plan the Stop-Motion Assembly

For each approved still frame, prepare a motion plan that treats the approved still as the completed final frame.

Default motion order:

1. Start from a clean paper color field matching the approved still's main background, not a mismatched kraft-paper or brown base.
2. Base structure appears as paper pieces that slide, pop, or press into place.
3. Character, card, object, or main metaphor element enters.
4. Secondary objects assemble one by one with small bounce, press-flat, and pause beats.
5. Final relationship locks into the approved composition.
6. Hold the completed still-like frame briefly at the end.

Motion prompt guidance:

```text
Paper-collage stop-motion assembly. Start on a clean paper field matching the approved still's background color and tone. Assemble the scene piece by piece with tactile stop-motion timing: foreground, midground, and background paper groups appear in a readable order; each paper object appears, slides or pops in, lightly bounces, presses flat, pauses, and locks into place. Preserve the halftone dots, cream keylines, subtle fibrous torn edges, layered paper seams, soft shadows, paper grain, color field, and approved aspect ratio. End by holding the completed composition matching the approved still frame. Sound design: tactile collage SFX only, synchronized to paper-piece motion: soft paper slide, pop-in, press-flat tap, light rustle, and tiny paper snap where appropriate. Default: do not add BGM, voiceover, dialogue, or subtitles unless the user explicitly requested them. No mismatched kraft-paper opening, no brown/yellowed base unless approved, no fast spinning, no chaotic object flight, no smooth digital layer movement, no scene cuts, no camera move, no zoom, no morphing, no new objects, no text, no logos, no watermark, no UI.
```

## STEP 6: Generate Collage-SFX B-roll Clips

After Gate 2 approval, generate one video clip per approved still frame.

Generation requirements:

- Use the approved still as the visual reference / final-frame anchor whenever the selected Hub video model supports it
- Use the same ratio as the still frame
- Target about four seconds unless the user requested a different length
- Generate or preserve tactile collage SFX by default when supported: paper slide, pop, press-flat tap, light rustle, tiny snap
- Default: do not generate or add BGM unless the user explicitly requested music
- Default: do not generate or add voiceover, spoken narration, or presenter audio unless the user explicitly requested narration/口播
- If a generated clip contains useful collage SFX, preserve it during post-production and final assembly
- If a generated clip contains unwanted speech, music, fake UI sound, or noisy ambience, remove or regenerate the audio according to the user goal
- Keep the shot visually stable; do not add camera movement unless the user asks
- Preserve the collage material language, final composition, and established batch style
- When revising later clips, use the best approved previous clips as style/cadence references if helpful

Use `MiniMax-H3` as the default video generation model for this Skill. Do not ask the user to choose a model unless `MiniMax-H3` fails, is unavailable, or cannot satisfy a hard capability requirement. If the user explicitly specified another model, keep that model as primary and adapt parameters only when needed.

## STEP 7: Quality Review

Review stills and clips against these standards:

- The metaphor or story beat is understandable without writing it on screen
- The still has a strong clean color field, attractive 16:9 composition by default, clear object hierarchy, and visible foreground / midground / background depth
- The number of object groups stays readable, not a screen full of fragments
- The clip begins from a color field matching the approved still or batch palette
- The assembly is visible piece by piece, not a global fade-in
- Motion reads as stop-motion collage: separate paper groups pop in or slide in, lightly bounce, press flat, pause, and lock
- Defaults: tactile collage SFX are desirable; BGM, voiceover/口播, and subtitles are absent unless explicitly requested
- Paper texture is controlled: not too flat, not over-aged, not brown/kraft unless approved
- Color tone is consistent across the batch
- The final frame remains close to the approved still
- There are no unwanted readable letters, fake UI, subtitles, watermark, or logos

Minor drift in tiny details is acceptable if the metaphor remains clear and the final composition still resembles the approved still. Major drift, added text, fake UI, mismatched opening color, unwanted kraft-paper base, over-wrinkled/dirty paper texture, lost stop-motion assembly, unwanted BGM, or unwanted voiceover should be regenerated or post-processed depending on where the issue appears.

## STEP 8: Optional Assembly, Music, Narration, and Delivery

If the user asks for a multi-clip story video, assemble the approved clips in the confirmed storyboard order and preserve each clip's original collage SFX by default.

Add BGM, narration/口播, or subtitles only when explicitly requested. When the user requests BGM or narration after clips already exist, mix it with the collage SFX instead of replacing SFX, unless the user asks to mute the original SFX.

For each completed item, deliver:

- Final B-roll video path or final assembled video path
- Approved still frame path when relevant
- A one-sentence explanation of how the script or visual beat became a visual metaphor
- Media note: collage SFX by default, and list any explicitly requested BGM, narration/口播, or subtitle additions
- Any QA caveat if a clip passed with minor acceptable drift

For batch work, group the outputs by item number and keep the user's original line or story beat attached to its result.

## Failure Handling

- If the production plan defaults to silent video, correct it to collage SFX only unless the user explicitly asked for silence.
- If the production plan includes BGM, voiceover/口播, or subtitles without explicit user request, remove them and present BGM, voiceover/口播, and subtitles as optional choices.
- If the production plan feels too literal, return to Gate 1 and make the visual proposition more object-driven.
- If the still contains fake text or UI, regenerate the still before video generation; do not try to fix it only in the video prompt.
- If the still or clip is too visually busy, reduce paper layers, simplify the object count, and restore negative space.
- If the video lacks assembly motion, reduce the number of objects and make the step-by-step arrival order more explicit.
- If motion feels like smooth digital layers, rewrite the prompt around stop-motion beats: appear → light bounce → press flat → pause → lock.
- If the paper looks too flat, add subtle torn edges, fiber texture, layered seams, and soft shadows.
- If the paper looks over-aged or dirty, remove kraft-paper, brown/yellowed base, heavy wrinkles, stains, and distressed treatment.
- If the opening color clashes with the approved still, start from the same main background color as the still.
- If the final frame drifts too far from the still, strengthen the instruction that the approved still is the completed final composition.
- If generated audio lacks collage SFX but the visual is otherwise strong, ask the user whether to keep it, regenerate with stronger SFX direction, or add/post-sync SFX.
- If generated audio contains unwanted BGM or voiceover, remove or regenerate it before final delivery.
- If precise layer control is required, recommend a layered animation or editing workflow instead of continuing this Skill.


---

## FILE: skills/paper-collage-explainer-generator/meta.yaml (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/paper-collage-explainer-generator/meta.yaml)

display-name-zh: 纸拼贴讲解动画生成器
version: 0.3.9
tag-en: "Education"
tag-cn: "教育"
complete-tags-en:
  - "Education / Full Production"
  - "Animation / Creative Generation"
  - "Creative & Experimental / Creative Generation"
complete-tags-cn:
  - "教育 / 成片制作"
  - "动画 / 创作生成"
  - "创意实验 / 创作生成"
summary-en: "Turn narration or abstract ideas into tactile paper-collage explainer animations."
summary-cn: "基于文案或抽象概念，完成隐喻设计、拼贴分镜和动画生成，适用于讲解类内容。"
desc-en: "For creators, educators, and social-video editors who need a tactile visual language for narration, knowledge points, opinions, or abstract topics. Users provide source copy, story beats, or a core concept, with optional aspect ratio, duration, palette, and audio needs. The Skill extracts meaning, proposes visual metaphors, prepares a production plan and storyboard, generates approval-ready halftone collage stills, then creates stop-motion clips with paper movement and tactile sound effects. It delivers cohesive paper-collage animation assets or an assembled explainer video, and is not intended for realistic presenter ads, precisely editable layer files, complex typography systems, or prompt-only handoff."
desc-cn: "面向内容创作者、教育讲解者和社交视频编辑，用触感纸拼贴语言表现口播句、知识点、观点或抽象主题。用户提供短文案、故事节点或核心概念，也可补充画幅、时长、色调和音频需求。Skill 会提炼含义与视觉隐喻，制定制作方案和分镜，生成可确认的半调纸拼贴静帧，再制作带纸片滑入、弹入、轻敲、压平和摩擦声的停格动画片段。最终交付统一的纸拼贴动画素材或完整讲解视频，不适合真人口播广告、精确可编辑图层、复杂文字排版或仅输出提示词。"
author-en: "MiniMax Hub"
author-cn: "MiniMax Hub"
source: official-featured


---

## FILE: skills/papercraft-stop-motion-explainer/SKILL.md (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/papercraft-stop-motion-explainer/SKILL.md)

---
name: papercraft-stop-motion-explainer
description: For creators explaining science, education, or general knowledge through tactile handmade papercraft visuals. Users provide a topic, core knowledge points, or source material and may specify audience, duration, aspect ratio, and deliverable type. The Skill extracts the learning goal and visual metaphor, proposes creative directions, designs paper characters, layered diorama sets, and props, creates preview concepts plus image and video prompts, and plans storyboards, camera movement, transitions, and sound with staged approvals and review checklists. It outputs a production-ready papercraft stop-motion explainer package, or selected assets such as still prompts, image-series prompts, short-video prompts, or storyboards. Best for cut-paper, pop-up-book, layered diorama, and miniature stop-motion explainers; not for standard 2D cartoons, line doodles, live action, or explainers without a paper-art look.
compatibility: Requires the MiniMax Hub agent (canvas workspace and MiniMax H3 generation); not portable to generic agent harnesses.
---

# Papercraft Stop-Motion Explainer

Create a complete papercraft stop-motion explainer package from a science, education, or knowledge topic. The output is a production-ready creative plan: style rules, character and set design, asset planning, prompts, storyboard structures, motion language, sound direction, negative prompts, and review criteria.

Use this Skill when the user wants a tactile handmade explainer style: layered paper, cardboard cutouts, miniature diorama, stop-motion puppet movement, pop-up-book staging, paper props, and physical shadows. The default assumption is that the user wants a complete explainer video package. Prompts, storyboards, asset plans, and motion notes are reviewable production assets inside that complete process, not isolated deliverables unless the user says so.

When this Skill proceeds from planning/prompting into actual video generation, use MiniMax-H3 as the default video generation model. Treat MiniMax-H3 as the default unless the user explicitly selects another available model or MiniMax-H3 is unavailable.


## Canvas Document Delivery Rule

For complete video packages, production plans, storyboards, or any long multi-section planning output, write the full production document to the canvas as a text document. In the chat reply, give only a brief summary, the document filename, and the next recommended action. Do not paste long production plans, large storyboard tables, full prompt libraries, or full checklists into the conversation unless the user explicitly asks to see them in chat.

Use chat for concise guidance and decision points; use the canvas document as the source of the detailed plan. When the user asks to revise the plan, update the existing canvas document rather than creating a duplicate unless a new version is intentionally needed.

## Visual Depth and Papercraft Motion Priority

Every visual plan, preview prompt, video prompt, and review pass must prioritize both layered image depth and unmistakable papercraft stop-motion qualities.

Required visual-depth emphasis:

- Build scenes with clear foreground, midground, background, and far-background planes.
- Use foreground occluders such as paper leaves, clouds, frames, curtains, props, or cutout silhouettes to create depth.
- Keep the main knowledge object or focal action in a readable middle plane.
- Add background parallax, small environmental motions, and layer separation instead of flat static backdrops.
- Use circular, diagonal, tunnel, pop-up, or cross-section compositions when they help the viewer read depth.

Required papercraft stop-motion emphasis:

- Make all visible objects feel physically made from paper: layered cardboard, cut edges, paper fibers, folds, seams, tabs, brads, joints, pull-tabs, sliders, rotating discs, and visible thickness.
- Motion should feel frame-by-frame and hand-manipulated: small stepped movements, tiny pauses, slight rebounds, hinged gestures, sliding paper mechanisms, page flips, pull-tab reveals, and paper pieces settling.
- Background elements should also be paper mechanisms when possible: paper clouds sliding on rails, paper moons/discs moving on tracks, layered scenery shifting in parallax, paper leaves falling, paper lamps swaying, and cardboard doors opening.
- Avoid smooth CG motion, plastic 3D surfaces, flat vector backdrops, overly static backgrounds, and large character motion that breaks the miniature stop-motion feel.

## Lightweight Request Bypass

If the user explicitly asks for only one production asset, do not force the full 18-step package workflow. Route directly to the relevant step while preserving the papercraft style rules.

Use these shortcuts:

- Single-image prompt only: run STEP 1, STEP 2, STEP 9, and STEP 17.
- Image-series prompts only: run STEP 1, STEP 2, STEP 10, and STEP 17.
- 5-second image-to-video prompt only: run STEP 1, STEP 2, STEP 11, and STEP 17.
- Storyboard only: run STEP 1, STEP 2, STEP 12, STEP 13, STEP 14, STEP 15, and STEP 16.
- Creative directions only: run STEP 1, STEP 2, and STEP 3.

Use the full phased confirmation workflow only when the user asks for a complete video package, a full production plan, or does not specify a narrower deliverable.

## Interaction Rule: Confirmation Cards Between Phases

After each major phase, pause and ask the user with a selection card before moving forward. All polling, multi-choice decisions, confirmations, continue/revise/stop gates, and user decision points must use cards rather than plain text asking the user to reply with a number. The card must give practical next-step choices, not an open-ended question. After the creative directions phase, the next confirmation card must ask the user to choose the target video duration before detailed character, scene, preview, prompt, or storyboard work continues.

Use choices like:

1. Continue to the next phase with the current direction.
2. Revise the current phase before continuing.
3. Switch to one of the other creative directions.
4. Jump to a specific production asset: visual preview images, single-image prompt, image-series prompts, 5-second image-to-video prompt, or storyboard.
5. Stop here and export the current package.

Keep each card short. Include the recommended option first when the current result is strong. This staged confirmation protects the full-video workflow: the user can adjust each asset before it becomes the basis for the next step.

## STEP 1: Understand the Input

Analyze the user's topic and production goal. Preserve the user's domain words and do not simplify the science into a different topic.

Output a concise understanding block:

- Topic or knowledge point
- Target audience: children, general audience, classroom, social media, brand education, or specialist audience
- Intended duration: 15s, 30s, 60s, or unspecified
- Platform or format when stated
- Required output type: single image, image series, 5-second image-to-video prompt, storyboard, or full package
- Core learning outcome: the one sentence the viewer should remember
- Visual metaphor: how the concept becomes a paper object, model, puppet, layer, path, or mechanism

If crucial information is missing, choose reasonable defaults instead of stopping: general audience, 30-second short, horizontal 16:9 video by default unless the user specifies another ratio, and one paper narrator plus one core paper model.

## STEP 2: Summarize the Style DNA

Before designing the video, state the style rules that must stay consistent.

Include these traits:

- Handmade papercraft stop-motion look
- Miniature paper diorama or shadow-box stage
- Layered cardboard cutouts with visible thickness
- Matte tactile paper textures, fibers, folds, seams, torn or cut edges
- Paper puppet characters built from separate overlapping parts
- Real physical drop shadows between layers
- Multi-plane foreground, midground, background, and far background
- Macro miniature photography, slight depth of field, and 2.5D parallax
- Educational labels, arrows, cards, and simple symbols made from paper
- Clear foreground / midground / background / far-background depth, with foreground occlusion and readable parallax
- Background layers with paper-mechanism motion, not static painted scenery
- Stop-motion motion language: stepped movement, tiny pauses, slight rebounds, hinged joints, pull-tabs, sliders, rotating discs, paper pieces settling

Explain the reason: the paper medium turns abstract science into touchable objects and makes layered explanation easy to understand. The scene must feel physically built and animated frame by frame, not merely illustrated in a paper-like texture.

## STEP 3: Propose 3-5 Creative Directions

Generate 3 to 5 distinct concepts. Each direction must explain the same topic through a different visual metaphor or narrative structure.

For each direction, provide:

- Title
- Core idea
- Explainer angle
- Paper visual metaphor
- Best duration
- Best audience or platform
- Signature visual moment
- Risk or limitation

After presenting directions, ask the user to choose both the creative direction and the target duration. Duration options should be concise: 15s quick version, 30s standard version, 60s full version, or custom duration. Do not proceed into detailed assets until a duration is chosen.

Recommended direction archetypes:

1. Pop-up-book journey: each page reveals one layer of the concept.
2. Paper scientist laboratory: a paper host demonstrates the mechanism on a table.
3. Layered cross-section model: a paper object opens like a sectional diagram.
4. Miniature nature theater: ecological, geographic, or astronomical topics unfold in a paper landscape.
5. Paper mechanism board: gears, arrows, sliders, labels, and moving parts explain cause and effect.

## STEP 4: Design Paper Characters

Design characters only when they help communication. A character may be a host, assistant, animal guide, personified molecule, immune cell, planet, machine part, or natural force.

For each character, output:

- Name and role in the explanation
- Shape language and proportions
- Paper construction: cutout face, layered hair or body, cardboard limbs, joint nodes, tabs, brads, or folded parts
- Expression system: simple paper eyes, eyebrows, mouth shapes, replaceable emotion cards
- Movement style: stop-motion nudges, hinged arm gestures, pointing, bouncing, sliding, or flip-card expressions
- Reusable identity details for image series consistency

Keep characters simple enough to remain readable in short videos.

## STEP 5: Design Paper Scenes

Treat every scene as a physical paper stage, not a flat background.

For each scene, output:

- Scene name
- Scientific purpose
- Foreground elements
- Midground subject and action
- Background environment
- Far-background board or sky
- Movable paper mechanisms
- Educational information carriers: labels, arrows, charts, measurement tags, captions, or cards

Use at least four depth planes whenever possible. Keep the main knowledge model separated from the background through spacing and shadow.

## STEP 6: Plan Layered Diorama Staging

Create a multi-plane staging table.

Required columns:

- Layer number
- Plane role
- Paper elements
- Material and edge treatment
- Motion or parallax behavior
- Shadow interaction
- Knowledge function

Rules:

- Use 4-7 layers for most shots.
- Place reading-heavy labels on stable layers.
- Let foreground elements partially occlude the stage for miniature realism.
- Keep the core concept in the midground, where attention is strongest.
- Use layer separation to explain hierarchy, sequence, anatomy, causality, or scale.

## STEP 7: Plan Prop and Asset Library

Create an asset list grouped by function.

Asset groups:

- Host and character assets
- Core scientific object assets
- Explanation props: arrows, labels, cards, charts, magnifiers, rulers, meters
- Scene props: lab bench, hills, books, shelves, clouds, stars, trees, water, tubes, gears
- Motion props: paper strips, sliders, rotating discs, pop-up tabs, pull-out layers, paper confetti
- Sound cue props when useful: page flip, scissors, paper rustle, click, pop, tape peel

For each asset, define:

- Name
- Purpose
- Paper material
- Layer placement
- Static or movable
- Appearance timing
- Consistency notes

## STEP 8: Plan or Generate 1-3 Visual Preview Images

Before writing final prompts or storyboards, plan 1 to 3 visual preview images based on the approved creative direction, character design, scene design, and layered staging. If image generation is available and the user wants actual previews, generate them; otherwise provide preview briefs and prompts. These previews are for style and concept confirmation, not final production frames.

Default preview ratio follows the project video ratio: horizontal 16:9 unless the user specifies another ratio. If the user requested vertical, square, or another format, use that ratio consistently for previews and later video planning.

For each preview, define:

- Preview number and purpose
- Which approved direction, character, or scene it visualizes
- Key visual focus: overall world, character look, mechanism clarity, cultural atmosphere, or ending memory image
- What the user should judge before continuing

Recommended preview set:

1. Overall world preview: shows the full papercraft diorama, main character, core object, and atmosphere.
2. Mechanism preview: shows the science model, paper arrows, labels, and movable explanatory props.
3. Emotional or ending preview: shows the cultural connection, summary card, or final memory image.

After presenting previews, pause with a confirmation card. Offer choices such as: continue to prompt writing, revise preview 1, revise preview 2, revise preview 3, reduce to one visual direction, or switch creative direction. Do not write final single-image, image-series, or 5-second image-to-video prompts until the user confirms the visual preview direction.

## STEP 9: Write Single-Image Prompt

Create a prompt for one concept image or key visual.

The prompt must include:

- User's topic and core object
- Papercraft stop-motion style
- Miniature diorama stage
- Layered cardboard cutouts
- Visible paper fibers, folds, seams, cut edges, and thickness
- Real physical shadows between layers
- Macro miniature photography and slight depth of field
- Educational paper labels, arrows, or cards when needed

Avoid overloading the image with every knowledge detail. One image should communicate one main idea.

## STEP 10: Write Image-Series Prompts

Create a sequence of prompts when the user needs multiple keyframes or a storyboard image set.

For each image, provide:

- Image number and purpose
- What changes from the previous image
- Shared style anchors
- Character and scene consistency anchors
- Prompt

Series rules:

- Keep the same paper material language, light direction, depth planes, and character identity.
- Each image should explain one step of the knowledge point.
- Preserve domain-specific words from the user's topic in every prompt.
- Use the same aspect ratio across the series unless the user requests variations.

## STEP 11: Write 5-Second Image-to-Video Prompt

Create a short prompt that animates a single reference image while preserving the paper style.

Include:

- Preserve paper texture, cut edges, layered set, and physical shadows
- Slow push-in, gentle pan, or parallax move
- Small stop-motion-like puppet gestures
- Paper arrows, labels, sliders, or mechanisms moving slightly
- Foreground and background parallax
- Avoid smooth CG transformation, melting, plastic surfaces, or high-speed camera moves

Keep motion limited and physically plausible for paper objects.

## STEP 12: Create Storyboard for the Chosen Duration

Use the duration chosen after the creative directions phase. Do not present three full storyboard versions by default. First give a brief content overview, then provide a concise storyboard table for the chosen duration only.

The content overview should be short and clear:

- One-sentence video premise
- Target duration and ratio
- Main visual structure
- Knowledge path: hook → explanation → example or cultural connection → memory sentence

Storyboard table rules:

- Keep text compact. Avoid long paragraphs inside table cells.
- Use only the shots needed for the chosen duration.
- 15s: usually 4 shots.
- 30s: usually 5-6 shots.
- 60s: usually 7-9 shots.
- Each shot explains one knowledge beat.

Required columns:

- Time
- Knowledge beat
- Visual action
- Paper movement
- Camera / transition
- Sound cue

After the table, ask the user with a confirmation card: continue to editing rhythm and camera rules, revise storyboard, change duration, or return to visual previews.

## STEP 13: Define Editing Rhythm

Set the rhythm according to duration and educational clarity.

Guidelines:

- 15s: 4-6 shots, fast but readable, 2-4 seconds per shot.
- 30s: 6-8 shots, 3-5 seconds per shot.
- 60s: 8-12 shots, 4-7 seconds per shot.
- Pause briefly when a label card appears.
- Do not cut before the viewer understands the paper mechanism.
- Use rhythmic paper movements instead of aggressive digital edits.

## STEP 14: Define Camera Rules

Use camera movement that feels like filming a miniature paper stage.

Recommended moves:

- Slow push-in to enter the paper world
- Lateral pan with multi-plane parallax
- Static medium shot for host explanation
- Macro close-up for paper texture and key props
- Slight top-down angle for cross-section diagrams
- Layer pass-through when moving into internal structures

Avoid:

- High-speed flying camera
- Full 360-degree orbit unless the set is explicitly built for it
- Digital glitch motion
- Liquid morphing
- Hyper-real CG camera behavior

## STEP 15: Define Transitions

Use transitions that obey paper physics.

Recommended transitions:

- Page flip
- Pop-up-book unfold
- Pull-tab slide
- Paper label wipe
- Paper cloud pass
- Circular paper mask
- Cut-paper door opening
- Cross-section layer split
- Paper confetti burst
- Tape or sticker reveal

Avoid electronic scanlines, neon glitches, glass shatter, metallic wipes, and sci-fi particle transitions unless the user explicitly requests a contrast effect.

## STEP 16: Define Sound Design

Build a tactile handmade sound palette.

Recommended sound cues:

- Paper flip
- Scissor snip
- Cardboard slide
- Paper rustle
- Small wooden click
- Soft pop
- Tape peel
- Puppet joint tap
- Box opening
- Paper confetti scatter

Music direction:

- Match the BGM to the user's topic, culture, and emotional tone, not only to the generic papercraft style.
- For culturally specific topics, design BGM around the topic's relevant musical language. For example, a Mid-Autumn Festival explainer should lean toward restrained Chinese traditional color: guzheng, pipa, bamboo flute or xiao, light percussion, soft strings, and enough silence for narration.
- For science or classroom topics without a strong cultural identity, use light marimba, xylophone, pizzicato strings, soft percussion, and a warm educational tone.
- Keep music light under narration and apply ducking in the final mix.
- Avoid heavy cinematic bass, aggressive EDM, futuristic textures, or overdramatic scoring unless requested.

Paper-motion SFX direction:

- Design SFX from the actual video motion beats: page flip, paper door opening, paper rail slide, card flip, cardboard drawer pull, paper box opening, paper layer stack, soft pop, puppet joint tap, and paper rustle.
- Use SFX sparingly as tactile accents. They should make paper motion feel physical, not become exaggerated cartoon sounds.
- Map SFX to the storyboard timeline before final mixing so action, narration, and sound reinforce each other.

## STEP 17: Generate and Clean Voiceover Audio

When the workflow includes narration, generate the voiceover after the script is confirmed. Then check the generated audio before final assembly.

Voiceover rules:

- Match voice tone to the topic and audience: warm, clear, and gentle for family or education topics; more neutral for classroom or technical explainers.
- Compare generated voiceover duration with the chosen video duration. If the deviation is over 20%, ask the user with a card whether to shorten the script, extend the video, or accept the mismatch.
- Listen for tail noise, clicks, breath artifacts, abrupt cutoffs, or model residue at the end.
- If tail noise exists, repair the existing audio first: trim the noisy tail and add a short fade-out. Prefer repair over regenerating when the voice performance is otherwise good.
- If regenerating, explicitly request a clean ending with no tail noise, but still verify after generation.

## STEP 18: Provide Negative Prompts

Always include a concise negative prompt block to protect the style.

Recommended negatives:

- smooth plastic 3D
- glossy CG render
- live-action realism
- flat vector illustration
- generic cartoon with no paper texture
- metallic sci-fi surfaces
- glass material
- cyberpunk neon
- digital glitch effects
- oil painting strokes
- realistic hair or skin pores
- no paper fibers
- no cut edges
- no layer shadows
- overly smooth edges
- high-speed camera orbit
- melting or liquid morphing

## STEP 19: Run Review Checklist

End with a checklist. Mark problems only when they matter to the user's requested output.

### Style checklist

- Paper fibers, folds, seams, and cut edges are visible.
- Cardboard thickness is visible.
- Layers cast physical shadows.
- The image feels like miniature photography, not flat illustration.
- The style avoids plastic CG and live-action realism.

### Character checklist

- Characters read as paper puppets.
- Faces and expressions are cut-paper simple.
- Joints and limbs feel assembled from separate parts.
- Motion remains stop-motion-like and physically plausible.

### Scene checklist

- Foreground, midground, background, and far background are clear.
- Props are paper-made and support the explanation.
- Labels, arrows, and cards do not block the core model.
- Each shot explains only one knowledge beat.

### Video checklist

- Camera moves match miniature filming.
- Transitions follow paper physics.
- Sound cues match paper, cardboard, and puppet movement.
- The storyboard duration fits the target length.
- The final memory sentence is clear.

### Audio checklist

- Voiceover duration fits the chosen video duration or the mismatch has been resolved with user confirmation.
- Voiceover ending is clean, with no tail noise, click, breath residue, or abrupt cutoff.
- BGM matches the topic's culture and emotional tone, not only the generic papercraft style.
- BGM stays under narration and does not compete with spoken information.
- Paper-motion SFX align with actual visual motion beats and remain subtle.


---

## FILE: skills/papercraft-stop-motion-explainer/meta.yaml (https://github.com/MiniMax-AI/MiniMax-H3/blob/HEAD/skills/papercraft-stop-motion-explainer/meta.yaml)

display-name-zh: 纸艺定格科普视频生成器
version: 0.6.5
tag-en: Education
tag-cn: 教育
complete-tags-en:
- Education / Creative Generation
- Animation / Creative Generation
- Creative & Experimental / Creative Generation
complete-tags-cn:
- 教育 / 创作生成
- 动画 / 创作生成
- 创意实验 / 创作生成
summary-en: Build a production-ready papercraft stop-motion explainer package from a knowledge topic.
summary-cn: 根据知识主题，完成纸偶、分层布景、分镜与提示词设计，输出可执行的纸艺定格科普视频创作包。
desc-en: For creators explaining science, education, or general knowledge through tactile handmade papercraft visuals. Users provide a topic, core knowledge points, or source material and may specify audience, duration, aspect ratio, and deliverable type. The Skill extracts the learning goal and visual metaphor, proposes creative directions, designs paper characters, layered diorama sets, and props, creates preview concepts plus image and video prompts, and plans storyboards, camera movement, transitions, and sound with staged approvals and review checklists. It outputs a production-ready papercraft stop-motion explainer package, or selected assets such as still prompts, image-series prompts, short-video prompts, or storyboards. Best for cut-paper, pop-up-book, layered diorama, and miniature stop-motion explainers; not for standard 2D cartoons, line doodles, live action, or explainers without a paper-art look.
desc-cn: 面向希望用手工纸艺视觉讲解科学、教育或泛知识内容的创作者。用户需提供主题、核心知识点或原始材料，并可指定受众、时长、画幅和交付类型。Skill 会先提炼学习目标与视觉隐喻，提出创意方向，设计纸偶角色、分层纸雕布景和道具资产，制作预览概念、图像与视频提示词，规划分镜、运镜、转场和声音，并通过阶段确认与审核清单控制质量。最终输出可直接进入制作的纸艺定格科普视频创作包，也可按需交付单图、系列图、短视频提示词或分镜。适用于纸雕、剪纸、立体书和微缩定格讲解，不适用于普通二维卡通、线稿涂鸦、真人写实视频或无纸艺质感的标准科普。
author-en: MiniMax Hub
author-cn: MiniMax Hub
source: official
