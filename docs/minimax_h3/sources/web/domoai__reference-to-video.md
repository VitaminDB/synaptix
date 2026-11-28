# Source: https://domoai.app/blog/minimax-h3-reference-to-video

Fetched: 2026-11-28 (curl + w3m text dump)

---

August 21, 2026

14 min read

Try MiniMax H3!
minimax-h3-reference-to-video

Ask this post with:

PerplexityClaudeChatGPT
[Inko_AI_Ag]

InkoAI Agent

Meet InkoAI Agent, a creative assistant from DomoAI. Describe your idea, create images and videos, and refine the results just by chatting.

Try InkoAI Agent

MiniMax H3 reference-to-video works best when every source has a named job. Use a picture for appearance or composition, video for motion or camera, and audio for a sound signal. Then state what H3
should retain, transfer, change, or treat as a loose reference.

One asset can provide several subjects or attributes. The “one job” rule sets one primary responsibility for each source. It is a planning method, not a technical one-asset limit.

What Does Reference-to-Video Mean in MiniMax H3?

Reference-to-video, or Ref2VA, uses text plus reference images, videos, or audio to guide a new audiovisual clip. The sources can describe identity, appearance, style, movement, camera, timing,
voice, sound effects, or music.

H3 does not treat every file as a frame. A picture can define appearance, video can guide camera, and audio can guide timbre.

MiniMax's official full-reference prompt guide formalizes those relationships. It uses labels, retention markers, and a six-section planning format.

That format describes intended relationships. It does not guarantee exact identity, physics, lip sync, timing, text, or pixel-level reproduction.

Ref2VA vs FL2VA: Which H3 Mode Should You Use?

Choose FL2VA when a supplied image must occupy the first frame, last frame, or both. Choose Ref2VA when sources should guide attributes or behavior without serving as fixed endpoints.

                               Need                                             Recommended H3 mode                                             Why
Start from one opening image                                       FL2VA checkpoint in first-frame I2VA mode     The supplied image anchors the opening state
Land on one supplied ending image                                  FL2VA checkpoint in L2VA mode                 The prompt builds a path to the final state
Constrain both first and last images                               FL2VA checkpoint in first-and-last-frame mode Both endpoints are explicit
Use identity, style, motion, camera, or sound from several sources Ref2VA                                        Each source can receive a different reference role
Mix first/last-frame API roles with reference roles in one request Not supported in a single API request         The official API treats these role families as mutually exclusive

The Video Generation V2 API reference separates endpoint-frame roles from reference-media roles. One API request cannot mix those role families.

Prepare Only References That Have a Clear Purpose

Every reference adds another relationship that H3 must interpret.

Use this five-check filter before upload:

 1. Permission: You own the asset or have permission to use the identity, footage, audio, trademark, and creative work.
 2. Relevance: The file clearly shows the attribute you want to guide.
 3. Clarity: The important subject, movement, camera path, or audio event is easy to identify.
 4. Compatibility: The file meets the current local or API limits for your route.
 5. Priority: You can state what this asset controls when another source disagrees.

A clean packshot usually guides geometry better than a busy collage. A stable movement clip communicates camera intent better than a fast montage.

Plan the shot before gathering references. This AI storyboard guide helps separate composition, action, and camera decisions before animation.

Official H3 reference limits

MiniMax's current H3 repository and model documentation list these local Ref2VA limits:

  • Up to 9 reference images.
  • Up to 3 reference videos, each 2–15 seconds, with no more than 15 seconds total.
  • Up to 3 reference audio clips, each 2–15 seconds, with no more than 15 seconds total.
  • No more than 12 mixed local Ref2VA media files in total.
  • Reference audio must accompany an image or video in the local workflow. It cannot be the sole media input.

The hosted API requires a non-empty text prompt in every request. Its current request body limit is 64 MB.

API reference images can use JPG, JPEG, PNG, WEBP, HEIC, or HEIF. Reference video accepts MP4 or MOV, while reference audio accepts WAV or MP3.

API upload requirements can change, so check the current Video Generation V2 API reference before submitting a request.

MiniMax lists ComfyUI as one local inference option. Local input rules and hosted API request rules may differ.

Give Every Reference One Clear Primary Job

Create a Reference Job Sheet before writing the prompt. Give each file one primary job and, when needed, one secondary job.

Source or label                                         Good primary jobs                                                              What it does not guarantee
<Subject N>     Reusable person, object, environment, action, pose, effect, or style abstracted from sources       Perfect identity lock or a separate uploaded file
<Picture N>     Concrete frame, composition, storyboard anchor, spatial arrangement, or lighting anchor            Complete motion, temporal structure, or perfect geometry retention
<Video N>       Source edit, continuation, motion pattern, camera path, cuts, rhythm, or temporal structure        Guaranteed replacement, physics, or pixel-level copying
<Audio N>       Voice timbre, beat, dialogue signal, sound effect, music traits, continuity, or direct audio reuse Guaranteed lip sync or audio-only local Ref2VA input

Here is the worksheet format:

Asset ID  Type  Rights cleared?         Primary job            Secondary job         Retain, transfer, or change      Shot range     Conflict or acceptance check
Picture 1 Image Yes             Bottle appearance and label Material color      Retain geometry and matte-black cap   All shots  Label remains front-facing
Picture 2 Image Yes             Greenhouse environment      Dawn palette        Transfer setting, not its props       All shots  No second product appears
Video 1   Video Yes             Clockwise camera arc        Slow pacing         Transfer camera path only             Shot 1     Bottle does not inherit source shape
Audio 1   Audio Yes             Glass-chime texture         Final accent timing Reference texture, do not copy signal Shot 2     One clean chime after cap click

Acceptance checks make each role reviewable.

For product work, the product-video consistency guide adds geometry and label checks.

Resolve conflicts before prompting

This direction has no priority:

Use the silver cap from Picture 1 and the black cap from Picture 2. Keep both exact.

This version assigns responsibility:

Picture 1 controls the bottle geometry, amber glass, matte-black cap, and label. Picture 2 controls only the greenhouse environment and dawn lighting; ignore the container shown in Picture 2.

When a reference performance matters more than its performer, state that boundary explicitly: Video 1 controls motion and camera only; ignore its performer and setting.

Label Subjects, Pictures, Videos, and Audio Correctly

Keep label numbers stable across all six sections.

A Subject is an abstraction

<Subject N> means reusable or editable visible content, such as a person, product, environment, action, pose, interface, or style.

<Subject 1> is the square amber-glass perfume bottle whose geometry, matte-black cap, and cream "NORTHLINE" label come from <Picture 1>.

A Picture is a concrete frame or planning anchor

Use standalone <Picture N> when the image acts as a concrete frame or composition anchor.

<Picture 3> is the first-frame composition anchor for [Shot 1], defining the bottle position, plinth angle, and empty space on the right.

If an image only defines a subject, cite it inside the Subject definition.

A Video describes a whole-video relationship

Use <Video N> for source editing, continuation, camera, cuts, rhythm, or temporal structure. Extracted people and objects still need Subject labels.

Audio describes the signal and its job

Use <Audio N> for direct reuse or reference to voice, timing, beat, effects, or music traits. Audio and Video numbers remain independent.

Write the Six Parts of an H3 Full-Reference Plan

MiniMax's full-reference guide uses six sections in this order:

 1. subject_definitions
 2. summary
 3. retention_analysis
 4. detailed_description
 5. overall_soundscape
 6. non_diegetic_music

The hosted API requires text, but it does not require every user to hand-author this full rewrite. H3-Context-IR can process free-form multimodal input. Use the six sections as a precise planning
language, not a mandatory raw API schema.

The original example below applies MiniMax's documented format to a fictional product shot.

Example run card: Ref2VA · 8 seconds · 16:9 · three images · one video · one audio reference.

subject_definitions: <Subject 1> is the square amber-glass perfume bottle whose geometry, matte-black cap, and cream "NORTHLINE" label come from <Picture 1>. <Subject 2> is the greenhouse environment
from <Picture 2>, with wet black-basalt surfaces, tall fern leaves, glass roof panels, and cold blue dawn light. <Picture 3> is the first-frame composition anchor for [Shot 1], defining the bottle
position, plinth angle, and empty space on the right. <Video 1> provides the slow clockwise camera arc and measured pacing for [Shot 1], without providing product appearance. <Audio 1> is the
sound-texture reference for one clear glass chime heard after the cap click in [Shot 2]. summary: [keyframe completion + reference generation + audio reference] The target video presents <Subject 1>
inside <Subject 2>. It begins from the composition in <Picture 3>, follows the camera path from <Video 1>, and uses <Audio 1> only as a sound-texture reference for the final chime.
retention_analysis: <Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - the square amber-glass body, matte-black cap, cream "NORTHLINE" label, and front-facing product identity are
retained. <Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the wet basalt, fern leaves, glass roof, and cold blue dawn lighting are retained. <Picture 3> ([Shot 1] first frame):
fully_preserved - the opening bottle position, plinth angle, and right-side negative space are retained. <Video 1> (camera path and pacing): partially_preserved - its slow clockwise arc and measured
speed guide [Shot 1], while its source subjects and setting are excluded. <Audio 1>: reference - its clean glass-chime texture guides one newly generated accent without copying the original signal.
detailed_description: The target video uses a live-action cinematic product-film style with restrained movement and cool dawn lighting. [Shot 1] The shot begins from <Picture 3>. <Subject 1>, the
square amber-glass perfume bottle with a matte-black cap and cream "NORTHLINE" label, stands on the wet basalt plinth inside <Subject 2>. The camera follows the slow clockwise arc and measured pacing
referenced from <Video 1>. Cold blue light enters from the right through the greenhouse roof. Water beads slide down the amber glass. Fern leaves move slightly behind the product. The bottle stays
fixed, front-facing, and unchanged while the camera completes the arc. No second bottle, subtitle, or watermark appears. [Shot 2] At 00:05.000, the camera cuts to a static close-up of <Subject 1>. A
clean hand enters from the upper right, presses the matte-black cap down once, and releases it. The cap makes a dry click. A single newly generated glass chime follows, using the clear texture
referenced from <Audio 1> without copying its signal. A narrow amber reflection crosses the cream "NORTHLINE" label from left to right. The hand exits. The bottle remains centered and readable
through the final frame. overall_soundscape: Light rain taps on the glass roof throughout the video. Fern leaves brush softly in the background, while water drops strike the basalt and the cap
produces one dry mechanical click. non_diegetic_music: N/A

The plan states what each source controls and excludes.

Turn Reference Roles Into a Shot Timeline

Assign roles by shot because a reference may matter for only part of the clip.

Use this order for each shot:

 1. Establish the current composition and active subjects.
 2. Name the reference when its influence begins.
 3. Describe physical action and state changes.
 4. State camera motion and cut timing.
 5. Place dialogue or synchronized sound at its event.
 6. End with the state that the next shot needs.

The first shot has no timestamp. Later shots use [Shot N] At MM:SS.mmm, ... with increasing cut times. If two videos guide movement, assign body motion to one and camera to the other.

Decide Whether Audio Is Copied or Referenced

Choose the audio marker that matches the intended signal relationship.

    Marker                                   Plain-language meaning
fully_copy     Reuse the entire source audio as the complete final audio track
partially_copy Copy part of the timeline or selected layers, with changes elsewhere
reference      Generate new audio guided by timbre, rhythm, style, words, beat, or sound texture
weak_reference Keep only broad category or atmospheric similarity

When audio guides a target speaker, bind it to the same stable (Sx) speaker ID used in detailed_description. If only the timbre transfers, do not copy the source dialogue into the target.

Remember the local restriction: audio cannot be the only media reference for H3-Base-Ref2VA. Include an image or video.

Review the First Result With One Acceptance Rubric

Score each intended job as kept, partly kept, missing, or conflicted.

  Check                                           Question
Identity   Does the intended person or object remain recognizable across its shots?
Appearance Are the named colors, clothing, material, label, or style present?
Geometry   Did the product keep its defining shape and part relationships?
Motion     Does the requested action follow the intended path and pace?
Camera     Did the camera use the requested direction without inheriting unwanted source content?
Timing     Did each reference begin and stop in the intended shot range?
Sound      Was audio copied, referenced, or newly generated as planned?
Conflicts  Did two references compete for the same attribute?

Revise one relationship at a time. If only the camera failed, clarify Video 1's camera-only role and exclude its source product.

Why Might H3 Weaken or Ignore a Reference?

Check the official inputs and your role plan before blaming model behavior.

  • The job is implicit: Name the exact subject, attribute, shot range, and intended relationship.
  • Two sources conflict: Choose which asset controls that attribute and exclude the other source's version.
  • The source is irrelevant: Replace a busy or ambiguous file with one that displays the target job clearly.
  • The wrong mode is active: Use FL2VA for fixed endpoints and Ref2VA for multimodal guidance.
  • The checkpoint is wrong: Local mixed-reference work uses H3-Base-Ref2VA, not the FL2VA checkpoint.
  • The API roles conflict: Do not mix first/last-frame roles with reference-media roles.
  • An input exceeds a limit: Recheck counts, durations, formats, sizes, dimensions, frame rates, and request-body size.
  • The prompt asks for too much: Remove lower-priority references and test the core relationship first.

Cache, quantization, step count, nodes, and VRAM can affect a specific local setup. Their effects vary by configuration.

How 768p H3-Base and Regenerate-2K Fit Together

Local H3-Base produces 768p video with native stereo audio. The public release includes separate FL2VA and Ref2VA checkpoints.

The complete system also includes hosted H3-Context-IR and H3-Regenerate-2K. Regenerate-2K uses the 768p result plus the original context to regenerate at up to 2K.

Regenerate-2K is not a conventional upscaler, and it is not in the current open-weight release. Downloading Ref2VA alone therefore does not provide the complete local 2K workflow.

MiniMax H3 Reference-to-Video FAQ

What is the difference between H3 FL2VA and Ref2VA?

FL2VA anchors a first frame, last frame, or both. Ref2VA assigns identity, style, motion, camera, timing, or sound roles to reference images, video, and audio.

How many reference files can local H3 Ref2VA use?

The local model card lists up to 9 images, 3 videos, and 3 audio clips, with no more than 12 mixed files. Video and audio duration totals each cannot exceed 15 seconds.

Can audio be the only reference in MiniMax H3?

Not in the documented local Ref2VA workflow. Reference audio must accompany an image or video. The hosted API follows a separate set of request rules.

Do I need all six Ref2VA sections for the API?

No. The API requires text, but users do not have to submit all six sections manually. The structure is a precise planning method.

Can H3 copy movement from a reference video?

Video can guide a motion pattern, camera path, rhythm, or temporal structure. It should be treated as guidance, not guaranteed reproduction.

Can local H3 Ref2VA generate 2K video?

H3-Base generates 768p locally. The official 2K path depends on the hosted Regenerate-2K stage, which is not included in the current open-weight release.

Plan Your Next H3 Reference-to-Video Test

Build the Reference Job Sheet before your next Ref2VA request. Give each source one primary job, then review the first output against that job.

MiniMax H3 is now available in DomoAI through Omni Reference.

Recent articles
