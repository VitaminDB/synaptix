# Source: https://www.jxp.com/minimax/blog/minimax-h3-prompt-guide

Fetched: 2026-11-28 (curl + w3m text dump)

---

  • Language
      □ English Current language
      □ 日本語
      □ 한국어
  • My Account
  • Logout

Login
(Open navigation menu)
MiniMaxMiniMax H3 MaxNewMiniMax H3Hailuo 2.3MiniMax Blog
JXPMiniMaxMiniMax BlogMiniMax H3 Prompt Guide: Prompts, Exampl...

MiniMax H3 Prompt Guide: Prompts, Examples & Best Practices

Learn MiniMax H3 prompting with the field structure, shot timestamps, camera syntax, dialogue tags, reference labels, and copy-ready examples for text, image, product, and cinematic video.

MiniMax H3 Prompt Guide: Prompts, Examples & Best Practices
JXP TeamAugust 5, 202614 min read
Contents
What MiniMax H3 Gives You to Work WithMiniMax H3 Prompt Guide: The Quick FormulaThe Four MiniMax H3 Input ModesThe Three Core MiniMax H3 Prompt FieldsHow to Write Better MiniMax H3 PromptsMiniMax H3
Text-to-Video Prompt ExampleMiniMax H3 Image-to-Video Prompt ExamplesMiniMax H3 Character Consistency PromptMiniMax H3 Product Video PromptMiniMax H3 Camera Movement PromptMiniMax H3 Reference Prompt
GuideMiniMax H3 Prompt TemplateCommon MiniMax H3 Prompt MistakesFAQFinal MiniMax H3 Prompt Guide Checklist
Contents
What MiniMax H3 Gives You to Work WithMiniMax H3 Prompt Guide: The Quick FormulaThe Four MiniMax H3 Input ModesThe Three Core MiniMax H3 Prompt FieldsHow to Write Better MiniMax H3 PromptsMiniMax H3
Text-to-Video Prompt ExampleMiniMax H3 Image-to-Video Prompt ExamplesMiniMax H3 Character Consistency PromptMiniMax H3 Product Video PromptMiniMax H3 Camera Movement PromptMiniMax H3 Reference Prompt
GuideMiniMax H3 Prompt TemplateCommon MiniMax H3 Prompt MistakesFAQFinal MiniMax H3 Prompt Guide Checklist

A useful MiniMax H3 prompt guide should do more than collect cinematic adjectives. Because MiniMax H3 works with text, images, video, and audio in one request, an effective prompt has to describe
what happens over time, how the camera moves, how characters behave, and what the viewer should hear.

The strongest MiniMax H3 prompts read like short production plans. This guide covers the natural-language formula plus the field structure, shot timestamps, dialogue tags, and reference labels used
for multi-shot and reference-based work.

Try MiniMax H3 on JXP and turn one of the prompts below into a video.

What MiniMax H3 Gives You to Work With

MiniMax H3, widely called Hailuo 3.0 after the Hailuo AI app it runs in, is a multimodal video model rather than a text-to-video model with add-ons. Public listings describe output at up to 2K and
24fps, clips of roughly 5 to 15 seconds, and native stereo audio produced in the same pass as the picture, with a single generation reportedly accepting up to nine reference images, three video
clips, and three audio clips within a combined file cap.

Two of those numbers change how you write. Native audio means silence is never the default, so an undefined soundtrack is still a generated one. A 5 to 15 second ceiling means dialogue and action
must be budgeted, not just listed.

MiniMax H3 Prompt Guide: The Quick Formula

For most clips, start with this formula:

Subject + Action + Environment + Camera + Timing + Visual Style + Audio

Prompt:

A young woman in a long red coat walks through a rain-soaked Tokyo alley at night. She looks over her shoulder as neon signs reflect across the wet pavement. The camera begins with a medium tracking
shot behind her, moves slowly to her left side, then pushes into a close-up as she stops beneath a glowing sign. Realistic cinematic lighting, shallow depth of field, natural cloth movement, subtle
rain on the lens. Distant traffic, footsteps splashing through puddles, soft city ambience, and low atmospheric electronic music.

Compare that with:

Prompt:

A cinematic woman walking through Tokyo at night, beautiful lighting, realistic, 4K.

The second describes appearance but says little about motion, progression, camera behavior, or sound. A strong MiniMax H3 prompt needs verbs and temporal relationships.

The Four MiniMax H3 Input Modes

Before writing anything, decide what you are starting from. MiniMax H3 prompting distinguishes four base modes, and choosing the wrong one is a common reason a prompt underdelivers.

T2VA: Text to Video with Audio

No visual reference is supplied, so the prompt must establish the full audiovisual scene.

I2VA: Image to Video with Audio

The supplied image is the starting frame. Focus on what changes: motion, camera behavior, sound, and details that must stay stable.

FL2VA: First and Last Frame to Video

You provide both endpoints and describe the movement or transformation between them. This suits controlled transitions and product reveals.

L2VA: Last Frame to Video

You provide the final image and describe how the shot should arrive there, useful for packshots, logos, or fixed ending poses.

The Three Core MiniMax H3 Prompt Fields

For more structured MiniMax H3 prompting, a standard prompt can be organized around three jobs:

integrated_multimodal_description covers the main visual and narrative instructions: subjects, actions, camera movement, shot changes, dialogue, and in-scene events.

overall_soundscape covers environmental and physical sound: rain, traffic, footsteps, fabric, breathing, room tone, engines, glass, and other sounds that belong inside the scene.

non_diegetic_music covers music heard by the audience rather than the characters. Describe instrumentation, tempo, mood, and how the score changes over time.

A MiniMax H3 prompt does not have to look like a technical form, but thinking in these three layers prevents detailed visuals from being paired with undefined audio.

How to Write Better MiniMax H3 Prompts

1. Define the Main Subject in Every MiniMax H3 Prompt

Weak:

A girl walks outside.

Better:

A young woman with short black hair, a beige trench coat, and a small leather shoulder bag walks along a narrow European street.

Identity details matter most when the same character appears across several shots. For a MiniMax H3 image-to-video prompt, though, avoid re-describing the source image, which already establishes
appearance, clothing, and environment. Spend those words on movement instead.

2. Use Specific Motion Verbs

MiniMax H3 prompts need actions the model can animate: walks, turns, reaches, glances, opens, pours, accelerates, slows, lands, reacts.

Weak:

A man beside a sports car.

Better:

A man walks toward the sports car, runs his hand across the hood, opens the driver’s door, pauses, and looks toward the camera.

The second creates a sequence rather than a static composition.

3. Use MiniMax H3 Shot Timestamps When Order Matters

For a simple single-shot video, natural language is often enough. For a multi-shot sequence, timestamps make the order clearer.

A compact structure looks like this:

Prompt:

[Shot 1] Live-action, cinematic. A chef places a steak into a hot pan as oil begins to sizzle.

[Shot 2] At 00:03.500, the camera cuts to a close-up as he tilts the pan and spoons melted butter over the steak.

The first shot establishes style and action; later shots introduce new framing at a specific point. Use a cut only when the subject, space, or angle genuinely changes. For a tighter composition, a
push-in reads cleaner than another cut.

4. Describe MiniMax H3 Camera Motion as a Path

Camera language is one of the highest-value elements in any MiniMax H3 prompt guide. Useful directions include push in, pull out, pan, truck, tilt, pedestal, tracking, arc, static, POV, handheld
movement, and camera roll. When necessary, also describe amplitude and speed.

Weak:

Camera pushes in, small amplitude, slow speed, folded letter.

Better:

The camera pushes in with small amplitude at slow speed toward the folded letter resting in her hands.

Write camera movement as part of the action. Do not stack five unrelated movements into a short clip.

5. Use Visual Style to Support the Action

Style should reinforce the scene rather than replace it. Pair a specific direction such as cinematic realism, luxury commercial photography, documentary realism, anime-inspired animation, or natural
smartphone video with concrete lighting. In a MiniMax H3 prompt, “epic” or “cinematic” alone cannot substitute for action and camera direction.

6. Structure MiniMax H3 Dialogue and Native Audio

A strong MiniMax H3 dialogue prompt should identify who speaks and exactly what they say. Speaking characters can use stable IDs such as (S1) and (S2), while the spoken line sits inside a language
tag.

Prompt:

The young filmmaker with a calm, warm mid-range voice (S1) says: <d>[English] The hardest part isn't getting an idea. It's turning the idea into a shot.</d>

Keep the voice description outside the dialogue tag and the exact words inside it. Keep spoken lines short enough to fit naturally inside the clip.

For voiceover, state that the speech is off-screen and that the on-screen character does not lip-sync unless intended.

For on-screen wording such as signage, labels, or neon text, write the desired text exactly:

A red neon sign reading “OPEN LATE” glows above the doorway.

Then separate the sound layers: ambient sound for the environment, action sound for physical events, and music for instrumentation, tempo, intensity, and dynamic changes. Avoid vague labels such as
“epic music.”

7. Define the Ending of the MiniMax H3 Clip

When the final frame matters, say what it should look like.

The shot ends with the perfume bottle centered in frame, the camera completely still, and the label facing directly toward the viewer.

Ending instructions matter most for product videos, transitions, and clips built to cut into another shot.

MiniMax H3 Text-to-Video Prompt Example

A MiniMax H3 text-to-video prompt needs to establish the scene more completely because no starting image is available.

MiniMax H3 Cinematic Street Scene

Prompt:

[Shot 1] Live-action, cinematic. A lone man in a dark wool coat walks through a foggy London street just before sunrise. Streetlights glow faintly through the mist while an old red bus passes in the
distance. The camera tracks behind him at shoulder height.

[Shot 2] At 00:05.000, the shot transitions to a low-angle medium shot as he stops at the corner and looks toward an approaching taxi. Muted blue-gray palette, realistic fog, subtle film grain.

overall_soundscape: Soft footsteps, distant tires on wet pavement, light wind, faint engine noise.

non_diegetic_music: Restrained low strings at a slow tempo, swelling once as the taxi appears.

MiniMax H3 Image-to-Video Prompt Examples

The best MiniMax H3 image-to-video prompts describe what changes while protecting what should remain stable.

MiniMax H3 Portrait Animation Prompt

Prompt:

The supplied image is the first frame at 00:00. Preserve the person’s identity, hairstyle, clothing, lighting, and environment.

[Shot 1] She initially looks slightly away from the camera. A soft breeze moves several strands of hair. She turns toward the camera, smiles naturally, and blinks once. The camera pushes in with
small amplitude at slow speed. Keep facial proportions stable and motion understated.

overall_soundscape: Quiet room tone, faint traffic beyond the window.

non_diegetic_music: None.

MiniMax H3 Character Consistency Prompt

Character consistency improves when a MiniMax H3 prompt repeats a few strong identity anchors instead of rewriting a long physical description in every scene.

Prompt:

Maintain the same character identity as the reference: the same facial structure, short wavy brown hair, green jacket, white shirt, and silver necklace.

[Shot 1] She enters a quiet record store and walks slowly beside a shelf of vinyl records while the camera tracks with her.

[Shot 2] At 00:06.000, the camera cuts to a close-up as she selects one record, turns it over in her hands, and smiles. Keep her facial identity, hairstyle, clothing, and proportions consistent
throughout.

overall_soundscape: Soft indoor ambience, sleeve rustle, subtle footsteps on wood.

non_diegetic_music: Faint vinyl-warm jazz.

For multiple clips, reuse the same identity anchors and change only location, action, and camera plan.

MiniMax H3 Product Video Prompt

Product advertising suits structured prompting because product integrity, motion, framing, and the final hero shot can all be specified.

Prompt:

[Shot 1] Premium commercial realism. A transparent luxury perfume bottle stands on glossy black stone surrounded by thin drifting mist. A narrow beam of warm light sweeps across the glass, revealing
reflections and condensation. The camera begins in extreme close-up on the cap, pulls back with large amplitude at slow speed, then arcs gently around the product. Keep the bottle shape, label
placement, colors, and proportions unchanged, and end with the bottle centered and the label facing the viewer.

overall_soundscape: Faint glass resonance and a soft droplet sliding down the bottle.

non_diegetic_music: Delicate glass-like chimes over a low warm pad, resolving on the final frame.

Open MiniMax H3 on JXP and test this prompt structure with your own product image.

MiniMax H3 Camera Movement Prompt

A camera prompt works best when it describes one continuous path instead of a list of cinematic terms.

Prompt:

[Shot 1] Warm interior tones against cool rain outside. The camera opens on a close-up of a steaming coffee cup, pulls back with large amplitude at slow speed to reveal a woman reading beside the
window, continues past her shoulder toward the glass, then rotates outward to show the rainy street beyond. One continuous move, no cuts.

overall_soundscape: Quiet cafe conversation, rain against the glass, cups and plates in the background.

non_diegetic_music: None.

Naming the order of the movements is what makes a MiniMax H3 camera prompt readable. Listing dolly, orbit, handheld, and zoom leaves the sequencing undefined.

MiniMax H3 Reference Prompt Guide

Reference-based generation is where precise labels become especially useful. A prompt may use one asset for identity, another for motion, and another for audio or editing rhythm.

Four useful reference labels are:

<Subject N> for a reusable visible subject such as a person, animal, product, outfit, setting, style, or pose.

<Picture N> for an image used as a literal visual frame, such as a first frame, last frame, or storyboard anchor.

<Video N> for a video used for editing relationships, continuation, motion, or cutting rhythm.

<Audio N> for an audio source used for voice, sound, beat, or another audio reference.

A reference prompt can also separate the task into sections such as subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, and non_diegetic_music.

Prompt:

subject_definitions: <Subject 1> is the woman in <Picture 1>, with a cropped denim jacket and shoulder-length auburn hair. <Video 1> is a handheld tracking shot through a market street. <Audio 1> is
a female voice recording.

summary: [reference generation + audio reference]

retention_analysis: <Subject 1> fully_preserved. <Video 1> attribute_transfer for camera movement and cutting rhythm only. <Audio 1> reference for vocal timbre.

detailed_description: Cinematic, slightly desaturated, handheld. [Shot 1] <Subject 1> walks through the market street while the camera tracks beside her at shoulder height using the movement rhythm
from <Video 1>.

overall_soundscape: Busy market ambience, footsteps, fabric movement, distant vendors.

non_diegetic_music: None.

Retention wording is what makes this work. Saying a subject is fully preserved while a clip contributes camera behavior only is far clearer than “use these references,” because it tells MiniMax H3
whether each asset controls identity, composition, motion, style, or audio.

MiniMax H3 Prompt Template

Use this reusable MiniMax H3 prompt template for multi-shot generations:

Prompt:

[Shot 1] [visual style]. [Main subject and identity] is in [environment], [first action]. The camera [shot type] and [movement with amplitude and speed].

[Shot 2] At [00:0X.000], the camera cuts to [new framing] as [second action or reaction].

[Final beat and ending composition.]

overall_soundscape: [ambient layer], [action sounds].

non_diegetic_music: [instrument], [tempo], [how dynamics change].

For image-to-video, add:

The supplied image is the first frame at 00:00. Preserve identity, clothing, composition, and lighting.

For product video, add:

Keep the product shape, packaging, colors, label placement, and proportions unchanged throughout.

For dialogue, add:

The [voice description] character (S1) says: <d>[English] ...</d>

Common MiniMax H3 Prompt Mistakes

Writing an Image Prompt Instead of a Video Prompt

“Beautiful cinematic woman in New York” describes a frame; “she exits a taxi, looks upward, and walks toward the building as the camera follows” describes a video.

Packing in Too Many Actions

Prioritize two or three beats instead of forcing many actions, camera changes, and long dialogue into a clip that may only run 5 to 15 seconds.

Depending on Style Words

“Masterpiece,” “epic,” “8K,” and “ultra cinematic” cannot replace motion, camera, lighting, and sound instructions.

Contradicting the Reference

Do not change clothing, identity, or scene details unless the change is intentional.

Ignoring Sound

Because audio is generated alongside the picture, leaving it undefined does not produce a quiet clip. Define ambience, physical effects, dialogue, and music where they matter.

Changing Everything at Once

When a take is close, revise one element only: verbs for motion, the camera path, the identity anchors, or the sound layers. Single-variable iteration is what makes MiniMax H3 results diagnosable.

Failing to Define Reference Roles

State what each asset should preserve or transfer. In MiniMax H3, a face reference, a motion reference, and an audio reference serve different jobs.

FAQ

What is the best MiniMax H3 prompt format?

Combine subject, action, environment, camera movement, timing, visual style, and audio. Simple scenes fit one paragraph; complex ones benefit from shot timestamps plus separate soundscape and music
lines.

How do you write a MiniMax H3 text-to-video prompt?

Describe subject and environment first, then chronological action, camera behavior, style, ambient sound, and music. Because text-to-video has no starting image, it needs more visual description than
other MiniMax H3 modes.

How do you write a MiniMax H3 image-to-video prompt?

State that the supplied image is the starting frame, then focus on movement, camera behavior, sound, and details that must stay stable. Avoid repeating what the image already shows.

Can MiniMax H3 generate audio from a prompt?

Yes. MiniMax H3 supports native stereo audio alongside the picture, so a prompt can specify dialogue, environmental sound, effects, and music. Because audio is produced in the same pass, leaving it
undescribed still yields generated sound.

How many references can one MiniMax H3 generation use?

Public listings describe support for up to nine reference images, three reference video clips, and three audio clips within a combined file cap, with audio generally needing to accompany at least one
image or video rather than being submitted alone.

Is MiniMax H3 the same model as Hailuo 3.0?

MiniMax H3 is the model name, while Hailuo 3.0, sometimes written Hailuo 03, is the name commonly used after the Hailuo AI app. Both refer to the same model, which follows Hailuo 2.3 in the same
line.

How do you improve character consistency in MiniMax H3?

Reuse the strongest identity anchors across every MiniMax H3 prompt in the set: facial identity, hairstyle, clothing, proportions, and distinctive accessories. State clearly which details must remain
preserved.

Should MiniMax H3 prompts include timestamps?

Use them when a clip contains multiple shots or when actions must happen in sequence. A single-shot clip usually does not need them.

Final MiniMax H3 Prompt Guide Checklist

Before generating, confirm your MiniMax H3 prompt defines the input mode, the main subject, the opening and following actions, a coherent camera path, timestamps where sequencing matters, visual
style and lighting, identity or product constraints, dialogue with speaker identity, on-screen text, ambient and physical sound, music direction, reference roles, and the desired ending.

The habit this MiniMax H3 prompt guide is really recommending is thinking in time, movement, camera, references, and sound rather than describing a beautiful frame. Start with a clear subject, give
it an action, sequence that action, tell the camera how to observe it, preserve the details that matter, define what each reference contributes, then add the audio that completes the scene.

That structure carries across MiniMax H3 text-to-video prompts, image-to-video prompts, character consistency work, product ads, dialogue scenes, and reference-based generation alike.

Try these MiniMax H3 prompt examples on JXP and refine them for your own workflow.

(Share this article) X (Copy article link)
jxp ai video generator

JXP COM LIMITED.

support@jxp.com
Join our Discord
Follow us on X
(English — change language)English
EnglishCurrent language日本語한국어
Products

  • Android App (Beta)
  • iOS App
  • Mobile App
  • Brand Upgrade

About

  • How To Use
  • Inspiration
  • Prompt Guide
  • FAQ
  • Blog

Legal

  • Terms of Service
  • Privacy Policy
  • Refund Policy

Disclaimer: JXP provides AI-generated video and image outputs for creative and lawful use only. Results may be inaccurate, incomplete, or unsuitable for your intended purpose. You are solely
responsible for reviewing outputs and for compliance with applicable laws. This service is provided "as is" without warranties of any kind.
