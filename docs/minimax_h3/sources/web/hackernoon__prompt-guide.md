# Source: https://hackernoon.com/minimax-h3-prompt-guide-how-to-write-better-hailuo-30-video-prompts

Fetched: 2026-11-28 (curl + w3m text dump)

---

     

Zeiki

    Zeiki
    byZeiki|@wushi

    AI prompt engineer

    [                    ]Subscribe
    Story's Credibility
    Original Reporting
    AI-assisted

MiniMax H3 prompts work best when they describe a timed audiovisual sequence, not a pile of cinematic adjectives. This MiniMax H3 prompt guide turns that idea into a repeatable method for
text-to-video and reference-driven Hailuo 3.0 generation: define the opening, map the action, direct the camera, separate sound layers, and state what must stay consistent.

The key shift is simple: write the prompt like a timeline specification with a clear start, change, camera, sound, and ending.

The Short Answer: Prompt the Timeline, Not a Still Image

Many AI video prompts begin with a subject, a style, and a mood. That suggests a frame, but video also needs order, causality, camera behavior, sound, and an ending state.

MiniMax H3, often described as Hailuo 3.0, is designed for short audiovisual clips with text, image, video, and audio context. The current Hailuo workspace supports 4-15 second clips, up to 2K
output, native stereo sound, and up to 12 mixed reference items. Structure still matters.

Use this six-part planning pass before writing the final text:

 1. Context: Establish the opening composition and references.
 2. Timeline: Describe actions in order, including the ending state.
 3. Camera: Name movement, scale, and speed when they matter.
 4. Sound: Separate dialogue, physical sound, ambience, and music.
 5. Constraints: State what must remain stable.
 6. QA: Decide what you will inspect before acceptance.

Choose the Right MiniMax H3 Prompt Mode

The prompt structure changes when a reference image anchors the video. MiniMax H3 uses four useful timeline modes:

Mode          Reference anchor                              Prompt opening                                              Best use

T2VA  No frame reference               Start with the three core fields                          Build the complete audiovisual sequence from text

I2VA  One image at 0.00 seconds        State that the first picture is fully referenced          Animate a supplied starting frame forward

FL2VA First and last images            Align Picture 1 with the start and Picture 2 with the end Describe a continuous transformation between two states

L2VA  One image at the final timestamp Align the picture with the effective end time             Build a plausible path that lands on a final frame

For T2VA, begin directly with the core fields. For the reference modes, put the frame-alignment instruction first, followed by a blank line. Describe the path between reference states: subject
movement, object changes, camera travel, and the final composition.

FL2VA usually benefits from one continuous shot. Use multiple shots only when the cut itself is part of the creative intent.

Build the Core Prompt with Three Audiovisual Fields

The H3 prompt format separates the main timeline from the two sound layers that are easiest to confuse:

              Field                                                               Put here                                                                    Keep out of it

integrated_multimodal_description Visual style, composition, actions, shots, speakers, dialogue, singing, visible text, and diegetic sound A separate paragraph about background music

overall_soundscape                Ambience, footsteps, impacts, fabric, wind, rain, breathing, and other physical or non-verbal sounds     Repeating full dialogue or lyrics

non_diegetic_music                Instrumentation, tempo, rhythm, and dynamic changes in music only the audience hears                     Music playing from a radio or phone inside the scene

The first field is the main body. Start [Shot 1] without a timestamp; later shots use increasing cut times. A cut should add information. If only the angle changes, camera motion is clearer.

Describe camera motion naturally inside the shot, adding amplitude or speed only when those details matter:

integrated_multimodal_description: [Shot 1] Live-action, a medium-wide shot frames a cyclist waiting at a wet intersection. The camera tracks right with small amplitude at slow speed as she releases the brake and sends water from the rear tire. The cyclist (S1) says: <d>[English] Take the long way home.</d> [Shot 2] At 00:04.500, the camera cuts to the tire crossing the painted line.

overall_soundscape: Rain taps on the road and the chain clicks under load. Tires hiss through shallow water while distant traffic remains soft.

non_diegetic_music: Sparse electronic pulses at a slow tempo, joined by a restrained bass note.

Use stable speaker IDs such as (S1) and (S2). Put only exact spoken content inside <d>[Language] ... </d>. Put visible signs, labels, subtitles, or end cards in English double quotation marks, and
keep them short enough to inspect.

A visual map of the structured MiniMax H3 prompt workflowA visual map of the structured MiniMax H3 prompt workflow

Use References as Roles, Not a Pile of Assets

More references do not automatically produce more control. Give each one a distinct role:

  • Identity: Which person, character, animal, or product must remain recognizable?
  • Scene: Which environment, lighting, or composition should carry through?
  • Motion: Which video supplies movement, posture, rhythm, or camera language?
  • Voice and style: Which audio or visual source defines delivery, texture, or pacing?

The current Hailuo workspace accepts up to 12 reference items in Omni Reference mode. Treat that as a boundary, not a target. A small set is easier to diagnose.

   Workflow                               Lock first                                              Common failure                                            Better correction

I2VA           The first-frame subject, layout, clothing, and lighting           The prompt jumps to a new composition immediately Restate the opening anchors, then describe the next action

FL2VA          The first state, final state, and continuous path                 The output cuts or morphs instead of transforming Use one shot and describe intermediate object and pose changes

L2VA           The final frame and the state that plausibly precedes it          The model reaches the ending too early            Add a gradual approach and reserve the final moment for convergence

Full reference A distinct role for each subject, picture, video, and audio asset References compete for identity or motion control Define role, placement, and retention before writing the timeline

For an image-to-video prompt, the first line can make the frame relationship explicit:

For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, the woman shown in <Picture 1> remains beside the train window with the same coat and seat position. The camera pushes in slowly as she folds the letter and turns toward the passing lights. The woman (S1) says: <d>[English] I get off at the next station.</d>

overall_soundscape: Train wheels create a metallic rhythm beneath a low ventilation hum. Rain ticks against the glass while paper moves in her hands.

non_diegetic_music: Sustained cello notes at a slow tempo with widely spaced piano tones.

This makes the reference useful: it defines a starting state and a forward path instead of asking the model to freeze the image while unrelated actions happen around it.

Text, image, video, and audio references converging into one MiniMax H3 video workflowText, image, video, and audio references converging into one MiniMax H3 video workflow

A Reliable Iteration Loop in ImagineVid

You can run this workflow in the MiniMax H3 video generator in ImagineVid, which provides text-to-video, image-to-video, and reference-to-video entry points without requiring an API or local setup.

Use this five-step loop:

 1. Choose the input mode. Use T2VA for a fully authored scene, I2VA when the opening frame matters, or a reference workflow when identity, motion, voice, or style needs an anchor.
 2. Assign roles before uploading. Write one sentence for each asset, then remove anything without a clear job.
 3. Write one timeline. Establish the opening, describe cause-and-effect action, and specify the ending. Avoid several unrelated locations in one short clip.
 4. Set duration and composition. Choose 4-15 seconds and an aspect ratio that matches the final placement. The 15-second ceiling is a story constraint.
 5. Review the full result. Check identity, motion, crop, text, dialogue, stereo sound, music, and continuity. Change one category at a time.

For a sequence longer than 15 seconds, split it into short clips with explicit start and end states.

Common MiniMax H3 Prompt Failures and Fixes

Use the output as a diagnostic signal. A failed generation usually tells you which part of the prompt is underspecified.

                      Symptom                                                  Likely cause                                                                  Fix

The output looks like a polished still with weak     The prompt lists appearance but not an observable sequence       Add an action trigger, an intermediate change, and an ending state
motion

The camera moves in a way that contradicts the       Camera language is stacked as disconnected labels                Write one natural sentence with motion type, direction, scale, and speed
subject

A character adds unwanted speech                     Dialogue is implied, loosely quoted, or missing a stable speaker Use (S1) and an exact <d>[Language] ... </d> block, then describe non-speaking
                                                     ID                                                               actions

The output jumps between reference images            The prompt names endpoints but not the path                      Use FL2VA logic and describe pose, object, camera, and lighting changes between
                                                                                                                      them

On-screen text is unreadable, or changes shape       Text is treated as a mood instead of a visible object            Quote the exact short text and inspect typography as a separate QA item

Do not rewrite every sentence after every failure. Change only the category that failed: dialogue, identity, camera, or sound. One-variable iteration makes the prompt easier to learn from.

MiniMax H3 Prompt Guide FAQ

How long should a MiniMax H3 prompt be, and can it use multiple references?

There is no universal word count. Cover visible changes and audio events, with a clear opening and ending. The current Hailuo workspace supports up to 12 mixed reference items in Omni Reference mode;
use distinct roles instead of the maximum by default.

How do I control dialogue?

Give each speaking subject a stable ID, preserve the exact words inside a <d>[Language] ... </d> block, and describe the delivery outside the block. Add enough action after a short line to fill the
remaining timeline without inviting improvised speech.

When should I split a scene into separate clips?

Split when the idea exceeds the 15-second generation window, changes location several times, needs unrelated camera styles, or requires dialogue and action that cannot fit naturally in one timeline.
Explicit clip boundaries make later editing easier.

Conclusion

The strongest MiniMax H3 prompts read like compact production plans. They establish a starting state, define visible changes, direct the camera, separate sound, assign reference roles, and finish
with a review checklist. Hailuo 3.0 prompting becomes less about guessing magic words and more about making the intended video legible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Photo by Jeremy Yap on Unsplash
