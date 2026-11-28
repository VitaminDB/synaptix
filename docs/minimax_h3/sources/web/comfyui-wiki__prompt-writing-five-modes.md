# Source: https://comfyui-wiki.com/en/tutorial/advanced/video/minimax/minimax-h3-prompt-writing

Fetched: 2026-11-28 (curl + w3m text dump)

---

 1. Home
 2.
 3. Tutorials
 4.
 5. Advanced
 6.
 7. Video
 8.
 9. Minimax
10.
11. MiniMax H3 Prompt Writing Guide: Official Structure for All Five Modes

MiniMax H3 Prompt Writing Guide: Official Structure for All Five Modes

Write structured MiniMax H3 prompts with the official skill: five generation modes, the three core fields, camera motion vocabulary, speaker IDs and the six-section format

MiniMax published an official prompt-writing skill in the MiniMax-H3 repository on August 9-10, 2026 (skills/h3-prompt-writing). This tutorial mirrors that official guide so you can write
production-grade prompts without pulling the skill files yourself. It covers the three core fields, the keyframe alignment instructions, camera and dialogue vocabulary, and the six-section
full-reference format.

Tutorial Overview

MiniMax H3 prompts are structured, not free-form. The model expects field names, section order, and timing notation exactly as the official guide specifies. This tutorial explains the structure for
all five generation modes: T2VA, I2VA, FL2VA, L2VA, and Ref2VA.

The Five Generation Modes

 Mode                              Meaning                                        Reference input
T2VA   Text-to-video: complete audiovisual timeline from text           none
I2VA   First-frame instruction + path developing forward from it        1 image (first frame)
FL2VA  First-and-last-frame instruction + continuous path between them  2 images (start + end)
L2VA   Last-frame instruction + path converging to it                   1 image (last frame)
Ref2VA Full-reference rewrite with labeled subjects, videos, audio      up to 9 images + 3 videos + 3 audio

The Three Core Fields (T2VA / I2VA / FL2VA / L2VA)

Every base-mode prompt ends with three fields, in this exact order:

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

  • integrated_multimodal_description: the main body. Describes visuals, actions, shots, speakers, dialogue, singing, and diegetic audio along the timeline.
  • overall_soundscape: 1-4 English sentences summarizing ambient sound, physical action sounds, and non-verbal human sounds (wind, rain, traffic, footsteps, impacts, breathing). Do not repeat
    dialogue or music here. Use N/A only for full silence.
  • non_diegetic_music: 1-3 English sentences describing background music only the audience hears. State instrumentation, speed, rhythm, dynamics; no abstract mood words. Use N/A when none.

Keyframe alignment instructions

For keyframe modes, the instruction line comes first, followed by a blank line, then the three core fields:

I2VA:

For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

FL2VA:

How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.

L2VA:

How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.

N is the index of the actual final shot, and S.SS is the video duration formatted to exactly two decimal places.

Writing the Multimodal Description

Shot structure and cuts

  • No timestamp on the first shot. Later shots start with a strictly increasing cut time:

[Shot 2] At 00:03.500, the camera cuts to...

  • Cut vocabulary: the camera cuts to, the shot cuts to, the shot transitions to, the shot changes to, the shot switches to. Cross-dissolve/fade/wipe only when explicitly requested.
  • A cut must introduce new information (subject, space, state, viewpoint, time). For distance or slight angle changes, prefer camera motion.

Camera motion: type + amplitude + speed

Write camera motion as natural English inside the shot, not stacked labels:

The camera pushes in with small amplitude at slow speed toward the folded letter in her hands.
The camera pans right with large amplitude at fast speed, revealing the open doorway.
The camera holds a static shot as the runner exits the frame.

 Dimension                                                                                          Expressions
Motion type Zoom In/Out, Push In/Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down, Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly, POV, Roll Clockwise/
            Counterclockwise
Amplitude   with small amplitude, with large amplitude
Speed       at slow speed, at fast speed

Omit amplitude and speed when medium/normal (the default).

Speakers, dialogue, and singing

  • Stable speaker IDs (S1), (S2); compound (S1,S2) for simultaneous speech. IDs persist across shots; non-vocalizing characters get none.
  • On first appearance, establish identity: character type, age, gender, on/off-screen, pitch, timbre, rate, accent.
  • Keep the identifying phrase, ID, action, and delivery outside <d>; inside <d> only the language tag and verbatim user content:

The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>

  • Voiceover: use exactly says in an off-screen voiceover, then state the on-screen character's lips remain closed:

The man (S1) says in an off-screen voiceover: <d>[English] I still remember that road.</d> while his lips remain completely closed.

  • Dialogue across cuts: use <scenetrans> at both connecting points and state the audio continues across the cut (continues seamlessly across the cut). Use <cutoff> when speech truncates at video
    end.

On-screen text

Any visible banner, sign, label, subtitle, or neon text goes in English double quotation marks, verbatim:

A red neon sign reading "营业中" glows above the doorway.

Full-Reference Mode (Ref2VA): Six-Section Format

Ref2VA rewrites use six sections in this exact order:

      Section                                Purpose
subject_definitions  Defines referenced content and its labels
summary              Task type + target video + main reference relationships
retention_analysis   How each referenced item is preserved/transferred
detailed_description Shot-by-shot description in playback order
overall_soundscape   Ambience and physical sounds
non_diegetic_music   Audience-only background music

Reference labels

   Label                               Meaning
<Subject N> Reusable visible content abstracted from references
<Picture N> Reference image as a frame anchor or planning reference
<Video N>   Reference video as editing source / continuation / structure
<Audio N>   Audio signal copied or referenced

Labels keep their meaning across all six sections. <Video N> and <Audio N> are numbered independently.

Task types in summary

     Task type                              Use when
keyframe completion  Image serves as a concrete frame anchor
reference generation Asset provides guidance without being a concrete frame
video editing        A source video is directly modified
video continuation   New content continues from a source video
audio reuse          The same audio signal is reused
audio reference      Only style/timbre/content of audio is referenced

Combine multiple types with +, e.g. [video continuation + keyframe completion].

Relationship markers in retention_analysis

Visible content: fully_preserved, partially_preserved, attribute_transfer, weak_reference Audio: fully_copy, partially_copy, reference, weak_reference

<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...
<Audio 1>: reference - the target speaker follows <Audio 1>'s voice timbre without copying the original signal.

detailed_description differences

  • Style is established in one or two English sentences before [Shot 1] (not after it).
  • Insert <Subject N>, <Picture N>, <Video N>, <Audio N> at their first appearance and where their roles apply.
  • Cite <Audio N> in the shot or phase where the audio relationship is active, and state whether the signal is copied or referenced.
  • Generation tasks normally run 350-500 English words; dialogue-dense content prioritizes the full spoken timeline over word count.
  • Use <Subject N> (Sx) when a referenced subject physically speaks. Do not invent (Sx) for voice cues inside a directly reused soundtrack (use <Audio N> instead).
  • Reused dialogue/lyrics go inside <d> verbatim; write [unclear] for unintelligible spans; end statements with ./?/!.

Complete Example (T2VA)

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and says: <d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of steam rising from the sliced bread while the baker's final words carry over from the previous shot.

overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the bakery. The doorbell rings once, followed by light footsteps and the crisp sound of bread being sliced.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and a gentle fade at the end.

Official Resources

  • Prompt-writing skill: MiniMax-AI/MiniMax-H3/skills/h3-prompt-writing
  • Base-mode guide (full): base-en.txt
