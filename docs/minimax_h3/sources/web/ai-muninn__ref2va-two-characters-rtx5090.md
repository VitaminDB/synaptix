# Source: https://ai-muninn.com/en/blog/minimax-h3-ref2va-two-characters

Fetched: 2026-11-28 (curl + w3m text dump)

---

❯[                    ](Close search)esc

Loading…

bloglocal modelsmodel anatomygithub中
~/blog/minimax-h3-ref2va-two-characters

MiniMax-H3 on RTX 5090 · part 6

MiniMax-H3 Ref2VA: Two Characters From Two Reference Images, 437 s on a 5090

2026-09-06updated 2026-11-1320 min read#minimax-h3#ref2va#comfyui#rtx-5090中文版
❯ cat --toc

  • Hand a costume department two photos and the first question back is which robe goes on which person
  • Ref2VA is the only H3 mode where images mean "this is who", not "this is when"
  • The references are stills, not frames — two character sheets at 45 s each
  • The node is MiniMaxH3ReferenceToVideo, and it wants an audio_vae your ImageToVideo graph does not have
  • length has step 17: 243 frames is 10.125 s and 362 frames is the ceiling
  • ref_image_size=match scales references to the render; max makes you pay on every step
  • The prompt is six sections, and a label is defined exactly once
  • A `<Subject 1>` label is a pointer, and every shot has to resolve it
  • Dialogue goes inside the tag, everything about the speaker goes outside
  • 243 frames with two references: 437 s on one RTX 5090
  • Four common pitfalls before your first render
  • Deep dive: four unknowns on a machine that had never run this node
  • Unknown 1: do two reference images contaminate each other?
  • Unknown 2: does the speaker ID actually route the voice?
  • Unknown 3: can I control *when* a character walks in?
  • Unknown 4: does the Chinese dialogue survive?
  • Why 6.8% is not the same kind of 6.8% the gate usually sees
  • Whisper's segment boundaries say the timing directives are a target, not a contract
  • Full environment
  • Untested

❯ toc

  • Hand a costume department two photos and the first question back is which robe goes on which person
  • Ref2VA is the only H3 mode where images mean "this is who", not "this is when"
  • The references are stills, not frames — two character sheets at 45 s each
  • The node is MiniMaxH3ReferenceToVideo, and it wants an audio_vae your ImageToVideo graph does not have
  • length has step 17: 243 frames is 10.125 s and 362 frames is the ceiling
  • ref_image_size=match scales references to the render; max makes you pay on every step
  • The prompt is six sections, and a label is defined exactly once
  • A `<Subject 1>` label is a pointer, and every shot has to resolve it
  • Dialogue goes inside the tag, everything about the speaker goes outside
  • 243 frames with two references: 437 s on one RTX 5090
  • Four common pitfalls before your first render
  • Deep dive: four unknowns on a machine that had never run this node
  • Unknown 1: do two reference images contaminate each other?
  • Unknown 2: does the speaker ID actually route the voice?
  • Unknown 3: can I control *when* a character walks in?
  • Unknown 4: does the Chinese dialogue survive?
  • Why 6.8% is not the same kind of 6.8% the gate usually sees
  • Whisper's segment boundaries say the timing directives are a target, not a contract
  • Full environment
  • Untested

TL;DR

Ref2VA is the only MiniMax-H3 mode that takes more than one reference image; the other four take a first frame, a last frame, or both. Two character sheets, one six-section prompt, 243 frames at
1344x768: 437 s on one RTX 5090, two distinct voices, no bleed between the two faces (n=1, and I made the designs as different as I could). ASR on the dialogue came back at CER 6.8%, and all three
errors were homophones. The trap: a <Subject 1> label carries no identity, so restate the full appearance in every shot or the model invents a person.

Hand a costume department two photos and the first question back is which robe goes on which person

That question is the entire problem with putting two characters in one AI-generated shot. One reference image is unambiguous: there is one face, and it belongs to the only person on screen. Two
reference images need a binding, and nothing about "here are two pictures" tells the model which is which.

Part 3 of this series locked one character's face with reference portraits. This part does two people in one frame, talking to each other. The scene is the Chinese idiom 自相矛盾 (zì xiāng máo dùn,
"self-contradiction" — literally, his own spear against his own shield): a market vendor claims his shield stops everything and his spear pierces everything, a young scholar asks what happens when
the one meets the other, and the vendor has no answer.

What comes out the other end is a 10.125-second clip with three spoken lines, two distinct voices, and two characters who stay themselves, in 437 seconds on a single RTX 5090. Here is the order to do
it in, and the four things that will stop you.

Ref2VA is the only H3 mode where images mean "this is who", not "this is when"

H3 has five generation modes. Four of them take images, and the difference that matters is not how many but what the images are for — read the right-hand column.

 Mode                          What the images mean
T2VA   none — text only
I2VA   one image = the first frame
FL2VA  two images = first frame + last frame
L2VA   one image = the last frame, model infers the opening
Ref2VA N images = <Subject N> is an identity, <Picture N> is still a frame

FL2VA also takes two images, which makes it look like a candidate until you notice that its two slots are already spoken for by time. Ref2VA is the only mode whose image list is open-ended. It also
splits the job: an image tagged <Subject N> says this is who and the model picks the frames, while one tagged <Picture N> still works as a concrete target frame or shot-planning anchor. This run used
<Subject N> only.

The references are stills, not frames — two character sheets at 45 s each

I generated both character sheets with z-image, a still-image model, taking about 45 seconds each. Full-body, neutral pose, plain background. They are never shown in the video; they exist only to be
pointed at.

Two character sheets side by side: a stout middle-aged vendor in an ochre-yellow robe with a dark-brown sash and a black cap, and a slim young scholar in a pale blue-grey robe with a dark sash. The
two designs are deliberately far apart in build, colour and age so any bleed between them would be visible immediately.

The distance between those two designs is deliberate and I come back to it in the deep dive. If you are doing this for the first time, make your two characters as unlike each other as the scene
allows. It costs nothing and it turns "did the identities hold?" from a judgement call into something you can see at a glance.

The node is MiniMaxH3ReferenceToVideo, and it wants an audio_vae your ImageToVideo graph does not have

If you already have an H3 graph running, it is almost certainly built on MiniMaxH3ImageToVideo. Ref2VA is a different node:

MiniMaxH3ReferenceToVideo
  required: clip · vae · audio_vae · prompt · width · height · length · ref_image_size
  optional: ref_images · ref_videos · ref_video_audios · ref_audios
  output:   ['CONDITIONING', 'LATENT']

Two required inputs are new compared to ImageToVideo: audio_vae (minimax_h3_audio_vae_fp32.safetensors) and ref_image_size. The outputs are identical — CONDITIONING and LATENT — so everything
downstream of the node, the whole sampler chain, transfers unchanged. Converting an existing graph is a node swap plus two wires, and forgetting audio_vae is the single most likely reason your
converted graph will not run.

ref_images has type COMFY_AUTOGROW_V3, which sounds exotic and is not. In API-format JSON it is a plain array of links:

"ref_images": [["20", 0], ["21", 0]]

Nodes 20 and 21 are LoadImage nodes. The images themselves have to be POST /upload/image'd to ComfyUI first, then loaded — you cannot inline them into the prompt payload.

length has step 17: 243 frames is 10.125 s and 362 frames is the ceiling

The node schema declares length with step: 17 and min: 5. So legal frame counts are 17k+5 — 5, 22, 39, ... 243, ... 362. That grid is enforced by the node, not by a convention downstream of it, which
means a round number like 240 fails at the node rather than producing something slightly off.

The two values worth memorising: 243 frames = 10.125 s and 362 frames = 15.08 s, the latter being the ceiling for a single generation. Anything longer is a stitching problem, not a parameter.

ref_image_size=match scales references to the render; max makes you pay on every step

The other new required input has two settings. match scales the reference images down to the pixel area of the generation. max uses a 2048 px short edge. The node's own tooltip explains the cost:

    Reference tokens ride through every sampling step, so 'max' can be several times slower.

That is the mechanism in one sentence: reference images are not read once at the start; they are conditioning that participates in every denoising step, so their token count multiplies by your step
count. I ran match. There is a wrinkle in that "several times slower" claim which I get to in the untested list at the end.

The prompt is six sections, and a label is defined exactly once

This is where Ref2VA diverges most from the three-part prompts you may be used to. The format has six named sections, in order: subject_definitions → summary → retention_analysis →
detailed_description → overall_soundscape → non_diegetic_music.

The six Ref2VA prompt sections, each with the one question it answers: subject_definitions asks who shows up, summary asks what the clip is, retention_analysis asks how much of each reference
survives, detailed_description asks what happens second by second, overall_soundscape asks what the world sounds like, non_diegetic_music asks what the score is. A subject label is defined once in
section one and referenced in five of them.

The design point is the arrow on the right of that diagram. A character label is defined once in subject_definitions and reused across the other five sections. Change what a character looks like and
you edit section one; every downstream reference follows. That is the payoff for a format that otherwise looks like bureaucracy.

retention_analysis is the section people skip and should not. It does not take free text. Visible references (<Subject N>, <Picture N>, <Video N>) take one of four fixed markers: fully_preserved,
partially_preserved, attribute_transfer, weak_reference. Audio references use a different set — fully_copy, partially_copy, reference, weak_reference — which I did not need here. Both characters in
my run were tagged fully_preserved. The full spec lives in the official repo at MiniMax-AI/MiniMax-H3, under skills/h3-prompt-writing/references/ref-en.txt.

Here is the skeleton, with the parts I am not quoting marked as placeholders:

subject_definitions:
  <Subject 1> — a stout middle-aged man with a round fleshy face, a thick
                drooping moustache, an ochre-yellow robe with a dark-brown
                collar band
  <Subject 2> — a slim young scholar, pale blue-grey robe, dark sash

summary:
  [reference generation] the plot in one paragraph

retention_analysis:
  <Subject 1>: fully_preserved
  <Subject 2>: fully_preserved

detailed_description:
  [Shot 1] 0-3s   <Subject 1> (S1), <full appearance restated>, stands behind
                  the stall and lifts the shield, voice loud and boastful:
                  <d>[Chinese] 原文</d>
  [Shot 2] 3-6s   ...
  [Shot 3] 6-8s   ...

overall_soundscape:
  sounds made by things on screen — no score here

non_diegetic_music:
  N/A

A <Subject 1> label is a pointer, and every shot has to resolve it

Writing <Subject 1> stands behind the stall and lifts the shield is not enough. The model invents a person. The label is a pointer that gets resolved from the text around it, and if the text around
it says nothing about a face, there is nothing to resolve.

Label only versus label plus the look. On the left, a shot that names the subject label and nothing else — the model invents a person, and the face, clothes and age all drift. On the right, the same
shot with the appearance spelled out inline (a stout middle-aged man with a round fleshy face, a thick drooping moustache, an ochre-yellow robe with a dark-brown collar band) — the look holds and the
two reference images stay uncontaminated.

What do the official examples do? Shot 1 describes each subject head to toe. Later shots shorten it, but the short version still carries something you could pick the person out by:

[Shot 2] … <Subject 4> (S2), the young man in the dark-grey hoodie from Shot 1, …
[Shot 3] … <Subject 3> (S1), the blonde woman in the light-pink shirt from Shot 1.

Not a full restatement, and not a bare label either. The spec asks each shot to establish the current appearance; it never says how many times you have to spell it out.

I ran into the same problem later that day while generating stills. I used local-image's --reference flag with a proper reference image, but the prompt only said "the same man" plus the scene. Output
number two was a Western man in a navy shirt and trousers wearing a wristwatch — wrong person, wrong century, wrong continent. Restating the look item by item (head-wrap, coarse brown hemp tunic,
indigo sash) locked it on the first try. A reference image gives the model something to draw from; it does not tell the model that this is the thing to draw.

Dialogue goes inside the tag, everything about the speaker goes outside

Spoken lines use a tag: <d>[Chinese] 原文</d>. Inside the tag there are exactly two things — the language marker and the verbatim line. Nothing else. Who is speaking, what their tone is, what their
voice sounds like: all of that goes in the prose outside the tag. Speaker IDs (S1) and (S2) stay stable for the whole clip — assigned once, reused at every line. The official example introduces (S2)
in the shot where he first speaks rather than up front.

And one hard requirement: Chinese dialogue must be in simplified characters. Traditional characters break sentences part-way through. This is measured, not folklore — a prior test on the same scene
with the same seed scored word-for-word ASR accuracy at 0.667 with traditional characters against 1.000 with simplified. If your script lives in traditional Chinese, convert it before it goes into
the prompt.

243 frames with two references: 437 s on one RTX 5090

Look at the two rows as separate data points rather than a delta — two things changed between them.

         Run               Frames     Reference images ref_image_size Time
This article           243 (10.125 s) 2                match          437 s
Earlier, same machine  124 (5.17 s)   1                —              136 s

Frame count and reference count both moved, so I cannot tell you what the second reference image cost. What the table does give you is a planning number: about seven and a half minutes for ten
seconds of two-character dialogue at 1344x768 on one 5090, with the config in the deep dive below.

Four common pitfalls before your first render

 1. A frame count off the 17k+5 grid. The node rejects it. 243 and 362 are the two you want.
 2. A missing audio_vae after converting a graph from MiniMaxH3ImageToVideo. It is a new required input and the old graph has no wire for it.
 3. Reference images that carry no appearance in the text. The image alone is not a binding — restate the look in every shot.
 4. Traditional-character dialogue. It degrades mid-sentence and the failure is not obvious until you listen.

Deep dive: four unknowns on a machine that had never run this node

You can skip this section and still run Ref2VA; everything required is covered above. What follows is why I treated four specific things as unknown, what I expected each time, and where my
expectations were wrong.

The reason these were unknowns at all: this machine had zero prior MiniMaxH3ReferenceToVideo jobs in its ComfyUI history. Nothing to copy, no known-good graph, no previous run to diff against. I read
the node's wiring live from /object_info and worked out from the schema what each input wanted. Four things I could not settle from the schema, so I designed the shot to answer them.

Unknown 1: do two reference images contaminate each other?

The problem. Two images go into one ref_images list. There is no per-image slot in the schema saying "this face belongs to subject 1" — the binding has to come from the text.

What I expected. Contamination. My mental model was that both images land in one shared conditioning budget and get averaged into some composite person, most likely showing up as the wrong robe
colour on the wrong body, or one character borrowing the other's face shape.

What I did. Rather than test it gently, I made the two designs maximally different so that any bleed would be unmissable: stout, middle-aged, ochre-yellow robe, drooping moustache versus slim, young,
pale blue-grey robe, clean-shaven. Two axes of separation — build and colour — plus age.

Result. It did not happen. Both characters came out as themselves and stayed that way across all three shots.

Reflecting on the expectation. My model of a shared averaged budget was wrong, or at least it is not what dominates at n=2 references. But I should be precise about what I actually tested: this was
one run, with two characters whose appearances were deliberately very different. I built the experiment to make failure visible, which is also what makes it the easy case. Two characters who look
alike — same age bracket, same robe colour, same build — is untested and is exactly where an averaging failure would hide.

Unknown 2: does the speaker ID actually route the voice?

The problem. (S1) and (S2) are just parenthesised text in a prompt. Nothing in the node schema says they mean anything to the audio path.

What I expected. Honestly, no strong prior. The plausible failure was one voice reading all three lines, or the voices swapping between shots.

What I did. Wrote the scene so (S1) speaks twice and (S2) speaks once, with the second (S1) line arriving after (S2) has spoken. If the routing were positional rather than by ID, the third line would
come out in the scholar's voice.

Result. Two distinct voices, each matching its character, including on the return to (S1).

Reflecting on the expectation. The speaker IDs are doing real work, not decorating the prompt. The specific thing this rules out is positional assignment — voice follows the ID, not the order of
appearance. Still one run with two speakers; I have not tested three.

Unknown 3: can I control when a character walks in?

The problem. The scholar has to enter partway through. For the first six seconds he must not be in the frame at all.

What I expected. This was the one I expected to fail, and it is the reason I picked this scene. Every per-second directive I had tested before was of the form "someone already on screen does an
action". Presence felt like a different category — the model decides who is in the shot, and the timeline block only choreographs them once they are there. My expectation was that both characters
would be standing there from frame one and the scholar would just start talking at the six-second mark.

What I did. Wrote the entrance as a 6-8s directive in detailed_description, then extracted frames from the output at 1.5 s, 4.5 s, 7.0 s and 9.5 s and looked at them.

Result. Absent at 1.5 s. Absent at 4.5 s. Present at 7.0 s.

Four frames from the output at 1.5, 4.5, 7.0 and 9.5 seconds. In the first two the vendor is alone at his stall with the shield and then the spear. At 7.0 s the young scholar in the blue-grey robe
has entered from the left, and at 9.5 s the two are facing each other — the scholar was written to enter at 6-8s and does not exist in the frame before that.

Reflecting on the expectation. My model was wrong at the category level, not in the details. I had assumed the timeline directives were an animation layer sitting on top of a fixed cast, when they
are closer to a staging layer that also decides occupancy. The reason I believed the narrower version is that my prior tests only ever contained already-present characters — I had never given the
directives a chance to control presence, so of course I had no evidence they could.

Unknown 4: does the Chinese dialogue survive?

The problem. Three spoken lines, and no way to judge them by ear reliably — I know what the lines are supposed to be, which makes me a terrible listener.

What I expected. Some degradation. Prior H3 work in this series has a standing audio gate precisely because dialogue is the fragile part.

What I did. Ran the output audio through mlx-community/whisper-large-v3-turbo, normalised traditional to simplified with opencc so the comparison is not measuring script conversion, and scored
character error rate with jiwer.

Result. CER 6.8%, with the ASR returning 44 characters against 44 expected. Three substitutions:

Expected ASR heard    Pinyin
支       只         zhī / zhī
矛       毛         máo / máo
矛       毛         máo / máo

All three are homophones. The character count was exact and the sentence structure held, with no garbling and no repetition.

Reflecting on the expectation. I expected degradation and I got a number that is not zero, so the naive reading is "mild degradation, as predicted". That reading is wrong, and the correction is the
useful part of this whole section.

Why 6.8% is not the same kind of 6.8% the gate usually sees

The audio gate's own history is what nearly tripped me up here. Every clip it has logged with CER above zero was also flagged as having an audio problem, which makes non-zero CER look like a reliable
failure signal. But those clips fail in different shapes:

  • one single-character mispronunciation
  • one robotic-audio case that turned 探討 into 湯頭
  • one at 75%, where a repeated phrase was literally transcribed twice

A homophone substitution and a mispronunciation produce identical CER and completely different pinyin. 探討 (tàn tǎo) → 湯頭 (tāng tóu) is a different sound coming out of the model's mouth. 矛 → 毛
is the same sound, and the transcriber picking the wrong character for it. CER cannot tell those apart; pinyin can. If I had only looked at the number I would have called this a failed clip, and if I
had only looked at the number and it had been 0%, I might have called a genuinely broken clip clean.

Whisper's segment boundaries say the timing directives are a target, not a contract

One more thing came out of the ASR pass for free, because Whisper reports segment boundaries:

Written directive  Whisper segment
0-3s               0.0–3.0 s
3-6s               3.0–6.3 s
6-8s               6.3–10.1 s

The first two land essentially on the mark. The third runs about two seconds long. Combined with Unknown 3 — where the entrance did land inside the written window — the picture is that the directives
control ordering and presence tightly and duration loosely. Budget for the last shot overrunning if you are cutting to music.

Full environment

Every number above comes from this configuration:

RTX 5090 · ComfyUI 0.34.0 · torch 2.11.0+cu128
1344x768 · 243 frames · 14 steps
sampler res_multistep + simple · shift 12/3 · Spectrum ON
MiniMaxLowVRAMAttention head_chunks=16
MiniMaxChunkFeedForward chunks=8
node MiniMaxH3ReferenceToVideo · ref_image_size=match · 2 ref images
audio VAE minimax_h3_audio_vae_fp32.safetensors
character sheets: z-image, ~45 s each

The chunk settings come from Part 5, which measured them on the same machine. Note the torch line: this run is on cu128, and Part 5 measured torch 2.13.0+cu130 as worth 1.48x on the NVFP4 config. I
have not re-measured the Ref2VA path on cu130, and I do not know whether these weights take the same kernel path, so treat 437 s as the number this configuration produced rather than the floor for
this machine.

Untested

  • ref_image_size=max with two or more reference images. I only ran max on a single-reference generation elsewhere: same seed, 22% slower, no visible identity gain. That is worth flagging because
    22% is nothing like the tooltip's "several times slower" — but a single reference at max is also the cheapest possible version of that setting, so the tooltip may well be right at higher
    reference counts. Unresolved.
  • ref_videos, ref_video_audios and ref_audios. I have not touched either of these. The official spec allows an <Audio N> reference for voice timbre, which is the obvious next thing to try for a
    recurring character.
  • Two similar-looking characters. Everything in Unknown 1 was measured on a deliberately extreme appearance gap.
  • Three or more speakers. Speaker ID routing is verified at two.

Back to the costume department: the two photographs turned out to be enough, but only because the prompt kept saying out loud which robe went on which person. Ref2VA gives you the slots for a cast;
the text is still what does the casting. Part 3 is where the single-character version of this problem got solved, and Part 5 is where every performance setting in the environment block above came
from.

FAQ

Which MiniMax-H3 mode accepts more than one reference image?
    Only Ref2VA. I2VA takes one image and treats it as the first frame, FL2VA takes two as first and last frame, and L2VA takes one as the last frame. Ref2VA is the only mode with an open-ended image
    list, and it splits the roles: an image tagged <Subject N> is an identity the model can place anywhere, while <Picture N> still works as a concrete frame. That is why it is the one that handles
    two characters in the same shot.
Why does a MiniMax-H3 prompt draw the wrong person when I write <Subject 1>?
    The label carries no identity on its own. It is a pointer that the model resolves from the surrounding text, so if a shot only says '<Subject 1> lifts the shield' the model invents whoever fits.
    Every official example restates the full appearance in every shot, and that is the spec rather than verbose example-writing.
How long does a two-reference Ref2VA render take on an RTX 5090?
    243 frames at 1344x768 with two reference images and ref_image_size=match took 437 seconds on one RTX 5090. A 124-frame run with a single reference image on the same machine took 136 seconds, but
    frame count and reference count both changed between those two, so it is a comparison point rather than a cost attribution.
Do two reference images contaminate each other in Ref2VA?
    In my run they did not — two characters stayed visually separate with no bleed. That is n=1 with two deliberately dissimilar designs (stout in an ochre robe versus slim in a blue-grey robe),
    chosen so contamination would be obvious. Similar-looking characters are untested here.
Why must Chinese dialogue for MiniMax-H3 be in simplified characters?
    Traditional characters break sentences mid-way. A prior measurement on the same scene and the same seed scored word-for-word ASR accuracy of 0.667 with traditional characters against 1.000 with
    simplified.
What frame counts are legal for MiniMaxH3ReferenceToVideo?
    The node schema declares length with step 17 and min 5, so legal values follow the 17k+5 grid. 243 frames is 10.125 seconds and 362 frames is 15.08 seconds, which is the single-generation
    ceiling. An arbitrary number like 240 is rejected by the node itself, not by a downstream convention.

MiniMax-H3 on RTX 5090

← Previous[Benchmark] From OOM to 8.6 minutes: a MiniMax-H3 config stack for one RTX 5090Next →[Benchmark] LTX-2.5 vs MiniMax-H3 on one RTX 5090: 29s vs 81s for the same Chinese dialogue clip
All posts in this series

 1. 1. [Benchmark] Running MiniMax-H3 on one RTX 5090: four files, 31.7 GB, 175s per talking clip
 2. 2. [Benchmark] Twice as Fast: MiniMax-H3 on an RTX 5090, 625s Down to 314s
 3. 3. [Benchmark] MiniMax-H3 1080p Isn't Unsupported, It's Three and a Half Hours
 4. 4. [Benchmark] Ten effect embeddings for MiniMax-H3 — 10 MB that buys you bullet time, fire breath and a full year of seasons
 5. 5. [Benchmark] From OOM to 8.6 minutes: a MiniMax-H3 config stack for one RTX 5090
 6. 6. MiniMax-H3 Ref2VA: Two Characters From Two Reference Images, 437 s on a 5090[Current post]
 7. 7. [Benchmark] LTX-2.5 vs MiniMax-H3 on one RTX 5090: 29s vs 81s for the same Chinese dialogue clip
 8. 8. [Benchmark] Roll cheap, then enhance the keeper: NVIDIA's two-stage MiniMax-H3 + LTX-2.5 pipeline explained
 9. 9. [AI Workflow] Block a scene in Blender, let MiniMax-H3 film it: local Ref2VA on an RTX 5090
10. 10. A 22-second Taiwan FPV flight without a drone: Blender blockout + MiniMax-H3 on one RTX 5090
11. 11. A 25-second no-fog FPV one-take with MiniMax-H3 and ComfyUI-H3-Continuum on one RTX 5090
12. 12. Turn a real dance into a depth video and have MiniMax-H3 restage it in a palace: native 768 on one RTX 5090
13. 13. [MiniMax-H3] A storyboard image isn't a prompt: getting local H3 to hit 13 of 16 shots
14. 14. [MiniMax-H3] Open-H3-Regenerate-2K on an RTX 5090: more detail than VSR at 133 s a clip
15. 15. Making a character do parkour with NVIDIA MotionBricks on one RTX 5090: 1.66 ms per generation
16. 16. Pika-H3-Flex on an RTX 5090: 2x faster MiniMax-H3, and a Blender blockout makes the camera obey
17. 17. [Benchmark] No GPU? Run MiniMax-H3 on a Colab G4: 66 s per clip, ~49 clips an hour
18. 18. How to run Viggle Remake character swap on an RTX 5090 (int8, ComfyUI)

Read next

  • 2026-11-04
    Pika-H3-Flex on an RTX 5090: 2x faster MiniMax-H3, and a Blender blockout makes the camera obey

    Pika-H3-Flex on an RTX 5090: 20.8 s per 5-second clip vs 38 s for MiniMax-H3. Text-only orbits drift; a Blender blockout got 6 of 6 clips around.

  • 2026-10-25
    [MiniMax-H3] A storyboard image isn't a prompt: getting local H3 to hit 13 of 16 shots

    A 4x4 storyboard plus a character sheet into MiniMax-H3 Ref2VA on an RTX 5090: one sentence hit 3/16 panels, an official-format shot timeline hit 13/16.

  • 2026-10-19
    Turn a real dance into a depth video and have MiniMax-H3 restage it in a palace: native 768 on one RTX 5090

    Real dance to grey depth video to MiniMax-H3 Ref2VA with a first-frame guide: a new character dances the routine in a palace, native 768x1376 on one RTX 5090.

  • 2026-10-18
    A 25-second no-fog FPV one-take with MiniMax-H3 and ComfyUI-H3-Continuum on one RTX 5090

    A 25-second FPV one-take over Taiwan with no fog: MiniMax-H3 Ref2VA and ComfyUI-H3-Continuum on one RTX 5090, each splice hidden in a turn through the sky.

ShareCopyRedditLinkedInXFacebook
❯echo "thanks for reading"

If this helped, buying me a coffee is all it takes — it lets me run more hands-on tests and share what I find.

Buy me a coffee

Don't miss the next one
