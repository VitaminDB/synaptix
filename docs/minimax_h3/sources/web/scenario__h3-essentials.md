# Source: https://help.scenario.com/articles/1593651655-minimax-h3-the-essentials

Fetched: 2026-11-28 (curl + w3m text dump)

---


Scenario Knowledge BaseScenario Knowledge Base
[                    ]
All Collections›Video Generation

MiniMax H3: The Essentials

Last updated 20 hours ago
[2510f347-0eaa-468f-8bdb-0d085320e766]

Provider: MiniMax (H3 Max and H3 Max Turbo post-trained by Fal)  |  Modality: Text / Image / Video to Video  |  Model IDs: model_minimax-h3, model_minimax-h3-fast, model_minimax-h3-max-t2v,
model_minimax-h3-max-i2v, model_minimax-h3-max-turbo-t2v, model_minimax-h3-max-turbo-i2v, model_minimax-h3-max-lip-sync, model_minimax-h3-max-3d-to-video

MiniMax H3 generates short video clips with native stereo audio from a text prompt, first and last frames, or multimodal references, running 5 to 15 seconds at up to 2K and 24 fps. MiniMax H3 Fast
takes the same inputs as MiniMax H3, including reference images, videos and audio, and renders 4 to 15 second clips at 480P or 768P. MiniMax H3 Max is a faster, simpler sibling from Fal:
text-to-video and image-to-video only, at 480p or 768p, with no reference images or videos, tuned for stronger prompt adherence and richer aesthetics. Use H3 for brand films, product spots, story
beats with dialogue, motion design, and game UI trailers; use H3 Max when you want a fast, cheap, on-model clip from a prompt or a single image. MiniMax H3 Max Turbo pushes that further: the same
text-to-video and image-to-video pair, tuned again by Fal for even stronger prompt adherence and faster inference, best when a detailed prompt needs to land exactly as written. MiniMax H3 Max Lip
Sync is a different kind of model in this family: it takes no text prompt at all, only a still portrait and 5 to 15 seconds of audio, and returns that portrait speaking in sync with the audio track.
MiniMax H3 Max 3D to Video turns a Blender render or 3D previs clip into a finished video, keeping the layout, camera movement, and timing of the source clip.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Which Model Should I Use?

      Model                       ID                                         Input                                                                    Best for

MiniMax H3         model_minimax-h3                 Text, first/last frame, or up to 9 reference images/     Native audio and dialogue, multi-reference brand continuity, up to 2K.
                                                    videos/audio

MiniMax H3 Fast    model_minimax-h3-fast            Text, first/last frame, or up to 9 reference images, 3   The same inputs as MiniMax H3 at 480P or 768P, 4 to 15 seconds, with native audio and
                                                    videos and 3 audio clips                                 dialogue.

MiniMax H3 Max T2V model_minimax-h3-max-t2v         Text prompt                                              Fast, cheap, on-model clips from a prompt alone, with a full spread of aspect ratios.

MiniMax H3 Max I2V model_minimax-h3-max-i2v         Starting image, optional end image                       Animating an existing still with control over exactly how the shot ends.

MiniMax H3 Max     model_minimax-h3-max-turbo-t2v   Text prompt                                              The fastest, most prompt-accurate text-to-video in the family; best when a detailed prompt
Turbo T2V                                                                                                    needs to land exactly as written.

MiniMax H3 Max     model_minimax-h3-max-turbo-i2v   Starting image, optional end image                       The fastest way to animate a still with precise control over the ending, when speed
Turbo I2V                                                                                                    matters more than 2K output.

MiniMax H3 Max Lip model_minimax-h3-max-lip-sync    A portrait image plus 5 to 15 seconds of audio           Turning a still photo or character portrait into a talking video synced to a specific
Sync                                                                                                         voice recording, up to 2K.

MiniMax H3 Max 3D  model_minimax-h3-max-3d-to-video A Blender render or 3D previs video, optional prompt and Turning a cinematic blockout or layout pass into a photorealistic shot, up to 1080P, with
to Video                                            reference images                                         the camera and motion of your source.

Reach for MiniMax H3 when you need synced dialogue, reference-driven consistency, or up to 2K output. Reach for H3 Fast when you need those same inputs but 480P or 768P output is enough. Reach for H3
Max when you want a faster and cheaper clip from just a prompt or a single image, and do not need multi-reference input.

To teach MiniMax H3 your own style, characters or motion, train a video LoRA on your images and clips, then generate with it on MiniMax H3 LoRA or MiniMax H3 Ref LoRA. See Train a MiniMax H3 Video
LoRA in Train Custom AI Models.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

How MiniMax H3 Works

Start with a Prompt that describes the scene, camera, and audio. Then choose one input mode:

  • Text only. Generate from the Prompt alone. Good for open creative exploration.

  • First Frame / Last Frame. Lock the opening shot, and optionally the closing shot. Aspect Ratio follows the first frame. Do not mix frames with Reference Images, Videos, or Audio.

  • Reference Images. Pass up to 9 images to guide subjects and style. First 5 are free; additional images are billed. Cannot combine with first or last frame.

  • Reference Videos. Pass up to 3 clips (2 to 15 seconds each, 2 to 15 seconds total) to guide motion.

  • Reference Audio. Optional, and only with at least one reference image or video.

Native audio (dialogue, SFX, ambience) is generated in the same pass. Put spoken lines in the Prompt when you need clear lip-synced dialogue.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Writing Prompts for MiniMax H3

H3 takes a single Prompt, but it responds well when that Prompt reads like a tiny shooting script rather than one long sentence. MiniMax's own guidance boils down to a simple habit: walk the clip
forward in time, and describe three things as you go, the picture and action, the in-scene sound, and any background music. You can write all of it in plain prose.

Walk the clip forward in time

Open with a look, for example "Live-action, cinematic" or "2D anime", then describe what happens beat by beat. If the clip has more than one shot, call the cuts out in order and, when timing matters,
add a rough timestamp such as "around the four-second mark, cut to a close-up of the coffee cup". Keep any readable on-screen text in quotes so it renders as written.

Direct the camera in plain words

Name the move, how big it is, and how fast, all in one phrase. Something like "the camera slowly pushes in on the folded letter" is enough. H3 understands the usual moves, push in and pull out, pan,
tilt, truck, arc, tracking, a static hold, a slight handheld shake, or a POV, so describe the one you want and let it sit inside the action.

Write dialogue as spoken lines

Put spoken words in quotes and make it clear who is talking, for example: the waitress leans in and says, "We close in five, honey." Short lines lip-sync best. If a voice is narration rather than
someone on screen, say so, and note that their lips stay closed.

Describe the sound you want

Because H3 generates audio in the same pass, a sentence or two of ambience goes a long way: rain on glass, a fridge humming, a cup set on the counter. If you want a musical bed, describe it briefly
by instrument and mood, for example "a slow, sparse piano, quiet and unhurried". Ask for silence explicitly when you want it.

MiniMax H3 Max shares this same approach: describe the shot in timestamped beats, name the camera move in plain words, and describe the sound you want. See the MiniMax H3 Max section below for its
own worked examples.

A short worked example

A text-only beat, written as plain prose:

Live-action, cinematic. A rain-streaked diner window at night, warm interior light. The camera slowly pushes in on a tired waitress as she leans over the counter and says, "We close in five, honey."
Around the four-second mark, cut to a close-up of a full coffee cup with steam rising. Ambient sound: steady rain on the glass, the low hum of a refrigerator, distant traffic. Under it, a slow,
sparse piano over soft upright bass, quiet and unhurried.

Your input mode shapes where the description starts. With Reference Images, ground the opening in the reference and develop forward while keeping the subject consistent. With First and Last Frame,
describe the motion that carries the opening frame to the closing one. Last Frame only works when a First Frame is also set, so there is no Last Frame only mode.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MiniMax H3 Fast: The Full H3 Input Set at 480P or 768P

MiniMax H3 Fast (model_minimax-h3-fast) is the Fast tier of MiniMax H3, from MiniMax. It takes exactly the same inputs as MiniMax H3: a Prompt alone, a First Frame with an optional Last Frame, or up
to 9 Reference Images, 3 Reference Videos and 3 Reference Audio clips. It generates 4 to 15 second videos with native dialogue and sound at 480P or 768P. One model covers text to video, image to
video and reference to video.

H3 Fast, H3 and H3 Max compared

                                                MiniMax H3                                            MiniMax H3 Fast                                         MiniMax H3 Max

Model IDs         model_minimax-h3                                                       model_minimax-h3-fast                      model_minimax-h3-max-t2v, model_minimax-h3-max-i2v

Inputs            Prompt, First/Last Frame, or up to 9 Reference Images, 3 Reference     Same as MiniMax H3                         Prompt (T2V) or First Frame with optional Last Frame (I2V); no
                  Videos and 3 Reference Audio clips                                                                                reference images, videos or audio

Prompt length     Up to 7,000 characters                                                 Up to 7,000 characters                     Up to 7,000 characters (50,000 on H3 Max Turbo)

Duration          5 to 15 seconds                                                        4 to 15 seconds                            5 to 15 seconds

Resolution        768P or 2K (default 2K)                                                480P or 768P (default 768P)                480P or 768P (default 768P)

Aspect ratio      21:9, 16:9, 4:3, 1:1, 3:4, 9:16 or Auto (adaptive in the API)          21:9, 16:9, 4:3, 1:1, 3:4, 9:16 or Auto    T2V only: 21:9, 16:9, 4:3, 1:1, 3:4 or 9:16
                                                                                         (adaptive in the API)

Prompt Expansion  No                                                                     No                                         Yes
and Seed

Native audio      Yes, including dialogue                                                Yes, including dialogue                    Yes, ambience and SFX first; dialogue less consistent

When to use H3 Fast

  • Pick H3 Fast over MiniMax H3 when you need H3’s reference images, videos, audio or first and last frames, and 480P or 768P output is enough: drafts, previews, social clips. Pick MiniMax H3 when
    the final clip needs 2K.

  • Pick H3 Fast over H3 Max when the clip has to follow reference images, reference videos or reference audio, or when you want text, image and reference input in one model. Pick H3 Max or H3 Max
    Turbo when you work from a prompt or a single still and want Prompt Expansion or a seed, and H3 Max Turbo when the prompt is longer than 7,000 characters (up to 50,000).

  • Short clips: among the text and image to video models in this family (MiniMax H3, H3 Fast, H3 Max and H3 Max Turbo), H3 Fast is the only one that accepts a 4 second duration.

Parameters (MiniMax H3 Fast)

The controls match MiniMax H3 (see Parameters (MiniMax H3) below), except Duration and Resolution.

prompt

Required, up to 7,000 characters. Describe subject, action, camera, lighting and audio, and write spoken lines in quotes. The prompt-writing guidance above for MiniMax H3 applies unchanged.

firstFrameImage and lastFrameImage

Optional opening and closing frames (max 30 MB each). The video shape follows the First Frame. Last Frame only works when First Frame is set. Neither can be combined with Reference Images, Videos or
Audio.

referenceImages

Optional. Up to 9 images (max 30 MB each) to guide subjects or style. Reference images beyond the first five add to the price shown on the Generate button. Cannot combine with First Frame or Last
Frame.

referenceVideos

Optional. Up to 3 videos to guide motion (each 2 to 15 seconds, 2 to 15 seconds total, max 50 MB each). Reference videos are billed by their uploaded duration, which adds to the price shown on the
Generate button. Cannot combine with First Frame or Last Frame.

referenceAudio

Optional. Up to 3 audio clips (each 2 to 15 seconds, 2 to 15 seconds total, max 15 MB each). Requires at least one reference image or video.

duration

Clip length in seconds, from 4 to 15. Default 5.

resolution

480P or 768P. Default 768P.

aspectRatio

21:9, 16:9, 4:3, 1:1, 3:4, 9:16 or Auto (adaptive in the API, the default). Auto follows the reference media, or uses 16:9 for a prompt alone. Ignored when First Frame is set, because the shape
follows that image.

Tips and Known Limitations (MiniMax H3 Fast)

  • Reference Audio guides rhythm, it is not copied. In testing, the generated soundtrack followed the rhythm of the reference audio but did not reproduce the clip itself. To use an exact track, add
    it in editing.

  • Describe props positively. Negative wording (asking for no labels) still produced labels in testing. Describe the surface you want instead, for example "a plain, smooth glass bottle".

  • Do not mix modes. As on MiniMax H3, First/Last Frame and Reference Images/Videos/Audio are mutually exclusive.

  • No 2K output. H3 Fast stops at 768P. For 2K, use MiniMax H3.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MiniMax H3 Max: A Faster, Simpler Sibling

MiniMax H3 Max is Fal's post-trained variant of MiniMax H3: stronger prompt adherence and richer aesthetics, at 480p or 768p. It takes only a prompt or a first/last frame image, with no reference
images, videos, or reference audio inputs, though it does generate its own synced audio in the output. Two models cover it: model_minimax-h3-max-t2v for text-to-video, and model_minimax-h3-max-i2v
for image-to-video with optional end-frame control.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

How MiniMax H3 Max Works

Both models take one plain-text Prompt describing the scene, action, camera, and mood. H3 Max responds best to long, detailed prompts written as timestamped beats (what happens from 0 to 3 seconds, 3
to 6, and so on) rather than a short caption. Set Duration (5 to 15 seconds) and Resolution (480P or 768P), and optionally raise Prompt Expansion to let the model rewrite a thin prompt into a richer
one before generating.

Text to Video

H3 Max T2V also exposes Aspect Ratio (21:9, 16:9, 4:3, 1:1, 3:4, 9:16), so the same model covers widescreen cinematics, square product loops, and vertical social clips.

A veteran wildland firefighter in soot-streaked turnout gear stands at the edge of a smoldering ridge at dusk. 0-3s: she plants her boots on the ash-covered rock, exhaling as embers drift past her face, the orange glow of the distant fire line reflecting off her visor. 3-6s: the camera slowly pushes in as she raises a hand to shield her eyes, wind picking up and blowing ash sideways across the frame. 6-10s: she turns and starts walking down the slope toward a line of parked engines, silhouetted against the smoke-filled sky, the rumble of distant flames and crackling radio chatter building under a low ambient wind. Cinematic wide shot, warm amber and deep orange grade, volumetric smoke, handheld camera with subtle shake.

Duration 10s, Resolution 768P, Aspect Ratio 16:9 · Open on Scenario

Prompt Expansion set to Quality spends extra time enriching a shorter prompt before generation. It paid off on this neon alleyway shot, where the model added rain, drone lighting, and background
detail beyond what was explicitly described:

A cyberpunk mercenary in a weathered tactical jacket strides down a narrow neon-lit alley at night. 0-4s: rain-slicked pavement reflects flickering holographic ads as he walks toward camera, breath visible in the cold air. 4-7s: the camera cranes low as he passes a food stall, steam rising and neon signage in Japanese and English glowing pink and cyan overhead. 7-10s: he glances up at a passing drone, its spotlight briefly sweeping across his face before he ducks into a doorway, distant synth music and rain underneath. Moody cyberpunk cinematography, high contrast neon color grade, volumetric rain, slow tracking shot.

Duration 10s, Resolution 768P, Prompt Expansion: Quality · Open on Scenario

Duration goes up to a full 15 seconds, enough room for a small multi-beat story in one clip:

An elderly fisherman in a knit sweater sits on a weathered wooden dock at dawn, mending a torn fishing net draped across his lap. 0-4s: his weathered hands work the twine methodically, mist rising off the still water behind him as the sun begins to break over the horizon. 4-8s: he pauses, looks up toward the open sea with a quiet, reflective expression, gulls calling faintly in the distance. 8-12s: he returns to his mending, a small wooden boat gently rocking at its mooring beside the dock, the light growing warmer as the sun climbs. 12-15s: he ties off the final knot and sets the net down, exhaling slowly as calm water laps against the dock pilings, soft ambient waves and distant gulls underneath.

Duration 15s (the model maximum), Resolution 768P · Open on Scenario

Aspect Ratio adapts the same model to square product loops, vertical social clips, and ultra-wide cinematics:

Aspect Ratio 9:16, product/marketing

Aspect Ratio 21:9, fantasy/game cinematic

Aspect Ratio 1:1, Resolution 480P, product loop

Aspect Ratio 21:9, travel/lifestyle

Image to Video

H3 Max I2V takes a First Frame image and animates it; the output shape follows that image, so there is no separate Aspect Ratio control. It stayed fully on-model with an illustrated, non-photoreal
source, keeping the exact art style while inventing a consistent kitchen environment behind the original flat background:

      Source image (First Frame)        H3 Max I2V output

[2510f347-0eaa-468f-8bdb-0d085320e766]  

0-3s: the chef throws his head back mid-laugh as he tosses the wok, the tower of orange flame surging upward and the noodles arcing through the air before landing back in the pan. 3-6s: he swirls the ladle through the sizzling noodles, sparks and embers flicking off the flame as steam curls from the wok, his straw hat tilting with the motion. 6-10s: he plants the wok back down with a satisfied grin, the flame settling into a steady roaring plume behind him, a few stray noodles still drifting down into frame. Energetic hand-drawn animation style, bold flat colors, dynamic flame and motion lines, playful comedic energy, upbeat sizzling and crackling fire sounds.

Duration 10s, Resolution 768P, source: illustrated chef still · Open on Scenario

Product stills work just as well. A studio photo of a watch became a slow hero-angle rotation with light sweeping across the dial, nothing invented beyond what the prompt asked for:

      Source image (First Frame)        H3 Max I2V output

[2510f347-0eaa-468f-8bdb-0d085320e766]  

Duration 8s, Resolution 768P, product commercial style · Open on Scenario

Locking the Closing Shot with Last Frame

Last Frame only works alongside First Frame. Set both and the model animates a transition between the two exact images instead of improvising an ending. Here, a perched dragon still was paired with a
separately generated airborne pose as the Last Frame, and the model built a full launch-and-climb sequence that lands on it:

0-4s: the bronze-scaled dragon perched on the cliff edge slowly unfolds its wings, muscles rippling beneath its scales as it crouches lower. 4-8s: with a powerful downward beat of its wings it launches off the rock into the golden hour sky, talons releasing from the stone as dust and loose pebbles scatter below. 8-12s: it banks upward against the sunset, wings fully spread and flapping steadily as it climbs over the misty coastline, a deep guttural roar echoing across the valley.

Duration 12s, Resolution 768P, First Frame: perched dragon, Last Frame: airborne dragon · Open on Scenario

Last Frame only takes effect when a First Frame is also set. Without a First Frame, Last Frame has no visible effect.

A few more I2V examples across verticals:

Classroom robot assistant, education

Lighthouse keeper, film/narrative

Surfer in the barrel, sports/action

Cinematic Showcase

H3 Max is not limited to simple product turns. It follows aggressive, named camera moves (barrel-roll orbits, Hitchcock dolly-zooms, drone launches, crash-zooms) as reliably as it follows plain
description. The clips below all came from this article's test batch, full prompts included.

A getaway driver in a black leather jacket grips the wheel of a matte-black muscle car inside a rain-slicked, neon-lit parking garage at night. 0-3s: tires screech as he yanks the handbrake into a full 180-degree drift, sparks flashing off the rear bumper against a concrete pillar. 3-6s: the camera whips into a full barrel-roll orbit around the spinning car, neon pink and blue reflections streaking across the wet floor and windshield. 6-9s: the car snaps straight and launches forward through the exit ramp, tires smoking, engine roaring under screeching rubber and distant sirens. High-octane heist-film cinematography, anamorphic lens flares, high contrast neon and shadow, dynamic orbiting camera.

T2V, barrel-roll orbit camera, Aspect Ratio 21:9 · Open on Scenario

A window washer in a harness stands on a narrow platform against the glass facade of a skyscraper, hundreds of feet above a city at dusk. 0-3s: he leans out to wipe a streak of glass, wind tugging at his jacket, city lights twinkling far below. 3-7s: the camera performs a slow dolly-zoom, pulling back while zooming in, the background plunging into vertiginous distortion as his eyes widen slightly at the drop beneath him. 7-10s: he steadies himself, exhales, and resumes wiping calmly as the skyline glows gold in the fading light, wind and distant traffic hum underneath. Vertigo-style Hitchcock zoom, warm dusk lighting, razor-sharp skyline detail, unsettling scale.

T2V, Hitchcock dolly-zoom, Aspect Ratio 16:9 · Open on Scenario

0-4s: the camera begins a slow 360-degree orbit around the model as the liquid mercury dress ripples and reforms in constant slow motion, silver highlights sliding across its surface like living metal. 4-7s: as the orbit completes, droplets of mercury lift briefly off the hem and hover before falling back into the fabric, rim light catching every ripple. 7-9s: she tilts her head slightly, the dress settling into a final rippling wave down its length, a low resonant hum and soft metallic chime underneath. High-fashion editorial cinematography, dramatic rim lighting, full uninterrupted orbital camera move, dark and glossy.

I2V, full 360-degree orbit, from a liquid-mercury fashion still · Open on Scenario

T2V, drone skim into vertical launch, bioluminescent wave

T2V, chase camera whip-pan, wingsuit canyon dive

I2V, macro crash-zoom to vertigo pull-out, lava cake

A few more from the same batch, spanning scale from zero-gravity to extreme macro:

T2V, full orbital barrel roll, zero-gravity EVA

T2V, low-angle push-in to whip-pan reveal, storm chaser

T2V, slow dolly-in, self-folding paper crane

T2V, tracking shot cut to onboard POV, race chicane

I2V, deep glide to upward tilt reveal, ice cave

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Examples (MiniMax H3)

All clips below were generated with MiniMax H3 (model_minimax-h3) in Public Data.

Reference Images: desert fashion campaign

Four stills (car, portrait, handbag, brand mark) passed as Reference Images for a premium editorial brand film.

Desert Fashion Campaign · Reference Images · Open on Scenario

Text to video: kitchen creature

Prompt-only smoke test: lived-in kitchen at dusk with soft hand-drawn luminous creatures and quiet room-tone audio.

Hand-Drawn Kitchen Creature · Text to Video · Open on Scenario

Reference Images: game equipment UI

Character key art plus HUD panels as references for an interactive game inventory reveal.

Interactive Game Equipment UI · Reference Images · Open on Scenario

Dialogue: vertical family confrontation

9:16 short drama with explicit spoken English lines in the Prompt and Reference Images for the two-shot and close-up.

Vertical Family Confrontation · Dialogue · Open on Scenario

Dialogue: cafe reunion

9:16 reunion scene with clear synced dialogue under rainy cafe light.

Cafe Reunion · Dialogue (9:16) · Open on Scenario

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parameters (MiniMax H3)

These are the dials you will touch most often on MiniMax H3.

prompt

Required. Describe the video: subject, action, camera, lighting, and audio. For dialogue, write the spoken lines in English inside the Prompt so lip sync has a clear target. Max length 7000
characters.

firstFrameImage

Optional image used as the opening frame. The video shape follows this image. Required if Last Frame is set. Cannot combine with Reference Images, Videos, or Audio.

lastFrameImage

Optional closing frame. Only works when First Frame is also set. Cannot combine with reference media.

referenceImages

Optional. Up to 9 images (max 30 MB each) to guide subjects or style. First 5 are free; additional images are billed. Cannot combine with First Frame or Last Frame.

referenceVideos

Optional. Up to 3 videos to guide motion (each 2 to 15 seconds, 2 to 15 seconds total, max 50 MB each). Billed per second of uploaded duration. Cannot combine with First Frame or Last Frame.

referenceAudio

Optional. Up to 3 audio clips (each 2 to 15 seconds, 2 to 15 seconds total, max 15 MB each). Requires at least one reference image or video.

duration

Clip length in seconds, from 5 to 15. Default 5. Longer clips cost more.

aspectRatio

Output shape: 21:9, 16:9, 4:3, 1:1, 3:4, 9:16, or Auto (adaptive in the API, the default). Ignored when First Frame is set, because the shape follows that image.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parameters (MiniMax H3 Max)

T2V and I2V share most of the same controls. Differences are called out below.

prompt

Required on both models, up to 7,000 characters. Write it as a full, detailed paragraph, ideally broken into timestamped beats (0 to 3s, 3 to 6s, and so on) describing action, camera, and mood. See
the firefighter and fisherman examples above.

firstFrameImage

I2V only, required. The opening frame of the video. The output's shape follows this image, so there is no separate Aspect Ratio control on I2V. See the chef and watch examples above.

lastFrameImage

I2V only, optional. The closing frame of the video. Only takes effect when firstFrameImage is also set. See the dragon launch example above.

duration

5 to 15 seconds on both models. Longer durations cost more. The fisherman example above used the full 15 seconds for a small multi-beat story in one clip.

resolution

480P or 768P, default 768P, on both models. Higher resolution costs more. 480P is a fast, cheap option for drafts and quick product loops (see the sneaker example above); 768P is the better default
for hero shots.

aspectRatio

T2V only, default 16:9. Choose from 21:9, 16:9, 4:3, 1:1, 3:4, or 9:16. See the demo grid above for 9:16, 21:9, and 1:1 in practice.

promptExpansionMode

Disabled, Balanced (default), or Quality, on both models. Disabled skips prompt rewriting entirely. Balanced spends about a second on it. Quality spends up to about 30 seconds letting the model
enrich a shorter prompt before generating, which added convincing background detail in the cyberpunk alley example above.

seed

Optional on both models. Leave blank for a random seed, or set one for reproducible results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use Cases

  • Brand and fashion films: Multi-still Reference Images for campaign continuity across mood, talent, product, and end card.

  • Product and ecommerce: Orbit and hero reveals for chairs, eyewear, sneakers, and pack shots with native ambience.

  • Story and dialogue shorts: Vertical 9:16 beats with explicit spoken lines for social and drama concepts.

  • Game and UI trailers: Character plus HUD references for inventory reveals and concept trailers.

  • Motion design and stylized spots: Text-only or image-guided clips for neon, anime, and editorial looks.

  • Film and teaser language: Space opera and mystery teasers with orchestral or drone beds generated in-pass.

  • Fast product and social loops (H3 Max): the watch, sneaker, and perfume I2V examples turn a plain product still into a hero rotation without a physical shoot, in a fraction of the time and cost
    of a reference-driven H3 render.

  • Bringing illustrated or concept art to life (H3 Max): the wok chef example shows H3 Max preserving a non-photoreal art style rather than pulling the output toward photorealism.

  • Marketing and advertising concepts across art styles (H3 Max): flat-illustration ads, retro synthwave spots, and stop-motion or claymation-style pieces all hold their intended style rather than
    defaulting to realism.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tips for Better Results

  • Prefer Reference Images for showcase work. Generate clean stills first (one scene per still), then pass those asset IDs into Reference Images.

  • Write dialogue as quoted lines in the Prompt. Keep lines short and assign speakers so lip sync stays readable.

  • Do not mix modes. First/Last Frame and Reference Images/Videos/Audio are mutually exclusive.

  • Match Aspect Ratio to the story. Use 9:16 for vertical dialogue; 16:9 for brand and product films. Auto is fine when you are exploring.

  • Describe audio explicitly. Name room tone, music bed, SFX, and whether dialogue should lead the mix.

  • Avoid packing multiple unrelated scenes into one Prompt without references. Split concepts across stills or separate runs.

  • Write prompts as timestamped beats. Breaking the action into 0-3s, 3-6s, and so on gave consistently more coherent motion than a single unstructured description across every example in this
    batch.

  • Give every beat a clear end state. Actions that trail off without a visible resolution risk the model losing track of the object; the wok chef and dragon examples both close each beat on a
    concrete pose.

  • Ground I2V prompts in the actual source image. The wok chef and watch examples describe exactly what is already in the still (the straw hat, the flame, the dial layout) rather than a generic
    scene.

  • Use Last Frame for a specific ending, not just a starting point. Pairing a matched First and Last Frame, as in the dragon example, gets a coherent full sequence instead of an improvised close.

  • Reach for Prompt Expansion: Quality on thinner prompts. It filled in convincing environmental detail on the cyberpunk alley clip beyond what was explicitly written.

  • Match resolution to the job. 480P was plenty for the sneaker product loop; save 768P for hero shots where detail matters.

  • Push duration to 15 seconds for anything with more than one beat. The fisherman example used the full range to fit a small four-beat story in a single clip.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Known Limitations

  • Duration is capped at 15 seconds. There is no schema-supported extend-to-30s path on this model.

  • Reference Audio requires at least one reference image or video.

  • First Frame / Last Frame cannot be combined with Reference Images, Videos, or Audio.

  • Aspect Ratio is ignored when First Frame is set.

  • Provider jobs can occasionally fail with a temporary internal error; retry the same payload.

  • On-screen readable logos and brand text can drift; keep brand marks abstract or out of frame when fidelity matters.

  • Audio is real but ambience/SFX-oriented, not reliable for dialogue. H3 Max does generate synced stereo audio matching the scene (footsteps, engines, wind, crowd noise, and similar), confirmed by
    measuring output levels across this batch. It is less consistent at rendering clear spoken dialogue than the base MiniMax H3 model, so write prompts expecting ambience and SFX first, with
    dialogue as a bonus rather than a guarantee.

  • No reference images or reference videos. H3 Max only accepts a text prompt (T2V) or a first/last frame image (I2V). For multi-image style or subject consistency, use the base MiniMax H3 model
    instead.

  • I2V has no Aspect Ratio control. The output shape always follows the First Frame image, so plan the source image's orientation before generating.

  • Last Frame is inert without First Frame. Setting only Last Frame has no effect; both must be set together.

  • Prompt Expansion: Quality adds latency. It can spend up to about 30 seconds rewriting the prompt before generation even starts, on top of generation time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MiniMax H3 Max Turbo: Faster Still

MiniMax H3 Max Turbo is Fal's second pass on H3 Max: the same two models, model_minimax-h3-max-turbo-t2v and model_minimax-h3-max-turbo-i2v, tuned again for even stronger prompt adherence and faster
inference. The parameter surface is identical to H3 Max (Duration, Resolution, Prompt Expansion, plus Aspect Ratio on T2V and First/Last Frame on I2V), so everything in the sections above about
writing prompts as timestamped beats applies directly. Turbo is the pick when a detailed prompt needs to land exactly as written, fast.

Prompt Detail Changes Everything

The same subject, shot three ways with Prompt Expansion disabled, shows how much the model actually listens. A one-line prompt gets a generic, forgettable read. One paragraph of detail (bike type,
camera move, lighting, sound) gets a materially closer result. A full timestamped shooting script gets a shot-for-shot match, down to the exact camera move scripted for the closing beat.

Simple: "A motorcyclist riding down a desert road at sunset." No scene, bike, or camera detail specified. · Open on Scenario

Detailed: one paragraph naming the bike, jacket, camera move, and lighting. · Open on Scenario

Very detailed: a timestamped 3-beat script. The clip ends on the exact scripted beat, a wide aerial pull-back with the bike a small silhouette and its shadow stretched across the rock. · Open on
Scenario

Prompt Expansion: Quality Turns a Short Brief Into a Full Scene

Prompt Expansion set to Quality took a two-sentence brief and filled in the atmosphere, steam, sparks, and neon signage on its own, at 480P for a quick, inexpensive social clip:

Quality-mode Prompt Expansion from a short brief, Duration 8s, Resolution 480P · Open on Scenario

Across Genres, Fast

H3 Max Turbo holds up across very different visual languages in the same batch: sci-fi, epic fantasy, 2D anime, emotional drama, and on-screen text.

Sci-fi EVA repair, slow orbital camera drift · Open on Scenario

Epic fantasy creature reveal, wide dramatic camera · Open on Scenario

2D anime style, dynamic rooftop chase, vertical framing · Open on Scenario

Emotional drama, slow push-in · Open on Scenario

On-screen text rendered exactly as scripted, slow orbital camera · Open on Scenario

MiniMax H3 Max Turbo I2V: Animating Stills, Fast

Turbo's image-to-video side takes the same First Frame plus optional Last Frame inputs as H3 Max I2V, and stays faithful to the source, whether it is illustrated art or a plain studio product shot.

Illustrated art style preserved, gentle character animation · Open on Scenario

Product rotation reveal from a single studio still · Open on Scenario

Travel photography brought to life with a slow drone push · Open on Scenario

Turbo is not a different feature set from H3 Max, just a faster, more literal read of the same prompt. When you want the model to hedge less and follow more, reach for Turbo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MiniMax H3 Max Lip Sync: Animate a Photo to Speech

MiniMax H3 Max Lip Sync is the odd one out in this family: it does not generate a scene from a prompt, it animates a still image to match a specific audio recording. Feed it a portrait and a 5 to 15
second clip of speech, and it renders natural, phoneme-accurate mouth movement synced to that exact audio, no source video and no scripted prompt required.

How It Works

Two inputs, both required. Image is a still of the character or person to animate, with an aspect ratio between 0.4 and 2.5 (tall portrait to wide landscape); the output video takes its shape from
this image. Audio is the voice track to lip-sync to, 5 to 15 seconds long (anything above 15 seconds is trimmed). There is no text prompt field at all, so the entire result comes from the pairing of
image and audio you provide.

A front-facing, sharp, evenly lit portrait looking at the camera gives the most natural sync. Profile shots, masks, or hair covering the mouth and jaw noticeably degrade the result. The model is not
limited to photoreal faces either: stylized illustrations, 3D character renders, and even non-human mascots animate cleanly as long as the face is clear and forward-facing.

Examples Across Use Cases

The clips below all came from this article’s test batch. Prompts are not shown because the model takes none, only the source image and audio.

Marketing spokesperson, Resolution 768P, Enable Transcription on · Open on Scenario

Esports commentator, Resolution 1080P, games/live-event use case · Open on Scenario

Documentary narrator, Resolution 768P, Enable Transcription off (raw audio-only sync) · Open on Scenario

Spanish-language marketing spokesperson, Resolution 768P · Open on Scenario

French-language tutor, Resolution 768P, natural laugh and expression carried through · Open on Scenario

Stylized plush mascot character, Resolution 768P, non-photoreal source image · Open on Scenario

Sci-fi android character, Resolution 1080P, Enable Transcription off · Open on Scenario

Film trailer voice-over, Resolution 2K, fixed Seed for reproducibility · Open on Scenario

Parameters

The parameter surface is small and every field matters.

image

Required. A still of the character or person to animate. Aspect ratio must be between 0.4 and 2.5 (roughly tall portrait to wide landscape); the output video shape follows this image. A sharp,
front-facing, evenly lit face gives the most natural mouth movement, see the fox mascot example above for how well this holds up on a non-photoreal source.

audio

Required. The voice track to lip-sync to. Must be between 5 and 15 seconds. In practice the platform enforces roughly 5 to 14.8 seconds, audio outside that window fails generation rather than being
silently clamped, so trim or pad short clips before sending them.

resolution

Optional, default 768P. Options are 480P, 768P, 1080P, and 2K. Aspect ratio always follows the source image regardless of resolution. The film trailer example above used 2K for the sharpest result.

enableTranscription

Optional, default false (off). When on, the audio is transcribed first and that transcript guides lip synchronization alongside the raw waveform. When off, the mouth syncs to the audio signal alone.
Both modes produced clean, natural results in testing, see the documentary narrator and sci-fi android examples above, which ran with transcription off.

seed

Optional. Fixes randomness for reproducible results across repeat generations with the same image and audio. Leave blank for a random seed each run.

Tips for Better Results

 1. Use a front-facing, evenly lit portrait. Direct eye contact with the camera and clear, unobstructed features produce the most natural mouth movement. Side angles and shadowed faces read as
    noticeably stiffer.

 2. Keep the source expression neutral, mouth closed. A calm starting expression gives the model room to animate speech without fighting an existing smile or open mouth in the source still.

 3. Check your audio duration before sending it. The model requires roughly 5 to 14.8 seconds and fails outside that range rather than trimming automatically, verify length with any audio tool first.

 4. Use clean, single-speaker audio. Background music or overlapping voices reduce sync accuracy, since the model has only the waveform (and optionally a transcript) to work from.

 5. Try both Enable Transcription settings. Transcription on tends to help with less clean or accented audio; transcription off works well for clear, single-speaker recordings and skips the extra
    processing step.

 6. Non-photoreal art styles work fine. Illustrated characters, 3D renders, and mascots animate as cleanly as photographic portraits, as long as the face is clear and forward-facing.

 7. The model is language-agnostic. It follows phonemes from the audio itself, so it syncs equally well to non-English speech, see the Spanish and French examples above.

Known Limitations

  • No text prompt control. The entire output is determined by the image and audio you provide, there is no way to direct camera, background motion, or performance style through text.

  • Hard audio window. Audio must fall roughly between 5 and 14.8 seconds. Shorter or longer clips are rejected, so long-form dialogue needs to be split into segments and generated separately.

  • Single subject only. The model animates one face per generation. Multi-character dialogue in one shot is not supported, generate each character separately and edit the clips together.

  • Output shape is fixed to the source image. There is no independent aspect ratio control, crop or reframe the source image first if you need a specific output shape.

MiniMax H3 Max 3D to Video: Turn Previs into Video

MiniMax H3 Max 3D to Video is a video-to-video model for 3D artists. You upload a Blender render or a 3D previs clip, and it generates a photorealistic video at up to 1080P that follows your source.
The layout, camera movement, and timing carry over from the clip, while simple blocks and stand-ins become the scene you describe. This turns cinematic blockouts and layout passes into finished
shots.

How It Works

Video is the only required input: a Blender render or 3D previs clip of up to 15 seconds and 32 shots, up to 50 MB. It sets the layout, camera, motion, and timing, and the output has the same length
as the clip. Use Prompt (optional) to explain what the proxies represent or how they move, for example: "The moving block is a running person."

For the look of the scene, add up to 8 Reference Images of the environment, subjects, or details you want. When you supply references, no new ones are generated. When you do not, the model can
generate up to Max Generated Reference Images of its own (1 to 8, default 2) to guide the look.

Parameters

video

Required. Your Blender render or 3D previs clip, up to 15 seconds and 32 shots. It is billed per second of video, with a 5-second minimum.

prompt

Optional, up to 2,000 characters. Describe what simple shapes in your previs stand for and how they move.

referenceImages

Optional, up to 8 images. Environment, subject, or detail references that guide the look. Each image adds to the cost.

maxGeneratedReferenceImages

Optional, 1 to 8, default 2. How many reference images the model may generate when you supply none. It is ignored when you provide reference images, and more generated references cost more.

resolution

Optional, default 768P. Options are 480P, 768P, and 1080P. Higher resolutions cost more.

Tips for Better Results

 1. Lock your camera and timing in 3D first. The model keeps them exactly, so fix them in your previs before you generate.

 2. Name your stand-ins in the prompt. If a moving block is a person or a vehicle, say so, so the model gives it the right motion and shape.

 3. Supply reference images when the look matters. Your own references keep the environment and subjects on model, instead of letting the model invent them.

 4. Test at 480P, then finish at 1080P. Higher resolutions cost more, so a low-resolution pass shows whether the shot reads before you pay for the final.

Related Articles
Wan 3.0: The Essentials How to Choose the Right Video Model (2026 Edition) Wan 2.7: Complete Video Generation Suite Happy Horse 1.1: The Essentials
Scenario Knowledge Base Powered by [logo_subtl]Pylon
