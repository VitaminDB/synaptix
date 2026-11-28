# Source: https://lumalabs.ai/learning-center/articles/minimax-h3-complete-guide

Fetched: 2026-11-28 (curl + w3m text dump)

---

      □ A helpful reminder

MiniMax H3 Complete Guide

Start with the whole scene

A useful H3 prompt describes the creative result rather than listing disconnected keywords.

Instead of:

Woman, train, cinematic, rain, camera movement.

Try:

A cinematic close shot of a woman sitting beside the window of a moving train at night. Rain runs across the glass as city lights streak through her reflection. She looks down at a folded letter,
then slowly raises her eyes toward the window. The camera gently tracks to the right. Train wheels rumble underneath quiet rainfall, with sparse piano playing in the background.

The second version gives H3 relationships to work with: subject, environment, action, camera, timing, ambience, and music.

Choose the right starting point

H3 can work from several kinds of creative context.

T2V (text-to-video), describe the complete scene from scratch, text only.

I2V (image-to-video) with the image as a start frame, treat the image as the opening state. Describe what begins moving and how the scene develops from there.

I2V with start and end frames, focus on generating the "in-between". Describe the physical changes, camera movement, subject motion, and transitions that should naturally lead from the starting image
to the ending image.

R2V (reference-to-video) with media as references that contribute creative direction for different parts of the result. A reference image might define the character, a reference video might define
camera movement, and reference audio might define a voice or musical direction. MiniMax calls this its Omni Reference workflow.

Think beyond the picture

H3 generates audio together with video, so sound can be part of the original creative direction rather than an afterthought.

You can describe:

Dialogue
Without looking away from the window, she quietly says, ‘I think this is my stop.’

Environmental audio
Rain taps against the glass while the train produces a steady metallic rhythm.

Physical sounds
The paper softly unfolds in her hands.

Music
Sparse piano notes play at a slow tempo, joined by a sustained low cello.

The official H3 prompting guidance separates these ideas into the visual/action timeline, the overall soundscape, and non-diegetic music.

You do not need to make every prompt complicated. The useful principle is simply to think about the complete audiovisual moment.

Where H3 is especially useful

H3 is a strong fit when the video depends on several kinds of creative direction working together.

That includes character-driven scenes, commercials, product films, stylized content, music-driven work, narrative clips, motion references, dialogue, and scenes where sound is an important part of
the result. MiniMax's own examples span brand films, social creative, short-form narrative, product advertising, UI motion, games, and stylized animation.

More inputs do not automatically make a generation better.

Give each instruction or reference a clear job. If an image defines the character, a video defines motion, and audio defines the voice, say so. Clear relationships give the model a stronger creative
hierarchy than a collection of unexplained references.

Think of H3 as an audiovisual scene generator rather than only a video generator.

Describe what should happen, how it should look, how it should move, and what the audience should hear. When you use references, tell H3 what each reference contributes to the final result.

How to prompt MiniMax H3

A strong H3 prompt describes a video as a sequence of visible and audible events.

Instead of only describing a subject and a style, think through three layers:

What happens on screen → what happens in the soundscape → what music accompanies it.

This mirrors the structure MiniMax uses in its own H3 prompting system and helps give the model enough information to coordinate visuals, motion, dialogue, sound, and music.

1. Establish the scene

Start by defining the visual world.

Include the most important details about:

  • visual style
  • subject
  • environment
  • composition
  • lighting
  • important objects

For example:

Live-action cinematic footage. A young woman sits alone beside the window of a nearly empty train at night. Cool fluorescent light fills the carriage while blurred city lights pass outside.

You do not need to describe every visible object. Focus on details that should stay recognizable or influence what happens next.

2. Describe observable action

Next, describe what actually changes during the clip.

A folded letter rests in her hands. She slowly opens it, reads the first line, then raises her gaze toward the passing city.

Concrete actions give H3 a path through time.

When possible, describe physical changes rather than abstract intentions. “She tightens her grip on the letter and looks away” gives the model something more visible to generate than “she becomes
emotional.”

3. Direct the camera when it matters

H3 understands common camera language including push, pull, pan, truck, tilt, tracking, arc, static shots, camera shake, POV, and roll. MiniMax's prompting guide recommends thinking about camera
motion in terms of the movement itself, its range, and its speed.

For example:

The camera slowly pushes toward the letter in her hands.

The camera tracks beside her as she walks through the station.

The camera rapidly pans right to reveal the approaching train.

Treat camera direction as part of the scene rather than a list of technical tags.

4. Use cuts intentionally

If the video needs multiple shots, introduce a cut when the next shot reveals genuinely new information.

For example:

An establishing shot follows her through the crowded station. The scene cuts to a close-up as she stops and unfolds the letter.

If the only change is moving slightly closer or viewing the same action from a nearby angle, a camera move may be simpler than adding another shot.

H3 was trained with native multi-shot modeling, so a prompt can describe a sequence rather than forcing every idea into a single continuous composition.

5. Treat dialogue as part of the performance

Dialogue can be included directly in the scene description.

For example:

She glances toward the empty seat beside her and quietly says, ‘You would have liked this place.’

Include useful delivery information when it matters:

She says it softly, with a low, slightly breathy voice.

Keep the actual line concise enough to fit comfortably within the clip. The visual performance, spoken line, and surrounding action all need time to happen.

6. Describe the soundscape

Next, think about sounds that belong inside the physical scene.

For example:

Train wheels produce a steady metallic rhythm. Rain taps against the window. Paper rustles quietly as she unfolds the letter.

Useful soundscape details include footsteps, wind, fabric, machinery, impacts, room tone, traffic, breathing, water, crowds, doors, or object interaction.

Tie important sounds to the actions that create them.

7. Add music separately

If you want a score that exists for the audience rather than inside the scene, describe it separately from physical sound.

For example:

Sparse piano at a slow tempo, joined by sustained low strings. The music gradually decreases in volume toward the final frame.

MiniMax recommends describing concrete musical qualities such as instrumentation, tempo, rhythm, and dynamic change instead of relying entirely on abstract mood words.

Prompt example

Live-action cinematic footage. A woman in her late twenties sits beside a rain-covered train window at night, holding a folded handwritten letter. Cool carriage lighting contrasts with warm city
lights moving outside.

She slowly unfolds the letter and begins reading. The camera gently tracks to the right as her reflection moves across the wet glass. She pauses, raises her eyes toward the city, and quietly says, ‘I
get off at the next station.’ She folds the letter again along the existing crease.

Train wheels create a steady metallic rhythm beneath a low ventilation hum. Rain taps against the window and the paper softly rustles in her hands.

Sparse piano notes play at a slow tempo with sustained cello underneath, gradually fading toward the end.

When you're starting from an image

Treat the image as the established opening state.

Preserve the important identity, clothing, environment, composition, and objects from the image, then describe what starts changing.

A useful pattern is:

Opening image → first movement → continued action → final reaction.

When you're using first and last frames

Do not simply describe both images.

Describe the motion that connects them.

Think through:

Starting state → intermediate physical changes → approach toward the final composition → ending state.

This is also the pattern recommended in MiniMax's official H3 prompt guide.

A helpful reminder

Every instruction competes for time inside a short clip.

If you ask for four camera moves, three character actions, two lines of dialogue, a transformation, and several cuts, the model has to compress all of them into the available duration.

Prioritize the moments that matter most.

Key takeaway

Write H3 prompts as audiovisual timelines.

Establish the scene, describe the actions in order, direct the camera where necessary, and make sound and music explicit. The clearer the sequence of events is, the easier it is for H3 to understand
the video you are trying to create.

How to direct MiniMax H3 with image, video, and audio references

References are most useful when you tell H3 what to take from each one.

An image can establish a character. A video can establish movement. Another image can establish a location or product. Audio can provide a voice or musical reference.

H3 can understand those relationships together and use them to create a new audiovisual result.

The key is to assign each reference a clear creative role.

Reference appearance with images

Use an image when something needs to remain visually recognizable.

That might include:

  • a person or character
  • clothing
  • a product
  • an environment
  • a visual effect
  • an art direction
  • a composition

Instead of simply attaching an image and saying:

Use this.

Describe what matters:

Use Image 1 for the woman's appearance, hairstyle, clothing, and accessories.

Or:

Use Image 2 as the visual reference for the motorcycle design. Preserve the body shape, headlight, wheels, and red-and-black color treatment.

This tells H3 which properties are important to carry forward.

Reference motion with video

A video reference can contribute movement without requiring you to recreate its visual content.

For example:

Use Video 1 as the reference for the dancer's choreography and body timing.

Reference the slow circular camera movement from Video 1.

Use Video 1 for the editing rhythm and shot pacing.

H3's reference system can reason about actions, camera movement, temporal structure, and editing style as distinct parts of a video reference.

That means you can borrow how something moves without necessarily borrowing what it looks like.

Reference voice and audio

Audio can also provide creative direction.

For example:

Use Audio 1 as the reference for her voice timbre and delivery.

Use Audio 2 as the musical reference, preserving the beat and overall instrumentation.

Use Audio 1 as the soundtrack while generating new visuals that follow its rhythm.

Be precise about whether you want to reuse an audio signal or simply reference characteristics such as voice, rhythm, timbre, or musical style.

The official H3 reference guide treats those as different relationships.

Combine references by role

The real advantage appears when multiple references work together.

Imagine you have:

  • Image 1 — character
  • Image 2 — wardrobe
  • Video 1 — choreography
  • Video 2 — camera movement
  • Audio 1 — voice

Your prompt could say:

Create a cinematic performance featuring the woman from Image 1 wearing the outfit from Image 2. Use the choreography and body timing from Video 1. Follow the slow orbiting camera movement from Video
2. When she speaks, use Audio 1 as the reference for her voice timbre and calm delivery. Place the performance on a minimal dark stage with a single overhead spotlight.

Each reference contributes one part of the result.

H3 then has a much clearer map of what should be preserved and what can be newly generated.

Separate identity from performance

One particularly useful pattern is to separate who the character is from what the character does.

For example:

Use Image 1 for the character's appearance. Use Video 1 only for movement.

This allows the character from one source to perform an action from another.

The same idea works for camera:

Preserve the character and environment from Image 1, but reference only the camera motion from Video 1.

Or voice:

Keep the character from Image 1. Use Audio 1 only for voice timbre; use the new dialogue written in this prompt.

The more clearly these responsibilities are separated, the less ambiguity the model needs to resolve.

Use references for style and composition too

References do not have to represent literal subjects.

An image can provide:

  • lighting
  • color treatment
  • composition
  • texture
  • visual atmosphere

A video can provide:

  • camera language
  • cut structure
  • pacing
  • motion style

Audio can provide:

  • rhythm
  • voice qualities
  • sound texture
  • musical direction

This makes reference workflows useful for creative direction as well as consistency.

Example: product campaign

Create a premium 9:16 eyewear campaign.

Use Image 1 for the model's identity and Image 2 for the exact eyewear design. Preserve the glasses' wraparound silhouette, mirrored lenses, and narrow temples.

Use Video 1 for the model's confident walking pace and Video 2 for the low tracking camera movement.

The model crosses a minimal white studio while the camera tracks beside her. Reflections move across the lenses as she turns toward camera for the final close-up.

Sharp footsteps echo through the studio. A restrained electronic beat plays underneath with a steady tempo and clean percussion.

Start with fewer references

H3 can reason across many inputs, but that does not mean every generation needs them.

Start with the minimum set required to express the idea.

If character identity matters, start with the character reference.

If motion is missing, add a motion reference.

If the voice matters, add the voice reference.

Build the reference stack around specific problems instead of attaching additional assets simply because they are available.

A helpful reminder

References work best when their responsibilities are explicit.

If two references both appear to define the same character, style, motion, or composition without further explanation, H3 has to decide how those signals should be reconciled.

Tell the model what matters in each source and what is allowed to change.

Treat references like members of a creative team.

Give every image, video, or audio asset a job: identity, appearance, motion, camera, composition, voice, sound, rhythm, or style.

The clearer those roles are, the easier it becomes to combine multiple sources into one controlled H3 generation.

Try Luma Now
Luma
