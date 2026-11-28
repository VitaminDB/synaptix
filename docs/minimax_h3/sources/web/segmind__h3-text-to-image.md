# Source: https://www.segmind.com/models/minimax-h3-text-to-image

Fetched: 2026-11-28 (curl + w3m text dump)

---

(Switch to classic interface)
(Get API code)(Edit request as JSON)(Parameter info)
Try an example4
(Example 1)[minimax-h3]1
(Example 2)[minimax-h3]2
(Example 3)[minimax-h3]3
(Example 4)[minimax-h3]4
AboutGuideExamplesFAQsResources
AboutGuideExamplesFAQsResources

MiniMax H3 Text to Image (Text-to-Image Model)

What is MiniMax H3 Text to Image?

MiniMax H3 Text to Image is MiniMax's H3-Image model for turning a text prompt into a high-resolution still image. Built on the H3 architecture, it generates natively at about 16 megapixels
(4096x4096) in a single pass, with no separate upscaler, and follows detailed instructions closely, including legible in-image text and precise layouts. You can request 1 to 10 images per call,
choose any aspect ratio from 1:3 to 3:1, and pick a standard or highspeed tier.

Key Features

  • •Native ~16 MP output (up to about 20 MP) in one pass, no upscaler
  • •Legible in-image text: put words in quotes and they render correctly spelled
  • •Strong instruction following for multi-element scenes and poster or product layouts
  • •Aspect ratios from 1:3 to 3:1, or an exact WIDTHxHEIGHT
  • •1 to 10 images per request; jpeg, png or webp output
  • •Optional prompt enhancement, with the rewritten prompt returned

Best Use Cases

Reach for it when text has to be readable: posters, packaging and product mockups, signage, menus, title cards and information boards. Our test rendered a shop header and a row of labels with every
word correctly spelled. The 16 MP native output suits print and large-format work, and the wide 1:3 to 3:1 range covers banners, billboards and tall social formats without cropping. It also handles
photoreal scenes, concept art, characters and illustration styles.

Prompt Tips and Output Quality

Write specific prompts with concrete nouns, and wrap any text you want rendered in single quotes. Keep enhancement on for richer scenes, or turn it off to use your prompt verbatim (then set an aspect
ratio or an exact size). Choose 4K for crisp small text and fine detail, and a less-wide ratio to keep lettering large. Output in our tests was clean and artifact-free at 4K.

How to Use MiniMax H3 Text to Image

MiniMax H3 Text to Image turns a written description into a high-resolution image. The model reads your prompt, optionally rewrites it to add detail, and renders natively at up to about 16
megapixels. It is strongest when a prompt is specific, and when you need words to appear correctly inside the image.

Writing the prompt

Name concrete subjects, materials, era and composition rather than stacking adjectives. For in-image text, put the exact words in single quotes (for example 'OPEN' or 'EST. 1894'); the model keeps
the spelling and places the text where you describe it. Order the sentence the way you want the scene read, from foreground to background.

Prompt enhancement

enhance_prompt is on by default and expands a short request into the detailed caption the model reads best; the rewritten prompt is returned so you can see exactly what was generated. Turn it off to
render your words verbatim, and then set an aspect ratio (not auto) or an exact size.

Choosing resolution and shape

area sets detail: use 4K for print, posters and crisp small text, 2K for faster everyday work, and 1K for drafts. Set aspect_ratio from 1:3 to 3:1, or give an exact size (WIDTHxHEIGHT, 655,360 to
20,000,000 pixels). A less-wide ratio such as 4:3 keeps lettering large and legible.

Recommended Settings

  • •Legible text, posters and signage: area 4K, an explicit aspect_ratio such as 4:3, enhancement on, and text in quotes. This is how the default example, a labelled herbalist's cabinet, rendered
    every word correctly.
  • •Print and large format: 4K, with an exact size for the final dimensions.
  • •Fast iteration: 2K, num_images up to 10 to compare options, and the highspeed speed tier.
  • •Pick output_format png for sharp text or transparency, jpeg for smaller files.

Examples to try

Each example below ran end to end through the model at the settings shown - area 4K, standard speed and prompt enhancement on - and the card plays the exact prompt back into the form. Only the aspect
ratio changes from one to the next.

  • •

    Legible in-image text and layout. Prompt: "A straight-on view of a Victorian herbalist's cabinet: a grid of small oak drawers, each with a round brass cup handle and a cream card label in a brass
    frame. The front labels read 'VALERIANA', 'LAVANDULA', 'CALENDULA' and 'ARNICA' in black serif capitals. A hand-painted wooden header board above the drawers reads 'HOLLOWAY HERBALIST' with 'EST.
    1894' beneath in gold leaf. A brass mortar and pestle and a set of apothecary scales rest on the counter below. Even daylight from a side window, aged wood grain and tarnished brass." Every
    requested word renders correctly spelled and placed, from the gold-leaf header to the four drawer labels. Quote the exact words in single quotes and keep a less-wide ratio like 4:3 so lettering
    stays large.

  • •

    Ultra-wide 3:1 framing. Prompt: "An ultra-wide view straight down a car-factory assembly line: a long row of unpainted silver sedan shells on a steel conveyor recedes to a vanishing point,
    flanked on both sides by orange robotic welding arms frozen mid-weld, each throwing a short burst of white sparks. Polished concrete floor, yellow painted safety lines, cool even overhead
    lighting, no people." The 3:1 output (7104x2368) fills the banner edge to edge with no crop. Set aspect_ratio anywhere from 1:3 to 3:1 up front, and give the scene depth so the wide frame stays
    busy.

  • •

    Photoreal material realism. Prompt: "A cast-iron pan of saffron paella on a dark slate counter, shot three-quarter from above: glossy mussels and seared prawns on golden rice, a charred socarrat
    crust at the pan's edge, two lemon wedges and a scatter of chopped parsley, faint steam rising. Side daylight, visible grains of rice and an oil sheen." Shellfish sheen, separate rice grains and
    rising steam read as a real photograph. Name the materials and one light direction, and keep enhancement on to fill in fine texture.

  • •

    Illustration and concept-art styles. Prompt: "A gouache concept illustration of a whale-shaped airship drifting low over terraced green rice paddies, its wooden gondola slung beneath on rope
    rigging, two tiny farmers looking up from a flooded terrace. Loose visible brushstrokes, flat poster-like colour, soft paper grain." Visible brushwork, flat poster colour and paper grain keep it
    clearly painted, not photoreal. State the medium first so the whole frame commits to the style.

Other prompts the model is built for: "a vertical 9:16 movie poster with the title in quotes", "a product-label mockup on a frosted glass jar", "a watercolour storybook spread of a fox and a copper
kettle".

MiniMax H3 Text to Image examples

Requests run on MiniMax H3 Text to Image in the playground: what each was given, what came back, and the API call that makes it.

 1. 1.MiniMax H3 Text to Image typography and sign design
 2. 2.MiniMax H3 Text to Image stock photo generator
 3. 3.MiniMax H3 Text to Image food photography generator
 4. 4.MiniMax H3 Text to Image concept art generator

MiniMax H3 Text to Image typography and sign design

Prompt

    A straight-on view of a Victorian herbalist's cabinet: a grid of small oak drawers, each with a round brass cup handle and a cream card label in a brass frame. The front labels read 'VALERIANA',
    'LAVANDULA', 'CALENDULA' and 'ARNICA' in black serif capitals. A hand-painted wooden header board above the drawers reads 'HOLLOWAY HERBALIST' with 'EST. 1894' beneath in gold leaf. A brass
    mortar and pestle and a set of apothecary scales rest on the counter below. Even daylight from a side window, aged wood grain and tarnished brass.

Settings

speed
    standard
num_images
    1
area
    4K
aspect_ratio
    4:3

Try it in the playground

A straight-on view of a Victorian herbalist's cabinet: a grid of small oak drawers, each with a round brass cup handle and a cream card label in a brass frame.…
Output from MiniMax H3 Text to Image

API request

Python SDKcURL
(Copy Python SDK code)Copy

# pip install "segmind>=1.1.0"
# export SEGMIND_API_KEY="YOUR_API_KEY"
import segmind

result = segmind.run(
    "minimax-h3-text-to-image",
    prompt="A straight-on view of a Victorian herbalist's cabinet: a grid of small oak drawers, each with a round brass cup handle and a cream card label in a brass frame. The front labels read 'VALERIANA', 'LAVANDULA', 'CALENDULA' and 'ARNICA' in black serif capitals. A hand-painted wooden header board above the drawers reads 'HOLLOWAY HERBALIST' with 'EST. 1894' beneath in gold leaf. A brass mortar and pestle and a set of apothecary scales rest on the counter below. Even daylight from a side window, aged wood grain and tarnished brass.",
    speed="standard",
    num_images=1,
    area="4K",
    aspect_ratio="4:3",
)
print(result.get("output"))

# 1. Submit the request; the reply carries its request_id
REQUEST_ID=$(curl -s -X POST "https://api.segmind.com/v2/minimax-h3-text-to-image" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "prompt": "A straight-on view of a Victorian herbalist'\''s cabinet: a grid of small oak drawers, each with a round brass cup handle and a cream card label in a brass frame. The front labels read '\''VALERIANA'\'', '\''LAVANDULA'\'', '\''CALENDULA'\'' and '\''ARNICA'\'' in black serif capitals. A hand-painted wooden header board above the drawers reads '\''HOLLOWAY HERBALIST'\'' with '\''EST. 1894'\'' beneath in gold leaf. A brass mortar and pestle and a set of apothecary scales rest on the counter below. Even daylight from a side window, aged wood grain and tarnished brass.",
  "speed": "standard",
  "num_images": 1,
  "area": "4K",
  "aspect_ratio": "4:3"
}' | jq -r .request_id)

# 2. Check its status until it is COMPLETED
curl -s "https://api.segmind.com/v2/requests/$REQUEST_ID/status" \
  -H "x-api-key: YOUR_API_KEY"

# 3. Fetch the result
curl -s "https://api.segmind.com/v2/requests/$REQUEST_ID" \
  -H "x-api-key: YOUR_API_KEY"

MiniMax H3 Text to Image stock photo generator

Prompt

    An ultra-wide view straight down a car-factory assembly line: a long row of unpainted silver sedan shells on a steel conveyor recedes to a vanishing point, flanked on both sides by orange robotic
    welding arms frozen mid-weld, each throwing a short burst of white sparks. Polished concrete floor, yellow painted safety lines, cool even overhead lighting, no people.

Settings

speed
    standard
num_images
    1
area
    4K
aspect_ratio
    3:1

Try it in the playground

An ultra-wide view straight down a car-factory assembly line: a long row of unpainted silver sedan shells on a steel conveyor recedes to a vanishing point, fla…
Output from MiniMax H3 Text to Image

API request

Python SDKcURL
(Copy Python SDK code)Copy

# pip install "segmind>=1.1.0"
# export SEGMIND_API_KEY="YOUR_API_KEY"
import segmind

result = segmind.run(
    "minimax-h3-text-to-image",
    prompt="An ultra-wide view straight down a car-factory assembly line: a long row of unpainted silver sedan shells on a steel conveyor recedes to a vanishing point, flanked on both sides by orange robotic welding arms frozen mid-weld, each throwing a short burst of white sparks. Polished concrete floor, yellow painted safety lines, cool even overhead lighting, no people.",
    speed="standard",
    num_images=1,
    area="4K",
    aspect_ratio="3:1",
)
print(result.get("output"))

# 1. Submit the request; the reply carries its request_id
REQUEST_ID=$(curl -s -X POST "https://api.segmind.com/v2/minimax-h3-text-to-image" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "prompt": "An ultra-wide view straight down a car-factory assembly line: a long row of unpainted silver sedan shells on a steel conveyor recedes to a vanishing point, flanked on both sides by orange robotic welding arms frozen mid-weld, each throwing a short burst of white sparks. Polished concrete floor, yellow painted safety lines, cool even overhead lighting, no people.",
  "speed": "standard",
  "num_images": 1,
  "area": "4K",
  "aspect_ratio": "3:1"
}' | jq -r .request_id)

# 2. Check its status until it is COMPLETED
curl -s "https://api.segmind.com/v2/requests/$REQUEST_ID/status" \
  -H "x-api-key: YOUR_API_KEY"

# 3. Fetch the result
curl -s "https://api.segmind.com/v2/requests/$REQUEST_ID" \
  -H "x-api-key: YOUR_API_KEY"

MiniMax H3 Text to Image food photography generator

Prompt

    A cast-iron pan of saffron paella on a dark slate counter, shot three-quarter from above: glossy mussels and seared prawns on golden rice, a charred socarrat crust at the pan's edge, two lemon
    wedges and a scatter of chopped parsley, faint steam rising. Side daylight, visible grains of rice and an oil sheen.

Settings

speed
    standard
num_images
    1
area
    4K
aspect_ratio
    4:3

Try it in the playground

A cast-iron pan of saffron paella on a dark slate counter, shot three-quarter from above: glossy mussels and seared prawns on golden rice, a charred socarrat c…
Output from MiniMax H3 Text to Image

API request

Python SDKcURL
(Copy Python SDK code)Copy

# pip install "segmind>=1.1.0"
# export SEGMIND_API_KEY="YOUR_API_KEY"
import segmind

result = segmind.run(
    "minimax-h3-text-to-image",
    prompt="A cast-iron pan of saffron paella on a dark slate counter, shot three-quarter from above: glossy mussels and seared prawns on golden rice, a charred socarrat crust at the pan's edge, two lemon wedges and a scatter of chopped parsley, faint steam rising. Side daylight, visible grains of rice and an oil sheen.",
    speed="standard",
    num_images=1,
    area="4K",
    aspect_ratio="4:3",
)
print(result.get("output"))

# 1. Submit the request; the reply carries its request_id
REQUEST_ID=$(curl -s -X POST "https://api.segmind.com/v2/minimax-h3-text-to-image" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "prompt": "A cast-iron pan of saffron paella on a dark slate counter, shot three-quarter from above: glossy mussels and seared prawns on golden rice, a charred socarrat crust at the pan'\''s edge, two lemon wedges and a scatter of chopped parsley, faint steam rising. Side daylight, visible grains of rice and an oil sheen.",
  "speed": "standard",
  "num_images": 1,
  "area": "4K",
  "aspect_ratio": "4:3"
}' | jq -r .request_id)

# 2. Check its status until it is COMPLETED
curl -s "https://api.segmind.com/v2/requests/$REQUEST_ID/status" \
  -H "x-api-key: YOUR_API_KEY"

# 3. Fetch the result
curl -s "https://api.segmind.com/v2/requests/$REQUEST_ID" \
  -H "x-api-key: YOUR_API_KEY"

MiniMax H3 Text to Image concept art generator

Prompt

    A gouache concept illustration of a whale-shaped airship drifting low over terraced green rice paddies, its wooden gondola slung beneath on rope rigging, two tiny farmers looking up from a
    flooded terrace. Loose visible brushstrokes, flat poster-like colour, soft paper grain.

Settings

speed
    standard
num_images
    1
area
    4K
aspect_ratio
    4:3

Try it in the playground

A gouache concept illustration of a whale-shaped airship drifting low over terraced green rice paddies, its wooden gondola slung beneath on rope rigging, two t…
Output from MiniMax H3 Text to Image

API request

Python SDKcURL
(Copy Python SDK code)Copy

# pip install "segmind>=1.1.0"
# export SEGMIND_API_KEY="YOUR_API_KEY"
import segmind

result = segmind.run(
    "minimax-h3-text-to-image",
    prompt="A gouache concept illustration of a whale-shaped airship drifting low over terraced green rice paddies, its wooden gondola slung beneath on rope rigging, two tiny farmers looking up from a flooded terrace. Loose visible brushstrokes, flat poster-like colour, soft paper grain.",
    speed="standard",
    num_images=1,
    area="4K",
    aspect_ratio="4:3",
)
print(result.get("output"))

# 1. Submit the request; the reply carries its request_id
REQUEST_ID=$(curl -s -X POST "https://api.segmind.com/v2/minimax-h3-text-to-image" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "prompt": "A gouache concept illustration of a whale-shaped airship drifting low over terraced green rice paddies, its wooden gondola slung beneath on rope rigging, two tiny farmers looking up from a flooded terrace. Loose visible brushstrokes, flat poster-like colour, soft paper grain.",
  "speed": "standard",
  "num_images": 1,
  "area": "4K",
  "aspect_ratio": "4:3"
}' | jq -r .request_id)

# 2. Check its status until it is COMPLETED
curl -s "https://api.segmind.com/v2/requests/$REQUEST_ID/status" \
  -H "x-api-key: YOUR_API_KEY"

# 3. Fetch the result
curl -s "https://api.segmind.com/v2/requests/$REQUEST_ID" \
  -H "x-api-key: YOUR_API_KEY"

Can it render readable text?

Yes, quote the exact words in your prompt and they appear correctly spelled.

What is the maximum resolution?

About 16 to 20 megapixels natively, generated in one pass without an upscaler.

Which aspect ratios are supported?

Any ratio from 1:3 to 3:1, or an exact WIDTHxHEIGHT size.

How many images can I generate per request?

From 1 to 10.

Does it return the enhanced prompt?

Yes, the rewritten prompt is returned as a revised prompt.

  • MiniMax H3-Image generation guide
    Reference
    how to prompt, output size/aspect-ratio/area, formats, references and limits for text-to-image and editing
  • MiniMax-H3-Image model card
    Reference
    the model's three highlights (16 MP, realism, instruction following), architecture, model zoo, showcase and license
  • MiniMax H3-Image text-to-image API reference
    Reference
    the full request schema - prompt, size, aspect_ratio, area, n, formats - plus streaming, error codes and the revised_prompt field
  • MiniMax H3-Image release post
    Reference
    the open-weight release, the 4K-in-10-seconds positioning and a capabilities overview
  • MiniMax-H3-Image on GitHub
    Reference
