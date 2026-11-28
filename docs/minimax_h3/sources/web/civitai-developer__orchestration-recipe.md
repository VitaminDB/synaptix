# Source: https://developer.civitai.com/orchestration/recipes/minimax-h3

Fetched: 2026-11-28 (curl + w3m text dump)

---


Flux 1 LoRA training

Flux 2 Klein LoRA training

Wan video LoRA training

LTX2 video LoRA training

Chroma / ERNIE / Qwen / Z-Image LoRA training

On this page

Copy page
Are you an LLM? You can read better optimized documentation at /orchestration/recipes/minimax-h3.md for this page in Markdown format

MiniMax H3 video generation ​

MiniMax's H3 model, called directly against MiniMax rather than through a broker. It generates 5–15 second clips at 2K (2560×1440) at 24 fps with a native stereo audio track, and covers five input
modes: text-to-video, first-frame, last-frame, first-and-last-frame, and reference-to-video with up to nine reference images plus three video and three audio clips.

Unlike most engines here there is no operation discriminator — the mode is inferred from which media fields you populate:

        Mode                                         Populate                                           aspectRatio
Text-to-video        nothing but prompt                                                       a concrete ratio (not adaptive)
Image-to-video       firstFrameImage                                                          ignored — derived from the image
Last-frame           lastFrameImage                                                           ignored
First and last frame both firstFrameImage and lastFrameImage                                  ignored
Reference-to-video   any of referenceImages (≤9), referenceVideos (≤3), referenceAudios (≤3)  ignored

Default choice: resolution: "2K", duration: 5, aspectRatio: "16:9". Every H3 job runs for minutes, longer than your HTTP client's timeout — always submit with wait=0.

The request shape ​

A single videoGen step on SubmitWorkflow. engine is the only discriminator:

json

{
  "$type": "videoGen",
  "input": {
    "engine": "minimax-h3",
    "prompt": "…",
    "resolution": "2K",
    "duration": 5
  }
}

Text-to-video ​

Text-to-video is the one mode that requires a concrete aspectRatio — MiniMax rejects adaptive when there is no image to derive geometry from. If you leave it at the default, 16:9 is substituted for
you.

http

POST https://orchestration.civitai.com/v2/consumer/workflows?wait=0
Authorization: Bearer <your-token>
Content-Type: application/json

{
  "steps": [{
    "$type": "videoGen",
    "input": {
      "engine": "minimax-h3",
      "prompt": "A golden retriever puppy running along a sunny beach, waves splashing, cinematic slow motion",
      "aspectRatio": "16:9",
      "resolution": "2K",
      "duration": 5
    }
  }]
}

POST/v2/consumer/workflows
Preview costSubmit for real
Set your Civitai API token via the Token button in the navbar to enable Try It.
                                                                    [{                   ]
                                                                    [  "steps": [        ]
                                                                    [    {               ]
                                                                    [      "$type": "vide]
                                                                    [      "input": {    ]
                                                                    [        "engine": "m]
                                                                    [        "prompt": "A]
                                                                    [        "aspectRatio]
                                                                    [        "resolution"]
                                                                    [        "duration": ]
                                                                    [      }             ]
Request body — edit to customize (e.g. swap the image URL or prompt)[    }               ]
Valid JSONRevert to example

Image-to-video ​

Pass firstFrameImage to animate forward from a still. aspectRatio is ignored — the output geometry follows the image.

json

{
  "engine": "minimax-h3",
  "prompt": "Pull focus to the people in the background and add steam rising from the bowl",
  "firstFrameImage": "https://image.civitai.com/.../first-frame.jpeg",
  "resolution": "2K",
  "duration": 5
}

POST/v2/consumer/workflows
Preview costSubmit for real
Set your Civitai API token via the Token button in the navbar to enable Try It.
                                                                    [{                   ]
                                                                    [  "steps": [        ]
                                                                    [    {               ]
                                                                    [      "$type": "vide]
                                                                    [      "input": {    ]
                                                                    [        "engine": "m]
                                                                    [        "prompt": "P]
                                                                    [        "firstFrameI]
                                                                    [        "resolution"]
                                                                    [        "duration": ]
                                                                    [      }             ]
Request body — edit to customize (e.g. swap the image URL or prompt)[    }               ]
Valid JSONRevert to example

Last frame ​

Pass lastFrameImage instead to generate the run-up that ends on the supplied still.

json

{
  "engine": "minimax-h3",
  "prompt": "The camera slowly pushes in until it settles on this composition",
  "lastFrameImage": "https://image.civitai.com/.../final-frame.jpeg",
  "resolution": "2K",
  "duration": 5
}

POST/v2/consumer/workflows
Preview costSubmit for real
Set your Civitai API token via the Token button in the navbar to enable Try It.
                                                                    [{                   ]
                                                                    [  "steps": [        ]
                                                                    [    {               ]
                                                                    [      "$type": "vide]
                                                                    [      "input": {    ]
                                                                    [        "engine": "m]
                                                                    [        "prompt": "T]
                                                                    [        "lastFrameIm]
                                                                    [        "resolution"]
                                                                    [        "duration": ]
                                                                    [      }             ]
Request body — edit to customize (e.g. swap the image URL or prompt)[    }               ]
Valid JSONRevert to example

First and last frame ​

Supply both and H3 interpolates between them.

json

{
  "engine": "minimax-h3",
  "prompt": "A smooth cinematic transition between the two compositions",
  "firstFrameImage": "https://image.civitai.com/.../start.jpeg",
  "lastFrameImage": "https://image.civitai.com/.../end.jpeg",
  "resolution": "2K",
  "duration": 5
}

POST/v2/consumer/workflows
Preview costSubmit for real
Set your Civitai API token via the Token button in the navbar to enable Try It.
                                                                    [{                   ]
                                                                    [  "steps": [        ]
                                                                    [    {               ]
                                                                    [      "$type": "vide]
                                                                    [      "input": {    ]
                                                                    [        "engine": "m]
                                                                    [        "prompt": "A]
                                                                    [        "firstFrameI]
                                                                    [        "lastFrameIm]
                                                                    [        "resolution"]
                                                                    [        "duration": ]
Request body — edit to customize (e.g. swap the image URL or prompt)[      }             ]
Valid JSONRevert to example

Reference-to-video ​

Guide generation with up to nine reference images, three reference video clips, and three reference audio clips. Reference media cannot be combined with firstFrameImage/lastFrameImage — MiniMax
treats those as different task types, and mixing them is rejected with a 400.

json

{
  "engine": "minimax-h3",
  "prompt": "The character from the references walks through a rain-soaked night market",
  "referenceImages": [
    "https://image.civitai.com/.../subject-a.jpeg",
    "https://image.civitai.com/.../subject-b.jpeg"
  ],
  "resolution": "2K",
  "duration": 5
}

Reference audio is how you drive a spoken performance — describe the line in the prompt and the voice timbre follows the supplied clip.

POST/v2/consumer/workflows
Preview costSubmit for real
Set your Civitai API token via the Token button in the navbar to enable Try It.
                                                                    [{                   ]
                                                                    [  "steps": [        ]
                                                                    [    {               ]
                                                                    [      "$type": "vide]
                                                                    [      "input": {    ]
                                                                    [        "engine": "m]
                                                                    [        "prompt": "T]
                                                                    [        "referenceIm]
                                                                    [          "https://i]
                                                                    [          "https://i]
                                                                    [        ],          ]
Request body — edit to customize (e.g. swap the image URL or prompt)[        "resolution"]
Valid JSONRevert to example

Reference media must be publicly reachable

MiniMax fetches referenceVideos and referenceAudios server-side, from its own infrastructure. Signed URLs that expire, private buckets, and internal hostnames all fail with an opaque 400. Host the
media somewhere publicly fetchable for the lifetime of the job. Images passed via firstFrameImage, lastFrameImage, and referenceImages go through the Civitai image pipeline first and don't have this
restriction.

Parameters ​

The schema in the API reference is authoritative.

      Field        Default                                                           Notes
engine            — ✅        "minimax-h3"
prompt            — ✅        Up to 7000 characters.
resolution        "2K"       "2K" is the only value H3 accepts today. H3's 2K is natively generated rather than upscaled.
duration          5          Integer seconds, 5–15.
aspectRatio       "adaptive" "adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16". Only read for text-to-video, where adaptive is not allowed.
firstFrameImage   —          Single image used as the opening frame.
lastFrameImage    —          Single image used as the closing frame.
referenceImages[] []         Up to 9. First 5 free, then billed. Cannot be combined with first/last frame.
referenceVideos[] []         Up to 3, 2–15 s each and ≤15 s in total, ≤50 MB, MP4/MOV.
referenceAudios[] []         Up to 3, 2–15 s each, ≤15 MB, WAV/MP3. Must accompany a reference image or video.
watermark         false      Adds MiniMax's AIGC watermark to the output.

Image limits: 256–5760 px per side, aspect ratio between 0.4 and 2.5, ≤30 MB, JPG/PNG/WEBP/HEIC/HEIF.

Cost ​

Billed per second. Preview the exact charge with whatif=true (Payments (Buzz)).

total = 170 × (duration + referenceVideoSeconds) + 52 × max(0, referenceImages − 5)

       duration         5 s 10 s  15 s
2K, no reference media  850 1 700 2 550

Three rules govern reference input:

Reference input                              Billing
Audio           Free
Image           First 5 free, then 52 Buzz each
Video           Billed at the output rate — 170 Buzz per second of uploaded clip

Reference video is charged on the length of the clip you upload, not the output. A 10-second generation over a 5-second reference clip bills 15 seconds, so whatif=true measures your clips before
quoting. Some worked examples:

          Scenario                    Total
10 s, no reference            1 700
10 s + 3 reference images     1 700
10 s + 8 reference images     1 700 + 3 × 52 = 1 856
10 s + a 5 s reference video  15 × 170 = 2 550

Failed generations and clips rejected by MiniMax's content review are not charged.

Reading the result ​

Standard videoGen output: a single video blob (shape and URL expiry).

The output is H.264 at 2560×1440 in an MP4 container, with an AAC audio track generated alongside the video. A 5-second clip is roughly 6 MB.

Runtime ​

A 5-second 2K clip takes around three minutes end to end, and 15-second clips considerably longer — longer than your HTTP client's timeout. Submit with wait=0 and collect the result via a webhook or
by polling — see Results and webhooks.

Troubleshooting ​

                    Symptom                                                      Cause                                                                     Fix
400 mentioning ratio                             adaptive on a text-to-video request                                    Pass a concrete aspectRatio, or supply a firstFrameImage.
400 mentioning resolution                        Anything other than 2K                                                 H3 supports only 2K.
400 on a reference-media request                 URL not publicly fetchable, or a clip outside the size/duration limits Check the limits above and that the URL resolves from the public internet.
400 combining reference media with a frame image Different MiniMax task types                                           Use one or the other.
Content-policy rejection                         MiniMax's own input review                                             Rephrase the prompt or swap the source media. Not billed.
Job retries across workers                       Provider rate limit                                                    Expected under load.

Related ​
