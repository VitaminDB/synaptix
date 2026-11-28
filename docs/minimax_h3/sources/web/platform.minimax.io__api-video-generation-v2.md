# Source: https://platform.minimax.io/docs/api-reference/video-generation-v2-create

Fetched: 2026-11-28 (curl + w3m text dump)

---

Text-to-video (t2va)

(Copy the contents from the code block)

curl --request POST \
  --url https://api.minimax.io/v2/video_generation \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: <content-type>' \
  --data '
{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as the last fleet gathers and jumps away in a blinding flash, the bridge shaking, leaving her behind."
    }
  ],
  "resolution": "2K",
  "duration": 5,
  "ratio": "16:9"
}
'

200
400
401
402
422
429
500
(Copy the contents from the code block)

{
  "task_id": "424010985738629"
}

{
  "type": "error",
  "error": {
    "type": "bad_request_error",
    "message": "invalid params, content must include a non-empty text item (prompt is required) (2013)",
    "http_code": "400"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

{
  "type": "error",
  "error": {
    "type": "authorized_error",
    "message": "login fail: Please carry the API secret key in the 'Authorization' field of the request header (1004)",
    "http_code": "401"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

{
  "type": "error",
  "error": {
    "type": "insufficient_balance_error",
    "message": "insufficient balance (1008)",
    "http_code": "402"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

{
  "type": "error",
  "error": {
    "type": "unprocessable_entity_error",
    "message": "video description contains sensitive content (1026)",
    "http_code": "422"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "rate limit, please retry later (1002)",
    "http_code": "429"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

{
  "type": "error",
  "error": {
    "type": "server_error",
    "message": "internal error (1000)",
    "http_code": "500"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

MiniMax-H3

Create Video Generation Task

(Copy page)
Copy pageCopy page
(More actions)

Change the model field to switch between the flagship models (MiniMax H3, MiniMax H3 Highspeed, and MiniMax H3 Lite) and other extended models. This video generation V2 endpoint accepts multimodal
input via the content array (text / image / video / audio) to support text-to-video, image-to-video (first & last frame), and reference-to-video, with output up to 2K.

Note: To use the flagship models or other extended models, please select the Pay-as-you-go API.

(Copy page)
Copy pageCopy page
(More actions)
POST
/
v2
/
video_generation
(Copy)
(Try it)Try it

cURL

Text-to-video (t2va)

(Copy the contents from the code block)

curl --request POST \
  --url https://api.minimax.io/v2/video_generation \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: <content-type>' \
  --data '
{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as the last fleet gathers and jumps away in a blinding flash, the bridge shaking, leaving her behind."
    }
  ],
  "resolution": "2K",
  "duration": 5,
  "ratio": "16:9"
}
'

200
400
401
402
422
429
500
(Copy the contents from the code block)

{
  "task_id": "424010985738629"
}

{
  "type": "error",
  "error": {
    "type": "bad_request_error",
    "message": "invalid params, content must include a non-empty text item (prompt is required) (2013)",
    "http_code": "400"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

{
  "type": "error",
  "error": {
    "type": "authorized_error",
    "message": "login fail: Please carry the API secret key in the 'Authorization' field of the request header (1004)",
    "http_code": "401"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

{
  "type": "error",
  "error": {
    "type": "insufficient_balance_error",
    "message": "insufficient balance (1008)",
    "http_code": "402"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

{
  "type": "error",
  "error": {
    "type": "unprocessable_entity_error",
    "message": "video description contains sensitive content (1026)",
    "http_code": "422"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "rate limit, please retry later (1002)",
    "http_code": "429"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

{
  "type": "error",
  "error": {
    "type": "server_error",
    "message": "internal error (1000)",
    "http_code": "500"
  },
  "request_id": "021785229015510a2c883cf675b9804d"
}

Authorizations

 
Authorization
string
header
required

HTTP: Bearer Auth

  • Security Scheme Type: http
  • HTTP Authorization Scheme: Bearer API_key, used to verify account information, can be found in Account Management>API Keys.

Headers

 
Content-Type
enum<string>
default:application/json
required

Media type of the request body. Set it to application/json.

Available options:
application/json

Body

application/json
 
model
enum<string>
required

Model name. Currently available: MiniMax-H3, MiniMax-H3-Highspeed, MiniMax-H3-Lite, MiniMax-H3-Max, MiniMax-H3-Max-Turbo, MiniMax-H3-MG, MiniMax-H3-NC.

  • MiniMax-H3: the variant for high-quality video creation. Supports text-to-video, image-to-video (first / last frame), and reference-to-video; 480P / 768P / 2K resolution, 4–15s duration. 480P
    output is served by RunningHub and billed at the MiniMax-H3 480P price under H3 Series Extended Models; no separate model name is needed.
  • MiniMax-H3-Highspeed: balances generation quality and speed. Supports text-to-video, image-to-video (first / last frame), and reference-to-video (reference image / video / audio); 480P / 768P
    resolution, 2K is not supported; 4–15s duration.
  • MiniMax-H3-Lite: the lightweight variant. Supports text-to-video, image-to-video (first / last frame), and reference-to-video (reference image / video / audio); 480P / 768P resolution, 2K is not
    supported; 4–15s duration.
  • MiniMax-H3-Max: the fast generation variant. Supports text-to-video, image-to-video (first / last frame), and reference-to-video (reference image / video / audio); 480P / 768P resolution, 2K is
    not supported; 5–15s duration.
  • MiniMax-H3-Max-Turbo: a faster, lower-cost version of MiniMax-H3-Max. Supports text-to-video, image-to-video (first / last frame), and reference-to-video (reference image / video / audio); 480P /
    768P resolution, 2K is not supported; 5–15s duration.
  • MiniMax-H3-MG: MiniMax H3 hosted by MachGen, with output up to 2K. Supports text-to-video, image-to-video (first / last frame), and reference-to-video (reference image / video / audio); 768P / 2K
    resolution, 480P is not supported; 4–15s duration.
  • MiniMax-H3-NC: MiniMax H3 Fast by Nunchux AI, for fast previews and rapid iteration. Supports text-to-video, image-to-video (first / last frame), and reference-to-video (reference image / video /
    audio); 480P / 768P resolution, 2K is not supported; 4–15s duration.

Available options:
MiniMax-H3,
MiniMax-H3-Highspeed,
MiniMax-H3-Lite,
MiniMax-H3-Max,
MiniMax-H3-Max-Turbo,
MiniMax-H3-MG,
MiniMax-H3-NC
 
content
object[]
required

Array of multimodal input describing the information used to generate the video. Each element is distinguished by type (text / image_url / video_url / audio_url) and can be labeled with a role.

Every request must include one non-empty text item (the prompt is required); otherwise a parameter error is returned.

Supported input combinations (corresponding to different generation scenarios):

  • Text-to-video: a single text element only.
  • Image-to-video, first frame: text + 1 image_url (role=first_frame, or omitted).
  • Image-to-video, last frame: text + 1 image_url (role=last_frame).
  • Image-to-video, first & last frame: text + 2 image_url items with role set to first_frame and last_frame respectively.
  • Reference-to-video: text + any combination of reference images (role=reference_image), reference videos (role=reference_video), and reference audio (role=reference_audio).

    Image-to-video and reference-to-video are mutually exclusive: if any reference_image / reference_video / reference_audio role appears in content, then first_frame / last_frame must not appear
    (and vice versa); the two cannot be mixed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input media limits (total request body ≤ 64 MB; use public URLs for large files, avoid Base64)

Image image_url:

       Item                                Limit
Format             JPG, JPEG, PNG, WEBP, HEIC, HEIF
Single file size   ≤ 30 MB
Width/height range [256, 5760] px
Aspect ratio (w/h) [0.4, 2.5]
Count              first frame ≤ 1, last frame ≤ 1, reference images ≤ 9

Video video_url (reference scenario only):

       Item                           Limit
Container / format MP4 (.mp4), MOV (.mov)
Codec              Video H.264/AVC, H.265/HEVC; audio AAC, MP3
Single file size   ≤ 50 MB
Count              ≤ 3
Per-clip duration  [2, 15] s; total ≤ 15 s
Width/height range [256, 5760] px
Aspect ratio (w/h) [0.4, 2.5]
Frame rate         [23.976, 60]

Audio audio_url (reference scenario only):

       Item                 Limit
Format             WAV, MP3
Single file size   ≤ 15 MB
Count              ≤ 3
Per-clip duration  [2, 15] s; total ≤ 15 s

Show child attributes

 
resolution
enum<string>
required

Video resolution. Availability differs by model:

  • MiniMax-H3: 480P, 768P, 2K (480P is served by RunningHub; keep model as MiniMax-H3 and set resolution to 480P)
  • MiniMax-H3-Highspeed: 480P, 768P (2K is not supported)
  • MiniMax-H3-Lite: 480P, 768P (2K is not supported)
  • MiniMax-H3-Max: 480P, 768P (2K is not supported)
  • MiniMax-H3-Max-Turbo: 480P, 768P (2K is not supported)
  • MiniMax-H3-MG: 768P, 2K (480P is not supported)
  • MiniMax-H3-NC: 480P, 768P (2K is not supported)

Available options:
480P,
768P,
2K
 
duration
enum<integer>
required

Duration of the generated video in seconds. Required, integer. Available values differ by model:

  • MiniMax-H3: 4–15
  • MiniMax-H3-Highspeed: 4–15
  • MiniMax-H3-Lite: 4–15
  • MiniMax-H3-Max: 5–15 (4 seconds is not supported)
  • MiniMax-H3-Max-Turbo: 5–15 (4 seconds is not supported)
  • MiniMax-H3-MG: 4–15
  • MiniMax-H3-NC: 4–15

Available options:
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15
 
ratio
enum<string>

Aspect ratio of the generated video. Defaults to adaptive (the most suitable ratio is chosen automatically based on the input; the actual ratio can be read from the ratio field of the query
endpoint). Available values: adaptive, 21:9, 16:9, 4:3, 1:1, 3:4, 9:16.

Text-to-video (t2va, content contains only text): ratio is required and cannot be adaptive; available values 21:9, 16:9, 4:3, 1:1, 3:4, 9:16.

Image-to-video (i2va, content contains a first_frame / last_frame image): the aspect ratio is determined by the input image and ratio is always adaptive; passing another valid value does not error
but is ignored and treated as adaptive.

Reference-to-video (r2va, content contains reference_image / reference_video / reference_audio): ratio is optional and defaults to adaptive; you may also explicitly specify any of the concrete ratios
above. MiniMax-H3-Lite and MiniMax-H3-NC do not support adaptive yet; specify a concrete ratio.

Available options:
adaptive,
21:9,
16:9,
4:3,
1:1,
3:4,
9:16
 
extra
object

Additional generation options for MiniMax-H3-Max and MiniMax-H3-Max-Turbo. Currently only prompt_expansion_mode is supported. If extra or this property is omitted, balanced is used. Do not pass 
balance, an empty string, a Boolean value, or other undeclared fields.

Show child attributes

 
generate_audio
boolean
default:true

Controls whether the generated video includes audio. Optional; defaults to true.

  • true: output a video with audio.
  • false: output a silent video.

 
callback_url
string

Callback URL for task status changes. Once configured, the MiniMax server first sends a verification request containing a challenge field (you must return the challenge unchanged within 3 seconds to
complete verification); after verification succeeds, it POSTs an update to this URL whenever the task status changes. The push body has the same structure as the response of the Query Task endpoint.

Callback status values: queued, running, succeeded, failed, cancelled.

 
response_format
enum<string>
default:task

Response format.

  • task (default): returns {"task": {...}}, with task fields nested under the task object.
  • flat: task fields are flattened at the top level of the response (including usage), matching the shape of items[] in the task list response.

Available options:
task,
flat

Response

200
application/json

The create endpoint returns a task_id. Use this task_id with the Query Task endpoint to retrieve the task status and result.

Successful Query Task response example

{
  "task": {
    "id": "424010985738629",
    "model": "MiniMax-H3",
    "status": "succeeded",
    "created_at": 1785125529,
    "updated_at": 1785125946,
    "content": {
      "url": "https://your-cdn.example.com/h3-generated-2k-output.mp4"
    },
    "resolution": "2K",
    "duration": 5,
    "usage": {
      "total_seconds": 5,
      "input_seconds": 0,
      "output_seconds": 5,
      "input_image_count": 0
    },
    "ratio": "16:9",
    "task_type": "generation",
    "modality": "video"
  }
}

 
task_id
string

ID of the task, used to query the task status and result later.

Estimate Input Tokens
Query Task
[                    ]               
[                    ] (Send message)
MiniMax API Docs home pagelight logodark logo
discordxlinkedingithub

Model

MiniMax M3.1 FlashMiniMax M3MiniMax H3MiniMax Speech 2.8MiniMax Music 3.0

Product

MiniMax CodeMiniMax DesignMiniMax Audio

API Platform

Developer DocsM Plan SubscriptionPricingConsole LoginStatus

Company

About UsNewsInvestor RelationsCareers
{"docsConfig":{"theme":"mint","$schema":"https://mintlify.com/docs.json","name":"MiniMax API Docs","colors":{"primary":"#181E25","dark":"#181E25"},"logo":{"light":"https://mintcdn.com/
minimax-cac98058/XYzsL2L2ynonu2Q_/logo/light.svg?fit=max\u0026auto=format\u0026n=XYzsL2L2ynonu2Q_\u0026q=85\u0026s=9c99bb2678d62388d8a4e7cce1750858","dark":"https://mintcdn.com/minimax-cac98058/
XYzsL2L2ynonu2Q_/logo/dark.svg?fit=max\u0026auto=format\u0026n=XYzsL2L2ynonu2Q_\u0026q=85\u0026s=e62ef6e36d7512de91c8c48322614594","href":"https://minimax.io"},"favicon":"https://mintcdn.com/
minimax-cac98058/3KE0RwwD41Lh3xt2/favicon.ico?fit=max\u0026auto=format\u0026n=3KE0RwwD41Lh3xt2\u0026q=85\u0026s=6a58247013972686c0a551c6c6b98d66","api":{"examples":{"prefill":true}},"appearance":
{"default":"system","strict":false},"navbar":{"primary":{"type":"button","label":"Console","href":"https://platform.minimax.io/console/home"}},"footer":{"socials":{"discord":"https://www.minimax.io/
discord","x":"https://x.com/MiniMax_AI","linkedin":"https://www.linkedin.com/company/minimax-ai","github":"https://github.com/MiniMax-AI"},"links":[{"header":"Model","items":[{"label":"MiniMax M3.1
Flash","href":"https://www.minimax.io/blog/minimax-m31f"},{"label":"MiniMax M3","href":"https://www.minimax.io/blog/minimax-m3"},{"label":"MiniMax H3","href":"https://www.minimax.io/blog/
minimax-h3"},{"label":"MiniMax Speech 2.8","href":"https://www.minimax.io/news/minimax-speech-28"},{"label":"MiniMax Music 3.0","href":"https://www.minimax.io/blog/
minimax-music-3-0-next-generation-open-weights-production-ready-versatile-music-model"}]},{"header":"Product","items":[{"label":"MiniMax Code","href":"https://agent.minimax.io/"},{"label":"MiniMax
Design","href":"https://design.minimax.io/"},{"label":"MiniMax Audio","href":"https://www.minimax.io/audio"}]},{"header":"API Platform","items":[{"label":"Developer Docs","href":"https://
platform.minimax.io/docs/guides/models-intro"},{"label":"M Plan Subscription","href":"https://www.minimax.io/m-plan"},{"label":"Pricing","href":"https://platform.minimax.io/docs/pricing/overview"},
{"label":"Console Login","href":"https://platform.minimax.io/console/home"},{"label":"Status","href":"https://status.minimax.io/"}]},{"header":"Company","items":[{"label":"About Us","href":"https://
www.minimax.io/about"},{"label":"News","href":"https://www.minimax.io/news"},{"label":"Investor Relations","href":"https://ir.minimax.io/"},{"label":"Careers","href":"https://www.minimax.io/
careers"}]}]},"seo":{"metatags":{"title":"MiniMax API Docs","description":"MiniMax Open Platform developer documentation with model API references, integration guides, tool setup, and billing
information for building AI applications.","charset":"UTF-8","canonical":"https://platform.minimax.io/docs","viewport":"width=device-width, initial-scale=1.0"},"indexing":"all"},"fonts":{"family":"DM
Sans"},"icons":{"library":"lucide"},"integrations":{"telemetry":{"enabled":true},"cookies":{"key":"minimax_docs_feedback_consent","value":"v1:granted"}},"banner":{"content":"The next-generation image
model MiniMax H3 Image is now available! [View docs →](/docs/guides/image-generation)","dismissible":true},"contextual":{"options":
["copy","view","mcp","cursor","vscode"],"display":"header"},"styling":{"latex":true},"navigation":{"pages":[]}},"deployment":{"deploymentMetadataValue":
{"subdomain":"minimax-cac98058","hasAudioNarration":false,"actualSubdomain":"minimax-cac98058","customDomains":["platform.minimax.io"],"gitSource":
{"type":"github","owner":"minimax-ai-dev","repo":"docs","deployBranch":"main","contentDirectory":"packages/docs-en","isPrivate":true,"mountPath":"","label":""},"feedback":
{"issues":false,"thumbs":true,"edits":false,"agentFeedback":true},"entitlements":{"AI_CHAT":{"status":"ENABLED"},"REMOVE_BRANDING":{"status":"ENABLED"},"CONTEXTUAL_FEEDBACK":
{"status":"ENABLED"},"CODE_SNIPPET_FEEDBACK":{"status":"ENABLED"},"WEB_MCP":{"status":"ENABLED"},"NARRATION":
{"status":"ENABLED"}},"buildId":"6b0659c8a63409eb080e1c3f","clientVersion":"0.0.4342","searchFilterCounts":[{"language":"default","tag":"API","count":49},{"language":"default","tag":"Developer
Guides","count":33},{"language":"default","tag":"Use Guides","count":23},{"language":"default","tag":"M Plan","count":18},{"language":"default","tag":"Speech","count":17},
{"language":"default","tag":"Text","count":14},{"language":"default","tag":"Guides","count":12},{"language":"default","tag":"Tools","count":9},{"language":"default","tag":"Legacy APIs","count":8},
{"language":"default","tag":"Video","count":8}]},"assistantConfig":{"enableAskAiButton":false,"enableAskAiSearchBar":false},"basePath":"/
docs","hasChatPermissions":false,"hasWebMcpPermissions":true,"toggles":[{"name":"kill-isr","enabled":true,"variant":{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},
{"name":"custom-rbac-roles","enabled":true,"variant":{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"agent-mintlify-slack","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"agent-integrations-settings","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"rbac-expansion","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"automations-pr-grouping","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"custom-prefix","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"analytics-v7","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"user-flows-analytics","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"no-publish-flow","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"snapshot-serving-live","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"stripTemperatureMiddleware","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"support-platform-pylon-zendesk","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"snapshot-serving","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"instant-rollback","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"granular-notifications-server","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"granular-notifications","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"automations-fable","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"fixed-ai-action-pricing","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"site-roles","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"automation-resolve-merge-conflict","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"scoped-publishing","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"mintql","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"page-level-analytics","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"editor-version-history-v2","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"billing-page-v2","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"full-page-search","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"workspace-projects","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"fork-pr-previews","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"revalidation-jwt","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"hobby-abuse-scan","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"dashboard-inbox","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"turbopuffer-dual-write","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"approve-review-flow","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"editor-metadata","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"turbopuffer-serve","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"live-agents","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"starter-pricing","enabled":true,"variant":{"name":"control","enabled":true,"payload":
{"type":"string","value":"control"},"feature_enabled":true},"impressionData":false},{"name":"llms-txt-v2","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false},{"name":"wildcard-email-auth","enabled":true,"variant":
{"name":"disabled","enabled":false,"feature_enabled":true},"impressionData":false}],"starterQuestions":[],"googleFontsInlined":true},"customCss":[{"filePath":"style.css","content":".audio-container
{height:50px;margin:16px 0}.video-agent-templates{width:300px;height:180px;object-fit:cover}video{border-radius:12px}table{overflow-x:auto}th,td{vertical-align:middle!important}td
.video-agent-templates{margin:0}#navbar .docs-platform-brand{display:inline-flex;align-items:center;flex-shrink:0}#navbar .docs-platform-name{margin-left:20px;padding-left:20px;border-left:1px solid
#e5e7eb;color:#6b7280;font-size:14px;font-weight:400;line-height:20px;white-space:nowrap}.dark #navbar .docs-platform-name{border-color:#374151;color:#9ca3af}#topbar-cta-button>a>span,#navbar
.docs-console-mobile>span{background-color:#242424;border-radius:8px}#topbar-cta-button>a:hover>span,#navbar .docs-console-mobile:hover>span{background-color:#3a3a3a}#navbar .docs-console-mobile
{display:none}#navbar nav:has(#topbar-cta-button){margin-right:0}@media (max-width: 1023px){#navbar .docs-platform-name{margin-left:12px;padding-left:12px;font-size:12px}#navbar .docs-console-mobile
{position:relative;display:inline-flex;align-items:center;flex-shrink:0;padding:8px 10px;font-size:12px;font-weight:500;white-space:nowrap}}@media (max-width: 480px){#navbar .docs-topbar-row
{column-gap:4px}#navbar .docs-platform-brand .nav-logo{width:80px;height:auto}#navbar .docs-platform-name{max-width:60px;margin-left:8px;padding-left:8px;font-size:11px;line-height:14px;
white-space:normal}#navbar .docs-console-mobile{padding:8px;font-size:11px}#navbar :has(>#search-bar-entry-mobile){gap:6px}#navbar #search-bar-entry-mobile{width:24px}}#content-container p
{font-size:1rem;line-height:1.5rem}.eyebrow{text-transform:uppercase}.method-nav-pill>span{font-weight:400}*:has(>.method-pill){border:none}.navbar-link>a{font-weight:400}.nav-tabs-item
{font-weight:400}.nav-tabs-item.text-gray-800{font-weight:500}*:has(>.navbar-link)>.navbar-link:first-child>a{color:#181e25;font-weight:500}.dark *:has(>.navbar-link)>.navbar-link:first-child>a
{color:#fff}#navbar,#sidebar{letter-spacing:0.02em}.callout{margin-top:2rem;margin-bottom:2rem}#content:has(>.model-retirement-notice:first-child){margin-top:1rem}.callout.model-retirement-notice
{margin-top:0}.mplan-tool-intro{margin:16px 0;padding:12px 16px;border-left:4px solid #d97706;border-radius:6px;background:#fffbeb}.dark .mplan-tool-intro{background:rgba(120, 53, 15, 0.22);color:#
e5e7eb}.dark .mplan-tool-intro a{color:#fbbf24!important;border-bottom-color:#fbbf24!important}.card span{line-height:1.4rem;margin-top:8px!important}#banner a{border-bottom-color:#181E25;color:#FFF}
#footer a[href=\"mailto:api@minimax.io\"] svg{-webkit-mask-image:url('https://filecdn.minimax.chat/public/45f6ad6e-f42e-45ed-938e-7de67bfd367f.svg')!important;mask-image:url('https://
filecdn.minimax.chat/public/45f6ad6e-f42e-45ed-938e-7de67bfd367f.svg')!important}#sidebar:has(a[href$=\"/pricing/overview\"]),#sidebar:has(a[href$=\"/pricing/overview/\"])
{visibility:hidden}.pricing-sidebar-navigation{display:none}@media (min-width: 1024px){#content:has([data-pricing-sidebar]){container-type:normal;contain:none;
isolation:auto}.pricing-sidebar-navigation{--pricing-sidebar-top: calc(var(--mintlify-slot-header-height, calc(4rem + var(--topbar-tabs-height, 3rem))) + var(--banner-height, 0px));display:block;
position:fixed;z-index:20;top:var(--pricing-sidebar-top);left:max(2rem, calc((100vw - 92rem) / 2 + 2rem));width:var(--sidebar-width, 18rem);height:calc(100dvh - var(--pricing-sidebar-top));
box-sizing:border-box;overflow-y:auto;overscroll-behavior:contain;padding:40px 24px 32px 0;margin:0}}[data-pricing-overview-active] [data-pricing-original]{display:none!
important}.pricing-anchor-navigation{font-size:14px;line-height:1.5}.pricing-navigation-group+.pricing-navigation-group{margin-top:28px}.pricing-navigation-title{display:block;margin:0 0 10px;
padding-left:16px;color:#111827;font-size:inherit;font-weight:600;line-height:1.5}.pricing-anchor-navigation .pricing-anchor-link{display:block;padding:6px 12px 6px 16px;margin:1px 0;border:0;
border-radius:12px;color:#374151;line-height:1.5;text-decoration:none;overflow-wrap:anywhere}.pricing-anchor-navigation .pricing-anchor-link:hover{background:rgba(75, 85, 99, .05);color:#
111827}.pricing-anchor-navigation .pricing-anchor-link[aria-current=\"location\"]{background:rgba(24, 30, 37, .075);color:#181e25;font-weight:600}.pricing-anchor-link:focus-visible{outline:2px solid
currentColor;outline-offset:-2px}.pricing-section-toc{font-size:14px;line-height:1.5}.pricing-toc-title{margin-bottom:12px;color:#374151;font-weight:500}.pricing-section-toc .pricing-anchor-link
{display:block;border:0;padding:5px 0 5px 12px;border-left:1px solid #e5e7eb;color:#6b7280;text-decoration:none;overflow-wrap:anywhere}.pricing-section-toc .pricing-anchor-link[aria-current=\
"location\"]{border-left:2px solid #181e25;color:#181e25;font-weight:500}.pricing-mobile-navigation{display:none}#pricing-document .pricing-section,#pricing-document .pricing-section [id]
{scroll-margin-top:10rem}.dark .pricing-navigation-title,.dark .pricing-toc-title{color:#e5e7eb}.dark .pricing-anchor-navigation .pricing-anchor-link,.dark .pricing-section-toc .pricing-anchor-link
{color:#9ca3af}.dark .pricing-anchor-navigation .pricing-anchor-link:hover{background:rgba(229, 231, 235, .05);color:#d1d5db}.dark .pricing-anchor-navigation .pricing-anchor-link[aria-current=\
"location\"]{background:rgba(229, 231, 235, .1);color:#f3f4f6}.dark .pricing-section-toc .pricing-anchor-link{border-left-color:#374151}.dark .pricing-section-toc .pricing-anchor-link[aria-current=\
"location\"]{border-left-color:#e5e7eb;color:#e5e7eb}@media (max-width: 1023px){#pricing-document .pricing-mobile-navigation{display:block;margin:0 0 24px}#pricing-document
.pricing-mobile-navigation>.accordion{margin-bottom:0}#pricing-document .pricing-mobile-navigation .pricing-anchor-navigation{max-height:min(65vh, 560px);overflow-y:auto;padding:8px 12px 20px;
overscroll-behavior:contain}}#pricing-document>.pricing-section{margin-top:56px;padding-top:40px;border-top:1px solid #e5e7eb}#pricing-document>.pricing-overview+.pricing-section{margin-top:36px;
border-top:0}#pricing-document>.pricing-section>h2:first-child{margin-top:0}#pricing-document .pricing-overview>span[data-as=\"p\"],#pricing-document .pricing-section>span[data-as=\"p\"]
{display:block;margin:1em 0}.dark #pricing-document>.pricing-section{border-color:#374151}@media (max-width: 767px){#pricing-document>.pricing-section{margin-top:40px;padding-top:28px}}#
pricing-document section[id=\"models/text\"] table{table-layout:fixed;width:100%}#pricing-document section[id=\"models/text\"] table th:first-child,#pricing-document section[id=\"models/text\"] table
td:first-child{width:32%}#pricing-document section[id=\"models/video\"] table{table-layout:fixed;width:100%}#pricing-document section[id=\"models/video\"] table th:first-child,#pricing-document
section[id=\"models/video\"] table td:first-child{width:30%}#pricing-document .video-price-table,#pricing-document .video-task-table{display:table;table-layout:fixed;width:100%}#pricing-document
.video-price-table td:has(>strong)+td,#pricing-document .video-price-table tbody tr:not(:has(strong))>td:first-child{padding-left:24px}#pricing-document .video-price-table thead th:nth-child(2)
{padding-left:24px}.restricted-doc{--restricted-border: rgba(15, 23, 42, .12);--restricted-border-strong: rgba(15, 23, 42, .18);--restricted-muted: #64748b;--restricted-surface: rgba(15, 23, 42,
.035);--restricted-surface-hover: rgba(15, 23, 42, .055);--restricted-link: #2563eb;color:inherit}.dark .restricted-doc{--restricted-border: rgba(255, 255, 255, .12);--restricted-border-strong: rgba
(255, 255, 255, .18);--restricted-muted: #94a3b8;--restricted-surface: rgba(255, 255, 255, .045);--restricted-surface-hover: rgba(255, 255, 255, .07);--restricted-link: #
60a5fa}.restricted-doc-loading{min-height:12rem;display:flex;align-items:center;justify-content:center}.restricted-doc-spinner{width:1.75rem;height:1.75rem;border:2px solid rgba(98,112,132,.24);
border-top-color:#627084;border-radius:50%;animation:restricted-doc-spin .75s linear infinite}@keyframes restricted-doc-spin{to{transform:rotate(360deg)}}@media (prefers-reduced-motion: reduce)
{.restricted-doc-spinner{animation-duration:1.5s}}.restricted-doc>h1{margin:0 0 .75rem;font-size:2.25rem;line-height:1.2;font-weight:700;letter-spacing:-.025em}.restricted-doc-description{margin:0 0
2rem;color:var(--restricted-muted);font-size:1.05rem;line-height:1.7}.restricted-doc h2,.restricted-doc h3,.restricted-doc h4{scroll-margin-top:6rem;font-weight:650;
letter-spacing:-.015em}.restricted-doc h2{margin:2.75rem 0 1rem;padding-top:.25rem;font-size:1.65rem;line-height:1.35}.restricted-doc h3{margin:2.1rem 0 .75rem;font-size:1.3rem;
line-height:1.4}.restricted-doc h4{margin:1.6rem 0 .5rem;font-size:1.1rem;line-height:1.45}.restricted-doc>p{margin:1rem 0}.restricted-doc p,.restricted-doc li{line-height:1.75}.restricted-doc
ul,.restricted-doc ol{margin:1rem 0;padding-left:1.5rem}.restricted-doc li{padding-left:.15rem}.restricted-doc li+li{margin-top:.3rem}.restricted-doc li>p{margin:.25rem 0}.restricted-doc a{color:var
(--restricted-link);text-decoration:underline;text-decoration-color:color-mix(in srgb, var(--restricted-link) 35%, transparent);text-underline-offset:.2em}.restricted-doc a:hover
{text-decoration-color:currentColor}.restricted-doc :not(pre)>code{padding:.15rem .4rem;border:1px solid var(--restricted-border);border-radius:.35rem;background:var(--restricted-surface);
font-size:.875em;font-weight:500}.restricted-doc-table-wrap{max-width:100%;margin:1.5rem 0;overflow-x:auto;border:1px solid var(--restricted-border-strong);border-radius:.75rem}.restricted-doc table
{width:100%;border-collapse:separate;border-spacing:0;font-size:.925rem;line-height:1.55}.restricted-doc th,.restricted-doc td{min-width:7rem;padding:.75rem 1rem;border-bottom:1px solid var
(--restricted-border);vertical-align:top;text-align:left}.restricted-doc th+th,.restricted-doc td+td{border-left:1px solid var(--restricted-border)}.restricted-doc th{background:var
(--restricted-surface);font-weight:600;color:inherit}.restricted-doc tbody tr{transition:background-color .15s ease}.restricted-doc tbody tr:hover{background:var
(--restricted-surface-hover)}.restricted-doc tr:last-child td{border-bottom:0}.restricted-doc-code{position:relative;margin:1.5rem 0;border:1px solid rgba(255,255,255,.09);border-radius:.75rem;
background:#0f172a;color:#e5e7eb;box-shadow:0 1px 2px rgba(15,23,42,.08);overflow:hidden}.restricted-doc-code pre{margin:0;padding:1.1rem 1.2rem;overflow-x:auto;
background:transparent}.restricted-doc-code code{display:block;min-width:max-content;padding-right:3.5rem;font-size:.85rem;line-height:1.7;white-space:pre;tab-size:2}.restricted-doc-code button
{position:absolute;z-index:1;top:.65rem;right:.65rem;padding:.3rem .6rem;border:1px solid rgba(255,255,255,.16);border-radius:.4rem;background:rgba(30,41,59,.92);color:#cbd5e1;font-size:.75rem;
line-height:1.25rem;cursor:pointer;transition:color .15s ease, border-color .15s ease, background-color .15s ease}.restricted-doc-code button:hover{border-color:rgba(255,255,255,.28);background:#
334155;color:#fff}.restricted-doc-code .hljs-comment,.restricted-doc-code .hljs-quote{color:#7f8ea3;font-style:italic}.restricted-doc-code .hljs-keyword,.restricted-doc-code
.hljs-selector-tag,.restricted-doc-code .hljs-literal,.restricted-doc-code .hljs-section,.restricted-doc-code .hljs-link{color:#c792ea}.restricted-doc-code .hljs-string,.restricted-doc-code
.hljs-title,.restricted-doc-code .hljs-name,.restricted-doc-code .hljs-type,.restricted-doc-code .hljs-attribute,.restricted-doc-code .hljs-symbol,.restricted-doc-code
.hljs-bullet,.restricted-doc-code .hljs-addition,.restricted-doc-code .hljs-variable,.restricted-doc-code .hljs-template-tag,.restricted-doc-code .hljs-template-variable{color:#
addb67}.restricted-doc-code .hljs-number,.restricted-doc-code .hljs-meta,.restricted-doc-code .hljs-built_in,.restricted-doc-code .hljs-builtin-name,.restricted-doc-code .hljs-params{color:#
f78c6c}.restricted-doc-code .hljs-attr,.restricted-doc-code .hljs-selector-id,.restricted-doc-code .hljs-selector-class,.restricted-doc-code .hljs-selector-attr,.restricted-doc-code
.hljs-selector-pseudo{color:#ffcb6b}.restricted-doc-code .hljs-function,.restricted-doc-code .hljs-class .hljs-title{color:#82aaff}.restricted-doc-code .hljs-regexp,.restricted-doc-code
.hljs-deletion{color:#ff5370}.restricted-doc-code .hljs-emphasis{font-style:italic}.restricted-doc-code .hljs-strong{font-weight:700}.restricted-doc-callout{margin:1.5rem 0;padding:.9rem 1rem;
border:1px solid rgba(59,130,246,.22);border-radius:.65rem;background:rgba(59,130,246,.07)}.restricted-doc-callout>:first-child{margin-top:0}.restricted-doc-callout>:last-child
{margin-bottom:0}.restricted-doc blockquote{margin:1.5rem 0;padding:.1rem 0 .1rem 1.1rem;border-left:2px solid var(--restricted-border-strong);color:var(--restricted-muted);
font-style:italic}.restricted-doc blockquote>:first-child{margin-top:0}.restricted-doc blockquote>:last-child{margin-bottom:0}.restricted-doc-toc ul{margin:.6rem 0 0;padding:0;
list-style:none}.restricted-doc-toc li{margin:.35rem 0;line-height:1.35}.restricted-doc-toc li[data-level=\"3\"]{padding-left:.75rem}.restricted-doc-toc li[data-level=\"4\"]
{padding-left:1.5rem}.restricted-doc-toc a{color:var(--restricted-muted);text-decoration:none}.restricted-doc-toc a:hover{color:var(--restricted-link)}.restricted-doc-toc-title
{font-weight:600}.restricted-doc-not-found{min-height:20rem;padding:4rem 0;text-align:center}.restricted-doc-not-found h1{font-size:3rem}@media (max-width: 768px){.restricted-doc>h1
{font-size:1.85rem}.restricted-doc th,.restricted-doc td{min-width:6.5rem;padding:.65rem .75rem}.restricted-doc-code pre{padding:1rem}}"}]} {"page":{"pagePath":"api-reference/
video-generation-v2-create","slug":"/api-reference/video-generation-v2-create","pageMetadata":{"title":"Create Video Generation Task","sidebarTitle":"Create Video Generation
Task","description":"Change the `model` field to switch between the flagship models (MiniMax H3, MiniMax H3 Highspeed, and MiniMax H3 Lite) and other extended models. This video generation V2
endpoint accepts multimodal input via the content array (text / image / video / audio) to support text-to-video, image-to-video (first \u0026 last frame), and reference-to-video, with output up to
2K.\u003cbr />\u003cbr />Note: To use the flagship models or other extended models, please select the [Pay-as-you-go API](https://platform.minimax.io/docs/pricing/overview#models/
video).","openapi":"api-reference/video/generation/api/v2-video-generation.json POST /v2/video_generation","href":"/api-reference/video-generation-v2-create"},"description":{"compiledSource":"\"use
strict\";\nconst {jsx: _jsx, jsxs: _jsxs} = arguments[0];\nconst {useMDXComponents: _provideComponents} = arguments[0];\nfunction _createMdxContent(props) {\n const _components = {\n a: \"a\",\n
code: \"code\",\n p: \"p\",\n ..._provideComponents(),\n ...props.components\n };\n return _jsxs(_components.p, {\n children: [\"Change the \", _jsx(_components.code, {\n children: \"model\"\n }), \"
field to switch between the flagship models (MiniMax H3, MiniMax H3 Highspeed, and MiniMax H3 Lite) and other extended models. This video generation V2 endpoint accepts multimodal input via the
content array (text / image / video / audio) to support text-to-video, image-to-video (first \u0026 last frame), and reference-to-video, with output up to 2K.\", _jsx(\"br\", {}), _jsx(\"br\", {}), \
"Note: To use the flagship models or other extended models, please select the \", _jsx(_components.a, {\n href: \"https://platform.minimax.io/docs/pricing/overview#models/video\",\n children: \
"Pay-as-you-go API\"\n }), \".\"]\n });\n}\nfunction MDXContent(props = {}) {\n const {wrapper: MDXLayout} = {\n ..._provideComponents(),\n ...props.components\n };\n return MDXLayout ? _jsx
(MDXLayout, {\n ...props,\n children: _jsx(_createMdxContent, {\n ...props\n })\n }) : _createMdxContent(props);\n}\nreturn {\n default: MDXContent\n};\n","frontmatter":{},"scope":
{}},"pageType":"default","mdx":{"mdxSource":{"compiledSource":"\"use strict\";const{Fragment:_Fragment,jsx:_jsx}=arguments[0];const{useMDXComponents:_provideComponents}=arguments[0];function
_createMdxContent(props){return _jsx(_Fragment,{})}function MDXContent(props={}){const{wrapper:MDXLayout}={..._provideComponents(),...props.components};return MDXLayout?_jsx(MDXLayout,
{...props,children:_jsx(_createMdxContent,{...props})}):_createMdxContent(props)}return{default:MDXContent}","frontmatter":{},"scope":{"config":{},"pageMetadata":{"title":"Create Video Generation
Task","sidebarTitle":"Create Video Generation Task","description":"Change the `model` field to switch between the flagship models (MiniMax H3, MiniMax H3 Highspeed, and MiniMax H3 Lite) and other
extended models. This video generation V2 endpoint accepts multimodal input via the content array (text / image / video / audio) to support text-to-video, image-to-video (first \u0026 last frame),
and reference-to-video, with output up to 2K.\u003cbr />\u003cbr />Note: To use the flagship models or other extended models, please select the [Pay-as-you-go API](https://platform.minimax.io/docs/
pricing/overview#models/video).","openapi":"api-reference/video/generation/api/v2-video-generation.json POST /v2/video_generation","href":"/api-reference/
video-generation-v2-create"}}},"mdxSourceWithNoJs":null},"mdxExtracts":{"tableOfContents":[],"codeExamples":{}},"jsLinks":["/docs/_mintlify/custom/minimax-cac98058/74cbedfd7ddc6a81/scripts/
auto-token.js","/docs/_mintlify/custom/minimax-cac98058/08175b8daa74cd64/scripts/logo.js","/docs/_mintlify/custom/minimax-cac98058/bdfa294452778802/scripts/head-userinfo.js","/docs/_mintlify/custom/
minimax-cac98058/ee26a6e83cc5ae7f/scripts/navbar-same-tab.js","/docs/_mintlify/custom/minimax-cac98058/b2a91de372d716f8/scripts/pricing-dropdown.js","/docs/_mintlify/custom/minimax-cac98058/
c76999ca41eb4aec/scripts/domain-link-rewrite.js","/docs/_mintlify/custom/minimax-cac98058/4fe58d73df53e822/scripts/cookie-consent.js","/docs/_mintlify/custom/minimax-cac98058/444224329a77c4b1/scripts
/page-view-tracking.js","/docs/_mintlify/custom/minimax-cac98058/07d15e8ada8d8bbf/scripts/restricted-docs-core.js","/docs/_mintlify/custom/minimax-cac98058/409396746d530a03/scripts/
restricted-docs-highlight.js","/docs/_mintlify/custom/minimax-cac98058/728362c2608a54be/scripts/restricted-docs.js","/docs/_mintlify/custom/minimax-cac98058/50d87c5c4c9a98b0/scripts/rum.js","/docs/
_mintlify/custom/minimax-cac98058/fbc61270b1626c86/scripts/pricing-navigation.js","/docs/_mintlify/custom/minimax-cac98058/b067abec70d0dbb6/scripts/feedback-consent.js"],"openApiReferenceData":
{"operation":{"summary":"Create video generation task","description":"Change the `model` field to switch between the flagship models (MiniMax H3, MiniMax H3 Highspeed, and MiniMax H3 Lite) and other
extended models. This video generation V2 endpoint accepts multimodal input via the content array (text / image / video / audio) to support text-to-video, image-to-video (first \u0026 last frame),
and reference-to-video, with output up to 2K.\u003cbr />\u003cbr />Note: To use the flagship models or other extended models, please select the [Pay-as-you-go API](https://platform.minimax.io/docs/
pricing/overview#models/video).","operationId":"videoGenerationV2Create","tags":["Video V2"],"parameters":
["c25778a1-6dc4-473f-a321-aff36d5b64af"],"requestBody":"3d94362c-9d19-45f3-801e-878bb6829c89","responses":
{"200":"5035c119-5393-49ec-8485-0c7828fbfccb","400":"95168244-84a3-4ba4-bd1a-f7a23652ab61","401":"5a0af1d7-95db-470b-ba15-efadadeadec8","402":"f2ff4df4-f359-42f2-baeb-dc44c65b0cfd","422":"44e61ea7-ea9d-40ff-98fb-b0bbca46accb","429":"c0c50602-c403-4410-94d3-f9433d3f7abf","500":"e6330e2c-5938-46a4-8383-85d0d319677a"},"method":"post","type":"path","servers":
["a820e153-6e3f-4d43-93a8-e0e967aa5027"],"security":["18031243-2e03-4249-9454-d0947c3b2763"],"path":"/v2/video_generation","baseUrlOptions":["https://api.minimax.io"],"title":"Create video generation
task","codeSamples":[],"requiredOnlyExamples":false,"requestExampleType":"generated","responseExampleType":"generated","prefillPlaygroundWithExample":true,"prefillPlaygroundExamples":
[{"body.model":"MiniMax-H3","body.content.items.0.type":"text","body.content.items.0.text":"Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as
the last fleet gathers and jumps away in a blinding flash, the bridge shaking, leaving her behind.","body.resolution":"2K","body.duration":5,"body.ratio":"16:9"},
{"body.model":"MiniMax-H3","body.content.items.0.type":"text","body.content.items.0.text":"Pull focus to the people in the background and add more steam to the ramen
bowl.","body.content.items.1.type":"image_url","body.content.items.1.image_url.url":"https://cdn.hailuoai.com/prod/hailuo_demo/testsets/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/
4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png","body.content.items.1.role":"first_frame","body.resolution":"2K","body.duration":5,"body.ratio":"adaptive"},
{"body.model":"MiniMax-H3","body.content.items.0.type":"text","body.content.items.0.text":"Character speaks: Follow the wind, live free. Leave worries behind, enjoy the moment. Voice timbre follows
reference audio 1.","body.content.items.1.type":"video_url","body.content.items.1.video_url.url":"https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/
sr_v2p26_trio_seed42_20260724/inputs/
297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4","body.content.items.1.role":"reference_video","body.content.items.2.type":"audio_url","body.content.items.2.audio_url.url":"https:
//cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5
(1).mp3","body.content.items.2.role":"reference_audio","body.resolution":"2K","body.duration":5,"body.ratio":"adaptive"},
{"body.model":"MiniMax-H3","body.content.items.0.type":"text","body.content.items.0.text":"A small paper boat drifts across a calm
pond.","body.resolution":"768P","body.duration":6,"body.ratio":"4:3","body.generate_audio":false}]},"dependencies":{"parameters":{"c25778a1-6dc4-473f-a321-aff36d5b64af":
{"name":"Content-Type","in":"header","required":true,"description":"Media type of the request body. Set it to `application/json`.","schema":{"type":"string","enum":["application/
json"],"default":"application/json","isRequired":true,"description":"Media type of the request body. Set it to `application/json`.","uniqueKey":"","typeLabel":"enum\u003cstring>"}}},"requestBody":
{"content":{"application/json":{"schema":{"type":"object","required":["model","content","resolution","duration"],"properties":{"model":{"type":"string","description":"Model name. Currently available:
`MiniMax-H3`, `MiniMax-H3-Highspeed`, `MiniMax-H3-Lite`, `MiniMax-H3-Max`, `MiniMax-H3-Max-Turbo`, `MiniMax-H3-MG`, `MiniMax-H3-NC`.\n\n- **`MiniMax-H3`**: **the variant for high-quality video
creation.** Supports text-to-video, image-to-video (first / last frame), and reference-to-video; `480P` / `768P` / `2K` resolution, 4–15s duration. `480P` output is served by RunningHub and billed at
the MiniMax-H3 480P price under H3 Series Extended Models; no separate model name is needed.\n- **`MiniMax-H3-Highspeed`**: **balances generation quality and speed.** Supports text-to-video,
image-to-video (first / last frame), and reference-to-video (reference image / video / audio); `480P` / `768P` resolution, **`2K` is not supported**; 4–15s duration.\n- **`MiniMax-H3-Lite`**: **the
lightweight variant.** Supports text-to-video, image-to-video (first / last frame), and reference-to-video (reference image / video / audio); `480P` / `768P` resolution, **`2K` is not supported**;
4–15s duration.\n- **`MiniMax-H3-Max`**: **the fast generation variant.** Supports text-to-video, image-to-video (first / last frame), and reference-to-video (reference image / video / audio); `480P`
/ `768P` resolution, **`2K` is not supported**; 5–15s duration.\n- **`MiniMax-H3-Max-Turbo`**: **a faster, lower-cost version of `MiniMax-H3-Max`.** Supports text-to-video, image-to-video (first /
last frame), and reference-to-video (reference image / video / audio); `480P` / `768P` resolution, **`2K` is not supported**; 5–15s duration.\n- **`MiniMax-H3-MG`**: **MiniMax H3 hosted by MachGen,
with output up to 2K.** Supports text-to-video, image-to-video (first / last frame), and reference-to-video (reference image / video / audio); `768P` / `2K` resolution, **`480P` is not supported**;
4–15s duration.\n- **`MiniMax-H3-NC`**: **MiniMax H3 Fast by Nunchux AI, for fast previews and rapid iteration.** Supports text-to-video, image-to-video (first / last frame), and reference-to-video
(reference image / video / audio); `480P` / `768P` resolution, **`2K` is not supported**; 4–15s duration.","enum":
["MiniMax-H3","MiniMax-H3-Highspeed","MiniMax-H3-Lite","MiniMax-H3-Max","MiniMax-H3-Max-Turbo","MiniMax-H3-MG","MiniMax-H3-NC"],"isRequired":true,"uniqueKey":"model","typeLabel":"enum\u003cstring>
"},"content":{"type":"array","description":"Array of multimodal input describing the information used to generate the video. Each element is distinguished by `type` (`text` / `image_url` / `video_url
` / `audio_url`) and can be labeled with a `role`.\n\n**Every request must include one non-empty `text` item (the prompt is required)**; otherwise a parameter error is returned.\n\nSupported input
combinations (corresponding to different generation scenarios):\n- **Text-to-video**: a single `text` element only.\n- **Image-to-video, first frame**: `text` + 1 `image_url` (`role=first_frame`, or
omitted).\n- **Image-to-video, last frame**: `text` + 1 `image_url` (`role=last_frame`).\n- **Image-to-video, first \u0026 last frame**: `text` + 2 `image_url` items with `role` set to `first_frame`
and `last_frame` respectively.\n- **Reference-to-video**: `text` + any combination of reference images (`role=reference_image`), reference videos (`role=reference_video`), and reference audio (`role=
reference_audio`).\n\n> **Image-to-video and reference-to-video are mutually exclusive**: if any `reference_image` / `reference_video` / `reference_audio` role appears in content, then `first_frame`
/ `last_frame` must not appear (and vice versa); the two cannot be mixed.\n\n---\n\n**Input media limits** (total request body ≤ 64 MB; use public URLs for large files, avoid Base64)\n\nImage
`image_url`:\n\n| Item | Limit |\n| :--- | :--- |\n| Format | JPG, JPEG, PNG, WEBP, HEIC, HEIF |\n| Single file size | ≤ 30 MB |\n| Width/height range | [256, 5760] px |\n| Aspect ratio (w/h) | [0.4,
2.5] |\n| Count | first frame ≤ 1, last frame ≤ 1, reference images ≤ 9 |\n\nVideo `video_url` (reference scenario only):\n\n| Item | Limit |\n| :--- | :--- |\n| Container / format | MP4 (`.mp4`),
MOV (`.mov`) |\n| Codec | Video H.264/AVC, H.265/HEVC; audio AAC, MP3 |\n| Single file size | ≤ 50 MB |\n| Count | ≤ 3 |\n| Per-clip duration | [2, 15] s; total ≤ 15 s |\n| Width/height range | [256,
5760] px |\n| Aspect ratio (w/h) | [0.4, 2.5] |\n| Frame rate | [23.976, 60] |\n\nAudio `audio_url` (reference scenario only):\n\n| Item | Limit |\n| :--- | :--- |\n| Format | WAV, MP3 |\n| Single
file size | ≤ 15 MB |\n| Count | ≤ 3 |\n| Per-clip duration | [2, 15] s; total ≤ 15 s |","items":{"type":"object","required":["type"],"properties":{"type":{"type":"string","description":"Type of the
input content.","enum":["text","image_url","video_url","audio_url"],"isRequired":true,"uniqueKey":"content.items.[INDEX].type","typeLabel":"enum\u003cstring>"},"text":
{"type":"string","description":"Text prompt, **required**: every scenario must include one non-empty `text` describing the desired video. Length is counted by characters, with a maximum of 7000
characters per `text`.","uniqueKey":"content.items.[INDEX].text","typeLabel":"string"},"image_url":{"type":"object","description":"Image object when `type=image_url` (see the content description
above for format / size / dimension / count limits).","required":["url"],"properties":{"url":{"type":"string","description":"Image location. Supported: a public URL; `mm_file://{file_id}` (reference
an existing platform file, e.g. an uploaded file or a previous output's file_id); a `data:image/\u003cformat>;base64,\u003cBase64>` data URI (`\u003cformat>`
lowercase).","isRequired":true,"uniqueKey":"content.items.[INDEX].image_url.url","typeLabel":"string"}},"uniqueKey":"content.items.[INDEX].image_url","typeLabel":"object"},"video_url":
{"type":"object","description":"Video object when `type=video_url` (reference video, reference scenario only; see the content description above for format / size / duration limits).","required":
["url"],"properties":{"url":{"type":"string","description":"Video location. Supported: a public URL; `mm_file://{file_id}` (reference an existing platform file's file_id); a `data:video/mp4;base64,\
u003cBase64>` data URI. Note the total request body must be ≤ 64 MB and Base64 inflates size by about 33%, so use a public URL or mm_file:// for large
videos.","isRequired":true,"uniqueKey":"content.items.[INDEX].video_url.url","typeLabel":"string"}},"uniqueKey":"content.items.[INDEX].video_url","typeLabel":"object"},"audio_url":
{"type":"object","description":"Audio object when `type=audio_url` (reference audio, reference scenario only; see the content description above for format / size / duration limits).","required":
["url"],"properties":{"url":{"type":"string","description":"Audio location. Supported: a public URL; `mm_file://{file_id}` (reference an existing platform file's file_id); a `data:audio/\
u003cformat>;base64,\u003cBase64>` data URI (`\u003cformat>` lowercase).","isRequired":true,"uniqueKey":"content.items.[INDEX].audio_url.url","typeLabel":"string"}},"uniqueKey":"content.items.
[INDEX].audio_url","typeLabel":"object"},"role":{"type":"string","description":"Position or purpose of the content, conditionally required:\n- `first_frame`: first-frame image (image-to-video; when
there is a single image and no role is set, it defaults to first_frame).\n- `last_frame`: last-frame image (image-to-video first \u0026 last frame, must be paired with first_frame).\n-
`reference_image`: reference image (reference-to-video).\n- `reference_video`: reference video (reference-to-video).\n- `reference_audio`: reference audio (reference-to-video).","enum":
["first_frame","last_frame","reference_image","reference_video","reference_audio"],"uniqueKey":"content.items.[INDEX].role","typeLabel":"enum\u003cstring>"}},"uniqueKey":"content.items.[INDEX]
","typeLabel":"object"},"isRequired":true,"uniqueKey":"content","typeLabel":"object[]"},"resolution":{"type":"string","description":"Video resolution. Availability differs by model:\n\n- **
`MiniMax-H3`**: `480P`, `768P`, `2K` (`480P` is served by RunningHub; keep `model` as `MiniMax-H3` and set `resolution` to `480P`)\n- **`MiniMax-H3-Highspeed`**: `480P`, `768P` (`2K` is not
supported)\n- **`MiniMax-H3-Lite`**: `480P`, `768P` (`2K` is not supported)\n- **`MiniMax-H3-Max`**: `480P`, `768P` (`2K` is not supported)\n- **`MiniMax-H3-Max-Turbo`**: `480P`, `768P` (`2K` is not
supported)\n- **`MiniMax-H3-MG`**: `768P`, `2K` (`480P` is not supported)\n- **`MiniMax-H3-NC`**: `480P`, `768P` (`2K` is not supported)","enum":
["480P","768P","2K"],"isRequired":true,"uniqueKey":"resolution","typeLabel":"enum\u003cstring>"},"duration":{"type":"integer","description":"Duration of the generated video in seconds. Required,
integer. Available values differ by model:\n\n- **`MiniMax-H3`**: `4`–`15`\n- **`MiniMax-H3-Highspeed`**: `4`–`15`\n- **`MiniMax-H3-Lite`**: `4`–`15`\n- **`MiniMax-H3-Max`**: `5`–`15` (4 seconds is
not supported)\n- **`MiniMax-H3-Max-Turbo`**: `5`–`15` (4 seconds is not supported)\n- **`MiniMax-H3-MG`**: `4`–`15`\n- **`MiniMax-H3-NC`**: `4`–`15`","enum":
[4,5,6,7,8,9,10,11,12,13,14,15],"isRequired":true,"uniqueKey":"duration","typeLabel":"enum\u003cinteger>"},"ratio":{"type":"string","description":"Aspect ratio of the generated video. Defaults to
`adaptive` (the most suitable ratio is chosen automatically based on the input; the actual ratio can be read from the `ratio` field of the query endpoint). Available values: `adaptive`, `21:9`, `16:9
`, `4:3`, `1:1`, `3:4`, `9:16`.\n\n**Text-to-video (t2va, content contains only `text`)**: `ratio` is required and cannot be `adaptive`; available values `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`.\
n\n**Image-to-video (i2va, content contains a `first_frame` / `last_frame` image)**: the aspect ratio is determined by the input image and `ratio` is always `adaptive`; passing another valid value
does not error but is ignored and treated as `adaptive`.\n\n**Reference-to-video (r2va, content contains `reference_image` / `reference_video` / `reference_audio`)**: `ratio` is optional and defaults
to `adaptive`; you may also explicitly specify any of the concrete ratios above. `MiniMax-H3-Lite` and `MiniMax-H3-NC` do not support `adaptive` yet; specify a concrete ratio.","enum":
["adaptive","21:9","16:9","4:3","1:1","3:4","9:16"],"uniqueKey":"ratio","typeLabel":"enum\u003cstring>"},"extra":{"type":"object","description":"Additional generation options for `MiniMax-H3-Max` and
`MiniMax-H3-Max-Turbo`. Currently only `prompt_expansion_mode` is supported. If `extra` or this property is omitted, `balanced` is used. Do not pass `balance`, an empty string, a Boolean value, or
other undeclared fields.","properties":{"prompt_expansion_mode":{"type":"string","description":"Prompt expansion mode. `disabled` disables prompt expansion; `balanced` uses balanced expansion and is
the default; `quality` prioritizes expansion quality.","enum":["disabled","balanced","quality"],"default":"balanced","uniqueKey":"extra.prompt_expansion_mode","typeLabel":"enum\u003cstring>
"}},"additionalProperties":false,"uniqueKey":"extra","typeLabel":"object"},"generate_audio":{"type":"boolean","description":"Controls whether the generated video includes audio. Optional; defaults to
`true`.\n\n- `true`: output a video with audio.\n- `false`: output a silent video.","default":true,"uniqueKey":"generate_audio","typeLabel":"boolean"},"callback_url":
{"type":"string","description":"Callback URL for task status changes. Once configured, the MiniMax server first sends a verification request containing a `challenge` field (you must return the
`challenge` unchanged within 3 seconds to complete verification); after verification succeeds, it POSTs an update to this URL whenever the task status changes. The push body has the same structure as
the response of the [Query Task](/api-reference/video-generation-v2-query) endpoint.\n\nCallback `status` values: `queued`, `running`, `succeeded`, `failed`, `cancelled
`.","uniqueKey":"callback_url","typeLabel":"string"},"response_format":{"type":"string","enum":["task","flat"],"default":"task","description":"Response format.\n\n- `task` (default): returns `{\"task
\": {...}}`, with task fields nested under the `task` object.\n- `flat`: task fields are flattened at the top level of the response (including `usage`), matching the shape of `items[]` in the task
list response.","uniqueKey":"response_format","typeLabel":"enum\u003cstring>"}},"example":{"model":"MiniMax-H3","content":[{"type":"text","text":"A boy playing basketball by the
sea"}],"resolution":"2K","duration":5,"ratio":"16:9"},"uniqueKey":"","typeLabel":"object"},"examples":{"Text-to-video (t2va)":{"title":"Text-to-video (t2va)","value":{"model":"MiniMax-H3","content":
[{"type":"text","text":"Epic space-opera theatrical teaser: a female captain stands alone before a massive observation window as the last fleet gathers and jumps away in a blinding flash, the bridge
shaking, leaving her behind."}],"resolution":"2K","duration":5,"ratio":"16:9"}},"Image-to-video (i2va)":{"title":"Image-to-video (i2va)","value":{"model":"MiniMax-H3","content":
[{"type":"text","text":"Pull focus to the people in the background and add more steam to the ramen bowl."},{"type":"image_url","image_url":{"url":"https://cdn.hailuoai.com/prod/hailuo_demo/testsets/
H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/
4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png"},"role":"first_frame"}],"resolution":"2K","duration":5,"ratio":"adaptive"}},"Reference-to-video (r2va)":{"title":"Reference-to-video (r2va)","value":
{"model":"MiniMax-H3","content":[{"type":"text","text":"Character speaks: Follow the wind, live free. Leave worries behind, enjoy the moment. Voice timbre follows reference audio 1."},
{"type":"video_url","video_url":{"url":"https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/
297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4"},"role":"reference_video"},{"type":"audio_url","audio_url":{"url":"https://cdn.hailuoai.com/prod/hailuo_demo/
testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5
(1).mp3"},"role":"reference_audio"}],"resolution":"2K","duration":5,"ratio":"adaptive"}},"Silent video (generate_audio=false)":{"title":"Silent video (generate_audio=false)","summary":"Generate a
silent video","value":{"model":"MiniMax-H3","content":[{"type":"text","text":"A small paper boat drifts across a calm
pond."}],"resolution":"768P","duration":6,"ratio":"4:3","generate_audio":false}}},"exampleType":"examples"}},"required":true,"exampleType":"examples"},"responses":
{"5035c119-5393-49ec-8485-0c7828fbfccb":{"description":"The create endpoint returns a `task_id`. Use this `task_id` with the [Query Task](/api-reference/video-generation-v2-query) endpoint to
retrieve the task status and result.\n\n**Successful Query Task response example**\n\n```json\n{\n \"task\": {\n \"id\": \"424010985738629\",\n \"model\": \"MiniMax-H3\",\n \"status\": \"succeeded\",
\n \"created_at\": 1785125529,\n \"updated_at\": 1785125946,\n \"content\": {\n \"url\": \"https://your-cdn.example.com/h3-generated-2k-output.mp4\"\n },\n \"resolution\": \"2K\",\n \"duration\": 5,\
n \"usage\": {\n \"total_seconds\": 5,\n \"input_seconds\": 0,\n \"output_seconds\": 5,\n \"input_image_count\": 0\n },\n \"ratio\": \"16:9\",\n \"task_type\": \"generation\",\n \"modality\": \"video
\"\n }\n}\n```","content":{"application/json":{"schema":{"type":"object","properties":{"task_id":{"type":"string","description":"ID of the task, used to query the task status and result
later.","uniqueKey":"task_id","typeLabel":"string"}},"example":{"task_id":"424010985738629"},"uniqueKey":"","typeLabel":"object"},"examples":{"Example":{"title":"Example","value":
{"task_id":"424010985738629"}}},"exampleType":"generated"}},"exampleType":"generated"},"95168244-84a3-4ba4-bd1a-f7a23652ab61":{"description":"Invalid parameters","content":{"application/json":
{"schema":{"type":"object","description":"OpenAI-style error response. On error the HTTP status is the real error code (401/400/429/402/422/500…) and the body is this object.","properties":{"type":
{"type":"string","description":"Always `error`.","example":"error","uniqueKey":"type","typeLabel":"string"},"error":{"type":"object","properties":{"type":{"type":"string","description":"Error type:
`authorized_error`(401)/`bad_request_error`(400)/`rate_limit_error`(429)/`insufficient_balance_error`(402)/`unprocessable_entity_error`(422)/`overloaded_error`(529)/`server_error`(500),
etc.","uniqueKey":"error.type","typeLabel":"string"},"message":{"type":"string","description":"Error detail; the trailing parenthesis holds the internal error code (e.g. `... (1004)
`).","uniqueKey":"error.message","typeLabel":"string"},"http_code":{"type":"string","description":"HTTP status code as a string, e.g. `401
`.","uniqueKey":"error.http_code","typeLabel":"string"}},"uniqueKey":"error","typeLabel":"object"},"request_id":{"type":"string","description":"Request trace ID (for
troubleshooting).","uniqueKey":"request_id","typeLabel":"string"}},"uniqueKey":"","typeLabel":"object"},"example":{"type":"error","error":{"type":"bad_request_error","message":"invalid params,
content must include a non-empty text item (prompt is required) (2013)","http_code":"400"},"request_id":"021785229015510a2c883cf675b9804d"},"examples":{"Example":{"title":"Example","value":
{"type":"error","error":{"type":"bad_request_error","message":"invalid params, content must include a non-empty text item (prompt is required) (2013)
","http_code":"400"},"request_id":"021785229015510a2c883cf675b9804d"}}},"exampleType":"examples"}},"exampleType":"examples"},"5a0af1d7-95db-470b-ba15-efadadeadec8":{"description":"Authentication
failed","content":{"application/json":{"schema":{"type":"object","description":"OpenAI-style error response. On error the HTTP status is the real error code (401/400/429/402/422/500…) and the body is
this object.","properties":{"type":{"type":"string","description":"Always `error`.","example":"error","uniqueKey":"type","typeLabel":"string"},"error":{"type":"object","properties":{"type":
{"type":"string","description":"Error type: `authorized_error`(401)/`bad_request_error`(400)/`rate_limit_error`(429)/`insufficient_balance_error`(402)/`unprocessable_entity_error`(422)/
`overloaded_error`(529)/`server_error`(500), etc.","uniqueKey":"error.type","typeLabel":"string"},"message":{"type":"string","description":"Error detail; the trailing parenthesis holds the internal
error code (e.g. `... (1004)`).","uniqueKey":"error.message","typeLabel":"string"},"http_code":{"type":"string","description":"HTTP status code as a string, e.g. `401
`.","uniqueKey":"error.http_code","typeLabel":"string"}},"uniqueKey":"error","typeLabel":"object"},"request_id":{"type":"string","description":"Request trace ID (for
troubleshooting).","uniqueKey":"request_id","typeLabel":"string"}},"uniqueKey":"","typeLabel":"object"},"example":{"type":"error","error":{"type":"authorized_error","message":"login fail: Please
carry the API secret key in the 'Authorization' field of the request header (1004)","http_code":"401"},"request_id":"021785229015510a2c883cf675b9804d"},"examples":{"Example":
{"title":"Example","value":{"type":"error","error":{"type":"authorized_error","message":"login fail: Please carry the API secret key in the 'Authorization' field of the request header (1004)
","http_code":"401"},"request_id":"021785229015510a2c883cf675b9804d"}}},"exampleType":"examples"}},"exampleType":"examples"},"f2ff4df4-f359-42f2-baeb-dc44c65b0cfd":{"description":"Insufficient
balance/quota","content":{"application/json":{"schema":{"type":"object","description":"OpenAI-style error response. On error the HTTP status is the real error code (401/400/429/402/422/500…) and the
body is this object.","properties":{"type":{"type":"string","description":"Always `error`.","example":"error","uniqueKey":"type","typeLabel":"string"},"error":{"type":"object","properties":{"type":
{"type":"string","description":"Error type: `authorized_error`(401)/`bad_request_error`(400)/`rate_limit_error`(429)/`insufficient_balance_error`(402)/`unprocessable_entity_error`(422)/
`overloaded_error`(529)/`server_error`(500), etc.","uniqueKey":"error.type","typeLabel":"string"},"message":{"type":"string","description":"Error detail; the trailing parenthesis holds the internal
error code (e.g. `... (1004)`).","uniqueKey":"error.message","typeLabel":"string"},"http_code":{"type":"string","description":"HTTP status code as a string, e.g. `401
`.","uniqueKey":"error.http_code","typeLabel":"string"}},"uniqueKey":"error","typeLabel":"object"},"request_id":{"type":"string","description":"Request trace ID (for
troubleshooting).","uniqueKey":"request_id","typeLabel":"string"}},"uniqueKey":"","typeLabel":"object"},"example":{"type":"error","error":{"type":"insufficient_balance_error","message":"insufficient
balance (1008)","http_code":"402"},"request_id":"021785229015510a2c883cf675b9804d"},"examples":{"Example":{"title":"Example","value":{"type":"error","error":
{"type":"insufficient_balance_error","message":"insufficient balance (1008)
","http_code":"402"},"request_id":"021785229015510a2c883cf675b9804d"}}},"exampleType":"examples"}},"exampleType":"examples"},"44e61ea7-ea9d-40ff-98fb-b0bbca46accb":{"description":"Input contains
sensitive content","content":{"application/json":{"schema":{"type":"object","description":"OpenAI-style error response. On error the HTTP status is the real error code (401/400/429/402/422/500…) and
the body is this object.","properties":{"type":{"type":"string","description":"Always `error`.","example":"error","uniqueKey":"type","typeLabel":"string"},"error":{"type":"object","properties":
{"type":{"type":"string","description":"Error type: `authorized_error`(401)/`bad_request_error`(400)/`rate_limit_error`(429)/`insufficient_balance_error`(402)/`unprocessable_entity_error`(422)/
`overloaded_error`(529)/`server_error`(500), etc.","uniqueKey":"error.type","typeLabel":"string"},"message":{"type":"string","description":"Error detail; the trailing parenthesis holds the internal
error code (e.g. `... (1004)`).","uniqueKey":"error.message","typeLabel":"string"},"http_code":{"type":"string","description":"HTTP status code as a string, e.g. `401
`.","uniqueKey":"error.http_code","typeLabel":"string"}},"uniqueKey":"error","typeLabel":"object"},"request_id":{"type":"string","description":"Request trace ID (for
troubleshooting).","uniqueKey":"request_id","typeLabel":"string"}},"uniqueKey":"","typeLabel":"object"},"example":{"type":"error","error":{"type":"unprocessable_entity_error","message":"video
description contains sensitive content (1026)","http_code":"422"},"request_id":"021785229015510a2c883cf675b9804d"},"examples":{"Example":{"title":"Example","value":{"type":"error","error":
{"type":"unprocessable_entity_error","message":"video description contains sensitive content (1026)
","http_code":"422"},"request_id":"021785229015510a2c883cf675b9804d"}}},"exampleType":"examples"}},"exampleType":"examples"},"c0c50602-c403-4410-94d3-f9433d3f7abf":{"description":"Rate limit
triggered","content":{"application/json":{"schema":{"type":"object","description":"OpenAI-style error response. On error the HTTP status is the real error code (401/400/429/402/422/500…) and the body
is this object.","properties":{"type":{"type":"string","description":"Always `error`.","example":"error","uniqueKey":"type","typeLabel":"string"},"error":{"type":"object","properties":{"type":
{"type":"string","description":"Error type: `authorized_error`(401)/`bad_request_error`(400)/`rate_limit_error`(429)/`insufficient_balance_error`(402)/`unprocessable_entity_error`(422)/
`overloaded_error`(529)/`server_error`(500), etc.","uniqueKey":"error.type","typeLabel":"string"},"message":{"type":"string","description":"Error detail; the trailing parenthesis holds the internal
error code (e.g. `... (1004)`).","uniqueKey":"error.message","typeLabel":"string"},"http_code":{"type":"string","description":"HTTP status code as a string, e.g. `401
`.","uniqueKey":"error.http_code","typeLabel":"string"}},"uniqueKey":"error","typeLabel":"object"},"request_id":{"type":"string","description":"Request trace ID (for
troubleshooting).","uniqueKey":"request_id","typeLabel":"string"}},"uniqueKey":"","typeLabel":"object"},"example":{"type":"error","error":{"type":"rate_limit_error","message":"rate limit, please
retry later (1002)","http_code":"429"},"request_id":"021785229015510a2c883cf675b9804d"},"examples":{"Example":{"title":"Example","value":{"type":"error","error":
{"type":"rate_limit_error","message":"rate limit, please retry later (1002)
","http_code":"429"},"request_id":"021785229015510a2c883cf675b9804d"}}},"exampleType":"examples"}},"exampleType":"examples"},"e6330e2c-5938-46a4-8383-85d0d319677a":{"description":"Server
error","content":{"application/json":{"schema":{"type":"object","description":"OpenAI-style error response. On error the HTTP status is the real error code (401/400/429/402/422/500…) and the body is
this object.","properties":{"type":{"type":"string","description":"Always `error`.","example":"error","uniqueKey":"type","typeLabel":"string"},"error":{"type":"object","properties":{"type":
{"type":"string","description":"Error type: `authorized_error`(401)/`bad_request_error`(400)/`rate_limit_error`(429)/`insufficient_balance_error`(402)/`unprocessable_entity_error`(422)/
`overloaded_error`(529)/`server_error`(500), etc.","uniqueKey":"error.type","typeLabel":"string"},"message":{"type":"string","description":"Error detail; the trailing parenthesis holds the internal
error code (e.g. `... (1004)`).","uniqueKey":"error.message","typeLabel":"string"},"http_code":{"type":"string","description":"HTTP status code as a string, e.g. `401
`.","uniqueKey":"error.http_code","typeLabel":"string"}},"uniqueKey":"error","typeLabel":"object"},"request_id":{"type":"string","description":"Request trace ID (for
troubleshooting).","uniqueKey":"request_id","typeLabel":"string"}},"uniqueKey":"","typeLabel":"object"},"example":{"type":"error","error":{"type":"server_error","message":"internal error (1000)
","http_code":"500"},"request_id":"021785229015510a2c883cf675b9804d"},"examples":{"Example":{"title":"Example","value":{"type":"error","error":{"type":"server_error","message":"internal error (1000)
","http_code":"500"},"request_id":"021785229015510a2c883cf675b9804d"}}},"exampleType":"examples"}},"exampleType":"examples"}},"security":{"18031243-2e03-4249-9454-d0947c3b2763":{"bearerAuth":
{"type":"http","scheme":"bearer","bearerFormat":"JWT","description":"`HTTP: Bearer Auth`\n- Security Scheme Type: http\n- HTTP Authorization Scheme: `Bearer API_key`, used to verify account
information, can be found in [Account Management>API Keys](https://platform.minimax.io/user-center/basic-information/interface-key).","scopes":[],"schemeName":"bearerAuth"}}},"servers":
{"a820e153-6e3f-4d43-93a8-e0e967aa5027":{"url":"https://api.minimax.io"}},"processedSecurityOptions":[{"schemes":[{"scheme":{"type":"http","scheme":"bearer","bearerFormat":"JWT","description":"`HTTP:
Bearer Auth`\n- Security Scheme Type: http\n- HTTP Authorization Scheme: `Bearer API_key`, used to verify account information, can be found in [Account Management>API Keys](https://
platform.minimax.io/user-center/basic-information/interface-key).","scopes":[],"schemeName":"bearerAuth"},"schema":{"title":"Bearer Auth","type":"string","isRequired":true,"description":"`HTTP:
Bearer Auth`\n- Security Scheme Type: http\n- HTTP Authorization Scheme: `Bearer API_key`, used to verify account information, can be found in [Account Management>API Keys](https://
platform.minimax.io/user-center/basic-information/interface-key).","format":"bearer","typeLabel":"string\u003cbearer>
","deprecated":false},"schemeName":"bearerAuth","name":"Authorization"}],"title":"bearerAuth"}]},"originalFileLocation":"api-reference/video/generation/api/
v2-video-generation.json"},"subdomain":"minimax-cac98058","siteName":"MiniMax API Docs","title":"Create Video Generation Task - MiniMax API Docs","contextualStarterQuestions":[],"navMaps":
{"firstHrefInVersion":{},"firstHrefInLanguage":{}},"lastModified":"2026-11-23T16:11:07.998Z","originalFileLocation":"api-reference/video-generation-v2-create.mdx"},"headHtml":"\u003cscript
data-mint-head>(function(){var d=document.documentElement,p=null;if(!p){try{p=localStorage.getItem(\"isDarkMode\")}catch(e){}if(p==='true')p='dark';else if(p==='false')p='light';if(p!=='light'\u0026\
u0026p!=='dark'\u0026\u0026p!=='system')p=\"system\"}var r=p==='system'?(matchMedia(\"(prefers-color-scheme: dark)\").matches?'dark':'light'):p;d.classList.remove(r==='dark'?'light':'dark');
d.classList.add(r);d.setAttribute(\"data-theme-preference\",p)})();\u003c/script>\u003cmeta data-mint-head name=\"description\" content=\"Change the model field to switch between the flagship models
(MiniMax H3, MiniMax H3 Highspeed, and MiniMax H3 Lite) and other extended models. This video generation V2 endpoint accepts multimodal input via the content array (text / image / video / audio) to
support text-to-video, image-to-video (first \u0026amp; last frame), and reference-to-video, with output up to 2K.Note: To use the flagship models or other extended models, please select the
Pay-as-you-go API.\">\u003cmeta data-mint-head name=\"application-name\" content=\"MiniMax API Docs\">\u003cmeta data-mint-head name=\"generator\" content=\"Mintlify\">\u003clink data-mint-head rel=\
"icon\" type=\"image/png\" sizes=\"192x192\" href=\"/docs/mintlify-assets/_mintlify/favicons/minimax-cac98058/-WSO_KEoennf5XoM/_generated/favicon/android-chrome-192x192.png\" media=\"
(prefers-color-scheme: light)\">\u003clink data-mint-head rel=\"apple-touch-icon\" type=\"image/png\" sizes=\"180x180\" href=\"/docs/mintlify-assets/_mintlify/favicons/minimax-cac98058/
-WSO_KEoennf5XoM/_generated/favicon/apple-touch-icon.png\">\u003clink data-mint-head rel=\"icon\" type=\"image/png\" sizes=\"16x16\" href=\"/docs/mintlify-assets/_mintlify/favicons/minimax-cac98058/
-WSO_KEoennf5XoM/_generated/favicon/favicon-16x16.png\" media=\"(prefers-color-scheme: light)\">\u003clink data-mint-head rel=\"icon\" type=\"image/png\" sizes=\"32x32\" href=\"/docs/mintlify-assets/
_mintlify/favicons/minimax-cac98058/-WSO_KEoennf5XoM/_generated/favicon/favicon-32x32.png\" media=\"(prefers-color-scheme: light)\">\u003clink data-mint-head rel=\"shortcut icon\" type=\"image/x-icon
\" href=\"/docs/mintlify-assets/_mintlify/favicons/minimax-cac98058/-WSO_KEoennf5XoM/_generated/favicon/favicon.ico\" media=\"(prefers-color-scheme: light)\">\u003clink data-mint-head rel=\"icon\"
type=\"image/png\" sizes=\"192x192\" href=\"/docs/mintlify-assets/_mintlify/favicons/minimax-cac98058/-WSO_KEoennf5XoM/_generated/favicon-dark/android-chrome-192x192.png\" media=\"
(prefers-color-scheme: dark)\">\u003clink data-mint-head rel=\"icon\" type=\"image/png\" sizes=\"16x16\" href=\"/docs/mintlify-assets/_mintlify/favicons/minimax-cac98058/-WSO_KEoennf5XoM/_generated/
favicon-dark/favicon-16x16.png\" media=\"(prefers-color-scheme: dark)\">\u003clink data-mint-head rel=\"icon\" type=\"image/png\" sizes=\"32x32\" href=\"/docs/mintlify-assets/_mintlify/favicons/
minimax-cac98058/-WSO_KEoennf5XoM/_generated/favicon-dark/favicon-32x32.png\" media=\"(prefers-color-scheme: dark)\">\u003clink data-mint-head rel=\"shortcut icon\" type=\"image/x-icon\" href=\"/docs
/mintlify-assets/_mintlify/favicons/minimax-cac98058/-WSO_KEoennf5XoM/_generated/favicon-dark/favicon.ico\" media=\"(prefers-color-scheme: dark)\">\u003cmeta data-mint-head property=\"og:title\"
content=\"Create Video Generation Task - MiniMax API Docs\">\u003cmeta data-mint-head property=\"og:description\" content=\"Change the model field to switch between the flagship models (MiniMax H3,
MiniMax H3 Highspeed, and MiniMax H3 Lite) and other extended models. This video generation V2 endpoint accepts multimodal input via the content array (text / image / video / audio) to support
text-to-video, image-to-video (first \u0026amp; last frame), and reference-to-video, with output up to 2K.Note: To use the flagship models or other extended models, please select the Pay-as-you-go
API.\">\u003cmeta data-mint-head property=\"og:site_name\" content=\"MiniMax API Docs\">\u003cmeta data-mint-head property=\"og:url\" content=\"https://platform.minimax.io/docs/api-reference/
video-generation-v2-create\">\u003cmeta data-mint-head property=\"og:type\" content=\"website\">\u003cmeta data-mint-head property=\"og:image\" content=\"https://minimax-cac98058.mintlify.site/docs/
_mintlify/api/og/v2?division=MiniMax-H3\u0026amp;title=Create+Video+Generation+Task\u0026amp;description=
Change+the+%60model%60+field+to+switch+between+the+flagship+models+%28MiniMax+H3%2C+MiniMax+H3+Highspeed%2C+and+MiniMax+H3+Lite%29+and+other+extended+models.+This+video+gen\u0026amp;theme=
329045bc9618e343e09b4393\u0026amp;signature=43057cb00dc25566644a07571a5ab3ad771ca61ec083da32a2b1d7fa23ae2f84\">\u003cmeta data-mint-head property=\"og:image:width\" content=\"1200\">\u003cmeta
data-mint-head property=\"og:image:height\" content=\"630\">\u003cmeta data-mint-head name=\"twitter:card\" content=\"summary_large_image\">\u003cmeta data-mint-head name=\"twitter:title\" content=\
"Create Video Generation Task - MiniMax API Docs\">\u003cmeta data-mint-head name=\"twitter:description\" content=\"Change the model field to switch between the flagship models (MiniMax H3, MiniMax
H3 Highspeed, and MiniMax H3 Lite) and other extended models. This video generation V2 endpoint accepts multimodal input via the content array (text / image / video / audio) to support text-to-video,
image-to-video (first \u0026amp; last frame), and reference-to-video, with output up to 2K.Note: To use the flagship models or other extended models, please select the Pay-as-you-go API.\">\u003cmeta
data-mint-head name=\"twitter:image\" content=\"https://minimax-cac98058.mintlify.site/docs/_mintlify/api/og/v2?division=MiniMax-H3\u0026amp;title=Create+Video+Generation+Task\u0026amp;description=
Change+the+%60model%60+field+to+switch+between+the+flagship+models+%28MiniMax+H3%2C+MiniMax+H3+Highspeed%2C+and+MiniMax+H3+Lite%29+and+other+extended+models.+This+video+gen\u0026amp;theme=
329045bc9618e343e09b4393\u0026amp;signature=43057cb00dc25566644a07571a5ab3ad771ca61ec083da32a2b1d7fa23ae2f84\">\u003cmeta data-mint-head name=\"twitter:image:width\" content=\"1200\">\u003cmeta
data-mint-head name=\"twitter:image:height\" content=\"630\">\u003cmeta data-mint-head name=\"msapplication-config\" content=\"/docs/mintlify-assets/_mintlify/favicons/minimax-cac98058/
-WSO_KEoennf5XoM/_generated/favicon/browserconfig.xml\">\u003cmeta data-mint-head name=\"apple-mobile-web-app-title\" content=\"MiniMax API Docs\">\u003cmeta data-mint-head name=\
"msapplication-TileColor\" content=\"#181E25\">\u003cmeta data-mint-head name=\"charset\" content=\"UTF-8\">\u003cmeta data-mint-head name=\"viewport\" content=\"width=device-width, initial-scale=1.0
\">\u003clink data-mint-head rel=\"canonical\" href=\"https://platform.minimax.io/docs/api-reference/video-generation-v2-create\">\u003clink data-mint-head rel=\"alternate\" type=\"application/xml\"
href=\"/docs/sitemap.xml\">\u003clink data-mint-head rel=\"alternate\" type=\"text/markdown\" href=\"/docs/api-reference/video-generation-v2-create.md\">\u003cscript data-mint-head type=\"application
/ld+json\">{\"@context\":\"https://schema.org\",\"@graph\":[{\"@type\":\"Organization\",\"@id\":\"https://platform.minimax.io/#organization\",\"name\":\"MiniMax API Docs\",\"url\":\"https://
platform.minimax.io\",\"logo\":{\"@type\":\"ImageObject\",\"url\":\"https://mintcdn.com/minimax-cac98058/XYzsL2L2ynonu2Q_/logo/light.svg?fit=max\u0026auto=format\u0026n=XYzsL2L2ynonu2Q_\u0026q=85\
u0026s=9c99bb2678d62388d8a4e7cce1750858\"}},{\"@type\":\"WebSite\",\"@id\":\"https://platform.minimax.io/docs#website\",\"name\":\"MiniMax API Docs\",\"url\":\"https://platform.minimax.io/docs\",\
"publisher\":{\"@id\":\"https://platform.minimax.io/#organization\"}},{\"@type\":\"WebPage\",\"@id\":\"https://platform.minimax.io/docs/api-reference/video-generation-v2-create#webpage\",\"url\":\
"https://platform.minimax.io/docs/api-reference/video-generation-v2-create\",\"name\":\"Create Video Generation Task\",\"description\":\"Change the model field to switch between the flagship models
(MiniMax H3, MiniMax H3 Highspeed, and MiniMax H3 Lite) and other extended models. This video generation V2 endpoint accepts multimodal input via the content array (text / image / video / audio) to
support text-to-video, image-to-video (first \u0026 last frame), and reference-to-video, with output up to 2K.Note: To use the flagship models or other extended models, please select the
Pay-as-you-go API.\",\"dateModified\":\"2026-11-23T16:11:07.998Z\",\"isPartOf\":{\"@id\":\"https://platform.minimax.io/docs#website\"},\"breadcrumb\":{\"@id\":\"https://platform.minimax.io/docs/
api-reference/video-generation-v2-create#breadcrumb\"}},{\"@type\":\"BreadcrumbList\",\"@id\":\"https://platform.minimax.io/docs/api-reference/video-generation-v2-create#breadcrumb\",\
"itemListElement\":[{\"@type\":\"ListItem\",\"position\":1,\"name\":\"MiniMax API Docs\",\"item\":\"https://platform.minimax.io/docs\"},{\"@type\":\"ListItem\",\"position\":2,\"name\":\"Create Video
Generation Task\",\"item\":\"https://platform.minimax.io/docs/api-reference/video-generation-v2-create\"}]},{\"@type\":[\"TechArticle\",\"APIReference\"],\"@id\":\"https://platform.minimax.io/docs/
api-reference/video-generation-v2-create#apireference\",\"headline\":\"Create Video Generation Task\",\"name\":\"Create Video Generation Task\",\"description\":\"Change the model field to switch
between the flagship models (MiniMax H3, MiniMax H3 Highspeed, and MiniMax H3 Lite) and other extended models. This video generation V2 endpoint accepts multimodal input via the content array (text /
image / video / audio) to support text-to-video, image-to-video (first \u0026 last frame), and reference-to-video, with output up to 2K.Note: To use the flagship models or other extended models,
please select the Pay-as-you-go API.\",\"url\":\"https://platform.minimax.io/docs/api-reference/video-generation-v2-create\",\"mainEntityOfPage\":{\"@id\":\"https://platform.minimax.io/docs/
api-reference/video-generation-v2-create#webpage\"},\"dateModified\":\"2026-11-23T16:11:07.998Z\",\"publisher\":{\"@id\":\"https://platform.minimax.io/#organization\"},\"isPartOf\":{\"@id\":\"https:/
/platform.minimax.io/docs#website\"}}]}\u003c/script>","agentPointerHref":"/docs/api-reference/video-generation-v2-create.md","siteName":"MiniMax API Docs","nav":{"global":null,"tabs":
[{"tab":"Developer Guides","pages":[{"title":"Developer Guides","href":"/guides/models-intro"}]},{"tab":"API","groups":[{"group":"Using the API","pages":[{"title":"API Overview","sidebarTitle":"API
Overview","href":"/api-reference/api-overview"},{"title":"Rate Limits","href":"/guides/rate-limits"},{"title":"Error Codes","sidebarTitle":"Error Code Reference","href":"/api-reference/errorcode"}]},
{"group":"Text","icon":"text-align-start","pages":[{"title":"Anthropic SDK","sidebarTitle":"Anthropic SDK (Recommended)","href":"/api-reference/text-anthropic-api"},{"title":"OpenAI
SDK","sidebarTitle":"OpenAI SDK","href":"/api-reference/text-openai-api"},{"title":"AI SDK","sidebarTitle":"AI SDK","href":"/api-reference/text-ai-sdk"},{"group":"Anthropic Messages API","pages":
[{"title":"Create Message","sidebarTitle":"Create Message","openapi":"api-reference/text/api/openapi-chat-anthropic.json POST /anthropic/v1/messages","href":"/api-reference/text-chat-anthropic"}]},
{"group":"OpenAI Chat Completions API","pages":[{"title":"Create Chat Completion","sidebarTitle":"Create Chat Completion","openapi":"api-reference/text/api/openapi-chat-openai.json POST /v1/chat/
completions","href":"/api-reference/text-chat-openai"}]},{"group":"OpenAI Responses API","pages":[{"title":"Create Response","sidebarTitle":"Create Response","openapi":"api-reference/text/api/
openapi-responses.json POST /v1/responses","href":"/api-reference/responses-create"},{"title":"Estimate Input Tokens","sidebarTitle":"Estimate Input Tokens","description":"Estimate the input token
count of a request without invoking the model. Useful for evaluating request cost or checking context length limits before calling the main endpoint.","openapi":"api-reference/text/api/
openapi-responses.json POST /v1/responses/input_tokens","href":"/api-reference/responses-input-tokens"}]}]},{"group":"Video","icon":"video","pages":[{"group":"MiniMax-H3","pages":[{"title":"Create
Video Generation Task","sidebarTitle":"Create Video Generation Task","openapi":"api-reference/video/generation/api/v2-video-generation.json POST /v2/video_generation","href":"/api-reference/
video-generation-v2-create"},{"title":"Query Task","sidebarTitle":"Query Task","description":"Query the status and result of a single video generation, H3-Context-IR, or video regeneration task from
the last 7 days by task_id.","openapi":"api-reference/video/generation/api/v2-video-generation.json GET /v2/query/video_generation/{task_id}","href":"/api-reference/video-generation-v2-query"},
{"title":"List Tasks","sidebarTitle":"List Tasks","openapi":"api-reference/video/generation/api/v2-video-generation.json GET /v2/query/video_generation","href":"/api-reference/
video-generation-v2-list"},{"title":"Cancel or Delete Task","sidebarTitle":"Cancel or Delete Task","openapi":"api-reference/video/generation/api/v2-video-generation.json DELETE /v2/video_generation/
{task_id}","href":"/api-reference/video-generation-v2-delete"},{"title":"Create H3-Context-IR Task","sidebarTitle":"Create H3-Context-IR Task","openapi":"api-reference/video/generation/api/
v2-video-generation.json POST /v2/h3_context_ir","href":"/api-reference/video-generation-v2-h3-context-ir"},{"title":"Create Video Regeneration Task","sidebarTitle":"Create Video Regeneration
