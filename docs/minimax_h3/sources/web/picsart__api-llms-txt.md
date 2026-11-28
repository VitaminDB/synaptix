# Source: https://picsart.com/api-platform/models/minimax-h3/llms.txt

Fetched: 2026-11-28 (raw markdown)

---

# MiniMax H3

> 2K video up to 15s from text, start and end frames, or image, video and audio references.

## Overview

- **Model ID**: `minimax-h3`
- **Workflow**: `minimax/v2/video-generation`
- **Provider**: MiniMax
- **Origin**: Partner Model
- **Category**: Text → Video (video)
- **Output**: A video URL.
- **Tags**: start frame, end frame, reference video, 2k, 15 sec
- **Unified endpoint**: `POST https://api.picsart.com/workflows/v1/models/execute` (model id in the body)
- **Compatibility endpoint**: `POST https://api.picsart.com/workflows/minimax/v2/video-generation/execute` (vendor's own parameters)
- **Execution**: long-running — use the async submit/poll pair or the SDK, which polls for you
- **Model page**: https://picsart.com/api-platform/models/minimax-h3

## Pricing

Rates are in USD. Every row below is an exact rate for the settings it names — the columns are the settings that change the price for this model. Every response reports what was actually charged in `response.usage.credits`.

These rates are charged after the request finishes, on what it actually used.

| Tier | Rate |
| --- | --- |
| Reference image | $0.04 / input reference image |
| 768p | $0.08 / sec |
| 2K | $0.13 / sec |

Published rates. Volume pricing, invoicing and annual agreements: https://picsart.com/api-platform/contact-sales

## Authentication

The SDK and the REST API authenticate with a Picsart API key sent as a bearer token. Create a key in the console at https://picsart.com/api-platform/dashboard/api-keys, keep it in a server-side environment variable, and never ship it in browser code or commit it.

```bash
export PICSART_API_KEY="YOUR_KEY"
```

```http
Authorization: Bearer $PICSART_API_KEY
```

The CLI and the MCP server do not use an API key: `gen-ai login` signs in through the browser once per machine and stores a token locally.

## Interfaces

- **SDK (TypeScript)** — `@picsart/ai-sdk` (Node 20+, ESM). Type-safe `ai.generate()` / `ai.generateText()` with per-model parameter types; polls long jobs for you.
- **SDK (Python)** — `picsart-ai-sdk` (Python 3.10+, async). The same surface in snake_case: `await ai.generate()` / `await ai.generate_text()`, with a typed input per model.
- **REST** — the API gateway at `https://api.picsart.com`. Sync `/execute`, or async `/submit` + `/{taskId}/result`.
- **CLI** — `@picsart/gen-ai` (`gen-ai`). Generate from a terminal, quote cost before spending, emit JSON for pipelines.
- **MCP** — the same models inside Claude Code, Cursor, Windsurf and other MCP hosts. Setup: https://picsart.com/gen-ai-mcp/

```bash
npm install @picsart/ai-sdk      # SDK (TypeScript)
pip install picsart-ai-sdk       # SDK (Python)
npm install -g @picsart/gen-ai   # CLI (also drives the MCP server)
```

## Unified API (recommended)

One parameter set, used by the SDK and by REST alike: the names below are what `ai.generate('minimax-h3', …)` takes, and what goes inside the inner `params` of a REST call. Over REST the model id travels in the body, not the path — one workflow, `v1/models`, fronts the whole catalogue.

### Execution modes

- **Sync** — `POST /workflows/v1/models/execute` returns the result in one call. Only for jobs that finish within ~20 seconds; otherwise it times out.
- **Async** — `POST /workflows/v1/models/submit` returns a task id; poll `GET /workflows/v1/models/{taskId}/result` until `response.status` is `COMPLETED` or `FAILED`. Task statuses: `ACCEPTED` · `IN_PROGRESS` · `COMPLETED` · `FAILED`.
- **SDK** — `ai.generate()` always awaits the finished result; polling is hidden.

Video jobs routinely outrun the sync limit: use async over REST, or the SDK, which polls for you.

### Input parameters

- **`prompt`** (`string`, _required_): What to make, in plain language: the subject, style, composition and motion. With an input image or video, what to change.
  - Max length: 7000 characters
- **`startFrame`** (`string`, _optional_): The first frame, as an image URL. The video starts on this image.
  - Accepts: image URL
- **`endFrame`** (`string`, _optional_): The last frame, as an image URL. The video ends on this image.
  - Accepts: image URL
- **`imageUrls`** (`list<string>`, _optional_): Input images, as URLs: frames to animate, or references for subject, style or composition.
  - Accepts: image URLs (max 9)
- **`videoUrls`** (`list<string>`, _optional_): Input videos, as URLs.
  - Accepts: video URLs (max 3), up to 15s
- **`audioUrls`** (`list<string>`, _optional_): Input audio files, as URLs.
  - Accepts: audio URLs (max 3), up to 15s
- **`resolution`** (`string`, _optional_): Output size. Higher costs more and takes longer.
  - Default: `"2K"`
  - Options: `768P`, `2K`
- **`duration`** (`number`, _optional_): Length of the output, in seconds.
  - Default: `5`
  - Range: 5–15 (step 1)
- **`aspectRatio`** (`string`, _optional_): Shape of the output as width:height. An auto or adaptive option follows the input image.
  - Default: `"adaptive"`
  - Options: `adaptive` (follows the input), `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`

### Input JSON Schema

The inner `params` object. Over REST it is wrapped as `{ "params": { "model": "<id>", "params": <this> } }`; the SDK takes it directly.

```json
{
  "type": "object",
  "required": [
    "prompt"
  ],
  "properties": {
    "prompt": {
      "type": "string",
      "maxLength": 7000,
      "description": "What to make, in plain language: the subject, style, composition and motion. With an input image or video, what to change."
    },
    "startFrame": {
      "type": "string",
      "format": "uri",
      "description": "The first frame, as an image URL. The video starts on this image."
    },
    "endFrame": {
      "type": "string",
      "format": "uri",
      "description": "The last frame, as an image URL. The video ends on this image."
    },
    "imageUrls": {
      "type": "array",
      "items": {
        "type": "string",
        "format": "uri",
        "description": "image URL"
      },
      "maxItems": 9,
      "description": "Input images, as URLs: frames to animate, or references for subject, style or composition."
    },
    "videoUrls": {
      "type": "array",
      "items": {
        "type": "string",
        "format": "uri",
        "description": "video URL"
      },
      "maxItems": 3,
      "description": "Input videos, as URLs."
    },
    "audioUrls": {
      "type": "array",
      "items": {
        "type": "string",
        "format": "uri",
        "description": "audio URL"
      },
      "maxItems": 3,
      "description": "Input audio files, as URLs."
    },
    "resolution": {
      "type": "string",
      "enum": [
        "768P",
        "2K"
      ],
      "default": "2K",
      "description": "Output size. Higher costs more and takes longer."
    },
    "duration": {
      "type": "number",
      "minimum": 5,
      "maximum": 15,
      "multipleOf": 1,
      "default": 5,
      "description": "Length of the output, in seconds."
    },
    "aspectRatio": {
      "type": "string",
      "enum": [
        "adaptive",
        "21:9",
        "16:9",
        "4:3",
        "1:1",
        "3:4",
        "9:16"
      ],
      "default": "adaptive",
      "description": "Shape of the output as width:height. An auto or adaptive option follows the input image."
    }
  },
  "additionalProperties": false
}
```

### Required parameters example

The required parameters plus the defaults that shape the output, as a REST body:

```json
{
  "params": {
    "model": "minimax-h3",
    "params": {
      "prompt": "A serene mountain lake at golden hour, ultra detailed",
      "resolution": "2K",
      "duration": 5,
      "aspectRatio": "adaptive"
    }
  }
}
```

### Response

The body is the gateway's status envelope: the output is at `response.result`, next to the task id, its status and the credits charged. Asset models return `url` plus an `items` array of results.

```json
{
  "status": "success",
  "response": {
    "id": "task_abc123",
    "status": "COMPLETED",
    "result": {
      "url": "https://cdn.picsart.com/…/result",
      "items": [
        {
          "url": "https://cdn.picsart.com/…/result"
        }
      ],
      "model": "minimax-h3"
    }
  }
}
```

What `ai.generate()` / `ai.generateText()` resolve to (the envelope already unwrapped):

```json
{
  "items": [
    {
      "url": "https://cdn.picsart.com/…/result"
    }
  ],
  "model": "minimax-h3"
}
```

### Usage examples

#### TypeScript (SDK)

```typescript
import { createClient } from '@picsart/ai-sdk';

const ai = createClient({
  apiKey: process.env.PICSART_API_KEY,
  apiUrl: 'https://api.picsart.com',
});

const result = await ai.generate('minimax-h3', {
  prompt: "A serene mountain lake at golden hour, ultra detailed",
  resolution: "2K",
  duration: 5,
  aspectRatio: "adaptive"
});

console.log(result.url); // video URL
```

#### Python (SDK)

```python
import asyncio
import os

from picsart_ai_sdk import create_client


async def main() -> None:
    async with create_client(
        api_key=os.environ["PICSART_API_KEY"],
        api_url="https://api.picsart.com",
    ) as ai:
        result = await ai.generate("minimax-h3", {
            "prompt": "A serene mountain lake at golden hour, ultra detailed",
            "resolution": "2K",
            "duration": 5,
            "aspectRatio": "adaptive",
        })

        print(result.url)  # video URL


asyncio.run(main())
```

#### cURL (sync)

```bash
curl -X POST 'https://api.picsart.com/workflows/v1/models/execute' \
  -H 'Authorization: Bearer $PICSART_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
  "params": {
    "model": "minimax-h3",
    "params": {
      "prompt": "A serene mountain lake at golden hour, ultra detailed",
      "resolution": "2K",
      "duration": 5,
      "aspectRatio": "adaptive"
    }
  }
}'
```

#### cURL (async: submit, then poll)

```bash
# 1. Submit the job
curl -X POST 'https://api.picsart.com/workflows/v1/models/submit' \
  -H 'Authorization: Bearer $PICSART_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
  "params": {
    "model": "minimax-h3",
    "params": {
      "prompt": "A serene mountain lake at golden hour, ultra detailed",
      "resolution": "2K",
      "duration": 5,
      "aspectRatio": "adaptive"
    }
  }
}'
# → { "status": "success", "response": { "id": "task_abc123" } }

# 2. Poll until response.status is COMPLETED
curl 'https://api.picsart.com/workflows/v1/models/task_abc123/result' \
  -H 'Authorization: Bearer $PICSART_API_KEY'
```

#### CLI

```bash
gen-ai login  # once per machine
gen-ai models info minimax-h3  # this model's parameters
gen-ai pricing minimax-h3  # quote before spending credits
gen-ai generate -m minimax-h3 \
  -p "a fox running through autumn leaves" \
  --ar adaptive \
  -d 5
```

## Compatibility API (vendor parameters)

Alongside the unified API, the `minimax/v2/video-generation` workflow takes MiniMax's own parameters exactly as the vendor defines them — nothing renamed or reshaped. It is the shortest path onto Picsart for request bodies already written against the vendor; the trade is that a request is written for one vendor, and the result comes back in the vendor's own shape. Same authentication.

- **Endpoint**: `POST https://api.picsart.com/workflows/minimax/v2/video-generation/execute` (sync) · `POST …/submit` + `GET …/{taskId}/result` (async)
- **SDK**: `ai.apis.run('minimax/v2/video-generation', params, { mode })` in TypeScript, `ai.apis.run("minimax/v2/video-generation", params, mode=…)` in Python — params forwarded verbatim

### Vendor parameters

| Parameter | Type | Required | Default | Details |
| --- | --- | --- | --- | --- |
| `model` | `string` | no | `MiniMax-H3` | Model id: MiniMax-H3 (768P/2K), MiniMax-H3-Highspeed (balanced quality and speed, 480P/768P) or MiniMax-H3-Lite (fastest and cheapest, 480P/768P). · Options: `MiniMax-H3`, `MiniMax-H3-Highspeed`, `MiniMax-H3-Lite` |
| `content` | `MinimaxContentItem[]` | yes | — | Content items. Must contain one non-empty text item. Add image/video/audio items with a role to select first-frame, first-to-last keyframe, or multimodal reference modes. Frame and reference roles are mutually exclusive. |
| └ `type` | `string` | yes | — | Media type of this content item · Options: `text`, `image_url`, `video_url`, `audio_url` |
| └ `text` | `string` | no | — | Text prompt; required for a `text` item |
| └ `image_url` | `object` | no | — | Image URL for an `image_url` item |
| └   `url` | `string` | yes | — | Publicly reachable URL, an mm_file://{file_id} handle, or a data: URI (small assets only) |
| └ `video_url` | `object` | no | — | Video URL for a `video_url` item |
| └   `url` | `string` | yes | — | Publicly reachable URL, an mm_file://{file_id} handle, or a data: URI (small assets only) |
| └ `audio_url` | `object` | no | — | Audio URL for an `audio_url` item |
| └   `url` | `string` | yes | — | Publicly reachable URL, an mm_file://{file_id} handle, or a data: URI (small assets only) |
| └ `role` | `string` | no | — | Role of a media item; omit for text · Options: `first_frame`, `last_frame`, `reference_image`, `reference_video`, `reference_audio` |
| `resolution` | `string` | no | — | Output resolution, case-insensitive. MiniMax-H3: 768P or 2K (default 2K); MiniMax-H3-Highspeed and MiniMax-H3-Lite: 480P or 768P (default 768P). · Options: `768P`, `2K`, `480P`, `768p`, `2k`, `480p` |
| `duration` | `number` | yes | `5` | Video duration in seconds (5–15) · 5–15 |
| `ratio` | `string` | no | — | Aspect ratio. Ignored (treated as adaptive) for first-/last-frame generation. Text-only generation, and MiniMax-H3-Lite with references, need a concrete ratio: adaptive or none becomes 16:9. · Options: `adaptive`, `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16` |
| `aigc_watermark` | `boolean` | no | `false` | Embed an AIGC watermark in the output |
| `callback_url` | `string` | no | — | Webhook URL to receive task state changes after challenge verification |
| `options` | `object` | no | — | GenAI options (safety checks, Picsart Drive) |
| └ `safety_checks` | `object` | no | — | Safety check settings |
| └   `enabled` | `boolean` | no | — | Whether to run content moderation. Defaults to true. |
| └ `drive` | `object` | no | — | Save result to Picsart Drive |
| └   `name` | `string` | yes | — | File name in Picsart Drive |
| └   `attributes` | `object` | no | — | Custom attributes to attach to the file |
| └   `folder` | `object` | no | — | Target folder in Picsart Drive |
| └ `inputs_transformation` | `object` | no | — | Input transformation settings |
| └   `downscale_oversized_images` | `boolean` | no | — | Whether to downscale oversized input images. Defaults to false. |
| └ `service_tier` | `string` | no | `auto` | How the generation is served. "auto" (default) uses the user's discount when one applies, with its limits. "priority" charges the regular price, with no concurrency limit and no wait. · Options: `auto`, `priority` |

### Example request body

```json
{
  "params": {
    "content": [
      {
        "type": "text"
      }
    ],
    "duration": 5
  }
}
```

### Usage examples

#### TypeScript (`ai.apis.run`)

```typescript
import { createClient, ApiRunMode } from '@picsart/ai-sdk';

const ai = createClient({
  apiKey: process.env.PICSART_API_KEY,
  apiUrl: 'https://api.picsart.com',
});

// Calls the 'minimax/v2/video-generation' workflow directly — params are sent as-is.
const { result, usage } = await ai.apis.run('minimax/v2/video-generation', {
  content: [
    {
      type: "text"
    }
  ],
  duration: 5
}, {
  mode: ApiRunMode.SYNC,
});

console.log(result); // workflow-specific output
console.log(usage?.credits); // amount charged, in $
```

#### Python (`ai.apis.run`)

```python
import asyncio
import os

from picsart_ai_sdk import ApiRunMode, create_client


async def main() -> None:
    async with create_client(
        api_key=os.environ["PICSART_API_KEY"],
        api_url="https://api.picsart.com",
    ) as ai:
        # Calls the 'minimax/v2/video-generation' workflow directly — params are sent as-is.
        response = await ai.apis.run("minimax/v2/video-generation", {
            "content": [
                {
                    "type": "text",
                },
            ],
            "duration": 5,
        }, mode=ApiRunMode.SYNC)

        print(response["result"])  # workflow-specific output
        print((response.get("usage") or {}).get("credits"))  # amount charged, in $


asyncio.run(main())
```

#### cURL (sync)

```bash
curl -X POST 'https://api.picsart.com/workflows/minimax/v2/video-generation/execute' \
  -H 'Authorization: Bearer $PICSART_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
  "params": {
    "content": [
      {
        "type": "text"
      }
    ],
    "duration": 5
  }
}'
```

#### cURL (async: submit, then poll)

```bash
# 1. Submit the job
curl -X POST 'https://api.picsart.com/workflows/minimax/v2/video-generation/submit' \
  -H 'Authorization: Bearer $PICSART_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
  "params": {
    "content": [
      {
        "type": "text"
      }
    ],
    "duration": 5
  }
}'
# → { "status": "success", "response": { "id": "wf_abc123" } }

# 2. Poll until response.status is COMPLETED
curl 'https://api.picsart.com/workflows/minimax/v2/video-generation/wf_abc123/result' \
  -H 'Authorization: Bearer $PICSART_API_KEY'
```

### Response

Over HTTP the vendor-shaped result arrives inside the gateway envelope at `response.result`:

```json
{
  "status": "success",
  "response": {
    "id": "wf_abc123",
    "status": "COMPLETED",
    "result": {
      "url": "https://cdn.picsart.com/…/result.mp4",
      "mimeType": "video/mp4",
      "duration": 0,
      "ratio": "…",
      "driveFile": {
        "id": "wf_abc123"
      }
    }
  }
}
```

`ai.apis.run()` unwraps that envelope and resolves to `{ result, usage }`:

```json
{
  "result": {
    "url": "https://cdn.picsart.com/…/result.mp4",
    "mimeType": "video/mp4",
    "duration": 0,
    "ratio": "…",
    "driveFile": {
      "id": "wf_abc123"
    }
  }
}
```

## Additional resources

- Model page (playground): https://picsart.com/api-platform/models/minimax-h3
- SDK and REST API (unified): https://picsart.com/api-platform/models/minimax-h3#api
- Compatibility API (vendor parameters): https://picsart.com/api-platform/models/minimax-h3#advanced
- MCP (use it from Claude, ChatGPT, Cursor and other assistants): https://picsart.com/api-platform/models/minimax-h3#mcp
- CLI (run it from a terminal): https://picsart.com/api-platform/models/minimax-h3#cli
- Pricing: https://picsart.com/api-platform/models/minimax-h3#pricing
- This document: https://picsart.com/api-platform/models/minimax-h3/llms.txt
- All models: https://picsart.com/api-platform/models
- Docs — quickstart https://picsart.com/api-platform/docs/quickstart, authentication https://picsart.com/api-platform/docs/authentication, SDK https://picsart.com/api-platform/docs/sdk, REST https://picsart.com/api-platform/docs/api-reference, CLI https://picsart.com/api-platform/docs/cli, MCP https://picsart.com/api-platform/docs/mcp
- CLI reference: https://picsart.github.io/picsart-mcp-cli-docs/guide/introduction
