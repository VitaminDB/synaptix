# Source: https://docs.gmicloud.ai/model-quickstarts/video/minimax-h3.md

Fetched: 2026-11-28 (raw markdown)

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.gmicloud.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# MiniMax-H3

> API usage guide for MiniMax-H3.

**Model ID**

```bash theme={null}
MiniMax-H3
```

**Calling method:** async

# MiniMax-H3 API Usage Guide

## Overview

MiniMax-H3 generates videos from text prompts and optional image, video, or audio references. It supports text-to-video, first/last-frame image-to-video, and reference-to-video generation.

## Submit a Request

```json theme={null}
{
  "model": "MiniMax-H3",
  "payload": {
    "prompt": "A cinematic view of a spacecraft launching at sunset",
    "resolution": "2K",
    "duration": 5,
    "ratio": "16:9"
  }
}
```

## Supported Inputs

* `prompt`: required text prompt, up to 7000 characters.
* `first_frame_image` and `last_frame_image`: frame image URLs.
* `reference_images`: up to 9 reference images.
* `reference_videos`: up to 3 reference videos.
* `reference_audios`: up to 3 reference audio files; audio cannot be used alone.

Frame-based inputs and reference-media inputs cannot be mixed in the same request. MiniMax performs final validation.

## Output

Completed requests return a permanent video URL and generated thumbnail in the request outcome.


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.