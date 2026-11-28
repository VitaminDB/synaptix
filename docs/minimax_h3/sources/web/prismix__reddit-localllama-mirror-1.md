# Source: https://prismix.dev/news/0c187cf8f98d

Fetched: 2026-11-28 (curl + w3m text dump)

---

Home Status News MCP Pricing Sign in (Toggle theme)
(Toggle theme) (Toggle menu)
Home Status News MCP Pricing Sign in
News / r/LocalLLaMA
● r/LocalLLaMA · August 9, 2026 · 2 min read

Open-weight video gen that actually delivers. Five days with MiniMax H3 on local hardware.

#video-gen #open-source #music

Mirrored from r/LocalLLaMA for archival readability. Support the source by reading on the original site.

Like Read original ↗

H3 weights went live on HuggingFace August 3rd and I started pulling them immediately. An omni-modal video model with native stereo audio in the same forward pass, where audio can actually drive the
video generation? On open weights? I had to try it.

Five days in, the quality is legit. 2K at 24fps, 5 to 15 second clips, text and image and video and audio all in one shared context. You can throw 9 reference images, 3 videos, 3 audio clips at it
per generation. The audio-driven mode where you feed it a track and it generates video that moves to the sound is genuinely unlike anything I've run from open weights before. But here's the catch:
running this locally on a consumer GPU means your iteration speed is painful. Every failed generation costs you real time and power, and you will need retakes. Motion consistency is much better than
I expected, but complex camera moves and longer clips still need multiple attempts to get right.

What actually saved my workflow is that APOB AI is self-hosting MiniMax H3 unlimited and free right now since the weights are open. I stopped grinding my local card for every test run. I prototype on
their hosted H3, figure out which prompts and references produce clean results, then switch back to my own hardware when I want full control or need to run something custom.

For comparison, Seedance 2.5 from ByteDance also launched July 31st. Completely different approach: up to 30 seconds in a single pass at 4K with native audio, 50 multimodal references, and
region-level editing to fix part of a shot without regenerating the whole clip. But API-only through BytePlus ModelArk, no published weights. If you actually want to run the model yourself, H3 is the
one. Having open-weight video gen at this quality level sitting on HuggingFace is a real milestone for local inference.

 submitted by  /u/Binary_orchid
[link]  [comments]
Continue reading on r/LocalLLaMA ↗
Share X / Twitter LinkedIn Reddit Hacker News Copy link

Copyright belongs to r/LocalLLaMA. Prismix mirrors public-feed content for archival + accessibility purposes and credits + links to the original publisher on every page.

Discussion (0)

Sign in to join the discussion. Free account, 30 seconds — email code or GitHub.

Sign in →

No comments yet. Sign in and be the first to say something.

More from r/LocalLLaMA

  • The AI race: China's open-weight models trail US closed models by just 7 months

    Oct 14

  • Anthropic Executive Says Open-Source AI Models Are Higher Risk

    Oct 14

  • ComfyUI v0.40.0 released

    Oct 14

  • So this is how they do it --> RTX 5090, once a consumer gaming card, is now on smugglers’ menu alongside cigarettes

    Oct 14

Product
