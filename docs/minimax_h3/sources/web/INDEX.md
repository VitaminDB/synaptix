# MiniMax H3 — open-web sources (fetched 2026-11-28)

Raw material from the open web (no GitHub / HF model cards — other collectors cover those).
Most files are `curl` + `w3m -dump` text of the page with nav/footer trimmed (body is verbatim; tables flattened to text).
`docs.comfy.org__*`, `apiyi__*`, `gmicloud__*`, `picsart__*` are the sites' own raw Markdown/llms.txt.
`civitai__*` model pages come from the public Civitai API (`/api/v1/models/<id>`), description HTML converted to text.
Caveat: many vendor blogs (Atlas Cloud, WeShop, JXP, Kapwing, DomoAI, Seedance, Dreamina, PixVerse, Krea) are SEO/marketing; treat numbers as unverified.

## Official MiniMax
| File | URL | Note |
|---|---|---|
| minimax.io__blog-h3-announcement.md | https://minimax.io/blog/minimax-h3 | Official launch blog: omni-modal, 2K, ≤15 s, native stereo audio, multi-shot |
| minimax.io__news-h3-open-source.md | https://www.minimax.io/news/minimax-h3-open-source | Official open-source announcement: what is released (FL2VA/Ref2VA, Image), what is held back, license |
| platform.minimax.io__api-video-generation-v2.md | https://platform.minimax.io/docs/api-reference/video-generation-v2-create | Official API v2 `/v2/video_generation`: models (H3, Highspeed, Lite, Max), content items, resolutions, durations |

## Prompt guides (official structure, applied)
| File | URL | Note |
|---|---|---|
| docs.comfy.org__prompt-guide.md | https://docs.comfy.org/tutorials/video/minimax/minimax-h3-prompt-guide.md | **Best compact summary**: 3 fields, shots/cuts, `<d>` dialogue, speaker IDs, R2V audio markers, cfg 1 / no negatives, 1344×768, length 17k+5, 10 style embeddings |
| comfyui-wiki__prompt-writing-five-modes.md | https://comfyui-wiki.com/en/tutorial/advanced/video/minimax/minimax-h3-prompt-writing | **Full official structure** for T2VA/I2VA/FL2VA/L2VA/Ref2VA: alignment lines, camera vocab (type+amplitude+speed), 6-section Ref2VA, retention markers, task types, full example |
| atlascloud__prompt-guide-45-official.md | https://www.atlascloud.ai/blog/guides/minimax-h3-prompt-guide | Analysis of all 45 official example prompts; six-block structure, end-image technique, audio cues |
| fal.ai__prompting-guide-44-examples.md | https://fal.ai/learn/devs/minimax-h3-prompting-guide | Long guide with 44 example videos + exact prompts across T2V/I2V/Ref |
| civitai__article-34646.md | https://civitai.com/articles/34646 | "Production plan" prompt structure + 5 copy-ready T2V/I2V/Ref2V prompts |
| kapwing__how-to-prompt.md | https://www.kapwing.com/resources/how-to-prompt-minimax-h3-hailuo-3-0-a-guide-for-ai-video-creators/ | Creator guide, structured order, examples ("Hailuo 3.0" alias) |
| jxp__prompt-guide.md | https://www.jxp.com/minimax/blog/minimax-h3-prompt-guide | Formula subject/action/env/camera/timing/style/audio; camera language as path |
| hackernoon__prompt-guide.md | https://hackernoon.com/minimax-h3-prompt-guide-how-to-write-better-hailuo-30-video-prompts | Prompt guide; one-variable iteration by failure category |
| weshop__prompt-formula-mistakes.md | https://www.weshop.ai/solutions/models/mini-max-h3-prompt-guide-formula-storyboards-templates-and-common-mistakes | Formula, storyboard templates, common mistakes |
| weshop__dialogue-and-timing.md | https://www.weshop.ai/solutions/models/mini-max-h3-prompt-guide-for-dialogue-and-timing | Dialogue/speaker attribution/cut timing; A/B: vague prompt → male voice on female lips |
| wavespeed__how-to-write-prompts.md | https://wavespeed.ai/blog/minimax-h3/how-to-write-minimax-h3-prompts/ | Short: subject/action/camera/audio brief, one idea per clip |
| lumalabs__complete-guide.md | https://lumalabs.ai/learning-center/articles/minimax-h3-complete-guide | Complete guide: multi-shot timelines, dialogue as performance (the two Luma "prompt guide" URLs returned 404) |
| scenario__h3-essentials.md | https://help.scenario.com/articles/1593651655-minimax-h3-the-essentials | Platform essentials; put spoken lines in prompt for lip-sync; modes/params |
| morphic__complete-guide.md | https://morphic.com/resources/how-to/minimax-h3-guide | References/editing/audio guide; limits (12 files, refs ≤15 s total) |
| krea__h3-model-page.md | (krea blog URLs) | Both Krea blog URLs redirected to the Krea H3 model page (specs, few prompts) |

## Reference-to-video / identity
| File | URL | Note |
|---|---|---|
| domoai__reference-to-video.md | https://domoai.app/blog/minimax-h3-reference-to-video | Assign each reference a role (image=look, video=motion/camera, audio=voice); FL2VA vs Ref2VA choice |
| domoai__character-consistency.md | https://domoai.app/blog/minimax-h3-character-consistency-guide | Clean reference pack, locked character description, change one shot variable at a time |
| wavespeed__character-consistency.md | https://wavespeed.ai/blog/minimax-h3/minimax-h3-character-consistency/ | Short character-consistency tips |
| runware__reference-driven-consistency.md | https://runware.ai/docs/models/minimax-h3/guides/reference-driven-consistency | Provider guide on reference-driven consistency (with request examples) |
| runware__guides-index.md | https://runware.ai/docs/models/minimax-h3/guides | Runware H3 guides index/overview with parameters |

## Local running, speed, VRAM (practitioner write-ups)
| File | URL | Note |
|---|---|---|
| ai-muninn__minimax-h3-nvfp4-rtx5090.md | https://ai-muninn.com/en/blog/minimax-h3-nvfp4-rtx5090 | Part 1: pruned+quantized 4 files 31.7 GB on one 5090; 175 s for 864×480 10 s |
| ai-muninn__minimax-h3-rtx5090-speedup-vsr.md | https://ai-muninn.com/en/blog/minimax-h3-rtx5090-speedup-vsr | Part 2: 14 steps + SageAttention + RTX VSR halves time |
| ai-muninn__minimax-h3-rtx5090-character-lock.md | https://ai-muninn.com/en/blog/minimax-h3-rtx5090-character-lock | Part 3: locking a character with references; "name the face, don't describe it"; blur upscaling can't fix |
| ai-muninn__minimax-h3-effect-embeddings-rtx5090.md | https://ai-muninn.com/en/blog/minimax-h3-effect-embeddings-rtx5090 | Part 4: test of the 10 community effect embeddings |
| ai-muninn__minimax-h3-rtx5090-config-stack.md | https://ai-muninn.com/en/blog/minimax-h3-rtx5090-config-stack | **Accounting of 6 speed levers**: KJNodes LowVRAMAttention+ChunkFF (47 GiB OOM → fits), chunk 16/8, torch cu130 NVFP4 kernel (−32..45%), Sage/Sol 2–3%, TRT VAE 1.7% |
| ai-muninn__ref2va-two-characters-rtx5090.md | https://ai-muninn.com/en/blog/minimax-h3-ref2va-two-characters | Part 6: Ref2VA with two characters; needs audio_vae; which-robe-on-whom binding |
| ai-muninn__ltx25-vs-h3-rtx5090.md | https://ai-muninn.com/en/blog/ltx25-vs-minimax-h3-rtx5090 | Part 7: LTX-2.5 vs H3 same Chinese dialogue clip on 5090 (≈29 s vs 81–95 s) |
| ai-muninn__storyboard-grid-h3-ref2va-rtx5090.md | https://ai-muninn.com/en/blog/storyboard-grid-h3-ref2va-rtx5090 | 4×4 storyboard into Ref2VA: one sentence 3/16 panels hit vs official-format shot timeline 13/16 |
| ai-muninn__pika-h3-flex-rtx5090-whitebox-orbit.md | https://ai-muninn.com/en/blog/pika-h3-flex-rtx5090-whitebox-orbit | Pika-H3-Flex turbo 20.8 s vs 38 s per 5 s clip; text-only orbits drift, Blender blockout 6/6 |
| ai-muninn__minimax-h3-on-modded-2080ti-22gb.md | https://ai-muninn.com/en/blog/minimax-h3-on-modded-2080ti-22gb | 2080 Ti 22G: 15 s 1080p in 23 min; INT8 DiT ok, W4A4 DiT tears frame, W4A4 TE fine |
| dev.to__vram-6gb-to-20gb.md | https://dev.to/tf6cool/i-compared-minimax-h3-runs-from-6gb-to-20gb-vram-heres-what-i-learned-2jpj | Why "minimum VRAM" claims conflict; compare precision/res/frames/steps |
| hackernews__h3-thread.md | https://hn.svelte.dev/item/49252179 | HN thread (mirror): Mac speeds (1–1.5 h per clip), 5090 ~2 min, GGUF vs int8_convrot, MPS lacks `_int_mm`, sparse-attention AMA |
| prismix__reddit-localllama-mirror-1.md | https://prismix.dev/news/0c187cf8f98d | Mirror of r/LocalLLaMA post: first-week impressions, audio-driven mode, 9 img + 3 vid + 3 audio refs |
| prismix__reddit-localllama-mirror-2.md | https://prismix.dev/news/c211905a6acc | Mirror of r/LocalLLaMA launch post (mostly announcement text) |
| simonwillison__minimax-tag.md | https://simonwillison.net/tags/minimax/ | Simon Willison: running H3 FL2VA on Mac via MLX 8-bit port (commands) |

## ComfyUI docs and blog
| File | URL | Note |
|---|---|---|
| docs.comfy.org__h3-local-tutorial.md | https://docs.comfy.org/tutorials/video/minimax/minimax-h3.md | Overview: local ≈1 MP/768 short edge 24 fps ≤15 s, 2K only hosted; 5 templates; version reqs |
| docs.comfy.org__h3-native-workflows.md | https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native.md | T2V/I2V/R2V templates: model files, node settings, R2V prompt notes |
| docs.comfy.org__h3-multiframe-reference.md | https://docs.comfy.org/tutorials/video/minimax/minimax-h3-multiframe.md | Anchoring frames along the timeline with chained Add Guide nodes |
| docs.comfy.org__h3-fun-controlnet.md | https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fun-controlnet.md | Fun ControlNet Union (canny/depth/HED/MLSD/pose, inpaint), sparse attention backend |
| docs.comfy.org__h3-fastvideo.md | https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fastvideo.md | FastH3: DMD2-distilled 8-step checkpoint (FastVideo) |
| docs.comfy.org__partner-nodes-api.md | https://docs.comfy.org/tutorials/partner-nodes/minimax/minimax-h3.md | Hosted (API) H3 partner nodes: params, 2K, modes |
| blog.comfy.org__day0-support.md | https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui | Day-0 ComfyUI support announcement, local optimizations |
| blog.comfy.org__sync-sound-challenge.md | https://blog.comfy.org/p/comfy-h3-sync-sound-community-challenge | Community "Sync Sound" challenge rules |
| blog.comfy.org__sync-sound-challenge-results.md | https://blog.comfy.org/p/comfy-h3-sync-sound-challenge-the | Challenge results; entrant workflows, render times (2 h 35 min on 5090 for one entry) |

## Turbo / distillation LoRAs and community models
| File | URL | Note |
|---|---|---|
| comfyui-wiki__turbo-lora-4step-news.md | https://comfyui-wiki.com/en/news/2026-08-07-minimax-h3-turbo-lightx2v | lightx2v/ModelTC FL2V Turbo 4-step v0.1; Kijai: strength 0.75, 4 steps, er_sde/sa_solver; ComfyUI audio bug fixed Aug 8 |
| comfyui-wiki__turbo-lora-v1-8step-news.md | https://comfyui-wiki.com/en/news/2026-08-11-minimax-h3-turbo-lightx2v-v1 | Turbo v1.0: 8-step (shifts 12/3, 8 or 4 steps) and 4-step 768p (shifts 6/3) |
| mindstudio__turbo-lora-comfyui-local.md | https://www.mindstudio.ai/blog/minimax-h3-turbo-lora-comfyui-local | Turbo LoRA local: 20 → ~6 steps, VRAM peaked ~50 GB in their run |
| instasd__turbo-lora-4-steps.md | https://www.instasd.com/post/minimax-h3-turbo-lora-comfyui-4-steps | Turbo 4-step setup; `low_vram` (merge) switch on LoRA node; 10 steps without LoRA alternative |
| civitai__turbo-loras-collection.md | https://civitai.com/models/2837571 | **Collection of all turbo LoRAs** (larryvrh v4, lightx2v, DMAD, taomate, pika-h3-flex) with best settings per file |
| civitai__top-h3-models-by-downloads.md | https://civitai.com/api/v1/models?baseModels=MiniMax%20H3 | Top 59 SFW H3 models/LoRAs/workflows on Civitai by downloads, with descriptions |
| stablediffusiontutorials__19-lora-models.md | https://www.stablediffusiontutorials.com/2026/08/minimax-h3-lora-models.html | Roundup of 19 H3 LoRAs (acc, style, concept) with what they do |
| civitai__third-person-view-lora.md | https://civitai.com/models/2945172 | Concept LoRA (third-person/HUD camera) with usage notes |
| civitai__grimdark-3d-style-lora.md | https://civitai.com/models/2918749 | Example style LoRA page with settings |
| civitai__workflow-audio-pseudo-chinese-fix.md | https://civitai.com/models/2863848 | Workflow fixing "input audio regenerated into pseudo-Chinese vocals": pass original audio through; Ref2VA+Turbo ~180–200 s/15 s on 4090/5090 |
| comfyui-wiki__endless-h3-long-lipsync.md | https://comfyui-wiki.com/en/news/2026-09-03-endless-minimax-h3 | Endless MiniMax H3: chained Motion Context clips, latent stitching, 1:30 lip-sync on 3060 12 GB |
| civitai__article-35316.md | https://civitai.com/articles/35316 | Civitai H3 LoRA training contest; 670+ H3 models in first weeks; on-site trainer |
| fal.ai__how-to-train-lora.md | https://fal.ai/learn/devs/how-to-train-a-lora-for-minimax-h3 | LoRA training guide (dataset, captions, steps) |
| civitai-developer__orchestration-recipe.md | https://developer.civitai.com/orchestration/recipes/minimax-h3 | Civitai orchestration API recipe for H3 (params) |

## Audio / dialogue problems
| File | URL | Note |
|---|---|---|
| seedance.tv__gibberish-fix.md | https://www.seedance.tv/blog/minimax-h3-gibberish-fix | Gibberish vs desync vs looping; 6 fixes; 4 minimal prompt templates (competitor site) |
| atlascloud__chinese-dialogue-accuracy.md | https://www.atlascloud.ai/blog/tips/minimax-h3-chinese-dialogue-accuracy | Vendor test: ~83% Chinese dialogue accuracy; failure modes |
| atlascloud__lip-sync-and-audio.md | https://www.atlascloud.ai/blog/minimax-h3-lip-sync-and-audio | Lip-sync/audio tests: non-human faces, lip-sync loss after 2K upscale |

## News, reviews, comparisons
| File | URL | Note |
|---|---|---|
| deeplearning.ai__the-batch-minimally-open.md | https://www.deeplearning.ai/the-batch/minimaxs-state-of-the-art-video-model-is-only-minimally-open | The Batch: license restrictions, what is open |
| nyu-rits__h3-release-open-weights-promised.md | https://rits.shanghai.nyu.edu/ai/minimax-releases-h3-2k-video-with-native-audio-open-weights-promised/ | Launch report, pricing ($0.13/s 2K, $0.08/s 768p) |
| nyu-rits__h3-weights-us-eu-excluded.md | https://rits.shanghai.nyu.edu/ai/minimax-ships-h3-weights-with-the-us-and-eu-excluded/ | Weights release Aug 3; territory exclusions in license |
| eweek__h3-2k-open-weights.md | https://www.eweek.com/news/minimax-h3-2k-ai-video-open-weights/ | News: 2K, stereo, proprietary parts |
| the-decoder__tops-video-ranking.md | https://the-decoder.com/chinas-minimax-h3-is-the-first-open-model-to-top-an-ai-video-ranking/ | Artificial Analysis ranking (1st video editing, 2nd T2V) |
| programminginsider__33b-open-weight.md | https://programminginsider.com/minimax-h3-open-weight-release-gives-developers-a-33-billion-parameter-video-model-with-full-api-access/ | 33B open-weight release overview |
| programminginsider__h3-wan-ltx-comparison.md | https://programminginsider.com/creator-marketer-or-developer-heres-the-ai-video-model-built-for-you/ | H3 vs Wan vs LTX positioning (local 768p tier) |
| atlascloud__open-source-weights.md | https://www.atlascloud.ai/blog/guides/minimax-h3-open-source-weights | What the open weights include, hardware, license |
| atlascloud__h3-vs-ltx23.md | https://www.atlascloud.ai/blog/tips/minimax-h3-vs-ltx-2.3 | Same-prompt H3 vs LTX 2.3 tests |
| wavespeed__h3-vs-ltx.md | https://wavespeed.ai/blog/minimax-h3/minimax-h3-vs-ltx-video/ | Short H3 vs LTX comparison |
| dreamina__h3-review.md | https://dreamina.capcut.com/ai-model/minimax-h3-review | Review: limits (competitor) |
| pixverse__h3-review.md | https://pixverse.ai/en/blog/minimax-h3-review | Review: specs, demos, limits (competitor) |

## MiniMax-H3-Image
| File | URL | Note |
|---|---|---|
| mindstudio__h3-image-benchmark.md | https://www.mindstudio.ai/blog/minimax-h3-image-benchmark-test/ | Hands-on tests: text, counting, layout, edit identity, realism |
| hackernoon__h3-image-high-res.md | https://hackernoon.com/how-minimax-h3-image-handles-high-resolution-ai-image-generation | ~16 MP target, sizes 4096², 5472×3072 |
| airmore__h3-image-review.md | https://airmore.ai/ai-review/minimax-h3-image-review | 4K, 16-reference editing, VRAM, license limits |
| segmind__h3-text-to-image.md | https://www.segmind.com/models/minimax-h3-text-to-image | Settings (aspect 1:3–3:1, prompt enhancement), examples |

## Third-party API docs
| File | URL | Note |
|---|---|---|
| apiyi__api-overview.md | https://docs.apiyi.com/api-capabilities/minimax-h3/overview.md | API proxy overview: modes, params |
| gmicloud__api-quickstart.md | https://docs.gmicloud.ai/model-quickstarts/video/minimax-h3.md | Short quickstart |
| picsart__api-llms-txt.md | https://picsart.com/api-platform/models/minimax-h3/llms.txt | Picsart API H3 params (llms.txt) |

## Failed / blocked / not saved
- reddit.com (r/StableDiffusion, r/comfyui, r/LocalLLaMA): blocked for both WebSearch (domain not accessible) and curl (bot wall). Only indirect: prismix.dev mirrors, HN thread, comfyui-wiki news citing Reddit posts.
- lumalabs.ai `/minimax-h3-prompt-guide` and `/minimax-h3-prompting-guide`: HTTP 404 (complete-guide saved instead).
- tensor.art/articles/1035804035619278845: HTTP 403.
- scmp.com article on H3 pricing/open weights: HTTP 403.
- dev.to/tekmag H3 overview: HTTP 404.
- krea.ai blog posts: redirect to generic model page (saved once as krea__h3-model-page.md).
- picsart.com/blog/?p=264378 (blog post): not fetched (only API llms.txt saved).
