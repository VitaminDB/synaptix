# D3voz__openjutsu

Source: https://github.com/D3voz/openjutsu
Fetched: 2026-11-28

Description: One-click character swap for videos: an open-source, local Higgsfield Genjutsu alternative built on ComfyUI and MiniMax H3, with FLUX.3 Klein relighting.
Stars: 2697  Updated: 2026-10-30T07:49:51Z



---

## FILE: README.md (https://github.com/D3voz/openjutsu/blob/HEAD/README.md)

# Openjutsu

An open-source, Genjutsu-style character and object swap for video. Drop a video and a character photo, press **Generate**, and the main person in the video is replaced by your character. Or change their hair or clothes, remove something, or put them somewhere else. The camera, cuts, motion and soundtrack stay. Everything runs on open weights on your own GPU: MiniMax H3 for the video, SAM 3.1 to track people, and Qwen3-VL 8B to describe the pictures and shots.

The app is a small Next.js front end over a ComfyUI workflow. It uses [`@comfyorg/sdk`](https://www.npmjs.com/package/@comfyorg/sdk), following Comfy's [comfy-reshoot](https://github.com/Comfy-Org/comfy-reshoot) reference app.

https://github.com/user-attachments/assets/2d8d763a-1764-49a6-9116-d6d1be661ff6

| Advanced: several people, pick who becomes who | Object swap: paint where it goes |
| --- | --- |
| ![Two people swapped, each from their own photo](docs/media/advanced_swap.jpg) | ![Sunglasses painted onto a person's eyes](docs/media/object_swap.jpg) |
| **From picture: a new look and place from one picture** | **Post-processing: fix faces** |
| ![A video frame and a picture combined into a posed picture, then a video](docs/media/from_picture.jpg) | ![A small face redrawn at close-up size](docs/media/fix_faces.jpg) |

**From picture → Relight: new light in the same pass**

![A video frame and a picture combined into a posed picture under golden backlight, then a video that keeps the light, with the seventeen light looks below](docs/media/relight.jpg)

## First visit: visual describer

The welcome page lets you choose the vision-language model that describes pictures and video frames and writes prompts. Qwen3-VL 8B INT8 remains the default; Qwen3-VL 4B FP8 and Gemma 4 E2B INT8 are smaller alternatives. Install a compatible single-file model including its vision weights in `ComfyUI/models/text_encoders`, then press **Refresh models**. Other installed models can be selected if they support image/video input and text generation through ComfyUI's CLIPLoader and Generate Text nodes. Text-only models and files requiring a different loader are unsuitable.

Choose according to available VRAM and leave room for image/video context and working memory; file size is not a VRAM requirement, and H3 generation has separate memory needs. Your selection is saved in this browser and used for swaps, prompt descriptions and **Find people**. Reopen **Welcome & models** to change it, or use **Visual describer** under Model components. Alternative describers may change prompt quality and person matching. **Set up later** opens the editor; generation still checks the required files. The separate H3 text encoder remains unchanged.

## Modes

- **Simple motion transfer**: available. One character replaces the main person (the one who fills the most of the frame). Runs `workflows/simple.api.json`, exported from `H3 One-Click Single Swap (DMAD 6 step).json`.
- **Advanced motion transfer**: available. 1–4 characters, each with an optional role; every tracked person is matched to a role per shot, and people without a character stay untouched. Runs `workflows/advanced.api.json`, exported from `H3 One-Click Advanced Swap (DMAD 6 step).json`; **One pass per person** (two characters) runs `workflows/advanced_maxq.api.json`.
- **Object swap**: available. **Change** what SAM tracks (hair, a jacket, a car) into what you describe or show in a picture; **Remove** it; or change the **Background** around it (`person` stays). Only that area is regenerated. Runs `workflows/object.api.json`, exported from `H3 One-Click Object Swap (DMAD 6 step).json`.
- **From picture** (under Object swap): FLUX prepares the character in the raw video frame's pose and body cut while keeping the character picture's background. Review or regenerate the image, then choose **Use this image** to set the video length, video and motion-reference resolutions, models, LoRAs, steps and seed. SolarRenderer defaults to four steps with a selectable rank-384 or rank-64 four-step/control pair and supports up to 15 seconds, like the native H3 renderer. SolarRenderer clips longer than six seconds can take considerably longer to render, especially at higher resolutions; see the workflow requirements for the extended-silence node fix. Both LoRA file pickers support `models/loras` and its subfolders. **Use this image** also has the visual describer watch the part and write what happens (actions, held objects, expressions), then rewrite it for the prepared picture; check and edit that video description before rendering. **Render video** reuses the saved preview with normal-map motion guidance, the video description and the source soundtrack, in one clip without a context loop. H3 rounds the requested length down to its valid frame grid. **Relight** (off by default) changes the light in the same FLUX pass: pick one of seventeen looks, each shown with an example picture, or describe your own light. FLUX then sees only your picture, and the Qwen 3.5 9B prompt upsampler reads the pose from the video frame. Magenta club and Teal lasers mostly recolour the background. **Relight the video's own person** skips the character picture: FLUX relights the video frame itself, keeping its person, pose and setting. See [the workflow requirements](workflows/picture/README.md).
- **Post-processing**: available, for a finished video. **Fix faces**: H3 draws faces poorly when the person is small in the frame, so this redraws the face of the person in a photo at close-up size and pastes it back. Runs `workflows/face.api.json`, written by `workflows/build/openjutsu_face_build.py`. **Fix colours** softens H3's harsh look without an H3 run: Image Detail Contrast takes some fine detail contrast out, then Orion4D's DCTL Tone Mapper grades every frame. Runs `workflows/color.api.json`.

## Requirements

- An NVIDIA GPU. Tested on an RTX 4060 Ti 16 GB with 64 GB of system RAM; the H3 int8 models stream from RAM.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 0.41.2 or newer. Core provides MiniMax H3, SAM 3.1 and TextGenerate.
- Node.js 22.6 or newer, and Python for `comfy-api-proxy`.

### Custom nodes

Install these into `ComfyUI/custom_nodes` (ComfyUI Manager finds the first three by name):

| Pack | Used for |
| --- | --- |
| [SolarRenderer](https://github.com/Enderfga/SolarRenderer/tree/main/comfyui) | From picture: SolarRenderer LoRA loading, conditioning and four-step sampling |
| [ComfyUI-MiniMaxH3-Context-Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop) 0.7.3+ | clip loop, checkpoints, final assembly |
| [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder) | masked video-edit latent |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Grow Mask With Blur; From picture: VRAM cleanup |
| [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | From picture: Metric3D normal maps |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | video loading |
| [comfyui-h3-oneclick-swap](https://github.com/D3voz/comfyui-h3-oneclick-swap) | swap and object plans, people and role masks, grey mannequins, track labelling, cut finder, shot list, face-fix chunks |
| [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine) | Fix faces: face tracking and cropping, SAM face masks, pasting back |
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | Fix faces: SAM loader |
| [comfyui-image-detail-contrast](https://github.com/D3voz/comfyui-image-detail-contrast) | Fix colours: Image Detail Contrast |
| [Orion4D_FXMax](https://github.com/orion4d/Orion4D_FXMax) | Fix colours: DCTL Tone Mapper and its presets |
| [comfyui-websocket-reconnect-fix](https://github.com/D3voz/comfyui-websocket-reconnect-fix) | every job: keeps live progress coming after the proxy reconnects to ComfyUI (adds no nodes) |

Restart ComfyUI after installing them.

Before uploading a job, Openjutsu checks the selected workflow against ComfyUI's node and model lists. Missing packs appear with installation links; missing model files appear with their destination folders. Disabled steps are excluded from the check. This checks registered nodes and listed model files, not Python dependencies, model integrity or available VRAM. `COMFYUI_URL` must point at ComfyUI itself (normally `http://127.0.0.1:8188`).

### Models

For **From picture → SolarRenderer**, download [solarrenderer_4step_lora.safetensors](https://huggingface.co/Enderfga/SolarRenderer/blob/main/solarrenderer_4step_lora.safetensors) and [solarrenderer_control_lora.safetensors](https://huggingface.co/Enderfga/SolarRenderer/blob/main/solarrenderer_control_lora.safetensors) from [Enderfga/SolarRenderer](https://huggingface.co/Enderfga/SolarRenderer), or the corresponding rank-64 pair from [mdevoz/SolarRendererRank64](https://huggingface.co/mdevoz/SolarRendererRank64). Put both in `ComfyUI/models/loras` or a subfolder, reload the app and choose their paths in the render settings.

| Folder in `ComfyUI/models` | File |
| --- | --- |
| `diffusion_models` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| `loras` | `minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors` |
| `text_encoders` | `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` (H3 text encoder) |
| `text_encoders` | `qwen3vl_8b_int8_convrot.safetensors` (describer) |
| `vae` | `minimax_h3_video_vae_int8_convrot.safetensors`, `minimax_h3_audio_vae_fp32.safetensors` |
| `checkpoints` | `sam3.1_multiplex_fp16.safetensors` |
| `ultralytics/bbox`, `ultralytics/segm` | `face_yolov8m.pt`, `person_yolov8m-seg.pt` ([Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer)), for Fix faces |
| `sams` | `sam_vit_b_01ec64.pth`, for Fix faces |
| `insightface/models` | `buffalo_l` (InsightFace downloads it on first use), for Fix faces |

## Run it locally

On Windows, double-click `start_openjutsu.bat` (ComfyUI must be running, or started while it waits). The first run installs `comfy-api-proxy` 0.1.6 into `.venv-proxy` and the node packages, and `patch_proxy.py` fixes the proxy's live progress (node changes and mid-job previews). Each run builds the app, then opens the proxy and the app (`npm start`, the production server) in two windows and the page in your browser. Close those two windows to stop Openjutsu. The manual steps:

1. Start ComfyUI on `127.0.0.1:8188`. On Windows, set `AIOHTTP_NOSENDFILE=1` first (PowerShell: `$env:AIOHTTP_NOSENDFILE = "1"`). Without it, the first partial-file request (the video player sends one when you seek) can stall ComfyUI's `/view`, and every later file download hangs until ComfyUI restarts.
2. Install the proxy, fix its live progress with this repo's `patch_proxy.py` (run it with the same Python), and start it. It serves the Comfy API v2 on `127.0.0.1:8189`. Raise the upload cap for longer videos (the default is 100 MB):

   ```sh
   pip install comfy-api-proxy==0.1.6
   python openjutsu/patch_proxy.py
   comfy-api-proxy run --max-upload-mb 500
   ```

3. Install and start the app:

   ```sh
   git clone <this repo> openjutsu
   cd openjutsu
   cp .env.example .env.local
   npm ci
   npm run build
   npm start
   ```

   While changing the code, `npm run dev` runs the development server instead.

4. Open <http://127.0.0.1:3000>.

`.env.local` sets `COMFY_BASE_URL=http://127.0.0.1:8189`. Leave `COMFY_API_KEY` empty for a local proxy. The base model and LoRA pickers read ComfyUI's model lists from `COMFYUI_URL` (default `http://127.0.0.1:8188`); without it they become text fields.

Uploads are limited to 512 MB total per request, including reference pictures. Uploads are streamed to a temporary folder, and up to three can arrive at once per app process; a fourth is asked to retry shortly. After a job ends, its uploads are removed from comfy-api-proxy, but the files stay in ComfyUI's `input/` folder. Generation continues in ComfyUI independently.

A 6 s clip at 480p takes about 6–13 minutes on a 4060 Ti, most of it in H3 sampling. 736p is the default; 1080p is about 2.5× slower than 736p and needs much more memory. Custom sets an exact width × height from 256 to 2048 px a side (rounded to 32); the video is scaled and centre-cropped to that shape first. It needs the `short_edge` input of the H3 Swap Plan nodes in comfyui-h3-oneclick-swap.

Advanced motion transfer has a **Montage** tab for edited clips where different people take turns within one shot without a hard cut (whip pans, selfie handovers, travel montages). Write roles as "who - looks" ("the dancer - white polo, jeans"): the looks help match the people, and only the part before " - " reaches H3. **Generate** tracks everyone with SAM over the whole video, matches each appearance to the roles with Qwen (a person SAM loses and finds again is matched again, so they keep their photo; Qwen sees each appearance in its frame and close up at three moments, a few appearances at a time, and is asked again about any it skips or matches to one photo twice at the same moment) and makes the video as one chain like Standard, with overlapping clips: each clip has one H3 pass per photo on screen in it (the two with the most screen time) that masks only that photo's appearances, with a prompt about that person only. Appearances on screen for under 12 frames or at under 2% of the frame stay as they are, since H3 draws them as grey figures; the result page lists them. **Find people** (about 3 minutes for 18 s of video) is the optional review: it does the tracking and matching alone, lists the appearances with a photo picker each and saves the tracks, and Generate from that list masks with exactly those tracks. The saved tracks live in ComfyUI's temp folder and are checked against the video, part and size, so after a ComfyUI restart or a change, Find people again. It needs the pack's `H3TrackCrops` and `H3TrackRoles` (which now runs Qwen itself), `H3MontagePeople`, `H3MontageTracks`, `H3MontagePicks`, `H3MontagePlan` and `H3MontagePass` nodes.

## Tips

- **Paint an edit area:** under Object swap -> Change, expand **Paint edit area** and brush extra space around the tracked part. Use the eraser, brush size and keyframe controls to refine it. During generation the painting follows SAM's bounding box in position and scale; it switches off when the track is missing. The edit includes both SAM's mask and the paint, followed by the chosen mask margin. With no paint, Change is unchanged. Update `comfyui-h3-oneclick-swap` and restart ComfyUI for `H3PaintedEditArea`.
- **Add an object:** under Object swap, choose **Add**. Describe the object (for example `a chocolate cake on a plate`) and where it goes (`on the table`). Draw a box covering the whole object and its shadow in the placement editor. An optional reference picture controls its appearance. One box stays fixed on screen; for camera movement, scrub forward and draw more boxes to animate the placement. This uses manual placement keyframes, not SAM tracking. Changing the source video, selected section or resolution clears the boxes. Update `comfyui-h3-oneclick-swap` and restart ComfyUI to load the new `H3ObjectPlacementMask` node.

- Only the main person is regenerated; anyone else in the video stays as they are.
- **Part to use** (under the source video) swaps only part of the video: **Whole**, or a **Length** placed at the **Start**, **Middle** or **End**, fine-tuned with **Starts at**. The sound is cut to the same part, the preview loops it, and Check mask uses it too. A shorter part renders faster (fewer clips).
- **Mirrors and reflections do not work yet**, in either mode. The reflection keeps the original person, and H3 copies their look from it: hair, outfit and build come back on your character. Use videos without a mirror, window or other reflection of the person.
- **Who do they replace?** names the person for the prompt by role (`the singer`), never by clothes: clothing words leak into the result. Leave it blank and the describer names them.
- **What to replace** (Simple mode and Check mask) is the text SAM tracks: `person` by default, an animal such as `cat`, or a short description (`black and white cat`) when the biggest match is the wrong one. The biggest match in the frame is replaced.
- **Hat fix** also masks the hat, so it is replaced too (about 45 s more per clip).
- **No lip-sync** (off by default) is for songs the people in the video do not sing or talk along to. Otherwise H3 makes the new people lip-sync any vocals. It gives H3 silent audio and a prompt saying nobody sings, and shows it the source faces for head turns (about 30 s more per clip); the final video still has the original soundtrack. Mouths can still smile more widely than the source.
- **Check mask** runs only SAM and the mask (about a minute per 3 s of video) and shows a preview you can scrub frame by frame: cyan is the area H3 regenerates, orange the tracked person, and on the right the grey mannequin H3 copies the motion from. Anything outside the cyan area stays as it is. If the character is cut off or too much background changes, set **Mask margin** (Tight, Normal, Loose) and check again; a re-check takes seconds.
- **Motion reference** (under Advanced) is the size of the grey figures H3 copies the motion from: 512 px by default, or 768 or 1376 px. For general, simple videos 512 px is probably enough, but a bigger size does seem to improve quality a bit: H3 follows the source poses and placement more closely. Sampling takes about a third longer at 768 px and about 2.4× as long at 1376 px (H3 uses at most 768×1344 for references). Each size keeps its own saved clips, so switching size renders again instead of resuming.
- **Model components** (under Advanced): choose the H3 Qwen3-VL 32B text encoder quantization and the video and audio VAEs from ComfyUI's installed files. **Use LoRAs** turns all selected LoRAs on or off without removing them. Each LoRA also has its own **Use this LoRA** or **Use speed LoRA** switch; turn off only the speed LoRA to keep style LoRAs and enable Spectrum. **Acceleration → Spectrum** requires [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) and is unavailable with an active speed LoRA. DMAD, Turbo, LightX2V and common few-step filenames are recognized; mark renamed speed LoRAs with **Speed LoRA**. Without a speed LoRA, use 20 or more steps. Spectrum is approximate and can change the output.
- Change the **seed** (under Advanced) for a new take. Running the same inputs again resumes from the last finished clip.
- **Steps**, **Base model** and **LoRAs** (under Advanced) default to the H3 pruned INT8 model with the DMAD speed LoRA at 6 steps. LoRAs apply in order; without a speed LoRA, use 20 steps or more. Check mask ignores them.
- **Object swap:** name what SAM tracks in a word or two (`hair`, `jacket`, `poster`), or several separated by commas (`hair, top`) to change them together. Every match changes unless **Only the biggest one** is on (one of each thing). A picture of an outfit, hairstyle, product or place is copied onto it. For **Remove**, say what shows instead only when it is not obvious (`a plain wall`). Mask margin is smaller here (10, 20 or 40 px). Change keeps the outline of what it replaces: a tank top turned into a flannel shirt stays sleeveless.
- **Fix faces:** drop the finished video and the photo the person was made from. **Match the photo** redraws the face from the photo: the closest likeness, but H3 makes up the expressions and mouth movements. **Follow the mouth** keeps the video's expressions and mouth movements, with a little less likeness. Only that person's face changes (matched by face identity); faces taller than a fifth of the frame are left alone, and stretches with only such faces are skipped. A 17 s video at 768p (4 chunks) took about 28 minutes on a 4060 Ti, including model loading. The video is read at 24 fps.
- **Fix colours:** drop the finished video. **Detail contrast** (-0.20 by default) softens crunchy fine detail; above 0 it sharpens instead. The **Tone mapper** starts from the **H3 soft** preset (Reinhard at 20% strength: softer contrast and highlights, blacks and colours kept). It is zuanfilm's H3 preset (also in the list) with less shadow lift and black floor, saturation 1.05 and 20% strength instead of 30%: zuanfilm's version washed out the blacks and dimmed whites by about 12% on H3 videos. The preset list is Orion's saved presets, read from ComfyUI; changing a value under Settings makes it Custom, and the values are sent with the job, so Orion's preset files are not changed. Save `H3 soft.json` in Orion's `presets/color_fx_dctl_tone_mapper` folder to have it in the list; without it the page uses its built-in copy. A 1.75 s 480p clip took 5 s. The video is read at 24 fps.
- **Advanced mode:** give roles as `the drummer`, optionally with how they look after ` - ` (`the drummer - man in a cap`); the look only helps match people and never reaches H3. **One pass per person** gives each of two characters their own H3 pass with only their photo (about twice the time). Check mask is Simple-only for now.

## How it works

Each clip, SAM 3.1 tracks everyone at 1008 px (its own working size), and H3 People Mask keeps the largest person. The edit mask adds 10% headroom above the head, holds each pose for 2 frames and grows by 40 px, so a differently built character has room; it does not affect the motion. The motion reference H3 follows is the source clip with that person turned into a sharp grey mannequin of their exact silhouette, so the pose, arms and fingers stay readable.

`workflows/simple.api.json` is one API export of that workflow with the hat fix and No lip-sync enabled. For each request, `lib/simple.ts`:

- without the hat fix, removes the hat nodes and connects Grow Mask to the People Mask output;
- without No lip-sync, removes the face and mute nodes, so H3 hears the soundtrack and the plan keeps its lip-sync line;
- sets the Swap Plan's resolution, seed, clip length and role.

`lib/model.ts` then sets the Swap Plan's steps and the H3 Diffusion Model file, and replaces the DMAD LoRA node with the chosen LoRAs in a chain (or links the model straight through when there are none). Advanced mode does the same.

`lib/comfy.ts` binds the uploaded video to `VHS_LoadVideo` and the photo to the `LoadImage` node, then submits with the SDK. `/api/jobs/[id]/events` reads the job's raw event stream (the SDK's low-level `getJobEvents`) and forwards it to the page as server-sent events. Node ids are turned into node titles and clip numbers there: each clip after the first is a graph expansion of the loop, whose node ids carry a prefix per clip (`64`, `66.0.0.64`, `66.0.0.Recurse.0.0.64` for Save Clip in clips 1–3). The final MP4 is the output of the Final Video node (`MiniMaxH3ChainAssemble`, node `67`).

`workflows/mask.api.json` is the mask check: the same SAM, People Mask and grow settings over the whole video, with the mask tinted over the frames next to the mannequin, saved as an MP4.

Advanced mode labels the people before it masks them. Per clip, H3 Track Crops shows each SAM track once per shot (the person alone, and the frame with them boxed) to Qwen3-VL 8B, which names the subject or "other"; H3 Track Roles reads that answer and settles a subject named twice in one shot by colour. H3 Role Mask builds the edit mask from the labelled tracks only, and the shot-list describer sees each labelled person outlined in their subject's colour, so the prompt names the same people the masks do. `lib/advanced.ts` finds nodes by title, drops unused picture branches and, without the hat fix or No lip-sync, their nodes; the exports are made from `workflows/build/oneclick_advanced_build.js` with every optional group on.

Object swap uses the same masked edit for the tracked things; SAM 3 tracks each comma-separated thing as its own prompt. The H3 Object Swap Plan writes a MiniMax video-editing prompt (`[video editing + reference generation + audio reuse]`), and the shot-list describer is told not to mention the thing being changed. **Change**: the tracked area is regenerated, and the motion reference shows it as a grey shape that keeps its shape and movement but not its colours. **Background**: the grown person mask is inverted (`flip_input`), so the edit reaches a little into the person's edge, and the whole background turns into a grey blur that keeps the camera movement. **Remove**: there is no motion reference (a grey shape would come back as a grey object), and the prompt describes what fills the area without naming what was there. `lib/object.ts` makes these changes by node title.

Fix faces works on the face, not the frame. H3 Face Track + Crop follows the face of the person in the photo over the whole video (InsightFace picks them out when there are several) and crops it to 512×512, so a face a few dozen pixels tall fills the canvas. H3 Face Fix Chunks splits the crops into independent chunks (6.5 s by default, overlapping by at least 1 s) and outputs them as lists, so ComfyUI runs the H3 nodes once per chunk. Each chunk is a masked edit: only the SAM face mask is regenerated with the photo as reference, while the hair, ears, collar and background around it stay, which keeps the new face from sliding against the head. For Follow the mouth, a blurred copy of the crop is the motion reference. H3 Face Fix Join crossfades the chunk overlaps and lowers the per-frame paste weight where the face is large, and H3 Face Stitch pastes the face back through a slightly smaller mask with the original soundtrack. The chunks are not chained through the context loop: each one starts from the swap's own frames, so colours do not drift. `lib/face.ts` removes the blurred face for Match the photo and sets the seed and chunk length.

The swap workflows are built by the scripts in `workflows/build/`: `oneclick_single_build.js` (Simple), `oneclick_advanced_build.js` (Advanced; set `globalThis.H3_MAX_QUALITY = true` first for One pass per person), and `oneclick_object_build.js`. To rebuild one, open ComfyUI in the browser with an empty workflow tab (each script clears the active tab), run `const app = window.comfyAPI.app.app;` in the developer console, paste the script, enable its optional groups (hat fix, No lip-sync) where the export needs them, export **Workflow (API)** and replace the matching `workflows/*.api.json`. Fix faces is written by `python workflows/build/openjutsu_face_build.py`. If node ids change, update `NODES` in `lib/simple.ts`. `npm test` checks the pruning against the exports.

## Deploy on the Comfy Developer Platform

Not tested yet. The app code does not change, only `.env.local`:

1. Package the prepared ComfyUI install (with the custom nodes and models above) as a Build, and deploy it:

   ```sh
   pip install comfy-cli
   comfy setup --where cloud
   cd /path/to/ComfyUI
   comfy build init --name openjutsu --models-dir ./models --custom-nodes-dir ./custom_nodes
   comfy build status
   comfy build push --release --target linux/nvidia
   comfy deploy refs compute
   ```

2. Create a deployment from the released Build on the Developer Platform. The workflow has only been run on a 16 GB card with 64 GB of system RAM, so choose at least that.
3. Set `COMFY_BASE_URL` to the deployment URL and `COMFY_API_KEY` to a key from the workspace that owns it. The key stays on the server; never put it in a `NEXT_PUBLIC_` variable.
4. Before hosting or tunneling the app, set a strong `OPENJUTSU_PASSWORD` and use HTTPS. Without a password, Openjutsu refuses to start on any address but loopback. It also refuses a password shorter than 12 characters. After 5 wrong passwords a client must wait before each try, from 1 second doubling to 5 minutes; clients are told apart by their `X-Forwarded-For` address, which only a proxy or tunnel that sets it makes reliable, so 20 failures spread across clients make everyone wait too. Open the page and sign in as `openjutsu` with that password; the same browser login protects uploads, job controls, progress and downloads. This is a single-owner app: everyone given the password shares access to its jobs. The start-up check cannot see a tunnel or reverse proxy on the same machine, so never point one at a password-free server.
5. Host the Next.js app anywhere that runs Node. `npm start` listens on `127.0.0.1` only; to serve on a network address, build and start Next with the password set (Next uses `PORT` if the host sets it, otherwise 3000):

   ```sh
   npm run build
   OPENJUTSU_PASSWORD='<strong password>' npx next start --hostname 0.0.0.0
   ```

   Streaming progress needs a host that keeps long responses open. Preserve the public Host and scheme when reverse proxying so same-origin API checks work.

Notes:

- Resuming relies on clip checkpoints in ComfyUI's `output/` folder, which a fresh serverless worker does not have.
- A job runs for many minutes; check the deployment's request time limit.
- Stop or delete the deployment when you finish, to end GPU charges.

## License

MIT, see [`LICENSE`](LICENSE). Model licences are separate. Fix faces depends on InsightFace `buffalo_l`, which is licensed for non-commercial research only, and on the Ultralytics YOLO weights from Bingsu/adetailer, which are AGPL-3.0.


---

## FILE: workflows/picture/README.md (https://github.com/D3voz/openjutsu/blob/HEAD/workflows/picture/README.md)

# From picture pipelines

These are ComfyUI API-format workflows. Drop a JSON file onto ComfyUI, or use **File → Open**.

The app uses them under **Object swap → From picture**. Choose the video and character picture, then **Generate preview image**. Compare the FLUX image with the raw frame, adjust the prompt or generate another image if needed, and click **Use this image** when ready. Set the video length, video resolution, motion-reference resolution, models, LoRAs, steps and seed, then click **Render video**. Previewing never starts H3 automatically. The video stage loads the saved preview image and does not rerun FLUX. Both stages use `COMFYUI_URL` directly; comfy-api-proxy is not required for this tab. Pending jobs and the selected preview resume when returning to the tab or refreshing the page.

| Workflow | Result |
| --- | --- |
| `from_picture_prepare_flux3.api.json` | FLUX 3 prepares and saves the first-frame picture. |
| `from_picture_motion_flux3.api.json` | FLUX preparation → Metric3D normals → H3 video. |
| `from_picture_solar_flux3.api.json` | FLUX preparation → Metric3D normals → SolarRenderer with its four-step and control LoRAs. |
| `from_picture_describe_flux3.api.json` | The visual describer writes what happens in the source part, then rewrites it for the prepared picture. |

## Video description

The normal map shows H3 how the body moves but not what is happening: held objects become unlabeled shapes, and faces lose their expression. **Use this image** therefore runs the visual describer chosen in the app (Qwen3-VL 8B by default) twice. Pass 1 watches the part at two frames per second and writes its actions, held objects, mouth actions, expressions, gaze, other people and camera, without the person's looks or the light. Pass 2 describes the prepared picture's person and setting in words, then rewrites pass 1 for them: it keeps every action and held object, drops hair or clothing movements the new character cannot make, moves the action into the picture's setting and names other people only by what they do. The rewrite reads the picture as text because, shown the image, the 8B model described the picture's pose instead of the video's actions. **Relight the video's own person** keeps the person, so it uses pass 1 directly.

The text appears in an editable box before **Render video** and opens the prompt's `detailed_description`. Remove anything about the source person's looks or setting; leave the box empty to render without it. Describe again after changing the video length. On a 4060 Ti 16 GB both passes take about 15 s with the model loaded, 40 s without.

## SolarRenderer

The app defaults to **SolarRenderer**, four steps and the rank-384 pair. Install the [official SolarRenderer custom nodes](https://github.com/Enderfga/SolarRenderer/tree/main/comfyui) and restart ComfyUI. These nodes run inside ComfyUI's existing Python environment.

Download both [solarrenderer_4step_lora.safetensors](https://huggingface.co/Enderfga/SolarRenderer/blob/main/solarrenderer_4step_lora.safetensors) and [solarrenderer_control_lora.safetensors](https://huggingface.co/Enderfga/SolarRenderer/blob/main/solarrenderer_control_lora.safetensors) from [Enderfga/SolarRenderer](https://huggingface.co/Enderfga/SolarRenderer). For the compressed pair, download the rank-64 four-step and control files from [mdevoz/SolarRendererRank64](https://huggingface.co/mdevoz/SolarRendererRank64).

Put the pair in `ComfyUI/models/loras` or any subfolder, such as `ComfyUI/models/loras/solar`. Reload the app, select **SolarRenderer LoRA rank**, then check the **four-step LoRA** and **control LoRA** file pickers. Both show the full installed relative paths; files can be selected manually, including renamed files. The selection persists across refreshes. The exported SolarRenderer workflow uses root filenames; select subfolder paths in its two loader nodes when importing directly into ComfyUI.

The four-step LoRA feeds the control LoRA, both at strength 1, through the official SolarRenderer loader. The workflow uses frame-aligned normal-map conditioning, held silent audio during sampling, the SolarRenderer four-step schedule and LCM. The output keeps the original source soundtrack. It needs no audio VAE. Both SolarRenderer and native H3 support up to 15 seconds in one clip, without a context loop. SolarRenderer needs at least 22 frames. Clips longer than six seconds can take considerably longer to render, especially at higher resolutions. Existing saved H3 settings keep using H3.

Both rank pairs completed a 141-frame 768×1376 test on an RTX 4060 Ti 16 GB. Rank 64 reduced adapter file size, with essentially unchanged sampling time and peak VRAM in that test.

Longer SolarRenderer clips need its internal silence buffer to extend beyond the released fixed template. If your node pack still reports "the held silence covers 170 frames", apply the included [extended-silence patch](../../patches/solarrenderer-long-clips.patch) to the official SolarRenderer pack and restart ComfyUI:

```sh
git -C /path/to/ComfyUI/custom_nodes/SolarRenderer apply /path/to/openjutsu/patches/solarrenderer-long-clips.patch
```

The patch preserves the silence template's beginning and end and repeats its steady middle, following the causal sampler's existing extension pattern. It does not enable causal sampling or context looping. Node-pack updates may require checking the fix again.

A rank-64 768×1376 single-clip nine-second selection completed with the fix: 209 frames / 8.708 seconds after frame-grid alignment, 34m 23s total and 15,898 MiB peak measured total VRAM. This confirms that selection on the tested installation; longer selections remain hardware-dependent.



Start with a **prepare** workflow. In **Source video**, choose the uploaded video and its `skip_first_frames`; in **Image 1 / character and background**, choose the original character picture. The first selected raw frame becomes image 2. There is no DWPose or skeleton input.

The edit request is editable and defaults to:

```text
change <image 1> character pose follow the  <image 2> pose. keep the final image as same body cut as <image 2> . do not change the background of <image 1> and do not add anything extra in <image 1>.
```

The image-aware prompt upsampler sees both references and expands that request into an editing instruction. Both references also reach FLUX directly, in the same order. The expansion rules use gender-neutral wording. This first version uses FLUX; the Qwen Image trial copied the source setting into the result.

**Relight** (off by default) poses and relights the character in the same FLUX pass; a second edit on top of the prepared picture degrades it. Pick one of seventeen light looks or write your own. The app then sends the prepare workflow with these changes (`applyRelight` in `lib/picture.ts`): FLUX gets only image 1, resized to 3 MP; image 2 reaches it only through the upsampler's written pose; the canvas is image 2's shape at 3 MP; the request's `{look}` is replaced by the light. The light usually changes the background too, and results vary a lot by seed. Relit pictures save as `relit_frame_*.png`.

**Relight the video's own person** turns the character picture off. FLUX gets only the raw frame at 3 MP, the upsampler sees the same frame, and the request asks for the same person, pose, framing and setting under the chosen light. **Framing** decides how free FLUX is: by default it may recompose the shot (strongest light, but it can zoom out to fit a light above the head); **Keep the frame's crop** starts sampling from the frame at denoise 0.9, which keeps the crop but can leave the old image showing through, such as see-through hair under Cold beam. The relit frame is saved as `relit_video_frame_*` and renders like any prepared picture.

Preparation defaults to a 768×1376 canvas and saves under `ComfyUI/output/openjutsu_picture/flux3/prepared_frame_*.png`. Compare the prepared picture against the raw first-frame preview before using it for video; a successful execution does not guarantee that the requested pose was followed.

The **motion** and **solar** variants wire the decoded prepared picture directly into H3. No manual re-upload is needed. They default to **141 frames / 5.875 seconds at 24 fps**, 480×864, four SolarRenderer steps or six H3/DMAD steps, and the original soundtrack (or silence if absent). Both renderers let you select up to 15 seconds, with no context loop. It rounds down to the H3 17k+5 frame grid: 6 seconds becomes 141 frames / 5.875 seconds; 15 seconds becomes 345 frames / 14.375 seconds. The duration shown in the settings is the actual render length. Models, LoRAs (including individual switches and strengths), H3 text encoder, VAEs, steps and seed can be changed before rendering. These settings persist across refreshes. Change `frame_load_cap` and H3's `length` together to select another valid H3 clip length.

By default normals match the H3 output canvas. The app also offers independent 512, 768 and 1376 px motion-reference sizes, rounded to multiples of 32; native H3 can cap large reference canvases. This is the motion/mask resolution control for this mode; no SAM edit mask is used. If changing that canvas, change both **Normals match the output canvas** and **Prepared picture + normal motion reference**. Normals are soft reference-video conditioning, not a trained normal ControlNet. The VRAM cleanup node runs after the image edit and before Metric3D to make room for its model.

Required installed models are selected in the loader nodes. Defaults match this installation:

- FLUX: `flux3_image_klein_sd_int8_convrot.safetensors`, `qwen3vl_4b_fp8_scaled.safetensors`, `flux2-vae.safetensors`, plus `qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors`.
- Video: the H3 pruned INT8 ref2va model, H3 text encoder and video VAE. SolarRenderer needs its four-step/control pair; H3 needs the DMAD LoRA and audio VAE used by the existing workflows.
- Custom nodes: VideoHelperSuite; KJNodes for VRAM cleanup; `comfyui_controlnet_aux` for Metric3D; the one-click swap pack for soundless-video handling. Metric3D-small's checkpoint is needed for the motion variants.

Regenerate the files from the repository root:

```powershell
python workflows/build/openjutsu_picture_pipeline.py --video "video.mp4" --picture "character.png"
```

Names refer to files in ComfyUI's `input` directory. The generated defaults use the current Instagram test clip and original café character picture; choose your own files after importing.

Validated locally: image preparation → review → explicit video render completed in the app. The saved preview and H3's picture input were byte-identical. The result contains 141 frames at 480×864 and 24 fps, with audio, and the app supports video seeking. Temporary server outages and page refreshes preserve pending jobs. Automated tests, TypeScript checking and the production build pass. The café stayed consistent; precise pose and framing remain approximate.

H3 video rendering selects Comfy Kitchen INT8 attention, matching the other video tabs. The backend change applies to newly submitted jobs.


---

## FILE: CUSTOM_NODES_REPORT.md (https://github.com/D3voz/openjutsu/blob/HEAD/CUSTOM_NODES_REPORT.md)

# Openjutsu: ComfyUI custom-node installation notes

Updated 25 October 2026. The README lists the packs, their links and the model files; this file adds what it leaves out: which workflow needs which pack, the Python and core requirements, and what to check when a pack does not load. The exact node IDs per pack are in `PACKS` in `lib/setup.ts`, which the setup check uses before every job.

## 1. Packs by workflow

Openjutsu uses the custom-node packs in the README's Custom nodes table. Install them under `ComfyUI/custom_nodes/`.

| Feature / shipped workflow | Required custom packs |
| --- | --- |
| Simple: `simple.api.json` (also Object swap From picture) | Context Loop, Fantastic PromptBuilder, KJNodes, VideoHelperSuite, Oneclick Swap |
| Advanced: `advanced.api.json` | Same five |
| One pass per person: `advanced_maxq.api.json` | Same five |
| Object Change / Remove / Background / Add: `object.api.json` | Same five |
| Montage generation: `montage.api.json` | Same five |
| Object swap ControlNet: `control.api.json` | Context Loop, Fantastic PromptBuilder, VideoHelperSuite, Oneclick Swap, comfyui_controlnet_aux |
| Check mask: `mask.api.json` | KJNodes, VideoHelperSuite, Oneclick Swap |
| Find people: `people.api.json` | VideoHelperSuite, Oneclick Swap |
| Fix faces: `face.api.json` | Context Loop, Fantastic PromptBuilder, VideoHelperSuite, Oneclick Swap, H3-FaceRefine, Impact Pack |
| Fix colours: `color.api.json` | VideoHelperSuite, Image Detail Contrast, Orion4D_FXMax |

Fix colours removes the detail or tone node when that step is off, so each of its two packs is needed only for its own step. Turning off hat fix or No lip-sync does not remove any pack from swapping.

Every workflow also needs [comfyui-websocket-reconnect-fix](https://github.com/D3voz/comfyui-websocket-reconnect-fix). It adds no nodes, so the setup check looks for its `/websocket_reconnect_fix` route instead of `/object_info`. Without it, when comfy-api-proxy reconnects to ComfyUI (a page reload, for example), ComfyUI drops the new connection and the page stops showing progress while the job keeps running.

Release notes worth knowing:

- Context Loop must be **0.7.3+**. Its Registry ID keeps the spelling `comfyui-minimaxh3-contex-loop`, and the `Contex` in its node IDs is intentional.
- Fantastic PromptBuilder must include `MiniMaxH3FantasticVideoEditLatent`; an older prompt-builder-only release does not.
- Oneclick Swap must be current: Montage, Fix faces, custom resolution (`short_edge`), Object `keep_outline` and Add (`H3ObjectPlacementMask`) all need nodes or inputs from recent versions. Unrelated H3 packs do not supply these IDs.
- Orion4D_FXMax must expose `Orion4D_FXMax_DCTLToneMapper` with embedded `raw_settings` and the preset-list routes. The workflow's `Openjutsu` preset label needs no matching preset file, because `lib/color.ts` sends the settings with the job.
- Context Loop's own install notes recommend [ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef). Openjutsu does not use it directly; Context Loop falls back without it.

## 2. ComfyUI core and Python

Openjutsu needs **ComfyUI 0.41.2 or newer**, for native H3, SAM 3.1 and text generation (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3SigmaShift`, `MiniMaxH3FunControlNetApply`, `SAM3_VideoTrack`, `SAM3_TrackToMask`, `TextGenerate`). The setup check lists missing core nodes as "ComfyUI (update required)". The app can also insert the core nodes `ImageScale` and `StringConcatenate`.

Install each pack's requirements into **the Python environment that runs ComfyUI**, not into Openjutsu's `.venv-proxy` (that one only runs `comfy-api-proxy`):

```text
<ComfyUI-Python> -m pip install -r <ComfyUI>/custom_nodes/<pack>/requirements.txt
```

For Fix faces, H3-FaceRefine needs `ultralytics`, `scipy` and `insightface`, plus an ONNX Runtime build for InsightFace (see its [requirements](https://github.com/Carasibana/ComfyUI-H3-FaceRefine#requirements) for the CPU/GPU choice). Some of these imports only happen when a job runs, so a registered node does not prove its dependencies are installed. InsightFace downloads `buffalo_l` on first use, and comfyui_controlnet_aux downloads `depth_anything_v2_vitb.pth`; fetch both in advance for an offline install.

Video handling needs FFmpeg: VideoHelperSuite can use the binary from `imageio-ffmpeg`, and Context Loop wants FFmpeg on `PATH` for assembly and review. Restart ComfyUI after installing packs or dependencies.

## 3. When the setup check lists an installed pack

The check reads ComfyUI's `/object_info`, so a pack that is installed but failed to import looks the same as a missing one. Common causes:

- **ComfyUI is too old.** comfyui-h3-oneclick-swap uses ComfyUI's SAM 3 code, so on a ComfyUI older than 0.41.2 it can fail to load and all its nodes disappear. Update ComfyUI first.
- **A Python dependency is missing** from ComfyUI's environment (see section 2).
- **The pack is an old release** that does not have the nodes listed (see the release notes in section 1).

ComfyUI's console shows the import error for each pack that did not load, and marks it `(IMPORT FAILED)` in the start-up list.

## 4. Still open

1. **Pin pack revisions.** The tested setup used Context Loop 0.7.3, Fantastic PromptBuilder 1.8.1 (`01b6d8d`), KJNodes 1.5.2, VideoHelperSuite 1.7.9 (`115de7a`), H3-FaceRefine 1.0.0 (`79a97ce`), Impact Pack 8.28.3 and Orion4D_FXMax 1.0.0. These are known to work, not proven minimums. The two D3voz packs and comfyui_controlnet_aux have no recorded revision yet.
2. **Link exact model downloads.** The README links the YOLO and ControlNet files; the INT8/INT4 ConvRot H3 and Qwen files and the DMAD LoRA still have no direct links.
3. **Validate on a clean PC.** Install from the README alone, restart ComfyUI, and run a short job in every mode, including custom resolution, Montage with Find people, ControlNet, Fix faces and the default Fix colours.

Openjutsu does not need ComfyUI-GGUF, NativeAudioLock, H3 upscalers or other packs that a larger ComfyUI install may have.


---

# ISSUES / PRs of D3voz/openjutsu


### #52 [open] PR: API v1 en espanol, notebook para Google Colab y base de datos SQLite
URL: https://github.com/D3voz/openjutsu/pull/52  by cristhianelite2 2026-11-11T02:58:55Z

## Resumen

- **API v1 en español** (`app/api/v1/*`, `lib/api-v1.ts`, `docs/API.es.md`): health, jobs CRUD, models, stats y export/import de la base de datos.
- **Notebook para Google Colab** (`notebooks/openjutsu_colab.ipynb`): instala el worker, abre tunel publico y guarda la base de datos en Google Drive para que sobreviva a los reinicios.
- **Base de datos SQLite** (`lib/db.ts`, `scripts/db.ts`, `test/db.test.ts`): node:sqlite sin dependencias nativas, copias con VACUUM INTO y copias automaticas cada 5 minutos desde el notebook.

## Cómo probarlo

1. Ejecuta el notebook en Colab (celda 2 en adelante).
2. Usa la API v1 con la clave generada en la celda 2: `curl -H "Authorization: Bearer $OPENJUTSU_API_KEY" "$URL/api/v1/health"`.
3. Reinicia el worker: los trabajos vuelven desde Drive (celda 6).


### #51 [open] ISSUE: Unable to fix faces
URL: https://github.com/D3voz/openjutsu/issues/51  by Zhayton 2026-11-07T10:28:36Z

When trying to run face post-processing I'm given this error:

Setup needs attention
Add these models to ComfyUI, or choose an installed base model or LoRA, then try again.

bbox\face_yolov8m.pt
ComfyUI/models/ultralytics

segm\person_yolov8m-seg.pt
ComfyUI/models/ultralytics

Are these typos in the .pt filepaths? \ rather than / ?


### #50 [closed] ISSUE: fetch failed
URL: https://github.com/D3voz/openjutsu/issues/50  by cobjtype 2026-11-05T14:29:06Z




### #49 [open] PR: Add a Download model button that installs the describer into ComfyUI
URL: https://github.com/D3voz/openjutsu/pull/49  by waltherhuggo 2026-11-05T13:02:30Z

## What

Adds a **Download model** button next to the *Visual describer* label on the welcome page. When the selected model is one of the suggested ones (Qwen3-VL 8B INT8, Qwen3-VL 4B FP8, Gemma 4 E2B INT8) and is not installed, clicking it makes the server download the file from Hugging Face straight into `COMFYUI_DIR/models/text_encoders`. The button shows progress, and the model list refreshes by itself when the download finishes.

- New `POST/GET /api/models/download` (same access and same-origin checks as the other API routes). Only the three suggested file names are accepted, one download at a time, written to a `.part` file and renamed when complete.
- New optional `COMFYUI_DIR` in `.env.example` (your ComfyUI folder). Without it the button explains what to set and offers the plain browser download link as a fallback.
- The suggested-model list moved to `lib/describers.ts` so the page and the server share it. README documents `COMFYUI_DIR`.
- `test/download.test.ts` covers the guards and a stubbed download.

## Notes

- Checked with `tsc --noEmit` and the new tests. I did not download the real multi-GB files or check the page visually in a browser.
- The download runs in the app server process; if the app restarts mid-download, it has to be started again.

🤖 Generated with [Claude Code](https://claude.com/claude-code)


### #48 [open] ISSUE: support for nix package manager
URL: https://github.com/D3voz/openjutsu/issues/48  by eaves-dropper 2026-11-05T05:17:34Z

**What you want**
please add support for nix package manager by adding a flake.nix file

**Why good**
people who are not familiar with npm/npx can easily install through nix


### #47 [open] PR: docs: fix tense mismatch in CUSTOM_NODES_REPORT.md
URL: https://github.com/D3voz/openjutsu/pull/47  by lunar-me 2026-11-03T15:03:24Z

## Documentation Fix

**File:** `CUSTOM_NODES_REPORT.md`
**Line:** 50
**Category:** grammar
**Severity:** minor

### Problem

The relative clause `that is installed but failed to import` joins a passive present-tense verb (`is installed`) with an active past-tense verb (`failed`) under one `that`. The second predicate needs its own auxiliary so both halves read consistently: `that is installed but has failed to import`.

> so a pack that is installed but failed to import looks the same as a missing one

### Fix

```diff
- so a pack that is installed but failed to import looks the same as a missing one
+ so a pack that is installed but has failed to import looks the same as a missing one
```

### Verification

- [x] Fix applied to the exact sentence
- [x] No other occurrences in the file
- [x] No functional code changes - documentation only

---
*Found by a documentation audit (fingerprint `48c2c26bf8fe2c62`).*



### #46 [open] ISSUE: Fails if input video has no audio
URL: https://github.com/D3voz/openjutsu/issues/46  by dvngnt 2026-11-02T23:39:18Z

I fixed by adding a silent audio track to videos 


### #45 [open] ISSUE: Fix face error on variable
URL: https://github.com/D3voz/openjutsu/issues/45  by Chardrolls 2026-11-01T15:31:03Z

On using fix face, UI gives error:
Openjutsu
Tab: post/face
Error: Prompt outputs failed validation
Browser: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36 Edg/156.0.0.0
Time: 2026-11-01T15:30:45.675Z

comfy gives error:
[INFO] got prompt
[ERROR] Failed to validate prompt for output 30:
[ERROR] * H3FaceTrackCrop 7:
[ERROR]   - Value not in list: select: 'largest' not in ['largest_face', 'smallest_face', 'left_most', 'right_most', 'top_most', 'bottom_most', 'centre_most', 'closest_to_xy', 'detector_score']
[ERROR] Output will be ignored
[WARNING] invalid prompt: {'type': 'prompt_outputs_failed_validation', 'message': 'Prompt outputs failed validation', 'details': '', 'extra_info': {}}

> **Comment by Chardrolls (2026-11-01T22:53:52Z):**

Fix is to update largest to largest_face on face.api.json



### #44 [open] ISSUE: Simple mode fails with MemoryError on long 1080p videos: the whole video is decoded at native size
URL: https://github.com/D3voz/openjutsu/issues/44  by fuleinist 2026-11-01T10:03:01Z

## Summary

Simple mode fails with `MemoryError: cannot allocate array memory` on a long 1080p video. The video loader decodes every frame at the native size before the workflow scales anything. The default part is the whole video, so a two-minute clip asks for more RAM than the machine has.

## Environment

- ComfyUI 0.42.5, with the custom-node packs from the README at current versions
- GPU with 24 GB VRAM and 64 GB of system RAM
- Source: 1920x1080, 125 s, 30 fps, VP9 WebM
- Settings: Simple motion transfer, 736p, 6 s clips, DMAD speed LoRA, default describer

## Steps to reproduce

1. Open Simple motion transfer.
2. Drop a 1920x1080 video longer than about one minute.
3. Keep **Part to use** on **Whole**.
4. Add a character photo.
5. Press **Generate**.

## Expected

The job runs. Or the page reports that the part is too long for the available memory.

## Actual

The job fails at the video loading step with:

```
File ".../videohelpersuite/load_video_nodes.py", line 379, in load_video
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (new_height, new_width, 4 if alpha else 3)))))
MemoryError: cannot allocate array memory
```

The job reports the raw exception. Nothing on the page points to the cause.

## Root cause

`VHS_LoadVideo` decodes every frame into one array of float32 RGB. The cost is `width * height * 12` bytes per frame:

- 1920 x 1080: 23.7 MB per frame
- 125 s at 24 fps: 3000 frames
- Total: **69.5 GB**, against **64 GB** of system RAM

Three facts combine to make the failure certain:

1. `workflows/simple.api.json` ships the loader with `custom_width: 0` and `custom_height: 0`. A zero value means "decode at the native size". The loader does scale inside ffmpeg when both inputs have a non-zero value, so the fix needs no new node.
2. `lib/simple.ts` changes only `skip_first_frames` and `frame_load_cap` (in `setSection`). Nothing sets the load size. The 736p setting reaches the Swap Plan, but it never reaches the loader.
3. `components/openjutsu.tsx` sets the default part to the whole video when the duration probe returns (`setPart({ start: 0, length: duration })`). A long video is therefore loaded in full by default.

The loader has a memory guard, but it estimates `width * height * 3 * 0.1` bytes per frame (`max_loadable_frames` in `load_video_nodes.py`). The real cost is about 120 times higher, so the guard never trips before the allocation fails.

## Suggested fix

1. **Set the load size to the canvas.** Pass `canvasSize(width, height, preset)` from `setResolution` into `custom_width` and `custom_height`. Skip the change when the canvas is larger than the source, so the loader never enlarges frames. With this change in a local patch, a 6 s 736p part costs 11.6 MB per frame instead of 23.7 MB, and the job completes.
2. **Warn before the job starts.** Estimate `frames * width * height * 12` from the part, the probe size and the resolution. If the total exceeds the free RAM, warn the user or clamp the part.
3. **Consider a bounded default part** for long videos. A one-minute default would avoid most of these failures.

## Upstream context

The same crash is open upstream three times: Kosinkadink/ComfyUI-VideoHelperSuite #163, #671 and #67. The maintainer states that the loader holds images as uncompressed float32. He also states that resizing happens after the load, which is not true for `custom_width` and `custom_height`: those scale inside ffmpeg and reduce the array.

## Measurements from this setup

- Whole part, 736p: fails with the error above.
- 6 s part, 736p, load size set to the canvas: completes in 495 s and writes a 1312x736 result.
- Whole part, 480p, same change: 14.6 GB for frames, which fits.

I can open a pull request for the first change if you want it.


### #43 [closed] ISSUE: 收录与创作者实测工作流通知：Openjutsu 已收录至 WaveWeaver (织浪 AI) 创作者视频工具指南
URL: https://github.com/D3voz/openjutsu/issues/43  by aronezhang 2026-11-01T02:07:13Z

作者你好！👋

非常赞赏 `openjutsu` 的架构设计！把 **MiniMax H3 (视频生成) + SAM 3.1 (人体与服装追踪) + Qwen3-VL 8B (多模态视觉打标) + FLUX.3 Klein (17 组环境光重打光)** 封装在极简的 Next.js + `@comfyorg/sdk` 界面下，真正把 Higgsfield Genjutsu 的商业级视频角色替换能力带入了开源生态，彻底免去了高昂的月费订阅！

我们刚刚在 **WaveWeaver（织浪 AI）** 创作者工具专栏为 `D3voz/openjutsu` 编写并发布了一篇面向自媒体博主与 AI 视频创作者的英文实测与全链路工作流指南（详细对比了 Openjutsu vs. Higgsfield vs. HeyGen，并提供了 16GB VRAM 本地 ComfyUI 避坑指南以及从爆款选题/母版脚本到多平台排期分发的串联方案）：

- 🔗 **评测与工作流指南地址**：https://waveweaver.io/tools/openjutsu
- 🔗 **专栏首页推荐位**：https://waveweaver.io/

指南中已完整附上本开源仓库链接与 Star 引导。感谢作者开源这么硬核的视频换人工具，祝项目早日破万 Star！🚀


### #42 [open] ISSUE: 收录与创作者实测工作流通知：Openjutsu 已收录至 WaveWeaver (织浪 AI) 创作者视频工具指南
URL: https://github.com/D3voz/openjutsu/issues/42  by aronezhang 2026-11-01T02:07:02Z

作者你好！👋

非常赞赏 `openjutsu` 的架构设计！把 **MiniMax H3 (视频生成) + SAM 3.1 (人体与服装追踪) + Qwen3-VL 8B (多模态视觉打标) + FLUX.3 Klein (17 组环境光重打光)** 封装在极简的 Next.js + `@comfyorg/sdk` 界面下，真正把 Higgsfield Genjutsu 的商业级视频角色替换能力带入了开源生态，彻底免去了高昂的月费订阅！

我们刚刚在 **WaveWeaver（织浪 AI）** 创作者工具专栏为 `D3voz/openjutsu` 编写并发布了一篇面向自媒体博主与 AI 视频创作者的英文实测与全链路工作流指南（详细对比了 Openjutsu vs. Higgsfield vs. HeyGen，并提供了 16GB VRAM 本地 ComfyUI 避坑指南以及从爆款选题/母版脚本到多平台排期分发的串联方案）：

- 🔗 **评测与工作流指南地址**：https://waveweaver.io/tools/openjutsu
- 🔗 **专栏首页推荐位**：https://waveweaver.io/

指南中已完整附上本开源仓库链接与 Star 引导。感谢作者开源这么硬核的视频换人工具，祝项目早日破万 Star！🚀

> **Comment by goldicom789-lgtm (2026-11-12T10:29:56Z):**

<img width="1024" height="1536" alt="file_00000000d2848211bc0900db7b0bb496" src="https://github.com/user-attachments/assets/3e76aa76-d008-40a9-a545-41e0d23f3c6f" />


> **Comment by goldicom789-lgtm (2026-11-12T10:32:29Z):**

https://github.com/user-attachments/assets/5daa9193-7daf-4cb5-a19a-38c2161d5c5b



### #41 [open] ISSUE: Official Links for models
URL: https://github.com/D3voz/openjutsu/issues/41  by adrnga7 2026-10-31T22:33:21Z

Thanks for your work! Could you provide official links in README for models ?

> **Comment by oussamadjaoui90-cmyk88 (2026-11-01T00:11:59Z):**

good



> **Comment by alisson-anjos (2026-11-02T22:07:18Z):**

https://huggingface.co/Enderfga/SolarRenderer



### #40 [open] PR: Add English and Simplified Chinese language switching
URL: https://github.com/D3voz/openjutsu/pull/40  by H-Bole 2026-10-31T02:57:07Z

The welcome screen and editor were English-only. This adds a 语言 / Language selector to both headers, with English and Simplified Chinese UI text across the editing modes, settings, progress, result views and accessibility labels.

The first visit follows the browser's first supported language and falls back to English. An explicit choice is saved in localStorage; switching languages preserves the current inputs, settings and job. Model filenames, generation prompts, API values and diagnostic details retain their original values. No new dependencies.

Validation:
- `npm test`: all 134 tests pass, including locale selection, interpolation, fallback and UI translation coverage.
- Production build passes (Turbopack in the installed app; webpack in an isolated worktree).
- Browser checks cover Chinese onboarding, English/Chinese switching, persistence after reload, retained target/resolution settings, and object, picture and post-processing views; no browser console errors.



### #39 [open] PR: Recognize completed outputs with valid MIME case and parameters
URL: https://github.com/D3voz/openjutsu/pull/39  by rudycelekli 2026-10-29T06:49:47Z

Completed videos or Find people JSON can disappear from the public job result when the backend reports valid MIME capitalization or parameters, such as `VIDEO/MP4` or `application/json; charset=utf-8`. Discover outputs by the normalized base media type, consistent with how the download route already accepts them. Media type tokens are case-insensitive and can have parameters ([RFC 9110, section 8.3.1](https://www.rfc-editor.org/rfc/rfc9110.html#section-8.3.1)).

Validation: three native loopback job-result cases fail before the fix. All 16 focused output/submission regressions and TypeScript checks pass afterward; the 5 new cases pass on Node 22. Canonical outputs are unchanged, and unrelated images/problem documents are not selected as video or track results. Batch production-build qualification will follow when the backend fixes are combined. No GPU/actual ComfyUI execution claimed.

Prepared with AI assistance. This branch is independent of the existing stream framing changes.



### #38 [open] PR: Return 404 for missing jobs during lookup and cancellation
URL: https://github.com/D3voz/openjutsu/pull/38  by rudycelekli 2026-10-29T06:47:27Z

The job lookup and cancellation routes turn the SDK's typed missing-job response into HTTP 502. Return 404 for a genuinely missing or expired job, including one removed between lookup and cancellation, while retaining 400 for invalid IDs and 502 for backend outages.

Validation: five native loopback regressions fail before the fix. Focused job/HTTP regressions and TypeScript checks pass afterward; the 8 new cases also pass on Node 22. Controls cover both supported not-found codes, GET/DELETE, a cancellation race, a 503 outage and invalid input. Batch production-build qualification will follow once the backend fixes are combined. No GPU/actual ComfyUI execution claimed.

Prepared with AI assistance. This change is independent of the pending transport/body-streaming fixes.



### #37 [open] PR: Fit Solar preset canvases within native render bounds
URL: https://github.com/D3voz/openjutsu/pull/37  by rudycelekli 2026-10-29T06:45:28Z

SolarRenderer's conditioning node accepts 256–2048 px per side, but a preset only fixes the short edge. A 1856×768 source at 1080p becomes 2624×1088 and exceeds the native width limit, even though the app offers that preset.

Use a shared renderer-aware canvas helper in graph construction and the picture editor. Solar preset canvases now fit its native bounds on the 32-pixel grid, and the editor shows the actual canvas plus a limit hint when it is constrained. H3 presets, explicit valid custom dimensions and the default Solar canvas stay unchanged. Extremely wide/tall sources use the closest legal canvas because the native minimum side is 256 px.

Validation:

- The export regression fails before the fix and passes afterward across wide/tall sources and all presets; positive controls cover H3, custom and default behavior. All **128 standalone tests plus TypeScript pass**.
- The mounted editor reproduces the 21:9 failure before and passes afterward: the displayed and constructed render canvas are both 2048×864; switching to H3 still shows 2624×1088.
- The unmodified [current native INPUT_TYPES schema](https://github.com/Enderfga/SolarRenderer/blob/41e136e64b30d4aa0095070ce055ff00ae90fe3b/comfyui/nodes.py#L137) was evaluated directly: the old width violates its published bound, and the new dimensions satisfy min/max/step.
- This and the dormant-settings fix merge cleanly with the prior integration. Combined qualification is still running; no combined-suite pass is claimed.

A new production build is pending the campaign's shared disk-reserve check; no per-PR build pass is claimed here. The existing cached build artifacts were removed to preserve the 1GiB reserve. No installs, models, paid calls or GPU inference were used. Prepared with AI assistance.



### #36 [open] PR: Accept verified photos when MIME metadata is missing
URL: https://github.com/D3voz/openjutsu/pull/36  by rudycelekli 2026-10-29T06:41:28Z

The editor accepts PNG/JPEG/WebP filenames when a drag source omits MIME metadata, and multipart parsing verifies their signatures. The submission API then rejects those photos solely because their MIME string is empty or generic. Accept a supported picture suffix after the existing byte checks, matching the video-input fallback.

Validation: PNG/JPEG/WebP missing/generic MIME cases fail before the change. All 16 focused photo/submission regressions and TypeScript checks pass afterward; the 5 new cases also pass on Node 22. Invalid bytes with a photo suffix and unknown suffixes without MIME remain rejected before backend upload. Tests use real multipart temp files and a loopback SDK proxy fixture; no GPU/actual ComfyUI execution claimed. Batch production-build qualification will be reported after the related backend fixes are combined.

Prepared with AI assistance. This branch is independent of the pending editor changes and contains only the API fallback and its regression tests.



### #35 [open] PR: Keep mask playback state accurate after play rejection
URL: https://github.com/D3voz/openjutsu/pull/35  by rudycelekli 2026-10-29T06:38:00Z

The mask review switches its button to Pause immediately after calling `video.play()`. If playback is rejected, the rejected promise goes unhandled and the UI claims the video is playing.

Track playback through native play/pause events, handle play rejection with a retryable error, and discard errors from superseded attempts or an expired review. Frame stepping still pauses playback.

Validation: mounted React regression fails before the fix because rejected playback leaves Pause visible. Afterward it verifies the error, successful retry, pause, pending-play state, and an obsolete rejection arriving after a newer successful attempt. `npm test` passes 127 tests and TypeScript. The media API is controlled in jsdom; native codec/GPU behavior was not qualified.



### #34 [open] PR: Ignore dormant picture-render component requirements
URL: https://github.com/D3voz/openjutsu/pull/34  by rudycelekli 2026-10-29T06:37:31Z

Switching renderers can leave a valid render blocked by hidden settings. For example, clear H3's Audio VAE and switch to Solar: the field disappears, but the server parser still requires it even though Solar uses no audio VAE. H3 similarly requires hidden Solar filenames. Turning off H3 LoRAs can leave an unused out-of-range strength blocking the selected render.

Require nonempty filenames only for the selected renderer, and enforce strength bounds only for enabled H3 LoRAs when LoRAs are on. Keep dormant preferences, basic types, finite values and shared list limits; active components remain validated. No graph-generation or provider behavior changes.

Validation on Node24.22.1/macOS arm64:

- Three regression scenarios fail before; all **130 tests plus TypeScript and the production Webpack build pass** afterward. The tests traverse actual H3/Solar exports from the same saved-image binding as the render route, proving unused components are absent from submitted dependencies. Active-option negative controls remain rejected.
- Additional mounted-editor evidence clears Audio VAE, switches to Solar and captures the exact request: the public server parser rejects it before and accepts it after. This uses the existing campaign DOM harness, without adding test dependencies here.
- Clean local integration onto `1cb90b6d6f46c4fd4fa293f955287cf09aa2f32f`: **248 tests plus TypeScript and production Webpack build pass**. No upstream merge is implied.
- Current [SolarRendererConditioning schema](https://github.com/Enderfga/SolarRenderer/blob/41e136e64b30d4aa0095070ce055ff00ae90fe3b/comfyui/nodes.py#L137) requires a video VAE and no audio VAE, matching the checked-in export.

Webpack supports the external node_modules symlink in the isolated worktree. No installs, models, paid calls or GPU render were performed; export/DOM/parser qualification is not a provider inference claim. Prepared with AI assistance.



### #33 [open] PR: Ignore submission responses after the editor unmounts
URL: https://github.com/D3voz/openjutsu/pull/33  by rudycelekli 2026-10-29T06:36:08Z

A submission can outlive the editor: a delayed setup JSON response starts an upload after unmount, and a delayed jobs response overwrites the saved run belonging to a newer editor. Ignore submission continuations from an expired component lifetime before starting the upload, saving the run, or updating state.

Validation: mounted React tests defer setup/jobs JSON, unmount the actual editor, and place a newer run in localStorage. Both stale paths fail before the fix and pass afterward; both still-mounted controls complete normally. `npm test` passes 130 tests and TypeScript. An already submitted server job continues independently; this change prevents stale client continuation and storage writes.



### #32 [open] PR: Bind person classification reviews to their ordered roles
URL: https://github.com/D3voz/openjutsu/pull/32  by rudycelekli 2026-10-29T06:35:39Z

Find people classifies tracks using the ordered role descriptions. After changing a role in settings, reopening the previous review still enabled Generate, applying old classifications to the new role mapping.

Save the ordered roles with each people review and enable reuse only when they still match. Saved reviews without a role snapshot require a fresh check. The recovery hint explains this condition.

Validation: mounted React reproduction changes a role, reopens history, and observes Generate incorrectly enabled before the fix; it is disabled afterward. An unchanged-role control remains enabled. `npm test` passes 128 tests and TypeScript.



### #31 [open] PR: Keep source and reference uploads accessible from the keyboard
URL: https://github.com/D3voz/openjutsu/pull/31  by rudycelekli 2026-10-29T06:35:15Z

The source and reference upload inputs use `hidden`, which removes the only native upload control from keyboard navigation. Keep real file inputs focusable with visually clipped styling and show a visible outline on the enclosing upload label when focused.

Validation: mounted React regression fails on the original hidden inputs and passes after the fix; source selection via the native input change event still works. `npm test` passes 127 tests and TypeScript. This verifies DOM focus and change handling; an operating-system file dialog was not exercised in jsdom.



### #30 [open] PR: Preserve HTTP framing for decoded media responses
URL: https://github.com/D3voz/openjutsu/pull/30  by rudycelekli 2026-10-29T04:55:56Z

Native fetch decodes compressed media but retains its upstream headers. Both media routes then copy the compressed Content-Length onto the decoded body: a real gzip response produced 54 declared bytes for 3013 JSON bytes, and 86 declared bytes for 68 PNG bytes. Native downstream HTTP clients truncate the JSON or wait for missing image bytes.

Request identity encoding for media reads. If an upstream still sends a supported gzip/deflate/br full response, serve the decoded body without the coded length, range metadata or strong validators. Reject unknown codings and coded 206 responses, canceling their unused bodies, since their byte ranges cannot describe the decoded representation. Identity 200/206/416 behavior is preserved.

Validation: native upstream and downstream HTTP tests fail against the all 27 integration before the fix. All 164 Node 24 tests and TypeScript checks pass afterward; 35 focused Node 22 checks pass, including 16 wire cases across job outputs and picture media. The fixtures exercise gzip/deflate/br, identity full/partial/unsatisfiable responses, unexpected coded 206 and unknown encodings. Production webpack build passes. No browser, GPU or actual ComfyUI execution is claimed.

Dependencies: pending #22 and updated #26 are included, which already carry #6/#10. Pending #5 is normal-merged because its adjacent Unicode filename block otherwise conflicts with the transfer-header change. No unrelated integrated production changes are included. Default local Turbopack cannot follow the shared dependency symlink outside this isolated worktree; root separately qualifies the normal build. Hosted Actions currently require owner approval. Prepared with AI assistance.



### #29 [open] PR: fix(editor): discard abandoned post-processing downloads
URL: https://github.com/D3voz/openjutsu/pull/29  by rudycelekli 2026-10-29T04:48:47Z

Downloading a finished result for Fix faces/Fix colours can take long enough for the user to start a new swap. The old download currently continues and, when its body arrives, replaces the newly selected video and resets the current work. Its late failure can also appear on the unrelated new swap.

Keep the download tied to its result view. Leaving the view or starting another result download aborts the old request; a response body that settles late cannot change the source, mode or current job. Normal successful downloads still open the requested post-processing tool.

Mounted React DOM proof: two failures and one successful control before the fix; all three pass after. The body is deliberately deferred past navigation, so the state guard is tested even when cancellation cannot stop a late callback.

Validation on macOS arm64:
- Complete Node24.22.1 `npm test`:132 passed, including TypeScript.
- Node22.19.0 focused post-processing/cancellation controls:6 passed.
- Normal production `npm run build` and `git diff --check` pass.

Depends on #13's pending cancellation lifecycle change; its normal merge was resolved ahead of review so both protections are preserved. Uses the existing dev-only DOM test harness. Download bodies/events are controlled fixtures; no native decoding or GPU render is claimed. Prepared with OpenAI Codex assistance.



### #28 [open] PR: fix(editor): handle unavailable result metadata safely
URL: https://github.com/D3voz/openjutsu/pull/28  by rudycelekli 2026-10-29T03:04:55Z

A JSON error response from a result-data download is currently treated as successful metadata. Find people throws while reading `people.map`, montage results crash while reading `kept.length`, and Fit check can misleadingly say nobody was found.

Check HTTP status before parsing result metadata, catch failures in the complete async chain, and show a recoverable metadata alert while keeping the successful video available. Abort pending metadata reads when leaving the view, clear stale result state when the URL changes, and release unused error bodies. Normal successful metadata keeps its existing behavior.

Mounted React DOM controls exercise actual completed People/Fit/Montage views: 6 fail / 3 pass before; all 9 pass after (HTTP-error handling, successful results and pending-request cancellation on Back).

macOS arm64 / Node24.22.1: complete `npm test` passes 135 tests plus TypeScript; normal `npm run build` and `git diff --check` pass. Uses common dev-only DOM harness ancestry without the other editor production fixes. Metadata responses are controlled fixtures; no native video decoding or GPU inference is claimed. Prepared with OpenAI Codex assistance.



### #27 [open] PR: Bind saved render histories to complete media contents
URL: https://github.com/D3voz/openjutsu/pull/27  by rudycelekli 2026-10-29T03:02:48Z

The native Swap Plan's saved-run identity samples source frames and averages RGB before hashing; it also omits audio. Distinct media can therefore share a history even when the new job should generate different content. For example, the actual native hash returns the same value for red-only and blue-only CPU tensors. Replacing a soundtrack while keeping the video frames can likewise leave the native identity unchanged.

Bind each chain submission to streamed SHA-256 digests of the complete video and its ordered, actually used references, plus the selected source interval. Identical bytes under different temporary filenames retain the same identity; video/reference content, order, count or interval changes separate it. Remove mode excludes its unused picture.

The helper appends a compact media namespace to the existing run-name graph, preserving model/motion/mask/per-person namespaces and staying within native's 96-character name limit. The full 256-bit digest also reaches `generation_fingerprint`, preserving any prior compatibility tag. A rare compact folder-name collision therefore requires regeneration through the full native compatibility check instead of resuming different media. Existing outputs are preserved; the first run with this identity starts a new history.

Validation on Node24.22.1/macOS arm64:

- All five SDK loopback regressions fail on main before the fix (Simple, Advanced, per-person Advanced, Object and Montage). They exercise the actual installed SDK0.5.0 upload/submission paths and inspect the transmitted production graphs; after the fix they pass alongside retry/reference/interval controls and unused-picture removal.
- Full `npm test`: **135 passed plus TypeScript**. `npm run build -- --webpack` and `git diff --check` pass. Webpack supports the external node_modules symlink used by this isolated worktree.
- Clean local integration with the first 20 contribution branches (`ffebbb52`): **189 passed plus TypeScript and production Webpack build**. No upstream merge is implied.
- Unmodified [native run_hash](https://github.com/D3voz/comfyui-h3-oneclick-swap/blob/9cf2d1a5e6f5f4e5fc3c11a82d9bc897e85085da/__init__.py#L291) executes on real CPU tensors and confirms the color collision. Unmodified [Context Loop comparison functions](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/43afeffcc570beb5f7382593e79e4b211c7e02f6/chain_nodes.py#L6741) preserve the full tag and require regeneration when synthetic dependency records differ in that tag.

This adds one complete sequential file read before SDK upload/hash work, using 64KiB stream chunks and no persistent cache or dependency. Loopback file contents and checkpoint records are controlled fixtures; no GPU render, real ComfyUI checkpoint-resume session, model download or paid provider execution is claimed. Prepared with AI assistance.



### #26 [open] PR: Preserve healthy picture media streams beyond the header deadline
URL: https://github.com/D3voz/openjutsu/pull/26  by rudycelekli 2026-10-29T03:00:07Z

Picture media inherits the helper's 30-second total response deadline, so a healthy video transfer is aborted even after headers arrived successfully. The helper also replaces the caller signal. For `/view` transfers, apply that deadline while waiting for headers (and reading error JSON), then detach it from a healthy media body. Compose the caller signal so leaving the request still closes the transfer. Metadata JSON retains its original bounded lifetime.

Validation: native loopback tests with a controlled expiry of the configured 30,000ms signal reproduce both healthy-body truncation and ignored caller cancellation on main `47c3724` and the first 20 fixes integrated at `ffebbb52`. Controls retain deadlines for missing headers, stalled metadata JSON and stalled media-error JSON. After the fix, all 137 Node 24 tests and TypeScript checks pass; all 11 focused media/range/polling regressions pass on Node 22. Production webpack build passes.

The deadline event is controlled to avoid a 30-second test delay; the bytes and socket closure use real local HTTP, not a mocked fetch. No GPU/ComfyUI execution claimed. Default local Turbopack cannot resolve the shared dependency symlink outside the isolated worktree; normal CI is unchanged and currently needs owner approval. Prepared with AI assistance.

Dependency: #6 is included through a normal merge commit so its pending HTTP416 behavior is preserved alongside the streaming lifetime fix. Its range regressions are included in the 133-test result.

Polling lifetime follow-up: pass the same caller signal into picture history and queue lookups, including their JSON bodies. Four native loopback regressions failed against the all-27 integration before this follow-up; history/queue header and body waits now stop when polling is canceled. This extends the existing request-lifetime contract and does not add a separate PR.



### #25 [open] PR: Keep object placement gestures with their starting pointer
URL: https://github.com/D3voz/openjutsu/pull/25  by rudycelekli 2026-10-29T02:59:51Z

A second pointer can replace the canvas origin and commit the first pointer's unfinished placement box. An unrelated `pointercancel` also discards the active drag. Bind each gesture to its starting pointer ID and ignore other pointer events until the owner ends or cancels it.

Actual mounted React DOM regressions reproduce both failures before the change and pass afterward. A control retains the existing behavior of discarding a box when its own pointer is cancelled. Node 24.22.1: `npm test` passes 129 tests plus TypeScript; `npm run build -- --webpack` and `git diff --check` pass. Webpack is used in this isolated worktree because dependencies are linked from the main checkout.

Only `components/object-placement.tsx` changes production behavior. Shared test-only DOM scaffolding is independent of the other editor fixes. These tests dispatch distinct pointer identifiers through the actual component; they do not claim hardware touch, native canvas/media decoding, GPU or ComfyUI qualification. Prepared with OpenAI Codex assistance.



### #24 [open] PR: Clamp object placement to the shortened clip
URL: https://github.com/D3voz/openjutsu/pull/24  by rudycelekli 2026-10-29T02:59:17Z

After scrubbing placement to 8 seconds, shortening the selected section from 0–10 to 0–4 seconds clears its boxes but leaves this editor seeking to 8 seconds. The next drag saves frame 192 outside the chosen clip. Clamp the selected time to the last 24fps frame of the new interval, while preserving a selected frame that still fits.

Actual mounted React DOM regressions reproduce the out-of-range saved frame before the change and pass afterward, with an in-range control. Node 24.22.1: `npm test` passes 128 tests plus TypeScript; `npm run build -- --webpack` and `git diff --check` pass. Webpack is used in this isolated worktree because dependencies are linked from the main checkout.

Only `components/object-placement.tsx` changes production behavior. Shared test-only DOM scaffolding is independent of the other editor fixes. Tests control metadata and dispatch real DOM events through React; no native media decoding, GPU rendering or live ComfyUI result is claimed. Prepared with OpenAI Codex assistance.



### #23 [open] PR: fix(mask): preview the custom generation canvas
URL: https://github.com/D3voz/openjutsu/pull/23  by rudycelekli 2026-10-29T02:55:59Z

Mask check currently ignores custom output dimensions while Simple generation centre-crops before SAM. For a landscape source cropped to portrait, the preview can therefore qualify a mask/person outside the canvas generation actually uses.

Carry the selected resolution through the editor's setup request and form, validate it in the mask route, and crop the mask workflow before both SAM and preview consumers. Setup and actual SDK submission now see the same custom canvas. Preset aspect-ratio behavior and callers that omit resolution remain compatible.

Evidence on macOS arm64 / Node24.22.1:
- Before: 4 failures / 2 passing controls across mounted editor, actual setup graph and real SDK/owned-loopback submission.
- After: all 6 controls pass; complete `npm test` passes 132 tests and TypeScript.
- Normal `npm run build` and `git diff --check` pass.

The tests check actual exported graph rewiring, request fields and rejection before upload. They do not run SAM/GPU inference or claim a rendered-video result. Uses the shared dev-only DOM harness ancestry without the other editor production fixes. Prepared with OpenAI Codex assistance; no release or deployment.



### #22 [open] PR: Stop abandoned output reads and release upstream media streams
URL: https://github.com/D3voz/openjutsu/pull/22  by rudycelekli 2026-10-29T02:55:41Z

Canceling an output request currently leaves its job/asset lookups and media transfer running upstream. Forward the request signal through SDK-backed job/output handles, low-level asset calls and the final byte fetch, composing cancellation with the SDK's existing request deadline. Also cancel an unused upstream error body before returning502.

Validation: six native loopback cancellation regressions fail against both main `47c3724` and the first 20 fixes integrated at `ffebbb52`. They observe the upstream socket during job lookup, preview metadata/URL lookup, listed-output URL lookup, response headers and body. A separate control reproduces an unused500 media stream remaining open. After the fix, all 138 Node 24 tests and TypeScript checks pass; all 12 focused cancellation/range/error regressions pass on Node 22. Production webpack build passes.

The default local Turbopack build cannot resolve shared dependencies outside this isolated worktree; normal CI remains unchanged. Hosted Actions currently require owner approval. No GPU execution or backend job cancellation is claimed: this stops abandoned read requests. Prepared with AI assistance.

Dependencies: #6 and #10 are included through normal merge commits so the pending range and typed asset-error fixes compose with cancellation. Their existing regression tests are included in the 138-test result.



### #21 [open] PR: Recover the picture editor when a saved preview expires
URL: https://github.com/D3voz/openjutsu/pull/21  by rudycelekli 2026-10-28T18:04:57Z

A saved approved preview is rechecked on restore. If that preparation has expired or failed, polling clears the job but leaves approval true and preview null. Both source inputs and render/review controls are then hidden. Clear the failed preparation and its approval together so the user can choose inputs and generate again while retaining the error message.

Validation: mounted React regressions for HTTP400/404/422 fail before the fix; all130 Node tests and TypeScript checks pass afterward. A failed-render control retains its valid approved preparation and enabled retry action. Production webpack build passes. Tests use actual React/jsdom/SWC with simulated API responses, not browser/GPU execution; default Turbopack cannot follow shared dependencies outside the isolated worktree.

Prepared with AI assistance. This branch includes shared dev-only DOM test scaffolding and no other production fix.



### #20 [open] PR: Restore the correct canvas for saved picture renders
URL: https://github.com/D3voz/openjutsu/pull/20  by rudycelekli 2026-10-28T18:03:42Z

Restoring a saved render restores its preview URLs but loses the prepared source dimensions. The settings panel then claims a fallback portrait canvas even for a landscape source, and selecting Custom sends those incorrect dimensions. Read the saved preparation's metadata, preserve its aspect ratio, and make Custom wait until dimensions are known. Cancel the lookup when review/source identity changes.

Validation: a mounted landscape restore regression fails before the fix; all129 Node tests and TypeScript checks pass afterward. Controls check the actual submitted Custom864×480 settings, pending metadata, and a late old response after selecting a new portrait source. Production webpack build passes. These are actual React/jsdom/SWC tests with simulated API responses, not browser/GPU execution. Default Turbopack cannot follow shared dependencies outside the isolated local worktree.

Prepared with AI assistance. The branch includes shared dev-only DOM test scaffolding and no other production fix.



### #19 [open] PR: fix(editor): accept supported drops without MIME metadata
URL: https://github.com/D3voz/openjutsu/pull/19  by rudycelekli 2026-10-28T18:01:28Z

Supported MP4/WebM clips and PNG/JPEG/WebP reference pictures are silently ignored when dragged from a source that supplies an empty or generic MIME type. The drop filter has a filename fallback only for MKV/MOV/AVI, even though those other formats are documented and can be selected using the file picker.

Extend that same filename fallback to all supported media extensions. The server's byte-signature checks remain the authority before submission; this does not replace them with filename trust.

Actual mounted React DOM drop events reproduce 6 failures / 5 passing controls before the change. All 11 cases pass after: nine supported extensions (including existing MKV/MOV/AVI controls) and two unrelated-file refusals.

macOS arm64 / Node24.22.1: `npm test` passes 137 tests and TypeScript; normal `npm run build` and `git diff --check` pass. Uses the shared dev-only DOM harness without the other editor production fixes. Fixtures test drag/drop state, not decoding or GPU rendering. Prepared with OpenAI Codex assistance.



### #18 [open] PR: Keep late picture responses from overwriting newer saved jobs
URL: https://github.com/D3voz/openjutsu/pull/18  by rudycelekli 2026-10-28T18:00:33Z

Leaving From picture after response headers arrive does not stop a pending JSON body from resolving. The old poll, preparation or render handler can then overwrite a newer saved job in localStorage. Check the polling request's abort signal after JSON resolves and guard submission continuations with the mounted component's lifetime.

Validation: three mounted React regressions fail before the fix, each deferring JSON across unmount and checking a newer saved job survives. All130 Node tests and TypeScript checks pass afterward, including a positive mounted preparation→approval→render flow. Production webpack build passes; default Turbopack cannot follow shared dependencies outside the isolated worktree. Tests use jsdom and the app's Next SWC compiler, not a browser/GPU run. This guards UI state; it does not claim to cancel a backend job.

Prepared with AI assistance. This branch includes the shared dev-only React/jsdom test scaffold used by the other editor fixes; no other production fix is included.



### #17 [open] PR: fix(editor): inspect the moving object graph before uploading
URL: https://github.com/D3voz/openjutsu/pull/17  by rudycelekli 2026-10-28T18:00:16Z

An Add request sends its action to job submission but omits it from setup inspection. For “she dances the same routine in sync with the others”, setup checks the static-placement graph; the actual render uses `H3AddMovingArea`, so a missing required node is discovered only after the video upload.

Send the same action with the setup selection. A mounted React DOM regression fills the real Add controls, draws a placement box, requests generation and runs the captured setup request through `setupGraph` with the actual exported workflow. The moving case fails before the fix; after, both moving and static controls inspect the correct node set and stop before upload when setup is unavailable.

macOS arm64 / Node24.22.1: `npm test` passes 128 tests plus TypeScript, normal `npm run build` and `git diff --check` pass. Reuses the common dev-only DOM harness without depending on the other editor production fixes. No ComfyUI/GPU render is claimed. Prepared with OpenAI Codex assistance.



### #16 [open] PR: fix(editor): bind saved checks to the selected source
URL: https://github.com/D3voz/openjutsu/pull/16  by rudycelekli 2026-10-28T17:57:50Z

Saved Find people, mask and fit checks currently identify their source with only filename, byte count and section. Replacing a clip with different contents under the same name and size lets the old check be treated as matching the newly selected clip.

Include the selected video's object URL in the browser-local frame identity. This conservatively invalidates old checks whenever the source is replaced, while reopening a check for the unchanged selection still works. It does not claim a content hash or replace ComfyUI's own track validation.

The mounted-editor regression performs Find people, returns to settings, replaces `clip.mp4` with different same-size bytes, and reopens the saved check. Before: Generate is enabled with stale tracks. After: it is disabled and asks the user to find people again. The unchanged-source positive control stays enabled.

macOS arm64 / Node24.22.1: `npm test` (128 passed + TypeScript), normal `npm run build`, and `git diff --check` pass. Uses the common dev-only DOM harness commits from #7/#13 without their production changes; these editor PRs can merge in either order. Responses/metadata are controlled fixtures, with no GPU rendering claim. Prepared with OpenAI Codex assistance.



### #15 [open] PR: Honor explicit negation of copied object routines
URL: https://github.com/D3voz/openjutsu/pull/15  by rudycelekli 2026-10-28T17:55:41Z

An Add action such as `not in sync` or `do not synchronize` currently selects the copied-routine moving mask because the synchronization keywords match inside the negated phrase. Both the fit preview and render then track a person and wire H3AddMovingArea even though that specific instruction disables copying.

Exclude explicitly negated copying phrases before looking for a positive copying instruction. This is deliberately bounded to the existing phrase heuristic; it does not claim general natural-language interpretation. The action text itself is preserved, and positive copying instructions elsewhere in the same text still work.

The new regression fails before the fix and passes afterward. It loads the actual fit and object workflow exports and checks branch selection for five negated phrases plus four positive controls, retains the action, and verifies links. All 128 tests, TypeScript checking, and `npm run build -- --webpack` pass. Webpack supports the shared external node_modules symlink used by this worktree; no GPU/model execution is claimed.

Prepared with AI assistance.

Current exact head: `f2491feb65f9f79586f00fda9492595ea88bf1a0`. Combined local integration of #4, #11, #14 and #15 merges cleanly and passes all 134 tests, TypeScript checking and the Webpack production build (integration head `cac56458991e0bf7dd7ca6596d1d8a2c1bed6abc`). This is a separate local integration check, not an upstream merge. As of `2026-10-28T17:59:53.948656+00:00`, GitHub reports no visible check rollup or review decision for this PR; hosted CI success and maintainer approval are not claimed.



### #14 [open] PR: Keep per-person Advanced renders in a distinct saved history
URL: https://github.com/D3voz/openjutsu/pull/14  by rudycelekli 2026-10-28T17:54:22Z

Switching between ordinary Advanced and **One pass per person** can resume the same saved clip history. Both exports feed equivalent inputs into the same native H3AdvancedSwapPlan and use its same run-name output, but ordinary Advanced has one sampler and per-person mode has two. Neither export supplies an external generation fingerprint for that difference.

Give the per-person API export and its workflow builder their own `_per_person` run-name namespace before adding any motion/model suffixes. Ordinary Advanced retains its existing name. This preserves normal resume within each strategy while keeping the other strategy's completed clips out of that history.

The new regression loads both actual exports, compares native planning inputs by semantic node/output rather than node IDs, verifies the one-versus-two sampler difference, and resolves the real run-name concatenation graph. It fails before the fix and passes afterward for both 512 and 768 motion settings, with no dangling links. All 127 tests plus TypeScript checking and `npm run build -- --webpack` pass. Webpack was used because this isolated worktree shares an external node_modules symlink that Turbopack refuses.

This is independent of #11's mask-setting identity fix and composes with it. No models, dependencies, output files, or ComfyUI node packages changed. GPU rendering and full checkpoint-resume execution are not claimed. Prepared with AI assistance.

The workflow builder passes `node --check`; its execution inside a live ComfyUI editor was not tested.

Current exact head: `d8bd5160854e3ccf3f4a4bc348ad831e0512ce5a`. Combined local integration of #4, #11, #14 and #15 merges cleanly and passes all 134 tests, TypeScript checking and the Webpack production build (integration head `cac56458991e0bf7dd7ca6596d1d8a2c1bed6abc`). This is a separate local integration check, not an upstream merge. As of `2026-10-28T17:59:53.948656+00:00`, GitHub reports no visible check rollup or review decision for this PR; hosted CI success and maintainer approval are not claimed.



### #13 [open] PR: fix(editor): isolate cancellation from later jobs
URL: https://github.com/D3voz/openjutsu/pull/13  by rudycelekli 2026-10-28T17:51:47Z

A cancellation request can finish after the user goes Back and reopens another recent job. Its result then replaces the new job with `Canceled`, or its failure appears on the unrelated job. Record the editor's run generation when cancelling and apply the result/error only while that generation remains current.

Two mounted React DOM regressions defer the original DELETE response, reopen a different job and then resolve the old success/error. Both fail on the original component and pass with the guard. A control verifies cancelling the current job still works.

Validation: `npm test` passes 129 tests and TypeScript; normal `npm run build` and `git diff --check` pass on macOS arm64 / Node 24.22.1. Uses the same shared dev-only DOM harness commit as #7, plus controlled storage/events/requests; this production fix does not depend on #7's video fix and can merge in either order. No GPU or ComfyUI rendering qualification is claimed. Prepared with OpenAI Codex assistance.



### #12 [open] PR: Validate all proxy patches before changing package files
URL: https://github.com/D3voz/openjutsu/pull/12  by rudycelekli 2026-10-28T17:51:26Z

The proxy patcher currently writes app.py before validating realtime.py. If the latter is incompatible or partially upgraded, the script aborts after leaving the two sides of JobEventBridge's constructor contract inconsistent. Prepare and validate every replacement before writing either file.

Validation: the unsupported-second-file regression fails before the change; both Python3.13 standard-library unittest fixtures pass afterward, including a complete supported patch and an idempotent rerun. All 126 Node tests, TypeScript checks and a production webpack build pass. Run the Python fixtures with `python -m unittest discover -s test -p test_proxy_patch.py` using Python3.10+. These are synthetic package-source fixtures, not an installed proxy/GPU execution test. A failure during actual filesystem writes is not made transactional by this preflight.

Prepared with AI assistance. Default Turbopack cannot follow shared local dependencies outside the isolated worktree; upstream CI keeps the normal build.



### #11 [open] PR: Separate saved swap clips by mask settings
URL: https://github.com/D3voz/openjutsu/pull/11  by rudycelekli 2026-10-28T17:50:50Z

Changing mask settings can reuse the same saved swap clips: the native Swap Plan hashes source media, prompts, dimensions, steps and seed, while hat fix/mask growth/SAM target live elsewhere in the graph. The exports leave the external generation fingerprint blank. Consequently a rerun can resume clips generated with a different mask.

Append a deterministic mask-settings namespace to the actual Production Plan run name for Simple, Advanced (including per-person and Montage), and Object generation. The existing motion/model suffix chains remain connected. All mask configurations, including defaults, receive the namespace so legacy clips whose mask configuration was never recorded cannot be mistaken for a qualified match; their files are preserved.

Validation:

- Three export-based regressions fail before the fix and pass afterward, covering every mask setting above and deterministic same-settings resume.
- All 129 tests and TypeScript checking pass; `npm run build -- --webpack` passes. Webpack supports the external node_modules symlink used by the isolated worktree; default Turbopack does not.
- The actual upstream CPU planning functions were extracted unchanged from [comfyui-h3-oneclick-swap at 9cf2d1a](https://github.com/D3voz/comfyui-h3-oneclick-swap/blob/9cf2d1a5e6f5f4e5fc3c11a82d9bc897e85085da/__init__.py#L238) and executed on real torch tensors, producing `swap_c971cf1fb7`; mask settings are absent from that planning signature. The [Context Loop contract](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/43afeffcc570beb5f7382593e79e4b211c7e02f6/chain_nodes.py#L15392) calls out the need to identify generation inputs outside the plan.

The regression resolves actual StringConcatenate graph edges over the same native base identity. This verifies graph submission and identity separation; it does not claim a GPU render or checkpoint-resume end-to-end run. No models, runtime dependencies, output files, or native node packages are changed. Prepared with AI assistance.

Current exact head: `ca0f0e29dc5e32e8fb377e2e55ccf8b332eb5d59`. Combined local integration of #4, #11, #14 and #15 merges cleanly and passes all 134 tests, TypeScript checking and the Webpack production build (integration head `cac56458991e0bf7dd7ca6596d1d8a2c1bed6abc`). This is a separate local integration check, not an upstream merge. As of `2026-10-28T17:59:53.948656+00:00`, GitHub reports no visible check rollup or review decision for this PR; hosted CI success and maintainer approval are not claimed.



### #10 [open] PR: Distinguish invalid requests, missing previews and backend outages
URL: https://github.com/D3voz/openjutsu/pull/10  by rudycelekli 2026-10-28T17:50:09Z

Malformed picture JSON currently returns502 claiming ComfyUI is unreachable, while a preview-asset backend503 is swallowed and becomes404. Validate picture JSON before backend access and suppress only the SDK's typed asset-not-found error during preview lookup.

Validation: malformed JSON and upstream503 regressions fail before the change; all 129 tests and TypeScript checks pass afterward. Controls retain genuine404 and reject null/array/string JSON without backend calls. Production webpack build passes; default Turbopack cannot follow the shared dependency symlink used in the isolated local worktree.

Prepared with AI assistance. These tests simulate backend responses; no GPU execution claimed.



### #9 [open] PR: Keep healthy tone presets when another preset fails
URL: https://github.com/D3voz/openjutsu/pull/9  by rudycelekli 2026-10-28T17:49:03Z

One preset file returning404 or invalid settings currently makes the entire optional tone-preset list empty, hiding healthy saved presets. Isolate each preset fetch and discard only failed/malformed entries while retaining the original order of healthy presets.

Validation: the mixed healthy/missing/malformed regression fails before the change; all 128 tests and TypeScript checks pass after it. A wholly unavailable optional preset service still returns an empty list. Production webpack build passes; default Turbopack cannot resolve shared dependencies outside the isolated local worktree. No GPU execution claimed.

Prepared with AI assistance.



### #8 [open] PR: Show jobs missing from the raw event stream as expired
URL: https://github.com/D3voz/openjutsu/pull/8  by rudycelekli 2026-10-28T17:47:24Z

If a job disappears after the initial snapshot but before the raw event connection opens, the low-level SDK throws its own NotFound class. The route catches only the high-level SDK class, so it sends a generic stream error instead of the terminal expired state. Recognize both typed exceptions.

Validation: the raw-event404 regression fails before the fix; all 129 tests and TypeScript checks pass afterward. Controls cover a missing initial snapshot and a503 outage that must remain an error rather than being mislabeled expired. Production webpack build passes; default Turbopack cannot resolve the shared dependency symlink in the isolated local worktree. No GPU execution claimed.

Prepared with AI assistance.



### #7 [open] PR: fix(editor): ignore late metadata from replaced videos
URL: https://github.com/D3voz/openjutsu/pull/7  by rudycelekli 2026-10-28T17:47:02Z

Replacing a video while its metadata is still loading can let the old probe reset the new video's selected section. The editor can display a 20-second source while submitting a 90-second section taken from the replaced file. A selection token now prevents an old probe from changing either the source metadata or its section.

The regression mounts the actual editor with React DOM, selects two files, and delivers their metadata out of order. It fails on the original component and passes with the guard. JSDOM and its types are dev-only additions for this browser-state regression; TSX is compiled with Next's existing SWC bindings. The shared harness is commit `34af650` so later independent editor fixes can reuse it without depending on this production change.

Validation on macOS arm64 / Node 24.22.1:
- `npm test`: 127 passed, including TypeScript validation.
- `npm run build`: passed with the normal production build.
- `git diff --check`: passed.

Metadata events and model-list responses are controlled fixtures. This tests the mounted component's state transitions, not native video decoding or ComfyUI/GPU rendering. Prepared with OpenAI Codex assistance; no release or deployment performed.



### #6 [open] PR: Preserve HTTP416 for unsatisfiable media ranges
URL: https://github.com/D3voz/openjutsu/pull/6  by rudycelekli 2026-10-28T17:46:07Z

An unsatisfiable video/image byte range is currently converted from upstream HTTP416 into a generic HTTP502. Preserve 416 and its Content-Range in both job-output and picture media routes, while discarding the upstream error body.

Validation: both media-route regressions fail before the change; all 128 tests and TypeScript checks pass after it. The tests check URL discovery, the forwarded Range header, `bytes */5`, an empty error body, and that upstream HTML is not exposed as media. Existing206 seeking tests remain passing. Production webpack build passes; the normal Turbopack build is left to CI because shared local dependencies live outside the isolated worktree.

Prepared with AI assistance; no GPU execution claimed.



### #5 [open] PR: Support Unicode output download filenames
URL: https://github.com/D3voz/openjutsu/pull/5  by rudycelekli 2026-10-28T17:44:57Z

Downloading an output named `結果.mp4` currently throws when the Unicode name is put directly into the Content-Disposition header, returning HTTP502. Emit an ASCII fallback and an RFC5987 UTF-8 filename instead, with path segments and control characters removed from the suggested filename.

Validation: the regression fails with the original ByteString error; all 127 tests and TypeScript checks pass after the change. The test round-trips the Unicode name and quotes from the extended filename and checks the body survives. A production webpack build passes. Default Turbopack cannot follow the shared dependency symlink used in the isolated worktree; normal upstream CI is unchanged. No GPU execution claimed.

Prepared with AI assistance.



### #4 [open] PR: Bound custom From picture video render dimensions
URL: https://github.com/D3voz/openjutsu/pull/4  by rudycelekli 2026-10-28T17:44:26Z

Custom From picture video settings accept dimensions such as 32×512 or 10000×512 and forward them to the render graph, bypassing the 256–2048 px custom-canvas bounds used by the main generation route. Reuse the shared step/bounds after snapping, preserving existing 650×970 → 640×960 behavior.

Validation: the new boundary regression fails before the fix; all 128 tests and TypeScript checking pass after it. The second regression loads the actual H3 and Solar workflow exports and verifies both conditioning and motion-reference canvases. `npm run build -- --webpack` passes. Default Turbopack cannot follow the external node_modules symlink in this isolated worktree, so the production build used Webpack; no GPU/provider execution is claimed.

Prepared with AI assistance.

Current exact head: `bb0b75550f25c287fc733b01b57858a3153ff5aa`. Combined local integration of #4, #11, #14 and #15 merges cleanly and passes all 134 tests, TypeScript checking and the Webpack production build (integration head `cac56458991e0bf7dd7ca6596d1d8a2c1bed6abc`). This is a separate local integration check, not an upstream merge. As of `2026-10-28T17:59:53.948656+00:00`, GitHub reports no visible check rollup or review decision for this PR; hosted CI success and maintainer approval are not claimed.



### #3 [open] PR: Protect deduplicated inputs from concurrent job cleanup
URL: https://github.com/D3voz/openjutsu/pull/3  by rudycelekli 2026-10-28T17:43:45Z

When one job finishes while a second request is resolving the same deduplicated upload, cleanup can delete the asset before the second job holds it. The second submission then fails with an unresolvable ASSET reference. Serialize input commit+hold with release so deletion cannot invalidate an acquisition in flight.

This serializes input network operations across upload requests; GPU jobs remain concurrent. The SDK bounds these operations with its existing 30-second request timeout. The queue unlocks on acquisition/submission failures.

Validation: a real loopback fake proxy reproduces the race before the change; all 127 tests and TypeScript checks pass afterward. Regression controls cover final cleanup, failed acquisition, rejected submission and subsequent recovery without orphaned inputs. Production webpack build passes; default Turbopack cannot follow shared dependencies outside this isolated worktree. No actual GPU execution claimed.

Prepared with AI assistance.



### #2 [open] PR: Preserve multipart upload bytes across stream boundaries
URL: https://github.com/D3voz/openjutsu/pull/2  by rudycelekli 2026-10-28T17:42:19Z

Multipart payloads containing a boundary prefix followed by ordinary data are currently rejected. Short FileHandle writes also silently truncate uploads, and the header cap depends on incoming chunk size. This waits for a valid delimiter suffix, completes partial writes, and applies the header cap to complete headers too.

Validation: the integrity regressions fail before the change; all 130 Node tests and TypeScript checks pass on both Node22.19 and Node24.22.1 after it. A production webpack build passes. The default Turbopack build cannot resolve the shared dependency symlink in the isolated local worktree; CI retains the normal build. No GPU or ComfyUI execution claimed.

Prepared with AI assistance. Regression evidence includes real temp files and a permitted short-write simulation.

The existing fake proxy subtracted a terminal CRLF that Node22 does not append, causing the unchanged base to report file sizes two bytes too small. Its bookkeeping now accepts both multipart endings; independent controls cover 0/1/5/100000-byte payloads with and without CRLF. This changes fixture measurement, not upload bytes.



### #1 [closed] PR: KAFEIN fleet profile: 8-step turbo LoRA, research-mode gates, license docs
URL: https://github.com/D3voz/openjutsu/pull/1  by leolambertini 2026-10-27T22:14:38Z

## KAFEIN fleet profile — reviewable fork (no deploy)

Closes the `kafein/fleet-absorption` fork into a reviewable deliverable. Implementation phase only: nothing is deployed, `main` is untouched.

### Profile (zero new base models)
Fleet defaults in `lib/model.ts` now match the weights actually in S3:

| Component | Before | After (fleet) |
| --- | --- | --- |
| Base model | H3 ref2va pruned INT8 | **unchanged** |
| LoRA | DMAD 4-step full (rank 39) | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16` |
| Steps | 6 | 8 |
| Text encoder | Qwen 32B INT4 | Qwen 32B NVFP4 AWQ |
| Describer | Qwen3-VL 8B INT8 | Qwen3-VL 8B FP8 scaled |

- The LoRA node title stays `"DMAD LoRA"` — `applyModel()` and every workflow export resolve it by title (comment in `lib/model.ts`).
- All workflow exports (`workflows/*.api.json`) and the build scripts under `workflows/build/` were regenerated to the fleet profile; `test/model.test.ts` verifies the exports match the defaults.
- **One LoRA, justified:** the 8-step turbo is a fleet weight already in S3 (replaces DMAD); it is *not* a new base model.

### Settings migration (no rewrites)
`migrateFleetSettings()` in `lib/model.ts` renames saved upstream names (DMAD LoRA, INT4 encoder, INT8 describer) to the fleet equivalents and lifts 6→8 steps when the DMAD LoRA was active. Custom LoRA strengths survive the migration (covered by `test/model.test.ts`).

### License gates (default OFF)
- **Fix faces** (InsightFace NC + YOLOv8 AGPL-3.0) and **From picture** (FLUX 3 Klein NC) are hidden in the UI (`welcome.tsx`, `from-picture.tsx`) and server-blocked with 403 (`app/api/jobs/route.ts`, `app/api/picture/route.ts`) unless `KAFEIN_RESEARCH_MODES=1` + `NEXT_PUBLIC_KAFEIN_RESEARCH_MODES=1`.
- `lib/license.ts` is the single policy source; `test/license.test.ts` covers the default-off behaviour; route tests cover both 403s.
- Core modes (Simple/Advanced/Object/Montage/Find people/Check mask/Fit/Fix colours) add **no new licenses** (H3 ref2va, SAM 3.1, Qwen3-VL, fleet VAEs already accounted for).

### Packaging
Next 16.4.3 lineup; `npm audit --audit-level=high` → **0 vulnerabilities**; CI already runs the audit.

### Docs
- `KAFEIN.md` (new): profile delta, gate policy, S3 status — 13/32 models registered in `licenses.ts`; R1 must close 4 `ModelLicense` gaps (H3 ref2va NVFP4, H3 fl2va NVFP4, LoRA taomate_fl2va_3step, and the turbo 8-step LoRA this profile uses).
- `README.md`: KAFEIN edition note + fleet model table; upstream text otherwise left intact (no app rewrite).

### Tests & verification (local)
- `npm test` (node --test + tsc): **128/128 pass**
- `npm run build`: clean
- `npm audit --audit-level=high`: 0 vulnerabilities
- `git diff --check`: clean

### Rollback
Revert the single commit; no DB migrations, no new base models, no env vars required to go back.

