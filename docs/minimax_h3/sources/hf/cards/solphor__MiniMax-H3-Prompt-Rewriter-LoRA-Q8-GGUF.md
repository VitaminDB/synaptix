# solphor/MiniMax-H3-Prompt-Rewriter-LoRA-Q8-GGUF

Источник: https://huggingface.co/solphor/MiniMax-H3-Prompt-Rewriter-LoRA-Q8-GGUF · скачано 2026-11-28
Загрузок: 106 · лайков: 1 · базовая: ["lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA","Qwen/Qwen3.6-27B"] · теги: gguf,lora · обновлено: 2026-08-10T15:42:17.000Z

---

---
base_model:
- lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA
- Qwen/Qwen3.6-27B
library_name: peft
tags:
- lora
- gguf
- llama.cpp
- prompt-rewriting
- minimax-h3
- text-to-audio-video
language:
- en
---

# gguf conversion of lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA

This is a GGUF conversion of [lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA)

## Files

| file | precision | size |
|---|---|---|
| `qwen36-27b-rewriter-f16.gguf` | F16 | 3.7 GB |
| `qwen36-27b-rewriter-q8_0.gguf` | Q8_0 | 2.0 GB |
| `convert_lora_qwen35.py` | — | conversion script |
| `base_config.json` | — | Qwen3.6-27B config (for `--base`) |

## Usage

Load with `llama-server` against a `qwen35` base GGUF (e.g. `Qwen3.6-27B-IQ4_XS.gguf`):

```bash
llama-server -m Qwen3.6-27B-IQ4_XS.gguf \
  --lora qwen36-27b-rewriter-q8_0.gguf \
  -c 32768 -ngl 999 --jinja
```

Send a chat completion with the system prompt from `prompt_template.py`:

```json
{
  "messages": [
    {"role": "system", "content": "content": "You are a professional prompt rewriter for joint audio-video generation.
Rewrite the user's original prompt into one coherent, production-ready multimodal description for the requested output aspect ratio and duration.

Return only these three fields, in this exact order:
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...

Requirements:
- Expand the visual narrative into clearly numbered shots such as [Shot 1], [Shot 2], and include timestamps for cuts after the first shot when useful.
- Make the number, timing, and pacing of shots appropriate for the requested duration.
- Compose the scene for the requested aspect ratio.
- Preserve the user's intent while adding concrete subjects, appearance, environment, lighting, composition, camera movement, physical motion, and temporal continuity.
- Keep characters, objects, wardrobe, locations, and spatial relationships consistent across shots.
- Describe synchronized diegetic audio in overall_soundscape and external score in non_diegetic_music.
- Do not add explanations, Markdown fences, safety commentary, or fields other than the three requested fields."},
    {"role": "user", "content": "resolution: 16:9
duration: 10s
original_prompt: A red fox walks through a snowy forest at dawn."}
  ],
  "temperature": 0.7,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

## Speed

IQ4_XS base, `-c 32768`, `-ctk/-ctv q8_0`, `-ngl 999`, `-b 2048 -ub 512 -np 1`, no spec-decoding. 3263-token prompt, 256 tokens generated, warmup discarded, 3 runs averaged. RTX 3090

| config | prompt (t/s) | vs base | generation (t/s) | vs base |
|---|---:|---:|---:|---:|
| no LoRA | 1238.4 | 100% | 40.28 | 100% |
| LoRA f16 | 896.3 | 72.4% | 27.89 | 69.2% |
| LoRA q8_0 | 906.8 | 73.2% | 30.87 | 76.6% |

The adapter targets every matmul in all 64 layers (992 tensors, r=256), so each becomes three matmuls — ~27% prompt and ~23–31% generation overhead.
q8_0 is ~11% faster than f16 and 1.75 GB less VRAM, with barelly any difference.

## Output comparison

Same prompt, `temperature: 0`, `seed: 42`.

### no LoRA

> integrated_multimodal_description: [Shot 1] (0s-4s) A wide, cinematic 16:9 shot captures a serene, snow-covered pine forest at dawn. Soft, cool blue ambient light filters through the dense trees, contrasting with the warm, golden hues of the rising sun peeking through the branches in the background. A vibrant red fox, its fur thick and pristine white-tipped, walks gracefully from the left mid-ground towards the right foreground, its paws sinking slightly into the fresh powder. The camera tracks slowly alongside the fox, maintaining a low angle to emphasize the texture of the snow and the animal's alert posture. [Shot 2] (4s-10s) The camera cuts to a medium close-up of the fox as it pauses, turning its head slightly to look directly into the lens. Its amber eyes reflect the morning light, and its nose twitches as it sniffs the cold air. Snowflakes gently drift down around it. The background is softly blurred (bokeh), highlighting the fox's detailed fur and the steam rising from its breath.
>
> overall_soundscape: The crisp, muffled crunch of snow under the fox's paws is prominent and rhythmic. There is a subtle, high-frequency wind whistling through the pine needles. Occasional distant bird calls add depth to the forest atmosphere. The sound of the fox's breath is audible as a soft hiss in the cold air.
>
> non_diegetic_music: A minimal, ambient piano melody with slow, resonant notes that evoke a sense of solitude and tranquility. The music swells slightly as the fox looks at the camera, then fades out gently as it walks away, leaving only the natural soundscape.

### LoRA f16

> integrated_multimodal_description: [Shot 1] Cinematic, a medium-wide tracking shot follows a vibrant red fox as it walks steadily from left to right through a dense, snow-blanketed pine forest at dawn. The adult fox features thick, bright rust-orange fur contrasted by a stark white chest, jaw, and the prominent white tip of its bushy tail. Its dark, pointed ears are perked up, and its glossy black nose twitches slightly as it navigates the terrain. The environment is enveloped in the cool, soft blue-hour light of early morning, with pale violet and warm amber rays piercing through the tall, dark green cedar trees in the background, highlighting gently drifting snow crystals in the crisp air. The ground is heavily layered with untouched, powdery snow, interrupted only by the deep, rhythmic footprints the fox leaves behind. As the camera tracks alongside the animal, the fox briefly pauses, turning its head slightly toward the camera to sniff the air, before lowering its head and continuing its steady, silent march through the tranquil winter landscape.
>
> overall_soundscape: A quiet, ambient winter morning tone features a gentle, continuous breeze rustling softly through the pine needles. In the foreground, the distinct, rhythmic crunch and soft thud of padded paws pressing into deep, fresh snow are clearly heard with every step the fox takes, punctuated by a subtle, sharp sniff as the animal pauses to smell the air.
>
> non_diegetic_music: A serene, slow-tempo ambient orchestral piece featuring a gentle, sustained cello melody layered over soft, atmospheric violin textures, creating a tranquil and mystical morning mood without any percussion.

### LoRA q8_0

> integrated_multimodal_description: [Shot 1] Cinematic, a medium wide tracking shot follows a red fox as it walks steadily from left to right through a dense, snow-covered pine forest at dawn. The adult fox features vibrant rust-orange fur dusted with tiny, undisturbed snowflakes, a bright white chest, and dark black lower legs. It navigates a softly packed snowbed, its padded paws pressing deeply into the powder. Towering, dark-barked pine trees rise in the midground and background, their heavy branches laden with thick, fresh white snow. Pale, cool-toned dawn light filters diagonally through the dense timber from the upper left, casting long, crisp blue shadows across the snowy forest floor while illuminating the floating frost particles in the crisp air. The camera smoothly tracks the animal's horizontal movement, keeping it centered in the frame. As the fox progresses, its pointed ears twitch slightly, and its bushy, white-tipped tail sways in rhythm with its deliberate strides.
>
> overall_soundscape: A steady, gentle wind hisses softly through the pine branches, establishing a quiet, frozen ambience. In the foreground, the distinct, rhythmic crunch and soft thud of the fox's paws sinking into the fresh snow are clearly heard with every step. A faint, brief sniff is audible near the end as the animal pauses to smell the air.
>
> non_diegetic_music: A solo cello plays a slow, sustained melody with long, bowing notes, accompanied by sparse, resonant acoustic guitar plucks underneath, maintaining a steady, quiet dynamic throughout without any sudden swells.

## Reproduce

The upstream PEFT adapter cannot be converted with stock (convert_lora_to_gguf.py)[https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py] from llama.cpp repo — the qwen35 architecture reorders linear-attention V heads, and `LoraTorchTensor.reshape` rejects the dim=1 `out_proj` reorder. The included `convert_lora_qwen35.py` includes the fix to reproduce.
