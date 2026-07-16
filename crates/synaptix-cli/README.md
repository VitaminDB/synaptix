# synaptix CLI

Единый CLI инференса synaptix: LLM (run/chat/bench), картинки (imagine: SDXL/FLUX),
видео (video: LTX-2.3 — все режимы), ASR (transcribe: Whisper), инспекция/конвертация
весов.

```bash
cargo build --release -p synaptix-cli --bin synaptix --features cuda
target/release/synaptix <команда> --help
```

---

## Команды

| Команда | Что делает |
|---|---|
| `inspect` | Просмотр содержимого safetensors/syn/ckpt |
| `convert` | Конвертация весов (→ syn и др.) |
| `bench` | Бенчмарк LLM (tok/s) |
| `run` | Одноразовая LLM-генерация |
| `chat` | Интерактивный LLM-чат |
| `diff` | Сравнение двух файлов весов |
| `train` | LoRA-тренировка |
| `transcribe` | Whisper ASR: WAV → текст |
| `imagine` | Текст → картинка (SDXL / FLUX, авто-детект) |
| `video` | Текст/фото/видео/аудио → видео (LTX-2.3, все режимы) |

---

## video — LTX-2.3 (все режимы)

```
synaptix video <MODEL.safetensors> "<PROMPT>" [флаги]
```

`MODEL` — единый чекпойнт LTX-2.3 (DiT+VAE+audio-VAE+vocoder+проекции):
`ltx-2.3-22b-distilled-*.safetensors` (быстрый, 8+3 шага) или
`ltx-2.3-22b-dev.safetensors` (полный, нужен guidance).

### Общие флаги

| Флаг | Дефолт | Описание |
|---|---|---|
| `-o, --output` | `out.mp4` | Выходной mp4 (видео+звук) |
| `--gemma <dir>` | `models/gemma-3-12b-qat` | Текст-энкодер Gemma-3-12B |
| `--frames N` | 105 | Кадры; округляется к `8k+1` (121 ≈ 5с @24fps) |
| `--width / --height` | 1024×576 | Целевое разрешение. two-stage: кратно 64 |
| `--fps` | 24 | Кадров/с |
| `--no-audio` | — | Отключить аудио-поток |
| `--device` | `cuda` | `cpu` \| `cuda` |
| `--compute-dtype` | `bf16` | `f16` \| `bf16` \| `f32` |
| `--quant-transformer` | none | `mxfp8`/`nvfp4` → DiT квантуется и живёт **резидентно** (быстро, GPU ~96%); `none` → dense bf16 + **offload-стриминг** (медленнее, но VRAM свободен) |
| `--quant-encoder` | `mxfp8` | Квант Gemma (12B → ~12GB) |

**Выбор кванта**: видео-only → `--quant-transformer mxfp8` (быстро).
Аудио-режимы (av/two-stage со звуком/lipdub) → **без** кванта (AvDit ≈21GB
резидентно не влезает, offload обязателен). HD VAE-decode сам освобождает DiT
(дроп перед декодом) — HD работает в обоих режимах.

### Пайплайны (`--pipeline`, `--list-pipelines`)

| Имя | Статус | Что |
|---|---|---|
| `one-stage` | ✓ | txt→video, одна стадия на целевом разрешении |
| `two-stage` | ✓ | txt→video+audio: stage1 (½ разрешения) → upscaler ×2 → stage2-refine |
| `av` | ✓ | txt→video+audio, одна стадия (joint AvDit) |
| `ti2v-two-stage`, `a2v`, `keyframe`, `ic-lora`, `hdr-ic-lora`, `retake`, `lipdub` | реестр | Фактически доступны через флаги ниже |

Без `--pipeline`: `--two-stage` → two-stage; `--no-audio` → one-stage; иначе av.

Two-stage: выход = `(width, height)`, оба кратны 64; stage1 генерится на половине.
`--no-refine` — пропустить stage2 (быстрее, мягче). Требует
`--upscaler <ltx-2.3-spatial-upscaler-x2-*.safetensors>`.

### LoRA (`--lora <file>`)

| Флаг | Дефолт | Описание |
|---|---|---|
| `--lora` | — | LoRA-адаптер (мерджится в веса DiT при загрузке) |
| `--lora-strength` | 1.0 | Сила (одностадийные пути; для two-stage = дефолт stage2) |
| `--lora-strength-stage1` | **0.0** | two-stage: сила на stage1 |
| `--lora-strength-stage2` | =`--lora-strength` | two-stage: сила на stage2-refine |

**⚠ Правила distilled-lora-384** (выяснено экспериментально + официальный args.py):
- **distilled-чекпойнт: БЕЗ этой LoRA** (официальный distilled-пайплайн её не
  принимает; strength 1.0 на обе стадии = накапливающаяся деградация на длинных
  видео — «расплавленные» лица).
- **dev-чекпойнт + guidance: LoRA только stage2**, ~0.8 (дефолтное поведение:
  stage1=0).
- Официальный HQ-паттерн (dev): stage1 0.25 / stage2 0.5.

### Guidance (dev-чекпойнт)

| Флаг | Дефолт | Описание |
|---|---|---|
| `--cfg-scale` | 1.0 (выкл) | CFG; официально 3.0 |
| `--stg-scale` | 0.0 (выкл) | Spatio-temporal guidance; официально 1.0 |
| `--negative-prompt` | "" | Негатив для CFG |
| `--steps` | 30 | Шаги guided stage1 (LTX2Scheduler; официально 40) |

`cfg>1 || stg>0` → guided two-stage (только видео): stage1 = dev+CFG/STG,
stage2 = distilled-LoRA refine. Все guidance-проходы идут одним стрим-свипом
блоков (forward_multi).

### Conditioning-режимы

| Режим | Флаги | Описание |
|---|---|---|
| **image→video** | `--image <png/jpg>` [`--image-strength 1.0`] | Фото = кадр 0, дальше движение. Работает и в two-stage (HQ) |
| **keyframe** | `--image <...> --image-frame N` | Фото притягивает кадр N (append-механизм), N — пиксель-кадр |
| **retake** | `--video <src.mp4> --retake-start S --retake-end E` | Перегенерация региона `[S,E]` секунд исходного видео (остальное frozen). Промпт описывает новый контент региона |
| **IC-LoRA v2v** | `--ref-video <control.mp4> --ref-downscale {1,2} --ref-strength 1.0` + `--lora <ic-lora>` | Control-видео (depth/pose/edges) ведёт генерацию. `ref-downscale` из метаданных LoRA (union/motion-track = 2, lipdub = 1) |
| **Canny-препроцесс** | `--ref-preprocess canny` [`--canny-low 0.1 --canny-high 0.2`] | Обычное видео → контурная карта на лету (для union-control; превью `/tmp/synaptix_canny_f0.png`) |
| **Depth-препроцесс** | `--ref-preprocess depth` [`--depth-model <dir>`] | Видео → карты глубины (Depth Anything V2 Small, bit-exact порт; превью `/tmp/synaptix_depth_f0.png`) |
| **lipdub** | `--ref-video <лицо.mp4> --audio <речь.wav>` + `--lora <lipdub-ic-lora>` | Губы под аудио. wav любого SR (авто 16k). В mp4 муксится оригинальный звук |

### Примеры (проверенные)

```bash
M=models/ltx2.3_v1.1
BIN=target/release/synaptix

# 1) Быстрый HQ: distilled two-stage БЕЗ LoRA, видео+звук, 5с HD (~4.5 мин)
$BIN video $M/ltx-2.3-22b-distilled-1.1.safetensors \
  "a ripe red apple on a rustic wooden table, soft window light" \
  --pipeline two-stage --quant-transformer mxfp8 \
  --upscaler $M/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --frames 121 --width 1280 --height 704 -o apple.mp4

# 2) ЛУЧШЕЕ качество: dev + guidance + LoRA stage2 (~17 мин на 5с HD)
$BIN video $M/ltx-2.3-22b-dev.safetensors \
  "a close-up portrait of a man with a beard talking to the camera, photorealistic" \
  --pipeline two-stage --quant-transformer mxfp8 \
  --upscaler $M/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --lora $M/ltx-2.3-22b-distilled-lora-384-1.1.safetensors --lora-strength 0.8 \
  --cfg-scale 3.0 --stg-scale 1.0 --steps 40 \
  --negative-prompt "blurry, distorted, deformed face, low quality" \
  --frames 121 --width 1280 --height 704 -o man_hd.mp4

# 3) image→video: фото оживает (кадр 0 = фото)
$BIN video $M/ltx-2.3-22b-distilled-1.1.safetensors \
  "the camera slowly pushes in, gentle motion" \
  --quant-transformer mxfp8 --image photo.png \
  --frames 17 --width 512 --height 512 -o i2v.mp4

# 4) retake: перегенерить хвост видео под новый промпт
$BIN video $M/ltx-2.3-22b-distilled-1.1.safetensors \
  "a green apple on a wooden table" \
  --quant-transformer mxfp8 \
  --video src.mp4 --retake-start 0.35 --retake-end 1.0 \
  --frames 17 --width 512 --height 512 -o retake.mp4

# 5) union-control + canny/depth: структура из видео, контент из промпта
#    (--ref-preprocess canny → контуры; depth → карты глубины DAv2)
$BIN video $M/ltx-2.3-22b-distilled-1.1.safetensors \
  "a marble statue of a bearded man talking, museum lighting" \
  --quant-transformer mxfp8 \
  --lora $M/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors \
  --ref-video src.mp4 --ref-preprocess canny --ref-downscale 2 \
  --frames 49 --width 768 --height 448 -o statue.mp4

# 6) lipdub: лицо + речь → губы под аудио (AvDit → БЕЗ кванта)
$BIN video $M/ltx-2.3-22b-distilled-1.1.safetensors \
  "a man talking to the camera, lips moving in sync with speech" \
  --pipeline two-stage \
  --upscaler $M/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --lora $M/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors \
  --ref-video face.mp4 --audio speech.wav \
  --frames 81 --width 768 --height 448 -o lipdub.mp4
```

---

## imagine — SDXL / FLUX (текст → картинка)

```
synaptix imagine <MODEL_DIR> "<PROMPT>" [флаги]
```

`MODEL_DIR` — HF-директория (SDXL: `text_encoder/ unet/ vae/...`; FLUX
авто-детектится).

| Флаг | Дефолт | Описание |
|---|---|---|
| `-o, --output` | `out.png` | PNG-выход |
| `-n, --negative` | "" | Негативный промпт |
| `--steps` | 30 | Шаги денойза |
| `--cfg` | 5.0 | CFG (SDXL ~5-8) |
| `--width / --height` | 1024×1024 | Разрешение |
| `--seed` | 0 | Сид |
| `--device` | `cpu` | `cuda` сильно быстрее (SDXL 1024² ≈ 7с) |
| `--compute-dtype` | bf16(cuda)/f32(cpu) | VAE всегда F32 |
| `--quant` / `--storage-dtype` | none | `nvfp4`/`mxfp8`: FLUX 23GB → ~6/12GB резидентно |

```bash
synaptix imagine ~/models/sdxl "a photo of an apple" --device cuda -o apple.png
synaptix imagine ~/models/flux1-dev "a cat astronaut" --device cuda --quant nvfp4
```

---

## chat / run — LLM

```
synaptix chat <MODEL> [флаги]          # интерактивный чат
synaptix run  <MODEL> "<PROMPT>" [флаги]  # одна генерация
```

`MODEL` — safetensors/syn/ckpt (qwen3, qwen3.6-hybrid, gemma3 и др.).

| Флаг (общие) | Дефолт | Описание |
|---|---|---|
| `--quant` | none | Пресет: `nvfp4` \| `fp8` (mxfp8) |
| `--kv-dtype` | bf16 | `mxfp8` — block-scale KV (длинный контекст) |
| `--compute-dtype` | bf16 | Активации: `f16`/`bf16`/`f32` |
| `--storage-dtype` | — | Override весов attn+mlp |
| `--lm-head-dtype` / `--embed-dtype` | bf16 | Override головы/эмбеддингов |
| `--attn` | auto | `flash-decode` \| `fa2` \| `fa4` |
| `--device` | cuda | |
| `--temperature` | 1.0 / 0.7(chat) | |
| `--seed` | 0 (=время в chat) | |

`run` дополнительно: `--max-tokens 128`, `--max-seq` (KV+RoPE capacity),
`--no-graph` (CUDA-graph decode выкл), `--warmup`.

`chat` дополнительно: `--system`, `--context 4096`, `--prefill-batch 0`
(0 = single-shot), `--max-tokens 0` (без лимита), сэмплинг `--top-k 40
--top-p 1.0 --min-p 0 --repetition-penalty 1.0`, `--no-think`
(отключить reasoning у qwen3.6).

```bash
# флагман: NVFP4 веса + MXFP8 KV + CUDA-graph ≈ 90 tok/s на 27B
synaptix chat ~/models/qwen3.6-27b.safetensors --quant nvfp4 --kv-dtype mxfp8 --context 32768
```

---

## transcribe — Whisper ASR

```
synaptix transcribe <MODEL> <AUDIO.wav> [флаги]
```

| Флаг | Дефолт | Описание |
|---|---|---|
| `--language` | авто | ISO-639-1 (`ru`, `en`, ...) |
| `--task` | transcribe | `transcribe` \| `translate` (→ англ.) |
| `--timestamps` | — | Сегменты с таймкодами |
| `--device` | cpu | |
| `--compute-dtype` | f32 | `f16`/`bf16` |

---

## bench — LLM-бенч

```
synaptix bench <MODEL> [--n-tokens 128] [--prompt-tokens 0] [--batch-size 1]
               [--warmup 3] [--device cuda] [--attn auto] [--dtype bf16]
```

---

## inspect / convert / diff / train

```
synaptix inspect <FILE> [-v] [-f <фильтр-имени>]
synaptix convert <IN> <OUT> [--format syn] [--arch <a>] [--component <c>]
synaptix diff <A> <B> [--atol 1e-4] [--rtol 1e-3]
synaptix train <MODEL> <DATA> <OUT> [--lora-r 8] [--lora-alpha 16]
               [--lr 1e-4] [--epochs 3] [--batch-size 4]
```

---

## Модели LTX-2.3 (ожидаемая раскладка)

```
models/ltx2.3_v1.1/
  ltx-2.3-22b-distilled-1.1.safetensors        # быстрый (8+3 шага, без CFG)
  ltx-2.3-22b-dev.safetensors                  # полный (CFG/STG, лучшее качество)
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors  # для two-stage
  ltx-2.3-22b-distilled-lora-384-1.1.safetensors  # ТОЛЬКО для dev (stage2 ~0.8)
  ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors      # lipdub (ref-downscale 1)
  ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors      # depth/pose/edges (ref-downscale 2)
  ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors  # motion (ref-downscale 2)
models/gemma-3-12b-qat/           # текст-энкодер
models/depth-anything-v2-small/   # depth-препроцессор (DAv2-S hf)
```
