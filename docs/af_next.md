# Audio Flamingo Next Captioner (NVIDIA, `model_type = musicflamingo`)

`nvidia/audio-flamingo-next-captioner-hf` (лицензия — некоммерческая, исследовательская). Тот же код подходит к
`audio-flamingo-next-hf` (instruct) и `-think-hf` — архитектура одна.

Башня AF3: энкодер Whisper-large (32 слоя, d=1280, обучаемые позиции) по окнам 30 с, внимание только по валидным
кадрам (маска оригинала = обрезка после стема), AvgPool ÷2, LayerNorm; RoTE на первых 512 измерениях (оси
«номер окна» и «кадр в окне», углы `−2π·t` абсолютного времени, таблицы f32, поворот f64); проектор
Linear–GELU–Linear в 3584. LLM — Qwen2.5-7B (смещения q/k/v, без qk-норм) в гибридном пайплайне.

## Бандл

```sh
syn-pack _hf/audio-flamingo-next-captioner-hf -o llm/af-next-captioner-bf16.syn --id af-next-captioner
synaptix quantize …-bf16.syn llm/af-next-captioner.syn --format mxfp8 --rule audio_tower.=none --rule multi_modal_projector.=none
```

9,2 ГБ. Профиль (`optimal_profile_blackwell`): счёт, эмбеддинги и **KV в BF16** — MXFP8-KV съедает выбросы K у
Qwen2 (смещения `k_proj`) и переворачивал ответ: бас-дроп «high-pitched beep» вместо «low sustained bass note».
Башня кодирует в F32 и на карте (BF16 давал у ~1 % строк cos до 0,3), 0,9 с на 43 с звука. Декод ~58 ток/с.

## Вход

Звук — ffmpeg моно 16 кГц (torchcodec у оригинала), окна по 30 с, log-mel Whisper с добивкой нулями, маска
`ceil(n/160)` кадров, токенов окна `((L−1)/2+1−2)/2+1`. Промпт шаблона: системная реплика AF-Next (своя
по умолчанию), `<|sound_bos|>` + `<sound>`×N + `<|sound_eos|>` и вопрос без `\n`. README советует
`repetition_penalty=1.2`.

## Сверка с transformers (24.11.2026)

input_ids совпадают (211; 1182 на 43-с звуке из двух окон); выход башни — rel 6e-5 (F32 на карте), 2e-5 на CPU;
жадный ответ BF16-бандла совпадает с эталоном дословно («A low-pitched, resonant, and sustained musical
note…»), MXFP8 — по смыслу («A low, sustained synth bass note…»).

## Вызов

```sh
MEDIA=звук.wav MEDIA_NOMARK=1 GEN_MAX=800 synaptixctl gen af-next-captioner "<промпт>"
```

Пишет по-английски даже при просьбе ответить по-русски (вкрапления китайского); размышлений у captioner нет.
