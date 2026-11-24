# MOSS-Audio-8B-Thinking (OpenMOSS)

Аудио-LLM: энкодер MOSS (Conv2d-стем ÷8 + 32 слоя Whisper, d=1280, куски по 4 с), адаптер GatedMLP
1280→8192→4096, deepstack — выходы слоёв энкодера 8/16/24 своими GatedMLP прибавляются к аудиопозициям после
блоков LLM 0/1/2; LLM — Qwen3-8B (обычный, в гибридном пайплайне с `plain_norm`). Код оригинала:
github.com/OpenMOSS/MOSS-Audio (`src/modeling_moss_audio.py`, `processing_moss_audio.py`), в HF-репозитории
его нет; нет и `tokenizer.json` (собран из vocab/merges в `_hf/MOSS-Audio-8B-Thinking-aux`).

## Бандл

```sh
syn-pack -o llm/moss-audio-8b-thinking-bf16.syn --id moss-audio-8b-thinking --arch moss_audio \
  --component main:_hf/MOSS-Audio-8B-Thinking --files _hf/MOSS-Audio-8B-Thinking-aux
synaptix quantize …-bf16.syn llm/moss-audio-8b-thinking.syn --format mxfp8 --rule audio_=none --rule deepstack_=none
```

10,3 ГБ: LLM MXFP8 (7,8 ГБ весов), башня BF16 (грузится на время кодирования). Декод ~56 ток/с (MXFP8 GEMV
≈0,5 ТБ/с; NVFP4 для MLP ускорил бы ~вдвое ценой качества).

## Вход (как у оригинала)

Звук — torchaudio: среднее каналов, `resample` sinc-hann (повторён бит в бит, `resample_torchaudio`); log-mel
Whisper 128 полос по всей длине (без добивки до 30 с) в bf16; 12,5 токена/с; промпт процессора:
system «You are a helpful assistant.», `<|audio_bos|>` + `<|AUDIO|>`… с метками секунд цифрами через 2 с +
`<|audio_eos|>` (в словаре это `<|system|>`, `<|vision_pad|>`, `<|user|>`), `\n` и вопрос; `<think>` модель
открывает сама (`enable_thinking=false` в шаблоне закрывает его заранее).

## Сверка (24.11.2026)

Звук rel 7e-8, mel 3e-5, input_ids совпадают (Bass_Drop 75, Bubble 184 — три куска энкодера и метки), выход
башни и deepstack cos ≥ 0,9997 (rel ~1 %, BF16). Жадная генерация совпадает с эталоном F32 (transformers,
teacher forcing) до почти-ничьих: наш токен там — его №2 с отрывом 0,55 и 0,08 ната.

Эталон под transformers 5.22 требует заплатки: `WhisperEncoderLayer` возвращает тензор, а не кортеж, и
`generate` передаёт аудио на каждом шаге — генерация вручную с KV.

## Вызов

```sh
MEDIA=звук.wav MEDIA_NOMARK=1 THINK=1 GEN_MAX=2500 synaptixctl gen moss-audio-8b-thinking "<промпт>"
```

По-русски модель пишет слабо (вкрапления китайского, обрывки английских слов) — лучше просить ответ
по-английски.
