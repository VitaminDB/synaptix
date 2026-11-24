# Qwen3-Omni-30B-A3B-Captioner

Только звук на входе, текст на выходе (без промпта — карточка: «does not accept any text prompts»).
Thinker: энкодер AuT (`thinker.audio_tower.*`) + Qwen3-MoE 30B-A3B (48 слоёв, 128 экспертов / 8, qk-нормы,
M-RoPE — у звука все оси равны, т. е. обычные позиции) в гибридном пайплайне. Зрение в бандле есть
(`thinker.visual.*`), но не подключено.

## Бандл

```sh
syn-pack -o llm/qwen3-omni-captioner-bf16.syn --id qwen3-omni-30b-a3b-captioner --arch qwen3_omni_moe \
  --component main:_hf/Qwen3-Omni-30B-A3B-Captioner --files _hf/Qwen3-Omni-30B-A3B-Captioner-aux
synaptix quantize …-bf16.syn llm/qwen3-omni-30b-a3b-captioner.syn --policy llm \
  --rule thinker.audio_tower.=none --rule thinker.visual.=none
```

20,9 ГБ файл, 18,5 ГБ весов (эксперты NVFP4, крайние слои и внимание MXFP8). В aux — `tokenizer.json`
(из vocab/merges) и `chat_template.jinja` (из `chat_template.json`). Декод ~115 ток/с.

## Вход

Звук — ffmpeg моно 16 кГц (у оригинала `qwen_omni_utils`/librosa — ресемплинг не повторён); log-mel Whisper по
всей длине (`len/160` кадров, без добивки) в bf16; куски по 100 кадров (хвост добит нулями до 100, как
`pad_sequence` — за краем свёртки видят `gelu(bias)`), 13 строк на кусок, окна внимания по 104 строки,
`ln_post → proj1 → GELU → proj2`. Промпт: `<|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|>
<|im_end|>\n<|im_start|>assistant\n` — без системы и текста; пустой `<think>` шаблона при
`enable_thinking=false` снимается (`Llm::finalize_prompt`). Сэмплинг: t 0,6, top_p 0,95, top_k 20.

## Сверка с transformers (24.11.2026)

input_ids совпадают (60), mel 2e-5, выход башни 3,5e-5 (F32); жадная генерация совпадает с эталоном BF16
(teacher forcing) до почти-ничьих — наш токен там его №2 с отрывом 0,25–0,5 ната.

## Вызов

```sh
MEDIA=звук.wav MEDIA_NOMARK=1 GEN_MAX=1500 synaptixctl gen qwen3-omni-30b-a3b-captioner ""
```

Текст запроса отбрасывается; ответ всегда по-английски.
