# Audio8-TTS (Edge0/Audio8-TTS-Realistic-Air) — нативный порт в synaptix

Источник истины: HF `Edge0/Audio8-TTS-Realistic-Air` (rev 23c305cc), `modeling_arktts.py`,
`modeling_arktts_codec.py`, `processing_arktts.py` (transformers 4.57.3, trust_remote_code).
`config.json` → `model_type = arktts`; в службе и каталоге — движок `audio8`.

## Бандл
- `audio8-tts-realistic-air.syn`, arch `audio8tts`, компоненты `main` (модель, BF16 как в
  оригинале) и `codec` (кодек 44.1 кГц, F32 как в `codec.pth`, имена upstream, weight-norm
  не свёрнут). Файлы: config.json, generation_config.json, tokenizer*.json, processor/preprocessor.
- Сборка: `reference/stage_bundle.py <снимок HF> <стадия>` (codec.pth → codec/model.safetensors),
  затем `synaptix convert <стадия> audio8-tts-realistic-air.syn --arch audio8tts --id audio8-tts-realistic-air`.
- Квант: LM — плотный BF16. MXFP8 проверен (`synaptix quantize --policy dit --rule attention.wqkv=none`):
  ошибка логитов к F32 у медленного AR 2.2 % (BF16 — 0.2 %), у быстрого 9–19 % (BF16 — 0.4 %),
  скорости не даёт (декод 0.6B упирается в запуски ядер), экономит 0.38 ГБ из 2.57 — не берём.
  `attention.wqkv` в любом случае должен оставаться плотным (перестановка строк под RoPE).
  Квант на лету в службе (`precision`) работает для остальных линейных слоёв медленного AR.

## Архитектура
- Медленный AR: 24 блока, dim 896, 14 голов / 2 KV, head 64, FFN 4864 (SwiGLU w1/w3/w2),
  `wqkv` с bias, RMSNorm eps 1e-6, RoPE base 1e6, контекст 2048. Голова связана с `embeddings`.
- Вход столбца: `embeddings[tok] + Σ_i codebook_embeddings[code_i + i·4096]` (сумма — только
  у семантических токенов 151678..=155773).
- Быстрый AR: 4 блока той же формы без bias; вход шага 0 — нормированный hidden медленного
  (`norm_fastlayer_input`), дальше `fast_embeddings[code]`; позиции 0..9, KV на 10.
- RoPE upstream — комплексная (пары 2j, 2j+1), таблицы cos/sin в BF16. У нас строки q/k
  `wqkv` переставлены по головам (j←2j, j+32←2j+1) → split-RoPE слитым ядром; таблицы
  тоже округлены до BF16. То же в трансформерах кодека (base 1e4).
- Сэмплинг (`sample.rs`): фильтр семантика+EOS → top-k/top-p legacy по сырым логитам →
  /temperature → argmax(p/−ln U). RAS: второй выбор с top_p 0.9 / temp 1.0 берётся, если
  обычный уже был в окне 10 последних семантических (окно после шага 0 — нули).
  Шум — счётчиковый ГПСЧ (splitmix64, 23 бита), расход как `torch.rand(V)` upstream:
  2×155776 на семантику + 9×4096 на кодбуки за кадр (кодбуки считаются и на кадре EOS).
- Кодек: см. шапку `codec.rs`. Каузальные свёртки: слева k_eff−stride, справа добивка.
  Snake с α>0 — слитым ядром (`snake(ln α, ln α, 1e-9)`).

## Промпт (processor)
Части кодируются отдельно и склеиваются:
- без образца: `<|im_start|>system\n` · `convert the provided text to speech` · `<|im_end|>\n` ·
  `<|im_start|>user\n` · текст · `<|im_end|>\n` · `<|im_start|>assistant\n<|voice|>`;
- с образцом: system `…reference to the following:\n\nText:\n` + `<|speaker:0|>`расшифровка +
  `\n\nSpeech:\n`, затем коды образца (семантика + 151678 в строке 0, коды в строках 1..10),
  затем `<|im_end|>\n<|im_start|>user\n…<|voice|>`.

## Сверка (examples/audio8_audit.rs, reference/audio8_ref.py)
- CPU F32, одно зерно: EN (81 кадр), RU (122), клон EN по образцу (88) — все коды 10 кодбуков
  совпали с upstream кадр в кадр; логиты 1-го шага отн. 2–4e-7; кодек-декод отн. 1.5e-6;
  кодирование образца — 790/790 кодов.
- CUDA BF16: логиты 2e-3 (медленный) / 4e-3 (кодбук 1).
- Whisper large-v3-turbo по выходам GPU (RU ×3 зерна, EN ×2, клон): все слова распознаны.
- Скорость (RTX 5090 Laptop, BF16, без графов): ~157 кадров/с (7.3× реального времени),
  кодек 4–6 с речи за 0.11–0.17 с; RTF ≈ 0.17. Образец 6.4 с кодируется за 0.11 с.

## Память
Таблица `embeddings` (279 МБ) на CUDA живёт в RAM; на карте — семантическая голова (4097 строк).
Энкодер кодека (~0.8 ГБ F32) грузится при первом клонировании (`SYN_AUDIO8_ENCODER=eager` — сразу).
Отладка: `SYN_AUDIO8_DUMP=каталог` — промежуточные тензоры энкода; `SYN_AUDIO8_ROPE=interleaved`.
