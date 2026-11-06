# F5-TTS (русские версии Misha24-10/F5-TTS_RUSSIAN)

Порт `SWivid/F5-TTS` v1 Base (DiT 22×1024, 16 голов, ConvNeXtV2-текст, условный flow matching) и вокодера
`charactr/vocos-mel-24khz` на synaptix. Голос — всегда клон: образец (до 12 с) + его точный текст.

## Бандлы (`~/Storage/Нейросети/Модели/tts`, arch `f5-tts`, ~730 МБ, веса DiT в F16)

| id | исходник | шаг |
|---|---|---|
| f5-tts-ru-v1-base | F5TTS_v1_Base/model_240000_inference | 2 663 074 |
| f5-tts-ru-accent-10k | F5TTS_v1_Base_accent_tune/model_10000_inference | 2 673 074 |
| f5-tts-ru-accent-20k | F5TTS_v1_Base_accent_tune/model_20000_inference | 2 683 074 |
| f5-tts-ru-accent | F5TTS_v1_Base_accent_tune/model_last_inference | 2 688 421 |
| f5-tts-ru-v2 | F5TTS_v1_Base_v2/model_last_inference | 3 077 074 |
| f5-tts-ru-v4-winter-102k | F5TTS_v1_Base_v4_winter/all-checkpoints/model_102000 | 3 179 074 |
| f5-tts-ru-v4-winter-150k | F5TTS_v1_Base_v4_winter/all-checkpoints/model_150000 | 3 227 074 |
| f5-tts-ru-v4-winter | F5TTS_v1_Base_v4_winter/model_212000 | 3 289 074 |

Компоненты: `dit` (ключи без `ema_model.transformer.`), `vocoder` (`backbone.*`, `head.out`); файлы
`vocab.txt` (CRLF — разбирается как Python в текстовом режиме), `config.json`. Лицензия весов — CC BY-NC 4.0.

## Детали, важные для совпадения

- Текст: `ref_text + gen` посимвольно (`convert_char_to_pinyin` без китайского: `;`→`,`, кавычки, пробел перед
  ASCII-сегментом длиннее символа); индекс +1 (0 — заполнитель); ударение — `+` перед гласной (символ словаря).
- RoPE x_transformers — пары (2j, 2j+1): строки `to_q`/`to_k` переставлены в раскладку половинами.
- Длительность: `ref_len + ref_len / байт(ref_text) · байт(gen) / speed` (speed ×0.3 при gen < 10 байт).
- Эйлер по `t = linspace(0, 1, nfe+1)` (EPSS для 5/6/7/10/12/16), sway −1, CFG 2 батчем (без образца и текста).
- Образец: громкость до RMS 0.1 (выход возвращается назад), ≤ 12 с, тишина по краям срезается.

## Сверка (`examples/f5_audit.rs` против `reference/f5_ref.py`, веса округлены до F16)

v1-base, образец 4,5 с, 987 кадров: ids совпали; мел образца 6,7e-6; мел (CUDA F32) 4,0e-5, (CPU F32) 1,3e-5,
(CUDA F16) 3,2e-3; вокодер 2,2e-5. Whisper распознаёт все 8 версий дословно. RTX: RTF ≈ 0,12–0,17 (F16, NFE 32).
