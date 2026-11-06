# Supertonic 3 (Supertone/supertonic-3) — нативный порт в synaptix

Оригинал опубликован только графами ONNX (`onnx/{duration_predictor,text_encoder,vector_estimator,vocoder}.onnx`,
rev 3cadd1ee) + `tts.json`, `unicode_indexer.json`, готовые голоса `voice_styles/*.json`; логика инференса —
`py/helper.py` из github supertone-inc/supertonic. PyTorch-кода архитектуры нет: она восстановлена по графам
собственным читателем ONNX (`synaptix_io::onnx`, пример `onnx_dump --filter <узел>`).

## Бандл
`supertonic-3.syn` (arch `supertonic`, ~0.4 ГБ F32): компоненты `duration_predictor`, `text_encoder`,
`vector_estimator`, `vocoder` — веса из инициализаторов ONNX (`onnx_dump model.onnx --weights x.safetensors`;
безымянные `onnx::*` названы по узлу-потребителю: `<узел>.w<вход>`), файлы `tts.json`, `unicode_indexer.json`,
`voices.json` (все 10 голосов: `style_ttl` [50,256], `style_dp` [8,16]), LICENSE (OpenRAIL), README.
Квант не нужен (модель ~100M, весь счёт F32 как в onnxruntime).

Сборка: каталог `<компонент>/model.safetensors` + файлы → `synaptix convert <каталог> supertonic-3.syn --arch supertonic --id supertonic-3`.

## Архитектура (батч 1: маски — единицы)
- Текст: NFKD, чистка (как `_preprocess_text`), `<lang>…</lang>`, символы → `unicode_indexer` (−1 → последняя строка).
- ConvNeXt: depthwise k (5/7) с паддингом краями (`edge`): симметрично 2·d (энкодеры, поле), каузально (k−1)·d
  слева (вокодер); LayerNorm по каналам eps 1e-6; 1×1 ×4; GELU(erf); 1×1; γ; остаток.
- Энкодер VITS (DP: 2 слоя × 2 головы, 64; текст: 4 × 4, 256): относительное внимание с окном 4
  (`emb_rel_k/v` [1,9,d], общие для голов), постнорма, FFN 1×1 ReLU; выход + вход (обход вокруг энкодера).
- DP: эмбеддинг 64, токен предложения в начало, ConvNeXt ×6, VITS, позиция 0 → 1×1 → concat(style_dp 128) →
  Linear 192→128 → PReLU → Linear → exp = длительность, с; делится на темп (1.05).
- Текстовый энкодер: эмбеддинг 256, ConvNeXt ×6 (d 1,1,2,2,4,4), VITS → x0; два внимания к 50 токенам
  стиля (ключи — tanh(W_k·style_key), значения — W_v·style_ttl, 2 головы ×128, масштаб 1/16):
  h1 = x0 + A1(x0), h2 = x0 + A2(h1); LayerNorm.
- Векторное поле (latent 144 = 24×6, L = ceil(dur·44100/3072)): proj_in 144→512; время t = step/total:
  sin/cos(1000·t·f), f_i = 10000^(−i/31) → Linear 64→256 → Mish → 256→64; 4 группы по 6 блоков:
  ConvNeXt ×4 (d 1,2,4,8) → +Linear(t) → ConvNeXt → внимание к тексту (8×64, RoPE половинами по
  нормированным позициям i/L и j/T, θ_i = 10·10000^(−i/32), масштаб 1/16) + остаток + LN → ConvNeXt →
  внимание к стилю (2×128) + остаток + LN; ConvNeXt ×4; proj_out 512→144.
  CFG батчем 2: безусловный — `text_special_token`, `style_key/value_special_token`;
  v = 4·v_c − 3·v_u; x ← x + v/total (Эйлер, 8 шагов).
- Вокодер: /0.25 → [24, 6L] → ·std + mean → conv 24→512 k7 (каузально) → ConvNeXt ×10 k7
  (d 1,2,4,1,2,4,1,1,1,1, каузально) → BatchNorm → conv 512→2048 k3 → PReLU → 1×1 → 512 семплов на шаг;
  обрезка до int(44100·dur).
- Шум: splitmix64 + Бокс — Мюллер (`noise`), тот же в эталоне.

## Сверка (examples/supertonic_audit.rs, reference/supertonic_ref.py — onnxruntime + helper.py)
Индексы текста совпадают, длительность — до 1e-6 с, звук (EN 4.1 с, RU 4.6 с, голос M1, 8 шагов) —
отн. 7.7e-5…1.4e-4 на CPU, 5.5e-5…4.3e-4 на CUDA. Whisper large-v3-turbo распознаёт RU/EN всех проверенных голосов.

## Скорость (RTX 5090 Laptop, F32)
~5 с речи за 0.10 с (RTF ≈ 0.02, ~50× быстрее реального времени; DP+текст 4 мс, 8 шагов поля 85 мс,
вокодер 12 мс); загрузка 0.07–0.13 с; CPU RTF ≈ 0.07.

## Голоса и персонажи службы (по медиане F0)
F1 185 Гц, F2 195, F3 158, F4 200, F5 162, M1 130, M2 87, M3 94, M4 118, M5 92.
female_young→F4, female_adult→F1, female_elderly→F3, male_young→M1, male_adult→M2, male_elderly→M3,
girl→F2, boy→M4 (детских голосов нет). Клона по записи нет: энкодер стиля в открытых весах не опубликован.
Отладка: `SYN_ST_DUMP=каталог` (промежуточные поля), `SYN_ST_PROF=1` (время частей).
