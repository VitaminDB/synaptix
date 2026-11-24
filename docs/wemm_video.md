# WeMM-Video-35BA3B (tencent)

Видео-LLM: текст — Qwen3.5-MoE 35B-A3B (`model.language_model.*`, 40 слоёв, 256 экспертов / 8 активных),
зрение — башня Qwen3-VL (`model.visual.*`, 27 блоков, patch 16, merge 2, temporal 2, без deepstack),
аудио — энкодер Whisper-large + Q-Former BERT (`model.audio.*`). `model_type = wemm_qwen3_5_moe`
распознаётся как гибрид Qwen3.5-MoE (`facade/arch.rs`). Аудио пока не исполняется — веса лежат в бандле
BF16 и исполняются (`--audio`, см. ниже). MTP в весах нет (в конфиге `mtp_num_hidden_layers=1`) — загрузчик это переносит.

## Сборка бандла

```sh
# оригинал: ~/Storage/Нейросети/Модели/_hf/WeMM-Video-35BA3B (17 шардов, 67 ГБ, сверка по SHA256SUMS)
target/fast-release/syn-pack ~/Storage/Нейросети/Модели/_hf/WeMM-Video-35BA3B \
  -o ~/Storage/Нейросети/Модели/llm/wemm-video-35b-a3b-bf16.syn --id wemm-video-35b-a3b
target/fast-release/synaptix quantize wemm-video-35b-a3b-bf16.syn wemm-video-35b-a3b.syn \
  --policy llm --rule model.visual.=none --rule model.audio.=none \
  --rule .layers.0.mlp.experts=nvfp4 --rule .layers.39.mlp.experts=nvfp4
```

Эксперты крайних слоёв — NVFP4, как у qwen3.6-35b-a3b: с MXFP8 в них языковая часть 23,6 ГБ не влезает
в 24 ГБ, эксперты уходят в RAM и декод падает до ~60 ток/с. Башня зрения — BF16: с MXFP8 её выход
отличается от эталона F32 на 9,6 %, с BF16 (вычисления F16) — на 1,3 % (+0,4 ГБ VRAM, всё равно влезает).
Итог: 23,7 ГБ файл (из них ~2,5 ГБ аудио BF16, грузится только на время кодирования звука).

## Проверка (RTX 5090)

```sh
synaptix run ~/Storage/Нейросети/Модели/llm/wemm-video-35b-a3b.syn "<промпт>" \
  --video trans_Elegant_01.mp4 --chat --no-think --max-tokens 700
```

Клип 1080×1920 → 3990 видеотокенов; префилл 4224 ток за 1,2 с (3580 ток/с), декод 108 ток/с, всё на карте.
Ответ: слайд справа налево, диагональная граница, 9:16.

## Паритет с эталоном transformers (24.11.2026)

Сверка с `WeMMProcessor` + модулями `modeling_wemm_moe.py` / `h5_modeling_*.py` на CPU (скрипты — дампы
`SYN_DUMP_IDS`, `SYN_DUMP_VIDEO`, `SYN_WEMM_AUDIO_DUMP`, `examples/tower_dump`):

- input_ids видео и звука совпадают побитно: системная реплика «You are a helpful assistant.», `\n` после
  медиаблока, кадры linspace по 2 fps (4…128), ≤768 токенов на группу (5-с клип 1080×1920 → 10 кадров,
  сетка 72×40, 3600 токенов), метки `<t seconds>` — середина группы; звук — группы по 1 с, паддинг 30-с куска
  в последней группе;
- пиксели кадров — бит в бит (бикубика torchvision на uint8, `synaptix_io::image::resample`);
- башня зрения: F32 — 2e-5 (после исправления интерполяции pos_embed и GELU merger'а — до этого 31 %);
- аудио-башня (Whisper + Q-Former): cos 0,9998, rel 1,9 % (BF16).

Логиты всей LLM BF16 против нашего NVFP4 не сверялись (70 ГБ BF16 на CPU) — языковая часть та же, что у
Qwen3.5/3.6-MoE.

## Звук

```sh
synaptix run wemm-video-35b-a3b.syn "<промпт>" --audio файл.wav --chat --no-think --max-tokens 600
MEDIA=файл.wav MEDIA_NOMARK=1 synaptixctl gen wemm-video-35b-a3b "<промпт>"   # через службу
```

Стерео сводится в моно (как у оригинала) — панорамы модель не слышит. Видео со звуком (`video_audio`) — пока
нет. В режиме размышлений при жадном выборе (`top_k 1`) модель зацикливается в размышлениях — брать пресет.
