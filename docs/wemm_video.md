# WeMM-Video-35BA3B (tencent)

Видео-LLM: текст — Qwen3.5-MoE 35B-A3B (`model.language_model.*`, 40 слоёв, 256 экспертов / 8 активных),
зрение — башня Qwen3-VL (`model.visual.*`, 27 блоков, patch 16, merge 2, temporal 2, без deepstack),
аудио — энкодер Whisper-large + Q-Former BERT (`model.audio.*`). `model_type = wemm_qwen3_5_moe`
распознаётся как гибрид Qwen3.5-MoE (`facade/arch.rs`). Аудио пока не исполняется — веса лежат в бандле
BF16 под второй этап. MTP в весах нет (в конфиге `mtp_num_hidden_layers=1`) — загрузчик это переносит.

## Сборка бандла

```sh
# оригинал: ~/Storage/Нейросети/Модели/_hf/WeMM-Video-35BA3B (17 шардов, 67 ГБ, сверка по SHA256SUMS)
target/fast-release/syn-pack ~/Storage/Нейросети/Модели/_hf/WeMM-Video-35BA3B \
  -o ~/Storage/Нейросети/Модели/llm/wemm-video-35b-a3b-bf16.syn --id wemm-video-35b-a3b
target/fast-release/synaptix quantize wemm-video-35b-a3b-bf16.syn wemm-video-35b-a3b.syn \
  --policy llm --vision mxfp8 --rule model.audio.=none \
  --rule .layers.0.mlp.experts=nvfp4 --rule .layers.39.mlp.experts=nvfp4
```

Эксперты крайних слоёв — NVFP4, как у qwen3.6-35b-a3b: с MXFP8 в них языковая часть 23,6 ГБ не влезает
в 24 ГБ, эксперты уходят в RAM и декод падает до ~60 ток/с. Итог: 23,4 ГБ файл (из них ~2,5 ГБ аудио BF16,
на карту не грузится), веса 20,9 ГБ. `visual.blocks.N.mlp.linear_fc2` остаётся BF16 (K=4304 не кратно 32).

## Проверка (RTX 5090)

```sh
synaptix run ~/Storage/Нейросети/Модели/llm/wemm-video-35b-a3b.syn "<промпт>" \
  --video trans_Elegant_01.mp4 --chat --no-think --max-tokens 700
```

Клип 1080×1920 → 3990 видеотокенов; префилл 4224 ток за 1,2 с (3580 ток/с), декод 108 ток/с, всё на карте.
Ответ: слайд справа налево, диагональная граница, 9:16.
