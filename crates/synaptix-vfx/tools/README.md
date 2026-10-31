# Конвертеры весов synaptix-vfx

Оригинальные веса (.pth/.pkl/.safetensors) → `model.safetensors` с понятными ключами и метаданными
(`arch`, размеры) + эталон `ref-*.safetensors` для `examples/vfx_check` / `examples/rife_check`;
затем `syn-pack КАТАЛОГ -o vfx/ID.syn --id ID --arch ARCH --purpose vfx`.

Нужен только torch (CPU); архитектуры SPAN/RealPLKSR/DySample — из spandrel (MIT), RIFE — из
vs-rife (MIT), в `arch/` без их загрузчиков. Запуск: `python -I tools/X_convert.py ВХОД КАТАЛОГ ИМЯ`.

| Модель | Источник | Лицензия |
|---|---|---|
| NanoVSR 226k/644k/1.7M/5.4M (×4) | github.com/filippawlicki/nanovsr releases v1.0 | MIT |
| 2xLiveActionV1_SPAN | github.com/jcj83429/upscaling | Apache-2.0 |
| 2xNomosUni_span_multijpg | github.com/Phhofm/models | CC-BY-4.0 |
| 2xPublic_realplksr_dysample_layernorm_real | github.com/Phhofm/models | Apache-2.0 |
| 2xVHS2HD-RealPLKSR | github.com/starinspace/StarinspaceUpscale | CC-BY-SA-4.0 |
| AnimeJaNai HD V3 (Compact/UltraCompact/SuperUltraCompact) | github.com/the-database/mpv-upscale-2x_animejanai 3.0.0 | CC-BY-NC-SA-4.0 |
| realesr-animevideov3 (×4) | github.com/xinntao/Real-ESRGAN v0.2.5.0 | BSD-3-Clause |
| RIFE 4.25 / 4.25.lite / 4.26 | github.com/HolyWu/vs-rife releases «model» | MIT |
