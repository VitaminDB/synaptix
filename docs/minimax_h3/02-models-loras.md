# MiniMax-H3: Модели, варианты, кванты, LoRA и экосистема MiniMax-H3

Черновик. Собран 2026-11-28 только по локальным материалам `docs/minimax_h3/sources/` (сеть не использовалась, веса не скачивались).
Пути к источникам даны относительно `docs/minimax_h3/`. Загрузки — из `sources/hf/meta.tsv` (снимок HF на 2026-11-28; это счётчик HF, а не число уникальных пользователей).
Пометка **«не проверено»** — маркетинговые или авторские цифры, которые мы не воспроизводили.
Единицы: HF и Atlas Cloud пишут размеры в ГБ (10⁹), список wildminder — в ГиБ (2³⁰). Отсюда «66,28 GB» = «61,73 GB» для одного и того же файла (`sources/web/atlascloud__open-source-weights.md`, `sources/github/wildminder__awesome-minimax-H3.md`).

## 1. Официальные веса

### 1.1. Что открыто

| Репозиторий | Что внутри | Загрузки / лайки | Источник |
|---|---|---|---|
| `MiniMaxAI/MiniMax-H3` | Два чекпойнта H3-Base: **FL2VA** (t2va + первый/последний кадр) и **Ref2VA** (омни-референсы), BF16; оригинальный формат (`FL2VA/`, `Ref2VA/`) и diffusers рядом | 5 974 796 / 7 448 | `sources/hf/cards/MiniMaxAI__MiniMax-H3.md` |
| `MiniMaxAI/MiniMax-H3-Image` | Картинки и редактирование, один чекпойнт, DiT без звуковых слоёв, дистиллирован в 6 шагов без CFG | 1 102 421 / 2 631 | `sources/hf/cards/MiniMaxAI__MiniMax-H3-Image.md` |
| `MiniMaxAI/MiniMax-H3-Image-PE` | Расширитель промптов для T2I: дообученный Qwen3.8-27B в FP4 (17,9 GB); PE для редактирования пока только в виде промпта | 6 044 / 120 | `sources/hf/cards/MiniMaxAI__MiniMax-H3-Image-PE.md`, `sources/hf/cards/MiniMaxAI__MiniMax-H3-Image.md` |
| `Comfy-Org/MiniMax-H3`, `Comfy-Org/MiniMax-H3-Image` | Официальная переупаковка для ComfyUI (подробности в разделе 2) | 33 871 716 / 5 052 954 | `sources/hf/cards/Comfy-Org__MiniMax-H3.md`, `sources/hf/cards/Comfy-Org__MiniMax-H3-Image.md` |

- Объявление вышло 30–31.07.2026, а веса опубликованы на HF 03.08.2026 (`sources/web/nyu-rits__h3-release-open-weights-promised.md`, `sources/web/nyu-rits__h3-weights-us-eu-excluded.md`, `sources/web/minimax.io__news-h3-open-source.md`). Лицензия вступила в силу 02.08.2026 (`sources/web/atlascloud__open-source-weights.md`).
- Режимы FL2VA: без картинки — T2VA, одна картинка — первый или последний кадр, две — FL2VA. Ref2VA: до 9 картинок, до 3 видео и до 3 аудио (каждое по 2–15 с, в сумме не дольше 15 с), всего не больше 12 файлов. Аудио нельзя подать как единственный вход (`sources/web/minimax.io__news-h3-open-source.md`, `sources/hf/cards/MiniMaxAI__MiniMax-H3.md`).
- Выход: 4–15 с, 24 fps, по умолчанию короткая сторона 768 px, звук 32 кГц стерео. Стабильно поддерживаются 11 языков диалога, включая русский (`sources/hf/cards/MiniMaxAI__MiniMax-H3.md`). Расхождение: в метаданных карточки HF `language:` указаны только sv, en, fr, de, es, nl — это похоже на шаблон, а не на реальный список.
- Выпущенные чекпойнты **CFG-дистиллированы** (`sources/hf/cards/MiniMaxAI__MiniMax-H3.md`). Официальные воркфлоу ComfyUI работают с cfg 1 и без негатива; в vllm-omni прямо написано «has no negative branch» (`sources/local/synaptix-probe-findings.md`). Часть сторонних гайдов советует `guidance_scale=7.5` и 50 шагов (гайд для 8 ГБ в `sources/github/wildminder__awesome-minimax-H3.md`) — это противоречит официальному описанию.
- Рекомендуемые рантаймы: SGLang (в примере 4 GPU, Ulysses 4), vLLM / vLLM-Omni, diffusers `ModularPipeline`, ComfyUI (`sources/hf/cards/MiniMaxAI__MiniMax-H3.md`). Для H3-Image дополнительно указан DiffSynth-Studio (там есть low-VRAM инференс и обучение) (`sources/hf/cards/MiniMaxAI__MiniMax-H3-Image.md`).

### 1.2. Что не открыто

| Компонент / вариант | Статус | Источник |
|---|---|---|
| **H3-Context-IR** (многоэтапный препроцессор: разбор инструкции, связи модальностей, дописывание деталей) | Не открыт, есть только API `/video-generation-v2-h3-context-ir`. Взамен даны гайды по промптам и skills | `sources/hf/cards/MiniMaxAI__MiniMax-H3.md` |
| **H3-Regenerate-2K** (in-context перегенерация результата 768p в 2K той же базовой моделью) | Не открыт: «not yet open-sourced… will release it once it is ready». Только API `/video-generation-v2-regeneration` | `sources/hf/cards/MiniMaxAI__MiniMax-H3.md` |
| **Разреженное внимание** (native sparse attention, введено на последнем этапе обучения) | В релизе только полное внимание, обещан отдельный выпуск. Issue #89 (10.10.2026) с вопросом о сроках открыт | `sources/hf/cards/MiniMaxAI__MiniMax-H3.md`, `sources/github/MiniMax-AI__MiniMax-H3__issues.md` |
| API-модели `MiniMax-H3-Highspeed`, `-Lite`, `-Max`, `-Max-Turbo`, `-MG` (хостинг MachGen), `-NC` («H3 Fast by Nunchux AI») | Только API. 2K есть лишь у `MiniMax-H3` и `-MG`; 480P у `MiniMax-H3` отдаёт RunningHub | `sources/web/platform.minimax.io__api-video-generation-v2.md` |
| Image editing PE | Выпущен только task-prompt, весов нет | `sources/hf/cards/MiniMaxAI__MiniMax-H3-Image-PE.md` |

Локальный H3 выдаёт **только 768p** (≈1 MP). 2K — либо через API, либо сторонним апскейлом (`sources/web/docs.comfy.org__h3-local-tutorial.md`, `sources/web/atlascloud__open-source-weights.md`). Интерфейс Hailuo вдобавок умеет «Extend Video» примерно до 30 с (`sources/web/nyu-rits__h3-release-open-weights-promised.md`); в открытых весах такого инструмента нет.

### 1.3. Архитектура H3-Base

Всё ниже — из `sources/hf/cards/MiniMaxAI__MiniMax-H3.md` (тот же текст в `sources/web/minimax.io__news-h3-open-source.md`), если не указано иное.

- **H3-Omni-Transformer**: плотный single-stream трансформер на 33B, из них ≈13B приходится на ветви AdaLN. Выходы модуляции AdaLN можно предвычислить и закэшировать, поэтому при инференсе эти параметры можно не грузить. Опубликованы полные веса, пригодные для дообучения. У слоёв внимания и FFN нет модально-специфичных частей: такие параметры есть только во входных и выходных слоях и в AdaLN (модально-специфичный AdaLN). Позиции кодируются 3D MM-RoPE `(t, h, w)`.
- Размеры (из карточки H3-Image, где DiT «такой же, как H3-Omni-Transformer»): 33,1B, **50 слоёв, hidden 5376, 56 голов, FFN 14336**. Веса в BF16, эмбеддинг, timestep и выходной слой — в FP32 (`sources/hf/cards/MiniMaxAI__MiniMax-H3-Image.md`). Наши константы порта такие же: `adaln_out_features 96768` = 18 × 5376 (3 модальности × 6 чанков) (`sources/local/synaptix-port-code.md`).
- **H3-Encoder** — полные веса **Qwen3-VL-32B**. В DiT идут скрытые состояния **50-го слоя**. В токенизатор добавлены спецтокены (`<d>` и др.), поэтому нужен токенизатор из репозитория H3. Картинки кодируются и энкодером, и VisualVAE, звук — только AudioVAE.
  Следствие, которым пользуется сообщество: слои после 49-го и `lm_head` для кондиционирования не нужны. Так появились урезанные энкодеры (`sources/hf/cards/nif0__Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF.md` — «Pruned layers after 49»). У ethanfel энкодер разделён на «слои 0–49 + vision» и отдельный «хвост 50–63» для генерации промптов (`sources/github/wildminder__awesome-minimax-H3.md`).
- **H3-VisualVAE**: временно-каузальный, **f16t4d24** (16× по пространству, 4× по времени, 24 канала). Перед DiT латент патчится 1×2×2, так что эффективно токен покрывает 32×32 px и 4 кадра. Декодер обучен отдельно, он на ViT. Длина ролика задаётся сеткой кадров `17k+5` (`sources/web/docs.comfy.org__prompt-guide.md`; про сетку 5+17k в нашем порте — `sources/local/synaptix-probe-findings.md`). Размер кадра должен быть кратен 32 (`sources/local/synaptix-perf.md`).
- **H3-AudioVAE**: один кодек на оба канала (каналы обрабатываются независимо), 32 кГц → латент 40 Гц на канал. Латент оптимизирован в духе VA-VAE.
- Расхождение в описании блоков: NYU RITS пишет «shared DiT backbone repeated 50 times» (`sources/web/nyu-rits__h3-weights-us-eu-excluded.md`), что совпадает с 50 слоями. The Batch насчитала «три энкодера… отдельный VAE» (`sources/web/deeplearning.ai__the-batch-minimally-open.md`); официально энкодеров два — Qwen3-VL и AudioVAE, плюс VisualVAE.
- **H3-Image**: та же архитектура без звуковой ветки, тот же энкодер без изменений (из H3 его берут через `model_index.json`). Image VAE — CNN-энкодер + 2D ViT-декодер на 24 слоя, f16, 24 канала. Дистиллирован в **6 NFE без CFG**: `pdd_fused_heads.safetensors` (12 MB) — пошаговые выходные головы вместо выходного слоя, плюс `pdd_config.json` с расписанием. Генерирует ~16 MP (4096², до 6272×2688), рабочий диапазон 4–20 MP. Другие значения шагов и guidance API отклоняет с HTTP 400 (`sources/hf/cards/MiniMaxAI__MiniMax-H3-Image.md`).

### 1.4. Файлы и размеры

Официальный репозиторий целиком весит **498 GB**, зеркало Comfy-Org — **343 GB** (по индексам HF, август 2026) (`sources/web/atlascloud__open-source-weights.md`). Таблица ниже — по Comfy-Org (ГБ из Atlas Cloud, ГиБ из списка wildminder):

| Файл (Comfy-Org) | ГБ | ГиБ | Комментарий |
|---|---|---|---|
| `minimax_h3_{fl2va,ref2va}_bf16` | 66,28 | 61,73 | полный DiT |
| `minimax_h3_{fl2va,ref2va}_int8_convrot` | 34,04 | 31,70 | |
| `minimax_h3_{fl2va,ref2va}_pruned_bf16` | 40,24 | 37,46 | AdaLN свёрнут в таблицу ранга 8 (раздел 2.1) |
| `minimax_h3_{fl2va,ref2va}_pruned_int8_convrot` | 20,97 | 19,53 | вариант ComfyUI по умолчанию |
| `minimax_h3_{fl2va,ref2va}_pruned_fp8_scaled` | — | 19,52 | только если нельзя int8_convrot |
| `minimax_h3_{fl2va,ref2va}_pruned_w6a8` | — | — | есть в списке файлов, размер не указан |
| `qwen3vl_32b_minimax_h3_bf16` | 51,51 | 47,97 | энкодер |
| `qwen3vl_32b_minimax_h3_int8_convrot` | 27,14 | 25,28 | |
| `qwen3vl_32b_minimax_h3_nvfp4_awq` | 15,69 | 14,61 | работает и без Blackwell |
| `minimax_h3_video_vae_fp16` | 5,21 | 4,85 | |
| `minimax_h3_video_vae_int8_convrot` | — | — | в списке Comfy-Org; у Kijai — 2,95 GB |
| `minimax_h3_audio_vae_fp32` | 0,61 | 0,577 | держать в fp32 |

Источники: `sources/hf/cards/Comfy-Org__MiniMax-H3.md`, `sources/web/atlascloud__open-source-weights.md`, `sources/github/wildminder__awesome-minimax-H3.md`.

- Минимальный рабочий набор на одну задачу — **42,5 GB** (pruned int8 DiT + nvfp4 энкодер + оба VAE). Оба чекпойнта — 63,4 GB, всё в bf16 — 123,6 GB (`sources/web/atlascloud__open-source-weights.md`, `sources/web/blog.comfy.org__day0-support.md`).
- Что значит **pruned**: веса модуляции (≈40% параметров) заменены «функционально эквивалентной таблицей» — по словам ComfyUI, «без потери качества» (**не проверено**). Это та же идея, что официальный кэш выходов AdaLN (`sources/web/blog.comfy.org__day0-support.md`, `sources/hf/cards/MiniMaxAI__MiniMax-H3.md`). Утверждение ComfyUI «работает на RTX 3060 12 ГБ» держится только на динамической выгрузке в RAM (`sources/web/nyu-rits__h3-weights-us-eu-excluded.md`, `sources/web/atlascloud__open-source-weights.md`).
- Расхождение по энкодеру: в карточке H3-Image «Text encoder… 66.7 GB» в репозитории MiniMax-H3 (`sources/hf/cards/MiniMaxAI__MiniMax-H3-Image.md`), а bf16-файл Comfy-Org — 51,51 GB. Вероятно, в оригинале часть весов в FP32 или есть лишние слои/`lm_head`. Это наше предположение, проверить по индексу файлов.
- Расхождение по Image VAE: 2,34 GB FP32 в оригинале (`sources/hf/cards/MiniMaxAI__MiniMax-H3-Image.md`) и `minimax_h3_image_vae_fp16` у Comfy-Org, размер не указан (`sources/hf/cards/Comfy-Org__MiniMax-H3-Image.md`).
- H3-Image: `transformer/` 66,28 GB, `vae/` 2,34 GB, всего с энкодером в модельном зоопарке 68,65 GB (`sources/hf/cards/MiniMaxAI__MiniMax-H3-Image.md`). У Comfy-Org — только `minimax_h3_image_pruned_int8_convrot`, энкодеры int8/nvfp4 и PE `qwen3.8_27b_minimax_h3_image_pe_w4a8_gptq` (`sources/hf/cards/Comfy-Org__MiniMax-H3-Image.md`).

### 1.5. Лицензия и территориальные ограничения

**MiniMax H3 Community License** (видео-H3). Пункты в пересказе Atlas Cloud (`sources/web/atlascloud__open-source-weights.md`):
- I.3, I.5 — «Applicable Territory»: весь мир **кроме ЕС, Великобритании, Республики Корея и США** («Excluded Territories»).
- V.4 — запрещено использовать, воспроизводить, изменять, распространять и показывать веса **и их выходы** за пределами Applicable Territory.
- II — для исключённых регионов работает отдельный канал лицензирования: заявка через форму «only for USA/EU/UK/South Korea» (`sources/hf/cards/MiniMaxAI__MiniMax-H3.md`). Формулировка «not yet, not not ever» — из `docs/QA-about-License.md` репозитория (в пересказе Atlas Cloud).
- IV.2 — коммерческий продукт обязан **на видном месте показывать «MiniMax H3»** в интерфейсе. IV.1 — при выручке **больше 20 млн $ в год** нужно отдельное письменное разрешение.
- V.3 — запрещено использовать H3 и его выходы «to improve any other AI model», то есть дистиллировать в свою модель нельзя.
- III.1, III.4 — при распространении прикладывать договор и NOTICE, помечать изменённые файлы. IX — право Гонконга. Дополнительное примечание: сам энкодер Qwen3-VL-32B распространяется под Apache 2.0.

**MiniMax-H3-Image Community License** — отдельная и более строгая. Письменное разрешение нужно при выручке **больше 1 млн $** или при предоставлении хостинг-сервиса третьим лицам (цитата в `sources/hf/cards/Comfy-Org__MiniMax-H3-Image.md`). PE подпадает под ту же лицензию (`sources/hf/cards/MiniMaxAI__MiniMax-H3-Image.md`). Есть ли в ней территориальная оговорка, по нашим материалам не видно.

**Расхождения источников:**
- The Batch (deeplearning.ai) трактует исключённые территории как требование «подать заявку, чтобы пользоваться на тех же условиях», а про 20 млн $ не пишет: там сказано «Weights free for noncommercial and commercial uses» (`sources/web/deeplearning.ai__the-batch-minimally-open.md`). NYU RITS и Atlas Cloud формулируют жёстче: грант на эти страны просто не распространяется (`sources/web/nyu-rits__h3-weights-us-eu-excluded.md`, `sources/web/atlascloud__open-source-weights.md`).
- NYU RITS отмечает, что у MiniMax M3 (июнь 2026) территориальной оговорки не было, а лицензию M2.7 после выпуска ужесточали. Поэтому текст лицензии H3 стоит считать изменчивым (`sources/web/nyu-rits__h3-weights-us-eu-excluded.md`).
- Дата выпуска: в новости MiniMax — 2026-08-03, лицензия действует с 2026-08-02 (`sources/web/minimax.io__news-h3-open-source.md`, `sources/web/atlascloud__open-source-weights.md`).
- У производных репозиториев своя пометка: например, переписчики lightx2v 27B выложены без файла LICENSE (`sources/github/wildminder__awesome-minimax-H3.md`). При этом пункт V.3 формально запрещает учить чужие модели на выходах H3, а многие сторонние ускорители и переписчики (раздел 3, 8) учены именно на них. Это юридический риск для тех, кто ими пользуется.

Для нас (Казахстан) территориальная оговорка не действует. Остаются атрибуция в UI коммерческого продукта и запрет V.3 на дистилляцию.
## 2. Переупаковки и кванты

### 2.1. Термины

- **pruned / «curve form»** (AdaLN-pruned). Аудио-, референсная ветка и блоки не удаляются. В каждом из 50 блоков есть `adaln_proj.linear` (2688→96768), плюс один `norm_out.linear`: вместе 13,03B параметров, 39,3% чекпойнта. Все они зависят только от `silu(time_embedder(t))`, то есть от таймстепа. Поэтому их сворачивают в аффинное подпространство ранга 8: MLP таймстепа заменяется таблицей `adaln_t_table` на 1025 точек, а проекции AdaLN получают вход шириной 8. В итоге DiT сокращается с 33,14B до 20,11B параметров, файл bf16 — с 66,28 до 40,24 GB. Ошибка, по замеру автора, примерно в 250 раз меньше одного шага округления bf16 («effectively lossless», **не проверено нами**) (`sources/hf/cards/multimodalart__MiniMax-H3-Pruned.md`, `sources/hf/cards/coolthor__MiniMax-H3-pruned-NVFP4.md`, `sources/hf/cards/joeygambino__MiniMax-H3-curve-GGUF.md`). Pruned-файлы распознаются ComfyUI по форме тензоров, поэтому вне ComfyUI безопаснее брать непрореженные (`sources/hf/cards/rockerBOO__minimax-h3-nvfp4-convrot.md`).
- **ConvRot**. Каждый квантованный Linear домножается на блочно-диагональную матрицу Адамара 256×256: W·H, x·H. Функция слоя не меняется, но выбросы в отдельных каналах размазываются. Калибровка не нужна, вторая копия весов не хранится. Замер на t2va, H100, 20 шагов (косинус видео к bf16): int8 dynamic + ConvRot — 0,9893; int8 dynamic — 0,9819; int8 weight-only — 0,9773; fp8 + ConvRot — 0,9678; нижняя граница (bf16 против скомпилированного bf16) — 0,9940. Вывод автора: «ConvRot buys almost nothing for fp8 and everything for int8» (`sources/hf/cards/multimodalart__MiniMax-H3-Pruned.md`). В issues ComfyUI его описывают как «QuaRot variant… row-wise INT8» (`sources/github/Comfy-Org__ComfyUI__issues-minimax-h3.md`).
- **Hybrid bNN-49**. Основа — fl2va, но `adaln_proj` блоков NN–49 взяты из ref2va. Мотив: fl2va даёт лучшее качество, у ref2va «known training-quality issue». Остальные веса двух чекпойнтов почти совпадают (cos ≥ 0,9997). Шкала: b30 сильнее держит качество, b15 — следование референсам; автор советует начинать с b25 (`sources/hf/cards/smhfacct__Minimax-H3-fl2va-ref2va-hybrid-models.md`). Отличие от **ref-delta r1024** (xmarre): там к pruned fl2va добавлено SVD ранга 1024 от разницы ref2va−fl2va, и один файл обслуживает оба режима (`sources/hf/cards/xmarre__MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI.md`).
- **H3-Tiny** — структурное прореживание другого рода: AdaLN ранга 8 и удаление 8 из 50 блоков (`[6,7,8,9,10,11,13,15]`), итого 17,07B без дообучения (`sources/hf/cards/H3-Tiny__MiniMax-H3-Tiny-17B-Pruned-FL2VA.md`).

### 2.2. Таблица (видео-H3)

Размеры — из карточек. Где карточка их не даёт, взяты из списка wildminder (`sources/github/wildminder__awesome-minimax-H3.md`), он вторичен. ГиБ помечены.

| Репозиторий | Формат | Размер (DiT на вариант) | Для чего / оговорки | Загрузки | Источник |
|---|---|---|---|---|---|
| Comfy-Org/MiniMax-H3 | bf16, int8_convrot, pruned bf16/fp8_scaled/int8_convrot/w6a8; TE bf16/int8/nvfp4_awq; VAE | 61,73 / 31,70 / 37,46 / 19,52 / 19,53 ГиБ; w6a8 — нет данных | основной путь ComfyUI. Карточка: int8_convrot, если есть torch cu130, fp8 — только если int8_convrot недоступен | 33 871 716 | `sources/hf/cards/Comfy-Org__MiniMax-H3.md` |
| Kijai/MiniMax-H3-experimental, Kijai/MiniMax-H3_comfy, Kijai/MiniMax-H3-TAE | pruned w4a8_mixed (кодбук Lloyd-Max, групповые шкалы fp8), video VAE int8_convrot, LoRA Regenerate-2K, TAE для превью | 11,68 GB; VAE 2,95 GB; TAE 9 MB | экспериментальный. TAE работает только с KJNodes `ModelPreviewOverride` | 0 в meta (лайки 558/721/218) | `sources/github/wildminder__awesome-minimax-H3.md` |
| unsloth/MiniMax-H3-GGUF | GGUF Q2_K…Q8_0, только pruned; TE Q2_K_M/Q4_K_M | Q4_K 10,64 ГиБ, Q8_0 19,97 ГиБ; TE 12,2/17,0 ГиБ | sd.cpp / ComfyUI. `--cfg-scale 1.0` обязателен. На ROCm Q4_K «full of artifacts, A/V desync», Q2 даёт кашу | 3 774 406 | `sources/hf/cards/unsloth__MiniMax-H3-GGUF.md`, `sources/github/leejet__stable-diffusion.cpp__minimax-h3.md` |
| unsloth/MiniMax-H3-FP8 | diffusers+torchao: INT8, INT8-ConvRot, FP8 (pruned) | 18,86–18,87 GB | INT8 — с torch.compile, FP8 — без него | 0 | `sources/github/wildminder__awesome-minimax-H3.md` |
| DeepBeepMeep/MiniMax-H3 | bf16 / int8_convrot; pruned; pruned_rank8; VAE fp16/fp8mix; LightVAE | 66,3 / 34; 41,4 / 22,1; 40,3 / 21,1 GB | для WanGP | 1 786 632 | `sources/hf/cards/DeepBeepMeep__MiniMax-H3.md`, `sources/github/wildminder__awesome-minimax-H3.md` |
| Abiray/MiniMax-H3-Pruned-GGUF | GGUF pruned | Q4_K_M 11,6, Q5_K_M 14,1, Q8_0 21,6 GB | Q4_K_M — для 16 ГБ, Q5_K_M — для 24 ГБ | 936 048 | `sources/hf/cards/Abiray__MiniMax-H3-Pruned-GGUF.md` |
| Abiray/MiniMax-H3-GGUF | GGUF непрореженный; TE int4_convrot | Q4_K_M 19,9, Q8_0 36 GB; TE 15 GB | на HN: Q5_K_M работает на M5 Pro с 64 ГБ | 206 197 | `sources/hf/cards/Abiray__MiniMax-H3-GGUF.md`, `sources/web/hackernews__h3-thread.md` |
| Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot | pruned int8 / mixed int4+int8 / int4 / nvfp4; TE int4/int8/nvfp4 | ~21 / ~15,5 / ~11,3 / 12,5 GB | nvfp4 — только Blackwell | 122 734 | `sources/hf/cards/Abiray__Minimax-H3-nvfp4-INT4-INT8-Convrot.md` |
| realrebelai/MiniMax-H3_GGUFs | GGUF непрореженный + TE Q4_K_M | Q3_K_M 14,5, Q4_K_M 18,5 GB | лицензия «unknown» | 411 597 | `sources/hf/cards/realrebelai__MiniMax-H3_GGUFs.md` |
| leejet/MiniMax-H3-GGUF | GGUF (stable-diffusion.cpp) | в карточке нет; TE Q4_K_M 18,2 GB | своя ветка ComfyUI-GGUF. Для RDNA3 лучший вариант — INT8 ConvRot | 341 034 | `sources/hf/cards/leejet__MiniMax-H3-GGUF.md`, `sources/github/leejet__stable-diffusion.cpp__minimax-h3.md` |
| molbal/MiniMax-H3-GGUF | Q4_0, Q8_0, Q8_CR (Q8 + ConvRot), U16G (смесь INT8 и Q4_0) | 11,4 / 20,2 / 20,2 / 15,0 GB | U16G помещается в 16 ГБ и быстрее Q4_0; нужна ветка molbal | 75 253 | `sources/hf/cards/molbal__MiniMax-H3-GGUF.md` |
| joeygambino/MiniMax-H3-curve-GGUF, -GGUF, -encoder-GGUF | GGUF curve (pruned) и полный; энкодер на 50 слоёв | curve Q8_0 21,5, Q4_0 11,5; полный Q5_1 25,9; энкодер Q4_K_M 19,8 + mmproj 1,2 GB | mmproj обязателен для картинок и референсов | 11 182 / 20 744 / 52 129 | `sources/hf/cards/joeygambino__MiniMax-H3-curve-GGUF.md`, `sources/hf/cards/joeygambino__MiniMax-H3-encoder-GGUF.md` |
| MarxistLeninist/…-Pruned-IQ1-GGUF | IQ1_S/M и смешанные QF | 4,06–4,53 GB (провал); QF ~10,9 GB | равномерный IQ1 — «perceptual FAIL». QF на 8 шагах: картинка в порядке, звук ломается | 7 695 | `sources/hf/cards/MarxistLeninist__MiniMax-H3-FL2VA-Pruned-IQ1-GGUF.md` |
| brurpo/MiniMax-H3-NVFP4-GGUF | нативный GGML NVFP4 (sd.cpp, Blackwell) | 18,7 / pruned 11,4 GB | | 5 312 | `sources/hf/cards/brurpo__MiniMax-H3-NVFP4-GGUF.md` |
| PurpleBlazey/Minimax-H3-pruned-GGUF | imatrix и UD-кванты | — | автор: «Q5_K is the sweetspot», Q2_K — «meme quant» | 4 043 | `sources/hf/cards/PurpleBlazey__Minimax-H3-pruned-GGUF.md` |
| Merserk/MiniMax-H3-INT4-ConvRot | INT4 ConvRot pruned/full; TE int4 | 11,3 / 17,9; TE 15,0 GB | для 12 ГБ при RAM от 32 ГБ | 166 669 | `sources/hf/cards/Merserk__MiniMax-H3-INT4-ConvRot.md` |
| AX1Y2JP/MiniMax-H3-W4A8-ConvRot | sym W4A8 ConvRot pruned | 11,68 GB | нативно в свежем ComfyUI + comfy-kitchen | 39 322 | `sources/hf/cards/AX1Y2JP__MiniMax-H3-W4A8-ConvRot.md` |
| koongrizzly/…int4_W4A8_ConvRot_Pruned | зеркало W4A8 от Winnougan + VAE | — | для standalone-приложения | 29 919 | `sources/hf/cards/koongrizzly__MiniMax_H3_int4_W4A8_ConvRot_Pruned.md` |
| rockerBOO/minimax-h3-nvfp4-convrot | NVFP4 на fc1/fc2, qkv в INT8 ConvRot или FP8, блоки 0, 1, 47–49 в BF16; варианты INT4 | 20 GB (pruned) / 34 GB | «Quality has not been measured anywhere in this repo» | 34 284 | `sources/hf/cards/rockerBOO__minimax-h3-nvfp4-convrot.md` |
| coolthor/MiniMax-H3-pruned-NVFP4 | weight-only NVFP4 (200 линейных слоёв) + Heretic TE + turbo LoRA + узел CondCache | 11,67 ГиБ | пик 26 914 MiB на карте 32 ГБ; тот же seed даёт другой клип, чем INT8 | 7 602 | `sources/hf/cards/coolthor__MiniMax-H3-pruned-NVFP4.md` |
| MATLOWAI/minimax-h3-nvfp4 | все 200 линейных слоёв блоков в NVFP4 | 12,5 GB | медианная ошибка весов 9,4% (W4A8 — 7,3%, int8_convrot — 1,0%). На RTX PRO 6000 ×0,84–0,90 по времени; без Blackwell медленнее int8 | 5 689 | `sources/hf/cards/MATLOWAI__minimax-h3-nvfp4.md` |
| rzgar/minimax_h3_{fl2va,ref2va}_fp8_e4m3fn | fp8_e4m3fn, mxfp8, fp16attn_fp8 | 47 / 47,6 / 28,7 GB | в карточке fl2va: не меньше 8 шагов, dpmpp_2m / sgm_uniform | 3 665 / 2 958 | `sources/hf/cards/rzgar__minimax_h3_fl2va_fp8_e4m3fn.md` |
| intel-ai/MiniMax-H3-MXFP8-CT-AutoRound | MXFP8 (AutoRound), DiT + TE | TE 33,98 GB | | — | `sources/github/wildminder__awesome-minimax-H3.md` |
| AcademiaSD/MiniMax-H3-NF4, DiffSynth-Studio/MiniMax-H3-NF4 | NF4 (DiT + TE + VAE) | 41 GB комплект; 15,98 GB DiT | для обучения LoRA (LoRAlab); DiffSynth — от 8 ГБ VRAM | 778 / 0 | `sources/hf/cards/AcademiaSD__MiniMax-H3-NF4.md`, `sources/github/wildminder__awesome-minimax-H3.md` |
| WaveCut/MiniMax-H3-OrbitQuant-W4A4 | W4A4 OrbitQuant (DiT + TE) | 17,03 GB | на RTX 3090: 115 с против 175 с (**не проверено**) | 36 | `sources/hf/cards/WaveCut__MiniMax-H3-OrbitQuant-W4A4.md` |
| pottokao/…NVFP4-rotated / -SVDQuant-vLLM-Omni | rotated NVFP4 (nunchaku W4A4), SVDQuant r32 | ~12,5 GB / ~12,3 ГиБ | turbo v1.2 уже встроен; нужен свой загрузчик или патч SM120 | 0 / 21 | `sources/hf/cards/pottokao__MiniMax-H3-FL2VA-turbo-4step-v1.2-768p-NVFP4-rotated-T1.md` |
| yitongl/minimax-h3-nvfp4-lora-recovery | LoRA r128 поверх замороженного NVFP4 | — | ошибка слоёв 0,0913 → 0,0807; на всём ролике +0,07 дБ PSNR, «inside the noise» | 0 | `sources/hf/cards/yitongl__minimax-h3-nvfp4-lora-recovery.md` |
| multimodalart/MiniMax-H3-Pruned | diffusers AdaLN-pruned; TE на 50 слоёв; ConvRot при загрузке | 40,24 GB; TE 52,48 GB | эталон по pruned и ConvRot | 15 128 | `sources/hf/cards/multimodalart__MiniMax-H3-Pruned.md` |
| xmarre/…Pruned-Ref-Delta-Fused-r1024 | pruned fl2va + SVD-дельта ref2va | bf16 40,2; int8 ~21 GB | один файл на оба режима | 24 115 | `sources/hf/cards/xmarre__MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI.md` |
| smhfacct/…hybrid-models, hoidhxd/…Hybrid-b25-49-GGUF, berryber09/…hybrid-w4a8 | гибриды b15–b30; GGUF Q4_0/Q5_0; W4A8 (блоки 20–49 из ref2va) | ~21 / 10,6 / ~12,5 GB | | 0 / 6 021 / 2 604 | `sources/hf/cards/smhfacct__Minimax-H3-fl2va-ref2va-hybrid-models.md`, `sources/hf/cards/hoidhxd__MiniMax-H3-Hybrid-b25-49-GGUF.md`, `sources/hf/cards/berryber09__MiniMax-H3-ref2va-fl2va-hybrid-w4a8.md` |
| H3-Tiny/MiniMax-H3-Tiny-ComfyUI (+ 17B-Pruned-FL2VA, 17B-FewStep-Ref2VA) | 17B (−8 блоков), 14B OKD-Ref2VA; bf16 / int8 | 34,2 / 18,1 GB | FewStep — 6 шагов; Turbo LoRA с ними не проверены | 47 824 | `sources/hf/cards/H3-Tiny__MiniMax-H3-Tiny-ComfyUI.md` |
| ddalcu/MiniMax-H3-FL2VA-MLX-Serve-{8bit,4bit}, Vayden/…MLX-q8-extended-paged | MLX affine gs64; Q8 с пейджингом по 4 блока | — / ~33,4 GB | «Quantization buys footprint, not speed». У Vayden пик 14,95 GB на 640×384 | 4 012 / 1 956 / 14 | `sources/hf/cards/ddalcu__MiniMax-H3-FL2VA-MLX-Serve-8bit.md`, `sources/hf/cards/Vayden__MiniMax-H3-MLX-q8-extended-paged.md` |

**Слитые Turbo + квант** (LoRA уже вшита, отдельный загрузчик не нужен):
- `MATLOWAI/minimax-h3-fused-turbo-int8-convrot`, 422 813 загрузок. Состав: ref-delta r1024 + lightx2v FL2VA turbo 8-step v1.0 @1,0 (ранг 24 после ресайза Kijai) + стилевая LoRA Mystic v2.0 @0,7, всё в INT8 ConvRot g256. 21 GB, 4 шага. По карточке 76 с на 1152×640×243 кадра (RTX PRO 6000); вшитая LoRA экономит ~21 GB VRAM по сравнению с живой: пик 47,8 против 68,9 GB (`sources/hf/cards/MATLOWAI__minimax-h3-fused-turbo-int8-convrot.md`). Зеркала с той же карточкой: Jackxuanxuan, FenomAI, cgc-0521, chfm (`sources/hf/meta.tsv`).
- `PulpCut/MiniMax-H3-{,Ref2VA-}Turbo-INT8-ConvRot` (3 053 / 3 587); `ChrisColeTech/minimax-h3-turbo-GGUF` (25 866): FL2VA с 8-step, REF2VA с 4-step, Q4_K_M 11,4 GB; `molbal/MiniMax-H3-Turbo-GGUF`; `releaseworthy/dasiwa-minimax-h3-turbo-int8` (20 102; зеркало ref2va-файла с Civitai) (`sources/hf/cards/PulpCut__MiniMax-H3-Turbo-INT8-ConvRot.md`, `sources/hf/cards/ChrisColeTech__minimax-h3-turbo-GGUF.md`, `sources/hf/cards/releaseworthy__dasiwa-minimax-h3-turbo-int8.md`).

### 2.3. Энкодер и VAE отдельно

- Энкодеры: Comfy-Org bf16 / int8 / nvfp4_awq — 47,97 / 25,28 / 14,61 ГиБ; Abiray Q4_K_M — 13,58 GB; FP32 pruned от Felldude — 95,9 GB; intel-ai MXFP8 — 33,98 GB (`sources/github/wildminder__awesome-minimax-H3.md`). У ai-muninn W4A4-энкодер Qwen3-VL занимает 24 → 13,2 ГиБ и работает на 8,6% быстрее без видимой потери (`sources/web/ai-muninn__minimax-h3-on-modded-2080ti-22gb.md`).
- Видео-VAE: fp16 4,85 ГиБ; int8_convrot (Kijai) 2,95 GB; fp8mix 2,60 GB (dummy9996); ONNX/TensorRT от lihaoyun6 (декодер 4,5 GB или w4a16 1,2 GB). Дообученный аниме-VAE dully3865 убирает сеточные артефакты 16×16 и 256×256 (**не проверено**). LightVAE (lightx2v) и LightVAE от corechan — дистиллированный декодер на 26 блоков вместо 36 (`sources/github/wildminder__awesome-minimax-H3.md`).
- Аудио-VAE: fp32 577 MB. Сообщество предупреждает: его нужно держать в fp32, иначе рассинхрон A/V (гайд для 8 ГБ в `sources/github/wildminder__awesome-minimax-H3.md`). У dummy9996 есть bf16 на 289 MB (там же).

### 2.4. Качество квантов: что известно

- **W4A4 для DiT не годится, для энкодера — годится.** Относительная ошибка 0,2005 против 0,0110 у INT8; картинка «full-frame colour tearing». Пример набора: pruned INT8 DiT + W4A4 TE + VAE = 38,14 ГиБ на диске, ролик 15 с 1080p за 23 мин на 2080 Ti 22G (`sources/web/ai-muninn__minimax-h3-on-modded-2080ti-22gb.md`).
- **NVFP4 против INT8-ConvRot на RTX 5090** (864×480, 10 с): 11,67 против 19,53 ГиБ; 175 против 185 с; пик 26 914 против 28 581 MiB. Разницу в 5% автор объясняет тепловыми частотами, а не форматом (`sources/web/ai-muninn__minimax-h3-nvfp4-rtx5090.md`). С torch cu130 и ядром NVFP4 — минус 32–45% времени (`sources/web/ai-muninn__minimax-h3-rtx5090-config-stack.md`).
- INT8 ConvRot на A100: сначала в 2–3 раза медленнее FP8, позже в чистом окружении «~1.4× faster than FP8» (`sources/github/Comfy-Org__ComfyUI__issues-minimax-h3.md`). На Apple MPS он не работает — нет `torch._int_mm` (`sources/web/hackernews__h3-thread.md`).
- Ориентиры по VRAM (сообщество): pruned INT8 + NVFP4 TE — ~24 ГБ (12 ГБ возможно при большом RAM). Непрореженный int8 — ~48 ГБ, bf16 — 80 ГБ или 4 GPU (`sources/github/wildminder__awesome-minimax-H3.md`, `sources/web/dev.to__vram-6gb-to-20gb.md`).

### 2.5. Кванты H3-Image

| Репозиторий | Формат | Размер | Заметки | Загрузки |
|---|---|---|---|---|
| Hearthscale/MiniMax-H3-Image-GGUF | Q3_K_M…Q8_0, AdaLN в F16 | Q4_K_M 11,4 GB | Q2 не выпущен («lettering breaks»); на 3080 10 ГБ — 61 с | 840 375 |
| Abiray/MiniMax-H3-Image-GGUF | pruned Q2_K…Q5_K_M | Q4 11,6 GB | «indistinguishable from BF16» — **не проверено** | 338 350 |
| unsloth/MiniMax-H3-Image-GGUF | Q2_K…Q8_0 с замерами LPIPS | UD-Q4_K_XL 13,61 GB, LPIPS 0,172 | Q8_0: LPIPS 0,083 | 324 826 |
| unsloth/MiniMax-H3-Image-FP8 | INT8-ConvRot / FP8 | 20,27 GB | LPIPS 0,140 против 0,255 — лучше INT8-ConvRot | 24 835 |
| AtomicChat/MiniMax-H3-Image-GGUF | AD-Q2_K…Q8_0 | 7,57–21,45 GB | Q8_0 LPIPS 0,090; у Comfy int8 — 0,129, w4a8 — 0,274 | 59 617 |
| BennyDaBall/…Image-ComfyUI-NVFP4 | W4A4 | 12,54 GB | на 5090 на 20% быстрее W4A8, LPIPS тот же | 20 219 |
| speedyrulz/…image-int8-w4a8-hybrid | w4a8 в блоках 0–29, остальное int8 | 14,8 ГиБ | ошибка 5,2%; NVFP4 теряет 8–12% на слой | 4 256 |
| Mamad8/MiniMax-H3-Image-VAE | видео-VAE, дообученный под T=1 | 4,85 GB | только для картинок, видео портит | 10 679 |

Источники: `sources/hf/cards/Hearthscale__MiniMax-H3-Image-GGUF.md`, `sources/hf/cards/Abiray__MiniMax-H3-Image-GGUF.md`, `sources/hf/cards/unsloth__MiniMax-H3-Image-GGUF.md`, `sources/hf/cards/unsloth__MiniMax-H3-Image-FP8.md`, `sources/hf/cards/AtomicChat__MiniMax-H3-Image-GGUF.md`, `sources/hf/cards/BennyDaBall__MiniMax-H3-Image-ComfyUI-NVFP4.md`, `sources/hf/cards/speedyrulz__minimax-h3-image-int8-w4a8-hybrid.md`, `sources/hf/cards/Mamad8__MiniMax-H3-Image-VAE.md`.

Пробелы: ни в одном источнике нет размеров Comfy-Org `pruned_w6a8`, файлов Cobijada, MLX ddalcu и полного списка leejet. У Kijai, DmitryDB, smhfacct, DiffSynth и unsloth-FP8 в meta.tsv 0 загрузок при ненулевых лайках — скорее всего, счётчик просто не отдан.
## 3. Ускорители: Turbo- и дистилл-LoRA, дистиллированные чекпойнты

### 3.1. Опорная точка: база без дистилляции

- Чекпойнты уже CFG-дистиллированы: guider, `negative_prompt` и `guidance_scale` не используются, на шаг — один проход. Shift видео **12**, аудио **3** (`scheduler_config.json` и `audio_scheduler`) (`sources/github/MiniMax-AI__MiniMax-H3__README.md`, `sources/github/MiniMax-AI__MiniMax-H3__scripts-and-configs.md`). В diffusers и vLLM `num_inference_steps` считает точки сетки σ, проходов модели на один меньше (`sources/github/huggingface__diffusers__minimax-h3-docs.md`).
- Шагов по умолчанию: в vLLM-Omni — 50, euler; доступны также res_multistep, heun, dpmpp_2m (`sources/github/vllm-project__vllm-omni__recipe-MiniMax-H3.md`). В LightX2V — 29, планировщик `reference_blend` (`sources/github/ModelTC__LightX2V__minimax-h3.md`). В шаблонах ComfyUI — 20, res_multistep + simple (`sources/github/Comfy-Org__workflow_templates__minimax-h3-templates.md`). У ai-muninn переход с 20 на 14 шагов «без видимой разницы» сэкономил 18,9% времени (`sources/web/ai-muninn__minimax-h3-rtx5090-speedup-vsr.md`).
- Как дистилляция бьёт по звуку. Сетка из 25 конфигураций (RTX 5080, 960×544) показала: 4 шага портят тихие и протяжные вокальные регистры (шёпот, долгие ноты), сводят акценты и смену языков к усреднённому звучанию, жёсткие склейки превращаются в наплывы. Сила LightX2V 0,75 даёт больше стабильности, 1,0 — больше скорости. Нужен ComfyUI ≥ 0.30.0 с исправлением звука в сэмплере от Kijai (`sources/github/wildminder__awesome-minimax-H3.md`). Замер на RTX 4070 12 ГБ (576×832, 124 кадра): база 20 шагов — 273 с; Larry v4 на 8 шагах — 130 с; LightX2V 8-step v1.0 на 4 шагах — 79 с (×3,44), VRAM та же (там же).

### 3.2. Таблица

Сокращения: «Ш» — шаги (NFE); «σ-shift» — видео/аудио. Все ускорители работают с CFG 1.

| Ускоритель | Тип | Ш | Сила | Сэмплер / планировщик | σ-shift | Режим | Обучено на | Особенности | Загрузки | Источник |
|---|---|---|---|---|---|---|---|---|---|---|
| lightx2v `fl2v_turbo_4step_v0.1` | LoRA r128 (alpha нет → 8) | 4 | 0,75 (Kijai), 1,0 | er_sde / sa_solver (Kijai), euler | 12/3 | FL2VA/T2VA | 544p | preview; ~1,3 GB; ресайз Kijai до r21 — 300 MB | 1 835 232 (весь репозиторий) | `sources/hf/cards/lightx2v__Minimax-h3-Turbo.md`, `sources/web/comfyui-wiki__turbo-lora-4step-news.md` |
| lightx2v `fl2v_turbo_8step_v1.0` | LoRA r128, α8 | 8 (работает и на 4) | 1,0 | euler | 12/3 | FL2VA/T2VA | 544p | по умолчанию в шаблонах ComfyUI и ModelTC | ↑ | `sources/web/comfyui-wiki__turbo-lora-v1-8step-news.md`, `sources/github/ModelTC__Minimax-H3-Turbo__README-setup.md` |
| lightx2v `fl2v_turbo_{4,8}step_v1.0_768p` | LoRA r128 (α128 / α8) | 4 / 8 | 1,0 | euler | **6/3** | FL2VA | 768p | ComfyUI-версии 1,82 GB (QKV слиты, vLLM их не принимает), diffusers — 1,29 GB | ↑ | `sources/hf/cards/lightx2v__Minimax-h3-Turbo.md` |
| lightx2v `fl2v_turbo_4step_v1.1/v1.2_768p` | LoRA | 4 (есть и 8) | 1,0 | euler | 6/3 | FL2VA | 768p | v1.2 входит в Pika Flex Turbo | ↑ | там же |
| lightx2v `fl2v_turbo_8step_v1.2_768p` | LoRA r128 + адаптеры AdaLN r16 (51 пара) | 8 | 1,0 | euler | **12/3** | FL2VA | 768p | на pruned/curve-базе 51 пара AdaLN теряется (размерности 2688 и 8 не совпадают, issue #7). Нужен кэш AdaLN | ↑ | `sources/hf/cards/lightx2v__Minimax-h3-Turbo.md`, `sources/github/ModelTC__Minimax-H3-Turbo__README-setup.md` |
| lightx2v `ref2v_turbo_4step_v0.1` / `ref2v_turbo_8step_v1.0_768p` (+ v1.1 8step) | LoRA r128, α8 | 4 / 8 | 1,0 | euler | 12/3 и **6/3** (у 768p) | Ref2VA | 544p / 768p | issue #10: «металлический» звук, особенно у ref2v; выше 0,5 MP картинка разваливается | ↑ | там же |
| lightx2v/Minimax-h3-Turbo-SLA | LoRA + разреженное внимание SLA (sparsity 0,85, sage2) | 4 | 1,0 | `training_euler` | 6/3 | FL2VA | 768p | «~2,5× на 5090» (**не проверено**) | 0 (114 лайков) | `sources/hf/cards/lightx2v__Minimax-h3-Turbo-SLA.md` |
| lightx2v/Minimax-H3-Turbo-Fusion | слияние: LightX2V ref2v ×1,0 + Pika Flex ×0,4 + Larry v4 EMA ×0,4; Full и Lite r32 | 4 / 8 | 1,0 | euler | 12/3 | только Ref2VA | — | другие компоненты поверх не добавлять | 24 351 | `sources/hf/cards/lightx2v__Minimax-H3-Turbo-Fusion.md` |
| larryvrh/MiniMax-H3-Turbo-Lora `v4_step600_ema` (рекомендован), v4_step600, линия v1 (ckpt500/850, ema) | LoRA (attn/MLP r64, AdaLN r16), ~744 MB | 4–8, лучше 6–8 | 1,0 (1,05–1,2 при размытии; 0,8–0,95 при пересвете) | свой Turbo Sampler или euler, simple | 12/3 | FL2VA (подходит к любой базе) | — | выше 8 шагов пересвет; на 4 шагах шлейфы при быстром движении. На pruned-базе штатный `LoraLoaderModelOnly` применяет 208 из 259 модулей (без AdaLN). Свой узел заново вводит AdaLN. Кэш-узлы поверх не ставить | 195 918 | `sources/hf/cards/larryvrh__MiniMax-H3-Turbo-Lora.md`, `sources/github/Larryvrh__ComfyUI-MiniMax-H3-Turbo__README.md`, `sources/web/instasd__turbo-lora-4-steps.md` |
| drbaph/MiniMax-H3-Turbo-Lora-ComfyUI | конверсии Larry под pruned-базу (208 пар), SVD-ресайзы LightX2V (r21–64), HyperFlow 8-step, TaoMate 3-step, VDN DMD 8-step | 6–8 (Larry) | 1,0 | euler + beta | 12 / **4–6** | FL2VA, Ref2VA | — | в карточке два разных списка σ для HyperFlow, расхождение не разрешено | 250 564 | `sources/hf/cards/drbaph__MiniMax-H3-Turbo-Lora-ComfyUI.md` |
| Pika-Labs/Pika-H3-Flex-Turbo-LoRA | склейка рангов: стиль Flex ×0,5–0,6 + turbo (LightX2V v1.2 / ref2v v0.1 / pDMD 2-NFE); 3,9 GB, Lite r32 0,31 GB | 2 (черновик, Ref2VA) / 4 / 8 HD | 1,0 | euler / lcm / res_multistep, simple | **6/3** FL2VA, **12/3** Ref2VA | оба | — | звук хуже ниже ~6 шагов. На NVFP4-базе тихо портит вывод, нужен int8. 848×480 падает, брать 864×480. 5090, 864×480, 5 с: база 14 шагов — 38,0 с, Flex — 20,8 с, Turbo-4 — 16,3 с | 223 682 | `sources/hf/cards/Pika-Labs__Pika-H3-Flex-Turbo-LoRA.md`, `sources/web/ai-muninn__pika-h3-flex-rtx5090-whitebox-orbit.md` |
| Pika-H3-Flex (сам по себе) | адаптеры 128/128 | 6 | 1,0 | LCM | 12/3; σ `1,.75,.5,.25,.1,.04,0` | FL2VA, Ref2VA | — | — | — | `sources/github/ModelTC__LightX2V__minimax-h3.md` |
| FastH3 8-Step V2 (FastVideo, DMD2) | полный чекпойнт + BlockSparse VSA (keep 10% с 20% расписания) | 8 | — | res_multistep, simple | **10/3** | t2va / FL2VA | — | «жертвует частью качества движения и звука»; нужен ComfyUI ≥ 0.36 | — | `sources/web/docs.comfy.org__h3-fastvideo.md` |
| FastH3 4-step Preview V1 / Omni 8-Step V1 | LoRA, вливается при загрузке (Dense / VSA); Omni — NVFP4 QAD и FP8 | 4 / 8 | 1,0 | — | — | t2va / Ref2VA | — | Omni: 124 кадра за 41 с на 8×RTX PRO 5000 | — | `sources/github/vllm-project__vllm-omni__recipes-FastH3-Regenerate2K-TaoMate.md` |
| drozbay/MiniMax-H3-FastH3-Preview-LoRA | SVD-извлечение из FastH3 v0.2 (pruned r128 1,33 GB) | 4 | 1,0 | euler, simple | 12/3 | fl2va/t2va | — | PSNR к FastH3 — 17,6 дБ при потолке 19,9: дельта плохо приближается низким рангом | 2 169 | `sources/hf/cards/drozbay__MiniMax-H3-FastH3-Preview-LoRA.md` |
| Iwannapose pDMD 2nfe / 4nfe | LoRA r128 (attn + FFN, 208 ключей) | 2 / 4 | строго 1,0 | — | 12/3 | проверено на ref2va int8 | — | для 4 NFE брать только `_v6` | 4 214 / 2 494 | `sources/hf/cards/Iwannapose__minimax_h3_pdmd_2nfe_comfyui.md`, `sources/hf/cards/Iwannapose__minimax_h3_pdmd_4nfe_comfyui.md` |
| OliviaRossi/Minimax-H3-Hybrid-DMAD-PDMD | чекпойнт: FL2VA (0–19) → S-кривая → Ref2VA (35–49), вшиты DMAD ×0,85 и PDMD ×0,333; W6A8 14,95 / INT8 18,89 GB | 4–8 | — | не указан | не указан | гибрид | — | заявления о качестве — маркетинг | 32 154 | `sources/hf/cards/OliviaRossi__Minimax-H3-Hybrid-DMAD-PDMD.md` |
| DMAD (ZhengmingYu, ByteDance + TAMU) | LoRA `dmad_…4step_full_critic` | 6–8 | 1,0 | er_sde / euler, simple | **12/2** | — | — | «видео на уровне Larry v4, звук лучше» — мнение автора страницы | — | `sources/web/civitai__turbo-loras-collection.md` |
| S3ckn11g0r/CROWN-HyperPDMD-HD | LoRA (средний r38 / r64) | 3–4 | 1,0 | euler, beta (α=β=0,6) | FL2VA 12/3; Ref2VA **9/3** + `ref_image_size="max"` | оба | — | «одна LoRA на bf16/int8/GGUF, <9 GiB» — **не проверено** | 2 609 | `sources/hf/cards/S3ckn11g0r__CROWN-MiniMax-H3-HyperPDMD-HD-3-4Step-LoRA.md` |
| alibaba-pai/MiniMax-H3-Acc-LoRAs (PDD) | LoRA r64 + 32 копии выходных голов (по 4 на каждый проход) | 8 | строго 1,0 | только euler | 12/3 | FL2VA, Ref2VA | — | не совмещается с другими адаптерами; нужен diffusers ≥ 0.40 | 46 784 | `sources/hf/cards/alibaba-pai__MiniMax-H3-Acc-LoRAs.md` |
| aptech0081/MiniMax-H3-PDD-ComfyUI | Ref2VA с вшитым PDD (pruned int8 21,06 / полный 34,12 GB) + LoRA | 8 (4) | — | euler, simple | 12/3 | Ref2VA, FL2VA | — | нативно с ComfyUI PR #17157 | 16 004 | `sources/hf/cards/aptech0081__MiniMax-H3-PDD-ComfyUI.md` |
| taobao-content-ai PTD-RefAsymOPD | своя LoRA r128 на каждый шаг поверх голов PDD | 8 / 6 | — | — | 12/3 | Ref2VA | — | опорные картинки — 768 по короткой стороне, на дефолтных 2048 качество падает | 20 969 | `sources/hf/cards/taobao-content-ai__MiniMax-H3-PTD-RefAsymOPD-LoRAs.md` |
| TaoMate-H3 (TaoLive) | LoRA r128, потоковая генерация окнами по 5 с | 3 на фазу (аудио ведёт учитель, 9 шагов) | 0,75 (Civitai) | — | — | FL2VA | — | не сливать с FP8: дельта теряется, картинка мылится | — | `sources/github/vllm-project__vllm-omni__recipes-FastH3-Regenerate2K-TaoMate.md`, `sources/web/civitai__turbo-loras-collection.md` |
| Wuli-art/MiniMax-H3-Turbo; sandpies Wuli-…-ComfyUI | 4 шага — LoRA + выходные проекции; 6 и 8 — полные трансформеры | 4 / 6 / 8 | — | euler, simple | 4 шага **6/3**, 8 шагов 12/3 | FL2VA, Ref2VA 8 | — | у sandpies проверено только на CPU | 10 277 / 5 621 | `sources/hf/cards/Wuli-art__MiniMax-H3-Turbo.md`, `sources/hf/cards/sandpies__Wuli-MiniMax-H3-Turbo-ComfyUI.md` |
| Beidouqixing flashgen | VSD/DMD data-free, r64 (qkv/out/fc/adaln) | 4 | строго 1,0 | euler; σ [1,.7,.4,.15,0] | 12/3 | v1 T2VA; v2 I2VA/Ref2VA | 1344×768 | обучено на Ascend NPU | 3 673 | `sources/hf/cards/Beidouqixing__minimax-h3-4step-lora-flashgen.md` |
| ModelsLab/MiniMax-H3-Turbo-RL | Larry v4 + Flow-DPO LoRA r64 (ветки не сливать) | 10 NFE («E4a»: Heun + узел u=0,04) | — | Heun / euler | — | Ref2VA | — | авторы признают, что прирост почти целиком от сэмплера | 710 | `sources/hf/cards/ModelsLab__MiniMax-H3-Turbo-RL.md` |
| tutututututu 20→8 NFE | LoRA (step100/200/300) | 8 | 0,8 | euler; свои σ | 12/3 | FL2VA | — | — | 0 | `sources/hf/cards/tutututututu__Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA.md` |
| moe-kill/H3-Turbo-Neat | LoRA Classic / Prime (Neat × DMAD) | 4 / 7 | — | «DMAD re-noise» | 12/**2** | — | — | у Classic «электронный» звук | 3 101 | `sources/hf/cards/moe-kill__H3-Turbo-Neat.md` |
| Raretutor Dual-Turbo | две LoRA (Larry + TaoMate + DMAD) | 6 (5–9) | — | lcm, simple | — | — | — | способ слияния описан расплывчато | 1 634 | `sources/hf/cards/Raretutor__MiniMax-H3-Dual-Turbo-Lora-ComfyUI.md` |
| SanDiegoDude 6-step; InstantX diffusers; NaukNauk merged | переименование / PEFT-конверсия / вливание Larry | 6 / 4 / 8 | 4,0 → 1,0 (расписание силы) / — / 1,0 | euler beta57 / — / SGLang | — | FL2VA (+Ref2VA у NaukNauk) | — | InstantX не поддерживает pruned-базу | 0 / 3 365 / 0 | `sources/hf/cards/SanDiegoDude__H3-Turbo-6-Step-LoRA-Comfy.md`, `sources/hf/cards/InstantX__MiniMax-H3-Turbo-Lora-Diffusers.md`, `sources/hf/cards/NaukNauk__minimax-h3-turbo-fl2va-merged.md` |
| rzgar 8Step motion enhancer | NSFW-дообучение поверх LightX2V 8step | 8 | — | — | — | fl2v / ref2va | — | исходную LightX2V при этом отключать | 6 842 / 2 419 | `sources/hf/cards/rzgar__minimax-h3_fl2v_8Step_motion_enhancer.md` |
| H3-Tiny 17B FewStep-Ref2VA | дистиллированный pruned-чекпойнт на 17B | 6 (4–8) | — | euler + свои Few-Step Sigmas | — | Ref2VA | — | bf16 34,2 / int8 18,1 GB | 2 633 | `sources/hf/cards/H3-Tiny__MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1.md` |
| Veda-Sparse R2VA Preview | обученный предиктор разреженного внимания (FP8) | 8 (с Larry) | — | — | — | R2VA | — | внимание быстрее в 2,06–2,65×, весь прогон — в 1,37–1,69× (замер авторов) | 26 984 | `sources/hf/cards/Veda-Sparse__Minimax-H3-R2VA-Veda-Preview.md` |
| mvp-lab RAVEN Streaming | каузальная потоковая LoRA r128 | 4 | — | — | — | — | 768×1376, 192 кадра | недообучена по словам авторов | 0 | `sources/hf/cards/mvp-lab__MiniMax-H3-RAVEN-Streaming-LoRA.md` |
| akhaliq/minimax-h3-image-pdmd-2step | pDMD r128 (и влитый чекпойнт 40 GB) | 2 (у стока 6) | 1,0 | euler, simple | 12/3 | **H3-Image** | — | — | 48 132 | `sources/hf/cards/akhaliq__minimax-h3-image-pdmd-2step.md` |

### 3.3. Что важно знать про LoRA-ускорители

- **Масштаб alpha.** Дельта считается как `scale·alpha/rank`. Файлы LightX2V с α8 при rank 128 работают в 16 раз слабее файлов с α128; alpha берётся из метаданных. В конверсии Kijai v0.1 матрица B уже домножена на 0,125, у T8 — на alpha 8/128 = 0,0625 (`sources/hf/cards/lightx2v__Minimax-h3-Turbo.md`, `sources/hf/cards/t8star__minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.md`). Загрузчик должен читать alpha из файла, иначе сила будет неверной.
- **Shift для 768p-версий** — 6/3, для 544p-версий и v1.2 8step — 12/3. Смешивать нельзя. vLLM жёстко проверяет пару у Pika (`sources/hf/cards/lightx2v__Minimax-h3-Turbo.md`, `sources/hf/cards/Pika-Labs__Pika-H3-Flex-Turbo-LoRA.md`).
- **Опорные картинки** для дистиллированных моделей: режим `match` или 768 по короткой стороне, как при обучении; дефолт diffusers (2048) хуже (`sources/hf/cards/lightx2v__Minimax-h3-Turbo.md`, `sources/hf/cards/taobao-content-ai__MiniMax-H3-PTD-RefAsymOPD-LoRAs.md`).
- **Цена LoRA в vLLM** (8×B300, 1344×768, 15 с): шаг с LoRA дороже на ~10%, переключение запросов с LoRA и без стоит ~24 с. База 49 проходов ~105 с против Turbo v1.2 на 4 проходах ~9 с (×11,7 по денойзу, ×6,8 по всему конвейеру) (`sources/hf/cards/lightx2v__Minimax-h3-Turbo.md`).
- **Конверсии t8star и RunningHubAI** — в основном зеркала с нулём загрузок. Варианты `curveproj1025` спроецированы на AdaLN конкретных дообученных баз и подходят только к ним (`sources/hf/cards/t8star__*.md`, `sources/hf/cards/RunningHubAI__*.md`).
- В нашем порте: Turbo lightx2v на 720p × 10 с — 20 шагов 18,4 мин, 8 шагов 8 мин, 4 шага 4,2 мин. Разные LoRA дают разную цветопередачу, поэтому на фильм берётся одна LoRA. По умолчанию сейчас полные шаги без LoRA, а для Ref2VA — Turbo-Fusion (`sources/local/synaptix-perf.md`, `sources/local/synaptix-probe-findings.md`, `sources/local/synaptix-git-log-h3.md`).
## 4. Управление и инструментальные модели

### 4.1. ControlNet

| Модель | Что умеет | Как подключается | Загрузки | Источник |
|---|---|---|---|---|
| alibaba-pai/MiniMax-H3-Fun-Controlnet-Union (v1, устарела) | Canny, Depth, HED, MLSD, Pose + инпейнт (`control_in_dim=49`: латент, маскированный латент, маска) | ветка управления в 5 блоках (0/10/20/30/40), проекция с нулевым гейтом; `guidance_scale=1`; сила через `control_context_scale` (0–1) | 37 480 | `sources/hf/cards/alibaba-pai__MiniMax-H3-Fun-Controlnet-Union.md` |
| alibaba-pai/…-Union-2.0 | 8 условий: + Scribble, Layout (боксы как в Wan-VACE), Gray; инпейнт по рецепту `post_norm`; `control_apply_audio=false` | 10 блоков управления (через каждые 5 слоёв); 12,6 GB fp32. **Конфиг v1 на этом чекпойнте падает молча**: `strict=False` выбрасывает блоки 5–9 | 31 655 | `sources/hf/cards/alibaba-pai__MiniMax-H3-Fun-Controlnet-Union-2.0.md`, `sources/github/wildminder__awesome-minimax-H3.md` |
| Comfy-Org `minimax_h3_fun_controlnet_union_pruned_{bf16,int8_convrot}` | патч поверх Ref2VA pruned int8; поза — через подграф SDPose | ComfyUI ≥ 0.35; силу выше 1 поднимать, только если результат уходит от контроля; с 4-шаговым turbo звук и движение чуть хуже | — | `sources/web/docs.comfy.org__h3-fun-controlnet.md`, `sources/hf/cards/Comfy-Org__MiniMax-H3.md` |
| akhaliq/MiniMax-H3-Image-Fun-Controlnet-Union-2.0 | для картинок: работает только через видеостек (22 одинаковых кадра canny, 40 шагов, scale 0,6–1,0) | на дистиллированной H3-Image ветка «искажает, но не управляет»; надёжное условие — canny | 504 | `sources/hf/cards/akhaliq__MiniMax-H3-Image-Fun-Controlnet-Union-2.0.md` |

Обучение ControlNet в VideoX-Fun: lr 2e-5, warmup 100, 311 кадров, max_grad_norm 0,05, 10% условий обнуляются (`sources/github/aigc-apps__VideoX-Fun__minimax-h3-fun-controlnet.md`).

### 4.2. Апскейлеры и 2K без API

- **LBH-123-AI/Minimax_h3_latent_Upscaler** (452 149, Apache-2.0). Работает прямо на 24-канальном латенте: декод, апскейл и энкод через VAE не нужны. Экономит время, а не VRAM. 3D v1 — 345M параметров, масштабы 2–4×, 691 MB bf16. 2D v1 (10.11.2026) — 182M, 6 уровней резкости, масштабы 1,1–4×; заявлено латентное MSE −84,7% и PSNR 30,47 против 23,54 у трилинейного (**не проверено**) (`sources/hf/cards/LBH-123-AI__Minimax_h3_latent_Upscaler.md`, `sources/github/LBH-123-AI__Comfyui_Minimax_h3_latent_Upscaler.md`). Используется в ResCascade: генерация на 1 MP, затем доводка до 1080p/1440p (`sources/github/wildminder__awesome-minimax-H3.md`).
- **SOLRICKS/MiniMax-H3-Upscaler** (6 419): 3D-апскейлер fp16 + LoRA-рефайнер `h3_upscale_refiner_int8`. Старт от 2 MP, рефайнер — на базе без turbo и стилевых LoRA (`sources/hf/cards/SOLRICKS__MiniMax-H3-Upscaler.md`).
- **BobMcoy/Minimax-H3-Temporal-Latent-Upscaler** (0): во втором проходе растягивает быстрые 17-кадровые блоки в 2–3× против смаза при быстром движении (`sources/hf/cards/BobMcoy__Minimax-H3-Temporal-Latent-Upscaler.md`).
- **Kijai Regenerate-2K refiner LoRAs** (rank 16/384) — попытка повторить закрытый H3-Regenerate-2K (`sources/github/wildminder__awesome-minimax-H3.md`). Карточки в наших материалах нет.
- Альтернатива без модели — RTX VSR после генерации: вдвое быстрее, но размытие он не исправляет (`sources/web/ai-muninn__minimax-h3-rtx5090-speedup-vsr.md`, `sources/web/ai-muninn__minimax-h3-rtx5090-character-lock.md`).

### 4.3. Замена персонажа, лица, удаление, инпейнт

| Модель | Режим | Как пользоваться | Загрузки | Источник |
|---|---|---|---|---|
| akatz-ai/MiniMax-H3-Character-Swap-LoRA (V1.5) | Ref2VA | сила 1,0, без триггера. Проверенный рецепт: Ref2VA INT8 + turbo 8-step, er_sde, CFG 1; маски SAM через H3 Relay 2.0; обучена в ai-toolkit | 129 428 | `sources/hf/cards/akatz-ai__MiniMax-H3-Character-Swap-LoRA.md` |
| akatz-ai/…-Image-Character-Swap-LoRA | H3-Image | 6 шагов; `<Picture 1>` — сцена, `<Picture 2>` — лист персонажа; r16, 140 пар | 18 604 | `sources/hf/cards/akatz-ai__MiniMax-H3-Image-Character-Swap-LoRA.md` |
| akhaliq/MiniMax-H3-Image-Character-Swap | H3-Image | видео-LoRA akatz вшита в DiT картинок; заявлена замена между стилями (**не проверено**) | 484 | `sources/hf/cards/akhaliq__MiniMax-H3-Image-Character-Swap.md` |
| Alissonerdx/Minimax-H3-ComfyUI | Ref2VA | набор экспериментальных LoRA с триггерами: `body_swap:`, `head_swap:`, `motion_transfer:` (поверх адаптера ostris v2), `vfx_edit:`, `style_transfer:`, `live_wallpaper:`, LMS-резкость | 329 372 | `sources/hf/cards/Alissonerdx__Minimax-H3-ComfyUI.md` |
| UntMods/FaceSwap_MiniMaxH3_REF2VA | Ref2VA | триггер `Faceswap`, сила 1 (Civitai 3 881) | 0 | `sources/hf/cards/UntMods__FaceSwap_MiniMaxH3_REF2VA.md` |
| akatz-ai/…-Person-Remover-LoRA | Ref2VA | маска SAM 3.1, залитая зелёным; сила 1,0, 12 шагов | 2 916 | `sources/hf/cards/akatz-ai__MiniMax-H3-Person-Remover-LoRA.md` |
| jialidddh/…People-Vehicle-Removal-LoRA | Ref2VA | убирает людей и машины, в том числе на 360°; 50 шагов, сила 1,0 | 299 | `sources/hf/cards/jialidddh__MiniMax-H3-People-Vehicle-Removal-LoRA.md` |
| diffusers-modular/minimax-h3-inpainting | FL2VA/Ref2VA | модульный инпейнт видео **и звука**: маска звука считается в 40 латентах/с, а не в 24 fps | 0 | `sources/hf/cards/diffusers-modular__minimax-h3-inpainting.md` |
| VoidEclipseAI/…ReFrame-VideoInpainting-LoRA | Ref2VA | дорисовывает грубый монтаж; 768p, 6 шагов вместе с Pika Flex | 157 | `sources/hf/cards/VoidEclipseAI__MiniMax-H3-ReFrame-VideoInpainting-LoRA.md` |
| ausboss/MiniMax-H3-Image-Outpaint-LoRA | H3-Image | заливка `#808080`, до ~4 MP, LCM 6 шагов | 784 | `sources/hf/cards/ausboss__MiniMax-H3-Image-Outpaint-LoRA.md` |
| supermansundies/…Image-Pose-Transfer-Lora | H3-Image | «Put the person from Picture 2 into the reference pose shown in Picture 1…» | 2 784 | `sources/hf/cards/supermansundies__Minimax-H3-Image-Pose-Transfer-Lora.md` |
| Carasibana/ComfyUI-H3-FaceRefine | узел | FaceDetailer для видео: трекинг лица → кроп → перегенерация в H3 → вклейка | — | `sources/github/Carasibana__ComfyUI-H3-FaceRefine.md` |

### 4.4. Ракурсы, движение, потоковая генерация, служебные модули

- Ракурсы:
  - **pablodawson/360-Orbit** (24 050, FL2VA): одно фото как первый и последний кадр даёт замкнутую орбиту; сила 1,0, 28 шагов.
  - **MATLOWAI/ORB360-CardSpin** (4 426, Ref2VA): триггеры `ORB360_CW` / `ORB360_CARDSPIN`.
  - **matlod/turnaround** (2 487): 5 ракурсов за проход.
  - **Cseti/CrossView-Warp** (11 003, Ref2VA): новый ракурс по depth-warp, триггер `crossview`, 0,8–1,0.
  - **rehan-fal/360-equirect** (`eqr360`, 0,75).

  Источники: `sources/hf/cards/pablodawson__MiniMax-H3-360-Orbit-LoRA.md`, `sources/hf/cards/MATLOWAI__MiniMax-H3-ORB360-CardSpin.md`, `sources/hf/cards/matlod__minimax-h3-turnaround.md`, `sources/hf/cards/Cseti__MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1.md`, `sources/hf/cards/rehan-fal__minimax-h3-360-equirect-lora.md`. Для орбиты ai-muninn предупреждает: текстовые орбиты «плывут», а с черновой сценой из Blender выходит 6 из 6 (`sources/web/ai-muninn__pika-h3-flex-rtx5090-whitebox-orbit.md`).
- Движение:
  - **LiconStudio/Minimax-H3-VBVR** (16 562, I2V): меньше дублирования объектов и прохождения сквозь них (`sources/hf/cards/LiconStudio__Minimax-H3-VBVR.md`).
  - **MATLOWAI/Variable-Clock** (340): нативный slow-motion, результаты смешанные (`sources/hf/cards/MATLOWAI__MiniMax-H3-Variable-Clock.md`).
  - **MATLOWAI/Motion-Adapter** (5 711): работает только в проходе de-rope ComfyUI-MAINodes (`sources/hf/cards/MATLOWAI__MiniMax-H3-Motion-Adapter.md`).
  - **kentskooking/AnimateDiff-LoRAs** (FL2VA, триггер `animatediff`) (`sources/hf/cards/kentskooking__MiniMax-H3-AnimateDiff-LoRAs.md`).
- Потоковая генерация и длинные ролики: RAVEN (4 NFE, 192 кадра в 24 GiB) и TaoMate (окна по 5 с) — раздел 3. Узлы `NikoDemon80__ComfyUI-H3-Motion-Context`, `knoic__ComfyUI-MiniMaxH3-PrefixStream`, `tritant__ComfyUI_MiniMax_H3_Extender` (`sources/github/`).
- Служебные модули:
  - **ReDiverse** (r32, только первый шаг turbo-студента) возвращает разнообразие между seed (`sources/hf/cards/ReDiverse__ReDiverse-MiniMax-H3.md`).
  - **lihaoyun6 Ref-Patch** (148 MB): FL2VA частично ведёт себя как Ref2VA.
  - **RefMods**: крохотные (0,5–3,5 MB) пакеты референсов на персонажа для узла MiniMaxH3Mod.
  - **MPEG-Diffusion**: смешанное разрешение по регионам (исследовательские веса).
  - **TAE** Kijai: превью.

  Источник по Ref-Patch, RefMods, MPEG-Diffusion и TAE: `sources/github/wildminder__awesome-minimax-H3.md`.

## 5. LoRA качества, стиля и концептов

NSFW-коллекции заметны по загрузкам: Hearmeman 78 677, lynaNSFW 11 123, наборы EllaPriest45 на сотни LoRA. Здесь они не разбираются (`sources/hf/meta.tsv`, `sources/github/wildminder__awesome-minimax-H3.md`).

### 5.1. Топ по загрузкам (HF, без turbo и квантов)

| # | Модель | Группа | Загрузки |
|---|---|---|---|
| 1 | Alissonerdx/Minimax-H3-ComfyUI | инструменты (swap / motion / VFX) | 329 372 |
| 2 | fal/MiniMax-H3-Realism-People-LoRA | реализм | 143 314 |
| 3 | akatz-ai/Character-Swap | инструменты | 129 428 |
| 4 | JOKER141/MiniMax-H3-Combat-Base-V2 | экшн | 39 124 |
| 5 | pablodawson/360-Orbit | камера | 24 050 |
| 6 | akatz-ai/Image-Character-Swap | H3-Image | 18 604 |
| 7 | Jojocodex/spatial-physics | физика | 17 927 |
| 8 | Jojocodex/wushu-action v7 | экшн | 17 894 |
| 9 | Jojocodex/Camera-Motion | камера | 17 469 |
| 10 | LiconStudio/VBVR | физика / движение | 16 562 |
| 11 | Cseti/CrossView-Warp | ракурс | 11 003 |

Источник загрузок — `sources/hf/meta.tsv`. Топ Civitai (тип LORA): InstaPic 33 057, Furry Enhancer 22 286, Better Human Motion 20 721, Realism Slider 13 617, BUNNY Motion Repair 10 851, Cinematic V2 10 021. У части моделей несколько базовых, и счётчик у них общий (`sources/web/civitai__top-h3-models-by-downloads.md`).

### 5.2. По группам: сила и триггеры

**Реализм и улучшайзеры**
- **fal Realism-People**: T2V/I2V/R2V, триггер `r34l1sm`, сила 1,0 (0,6–0,8 мягче). Расхождение: в карточке «rank 32, 1500 шагов, 176 клипов, высокое разрешение», а гайд fal выигрышным считает rank 16, 5000 шагов, lr 1e-4 и пишет, что высокое разрешение не окупилось (`sources/hf/cards/fal__MiniMax-H3-Realism-People-LoRA.md`, `sources/web/fal.ai__how-to-train-lora.md`).
- **Zakyrie beauty-clarity-enhancer** (1 518; зеркало Civitai vrgamedevgirl): FL2VA, эпоха 5 мягче, эпоха 10 сильнее, сила 1,0, только в первом проходе (`sources/hf/cards/Zakyrie__minimax-h3-beauty-clarity-enhancer.md`).
- Civitai:
  - Realism Slider (rank 1, против жирного блеска кожи);
  - Detail Slider (от −2 до 2);
  - Polyhedron eyes/skin/hands (`perfe8ct`);
  - Natural Face & Speech (0,6–0,8);
  - Photorealistic (`ph0t0r34l`).

  Источник: `sources/web/civitai__top-h3-models-by-downloads.md`, `sources/web/stablediffusiontutorials__19-lora-models.md`.

**Стили**
- **lovis93 STUDIO 1939** (2 228): анимация 1930-х, триггер `gulliv3r,`, сила 1,0 (0,4–0,8 для смешения) (`sources/hf/cards/lovis93__studio-1939-old-animation-lora-minimax-h3.md`).
- **Monet** (`monetstyle`, вес ≥ 2) (`sources/hf/cards/neonforestmist__Minimax_H3_Monet_Style_LoRA.md`).
- **SOLRICKS 3D Animation** (`3dsrx`; 0,8 для людей, 1,0 для зверей и окружения) (`sources/hf/cards/SOLRICKS__3D-Animation-Style-MiniMax-H3.md`).
- **Grimdark 3D** (Civitai): `grimdark3dstyle`, сила 1,3–2,0, потому что база тянет к фотореализму (`sources/web/civitai__grimdark-3d-style-lora.md`).
- **DiffSynth LineartAnime**: раскраска лайн-арта через Ref2VA (`sources/hf/cards/DiffSynth-Studio__MiniMax-H3-LoRA-LineartAnime.md`).
- Ещё из сводного списка:
  - vh5tape, 16Bit Pixel;
  - Claymation (`claymation`), 2D Anime (`2danime`);
  - Anime2Realism (`LumiReal`), Real2Anime (`r2anime style`);
  - кино-стили 1950–80-х (neph1).

  Источник: `sources/github/wildminder__awesome-minimax-H3.md`.
- Общее наблюдение: стилевым LoRA на H3 обычно нужна сила **выше 1** (Monet ≥ 2, Grimdark 1,3–2,0), потому что база сильно тянет к фотореализму (источники выше).

**Движение, экшн, камера**
- **JOKER141 Combat-Base-V2**: без триггера или `prfight2` / `prfin1`; res_multistep + simple или euler + beta (`sources/hf/cards/JOKER141__MiniMax-H3-Combat-Base-V2.md`).
- **GunFu**: триггер `BUNNY`, 0,9 (`sources/hf/cards/JOKER141__MiniMaxH3-GunFu-Gunfight-LORA.md`).
- **Motion Continuity Repair**: `bunny_crisp_motion`; 0,9 отдельно, 0,5–0,7 вместе с Combat (`sources/hf/cards/JOKER141__MiniMax-H3-General-Motion-Continuity-Repair.md`).
- **Jojocodex Wushu v7** (FL2VA/Ref2VA): `wushu_action` в начале промпта, 0,8–1,0 или 0,5 вместе со spatial-physics 0,3 (`sources/hf/cards/Jojocodex__wushu-action-v7-minimax-h3-fl2va-ref2va-lora.md`).
- **Jojocodex Spatial-Physics**: 0,3–0,5, автор называет её нестабильной (`sources/hf/cards/Jojocodex__minimax-h3-spatial-physics-lora.md`).
- **Jojocodex Camera-Motion**: `camera motion` в начале, 0,8–1,0 (`sources/hf/cards/Jojocodex__minimax-h3-Camera-Motion-lora.md`).
- Civitai:
  - Better Human Motion (0,4–0,8);
  - Motion-Speed slider;
  - POV (`mpov` / `fpov`);
  - Third Person View (0,6–0,85, база Ref INT8);
  - Cinematic V2 (0,85);
  - Cinematic Look (`DY`, 0,7).

  Источники: `sources/web/civitai__top-h3-models-by-downloads.md`, `sources/web/civitai__third-person-view-lora.md`.

**Звук**
- **SimpleTuner/minimaxh3-suno-reggae-rank128** (45): r128, 1000 шагов на 8 GPU, датасет из 123 элементов «suno-reggae-audio». Что это чисто звуковая LoRA, видно только по названию датасета (**не проверено**) (`sources/hf/cards/SimpleTuner__minimaxh3-suno-reggae-rank128.md`). Других специализированных звуковых LoRA в материалах нет.

**Персонажи**
- **masafy** (`masafy_character`, Ref2VA, чекпойнт 500, сила 0,8) (`sources/hf/cards/masafy__minimax-h3-masafy-lora.md`).
- RefMods — альтернатива LoRA (раздел 4.4).

## 6. Файнтюны, мерджи, эмбеддинги

| Модель | Что это | Форматы | Загрузки | Источник |
|---|---|---|---|---|
| WarmBloodAban/Minimax-h3_Singularity (V1/V2) | слияние ref + fl + b25-49, затем долгий файнтюн. Заявлены HDR-данные, дальние лица, экшн/VFX; в V2 исправлено условие по аудио-референсу; промпт до ~5000 символов (**не проверено**) | INT8 | 1 125 221 | `sources/hf/cards/WarmBloodAban__Minimax-h3_Singularity.md` |
| WAS/Minimax-h3_Singularity_v2_pruned | Ref2VA v2 | int8 21 / w6a8 16 / w4a8 11,8 GB; сборка r2: PSNR 26,1 против 24,6 | 105 818 | `sources/hf/cards/WAS__Minimax-h3_Singularity_v2_pruned.md` |
| Abiray, indhic-ai | копии Singularity v2 pruned и GGUF | | 20 654 / 31 130 / 6 536 | `sources/hf/meta.tsv` |
| OliviaRossi/…Singularity-x-10Eros-Max_PRISM | мердж Singularity V2 × 10Eros-Max turbo своим алгоритмом (rSVD-денойз дельты, геодезическая интерполяция, разрешение конфликтов знаков); вариант UPD5 DMAD_PDMD — 8 шагов, Euler/Beta57, 12/3 | | 579 945 | `sources/hf/cards/OliviaRossi__Minimax-h3_Singularity-x-10Eros-Max_PRISM.md` |
| OliviaRossi TriPrism-TurboFusion | Sulphur-3 + 10Eros + Singularity V2 + TurboFusion + Realism Slider | | 0 (Civitai 23 333) | `sources/hf/cards/OliviaRossi__Minimax-H3-Sulphur3-10Erosmax-SingularityV2-TriPrism-TurboFusion.md` |
| 10Eros-Max (TenStrip) | NSFW-граф с данными LTX 2.3 / Wan 2.2 / Krea 2 | | — | `sources/github/wildminder__awesome-minimax-H3.md` |
| DaSiWa (Civitai, GXS v5) | Ref2VA turbo-мердж; зеркало releaseworthy — int8, 4 шага (звук хуже) / 6 / 8 | int8 ~21 GB | Civitai 87 685; HF 20 102 | `sources/web/civitai__top-h3-models-by-downloads.md`, `sources/hf/cards/releaseworthy__dasiwa-minimax-h3-turbo-int8.md` |
| Deno2026/MiniMax-H3-DenoMix | Ref2VA pruned INT8 v3 на основе Singularity | int8 | 98 993 | `sources/hf/cards/Deno2026__MiniMax-H3-DenoMix.md` |
| Winnougan/Cobijada | гибрид b25-49 из uncensored-источников, без обучения | bf16 / int8 / w4a8 | 448 349 | `sources/hf/cards/Winnougan__Cobijada_Minimax-H3_Hybrid_Pruned_ComfyUI.md` |
| webmp3/Sakura-…-Uncensored-GGUF | **не DiT**: аблитерированный энкодер Qwen3-VL-32B | GGUF | 95 346 | `sources/hf/cards/webmp3__Sakura-MiniMax-H3-Uncensored-GGUF.md` |
| joeygambino/x-Z-Image; hoidhxd/x-Z-Image-hybrid | «пересадка» профиля внимания Z-Image: масштаб `q_norm` в поздних блоках (`zs05`); гибрид B00–24 FL2VA + B25–49 REF2VA | много квантов (раздел 2) | 22 587 / 15 172 | `sources/hf/cards/joeygambino__MiniMax-H3-x-Z-Image-native.md`, `sources/hf/cards/hoidhxd__MiniMax-H3-x-Z-Image-hybrid.md` |
| OpenVDN/vdn-minimax-h3 | гибрид линейного (DeltaNet) и softmax-внимания + 2 LoRA, DMD на 8 шагов; заявлено 14,4 с клипа за ~9 с на 8×B200 (**не проверено**); на нём FreeVideo (от 8 ГБ) | | 2 124 (364 лайка) | `sources/hf/cards/OpenVDN__vdn-minimax-h3.md`, `sources/github/OpenVDN__vdn-minimax-h3.md`, `sources/github/FlashML-org__FreeVideo.md` |
| Viggle-Animate | полный файнтюн Ref2VA 33,1B, DMD на 3 прохода | | — | `sources/github/wildminder__awesome-minimax-H3.md` |
| t8star/Minimax-H3-Image-Exp-Comfy | эксперимент с H3-Image «direct MLP100», 50 шагов | BF16 | 34 314 | `sources/hf/cards/t8star__Minimax-H3-Image-Exp-Comfy.md` |

Ещё в сводном списке: Remix FeiHou, Refusion Aki (Civitai 12 636), Singularity-Turbo от ms13d, Realism-Turbo-Remix и др. (`sources/github/wildminder__awesome-minimax-H3.md`).

**Эмбеддинги эффектов (10 штук, `Comfy-Org/MiniMax-H3/embeddings`, автор silveroxides)**
- Список:
  - `minimaxh3_art_is_explosion`, `_blooming_flowers`, `_bullet_time`, `_dark_magic`, `_fire_breath`;
  - `_four_seasons`, `_kiss_camera`, `_spiral_ascent`, `_storm_magic`, `_truman_show`.

  Вызов: `embedding:<имя>` (ComfyUI PR #15697) (`sources/hf/cards/Comfy-Org__MiniMax-H3.md`, `sources/web/docs.comfy.org__prompt-guide.md`).
- По разбору ai-muninn это не триггеры, а готовые тензоры [N, 5120], N от 50 до 142, всего 10,7 MB. По сути это заранее закодированный абзац выхода энкодера: он вставляется вместо N токенов (`sources/web/ai-muninn__minimax-h3-effect-embeddings-rtx5090.md`).
- Как применять (тот же источник):
  - ставить в середину фразы — в самом начале промпта эффект молча не срабатывает;
  - эффект не описывать словами;
  - оставлять сцене «место» для эффекта.
- Группы: камера (bullet_time, spiral_ascent, truman_show), события (fire_breath, dark_magic, storm_magic, art_is_explosion), течение времени (four_seasons, blooming_flowers), камера + действие (kiss_camera) (там же).

## 7. Обучение своих LoRA

**Инструменты**
- **VideoX-Fun** (официально от PAI). Датасет — `metadata.json` с `file_path`, `audio_path` (обязательно, ресемплинг в 32 кГц), `text` (по-английски), `type`, `width`, `height`; демо-набор `PAI/X-Fun-Videos-Audios-Demo`.
  - LoRA: обязателен FSDP FULL_SHARD (трансформер ~62 ГБ + Qwen3-VL ~62 ГБ, ZeRO-2 их не шардирует). Rank 64, alpha 32. Цели: `to_q,to_k,to_v,ff.0,ff.2,proj_in,audio_proj_in,context_embedder`. lr 1e-4, wd 3e-2, `video_sample_size` 960, 124 кадра (17n+5), batch 1, `--low_vram`, `gradient_checkpointing_save_on_cpu`, `train_mode=fl2va`, `t2v_ratio=0.25`. Вес на инференсе — 0,55.
  - Полный файнтюн — lr 1e-5.

  Источник: `sources/github/aigc-apps__VideoX-Fun__minimax-h3-fun-controlnet.md`.
- **ai-toolkit (ostris)**: архитектуры `minimax_h3` / `minimax_h3_image`, разделы `fl2va_pruned` / `ref2va_pruned`. Во время обучения подключается **`ostris/minimax_h3_training_adapter`** (v1/v2, «де-дистилляция» CFG-дистиллированной базы), на инференсе он не нужен (`sources/hf/cards/pablodawson__MiniMax-H3-360-Orbit-LoRA.md`, `sources/hf/cards/matlod__minimax-h3-turnaround.md`, `sources/hf/cards/ausboss__MiniMax-H3-Image-Outpaint-LoRA.md`).
- **musubi-tuner**: `--task ref2va`, `--prune_adaln`, адаптер через `--base_weights` (`sources/hf/cards/MATLOWAI__MiniMax-H3-ORB360-CardSpin.md`, `sources/hf/cards/Cseti__MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1.md`).
- **Тренер fal**: 4 эндпоинта — t2v, i2v, flf2v, ref2va (`sources/web/fal.ai__how-to-train-lora.md`).
- Ещё: SimpleTuner, DiffSynth-Studio (LoRA и полный файнтюн, есть NF4 для low-VRAM), diffusion-pipe, Fizgig (LoRA/LoKr), VeOmni, MiniMax-H3-FineTuning (звук в лоссе); RL — verl-omni, Flow-Factory, miles_diffusion, UniRL (`sources/github/MiniMax-AI__awesome-minimax-h3-integration.md`). Тренер Civitai и конкурс LoRA (только SFW, Style/Concept/Camera, до 23.10.2026) (`sources/web/civitai__article-35316.md`).

**Датасет и подписи (fal)** (`sources/web/fal.ai__how-to-train-lora.md`)
- Объём: от 10 файлов, лучше 50–200 клипов по 3–15 с, строго 24,000 fps.
- **Звук сохранять**: без него модель научится тишине.
- Подготовка: убрать затемнения в начале и конце, замедленные клипы перетаймить к нормальной скорости.
- Бакеты кратны 32: 1280×704, 1280×544, 960×704.
- Подписи: один абзац на клип. Триггер — либо `trigger_phrase`, либо прямо в подписях, не в обоих местах.
- Параметры по умолчанию: lr 2e-4, rank 32, 2000 шагов, 73 кадра (кадры % 17 == 5).
- Выводы автора:
  - для стиля rank 16 лучше 32 и 64;
  - ~1500 шагов на 53 клипа, 3000+ на 176;
  - сравнивать парами с одним seed при scale 0 и 1.

**Типичные параметры из карточек**
| LoRA | Инструмент | rank/alpha | lr | Шаги | Данные | Железо | Источник |
|---|---|---|---|---|---|---|---|
| 360-Orbit | ai-toolkit, fl2va_pruned int8 | 16, все линейные слои, кроме `adaln_proj` | 1e-4 AdamW8bit | 3000 | 28 клипов | — | `sources/hf/cards/pablodawson__MiniMax-H3-360-Orbit-LoRA.md` |
| Turnaround | ai-toolkit, ref2va_pruned | 16, только attention | 5e-5 | — | 90 клипов | — | `sources/hf/cards/matlod__minimax-h3-turnaround.md` |
| Person-Remover | ai-toolkit | 16/16 без `adaln_proj` | 5e-5 AdamW8bit | 2000 | — | при полной загрузке в GPU 24 ГБ не хватило | `sources/hf/cards/akatz-ai__MiniMax-H3-Person-Remover-LoRA.md` |
| ORB360-CardSpin | musubi v0.3.5 | 32/32 | 1e-4 | 1500 | 345 рендеров Blender; лосс только по видео | RTX PRO 6000 96 ГБ, ~26 ч (~62 с/шаг) | `sources/hf/cards/MATLOWAI__MiniMax-H3-ORB360-CardSpin.md` |
| CrossView-Warp | musubi (форк) | 32 | — | 6000 | — | RTX PRO 6000 | `sources/hf/cards/Cseti__MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1.md` |
| 360-equirect | тренер fal | PEFT r32, attention 50 блоков + 2 token-refiner; совместный лосс видео и звука | 2e-4 | 3500 | — | — | `sources/hf/cards/rehan-fal__minimax-h3-360-equirect-lora.md` |
| People-Vehicle-Removal | DiffSynth | 32 на `qkv_proj`/`out_proj` | 5e-5, batch 16 | 1000 | 6 635 клипов | — | `sources/hf/cards/jialidddh__MiniMax-H3-People-Vehicle-Removal-LoRA.md` |
| Image Outpaint | ai-toolkit `minimax_h3_image` (DiT INT8, TE NVFP4) | 16/16 | 1e-4 AdamW8bit | 2000 (выпущен ckpt 1500) | 412 пар, бакеты 2048 | 1× RTX 3090, ~5 ч | `sources/hf/cards/ausboss__MiniMax-H3-Image-Outpaint-LoRA.md` |
| Grimdark 3D | Fizgig | 8 | 2e-4 | 50 эпох | 150 картинок | — | `sources/web/civitai__grimdark-3d-style-lora.md` |

- Узкое место по памяти — системная RAM, а не VRAM. У masafy пик VRAM при обучении — всего 3 959 MiB, но набор весов 41,2 ГБ: с 50 ГБ RAM процесс убивал OOM, с 188 ГБ дошёл до конца (`sources/hf/cards/masafy__minimax-h3-masafy-lora.md`).
- Цели LoRA: большинство авторов **исключают `adaln_proj`** и обучают на pruned-базе. VideoX-Fun добавляет `audio_proj_in` и `context_embedder` (источники выше).
- Звук: VideoX-Fun и fal учат видео и звук совместно, ORB360 — только видео (источники выше).
- Юридически: пункт V.3 лицензии запрещает использовать выходы H3 для улучшения других моделей (раздел 1.5). На обучение LoRA для самой H3 этот пункт, по формулировке, не распространяется.
## 8. Переписчики промптов и альтернативные энкодеры (кратко)

Подробно — в разделе о промптах. Официальный H3-Context-IR закрыт (раздел 1.2), поэтому вокруг него выросли открытые замены.

| Модель | База | Режимы | Загрузки | Источник |
|---|---|---|---|---|
| `lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA` | LoRA на Qwen3.6-27B | T2VA | 198 (GGUF pytraveler 2 181) | `sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-LoRA.md` |
| `lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B` | LoRA на Qwen3-VL-8B (2,79 GB fp32; GGUF Q8_0 0,69 GB) | T2VA/I2VA/FL2VA (4 задачи), без Ref2VA | 408 | `sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-LoRA-8B.md`, `sources/github/wildminder__awesome-minimax-H3.md` |
| `lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni` | LoRA на Qwen2.5-Omni-7B | текст + картинка/видео/аудио, en/zh | 273 | `sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-LoRA-Omni.md` |
| `lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-{FL2VA,Ref2VA}` | полный SFT Qwen3.5-9B, vLLM | FL2VA / Ref2VA | 4 234 / 8 509 | `sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA.md`, `sources/hf/meta.tsv` |
| `lightx2v/…-Qwen3.8-27B-{FL2VA,Ref2VA}` | 27B, ~51 ГиБ, без LICENSE | FL2VA / Ref2VA | — | `sources/github/wildminder__awesome-minimax-H3.md` |
| `Wuli-art/Gemma4-for-MiniMax-H3-Prompt-Extend` | Gemma4-12B-IT; одни веса на FL2VA и Ref2VA; BF16 22,3 GB / FP8 12,1 GB | FL2VA, Ref2VA | 6 827 | `sources/hf/cards/Wuli-art__Gemma4-for-MiniMax-H3-Prompt-Extend.md`, `sources/github/wildminder__awesome-minimax-H3.md` |
| `MiniMaxAI/MiniMax-H3-Image-PE` | Qwen3.8-27B FP4, только для H3-Image | T2I | 6 044 | `sources/hf/cards/MiniMaxAI__MiniMax-H3-Image-PE.md` |
| «heretic»/abliterated варианты переписчиков 27B (GGUF у mradermacher, MLX oQ4e) | — | FL2VA / Ref2VA | 5 850 / 2 180 | `sources/hf/meta.tsv`, `sources/github/wildminder__awesome-minimax-H3.md` |

Узлы и инструменты ComfyUI для промптов: `sources/github/1038lab__ComfyUI-MiniMax-H3-Promptor.md`, `sources/github/pytraveler__MiniMax-H3-Prompt-Rewriter-ComfyUI.md`, `sources/github/T8mars__comfyui-minimax-h3-prompt-enhancer-T8.md`, `sources/github/ethanfel__ComfyUI-MiniMax-H3-Guide.md` (подгружает «хвост» энкодера, слои 50–63, как генератор промптов); Omni-Rewriter и OpenH3-IR (`sources/github/wildminder__awesome-minimax-H3.md`). Официальные skills для промптов лежат в `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`.

Вне промптов, **замены энкодера** Qwen3-VL-32B на модель поменьше (экспериментальные, нужен свой узел):
- `Alissonerdx/Qwen3.5-9B-Minimax-H3` — проекция 4096→5120 и LoRA rank 16 на Qwen3.5-9B, дистиллированные с учителя Qwen3-VL-32B; только Ref2VA. Метрики автора: косинус текста 0,95, картинок 0,89, PSNR видео 14,8 дБ против 12,9 дБ у одной проекции (**не проверено**) (`sources/hf/cards/Alissonerdx__Qwen3.5-9B-Minimax-H3.md`).
- `NicoLab28/ClipProj-MiniMax-H3` — линейные/MLP-проекции с Qwen3-VL 4B/8B; энкодер ~15,7 → 4,5 ГБ VRAM (**не проверено**). `Alissonerdx/Molmo2-8B-Minimax-H3` — пилотный проектор с Molmo2-8B (`sources/github/wildminder__awesome-minimax-H3.md`).
- Uncensored-энкодеры «Ultra-Heretic» (ethanfel; NVFP4-пересборки Momoking, sakamakismile и др.) заметны по лайкам, но не по загрузкам (`sources/hf/meta.tsv`, `sources/github/wildminder__awesome-minimax-H3.md`).
## 9. Что стоит попробовать у нас (Rust-порт, одна карта 32 ГБ)

Только идеи по материалам выше, ничего не скачивалось. Порядок — от дешёвых к дорогим.

1. **Свернуть AdaLN в таблицу ранга 8 вместо дискового кэша на 52 ГБ.** Сейчас мы кэшируем выходы AdaLN по набору шагов: 52 ГБ bf16, каждый новый набор шагов означает новый прогон (`sources/local/synaptix-perf.md`). Pruned/curve-форма даёт 1025-точечную таблицу `adaln_t_table` и проекции с входом 8: 13B → ~44M параметров, ошибка ниже шага bf16. Таблица годится для любого расписания σ (`sources/hf/cards/multimodalart__MiniMax-H3-Pruned.md`, `sources/hf/cards/coolthor__MiniMax-H3-pruned-NVFP4.md`). Свёртку можно сделать у себя из имеющихся весов. Минус 13B параметров — это ~20 ГБ bf16 / ~10 ГБ FP8 на диске и в RAM. Учесть, что LoRA с адаптерами AdaLN (lightx2v v1.2 8step: 51 пара r16) на свёрнутой базе требуют пересчёта (`sources/github/ModelTC__Minimax-H3-Turbo__README-setup.md`, `sources/github/Larryvrh__ComfyUI-MiniMax-H3-Turbo__README.md`).
2. **Обрезать энкодер до слоя 50.** DiT читает `hidden_states[50]`, слои 51–63 и `lm_head` не нужны: 66,7 → 52,5 ГБ в bf16 (`sources/hf/cards/multimodalart__MiniMax-H3-Pruned.md`, `sources/hf/cards/nif0__Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF.md`). Проверить, не грузим ли мы их сейчас. Энкодер у нас стримит слои с диска, и служба в ~1,3 раза медленнее CLI (`sources/local/synaptix-perf.md`). Плюс W4A4/NVFP4 для энкодера: по ai-muninn без видимой потери, 24 → 13,2 ГиБ (`sources/web/ai-muninn__minimax-h3-on-modded-2080ti-22gb.md`).
3. **ConvRot (Адамар 256) перед INT8 / NVFP4 в нашем `QuantLinear`.** Для int8 это почти бесплатный прирост косинуса: 0,982 → 0,989 (`sources/hf/cards/multimodalart__MiniMax-H3-Pruned.md`). У нас уже есть «MXFP8 rot» (`sources/local/synaptix-git-log-h3.md`). Сравнить наш NVFP4-DiT с вариантами rockerBOO: блоки 0, 1, 47–49 оставлены в BF16, qkv в INT8 ConvRot. И помнить, что W4A4 для DiT рвёт кадр (`sources/hf/cards/rockerBOO__minimax-h3-nvfp4-convrot.md`, `sources/web/ai-muninn__minimax-h3-on-modded-2080ti-22gb.md`).
4. **Проверить загрузчик LoRA на alpha из метаданных и shift под версию.** Файлы LightX2V с α8 и α128 отличаются по силе в 16 раз. 768p-версии требуют shift 6/3, остальные 12/3. Если шаги определяются по имени («Nstep»), стоит так же определять и shift (`sources/hf/cards/lightx2v__Minimax-h3-Turbo.md`, `sources/local/synaptix-port-code.md`).
5. **Попробовать новые ускорители на нашей базе без LoRA-оверхеда:**
   - PDD (alibaba-pai Acc, 8 шагов): 32 выходные головы, не LoRA на блоки. Шаг почти не дорожает, но несовместим с другими адаптерами.
   - PTD-RefAsymOPD для Ref2VA: 6–8 NFE, опорные картинки 768.
   - FastH3 Omni 8-step (Ref2VA, есть FP8 для одной карты).
   - DMAD (звук лучше, shift 12/2).

   Сравнивать прежде всего по звуку: у 4-шаговых тихие регистры и акценты деградируют (`sources/hf/cards/alibaba-pai__MiniMax-H3-Acc-LoRAs.md`, `sources/hf/cards/taobao-content-ai__MiniMax-H3-PTD-RefAsymOPD-LoRAs.md`, `sources/github/vllm-project__vllm-omni__recipes-FastH3-Regenerate2K-TaoMate.md`, `sources/web/civitai__turbo-loras-collection.md`, `sources/github/wildminder__awesome-minimax-H3.md`). Для единой цветопередачи на фильм лучше вшивать одну LoRA в веса (как MATLOWAI fused): это экономит VRAM на живой LoRA (`sources/hf/cards/MATLOWAI__minimax-h3-fused-turbo-int8-convrot.md`).
6. **Гибрид b25-49 (adaln из ref2va поверх fl2va) или ref-delta r1024** — держать один DiT вместо двух на диске и в RAM: FL2VA-кадры и Ref2VA-референсы одной моделью (`sources/hf/cards/smhfacct__Minimax-H3-fl2va-ref2va-hybrid-models.md`, `sources/hf/cards/xmarre__MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI.md`).
7. **Латентный апскейлер LBH (2D v1, 182M; 3D v1, 345M; Apache-2.0)** как дешёвая замена закрытому Regenerate-2K. Это простая 3D/2D-свёрточная сеть на 24-канальном латенте, её легко портировать. Затем короткий второй проход H3 (highres-fix у нас уже есть) (`sources/hf/cards/LBH-123-AI__Minimax_h3_latent_Upscaler.md`, `sources/local/synaptix-git-log-h3.md`).
8. **Облегчённые VAE:** LightVAE (26 блоков ViT вместо 36), int8 video VAE (2,95 ГБ), TAE на 9 МБ для превью. Аудио-VAE держать в fp32 (`sources/github/wildminder__awesome-minimax-H3.md`). У нас декод 1344×768×90 занимает 25,5 с, а на 720p × 10 с — 40,8 с (`sources/local/synaptix-perf.md`).
9. **Эмбеддинги эффектов** (10 файлов по [N, 5120], 10,7 МБ): в нашем энкодере это вставка готовых строк вместо токенов. Дёшево и даёт эффекты, которые текстом не получить. Ставить в середину фразы (`sources/web/ai-muninn__minimax-h3-effect-embeddings-rtx5090.md`).
10. **Fun-ControlNet-Union 2.0** (10 блоков управления, 12,6 ГБ fp32 → ~3,2 ГБ в FP8): поза, глубина и инпейнт для story-video. Брать конфиг 2.0, иначе ошибка будет молчаливой (`sources/hf/cards/alibaba-pai__MiniMax-H3-Fun-Controlnet-Union-2.0.md`).
11. **Локальный переписчик промптов** вместо закрытого Context-IR: lightx2v Qwen3.5-9B FL2VA/Ref2VA (полный SFT, vLLM) или Wuli Gemma4-12B (один набор весов на оба режима, FP8 12 ГБ). Запускать до загрузки DiT (`sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA.md`, `sources/hf/cards/Wuli-art__Gemma4-for-MiniMax-H3-Prompt-Extend.md`).
12. **Разреженное внимание.** Официальное пока не вышло. Из сторонних есть обученные: Veda (Ref2VA; внимание в 2–2,6 раза быстрее, по замеру авторов), SLA lightx2v, VSA FastH3. У нас уже есть разреженное FP8-внимание Sol. Стоит сравнить, что даёт обученный предиктор (`sources/hf/cards/Veda-Sparse__Minimax-H3-R2VA-Veda-Preview.md`, `sources/hf/cards/lightx2v__Minimax-h3-Turbo-SLA.md`, `sources/local/synaptix-git-log-h3.md`).
