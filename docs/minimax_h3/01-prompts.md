# MiniMax-H3: промпты

Черновик собран 2026-11-28 только по локальным материалам `docs/minimax_h3/sources/`. После каждого утверждения в скобках —
источник (путь относительно `docs/minimax_h3/`). «Не проверено» — цифры поставщиков и замеры одного автора на одном зерне,
которые мы сами не повторяли.

Обозначения режимов (официальные): **T2VA** — текст→видео+звук; **I2VA** — первый кадр; **FL2VA** — первый и последний
кадр; **L2VA** — только последний кадр; **Ref2VA** — полный режим референсов (картинки, видео, звук)
(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Первые четыре режима обслуживает чекпойнт
H3-Base-FL2VA (0, 1 или 2 картинки), Ref2VA — отдельный чекпойнт H3-Base-Ref2VA
(`sources/github/MiniMax-AI__MiniMax-H3__README.md`; `sources/web/docs.comfy.org__h3-native-workflows.md`).

---

## 1. Общие принципы

### 1.1 Главное: открытые веса ждут уже «переписанный» промпт

- Полная система H3 = **H3-Context-IR** (размещённый у MiniMax препроцессор: разбирает свободный запрос и медиа и
  сериализует их в структурированное представление) + **H3-Base** (768p) + **H3-Regenerate-2K**. Context-IR в открытый
  релиз не входит; MiniMax «настоятельно рекомендует» встроить его или построить свой по Prompting Guidance
  (`sources/github/MiniMax-AI__MiniMax-H3__README.md`; `sources/hf/cards/MiniMaxAI__MiniMax-H3.md`).
- Context-IR принимает короткий текст («Pull focus to the people in the background and add more steam to the ramen bowl.»)
  плюс медиа с ролями `first_frame`/`reference_video`/… и возвращает развёрнутый промпт в официальном формате
  (`sources/github/MiniMax-AI__MiniMax-H3__scripts-and-configs.md`, скрипты `full-2k-*-h3-context-ir.sh`).
- Отсюда следствие: свободные промпты из галерей облачных API (fal, Atlas Cloud) прошли через расширение и на локальных
  весах так не работают. Прямая проверка: на локальном Ref2VA одна свободная фраза к раскадровке дала 3/16 совпавших
  панелей, тот же замысел в официальном формате — 13/16 (не проверено, n=1)
  (`sources/web/ai-muninn__storyboard-grid-h3-ref2va-rtx5090.md`). В облачном API v2 расширение для моделей Max
  включено по умолчанию: `extra.prompt_expansion_mode` = `disabled` / `balanced` (умолчание) / `quality`
  (`sources/web/platform.minimax.io__api-video-generation-v2.md`).

### 1.2 Язык

- Все служебные поля и описание — **по-английски**; на исходном языке остаются только реплики и тексты песен внутри `<d>` и
  видимый в кадре текст (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`;
  `sources/web/docs.comfy.org__prompt-guide.md`).
- Устойчиво поддержаны реплики на 11 языках, в том числе русском и китайском; остальные — «в разной степени»
  (`sources/github/MiniMax-AI__MiniMax-H3__README.md`).
- Наш опыт: русское описание сцены H3 зачитывает вслух; английское описание с репликами просто в кавычках «…» — герои говорят
  по-английски; работает английское описание + `<d>[Russian] …</d>` (`sources/local/syn-video-editor-skill-minimax-h3.md`;
  `sources/local/synaptix-probe-findings.md`).
- Китайские реплики — только упрощёнными иероглифами: на одном зерне точность распознавания 0,667 (традиционные) против 1,000
  (упрощённые) (не проверено, n=1) (`sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`).

### 1.3 Длина промпта

- Облачный API: до 7000 символов (`sources/web/platform.minimax.io__api-video-generation-v2.md`;
  `sources/web/fal.ai__prompting-guide-44-examples.md`); для open-весов тот же вопрос в issue #61 остался без ответа
  (`sources/github/MiniMax-AI__MiniMax-H3__issues.md`).
- Официальный ориентир для Ref2VA: `detailed_description` для генерации — обычно 350–500 английских слов; для насыщенных
  диалогом сцен важнее уложить всю речь, чем добрать слова; «один план» не повод писать короче
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- Образцы Context-IR длинные: T2VA на 10 с — 2915 выходных токенов, I2VA на 8 с — ~10 тыс.
  (`sources/github/MiniMax-AI__MiniMax-H3__README.md`).
- В свободных промптах облака длина обратно пропорциональна вкладу референсов: медиана 45 официальных примеров ≈130
  иероглифов, самые длинные — 657 и 858 (не проверено) (`sources/web/atlascloud__prompt-guide-45-official.md`).

### 1.4 Что работает

- Писать **таймлайн, а не описание кадра**: «старт → изменение → камера → звук → финал» (`sources/web/hackernoon__prompt-guide.md`;
  `sources/web/atlascloud__prompt-guide-45-official.md`).
- Каждая деталь должна быть видимой или слышимой: стиль, начальная композиция, внешность и положение героев, обстановка,
  действия и реакции, смена планов, речь, звук (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- Конкретика вместо абстракций вроде «cinematic», «beautiful» (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- Хронометраж описания должен совпадать с длиной ролика (4–15 с) (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- Длина ролика на локальных весах — кадры при 24 fps на сетке **17k+5** (124 ≈ 5 с, 243 = 10,125 с, 362 ≈ 15,08 с — потолок)
  (`sources/web/docs.comfy.org__prompt-guide.md`; `sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`). Длина вне сетки
  округляется вверх; если звуковой референс короче, в хвосте H3 придумывает речь
  (`sources/local/synaptix-probe-findings.md`).

### 1.5 Что не работает / чего избегать

- **Негатив при cfg 1.** Выпущенные веса CFG-дистиллированы (`sources/github/MiniMax-AI__MiniMax-H3__README.md`); ComfyUI
  гоняет `BasicGuider` с cfg 1 без негативной ветки, поэтому поля «negative» нет, а списки запретов в позитиве «добавляют
  нежелательные понятия» — надо писать, что должно быть (`sources/web/docs.comfy.org__prompt-guide.md`). vLLM-Omni в
  пайплайне прямо пишет «CFG-distilled and has no negative branch»; в нашем порте CFG 5 + негатив — только опция
  (`sources/local/synaptix-probe-findings.md`). Спорность — в разделе 7.
- Описание сюжета пересказом, неразрешённые метки референсов, таймкоды, не совпадающие с длиной
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- Перегруз: два говорящих + много действий + музыка + шумы + движения камеры + склейки в коротком ролике снижают надёжность речи
  (`sources/web/seedance.tv__gibberish-fix.md`).
- Многоточия в репликах ухудшают разборчивость (`sources/web/docs.comfy.org__prompt-guide.md`).

---

## 2. Официальная структура по режимам

### 2.1 Общий каркас

Базовые режимы (T2VA/I2VA/FL2VA/L2VA) — три поля в строгом порядке; в режимах с кадрами перед ними стоит строка
выравнивания, затем **одна пустая строка** (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`):

```text
<строка выравнивания — только I2VA / FL2VA / L2VA>

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

| Поле | Что в нём | Объём |
|---|---|---|
| `integrated_multimodal_description` | визуал, действия, планы, говорящие, реплики, пение и **внутрикадровый** звук по времени | основной объём |
| `overall_soundscape` | фон, звуки физических действий, неречевые звуки людей (дыхание, смех) за весь ролик; без реплик и музыки | 1–4 предложения; `N/A` только при явной полной тишине |
| `non_diegetic_music` | закадровая музыка, которую слышит только зритель: инструменты, темп, ритм, динамика; без «настроенческих» слов | 1–3 предложения; `N/A`, если музыки нет |

(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`; `sources/web/comfyui-wiki__prompt-writing-five-modes.md`)

Ref2VA — шесть разделов, порядок обязателен: `subject_definitions` → `summary` → `retention_analysis` →
`detailed_description` → `overall_soundscape` → `non_diegetic_music`
(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).

### 2.2 Строки выравнивания ключевых кадров (дословно)

| Режим | Первая строка промпта |
|---|---|
| T2VA | нет; промпт начинается прямо с полей |
| I2VA | `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.` |
| FL2VA | `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.` |
| L2VA | `How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.` |

`N` — номер фактического **последнего** плана, `S.SS` — эффективная длительность ролика ровно с двумя знаками после запятой
(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Обратите внимание: в официальном шаблоне FL2VA метки
написаны **без** угловых и квадратных скобок (`Picture 1 (from Shot 1)`), а в I2VA и L2VA — со скобками; так в оригинале
(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`; `sources/web/comfyui-wiki__prompt-writing-five-modes.md`).
«Эффективная длительность» на локальных весах — длина по сетке 17k+5 (например, 243 кадра → `10.13`), вывод наш из
(`sources/web/docs.comfy.org__prompt-guide.md`); официальных пояснений об округлении нет.

### 2.3 T2VA

Нет строки выравнивания; стиль и начальная композиция — в начале `[Shot 1]`; сцену, героев, действие и звук можно
достраивать в рамках замысла (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).

Шаблон:

```text
integrated_multimodal_description: [Shot 1] <стиль: Live-action / 2D-animated / 3D CG / claymation …>, <крупность> frames <кто, где>. The camera <тип движения> [with small|large amplitude] [at slow|fast speed] as <герой: возраст, пол, голос> (S1) <действие> and says: <d>[<Язык>] <реплика></d> [Shot 2] At 00:SS.mmm, the camera cuts to <новая информация: крупность/объект/время>.

overall_soundscape: <фон>. <звуки действий по порядку>.

non_diegetic_music: <инструменты, темп, динамика> | N/A
```

Официальный пример (дословно, `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`):

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and says: <d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of steam rising from the sliced bread while the baker's final words carry over from the previous shot.

overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the bakery. The doorbell rings once, followed by light footsteps and the crisp sound of bread being sliced.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and a gentle fade at the end.
```

Более длинный образец — выход Context-IR для T2VA на 10 с (корабельный мостик, два плана, склейка `At 00:04.500`) — в
`sources/github/MiniMax-AI__MiniMax-H3__README.md` (case-T2VA).

### 2.4 I2VA (первый кадр)

`<Picture 1>` — реальный первый кадр (0.00 с), принадлежит `[Shot 1]`. Сначала закрепить стиль, героев, композицию и
опоры сцены из картинки, затем следующее действие; внешность, одежда, цвета, ключевые предметы и расстановка неизменны.
Рекомендуемый ход: **якорь первого кадра → начало действия → развитие → результат/реакция**
(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Не только заимствовать стиль картинки, а начинать с неё
(`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`).

Шаблон:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] <стиль>, the <герой> shown in <Picture 1> remains <где>, preserving <внешность, одежда, положение, обстановка>. The camera <движение> as <первое действие>. <развитие>. <результат/реакция>.

overall_soundscape: ...

non_diegetic_music: ... | N/A
```

Официальный пример (дословно, `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`):

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the young woman shown in <Picture 1> remains beside the rain-covered train window, preserving her appearance, clothing, seat position, and the carriage layout. The camera trucks right with small amplitude at slow speed as she lifts her gaze from the folded letter toward the passing city lights. Her reflection moves across the glass while the quiet, breathy young woman (S1) says: <d>[English] I get off at the next station.</d> She folds the letter along its existing crease.

overall_soundscape: The train wheels produce a steady metallic rhythm beneath a low ventilation hum. Rain ticks against the window while paper rustles softly in her hands.

non_diegetic_music: Sustained cello notes at a slow tempo with widely spaced piano tones, gradually decreasing in volume.
```

Практика: первый кадр в ComfyUI растягивается до холста без сохранения пропорций, а последний кадр обрезается по центру —
пропорции картинки надо подгонять заранее (`sources/web/docs.comfy.org__h3-native-workflows.md`). В облачном I2V форма
кадра побеждает заданный `ratio` (не проверено, n=1) (`sources/web/atlascloud__prompt-guide-45-official.md`).

### 2.5 FL2VA (первый + последний кадр)

Picture 1 — начало, Picture 2 — конец. Описывать **путь** между ними (движение, смена поз, работа с предметами, эволюция
композиции, смена света), а не два статичных описания. Режим «в целом предпочитает один план»; несколько планов — только если
заданы явно; последний кадр достигается финальным `[Shot N]` в конце ролика. Ход: **состояние первого кадра → наблюдаемые
промежуточные изменения → сужающаяся разница → состояние последнего кадра**
(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Обе картинки — схожих пропорций, иначе переход уродлив
(`sources/web/atlascloud__prompt-guide-45-official.md`).

Шаблон:

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the <длительность>.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] <стиль>, <герой> begins in the position and framing established by Picture 1, <исходное состояние>. The camera <движение> as <действие 1>, <действие 2>, <промежуточное изменение>, and settles into the pose, spacing, and composition established by Picture 2 at the end of the shot.

overall_soundscape: ...

non_diegetic_music: ... | N/A
```

Официальный пример, один план 8 с (дословно, `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`):

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a rain-soaked cyclist begins in the position and framing established by Picture 1, holding a closed black umbrella beside a silver bicycle. The camera pulls out with small amplitude at slow speed as she releases the bicycle handle, raises the umbrella above her shoulder, and presses the runner upward until the canopy opens. Water rolls from the expanding fabric while she steps beneath it, rotates the handle into the final angle, and settles into the pose, spacing, and composition established by Picture 2 at the end of the shot.

overall_soundscape: Rain falls steadily on the pavement, followed by the metallic click of the umbrella runner and the soft snap of the canopy opening. Water drips from the bicycle frame as distant traffic passes.

non_diegetic_music: N/A
```

### 2.6 L2VA (только последний кадр)

`<Picture 1>` — **последний** кадр и принадлежит последнему `[Shot N]`, а не `[Shot 1]`. Вывести правдоподобное начальное
состояние, затем постепенно привести героев, предметы, камеру и сцену к картинке. Ход: **правдоподобное предшествующее
состояние → явное действие и путь → постепенная сходимость в финальном плане → посадка на последний кадр**
(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Типичный сбой — модель приходит к финалу слишком рано;
лечится постепенным подходом и резервированием последнего момента под сходимость (`sources/web/hackernoon__prompt-guide.md`).

Шаблон:

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the <длительность>.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] <стиль>, <крупность> begins with <более раннее совместимое состояние>, while the same <рука/герой/предмет> visible in <Picture 1> <подходит>. The camera <движение> as <событие>. Toward the end, <движение затухает> and settles into the exact <расстановка, ракурс, свет, композиция> established by <Picture 1>.

overall_soundscape: ...

non_diegetic_music: ... | N/A
```

Официальный пример, один план 6 с (дословно, `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`):

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a close shot begins with an intact drinking glass near the edge of a dark wooden table, while the same hand and sleeve visible in <Picture 1> approach from the right. The camera pushes in with small amplitude at slow speed as the fingertips strike the rim. The glass tips, falls, and hits the floor with a sharp impact; cracks spread through it as fragments slide outward. Toward the end, the moving pieces lose momentum and settle into the exact broken arrangement, hand position, camera angle, lighting, and final composition established by <Picture 1>.

overall_soundscape: Fingertips tap the glass before it scrapes across the tabletop, falls, and breaks with a sharp crash. Small fragments scatter and gradually stop sliding across the floor.

non_diegetic_music: A low electronic pulse at a slow tempo, ending immediately after the glass breaks.
```

### 2.7 Ref2VA (полный режим референсов)

#### Шесть разделов

| Раздел | Назначение | Ключевые правила |
|---|---|---|
| `subject_definitions` | определяет каждый отслеживаемый референс и его метку | по строке на элемент: что обозначает метка, роль, главные приметы; источник — если нужно снять неоднозначность |
| `summary` | одна короткая фраза о ролике и связях референсов | начинается с префикса типа задачи в `[]`; новых меток не вводит |
| `retention_analysis` | как сохраняется / переносится каждый референс | по строке на метку, фиксированные маркеры; `(Sx)` здесь не пишут |
| `detailed_description` | основное тело: планы в порядке воспроизведения | стиль — 1–2 предложения **до** `[Shot 1]`; метки — при первом появлении и там, где действуют |
| `overall_soundscape` | фон и физические звуки | при звуковом референсе — его связь здесь, если это слой фона/шумов |
| `non_diegetic_music` | закадровая музыка | при звуковом референсе — его связь здесь, если это партитура |

(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`)

#### Метки

| Метка | Смысл |
|---|---|
| `<Subject N>` | видимое содержание, абстрагированное из референсов: человек, животное, предмет, сцена, одежда, интерфейс, эффект, стиль, действие, поза. Один субъект может собираться из нескольких файлов, один файл — давать несколько субъектов |
| `<Picture N>` | картинка как конкретный кадр (первый, ключевой, последний, отредактированный) или опора планирования (раскадровка). Если картинка лишь определяет героя/сцену/стиль — отдельной строки не заводят, её цитируют внутри `<Subject N>` |
| `<Video N>` | только отношения «весь ролик»: исходник для редактирования, старт для продолжения, образец движения камеры / склеек / ритма. Люди и предметы из видео всё равно идут как `<Subject N>` |
| `<Audio N>` | отдельный звуковой файл или включённая дорожка референсного видео. Обычное видео со звуком не порождает `<Audio N>` само по себе |

`<Video N>` и `<Audio N>` нумеруются независимо; одна и та же видеозапись может быть `<Video 1>` и `<Audio 2>`. Метка
сохраняет смысл во всех шести разделах (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).

Нумерация в ComfyUI: `ref_image_0` → `<Picture 1>`; порядок в узле — картинки, затем видео, затем звук; звуковые дорожки
видео получают `<Audio N>` первыми (в порядке видео), затем отдельные аудиофайлы; дорожка видео используется только если
подключён `ref_video_audio_N` (`sources/web/docs.comfy.org__h3-native-workflows.md`). В нашей службе метки нумеруются по
типу и порядку в `refs`, и движок по-сценно отбирает и перенумеровывает упомянутые `<Picture k>`/`<Audio k>`
(`sources/local/syn-video-editor-skill-minimax-h3.md`).

#### Типы задач в `summary`

| Тип | Когда |
|---|---|
| `keyframe completion` | картинка — конкретный кадр (первый, ключевой, последний, отредактированный) |
| `reference generation` | картинка/видео/звук направляют героя, сцену, стиль, действие, камеру, раскадровку, не будучи кадром или исходником |
| `video editing` | исходное видео прямо изменяется (редактирование картинки или генерация между стоп-кадрами сюда не относится) |
| `video continuation` | новое содержание продолжает исходное видео |
| `audio reuse` | тот же звуковой сигнал используется целиком или частично |
| `audio reference` | сигнал не копируется; берутся стиль музыки, тембр, текст, фактура, ритм |

Несколько типов — через ` + ` без повторов: `[video continuation + keyframe completion]`, `[video editing + audio reuse]`.
Наличие видео/звука само по себе тип не создаёт: видео только ради движения камеры — это `reference generation`. При
редактировании `summary` после префикса начинается фразой `The target video is an edited version of <Video 1>.`
(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).

#### Маркеры `retention_analysis`

| Для картинок/видео/субъектов | Смысл | Для звука | Смысл |
|---|---|---|---|
| `fully_preserved` | роль референса сохранена полностью | `fully_copy` | весь исходный звук = вся итоговая дорожка |
| `partially_preserved` | используется, но часть примет изменена / сохранена частично | `partially_copy` | копируется часть времени или слоёв, либо после копирования что-то добавлено/убрано/заменено |
| `attribute_transfer` | признаки переносятся на **другой** опознаваемый субъект | `reference` | сигнал не копируется: только тембр, ритм, стиль, текст, фактура |
| `weak_reference` | только общее сходство стиля, категории, композиции, атмосферы | `weak_reference` | только общее сходство категории или атмосферы |

Формы строк: `<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...`; `<Picture 2> ([Shot 1] first frame):
fully_preserved - ...`; `<Video 1> (cut and pacing structure): weak_reference - ...`; `<Audio 1>: fully_copy - ...`.
Маркер выбирается только в пределах роли, заданной в `subject_definitions`; новые действия и фон не считаются потерей
верности (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).

#### Говорящие в Ref2VA

- Если говорит референсный субъект: `<Subject 2> (S1) turns toward the woman and says, <d>[English] …</d>`; вне кадра —
  та же форма плюс `off-screen`; если голос не соответствует субъекту — устойчивое описание голоса + `(Sx)`
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- `(Sx)` раздаются один раз в порядке реальных речевых событий ролика; в `subject_definitions` голосовой референс
  переиспользует номер (`<Audio 1> is the voice-timbre reference for <Subject 1> (S1).`), но не назначает новый
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- Слова внутри напрямую переиспользуемой фонограммы (BGM, песня) — источник `<Audio N>`, отдельный `(Sx)` не придумывают
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- При переиспользовании речи/лирики из звука — точные слова на исходном языке, `[unclear]` для неразборчивого, пунктуация
  нормализуется до `, . ? !`; если берётся только тембр/ритм — исходный текст в ролик не переносится
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).

#### Лимиты референсов

| Что | Лимит |
|---|---|
| Картинки | ≤ 9 |
| Видео | ≤ 3, каждое 2–15 с, суммарно ≤ 15 с |
| Звук | ≤ 3, каждый 2–15 с, суммарно ≤ 15 с |
| Всего файлов | ≤ 12 |
| Звук без картинки/видео | нельзя |

(`sources/github/MiniMax-AI__MiniMax-H3__README.md`; `sources/github/ethanfel__ComfyUI-MiniMax-H3-Guide.md`;
`sources/web/atlascloud__prompt-guide-45-official.md`; `sources/local/syn-video-editor-skill-minimax-h3.md`). Облачный узел
Context-IR дополнительно требует картинки ≥ 256×256 с пропорциями 0,4–2,5 и видео 23,9–60,5 fps
(`sources/github/Comfy-Org__docs__minimax-h3-node-reference.md`); локальный узел читает референсное видео как 24 fps без
метаданных — другие частоты надо конвертировать заранее (`sources/web/docs.comfy.org__h3-native-workflows.md`).

#### Шаблон Ref2VA

```text
subject_definitions:
<Subject 1> is the <герой A: возраст, волосы, одежда, приметы> in <Picture 1>.
<Subject 2> is the <локация: ключевые предметы> in <Picture 2>.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] The target video shows <Subject 1> <действие> in <Subject 2>. <течение планов одной-двумя фразами; роль <Audio 1>>.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - <что сохраняется>.
<Subject 2> (appears in [Shot 1], [Shot 2]): partially_preserved - <что сохраняется, что может меняться>.
<Audio 1>: reference - <чем направляет, без копирования сигнала>.

detailed_description:
The target video is in <стиль, свет, палитра>.
[Shot 1] <крупность> establishes <Subject 2>, <приметы>. <Subject 1> (S1), <полная внешность>, <положение в кадре, действие>. The camera <движение>. <Subject 1> (S1), using the <описание голоса> voice timbre referenced from <Audio 1>, says, <d>[<Язык>] <реплика></d> <после реплики: губы сомкнуты, действие>.
[Shot 2] At 00:SS.mmm, the shot cuts to <крупность> of <Subject 1> (S1), the <краткие приметы> from Shot 1, ...

overall_soundscape:
<фон и шумы>

non_diegetic_music:
<партитура> | N/A
```

#### Полный официальный пример Ref2VA (дословно, `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`)

```text
subject_definitions:
<Subject 1> is the coffee-shop environment in <Picture 1>, featuring an exposed brick wall, an orange tufted sofa with patterned pillows, a neon sign, and a wooden coffee table.
<Subject 2> is the fluffy white Samoyed in <Picture 2>, <Picture 3>, and <Picture 4>, with thick white fur, pointed ears, a dark nose, and a curved tail.
<Subject 3> is the young blonde woman in <Video 1>, with long blonde hair and a light-pink button-down shirt with rolled-up sleeves.
<Subject 4> is the young man in <Video 2>, with short wavy brown hair and a dark-grey hoodie with drawstrings.
<Audio 1> is the voice-timbre reference for <Subject 3> (S1), containing a spoken English vocal layer.

summary:
[reference generation + audio reference] The target video shows <Subject 3> eating a cookie in <Subject 1>. <Subject 4> enters with <Subject 2>, which lunges toward the cookie. The three-shot exchange uses <Audio 1> as the voice-timbre reference for <Subject 3> and ends with a canned audience laugh.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the Samoyed's thick white fur, pointed ears, dark nose, and curved tail are retained.
<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the blonde woman's identity, long hair, and light-pink shirt are retained.
<Subject 4> (appears in [Shot 1], [Shot 2]): fully_preserved - the young man's short wavy brown hair and dark-grey hoodie are retained.
<Audio 1>: reference - its vocal timbre guides the dialogue delivery of <Subject 3> without copying the original signal.

detailed_description:
The target video uses a realistic multi-camera sitcom style with warm indoor lighting.
[Shot 1] A medium shot establishes <Subject 1>, the coffee shop with its exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table. <Subject 3> (S1), the young woman with long blonde hair and a light-pink button-down shirt with rolled-up sleeves, sits on the sofa holding a chocolate-chip cookie. From the left, <Subject 4>, the young man with short wavy brown hair and a dark-grey hoodie with drawstrings, enters holding the leash of <Subject 2>, the thick-furred white Samoyed with pointed ears, a dark nose, and a curved tail. The dog lunges toward the cookie and pulls the leash taut. <Subject 3> (S1) jerks her hand back and, using the clear youthful voice timbre referenced from <Audio 1>, exclaims with light annoyance, <d>[English] Hey! Watch your dog!</d> She closes her lips and guards the cookie while <Subject 4> pulls the dog back.
[Shot 2] At 00:03.000, the shot cuts to a close-up of <Subject 4> (S2), the young man in the dark-grey hoodie from Shot 1, sitting beside <Subject 3> on the sofa and holding <Subject 2> securely in his arms. <Subject 4> (S2) says in a casual young male voice with a playful tone and an easy conversational pace, <d>[English] He just likes cookies more than me.</d> He closes his mouth into an apologetic smile and strokes the dog's thick white fur.
[Shot 3] At 00:05.000, the shot cuts to a close-up of <Subject 3> (S1), the blonde woman in the light-pink shirt from Shot 1. Her annoyance softens as she looks toward the Samoyed. <Subject 3> (S1) replies in the same clear youthful voice referenced from <Audio 1> with an amused cadence, <d>[English] Well, he has good taste at least.</d> She smiles and raises the cookie in a small toast-like gesture. A classic canned audience laugh begins immediately after the line and continues through the final frame.

overall_soundscape:
Soft indoor coffee-shop room tone continues throughout the scene.

non_diegetic_music:
N/A
```

Второй официальный образец — Ref2VA-редактирование `[video editing + audio reference + audio reuse]` (видео + две
звуковые роли: `<Audio 1>` partially_copy как фон, `<Audio 2>` reference как тембр) — в
`sources/github/MiniMax-AI__MiniMax-H3__README.md` (case-Ref2VA). Он полезен как образец формулировки «губы смыкаются ровно
когда голос стихает»: `Exactly as his voice stops, his lips meet in a relaxed, peaceful smile, and his jaw ceases speaking motion.`

#### Наш рабочий вариант Ref2VA с готовой озвучкой

Озвучка синтезируется заранее, на каждого говорящего — своя дорожка длиной во всю сцену (реплики на местах, вокруг тишина);
`subject_definitions`: `<Audio k> is the exact dialogue track of <имя>`; `retention_analysis`: `<Audio k>: fully_copy -
<имя>'s speech, copied exactly and lip-synced`; в описании — только действия без глаголов речи, реплика нейтрально
`(Sk): <d>[Russian] …</d>` плюс правила `All speech comes only from the audio tracks; outside them nobody speaks, hums or
makes vocal sounds.` и `Speech timing: each line is heard exactly once, only at (Sk) at <начало>–<конец>. Silence at <окна>:
…` (`sources/local/syn-video-editor-skill-minimax-h3.md`; `sources/local/synaptix-probe-findings.md`). Это не официальный
приём, а наш вывод из проб; его обоснование — звуковой референс не привязан ко времени ролика: без таких строк H3 сдвигает
реплику, повторяет её или заполняет паузы бормотанием (`sources/local/synaptix-probe-findings.md`).

---

## 3. Планы, камера, реплики, текст, звук

### 3.1 Планы и склейки

- `[Shot 1]` — **без** таймкода (начинается в 0.00 с). Следующие: `[Shot N] At MM:SS.mmm, the camera cuts to …` — номера
  подряд, время строго возрастает и лежит внутри длительности (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- Только одиночная отметка (`At 00:05.000`), не диапазон вида `[5s-8s]` (`sources/web/docs.comfy.org__prompt-guide.md`).
- Глаголы склейки: `the camera cuts to`, `the shot cuts to`, `the shot transitions to`, `the shot changes to`,
  `the shot switches to`; наплыв, затемнение, шторка — только по явной просьбе
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- Склейка должна приносить новую информацию (субъект, пространство, состояние, точка зрения, время); если меняется только
  крупность или чуть-чуть угол — лучше движение камеры (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
  Непрерывный морф — внутри одного плана с внутренним хронометражем (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`).
- Промежуток между отметками — это время, отведённое на действие и реплику; если реплики накладываются или действие
  пролетает — сократить сценарий или удлинить ролик (`sources/web/weshop__dialogue-and-timing.md`).
- Бюджет планов (неофициальный): 4–6 с → 1–2 плана; 7–10 с → 2–3; 11–15 с → 3–5
  (`sources/github/benjiyaya__Minimax-H3-Prompt-AgentSkill.md`). Один план дольше ~10 с — «скучно»
  (`sources/local/syn-video-editor-skill-minimax-h3.md`).
- Склейки соблюдаются «приблизительно»: в A/B поставщика структурный промпт дал склейки на ~3,92 и ~7,88 с при заказанных 4 и
  8 (короткий промпт — 3,54 и 6,79) (не проверено, один тест) (`sources/web/weshop__dialogue-and-timing.md`). По
  Whisper-разметке первые окна легли «в метку», последнее затянулось примерно на 2 с: порядок и присутствие соблюдаются
  жёстко, длительность — свободно (не проверено, n=1) (`sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`).

### 3.2 Словарь камеры: тип + амплитуда + скорость

| Измерение | Выражение | Значение |
|---|---|---|
| Тип | `Zoom In / Zoom Out` | меняется фокусное, камера стоит |
| Тип | `Push In / Pull Out` | камера едет вперёд / назад |
| Тип | `Pan Left / Pan Right` | поворот на месте по горизонтали |
| Тип | `Truck Left / Truck Right` | смещение вбок |
| Тип | `Tilt Up / Tilt Down` | поворот на месте по вертикали |
| Тип | `Pedestal Up / Pedestal Down` | камера целиком вверх / вниз |
| Тип | `Arc Shot` | дуга вокруг субъекта |
| Тип | `Tracking Shot` | сопровождение движущегося субъекта |
| Тип | `Static Shot` | камера и объектив неподвижны |
| Тип | `Shake Slightly / Shake Strongly` | лёгкая / сильная тряска |
| Тип | `POV` | взгляд субъекта |
| Тип | `Roll Clockwise / Roll Counterclockwise` | вращение вокруг оси объектива |
| Амплитуда | `with small amplitude` / `with large amplitude` | малое / большое изменение композиции |
| Скорость | `at slow speed` / `at fast speed` | медленно / быстро |

Средняя амплитуда и обычная скорость не пишутся. Камеру писать естественным действием внутри плана, а не списком ярлыков в
конце: `The camera pushes in with small amplitude at slow speed toward the folded letter in her hands.`
(`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Каждому плану — явное движение, хотя бы `Static Shot`
(неофициально) (`sources/github/benjiyaya__Minimax-H3-Prompt-AgentSkill.md`).

Дополнения практиков:
- H3 напрямую понимает плёночную лексику: фокус-перевод, «exposure breathing», зерно, халация (`sources/web/fal.ai__prompting-guide-44-examples.md`).
- «Не называть эффект (`low angle`), а описывать физические условия, которые его дают»; отказ от градусов (`ARC 40°`) в пользу
  START POSITION → PHYSICAL PATH → SUBJECT TRIGGER → END POSITION → FINAL FRAMING; одно главное движение камеры на сегмент
  (`sources/github/LoveRain1997__h3-prompt-journal.md`).
- Облёт на 180° только текстом нестабилен; с серой Blender-болванкой как референсным видео — 6 из 6 (не проверено, 1–2 зерна)
  (`sources/web/ai-muninn__pika-h3-flex-rtx5090-whitebox-orbit.md`).
- Явно указывать, куда смотрит камера и куда — каждый герой: неявное «смотрит вперёд» модель читает как «в камеру»
  (`sources/local/syn-video-editor-skill-minimax-h3.md`).

### 3.3 Говорящие и реплики

- Каждый голос (в том числе за кадром и пение) — устойчивый ID `(S1)`, `(S2)`; хор — `(S1,S2)`; ID не меняется между
  планами; молчащие персонажи ID не получают. ID нумерует голоса, а не персонажей
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`; `sources/web/docs.comfy.org__prompt-guide.md`).
- При первом появлении — приметы голоса: тип, возраст, пол, в кадре ли, высота, тембр, темп, акцент. Описание, ID, действие и
  подача — **снаружи** `<d>`; внутри — только `[Язык]` и дословные слова пользователя без перевода и правок:
  `The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>`;
  `The two children (S1,S2) shout together, <d>[English] Wait for us!</d>`
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- Пунктуация внутри `<d>`: оставлять `, . ? !`, убирать повторы и эмодзи, заканчивать `.`/`?`/`!` перед `</d>`,
  `[unclear]` — если не уверены (`sources/web/docs.comfy.org__prompt-guide.md`).
- **Закадровый голос** — точная фраза `says in an off-screen voiceover`, и сразу после `</d>` — что губы видимого героя
  сомкнуты: `The man (S1) says in an off-screen voiceover: <d>[English] I still remember that road.</d> while his lips remain
  completely closed.` (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). В A/B поставщика без этой формулы
  ведущая шевелила губами под мужской закадровый голос, с ней — нет (не проверено, один тест)
  (`sources/web/weshop__dialogue-and-timing.md`).
- **`<scenetrans>`** — реплика или строка песни, пересекающая склейку: метка в обоих точках стыка плюс явная фраза о
  непрерывности (`continues seamlessly across the cut`, `continues uninterrupted into the next shot`, `carries over from the
  previous shot`, `remains audible across the transition`). **`<cutoff>`** — речь обрывается концом ролика
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Точное положение тега внутри/снаружи `<d>` в официальных
  текстах примером не показано (наш вывод из `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`); у ethanfel
  компилятор ставит `<scenetrans>`/`<cutoff>` сам по флагу непрерывности (`sources/github/ethanfel__ComfyUI-MiniMax-H3-Guide.md`).
- Привязывать реплики к видимым событиям, а не к таймкодам (`sources/web/docs.comfy.org__prompt-guide.md`); при этом наш
  опыт с готовой озвучкой требует явных окон речи/тишины с таймкодами (`sources/local/syn-video-editor-skill-minimax-h3.md`) —
  см. раздел 7.
- Длина реплики под длину ролика: эмпирика ~0,36 с на слово (англ., не проверено) (`sources/github/MiniMax-AI__MiniMax-H3__issues.md`,
  #17); ~2,5 слога/с, реплики 3–8 слов и молчаливое действие после последней реплики (наши A/B)
  (`sources/local/synaptix-git-log-h3.md`). Пятисекундная фраза в 15-секундной сцене провоцирует заполнитель, дыхание и
  повторы — лишнее время убирать (`sources/web/seedance.tv__gibberish-fix.md`); после короткой реплики добавить действие,
  чтобы заполнить таймлайн без импровизированной речи (`sources/web/hackernoon__prompt-guide.md`).
- Ролик без речи — сказать прямо: `No one speaks, shouts, hums or makes any vocal sound in this shot; all mouths stay closed`
  (`sources/local/syn-video-editor-skill-minimax-h3.md`). Если появляется непрошеная речь — заполнить оба звуковых поля явно
  (`sources/web/docs.comfy.org__prompt-guide.md`).

### 3.4 Текст в кадре

- Любой видимый баннер, вывеска, ярлык, субтитр, неон — в английских двойных кавычках, дословно на исходном языке:
  `A red neon sign reading "营业中" glows above the doorway.` Реплики — в `<d>`, не в кавычках
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`; `sources/web/docs.comfy.org__prompt-guide.md`).
- Невыписанный текст рисуется как «текстура букв»: HUD, названный в промпте лишь «HUD elements», вышел мусором («ETR METNO
  CITFEP»), а выписанные строки — чисто (`sources/web/atlascloud__prompt-guide-45-official.md`). Если текст нечитаем —
  уменьшить его количество (`sources/github/MiniMax-AI__MiniMax-H3__video-skills.md`, co-op-game-intro, STEP 7). Короткие
  строки проще проверить (`sources/web/hackernoon__prompt-guide.md`).
- Наличие текста в промпте — не доказательство: буквы и синхрон с битом проверять по результату
  (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`).

### 3.5 Звук и музыка

- Внутрикадровое (диегетическое) — в описании плана: речь, пение, радио, телефон, инструмент в кадре. `overall_soundscape` —
  фон и физика за весь ролик, без повтора реплик и музыки. `non_diegetic_music` — только партитура для зрителя:
  инструменты, темп, ритм, динамика, без «эмоциональных» слов и объяснения функции музыки
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Одну и ту же музыку не писать в оба места
  (`sources/github/benjiyaya__Minimax-H3-Prompt-AgentSkill.md`).
- Звуки привязывать к событию, которое их производит (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`); каждый —
  конкретным звуком (`kettle whistling`, `door slams`) (`sources/local/syn-video-editor-skill-minimax-h3.md`).
- Если ничего не сказать о звуке, модель всё равно сделает дорожку — «какую захочет»; сильные официальные примеры дают
  музыке посекундную «партитуру» («at 6 seconds the jazz bass groove joins») (`sources/web/atlascloud__prompt-guide-45-official.md`;
  `sources/web/fal.ai__prompting-guide-44-examples.md`).
- Звуковой референс: в `subject_definitions` — роль, в `retention_analysis` — `fully_copy`/`partially_copy`/`reference`/
  `weak_reference`, в звуковых полях — связь только в том слое, к которому она относится:
  `overall_soundscape: The copied ambience layer from <Audio 1> continues throughout the target video.` /
  `non_diegetic_music: <Audio 2> is directly reused as the complete audience-only score.`
  (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
- Голос: аудио несёт тембр, а не слова; подачу указывать текстом; при точной привязке — по одному аудио на говорящего
  (`sources/web/docs.comfy.org__h3-native-workflows.md`). Формулировки ролей: `use <Audio 1> exactly as it is` против
  `follow the timbre of <Audio 2>` (`sources/web/docs.comfy.org__h3-native-workflows.md`).
- Даже при «точном копировании» реконструкция через аудио-VAE не побитовая; если нужна исходная дорожка — накладывать её
  потоком при сборке (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`). Сообщество делает так же: оригинальный
  MP3 пропускается в итог в обход генерации, потому что H3 «перегенерирует» вход в «псевдокитайский вокал»
  (`sources/web/civitai__workflow-audio-pseudo-chinese-fix.md`). Мы кладём исходную озвучку поверх при монтаже
  (`sources/local/syn-video-editor-skill-minimax-h3.md`).
- Чистые голосовые дорожки без музыки дают лучшую синхронизацию губ (`sources/local/syn-video-editor-skill-minimax-h3.md`).
- Заданный BPM — цель, а не измеренный темп (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`).

### 3.6 Встраивания-эффекты

ComfyUI понимает `embedding:имя` в промпте H3; 10 неофициальных эффектов (`minimaxh3_bullet_time`, `minimaxh3_four_seasons`,
…) (`sources/web/docs.comfy.org__prompt-guide.md`). Ставить в середину предложения: в самом начале промпта эффект молча не
срабатывает (не проверено) (`sources/web/ai-muninn__minimax-h3-effect-embeddings-rtx5090.md`).

---

## 4. Приёмы из практики

### 4.1 Раскадровка

- Картинка-раскадровка не заменяет текст: H3 читает сетку 4×4 как одну картинку и берёт из неё лицо, костюм, палитру и позы,
  но не порядок. Одна фраза — 2–3/16 панелей; поканельный таймлайн — 12/16; тот же таймлайн в «официальном формате
  shot-table» (пять элементов на секунду + передача состояния + якоря) — 13/16; без картинок вообще — 12/16, но лицо другое
  (не проверено, n=1, одно зерно) (`sources/web/ai-muninn__storyboard-grid-h3-ref2va-rtx5090.md`).
- Квадратная раскадровка при `ref_image_size=match` сжимается до ~1024², панель ≈ 256² — лицу не хватает пикселей; герою нужен
  отдельный лист персонажа (`sources/web/ai-muninn__storyboard-grid-h3-ref2va-rtx5090.md`). Сетку и подписи H3 в кадр не
  переносил (`sources/web/ai-muninn__storyboard-grid-h3-ref2va-rtx5090.md`).
- В конце строки каждого плана — «передача» (состояние реквизита и позы на вход следующего): ровно на этих местах формат дал
  выигрыш (`sources/web/ai-muninn__storyboard-grid-h3-ref2va-rtx5090.md`).
- Официальный формат таблицы планов (навык 3D-мультфильма): шесть колонок — ID и длительность; Continuity Handoff; Reference
  Anchors (фиксированные ориентиры с экранными позициями, позиции и взгляды героев, статус вышедших героев, базовый свет);
  Hook Type; описание с посекундными директивами из пяти элементов (действие/поза/мимика; камера; положение; звук;
  передача на следующую секунду); звуковая дорожка. План ≤ 15 с, ≤ 3 важных героя, каждая секунда покрыта
  (`sources/github/MiniMax-AI__MiniMax-H3__video-skills.md`). Там же: для H3 посекундную директиву «можно отправлять почти
  дословно» (`sources/github/MiniMax-AI__MiniMax-H3__video-skills.md`).
- Рабочие подписи раскадровки (`[char:…]`, `[shot:…]`, стрелки, номера) перед рендером удалять
  (`sources/github/MiniMax-AI__MiniMax-H3__video-skills.md`); стрелки и рамки на референсах могут проявиться в кадре —
  инструкции держать в тексте (`sources/github/ethanfel__ComfyUI-MiniMax-H3-Guide.md`).
- Грубая композиция (эскиз, коллаж, болванка) работает, если ясны пропорции, точка и высота камеры, число героев, их
  экранные позиции и взгляды, планы глубины; в описании роли — что игнорировать (`Ignore the sketch style, placeholder
  colors…`) (`sources/github/ethanfel__ComfyUI-MiniMax-H3-Guide.md`).

### 4.2 Фиксация персонажа

- «Назвать лицо, а не описать»: текст «седые виски, грубая одежда» дал старика вместо героя; помог портрет-референс из
  картиночной модели и переход T2V → R2V (`sources/web/ai-muninn__minimax-h3-rtx5090-character-lock.md`).
- На 480p лицо в общем плане — несколько десятков пикселей; апскейл размытость не лечит — снимать крупнее
  (`sources/web/ai-muninn__minimax-h3-rtx5090-character-lock.md`).
- Метка `<Subject N>` — лишь указатель: без внешности в тексте плана модель придумывает человека; официальный пример даёт
  полное описание при первом появлении и укороченное, но опознаваемое — дальше (`the young man in the dark-grey hoodie from
  Shot 1`) (`sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`;
  `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Спорность — раздел 7.
- Один «замороженный» абзац внешности, вставляемый в каждый план без перефразировок; меняется только действие
  (`sources/web/domoai__character-consistency.md`); перечислять сохраняемые приметы
  (`sources/web/fal.ai__prompting-guide-44-examples.md`). Пакет референсов: нейтральный крупный план, во весь рост, профиль,
  эмоции, детали костюма, палитра — без противоречий между картинками (`sources/web/domoai__character-consistency.md`);
  эталоны героев — во весь рост, внешность одними и теми же словами в каждой сцене
  (`sources/local/syn-video-editor-skill-minimax-h3.md`).
- Апскейл Open-H3-Regenerate-2K добавляет деталь, но лица могут «плыть», а мелкий текст вывесок — превращаться в псевдоиероглифы (не проверено)
  (`sources/web/INDEX.md`, аннотация к Open-H3-Regenerate-2K).

### 4.3 Два и больше персонажей

- Два листа персонажа, шесть разделов, 243 кадра: два разных голоса, без «перетекания» лиц; дизайн героев делать максимально
  непохожим (рост, цвет, возраст) (не проверено, n=1) (`sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`).
- ID голоса работает по ID, а не по порядку появления (A→B→A проверено на двух говорящих, n=1)
  (`sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`); но в официальных issues — массовые жалобы: с двумя
  голосовыми референсами чередование A→B→A теряет или меняет голос, A→B надёжно; однополые голоса сливаются, в том числе в
  платной закрытой версии; лицо и губы при этом назначены верно — «плывёт» только голос
  (`sources/github/MiniMax-AI__MiniMax-H3__issues.md`, #17, #100).
- Обходы из issues (каждый — частичный): (1) отдельный слот и новый `(Sx)` на каждый возврат говорящего (дубль того же файла
  как `<Audio 3>`, `S1/S2/S3`) — у одного автора помогло, в контролируемом повторе #100 — полная перестановка голосов;
  (2) изменить только длительность на ±1–2 с и перегенерировать; (3) явно описывать слушающего (рот закрыт) в каждой реплике;
  (4) ≤ 6 смен говорящих на план; (5) убрать даже мимолётные вокализации неговорящих («вместе хихикают») — они считаются
  речью и сбивают привязку; (6) референс голоса длиной 30 с — слова референса просачиваются в речь
  (`sources/github/MiniMax-AI__MiniMax-H3__issues.md`, #17, #100).
- Появление героя в нужный момент управляемо директивой в таймлайне («входит на 6–8 с» — отсутствует на 1,5 и 4,5 с, есть на
  7 с) (не проверено, n=1) (`sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`).
- Позиции — в экранных координатах (левая/правая треть, проценты ширины, передний/средний/задний план, взгляд, масштаб)
  (`sources/github/ethanfel__ComfyUI-MiniMax-H3-Guide.md`); в разговоре у каждого явно — к кому повернут и куда смотрит
  (`sources/local/syn-video-editor-skill-minimax-h3.md`).
- Замена людей в видео: явная взаимно однозначная привязка `<Subject 3> → <Subject 1>`; цели — `fully_preserved`, треки
  исходника — `attribute_transfer`, видео — `partially_preserved`; общий абзац ограничений вместо повторения в каждом разделе
  (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`).

### 4.4 Концовка-кадр, продолжение, длинные сцены

- Закреплённый последний кадр (FL2VA/L2VA, `end_image` в облаке) — ещё и экономия: большинство пересъёмок — из-за «уплывшей»
  концовки (`sources/web/atlascloud__prompt-guide-45-official.md`).
- Опорные кадры в любой точке: `MiniMaxH3AddGuide` с `frame_idx` (отрицательный — от конца). Картинки, подключённые только к
  AddGuide, кодировщик текста не видит — чтобы промпт мог ссылаться на них как на `<Picture N>`, их подают ещё и в
  `ref_images` (`sources/web/docs.comfy.org__h3-multiframe-reference.md`; `sources/web/docs.comfy.org__h3-native-workflows.md`).
- Продолжение с латентного хвоста описывать как тот же ракурс, ту же расстановку, позы и свет (`This shot continues seamlessly
  from the previous shot`); иначе H3 «растворяет» хвост и по пути показывает эталоны на белом фоне
  (`sources/local/synaptix-probe-findings.md`; `sources/local/syn-video-editor-skill-minimax-h3.md`). Смена сцены — новыми
  референсами сцены: латентный хвост прошлой сцены может сохранить не ту обстановку
  (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`).
- Длиннее 15 с — резать на ролики с явными стартовыми и конечными состояниями (`sources/web/hackernoon__prompt-guide.md`);
  сегменты самодостаточны, выход сегмента = вход следующего, стыки — «локальный крупный план части тела» с продолжением
  движения (`sources/github/LoveRain1997__h3-prompt-journal.md`). Перенос движения — не длиннее 8–9 с за раз
  (`sources/github/LoveRain1997__h3-prompt-journal.md`).
- Музыка как каркас: таймлайн от реальной дорожки (onset-сетка) или подробный текстовый BGM с акцентами, и прямо сказать, что
  движение ведёт эта музыка (`sources/github/LoveRain1997__h3-prompt-journal.md`).

### 4.5 Перенос движения и референсное видео

- Прокси движения: точка передаёт только путь и тайминг, для поворотов нужен ориентированный прокси (круг с «носом»), для
  суставов — скелет; в роли явно: что переносится, что неподвижно, что игнорировать
  (`sources/github/ethanfel__ComfyUI-MiniMax-H3-Guide.md`).
- Каждому референсу — работа одной фразой в тексте («Use Image 1 for the overall mood…; Image 2 for the talent…»)
  (`sources/web/fal.ai__prompting-guide-44-examples.md`; `sources/web/atlascloud__prompt-guide-45-official.md`); стиль-референс
  не должен приносить своих людей, пейзаж, шрифт в чужие планы (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`).
- Редактирование видео — список замен вместе с тем, что остаётся неизменным (`sources/web/fal.ai__prompting-guide-44-examples.md`).

### 4.6 Постановка (наш опыт)

- Локация `fully_preserved` и неизменна; входы/выходы только через то, что есть на эталоне: «входит в дверь» при двери вне
  кадра заставляет H3 переделать стену (`sources/local/syn-video-editor-skill-minimax-h3.md`).
- Без действия H3 снимает «говорящие головы» — в каждом ролике биты действия с таймкодами; ключевое событие — отдельным
  планом, лицо видно, ничто не закрывает (`sources/local/syn-video-editor-skill-minimax-h3.md`).
- Ночь — читаемая, с источником света (`moonlit rim light`), не `pitch black` (`sources/local/syn-video-editor-skill-minimax-h3.md`).
- Turbo/другие LoRA — только на весь фильм: у разных LoRA разная цветопередача, стык виден
  (`sources/local/synaptix-probe-findings.md`).

### 4.7 Итерации

- Менять **один** параметр/категорию за раз (диалог, личность, камера, звук) (`sources/web/hackernoon__prompt-guide.md`;
  `sources/web/domoai__character-consistency.md`); для сравнений фиксировать входы, модель, зерно, LoRA и семплинг
  (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`).
- Сначала перегенерировать с другим зерном; после трёх дисциплинированных неудач — сократить реплику, перевести в B-roll с
  закадровым голосом или переозвучить в монтаже (`sources/web/seedance.tv__gibberish-fix.md`).
- Пробовать на 5 с, сдавать на 10–15 (облако, оплата посекундно) (`sources/web/atlascloud__prompt-guide-45-official.md`);
  540p — проба, 768p — чистовик (`sources/local/syn-video-editor-skill-minimax-h3.md`).
- Без лимита — одна целевая перегенерация, затем отчёт о дефекте (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`).
- Удачную конфигурацию диалога сохранять целиком: похожая сцена с тем же таймингом может не сработать
  (`sources/github/MiniMax-AI__MiniMax-H3__issues.md`, #17).

---

## 5. Типичные ошибки → исправление

| Симптом | Вероятная причина | Исправление | Источник |
|---|---|---|---|
| Красивый «живой стоп-кадр», почти нет движения; один медленный наезд на 10 с | описан момент, а не последовательность | действие-триггер, промежуточное изменение, конечное состояние; таймлайн по планам | `sources/web/hackernoon__prompt-guide.md`; `sources/web/atlascloud__prompt-guide-45-official.md` |
| Камера противоречит действию | ярлыки камеры списком | одно естественное предложение: тип, направление, амплитуда, скорость | `sources/web/hackernoon__prompt-guide.md`; `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md` |
| Герой говорит лишнее / импровизирует | реплика подразумевается или в простых кавычках, нет ID | `(S1)` + точный `<d>[Язык] …</d>`, описать действия молчащих; короче ролик | `sources/web/hackernoon__prompt-guide.md`; `sources/web/seedance.tv__gibberish-fix.md` |
| Бормотание, «зачитывание» описания | описание сцены не по-английски | описание по-английски, речь только в `<d>[Russian] …</d>` | `sources/local/synaptix-probe-findings.md` |
| Герои говорят по-английски вместо русского | реплики не в `<d>[Russian]` | `<d>[Russian] …</d>` | `sources/local/syn-video-editor-skill-minimax-h3.md` |
| Голос не тому, чужие губы шевелятся под закадровый голос | нет формулы закадрового голоса | `says in an off-screen voiceover` + `while … lips remain completely closed` | `sources/web/weshop__dialogue-and-timing.md`; `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md` |
| Повтор реплики до конца ролика, речь в паузах, хвост с выдуманной речью | звуковой референс короче ролика / не привязан ко времени | длина ролика по дорожке; окна речи и тишины словами с таймкодами; «вся речь только из дорожек» | `sources/local/synaptix-probe-findings.md` |
| Лишние/повторённые фразы и срезанное начало при отдельном голосовом референсе | (не установлено) референс-звук | в issue — отключение референса как обход; эталонного решения нет | `sources/github/MiniMax-AI__MiniMax-H3__issues.md` (#95) |
| Голоса меняются местами в A→B→A | слабая привязка голоса к субъекту | см. 4.3: слот на каждый возврат, ±1–2 с длительности, слушающий с закрытым ртом, без вокализаций молчащих | `sources/github/MiniMax-AI__MiniMax-H3__issues.md` (#17, #100) |
| Лишний текст, субтитры, мусорные буквы | текст не выписан | выписать каждую строку в кавычках; если читаемость плохая — меньше текста | `sources/web/atlascloud__prompt-guide-45-official.md`; `sources/github/MiniMax-AI__MiniMax-H3__video-skills.md` |
| Чужое лицо при описанной внешности | текст описывает тип, а не человека | Ref2VA + портрет-референс | `sources/web/ai-muninn__minimax-h3-rtx5090-character-lock.md` |
| Лицо дрейфует, «придуман» человек | метка `<Subject N>` без примет | приметы в каждом плане (минимум — укороченно) | `sources/web/ai-muninn__ref2va-two-characters-rtx5090.md` |
| Идентичности перепутаны/смешаны (замена людей) | порядок картинок ≠ порядку входов, нет однозначной привязки | сначала проверить порядок и фактические входы, затем привязки, потом прозу | `sources/github/vllm-project__vllm-omni__agent-skills-h3.md` |
| Раскадровка проигнорирована | раскадровка + одна фраза | поканельный таймлайн с передачей состояния | `sources/web/ai-muninn__storyboard-grid-h3-ref2va-rtx5090.md` |
| FL2VA: склейка или морф вместо превращения | заданы концы, но не путь | один план, промежуточные позы/предметы/свет | `sources/web/hackernoon__prompt-guide.md` |
| L2VA: финал достигнут слишком рано | нет постепенного подхода | оставить последний момент под сходимость | `sources/web/hackernoon__prompt-guide.md` |
| Стена/обстановка перестраивается | действие требует того, чего нет на эталоне (дверь вне кадра) | входы только через видимое; `the room layout does not change` | `sources/local/syn-video-editor-skill-minimax-h3.md` |
| Герой развёрнут в камеру | неявное «смотрит вперёд» | явно: куда смотрит, к кому повернут | `sources/local/syn-video-editor-skill-minimax-h3.md` |
| На стыке с хвоста — чужой кадр, эталоны на белом | продолжение описывает другой кадр | тот же ракурс и расстановка, новый план — движением камеры или `[CUT]` | `sources/local/synaptix-probe-findings.md` |
| Референсы молча не применились (ComfyUI API) | вложенный словарь `ref_images` в JSON | ключи `ref_images.ref_image_0` верхнего уровня | `sources/web/ai-muninn__storyboard-grid-h3-ref2va-rtx5090.md` |
| Промпт ссылается на `<Audio 1>`, а звук не подключён | несвязанная метка | убрать метку и связанные с ней фразы | `sources/github/MiniMax-AI__MiniMax-H3__issues.md` (#95); `sources/github/vllm-project__vllm-omni__agent-skills-h3.md` |
| Тарабарщина в многоязычной/многоголосой сцене | перегруз, смешение языков | один язык, одна реплика/один говорящий на клип, стабильные ID | `sources/web/seedance.tv__gibberish-fix.md` |

---

## 6. Переписчики промптов (PE)

| Что | Тип | Вход | Режимы | Как применять | Источник |
|---|---|---|---|---|---|
| **H3-Context-IR** (официальный) | облачный сервис MiniMax, `POST /v2/h3_context_ir`; узел ComfyUI «MiniMax H3 Context IR (Prompt Enhancer)» | короткий текст + медиа с ролями (first/last frame **или** референсы; не вместе), длительность 4–15, `ratio` | все | выход → в промпт локального H3; те же медиа в том же порядке (промпт ссылается по позиции). Открытого кода нет | `sources/github/MiniMax-AI__MiniMax-H3__README.md`; `sources/github/MiniMax-AI__MiniMax-H3__scripts-and-configs.md`; `sources/github/Comfy-Org__docs__minimax-h3-node-reference.md` |
| `prompt_expansion_mode` | параметр облачного API v2 (модели Max) | — | — | `disabled` / `balanced` / `quality` | `sources/web/platform.minimax.io__api-video-generation-v2.md` |
| Навык `h3-prompt-writing` (официальный) | правила для любого LLM-агента (SKILL.md + base-en.txt + ref-en.txt) | бриф + описание медиа | все | агент пишет промпт по правилам; без внешних API | `sources/github/MiniMax-AI__MiniMax-H3__README.md`; `sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md` |
| lightx2v Prompt-Rewriter-LoRA | LoRA на Qwen3.6-27B | только текст + пропорции + длительность | T2VA | `infer.py --greedy`; «приближение, а не копия Context-IR»; может добавлять детали — проверять речь, личность, тайминг | `sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-LoRA.md` |
| lightx2v …-LoRA-8B | LoRA на Qwen3-VL-8B | текст + первый/последний кадр (видит картинки) | T2VA, I2VA, L2VA, FL2VA; Ref2VA нет | то же | `sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-LoRA-8B.md` |
| lightx2v …-LoRA-Omni | LoRA на Qwen2.5-Omni-7B | картинки, видео, звук | все пять | выдаёт `enhanced_prompt`; те же медиа — в генератор | `sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-LoRA-Omni.md` |
| lightx2v Qwen3.5-9B-FL2VA / -Ref2VA (и 27B-версии на Qwen3.8) | полные SFT-модели под vLLM | FL2VA: текст + кадры; Ref2VA: картинки + кадры из видео (по 8), звук — только метками, содержимое не читается | FL2VA-чекпойнт: T2VA/I2VA/L2VA/FL2VA; Ref2VA-чекпойнт: Ref2VA | `vllm_server.sh` + `post_vllm.sh`; контекст 32768 токенов | `sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-FL2VA.md`; `sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-Qwen3.5-9B-Ref2VA.md`; `sources/github/pytraveler__MiniMax-H3-Prompt-Rewriter-ComfyUI.md` |
| LeViAtHaN2K …27B-Ref2VA-heretic | «расцензуренная» версия Ref2VA-переписчика | как у исходника | Ref2VA | отказы 3/100 против 98/100 у исходника (не проверено) | `sources/hf/cards/LeViAtHaN2K__MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.md` |
| Wuli-art Gemma4-for-MiniMax-H3-Prompt-Extend | дообученная Gemma-4-12B-IT, одни веса на FL2VA и Ref2VA; BF16/FP8/INT8 | инструкция + кадры/референсы | FL2VA-формат (T2VA/I2VA/L2VA/FL2VA) и Ref2VA | два готовых воркфлоу ComfyUI; сравнение «Raw / Gemma4 / официальный PE / lightx2v / наш» на 9 случаях (не проверено, выбор самих авторов) | `sources/hf/cards/Wuli-art__Gemma4-for-MiniMax-H3-Prompt-Extend.md` |
| pytraveler MiniMax-H3-Prompt-Rewriter-ComfyUI | пак узлов: LoRA-переписчики (27B/8B/Omni), Qwen SFT/GGUF, «guide-based» писатели на любом GGUF, API-писатель, капшенеры референсов, **Prompt Check** (проверка чужого промпта без модели), **Prompt Reducer** (сжатие обратно в одну фразу), 1000 готовых пресетов | промпт на любом языке → ответ по-английски | все | 27B: ~52 ГБ диска, ~16 ГБ VRAM в nf4; кодировщик H3 переиспользовать нельзя | `sources/github/pytraveler__MiniMax-H3-Prompt-Rewriter-ComfyUI.md` |
| ethanfel ComfyUI-MiniMax-H3-Guide | компилятор плана (Project → References → Shot → Prompt Merge) + опц. Qwen-«улучшатель» | типизированные роли референсов, планы, реплики | все | LLM правит только прозу; метки, роли, таймкоды, реплики, порядок планов заблокированы и проверяются; есть Review Gate для ручной правки | `sources/github/ethanfel__ComfyUI-MiniMax-H3-Guide.md` |
| 1038lab H3-Promptor | анализ медиа + структурирование любым LLM (облачные, Ollama, QwenVL) | медиа + текст | все (+ свои V2VA, A2V) | «заморозка» заголовков планов и меток при правке; шапки планов в виде диапазонов `[Shot 1: 00:00.000 – 00:02.800]` | `sources/github/1038lab__ComfyUI-MiniMax-H3-Promptor.md` |
| T8mars prompt-enhancer | китайский пак-улучшатель (API/GGUF), конфиги цвета и «режиссёрской манеры», экспериментальные пространственные метки | текст + картинки | H3 и Seedance | опции по умолчанию выключены; эффект ряда функций «ещё ждёт A/B» | `sources/github/T8mars__comfyui-minimax-h3-prompt-enhancer-T8.md` |
| benjiyaya Minimax-H3-Prompt-AgentSkill | навык для агента (Hermes, Claude, Codex) | идея + медиа | все | «творческое обогащение» по 7 измерениям + официальный формат | `sources/github/benjiyaya__Minimax-H3-Prompt-AgentSkill.md` |
| Omni-Rewriter (сообщество) | типизированный черновик → проверка → ограниченная правка | бриф | все | авторы прямо не претендуют на воспроизведение Context-IR | `sources/github/MiniMax-AI__MiniMax-H3__issues.md` (#41) |

Не путать с переписчиками:
- **MiniMax-H3-Image-PE** — официальный расширитель промптов для **картиночной** модели H3-Image (Qwen3.8-27B), не для видео
  (`sources/hf/cards/MiniMaxAI__MiniMax-H3-Image-PE.md`).
- **Alissonerdx/Qwen3.5-9B-Minimax-H3** и **CondBridge-Qwen3.5-4B** — замены 33B-кодировщика текста (проекция/адаптер к
  пространству кондиционирования), а не переписчики; их метрики — косинусы к учителю (не проверено)
  (`sources/hf/cards/Alissonerdx__Qwen3.5-9B-Minimax-H3.md`; `sources/hf/cards/SpXMerlin1D__MiniMaxH3-CondBridge-Qwen3.5-4B.md`).
- **KJNodes MiniMaxH3PromptWeight** — веса токенов `(word:1.5)`, `(word:-1)` «удаляет понятие», смешивание `[cat|fox]`;
  описан как кондиционирование H3-Image, но с параметрами для видеорежимов — применимость к видео не проверена
  (`sources/github/kijai__ComfyUI-KJNodes__minimax-nodes-source.md`).

Общие правила применения: решить, что сдаётся (промпт или видео) — промпт-задача заканчивается промптом; бриф — на языке
пользователя, рендер-промпт — по-английски; перед отправкой проверить порядок полей, номера планов, таймкоды, метки,
дословность реплик, язык, длительность и отсутствие незаполненных заполнителей
(`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`). Для честного сравнения переписчиков — одинаковое
декодирование и зерно, проще всего greedy (`sources/hf/cards/lightx2v__MiniMax-H3-Prompt-Rewriter-LoRA.md`).

---

## 7. Расхождения между источниками

1. **Негатив и запреты.**
   - ComfyUI: cfg 1, негативной ветки нет, списки запретов в позитиве вредят — писать желаемое
     (`sources/web/docs.comfy.org__prompt-guide.md`).
   - Atlas Cloud и fal: «негативный список» — один из шести обязательных блоков и «необычайно эффективен» («No soft
     dissolves…», «Do not add subtitles») (`sources/web/atlascloud__prompt-guide-45-official.md`;
     `sources/web/fal.ai__prompting-guide-44-examples.md`). Но это облако с Context-IR, который переписывает текст.
   - Официальные навыки MiniMax сами кладут «negative prompt» в описание плана и «negative constraints» в финальный промпт
     (`sources/github/MiniMax-AI__MiniMax-H3__video-skills.md`) — для облачного H3.
   - vLLM-Omni: не копить избыточные запреты и не называть отсутствующий во входах аксессуар — описывать желаемое
     положительно (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`).
   - Локальные примеры с запретами в тексте существуют (BeatAPI, RTX 5070: «No subtitles, narration, or additional
     dialogue.»), и наши рабочие строки тоже отрицательные («No one speaks…», «Silence at …: nobody speaks») — и помогают
     (`sources/github/BeatAPI__awesome-minimax-h3-prompts.md`; `sources/local/syn-video-editor-skill-minimax-h3.md`).
   - Вывод для итогового документа: запрет как отдельное поле при cfg 1 невозможен; короткие утверждения о том, чего
     «нет» в кадре/звуке, на практике работают, длинные списки — под вопросом. Нужен свой A/B.
2. **Таймкоды: отметка или диапазон.** Официально — одиночная отметка `At MM:SS.mmm`, первый план без неё; ComfyUI прямо
   запрещает диапазоны `[5s-8s]` (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`;
   `sources/web/docs.comfy.org__prompt-guide.md`). Atlas Cloud называет диапазоны `[0s-2s]` признаком сильнейших официальных
   примеров (`sources/web/atlascloud__prompt-guide-45-official.md`); ai-muninn в скелете пишет `[Shot 1] 0-3s`
   (`sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`); 1038lab — `[Shot 1: 00:00.000 – 00:02.800]`
   (`sources/github/1038lab__ComfyUI-MiniMax-H3-Promptor.md`); официальная таблица планов — посекундные `0–1s`
   (`sources/github/MiniMax-AI__MiniMax-H3__video-skills.md`); у нас — `At 00:06.0` с одним знаком
   (`sources/local/syn-video-editor-skill-minimax-h3.md`).
3. **Где стоит фраза стиля в базовых режимах.** Официально — после `[Shot 1]` (в Ref2VA — до `[Shot 1]`)
   (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). benjiyaya требует 1–2 предложения стиля **до**
   `[Shot 1]` и в базовых режимах («MUST») (`sources/github/benjiyaya__Minimax-H3-Prompt-AgentSkill.md`).
4. **Повторять ли внешность в каждом плане.** ai-muninn: метка без примет — модель придумывает человека, повторять в каждом
   плане (`sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`); domoai — один неизменный абзац в каждый план
   (`sources/web/domoai__character-consistency.md`); официальный пример — полностью при первом появлении, далее укороченно
   (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`); LoveRain — «图1零外貌»: при картинке-замке внешность
   текстом не переписывать вовсе (`sources/github/LoveRain1997__h3-prompt-journal.md`).
5. **Лист персонажа: отдельная `<Picture N>` или внутри `<Subject N>`.** Официально: картинка, которая только определяет
   героя/стиль, отдельной строки не получает (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`); то же как
   «частая ошибка №1» у benjiyaya (`sources/github/benjiyaya__Minimax-H3-Prompt-AgentSkill.md`). ComfyUI: многопанельному
   листу — собственную `<Picture N>`, «не только внутри `<Subject N>`», и указывать планам конкретную панель
   (`sources/web/docs.comfy.org__prompt-guide.md`).
6. **Реплики: к событиям или к таймкодам.** ComfyUI: привязывать к видимым событиям, не к таймкодам
   (`sources/web/docs.comfy.org__prompt-guide.md`). Наш опыт с готовой озвучкой и автор issue #17: явные окна времени
   обязательны/полезны (`sources/local/syn-video-editor-skill-minimax-h3.md`; `sources/github/MiniMax-AI__MiniMax-H3__issues.md`).
7. **Что делает звуковой референс.** ComfyUI: «аудио несёт тембр, а не слова» (`sources/web/docs.comfy.org__h3-native-workflows.md`);
   официальный формат допускает `fully_copy` всей дорожки (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`),
   и у нас `fully_copy` даёт синхрон губ под готовую речь (`sources/local/synaptix-probe-findings.md`); но сообщество видит
   «перегенерацию в псевдокитайский вокал» (`sources/web/civitai__workflow-audio-pseudo-chinese-fix.md`), а vLLM-Omni — что
   реконструкция не побитовая (`sources/github/vllm-project__vllm-omni__agent-skills-h3.md`). Issue #100: в ComfyUI в текст
   уходит только `"<Audio j>: "` без содержимого, голоса различаются лишь позицией на оси времени
   (`sources/github/MiniMax-AI__MiniMax-H3__issues.md`).
8. **Возврат говорящего A→B→A.** Обход «новый `(S3)` + дубль аудио» помог в #17 и не помог в контролируемом повторе #100
   (`sources/github/MiniMax-AI__MiniMax-H3__issues.md`). ai-muninn на одной сцене получил верные голоса и при возврате S1
   (`sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`). Возможное объяснение — разнополые/сильно разные голоса
   у ai-muninn против однополых в #17/#100 (наш вывод).
9. **«Одно действие на план».** benjiyaya — «жёсткое ограничение H3» (`sources/github/benjiyaya__Minimax-H3-Prompt-AgentSkill.md`);
   официальные примеры дают несколько действий в одном плане (зонтик в FL2VA, разбитый стакан в L2VA)
   (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`); официальный навык мультфильма требует пять элементов
   на каждую секунду (`sources/github/MiniMax-AI__MiniMax-H3__video-skills.md`).
10. **Музыка без «настроения».** Правило: без абстрактных слов настроения и без объяснения функции музыки
    (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Выход самого Context-IR для I2VA: «heartwarming… enhances
    the cozy, nostalgic, and joyful atmosphere»; T2VA — «mournful French horn» (`sources/github/MiniMax-AI__MiniMax-H3__README.md`).
11. **Разделитель полей.** Навык: пустая строка между полями и после строки выравнивания
    (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`). Выход Context-IR для T2VA — поля через одиночный `\n`
    (`sources/github/MiniMax-AI__MiniMax-H3__README.md`); наш навык — «одной строкой, поля по порядку»
    (`sources/local/syn-video-editor-skill-minimax-h3.md`). Влияние формы разделителя никто не измерял.
12. **Язык описания при русской речи.** Наш коммит f06332f5: писать строки сцен целиком по-русски («H3 говорит на языке
    описания»); через несколько коммитов в тот же день (91448b2a) — описание по-английски, потому что русское описание H3
    зачитывал вслух (`sources/local/synaptix-git-log-h3.md`). Действующее правило — английское описание +
    `<d>[Russian]` (`sources/local/syn-video-editor-skill-minimax-h3.md`); официально — тоже английский
    (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`).
13. **Названия движений камеры.** Официально — именованные типы (`Arc Shot`, `Pan Left`…)
    (`sources/github/MiniMax-AI__MiniMax-H3__prompt-writing-skill.md`); LoveRain: имена эффектов и градусы H3 «усредняет»,
    надо описывать физическое положение и путь камеры (`sources/github/LoveRain1997__h3-prompt-journal.md`).
14. **Метки картинок.** Официально `<Picture N>`; в свободных облачных промптах — «Image 1», `@Image 1`
    (`sources/web/fal.ai__prompting-guide-44-examples.md`); узел Context-IR тоже ждёт в тексте «Image 1»…«Image 9»
    (`sources/github/Comfy-Org__docs__minimax-h3-node-reference.md`); в локальном примере BeatAPI внутри шести полей —
    `@Image1` (`sources/github/BeatAPI__awesome-minimax-h3-prompts.md`). Для локальных весов надёжность чужих меток не измерена.
15. **Разрешение/длительность.** README: по умолчанию короткая сторона 768, 2K — через Regenerate, 4–15 с
    (`sources/github/MiniMax-AI__MiniMax-H3__README.md`); Atlas Cloud на момент статьи: в облаке работал только 2K, T2V/I2V —
    5–10 с, R2V — 5–15 с (`sources/web/atlascloud__prompt-guide-45-official.md`). Для промпта важно: хронометраж в тексте
    подгонять под то, что реально примет конкретный бэкенд.

### Цифры, которые мы не проверяли

- 13/16 панелей против 3/16, CER 6,8 %, 437 с — один автор, одно зерно (`sources/web/ai-muninn__storyboard-grid-h3-ref2va-rtx5090.md`;
  `sources/web/ai-muninn__ref2va-two-characters-rtx5090.md`).
- Склейки 3,92/7,88 с против 3,54/6,79 с — один A/B поставщика (`sources/web/weshop__dialogue-and-timing.md`).
- ~83 % точности китайских реплик — тест поставщика (`sources/web/INDEX.md`, аннотация к atlascloud__chinese-dialogue-accuracy).
- 60–70 % успеха диалогов по методу 0,36 с/слово — один автор (`sources/github/MiniMax-AI__MiniMax-H3__issues.md`, #17).
- Метрики переписчиков и кодировщиков (Wuli, Alissonerdx, CondBridge, heretic) — заявления авторов
  (`sources/hf/cards/Wuli-art__Gemma4-for-MiniMax-H3-Prompt-Extend.md`; `sources/hf/cards/Alissonerdx__Qwen3.5-9B-Minimax-H3.md`;
  `sources/hf/cards/SpXMerlin1D__MiniMaxH3-CondBridge-Qwen3.5-4B.md`;
  `sources/hf/cards/LeViAtHaN2K__MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA-heretic.md`).

---

## 8. Что проверить на нашем порте (для итогового документа)

- A/B: короткие отрицательные утверждения против их удаления (раздел 7.1), с распознаванием речи.
- `<scenetrans>`/`<cutoff>`: где ставить тег относительно `<d>` — официального примера нет (раздел 3.3).
- Разделитель полей: пустая строка против одной строки (раздел 7.11).
- Повтор внешности: полное / укороченное / только метка при Ref2VA (раздел 7.4).
- Возврат говорящего A→B→A с двумя дорожками `fully_copy` против `reference` (раздел 7.8).
- Локальные переписчики (lightx2v 9B FL2VA/Ref2VA, Wuli Gemma4) против нашего агента с навыком — на одинаковых брифах и зерне.
