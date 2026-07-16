# FlowMatchEulerDiscreteScheduler (rectified flow, dynamic shifting) — FLUX.1-dev


# Спецификация: FlowMatchEulerDiscreteScheduler (FLUX.1-dev)

Bit-exact порт rectified-flow Euler-планировщика с dynamic-shifting. Источник:
`diffusers/schedulers/scheduling_flow_match_euler_discrete.py` + связка из
`diffusers/pipelines/flux/pipeline_flux.py`. Это НЕ обучаемый модуль — весов нет,
только числовая логика. Все вычисления в **f64/f32** (см. ГОЧИ по точности), тензоров
с весами в state_dict НЕТ.

---

## 0. Конфиг FLUX.1-dev (scheduler_config.json)

```
num_train_timesteps   = 1000      (целое)
shift                 = 3.0       (не используется при dynamic — см. §3)
use_dynamic_shifting  = true      → активна ветка time_shift(mu, 1.0, sigma)
base_shift            = 0.5
max_shift             = 1.15
base_image_seq_len    = 256
max_image_seq_len     = 4096
```
Не заданы в конфиге → дефолты класса:
```
invert_sigmas          = false
shift_terminal         = None     (falsy → шаг stretch НЕ выполняется)
use_karras_sigmas      = false
use_exponential_sigmas = false
use_beta_sigmas        = false
time_shift_type        = "exponential"
stochastic_sampling    = false
order                  = 1
```
Для FLUX.1-dev все «sigma-conversion» ветки (karras/exponential/beta), stretch_terminal,
invert, stochastic — **ОТКЛЮЧЕНЫ**. Их можно НЕ реализовывать для bit-exact dev-пути,
но логику ниже привожу для полноты.

---

## 1. Состояние объекта (поля)

После конструктора и после `set_timesteps` объект хранит:
```
config.*            (значения из §0, неизменны)
_shift              = 3.0                     (mutable через set_shift, но не трогается)
num_inference_steps : int                     (устанавливается в set_timesteps)
timesteps : Tensor[f32]                        (см. §4; длина = num_steps)
sigmas    : Tensor[f32]                        (см. §4; длина = num_steps + 1, последний = 0.0)
sigma_min, sigma_max : float                   (из конструктора, см. §2)
_step_index  : Optional[int]                   (None до первого step / после set_timesteps)
_begin_index : Optional[int]                   (pipeline ставит = 0 перед циклом)
```

### Конструктор (__init__) — вычисляет sigma_min/sigma_max
```
timesteps = linspace(1, 1000, 1000, f32)        # [1, 2, ..., 1000]
timesteps = reverse(timesteps)                  # [1000, 999, ..., 1]  (np [::-1])
sigmas = timesteps / 1000                       # [1.0, 0.999, ..., 0.001]
# use_dynamic_shifting=True → ветку "shift*sigmas/(1+(shift-1)*sigmas)" ПРОПУСКАЕМ
self.timesteps = sigmas * 1000                  # = исходный timesteps (не используется далее)
sigma_min = sigmas[-1] = 0.001                  # = 1/1000
sigma_max = sigmas[0]  = 1.0
```
Важно: для FLUX `sigma_max = 1.0`, `sigma_min = 0.001` (это 1.0 и 1/num_train_timesteps).
Эти значения нужны как границы default-linspace в `set_timesteps` (§3, шаг A).

---

## 2. calculate_shift(image_seq_len) — в pipeline, НЕ в scheduler

Источник: `pipeline_flux.py:74-84`. Линейная интерполяция mu по длине latent-последовательности.

```
m  = (max_shift - base_shift) / (max_seq_len - base_seq_len)
   = (1.15 - 0.5) / (4096 - 256)
   = 0.65 / 3840
   = 0.00016927083333...                        # точное f64
b  = base_shift - m * base_seq_len
   = 0.5 - m * 256
   = 0.5 - 0.043333... = 0.456666...
mu = image_seq_len * m + b
```
Где `image_seq_len = latents.shape[1]` = число latent-токенов (после pack 2x2).
Для FLUX latent-токены = (H/16)*(W/16). Пример 1024x1024 → latent 128x128 → pack →
64x64 = 4096 токенов → mu = 4096*m + b = 0.65 + 0.456666... = ~1.1006... (sanity: при
image_seq_len = max_seq_len(4096) mu = max_shift; при = base_seq_len(256) mu = base_shift=0.5).

ПОРЯДОК ВЫЧИСЛЕНИЯ важен для bit-exact: сначала `m`, затем `b = base_shift - m*base_seq_len`,
затем `mu = image_seq_len*m + b`. НЕ упрощать алгебраически (иначе разойдётся в последнем бите).

`mu` — скаляр f64 (Python float). Передаётся в `set_timesteps(mu=mu)`.

---

## 3. set_timesteps(num_inference_steps, mu) — построение sigmas/timesteps

В pipeline вызывается через `retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)`,
где **`sigmas` ПЕРЕДАЁТСЯ ЯВНО** из pipeline (строка 868):
```
sigmas = np.linspace(1.0, 1/num_inference_steps, num_inference_steps)   # pipeline, dtype f64
```
(ветка `use_flow_sigmas` для FLUX отсутствует в конфиге → sigmas НЕ обнуляется).
Поскольку `sigmas is not None`, внутри `set_timesteps` срабатывает ветка `else` для sigmas
(строки 341-343), а НЕ default-linspace в t-домене.

Точная последовательность внутри `set_timesteps` для FLUX-dev (sigmas задан, mu задан,
timesteps=None, num_inference_steps=N):

### Шаг 0. Валидация
`use_dynamic_shifting=True` и `mu` не None — OK. `num_inference_steps = N`.
`is_timesteps_provided = False`.

### Шаг 1. Базовые sigmas
```
sigmas = np.array(sigmas).astype(f32)     # из pipeline: linspace(1.0, 1/N, N), приведён к f32
num_inference_steps = len(sigmas) = N
```
ВНИМАНИЕ: pipeline считает linspace в **f64**, затем `set_timesteps` кастует в **f32**.
Реализуй так же: linspace в f64, потом truncate в f32. Массив (длина N):
```
sigmas[i] = 1.0 + i * ( (1/N - 1.0) / (N-1) )   для i = 0..N-1
sigmas[0]   = 1.0
sigmas[N-1] = 1/N
```
(np.linspace endpoint=True: первый ровно 1.0, последний ровно 1/N).

### Шаг 2. Динамический сдвиг (time_shift, exponential)
`use_dynamic_shifting=True` → применяем поэлементно (строка 348):
```
sigmas = time_shift(mu, 1.0, sigmas)
```
где (time_shift_type="exponential", _time_shift_exponential, строка 649):
```
time_shift(mu, sigma=1.0, t) = exp(mu) / ( exp(mu) + (1/t - 1)**1.0 )
                             = exp(mu) / ( exp(mu) + (1/t - 1) )
```
Поэлементно по каждому t = sigmas[i]. `**1.0` — степень 1, можно опустить, но НЕ
заменять алгебраически (для bit-exact вычисляй `exp(mu)` один раз, затем на каждый элемент
`em / (em + (1.0/t - 1.0))`). `exp` от скаляра mu — в f64 (math.exp), затем арифметика —
тензор f32 (numpy broadcast f32 после .astype(f32)). На практике: `em = exp(mu)` (f64
скаляр), операции `1/t`, `-1`, деление идут в dtype массива (f32). Воспроизводи: em как
f64-константа, тензорная арифметика в f32.

После этого шага монотонность сохраняется: sigmas убывают от ~time_shift(1.0)→близко к 1,
до time_shift(1/N)→малое значение.

### Шаг 3. shift_terminal
`config.shift_terminal = None` (falsy) → ПРОПУСК (stretch_shift_to_terminal НЕ вызывается).

### Шаг 4. karras/exponential/beta
Все три флага False → ПРОПУСК.

### Шаг 5. timesteps из sigmas
```
sigmas = torch.from_numpy(sigmas).to(f32, device)
# is_timesteps_provided == False:
timesteps = sigmas * num_train_timesteps = sigmas * 1000      # f32, длина N
```
`timesteps[i] = sigmas[i] * 1000`. Это финальный массив `self.timesteps` (длина N, БЕЗ
терминального нуля).

### Шаг 6. Append терминальной sigma
`invert_sigmas=False` → ветка else (строка 379):
```
sigmas = cat([sigmas, zeros(1)])      # длина N+1, последний элемент = 0.0
```
`timesteps` НЕ получает терминальный элемент (остаётся длины N).

### Финал
```
self.timesteps   = timesteps   (f32, длина N)         # убывающий, timesteps[0] ~= time_shift(1.0)*1000
self.sigmas      = sigmas       (f32, длина N+1)        # sigmas[N] = 0.0
self._step_index  = None
self._begin_index = None
```

Затем pipeline вызывает `scheduler.set_begin_index(0)` → `_begin_index = 0`.

#### Итоговые массивы (для N шагов), порядок УБЫВАЮЩИЙ:
```
sigmas    = [ s_0, s_1, ..., s_{N-1}, 0.0 ]         # len N+1, s_0 > s_1 > ... > s_{N-1} > 0
timesteps = [ s_0*1000, s_1*1000, ..., s_{N-1}*1000 ] # len N
```
где `s_i = time_shift(mu, 1.0, lin_i)`, `lin_i = 1.0 + i*((1/N - 1)/(N-1))`.

---

## 4. step(model_output, timestep, sample) — Euler-шаг flow-matching

Источник: строки 425-524. Для FLUX-dev `stochastic_sampling=False`,
`per_token_timesteps=None`. Логика:

### 4.1 Инициализация step_index (первый вызов)
Pipeline уже выставил `_begin_index = 0`. На первом `step`:
```
if _step_index is None:
    _init_step_index(timestep):
        if begin_index is not None:           # = 0
            _step_index = _begin_index = 0
        else: _step_index = index_for_timestep(timestep)   # (НЕ путь FLUX-dev)
```
Т.е. для FLUX благодаря `set_begin_index(0)` step_index стартует с 0 и просто
инкрементируется. `index_for_timestep` (поиск по равенству) в проде НЕ задействован.
(Для надёжности: `index_for_timestep` ищет `(schedule_timesteps == timestep).nonzero()`,
берёт `pos=1 if len>1 else 0` — второй матч или единственный; но при begin_index=0 не нужен.)

### 4.2 Upcast sample → f32
```
sample = sample.to(f32)           # ГОЧА: вход model_output остаётся в своём dtype (bf16),
                                  # sample апкастится в f32 ДО арифметики
```

### 4.3 Выбор sigma / sigma_next
```
i          = step_index
sigma      = sigmas[i]            # current_sigma
sigma_next = sigmas[i+1]          # next_sigma  (для последнего шага i=N-1: sigmas[N]=0.0)
dt         = sigma_next - sigma   # ОТРИЦАТЕЛЬНОЕ (sigma убывает)
```

### 4.4 Euler-обновление (детерминированная ветка)
```
prev_sample = sample + dt * model_output
            = sample + (sigma_next - sigma) * model_output
```
ВАЖНО:
- НЕТ деления на что-либо; НЕТ масштабирования входа (scale_model_input = identity,
  её в этом классе вообще нет → latents подаются в трансформер как есть).
- `model_output` интерпретируется как velocity (предсказание скорости потока v = dx/dt).
- Порядок: `dt * model_output` (тензор bf16*f32 → f32 по промоушену), затем `sample(f32) + ...`.
  Воспроизводи: dt — скаляр f32 (элемент sigmas), умножение на model_output, сложение с
  sample в f32.

### 4.5 Инкремент и каст обратно
```
_step_index += 1
# per_token_timesteps is None:
prev_sample = prev_sample.to(model_output.dtype)   # обратно в bf16 (dtype model_output)
return prev_sample
```
ГОЧА: результат кастуется обратно в dtype **model_output** (не sample-исходного). В FLUX
оба bf16, так что prev_sample выходит bf16. Внутренняя арифметика — f32, финальный каст — bf16.

#### Псевдокод полного цикла денойза (для понимания связки):
```
sigmas, timesteps = set_timesteps(N, mu)     # §3
set_begin_index(0)
latents = init_noise                          # bf16, форма [B, image_seq_len, C_packed]
for i, t in enumerate(timesteps):             # t = timesteps[i], скаляр f32
    timestep = t.expand(B).to(latents.dtype)  # [B], bf16  (значение = sigmas[i]*1000)
    v = transformer(latents, timestep = timestep/1000, ...)   # см. §5
    latents = scheduler.step(v, t, latents).prev_sample       # §4
```

---

## 5. Связка pipeline.timesteps → transformer (КРИТИЧНО, bit-exact)

Это часть, где чаще всего ошибаются. Точная цепочка:

1. `scheduler.timesteps[i]` = `sigmas[i] * 1000` (диапазон ~ (0, 1000), убывает).
2. В цикле: `timestep = t.expand(B).to(latents.dtype)` → тензор [B] (bf16), значение = `sigmas[i]*1000`.
3. В трансформер передаётся **`timestep / 1000`** (pipeline_flux.py:951):
   ```
   transformer(..., timestep = timestep / 1000, ...)
   ```
   → то есть в трансформер приходит ЗНАЧЕНИЕ ≈ `sigmas[i]` (доля 0..1), а НЕ t*1000.
   Деление на 1000 делает PIPELINE, не scheduler и не transformer.

4. Внутри трансформера (`CombinedTimestepGuidanceTextProjEmbeddings` / для FLUX-dev — с guidance):
   `time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1)`
   (embeddings.py:1588 для CombinedTimestepTextProjEmbeddings; guidance-вариант FLUX —
   аналогичный, downscale_freq_shift=0). Внутри `time_proj` вызывается `get_timestep_embedding`
   с `scale=1` → **НИКАКОГО умножения на 1000 обратно нет**. Sinusoidal-эмбеддинг строится
   напрямую от значения ≈ sigma ∈ (0,1):
   ```
   get_timestep_embedding(timesteps≈sigma, dim=256, flip_sin_to_cos=True,
                          downscale_freq_shift=0, scale=1, max_period=10000)
   ```
   ГОЧА: важно, что `Timesteps` для FLUX (`downscale_freq_shift=0`), а НЕ
   `CombinedTimestepLabelEmbeddings` (`downscale_freq_shift=1`). И `scale=1` — вход в синусоиду
   не масштабируется; используется именно дробное sigma∈(0,1), а не t∈(0,1000).

ИТОГ для loader/forward: то, что трансформер видит как «timestep», есть **sigma_i = sigmas[i]**
(доля, эквивалентна `timesteps[i]/1000`). Реализуя scheduler отдельно, отдавай наружу массив
`timesteps[i] = sigmas[i]*1000`; деление обратно на 1000 — обязанность вызывающего цикла
(как в pipeline). Не зашивай /1000 в scheduler.

---

## 6. scale_noise (forward-процесс, для img2img/инициализации) — для полноты

Источник: строки 187-235. В чистом txt2img FLUX-dev НЕ нужен (init = чистый шум). Формула
линейной интерполяции flow-matching:
```
sigma подбирается по step_index/begin_index/index_for_timestep
sample = sigma * noise + (1.0 - sigma) * sample
```
(sigma broadcast до ранга sample через unsqueeze(-1)). Реализовать только если поддерживаешь img2img.

---

## 7. Формы тензоров (txt2img, B = batch)

```
latents (sample)     : [B, image_seq_len, C]   bf16   (C = in_channels = 64 packed, image_seq_len=(H/16)*(W/16))
model_output (v)     : [B, image_seq_len, C]   bf16
timesteps (scheduler): [N]                       f32
sigmas (scheduler)   : [N+1]                     f32
t (один шаг)         : скаляр f32 → expand → [B] bf16
mu                   : скаляр f64
prev_sample (выход)  : [B, image_seq_len, C]   bf16
```
Scheduler-арифметика покомпонентна по последним осям; sigma/dt — скаляры, broadcast по [B,seq,C].

---

## 8. ГОЧИ bit-exact (чек-лист)

1. **Деление /1000 — в pipeline, не в scheduler/transformer.** Scheduler отдаёт
   `timesteps = sigmas*1000`; цикл делит обратно перед трансформером. transformer.time_proj
   `scale=1`, `downscale_freq_shift=0` — обратного умножения нет.
2. **sigmas базовый = linspace(1.0, 1/N, N) в f64** (считается в PIPELINE, строка 868),
   потом каст в f32 внутри set_timesteps. НЕ default-linspace в t-домене (та ветка не
   срабатывает, т.к. sigmas передан явно).
3. **time_shift exponential**: `exp(mu)/(exp(mu) + (1/t - 1))`. `exp(mu)` от f64-скаляра
   (math.exp), `(1/t - 1)` поэлементно в f32. `**1.0` опускаемо. Считать `em=exp(mu)` один раз.
4. **calculate_shift порядок**: m → b=base_shift−m·base_seq → mu=seq·m+b. Не упрощать.
5. **dt = sigma_next − sigma ОТРИЦАТЕЛЬНО**; `prev = sample + dt*v` (плюс, не минус).
   Никакого деления, никакого scale_model_input.
6. **Upcast в f32**: `sample.to(f32)` ДО Euler-арифметики; финальный `prev.to(model_output.dtype)`
   (bf16). Внутренняя точность — f32.
7. **sigmas длиной N+1** (терминальный 0.0 добавлен), **timesteps длиной N** (без нуля).
   Последний шаг i=N-1 использует sigma_next = sigmas[N] = 0.0 → выводит латент при sigma=0
   (чистый сигнал).
8. **begin_index=0** ставится pipeline → step_index просто инкрементируется 0,1,...,N-1;
   index_for_timestep (поиск по равенству t) в проде не задействован.
9. **shift=3.0 из конфига НЕ используется** при use_dynamic_shifting=True (ни в конструкторе
   sigmas, ни в set_timesteps; вместо него — time_shift(mu)). Не применяй формулу
   `shift*s/(1+(shift-1)*s)`.
10. **shift_terminal/karras/exponential/beta/invert/stochastic — все OFF** для FLUX-dev.
11. **mu — f64 скаляр**; `exp(mu)` в f64. Расхождение в mu в 1e-7 даёт видимый дрейф к
    концу 50 шагов — держи f64 в calculate_shift и в exp.
12. **Порядок sigmas — строго убывающий** (от ~1 к ~0). Если реализуешь linspace вручную:
    sigmas[0]=1.0 точно, sigmas[N-1]=1/N точно (endpoint inclusive).


## WEIGHT KEYS
У FlowMatchEulerDiscreteScheduler НЕТ обучаемых весов и НЕТ ключей в HF state_dict — это чисто числовой планировщик. Конфигурация грузится из `scheduler/scheduler_config.json` (не safetensors). Поля конфига: `num_train_timesteps=1000`, `shift=3.0`, `use_dynamic_shifting=true`, `base_shift=0.5`, `max_shift=1.15`, `base_image_seq_len=256`, `max_image_seq_len=4096`. Дефолты класса (отсутствуют в json, брать из __init__): `invert_sigmas=false`, `shift_terminal=None`, `use_karras_sigmas=false`, `use_exponential_sigmas=false`, `use_beta_sigmas=false`, `time_shift_type="exponential"`, `stochastic_sampling=false`. Loader должен лишь распарсить JSON в config-структуру; буферы sigmas/timesteps вычисляются в set_timesteps на лету.

## GOTCHAS
КРИТИЧЕСКИЕ подводные камни (bit-exact):

1. Деление timestep/1000 делает PIPELINE (pipeline_flux.py:951), НЕ scheduler и НЕ transformer. Scheduler отдаёт timesteps[i]=sigmas[i]*1000; цикл делит обратно перед вызовом трансформера, в трансформер приходит значение ≈ sigma_i (доля 0..1). transformer.time_proj = Timesteps(scale=1, downscale_freq_shift=0) — обратного умножения на 1000 нет (это НЕ CombinedTimestepLabelEmbeddings c downscale_freq_shift=1). Не зашивать /1000 в scheduler.

2. Базовый sigmas = np.linspace(1.0, 1/N, N) считается в pipeline в f64 (строка 868), затем кастуется в f32 внутри set_timesteps. Поскольку sigmas передаётся явно через retrieve_timesteps(sigmas=...), внутренняя default-ветка linspace в t-домене (_sigma_to_t) НЕ срабатывает.

3. time_shift (exponential): exp(mu)/(exp(mu)+(1/t-1)**1.0). exp(mu) от f64-скаляра (math.exp), вычислять ОДИН раз; (1/t-1) поэлементно в f32. **1.0 опускаемо но не упрощать алгебраически.

4. calculate_shift (pipeline): m=(max_shift-base_shift)/(max_seq-base_seq); b=base_shift-m*base_seq; mu=image_seq_len*m+b. Порядок не нарушать, держать f64 — расхождение mu в 1e-7 даёт дрейф за 50 шагов.

5. Euler step: dt = sigma_next - sigma (ОТРИЦАТЕЛЬНО, sigma убывает); prev_sample = sample + dt*model_output (ПЛЮС). НЕТ деления, НЕТ scale_model_input (её в классе вообще нет — latents подаются как есть). model_output = velocity.

6. Upcast: sample.to(f32) ДО арифметики; результат prev_sample.to(model_output.dtype)=bf16 (dtype model_output, не sample-исходного). Внутренние вычисления f32.

7. sigmas длиной N+1 (терминальный 0.0 добавлен в конце через cat([sigmas, zeros(1)])); timesteps длиной N (без нуля). Последний шаг i=N-1 берёт sigma_next=sigmas[N]=0.0.

8. set_begin_index(0) вызывается pipeline → step_index инкрементируется 0..N-1; index_for_timestep (поиск по равенству) в проде FLUX не нужен. _step_index/_begin_index сбрасываются в None внутри set_timesteps, потом pipeline ставит begin_index=0.

9. shift=3.0 из конфига НЕ используется при use_dynamic_shifting=true — ни формула shift*s/(1+(shift-1)*s) в конструкторе, ни в set_timesteps. Вместо неё time_shift(mu).

10. Для FLUX-dev ОТКЛЮЧЕНЫ: shift_terminal(None→falsy), use_karras/exponential/beta_sigmas, invert_sigmas, stochastic_sampling, per_token_timesteps. Эти ветки можно не реализовывать.

11. Порядок массивов строго убывающий (sigma от ~1 к 0). linspace endpoint inclusive: sigmas[0]=1.0 ровно, sigmas[N-1]=1/N ровно.
