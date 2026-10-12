# CPU-бэкенд synaptix

Состояние, рецепты и что осталось. Крейт — `crates/synaptix-kernels-cpu`.

## Что есть (2026-10-11)

До этого дня CPU-бэкенд реализовывал 13 из 98 методов `Backend`; всё остальное шло разложенным путём
(`synaptix-ops`), а квантованные веса LLM на CPU не исполнялись вообще (`linear_quant` → Unsupported).
Теперь реализованы:

| Область | Файл | Методы |
|---|---|---|
| Общее | `common.rs`, `threads.rs`, `dispatch.rs` | SIMD-преобразования F16/BF16↔F32 (AVX2+F16C, NEON), `par_chunks`/`par_chunks_cost`, пул потоков (`SYNAPTIX_THREADS`), определение AVX-VNNI/dotprod/i8mm |
| Квант | `quant/dequant.rs`, `quant/ggml_simd.rs`, `quant/encode.rs`, `quant/gemv.rs` | `linear_quant` (SQ1…SQ8, NVFP4, MXFP8, все ggml), `quant_gemv_batched`, `block_dequant`, `block_gather_dequant`, `nvfp4_dequant`, `mxfp8_dequant`, энкодеры `quantize_sq/mxfp8/nvfp4` |
| Поэлементные | `elementwise.rs`, `reduction.rs`, `copy.rs` | unary/binary/cast/reduce/copy: contiguous и broadcast fast paths, SIMD, rayon |
| Нормы, RoPE, активации | `norms.rs`, `rope.rs`, `act.rs` | `rms_norm(_residual)`, `layer_norm`, `group_rms`, `group_norm`, `pixel_norm`, `rope_split(_partial)`, `rope_interleaved`, `rope_apply_dev`, `silu_and_mul`, `geglu_split`, `scale_act`, `ternary_fused`, `snake`, `linear_epilogue`, `hc_mix/inject`, `weighted_rows_sum`, `moe_combine` |
| Внимание | `attention.rs` | `flash_attention(_dev/_prefill_dev/_window/_window_dev/_bshd/_blocks)`, MXFP8-KV (`kv_append_quant_mxfp8(_dev)`, `flash_attention_mxfp8kv(_dev)`, `_blocks_mxfp8`) |
| Сборка данных, свёртки | `gather.rs`, `conv.rs` | `embed_gather(_mxfp8)`, `topk_rows`, `topk_wide`, `kv_append_dev`, `upsample_nearest2x`, `nchw↔nhwc`, `conv_epilogue`, `conv2d`, `conv3d`, `dwconv1d` |
| GEMM | `gemm/` | блочный f32 SGEMM (AVX2/NEON), `linear_panel` для квант-префилла, SIMD dot f32/f16/bf16 на x86 и NEON |

Не реализованы (остаются Unsupported, вызывающий код идёт разложенным путём или не нужен на CPU):
`linear_quant_prequant`, `nvfp4_quantize_act`, `mxfp8_quantize_act`, `silu_mul_quant_nvfp4`, `gelu_tanh_mul_quant_nvfp4`,
`rms_mod_quant_*`, `nvfp4_repack`, `nvfp4_gemm_grouped`, `nvfp4_gemv_*`, `quant_gemv_indexed(_acc)`, `quant_gemm_grouped`,
`quant_gemv_grouped`, семейство `dec_*` (слитые CUDA-graph ядра), `linear_attn_decode_step`, `linear_attn_chunk_prefill`,
`gated_delta_rule_prefill` (гибриды считают через host-путь), `conv2d_implicit_nhwc`.

## Политика точности на CPU
`optimal_profile_for_device(path, device)`: на CPU — счёт F32, веса SQ4 (внимание Gemma — SQ8), голова SQ8,
эмбеддинг F16, KV F16, без CUDA-графов и спекуляции. Уже квантованные источники (GGUF-блоки, SQ/NVFP4/MXFP8-бандлы)
исполняются как есть. Плотный `.syn` на CPU квантуется при первой загрузке через `transcode` с дисковым кэшем
(`Placement::Disk`, см. `synaptix_io::weights::transcode::cache_dir`), дальше читается готовым.
`PrecisionConfig::validate` допускает F32-счёт с квант-весами; `QLinear::forward` для F32 пробует `linear_quant` напрямую.

## Запуск и замеры
```sh
synaptix run model.gguf "…" --device cpu            # GGUF-блоки как есть
synaptix run model.syn  "…" --device cpu            # плотный бандл → SQ4 в кэше
SYNAPTIX_THREADS=8 synaptix run …                     # число потоков rayon
```
Ноутбук (Core Ultra 9 275HX, 24 потока), Qwen3-0.6B, 48 токенов:

| Источник | до | после |
|---|---|---|
| GGUF Q8_0 | не запускался (блочный gather) | декод 20,7 ток/с |
| GGUF BF16 → SQ4 при загрузке | не запускался | декод 14,3 ток/с |
| Whisper-tiny, 7,3 с речи | — | 0,84 с (RTF 0,12) |

Профиль (`perf record … --sort symbol`): на декоде 0.6B до трети времени — `rayon` steal/wait, то есть
слишком мелкие параллельные области; `par_chunks_cost` даёт каждой задаче ≥60 мкс работы.

## Префилл/кодировщики: GEMM и накладные расходы (2026-10-12)
Замер — OmniVoice (бэкбон Qwen3 0,6B, двунаправленный, 16 шагов × 2 прохода CFG по ~100 токенов), фраза 3,4 с,
`cargo run --profile fast-release -p synaptix-tts-omnivoice --example cpu_bench <omnivoice.syn> f32 16`:

| Шаг | F32 | F16 | int8 (SQ8) |
|---|---|---|---|
| было | 17,2 с | 16,1 с | 20,5 с |
| `gemm/xwt.rs`: Linear без транспонирования веса | 8,3 с | 8,5 с | — |
| `alloc_uninit` на CPU без обнуления | 6,1 с | | |
| батч-matmul параллельно по батчу + микроядро 6×16 | 4,4 с | 5,6 с | 8,1 с |
| префилл квант-весов через `xwt`, слитый `silu_and_mul` | 4,3 с | | 5,8 с |

Что было не так:
- `blocked::linear` на каждом вызове перекладывал **весь** вес `[N,K]` в панель `[K,NB]` скалярным циклом — треть
  времени синтеза. `xwt::linear` читает вес как есть: микроядро 4 строки `x` × 3 строки `w` (12 ymm-аккумуляторов),
  скалярные произведения по 8, горизонтальные суммы в конце, K кусками по 512 (строки `x` в L1, полоса весов в L2),
  тайлы задач 32×48. Вес F16/BF16 переводится в F32 при загрузке (F16C) — читается вдвое меньше памяти.
  `gemm_bench`: 300×1024×1024 — 0,41 → 1,35 TFLOPS (F32), 1,1–1,7 с весом F16; декод m=1 — в 35–60 раз быстрее.
- Микроядро SGEMM 8×16 требовало 19 ymm при 16 — компилятор выгружал аккумуляторы на каждой итерации K;
  6×16 (15 регистров) — упакованный путь 0,41 → 0,77–0,83 TFLOPS.
- `Backend::alloc_uninit` по умолчанию звал `alloc_zeros`: каждый выход каждой операции обнулялся `memset`
  в главном потоке, пока 23 потока ждали (больше половины времени главного потока). Теперь CPU отдаёт
  неинициализированную память, как CUDA; `SYNAPTIX_CPU_ZERO_ALLOC=1` — по-старому (найти ядро, читающее выход).
  Проверено побитовое совпадение выходов Qwen3-1.7B, Gemma-4 E2B, Whisper-tiny, VoxCPM2, OmniVoice.
- `matmul` F32 с батчем (головы внимания) шёл по батчу последовательно, а матрица 114×114 даёт 4 тайла —
  внимание занимало 2,1 с из 5,6. Теперь батч параллельно: 0,55 с.
- `Tensor::cat` на CPU: проверка `is_contiguous` на каждом куске, обнуление выхода, копия в одном потоке.
- OmniVoice: проходы cond/uncond на CPU — одним батчем (`Backbone::batch_cfg`, `SYNAPTIX_CFG_BATCH=0` — раздельно).

Осталось: int8 через AVX-VNNI (`vpdpbusd`) для префилла — вчетверо больше умножений за такт, чем F32 FMA;
GQA во внимании без `repeat_kv`; перестановки осей q/k/v после проекций (~0,3 с на прогон OmniVoice).

## Тесты
```sh
cargo test --profile fast-release -p synaptix-kernels-cpu      # ref_* требуют tests/reference_data (нет в репозитории)
cargo check -p synaptix-kernels-cpu --target aarch64-unknown-linux-gnu
```
Каждый модуль сравнивается с наивным эталоном в своих `#[cfg(test)]`; на хосте есть `qemu-aarch64` — NEON-ветки
можно проверить, собрав тесты под `aarch64-unknown-linux-gnu` (sysroot телефона, `scratchpad/cross.sh`).

## Что осталось
1. Int8-путь GEMV/GEMM: активации в Q8-блоках по 32 с f32-шкалой, веса SQ/Q4/Q8 как целые, скалярное
   произведение через AVX-VNNI (`vpdpbusd`) на Intel и `sdot`/`smmla` (dotprod/i8mm) на ARM — это «ускорение Intel»
   и главный выигрыш на телефоне (Cortex-X2/A710 умеют i8mm). Флаги уже определяются в `dispatch.rs`.
2. Блочный GEMM F16/BF16 на префилле (сейчас построчный dot) и общий batched matmul без strided-чтения колонок.
3. Накладные расходы rayon на декоде: меньше параллельных областей на шаг (слить нормы с проекциями, не
   параллелить операции меньше ~50 мкс).
4. Кэш квантования для GGUF с плотными типами (BF16/F16) — сейчас квантуется при каждой загрузке.
5. Проверка NEON-веток на телефоне/qemu, замеры на SM8475.
6. Intel iGPU (OpenCL/Level Zero есть в системе): Xe-LPG Arrow Lake по пропускной способности сравним с
   24 потоками AVX2, отдельный бэкенд не оправдан — не планируется.

## int8 по каналу (q8row) на CPU
Веса `DType::Q8Row` (i8 `[N,K]` + шкала строки, см. docs/npu.md) исполняются без распаковки:
- декод (m < 8): активация квантуется на лету в int8 блоками по 32 (как Q8_0 у llama.cpp), свёртка — `sdot`
  (ARM dotprod, через `asm!`: `vdotq_s32` в стабильном Rust ещё нестабилен) или AVX2 `maddubs`; итог × шкала блока ×
  шкала строки (`quant/q8dot.rs`);
- префилл (m ≥ 8) — тоже int8: активации квантуются построчно, микроядро 4 строки активаций × 1 строка весов
  (`q8dot::dot4`, загрузка весов делится на четыре `sdot`), веса режутся по потокам кусками ≤ 128 строк. Раньше
  строки весов распаковывались в f32 под панельный GEMM: префилл Gemma-4 E2B (~400 токенов) на телефоне
  20,3 → 8,3 с, на ноутбуке 9,9 → 5,6 с (до первого токена);
- `QuantWeight::embed_gather` для q8row — выборка и деквант строк таблицы (эмбеддинги Gemma-4 в int8);
- внимание на декоде — CPU-ядро по KV-буферу без `repeat_kv` (`flash_attention_dev`/`window_dev`): 1,97 → 0,23 мс на
  слой у Gemma-4 E2B на телефоне.

Потоки по умолчанию — только быстрые ядра (частота ≥ 70 % максимальной, `/sys/…/cpuinfo_max_freq`): на Snapdragon 8+
Gen 1 это 4 из 8, малые A510 тормозили общий проход (8 потоков — 4,3 ток/с, 4 — 6,7). `SYNAPTIX_THREADS` — вручную.

Gemma-4 q8row (`synaptix quantize … --format q8row --include model.language_model. --embed q8row`), Xiaomi 12T Pro,
CPU: E2B — ~6,9 ток/с, E4B — ~3,1 ток/с; ноутбук (x86, AVX2): E2B — 9 ток/с. Декод упирается в память: на токен
читается ~2 ГБ int8 (E2B), голова на 262 тыс. слов — 400 МБ.

В чате префилл истории не повторяется: synaptixd держит `LlmKvSession` на модель и дописывает только новый хвост
промпта (`generate_streaming_cached`) — следующий ход наставника synspeak на телефоне начинает отвечать через
0,6–1 с вместо 8–13 с.
