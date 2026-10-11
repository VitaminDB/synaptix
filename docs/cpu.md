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
