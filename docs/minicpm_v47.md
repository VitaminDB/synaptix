# MiniCPM-V 4.7 (OpenBMB, октябрь 2026)

Две модели семейства: **35B-A3B** (MoE: 256 экспертов, 8 активных, 40 слоёв,
~70 ГБ bf16) и **1B** (плотная, 1.55B, 3.1 ГБ bf16). Обе —
`model_type = minicpmv4_7`, контекст 262 144, Apache-2.0, HF:
`openbmb/MiniCPM-V-4.7-35B-A3B`, `openbmb/MiniCPM-V-4.7-1B`.

## Как устроена

- **Текст** — гибрид Qwen3.5 (`text_config.model_type` `qwen3_5_moe_text` /
  `qwen3_5_text`): GatedDeltaNet + полное внимание каждый 4-й слой,
  `attn_output_gate`, partial RoPE 0.25, M-RoPE `[11, 11, 10]` с интерливингом.
  Исполняется конвейером `synaptix-llm-qwen3-next-hybrid`; у 35B вместо
  плотного MLP — Qwen-MoE (`mlp.gate` роутер, стопки
  `mlp.experts.{gate_up_proj,down_proj}`, `shared_expert` с sigmoid-гейтом) —
  общий декодер получил `Ffn::Moe` (`synaptix_llm_common::model::Ffn`,
  спецификация `DecoderExt::moe_ffn`). Веса MTP в чекпоинте отсутствуют.
- **Зрение** — крейт `synaptix-vlm-minicpm` (`crates/synaptix-models/vlm/minicpm_v`):
  SigLIP 27 слоёв (1152, патч 14, NaViT без фиксированного размера, позиции
  по ближайшему соседу из таблицы 70×70), после 7-го слоя — оконный
  объединитель 2×2 (`vit_merger`: внимание внутри окна + MLP 4D→D с
  остатком-средним), `post_layernorm`, затем `model.merger.mlp.0` (ещё 2×2 →
  LLM-размерность, GELU точный). Итого 16 патчей → 1 токен.
- **Препроцессинг** — как `MiniCPMV4_6ImageProcessor`: миниатюра
  (`find_best_resize`, масштаб 448, стороны кратны 56) и до 9 срезов
  (`get_sliced_grid` по log-ratio), бикубический ресайз с антиалиасингом,
  нормализация `(x − 0.5)/0.5`.
- **Промпт** — `<image_id>N</image_id><image>…</image>` и строки
  `<slice>…</slice>` через `\n`; номер картинки проставляется при рендере
  chat-шаблона (`facade::llm::number_image_ids`, метка
  `MINICPM_IMAGE_ID_MARK` в `prompt_block`).
- **Позиции** — «холстовый» M-RoPE (`mrope_mode = canvas`): миниатюра
  растягивается на холст срезов, `<image>`/`</image>`/`<slice>`/`</slice>`/`\n`
  прибиты к углам (порт `get_rope_index` — `synaptix_vlm_minicpm::layout::canvas_positions`;
  id токенов — `HybridConfig::canvas`).

## Сверка с transformers 5.19 (1B, CPU f32 против synaptix bf16, greedy)

- Текст: первые 32 токена ответа совпадают бит в бит.
- Картинка: см. ниже (обновляется).

## Файлы

- `~/Storage/Нейросети/Модели/llm/minicpm-v-4.7-1b.syn` — плотный bf16
  (`syn-pack <hf-dir> -o … --id minicpm-v-4.7-1b`); модель < 3 ГБ по политике
  не квантуется, служба берёт профиль «optimal» на лету.
- `~/Storage/Нейросети/Модели/llm/minicpm-v-4.7-35b-a3b.syn` — политика
  `llm`: эксперты и MLP NVFP4, внимание/эмбеддинги/голова/крайние слои и башня
  зрения MXFP8 (`synaptix quantize --policy llm --vision mxfp8 --rule model.merger.=mxfp8`).
- Телефон: 1B в q8row (`synaptix quantize --format q8row --device cpu`),
  каталог `llm/minicpm-v-4.7-1b/` с `model.toml` (`precision = "w8"`, `devices = ["cpu"]`).
