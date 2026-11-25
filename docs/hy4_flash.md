# Hy4-flash (tencent, `hy_v4_flash`)

220 млрд параметров (63,5 млрд — engram), 12 млрд активных на токен, контекст 1M. Оригинал BF16 —
`~/Storage/Нейросети/Модели/_hf/Hy4-flash` (148 шардов, 440 ГБ; не удалять). Крейт —
`crates/synaptix-models/llm/hy_v4_flash`, ядра — `synaptix-kernels-cuda/src/cu/fused/hy4.cu`.

Эталоны: transformers PR #50903 (`modeling_hy_v4_flash.py`) и vLLM (`vllm/models/hy_v4_flash`,
`vllm/models/hy_v4`). Где они расходятся (engram), движок следует vLLM.

## Архитектура (text_config)

- 42 слоя, hidden 3072, словарь 120832, RMS eps 1e-5. Слой 0 — плотный MLP (13824), 1–41 — MoE.
- **iHC** (identity Hyper-Connections, 4 потока `[T, 4, H]`): `mixes = flat(X)·fnᵀ · rsqrt(mean(X²)+eps)`;
  `pre = σ(mix·s0 + b) + 1e-6`, `post = 2·σ(mix'·s1 + b') + 1e-6`; вход подслоя `Σ pre_i·X_i`,
  возврат `X_i += post_i · y`. Голова: те же 4 `pre`, затем `model.norm` и `lm_head`.
  Веса: `hc_attn_layer.hc_pre.hc_{fn,base,scale}`, `hc_mlp_layer…`, `hc_head.hc_head_*` (F32).
- **Внимание** — MLA с гейтом и синками: `q = q_b(rms(q_a(x)))` (q_lora 1024), `ckv = kv_a(x)` →
  латент 512 (`kv_a_layernorm`) + `k_rot` 64; `kv_b` → `k_nope` 192 и `v` 256 на голову; RoPE —
  чередующиеся пары на последних 64 каналах q/k; масштаб `1/16` (qk 256); синк на голову —
  лишний логит без value; выход `·σ(linear_gate(x))`, `o_proj`, затем `post_proj_norm`.
  - `sliding_attention` (30 слоёв): 64 головы, окно 512 (своя позиция и 511 прошлых), θ = 1e4.
  - `deepseek_sparse_attention` (12 слоёв: 0, 1, 5, 9, …, 41): 32 головы, θ = 1e7, сжатый индексатор:
    ключ токена `rms(token_key_projection(x))` (128), группы по 4 токена свёртываются
    (`softmax(Wg·k + ape)` по слотам × `Wv·k`, `key_norm`, RoPE на позиции первого токена группы),
    запрос `query_projection(q_resid)` 16×128 с RoPE, оценка `Σ_h w_h·relu(q_h·K_g)`
    (`w = head_weight_projection(x)·16^-½·128^-½`), выбор top-511 закрытых групп раньше своей
    плюс своя. Пока прошлых групп ≤ 511 — это плотное каузальное внимание.
- **MoE** латентный: роутер на полном x (σ, поправка выбора `e_score_correction_bias`, top-16 из 512,
  нормировка, ×3,998); эксперты в ширине 1536: `routed_expert_down_proj` (3072→1536) → эксперты
  (`silu(min(g,10))·clamp(u,±10)`) → `routed_expert_up_proj` (1536→3072); общий эксперт (1536,
  без ограничения) на полном x; сумма → `combine_output_norm`. У плотного слоя — `post_fc2_norm`.
- **Engram** (слои 1, 11, 21, 31 — перед вниманием): id → `lookup_table` (карта нормализованного
  словаря, I32), 2- и 3-граммы `[t, t−1, t−2]` (за началом — заполнитель), `rolling ^= id_s·mult_s`,
  по 8 голов на порядок: строка `rolling mod prime + offset`. Таблица `[164994380, 96]` на слой.
  Строки норм. по головам (eps F32), `value_proj`/`key_projs.c` (1536→3072, со смещением),
  гейт потока `σ(sign·√max(|⟨rms(k_c)·kn_c, rms(X_c)·qn_c⟩/√H|, 1e-6))`,
  `X_c += (gate + hc_scale_c)·pre_gamma·rms(value)`. Заполнитель (`engram_pad_id`) памяти не получает.
  Множители хэша эталон тянет из `numpy.random.default_rng(seed + 10007·id)` с границей по
  `max(lookup_table)+1` — посчитаны один раз и лежат в `hy4_engram.json` (кладётся в каталог HF
  до упаковки и попадает в бандл).
- DFly2 (`model.dfly_layers.*`, 5 слоёв драфтера для спекуляции) и башня зрения `vit.*` в бандле
  есть, движком пока не исполняются.

Расхождения эталонов: у transformers `nn.RMSNorm` engram без eps → eps bf16 = 2⁻⁷ (заметно на строках
таблицы ~1e-2); vLLM — eps F32. vLLM также отображает id вне карты в себя (transformers — в заполнитель).

## Бандл

```sh
cp scratchpad/hy4_engram.json _hf/Hy4-flash/   # множители хэша engram
synaptix quantize _hf/Hy4-flash llm/hy4-flash.syn --policy llm \
  --rule compressed_indexer.=none --rule vit.=none
```

Эксперты NVFP4 (~85 ГБ), внимание, общие эксперты, крайние слои, эмбеддинг, голова и engram-таблицы —
MXFP8; индексатор (F32-проекции, позиционная таблица) и зрение — как есть. Engram-таблица 31,7 ГБ
одним тензором кодируется кусками строк (`SYN_QUANT_ROW_CHUNK`).

## Движок

- Потоки iHC — F32; проекции — BF16 на квант-весах; эксперты — F16 через общий `MoeFfn`
  (`MoeExt::router_hidden`, `Activation::SiluClamp`), лениво из бандла с кэшем на карте
  (`SYN_HY4_EXPERT_CACHE_GB`, 8) и зеркалом в RAM (сколько влезает).
- Внимание — поглощённая форма по латентному кэшу (`q_lat = q_nope·W_uk`, `v = o_lat·W_uvᵀ`, W из
  `kv_b_proj` в BF16): ключей на запрос не больше 2048, ядро `hy4_mla_attn` держит оценки в shared.
  Кэш индексированных слоёв — F32 (`HY4_KV=bf16`), у SWA — кольцо окно + чанк префилла.
- Engram: хэш и строки таблицы — на хосте (строки MXFP8 читаются из блобов бандла), гейт — на карте.

Проверка: `cargo run --example hy4_run -- llm/hy4-flash.syn --chat "…" --gen 64`;
сверка: `HY4_DUMP=dir … --save-ids ids.txt`, затем `scripts/reference/hy4_ref.py --ids ids.txt --out ref/`
и `hy4_cmp.py ref/ dir/`.
