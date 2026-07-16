# ACE-Step v1.5 — нативный порт в synaptix (SPEC)

Источник истины: официальный Python-репозиторий `~/Temp/ACE-Step-1.5/acestep/`
(модели `models/xl_base|xl_turbo`, `llm_inference.py`, `constrained_logits_processor.py`,
`acestep_v15_pipeline.py`, VAE = HF `AutoencoderOobleck`).

Веса уже сконвертированы в `.syn` (`storage/syn_models/`):
- `acestep_v15_xl_base.syn` / `acestep_v15_xl_turbo.syn` — DiT-бандл (5 компонентов, см. ниже)
- `acestep_5hz_lm_1.7b.syn` / `acestep_5hz_lm_4b.syn` — 5Hz AR LM (arch `qwen`, purpose `music-lm`)
- `acestep_vae.syn` — Oobleck VAE (arch `ace-step-vae`)
- text-энкодер: `bge-m3.syn` / `qwen3-embedding-0.6b.syn` (оба hidden=1024; ОТКРЫТО — какой именно, см. §Open)

Паттерн порта = `tts/voxcpm` и `video/ltx23`: модули `config / loader / model / pipeline`,
загрузка только из `.syn` через `synaptix-bundle`, бит-гейт против Python-эталона.

---

## 1. Раскладка `.syn` (ground truth)

### DiT-бандл `acestep_v15_xl_base.syn` (arch `ace-step-dit`), компоненты-по-префиксу:
| prefix | тензоров | что это |
|---|---|---|
| `decoder` | 628 | сам **DiT** (32 слоя) |
| `encoder` | 140 | **condition encoder** (фьюз text+lyric+timbre) |
| `detokenizer` | 28 | **AudioTokenDetokenizer** (5Hz codes → 25Hz lm_hints) |
| `tokenizer` | 32 | **FSQ audio tokenizer** (+ quantizer) |
| `null_condition_emb` | 1 | null-эмбеддинг для CFG |

Бандл также несёт `silence_latent.safetensors` (silence-латент для text2music), `config.json`.

### DiT config (`config.json`, xl_base — turbo отличается димами):
```
hidden_size            2560        (turbo: 2048)
num_hidden_layers      32
num_attention_heads    32          (turbo: 16)
num_key_value_heads    8           GQA n_rep=4
head_dim               128
intermediate_size      9728        SwiGLU (turbo: 6144)
in_channels            192         = context_latents(128) + audio(64)
audio_acoustic_hidden_dim 64       выход DiT (латент VAE)
patch_size             2           Conv1d k=s=2 (proj_in) / ConvTranspose1d (proj_out)
rope_theta             1e6
rms_norm_eps           1e-6
layer_types            sliding/full чередуются; sliding_window=128
qk_norm                есть (Qwen3RMSNorm на head_dim, q_norm/k_norm)
encoder_hidden_size    2048        выход condition encoder → condition_embedder Linear(2048→2560)
text_hidden_dim        1024        вход text-энкодера в condition encoder
```
Condition-encoder: `encoder_num_attention_heads 16`, `encoder_num_key_value_heads 8`,
`encoder_intermediate_size 6144`, `num_lyric_encoder_hidden_layers 8`,
`num_timbre_encoder_hidden_layers 4`, `timbre_hidden_dim 64`, `timbre_fix_frame 750`.
Detokenizer: `num_attention_pooler_hidden_layers 2`, `pool_window_size 5` (5→25 Hz),
`num_audio_decoder_hidden_layers 24`.
FSQ: `fsq_dim 2048`, `fsq_input_levels [8,8,8,5,5,5]` (∏=64000 кодов), `fsq_input_num_quantizers 1`,
`vocab_size 64003` (64000 кодов + 3 спец).
Sampler: `timestep_mu -0.4`, `timestep_sigma 1.0` (logit-normal — обучение; инференс linspace+shift).

### 5Hz LM `acestep_5hz_lm_1.7b.syn` (arch `qwen`, `Qwen3Model`):
```
hidden_size 2048, num_hidden_layers 24, num_attention_heads 16, num_key_value_heads 8,
head_dim 128, intermediate_size 6144, rope_theta 1e6, vocab_size 64003,
layer_types все full_attention, bos 151643 eos 151645, dtype bf16, qk_norm есть.
```
Тензоры: `embed_tokens`, `layers.{0..23}.*` (308), `norm`. lm_head нет → tied к embed_tokens.
В бандле токенайзер (vocab/merges/tokenizer.json), `added_tokens.json` (audio_code_N), chat_template.

### VAE `acestep_vae.syn` (`AutoencoderOobleck`):
```
audio_channels 2 (stereo), sampling_rate 48000,
encoder_hidden_size 128, decoder_channels 128, decoder_input_channels 64 (латент),
channel_multiples [1,2,4,8,16], downsampling_ratios [2,4,4,6,10] → hop 1920 → 25 Hz латент.
```
Тензоры (weight-norm `weight_g`/`weight_v`): `encoder.conv1/2`, `encoder.block.{0..4}.{res_unit1..3,snake1,conv1}`,
`decoder.conv1/2`, `decoder.block.{0..4}.{snake1,conv_t1,res_unit1..3}`. ResUnit = snake1,conv1(dil),snake2,conv2(k1).
Латент = diagonal Gaussian: encoder→[mean|scale], std=softplus(scale)+1e-4, z=mean+std·noise.

---

## 2. Карта переиспользования synaptix (что НЕ писать заново)

| Нужно | Готовое в synaptix |
|---|---|
| 5Hz AR LM (Qwen3Model) | `crates/synaptix-models/llm/qwen3` — модель/лоадер/KV-кэш/sampling |
| Oobleck VAE (snake+weight_norm+ResUnit+ConvT) | `tts/voxcpm/src/audiovae.rs` — Snake, WnConv/WnConvT, ResUnit — ТОТ ЖЕ паттерн, адаптировать имена/strides |
| conv1d / conv_transpose1d / depthwise | `synaptix-ops::conv` |
| RMSNorm / Linear / attention / RoPE / SwiGLU | `synaptix-nn`, ltx23/qwen3 attention-блоки |
| schedulers | `synaptix-diffusion/src/schedulers` (flow-match euler + shift — добавить/адаптировать) |
| `.syn` загрузка | `synaptix-bundle::Bundle`, `tensors_slice_for(component)` |

---

## 3. Архитектура форварда (для реализации)

**DiT-слой** (×32, AdaLN-Single): `scale_shift_table[1,6,D] + temb` → chunk6
(shift/scale/gate ×2). Порядок: AdaLN→self-attn(RoPE,qk-norm,GQA, sliding/full)→×gate→residual;
AdaLN→cross-attn(K/V=encoder_hidden, без RoPE)→residual; AdaLN→SwiGLU-MLP→×gate→residual.
Финал: norm_out с (scale_shift_table+temb).chunk2, proj_out=ConvTranspose1d→[B,T,64], crop.
proj_in: cat([context_latents(128), x(64)],dim=-1)=192 → pad T до кратного 2 → Conv1d k=s=2 → [B,T/2,2560].
timestep: два TimestepEmbedding (t и t−r), temb=temb_t+temb_r, timestep_proj=proj_t+proj_r (256→D, time_proj→6D).

**Condition encoder**: text(1024)/lyric(1024 через lyric-encoder 8L)/timbre(64 через timbre-encoder 4L,
fix_frame 750) → пакуется → encoder_hidden[B,L,2048] + mask → condition_embedder Linear→2560.

**LM→DiT мост** (prepare_condition): 5Hz LM AR-генерит audio_code IDs (constrained, phase1 CoT-метаданные /
phase2 коды) → FSQ.get_output_from_indices → 5Hz codes → detokenizer(pool 5) → 25Hz lm_hints →
если is_covers: src_latents=lm_hints (иначе silence_latent) → context_latents=cat([src_latents, chunk_masks]).

**Sampler**: t=linspace(1,0,steps+1); shift: t=shift·t/(1+(shift−1)·t); Euler: x−=Δt·v;
CFG (cfg_interval), APG (momentum=−0.75, проекция по T-оси), DCW per-step correction.

---

## 4. Фазовый план (каждая фаза бит-гейтится против Python)

- [x] **Ф0** — скелет крейта: config.rs (реальные димы), loader.rs (CompLoader), ошибки, lib.rs.
- [x] **Ф1 — VAE** (Oobleck, decode+encode_mean): vae.rs, гейт shape+finite (vae_smoke). Бит-гейт vs Python — TODO (GPU).
- [x] **Ф2a — 5Hz LM** load+forward поверх `DecoderModel` (lm.rs, BundleWeightSource). Гейт lm_smoke.
- [x] **Ф2b — токенайзер + AR codes** (tokenizer.rs, ar.rs): audio-code маппинг, codes-фаза constrained. Гейт tokenizer/ar_smoke. Отложено: Phase-1 CoT FSM, lm_cfg.
- [x] **Ф3a — FSQ** get_output_from_indices (fsq.rs). Гейт fsq_smoke.
- [x] **Ф3b — AudioTokenDetokenizer** (detokenizer.rs) + общий bidir encoder-слой (encoder.rs:
      RMSNorm+attn qk_norm+RoPE split-half+GQA+SwiGLU). Гейт detokenizer_smoke: codes→lm_hints [1,T·5,64].
- [x] **Ф4 — condition encoder** (cond_encoder.rs): text_projector(1024→2048) + lyric_encoder(8сл) + фьюз cat([lyric,text]). Гейт cond_encoder_smoke. ОТЛОЖЕНО: timbre (cover-only), padding-pack, реальный text-энкодер (Ф6).
- [x] **Ф5 — DiT** (dit.rs): proj_in Conv1d + двойной TimestepEmbedding + 32 AdaLN-слоя (self-attn RoPE/qk_norm/GQA + cross-attn на encoder_hidden + SwiGLU) + norm_out + proj_out ConvT. Гейт dit_smoke: velocity [1,8,64]. TODO: sliding-window маска (пока full), бит-гейт vs Python.
- [~] **Ф6a — scheduler + sampler-ядро** (scheduler.rs timestep_schedule + pipeline.rs denoise Euler+CFG). Гейт: scheduler unit 2/2.
- [x] **Ф6b — e2e generate_music** (pipeline.rs) + cuda-пример music_gen: LM→codes→FSQ→detok→lm_hints ‖ Qwen3-Emb→cond→enc → context(src+chunk_masks=ones) → denoise(DiT) → VAE→wav. GPU-прогон: 8s аудио за 15.6с, сигнал валиден (rms 0.26, 0 NaN). TODO качество: APG(-0.75)/DCW, CFG-тюнинг, chunk_masks/is_covers сверка vs Python, timbre.
- [x] **Ф7 — перепроводка 8 нод synthos** на synaptix (types/shared/mod/8 нод; cargo check 0 ошибок). Допорчены timbre/APG/DCW/чанк-VAE/Phase1-CoT. Упрощения: lm_cfg, реальный silence_latent — квалити-проход.
- [x] **КАЧЕСТВО ПОЧИНЕНО**: 2 корневых бага (структурный text-промпт + FSQ preserve_symmetry). Все компоненты cos 1.0 vs Python reference. Пайплайн prompt-зависим (rock громко / ambient тихо). + timbre в enc.
- [ ] **Ф8 — CLI** `synaptix music` (опц.; пока example music_gen) + перф/квант.

### Заметки для Ф3b/Ф4 (encoder-слой, общий)
AceStepEncoderLayer (modeling base:380) = Qwen3 bidir: input_layernorm→self_attn→+res;
post_attention_layernorm→mlp(SwiGLU)→+res. attn: q_proj→q_norm(RMSNorm head_dim)→view→T(1,2);
k/v аналогично (k_norm); RoPE split-half (theta 1e6) на q,k; eager, scale=head_dim^-0.5; GQA n_rep.
Билдинг-блоки synaptix: `Tensor::rms_norm_fused(w,eps,false)`, `rope_split_fused(cos,sin)`,
`RopeCache::new(head_dim,max_seq,theta,dev).select_positions`, `synaptix_ops::attention::softmax_dim`,
`synaptix_nn::linear::Linear`. Детокенайзер: seq=5 (pool_window), bidir, маска тривиальна.

## Open questions (уточнить у пользователя / в pipeline)
1. text-энкодер для text_hidden_dim=1024: `bge-m3` или `qwen3-embedding-0.6b`? (оба 1024).
2. Метаданные/CoT планируются 5Hz LM (vocab 64003 = только аудио-коды?) или отдельным planner-LM?
   В наших весах только 5hz_lm — возможно phase1 идёт на внешнем Qwen3 (qwen3.6 27B / отдельный).
3. Целевой вариант первым: xl_base vs xl_turbo (turbo = меньше шагов, быстрее гейт); LM 1.7b vs 4b.
4. Бит-гейт: есть Python `.venv` в `~/Temp/ACE-Step-1.5` для дампа эталонных тензоров (нужен GPU — только с явного «да»).
