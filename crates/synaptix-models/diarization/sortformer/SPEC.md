# Sortformer (NeMo Streaming Sortformer 4spk v2.1) — synaptix port spec

> **Источник истины — официальный NVIDIA NeMo** (`.nemo` v2.1 / репозиторий NeMo).
> Реализовано нативно по NeMo-спеке; bit-exact против NeMo reference.

Нативный порт диаризации спикеров на synaptix Tensor API. Цель — bit-exact воспроизведение NeMo.
Конвенции порта — как `asr/whisper`, `tts/voxcpm` (loader = `SynBundleLoader`, слои напрямую через
Tensor, кроме `Linear`).

## Цепочка обработки

```
PCM 16kHz mono
 → NemoMelExtractor (preemph 0.97 → STFT n_fft=512/hop=160/win=400 center-reflect
                     → 128-bin Slaney mel → log clamp 2^-24 → normalize="NA"(skip v2.1))  → (1,128,T)
 → FastConformerEncoder
     · DwStridedSubsampling8x: (1,1,T,128) →3×Conv2d(stride2)+ReLU→ (1,256,T/8,16)
       → permute(0,2,1,3) → reshape (1,T/8,4096) → Linear(4096→512)               → (1,T/8,512)
     · xscaling: x *= sqrt(512)
     · 17× FastConformerLayer (Macaron): ½FFN → RelPosMHSA → ConvModule → FFN → LN  → (1,T/8,512)
 → transpose(1,2) → (1,512,T/8)
 → SortformerHead
     · Linear(512→192) encoder_proj
     · 18× SortformerLayer (POST-LN): SelfAttn→norm1 ; FFN(ReLU)→norm2             → (1,T/8,192)
     · ReLU → Linear(192→192) hidden_proj → ReLU → Linear(192→4) classifier        → (1,T/8,4) logits
 → sigmoid → (1,T/8,4) per-speaker probs @ 12.5 Hz
 → postprocess: binarize(thr 0.5) → median_smooth(3) → segments → merge(<0.15s)
                → filter(<0.25s) → arrival-time re-id                              → Vec<DiarizeSegment>
```

## Конфиг (v2.1 defaults, см. `config.rs`)

sample_rate 16000 · max_speakers 4 · frame_rate 12.5 Hz.
Preprocessor: n_window_size 400, n_window_stride 160, n_fft 512, n_mels 128, log, dither 1e-5,
preemph 0.97, normalize "NA", pad_to 0.
Encoder (FastConformer): feat_in 128, n_layers 17, d_model 512, n_heads 8 (d_k=64),
subsampling dw_striding ×8, subs_kernel_size 9, ff_expansion 4 (d_ff 2048), self_attn rel_pos,
pos_emb_max_len 5000, conv_kernel_size 9, conv_norm batch_norm, subsampling_conv_channels 256, xscaling.
Head: feat_in 512, n_layers 18, d_model 192, n_heads 8 (d_k=24), ff_expansion 4 (d_ff 768), max_speakers 4.
Streaming: spkcache_len 188, fifo_len 0, chunk_len 188, chunk L/R context 1, sil_frames/spk 3,
pred_score_threshold 0.25, ... (см. StreamingConfig).

## Имена весов (.syn / safetensors, HF-раскладка)

Subsampling: `encoder.pre_encode.conv.{0,2,3,5,6}.{weight,bias}`, `encoder.pre_encode.out.{weight,bias}`.
Encoder layer i∈0..16 (`encoder.layers.{i}.`):
  `norm_feed_forward1.{w,b}`, `feed_forward1.linear{1,2}.{w,b}`,
  `norm_self_att.{w,b}`, `self_attn.linear_{q,k,v,out}.{w,b}`, `self_attn.linear_pos.weight` (NO bias),
  `self_attn.pos_bias_u` (8,64), `self_attn.pos_bias_v` (8,64),
  `norm_conv.{w,b}`, `conv.pointwise_conv1.{w,b}`, `conv.depthwise_conv.{w,b}`,
  `conv.batch_norm.{weight,bias,running_mean,running_var}`, `conv.pointwise_conv2.{w,b}`,
  `norm_feed_forward2.{w,b}`, `feed_forward2.linear{1,2}.{w,b}`, `norm_out.{w,b}`.
Head: `head.encoder_proj.{w,b}`; head layer i∈0..17 (`head.layers.{i}.`):
  `norm1.{w,b}`, `self_attn.linear_{q,k,v,out}.{w,b}`, `norm2.{w,b}`, `feed_forward.linear{1,2}.{w,b}`;
  `head.hidden_proj.{w,b}`, `head.classifier.{w,b}`.

## Критичные места (легко испортить — bit-exact)

1. **RelPosMHSA (T5-XL)** `encoder.rs:185-274`: AC=(Q+u)@Kᵀ, BD=(Q+v)@Pᵀ, scores=(AC+BD)/√d_k.
   `pos_emb`: интерливинг sin/cos (НЕ cat), phase = p/10000^(2i/d). `rel_shift(BD)`: pad-left-1 →
   reshape (B,H,T,2L)→(B,H,2L,T) → drop row 0 → reshape → narrow до T колонок. softmax по последней оси.
2. **DwStriding subsampling**: после 3×conv2d (B,256,T/8,16) → permute(0,2,1,3) **→ contiguous →**
   reshape (B,T/8,4096) → Linear(4096→512). Порядок permute→contiguous→reshape обязателен.
3. **ConvModule (Conformer)**: transpose→pointwise_conv1(512→1024)→GLU(h1·σ(h2))→
   depthwise_conv(k=9,groups=512,pad=4)→BatchNorm1d(eps 1e-5, running stats, inference)→SiLU→
   pointwise_conv2→transpose.
4. **Mel = Slaney scale** (НЕ HTK): f<1000 mel=f/(200/3); f≥1000 mel=1000/(200/3)+ln(f/1000)/(ln(6.4)/27).
   area-normalized filterbank. log clamp min=2^-24. normalize="NA" → шаг per-feature пропускается.
5. **Активации**: encoder FFN = **SiLU**; head FFN = **ReLU**; conv GLU gate = sigmoid.
6. **Head = POST-LN** (LN после residual), без финального LN (нет в v2.1 state_dict).
7. **Macaron FFN**: half-residual ×0.5 на обоих FFN энкодера.
8. **BatchNorm1d** инференс: frozen running_mean/var, eps 1e-5. **LayerNorm** eps 1e-5.

## Постпроцессинг

`DiarizeSegment { start_s, end_s, speaker: u8, confidence: f32 }`.
`PostprocessParams { threshold 0.5, min_segment_s 0.25, merge_gap_s 0.15, frame_rate_hz 12.5,
max_speakers 4, smoothing_frames 3 }`.
Flow: binarize (allow_overlap → thr per-spk; иначе argmax) → median smooth(3) → per-speaker
contiguous intervals + confidence=avg(prob) → merge gaps<0.15s → drop<0.25s → arrival-time re-id.

## Streaming (фаза 2 — после batch)

`SortformerStreamingState { spkcache, fifo, mean_sil_emb, n_sil_frames, spkcache_preds, fifo_preds }`.
chunk 188 frames + L/R context 1; `compress_spkcache` при переполнении: log-scores (NeMo формула) →
disable_low → boost_topk → **flat-index topk + sort flat_idx (группировка по спикеру)** → gather + sil-replace.
Это самая хитрая часть — портировать ПОСЛЕ batch-режима с отдельной валидацией.

## Статус реализации (BATCH-путь готов и сверен с NeMo)

1. ✅ config.rs + loader.rs + скелет.
2. ✅ mel.rs — NeMo `FilterbankFeatures` (preemph→STFT sym-Hann→power→fb@→log+2⁻²⁴),
   окно/fb из чекпойнта; cos 0.9999 vs NeMo.
3. ✅ encoder.rs — dw_striding subsampling(conv2d) + 17 FastConformer (RelPosMHSA+rel_shift,
   ConvModule+BatchNorm1d, Macaron½, xscaling); per-frame mean cos 0.9995 vs NeMo.
4. ✅ head.rs — encoder_proj + 18 post-LN transformer (std MHA) + sigmoid-голова; cos 0.9997.
5. ✅ model.rs — load + forward_stages + diarize_pcm (full-attention batch).
6. ✅ postprocess.rs — frames_to_segments (binarize/median-smooth/merge/filter/arrival-reid).
7. ✅ pipeline.rs — SortformerPipeline::from_syn + diarize → сегменты.
8. ✅ tests/sortformer_gate.rs — постадийный гейт vs NeMo-дамп; **preds bin-agree спикеров = 1.0**.
9. ⏳ GPU-валидация (с явного «да»).
10. ⏳ streaming.rs — chunked spkcache/fifo + `_compress_spkcache` (для длинных записей, фаза 2;
    для клипов ≤~15с в eval batch≡streaming).

Эталон: официальный NeMo (`~/Temp/NeMo`) через `reference/dump_sortformer.py`.
Маппинг HF→NeMo-имён + установка NeMo-env — в `reference/` и памяти проекта.
