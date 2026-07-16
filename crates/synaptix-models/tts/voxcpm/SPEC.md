# VoxCPM2 — нативный порт в synaptix (спецификация)

Источник истины: репозиторий `Temp/VoxCPM` (Python, conf/voxcpm_v2) +
бандл `storage/syn_models/voxcpm2.syn` (config.json внутри = ground truth).
Реализация по официальному upstream (VoxCPM).

Инспекция бандла: `cargo run -p synaptix-io --example dump_voxcpm -- <bundle.syn>`
(пример лежит в synaptix-io/examples; печатает meta/files/config.json + карту тензоров).

## Бандл
- arch="voxcpm2", purpose="tts", ver 1.0
- Компоненты: `base` (prefix `''`) и `audiovae` (prefix `audiovae`).
- Файлы: config.json, tokenizer.json (LlamaTokenizerFast/SP BPE), tokenizer_config.json,
  special_tokens_map.json, tokenization_voxcpm2.py, README.md.
- LM/DiT/enc/проекции — BF16; AudioVAE — F32 (всегда fp32 на инференсе).

## Гиперпараметры (config.json, подтверждено)
- lm_config: hidden 2048, ffn 6144, layers 28, heads 16, kv_heads 2, kv_channels(head_dim) 128,
  rms_eps 1e-5, rope_theta 10000, vocab 73448, **use_mup=false**, max_pos 32768.
  rope_scaling type=longrope, long_factor==short_factor (64 значения), original_max_pos 32768.
- patch_size P=4, feat_dim/latent D=64, fsq latent 512 scale 9.
- residual_lm: 8 слоёв, **no_rope=true**, без embed_tokens.
- encoder_config: hidden 1024, ffn 4096, heads 16, layers 12, kv_channels 128 (RoPE on, non-causal).
- dit_config: hidden 1024, ffn 4096, heads 16, layers 12, kv_channels 128, mean_mode false,
  cfm: sigma_min 1e-6, solver euler, t_scheduler log-norm, inference_cfg_rate 2.0.
- audio_vae_config: encoder_dim 128, encoder_rates [2,5,8,8], latent 64, decoder_dim 2048,
  decoder_rates [8,6,5,2,2,2], sr_bin_boundaries [20000,30000,40000], sample_rate 16000, out 48000.
- max_length(KV) 8192.

## Спецтокены
audio_start=101, audio_end=102, ref_audio_start=103, ref_audio_end=104.

## ⚠ Критичные нюансы (легко ошибиться)
1. **use_mup=false** → scale_emb=1 (без масштаба эмбеддингов), без scale_depth, обычные residual.
2. **LongRoPE**: scaling_factor=1.0 (max_pos==original), short==long → каждую inv_freq[i] делим
   на factor[i] (64 разных значения). inv_freq = 1/theta^(2i/dim), dim=128. cos/sin в fp32.
3. **head_dim=kv_channels=128 ВЕЗДЕ** (вкл. 1024-мерные enc/dit: q_proj out=2048, o_proj in=2048).
4. RMSNorm: variance в fp32, умножение на weight в исходном dtype, eps 1e-5.
5. q/k → fp32 перед RoPE.
6. **FSQ** только на audio-позициях; в AR-цикле применяется к каждому свежему lm_hidden до residual LM.
   round(tanh(in_proj(h))·9)/9, in 2048→512, out 512→2048.
7. **LocDiT v2**: mu (2048) → 2 токена [N,2,1024]; seq = [mu(2), t(1), cond(P'), x(P)];
   на выходе берём последние P токенов (slice start = P' + 2 + 1). delta_time_mlp(sinusoid(0))
   входит в t даже при dt=0 (ненулевой вклад). SinusoidalPosEmb scale=1000, dim 1024.
8. **CFM**: t_span = linspace(1,0,n+1) + sway(coef=1)·(cos(π/2·t)-1+t); zero_init_steps=max(1,int(len·0.04))=1
   (на 1-м шаге velocity=0); CFG-zero-star: st_star=Σ(pos·neg)/(Σneg²+1e-8); guided = uncond·st + cfg·(cond - uncond·st);
   Эйлер x -= dt·v; uncond = mu обнулён (2-я половина CFG-батча). noise randn(b,64,4)·temperature.
9. **AudioVAE fp32**: weight_norm w=g·v/‖v‖ (норма по всем осям кроме 0); Snake x+(α+1e-9)^-1·sin(αx)²;
   CausalConv left-pad=2·padding-output_padding; depthwise groups; SR-cond (scale_bias, bucket idx 3 для 48к)
   применяется ПЕРЕД каждым CausalDecoderBlock; tensor name weight_g/weight_v/alpha/sr_cond_model.*.{scale,bias}_embed.
10. hop=640, chunk=640, decode_chunk=960, P=4. encode patch_len=2560 семплов; decode_patch_len=3840; latent fps=25; out 48к.
11. Токенизатор БЕЗ BOS/EOS: text_token = subword ids ++ [101]; multi-char CJK split на одиночные.
12. KV-cache обнуляется при prefill-fill; current_len=prefill k.size(2); decode пишет в pos, маска arange<=pos.
13. Дефолты: cfg_value=2.0, n_timesteps=10, min_len=2, retry ratio 6.0.

## Карта тензоров (ключевые)
- base_lm.embed_tokens.weight [73448,2048]; base_lm.layers.{0..27}.{input_layernorm,post_attention_layernorm}.weight;
  .self_attn.{q[2048,2048],k[256,2048],v[256,2048],o[2048,2048]}_proj.weight; .mlp.{gate,up[6144,2048],down[2048,6144]}_proj.weight;
  base_lm.norm.weight.
- residual_lm.layers.{0..7}.* (как base, но no_rope, без embed); residual_lm.norm.weight.
- feat_encoder.special_token [1,1,1,1024]; feat_encoder.in_proj.{weight[1024,64],bias}; feat_encoder.encoder.layers.{0..11}.*
  (q[2048,1024],k/v[256,1024],o[1024,2048]); feat_encoder.encoder.norm.weight.
- enc_to_lm_proj.{weight[2048,1024],bias}; lm_to_dit_proj.{weight[1024,2048],bias}; res_to_dit_proj.{[1024,2048],bias};
  fusion_concat_proj.{weight[2048,4096],bias}.
- fsq_layer.in_proj.{weight[512,2048],bias}; fsq_layer.out_proj.{weight[2048,512],bias}.
- feat_decoder.estimator.{in_proj[1024,64],cond_proj[1024,64],out_proj[64,1024]}.{weight,bias};
  .time_mlp.{linear_1,linear_2}[1024,1024].{weight,bias}; .delta_time_mlp.{linear_1,linear_2}.{weight,bias};
  .decoder.layers.{0..11}.* (q[2048,1024],k/v[256,1024],o[1024,2048]); .decoder.norm.weight.
- stop_proj.{weight[2048,2048],bias}; stop_head.weight [2,2048] (bias=False).
- audiovae.* (F32, 889 тензоров): encoder.block.*, encoder.fc_mu/fc_logvar (weight_g/v+bias),
  decoder.model.* (weight_g/v, alpha), decoder.sr_cond_model.{i}.{scale,bias}_embed.weight [4, dim],
  decoder.sr_bin_boundaries I32 [3].

## Forward (zero-shot) и 4 режима cloning — см. spec агента в истории / Python core.py _inference/_generate.
```
prefill: feat_embed=enc_to_lm_proj(feat_encoder(audio_feat)); text_embed=embed(text)*1;
combined=text_mask·text_embed+audio_mask·feat_embed; base_lm(causal)→enc; enc=fsq(enc)·a_mask+enc·t_mask;
lm_hidden=enc[:,-1]; residual_in=fusion_concat_proj(cat(enc, a_mask·feat_embed)); residual_lm(causal)→res_hidden.
AR loop: mu=cat(lm_to_dit_proj(lm_h), res_to_dit_proj(res_h)); pred=CFM(mu,cond=prefix_feat); curr=enc_to_lm_proj(feat_encoder(pred));
stop=argmax(stop_head(silu(stop_proj(lm_h)))); lm_h=fsq(base_lm.step(curr)); res_h=residual_lm.step(fusion_concat_proj(cat(lm_h,curr))).
out: cat pred_feat → [1,64,T·P] → audiovae.decode(fp32, sr=48000→bucket3) → PCM.
```

## СТАТУС (2026-06-17)
✅ Полный нативный порт инференса РЕАЛИЗОВАН и РАБОТАЕТ end-to-end на CPU (f32).
Модули: config/loader/minicpm/locenc/locdit/cfm/fsq/audiovae/tokenizer/audio_io/model/pipeline.
Сквозной прогон `synthesize("Hello world.")` (max_len=4) → 30720 семплов @ 48кГц, всё конечно (163с CPU/f32).
Тесты `tests/smoke.rs`: загрузка (валидирует все имена тензоров), токенайзер, VAE-декод формы — PASS;
полный синтез за `VOXCPM_GENERATE=1` — PASS. Реализованы все 4 режима (zero-shot/reference/continuation/combined).
Уроки CPU: silu_and_mul/embed_gather/rope_split_fused не на CPU → fallback'и; gqa repeat_interleave даёт
non-contiguous → расширяю KV сам + scaled_dot; narrow→reshape требует .contiguous().
✅ CLI `synaptix speak <bundle> <text> [-o out.wav --device cuda --reference/--prompt-wav/--prompt-text ...]` — ГОТОВО.
✅ GPU-прогон CUDA bf16: загрузка 0.89с; «The quick brown fox...» → 3.84с аудио (20 патчей, stop сработал),
RTF 2.46 (без опт); сигнал реальный peak 0.846 / rms 0.082 / ~50% озвучено (без клиппинга). Файл /tmp/vox_full.wav.
ОСТАЁТСЯ: streaming-декод; mel-cosine сверка с Python (БЛОКЕР: нужен HF-каталог весов, у нас только .syn);
нода в synthos + удаление старой страницы voxcpm2; перф-оптимизация (RTF 2.46 → цель <1).

## План модулей крейта
config.rs · loader.rs (VoxCheckpoint over SynBundleLoader: base+audiovae) · minicpm.rs (общий гибкий
трансформер: causal/no_rope/longrope/non-causal, KV-cache, prefill+step) · locenc.rs · locdit.rs · cfm.rs ·
fsq.rs · audiovae.rs (encoder+decoder+streaming) · tokenizer.rs · audio_io.rs · pipeline.rs (4 режима + streaming) · lib.rs.
```
