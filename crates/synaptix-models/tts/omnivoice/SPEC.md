# OmniVoice (k2-fsa) — synaptix port spec

Нативный порт массмультиязычной zero-shot TTS **OmniVoice** в synaptix.

> 🛑 **Источник истины — официальный upstream**, по upstream-спеке
>
> - модель: `~/Temp/OmniVoice/omnivoice/models/omnivoice.py` (k2-fsa, Apache-2.0);
> - аудио-кодек: HF `transformers` `models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py`
>   (в `synaptix/scripts/reference/.venv`);
> - веса/конфиги: бандл `storage/syn_models/omnivoice.syn` (распакованные конфиги в
>   `tmp/ov_unpack/`).

## Что это за модель

OmniVoice — **masked-diffusion language model** (MaskGIT-стиль, НЕ autoregressive):
двунаправленный Qwen3-бэкбон над 8 RVQ-аудио-кодбуками, итеративное параллельное раскрытие
маскированных токенов за `num_step` шагов с classifier-free guidance. Отдельный нейро-кодек
**HiggsAudioV2** энкодит референс-аудио в коды и декодит сгенерированные коды в волну 24 кГц.
Режимы: voice-cloning (ref_audio[+ref_text]), voice-design (instruct), auto. 600+ языков.

## Цепочка инференса

```
ref PCM (опц.) → HiggsAudioV2.encode → ref_audio_tokens (8, T_ref)
text → frontend (combine ref_text+text, nonverbal-теги, special-токены)
  style = <|denoise|>?<|lang_start|>LANG<|lang_end|><|instruct_start|>INSTR<|instruct_end|>
  cond_input = [style ; <|text_start|>text<|text_end|> ; ref_audio_tokens? ; MASK×T_target]  (8 кодбуков)
  audio_mask = true на хвосте (ref+target)
→ batch 2B: cond[0..B] + uncond[B..2B] (uncond = только target-хвост)
→ Qwen3 bidirectional backbone (full-attention, БЕЗ causal, БЕЗ KV-cache)
    embeds = where(audio_mask, Σ_c audio_embeddings[codes_c + c·1025], text_embeds[ids])
    → 28 слоёв → hidden (B,S,1024)
→ audio_heads: Linear(1024 → 8·1025) → logits (B,8,S,1025)
→ iterative unmask (num_step):
    CFG: lp = logsoftmax(c + g·(c−u)); lp[mask_id]=−inf
    pred = argmax(lp) (или gumbel при class_temperature>0); score = max(lp)
    score −= layer_id·layer_penalty_factor; score += gumbel(position_temperature)
    score[уже раскрытые]=−inf; раскрыть top-k(score) по schedule[step]
→ codes (8, T_target)
→ HiggsAudioV2.decode → wav 24 кГц → postprocess (remove_silence, rms, fade/pad)
```

## Конфиг (из `ov_unpack/config.json`)

OmniVoice: `audio_vocab_size 1025`, `audio_mask_id 1024`, `num_audio_codebook 8`,
`audio_codebook_weights [8,8,6,6,4,4,2,2]` (только для train-loss), `eos 151645`, `pad 151643`.
**Бэкбон = Qwen3** (`Qwen3ForCausalLM`, но используется двунаправленно): hidden 1024, 28 слоёв,
16 q-голов / 8 kv-голов (GQA), head_dim 128 (q_proj→2048, kv→1024, o_proj 2048→1024),
intermediate 3072, silu, rms_norm_eps 1e-6, qk-norm (Qwen3), rope_theta 1e6, vocab 151676,
tie_word_embeddings true, все слои full_attention. → переиспользовать слой-примитивы
`synaptix-llm-qwen3`/`synaptix-llm-common`/`synaptix-nn`, но в bidirectional-режиме.

HiggsAudioV2 (`ov_unpack/audio_tokenizer/config.json`, `model_type higgs_audio_v2_tokenizer`):
- акустика (`model_type dac`, Descript Audio Codec): encoder_hidden 64, hidden 256, decoder_hidden 1024,
  downsampling/upsampling [8,5,4,2,3] (hop 960), n_codebooks 9, codebook_size 1024, codebook_dim 8,
  sampling_rate 16000;
- semantic (`model_type hubert`): 12 слоёв, hidden 768, conv-feature-extractor (7 слоёв,
  conv_dim 512, kernel [10,3,3,3,3,2,2], stride [5,2,2,2,2,2,2]), 16 кГц;
- top: `sample_rate 24000`, `downsample_factor 320`, codebook_dim 64, codebook_size 1024,
  target_bandwidths [0.5,1,1.5,2]. OmniVoice использует 8 из codec-кодбуков.

GenerationConfig (defaults): num_step 32, guidance_scale 2.0, t_shift 0.1, layer_penalty_factor 5.0,
position_temperature 5.0, class_temperature 0.0 (greedy), denoise true, audio_chunk_duration 15.0,
audio_chunk_threshold 30.0.

## Веса (.syn `omnivoice.syn`, arch `qwen3-omni`)

2 tensors-компонента. File-чанки: config.json, tokenizer.json (151676), tokenizer_config.json,
chat_template.jinja, audio_tokenizer/{config,preprocessor_config}.json.
`syn-unpack` ПОЧИНЕН (покомпонентно: `model.safetensors`=lm, `model-codec.safetensors`); HF-раскладку
для Python-эталона собрать: lm→`model.safetensors`, codec→`audio_tokenizer/model.safetensors`.
Распакованный снапшот: `tmp/ov_unpack/`.

**`lm` (313 тензоров, F32):**
- `audio_embeddings.weight` [8200,1024] (8·1025); `audio_heads.weight` [8200,1024]; `codebook_layer_offsets` [8] i64.
- `llm.embed_tokens.weight` [151676,1024] (tied; отдельного lm_head НЕТ — выход через audio_heads); `llm.norm.weight`.
- `llm.layers.{i}.` (28): `input_layernorm`/`post_attention_layernorm`; `self_attn.{q_proj[2048,1024],k_proj[1024,1024],v_proj[1024,1024],o_proj[1024,2048],q_norm[128],k_norm[128]}`; `mlp.{gate_proj[3072,1024],up_proj[3072,1024],down_proj[1024,3072]}`. = чистый Qwen3 (префикс `llm.`).

**`codec` (527 тензоров, F32) — HiggsAudioV2:**
- DAC `acoustic_encoder` (conv1[64,1,7] → block.{i}{ResidualUnit×3(conv1 k7/conv2 k1/Snake), Snake, conv stride} → conv2[256,2048,3], Snake) и `acoustic_decoder` (conv1[1024,256,7] → block.{i}{Snake, conv_t1 transpose, res_unit×3} → Snake, conv2[1,32,7]). Snake-активация (`.alpha`).
- `semantic_model` (HuBERT, 12 сл: feature_extractor + encoder.layers.{i}.{attention.{q,k,v,out}_proj, feed_forward.{intermediate_dense,output_dense}, layer_norm}).
- `encoder_semantic`/`decoder_semantic` (conv_blocks 768-dim над semantic-фичами).
- RVQ `quantizer.quantizers.{i}.{codebook.embed[1024,64], project_in[64,1024], project_out[1024,64], cluster_size, embed_avg, inited}`.
- fusion `fc[1024,1024]`, `fc1[768,1024]`, `fc2[256,1024]`.
- decode (для генерации): codes → project_out → fusion → acoustic_decoder → wav 24кГц. encode (ref): acoustic_encoder + semantic_model + quantizer → codes.

## Критичные места (bit-exact, per-row vs Python-эталон)

1. Audio-embed: `shifted = codes·audio_mask + c·1025`; `Σ_c audio_embeddings(shifted)`;
   `where(audio_mask, audio_embeds, text_embeds)` (text — tied input-embed по `ids[:,0,:]`).
2. Qwen3 backbone: qk-norm, rope θ1e6, GQA 16/8, **FULL bidirectional attention** (block-mask по
   документу, без causal!), без KV-cache (каждый шаг — полный forward).
3. CFG: `logsoftmax(c_lp + g·(c_lp − u_lp))`, затем `[mask_id] = −inf` (порядок важен).
4. Schedule: `_get_time_steps` (linspace+t_shift трансформ), per-item `ceil(total·Δt)`, остаток на
   последний шаг. topk по `scores.flatten()`. layer-penalty `score − layer_id·5`. gumbel position/class.
5. HiggsAudioV2: DAC weight-norm свёртки, RVQ codebook-lookup/проекции, HuBERT semantic, апсэмпл до 24 кГц.
6. cond/uncond батч 2B; uncond = только target-хвост; padding-diag в attention-маске uncond.

## План порта (каждый шаг — bit-exact per-row vs upstream-дамп, НЕ cos)

1. ✅ Разбор архитектуры + конфигов + этот SPEC + скелет крейта.
2. Покомпонентная распаковка `.syn` (`tensors:lm`/`tensors:codec`) — инструмент/loader.
3. Python-эталон: прогнать upstream OmniVoice на `ov_unpack/` (нужен transformers-venv), сдампить
   ref-tokens / backbone-hidden / audio-logits по шагам / codes / wav.
4. text.rs — frontend: токенайзер (Qwen3), special-токены, nonverbal-теги, `_combine_text`, style;
   `RuleDurationEstimator`; lang_map; voice_design instruct. Гейт: token-ids.
5. backbone.rs — Qwen3 bidirectional + audio_embeddings + audio_heads. Гейт: hidden+logits (1 forward).
6. masked_decode.rs — итеративное раскрытие (CFG/schedule/gumbel/layer-penalty/topk). Гейт: codes (greedy/seeded).
7. audio_codec.rs / audio_encode.rs — HiggsAudioV2 codec.
   ✅ decode (audio_codec.rs): codes → RVQ.decode → fc2 → DAC acoustic_decoder → wav.
   ✅ encode (audio_encode.rs, voice-clone): ref-wav 24к → [resample 24→16к (torchaudio
      sinc-hann bit-faithful порт) → HuBERT(13 hidden_states mean) → ::2 downsample] →
      encoder_semantic (ELU res-units) ‖ DAC acoustic_encoder → fc(cat) → RVQ.encode
      (L2-nearest, residual-loop, n_q=8 из bandwidth=2.0). Гейт `encoder_gate`: 100%
      совпадение кодов (8×150 и 8×300), stage cos=1.0 (semfeat/e_sem/e_ac/emb, max-abs ~4e-5).
8. pipeline.rs — from_syn + generate (clone/design/auto) + create_voice_clone_prompt + postprocess. Гейт: e2e wav.
9. Опц. ASR ref-auto-transcribe — переиспользовать `synaptix-asr-whisper`.
10. Перф (после корректности). 11. Проводка synthos-ноды → `synaptix-tts-omnivoice` (заменить стаб `app/synthos/stubs/tts-omnivoice`).

Конвенции порта — как `tts/voxcpm`, `music/acestep` (loader=synaptix-bundle, слои через Tensor/synaptix-nn).
```
