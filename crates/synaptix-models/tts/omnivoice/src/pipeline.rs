//! Высокоуровневый пайплайн OmniVoice (e2e auto-режим: текст → волна).
//!
//! `from_syn`/`from_unpacked` грузят config + tokenizer + lm/codec-веса, строят
//! `Backbone` + `TextFrontend` + `CodecDecoder`. `generate` (auto, без ref):
//! duration → num_target_tokens → text-frontend → masked_decode → codes →
//! codec.decode → wav. Источник: `~/Temp/OmniVoice/omnivoice/models/omnivoice.py`
//! (`generate`, `_preprocess_all`, `_generate_iterative`, `_decode_and_post_process`).
//! См. SPEC.md «План порта» п.8.
//!
//! Voice-clone путь (`create_voice_clone_prompt` + `generate_clone`) реализован.
//! Postprocess / chunking / synthos-нода — TODO (см. отчёт).

use std::path::Path;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::audio_codec::CodecDecoder;
use crate::audio_encode::CodecEncoder;
use crate::backbone::Backbone;
use crate::config::{HiggsAudioConfig, OmniVoiceConfig, OmniVoiceGenerationConfig};
use crate::loader::{OmniVoiceCodecWeights, OmniVoiceLmWeights};
use crate::masked_decode::generate_iterative;
use crate::prompt::{create_voice_clone_prompt, VoiceClonePrompt};
use crate::text::{DurationEstimator, TextFrontend};
use crate::{OmniVoiceError, Result};

fn err<E: std::fmt::Display>(e: E) -> OmniVoiceError {
    OmniVoiceError::Other(e.to_string())
}

/// Высокоуровневый пайплайн OmniVoice.
pub struct OmniVoicePipeline {
    backbone: Backbone,
    text: TextFrontend,
    codec: CodecDecoder,
    codec_encoder: CodecEncoder,
    duration: DurationEstimator,
    cfg: OmniVoiceConfig,
    /// frame_rate кодека (токенов/с) для duration-фоллбэка авто-режима.
    frame_rate: usize,
    /// sample_rate кодека (для load_audio в voice-clone).
    sample_rate: usize,
    /// hop_length кодека (= ∏ downsampling_ratios) для clip ref-аудио.
    hop_length: usize,
    #[allow(dead_code)]
    device: Device,
}

impl OmniVoicePipeline {
    /// Собрать из `.syn`-бандла (компоненты `tensors:lm` / `tensors:codec`,
    /// файлы `config.json`, `tokenizer.json`, `audio_tokenizer/config.json`).
    pub fn from_syn(
        syn_path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_syn_quant(syn_path, device, dtype, None)
    }

    /// То же, но линейные слои LM квантуются при загрузке в `quant`
    /// (NVFP4/MXFP8/Sq) независимо от формата весов в бандле; `dtype` —
    /// точность счёта и остальных тензоров.
    pub fn from_syn_quant(
        syn_path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        quant: Option<DType>,
    ) -> Result<Self> {
        let bundle = Bundle::open(syn_path.as_ref())
            .map_err(|e| OmniVoiceError::Bundle(e.to_string()))?;

        let cfg = OmniVoiceConfig::from_bundle(&bundle)?;
        let higgs = HiggsAudioConfig::from_bundle(&bundle)?;

        let tok_bytes = bundle
            .read_file("tokenizer.json")
            .map_err(|e| {
                OmniVoiceError::Bundle(format!(
                    "tokenizer.json: {e} — бандл `{}` не похож на модель OmniVoice \
                     (выберите OmniVoice `.syn`, например omnivoice.syn)",
                    bundle.id()
                ))
            })?;
        let text = TextFrontend::from_tokenizer_bytes(
            &tok_bytes,
            cfg.num_audio_codebook,
            cfg.audio_mask_id as i64,
        )?;

        // lm: dedicated `tensors:lm` chunk (распакованный размер ~2.45ГБ).
        let lm_bytes = bundle
            .tensors_slice_named("lm")
            .map_err(|e| OmniVoiceError::Bundle(format!("tensors:lm chunk: {e}")))?;
        let manifest = synaptix_bundle::QuantManifest::read_from(&bundle);
        let host: &[&str] = if matches!(device, Device::Npu(_)) { &["llm.embed_tokens.weight", "audio_embeddings.weight"] } else { &[] };
        let mut lm = OmniVoiceLmWeights::load_bundle_bytes(lm_bytes, device, dtype, manifest.as_ref(), host)?;
        lm.quant_dtype = quant;

        // codec: dedicated `tensors:codec` chunk. ВНИМАНИЕ: некоторые omnivoice.syn
        // спакованы только с lm-компонентом (codec отсутствует как отдельный chunk;
        // legacy-fallback вернул бы lm-байты под codec → молча неверно). Требуем
        // именно `tensors:codec`; иначе понятная ошибка.
        let codec_bytes = bundle.tensors_slice_named("codec").map_err(|_| {
            OmniVoiceError::Load(
                "bundle has no `tensors:codec` chunk (lm-only pack); cannot build codec \
                 from .syn — use from_unpacked or repack with the codec component"
                    .into(),
            )
        })?;
        let codec_dev = if matches!(device, Device::Npu(_)) { Device::Cpu } else { device };
        let codec_dtype = if matches!(device, Device::Npu(_)) { DType::F32 } else { dtype };
        let codec_w = OmniVoiceCodecWeights::load_safetensors_bytes(codec_bytes, codec_dev, codec_dtype)?;

        Self::assemble(cfg, higgs, text, lm, codec_w, device)
    }

    /// Собрать из распакованного HF-снапшота (`dir/model.safetensors`,
    /// `dir/audio_tokenizer/model.safetensors`, `dir/{config,tokenizer}.json`,
    /// `dir/audio_tokenizer/config.json`). Удобно для гейта.
    pub fn from_unpacked(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = OmniVoiceConfig::from_json_bytes(
            &std::fs::read(dir.join("config.json")).map_err(err)?,
        )?;
        let higgs = HiggsAudioConfig::from_json_bytes(
            &std::fs::read(dir.join("audio_tokenizer/config.json")).map_err(err)?,
        )?;
        let text = TextFrontend::from_tokenizer_file(
            dir.join("tokenizer.json"),
            cfg.num_audio_codebook,
            cfg.audio_mask_id as i64,
        )?;
        let lm = OmniVoiceLmWeights::load_safetensors(
            dir.join("model.safetensors"),
            device,
            dtype,
        )?;
        let codec_w = OmniVoiceCodecWeights::load_safetensors(
            dir.join("audio_tokenizer/model.safetensors"),
            device,
            dtype,
        )?;
        Self::assemble(cfg, higgs, text, lm, codec_w, device)
    }

    fn assemble(
        cfg: OmniVoiceConfig,
        higgs: HiggsAudioConfig,
        text: TextFrontend,
        lm: OmniVoiceLmWeights,
        codec_w: OmniVoiceCodecWeights,
        device: Device,
    ) -> Result<Self> {
        // rope-capacity с запасом: style+text (десятки) + ref + target (тысячи на
        // длинных текстах); 8192 покрывает типичные одно-чанковые генерации.
        let backbone = Backbone::build(&cfg, &lm, 8192)?;
        let codec_w = if codec_w.dtype == DType::F32 { codec_w } else { codec_w.to_dtype(DType::F32)? };
        let codec = CodecDecoder::build(&higgs, &codec_w, cfg.num_audio_codebook)?;
        let codec_encoder = CodecEncoder::build(&higgs, &codec_w)?;
        // frame_rate = sample_rate / downsample_factor (HiggsAudioV2: 24000/320=75).
        let frame_rate = higgs.sample_rate / higgs.downsample_factor.max(1);
        let sample_rate = higgs.sample_rate;
        let hop_length = higgs.hop_length();
        Ok(Self {
            backbone,
            text,
            codec,
            codec_encoder,
            duration: DurationEstimator::new(),
            cfg,
            frame_rate,
            sample_rate,
            hop_length,
            device,
        })
    }

    pub fn prepare(&self, text: &str, lang: Option<&str>) -> Result<(Vec<i64>, Vec<u8>, usize)> {
        let target_len = self.duration.estimate_target_tokens(text, None, None, 1.0);
        let p = self.text.prepare_inference_inputs(text, target_len, None, None, lang, None, true)?;
        let ids = p.input_ids.to_dtype(DType::I64).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<i64>()).map_err(err)?;
        let mask = p.audio_mask.to_dtype(DType::U8).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<u8>()).map_err(err)?;
        let c = mask.len();
        Ok((ids, mask, c))
    }

    pub fn backbone(&self) -> &Backbone {
        &self.backbone
    }

    fn codes_for_codec(&self, codes: &Tensor) -> Result<Tensor> {
        if matches!(self.device, Device::Npu(_)) {
            codes.to_device(Device::Cpu).map_err(err)
        } else {
            Ok(codes.clone())
        }
    }

    pub fn config(&self) -> &OmniVoiceConfig {
        &self.cfg
    }

    pub fn frame_rate(&self) -> usize {
        self.frame_rate
    }

    /// Оценка числа target-токенов авто-режима для `text` (порт
    /// `_estimate_target_tokens` с fallback ref="Nice to meet you.", num_ref=25).
    pub fn estimate_target_tokens(&self, text: &str) -> usize {
        self.duration.estimate_target_tokens(text, None, None, 1.0)
    }

    /// e2e auto-режим (без ref): текст → волна 24кГц (`Vec<f32>`).
    /// `num_target_tokens` оценивается duration-estimator'ом.
    pub fn generate(
        &self,
        text: &str,
        gen: &OmniVoiceGenerationConfig,
    ) -> Result<Vec<f32>> {
        self.generate_styled(text, None, None, 1.0, gen)
    }

    /// e2e auto/design-режим (без ref) с опциональными `lang`/`instruct` и `speed`.
    /// `generate` = `generate_styled(text, None, None, 1.0, gen)`; поведение auto
    /// (lang=None, instruct=None, speed=1.0) идентично прежнему `generate`.
    pub fn generate_styled(
        &self,
        text: &str,
        lang: Option<&str>,
        instruct: Option<&str>,
        speed: f64,
        gen: &OmniVoiceGenerationConfig,
    ) -> Result<Vec<f32>> {
        let target_len =
            self.duration.estimate_target_tokens(text, None, None, speed);
        self.generate_styled_with_target(text, lang, instruct, target_len, gen)
    }

    /// e2e auto-режим с явным `target_len` (детерминированно убирает зависимость
    /// от duration-оценки — для гейта).
    pub fn generate_with_target(
        &self,
        text: &str,
        target_len: usize,
        gen: &OmniVoiceGenerationConfig,
    ) -> Result<Vec<f32>> {
        self.generate_styled_with_target(text, None, None, target_len, gen)
    }

    /// e2e auto/design с явным `target_len` и опциональными `lang`/`instruct`
    /// (прокидываются в text-frontend; instruct-валидацию не делаем).
    pub fn generate_styled_with_target(
        &self,
        text: &str,
        lang: Option<&str>,
        instruct: Option<&str>,
        target_len: usize,
        gen: &OmniVoiceGenerationConfig,
    ) -> Result<Vec<f32>> {
        // text-frontend → cond input_ids / audio_mask (auto/design: без ref).
        let prepared = self.text.prepare_inference_inputs(
            text,
            target_len,
            None,
            None,
            lang,
            instruct,
            gen.denoise,
        )?;

        // masked-decode → codes [8, T].
        let codes = generate_iterative(
            &self.backbone,
            &prepared.input_ids,
            &prepared.audio_mask,
            target_len,
            gen,
        )?;

        // codec.decode → wav [samples].
        let wav = self.codec.decode(&self.codes_for_codec(&codes)?)?;
        wav.to_dtype(DType::F32)
            .and_then(|w| w.flatten_all())
            .and_then(|w| w.to_vec1::<f32>())
            .map_err(err)
    }

    pub fn output_sample_rate(&self) -> usize {
        self.sample_rate
    }

    #[allow(clippy::too_many_arguments)]
    pub fn synthesize(
        &self,
        text: &str,
        prompt: Option<&VoiceClonePrompt>,
        lang: Option<&str>,
        instruct: Option<&str>,
        speed: f64,
        target_len: Option<usize>,
        gen: &OmniVoiceGenerationConfig,
    ) -> Result<Vec<f32>> {
        self.synthesize_codes(text, prompt, lang, instruct, speed, target_len, gen).map(|(wav, _)| wav)
    }

    /// Референс голоса из уже сгенерированной речи (`codes` — из
    /// [`Self::synthesize_codes`]): следующие фразы выйдут тем же голосом.
    pub fn prompt_from_generated(&self, codes: &Tensor, text: &str, wav: &[f32]) -> Result<VoiceClonePrompt> {
        let ref_audio_tokens = codes.to_device(Device::Cpu).and_then(|c| c.to_dtype(DType::I64)).map_err(err)?;
        Ok(VoiceClonePrompt { ref_audio_tokens, ref_text: crate::prompt::add_punctuation(text), ref_rms: crate::prompt::rms_f32(wav) })
    }

    /// Озвучка текста фразами одним голосом. `voice` — референс из
    /// предыдущих фраз: `None` — голос выбирается по `instruct` (или
    /// случайно) и запоминается, иначе фраза синтезируется как продолжение
    /// референса, а `instruct` уже не нужен (описание голоса в промпте
    /// модель порой читает вслух). Референс дополняется фразами до
    /// `max_ref_sec` секунд.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_continuing(
        &self,
        text: &str,
        voice: &mut Option<VoiceClonePrompt>,
        lang: Option<&str>,
        instruct: Option<&str>,
        speed: f64,
        gen: &OmniVoiceGenerationConfig,
        max_ref_sec: f32,
    ) -> Result<Vec<f32>> {
        let (wav, codes) = match voice.as_ref() {
            Some(p) => self.synthesize_codes(text, Some(p), lang, None, speed, None, gen)?,
            None => self.synthesize_codes(text, None, lang, instruct, speed, None, gen)?,
        };
        let part = self.prompt_from_generated(&codes, text, &wav)?;
        // Кадров в секунду — по самим кодам: `frame_rate` (24000/320 = 75) втрое
        // больше реальных 25 кадров/с кодов, и «12 с» референса были 36 с —
        // вход рос с каждой фразой, синтез замедлялся на любом устройстве.
        let sec = wav.len() as f32 / self.output_sample_rate() as f32;
        let t = codes.dims().last().copied().unwrap_or(0);
        let fps = if sec > 0.1 && t > 0 { t as f32 / sec } else { self.frame_rate as f32 };
        let max_tokens = (max_ref_sec.max(0.0) * fps) as usize;
        match voice {
            Some(p) => {
                p.extend(&part, max_tokens)?;
            }
            None => *voice = Some(part),
        }
        Ok(wav)
    }

    /// [`Self::synthesize`], возвращающий и коды кодека `[C, T]` сгенерированной
    /// речи — для референса голоса ([`Self::prompt_from_generated`]).
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_codes(
        &self,
        text: &str,
        prompt: Option<&VoiceClonePrompt>,
        lang: Option<&str>,
        instruct: Option<&str>,
        speed: f64,
        target_len: Option<usize>,
        gen: &OmniVoiceGenerationConfig,
    ) -> Result<(Vec<f32>, Tensor)> {
        let target_len = target_len.unwrap_or_else(|| match prompt {
            Some(p) => self.estimate_target_tokens_clone(text, p, speed),
            None => self.duration.estimate_target_tokens(text, None, None, speed),
        });
        let prepared = self.text.prepare_inference_inputs(
            text,
            target_len,
            prompt.map(|p| p.ref_text.as_str()),
            prompt.map(|p| &p.ref_audio_tokens),
            lang,
            instruct,
            gen.denoise,
        )?;
        let timing = std::env::var_os("SYNAPTIX_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let codes = generate_iterative(&self.backbone, &prepared.input_ids, &prepared.audio_mask, target_len, gen)?;
        let t1 = std::time::Instant::now();
        let wav = self.codec.decode(&self.codes_for_codec(&codes)?)?;
        let out = wav.to_dtype(DType::F32).and_then(|w| w.flatten_all()).and_then(|w| w.to_vec1::<f32>()).map_err(err)?;
        if timing {
            eprintln!("omnivoice: декод {:.2} с ({} шагов, T={target_len}), кодек {:.2} с", (t1 - t0).as_secs_f32(), gen.num_step, t1.elapsed().as_secs_f32());
        }
        Ok((out, codes))
    }

    /// Доступ к encode-пути нейро-кодека (ref-аудио → коды). Для гейтов/дебага.
    pub fn codec_encoder(&self) -> &CodecEncoder {
        &self.codec_encoder
    }

    /// `create_voice_clone_prompt`: ref-аудио (путь) + ref-текст → переиспользуемый
    /// `VoiceClonePrompt` (ref_audio_tokens + ref_text+punct + ref_rms). Порт
    /// `OmniVoice.create_voice_clone_prompt` (preprocess_prompt=true:
    /// remove_silence 200/100/200, без trim т.к. ref_text задан, add_punctuation).
    pub fn create_voice_clone_prompt(
        &self,
        ref_audio_path: impl AsRef<Path>,
        ref_text: &str,
    ) -> Result<VoiceClonePrompt> {
        create_voice_clone_prompt(
            &self.codec_encoder,
            ref_audio_path,
            ref_text,
            self.sample_rate,
            self.hop_length,
            true,
        )
    }

    /// Оценка числа target-токенов voice-clone (порт `_estimate_target_tokens` с
    /// реальным ref_text/num_ref из промпта). `speed` > 1.0 — быстрее (короче),
    /// < 1.0 — медленнее; upstream передаёт его и в clone-путь.
    pub fn estimate_target_tokens_clone(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        speed: f64,
    ) -> usize {
        let num_ref = prompt.ref_audio_tokens.dims()[prompt.ref_audio_tokens.dims().len() - 1];
        self.duration
            .estimate_target_tokens(text, Some(&prompt.ref_text), Some(num_ref), speed)
    }

    /// Voice-clone e2e: текст → волна 24кГц с клон-голосом из `prompt`.
    /// `target_len` оценивается duration-estimator'ом с учётом ref.
    /// = `generate_clone_styled(text, prompt, 1.0, gen)`.
    pub fn generate_clone(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        gen: &OmniVoiceGenerationConfig,
    ) -> Result<Vec<f32>> {
        self.generate_clone_styled(text, prompt, 1.0, gen)
    }

    /// Voice-clone e2e со скоростью речи: `speed` масштабирует duration-оценку
    /// (est /= speed), как `generate_styled` в auto/design-режиме.
    pub fn generate_clone_styled(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        speed: f64,
        gen: &OmniVoiceGenerationConfig,
    ) -> Result<Vec<f32>> {
        let target_len = self.estimate_target_tokens_clone(text, prompt, speed);
        self.generate_clone_with_target(
            text,
            &prompt.ref_audio_tokens,
            &prompt.ref_text,
            target_len,
            gen,
        )
    }

    /// Voice-clone с явными ref_audio_tokens / ref_text / target_len (детерминизм
    /// для гейта; ref_tokens могут приходить из дампа). Порт ref-ветки
    /// `_prepare_inference_inputs` (denoise + ref-токены в cond) → masked_decode →
    /// codec.decode. Возврат wav [samples] (БЕЗ post-process).
    pub fn generate_clone_with_target(
        &self,
        text: &str,
        ref_audio_tokens: &Tensor,
        ref_text: &str,
        target_len: usize,
        gen: &OmniVoiceGenerationConfig,
    ) -> Result<Vec<f32>> {
        // text-frontend ref-путь: ref_text + ref_audio_tokens + denoise.
        let prepared = self.text.prepare_inference_inputs(
            text,
            target_len,
            Some(ref_text),
            Some(ref_audio_tokens),
            None,
            None,
            gen.denoise,
        )?;

        // masked-decode (cond=style+text+ref+target; uncond=target-хвост) → [8,T].
        let codes = generate_iterative(
            &self.backbone,
            &prepared.input_ids,
            &prepared.audio_mask,
            target_len,
            gen,
        )?;

        // codec.decode → wav.
        let wav = self.codec.decode(&self.codes_for_codec(&codes)?)?;
        wav.to_dtype(DType::F32)
            .and_then(|w| w.flatten_all())
            .and_then(|w| w.to_vec1::<f32>())
            .map_err(err)
    }
}
