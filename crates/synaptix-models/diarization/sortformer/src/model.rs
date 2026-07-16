//! `SortformerModel` — mel + FastConformer + Sortformer-head (BATCH / full-attention).
//!
//! Цепочка (NeMo `SortformerEncLabelModel.forward`, non-streaming):
//!   process_signal: audio ·= 1/(max(audio)+1e-3) → mel → encoder → transpose(1,2)
//!   → encoder_proj → transformer-head → forward_speaker_sigmoids → preds (1,T',n_spk).
//! ⚠ Это full-attention путь. Для клипов ≤~15с (1 чанк, eval) ≡ streaming; для длинных
//! записей нужен streaming-режим (см. `streaming.rs`, фаза 2).
//!
//! `forward_stages` отдаёт промежуточные тензоры (mel/encoder_out/emb_seq/trans/preds)
//! для постадийного bit-exact гейта против NeMo-дампа.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::config::SortformerConfig;
use crate::encoder::FastConformer;
use crate::head::SortformerHead;
use crate::loader::SortformerWeights;
use crate::mel::MelFrontend;
use crate::Result;

/// Промежуточные активации для постадийного гейта.
pub struct Stages {
    pub mel: Tensor,         // (1,128,T)
    pub encoder_out: Tensor, // (1,512,T')
    pub emb_seq: Tensor,     // (1,T',192)
    pub trans: Tensor,       // (1,T',192)
    pub preds: Tensor,       // (1,T',n_spk)
}

pub struct SortformerModel {
    mel: MelFrontend,
    encoder: FastConformer,
    head: SortformerHead,
    config: SortformerConfig,
    device: Device,
    dtype: DType,
}

const NORM_EPS: f32 = 1e-3;

impl SortformerModel {
    pub fn load(w: &SortformerWeights) -> Result<Self> {
        let n_mels = w.config.preprocessor.n_mels;
        let n_freqs = w.config.preprocessor.n_fft / 2 + 1;
        let window =
            w.get_dtype("preprocessor.featurizer.window", DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let fb = w.get_dtype("preprocessor.featurizer.fb", DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mel = MelFrontend::nemo_v21(window, fb, n_mels, n_freqs);
        let encoder = FastConformer::load(w)?;
        let head = SortformerHead::load(w)?;
        Ok(Self { mel, encoder, head, config: w.config.clone(), device: w.device, dtype: w.dtype })
    }

    /// NeMo `process_signal` (non-streaming) → mel-тензор `[1,n_mels,T]` в compute-dtype.
    fn mel_tensor(&self, audio: &[f32]) -> Result<Tensor> {
        let max = audio.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = 1.0f32 / (max + NORM_EPS);
        let normed: Vec<f32> = audio.iter().map(|&v| v * scale).collect();
        let (flat, t) = self.mel.forward(&normed);
        let mel = Tensor::from_vec(flat, (1, self.config.preprocessor.n_mels, t), self.device)?;
        Ok(mel.to_dtype(self.dtype)?)
    }

    /// Полный forward с промежуточными активациями (для гейта).
    pub fn forward_stages(&self, audio: &[f32]) -> Result<Stages> {
        let mel = self.mel_tensor(audio)?;
        let encoder_out = self.encoder.forward(&mel)?; // (1,512,T')
        let emb_t = encoder_out.transpose(1, 2)?.contiguous()?; // (1,T',512)
        let emb_seq = self.head.project(&emb_t)?; // (1,T',192)
        let trans = self.head.transformer(&emb_seq)?; // (1,T',192)
        let preds = self.head.sigmoids(&trans)?; // (1,T',n_spk)
        Ok(Stages { mel, encoder_out, emb_seq, trans, preds })
    }

    fn stream_cfg(&self) -> crate::streaming::StreamCfg {
        let s = &self.config.streaming;
        crate::streaming::StreamCfg {
            spkcache_len: s.spkcache_len,
            fifo_len: s.fifo_len,
            chunk_len: s.chunk_len,
            subsampling_factor: self.config.encoder.subsampling_factor,
            chunk_left_context: s.chunk_left_context,
            chunk_right_context: s.chunk_right_context,
            spkcache_sil_frames_per_spk: s.spkcache_sil_frames_per_spk,
            spkcache_update_period: s.spkcache_update_period,
            n_spk: self.config.max_speakers,
            d_model: self.config.encoder.d_model,
            pred_score_threshold: s.pred_score_threshold,
            scores_boost_latest: s.scores_boost_latest,
            sil_threshold: s.sil_threshold,
            strong_boost_rate: s.strong_boost_rate,
            weak_boost_rate: s.weak_boost_rate,
            min_pos_scores_rate: s.min_pos_scores_rate,
            max_index: s.max_index,
        }
    }

    /// Streaming-диаризация (NeMo `forward_streaming`): чанки по chunk_len·8 mel-кадров,
    /// spkcache/fifo + compress. Для ДЛИННЫХ записей (точное воспроизведение NeMo-поведения).
    /// PCM 16кГц → per-speaker probs `[1,T',n_spk]`.
    pub fn diarize_pcm_streaming(&self, audio: &[f32]) -> Result<Tensor> {
        use crate::streaming::{streaming_update, StreamState};
        let cfg = self.stream_cfg();
        let mel = self.mel_tensor(audio)?; // (1,128,T)
        let t_mel = mel.dims()[2];
        let sf = cfg.subsampling_factor;
        let chunk_mel = cfg.chunk_len * sf;

        let mut state = StreamState::new(self.device, self.dtype, cfg.d_model)?;
        let mut all_preds: Vec<Tensor> = Vec::new();

        let mut stt = 0usize;
        while stt < t_mel {
            let lc_off = (cfg.chunk_left_context * sf).min(stt);
            let end = (stt + chunk_mel).min(t_mel);
            let rc_off = (cfg.chunk_right_context * sf).min(t_mel - end);
            let beg = stt - lc_off;
            let len = end + rc_off - beg;
            let chunk_mel_t = mel.narrow(2, beg, len)?.contiguous()?; // (1,128,len)

            let chunk_embs = self.encoder.pre_encode(&chunk_mel_t)?; // (1,cf,512)

            // concat [spkcache, chunk_embs] (fifo_len=0 → fifo пуст; пустой spkcache пропускаем).
            let combined = if state.spkcache.dims()[1] > 0 {
                Tensor::cat(&[&state.spkcache, &chunk_embs], 1)?
            } else {
                chunk_embs.clone()
            };

            let enc = self.encoder.encode_bypass(&combined)?; // (1,512,L)
            let emb_t = enc.transpose(1, 2)?.contiguous()?; // (1,L,512)
            let emb_seq = self.head.project(&emb_t)?;
            let trans = self.head.transformer(&emb_seq)?;
            let preds = self.head.sigmoids(&trans)?; // (1,L,n_spk)

            let lc = ((lc_off as f32) / sf as f32).round() as usize;
            let rc = (rc_off as f32 / sf as f32).ceil() as usize;
            let chunk_preds = streaming_update(&mut state, &chunk_embs, &preds, lc, rc, &cfg)?;
            all_preds.push(chunk_preds);
            stt = end;
        }

        let total = if all_preds.len() == 1 {
            all_preds.pop().unwrap()
        } else {
            let refs: Vec<&Tensor> = all_preds.iter().collect();
            Tensor::cat(&refs, 1)?
        };
        // trim до ceil(T_mel / subsampling_factor) (NeMo discard padding).
        let n_frames = t_mel.div_ceil(sf).min(total.dims()[1]);
        Ok(total.narrow(1, 0, n_frames)?.contiguous()?)
    }

    /// Постадийная отладка энкодера (preenc/enc_l0/enc_l8/enc_l16/final) для локализации.
    pub fn encoder_debug(&self, audio: &[f32]) -> Result<Vec<(String, Tensor)>> {
        let mel = self.mel_tensor(audio)?;
        self.encoder.forward_debug(&mel)
    }

    /// PCM 16кГц моно → per-speaker probs `[1,T',n_spk]` @ frame_rate_hz.
    pub fn diarize_pcm(&self, audio: &[f32]) -> Result<Tensor> {
        let mel = self.mel_tensor(audio)?;
        let encoder_out = self.encoder.forward(&mel)?;
        let emb_t = encoder_out.transpose(1, 2)?.contiguous()?;
        self.head.forward(&emb_t)
    }

    pub fn config(&self) -> &SortformerConfig {
        &self.config
    }
    pub fn device(&self) -> Device {
        self.device
    }
}
