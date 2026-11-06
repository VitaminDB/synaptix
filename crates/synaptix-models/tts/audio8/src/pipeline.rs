//! Синтез Audio8: промпт → цикл кадров (медленный шаг → семантика с RAS → быстрый
//! AR по 9 кодбукам) → кодек. Порядок сэмплинга и расход ГПСЧ — как у
//! `ArkttsModel.generate` upstream (батч 1).

use std::path::Path;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;

use crate::codec::Codec;
use crate::config::Audio8Config;
use crate::dualar::DualAr;
use crate::loader::Checkpoint;
use crate::sample::{argmax, processed, sample, CounterRng};
use crate::text::TextTokenizer;
use crate::{Audio8Error, Result, FRAME_SAMPLES, SAMPLE_RATE};

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    /// Предел кадров (≈ 21.5 кадра на секунду звука).
    pub max_frames: usize,
    pub seed: u64,
    /// Жадный выбор (без сэмплинга) — для сверки с эталоном.
    pub greedy: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self { temperature: 0.7, top_p: 0.9, top_k: 50, max_frames: 1024, seed: 0, greedy: false }
    }
}

/// Образец голоса: коды кодека и его расшифровка.
#[derive(Debug, Clone)]
pub struct VoicePrompt {
    pub codes: Vec<Vec<u32>>,
    pub text: String,
}

impl VoicePrompt {
    pub fn frames(&self) -> usize {
        self.codes.first().map_or(0, |c| c.len())
    }
}

/// Замеры последнего синтеза.
#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub prompt_tokens: usize,
    pub frames: usize,
    pub finished: bool,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub codec_s: f64,
}

pub struct Audio8Pipeline {
    ck: Checkpoint,
    pub model: DualAr,
    pub codec: Codec,
    tok: TextTokenizer,
    pub config: Audio8Config,
    defaults: GenerateOptions,
    pub device: Device,
    last: std::sync::Mutex<Timings>,
}

impl Audio8Pipeline {
    /// `compute` — счётный тип LM (BF16 на карте, F32 на CPU); `quant` —
    /// квант линейных слоёв медленного AR на лету (NVFP4/MXFP8/Sq), если
    /// в бандле они плотные.
    pub fn from_syn_quant(path: &Path, device: Device, compute: DType, quant: Option<DType>) -> Result<Self> {
        let compute = if device.is_cuda() { compute } else { DType::F32 };
        let ck = Checkpoint::open(path, device, compute, quant)?;
        let tok = TextTokenizer::from_bytes(&ck.read_file("tokenizer.json")?)?;
        let model = DualAr::load(&ck)?;
        let eager = std::env::var("SYN_AUDIO8_ENCODER").as_deref() == Ok("eager");
        let codec = Codec::load(&ck, eager)?;
        let g = &ck.gen;
        let defaults = GenerateOptions { temperature: g.temperature, top_p: g.top_p, top_k: g.top_k, ..GenerateOptions::default() };
        Ok(Self { model, codec, tok, config: ck.config.clone(), defaults, device, last: Default::default(), ck })
    }

    pub fn from_syn(path: &Path, device: Device) -> Result<Self> {
        Self::from_syn_quant(path, device, if device.is_cuda() { DType::BF16 } else { DType::F32 }, None)
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Умолчания сэмплинга модели (`generation_config.json`).
    pub fn default_options(&self) -> GenerateOptions {
        self.defaults.clone()
    }

    pub fn last_timings(&self) -> Timings {
        self.last.lock().map(|t| t.clone()).unwrap_or_default()
    }

    /// Образец голоса из PCM (моно, любая частота) и его точной расшифровки.
    pub fn voice_prompt(&self, pcm: &[f32], sample_rate: u32, text: &str) -> Result<VoicePrompt> {
        let pcm = if sample_rate != SAMPLE_RATE {
            synaptix_audio::resample::resample(pcm, sample_rate, SAMPLE_RATE).map_err(|e| Audio8Error::Audio(e.to_string()))?
        } else {
            pcm.to_vec()
        };
        self.codec.ensure_encoder(&self.ck)?;
        let codes = self.codec.encode(&pcm)?;
        Ok(VoicePrompt { codes, text: text.to_string() })
    }

    /// Строки промпта `[1 + num_codebooks][T]`.
    pub fn prompt_rows(&self, text: &str, voice: Option<&VoicePrompt>) -> Result<Vec<Vec<u32>>> {
        let c = &self.config;
        let p = self.tok.prompt(text, voice.map(|v| v.text.as_str()))?;
        let nc = c.num_codebooks;
        let mut rows = vec![Vec::new(); nc + 1];
        let push_text = |rows: &mut Vec<Vec<u32>>, ids: &[u32]| {
            for &id in ids {
                rows[0].push(id);
                for r in rows.iter_mut().skip(1) {
                    r.push(0);
                }
            }
        };
        push_text(&mut rows, &p.prefix);
        if let Some(v) = voice {
            if v.codes.len() != nc {
                return Err(Audio8Error::Other(format!("образец: {} кодбуков вместо {nc}", v.codes.len())));
            }
            for t in 0..v.frames() {
                rows[0].push(v.codes[0][t] + c.semantic_begin_id);
                for (i, r) in rows.iter_mut().skip(1).enumerate() {
                    r.push(v.codes[i][t]);
                }
            }
        }
        push_text(&mut rows, &p.suffix);
        Ok(rows)
    }

    fn sample_semantic(&self, logits: &[f32], previous: Option<&[u32]>, o: &GenerateOptions, rng: &mut CounterRng) -> u32 {
        let c = &self.config;
        let n_sem = c.semantic_count();
        let to_id = |i: usize| if i < n_sem { c.semantic_begin_id + i as u32 } else { c.eos_token_id };
        let regular = processed(logits, o.top_k, o.top_p, o.temperature);
        if o.greedy {
            return to_id(argmax(&regular));
        }
        let vocab = c.vocab_size as u64;
        let index = |i: usize| to_id(i) as u64;
        let normal = to_id(sample(&regular, &index, vocab, rng));
        let high_scores = processed(logits, o.top_k, c.ras_top_p, c.ras_temperature);
        let high = to_id(sample(&high_scores, &index, vocab, rng));
        match previous {
            None => normal,
            Some(prev) => {
                let semantic = normal >= c.semantic_begin_id && normal <= c.semantic_end_id;
                if semantic && prev.contains(&normal) {
                    high
                } else {
                    normal
                }
            }
        }
    }

    fn codebooks(&self, fast_in: &synaptix_core::tensor::Tensor, semantic: u32, kv: &mut crate::dualar::Kv, o: &GenerateOptions, rng: &mut CounterRng) -> Result<Vec<u32>> {
        let c = &self.config;
        self.model.fast_prime(fast_in, kv)?;
        let first = (semantic as i64 - c.semantic_begin_id as i64).clamp(0, c.codebook_size as i64 - 1) as u32;
        let mut out = Vec::with_capacity(c.num_codebooks);
        out.push(first);
        let mut x = self.model.fast_embed(first)?;
        for pos in 1..c.num_codebooks {
            let logits = self.model.fast_step(&x, kv, pos)?;
            let s = processed(&logits, o.top_k, o.top_p, o.temperature);
            let cur = if o.greedy { argmax(&s) } else { sample(&s, &|i| i as u64, logits.len() as u64, rng) } as u32;
            out.push(cur);
            x = self.model.fast_embed(cur)?;
        }
        Ok(out)
    }

    /// Коды кадров `[num_codebooks][F]` без кодека.
    pub fn generate_codes(&self, text: &str, voice: Option<&VoicePrompt>, o: &GenerateOptions) -> Result<Vec<Vec<u32>>> {
        let c = &self.config;
        let t0 = Instant::now();
        let rows = self.prompt_rows(text, voice)?;
        let width = rows[0].len();
        if width >= c.max_seq_len {
            return Err(Audio8Error::Other(format!("промпт {width} токенов ≥ контекста {}", c.max_seq_len)));
        }
        let max_new = o.max_frames.min(c.max_seq_len - width);
        let mut kv = self.model.slow.new_cache(c.max_seq_len)?;
        let mut fkv = self.model.fast.new_cache(c.num_codebooks)?;
        let (mut logits, mut fast_in) = self.model.slow_step(&rows, &mut kv, 0)?;
        let t_prefill = t0.elapsed().as_secs_f64();
        let mut rng = CounterRng::new(o.seed);
        let mut previous: Option<Vec<u32>> = None;
        let mut frames: Vec<Vec<u32>> = vec![Vec::new(); c.num_codebooks];
        let mut finished = false;
        for step in 0..max_new {
            let semantic = self.sample_semantic(&logits, previous.as_deref(), o, &mut rng);
            let cbs = self.codebooks(&fast_in, semantic, &mut fkv, o, &mut rng)?;
            if semantic == c.eos_token_id {
                finished = true;
                break;
            }
            for (i, v) in cbs.iter().enumerate() {
                frames[i].push(*v);
            }
            match previous.as_mut() {
                None => previous = Some(vec![0; c.ras_window_size]),
                Some(p) => {
                    p.rotate_left(1);
                    *p.last_mut().unwrap() = semantic;
                }
            }
            if step + 1 == max_new {
                break;
            }
            let mut col = Vec::with_capacity(c.num_codebooks + 1);
            col.push(vec![semantic]);
            col.extend(cbs.iter().map(|&v| vec![v]));
            let (l, f) = self.model.slow_step(&col, &mut kv, width + step)?;
            logits = l;
            fast_in = f;
        }
        if let Ok(mut t) = self.last.lock() {
            *t = Timings {
                prompt_tokens: width,
                frames: frames[0].len(),
                finished,
                prefill_s: t_prefill,
                decode_s: t0.elapsed().as_secs_f64() - t_prefill,
                codec_s: 0.0,
            };
        }
        Ok(frames)
    }

    /// Текст → PCM 44.1 кГц.
    pub fn synthesize(&self, text: &str, voice: Option<&VoicePrompt>, o: &GenerateOptions) -> Result<Vec<f32>> {
        let codes = self.generate_codes(text, voice, o)?;
        let t = Instant::now();
        let pcm = self.codec.decode(&codes)?;
        if let Ok(mut l) = self.last.lock() {
            l.codec_s = t.elapsed().as_secs_f64();
        }
        debug_assert_eq!(pcm.len(), codes[0].len() * FRAME_SAMPLES);
        Ok(pcm)
    }
}
