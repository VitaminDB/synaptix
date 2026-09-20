//! AR-ветка YuE2: партитура (ABC) и семантические токены.
//!
//! Слои AR-пути лежат в бандле в обычной HF-раскладке Llama, поэтому ветку
//! ведёт общий [`DecoderModel`] движка — с его префиллом, flash-вниманием,
//! квантованием весов, оффлоадом блоков и CUDA-графом на декоде. NAR-двойники
//! (`nar_*`) ему не нужны и не запрашиваются.
//!
//! Две фазы идут по одной сессии: после партитуры в тот же KV дописываются
//! `</abc>` и `<music>`, и генерация продолжается. Повторный префилл текста и
//! партитуры (а это тысячи токенов) не делается.

use std::path::Path;
use std::time::Instant;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_llm_common::{
    Activation, DecodeState, DecoderConfig, DecoderModel, KvCache, LayerKind, NormGain, RopeSpec,
};
use synaptix_ops::rng::Philox4x32;

use crate::config::Yue2Config;
use crate::loader::{read_bundle_file, BundleWeightSource, CompLoader};
use crate::protocol::{Sampling, ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END};
use crate::YueError;

/// Сколько токенов префилла уходит в модель за раз.
const PREFILL_CHUNK: usize = 1024;

/// Какую фазу генерируем — от этого зависит, какие токены вообще разрешены.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Партитура: обычный текстовый словарь плюс `</abc>`.
    Abc,
    /// Музыка: только семантические токены плюс `</music>`.
    Semantic,
}

impl Phase {
    pub fn end_token(self) -> u32 {
        match self {
            Phase::Abc => ABC_END,
            Phase::Semantic => MUSIC_END,
        }
    }

    /// Диапазон разрешённых токенов `[начало, длина)` (без конца фазы).
    fn allowed_range(self) -> (u32, u32) {
        match self {
            Phase::Abc => (0, EOD),
            Phase::Semantic => (CODEC_OFFSET, CODEC_SIZE),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Abc => "abc",
            Phase::Semantic => "semantic",
        }
    }
}

/// Замеры одной фазы.
#[derive(Debug, Clone, Default)]
pub struct PhaseTiming {
    pub prefill_seconds: f64,
    pub seconds: f64,
    pub output_tokens: usize,
    pub prefix_tokens: usize,
    pub tokens_per_second: f64,
    /// Ветки CFG: 1 — без guidance, 2 — с отрицательным префиксом.
    pub cfg_branches: usize,
    /// `true` — декод шёл захваченным CUDA-графом.
    pub cuda_graph: bool,
}

pub struct Yue2Ar {
    pub model: DecoderModel,
    pub config: Yue2Config,
    pub device: Device,
    pub compute: DType,
}

fn to_decoder_config(cfg: &Yue2Config) -> DecoderConfig {
    DecoderConfig {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        max_position_embeddings: cfg.max_position_embeddings,
        rms_norm_eps: cfg.rms_norm_eps,
        norm_gain: NormGain::Plain,
        activation: Activation::Silu,
        sandwich_norms: false,
        post_norm_eps: None,
        // У YuE2 есть q_norm/k_norm на голове — как у Qwen3.
        qk_norm: true,
        attn_output_gate: false,
        attn_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        embed_scale: None,
        embed_rms_norm: false,
        logit_scale: None,
        logit_softcap: None,
        rope_global: RopeSpec { theta: cfg.rope_theta, rotary_dim: cfg.head_dim, scaled_freqs: None },
        rope_local: None,
        sliding_window: None,
        sliding_window_pattern: 0,
        layer_kinds: vec![LayerKind::Full; cfg.num_hidden_layers],
        linear: None,
        tie_word_embeddings: false,
        bos_token_id: None,
        eos_token_ids: vec![MUSIC_END],
        ext: None,
    }
}

impl Yue2Ar {
    /// Открыть бандл `yue2-3b.syn`. `quant_w` — квант весов внимания и MLP
    /// (`None` — плотно в `compute`).
    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        compute: DType,
        quant_w: Option<DType>,
        rope_capacity: usize,
    ) -> Result<Self, YueError> {
        let path = path.as_ref();
        let loader = CompLoader::open(path, None, device)?;
        let src = BundleWeightSource::new(loader);
        let config = match read_bundle_file(path, "config.json") {
            Ok(bytes) => Yue2Config::from_json(&bytes)?,
            Err(_) => Yue2Config::default(),
        };
        let dcfg = to_decoder_config(&config);
        let w = quant_w.unwrap_or(compute);
        let model = DecoderModel::build_auto(
            &dcfg,
            &src,
            device,
            compute,
            w,
            w,
            compute,
            compute,
            rope_capacity.max(config.max_position_embeddings),
        )
        .map_err(|e| YueError::Load(e.to_string()))?
        // KV держим плотным: NAR-ветка читает его как контекст своих чанков,
        // а квантованный кэш умеет читать только flash-ядро.
        .with_kv_cache_dtype(compute);
        Ok(Self { model, config, device, compute })
    }

    pub fn make_kv(&self, max_seq: usize) -> Result<KvCache, YueError> {
        self.model
            .make_kv_cache(1, max_seq)
            .map_err(|e| YueError::Other(e.to_string()))
    }
}

/// Сессия AR-генерации: KV-кэш плюс логиты последней позиции.
pub struct ArSession<'a> {
    pub ar: &'a Yue2Ar,
    kv: KvCache,
    /// Сколько токенов уже в кэше.
    pos: usize,
    logits: Option<Tensor>,
    decode: Option<DecodeState>,
    prefill_seconds: f64,
}

impl<'a> ArSession<'a> {
    pub fn new(ar: &'a Yue2Ar, max_seq: usize) -> Result<Self, YueError> {
        Ok(Self {
            kv: ar.make_kv(max_seq)?,
            ar,
            pos: 0,
            logits: None,
            decode: None,
            prefill_seconds: 0.0,
        })
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    /// Секунды последнего префилла.
    pub fn prefill_seconds(&self) -> f64 {
        self.prefill_seconds
    }

    /// Дописать токены в кэш (префилл). Логиты после этого — от последнего.
    pub fn feed(&mut self, ids: &[u32]) -> Result<(), YueError> {
        if ids.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        for chunk in ids.chunks(PREFILL_CHUNK) {
            let t = Tensor::from_vec(chunk.to_vec(), vec![1usize, chunk.len()], self.ar.device)?;
            let logits = self
                .ar
                .model
                .forward(&t, &mut self.kv)
                .map_err(|e| YueError::Other(e.to_string()))?;
            self.logits = Some(logits);
            self.pos += chunk.len();
        }
        self.prefill_seconds = started.elapsed().as_secs_f64();
        Ok(())
    }

    /// Шаг декода без графа.
    fn eager_step(&mut self, token: u32) -> Result<(), YueError> {
        let t = Tensor::from_vec(vec![token], vec![1usize, 1usize], self.ar.device)?;
        let logits = self
            .ar
            .model
            .forward(&t, &mut self.kv)
            .map_err(|e| YueError::Other(e.to_string()))?;
        self.pos += 1;
        self.logits = Some(logits);
        Ok(())
    }

    /// Логиты последней позиции — для сверок и диагностики.
    pub fn logits_snapshot(&self) -> Option<Tensor> {
        self.logits.clone()
    }

    fn logits(&self) -> Result<&Tensor, YueError> {
        self.logits
            .as_ref()
            .ok_or_else(|| YueError::Other("логитов ещё нет — сессия не получила префикса".into()))
    }

    /// Отдать KV-кэш наружу (NAR-ветка переиспользует его как AR-контекст).
    pub fn into_kv(self) -> KvCache {
        self.kv
    }

    /// Готова ли модель к захвату графа декода (все блоки на карте, CUDA).
    fn graph_ready(&self) -> Option<usize> {
        match self.ar.device {
            Device::Cuda(ord) if self.ar.model.graph_decode_ready() => Some(ord),
            _ => None,
        }
    }
}

/// Логиты фазы: значения разрешённого диапазона плюс токен конца фазы.
fn phase_logits(logits: &Tensor, phase: Phase) -> Result<(Vec<f32>, f32), YueError> {
    let flat = if logits.rank() == 1 { logits.clone() } else { logits.flatten_all()? };
    let (start, len) = phase.allowed_range();
    // `narrow` отдаёт вид; при F32-логитах `to_dtype` — no-op, и вид остался бы
    // нераскладанным (`to_vec1` тогда отвечает NonContiguous).
    let body: Vec<f32> = flat
        .narrow(0, start as usize, len as usize)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .to_vec1()?;
    let end: Vec<f32> = flat
        .narrow(0, phase.end_token() as usize, 1)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .to_vec1()?;
    Ok((body, end[0]))
}

/// Штраф за повтор — частотный и оконный: токен, встретившийся в последних
/// `penalty_window` шагах `n` раз, получает `penalty^n` (отрицательный логит
/// умножается, положительный делится).
fn penalised(logit: f32, freq: u32, penalty: f32) -> f32 {
    if freq == 0 || penalty == 1.0 {
        return logit;
    }
    let alpha = penalty.powi(freq as i32);
    if logit < 0.0 {
        logit * alpha
    } else {
        logit / alpha
    }
}

/// Выбрать следующий токен: маска фазы → штраф → температура → top-k → top-p.
#[allow(clippy::too_many_arguments)]
fn sample_token(
    cond: &Tensor,
    uncond: Option<&Tensor>,
    cfg_scale: f32,
    phase: Phase,
    sampling: &Sampling,
    history: &[u32],
    step: usize,
    rng: &mut Philox4x32,
) -> Result<u32, YueError> {
    let (mut body, mut end_logit) = phase_logits(cond, phase)?;
    if let Some(u) = uncond {
        let (ubody, uend) = phase_logits(u, phase)?;
        if ubody.len() != body.len() {
            return Err(YueError::Other("ветки CFG разной ширины".into()));
        }
        for (c, u) in body.iter_mut().zip(ubody.iter()) {
            *c = u + cfg_scale * (*c - u);
        }
        end_logit = uend + cfg_scale * (end_logit - uend);
    }
    let (start, _) = phase.allowed_range();
    let end_token = phase.end_token();

    // Частоты по окну истории.
    let window = history.len().saturating_sub(sampling.penalty_window);
    let mut freq: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &t in &history[window..] {
        *freq.entry(t).or_insert(0) += 1;
    }

    let mut cand: Vec<(u32, f32)> = Vec::with_capacity(body.len() + 1);
    for (i, &v) in body.iter().enumerate() {
        let id = start + i as u32;
        let f = freq.get(&id).copied().unwrap_or(0);
        cand.push((id, penalised(v, f, sampling.repetition_penalty)));
    }
    // До `min_tokens` фазу закончить нельзя — эталон ставит её концу −inf.
    if step >= sampling.min_tokens {
        let f = freq.get(&end_token).copied().unwrap_or(0);
        cand.push((end_token, penalised(end_logit, f, sampling.repetition_penalty)));
    }
    if sampling.temperature == 0.0 {
        let mut best = cand[0];
        for &c in cand.iter() {
            if c.1 > best.1 {
                best = c;
            }
        }
        return Ok(best.0);
    }
    if (sampling.temperature - 1.0).abs() > f32::EPSILON {
        let inv = 1.0 / sampling.temperature;
        for c in cand.iter_mut() {
            c.1 *= inv;
        }
    }
    if sampling.top_k < cand.len() {
        cand.select_nth_unstable_by(sampling.top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        cand.truncate(sampling.top_k);
    }
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if sampling.top_p < 1.0 {
        let max = cand[0].1;
        let exps: Vec<f32> = cand.iter().map(|&(_, v)| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>().max(1e-30);
        let mut cumsum = 0.0f32;
        let mut keep = cand.len();
        for (i, e) in exps.iter().enumerate() {
            // Отбрасывается то, что выходит за top_p ДО прибавления своей
            // доли; первый кандидат остаётся всегда.
            if i > 0 && cumsum > sampling.top_p {
                keep = i;
                break;
            }
            cumsum += e / sum;
        }
        cand.truncate(keep.max(1));
    }
    if cand.len() == 1 {
        return Ok(cand[0].0);
    }
    let max = cand[0].1;
    let exps: Vec<f32> = cand.iter().map(|&(_, v)| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-30);
    let u = rng.next_f32_uniform();
    let mut cumsum = 0.0f32;
    for (i, e) in exps.iter().enumerate() {
        cumsum += e / sum;
        if u < cumsum {
            return Ok(cand[i].0);
        }
    }
    Ok(cand[cand.len() - 1].0)
}

/// Результат фазы: выданные токены (без токена конца) и был ли обрыв по лимиту.
pub struct PhaseOutput {
    pub tokens: Vec<u32>,
    /// `true` — фаза упёрлась в `max_tokens`, а не закончилась сама.
    pub truncated: bool,
    pub timing: PhaseTiming,
}

/// Прогнать одну фазу. `negative` — сессия отрицательной ветки CFG (нужна при
/// `cfg_scale != 1`). Декод идёт CUDA-графом, когда модель целиком на карте.
#[allow(clippy::too_many_arguments)]
pub fn run_phase(
    cond: &mut ArSession<'_>,
    mut negative: Option<&mut ArSession<'_>>,
    phase: Phase,
    sampling: &Sampling,
    cfg_scale: f32,
    seed: u64,
    cancel: &dyn Fn() -> bool,
    on_token: &dyn Fn(Phase, u32, usize),
) -> Result<PhaseOutput, YueError> {
    use synaptix_core::grad::no_grad;
    use synaptix_infer::error::InferError;
    use synaptix_infer::graph_capture::GraphCapturer;

    sampling.validate()?;
    if cfg_scale != 1.0 && negative.is_none() {
        return Err(YueError::Config("CFG требует отрицательной ветки".into()));
    }
    // Сид у каждой фазы свой — так устроен пресет.
    let mut rng = Philox4x32::new(seed);
    let started = Instant::now();
    let prefix_tokens = cond.position();
    let prefill_seconds = cond.prefill_seconds();
    let mut history: Vec<u32> = Vec::with_capacity(sampling.max_tokens.min(4096));
    let mut truncated = true;
    let end_token = phase.end_token();

    // Графы декода живут до конца фазы: буферы KV и состояния предвыделены,
    // меняются только содержимое и позиция. Смена фазы захватывает их заново
    // (это доли миллисекунды против тысяч шагов декода).
    let ord = cond.graph_ready();
    let mut capturers: Vec<GraphCapturer> = Vec::new();
    let mut graphs = Vec::new();
    let mut graph_mode = false;

    for step in 0..sampling.max_tokens {
        if cancel() {
            return Err(YueError::Cancelled(phase.as_str()));
        }
        let ulogits = match negative.as_deref() {
            Some(s) => Some(s.logits()?.clone()),
            None => None,
        };
        let token = sample_token(
            cond.logits()?,
            ulogits.as_ref(),
            cfg_scale,
            phase,
            sampling,
            &history,
            step,
            &mut rng,
        )?;
        on_token(phase, token, step);
        if token == end_token {
            truncated = false;
            break;
        }
        history.push(token);
        if step + 1 >= sampling.max_tokens {
            break;
        }

        let Some(ord) = ord else {
            cond.eager_step(token)?;
            if let Some(s) = negative.as_deref_mut() {
                s.eager_step(token)?;
            }
            continue;
        };
        let stream = synaptix_core::device::cuda::default_stream(ord)
            .map_err(|e| YueError::Other(format!("поток: {e}")))?;
        // Тип графа приходит из cudarc, и назвать его в сигнатуре помощника
        // было бы лишней зависимостью крейта, поэтому ветки разворачиваются
        // макросом на месте: типы выводятся, а `&mut` не складываются в
        // контейнер (что ссорило лайфтаймы двух сессий).
        macro_rules! capture_branch {
            ($session:expr) => {{
                let session: &mut ArSession<'_> = $session;
                let mut state = match session.decode.take() {
                    Some(s) => s,
                    None => session
                        .ar
                        .model
                        .make_decode_state()
                        .map_err(|e| YueError::Other(e.to_string()))?,
                };
                state
                    .update(token, session.pos as u32)
                    .map_err(|e| YueError::Other(e.to_string()))?;
                let mut capturer = GraphCapturer::new(3);
                let graph = {
                    let model = &session.ar.model;
                    let sr = &mut state;
                    let kr = &mut session.kv;
                    no_grad(|| {
                        capturer.capture_with(&stream, |_| {
                            model
                                .forward_decode_dev(sr, kr)
                                .map_err(|e| InferError::Other(e.to_string()))
                        })
                    })
                    .map_err(|e| YueError::Other(format!("захват графа декода: {e}")))?
                };
                graph
                    .upload()
                    .map_err(|e| YueError::Other(format!("загрузка графа декода: {e}")))?;
                session.pos += 1;
                session.logits = Some(state.logits.clone());
                session.decode = Some(state);
                capturers.push(capturer);
                graphs.push(graph);
            }};
        }
        macro_rules! launch_branch {
            ($session:expr, $graph:expr) => {{
                let session: &mut ArSession<'_> = $session;
                let state = session
                    .decode
                    .as_mut()
                    .ok_or_else(|| YueError::Other("состояние декода потеряно".into()))?;
                state
                    .update(token, session.pos as u32)
                    .map_err(|e| YueError::Other(e.to_string()))?;
                $graph
                    .launch()
                    .map_err(|e| YueError::Other(format!("запуск графа декода: {e:?}")))?;
                session.pos += 1;
            }};
        }
        if !graph_mode {
            // Первый шаг фазы — захват. Состояние декода у сессии своё и
            // переживает фазы, граф ссылается именно на него.
            capture_branch!(&mut *cond);
            if let Some(s) = negative.as_deref_mut() {
                capture_branch!(s);
            }
            graph_mode = true;
            continue;
        }
        launch_branch!(&mut *cond, graphs[0]);
        if let Some(s) = negative.as_deref_mut() {
            launch_branch!(s, graphs[1]);
        }
        stream
            .synchronize()
            .map_err(|e| YueError::Other(format!("синхронизация: {e:?}")))?;
    }
    let seconds = started.elapsed().as_secs_f64();
    let output_tokens = history.len() + usize::from(!truncated);
    Ok(PhaseOutput {
        timing: PhaseTiming {
            prefill_seconds,
            seconds,
            output_tokens,
            prefix_tokens,
            tokens_per_second: if seconds > 0.0 { output_tokens as f64 / seconds } else { 0.0 },
            cfg_branches: if cfg_scale == 1.0 { 1 } else { 2 },
            cuda_graph: graph_mode,
        },
        tokens: history,
        truncated,
    })
}
