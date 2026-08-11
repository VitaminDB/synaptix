use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result as CoreResult;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::linear::{
    gated_delta_decay_beta, gated_delta_net_recurrent, GatedDeltaNetState,
};
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::causal_conv1d::causal_conv1d_stateful;
use synaptix_ops::embed::token_embedding;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_range, RopeLayout};
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::config::{Activation, DecoderConfig, LayerKind, NormGain};
use crate::weights::{QLinear, WeightSource};

const MASK_NEG: f32 = -1.0e4;

static PREFILL_PROF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Включить/выключить per-op prefill-профиль (аналог [`set_decode_prof`], для
/// prefill-пути). Дефолт ВЫКЛ; перф-инструмент для examples/бенчей.
pub fn set_prefill_prof(on: bool) {
    PREFILL_PROF.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn prefill_prof_on() -> bool {
    PREFILL_PROF.load(std::sync::atomic::Ordering::Relaxed)
}

thread_local! {
    static PROF_ACC: std::cell::RefCell<std::collections::BTreeMap<&'static str, (f64, u64)>> =
        std::cell::RefCell::new(std::collections::BTreeMap::new());
    static DECODE_PROF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Включить/выключить per-op decode-профиль (аналог [`prefill_prof_on`], но для
/// `forward_decode_dev`-пути). ОБЯЗАН быть ВЫКЛ во время CUDA-graph capture:
/// `prof` синхронизирует stream, а sync под capture нелегален.
pub fn set_decode_prof(on: bool) {
    DECODE_PROF.with(|c| c.set(on));
}

#[inline]
fn prof_on() -> bool {
    prefill_prof_on() || DECODE_PROF.with(|c| c.get())
}

#[inline]
fn prof<T>(device: Device, name: &'static str, f: impl FnOnce() -> T) -> T {
    if !prof_on() {
        return f();
    }
    if device.is_cuda() {
        let _ = synaptix_core::device::cuda::synchronize(device.ordinal());
    }
    let t0 = std::time::Instant::now();
    let r = f();
    if device.is_cuda() {
        let _ = synaptix_core::device::cuda::synchronize(device.ordinal());
    }
    let dt = t0.elapsed().as_secs_f64() * 1000.0;
    PROF_ACC.with(|m| {
        let mut m = m.borrow_mut();
        let e = m.entry(name).or_insert((0.0, 0));
        e.0 += dt;
        e.1 += 1;
    });
    r
}

fn prof_report_and_clear(phase: &str) -> String {
    PROF_ACC.with(|m| {
        let mut m = m.borrow_mut();
        let mut lines: Vec<(&'static str, f64, u64)> =
            m.iter().map(|(k, (t, c))| (*k, *t, *c)).collect();
        lines.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let total: f64 = lines.iter().map(|x| x.1).sum();
        let mut s = format!("\n=== {phase} phase breakdown (synced total {total:.1} ms) ===\n");
        for (k, t, c) in lines {
            let pct = if total > 0.0 { 100.0 * t / total } else { 0.0 };
            let per = if c > 0 { t / c as f64 } else { 0.0 };
            s += &format!("  {k:24} {t:9.3} ms  {pct:5.1}%  ({c} calls, {per:.4} ms/call)\n");
        }
        m.clear();
        s
    })
}

pub fn prefill_prof_report_and_clear() -> String {
    prof_report_and_clear("prefill")
}

pub fn decode_prof_report_and_clear() -> String {
    prof_report_and_clear("decode")
}

/// Режим резидент/offload для [`DecoderModel::build_auto`].
/// `Auto` (default) — резидент с откатом в offload при OOM;
/// `Resident` — всегда GPU-резидент; `Offload` — всегда host-stream.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OffloadMode {
    #[default]
    Auto,
    Resident,
    Offload,
}

static OFFLOAD_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_offload_mode(mode: OffloadMode) {
    let v = match mode {
        OffloadMode::Auto => 0,
        OffloadMode::Resident => 1,
        OffloadMode::Offload => 2,
    };
    OFFLOAD_MODE.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn offload_mode() -> OffloadMode {
    match OFFLOAD_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => OffloadMode::Resident,
        2 => OffloadMode::Offload,
        _ => OffloadMode::Auto,
    }
}

/// Эвристика «ошибка = нехватка VRAM» для авто-отката резидент→offload
/// (`build_auto`). CUDA-аллокаторы рапортуют OOM по-разному (alloc_zeros/
/// out of memory / OOM) — матчим по подстроке.
fn is_oom_err(e: &ModelError) -> bool {
    let s = e.to_string().to_ascii_lowercase();
    s.contains("oom")
        || s.contains("out of memory")
        || s.contains("outofmemory")
        || s.contains("alloc_zeros")
        || s.contains("alloc failed")
}

pub struct FullAttn {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    attn_output_gate: bool,
    attn_scale: f32,
    rms_eps: f32,
    use_flash: bool,
    sliding_window: Option<usize>,
}

pub struct LinearAttn {
    in_proj_qkv: QLinear,
    in_proj_a: QLinear,
    in_proj_b: QLinear,
    in_proj_z: QLinear,
    out_proj: QLinear,
    conv_w: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm_weight: Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    dk: usize,
    dv: usize,
    conv_k: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    q_scale: f32,
    rms_eps: f32,
    conv_w_dev: Option<Tensor>,
    a_log_dev: Option<Tensor>,
    dt_bias_dev: Option<Tensor>,
    norm_w_f16: Option<Tensor>,
}

pub enum Mixer {
    Full(FullAttn),
    Linear(LinearAttn),
}

pub struct Mlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
    activation: Activation,
}

pub struct Block {
    pre_attn_norm: Tensor,
    post_attn_norm: Option<Tensor>,
    pre_mlp_norm: Tensor,
    post_mlp_norm: Option<Tensor>,
    mixer: Mixer,
    mlp: Mlp,
    rms_eps: f32,
}

impl FullAttn {
    fn to_device(&self, dev: Device) -> Result<Self, ModelError> {
        let t = |x: &Tensor| x.to_device(dev).map_err(|e| ModelError::Load(e.to_string()));
        let ot = |x: &Option<Tensor>| -> Result<Option<Tensor>, ModelError> {
            Ok(match x { Some(v) => Some(t(v)?), None => None })
        };
        Ok(Self {
            q_proj: self.q_proj.to_device(dev)?,
            k_proj: self.k_proj.to_device(dev)?,
            v_proj: self.v_proj.to_device(dev)?,
            o_proj: self.o_proj.to_device(dev)?,
            q_norm: ot(&self.q_norm)?,
            k_norm: ot(&self.k_norm)?,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            rotary_dim: self.rotary_dim,
            attn_output_gate: self.attn_output_gate,
            attn_scale: self.attn_scale,
            rms_eps: self.rms_eps,
            use_flash: self.use_flash,
            sliding_window: self.sliding_window,
        })
    }
}

impl Mlp {
    fn to_device(&self, dev: Device) -> Result<Self, ModelError> {
        Ok(Self {
            gate_proj: self.gate_proj.to_device(dev)?,
            up_proj: self.up_proj.to_device(dev)?,
            down_proj: self.down_proj.to_device(dev)?,
            activation: self.activation,
        })
    }
}

impl LinearAttn {
    /// Перенос на устройство (host-stream Hybrid-блоков: CPU-резидент → GPU по
    /// требованию). Device-зеркала (conv_w_dev/a_log_dev/dt_bias_dev/norm_w_f16)
    /// пересоздаются из host-векторов на целевом устройстве (как в `build_ext`),
    /// чтобы CUDA-пути decode/prefill (требуют conv_w_dev) работали и на
    /// стриминговом блоке. На CPU зеркала = None.
    fn to_device(&self, dev: Device) -> Result<Self, ModelError> {
        let norm_weight = self.norm_weight.to_device(dev).map_err(|e| ModelError::Load(e.to_string()))?;
        let (conv_w_dev, a_log_dev, dt_bias_dev, norm_w_f16) = if dev.is_cpu() {
            (None, None, None, None)
        } else {
            let mk = |v: &Vec<f32>, shape: Vec<usize>| -> Result<Tensor, ModelError> {
                Tensor::from_vec(v.clone(), shape, dev).map_err(|e| ModelError::Load(e.to_string()))
            };
            (
                Some(
                    mk(&self.conv_w, vec![self.conv_dim, self.conv_k])?
                        .to_dtype(DType::F16)
                        .map_err(|e| ModelError::Load(e.to_string()))?,
                ),
                Some(mk(&self.a_log, vec![self.num_v_heads])?),
                Some(mk(&self.dt_bias, vec![self.num_v_heads])?),
                Some(
                    norm_weight
                        .to_dtype(DType::F16)
                        .map_err(|e| ModelError::Load(e.to_string()))?,
                ),
            )
        };
        Ok(Self {
            in_proj_qkv: self.in_proj_qkv.to_device(dev)?,
            in_proj_a: self.in_proj_a.to_device(dev)?,
            in_proj_b: self.in_proj_b.to_device(dev)?,
            in_proj_z: self.in_proj_z.to_device(dev)?,
            out_proj: self.out_proj.to_device(dev)?,
            conv_w: self.conv_w.clone(),
            a_log: self.a_log.clone(),
            dt_bias: self.dt_bias.clone(),
            norm_weight,
            num_k_heads: self.num_k_heads,
            num_v_heads: self.num_v_heads,
            dk: self.dk,
            dv: self.dv,
            conv_k: self.conv_k,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            conv_dim: self.conv_dim,
            q_scale: self.q_scale,
            rms_eps: self.rms_eps,
            conv_w_dev,
            a_log_dev,
            dt_bias_dev,
            norm_w_f16,
        })
    }
}

impl Block {
    /// Перенос блока на устройство (host-stream: CPU-резидент → GPU по
    /// требованию). Linear-mixer не поддержан (gemma/llama full-attention only).
    fn to_device(&self, dev: Device) -> Result<Self, ModelError> {
        let t = |x: &Tensor| x.to_device(dev).map_err(|e| ModelError::Load(e.to_string()));
        let ot = |x: &Option<Tensor>| -> Result<Option<Tensor>, ModelError> {
            Ok(match x { Some(v) => Some(t(v)?), None => None })
        };
        let mixer = match &self.mixer {
            Mixer::Full(fa) => Mixer::Full(fa.to_device(dev)?),
            Mixer::Linear(la) => Mixer::Linear(la.to_device(dev)?),
        };
        Ok(Self {
            pre_attn_norm: t(&self.pre_attn_norm)?,
            post_attn_norm: ot(&self.post_attn_norm)?,
            pre_mlp_norm: t(&self.pre_mlp_norm)?,
            post_mlp_norm: ot(&self.post_mlp_norm)?,
            mixer,
            mlp: self.mlp.to_device(dev)?,
            rms_eps: self.rms_eps,
        })
    }
}

pub struct DecoderModel {
    pub config: DecoderConfig,
    pub device: Device,
    pub dtype: DType,
    pub kv_dtype: DType,
    embed: Option<Tensor>,
    embed_q: Option<QuantWeight>,
    final_norm: Tensor,
    lm_head: QLinear,
    blocks: Vec<Block>,
    rope_global: RopeCache,
    rope_local: Option<RopeCache>,
    rope_capacity: usize,
    embed_scale: Option<f32>,
    /// Блоки CPU-резидентны и стримятся на GPU per-block в forward (pinned-H2D
    /// с префетчем) — bf16-энкодер 24GB на 24GB-карте. Декод-петли не поддержаны.
    host_stream_blocks: bool,
}

pub struct KvCacheLayer {
    pub k: Tensor,
    pub v: Tensor,
    pub k_scale: Option<Tensor>,
    pub v_scale: Option<Tensor>,
}

pub enum LayerCache {
    Full(KvCacheLayer),
    Linear(GatedDeltaNetState),
}

pub struct KvCache {
    pub layers: Vec<LayerCache>,
    pub seq_len: usize,
    pub max_seq: usize,
}

fn deep_copy(src: &Tensor) -> Result<Tensor, ModelError> {
    let mut dst = Tensor::zeros(src.dims().to_vec(), src.dtype(), src.device())
        .map_err(|e| ModelError::Forward(e.to_string()))?;
    dst.copy_from(src).map_err(|e| ModelError::Forward(e.to_string()))?;
    Ok(dst)
}

pub struct LinearSnapshot {
    conv_dev: Option<Tensor>,
    ssm_dev: Option<Tensor>,
    conv_host: Option<Vec<f32>>,
    ssm_host: Option<Vec<f32>>,
}

impl KvCache {
    pub fn alloc_linear_snapshot(&self) -> Result<Vec<LinearSnapshot>, ModelError> {
        self.snapshot_linear()
    }

    pub fn save_linear_into(&self, snap: &mut [LinearSnapshot]) -> Result<(), ModelError> {
        let mut i = 0;
        for l in &self.layers {
            let LayerCache::Linear(st) = l else { continue };
            let s = snap.get_mut(i).ok_or_else(|| {
                ModelError::Shape("save_linear_into: снапшот короче числа linear-слоёв".into())
            })?;
            i += 1;
            if let (Some(src), Some(dst)) = (&st.conv_state_dev, s.conv_dev.as_mut()) {
                dst.copy_from(src).map_err(|e| ModelError::Forward(e.to_string()))?;
            }
            if let (Some(src), Some(dst)) = (&st.ssm_state_dev, s.ssm_dev.as_mut()) {
                dst.copy_from(src).map_err(|e| ModelError::Forward(e.to_string()))?;
            }
            if let Some(v) = s.conv_host.as_mut() {
                v.copy_from_slice(&st.conv_state);
            }
            if let Some(v) = s.ssm_host.as_mut() {
                v.copy_from_slice(&st.ssm_state);
            }
        }
        Ok(())
    }

    pub fn snapshot_linear(&self) -> Result<Vec<LinearSnapshot>, ModelError> {
        let mut out = Vec::new();
        for l in &self.layers {
            let LayerCache::Linear(st) = l else { continue };
            let snap = if st.conv_state_dev.is_some() || st.ssm_state_dev.is_some() {
                LinearSnapshot {
                    conv_dev: match &st.conv_state_dev {
                        Some(t) => Some(deep_copy(t)?),
                        None => None,
                    },
                    ssm_dev: match &st.ssm_state_dev {
                        Some(t) => Some(deep_copy(t)?),
                        None => None,
                    },
                    conv_host: None,
                    ssm_host: None,
                }
            } else {
                LinearSnapshot {
                    conv_dev: None,
                    ssm_dev: None,
                    conv_host: Some(st.conv_state.clone()),
                    ssm_host: Some(st.ssm_state.clone()),
                }
            };
            out.push(snap);
        }
        Ok(out)
    }

    pub fn restore_linear(&mut self, snap: &[LinearSnapshot]) -> Result<(), ModelError> {
        let mut i = 0;
        for l in self.layers.iter_mut() {
            let LayerCache::Linear(st) = l else { continue };
            let s = snap.get(i).ok_or_else(|| {
                ModelError::Shape("restore_linear: снапшот короче числа linear-слоёв".into())
            })?;
            i += 1;
            if let (Some(src), Some(dst)) = (&s.conv_dev, st.conv_state_dev.as_mut()) {
                dst.copy_from(src).map_err(|e| ModelError::Forward(e.to_string()))?;
            }
            if let (Some(src), Some(dst)) = (&s.ssm_dev, st.ssm_state_dev.as_mut()) {
                dst.copy_from(src).map_err(|e| ModelError::Forward(e.to_string()))?;
            }
            if let Some(v) = &s.conv_host {
                st.conv_state.copy_from_slice(v);
            }
            if let Some(v) = &s.ssm_host {
                st.ssm_state.copy_from_slice(v);
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        for l in &mut self.layers {
            if let LayerCache::Linear(s) = l {
                s.reset();
            }
        }
    }
}

pub struct DecodeState {
    pub input: Tensor,
    pub pos_dev: Tensor,
    pub tcache_dev: Tensor,
    pub rope_cos: Tensor,
    pub rope_sin: Tensor,
    pub logits: Tensor,
}

impl DecodeState {
    pub fn update(&mut self, token: u32, pos: u32) -> Result<(), ModelError> {
        self.input.write_host_u32(&[token]).coerr()?;
        self.pos_dev.write_host_u32(&[pos]).coerr()?;
        self.tcache_dev.write_host_u32(&[pos + 1]).coerr()?;
        Ok(())
    }

    /// Batched per-row update: `tokens`/`positions` length = batch (the state's
    /// row count). Each row appends its token at its own absolute position
    /// (per-row RoPE + KV length). Used for batched CFG decode (cond+uncond).
    pub fn update_batched(&mut self, tokens: &[u32], positions: &[u32]) -> Result<(), ModelError> {
        self.input.write_host_u32(tokens).coerr()?;
        self.pos_dev.write_host_u32(positions).coerr()?;
        let tcache: Vec<u32> = positions.iter().map(|p| p + 1).collect();
        self.tcache_dev.write_host_u32(&tcache).coerr()?;
        Ok(())
    }
}

/// Device-резидентное состояние для CUDA-graph **prefill chunk'а** (`forward_prefill_dev`).
///
/// Аналог [`DecodeState`] для случая T = `chunk_size` > 1. Все буферы аллоцированы
/// один раз, размеры от значений `pos_start`/токенов не зависят → один граф
/// валиден для любого chunk'а (одного и того же размера). Перед replay'ем
/// `PrefillState::update` host→device-копирует ids + позицию в стабильные адреса.
///
/// Поля:
/// - `chunk_size` — T токенов в одном chunk'е, известно при capture.
/// - `input` `[1, chunk_size]` U32 — ids текущего chunk'а.
/// - `pos_start` `[1]` U32 — абсолютная позиция первого токена chunk'а (=
///   `seq_pos` для `kv_append_dev`, `start_pos` для `rope_apply_dev`).
/// - `tcache_dev` `[1]` U32 — `pos_start + chunk_size` (активная длина KV после
///   `kv_append_dev`, передаётся в `flash_attention_dev`).
/// - `rope_cos`/`rope_sin` — те же дублированные таблицы что и в `DecodeState`
///   (`[rope_capacity, rotary_dim]`, dtype = compute), ядро rope сэмплит по
///   `(pos_start + t) * rotary_dim + d`.
/// - `logits` `[1, vocab_size]` — выход lm_head для **последнего** токена chunk'а
///   (только он используется для сэмплинга следующего токена в decode).
pub struct PrefillState {
    pub chunk_size: usize,
    pub input: Tensor,
    pub pos_start: Tensor,
    pub tcache_dev: Tensor,
    pub rope_cos: Tensor,
    pub rope_sin: Tensor,
    pub logits: Tensor,
    pub hidden: Tensor,
}

impl PrefillState {
    /// In-place host→device запись ids и позиции в стабильные буферы. Длина
    /// `tokens` должна совпадать с `chunk_size` (граф captured под фиксированный
    /// T — частичные chunk'и обрабатывает host-fallback в pipeline'е).
    pub fn update(&mut self, tokens: &[u32], pos_start: u32) -> Result<(), ModelError> {
        if tokens.len() != self.chunk_size {
            return Err(ModelError::Shape(format!(
                "PrefillState::update: tokens.len {} != chunk_size {}",
                tokens.len(),
                self.chunk_size
            )));
        }
        self.input.write_host_u32(tokens).coerr()?;
        self.pos_start.write_host_u32(&[pos_start]).coerr()?;
        self.tcache_dev
            .write_host_u32(&[pos_start + self.chunk_size as u32])
            .coerr()?;
        Ok(())
    }
}

impl DecoderModel {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        cfg: &DecoderConfig,
        weights: &dyn WeightSource,
        device: Device,
        compute: DType,
        attn_w: DType,
        mlp_w: DType,
        lm_head_dtype: DType,
        embed_dtype: DType,
        rope_capacity: usize,
    ) -> Result<Self, ModelError> {
        Self::build_ext(cfg, weights, device, None, compute, attn_w, mlp_w, lm_head_dtype, embed_dtype, rope_capacity)
    }

    /// Авто-выбор резидент/offload для генерации. Пробует резидентную загрузку
    /// (блоки на GPU → быстрый decode); при нехватке VRAM (OOM на любом весе) —
    /// пере-собирает блоки CPU-резидентно с host-stream (pinned-H2D per-block в
    /// `forward`, как DiT-блоки LTX) → работает при ЛЮБОМ объёме свободной VRAM
    /// ценой PCIe-стрима каждого блока. Управление [`set_offload_mode`]:
    /// `Resident` — всегда резидент; `Offload` — всегда offload; иначе
    /// (`Auto`, default) — резидент с откатом в offload при OOM. На
    /// CPU-устройстве всегда резидент.
    #[allow(clippy::too_many_arguments)]
    pub fn build_auto(
        cfg: &DecoderConfig,
        weights: &dyn WeightSource,
        device: Device,
        compute: DType,
        attn_w: DType,
        mlp_w: DType,
        lm_head_dtype: DType,
        embed_dtype: DType,
        rope_capacity: usize,
    ) -> Result<Self, ModelError> {
        let mode = offload_mode();
        let resident = |()| Self::build_ext(cfg, weights, device, None, compute, attn_w, mlp_w, lm_head_dtype, embed_dtype, rope_capacity);
        let offload = |()| Self::build_ext(cfg, weights, device, Some(Device::Cpu), compute, attn_w, mlp_w, lm_head_dtype, embed_dtype, rope_capacity);

        if device.is_cpu() || mode == OffloadMode::Resident {
            return resident(());
        }
        if mode == OffloadMode::Offload {
            return offload(());
        }
        match resident(()) {
            Ok(m) => Ok(m),
            Err(e) if is_oom_err(&e) => {
                if let Device::Cuda(o) = device {
                    let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(o);
                }
                offload(())
            }
            Err(e) => Err(e),
        }
    }

    /// Как [`Self::build`], но блоки строятся на `block_device` (Some(Cpu) =
    /// host-stream: CPU-резидент, per-block стрим на GPU в forward с pinned-H2D
    /// префетчем). Embed/rope/lm_head остаются на `device`. Host-stream работает
    /// в `forward_hidden_states` (текст-энкодер) и `forward` (генерация: prefill
    /// + decode, full + linear-mixer, персистентный KV).
    #[allow(clippy::too_many_arguments)]
    pub fn build_ext(
        cfg: &DecoderConfig,
        weights: &dyn WeightSource,
        device: Device,
        block_device: Option<Device>,
        compute: DType,
        attn_w: DType,
        mlp_w: DType,
        lm_head_dtype: DType,
        embed_dtype: DType,
        rope_capacity: usize,
    ) -> Result<Self, ModelError> {
        let eps = cfg.rms_norm_eps;
        let one_plus = cfg.norm_gain == NormGain::OnePlus;
        let b_dev = block_device.unwrap_or(device);

        // Квантование выполняется на GPU (CPU-backend не реализует quantize_nvfp4/
        // mxfp8), результат кладётся на `b_dev`:
        //  • резидент (b_dev==device==GPU): квант остаётся на GPU;
        //  • offload (b_dev==Cpu): F16 материализуется на GPU временно, квантуется,
        //    компактный квант переносится на CPU; host-stream вернёт его на GPU
        //    per-block в forward. На GPU при загрузке живёт ~1 вес → влезает при
        //    скромной VRAM;
        //  • device==Cpu (чистый CPU-инференс): квант недоступен → плотный путь.
        let qlin = |key: &str, wq: DType| -> Result<QLinear, ModelError> {
            if wq.is_quantized() && matches!(device, Device::Cuda(_)) {
                let w = weights.tensor(key, device, DType::F16)?;
                let q = QLinear::build(w, wq, compute)?;
                return if b_dev == device { Ok(q) } else { q.to_device(b_dev) };
            }
            let qd = if matches!(device, Device::Cuda(_)) { wq } else { compute };
            let wdt = if qd.is_quantized() { DType::F16 } else { compute };
            let w = weights.tensor(key, b_dev, wdt)?;
            QLinear::build(w, qd, compute)
        };
        let norm = |key: &str| -> Result<Tensor, ModelError> {
            let w = weights.tensor(key, b_dev, if one_plus { DType::F32 } else { compute })?;
            if one_plus {
                w.add_scalar(1.0)
                    .and_then(|t| t.to_dtype(compute))
                    .map_err(|e| ModelError::Load(e.to_string()))
            } else {
                Ok(w)
            }
        };
        let host_f32 = |key: &str| -> Result<Vec<f32>, ModelError> {
            let t = weights.tensor(key, Device::Cpu, DType::F32)?;
            t.flatten_all()
                .and_then(|x| x.to_vec1::<f32>())
                .map_err(|e| ModelError::Load(e.to_string()))
        };

        let mut embed_dense = Some(weights.tensor("model.embed_tokens.weight", device, compute)?);
        let embed_quant = if embed_dtype == DType::MXFP8
            && !cfg.tie_word_embeddings
            && matches!(device, Device::Cuda(_))
            && cfg.hidden_size % 32 == 0
        {
            let q = embed_dense
                .as_ref()
                .unwrap()
                .quantize_to_mxfp8()
                .map_err(|e| ModelError::Build(format!("quantize embed to mxfp8: {e}")))?;
            embed_dense = None;
            if let Device::Cuda(o) = device {
                let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
            }
            Some(q)
        } else {
            None
        };
        let final_norm = norm("model.norm.weight")?
            .to_device(device)
            .map_err(|e| ModelError::Load(e.to_string()))?;
        // lm_head: при tie_word_embeddings = embed (Dense, не квантуем — embed нужен
        // для gather). Иначе грузим lm_head.weight и квантуем по `lm_head_dtype`
        // (NVFP4 [vocab,hidden] %64==0 → GEMV; экономит 2.5GB→0.7GB чтения/токен).
        let lm_head = if cfg.tie_word_embeddings {
            QLinear::build(
                embed_dense
                    .clone()
                    .ok_or_else(|| ModelError::Build("tied lm_head без embed".into()))?,
                compute,
                compute,
            )?
        } else {
            // lm_head всегда резидентен на `device` (даже при offload); квант
            // считается на GPU. На CPU-устройстве квант недоступен → плотный.
            let ld = if matches!(device, Device::Cuda(_)) { lm_head_dtype } else { compute };
            let wdt = if ld.is_quantized() { DType::F16 } else { compute };
            let w = weights.tensor("lm_head.weight", device, wdt)?;
            QLinear::build(w, ld, compute)?
        };

        let use_flash = cfg.simple_profile() || matches!(cfg.head_dim, 64 | 128 | 256);
        let lin = cfg.linear.as_ref();
        let q_scale = lin.map(|l| 1.0 / (l.key_head_dim as f32).sqrt()).unwrap_or(1.0);

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let key = |s: &str| format!("model.layers.{l}.{s}");
            let mixer = match cfg.layer_kind(l) {
                LayerKind::Full => Mixer::Full(FullAttn {
                    q_proj: qlin(&key("self_attn.q_proj.weight"), attn_w)?,
                    k_proj: qlin(&key("self_attn.k_proj.weight"), attn_w)?,
                    v_proj: qlin(&key("self_attn.v_proj.weight"), attn_w)?,
                    o_proj: qlin(&key("self_attn.o_proj.weight"), attn_w)?,
                    q_norm: if cfg.qk_norm { Some(norm(&key("self_attn.q_norm.weight"))?) } else { None },
                    k_norm: if cfg.qk_norm { Some(norm(&key("self_attn.k_norm.weight"))?) } else { None },
                    num_heads: cfg.num_attention_heads,
                    num_kv_heads: cfg.num_key_value_heads,
                    head_dim: cfg.head_dim,
                    rotary_dim: cfg.rope_for(l).rotary_dim,
                    attn_output_gate: cfg.attn_output_gate,
                    attn_scale: cfg.attn_scale,
                    rms_eps: eps,
                    use_flash,
                    sliding_window: cfg.window_for(l),
                }),
                LayerKind::Linear => {
                    let lc = lin.ok_or_else(|| ModelError::Build("linear layer без LinearAttnConfig".into()))?;
                    let conv_w = host_f32(&key("linear_attn.conv1d.weight"))?;
                    let a_log = host_f32(&key("linear_attn.A_log"))?;
                    let dt_bias = host_f32(&key("linear_attn.dt_bias"))?;
                    let norm_weight = weights.tensor(&key("linear_attn.norm.weight"), device, compute)?;
                    let (conv_dim, ck, nv) = (lc.conv_dim(), lc.conv_kernel, lc.num_value_heads);
                    // Device-зеркала весов для CUDA-graph decode (F16/F32). На CPU не нужны.
                    let (conv_w_dev, a_log_dev, dt_bias_dev, norm_w_f16) = if device.is_cpu() {
                        (None, None, None, None)
                    } else {
                        (
                            Some(Tensor::from_vec(conv_w.clone(), vec![conv_dim, ck], device).coerr()?
                                .to_dtype(DType::F16).coerr()?),
                            Some(Tensor::from_vec(a_log.clone(), vec![nv], device).coerr()?),
                            Some(Tensor::from_vec(dt_bias.clone(), vec![nv], device).coerr()?),
                            Some(norm_weight.to_dtype(DType::F16).coerr()?),
                        )
                    };
                    Mixer::Linear(LinearAttn {
                        in_proj_qkv: qlin(&key("linear_attn.in_proj_qkv.weight"), attn_w)?,
                        in_proj_a: qlin(&key("linear_attn.in_proj_a.weight"), attn_w)?,
                        in_proj_b: qlin(&key("linear_attn.in_proj_b.weight"), attn_w)?,
                        in_proj_z: qlin(&key("linear_attn.in_proj_z.weight"), attn_w)?,
                        out_proj: qlin(&key("linear_attn.out_proj.weight"), attn_w)?,
                        conv_w,
                        a_log,
                        dt_bias,
                        norm_weight,
                        num_k_heads: lc.num_key_heads,
                        num_v_heads: lc.num_value_heads,
                        dk: lc.key_head_dim,
                        dv: lc.value_head_dim,
                        conv_k: lc.conv_kernel,
                        key_dim: lc.key_dim(),
                        value_dim: lc.value_dim(),
                        conv_dim: lc.conv_dim(),
                        q_scale,
                        rms_eps: eps,
                        conv_w_dev,
                        a_log_dev,
                        dt_bias_dev,
                        norm_w_f16,
                    })
                }
            };
            let (post_attn_key, pre_mlp_key, post_mlp_key) = if cfg.sandwich_norms {
                (
                    Some("post_attention_layernorm.weight"),
                    "pre_feedforward_layernorm.weight",
                    Some("post_feedforward_layernorm.weight"),
                )
            } else {
                (None, "post_attention_layernorm.weight", None)
            };
            let mlp = Mlp {
                gate_proj: qlin(&key("mlp.gate_proj.weight"), mlp_w)?,
                up_proj: qlin(&key("mlp.up_proj.weight"), mlp_w)?,
                down_proj: qlin(&key("mlp.down_proj.weight"), mlp_w)?,
                activation: cfg.activation,
            };
            blocks.push(Block {
                pre_attn_norm: norm(&key("input_layernorm.weight"))?,
                post_attn_norm: match post_attn_key {
                    Some(k) => Some(norm(&key(k))?),
                    None => None,
                },
                pre_mlp_norm: norm(&key(pre_mlp_key))?,
                post_mlp_norm: match post_mlp_key {
                    Some(k) => Some(norm(&key(k))?),
                    None => None,
                },
                mixer,
                mlp,
                rms_eps: eps,
            });
        }

        let rope_capacity = rope_capacity.max(1);
        let build_rope = |spec: &crate::config::RopeSpec| -> Result<RopeCache, ModelError> {
            match &spec.scaled_freqs {
                Some(freqs) => RopeCache::with_scaled_freqs(spec.rotary_dim, rope_capacity, spec.theta, freqs, device),
                None => RopeCache::new(spec.rotary_dim, rope_capacity, spec.theta, device),
            }
            .map_err(|e| ModelError::Build(e.to_string()))
        };
        let rope_global = build_rope(&cfg.rope_global)?;
        let rope_local = match &cfg.rope_local {
            Some(s) => Some(build_rope(s)?),
            None => None,
        };

        Ok(Self {
            config: cfg.clone(),
            device,
            dtype: compute,
            kv_dtype: compute,
            embed: embed_dense,
            embed_q: embed_quant,
            final_norm,
            lm_head,
            blocks,
            rope_global,
            rope_local,
            rope_capacity,
            embed_scale: cfg.embed_scale,
            host_stream_blocks: block_device.is_some_and(|d| d != device),
        })
    }

    pub fn with_kv_cache_dtype(mut self, kv_dtype: DType) -> Self {
        self.kv_dtype = kv_dtype;
        self
    }

    fn rope_at(&self, idx: usize) -> &RopeCache {
        if self.config.is_global_layer(idx) {
            &self.rope_global
        } else {
            self.rope_local.as_ref().unwrap_or(&self.rope_global)
        }
    }

    pub fn rope_capacity(&self) -> usize {
        self.rope_capacity
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        let c = &self.config;
        let mxfp8 = self.kv_dtype == DType::MXFP8;
        let elem = if mxfp8 {
            1 + 1usize.div_ceil(32)
        } else {
            (self.dtype.size_in_bits() / 8).max(1)
        };
        let per_full = c.num_key_value_heads * c.head_dim * 2 * elem;
        let full_layers = (0..self.blocks.len())
            .filter(|l| matches!(c.layer_kind(*l), LayerKind::Full))
            .count();
        per_full * full_layers
    }

    pub fn has_mxfp8_head_or_embed(&self) -> bool {
        self.embed_q
            .as_ref()
            .map(|q| q.dtype() == DType::MXFP8)
            .unwrap_or(false)
            || self.lm_head.quant_dtype() == Some(DType::MXFP8)
    }

    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor, ModelError> {
        match (&self.embed_q, &self.embed) {
            (Some(q), _) => {
                let mut dims = input_ids.dims().to_vec();
                let flat = input_ids
                    .contiguous()
                    .and_then(|t| t.reshape(vec![input_ids.numel()]))
                    .coerr()?;
                let emb = q.embed_gather(&flat).coerr()?;
                dims.push(self.config.hidden_size);
                emb.reshape(dims).coerr()
            }
            (None, Some(t)) => token_embedding(input_ids, t).coerr(),
            (None, None) => Err(ModelError::Build("embed отсутствует".into())),
        }
    }

    fn embed_rows(&self, ids_flat: &Tensor) -> Result<Tensor, ModelError> {
        match (&self.embed_q, &self.embed) {
            (Some(q), _) => q.embed_gather(ids_flat).coerr(),
            (None, Some(t)) => t.embed_gather(ids_flat).coerr(),
            (None, None) => Err(ModelError::Build("embed отсутствует".into())),
        }
    }

    pub fn make_kv_cache(&self, batch: usize, max_seq: usize) -> Result<KvCache, ModelError> {
        if max_seq == 0 {
            return Err(ModelError::Shape("make_kv_cache: max_seq must be > 0".into()));
        }
        if max_seq > self.rope_capacity {
            return Err(ModelError::Shape(format!(
                "make_kv_cache: max_seq {max_seq} > RoPE capacity {}",
                self.rope_capacity
            )));
        }
        let c = &self.config;
        let n_kv = c.num_key_value_heads;
        let hd = c.head_dim;
        let mxfp8 = self.kv_dtype == DType::MXFP8;
        if mxfp8 && hd % 32 != 0 {
            return Err(ModelError::Shape(format!(
                "make_kv_cache: --kv-dtype mxfp8 требует head_dim % 32 == 0 (hd={hd})"
            )));
        }
        let kv_dt = if mxfp8 { DType::MXFP8 } else { self.dtype };
        let mut layers = Vec::with_capacity(self.blocks.len());
        for l in 0..self.blocks.len() {
            let lc = match c.layer_kind(l) {
                LayerKind::Full => {
                    let k = Tensor::zeros(vec![batch, n_kv, max_seq, hd], kv_dt, self.device).coerr()?;
                    let v = Tensor::zeros(vec![batch, n_kv, max_seq, hd], kv_dt, self.device).coerr()?;
                    let (k_scale, v_scale) = if mxfp8 {
                        let nb = hd / 32;
                        (
                            Some(Tensor::zeros(vec![batch, n_kv, max_seq, nb], DType::U8, self.device).coerr()?),
                            Some(Tensor::zeros(vec![batch, n_kv, max_seq, nb], DType::U8, self.device).coerr()?),
                        )
                    } else {
                        (None, None)
                    };
                    LayerCache::Full(KvCacheLayer { k, v, k_scale, v_scale })
                }
                LayerKind::Linear => {
                    let lin = c.linear.as_ref().unwrap();
                    LayerCache::Linear(GatedDeltaNetState::new(
                        lin.conv_dim(), lin.conv_kernel, lin.num_value_heads, lin.key_head_dim, lin.value_head_dim,
                    ))
                }
            };
            layers.push(lc);
        }
        Ok(KvCache { layers, seq_len: 0, max_seq })
    }

    pub fn forward(&self, input_ids: &Tensor, kv_cache: &mut KvCache) -> Result<Tensor, ModelError> {
        let hidden = self.forward_trunk(input_ids, kv_cache)?;
        self.head_at(&hidden, hidden.dims()[1] - 1)
    }

    pub fn embed_ids(&self, input_ids: &Tensor) -> Result<Tensor, ModelError> {
        let mut hidden = self.embed_tokens(input_ids)?;
        if let Some(scale) = self.embed_scale {
            hidden = hidden.mul_scalar(scale).coerr()?;
        }
        Ok(hidden)
    }

    pub fn normed_at(&self, hidden: &Tensor, idx: usize) -> Result<Tensor, ModelError> {
        let dev = self.device;
        let normed = prof(dev, "norm", || {
            rms_norm(hidden, &self.final_norm, self.config.rms_norm_eps).coerr()
        })?;
        normed.narrow(1, idx, 1).coerr()?.squeeze(1).coerr()
    }

    pub fn lm_head_forward(&self, x: &Tensor) -> Result<Tensor, ModelError> {
        prof(self.device, "lm_head", || self.lm_head.forward(x))
    }

    pub fn head_at(&self, hidden: &Tensor, idx: usize) -> Result<Tensor, ModelError> {
        let row = self.normed_at(hidden, idx)?;
        self.lm_head_forward(&row)
    }

    pub fn forward_trunk(&self, input_ids: &Tensor, kv_cache: &mut KvCache) -> Result<Tensor, ModelError> {
        if input_ids.rank() != 2 {
            return Err(ModelError::Shape(format!("input_ids must be [B, S], got {:?}", input_ids.dims())));
        }
        let batch = input_ids.dims()[0];
        let s = input_ids.dims()[1];
        let past = kv_cache.seq_len;
        let hidden = self.embed_ids(input_ids)?;
        if dump_layers_on() { record_layer_norm(999, "embed", &hidden, s, past); }
        self.run_blocks(hidden, kv_cache, batch, s)
    }

    pub fn forward_from_hidden(&self, hidden: &Tensor, kv_cache: &mut KvCache) -> Result<Tensor, ModelError> {
        if hidden.rank() != 3 {
            return Err(ModelError::Shape(format!("hidden must be [B, S, H], got {:?}", hidden.dims())));
        }
        let batch = hidden.dims()[0];
        let s = hidden.dims()[1];
        self.run_blocks(hidden.clone(), kv_cache, batch, s)
    }

    fn run_blocks(
        &self,
        mut hidden: Tensor,
        kv_cache: &mut KvCache,
        batch: usize,
        s: usize,
    ) -> Result<Tensor, ModelError> {
        let past = kv_cache.seq_len;
        if past + s > kv_cache.max_seq {
            return Err(ModelError::Shape(format!("KV overflow: past {past} + s {s} > max_seq {}", kv_cache.max_seq)));
        }
        let dev = self.device;
        let dump_layers = dump_layers_on();

        // Один блок (attn/linear-mixer + MLP) → новый hidden. Вынесено в замыкание,
        // чтобы общий код работал и в резидентном цикле, и в host-stream (блок
        // приходит уже на GPU). KV персистентный (kv_cache.layers[idx]).
        let step = |idx: usize, blk: &Block, hidden: &Tensor, kv_cache: &mut KvCache|
            -> Result<Tensor, ModelError> {
            let residual = hidden.clone();
            let h = prof(dev, "norm", || rms_norm(hidden, &blk.pre_attn_norm, blk.rms_eps).coerr())?;
            let is_lin = matches!(&blk.mixer, Mixer::Linear(_));
            let mixed = match &blk.mixer {
                Mixer::Full(fa) => prof(dev, "attn_full", || fa.forward(&h, &mut kv_cache.layers[idx], past, s, batch, self.rope_at(idx), self.kv_dtype, self.device, self.dtype, None))?,
                Mixer::Linear(la) => prof(dev, "attn_linear", || la.forward(&h, &mut kv_cache.layers[idx], s, self.device, self.dtype))?,
            };
            let mixed = apply_opt_norm(&mixed, blk.post_attn_norm.as_ref(), blk.rms_eps)?;
            let hidden = prof(dev, "residual", || residual.add(&mixed).coerr())?;
            if dump_layers { record_layer_norm(idx, if is_lin { "lin_attn" } else { "full_attn" }, &hidden, s, past); }

            let residual2 = hidden.clone();
            let h = prof(dev, "norm", || rms_norm(&hidden, &blk.pre_mlp_norm, blk.rms_eps).coerr())?;
            let mlp_out = prof(dev, "mlp", || blk.mlp.forward(&h))?;
            let mlp_out = apply_opt_norm(&mlp_out, blk.post_mlp_norm.as_ref(), blk.rms_eps)?;
            let out = prof(dev, "residual", || residual2.add(&mlp_out).coerr())?;
            if dump_layers { record_layer_norm(idx, "mlp", &out, s, past); }
            Ok(out)
        };

        if self.host_stream_blocks && matches!(dev, Device::Cuda(_)) {
            // host-stream: блоки CPU-резидентны, текущий стримится на GPU, следующий
            // префетчится на loader-стриме с pinned-H2D (как DiT-блоки LTX). Работает
            // при ЛЮБОМ объёме свободной VRAM — на GPU одновременно живёт ~1-2 блока.
            // KV-кэш резидентен на device, lm_head/embed/rope тоже.
            let ord = if let Device::Cuda(o) = dev { o } else { 0 };
            synaptix_core::device::cuda::set_offload_pinned(true);
            let ls = synaptix_core::device::cuda::loader_stream(ord)
                .map_err(|e| ModelError::Forward(e.to_string()))?;
            let mut cur = self.blocks[0].to_device(dev)?;
            for idx in 0..self.blocks.len() {
                let lsc = ls.clone();
                let next: Option<Block> = std::thread::scope(
                    |sp| -> Result<Option<Block>, ModelError> {
                        let h = if idx + 1 < self.blocks.len() {
                            Some(sp.spawn(move || -> Result<Block, ModelError> {
                                synaptix_core::device::cuda::set_alloc_stream(Some(lsc.clone()));
                                synaptix_core::device::cuda::set_offload_pinned(true);
                                let r = self.blocks[idx + 1].to_device(dev);
                                let _ = lsc.synchronize();
                                synaptix_core::device::cuda::set_offload_pinned(false);
                                synaptix_core::device::cuda::set_alloc_stream(None);
                                r
                            }))
                        } else {
                            None
                        };
                        hidden = step(idx, &cur, &hidden, kv_cache)?;
                        match h {
                            Some(h) => Ok(Some(h.join().map_err(|_| {
                                ModelError::Forward("llm prefetch thread panicked".into())
                            })??)),
                            None => Ok(None),
                        }
                    },
                )?;
                if let Some(nb) = next {
                    cur = nb;
                }
            }
            synaptix_core::device::cuda::set_offload_pinned(false);
        } else {
            let sync_ord = match self.device {
                Device::Cuda(o) => Some(o),
                _ => None,
            };
            for (idx, blk) in self.blocks.iter().enumerate() {
                hidden = step(idx, blk, &hidden, kv_cache)?;
                if let Some(o) = sync_ord {
                    synaptix_core::device::cuda::layer_sync(o, s > 1);
                }
            }
        }
        kv_cache.seq_len = past + s;
        Ok(hidden)
    }

    /// Энкодер-проход: вернуть ВСЕ hidden states как HF `output_hidden_states=True`:
    /// `[emb после ×embed_scale, выход слоёв 0..N-2, final_norm(выход слоя N-1)]` —
    /// итого `num_hidden_layers + 1` тензоров формы `[B, S, hidden]`. Без narrow
    /// последнего токена и без lm_head (нужно для текст-кондишена LTX-2.3).
    /// `attention_mask` (`[B,S]`, 1=valid / 0=pad) → аддитивная key-padding маска
    /// (LTX Gemma токенизирует left-pad); позиции RoPE — абсолютные `[0..S)` (HF при
    /// паддинге не сдвигает position_ids). Только full-attention модели (как Gemma).
    pub fn forward_hidden_states(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>, ModelError> {
        if input_ids.rank() != 2 {
            return Err(ModelError::Shape(format!("input_ids must be [B, S], got {:?}", input_ids.dims())));
        }
        let batch = input_ids.dims()[0];
        let s = input_ids.dims()[1];
        let dev = self.device;
        let mut kv = self.make_kv_cache(batch, s)?;

        // key-padding bias [1,S]: 0 для valid, MASK_NEG для pad → broadcast_add к
        // causal-маске [s,s] внутри FullAttn (scaled_dot_attention бродкастит [s,s]
        // по batch/heads). (mask-1)*|MASK_NEG|: 1→0, 0→MASK_NEG. Энкодер LTX подаёт
        // один промпт за раз (batch=1) — общая [s,s]-маска корректна.
        if attention_mask.is_some() && batch != 1 {
            return Err(ModelError::Shape("forward_hidden_states: key-padding поддержан только для batch=1".into()));
        }
        let pad_bias = match attention_mask {
            Some(m) => {
                let m = m.reshape(vec![1, s]).coerr()?.to_dtype(DType::F32).coerr()?;
                let b = m.add_scalar(-1.0).coerr()?.mul_scalar(-MASK_NEG).coerr()?;
                Some(b.to_dtype(self.dtype).coerr()?)
            }
            None => None,
        };
        let pad_ref = pad_bias.as_ref();

        let mut hidden = self.embed_tokens(input_ids)?;
        if let Some(scale) = self.embed_scale {
            hidden = hidden.mul_scalar(scale).coerr()?;
        }
        let mut states: Vec<Tensor> = Vec::with_capacity(self.blocks.len() + 1);
        let mut step = |idx: usize, blk: &Block, hidden: &Tensor, kv: &mut KvCache|
            -> Result<Tensor, ModelError> {
            states.push(hidden.clone()); // HF hidden_states[idx] = вход блока idx
            let residual = hidden.clone();
            let h = rms_norm(hidden, &blk.pre_attn_norm, blk.rms_eps).coerr()?;
            let mixed = match &blk.mixer {
                Mixer::Full(fa) => fa.forward(
                    &h, &mut kv.layers[idx], 0, s, batch, self.rope_at(idx),
                    self.kv_dtype, dev, self.dtype, pad_ref,
                )?,
                Mixer::Linear(_) => {
                    return Err(ModelError::Forward(
                        "forward_hidden_states: linear-слои не поддержаны (только full-attention)".into(),
                    ))
                }
            };
            let mixed = apply_opt_norm(&mixed, blk.post_attn_norm.as_ref(), blk.rms_eps)?;
            let hidden = residual.add(&mixed).coerr()?;

            let residual2 = hidden.clone();
            let h = rms_norm(&hidden, &blk.pre_mlp_norm, blk.rms_eps).coerr()?;
            let mlp_out = blk.mlp.forward(&h)?;
            let mlp_out = apply_opt_norm(&mlp_out, blk.post_mlp_norm.as_ref(), blk.rms_eps)?;
            residual2.add(&mlp_out).coerr()
        };
        if self.host_stream_blocks && matches!(dev, Device::Cuda(_)) {
            // host-stream: CPU-блоки стримятся на GPU per-block, префетч i+1 на
            // loader-стриме с pinned-H2D (как DiT-блоки LTX). Только прямой
            // проход — KV-кэш одноразовый.
            let ord = if let Device::Cuda(o) = dev { o } else { 0 };
            synaptix_core::device::cuda::set_offload_pinned(true);
            let ls = synaptix_core::device::cuda::loader_stream(ord)
                .map_err(|e| ModelError::Forward(e.to_string()))?;
            let mut cur = self.blocks[0].to_device(dev)?;
            for idx in 0..self.blocks.len() {
                let lsc = ls.clone();
                let next: Option<Block> = std::thread::scope(
                    |sp| -> Result<Option<Block>, ModelError> {
                        let h = if idx + 1 < self.blocks.len() {
                            Some(sp.spawn(move || -> Result<Block, ModelError> {
                                synaptix_core::device::cuda::set_alloc_stream(Some(lsc.clone()));
                                synaptix_core::device::cuda::set_offload_pinned(true);
                                let r = self.blocks[idx + 1].to_device(dev);
                                let _ = lsc.synchronize();
                                synaptix_core::device::cuda::set_offload_pinned(false);
                                synaptix_core::device::cuda::set_alloc_stream(None);
                                r
                            }))
                        } else {
                            None
                        };
                        hidden = step(idx, &cur, &hidden, &mut kv)?;
                        match h {
                            Some(h) => Ok(Some(h.join().map_err(|_| {
                                ModelError::Forward("llm prefetch thread panicked".into())
                            })??)),
                            None => Ok(None),
                        }
                    },
                )?;
                if let Some(nb) = next {
                    cur = nb;
                }
            }
            synaptix_core::device::cuda::set_offload_pinned(false);
        } else {
            for (idx, blk) in self.blocks.iter().enumerate() {
                hidden = step(idx, blk, &hidden, &mut kv)?;
            }
        }
        // HF: последнее состояние = final_norm(выход последнего слоя).
        states.push(rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps).coerr()?);
        Ok(states)
    }

    pub fn make_decode_state(&self) -> Result<DecodeState, ModelError> {
        self.make_decode_state_batched(1)
    }

    /// Batched decode state for `batch` independent rows (per-row token +
    /// position). `batch==1` is the ordinary single-sequence decode.
    pub fn make_decode_state_batched(&self, batch: usize) -> Result<DecodeState, ModelError> {
        if !self.config.graph_decode_ok() {
            return Err(ModelError::Forward("make_decode_state: профиль не поддержан CUDA-graph (sandwich/sliding/local-rope)".into()));
        }
        let dev = self.device;
        let input = Tensor::from_vec(vec![0u32; batch], vec![batch, 1], dev).coerr()?;
        let pos_dev = Tensor::from_vec(vec![0u32; batch], vec![batch], dev).coerr()?;
        let tcache_dev = Tensor::from_vec(vec![0u32; batch], vec![batch], dev).coerr()?;
        let cos = self.rope_global.cos();
        let sin = self.rope_global.sin();
        let rope_cos = Tensor::cat(&[cos, cos], 1).coerr()?.to_dtype(self.dtype).coerr()?;
        let rope_sin = Tensor::cat(&[sin, sin], 1).coerr()?.to_dtype(self.dtype).coerr()?;
        let logits = Tensor::zeros(vec![batch, self.config.vocab_size], self.dtype, dev).coerr()?;
        Ok(DecodeState { input, pos_dev, tcache_dev, rope_cos, rope_sin, logits })
    }

    /// Засеять/восстановить device-зеркала linear-слоёв из host-состояния KV.
    /// Вызывается (а) после prefill — создать dev-state из накопленного host-scan;
    /// (б) после graph-capture — восстановить S0 (capture/warmup продвинули
    /// dev-state, т.к. рекуррентность НЕ идемпотентна, а host-векторы не тронуты).
    /// No-op для моделей без linear-слоёв.
    pub fn sync_decode_dev_state(&self, kv: &mut KvCache) -> Result<(), ModelError> {
        let lin = match self.config.linear.as_ref() {
            Some(l) => l,
            None => return Ok(()),
        };
        for lc in kv.layers.iter_mut() {
            if let LayerCache::Linear(s) = lc {
                s.sync_to_device(
                    self.device,
                    lin.conv_dim(),
                    lin.conv_kernel,
                    lin.num_value_heads,
                    lin.key_head_dim,
                    lin.value_head_dim,
                )
                .coerr()?;
            }
        }
        Ok(())
    }

    /// Обратный синк device→host linear-состояния (после graph-decode). Нужен для
    /// prefix-KV-кэша: следующий ход продолжает host-scan с верного состояния.
    /// No-op для моделей без linear-слоёв.
    pub fn sync_decode_host_state(&self, kv: &mut KvCache) -> Result<(), ModelError> {
        if self.config.linear.is_none() {
            return Ok(());
        }
        for lc in kv.layers.iter_mut() {
            if let LayerCache::Linear(s) = lc {
                s.sync_to_host().coerr()?;
            }
        }
        Ok(())
    }

    /// Есть ли в модели linear-слои (GatedDeltaNet). Для них рекуррентное состояние
    /// нельзя «отмотать» к произвольному префиксу — кэш переиспользуем только как
    /// полное расширение последовательности.
    pub fn has_linear_layers(&self) -> bool {
        self.config.linear.is_some()
    }

    pub fn forward_decode_dev(&self, state: &mut DecodeState, kv: &mut KvCache) -> Result<(), ModelError> {
        if !self.config.graph_decode_ok() {
            return Err(ModelError::Forward("forward_decode_dev: профиль не поддержан (sandwich/sliding/local-rope)".into()));
        }
        let dev = self.device;
        // Batch B: state.input is [B, 1]; B>1 runs a batched decode (e.g. CFG
        // cond+uncond in one forward) with per-row positions in state.pos_dev.
        let b = state.input.dims()[0];
        let ids_flat = state.input.reshape(vec![b]).coerr()?;
        let emb = prof(dev, "embed_gather", || self.embed_rows(&ids_flat))?;
        let mut hidden = emb.reshape(vec![b, 1, self.config.hidden_size]).coerr()?;
        if let Some(scale) = self.embed_scale {
            hidden = hidden.mul_scalar(scale).coerr()?;
        }

        // Fused residual+norm цепочкой: каждая пара (add, следующий rms_norm)
        // сливается в один launch (rmsnorm_residual). `h` — нормированный вход в
        // attn слоя idx; внутри слоя сначала fuse(add_attn, mlp_norm), затем
        // fuse(add_mlp, next_attn_norm). Стартовый attn-norm слоя 0 — отдельно.
        let nb = self.blocks.len();
        let mut h = prof(dev, "rms_norm", || {
            rms_norm(&hidden, &self.blocks[0].pre_attn_norm, self.blocks[0].rms_eps)
        })
        .coerr()?;
        for idx in 0..nb {
            let blk = &self.blocks[idx];
            let mixed = match &blk.mixer {
                Mixer::Full(fa) => fa.forward_decode_dev(&h, &mut kv.layers[idx], state)?,
                Mixer::Linear(la) => la.forward_decode_dev(&h, &mut kv.layers[idx])?,
            };
            // hidden = hidden + mixed; h_mlp = norm(hidden, pre_mlp).
            let (new_hidden, h_mlp) =
                fused_add_norm(dev, &mixed, &hidden, &blk.pre_mlp_norm, blk.rms_eps)?;
            hidden = new_hidden;
            let mlp_out = blk.mlp.forward(&h_mlp)?;
            if idx + 1 < nb {
                // hidden = hidden + mlp_out; h = norm(hidden, next.pre_attn).
                let nb_next = &self.blocks[idx + 1];
                let (new_hidden2, h_next) =
                    fused_add_norm(dev, &mlp_out, &hidden, &nb_next.pre_attn_norm, nb_next.rms_eps)?;
                hidden = new_hidden2;
                h = h_next;
            } else {
                hidden = prof(dev, "residual_add", || hidden.add(&mlp_out)).coerr()?;
            }
        }

        let normed = prof(dev, "rms_norm", || rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)).coerr()?;
        let last = normed.narrow(1, 0, 1).coerr()?.squeeze(1).coerr()?;
        let logits = prof(dev, "lm_head", || self.lm_head.forward(&last))?;
        prof(dev, "logits_copy", || state.logits.copy_from(&logits)).coerr()?;
        Ok(())
    }

    /// Аллоцирует [`PrefillState`] для фиксированного `chunk_size`. Все буферы
    /// device-резидентные и стабильно адресуемые → один capture валиден для всех
    /// последующих replay'ев (с разными ids/`pos_start`, обновляемыми через
    /// [`PrefillState::update`]).
    ///
    /// Доступно только для профилей, проходящих [`DecoderConfig::graph_prefill_ok`]
    /// (full-attn only, без hybrid). RoPE-таблицы дублируются (`[capacity,
    /// rotary_dim]`, dtype = compute) — формат, который ждёт `rope_apply_dev`.
    pub fn make_prefill_state(&self, chunk_size: usize) -> Result<PrefillState, ModelError> {
        if !self.config.graph_prefill_ok() {
            return Err(ModelError::Forward(
                "make_prefill_state: профиль не поддержан CUDA-graph prefill (sandwich/sliding/local-rope/hybrid)".into(),
            ));
        }
        if chunk_size == 0 {
            return Err(ModelError::Shape("make_prefill_state: chunk_size > 0".into()));
        }
        if self.kv_dtype == DType::MXFP8 {
            return Err(ModelError::Forward(
                "make_prefill_state: FP8-KV не поддержан dev-путём".into(),
            ));
        }
        let dev = self.device;
        let input = Tensor::from_vec(vec![0u32; chunk_size], vec![1usize, chunk_size], dev).coerr()?;
        let pos_start = Tensor::from_vec(vec![0u32], vec![1usize], dev).coerr()?;
        let tcache_dev = Tensor::from_vec(vec![0u32], vec![1usize], dev).coerr()?;
        let cos = self.rope_global.cos();
        let sin = self.rope_global.sin();
        let rope_cos = Tensor::cat(&[cos, cos], 1).coerr()?.to_dtype(self.dtype).coerr()?;
        let rope_sin = Tensor::cat(&[sin, sin], 1).coerr()?.to_dtype(self.dtype).coerr()?;
        let logits =
            Tensor::zeros(vec![chunk_size, self.config.vocab_size], self.dtype, dev).coerr()?;
        let hidden =
            Tensor::zeros(vec![chunk_size, self.config.hidden_size], self.dtype, dev).coerr()?;
        Ok(PrefillState {
            chunk_size,
            input,
            pos_start,
            tcache_dev,
            rope_cos,
            rope_sin,
            logits,
            hidden,
        })
    }

    /// Device-резидентный prefill одного chunk'а (T = `state.chunk_size`). Аналог
    /// [`Self::forward_decode_dev`], но обрабатывает batch T токенов за один проход
    /// и пишет в KV `state.chunk_size` новых слотов. Все позиционно-зависимые
    /// параметры (RoPE start, KV append slot, активная длина KV для causal-mask
    /// flash-decode) — device-резидентные U32-буферы, обновляются между replay'ями
    /// через [`PrefillState::update`]. Один capture валиден для всех полных
    /// chunk'ов одного размера в пределах prefill'а.
    ///
    /// Главное упрощение vs план: kernel-сторона `rope_apply_dev`/`kv_append_dev`/
    /// `flash_attention_dev` уже T>1-aware — `t = idx % T_seq`, `pos = start_pos + t`,
    /// `q_pos = Tkv - Tq + ti`. Никаких kernel-правок не потребовалось.
    pub fn forward_prefill_dev(&self, state: &mut PrefillState, kv: &mut KvCache) -> Result<(), ModelError> {
        if !self.config.graph_prefill_ok() {
            return Err(ModelError::Forward(
                "forward_prefill_dev: профиль не поддержан (sandwich/sliding/local-rope/hybrid)".into(),
            ));
        }
        if self.kv_dtype == DType::MXFP8 {
            return Err(ModelError::Forward("forward_prefill_dev: FP8-KV не поддержан dev-путём".into()));
        }
        let chunk = state.chunk_size;
        let ids_flat = state.input.reshape(vec![chunk]).coerr()?;
        let emb = self.embed_rows(&ids_flat)?;
        let mut hidden = emb.reshape(vec![1usize, chunk, self.config.hidden_size]).coerr()?;
        if let Some(scale) = self.embed_scale {
            hidden = hidden.mul_scalar(scale).coerr()?;
        }

        for (idx, blk) in self.blocks.iter().enumerate() {
            let residual = hidden.clone();
            let want_attn = match &blk.mixer {
                Mixer::Full(fa) => fa.q_proj.quant_dtype(),
                _ => None,
            };
            let (h, pq) = rms_norm_quant(&hidden, &blk.pre_attn_norm, blk.rms_eps, want_attn)?;
            let mixed = match &blk.mixer {
                Mixer::Full(fa) => fa
                    .forward_prefill_dev(&h, &mut kv.layers[idx], state, pq.as_ref())
                    .map_err(|e| ModelError::Forward(format!("prefill_dev full[{idx}]: {e}")))?,
                Mixer::Linear(la) => la
                    .forward_prefill_dev(&h, &mut kv.layers[idx])
                    .map_err(|e| ModelError::Forward(format!("prefill_dev linear[{idx}]: {e}")))?,
            };
            hidden = residual.add(&mixed).coerr()?;

            let residual2 = hidden.clone();
            let want_mlp = blk.mlp.gate_proj.quant_dtype();
            let (h, pq2) = rms_norm_quant(&hidden, &blk.pre_mlp_norm, blk.rms_eps, want_mlp)?;
            let mlp_out = blk.mlp.forward_pq(&h, pq2.as_ref())?;
            hidden = residual2.add(&mlp_out).coerr()?;
        }

        let trunk = hidden
            .contiguous()
            .coerr()?
            .reshape(vec![chunk, self.config.hidden_size])
            .coerr()?;
        state
            .hidden
            .copy_from(&trunk)
            .map_err(|e| ModelError::Forward(format!("prefill_dev hidden copy: {e}")))?;
        let normed = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps).coerr()?;
        let rows = normed.reshape(vec![chunk, self.config.hidden_size]).coerr()?;
        let logits = self.lm_head.forward(&rows)?;
        state.logits.copy_from(&logits).coerr()?;
        Ok(())
    }
}

impl Mlp {
    fn forward(&self, h: &Tensor) -> Result<Tensor, ModelError> {
        self.forward_pq(h, None)
    }
    fn forward_pq(
        &self,
        h: &Tensor,
        pq: Option<&(Tensor, Tensor, DType)>,
    ) -> Result<Tensor, ModelError> {
        let dev = h.device();
        // gate и up берут один и тот же `h`: prequant из эпилога нормы, иначе 1×.
        let act = pq.cloned().or_else(|| quant_act_shared(h));
        let gate = proj_shared(&self.gate_proj, h, &act, dev, "mlp_gate")?;
        let up = proj_shared(&self.up_proj, h, &act, dev, "mlp_up")?;
        let gated = prof(dev, "mlp_act", || match self.activation {
            Activation::Silu => match gate.silu_and_mul(&up) {
                Ok(g) => Ok(g),
                Err(SynaptixError::Unsupported(_)) => {
                    Ok(gate.silu().coerr()?.mul(&up).coerr()?)
                }
                Err(e) => Err(ModelError::Forward(e.to_string())),
            },
            Activation::GeluTanh => Ok(gate.gelu_tanh().coerr()?.mul(&up).coerr()?),
        })?;
        prof(dev, "mlp_down", || self.down_proj.forward(&gated))
    }
}

impl FullAttn {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        h: &Tensor,
        cache: &mut LayerCache,
        past: usize,
        s: usize,
        batch: usize,
        rope: &RopeCache,
        kv_dtype: DType,
        device: Device,
        compute: DType,
        pad_bias: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        let kv = match cache {
            LayerCache::Full(k) => k,
            LayerCache::Linear(_) => return Err(ModelError::Shape("full layer got linear cache".into())),
        };
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

        let qg = prof(device, "attn_qproj", || self.q_proj.forward(h))?;
        let (q, gate) = if self.attn_output_gate {
            let qg = qg.reshape(vec![batch, s, nh, 2 * hd]).coerr()?;
            let q = qg.narrow(3, 0, hd).coerr()?.contiguous().coerr()?;
            let gate = qg.narrow(3, hd, hd).coerr()?.contiguous().coerr()?;
            (q, Some(gate))
        } else {
            (qg.reshape(vec![batch, s, nh, hd]).coerr()?, None)
        };
        let q = q.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let k = prof(device, "attn_kproj", || self.k_proj.forward(h))?.reshape(vec![batch, s, nkv, hd]).coerr()?
            .permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let v = prof(device, "attn_vproj", || self.v_proj.forward(h))?.reshape(vec![batch, s, nkv, hd]).coerr()?
            .permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;

        let q = prof(device, "attn_qknorm", || apply_opt_head_norm(&q, self.q_norm.as_ref(), self.rms_eps))?;
        let k = prof(device, "attn_qknorm", || apply_opt_head_norm(&k, self.k_norm.as_ref(), self.rms_eps))?;

        let q = prof(device, "attn_rope", || partial_rope(&q, rope, past, s, self.rotary_dim, hd).coerr())?;
        let k = prof(device, "attn_rope", || partial_rope(&k, rope, past, s, self.rotary_dim, hd).coerr())?;

        let new_len = past + s;
        let group = nh / nkv;
        // pad_bias (key-padding для энкодера) несовместим с flash (flash маскирует
        // только чисто-causal) → форсим sdpa-путь, где маску можно дополнить.
        let flash_eligible =
            self.use_flash && self.sliding_window.is_none() && pad_bias.is_none();
        let _core_dev = device;
        let attn = prof(_core_dev, "attn_core", || -> Result<Tensor, ModelError> {
        Ok(if flash_eligible && kv_dtype == DType::MXFP8 {
            let KvCacheLayer { k: kc, v: vc, k_scale: ksc, v_scale: vsc } = kv;
            kc.kv_append_quant_mxfp8_inplace(ksc.as_mut().unwrap(), &k, past).coerr()?;
            vc.kv_append_quant_mxfp8_inplace(vsc.as_mut().unwrap(), &v, past).coerr()?;
            let k_q = kc.narrow(2, 0, new_len).coerr()?;
            let v_q = vc.narrow(2, 0, new_len).coerr()?;
            // narrow по dim-2 (max_seq); блочная ось hd/32 — dim-3, не задета.
            let ks = ksc.as_ref().unwrap().narrow(2, 0, new_len).coerr()?;
            let vs = vsc.as_ref().unwrap().narrow(2, 0, new_len).coerr()?;
            q.flash_attention_mxfp8kv(&k_q, &v_q, &ks, &vs, self.attn_scale, true)
                .map_err(|e| ModelError::Forward(e.to_string()))?
        } else {
            kv.k.kv_append_inplace(&k, past).coerr()?;
            kv.v.kv_append_inplace(&v, past).coerr()?;
            let k_total = kv.k.narrow(2, 0, new_len).coerr()?;
            let v_total = kv.v.narrow(2, 0, new_len).coerr()?;
            let do_flash = flash_eligible;
            let flashed = if do_flash {
                match q.flash_attention(&k_total, &v_total, self.attn_scale, true) {
                    Ok(a) => Some(a),
                    Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => None,
                    Err(e) => return Err(ModelError::Forward(e.to_string())),
                }
            } else {
                None
            };
            match flashed {
                Some(a) => a,
                None => {
                    let k_rep = repeat_kv(&k_total, group).coerr()?;
                    let v_rep = repeat_kv(&v_total, group).coerr()?;
                    let window = self.sliding_window;
                    if s == 1 && window.is_none() && pad_bias.is_none() {
                        scaled_dot_attention(&q, &k_rep, &v_rep, self.attn_scale, None).coerr()?
                    } else {
                        let mask = build_mask(s, new_len, past, window, device, compute).coerr()?;
                        let mask = match pad_bias {
                            Some(pb) => mask.broadcast_add(pb).coerr()?,
                            None => mask,
                        };
                        scaled_dot_attention(&q, &k_rep, &v_rep, self.attn_scale, Some(&mask)).coerr()?
                    }
                }
            }
        })
        })?;

        let attn = attn.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let attn = match gate {
            Some(g) => attn.mul(&g.sigmoid().coerr()?).coerr()?,
            None => attn,
        };
        let attn = attn.reshape(vec![batch, s, nh * hd]).coerr()?;
        prof(device, "attn_oproj", || self.o_proj.forward(&attn))
    }

    /// Device-резидентный decode-шаг (T=1) для CUDA-graph. Как [`Self::forward`]
    /// при s=1, но: позиция/длина KV — device-резидентные (`state.pos_dev`/
    /// `tcache_dev`), RoPE и flash через `*_dev`-ядра. Поддерживает partial-RoPE
    /// (`rotary_dim < head_dim`), Q/K-norm и attn-output-gate. Без host round-trip.
    fn forward_decode_dev(
        &self,
        h: &Tensor,
        cache: &mut LayerCache,
        state: &DecodeState,
    ) -> Result<Tensor, ModelError> {
        let kvl = match cache {
            LayerCache::Full(k) => k,
            LayerCache::Linear(_) => return Err(ModelError::Shape("full layer got linear cache".into())),
        };
        let dev = h.device();
        // Batch B (>1 for batched CFG decode); seq dim stays 1. Per-row RoPE/KV
        // positions come from `state.pos_dev`/`tcache_dev` ([B]).
        let b = h.dims()[0];
        // q/k/v берут один `h` → квантуем 1× и переиспользуем во всех трёх.
        let act = quant_act_shared(h);
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let qg = proj_shared(&self.q_proj, h, &act, dev, "attn_qproj")?;
        let (q, gate) = if self.attn_output_gate {
            let qg = qg.reshape(vec![b, 1, nh, 2 * hd]).coerr()?;
            let q = qg.narrow(3, 0, hd).coerr()?.contiguous().coerr()?;
            let gate = qg.narrow(3, hd, hd).coerr()?.contiguous().coerr()?;
            (q, Some(gate))
        } else {
            (qg.reshape(vec![b, 1, nh, hd]).coerr()?, None)
        };
        let q = q.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let k = proj_shared(&self.k_proj, h, &act, dev, "attn_kproj")?
            .reshape(vec![b, 1, nkv, hd]).coerr()?.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let v = proj_shared(&self.v_proj, h, &act, dev, "attn_vproj")?
            .reshape(vec![b, 1, nkv, hd]).coerr()?.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let q = prof(dev, "attn_qknorm", || apply_opt_head_norm(&q, self.q_norm.as_ref(), self.rms_eps))?;
        let k = prof(dev, "attn_qknorm", || apply_opt_head_norm(&k, self.k_norm.as_ref(), self.rms_eps))?;
        let q = prof(dev, "attn_rope", || q.rope_apply_dev(&state.rope_cos, &state.rope_sin, &state.pos_dev, self.rotary_dim)).coerr()?;
        let k = prof(dev, "attn_rope", || k.rope_apply_dev(&state.rope_cos, &state.rope_sin, &state.pos_dev, self.rotary_dim)).coerr()?;

        let attn = if kvl.k.dtype() == DType::MXFP8 {
            let KvCacheLayer { k: kc, v: vc, k_scale: ksc, v_scale: vsc } = kvl;
            prof(dev, "attn_kv_append", || -> Result<(), ModelError> {
                kc.kv_append_quant_mxfp8_dev(ksc.as_mut().unwrap(), &k, &state.pos_dev).coerr()?;
                vc.kv_append_quant_mxfp8_dev(vsc.as_mut().unwrap(), &v, &state.pos_dev).coerr()
            })?;
            prof(dev, "attn_flash", || q.flash_attention_mxfp8kv_dev(
                kc,
                vc,
                ksc.as_ref().unwrap(),
                vsc.as_ref().unwrap(),
                &state.tcache_dev,
                self.attn_scale,
                true,
            ))
            .map_err(|e| ModelError::Forward(e.to_string()))?
        } else {
            prof(dev, "attn_kv_append", || -> Result<(), ModelError> {
                kvl.k.kv_append_dev(&k, &state.pos_dev).coerr()?;
                kvl.v.kv_append_dev(&v, &state.pos_dev).coerr()
            })?;
            prof(dev, "attn_flash", || q.flash_attention_dev(&kvl.k, &kvl.v, &state.tcache_dev, self.attn_scale, true))
                .map_err(|e| ModelError::Forward(e.to_string()))?
        };
        let attn = attn.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let attn = match gate {
            Some(g) => prof(dev, "attn_gate", || attn.mul(&g.sigmoid()?)).coerr()?,
            None => attn,
        };
        let attn = attn.reshape(vec![b, 1, nh * hd]).coerr()?;
        prof(dev, "attn_oproj", || self.o_proj.forward(&attn))
    }

    /// Device-резидентный prefill-шаг (T = `state.chunk_size`) для CUDA-graph.
    /// Структурно зеркалит [`Self::forward_decode_dev`]; отличия от decode:
    /// - `hidden` имеет форму `[1, T, hidden]` (T = chunk_size, decode = 1);
    /// - `rope_apply_dev`/`kv_append_dev`/`flash_attention_dev` обрабатывают T
    ///   токенов в одном launch'е (ядра уже `t = idx % T`-aware), позиция первого
    ///   токена — `state.pos_start`, активная длина KV для causal-mask —
    ///   `state.tcache_dev = pos_start + T`;
    /// - causal-mask в flash формируется автоматически по формуле
    ///   `q_pos[ti] = Tkv - T + ti = pos_start + ti`, q[ti] видит k[0..q_pos]
    ///   (т.е. себя и весь префикс).
    fn forward_prefill_dev(
        &self,
        h: &Tensor,
        cache: &mut LayerCache,
        state: &PrefillState,
        pq: Option<&(Tensor, Tensor, DType)>,
    ) -> Result<Tensor, ModelError> {
        let kvl = match cache {
            LayerCache::Full(k) => k,
            LayerCache::Linear(_) => return Err(ModelError::Shape("full layer got linear cache".into())),
        };
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let t = state.chunk_size;
        let dev = h.device();
        // q/k/v шарят prequant из эпилога нормы (раньше квантовали h ТРИЖДЫ).
        let act = pq.cloned();
        let qg = proj_shared(&self.q_proj, h, &act, dev, "attn_qproj")?;
        let (q, gate) = if self.attn_output_gate {
            let qg = qg.reshape(vec![1, t, nh, 2 * hd]).coerr()?;
            let q = qg.narrow(3, 0, hd).coerr()?.contiguous().coerr()?;
            let gate = qg.narrow(3, hd, hd).coerr()?.contiguous().coerr()?;
            (q, Some(gate))
        } else {
            (qg.reshape(vec![1, t, nh, hd]).coerr()?, None)
        };
        let q = q.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let k = proj_shared(&self.k_proj, h, &act, dev, "attn_kproj")?
            .reshape(vec![1, t, nkv, hd]).coerr()?.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let v = proj_shared(&self.v_proj, h, &act, dev, "attn_vproj")?
            .reshape(vec![1, t, nkv, hd]).coerr()?.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let q = apply_opt_head_norm(&q, self.q_norm.as_ref(), self.rms_eps)?;
        let k = apply_opt_head_norm(&k, self.k_norm.as_ref(), self.rms_eps)?;
        let q = q.rope_apply_dev(&state.rope_cos, &state.rope_sin, &state.pos_start, self.rotary_dim).coerr()?;
        let k = k.rope_apply_dev(&state.rope_cos, &state.rope_sin, &state.pos_start, self.rotary_dim).coerr()?;

        kvl.k.kv_append_dev(&k, &state.pos_start).coerr()?;
        kvl.v.kv_append_dev(&v, &state.pos_start).coerr()?;
        // Prefill (Tq>1) → FA-4 device-resident-Tkv (Q-тайлы по BM=16, WMMA m16n8k16).
        // `flash_attention_dev` (= flash_decode_split) — decode-only: split по KV, без
        // Q-тайлинга → ~4× медленнее на Tq=256. Здесь нужен именно prefill-вариант.
        let attn = q
            .flash_attention_prefill_dev(&kvl.k, &kvl.v, &state.tcache_dev, self.attn_scale, true)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let attn = attn.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
        let attn = match gate {
            Some(g) => attn.mul(&g.sigmoid().coerr()?).coerr()?,
            None => attn,
        };
        let attn = attn.reshape(vec![1, t, nh * hd]).coerr()?;
        self.o_proj.forward(&attn)
    }
}

impl LinearAttn {
    fn forward(&self, h: &Tensor, cache: &mut LayerCache, s: usize, device: Device, compute: DType) -> Result<Tensor, ModelError> {
        let (dk, dv, h_v, h_k, conv_dim, k) = (self.dk, self.dv, self.num_v_heads, self.num_k_heads, self.conv_dim, self.conv_k);

        // CUDA fast-path: device-резидентная цепочка (chunk_conv1d + silu +
        // prep_scatter + chunk_gated_delta_rule) одним Backend op'ом — без
        // host_vec'ов на qkv/a/b и без host scatter qe/ke/vv. Требует _dev
        // веса (build в non-CPU) и compute = F16/BF16/F32.
        if matches!(device, Device::Cuda(_))
            && self.conv_w_dev.is_some()
            && self.a_log_dev.is_some()
            && self.dt_bias_dev.is_some()
            && matches!(compute, DType::F16 | DType::BF16 | DType::F32)
        {
            if s <= SMALL_CHUNK_DEV && compute == DType::F16 && self.norm_w_f16.is_some() {
                let mut parts = Vec::with_capacity(s);
                for t in 0..s {
                    let ht = h.narrow(1, t, 1).coerr()?.contiguous().coerr()?;
                    parts.push(self.forward_decode_dev(&ht, cache)?);
                }
                let refs: Vec<&Tensor> = parts.iter().collect();
                return Tensor::cat(&refs, 1).coerr();
            }
            let state = match cache {
                LayerCache::Linear(s) => s,
                LayerCache::Full(_) => {
                    return Err(ModelError::Shape("linear layer got full cache".into()))
                }
            };
            return self.forward_cuda_chunk_prefill(h, state, s, dk, dv, h_v, h_k, conv_dim, k, device, compute);
        }
        let state = match cache {
            LayerCache::Linear(s) => s,
            LayerCache::Full(_) => return Err(ModelError::Shape("linear layer got full cache".into())),
        };

        // CPU path (host-mix): полная host цепочка, как раньше.
        let dbg = dump_layers_on();
        let group = h_v / h_k;
        if dbg { record_layer_norm(1003, "h_in", h, s, 0); }
        let qkv = self.in_proj_qkv.forward(h)?;
        if dbg { record_layer_norm(1000, "qkv", &qkv, s, 0); }
        let qkv_v = host_vec(&qkv)?;
        let mut conv_out = causal_conv1d_stateful(&mut state.conv_state, &qkv_v, &self.conv_w, s, conv_dim, k);
        for x in conv_out.iter_mut() {
            *x /= 1.0 + (-*x).exp();
        }

        let a_v = host_vec(&self.in_proj_a.forward(h)?)?;
        let b_v = host_vec(&self.in_proj_b.forward(h)?)?;
        let (g, beta) = gated_delta_decay_beta(&a_v, &b_v, &self.a_log, &self.dt_bias, s, h_v);

        let mut qe = vec![0.0f32; h_v * s * dk];
        let mut ke = vec![0.0f32; h_v * s * dk];
        let mut vv = vec![0.0f32; h_v * s * dv];
        let v_off0 = self.key_dim * 2;
        for hi in 0..h_v {
            let kh = hi / group;
            for t in 0..s {
                let row = t * conv_dim;
                let qsrc = row + kh * dk;
                let ksrc = row + self.key_dim + kh * dk;
                let vsrc = row + v_off0 + hi * dv;
                let qdst = (hi * s + t) * dk;
                let vdst = (hi * s + t) * dv;
                qe[qdst..qdst + dk].copy_from_slice(&conv_out[qsrc..qsrc + dk]);
                ke[qdst..qdst + dk].copy_from_slice(&conv_out[ksrc..ksrc + dk]);
                vv[vdst..vdst + dv].copy_from_slice(&conv_out[vsrc..vsrc + dv]);
            }
        }
        let core = gated_delta_net_recurrent(
            &mut state.ssm_state, &qe, &ke, &vv, &g, &beta, h_v, s, dk, dv, self.q_scale,
        );
        let mut core_sh = vec![0.0f32; s * h_v * dv];
        for hi in 0..h_v {
            for t in 0..s {
                let src = (hi * s + t) * dv;
                let dst = (t * h_v + hi) * dv;
                core_sh[dst..dst + dv].copy_from_slice(&core[src..src + dv]);
            }
        }
        let core_t = Tensor::from_vec(core_sh, vec![1, s, h_v, dv], device).coerr()?.to_dtype(compute).coerr()?;
        if dbg { record_layer_norm(1002, "core", &core_t.reshape(vec![1, s, self.value_dim]).coerr()?, s, 0); }
        let z = self.in_proj_z.forward(h)?.reshape(vec![1, s, h_v, dv]).coerr()?;
        let normed = rms_norm(&core_t, &self.norm_weight, self.rms_eps).coerr()?;
        let normed = normed.mul(&z.silu().coerr()?).coerr()?;
        let normed = normed.reshape(vec![1, s, self.value_dim]).coerr()?;
        self.out_proj.forward(&normed)
    }

    /// Device-резидентный prefill через `Tensor::linear_attn_chunk_prefill`.
    /// Заменяет 4 host_vec'а (qkv/a/b/conv_state) + scatter-loop одним Backend
    /// op'ом. conv_state/ssm_state мигрируют host↔device временно. Bit-exact
    /// против host-mix пути для F32; для F16/BF16 compute — квант-tolerance.
    #[allow(clippy::too_many_arguments)]
    fn forward_cuda_chunk_prefill(
        &self,
        h: &Tensor,
        state: &mut GatedDeltaNetState,
        s: usize,
        dk: usize,
        dv: usize,
        h_v: usize,
        _h_k: usize,
        conv_dim: usize,
        k: usize,
        device: Device,
        compute: DType,
    ) -> Result<Tensor, ModelError> {
        const CS: usize = 64;
        let qkv = prof(device, "la_inproj", || self.in_proj_qkv.forward(h))?;
        let a = self.in_proj_a.forward(h)?;
        let b = self.in_proj_b.forward(h)?;
        let conv_w = self.conv_w_dev.as_ref().ok_or_else(|| missing("conv_w_dev"))?;
        let dt_bias = self.dt_bias_dev.as_ref().ok_or_else(|| missing("dt_bias_dev"))?;
        let a_log = self.a_log_dev.as_ref().ok_or_else(|| missing("a_log_dev"))?;
        // prep_scatter ожидает a/b в F16 (как decode-путь); cast если compute другой.
        let a_f16 = if a.dtype() == DType::F16 { a } else { a.to_dtype(DType::F16).coerr()? };
        let b_f16 = if b.dtype() == DType::F16 { b } else { b.to_dtype(DType::F16).coerr()? };

        // Device-резидентный стейт: сеем из host один раз (когда зеркало None —
        // первый чанк / свежий KV), дальше переиспользуем между чанками без
        // host↔device round-trip'а (раньше каждый чанк делал from_vec read +
        // host_vec write = clone_dtoh sync на КАЖДЫЙ слой; это был host-stall).
        // host остаётся источником истины для decode-handoff — pipeline обновляет
        // его из dev ОДИН раз после всего prefill (sync_decode_host_state).
        prof(device, "la_state_in", || -> Result<(), ModelError> {
            if state.conv_state_dev.is_none() {
                let cs = Tensor::from_vec(state.conv_state.clone(), vec![k - 1, conv_dim], device)
                    .coerr()?
                    .to_dtype(compute)
                    .coerr()?;
                state.conv_state_dev = Some(cs);
            }
            if state.ssm_state_dev.is_none() {
                let ss = Tensor::from_vec(state.ssm_state.clone(), vec![h_v, dk, dv], device).coerr()?;
                state.ssm_state_dev = Some(ss);
            }
            Ok(())
        })?;

        // Backend op: chunk_conv1d + silu + prep_scatter + chunk_gated_delta_rule.
        // out = [h_v, s, dv] F32. Мутирует cs_t/ss_t (dev-зеркала) in-place.
        let conv_w_c;
        let conv_w = if conv_w.dtype() == compute {
            conv_w
        } else {
            conv_w_c = conv_w.to_dtype(compute).coerr()?;
            &conv_w_c
        };
        let out = {
            let cs_t = state.conv_state_dev.as_mut().ok_or_else(|| missing("conv_state_dev"))?;
            let ss_t = state.ssm_state_dev.as_mut().ok_or_else(|| missing("ssm_state_dev"))?;
            prof(device, "la_kernel", || qkv.linear_attn_chunk_prefill(
                conv_w, &a_f16, &b_f16, dt_bias, a_log,
                cs_t, ss_t,
                self.num_k_heads, h_v, dk, dv, k, CS, self.q_scale, true,
            ).coerr())?
        };

        prof(device, "la_post", || {
        // Layout перевод: [h_v, s, hv] → [1, s, h_v, hv] (старая цепочка ниже
        // ожидает (t·h_v+hi)·dv stride). transpose(0,1)+contiguous+reshape.
        let core_t = out
            .transpose(0, 1)
            .coerr()?
            .contiguous()
            .coerr()?
            .reshape(vec![1, s, h_v, dv])
            .coerr()?
            .to_dtype(compute)
            .coerr()?;
        let z = self.in_proj_z.forward(h)?.reshape(vec![1, s, h_v, dv]).coerr()?;
        let normed = rms_norm(&core_t, &self.norm_weight, self.rms_eps).coerr()?;
        let normed = normed.mul(&z.silu().coerr()?).coerr()?;
        let normed = normed.reshape(vec![1, s, self.value_dim]).coerr()?;
        self.out_proj.forward(&normed)
        })
    }

    /// Device-резидентный decode-шаг (T=1) для CUDA-graph: GEMM-проекции +
    /// fused linear-attn ядро (conv1d-update + prep + gated-delta-rule +
    /// RmsNormGated), всё на device без host round-trip. Требует засеянного
    /// `conv_state_dev`/`ssm_state_dev` (см. [`DecoderModel::sync_decode_dev_state`])
    /// и dev-весов (build при non-CPU). Compute-dtype должен быть F16.
    fn forward_decode_dev(&self, h: &Tensor, cache: &mut LayerCache) -> Result<Tensor, ModelError> {
        let state = match cache {
            LayerCache::Linear(s) => s,
            LayerCache::Full(_) => return Err(ModelError::Shape("linear layer got full cache".into())),
        };
        let conv_w = self.conv_w_dev.as_ref().ok_or_else(|| missing("conv_w_dev"))?;
        let a_log = self.a_log_dev.as_ref().ok_or_else(|| missing("a_log_dev"))?;
        let dt_bias = self.dt_bias_dev.as_ref().ok_or_else(|| missing("dt_bias_dev"))?;
        let norm_w = self.norm_w_f16.as_ref().ok_or_else(|| missing("norm_w_f16"))?;
        let cs = state.conv_state_dev.as_mut().ok_or_else(|| missing("conv_state_dev (sync_to_device?)"))?;
        let ss = state.ssm_state_dev.as_mut().ok_or_else(|| missing("ssm_state_dev (sync_to_device?)"))?;

        let dev = h.device();
        // in_qkv/a/b/z берут один `h` → квантуем 1×; qkv/z через prequant (NVFP4),
        // a/b — Dense [N=48] (forward сам, prequant их не касается).
        let act = quant_act_shared(h);
        let qkv = proj_shared(&self.in_proj_qkv, h, &act, dev, "lin_in_qkv")?;
        let a = prof(dev, "lin_in_a", || self.in_proj_a.forward(h))?;
        let b = prof(dev, "lin_in_b", || self.in_proj_b.forward(h))?;
        let z = proj_shared(&self.in_proj_z, h, &act, dev, "lin_in_z")?;
        let out = prof(dev, "lin_gdr_step", || qkv
            .linear_attn_decode_step(
                conv_w, &a, &b, dt_bias, a_log, &z, norm_w, cs, ss,
                self.num_k_heads, self.num_v_heads, self.dk, self.dv, self.conv_k,
                self.q_scale, self.rms_eps,
            ))
            .coerr()?;
        let out = out.reshape(vec![1, 1, self.value_dim]).coerr()?;
        prof(dev, "lin_oproj", || self.out_proj.forward(&out))
    }

    /// Device-резидентный prefill линейного слоя — **ждёт hybrid-сессии**. Для
    /// hybrid-моделей нужен chunked GatedDeltaNet device-резидентный путь
    /// (conv1d-update по chunk'у + chunk_gated_delta_rule с device-state). Сейчас
    /// `DecoderConfig::graph_prefill_ok` режет hybrid (`linear.is_none()`)
    /// заранее, так что в норме сюда не приходим; метод оставлен для полноты
    /// `Mixer`-енама и чтобы вызывающий `forward_prefill_dev` компилировался.
    fn forward_prefill_dev(&self, h: &Tensor, cache: &mut LayerCache) -> Result<Tensor, ModelError> {
        let device = h.device();
        let compute = h.dtype();
        if !matches!(device, Device::Cuda(_))
            || self.conv_w_dev.is_none()
            || self.a_log_dev.is_none()
            || self.dt_bias_dev.is_none()
            || !matches!(compute, DType::F16 | DType::BF16 | DType::F32)
        {
            return Err(ModelError::Forward(
                "forward_prefill_dev: linear mixer требует CUDA и device-зеркал весов".into(),
            ));
        }
        let s = h.dims()[1];
        if let Some(out) = self.try_small_chunk_dev(h, cache, s, compute)? {
            return Ok(out);
        }
        let state = match cache {
            LayerCache::Linear(s) => s,
            LayerCache::Full(_) => return Err(ModelError::Shape("linear layer got full cache".into())),
        };
        self.forward_cuda_chunk_prefill(
            h, state, s, self.dk, self.dv, self.num_v_heads, self.num_k_heads, self.conv_dim,
            self.conv_k, device, compute,
        )
    }

    fn try_small_chunk_dev(
        &self,
        h: &Tensor,
        cache: &mut LayerCache,
        s: usize,
        compute: DType,
    ) -> Result<Option<Tensor>, ModelError> {
        if s == 0 || s > SMALL_CHUNK_DEV || compute != DType::F16 || self.norm_w_f16.is_none() {
            return Ok(None);
        }
        let mut parts = Vec::with_capacity(s);
        for t in 0..s {
            let ht = h.narrow(1, t, 1).coerr()?.contiguous().coerr()?;
            parts.push(self.forward_decode_dev(&ht, cache)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Ok(Some(Tensor::cat(&refs, 1).coerr()?))
    }
}

const SMALL_CHUNK_DEV: usize = 8;

fn missing(what: &str) -> ModelError {
    ModelError::Forward(format!("forward_decode_dev: {what} не инициализирован"))
}

/// Fused `hidden = x + residual; normed = RMSNorm(hidden) * weight` (один launch
/// вместо add + rms_norm). `weight` уже с пред-baked гейном (+1 для OnePlus при
/// load) → Plain-вариант. Fallback на decomposed при Unsupported (CPU/нет ядра).
fn fused_add_norm(
    dev: Device,
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor), ModelError> {
    prof(dev, "rms_norm_residual", || {
        match x.rms_norm_residual_fused(residual, weight, eps, false) {
            Ok(pair) => Ok(pair),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                let hidden = residual.add(x)?;
                let normed = rms_norm(&hidden, weight, eps)?;
                Ok((hidden, normed))
            }
            Err(e) => Err(e),
        }
    })
    .coerr()
}

/// rms_norm + (опц.) prequant ОДНИМ ядром (эпилог нормы; бит-в-бит с
/// rms_norm_fused→quantize_act, гейт cuda_rms_mod_quant::rms_w). `want` =
/// формат веса потребителей (NVFP4|MXFP8).
#[allow(clippy::type_complexity)]
fn rms_norm_quant(
    x: &Tensor,
    w: &Tensor,
    eps: f32,
    want: Option<DType>,
) -> Result<(Tensor, Option<(Tensor, Tensor, DType)>), ModelError> {
    if matches!(x.device(), Device::Cuda(_))
        && matches!(x.dtype(), DType::F16 | DType::BF16)
    {
        let fused = match want {
            Some(DType::NVFP4) => x.rms_quant_nvfp4(w, eps, false).ok(),
            Some(DType::MXFP8) => x.rms_quant_mxfp8(w, eps, false).ok(),
            _ => None,
        };
        if let Some((y, p, sc)) = fused {
            return Ok((y, Some((p, sc, want.unwrap()))));
        }
    }
    Ok((rms_norm(x, w, eps).coerr()?, None))
}

/// Квантует `h` в NVFP4 ОДИН раз для шаринга между проекциями из него (q/k/v;
/// in_qkv/z; gate/up). None если backend не умеет (CPU) → проекции квантуют
/// сами. Decode (m=1) с MXFP8-весом остаётся на gemv-пути (без prequant) —
/// поэтому формат тут только NVFP4.
fn quant_act_shared(h: &Tensor) -> Option<(Tensor, Tensor, DType)> {
    h.nvfp4_quantize_act().ok().map(|(p, s)| (p, s, DType::NVFP4))
}

/// Проекция из `h` через общую квант-активацию `act` (без повторного quantize),
/// если формат веса совпадает с форматом пары; иначе обычный `forward`. Форма
/// выхода = как у `forward`: ведущие dims `h` + `[N]`. Обёрнута в `prof`.
fn proj_shared(
    ql: &QLinear,
    h: &Tensor,
    act: &Option<(Tensor, Tensor, DType)>,
    dev: Device,
    name: &'static str,
) -> Result<Tensor, ModelError> {
    prof(dev, name, || {
        if let Some((p, s, fmt)) = act {
            if ql.quant_dtype() == Some(*fmt) {
                let lead = &h.dims()[..h.rank() - 1];
                let m: usize = lead.iter().product();
                let out = ql.forward_prequant(p, s, m)?; // [m, N]
                let mut shape = lead.to_vec();
                shape.push(out.dims()[out.rank() - 1]);
                return out.reshape(shape).map_err(|e| ModelError::Forward(e.to_string()));
            }
        }
        ql.forward(h)
    })
}

fn apply_opt_norm(x: &Tensor, w: Option<&Tensor>, eps: f32) -> Result<Tensor, ModelError> {
    match w {
        Some(w) => rms_norm(x, w, eps).coerr(),
        None => Ok(x.clone()),
    }
}

fn apply_opt_head_norm(x: &Tensor, w: Option<&Tensor>, eps: f32) -> Result<Tensor, ModelError> {
    match w {
        Some(w) => rms_norm(x, w, eps).coerr(),
        None => Ok(x.clone()),
    }
}

fn host_vec(t: &Tensor) -> Result<Vec<f32>, ModelError> {
    t.to_dtype(DType::F32)
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .map_err(|e| ModelError::Forward(e.to_string()))
}

thread_local! {
    static LAYER_DUMP: std::cell::RefCell<Vec<(usize, String, Vec<f32>)>> = const { std::cell::RefCell::new(Vec::new()) };
}

static DUMP_LAYERS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static DUMP_GTOK: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Включить/выключить per-слойный дамп скрытых состояний (диагностика
/// chunked-prefill бага, см. [`record_layer_norm`]). Дефолт ВЫКЛ.
pub fn set_dump_layers(on: bool) {
    DUMP_LAYERS.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn dump_layers_on() -> bool {
    DUMP_LAYERS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Глобальный токен, чьё состояние пишется в дамп. `None` (default) — последний
/// токен текущего чанка. См. [`record_layer_norm`].
pub fn set_dump_gtok(gtok: Option<usize>) {
    DUMP_GTOK.store(gtok.unwrap_or(usize::MAX), std::sync::atomic::Ordering::Relaxed);
}

fn dump_gtok() -> Option<usize> {
    match DUMP_GTOK.load(std::sync::atomic::Ordering::Relaxed) {
        usize::MAX => None,
        g => Some(g),
    }
}

/// Записать L2-норму + первый элемент скрытого состояния токена после под-слоя
/// `tag` слоя `idx`. По умолчанию — последний токен (позиция past+s-1); если
/// задан [`set_dump_gtok`] — глобальный токен этой позиции (локальный idx =
/// gtok-past), если он попадает в текущий чанк [past, past+s). Диагностика
/// chunked-prefill.
fn record_layer_norm(idx: usize, tag: &str, hidden: &Tensor, s: usize, past: usize) {
    let local = match dump_gtok() {
        Some(g) => {
            if g < past || g >= past + s { return; }
            g - past
        }
        None => s - 1,
    };
    let last = match hidden.narrow(1, local, 1).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(_) => return,
    };
    LAYER_DUMP.with(|d| d.borrow_mut().push((idx, tag.to_string(), last)));
}

/// Забрать и очистить накопленный per-layer дамп полных векторов. См. [`record_layer_norm`].
pub fn layer_dump_take() -> Vec<(usize, String, Vec<f32>)> {
    LAYER_DUMP.with(|d| std::mem::take(&mut *d.borrow_mut()))
}

fn partial_rope(x: &Tensor, rope: &RopeCache, start: usize, len: usize, rotary_dim: usize, head_dim: usize) -> CoreResult<Tensor> {
    if rotary_dim == head_dim {
        return apply_rope_range(x, rope, start, len, RopeLayout::Split);
    }
    let dev = x.device();
    let x_rot = prof(dev, "rope_split_in", || x.narrow(3, 0, rotary_dim)?.contiguous())?;
    let x_pass = x.narrow(3, rotary_dim, head_dim - rotary_dim)?.contiguous()?;
    let rotated = prof(dev, "rope_kernel", || apply_rope_range(&x_rot, rope, start, len, RopeLayout::Split))?;
    prof(dev, "rope_cat", || Tensor::cat(&[&rotated, &x_pass], 3))
}

fn repeat_kv(x: &Tensor, group_size: usize) -> CoreResult<Tensor> {
    if group_size == 1 {
        return Ok(x.clone());
    }
    let dims = x.dims();
    let (b, n_kv, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    let x_un = x.unsqueeze(2)?;
    let reps = Tensor::zeros(vec![b, n_kv, group_size, s, d], x.dtype(), x.device())?;
    let x_b = x_un.broadcast_add(&reps)?;
    x_b.reshape(vec![b, n_kv * group_size, s, d])
}

fn build_mask(s_new: usize, s_total: usize, past: usize, window: Option<usize>, device: Device, dtype: DType) -> CoreResult<Tensor> {
    let mut data = vec![0.0_f32; s_new * s_total];
    for i in 0..s_new {
        let qi = past + i;
        for j in 0..s_total {
            let causal_ok = j <= qi;
            let window_ok = match window {
                Some(w) => qi < j + w,
                None => true,
            };
            if !(causal_ok && window_ok) {
                data[i * s_total + j] = MASK_NEG;
            }
        }
    }
    let m = Tensor::from_vec(data, vec![s_new, s_total], device)?;
    if dtype != DType::F32 {
        m.to_dtype(dtype)
    } else {
        Ok(m)
    }
}

trait CoreResultExt<T> {
    fn coerr(self) -> Result<T, ModelError>;
}
impl<T> CoreResultExt<T> for CoreResult<T> {
    fn coerr(self) -> Result<T, ModelError> {
        self.map_err(|e: SynaptixError| ModelError::Forward(e.to_string()))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model load: {0}")]
    Load(String),
    #[error("model build: {0}")]
    Build(String),
    #[error("model shape: {0}")]
    Shape(String),
    #[error("model forward: {0}")]
    Forward(String),
}
