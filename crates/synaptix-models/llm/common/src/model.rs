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
use synaptix_ops::pos::rope::{apply_rope_range, apply_rope_with_cossin, RopeLayout};

/// Откуда брать позиции RoPE для прогона блоков.
///
/// Обычный текст — [`Self::Sequential`]: позиция токена равна его индексу в
/// последовательности (`past + i`). Мультимодальный промпт Qwen-VL /
/// Qwen3.5 живёт на M-RoPE: у патчей картинки три оси позиций (время,
/// строка, столбец), а текст после блока продолжается не с `h·w`, а с
/// `max(h, w)`. Такие таблицы cos/sin строятся снаружи на весь промпт
/// ([`Self::Tables`]), а декод после него идёт по 1D-позициям со сдвигом
/// ([`Self::Shifted`]: `past + i + delta`, где `delta = max_pos + 1 − L`).
#[derive(Clone, Copy)]
pub enum RopePositions<'a> {
    Sequential,
    Shifted(i64),
    /// `[L, rotary_dim/2]` F32 на весь промпт; берутся строки `[past, past+s)`.
    Tables { cos: &'a Tensor, sin: &'a Tensor },
}
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
    post_eps: f32,
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

impl FullAttn {
    /// Вес слоя в байтах — для планирования частичного оффлоада.
    fn bytes(&self) -> usize {
        self.q_proj.bytes()
            + self.k_proj.bytes()
            + self.v_proj.bytes()
            + self.o_proj.bytes()
            + tensor_bytes(self.q_norm.as_ref())
            + tensor_bytes(self.k_norm.as_ref())
    }
}

impl LinearAttn {
    fn bytes(&self) -> usize {
        self.in_proj_qkv.bytes()
            + self.in_proj_a.bytes()
            + self.in_proj_b.bytes()
            + self.in_proj_z.bytes()
            + self.out_proj.bytes()
            + tensor_bytes(Some(&self.norm_weight))
            + tensor_bytes(self.conv_w_dev.as_ref())
            + tensor_bytes(self.a_log_dev.as_ref())
            + tensor_bytes(self.dt_bias_dev.as_ref())
            + tensor_bytes(self.norm_w_f16.as_ref())
    }
}

impl Mlp {
    fn bytes(&self) -> usize {
        self.gate_proj.bytes() + self.up_proj.bytes() + self.down_proj.bytes()
    }
}

impl Block {
    /// Сколько памяти занимает блок целиком. По нему планируется частичный
    /// оффлоад: сколько блоков оставить на карте, чтобы под контекст осталось
    /// нужное количество памяти.
    pub fn bytes(&self) -> usize {
        let mixer = match &self.mixer {
            Mixer::Full(fa) => fa.bytes(),
            Mixer::Linear(la) => la.bytes(),
        };
        mixer
            + self.mlp.bytes()
            + tensor_bytes(Some(&self.pre_attn_norm))
            + tensor_bytes(self.post_attn_norm.as_ref())
            + tensor_bytes(Some(&self.pre_mlp_norm))
            + tensor_bytes(self.post_mlp_norm.as_ref())
    }

    /// Блок лежит на этом устройстве (по первому весу — они переезжают вместе).
    fn on_device(&self, dev: Device) -> bool {
        self.pre_attn_norm.device() == dev
    }

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
            post_eps: self.post_eps,
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
    embed_norm: Option<Tensor>,
    final_norm: Tensor,
    lm_head: QLinear,
    blocks: Vec<Block>,
    rope_global: RopeCache,
    rope_local: Option<RopeCache>,
    rope_capacity: usize,
    embed_scale: Option<f32>,
    /// Блоки CPU-резидентны и стримятся на GPU per-block в forward (pinned-H2D
    /// с префетчем) — bf16-энкодер 24GB на 24GB-карте. Декод-петли не поддержаны.
    /// Сколько первых блоков живут на устройстве. Остальные — на хосте и
    /// стримятся по одному во время forward'а с префетчем следующего.
    /// Равно числу блоков — вся модель резидентна (обычный путь).
    resident_blocks: usize,
}

pub struct KvCacheLayer {
    pub k: Tensor,
    pub v: Tensor,
    pub k_scale: Option<Tensor>,
    pub v_scale: Option<Tensor>,
    pub start: usize,
}

pub const RING_SLACK: usize = 2048;

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

    /// Полный снимок linear-состояния: и host-векторы, и device-зеркала.
    ///
    /// [`Self::snapshot_linear`] берёт ТОЛЬКО одну половину (dev, если зеркала
    /// уже созданы, иначе host) — для префикс-KV между ходами этого мало:
    /// префилл живёт в dev-зеркалах, а `sync_decode_dev_state` перед декодом
    /// пересеивает их из host. Расхождение любой из половин ломает
    /// продолжение диалога с сохранённой точки.
    pub fn snapshot_linear_full(&self) -> Result<Vec<LinearSnapshot>, ModelError> {
        let mut out = Vec::new();
        for l in &self.layers {
            let LayerCache::Linear(st) = l else { continue };
            out.push(LinearSnapshot {
                conv_dev: match &st.conv_state_dev {
                    Some(t) => Some(deep_copy(t)?),
                    None => None,
                },
                ssm_dev: match &st.ssm_state_dev {
                    Some(t) => Some(deep_copy(t)?),
                    None => None,
                },
                conv_host: Some(st.conv_state.clone()),
                ssm_host: Some(st.ssm_state.clone()),
            });
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

    /// Самое дальнее начало ring-окна среди full-attention слоёв: ниже этой
    /// позиции префикс уже вытеснен, и продолжать с неё нельзя (для
    /// sliding-слоёв кэш держит только последние W токенов).
    pub fn ring_start_max(&self) -> usize {
        self.layers
            .iter()
            .filter_map(|l| match l {
                LayerCache::Full(f) => Some(f.start),
                LayerCache::Linear(_) => None,
            })
            .max()
            .unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        for l in &mut self.layers {
            match l {
                LayerCache::Linear(s) => s.reset(),
                LayerCache::Full(f) => f.start = 0,
            }
        }
    }

    /// Переселяет содержимое кэша в host-RAM: буферы слоёв заменяются
    /// CPU-копиями, VRAM возвращается пулу. Возвращает, сколько байт VRAM
    /// освободилось (не всё из этого едет через шину: device-зеркала GDN
    /// просто сбрасываются — их источник истины и так в RAM).
    ///
    /// Зачем: пока идёт вложенная генерация (субагент, сводка), посчитанный
    /// контекст диалога держит гигабайты VRAM, а вложенному прогону их не
    /// хватает. Перевезти префикс через PCIe и привезти обратно дешевле, чем
    /// потерять его и префиллить историю заново: 220 МБ идут ~40 мс в обе
    /// стороны против единиц секунд на префилл тех же токенов.
    ///
    /// Состояние кэша (`seq_len`, `start` слоёв) не меняется — переезжают
    /// только данные. Forward по «припаркованному» кэшу упадёт на
    /// несовпадении устройств, это осознанно: молча считать по пустому
    /// буферу хуже, чем громко отказать.
    pub fn park_to_host(&mut self) -> Result<usize, ModelError> {
        let mut moved = 0;
        for l in self.layers.iter_mut() {
            match l {
                LayerCache::Full(f) => {
                    moved += park_tensor(&mut f.k)?;
                    moved += park_tensor(&mut f.v)?;
                    if let Some(t) = f.k_scale.as_mut() {
                        moved += park_tensor(t)?;
                    }
                    if let Some(t) = f.v_scale.as_mut() {
                        moved += park_tensor(t)?;
                    }
                }
                LayerCache::Linear(st) => {
                    // У GDN источник истины — host-векторы, а device-зеркала
                    // пересеиваются из них (`sync_to_device`). Поэтому их не
                    // возим, а сбрасываем — предварительно считав то, что
                    // дописал graph-декод.
                    st.sync_to_host().coerr()?;
                    moved += tensor_bytes(st.conv_state_dev.as_ref());
                    moved += tensor_bytes(st.ssm_state_dev.as_ref());
                    st.conv_state_dev = None;
                    st.ssm_state_dev = None;
                }
            }
        }
        Ok(moved)
    }

    /// Обратный переезд: [`Self::park_to_host`] наоборот.
    pub fn unpark_to(&mut self, device: Device) -> Result<usize, ModelError> {
        let mut moved = 0;
        for l in self.layers.iter_mut() {
            let LayerCache::Full(f) = l else { continue };
            moved += unpark_tensor(&mut f.k, device)?;
            moved += unpark_tensor(&mut f.v, device)?;
            if let Some(t) = f.k_scale.as_mut() {
                moved += unpark_tensor(t, device)?;
            }
            if let Some(t) = f.v_scale.as_mut() {
                moved += unpark_tensor(t, device)?;
            }
        }
        Ok(moved)
    }

    /// Содержимое лежит в host-RAM — до forward'а нужен [`Self::unpark_to`].
    pub fn is_parked(&self) -> bool {
        self.layers.iter().any(|l| match l {
            LayerCache::Full(f) => f.k.device() == Device::Cpu,
            LayerCache::Linear(_) => false,
        })
    }

    /// Сколько VRAM держат буферы слоёв прямо сейчас.
    pub fn device_bytes(&self) -> usize {
        let mut total = 0;
        for l in &self.layers {
            match l {
                LayerCache::Full(f) => {
                    for t in [Some(&f.k), Some(&f.v), f.k_scale.as_ref(), f.v_scale.as_ref()] {
                        if let Some(t) = t {
                            if t.device() != Device::Cpu {
                                total += tensor_bytes(Some(t));
                            }
                        }
                    }
                }
                LayerCache::Linear(st) => {
                    total += tensor_bytes(st.conv_state_dev.as_ref());
                    total += tensor_bytes(st.ssm_state_dev.as_ref());
                }
            }
        }
        total
    }
}

fn tensor_bytes(t: Option<&Tensor>) -> usize {
    t.map(|t| t.dtype().bytes_for_numel(t.numel())).unwrap_or(0)
}

fn park_tensor(t: &mut Tensor) -> Result<usize, ModelError> {
    if t.device() == Device::Cpu {
        return Ok(0);
    }
    let bytes = tensor_bytes(Some(t));
    *t = t.to_device(Device::Cpu).coerr()?;
    Ok(bytes)
}

fn unpark_tensor(t: &mut Tensor, device: Device) -> Result<usize, ModelError> {
    if t.device() == device {
        return Ok(0);
    }
    let bytes = tensor_bytes(Some(t));
    *t = t.to_device(device).coerr()?;
    Ok(bytes)
}

impl LinearSnapshot {
    /// Переселить снимок linear-состояния в host-RAM — см.
    /// [`KvCache::park_to_host`]. Device-половина снимка переезжает как есть:
    /// host-половина (`*_host`) и так лежит в RAM.
    pub fn park_to_host(&mut self) -> Result<usize, ModelError> {
        let mut moved = 0;
        if let Some(t) = self.conv_dev.as_mut() {
            moved += park_tensor(t)?;
        }
        if let Some(t) = self.ssm_dev.as_mut() {
            moved += park_tensor(t)?;
        }
        Ok(moved)
    }

    pub fn unpark_to(&mut self, device: Device) -> Result<usize, ModelError> {
        let mut moved = 0;
        if let Some(t) = self.conv_dev.as_mut() {
            moved += unpark_tensor(t, device)?;
        }
        if let Some(t) = self.ssm_dev.as_mut() {
            moved += unpark_tensor(t, device)?;
        }
        Ok(moved)
    }

    /// Сколько VRAM держит device-половина снимка прямо сейчас.
    pub fn device_bytes(&self) -> usize {
        [self.conv_dev.as_ref(), self.ssm_dev.as_ref()]
            .into_iter()
            .flatten()
            .filter(|t| t.device() != Device::Cpu)
            .map(|t| tensor_bytes(Some(t)))
            .sum()
    }
}

pub struct DecodeState {
    pub input: Tensor,
    pub pos_dev: Tensor,
    pub tcache_dev: Tensor,
    pub ring_pos_dev: Tensor,
    pub ring_len_dev: Tensor,
    pub rope_cos: Tensor,
    pub rope_sin: Tensor,
    pub logits: Tensor,
}

impl DecodeState {
    pub fn update(&mut self, token: u32, pos: u32) -> Result<(), ModelError> {
        self.update_ring(token, pos, 0)
    }

    pub fn update_ring(&mut self, token: u32, pos: u32, ring_start: u32) -> Result<(), ModelError> {
        self.input.write_host_u32(&[token]).coerr()?;
        self.pos_dev.write_host_u32(&[pos]).coerr()?;
        self.tcache_dev.write_host_u32(&[pos + 1]).coerr()?;
        self.ring_pos_dev.write_host_u32(&[pos - ring_start]).coerr()?;
        self.ring_len_dev.write_host_u32(&[pos + 1 - ring_start]).coerr()?;
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
            // Веса, уже упакованные в бандле, берём как есть: ни чтения F16,
            // ни повторного квантования на загрузке.
            if let Some(prequant) = weights.quant(key, if b_dev == device { device } else { b_dev }) {
                let qw = prequant?;
                return Ok(QLinear::Quant(qw));
            }
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

        // Эмбеддинги, уже упакованные в бандле: плотной копии там нет,
        // поэтому и читать её не пробуем, и квантовать заново незачем.
        let prequant_embed = weights
            .quant("model.embed_tokens.weight", device)
            .transpose()?;
        let mut embed_dense = match prequant_embed {
            Some(_) => None,
            None => Some(weights.tensor("model.embed_tokens.weight", device, compute)?),
        };
        let embed_quant = if let Some(q) = prequant_embed {
            Some(q)
        } else if embed_dtype == DType::MXFP8
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
        let embed_norm = if cfg.embed_rms_norm {
            Some(
                Tensor::from_vec(vec![1.0_f32; cfg.hidden_size], vec![cfg.hidden_size], device)
                    .and_then(|t| t.to_dtype(compute))
                    .map_err(|e| ModelError::Load(e.to_string()))?,
            )
        } else {
            None
        };
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
        } else if let Some(prequant) = weights.quant("lm_head.weight", device) {
            // Голова уже упакована в бандле — берём как есть.
            QLinear::Quant(prequant?)
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
                post_eps: cfg.post_norm_eps.unwrap_or(eps),
            });
        }

        let rope_capacity = rope_capacity.max(1);
        let build_rope = |spec: &crate::config::RopeSpec| -> Result<RopeCache, ModelError> {
            if spec.rotary_dim == 0 {
                return RopeCache::new(2, 1, 10_000.0, device)
                    .map_err(|e| ModelError::Build(e.to_string()));
            }
            match &spec.scaled_freqs {
                Some(freqs) => RopeCache::with_scaled_freqs(spec.rotary_dim, rope_capacity, spec.theta, freqs, device),
                None => RopeCache::new(spec.rotary_dim, rope_capacity, spec.theta, device),
            }
            .map_err(|e| ModelError::Build(e.to_string()))
        };
        let n_blocks = blocks.len();
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
            embed_norm,
            final_norm,
            lm_head,
            blocks,
            rope_global,
            rope_local,
            rope_capacity,
            embed_scale: cfg.embed_scale,
            resident_blocks: if block_device.is_some_and(|d| d != device) { 0 } else { n_blocks },
        })
    }

    /// Сколько блоков у модели.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Вес одного блока в байтах (по первому — они одинаковы по форме).
    /// Единица планирования частичного оффлоада.
    pub fn block_bytes(&self) -> usize {
        self.blocks.first().map(Block::bytes).unwrap_or(0)
    }

    /// Сколько блоков сейчас живёт на устройстве.
    pub fn resident_blocks(&self) -> usize {
        self.resident_blocks
    }

    /// Вся модель на устройстве — обычный резидентный путь. При частичном
    /// оффлоаде CUDA-графы недоступны: адреса весов меняются каждый ход.
    pub fn blocks_all_resident(&self) -> bool {
        self.resident_blocks >= self.blocks.len()
    }

    /// Оставить на устройстве первые `resident` блоков, остальные отправить на
    /// хост. Блоки переезжают по одному, поэтому пик памяти — один блок сверх
    /// уже резидентных.
    ///
    /// Возвращает, сколько блоков в итоге резидентно: если перевозка упёрлась
    /// в память, останавливаемся на достигнутом, а не роняем загрузку.
    pub fn set_block_residency(&mut self, resident: usize) -> usize {
        let want = resident.min(self.blocks.len());
        let dev = self.device;
        // Сначала выселяем лишние — так освобождается место под въезд.
        for idx in (want..self.blocks.len()).rev() {
            if self.blocks[idx].on_device(dev) {
                match self.blocks[idx].to_device(Device::Cpu) {
                    Ok(b) => self.blocks[idx] = b,
                    Err(e) => {
                        eprintln!("[llm] блок {idx} не уехал на хост: {e}");
                    }
                }
            }
        }
        for idx in 0..want {
            if self.blocks[idx].on_device(dev) {
                continue;
            }
            match self.blocks[idx].to_device(dev) {
                Ok(b) => self.blocks[idx] = b,
                Err(e) => {
                    eprintln!("[llm] блок {idx} не въехал на карту ({e}) — остаток стримим");
                    self.resident_blocks = idx;
                    return idx;
                }
            }
        }
        self.resident_blocks = want;
        want
    }

    /// Проход по блокам по порядку: резидентные отдаются как есть,
    /// нерезидентные приезжают с хоста, а следующий за ними префетчится на
    /// loader-стриме параллельно текущему шагу (pinned-H2D, как DiT-блоки LTX).
    fn for_each_block<F>(&self, dev: Device, mut body: F) -> Result<(), ModelError>
    where
        F: FnMut(usize, &Block) -> Result<(), ModelError>,
    {
        let n = self.blocks.len();
        if self.blocks_all_resident() || !matches!(dev, Device::Cuda(_)) {
            for (idx, blk) in self.blocks.iter().enumerate() {
                body(idx, blk)?;
            }
            return Ok(());
        }
        let ord = if let Device::Cuda(o) = dev { o } else { 0 };
        synaptix_core::device::cuda::set_offload_pinned(true);
        let ls = synaptix_core::device::cuda::loader_stream(ord)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let first_streamed = self.resident_blocks;
        // Первый нерезидентный блок везём заранее — дальше каждый следующий
        // едет во время счёта предыдущего.
        let mut staged: Option<Block> = if first_streamed < n {
            Some(self.blocks[first_streamed].to_device(dev)?)
        } else {
            None
        };
        let mut result = Ok(());
        for idx in 0..n {
            if idx < first_streamed {
                if let Err(e) = body(idx, &self.blocks[idx]) {
                    result = Err(e);
                    break;
                }
                continue;
            }
            let cur = match staged.take() {
                Some(b) => b,
                None => self.blocks[idx].to_device(dev)?,
            };
            let lsc = ls.clone();
            let next: Result<Option<Block>, ModelError> = std::thread::scope(
                |sp| -> Result<Option<Block>, ModelError> {
                    let h = if idx + 1 < n {
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
                    let step = body(idx, &cur);
                    let prefetched = match h {
                        Some(h) => Some(h.join().map_err(|_| {
                            ModelError::Forward("llm prefetch thread panicked".into())
                        })??),
                        None => None,
                    };
                    step?;
                    Ok(prefetched)
                },
            );
            match next {
                Ok(nb) => staged = nb,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        synaptix_core::device::cuda::set_offload_pinned(false);
        result
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
        // Ровно то, что аллоцирует `make_kv_cache`: K и V по слою и токену.
        // MXFP8 — байт на элемент плюс один E8M0-байт на блок из 32 (а не
        // «2 байта на элемент», как считала прежняя формула: она завышала
        // ринг почти вдвое, и бюджет контекста выходил вполовину меньше
        // реального). Dtype — ПОСЛОЙНЫЙ: sliding-слои остаются плотными
        // (см. `layer_kv_mxfp8`), поэтому суммируем по слоям, а не умножаем
        // одну ставку на их количество.
        let dense = {
            let elem = (self.dtype.size_in_bits() / 8).max(1);
            2 * c.num_key_value_heads * c.head_dim * elem
        };
        let quant = 2 * c.num_key_value_heads * (c.head_dim + c.head_dim.div_ceil(32));
        let ring_ok = self.ring_kv_ok();
        (0..self.blocks.len())
            .filter(|l| matches!(c.layer_kind(*l), LayerKind::Full))
            .map(|l| {
                // Ring-KV: sliding-слой держит окно фиксированного размера
                // (w + RING_SLACK), с длиной контекста не растёт → в ставке
                // «на токен» его нет.
                if ring_ok && c.window_for(l).is_some() {
                    0
                } else if self.layer_kv_mxfp8(l) {
                    quant
                } else {
                    dense
                }
            })
            .sum()
    }

    /// VRAM под те KV-слои, чей размер НЕ зависит от длины контекста:
    /// sliding-слои на ring-KV держат окно `w + RING_SLACK` и с ростом
    /// контекста не растут (в [`Self::kv_bytes_per_token`] их ставка = 0).
    /// Полный размер кэша ≈ `kv_fixed_bytes(batch, max_seq) + ctx *
    /// kv_bytes_per_token()`. Без ring-KV (CPU, hd≠128) — 0: там все слои
    /// считаются на токен.
    pub fn kv_fixed_bytes(&self, batch: usize, max_seq: usize) -> usize {
        if !self.ring_kv_ok() {
            return 0;
        }
        let c = &self.config;
        let elem = (self.dtype.size_in_bits() / 8).max(1);
        (0..self.blocks.len())
            .filter(|l| matches!(c.layer_kind(*l), LayerKind::Full))
            .filter_map(|l| c.window_for(l))
            .map(|w| 2 * batch * c.num_key_value_heads * max_seq.min(w + RING_SLACK) * c.head_dim * elem)
            .sum()
    }

    /// Держится ли KV слоя `l` в MXFP8. Квантованный кэш умеет читать только
    /// `flash_attention_mxfp8kv` (causal, без окна), поэтому MXFP8 достаётся
    /// строго слоям, которые по этому ядру и пойдут: full-attention (без
    /// sliding-окна) и с включённым flash. Остальные слои — плотный
    /// compute-dtype: иначе `kv_append` получил бы BF16-источник на
    /// MXFP8-буфер и падал с «dtype mismatch src/dst».
    fn layer_kv_mxfp8(&self, l: usize) -> bool {
        if self.kv_dtype != DType::MXFP8 || self.config.head_dim % 32 != 0 {
            return false;
        }
        match self.blocks.get(l).map(|b| &b.mixer) {
            Some(Mixer::Full(fa)) => fa.use_flash && fa.sliding_window.is_none(),
            _ => false,
        }
    }

    fn ring_kv_ok(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
            && self.config.head_dim == 128
            && matches!(self.dtype, DType::F16 | DType::BF16)
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

    /// Сырой gather эмбеддингов без `embed_scale`/`embed_rms_norm` — DFlash-драфтер
    /// эмбеддит anchor/mask-токены именно так (норма target'а к ним не применяется).
    pub fn embed_rows(&self, ids_flat: &Tensor) -> Result<Tensor, ModelError> {
        match (&self.embed_q, &self.embed) {
            (Some(q), _) => q.embed_gather(ids_flat).coerr(),
            (None, Some(t)) => t.embed_gather(ids_flat).coerr(),
            (None, None) => Err(ModelError::Build("embed отсутствует".into())),
        }
    }

    pub fn make_kv_cache(&self, batch: usize, max_seq: usize) -> Result<KvCache, ModelError> {
        self.make_kv_cache_ext(batch, max_seq, true)
    }

    /// Как [`Self::make_kv_cache`], но `allow_mxfp8=false` форсит плотный KV на
    /// всех слоях. Нужен вызывающим, которые читают кэш не flash-ядром —
    /// например энкодерный проход с key-padding маской
    /// ([`Self::forward_hidden_states`]): MXFP8-кэш там прочитать нечем.
    pub fn make_kv_cache_ext(
        &self,
        batch: usize,
        max_seq: usize,
        allow_mxfp8: bool,
    ) -> Result<KvCache, ModelError> {
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
        if self.kv_dtype == DType::MXFP8 && hd % 32 != 0 {
            return Err(ModelError::Shape(format!(
                "make_kv_cache: --kv-dtype mxfp8 требует head_dim % 32 == 0 (hd={hd})"
            )));
        }
        let ring_ok = self.ring_kv_ok();
        let mut layers = Vec::with_capacity(self.blocks.len());
        for l in 0..self.blocks.len() {
            let lc = match c.layer_kind(l) {
                LayerKind::Full => {
                    let cap = match c.window_for(l) {
                        Some(w) if ring_ok => max_seq.min(w + RING_SLACK),
                        _ => max_seq,
                    };
                    // Послойно: MXFP8 достаётся только слоям, чей путь чтения —
                    // flash_attention_mxfp8kv. Sliding-слои и любой слой без
                    // flash остаются в compute-dtype.
                    let mxfp8 = allow_mxfp8 && self.layer_kv_mxfp8(l);
                    let kv_dt = if mxfp8 { DType::MXFP8 } else { self.dtype };
                    let k = Tensor::zeros(vec![batch, n_kv, cap, hd], kv_dt, self.device).coerr()?;
                    let v = Tensor::zeros(vec![batch, n_kv, cap, hd], kv_dt, self.device).coerr()?;
                    let (k_scale, v_scale) = if mxfp8 {
                        let nb = hd / 32;
                        (
                            Some(Tensor::zeros(vec![batch, n_kv, cap, nb], DType::U8, self.device).coerr()?),
                            Some(Tensor::zeros(vec![batch, n_kv, cap, nb], DType::U8, self.device).coerr()?),
                        )
                    } else {
                        (None, None)
                    };
                    LayerCache::Full(KvCacheLayer { k, v, k_scale, v_scale, start: 0 })
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
        if let Some(w) = &self.embed_norm {
            hidden = rms_norm(&hidden, w, self.config.rms_norm_eps).coerr()?;
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
        let mut logits = prof(self.device, "lm_head", || self.lm_head.forward(x))?;
        if let Some(scale) = self.config.logit_scale {
            logits = logits.mul_scalar(scale).coerr()?;
        }
        if let Some(cap) = self.config.logit_softcap {
            logits = logits
                .mul_scalar(1.0 / cap)
                .and_then(|t| t.tanh())
                .and_then(|t| t.mul_scalar(cap))
                .coerr()?;
        }
        Ok(logits)
    }

    pub fn head_at(&self, hidden: &Tensor, idx: usize) -> Result<Tensor, ModelError> {
        let row = self.normed_at(hidden, idx)?;
        self.lm_head_forward(&row)
    }

    pub fn heads_all(&self, hidden: &Tensor) -> Result<Tensor, ModelError> {
        let normed = prof(self.device, "norm", || {
            rms_norm(hidden, &self.final_norm, self.config.rms_norm_eps).coerr()
        })?;
        let dims = normed.dims().to_vec();
        let flat = normed
            .reshape(vec![dims[0] * dims[1], dims[2]])
            .coerr()?;
        self.lm_head_forward(&flat)
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
        self.run_blocks(hidden, kv_cache, batch, s, RopePositions::Sequential)
    }

    pub fn forward_from_hidden(&self, hidden: &Tensor, kv_cache: &mut KvCache) -> Result<Tensor, ModelError> {
        self.forward_from_hidden_pos(hidden, kv_cache, RopePositions::Sequential)
    }

    /// Как [`Self::forward_from_hidden`], но с явными позициями RoPE
    /// (M-RoPE-таблицы мультимодального промпта).
    pub fn forward_from_hidden_pos(
        &self,
        hidden: &Tensor,
        kv_cache: &mut KvCache,
        rope_pos: RopePositions,
    ) -> Result<Tensor, ModelError> {
        if hidden.rank() != 3 {
            return Err(ModelError::Shape(format!("hidden must be [B, S, H], got {:?}", hidden.dims())));
        }
        let batch = hidden.dims()[0];
        let s = hidden.dims()[1];
        self.run_blocks(hidden.clone(), kv_cache, batch, s, rope_pos)
    }

    /// Как [`Self::forward`], но с явными позициями RoPE — декод после
    /// M-RoPE-промпта идёт по 1D-позициям со сдвигом
    /// ([`RopePositions::Shifted`]).
    pub fn forward_pos(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut KvCache,
        rope_pos: RopePositions,
    ) -> Result<Tensor, ModelError> {
        if input_ids.rank() != 2 {
            return Err(ModelError::Shape(format!("input_ids must be [B, S], got {:?}", input_ids.dims())));
        }
        let batch = input_ids.dims()[0];
        let s = input_ids.dims()[1];
        let hidden = self.embed_ids(input_ids)?;
        let hidden = self.run_blocks(hidden, kv_cache, batch, s, rope_pos)?;
        self.head_at(&hidden, hidden.dims()[1] - 1)
    }

    /// Как [`Self::forward_trunk`], но дополнительно возвращает выходы блоков
    /// `taps` (в порядке `taps`) — hidden-фичи для внешних потребителей
    /// (DFlash-драфтер берёт слои target'а). Индексация как у HF
    /// `hidden_states[i+1]` = выход блока `i`.
    pub fn forward_trunk_tapped(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut KvCache,
        taps: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>), ModelError> {
        if input_ids.rank() != 2 {
            return Err(ModelError::Shape(format!("input_ids must be [B, S], got {:?}", input_ids.dims())));
        }
        let batch = input_ids.dims()[0];
        let s = input_ids.dims()[1];
        let hidden = self.embed_ids(input_ids)?;
        let mut tapped = Vec::with_capacity(taps.len());
        let out = self.run_blocks_tapped(hidden, kv_cache, batch, s, taps, &mut tapped, RopePositions::Sequential)?;
        Ok((out, tapped))
    }

    fn run_blocks(
        &self,
        hidden: Tensor,
        kv_cache: &mut KvCache,
        batch: usize,
        s: usize,
        rope_pos: RopePositions,
    ) -> Result<Tensor, ModelError> {
        let mut sink = Vec::new();
        self.run_blocks_tapped(hidden, kv_cache, batch, s, &[], &mut sink, rope_pos)
    }

    fn run_blocks_tapped(
        &self,
        mut hidden: Tensor,
        kv_cache: &mut KvCache,
        batch: usize,
        s: usize,
        taps: &[usize],
        tapped: &mut Vec<Tensor>,
        rope_pos: RopePositions,
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
                Mixer::Full(fa) => prof(dev, "attn_full", || fa.forward(&h, &mut kv_cache.layers[idx], past, s, batch, self.rope_at(idx), self.kv_dtype, self.device, self.dtype, None, rope_pos))?,
                Mixer::Linear(la) => prof(dev, "attn_linear", || la.forward(&h, &mut kv_cache.layers[idx], s, self.device, self.dtype))?,
            };
            let mixed = apply_opt_norm(&mixed, blk.post_attn_norm.as_ref(), blk.post_eps)?;
            let hidden = prof(dev, "residual", || residual.add(&mixed).coerr())?;
            if dump_layers { record_layer_norm(idx, if is_lin { "lin_attn" } else { "full_attn" }, &hidden, s, past); }

            let residual2 = hidden.clone();
            let h = prof(dev, "norm", || rms_norm(&hidden, &blk.pre_mlp_norm, blk.rms_eps).coerr())?;
            let mlp_out = prof(dev, "mlp", || blk.mlp.forward(&h))?;
            let mlp_out = apply_opt_norm(&mlp_out, blk.post_mlp_norm.as_ref(), blk.post_eps)?;
            let out = prof(dev, "residual", || residual2.add(&mlp_out).coerr())?;
            if dump_layers { record_layer_norm(idx, "mlp", &out, s, past); }
            Ok(out)
        };

        // Резидентные блоки идут как есть, нерезидентные приезжают с хоста —
        // решает `for_each_block`, здесь разница только в послойной синхре
        // (на стриме её делает сам префетч).
        let sync_ord = match (self.device, self.blocks_all_resident()) {
            (Device::Cuda(o), true) => Some(o),
            _ => None,
        };
        self.for_each_block(dev, |idx, blk| {
            hidden = step(idx, blk, &hidden, kv_cache)?;
            if taps.contains(&idx) {
                tapped.push(hidden.clone());
            }
            if let Some(o) = sync_ord {
                synaptix_core::device::cuda::layer_sync(o, s > 1);
            }
            Ok(())
        })?;
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
        // Энкодерный проход читает KV через sdpa/flash-window, не через
        // mxfp8kv-ядро → кэш всегда плотный, даже если политика модели MXFP8.
        let mut kv = self.make_kv_cache_ext(batch, s, false)?;

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

        let mut hidden = self.embed_ids(input_ids)?;
        let mut states: Vec<Tensor> = Vec::with_capacity(self.blocks.len() + 1);
        let mut step = |idx: usize, blk: &Block, hidden: &Tensor, kv: &mut KvCache|
            -> Result<Tensor, ModelError> {
            states.push(hidden.clone()); // HF hidden_states[idx] = вход блока idx
            let residual = hidden.clone();
            let h = rms_norm(hidden, &blk.pre_attn_norm, blk.rms_eps).coerr()?;
            let mixed = match &blk.mixer {
                Mixer::Full(fa) => fa.forward(
                    &h, &mut kv.layers[idx], 0, s, batch, self.rope_at(idx),
                    self.kv_dtype, dev, self.dtype, pad_ref, RopePositions::Sequential,
                )?,
                Mixer::Linear(_) => {
                    return Err(ModelError::Forward(
                        "forward_hidden_states: linear-слои не поддержаны (только full-attention)".into(),
                    ))
                }
            };
            let mixed = apply_opt_norm(&mixed, blk.post_attn_norm.as_ref(), blk.post_eps)?;
            let hidden = residual.add(&mixed).coerr()?;

            let residual2 = hidden.clone();
            let h = rms_norm(&hidden, &blk.pre_mlp_norm, blk.rms_eps).coerr()?;
            let mlp_out = blk.mlp.forward(&h)?;
            let mlp_out = apply_opt_norm(&mlp_out, blk.post_mlp_norm.as_ref(), blk.post_eps)?;
            residual2.add(&mlp_out).coerr()
        };
        self.for_each_block(dev, |idx, blk| {
            hidden = step(idx, blk, &hidden, &mut kv)?;
            Ok(())
        })?;
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
            return Err(ModelError::Forward("make_decode_state: профиль не поддержан CUDA-graph (два реальных RoPE)".into()));
        }
        if self.config.sliding_window.is_some() && !self.ring_kv_ok() {
            return Err(ModelError::Forward(
                "make_decode_state: sliding-профиль требует ring-KV (CUDA, f16/bf16, hd=128)".into(),
            ));
        }
        let dev = self.device;
        let input = Tensor::from_vec(vec![0u32; batch], vec![batch, 1], dev).coerr()?;
        let pos_dev = Tensor::from_vec(vec![0u32; batch], vec![batch], dev).coerr()?;
        let tcache_dev = Tensor::from_vec(vec![0u32; batch], vec![batch], dev).coerr()?;
        let ring_pos_dev = Tensor::from_vec(vec![0u32; batch], vec![batch], dev).coerr()?;
        let ring_len_dev = Tensor::from_vec(vec![0u32; batch], vec![batch], dev).coerr()?;
        let rope = if self.config.rope_global.rotary_dim > 0 {
            &self.rope_global
        } else {
            self.rope_local.as_ref().unwrap_or(&self.rope_global)
        };
        let cos = rope.cos();
        let sin = rope.sin();
        let rope_cos = Tensor::cat(&[cos, cos], 1).coerr()?.to_dtype(self.dtype).coerr()?;
        let rope_sin = Tensor::cat(&[sin, sin], 1).coerr()?.to_dtype(self.dtype).coerr()?;
        let logits = Tensor::zeros(vec![batch, self.config.vocab_size], self.dtype, dev).coerr()?;
        Ok(DecodeState { input, pos_dev, tcache_dev, ring_pos_dev, ring_len_dev, rope_cos, rope_sin, logits })
    }

    pub fn ring_prepare_decode(&self, kv: &mut KvCache, pos: usize) -> Result<usize, ModelError> {
        let Some(w) = self.config.sliding_window else { return Ok(0) };
        if !self.ring_kv_ok() {
            return Ok(0);
        }
        let mut start = 0usize;
        for l in 0..self.blocks.len() {
            if self.config.window_for(l).is_none() {
                continue;
            }
            let LayerCache::Full(kvl) = &mut kv.layers[l] else { continue };
            let cap = kvl.k.dims()[2];
            if pos - kvl.start + 1 > cap {
                let lo_global = (pos + 1).saturating_sub(w);
                let keep = pos - lo_global;
                if keep > 0 {
                    let src = lo_global - kvl.start;
                    let tk = kvl.k.narrow(2, src, keep).coerr()?.contiguous().coerr()?;
                    let tv = kvl.v.narrow(2, src, keep).coerr()?.contiguous().coerr()?;
                    kvl.k.kv_append_inplace(&tk, 0).coerr()?;
                    kvl.v.kv_append_inplace(&tv, 0).coerr()?;
                }
                kvl.start = lo_global;
            }
            start = kvl.start;
        }
        Ok(start)
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
            return Err(ModelError::Forward("forward_decode_dev: профиль не поддержан (два реальных RoPE)".into()));
        }
        if self.config.sliding_window.is_some() && !self.ring_kv_ok() {
            return Err(ModelError::Forward("forward_decode_dev: sliding без ring-KV не поддержан".into()));
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
        if let Some(w) = &self.embed_norm {
            hidden = rms_norm(&hidden, w, self.config.rms_norm_eps).coerr()?;
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
            let mixed = apply_opt_norm(&mixed, blk.post_attn_norm.as_ref(), blk.post_eps)?;
            // hidden = hidden + mixed; h_mlp = norm(hidden, pre_mlp).
            let (new_hidden, h_mlp) =
                fused_add_norm(dev, &mixed, &hidden, &blk.pre_mlp_norm, blk.rms_eps)?;
            hidden = new_hidden;
            let mlp_out = blk.mlp.forward(&h_mlp)?;
            let mlp_out = apply_opt_norm(&mlp_out, blk.post_mlp_norm.as_ref(), blk.post_eps)?;
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
        let logits = self.lm_head_forward(&last)?;
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
        rope_pos: RopePositions,
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

        let q = prof(device, "attn_rope", || partial_rope(&q, rope, past, s, self.rotary_dim, hd, rope_pos).coerr())?;
        let k = prof(device, "attn_rope", || partial_rope(&k, rope, past, s, self.rotary_dim, hd, rope_pos).coerr())?;

        let new_len = past + s;
        let group = nh / nkv;
        // pad_bias (key-padding для энкодера) несовместим с flash (flash маскирует
        // только чисто-causal) → форсим sdpa-путь, где маску можно дополнить.
        let flash_eligible =
            self.use_flash && self.sliding_window.is_none() && pad_bias.is_none();
        let _core_dev = device;
        // Квант KV — по фактическому dtype буфера слоя, а не по политике модели:
        // `make_kv_cache` держит MXFP8 только там, где кэш читает
        // flash_attention_mxfp8kv (full-attention + flash). Sliding-слои того же
        // прогона плотные, и ветка ниже для них — обычный `kv_append`.
        let kv_mxfp8 = kv.k.dtype() == DType::MXFP8;
        let attn = prof(_core_dev, "attn_core", || -> Result<Tensor, ModelError> {
        Ok(if kv_mxfp8 {
            if !flash_eligible {
                return Err(ModelError::Forward(format!(
                    "MXFP8 KV читается только flash-ядром (causal, без окна и key-padding): \
                     use_flash={} sliding={:?} pad_bias={}",
                    self.use_flash,
                    self.sliding_window,
                    pad_bias.is_some()
                )));
            }
            let KvCacheLayer { k: kc, v: vc, k_scale: ksc, v_scale: vsc, .. } = kv;
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
            let cap = kv.k.dims()[2];
            let mut local_past = past - kv.start;
            if local_past + s > cap {
                let Some(w) = self.sliding_window else {
                    return Err(ModelError::Shape(format!(
                        "KV overflow: past {past} + s {s} > cap {cap}"
                    )));
                };
                let lo_global = (past + 1).saturating_sub(w);
                if lo_global <= kv.start || past + s - lo_global > cap {
                    return Err(ModelError::Shape(format!(
                        "ring KV: chunk s={s} не помещается (cap {cap}, окно {w}) — уменьшите prefill-chunk"
                    )));
                }
                let keep = past - lo_global;
                if keep > 0 {
                    let src = lo_global - kv.start;
                    let tk = kv.k.narrow(2, src, keep).coerr()?.contiguous().coerr()?;
                    let tv = kv.v.narrow(2, src, keep).coerr()?.contiguous().coerr()?;
                    kv.k.kv_append_inplace(&tk, 0).coerr()?;
                    kv.v.kv_append_inplace(&tv, 0).coerr()?;
                }
                kv.start = lo_global;
                local_past = past - kv.start;
            }
            kv.k.kv_append_inplace(&k, local_past).coerr()?;
            kv.v.kv_append_inplace(&v, local_past).coerr()?;
            let local_len = local_past + s;
            let (att_lo, att_len) = match self.sliding_window {
                Some(w) => {
                    let lo_global = (past + 1).saturating_sub(w).max(kv.start);
                    (lo_global - kv.start, past + s - lo_global)
                }
                None => (0, local_len),
            };
            // T>1 на CUDA: FA-4 prefill ЧИТАЕТ preallocated KV как есть
            // (strided, активная длина — device-скаляр). Без этого путь ниже
            // отдаёт `flash_attention` narrow-view'ы, та отвечает
            // NonContiguous, и всё скатывается в SDPA с `repeat_kv`: на
            // каждый слой каждого чанка материализуются k_rep/v_rep размером
            // nh×Tkv×hd (на 6.6k контекста ≈160 МБ), пул с
            // RELEASE_THRESHOLD=MAX растёт с позицией и на 8k промпте
            // упирается в OOM при неизменном used.
            let dev_prefill = if s > 1
                && flash_eligible
                && self.sliding_window.is_none()
                && matches!(device, Device::Cuda(_))
                && matches!(kv.k.dtype(), DType::F16 | DType::BF16)
                && matches!(hd, 64 | 128 | 256)
            {
                let tc = Tensor::from_vec(vec![local_len as u32], vec![1usize], device).coerr()?;
                match q.flash_attention_prefill_dev(&kv.k, &kv.v, &tc, self.attn_scale, true) {
                    Ok(a) => Some(a),
                    Err(e) => {
                        if std::env::var("SYN_TRACE_PREFILL_MEM").is_ok() {
                            eprintln!("[FA4_PREFILL_SKIP] {e:?}");
                        }
                        match e {
                            SynaptixError::Unsupported(_) | SynaptixError::NonContiguous => None,
                            other => return Err(ModelError::Forward(other.to_string())),
                        }
                    }
                }
            } else {
                None
            };
            if let Some(a) = dev_prefill {
                return Ok(a);
            }
            let k_total = kv.k.narrow(2, att_lo, att_len).coerr()?;
            let v_total = kv.v.narrow(2, att_lo, att_len).coerr()?;
            let flash_win = self.sliding_window.is_some()
                && self.use_flash
                && pad_bias.is_none()
                && hd == 128;
            let flashed = if flash_eligible {
                match q.flash_attention(&k_total, &v_total, self.attn_scale, true) {
                    Ok(a) => Some(a),
                    Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => None,
                    Err(e) => return Err(ModelError::Forward(e.to_string())),
                }
            } else if flash_win {
                let w = self.sliding_window.unwrap();
                match q.flash_attention_window(&k_total, &v_total, self.attn_scale, (w - 1) as i32, true) {
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
                    if std::env::var("SYN_TRACE_PREFILL_MEM").is_ok() {
                        eprintln!("[ATTN_FALLBACK] s={s} att_len={att_len} hd={hd} nh={nh} nkv={nkv} flash_eligible={flash_eligible} kv_dtype={kv_dtype:?} kv_buf={:?} sw={:?}", kv.k.dtype(), self.sliding_window);
                    }
                    let k_rep = repeat_kv(&k_total, group).coerr()?;
                    let v_rep = repeat_kv(&v_total, group).coerr()?;
                    let window = self.sliding_window;
                    if s == 1 && window.is_none() && pad_bias.is_none() {
                        scaled_dot_attention(&q, &k_rep, &v_rep, self.attn_scale, None).coerr()?
                    } else {
                        let past_rel = past - kv.start - att_lo;
                        let mask = build_mask(s, att_len, past_rel, window, device, compute).coerr()?;
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
        let (q, k) = if self.rotary_dim > 0 {
            let q = prof(dev, "attn_rope", || q.rope_apply_dev(&state.rope_cos, &state.rope_sin, &state.pos_dev, self.rotary_dim)).coerr()?;
            let k = prof(dev, "attn_rope", || k.rope_apply_dev(&state.rope_cos, &state.rope_sin, &state.pos_dev, self.rotary_dim)).coerr()?;
            (q, k)
        } else {
            (q, k)
        };

        if let Some(w) = self.sliding_window {
            prof(dev, "attn_kv_append", || -> Result<(), ModelError> {
                kvl.k.kv_append_dev(&k, &state.ring_pos_dev).coerr()?;
                kvl.v.kv_append_dev(&v, &state.ring_pos_dev).coerr()
            })?;
            let attn = prof(dev, "attn_flash", || {
                q.flash_attention_window_dev(
                    &kvl.k,
                    &kvl.v,
                    &state.ring_len_dev,
                    self.attn_scale,
                    (w - 1) as i32,
                    true,
                )
            })
            .map_err(|e| ModelError::Forward(e.to_string()))?;
            let attn = attn.permute(vec![0, 2, 1, 3]).coerr()?.contiguous().coerr()?;
            let attn = match gate {
                Some(g) => prof(dev, "attn_gate", || attn.mul(&g.sigmoid()?)).coerr()?,
                None => attn,
            };
            let attn = attn.reshape(vec![b, 1, nh * hd]).coerr()?;
            return prof(dev, "attn_oproj", || self.o_proj.forward(&attn));
        }

        let attn = if kvl.k.dtype() == DType::MXFP8 {
            let KvCacheLayer { k: kc, v: vc, k_scale: ksc, v_scale: vsc, .. } = kvl;
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
                return self.forward_small_batch_dev(h, cache, s);
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
        // Свежий KV (короткий s<=SMALL_CHUNK_DEV префилл идёт этим же путём) —
        // зеркала ещё не созданы: засеять из host-векторов (нулевое состояние).
        // Существующие зеркала не трогаем: во время decode host отстаёт от dev.
        if state.conv_state_dev.is_none() || state.ssm_state_dev.is_none() {
            state
                .sync_to_device(h.device(), self.conv_dim, self.conv_k, self.num_v_heads, self.dk, self.dv)
                .coerr()?;
        }
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
        Ok(Some(self.forward_small_batch_dev(h, cache, s)?))
    }

    /// s <= SMALL_CHUNK_DEV токенов одним проходом: проекции (in_qkv/a/b/z) и
    /// out_proj — батчем M=s (веса читаются ОДИН раз, а не s раз, как в старом
    /// цикле полных decode-шагов), рекуррентное ядро (conv1d-update +
    /// gated-delta-rule + RmsNormGated) — последовательно по токенам: оно
    /// state-зависимо, но весов не читает. Для MTP verify (s=2) это убирает
    /// ~2× чтение весов linear-слоёв (~3 ГБ/шаг на 27B).
    /// SYN_LA_SMALLBATCH=0 возвращает старый цикл (A/B: нумерика M=2-GEMM
    /// слегка отличается от GEMV и меняет greedy-траекторию).
    fn forward_small_batch_dev(
        &self,
        h: &Tensor,
        cache: &mut LayerCache,
        s: usize,
    ) -> Result<Tensor, ModelError> {
        if s > 1 && !small_batch_on() {
            let mut parts = Vec::with_capacity(s);
            for t in 0..s {
                let ht = h.narrow(1, t, 1).coerr()?.contiguous().coerr()?;
                parts.push(self.forward_decode_dev(&ht, cache)?);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            return Tensor::cat(&refs, 1).coerr();
        }
        let state = match cache {
            LayerCache::Linear(st) => st,
            LayerCache::Full(_) => return Err(ModelError::Shape("linear layer got full cache".into())),
        };
        let conv_w = self.conv_w_dev.as_ref().ok_or_else(|| missing("conv_w_dev"))?;
        let a_log = self.a_log_dev.as_ref().ok_or_else(|| missing("a_log_dev"))?;
        let dt_bias = self.dt_bias_dev.as_ref().ok_or_else(|| missing("dt_bias_dev"))?;
        let norm_w = self.norm_w_f16.as_ref().ok_or_else(|| missing("norm_w_f16"))?;
        if state.conv_state_dev.is_none() || state.ssm_state_dev.is_none() {
            state
                .sync_to_device(h.device(), self.conv_dim, self.conv_k, self.num_v_heads, self.dk, self.dv)
                .coerr()?;
        }
        let cs = state.conv_state_dev.as_mut().ok_or_else(|| missing("conv_state_dev (sync_to_device?)"))?;
        let ss = state.ssm_state_dev.as_mut().ok_or_else(|| missing("ssm_state_dev (sync_to_device?)"))?;

        let dev = h.device();
        let act = quant_act_shared(h);
        let qkv = proj_shared(&self.in_proj_qkv, h, &act, dev, "lin_in_qkv")?;
        let a = prof(dev, "lin_in_a", || self.in_proj_a.forward(h))?;
        let b = prof(dev, "lin_in_b", || self.in_proj_b.forward(h))?;
        let z = proj_shared(&self.in_proj_z, h, &act, dev, "lin_in_z")?;

        let mut parts = Vec::with_capacity(s);
        for t in 0..s {
            let sl = |x: &Tensor| -> Result<Tensor, ModelError> {
                x.narrow(1, t, 1).coerr()?.contiguous().coerr()
            };
            let (qkv_t, a_t, b_t, z_t) = (sl(&qkv)?, sl(&a)?, sl(&b)?, sl(&z)?);
            let out_t = prof(dev, "lin_gdr_step", || qkv_t
                .linear_attn_decode_step(
                    conv_w, &a_t, &b_t, dt_bias, a_log, &z_t, norm_w, cs, ss,
                    self.num_k_heads, self.num_v_heads, self.dk, self.dv, self.conv_k,
                    self.q_scale, self.rms_eps,
                ))
                .coerr()?;
            parts.push(out_t.reshape(vec![1usize, 1, self.value_dim]).coerr()?);
        }
        let normed = if s == 1 {
            parts.pop().expect("s >= 1")
        } else {
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, 1).coerr()?
        };
        prof(dev, "lin_oproj", || self.out_proj.forward(&normed))
    }
}

const SMALL_CHUNK_DEV: usize = 8;

/// Батчевые проекции в small-s пути linear-attn (см.
/// `forward_small_batch_dev`). Дефолт ВКЛ; SYN_LA_SMALLBATCH=0 — старый
/// потокенный цикл (для A/B-сравнений нумерики/скорости).
fn small_batch_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SYN_LA_SMALLBATCH").as_deref() != Ok("0"))
}

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

pub fn partial_rope(
    x: &Tensor,
    rope: &RopeCache,
    start: usize,
    len: usize,
    rotary_dim: usize,
    head_dim: usize,
    pos: RopePositions,
) -> CoreResult<Tensor> {
    if rotary_dim == 0 {
        return Ok(x.clone());
    }
    let rotate = |x_rot: &Tensor| -> CoreResult<Tensor> {
        match pos {
            RopePositions::Sequential => apply_rope_range(x_rot, rope, start, len, RopeLayout::Split),
            RopePositions::Shifted(delta) => {
                let shifted = start as i64 + delta;
                if shifted < 0 {
                    return Err(synaptix_core::error::SynaptixError::Other(format!(
                        "rope: позиция {start} со сдвигом {delta} отрицательна"
                    )));
                }
                apply_rope_range(x_rot, rope, shifted as usize, len, RopeLayout::Split)
            }
            RopePositions::Tables { cos, sin } => {
                let cos = cos.narrow(0, start, len)?.contiguous()?;
                let sin = sin.narrow(0, start, len)?.contiguous()?;
                apply_rope_with_cossin(x_rot, &cos, &sin, RopeLayout::Split)
            }
        }
    };
    if rotary_dim == head_dim {
        return rotate(x);
    }
    let dev = x.device();
    let x_rot = prof(dev, "rope_split_in", || x.narrow(3, 0, rotary_dim)?.contiguous())?;
    let x_pass = x.narrow(3, rotary_dim, head_dim - rotary_dim)?.contiguous()?;
    let rotated = prof(dev, "rope_kernel", || rotate(&x_rot))?;
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
