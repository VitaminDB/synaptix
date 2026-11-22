//! Прямой проход Inkling-2 (схема — llama.cpp `inkling2.cpp` + `inkling2-loop.cpp`).
//!
//! Поток между слоями и блоки attention-residuals — F32; проекции — BF16-веса с
//! BF16-активацией (как в эталоне); KV — BF16; эксперты — квант-веса движка с
//! плотной BF16-активацией (только веса квантованы, как у modelopt-эталона).

use synaptix_core::backend::Ink2Launch;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::{ExpertTable, QuantWeight};
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::moe_zc::{ZcLayer, ZcPool};
use synaptix_llm_common::ModelError;
use synaptix_nn::linear::Linear;
use synaptix_nn::Module;

use crate::config::Inkling2Config;
use crate::kern::{self, addr, bf16c, empty, f32c, fe, launch, R};
use crate::loader::{ExpertStack, Inkling2Weights, LoadError, LLM};

/// Проекция: BF16-вес `[out, in]` либо квант-вес бандла с общим множителем.
/// Наибольший чанк одного прогона: кольцо KV SWA-слоя — окно + столько.
pub const MAX_CHUNK: usize = 512;

pub enum Lin {
    Dense(Linear),
    Quant(synaptix_llm_common::QLinear, f32),
}

impl Lin {
    fn load(w: &Inkling2Weights, name: &str, dev: Device) -> Result<Self, LoadError> {
        if let Some((q, alpha)) = w.quant(name, dev)? {
            return Ok(Self::Quant(synaptix_llm_common::QLinear::Quant(q), alpha));
        }
        let t = w.tensor(name, dev, DType::BF16)?;
        Ok(Self::Dense(Linear::new(t, None).map_err(|e| LoadError::Weights(format!("{name}: {e}")))?))
    }

    /// Плотные BF16-матрицы с общим входом, склеенные по выходам — один GEMV вместо нескольких.
    fn cat(w: &Inkling2Weights, names: &[String], dev: Device) -> Result<Self, LoadError> {
        let ts: Vec<Tensor> = names.iter().map(|n| w.tensor(n, dev, DType::BF16)).collect::<Result<_, _>>()?;
        let refs: Vec<&Tensor> = ts.iter().collect();
        let t = Tensor::cat(&refs, 0).map_err(lerr)?.contiguous().map_err(lerr)?;
        Ok(Self::Dense(Linear::new(t, None).map_err(lerr)?))
    }

    /// `x [T, in]` (любой dtype) → `[T, out]` в dtype движка (BF16).
    fn raw(&self, x: &Tensor) -> R<Tensor> {
        let x = bf16c(x)?;
        match self {
            Self::Dense(l) => l.forward(&x).map_err(fe),
            Self::Quant(q, a) => {
                let y = q.forward(&x)?;
                if *a == 1.0 { Ok(y) } else { y.mul_scalar(*a).map_err(fe) }
            }
        }
    }
}

/// Контроллеры входов повторного прохода (q, k, v, mlp): `(1 − tanh(F·d))·x + tanh(G·d)·m`,
/// `d = Down·[rms(x); rms(m)]`. Веса F32 четырёх контроллеров подряд (запуск сразу на несколько —
/// по `blockIdx.z`): down `[4, R, 2H]`, gain/forget транспонированы в `[4, R, H]`.
struct LoopLayer {
    /// Запросы сообщений `[4, H]`: q, k, v, mlp.
    msg_q: Tensor,
    down: Tensor,
    gain: Tensor,
    forget: Tensor,
    rank: usize,
}

/// Где лежат веса экспертов слоя.
pub(crate) enum ExpertStore {
    /// Ещё не подняты ([`Inkling2Model::load_base`]).
    Pending,
    /// Все эксперты на карте: таблицы адресов (плотная активация) и сами веса.
    Resident {
        gate_up: ExpertTable,
        down: ExpertTable,
        /// Держит веса живыми (таблицы — только адреса).
        _weights: (Vec<QuantWeight>, Vec<QuantWeight>),
    },
    /// Zero-copy: эксперты в закреплённом зеркале RAM, на карте — кэш слотов
    /// (`synaptix_llm_common::moe_zc`). Один слой на физический слой: три
    /// прохода цикла делят его слоты.
    Zc(ZcLayer),
}

/// Множители экспертов слоя (сами веса — в [`ExpertRt`]: режим меняется на ходу).
struct Experts {
    alpha_gate_up: Vec<f32>,
    /// Множители на карте `[E]`: gate_up — для активации, down — в вес маршрута.
    alpha_gate_up_dev: Tensor,
    alpha_down_dev: Tensor,
}

impl Experts {
    fn new(alpha_gate_up: Vec<f32>, alpha_down: Vec<f32>, dev: Device) -> Result<Self, LoadError> {
        let e = alpha_gate_up.len();
        Ok(Self {
            alpha_gate_up_dev: dev_f32(alpha_gate_up.clone(), vec![e], dev)?,
            alpha_down_dev: dev_f32(alpha_down, vec![alpha_gate_up.len()], dev)?,
            alpha_gate_up,
        })
    }
}

/// Веса экспертов всех слоёв и пул слотов zero-copy. Режим (на карте /
/// zero-copy, число слотов) меняется на ходу по длине контекста
/// (`crate::experts::ExpertHub`), поэтому состояние общее и под замком:
/// прямой проход берёт его на чтение на время MoE слоя.
pub(crate) struct ExpertRt {
    /// По слою на физический слой (`stores[i]` — слой `i`).
    pub(crate) stores: Vec<ExpertStore>,
    /// Пул слотов zero-copy (режим экспертов в RAM).
    pub(crate) pool: Option<std::sync::Arc<ZcPool>>,
}

pub(crate) type SharedExperts = std::sync::Arc<std::sync::RwLock<ExpertRt>>;

/// Резидентные эксперты всех `layers` слоёв (веса — в пул весов карты).
/// Читаются срезами бандла; если стопки лежат в закреплённом зеркале, копия
/// идёт оттуда (DMA, без страничного кэша).
pub(crate) fn build_resident(w: &Inkling2Weights, dev: Device, layers: usize) -> Result<Vec<ExpertStore>, LoadError> {
    let _guard = synaptix_core::device::cuda::WeightsAllocGuard::for_device(dev);
    let mut out = Vec::with_capacity(layers);
    for i in 0..layers {
        let ExpertStack { gate_up, down, .. } = w.experts(i, dev)?;
        let gu_refs: Vec<&QuantWeight> = gate_up.iter().collect();
        let dn_refs: Vec<&QuantWeight> = down.iter().collect();
        let gt = ExpertTable::build_dense(&gu_refs).ok_or_else(|| LoadError::Weights(format!("слой {i}: таблица gate_up")))?;
        let dt = ExpertTable::build_dense(&dn_refs).ok_or_else(|| LoadError::Weights(format!("слой {i}: таблица down")))?;
        out.push(ExpertStore::Resident { gate_up: gt, down: dt, _weights: (gate_up, down) });
    }
    Ok(out)
}

/// Поставить экспертов всех слоёв (и пул слотов zero-copy). Захваченные
/// графы декода держат адреса таблиц прежних слоёв: поколение zero-copy
/// растёт, и графы переснимаются.
pub(crate) fn set_experts(xrt: &SharedExperts, stores: Vec<ExpertStore>, pool: Option<std::sync::Arc<ZcPool>>) {
    let mut g = xrt.write().unwrap_or_else(|e| e.into_inner());
    g.stores = stores;
    g.pool = pool;
    synaptix_llm_common::moe_zc::ZC_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Снять экспертов всех слоёв (после синхронизации карты: ничего в потоке
/// их уже не читает). Возвращает частоты выбора экспертов слоёв zero-copy —
/// затравку следующей сборки.
pub(crate) fn take_experts(xrt: &SharedExperts, dev: Device) -> Vec<Option<Vec<f32>>> {
    let mut g = xrt.write().unwrap_or_else(|e| e.into_inner());
    let seeds = g.stores.iter().map(|s| match s { ExpertStore::Zc(z) => Some(z.counts().to_vec()), _ => None }).collect();
    if let Device::Cuda(o) = dev {
        let _ = synaptix_core::device::cuda::synchronize_all(o);
    }
    for s in g.stores.iter_mut() {
        *s = ExpertStore::Pending;
    }
    g.pool = None;
    synaptix_llm_common::moe_zc::ZC_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    seeds
}

pub struct Layer {
    swa: bool,
    attn_norm: Option<Tensor>,
    qkv_norm: Option<[Tensor; 3]>,
    /// Слитые проекции с общим входом: `pa` = q | r | gate | sink (вход q),
    /// `pk` = k | отводы k-свёртки, `pv` = v | отводы v-свёртки.
    pa: Lin,
    pk: Lin,
    pv: Lin,
    wo: Lin,

    k_conv: Tensor,
    v_conv: Tensor,
    dyn_conv: bool,
    q_clip: Tensor,
    r_clip: Tensor,
    rel_proj: Tensor,
    extent: usize,
    attn_res_q: Option<Tensor>,
    ffn_res_q: Tensor,
    attn_post_conv: Tensor,
    ffn_post_conv: Tensor,
    ffn_norm: Tensor,
    router: Tensor,
    router_bias: Tensor,
    experts: Experts,
    lp: Option<LoopLayer>,
}

/// Формат KV слотов полного внимания (SWA-кольца всегда BF16: они малы).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFormat {
    /// BF16, 2 байта на элемент.
    Bf16,
    /// MXFP8: E4M3 + масштаб E8M0 на 32 канала (8,25 бита на элемент).
    Mxfp8,
    /// INT8 + масштаб E8M0 на 32 канала (8,25 бита на элемент).
    Int8,
}

impl KvFormat {
    /// Код формата для ядер (`u11`).
    pub fn code(self) -> u32 {
        match self {
            Self::Bf16 => 0,
            Self::Mxfp8 => 1,
            Self::Int8 => 2,
        }
    }

    /// Байт на элемент KV (с масштабами).
    pub fn bytes_per_elem(self) -> f64 {
        match self {
            Self::Bf16 => 2.0,
            Self::Mxfp8 | Self::Int8 => 1.0 + 1.0 / 32.0,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Mxfp8 => "mxfp8",
            Self::Int8 => "int8",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "bf16" => Some(Self::Bf16),
            "mxfp8" | "fp8" => Some(Self::Mxfp8),
            "int8" | "q8" => Some(Self::Int8),
            _ => None,
        }
    }
}

/// Кэш одного слота (физический слой на своём проходе цикла).
pub struct SlotCache {
    /// Кольцо KV SWA-слоя (окно + запас на чанк префилла); 0 — линейный кэш на весь контекст.
    ring: usize,
    /// Формат KV слота: у квантованного `k`/`v` — байты `[nkv, cap, d]` U8, масштабы E8M0 — в `scales`.
    fmt: KvFormat,
    k: Tensor,
    v: Tensor,
    /// Масштабы квант-KV `[2, nkv, cap, d/32]` U8 (k, затем v).
    scales: Option<Tensor>,
    k_state: Tensor,
    v_state: Tensor,
    attn_state: Tensor,
    ffn_state: Tensor,
}

/// CUDA-граф шага декода. Сырые указатели драйвера не `Send`; граф запускается
/// только через `&mut ModelCache` — по одному потоку за раз.
pub struct GraphHandle(std::sync::Arc<cudarc::driver::CudaGraph>);
// SAFETY: доступ исключительный (через `&mut` кэша), драйвер допускает вызовы из разных потоков по очереди.
unsafe impl Send for GraphHandle {}
unsafe impl Sync for GraphHandle {}

pub struct ModelCache {
    pub slots: Vec<SlotCache>,
    pub seq_len: usize,
    pub capacity: usize,
    /// Позиция начала прогона (U32 `[1]`): ядра читают её с карты, поэтому шаг
    /// декода от позиции не зависит и записывается в граф один раз.
    pos_dev: Tensor,
    /// Токен шага декода (U32 `[1]`) и логиты шага (F32 `[1, V]`) — постоянные буферы графа.
    tok_dev: Tensor,
    out_dev: Option<Tensor>,
    graph: Option<GraphHandle>,
    warm: u32,
    /// Поколение слотов zero-copy, при котором записан граф (адреса таблиц слоёв).
    zc_gen: u64,
}

impl ModelCache {
    /// Граф держит адреса KV и состояний: после их замены он негоден.
    fn drop_graph(&mut self) {
        // Последний запуск графа может ещё идти: его буферы нельзя отдавать, пока он не кончился.
        if self.graph.is_some() {
            if let Device::Cuda(o) = self.pos_dev.device() {
                let _ = synaptix_core::device::cuda::synchronize(o);
            }
        }
        self.graph = None;
        self.warm = 0;
    }

    /// Переселить KV и состояния свёрток в host-RAM, освободив VRAM. Позиция
    /// (`seq_len`) не меняется: после [`Self::unpark_to`] ход продолжается с
    /// той же точки. Граф шага держит адреса прежних буферов — сбрасывается и
    /// переснимается после возврата. Возвращает байты, уехавшие с карты.
    pub fn park_to_host(&mut self) -> R<usize> {
        self.drop_graph();
        self.out_dev = None;
        let mut moved = 0;
        for s in self.slots.iter_mut() {
            for t in [&mut s.k, &mut s.v, &mut s.k_state, &mut s.v_state, &mut s.attn_state, &mut s.ffn_state] {
                moved += move_tensor(t, Device::Cpu)?;
            }
            if let Some(t) = s.scales.as_mut() {
                moved += move_tensor(t, Device::Cpu)?;
            }
        }
        for t in [&mut self.pos_dev, &mut self.tok_dev] {
            moved += move_tensor(t, Device::Cpu)?;
        }
        Ok(moved)
    }

    /// Обратный переезд на карту `device`.
    pub fn unpark_to(&mut self, device: Device) -> R<usize> {
        self.drop_graph();
        let mut moved = 0;
        for s in self.slots.iter_mut() {
            for t in [&mut s.k, &mut s.v, &mut s.k_state, &mut s.v_state, &mut s.attn_state, &mut s.ffn_state] {
                moved += move_tensor(t, device)?;
            }
            if let Some(t) = s.scales.as_mut() {
                moved += move_tensor(t, device)?;
            }
        }
        for t in [&mut self.pos_dev, &mut self.tok_dev] {
            moved += move_tensor(t, device)?;
        }
        // Копии с хоста идут на стримах буферов: дождаться их до первого ядра хода.
        if let Device::Cuda(o) = device {
            synaptix_core::device::cuda::synchronize(o).map_err(fe)?;
        }
        Ok(moved)
    }

    /// Кэш сейчас в host-RAM.
    pub fn is_parked(&self) -> bool {
        self.slots.first().is_some_and(|s| s.k.device() == Device::Cpu)
    }

    /// Формат KV полных слотов (`None` — их нет).
    pub fn full_format(&self) -> Option<KvFormat> {
        self.slots.iter().find(|s| s.ring == 0 && s.fmt != KvFormat::Bf16).map(|s| s.fmt).or_else(|| self.slots.first().map(|_| KvFormat::Bf16))
    }

    /// Байт KV полных слотов (растущая часть кэша: K, V и масштабы),
    /// где бы кэш ни лежал.
    pub fn full_kv_bytes(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.ring == 0)
            .map(|s| {
                let b = |t: &Tensor| t.dtype().bytes_for_numel(t.numel());
                b(&s.k) + b(&s.v) + s.scales.as_ref().map_or(0, b)
            })
            .sum()
    }

    pub fn reset(&mut self) -> R<()> {
        self.drop_graph();
        self.seq_len = 0;
        for s in self.slots.iter_mut() {
            for t in [&mut s.k_state, &mut s.v_state, &mut s.attn_state, &mut s.ffn_state] {
                *t = Tensor::zeros(t.dims().to_vec(), DType::F32, t.device()).map_err(fe)?;
            }
        }
        Ok(())
    }

    /// Куски снимка в позиции `pos` с окном SWA `window`: (адрес на карте, байт)
    /// в каноническом порядке хранения — по слотам: четыре состояния свёрток,
    /// затем у SWA-слота строки K и V позиций `[pos − W, pos)` (W = min(pos,
    /// window)) по головам. Раскладка снимка от кольца не зависит: строка `p`
    /// лежит в кольце по `p % ring`, в линейном слоте — по `p` (слот, бывший
    /// линейным, после роста кэша становится кольцом).
    fn snap_pieces(&self, swa: &[bool], pos: usize, window: usize) -> R<Vec<(u64, usize)>> {
        let w = pos.min(window);
        let lo = pos - w;
        let mut out = Vec::new();
        for (s, &is_swa) in self.slots.iter().zip(swa) {
            for t in [&s.k_state, &s.v_state, &s.attn_state, &s.ffn_state] {
                out.push(t.cuda_region().map_err(fe)?);
            }
            if !is_swa || w == 0 {
                continue;
            }
            let (nkv, cap, d) = (s.k.dims()[0], s.k.dims()[1], s.k.dims()[2]);
            let row = s.k.dtype().bytes_for_numel(d);
            for t in [&s.k, &s.v] {
                let (base, _) = t.cuda_region().map_err(fe)?;
                for h in 0..nkv {
                    let head = base + (h * cap * row) as u64;
                    if s.ring == 0 {
                        out.push((head + (lo * row) as u64, w * row));
                        continue;
                    }
                    // Строки lo..pos в кольце: один отрезок или два с переходом через конец.
                    let a = lo % s.ring;
                    let first = w.min(s.ring - a);
                    out.push((head + (a * row) as u64, first * row));
                    if first < w {
                        out.push((head, (w - first) * row));
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Где лежит снимок точки возврата.
enum SnapStore {
    /// На карте одним буфером U8.
    Device(Tensor),
    /// В закреплённой RAM (карта занята либо сессия уехала в RAM).
    Host(synaptix_core::memory::pinned::PinnedBuf),
}

/// Снимок кэша в позиции `pos`: всё, что ход после неё портит безвозвратно, —
/// состояния коротких свёрток всех слотов и строки колец SWA в окне перед
/// `pos` (кольцо перезаписывается по кругу). KV полных слотов линейный и
/// ниже `pos` не переписывается — его не копируем, достаточно вернуть
/// `seq_len`.
pub struct CacheSnapshot {
    pos: usize,
    bytes: usize,
    /// Карта кэша (порядок копий с закреплённого буфера).
    ord: usize,
    /// Снимок был на карте и уехал в RAM вместе с кэшем ([`Self::park_to_host`]).
    parked: bool,
    store: SnapStore,
}

/// Снимок держать на карте, если после него там остаётся столько свободного.
/// `SYN_INKLING2_SNAP=device|host` — выбрать вручную.
const SNAP_DEVICE_MARGIN: usize = 1 << 30;

impl CacheSnapshot {
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn on_device(&self) -> bool {
        matches!(self.store, SnapStore::Device(_))
    }

    pub fn device_bytes(&self) -> usize {
        if self.on_device() { self.bytes } else { 0 }
    }

    /// Перевезти снимок в закреплённую RAM (на время вложенной генерации).
    /// Возвращает байты, ушедшие с карты.
    pub fn park_to_host(&mut self) -> R<usize> {
        let SnapStore::Device(t) = &self.store else { return Ok(0) };
        let Device::Cuda(ord) = t.device() else { return Ok(0) };
        let mut buf = synaptix_core::memory::pinned::PinnedBuf::new_uninit(self.bytes);
        let (src, _) = t.cuda_region().map_err(fe)?;
        let st = synaptix_core::device::cuda::default_stream(ord).map_err(fe)?;
        // SAFETY: буфер снимка жив, хост-буфер своей длины; ждём копию до отдачи буфера карты.
        unsafe { cudarc::driver::result::memcpy_dtoh_async(buf.as_mut_slice(), src, st.cu_stream()) }
            .map_err(|e| ModelError::Forward(format!("снимок в RAM: {e:?}")))?;
        synaptix_core::device::cuda::synchronize(ord).map_err(fe)?;
        self.store = SnapStore::Host(buf);
        self.parked = true;
        Ok(self.bytes)
    }

    /// Вернуть на карту снимок, уехавший с неё в [`Self::park_to_host`].
    /// Места нет — остаётся в RAM (восстановление идёт и оттуда). Возвращает
    /// байты, вернувшиеся на карту.
    pub fn unpark(&mut self) -> R<usize> {
        let SnapStore::Host(host) = &self.store else { return Ok(0) };
        if !self.parked {
            return Ok(0);
        }
        self.parked = false;
        let ord = self.ord;
        let free = synaptix_core::device::cuda::mem_info(ord).map(|(f, _)| f).unwrap_or(0);
        if free < self.bytes + SNAP_DEVICE_MARGIN {
            return Ok(0);
        }
        let Ok(buf) = empty(vec![self.bytes.max(1)], DType::U8, Device::Cuda(ord)) else { return Ok(0) };
        let (dst, _) = buf.cuda_region().map_err(fe)?;
        let st = synaptix_core::device::cuda::default_stream(ord).map_err(fe)?;
        // SAFETY: свежий буфер своей длины; хост-буфер жив до синхронизации ниже.
        unsafe { cudarc::driver::result::memcpy_htod_async(dst, &host.as_slice()[..self.bytes], st.cu_stream()) }
            .map_err(|e| ModelError::Forward(format!("снимок на карту: {e:?}")))?;
        synaptix_core::device::cuda::synchronize(ord).map_err(fe)?;
        self.store = SnapStore::Device(buf);
        Ok(self.bytes)
    }
}

impl Drop for CacheSnapshot {
    fn drop(&mut self) {
        // Копии с закреплённого буфера асинхронны: не отдавать его, пока они идут.
        if let SnapStore::Host(_) = self.store {
            let _ = synaptix_core::device::cuda::synchronize(self.ord);
        }
    }
}

/// Готовые эмбеддинги (F32 `[len, H]` на карте) вместо строк чанка с `row`.
pub struct EmbedPatch {
    pub row: usize,
    pub embeds: Tensor,
}

pub struct Inkling2Model {
    pub cfg: Inkling2Config,
    pub device: Device,
    embed: Tensor,
    embed_norm: Tensor,
    unembed: Lin,
    norm: Tensor,
    output_q: Tensor,
    layers: Vec<Layer>,
    /// Веса экспертов и пул zero-copy (режим меняется на ходу).
    pub(crate) xrt: SharedExperts,
    /// Формат KV слотов полного внимания у новых кэшей ([`Self::make_cache`]).
    kv_fmt: KvFormat,
}

fn lerr(e: impl std::fmt::Display) -> LoadError {
    LoadError::Weights(e.to_string())
}

/// Тензор F32 на карте из host-вектора.
fn dev_f32(v: Vec<f32>, shape: Vec<usize>, dev: Device) -> Result<Tensor, LoadError> {
    Tensor::from_vec(v, shape, dev).map_err(lerr)
}

/// Матрица `[r, c]` → транспонированная `[c, r]`.
fn transpose(v: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut t = vec![0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = v[i * c + j];
        }
    }
    t
}

/// Веса свёртки → `[C, K]`, последний отвод — текущий токен.
/// `conv1d.weight [C, 1, K]` уже так; `static_w [K, C]` — отвод по расстоянию назад.
fn conv_w(w: &Inkling2Weights, name: &str, k: usize, dev: Device) -> Result<Tensor, LoadError> {
    let v = w.host_f32(name)?;
    if name.ends_with("static_w") {
        let c = v.len() / k;
        let mut out = vec![0f32; c * k];
        for lag in 0..k {
            for ch in 0..c {
                out[ch * k + (k - 1 - lag)] = v[lag * c + ch];
            }
        }
        dev_f32(out, vec![c, k], dev)
    } else {
        let c = v.len() / k;
        dev_f32(v, vec![c, k], dev)
    }
}

impl Inkling2Model {
    /// Всё, кроме весов экспертов: их поднимает [`Self::load_resident_experts`]
    /// либо подключает [`Self::attach_zc`].
    pub fn load_base(w: &Inkling2Weights, dev: Device) -> Result<Self, LoadError> {
        let cfg = w.config.clone();
        let t = &cfg.text;
        let h = t.hidden_size;
        let k = t.sconv_kernel_size;
        let f32t = |name: &str| -> Result<Tensor, LoadError> { w.tensor(name, dev, DType::F32) };
        let embed = w.tensor(&format!("{LLM}.embed.weight"), dev, DType::BF16)?;
        let embed_norm = f32t(&format!("{LLM}.embed_norm.weight"))?;
        let unembed = Lin::load(w, &format!("{LLM}.unembed.weight"), dev)?;
        let norm = f32t(&format!("{LLM}.norm.weight"))?;
        let output_q = f32t(&format!("{LLM}.attn_res.output_query"))?.reshape(vec![1, h]).map_err(lerr)?;
        let (cs, _) = cfg.core();
        let msg = if t.loop_config.is_some() {
            Some((
                w.host_f32(&format!("{LLM}.middle_cycle_cross_pass_queries"))?,
                w.host_f32(&format!("{LLM}.middle_cycle_cross_pass_kv_queries"))?,
            ))
        } else {
            None
        };
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for i in 0..t.num_hidden_layers {
            let p = format!("{LLM}.layers.{i}");
            let a = format!("{p}.attn");
            let swa = cfg.is_swa(i);
            let extent = cfg.rel_extent(i);
            let (attn_norm, qkv_norm) = if swa {
                (Some(f32t(&format!("{p}.attn_norm.weight"))?), None)
            } else {
                (
                    None,
                    Some([
                        f32t(&format!("{p}.q_input_norm.weight"))?,
                        f32t(&format!("{p}.k_input_norm.weight"))?,
                        f32t(&format!("{p}.v_input_norm.weight"))?,
                    ]),
                )
            };
            let (k_conv, v_conv, k_conv_proj, v_conv_proj) = if swa {
                (conv_w(w, &format!("{a}.k_sconv.weight"), k, dev)?, conv_w(w, &format!("{a}.v_sconv.weight"), k, dev)?, None, None)
            } else {
                (
                    conv_w(w, &format!("{a}.k_sconv.static_w"), k, dev)?,
                    conv_w(w, &format!("{a}.v_sconv.static_w"), k, dev)?,
                    Some(format!("{a}.k_sconv.weight_proj")),
                    Some(format!("{a}.v_sconv.weight_proj")),
                )
            };
            let attn_res_q = if i > 0 {
                let q = w.host_f32(&format!("{LLM}.attn_res.attn_queries.{}", i - 1))?;
                let nq = q.len() / h;
                Some(dev_f32(q, vec![nq, h], dev)?)
            } else {
                None
            };
            let ffn_res_q = dev_f32(w.host_f32(&format!("{LLM}.attn_res.mlp_queries.{i}"))?, vec![1, h], dev)?;
            let router = w.host_f32(&format!("{p}.mlp.gate.weight"))?;
            let e = t.n_routed_experts;
            let router = dev_f32(router, vec![e, h], dev)?;
            let router_bias = dev_f32(w.host_f32(&format!("{p}.mlp.gate.bias"))?, vec![e], dev)?;
            let (alpha_gate_up, alpha_down) = w.expert_alpha(i)?.unwrap_or_else(|| (vec![1.0; e], vec![1.0; e]));
            let lp = if let (Some((q, kv)), true) = (&msg, cfg.loop_layer(i)) {
                let j = i - cs;
                let mut mq = Vec::with_capacity(4 * h);
                for row in [&q[2 * j * h..(2 * j + 1) * h], &kv[2 * j * h..(2 * j + 1) * h], &kv[(2 * j + 1) * h..(2 * j + 2) * h], &q[(2 * j + 1) * h..(2 * j + 2) * h]] {
                    mq.extend_from_slice(row);
                }
                let rank = t.loop_config.as_ref().map(|l| l.gate_rank).unwrap_or(64);
                let (mut down, mut gain, mut forget) = (Vec::new(), Vec::new(), Vec::new());
                for name in ["attn", "k", "v", "mlp"] {
                    let c = format!("{p}.controllers.{name}");
                    down.extend(w.host_f32(&format!("{c}.down_weight"))?);
                    gain.extend(transpose(&w.host_f32(&format!("{c}.gain_weight"))?, h, rank));
                    forget.extend(transpose(&w.host_f32(&format!("{c}.forget_weight"))?, h, rank));
                }
                Some(LoopLayer {
                    msg_q: dev_f32(mq, vec![4, h], dev)?,
                    down: dev_f32(down, vec![4, rank, 2 * h], dev)?,
                    gain: dev_f32(gain, vec![4, rank, h], dev)?,
                    forget: dev_f32(forget, vec![4, rank, h], dev)?,
                    rank,
                })
            } else {
                None
            };
            layers.push(Layer {
                swa,
                attn_norm,
                qkv_norm,
                pa: Lin::cat(w, &[format!("{a}.wq_du.weight"), format!("{a}.wr_du.weight"), format!("{a}.gate_attn.weight"), format!("{a}.attn_sink_proj.weight")], dev)?,
                pk: Lin::cat(w, &[&[format!("{a}.wk_dv.weight")][..], k_conv_proj.as_slice()].concat(), dev)?,
                pv: Lin::cat(w, &[&[format!("{a}.wv_dv.weight")][..], v_conv_proj.as_slice()].concat(), dev)?,
                wo: Lin::load(w, &format!("{a}.wo_ud.weight"), dev)?,

                k_conv,
                v_conv,
                dyn_conv: !swa,
                q_clip: f32t(&format!("{a}.q_l2_clip.scale"))?,
                r_clip: f32t(&format!("{a}.rel_l2_clip.scale"))?,
                rel_proj: f32t(&format!("{a}.rel_logits_proj.proj"))?,
                extent,
                attn_res_q,
                ffn_res_q,
                attn_post_conv: conv_w(w, &format!("{p}.attn_sconv.weight"), k, dev)?,
                ffn_post_conv: conv_w(w, &format!("{p}.mlp_sconv.weight"), k, dev)?,
                ffn_norm: f32t(&format!("{p}.mlp_norm.weight"))?,
                router,
                router_bias,
                experts: Experts::new(alpha_gate_up, alpha_down, dev)?,
                lp,
            });
        }
        let stores = (0..layers.len()).map(|_| ExpertStore::Pending).collect();
        let xrt = std::sync::Arc::new(std::sync::RwLock::new(ExpertRt { stores, pool: None }));
        Ok(Self { cfg, device: dev, embed, embed_norm, unembed, norm, output_q, layers, xrt, kv_fmt: KvFormat::Bf16 })
    }

    /// Модель целиком, эксперты резидентны на карте.
    pub fn load(w: &Inkling2Weights, dev: Device) -> Result<Self, LoadError> {
        let mut m = Self::load_base(w, dev)?;
        m.load_resident_experts(w)?;
        Ok(m)
    }

    /// Поднять экспертов всех слоёв на карту (резидентный режим).
    pub fn load_resident_experts(&mut self, w: &Inkling2Weights) -> Result<(), LoadError> {
        let stores = build_resident(w, self.device, self.layers.len())?;
        set_experts(&self.xrt, stores, None);
        Ok(())
    }

    /// Режим экспертов в RAM: по [`ZcLayer`] на физический слой (`layers[i]` —
    /// слой `i`, свой dtype) и общий пул слотов.
    pub fn attach_zc(&mut self, pool: std::sync::Arc<ZcPool>, layers: Vec<ZcLayer>) -> Result<(), LoadError> {
        if layers.len() != self.layers.len() {
            return Err(LoadError::Weights(format!("zero-copy: слоёв {} из {}", layers.len(), self.layers.len())));
        }
        set_experts(&self.xrt, layers.into_iter().map(ExpertStore::Zc).collect(), Some(pool));
        Ok(())
    }

    /// Эксперты в RAM (zero-copy), а не на карте.
    pub fn zc_active(&self) -> bool {
        self.xrt.read().map(|g| g.pool.is_some()).unwrap_or(false)
    }

    /// Эксперты подняты (на карте или zero-copy).
    pub fn experts_ready(&self) -> bool {
        self.xrt.read().map(|g| g.stores.iter().all(|s| !matches!(s, ExpertStore::Pending))).unwrap_or(false)
    }

    /// Слотов VRAM на слой в режиме zero-copy.
    pub fn zc_slots(&self) -> Option<usize> {
        let g = self.xrt.read().ok()?;
        g.pool.as_ref()?;
        g.stores.iter().find_map(|s| match s { ExpertStore::Zc(z) => Some(z.slots()), _ => None })
    }

    /// Счётчики zero-copy по всем слоям: (попадания в слоты, выборы
    /// экспертов). Синхронное чтение — только для журналов и замеров.
    pub fn zc_stats(&self) -> Option<(u64, u64)> {
        let g = self.xrt.read().ok()?;
        g.pool.as_ref()?;
        let mut acc = (0, 0);
        for s in &g.stores {
            if let ExpertStore::Zc(z) = s {
                let (h, p) = z.stats();
                acc.0 += h;
                acc.1 += p;
            }
        }
        Some(acc)
    }

    /// Формат KV полных слотов для кэшей, созданных после вызова.
    pub fn set_kv_format(&mut self, f: KvFormat) {
        self.kv_fmt = f;
    }

    pub fn kv_format(&self) -> KvFormat {
        self.kv_fmt
    }

    /// Физический слой слота кэша.
    fn slot_phys(&self, slot: usize) -> usize {
        let n = self.cfg.text.num_hidden_layers;
        if slot < n {
            return slot;
        }
        let (s, e) = self.cfg.core();
        s + (slot - n) % (e - s)
    }

    /// Кольцо KV слота: у SWA-слоёв — окно + наибольший чанк префилла, если это меньше контекста.
    fn slot_ring(&self, slot: usize, capacity: usize) -> usize {
        let ring = self.cfg.text.sliding_window_size + MAX_CHUNK;
        if self.cfg.is_swa(self.slot_phys(slot)) && ring < capacity { ring } else { 0 }
    }

    pub fn make_cache(&self, capacity: usize) -> R<ModelCache> {
        self.make_cache_fmt(capacity, self.kv_fmt)
    }

    /// Кэш на `capacity` позиций с KV полных слотов в формате `fmt`.
    pub fn make_cache_fmt(&self, capacity: usize, fmt_full: KvFormat) -> R<ModelCache> {
        let t = &self.cfg.text;
        let (nkv, d, h, k) = (t.num_key_value_heads, t.head_dim, t.hidden_size, t.sconv_kernel_size);
        let dev = self.device;
        let z = |shape: Vec<usize>, dt: DType| Tensor::zeros(shape, dt, dev).map_err(fe);
        let mut slots = Vec::with_capacity(self.cfg.num_slots());
        for si in 0..self.cfg.num_slots() {
            let ring = self.slot_ring(si, capacity);
            let cap = if ring > 0 { ring } else { capacity };
            let fmt = if self.cfg.is_swa(self.slot_phys(si)) { KvFormat::Bf16 } else { fmt_full };
            let (kdt, scales) = match fmt {
                KvFormat::Bf16 => (DType::BF16, None),
                _ => (DType::U8, Some(z(vec![2, nkv, cap, d / 32], DType::U8)?)),
            };
            slots.push(SlotCache {
                ring,
                fmt,
                k: z(vec![nkv, cap, d], kdt)?,
                v: z(vec![nkv, cap, d], kdt)?,
                scales,
                k_state: z(vec![k - 1, nkv * d], DType::F32)?,
                v_state: z(vec![k - 1, nkv * d], DType::F32)?,
                attn_state: z(vec![k - 1, h], DType::F32)?,
                ffn_state: z(vec![k - 1, h], DType::F32)?,
            });
        }
        Ok(ModelCache {
            slots,
            seq_len: 0,
            capacity,
            pos_dev: z(vec![1], DType::U32)?,
            tok_dev: z(vec![1], DType::U32)?,
            out_dev: None,
            graph: None,
            warm: 0,
            zc_gen: 0,
        })
    }

    /// SWA ли слот (по слотам кэша).
    fn swa_slots(&self) -> Vec<bool> {
        (0..self.cfg.num_slots()).map(|si| self.cfg.is_swa(self.slot_phys(si))).collect()
    }

    /// Снять точку возврата в текущей позиции кэша (`seq_len`): копии ставятся
    /// в поток по порядку с ядрами, так что снимок — ровно состояние после
    /// последнего прогона. Хранится на карте, если после него там остаётся
    /// запас, иначе — в закреплённой RAM.
    pub fn snapshot(&self, cache: &ModelCache) -> R<CacheSnapshot> {
        let Device::Cuda(ord) = self.device else {
            return Err(ModelError::Forward("inkling2: снимок кэша — только на CUDA".into()));
        };
        if cache.is_parked() {
            return Err(ModelError::Forward("inkling2: снимок припаркованного кэша".into()));
        }
        let pos = cache.seq_len;
        let pieces = cache.snap_pieces(&self.swa_slots(), pos, self.cfg.text.sliding_window_size)?;
        let bytes: usize = pieces.iter().map(|p| p.1).sum();
        let on_device = match std::env::var("SYN_INKLING2_SNAP").ok().as_deref().map(str::trim) {
            Some("host" | "ram") => false,
            Some("device" | "vram" | "gpu") => true,
            _ => synaptix_core::device::cuda::mem_info(ord).map(|(f, _)| f).unwrap_or(0) >= bytes + SNAP_DEVICE_MARGIN,
        };
        let st = synaptix_core::device::cuda::default_stream(ord).map_err(fe)?;
        let store = if on_device {
            let buf = empty(vec![bytes.max(1)], DType::U8, self.device)?;
            let (base, _) = buf.cuda_region().map_err(fe)?;
            let mut off = 0u64;
            for &(src, len) in &pieces {
                synaptix_core::device::cuda::dtod_into_region(&st, base + off, src, len).map_err(fe)?;
                off += len as u64;
            }
            SnapStore::Device(buf)
        } else {
            let mut buf = synaptix_core::memory::pinned::PinnedBuf::new_uninit(bytes.max(1));
            let mut off = 0usize;
            for &(src, len) in &pieces {
                // SAFETY: куски — регионы живых буферов кэша; хост-буфер живёт в снимке,
                // а его отдача ждёт поток (Drop снимка).
                unsafe { cudarc::driver::result::memcpy_dtoh_async(&mut buf.as_mut_slice()[off..off + len], src, st.cu_stream()) }
                    .map_err(|e| ModelError::Forward(format!("снимок кэша в RAM: {e:?}")))?;
                off += len;
            }
            SnapStore::Host(buf)
        };
        Ok(CacheSnapshot { pos, bytes, ord, parked: false, store })
    }

    /// Вернуть кэш в точку снимка: состояния и строки колец копируются в те же
    /// буферы (граф декода остаётся годным — адреса прежние), `seq_len` — на
    /// позицию снимка. Годен, только если с момента снимка кэш не сбрасывался
    /// (строки полных слотов ниже позиции целы) — за этим следит сессия.
    pub fn restore(&self, cache: &mut ModelCache, snap: &CacheSnapshot) -> R<()> {
        let Device::Cuda(ord) = self.device else {
            return Err(ModelError::Forward("inkling2: снимок кэша — только на CUDA".into()));
        };
        if cache.is_parked() {
            return Err(ModelError::Forward("inkling2: кэш в RAM (unpark_to перед ходом)".into()));
        }
        if snap.pos > cache.capacity {
            return Err(ModelError::Forward(format!("inkling2: снимок на {} за ёмкостью {}", snap.pos, cache.capacity)));
        }
        let pieces = cache.snap_pieces(&self.swa_slots(), snap.pos, self.cfg.text.sliding_window_size)?;
        let bytes: usize = pieces.iter().map(|p| p.1).sum();
        if bytes != snap.bytes {
            return Err(ModelError::Forward(format!("inkling2: снимок {} Б, кэшу нужно {bytes}", snap.bytes)));
        }
        let st = synaptix_core::device::cuda::default_stream(ord).map_err(fe)?;
        match &snap.store {
            SnapStore::Device(buf) => {
                let (base, _) = buf.cuda_region().map_err(fe)?;
                let mut off = 0u64;
                for &(dst, len) in &pieces {
                    synaptix_core::device::cuda::dtod_into_region(&st, dst, base + off, len).map_err(fe)?;
                    off += len as u64;
                }
            }
            SnapStore::Host(buf) => {
                let mut off = 0usize;
                for &(dst, len) in &pieces {
                    // SAFETY: регионы буферов кэша; хост-буфер живёт в снимке до конца копий (Drop ждёт поток).
                    unsafe { cudarc::driver::result::memcpy_htod_async(dst, &buf.as_slice()[off..off + len], st.cu_stream()) }
                        .map_err(|e| ModelError::Forward(format!("снимок кэша из RAM: {e:?}")))?;
                    off += len;
                }
            }
        }
        cache.seq_len = snap.pos;
        Ok(())
    }

    /// Расширить кэш до `capacity` позиций: KV копируется, состояния свёрток те же.
    pub fn grow_cache(&self, cache: &mut ModelCache, capacity: usize) -> R<()> {
        self.grow_cache_fmt(cache, capacity, None)
    }

    /// Как [`Self::grow_cache`], но KV полных слотов ещё и переводится в
    /// формат `fmt` (если он другой): BF16 → MXFP8/INT8 переквантуется на
    /// карте (`ink2_kv_requant`, то же кодирование, что у записи `ink2_qkv`).
    /// Позиции, `seq_len` и точка возврата остаются годными: снимок полные
    /// слоты не хранит. Слоты идут по одному — пик памяти сверх итоговой
    /// ёмкости не больше одного тензора K или V прежнего кэша.
    pub fn grow_cache_fmt(&self, cache: &mut ModelCache, capacity: usize, fmt: Option<KvFormat>) -> R<()> {
        let capacity = capacity.max(cache.capacity);
        let convert = fmt.is_some_and(|f| cache.full_format().is_some_and(|c| c != f));
        if capacity == cache.capacity && !convert {
            return Ok(());
        }
        if cache.is_parked() {
            return Err(ModelError::Forward("inkling2: рост кэша в RAM (unpark_to перед ростом)".into()));
        }
        cache.drop_graph();
        let used = cache.seq_len;
        for (si, s) in cache.slots.iter_mut().enumerate() {
            if s.ring > 0 {
                continue;
            }
            let ring = self.slot_ring(si, capacity);
            if ring > 0 {
                // Слот, бывший линейным на малом контексте, становится кольцом: переложить хвост окна.
                let (nkv, d) = (s.k.dims()[0], s.k.dims()[2]);
                let lo = used.saturating_sub(ring);
                for kv in [&mut s.k, &mut s.v] {
                    let host: Vec<half::bf16> = kv.to_device(Device::Cpu).map_err(fe)?.flatten_all().map_err(fe)?.to_vec1::<half::bf16>().map_err(fe)?;
                    let cap_old = kv.dims()[1];
                    let mut buf = vec![half::bf16::ZERO; nkv * ring * d];
                    for h in 0..nkv {
                        for p in lo..used {
                            let src = (h * cap_old + p) * d;
                            let dst = (h * ring + p % ring) * d;
                            buf[dst..dst + d].copy_from_slice(&host[src..src + d]);
                        }
                    }
                    *kv = Tensor::from_vec(buf, vec![nkv, ring, d], self.device).map_err(fe)?;
                }
                s.ring = ring;
                continue;
            }
            let swa = self.cfg.is_swa(self.slot_phys(si));
            if let Some(f) = fmt.filter(|&f| !swa && f != s.fmt) {
                requant_slot(s, f, used, capacity, self.device)?;
                continue;
            }
            for kv in [&mut s.k, &mut s.v] {
                grow_rows(kv, 1, used, capacity)?;
            }
            if let Some(sc) = s.scales.as_mut() {
                grow_rows(sc, 2, used, capacity)?;
            }
        }
        cache.capacity = capacity;
        Ok(())
    }

    /// Прогон токенов `ids` с позиции `cache.seq_len`. Возвращает логиты
    /// `[1, V]` последнего токена (или всех `[T, V]` при `all_logits`), F32,
    /// словарь без паддинга.
    pub fn forward(&self, ids: &[u32], cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        self.forward_with_embeds(ids, &[], cache, all_logits)
    }

    /// Как [`Self::forward`], но строки `patches` заменяются готовыми
    /// эмбеддингами (признаки картинки) — после `embed_norm`, без нормировки.
    /// `patches[i].row` — строка относительно всего `ids`; длинный прогон
    /// режется на куски по [`MAX_CHUNK`], патчи — по кускам.
    pub fn forward_with_embeds(&self, ids: &[u32], patches: &[EmbedPatch], cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        if ids.len() <= MAX_CHUNK {
            return self.forward_chunk(ids, patches, cache, all_logits);
        }
        let mut outs = Vec::new();
        let mut off = 0;
        while off < ids.len() {
            let end = (off + MAX_CHUNK).min(ids.len());
            let mut part = Vec::new();
            for p in patches {
                let len = p.embeds.dims()[0];
                let (a, b) = (p.row.max(off), (p.row + len).min(end));
                if a < b {
                    part.push(EmbedPatch { row: a - off, embeds: p.embeds.narrow(0, a - p.row, b - a).map_err(fe)? });
                }
            }
            outs.push(self.forward_chunk(&ids[off..end], &part, cache, all_logits)?);
            off = end;
        }
        if !all_logits {
            return Ok(outs.pop().expect("непустой прогон"));
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        Tensor::cat(&refs, 0).map_err(fe)
    }

    fn forward_chunk(&self, ids: &[u32], patches: &[EmbedPatch], cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        let n = ids.len();
        let past = cache.seq_len;
        if cache.is_parked() {
            return Err(ModelError::Forward("inkling2: кэш в RAM (unpark_to перед ходом)".into()));
        }
        if past + n > cache.capacity {
            return Err(ModelError::Forward(format!("inkling2: кэш {} < {}", cache.capacity, past + n)));
        }
        set_u32(&cache.pos_dev, past as u32)?;
        let idt = Tensor::from_vec(ids.to_vec(), vec![n], self.device).map_err(fe)?;
        let out = self.forward_dev_with(&idt, patches, cache, all_logits)?;
        cache.seq_len = past + n;
        Ok(out)
    }

    /// Шаг декода одного токена: после прогрева — проигрыш CUDA-графа.
    /// Возвращает логиты `[1, V]` (буфер графа: годен до следующего шага).
    pub fn decode(&self, tok: u32, cache: &mut ModelCache) -> R<Tensor> {
        let past = cache.seq_len;
        if cache.is_parked() {
            return Err(ModelError::Forward("inkling2: кэш в RAM (unpark_to перед ходом)".into()));
        }
        if past + 1 > cache.capacity {
            return Err(ModelError::Forward(format!("inkling2: кэш {} < {}", cache.capacity, past + 1)));
        }
        let graph_on = std::env::var("SYN_INKLING2_GRAPH").as_deref() != Ok("0") && self.device.is_cuda();
        set_u32(&cache.pos_dev, past as u32)?;
        set_u32(&cache.tok_dev, tok)?;
        // Слои zero-copy пересобраны — таблицы на новых адресах, граф негоден.
        let zc_gen = synaptix_llm_common::moe_zc::ZC_GENERATION.load(std::sync::atomic::Ordering::Relaxed);
        if cache.graph.is_some() && cache.zc_gen != zc_gen {
            cache.drop_graph();
        }
        if let (true, Some(g)) = (graph_on, &cache.graph) {
            g.0.launch().map_err(|e| ModelError::Forward(format!("граф декода: {e:?}")))?;
            cache.seq_len = past + 1;
            return Ok(cache.out_dev.clone().expect("буфер графа"));
        }
        if !graph_on || cache.warm < 2 {
            let tokt = cache.tok_dev.clone();
            let out = self.forward_dev(&tokt, cache, false)?;
            cache.warm += 1;
            cache.seq_len = past + 1;
            return Ok(out);
        }
        let Device::Cuda(ord) = self.device else { unreachable!() };
        let vu = self.cfg.text.unpadded_vocab_size;
        if cache.out_dev.is_none() {
            cache.out_dev = Some(Tensor::zeros(vec![1, vu], DType::F32, self.device).map_err(fe)?);
        }
        let stream = synaptix_core::device::cuda::default_stream(ord).map_err(fe)?;
        let tokt = cache.tok_dev.clone();
        let out_buf = cache.out_dev.clone().expect("буфер");
        let mut err: Option<ModelError> = None;
        let mut cap = synaptix_infer::graph_capture::GraphCapturer::new(0);
        let res = cap.capture_with(&stream, |_| {
            let r = self.forward_dev(&tokt, cache, false).and_then(|l| copy_into(&out_buf, &l));
            r.map_err(|e| {
                let m = e.to_string();
                err = Some(e);
                synaptix_infer::InferError::Other(m)
            })
        });
        match (res, err) {
            (Ok(g), None) => {
                g.launch().map_err(|e| ModelError::Forward(format!("граф декода: {e:?}")))?;
                cache.graph = Some(GraphHandle(g));
                cache.zc_gen = zc_gen;
                cache.seq_len = past + 1;
                Ok(out_buf)
            }
            (_, Some(e)) => Err(e),
            (Err(e), None) => Err(ModelError::Forward(format!("граф декода: захват: {e}"))),
        }
    }

    /// Прогон по токенам на карте (`idt` U32 `[T]`) с позиции из `cache.pos_dev`;
    /// хост в нём не участвует, кроме маршрута MoE префилла (T > 1).
    fn forward_dev(&self, idt: &Tensor, cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        self.forward_dev_with(idt, &[], cache, all_logits)
    }

    fn forward_dev_with(&self, idt: &Tensor, patches: &[EmbedPatch], cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        let t = &self.cfg.text;
        let n = idt.dims()[0];
        let past = cache.seq_len;
        let dev = self.device;
        let x = self.embed.embed_gather(idt).map_err(fe)?;
        let x = f32c(&x.reshape(vec![n, t.hidden_size]).map_err(fe)?)?;
        let mut cur = kern::rms(&x, Some(&self.embed_norm), t.rms_norm_eps, false)?;
        for p in patches {
            let len = p.embeds.dims()[0];
            if p.row + len > n || p.embeds.dims().get(1) != Some(&t.hidden_size) {
                return Err(ModelError::Shape(format!("inkling2: эмбеддинги {:?} с строки {} в чанк {n}", p.embeds.dims(), p.row)));
            }
            copy_into(&cur.narrow(0, p.row, len).map_err(fe)?, &f32c(&p.embeds)?)?;
        }

        let (cs, ce) = self.cfg.core();
        let passes = self.cfg.num_passes();
        let mut blocks: Vec<Tensor> = Vec::new();
        let first_loop = if t.loop_config.is_some() { cs } else { t.num_hidden_layers };
        for ip in 0..first_loop {
            cur = self.layer(ip, ip, cur, &mut blocks, None, cache, past)?;
        }
        if t.loop_config.is_some() {
            let prelude = cur.clone();
            let prelude_blocks = blocks.clone();
            for pass in 0..passes {
                let mut bank: Vec<Tensor> = Vec::new();
                if pass > 0 {
                    bank.extend(blocks[prelude_blocks.len() + 1..].iter().cloned());
                    bank.push(cur.clone());
                    cur = prelude.clone();
                    blocks = prelude_blocks.clone();
                }
                for ip in cs..ce {
                    let slot = self.cfg.slot(ip, pass);
                    let messages = if pass > 0 {
                        let lp = self.layers[ip].lp.as_ref().ok_or_else(|| ModelError::Forward("inkling2: нет контроллеров".into()))?;
                        let refs: Vec<&Tensor> = bank.iter().collect();
                        Some(kern::mix(&refs, &lp.msg_q)?)
                    } else {
                        None
                    };
                    cur = self.layer(ip, slot, cur, &mut blocks, messages.as_ref(), cache, past)?;
                }
            }
            for ip in ce..t.num_hidden_layers {
                cur = self.layer(ip, ip, cur, &mut blocks, None, cache, past)?;
            }
        }
        let mut states: Vec<&Tensor> = blocks.iter().collect();
        states.push(&cur);
        let out = kern::pick(&kern::mix(&states, &self.output_q)?, 0)?;
        let out = f32c(&out)?;
        let out = if all_logits { out } else { f32c(&out.narrow(0, n - 1, 1).map_err(fe)?)? };
        let rows = out.dims()[0];
        let xn = kern::rms(&out, Some(&self.norm), t.rms_norm_eps, true)?;
        let logits = bf16c(&self.unembed.raw(&xn)?)?;
        let vu = t.unpadded_vocab_size;
        let res = empty(vec![rows, vu], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&logits)?;
        a.a[1] = addr(&res)?;
        a.u[0] = rows as u32;
        a.u[1] = logits.dims()[1] as u32;
        a.u[2] = vu as u32;
        a.u[3] = 1;
        a.f[0] = t.final_logit_softcapping;
        launch(dev, "ink2_logits", &a, ((vu as u32).div_ceil(256), rows as u32, 1), 256)?;
        Ok(res)
    }

    /// Контроллеры `first..first+nz` слоя одним запуском на этап: вход `j` — строка `j` из `x`
    /// (`x_z` — шаг между входами в строках `[T, H]`, 0 — общий вход), сообщение — строка
    /// `first + j` из `msg [4, T, H]`. Выход `[nz, T, H]` F32.
    fn control(&self, x: &Tensor, x_z: usize, msg: &Tensor, lp: &LoopLayer, first: usize, nz: usize) -> R<Tensor> {
        let (n, h) = (msg.dims()[1], msg.dims()[2]);
        let dev = self.device;
        let th = n * h;
        let r = lp.rank;
        let d = empty(vec![nz, n, r], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(x)?;
        a.a[1] = addr(msg)? + (first * th * 4) as u64;
        a.a[2] = addr(&lp.down)? + (first * r * 2 * h * 4) as u64;
        a.a[3] = addr(&d)?;
        a.u[0] = n as u32;
        a.u[1] = h as u32;
        a.u[2] = r as u32;
        a.u[3] = (x_z * th) as u32;
        a.u[4] = th as u32;
        a.u[5] = (r * 2 * h) as u32;
        a.u[6] = (n * r) as u32;
        launch(dev, "ink2_ctrl_down", &a, (n as u32, r as u32, nz as u32), 256)?;
        let out = empty(vec![nz, n, h], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(x)?;
        a.a[1] = addr(msg)? + (first * th * 4) as u64;
        a.a[2] = addr(&d)?;
        a.a[3] = addr(&lp.gain)? + (first * r * h * 4) as u64;
        a.a[4] = addr(&lp.forget)? + (first * r * h * 4) as u64;
        a.a[5] = addr(&out)?;
        a.u[0] = n as u32;
        a.u[1] = h as u32;
        a.u[2] = r as u32;
        a.u[3] = (x_z * th) as u32;
        a.u[4] = th as u32;
        a.u[5] = (r * h) as u32;
        a.u[6] = (n * r) as u32;
        launch(dev, "ink2_ctrl_out", &a, ((h as u32).div_ceil(256), n as u32, nz as u32), 256)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(&self, ip: usize, slot: usize, cur: Tensor, blocks: &mut Vec<Tensor>, messages: Option<&Tensor>, cache: &mut ModelCache, past: usize) -> R<Tensor> {
        let t = &self.cfg.text;
        let l = &self.layers[ip];
        let n = cur.dims()[0];
        let (nh, nkv, d, dr) = (t.num_attention_heads, t.num_key_value_heads, t.head_dim, t.d_rel);
        let eps = t.rms_norm_eps;
        let dev = self.device;
        let inp_sa = cur.clone();

        // Входы q/k/v: строки смеси блоков (у SWA — одна на всех), затем контроллеры (проход > 0)
        // и RMS-нормы — по одному запуску на этап для всех трёх входов (`blockIdx.z`).
        let mixed = match &l.attn_res_q {
            Some(q) => {
                let mut states: Vec<&Tensor> = blocks.iter().collect();
                states.push(&cur);
                Some(kern::stage("mix", dev, || kern::mix(&states, q))?)
            }
            None => None,
        };
        let src = mixed.as_ref().unwrap_or(&cur);
        let src_z = if mixed.is_some() && !l.swa { 1 } else { 0 };
        let normed: [Tensor; 3] = {
            let (x, x_z, nz) = match messages {
                Some(m) => {
                    let lp = l.lp.as_ref().expect("контроллеры есть, раз есть сообщения");
                    (kern::stage("control", dev, || self.control(src, src_z, m, lp, 0, 3))?, 1, 3)
                }
                None => (src.clone(), src_z, if l.swa { 1 } else { 3 }),
            };
            let ws: Vec<&Tensor> = match (&l.attn_norm, &l.qkv_norm) {
                (Some(w), _) => vec![w; nz],
                (None, Some(w)) => w.iter().collect(),
                _ => unreachable!("нормы входа"),
            };
            let y = kern::rms_z(&x, x_z, n, &ws, eps)?;
            let row = |j: usize| y[j.min(nz - 1)].clone();
            [row(0), row(1), row(2)]
        };

        let sc = &mut cache.slots[slot];
        let k_scale = 1.0 / ((t.sconv_kernel_size + 1) as f32).sqrt();
        let kd_scale = 1.0 / ((2 * t.sconv_kernel_size + 1) as f32).sqrt();
        // Выходы слитых проекций (BF16) читаются ядрами срезами с шагом строки — без приведения и копий.
        let ya = kern::stage("proj_a", dev, || l.pa.raw(&normed[0]))?;
        let (q, r, gate, sink) = (kern::cols(&ya, 0)?, kern::cols(&ya, nh * d)?, kern::cols(&ya, nh * d + nh * dr)?, kern::cols(&ya, 2 * nh * d + nh * dr)?);
        let (yk, yv) = kern::stage("proj_kv", dev, || -> R<_> { Ok((l.pk.raw(&normed[1])?, l.pv.raw(&normed[2])?)) })?;
        let ckv = nkv * d;
        // Свёртки k и v — одним запуском.
        let dynk = if l.dyn_conv { Some((kern::cols(&yk, ckv)?, nkv)) } else { None };
        let dynv = if l.dyn_conv { Some((kern::cols(&yv, ckv)?, nkv)) } else { None };
        let jobs = [
            kern::ConvJob { x: kern::cols(&yk, 0)?, w: &l.k_conv, state: &sc.k_state, dynamic: dynk },
            kern::ConvJob { x: kern::cols(&yv, 0)?, w: &l.v_conv, state: &sc.v_state, dynamic: dynv },
        ];
        let mut kv = kern::conv(&jobs, n, ckv, None, if l.dyn_conv { kd_scale } else { k_scale }, dev)?;
        let (vc, kc) = (kv.pop().expect("v"), kv.pop().expect("k"));
        let q_out = empty(vec![n, nh * d], DType::F32, dev)?;
        let r_out = empty(vec![n, nh * dr], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = q.addr;
        a.a[1] = addr(&kc)?;
        a.a[2] = addr(&vc)?;
        a.a[3] = r.addr;
        a.u[12] = q.bf16;
        a.u[13] = q.stride;
        a.a[4] = addr(&l.q_clip)?;
        a.a[5] = addr(&l.r_clip)?;
        a.a[6] = addr(&q_out)?;
        a.a[7] = addr(&r_out)?;
        a.a[8] = addr(&sc.k)?;
        a.a[9] = addr(&sc.v)?;
        a.u[0] = n as u32;
        a.u[1] = nh as u32;
        a.u[2] = nkv as u32;
        a.u[3] = d as u32;
        a.u[4] = dr as u32;
        a.u[5] = sc.k.dims()[1] as u32;
        a.u[6] = past as u32;
        a.u[7] = (!l.swa) as u32;
        a.u[9] = sc.ring as u32;
        a.u[8] = t.log_scaling_n_floor as u32;
        if let Some(scl) = &sc.scales {
            a.a[11] = addr(scl)?;
            a.u[11] = sc.fmt.code();
        }
        a.f[0] = t.qk_l2_clip_threshold;
        a.f[1] = t.rel_l2_clip_threshold;
        a.f[2] = eps;
        a.f[3] = t.log_scaling_alpha;
        a.a[10] = addr(&cache.pos_dev)?;
        kern::stage("k_qkv", dev, || launch(dev, "ink2_qkv", &a, (n as u32, (nh + 2 * nkv) as u32, 1), d as u32))?;

        let att = empty(vec![n, nh * d], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&q_out)?;
        a.a[1] = addr(&r_out)?;
        a.a[2] = addr(&l.rel_proj)?;
        a.a[3] = sink.addr;
        a.a[4] = addr(&sc.k)?;
        a.a[5] = addr(&sc.v)?;
        a.a[6] = gate.addr;
        a.u[12] = sink.bf16;
        a.u[13] = sink.stride;
        a.a[7] = addr(&att)?;
        a.u[0] = n as u32;
        a.u[1] = nh as u32;
        a.u[2] = nkv as u32;
        a.u[3] = d as u32;
        a.u[4] = dr as u32;
        a.u[5] = sc.k.dims()[1] as u32;
        a.u[6] = past as u32;
        a.u[7] = if l.swa { t.sliding_window_size as u32 } else { 0 };
        a.u[10] = sc.ring as u32;
        a.u[8] = l.extent as u32;
        if let Some(scl) = &sc.scales {
            a.a[11] = addr(scl)?;
            a.u[11] = sc.fmt.code();
        }
        a.f[0] = 1.0 / (d as f32).sqrt();
        a.a[10] = addr(&cache.pos_dev)?;
        // Части по ключам на декоде: число фиксировано от ёмкости окна (граф), ~64 позиции на часть.
        let span = if l.swa { t.sliding_window_size.min(cache.capacity) } else { cache.capacity };
        // Префилл (T > 1) — flash-ядро на тензорных ядрах ink2_attn_fa (bias, окно, кольцо, синк, гейт);
        // части по ключам — от длины контекста, сборка — ink2_attn_merge.
        // Декод (T = 1): прежнее ink2_attn, число частей фиксировано от ёмкости (CUDA-граф).
        let fa = n > 1 && d == 128 && dr == 16;
        let splits = if fa {
            let keys = if l.swa { (t.sliding_window_size + 64).min(past + n) } else { past + n };
            kern::attn_fa_splits(n, nh, keys)
        } else if n == 1 {
            span.div_ceil(64).clamp(1, 32)
        } else {
            1
        };
        let parts = if splits > 1 { Some(empty(vec![n * nh * splits * 130], DType::F32, dev)?) } else { None };
        if let Some(p) = &parts {
            a.a[8] = addr(p)?;
        }
        a.u[9] = splits as u32;
        kern::stage("k_attn", dev, || -> R<()> {
            if fa {
                launch(dev, "ink2_attn_fa", &a, ((n as u32).div_ceil(64), nh as u32, splits as u32), 128)?;
            } else {
                launch(dev, "ink2_attn", &a, (n as u32, nh as u32, splits as u32), 128)?;
            }
            if parts.is_some() {
                launch(dev, "ink2_attn_merge", &a, (n as u32, nh as u32, 1), 128)?;
            }
            Ok(())
        })?;
        let o = kern::stage("proj_o", dev, || l.wo.raw(&att))?;
        // Первый слой блока начинает новый поток; иначе остаток прибавляется в той же свёртке.
        let first = ip % t.n_layers_per_attn_res_block == 0;
        let job = kern::ConvJob { x: kern::cols(&o, 0)?, w: &l.attn_post_conv, state: &sc.attn_state, dynamic: None };
        let cur = kern::conv(&[job], n, t.hidden_size, (!first).then_some(&inp_sa), k_scale, dev)?.pop().expect("свёртка");
        if first {
            blocks.push(inp_sa.clone());
        }
        let ffn_inp = cur.clone();
        let mut states: Vec<&Tensor> = blocks.iter().collect();
        states.push(&cur);
        let mut m = kern::mix(&states, &l.ffn_res_q)?;
        if let Some(msg) = messages {
            let lp = l.lp.as_ref().expect("контроллеры");
            m = self.control(&m, 0, msg, lp, 3, 1)?;
        }
        let m = kern::pick(&m, 0)?;
        let xn = kern::rms(&m, Some(&l.ffn_norm), eps, true)?;
        let y = kern::stage("moe", dev, || self.moe(ip, l, &xn))?;
        let sc = &mut cache.slots[slot];
        let job = kern::ConvJob { x: kern::cols(&y, 0)?, w: &l.ffn_post_conv, state: &sc.ffn_state, dynamic: None };
        Ok(kern::conv(&[job], n, t.hidden_size, Some(&ffn_inp), k_scale, dev)?.pop().expect("свёртка"))
    }

    /// MoE: `xn [T, H]` BF16 → `[T, H]` F32. Декод (T = 1) — целиком на карте,
    /// без чтения маршрута хостом; префилл — групповой GEMM по экспертам.
    fn moe(&self, ip: usize, l: &Layer, xn: &Tensor) -> R<Tensor> {
        let t = &self.cfg.text;
        let n = xn.dims()[0];
        let (e, k, hdim, inter) = (t.n_routed_experts, t.num_experts_per_tok, t.hidden_size, t.intermediate_size);
        let dev = self.device;
        let ex = &l.experts;
        let rt = self.xrt.read().unwrap_or_else(|e| e.into_inner());
        let store = &rt.stores[ip];
        let xin = kern::cols(xn, 0)?;
        let logits = empty(vec![n, e], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = xin.addr;
        a.a[1] = addr(&l.router)?;
        a.a[2] = addr(&logits)?;
        a.u[0] = n as u32;
        a.u[1] = hdim as u32;
        a.u[2] = e as u32;
        a.u[3] = xin.bf16;
        launch(dev, "ink2_router", &a, (n as u32, (e as u32).div_ceil(8), 1), 256)?;
        let idx = empty(vec![n * k], DType::U32, dev)?;
        let wts = empty(vec![n * k], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&logits)?;
        a.a[1] = addr(&l.router_bias)?;
        a.a[2] = addr(&idx)?;
        a.a[3] = addr(&wts)?;
        a.a[4] = addr(&ex.alpha_down_dev)?;
        a.u[0] = n as u32;
        a.u[1] = e as u32;
        a.u[2] = k as u32;
        a.f[0] = t.route_scale;
        launch(dev, "ink2_route", &a, (n as u32, 1, 1), 32)?;
        if n == 1 {
            // Декод — та же схема, что у префилла и эталона: выход эксперта в BF16 по парам
            // `[k, H]`, затем `ink2_combine` в фиксированном порядке k = 0..K−1 (F32) с весами
            // маршрута (alpha_down уже в `wts`). Без атомарного накопления — детерминированно.
            let y = match store {
                ExpertStore::Resident { gate_up, down, .. } => {
                    let gu = gate_up.gemv_indexed_dense(&idx, xn, false).map_err(fe)?;
                    let h = self.act(&gu, &ex.alpha_gate_up_dev, Some(&idx), inter)?;
                    down.gemv_indexed_dense(&idx, &h, true).map_err(fe)?
                }
                ExpertStore::Zc(z) => {
                    // Те же ядра по таблицам пар: запись `p` — эксперт `idx[p]` (слот или RAM);
                    // alpha и вес маршрута — по настоящему `idx`.
                    let pool = rt.pool.as_ref().expect("пул zero-copy");
                    let (gu_t, dn_t, iota) = z.stage(pool, &idx)?;
                    let gu = gu_t.gemv_indexed_dense(&iota, xn, false).map_err(fe)?;
                    let h = self.act(&gu, &ex.alpha_gate_up_dev, Some(&idx), inter)?;
                    dn_t.gemv_indexed_dense(&iota, &h, true).map_err(fe)?
                }
                ExpertStore::Pending => return Err(ModelError::Forward("inkling2: эксперты не подняты".into())),
            };
            return self.combine(&y, &wts, None, n, hdim, k);
        }
        let idx_h: Vec<u32> = idx.to_device(Device::Cpu).map_err(fe)?.to_vec1::<u32>().map_err(fe)?;
        // Zero-copy: эксперты, уже лежащие в слотах, считаются первыми — подкачка
        // промахов вытесняет только тех, чей GEMM уже поставлен в поток.
        let resident: Option<Vec<bool>> = match store {
            ExpertStore::Zc(z) => {
                let mut r = vec![false; e];
                for o in z.slot_owners()? {
                    if o >= 0 && (o as usize) < e {
                        r[o as usize] = true;
                    }
                }
                Some(r)
            }
            _ => None,
        };
        let pairs = n * k;
        let mut order: Vec<usize> = (0..pairs).collect();
        match &resident {
            Some(r) => order.sort_by_key(|&p| (!r[idx_h[p] as usize], idx_h[p], p)),
            None => order.sort_by_key(|&p| (idx_h[p], p)),
        }
        let mut rowmap = vec![0u32; pairs];
        let mut rows = Vec::with_capacity(pairs);
        let mut alpha_rows = Vec::with_capacity(pairs);
        let mut segments: Vec<(u32, u32, u32)> = Vec::new();
        for (pos, &p) in order.iter().enumerate() {
            let ei = idx_h[p];
            rowmap[p] = pos as u32;
            rows.push((p / k) as u32);
            alpha_rows.push(ex.alpha_gate_up[ei as usize]);
            match segments.last_mut() {
                Some(s) if s.0 == ei => s.2 = pos as u32 + 1,
                _ => segments.push((ei, pos as u32, pos as u32 + 1)),
            }
        }
        let y = match store {
            ExpertStore::Resident { gate_up, down, .. } => {
                let row_idx = Tensor::from_vec(rows, vec![pairs], dev).map_err(fe)?;
                let at = Tensor::from_vec(alpha_rows, vec![pairs], dev).map_err(fe)?;
                let gu = gate_up.gemm_grouped_dense_rows(xn, Some(&row_idx), &segments, false).map_err(fe)?;
                let h = self.act(&gu, &at, None, inter)?;
                bf16c(&down.gemm_grouped_dense_rows(&h, None, &segments, false).map_err(fe)?)?
            }
            ExpertStore::Zc(z) => {
                let r = resident.as_ref().expect("резиденты слоя");
                let pool = rt.pool.as_ref().expect("пул zero-copy");
                self.moe_zc_prefill(z, pool, xn, &rows, &alpha_rows, &segments, r, inter)?
            }
            ExpertStore::Pending => return Err(ModelError::Forward("inkling2: эксперты не подняты".into())),
        };
        let rm = Tensor::from_vec(rowmap, vec![pairs], dev).map_err(fe)?;
        self.combine(&y, &wts, Some(&rm), n, hdim, k)
    }

    /// Сборка MoE: `out[t] = Σ_k w[t·K+k] · y[rowmap[t·K+k]]` (F32, порядок k = 0..K−1).
    /// `y` — `[P, H]` BF16; `rowmap` = None — тождественный (строка пары `p` = `p`).
    fn combine(&self, y: &Tensor, wts: &Tensor, rowmap: Option<&Tensor>, n: usize, hdim: usize, k: usize) -> R<Tensor> {
        let out = empty(vec![n, hdim], DType::F32, self.device)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(y)?;
        a.a[1] = addr(wts)?;
        a.a[2] = match rowmap { Some(r) => addr(r)?, None => 0 };
        a.a[3] = addr(&out)?;
        a.u[0] = n as u32;
        a.u[1] = hdim as u32;
        a.u[2] = k as u32;
        launch(self.device, "ink2_combine", &a, ((hdim as u32).div_ceil(256), n as u32, 1), 256)?;
        Ok(out)
    }

    /// Префилл MoE по слотам zero-copy: сегменты (уже по порядку «резиденты, затем
    /// промахи») группами не больше числа слотов — группа целиком помещается в
    /// слоты, её промахи копируются в слоты прошлых групп (их GEMM в потоке
    /// раньше). Групповой GEMM — тот же, что у резидентного пути, по таблице группы.
    #[allow(clippy::too_many_arguments)]
    fn moe_zc_prefill(&self, z: &ZcLayer, pool: &ZcPool, xn: &Tensor, row_idx: &[u32], alpha_rows: &[f32], segments: &[(u32, u32, u32)], resident: &[bool], inter: usize) -> R<Tensor> {
        let dev = self.device;
        let cap = z.slots().max(1);
        // Группы: сначала все попадания одной группой (сколько бы их ни было — они
        // уже в слотах), промахи — по `cap`.
        let hits = segments.iter().take_while(|s| resident[s.0 as usize]).count();
        let mut groups: Vec<&[(u32, u32, u32)]> = Vec::new();
        if hits > 0 {
            groups.push(&segments[..hits]);
        }
        groups.extend(segments[hits..].chunks(cap));
        let xn = bf16c(xn)?;
        let mut parts: Vec<Tensor> = Vec::with_capacity(groups.len());
        for g in groups {
            let ids: Vec<u32> = g.iter().map(|s| s.0).collect();
            let idx_g = Tensor::from_vec(ids, vec![g.len()], dev).map_err(fe)?;
            let (gu_t, dn_t, _) = z.stage(pool, &idx_g)?;
            let (r0, r1) = (g[0].1, g[g.len() - 1].2);
            let segs: Vec<(u32, u32, u32)> = g.iter().enumerate().map(|(u, s)| (u as u32, s.1 - r0, s.2 - r0)).collect();
            let (r0u, r1u) = (r0 as usize, r1 as usize);
            let rows = Tensor::from_vec(row_idx[r0u..r1u].to_vec(), vec![r1u - r0u], dev).map_err(fe)?;
            let gu = gu_t.gemm_grouped_dense_rows(&xn, Some(&rows), &segs, false).map_err(fe)?;
            let at = Tensor::from_vec(alpha_rows[r0u..r1u].to_vec(), vec![r1u - r0u], dev).map_err(fe)?;
            let h = self.act(&gu, &at, None, inter)?;
            parts.push(bf16c(&dn_t.gemm_grouped_dense_rows(&h, None, &segs, false).map_err(fe)?)?);
        }
        if parts.len() == 1 {
            return Ok(parts.pop().expect("одна группа"));
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 0).map_err(fe)
    }

    /// Активация экспертов: `gu [R, 2I]`, множитель — `alpha[r]` либо `alpha[idx[r]]` → `[R, I]` BF16.
    fn act(&self, gu: &Tensor, alpha: &Tensor, idx: Option<&Tensor>, inter: usize) -> R<Tensor> {
        let gu = bf16c(gu)?;
        let rows = gu.dims()[0];
        let out = empty(vec![rows, inter], DType::BF16, self.device)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&gu)?;
        a.a[1] = addr(alpha)?;
        a.a[2] = addr(&out)?;
        a.a[3] = match idx { Some(i) => addr(i)?, None => 0 };
        a.u[0] = rows as u32;
        a.u[1] = inter as u32;
        a.f[0] = self.cfg.text.mlp_gate_softcap;
        a.f[1] = self.cfg.text.mlp_up_softcap;
        launch(self.device, "ink2_act", &a, ((inter as u32).div_ceil(256), rows as u32, 1), 256)?;
        Ok(out)
    }
}

/// Перевести KV полного слота в формат `fmt` с ёмкостью `capacity`: первые
/// `used` позиций переквантуются на карте (`ink2_kv_requant`: BF16 → MXFP8/INT8
/// тем же кодированием, что у записи; квант → квант того же формата — копией).
/// K, V и масштабы — по очереди: прежний тензор отдаётся сразу после своего.
fn requant_slot(s: &mut SlotCache, fmt: KvFormat, used: usize, capacity: usize, dev: Device) -> R<()> {
    let (nkv, cap_old, d) = (s.k.dims()[0], s.k.dims()[1], s.k.dims()[2]);
    let used = used.min(cap_old);
    if s.fmt != KvFormat::Bf16 || fmt == KvFormat::Bf16 {
        return Err(ModelError::Forward(format!("inkling2: перевод KV {} → {} не поддержан", s.fmt.name(), fmt.name())));
    }
    let scales = Tensor::zeros(vec![2, nkv, capacity, d / 32], DType::U8, dev).map_err(fe)?;
    for (j, kv) in [&mut s.k, &mut s.v].into_iter().enumerate() {
        let fresh = Tensor::zeros(vec![nkv, capacity, d], DType::U8, dev).map_err(fe)?;
        if used > 0 {
            let mut a = Ink2Launch::default();
            a.a[0] = addr(kv)?;
            a.a[1] = addr(&fresh)?;
            a.a[2] = addr(&scales)? + (j * nkv * capacity * (d / 32)) as u64;
            a.u[0] = nkv as u32;
            a.u[1] = used as u32;
            a.u[2] = cap_old as u32;
            a.u[3] = capacity as u32;
            a.u[4] = d as u32;
            a.u[5] = fmt.code();
            launch(dev, "ink2_kv_requant", &a, (used as u32, nkv as u32, 1), d as u32)?;
        }
        *kv = fresh;
    }
    s.scales = Some(scales);
    s.fmt = fmt;
    Ok(())
}

/// Дорастить по оси `axis` (ёмкость) тензор `[.., cap, ..]` до `capacity`: первые `used`
/// позиций копируются, остальное — нули (BF16 KV, U8 квант-KV и его масштабы). Копии —
/// сырыми D2D по отрезкам (`narrow`/`cat` по U8 на карте не поддержаны).
fn grow_rows(t: &mut Tensor, axis: usize, used: usize, capacity: usize) -> R<()> {
    let mut shape = t.dims().to_vec();
    let cap_old = shape[axis];
    shape[axis] = capacity;
    let fresh = Tensor::zeros(shape.clone(), t.dtype(), t.device()).map_err(fe)?;
    let used = used.min(cap_old);
    if used > 0 {
        let Device::Cuda(ord) = t.device() else {
            return Err(ModelError::Forward("inkling2: рост кэша — только на CUDA".into()));
        };
        let outer: usize = shape[..axis].iter().product();
        let row = t.dtype().bytes_for_numel(shape[axis + 1..].iter().product());
        let (src, _) = t.cuda_region().map_err(fe)?;
        let (dst, _) = fresh.cuda_region().map_err(fe)?;
        let st = synaptix_core::device::cuda::default_stream(ord).map_err(fe)?;
        for o in 0..outer {
            synaptix_core::device::cuda::dtod_into_region(&st, dst + (o * capacity * row) as u64, src + (o * cap_old * row) as u64, used * row)
                .map_err(fe)?;
        }
    }
    *t = fresh;
    Ok(())
}

/// Перевезти тензор на `dev` (карта ↔ host-RAM); байты переезда, 0 — уже там.
fn move_tensor(t: &mut Tensor, dev: Device) -> R<usize> {
    if t.device() == dev {
        return Ok(0);
    }
    let bytes = t.dtype().bytes_for_numel(t.numel());
    *t = t.to_device(dev).map_err(fe)?;
    Ok(bytes)
}

/// D2D-копия `src` в постоянный буфер `dst` той же формы (на default-стриме).
fn copy_into(dst: &Tensor, src: &Tensor) -> R<()> {
    let src = if src.is_contiguous() { src.clone() } else { src.contiguous().map_err(fe)? };
    let (d, dl) = dst.cuda_region().map_err(fe)?;
    let (s, sl) = src.cuda_region().map_err(fe)?;
    if dl != sl {
        return Err(ModelError::Shape("inkling2: размеры буферов графа".into()));
    }
    let Device::Cuda(ord) = dst.device() else { return Err(ModelError::Forward("не CUDA".into())) };
    let ds = synaptix_core::device::cuda::default_stream(ord).map_err(fe)?;
    synaptix_core::device::cuda::dtod_into_region(&ds, d, s, dl).map_err(fe)
}

/// Записать число в буфер на карте по порядку потока: синхронная копия с хоста
/// не ждёт ядер, которые ещё читают прежнее значение (прошлый чанк префилла).
fn set_u32(dst: &Tensor, v: u32) -> R<()> {
    let src = Tensor::from_vec(vec![v], vec![1], dst.device()).map_err(fe)?;
    copy_into(dst, &src)
}
