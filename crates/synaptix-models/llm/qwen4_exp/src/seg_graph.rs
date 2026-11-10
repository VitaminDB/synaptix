//! CUDA-графы шага декода: подряд идущие слои линейного внимания (HC + GDN +
//! MoE zero-copy) от токена к токену запускают одни и те же ядра с одними и
//! теми же адресами — состояние, таблицы слотов и «часы» кэша экспертов живут
//! на карте, хост в шаге не участвует. Такой отрезок записывается в граф один
//! раз и дальше проигрывается одним запуском вместо сотен.
//!
//! Слои разреженного внимания (зависят от позиции и длины контекста) и PLE
//! (история токенов на хосте) идут обычным путём между графами.
//!
//! Граф привязан к адресам: состояние GDN слоёв отрезка, буферы снимка пары,
//! поколение слотов zero-copy. Любая их смена (новый ход, уплотнение VRAM,
//! роспуск слотов перед префиллом) — отпечаток не сходится, граф пересобирается.
//!
//! `SYN_QWEN4EXP_GRAPH=1` — включить (пока не проверено замерами — выключено).

use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::ModelError;

/// Сколько обычных проходов отрезка с тем же отпечатком должно пройти до
/// захвата: первые шаги хода достраивают ленивое (слои слотов, перемешанные
/// копии весов, компиляция ядер).
pub const WARM_STEPS: u32 = 3;

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SYN_QWEN4EXP_GRAPH").as_deref() == Ok("1"))
}

/// Счётчики для отладки (`SYN_QWEN4EXP_GRAPH_TRACE=1` печатает их в логе хода).
pub static CAPTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static EAGER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SYN_QWEN4EXP_GRAPH_TRACE").is_ok())
}

pub struct SegGraph {
    pub graph: Arc<cudarc::driver::CudaGraph>,
    pub input: Tensor,
    pub output: Tensor,
}

#[derive(Default)]
pub struct SegSlot {
    /// Отпечаток адресов, под который считались прогрев и граф.
    pub stamp: Vec<u64>,
    pub warm: u32,
    pub graph: Option<SegGraph>,
    /// Захват не удался — больше не пробуем с этим отпечатком.
    pub failed: bool,
}

/// Графы сессии: ключ — (первый слой отрезка, строк в шаге).
#[derive(Default)]
pub struct SegGraphs {
    pub slots: std::collections::HashMap<(usize, usize), SegSlot>,
    /// Буферы снимка состояния GDN на границе пары «токен + драфт», по слоям.
    pub pair: Vec<Option<(Tensor, Tensor)>>,
}

impl SegGraphs {
    pub fn clear(&mut self) {
        self.slots.clear();
        self.pair.clear();
    }

    pub fn graphs(&self) -> usize {
        self.slots.values().filter(|s| s.graph.is_some()).count()
    }
}

/// D2D-копия `src` в постоянный буфер `dst` (та же форма и тип) на
/// default-стриме; уникальности `Arc` у приёмника не требует.
pub fn copy_into_fixed(dst: &Tensor, src: &Tensor) -> Result<(), ModelError> {
    let e = |e: SynaptixError| ModelError::Forward(format!("граф декода: копия в буфер: {e}"));
    if dst.dims() != src.dims() || dst.dtype() != src.dtype() {
        return Err(ModelError::Shape(format!("граф декода: буфер {:?} {:?}, источник {:?} {:?}", dst.dims(), dst.dtype(), src.dims(), src.dtype())));
    }
    let src_c = if src.is_contiguous() { src.clone() } else { src.contiguous().map_err(e)? };
    let (d, dl) = dst.cuda_region().map_err(e)?;
    let (s, sl) = src_c.cuda_region().map_err(e)?;
    if dl != sl {
        return Err(ModelError::Shape("граф декода: размеры буферов".into()));
    }
    if d == s {
        return Ok(());
    }
    let Device::Cuda(ord) = dst.device() else { return Err(ModelError::Forward("граф декода: не CUDA".into())) };
    let ds = synaptix_core::device::cuda::default_stream(ord).map_err(e)?;
    synaptix_core::device::cuda::dtod_into_region(&ds, d, s, dl).map_err(e)
}

pub fn like_fixed(src: &Tensor) -> Result<Tensor, ModelError> {
    Tensor::empty_uninit(src.dims().to_vec(), src.dtype(), src.device()).map_err(|e| ModelError::Forward(format!("граф декода: буфер: {e}")))
}

/// Записать `step` в граф на default-стриме. Под записью ядра НЕ исполняются.
pub fn capture(ord: usize, step: impl FnMut() -> Result<(), ModelError>) -> Result<Arc<cudarc::driver::CudaGraph>, ModelError> {
    let mut step = step;
    let stream = synaptix_core::device::cuda::default_stream(ord).map_err(|e| ModelError::Forward(e.to_string()))?;
    let mut err: Option<ModelError> = None;
    let mut cap = synaptix_infer::graph_capture::GraphCapturer::new(0);
    let r = cap.capture_with(&stream, |_| {
        step().map_err(|e| {
            let msg = e.to_string();
            err = Some(e);
            synaptix_infer::InferError::Other(msg)
        })
    });
    match (r, err) {
        (Ok(g), None) => Ok(g),
        (_, Some(e)) => Err(e),
        (Err(e), None) => Err(ModelError::Forward(format!("граф декода: захват: {e}"))),
    }
}
