use std::sync::{Arc, Mutex};

use once_cell::sync::OnceCell;

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;
use crate::tensor::storage::Storage;

pub struct QuantWeight {
    packed: Mutex<Option<Arc<Storage>>>,
    scales: Arc<Storage>,
    dtype: DType,
    n: usize,
    k: usize,
    device: Device,
    shuffled: OnceCell<Arc<Storage>>,
    /// Вес живёт в кэше экспертов MoE: его перемешанная копия должна лечь в
    /// пул экспертов, а не в default'ный рядом с резидентными весами — иначе
    /// вытеснение эксперта не возвращает драйверу ничего (см.
    /// `synaptix_core::device::cuda::experts_pool`).
    expert_pool: std::sync::atomic::AtomicBool,
}

impl QuantWeight {
    pub fn new(
        packed: Arc<Storage>,
        scales: Arc<Storage>,
        dtype: DType,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        if !dtype.is_quantized() {
            return Err(SynaptixError::Unsupported(
                "QuantWeight: dtype должен быть quantized (NVFP4/MXFP8/...)",
            ));
        }
        if packed.device() != scales.device() {
            return Err(SynaptixError::device_mismatch(
                packed.device(),
                scales.device(),
            ));
        }
        let device = packed.device();
        Ok(Self {
            packed: Mutex::new(Some(packed)),
            scales,
            dtype,
            n,
            k,
            device,
            shuffled: OnceCell::new(),
            expert_pool: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Вес, от которого остались только перемешанные байты: так выглядит
    /// NVFP4 после первого умножения — `shuffled` построен, сырой packed
    /// освобождён. Устройство берём по scales: packed'а больше нет.
    pub fn from_shuffled(
        shuffled: Arc<Storage>,
        scales: Arc<Storage>,
        dtype: DType,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        if !dtype.is_quantized() {
            return Err(SynaptixError::Unsupported(
                "QuantWeight: dtype должен быть quantized (NVFP4/MXFP8/...)",
            ));
        }
        if shuffled.device() != scales.device() {
            return Err(SynaptixError::device_mismatch(
                shuffled.device(),
                scales.device(),
            ));
        }
        let device = scales.device();
        let cell = OnceCell::new();
        let _ = cell.set(shuffled);
        Ok(Self {
            packed: Mutex::new(None),
            scales,
            dtype,
            n,
            k,
            device,
            shuffled: cell,
            expert_pool: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Пометить вес как эксперта MoE — см. поле `expert_pool`.
    /// Адрес весов на устройстве — по масштабам: они есть у любой схемы и
    /// живут рядом с `packed`, в том же slab'е арены экспертов.
    pub fn device_address(&self) -> Option<u64> {
        self.scales.device_address()
    }

    pub fn mark_expert_pool(&self) {
        self.expert_pool.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn is_expert_pool(&self) -> bool {
        self.expert_pool.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Гард пула под аллокации, которые принадлежат этому весу (перемешанная
    /// копия). `None` — вес не из кэша экспертов, пул выбирается как раньше.
    pub fn alloc_guard(&self) -> Option<crate::device::cuda::ExpertsAllocGuard> {
        self.is_expert_pool()
            .then(|| crate::device::cuda::ExpertsAllocGuard::for_device(self.device))
            .flatten()
    }

    pub fn from_tensors(packed: &Tensor, scales: &Tensor, n: usize, k: usize) -> Result<Self> {
        Self::new(
            packed.storage_arc(),
            scales.storage_arc(),
            packed.dtype(),
            n,
            k,
        )
    }

    pub fn packed_arc(&self) -> Option<Arc<Storage>> {
        self.packed.lock().unwrap().clone()
    }

    pub fn release_packed(&self) {
        *self.packed.lock().unwrap() = None;
    }

    pub fn scales(&self) -> &Storage {
        &self.scales
    }

    pub fn embed_gather(&self, ids: &Tensor) -> Result<Tensor> {
        use crate::backend::registry;
        use crate::stream::Stream;
        use crate::tensor::layout::Layout;
        use crate::tensor::shape::Shape;

        if self.dtype != DType::MXFP8 {
            return Err(SynaptixError::Unsupported(
                "QuantWeight::embed_gather: поддержан только MXFP8",
            ));
        }
        let packed = self.packed_arc().ok_or(SynaptixError::Unsupported(
            "QuantWeight::embed_gather: packed освобождён",
        ))?;
        let ids_c = if ids.is_contiguous() { ids.clone() } else { ids.contiguous()? };
        let n = ids_c.numel();
        let out_layout = Layout::contiguous(Shape::new(vec![n, self.k]), DType::F16);
        let backend = registry::backend_for(self.device)?;
        let mut storage =
            backend.alloc_zeros(DType::F16.bytes_for_numel(n * self.k), self.device)?;
        let stream = Stream::default_for(self.device)?;
        backend.embed_gather_mxfp8(
            &packed,
            &self.scales,
            (&ids_c.storage, &ids_c.layout),
            (&mut storage, &out_layout),
            self.n,
            self.k,
            &stream,
        )?;
        Ok(Tensor::from_parts(Arc::new(storage), out_layout))
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Перенос веса на `dev` (host-stream квант-весов: квантуем 1× на GPU →
    /// храним на CPU → стримим обратно по требованию). Байты не меняются →
    /// bit-identical с резидентным квантом.
    ///
    /// Везём ту копию, которая у веса осталась. У NVFP4 первое же умножение
    /// строит перемешанную копию и освобождает сырой packed
    /// ([`Self::release_packed`]) — с этого момента вес это и есть `shuffled`,
    /// и переносить надо его: раскладка перемешивания от устройства не
    /// зависит, а обратный repack из неё не собрать. Пока packed на месте,
    /// едет packed, а перемешанная копия пере-строится на новом устройстве
    /// первым умножением.
    pub fn to_device(&self, dev: Device) -> Result<Self> {
        let same_device = self.device == dev;
        let moved = |s: &Arc<Storage>| -> Result<Arc<Storage>> {
            if same_device {
                Ok(s.clone())
            } else {
                Ok(Arc::new(crate::tensor::conversion::storage_to_device(s, dev)?))
            }
        };
        let scales = moved(&self.scales)?;
        let out = match (self.packed_arc(), self.shuffled.get()) {
            (Some(packed), _) => Self::new(moved(&packed)?, scales, self.dtype, self.n, self.k)?,
            (None, Some(shuffled)) => {
                Self::from_shuffled(moved(shuffled)?, scales, self.dtype, self.n, self.k)?
            }
            (None, None) => {
                return Err(SynaptixError::Unsupported(
                    "QuantWeight::to_device: у веса нет ни packed, ни shuffled",
                ));
            }
        };
        if self.is_expert_pool() {
            out.mark_expert_pool();
        }
        Ok(out)
    }

    /// Посчитать сразу несколько NVFP4-GEMV одним запуском: `out[e]` = `W_e ·
    /// x_e`. Веса обязаны быть одной формы и уже иметь перемешанную копию —
    /// её строит первое обычное умножение, поэтому первый вызов эксперта
    /// идёт привычным путём, а батч подхватывает готовое.
    /// Построить перемешанную копию заранее и освободить исходную. Без этого
    /// её строит первое умножение — и до тех пор вес не годится для батча.
    pub fn ensure_shuffled(&self) -> Result<()> {
        use crate::backend::registry;
        use crate::stream::Stream;

        if self.dtype != DType::NVFP4 {
            return Err(SynaptixError::Unsupported("ensure_shuffled: только NVFP4"));
        }
        if self.shuffled.get().is_some() {
            return Ok(());
        }
        let packed = self
            .packed_arc()
            .ok_or(SynaptixError::Unsupported("ensure_shuffled: packed освобождён"))?;
        let backend = registry::backend_for(self.device)?;
        let stream = Stream::default_for(self.device)?;
        let bytes = DType::NVFP4.bytes_for_numel(self.n * self.k);
        // Перемешанная копия — это вес, а не активация: в общем пуле она дробит
        // free-list, из которого потом нечем выделить крупный буфер. У эксперта
        // MoE пул свой — он вытесняется вместе с ним.
        let _expert_pool = self.alloc_guard();
        let _weights_pool = crate::device::cuda::WeightsAllocGuard::new();
        let mut out = backend.alloc_zeros(bytes, self.device)?;
        backend.nvfp4_repack(&packed, &mut out, self.n, self.k, &stream)?;
        let _ = self.shuffled.set(std::sync::Arc::new(out));
        self.release_packed();
        Ok(())
    }

    pub fn gemv_batched(
        weights: &[&QuantWeight],
        acts: &[(&Tensor, &Tensor)],
        x_rows: &[usize],
    ) -> Result<Tensor> {
        use crate::backend::registry;
        use crate::stream::Stream;
        use crate::tensor::layout::Layout;
        use crate::tensor::shape::Shape;

        if weights.is_empty() || weights.len() != acts.len() || weights.len() != x_rows.len() {
            return Err(SynaptixError::Unsupported("gemv_batched: пустой или неровный батч"));
        }
        let first = weights[0];
        if first.dtype != DType::NVFP4 {
            return Err(SynaptixError::Unsupported("gemv_batched: только NVFP4"));
        }
        let (n, k, device) = (first.n, first.k, first.device);
        let mut w_shuf: Vec<&Storage> = Vec::with_capacity(weights.len());
        let mut w_scales: Vec<&Storage> = Vec::with_capacity(weights.len());
        for w in weights {
            if w.n != n || w.k != k || w.device != device || w.dtype != DType::NVFP4 {
                return Err(SynaptixError::Unsupported("gemv_batched: разнородные веса"));
            }
            let Some(shuf) = w.shuffled() else {
                return Err(SynaptixError::Unsupported("gemv_batched: нет перемешанной копии"));
            };
            w_shuf.push(shuf);
            w_scales.push(&w.scales);
        }
        let x_packed: Vec<&Storage> = acts.iter().map(|(p, _)| &p.storage as &Storage).collect();
        let x_scales: Vec<&Storage> = acts.iter().map(|(_, s)| &s.storage as &Storage).collect();

        let out_layout = Layout::contiguous(Shape::new(vec![weights.len(), n]), DType::F16);
        let backend = registry::backend_for(device)?;
        let mut storage = backend.alloc_zeros(DType::F16.bytes_for_numel(weights.len() * n), device)?;
        let stream = Stream::default_for(device)?;
        backend.nvfp4_gemv_batched(
            &w_shuf,
            &w_scales,
            &x_packed,
            &x_scales,
            x_rows,
            (&mut storage, &out_layout),
            n,
            k,
            &stream,
        )?;
        Ok(Tensor::from_parts(std::sync::Arc::new(storage), out_layout))
    }

    /// Адрес перемешанной копии и её масштабов на карте — из них строится
    /// таблица [`ExpertTable`].
    pub fn shuffled_addr(&self) -> Option<(u64, u64)> {
        let shuf = self.shuffled()?;
        Some((storage_addr(shuf)?, storage_addr(&self.scales)?))
    }

    pub fn shuffled(&self) -> Option<&Storage> {
        self.shuffled.get().map(|s| s.as_ref())
    }

    pub fn shuffled_or_try_init(
        &self,
        init: impl FnOnce() -> Result<Arc<Storage>>,
    ) -> Result<&Storage> {
        self.shuffled.get_or_try_init(init).map(|s| s.as_ref())
    }
}


fn storage_addr(st: &Storage) -> Option<u64> {
    st.device_address()
}

/// Стопка NVFP4-экспертов, адресуемая индексом НА КАРТЕ.
///
/// Таблица адресов перемешанных весов строится один раз (адреса стабильны:
/// копии живут столько же, сколько сама модель), а какой эксперт считать —
/// решает тензор индексов на устройстве. Нужна графовому декоду MoE: под
/// захватом CUDA-графа выбор не может проходить через хост.
pub struct ExpertTable {
    /// I64[E] — адреса перемешанных весов.
    w_table: Tensor,
    /// I64[E] — адреса масштабов весов.
    s_table: Tensor,
    n: usize,
    k: usize,
    count: usize,
    device: Device,
}

impl ExpertTable {
    /// `None` — веса не NVFP4, разной формы либо без перемешанной копии
    /// (её строит `ensure_shuffled` на загрузке).
    pub fn build(weights: &[&QuantWeight]) -> Option<Self> {
        let first = weights.first()?;
        if first.dtype != DType::NVFP4 || !first.device.is_cuda() {
            return None;
        }
        let (n, k, device) = (first.n, first.k, first.device);
        let mut w = Vec::with_capacity(weights.len());
        let mut sc = Vec::with_capacity(weights.len());
        for it in weights {
            if it.n != n || it.k != k || it.device != device || it.dtype != DType::NVFP4 {
                return None;
            }
            let (wa, sa) = it.shuffled_addr()?;
            w.push(wa as i64);
            sc.push(sa as i64);
        }
        let count = weights.len();
        let w_table = Tensor::from_vec::<_, i64>(w, vec![count], device).ok()?;
        let s_table = Tensor::from_vec::<_, i64>(sc, vec![count], device).ok()?;
        Some(Self { w_table, s_table, n, k, count, device })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Один GEMV на пару: `out[p] = W[idx[p]] · x`. Активация — ОДНА строка
    /// (декод), квантованная заранее; `idx` — U32 на устройстве.
    /// `rows_per_pair = false` — все пары читают строку 0 активации (первая
    /// проекция эксперта); `true` — пара `p` читает строку `p` (вторая
    /// проекция, где активация уже своя на пару).
    pub fn gemv_indexed(
        &self,
        idx: &Tensor,
        packed: &Tensor,
        scales: &Tensor,
        rows_per_pair: bool,
    ) -> Result<Tensor> {
        use crate::backend::registry;
        use crate::stream::Stream;
        use crate::tensor::layout::Layout;
        use crate::tensor::shape::Shape;

        if idx.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported("gemv_indexed: idx должен быть U32"));
        }
        let pairs = idx.numel();
        let out_layout = Layout::contiguous(Shape::new(vec![pairs, self.n]), DType::F16);
        let backend = registry::backend_for(self.device)?;
        let mut storage =
            backend.alloc_zeros(DType::F16.bytes_for_numel(pairs * self.n), self.device)?;
        let stream = Stream::default_for(self.device)?;
        let idx_c = if idx.is_contiguous() { idx.clone() } else { idx.contiguous()? };
        backend.nvfp4_gemv_indexed(
            &self.w_table.storage,
            &self.s_table.storage,
            (&idx_c.storage, &idx_c.layout),
            &packed.storage,
            &scales.storage,
            (&mut storage, &out_layout),
            self.n,
            self.k,
            self.count,
            pairs,
            rows_per_pair,
            &stream,
        )?;
        Ok(Tensor::from_parts(std::sync::Arc::new(storage), out_layout))
    }
}
