//! Декод MoE без синхронизации с хостом (zero-copy).
//!
//! Обычный путь на каждом слое читает выбор роутера на хост, подкачивает
//! промахи и только потом считает: 48 синхронизаций на токен, карта простаивает
//! треть времени. Здесь всё на карте:
//!
//! 1. роутер → top-k → softmax;
//! 2. кэш экспертов слоя — слоты VRAM с политикой «часов» (`zc_lru`): эксперт
//!    либо в слоте, либо его блоб читается из закреплённого зеркала RAM прямо
//!    в слот вытесненного (`zc_copy`, широкие загрузки ~46 ГБ/с — как DMA);
//! 3. индексные GEMV gate_up / down по адресам пар.
//!
//! Хост в шаге не участвует вовсе; затравка слотов — частоты выбора на
//! префилле. Форматы ([`zc_format`]):
//!
//! - одноблобные (IQ3_S-128, SQ, типы ggml): блоб эксперта в зеркале байт в
//!   байт тот, что читает ядро; слот — `[gate_up | down]`;
//! - с раздельными масштабами (NVFP4, MXFP8): два блоба на матрицу, слот —
//!   `[gu_packed | gu_scales | dn_packed | dn_scales]`. NVFP4 читается в
//!   натуральной раскладке packed (зеркало — [`crate::expert_mirror::mirror_ranges_natural`],
//!   не перемешанное), масштабы — тайловые E4M3; MXFP8 — `[N, K]` + E8M0
//!   `[N, K/32]`. Таблицы шага несут адреса и packed, и масштабов
//!   ([`ExpertTable::from_tensors_dense`]), GEMV — ядра `GEMV_SYN`.
//!
//! # Подключение из своей модели (без [`crate::moe::MoeFfn`])
//!
//! ```ignore
//! use synaptix_llm_common::moe_zc::{ExpertBlobs, ZcLayer, ZcPool};
//! // 1. Пул на модель: слотов на слой (сколько VRAM отдать кэшу экспертов).
//! let pool = ZcPool::new(device);
//! pool.set_slots_per_layer(64);
//! // 2. Адреса экспертов слоя в закреплённой RAM: из зеркала бандла
//! //    (`BundleExperts::blob_addrs4`, зеркало `mirror_ranges_natural`) или
//! //    из своих pinned-буферов (`ExpertBlobs { gu, gu_s, dn, dn_s }`).
//! let blobs: Vec<ExpertBlobs> = (0..experts).map(|e| src.blob_addrs4(layer, e)).collect::<Option<_>>()?;
//! let zc = ZcLayer::build_from_blobs(&blobs, DType::NVFP4, (2 * inter, hidden), (hidden, inter), &pool, None)?
//!     .expect("слотов/зеркала хватает");
//! // 3. Шаг: свой роутинг даёт idx (U32 [pairs] на карте) → кэш на карте
//! //    обновлён, промахи скопированы → таблицы по ПАРАМ (не по экспертам).
//! let (gu_t, dn_t, iota) = zc.stage(&pool, &idx)?;
//! let gu = gu_t.gemv_indexed_dense(&iota, &x_rows, true)?;   // [pairs, 2·inter]
//! let h = swiglu(&gu);                                       // своя активация
//! let rows = dn_t.gemv_indexed_dense(&iota, &h, true)?;      // [pairs, hidden]
//! // Веса пар, alpha эксперта и т. п. модель применяет сама по своему idx.
//! ```
//!
//! Таблицы `gu_t`/`dn_t` индексируются номером пары (`iota`), а не
//! эксперта: запись `p` — адрес эксперта `idx[p]` (слот VRAM или RAM).
//! Шаг ставится только в default-стрим, хост не синхронизируется — годится
//! для захвата CUDA-графом (адреса таблиц стабильны, пока
//! [`ZC_GENERATION`] не изменилось).

use std::sync::{Arc, Mutex};

use synaptix_core::backend::ZcLaunch;
use synaptix_core::device::cuda::LoaderDone;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::ExpertTable;
use synaptix_core::tensor::Tensor;

use crate::model::ModelError;
use crate::moe::ExpertSource;

fn zerr(what: impl std::fmt::Display) -> ModelError {
    ModelError::Forward(format!("zero-copy: {what}"))
}

fn addr(t: &Tensor) -> Result<u64, ModelError> {
    t.device_address().ok_or_else(|| zerr("тензор не на карте"))
}

/// Формат годится для zero-copy: одноблобный или с раздельными масштабами
/// (NVFP4, MXFP8).
pub fn zc_format(dtype: DType) -> bool {
    dtype.is_block_quant() || matches!(dtype, DType::NVFP4 | DType::MXFP8)
}

/// Адреса блобов эксперта в закреплённой RAM (единая адресация: ядра читают
/// их напрямую). `gu_s`/`dn_s` — масштабы у NVFP4/MXFP8; у одноблобных
/// форматов 0 (не читаются). Все адреса выровнены на 16.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertBlobs {
    pub gu: u64,
    pub gu_s: u64,
    pub dn: u64,
    pub dn_s: u64,
}

impl ExpertBlobs {
    /// Адреса по срезам mmap, покрытым закреплённым зеркалом
    /// (`mirror_addr`). `None` — какой-то срез не в зеркале (или зеркало
    /// ещё не догнало его, или хранит его перемешанным).
    pub fn from_mirror(gu: &[u8], gu_s: Option<&[u8]>, dn: &[u8], dn_s: Option<&[u8]>) -> Option<Self> {
        use synaptix_core::device::cuda::mirror_addr;
        let opt = |b: Option<&[u8]>| match b {
            Some(b) => mirror_addr(b),
            None => Some(0),
        };
        Some(Self { gu: mirror_addr(gu)?, gu_s: opt(gu_s)?, dn: mirror_addr(dn)?, dn_s: opt(dn_s)? })
    }
}

/// Общее для слоёв: размер пула слотов и тождественные индексы.
/// Поколение слотов zero-copy: растёт при каждой сборке и роспуске слоя.
/// Захваченные CUDA-графы шага декода помнят адреса таблиц слоя и годятся,
/// пока поколение то же.
pub static ZC_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub struct ZcPool {
    pub device: Device,
    /// Слотов на слой; 0 — размер ещё не назначен (слои не собираются).
    slots_per_layer: std::sync::atomic::AtomicUsize,
    /// Меньше стольких слотов слой не собирается (по умолчанию 64).
    min_slots: std::sync::atomic::AtomicUsize,
    iota: Mutex<Vec<Option<Tensor>>>,
}

impl ZcPool {
    pub fn new(device: Device) -> Arc<Self> {
        Arc::new(Self { device, slots_per_layer: Default::default(), min_slots: 64.into(), iota: Mutex::new(Vec::new()) })
    }

    /// Порог слотов на слой, ниже которого слой не собирается (по умолчанию
    /// 64: при меньшем числе часы вытесняют почти на каждом шаге).
    pub fn set_min_slots(&self, n: usize) {
        self.min_slots.store(n.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    pub fn slots_per_layer(&self) -> usize {
        self.slots_per_layer.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Назначить размер (до сборки слоёв; 0 — сбросить перед перестройкой).
    pub fn set_slots_per_layer(&self, n: usize) {
        self.slots_per_layer.store(n, std::sync::atomic::Ordering::Relaxed);
    }

    /// Индексы `start..start+len` (U32, на карте), из кэша.
    pub fn iota_range(&self, start: usize, len: usize) -> Result<Tensor, ModelError> {
        static RANGES: std::sync::OnceLock<Mutex<std::collections::HashMap<(usize, usize, usize), Tensor>>> = std::sync::OnceLock::new();
        let key = (self as *const Self as usize, start, len);
        let mut g = RANGES.get_or_init(|| Mutex::new(Default::default())).lock().map_err(|_| zerr("индексы отравлены"))?;
        if let Some(t) = g.get(&key) {
            return Ok(t.clone());
        }
        let t = Tensor::from_vec::<_, u32>((start as u32..(start + len) as u32).collect(), vec![len], self.device).map_err(zerr)?;
        g.insert(key, t.clone());
        Ok(t)
    }

    fn iota(&self, pairs: usize) -> Result<Tensor, ModelError> {
        let mut g = self.iota.lock().map_err(|_| zerr("индексы отравлены"))?;
        if g.len() <= pairs {
            g.resize(pairs + 1, None);
        }
        if let Some(t) = &g[pairs] {
            return Ok(t.clone());
        }
        let t = Tensor::from_vec::<_, u32>((0..pairs as u32).collect(), vec![pairs], self.device).map_err(zerr)?;
        g[pairs] = Some(t.clone());
        Ok(t)
    }
}

/// Состояние слоя — всё на карте.
pub struct ZcLayer {
    tab_gu: Tensor,
    tab_dn: Tensor,
    host_gu: Tensor,
    host_dn: Tensor,
    /// Адреса масштабов в RAM (только NVFP4/MXFP8).
    host_s: Option<(Tensor, Tensor)>,
    owner: Tensor,
    used: Tensor,
    clock: Tensor,
    stats: Tensor,
    _slots: Tensor,
    slot_base: u64,
    gu_shape: (usize, usize),
    dn_shape: (usize, usize),
    dtype: DType,
    gu_bytes: usize,
    dn_bytes: usize,
    /// Масштабы gate_up / down (0 — одноблобный формат).
    gu_s_bytes: usize,
    dn_s_bytes: usize,
    experts: usize,
    n_slots: usize,
    /// Частоты выбора на префилле — затравка следующей сборки.
    seed: Vec<f32>,
    ord: usize,
    /// Скретч упреждающей подкачки (живёт со слоем: его читают ядра второго стрима).
    pf: Mutex<Option<PfScratch>>,
    /// Незавершённая подкачка: событие второго стрима и вход, который она читает.
    pending: Mutex<Option<(LoaderDone, Tensor)>>,
}

struct PfScratch {
    logits: Tensor,
    idx: Tensor,
    wts: Tensor,
    cp: Tensor,
    eff: Tensor,
    rows: usize,
    k: usize,
}

impl Drop for ZcLayer {
    fn drop(&mut self) {
        // Ядра второго стрима могут ещё читать слоты и скретч.
        if self.pending.lock().map(|p| p.is_some()).unwrap_or(true) {
            let _ = synaptix_core::device::cuda::synchronize_all(self.ord);
        }
    }
}

impl ZcLayer {
    /// Собрать слой: самые частые на префилле эксперты — сразу в слоты (DMA
    /// в loader-стриме, default ждёт событие — хост нет), остальные — в RAM.
    /// `None` — зеркало не покрывает какого-то эксперта, размер пула не
    /// назначен или слотов слишком мало.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        source: &dyn ExpertSource,
        layer: usize,
        experts: usize,
        dtype: DType,
        gu_shape: (usize, usize),
        dn_shape: (usize, usize),
        pool: &ZcPool,
        seed_counts: Option<&[f32]>,
    ) -> Result<Option<Self>, ModelError> {
        if !matches!(pool.device, Device::Cuda(_)) || pool.slots_per_layer().min(experts) < pool.min_slots.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(None);
        }
        let mut blobs = Vec::with_capacity(experts);
        for e in 0..experts {
            match source.blob_addrs4(layer, e) {
                Some(b) => blobs.push(b),
                None => return Ok(None),
            }
        }
        Self::build_from_blobs(&blobs, dtype, gu_shape, dn_shape, pool, seed_counts)
    }

    /// Собрать слой по готовым адресам экспертов в закреплённой RAM — для
    /// модели без [`ExpertSource`]. `gu_shape`/`dn_shape` — `(N, K)` матриц
    /// gate_up и down; формат — [`zc_format`]. Самые частые по `seed_counts`
    /// эксперты сразу копируются в слоты. `None` — формат/размеры не годятся
    /// (блобы не кратны 16, адрес не выровнен, нет адреса масштабов у
    /// NVFP4/MXFP8), размер пула не назначен или слотов меньше порога пула.
    pub fn build_from_blobs(
        blobs: &[ExpertBlobs],
        dtype: DType,
        gu_shape: (usize, usize),
        dn_shape: (usize, usize),
        pool: &ZcPool,
        seed_counts: Option<&[f32]>,
    ) -> Result<Option<Self>, ModelError> {
        let Device::Cuda(ord) = pool.device else { return Ok(None) };
        let experts = blobs.len();
        let n_slots = pool.slots_per_layer().min(experts);
        if experts == 0 || n_slots < pool.min_slots.load(std::sync::atomic::Ordering::Relaxed).min(experts) || !zc_format(dtype) {
            return Ok(None);
        }
        let sizes = |shape: (usize, usize)| synaptix_core::quant::matrix_blob_bytes(dtype, shape.0, shape.1);
        let (Some((gu_bytes, gu_s_bytes)), Some((dn_bytes, dn_s_bytes))) = (sizes(gu_shape), sizes(dn_shape)) else { return Ok(None) };
        if (gu_bytes | dn_bytes | gu_s_bytes | dn_s_bytes) % 16 != 0 || gu_bytes == 0 || dn_bytes == 0 {
            return Ok(None);
        }
        let split = gu_s_bytes > 0;
        if split != (dn_s_bytes > 0) {
            return Ok(None);
        }
        let ok = |a: u64| a != 0 && a % 16 == 0;
        if !blobs.iter().all(|b| ok(b.gu) && ok(b.dn) && (!split || (ok(b.gu_s) && ok(b.dn_s)))) {
            return Ok(None);
        }
        let slot_bytes = gu_bytes + gu_s_bytes + dn_bytes + dn_s_bytes;
        let dn_off = gu_bytes + gu_s_bytes;
        let slots = Tensor::zeros(vec![n_slots * slot_bytes], DType::U8, pool.device).map_err(zerr)?;
        let slot_base = addr(&slots)?;
        // Затравка: частые — в слоты.
        let seed = seed_counts.filter(|c| c.len() == experts).map(|c| c.to_vec()).unwrap_or_else(|| vec![0.0; experts]);
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|a, b| seed[*b].total_cmp(&seed[*a]));
        let mut tab_gu: Vec<i64> = blobs.iter().map(|a| (a.gu | 1) as i64).collect();
        let mut tab_dn: Vec<i64> = blobs.iter().map(|a| (a.dn | 1) as i64).collect();
        let mut owner = vec![-1i32; n_slots];
        let mut copies = 0usize;
        // Буфер слотов выделен и обнулён в default-стриме: копии loader-стрима
        // обязаны идти после этого, иначе обнуление сотрёт затравку (гонка
        // проявлялась только на занятой карте).
        synaptix_core::device::cuda::loader_wait_default(ord).map_err(zerr)?;
        for (s, e) in order.into_iter().take(n_slots).enumerate() {
            if seed[e] <= 0.0 {
                break;
            }
            let dst = slot_base + (s * slot_bytes) as u64;
            let b = &blobs[e];
            let cp = |off: usize, src: u64, len: usize| synaptix_core::device::cuda::loader_copy_from_pinned(ord, dst + off as u64, src, len).map_err(zerr);
            cp(0, b.gu, gu_bytes)?;
            cp(dn_off, b.dn, dn_bytes)?;
            if split {
                cp(gu_bytes, b.gu_s, gu_s_bytes)?;
                cp(dn_off + dn_bytes, b.dn_s, dn_s_bytes)?;
            }
            tab_gu[e] = dst as i64;
            tab_dn[e] = (dst + dn_off as u64) as i64;
            owner[s] = e as i32;
            copies += 1;
        }
        if copies > 0 {
            LoaderDone::record(ord).and_then(|d| d.wait_default()).map_err(zerr)?;
        }
        let dev = pool.device;
        let t_i64 = |v: Vec<i64>| Tensor::from_vec::<_, i64>(v, vec![experts], dev).map_err(zerr);
        let host_s = if split {
            Some((t_i64(blobs.iter().map(|a| a.gu_s as i64).collect())?, t_i64(blobs.iter().map(|a| a.dn_s as i64).collect())?))
        } else {
            None
        };
        let me = Self {
            tab_gu: t_i64(tab_gu)?,
            tab_dn: t_i64(tab_dn)?,
            host_gu: t_i64(blobs.iter().map(|a| a.gu as i64).collect())?,
            host_dn: t_i64(blobs.iter().map(|a| a.dn as i64).collect())?,
            host_s,
            owner: Tensor::from_vec::<_, i32>(owner, vec![n_slots], dev).map_err(zerr)?,
            used: Tensor::zeros(vec![n_slots], DType::U8, dev).map_err(zerr)?,
            clock: Tensor::zeros(vec![1], DType::U32, dev).map_err(zerr)?,
            stats: Tensor::zeros(vec![4], DType::U32, dev).map_err(zerr)?,
            _slots: slots,
            slot_base,
            gu_shape,
            dn_shape,
            dtype,
            gu_bytes,
            dn_bytes,
            gu_s_bytes,
            dn_s_bytes,
            experts,
            n_slots,
            seed,
            ord,
            pf: Mutex::new(None),
            pending: Mutex::new(None),
        };
        Ok(Some(me))
    }

    /// Общая часть аргументов ядер: таблицы, слоты, размеры.
    fn launch_base(&self) -> Result<ZcLaunch, ModelError> {
        let (hgs, hds) = match &self.host_s {
            Some((g, d)) => (addr(g)?, addr(d)?),
            None => (0, 0),
        };
        Ok(ZcLaunch {
            tab_gu: addr(&self.tab_gu)?,
            tab_dn: addr(&self.tab_dn)?,
            host_gu: addr(&self.host_gu)?,
            host_dn: addr(&self.host_dn)?,
            owner: addr(&self.owner)?,
            used: addr(&self.used)?,
            clock: addr(&self.clock)?,
            stats: addr(&self.stats)?,
            slot_base: self.slot_base,
            slot_bytes: self.slot_bytes() as u64,
            gu_bytes: self.gu_bytes as u64,
            dn_bytes: self.dn_bytes as u64,
            experts: self.experts as u32,
            slots: self.n_slots as u32,
            host_gu_s: hgs,
            host_dn_s: hds,
            gu_s_bytes: self.gu_s_bytes as u64,
            dn_s_bytes: self.dn_s_bytes as u64,
            ..Default::default()
        })
    }

    /// Масштабы отдельным блобом (NVFP4, MXFP8).
    pub fn split_scales(&self) -> bool {
        self.host_s.is_some()
    }

    /// Байт на слот (packed и масштабы обеих матриц).
    pub fn slot_bytes(&self) -> usize {
        self.gu_bytes + self.gu_s_bytes + self.dn_bytes + self.dn_s_bytes
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Частоты префилла — затравка следующей сборки.
    pub fn counts(&self) -> &[f32] {
        &self.seed
    }

    /// Таблицы по парам шага: кэш на карте обновлён, промахи скопированы в
    /// свои новые слоты.
    pub fn stage(&self, pool: &ZcPool, idx: &Tensor) -> Result<(ExpertTable, ExpertTable, Tensor), ModelError> {
        self.stage_routed(pool, idx, None)
    }

    /// Как [`Self::stage`], но выбор экспертов считается тут же на карте:
    /// `route = (вес роутера f32 [E, H], x f16 [rows, H], wts f32 [rows·k])` —
    /// `idx` и `wts` заполняются ядрами (роутер → top-k → softmax).
    pub fn stage_routed(
        &self,
        pool: &ZcPool,
        idx: &Tensor,
        route: Option<(&Tensor, &Tensor, &Tensor)>,
    ) -> Result<(ExpertTable, ExpertTable, Tensor), ModelError> {
        self.resolve_pending()?;
        let pairs = idx.numel();
        let dev = pool.device;
        let split = self.split_scales();
        let eff_gu = Tensor::zeros(vec![pairs], DType::I64, dev).map_err(zerr)?;
        let eff_dn = Tensor::zeros(vec![pairs], DType::I64, dev).map_err(zerr)?;
        // Масштабы — отдельными тензорами: таблица GEMV берёт хранилище целиком.
        let eff_s = if split {
            Some((Tensor::zeros(vec![pairs], DType::I64, dev).map_err(zerr)?, Tensor::zeros(vec![pairs], DType::I64, dev).map_err(zerr)?))
        } else {
            None
        };
        let cp = Tensor::zeros(vec![if split { 5 } else { 3 }, pairs], DType::I64, dev).map_err(zerr)?;
        let cpa = addr(&cp)?;
        let row = 8 * pairs as u64;
        let mut a = self.launch_base()?;
        a.idx = addr(idx)?;
        a.eff_gu = addr(&eff_gu)?;
        a.eff_dn = addr(&eff_dn)?;
        a.cp_src_gu = cpa;
        a.cp_src_dn = cpa + row;
        a.cp_dst = cpa + 2 * row;
        a.pairs = pairs as u32;
        if let Some((g, d)) = &eff_s {
            a.eff_gu_s = addr(g)?;
            a.eff_dn_s = addr(d)?;
            a.cp_src_gu_s = cpa + 3 * row;
            a.cp_src_dn_s = cpa + 4 * row;
        }
        let mut _logits = None;
        if let Some((router, x, wts)) = route {
            let rows = x.dims()[0];
            let h = x.dims()[1];
            let lg = Tensor::empty_uninit(vec![rows, self.experts], DType::F32, dev).map_err(zerr)?;
            a.router_w = addr(router)?;
            a.x = addr(x)?;
            a.logits = addr(&lg)?;
            a.wts = addr(wts)?;
            a.rows = rows as u32;
            a.k = (pairs / rows.max(1)) as u32;
            a.h = h as u32;
            _logits = Some(lg);
        }
        synaptix_core::tensor::ops::moe_zc_lru(dev, &a).map_err(zerr)?;
        drop(cp);
        let (gu, dn) = match eff_s {
            Some((gs, ds)) => (
                ExpertTable::from_tensors_dense(self.gu_shape.0, self.gu_shape.1, self.dtype, eff_gu, gs),
                ExpertTable::from_tensors_dense(self.dn_shape.0, self.dn_shape.1, self.dtype, eff_dn, ds),
            ),
            None => (
                ExpertTable::from_tensor_dense(self.gu_shape.0, self.gu_shape.1, self.dtype, eff_gu),
                ExpertTable::from_tensor_dense(self.dn_shape.0, self.dn_shape.1, self.dtype, eff_dn),
            ),
        };
        let gu = gu.ok_or_else(|| zerr("таблица gate_up"))?;
        let dn = dn.ok_or_else(|| zerr("таблица down"))?;
        Ok((gu, dn, pool.iota(pairs)?))
    }

    /// Дождаться (на карте, не на хосте) упреждающей подкачки этого слоя:
    /// default-стрим ждёт событие второго стрима.
    fn resolve_pending(&self) -> Result<(), ModelError> {
        let p = self.pending.lock().map_err(|_| zerr("состояние отравлено"))?.take();
        if let Some((done, _x)) = p {
            done.wait_default().map_err(zerr)?;
            // `_x` освобождается здесь — после ожидания в порядке default-стрима.
        }
        Ok(())
    }

    /// Упреждающая подкачка: по входу MoE предыдущего слоя `x` (f16 [rows, H])
    /// роутер ЭТОГО слоя предсказывает top-`kpf`; до `budget` промахов
    /// поднимаются в слоты вторым стримом, пока default считает предыдущий
    /// слой. Шаг слоя дождётся события ([`Self::resolve_pending`]).
    pub fn prefetch(&self, pool: &ZcPool, router: &Tensor, x: &Tensor, kpf: usize, budget: usize) -> Result<(), ModelError> {
        if self.pending.lock().map(|p| p.is_some()).unwrap_or(true) {
            return Ok(());
        }
        let rows = x.dims()[0];
        let h = x.dims()[1];
        let dev = pool.device;
        let mut pf = self.pf.lock().map_err(|_| zerr("скретч отравлен"))?;
        if pf.as_ref().is_none_or(|s| s.rows < rows || s.k != kpf) {
            let r = rows.max(4);
            *pf = Some(PfScratch {
                logits: Tensor::zeros(vec![r, self.experts], DType::F32, dev).map_err(zerr)?,
                idx: Tensor::zeros(vec![r * kpf], DType::U32, dev).map_err(zerr)?,
                wts: Tensor::zeros(vec![r * kpf], DType::F32, dev).map_err(zerr)?,
                cp: Tensor::zeros(vec![if self.split_scales() { 5 } else { 3 }, r * kpf], DType::I64, dev).map_err(zerr)?,
                eff: Tensor::zeros(vec![if self.split_scales() { 4 } else { 2 }, r * kpf], DType::I64, dev).map_err(zerr)?,
                rows: r,
                k: kpf,
            });
        }
        let sc = pf.as_ref().expect("создан выше");
        let pairs = rows * kpf;
        let cpa = addr(&sc.cp)?;
        let cap = (sc.rows * kpf) as u64;
        let mut a = self.launch_base()?;
        a.idx = addr(&sc.idx)?;
        a.eff_gu = addr(&sc.eff)?;
        a.eff_dn = addr(&sc.eff)? + 8 * cap;
        a.cp_src_gu = cpa;
        a.cp_src_dn = cpa + 8 * cap;
        a.cp_dst = cpa + 16 * cap;
        a.pairs = pairs as u32;
        a._pad = 1;
        a.router_w = addr(router)?;
        a.x = addr(x)?;
        a.logits = addr(&sc.logits)?;
        a.wts = addr(&sc.wts)?;
        a.rows = rows as u32;
        a.k = kpf as u32;
        a.h = h as u32;
        a._pad2 = budget as u32;
        if self.split_scales() {
            // Таблицы eff в упреждении не пишутся, но адреса — в свой скретч.
            a.eff_gu_s = addr(&sc.eff)? + 16 * cap;
            a.eff_dn_s = addr(&sc.eff)? + 24 * cap;
            a.cp_src_gu_s = cpa + 24 * cap;
            a.cp_src_dn_s = cpa + 32 * cap;
        }
        // Второй стрим стартует после всего, что уже поставлено в default
        // (вход `x`, прошлые шаги слоя), и считает параллельно дальнейшему.
        synaptix_core::device::cuda::loader_wait_default(self.ord).map_err(zerr)?;
        synaptix_core::tensor::ops::moe_zc_lru(dev, &a).map_err(zerr)?;
        let done = LoaderDone::record(self.ord).map_err(zerr)?;
        *self.pending.lock().map_err(|_| zerr("состояние отравлено"))? = Some((done, x.clone()));
        Ok(())
    }

    /// (попадания, выборы) — синхронное чтение счётчиков, только для логов.
    pub fn stats(&self) -> (u64, u64) {
        self.stats
            .to_device(Device::Cpu)
            .and_then(|t| t.to_vec1::<u32>())
            .map(|v| (v[0] as u64, v[1] as u64))
            .unwrap_or((0, 0))
    }

    pub fn slots(&self) -> usize {
        self.n_slots
    }

    /// Какие эксперты сейчас в слотах (синхронное чтение `owner`, −1 — слот
    /// пуст). Для префилла: группы с попаданиями ставятся вперёд, чтобы
    /// подкачка промахов не вытесняла ещё не посчитанных резидентов.
    pub fn slot_owners(&self) -> Result<Vec<i32>, ModelError> {
        self.owner.to_device(Device::Cpu).and_then(|t| t.to_vec1::<i32>()).map_err(zerr)
    }
}
