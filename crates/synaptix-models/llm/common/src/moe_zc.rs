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
//! префилле. Только одноблобные форматы (IQ3_S-128, SQ, типы ggml): блоб
//! эксперта в зеркале байт в байт тот, что читает ядро.

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

/// Общее для слоёв: размер пула слотов и тождественные индексы.
/// Поколение слотов zero-copy: растёт при каждой сборке и роспуске слоя.
/// Захваченные CUDA-графы шага декода помнят адреса таблиц слоя и годятся,
/// пока поколение то же.
pub static ZC_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub struct ZcPool {
    pub device: Device,
    /// Слотов на слой; 0 — размер ещё не назначен (слои не собираются).
    slots_per_layer: std::sync::atomic::AtomicUsize,
    iota: Mutex<Vec<Option<Tensor>>>,
}

impl ZcPool {
    pub fn new(device: Device) -> Arc<Self> {
        Arc::new(Self { device, slots_per_layer: Default::default(), iota: Mutex::new(Vec::new()) })
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
        let Device::Cuda(ord) = pool.device else { return Ok(None) };
        let n_slots = pool.slots_per_layer().min(experts);
        if n_slots < 64 {
            return Ok(None);
        }
        let mut host = Vec::with_capacity(experts);
        for e in 0..experts {
            match source.blob_addrs(layer, e) {
                Some(a) if a.0 % 16 == 0 && a.1 % 16 == 0 => host.push(a),
                _ => return Ok(None),
            }
        }
        let row = |shape: (usize, usize)| synaptix_core::quant::block_row_bytes(dtype, shape.1).map(|r| r * shape.0);
        let (Some(gu_bytes), Some(dn_bytes)) = (row(gu_shape), row(dn_shape)) else { return Ok(None) };
        if gu_bytes % 16 != 0 || dn_bytes % 16 != 0 {
            return Ok(None);
        }
        let slot_bytes = gu_bytes + dn_bytes;
        let slots = Tensor::zeros(vec![n_slots * slot_bytes], DType::U8, pool.device).map_err(zerr)?;
        let slot_base = addr(&slots)?;
        // Затравка: частые — в слоты.
        let seed = seed_counts.map(|c| c.to_vec()).unwrap_or_else(|| vec![0.0; experts]);
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|a, b| seed[*b].total_cmp(&seed[*a]));
        let mut tab_gu: Vec<i64> = host.iter().map(|a| (a.0 | 1) as i64).collect();
        let mut tab_dn: Vec<i64> = host.iter().map(|a| (a.1 | 1) as i64).collect();
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
            synaptix_core::device::cuda::loader_copy_from_pinned(ord, dst, host[e].0, gu_bytes).map_err(zerr)?;
            synaptix_core::device::cuda::loader_copy_from_pinned(ord, dst + gu_bytes as u64, host[e].1, dn_bytes).map_err(zerr)?;
            tab_gu[e] = dst as i64;
            tab_dn[e] = (dst + gu_bytes as u64) as i64;
            owner[s] = e as i32;
            copies += 1;
        }
        if copies > 0 {
            LoaderDone::record(ord).and_then(|d| d.wait_default()).map_err(zerr)?;
        }
        let dev = pool.device;
        let t_i64 = |v: Vec<i64>| Tensor::from_vec::<_, i64>(v, vec![experts], dev).map_err(zerr);
        let me = Self {
            tab_gu: t_i64(tab_gu)?,
            tab_dn: t_i64(tab_dn)?,
            host_gu: t_i64(host.iter().map(|a| a.0 as i64).collect())?,
            host_dn: t_i64(host.iter().map(|a| a.1 as i64).collect())?,
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
            experts,
            n_slots,
            seed,
            ord,
            pf: Mutex::new(None),
            pending: Mutex::new(None),
        };
        Ok(Some(me))
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
        let eff_gu = Tensor::zeros(vec![pairs], DType::I64, dev).map_err(zerr)?;
        let eff_dn = Tensor::zeros(vec![pairs], DType::I64, dev).map_err(zerr)?;
        let cp = Tensor::zeros(vec![3, pairs], DType::I64, dev).map_err(zerr)?;
        let cpa = addr(&cp)?;
        let a = ZcLaunch {
            idx: addr(idx)?,
            tab_gu: addr(&self.tab_gu)?,
            tab_dn: addr(&self.tab_dn)?,
            host_gu: addr(&self.host_gu)?,
            host_dn: addr(&self.host_dn)?,
            owner: addr(&self.owner)?,
            used: addr(&self.used)?,
            clock: addr(&self.clock)?,
            stats: addr(&self.stats)?,
            eff_gu: addr(&eff_gu)?,
            eff_dn: addr(&eff_dn)?,
            cp_src_gu: cpa,
            cp_src_dn: cpa + 8 * pairs as u64,
            cp_dst: cpa + 16 * pairs as u64,
            slot_base: self.slot_base,
            slot_bytes: (self.gu_bytes + self.dn_bytes) as u64,
            gu_bytes: self.gu_bytes as u64,
            dn_bytes: self.dn_bytes as u64,
            pairs: pairs as u32,
            experts: self.experts as u32,
            slots: self.n_slots as u32,
            _pad: 0,
            ..Default::default()
        };
        let mut a = a;
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
        let gu = ExpertTable::from_tensor_dense(self.gu_shape.0, self.gu_shape.1, self.dtype, eff_gu).ok_or_else(|| zerr("таблица gate_up"))?;
        let dn = ExpertTable::from_tensor_dense(self.dn_shape.0, self.dn_shape.1, self.dtype, eff_dn).ok_or_else(|| zerr("таблица down"))?;
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
                cp: Tensor::zeros(vec![3, r * kpf], DType::I64, dev).map_err(zerr)?,
                eff: Tensor::zeros(vec![2, r * kpf], DType::I64, dev).map_err(zerr)?,
                rows: r,
                k: kpf,
            });
        }
        let sc = pf.as_ref().expect("создан выше");
        let pairs = rows * kpf;
        let cpa = addr(&sc.cp)?;
        let cap = (sc.rows * kpf) as u64;
        let a = ZcLaunch {
            idx: addr(&sc.idx)?,
            tab_gu: addr(&self.tab_gu)?,
            tab_dn: addr(&self.tab_dn)?,
            host_gu: addr(&self.host_gu)?,
            host_dn: addr(&self.host_dn)?,
            owner: addr(&self.owner)?,
            used: addr(&self.used)?,
            clock: addr(&self.clock)?,
            stats: addr(&self.stats)?,
            eff_gu: addr(&sc.eff)?,
            eff_dn: addr(&sc.eff)? + 8 * cap,
            cp_src_gu: cpa,
            cp_src_dn: cpa + 8 * cap,
            cp_dst: cpa + 16 * cap,
            slot_base: self.slot_base,
            slot_bytes: (self.gu_bytes + self.dn_bytes) as u64,
            gu_bytes: self.gu_bytes as u64,
            dn_bytes: self.dn_bytes as u64,
            pairs: pairs as u32,
            experts: self.experts as u32,
            slots: self.n_slots as u32,
            _pad: 1,
            router_w: addr(router)?,
            x: addr(x)?,
            logits: addr(&sc.logits)?,
            wts: addr(&sc.wts)?,
            rows: rows as u32,
            k: kpf as u32,
            h: h as u32,
            _pad2: budget as u32,
        };
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
}
