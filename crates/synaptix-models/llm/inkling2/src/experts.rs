//! Режим памяти Inkling-2 по фактической длине сессии.
//!
//! Эксперты (14,6 ГБ) и KV полных слотов (32 КБ/токен BF16) делят карту. Режим
//! не выбирается раз навсегда при загрузке — он следует за ёмкостью сессии на
//! карте ([`ExpertHub::arrange`]):
//!
//! 1. эксперты на карте и KV BF16, пока помещаются вместе с запасом;
//! 2. иначе эксперты zero-copy (зеркало стопок в закреплённой RAM, на карте —
//!    слоты): слотов столько, сколько помещается после KV и запаса, по мере
//!    роста KV их становится меньше (слои пересобираются, частоты выбора —
//!    затравка);
//! 3. только если и минимума слотов мало — KV полных слотов в MXFP8: уже
//!    записанный BF16 переквантуется на карте, префикс и точка возврата годны.
//!
//! Обратно (эксперты снова на карту, больше слотов) — только при создании
//! новой сессии: рост и возврат припаркованной сессии режим лишь ужимают.
//! Режим экспертов общий для модели, формат KV — у каждой сессии свой.
//! Ручные переключатели: `SYN_INKLING2_EXPERTS=resident|zc`,
//! `SYN_INKLING2_ZC_SLOTS=N`, `SYN_INKLING2_KV=bf16|mxfp8|int8`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_llm_common::expert_mirror::{self as em, ExpertStacks};
use synaptix_llm_common::moe::ExpertSource;
use synaptix_llm_common::moe_zc::{ExpertBlobs, ZcLayer, ZcPool};

use crate::loader::Inkling2Weights;
use crate::model::{self, ExpertStore, KvFormat, SharedExperts};
use crate::pipeline::PipelineError;

const GB: f64 = (1u64 << 30) as f64;

/// Где держать экспертов MoE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertMode {
    /// Все на карте.
    Resident,
    /// В закреплённом зеркале RAM, на карте — кэш слотов (zero-copy).
    Zc,
}

/// Ручной выбор: `SYN_INKLING2_EXPERTS=resident|zc` (иначе — по длине сессии).
pub fn expert_mode_env() -> Option<ExpertMode> {
    match std::env::var("SYN_INKLING2_EXPERTS").ok()?.trim() {
        "resident" | "vram" | "gpu" => Some(ExpertMode::Resident),
        "zc" | "ram" | "zero-copy" => Some(ExpertMode::Zc),
        _ => None,
    }
}

/// Ручной формат KV полных слотов: `SYN_INKLING2_KV=bf16|mxfp8|int8`.
pub fn kv_format_env() -> Option<KvFormat> {
    std::env::var("SYN_INKLING2_KV").ok().as_deref().and_then(KvFormat::parse)
}

pub(crate) fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// Запас VRAM сверх весов и KV: активации чанка префилла, граф декода,
/// фрагментация пулов. `SYN_INKLING2_ACT_RESERVE_MB` (1536).
pub(crate) fn activation_reserve() -> usize {
    env_usize("SYN_INKLING2_ACT_RESERVE_MB").unwrap_or(1536) << 20
}

/// Место соседям по карте в режиме zero-copy (малые модели службы),
/// `SYN_INKLING2_ZC_HEADROOM_MB` (512).
pub(crate) fn zc_headroom() -> usize {
    env_usize("SYN_INKLING2_ZC_HEADROOM_MB").unwrap_or(512) << 20
}

/// Меньше стольких слотов на слой zero-copy не собирается (`SYN_INKLING2_ZC_MIN_SLOTS`, 8).
fn min_slots() -> usize {
    env_usize("SYN_INKLING2_ZC_MIN_SLOTS").unwrap_or(8).max(1)
}

/// Прибавка слотов, ради которой при новой сессии слои пересобираются (иначе
/// режим остаётся прежним — без лишних пересборок).
const UPGRADE_SLOTS: usize = 8;

/// Режим экспертов и формат KV, которые выбрал план.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    mode: ExpertMode,
    /// Слотов на слой (zero-copy).
    slots: usize,
    fmt: KvFormat,
}

/// Сколько памяти нужно сессии: KV полных слотов в формате `fmt` плюс то,
/// что вернётся на карту вместе с ним (кольца SWA при возврате из RAM).
pub(crate) struct KvNeed<'a> {
    /// Байт на карте, которые сессия держит сейчас и которые операция заменит
    /// (KV полных слотов при росте; 0 — новой или припаркованной).
    pub held: usize,
    /// Байт на карте после операции в формате `fmt`.
    pub need: &'a dyn Fn(KvFormat) -> usize,
    /// Формат уже задан (сессия квантована — назад в BF16 не переводится).
    pub fixed: Option<KvFormat>,
    /// Можно поднять режим (эксперты на карту, больше слотов) — только при
    /// новой сессии.
    pub upgrade: bool,
}

struct HubState {
    /// Закреплённое зеркало стопок (поднимается при первой нужде в zero-copy
    /// и остаётся: следующие переходы и подъём экспертов на карту идут из него).
    mirror: Option<synaptix_core::device::cuda::OffloadPinCacheGuard>,
    /// Зеркало поднять не вышло (RAM, чужое зеркало) — zero-copy недоступен.
    mirror_failed: bool,
    /// Режим сейчас; `None` — эксперты не подняты (переход сорвался).
    mode: Option<ExpertMode>,
    slots: usize,
    /// VRAM под экспертов на карте (замер при подъёме).
    resident_bytes: usize,
}

/// Режим экспертов модели: план по памяти и переходы.
pub struct ExpertHub {
    /// Эксперты модели (общие с прямым проходом). Поле первое: слои
    /// zero-copy уходят раньше зеркала, зеркало — раньше mmap бандла.
    xrt: SharedExperts,
    st: Mutex<HubState>,
    device: Device,
    layers: usize,
    path: PathBuf,
    stacks: Option<Arc<ExpertStacks>>,
    /// Формат экспертов по слоям (zero-copy; есть только у бандла).
    dtypes: Vec<DType>,
    gu_shape: (usize, usize),
    dn_shape: (usize, usize),
    experts: usize,
    /// Байт слота zero-copy (gate_up + down) всех слоёв.
    per_slot: usize,
    /// Байт всех стопок (зеркало, оценка VRAM экспертов на карте).
    experts_bytes: usize,
    zc_capable: bool,
}

/// Свободная VRAM после ужатия пулов (освобождённое, но удержанное пулами — тоже свободно).
pub(crate) fn free_vram(device: Device) -> usize {
    match device {
        Device::Cuda(ord) => {
            let _ = synaptix_core::device::cuda::synchronize_all(ord);
            let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(ord);
            synaptix_core::device::cuda::mem_info(ord).map(|(f, _)| f).unwrap_or(0)
        }
        _ => usize::MAX / 4,
    }
}

impl ExpertHub {
    pub(crate) fn new(w: &Inkling2Weights, xrt: SharedExperts, path: PathBuf, device: Device) -> Self {
        let t = &w.config.text;
        let (hidden, inter, ne) = (t.hidden_size, t.intermediate_size, t.n_routed_experts);
        let stacks = w.expert_stacks().map(Arc::new);
        let experts_bytes: usize = stacks.as_deref().map_or(0, |s| em::mirror_ranges_natural(s, device).iter().map(|r| r.bytes.len()).sum());
        let mut per_slot = 0usize;
        let mut dtypes = Vec::with_capacity(t.num_hidden_layers);
        for l in 0..t.num_hidden_layers {
            if let Some(dt) = w.expert_dtype(l) {
                let b = |n, k| synaptix_core::quant::matrix_blob_bytes(dt, n, k).map(|(p, s)| p + s).unwrap_or(0);
                per_slot += b(2 * inter, hidden) + b(hidden, inter);
                dtypes.push(dt);
            }
        }
        let zc_capable = device.is_cuda()
            && dtypes.len() == t.num_hidden_layers
            && stacks.as_deref().is_some_and(|s| em::experts_zc_capable(s));
        Self {
            xrt,
            st: Mutex::new(HubState { mirror: None, mirror_failed: false, mode: None, slots: 0, resident_bytes: experts_bytes }),
            device,
            layers: t.num_hidden_layers,
            path,
            stacks,
            dtypes,
            gu_shape: (2 * inter, hidden),
            dn_shape: (hidden, inter),
            experts: ne,
            per_slot,
            experts_bytes,
            zc_capable,
        }
    }

    /// Режим экспертов сейчас (`None` — не подняты).
    pub fn mode(&self) -> Option<ExpertMode> {
        self.lock().mode
    }

    /// Слотов zero-copy на слой (0 — эксперты на карте).
    pub fn slots(&self) -> usize {
        let g = self.lock();
        if g.mode == Some(ExpertMode::Zc) { g.slots } else { 0 }
    }

    /// Байт всех стопок экспертов.
    pub fn experts_bytes(&self) -> usize {
        self.experts_bytes
    }

    /// Байт слота zero-copy всех слоёв.
    pub fn slot_bytes(&self) -> usize {
        self.per_slot
    }

    /// Сколько VRAM держат эксперты сейчас.
    pub fn device_bytes(&self) -> usize {
        let g = self.lock();
        match g.mode {
            Some(ExpertMode::Resident) => g.resident_bytes,
            Some(ExpertMode::Zc) => g.slots * self.per_slot,
            None => 0,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HubState> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Варианты по порядку предпочтения: формат KV (BF16, затем MXFP8), для
    /// каждого — эксперты на карте, затем zero-copy.
    fn plan(&self, g: &HubState, avail: usize, kv: &KvNeed<'_>) -> Option<Plan> {
        let fmts: Vec<KvFormat> = match kv.fixed.or_else(kv_format_env) {
            Some(f) => vec![f],
            None => vec![KvFormat::Bf16, KvFormat::Mxfp8],
        };
        let forced = expert_mode_env();
        let zc_ok = self.zc_capable && !g.mirror_failed && forced != Some(ExpertMode::Resident);
        let resident_ok = forced != Some(ExpertMode::Zc) && (kv.upgrade || g.mode != Some(ExpertMode::Zc));
        let reserve = activation_reserve();
        // Последний вариант (окно 1M целиком): последний формат без запаса соседям.
        let last = (*fmts.last().expect("формат"), true);
        for (fmt, bare) in fmts.iter().map(|&f| (f, false)).chain(std::iter::once(last)) {
            let need = (kv.need)(fmt);
            if !bare && resident_ok && avail >= g.resident_bytes + need + reserve {
                return Some(Plan { mode: ExpertMode::Resident, slots: 0, fmt });
            }
            if zc_ok {
                let room = avail.saturating_sub(need + reserve + if bare { 0 } else { zc_headroom() });
                let mut slots = (room / self.per_slot.max(1)).min(self.experts);
                if !kv.upgrade && g.mode == Some(ExpertMode::Zc) {
                    slots = slots.min(g.slots);
                }
                if let Some(n) = env_usize("SYN_INKLING2_ZC_SLOTS") {
                    slots = n.min(self.experts);
                }
                if slots >= min_slots().min(self.experts) {
                    return Some(Plan { mode: ExpertMode::Zc, slots, fmt });
                }
            }
        }
        // Ручной режим «на карте» без места: KV — в последнем формате, пусть решает аллокатор.
        if forced == Some(ExpertMode::Resident) || !self.zc_capable {
            return Some(Plan { mode: ExpertMode::Resident, slots: 0, fmt: *fmts.last().expect("формат") });
        }
        None
    }

    /// Привести экспертов к памяти, которую займёт сессия, и выполнить шаг KV
    /// (`kv_step` с выбранным форматом: рост, возврат из RAM, создание).
    /// Порядок: если нынешний режим помещается — только шаг KV. Иначе эксперты
    /// снимаются, шаг KV идёт на освободившейся карте, затем эксперты
    /// поднимаются в новом режиме (слоты — по фактически свободному).
    pub(crate) fn arrange<T>(
        &self,
        kv: KvNeed<'_>,
        kv_step: impl FnOnce(KvFormat) -> Result<T, PipelineError>,
    ) -> Result<T, PipelineError> {
        let mut g = self.lock();
        let held_x = match g.mode {
            Some(ExpertMode::Resident) => g.resident_bytes,
            Some(ExpertMode::Zc) => g.slots * self.per_slot,
            None => 0,
        };
        // Быстрый путь без ужатия пулов (оно стоит им повторного роста): рост или
        // возврат, при котором нынешний режим с нынешним форматом и так помещается.
        if !kv.upgrade && g.mode.is_some() {
            if let (Device::Cuda(ord), Some(f)) = (self.device, kv.fixed.or(Some(KvFormat::Bf16))) {
                let quick = synaptix_core::device::cuda::mem_info(ord).map(|(f, _)| f).unwrap_or(0);
                let extra = if g.mode == Some(ExpertMode::Zc) { zc_headroom() } else { 0 };
                if quick + kv.held >= (kv.need)(f) + activation_reserve() + extra {
                    drop(g);
                    return kv_step(f);
                }
            }
        }
        let free = free_vram(self.device);
        let avail = free + held_x + kv.held;
        let Some(mut plan) = self.plan(&g, avail, &kv) else {
            return Err(PipelineError::Load(format!(
                "inkling2: VRAM мало и для минимума слотов zero-copy: свободно {:.1} ГБ (+ эксперты {:.1} + KV {:.1}), нужно KV {:.1} ГБ (MXFP8)",
                free as f64 / GB,
                held_x as f64 / GB,
                kv.held as f64 / GB,
                (kv.need)(KvFormat::Mxfp8) as f64 / GB
            )));
        };
        // Нынешний режим помещается с форматом плана и план не лучше заметно — оставить.
        let reserve = activation_reserve();
        let need = (kv.need)(plan.fmt);
        let cur_fits = match g.mode {
            Some(ExpertMode::Resident) => avail >= held_x + need + reserve,
            Some(ExpertMode::Zc) => avail >= held_x + need + reserve + zc_headroom(),
            None => false,
        };
        let better = kv.upgrade
            && match (g.mode, plan.mode) {
                (Some(ExpertMode::Zc), ExpertMode::Resident) => true,
                (Some(ExpertMode::Zc), ExpertMode::Zc) => plan.slots >= (g.slots + UPGRADE_SLOTS).min(self.experts) && plan.slots > g.slots,
                _ => false,
            };
        let forced_slots = env_usize("SYN_INKLING2_ZC_SLOTS").map(|n| n.min(self.experts));
        let same = g.mode == Some(plan.mode) && (plan.mode == ExpertMode::Resident || forced_slots.is_some_and(|n| n == g.slots) || plan.slots == g.slots);
        if same || (cur_fits && !better) {
            drop(g);
            return kv_step(plan.fmt);
        }
        // Zero-copy без зеркала: поднять его (из страничного кэша — секунды).
        if plan.mode == ExpertMode::Zc && g.mirror.is_none() {
            if let Err(e) = self.ensure_mirror(&mut g) {
                eprintln!("[inkling2] zero-copy недоступен: {e}");
                g.mirror_failed = true;
                plan = self.plan(&g, avail, &kv).ok_or(e)?;
                if g.mode == Some(plan.mode) && plan.mode == ExpertMode::Resident {
                    drop(g);
                    return kv_step(plan.fmt);
                }
            }
        }
        let t0 = Instant::now();
        let from = (g.mode, g.slots);
        // Снять экспертов, освободить карту.
        let seeds = model::take_experts(&self.xrt, self.device);
        g.mode = None;
        g.slots = 0;
        let freed = free_vram(self.device);
        // Шаг KV на освободившейся карте.
        let out = kv_step(plan.fmt);
        // Поднять экспертов в новом режиме даже после ошибки шага KV: модель не должна остаться без них.
        let raised = self.raise(&mut g, plan, &seeds);
        let after = free_vram(self.device);
        let desc = |m: Option<ExpertMode>, s: usize| match m {
            Some(ExpertMode::Resident) => "эксперты на карте".to_string(),
            Some(ExpertMode::Zc) => format!("zero-copy, {s} слотов"),
            None => "эксперты не подняты".to_string(),
        };
        eprintln!(
            "[inkling2] режим памяти: {} → {}, KV {} ({:.1} ГБ), за {:.2} с; свободно {:.1} → {:.1} ГБ",
            desc(from.0, from.1),
            desc(g.mode, g.slots),
            plan.fmt.name(),
            need as f64 / GB,
            t0.elapsed().as_secs_f64(),
            freed as f64 / GB,
            after as f64 / GB
        );
        raised?;
        out
    }

    /// Поднять экспертов в режиме плана (после `model::take_experts`).
    /// Эксперты на карте не влезли — zero-copy; слотов — сколько свободно
    /// сейчас, не больше плана.
    fn raise(&self, g: &mut HubState, plan: Plan, seeds: &[Option<Vec<f32>>]) -> Result<(), PipelineError> {
        if plan.mode == ExpertMode::Resident {
            let before = free_vram(self.device);
            let res = Inkling2Weights::open(&self.path)
                .map_err(|e| PipelineError::Load(e.to_string()))
                .and_then(|w| model::build_resident(&w, self.device, self.layers).map_err(|e| PipelineError::Load(e.to_string())));
            match res {
                Ok(stores) => {
                    model::set_experts(&self.xrt, stores, None);
                    let after = free_vram(self.device);
                    if before > after {
                        g.resident_bytes = before - after;
                    }
                    g.mode = Some(ExpertMode::Resident);
                    g.slots = 0;
                    return Ok(());
                }
                Err(e) if self.zc_capable && !g.mirror_failed => {
                    eprintln!("[inkling2] эксперты на карту не поднялись ({e}) — zero-copy");
                    let _ = free_vram(self.device);
                    self.ensure_mirror(g)?;
                }
                Err(e) => return Err(e),
            }
        }
        let free = free_vram(self.device);
        let ps = self.per_slot.max(1);
        let mut by_room = free.saturating_sub(activation_reserve() + zc_headroom()) / ps;
        if by_room < min_slots() {
            // План без запаса соседям (окно целиком).
            by_room = free.saturating_sub(activation_reserve()) / ps;
        }
        let cap = if plan.mode == ExpertMode::Zc { plan.slots } else { self.experts };
        let slots = match env_usize("SYN_INKLING2_ZC_SLOTS") {
            Some(n) => n.min(self.experts),
            None => by_room.min(cap).min(self.experts),
        };
        self.build_zc(slots, seeds)?;
        g.mode = Some(ExpertMode::Zc);
        g.slots = slots;
        Ok(())
    }

    /// Собрать слои zero-copy со `slots` слотами (зеркало уже поднято).
    fn build_zc(&self, slots: usize, seeds: &[Option<Vec<f32>>]) -> Result<(), PipelineError> {
        let stacks = self.stacks.as_ref().ok_or_else(|| PipelineError::Load("zero-copy: нужен .syn-бандл".into()))?;
        let pool = ZcPool::new(self.device);
        pool.set_min_slots(min_slots());
        pool.set_slots_per_layer(slots);
        let src = em::BundleExperts::new(stacks.clone());
        let mut stores = Vec::with_capacity(self.dtypes.len());
        for (l, &dt) in self.dtypes.iter().enumerate() {
            let blobs: Vec<ExpertBlobs> = (0..self.experts)
                .map(|e| src.blob_addrs4(l, e))
                .collect::<Option<_>>()
                .ok_or_else(|| PipelineError::Load(format!("zero-copy: слой {l}: эксперт не в зеркале")))?;
            let seed = seeds.get(l).and_then(|s| s.as_deref());
            let z = ZcLayer::build_from_blobs(&blobs, dt, self.gu_shape, self.dn_shape, &pool, seed)?
                .ok_or_else(|| PipelineError::Load(format!("zero-copy: слой {l}: слотов {slots} мало (минимум {})", min_slots())))?;
            stores.push(ExpertStore::Zc(z));
        }
        model::set_experts(&self.xrt, stores, Some(pool));
        Ok(())
    }

    /// Поднять закреплённое зеркало стопок и дождаться его целиком.
    fn ensure_mirror(&self, g: &mut HubState) -> Result<(), PipelineError> {
        if g.mirror.is_some() {
            return Ok(());
        }
        if !self.zc_capable {
            return Err(PipelineError::Load("zero-copy экспертов: нужен .syn-бандл NVFP4/MXFP8 и CUDA".into()));
        }
        let stacks = self.stacks.as_ref().expect("zc_capable");
        let ranges = em::mirror_ranges_natural(&**stacks, self.device);
        let (mirror, budget) =
            em::host_mirror(&ranges, "inkling2").ok_or_else(|| PipelineError::Load("zero-copy: зеркало экспертов в RAM не поднято".into()))?;
        if budget < self.experts_bytes {
            return Err(PipelineError::Load(format!(
                "zero-copy: RAM под зеркало {:.1} ГБ < стопок {:.1} ГБ (SYN_HOST_MIRROR_GB / SYN_MIRROR_RESERVE_GB)",
                budget as f64 / GB,
                self.experts_bytes as f64 / GB
            )));
        }
        mirror.resume();
        wait_mirror()?;
        g.mirror = Some(mirror);
        Ok(())
    }

    /// Поднять экспертов при загрузке под первую сессию `need` (байт KV по
    /// формату): план как у новой сессии.
    pub(crate) fn initial(&self, need: &dyn Fn(KvFormat) -> usize) -> Result<KvFormat, PipelineError> {
        self.arrange(KvNeed { held: 0, need, fixed: None, upgrade: true }, Ok)
    }

    /// Начать заполнять зеркало в фоне (`SYN_INKLING2_MIRROR=eager`): первый
    /// переход в zero-copy тогда не ждёт копирования.
    pub(crate) fn prefetch_mirror(&self) {
        let mut g = self.lock();
        if g.mirror.is_some() || !self.zc_capable {
            return;
        }
        let stacks = self.stacks.as_ref().expect("zc_capable");
        let ranges = em::mirror_ranges_natural(&**stacks, self.device);
        if let Some((mirror, budget)) = em::host_mirror(&ranges, "inkling2") {
            if budget >= self.experts_bytes {
                mirror.resume();
                g.mirror = Some(mirror);
            }
        }
    }
}

/// Дождаться, пока зеркало экспертов заполнится целиком: слои zero-copy
/// собираются по адресам в нём.
fn wait_mirror() -> Result<(), PipelineError> {
    let t0 = Instant::now();
    let mut last = 0usize;
    loop {
        let Some((ready, total)) = synaptix_core::device::cuda::offload_pin_cache_progress() else {
            return Err(PipelineError::Load("zero-copy: зеркало пропало".into()));
        };
        if ready >= total {
            eprintln!("[inkling2] зеркало экспертов: {:.1} ГБ за {:.1} с", total as f64 / GB, t0.elapsed().as_secs_f64());
            return Ok(());
        }
        if t0.elapsed().as_secs() > 900 {
            return Err(PipelineError::Load(format!("zero-copy: зеркало {:.1}/{:.1} ГБ за 15 мин", ready as f64 / GB, total as f64 / GB)));
        }
        if ready >= last + (2usize << 30) {
            eprintln!("[inkling2] зеркало экспертов: {:.1}/{:.1} ГБ", ready as f64 / GB, total as f64 / GB);
            last = ready;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}
