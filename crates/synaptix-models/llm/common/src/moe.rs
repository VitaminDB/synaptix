//! MoE-FFN в раскладке Qwen (`Qwen2Moe`, `Qwen3Moe`, `Qwen4Exp`).
//!
//! Слой считает то же, что `Qwen2MoeSparseMoeBlock`:
//!
//! ```text
//! logits = x · gateᵀ                              # [T, E]
//! w, idx = top_k(softmax(logits))                 # k экспертов на токен
//! y      = Σ_i w_i · down_i(silu(gate_i(x)) * up_i(x))
//! y     += sigmoid(x · shared_gateᵀ) · shared(x)  # всегда активный эксперт
//! ```
//!
//! Веса экспертов лежат стопками: `experts.gate_up_proj [E, 2I, H]` и
//! `experts.down_proj [E, H, I]` — по одной матрице на эксперта в конвенции
//! `nn.Linear` (`[out, in]`). Из квантованного бандла они приходят готовыми
//! парами `packed`/`scales` ([`WeightSource::quant_stack`]), из плотного —
//! режутся по ведущей оси.
//!
//! Токены не гоняются через эксперта по одному: пары `(токен, эксперт)`
//! сортируются по эксперту, строки собираются `index_select`'ом в одну
//! матрицу, и каждый эксперт получает один GEMM на все свои токены. Обратная
//! перестановка возвращает строки на места, а сумма по оси слотов складывает
//! вклады k экспертов. Всё это — операции на устройстве; на хост уходят
//! только логиты роутера (см. [`MoeFfn::route`]).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;

use crate::model::ModelError;
use crate::weights::{QLinear, WeightSource};

/// Форма и режим MoE-слоя.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub hidden_size: usize,
    /// Ширина одного эксперта (`moe_intermediate_size`).
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    /// Сколько экспертов активируется на токен (`num_experts_per_tok`).
    pub num_experts_per_tok: usize,
    /// Ширина всегда активного эксперта; `0` — его нет.
    pub shared_intermediate_size: usize,
    /// Делить ли веса top-k на их сумму. `true` — как в Qwen3-MoE (веса
    /// токена дают в сумме единицу), `false` — как в Qwen2-MoE (веса берутся
    /// из общей softmax и в сумме меньше единицы).
    pub norm_topk_prob: bool,
    /// Сколько токенов обрабатывать за один проход. Пик памяти — примерно
    /// `chunk · k · H`, поэтому длинный prefill режется на части.
    pub chunk: usize,
    /// Аварийный клапан оффлоада: пара «токен—эксперт» с весом ниже порога
    /// пропускается, если эксперта нет в кэше на устройстве. Экономит перенос
    /// весов ценой небольшой потери массы у последних слотов top-k. `0` —
    /// выключено, и это значение по умолчанию.
    pub skip_below: f32,
}

impl MoeConfig {
    /// Значения по умолчанию для Qwen4Exp (`Qwen3.8-Flash-Next`).
    pub fn qwen4_exp(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            moe_intermediate_size: 640,
            num_experts: 512,
            num_experts_per_tok: 10,
            shared_intermediate_size: 640,
            norm_topk_prob: true,
            chunk: 512,
            skip_below: 0.0,
        }
    }
}

/// В каком виде читать плотные веса: под квантование — F16 (его требуют
/// квант-ядра), иначе сразу в рабочем dtype, чтобы не терять точность на
/// лишнем проходе через половинную.
fn is_oom(e: &ModelError) -> bool {
    let s = e.to_string().to_lowercase();
    s.contains("out of memory") || s.contains("oom")
}

fn to_f16(x: &Tensor) -> Result<Tensor, ModelError> {
    x.to_dtype(DType::F16)
        .map_err(|e| ModelError::Forward(format!("MoE: приведение к F16: {e}")))
}

fn weight_dtype(quant: DType, compute: DType) -> DType {
    if quant.is_quantized() {
        DType::F16
    } else {
        compute
    }
}

pub struct Expert {
    /// `[2I, H]` — первая половина строк gate, вторая up.
    gate_up: QLinear,
    /// `[H, I]`.
    down: QLinear,
}

impl Expert {
    fn bytes(&self) -> usize {
        self.gate_up.bytes() + self.down.bytes()
    }

    /// Slab арены, в котором лежат веса эксперта. `None` — арена выключена
    /// либо вес не квантован (плотный путь адреса не отдаёт); тогда кэш
    /// вытесняет по одному, как раньше.
    fn slab(&self) -> Option<u64> {
        for l in [&self.gate_up, &self.down] {
            let addr = l.quant_weight().and_then(|w| w.device_address());
            if let Some(slab) = addr.and_then(synaptix_core::memory::expert_arena::slab_of) {
                return Some(slab);
            }
        }
        None
    }

    fn to_device(&self, dev: Device) -> Result<Self, ModelError> {
        Ok(Self {
            gate_up: self.gate_up.to_device(dev)?,
            down: self.down.to_device(dev)?,
        })
    }
}

/// Кэш резидентных экспертов, общий для всех MoE-слоёв модели.
///
/// Веса экспертов лежат в системной памяти (их суммарный объём в разы больше
/// VRAM), а на устройство едет только то, что выбрал роутер. Вытеснение —
/// FIFO: MoE обучается с балансировкой нагрузки, поэтому «горячих» экспертов,
/// ради которых стоило бы держать возраст обращения, там нет.
pub struct ExpertCache {
    device: Device,
    capacity_bytes: AtomicUsize,
    /// Потолок ёмкости: то, что задано настройкой. `capacity_bytes` ходит под
    /// ним туда-сюда — вниз на нехватке VRAM, обратно вверх, когда пик хода
    /// пройден.
    ceiling_bytes: AtomicUsize,
    /// Размер последнего поднятого эксперта: подсказка арене, чтобы весь
    /// эксперт лёг в один slab, а не хвостом в соседний.
    last_expert_bytes: AtomicUsize,
    /// Лимит временного набора префилла — ЧАСТЬ общего бюджета
    /// `capacity_bytes`, а не добавка к нему: раньше на карте оказывалось
    /// `capacity + scratch` байт экспертов (12 + 3 ГБ при 24 ГБ VRAM), и
    /// активациям префилла не оставалось ничего.
    scratch_bytes: usize,
    scratch_mode: std::sync::atomic::AtomicBool,
    inner: Mutex<CacheInner>,
    /// Pinned-зеркало host-весов (`SYN_MOE_PINNED=1`, по умолчанию выключено).
    /// Первая отправка эксперта копирует его в закреплённую память — на слое
    /// 512 экспертов это втрое дороже обычной pageable-копии, — зато повторные
    /// отправки того же эксперта идут DMA без staging. Окупается только при
    /// тесном кэше с частым вытеснением, и ценой второй копии каждого
    /// отправленного эксперта в RAM.
    _mirror: Option<synaptix_core::device::cuda::PinMirrorGuard>,
}

/// Резидент кэша с битом обращения: по нему идёт вытеснение «часами» —
/// приближение LRU, которому не нужен ни список, ни полный обход.
struct Resident {
    expert: Arc<Expert>,
    used: bool,
    /// Slab арены — по нему идёт вытеснение: драйверу возвращается только
    /// slab целиком, поэтому выкидывать резидентов надо группой.
    slab: Option<u64>,
}

struct CacheInner {
    map: HashMap<(usize, usize), Resident>,
    order: VecDeque<(usize, usize)>,
    bytes: usize,
    /// Эксперты, поднятые под префилл: он обходит почти всю стопку, и класть
    /// их в основной кэш значит вытеснить всё, что прогрел предыдущий диалог.
    /// Живут до конца префилла, лимит свой.
    scratch: HashMap<(usize, usize), Resident>,
    scratch_order: VecDeque<(usize, usize)>,
    scratch_bytes: usize,
    hits: u64,
    misses: u64,
    skipped: u64,
    fetched: u64,
    fetch_nanos: u64,
    batched: u64,
    unbatched: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Сколько пар «токен—эксперт» отброшено аварийным клапаном
    /// ([`MoeConfig::skip_below`]).
    pub skipped: u64,
    /// Сколько экспертов поднято предзагрузкой чанка.
    pub fetched: u64,
    /// Суммарное время предзагрузки.
    pub fetch_millis: u64,
    /// Сколько шагов декода посчитано батчем и сколько — поэкспертно.
    pub batched: u64,
    pub unbatched: u64,
    pub resident: usize,
    pub bytes: usize,
}

impl ExpertCache {
    pub fn new(device: Device, capacity_bytes: usize) -> Arc<Self> {
        let me = Self::build(device, capacity_bytes);
        // Кэш перечитывается из бандла, поэтому на чужом OOM он обязан
        // подвинуться — регистрируем как отдаваемую память устройства.
        let dynamic: Arc<dyn synaptix_core::memory::reclaim::Reclaimable> = me.clone();
        synaptix_core::memory::reclaim::register(&dynamic);
        me
    }

    fn build(device: Device, capacity_bytes: usize) -> Arc<Self> {
        let pinned = matches!(device, Device::Cuda(_))
            && std::env::var("SYN_MOE_PINNED").map(|v| v.trim() == "1").unwrap_or(false);
        Arc::new(Self {
            device,
            capacity_bytes: AtomicUsize::new(capacity_bytes),
            ceiling_bytes: AtomicUsize::new(capacity_bytes),
            last_expert_bytes: AtomicUsize::new(0),
            scratch_bytes: (capacity_bytes / 4).min(2 << 30),
            scratch_mode: std::sync::atomic::AtomicBool::new(false),
            _mirror: pinned.then(synaptix_core::device::cuda::PinMirrorGuard::new),
            inner: Mutex::new(CacheInner {
                map: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
                scratch: HashMap::new(),
                scratch_order: VecDeque::new(),
                scratch_bytes: 0,
                hits: 0,
                misses: 0,
                skipped: 0,
                fetched: 0,
                fetch_nanos: 0,
                batched: 0,
                unbatched: 0,
            }),
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Сказать арене, что сейчас поднимут эксперта: если в текущем slab'е
    /// столько не осталось, она откроет новый заранее. Иначе хвост эксперта
    /// уехал бы в соседний slab и держал бы его от освобождения.
    fn arena_group(&self) {
        if let Device::Cuda(ord) = self.device {
            let hint = self.last_expert_bytes.load(Ordering::Relaxed);
            synaptix_core::memory::expert_arena::begin_group(ord, hint);
        }
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes.load(Ordering::Relaxed)
    }

    /// Сменить ёмкость и сразу освободить лишнее. Префилл трогает почти всех
    /// экспертов слоя, так что держать под них большой кэш бессмысленно — он
    /// всё равно вытеснится, зато отнимет память у активаций; на декоде,
    /// наоборот, кэш и есть источник скорости.
    pub fn set_capacity(&self, bytes: usize) {
        self.capacity_bytes.store(bytes, Ordering::Relaxed);
        self.trim_to(bytes);
    }

    /// Освободить резидентов, пока занято больше `bytes` (считая набор
    /// префилла — бюджет общий). Освобождённое сразу возвращаем драйверу:
    /// в своём пуле трим отдаёт целые сегменты, ради этого пул и заведён.
    pub fn trim_to(&self, bytes: usize) {
        {
            let Ok(mut inner) = self.inner.lock() else { return };
            if inner.bytes + inner.scratch_bytes <= bytes {
                return;
            }
            while inner.bytes + inner.scratch_bytes > bytes {
                if !inner.evict() {
                    break;
                }
            }
        }
        if let Device::Cuda(ord) = self.device {
            // Пустые slab'ы арены — сначала: они и есть та память, ради
            // которой всё это затевалось. Трим пула добирает остальное
            // (эксперты, не попавшие в арену, и её же отпущенные блоки).
            synaptix_core::memory::expert_arena::release_empty(ord);
            let _ = synaptix_core::device::cuda::synchronize_all(ord);
            let _ = synaptix_core::device::cuda::trim_experts_pool(ord);
        }
    }

    /// Потолок ёмкости (то, что задано настройкой).
    pub fn ceiling_bytes(&self) -> usize {
        self.ceiling_bytes.load(Ordering::Relaxed)
    }

    /// Опустить потолок — например, когда при загрузке видно, что заказанный
    /// кэш не оставляет места ни KV, ни активациям.
    pub fn set_ceiling(&self, bytes: usize) {
        self.ceiling_bytes.store(bytes, Ordering::Relaxed);
        if self.capacity_bytes.load(Ordering::Relaxed) > bytes {
            self.set_capacity(bytes);
        }
    }

    /// Подогнать ёмкость под текущую VRAM, оставив `reserve` байт свободными
    /// под активации и KV предстоящей фазы. Кэш и растёт (после пика хода
    /// потолок возвращается), и ужимается — а `reclaim` на OOM только режет,
    /// иначе первая же нехватка навсегда оставила бы модель без кэша.
    pub fn fit_to_vram(&self, reserve: usize) -> usize {
        let ceiling = self.ceiling_bytes();
        let Device::Cuda(ord) = self.device else {
            return ceiling;
        };
        let Ok((free, _total)) = synaptix_core::device::cuda::mem_info(ord) else {
            return self.capacity_bytes();
        };
        // Слабина своего пула — тоже наша: она уйдёт под следующего эксперта,
        // не занимая у драйвера.
        let slack = synaptix_core::device::cuda::experts_pool_stats(ord)
            .map(|(rsv, used)| rsv.saturating_sub(used) as usize)
            .unwrap_or(0);
        // Держим мы не байты экспертов, а slab'ы арены: в них есть и огрызки,
        // и место под ещё не поднятых. Считать по `used_bytes` значило бы
        // недосчитать своё же и раздуть ёмкость сверх того, что на карте.
        let held = self
            .used_bytes()
            .max(synaptix_core::memory::expert_arena::reserved_bytes());
        let room = held + free + slack;
        let want = room.saturating_sub(reserve).clamp(MIN_CACHE_BYTES, ceiling);
        self.set_capacity(want);
        want
    }

    pub fn stats(&self) -> ExpertCacheStats {
        let inner = self.inner.lock().expect("кэш экспертов отравлен");
        ExpertCacheStats {
            hits: inner.hits,
            misses: inner.misses,
            skipped: inner.skipped,
            fetched: inner.fetched,
            fetch_millis: inner.fetch_nanos / 1_000_000,
            batched: inner.batched,
            unbatched: inner.unbatched,
            resident: inner.map.len() + inner.scratch.len(),
            bytes: inner.bytes + inner.scratch_bytes,
        }
    }

    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("кэш экспертов отравлен");
        inner.map.clear();
        inner.order.clear();
        inner.bytes = 0;
        inner.scratch.clear();
        inner.scratch_order.clear();
        inner.scratch_bytes = 0;
    }

    fn note_batch(&self, ok: bool) {
        if let Ok(mut inner) = self.inner.lock() {
            if ok {
                inner.batched += 1;
            } else {
                inner.unbatched += 1;
            }
        }
    }

    fn note_fetch_time(&self, elapsed: std::time::Duration) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.fetch_nanos += elapsed.as_nanos() as u64;
        }
    }

    fn note_fetched(&self, count: u64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.fetched += count;
        }
    }

    fn note_skipped(&self, count: u64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.skipped += count;
        }
    }

    fn get(&self, key: (usize, usize)) -> Option<Arc<Expert>> {
        let mut inner = self.inner.lock().ok()?;
        let found = match inner.map.get_mut(&key) {
            Some(r) => {
                r.used = true;
                Some(r.expert.clone())
            }
            None => inner.scratch.get(&key).map(|r| r.expert.clone()),
        };
        match found {
            Some(e) => {
                inner.hits += 1;
                Some(e)
            }
            None => {
                inner.misses += 1;
                None
            }
        }
    }

    /// Освободить экспертов, поднятых под префилл. Прогретые остаются.
    pub fn clear_scratch(&self) {
        {
            let Ok(mut inner) = self.inner.lock() else { return };
            inner.scratch.clear();
            inner.scratch_order.clear();
            inner.scratch_bytes = 0;
        }
        // Набор префилла — это сотни экспертов разом; его slab'ы освобождаются
        // целиком, и держать их до следующего трима незачем: как раз сейчас
        // память нужна активациям декода.
        if let Device::Cuda(ord) = self.device {
            synaptix_core::memory::expert_arena::release_empty(ord);
        }
    }

    /// Куда класть поднятых экспертов: `true` — во временный набор префилла.
    pub fn set_scratch_mode(&self, on: bool) {
        self.scratch_mode.store(on, Ordering::Relaxed);
    }

    fn contains(&self, key: (usize, usize)) -> bool {
        self.inner
            .lock()
            .map(|inner| inner.map.contains_key(&key) || inner.scratch.contains_key(&key))
            .unwrap_or(false)
    }

    fn insert(&self, key: (usize, usize), expert: Arc<Expert>) {
        let bytes = expert.bytes();
        let scratch = self.scratch_mode.load(Ordering::Relaxed);
        let Ok(mut inner) = self.inner.lock() else { return };
        if inner.map.contains_key(&key) || inner.scratch.contains_key(&key) {
            return;
        }
        let capacity = self.capacity_bytes.load(Ordering::Relaxed);
        let slab = expert.slab();
        self.last_expert_bytes.store(bytes, Ordering::Relaxed);
        if scratch {
            while inner.scratch_bytes + bytes > self.scratch_bytes {
                let Some(victim) = inner.scratch_order.pop_front() else { break };
                if let Some(old) = inner.scratch.remove(&victim) {
                    inner.scratch_bytes = inner.scratch_bytes.saturating_sub(old.expert.bytes());
                }
            }
            // Набор префилла — часть общего бюджета: под него уступает
            // прогретый кэш, а не свободная VRAM (её ждут активации).
            while inner.bytes + inner.scratch_bytes + bytes > capacity {
                if !inner.evict() {
                    break;
                }
            }
            inner.scratch_bytes += bytes;
            inner.scratch_order.push_back(key);
            inner.scratch.insert(key, Resident { expert, used: false, slab });
            return;
        }
        while inner.bytes + inner.scratch_bytes + bytes > capacity {
            if !inner.evict() {
                break;
            }
        }
        inner.bytes += bytes;
        inner.order.push_back(key);
        inner.map.insert(key, Resident { expert, used: false, slab });
    }

    /// Занято экспертами всего (прогретые + набор префилла).
    pub fn used_bytes(&self) -> usize {
        self.inner.lock().map(|i| i.bytes + i.scratch_bytes).unwrap_or(0)
    }
}

impl synaptix_core::memory::reclaim::Reclaimable for ExpertCache {
    /// Отдать не меньше `want` байт: сперва набор префилла (он перечитается
    /// на следующем чанке), потом прогретые эксперты. Ёмкость опускаем до
    /// того, что осталось, — иначе кэш тут же наберёт обратно ровно те же
    /// гигабайты, и ретрай аллокации снова упрётся в OOM.
    fn reclaim(&self, device: Device, want: usize) -> usize {
        if device != self.device || want == 0 {
            return 0;
        }
        // Нас зовут из аллокатора — возможно, из-под этого же мьютекса.
        let Ok(mut inner) = self.inner.try_lock() else { return 0 };
        let before = inner.bytes + inner.scratch_bytes;
        if !inner.scratch.is_empty() {
            inner.scratch.clear();
            inner.scratch_order.clear();
            inner.scratch_bytes = 0;
        }
        while before.saturating_sub(inner.bytes + inner.scratch_bytes) < want {
            if !inner.evict() {
                break;
            }
        }
        let freed = before.saturating_sub(inner.bytes + inner.scratch_bytes);
        let left = inner.bytes + inner.scratch_bytes;
        drop(inner);
        if freed == 0 {
            return 0;
        }
        self.capacity_bytes.fetch_min(left.max(MIN_CACHE_BYTES), Ordering::Relaxed);
        // Пул отдаёт драйверу только по триму, а звавший нас аллокатор
        // синкает стримы сам — здесь достаточно вернуть сегменты.
        if let Device::Cuda(ord) = device {
            synaptix_core::memory::expert_arena::release_empty(ord);
            let _ = synaptix_core::device::cuda::synchronize_all(ord);
            let _ = synaptix_core::device::cuda::trim_experts_pool(ord);
        }
        freed
    }
}

/// Ниже этого кэш не ужимаем: без резидентных экспертов каждый токен
/// перечитывает с диска все свои десять — генерация встаёт.
const MIN_CACHE_BYTES: usize = 1 << 30;

impl CacheInner {
    /// Шаг «часов»: эксперт, к которому обращались с прошлого круга, получает
    /// второй шанс, остальные уходят. Выбор экспертов сильно неравномерен, и
    /// простая очередь по возрасту вымывала как раз горячих.
    /// Освободить место под нового резидента.
    ///
    /// Когда эксперты живут в арене, вытеснять по одному бессмысленно:
    /// драйверу возвращается только slab целиком, а поштучное вытеснение
    /// оставляет в каждом slab'е жильца (кэш «ужимается» по учёту, а
    /// `cuMemGetInfo` не меняется — см. `expert_arena`). Поэтому с ареной
    /// уходит самый старый slab со всеми своими резидентами, и лишь без неё
    /// работает прежний обход часами.
    fn evict(&mut self) -> bool {
        if synaptix_core::memory::expert_arena::enabled() {
            if self.evict_slab() {
                return true;
            }
            // Слоя арены может не быть (плотные веса, слишком крупный
            // эксперт) — тогда обычный путь всё ещё уместен.
        }
        self.evict_one()
    }

    /// Выкинуть самый старый slab целиком. `false` — вытеснять нечего.
    fn evict_slab(&mut self) -> bool {
        for slab in synaptix_core::memory::expert_arena::slabs_by_age() {
            let mut freed = false;
            let doomed: Vec<(usize, usize)> = self
                .scratch
                .iter()
                .filter(|(_, r)| r.slab == Some(slab))
                .map(|(k, _)| *k)
                .collect();
            for key in doomed {
                if let Some(old) = self.scratch.remove(&key) {
                    self.scratch_bytes = self.scratch_bytes.saturating_sub(old.expert.bytes());
                    freed = true;
                }
            }
            let doomed: Vec<(usize, usize)> = self
                .map
                .iter()
                .filter(|(_, r)| r.slab == Some(slab))
                .map(|(k, _)| *k)
                .collect();
            for key in doomed {
                if let Some(old) = self.map.remove(&key) {
                    self.bytes = self.bytes.saturating_sub(old.expert.bytes());
                    freed = true;
                }
            }
            if freed {
                self.scratch_order.retain(|k| self.scratch.contains_key(k));
                self.order.retain(|k| self.map.contains_key(k));
                return true;
            }
        }
        false
    }

    fn evict_one(&mut self) -> bool {
        // Два круга, а не один: если на первом у ВСЕХ резидентов взведён бит
        // обращения (так бывает сразу после плотного чанка), круг только
        // гасит биты и не вытесняет ничего — а вызывающий по `false` решает,
        // что кэш пуст, и бросает трим на полном кэше.
        for _ in 0..(2 * self.order.len()).max(1) {
            let Some(key) = self.order.pop_front() else { return false };
            match self.map.get_mut(&key) {
                Some(r) if r.used => {
                    r.used = false;
                    self.order.push_back(key);
                }
                Some(_) => {
                    if let Some(old) = self.map.remove(&key) {
                        self.bytes = self.bytes.saturating_sub(old.expert.bytes());
                    }
                    return true;
                }
                None => {}
            }
        }
        false
    }
}

struct SharedExpert {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
    /// `[1, H]` — sigmoid-гейт всего shared-выхода.
    router: Tensor,
}

/// Откуда брать эксперта, которого нет на устройстве. Реализация читает его
/// из хранилища модели (квантованный `.syn` отдаёт срез стопки прямо из
/// mmap), так что в оперативной памяти эксперты не лежат вовсе.
/// Подкачка промахов одного слоя: живёт отдельно от `MoeFfn`, чтобы её можно
/// было отдать фоновому потоку.
struct FetchJob {
    source: Arc<dyn ExpertSource>,
    cache: Arc<ExpertCache>,
    layer_id: usize,
    prepare_batch: bool,
    missing: Vec<u32>,
}

impl FetchJob {
    fn run(self) {
        let started = std::time::Instant::now();
        let fetched: Vec<(u32, Expert)> = self
            .missing
            .par_iter()
            .filter_map(|e| {
                // Пул экспертов, а не общий пул весов: набор кэша живёт и
                // умирает целиком, и только в своём пуле трим после вытеснения
                // реально возвращает память драйверу (см. `experts_pool`).
                let _experts_pool =
                    synaptix_core::device::cuda::ExpertsAllocGuard::for_device(self.cache.device());
                let _staging = synaptix_core::device::cuda::PinnedStageGuard::new();
                let (gate_up, down) =
                    self.source.fetch(self.layer_id, *e as usize, self.cache.device()).ok()?;
                for w in [gate_up.quant_weight(), down.quant_weight()] {
                    if let Some(w) = w {
                        // Метка живёт с весом: перемешанную копию может строить
                        // и первое умножение — она обязана лечь в тот же пул.
                        w.mark_expert_pool();
                    }
                }
                if self.prepare_batch {
                    // Только на декоде: репак держит обе копии веса разом, а на
                    // префилле экспертов поднимается сотнями и пик не влезает.
                    for w in [gate_up.quant_weight(), down.quant_weight()] {
                        if let Some(w) = w {
                            let _ = w.ensure_shuffled();
                        }
                    }
                }
                Some((*e, Expert { gate_up, down }))
            })
            .collect();
        self.cache.note_fetch_time(started.elapsed());
        let loaded = fetched.len() as u64;
        for (expert, weights) in fetched {
            self.cache.insert((self.layer_id, expert as usize), Arc::new(weights));
        }
        self.cache.note_fetched(loaded);
    }
}

fn to_host_f32(t: &Tensor) -> Result<Vec<f32>, ModelError> {
    t.to_device(Device::Cpu)
        .and_then(|x| x.to_dtype(DType::F32))
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .map_err(|e| ModelError::Forward(format!("роутер MoE: выгрузка: {e}")))
}

/// Сбор строк по индексам. `index_select` на карте копирует строку за строкой
/// и на перестановке шестидесяти тысяч строк стоит дороже, чем все умножения
/// экспертов вместе взятые; embed-ядро читает индексы прямо с карты.
fn take_rows(src: &Tensor, ids: &Tensor) -> Result<Tensor, ModelError> {
    match src.embed_gather(ids) {
        Ok(t) => Ok(t),
        Err(synaptix_core::error::SynaptixError::Unsupported(_)) => src
            .index_select(0, ids)
            .map_err(|e| ModelError::Forward(format!("MoE: сбор строк: {e}"))),
        Err(e) => Err(ModelError::Forward(format!("MoE: сбор строк: {e}"))),
    }
}

/// До скольких токенов разом MoE считает батчем GEMV. Спекулятивный шаг
/// приносит пару, а батч — единственный путь, где вес уже лежит перемешанным:
/// GEMM пришлось бы читать освобождённую packed-копию и тянуть эксперта заново.
fn batch_tokens() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SYN_MOE_BATCH_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(4)
            .max(1)
    })
}

pub trait ExpertSource: Send + Sync {
    fn fetch(&self, layer: usize, expert: usize, device: Device)
        -> Result<(QLinear, QLinear), ModelError>;
}

enum ExpertStore {
    Resident(Vec<Expert>),
    Lazy { source: Arc<dyn ExpertSource>, count: usize },
}

impl ExpertStore {
    fn count(&self) -> usize {
        match self {
            ExpertStore::Resident(v) => v.len(),
            ExpertStore::Lazy { count, .. } => *count,
        }
    }
}

pub struct MoeFfn {
    cfg: MoeConfig,
    cache: Option<Arc<ExpertCache>>,
    layer_id: usize,
    /// `[E, H]` в F32: софтмакс роутера считается в полной точности, иначе
    /// на 512 экспертах порядок top-k пляшет от округления.
    router: Tensor,
    experts: ExpertStore,
    shared: Option<SharedExpert>,
    device: Device,
    compute: DType,
}

impl MoeFfn {
    /// Собрать слой из источника весов.
    ///
    /// `prefix` — путь до блока, например `model.layers.0.mlp`. Квантованные
    /// стопки берутся как есть; плотные режутся и, если `quant` — квант-dtype,
    /// квантуются на загрузке ([`QLinear::build`]).
    pub fn load(
        weights: &dyn WeightSource,
        prefix: &str,
        cfg: MoeConfig,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, ModelError> {
        Self::load_inner(weights, prefix, cfg, device, compute, quant, device)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_inner(
        weights: &dyn WeightSource,
        prefix: &str,
        cfg: MoeConfig,
        device: Device,
        compute: DType,
        quant: DType,
        expert_storage: Device,
    ) -> Result<Self, ModelError> {
        let gate_up = Self::load_stack(
            weights,
            &format!("{prefix}.experts.gate_up_proj"),
            cfg.num_experts,
            [2 * cfg.moe_intermediate_size, cfg.hidden_size],
            device,
            compute,
            quant,
            expert_storage,
        )?;
        let down = Self::load_stack(
            weights,
            &format!("{prefix}.experts.down_proj"),
            cfg.num_experts,
            [cfg.hidden_size, cfg.moe_intermediate_size],
            device,
            compute,
            quant,
            expert_storage,
        )?;
        let mut me = Self::load_parts(weights, prefix, cfg, device, compute, quant)?;
        me.experts = ExpertStore::Resident(
            gate_up
                .into_iter()
                .zip(down)
                .map(|(gate_up, down)| Expert { gate_up, down })
                .collect(),
        );
        Ok(me)
    }

    fn load_parts(
        weights: &dyn WeightSource,
        prefix: &str,
        cfg: MoeConfig,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, ModelError> {
        let router = weights
            .tensor(&format!("{prefix}.gate.weight"), device, DType::F32)?;
        if router.dims() != [cfg.num_experts, cfg.hidden_size] {
            return Err(ModelError::Load(format!(
                "{prefix}.gate.weight: форма {:?}, ожидалась [{}, {}]",
                router.dims(),
                cfg.num_experts,
                cfg.hidden_size
            )));
        }

        let shared = if cfg.shared_intermediate_size > 0 {
            let lin = |name: &str| -> Result<QLinear, ModelError> {
                let key = format!("{prefix}.shared_expert.{name}.weight");
                // В квантованном бандле плотной копии нет — берём готовый вес.
                if let Some(prequant) = weights.quant(&key, device) {
                    return Ok(QLinear::Quant(prequant?));
                }
                let w = weights.tensor(&key, device, weight_dtype(quant, compute))?;
                QLinear::build(w, quant, compute)
            };
            Some(SharedExpert {
                gate: lin("gate_proj")?,
                up: lin("up_proj")?,
                down: lin("down_proj")?,
                router: weights.tensor(
                    &format!("{prefix}.shared_expert_gate.weight"),
                    device,
                    DType::F32,
                )?,
            })
        } else {
            None
        };

        Ok(Self {
            cfg,
            cache: None,
            layer_id: 0,
            router,
            experts: ExpertStore::Resident(Vec::new()),
            shared,
            device,
            compute,
        })
    }

    /// Эксперты не материализуются вовсе: роутер и shared expert грузятся как
    /// обычно, а выбранный эксперт читается из [`ExpertSource`] при промахе
    /// кэша. Так модель со 120 миллиардами весов в экспертах не занимает ни
    /// системной памяти, ни памяти карты сверх кэша.
    #[allow(clippy::too_many_arguments)]
    pub fn load_lazy(
        weights: &dyn WeightSource,
        prefix: &str,
        cfg: MoeConfig,
        device: Device,
        compute: DType,
        quant: DType,
        cache: Arc<ExpertCache>,
        layer_id: usize,
        source: Arc<dyn ExpertSource>,
    ) -> Result<Self, ModelError> {
        let count = cfg.num_experts;
        let mut me = Self::load_parts(weights, prefix, cfg, device, compute, quant)?;
        me.experts = ExpertStore::Lazy { source, count };
        me.cache = Some(cache);
        me.layer_id = layer_id;
        Ok(me)
    }

    /// Веса экспертов остаются в системной памяти, а на устройство едет
    /// только выбранное роутером — через общий [`ExpertCache`]. Всё остальное
    /// (роутер, shared expert) резидентно на `device`, как обычно.
    #[allow(clippy::too_many_arguments)]
    pub fn load_offloaded(
        weights: &dyn WeightSource,
        prefix: &str,
        cfg: MoeConfig,
        device: Device,
        compute: DType,
        quant: DType,
        cache: Arc<ExpertCache>,
        layer_id: usize,
    ) -> Result<Self, ModelError> {
        let mut me = Self::load_inner(weights, prefix, cfg, device, compute, quant, Device::Cpu)?;
        me.cache = Some(cache);
        me.layer_id = layer_id;
        Ok(me)
    }

    /// Стопка `[E, N, K]` → по одной матрице на эксперта.
    #[allow(clippy::too_many_arguments)]
    fn load_stack(
        weights: &dyn WeightSource,
        key: &str,
        num_experts: usize,
        want: [usize; 2],
        device: Device,
        compute: DType,
        quant: DType,
        storage: Device,
    ) -> Result<Vec<QLinear>, ModelError> {
        // Квантованный бандл: плотной копии там нет, режем готовые пары.
        if let Some(stack) = weights.quant_stack(key, device) {
            let stack = stack?;
            if stack.len() != num_experts {
                return Err(ModelError::Load(format!(
                    "{key}: {} матриц, а экспертов {num_experts}",
                    stack.len()
                )));
            }
            for (i, w) in stack.iter().enumerate() {
                if (w.n(), w.k()) != (want[0], want[1]) {
                    return Err(ModelError::Load(format!(
                        "{key}: эксперт {i} имеет {}×{}, ожидалось {}×{}",
                        w.n(),
                        w.k(),
                        want[0],
                        want[1]
                    )));
                }
            }
            return stack
                .into_iter()
                .map(|w| {
                    let q = QLinear::Quant(w);
                    if storage == device {
                        Ok(q)
                    } else {
                        q.to_device(storage)
                    }
                })
                .collect();
        }

        // При оффлоаде плотная стопка читается в системную память, а на
        // устройство едет по одной матрице — только чтобы посчитать квант.
        let read_device = if storage == device { device } else { Device::Cpu };
        let dense = weights.tensor(key, read_device, weight_dtype(quant, compute))?;
        if dense.dims() != [num_experts, want[0], want[1]] {
            return Err(ModelError::Load(format!(
                "{key}: форма {:?}, ожидалась [{num_experts}, {}, {}]",
                dense.dims(),
                want[0],
                want[1]
            )));
        }
        (0..num_experts)
            .map(|i| {
                // `narrow` даёт вид со смещением — форму меняем только у
                // уплотнённой копии.
                let w = dense
                    .narrow(0, i, 1)
                    .and_then(|t| t.contiguous())
                    .and_then(|t| t.reshape((want[0], want[1])))
                    .map_err(|e| ModelError::Load(format!("{key}: эксперт {i}: {e}")))?;
                let w = if quant.is_quantized() && w.device() != device {
                    w.to_device(device).map_err(|e| ModelError::Load(e.to_string()))?
                } else {
                    w
                };
                let q = QLinear::build(w, quant, compute)?;
                if storage == device {
                    Ok(q)
                } else {
                    q.to_device(storage)
                }
            })
            .collect()
    }

    pub fn config(&self) -> &MoeConfig {
        &self.cfg
    }

    /// Кого позвать для каждого токена.
    ///
    /// Возвращает по `k` пар `(эксперт, вес)` на токен, плоско: слот `s`
    /// токена `t` лежит в позиции `t * k + s`. Логиты считаются на устройстве,
    /// а выбор top-k — на хосте: сортировка 512 чисел на токен ядром пока не
    /// оформлена, и для decode (`T = 1`) это ровно один D2H на 2 КБ.
    fn route(&self, x: &Tensor) -> Result<(Vec<u32>, Vec<f32>), ModelError> {
        let t = x.dims()[0];
        let e = self.cfg.num_experts;
        let k = self.cfg.num_experts_per_tok;
        let logits = x
            .to_dtype(DType::F32)
            .and_then(|xf| xf.linear(&self.router))
            .map_err(|err| ModelError::Forward(format!("роутер MoE: {err}")))?;

        // Выбор top-k считает карта: на хост тогда уезжают k пар «эксперт,
        // логит» вместо всей строки из сотен логитов. Нормировка без
        // `norm_topk_prob` требует суммы по всем экспертам, поэтому там
        // остаётся прежний путь.
        if self.cfg.norm_topk_prob {
            if let Ok((vals, idx)) = logits.topk_rows(k) {
                let host_vals = to_host_f32(&vals)?;
                let host_idx = idx
                    .to_device(Device::Cpu)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<u32>())
                    .map_err(|err| ModelError::Forward(format!("роутер MoE: индексы: {err}")))?;
                let mut experts = vec![0u32; t * k];
                let mut weights = vec![0f32; t * k];
                for i in 0..t {
                    let row = &host_vals[i * k..(i + 1) * k];
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
                    let denom: f32 = exps.iter().sum();
                    for s in 0..k {
                        experts[i * k + s] = host_idx[i * k + s];
                        weights[i * k + s] = exps[s] / denom;
                    }
                }
                return Ok((experts, weights));
            }
        }

        let logits = to_host_f32(&logits)?;

        let mut experts = vec![0u32; t * k];
        let mut weights = vec![0f32; t * k];
        // Выбор top-k по строкам независим, а строк на префилле тысячи:
        // последовательный проход по сотням логитов на токен стоил четверть
        // времени всей MoE.
        let norm_topk = self.cfg.norm_topk_prob;
        experts
            .par_chunks_mut(k)
            .zip(weights.par_chunks_mut(k))
            .enumerate()
            .for_each(|(i, (experts, weights))| {
                let row = &logits[i * e..(i + 1) * e];
                let mut order: Vec<u32> = (0..e as u32).collect();
                // Полной сортировки не нужно — важны только k первых.
                order.select_nth_unstable_by(k - 1, |a, b| {
                    row[*b as usize]
                        .partial_cmp(&row[*a as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let top = &mut order[..k];
                top.sort_unstable_by(|a, b| {
                    row[*b as usize]
                        .partial_cmp(&row[*a as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // softmax по k выбранным логитам — это и есть общая softmax,
                // поделённая на сумму top-k. Без нормировки делим на сумму по
                // всем экспертам.
                let max = top.iter().map(|j| row[*j as usize]).fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = top.iter().map(|j| (row[*j as usize] - max).exp()).collect();
                let denom: f32 = if norm_topk {
                    exps.iter().sum()
                } else {
                    row.iter().map(|v| (v - max).exp()).sum()
                };
                for (s, (idx, ex)) in top.iter().zip(exps.iter()).enumerate() {
                    experts[s] = *idx;
                    weights[s] = ex / denom;
                }
            });
        Ok((experts, weights))
    }

    /// `x: [T, H]` → `[T, H]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ModelError> {
        if x.rank() != 2 || x.dims()[1] != self.cfg.hidden_size {
            return Err(ModelError::Forward(format!(
                "MoE: вход {:?}, ожидался [T, {}]",
                x.dims(),
                self.cfg.hidden_size
            )));
        }
        let total = x.dims()[0];
        let mut parts: Vec<Tensor> = Vec::new();
        for start in (0..total).step_by(self.cfg.chunk.max(1)) {
            let len = self.cfg.chunk.max(1).min(total - start);
            let chunk = x
                .narrow(0, start, len)
                .and_then(|t| t.contiguous())
                .map_err(|e| ModelError::Forward(format!("MoE: срез токенов: {e}")))?;
            let out = match self.forward_chunk(&chunk) {
                Ok(out) => out,
                Err(e) if is_oom(&e) => {
                    // Кэш экспертов держит память, которой не хватило активациям:
                    // отдаём половину ЗАНЯТОГО (не ёмкости — она константа, и
                    // делить её пополам на полупустом кэше значит не отдать
                    // ничего) и опускаем ёмкость, иначе кэш наберёт обратно
                    // ровно те же гигабайты ещё до конца чанка.
                    let Some(cache) = &self.cache else { return Err(e) };
                    let want = (cache.used_bytes() / 2).max(MIN_CACHE_BYTES);
                    cache.set_capacity(want);
                    self.forward_chunk(&chunk)?
                }
                Err(e) => return Err(e),
            };
            parts.push(out);
        }
        let out = if parts.len() == 1 {
            parts.pop().expect("одна часть")
        } else {
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, 0).map_err(|e| ModelError::Forward(format!("MoE: сборка: {e}")))?
        };
        Ok(out)
    }

    fn forward_chunk(&self, x: &Tensor) -> Result<Tensor, ModelError> {
        use crate::profile::stage;
        let t = x.dims()[0];
        let k = self.cfg.num_experts_per_tok;
        let (experts, weights) = stage("moe:route", || self.route(x))?;

        // Промахи поднимаются одной параллельной пачкой до счёта: батчевый
        // путь берёт экспертов из кэша по одному, и очередь в один поток к
        // страницам бандла стоит дороже самого умножения.
        let batchable = t <= batch_tokens();
        stage("moe:prefetch", || self.prefetch(&experts, &weights, batchable));
        if batchable {
            if let Some(out) = self.forward_pairs_batched(x, &experts, &weights)? {
                return Ok(out);
            }
        }

        // Пары (токен, слот), сгруппированные по эксперту: каждый эксперт
        // получает один GEMM вместо GEMV на токен.
        let mut order: Vec<u32> = (0..(t * k) as u32).collect();
        order.sort_unstable_by_key(|p| (experts[*p as usize], *p));

        let gathered = stage("moe:gather", || -> Result<Tensor, ModelError> {
            let rows: Vec<u32> = order.iter().map(|p| *p / k as u32).collect();
            let row_idx = Tensor::from_vec::<_, u32>(rows, vec![t * k], self.device)
                .map_err(|e| ModelError::Forward(format!("MoE: индексы строк: {e}")))?;
            take_rows(x, &row_idx)
        })?;

        let mut outs: Vec<Tensor> = Vec::new();
        let mut pos = 0usize;
        let mut skipped = 0u64;
        while pos < order.len() {
            let expert = experts[order[pos] as usize];
            let mut end = pos + 1;
            while end < order.len() && experts[order[end] as usize] == expert {
                end += 1;
            }
            // Аварийный клапан: эксперт, которого нет на устройстве, а все его
            // пары в этом чанке весят меньше порога, не стоит переноса.
            if self.skip_expert(expert as usize, &order[pos..end], &weights) {
                skipped += (end - pos) as u64;
                outs.push(
                    Tensor::zeros(vec![end - pos, self.cfg.hidden_size], self.compute, self.device)
                        .map_err(|e| ModelError::Forward(format!("MoE: пропуск эксперта: {e}")))?,
                );
                pos = end;
                continue;
            }
            let slice = stage("moe:slice", || {
                gathered
                    .narrow(0, pos, end - pos)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| ModelError::Forward(format!("MoE: срез эксперта {expert}: {e}")))
            })?;
            let out = stage("moe:expert", || self.expert_forward(expert as usize, &slice))?;
            let out = if out.dtype() == self.compute {
                out
            } else {
                self.to_compute(out)?
            };
            outs.push(out);
            pos = end;
        }
        if skipped > 0 {
            if let Some(cache) = &self.cache {
                cache.note_skipped(skipped);
            }
        }

        let stacked = stage("moe:stack", || -> Result<Tensor, ModelError> {
            let refs: Vec<&Tensor> = outs.iter().collect();
            if refs.len() == 1 {
                return Ok(outs[0].clone());
            }
            Tensor::cat(&refs, 0)
                .map_err(|e| ModelError::Forward(format!("MoE: сборка экспертов: {e}")))
        })?;

        // Вес пары применяется до возврата строк на места.
        let scale: Vec<f32> = order.iter().map(|p| weights[*p as usize]).collect();
        let scale = Tensor::from_vec::<_, f32>(scale, vec![t * k, 1], self.device)
            .and_then(|s| s.to_dtype(stacked.dtype()))
            .map_err(|e| ModelError::Forward(format!("MoE: веса роутера: {e}")))?;
        let scaled = stacked
            .broadcast_mul(&scale)
            .map_err(|e| ModelError::Forward(format!("MoE: взвешивание: {e}")))?;

        // Обратная перестановка: строка, посчитанная на позиции `j`,
        // принадлежит паре `order[j]`.
        let mut inverse = vec![0u32; order.len()];
        for (j, p) in order.iter().enumerate() {
            inverse[*p as usize] = j as u32;
        }
        let inverse = Tensor::from_vec::<_, u32>(inverse, vec![t * k], self.device)
            .map_err(|e| ModelError::Forward(format!("MoE: обратные индексы: {e}")))?;
        let mixed = stage("moe:scatter", || -> Result<Tensor, ModelError> {
            let restored = take_rows(&scaled, &inverse)?;
            restored
                .reshape((t, k, self.cfg.hidden_size))
                .and_then(|m| m.sum([1usize]))
                .map_err(|e| ModelError::Forward(format!("MoE: сумма по экспертам: {e}")))
        })?;
        let mixed = self.to_compute(mixed)?;

        match &self.shared {
            Some(shared) => {
                let s = stage("moe:shared", || self.shared_forward(shared, x))?;
                mixed
                    .add(&s)
                    .map_err(|e| ModelError::Forward(format!("MoE: shared expert: {e}")))
            }
            None => Ok(mixed),
        }
    }

    /// Декод горстки токенов: все пары «токен × выбранный эксперт» считаются
    /// двумя батчевыми GEMV вместо пары умножений на каждого. На слой это
    /// четыре запуска ядер вместо трёх десятков, а именно они, а не
    /// арифметика, и определяют время шага. `None` — путь неприменим (не
    /// NVFP4, нет перемешанной копии весов, неподходящие формы), вызывающий
    /// считает как раньше.
    fn forward_pairs_batched(
        &self,
        x: &Tensor,
        experts: &[u32],
        weights: &[f32],
    ) -> Result<Option<Tensor>, ModelError> {
        if self.cache.is_none() && matches!(self.experts, ExpertStore::Lazy { .. }) {
            return Ok(None);
        }
        let t = x.dims()[0];
        let k = self.cfg.num_experts_per_tok;
        if experts.len() != t * k {
            return Ok(None);
        }

        let pairs: Vec<usize> = (0..experts.len()).collect();
        let Some(mixed) = self.batched_pairs(x, experts, weights, &pairs, t, k)? else {
            return Ok(None);
        };
        let mixed = self.to_compute(mixed)?;

        if let Some(cache) = &self.cache {
            cache.note_batch(true);
        }
        Ok(Some(match &self.shared {
            Some(shared) => {
                let s = self.shared_forward(shared, x)?;
                mixed
                    .add(&s)
                    .map_err(|e| ModelError::Forward(format!("MoE: shared expert: {e}")))?
            }
            None => mixed,
        }))
    }

    /// Сумма по указанным парам «токен × эксперт»: два батчевых GEMV и фьюз
    /// swiglu между ними. `None` — путь неприменим (не NVFP4, нет перемешанной
    /// копии весов, неподходящие формы).
    #[allow(clippy::too_many_arguments)]
    fn batched_pairs(
        &self,
        x: &Tensor,
        experts: &[u32],
        weights: &[f32],
        pairs: &[usize],
        t: usize,
        k: usize,
    ) -> Result<Option<Tensor>, ModelError> {
        let picked: Vec<Arc<Expert>> = {
            let mut out = Vec::with_capacity(pairs.len());
            for p in pairs {
                match &self.cache {
                    Some(cache) => out.push(self.resident_expert(experts[*p] as usize, cache)?),
                    None => return Ok(None),
                }
            }
            out
        };
        // Перемешанная копия нужна батчу; обычно её строит первое умножение,
        // но эксперт мог попасть в кэш и без него — тогда строим здесь, это
        // однократная работа на эксперта.
        let mut ready = true;
        for e in &picked {
            for w in [e.gate_up.quant_weight(), e.down.quant_weight()] {
                match w {
                    Some(w) if w.dtype() == DType::NVFP4 => {
                        if w.shuffled().is_none() && w.ensure_shuffled().is_err() {
                            ready = false;
                        }
                    }
                    _ => ready = false,
                }
            }
        }
        if !ready {
            if let Some(cache) = &self.cache {
                cache.note_batch(false);
            }
            return Ok(None);
        }

        let xf = if x.dtype() == DType::F16 { x.clone() } else { to_f16(x)? };
        let Ok((packed_x, scales_x)) = xf.nvfp4_quantize_act() else {
            return Ok(None);
        };
        let gate_up: Vec<&QuantWeight> =
            picked.iter().map(|e| e.gate_up.quant_weight().unwrap()).collect();
        let acts: Vec<(&Tensor, &Tensor)> = (0..picked.len()).map(|_| (&packed_x, &scales_x)).collect();
        // Пары идут в порядке `токен * k + слот`, поэтому строка активации у
        // пары — её номер, делённый на число экспертов на токен.
        let rows: Vec<usize> = pairs.iter().map(|p| p / k).collect();
        let Ok(gu) = QuantWeight::gemv_batched(&gate_up, &acts, &rows) else {
            return Ok(None);
        };

        let Ok((packed_h, scales_h)) = gu.silu_mul_quant_nvfp4(1.0) else {
            return Ok(None);
        };
        let down: Vec<&QuantWeight> = picked.iter().map(|e| e.down.quant_weight().unwrap()).collect();
        let acts: Vec<(&Tensor, &Tensor)> =
            (0..picked.len()).map(|_| (&packed_h, &scales_h)).collect();
        let rows: Vec<usize> = (0..picked.len()).collect();
        let Ok(parts) = QuantWeight::gemv_batched(&down, &acts, &rows) else {
            return Ok(None);
        };

        let scale: Vec<f32> = pairs.iter().map(|p| weights[*p]).collect();
        let scale = Tensor::from_vec::<_, f32>(scale, vec![pairs.len(), 1], self.device)
            .and_then(|s| s.to_dtype(parts.dtype()))
            .map_err(|e| ModelError::Forward(format!("MoE: веса роутера: {e}")))?;
        let scaled = parts
            .broadcast_mul(&scale)
            .map_err(|e| ModelError::Forward(format!("MoE: взвешивание: {e}")))?;
        // Слагаемые пары ложатся на свой токен: при одном токене это просто
        // сумма всех строк, иначе пары идут полными группами по k.
        let summed = if t == 1 {
            scaled
                .sum([0usize])
                .and_then(|m| m.reshape((1, self.cfg.hidden_size)))
                .map_err(|e| ModelError::Forward(format!("MoE: сумма по экспертам: {e}")))?
        } else {
            scaled
                .reshape((t, k, self.cfg.hidden_size))
                .and_then(|m| m.sum([1usize]))
                .and_then(|m| m.reshape((t, self.cfg.hidden_size)))
                .map_err(|e| ModelError::Forward(format!("MoE: сумма по экспертам: {e}")))?
        };
        Ok(Some(summed))
    }

    /// Поднять на устройство всех экспертов чанка, которых там ещё нет.
    /// Промахи читаются параллельно: у ленивого источника это страничные
    /// промахи mmap, и одна очередь к NVMe заметно медленнее нескольких.
    fn prefetch(&self, experts: &[u32], weights: &[f32], prepare_batch: bool) {
        if let Some(job) = self.fetch_job(experts, weights, prepare_batch) {
            job.run();
        }
    }

    fn fetch_job(&self, experts: &[u32], weights: &[f32], prepare_batch: bool) -> Option<FetchJob> {
        let cache = self.cache.as_ref()?;
        let ExpertStore::Lazy { source, .. } = &self.experts else {
            return None;
        };
        // Клапан отсекает экспертов ещё до подкачки: тянуть с диска того, чьи
        // пары всё равно будут отброшены, незачем.
        let mut wanted: Vec<u32> = experts
            .iter()
            .zip(weights)
            .filter(|(_, w)| **w >= self.cfg.skip_below)
            .map(|(e, _)| *e)
            .collect();
        wanted.sort_unstable();
        wanted.dedup();
        let missing: Vec<u32> = wanted
            .into_iter()
            .filter(|e| !cache.contains((self.layer_id, *e as usize)))
            .collect();
        if missing.is_empty() {
            return None;
        }
        Some(FetchJob {
            source: source.clone(),
            cache: cache.clone(),
            layer_id: self.layer_id,
            prepare_batch,
            missing,
        })
    }

    /// Стоит ли пропустить эксперта: только при включённом оффлоаде, когда его
    /// нет на устройстве и все его пары в чанке весят меньше порога.
    fn skip_expert(&self, expert: usize, pairs: &[u32], weights: &[f32]) -> bool {
        if self.cfg.skip_below <= 0.0 {
            return false;
        }
        let Some(cache) = &self.cache else { return false };
        if pairs.iter().any(|p| weights[*p as usize] >= self.cfg.skip_below) {
            return false;
        }
        !cache.contains((self.layer_id, expert))
    }

    fn expert_forward(&self, expert: usize, x: &Tensor) -> Result<Tensor, ModelError> {
        match (&self.experts, &self.cache) {
            (ExpertStore::Resident(all), None) => {
                let host = all
                    .get(expert)
                    .ok_or_else(|| ModelError::Forward(format!("MoE: нет эксперта {expert}")))?;
                let gu = host.gate_up.forward(x)?;
                self.down_from_gate_up(&host.down, &gu, x.dims()[0])
            }
            (_, Some(cache)) => {
                let resident = self.resident_expert(expert, cache)?;
                let gu = resident.gate_up.forward(x)?;
                self.down_from_gate_up(&resident.down, &gu, x.dims()[0])
            }
            (ExpertStore::Lazy { .. }, None) => Err(ModelError::Forward(
                "MoE: ленивые эксперты без кэша — некуда их класть".into(),
            )),
        }
    }

    fn resident_expert(
        &self,
        expert: usize,
        cache: &Arc<ExpertCache>,
    ) -> Result<Arc<Expert>, ModelError> {
        let key = (self.layer_id, expert);
        if let Some(e) = cache.get(key) {
            return Ok(e);
        }
        // Веса эксперта живут отдельно от активаций: смешанные в одном пуле
        // мелкие веса и крупные буферы префилла дробят free-list, и уже через
        // несколько слоёв не находится непрерывного куска. И отдельно от
        // резидентных весов модели: вытеснение обязано возвращать память.
        let _experts_pool =
            synaptix_core::device::cuda::ExpertsAllocGuard::for_device(cache.device());
        let _staging = synaptix_core::device::cuda::PinnedStageGuard::new();
        cache.arena_group();
        let fetched = match &self.experts {
            ExpertStore::Resident(all) => {
                let host = all
                    .get(expert)
                    .ok_or_else(|| ModelError::Forward(format!("MoE: нет эксперта {expert}")))?;
                let pinned = cache._mirror.is_some();
                if pinned {
                    synaptix_core::device::cuda::set_pin_mirror(true);
                }
                let moved = host.to_device(cache.device());
                if pinned {
                    synaptix_core::device::cuda::set_pin_mirror(false);
                }
                moved?
            }
            ExpertStore::Lazy { source, count } => {
                if expert >= *count {
                    return Err(ModelError::Forward(format!("MoE: нет эксперта {expert}")));
                }
                let (gate_up, down) = source.fetch(self.layer_id, expert, cache.device())?;
                Expert { gate_up, down }
            }
        };
        for w in [fetched.gate_up.quant_weight(), fetched.down.quant_weight()] {
            if let Some(w) = w {
                w.mark_expert_pool();
            }
        }
        let e = Arc::new(fetched);
        cache.insert(key, e.clone());
        Ok(e)
    }

    /// Статистика попаданий в кэш резидентных экспертов (`None` — оффлоад выключен).
    pub fn cache_stats(&self) -> Option<ExpertCacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }

    /// Вторая половина эксперта: `silu(gate) · up`, затем проекция вниз. У
    /// NVFP4-веса это один фьюз вместо пяти ядер — активация сразу выходит
    /// квантованной, ровно в том виде, какой нужен GEMM'у.
    fn down_from_gate_up(
        &self,
        down: &QLinear,
        gate_up: &Tensor,
        m: usize,
    ) -> Result<Tensor, ModelError> {
        if down.quant_dtype() == Some(DType::NVFP4) {
            if let Ok((packed, scales)) = gate_up.silu_mul_quant_nvfp4(1.0) {
                return down.forward_prequant(&packed, &scales, m);
            }
        }
        let h = self.swiglu(gate_up)?;
        down.forward(&h)
    }

    /// `gate_up: [m, 2I]` → `silu(gate) * up`.
    fn swiglu(&self, gate_up: &Tensor) -> Result<Tensor, ModelError> {
        let i = self.cfg.moe_intermediate_size;
        let gate = gate_up
            .narrow(1, 0, i)
            .and_then(|t| t.contiguous())
            .map_err(|e| ModelError::Forward(format!("MoE: gate: {e}")))?;
        let up = gate_up
            .narrow(1, i, i)
            .and_then(|t| t.contiguous())
            .map_err(|e| ModelError::Forward(format!("MoE: up: {e}")))?;
        // Fused-ядро есть не на всех устройствах — тогда обычные silu и mul.
        match gate.silu_and_mul(&up) {
            Ok(h) => Ok(h),
            Err(_) => gate
                .silu()
                .and_then(|g| g.mul(&up))
                .map_err(|e| ModelError::Forward(format!("MoE: swiglu: {e}"))),
        }
    }

    fn shared_forward(&self, shared: &SharedExpert, x: &Tensor) -> Result<Tensor, ModelError> {
        let gate = shared.gate.forward(x)?;
        let up = shared.up.forward(x)?;
        let h = match gate.silu_and_mul(&up) {
            Ok(h) => h,
            Err(_) => gate
                .silu()
                .and_then(|g| g.mul(&up))
                .map_err(|e| ModelError::Forward(format!("MoE: shared swiglu: {e}")))?,
        };
        let y = self.to_compute(shared.down.forward(&h)?)?;
        let g = x
            .to_dtype(DType::F32)
            .and_then(|xf| xf.linear(&shared.router))
            .and_then(|l| l.sigmoid())
            .and_then(|l| l.to_dtype(y.dtype()))
            .map_err(|e| ModelError::Forward(format!("MoE: гейт shared expert'а: {e}")))?;
        y.broadcast_mul(&g)
            .map_err(|e| ModelError::Forward(format!("MoE: shared expert: {e}")))
    }

    fn to_compute(&self, t: Tensor) -> Result<Tensor, ModelError> {
        if t.dtype() == self.compute {
            return Ok(t);
        }
        t.to_dtype(self.compute)
            .map_err(|e| ModelError::Forward(format!("MoE: приведение к {:?}: {e}", self.compute)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::memory::expert_arena;

    /// Эксперт под MXFP8: `[2I, H]` и `[H, I]` при H=1024, I=512 — около
    /// полутора мегабайт, как у настоящего Qwen4Exp.
    const H: usize = 1024;
    const I: usize = 512;

    fn quantized(n: usize, k: usize, device: Device) -> QLinear {
        let w = Tensor::zeros(vec![n, k], DType::F16, device).expect("zeros");
        QLinear::build(w, DType::MXFP8, DType::F16).expect("quantize")
    }

    fn expert(device: Device) -> Expert {
        Expert { gate_up: quantized(2 * I, H, device), down: quantized(H, I, device) }
    }

    /// Вытеснение обязано возвращать память драйверу, а не только «по учёту».
    ///
    /// Пока кэш вытеснял по одному, освобождённые эксперты оставались
    /// вперемешку с живыми, и `cuMemPoolTrimTo` не отдавал драйверу ничего:
    /// на живой сессии кэш «ужимался» 13 → 9.4 ГБ, а `cuMemGetInfo` показывал
    /// 40 МБ свободных, и префилл падал с OOM. Теперь единица вытеснения —
    /// slab арены.
    #[test]
    fn eviction_returns_vram_to_the_driver() {
        synaptix_kernels_cpu::ensure_registered();
        synaptix_kernels_cuda::ensure_registered();
        let device = Device::Cuda(0);
        if synaptix_core::device::cuda::mem_info(0).is_err() {
            eprintln!("CUDA недоступна — тест пропущен");
            return;
        }
        if !expert_arena::enabled() {
            eprintln!("арена выключена через SYN_EXPERT_ARENA — тест пропущен");
            return;
        }

        // Ёмкости хватает на всех: вытеснения на вставке быть не должно.
        let cache = ExpertCache::new(device, 4 << 30);
        let mut expected = 0usize;
        {
            let _pool = synaptix_core::device::cuda::ExpertsAllocGuard::for_device(device);
            for i in 0..384usize {
                cache.arena_group();
                let e = expert(device);
                expected = e.bytes();
                cache.insert((0, i), Arc::new(e));
            }
        }
        let _ = synaptix_core::device::cuda::synchronize_all(0);
        let before = expert_arena::stats();
        assert!(
            before.slabs >= 3,
            "ожидали несколько slab'ов под {} экспертов по {} КБ, получили {}",
            384,
            expected / 1024,
            before.slabs
        );
        let (free_before, _) = synaptix_core::device::cuda::mem_info(0).expect("mem_info");

        // Просим кэш ужаться вдвое — ровно то, что делает `fit_to_vram` перед
        // префиллом следующего хода.
        cache.trim_to(cache.used_bytes() / 2);
        let (free_after, _) = synaptix_core::device::cuda::mem_info(0).expect("mem_info");
        let returned = free_after.saturating_sub(free_before);
        let after = expert_arena::stats();
        eprintln!(
            "slab'ов {} → {}; кэш держит {} МБ; драйверу вернулось {} МБ",
            before.slabs,
            after.slabs,
            cache.used_bytes() / (1024 * 1024),
            returned / (1024 * 1024),
        );

        assert!(
            after.slabs < before.slabs,
            "вытеснение не освободило ни одного slab'а: {} → {}",
            before.slabs,
            after.slabs
        );
        assert!(
            returned >= expert_arena::slab_bytes(),
            "драйверу вернулось {} МБ — меньше одного slab'а",
            returned / (1024 * 1024)
        );
        drop(cache);
        expert_arena::release_empty(0);
    }
}
