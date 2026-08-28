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
use std::sync::{Arc, Mutex};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
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
    capacity_bytes: usize,
    inner: Mutex<CacheInner>,
    /// Pinned-зеркало host-весов (`SYN_MOE_PINNED=1`, по умолчанию выключено).
    /// Первая отправка эксперта копирует его в закреплённую память — на слое
    /// 512 экспертов это втрое дороже обычной pageable-копии, — зато повторные
    /// отправки того же эксперта идут DMA без staging. Окупается только при
    /// тесном кэше с частым вытеснением, и ценой второй копии каждого
    /// отправленного эксперта в RAM.
    _mirror: Option<synaptix_core::device::cuda::PinMirrorGuard>,
}

struct CacheInner {
    map: HashMap<(usize, usize), Arc<Expert>>,
    order: VecDeque<(usize, usize)>,
    bytes: usize,
    hits: u64,
    misses: u64,
    skipped: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Сколько пар «токен—эксперт» отброшено аварийным клапаном
    /// ([`MoeConfig::skip_below`]).
    pub skipped: u64,
    pub resident: usize,
    pub bytes: usize,
}

impl ExpertCache {
    pub fn new(device: Device, capacity_bytes: usize) -> Arc<Self> {
        let pinned = matches!(device, Device::Cuda(_))
            && std::env::var("SYN_MOE_PINNED").map(|v| v.trim() == "1").unwrap_or(false);
        Arc::new(Self {
            device,
            capacity_bytes,
            _mirror: pinned.then(synaptix_core::device::cuda::PinMirrorGuard::new),
            inner: Mutex::new(CacheInner {
                map: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
                hits: 0,
                misses: 0,
                skipped: 0,
            }),
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn stats(&self) -> ExpertCacheStats {
        let inner = self.inner.lock().expect("кэш экспертов отравлен");
        ExpertCacheStats {
            hits: inner.hits,
            misses: inner.misses,
            skipped: inner.skipped,
            resident: inner.map.len(),
            bytes: inner.bytes,
        }
    }

    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("кэш экспертов отравлен");
        inner.map.clear();
        inner.order.clear();
        inner.bytes = 0;
    }

    fn note_skipped(&self, count: u64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.skipped += count;
        }
    }

    fn get(&self, key: (usize, usize)) -> Option<Arc<Expert>> {
        let mut inner = self.inner.lock().ok()?;
        match inner.map.get(&key) {
            Some(e) => {
                let e = e.clone();
                inner.hits += 1;
                Some(e)
            }
            None => {
                inner.misses += 1;
                None
            }
        }
    }

    fn contains(&self, key: (usize, usize)) -> bool {
        self.inner
            .lock()
            .map(|inner| inner.map.contains_key(&key))
            .unwrap_or(false)
    }

    fn insert(&self, key: (usize, usize), expert: Arc<Expert>) {
        let bytes = expert.bytes();
        let Ok(mut inner) = self.inner.lock() else { return };
        if inner.map.contains_key(&key) {
            return;
        }
        while inner.bytes + bytes > self.capacity_bytes {
            let Some(victim) = inner.order.pop_front() else { break };
            if let Some(old) = inner.map.remove(&victim) {
                inner.bytes = inner.bytes.saturating_sub(old.bytes());
            }
        }
        inner.bytes += bytes;
        inner.order.push_back(key);
        inner.map.insert(key, expert);
    }
}

struct SharedExpert {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
    /// `[1, H]` — sigmoid-гейт всего shared-выхода.
    router: Tensor,
}

pub struct MoeFfn {
    cfg: MoeConfig,
    cache: Option<Arc<ExpertCache>>,
    layer_id: usize,
    /// `[E, H]` в F32: софтмакс роутера считается в полной точности, иначе
    /// на 512 экспертах порядок top-k пляшет от округления.
    router: Tensor,
    experts: Vec<Expert>,
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
        let experts = gate_up
            .into_iter()
            .zip(down)
            .map(|(gate_up, down)| Expert { gate_up, down })
            .collect();

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

        Ok(Self { cfg, cache: None, layer_id: 0, router, experts, shared, device, compute })
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
        let logits = logits
            .to_device(Device::Cpu)
            .and_then(|l| l.flatten_all())
            .and_then(|l| l.to_vec1::<f32>())
            .map_err(|err| ModelError::Forward(format!("роутер MoE: выгрузка: {err}")))?;

        let mut experts = vec![0u32; t * k];
        let mut weights = vec![0f32; t * k];
        let mut order: Vec<u32> = Vec::with_capacity(e);
        for i in 0..t {
            let row = &logits[i * e..(i + 1) * e];
            order.clear();
            order.extend(0..e as u32);
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
            let denom: f32 = if self.cfg.norm_topk_prob {
                exps.iter().sum()
            } else {
                row.iter().map(|v| (v - max).exp()).sum()
            };
            for (s, (idx, ex)) in top.iter().zip(exps.iter()).enumerate() {
                experts[i * k + s] = *idx;
                weights[i * k + s] = ex / denom;
            }
        }
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
            parts.push(self.forward_chunk(&chunk)?);
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
        let t = x.dims()[0];
        let k = self.cfg.num_experts_per_tok;
        let (experts, weights) = self.route(x)?;

        // Пары (токен, слот), сгруппированные по эксперту: каждый эксперт
        // получает один GEMM вместо GEMV на токен.
        let mut order: Vec<u32> = (0..(t * k) as u32).collect();
        order.sort_unstable_by_key(|p| (experts[*p as usize], *p));

        let rows: Vec<u32> = order.iter().map(|p| *p / k as u32).collect();
        let row_idx = Tensor::from_vec::<_, u32>(rows, vec![t * k], self.device)
            .map_err(|e| ModelError::Forward(format!("MoE: индексы строк: {e}")))?;
        let gathered = x
            .index_select(0, &row_idx)
            .map_err(|e| ModelError::Forward(format!("MoE: сбор токенов: {e}")))?;

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
            let slice = gathered
                .narrow(0, pos, end - pos)
                .and_then(|t| t.contiguous())
                .map_err(|e| ModelError::Forward(format!("MoE: срез эксперта {expert}: {e}")))?;
            let out = self.expert_forward(expert as usize, &slice)?;
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

        let refs: Vec<&Tensor> = outs.iter().collect();
        let stacked = if refs.len() == 1 {
            outs[0].clone()
        } else {
            Tensor::cat(&refs, 0)
                .map_err(|e| ModelError::Forward(format!("MoE: сборка экспертов: {e}")))?
        };

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
        let mixed = scaled
            .index_select(0, &inverse)
            .and_then(|m| m.reshape((t, k, self.cfg.hidden_size)))
            .and_then(|m| m.sum([1usize]))
            .map_err(|e| ModelError::Forward(format!("MoE: сумма по экспертам: {e}")))?;
        let mixed = self.to_compute(mixed)?;

        match &self.shared {
            Some(shared) => {
                let s = self.shared_forward(shared, x)?;
                mixed
                    .add(&s)
                    .map_err(|e| ModelError::Forward(format!("MoE: shared expert: {e}")))
            }
            None => Ok(mixed),
        }
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
        let host = self
            .experts
            .get(expert)
            .ok_or_else(|| ModelError::Forward(format!("MoE: нет эксперта {expert}")))?;
        match &self.cache {
            None => {
                let gu = host.gate_up.forward(x)?;
                let h = self.swiglu(&gu)?;
                host.down.forward(&h)
            }
            Some(cache) => {
                let key = (self.layer_id, expert);
                let resident = match cache.get(key) {
                    Some(e) => e,
                    None => {
                        let pinned = cache._mirror.is_some();
                        if pinned {
                            synaptix_core::device::cuda::set_pin_mirror(true);
                        }
                        let moved = host.to_device(cache.device());
                        if pinned {
                            synaptix_core::device::cuda::set_pin_mirror(false);
                        }
                        let e = Arc::new(moved?);
                        cache.insert(key, e.clone());
                        e
                    }
                };
                let gu = resident.gate_up.forward(x)?;
                let h = self.swiglu(&gu)?;
                resident.down.forward(&h)
            }
        }
    }

    /// Статистика попаданий в кэш резидентных экспертов (`None` — оффлоад выключен).
    pub fn cache_stats(&self) -> Option<ExpertCacheStats> {
        self.cache.as_ref().map(|c| c.stats())
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
