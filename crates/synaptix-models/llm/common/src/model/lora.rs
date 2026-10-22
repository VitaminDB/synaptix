//! Адаптеры LoRA при генерации: `y = W·x + Σ sᵢ · Bᵢ·(Aᵢ·x)` поверх
//! замороженной (в том числе квантованной) проекции, без слияния с весами.
//!
//! Адаптер — каталог PEFT (`adapter_model.safetensors` +
//! `adapter_config.json`), какой пишет `synaptix train`. На модели их может
//! быть несколько: каждый включается и выключается отдельно, у каждого своя
//! сила (множитель масштаба). Включённые складываются.
//!
//! Добавки живут в таблице по адресу проекции [`QLinear`]: так их видит
//! любой путь, что зовёт `QLinear::forward`, без правки сигнатур. Слитые
//! пути декода (графовый шаг, device-шаг, слитые блоки) читают веса мимо
//! `forward` — при активном адаптере модель их не выбирает
//! ([`DecoderModel::lora_active`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use synaptix_core::backend::train::TrainOp;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result as CoreResult;
use synaptix_core::tensor::Tensor;

use super::{DecoderModel, Mixer, ModelError};
use crate::weights::QLinear;

/// Добавка к одной проекции: `scale · B[N, r] · A[r, K]`, веса в F32.
pub struct LoraPair {
    pub a: Tensor,
    pub b: Tensor,
    pub scale: f32,
}

impl LoraPair {
    /// `y + scale · B·(A·x)`; `x` — `[.., K]`, `y` — `[.., N]`.
    fn apply(&self, x: &Tensor, y: Tensor) -> CoreResult<Tensor> {
        let k = x.dims()[x.rank() - 1];
        let n = y.dims()[y.rank() - 1];
        let rows = x.numel() / k.max(1);
        let r = self.a.dims()[0];
        let x2 = x.contiguous()?.reshape(vec![rows, k])?;
        let y2 = y.contiguous()?.reshape(vec![rows, n])?;
        let fused = || -> CoreResult<Tensor> {
            let x2 = if matches!(x2.dtype(), DType::F32 | DType::BF16 | DType::F16) { x2.clone() } else { x2.to_dtype(DType::F32)? };
            let z = Tensor::train_op(TrainOp::LoraDown, &[&x2, &self.a], &[(vec![rows, r], DType::F32)])?.remove(0);
            Ok(Tensor::train_op(TrainOp::LoraUpAdd { scale: self.scale }, &[&y2, &z, &self.b], &[(vec![rows, n], y2.dtype())])?.remove(0))
        };
        let out = match fused() {
            Ok(o) => o,
            // Без ядер (CPU): обычными умножениями.
            Err(synaptix_core::error::SynaptixError::Unsupported(_)) => {
                let z = x2.to_dtype(DType::F32)?.matmul(&self.a.t()?)?;
                let d = z.matmul(&self.b.t()?)?.mul_scalar(self.scale)?;
                y2.add(&d.to_dtype(y2.dtype())?)?
            }
            Err(e) => return Err(e),
        };
        out.reshape(y.dims().to_vec())
    }
}

// ── Таблица добавок ──────────────────────────────────────────────────────────

/// Сколько проекций сейчас с добавкой: ноль — путь `forward` не смотрит в таблицу.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

fn table() -> &'static RwLock<HashMap<usize, Vec<Arc<LoraPair>>>> {
    static T: OnceLock<RwLock<HashMap<usize, Vec<Arc<LoraPair>>>>> = OnceLock::new();
    T.get_or_init(|| RwLock::new(HashMap::new()))
}

fn key(q: &QLinear) -> usize {
    q as *const QLinear as usize
}

fn set_pairs(k: usize, pairs: Vec<Arc<LoraPair>>) {
    let mut t = table().write().unwrap_or_else(|e| e.into_inner());
    if pairs.is_empty() {
        t.remove(&k);
    } else {
        t.insert(k, pairs);
    }
    ACTIVE.store(t.len(), Ordering::Release);
}

/// Есть ли у проекции добавка.
pub(crate) fn has(q: &QLinear) -> bool {
    ACTIVE.load(Ordering::Acquire) != 0 && table().read().unwrap_or_else(|e| e.into_inner()).contains_key(&key(q))
}

/// Прибавить к выходу проекции включённые добавки.
pub(crate) fn apply(q: &QLinear, x: &Tensor, y: Tensor) -> CoreResult<Tensor> {
    if ACTIVE.load(Ordering::Acquire) == 0 {
        return Ok(y);
    }
    let pairs = match table().read().unwrap_or_else(|e| e.into_inner()).get(&key(q)) {
        Some(p) => p.clone(),
        None => return Ok(y),
    };
    let mut y = y;
    for p in &pairs {
        y = p.apply(x, y)?;
    }
    Ok(y)
}

// ── Адаптеры модели ──────────────────────────────────────────────────────────

/// Где в слое стоит проекция.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Slot {
    Q,
    K,
    V,
    O,
    Qkv,
    Z,
    A,
    B,
    Out,
    Gate,
    Up,
    Down,
    /// Shared-эксперт MoE (`mlp.shared_expert.*`).
    SharedGate,
    SharedUp,
    SharedDown,
}

impl Slot {
    fn parse(kind: &str, name: &str) -> Option<Self> {
        Some(match (kind, name) {
            ("self_attn", "q_proj") => Slot::Q,
            ("self_attn", "k_proj") => Slot::K,
            ("self_attn", "v_proj") => Slot::V,
            ("self_attn", "o_proj") => Slot::O,
            ("linear_attn", "in_proj_qkv") => Slot::Qkv,
            ("linear_attn", "in_proj_z") => Slot::Z,
            ("linear_attn", "in_proj_a") => Slot::A,
            ("linear_attn", "in_proj_b") => Slot::B,
            ("linear_attn", "out_proj") => Slot::Out,
            ("mlp", "gate_proj") => Slot::Gate,
            ("mlp", "up_proj") => Slot::Up,
            ("mlp", "down_proj") => Slot::Down,
            ("mlp.shared_expert", "gate_proj") => Slot::SharedGate,
            ("mlp.shared_expert", "up_proj") => Slot::SharedUp,
            ("mlp.shared_expert", "down_proj") => Slot::SharedDown,
            _ => return None,
        })
    }
}

/// Сведения об адаптере модели.
#[derive(Clone, Debug)]
pub struct LoraInfo {
    pub name: String,
    pub path: PathBuf,
    pub enabled: bool,
    /// Множитель силы (1 — как учился).
    pub strength: f32,
    pub rank: usize,
    pub alpha: f32,
    pub modules: usize,
    pub bytes: usize,
}

/// Адаптер, загруженный на модель.
pub struct LoadedLora {
    info: LoraInfo,
    /// `(слой, место) → (A, B)`; масштаб — `alpha / r · strength`.
    mats: Vec<(usize, Slot, Tensor, Tensor)>,
}

/// Адаптеры модели.
#[derive(Default)]
pub struct LoraState {
    loaded: Vec<LoadedLora>,
    /// Адреса проекций модели, у которых сейчас есть записи в таблице.
    keys: Vec<usize>,
}

impl LoraState {
    pub fn any_enabled(&self) -> bool {
        self.loaded.iter().any(|l| l.info.enabled)
    }
}

/// Прочитать адаптер PEFT с диска на `device`.
fn read_adapter(dir: &Path, device: Device, n_layers: usize) -> Result<(usize, f32, Vec<(usize, Slot, Tensor, Tensor)>, usize), ModelError> {
    let err = |m: String| ModelError::Load(m);
    let cfg: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("adapter_config.json")).map_err(|e| err(format!("{}: adapter_config.json: {e}", dir.display())))?,
    )
    .map_err(|e| err(format!("adapter_config.json: {e}")))?;
    let r = cfg.get("r").and_then(|v| v.as_u64()).ok_or_else(|| err("adapter_config.json: нет r".into()))? as usize;
    let alpha = cfg.get("lora_alpha").and_then(|v| v.as_f64()).unwrap_or(r as f64) as f32;
    let bytes = std::fs::read(dir.join("adapter_model.safetensors")).map_err(|e| err(format!("adapter_model.safetensors: {e}")))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| err(e.to_string()))?;
    let read = |name: &str| -> Result<Tensor, ModelError> {
        let v = st.tensor(name).map_err(|e| err(format!("{name}: {e}")))?;
        let data: Vec<f32> = match v.dtype() {
            safetensors::Dtype::F32 => v.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            safetensors::Dtype::BF16 => v.data().chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            safetensors::Dtype::F16 => v.data().chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            d => return Err(err(format!("{name}: тип {d:?}"))),
        };
        Tensor::from_vec(data, v.shape().to_vec(), device).map_err(|e| err(e.to_string()))
    };
    let mut mats = Vec::new();
    let mut total = 0usize;
    for name in st.names() {
        let Some(module) = name.strip_suffix(".lora_A.weight") else { continue };
        // `…layers.<i>.<kind>.<proj>`, `kind` — с точками (`mlp.shared_expert`).
        let Some(pos) = module.rfind("layers.") else { continue };
        let rest = &module[pos + "layers.".len()..];
        let Some((i, tail)) = rest.split_once('.') else { continue };
        let Some((kind, proj)) = tail.rsplit_once('.') else { continue };
        let Ok(i) = i.parse::<usize>() else { continue };
        if i >= n_layers {
            return Err(err(format!("{module}: у модели {n_layers} слоёв")));
        }
        let slot = Slot::parse(kind, proj).ok_or_else(|| err(format!("{module}: проекция не поддержана")))?;
        let a = read(name)?;
        let b = read(&format!("{module}.lora_B.weight"))?;
        total += (a.numel() + b.numel()) * 4;
        mats.push((i, slot, a, b));
    }
    if mats.is_empty() {
        return Err(err(format!("{}: в адаптере нет матриц LoRA", dir.display())));
    }
    Ok((r, alpha, mats, total))
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = match e {
        0 if m == 0 => s,
        0 => {
            let mut e = -14i32;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13)
        }
        31 => s | 0x7f80_0000 | (m << 13),
        _ => s | (((e - 15 + 127) as u32) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

impl DecoderModel {
    fn slot_linear(&self, layer: usize, slot: Slot) -> Option<&QLinear> {
        let b = self.blocks.get(layer)?;
        match (&b.mixer, slot) {
            (Mixer::Full(fa), Slot::Q) => Some(&fa.q_proj),
            (Mixer::Full(fa), Slot::K) => Some(&fa.k_proj),
            (Mixer::Full(fa), Slot::V) => fa.v_proj.as_ref(),
            (Mixer::Full(fa), Slot::O) => Some(&fa.o_proj),
            (Mixer::Linear(la), Slot::Qkv) => Some(&la.in_proj_qkv),
            (Mixer::Linear(la), Slot::Z) => Some(&la.in_proj_z),
            (Mixer::Linear(la), Slot::A) => Some(&la.in_proj_a),
            (Mixer::Linear(la), Slot::B) => Some(&la.in_proj_b),
            (Mixer::Linear(la), Slot::Out) => Some(&la.out_proj),
            (_, Slot::Gate) => b.mlp.dense().map(|m| &m.gate_proj),
            (_, Slot::Up) => b.mlp.dense().map(|m| &m.up_proj),
            (_, Slot::Down) => b.mlp.dense().map(|m| &m.down_proj),
            (_, Slot::SharedGate) => b.mlp.moe().and_then(|m| m.train_shared()).map(|s| s.gate),
            (_, Slot::SharedUp) => b.mlp.moe().and_then(|m| m.train_shared()).map(|s| s.up),
            (_, Slot::SharedDown) => b.mlp.moe().and_then(|m| m.train_shared()).map(|s| s.down),
            _ => None,
        }
    }

    /// Пересобрать таблицу добавок модели по включённым адаптерам.
    fn rebuild_lora(&mut self) {
        let mut per: HashMap<(usize, Slot), Vec<Arc<LoraPair>>> = HashMap::new();
        for l in self.lora.loaded.iter().filter(|l| l.info.enabled && l.info.strength != 0.0) {
            let scale = l.info.alpha / l.info.rank as f32 * l.info.strength;
            for (i, slot, a, b) in &l.mats {
                per.entry((*i, *slot)).or_default().push(Arc::new(LoraPair { a: a.clone(), b: b.clone(), scale }));
            }
        }
        for k in std::mem::take(&mut self.lora.keys) {
            set_pairs(k, Vec::new());
        }
        let mut keys = Vec::new();
        for ((i, s), pairs) in per {
            if let Some(q) = self.slot_linear(i, s) {
                keys.push(key(q));
                set_pairs(key(q), pairs);
            }
        }
        self.lora.keys = keys;
        // Захваченные графы декода читали веса без добавок.
        self.weights_gen += 1;
        self.lora_gen += 1;
    }

    /// Загрузить адаптер PEFT из `dir` под именем `name` (включённым).
    /// Имя уже есть — адаптер заменяется.
    pub fn load_lora(&mut self, name: &str, dir: &Path) -> Result<LoraInfo, ModelError> {
        if !self.blocks_all_resident() {
            return Err(ModelError::Build("LoRA: модель должна целиком лежать на карте (не в режиме офлоада блоков)".into()));
        }
        let (r, alpha, mats, bytes) = read_adapter(dir, self.device, self.blocks.len())?;
        for (i, slot, a, b) in &mats {
            let q = self
                .slot_linear(*i, *slot)
                .ok_or_else(|| ModelError::Build(format!("LoRA: у слоя {i} нет проекции {slot:?} (другая архитектура?)")))?;
            let (n, k) = match q {
                QLinear::Dense(l) => (l.weight().dims()[0], l.weight().dims()[1]),
                QLinear::Quant(w) => (w.n(), w.k()),
            };
            if a.dims() != [a.dims()[0], k] || b.dims() != [n, a.dims()[0]] {
                return Err(ModelError::Build(format!(
                    "LoRA: слой {i} {slot:?}: A {:?}, B {:?}, а проекция [{n}, {k}]",
                    a.dims(),
                    b.dims()
                )));
            }
        }
        let info = LoraInfo { name: name.to_string(), path: dir.to_path_buf(), enabled: true, strength: 1.0, rank: r, alpha, modules: mats.len(), bytes };
        self.lora.loaded.retain(|l| l.info.name != name);
        self.lora.loaded.push(LoadedLora { info: info.clone(), mats });
        self.rebuild_lora();
        Ok(info)
    }

    /// Включить или выключить адаптер (веса остаются загруженными).
    pub fn set_lora_enabled(&mut self, name: &str, enabled: bool) -> Result<(), ModelError> {
        let l = self.lora.loaded.iter_mut().find(|l| l.info.name == name).ok_or_else(|| ModelError::Build(format!("LoRA «{name}» не загружен")))?;
        if l.info.enabled != enabled {
            l.info.enabled = enabled;
            self.rebuild_lora();
        }
        Ok(())
    }

    /// Сила адаптера: множитель его масштаба (0 — как выключен).
    pub fn set_lora_strength(&mut self, name: &str, strength: f32) -> Result<(), ModelError> {
        let l = self.lora.loaded.iter_mut().find(|l| l.info.name == name).ok_or_else(|| ModelError::Build(format!("LoRA «{name}» не загружен")))?;
        if l.info.strength != strength {
            l.info.strength = strength;
            self.rebuild_lora();
        }
        Ok(())
    }

    /// Выгрузить адаптер.
    pub fn unload_lora(&mut self, name: &str) -> bool {
        let before = self.lora.loaded.len();
        self.lora.loaded.retain(|l| l.info.name != name);
        let removed = self.lora.loaded.len() != before;
        if removed {
            self.rebuild_lora();
        }
        removed
    }

    pub fn loras(&self) -> Vec<LoraInfo> {
        self.lora.loaded.iter().map(|l| l.info.clone()).collect()
    }

    /// Есть ли включённый адаптер: тогда слитые пути декода не годятся.
    pub fn lora_active(&self) -> bool {
        self.lora.any_enabled()
    }

    /// Поколение набора адаптеров: меняется при каждой смене — кэш KV,
    /// посчитанный при другом наборе, недействителен.
    pub fn lora_generation(&self) -> u64 {
        self.lora_gen
    }
}

impl Drop for LoraState {
    fn drop(&mut self) {
        // Модель уходит: по её адресам может лечь другая — записи убрать.
        for k in std::mem::take(&mut self.keys) {
            set_pairs(k, Vec::new());
        }
    }
}
