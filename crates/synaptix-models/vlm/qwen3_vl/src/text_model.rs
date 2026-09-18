use std::path::Path;

use serde::Deserialize;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_nn::module::Module as _;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::model::{VisionError, VisionWeights};

pub const LM: &str = "model.language_model";

type R<T> = Result<T, VisionError>;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub mrope_section: Vec<usize>,
    pub mrope_interleaved: bool,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 5120,
            intermediate_size: 25600,
            num_hidden_layers: 64,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            mrope_section: vec![24, 20, 20],
            mrope_interleaved: true,
        }
    }
}

impl TextConfig {
    pub fn from_hf_bytes(bytes: &[u8]) -> Result<Self, VisionError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| VisionError::Load(format!("config.json: {e}")))?;
        let tc = root
            .get("text_config")
            .cloned()
            .ok_or_else(|| VisionError::Load("нет text_config".into()))?;
        let mut cfg: Self = serde_json::from_value(tc.clone())
            .map_err(|e| VisionError::Load(format!("text_config: {e}")))?;
        if let Some(rs) = tc.get("rope_scaling") {
            if let Some(sec) = rs.get("mrope_section").and_then(|v| v.as_array()) {
                cfg.mrope_section =
                    sec.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
            }
            if let Some(b) = rs.get("mrope_interleaved").and_then(|v| v.as_bool()) {
                cfg.mrope_interleaved = b;
            }
        }
        Ok(cfg)
    }

    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, VisionError> {
        let p = dir.as_ref().join("config.json");
        let bytes = std::fs::read(&p)
            .map_err(|e| VisionError::Load(format!("{}: {e}", p.display())))?;
        Self::from_hf_bytes(&bytes)
    }

    /// Размер одного слоя: (в выбранном кванте, плотный в `compute`).
    pub fn layer_bytes(&self, quant: DType, compute: DType) -> (usize, usize) {
        let h = self.hidden_size;
        let numel = h * self.head_dim * (2 * self.num_attention_heads + 2 * self.num_key_value_heads)
            + 3 * h * self.intermediate_size;
        let dense = numel * compute.bytes_for_numel(1).max(1);
        let q = match quant {
            DType::NVFP4 => numel / 2 + numel / 16,
            DType::MXFP8 => numel + numel / 32,
            _ => dense,
        };
        (q, dense)
    }

    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

struct Lin(QuantLinear);

impl Lin {
    fn load(
        w: &dyn VisionWeights,
        key: &str,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> R<Self> {
        let raw = w.tensor(&format!("{key}.weight"), device, compute)?;
        QuantLinear::build(raw, None, quant, compute)
            .map(Self)
            .map_err(|e| VisionError::Load(format!("{key}: {e}")))
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        self.0.forward(x).map_err(|e| VisionError::Forward(e.to_string()))
    }
}

fn rms(x: &Tensor, w: &Tensor, eps: f32) -> R<Tensor> {
    if let Ok(y) = x.rms_norm_fused(w, eps, false) {
        return Ok(y);
    }
    synaptix_ops::norm::rms_norm::rms_norm(x, w, eps)
        .map_err(|e| VisionError::Forward(e.to_string()))
}

struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Lin,
    up: Lin,
    down: Lin,
}

impl Layer {
    fn load(
        w: &dyn VisionWeights,
        idx: usize,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> R<Self> {
        let p = format!("{LM}.layers.{idx}");
        Ok(Self {
            input_norm: w.tensor(&format!("{p}.input_layernorm.weight"), device, compute)?,
            post_norm: w.tensor(&format!("{p}.post_attention_layernorm.weight"), device, compute)?,
            q: Lin::load(w, &format!("{p}.self_attn.q_proj"), device, compute, quant)?,
            k: Lin::load(w, &format!("{p}.self_attn.k_proj"), device, compute, quant)?,
            v: Lin::load(w, &format!("{p}.self_attn.v_proj"), device, compute, quant)?,
            o: Lin::load(w, &format!("{p}.self_attn.o_proj"), device, compute, quant)?,
            q_norm: w.tensor(&format!("{p}.self_attn.q_norm.weight"), device, compute)?,
            k_norm: w.tensor(&format!("{p}.self_attn.k_norm.weight"), device, compute)?,
            gate: Lin::load(w, &format!("{p}.mlp.gate_proj"), device, compute, quant)?,
            up: Lin::load(w, &format!("{p}.mlp.up_proj"), device, compute, quant)?,
            down: Lin::load(w, &format!("{p}.mlp.down_proj"), device, compute, quant)?,
        })
    }
}

pub struct MRopeTables {
    pub cos: Tensor,
    pub sin: Tensor,
}

pub fn build_mrope(
    positions: &[[u32; 3]],
    cfg: &TextConfig,
    device: Device,
) -> Result<MRopeTables, VisionError> {
    let half = cfg.head_dim / 2;
    let s = positions.len();
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / cfg.head_dim as f32))
        .collect();

    let mut axis_of = vec![0usize; half];
    if cfg.mrope_interleaved {
        let sec = &cfg.mrope_section;
        for (ax, offset) in [(1usize, 1usize), (2, 2)] {
            let limit = (sec.get(ax).copied().unwrap_or(0) * 3).min(half);
            let mut i = offset;
            while i < limit {
                axis_of[i] = ax;
                i += 3;
            }
        }
    } else {
        let sec = &cfg.mrope_section;
        let mut off = 0usize;
        for (ax, n) in sec.iter().enumerate() {
            for i in off..(off + n).min(half) {
                axis_of[i] = ax;
            }
            off += n;
        }
    }

    let mut cos = vec![0f32; s * half];
    let mut sin = vec![0f32; s * half];
    for (si, p) in positions.iter().enumerate() {
        for i in 0..half {
            let ang = p[axis_of[i]] as f32 * inv[i];
            cos[si * half + i] = ang.cos();
            sin[si * half + i] = ang.sin();
        }
    }
    Ok(MRopeTables {
        cos: Tensor::from_vec(cos, vec![s, half], device)
            .map_err(|e| VisionError::Load(e.to_string()))?,
        sin: Tensor::from_vec(sin, vec![s, half], device)
            .map_err(|e| VisionError::Load(e.to_string()))?,
    })
}

pub struct VisionSpan {
    pub start: usize,
    pub len: usize,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

pub fn rope_positions(seq_len: usize, spans: &[VisionSpan]) -> Vec<[u32; 3]> {
    let mut out = vec![[0u32; 3]; seq_len];
    let mut cursor = 0u32;
    let mut i = 0usize;
    let mut span_idx = 0usize;
    while i < seq_len {
        if span_idx < spans.len() && spans[span_idx].start == i {
            let sp = &spans[span_idx];
            let base = cursor;
            let mut maxp = base;
            for t in 0..sp.grid_t {
                for h in 0..sp.grid_h {
                    for w in 0..sp.grid_w {
                        let idx = i + t * sp.grid_h * sp.grid_w + h * sp.grid_w + w;
                        if idx >= seq_len {
                            continue;
                        }
                        let p = [base + t as u32, base + h as u32, base + w as u32];
                        maxp = maxp.max(p[0]).max(p[1]).max(p[2]);
                        out[idx] = p;
                    }
                }
            }
            i += sp.len;
            cursor = maxp + 1;
            span_idx += 1;
            continue;
        }
        out[i] = [cursor, cursor, cursor];
        cursor += 1;
        i += 1;
    }
    out
}

/// Общий источник весов: нужен владеющий, чтобы слои, не влезшие на карту,
/// читать из него уже во время forward.
pub type SharedWeights = std::sync::Arc<dyn VisionWeights + Send + Sync>;

pub struct TextEncoder {
    pub config: TextConfig,
    embed: Tensor,
    /// Резидентный префикс слоёв.
    layers: Vec<Layer>,
    /// Сколько слоёв всего; слои `layers.len()..num_layers` стримятся.
    num_layers: usize,
    weights: Option<SharedWeights>,
    quant: DType,
    device: Device,
    dtype: DType,
}

fn free_vram(device: Device) -> Option<usize> {
    match device {
        Device::Cuda(ord) => synaptix_core::device::cuda::mem_info(ord).ok().map(|(f, _)| f),
        _ => None,
    }
}

fn trim_pool(device: Device) {
    if let Device::Cuda(ord) = device {
        let _ = synaptix_core::device::cuda::synchronize_all(ord);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(ord);
    }
}

impl TextEncoder {
    pub fn build(
        config: TextConfig,
        weights: &dyn VisionWeights,
        device: Device,
        compute: DType,
        quant: DType,
        num_layers: usize,
    ) -> Result<Self, VisionError> {
        let n = num_layers.min(config.num_hidden_layers);
        let embed = weights.tensor(&format!("{LM}.embed_tokens.weight"), device, compute)?;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(Layer::load(weights, i, device, compute, quant)?);
        }
        Ok(Self { config, embed, layers, num_layers: n, weights: None, quant, device, dtype: compute })
    }

    /// Как [`Self::build`], но на карту кладётся только то, что влезает:
    /// остальные слои читаются из `weights` по одному во время forward.
    /// Проход энкодера по слоям одноразовый, поэтому копия на хосте не
    /// нужна — слой едет из mmap-источника, считается и освобождается.
    /// Раньше MXFP8 (~25 ГБ) и dense (~49 ГБ) на карте 24 ГБ падали OOM
    /// посреди загрузки.
    pub fn build_shared(
        config: TextConfig,
        weights: SharedWeights,
        device: Device,
        compute: DType,
        quant: DType,
        num_layers: usize,
    ) -> Result<Self, VisionError> {
        let n = num_layers.min(config.num_hidden_layers);
        let embed = weights.tensor(&format!("{LM}.embed_tokens.weight"), device, compute)?;
        let (lq, ldense) = config.layer_bytes(quant, compute);
        // Остаток после резидентных слоёв: два стримящихся слоя (текущий +
        // префетч), сырой вес под квантование и активации (башня зрения на
        // референсах + внимание по всей презентации).
        let act = std::env::var("H3_ENCODER_RESERVE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096)
            << 20;
        let reserve = 2 * lq + ldense / 2 + act;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            if let Some(free) = free_vram(device) {
                if free < reserve + lq {
                    trim_pool(device);
                    if free_vram(device).unwrap_or(0) < reserve + lq {
                        eprintln!(
                            "[qwen3-vl] VRAM кончилась на слое {i}/{n} — слои {i}..{n} стримятся из источника в forward"
                        );
                        break;
                    }
                }
            }
            layers.push(Layer::load(weights.as_ref(), i, device, compute, quant)?);
        }
        Ok(Self {
            config,
            embed,
            layers,
            num_layers: n,
            weights: Some(weights),
            quant,
            device,
            dtype: compute,
        })
    }

    /// Слои по порядку: резидентные как есть, остальные грузятся из
    /// источника, следующий — параллельно счёту текущего.
    fn for_each_layer<F>(&self, mut body: F) -> R<()>
    where
        F: FnMut(usize, &Layer) -> R<()>,
    {
        for (i, l) in self.layers.iter().enumerate() {
            body(i, l)?;
        }
        let first = self.layers.len();
        let n = self.num_layers;
        if first >= n {
            return Ok(());
        }
        let Some(weights) = self.weights.as_ref() else {
            return Err(VisionError::Forward("стриминг слоёв без источника весов".into()));
        };
        let (dev, compute, quant) = (self.device, self.dtype, self.quant);
        let ls = match dev {
            Device::Cuda(ord) => Some(
                synaptix_core::device::cuda::loader_stream(ord)
                    .map_err(|e| VisionError::Forward(e.to_string()))?,
            ),
            _ => None,
        };
        let load = |idx: usize, on_loader: bool| -> R<Layer> {
            if let (true, Some(ls)) = (on_loader, ls.as_ref()) {
                synaptix_core::device::cuda::set_alloc_stream(Some(ls.clone()));
                let r = Layer::load(weights.as_ref(), idx, dev, compute, quant);
                let _ = ls.synchronize();
                synaptix_core::device::cuda::set_alloc_stream(None);
                r
            } else {
                Layer::load(weights.as_ref(), idx, dev, compute, quant)
            }
        };
        let mut staged = Some(load(first, false));
        for idx in first..n {
            let cur = match staged.take() {
                Some(r) => r?,
                None => unreachable!(),
            };
            let load = &load;
            let (step, next) = std::thread::scope(|sp| {
                let h = (idx + 1 < n).then(|| sp.spawn(move || load(idx + 1, true)));
                let step = body(idx, &cur);
                let next = h.map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(VisionError::Forward("поток префетча слоя упал".into()))
                    })
                });
                (step, next)
            });
            // Освобождение слоя стоит в хвосте compute-стрима — без синка пул
            // берёт под следующий слой новые сегменты.
            if let Device::Cuda(ord) = dev {
                if let Ok(cs) = synaptix_core::device::cuda::default_stream(ord) {
                    let _ = cs.synchronize();
                }
            }
            drop(cur);
            step?;
            staged = next;
        }
        Ok(())
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Сколько слоёв лежит на устройстве; остальные стримятся.
    pub fn resident_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor, VisionError> {
        let idx = Tensor::from_vec(ids.to_vec(), vec![ids.len()], self.device)
            .map_err(|e| VisionError::Forward(e.to_string()))?;
        self.embed
            .embed_gather(&idx)
            .map_err(|e| VisionError::Forward(e.to_string()))
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        rope: &MRopeTables,
        deepstack: &[(Tensor, Vec<usize>)],
    ) -> Result<Tensor, VisionError> {
        let cfg = &self.config;
        let s = hidden.dims()[0];
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let group = cfg.group_size();
        let scale = 1.0 / (hd as f32).sqrt();
        let e = |r: Result<Tensor, SynaptixError>| r.map_err(|x| VisionError::Forward(x.to_string()));

        let mut x = hidden.clone();
        self.for_each_layer(|li, layer| {
            if li < deepstack.len() {
                let (feat, rows) = &deepstack[li];
                x = scatter_add_rows(&x, feat, rows)?;
            }

            let h = rms(&x, &layer.input_norm, cfg.rms_norm_eps)?;
            let q = layer.q.forward(&h)?;
            let k = layer.k.forward(&h)?;
            let v = layer.v.forward(&h)?;

            let q = rms(&e(q.reshape(vec![s, nh, hd]))?, &layer.q_norm, cfg.rms_norm_eps)?;
            let k = rms(&e(k.reshape(vec![s, nkv, hd]))?, &layer.k_norm, cfg.rms_norm_eps)?;

            let q = e(e(q.transpose(0, 1))?.contiguous())?;
            let k = e(e(k.transpose(0, 1))?.contiguous())?;
            let v = e(e(e(v.reshape(vec![s, nkv, hd]))?.transpose(0, 1))?.contiguous())?;

            let q = apply_rope(&q, rope, hd)?;
            let k = apply_rope(&k, rope, hd)?;
            let k = repeat_kv(&k, group)?;
            let v = repeat_kv(&v, group)?;

            let q = e(q.reshape(vec![1, nh, s, hd]))?;
            let k = e(k.reshape(vec![1, nh, s, hd]))?;
            let v = e(v.reshape(vec![1, nh, s, hd]))?;
            let attn = match q.dtype() {
                DType::BF16 | DType::F16 => match q.flash_attention(&k, &v, scale, true) {
                    Ok(a) => a,
                    Err(_) => {
                        let m = causal_mask(s, q.dtype(), self.device)?;
                        e(scaled_dot_attention(&q, &k, &v, scale, Some(&m)))?
                    }
                },
                _ => {
                    let m = causal_mask(s, q.dtype(), self.device)?;
                    e(scaled_dot_attention(&q, &k, &v, scale, Some(&m)))?
                }
            };
            let attn = e(e(e(e(attn.reshape(vec![nh, s, hd]))?.transpose(0, 1))?.contiguous())?
                .reshape(vec![s, nh * hd]))?;
            x = e(x.add(&layer.o.forward(&attn)?))?;

            let h = rms(&x, &layer.post_norm, cfg.rms_norm_eps)?;
            let g = layer.gate.forward(&h)?;
            let u = layer.up.forward(&h)?;
            let act = e(g.silu_and_mul(&u))?;
            x = e(x.add(&layer.down.forward(&act)?))?;
            Ok(())
        })?;
        Ok(x)
    }
}

fn apply_rope(x: &Tensor, rope: &MRopeTables, head_dim: usize) -> R<Tensor> {
    if matches!(x.device(), Device::Cuda(_)) {
        if let Ok(y) = x.rope_split_partial_fused(&rope.cos, &rope.sin, head_dim) {
            return Ok(y);
        }
    }
    let dims = x.dims().to_vec();
    let last = dims.len() - 1;
    let half = head_dim / 2;
    let dt = x.dtype();
    let cos = rope.cos.to_dtype(dt).map_err(|e| VisionError::Forward(e.to_string()))?;
    let sin = rope.sin.to_dtype(dt).map_err(|e| VisionError::Forward(e.to_string()))?;
    let e = |r: Result<Tensor, SynaptixError>| r.map_err(|x| VisionError::Forward(x.to_string()));
    let x0 = e(e(x.narrow(last, 0, half))?.contiguous())?;
    let x1 = e(e(x.narrow(last, half, half))?.contiguous())?;
    let o0 = e(e(x0.mul(&cos))?.sub(&e(x1.mul(&sin))?))?;
    let o1 = e(e(x1.mul(&cos))?.add(&e(x0.mul(&sin))?))?;
    e(Tensor::cat(&[&o0, &o1], last))
}

fn repeat_kv(x: &Tensor, group: usize) -> R<Tensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let d = x.dims().to_vec();
    let (nkv, s, hd) = (d[0], d[1], d[2]);
    let e = |r: Result<Tensor, SynaptixError>| r.map_err(|x| VisionError::Forward(x.to_string()));
    let expanded = e(x.reshape(vec![nkv, 1, s, hd]))?;
    let mut parts = Vec::with_capacity(group);
    for _ in 0..group {
        parts.push(expanded.clone());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    let cat = e(Tensor::cat(&refs, 1))?;
    e(e(cat.reshape(vec![nkv * group, s, hd]))?.contiguous())
}

fn causal_mask(n: usize, dtype: DType, device: Device) -> R<Tensor> {
    let mut v = vec![0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            v[i * n + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(v, vec![1, 1, n, n], device)
        .and_then(|t| t.to_dtype(dtype))
        .map_err(|e| VisionError::Forward(e.to_string()))
}

/// `x[rows] += feat`. Vision-строки идут сплошными блоками (по блоку на
/// картинку или группу кадров), поэтому обходимся срезами: индексный тензор
/// `[rows, hidden]` для `scatter_add` — это сотни мегабайт на референсе в
/// 2048 px, а собрать его на карте нельзя — бинарных ядер для `U32` нет
/// (так падало любое кондиционирование картинкой: «cuda binary: dtype»).
fn scatter_add_rows(x: &Tensor, feat: &Tensor, rows: &[usize]) -> R<Tensor> {
    if rows.is_empty() {
        return Ok(x.clone());
    }
    let e = |r: Result<Tensor, SynaptixError>| r.map_err(|x| VisionError::Forward(x.to_string()));
    let feat = if feat.dtype() == x.dtype() {
        feat.clone()
    } else {
        e(feat.to_dtype(x.dtype()))?
    };
    let total = x.dims()[0];
    let mut parts: Vec<Tensor> = Vec::new();
    let mut cursor = 0usize;
    for (start, len, offset) in row_runs(rows) {
        if start < cursor || start + len > total {
            return Err(VisionError::Forward(format!(
                "deepstack: строки {start}..{} вне порядка или за пределами {total}",
                start + len
            )));
        }
        if start > cursor {
            parts.push(e(e(x.narrow(0, cursor, start - cursor))?.contiguous())?);
        }
        let base = e(e(x.narrow(0, start, len))?.contiguous())?;
        let add = e(e(feat.narrow(0, offset, len))?.contiguous())?;
        parts.push(e(base.add(&add))?);
        cursor = start + len;
    }
    if cursor < total {
        parts.push(e(e(x.narrow(0, cursor, total - cursor))?.contiguous())?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    e(Tensor::cat(&refs, 0))
}

/// Серии подряд идущих строк: `(первая строка, длина, смещение в feat)`.
fn row_runs(rows: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < rows.len() {
        let mut j = i + 1;
        while j < rows.len() && rows[j] == rows[j - 1] + 1 {
            j += 1;
        }
        runs.push((rows[i], j - i, i));
        i = j;
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_runs_split_at_gaps() {
        assert_eq!(row_runs(&[3, 4, 5, 9, 10, 20]), vec![(3, 3, 0), (9, 2, 3), (20, 1, 5)]);
        assert!(row_runs(&[]).is_empty());
    }

    #[test]
    fn scatter_add_rows_adds_blocks_in_place() {
        synaptix_kernels_cpu::ensure_registered();
        let x = Tensor::zeros(vec![6, 2], DType::F32, Device::Cpu).unwrap();
        let feat =
            Tensor::from_vec(vec![1f32, 1.0, 2.0, 2.0, 3.0, 3.0], vec![3, 2], Device::Cpu).unwrap();
        let out = scatter_add_rows(&x, &feat, &[1, 2, 4]).unwrap();
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0]);
    }
}
