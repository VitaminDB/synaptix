use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::{ModelError, QLinear, WeightSource};
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::pos::rope::{apply_rope_range, RopeLayout};
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::config::Qwen4ExpConfig;
use crate::norm::{coerr, load_one_plus, rms, stage};
use crate::qsa::{IndexerCache, QsaIndexer};

const MASK_NEG: f32 = -1.0e4;

/// Потолок памяти на собранные K/V одной группы запросов.
const SPARSE_KV_BUDGET: usize = 512 << 20;

/// Считать ли выбранные позиции гатером — работа тогда растёт с бюджетом
/// индексатора, а не с длиной контекста. `SYN_QWEN4EXP_QSA_GATHER=0`
/// возвращает прежний путь с маской поверх полного внимания.
fn gather_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SYN_QWEN4EXP_QSA_GATHER").map(|v| v.trim() != "0").unwrap_or(true)
    })
}

pub struct KvLayer {
    pub k: Tensor,
    pub v: Tensor,
}

impl KvLayer {
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self, ModelError> {
        let dims = vec![1, num_kv_heads, capacity, head_dim];
        Ok(Self {
            k: Tensor::zeros(dims.clone(), dtype, device).map_err(|e| ModelError::Build(e.to_string()))?,
            v: Tensor::zeros(dims, dtype, device).map_err(|e| ModelError::Build(e.to_string()))?,
        })
    }
}

pub struct QsaAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    pub indexer: QsaIndexer,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    scale: f32,
    eps: f32,
    device: Device,
    compute: DType,
}

impl QsaAttention {
    pub fn load(
        weights: &dyn WeightSource,
        prefix: &str,
        cfg: &Qwen4ExpConfig,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, ModelError> {
        let lin = |name: &str| -> Result<QLinear, ModelError> {
            let key = format!("{prefix}.{name}.weight");
            if let Some(prequant) = weights.quant(&key, device) {
                return Ok(QLinear::Quant(prequant?));
            }
            let w = weights.tensor(&key, device, if quant.is_quantized() { DType::F16 } else { compute })?;
            QLinear::build(w, quant, compute)
        };
        Ok(Self {
            q_proj: lin("q_proj")?,
            k_proj: lin("k_proj")?,
            v_proj: lin("v_proj")?,
            o_proj: lin("o_proj")?,
            q_norm: load_one_plus(weights, &format!("{prefix}.q_norm.weight"), device, compute)?,
            k_norm: load_one_plus(weights, &format!("{prefix}.k_norm.weight"), device, compute)?,
            indexer: QsaIndexer::load(weights, &format!("{prefix}.indexer"), cfg, device, compute, quant)?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.rope.rotary_dim,
            scale: cfg.attn_scale(),
            eps: cfg.rms_norm_eps,
            device,
            compute,
        })
    }

    fn partial_rope(&self, x: &Tensor, rope: &RopeCache, past: usize, s: usize) -> Result<Tensor, ModelError> {
        if self.rotary_dim == 0 {
            return Ok(x.clone());
        }
        if self.rotary_dim == self.head_dim {
            return coerr(apply_rope_range(x, rope, past, s, RopeLayout::Split));
        }
        let head = coerr(coerr(x.narrow(3, 0, self.rotary_dim))?.contiguous())?;
        let tail = coerr(coerr(x.narrow(3, self.rotary_dim, self.head_dim - self.rotary_dim))?
            .contiguous())?;
        let rotated = coerr(apply_rope_range(&head, rope, past, s, RopeLayout::Split))?;
        coerr(Tensor::cat(&[&rotated, &tail], 3))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        h: &Tensor,
        kv: &mut KvLayer,
        idx: &mut IndexerCache,
        past: usize,
        s: usize,
        rope: &RopeCache,
    ) -> Result<(Tensor, Option<Vec<Vec<u32>>>), ModelError> {
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let selected = stage("qsa:indexer", || self.indexer.forward(h, idx, past, s, rope))?;

        let qg = self.q_proj.forward(h)?;
        let qg = coerr(qg.reshape(vec![1, s, nh, 2 * hd]))?;
        let q = coerr(coerr(qg.narrow(3, 0, hd))?.contiguous())?;
        let gate = coerr(coerr(qg.narrow(3, hd, hd))?.contiguous())?;
        let q = rms(&q, &self.q_norm, self.eps)?;
        let q = coerr(coerr(q.permute(vec![0, 2, 1, 3]))?.contiguous())?;

        let k = coerr(self.k_proj.forward(h)?.reshape(vec![1, s, nkv, hd]))?;
        let k = rms(&k, &self.k_norm, self.eps)?;
        let k = coerr(coerr(k.permute(vec![0, 2, 1, 3]))?.contiguous())?;
        let v = coerr(coerr(coerr(self.v_proj.forward(h)?.reshape(vec![1, s, nkv, hd]))?
            .permute(vec![0, 2, 1, 3]))?
            .contiguous())?;

        let q = self.partial_rope(&q, rope, past, s)?;
        let k = self.partial_rope(&k, rope, past, s)?;

        let cap = kv.k.dims()[2];
        if past + s > cap {
            return Err(ModelError::Shape(format!(
                "KV overflow: {} + {s} > {cap}",
                past
            )));
        }
        kv.k.kv_append_inplace(&k, past).map_err(|e| ModelError::Forward(e.to_string()))?;
        kv.v.kv_append_inplace(&v, past).map_err(|e| ModelError::Forward(e.to_string()))?;

        let kv_len = past + s;
        let k_all = coerr(kv.k.narrow(2, 0, kv_len))?;
        let v_all = coerr(kv.v.narrow(2, 0, kv_len))?;

        let attn = match &selected {
            None if s == 1 => {
                let k_rep = repeat_kv(&k_all, nh / nkv)?;
                let v_rep = repeat_kv(&v_all, nh / nkv)?;
                coerr(scaled_dot_attention(&q, &k_rep, &v_rep, self.scale, None))?
            }
            None => {
                let flashed = match q.flash_attention(&k_all, &v_all, self.scale, true) {
                    Ok(a) => Some(a),
                    Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => None,
                    Err(e) => return Err(ModelError::Forward(e.to_string())),
                };
                match flashed {
                    Some(a) => a,
                    None => {
                        let k_rep = repeat_kv(&k_all, nh / nkv)?;
                        let v_rep = repeat_kv(&v_all, nh / nkv)?;
                        let mask = self.causal_mask(s, kv_len, past)?;
                        coerr(scaled_dot_attention(&q, &k_rep, &v_rep, self.scale, Some(&mask)))?
                    }
                }
            }
            Some(sel) if gather_selected() => {
                stage("qsa:sparse", || self.sparse_attention(&q, kv, sel))?
            }
            Some(sel) => stage("qsa:masked", || {
                let k_rep = repeat_kv(&k_all, nh / nkv)?;
                let v_rep = repeat_kv(&v_all, nh / nkv)?;
                let mask = self.selection_mask(sel, s, kv_len)?;
                coerr(scaled_dot_attention(&q, &k_rep, &v_rep, self.scale, Some(&mask)))
            })?,
        };

        let attn = coerr(coerr(attn.permute(vec![0, 2, 1, 3]))?.contiguous())?;
        let attn = coerr(attn.mul(&coerr(gate.sigmoid())?))?;
        let attn = coerr(attn.reshape(vec![s, nh * hd]))?;
        Ok((self.o_proj.forward(&attn)?, selected))
    }

    fn causal_mask(&self, s: usize, kv_len: usize, past: usize) -> Result<Tensor, ModelError> {
        let mut data = vec![0f32; s * kv_len];
        for i in 0..s {
            let qi = past + i;
            for (j, cell) in data[i * kv_len..(i + 1) * kv_len].iter_mut().enumerate() {
                if j > qi {
                    *cell = MASK_NEG;
                }
            }
        }
        coerr(coerr(Tensor::from_vec(data, vec![s, kv_len], self.device))?.to_dtype(self.compute))
    }

    fn selection_mask(&self, selected: &[Vec<u32>], s: usize, kv_len: usize) -> Result<Tensor, ModelError> {
        let mut data = vec![MASK_NEG; s * kv_len];
        for (i, row) in selected.iter().enumerate() {
            let base = i * kv_len;
            for t in row {
                let t = *t as usize;
                if t < kv_len {
                    data[base + t] = 0.0;
                }
            }
        }
        coerr(coerr(Tensor::from_vec(data, vec![s, kv_len], self.device))?.to_dtype(self.compute))
    }

    /// Attention по выбранным индексатором позициям, без построения маски на
    /// всю длину контекста: KV собираются гатером, и каждый запрос считает
    /// свой бюджет (≤ `budget + compress_ratio − 1` позиций) независимо от
    /// того, сколько токенов уже в кэше.
    ///
    /// Запросы группируются по числу выбранных позиций — внутри группы формы
    /// совпадают, поэтому ни паддинга, ни маски не нужно. Группы режутся по
    /// памяти собранного KV.
    fn sparse_attention(
        &self,
        q: &Tensor,
        kv: &KvLayer,
        selected: &[Vec<u32>],
    ) -> Result<Tensor, ModelError> {
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let cap = kv.k.dims()[2];
        let s = selected.len();
        let elem = (self.compute.size_in_bits() / 8).max(1);

        let mut by_len: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for (i, row) in selected.iter().enumerate() {
            by_len.entry(row.len()).or_default().push(i);
        }

        let mut order: Vec<usize> = Vec::with_capacity(s);
        let mut parts: Vec<Tensor> = Vec::new();
        for (m, rows) in by_len {
            if m == 0 {
                return Err(ModelError::Forward("QSA: пустой набор позиций".into()));
            }
            let per_query = 2 * nkv * m * hd * elem;
            let group = (SPARSE_KV_BUDGET / per_query.max(1)).clamp(1, rows.len());
            for slice in rows.chunks(group) {
                let g = slice.len();
                // Гатер идёт по KV-буферу как по таблице `[nkv · cap, hd]`:
                // строка головы `h` и позиции `p` лежит по индексу `h·cap + p`.
                // Так подходит быстрое embed-ядро, читающее индексы с карты, —
                // `index_select` копирует строку за строкой и на бюджете в две
                // тысячи позиций стоит дороже самого внимания.
                // Индексы сразу в порядке `[запрос, голова, позиция]` — тогда
                // результат гатера уже нужной формы и транспонировать
                // четверть гигабайта не приходится.
                let mut idx = Vec::with_capacity(g * nkv * m);
                for row in slice {
                    for head in 0..nkv {
                        let base = (head * cap) as u32;
                        idx.extend(selected[*row].iter().map(|p| base + *p));
                    }
                }
                let idx = coerr(Tensor::from_vec(idx, vec![g * nkv * m], self.device))?;
                let gather = |src: &Tensor| -> Result<Tensor, ModelError> {
                    let table = coerr(src.reshape(vec![nkv * cap, hd]))?;
                    let picked = match table.embed_gather(&idx) {
                        Ok(p) => p,
                        Err(SynaptixError::Unsupported(_)) => coerr(table.index_select(0, &idx))?,
                        Err(e) => return Err(ModelError::Forward(e.to_string())),
                    };
                    coerr(picked.reshape(vec![g, nkv, m, hd]))
                };
                let k_sel = gather(&kv.k)?;
                let v_sel = gather(&kv.v)?;

                let rows_idx: Vec<u32> = slice.iter().map(|r| *r as u32).collect();
                let rows_idx = coerr(Tensor::from_vec(rows_idx, vec![g], self.device))?;
                let q_sel = coerr(q.index_select(2, &rows_idx))?;
                let q_sel = coerr(coerr(q_sel.permute(vec![2, 1, 0, 3]))?.contiguous())?;

                let out = match q_sel.flash_attention(&k_sel, &v_sel, self.scale, false) {
                    Ok(a) => a,
                    Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                        let k_rep = repeat_kv(&k_sel, nh / nkv)?;
                        let v_rep = repeat_kv(&v_sel, nh / nkv)?;
                        coerr(scaled_dot_attention(&q_sel, &k_rep, &v_rep, self.scale, None))?
                    }
                    Err(e) => return Err(ModelError::Forward(e.to_string())),
                };
                order.extend_from_slice(slice);
                parts.push(coerr(out.reshape(vec![g, nh, hd]))?);
            }
        }

        let stacked = if parts.len() == 1 {
            parts.pop().expect("одна часть")
        } else {
            let refs: Vec<&Tensor> = parts.iter().collect();
            coerr(Tensor::cat(&refs, 0))?
        };
        let mut inverse = vec![0u32; s];
        for (place, row) in order.iter().enumerate() {
            inverse[*row] = place as u32;
        }
        let inverse = coerr(Tensor::from_vec(inverse, vec![s], self.device))?;
        let restored = coerr(stacked.index_select(0, &inverse))?;
        coerr(coerr(coerr(restored.permute(vec![1, 0, 2]))?.contiguous())?
            .reshape(vec![1, nh, s, hd]))
    }
}

fn repeat_kv(x: &Tensor, group: usize) -> Result<Tensor, ModelError> {
    if group == 1 {
        return coerr(x.contiguous());
    }
    let dims = x.dims();
    let (b, n_kv, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    let x_un = coerr(x.unsqueeze(2))?;
    let reps = coerr(Tensor::zeros(vec![b, n_kv, group, s, d], x.dtype(), x.device()))?;
    let x_b = coerr(x_un.broadcast_add(&reps))?;
    coerr(x_b.reshape(vec![b, n_kv * group, s, d]))
}
