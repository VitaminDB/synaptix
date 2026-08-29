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
use crate::qsa::{IndexerCache, QsaIndexer, Selection};

const MASK_NEG: f32 = -1.0e4;

/// Потолок памяти на собранные K/V одной группы запросов.
const SPARSE_KV_BUDGET: usize = 512 << 20;

/// Считать ли выбранные позиции гатером — работа тогда растёт с бюджетом
/// индексатора, а не с длиной контекста. `SYN_QWEN4EXP_QSA_GATHER=0`
/// возвращает прежний путь с маской поверх полного внимания.
fn gather_selected() -> bool {
    std::env::var("SYN_QWEN4EXP_QSA_GATHER").map(|v| v.trim() != "0").unwrap_or(true)
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
        // Ёмкость округляется вверх: блоки индексатора собираются из KV
        // строками по `compress_ratio` позиций, и хвост должен быть целым.
        let capacity = capacity.next_multiple_of(8);
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
    ) -> Result<(Tensor, Option<Selection>), ModelError> {
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

    fn selection_mask(&self, selected: &Selection, s: usize, kv_len: usize) -> Result<Tensor, ModelError> {
        let mut data = vec![MASK_NEG; s * kv_len];
        for i in 0..s {
            let base = i * kv_len;
            for t in selected.positions(i)? {
                let t = t as usize;
                if t < kv_len {
                    data[base + t] = 0.0;
                }
            }
        }
        coerr(coerr(Tensor::from_vec(data, vec![s, kv_len], self.device))?.to_dtype(self.compute))
    }

    /// Attention по выбранным индексатором позициям, без построения маски на
    /// всю длину контекста: KV собираются гатером, и работа растёт с бюджетом
    /// индексатора, а не с длиной контекста.
    ///
    /// Запросы идут тайлами подряд: соседние позиции выбирают почти одни и те
    /// же блоки, поэтому на тайл собирается объединение их наборов, а кто
    /// какие позиции видит, задаёт маска. На префилле это убирает почти весь
    /// трафик гатера — каждый запрос тянул свои две тысячи позиций отдельно.
    /// Прежний путь: запросы группируются по числу выбранных позиций, внутри
    /// группы формы совпадают, так что ни паддинга, ни маски не нужно, и
    /// работает flash. Каждому запросу собирается свой KV — на длинном
    /// контексте это дешевле, чем общее на всех объединение.
    fn attention_per_query(
        &self,
        q: &Tensor,
        kv: &KvLayer,
        selected: &Selection,
        offset: usize,
        len: usize,
    ) -> Result<Tensor, ModelError> {
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let cap = kv.k.dims()[2];
        let cr = selected.ratio.max(1);
        let elem = (self.compute.size_in_bits() / 8).max(1);
        if cap % cr != 0 {
            return Err(ModelError::Shape(format!(
                "QSA: ёмкость KV {cap} не кратна блоку {cr}"
            )));
        }
        let block_rows = cap / cr;
        let nb = selected.topk;

        let mut parts: Vec<Tensor> = Vec::new();
        let per_query = 2 * nkv * (nb * cr + cr) * hd * elem;
        let group = (SPARSE_KV_BUDGET / per_query.max(1)).clamp(1, len);
        let mut start = offset;
        while start < offset + len {
            let g = group.min(offset + len - start);
            let q_sel = coerr(coerr(q.narrow(2, start, g))?.contiguous())?;
            let q_sel = coerr(coerr(coerr(q_sel.reshape(vec![nh, g, hd]))?.permute(vec![1, 0, 2]))?
                .contiguous())?;

            // Ядро читает KV прямо по таблице блоков — она уже на карте, и
            // собранного буфера не нужно вовсе. На горстке запросов сетка из
            // `g · nkv` блоков карту не загружает, и там остаётся сборка.
            if nb > 0 && g >= BLOCK_KERNEL_MIN {
                let k3 = coerr(kv.k.reshape(vec![nkv, cap, hd]))?;
                let v3 = coerr(kv.v.reshape(vec![nkv, cap, hd]))?;
                match q_sel.flash_attention_blocks(
                    &k3,
                    &v3,
                    &selected.blocks,
                    &selected.tail_from,
                    &selected.tail_len,
                    cr,
                    self.scale,
                    start,
                ) {
                    Ok(out) => {
                        parts.push(out);
                        start += g;
                        continue;
                    }
                    Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
                    Err(e) => return Err(ModelError::Forward(e.to_string())),
                }
            }

            parts.push(self.gathered_attention(&q_sel, kv, selected, start, g, block_rows)?);
            start += g;
        }

        let stacked = if parts.len() == 1 {
            parts.pop().expect("одна часть")
        } else {
            let refs: Vec<&Tensor> = parts.iter().collect();
            coerr(Tensor::cat(&refs, 0))?
        };
        coerr(coerr(coerr(stacked.reshape(vec![len, nh, hd]))?.permute(vec![1, 0, 2]))?
            .contiguous())
            .and_then(|t| coerr(t.reshape(vec![1, nh, len, hd])))
    }

    /// Запасной путь: позиции собираются гатером, дальше обычный flash. Нужен
    /// там, где ядро по таблице неприменимо — на горстке запросов и на декоде.
    #[allow(clippy::too_many_arguments)]
    fn gathered_attention(
        &self,
        q_sel: &Tensor,
        kv: &KvLayer,
        selected: &Selection,
        offset: usize,
        g: usize,
        block_rows: usize,
    ) -> Result<Tensor, ModelError> {
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let cap = kv.k.dims()[2];
        let cr = selected.ratio.max(1);
        let host = selected.host_blocks()?;
        let tails = selected.tails();

        let mut parts: Vec<Tensor> = Vec::new();
        let mut order: Vec<usize> = Vec::with_capacity(g);
        let mut by_len: std::collections::BTreeMap<(usize, usize), Vec<usize>> = Default::default();
        for i in offset..offset + g {
            by_len.entry((host[i].len(), tails[i].1 as usize)).or_default().push(i);
        }
        for ((nb, tail), rows) in by_len {
            let m = nb * cr + tail;
            if m == 0 {
                return Err(ModelError::Forward("QSA: пустой набор позиций".into()));
            }
            let mut block_idx = Vec::with_capacity(rows.len() * nkv * nb);
            let mut tail_idx = Vec::with_capacity(rows.len() * nkv * tail);
            for row in &rows {
                for head in 0..nkv {
                    let base = (head * block_rows) as u32;
                    block_idx.extend(host[*row].iter().map(|b| base + *b));
                    let (from, count) = tails[*row];
                    let base = (head * cap) as u32;
                    tail_idx.extend((0..count).map(|j| base + from + j));
                }
            }
            let n = rows.len();
            let block_idx = coerr(Tensor::from_vec(block_idx, vec![n * nkv * nb], self.device))?;
            let tail_idx = coerr(Tensor::from_vec(tail_idx, vec![n * nkv * tail], self.device))?;
            let gather = |src: &Tensor| -> Result<Tensor, ModelError> {
                let blocks = coerr(src.reshape(vec![nkv * block_rows, cr * hd]))?;
                let picked = coerr(take_rows(&blocks, &block_idx)?.reshape(vec![n, nkv, nb * cr, hd]))?;
                if tail == 0 {
                    return Ok(picked);
                }
                let table = coerr(src.reshape(vec![nkv * cap, hd]))?;
                let rest = coerr(take_rows(&table, &tail_idx)?.reshape(vec![n, nkv, tail, hd]))?;
                coerr(Tensor::cat(&[&picked, &rest], 2))
            };
            let k_sel = gather(&kv.k)?;
            let v_sel = gather(&kv.v)?;

            let rows_idx: Vec<u32> = rows.iter().map(|r| (*r - offset) as u32).collect();
            let rows_idx = coerr(Tensor::from_vec(rows_idx, vec![n], self.device))?;
            let flat = coerr(q_sel.reshape(vec![g, nh * hd]))?;
            let picked_q = coerr(take_rows(&flat, &rows_idx)?.reshape(vec![n, nh, 1, hd]))?;

            let out = match picked_q.flash_attention(&k_sel, &v_sel, self.scale, false) {
                Ok(a) => a,
                Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                    let k_rep = repeat_kv(&k_sel, nh / nkv)?;
                    let v_rep = repeat_kv(&v_sel, nh / nkv)?;
                    coerr(scaled_dot_attention(&picked_q, &k_rep, &v_rep, self.scale, None))?
                }
                Err(e) => return Err(ModelError::Forward(e.to_string())),
            };
            order.extend(rows.iter().map(|r| *r - offset));
            parts.push(coerr(out.reshape(vec![n, nh, hd]))?);
        }

        let stacked = if parts.len() == 1 {
            parts.pop().expect("одна часть")
        } else {
            let refs: Vec<&Tensor> = parts.iter().collect();
            coerr(Tensor::cat(&refs, 0))?
        };
        let mut inverse = vec![0u32; g];
        for (place, row) in order.iter().enumerate() {
            inverse[*row] = place as u32;
        }
        let inverse = coerr(Tensor::from_vec(inverse, vec![g], self.device))?;
        let flat = coerr(stacked.reshape(vec![g, nh * hd]))?;
        coerr(take_rows(&flat, &inverse)?.reshape(vec![g, nh, hd]))
    }

    fn sparse_attention(
        &self,
        q: &Tensor,
        kv: &KvLayer,
        selected: &Selection,
    ) -> Result<Tensor, ModelError> {
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let cap = kv.k.dims()[2];
        let cr = selected.ratio.max(1);
        let s = selected.len();
        let elem = (self.compute.size_in_bits() / 8).max(1);
        if cap % cr != 0 {
            return Err(ModelError::Shape(format!(
                "QSA: ёмкость KV {cap} не кратна блоку {cr}"
            )));
        }
        let block_rows = cap / cr;
        let limit = TileLimit {
            kv_row: 2 * nkv * cr * hd * elem,
            score_row: nh * cr * 4,
            max_len: qsa_tile(),
            kv_budget: SPARSE_KV_BUDGET,
            score_budget: SPARSE_SCORE_BUDGET,
        };

        // Тайл общего объединения окупается, только пока контекст сам по себе
        // немногим шире бюджета индексатора: иначе соседние запросы смотрят в
        // разные места, объединение растёт до всего контекста и маскированное
        // внимание по нему дороже поштучного пути. Проверка идёт до выгрузки
        // таблицы на хост — на длинном контексте она и не понадобится.
        if selected.blocks_total > TILE_CONTEXT_RATIO * selected.topk {
            let mut parts: Vec<Tensor> = Vec::new();
            let mut start = 0usize;
            while start < s {
                let len = PER_QUERY_SPAN.min(s - start);
                parts.push(self.attention_per_query(q, kv, selected, start, len)?);
                start += len;
            }
            if parts.len() == 1 {
                return Ok(parts.pop().expect("одна часть"));
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            return coerr(Tensor::cat(&refs, 2));
        }

        let host = selected.host_blocks()?;
        let tails = selected.tails();
        let mut parts: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        let mut union = Union::new(block_rows);
        while start < s {
            let len = union.take_tile(host, tails, cr, start, &limit);
            // Общее объединение окупается не всегда: на длинном контексте
            // соседние запросы смотрят в разные места, и внимание по их
            // объединению считает почти весь контекст. Тогда дешевле собрать
            // каждому запросу свой KV — блоками, одним вызовом на группу.
            if !union.worth_tiling(len, &limit) {
                let end = (start + PER_QUERY_SPAN).min(s);
                parts.push(self.attention_per_query(q, kv, selected, start, end - start)?);
                start = end;
                continue;
            }
            let blocks = union.blocks();
            let u = blocks.len();

            let mut idx = Vec::with_capacity(nkv * u);
            for head in 0..nkv {
                let base = (head * block_rows) as u32;
                idx.extend(blocks.iter().map(|b| base + *b));
            }
            let idx = coerr(Tensor::from_vec(idx, vec![nkv * u], self.device))?;
            let gather = |src: &Tensor| -> Result<Tensor, ModelError> {
                let table = coerr(src.reshape(vec![nkv * block_rows, cr * hd]))?;
                coerr(take_rows(&table, &idx)?.reshape(vec![1, nkv, u * cr, hd]))
            };
            let k_sel = gather(&kv.k)?;
            let v_sel = gather(&kv.v)?;

            let q_tile = coerr(coerr(q.narrow(2, start, len))?.contiguous())?;
            let mask = union.mask(host, tails, cr, start, len);
            let mask = coerr(coerr(Tensor::from_vec(mask, vec![len, u * cr], self.device))?
                .to_dtype(self.compute))?;
            let k_rep = repeat_kv(&k_sel, nh / nkv)?;
            let v_rep = repeat_kv(&v_sel, nh / nkv)?;
            let out =
                coerr(scaled_dot_attention(&q_tile, &k_rep, &v_rep, self.scale, Some(&mask)))?;
            parts.push(coerr(out.reshape(vec![1, nh, len, hd]))?);
            start += len;
        }

        if parts.len() == 1 {
            return Ok(parts.pop().expect("одна часть"));
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        coerr(Tensor::cat(&refs, 2))
    }
}

/// Потолок матрицы скоров одного тайла.
const SPARSE_SCORE_BUDGET: usize = 192 << 20;

/// Сколько запросов уходит в поштучный путь за раз, когда тайлы не набираются.
const PER_QUERY_SPAN: usize = 512;

/// Во сколько раз контекст может быть шире бюджета индексатора, чтобы тайл
/// общего объединения ещё окупался.
const TILE_CONTEXT_RATIO: usize = 4;

/// От скольких запросов в группе включается ядро по таблице блоков.
const BLOCK_KERNEL_MIN: usize = 8;

/// Верхний предел длины тайла.
fn qsa_tile() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SYN_QWEN4EXP_QSA_TILE")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(512)
            .max(1)
    })
}

struct TileLimit {
    kv_row: usize,
    score_row: usize,
    max_len: usize,
    kv_budget: usize,
    score_budget: usize,
}

/// Объединение выбранных блоков по тайлу запросов: биты по блокам контекста
/// плюс номер блока внутри объединения. Тайл набирается по одному запросу и
/// закрывается, как только объединение упирается в бюджет памяти.
struct Union {
    bits: Vec<u64>,
    rank: Vec<u32>,
    blocks: Vec<u32>,
    widest: usize,
}

impl Union {
    fn new(total_blocks: usize) -> Self {
        Self {
            bits: vec![0; total_blocks.div_ceil(64)],
            rank: vec![u32::MAX; total_blocks],
            blocks: Vec::new(),
            widest: 0,
        }
    }

    fn set(&mut self, b: u32) -> bool {
        let (w, i) = ((b as usize) / 64, (b as usize) % 64);
        let was = self.bits[w] & (1 << i) != 0;
        self.bits[w] |= 1 << i;
        !was
    }

    fn clear(&mut self, b: u32) {
        let (w, i) = ((b as usize) / 64, (b as usize) % 64);
        self.bits[w] &= !(1u64 << i);
    }

    /// Блоки строки: выбранные индексатором плюс тот, в котором сидит хвост.
    fn row_blocks<'a>(
        blocks: &'a [Vec<u32>],
        tails: &'a [(u32, u32)],
        ratio: u32,
        i: usize,
    ) -> impl Iterator<Item = u32> + 'a {
        let (from, len) = tails[i];
        blocks[i].iter().copied().chain((len > 0).then_some(from / ratio))
    }

    /// Сколько запросов подряд разделят один собранный KV. Запросы
    /// добавляются по одному, пока собранный KV и матрица скоров влезают в
    /// бюджет: чем длиннее контекст, тем шире объединение и тем короче выходит
    /// тайл.
    fn take_tile(
        &mut self,
        blocks: &[Vec<u32>],
        tails: &[(u32, u32)],
        ratio: usize,
        offset: usize,
        limit: &TileLimit,
    ) -> usize {
        let stale = std::mem::take(&mut self.blocks);
        for b in stale {
            self.clear(b);
        }
        let rows = blocks.len() - offset;
        let mut size = 0usize;
        let mut widest = 0usize;
        let mut len = 0usize;
        let mut added: Vec<u32> = Vec::new();
        while len < rows && len < limit.max_len {
            let i = offset + len;
            added.clear();
            let mut row_size = 0usize;
            for b in Self::row_blocks(blocks, tails, ratio as u32, i) {
                row_size += 1;
                if self.set(b) {
                    added.push(b);
                }
            }
            let size_next = size + added.len();
            let widest_next = widest.max(row_size);
            let fits = size_next * limit.kv_row <= limit.kv_budget
                && (len + 1) * size_next * limit.score_row <= limit.score_budget;
            if len > 0 && !fits {
                for b in added.drain(..) {
                    self.clear(b);
                }
                break;
            }
            size = size_next;
            widest = widest_next;
            len += 1;
        }
        self.widest = widest;
        for (w, word) in self.bits.iter().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let b = (w * 64 + i) as u32;
                self.rank[b as usize] = self.blocks.len() as u32;
                self.blocks.push(b);
            }
        }
        len.max(1)
    }

    fn blocks(&self) -> &[u32] {
        &self.blocks
    }

    /// Стоит ли считать тайл общим объединением. Слева — что придётся прочитать
    /// на запрос при общем KV (собранные блоки, поделённые на длину тайла, плюс
    /// матрица скоров, которая пишется, читается и нормируется), справа —
    /// сколько стоит поштучный путь. Он вдвое дешевле своего трафика: там KV
    /// не собирается вовсе, ядро читает его прямо по таблице блоков.
    fn worth_tiling(&self, len: usize, limit: &TileLimit) -> bool {
        if len <= 1 {
            return false;
        }
        let u = self.blocks.len();
        let tiled = u * limit.kv_row / len + u * limit.score_row * 3;
        let per_query = self.widest * limit.kv_row / 2;
        tiled < per_query
    }

    /// Аддитивная маска `[len, |union| · ratio]`: запрос видит только свои
    /// блоки, а в блоке с хвостом — только те позиции, что уже есть.
    fn mask(
        &self,
        blocks: &[Vec<u32>],
        tails: &[(u32, u32)],
        ratio: usize,
        offset: usize,
        len: usize,
    ) -> Vec<f32> {
        let width = self.blocks.len() * ratio;
        let mut mask = vec![MASK_NEG; len * width];
        for i in 0..len {
            let row = offset + i;
            let base = i * width;
            for b in &blocks[row] {
                let j = self.rank[*b as usize];
                if j == u32::MAX {
                    continue;
                }
                let from = base + j as usize * ratio;
                mask[from..from + ratio].iter_mut().for_each(|c| *c = 0.0);
            }
            let (from, count) = tails[row];
            if count == 0 {
                continue;
            }
            let j = self.rank[(from / ratio as u32) as usize];
            if j == u32::MAX {
                continue;
            }
            let start = base + j as usize * ratio;
            mask[start..start + count as usize].iter_mut().for_each(|c| *c = 0.0);
        }
        mask
    }
}

fn take_rows(src: &Tensor, ids: &Tensor) -> Result<Tensor, ModelError> {
    match src.embed_gather(ids) {
        Ok(t) => Ok(t),
        Err(SynaptixError::Unsupported(_)) => coerr(src.index_select(0, ids)),
        Err(e) => Err(ModelError::Forward(e.to_string())),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(spec: Vec<(Vec<u32>, (u32, u32))>) -> (Vec<Vec<u32>>, Vec<(u32, u32)>) {
        spec.into_iter().unzip()
    }

    fn limit(max_len: usize, score_budget: usize) -> TileLimit {
        TileLimit {
            kv_row: 1,
            score_row: 1,
            max_len,
            kv_budget: SPARSE_KV_BUDGET,
            score_budget,
        }
    }

    #[test]
    fn tile_covers_every_selected_block() {
        let (blocks, tails) = rows(vec![
            (vec![0, 2], (12, 2)),
            (vec![2, 5], (24, 0)),
            (vec![9], (40, 1)),
        ]);
        let mut union = Union::new(64);
        let len = union.take_tile(&blocks, &tails, 4, 0, &limit(8, 1 << 20));
        assert_eq!(len, 3);
        assert_eq!(union.blocks(), &[0, 2, 3, 5, 9, 10]);

        let mask = union.mask(&blocks, &tails, 4, 0, len);
        let width = union.blocks().len() * 4;
        for i in 0..len {
            let mut allowed: Vec<u32> = Vec::new();
            for b in &blocks[i] {
                allowed.extend(b * 4..b * 4 + 4);
            }
            let (from, count) = tails[i];
            allowed.extend(from..from + count);
            for (j, b) in union.blocks().iter().enumerate() {
                for t in 0..4u32 {
                    let open = mask[i * width + j * 4 + t as usize] == 0.0;
                    assert_eq!(open, allowed.contains(&(b * 4 + t)), "строка {i}, блок {b}");
                }
            }
        }
    }

    #[test]
    fn tile_stops_on_budget_and_resumes() {
        let (blocks, tails) = rows(
            (0..8).map(|i| (vec![i as u32 * 3, i as u32 * 3 + 1], (100, 0))).collect(),
        );
        let mut union = Union::new(64);
        let first = union.take_tile(&blocks, &tails, 4, 0, &limit(8, 8));
        assert!(first >= 1 && first < blocks.len(), "тайл {first} не ограничен бюджетом");
        let rest = union.take_tile(&blocks, &tails, 4, first, &limit(8, 8));
        assert!(rest >= 1);
    }

    #[test]
    fn single_query_tile_is_not_worth_tiling() {
        let (blocks, tails) = rows(vec![(vec![1, 3], (16, 2))]);
        let mut union = Union::new(64);
        let lim = limit(512, 1 << 20);
        assert_eq!(union.take_tile(&blocks, &tails, 4, 0, &lim), 1);
        assert!(!union.worth_tiling(1, &lim), "тайл из одного запроса не нужен");
    }

    #[test]
    fn wide_union_is_not_worth_tiling() {
        // Наборы не пересекаются: объединение растёт как сумма, и общий KV
        // читается почти целиком каждым запросом.
        let (blocks, tails) = rows(
            (0..8).map(|i| (vec![i as u32 * 2, i as u32 * 2 + 1], (200, 0))).collect(),
        );
        let mut union = Union::new(64);
        let lim = TileLimit {
            kv_row: 1024,
            score_row: 64,
            max_len: 512,
            kv_budget: SPARSE_KV_BUDGET,
            score_budget: SPARSE_SCORE_BUDGET,
        };
        let len = union.take_tile(&blocks, &tails, 4, 0, &lim);
        assert_eq!(len, 8);
        assert!(!union.worth_tiling(len, &lim));
    }
}
