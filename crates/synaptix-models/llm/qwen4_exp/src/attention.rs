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
    /// всю длину контекста: KV собираются гатером, и работа растёт с бюджетом
    /// индексатора, а не с длиной контекста.
    ///
    /// Запросы идут тайлами подряд: соседние позиции выбирают почти одни и те
    /// же блоки, поэтому на тайл собирается объединение их наборов, а кто
    /// какие позиции видит, задаёт маска. На префилле это убирает почти весь
    /// трафик гатера — каждый запрос тянул свои две тысячи позиций отдельно.
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
        let limit = TileLimit {
            kv_row: 2 * nkv * hd * elem,
            score_row: nh * 4,
            max_len: qsa_tile(),
            kv_budget: SPARSE_KV_BUDGET,
            score_budget: SPARSE_SCORE_BUDGET,
            spread: qsa_spread(),
        };

        let mut parts: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        let mut union = Union::new(cap);
        while start < s {
            let len = union.take_tile(&selected[start..], &limit);
            let positions = union.positions();
            let u = positions.len();
            if u == 0 {
                return Err(ModelError::Forward("QSA: пустой набор позиций".into()));
            }

            // Гатер идёт по KV-буферу как по таблице `[nkv · cap, hd]`: строка
            // головы `h` и позиции `p` лежит по индексу `h·cap + p`. Так
            // подходит быстрое embed-ядро, читающее индексы с карты, —
            // `index_select` копирует строку за строкой и стоит дороже самого
            // внимания.
            let mut idx = Vec::with_capacity(nkv * u);
            for head in 0..nkv {
                let base = (head * cap) as u32;
                idx.extend(positions.iter().map(|p| base + *p));
            }
            let idx = coerr(Tensor::from_vec(idx, vec![nkv * u], self.device))?;
            let gather = |src: &Tensor| -> Result<Tensor, ModelError> {
                let table = coerr(src.reshape(vec![nkv * cap, hd]))?;
                let picked = match table.embed_gather(&idx) {
                    Ok(p) => p,
                    Err(SynaptixError::Unsupported(_)) => coerr(table.index_select(0, &idx))?,
                    Err(e) => return Err(ModelError::Forward(e.to_string())),
                };
                coerr(picked.reshape(vec![1, nkv, u, hd]))
            };
            let k_sel = gather(&kv.k)?;
            let v_sel = gather(&kv.v)?;

            let q_tile = coerr(coerr(q.narrow(2, start, len))?.contiguous())?;
            // Все запросы тайла смотрят на всё объединение — маска не нужна,
            // и тогда работает flash-путь (так идёт декод: тайл из одного).
            let full = selected[start..start + len].iter().all(|row| row.len() == u);
            let out = if full {
                match q_tile.flash_attention(&k_sel, &v_sel, self.scale, false) {
                    Ok(a) => a,
                    Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                        let k_rep = repeat_kv(&k_sel, nh / nkv)?;
                        let v_rep = repeat_kv(&v_sel, nh / nkv)?;
                        coerr(scaled_dot_attention(&q_tile, &k_rep, &v_rep, self.scale, None))?
                    }
                    Err(e) => return Err(ModelError::Forward(e.to_string())),
                }
            } else {
                let mask = union.mask(&selected[start..start + len]);
                let mask = coerr(coerr(Tensor::from_vec(mask, vec![len, u], self.device))?
                    .to_dtype(self.compute))?;
                let k_rep = repeat_kv(&k_sel, nh / nkv)?;
                let v_rep = repeat_kv(&v_sel, nh / nkv)?;
                coerr(scaled_dot_attention(&q_tile, &k_rep, &v_rep, self.scale, Some(&mask)))?
            };
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

/// Во сколько раз объединение тайла может быть шире одного набора.
fn qsa_spread() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SYN_QWEN4EXP_QSA_SPREAD")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(4)
            .max(1)
    })
}

/// Сколько подряд идущих запросов делят один собранный KV. На промпте в 6k
/// токенов сборка выбранных позиций стоила 9.3 с поштучно, 4.9 с тайлами по
/// 128 и 2.1 с тайлами по 512; дальше объединение разрастается и внимание
/// начинает считать лишнее.
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

/// Потолок матрицы скоров одного тайла.
const SPARSE_SCORE_BUDGET: usize = 192 << 20;

struct TileLimit {
    kv_row: usize,
    score_row: usize,
    max_len: usize,
    kv_budget: usize,
    score_budget: usize,
    spread: usize,
}

/// Объединение наборов позиций по тайлу запросов: биты по позициям контекста
/// плюс номер позиции внутри объединения. Тайл набирается по одному запросу и
/// закрывается, как только объединение перестаёт окупаться — считать по нему
/// внимание дороже, чем собрать KV каждому запросу отдельно.
struct Union {
    bits: Vec<u64>,
    rank: Vec<u32>,
    positions: Vec<u32>,
}

impl Union {
    fn new(cap: usize) -> Self {
        Self {
            bits: vec![0; cap.div_ceil(64)],
            rank: vec![u32::MAX; cap],
            positions: Vec::new(),
        }
    }

    fn set(&mut self, p: u32) -> bool {
        let (w, b) = ((p as usize) / 64, (p as usize) % 64);
        let was = self.bits[w] & (1 << b) != 0;
        self.bits[w] |= 1 << b;
        !was
    }

    fn clear(&mut self, p: u32) {
        let (w, b) = ((p as usize) / 64, (p as usize) % 64);
        self.bits[w] &= !(1u64 << b);
    }

    /// Сколько запросов подряд разделят один собранный KV. Запросы
    /// добавляются по одному, пока собранный KV и матрица скоров влезают в
    /// бюджет: чем длиннее контекст, тем шире объединение и тем короче выходит
    /// тайл — на 6k это около пятисот запросов, на 35k несколько десятков.
    fn take_tile(&mut self, rows: &[Vec<u32>], limit: &TileLimit) -> usize {
        let stale = std::mem::take(&mut self.positions);
        for p in stale {
            self.clear(p);
        }
        let mut size = 0usize;
        let mut widest = 0usize;
        let mut len = 0usize;
        let mut added: Vec<u32> = Vec::new();
        while len < rows.len() && len < limit.max_len {
            let row = &rows[len];
            added.clear();
            for p in row {
                if self.set(*p) {
                    added.push(*p);
                }
            }
            let size_next = size + added.len();
            let widest_next = widest.max(row.len());
            // Объединение шире учетверённого набора значит, что запросы тайла
            // смотрят в разные места: маска отбросит больше, чем оставит, а
            // внимание посчитает почти весь контекст — тайл тогда не нужен.
            let fits = size_next * limit.kv_row <= limit.kv_budget
                && (len + 1) * size_next * limit.score_row <= limit.score_budget
                && size_next <= limit.spread * widest_next.max(1);
            if len > 0 && !fits {
                for p in added.drain(..) {
                    self.clear(p);
                }
                break;
            }
            size = size_next;
            widest = widest_next;
            len += 1;
        }
        for (w, word) in self.bits.iter().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let p = (w * 64 + b) as u32;
                self.rank[p as usize] = self.positions.len() as u32;
                self.positions.push(p);
            }
        }
        len.max(1)
    }

    fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// Аддитивная маска `[len, |union|]`: запрос видит только свои позиции.
    fn mask(&self, rows: &[Vec<u32>]) -> Vec<f32> {
        let u = self.positions.len();
        let mut mask = vec![MASK_NEG; rows.len() * u];
        for (i, row) in rows.iter().enumerate() {
            let base = i * u;
            for p in row {
                let j = self.rank[*p as usize];
                if j != u32::MAX {
                    mask[base + j as usize] = 0.0;
                }
            }
        }
        mask
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

    fn limit(max_len: usize, score_row: usize) -> TileLimit {
        TileLimit {
            kv_row: 1,
            score_row,
            max_len,
            kv_budget: SPARSE_KV_BUDGET,
            score_budget: SPARSE_SCORE_BUDGET,
            spread: 64,
        }
    }

    #[test]
    fn tile_covers_every_selected_position() {
        let rows = vec![
            vec![0u32, 4, 8],
            vec![4u32, 12],
            vec![100u32],
        ];
        let mut union = Union::new(128);
        let len = union.take_tile(&rows, &limit(8, 1));
        assert_eq!(len, 3);
        assert_eq!(union.positions(), &[0, 4, 8, 12, 100]);

        let mask = union.mask(&rows);
        let u = union.positions().len();
        for (i, row) in rows.iter().enumerate() {
            for (j, p) in union.positions().iter().enumerate() {
                let open = mask[i * u + j] == 0.0;
                assert_eq!(open, row.contains(p), "строка {i}, позиция {p}");
            }
        }
    }

    #[test]
    fn tile_stops_on_budget_and_resumes() {
        let rows: Vec<Vec<u32>> = (0..8).map(|i| vec![i as u32 * 10, i as u32 * 10 + 1]).collect();
        let mut union = Union::new(128);
        // Бюджет скоров пускает только пару запросов: 2 * 4 позиции * 1 = 8.
        let lim = TileLimit {
            kv_row: 1,
            score_row: 1,
            max_len: 8,
            kv_budget: SPARSE_KV_BUDGET,
            score_budget: 8,
            spread: 64,
        };
        let first = union.take_tile(&rows, &lim);
        assert!(first >= 1 && first < rows.len(), "тайл {first} не ограничен бюджетом");
        let rest = union.take_tile(&rows[first..], &lim);
        assert!(rest >= 1);
        assert!(union.positions().iter().all(|p| *p >= rows[first][0]));
    }

    #[test]
    fn single_query_tile_keeps_its_own_positions() {
        let rows = vec![vec![3u32, 9, 27]];
        let mut union = Union::new(64);
        assert_eq!(union.take_tile(&rows, &limit(512, 4)), 1);
        assert_eq!(union.positions(), &[3, 9, 27]);
    }
}
