use rayon::prelude::*;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::RopePositions;
use synaptix_llm_common::{ModelError, QLinear, WeightSource};
use synaptix_ops::pos::rope::{apply_rope_with_cossin, RopeLayout};
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::config::{IndexerConfig, Qwen4ExpConfig};
use crate::norm::{coerr, load_one_plus, rms, stage};

/// Выбор индексатора: блоки, а не позиции. Блок — `compress_ratio` подряд
/// идущих токенов, и в KV они лежат сплошняком, поэтому и собирать их надо
/// блоками. Таблица живёт на карте — там же, где её посчитали, и туда же её
/// читает ядро внимания; на хост она выгружается только по требованию
/// (тайловый путь и трассировка).
pub struct Selection {
    pub ratio: usize,
    pub topk: usize,
    pub rows: usize,
    /// Сколько всего блоков в контексте на момент выбора.
    pub blocks_total: usize,
    /// `[rows, topk]` u32; пустой слот помечен `u32::MAX`.
    pub blocks: Tensor,
    /// `[rows]` u32 — начало и длина хвоста (токены после последнего блока).
    pub tail_from: Tensor,
    pub tail_len: Tensor,
    tails_host: Vec<(u32, u32)>,
    host: std::sync::OnceLock<Vec<Vec<u32>>>,
}

impl Selection {
    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn tails(&self) -> &[(u32, u32)] {
        &self.tails_host
    }

    /// Таблица блоков на хосте. Считается один раз по требованию.
    pub fn host_blocks(&self) -> Result<&Vec<Vec<u32>>, ModelError> {
        if let Some(v) = self.host.get() {
            return Ok(v);
        }
        let flat = self
            .blocks
            .to_device(Device::Cpu)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<u32>())
            .map_err(|e| ModelError::Forward(format!("QSA: выгрузка таблицы: {e}")))?;
        let rows: Vec<Vec<u32>> = flat
            .chunks(self.topk)
            .map(|r| r.iter().copied().filter(|b| *b != u32::MAX).collect())
            .collect();
        Ok(self.host.get_or_init(|| rows))
    }

    pub fn row_len(&self, i: usize) -> Result<usize, ModelError> {
        Ok(self.host_blocks()?[i].len() * self.ratio + self.tails_host[i].1 as usize)
    }

    /// Позиции строки по порядку: сперва выбранные блоки, потом хвост.
    pub fn positions(&self, i: usize) -> Result<Vec<u32>, ModelError> {
        let blocks = &self.host_blocks()?[i];
        let mut out = Vec::with_capacity(blocks.len() * self.ratio);
        for b in blocks {
            let start = b * self.ratio as u32;
            out.extend(start..start + self.ratio as u32);
        }
        let (from, len) = self.tails_host[i];
        out.extend(from..from + len);
        Ok(out)
    }
}

/// Снимок хвоста индексатора для сессии префикс-KV: счётчики и содержимое
/// `pending` (host-копия — переезд кэша в RAM его не касается).
#[derive(Clone)]
pub struct IndexerTail {
    blocks: usize,
    rows: usize,
    data: Vec<f32>,
}

pub struct IndexerCache {
    /// Ключи, не набравшие полный блок. Живут на карте: раньше они уезжали
    /// на хост целиком, сворачивались там циклом и возвращались обратно.
    pending: Tensor,
    pending_rows: usize,
    block_keys: Tensor,
    blocks: usize,
    capacity: usize,
    head_dim: usize,
    ratio: usize,
}

impl IndexerCache {
    pub fn new(
        capacity_blocks: usize,
        head_dim: usize,
        ratio: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self, ModelError> {
        let block_keys = Tensor::zeros(vec![1, 1, capacity_blocks.max(1), head_dim], dtype, device)
            .map_err(|e| ModelError::Build(e.to_string()))?;
        let ratio = ratio.max(1);
        let pending = Tensor::zeros(vec![1, ratio.max(2) - 1, head_dim], DType::F32, device)
            .map_err(|e| ModelError::Build(e.to_string()))?;
        Ok(Self {
            pending,
            pending_rows: 0,
            block_keys,
            blocks: 0,
            capacity: capacity_blocks.max(1),
            head_dim,
            ratio,
        })
    }

    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// Метка состояния: сколько блоков собрано и сколько сырых ключей ждёт
    /// в хвосте. По ней спекулятивный шаг откатывается.
    pub fn mark(&self) -> (usize, usize) {
        (self.blocks, self.pending_rows)
    }

    /// Какой будет метка через `n` токенов: ключи копятся в хвосте и каждые
    /// `ratio` штук сворачиваются в блок, так что считать её можно наперёд —
    /// прогону пары это избавляет от лишнего прохода индексатора.
    pub fn mark_after(&self, n: usize) -> (usize, usize) {
        let waiting = self.pending_rows + n;
        let new_blocks = waiting / self.ratio;
        (self.blocks + new_blocks, waiting % self.ratio)
    }

    /// Откат по метке восстанавливает счётчики ровно — не `min`: после
    /// свёртки текущий `pending_rows` бывает МЕНЬШЕ откатываемого, и `min`
    /// оставлял счётчик на свёрнутом значении. Дальше сетка блоков ехала
    /// относительно позиций токенов, а выбор блоков превращался в шум.
    pub fn rewind(&mut self, mark: (usize, usize)) {
        let (blocks, rows) = mark;
        self.blocks = blocks;
        self.pending_rows = rows;
    }

    /// Полный снимок хвоста: счётчики плюс СОДЕРЖИМОЕ ключей, не набравших
    /// блок. `mark`/`rewind` спасают только спекулятивную пару — она
    /// откатывается сразу и содержимое хвоста ниже метки ещё не перезаписано.
    /// Сессии префикс-KV этого мало: между снимком в конце промпта и
    /// рестором следующего хода декод многократно сворачивает `pending`,
    /// и ключи хвоста промпта затираются ключами сгенерированных токенов.
    pub fn tail_snapshot(&self) -> Result<IndexerTail, ModelError> {
        let data = if self.pending_rows == 0 {
            Vec::new()
        } else {
            coerr(self.pending.narrow(1, 0, self.pending_rows))?
                .contiguous()
                .and_then(|t| t.to_device(Device::Cpu))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| ModelError::Forward(e.to_string()))?
        };
        Ok(IndexerTail { blocks: self.blocks, rows: self.pending_rows, data })
    }

    /// Обратная часть [`Self::tail_snapshot`]: вернуть и счётчики, и ключи.
    pub fn restore_tail(&mut self, tail: &IndexerTail) -> Result<(), ModelError> {
        if tail.rows > 0 {
            let src = Tensor::from_vec(
                tail.data.clone(),
                vec![1, tail.rows, self.head_dim],
                self.pending.device(),
            )
            .map_err(|e| ModelError::Forward(e.to_string()))?;
            coerr(self.pending.copy_rows_from(0, &src))?;
        }
        self.blocks = tail.blocks;
        self.pending_rows = tail.rows;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.pending_rows = 0;
        self.blocks = 0;
    }

    /// Переселить буферы индексатора в host-RAM (см.
    /// `ModelCache::park_to_host`). Позиции (`blocks`, `pending_rows`) не
    /// трогаются — переезжают только данные.
    pub fn park_to_host(&mut self) -> Result<usize, ModelError> {
        Ok(park_tensor(&mut self.pending)? + park_tensor(&mut self.block_keys)?)
    }

    pub fn unpark_to(&mut self, device: Device) -> Result<usize, ModelError> {
        Ok(unpark_tensor(&mut self.pending, device)?
            + unpark_tensor(&mut self.block_keys, device)?)
    }

    pub fn is_parked(&self) -> bool {
        self.block_keys.device() == Device::Cpu
    }

    pub fn device_bytes(&self) -> usize {
        [&self.pending, &self.block_keys]
            .into_iter()
            .filter(|t| t.device() != Device::Cpu)
            .map(|t| t.dtype().bytes_for_numel(t.numel()))
            .sum()
    }
}

/// Общие помощники переезда тензора между картой и host-RAM.
pub(crate) fn park_tensor(t: &mut Tensor) -> Result<usize, ModelError> {
    if t.device() == Device::Cpu {
        return Ok(0);
    }
    let bytes = t.dtype().bytes_for_numel(t.numel());
    *t = t
        .to_device(Device::Cpu)
        .map_err(|e| ModelError::Forward(e.to_string()))?;
    Ok(bytes)
}

pub(crate) fn unpark_tensor(t: &mut Tensor, device: Device) -> Result<usize, ModelError> {
    if t.device() == device {
        return Ok(0);
    }
    let bytes = t.dtype().bytes_for_numel(t.numel());
    *t = t
        .to_device(device)
        .map_err(|e| ModelError::Forward(e.to_string()))?;
    Ok(bytes)
}

pub struct QsaIndexer {
    qk_proj: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    cfg: IndexerConfig,
    rotary_dim: usize,
    eps: f32,
    device: Device,
    compute: DType,
}

impl QsaIndexer {
    pub fn load(
        weights: &dyn WeightSource,
        prefix: &str,
        cfg: &Qwen4ExpConfig,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, ModelError> {
        let key = format!("{prefix}.index_qk_proj.weight");
        let qk_proj = if let Some(prequant) = weights.quant(&key, device) {
            QLinear::Quant(prequant?)
        } else {
            let w = weights.tensor(&key, device, if quant.is_quantized() { DType::F16 } else { compute })?;
            QLinear::build(w, quant, compute)?
        };
        Ok(Self {
            qk_proj,
            q_norm: load_one_plus(weights, &format!("{prefix}.q_layernorm.weight"), device, compute)?,
            k_norm: load_one_plus(weights, &format!("{prefix}.k_layernorm.weight"), device, compute)?,
            cfg: cfg.indexer,
            rotary_dim: cfg.rope.rotary_dim,
            eps: cfg.rms_norm_eps,
            device,
            compute,
        })
    }

    pub fn config(&self) -> IndexerConfig {
        self.cfg
    }

    pub fn make_cache(&self, max_seq: usize) -> Result<IndexerCache, ModelError> {
        let blocks = max_seq.div_ceil(self.cfg.compress_ratio) + 1;
        IndexerCache::new(
            blocks,
            self.cfg.head_dim,
            self.cfg.compress_ratio,
            self.device,
            self.compute,
        )
    }

    fn rope(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        positions: &[u32],
        sel: RopePositions,
    ) -> Result<Tensor, ModelError> {
        if self.rotary_dim == 0 {
            return Ok(x.clone());
        }
        let ids = |v: Vec<u32>| -> Result<Tensor, ModelError> {
            let n = v.len();
            coerr(Tensor::from_vec(v, vec![n], self.device))
        };
        let (cos, sin) = match sel {
            RopePositions::Tables { cos, sin } => {
                let idx = ids(positions.to_vec())?;
                (crate::attention::take_rows(cos, &idx)?, crate::attention::take_rows(sin, &idx)?)
            }
            RopePositions::Shifted(delta) => {
                let pos = ids(positions
                    .iter()
                    .map(|p| (*p as i64 + delta).max(0) as u32)
                    .collect())?;
                coerr(rope.select_positions(&pos))?
            }
            RopePositions::Sequential => {
                let pos = ids(positions.to_vec())?;
                coerr(rope.select_positions(&pos))?
            }
        };
        let d = x.dims()[3];
        if self.rotary_dim == d {
            return coerr(apply_rope_with_cossin(x, &cos, &sin, RopeLayout::Split));
        }
        let head = coerr(coerr(x.narrow(3, 0, self.rotary_dim))?.contiguous())?;
        let tail = coerr(coerr(x.narrow(3, self.rotary_dim, d - self.rotary_dim))?.contiguous())?;
        let rotated = coerr(apply_rope_with_cossin(&head, &cos, &sin, RopeLayout::Split))?;
        coerr(Tensor::cat(&[&rotated, &tail], 3))
    }

    /// Свернуть новые ключи в блоки. Ключи, не набравшие полный блок, ждут
    /// следующего вызова прямо на карте: раньше они уезжали на хост целиком,
    /// сворачивались там тройным циклом и возвращались обратно — на длинном
    /// промпте это и трафик, и синхронизация посреди слоя.
    fn push_keys(
        &self,
        cache: &mut IndexerCache,
        keys: &Tensor,
        rope: &RopeCache,
        sel: RopePositions,
    ) -> Result<(), ModelError> {
        let d = self.cfg.head_dim;
        let cr = self.cfg.compress_ratio;
        let fresh = keys.dims()[1];
        let waiting = cache.pending_rows + fresh;
        let new_blocks = waiting / cr;
        let keys = coerr(keys.to_dtype(DType::F32))?;
        let all = if cache.pending_rows == 0 {
            keys
        } else {
            let head =
                coerr(coerr(cache.pending.narrow(1, 0, cache.pending_rows))?.contiguous())?;
            coerr(Tensor::cat(&[&head, &keys], 1))?
        };
        if new_blocks == 0 {
            coerr(cache.pending.copy_rows_from(0, &all))?;
            cache.pending_rows = waiting;
            return Ok(());
        }
        if cache.blocks + new_blocks > cache.capacity {
            return Err(ModelError::Shape(format!(
                "индексатор: блоков {} больше ёмкости {}",
                cache.blocks + new_blocks,
                cache.capacity
            )));
        }
        let taken = new_blocks * cr;
        let full = coerr(coerr(all.narrow(1, 0, taken))?.contiguous())?;
        let pooled = coerr(coerr(coerr(full.reshape(vec![new_blocks, cr, d]))?.sum_keepdim(1))?
            .reshape(vec![new_blocks, d]))?;
        let pooled = coerr(pooled.mul_scalar(1.0 / cr as f32))?;
        let rest = waiting - taken;
        if rest > 0 {
            let tail = coerr(coerr(all.narrow(1, taken, rest))?.contiguous())?;
            coerr(cache.pending.copy_rows_from(0, &tail))?;
        }
        cache.pending_rows = rest;

        let pooled = coerr(pooled.to_dtype(self.compute))?;
        let normed = rms(&pooled, &self.k_norm, self.eps)?;
        let normed = coerr(normed.reshape(vec![1, 1, new_blocks, d]))?;
        let positions: Vec<u32> = (0..new_blocks)
            .map(|b| ((cache.blocks + b) * cr) as u32)
            .collect();
        let roped = self.rope(&normed, rope, &positions, sel)?;
        cache
            .block_keys
            .kv_append_inplace(&roped, cache.blocks)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        cache.blocks += new_blocks;
        Ok(())
    }

    /// Тот же отбор на процессоре: нужен там, где ядра нет (CPU-устройство и
    /// проверки паритета).
    fn topk_on_host(
        &self,
        scores: &Tensor,
        valid: &[u32],
        take: usize,
        total_blocks: usize,
        topk: usize,
    ) -> Result<Tensor, ModelError> {
        let host = scores
            .to_device(Device::Cpu)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let mut table = vec![u32::MAX; take * topk];
        table
            .par_chunks_mut(topk)
            .enumerate()
            .for_each(|(i, slot)| {
                let nb = (valid[i] as usize).min(total_blocks);
                if nb == 0 {
                    return;
                }
                let row = &host[i * total_blocks..i * total_blocks + nb];
                let mut order: Vec<u32> = (0..nb as u32).collect();
                if topk < order.len() {
                    order.select_nth_unstable_by(topk - 1, |a, b| {
                        row[*b as usize]
                            .partial_cmp(&row[*a as usize])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    order.truncate(topk);
                }
                for (s, b) in order.iter().enumerate() {
                    slot[s] = *b;
                }
            });
        coerr(Tensor::from_vec(table, vec![take, topk], self.device))
    }

    pub fn needs_selection(&self, kv_len: usize) -> bool {
        kv_len / self.cfg.compress_ratio > self.cfg.block_topk()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        h: &Tensor,
        cache: &mut IndexerCache,
        past: usize,
        s: usize,
        rope: &RopeCache,
        sel: RopePositions,
    ) -> Result<Option<Selection>, ModelError> {
        let d = self.cfg.head_dim;
        let nh = self.cfg.n_heads;
        let qk = self.qk_proj.forward(h)?;
        let qk = coerr(qk.reshape(vec![s, (nh + self.cfg.kv_heads) * d]))?;
        let q = coerr(coerr(qk.narrow(1, 0, nh * d))?.contiguous())?;
        let k = coerr(coerr(qk.narrow(1, nh * d, self.cfg.kv_heads * d))?.contiguous())?;

        let k_rows = coerr(k.reshape(vec![1, s * self.cfg.kv_heads, d]))?;
        stage("idx:pool", || self.push_keys(cache, &k_rows, rope, sel))?;

        let kv_len = past + s;
        if !self.needs_selection(kv_len) {
            return Ok(None);
        }

        let q = coerr(q.reshape(vec![1, s, nh, d]))?;
        let q = rms(&q, &self.q_norm, self.eps)?;
        let q = coerr(coerr(q.permute(vec![0, 2, 1, 3]))?.contiguous())?;
        let positions: Vec<u32> = (past..past + s).map(|p| p as u32).collect();
        let q = self.rope(&q, rope, &positions, sel)?;
        let q = coerr(coerr(coerr(q.permute(vec![0, 2, 1, 3]))?.contiguous())?
            .reshape(vec![s * nh, d]))?;

        let total_blocks = cache.blocks;
        let keys = coerr(coerr(cache.block_keys.narrow(2, 0, total_blocks))?
            .reshape(vec![total_blocks, d]))?;
        let keys_t = coerr(coerr(keys.t())?.contiguous())?;

        let cr = self.cfg.compress_ratio;
        let topk = self.cfg.block_topk().min(total_blocks);
        let mut tails: Vec<(u32, u32)> = Vec::with_capacity(s);
        let mut valid: Vec<u32> = Vec::with_capacity(s);
        for i in 0..s {
            let visible = past + i + 1;
            let nb = (visible / cr).min(total_blocks);
            valid.push(nb as u32);
            tails.push(((nb * cr) as u32, (visible - nb * cr) as u32));
        }

        // Скоры считаются чанками (матрица `[take, блоков]` на длинном
        // контексте занимает сотни мегабайт), а выбор блоков делает карта:
        // выгрузка строк логитов на хост и отбор на процессоре стоили дороже
        // всего остального в индексаторе.
        let mut parts: Vec<Tensor> = Vec::new();
        let chunk = (1 << 22) / total_blocks.max(1);
        let chunk = chunk.clamp(1, 256).min(s);
        let mut row = 0usize;
        while row < s {
            let take = chunk.min(s - row);
            let qs = coerr(coerr(q.narrow(0, row * nh, take * nh))?.contiguous())?;
            let scores = coerr(qs.matmul(&keys_t))?;
            let scores = coerr(coerr(scores.to_dtype(DType::F32))?.relu())?;
            let scores = coerr(coerr(scores.reshape(vec![take, nh, total_blocks]))?.sum_keepdim(1))?;
            let scores = coerr(coerr(scores.reshape(vec![take, total_blocks]))?
                .mul_scalar(1.0 / (d as f32).sqrt()))?;
            let valid_t = coerr(Tensor::from_vec(
                valid[row..row + take].to_vec(),
                vec![take],
                self.device,
            ))?;
            let picked = match scores.topk_wide(&valid_t, topk) {
                Ok(t) => t,
                Err(SynaptixError::Unsupported(_)) => {
                    self.topk_on_host(&scores, &valid[row..row + take], take, total_blocks, topk)?
                }
                Err(e) => return Err(ModelError::Forward(e.to_string())),
            };
            parts.push(picked);
            row += take;
        }
        let blocks = if parts.len() == 1 {
            parts.pop().expect("одна часть")
        } else {
            let refs: Vec<&Tensor> = parts.iter().collect();
            coerr(Tensor::cat(&refs, 0))?
        };
        let tail_from = coerr(Tensor::from_vec(
            tails.iter().map(|(f, _)| *f).collect::<Vec<u32>>(),
            vec![s],
            self.device,
        ))?;
        let tail_len = coerr(Tensor::from_vec(
            tails.iter().map(|(_, l)| *l).collect::<Vec<u32>>(),
            vec![s],
            self.device,
        ))?;
        Ok(Some(Selection {
            ratio: cr,
            topk,
            rows: s,
            blocks_total: total_blocks,
            blocks,
            tail_from,
            tail_len,
            tails_host: tails,
            host: std::sync::OnceLock::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tail_rows(cache: &IndexerCache, rows: usize) -> Vec<f32> {
        cache
            .pending
            .narrow(1, 0, rows)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("чтение pending")
    }

    fn put_rows(cache: &mut IndexerCache, data: Vec<f32>, rows: usize, d: usize) {
        let src = Tensor::from_vec(data, vec![1, rows, d], Device::Cpu).expect("src");
        cache.pending.copy_rows_from(0, &src).expect("запись pending");
    }

    /// Хвост индексатора обязан пережить цикл «снимок в конце промпта →
    /// свёртки декода → restore следующего хода»: и счётчики, и СОДЕРЖИМОЕ
    /// `pending`. Ровно здесь ломался префикс-KV чата qwen3.8-flash-next:
    /// откат по метке возвращал счётчики (да ещё через `min`), содержимое
    /// было затёрто свёртками декода, сетка блоков уезжала относительно
    /// позиций токенов — и выбор блоков QSA превращался в шум, модель со
    /// второго хода отвечала мусором.
    #[test]
    fn indexer_tail_roundtrip_survives_folds() {
        let d = 4usize;
        let ratio = 4usize;
        let mut cache = IndexerCache::new(16, d, ratio, Device::Cpu, DType::F32).expect("cache");

        // Состояние на конец промпта: 2 блока свёрнуто, 3 ключа ждут в хвосте.
        let prompt_tail: Vec<f32> = (0..3 * d).map(|x| x as f32 + 1.0).collect();
        put_rows(&mut cache, prompt_tail.clone(), 3, d);
        cache.pending_rows = 3;
        cache.blocks = 2;
        let snap = cache.tail_snapshot().expect("снимок");

        // «Декод»: свёртки много раз переписали pending и продвинули счётчики.
        put_rows(&mut cache, vec![-7.0; 3 * d], 3, d);
        cache.pending_rows = 1;
        cache.blocks = 5;

        cache.restore_tail(&snap).expect("restore");
        assert_eq!(cache.mark(), (2, 3), "счётчики обязаны вернуться ровно");
        assert_eq!(
            tail_rows(&cache, 3),
            prompt_tail,
            "содержимое хвоста обязано вернуться байт в байт"
        );
    }

    /// Откат по метке ставит счётчики ровно, а не через `min`: после свёртки
    /// текущий `pending_rows` меньше откатываемого, и `min` оставлял его на
    /// свёрнутом значении — сетка блоков уезжала.
    #[test]
    fn rewind_sets_counters_exactly() {
        let mut cache = IndexerCache::new(16, 4, 4, Device::Cpu, DType::F32).expect("cache");
        cache.blocks = 5;
        cache.pending_rows = 1;
        cache.rewind((2, 3));
        assert_eq!(cache.mark(), (2, 3));
    }
}
