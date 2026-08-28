use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::{ModelError, QLinear, WeightSource};
use synaptix_ops::pos::rope::{apply_rope_with_cossin, RopeLayout};
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::config::{IndexerConfig, Qwen4ExpConfig};
use crate::norm::{coerr, load_one_plus, rms};

pub struct IndexerCache {
    pending: Vec<f32>,
    block_keys: Tensor,
    blocks: usize,
    capacity: usize,
}

impl IndexerCache {
    pub fn new(capacity_blocks: usize, head_dim: usize, device: Device, dtype: DType) -> Result<Self, ModelError> {
        let block_keys = Tensor::zeros(vec![1, 1, capacity_blocks.max(1), head_dim], dtype, device)
            .map_err(|e| ModelError::Build(e.to_string()))?;
        Ok(Self { pending: Vec::new(), block_keys, blocks: 0, capacity: capacity_blocks.max(1) })
    }

    pub fn blocks(&self) -> usize {
        self.blocks
    }

    pub fn reset(&mut self) {
        self.pending.clear();
        self.blocks = 0;
    }
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
        IndexerCache::new(blocks, self.cfg.head_dim, self.device, self.compute)
    }

    fn rope(&self, x: &Tensor, rope: &RopeCache, positions: &[u32]) -> Result<Tensor, ModelError> {
        if self.rotary_dim == 0 {
            return Ok(x.clone());
        }
        let pos = Tensor::from_vec(positions.to_vec(), vec![positions.len()], self.device)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let (cos, sin) = coerr(rope.select_positions(&pos))?;
        let d = x.dims()[3];
        if self.rotary_dim == d {
            return coerr(apply_rope_with_cossin(x, &cos, &sin, RopeLayout::Split));
        }
        let head = coerr(coerr(x.narrow(3, 0, self.rotary_dim))?.contiguous())?;
        let tail = coerr(coerr(x.narrow(3, self.rotary_dim, d - self.rotary_dim))?.contiguous())?;
        let rotated = coerr(apply_rope_with_cossin(&head, &cos, &sin, RopeLayout::Split))?;
        coerr(Tensor::cat(&[&rotated, &tail], 3))
    }

    fn push_keys(
        &self,
        cache: &mut IndexerCache,
        raw: &[f32],
        rope: &RopeCache,
    ) -> Result<(), ModelError> {
        let d = self.cfg.head_dim;
        let cr = self.cfg.compress_ratio;
        cache.pending.extend_from_slice(raw);
        let available = cache.pending.len() / d;
        let new_blocks = available / cr;
        if new_blocks == 0 {
            return Ok(());
        }
        if cache.blocks + new_blocks > cache.capacity {
            return Err(ModelError::Shape(format!(
                "индексатор: блоков {} больше ёмкости {}",
                cache.blocks + new_blocks,
                cache.capacity
            )));
        }
        let mut pooled = vec![0f32; new_blocks * d];
        for b in 0..new_blocks {
            for j in 0..cr {
                let src = (b * cr + j) * d;
                for c in 0..d {
                    pooled[b * d + c] += cache.pending[src + c];
                }
            }
            for c in 0..d {
                pooled[b * d + c] /= cr as f32;
            }
        }
        cache.pending.drain(..new_blocks * cr * d);

        let pooled = Tensor::from_vec(pooled, vec![new_blocks, d], self.device)
            .and_then(|t| t.to_dtype(self.compute))
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        let normed = rms(&pooled, &self.k_norm, self.eps)?;
        let normed = coerr(normed.reshape(vec![1, 1, new_blocks, d]))?;
        let positions: Vec<u32> = (0..new_blocks)
            .map(|b| ((cache.blocks + b) * cr) as u32)
            .collect();
        let roped = self.rope(&normed, rope, &positions)?;
        cache
            .block_keys
            .kv_append_inplace(&roped, cache.blocks)
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        cache.blocks += new_blocks;
        Ok(())
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
    ) -> Result<Option<Vec<Vec<u32>>>, ModelError> {
        let d = self.cfg.head_dim;
        let nh = self.cfg.n_heads;
        let qk = self.qk_proj.forward(h)?;
        let qk = coerr(qk.reshape(vec![s, (nh + self.cfg.kv_heads) * d]))?;
        let q = coerr(coerr(qk.narrow(1, 0, nh * d))?.contiguous())?;
        let k = coerr(coerr(qk.narrow(1, nh * d, self.cfg.kv_heads * d))?.contiguous())?;

        let raw = k
            .to_device(Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| ModelError::Forward(e.to_string()))?;
        self.push_keys(cache, &raw, rope)?;

        let kv_len = past + s;
        if !self.needs_selection(kv_len) {
            return Ok(None);
        }

        let q = coerr(q.reshape(vec![1, s, nh, d]))?;
        let q = rms(&q, &self.q_norm, self.eps)?;
        let q = coerr(coerr(q.permute(vec![0, 2, 1, 3]))?.contiguous())?;
        let positions: Vec<u32> = (past..past + s).map(|p| p as u32).collect();
        let q = self.rope(&q, rope, &positions)?;
        let q = coerr(coerr(coerr(q.permute(vec![0, 2, 1, 3]))?.contiguous())?
            .reshape(vec![s * nh, d]))?;

        let total_blocks = cache.blocks;
        let keys = coerr(coerr(cache.block_keys.narrow(2, 0, total_blocks))?
            .reshape(vec![total_blocks, d]))?;
        let keys_t = coerr(coerr(keys.t())?.contiguous())?;

        let cr = self.cfg.compress_ratio;
        let topk = self.cfg.block_topk();
        let mut selected = Vec::with_capacity(s);
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
            let host = scores
                .to_device(Device::Cpu)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| ModelError::Forward(e.to_string()))?;
            for i in 0..take {
                let pos = past + row + i;
                let visible = pos + 1;
                let nb = visible / cr;
                let scores_row = &host[i * total_blocks..i * total_blocks + nb.min(total_blocks)];
                let mut tokens = Vec::with_capacity(topk * cr + cr);
                if nb > 0 {
                    let mut order: Vec<u32> = (0..scores_row.len() as u32).collect();
                    let take_blocks = topk.min(order.len());
                    if take_blocks < order.len() {
                        order.select_nth_unstable_by(take_blocks - 1, |a, b| {
                            scores_row[*b as usize]
                                .partial_cmp(&scores_row[*a as usize])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        order.truncate(take_blocks);
                    }
                    for b in order {
                        let start = b as usize * cr;
                        for t in start..start + cr {
                            tokens.push(t as u32);
                        }
                    }
                }
                for t in nb * cr..visible {
                    tokens.push(t as u32);
                }
                selected.push(tokens);
            }
            row += take;
        }
        Ok(Some(selected))
    }
}
