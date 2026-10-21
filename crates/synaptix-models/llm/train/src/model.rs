//! Модель на обучении: база [`DecoderModel`] (заморожена) плюс адаптеры
//! всех слоёв. Шаг — прямой проход с сохранением только входов слоёв,
//! лосс по токенам с меткой, обратный проход слой за слоем с пересчётом
//! активаций слоя.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::DecoderModel;
use synaptix_ops::norm::rms_norm::rms_norm;

use crate::block::{self, BlockLora, LinConst, SeqCtx};
use synaptix_llm_common::model::train_access::MixerRefs;
use crate::error::other;
use crate::linear::{self, BaseMatmul, Weights};
use crate::lora::{LoraConfig, LoraLayer};
use crate::Result;

/// Одна упакованная последовательность.
#[derive(Debug, Clone, Default)]
pub struct Batch {
    pub tokens: Vec<u32>,
    /// Метка позиции `t` — следующий токен, который модель должна
    /// предсказать после `t`; `-100` — без лосса.
    pub labels: Vec<i32>,
    /// Позиции RoPE (у каждого примера с нуля).
    pub positions: Vec<u32>,
    /// Номер примера в упаковке.
    pub seg: Vec<i32>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn n_targets(&self) -> usize {
        self.labels.iter().filter(|&&l| l >= 0).count()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StepStats {
    /// Сумма CE по токенам с меткой (без масштаба).
    pub loss_sum: f64,
    pub n_targets: usize,
}

/// Сколько строк логитов считать за раз (строка — словарь целиком).
const HEAD_CHUNK: usize = 1024;

pub struct TrainModel<'m> {
    pub base: &'m DecoderModel,
    pub lora: Vec<BlockLora>,
    pub cfg: LoraConfig,
    pub base_matmul: BaseMatmul,
}

impl<'m> TrainModel<'m> {
    /// Новые адаптеры на целевых проекциях всех слоёв.
    pub fn new(base: &'m DecoderModel, cfg: LoraConfig) -> Result<Self> {
        base.train_check()?;
        if !base.device.is_cuda() {
            return Err(other("обучение: нужна CUDA-карта"));
        }
        let dev = base.device;
        let mut lora = Vec::with_capacity(base.config.num_hidden_layers);
        for i in 0..base.config.num_hidden_layers {
            let b = base.train_block(i)?;
            let seed = cfg.seed.wrapping_mul(0x100000001B3).wrapping_add(i as u64 * 16);
            let mk = |name: &str, kind: &str, q: &synaptix_llm_common::QLinear, j: u64| -> Result<Option<LoraLayer>> {
                if !cfg.targets(name) {
                    return Ok(None);
                }
                let (n, k) = linear::dims(q);
                let module = format!("{}.{i}.{kind}.{name}", cfg.module_prefix);
                Ok(Some(LoraLayer::new(module, n, k, &cfg, seed + j, dev)?))
            };
            let mut bl = BlockLora::default();
            match &b.mixer {
                MixerRefs::Full(at) => {
                    let v = at.v_proj.ok_or_else(|| other("обучение: слой без v_proj"))?;
                    bl.q = mk("q_proj", "self_attn", at.q_proj, 1)?;
                    bl.k = mk("k_proj", "self_attn", at.k_proj, 2)?;
                    bl.v = mk("v_proj", "self_attn", v, 3)?;
                    bl.o = mk("o_proj", "self_attn", at.o_proj, 4)?;
                }
                MixerRefs::Linear(la) => {
                    bl.qkv = mk("in_proj_qkv", "linear_attn", la.in_proj_qkv, 8)?;
                    bl.z = mk("in_proj_z", "linear_attn", la.in_proj_z, 9)?;
                    bl.a = mk("in_proj_a", "linear_attn", la.in_proj_a, 10)?;
                    bl.b = mk("in_proj_b", "linear_attn", la.in_proj_b, 11)?;
                    bl.out = mk("out_proj", "linear_attn", la.out_proj, 12)?;
                    if la.dk != 128 || la.dv % 32 != 0 {
                        return Err(other(format!("GDN: ядро обучения — dk = 128 и dv кратно 32 (у слоя {i}: {} и {})", la.dk, la.dv)));
                    }
                    bl.lin = Some(LinConst::new(la, dev)?);
                }
            }
            bl.gate = mk("gate_proj", "mlp", b.mlp.gate_proj, 5)?;
            bl.up = mk("up_proj", "mlp", b.mlp.up_proj, 6)?;
            bl.down = mk("down_proj", "mlp", b.mlp.down_proj, 7)?;
            lora.push(bl);
        }
        Ok(Self { base, lora, cfg, base_matmul: BaseMatmul::default() })
    }

    pub fn layers(&self) -> impl Iterator<Item = &LoraLayer> {
        self.lora.iter().flat_map(|b| b.layers())
    }

    pub fn layers_mut(&mut self) -> impl Iterator<Item = &mut LoraLayer> {
        self.lora.iter_mut().flat_map(|b| b.layers_mut())
    }

    pub fn num_params(&self) -> usize {
        self.layers().map(|l| l.num_params()).sum()
    }

    pub fn zero_grad(&mut self) {
        for l in self.layers_mut() {
            l.zero_grad();
        }
    }

    fn ctx(&self, batch: &Batch) -> Result<SeqCtx> {
        let dev = self.base.device;
        let t = batch.len();
        let positions = Tensor::from_vec(batch.positions.clone(), vec![t], dev)?;
        let multi = batch.seg.iter().any(|&s| s != batch.seg[0]);
        let seg = if multi { Some(Tensor::from_vec(batch.seg.clone(), vec![t], dev)?) } else { None };
        Ok(SeqCtx::new(t, positions, seg))
    }

    fn embed(&self, batch: &Batch) -> Result<Tensor> {
        let t = batch.len();
        let ids = Tensor::from_vec(batch.tokens.clone(), vec![1, t], self.base.device)?;
        let x = self.base.embed_ids(&ids)?;
        Ok(x.reshape(vec![t, self.base.config.hidden_size])?)
    }

    /// Выходы всех слоёв (отладка и сверка с эталоном): `[вход, после слоя 0, …]`.
    pub fn hidden_states(&self, batch: &Batch) -> Result<Vec<Tensor>> {
        let ctx = self.ctx(batch)?;
        let mut x = self.embed(batch)?;
        let mut out = vec![x.clone()];
        for (i, l) in self.lora.iter().enumerate() {
            let b = self.base.train_block(i)?;
            x = block::forward(&b, l, &x, &ctx, &Weights::new(self.base_matmul))?.0;
            out.push(x.clone());
        }
        Ok(out)
    }

    /// Прямой проход без градиентов: средний CE по токенам с меткой.
    pub fn eval_loss(&self, batch: &Batch) -> Result<StepStats> {
        let ctx = self.ctx(batch)?;
        let mut x = self.embed(batch)?;
        for (i, l) in self.lora.iter().enumerate() {
            let b = self.base.train_block(i)?;
            x = block::forward(&b, l, &x, &ctx, &Weights::new(self.base_matmul))?.0;
        }
        let (stats, _) = self.head(&x, &batch.labels, 1.0, false)?;
        Ok(stats)
    }

    /// Шаг: прямой и обратный проход, градиенты копятся в адаптерах.
    /// `loss_scale` — множитель градиента лосса (обычно 1 / число токенов
    /// с меткой во всём шаге оптимизатора).
    pub fn forward_backward(&mut self, batch: &Batch, loss_scale: f32) -> Result<StepStats> {
        if batch.is_empty() {
            return Ok(StepStats::default());
        }
        if batch.labels.len() != batch.len() || batch.positions.len() != batch.len() || batch.seg.len() != batch.len() {
            return Err(other("Batch: длины tokens/labels/positions/seg различаются"));
        }
        let max_pos = batch.positions.iter().copied().max().unwrap_or(0) as usize;
        if max_pos >= self.base.rope_capacity() {
            return Err(other(format!("позиция {max_pos} за пределами RoPE-кэша ({})", self.base.rope_capacity())));
        }
        let ctx = self.ctx(batch)?;
        let n_layers = self.lora.len();
        let mut inputs: Vec<Tensor> = Vec::with_capacity(n_layers);
        let mut x = self.embed(batch)?;
        for i in 0..n_layers {
            let b = self.base.train_block(i)?;
            inputs.push(x.clone());
            x = crate::prof::prof(self.base.device, "Σ block_fwd", || block::forward(&b, &self.lora[i], &x, &ctx, &Weights::new(self.base_matmul)))?.0;
        }
        let (stats, dx) = crate::prof::prof(self.base.device, "head", || self.head(&x, &batch.labels, loss_scale, true))?;
        drop(x);
        let Some(mut dx) = dx else { return Ok(stats) };
        for i in (0..n_layers).rev() {
            let b = self.base.train_block(i)?;
            let xi = inputs.pop().unwrap();
            let w = Weights::new(self.base_matmul);
            let (_, saved) = crate::prof::prof(self.base.device, "Σ block_fwd", || block::forward(&b, &self.lora[i], &xi, &ctx, &w))?;
            drop(xi);
            dx = crate::prof::prof(self.base.device, "Σ block_bwd", || block::backward(&b, &mut self.lora[i], saved, &dx, &w))?;
        }
        Ok(stats)
    }

    /// Финальная норма, голова и CE. Логиты считаются только для строк с
    /// меткой и кусками по [`HEAD_CHUNK`]. `want_grad` — вернуть `dx` по
    /// выходу последнего слоя (`None`, если меток нет).
    fn head(&self, x: &Tensor, labels: &[i32], loss_scale: f32, want_grad: bool) -> Result<(StepStats, Option<Tensor>)> {
        let base = self.base;
        let dev = base.device;
        let (t, hdim) = (x.dims()[0], x.dims()[1]);
        let valid: Vec<u32> = (0..t as u32).filter(|&i| labels[i as usize] >= 0).collect();
        let nv = valid.len();
        if nv == 0 {
            return Ok((StepStats::default(), None));
        }
        let eps = base.config.rms_norm_eps;
        let final_norm = base.train_final_norm();
        let hn = rms_norm(x, final_norm, eps)?;
        let idx = Tensor::from_vec(valid.clone(), vec![nv], dev)?;
        let hv = hn.index_select(0, &idx)?;
        let lab: Vec<i32> = valid.iter().map(|&i| labels[i as usize]).collect();
        let lm_head = base.train_lm_head();
        let head_w = Weights::new(self.base_matmul);
        let w_head = if want_grad { Some(head_w.dense(lm_head)?) } else { None };
        let (scale, cap) = (base.config.logit_scale, base.config.logit_softcap);

        let mut loss_sum = 0f64;
        let mut dparts: Vec<Tensor> = Vec::new();
        let mut c0 = 0;
        while c0 < nv {
            let c = HEAD_CHUNK.min(nv - c0);
            let hc = hv.narrow(0, c0, c)?.contiguous()?;
            let raw = head_w.forward(lm_head, &hc)?;
            let mut z = raw.to_dtype(DType::F32)?;
            if let Some(s) = scale {
                z = z.mul_scalar(s)?;
            }
            let th = match cap {
                Some(cp) => Some(z.mul_scalar(1.0 / cp)?.tanh()?),
                None => None,
            };
            let logits = match (&th, cap) {
                (Some(th), Some(cp)) => th.mul_scalar(cp)?,
                _ => z,
            };
            let lt = Tensor::from_vec(lab[c0..c0 + c].to_vec(), vec![c], dev)?;
            let v = logits.dims()[1];
            let gdt = if th.is_some() || scale.is_some() { DType::F32 } else { DType::BF16 };
            let out = Tensor::train_op(
                synaptix_core::backend::train::TrainOp::CrossEntropy { scale: loss_scale },
                &[&logits, &lt],
                &[(vec![c], DType::F32), (vec![c, v], gdt)],
            )?;
            drop(logits);
            loss_sum += out[0].sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
            if let Some(w) = &w_head {
                let mut dl = out[1].clone();
                if let Some(th) = &th {
                    // d(cap·tanh(z/cap))/dz = 1 − tanh²
                    dl = dl.mul(&th.sqr()?.affine(-1.0, 1.0)?)?;
                }
                if let Some(s) = scale {
                    dl = dl.mul_scalar(s)?;
                }
                dparts.push(linear::backward_input_with(w, &dl)?);
            }
            c0 += c;
        }
        let stats = StepStats { loss_sum, n_targets: nv };
        if !want_grad {
            return Ok((stats, None));
        }
        let refs: Vec<&Tensor> = dparts.iter().collect();
        let dhv = if refs.len() == 1 { dparts[0].clone() } else { Tensor::cat(&refs, 0)? };
        // Обратно на все T строк: строки без метки берут нулевую строку в конце.
        let zero = Tensor::zeros(vec![1, hdim], DType::F32, dev)?;
        let padded = Tensor::cat(&[&dhv, &zero], 0)?;
        let mut inv = vec![nv as u32; t];
        for (j, &i) in valid.iter().enumerate() {
            inv[i as usize] = j as u32;
        }
        let inv = Tensor::from_vec(inv, vec![t], dev)?;
        let dhn = padded.index_select(0, &inv)?;
        let dx = block::rms_bwd(x, final_norm, &dhn, eps)?;
        Ok((stats, Some(dx)))
    }

    pub fn device(&self) -> Device {
        self.base.device
    }
}
