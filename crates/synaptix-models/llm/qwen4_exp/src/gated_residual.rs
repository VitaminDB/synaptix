use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::{ModelError, QLinear, WeightSource};

use crate::config::Qwen4ExpConfig;
use crate::norm::{coerr, group_rms, load_one_plus};

pub struct GatedResidual {
    hc_norm: Tensor,
    down: QLinear,
    up: QLinear,
    inject: Option<QLinear>,
    hc_count: usize,
    hidden: usize,
    eps: f32,
}

pub struct Mixed {
    pub mixed: Tensor,
    pub hyper: Tensor,
    pub inject_weights: Option<Tensor>,
}

impl GatedResidual {
    pub fn load(
        weights: &dyn WeightSource,
        prefix: &str,
        cfg: &Qwen4ExpConfig,
        device: Device,
        compute: DType,
        quant: DType,
        use_combine: bool,
    ) -> Result<Self, ModelError> {
        let lin = |name: &str| -> Result<QLinear, ModelError> {
            let key = format!("{prefix}.{name}.weight");
            if let Some(prequant) = weights.quant(&key, device) {
                return Ok(QLinear::Quant(prequant?));
            }
            let w = weights.tensor(&key, device, compute)?;
            QLinear::build(w, quant, compute)
        };
        let dense = |name: &str| -> Result<QLinear, ModelError> {
            let key = format!("{prefix}.{name}.weight");
            let w = weights.tensor(&key, device, compute)?;
            QLinear::build(w, compute, compute)
        };
        Ok(Self {
            hc_norm: load_one_plus(weights, &format!("{prefix}.hc_norm.weight"), device, compute)?,
            down: lin("input_mix_weight_down")?,
            up: lin("input_mix_weight_up")?,
            inject: if use_combine {
                Some(dense("block_inject_weight")?)
            } else {
                None
            },
            hc_count: cfg.hc_count,
            hidden: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
        })
    }

    pub fn forward(&self, hyper: &Tensor) -> Result<Mixed, ModelError> {
        let dims = hyper.dims().to_vec();
        let last = *dims.last().unwrap_or(&0);
        if last != self.hc_count * self.hidden {
            return Err(ModelError::Shape(format!(
                "gated residual: вход {last}, ожидалось {}",
                self.hc_count * self.hidden
            )));
        }
        let tokens: usize = dims[..dims.len() - 1].iter().product();
        let flat = coerr(coerr(hyper.contiguous())?.reshape(vec![tokens, last]))?;
        let normed = group_rms(&flat, &self.hc_norm, self.hidden, self.eps)?;

        let inv = 1.0 / self.hc_count as f32;
        let mix = self.down.forward(&normed)?;
        let mix = coerr(coerr(mix.mul_scalar(inv))?.silu())?;
        let mix = coerr(self.up.forward(&mix)?.sigmoid())?;

        let split = vec![tokens, self.hc_count, self.hidden];
        let weighted = coerr(mix.mul(&normed))?;
        let mixed = coerr(coerr(weighted.reshape(split))?.mean_keepdim(1))?;
        let mixed = coerr(mixed.reshape(vec![tokens, self.hidden]))?;

        let inject_weights = match &self.inject {
            Some(w) => {
                let g = w.forward(&normed)?;
                Some(coerr(coerr(coerr(g.mul_scalar(inv))?.sigmoid())?.mul_scalar(2.0))?)
            }
            None => None,
        };
        Ok(Mixed { mixed, hyper: flat, inject_weights })
    }

    pub fn inject(
        &self,
        hyper: &Tensor,
        block_out: &Tensor,
        inject_weights: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let tokens = hyper.dims()[0];
        let out = coerr(coerr(block_out.contiguous())?.reshape(vec![tokens, 1, self.hidden]))?;
        let w = coerr(inject_weights.reshape(vec![tokens, self.hc_count, 1]))?;
        let injection = coerr(coerr(out.broadcast_mul(&w))?
            .reshape(vec![tokens, self.hc_count * self.hidden]))?;
        coerr(hyper.add(&injection))
    }
}
