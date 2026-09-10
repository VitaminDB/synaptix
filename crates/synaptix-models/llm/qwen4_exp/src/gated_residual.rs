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

/// Слитое ядро неприменимо (не CUDA/не F16/неконтигуозно) — идём прежней цепочкой.
fn fallback(e: &synaptix_core::error::SynaptixError) -> bool {
    !crate::norm::fused_on() || matches!(
        e,
        synaptix_core::error::SynaptixError::Unsupported(_)
            | synaptix_core::error::SynaptixError::NonContiguous
    )
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
        let off = || Err(synaptix_core::error::SynaptixError::Unsupported("выключено"));
        let mix = match if crate::norm::fused_on() { mix.scale_act_fused(inv, 0) } else { off() } {
            Ok(t) => t,
            Err(e) if fallback(&e) => coerr(coerr(mix.mul_scalar(inv))?.silu())?,
            Err(e) => return Err(ModelError::Forward(e.to_string())),
        };
        let up = self.up.forward(&mix)?;
        // sigmoid(up) · normed, среднее по потокам — одним ядром.
        let mixed = match if crate::norm::fused_on() { up.hc_mix_fused(&normed, self.hc_count) } else { off() } {
            Ok(t) => t,
            Err(e) if fallback(&e) => {
                let mix = coerr(up.sigmoid())?;
                let split = vec![tokens, self.hc_count, self.hidden];
                let weighted = coerr(mix.mul(&normed))?;
                let mixed = coerr(coerr(weighted.reshape(split))?.mean_keepdim(1))?;
                coerr(mixed.reshape(vec![tokens, self.hidden]))?
            }
            Err(e) => return Err(ModelError::Forward(e.to_string())),
        };

        let inject_weights = match &self.inject {
            Some(w) => {
                let g = w.forward(&normed)?;
                Some(match if crate::norm::fused_on() { g.scale_act_fused(inv, 2) } else { off() } {
                    Ok(t) => t,
                    Err(e) if fallback(&e) => {
                        coerr(coerr(coerr(g.mul_scalar(inv))?.sigmoid())?.mul_scalar(2.0))?
                    }
                    Err(e) => return Err(ModelError::Forward(e.to_string())),
                })
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
        if crate::norm::fused_on() {
            match hyper.hc_inject_fused(block_out, inject_weights, self.hc_count) {
                Ok(t) => return Ok(t),
                Err(e) if fallback(&e) => {}
                Err(e) => return Err(ModelError::Forward(e.to_string())),
            }
        }
        let out = coerr(coerr(block_out.contiguous())?.reshape(vec![tokens, 1, self.hidden]))?;
        let w = coerr(inject_weights.reshape(vec![tokens, self.hc_count, 1]))?;
        let injection = coerr(coerr(out.broadcast_mul(&w))?
            .reshape(vec![tokens, self.hc_count * self.hidden]))?;
        coerr(hyper.add(&injection))
    }
}
