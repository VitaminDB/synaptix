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

    /// Декод (до 4 строк) двумя слитыми ядрами: веса down/up — из
    /// перемешанной NVFP4-копии с деквантом в регистрах, активации не
    /// квантуются. `None` — путь неприменим (не CUDA/F16, веса не NVFP4,
    /// `SYN_QWEN4EXP_HC_FUSED=0`).
    fn forward_fused(&self, flat: &Tensor, tokens: usize) -> Result<Option<Mixed>, ModelError> {
        use synaptix_core::tensor::ops::qwen4_hc_fused;
        if tokens > 4 || !hc_fused_on() || !crate::norm::fused_on() || !flat.device().is_cuda() || flat.dtype() != DType::F16 {
            return Ok(None);
        }
        let (Some(dw), Some(uw)) = (self.down.quant_weight(), self.up.quant_weight()) else { return Ok(None) };
        if dw.dtype() != DType::NVFP4 || uw.dtype() != DType::NVFP4 || self.hc_norm.dtype() != DType::F16 {
            return Ok(None);
        }
        let k = self.hc_count * self.hidden;
        let r = dw.n();
        if dw.k() != k || uw.n() != k || uw.k() != r || r % 64 != 0 || self.hidden % 64 != 0 || self.hc_count > 8 {
            return Ok(None);
        }
        if dw.ensure_shuffled().is_err() || uw.ensure_shuffled().is_err() {
            return Ok(None);
        }
        let (Some((dwa, dsa)), Some((uwa, usa))) = (dw.shuffled_addr(), uw.shuffled_addr()) else { return Ok(None) };
        let inj_w = match &self.inject {
            Some(QLinear::Dense(l)) => {
                let w = l.weight();
                if w.dtype() != DType::F16 || w.dims() != [self.hc_count, k] || !w.is_contiguous() {
                    return Ok(None);
                }
                Some(w)
            }
            Some(_) => return Ok(None),
            None => None,
        };
        let dev = flat.device();
        let addr = |t: &Tensor| t.device_address().ok_or_else(|| ModelError::Forward("hc: тензор не на карте".into()));
        let mix = coerr(Tensor::empty_uninit(vec![tokens, r], DType::F32, dev))?;
        let norms = coerr(Tensor::empty_uninit(vec![tokens, self.hc_count], DType::F32, dev))?;
        let mixed = coerr(Tensor::empty_uninit(vec![tokens, self.hidden], DType::F16, dev))?;
        let inj_out = match &inj_w {
            Some(_) => Some(coerr(Tensor::empty_uninit(vec![tokens, self.hc_count], DType::F16, dev))?),
            None => None,
        };
        let a = synaptix_core::backend::HcLaunch {
            x: addr(flat)?,
            norm_w: addr(&self.hc_norm)?,
            down_w: dwa,
            down_s: dsa,
            inj_w: match &inj_w {
                Some(w) => addr(w)?,
                None => 0,
            },
            up_w: uwa,
            up_s: usa,
            mix: addr(&mix)?,
            norms: addr(&norms)?,
            inj_out: match &inj_out {
                Some(t) => addr(t)?,
                None => 0,
            },
            mixed: addr(&mixed)?,
            rows: tokens as u32,
            hc: self.hc_count as u32,
            h: self.hidden as u32,
            r: r as u32,
            eps: self.eps,
            _pad: 0.0,
        };
        match qwen4_hc_fused(dev, &a) {
            Ok(()) => {
                Ok(Some(Mixed { mixed, hyper: flat.clone(), inject_weights: inj_out }))
            }
            Err(e) if fallback(&e) => Ok(None),
            Err(e) => Err(ModelError::Forward(e.to_string())),
        }
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
        if let Some(m) = self.forward_fused(&flat, tokens)? {
            return Ok(m);
        }
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

/// Слитый gated-residual на декоде (`SYN_QWEN4EXP_HC_FUSED=0` — прежняя цепочка).
fn hc_fused_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SYN_QWEN4EXP_HC_FUSED").as_deref() != Ok("0"))
}
