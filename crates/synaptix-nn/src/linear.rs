use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::{InitMethod, init_tensor};
use crate::module::{Module, join_path};
use crate::parameter::Parameter;

pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(SynaptixError::Unsupported("Linear: weight must be 2D (out, in)"));
        }
        let out_features = weight.dims()[0];
        let in_features = weight.dims()[1];
        if let Some(b) = &bias {
            if b.rank() != 1 || b.dims()[0] != out_features {
                return Err(SynaptixError::shape_mismatch(&[out_features], b.dims()));
            }
        }
        Ok(Self {
            weight: Parameter::new(weight).with_name("weight"),
            bias: bias.map(|b| Parameter::new(b).with_name("bias")),
            in_features,
            out_features,
        })
    }

    pub fn from_init(
        in_features: usize,
        out_features: usize,
        bias: bool,
        weight_init: InitMethod,
        bias_init: InitMethod,
        device: Device,
        dtype: DType,
        seed: u64,
    ) -> Result<Self> {
        let w = init_tensor(&[out_features, in_features], weight_init, dtype, seed, device)?;
        let b = if bias {
            Some(init_tensor(&[out_features], bias_init, dtype, seed.wrapping_add(1), device)?)
        } else {
            None
        };
        Self::new(w, b)
    }

    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }
    pub fn weight(&self) -> Tensor { self.weight.tensor() }
    pub fn bias(&self) -> Option<Tensor> { self.bias.as_ref().map(|p| p.tensor()) }

    pub fn forward_add(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let b = self.bias.as_ref().map(|p| p.tensor());
        x.linear_bias_residual(&self.weight.tensor(), b.as_ref(), Some(residual))
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // `linear` сам выбирает путь: CUDA decode (M=1) → GEMV напрямую с весом
        // в [out, in] (без транспонирования/dense-GEMM); иначе matmul(wᵀ).
        let b = self.bias.as_ref().map(|p| p.tensor());
        x.linear_bias_residual(&self.weight.tensor(), b.as_ref(), None)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut out = vec![&self.weight];
        if let Some(b) = &self.bias {
            out.push(b);
        }
        out
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter)> {
        let mut out = vec![(join_path(prefix, "weight"), &self.weight)];
        if let Some(b) = &self.bias {
            out.push((join_path(prefix, "bias"), b));
        }
        out
    }
}
