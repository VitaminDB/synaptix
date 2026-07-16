use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::activation::glu::glu;
use synaptix_ops::conv::{conv1d, depthwise_conv};
use synaptix_ops::norm::{batch_norm_inference, layer_norm};

use crate::init::InitMethod;
use crate::module::Module;
use crate::parameter::Parameter;

/// Conformer Convolution Module (torchaudio.models.Conformer._ConvolutionModule).
///
/// `x [B, S, C]` → LN → permute → pointwise_conv1d(2C) → GLU(dim=1) →
/// depthwise_conv1d(K, padding=(K-1)/2) → BatchNorm1d → SiLU →
/// pointwise_conv1d(C) → permute обратно → residual.
///
/// Все Conv1d представлены как weights `[C_out, C_in, K]`. Pointwise — K=1.
pub struct ConvModule {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    /// Pointwise expansion `[2C, C, 1]`.
    pub pw1_w: Parameter,
    pub pw1_b: Option<Parameter>,
    /// Depthwise `[C, 1, K]`.
    pub dw_w: Parameter,
    pub dw_b: Option<Parameter>,
    /// BatchNorm running stats `[C]`.
    pub bn_mean: Parameter,
    pub bn_var: Parameter,
    pub bn_w: Option<Parameter>,
    pub bn_b: Option<Parameter>,
    /// Pointwise contraction `[C, C, 1]`.
    pub pw2_w: Parameter,
    pub pw2_b: Option<Parameter>,
    pub hidden_size: usize,
    pub kernel_size: usize,
    pub eps: f32,
    pub bn_eps: f32,
}

impl ConvModule {
    pub fn new(hidden_size: usize, kernel_size: usize, device: Device, dtype: DType) -> Result<Self> {
        if kernel_size % 2 == 0 {
            return Err(SynaptixError::Unsupported(
                "ConvModule: kernel_size must be odd (symmetric padding)",
            ));
        }
        let norm_w = Tensor::ones(vec![hidden_size], dtype, device)?;
        let norm_b = Tensor::zeros(vec![hidden_size], dtype, device)?;
        let bn_mean = Tensor::zeros(vec![hidden_size], dtype, device)?;
        let bn_var = Tensor::ones(vec![hidden_size], dtype, device)?;
        let bn_w = Tensor::ones(vec![hidden_size], dtype, device)?;
        let bn_b = Tensor::zeros(vec![hidden_size], dtype, device)?;

        let init_pw1 = InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 };
        let init_dw = InitMethod::KaimingUniform { fan_in: kernel_size, a: 0.0 };
        let init_pw2 = InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 };
        let pw1_w = crate::init::init_tensor(&[hidden_size * 2, hidden_size, 1], init_pw1, dtype, 0, device)?;
        let dw_w = crate::init::init_tensor(&[hidden_size, 1, kernel_size], init_dw, dtype, 1, device)?;
        let pw2_w = crate::init::init_tensor(&[hidden_size, hidden_size, 1], init_pw2, dtype, 2, device)?;

        Ok(Self {
            norm_w: Parameter::new(norm_w).with_name("norm.weight"),
            norm_b: Parameter::new(norm_b).with_name("norm.bias"),
            pw1_w: Parameter::new(pw1_w).with_name("pw1.weight"),
            pw1_b: None,
            dw_w: Parameter::new(dw_w).with_name("dw.weight"),
            dw_b: None,
            bn_mean: Parameter::new(bn_mean).with_name("bn.running_mean"),
            bn_var: Parameter::new(bn_var).with_name("bn.running_var"),
            bn_w: Some(Parameter::new(bn_w).with_name("bn.weight")),
            bn_b: Some(Parameter::new(bn_b).with_name("bn.bias")),
            pw2_w: Parameter::new(pw2_w).with_name("pw2.weight"),
            pw2_b: None,
            hidden_size,
            kernel_size,
            eps: 1e-5,
            bn_eps: 1e-5,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        pw1_w: Tensor, pw1_b: Option<Tensor>,
        dw_w: Tensor, dw_b: Option<Tensor>,
        bn_mean: Tensor, bn_var: Tensor,
        bn_w: Option<Tensor>, bn_b: Option<Tensor>,
        pw2_w: Tensor, pw2_b: Option<Tensor>,
        eps: f32, bn_eps: f32,
    ) -> Result<Self> {
        // pw1 [2C, C, 1], dw [C, 1, K], pw2 [C, C, 1]
        if pw1_w.rank() != 3 || dw_w.rank() != 3 || pw2_w.rank() != 3 {
            return Err(SynaptixError::Unsupported(
                "ConvModule: pw1/dw/pw2 must be 3D [Cout, Cin, K]",
            ));
        }
        let hidden_size = pw2_w.dims()[0];
        if pw1_w.dims() != [hidden_size * 2, hidden_size, 1]
            || pw2_w.dims() != [hidden_size, hidden_size, 1]
            || dw_w.dims()[0] != hidden_size || dw_w.dims()[1] != 1
        {
            return Err(SynaptixError::Unsupported(
                "ConvModule: shape mismatch between pw1/dw/pw2",
            ));
        }
        let kernel_size = dw_w.dims()[2];
        Ok(Self {
            norm_w: Parameter::new(norm_w).with_name("norm.weight"),
            norm_b: Parameter::new(norm_b).with_name("norm.bias"),
            pw1_w: Parameter::new(pw1_w).with_name("pw1.weight"),
            pw1_b: pw1_b.map(|b| Parameter::new(b).with_name("pw1.bias")),
            dw_w: Parameter::new(dw_w).with_name("dw.weight"),
            dw_b: dw_b.map(|b| Parameter::new(b).with_name("dw.bias")),
            bn_mean: Parameter::new(bn_mean).with_name("bn.running_mean"),
            bn_var: Parameter::new(bn_var).with_name("bn.running_var"),
            bn_w: bn_w.map(|w| Parameter::new(w).with_name("bn.weight")),
            bn_b: bn_b.map(|b| Parameter::new(b).with_name("bn.bias")),
            pw2_w: Parameter::new(pw2_w).with_name("pw2.weight"),
            pw2_b: pw2_b.map(|b| Parameter::new(b).with_name("pw2.bias")),
            hidden_size, kernel_size, eps, bn_eps,
        })
    }

    /// `x: [B, S, C]` → `[B, S, C]`. Residual прибавляется внутри.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported(
                "ConvModule: expects x [B, S, hidden_size]",
            ));
        }
        let h = layer_norm(
            x,
            Some(&self.norm_w.tensor()),
            Some(&self.norm_b.tensor()),
            self.eps,
        )?;
        // [B, S, C] → [B, C, S] for Conv1d
        let h = h.permute(vec![0, 2, 1])?.contiguous()?;

        // Pointwise expansion: [B, 2C, S]
        let h = conv1d(
            &h,
            &self.pw1_w.tensor(),
            self.pw1_b.as_ref().map(|p| p.tensor()).as_ref(),
            1, 0,
        )?;
        // GLU along channel dim (1): [B, C, S]
        let h = glu(&h, 1)?;

        // Depthwise: symmetric padding (K-1)/2
        let pad = (self.kernel_size - 1) / 2;
        let h = depthwise_conv(
            &h,
            &self.dw_w.tensor(),
            self.dw_b.as_ref().map(|p| p.tensor()).as_ref(),
            1, pad, self.hidden_size,
        )?;

        // BatchNorm1d на канал-измерении
        let h = batch_norm_inference(
            &h,
            &self.bn_mean.tensor(),
            &self.bn_var.tensor(),
            self.bn_w.as_ref().map(|p| p.tensor()).as_ref(),
            self.bn_b.as_ref().map(|p| p.tensor()).as_ref(),
            self.bn_eps,
        )?;
        let h = h.silu()?;

        // Pointwise contraction: [B, C, S]
        let h = conv1d(
            &h,
            &self.pw2_w.tensor(),
            self.pw2_b.as_ref().map(|p| p.tensor()).as_ref(),
            1, 0,
        )?;
        // [B, C, S] → [B, S, C]
        let h = h.permute(vec![0, 2, 1])?.contiguous()?;
        x.add(&h)
    }
}

// Module trait — pass-through, residual внутри forward (matches conv_module spec).
impl Module for ConvModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}
