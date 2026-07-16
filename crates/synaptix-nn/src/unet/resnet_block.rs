use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// UNet ResNet-блок (Linear-stub: каналы трактуются как hidden_dim,
/// spatial-измерения flattened в seq).
///
/// Forward:
/// ```text
/// h = layer_norm(x, w=norm1, b=norm1_b)
/// h = silu(h)
/// h = conv1(h)                                # Linear in_channels → out_channels
/// t = silu(time_emb)
/// t = time_emb_proj(t).unsqueeze(seq_dim)     # [B, 1, out_channels]
/// h = h + broadcast(t)
/// h = layer_norm(h, w=norm2, b=norm2_b)
/// h = silu(h)
/// h = conv2(h)                                # Linear out_channels → out_channels
/// skip = shortcut(x) if in_channels != out_channels else x
/// output = h + skip
/// ```
pub struct ResNetBlock {
    pub norm1_w: Parameter,
    pub norm1_b: Parameter,
    pub conv1: Linear,
    pub norm2_w: Parameter,
    pub norm2_b: Parameter,
    pub conv2: Linear,
    pub time_emb_proj: Linear,
    pub shortcut: Option<Linear>,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub eps: f32,
}

impl ResNetBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let n1w = Tensor::ones(vec![in_channels], dtype, device)?;
        let n1b = Tensor::zeros(vec![in_channels], dtype, device)?;
        let n2w = Tensor::ones(vec![out_channels], dtype, device)?;
        let n2b = Tensor::zeros(vec![out_channels], dtype, device)?;
        let shortcut = if in_channels != out_channels {
            Some(Linear::from_init(
                in_channels, out_channels, false,
                InitMethod::KaimingUniform { fan_in: in_channels, a: 0.0 },
                InitMethod::Zeros, device, dtype, 99,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1_w: Parameter::new(n1w),
            norm1_b: Parameter::new(n1b),
            conv1: Linear::from_init(
                in_channels, out_channels, true,
                InitMethod::KaimingUniform { fan_in: in_channels, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            norm2_w: Parameter::new(n2w),
            norm2_b: Parameter::new(n2b),
            conv2: Linear::from_init(
                out_channels, out_channels, true,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1,
            )?,
            time_emb_proj: Linear::from_init(
                time_emb_dim, out_channels, true,
                InitMethod::KaimingUniform { fan_in: time_emb_dim, a: 0.0 },
                InitMethod::Zeros, device, dtype, 2,
            )?,
            shortcut,
            in_channels,
            hidden_size: out_channels,
            eps: 1e-5,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        norm1_w: Tensor, norm1_b: Tensor,
        conv1_w: Tensor, conv1_b: Option<Tensor>,
        norm2_w: Tensor, norm2_b: Tensor,
        conv2_w: Tensor, conv2_b: Option<Tensor>,
        time_emb_proj_w: Tensor, time_emb_proj_b: Option<Tensor>,
        shortcut_w: Option<Tensor>,
        eps: f32,
    ) -> Result<Self> {
        let conv1 = Linear::new(conv1_w, conv1_b)?;
        let conv2 = Linear::new(conv2_w, conv2_b)?;
        let time_emb_proj = Linear::new(time_emb_proj_w, time_emb_proj_b)?;
        let shortcut = shortcut_w.map(|w| Linear::new(w, None)).transpose()?;
        let in_channels = conv1.in_features();
        let hidden_size = conv1.out_features();
        Ok(Self {
            norm1_w: Parameter::new(norm1_w),
            norm1_b: Parameter::new(norm1_b),
            conv1,
            norm2_w: Parameter::new(norm2_w),
            norm2_b: Parameter::new(norm2_b),
            conv2,
            time_emb_proj,
            shortcut,
            in_channels,
            hidden_size,
            eps,
        })
    }

    /// `x: [B, T, in_channels]`, `time_emb: [B, time_emb_dim]`.
    pub fn forward(&self, x: &Tensor, time_emb: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.in_channels {
            return Err(SynaptixError::Unsupported("ResNetBlock: expects x [B, T, in_channels]"));
        }
        let h = layer_norm(x, Some(&self.norm1_w.tensor()), Some(&self.norm1_b.tensor()), self.eps)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;

        let t = time_emb.silu()?;
        let t = self.time_emb_proj.forward(&t)?;
        let t = t.unsqueeze(1)?;
        let t_b = t.expand(&[x.dims()[0], x.dims()[1], self.hidden_size])?.contiguous()?;
        let h = h.add(&t_b)?;

        let h = layer_norm(&h, Some(&self.norm2_w.tensor()), Some(&self.norm2_b.tensor()), self.eps)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;

        let skip = match &self.shortcut {
            Some(s) => s.forward(x)?,
            None => x.clone(),
        };
        h.add(&skip)
    }
}
