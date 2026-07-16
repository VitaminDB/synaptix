use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

pub struct Mamba2Block {
    pub in_proj: Linear,
    pub conv1d_weight: Parameter,
    pub conv1d_bias: Option<Parameter>,
    pub out_proj: Linear,
    pub a_log: Parameter,
    pub d: Parameter,
    pub dt_bias: Parameter,
    pub norm_weight: Parameter,
    pub hidden_size: usize,
    pub d_state: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub d_conv: usize,
    pub norm_eps: f32,
}

impl Mamba2Block {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        d_state: usize,
        num_heads: usize,
        head_dim: usize,
        d_conv: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let d_inner = num_heads * head_dim;
        let in_dim = 2 * d_inner + 2 * d_state + num_heads;
        let a_log = crate::init::init_tensor(&[num_heads], InitMethod::Normal { mean: 0.0, std: 1.0 }, dtype, 0, device)?;
        let d = crate::init::init_tensor(&[num_heads], InitMethod::Ones, dtype, 1, device)?;
        let dt_bias = Tensor::zeros(vec![num_heads], dtype, device)?;
        let norm_w = Tensor::ones(vec![d_inner], dtype, device)?;
        let conv_w = crate::init::init_tensor(
            &[d_inner, 1, d_conv],
            InitMethod::Normal { mean: 0.0, std: 0.02 },
            dtype, 2, device,
        )?;
        let conv_b = Tensor::zeros(vec![d_inner], dtype, device)?;
        Ok(Self {
            in_proj: Linear::from_init(hidden_size, in_dim, false, InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 }, InitMethod::Zeros, device, dtype, 3)?,
            conv1d_weight: Parameter::new(conv_w),
            conv1d_bias: Some(Parameter::new(conv_b)),
            out_proj: Linear::from_init(d_inner, hidden_size, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 4)?,
            a_log: Parameter::new(a_log),
            d: Parameter::new(d),
            dt_bias: Parameter::new(dt_bias),
            norm_weight: Parameter::new(norm_w),
            hidden_size,
            d_state,
            num_heads,
            head_dim,
            d_conv,
            norm_eps: 1e-5,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        in_proj_w: Tensor,
        conv_w: Tensor,
        conv_b: Option<Tensor>,
        out_proj_w: Tensor,
        a_log: Tensor,
        d: Tensor,
        dt_bias: Tensor,
        norm_weight: Tensor,
        hidden_size: usize,
        d_state: usize,
        num_heads: usize,
        head_dim: usize,
        d_conv: usize,
        norm_eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            in_proj: Linear::new(in_proj_w, None)?,
            conv1d_weight: Parameter::new(conv_w),
            conv1d_bias: conv_b.map(Parameter::new),
            out_proj: Linear::new(out_proj_w, None)?,
            a_log: Parameter::new(a_log),
            d: Parameter::new(d),
            dt_bias: Parameter::new(dt_bias),
            norm_weight: Parameter::new(norm_weight),
            hidden_size,
            d_state,
            num_heads,
            head_dim,
            d_conv,
            norm_eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use synaptix_ops::activation::silu;
        use synaptix_ops::activation::softplus::softplus;
        use synaptix_ops::conv::causal_conv1d::causal_conv1d;
        use synaptix_ops::norm::rms_norm::rms_norm_silu_gated;
        use synaptix_ops::ssm::mamba::{mamba_step, MambaState};

        if x.rank() != 3 {
            return Err(SynaptixError::Unsupported("Mamba2Block: x must be [B, L, hidden]"));
        }
        let dims = x.dims();
        let (b_sz, l, _) = (dims[0], dims[1], dims[2]);
        let d_inner = self.num_heads * self.head_dim;

        let projected = self.in_proj.forward(x)?;
        let z = projected.narrow(2, 0, d_inner)?.contiguous()?;
        let x_proj = projected.narrow(2, d_inner, d_inner)?.contiguous()?;
        let b_mat = projected.narrow(2, 2 * d_inner, self.d_state)?.contiguous()?;
        let c_mat = projected.narrow(2, 2 * d_inner + self.d_state, self.d_state)?.contiguous()?;
        let dt_raw = projected.narrow(2, 2 * d_inner + 2 * self.d_state, self.num_heads)?.contiguous()?;

        let x_conv_in = x_proj.permute(vec![0, 2, 1])?.contiguous()?;
        let conv_b_ref = self.conv1d_bias.as_ref().map(|p| p.tensor());
        let x_conv = causal_conv1d(&x_conv_in, &self.conv1d_weight.tensor(), conv_b_ref.as_ref(), 1)?;
        let x_acted = silu(&x_conv)?;
        let x_seq = x_acted.permute(vec![0, 2, 1])?.contiguous()?;

        let dt_biased = dt_raw.broadcast_add(&self.dt_bias.tensor())?;
        let dt = softplus(&dt_biased, 1.0, 20.0)?;
        let a_full = self.a_log.tensor().neg()?.exp()?;
        let a_expanded = self.expand_a(&a_full)?;
        let d_expanded_static = self.expand_per_head_param(&self.d.tensor())?;

        let mut state = MambaState {
            h: Tensor::zeros(vec![b_sz, d_inner, self.d_state], x.dtype(), x.device())?,
            conv_buf: Tensor::zeros(vec![0], x.dtype(), x.device())?,
        };

        let mut ys: Vec<Tensor> = Vec::with_capacity(l);
        for t in 0..l {
            let xt = x_seq.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
            let bt = b_mat.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
            let ct = c_mat.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
            let dt_t = dt.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
            let dt_expanded = self.expand_per_head_batched(&dt_t)?;
            let y = mamba_step(&xt, &mut state, &a_expanded, &bt, &ct, &dt_expanded)?;
            let skip = d_expanded_static.broadcast_mul(&xt)?;
            let y_with_skip = y.add(&skip)?;
            ys.push(y_with_skip.unsqueeze(1)?);
        }
        let refs: Vec<&Tensor> = ys.iter().collect();
        let y_seq = Tensor::cat(&refs, 1)?;

        let gated = rms_norm_silu_gated(&y_seq, &z, &self.norm_weight.tensor(), self.norm_eps)?;
        self.out_proj.forward(&gated)
    }

    fn expand_per_head_batched(&self, x_b_h: &Tensor) -> Result<Tensor> {
        let dims = x_b_h.dims();
        let b = dims[0];
        let h_dim = self.head_dim;
        let x_unsqueezed = x_b_h.unsqueeze(2)?;
        let mut chunks: Vec<&Tensor> = Vec::with_capacity(h_dim);
        let owned: Vec<Tensor> = (0..h_dim).map(|_| x_unsqueezed.clone()).collect();
        for c in owned.iter() {
            chunks.push(c);
        }
        let tiled = Tensor::cat(&chunks, 2)?;
        tiled.reshape(vec![b, self.num_heads * h_dim])
    }

    fn expand_per_head_param(&self, x_h: &Tensor) -> Result<Tensor> {
        let h_dim = self.head_dim;
        let x_unsqueezed = x_h.unsqueeze(1)?;
        let owned: Vec<Tensor> = (0..h_dim).map(|_| x_unsqueezed.clone()).collect();
        let refs: Vec<&Tensor> = owned.iter().collect();
        let tiled = Tensor::cat(&refs, 1)?;
        tiled.reshape(vec![self.num_heads * h_dim])
    }

    fn expand_a(&self, a_heads: &Tensor) -> Result<Tensor> {
        let n = self.d_state;
        let a_flat = self.expand_per_head_param(a_heads)?;
        let a_col = a_flat.unsqueeze(1)?;
        let owned: Vec<Tensor> = (0..n).map(|_| a_col.clone()).collect();
        let refs: Vec<&Tensor> = owned.iter().collect();
        Tensor::cat(&refs, 1)
    }
}
