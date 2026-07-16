use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Conformer Self-Attention Module: Pre-LN + MHA + opt. relative positional
/// bias + residual.
///
/// `forward(x, rel_pos_bias)` где `rel_pos_bias` (опц.) — Shaw 2018 / T5-style
/// per-head additive bias на attention scores, форма
/// `[num_heads, S, S]` или broadcastable `[1, num_heads, S, S]`. Без bias —
/// эквивалентно стандартному MHA (как в torchaudio Conformer).
///
/// Residual прибавляется внутри: `output = x + OutProj(MHA(LN(x), bias))`.
pub struct AttentionModule {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub eps: f32,
}

impl AttentionModule {
    pub fn new(hidden_size: usize, num_heads: usize, device: Device, dtype: DType) -> Result<Self> {
        if hidden_size % num_heads != 0 {
            return Err(SynaptixError::Unsupported(
                "AttentionModule: hidden_size must be divisible by num_heads",
            ));
        }
        let head_dim = hidden_size / num_heads;
        let norm_w = Tensor::ones(vec![hidden_size], dtype, device)?;
        let norm_b = Tensor::zeros(vec![hidden_size], dtype, device)?;
        Ok(Self {
            norm_w: Parameter::new(norm_w).with_name("norm.weight"),
            norm_b: Parameter::new(norm_b).with_name("norm.bias"),
            q_proj: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            k_proj: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            v_proj: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 2,
            )?,
            out_proj: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 3,
            )?,
            num_heads, head_dim, hidden_size, eps: 1e-5,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        q_w: Tensor, q_b: Option<Tensor>,
        k_w: Tensor, k_b: Option<Tensor>,
        v_w: Tensor, v_b: Option<Tensor>,
        o_w: Tensor, o_b: Option<Tensor>,
        num_heads: usize, eps: f32,
    ) -> Result<Self> {
        let q_proj = Linear::new(q_w, q_b)?;
        let k_proj = Linear::new(k_w, k_b)?;
        let v_proj = Linear::new(v_w, v_b)?;
        let out_proj = Linear::new(o_w, o_b)?;
        let hidden_size = q_proj.out_features();
        if hidden_size % num_heads != 0 {
            return Err(SynaptixError::Unsupported(
                "AttentionModule::from_weights: hidden_size must be divisible by num_heads",
            ));
        }
        let head_dim = hidden_size / num_heads;
        if norm_w.rank() != 1 || norm_w.dims()[0] != hidden_size
            || norm_b.rank() != 1 || norm_b.dims()[0] != hidden_size
        {
            return Err(SynaptixError::Unsupported(
                "AttentionModule::from_weights: norm_w/norm_b must be [hidden_size]",
            ));
        }
        Ok(Self {
            norm_w: Parameter::new(norm_w).with_name("norm.weight"),
            norm_b: Parameter::new(norm_b).with_name("norm.bias"),
            q_proj, k_proj, v_proj, out_proj,
            num_heads, head_dim, hidden_size, eps,
        })
    }

    /// Compute relative position indices `(i - j + max_distance)` clamped to
    /// `[0, 2*max_distance]`, форма `[S, S]`. Используется как индекс в
    /// learned bias table `[num_heads, 2*max_distance + 1]` для Shaw-style.
    ///
    /// Не часть `forward` — exposed чтобы пользователь мог сам собрать
    /// `rel_pos_bias [num_heads, S, S]` через gather по table.
    pub fn relative_position_indices(s: usize, max_distance: usize) -> Vec<i64> {
        let mut out = Vec::with_capacity(s * s);
        let m = max_distance as i64;
        for i in 0..s {
            for j in 0..s {
                let d = (i as i64) - (j as i64);
                let idx = (d + m).clamp(0, 2 * m);
                out.push(idx);
            }
        }
        out
    }

    /// `x: [B, S, hidden_size]`, `rel_pos_bias`: опц. `[num_heads, S, S]` или
    /// `[1, num_heads, S, S]`. `attn_mask`: опц. `[B, num_heads, S, S]` или
    /// broadcast-совместима, как в `scaled_dot_attention`.
    pub fn forward(
        &self,
        x: &Tensor,
        rel_pos_bias: Option<&Tensor>,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported(
                "AttentionModule: expects x [B, S, hidden_size]",
            ));
        }
        let b = x.dims()[0];
        let s = x.dims()[1];

        let h = layer_norm(
            x,
            Some(&self.norm_w.tensor()),
            Some(&self.norm_b.tensor()),
            self.eps,
        )?;

        let q = self.q_proj.forward(&h)?
            .reshape(vec![b, s, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let k = self.k_proj.forward(&h)?
            .reshape(vec![b, s, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let v = self.v_proj.forward(&h)?
            .reshape(vec![b, s, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // С rel_pos_bias или mask мы собираем effective mask и передаём в SDPA
        // как additive bias. scaled_dot_attention принимает mask с
        // broadcast-семантикой (добавляется к scores). Объединяем оба
        // источника через add, если есть.
        let combined = combine_bias(rel_pos_bias, attn_mask)?;
        let attn = scaled_dot_attention(&q, &k, &v, scale, combined.as_ref())?;

        let merged = attn.permute(vec![0, 2, 1, 3])?.contiguous()?
            .reshape(vec![b, s, self.hidden_size])?;
        let out = self.out_proj.forward(&merged)?;
        x.add(&out)
    }
}

fn combine_bias(a: Option<&Tensor>, b: Option<&Tensor>) -> Result<Option<Tensor>> {
    match (a, b) {
        (None, None) => Ok(None),
        (Some(t), None) | (None, Some(t)) => Ok(Some(t.clone())),
        (Some(a), Some(b)) => {
            // Broadcast-добавление двух additive bias'ов.
            Ok(Some(a.broadcast_add(b)?))
        }
    }
}

impl Module for AttentionModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        AttentionModule::forward(self, x, None, None)
    }
}
