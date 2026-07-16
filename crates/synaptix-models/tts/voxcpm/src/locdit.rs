use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::loader::{Lin, VoxCheckpoint};
use crate::minicpm::{dims_from_sub, MiniCpm};
use crate::VoxError;

const SIN_SCALE: f32 = 1000.0;

struct TimestepMlp {
    linear_1: Lin,
    linear_2: Lin,
}

impl TimestepMlp {
    fn load(ck: &VoxCheckpoint, prefix: &str) -> Result<Self, VoxError> {
        Ok(Self {
            linear_1: Lin::load(ck, prefix, "linear_1", true)?,
            linear_2: Lin::load(ck, prefix, "linear_2", true)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let h = self.linear_1.forward(x)?.silu()?;
        self.linear_2.forward(&h)
    }
}

pub struct LocDit {
    in_proj: Lin,
    cond_proj: Lin,
    out_proj: Lin,
    time_mlp: TimestepMlp,
    delta_time_mlp: TimestepMlp,
    decoder: MiniCpm,
    sin_freqs: Tensor,
    sin_dim: usize,
    hidden: usize,
    in_channels: usize,
    compute: DType,
}

impl LocDit {
    pub fn load(
        ck: &VoxCheckpoint,
        rope: synaptix_ops::pos::rope_cache::RopeCache,
    ) -> Result<Self, VoxError> {
        let cfg = &ck.config;
        let dims = dims_from_sub_dit(ck);
        let decoder = MiniCpm::load(ck, "feat_decoder.estimator.decoder", dims, Some(rope))?;
        let hidden = cfg.dit_config.hidden_dim;
        let sin_dim = hidden;
        let sin_freqs = build_sin_freqs(sin_dim, ck.device)?;
        Ok(Self {
            in_proj: Lin::load(ck, "feat_decoder.estimator", "in_proj", true)?,
            cond_proj: Lin::load(ck, "feat_decoder.estimator", "cond_proj", true)?,
            out_proj: Lin::load(ck, "feat_decoder.estimator", "out_proj", true)?,
            time_mlp: TimestepMlp::load(ck, "feat_decoder.estimator.time_mlp")?,
            delta_time_mlp: TimestepMlp::load(ck, "feat_decoder.estimator.delta_time_mlp")?,
            decoder,
            sin_freqs,
            sin_dim,
            hidden,
            in_channels: cfg.feat_dim,
            compute: ck.compute,
        })
    }

    fn proj_seq(&self, lin: &Lin, x_cl: &Tensor) -> Result<Tensor, VoxError> {
        let d = x_cl.dims();
        let (n, c, l) = (d[0], d[1], d[2]);
        let xt = x_cl.transpose(1, 2)?.contiguous()?.reshape((n * l, c))?;
        let y = lin.forward(&xt)?;
        Ok(y.reshape((n, l, self.hidden))?)
    }

    fn time_embed(&self, t: &Tensor, mlp: &TimestepMlp) -> Result<Tensor, VoxError> {
        let n = t.dims()[0];
        let args = t
            .to_dtype(DType::F32)?
            .reshape((n, 1usize))?
            .broadcast_mul(&self.sin_freqs.reshape((1usize, self.sin_dim / 2))?)?
            .mul_scalar(SIN_SCALE)?;
        let emb = Tensor::cat(&[&args.sin()?, &args.cos()?], 1)?.to_dtype(self.compute)?;
        mlp.forward(&emb)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor, VoxError> {
        let n = x.dims()[0];
        let p = x.dims()[2];
        let prefix = cond.dims()[2];

        let x_seq = self.proj_seq(&self.in_proj, x)?;
        let cond_seq = self.proj_seq(&self.cond_proj, cond)?;

        let t_e = self.time_embed(t, &self.time_mlp)?;
        let dt_e = self.time_embed(dt, &self.delta_time_mlp)?;
        let t_total = t_e.add(&dt_e)?;
        let t_tok = t_total.unsqueeze(1)?;

        let mu_tok = mu.reshape((n, mu.dims()[1] / self.hidden, self.hidden))?;
        let mu_count = mu_tok.dims()[1];

        let seq = Tensor::cat(&[&mu_tok, &t_tok, &cond_seq, &x_seq], 1)?;
        let hidden = self.decoder.forward(&seq, false)?;

        let start = prefix + mu_count + 1;
        let tail = hidden.narrow(1, start, p)?.contiguous()?;

        let out = self
            .out_proj
            .forward(&tail.reshape((n * p, self.hidden))?)?
            .reshape((n, p, self.in_channels))?;
        Ok(out.transpose(1, 2)?.contiguous()?)
    }
}

fn dims_from_sub_dit(ck: &VoxCheckpoint) -> crate::minicpm::Dims {
    let dit = &ck.config.dit_config;
    let sub = crate::config::SubTransformerConfig {
        hidden_dim: dit.hidden_dim,
        ffn_dim: dit.ffn_dim,
        num_heads: dit.num_heads,
        num_layers: dit.num_layers,
        kv_channels: dit.kv_channels,
    };
    dims_from_sub(&sub, &ck.config.lm_config)
}

fn build_sin_freqs(dim: usize, device: Device) -> Result<Tensor, VoxError> {
    let half = dim / 2;
    let emb_base = (10000f32).ln() / ((half - 1) as f32);
    let freqs: Vec<f32> = (0..half).map(|i| (-(i as f32) * emb_base).exp()).collect();
    Ok(Tensor::from_vec(freqs, half, device)?)
}
