use synaptix_core::tensor::Tensor;

use crate::loader::{Lin, VoxCheckpoint};
use crate::minicpm::{dims_from_sub, MiniCpm};
use crate::VoxError;

pub struct LocEnc {
    special_token: Tensor,
    in_proj: Lin,
    encoder: MiniCpm,
    hidden: usize,
}

impl LocEnc {
    pub fn load(ck: &VoxCheckpoint, rope: synaptix_ops::pos::rope_cache::RopeCache) -> Result<Self, VoxError> {
        let cfg = &ck.config;
        let dims = dims_from_sub(&cfg.encoder_config, &cfg.lm_config);
        let encoder = MiniCpm::load(ck, "feat_encoder.encoder", dims, Some(rope))?;
        Ok(Self {
            special_token: ck.get("feat_encoder.special_token")?,
            in_proj: Lin::load(ck, "feat_encoder", "in_proj", true)?,
            encoder,
            hidden: cfg.encoder_config.hidden_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let d = x.dims();
        let (b, t, p, feat) = (d[0], d[1], d[2], d[3]);
        let flat = x.contiguous()?.reshape((b * t * p, feat))?;
        let proj = self.in_proj.forward(&flat)?;
        let proj = proj.reshape((b * t, p, self.hidden))?;

        let cls = self
            .special_token
            .reshape((1usize, 1, self.hidden))?
            .expand((b * t, 1usize, self.hidden))?
            .contiguous()?;
        let seq = Tensor::cat(&[&cls, &proj], 1)?;
        let enc = self.encoder.forward(&seq, false)?;
        let cls_out = enc.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
        Ok(cls_out.reshape((b, t, self.hidden))?)
    }
}
