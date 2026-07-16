use synaptix_core::{dtype::DType, tensor::Tensor};

use crate::audiovae::AudioVae;
use crate::cfm::{self, CfmOptions};
use crate::config::VoxConfig;
use crate::fsq::Fsq;
use crate::loader::{Lin, VoxCheckpoint};
use crate::locdit::LocDit;
use crate::locenc::LocEnc;
use crate::minicpm::{dims_from_lm, MiniCpm};
use crate::VoxError;

pub struct InferOptions {
    pub min_len: usize,
    pub max_len: usize,
    pub n_timesteps: usize,
    pub cfg_value: f32,
    pub seed: u64,
}

pub struct VoxCpmModel {
    base_lm: MiniCpm,
    residual_lm: MiniCpm,
    feat_encoder: LocEnc,
    feat_decoder: LocDit,
    fsq: Fsq,
    pub audio_vae: AudioVae,
    embed_tokens: Tensor,
    enc_to_lm: Lin,
    lm_to_dit: Lin,
    res_to_dit: Lin,
    fusion_concat: Lin,
    stop_proj: Lin,
    stop_head: Lin,
    pub config: VoxConfig,
    compute: DType,
}

impl VoxCpmModel {
    pub fn load(ck: &VoxCheckpoint) -> Result<Self, VoxError> {
        let cfg = ck.config.clone();
        let lm = &cfg.lm_config;
        let device = ck.device;
        let max_seq = cfg.max_length;

        let base_rope = MiniCpm::build_longrope(lm, lm.kv_channels, max_seq, device)?;
        let enc_rope = MiniCpm::build_longrope(lm, cfg.encoder_config.kv_channels, max_seq, device)?;
        let dit_rope = MiniCpm::build_longrope(lm, cfg.dit_config.kv_channels, max_seq, device)?;

        let base_lm = MiniCpm::load(ck, "base_lm", dims_from_lm(lm, lm.num_hidden_layers), Some(base_rope))?;
        let residual_lm = MiniCpm::load(
            ck,
            "residual_lm",
            dims_from_lm(lm, cfg.residual_lm_num_layers),
            None,
        )?;
        let feat_encoder = LocEnc::load(ck, enc_rope)?;
        let feat_decoder = LocDit::load(ck, dit_rope)?;
        let fsq = Fsq::load(ck)?;
        let audio_vae = AudioVae::load(ck)?;

        Ok(Self {
            base_lm,
            residual_lm,
            feat_encoder,
            feat_decoder,
            fsq,
            audio_vae,
            embed_tokens: ck.get("base_lm.embed_tokens.weight")?,
            enc_to_lm: Lin::load_direct(ck, "enc_to_lm_proj", true)?,
            lm_to_dit: Lin::load_direct(ck, "lm_to_dit_proj", true)?,
            res_to_dit: Lin::load_direct(ck, "res_to_dit_proj", true)?,
            fusion_concat: Lin::load_direct(ck, "fusion_concat_proj", true)?,
            stop_proj: Lin::load_direct(ck, "stop_proj", true)?,
            stop_head: Lin::load_direct(ck, "stop_head", false)?,
            config: cfg,
            compute: ck.compute,
        })
    }

    fn embed(&self, ids: &Tensor) -> Result<Tensor, VoxError> {
        let l = ids.dims()[1];
        let e = match self.embed_tokens.embed_gather(ids) {
            Ok(e) => e,
            Err(_) => self.embed_tokens.index_select(0, &ids.reshape((l,))?)?,
        };
        Ok(e.reshape((1usize, l, self.config.lm_config.hidden_size))?)
    }

    pub fn feat_to_lm(&self, audio_feat: &Tensor) -> Result<Tensor, VoxError> {
        let h = self.feat_encoder.forward(audio_feat)?;
        self.enc_to_lm.forward(&h)
    }

    pub fn infer(
        &self,
        text_token: &Tensor,
        text_mask: &Tensor,
        audio_feat: &Tensor,
        audio_mask: &Tensor,
        context_len: usize,
        opts: &InferOptions,
    ) -> Result<(Tensor, usize), VoxError> {
        let feat_dim = self.config.feat_dim;
        let patch = self.config.patch_size;
        let l = text_token.dims()[1];

        let feat_embed = self.feat_to_lm(audio_feat)?;
        let text_embed = self.embed(text_token)?;
        let tmask3 = text_mask.unsqueeze(2)?;
        let amask3 = audio_mask.unsqueeze(2)?;
        let combined = text_embed
            .broadcast_mul(&tmask3)?
            .add(&feat_embed.broadcast_mul(&amask3)?)?;

        let mut base_cache = self.base_lm.make_cache(1)?;
        let enc = self.base_lm.prefill(&combined, &mut base_cache)?;
        let enc = self
            .fsq
            .forward(&enc)?
            .broadcast_mul(&amask3)?
            .add(&enc.broadcast_mul(&tmask3)?)?;
        let mut lm_hidden = enc.narrow(1, l - 1, 1)?.squeeze(1)?.contiguous()?;

        let masked_feat = feat_embed.broadcast_mul(&amask3)?;
        let residual_in = self.fusion_concat.forward(&Tensor::cat(&[&enc, &masked_feat], 2)?)?;
        let mut res_cache = self.residual_lm.make_cache(1)?;
        let res_out = self.residual_lm.prefill(&residual_in, &mut res_cache)?;
        let mut residual_hidden = res_out.narrow(1, l - 1, 1)?.squeeze(1)?.contiguous()?;

        let mut prefix_cond = audio_feat.narrow(1, l - 1, 1)?.squeeze(1)?.contiguous()?;

        let mut seq: Vec<Tensor> = Vec::new();
        for j in 0..context_len {
            let idx = l - context_len + j;
            seq.push(audio_feat.narrow(1, idx, 1)?.contiguous()?);
        }

        let cfm_opts = CfmOptions {
            n_timesteps: opts.n_timesteps,
            cfg_value: opts.cfg_value,
            ..CfmOptions::default()
        };

        for i in 0..opts.max_len {
            let mu = Tensor::cat(
                &[&self.lm_to_dit.forward(&lm_hidden)?, &self.res_to_dit.forward(&residual_hidden)?],
                1,
            )?;
            let cond = prefix_cond.transpose(1, 2)?.contiguous()?;
            let pred = cfm::sample(
                &self.feat_decoder,
                &mu,
                &cond,
                patch,
                feat_dim,
                &cfm_opts,
                self.embed_tokens.device(),
                self.compute,
                opts.seed.wrapping_add(i as u64),
            )?;

            let pred_t = pred.unsqueeze(1)?;
            let curr_embed = self.feat_to_lm(&pred_t)?;
            let curr_2d = curr_embed.squeeze(1)?.contiguous()?;
            seq.push(pred_t.contiguous()?);
            prefix_cond = pred;

            if i > opts.min_len && self.stop_flag(&lm_hidden)? {
                break;
            }

            let stepped = self.base_lm.step(&curr_embed, &mut base_cache)?.squeeze(1)?.contiguous()?;
            lm_hidden = self.fsq.forward(&stepped)?;
            let res_in = self
                .fusion_concat
                .forward(&Tensor::cat(&[&lm_hidden, &curr_2d], 1)?)?;
            residual_hidden = self
                .residual_lm
                .step(&res_in.unsqueeze(1)?, &mut res_cache)?
                .squeeze(1)?
                .contiguous()?;
        }

        let cat = Tensor::cat(&seq.iter().collect::<Vec<_>>(), 1)?;
        let tg = cat.dims()[1];
        let feat_pred = cat
            .permute([0usize, 3, 1, 2])?
            .contiguous()?
            .reshape((1usize, feat_dim, tg * patch))?;
        Ok((feat_pred, context_len))
    }

    fn stop_flag(&self, lm_hidden: &Tensor) -> Result<bool, VoxError> {
        let h = self.stop_proj.forward(lm_hidden)?.silu()?;
        let logits = self.stop_head.forward(&h)?;
        let v = logits.to_dtype(DType::F32)?.reshape((2usize,))?.to_vec1::<f32>()?;
        Ok(v[1] > v[0])
    }
}
