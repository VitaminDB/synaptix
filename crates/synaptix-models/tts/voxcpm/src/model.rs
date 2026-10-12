use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::audiovae::AudioVae;
use crate::cfm::{self, CfmOptions};
use crate::config::VoxConfig;
use crate::fsq::Fsq;
use crate::loader::{Lin, VoxCheckpoint};
use crate::locdit::LocDit;
use crate::locenc::LocEnc;
use crate::minicpm::{dims_from_lm, KvCache, MiniCpm, StaticKv};
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
        let device = self.embed_tokens.device();
        // Постоянные формы графов (NPU): префилл кусками и шаги декода по
        // KV-кэшу постоянной ёмкости — каждая операция собирается один раз
        // на все фразы. Входы, контекст и итог — на хосте.
        let fixed = fixed_shapes(device);
        let cap = kv_cap(l.div_ceil(CHUNK) * CHUNK + opts.max_len + 1);
        let (mut lm_hidden, mut residual_hidden, mut base_cache, mut res_cache) = if fixed {
            self.prefill_fixed(text_token, text_mask, audio_feat, audio_mask, l, cap)?
        } else {
            self.prefill_dyn(text_token, text_mask, audio_feat, audio_mask, l)?
        };
        let to_host = |t: Tensor| -> Result<Tensor, VoxError> { Ok(if fixed { t.to_device(Device::Cpu)? } else { t }) };
        let feat_src = to_host(audio_feat.clone())?;

        let mut prefix_cond = feat_src.narrow(1, l - 1, 1)?.squeeze(1)?.contiguous()?.to_device(device)?;

        let mut seq: Vec<Tensor> = Vec::new();
        for j in 0..context_len {
            let idx = l - context_len + j;
            seq.push(feat_src.narrow(1, idx, 1)?.contiguous()?);
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
            seq.push(to_host(pred_t.contiguous()?)?);
            prefix_cond = pred;

            if i > opts.min_len && self.stop_flag(&lm_hidden)? {
                break;
            }

            let stepped = base_cache.step(&self.base_lm, &curr_embed)?.squeeze(1)?.contiguous()?;
            lm_hidden = self.fsq.forward(&stepped)?;
            let res_in = self
                .fusion_concat
                .forward(&Tensor::cat(&[&lm_hidden, &curr_2d], 1)?)?;
            residual_hidden = res_cache
                .step(&self.residual_lm, &res_in.unsqueeze(1)?)?
                .squeeze(1)?
                .contiguous()?;
            synaptix_core::backend::registry::flush(device)?;
        }

        let cat = Tensor::cat(&seq.iter().collect::<Vec<_>>(), 1)?;
        let tg = cat.dims()[1];
        let feat_pred = cat
            .permute([0usize, 3, 1, 2])?
            .contiguous()?
            .reshape((1usize, feat_dim, tg * patch))?;
        Ok((feat_pred, context_len))
    }

    #[allow(clippy::type_complexity)]
    fn prefill_dyn(
        &self,
        text_token: &Tensor,
        text_mask: &Tensor,
        audio_feat: &Tensor,
        audio_mask: &Tensor,
        l: usize,
    ) -> Result<(Tensor, Tensor, Cache, Cache), VoxError> {
        let feat_embed = self.feat_to_lm(audio_feat)?;
        let text_embed = self.embed(text_token)?;
        let tmask3 = text_mask.unsqueeze(2)?;
        let amask3 = audio_mask.unsqueeze(2)?;
        let combined = text_embed.broadcast_mul(&tmask3)?.add(&feat_embed.broadcast_mul(&amask3)?)?;

        let mut base_cache = self.base_lm.make_cache(1)?;
        let enc = self.base_lm.prefill(&combined, &mut base_cache)?;
        let enc = self.fsq.forward(&enc)?.broadcast_mul(&amask3)?.add(&enc.broadcast_mul(&tmask3)?)?;
        let lm_hidden = enc.narrow(1, l - 1, 1)?.squeeze(1)?.contiguous()?;

        let masked_feat = feat_embed.broadcast_mul(&amask3)?;
        let residual_in = self.fusion_concat.forward(&Tensor::cat(&[&enc, &masked_feat], 2)?)?;
        let mut res_cache = self.residual_lm.make_cache(1)?;
        let res_out = self.residual_lm.prefill(&residual_in, &mut res_cache)?;
        let residual_hidden = res_out.narrow(1, l - 1, 1)?.squeeze(1)?.contiguous()?;
        Ok((lm_hidden, residual_hidden, Cache::Dyn(base_cache), Cache::Dyn(res_cache)))
    }

    /// Префилл кусками по [`CHUNK`] строк (вход добивается нулями — causal-маска
    /// не даёт добивке влиять на значимые строки, её K/V затираются шагами).
    #[allow(clippy::type_complexity)]
    fn prefill_fixed(
        &self,
        text_token: &Tensor,
        text_mask: &Tensor,
        audio_feat: &Tensor,
        audio_mask: &Tensor,
        l: usize,
        cap: usize,
    ) -> Result<(Tensor, Tensor, Cache, Cache), VoxError> {
        let dev = self.embed_tokens.device();
        let host = |t: &Tensor| t.to_device(Device::Cpu);
        let n = l.div_ceil(CHUNK);
        let (tok, tm, ft, am) = pad_inputs(&host(text_token)?, &host(text_mask)?, &host(audio_feat)?, &host(audio_mask)?, n * CHUNK - l)?;
        let mut base = self.base_lm.static_cache(cap, self.compute, dev)?;
        let mut res = self.residual_lm.static_cache(cap, self.compute, dev)?;
        let (last, row) = ((l - 1) / CHUNK, (l - 1) % CHUNK);
        let mut picked = None;
        for c in 0..n {
            let piece = |t: &Tensor| -> Result<Tensor, VoxError> { Ok(t.narrow(1, c * CHUNK, CHUNK)?.contiguous()?.to_device(dev)?) };
            let (tok, tm3, ft, am3) = (piece(&tok)?, piece(&tm)?.unsqueeze(2)?, piece(&ft)?, piece(&am)?.unsqueeze(2)?);
            let feat_embed = self.feat_to_lm(&ft)?;
            let combined = self.embed(&tok)?.broadcast_mul(&tm3)?.add(&feat_embed.broadcast_mul(&am3)?)?;
            let enc = self.base_lm.forward_static(&combined, &mut base, c * CHUNK)?;
            let enc = self.fsq.forward(&enc)?.broadcast_mul(&am3)?.add(&enc.broadcast_mul(&tm3)?)?;
            let masked_feat = feat_embed.broadcast_mul(&am3)?;
            let residual_in = self.fusion_concat.forward(&Tensor::cat(&[&enc, &masked_feat], 2)?)?;
            let r = self.residual_lm.forward_static(&residual_in, &mut res, c * CHUNK)?;
            if c == last {
                // Строка l − 1 — на хосте: срез с позицией в графе дал бы граф на каждую позицию.
                let pick = |t: &Tensor| -> Result<Tensor, VoxError> { Ok(host(t)?.narrow(1, row, 1)?.squeeze(1)?.contiguous()?.to_device(dev)?) };
                picked = Some((pick(&enc)?, pick(&r)?));
            }
            synaptix_core::backend::registry::flush(dev)?;
        }
        base.len = l;
        res.len = l;
        let (lm_hidden, residual_hidden) = picked.ok_or_else(|| VoxError::Other("пустой вход".into()))?;
        Ok((lm_hidden, residual_hidden, Cache::Static(base), Cache::Static(res)))
    }

    fn stop_flag(&self, lm_hidden: &Tensor) -> Result<bool, VoxError> {
        let h = self.stop_proj.forward(lm_hidden)?.silu()?;
        let logits = self.stop_head.forward(&h)?;
        let v = logits.to_dtype(DType::F32)?.reshape((2usize,))?.to_vec1::<f32>()?;
        Ok(v[1] > v[0])
    }
}

/// Кусок префилла при постоянных формах графов.
const CHUNK: usize = 32;

fn fixed_shapes(device: Device) -> bool {
    matches!(device, Device::Npu(_)) || std::env::var_os("VOXCPM_FIXED_SHAPES").is_some()
}

/// Ёмкость KV-кэша постоянной формы: 1024 хватает фразе до ~2 мин, длиннее — 4096 и 8192.
fn kv_cap(need: usize) -> usize {
    [1024, 4096, 8192].into_iter().find(|&c| c >= need).unwrap_or(need)
}

#[allow(clippy::type_complexity)]
fn pad_inputs(
    tokens: &Tensor,
    tmask: &Tensor,
    feat: &Tensor,
    amask: &Tensor,
    n: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor), VoxError> {
    let pad = |t: &Tensor| -> Result<Tensor, VoxError> {
        if n == 0 {
            return Ok(t.clone());
        }
        let mut d = t.dims().to_vec();
        d[1] = n;
        let z = Tensor::zeros(d, t.dtype(), t.device())?;
        Ok(Tensor::cat(&[t, &z], 1)?)
    };
    Ok((pad(tokens)?, pad(tmask)?, pad(feat)?, pad(amask)?))
}

enum Cache {
    Dyn(KvCache),
    Static(StaticKv),
}

impl Cache {
    fn step(&mut self, lm: &MiniCpm, x: &Tensor) -> Result<Tensor, VoxError> {
        match self {
            Cache::Dyn(c) => lm.step(x, c),
            Cache::Static(c) => {
                let start = c.len;
                lm.forward_static(x, c, start)
            }
        }
    }
}
