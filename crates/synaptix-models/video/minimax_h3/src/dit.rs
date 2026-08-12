use std::sync::Arc;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_nn::module::Module as _;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::adaln::{AdalnCache, AdalnPlan, AdalnProj, ModSegment, TimeEmbedder};
use crate::config::{ADALN_CHUNKS, FINAL_ADALN_CHUNKS, H3Config};
use crate::loader::H3Checkpoint;
use crate::rope::{read_inv_freq, RopeTables};
use crate::runtime;
use crate::H3Error;

type R<T> = Result<T, SynaptixError>;

const LIN_CHUNK_ROWS: usize = 16384;

pub struct Lin(QuantLinear);

impl Lin {
    pub fn load(
        ckpt: &H3Checkpoint,
        key: &str,
        bias: bool,
        qdt: DType,
        compute: DType,
    ) -> Result<Self, H3Error> {
        let mut w = ckpt.get_raw(&format!("{key}.weight"))?;
        if let Some(d) = ckpt.lora_delta(key, w.dtype())? {
            w = w.add(&d)?;
        }
        if let Some(d) = ckpt.lora_diff(key, w.dtype())? {
            w = w.add(&d)?;
        }
        let b = if bias { Some(ckpt.get_raw(&format!("{key}.bias"))?) } else { None };
        Ok(Self(QuantLinear::build(w, b, qdt, compute).map_err(H3Error::from)?))
    }

    pub fn load_qkv(
        ckpt: &H3Checkpoint,
        key: &str,
        heads: usize,
        head_dim: usize,
        qdt: DType,
        compute: DType,
    ) -> Result<Self, H3Error> {
        let mut w = ckpt.get_raw(&format!("{key}.weight"))?;
        if let Some(d) = ckpt.lora_delta(key, w.dtype())? {
            w = w.add(&d)?;
        }
        if let Some(d) = ckpt.lora_diff(key, w.dtype())? {
            w = w.add(&d)?;
        }
        let k = w.dims()[1];
        let w = w
            .reshape(vec![heads, 3, head_dim, k])?
            .permute([1, 0, 2, 3])?
            .contiguous()?
            .reshape(vec![heads * 3 * head_dim, k])?;
        Ok(Self(QuantLinear::build(w, None, qdt, compute).map_err(H3Error::from)?))
    }

    pub fn load_exact(
        ckpt: &H3Checkpoint,
        key: &str,
        bias: bool,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let mut w = ckpt.get_as(&format!("{key}.weight"), dtype)?;
        if let Some(d) = ckpt.lora_delta(key, dtype)? {
            w = w.add(&d)?;
        }
        if let Some(d) = ckpt.lora_diff(key, dtype)? {
            w = w.add(&d)?;
        }
        let b = if bias { Some(ckpt.get_as(&format!("{key}.bias"), dtype)?) } else { None };
        Ok(Self(QuantLinear::dense(w, b).map_err(H3Error::from)?))
    }

    pub fn from_quant(q: QuantLinear) -> Self {
        Self(q)
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let rows = x.dims()[0];
        if rows <= LIN_CHUNK_ROWS || !self.0.is_quant() {
            return self.0.forward(x);
        }
        let mut parts = Vec::with_capacity(rows.div_ceil(LIN_CHUNK_ROWS));
        let mut off = 0;
        while off < rows {
            let n = LIN_CHUNK_ROWS.min(rows - off);
            let chunk = x.narrow(0, off, n)?.contiguous()?;
            parts.push(self.0.forward(&chunk)?);
            off += n;
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 0)
    }
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> R<Tensor> {
    if let Ok(y) = x.rms_norm_fused(weight, eps, false) {
        return Ok(y);
    }
    synaptix_ops::norm::rms_norm::rms_norm(x, weight, eps)
}

pub struct Attention {
    qkv: Lin,
    out: Lin,
    q_norm: Tensor,
    k_norm: Tensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
    scale: f32,
}

impl Attention {
    pub fn load(
        ckpt: &H3Checkpoint,
        prefix: &str,
        cfg: &H3Config,
        qdt: DType,
        compute: DType,
    ) -> Result<Self, H3Error> {
        let heads = cfg.num_attention_heads;
        let head_dim = cfg.attention_head_dim;
        Ok(Self {
            qkv: Lin::load_qkv(ckpt, &format!("{prefix}.qkv_proj"), heads, head_dim, qdt, compute)?,
            out: Lin::load(ckpt, &format!("{prefix}.out_proj"), false, qdt, compute)?,
            q_norm: ckpt.get_as(&format!("{prefix}.q_norm.weight"), compute)?,
            k_norm: ckpt.get_as(&format!("{prefix}.k_norm.weight"), compute)?,
            heads,
            head_dim,
            eps: cfg.qk_norm_eps,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub fn forward(&self, x: &Tensor, rope: Option<&RopeTables>) -> R<Tensor> {
        let s = x.dims()[0];
        let inner = self.heads * self.head_dim;
        let qkv = self.qkv.forward(x)?;
        let q = qkv.narrow(1, 0, inner)?.contiguous()?;
        let k = qkv.narrow(1, inner, inner)?.contiguous()?;
        let v = qkv.narrow(1, 2 * inner, inner)?.contiguous()?;
        drop(qkv);

        let to_hsd = |t: Tensor| -> R<Tensor> {
            t.reshape(vec![s, self.heads, self.head_dim])?
                .transpose(0, 1)?
                .contiguous()
        };

        let q = rms_norm(&q.reshape(vec![s, self.heads, self.head_dim])?, &self.q_norm, self.eps)?;
        let k = rms_norm(&k.reshape(vec![s, self.heads, self.head_dim])?, &self.k_norm, self.eps)?;
        let mut q = q.transpose(0, 1)?.contiguous()?;
        let mut k = k.transpose(0, 1)?.contiguous()?;
        let v = to_hsd(v)?;

        if let Some(rt) = rope {
            q = rt.apply(&q).map_err(to_tensor_err)?;
            k = rt.apply(&k).map_err(to_tensor_err)?;
        }

        let q = q.reshape(vec![1, self.heads, s, self.head_dim])?;
        let k = k.reshape(vec![1, self.heads, s, self.head_dim])?;
        let v = v.reshape(vec![1, self.heads, s, self.head_dim])?;
        if runtime::h3_attn_prof() {
            let st = crate::pipeline::tensor_stats;
            eprintln!("[h3-attn] {} · {} · {}", st("q", &q), st("k", &k), st("v", &v));
        }

        let attn = match q.dtype() {
            DType::BF16 | DType::F16 => q
                .flash_attention(&k, &v, self.scale, false)
                .or_else(|_| scaled_dot_attention(&q, &k, &v, self.scale, None))?,
            _ => scaled_dot_attention(&q, &k, &v, self.scale, None)?,
        };
        let attn = attn
            .reshape(vec![self.heads, s, self.head_dim])?
            .transpose(0, 1)?
            .contiguous()?
            .reshape(vec![s, inner])?;
        self.out.forward(&attn)
    }
}

fn to_tensor_err(e: H3Error) -> SynaptixError {
    match e {
        H3Error::Tensor(t) => t,
        other => SynaptixError::Other(other.to_string()),
    }
}

pub struct Mlp {
    fc1: Lin,
    fc2: Lin,
    ffn: usize,
}

impl Mlp {
    pub fn load(
        ckpt: &H3Checkpoint,
        prefix: &str,
        cfg: &H3Config,
        qdt: DType,
        compute: DType,
    ) -> Result<Self, H3Error> {
        Ok(Self {
            fc1: Lin::load(ckpt, &format!("{prefix}.fc1"), false, qdt, compute)?,
            fc2: Lin::load(ckpt, &format!("{prefix}.fc2"), false, qdt, compute)?,
            ffn: cfg.ffn_hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let rows = x.dims()[0];
        if rows <= LIN_CHUNK_ROWS {
            return self.forward_chunk(x);
        }
        let mut parts = Vec::with_capacity(rows.div_ceil(LIN_CHUNK_ROWS));
        let mut off = 0;
        while off < rows {
            let n = LIN_CHUNK_ROWS.min(rows - off);
            let chunk = x.narrow(0, off, n)?.contiguous()?;
            parts.push(self.forward_chunk(&chunk)?);
            off += n;
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 0)
    }

    fn forward_chunk(&self, x: &Tensor) -> R<Tensor> {
        let h = self.fc1.forward(x)?;
        let gate = h.narrow(1, 0, self.ffn)?.contiguous()?;
        let up = h.narrow(1, self.ffn, self.ffn)?.contiguous()?;
        drop(h);
        let act = gate.silu_and_mul(&up)?;
        if runtime::h3_mlp_prof() {
            let st = crate::pipeline::tensor_stats;
            eprintln!("[h3-mlp] {} · {} · {} · {}", st("вход", x), st("gate", &gate), st("up", &up), st("act", &act));
            if x.dims()[0] > 1000 {
                crate::pipeline::dump_tensor("mlp_in", x);
                crate::pipeline::dump_tensor("mlp_gate", &gate);
                crate::pipeline::dump_tensor("mlp_up", &up);
                crate::pipeline::dump_tensor("mlp_act", &act);
            }
        }
        self.fc2.forward(&act)
    }
}

pub struct RefinerBlock {
    norm1: Tensor,
    norm2: Tensor,
    attn: Attention,
    mlp: Mlp,
    eps: f32,
}

impl RefinerBlock {
    pub fn load(
        ckpt: &H3Checkpoint,
        idx: usize,
        cfg: &H3Config,
        qdt: DType,
        compute: DType,
    ) -> Result<Self, H3Error> {
        let p = format!("token_refiner.blocks.{idx}");
        Ok(Self {
            norm1: ckpt.get_as(&format!("{p}.norm1.weight"), compute)?,
            norm2: ckpt.get_as(&format!("{p}.norm2.weight"), compute)?,
            attn: Attention::load(ckpt, &format!("{p}.attn"), cfg, qdt, compute)?,
            mlp: Mlp::load(ckpt, &format!("{p}.mlp"), cfg, qdt, compute)?,
            eps: cfg.norm_eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let h = rms_norm(x, &self.norm1, self.eps)?;
        let x = self.attn.forward(&h, None)?.add(x)?;
        let h = rms_norm(&x, &self.norm2, self.eps)?;
        self.mlp.forward(&h)?.add(&x)
    }
}

pub struct TokenRefiner {
    blocks: Vec<RefinerBlock>,
    final_norm: Tensor,
    eps: f32,
}

impl TokenRefiner {
    pub fn load(
        ckpt: &H3Checkpoint,
        cfg: &H3Config,
        qdt: DType,
        compute: DType,
    ) -> Result<Self, H3Error> {
        let mut blocks = Vec::with_capacity(cfg.token_refiner_num_layers);
        for i in 0..cfg.token_refiner_num_layers {
            blocks.push(RefinerBlock::load(ckpt, i, cfg, qdt, compute)?);
        }
        Ok(Self {
            blocks,
            final_norm: ckpt.get_as("token_refiner.final_norm.weight", compute)?,
            eps: cfg.final_norm_eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = x.clone();
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        rms_norm(&h, &self.final_norm, self.eps)
    }
}

pub struct DiTBlock {
    norm1: Tensor,
    norm2: Tensor,
    attn: Attention,
    mlp: Mlp,
    eps: f32,
}

impl DiTBlock {
    pub fn load(
        ckpt: &H3Checkpoint,
        idx: usize,
        cfg: &H3Config,
        qdt: DType,
        compute: DType,
    ) -> Result<Self, H3Error> {
        let p = format!("blocks.{idx}");
        Ok(Self {
            norm1: ckpt.get_as(&format!("{p}.norm1.weight"), compute)?,
            norm2: ckpt.get_as(&format!("{p}.norm2.weight"), compute)?,
            attn: Attention::load(ckpt, &format!("{p}.attn"), cfg, qdt, compute)?,
            mlp: Mlp::load(ckpt, &format!("{p}.mlp"), cfg, qdt, compute)?,
            eps: cfg.norm_eps,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mods: &BlockMods,
        segments: &[ModSegment],
        rope: &RopeTables,
        idx: usize,
    ) -> R<Tensor> {
        let prof = runtime::h3_adaln_prof() && idx == runtime::prof_block();
        let st = crate::pipeline::tensor_stats;
        if prof {
            eprintln!("[h3-mod] {} · {}", st("shift_msa", &mods.shift_msa), st("scale_msa", &mods.scale_msa));
            eprintln!("[h3-mod] {} · {}", st("gate_msa", &mods.gate_msa), st("gate_mlp", &mods.gate_mlp));
            eprintln!("[h3-mod] {} · {}", st("shift_mlp", &mods.shift_mlp), st("scale_mlp", &mods.scale_mlp));
        }
        let h = rms_norm(x, &self.norm1, self.eps)?;
        if prof {
            eprintln!("[h3-mod] {}", st("norm1", &h));
        }
        let h = mod_scale_shift(&h, &mods.shift_msa, &mods.scale_msa, segments)?;
        if prof {
            eprintln!("[h3-mod] {}", st("после mod1", &h));
        }
        let a = self.attn.forward(&h, Some(rope))?;
        if prof {
            eprintln!("[h3-mod] {}", st("attn", &a));
        }
        let x = mod_gate(x, &a, &mods.gate_msa, segments)?;
        if prof {
            eprintln!("[h3-mod] {}", st("x после attn", &x));
        }

        let h = rms_norm(&x, &self.norm2, self.eps)?;
        let h = mod_scale_shift(&h, &mods.shift_mlp, &mods.scale_mlp, segments)?;
        let m = self.mlp.forward(&h)?;
        if prof {
            eprintln!("[h3-mod] {}", st("mlp", &m));
        }
        mod_gate(&x, &m, &mods.gate_mlp, segments)
    }
}

pub struct BlockMods {
    pub shift_msa: Tensor,
    pub scale_msa: Tensor,
    pub gate_msa: Tensor,
    pub shift_mlp: Tensor,
    pub scale_mlp: Tensor,
    pub gate_mlp: Tensor,
}

impl BlockMods {
    pub fn from_cache(cache: &AdalnCache, block: usize, step: usize) -> R<Self> {
        Ok(Self {
            shift_msa: cache.chunk(block, step, 0)?,
            scale_msa: cache.chunk(block, step, 1)?,
            gate_msa: cache.chunk(block, step, 2)?,
            shift_mlp: cache.chunk(block, step, 3)?,
            scale_mlp: cache.chunk(block, step, 4)?,
            gate_mlp: cache.chunk(block, step, 5)?,
        })
    }
}

fn row_of(t: &Tensor, row: usize) -> R<Tensor> {
    t.narrow(0, row, 1)?.contiguous()
}

fn check_coverage(rows: usize, segments: &[ModSegment]) -> R<()> {
    let mut cursor = 0usize;
    for seg in segments {
        if seg.start != cursor {
            return Err(SynaptixError::Other(format!(
                "модуляция: разрыв сегментов, ожидалось {cursor}, получено {}",
                seg.start
            )));
        }
        cursor = seg.stop;
    }
    if cursor != rows {
        return Err(SynaptixError::Other(format!(
            "модуляция: сегменты покрывают {cursor} строк из {rows}"
        )));
    }
    Ok(())
}

fn mod_scale_shift(
    h: &Tensor,
    shift: &Tensor,
    scale: &Tensor,
    segments: &[ModSegment],
) -> R<Tensor> {
    let dims = h.dims().to_vec();
    check_coverage(dims[0], segments)?;
    let mut out = Tensor::empty_uninit(vec![1, dims[0], dims[1]], h.dtype(), h.device())?;
    for seg in segments {
        let n = seg.stop - seg.start;
        if n == 0 {
            continue;
        }
        let part = h.narrow(0, seg.start, n)?.contiguous()?;
        let s = row_of(scale, seg.row)?;
        let sh = row_of(shift, seg.row)?;
        let y = match part.fused_mod_row(&s, &sh) {
            Ok(y) => y,
            Err(_) => part.broadcast_mul(&s.add_scalar(1.0)?)?.broadcast_add(&sh)?,
        };
        out.copy_rows_from(seg.start, &y.reshape(vec![1, n, dims[1]])?)?;
    }
    out.reshape(dims)
}

fn mod_gate(x: &Tensor, other: &Tensor, gate: &Tensor, segments: &[ModSegment]) -> R<Tensor> {
    let dims = x.dims().to_vec();
    check_coverage(dims[0], segments)?;
    let mut out = Tensor::empty_uninit(vec![1, dims[0], dims[1]], x.dtype(), x.device())?;
    for seg in segments {
        let n = seg.stop - seg.start;
        if n == 0 {
            continue;
        }
        let xp = x.narrow(0, seg.start, n)?.contiguous()?;
        let op = other.narrow(0, seg.start, n)?.contiguous()?;
        let g = row_of(gate, seg.row)?;
        let y = match xp.fused_gate_residual(&op, &g) {
            Ok(y) => y,
            Err(_) => xp.add(&op.broadcast_mul(&g)?)?,
        };
        out.copy_rows_from(seg.start, &y.reshape(vec![1, n, dims[1]])?)?;
    }
    out.reshape(dims)
}

pub struct FinalLayer {
    norm: Tensor,
    video_out: Lin,
    audio_out: Lin,
    eps: f32,
}

impl FinalLayer {
    pub fn load(ckpt: &H3Checkpoint, cfg: &H3Config, compute: DType) -> Result<Self, H3Error> {
        Ok(Self {
            norm: ckpt.get_as("final_layer.norm.weight", compute)?,
            video_out: Lin::load_exact(ckpt, "final_layer.video_out", true, DType::F32)?,
            audio_out: Lin::load_exact(ckpt, "final_layer.audio_out", true, DType::F32)?,
            eps: cfg.final_norm_eps,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        shift: &Tensor,
        scale: &Tensor,
        video_seg: (usize, usize, usize),
        audio_seg: (usize, usize, usize),
    ) -> R<(Tensor, Tensor)> {
        let head = |seg: (usize, usize, usize), lin: &Lin| -> R<Tensor> {
            let (a, b, row) = seg;
            let part = x.narrow(0, a, b - a)?.contiguous()?;
            let n = rms_norm(&part, &self.norm, self.eps)?;
            let s = row_of(scale, row)?;
            let sh = row_of(shift, row)?;
            let m = match n.fused_mod_row(&s, &sh) {
                Ok(y) => y,
                Err(_) => n.broadcast_mul(&s.add_scalar(1.0)?)?.broadcast_add(&sh)?,
            };
            lin.forward(&m.to_dtype(DType::F32)?)
        };
        Ok((head(video_seg, &self.video_out)?, head(audio_seg, &self.audio_out)?))
    }
}

pub struct H3Dit {
    pub cfg: H3Config,
    video_patch: Lin,
    audio_patch: Lin,
    condition: Lin,
    token_refiner: TokenRefiner,
    blocks: Vec<DiTBlock>,
    final_layer: FinalLayer,
    time_embedder: TimeEmbedder,
    adaln_blocks: Vec<AdalnProj>,
    adaln_final: AdalnProj,
    inv_freq: Vec<f32>,
    device: Device,
    compute: DType,
    stream: Option<Arc<H3Checkpoint>>,
}

impl H3Dit {
    pub fn load(
        ckpt: &H3Checkpoint,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, H3Error> {
        Self::load_with(ckpt, device, compute, quant, false)
    }

    pub fn load_with(
        ckpt: &H3Checkpoint,
        device: Device,
        compute: DType,
        quant: DType,
        keep_adaln: bool,
    ) -> Result<Self, H3Error> {
        let cfg = ckpt.config.clone();
        let nblocks = runtime::nblocks_cap().unwrap_or(cfg.num_layers).min(cfg.num_layers);

        let video_patch = Lin::load_exact(ckpt, "video_patch_proj", true, DType::F32)?;
        let audio_patch = Lin::load_exact(ckpt, "audio_patch_proj", true, DType::F32)?;
        let condition = Lin::load(ckpt, "condition_proj", true, quant, compute)?;
        let token_refiner = TokenRefiner::load(ckpt, &cfg, quant, compute)?;

        let mut blocks = Vec::with_capacity(nblocks);
        for i in 0..nblocks {
            blocks.push(DiTBlock::load(ckpt, i, &cfg, quant, compute)?);
        }
        let final_layer = FinalLayer::load(ckpt, &cfg, compute)?;

        let time_embedder = TimeEmbedder::new(
            ckpt.get_as("time_embedder.proj_in.weight", DType::F32)?,
            ckpt.get_as("time_embedder.proj_in.bias", DType::F32)?,
            ckpt.get_as("time_embedder.proj_out.weight", DType::F32)?,
            ckpt.get_as("time_embedder.proj_out.bias", DType::F32)?,
            cfg.timestep_input_dim,
        );

        let mut adaln_blocks = Vec::new();
        if keep_adaln {
            for i in 0..nblocks {
                adaln_blocks.push(load_adaln(ckpt, &format!("blocks.{i}.adaln_proj"), &cfg, ADALN_CHUNKS, crate::config::ADALN_MODALITIES)?);
            }
        }
        let adaln_final = load_adaln(
            ckpt,
            "final_layer.adaln_proj",
            &cfg,
            FINAL_ADALN_CHUNKS,
            1,
        )?;

        let inv_freq = read_inv_freq(&ckpt.get_as("rope.inv_freq", DType::F32)?)?;

        Ok(Self {
            cfg,
            video_patch,
            audio_patch,
            condition,
            token_refiner,
            blocks,
            final_layer,
            time_embedder,
            adaln_blocks,
            adaln_final,
            inv_freq,
            device,
            compute,
            stream: None,
        })
    }

    pub fn with_stream(mut self, ckpt: Arc<H3Checkpoint>) -> Self {
        self.stream = Some(ckpt);
        self
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn compute_dtype(&self) -> DType {
        self.compute
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    pub fn rope_tables(&self, positions: &[[f64; 3]]) -> Result<RopeTables, H3Error> {
        RopeTables::build(positions, &self.inv_freq, self.device)
    }

    pub fn refine_text(&self, states: &Tensor) -> R<Tensor> {
        if states.dims()[states.rank() - 1] == self.cfg.hidden_size {
            return Ok(states.clone());
        }
        let x = states.to_dtype(self.compute)?;
        let x = if x.rank() == 3 {
            x.reshape(vec![x.dims()[1], x.dims()[2]])?
        } else {
            x
        };
        let h = self.condition.forward(&x)?;
        self.token_refiner.forward(&h)
    }

    pub fn embed_video(&self, rows: &Tensor) -> R<Tensor> {
        self.video_patch.forward(&rows.to_dtype(DType::F32)?)?.to_dtype(self.compute)
    }

    pub fn embed_audio(&self, rows: &Tensor) -> R<Tensor> {
        self.audio_patch.forward(&rows.to_dtype(DType::F32)?)?.to_dtype(self.compute)
    }

    pub fn time_embed(&self, vals: &[f32]) -> R<Tensor> {
        self.time_embedder.forward(vals, self.device)
    }

    pub fn adaln_block(&self, idx: usize) -> Option<&AdalnProj> {
        self.adaln_blocks.get(idx)
    }

    pub fn adaln_final(&self) -> &AdalnProj {
        &self.adaln_final
    }

    pub fn checkpoint(&self) -> Option<&Arc<H3Checkpoint>> {
        self.stream.as_ref()
    }

    pub fn build_adaln_cache(
        &self,
        plan: &AdalnPlan,
        ckpt: &H3Checkpoint,
        cache_dtype: DType,
    ) -> Result<AdalnCache, H3Error> {
        let steps = plan.steps();
        let rows = plan.rows();
        let hidden = self.cfg.hidden_size;
        let t_embs: Vec<Tensor> = plan
            .timesteps
            .iter()
            .map(|v| self.time_embed(v).map_err(H3Error::from))
            .collect::<Result<_, _>>()?;

        let mut blocks = Vec::with_capacity(self.blocks.len());
        for i in 0..self.blocks.len() {
            let proj = match self.adaln_blocks.get(i) {
                Some(p) => p,
                None => &load_adaln(
                    ckpt,
                    &format!("blocks.{i}.adaln_proj"),
                    &self.cfg,
                    ADALN_CHUNKS,
                    crate::config::ADALN_MODALITIES,
                )?,
            };
            let mut per_step = Vec::with_capacity(steps);
            for t in &t_embs {
                per_step.push(proj.forward(t)?);
            }
            blocks.push(crate::adaln::stack_steps(per_step, cache_dtype)?);
        }

        let mut per_step = Vec::with_capacity(steps);
        for t in &t_embs {
            per_step.push(self.adaln_final.forward(t)?);
        }
        let final_layer = crate::adaln::stack_steps(per_step, cache_dtype)?;

        Ok(AdalnCache::new(blocks, final_layer, rows, plan.final_rows(), hidden, steps))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden: &Tensor,
        cache: &AdalnCache,
        step: usize,
        segments: &[ModSegment],
        rope: &RopeTables,
        video_seg: (usize, usize, usize),
        audio_seg: (usize, usize, usize),
    ) -> R<(Tensor, Tensor)> {
        let prof = runtime::h3_blk_prof();
        if prof {
            eprintln!("[h3-blk] вход · {}", crate::pipeline::tensor_stats("hidden", hidden));
        }
        if step == crate::pipeline::dump_step() {
            crate::pipeline::dump_tensor("hidden", hidden);
            crate::pipeline::dump_tensor("rope_cos", &rope.cos);
            crate::pipeline::dump_tensor("rope_sin", &rope.sin);
            crate::pipeline::dump_text(
                "layout",
                &format!(
                    "{{\"segments\": {:?}, \"video_seg\": {:?}, \"audio_seg\": {:?}, \"rot_dim\": {}}}",
                    segments.iter().map(|s| [s.start, s.stop, s.row]).collect::<Vec<_>>(),
                    [video_seg.0, video_seg.1, video_seg.2],
                    [audio_seg.0, audio_seg.1, audio_seg.2],
                    rope.rot_dim
                ),
            );
            for i in 0..self.blocks.len() {
                for c in 0..ADALN_CHUNKS {
                    crate::pipeline::dump_tensor(
                        &format!("mod_b{i}_c{c}"),
                        &cache.chunk(i, step, c)?,
                    );
                }
            }
        }
        let mut h = hidden.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            let mods = BlockMods::from_cache(cache, i, step)?;
            h = block.forward(&h, &mods, segments, rope, i)?;
            if prof {
                eprintln!("[h3-blk] блок {i} · {}", crate::pipeline::tensor_stats("h", &h));
            }
        }
        let shift = cache.final_chunk(step, 0)?;
        let scale = cache.final_chunk(step, 1)?;
        let (v, a) = self.final_layer.forward(&h, &shift, &scale, video_seg, audio_seg)?;
        if step == crate::pipeline::dump_step() {
            crate::pipeline::dump_tensor("h_last", &h);
            crate::pipeline::dump_tensor("v_out", &v);
            crate::pipeline::dump_tensor("a_out", &a);
            crate::pipeline::dump_tensor("final_shift", &shift);
            crate::pipeline::dump_tensor("final_scale", &scale);
        }
        if prof {
            eprintln!(
                "[h3-blk] голова video_seg {video_seg:?} audio_seg {audio_seg:?} · {} · {}",
                crate::pipeline::tensor_stats("v_out", &v),
                crate::pipeline::tensor_stats("a_out", &a)
            );
        }
        Ok((v, a))
    }
}

fn load_adaln(
    ckpt: &H3Checkpoint,
    prefix: &str,
    cfg: &H3Config,
    expand: usize,
    modalities: usize,
) -> Result<AdalnProj, H3Error> {
    let mut w = ckpt.get_as(&format!("{prefix}.linear.weight"), DType::F32)?;
    if let Some(d) = ckpt.lora_delta(&format!("{prefix}.linear"), DType::F32)? {
        w = w.add(&d)?;
    }
    if let Some(d) = ckpt.lora_diff(&format!("{prefix}.linear"), DType::F32)? {
        w = w.add(&d)?;
    }
    let b = ckpt.get_as(&format!("{prefix}.linear.bias"), DType::F32)?;
    Ok(AdalnProj::new(w, b, expand, modalities, cfg.hidden_size))
}

pub fn dit_resident_bytes(
    ckpt: &H3Checkpoint,
    quant: DType,
    compute: DType,
    include_adaln: bool,
) -> usize {
    let qbits = |n: usize, k: usize| -> Option<usize> {
        match quant {
            DType::NVFP4 if n % 64 == 0 && k % 64 == 0 => Some(n * k / 2 + n * k / 16),
            DType::MXFP8 if k % 32 == 0 => Some(n * k + n * k / 32),
            _ => None,
        }
    };
    let cbytes = compute.bytes_for_numel(1).max(1);
    let mut total = 0usize;
    for (name, dt, shape) in ckpt.infos() {
        if !include_adaln && name.contains("adaln_proj") && name.contains("blocks.") {
            continue;
        }
        let numel: usize = shape.iter().product();
        let quantizable = shape.len() == 2
            && name.ends_with(".weight")
            && (name.contains(".attn.") || name.contains(".mlp."))
            && dt != DType::F32;
        total += if quantizable {
            qbits(shape[0], shape[1]).unwrap_or(numel * cbytes)
        } else if dt == DType::F32 {
            numel * 4
        } else {
            numel * cbytes
        };
    }
    total
}
