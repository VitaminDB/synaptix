//! T5-XXL encoder (google/t5-v1_1-xxl) — FLUX `text_encoder_2`.
//!
//! Encoder-only, bit-exact к HF `T5EncoderModel`. Особенности T5:
//! - T5LayerNorm = RMSNorm (variance-only, f32-upcast, `*weight`, без `1+w`/bias).
//! - attention БЕЗ масштаба `1/sqrt(d_kv)` (масштаб впитан в веса).
//! - relative position bias: считается один раз (block 0), шарится во все слои.
//! - gated-gelu FFN: `gelu_new(wi_0·x) * (wi_1·x) → wo`.
//! - НЕТ масштаба эмбеддингов, НЕТ BOS, FLUX не маскирует padding.

use synaptix_core::{
    device::Device,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;
use synaptix_ops::activation::gelu_tanh;
use synaptix_ops::attention::softmax_dim;
use synaptix_ops::norm::rms_norm;

#[derive(Debug, Clone)]
pub struct T5Config {
    pub d_model: usize,
    pub d_ff: usize,
    pub d_kv: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub num_buckets: usize,
    pub max_distance: usize,
    pub eps: f32,
}

impl T5Config {
    /// google/t5-v1_1-xxl (FLUX text_encoder_2).
    pub fn xxl() -> Self {
        Self {
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            num_buckets: 32,
            max_distance: 128,
            eps: 1e-6,
        }
    }
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.d_kv
    }
}

/// T5 relative-position bucket (bidirectional encoder). Точный порт
/// `_relative_position_bucket`: num_buckets//2=16, max_exact=8; малые `|rp|<8`
/// → напрямую, большие → `8 + trunc(log(|rp|/8)/log(16)*8)` clamp до 15; +16 если rp>0.
fn relative_position_bucket(rp: i64, num_buckets: i64, max_distance: i64) -> i64 {
    let mut ret = 0i64;
    let nb = num_buckets / 2; // 16
    if rp > 0 {
        ret += nb;
    }
    let n = rp.abs();
    let max_exact = nb / 2; // 8
    if n < max_exact {
        ret + n
    } else {
        let large = ((n as f64 / max_exact as f64).ln()
            / (max_distance as f64 / max_exact as f64).ln()
            * (nb - max_exact) as f64) as i64; // trunc к нулю = .to(long)
        let rp_if_large = (max_exact + large).min(nb - 1);
        ret + rp_if_large
    }
}

/// position_bias `[1, H, S, S]` = lookup `rel_bias[bucket(k-q)]`, permute в head-major.
fn build_position_bias(
    rel_bias_weight: &Tensor, // [num_buckets, H]
    s: usize,
    num_heads: usize,
    cfg: &T5Config,
    device: Device,
) -> Result<Tensor> {
    let mut buckets = vec![0u32; s * s];
    for q in 0..s {
        for k in 0..s {
            let b = relative_position_bucket(
                (k as i64) - (q as i64),
                cfg.num_buckets as i64,
                cfg.max_distance as i64,
            );
            buckets[q * s + k] = b as u32;
        }
    }
    let idx = Tensor::from_vec(buckets, (s * s,), device)?;
    let vals = rel_bias_weight.index_select(0, &idx)?; // [s*s, H]
    // [s*s,H] -> [s,s,H] -> [H,s,s] -> [1,H,s,s]
    vals.reshape((s, s, num_heads))?
        .permute([2, 0, 1])?
        .contiguous()?
        .unsqueeze(0)
}

struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: usize,
    d_kv: usize,
}

impl T5Attention {
    fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, s) = (d[0], d[1]);
        let (h, dh) = (self.num_heads, self.d_kv);
        // [b,s,inner] -> [b,h,s,dh]
        let to_heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, s, h, dh))?.transpose(1, 2)?.contiguous()
        };
        let q = to_heads(self.q.forward(x)?)?;
        let k = to_heads(self.k.forward(x)?)?;
        let v = to_heads(self.v.forward(x)?)?;
        // scores = q @ k^T (БЕЗ масштаба) + bias -> softmax -> @ v
        let scores = q.matmul(&k.transpose(2, 3)?)?; // [b,h,s,s]
        let scores = scores.broadcast_add(bias)?; // bias [1,h,s,s]
        let attn = softmax_dim(&scores, 3)?;
        let out = attn.matmul(&v)?; // [b,h,s,dh]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, s, h * dh))?;
        self.o.forward(&out)
    }
}

struct T5Block {
    ln0: Tensor,
    attn: T5Attention,
    ln1: Tensor,
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
    eps: f32,
}

impl T5Block {
    fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let n = rms_norm(x, &self.ln0, self.eps)?;
        let a = self.attn.forward(&n, bias)?;
        let x = x.add(&a)?;
        let n2 = rms_norm(&x, &self.ln1, self.eps)?;
        let g = gelu_tanh(&self.wi_0.forward(&n2)?)?;
        let l = self.wi_1.forward(&n2)?;
        let ff = self.wo.forward(&g.mul(&l)?)?;
        x.add(&ff)
    }
}

/// Источник весов T5 для стриминга блоков: имя тензора → тензор на
/// устройстве энкодера. Нужен владеющий и потокобезопасный — следующий блок
/// грузится в отдельном потоке параллельно счёту текущего.
pub type T5WeightSource = std::sync::Arc<dyn Fn(&str) -> Result<Tensor> + Send + Sync>;

pub struct T5Encoder {
    embed: Tensor, // shared.weight [vocab, d_model]
    rel_bias: Tensor,
    /// Резидентный префикс блоков; `blocks.len()..num_layers` стримятся.
    blocks: Vec<T5Block>,
    final_ln: Tensor,
    config: T5Config,
    stream: Option<T5WeightSource>,
}

fn load_block<F>(cfg: &T5Config, i: usize, get: &F) -> Result<T5Block>
where
    F: Fn(&str) -> Result<Tensor> + ?Sized,
{
    let lin = |name: &str| -> Result<Linear> { Linear::new(get(name)?, None) };
    let p = format!("encoder.block.{i}");
    Ok(T5Block {
        ln0: get(&format!("{p}.layer.0.layer_norm.weight"))?,
        attn: T5Attention {
            q: lin(&format!("{p}.layer.0.SelfAttention.q.weight"))?,
            k: lin(&format!("{p}.layer.0.SelfAttention.k.weight"))?,
            v: lin(&format!("{p}.layer.0.SelfAttention.v.weight"))?,
            o: lin(&format!("{p}.layer.0.SelfAttention.o.weight"))?,
            num_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
        },
        ln1: get(&format!("{p}.layer.1.layer_norm.weight"))?,
        wi_0: lin(&format!("{p}.layer.1.DenseReluDense.wi_0.weight"))?,
        wi_1: lin(&format!("{p}.layer.1.DenseReluDense.wi_1.weight"))?,
        wo: lin(&format!("{p}.layer.1.DenseReluDense.wo.weight"))?,
        eps: cfg.eps,
    })
}

impl T5Encoder {
    pub fn config(&self) -> &T5Config {
        &self.config
    }

    pub fn load<F>(cfg: &T5Config, get: &F) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let embed = get("shared.weight")?;
        let rel_bias =
            get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(load_block(cfg, i, get)?);
        }
        let final_ln = get("encoder.final_layer_norm.weight")?;
        Ok(Self { embed, rel_bias, blocks, final_ln, config: cfg.clone(), stream: None })
    }

    /// Байт на блок в `bytes_per_param` (2 для BF16).
    pub fn block_bytes(cfg: &T5Config, bytes_per_param: usize) -> usize {
        let (d, inner, ff) = (cfg.d_model, cfg.inner_dim(), cfg.d_ff);
        (4 * d * inner + 3 * d * ff) * bytes_per_param
    }

    /// Как [`Self::load`], но на карту кладутся блоки, пока свободной VRAM
    /// больше `reserve` + блок; остальные читаются из `get` по одному прямо в
    /// forward (следующий — параллельно счёту текущего). Проход энкодера
    /// одноразовый, поэтому копия на хосте не нужна: T5-XXL (9,5 ГБ в BF16)
    /// проходит и на карте в 7 ГБ.
    pub fn load_budgeted(cfg: &T5Config, get: T5WeightSource, device: Device, reserve: usize) -> Result<Self> {
        let embed = get("shared.weight")?;
        let rel_bias = get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")?;
        let final_ln = get("encoder.final_layer_norm.weight")?;
        let blk = Self::block_bytes(cfg, 2);
        let free = || match device {
            Device::Cuda(o) => synaptix_core::device::cuda::mem_info(o).map(|(f, _)| f).unwrap_or(0),
            _ => usize::MAX,
        };
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if free() < reserve + blk {
                if let Device::Cuda(o) = device {
                    let _ = synaptix_core::device::cuda::synchronize_all(o);
                    let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
                }
                if free() < reserve + blk {
                    eprintln!(
                        "[FLUX] T5: на карте {i}/{} блоков, остальные читаются из источника в forward",
                        cfg.num_layers
                    );
                    break;
                }
            }
            blocks.push(load_block(cfg, i, get.as_ref())?);
        }
        let stream = (blocks.len() < cfg.num_layers).then_some(get);
        Ok(Self { embed, rel_bias, blocks, final_ln, config: cfg.clone(), stream })
    }

    /// Сколько блоков лежит на карте.
    pub fn resident_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Блоки по порядку: резидентные как есть, остальные грузятся из
    /// источника, следующий — на loader-стриме параллельно счёту текущего.
    fn for_each_block<F>(&self, mut body: F) -> Result<()>
    where
        F: FnMut(&T5Block) -> Result<()>,
    {
        for b in &self.blocks {
            body(b)?;
        }
        let first = self.blocks.len();
        let n = self.config.num_layers;
        if first >= n {
            return Ok(());
        }
        let get = self
            .stream
            .as_ref()
            .ok_or_else(|| SynaptixError::Other("T5: стриминг блоков без источника весов".into()))?;
        let dev = self.embed.device();
        let ls = match dev {
            Device::Cuda(o) => Some(synaptix_core::device::cuda::loader_stream(o)?),
            _ => None,
        };
        let cfg = &self.config;
        let load = |idx: usize, on_loader: bool| -> Result<T5Block> {
            if let (true, Some(ls)) = (on_loader, ls.as_ref()) {
                synaptix_core::device::cuda::set_alloc_stream(Some(ls.clone()));
                let r = load_block(cfg, idx, get.as_ref());
                let _ = ls.synchronize();
                synaptix_core::device::cuda::set_alloc_stream(None);
                r
            } else {
                load_block(cfg, idx, get.as_ref())
            }
        };
        let mut staged = Some(load(first, false));
        for idx in first..n {
            let cur = staged.take().expect("staged")?;
            let load = &load;
            let (step, next) = std::thread::scope(|sp| {
                let h = (idx + 1 < n).then(|| sp.spawn(move || load(idx + 1, true)));
                let step = body(&cur);
                let next = h.map(|h| {
                    h.join().unwrap_or_else(|_| Err(SynaptixError::Other("T5: поток префетча блока упал".into())))
                });
                (step, next)
            });
            if let Device::Cuda(o) = dev {
                if let Ok(cs) = synaptix_core::device::cuda::default_stream(o) {
                    let _ = cs.synchronize();
                }
            }
            drop(cur);
            step?;
            staged = next;
        }
        Ok(())
    }

    /// `input_ids: [B, S]` (U32) → `last_hidden_state: [B, S, d_model]`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let d = input_ids.dims();
        let (b, s) = (d[0], d[1]);
        if input_ids.device() != self.embed.device() {
            return Err(SynaptixError::device_mismatch(
                input_ids.device(),
                self.embed.device(),
            ));
        }
        // embedding lookup (без масштаба): [b*s] -> [b*s, d] -> [b,s,d]
        let ids_flat = input_ids.reshape((b * s,))?;
        let h = self
            .embed
            .index_select(0, &ids_flat)?
            .reshape((b, s, self.config.d_model))?;
        let bias = build_position_bias(
            &self.rel_bias,
            s,
            self.config.num_heads,
            &self.config,
            input_ids.device(),
        )?;
        let mut h = h;
        self.for_each_block(|blk| {
            h = blk.forward(&h, &bias)?;
            Ok(())
        })?;
        rms_norm(&h, &self.final_ln, self.config.eps)
    }
}
