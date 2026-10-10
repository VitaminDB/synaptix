//! Нативный Wav2Vec2-энкодер + CTC-голова (batch = 1, без масок: все кадры
//! валидны, как у HF при отсутствии `attention_mask`).

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::log_softmax_dim;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv1d;
use synaptix_ops::norm::layer_norm;

use crate::config::Wav2Vec2Config;
use crate::loader::Wav2Vec2Weights;
use crate::{Result, Wav2Vec2Error};

pub(crate) struct Ln {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f32,
}

impl Ln {
    fn load(w: &Wav2Vec2Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?,
            bias: w.get(&format!("{prefix}.bias"))?,
            eps: w.config.layer_norm_eps as f32,
        })
    }

    /// LayerNorm по последней оси.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(layer_norm(x, Some(&self.weight), Some(&self.bias), self.eps)?)
    }
}

pub(crate) struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    fn load(w: &Wav2Vec2Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?,
            bias: w.get(&format!("{prefix}.bias"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.linear(&self.weight)?.broadcast_add(&self.bias)?)
    }
}

/// Нормализация после свёртки feature extractor'а.
pub(crate) enum ConvNorm {
    None,
    /// LayerNorm по каналам (`feat_extract_norm = "layer"`).
    Layer(Ln),
    /// GroupNorm с num_groups == channels: нормализация каждого канала по
    /// времени (`feat_extract_norm = "group"`, только первый слой).
    Group(Ln),
}

pub(crate) struct ConvLayer {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub norm: ConvNorm,
}

impl ConvLayer {
    /// `[B, Cin, L] → [B, Cout, L']`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = conv1d(x, &self.weight, self.bias.as_ref(), self.stride, 0)?;
        let h = match &self.norm {
            ConvNorm::None => h,
            ConvNorm::Layer(ln) => {
                let t = h.transpose(1, 2)?.contiguous()?;
                ln.forward(&t)?.transpose(1, 2)?.contiguous()?
            }
            ConvNorm::Group(ln) => {
                // По оси времени: [B,C,L], статистики по L, аффинно по C.
                let x32 = h.to_dtype(DType::F32)?;
                let mean = x32.mean_keepdim(2)?;
                let centered = x32.broadcast_sub(&mean)?;
                let var = centered.sqr()?.mean_keepdim(2)?;
                let normed = centered.broadcast_mul(&var.add_scalar(ln.eps)?.sqrt()?.recip()?)?;
                let w = ln.weight.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;
                let b = ln.bias.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;
                normed.broadcast_mul(&w)?.broadcast_add(&b)?.to_dtype(h.dtype())?
            }
        };
        Ok(h.gelu_exact()?)
    }
}

/// Позиционная grouped-свёртка (kernel K, pad K/2, groups G) + SamePad + GELU.
pub(crate) struct PosConv {
    pub weight: Tensor,
    pub bias: Tensor,
    pub groups: usize,
    pub kernel: usize,
}

impl PosConv {
    /// `[B, T, D] → [B, T, D]` (только позиционная добавка, без residual).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?.contiguous()?; // [B,D,T]
        let (d, t) = (xt.dims()[1], xt.dims()[2]);
        let cg = d / self.groups;
        let pad = self.kernel / 2;
        let mut outs = Vec::with_capacity(self.groups);
        for g in 0..self.groups {
            let xg = xt.narrow(1, g * cg, cg)?.contiguous()?;
            let wg = self.weight.narrow(0, g * cg, cg)?.contiguous()?;
            let bg = self.bias.narrow(0, g * cg, cg)?.contiguous()?;
            outs.push(conv1d(&xg, &wg, Some(&bg), 1, pad)?);
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        let mut h = Tensor::cat(&refs, 1)?; // [B,D,T(+1)]
        // SamePad: при чётном ядре свёртка даёт T+1 кадр — последний отбрасывается.
        if h.dims()[2] > t {
            h = h.narrow(2, 0, t)?.contiguous()?;
        }
        let h = h.gelu_exact()?;
        Ok(h.transpose(1, 2)?.contiguous()?)
    }
}

pub(crate) struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    fn heads(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        Ok(x.reshape(vec![d[0], d[1], self.n_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?
            .contiguous()?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, s) = (d[0], d[1]);
        let q = self.heads(&self.q.forward(x)?)?;
        let k = self.heads(&self.k.forward(x)?)?;
        let v = self.heads(&self.v.forward(x)?)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let a = scaled_dot_attention(&q, &k, &v, scale, None)?;
        let merged = a
            .permute(vec![0, 2, 1, 3])?
            .contiguous()?
            .reshape(vec![b, s, self.n_heads * self.head_dim])?;
        self.out.forward(&merged)
    }
}

pub(crate) struct EncoderLayer {
    pub attn_norm: Ln,
    pub attn: Attention,
    pub ff_norm: Ln,
    pub ff_in: Linear,
    pub ff_out: Linear,
}

impl EncoderLayer {
    fn ff(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.ff_in.forward(x)?.gelu_exact()?;
        self.ff_out.forward(&h)
    }

    /// pre-LN (`do_stable_layer_norm`).
    fn forward_stable(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.attn_norm.forward(x)?)?;
        let x = x.add(&h)?;
        let h = self.ff(&self.ff_norm.forward(&x)?)?;
        Ok(x.add(&h)?)
    }

    /// post-LN (base-модели): `attn_norm` = layer_norm, `ff_norm` = final_layer_norm.
    fn forward_post(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(x)?;
        let x = self.attn_norm.forward(&x.add(&h)?)?;
        let h = self.ff(&x)?;
        self.ff_norm.forward(&x.add(&h)?)
    }
}

/// Промежуточные тензоры прямого прохода (для гейтов и отладки).
pub struct ForwardTrace {
    /// Выход feature extractor'а `[1, C, T]`.
    pub features: Tensor,
    /// Feature projection `[1, T, D]`.
    pub proj: Tensor,
    /// Позиционная добавка `[1, T, D]`.
    pub pos: Tensor,
    /// Последний hidden state энкодера `[1, T, D]`.
    pub hidden: Tensor,
    /// Логиты `lm_head` `[1, T, V]` (до log-softmax).
    pub logits: Tensor,
}

pub struct Wav2Vec2Model {
    pub(crate) config: Wav2Vec2Config,
    pub(crate) convs: Vec<ConvLayer>,
    pub(crate) proj_norm: Ln,
    pub(crate) proj: Linear,
    pub(crate) pos_conv: PosConv,
    pub(crate) enc_norm: Ln,
    pub(crate) layers: Vec<EncoderLayer>,
    pub(crate) lm_head: Option<Linear>,
    device: Device,
    dtype: DType,
}

impl Wav2Vec2Model {
    pub fn load(w: &Wav2Vec2Weights) -> Result<Self> {
        let cfg = w.config.clone();
        let mut convs = Vec::with_capacity(cfg.conv_dim.len());
        for i in 0..cfg.conv_dim.len() {
            let p = format!("feature_extractor.conv_layers.{i}");
            let norm = match (cfg.feat_extract_norm.as_str(), i) {
                ("layer", _) => ConvNorm::Layer(Ln::load(w, &format!("{p}.layer_norm"))?),
                ("group", 0) => ConvNorm::Group(Ln::load(w, &format!("{p}.layer_norm"))?),
                _ => ConvNorm::None,
            };
            convs.push(ConvLayer {
                weight: w.get(&format!("{p}.conv.weight"))?,
                bias: if cfg.conv_bias { Some(w.get(&format!("{p}.conv.bias"))?) } else { None },
                stride: cfg.conv_stride[i],
                norm,
            });
        }

        let pos_conv = PosConv {
            weight: load_pos_conv_weight(w)?,
            bias: w.get("encoder.pos_conv_embed.conv.bias")?,
            groups: cfg.num_conv_pos_embedding_groups,
            kernel: cfg.num_conv_pos_embeddings,
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = |s: &str| format!("encoder.layers.{i}.{s}");
            layers.push(EncoderLayer {
                attn_norm: Ln::load(w, &p("layer_norm"))?,
                attn: Attention {
                    q: Linear::load(w, &p("attention.q_proj"))?,
                    k: Linear::load(w, &p("attention.k_proj"))?,
                    v: Linear::load(w, &p("attention.v_proj"))?,
                    out: Linear::load(w, &p("attention.out_proj"))?,
                    n_heads: cfg.num_attention_heads,
                    head_dim: cfg.head_dim(),
                },
                ff_norm: Ln::load(w, &p("final_layer_norm"))?,
                ff_in: Linear::load(w, &p("feed_forward.intermediate_dense"))?,
                ff_out: Linear::load(w, &p("feed_forward.output_dense"))?,
            });
        }

        let lm_head = if w.contains("lm_head.weight") {
            Some(Linear::load(w, "lm_head")?)
        } else {
            None
        };

        Ok(Self {
            proj_norm: Ln::load(w, "feature_projection.layer_norm")?,
            proj: Linear::load(w, "feature_projection.projection")?,
            enc_norm: Ln::load(w, "encoder.layer_norm")?,
            convs,
            pos_conv,
            layers,
            lm_head,
            device: w.device,
            dtype: w.dtype,
            config: cfg,
        })
    }

    pub fn config(&self) -> &Wav2Vec2Config {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn has_ctc_head(&self) -> bool {
        self.lm_head.is_some()
    }

    /// Нормализованный PCM `[1, N]` → признаки `[1, C, T]`.
    pub fn extract_features(&self, wav: &Tensor) -> Result<Tensor> {
        let mut h = wav.unsqueeze(1)?; // [1,1,N]
        for c in &self.convs {
            h = c.forward(&h)?;
        }
        Ok(h)
    }

    /// Признаки `[1, C, T]` → проекция `[1, T, D]`.
    pub fn project(&self, features: &Tensor) -> Result<Tensor> {
        let h = features.transpose(1, 2)?.contiguous()?;
        self.proj.forward(&self.proj_norm.forward(&h)?)
    }

    /// Проекция `[1, T, D]` → (позиционная добавка, выход энкодера).
    pub fn encode_with_pos(&self, proj: &Tensor) -> Result<(Tensor, Tensor)> {
        let pos = self.pos_conv.forward(proj)?;
        let mut h = proj.add(&pos)?;
        if self.config.do_stable_layer_norm {
            for l in &self.layers {
                h = l.forward_stable(&h)?;
            }
            h = self.enc_norm.forward(&h)?;
        } else {
            h = self.enc_norm.forward(&h)?;
            for l in &self.layers {
                h = l.forward_post(&h)?;
            }
        }
        Ok((pos, h))
    }

    /// Hidden state `[1, T, D]` → логиты `[1, T, V]`.
    pub fn head(&self, hidden: &Tensor) -> Result<Tensor> {
        let head = self
            .lm_head
            .as_ref()
            .ok_or_else(|| Wav2Vec2Error::Config("у модели нет lm_head (не CTC-чекпойнт)".into()))?;
        head.forward(hidden)
    }

    /// Полный прогон с промежуточными тензорами.
    pub fn forward_trace(&self, wav: &Tensor) -> Result<ForwardTrace> {
        let features = self.extract_features(wav)?;
        let proj = self.project(&features)?;
        let (pos, hidden) = self.encode_with_pos(&proj)?;
        let logits = self.head(&hidden)?;
        Ok(ForwardTrace { features, proj, pos, hidden, logits })
    }

    /// Нормализованный PCM `[1, N]` → log-probs `[1, T, V]`.
    pub fn log_probs(&self, wav: &Tensor) -> Result<Tensor> {
        let features = self.extract_features(wav)?;
        let proj = self.project(&features)?;
        let (_, hidden) = self.encode_with_pos(&proj)?;
        let logits = self.head(&hidden)?;
        Ok(log_softmax_dim(&logits.to_dtype(DType::F32)?, 2)?)
    }
}

/// Вес позиционной свёртки. Weight-norm (`weight = g · v / ‖v‖`, норма по всем
/// осям кроме оси ядра, dim=2) сворачивается при загрузке. Поддержаны старые
/// имена (`weight_g`/`weight_v`), parametrizations (`original0`/`original1`)
/// и уже свёрнутый `weight`.
fn load_pos_conv_weight(w: &Wav2Vec2Weights) -> Result<Tensor> {
    const P: &str = "encoder.pos_conv_embed.conv";
    if w.contains(&format!("{P}.weight")) {
        return w.get(&format!("{P}.weight"));
    }
    let (gn, vn) = if w.contains(&format!("{P}.weight_g")) {
        (format!("{P}.weight_g"), format!("{P}.weight_v"))
    } else {
        (
            format!("{P}.parametrizations.weight.original0"),
            format!("{P}.parametrizations.weight.original1"),
        )
    };
    let g = w.get_raw(&gn)?;
    let v = w.get_raw(&vn)?;
    let vd = v.dims().to_vec(); // [Cout, Cin/g, K]
    if vd.len() != 3 || g.dims().iter().product::<usize>() != vd[2] {
        return Err(Wav2Vec2Error::Load(format!(
            "pos_conv weight-norm: g {:?}, v {:?} (ожидается норма по dim=2)",
            g.dims(),
            vd
        )));
    }
    let k = vd[2];
    let gh = g.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let mut vh = v.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let mut norms = vec![0f64; k];
    for (i, &x) in vh.iter().enumerate() {
        norms[i % k] += (x as f64) * (x as f64);
    }
    let scale: Vec<f32> = (0..k).map(|j| (gh[j] as f64 / norms[j].sqrt().max(1e-12)) as f32).collect();
    for (i, x) in vh.iter_mut().enumerate() {
        *x *= scale[i % k];
    }
    let t = Tensor::from_vec(vh, vd, w.device)?;
    Ok(t.to_dtype(w.dtype)?)
}
