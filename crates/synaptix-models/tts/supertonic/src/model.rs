//! Сети Supertonic 3, восстановленные по графам ONNX (`onnx/*.onnx` HF Supertone/supertonic-3).
//! Батч 1 без паддинга: маски текста и латента — единицы, поэтому опущены.
//! Раскладка тензоров `[B, C, T]` (как у ONNX), внутри блоков — `[B, T, C]`.

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::attention::softmax_dim;
use synaptix_ops::conv::{conv1d_dilated, depthwise_conv};
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::{Result, StError};

/// Отладка: `SYN_ST_DUMP=каталог` — промежуточные тензоры (f32, сырые), только 1-й вызов имени.
pub fn dump(name: &str, x: &Tensor) {
    if let Ok(dir) = std::env::var("SYN_ST_DUMP") {
        let p = format!("{dir}/{name}.f32");
        if std::path::Path::new(&p).exists() {
            return;
        }
        if let Ok(v) = x.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
            let _ = std::fs::write(p, v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>());
        }
    }
}

/// Веса одного компонента бандла (F32 на устройстве).
pub struct Weights {
    l: SynBundleLoader,
    names: Vec<String>,
    pub device: Device,
}

impl Weights {
    pub fn open(path: &Path, component: &str, device: Device) -> Result<Self> {
        let l = SynBundleLoader::open(path).map_err(|e| StError::Load(e.to_string()))?.with_component(component).with_device(device);
        let names: Vec<String> = l.names().iter().map(|s| s.to_string()).collect();
        if names.is_empty() {
            return Err(StError::Load(format!("в бандле нет компонента {component}")));
        }
        Ok(Self { l, names, device })
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.l.load_to(name, self.device, DType::F32).map_err(|e| StError::Load(format!("{name}: {e}")))
    }

    /// Первый вес, имя которого оканчивается на `suffix` (у экспорта ONNX к
    /// именам прилипают префиксы вроде `vector_estimator.tts.ttl.`).
    /// Совпадение — всё имя или его хвост по границе `.`; несколько разных
    /// кандидатов — ошибка (порядок имён в индексе бандла не задан, и «первый
    /// попавшийся» менялся бы от запуска к запуску).
    pub fn find(&self, suffix: &str) -> Result<Tensor> {
        if self.names.iter().any(|n| n == suffix) {
            return self.get(suffix);
        }
        let dotted = format!(".{suffix}");
        let mut c: Vec<&String> = self.names.iter().filter(|n| n.ends_with(&dotted)).collect();
        c.sort();
        match c.as_slice() {
            [one] => self.get(&(*one).clone()),
            [] => Err(StError::Load(format!("нет веса *{suffix}"))),
            many => Err(StError::Load(format!("вес *{suffix} неоднозначен: {:?}", &many[..many.len().min(4)]))),
        }
    }

    pub fn host(&self, suffix: &str) -> Result<Vec<f32>> {
        Ok(self.find(suffix)?.flatten_all()?.to_vec1::<f32>()?)
    }
}

/// Свёртка 1×1 `[out, in, 1]` или матрица ONNX — как линейный слой по последней оси.
fn lin_conv(w: &Weights, weight: &str, bias: Option<&str>) -> Result<QuantLinear> {
    let t = w.find(weight)?;
    let d = t.dims().to_vec();
    let t = t.reshape(vec![d[0], d[1]])?;
    let b = bias.map(|b| w.find(b)).transpose()?;
    Ok(QuantLinear::dense(t, b)?)
}

/// MatMul ONNX `x · W` (`W [in, out]`).
fn lin_mm(w: &Weights, weight: &str, bias: Option<&str>) -> Result<QuantLinear> {
    let t = w.find(weight)?.transpose(0, 1)?.contiguous()?;
    let b = bias.map(|b| w.find(b)).transpose()?;
    Ok(QuantLinear::dense(t, b)?)
}

fn t_bct(x: &Tensor) -> Result<Tensor> {
    Ok(x.transpose(1, 2)?.contiguous()?)
}

/// Паддинг краевыми значениями по оси времени (`Pad mode=edge`).
fn pad_edge(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let d = x.dims().to_vec();
    let mut parts = Vec::with_capacity(3);
    if left > 0 {
        parts.push(x.narrow(2, 0, 1)?.broadcast_as(vec![d[0], d[1], left])?.contiguous()?);
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(x.narrow(2, d[2] - 1, 1)?.broadcast_as(vec![d[0], d[1], right])?.contiguous()?);
    }
    if parts.len() == 1 {
        return Ok(x.clone());
    }
    let r: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&r, 2)?)
}

/// Вес depthwise-свёртки с дилатацией — нулями между отводами.
fn dilate(w: &Tensor, d: usize) -> Result<Tensor> {
    if d <= 1 {
        return Ok(w.clone());
    }
    let dims = w.dims().to_vec();
    let (c, k) = (dims[0], dims[2]);
    let host = w.flatten_all()?.to_vec1::<f32>()?;
    let kk = (k - 1) * d + 1;
    let mut out = vec![0f32; c * kk];
    for ch in 0..c {
        for j in 0..k {
            out[ch * kk + j * d] = host[ch * k + j];
        }
    }
    Ok(Tensor::from_vec(out, vec![c, 1, kk], w.device())?)
}

/// Блок ConvNeXt: depthwise (паддинг краями: симметрично или каузально) → LayerNorm
/// по каналам → 1×1 ×4 → GELU(erf) → 1×1 → γ → остаток.
pub struct ConvNeXt {
    dw: Tensor,
    dw_b: Tensor,
    left: usize,
    right: usize,
    ln_w: Tensor,
    ln_b: Tensor,
    pw1: QuantLinear,
    pw2: QuantLinear,
    gamma: Tensor,
}

impl ConvNeXt {
    /// `p` — префикс блока (`…convnext.N`), `dw` — имя depthwise (`dwconv` или `dwconv.net`).
    pub fn load(w: &Weights, p: &str, dw: &str, dilation: usize, causal: bool) -> Result<Self> {
        let k = w.find(&format!("{p}.{dw}.weight"))?;
        let ks = k.dims()[2];
        let total = (ks - 1) * dilation;
        let (left, right) = if causal { (total, 0) } else { (total / 2, total - total / 2) };
        let c = k.dims()[0];
        Ok(Self {
            dw: dilate(&k, dilation)?,
            dw_b: w.find(&format!("{p}.{dw}.bias"))?,
            left,
            right,
            ln_w: w.find(&format!("{p}.norm.norm.weight"))?,
            ln_b: w.find(&format!("{p}.norm.norm.bias"))?,
            pw1: lin_conv(w, &format!("{p}.pwconv1.weight"), Some(&format!("{p}.pwconv1.bias")))?,
            pw2: lin_conv(w, &format!("{p}.pwconv2.weight"), Some(&format!("{p}.pwconv2.bias")))?,
            gamma: w.find(&format!("{p}.gamma"))?.reshape(vec![c])?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = x.dims()[1];
        let y = depthwise_conv(&pad_edge(x, self.left, self.right)?, &self.dw, Some(&self.dw_b), 1, 0, c)?;
        let y = layer_norm(&t_bct(&y)?, Some(&self.ln_w), Some(&self.ln_b), 1e-6)?;
        let y = self.pw2.forward(&self.pw1.forward(&y)?.gelu_exact()?)?.broadcast_mul(&self.gamma)?;
        Ok(x.add(&t_bct(&y)?)?)
    }
}

pub fn convnext_stack(w: &Weights, p: &str, dw: &str, dilations: &[usize], causal: bool) -> Result<Vec<ConvNeXt>> {
    dilations.iter().enumerate().map(|(i, &d)| ConvNeXt::load(w, &format!("{p}.{i}"), dw, d, causal)).collect()
}

fn run_stack(s: &[ConvNeXt], x: &Tensor) -> Result<Tensor> {
    let mut h = x.clone();
    for b in s {
        h = b.forward(&h)?;
    }
    Ok(h)
}

/// Слой энкодера VITS: относительное внимание (окно 4, эмбеддинги общие для
/// голов), постнорма, FFN 1×1 с ReLU.
struct VitsLayer {
    q: QuantLinear,
    k: QuantLinear,
    v: QuantLinear,
    o: QuantLinear,
    rel_k: Tensor,
    rel_v: Tensor,
    ln1: (Tensor, Tensor),
    ffn1: QuantLinear,
    ffn2: QuantLinear,
    ln2: (Tensor, Tensor),
    heads: usize,
    window: usize,
}

pub struct VitsEncoder {
    layers: Vec<VitsLayer>,
    device: Device,
}

impl VitsEncoder {
    pub fn load(w: &Weights, p: &str, n_layers: usize, heads: usize) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let a = |s: &str| format!("{p}.attn_layers.{i}.{s}");
            let lin = |n: &str| lin_conv(w, &a(&format!("{n}.weight")), Some(&a(&format!("{n}.bias"))));
            let rel_k = w.find(&a("emb_rel_k"))?;
            let window = (rel_k.dims()[1] - 1) / 2;
            let d = rel_k.dims()[2];
            layers.push(VitsLayer {
                q: lin("conv_q")?,
                k: lin("conv_k")?,
                v: lin("conv_v")?,
                o: lin("conv_o")?,
                rel_k: rel_k.reshape(vec![2 * window + 1, d])?,
                rel_v: w.find(&a("emb_rel_v"))?.reshape(vec![2 * window + 1, d])?,
                ln1: (w.find(&format!("{p}.norm_layers_1.{i}.norm.weight"))?, w.find(&format!("{p}.norm_layers_1.{i}.norm.bias"))?),
                ffn1: lin_conv(w, &format!("{p}.ffn_layers.{i}.conv_1.weight"), Some(&format!("{p}.ffn_layers.{i}.conv_1.bias")))?,
                ffn2: lin_conv(w, &format!("{p}.ffn_layers.{i}.conv_2.weight"), Some(&format!("{p}.ffn_layers.{i}.conv_2.bias")))?,
                ln2: (w.find(&format!("{p}.norm_layers_2.{i}.norm.weight"))?, w.find(&format!("{p}.norm_layers_2.{i}.norm.bias"))?),
                heads,
                window,
            });
        }
        Ok(Self { layers, device: w.device })
    }

    /// `x` `[1, C, T]` → `[1, C, T]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dims()[2];
        let mut h = t_bct(x)?; // [1, T, C]
        // Индексы относительное ↔ абсолютное (окно ±w): для (i, j) — столбец
        // j − i + w таблицы [T, 2w+1] (вне окна — нулевой столбец 2w+1).
        let wdw = self.layers.first().map(|l| l.window).unwrap_or(4);
        let r = 2 * wdw + 1;
        let mut rel2abs = Vec::with_capacity(t * t);
        let mut abs2rel = Vec::with_capacity(t * r);
        for i in 0..t {
            for j in 0..t {
                let off = j as i64 - i as i64;
                rel2abs.push(if off.unsigned_abs() as usize <= wdw { (i * (r + 1)) as i64 + off + wdw as i64 } else { (i * (r + 1) + r) as i64 });
            }
            for k in 0..r {
                let j = i as i64 + k as i64 - wdw as i64;
                abs2rel.push(if j >= 0 && (j as usize) < t { (i * (t + 1)) as i64 + j } else { (i * (t + 1) + t) as i64 });
            }
        }
        let prof = std::env::var_os("SYN_ST_PROF").is_some();
        let sync = |x: &Tensor| { if prof { let _ = x.flatten_all().and_then(|t| t.narrow(0, 0, 1)).and_then(|t| t.to_vec1::<f32>()); } };
        let mut tp = std::time::Instant::now();
        let mut lap = |name: &str, x: &Tensor| { if prof { sync(x); eprintln!("  vits {name}: {:.1} мс", tp.elapsed().as_secs_f64() * 1e3); tp = std::time::Instant::now(); } };
        for l in &self.layers {
            let c = h.dims()[2];
            let nh = l.heads;
            let d = c / nh;
            let split = |y: Tensor| -> Result<Tensor> { Ok(y.reshape(vec![t, nh, d])?.permute(&[1, 0, 2])?.contiguous()?) };
            let q = split(l.q.forward(&h)?.reshape(vec![t, c])?)?.affine(1.0 / (d as f32).sqrt(), 0.0)?;
            let k = split(l.k.forward(&h)?.reshape(vec![t, c])?)?;
            let v = split(l.v.forward(&h)?.reshape(vec![t, c])?)?;
            lap("qkv", &v);
            let mut scores = q.matmul(&k.transpose(1, 2)?.contiguous()?)?; // [H, T, T]
            lap("qk", &scores);
            // относительные логиты: q · E_kᵀ → [H, T, 2w+1] → абсолютные позиции
            let qe = q.matmul(&l.rel_k.transpose(0, 1)?.contiguous()?.unsqueeze(0)?.broadcast_as(vec![nh, d, r])?.contiguous()?)?;
            let qe = Tensor::cat(&[&qe, &Tensor::zeros(vec![nh, t, 1], DType::F32, self.device)?], 2)?.reshape(vec![nh, t * (r + 1)])?;
            let mut idx = Vec::with_capacity(nh * t * t);
            for hh in 0..nh {
                idx.extend(rel2abs.iter().map(|&x| x + (hh * t * (r + 1)) as i64));
            }
            let idx_t = Tensor::from_vec(idx, vec![nh * t * t], self.device)?;
            let rel = qe.reshape(vec![nh * t * (r + 1)])?.index_select(0, &idx_t)?.reshape(vec![nh, t, t])?;
            scores = scores.add(&rel)?;
            lap("rel_k", &scores);
            let p = softmax_dim(&scores, 2)?;
            lap("softmax", &p);
            let mut o = p.matmul(&v)?; // [H, T, d]
            let pp = Tensor::cat(&[&p, &Tensor::zeros(vec![nh, t, 1], DType::F32, self.device)?], 2)?.reshape(vec![nh * t * (t + 1)])?;
            let mut idx2 = Vec::with_capacity(nh * t * r);
            for hh in 0..nh {
                idx2.extend(abs2rel.iter().map(|&x| x + (hh * t * (t + 1)) as i64));
            }
            let idx2 = Tensor::from_vec(idx2, vec![nh * t * r], self.device)?;
            let prel = pp.index_select(0, &idx2)?.reshape(vec![nh, t, r])?;
            o = o.add(&prel.matmul(&l.rel_v.unsqueeze(0)?.broadcast_as(vec![nh, r, d])?.contiguous()?)?)?;
            lap("pv+rel_v", &o);
            let o = o.permute(&[1, 0, 2])?.contiguous()?.reshape(vec![1usize, t, c])?;
            let y = l.o.forward(&o)?;
            h = layer_norm(&h.add(&y)?, Some(&l.ln1.0), Some(&l.ln1.1), 1e-6)?;
            let y = l.ffn2.forward(&l.ffn1.forward(&h)?.relu()?)?;
            h = layer_norm(&h.add(&y)?, Some(&l.ln2.0), Some(&l.ln2.1), 1e-6)?;
            lap("o+ffn+ln", &h);
        }
        t_bct(&h)
    }
}

/// Внимание к токенам стиля: ключи — tanh(W_k·keys), значения — W_v·values,
/// масштаб 1/16, головы по 128.
pub struct StyleAttention {
    q: QuantLinear,
    k: QuantLinear,
    v: QuantLinear,
    o: QuantLinear,
    heads: usize,
}

impl StyleAttention {
    pub fn load(w: &Weights, p: &str, heads: usize) -> Result<Self> {
        let l = |n: &str| lin_mm(w, &format!("{p}.{n}.linear.MatMul.w1"), Some(&format!("{p}.{n}.linear.bias")));
        Ok(Self { q: l("W_query")?, k: l("W_key")?, v: l("W_value")?, o: l("out_fc")?, heads })
    }

    /// `x` `[B, T, Cq]`, `keys`/`values` `[B, S, 256]` → `[B, T, Cout]`.
    pub fn forward(&self, x: &Tensor, keys: &Tensor, values: &Tensor) -> Result<Tensor> {
        let (b, t) = (x.dims()[0], x.dims()[1]);
        let s = keys.dims()[1];
        let q = self.q.forward(x)?;
        let c = q.dims()[2];
        let d = c / self.heads;
        let heads = |y: Tensor, n: usize| -> Result<Tensor> { Ok(y.reshape(vec![b, n, self.heads, d])?.permute(&[0, 2, 1, 3])?.contiguous()?) };
        let q = heads(q, t)?;
        let k = heads(self.k.forward(keys)?.tanh()?, s)?;
        let v = heads(self.v.forward(values)?, s)?;
        let o = scaled_dot_attention(&q, &k, &v, 1.0 / 16.0, None)?;
        let o = o.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(vec![b, t, c])?;
        Ok(self.o.forward(&o)?)
    }
}

/// Предсказатель длительности (секунды).
pub struct DurationPredictor {
    embed: Tensor,
    token: Tensor,
    convnext: Vec<ConvNeXt>,
    attn: VitsEncoder,
    proj: QuantLinear,
    l0: QuantLinear,
    prelu: f32,
    l1: QuantLinear,
}

impl DurationPredictor {
    pub fn load(w: &Weights) -> Result<Self> {
        let p = "sentence_encoder";
        Ok(Self {
            embed: w.find(&format!("{p}.text_embedder.char_embedder.weight"))?,
            token: w.find(&format!("{p}.sentence_token"))?,
            convnext: convnext_stack(w, &format!("{p}.convnext.convnext"), "dwconv", &[1; 6], false)?,
            attn: VitsEncoder::load(w, &format!("{p}.attn_encoder"), 2, 2)?,
            proj: lin_conv(w, &format!("{p}.proj_out.net.weight"), None)?,
            l0: QuantLinear::dense(w.find("predictor.layers.0.weight")?, Some(w.find("predictor.layers.0.bias")?))?,
            prelu: w.host("predictor.activation.weight")?[0],
            l1: QuantLinear::dense(w.find("predictor.layers.1.weight")?, Some(w.find("predictor.layers.1.bias")?))?,
        })
    }

    pub fn forward(&self, ids: &Tensor, style_dp: &Tensor) -> Result<f32> {
        let t = ids.dims()[0];
        let e = self.embed.index_select(0, ids)?.transpose(0, 1)?.contiguous()?.reshape(vec![1usize, 64, t])?;
        let x = Tensor::cat(&[&self.token, &e], 2)?;
        let c = run_stack(&self.convnext, &x)?;
        let h = self.attn.forward(&c)?.add(&c)?;
        let s = h.narrow(2, 0, 1)?.contiguous()?.reshape(vec![1usize, 64])?;
        let s = self.proj.forward(&s)?;
        let z = Tensor::cat(&[&s, &style_dp.reshape(vec![1usize, 128])?], 1)?;
        let y = self.l0.forward(&z)?.flatten_all()?.to_vec1::<f32>()?;
        let y: Vec<f32> = y.iter().map(|&v| if v >= 0.0 { v } else { v * self.prelu }).collect();
        let n = y.len();
        let y = Tensor::from_vec(y, vec![1usize, n], ids.device())?;
        Ok(self.l1.forward(&y)?.flatten_all()?.to_vec1::<f32>()?[0].exp())
    }
}

/// Текстовый энкодер с подсказкой стилем → `text_emb` `[1, 256, T]`.
pub struct TextEncoder {
    embed: Tensor,
    convnext: Vec<ConvNeXt>,
    attn: VitsEncoder,
    style_key: Tensor,
    a1: StyleAttention,
    a2: StyleAttention,
    ln: (Tensor, Tensor),
}

impl TextEncoder {
    pub fn load(w: &Weights) -> Result<Self> {
        let p = "text_encoder";
        Ok(Self {
            embed: w.find(&format!("{p}.text_embedder.char_embedder.weight"))?,
            convnext: convnext_stack(w, &format!("{p}.convnext.convnext"), "dwconv", &[1, 1, 2, 2, 4, 4], false)?,
            attn: VitsEncoder::load(w, &format!("{p}.attn_encoder"), 4, 4)?,
            style_key: w.find("style_encoder.style_token_layer.style_key")?,
            a1: StyleAttention::load(w, "speech_prompted_text_encoder.attention1", 2)?,
            a2: StyleAttention::load(w, "speech_prompted_text_encoder.attention2", 2)?,
            ln: (w.find("speech_prompted_text_encoder.norm.norm.weight")?, w.find("speech_prompted_text_encoder.norm.norm.bias")?),
        })
    }

    pub fn forward(&self, ids: &Tensor, style_ttl: &Tensor) -> Result<Tensor> {
        let t = ids.dims()[0];
        let e = self.embed.index_select(0, ids)?.transpose(0, 1)?.contiguous()?.reshape(vec![1usize, 256, t])?;
        let c = run_stack(&self.convnext, &e)?;
        let h = self.attn.forward(&c)?.add(&c)?;
        let x0 = t_bct(&h)?;
        let h1 = x0.add(&self.a1.forward(&x0, &self.style_key, style_ttl)?)?;
        let h2 = x0.add(&self.a2.forward(&h1, &self.style_key, style_ttl)?)?;
        t_bct(&layer_norm(&h2, Some(&self.ln.0), Some(&self.ln.1), 1e-6)?)
    }
}

/// Внимание латента к тексту с RoPE по нормированным позициям (i/L, j/T).
struct TextAttention {
    q: QuantLinear,
    k: QuantLinear,
    v: QuantLinear,
    o: QuantLinear,
    ln: (Tensor, Tensor),
}

struct Group {
    cn0: Vec<ConvNeXt>,
    time: QuantLinear,
    cn1: Vec<ConvNeXt>,
    text: TextAttention,
    cn2: Vec<ConvNeXt>,
    style: StyleAttention,
    style_ln: (Tensor, Tensor),
}

/// Векторное поле flow matching (с CFG: батч 2 — условный и безусловный).
pub struct VectorField {
    proj_in: QuantLinear,
    t_freqs: Vec<f32>,
    t_l0: QuantLinear,
    t_l1: QuantLinear,
    groups: Vec<Group>,
    last: Vec<ConvNeXt>,
    proj_out: QuantLinear,
    theta: Vec<f32>,
    style_key: Tensor,
    key_uncond: Tensor,
    value_uncond: Tensor,
    text_uncond: Tensor,
    pub cfg: (f32, f32),
    device: Device,
}

impl VectorField {
    pub fn load(w: &Weights) -> Result<Self> {
        let mb = |i: usize, s: &str| format!("main_blocks.{i}.{s}");
        let mut groups = Vec::new();
        for g in 0..4 {
            let b = 6 * g;
            let tl = |n: &str| lin_mm(w, &format!("vector_field.{}.linear.MatMul.w1", mb(b + 3, &format!("attn.{n}"))), Some(&format!("vector_field.{}.linear.bias", mb(b + 3, &format!("attn.{n}")))));
            groups.push(Group {
                cn0: convnext_stack(w, &format!("vector_field.main_blocks.{b}.convnext"), "dwconv", &[1, 2, 4, 8], false)?,
                time: lin_mm(w, &format!("vector_field.{}", mb(b + 1, "linear.linear.MatMul.w1")), Some(&format!("vector_field.{}", mb(b + 1, "linear.linear.bias"))))?,
                cn1: convnext_stack(w, &format!("vector_field.main_blocks.{}.convnext", b + 2), "dwconv", &[1], false)?,
                text: TextAttention {
                    q: tl("W_query")?,
                    k: tl("W_key")?,
                    v: tl("W_value")?,
                    o: tl("out_fc")?,
                    ln: (w.find(&format!("vector_field.{}", mb(b + 3, "norm.norm.weight")))?, w.find(&format!("vector_field.{}", mb(b + 3, "norm.norm.bias")))?),
                },
                cn2: convnext_stack(w, &format!("vector_field.main_blocks.{}.convnext", b + 4), "dwconv", &[1], false)?,
                style: StyleAttention::load(w, &format!("vector_field.{}", mb(b + 5, "attention")), 2)?,
                style_ln: (w.find(&format!("vector_field.{}", mb(b + 5, "norm.norm.weight")))?, w.find(&format!("vector_field.{}", mb(b + 5, "norm.norm.bias")))?),
            });
        }
        // CFG: v = c·v_cond − u·v_uncond (корневые узлы Mul_1/Mul_2 графа).
        let cfg_c = w.host("Mul_1.w1")?[0];
        let cfg_u = w.host("Mul_2.w1")?[0];
        Ok(Self {
            proj_in: lin_conv(w, "vector_field.proj_in.net.weight", None)?,
            t_freqs: w.host("time_encoder.sinusoidal.Mul_1.w1")?,
            t_l0: QuantLinear::dense(w.find("time_encoder.mlp.0.linear.weight")?, Some(w.find("time_encoder.mlp.0.linear.bias")?))?,
            t_l1: QuantLinear::dense(w.find("time_encoder.mlp.2.linear.weight")?, Some(w.find("time_encoder.mlp.2.linear.bias")?))?,
            groups,
            last: convnext_stack(w, "vector_field.last_convnext.convnext", "dwconv", &[1; 4], false)?,
            proj_out: lin_conv(w, "vector_field.proj_out.net.weight", None)?,
            theta: w.host("main_blocks.3.attn.theta")?,
            style_key: w.find("vector_estimator.Tile.w0")?,
            key_uncond: w.find("uncond_masker.style_key_special_token")?,
            value_uncond: w.find("uncond_masker.style_value_special_token")?,
            text_uncond: w.find("uncond_masker.text_special_token")?,
            cfg: (cfg_c, cfg_u),
            device: w.device,
        })
    }

    /// cos/sin RoPE `[n, 32]` для позиций i/n.
    fn rope(&self, n: usize) -> Result<(Tensor, Tensor)> {
        let h = self.theta.len();
        let (mut c, mut s) = (vec![0f32; n * h], vec![0f32; n * h]);
        for i in 0..n {
            let p = i as f32 / n as f32;
            for k in 0..h {
                let a = p * self.theta[k];
                c[i * h + k] = a.cos();
                s[i * h + k] = a.sin();
            }
        }
        Ok((Tensor::from_vec(c, vec![n, h], self.device)?, Tensor::from_vec(s, vec![n, h], self.device)?))
    }

    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // x [B, H, N, 64]: половины (0..32, 32..64)
        let half = x.dims()[3] / 2;
        let a = x.narrow(3, 0, half)?.contiguous()?;
        let b = x.narrow(3, half, half)?.contiguous()?;
        let r1 = a.broadcast_mul(cos)?.sub(&b.broadcast_mul(sin)?)?;
        let r2 = a.broadcast_mul(sin)?.add(&b.broadcast_mul(cos)?)?;
        Ok(Tensor::cat(&[&r1, &r2], 3)?)
    }

    /// Вложение времени t ∈ [0, 1): sin/cos(1000·t·f) → MLP с Mish → `[1, 64]`.
    fn time(&self, t: f32) -> Result<Tensor> {
        let mut e: Vec<f32> = self.t_freqs.iter().map(|f| (1000.0 * t * f).sin()).collect();
        e.extend(self.t_freqs.iter().map(|f| (1000.0 * t * f).cos()));
        let n = e.len();
        let h = self.t_l0.forward(&Tensor::from_vec(e, vec![1usize, n], self.device)?)?;
        let hv = h.flatten_all()?.to_vec1::<f32>()?;
        let m: Vec<f32> = hv.iter().map(|&x| x * ((1.0 + x.exp()).ln()).tanh()).collect();
        let k = m.len();
        Ok(self.t_l1.forward(&Tensor::from_vec(m, vec![1usize, k], self.device)?)?)
    }

    /// Один шаг Эйлера: x `[1, 144, L]`, текст `[1, 256, T]`, стиль `[1, 50, 256]`.
    pub fn step(&self, x: &Tensor, text: &Tensor, style: &Tensor, step: usize, total: usize) -> Result<Tensor> {
        let l = x.dims()[2];
        let tt = text.dims()[2];
        let xb = Tensor::cat(&[x, x], 0)?;
        let text_u = self.text_uncond.broadcast_as(vec![1usize, 256, tt])?.contiguous()?;
        let text_b = t_bct(&Tensor::cat(&[text, &text_u], 0)?)?; // [2, T, 256]
        let keys = Tensor::cat(&[&self.style_key, &self.key_uncond], 0)?;
        let values = Tensor::cat(&[style, &self.value_uncond], 0)?;
        let temb = self.time(step as f32 / total as f32)?;
        let (qc, qs) = self.rope(l)?;
        let (kc, ks) = self.rope(tt)?;
        let mut h = t_bct(&self.proj_in.forward(&t_bct(&xb)?)?)?; // [2, 512, L]
        dump("vf_in", &h);
        dump("vf_temb", &temb);
        for (gi, g) in self.groups.iter().enumerate() {
            for (ci, c) in g.cn0.iter().enumerate() {
                h = c.forward(&h)?;
                dump(&format!("vf_g{gi}_cn0_{ci}"), &h);
            }
            let tv = g.time.forward(&temb)?.reshape(vec![1usize, 512, 1])?;
            h = h.broadcast_add(&tv)?;
            dump(&format!("vf_g{gi}_time"), &h);
            h = run_stack(&g.cn1, &h)?;
            dump(&format!("vf_g{gi}_cn1"), &h);
            // текстовое внимание
            let hb = t_bct(&h)?; // [2, L, 512]
            let heads = |y: Tensor, n: usize| -> Result<Tensor> { Ok(y.reshape(vec![2usize, n, 8, 64])?.permute(&[0, 2, 1, 3])?.contiguous()?) };
            let q = Self::apply_rope(&heads(g.text.q.forward(&hb)?, l)?, &qc, &qs)?;
            let k = Self::apply_rope(&heads(g.text.k.forward(&text_b)?, tt)?, &kc, &ks)?;
            let v = heads(g.text.v.forward(&text_b)?, tt)?;
            let o = scaled_dot_attention(&q, &k, &v, 1.0 / 16.0, None)?;
            let o = g.text.o.forward(&o.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(vec![2usize, l, 512])?)?;
            let hb = layer_norm(&hb.add(&o)?, Some(&g.text.ln.0), Some(&g.text.ln.1), 1e-6)?;
            dump(&format!("vf_g{gi}_text"), &hb);
            h = run_stack(&g.cn2, &t_bct(&hb)?)?;
            dump(&format!("vf_g{gi}_cn2"), &h);
            // внимание к стилю
            let hb = t_bct(&h)?;
            let o = g.style.forward(&hb, &keys, &values)?;
            let hb = layer_norm(&hb.add(&o)?, Some(&g.style_ln.0), Some(&g.style_ln.1), 1e-6)?;
            h = t_bct(&hb)?;
            dump(&format!("vf_g{gi}_style"), &h);
        }
        h = run_stack(&self.last, &h)?;
        let v = t_bct(&self.proj_out.forward(&t_bct(&h)?)?)?; // [2, 144, L]
        dump("vf_last", &h);
        dump("vf_v", &v);
        let vc = v.narrow(0, 0, 1)?.contiguous()?;
        let vu = v.narrow(0, 1, 1)?.contiguous()?;
        let guided = vc.affine(self.cfg.0, 0.0)?.sub(&vu.affine(self.cfg.1, 0.0)?)?;
        dump("vf_guided", &guided);
        let out = x.add(&guided.affine(1.0 / total as f32, 0.0)?)?;
        dump("vf_x0_in", x);
        dump("vf_out", &out);
        Ok(out)
    }
}

/// Вокодер: латент `[1, 144, L]` → PCM 44.1 кГц (3072 семпла на кадр латента).
pub struct Vocoder {
    scale: f32,
    mean: Tensor,
    std: Tensor,
    embed_w: Tensor,
    embed_b: Tensor,
    convnext: Vec<ConvNeXt>,
    bn_scale: Tensor,
    bn_shift: Tensor,
    head1_w: Tensor,
    head1_b: Tensor,
    prelu: f32,
    head2: QuantLinear,
}

impl Vocoder {
    pub fn load(w: &Weights) -> Result<Self> {
        let var = w.host("final_norm.norm.running_var")?;
        let mean = w.host("final_norm.norm.running_mean")?;
        let g = w.host("final_norm.norm.weight")?;
        let b = w.host("final_norm.norm.bias")?;
        let sc: Vec<f32> = g.iter().zip(&var).map(|(g, v)| g / (v + 1e-5).sqrt()).collect();
        let sh: Vec<f32> = b.iter().zip(&mean).zip(&sc).map(|((b, m), s)| b - m * s).collect();
        let n = sc.len();
        Ok(Self {
            scale: w.host("normalizer.scale")?[0],
            mean: w.find("latent_mean")?,
            std: w.find("latent_std")?,
            embed_w: w.find("decoder.embed.net.Conv.w1")?,
            embed_b: w.find("decoder.embed.net.Conv.w2")?,
            convnext: convnext_stack(w, "decoder.convnext", "dwconv.net", &[1, 2, 4, 1, 2, 4, 1, 1, 1, 1], true)?,
            bn_scale: Tensor::from_vec(sc, vec![1usize, n, 1], w.device)?,
            bn_shift: Tensor::from_vec(sh, vec![1usize, n, 1], w.device)?,
            head1_w: w.find("decoder.head.layer1.net.weight")?,
            head1_b: w.find("decoder.head.layer1.net.bias")?,
            prelu: w.host("decoder.head.act.PRelu.w1")?[0],
            head2: lin_conv(w, "decoder.head.layer2.weight", None)?,
        })
    }

    pub fn forward(&self, latent: &Tensor) -> Result<Vec<f32>> {
        let l = latent.dims()[2];
        let x = latent.affine(1.0 / self.scale, 0.0)?.reshape(vec![1usize, 24, 6, l])?.transpose(2, 3)?.contiguous()?.reshape(vec![1usize, 24, 6 * l])?;
        let x = x.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let x = conv1d_dilated(&pad_edge(&x, 6, 0)?, &self.embed_w, Some(&self.embed_b), 1, 0, 1)?;
        let x = run_stack(&self.convnext, &x)?;
        let x = x.broadcast_mul(&self.bn_scale)?.broadcast_add(&self.bn_shift)?;
        let x = conv1d_dilated(&pad_edge(&x, 2, 0)?, &self.head1_w, Some(&self.head1_b), 1, 0, 1)?;
        // PReLU с одним α: max(x,0) + α·min(x,0) = relu(x) − α·relu(−x)
        let x = x.relu()?.sub(&x.affine(-1.0, 0.0)?.relu()?.affine(self.prelu, 0.0)?)?;
        let y = self.head2.forward(&t_bct(&x)?)?; // [1, 6L, 512]
        Ok(y.flatten_all()?.to_vec1::<f32>()?)
    }
}
