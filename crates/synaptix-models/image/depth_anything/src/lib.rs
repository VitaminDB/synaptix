//! Depth Anything V2 Small (HF `depth-anything/Depth-Anything-V2-Small-hf`) —
//! монокулярная глубина для control-сигналов (IC-LoRA union-control depth, как
//! ComfyUI Depth-нода).
//!
//! Архитектура: DINOv2 ViT-S/14 (dim 384, 12 блоков, 6 голов, layer_scale,
//! pre-LN, GELU-exact) → DPT-neck (reassemble层 [48,96,192,384] × факторы
//! [4,2,1,0.5] + fusion 64) → голова (conv→bilinear→conv→relu→conv→relu).
//! Вход ФИКСИРОВАН 518×518 (без интерполяции pos-эмбеддингов), выход — карта
//! относительной глубины 518×518 (больше = ближе).

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_ops::attention::softmax_dim;
use synaptix_ops::conv::conv2d::conv2d;

type R<T> = Result<T, SynaptixError>;

pub const INPUT: usize = 518; // 37×37 патчей по 14
const DIM: usize = 384;
const HEADS: usize = 6;
const HEAD_DIM: usize = 64;
const GRID: usize = 37;
const EPS: f32 = 1e-6;
const NECK_CH: [usize; 4] = [48, 96, 192, 384];
const OUT_LAYERS: [usize; 4] = [3, 6, 9, 12]; // 1-based: после блоков 3/6/9/12
const FUSION: usize = 64;

fn lin(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> R<Tensor> {
    let y = x.matmul(&w.transpose(0, 1)?.contiguous()?)?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

struct Block {
    n1w: Tensor,
    n1b: Tensor,
    qw: Tensor,
    qb: Tensor,
    kw: Tensor,
    kb: Tensor,
    vw: Tensor,
    vb: Tensor,
    ow: Tensor,
    ob: Tensor,
    ls1: Tensor,
    n2w: Tensor,
    n2b: Tensor,
    f1w: Tensor,
    f1b: Tensor,
    f2w: Tensor,
    f2b: Tensor,
    ls2: Tensor,
}

impl Block {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let t = x.dims()[1];
        let h = x.layer_norm_fused(&self.n1w, Some(&self.n1b), EPS)?;
        let q = lin(&h, &self.qw, Some(&self.qb))?
            .reshape(vec![1, t, HEADS, HEAD_DIM])?.transpose(1, 2)?.contiguous()?;
        let k = lin(&h, &self.kw, Some(&self.kb))?
            .reshape(vec![1, t, HEADS, HEAD_DIM])?.transpose(1, 2)?.contiguous()?;
        let v = lin(&h, &self.vw, Some(&self.vb))?
            .reshape(vec![1, t, HEADS, HEAD_DIM])?.transpose(1, 2)?.contiguous()?;
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?
            .mul_scalar(1.0 / (HEAD_DIM as f32).sqrt())?;
        let attn = softmax_dim(&scores, 3)?.matmul(&v)?; // [1,H,T,dh]
        let attn = attn.transpose(1, 2)?.contiguous()?.reshape(vec![1, t, DIM])?;
        let attn = lin(&attn, &self.ow, Some(&self.ob))?;
        let x = x.add(&attn.broadcast_mul(&self.ls1)?)?;
        let h = x.layer_norm_fused(&self.n2w, Some(&self.n2b), EPS)?;
        let h = lin(&h, &self.f1w, Some(&self.f1b))?.gelu_exact()?;
        let h = lin(&h, &self.f2w, Some(&self.f2b))?;
        x.add(&h.broadcast_mul(&self.ls2)?)
    }
}

struct Conv {
    w: Tensor,
    b: Option<Tensor>,
}

/// Bilinear-resize `[1,C,H,W]` → `[1,C,oh,ow]` (CPU f32; align_corners как torch).
fn bilinear(x: &Tensor, oh: usize, ow: usize, align: bool) -> R<Tensor> {
    let (c, h, w) = (x.dims()[1], x.dims()[2], x.dims()[3]);
    if h == oh && w == ow {
        return Ok(x.clone());
    }
    let dev = x.device();
    let v: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut out = vec![0f32; c * oh * ow];
    let src = |d: usize, o: usize, i: usize| -> f32 {
        if align {
            if o == 1 { 0.0 } else { d as f32 * (i as f32 - 1.0) / (o as f32 - 1.0) }
        } else {
            ((d as f32 + 0.5) * i as f32 / o as f32 - 0.5).max(0.0)
        }
    };
    for y in 0..oh {
        let sy = src(y, oh, h).min(h as f32 - 1.0);
        let (y0, fy) = (sy.floor() as usize, sy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for xx in 0..ow {
            let sx = src(xx, ow, w).min(w as f32 - 1.0);
            let (x0, fx) = (sx.floor() as usize, sx.fract());
            let x1 = (x0 + 1).min(w - 1);
            for ch in 0..c {
                let p = |yy: usize, xc: usize| v[(ch * h + yy) * w + xc];
                let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
                let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
                out[(ch * oh + y) * ow + xx] = top * (1.0 - fy) + bot * fy;
            }
        }
    }
    Tensor::from_vec(out, vec![1, c, oh, ow], Device::Cpu)?.to_device(dev)
}

/// ConvTranspose2d с kernel=stride=f, pad 0 (reassemble upsample): эквивалент
/// 1×1-линейки `in→out·f²` + pixel-shuffle. `w` HF-layout `[in,out,f,f]`.
fn conv_transpose_up(x: &Tensor, w: &Tensor, b: &Tensor, f: usize) -> R<Tensor> {
    let (cin, cout) = (w.dims()[0], w.dims()[1]);
    let (h, ww) = (x.dims()[2], x.dims()[3]);
    // [in,out,f,f] → [in, out·f·f]
    let w2 = w.reshape(vec![cin, cout * f * f])?;
    let xf = x.reshape(vec![1, cin, h * ww])?.transpose(1, 2)?.contiguous()?; // [1,HW,in]
    let y = xf.matmul(&w2)?; // [1,HW,out·f·f]
    // [1,H,W,out,f,f] → [1,out,H,f,W,f] → [1,out,H·f,W·f]
    let y = y.reshape(vec![1, h, ww, cout, f, f])?
        .permute(vec![0, 3, 1, 4, 2, 5])?
        .contiguous()?
        .reshape(vec![1, cout, h * f, ww * f])?;
    y.broadcast_add(&b.reshape(vec![1, cout, 1, 1])?)
}

enum Resize {
    Up(Conv, usize), // conv_transpose f=4|2
    None,
    Down(Conv), // conv k3 s2 p1
}

struct Reassemble {
    proj: Conv, // 1×1 384→ch
    resize: Resize,
}

struct Fusion {
    proj: Conv,           // 1×1 64→64
    r1c1: Conv,
    r1c2: Conv,
    r2c1: Conv,
    r2c2: Conv,
}

fn preact(x: &Tensor, c1: &Conv, c2: &Conv) -> R<Tensor> {
    let h = conv2d(&x.relu()?, &c1.w, c1.b.as_ref(), (1, 1), (1, 1), (1, 1))?;
    let h = conv2d(&h.relu()?, &c2.w, c2.b.as_ref(), (1, 1), (1, 1), (1, 1))?;
    x.add(&h)
}

pub struct DepthAnything {
    cls: Tensor,
    pos: Tensor, // [1,1370,384]
    patch_w: Tensor,
    patch_b: Tensor,
    blocks: Vec<Block>,
    final_ln_w: Tensor,
    final_ln_b: Tensor,
    reassemble: Vec<Reassemble>,
    neck_convs: Vec<Conv>, // 3×3 ch→64, без bias
    fusion: Vec<Fusion>,
    head1: Conv,
    head2: Conv,
    head3: Conv,
    device: Device,
}

impl DepthAnything {
    pub fn load(dir: &std::path::Path, device: Device) -> Result<Self, String> {
        let ld = SafetensorsLoader::open(dir.join("model.safetensors"))
            .map_err(|e| e.to_string())?
            .with_device(device);
        let g = |n: &str| -> Result<Tensor, String> {
            ld.load_to(n, device, DType::F32).map_err(|e| format!("{n}: {e}"))
        };
        let conv = |p: &str, bias: bool| -> Result<Conv, String> {
            Ok(Conv {
                w: g(&format!("{p}.weight"))?,
                b: if bias { Some(g(&format!("{p}.bias"))?) } else { None },
            })
        };
        let mut blocks = Vec::with_capacity(12);
        for i in 0..12 {
            let p = format!("backbone.encoder.layer.{i}");
            blocks.push(Block {
                n1w: g(&format!("{p}.norm1.weight"))?,
                n1b: g(&format!("{p}.norm1.bias"))?,
                qw: g(&format!("{p}.attention.attention.query.weight"))?,
                qb: g(&format!("{p}.attention.attention.query.bias"))?,
                kw: g(&format!("{p}.attention.attention.key.weight"))?,
                kb: g(&format!("{p}.attention.attention.key.bias"))?,
                vw: g(&format!("{p}.attention.attention.value.weight"))?,
                vb: g(&format!("{p}.attention.attention.value.bias"))?,
                ow: g(&format!("{p}.attention.output.dense.weight"))?,
                ob: g(&format!("{p}.attention.output.dense.bias"))?,
                ls1: g(&format!("{p}.layer_scale1.lambda1"))?,
                n2w: g(&format!("{p}.norm2.weight"))?,
                n2b: g(&format!("{p}.norm2.bias"))?,
                f1w: g(&format!("{p}.mlp.fc1.weight"))?,
                f1b: g(&format!("{p}.mlp.fc1.bias"))?,
                f2w: g(&format!("{p}.mlp.fc2.weight"))?,
                f2b: g(&format!("{p}.mlp.fc2.bias"))?,
                ls2: g(&format!("{p}.layer_scale2.lambda1"))?,
            });
        }
        let factors = [4usize, 2, 1, 0]; // 0 = down (0.5)
        let mut reassemble = Vec::new();
        for (i, &f) in factors.iter().enumerate() {
            let p = format!("neck.reassemble_stage.layers.{i}");
            let resize = match f {
                4 | 2 => Resize::Up(conv(&format!("{p}.resize"), true)?, f),
                1 => Resize::None,
                _ => Resize::Down(conv(&format!("{p}.resize"), true)?),
            };
            reassemble.push(Reassemble { proj: conv(&format!("{p}.projection"), true)?, resize });
        }
        let mut neck_convs = Vec::new();
        for i in 0..4 {
            neck_convs.push(conv(&format!("neck.convs.{i}"), false)?);
        }
        let mut fusion = Vec::new();
        for i in 0..4 {
            let p = format!("neck.fusion_stage.layers.{i}");
            fusion.push(Fusion {
                proj: conv(&format!("{p}.projection"), true)?,
                r1c1: conv(&format!("{p}.residual_layer1.convolution1"), true)?,
                r1c2: conv(&format!("{p}.residual_layer1.convolution2"), true)?,
                r2c1: conv(&format!("{p}.residual_layer2.convolution1"), true)?,
                r2c2: conv(&format!("{p}.residual_layer2.convolution2"), true)?,
            });
        }
        Ok(Self {
            cls: g("backbone.embeddings.cls_token")?,
            pos: g("backbone.embeddings.position_embeddings")?,
            patch_w: g("backbone.embeddings.patch_embeddings.projection.weight")?,
            patch_b: g("backbone.embeddings.patch_embeddings.projection.bias")?,
            blocks,
            final_ln_w: g("backbone.layernorm.weight")?,
            final_ln_b: g("backbone.layernorm.bias")?,
            reassemble,
            neck_convs,
            fusion,
            head1: conv("head.conv1", true)?,
            head2: conv("head.conv2", true)?,
            head3: conv("head.conv3", true)?,
            device,
        })
    }

    /// `px` `[1,3,518,518]` (ImageNet-нормированный) → глубина `[1,518,518]`
    /// (относительная, больше = ближе).
    pub fn forward(&self, px: &Tensor) -> R<Tensor> {
        let px = px.to_device(self.device)?.to_dtype(DType::F32)?;
        // patch embed: conv 14×14 s14 → [1,384,37,37] → [1,1369,384] (+cls, +pos)
        let pe = conv2d(&px, &self.patch_w, Some(&self.patch_b), (14, 14), (0, 0), (1, 1))?;
        let tok = pe.reshape(vec![1, DIM, GRID * GRID])?.transpose(1, 2)?.contiguous()?;
        let mut x = Tensor::cat(&[&self.cls, &tok], 1)?.contiguous()?.broadcast_add(&self.pos)?;
        // 12 блоков; собираем после 3/6/9/12 (1-based) + финальный LN (Dinov2Backbone)
        let mut feats: Vec<Tensor> = Vec::with_capacity(4);
        for (i, b) in self.blocks.iter().enumerate() {
            x = b.forward(&x)?;
            if OUT_LAYERS.contains(&(i + 1)) {
                feats.push(x.layer_norm_fused(&self.final_ln_w, Some(&self.final_ln_b), EPS)?);
            }
        }
        // neck: reassemble (drop cls → [1,384,37,37] → proj → resize) → conv 3×3 → 64
        let mut level: Vec<Tensor> = Vec::with_capacity(4);
        for (i, f) in feats.iter().enumerate() {
            let h = f.narrow(1, 1, GRID * GRID)?.contiguous()?
                .transpose(1, 2)?.contiguous()?.reshape(vec![1, DIM, GRID, GRID])?;
            let ra = &self.reassemble[i];
            let h = conv2d(&h, &ra.proj.w, ra.proj.b.as_ref(), (1, 1), (0, 0), (1, 1))?;
            let h = match &ra.resize {
                Resize::Up(c, f) => conv_transpose_up(&h, &c.w, c.b.as_ref().unwrap(), *f)?,
                Resize::None => h,
                Resize::Down(c) => conv2d(&h, &c.w, c.b.as_ref(), (2, 2), (1, 1), (1, 1))?,
            };
            let nc = &self.neck_convs[i];
            level.push(conv2d(&h, &nc.w, None, (1, 1), (1, 1), (1, 1))?);
        }
        // fusion top-down: уровни в обратном порядке; размер шага = размер следующего
        let mut fused: Option<Tensor> = None;
        for (idx, li) in (0..4).rev().enumerate() {
            let fu = &self.fusion[idx];
            let hs = &level[li];
            let mut h = match &fused {
                None => hs.clone(),
                Some(prev) => {
                    let res = if prev.dims() != hs.dims() {
                        bilinear(hs, prev.dims()[2], prev.dims()[3], false)?
                    } else {
                        hs.clone()
                    };
                    prev.add(&preact(&res, &fu.r1c1, &fu.r1c2)?)?
                }
            };
            h = preact(&h, &fu.r2c1, &fu.r2c2)?;
            let (oh, ow) = if li > 0 {
                (level[li - 1].dims()[2], level[li - 1].dims()[3])
            } else {
                (h.dims()[2] * 2, h.dims()[3] * 2)
            };
            h = bilinear(&h, oh, ow, true)?;
            h = conv2d(&h, &fu.proj.w, fu.proj.b.as_ref(), (1, 1), (0, 0), (1, 1))?;
            fused = Some(h);
        }
        // head: conv1 → bilinear до 518 (align=true) → conv2 → relu → conv3 → relu
        let h = fused.unwrap();
        let h = conv2d(&h, &self.head1.w, self.head1.b.as_ref(), (1, 1), (1, 1), (1, 1))?;
        let h = bilinear(&h, INPUT, INPUT, true)?;
        let h = conv2d(&h, &self.head2.w, self.head2.b.as_ref(), (1, 1), (1, 1), (1, 1))?.relu()?;
        let h = conv2d(&h, &self.head3.w, self.head3.b.as_ref(), (1, 1), (0, 0), (1, 1))?.relu()?;
        h.reshape(vec![1, INPUT, INPUT])
    }

    /// RGB-кадр `[3,H,W]` в [0,1] → карта глубины `[3,H,W]` в [0,1] (нормирована
    /// per-frame, ближе = белее, реплицирована в RGB) — control-сигнал.
    pub fn depth_rgb(&self, frame: &Tensor) -> R<Tensor> {
        let (h, w) = (frame.dims()[1], frame.dims()[2]);
        let dev = frame.device();
        // resize до 518² + ImageNet-нормализация
        let r = bilinear(&frame.reshape(vec![1, 3, h, w])?, INPUT, INPUT, false)?;
        let mean = Tensor::from_vec(vec![0.485f32, 0.456, 0.406], vec![1, 3, 1, 1], dev)?;
        let std = Tensor::from_vec(vec![0.229f32, 0.224, 0.225], vec![1, 3, 1, 1], dev)?;
        let px = r.broadcast_sub(&mean)?.broadcast_div(&std)?;
        let d = self.forward(&px)?; // [1,518,518]
        // нормировка [0,1] per-frame
        let v: Vec<f32> = d.flatten_all()?.to_vec1()?;
        let (mut mn, mut mx) = (f32::MAX, f32::MIN);
        for &x in &v {
            mn = mn.min(x);
            mx = mx.max(x);
        }
        let s = if mx > mn { 1.0 / (mx - mn) } else { 0.0 };
        let n: Vec<f32> = v.iter().map(|&x| (x - mn) * s).collect();
        let d = Tensor::from_vec(n, vec![1, 1, INPUT, INPUT], Device::Cpu)?.to_device(dev)?;
        let d = bilinear(&d, h, w, false)?; // назад к размеру кадра
        let d3 = Tensor::cat(&[&d, &d, &d], 1)?.contiguous()?.reshape(vec![3, h, w])?;
        Ok(d3)
    }
}
