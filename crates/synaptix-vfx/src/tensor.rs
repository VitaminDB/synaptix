//! Модели на Tensor API — для любого устройства synaptix (CPU, NPU, CUDA): на телефоне нет CUDA,
//! поэтому здесь путь для процессора и NPU (Hexagon HTP: свёртки, нормы, поэлементные — в граф).
//! Формы фиксированы размером кадра: одно разрешение видео — один граф.
//!
//! Сейчас: SRVGGNetCompact (все варианты AnimeJaNai и realesr-animevideov3).

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::conv::conv2d::conv2d;

use crate::weights::Weights;
use crate::{Error, Frame, Result, Upscaler};

fn te(e: synaptix_core::error::SynaptixError) -> Error {
    Error::Cuda(format!("tensor: {e}"))
}

/// Вес на устройство как статика графа (на NPU — вшивается в граф).
fn param(v: &[f32], dims: Vec<usize>, dev: Device, dtype: DType) -> Result<Tensor> {
    let t = Tensor::from_vec(v.to_vec(), dims, Device::Cpu).and_then(|t| t.to_dtype(dtype)).map_err(te)?;
    synaptix_core::backend::with_param_upload(|| t.to_device(dev)).map_err(te)
}

/// PReLU по каналам: relu(x) − a·relu(−x), `a` — `[1, C, 1, 1]`.
fn prelu(x: &Tensor, a: &Tensor) -> std::result::Result<Tensor, synaptix_core::error::SynaptixError> {
    x.relu()?.sub(&x.affine(-1.0, 0.0)?.relu()?.broadcast_mul(a)?)
}

/// pixel shuffle: `[1, C·r², H, W]` → `[1, C, H·r, W·r]`.
fn pixel_shuffle(x: &Tensor, r: usize) -> std::result::Result<Tensor, synaptix_core::error::SynaptixError> {
    let d = x.dims().to_vec();
    let (c, h, w) = (d[1] / (r * r), d[2], d[3]);
    x.reshape(vec![1usize, c, r, r, h, w])?.permute(&[0, 1, 4, 2, 5, 3])?.contiguous()?.reshape(vec![1usize, c, h * r, w * r])
}

/// Ближайший пиксель ×r: `[1, C, H, W]` → `[1, C, H·r, W·r]`.
fn nearest(x: &Tensor, r: usize) -> std::result::Result<Tensor, synaptix_core::error::SynaptixError> {
    let d = x.dims().to_vec();
    let (c, h, w) = (d[1], d[2], d[3]);
    x.reshape(vec![1usize, c, h, 1, w, 1])?.broadcast_as(vec![1usize, c, h, r, w, r])?.contiguous()?.reshape(vec![1usize, c, h * r, w * r])
}

/// RGBA8 → `[1, 3, H, W]` в [0, 1].
fn rgba_to_chw(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let px = w * h;
    let mut out = vec![0f32; 3 * px];
    for i in 0..px {
        for c in 0..3 {
            out[c * px + i] = rgba[i * 4 + c] as f32 * (1.0 / 255.0);
        }
    }
    out
}

/// `[3, H, W]` в [0, 1] → RGBA8 (альфа 255).
fn chw_to_rgba(x: &[f32], w: usize, h: usize, out: &mut Vec<u8>) {
    let px = w * h;
    out.resize(px * 4, 255);
    for i in 0..px {
        for c in 0..3 {
            out[i * 4 + c] = (x[c * px + i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
        out[i * 4 + 3] = 255;
    }
}

/// SRVGGNetCompact на Tensor API.
pub struct CompactT {
    convs: Vec<(Tensor, Tensor, Option<Tensor>)>,
    scale: usize,
    device: Device,
    dtype: DType,
    out: Vec<u8>,
    name: String,
}

impl CompactT {
    pub fn load(wts: &Weights, device: Device, dtype: DType) -> Result<Self> {
        let feat = wts.meta_usize("feat")?;
        let n = wts.meta_usize("convs")?;
        let scale = wts.meta_usize("scale")?;
        let mut convs = Vec::new();
        for j in 0..n + 2 {
            let (cout, cin) = match j {
                0 => (feat, 3),
                j if j == n + 1 => (3 * scale * scale, feat),
                _ => (feat, feat),
            };
            let w = param(wts.get(&format!("conv{j}.w"))?, vec![cout, cin, 3, 3], device, dtype)?;
            let b = param(wts.get(&format!("conv{j}.b"))?, vec![cout], device, dtype)?;
            let a = if j <= n { Some(param(wts.get(&format!("act{j}"))?, vec![1, cout, 1, 1], device, dtype)?) } else { None };
            convs.push((w, b, a));
        }
        let name = wts.meta.get("name").cloned().unwrap_or_else(|| format!("Compact ×{scale}"));
        Ok(Self { convs, scale, device, dtype, out: Vec::new(), name: format!("{name} [{device:?}]") })
    }

    /// Кадр `[3, H, W]` (хост, [0, 1]) → `[3, H·r, W·r]` (хост).
    pub fn run(&self, x: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        let f = || -> std::result::Result<Vec<f32>, synaptix_core::error::SynaptixError> {
            let x = Tensor::from_vec(x.to_vec(), vec![1usize, 3, h, w], Device::Cpu)?.to_dtype(self.dtype)?.to_device(self.device)?;
            let mut y = x.clone();
            for (cw, cb, a) in &self.convs {
                y = conv2d(&y, cw, Some(cb), (1, 1), (1, 1), (1, 1))?;
                if let Some(a) = a {
                    y = prelu(&y, a)?;
                }
            }
            let y = pixel_shuffle(&y, self.scale)?.add(&nearest(&x, self.scale)?)?;
            y.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()
        };
        f().map_err(te)
    }
}

impl Upscaler for CompactT {
    fn name(&self) -> &str {
        &self.name
    }

    fn scale(&self) -> u32 {
        self.scale as u32
    }

    fn push(&mut self, rgba: &[u8], width: u32, height: u32, emit: &mut dyn FnMut(Frame)) -> Result<()> {
        let (w, h) = (width as usize, height as usize);
        if rgba.len() < w * h * 4 {
            return Err(Error::Unsupported("кадр меньше w×h×4".into()));
        }
        let y = self.run(&rgba_to_chw(rgba, w, h), w, h)?;
        let (ow, oh) = (w * self.scale, h * self.scale);
        let mut out = std::mem::take(&mut self.out);
        chw_to_rgba(&y, ow, oh, &mut out);
        emit(Frame { rgba: &out, width: ow as u32, height: oh as u32 });
        self.out = out;
        Ok(())
    }

    fn reset(&mut self) {}
}

/// Модель на Tensor API по `arch` (F32 на CPU, F16 на NPU).
pub fn open_on(wts: &Weights, device: Device) -> Result<Box<dyn Upscaler>> {
    let dtype = if matches!(device, Device::Npu(_)) { DType::F16 } else { DType::F32 };
    match wts.arch() {
        "compact" => Ok(Box::new(CompactT::load(wts, device, dtype)?)),
        a => Err(Error::Unsupported(format!("«{a}» на {device:?} пока не поддержана"))),
    }
}
